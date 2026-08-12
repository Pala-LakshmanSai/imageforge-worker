from __future__ import annotations

import asyncio
import base64
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID

from fastapi import Body, Depends, FastAPI, Query, Request
from fastapi import Path as PathParameter
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .auth import BearerAuthenticator, Principal
from .config import WorkerSettings
from .constants import API_SCHEMA_VERSION, WORKER_VERSION
from .controller import ArtifactDescriptor, GenerationController
from .coordination import (
    UUID4_PATTERN,
    CancelStopRequest,
    CreateStopRequest,
    FinalizeStopRequest,
    HeartbeatRequest,
    StopResponseRequest,
    StudioStateResponse,
)
from .domain import (
    BatchManifest,
    CreateBatchRequest,
    HealthPhase,
    ReceiptRequest,
    ReceiptResponse,
    StatusResponse,
)
from .errors import WorkerError
from .gpu_switch_models import (
    AdoptGpuSwitchRequestV1,
    CancelGpuSwitchRequestV1,
    CompleteGpuSwitchRequestV1,
    CreateGpuSwitchRequestV1,
    DeleteIntentGpuSwitchRequestV1,
    FinalizeGpuSwitchRequestV1,
    GpuSwitchLookupResponseV1,
    GpuSwitchResponseRequestV1,
    NativeWorkerGpuSwitchCreateResponseV1,
    NativeWorkerGpuSwitchOwnerLookupV1,
    SettleGpuSwitchCreateRequestV1,
    WorkerGpuSwitchRuntimeIdentityV1,
)
from .health import HealthTracker
from .inference import (
    FakeInferenceAdapter,
    FluxInferenceAdapter,
    InferenceAdapter,
    MageFlowInferenceAdapter,
)
from .model_profiles import profile_for_backend
from .persistence import FileManifestStore, ManifestStore

logger = logging.getLogger("imageforge_worker.app")


async def _authenticated_principal(request: Request) -> Principal:
    runtime: WorkerRuntime = request.app.state.runtime
    return await runtime.authenticator(request)


PrincipalDependency = Annotated[Principal, Depends(_authenticated_principal)]
BatchId = Annotated[UUID, PathParameter(description="Server-generated batch UUID")]
SubmissionId = Annotated[
    str,
    PathParameter(
        min_length=36,
        max_length=36,
        pattern=UUID4_PATTERN.pattern,
        description="Canonical lowercase UUIDv4 client submission ID",
    ),
]
ImageIndex = Annotated[int, PathParameter(ge=1)]
StudioId = Annotated[
    str,
    PathParameter(
        min_length=36,
        max_length=36,
        pattern=UUID4_PATTERN.pattern,
        description="Canonical lowercase UUIDv4",
    ),
]
StudioSessionQuery = Annotated[
    str,
    Query(
        min_length=36,
        max_length=36,
        pattern=UUID4_PATTERN.pattern,
        description="Canonical lowercase UUIDv4 live Studio session",
    ),
]


@dataclass(slots=True)
class WorkerRuntime:
    settings: WorkerSettings
    inference: InferenceAdapter
    store: ManifestStore
    controller: GenerationController
    health: HealthTracker
    authenticator: BearerAuthenticator
    boot_task: asyncio.Task[None] | None = None
    lag_task: asyncio.Task[None] | None = None
    # Worst event-loop scheduling delay seen, in milliseconds. `peak` is for the
    # whole process; `recent` covers the window since the last /v1/health read
    # and is reset by it, so a poller sees per-interval blocking rather than one
    # historical spike.
    loop_lag_peak_ms: float = 0.0
    loop_lag_recent_ms: float = 0.0


def create_app(
    settings: WorkerSettings | None = None,
    *,
    inference: InferenceAdapter | None = None,
    store: ManifestStore | None = None,
) -> FastAPI:
    configured = settings or WorkerSettings.from_env()
    selected_inference = inference or _build_inference(configured)
    selected_store = store or FileManifestStore(
        configured.data_root, fsync_writes=configured.fsync_writes
    )
    health = HealthTracker(dict(configured.runtime_metadata))
    controller = GenerationController(
        selected_store,
        selected_inference,
        max_attempts=configured.max_generation_attempts,
        retry_delay_seconds=configured.retry_delay_seconds,
        runtime_metadata=configured.runtime_metadata,
        data_root=configured.data_root,
    )
    runtime = WorkerRuntime(
        settings=configured,
        inference=selected_inference,
        store=selected_store,
        controller=controller,
        health=health,
        authenticator=BearerAuthenticator(configured.credentials),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Yield immediately after scheduling boot so /v1/health is available during model load.
        runtime.boot_task = asyncio.create_task(_boot(runtime), name="imageforge-worker-boot")
        runtime.lag_task = asyncio.create_task(
            _monitor_loop_lag(runtime), name="imageforge-worker-loop-lag"
        )
        try:
            yield
        finally:
            await runtime.controller.shutdown()
            for task in (runtime.boot_task, runtime.lag_task):
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            await runtime.inference.shutdown()

    app = FastAPI(
        title="ImageForge Worker",
        version=WORKER_VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.runtime = runtime
    _install_error_handlers(app)
    _install_routes(app, runtime)
    return app


async def _boot(runtime: WorkerRuntime) -> None:
    try:
        await runtime.health.transition(HealthPhase.PROCESS, 1.0)
        await runtime.health.transition(HealthPhase.STORAGE, 0.1)
        await runtime.controller.initialize()
        await runtime.health.transition(HealthPhase.STORAGE, 1.0)
        await runtime.inference.startup(runtime.health.transition)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error_id = uuid.uuid4().hex
        # Keep the public health response deliberately opaque, but allow an
        # explicitly enabled one-time diagnostic Pod to expose the sanitized
        # exception text in its private container log for release debugging.
        diagnostic_message = "<redacted>"
        if os.environ.get("IMAGEFORGE_BOOT_DIAGNOSTICS") == "1":
            diagnostic_message = " ".join(str(exc).split())[:240]
        logger.error(
            "worker boot failed error_id=%s error_type=%s error_message=%s",
            error_id,
            type(exc).__name__,
            diagnostic_message,
        )
        await runtime.controller.release_lease_after_boot_failure()
        await runtime.health.fail(error_id)


def _build_inference(settings: WorkerSettings) -> InferenceAdapter:
    if settings.inference_backend == "fake":
        return FakeInferenceAdapter()
    if settings.inference_backend == "flux":
        return FluxInferenceAdapter(settings.model_cache_dir)
    return MageFlowInferenceAdapter(
        settings.model_cache_dir,
        settings.comfyui_root,
        profile_for_backend(settings.inference_backend),
    )


async def _monitor_loop_lag(runtime: WorkerRuntime, tick: float = 0.05) -> None:
    """Record how late the loop reschedules a trivial sleep.

    A cooperative loop returns within roughly `tick`. Synchronous volume I/O
    inside a coroutine cannot be preempted, so the overshoot is a direct measure
    of how long the worker was blocked -- attributable without guessing from the
    outside, which is how the status-rescan stall stayed hidden.
    """

    loop = asyncio.get_running_loop()
    previous = loop.time()
    while True:
        await asyncio.sleep(tick)
        now = loop.time()
        lag_ms = max(0.0, (now - previous - tick) * 1000)
        previous = now
        if lag_ms > runtime.loop_lag_recent_ms:
            runtime.loop_lag_recent_ms = lag_ms
        if lag_ms > runtime.loop_lag_peak_ms:
            runtime.loop_lag_peak_ms = lag_ms


def _diagnostics(runtime: WorkerRuntime) -> dict[str, Any]:
    """Unauthenticated counters describing worker I/O and loop responsiveness.

    Additive only. The desktop health contract validates named fields, so this
    block cannot change how a release is accepted.
    """

    store = runtime.store
    recent = runtime.loop_lag_recent_ms
    runtime.loop_lag_recent_ms = 0.0
    return {
        "volume_manifest_reads": getattr(store, "volume_manifest_reads", None),
        "manifest_cache_hits": getattr(store, "manifest_cache_hits", None),
        "artifact_digest_computations": getattr(
            runtime.controller, "artifact_digest_computations", None
        ),
        "loop_lag_recent_ms": round(recent, 3),
        "loop_lag_peak_ms": round(runtime.loop_lag_peak_ms, 3),
    }


def _install_routes(app: FastAPI, runtime: WorkerRuntime) -> None:
    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        payload = await runtime.health.snapshot(runtime.inference.gpu_snapshot())
        payload["diagnostics"] = _diagnostics(runtime)
        return payload

    @app.get("/v1/status", response_model=StatusResponse)
    async def status(principal: PrincipalDependency) -> StatusResponse:
        return await runtime.controller.status(principal, ready=runtime.health.ready)

    @app.put("/v1/studio/sessions/{session_id}", response_model=StudioStateResponse)
    async def heartbeat_studio_session(
        principal: PrincipalDependency,
        session_id: StudioId,
        request: Annotated[HeartbeatRequest, Body()],
    ) -> StudioStateResponse:
        return await runtime.controller.studio_heartbeat(principal, session_id, request)

    @app.get("/v1/studio/sessions/{session_id}", response_model=StudioStateResponse)
    async def get_studio_state(
        principal: PrincipalDependency, session_id: StudioId
    ) -> StudioStateResponse:
        return await runtime.controller.studio_state(principal, session_id)

    @app.post("/v1/studio/stop-requests", response_model=StudioStateResponse, status_code=201)
    async def create_gpu_stop_request(
        principal: PrincipalDependency,
        request: Annotated[CreateStopRequest, Body()],
    ) -> StudioStateResponse:
        return await runtime.controller.request_gpu_stop(principal, request)

    @app.post(
        "/v1/studio/stop-requests/{request_id}/responses",
        response_model=StudioStateResponse,
    )
    async def respond_to_gpu_stop(
        principal: PrincipalDependency,
        request_id: StudioId,
        request: Annotated[StopResponseRequest, Body()],
    ) -> StudioStateResponse:
        return await runtime.controller.respond_to_gpu_stop(principal, request_id, request)

    @app.post(
        "/v1/studio/stop-requests/{request_id}/finalize",
        response_model=StudioStateResponse,
    )
    async def finalize_gpu_stop(
        principal: PrincipalDependency,
        request_id: StudioId,
        request: Annotated[FinalizeStopRequest, Body()],
    ) -> StudioStateResponse:
        return await runtime.controller.finalize_gpu_stop(principal, request_id, request)

    @app.post(
        "/v1/studio/stop-requests/{request_id}/cancel",
        response_model=StudioStateResponse,
    )
    async def cancel_gpu_stop(
        principal: PrincipalDependency,
        request_id: StudioId,
        request: Annotated[CancelStopRequest, Body()],
    ) -> StudioStateResponse:
        return await runtime.controller.cancel_gpu_stop(principal, request_id, request)

    @app.get("/v1/studio/gpu-switches/{switch_id}", response_model=GpuSwitchLookupResponseV1)
    async def get_gpu_switch(
        principal: PrincipalDependency,
        switch_id: StudioId,
        session_id: StudioSessionQuery,
    ) -> GpuSwitchLookupResponseV1:
        return await runtime.controller.get_gpu_switch(principal, switch_id, session_id)

    @app.get(
        "/v1/internal/gpu-switches/{switch_id}/owner",
        response_model=NativeWorkerGpuSwitchOwnerLookupV1,
    )
    async def get_gpu_switch_owner(
        principal: PrincipalDependency,
        switch_id: StudioId,
        session_id: StudioSessionQuery,
    ) -> NativeWorkerGpuSwitchOwnerLookupV1:
        return await runtime.controller.get_gpu_switch_owner(principal, switch_id, session_id)

    @app.get(
        "/v1/internal/gpu-switches/{switch_id}/runtime-identity",
        response_model=WorkerGpuSwitchRuntimeIdentityV1,
    )
    async def get_gpu_switch_runtime_identity(
        principal: PrincipalDependency,
        switch_id: StudioId,
        session_id: StudioSessionQuery,
    ) -> WorkerGpuSwitchRuntimeIdentityV1:
        return await runtime.controller.gpu_switch_runtime_identity(
            principal, switch_id, session_id
        )

    @app.post(
        "/v1/studio/gpu-switches",
        response_model=NativeWorkerGpuSwitchCreateResponseV1,
        status_code=201,
    )
    async def create_gpu_switch(
        principal: PrincipalDependency,
        request: Annotated[CreateGpuSwitchRequestV1, Body()],
    ) -> NativeWorkerGpuSwitchCreateResponseV1:
        return await runtime.controller.request_gpu_switch(principal, request)

    @app.post(
        "/v1/internal/gpu-switches/{switch_id}/settle-create",
        response_model=NativeWorkerGpuSwitchOwnerLookupV1,
    )
    async def settle_gpu_switch_create(
        principal: PrincipalDependency,
        switch_id: StudioId,
        request: Annotated[SettleGpuSwitchCreateRequestV1, Body()],
    ) -> NativeWorkerGpuSwitchOwnerLookupV1:
        return await runtime.controller.settle_gpu_switch_create(principal, switch_id, request)

    @app.post(
        "/v1/studio/gpu-switches/{switch_id}/responses",
        response_model=StudioStateResponse,
    )
    async def respond_to_gpu_switch(
        principal: PrincipalDependency,
        switch_id: StudioId,
        request: Annotated[GpuSwitchResponseRequestV1, Body()],
    ) -> StudioStateResponse:
        return await runtime.controller.respond_to_gpu_switch(principal, switch_id, request)

    @app.post(
        "/v1/studio/gpu-switches/{switch_id}/finalize",
        response_model=StudioStateResponse,
    )
    async def finalize_gpu_switch(
        principal: PrincipalDependency,
        switch_id: StudioId,
        request: Annotated[FinalizeGpuSwitchRequestV1, Body()],
    ) -> StudioStateResponse:
        return await runtime.controller.finalize_gpu_switch(principal, switch_id, request)

    @app.post(
        "/v1/studio/gpu-switches/{switch_id}/delete-intent",
        response_model=StudioStateResponse,
    )
    async def mark_gpu_switch_delete_intent(
        principal: PrincipalDependency,
        switch_id: StudioId,
        request: Annotated[DeleteIntentGpuSwitchRequestV1, Body()],
    ) -> StudioStateResponse:
        return await runtime.controller.mark_gpu_switch_delete_intent(principal, switch_id, request)

    @app.post(
        "/v1/studio/gpu-switches/{switch_id}/adopt",
        response_model=StudioStateResponse,
    )
    async def adopt_gpu_switch_replacement(
        principal: PrincipalDependency,
        switch_id: StudioId,
        request: Annotated[AdoptGpuSwitchRequestV1, Body()],
    ) -> StudioStateResponse:
        return await runtime.controller.adopt_gpu_switch_replacement(principal, switch_id, request)

    @app.post(
        "/v1/studio/gpu-switches/{switch_id}/complete",
        response_model=StudioStateResponse,
    )
    async def complete_gpu_switch(
        principal: PrincipalDependency,
        switch_id: StudioId,
        request: Annotated[CompleteGpuSwitchRequestV1, Body()],
    ) -> StudioStateResponse:
        return await runtime.controller.complete_gpu_switch(principal, switch_id, request)

    @app.post(
        "/v1/studio/gpu-switches/{switch_id}/cancel",
        response_model=StudioStateResponse,
    )
    async def cancel_gpu_switch(
        principal: PrincipalDependency,
        switch_id: StudioId,
        request: Annotated[CancelGpuSwitchRequestV1, Body()],
    ) -> StudioStateResponse:
        return await runtime.controller.cancel_gpu_switch(principal, switch_id, request)

    @app.post("/v1/batches", response_model=BatchManifest, status_code=201)
    async def create_batch(
        principal: PrincipalDependency,
        request: Annotated[CreateBatchRequest, Body()],
    ) -> BatchManifest:
        await runtime.controller.preflight_new_submission()
        _require_model_ready(runtime)
        return await runtime.controller.create_batch(principal, request)

    @app.get("/v1/submissions/{client_submission_id}", response_model=BatchManifest)
    async def get_submission(
        principal: PrincipalDependency,
        client_submission_id: SubmissionId,
    ) -> BatchManifest:
        return await runtime.controller.get_submission(principal, client_submission_id)

    @app.get("/v1/batches/{batch_id}", response_model=BatchManifest)
    async def get_batch(principal: PrincipalDependency, batch_id: BatchId) -> BatchManifest:
        return await runtime.controller.get_batch(principal, str(batch_id))

    @app.post("/v1/batches/{batch_id}/pause", response_model=BatchManifest)
    async def pause_batch(principal: PrincipalDependency, batch_id: BatchId) -> BatchManifest:
        return await runtime.controller.pause(principal, str(batch_id))

    @app.post("/v1/batches/{batch_id}/resume", response_model=BatchManifest)
    async def resume_batch(principal: PrincipalDependency, batch_id: BatchId) -> BatchManifest:
        _require_model_ready(runtime)
        return await runtime.controller.resume(principal, str(batch_id))

    @app.post("/v1/batches/{batch_id}/cancel", response_model=BatchManifest)
    async def cancel_batch(principal: PrincipalDependency, batch_id: BatchId) -> BatchManifest:
        return await runtime.controller.cancel(principal, str(batch_id))

    @app.post("/v1/batches/{batch_id}/retry-failed", response_model=BatchManifest)
    async def retry_failed(principal: PrincipalDependency, batch_id: BatchId) -> BatchManifest:
        _require_model_ready(runtime)
        return await runtime.controller.retry_failed(principal, str(batch_id))

    @app.get("/v1/batches/{batch_id}/artifacts/{index}")
    async def full_artifact(
        principal: PrincipalDependency, batch_id: BatchId, index: ImageIndex
    ) -> FileResponse:
        descriptor = await runtime.controller.artifact(
            principal, str(batch_id), index, preview=False
        )
        return _file_response(descriptor, attachment=True)

    @app.get("/v1/batches/{batch_id}/previews/{index}")
    async def preview_artifact(
        principal: PrincipalDependency, batch_id: BatchId, index: ImageIndex
    ) -> FileResponse:
        descriptor = await runtime.controller.artifact(
            principal, str(batch_id), index, preview=True
        )
        return _file_response(descriptor, attachment=False)

    @app.post("/v1/batches/{batch_id}/receipts", response_model=ReceiptResponse)
    async def receipts(
        principal: PrincipalDependency,
        batch_id: BatchId,
        request: Annotated[ReceiptRequest, Body()],
    ) -> ReceiptResponse:
        return await runtime.controller.accept_receipts(principal, str(batch_id), request)


def _require_model_ready(runtime: WorkerRuntime) -> None:
    if not runtime.health.ready:
        raise WorkerError(
            status_code=503,
            code="worker_not_ready",
            message="The worker model is not ready yet.",
            details={"phase": runtime.health.phase.value},
        )


def _file_response(descriptor: ArtifactDescriptor, *, attachment: bool) -> FileResponse:
    digest = base64.b64encode(bytes.fromhex(descriptor.sha256)).decode("ascii")
    headers = {
        "X-ImageForge-SHA256": descriptor.sha256,
        "X-Checksum-SHA256": descriptor.sha256,
        "Digest": f"sha-256={digest}",
        "ETag": f'"{descriptor.sha256}"',
        "Cache-Control": "private, max-age=86400, immutable",
        "Content-Length": str(descriptor.size_bytes),
    }
    return FileResponse(
        descriptor.path,
        media_type=descriptor.media_type,
        filename=descriptor.download_name if attachment else None,
        headers=headers,
    )


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(WorkerError)
    async def worker_error_handler(_: Request, exc: WorkerError) -> JSONResponse:
        # Every public worker error uses one exact three-field object. A null
        # details value is explicit rather than omitted, which keeps the
        # Python -> native -> renderer contract strict without disclosing
        # exception, path, owner, or submission-key internals.
        error: dict[str, Any] = {
            "code": exc.code,
            "message": exc.message,
            "details": dict(exc.details) if exc.details is not None else None,
        }
        return JSONResponse(
            status_code=exc.status_code,
            content={"schema_version": API_SCHEMA_VERSION, "error": error},
            headers=dict(exc.headers),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Do not echo Pydantic's `input` field; it may contain a prompt.
        issues = [
            {
                "location": ".".join(str(part) for part in issue.get("loc", ())),
                "type": issue.get("type", "validation_error"),
            }
            for issue in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "schema_version": API_SCHEMA_VERSION,
                "error": {
                    "code": "validation_error",
                    "message": "The request did not match the worker API contract.",
                    "details": {"issues": issues},
                },
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "schema_version": API_SCHEMA_VERSION,
                "error": {
                    "code": "not_found",
                    "message": "The endpoint does not exist.",
                    "details": None,
                },
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        error_id = uuid.uuid4().hex
        logger.error(
            "unhandled request failure method=%s path=%s error_id=%s error_type=%s",
            request.method,
            request.url.path,
            error_id,
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "schema_version": API_SCHEMA_VERSION,
                "error": {
                    "code": "internal_error",
                    "message": "The worker could not complete the request.",
                    "details": {"error_id": error_id},
                },
            },
        )
