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

from fastapi import Body, Depends, FastAPI, Request
from fastapi import Path as PathParameter
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .auth import BearerAuthenticator, Principal
from .config import WorkerSettings
from .constants import API_SCHEMA_VERSION, MAX_PROMPTS, WORKER_VERSION
from .controller import ArtifactDescriptor, GenerationController
from .domain import (
    BatchManifest,
    CreateBatchRequest,
    HealthPhase,
    ReceiptRequest,
    ReceiptResponse,
    StatusResponse,
)
from .errors import WorkerError
from .health import HealthTracker
from .inference import FakeInferenceAdapter, FluxInferenceAdapter, InferenceAdapter
from .persistence import FileManifestStore, ManifestStore

logger = logging.getLogger("imageforge_worker.app")


async def _authenticated_principal(request: Request) -> Principal:
    runtime: WorkerRuntime = request.app.state.runtime
    return await runtime.authenticator(request)


PrincipalDependency = Annotated[Principal, Depends(_authenticated_principal)]
BatchId = Annotated[UUID, PathParameter(description="Server-generated batch UUID")]
ImageIndex = Annotated[int, PathParameter(ge=1, le=MAX_PROMPTS)]


@dataclass(slots=True)
class WorkerRuntime:
    settings: WorkerSettings
    inference: InferenceAdapter
    store: ManifestStore
    controller: GenerationController
    health: HealthTracker
    authenticator: BearerAuthenticator
    boot_task: asyncio.Task[None] | None = None


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
        try:
            yield
        finally:
            await runtime.controller.shutdown()
            if runtime.boot_task is not None and not runtime.boot_task.done():
                runtime.boot_task.cancel()
                try:
                    await runtime.boot_task
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
    return FluxInferenceAdapter(settings.model_cache_dir)


def _install_routes(app: FastAPI, runtime: WorkerRuntime) -> None:
    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return await runtime.health.snapshot(runtime.inference.gpu_snapshot())

    @app.get("/v1/status", response_model=StatusResponse)
    async def status(principal: PrincipalDependency) -> StatusResponse:
        return await runtime.controller.status(principal, ready=runtime.health.ready)

    @app.post("/v1/batches", response_model=BatchManifest, status_code=201)
    async def create_batch(
        principal: PrincipalDependency,
        request: Annotated[CreateBatchRequest, Body()],
    ) -> BatchManifest:
        _require_model_ready(runtime)
        return await runtime.controller.create_batch(principal, request)

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
        error: dict[str, Any] = {"code": exc.code, "message": exc.message}
        if exc.details is not None:
            error["details"] = dict(exc.details)
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
                "error": {"code": "not_found", "message": "The endpoint does not exist."},
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
