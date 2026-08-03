from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from .auth import Principal
from .constants import MAX_REFERENCE_PIXELS
from .coordination import (
    FINALIZATION_TTL_SECONDS,
    CancelStopRequest,
    CoordinationClock,
    CoordinationIdentity,
    CreateStopRequest,
    FinalizeStopRequest,
    HeartbeatRequest,
    StopRequestView,
    StopResponseRequest,
    StudioCoordinator,
    StudioStateResponse,
)
from .domain import (
    LOCK_HOLDING_STATES,
    NONTERMINAL_IMAGE_STATES,
    AttemptRecord,
    BatchManifest,
    BatchOwner,
    BatchProgress,
    BatchState,
    BatchSummary,
    CreateBatchRequest,
    GenerationSettings,
    ImageRecord,
    ImageState,
    ReceiptRequest,
    ReceiptResponse,
    ReferenceInput,
    SafeImageError,
    StatusPermissions,
    StatusResponse,
    StoredReceipt,
    StoredReference,
    utc_now,
)
from .errors import InferenceFailure, WorkerError
from .inference import GenerationJob, InferenceAdapter, InferenceResult
from .persistence import ManifestStore, SharedGpuStopGuard

logger = logging.getLogger("imageforge_worker.controller")


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    path: Path
    sha256: str
    size_bytes: int
    media_type: str
    download_name: str


@dataclass(frozen=True, slots=True)
class _ClaimedAttempt:
    job: GenerationJob
    attempt: int
    started_at: str


@dataclass(frozen=True, slots=True)
class _PreparedReference:
    metadata: StoredReference
    payload: bytes


class GenerationController:
    """Owns the process-local controller and shared-volume generation lease."""

    def __init__(
        self,
        store: ManifestStore,
        inference: InferenceAdapter,
        *,
        max_attempts: int,
        retry_delay_seconds: float,
        coordination_clock: CoordinationClock | None = None,
        coordination_finalization_ttl_seconds: int = FINALIZATION_TTL_SECONDS,
    ) -> None:
        self.store = store
        self.inference = inference
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._lock = asyncio.Lock()
        self.coordination = StudioCoordinator(
            clock=coordination_clock,
            finalization_ttl_seconds=coordination_finalization_ttl_seconds,
        )
        self._active_batch_id: str | None = None
        self._runner_task: asyncio.Task[None] | None = None
        self._stop_guard: SharedGpuStopGuard | None = None
        self._stop_guard_is_local = False
        self._stop_guard_expiry_task: asyncio.Task[None] | None = None
        self._initialized = False
        self._closing = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            if self._closing:
                raise RuntimeError("generation controller is shutting down")
            if not self.store.try_acquire_worker_presence():
                raise RuntimeError("worker startup is disabled during retention maintenance")
            try:
                self.store.initialize()
                if self.store.try_acquire_active_lease():
                    self._adopt_or_clear_shared_stop_guard_locked()
                    active = self._discover_active_locked(recover=True)
                    self._active_batch_id = active.batch_id if active is not None else None
                    if active is None and self._stop_guard is None:
                        self.store.release_active_lease()
                else:
                    active = self._discover_active_locked(recover=False)
                    self._active_batch_id = active.batch_id if active is not None else None
            except BaseException:
                self._abandon_stop_guard_locked()
                self.store.release_active_lease()
                self.store.release_worker_presence()
                raise
            self._initialized = True

    def _recover_manifest(self, manifest: BatchManifest) -> bool:
        changed = False
        invalidated_artifact = False
        for image in manifest.images:
            if image.status in {ImageState.READY, ImageState.DOWNLOADED}:
                if (
                    image.artifacts_expired
                    and image.status == ImageState.DOWNLOADED
                    and image.receipt is not None
                ):
                    continue
                if not self.store.verify_record_artifacts(manifest.batch_id, image):
                    self.store.quarantine_artifacts(manifest.batch_id, image.index)
                    image.status = ImageState.PENDING
                    image.clear_artifact()
                    image.error = None
                    image.attempts_in_cycle = 0
                    invalidated_artifact = True
                    changed = True
            elif image.status in {ImageState.GENERATING, ImageState.RETRYING}:
                self.store.quarantine_artifacts(manifest.batch_id, image.index)
                image.status = ImageState.PENDING
                image.error = None
                image.attempts_in_cycle = 0
                changed = True

        if manifest.cancel_requested and manifest.state in LOCK_HOLDING_STATES:
            self._finalize_cancel(manifest)
            return True
        if manifest.state == BatchState.RUNNING:
            manifest.state = BatchState.INTERRUPTED
            manifest.interrupted_at = utc_now()
            manifest.pause_requested = False
            changed = True
        elif manifest.state == BatchState.COMPLETED and invalidated_artifact:
            manifest.state = BatchState.INTERRUPTED
            manifest.completed_at = None
            manifest.interrupted_at = utc_now()
            changed = True
        return changed

    async def shutdown(self) -> None:
        self._closing = True
        runner_task = self._runner_task
        if runner_task is not None and not runner_task.done():
            runner_task.cancel()
            try:
                await runner_task
            except asyncio.CancelledError:
                pass
        guard_task = self._stop_guard_expiry_task
        self._stop_guard_expiry_task = None
        if guard_task is not None and not guard_task.done():
            guard_task.cancel()
            try:
                await guard_task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            try:
                if not self.store.active_lease_held or self._active_batch_id is None:
                    return
                manifest = self.store.load(self._active_batch_id)
                if manifest.state == BatchState.RUNNING:
                    for image in manifest.images:
                        if image.status in {ImageState.GENERATING, ImageState.RETRYING}:
                            self.store.quarantine_artifacts(manifest.batch_id, image.index)
                            image.status = ImageState.PENDING
                            image.error = None
                            image.attempts_in_cycle = 0
                    manifest.state = BatchState.INTERRUPTED
                    manifest.interrupted_at = utc_now()
                    manifest.pause_requested = False
                    self.store.save(manifest)
            finally:
                self.store.release_active_lease()
                self.store.release_worker_presence()
                self._stop_guard = None
                self._stop_guard_is_local = False

    async def release_lease_after_boot_failure(self) -> None:
        """Allow a healthy replacement Pod to adopt an interrupted batch."""

        async with self._lock:
            task = self._stop_guard_expiry_task
            self._stop_guard_expiry_task = None
            if task is not None and not task.done():
                task.cancel()
            self.store.release_active_lease()
            self._stop_guard = None
            self._stop_guard_is_local = False

    async def status(self, principal: Principal, *, ready: bool) -> StatusResponse:
        self._ensure_initialized()
        async with self._lock:
            self._reconcile_stop_guard_locked()
            active = self._refresh_active_observation_locked()
            shared_stop_guard = self._stop_guard or self.store.read_gpu_stop_guard()
            is_owner = active is not None and active.owner.user_id == principal.user_id
            summary = (
                BatchSummary(
                    batch_id=active.batch_id,
                    owner=active.owner,
                    state=active.state,
                    progress=active.progress,
                    pause_requested=active.pause_requested,
                    cancel_requested=active.cancel_requested,
                )
                if active
                else None
            )
            return StatusResponse(
                ready=ready,
                active_batch=summary,
                permissions=StatusPermissions(
                    can_create=ready and active is None and shared_stop_guard is None,
                    can_manage_active=is_owner and self.store.active_lease_held,
                    is_owner=is_owner,
                ),
            )

    async def studio_heartbeat(
        self,
        principal: Principal,
        session_id: str,
        request: HeartbeatRequest,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._reconcile_stop_guard_locked()
            active = self._refresh_active_observation_locked()
            response = self.coordination.heartbeat(principal, session_id, request, active)
            self._reconcile_stop_guard_locked()
            return self._project_shared_stop_guard_locked(response)

    async def studio_state(
        self, principal: Principal, session_id: str
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._reconcile_stop_guard_locked()
            active = self._refresh_active_observation_locked()
            response = self.coordination.state(principal, session_id, active)
            self._reconcile_stop_guard_locked()
            return self._project_shared_stop_guard_locked(response)

    async def request_gpu_stop(
        self, principal: Principal, request: CreateStopRequest
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._reconcile_stop_guard_locked()
            if self._stop_guard is not None:
                self._raise_gpu_stop_pending(self._stop_guard)
            self._raise_shared_stop_guard_if_present()
            active = self._refresh_active_observation_locked()
            response = self.coordination.create_stop_request(principal, request, active)
            self._reconcile_stop_guard_locked()
            return response

    async def respond_to_gpu_stop(
        self,
        principal: Principal,
        request_id: str,
        request: StopResponseRequest,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._reconcile_stop_guard_locked()
            active = self._refresh_active_observation_locked()
            response = self.coordination.respond(principal, request_id, request, active)
            self._reconcile_stop_guard_locked()
            return response

    async def finalize_gpu_stop(
        self,
        principal: Principal,
        request_id: str,
        request: FinalizeStopRequest,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._reconcile_stop_guard_locked()
            active = self._refresh_active_observation_locked()
            response = self.coordination.finalize(principal, request_id, request, active)
            if self._stop_guard is not None:
                if self._stop_guard_is_local:
                    return response
                self.coordination.rollback_finalization(request_id, request.finalization_id)
                self._raise_gpu_stop_pending(self._stop_guard)

            try:
                self._acquire_stop_guard_lease_locked()
            except WorkerError as exc:
                self.coordination.rollback_finalization(
                    request_id,
                    request.finalization_id,
                    generation_started=exc.code == "stop_blocked_by_active_batch",
                )
                raise

            stop = response.stop_request
            if stop is None or stop.finalization_expires_at is None:
                self.coordination.rollback_finalization(request_id, request.finalization_id)
                self.store.release_active_lease()
                raise RuntimeError("finalized GPU Stop response omitted its shared guard")
            guard = SharedGpuStopGuard(
                server_instance_id=response.server_instance_id,
                request_id=request_id,
                finalization_id=request.finalization_id,
                pod_id=stop.pod_id,
                gpu_display_name=stop.gpu_display_name,
                requester=stop.requester.display_name,
                requested_at=stop.requested_at,
                response_deadline=stop.response_deadline,
                expires_at=stop.finalization_expires_at,
            )
            try:
                self.store.write_gpu_stop_guard(guard)
            except BaseException:
                self.coordination.rollback_finalization(request_id, request.finalization_id)
                try:
                    self.store.clear_stale_gpu_stop_guard()
                finally:
                    self.store.release_active_lease()
                raise
            self._stop_guard = guard
            self._stop_guard_is_local = True
            self._schedule_stop_guard_expiry_locked(guard)
            return response

    async def cancel_gpu_stop(
        self,
        principal: Principal,
        request_id: str,
        request: CancelStopRequest,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._reconcile_stop_guard_locked()
            active = self._refresh_active_observation_locked()
            response = self.coordination.cancel(principal, request_id, request, active)
            self._reconcile_stop_guard_locked()
            return response

    async def create_batch(
        self, principal: Principal, request: CreateBatchRequest
    ) -> BatchManifest:
        self._ensure_initialized()
        prepared_references = self._prepare_references(request.references)
        async with self._lock:
            await self._acquire_for_new_batch_locked()
            active = self._refresh_active_observation_locked()
            if active is not None:
                self._raise_busy(active)
            try:
                self._admit_generation_locked()
            except BaseException:
                self._release_if_no_active_locked()
                raise
            now = utc_now()
            batch_id = str(uuid.uuid4())
            images = [
                ImageRecord(index=index, prompt=prompt, seed=request.base_seed + index - 1)
                for index, prompt in enumerate(request.prompts, start=1)
            ]
            manifest = BatchManifest(
                batch_id=batch_id,
                owner=BatchOwner(user_id=principal.user_id, display_name=principal.display_name),
                state=BatchState.RUNNING,
                created_at=now,
                updated_at=now,
                settings=GenerationSettings.for_aspect_ratio(request.aspect_ratio),
                references=[reference.metadata for reference in prepared_references],
                images=images,
                progress=BatchProgress(total=len(images)),
            )
            try:
                self.store.create(
                    manifest,
                    reference_payloads=[
                        (reference.metadata.filename, reference.payload)
                        for reference in prepared_references
                    ],
                )
            except BaseException:
                self.store.release_active_lease()
                raise
            self._active_batch_id = batch_id
            self._launch_runner_locked(batch_id)
            return manifest

    async def get_batch(self, principal: Principal, batch_id: str) -> BatchManifest:
        self._ensure_initialized()
        async with self._lock:
            return self._load_owned(principal, batch_id)

    async def pause(self, principal: Principal, batch_id: str) -> BatchManifest:
        self._ensure_initialized()
        async with self._lock:
            manifest = self._load_owned(principal, batch_id)
            if manifest.state == BatchState.PAUSED:
                return manifest
            await self._require_mutation_lease_locked()
            manifest = self._load_owned(principal, batch_id)
            if manifest.state != BatchState.RUNNING:
                self._release_if_no_active_locked()
                self._invalid_state(manifest, "pause")
            if any(image.status == ImageState.GENERATING for image in manifest.images):
                manifest.pause_requested = True
            else:
                manifest.state = BatchState.PAUSED
                manifest.pause_requested = False
            self.store.save(manifest)
            return manifest

    async def resume(self, principal: Principal, batch_id: str) -> BatchManifest:
        self._ensure_initialized()
        async with self._lock:
            manifest = self._load_owned(principal, batch_id)
            if manifest.state == BatchState.RUNNING:
                if self.store.active_lease_held:
                    return manifest
                await self._require_mutation_lease_locked()
                manifest = self._load_owned(principal, batch_id)
                if manifest.state == BatchState.RUNNING:
                    return manifest
            if manifest.state not in {BatchState.PAUSED, BatchState.INTERRUPTED}:
                self._invalid_state(manifest, "resume")
            await self._acquire_for_new_batch_locked()
            manifest = self._load_owned(principal, batch_id)
            if manifest.state not in {BatchState.PAUSED, BatchState.INTERRUPTED}:
                self._release_if_no_active_locked()
                self._invalid_state(manifest, "resume")
            if self._active_batch_id not in {None, batch_id}:
                active = self._refresh_active_observation_locked()
                if active is not None:
                    self._raise_busy(active)
            self._admit_generation_locked()
            for image in manifest.images:
                if image.status in {ImageState.GENERATING, ImageState.RETRYING}:
                    self.store.quarantine_artifacts(batch_id, image.index)
                    image.status = ImageState.PENDING
                    image.attempts_in_cycle = 0
                    image.error = None
            manifest.state = BatchState.RUNNING
            manifest.interrupted_at = None
            manifest.pause_requested = False
            manifest.cancel_requested = False
            self.store.save(manifest)
            self._active_batch_id = batch_id
            self._launch_runner_locked(batch_id)
            return manifest

    async def cancel(self, principal: Principal, batch_id: str) -> BatchManifest:
        self._ensure_initialized()
        async with self._lock:
            manifest = self._load_owned(principal, batch_id)
            if manifest.state == BatchState.CANCELLED:
                self._release_if_no_active_locked()
                return manifest
            if manifest.state not in LOCK_HOLDING_STATES:
                self._release_if_no_active_locked()
                self._invalid_state(manifest, "cancel")
            await self._require_mutation_lease_locked()
            manifest = self._load_owned(principal, batch_id)
            if manifest.state == BatchState.CANCELLED:
                self._release_if_no_active_locked()
                return manifest
            if manifest.state not in LOCK_HOLDING_STATES:
                self._release_if_no_active_locked()
                self._invalid_state(manifest, "cancel")
            if manifest.state == BatchState.RUNNING and any(
                image.status == ImageState.GENERATING for image in manifest.images
            ):
                manifest.cancel_requested = True
                manifest.pause_requested = False
                self.store.save(manifest)
            else:
                self._finalize_cancel(manifest)
                self.store.save(manifest)
                self._release_batch_lease_locked(batch_id)
            return manifest

    async def retry_failed(self, principal: Principal, batch_id: str) -> BatchManifest:
        self._ensure_initialized()
        async with self._lock:
            manifest = self._load_owned(principal, batch_id)
            if manifest.state not in {BatchState.COMPLETED, BatchState.FAILED}:
                self._release_if_no_active_locked()
                self._invalid_state(manifest, "retry failed images in")
            failed = [image for image in manifest.images if image.status == ImageState.FAILED]
            if not failed:
                raise WorkerError(
                    status_code=409,
                    code="no_failed_images",
                    message="This batch has no failed images to retry.",
                )
            await self._acquire_for_new_batch_locked()
            active = self._refresh_active_observation_locked()
            if active is not None:
                self._raise_busy(active)
            manifest = self._load_owned(principal, batch_id)
            if manifest.state not in {BatchState.COMPLETED, BatchState.FAILED}:
                self._release_if_no_active_locked()
                self._invalid_state(manifest, "retry failed images in")
            failed = [image for image in manifest.images if image.status == ImageState.FAILED]
            if not failed:
                self._release_if_no_active_locked()
                raise WorkerError(
                    status_code=409,
                    code="no_failed_images",
                    message="This batch has no failed images to retry.",
                )
            try:
                self._admit_generation_locked()
            except BaseException:
                self._release_if_no_active_locked()
                raise
            for image in failed:
                image.status = ImageState.PENDING
                image.error = None
                image.attempts_in_cycle = 0
                image.retry_rounds += 1
                image.finished_at = None
            manifest.state = BatchState.RUNNING
            manifest.completed_at = None
            manifest.interrupted_at = None
            manifest.pause_requested = False
            manifest.cancel_requested = False
            self.store.save(manifest)
            self._active_batch_id = batch_id
            self._launch_runner_locked(batch_id)
            return manifest

    async def accept_receipts(
        self, principal: Principal, batch_id: str, request: ReceiptRequest
    ) -> ReceiptResponse:
        self._ensure_initialized()
        async with self._lock:
            self._load_owned(principal, batch_id)
            already_held = self.store.active_lease_held
            await self._require_mutation_lease_locked()
            try:
                return self._accept_receipts_locked(principal, batch_id, request)
            finally:
                if not already_held:
                    self._release_if_no_active_locked()

    def _accept_receipts_locked(
        self, principal: Principal, batch_id: str, request: ReceiptRequest
    ) -> ReceiptResponse:
        manifest = self._load_owned(principal, batch_id)
        records: list[tuple[ImageRecord, str, int]] = []
        for receipt in request.receipts:
            image = self._find_image(manifest, receipt.index)
            if image.artifacts_expired:
                raise WorkerError(
                    status_code=410,
                    code="artifact_expired",
                    message=f"Image {receipt.index} is no longer retained by the worker.",
                )
            if image.status not in {ImageState.READY, ImageState.DOWNLOADED}:
                raise WorkerError(
                    status_code=409,
                    code="artifact_not_ready",
                    message=f"Image {receipt.index} is not ready for acknowledgement.",
                )
            if image.sha256 is None or image.size_bytes is None:
                raise WorkerError(
                    status_code=409,
                    code="artifact_not_ready",
                    message=f"Image {receipt.index} is not ready for acknowledgement.",
                )
            if not secrets.compare_digest(image.sha256, receipt.sha256) or (
                image.size_bytes != receipt.size_bytes
            ):
                raise WorkerError(
                    status_code=409,
                    code="checksum_mismatch",
                    message=f"Image {receipt.index} did not match the server checksum and size.",
                    details={"index": receipt.index},
                )
            records.append((image, receipt.sha256, receipt.size_bytes))

        acknowledged_at = utc_now()
        for image, sha256, size_bytes in records:
            image.status = ImageState.DOWNLOADED
            image.receipt = StoredReceipt(
                user_id=principal.user_id,
                sha256=sha256,
                size_bytes=size_bytes,
                acknowledged_at=acknowledged_at,
            )
        self.store.save(manifest)
        return ReceiptResponse(
            batch_id=batch_id,
            accepted=[receipt.index for receipt in request.receipts],
            progress=manifest.progress,
        )

    async def artifact(
        self, principal: Principal, batch_id: str, index: int, *, preview: bool
    ) -> ArtifactDescriptor:
        self._ensure_initialized()
        async with self._lock:
            manifest = self._load_owned(principal, batch_id)
            image = self._find_image(manifest, index)
            if image.artifacts_expired:
                raise WorkerError(
                    status_code=410,
                    code="artifact_expired",
                    message=f"Image {index} is no longer retained by the worker.",
                )
            if image.status not in {ImageState.READY, ImageState.DOWNLOADED}:
                raise WorkerError(
                    status_code=409,
                    code="artifact_not_ready",
                    message=f"Image {index} is not ready.",
                )
            relative_name = image.preview_filename if preview else image.filename
            checksum = image.preview_sha256 if preview else image.sha256
            size = image.preview_size_bytes if preview else image.size_bytes
            if relative_name is None or checksum is None or size is None:
                raise WorkerError(
                    status_code=409,
                    code="artifact_not_ready",
                    message=f"Image {index} is not ready.",
                )
            path = self.store.artifact_path(batch_id, relative_name)
            if not self._file_matches(path, size, checksum):
                raise WorkerError(
                    status_code=409,
                    code="artifact_corrupt",
                    message=f"Image {index} failed server checksum validation.",
                )
            return ArtifactDescriptor(
                path=path,
                sha256=checksum,
                size_bytes=size,
                media_type="image/webp" if preview else "image/jpeg",
                download_name=path.name,
            )

    async def _claim_next_attempt(self, batch_id: str) -> _ClaimedAttempt | None:
        async with self._lock:
            if self._active_batch_id != batch_id:
                return None
            manifest = self.store.load(batch_id)
            if manifest.state != BatchState.RUNNING:
                return None
            if manifest.cancel_requested:
                self._finalize_cancel(manifest)
                self.store.save(manifest)
                self._release_batch_lease_locked(batch_id)
                return None
            if manifest.pause_requested:
                manifest.state = BatchState.PAUSED
                manifest.pause_requested = False
                self.store.save(manifest)
                return None
            image = next(
                (
                    candidate
                    for candidate in manifest.images
                    if candidate.status in {ImageState.PENDING, ImageState.RETRYING}
                ),
                None,
            )
            if image is None:
                manifest.state = BatchState.COMPLETED
                manifest.completed_at = utc_now()
                self.store.save(manifest)
                self._release_batch_lease_locked(batch_id)
                return None
            image.status = ImageState.GENERATING
            image.attempts += 1
            image.attempts_in_cycle += 1
            image.started_at = utc_now()
            image.finished_at = None
            image.error = None
            self.store.save(manifest)
            references = self._load_reference_images(manifest)
            return _ClaimedAttempt(
                job=GenerationJob(
                    index=image.index,
                    prompt=image.prompt,
                    seed=image.seed,
                    settings=manifest.settings,
                    references=references,
                ),
                attempt=image.attempts,
                started_at=image.started_at,
            )

    async def _record_success(
        self, batch_id: str, claimed: _ClaimedAttempt, result: InferenceResult
    ) -> None:
        async with self._lock:
            manifest = self.store.load(batch_id)
            image = self._find_image(manifest, claimed.job.index)
            if image.status != ImageState.GENERATING or image.attempts != claimed.attempt:
                raise RuntimeError("generation result no longer matches its manifest attempt")
            filename, preview_filename = self.store.write_artifacts(
                batch_id, image.index, result.jpeg, result.preview
            )
            finished_at = utc_now()
            image.filename = filename
            image.preview_filename = preview_filename
            image.sha256 = hashlib.sha256(result.jpeg).hexdigest()
            image.preview_sha256 = hashlib.sha256(result.preview).hexdigest()
            image.size_bytes = len(result.jpeg)
            image.preview_size_bytes = len(result.preview)
            image.generation_ms = (
                result.inference_ms + result.jpeg_encode_ms + result.preview_encode_ms
            )
            image.finished_at = finished_at
            image.status = ImageState.READY
            image.error = None
            image.artifacts_cleanup_started_at = None
            image.artifacts_deleted_at = None
            image.attempt_history.append(
                AttemptRecord(
                    attempt=claimed.attempt,
                    started_at=claimed.started_at,
                    finished_at=finished_at,
                    status="succeeded",
                    inference_ms=result.inference_ms,
                    jpeg_encode_ms=result.jpeg_encode_ms,
                    preview_encode_ms=result.preview_encode_ms,
                )
            )
            # Artifact writes and hashes are durable before this manifest publishes readiness.
            self.store.save(manifest)

    async def _record_failure(
        self, batch_id: str, claimed: _ClaimedAttempt, error_code: str
    ) -> bool:
        async with self._lock:
            manifest = self.store.load(batch_id)
            image = self._find_image(manifest, claimed.job.index)
            if image.status != ImageState.GENERATING or image.attempts != claimed.attempt:
                return False
            finished_at = utc_now()
            terminal = image.attempts_in_cycle >= self.max_attempts
            image.status = ImageState.FAILED if terminal else ImageState.RETRYING
            image.finished_at = finished_at if terminal else None
            image.error = SafeImageError(
                code=error_code,
                message=(
                    "Generation failed after three attempts."
                    if terminal
                    else "Generation attempt failed and will be retried."
                ),
            )
            image.attempt_history.append(
                AttemptRecord(
                    attempt=claimed.attempt,
                    started_at=claimed.started_at,
                    finished_at=finished_at,
                    status="failed",
                    error_code=error_code,
                )
            )
            self.store.save(manifest)
            return not terminal

    async def _run_batch(self, batch_id: str) -> None:
        try:
            while not self._closing:
                claimed = await self._claim_next_attempt(batch_id)
                if claimed is None:
                    return
                try:
                    result = await self.inference.generate(claimed.job)
                    self._validate_result(result, claimed.job.settings)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error_code = (
                        exc.code if isinstance(exc, InferenceFailure) else "inference_failed"
                    )
                    logger.warning(
                        "generation attempt failed batch_id=%s index=%d error_type=%s",
                        batch_id,
                        claimed.job.index,
                        type(exc).__name__,
                    )
                    will_retry = await self._record_failure(batch_id, claimed, error_code)
                    if will_retry and self.retry_delay_seconds:
                        await asyncio.sleep(self.retry_delay_seconds)
                    continue
                finally:
                    for reference in claimed.job.references:
                        reference.close()
                await self._record_success(batch_id, claimed, result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_id = uuid.uuid4().hex
            logger.error(
                "batch controller failed batch_id=%s error_id=%s error_type=%s",
                batch_id,
                error_id,
                type(exc).__name__,
            )
            await self._mark_fatal(batch_id)

    async def _mark_fatal(self, batch_id: str) -> None:
        async with self._lock:
            try:
                manifest = self.store.load(batch_id)
                for image in manifest.images:
                    if image.status in NONTERMINAL_IMAGE_STATES:
                        image.status = ImageState.FAILED
                        image.finished_at = utc_now()
                        image.error = SafeImageError(
                            code="worker_internal_error",
                            message="The worker could not finish this image.",
                        )
                manifest.state = BatchState.FAILED
                manifest.pause_requested = False
                manifest.cancel_requested = False
                manifest.completed_at = utc_now()
                self.store.save(manifest)
            finally:
                self._release_batch_lease_locked(batch_id)

    def _launch_runner_locked(self, batch_id: str) -> None:
        if self._closing:
            raise RuntimeError("generation controller is shutting down")
        if self._runner_task is not None and not self._runner_task.done():
            raise RuntimeError("a generation runner is already active")
        task = asyncio.create_task(self._run_batch(batch_id), name=f"batch-{batch_id}")
        self._runner_task = task
        task.add_done_callback(self._runner_finished)

    def _runner_finished(self, task: asyncio.Task[None]) -> None:
        if self._runner_task is task:
            self._runner_task = None

    def _admit_generation_locked(self) -> None:
        self._reconcile_stop_guard_locked(release_lease=False)
        if self._stop_guard is not None:
            self._raise_gpu_stop_pending(self._stop_guard)
        self.coordination.admit_generation()

    def _acquire_stop_guard_lease_locked(self) -> None:
        if self.store.active_lease_held:
            active = self._refresh_active_observation_locked()
            if active is not None:
                self._raise_stop_blocked_by_active_batch(active)
            adopted = self._adopt_or_clear_shared_stop_guard_locked()
            if adopted is not None:
                self._raise_gpu_stop_pending(adopted)
            return

        if not self.store.try_acquire_active_lease():
            self._raise_shared_stop_guard_if_present()
            active = self._discover_active_locked(recover=False)
            self._active_batch_id = active.batch_id if active is not None else None
            if active is not None:
                self._raise_stop_blocked_by_active_batch(active)
            raise WorkerError(
                status_code=423,
                code="worker_volume_locked",
                message="Another worker process is updating the shared volume.",
            )

        try:
            adopted = self._adopt_or_clear_shared_stop_guard_locked()
            if adopted is not None:
                self._raise_gpu_stop_pending(adopted)
            active = self._discover_active_locked(recover=True)
        except WorkerError:
            # An adopted guard intentionally keeps the lease until its expiry.
            raise
        except BaseException:
            self._abandon_stop_guard_locked()
            self.store.release_active_lease()
            raise
        self._active_batch_id = active.batch_id if active is not None else None
        if active is not None:
            # Recovery owns the batch lease now. Retain it while returning the
            # unconditional active-batch Stop veto.
            self._raise_stop_blocked_by_active_batch(active)

    def _adopt_or_clear_shared_stop_guard_locked(
        self,
    ) -> SharedGpuStopGuard | None:
        guard = self.store.read_gpu_stop_guard()
        if guard is not None and (
            0 < self._shared_guard_remaining_seconds(guard) <= FINALIZATION_TTL_SECONDS
        ):
            self._stop_guard = guard
            self._stop_guard_is_local = False
            self._schedule_stop_guard_expiry_locked(guard)
            return guard
        # A released lease proves there is no live owner. Only its new owner
        # may remove expired, malformed, or partial crash residue.
        self.store.clear_stale_gpu_stop_guard()
        return None

    def _reconcile_stop_guard_locked(self, *, release_lease: bool = True) -> None:
        guard = self._stop_guard
        if guard is None:
            return
        remaining = self._stop_guard_remaining_seconds(guard)
        if remaining is None or remaining <= 0:
            self._clear_stop_guard_locked(release_lease=release_lease)

    def _project_shared_stop_guard_locked(
        self, response: StudioStateResponse
    ) -> StudioStateResponse:
        """Expose an adopted guard without reviving its old deletion authority."""

        if response.stop_request is not None:
            return response
        guard = self._stop_guard or self.store.read_gpu_stop_guard()
        if guard is None:
            return response
        occupied_session_ids = {session.session_id for session in response.sessions}
        counter = 0
        while True:
            digest = bytearray(
                hashlib.sha256(
                    (
                        "imageforge-orphan-stop\0"
                        f"{guard.server_instance_id}\0{guard.request_id}\0"
                        f"{response.server_instance_id}\0{counter}"
                    ).encode()
                ).digest()[:16]
            )
            digest[6] = (digest[6] & 0x0F) | 0x40
            digest[8] = (digest[8] & 0x3F) | 0x80
            synthetic_session_id = str(uuid.UUID(bytes=bytes(digest)))
            if synthetic_session_id not in occupied_session_ids:
                break
            counter += 1
        return response.model_copy(
            update={
                "stop_request": StopRequestView(
                    request_id=guard.request_id,
                    pod_id=guard.pod_id,
                    gpu_display_name=guard.gpu_display_name,
                    requester=CoordinationIdentity(
                        session_id=synthetic_session_id,
                        display_name=guard.requester,
                    ),
                    state="finalizing",
                    reason=None,
                    requested_at=guard.requested_at,
                    response_deadline=guard.response_deadline,
                    finalization_expires_at=guard.expires_at,
                    waiting_for=[],
                    approved_by=[],
                    denied_by=[],
                    finalization_id=None,
                )
            }
        )

    def _stop_guard_remaining_seconds(
        self, guard: SharedGpuStopGuard
    ) -> float | None:
        if self._stop_guard_is_local:
            return self.coordination.finalization_remaining_seconds(
                guard.request_id, guard.finalization_id
            )
        return self._shared_guard_remaining_seconds(guard)

    def _shared_guard_remaining_seconds(self, guard: SharedGpuStopGuard) -> float:
        expires_at = datetime.fromisoformat(guard.expires_at.replace("Z", "+00:00"))
        now = self.coordination.clock.utcnow().astimezone(UTC)
        return max(0.0, (expires_at.astimezone(UTC) - now).total_seconds())

    def _clear_stop_guard_locked(self, *, release_lease: bool) -> None:
        guard = self._stop_guard
        if guard is None:
            return
        self.store.clear_gpu_stop_guard(guard)
        self._stop_guard = None
        self._stop_guard_is_local = False
        task = self._stop_guard_expiry_task
        self._stop_guard_expiry_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        if release_lease and self._active_batch_id is None:
            self.store.release_active_lease()

    def _abandon_stop_guard_locked(self) -> None:
        """Drop only process-local state when its lease cannot be retained."""

        task = self._stop_guard_expiry_task
        self._stop_guard_expiry_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        self._stop_guard = None
        self._stop_guard_is_local = False

    def _schedule_stop_guard_expiry_locked(self, guard: SharedGpuStopGuard) -> None:
        prior = self._stop_guard_expiry_task
        if prior is not None and not prior.done():
            prior.cancel()
        self._stop_guard_expiry_task = asyncio.create_task(
            self._expire_stop_guard(guard),
            name=f"gpu-stop-guard-{guard.request_id}",
        )

    async def _expire_stop_guard(self, guard: SharedGpuStopGuard) -> None:
        delay = self._stop_guard_remaining_seconds(guard)
        while True:
            await asyncio.sleep(max(delay or 0.0, 0.01))
            async with self._lock:
                if self._stop_guard != guard:
                    return
                remaining = self._stop_guard_remaining_seconds(guard)
                if remaining is None or remaining <= 0:
                    self._clear_stop_guard_locked(release_lease=True)
                    return
                delay = remaining

    def _raise_shared_stop_guard_if_present(self) -> None:
        guard = self.store.read_gpu_stop_guard()
        if guard is None:
            return
        self._raise_gpu_stop_pending(guard)

    @staticmethod
    def _raise_gpu_stop_pending(guard: SharedGpuStopGuard) -> None:
        raise WorkerError(
            status_code=423,
            code="gpu_stop_pending",
            message="GPU Stop is finalizing; new generation is temporarily blocked.",
            details={
                "request_id": guard.request_id,
                "requester": guard.requester,
                "expires_at": guard.expires_at,
            },
        )

    @staticmethod
    def _raise_stop_blocked_by_active_batch(active: BatchManifest) -> None:
        raise WorkerError(
            status_code=423,
            code="stop_blocked_by_active_batch",
            message=f"{active.owner.display_name} has an active generation batch.",
            details={
                "owner": active.owner.display_name,
                "completed": active.progress.completed,
                "total": active.progress.total,
            },
        )

    def _discover_active_locked(self, *, recover: bool) -> BatchManifest | None:
        active_manifests: list[BatchManifest] = []
        for batch_id in self.store.list_batch_ids():
            manifest = self.store.load(batch_id)
            changed = self._recover_manifest(manifest) if recover else False
            if manifest.state in LOCK_HOLDING_STATES:
                active_manifests.append(manifest)
            if changed:
                self.store.save(manifest)
        if len(active_manifests) > 1:
            raise RuntimeError("persistent volume contains multiple active batch leases")
        return active_manifests[0] if active_manifests else None

    def _refresh_active_observation_locked(self) -> BatchManifest | None:
        if self._active_batch_id is not None:
            try:
                active = self.store.load(self._active_batch_id)
            except FileNotFoundError:
                active = None
            if active is not None and active.state in LOCK_HOLDING_STATES:
                if self.store.active_lease_held:
                    return active
                # A standby worker may have observed this manifest while the
                # original Pod held the lease. Once that lease disappears,
                # status/refresh must be able to adopt it and perform the
                # same interruption recovery as a freshly booted worker.
                if self.store.try_acquire_active_lease():
                    try:
                        self._adopt_or_clear_shared_stop_guard_locked()
                        recovered = self._discover_active_locked(recover=True)
                    except BaseException:
                        self._abandon_stop_guard_locked()
                        self.store.release_active_lease()
                        raise
                    self._active_batch_id = recovered.batch_id if recovered is not None else None
                    if recovered is None and self._stop_guard is None:
                        self.store.release_active_lease()
                    return recovered
                return active
        active = self._discover_active_locked(recover=False)
        self._active_batch_id = active.batch_id if active is not None else None
        if (
            active is None
            and not self.store.active_lease_held
            and self.store.read_gpu_stop_guard() is not None
            and self.store.try_acquire_active_lease()
        ):
            try:
                self._adopt_or_clear_shared_stop_guard_locked()
            except BaseException:
                self._abandon_stop_guard_locked()
                self.store.release_active_lease()
                raise
            if self._stop_guard is None:
                self.store.release_active_lease()
            return None
        if (
            active is not None
            and not self.store.active_lease_held
            and self.store.try_acquire_active_lease()
        ):
            try:
                self._adopt_or_clear_shared_stop_guard_locked()
                recovered = self._discover_active_locked(recover=True)
            except BaseException:
                self._abandon_stop_guard_locked()
                self.store.release_active_lease()
                raise
            self._active_batch_id = recovered.batch_id if recovered is not None else None
            if recovered is None and self._stop_guard is None:
                self.store.release_active_lease()
            return recovered
        return active

    async def _acquire_for_new_batch_locked(self) -> None:
        if self.store.active_lease_held:
            return
        deadline = asyncio.get_running_loop().time() + 1.0
        while True:
            if self.store.try_acquire_active_lease():
                try:
                    adopted = self._adopt_or_clear_shared_stop_guard_locked()
                    if adopted is not None:
                        self._raise_gpu_stop_pending(adopted)
                    active = self._discover_active_locked(recover=True)
                except WorkerError:
                    # This process adopted the crash-safe guard and must retain
                    # the shared lease until cancellation or expiry.
                    raise
                except BaseException:
                    self._abandon_stop_guard_locked()
                    self.store.release_active_lease()
                    raise
                self._active_batch_id = active.batch_id if active is not None else None
                return
            self._raise_shared_stop_guard_if_present()
            active = self._discover_active_locked(recover=False)
            self._active_batch_id = active.batch_id if active is not None else None
            if active is not None:
                self._raise_busy(active)
            if asyncio.get_running_loop().time() >= deadline:
                raise WorkerError(
                    status_code=423,
                    code="worker_volume_locked",
                    message="Another worker process is updating the shared volume.",
                )
            await asyncio.sleep(0.01)

    async def _require_mutation_lease_locked(self) -> None:
        if self.store.active_lease_held:
            return
        if self.store.try_acquire_active_lease():
            try:
                adopted = self._adopt_or_clear_shared_stop_guard_locked()
                if adopted is not None:
                    self._raise_gpu_stop_pending(adopted)
                active = self._discover_active_locked(recover=True)
            except WorkerError:
                raise
            except BaseException:
                self._abandon_stop_guard_locked()
                self.store.release_active_lease()
                raise
            self._active_batch_id = active.batch_id if active is not None else None
            return
        self._raise_shared_stop_guard_if_present()
        active = self._discover_active_locked(recover=False)
        self._active_batch_id = active.batch_id if active is not None else None
        details = None
        if active is not None:
            details = {
                "owner": active.owner.display_name,
                "completed": active.progress.completed,
                "total": active.progress.total,
            }
        raise WorkerError(
            status_code=423,
            code="worker_standby",
            message="Another worker process owns the active shared-volume lease.",
            details=details,
        )

    @staticmethod
    def _raise_busy(active: BatchManifest) -> None:
        completed = active.progress.completed
        total = active.progress.total
        raise WorkerError(
            status_code=423,
            code="batch_busy",
            message=f"{active.owner.display_name} is generating {completed} of {total} images.",
            details={
                "owner": active.owner.display_name,
                "completed": completed,
                "total": total,
            },
        )

    def _release_batch_lease_locked(self, batch_id: str) -> None:
        if self._active_batch_id == batch_id:
            self._active_batch_id = None
            if self._stop_guard is None:
                self.store.release_active_lease()

    def _release_if_no_active_locked(self) -> None:
        if self._stop_guard is not None:
            return
        active = self._refresh_active_observation_locked()
        if active is None:
            self.store.release_active_lease()

    @staticmethod
    def _finalize_cancel(manifest: BatchManifest) -> None:
        for image in manifest.images:
            if image.status in NONTERMINAL_IMAGE_STATES:
                image.status = ImageState.CANCELLED
                image.finished_at = utc_now()
                image.error = None
        manifest.state = BatchState.CANCELLED
        manifest.cancel_requested = False
        manifest.pause_requested = False
        manifest.completed_at = utc_now()

    def _load_owned(self, principal: Principal, batch_id: str) -> BatchManifest:
        try:
            manifest = self.store.load(batch_id)
        except (FileNotFoundError, ValueError):
            raise self._not_found() from None
        if manifest.owner.user_id != principal.user_id:
            # Do not disclose another owner's prompt text or whether a guessed ID exists.
            raise self._not_found()
        return manifest

    @staticmethod
    def _find_image(manifest: BatchManifest, index: int) -> ImageRecord:
        if index < 1 or index > len(manifest.images):
            raise WorkerError(
                status_code=404,
                code="image_not_found",
                message="The requested image does not exist.",
            )
        return manifest.images[index - 1]

    @staticmethod
    def _prepare_references(references: list[ReferenceInput]) -> list[_PreparedReference]:
        prepared: list[_PreparedReference] = []
        extensions = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
        for index, reference in enumerate(references, start=1):
            payload = bytes.fromhex(reference.data_hex)
            try:
                GenerationController._validate_reference_payload(payload, reference.mime_type)
            except (OSError, ValueError, Image.DecompressionBombError):
                raise WorkerError(
                    status_code=422,
                    code="reference_invalid",
                    message="A reference image is malformed or does not match its declared type.",
                ) from None
            prepared.append(
                _PreparedReference(
                    metadata=StoredReference(
                        name=reference.name,
                        mime_type=reference.mime_type,
                        size_bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest(),
                        filename=f"references/{index:06d}.{extensions[reference.mime_type]}",
                    ),
                    payload=payload,
                )
            )
        return prepared

    def _load_reference_images(self, manifest: BatchManifest) -> tuple[Image.Image, ...]:
        images: list[Image.Image] = []
        try:
            for reference in manifest.references:
                payload = self.store.read_reference(manifest.batch_id, reference.filename)
                checksum_matches = hashlib.sha256(payload).hexdigest() == reference.sha256
                if len(payload) != reference.size_bytes or not checksum_matches:
                    raise ValueError("reference checksum mismatch")
                images.append(self._decode_reference_payload(payload, reference.mime_type))
        except Exception:
            for image in images:
                image.close()
            raise
        return tuple(images)

    @staticmethod
    def _validate_reference_payload(payload: bytes, mime_type: str) -> None:
        expected_format = {
            "image/jpeg": "JPEG",
            "image/png": "PNG",
            "image/webp": "WEBP",
        }[mime_type]
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != expected_format:
                raise ValueError("reference format does not match MIME type")
            if image.width * image.height > MAX_REFERENCE_PIXELS:
                raise ValueError("reference dimensions exceed the supported limit")
            image.verify()

    @classmethod
    def _decode_reference_payload(cls, payload: bytes, mime_type: str) -> Image.Image:
        cls._validate_reference_payload(payload, mime_type)
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")

    @staticmethod
    def _not_found() -> WorkerError:
        return WorkerError(
            status_code=404,
            code="batch_not_found",
            message="The requested batch does not exist.",
        )

    @staticmethod
    def _invalid_state(manifest: BatchManifest, action: str) -> None:
        raise WorkerError(
            status_code=409,
            code="invalid_batch_state",
            message=f"Cannot {action} a batch in the {manifest.state.value} state.",
            details={"state": manifest.state.value},
        )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise WorkerError(
                status_code=503,
                code="worker_starting",
                message="The worker storage is still being prepared.",
            )

    @staticmethod
    def _validate_result(result: InferenceResult, settings: GenerationSettings) -> None:
        checks = (
            (result.jpeg, "JPEG", (settings.width, settings.height)),
            (
                result.preview,
                "WEBP",
                (settings.preview_width, settings.preview_height),
            ),
        )
        for payload, expected_format, expected_size in checks:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != expected_format or image.size != expected_size:
                    raise InferenceFailure("invalid_inference_artifact")
                image.verify()

    @staticmethod
    def _file_matches(path: Path, size: int, checksum: str) -> bool:
        try:
            if path.stat().st_size != size:
                return False
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return secrets.compare_digest(digest.hexdigest(), checksum)
        except FileNotFoundError:
            return False
