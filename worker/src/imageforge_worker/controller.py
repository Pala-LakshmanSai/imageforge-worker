from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .auth import Principal
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
    SafeImageError,
    StatusPermissions,
    StatusResponse,
    StoredReceipt,
    utc_now,
)
from .errors import InferenceFailure, WorkerError
from .inference import GenerationJob, InferenceAdapter, InferenceResult
from .persistence import ManifestStore

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


class GenerationController:
    """Owns the process-local controller and shared-volume generation lease."""

    def __init__(
        self,
        store: ManifestStore,
        inference: InferenceAdapter,
        *,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> None:
        self.store = store
        self.inference = inference
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._lock = asyncio.Lock()
        self._active_batch_id: str | None = None
        self._runner_task: asyncio.Task[None] | None = None
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
                    active = self._discover_active_locked(recover=True)
                    self._active_batch_id = active.batch_id if active is not None else None
                    if active is None:
                        self.store.release_active_lease()
                else:
                    active = self._discover_active_locked(recover=False)
                    self._active_batch_id = active.batch_id if active is not None else None
            except BaseException:
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
        task = self._runner_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
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

    async def release_lease_after_boot_failure(self) -> None:
        """Allow a healthy replacement Pod to adopt an interrupted batch."""

        async with self._lock:
            self.store.release_active_lease()

    async def status(self, principal: Principal, *, ready: bool) -> StatusResponse:
        self._ensure_initialized()
        async with self._lock:
            active = self._refresh_active_observation_locked()
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
                    can_create=ready and active is None,
                    can_manage_active=is_owner and self.store.active_lease_held,
                    is_owner=is_owner,
                ),
            )

    async def create_batch(
        self, principal: Principal, request: CreateBatchRequest
    ) -> BatchManifest:
        self._ensure_initialized()
        async with self._lock:
            await self._acquire_for_new_batch_locked()
            active = self._refresh_active_observation_locked()
            if active is not None:
                self._raise_busy(active)
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
                images=images,
                progress=BatchProgress(total=len(images)),
            )
            try:
                self.store.create(manifest)
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
            return _ClaimedAttempt(
                job=GenerationJob(
                    index=image.index,
                    prompt=image.prompt,
                    seed=image.seed,
                    settings=manifest.settings,
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
                        recovered = self._discover_active_locked(recover=True)
                    except BaseException:
                        self.store.release_active_lease()
                        raise
                    self._active_batch_id = recovered.batch_id if recovered is not None else None
                    if recovered is None:
                        self.store.release_active_lease()
                    return recovered
                return active
        active = self._discover_active_locked(recover=False)
        self._active_batch_id = active.batch_id if active is not None else None
        if (
            active is not None
            and not self.store.active_lease_held
            and self.store.try_acquire_active_lease()
        ):
            try:
                recovered = self._discover_active_locked(recover=True)
            except BaseException:
                self.store.release_active_lease()
                raise
            self._active_batch_id = recovered.batch_id if recovered is not None else None
            if recovered is None:
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
                    active = self._discover_active_locked(recover=True)
                except BaseException:
                    self.store.release_active_lease()
                    raise
                self._active_batch_id = active.batch_id if active is not None else None
                return
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
                active = self._discover_active_locked(recover=True)
            except BaseException:
                self.store.release_active_lease()
                raise
            self._active_batch_id = active.batch_id if active is not None else None
            return
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
            self.store.release_active_lease()

    def _release_if_no_active_locked(self) -> None:
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
