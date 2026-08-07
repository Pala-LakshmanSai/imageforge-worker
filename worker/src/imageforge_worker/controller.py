from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import secrets
import uuid
from collections.abc import Callable, Mapping
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
    AdmissionMode,
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
from .gpu_switch import (
    GpuControlGuardConflictError,
    GpuSwitchCoordinator,
    GpuSwitchStore,
    GpuSwitchStoreCorruptError,
    RuntimeDeviceInspector,
    gpu_switch_error,
)
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
from .inference import GenerationJob, InferenceAdapter, InferenceResult
from .persistence import (
    ManifestStore,
    SharedGpuStopGuard,
    SubmissionMatch,
    SubmissionRecord,
    SubmissionStoreCorruptError,
)

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
        crash_hook: Callable[[str], None] | None = None,
        runtime_metadata: Mapping[str, str] | None = None,
        data_root: Path | None = None,
        runtime_device_inspector: RuntimeDeviceInspector | None = None,
    ) -> None:
        self.store = store
        self.inference = inference
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._lock = asyncio.Lock()
        # batch_id -> manifest fingerprint last observed in a terminal state.
        # Purely an I/O memo for observation-only discovery; a changed
        # fingerprint always falls back to reading the manifest.
        self._inactive_batch_fingerprints: dict[str, tuple[int, int, int]] = {}
        self.coordination = StudioCoordinator(
            clock=coordination_clock,
            finalization_ttl_seconds=coordination_finalization_ttl_seconds,
        )
        switch_root = data_root or getattr(store, "root", Path("."))
        self.gpu_switch = GpuSwitchCoordinator(
            GpuSwitchStore(
                switch_root,
                store,
                fsync_writes=bool(getattr(store, "fsync_writes", True)),
                crash_hook=crash_hook,
            ),
            self.coordination,
            runtime_metadata or {},
            switch_root,
            runtime_inspector=runtime_device_inspector,
        )
        self._active_batch_id: str | None = None
        self._runner_task: asyncio.Task[None] | None = None
        self._stop_guard: SharedGpuStopGuard | None = None
        self._stop_guard_is_local = False
        self._stop_guard_expiry_task: asyncio.Task[None] | None = None
        self._initialized = False
        self._closing = False
        self._gpu_switch_boot_error: str | None = None
        self._switch_inflight_index: int | None = None
        # Deterministic crash seams for the v2 submission commit tests. This
        # remains unset in production and is intentionally invoked only after
        # the named state is durable enough to recover from a process death.
        self._crash_hook = crash_hook

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
                    acquired_gpu_lock = self._acquire_gpu_control_lock_locked()
                    try:
                        self._adopt_or_clear_shared_stop_guard_locked()
                        active = self._discover_active_locked(recover=True)
                        try:
                            self.gpu_switch.initialize(active, self._stop_guard)
                        except GpuControlGuardConflictError:
                            self._gpu_switch_boot_error = "gpu_control_guard_conflict"
                        except GpuSwitchStoreCorruptError:
                            self._gpu_switch_boot_error = "gpu_switch_store_corrupt"
                        except WorkerError as exc:
                            if exc.code not in {
                                "gpu_control_guard_conflict",
                                "gpu_switch_store_corrupt",
                            }:
                                raise
                            self._gpu_switch_boot_error = exc.code
                    finally:
                        if acquired_gpu_lock:
                            self.store.release_gpu_control_lock()
                    self._active_batch_id = active.batch_id if active is not None else None
                    if (
                        active is None
                        and self._stop_guard is None
                        and self._gpu_switch_boot_error is None
                        and self.gpu_switch.store.read_marker() is None
                    ):
                        self.store.release_active_lease()
                else:
                    try:
                        self.gpu_switch.store.initialize()
                    except GpuControlGuardConflictError:
                        self._gpu_switch_boot_error = "gpu_control_guard_conflict"
                    except GpuSwitchStoreCorruptError:
                        self._gpu_switch_boot_error = "gpu_switch_store_corrupt"
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
                self.store.release_gpu_control_lock()
                self.store.release_submission_lease()
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
            self.store.release_gpu_control_lock()
            self.store.release_active_lease()
            self.store.release_submission_lease()
            self._stop_guard = None
            self._stop_guard_is_local = False

    async def status(self, principal: Principal, *, ready: bool) -> StatusResponse:
        self._ensure_initialized()
        async with self._lock:
            self._reconcile_stop_guard_locked()
            active = self._refresh_active_observation_locked()
            shared_stop_guard = self._stop_guard or self.store.read_gpu_stop_guard()
            submission_store_corrupt = self.store.submission_store_corrupt()
            is_owner = active is not None and active.owner.user_id == principal.user_id
            switch_block = self._gpu_switch_boot_error or self.gpu_switch.permission_block(
                principal, active, shared_stop_guard
            )
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
                    can_create=(
                        ready
                        and active is None
                        and shared_stop_guard is None
                        and not submission_store_corrupt
                        and switch_block
                        not in {
                            "gpu_switch_pending",
                            "gpu_control_guard_conflict",
                            "gpu_switch_store_corrupt",
                        }
                    ),
                    can_manage_active=is_owner and self.store.active_lease_held,
                    is_owner=is_owner,
                    create_block_reason=(
                        "submission_store_corrupt"
                        if submission_store_corrupt
                        else "gpu_stop_pending"
                        if shared_stop_guard is not None
                        else None
                    ),
                    can_switch=switch_block is None,
                    switch_block_code=switch_block,
                ),
            )

    async def preflight_new_submission(self) -> None:
        """Fail closed on corrupt v2 history before a readiness response.

        This intentionally does not require controller initialization: during
        boot, the durable shared volume is already the authoritative source for
        a corrupt envelope, while model readiness is still only provisional.
        A valid create therefore cannot mask the repair-required condition with
        ``worker_not_ready``.
        """

        async with self._lock:
            self._raise_submission_store_corrupt_if_present()

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
            shared_stop_guard = self._stop_guard or self.store.read_gpu_stop_guard()
            if shared_stop_guard is None:
                already_held, acquired_gpu = await self._enter_gpu_control_locked()
                try:
                    switch_view = self.gpu_switch.refresh(active)
                finally:
                    self._exit_gpu_control_locked(already_held, acquired_gpu)
            else:
                # Stop and Switch are mutually exclusive durable GPU-control
                # guards. A standby must still serve Studio presence while a
                # peer owns the Stop lease; there is no Switch state to mutate.
                switch_view = None
            self._reconcile_stop_guard_locked()
            return self._project_shared_stop_guard_locked(response).model_copy(
                update={
                    "gpu_switch_request": switch_view,
                    "gpu_switch_can_respond": self.gpu_switch.can_respond(
                        principal, session_id
                    )
                    if switch_view is not None
                    else False,
                }
            )

    async def studio_state(self, principal: Principal, session_id: str) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._reconcile_stop_guard_locked()
            active = self._refresh_active_observation_locked()
            self.coordination.state(principal, session_id, active)
            shared_stop_guard = self._stop_guard or self.store.read_gpu_stop_guard()
            if shared_stop_guard is None:
                already_held, acquired_gpu = await self._enter_gpu_control_locked()
                try:
                    switch_view = self.gpu_switch.refresh(active)
                    response = self.coordination.state(principal, session_id, active)
                finally:
                    self._exit_gpu_control_locked(already_held, acquired_gpu)
            else:
                switch_view = None
                response = self.coordination.state(principal, session_id, active)
            self._reconcile_stop_guard_locked()
            return self._project_shared_stop_guard_locked(response).model_copy(
                update={
                    "gpu_switch_request": switch_view,
                    "gpu_switch_can_respond": self.gpu_switch.can_respond(
                        principal, session_id
                    )
                    if switch_view is not None
                    else False,
                }
            )

    async def request_gpu_switch(
        self, principal: Principal, request: CreateGpuSwitchRequestV1
    ) -> NativeWorkerGpuSwitchCreateResponseV1:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            active = self._refresh_active_observation_locked()
            already_held, acquired_gpu = await self._enter_gpu_control_locked()
            try:
                return self.gpu_switch.create(
                    principal,
                    request,
                    active,
                    self._stop_guard or self.store.read_gpu_stop_guard(),
                )
            finally:
                self._exit_gpu_control_locked(already_held, acquired_gpu)

    async def get_gpu_switch(
        self, principal: Principal, switch_id: str, session_id: str
    ) -> GpuSwitchLookupResponseV1:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            return self.gpu_switch.public_lookup(principal, switch_id, session_id)

    async def get_gpu_switch_owner(
        self, principal: Principal, switch_id: str, session_id: str
    ) -> NativeWorkerGpuSwitchOwnerLookupV1:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            return self.gpu_switch.owner_lookup(principal, switch_id, session_id)

    async def settle_gpu_switch_create(
        self,
        principal: Principal,
        switch_id: str,
        request: SettleGpuSwitchCreateRequestV1,
    ) -> NativeWorkerGpuSwitchOwnerLookupV1:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            already_held, acquired_gpu = await self._enter_gpu_control_locked()
            try:
                return self.gpu_switch.settle_create(principal, switch_id, request)
            finally:
                self._exit_gpu_control_locked(already_held, acquired_gpu)

    async def respond_to_gpu_switch(
        self,
        principal: Principal,
        switch_id: str,
        request: GpuSwitchResponseRequestV1,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            active = self._refresh_active_observation_locked()
            already_held, acquired_gpu = await self._enter_gpu_control_locked()
            try:
                self.gpu_switch.respond(principal, switch_id, request, active)
                return self._studio_state_with_switch_locked(principal, request.session_id, active)
            finally:
                self._exit_gpu_control_locked(already_held, acquired_gpu)

    async def finalize_gpu_switch(
        self,
        principal: Principal,
        switch_id: str,
        request: FinalizeGpuSwitchRequestV1,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            active = self._refresh_active_observation_locked()
            already_held, acquired_gpu = await self._enter_gpu_control_locked()
            try:
                self.gpu_switch.finalize(
                    principal,
                    switch_id,
                    request,
                    active,
                    self._stop_guard or self.store.read_gpu_stop_guard(),
                )
                if active is not None:
                    current_image = next(
                        (
                            image.index
                            for image in active.images
                            if image.status in {ImageState.GENERATING, ImageState.RETRYING}
                        ),
                        None,
                    )
                    self._switch_inflight_index = current_image
                    if active.state == BatchState.RUNNING:
                        if current_image is None:
                            active.state = BatchState.PAUSED
                            active.pause_requested = False
                        else:
                            active.pause_requested = True
                        self.store.save(active)
                active = self._refresh_active_observation_locked()
                self.gpu_switch.mark_ready_to_delete(active)
                return self._studio_state_with_switch_locked(principal, request.session_id, active)
            finally:
                self._exit_gpu_control_locked(already_held, acquired_gpu)

    async def mark_gpu_switch_delete_intent(
        self,
        principal: Principal,
        switch_id: str,
        request: DeleteIntentGpuSwitchRequestV1,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            active = self._refresh_active_observation_locked()
            already_held, acquired_gpu = await self._enter_gpu_control_locked()
            try:
                self.gpu_switch.delete_intent(principal, switch_id, request)
                return self._studio_state_with_switch_locked(principal, request.session_id, active)
            finally:
                self._exit_gpu_control_locked(already_held, acquired_gpu)

    async def adopt_gpu_switch_replacement(
        self,
        principal: Principal,
        switch_id: str,
        request: AdoptGpuSwitchRequestV1,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            active = self._refresh_active_observation_locked()
            already_held, acquired_gpu = await self._enter_gpu_control_locked()
            try:
                self.gpu_switch.adopt(principal, switch_id, request)
                return self._studio_state_with_switch_locked(principal, request.session_id, active)
            finally:
                self._exit_gpu_control_locked(already_held, acquired_gpu)

    async def complete_gpu_switch(
        self,
        principal: Principal,
        switch_id: str,
        request: CompleteGpuSwitchRequestV1,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            active = self._refresh_active_observation_locked()
            already_held, acquired_gpu = await self._enter_gpu_control_locked()
            try:
                self.gpu_switch.complete(principal, switch_id, request)
                return self._studio_state_with_switch_locked(principal, request.session_id, active)
            finally:
                self._exit_gpu_control_locked(already_held, acquired_gpu)

    async def cancel_gpu_switch(
        self,
        principal: Principal,
        switch_id: str,
        request: CancelGpuSwitchRequestV1,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            active = self._refresh_active_observation_locked()
            already_held, acquired_gpu = await self._enter_gpu_control_locked()
            try:
                self.gpu_switch.cancel(principal, switch_id, request, active)
                return self._studio_state_with_switch_locked(principal, request.session_id, active)
            finally:
                self._exit_gpu_control_locked(already_held, acquired_gpu)

    async def gpu_switch_runtime_identity(
        self, principal: Principal, switch_id: str, session_id: str
    ) -> WorkerGpuSwitchRuntimeIdentityV1:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            return self.gpu_switch.runtime_identity(principal, switch_id, session_id)

    async def request_gpu_stop(
        self, principal: Principal, request: CreateStopRequest
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
            self._reconcile_stop_guard_locked()
            if self._stop_guard is not None:
                self._raise_gpu_stop_pending(self._stop_guard)
            self._raise_shared_stop_guard_if_present()
            active = self._refresh_active_observation_locked()
            already_held, acquired_gpu = await self._enter_gpu_control_locked()
            try:
                self.gpu_switch.block_stop()
                response = self.coordination.create_stop_request(principal, request, active)
                self._reconcile_stop_guard_locked()
                return response
            finally:
                self._exit_gpu_control_locked(already_held, acquired_gpu)

    async def respond_to_gpu_stop(
        self,
        principal: Principal,
        request_id: str,
        request: StopResponseRequest,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
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
            self._raise_gpu_switch_boot_error()
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

            acquired_gpu = self._acquire_gpu_control_lock_locked()
            try:
                self.gpu_switch.block_stop()

                stop = response.stop_request
                if stop is None or stop.finalization_expires_at is None:
                    self.coordination.rollback_finalization(request_id, request.finalization_id)
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
                        pass
                    raise
                self._stop_guard = guard
                self._stop_guard_is_local = True
                self._schedule_stop_guard_expiry_locked(guard)
                return response
            except BaseException:
                self.coordination.rollback_finalization(request_id, request.finalization_id)
                raise
            finally:
                if acquired_gpu:
                    self.store.release_gpu_control_lock()
                if self._stop_guard is None and self._active_batch_id is None:
                    self.store.release_active_lease()

    async def cancel_gpu_stop(
        self,
        principal: Principal,
        request_id: str,
        request: CancelStopRequest,
    ) -> StudioStateResponse:
        self._ensure_initialized()
        async with self._lock:
            self._raise_gpu_switch_boot_error()
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
        settings = GenerationSettings.for_aspect_ratio(request.aspect_ratio)
        fingerprint = self._submission_fingerprint(
            principal,
            request,
            prepared_references,
            settings,
        )
        async with self._lock:
            await self._acquire_submission_lease_locked()
            try:
                self._raise_submission_store_corrupt_if_present()

                # Replay is deliberately before the active lease/busy/Stop
                # checks. This separate shared-volume lease serializes lookup
                # with envelope publication even while a different batch owns
                # the long-running generation lease.
                replay = self._find_submission_or_raise(request.client_submission_id)
                if replay is not None:
                    return self._resolve_submission_replay(principal, fingerprint, replay)

                await self._acquire_for_new_batch_locked()
                # Re-read after active-lease acquisition before considering
                # busy or Stop admission. A process that had the history lease
                # immediately before us may have committed this exact key.
                self._raise_submission_store_corrupt_if_present()
                replay = self._find_submission_or_raise(request.client_submission_id)
                if replay is not None:
                    result = self._resolve_submission_replay(principal, fingerprint, replay)
                    self._release_if_no_active_locked()
                    return result
                active = self._refresh_active_observation_locked()
                if active is not None:
                    self._release_if_no_active_locked()
                    self._raise_busy(active)
                try:
                    self._admit_generation_locked(request.admission_mode)
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
                    owner=BatchOwner(
                        user_id=principal.user_id,
                        display_name=principal.display_name,
                    ),
                    state=BatchState.RUNNING,
                    created_at=now,
                    updated_at=now,
                    client_submission_id=request.client_submission_id,
                    settings=settings,
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
                        submission=SubmissionRecord(
                            client_submission_id=request.client_submission_id,
                            owner_user_id=principal.user_id,
                            request_fingerprint=fingerprint,
                        ),
                    )
                except BaseException:
                    self.store.release_active_lease()
                    raise
                self._active_batch_id = batch_id
                self._crash("after_active_memory_assignment")
                self._launch_runner_locked(batch_id)
                self._crash("after_runner_launch")
                # This is intentionally the final seam: an exception here
                # models response loss after admission, for which the next
                # request must replay rather than create a second batch.
                self._crash("before_create_response")
                return manifest
            finally:
                self.store.release_submission_lease()

    async def get_batch(self, principal: Principal, batch_id: str) -> BatchManifest:
        self._ensure_initialized()
        async with self._lock:
            return self._load_owned(principal, batch_id)

    async def get_submission(
        self, principal: Principal, client_submission_id: str
    ) -> BatchManifest:
        """Look up exactly one caller-owned durable submission association."""

        self._ensure_initialized()
        async with self._lock:
            await self._acquire_submission_lease_locked()
            try:
                self._raise_submission_store_corrupt_if_present()
                match = self._find_submission_or_raise(client_submission_id)
                if match is None or match.record.owner_user_id != principal.user_id:
                    raise self._submission_not_found()
                return match.manifest
            finally:
                self.store.release_submission_lease()

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
            acquired_gpu = self._acquire_gpu_control_lock_locked()
            try:
                self.gpu_switch.cancel_for_generation("batch_changed", queue_mode=False)
            finally:
                if acquired_gpu:
                    self.store.release_gpu_control_lock()
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
            allow_switch_retry = False
            acquired_gpu = self._acquire_gpu_control_lock_locked()
            try:
                try:
                    switch_marker = self.gpu_switch.store.read_marker()
                except GpuSwitchStoreCorruptError:
                    self._gpu_switch_boot_error = "gpu_switch_store_corrupt"
                    return None
                if switch_marker is not None:
                    if switch_marker.phase != "pausing":
                        return None
                    retrying_current = next(
                        (
                            image
                            for image in manifest.images
                            if image.status == ImageState.RETRYING
                            and image.index == self._switch_inflight_index
                        ),
                        None,
                    )
                    if retrying_current is None:
                        manifest.state = BatchState.PAUSED
                        manifest.pause_requested = False
                        self.store.save(manifest)
                        self.gpu_switch.mark_ready_to_delete(manifest)
                        return None
                    allow_switch_retry = True
            finally:
                if acquired_gpu:
                    self.store.release_gpu_control_lock()
            if manifest.cancel_requested:
                self._finalize_cancel(manifest)
                self.store.save(manifest)
                self._release_batch_lease_locked(batch_id)
                return None
            if manifest.pause_requested and not allow_switch_retry:
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
            if self._switch_inflight_index == image.index:
                self._switch_inflight_index = None

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
            if terminal and self._switch_inflight_index == image.index:
                self._switch_inflight_index = None
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

    def _admit_generation_locked(
        self, admission_mode: AdmissionMode = AdmissionMode.FOREGROUND
    ) -> None:
        self._raise_gpu_switch_boot_error()
        acquired_gpu = self._acquire_gpu_control_lock_locked()
        try:
            self.gpu_switch.cancel_for_generation(
                "generation_started", queue_mode=admission_mode == AdmissionMode.QUEUE
            )
        finally:
            if acquired_gpu:
                self.store.release_gpu_control_lock()
        self._reconcile_stop_guard_locked(release_lease=False)
        if self._stop_guard is not None:
            self._raise_gpu_stop_pending(self._stop_guard)
        if admission_mode == AdmissionMode.QUEUE:
            self.coordination.admit_queue_generation()
            return
        self.coordination.admit_generation()

    async def _enter_gpu_control_locked(self) -> tuple[bool, bool]:
        self._raise_gpu_switch_boot_error()
        already_held = self.store.active_lease_held
        await self._require_mutation_lease_locked()
        acquired_gpu = self._acquire_gpu_control_lock_locked()
        try:
            # Every mutation rereads the complete marker/envelope/tombstone
            # namespace after taking the cross-process GPU-control lock. This
            # closes the live-process seam where a post-delete cancellation
            # tombstone appears after boot but before adoption/completion.
            self.gpu_switch.store.initialize()
            self.gpu_switch.store.reconcile_terminal_commits()
        except GpuControlGuardConflictError:
            self._gpu_switch_boot_error = "gpu_control_guard_conflict"
            if acquired_gpu:
                self.store.release_gpu_control_lock()
            raise gpu_switch_error("gpu_control_guard_conflict") from None
        except GpuSwitchStoreCorruptError:
            self._gpu_switch_boot_error = "gpu_switch_store_corrupt"
            if acquired_gpu:
                self.store.release_gpu_control_lock()
            raise gpu_switch_error("gpu_switch_store_corrupt") from None
        return already_held, acquired_gpu

    def _exit_gpu_control_locked(self, already_held: bool, acquired_gpu: bool) -> None:
        if acquired_gpu:
            self.store.release_gpu_control_lock()
        if already_held or self._active_batch_id is not None or self._stop_guard is not None:
            return
        if self._gpu_switch_boot_error is not None:
            return
        try:
            marker = self.gpu_switch.store.read_marker()
        except GpuSwitchStoreCorruptError:
            self._gpu_switch_boot_error = "gpu_switch_store_corrupt"
            return
        if marker is None:
            self.store.release_active_lease()

    def _acquire_gpu_control_lock_locked(self) -> bool:
        if self.store.gpu_control_lock_held:
            return False
        if not self.store.try_acquire_gpu_control_lock():
            raise WorkerError(
                status_code=423,
                code="worker_volume_locked",
                message="Another worker process is updating GPU control state.",
            )
        return True

    def _raise_gpu_switch_boot_error(self) -> None:
        if self._gpu_switch_boot_error is not None:
            raise gpu_switch_error(self._gpu_switch_boot_error)

    def _studio_state_with_switch_locked(
        self, principal: Principal, session_id: str, active: BatchManifest | None
    ) -> StudioStateResponse:
        response = self.coordination.state(principal, session_id, active)
        switch_view = self.gpu_switch.refresh(active)
        return self._project_shared_stop_guard_locked(response).model_copy(
            update={
                "gpu_switch_request": switch_view,
                "gpu_switch_can_respond": self.gpu_switch.can_respond(
                    principal, session_id
                )
                if switch_view is not None
                else False,
            }
        )

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
        acquired_gpu = self._acquire_gpu_control_lock_locked()
        try:
            guard = self.store.read_gpu_stop_guard()
            try:
                marker = self.gpu_switch.store.read_marker()
            except GpuSwitchStoreCorruptError:
                marker = None
                self._gpu_switch_boot_error = "gpu_switch_store_corrupt"
            if guard is not None and marker is not None:
                self._gpu_switch_boot_error = "gpu_control_guard_conflict"
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
        except GpuSwitchStoreCorruptError:
            self._gpu_switch_boot_error = "gpu_switch_store_corrupt"
            raise gpu_switch_error("gpu_switch_store_corrupt") from None
        finally:
            if acquired_gpu:
                self.store.release_gpu_control_lock()

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

    def _stop_guard_remaining_seconds(self, guard: SharedGpuStopGuard) -> float | None:
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
        acquired_gpu = self._acquire_gpu_control_lock_locked()
        try:
            self.store.clear_gpu_stop_guard(guard)
        finally:
            if acquired_gpu:
                self.store.release_gpu_control_lock()
        self._stop_guard = None
        self._stop_guard_is_local = False
        task = self._stop_guard_expiry_task
        self._stop_guard_expiry_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        if (
            release_lease
            and self._active_batch_id is None
            and not self._switch_requires_lease_locked()
        ):
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
            if not recover and self._known_inactive_locked(batch_id):
                # Observation-only discovery runs on every `/v1/status` poll.
                # A manifest whose bytes have not changed since we last read it
                # cannot have entered a lock-holding state, so re-reading it
                # from the network volume can only produce the same answer.
                continue
            try:
                manifest = self.store.load(batch_id)
            except SubmissionStoreCorruptError:
                # A corrupt v2 envelope blocks every new admission through
                # status/create, but it must not make valid owner batch reads
                # or unrelated status observation crash with an internal error.
                continue
            changed = self._recover_manifest(manifest) if recover else False
            if manifest.state in LOCK_HOLDING_STATES:
                active_manifests.append(manifest)
            elif not changed:
                self._remember_inactive_locked(batch_id)
            if changed:
                self.store.save(manifest)
        if len(active_manifests) > 1:
            raise RuntimeError("persistent volume contains multiple active batch leases")
        return active_manifests[0] if active_manifests else None

    def _known_inactive_locked(self, batch_id: str) -> bool:
        remembered = self._inactive_batch_fingerprints.get(batch_id)
        if remembered is None:
            return False
        current = self.store.manifest_fingerprint(batch_id)
        if current == remembered:
            return True
        self._inactive_batch_fingerprints.pop(batch_id, None)
        return False

    def _remember_inactive_locked(self, batch_id: str) -> None:
        fingerprint = self.store.manifest_fingerprint(batch_id)
        if fingerprint is not None:
            self._inactive_batch_fingerprints[batch_id] = fingerprint

    def _refresh_active_observation_locked(self) -> BatchManifest | None:
        active = self._refresh_active_observation_unchecked_locked()
        if self._gpu_switch_boot_error is None:
            try:
                self.gpu_switch.validate_marker_batch_binding(active)
            except GpuSwitchStoreCorruptError:
                # A marker-bound manifest can disappear or become malformed
                # after boot. Convert that live observation into the same
                # fail-closed projection used during startup, while retaining
                # the guard and active-volume lease for explicit repair.
                self._gpu_switch_boot_error = "gpu_switch_store_corrupt"
        return active

    def _refresh_active_observation_unchecked_locked(self) -> BatchManifest | None:
        if self._active_batch_id is not None:
            try:
                active = self.store.load(self._active_batch_id)
            except (FileNotFoundError, SubmissionStoreCorruptError):
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
                        self._reconcile_gpu_switch_takeover_locked(recovered)
                    except BaseException:
                        self._abandon_stop_guard_locked()
                        self.store.release_active_lease()
                        raise
                    self._active_batch_id = recovered.batch_id if recovered is not None else None
                    if (
                        recovered is None
                        and self._stop_guard is None
                        and not self._switch_requires_lease_locked()
                    ):
                        self.store.release_active_lease()
                    return recovered
                return active
        active = self._discover_active_locked(recover=False)
        self._active_batch_id = active.batch_id if active is not None else None
        if (
            active is None
            and not self.store.active_lease_held
            and (
                self.store.read_gpu_stop_guard() is not None or self._switch_requires_lease_locked()
            )
            and self.store.try_acquire_active_lease()
        ):
            try:
                self._adopt_or_clear_shared_stop_guard_locked()
                self._reconcile_gpu_switch_takeover_locked(None)
            except BaseException:
                self._abandon_stop_guard_locked()
                self.store.release_active_lease()
                raise
            if self._stop_guard is None and not self._switch_requires_lease_locked():
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
                self._reconcile_gpu_switch_takeover_locked(recovered)
            except BaseException:
                self._abandon_stop_guard_locked()
                self.store.release_active_lease()
                raise
            self._active_batch_id = recovered.batch_id if recovered is not None else None
            if (
                recovered is None
                and self._stop_guard is None
                and not self._switch_requires_lease_locked()
            ):
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
                    if self.gpu_switch.requires_takeover_reconciliation(active):
                        self._reconcile_gpu_switch_takeover_locked(active)
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

    async def _acquire_submission_lease_locked(self) -> None:
        """Take the short cross-process key-history lease with a bounded wait."""

        if self.store.submission_lease_held:
            return
        deadline = asyncio.get_running_loop().time() + 1.0
        while not self.store.try_acquire_submission_lease():
            if asyncio.get_running_loop().time() >= deadline:
                raise WorkerError(
                    status_code=423,
                    code="worker_volume_locked",
                    message="Another worker process is updating submission history.",
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
                if self.gpu_switch.requires_takeover_reconciliation(active):
                    self._reconcile_gpu_switch_takeover_locked(active)
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

    def _reconcile_gpu_switch_takeover_locked(
        self,
        active: BatchManifest | None,
    ) -> None:
        """Run the complete Switch boot/adoption contract after lease takeover."""

        acquired_gpu = self._acquire_gpu_control_lock_locked()
        try:
            try:
                self.gpu_switch.initialize(active, self._stop_guard)
            except GpuControlGuardConflictError:
                self._gpu_switch_boot_error = "gpu_control_guard_conflict"
            except GpuSwitchStoreCorruptError:
                self._gpu_switch_boot_error = "gpu_switch_store_corrupt"
            except WorkerError as exc:
                if exc.code not in {
                    "gpu_control_guard_conflict",
                    "gpu_switch_store_corrupt",
                }:
                    raise
                self._gpu_switch_boot_error = exc.code
        finally:
            if acquired_gpu:
                self.store.release_gpu_control_lock()

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

    @staticmethod
    def _submission_fingerprint(
        principal: Principal,
        request: CreateBatchRequest,
        references: list[_PreparedReference],
        settings: GenerationSettings,
    ) -> str:
        """Compute the specified owner-bound immutable submission fingerprint.

        The client sends neither this digest nor the raw storage path. Reference
        hashes are computed only after strict image decoding has accepted the
        request bytes, so a claimed payload cannot choose its own fingerprint.
        """

        value = {
            "owner_user_id": principal.user_id,
            "admission_mode": request.admission_mode.value,
            "prompts": request.prompts,
            "base_seed": request.base_seed,
            "aspect_ratio": request.aspect_ratio,
            "references": [
                {
                    "name": reference.metadata.name,
                    "mime_type": reference.metadata.mime_type,
                    "size_bytes": reference.metadata.size_bytes,
                    "sha256": reference.metadata.sha256,
                }
                for reference in references
            ],
            "settings": settings.model_dump(mode="json"),
        }
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(b"imageforge-batch-submission-v1\n" + canonical).hexdigest()

    @staticmethod
    def _resolve_submission_replay(
        principal: Principal,
        fingerprint: str,
        match: SubmissionMatch,
    ) -> BatchManifest:
        # Deliberately use one opaque conflict response for a foreign key and
        # a same-owner changed request. Neither path reveals whether the key is
        # occupied by another person or exposes the fingerprint.
        if match.record.owner_user_id != principal.user_id or not secrets.compare_digest(
            match.record.request_fingerprint, fingerprint
        ):
            raise WorkerError(
                status_code=409,
                code="submission_conflict",
                message="This submission ID cannot be used for a different generation request.",
            )
        return match.manifest

    def _find_submission_or_raise(self, client_submission_id: str) -> SubmissionMatch | None:
        try:
            return self.store.find_submission(client_submission_id)
        except SubmissionStoreCorruptError:
            raise self._submission_store_corrupt() from None

    def _raise_submission_store_corrupt_if_present(self) -> None:
        try:
            corrupt = self.store.submission_store_corrupt()
        except SubmissionStoreCorruptError:
            corrupt = True
        if corrupt:
            raise self._submission_store_corrupt()

    @staticmethod
    def _submission_store_corrupt() -> WorkerError:
        return WorkerError(
            status_code=503,
            code="submission_store_corrupt",
            message=(
                "Worker submission history is unavailable. Repair the shared volume before "
                "starting generation."
            ),
        )

    @staticmethod
    def _submission_not_found() -> WorkerError:
        return WorkerError(
            status_code=404,
            code="submission_not_found",
            message="The requested submission does not exist.",
        )

    def _crash(self, point: str) -> None:
        if self._crash_hook is not None:
            self._crash_hook(point)

    def _release_batch_lease_locked(self, batch_id: str) -> None:
        if self._active_batch_id == batch_id:
            self._active_batch_id = None
            if self._stop_guard is None and not self._switch_requires_lease_locked():
                self.store.release_active_lease()

    def _release_if_no_active_locked(self) -> None:
        if self._stop_guard is not None or self._switch_requires_lease_locked():
            return
        active = self._refresh_active_observation_locked()
        if active is None:
            self.store.release_active_lease()

    def _switch_requires_lease_locked(self) -> bool:
        if self._gpu_switch_boot_error is not None:
            return True
        try:
            return self.gpu_switch.store.read_marker() is not None
        except GpuSwitchStoreCorruptError:
            self._gpu_switch_boot_error = "gpu_switch_store_corrupt"
            return True

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
        except (FileNotFoundError, ValueError, SubmissionStoreCorruptError):
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
