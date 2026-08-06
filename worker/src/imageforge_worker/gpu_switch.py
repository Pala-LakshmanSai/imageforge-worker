from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, get_args

from pydantic import ValidationError

from .auth import Principal
from .constants import MODEL_ID, MODEL_REVISION
from .coordination import StudioCoordinator
from .domain import BatchManifest, BatchState, ImageState, utc_now
from .errors import WorkerError
from .gpu_switch_models import (
    MAX_SAFE_REVISION,
    AdoptGpuSwitchRequestV1,
    CancelGpuSwitchRequestV1,
    CompleteGpuSwitchRequestV1,
    CreateGpuSwitchRequestV1,
    DeleteIntentGpuSwitchRequestV1,
    FinalizeGpuSwitchRequestV1,
    GpuRuntimeIdentityContractV1,
    GpuSwitchBatchOwnerV1,
    GpuSwitchBlockCode,
    GpuSwitchCodeContractV1,
    GpuSwitchLookupResponseV1,
    GpuSwitchParticipantV1,
    GpuSwitchRequestViewV1,
    GpuSwitchResponseRequestV1,
    NativeWorkerGpuSwitchCreateResponseV1,
    NativeWorkerGpuSwitchOwnerLookupV1,
    SettleGpuSwitchCreateRequestV1,
    SharedGpuSwitchMarkerV1,
    SharedGpuSwitchRequestEnvelopeV1,
    SharedGpuSwitchTombstoneV1,
    WorkerCudaDeviceIdentityV1,
    WorkerGpuSwitchRuntimeIdentityV1,
    require_gpu_identity,
    require_image_digest,
    require_pod_id,
)
from .persistence import ManifestStore, SharedGpuStopGuard

GPU_SWITCH_RESPONSE_TTL_SECONDS = 30
GPU_SWITCH_PAUSE_TTL_SECONDS = 15 * 60
MAX_GPU_SWITCH_MARKER_BYTES = 8 * 1024
MAX_GPU_SWITCH_ENVELOPE_BYTES = 32 * 1024
MAX_GPU_SWITCH_TOMBSTONE_BYTES = 4 * 1024
PACKAGED_RUNTIME_IDENTITIES_PATH = (
    Path(__file__).resolve().parent / "contracts" / "gpu-runtime-identities-v1.json"
)
PACKAGED_SWITCH_CODES_PATH = (
    Path(__file__).resolve().parent / "contracts" / "gpu-switch-codes-v1.json"
)

GPU_SWITCH_STORE_CORRUPT_MESSAGE = (
    "Worker GPU switch history is unavailable. Repair the shared volume before changing compute."
)
GPU_CONTROL_GUARD_CONFLICT_MESSAGE = (
    "Worker GPU control history conflicts. Repair the shared volume before changing compute."
)
GPU_SWITCH_RUNTIME_IDENTITY_UNAVAILABLE_MESSAGE = (
    "Worker runtime identity is unavailable. Repair the ImageForge template before "
    "changing compute."
)

_ERRORS: dict[str, tuple[int, str]] = {
    "gpu_switch_request_not_found": (404, "The GPU switch request does not exist."),
    "gpu_switch_request_in_progress": (409, "Another GPU switch request is already in progress."),
    "gpu_switch_identity_mismatch": (
        409,
        "The GPU switch ID cannot be used for different request details.",
    ),
    "gpu_switch_response_conflict": (
        409,
        "This user already sent a different GPU switch response.",
    ),
    "gpu_switch_response_not_allowed": (409, "This user cannot respond to the GPU switch request."),
    "gpu_switch_approval_pending": (409, "GPU switch approval is still pending."),
    "gpu_switch_not_approved": (409, "The GPU switch request is not approved for finalization."),
    "gpu_switch_finalization_mismatch": (
        409,
        "The GPU switch finalization identity does not match.",
    ),
    "gpu_switch_cancel_not_allowed": (
        409,
        "The GPU switch cannot be cancelled after delete intent.",
    ),
    "gpu_switch_adoption_mismatch": (
        409,
        "The replacement worker identity does not match the GPU switch.",
    ),
    "gpu_switch_batch_changed": (
        409,
        "The active batch no longer matches the GPU switch preflight.",
    ),
    "gpu_switch_completion_not_ready": (
        409,
        "The replacement worker is not ready to complete the GPU switch.",
    ),
    "gpu_switch_current_pod_unverified": (
        409,
        "The current worker Pod identity is not authoritative.",
    ),
    "gpu_switch_local_receipts_pending": (
        409,
        "Local image receipts must settle before changing compute.",
    ),
    "stop_request_in_progress": (409, "A coordinated GPU Stop request is already in progress."),
    "gpu_switch_requester_not_foreground": (
        423,
        "The GPU switch requester must remain foreground.",
    ),
    "switch_owner_unavailable": (
        423,
        "The active batch owner is not available to approve this GPU switch.",
    ),
    "gpu_switch_queue_dispatch_uncertain": (
        423,
        "The local queue submission must be reconciled before changing compute.",
    ),
    "gpu_stop_pending": (423, "GPU Stop is finalizing; a GPU switch is temporarily blocked."),
    "gpu_switch_pending": (423, "A finalized GPU switch blocks this worker operation."),
    "queue_switch_pending": (
        423,
        "A GPU switch is awaiting consent; the local queue remains parked.",
    ),
    "gpu_switch_store_corrupt": (503, GPU_SWITCH_STORE_CORRUPT_MESSAGE),
    "gpu_switch_runtime_identity_unavailable": (
        503,
        GPU_SWITCH_RUNTIME_IDENTITY_UNAVAILABLE_MESSAGE,
    ),
    "gpu_control_guard_conflict": (503, GPU_CONTROL_GUARD_CONFLICT_MESSAGE),
}


def gpu_switch_error(code: str) -> WorkerError:
    status, message = _ERRORS[code]
    return WorkerError(status_code=status, code=code, message=message)


class GpuSwitchStoreCorruptError(RuntimeError):
    pass


class GpuControlGuardConflictError(RuntimeError):
    pass


def _require_marker_view_binding(
    marker: SharedGpuSwitchMarkerV1,
    view: GpuSwitchRequestViewV1,
) -> None:
    """Validate one exact marker/view phase or its single write-order seam."""

    if (
        view.switch_id != marker.switch_id
        or view.old_pod_id != marker.old_pod_id
        or view.old_gpu_id != marker.old_gpu_id
        or view.initial_target_gpu_id != marker.initial_target_gpu_id
        or view.initial_replacement_attempt_id != marker.initial_replacement_attempt_id
        or view.requester.display_name != marker.requester_display_name
        or view.batch_id != marker.batch_id
    ):
        raise GpuSwitchStoreCorruptError("switch marker request binding mismatch")

    # Marker writes precede their matching request-view write. Startup may
    # therefore observe exactly the immediately preceding phase, but never an
    # arbitrary older phase. _view_from_marker deterministically completes
    # only these authored one-write crash seams.
    allowed_view_states = {
        "pausing": {"approved", "pausing", "needs_attention"},
        "ready_to_delete": {"pausing", "ready_to_delete"},
        "delete_intent": {"ready_to_delete", "delete_intent"},
        "replacement_ready": {"delete_intent", "replacement_ready"},
    }
    if view.state not in allowed_view_states[marker.phase]:
        raise GpuSwitchStoreCorruptError("switch marker and request phases disagree")
    if marker.phase == "ready_to_delete" and view.state == "pausing" and view.reason is not None:
        raise GpuSwitchStoreCorruptError("switch marker cannot advance a cancellation view")

    # The first pausing marker may precede publication of the finalization
    # state in an approved view. Every later/settled view must match exactly.
    if view.state == "approved":
        if view.batch_state_at_finalization is not None:
            raise GpuSwitchStoreCorruptError("approved marker seam has finalization state")
    elif view.batch_state_at_finalization != marker.batch_state_at_finalization:
        raise GpuSwitchStoreCorruptError("switch marker batch finalization state changed")

    if marker.phase == "ready_to_delete" and view.state == "ready_to_delete":
        if view.ready_to_delete_at != marker.updated_at:
            raise GpuSwitchStoreCorruptError("switch pause fixed-point timestamp changed")
    elif marker.phase in {"delete_intent", "replacement_ready"}:
        if view.ready_to_delete_at is None or view.ready_to_delete_at > marker.updated_at:
            raise GpuSwitchStoreCorruptError("switch pause fixed-point timestamp is invalid")

    marker_replacement = (
        marker.replacement_attempt_id,
        marker.replacement_attempt_revision,
        marker.replacement_pod_id,
        marker.actual_target_gpu_id,
    )
    view_replacement = (
        view.replacement_attempt_id,
        view.replacement_attempt_revision,
        view.replacement_pod_id,
        view.actual_target_gpu_id,
    )
    if marker.phase == "replacement_ready":
        if view.state == "replacement_ready":
            if view_replacement != marker_replacement:
                raise GpuSwitchStoreCorruptError("switch replacement identity changed")
        elif any(value is not None for value in view_replacement):
            raise GpuSwitchStoreCorruptError(
                "pre-adoption request seam carries replacement identity"
            )
    elif any(value is not None for value in marker_replacement + view_replacement):
        raise GpuSwitchStoreCorruptError("replacement identity appeared before adoption")


def _canonical_bytes(model: object) -> bytes:
    if hasattr(model, "model_dump"):
        value = model.model_dump(mode="json")  # type: ignore[union-attr]
    else:
        value = model
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(principal: Principal, request: CreateGpuSwitchRequestV1) -> str:
    payload = {
        "schema_version": 1,
        "requester_user_id": principal.user_id,
        "request": request.model_dump(mode="json"),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _new_uuid4() -> str:
    raw = bytearray(secrets.token_bytes(16))
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


def load_runtime_identity_contract(
    path: Path | None = None,
    *,
    allow_test_override: bool = False,
) -> GpuRuntimeIdentityContractV1:
    """Load the packaged strict runtime mapping without production overrides."""

    if path is not None and not allow_test_override:
        raise ValueError("runtime identity contract override is test-only")
    selected = path if path is not None else PACKAGED_RUNTIME_IDENTITIES_PATH
    if selected.stat().st_size > 64 * 1024:
        raise ValueError("runtime identity contract exceeds its byte cap")
    return GpuRuntimeIdentityContractV1.model_validate_json(selected.read_bytes())


def load_gpu_switch_code_contract(
    path: Path | None = None,
    *,
    allow_test_override: bool = False,
) -> GpuSwitchCodeContractV1:
    """Load the packaged exhaustive code registry without production overrides."""

    if path is not None and not allow_test_override:
        raise ValueError("GPU switch code contract override is test-only")
    selected = path if path is not None else PACKAGED_SWITCH_CODES_PATH
    if selected.stat().st_size > 256 * 1024:
        raise ValueError("GPU switch code contract exceeds its byte cap")
    return GpuSwitchCodeContractV1.model_validate_json(selected.read_bytes())


def _validate_worker_code_contract(contract: GpuSwitchCodeContractV1) -> None:
    actions = {
        entry.code: (entry.httpStatus, entry.retryable)
        for entry in contract.codes
        if entry.scope == "worker_action"
    }
    expected_actions = {code: (status, False) for code, (status, _message) in _ERRORS.items()}
    if actions != expected_actions:
        raise RuntimeError("packaged GPU switch worker action registry drifted")
    blocks = {entry.code for entry in contract.codes if entry.scope == "worker_block"}
    if blocks != set(get_args(GpuSwitchBlockCode)):
        raise RuntimeError("packaged GPU switch worker block registry drifted")


_GPU_SWITCH_CODE_CONTRACT = load_gpu_switch_code_contract()
_validate_worker_code_contract(_GPU_SWITCH_CODE_CONTRACT)


def _terminal_state(reason: str) -> str:
    if reason == "peer_denied":
        return "denied"
    if reason in {"response_timeout", "requester_expired"}:
        return "expired"
    return "cancelled"


class GpuSwitchStore:
    """Strict crash-atomic Switch envelope, marker, tombstone, and history store."""

    def __init__(
        self,
        root: Path,
        manifest_store: ManifestStore,
        *,
        fsync_writes: bool = True,
        crash_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root
        self.manifest_store = manifest_store
        self.fsync_writes = fsync_writes
        self.crash_hook = crash_hook
        self.requests_root = root / ".gpu-switch-requests-v1"
        self.tombstones_root = root / ".gpu-switch-tombstones-v1"
        self.history_root = root / ".gpu-switch-history-v1"
        self.marker_path = root / ".gpu-switch-v1.json"

    def initialize(self) -> None:
        for directory in (self.requests_root, self.tombstones_root, self.history_root):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._validate_no_orphan_temps()
        envelopes = self.list_envelopes()
        envelopes_by_id = {item.switch_id: item for item in envelopes}
        tombstones = {item.switch_id: item for item in self.list_tombstones()}
        marker = self.read_marker()
        for switch_id in tombstones:
            if switch_id not in envelopes_by_id:
                # Valid code always publishes a fingerprint-bearing envelope
                # before a direct settle tombstone, or retains the prior active
                # envelope during ordinary tombstone-first terminalization.
                # With no envelope, exact session/fingerprint recovery is
                # impossible, so fail closed rather than permit UUID reuse.
                raise GpuSwitchStoreCorruptError("orphan Switch tombstone has no envelope")
        for envelope in envelopes:
            expected_name = f"{hashlib.sha256(envelope.switch_id.encode()).hexdigest()}.json"
            if self._request_path(envelope.switch_id).name != expected_name:
                raise GpuSwitchStoreCorruptError("request path identity mismatch")
            tombstone = tombstones.get(envelope.switch_id)
            if envelope.terminal_tombstone is not None:
                if tombstone is not None and tombstone != envelope.terminal_tombstone:
                    raise GpuSwitchStoreCorruptError("terminal request history is inconsistent")
            elif tombstone is not None and (
                tombstone.principal_binding_id != envelope.principal_binding_id
                or tombstone.requester_user_id != envelope.requester_user_id
            ):
                raise GpuSwitchStoreCorruptError("terminal request identity is inconsistent")
            if (
                envelope.active_request is not None
                and envelope.active_request.state not in {"pending", "approved"}
                and (marker is None or marker.switch_id != envelope.switch_id)
            ):
                raise GpuSwitchStoreCorruptError(
                    "finalized Switch request has no matching durable marker"
                )
        if marker is not None:
            envelope = next(
                (item for item in envelopes if item.switch_id == marker.switch_id), None
            )
            if envelope is None:
                raise GpuSwitchStoreCorruptError("switch marker has no active request envelope")
            if (
                envelope.principal_binding_id != marker.principal_binding_id
                or envelope.requester_user_id != marker.requester_user_id
            ):
                raise GpuSwitchStoreCorruptError("switch marker identity mismatch")
            view = envelope.active_request
            if view is not None:
                _require_marker_view_binding(marker, view)
            effective_tombstone = envelope.terminal_tombstone or tombstones.get(envelope.switch_id)
            if effective_tombstone is not None:
                self._require_marker_terminal_clear_allowed(marker, effective_tombstone)

    def reconcile_terminal_commits(self) -> None:
        """Finish only the deterministic half of a tombstone/envelope crash seam."""

        self._require_gpu_control_lock()
        tombstones = {item.switch_id: item for item in self.list_tombstones()}
        for envelope in self.list_envelopes():
            separate = tombstones.get(envelope.switch_id)
            if envelope.terminal_tombstone is not None:
                if separate is None:
                    self._write_immutable(
                        self._tombstone_path(envelope.switch_id),
                        _canonical_bytes(envelope.terminal_tombstone),
                        "gpu_switch_tombstone",
                    )
                elif separate != envelope.terminal_tombstone:
                    raise GpuSwitchStoreCorruptError("terminal histories disagree")
            elif separate is not None:
                if (
                    separate.principal_binding_id != envelope.principal_binding_id
                    or separate.requester_user_id != envelope.requester_user_id
                ):
                    raise GpuSwitchStoreCorruptError("terminal histories disagree")
                marker = self.read_marker()
                if marker is not None and marker.switch_id == envelope.switch_id:
                    self._require_marker_terminal_clear_allowed(marker, separate)
                next_envelope = envelope.model_copy(
                    update={
                        "envelope_revision": envelope.envelope_revision + 1,
                        "active_request": None,
                        "terminal_tombstone": separate,
                        "updated_at": separate.terminal_at,
                    }
                )
                self.write_envelope(next_envelope, previous=envelope)

    def list_envelopes(self) -> list[SharedGpuSwitchRequestEnvelopeV1]:
        result: list[SharedGpuSwitchRequestEnvelopeV1] = []
        seen: set[str] = set()
        try:
            paths = sorted(self.requests_root.glob("*.json"))
        except OSError as exc:
            raise GpuSwitchStoreCorruptError("request history cannot be enumerated") from exc
        for path in paths:
            envelope = self._read_model(
                path, SharedGpuSwitchRequestEnvelopeV1, MAX_GPU_SWITCH_ENVELOPE_BYTES
            )
            if path != self._request_path(envelope.switch_id) or envelope.switch_id in seen:
                raise GpuSwitchStoreCorruptError("request envelope path or identity is duplicated")
            seen.add(envelope.switch_id)
            result.append(envelope)
        return result

    def list_tombstones(self) -> list[SharedGpuSwitchTombstoneV1]:
        result: list[SharedGpuSwitchTombstoneV1] = []
        seen: set[str] = set()
        try:
            paths = sorted(self.tombstones_root.glob("*.json"))
        except OSError as exc:
            raise GpuSwitchStoreCorruptError("tombstone history cannot be enumerated") from exc
        for path in paths:
            tombstone = self._read_model(
                path, SharedGpuSwitchTombstoneV1, MAX_GPU_SWITCH_TOMBSTONE_BYTES
            )
            if path != self._tombstone_path(tombstone.switch_id) or tombstone.switch_id in seen:
                raise GpuSwitchStoreCorruptError("tombstone path or identity is duplicated")
            seen.add(tombstone.switch_id)
            result.append(tombstone)
        return result

    def read_envelope(self, switch_id: str) -> SharedGpuSwitchRequestEnvelopeV1 | None:
        path = self._request_path(switch_id)
        if not path.exists():
            return None
        return self._read_model(
            path, SharedGpuSwitchRequestEnvelopeV1, MAX_GPU_SWITCH_ENVELOPE_BYTES
        )

    def read_tombstone(self, switch_id: str) -> SharedGpuSwitchTombstoneV1 | None:
        path = self._tombstone_path(switch_id)
        if not path.exists():
            return None
        return self._read_model(path, SharedGpuSwitchTombstoneV1, MAX_GPU_SWITCH_TOMBSTONE_BYTES)

    def read_marker(self) -> SharedGpuSwitchMarkerV1 | None:
        if not self.marker_path.exists():
            return None
        return self._read_model(
            self.marker_path, SharedGpuSwitchMarkerV1, MAX_GPU_SWITCH_MARKER_BYTES
        )

    def write_envelope(
        self,
        envelope: SharedGpuSwitchRequestEnvelopeV1,
        *,
        previous: SharedGpuSwitchRequestEnvelopeV1 | None,
    ) -> None:
        self._require_gpu_control_lock()
        envelope = SharedGpuSwitchRequestEnvelopeV1.model_validate(
            envelope.model_dump(mode="python")
        )
        if previous is None:
            if envelope.envelope_revision != 1 or self._request_path(envelope.switch_id).exists():
                raise GpuSwitchStoreCorruptError("new request revision is invalid")
        else:
            if envelope.switch_id != previous.switch_id:
                raise GpuSwitchStoreCorruptError("request identity changed")
            if envelope.envelope_revision != previous.envelope_revision + 1:
                raise GpuSwitchStoreCorruptError("request revision did not increase by one")
            if previous.envelope_revision >= MAX_SAFE_REVISION:
                raise GpuSwitchStoreCorruptError("request revision is exhausted")
            for field in (
                "request_fingerprint_sha256",
                "requester_user_id",
                "requester_session_id",
                "principal_binding_id",
                "created_at",
            ):
                if getattr(envelope, field) != getattr(previous, field):
                    raise GpuSwitchStoreCorruptError("immutable request identity changed")
            if previous.terminal_tombstone is not None:
                raise GpuSwitchStoreCorruptError("terminal request history is immutable")
            if previous.active_request is not None and envelope.active_request is not None:
                immutable_view_fields = (
                    "switch_id",
                    "old_pod_id",
                    "old_gpu_id",
                    "old_gpu_display_name",
                    "initial_target_gpu_id",
                    "initial_target_gpu_display_name",
                    "initial_replacement_attempt_id",
                    "requester",
                    "requested_at",
                    "response_deadline",
                    "batch_id",
                    "batch_owner",
                )
                if any(
                    getattr(envelope.active_request, field)
                    != getattr(previous.active_request, field)
                    for field in immutable_view_fields
                ):
                    raise GpuSwitchStoreCorruptError("immutable Switch request view changed")
        self._atomic_write(
            self._request_path(envelope.switch_id),
            _canonical_bytes(envelope),
            "gpu_switch_envelope",
        )

    def write_marker(
        self,
        marker: SharedGpuSwitchMarkerV1,
        *,
        previous: SharedGpuSwitchMarkerV1 | None,
    ) -> None:
        self._require_gpu_control_lock()
        marker = SharedGpuSwitchMarkerV1.model_validate(marker.model_dump(mode="python"))
        current = self.read_marker()
        if current != previous:
            raise GpuControlGuardConflictError("current switch marker changed")
        if previous is None:
            if marker.phase != "pausing":
                raise GpuSwitchStoreCorruptError("new marker must begin at pausing")
        else:
            immutable = (
                "switch_id",
                "finalization_id",
                "principal_binding_id",
                "requester_user_id",
                "requester_display_name",
                "old_pod_id",
                "old_gpu_id",
                "initial_target_gpu_id",
                "initial_replacement_attempt_id",
                "batch_id",
                "batch_owner_user_id",
                "batch_state_at_finalization",
                "expected_volume_id",
                "expected_data_center_id",
                "expected_image_digest",
                "expected_model_id",
                "expected_model_revision",
                "requested_at",
            )
            if any(getattr(marker, key) != getattr(previous, key) for key in immutable):
                raise GpuSwitchStoreCorruptError("immutable marker identity changed")
            allowed = {
                "pausing": {"pausing", "ready_to_delete"},
                "ready_to_delete": {"ready_to_delete", "delete_intent"},
                "delete_intent": {"delete_intent", "replacement_ready"},
                "replacement_ready": {"replacement_ready"},
            }
            if marker.phase not in allowed[previous.phase]:
                raise GpuSwitchStoreCorruptError("illegal marker transition")
        payload = _canonical_bytes(marker)
        if len(payload) > MAX_GPU_SWITCH_MARKER_BYTES:
            raise GpuSwitchStoreCorruptError("switch marker exceeds its byte cap")
        self._atomic_write(self.marker_path, payload, "gpu_switch_marker")

    def terminalize(
        self,
        envelope: SharedGpuSwitchRequestEnvelopeV1,
        tombstone: SharedGpuSwitchTombstoneV1,
    ) -> SharedGpuSwitchRequestEnvelopeV1:
        self._require_gpu_control_lock()
        tombstone = SharedGpuSwitchTombstoneV1.model_validate(tombstone.model_dump(mode="python"))
        existing_tombstone = self.read_tombstone(tombstone.switch_id)
        if existing_tombstone is not None:
            if existing_tombstone != tombstone:
                raise GpuSwitchStoreCorruptError("terminal tombstone identity mismatch")
        else:
            # Tombstone first: a crash leaves the original fingerprint-bearing
            # active envelope available for deterministic startup reconciliation.
            self._write_immutable(
                self._tombstone_path(tombstone.switch_id),
                _canonical_bytes(tombstone),
                "gpu_switch_tombstone",
            )
        if envelope.terminal_tombstone is not None:
            if envelope.terminal_tombstone != tombstone:
                raise GpuSwitchStoreCorruptError("request is already terminal with different data")
            next_envelope = envelope
        else:
            next_envelope = envelope.model_copy(
                update={
                    "envelope_revision": envelope.envelope_revision + 1,
                    "active_request": None,
                    "terminal_tombstone": tombstone,
                    "updated_at": tombstone.terminal_at,
                }
            )
            self.write_envelope(next_envelope, previous=envelope)
        return next_envelope

    def archive_marker_and_clear(
        self,
        marker: SharedGpuSwitchMarkerV1,
        tombstone: SharedGpuSwitchTombstoneV1,
    ) -> None:
        self._require_gpu_control_lock()
        current = self.read_marker()
        if current != marker:
            if current is None and self.read_tombstone(marker.switch_id) == tombstone:
                return
            raise GpuControlGuardConflictError("switch marker changed before completion")
        self._require_marker_terminal_clear_allowed(marker, tombstone)
        history_path = self.history_root / (
            f"{hashlib.sha256(marker.switch_id.encode()).hexdigest()}.json"
        )
        self._write_immutable(history_path, _canonical_bytes(marker), "gpu_switch_history")
        self.marker_path.unlink(missing_ok=True)
        self._fsync_directory(self.root)
        self._crash("gpu_switch_marker_clear")

    @staticmethod
    def _require_marker_terminal_clear_allowed(
        marker: SharedGpuSwitchMarkerV1,
        tombstone: SharedGpuSwitchTombstoneV1,
    ) -> None:
        """Reject a terminal history that would erase forward-only authority.

        A pre-delete cancellation may clear only its exact finalized marker.
        After delete intent, the only legal clear is exact replacement
        completion from ``replacement_ready``.  Any other independently valid
        marker/tombstone pair is a cross-authority conflict and both records
        remain on disk for explicit shared-volume repair.
        """

        common_identity_matches = (
            marker.switch_id == tombstone.switch_id
            and marker.principal_binding_id == tombstone.principal_binding_id
            and marker.requester_user_id == tombstone.requester_user_id
            and marker.finalization_id == tombstone.finalization_id
        )
        if not common_identity_matches:
            raise GpuControlGuardConflictError(
                "switch marker and terminal history authority disagree"
            )
        replacement_matches = (
            marker.replacement_attempt_id == tombstone.replacement_attempt_id
            and marker.replacement_attempt_revision == tombstone.replacement_attempt_revision
            and marker.replacement_pod_id == tombstone.replacement_pod_id
            and marker.actual_target_gpu_id == tombstone.actual_target_gpu_id
        )
        if marker.phase in {"pausing", "ready_to_delete"}:
            if (
                tombstone.terminal_state == "cancelled"
                and tombstone.terminal_reason
                in {"requester_cancelled", "target_changed_pre_delete"}
                and all(
                    value is None
                    for value in (
                        tombstone.replacement_attempt_id,
                        tombstone.replacement_attempt_revision,
                        tombstone.replacement_pod_id,
                        tombstone.actual_target_gpu_id,
                    )
                )
            ):
                return
        elif marker.phase == "replacement_ready":
            if (
                tombstone.terminal_state == "completed"
                and tombstone.terminal_reason == "replacement_completed"
                and replacement_matches
                and all(
                    value is not None
                    for value in (
                        tombstone.replacement_attempt_id,
                        tombstone.replacement_attempt_revision,
                        tombstone.replacement_pod_id,
                        tombstone.actual_target_gpu_id,
                    )
                )
            ):
                return
        raise GpuControlGuardConflictError(
            "forward-only switch marker conflicts with terminal history"
        )

    def tombstone_sha256(self, tombstone: SharedGpuSwitchTombstoneV1) -> str:
        return hashlib.sha256(_canonical_bytes(tombstone)).hexdigest()

    def _request_path(self, switch_id: str) -> Path:
        return self.requests_root / f"{hashlib.sha256(switch_id.encode()).hexdigest()}.json"

    def _tombstone_path(self, switch_id: str) -> Path:
        return self.tombstones_root / f"{hashlib.sha256(switch_id.encode()).hexdigest()}.json"

    def _require_gpu_control_lock(self) -> None:
        if not self.manifest_store.gpu_control_lock_held:
            raise RuntimeError("GPU switch mutation requires the shared GPU-control lock")

    def _read_model(self, path: Path, model_type: type, maximum_bytes: int):  # type: ignore[no-untyped-def]
        try:
            if path.stat().st_size > maximum_bytes:
                raise GpuSwitchStoreCorruptError("GPU switch record exceeds its byte cap")
            payload = path.read_bytes()
            return model_type.model_validate_json(payload)
        except GpuSwitchStoreCorruptError:
            raise
        except (OSError, UnicodeDecodeError, ValidationError, ValueError) as exc:
            raise GpuSwitchStoreCorruptError("GPU switch record is invalid") from exc

    def _write_immutable(self, path: Path, payload: bytes, seam: str) -> None:
        if path.exists():
            try:
                if secrets.compare_digest(path.read_bytes(), payload):
                    return
            except OSError as exc:
                raise GpuSwitchStoreCorruptError("immutable history cannot be read") from exc
            raise GpuSwitchStoreCorruptError("immutable history already has different bytes")
        self._atomic_write(path, payload, seam)

    def _atomic_write(self, path: Path, payload: bytes, seam: str) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                if self.fsync_writes:
                    os.fsync(handle.fileno())
            self._crash(f"{seam}_file_fsync")
            os.replace(temporary, path)
            self._crash(f"{seam}_rename")
            self._fsync_directory(path.parent)
            self._crash(f"{seam}_directory_fsync")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _validate_no_orphan_temps(self) -> None:
        directories = (self.root, self.requests_root, self.tombstones_root, self.history_root)
        try:
            if any(
                any(path.name.endswith(".tmp") for path in root.iterdir()) for root in directories
            ):
                raise GpuControlGuardConflictError("orphan GPU-control temporary file exists")
        except OSError as exc:
            raise GpuSwitchStoreCorruptError("GPU switch directory cannot be inspected") from exc

    def _fsync_directory(self, path: Path) -> None:
        if not self.fsync_writes or os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _crash(self, seam: str) -> None:
        if self.crash_hook is not None:
            self.crash_hook(seam)


@dataclass(frozen=True, slots=True)
class RuntimeDeviceObservation:
    device_count: int
    device: WorkerCudaDeviceIdentityV1


class RuntimeDeviceInspector(Protocol):
    def inspect(self) -> RuntimeDeviceObservation: ...


class NvidiaRuntimeDeviceInspector:
    """Use nvidia-smi/NVML for identity and torch CUDA for capability/count."""

    def inspect(self) -> RuntimeDeviceObservation:
        try:
            nvidia_smi = shutil.which("nvidia-smi")
            if nvidia_smi is None:
                raise ValueError("nvidia-smi is unavailable")
            completed = subprocess.run(  # noqa: S603
                [
                    nvidia_smi,
                    "--query-gpu=uuid,pci.device_id,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            rows = [row for row in completed.stdout.splitlines() if row]
            if len(rows) != 1:
                raise ValueError("exactly one NVML device is required")
            fields = [field.strip() for field in rows[0].split(",")]
            if len(fields) != 4:
                raise ValueError("NVML identity output is malformed")
            nvml_uuid, raw_pci_device_id, cuda_name, memory_mib = fields
            pci_device_id = _canonical_nvml_pci_device_id(raw_pci_device_id)
            import torch

            if torch.cuda.device_count() != 1:
                raise ValueError("exactly one CUDA device is required")
            major, minor = torch.cuda.get_device_capability(0)
            return RuntimeDeviceObservation(
                device_count=1,
                device=WorkerCudaDeviceIdentityV1(
                    deviceIndex=0,
                    nvmlUuid=nvml_uuid,
                    pciDeviceId=pci_device_id.lower(),
                    cudaName=cuda_name,
                    totalMemoryBytes=int(memory_mib) * 1024 * 1024,
                    computeCapabilityMajor=major,
                    computeCapabilityMinor=minor,
                ),
            )
        except Exception:
            raise gpu_switch_error("gpu_switch_runtime_identity_unavailable") from None


def _canonical_nvml_pci_device_id(value: str) -> str:
    """Project nvidia-smi's device+vendor token to the strict device ID wire form."""

    normalized = value.strip().lower()
    if not normalized.startswith("0x"):
        raise ValueError("NVML PCI identity is malformed")
    digits = normalized[2:]
    # nvidia-smi commonly returns DEVICE_ID followed by NVIDIA's vendor ID
    # (for example 0x268410DE).  The frozen worker wire contract intentionally
    # carries only the lowercase four-digit PCI device ID.
    if len(digits) == 8 and digits[4:] == "10de":
        digits = digits[:4]
    if len(digits) != 4 or any(character not in "0123456789abcdef" for character in digits):
        raise ValueError("NVML PCI identity is malformed")
    return f"0x{digits}"


@dataclass(slots=True)
class _DecisionState:
    participants: dict[str, GpuSwitchParticipantV1]
    approvals: set[str]
    denied: set[str]
    deadline_monotonic: float


class GpuSwitchCoordinator:
    """One controller-locked, volume-linearized worker Switch state machine."""

    def __init__(
        self,
        store: GpuSwitchStore,
        coordination: StudioCoordinator,
        runtime_metadata: Mapping[str, str],
        data_root: Path,
        *,
        runtime_inspector: RuntimeDeviceInspector | None = None,
    ) -> None:
        self.store = store
        self.coordination = coordination
        self.runtime_metadata = dict(runtime_metadata)
        self.data_root = data_root
        self.runtime_inspector = runtime_inspector or NvidiaRuntimeDeviceInspector()
        self._decisions: dict[str, _DecisionState] = {}
        self._response_replays: dict[tuple[str, str], tuple[str, str]] = {}

    def initialize(
        self, active: BatchManifest | None, stop_guard: SharedGpuStopGuard | None
    ) -> None:
        self.store.initialize()
        self.store.reconcile_terminal_commits()
        marker = self._safe_marker()
        if marker is not None and stop_guard is not None:
            raise GpuControlGuardConflictError("Stop and Switch guards coexist")
        if marker is not None:
            # Validate the manifest binding before any terminal crash recovery
            # is allowed to archive and clear the guard.  Otherwise a valid
            # pre-delete tombstone beside a missing/corrupt bound manifest
            # could erase the only generation veto during startup.
            self._require_marker_batch_binding(marker, active)
        now = utc_now()
        for envelope in self._safe_envelopes():
            if envelope.active_request is None:
                if marker is not None and marker.switch_id == envelope.switch_id:
                    tombstone = envelope.terminal_tombstone
                    assert tombstone is not None
                    self.store.archive_marker_and_clear(marker, tombstone)
                    marker = None
                continue
            if marker is not None and marker.switch_id == envelope.switch_id:
                continue
            # Consent authority is process-epoch local. A restart makes every
            # pre-finalization request terminal before coordination becomes ready.
            tombstone = self._tombstone_for(envelope, "requester_expired", now)
            self.store.terminalize(envelope, tombstone)
        if marker is not None:
            envelope = self._safe_envelope(marker.switch_id)
            if envelope is None or envelope.active_request is None:
                raise GpuSwitchStoreCorruptError("marker lost its active request envelope")
            projected = self._view_from_marker(envelope.active_request, marker)
            if projected != envelope.active_request:
                self._replace_active_view(envelope, projected)

    def create(
        self,
        principal: Principal,
        request: CreateGpuSwitchRequestV1,
        active: BatchManifest | None,
        stop_guard: SharedGpuStopGuard | None,
    ) -> NativeWorkerGpuSwitchCreateResponseV1:
        requester = self.coordination.require_foreground_session(principal, request.session_id)
        fingerprint = _fingerprint(principal, request)
        existing = self._safe_envelope(request.switch_id)
        existing_tombstone = self.store.read_tombstone(request.switch_id)
        if existing is None and existing_tombstone is not None:
            raise gpu_switch_error("gpu_switch_request_not_found")
        if existing is not None:
            if existing.terminal_tombstone is not None:
                raise gpu_switch_error("gpu_switch_request_not_found")
            if existing.requester_user_id != principal.user_id or (
                existing.requester_session_id != request.session_id
            ):
                raise gpu_switch_error("gpu_switch_request_not_found")
            if not secrets.compare_digest(existing.request_fingerprint_sha256, fingerprint):
                raise gpu_switch_error("gpu_switch_identity_mismatch")
            return NativeWorkerGpuSwitchCreateResponseV1(
                schema_version=1,
                request=existing.active_request,
                requester_user_id=existing.requester_user_id,
                principal_binding_id=existing.principal_binding_id,
            )
        self._raise_store_or_guard(stop_guard)
        if self.coordination.stop_request is not None and self.coordination.stop_request.state in {
            "pending",
            "approved",
        }:
            raise gpu_switch_error("stop_request_in_progress")
        if self.coordination.stop_request is not None and self.coordination.stop_request.state == (
            "finalizing"
        ):
            raise gpu_switch_error("gpu_stop_pending")
        other = next(
            (item for item in self._safe_envelopes() if item.active_request is not None), None
        )
        if other is not None:
            raise gpu_switch_error("gpu_switch_request_in_progress")
        self._validate_runtime_preflight(request)
        if (active.batch_id if active is not None else None) != request.expected_batch_id:
            raise gpu_switch_error("gpu_switch_batch_changed")
        foreground = self.coordination.foreground_principals()
        if active is not None and active.owner.user_id != principal.user_id:
            if active.owner.user_id not in foreground:
                raise gpu_switch_error("switch_owner_unavailable")
        participants = {
            user_id: GpuSwitchParticipantV1(
                session_id=identity.session_id, display_name=identity.display_name
            )
            for user_id, identity in foreground.items()
            if user_id != principal.user_id
        }
        requested_at = utc_now()
        response_deadline = (
            (
                self.coordination.clock.utcnow().astimezone(UTC)
                + timedelta(seconds=GPU_SWITCH_RESPONSE_TTL_SECONDS)
            )
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        view = GpuSwitchRequestViewV1(
            schema_version=1,
            switch_id=request.switch_id,
            old_pod_id=request.old_pod_id,
            old_gpu_id=request.old_gpu_id,
            old_gpu_display_name=request.old_gpu_display_name,
            initial_target_gpu_id=request.initial_target_gpu_id,
            initial_target_gpu_display_name=request.initial_target_gpu_display_name,
            initial_replacement_attempt_id=request.initial_replacement_attempt_id,
            requester=GpuSwitchParticipantV1(
                session_id=requester.session_id, display_name=requester.display_name
            ),
            state="pending" if participants else "approved",
            reason=None,
            requested_at=requested_at,
            response_deadline=response_deadline,
            ready_to_delete_at=None,
            waiting_for=self._sorted_participants(participants.values()),
            approved_by=[],
            denied_by=[],
            batch_id=active.batch_id if active is not None else None,
            batch_owner=(
                GpuSwitchBatchOwnerV1(display_name=active.owner.display_name)
                if active is not None
                else None
            ),
            batch_state_at_finalization=None,
            replacement_attempt_id=None,
            replacement_attempt_revision=None,
            replacement_pod_id=None,
            actual_target_gpu_id=None,
        )
        principal_binding_id = _new_uuid4()
        envelope = SharedGpuSwitchRequestEnvelopeV1(
            schema_version=1,
            envelope_revision=1,
            switch_id=request.switch_id,
            request_fingerprint_sha256=fingerprint,
            requester_user_id=principal.user_id,
            requester_session_id=request.session_id,
            principal_binding_id=principal_binding_id,
            active_request=view,
            terminal_tombstone=None,
            created_at=requested_at,
            updated_at=requested_at,
        )
        self.store.write_envelope(envelope, previous=None)
        self._decisions[request.switch_id] = _DecisionState(
            participants=participants,
            approvals=set(),
            denied=set(),
            deadline_monotonic=(
                self.coordination.clock.monotonic() + GPU_SWITCH_RESPONSE_TTL_SECONDS
            ),
        )
        return NativeWorkerGpuSwitchCreateResponseV1(
            schema_version=1,
            request=view,
            requester_user_id=principal.user_id,
            principal_binding_id=principal_binding_id,
        )

    def refresh(self, active: BatchManifest | None) -> GpuSwitchRequestViewV1 | None:
        marker = self._safe_marker()
        if marker is not None and marker.phase == "pausing":
            self._observe_pause_timeout(marker, active)
        envelope = self._active_envelope(marker.switch_id if marker is not None else None)
        if envelope is None or envelope.active_request is None:
            return None
        if marker is not None:
            return self._view_from_marker(envelope.active_request, marker)
        decision = self._decisions.get(envelope.switch_id)
        if decision is None:
            # A pre-marker envelope from an earlier process is never revived.
            return None
        view = envelope.active_request
        foreground = self.coordination.foreground_principals()
        if envelope.requester_user_id not in foreground:
            self._terminalize(envelope, "requester_expired")
            return None
        for user_id, identity in foreground.items():
            if user_id == envelope.requester_user_id or user_id in decision.denied:
                continue
            decision.participants.setdefault(
                user_id,
                GpuSwitchParticipantV1(
                    session_id=identity.session_id, display_name=identity.display_name
                ),
            )
        for user_id in list(decision.participants):
            if user_id not in foreground and user_id not in decision.approvals:
                decision.participants.pop(user_id)
        if self.coordination.clock.monotonic() >= decision.deadline_monotonic:
            self._terminalize(envelope, "response_timeout")
            return None
        next_view = self._decision_view(view, decision, active)
        if next_view != view:
            envelope = self._replace_active_view(envelope, next_view)
        return envelope.active_request

    def can_respond(self, principal: Principal, session_id: str) -> bool:
        """Project peer eligibility without exposing or trusting a principal ID."""

        try:
            self.coordination.require_foreground_session(principal, session_id)
        except WorkerError:
            return False
        envelope = self._active_envelope(None)
        if envelope is None or envelope.active_request is None:
            return False
        if envelope.active_request.state != "pending":
            return False
        decision = self._decisions.get(envelope.switch_id)
        return bool(
            decision is not None
            and principal.user_id != envelope.requester_user_id
            and principal.user_id in decision.participants
            and principal.user_id not in decision.approvals
            and principal.user_id not in decision.denied
        )

    def public_lookup(
        self, principal: Principal, switch_id: str, session_id: str
    ) -> GpuSwitchLookupResponseV1:
        self.coordination.require_foreground_session(principal, session_id)
        envelope = self._require_owner_envelope(principal, switch_id, session_id, private=False)
        return self._lookup(envelope)

    def owner_lookup(
        self, principal: Principal, switch_id: str, session_id: str
    ) -> NativeWorkerGpuSwitchOwnerLookupV1:
        self.coordination.require_foreground_session(principal, session_id)
        envelope = self._require_owner_envelope(principal, switch_id, session_id, private=True)
        lookup = self._lookup(envelope)
        marker = self._safe_marker()
        tombstone = envelope.terminal_tombstone
        return NativeWorkerGpuSwitchOwnerLookupV1(
            **lookup.model_dump(mode="python"),
            requester_user_id=envelope.requester_user_id,
            principal_binding_id=envelope.principal_binding_id,
            finalization_id=(
                marker.finalization_id
                if marker is not None and marker.switch_id == switch_id
                else tombstone.finalization_id
                if tombstone is not None
                else None
            ),
            terminal_tombstone_sha256=(
                self.store.tombstone_sha256(tombstone) if tombstone is not None else None
            ),
        )

    def settle_create(
        self,
        principal: Principal,
        switch_id: str,
        request: SettleGpuSwitchCreateRequestV1,
    ) -> NativeWorkerGpuSwitchOwnerLookupV1:
        if request.create_request.switch_id != switch_id:
            raise gpu_switch_error("gpu_switch_identity_mismatch")
        fingerprint = _fingerprint(principal, request.create_request)
        envelope = self._safe_envelope(switch_id)
        if envelope is None:
            existing_tombstone = self.store.read_tombstone(switch_id)
            if existing_tombstone is not None:
                raise gpu_switch_error("gpu_switch_store_corrupt")
            now = utc_now()
            tombstone = SharedGpuSwitchTombstoneV1(
                schema_version=1,
                switch_id=switch_id,
                principal_binding_id=_new_uuid4(),
                requester_user_id=principal.user_id,
                finalization_id=None,
                terminal_state="cancelled",
                terminal_reason="requester_cancelled",
                replacement_attempt_id=None,
                replacement_attempt_revision=None,
                replacement_pod_id=None,
                actual_target_gpu_id=None,
                terminal_at=now,
            )
            envelope = SharedGpuSwitchRequestEnvelopeV1(
                schema_version=1,
                envelope_revision=1,
                switch_id=switch_id,
                request_fingerprint_sha256=fingerprint,
                requester_user_id=principal.user_id,
                requester_session_id=request.create_request.session_id,
                principal_binding_id=tombstone.principal_binding_id,
                active_request=None,
                terminal_tombstone=tombstone,
                created_at=now,
                updated_at=now,
            )
            self.store.write_envelope(envelope, previous=None)
            self.store._write_immutable(
                self.store._tombstone_path(switch_id),
                _canonical_bytes(tombstone),
                "gpu_switch_tombstone",
            )
        else:
            if (
                envelope.requester_user_id != principal.user_id
                or envelope.requester_session_id != request.create_request.session_id
                or not secrets.compare_digest(envelope.request_fingerprint_sha256, fingerprint)
            ):
                raise gpu_switch_error("gpu_switch_identity_mismatch")
            marker = self._safe_marker()
            if marker is not None and marker.switch_id == switch_id:
                raise gpu_switch_error("gpu_switch_cancel_not_allowed")
            if envelope.active_request is not None:
                if envelope.active_request.state not in {"pending", "approved"}:
                    raise gpu_switch_error("gpu_switch_cancel_not_allowed")
                envelope = self._terminalize(envelope, "requester_cancelled")
        return self.owner_lookup(principal, switch_id, request.create_request.session_id)

    def respond(
        self,
        principal: Principal,
        switch_id: str,
        request: GpuSwitchResponseRequestV1,
        active: BatchManifest | None,
    ) -> GpuSwitchRequestViewV1 | None:
        response_identity = self.coordination.require_foreground_session(
            principal, request.session_id
        )
        envelope = self._safe_envelope(switch_id)
        decision = self._decisions.get(switch_id)
        replay = self._response_replays.get((switch_id, principal.user_id))
        if envelope is not None and envelope.terminal_tombstone is not None and replay is not None:
            if replay[0] == request.decision:
                return None
            raise gpu_switch_error("gpu_switch_response_conflict")
        if envelope is None or envelope.active_request is None or decision is None:
            raise gpu_switch_error("gpu_switch_request_not_found")
        if envelope.active_request.state not in {"pending", "approved"}:
            raise gpu_switch_error("gpu_switch_response_not_allowed")
        if principal.user_id == envelope.requester_user_id or principal.user_id not in (
            decision.participants
        ):
            raise gpu_switch_error("gpu_switch_response_not_allowed")
        prior = (
            "approve"
            if principal.user_id in decision.approvals
            else "deny"
            if principal.user_id in decision.denied
            else None
        )
        if prior is not None and prior != request.decision:
            raise gpu_switch_error("gpu_switch_response_conflict")
        if prior is not None:
            return envelope.active_request
        decision.participants[principal.user_id] = GpuSwitchParticipantV1(
            session_id=response_identity.session_id,
            display_name=response_identity.display_name,
        )
        self._response_replays[(switch_id, principal.user_id)] = (
            request.decision,
            request.session_id,
        )
        if request.decision == "deny":
            decision.denied.add(principal.user_id)
            self._terminalize(envelope, "peer_denied")
            return None
        decision.approvals.add(principal.user_id)
        next_view = self._decision_view(envelope.active_request, decision, active)
        return self._replace_active_view(envelope, next_view).active_request

    def finalize(
        self,
        principal: Principal,
        switch_id: str,
        request: FinalizeGpuSwitchRequestV1,
        active: BatchManifest | None,
        stop_guard: SharedGpuStopGuard | None,
    ) -> SharedGpuSwitchMarkerV1:
        self.coordination.require_foreground_session(principal, request.session_id)
        marker = self._safe_marker()
        if marker is not None:
            if (
                marker.switch_id == switch_id
                and marker.finalization_id == request.finalization_id
                and marker.requester_user_id == principal.user_id
            ):
                return self._resume_pause_failure(marker)
            raise gpu_switch_error("gpu_switch_finalization_mismatch")
        envelope = self._require_owner_envelope(
            principal, switch_id, request.session_id, private=False
        )
        self._raise_store_or_guard(stop_guard)
        view = self.refresh(active)
        if view is None:
            raise gpu_switch_error("gpu_switch_request_not_found")
        if view.state == "pending":
            raise gpu_switch_error("gpu_switch_approval_pending")
        if view.state != "approved":
            raise gpu_switch_error("gpu_switch_not_approved")
        if (active.batch_id if active is not None else None) != view.batch_id:
            self._terminalize(envelope, "batch_changed")
            raise gpu_switch_error("gpu_switch_batch_changed")
        if (
            active is not None
            and active.owner.user_id != principal.user_id
            and not (self.coordination.principal_has_foreground_session(active.owner.user_id))
        ):
            raise gpu_switch_error("switch_owner_unavailable")
        self._validate_runtime_marker_inputs(view)
        now = utc_now()
        batch_state = active.state.value if active is not None else None
        marker = SharedGpuSwitchMarkerV1(
            schema_version=1,
            switch_id=switch_id,
            finalization_id=request.finalization_id,
            principal_binding_id=envelope.principal_binding_id,
            requester_user_id=principal.user_id,
            requester_display_name=view.requester.display_name,
            old_pod_id=view.old_pod_id,
            old_gpu_id=view.old_gpu_id,
            initial_target_gpu_id=view.initial_target_gpu_id,
            initial_replacement_attempt_id=view.initial_replacement_attempt_id,
            batch_id=active.batch_id if active is not None else None,
            batch_owner_user_id=active.owner.user_id if active is not None else None,
            batch_state_at_finalization=batch_state,
            phase="pausing",
            replacement_attempt_id=None,
            replacement_attempt_revision=None,
            replacement_pod_id=None,
            actual_target_gpu_id=None,
            create_contract_revision=1,
            create_marker_sha256=None,
            create_intent_sha256=None,
            create_wire_body_sha256=None,
            expected_volume_id=self.runtime_metadata["RUNPOD_VOLUME_ID"],
            expected_data_center_id="EU-RO-1",
            expected_image_digest=self.runtime_metadata["IMAGEFORGE_IMAGE_DIGEST"],
            expected_model_id=MODEL_ID,
            expected_model_revision=MODEL_REVISION,
            requested_at=view.requested_at,
            updated_at=now,
        )
        self.store.write_marker(marker, previous=None)
        next_view = view.model_copy(
            update={"state": "pausing", "batch_state_at_finalization": batch_state}
        )
        self._replace_active_view(envelope, next_view)
        return marker

    def mark_ready_to_delete(self, active: BatchManifest | None) -> SharedGpuSwitchMarkerV1 | None:
        marker = self._safe_marker()
        if marker is None or marker.phase != "pausing":
            return marker
        self._require_marker_batch_binding(marker, active)
        envelope = self._safe_envelope(marker.switch_id)
        if envelope is None or envelope.active_request is None:
            raise GpuSwitchStoreCorruptError("marker lost its request envelope")
        view = envelope.active_request
        if view.state == "needs_attention":
            return marker
        active_is_safe = active is None or (
            active.state in {BatchState.PAUSED, BatchState.INTERRUPTED}
            and not any(
                image.status in {ImageState.GENERATING, ImageState.RETRYING}
                for image in active.images
            )
        )
        if not active_is_safe:
            if self._pause_deadline_elapsed(marker):
                self._replace_active_view(
                    envelope,
                    view.model_copy(update={"state": "needs_attention", "reason": "pause_failed"}),
                )
            return marker
        if envelope.active_request.reason == "requester_cancelled":
            tombstone = self._tombstone_for(
                envelope,
                "requester_cancelled",
                utc_now(),
                finalization_id=marker.finalization_id,
            )
            self.store.terminalize(envelope, tombstone)
            self.store.archive_marker_and_clear(marker, tombstone)
            self._decisions.pop(marker.switch_id, None)
            return None
        now = utc_now()
        next_marker = marker.model_copy(update={"phase": "ready_to_delete", "updated_at": now})
        self.store.write_marker(next_marker, previous=marker)
        next_view = envelope.active_request.model_copy(
            update={"state": "ready_to_delete", "ready_to_delete_at": now}
        )
        self._replace_active_view(envelope, next_view)
        return next_marker

    def _observe_pause_timeout(
        self,
        marker: SharedGpuSwitchMarkerV1,
        active: BatchManifest | None,
    ) -> None:
        self._require_marker_batch_binding(marker, active)
        envelope = self._safe_envelope(marker.switch_id)
        if envelope is None or envelope.active_request is None:
            raise GpuSwitchStoreCorruptError("marker lost its request envelope")
        view = envelope.active_request
        if view.state == "needs_attention":
            return
        fixed_point = active is None or (
            active.state in {BatchState.PAUSED, BatchState.INTERRUPTED}
            and not any(
                image.status in {ImageState.GENERATING, ImageState.RETRYING}
                for image in active.images
            )
        )
        if not fixed_point and self._pause_deadline_elapsed(marker):
            self._replace_active_view(
                envelope,
                view.model_copy(update={"state": "needs_attention", "reason": "pause_failed"}),
            )

    def _resume_pause_failure(self, marker: SharedGpuSwitchMarkerV1) -> SharedGpuSwitchMarkerV1:
        if marker.phase != "pausing":
            return marker
        envelope = self._safe_envelope(marker.switch_id)
        if envelope is None or envelope.active_request is None:
            raise GpuSwitchStoreCorruptError("marker lost its request envelope")
        view = envelope.active_request
        if view.state != "needs_attention" or view.reason != "pause_failed":
            return marker
        now = utc_now()
        resumed_marker = marker.model_copy(update={"updated_at": now})
        self.store.write_marker(resumed_marker, previous=marker)
        self._replace_active_view(
            envelope,
            view.model_copy(update={"state": "pausing", "reason": None}),
        )
        return resumed_marker

    def _require_marker_batch_binding(
        self,
        marker: SharedGpuSwitchMarkerV1,
        active: BatchManifest | None,
    ) -> None:
        if marker.batch_id is None:
            if active is not None:
                raise GpuSwitchStoreCorruptError(
                    "idle switch marker conflicts with an active batch manifest"
                )
            return
        if active is None:
            raise GpuSwitchStoreCorruptError(
                "switch marker references an unavailable active batch manifest"
            )
        if active.batch_id != marker.batch_id or active.owner.user_id != marker.batch_owner_user_id:
            raise GpuSwitchStoreCorruptError("switch marker batch binding changed")

    def _pause_deadline_elapsed(self, marker: SharedGpuSwitchMarkerV1) -> bool:
        started = datetime.fromisoformat(marker.updated_at.replace("Z", "+00:00"))
        return self.coordination.clock.utcnow().astimezone(UTC) >= started + timedelta(
            seconds=GPU_SWITCH_PAUSE_TTL_SECONDS
        )

    def delete_intent(
        self,
        principal: Principal,
        switch_id: str,
        request: DeleteIntentGpuSwitchRequestV1,
    ) -> SharedGpuSwitchMarkerV1:
        self.coordination.require_foreground_session(principal, request.session_id)
        marker = self._require_exact_marker(principal, switch_id, request.finalization_id)
        if marker.phase == "delete_intent":
            return marker
        if marker.phase != "ready_to_delete":
            raise gpu_switch_error("gpu_switch_completion_not_ready")
        next_marker = marker.model_copy(update={"phase": "delete_intent", "updated_at": utc_now()})
        self.store.write_marker(next_marker, previous=marker)
        self._set_marker_view_state(next_marker, "delete_intent")
        return next_marker

    def adopt(
        self,
        principal: Principal,
        switch_id: str,
        request: AdoptGpuSwitchRequestV1,
    ) -> SharedGpuSwitchMarkerV1:
        self.coordination.require_foreground_session(principal, request.session_id)
        marker = self._require_exact_marker(principal, switch_id, request.finalization_id)
        if marker.phase == "replacement_ready":
            expected = (
                marker.replacement_attempt_id,
                marker.replacement_attempt_revision,
                marker.replacement_pod_id,
                marker.actual_target_gpu_id,
                marker.create_marker_sha256,
                marker.create_intent_sha256,
                marker.create_wire_body_sha256,
            )
            supplied = (
                request.replacement_attempt_id,
                request.replacement_attempt_revision,
                request.replacement_pod_id,
                request.target_gpu_id,
                request.create_marker_sha256,
                request.create_intent_sha256,
                request.create_wire_body_sha256,
            )
            if expected != supplied:
                raise gpu_switch_error("gpu_switch_adoption_mismatch")
            return marker
        if marker.phase != "delete_intent":
            raise gpu_switch_error("gpu_switch_adoption_mismatch")
        self._verify_replacement_runtime(marker, request)
        next_marker = marker.model_copy(
            update={
                "phase": "replacement_ready",
                "replacement_attempt_id": request.replacement_attempt_id,
                "replacement_attempt_revision": request.replacement_attempt_revision,
                "replacement_pod_id": request.replacement_pod_id,
                "actual_target_gpu_id": request.target_gpu_id,
                "create_marker_sha256": request.create_marker_sha256,
                "create_intent_sha256": request.create_intent_sha256,
                "create_wire_body_sha256": request.create_wire_body_sha256,
                "updated_at": utc_now(),
            }
        )
        self.store.write_marker(next_marker, previous=marker)
        self._set_marker_view_state(next_marker, "replacement_ready")
        return next_marker

    def complete(
        self,
        principal: Principal,
        switch_id: str,
        request: CompleteGpuSwitchRequestV1,
    ) -> SharedGpuSwitchTombstoneV1:
        self.coordination.require_foreground_session(principal, request.session_id)
        envelope = self._safe_envelope(switch_id)
        if envelope is None or envelope.requester_user_id != principal.user_id:
            raise gpu_switch_error("gpu_switch_request_not_found")
        if envelope.terminal_tombstone is not None:
            tombstone = envelope.terminal_tombstone
            if tombstone.terminal_state != "completed" or (
                tombstone.finalization_id != request.finalization_id
                or tombstone.replacement_attempt_id != request.replacement_attempt_id
                or tombstone.replacement_attempt_revision != request.replacement_attempt_revision
                or tombstone.replacement_pod_id != request.replacement_pod_id
            ):
                raise gpu_switch_error("gpu_switch_finalization_mismatch")
            return tombstone
        marker = self._require_exact_marker(principal, switch_id, request.finalization_id)
        if marker.phase != "replacement_ready" or (
            marker.replacement_attempt_id != request.replacement_attempt_id
            or marker.replacement_attempt_revision != request.replacement_attempt_revision
            or marker.replacement_pod_id != request.replacement_pod_id
        ):
            raise gpu_switch_error("gpu_switch_completion_not_ready")
        tombstone = SharedGpuSwitchTombstoneV1(
            schema_version=1,
            switch_id=switch_id,
            principal_binding_id=marker.principal_binding_id,
            requester_user_id=marker.requester_user_id,
            finalization_id=marker.finalization_id,
            terminal_state="completed",
            terminal_reason="replacement_completed",
            replacement_attempt_id=marker.replacement_attempt_id,
            replacement_attempt_revision=marker.replacement_attempt_revision,
            replacement_pod_id=marker.replacement_pod_id,
            actual_target_gpu_id=marker.actual_target_gpu_id,
            terminal_at=utc_now(),
        )
        self.store.terminalize(envelope, tombstone)
        self.store.archive_marker_and_clear(marker, tombstone)
        self._decisions.pop(switch_id, None)
        return tombstone

    def cancel(
        self,
        principal: Principal,
        switch_id: str,
        request: CancelGpuSwitchRequestV1,
        active: BatchManifest | None,
    ) -> SharedGpuSwitchTombstoneV1 | None:
        self.coordination.require_foreground_session(principal, request.session_id)
        marker = self._safe_marker()
        if marker is None:
            envelope = self._require_owner_envelope(
                principal, switch_id, request.session_id, private=False
            )
        else:
            envelope = self._safe_envelope(switch_id)
            if envelope is None or envelope.requester_user_id != principal.user_id:
                raise gpu_switch_error("gpu_switch_request_not_found")
        if envelope.terminal_tombstone is not None:
            return envelope.terminal_tombstone
        if marker is None:
            if request.finalization_id is not None:
                raise gpu_switch_error("gpu_switch_finalization_mismatch")
            return self._terminalize(envelope, "requester_cancelled").terminal_tombstone
        if marker.switch_id != switch_id or request.finalization_id != marker.finalization_id:
            raise gpu_switch_error("gpu_switch_finalization_mismatch")
        if marker.phase in {"delete_intent", "replacement_ready"}:
            raise gpu_switch_error("gpu_switch_cancel_not_allowed")
        view = envelope.active_request
        if view is None:
            raise GpuSwitchStoreCorruptError("marker request is already terminal")
        if active is not None and (
            active.state == BatchState.RUNNING
            or any(
                image.status in {ImageState.GENERATING, ImageState.RETRYING}
                for image in active.images
            )
        ):
            self._replace_active_view(
                envelope,
                view.model_copy(update={"state": "pausing", "reason": "requester_cancelled"}),
            )
            return None
        tombstone = self._tombstone_for(
            envelope,
            "requester_cancelled",
            utc_now(),
            finalization_id=marker.finalization_id,
        )
        self.store.terminalize(envelope, tombstone)
        self.store.archive_marker_and_clear(marker, tombstone)
        self._decisions.pop(switch_id, None)
        return tombstone

    def cancel_for_generation(self, reason: str, *, queue_mode: bool) -> None:
        marker = self._safe_marker()
        if marker is not None:
            raise gpu_switch_error("gpu_switch_pending")
        envelope = self._active_envelope(None)
        if envelope is None or envelope.active_request is None:
            return
        if queue_mode:
            raise gpu_switch_error("queue_switch_pending")
        if envelope.active_request.state in {"pending", "approved"}:
            self._terminalize(envelope, reason)

    def block_stop(self) -> None:
        marker = self._safe_marker()
        if marker is not None:
            raise gpu_switch_error("gpu_switch_pending")
        envelope = self._active_envelope(None)
        if envelope is not None and envelope.active_request is not None:
            if envelope.active_request.state in {"pending", "approved"}:
                raise gpu_switch_error("gpu_switch_request_in_progress")

    def runtime_identity(
        self, principal: Principal, switch_id: str, session_id: str
    ) -> WorkerGpuSwitchRuntimeIdentityV1:
        self.coordination.require_foreground_session(principal, session_id)
        envelope = self._require_owner_envelope(principal, switch_id, session_id, private=True)
        marker = self._safe_marker()
        if marker is None or marker.switch_id != switch_id or marker.phase != "replacement_ready":
            raise gpu_switch_error("gpu_switch_runtime_identity_unavailable")
        observation = self.runtime_inspector.inspect()
        self._verify_device_mapping(marker.actual_target_gpu_id, observation.device)
        runtime_pod_id = self.runtime_metadata.get("RUNPOD_POD_ID")
        runtime_volume_id = self.runtime_metadata.get("RUNPOD_VOLUME_ID")
        image_digest = self.runtime_metadata.get("IMAGEFORGE_IMAGE_DIGEST")
        if runtime_pod_id is None or runtime_volume_id is None or image_digest is None:
            raise gpu_switch_error("gpu_switch_runtime_identity_unavailable")
        binding = hashlib.sha256(
            (
                "imageforge-data-root-binding-v1\n"
                f"{runtime_volume_id}\n{self.data_root.resolve(strict=False)}"
            ).encode()
        ).hexdigest()
        return WorkerGpuSwitchRuntimeIdentityV1(
            schema_version=1,
            switch_id=switch_id,
            principal_binding_id=envelope.principal_binding_id,
            server_instance_id=self.coordination.server_instance_id,
            runtime_pod_id=runtime_pod_id,
            runtime_volume_id=runtime_volume_id,
            runtime_data_center_id="EU-RO-1",
            data_root_binding_sha256=binding,
            expected_provider_gpu_id=marker.actual_target_gpu_id,
            device_count=observation.device_count,
            cuda_device=observation.device,
            image_digest=image_digest,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            create_contract_revision=1,
            create_marker_sha256=marker.create_marker_sha256,
            replacement_attempt_id=marker.replacement_attempt_id,
            replacement_attempt_revision=marker.replacement_attempt_revision,
        )

    def permission_block(
        self,
        principal: Principal,
        active: BatchManifest | None,
        stop_guard: SharedGpuStopGuard | None,
    ) -> GpuSwitchBlockCode | None:
        try:
            self.store.initialize()
            marker = self.store.read_marker()
            if marker is not None:
                self._require_marker_batch_binding(marker, active)
        except GpuControlGuardConflictError:
            return "gpu_control_guard_conflict"
        except GpuSwitchStoreCorruptError:
            return "gpu_switch_store_corrupt"
        if marker is not None and stop_guard is not None:
            return "gpu_control_guard_conflict"
        if stop_guard is not None:
            return "gpu_stop_pending"
        if marker is not None:
            return "gpu_switch_pending"
        stop = self.coordination.stop_request
        if stop is not None and stop.state in {"pending", "approved"}:
            return "stop_request_in_progress"
        if stop is not None and stop.state == "finalizing":
            return "gpu_stop_pending"
        envelope = self._active_envelope(None)
        if envelope is not None:
            return "gpu_switch_request_in_progress"
        if not self._runtime_preflight_available():
            return "runtime_identity_unavailable"
        if (
            active is not None
            and active.owner.user_id != principal.user_id
            and not (self.coordination.principal_has_foreground_session(active.owner.user_id))
        ):
            return "foreign_batch_owner_unavailable"
        if not self.coordination.principal_has_foreground_session(principal.user_id):
            return "requester_not_foreground"
        return None

    def validate_marker_batch_binding(self, active: BatchManifest | None) -> None:
        """Fail closed if the durable guard no longer names the active manifest."""

        marker = self._safe_marker()
        if marker is not None:
            self._require_marker_batch_binding(marker, active)

    def requires_takeover_reconciliation(self, active: BatchManifest | None) -> bool:
        """Identify state that is authoritative but not owned by this process epoch."""

        if active is not None or self._safe_marker() is not None:
            return True
        return any(
            envelope.active_request is not None and envelope.switch_id not in self._decisions
            for envelope in self._safe_envelopes()
        )

    def _safe_envelopes(self) -> list[SharedGpuSwitchRequestEnvelopeV1]:
        try:
            return self.store.list_envelopes()
        except GpuControlGuardConflictError:
            raise gpu_switch_error("gpu_control_guard_conflict") from None
        except GpuSwitchStoreCorruptError:
            raise gpu_switch_error("gpu_switch_store_corrupt") from None

    def _safe_envelope(self, switch_id: str) -> SharedGpuSwitchRequestEnvelopeV1 | None:
        try:
            return self.store.read_envelope(switch_id)
        except GpuSwitchStoreCorruptError:
            raise gpu_switch_error("gpu_switch_store_corrupt") from None

    def _safe_marker(self) -> SharedGpuSwitchMarkerV1 | None:
        try:
            return self.store.read_marker()
        except GpuSwitchStoreCorruptError:
            raise gpu_switch_error("gpu_switch_store_corrupt") from None

    def _active_envelope(self, switch_id: str | None) -> SharedGpuSwitchRequestEnvelopeV1 | None:
        matches = [
            item
            for item in self._safe_envelopes()
            if item.active_request is not None
            and (switch_id is None or item.switch_id == switch_id)
        ]
        if len(matches) > 1:
            raise gpu_switch_error("gpu_control_guard_conflict")
        return matches[0] if matches else None

    def _raise_store_or_guard(self, stop_guard: SharedGpuStopGuard | None) -> None:
        try:
            self.store.initialize()
            marker = self.store.read_marker()
        except GpuControlGuardConflictError:
            raise gpu_switch_error("gpu_control_guard_conflict") from None
        except GpuSwitchStoreCorruptError:
            raise gpu_switch_error("gpu_switch_store_corrupt") from None
        if marker is not None and stop_guard is not None:
            raise gpu_switch_error("gpu_control_guard_conflict")
        if stop_guard is not None:
            raise gpu_switch_error("gpu_stop_pending")
        if marker is not None:
            raise gpu_switch_error("gpu_switch_pending")

    def _validate_runtime_preflight(self, request: CreateGpuSwitchRequestV1) -> None:
        if not self._runtime_preflight_available():
            raise gpu_switch_error("gpu_switch_runtime_identity_unavailable")
        if self.runtime_metadata.get("RUNPOD_POD_ID") != request.old_pod_id:
            raise gpu_switch_error("gpu_switch_current_pod_unverified")
        expected_gpu = self.runtime_metadata.get("IMAGEFORGE_EXPECTED_GPU_TYPE_ID")
        if expected_gpu != request.old_gpu_id:
            raise gpu_switch_error("gpu_switch_current_pod_unverified")

    def _runtime_preflight_available(self) -> bool:
        required = {
            "RUNPOD_POD_ID",
            "RUNPOD_VOLUME_ID",
            "RUNPOD_DC_ID",
            "RUNPOD_GPU_COUNT",
            "IMAGEFORGE_IMAGE_DIGEST",
            "IMAGEFORGE_EXPECTED_GPU_TYPE_ID",
        }
        if any(not self.runtime_metadata.get(key) for key in required):
            return False
        try:
            require_pod_id(self.runtime_metadata["RUNPOD_POD_ID"])
            require_gpu_identity(self.runtime_metadata["IMAGEFORGE_EXPECTED_GPU_TYPE_ID"])
            require_image_digest(self.runtime_metadata["IMAGEFORGE_IMAGE_DIGEST"])
        except ValueError:
            return False
        return (
            self.runtime_metadata["RUNPOD_DC_ID"] == "EU-RO-1"
            and self.runtime_metadata["RUNPOD_GPU_COUNT"] == "1"
        )

    def _validate_runtime_marker_inputs(self, view: GpuSwitchRequestViewV1) -> None:
        request = CreateGpuSwitchRequestV1(
            schema_version=1,
            switch_id=view.switch_id,
            session_id=view.requester.session_id,
            old_pod_id=view.old_pod_id,
            old_gpu_id=view.old_gpu_id,
            old_gpu_display_name=view.old_gpu_display_name,
            initial_target_gpu_id=view.initial_target_gpu_id,
            initial_target_gpu_display_name=view.initial_target_gpu_display_name,
            initial_replacement_attempt_id=view.initial_replacement_attempt_id,
            expected_batch_id=view.batch_id,
            inventory_observed_at=view.requested_at,
        )
        self._validate_runtime_preflight(request)

    def _verify_replacement_runtime(
        self, marker: SharedGpuSwitchMarkerV1, request: AdoptGpuSwitchRequestV1
    ) -> None:
        exact = {
            "RUNPOD_POD_ID": request.replacement_pod_id,
            "RUNPOD_VOLUME_ID": marker.expected_volume_id,
            "RUNPOD_DC_ID": marker.expected_data_center_id,
            "RUNPOD_GPU_COUNT": "1",
            "IMAGEFORGE_IMAGE_DIGEST": marker.expected_image_digest,
            "IMAGEFORGE_EXPECTED_GPU_TYPE_ID": request.target_gpu_id,
            "IMAGEFORGE_GPU_SWITCH_ID": marker.switch_id,
            "IMAGEFORGE_REPLACEMENT_ATTEMPT_ID": request.replacement_attempt_id,
            "IMAGEFORGE_REPLACEMENT_ATTEMPT_REVISION": str(request.replacement_attempt_revision),
            "IMAGEFORGE_CREATE_CONTRACT_REVISION": "1",
            "IMAGEFORGE_CREATE_MARKER_SHA256": request.create_marker_sha256,
        }
        if any(self.runtime_metadata.get(key) != value for key, value in exact.items()):
            raise gpu_switch_error("gpu_switch_adoption_mismatch")
        observation = self.runtime_inspector.inspect()
        self._verify_device_mapping(request.target_gpu_id, observation.device)

    def _verify_device_mapping(
        self, expected_gpu_id: str | None, device: WorkerCudaDeviceIdentityV1
    ) -> None:
        if expected_gpu_id is None:
            raise gpu_switch_error("gpu_switch_runtime_identity_unavailable")
        try:
            contract = load_runtime_identity_contract()
            record = next(
                item for item in contract.identities if item.providerGpuId == expected_gpu_id
            )
            if device.cudaName not in record.cudaNames:
                raise ValueError("CUDA name mismatch")
            if device.pciDeviceId not in record.pciDeviceIds:
                raise ValueError("PCI identity mismatch")
            if device.totalMemoryBytes < record.minimumMemoryBytes:
                raise ValueError("GPU memory mismatch")
            minimum = record.minimumComputeCapability
            if (device.computeCapabilityMajor, device.computeCapabilityMinor) < (
                minimum.major,
                minimum.minor,
            ):
                raise ValueError("compute capability mismatch")
        except Exception:
            raise gpu_switch_error("gpu_switch_runtime_identity_unavailable") from None

    def _require_owner_envelope(
        self,
        principal: Principal,
        switch_id: str,
        session_id: str,
        *,
        private: bool,
    ) -> SharedGpuSwitchRequestEnvelopeV1:
        envelope = self._safe_envelope(switch_id)
        if envelope is None or envelope.requester_user_id != principal.user_id:
            raise gpu_switch_error("gpu_switch_request_not_found")
        marker = self._safe_marker()
        durable = marker is not None and marker.switch_id == switch_id
        terminal = envelope.terminal_tombstone is not None
        if not (durable or terminal) and envelope.requester_session_id != session_id:
            raise gpu_switch_error("gpu_switch_request_not_found")
        if not private and durable and envelope.requester_session_id != session_id:
            # Public old-session mutation routes never gain recovery authority.
            raise gpu_switch_error("gpu_switch_request_not_found")
        return envelope

    def _require_exact_marker(
        self, principal: Principal, switch_id: str, finalization_id: str
    ) -> SharedGpuSwitchMarkerV1:
        marker = self._safe_marker()
        if (
            marker is None
            or marker.switch_id != switch_id
            or (
                marker.requester_user_id != principal.user_id
                or marker.finalization_id != finalization_id
            )
        ):
            raise gpu_switch_error("gpu_switch_finalization_mismatch")
        return marker

    def _replace_active_view(
        self,
        envelope: SharedGpuSwitchRequestEnvelopeV1,
        view: GpuSwitchRequestViewV1,
    ) -> SharedGpuSwitchRequestEnvelopeV1:
        next_envelope = envelope.model_copy(
            update={
                "envelope_revision": envelope.envelope_revision + 1,
                "active_request": view,
                "updated_at": utc_now(),
            }
        )
        self.store.write_envelope(next_envelope, previous=envelope)
        return next_envelope

    def _terminalize(
        self, envelope: SharedGpuSwitchRequestEnvelopeV1, reason: str
    ) -> SharedGpuSwitchRequestEnvelopeV1:
        tombstone = self._tombstone_for(envelope, reason, utc_now())
        self._decisions.pop(envelope.switch_id, None)
        return self.store.terminalize(envelope, tombstone)

    @staticmethod
    def _tombstone_for(
        envelope: SharedGpuSwitchRequestEnvelopeV1,
        reason: str,
        terminal_at: str,
        *,
        finalization_id: str | None = None,
    ) -> SharedGpuSwitchTombstoneV1:
        view = envelope.active_request
        if view is None:
            assert envelope.terminal_tombstone is not None
            return envelope.terminal_tombstone
        return SharedGpuSwitchTombstoneV1(
            schema_version=1,
            switch_id=envelope.switch_id,
            principal_binding_id=envelope.principal_binding_id,
            requester_user_id=envelope.requester_user_id,
            finalization_id=finalization_id,
            terminal_state=_terminal_state(reason),
            terminal_reason=reason,
            replacement_attempt_id=view.replacement_attempt_id,
            replacement_attempt_revision=view.replacement_attempt_revision,
            replacement_pod_id=view.replacement_pod_id,
            actual_target_gpu_id=view.actual_target_gpu_id,
            terminal_at=terminal_at,
        )

    def _decision_view(
        self,
        view: GpuSwitchRequestViewV1,
        decision: _DecisionState,
        active: BatchManifest | None,
    ) -> GpuSwitchRequestViewV1:
        foreground = self.coordination.foreground_principals()
        waiting = {
            user_id: GpuSwitchParticipantV1(
                session_id=foreground[user_id].session_id,
                display_name=foreground[user_id].display_name,
            )
            for user_id in decision.participants
            if user_id in foreground and user_id not in decision.approvals
        }
        if (
            active is not None
            and active.owner.user_id != (self._safe_envelope(view.switch_id).requester_user_id)
            and active.owner.user_id not in foreground
        ):
            raise gpu_switch_error("switch_owner_unavailable")
        return view.model_copy(
            update={
                "state": "approved" if not waiting else "pending",
                "waiting_for": self._sorted_participants(waiting.values()),
                "approved_by": self._sorted_participants(
                    decision.participants[user_id]
                    for user_id in decision.approvals
                    if user_id in decision.participants
                ),
                "denied_by": self._sorted_participants(
                    decision.participants[user_id]
                    for user_id in decision.denied
                    if user_id in decision.participants
                ),
            }
        )

    def _set_marker_view_state(self, marker: SharedGpuSwitchMarkerV1, state: str) -> None:
        envelope = self._safe_envelope(marker.switch_id)
        if envelope is None or envelope.active_request is None:
            raise GpuSwitchStoreCorruptError("marker lost its request envelope")
        view = self._view_from_marker(envelope.active_request, marker).model_copy(
            update={"state": state}
        )
        self._replace_active_view(envelope, view)

    @staticmethod
    def _view_from_marker(
        view: GpuSwitchRequestViewV1, marker: SharedGpuSwitchMarkerV1
    ) -> GpuSwitchRequestViewV1:
        _require_marker_view_binding(marker, view)
        payload = view.model_dump(mode="python")
        payload.update(
            {
                "state": (view.state if view.state == "needs_attention" else marker.phase),
                "batch_state_at_finalization": marker.batch_state_at_finalization,
                "replacement_attempt_id": marker.replacement_attempt_id,
                "replacement_attempt_revision": marker.replacement_attempt_revision,
                "replacement_pod_id": marker.replacement_pod_id,
                "actual_target_gpu_id": marker.actual_target_gpu_id,
                "ready_to_delete_at": (
                    view.ready_to_delete_at
                    if view.ready_to_delete_at is not None
                    else marker.updated_at
                    if marker.phase
                    in {
                        "ready_to_delete",
                        "delete_intent",
                        "replacement_ready",
                    }
                    else None
                ),
            }
        )
        return GpuSwitchRequestViewV1.model_validate(payload)

    @staticmethod
    def _lookup(envelope: SharedGpuSwitchRequestEnvelopeV1) -> GpuSwitchLookupResponseV1:
        if envelope.active_request is not None:
            view = envelope.active_request
            return GpuSwitchLookupResponseV1(
                schema_version=1,
                switch_id=envelope.switch_id,
                state=view.state,
                replacement_attempt_id=view.replacement_attempt_id,
                replacement_attempt_revision=view.replacement_attempt_revision,
                replacement_pod_id=view.replacement_pod_id,
                actual_target_gpu_id=view.actual_target_gpu_id,
            )
        tombstone = envelope.terminal_tombstone
        assert tombstone is not None
        return GpuSwitchLookupResponseV1(
            schema_version=1,
            switch_id=envelope.switch_id,
            state=tombstone.terminal_state,
            replacement_attempt_id=tombstone.replacement_attempt_id,
            replacement_attempt_revision=tombstone.replacement_attempt_revision,
            replacement_pod_id=tombstone.replacement_pod_id,
            actual_target_gpu_id=tombstone.actual_target_gpu_id,
        )

    @staticmethod
    def _sorted_participants(
        values,
    ) -> list[GpuSwitchParticipantV1]:  # type: ignore[no-untyped-def]
        return sorted(values, key=lambda item: (item.display_name, item.session_id))
