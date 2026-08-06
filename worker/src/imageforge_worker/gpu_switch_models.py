from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .constants import API_SCHEMA_VERSION, MODEL_ID, MODEL_REVISION
from .domain import StrictModel, require_canonical_uuid4

MAX_SAFE_REVISION = 9_007_199_254_740_991
GPU_IDENTITY_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9 ._()+:-]{0,126}[A-Za-z0-9])?$")
POD_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,56}[A-Za-z0-9])?$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
NVML_UUID_PATTERN = re.compile(r"^GPU-[0-9A-Fa-f-]{36}$")
PCI_DEVICE_ID_PATTERN = re.compile(r"^0x[0-9a-f]{4}$")
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
RFC3339_MILLISECONDS_PATTERN = re.compile(
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\.[0-9]{3}Z$"
)

GpuSwitchWorkerState = Literal[
    "pending",
    "approved",
    "denied",
    "expired",
    "cancelled",
    "pausing",
    "ready_to_delete",
    "delete_intent",
    "replacement_ready",
    "completed",
    "needs_attention",
]
GpuSwitchReason = Literal[
    "peer_denied",
    "response_timeout",
    "requester_cancelled",
    "requester_expired",
    "generation_started",
    "batch_changed",
    "stop_started",
    "target_changed_pre_delete",
    "pause_failed",
    "replacement_mismatch",
    "completion_failed",
]
GpuSwitchBlockCode = Literal[
    "requester_not_foreground",
    "runtime_identity_unavailable",
    "current_pod_unverified",
    "local_receipts_pending",
    "queue_dispatch_uncertain",
    "foreign_batch_owner_unavailable",
    "stop_request_in_progress",
    "gpu_stop_pending",
    "gpu_switch_request_in_progress",
    "gpu_switch_pending",
    "gpu_control_guard_conflict",
    "gpu_switch_store_corrupt",
]


def require_gpu_identity(value: str) -> str:
    if len(value.encode("utf-8")) > 128 or GPU_IDENTITY_PATTERN.fullmatch(value) is None:
        raise ValueError("GPU identity is invalid")
    return value


def require_pod_id(value: str) -> str:
    if POD_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("Pod identifier is invalid")
    return value


def require_timestamp(value: str) -> str:
    if RFC3339_MILLISECONDS_PATTERN.fullmatch(value) is None:
        raise ValueError("timestamp must use RFC3339 UTC milliseconds")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp is invalid") from exc
    if (
        parsed.tzinfo is None
        or parsed.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z") != value
    ):
        raise ValueError("timestamp must be canonical UTC milliseconds")
    return value


def require_sha256(value: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("SHA-256 value is invalid")
    return value


def require_image_digest(value: str) -> str:
    if "@sha256:" not in value:
        raise ValueError("image identity must be an immutable digest")
    repository, digest = value.rsplit("@sha256:", 1)
    if (
        not repository
        or repository.lower() != repository
        or re.fullmatch(r"[a-z0-9][a-z0-9._/:\-]*[a-z0-9]", repository) is None
        or SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError("image identity must be a lowercase registry digest")
    return value


def require_user_id(value: str) -> str:
    if USER_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("authenticated user identity is invalid")
    return value


class CreateGpuSwitchRequestV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    switch_id: str
    session_id: str
    old_pod_id: str
    old_gpu_id: str
    old_gpu_display_name: str
    initial_target_gpu_id: str
    initial_target_gpu_display_name: str
    initial_replacement_attempt_id: str
    expected_batch_id: str | None
    inventory_observed_at: str

    @field_validator("switch_id", "session_id", "initial_replacement_attempt_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        return require_canonical_uuid4(value)

    @field_validator("expected_batch_id")
    @classmethod
    def validate_batch_id(cls, value: str | None) -> str | None:
        return None if value is None else require_canonical_uuid4(value)

    @field_validator("old_pod_id")
    @classmethod
    def validate_pod(cls, value: str) -> str:
        return require_pod_id(value)

    @field_validator(
        "old_gpu_id",
        "old_gpu_display_name",
        "initial_target_gpu_id",
        "initial_target_gpu_display_name",
    )
    @classmethod
    def validate_gpu(cls, value: str) -> str:
        return require_gpu_identity(value)

    @field_validator("inventory_observed_at")
    @classmethod
    def validate_inventory_time(cls, value: str) -> str:
        return require_timestamp(value)

    @model_validator(mode="after")
    def validate_different_target(self) -> CreateGpuSwitchRequestV1:
        if self.old_gpu_id == self.initial_target_gpu_id:
            raise ValueError("switch target must differ from the current GPU")
        return self


class GpuSwitchResponseRequestV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    session_id: str
    decision: Literal["approve", "deny"]

    @field_validator("session_id")
    @classmethod
    def validate_session(cls, value: str) -> str:
        return require_canonical_uuid4(value)


class FinalizeGpuSwitchRequestV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    session_id: str
    finalization_id: str

    @field_validator("session_id", "finalization_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        return require_canonical_uuid4(value)


class DeleteIntentGpuSwitchRequestV1(FinalizeGpuSwitchRequestV1):
    pass


class AdoptGpuSwitchRequestV1(FinalizeGpuSwitchRequestV1):
    replacement_attempt_id: str
    replacement_attempt_revision: int = Field(ge=1, le=MAX_SAFE_REVISION)
    replacement_pod_id: str
    target_gpu_id: str
    create_contract_revision: Literal[1]
    create_marker_sha256: str
    create_intent_sha256: str
    create_wire_body_sha256: str

    @field_validator("replacement_attempt_id")
    @classmethod
    def validate_attempt_id(cls, value: str) -> str:
        return require_canonical_uuid4(value)

    @field_validator("replacement_pod_id")
    @classmethod
    def validate_replacement_pod(cls, value: str) -> str:
        return require_pod_id(value)

    @field_validator("target_gpu_id")
    @classmethod
    def validate_target(cls, value: str) -> str:
        return require_gpu_identity(value)

    @field_validator("create_marker_sha256", "create_intent_sha256", "create_wire_body_sha256")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        return require_sha256(value)


class CompleteGpuSwitchRequestV1(FinalizeGpuSwitchRequestV1):
    replacement_attempt_id: str
    replacement_attempt_revision: int = Field(ge=1, le=MAX_SAFE_REVISION)
    replacement_pod_id: str

    @field_validator("replacement_attempt_id")
    @classmethod
    def validate_attempt_id(cls, value: str) -> str:
        return require_canonical_uuid4(value)

    @field_validator("replacement_pod_id")
    @classmethod
    def validate_replacement_pod(cls, value: str) -> str:
        return require_pod_id(value)


class CancelGpuSwitchRequestV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    session_id: str
    finalization_id: str | None

    @field_validator("session_id")
    @classmethod
    def validate_session(cls, value: str) -> str:
        return require_canonical_uuid4(value)

    @field_validator("finalization_id")
    @classmethod
    def validate_finalization(cls, value: str | None) -> str | None:
        return None if value is None else require_canonical_uuid4(value)


class SettleGpuSwitchCreateRequestV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    action: Literal["cancel"]
    create_request: CreateGpuSwitchRequestV1


class GpuSwitchParticipantV1(StrictModel):
    session_id: str
    display_name: str = Field(min_length=1, max_length=80)

    @field_validator("session_id")
    @classmethod
    def validate_session(cls, value: str) -> str:
        return require_canonical_uuid4(value)


class GpuSwitchBatchOwnerV1(StrictModel):
    display_name: str = Field(min_length=1, max_length=80)


class GpuSwitchRequestViewV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    switch_id: str
    old_pod_id: str
    old_gpu_id: str
    old_gpu_display_name: str
    initial_target_gpu_id: str
    initial_target_gpu_display_name: str
    initial_replacement_attempt_id: str
    requester: GpuSwitchParticipantV1
    state: GpuSwitchWorkerState
    reason: GpuSwitchReason | None
    requested_at: str
    response_deadline: str
    ready_to_delete_at: str | None
    waiting_for: list[GpuSwitchParticipantV1] = Field(max_length=16)
    approved_by: list[GpuSwitchParticipantV1] = Field(max_length=16)
    denied_by: list[GpuSwitchParticipantV1] = Field(max_length=16)
    batch_id: str | None
    batch_owner: GpuSwitchBatchOwnerV1 | None
    batch_state_at_finalization: Literal["running", "paused", "interrupted"] | None
    replacement_attempt_id: str | None
    replacement_attempt_revision: int | None = Field(default=None, ge=1, le=MAX_SAFE_REVISION)
    replacement_pod_id: str | None
    actual_target_gpu_id: str | None

    @field_validator("switch_id", "initial_replacement_attempt_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        return require_canonical_uuid4(value)

    @field_validator("batch_id", "replacement_attempt_id")
    @classmethod
    def validate_optional_uuid(cls, value: str | None) -> str | None:
        return None if value is None else require_canonical_uuid4(value)

    @field_validator("old_pod_id", "replacement_pod_id")
    @classmethod
    def validate_optional_pod(cls, value: str | None) -> str | None:
        return None if value is None else require_pod_id(value)

    @field_validator(
        "old_gpu_id",
        "old_gpu_display_name",
        "initial_target_gpu_id",
        "initial_target_gpu_display_name",
        "actual_target_gpu_id",
    )
    @classmethod
    def validate_optional_gpu(cls, value: str | None) -> str | None:
        return None if value is None else require_gpu_identity(value)

    @field_validator("requested_at", "response_deadline", "ready_to_delete_at")
    @classmethod
    def validate_times(cls, value: str | None) -> str | None:
        return None if value is None else require_timestamp(value)

    @model_validator(mode="after")
    def validate_participants_and_replacement(self) -> GpuSwitchRequestViewV1:
        for collection in (self.waiting_for, self.approved_by, self.denied_by):
            if collection != sorted(
                collection, key=lambda item: (item.display_name, item.session_id)
            ):
                raise ValueError("participant arrays must be canonically sorted")
            if len({item.session_id for item in collection}) != len(collection):
                raise ValueError("participant arrays cannot repeat a session")
        replacement_values = (
            self.replacement_attempt_id,
            self.replacement_attempt_revision,
            self.replacement_pod_id,
            self.actual_target_gpu_id,
        )
        if any(value is None for value in replacement_values) != all(
            value is None for value in replacement_values
        ):
            raise ValueError("replacement identity fields must be all null or all populated")
        if (self.batch_id is None) != (self.batch_owner is None):
            raise ValueError("batch lookup identity must be both null or both populated")
        if self.batch_id is None and self.batch_state_at_finalization is not None:
            raise ValueError("batch finalization state requires a batch identity")
        if self.state in {"denied", "expired", "cancelled", "completed"}:
            raise ValueError("terminal Switch state requires a tombstone, not an active view")

        finalized = self.state in {
            "pausing",
            "ready_to_delete",
            "delete_intent",
            "replacement_ready",
            "needs_attention",
        }
        if not finalized and self.batch_state_at_finalization is not None:
            raise ValueError("pre-finalization view cannot carry a batch finalization state")
        if finalized and self.batch_id is not None and self.batch_state_at_finalization is None:
            raise ValueError("batch-bound finalized view requires its finalization state")

        expected_reasons: dict[str, set[str | None]] = {
            "pending": {None},
            "approved": {None},
            "pausing": {None, "requester_cancelled"},
            "ready_to_delete": {None},
            "delete_intent": {None},
            "replacement_ready": {None},
            # Worker attention is deliberately closed to the one authored
            # fixed-point timeout. Provider/native attention codes never enter
            # this worker request envelope.
            "needs_attention": {"pause_failed"},
        }
        if self.reason not in expected_reasons[self.state]:
            raise ValueError("Switch request state and reason are incompatible")

        fixed_point = self.state in {
            "ready_to_delete",
            "delete_intent",
            "replacement_ready",
        }
        if fixed_point != (self.ready_to_delete_at is not None):
            raise ValueError("Switch pause fixed point timestamp is incompatible with state")

        has_replacement = all(value is not None for value in replacement_values)
        if has_replacement != (self.state == "replacement_ready"):
            raise ValueError("replacement identity is allowed only in replacement-ready state")

        if self.state == "needs_attention":
            if (
                self.batch_id is None
                or self.batch_owner is None
                or self.batch_state_at_finalization != "running"
            ):
                raise ValueError(
                    "pause-failed attention requires the bound batch finalization identity"
                )
        if self.state == "pausing" and self.reason == "requester_cancelled":
            if self.batch_id is None or self.batch_state_at_finalization != "running":
                raise ValueError(
                    "in-flight cancellation requires the running batch finalization identity"
                )
        return self


class GpuSwitchLookupResponseV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    switch_id: str
    state: GpuSwitchWorkerState
    replacement_attempt_id: str | None
    replacement_attempt_revision: int | None = Field(default=None, ge=1, le=MAX_SAFE_REVISION)
    replacement_pod_id: str | None
    actual_target_gpu_id: str | None

    @field_validator("switch_id", "replacement_attempt_id")
    @classmethod
    def validate_optional_uuid(cls, value: str | None) -> str | None:
        return None if value is None else require_canonical_uuid4(value)

    @field_validator("replacement_pod_id")
    @classmethod
    def validate_pod(cls, value: str | None) -> str | None:
        return None if value is None else require_pod_id(value)

    @field_validator("actual_target_gpu_id")
    @classmethod
    def validate_gpu(cls, value: str | None) -> str | None:
        return None if value is None else require_gpu_identity(value)

    @model_validator(mode="after")
    def validate_state_identity(self) -> GpuSwitchLookupResponseV1:
        replacement = (
            self.replacement_attempt_id,
            self.replacement_attempt_revision,
            self.replacement_pod_id,
            self.actual_target_gpu_id,
        )
        if any(value is None for value in replacement) != all(
            value is None for value in replacement
        ):
            raise ValueError("replacement lookup identity must be all null or all populated")
        if self.state in {"replacement_ready", "completed"} and not all(
            value is not None for value in replacement
        ):
            raise ValueError("replacement-ready and completed lookups require replacement identity")
        if self.state in {"denied", "expired", "cancelled"} and any(
            value is not None for value in replacement
        ):
            raise ValueError("pre-delete terminal lookups cannot carry replacement identity")
        return self


class NativeWorkerGpuSwitchCreateResponseV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    request: GpuSwitchRequestViewV1
    requester_user_id: str = Field(min_length=1, max_length=64)
    principal_binding_id: str

    @field_validator("requester_user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        return require_user_id(value)

    @field_validator("principal_binding_id")
    @classmethod
    def validate_binding(cls, value: str) -> str:
        return require_canonical_uuid4(value)


class NativeWorkerGpuSwitchOwnerLookupV1(GpuSwitchLookupResponseV1):
    requester_user_id: str = Field(min_length=1, max_length=64)
    principal_binding_id: str
    finalization_id: str | None
    terminal_tombstone_sha256: str | None

    @field_validator("requester_user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        return require_user_id(value)

    @field_validator("principal_binding_id")
    @classmethod
    def validate_binding(cls, value: str) -> str:
        return require_canonical_uuid4(value)

    @field_validator("finalization_id")
    @classmethod
    def validate_finalization(cls, value: str | None) -> str | None:
        return None if value is None else require_canonical_uuid4(value)

    @field_validator("terminal_tombstone_sha256")
    @classmethod
    def validate_tombstone_hash(cls, value: str | None) -> str | None:
        return None if value is None else require_sha256(value)

    @model_validator(mode="after")
    def validate_private_state(self) -> NativeWorkerGpuSwitchOwnerLookupV1:
        terminal = self.state in {"denied", "expired", "cancelled", "completed"}
        if terminal != (self.terminal_tombstone_sha256 is not None):
            raise ValueError("terminal owner lookup must carry exactly one tombstone hash")
        if self.state in {"pending", "approved", "denied", "expired", "cancelled"} and (
            self.finalization_id is not None
        ):
            raise ValueError("pre-finalization owner lookup cannot carry finalization identity")
        if (
            self.state
            in {
                "pausing",
                "ready_to_delete",
                "delete_intent",
                "replacement_ready",
                "completed",
            }
            and self.finalization_id is None
        ):
            raise ValueError("durably finalized owner lookup requires finalization identity")
        return self


class SharedGpuSwitchTombstoneV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    switch_id: str
    principal_binding_id: str
    requester_user_id: str = Field(min_length=1, max_length=64)
    finalization_id: str | None
    terminal_state: Literal["completed", "cancelled", "denied", "expired"]
    terminal_reason: Literal[
        "replacement_completed",
        "requester_cancelled",
        "peer_denied",
        "response_timeout",
        "requester_expired",
        "generation_started",
        "batch_changed",
        "stop_started",
        "target_changed_pre_delete",
    ]
    replacement_attempt_id: str | None
    replacement_attempt_revision: int | None = Field(default=None, ge=1, le=MAX_SAFE_REVISION)
    replacement_pod_id: str | None
    actual_target_gpu_id: str | None
    terminal_at: str

    @field_validator("requester_user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        return require_user_id(value)

    @field_validator("switch_id", "principal_binding_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        return require_canonical_uuid4(value)

    @field_validator("finalization_id", "replacement_attempt_id")
    @classmethod
    def validate_optional_uuid(cls, value: str | None) -> str | None:
        return None if value is None else require_canonical_uuid4(value)

    @field_validator("replacement_pod_id")
    @classmethod
    def validate_optional_pod(cls, value: str | None) -> str | None:
        return None if value is None else require_pod_id(value)

    @field_validator("actual_target_gpu_id")
    @classmethod
    def validate_optional_gpu(cls, value: str | None) -> str | None:
        return None if value is None else require_gpu_identity(value)

    @field_validator("terminal_at")
    @classmethod
    def validate_terminal_time(cls, value: str) -> str:
        return require_timestamp(value)

    @model_validator(mode="after")
    def validate_terminal_identity(self) -> SharedGpuSwitchTombstoneV1:
        replacement = (
            self.replacement_attempt_id,
            self.replacement_attempt_revision,
            self.replacement_pod_id,
            self.actual_target_gpu_id,
        )
        if any(value is None for value in replacement) != all(
            value is None for value in replacement
        ):
            raise ValueError("terminal replacement identity must be all null or all populated")
        if self.terminal_state == "completed":
            if self.terminal_reason != "replacement_completed" or self.finalization_id is None:
                raise ValueError("completed tombstone requires replacement completion identity")
            if not all(value is not None for value in replacement):
                raise ValueError("completed tombstone requires replacement identity")
        elif any(value is not None for value in replacement):
            raise ValueError("pre-delete terminal tombstone cannot carry replacement identity")
        if self.terminal_state in {"denied", "expired"} and self.finalization_id is not None:
            raise ValueError(
                "pre-finalization terminal tombstone cannot carry finalization identity"
            )
        return self


class SharedGpuSwitchRequestEnvelopeV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    envelope_revision: int = Field(ge=1, le=MAX_SAFE_REVISION)
    switch_id: str
    request_fingerprint_sha256: str
    requester_user_id: str = Field(min_length=1, max_length=64)
    requester_session_id: str
    principal_binding_id: str
    active_request: GpuSwitchRequestViewV1 | None
    terminal_tombstone: SharedGpuSwitchTombstoneV1 | None
    created_at: str
    updated_at: str

    @field_validator("requester_user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        return require_user_id(value)

    @field_validator("switch_id", "requester_session_id", "principal_binding_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        return require_canonical_uuid4(value)

    @field_validator("request_fingerprint_sha256")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return require_sha256(value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_times(cls, value: str) -> str:
        return require_timestamp(value)

    @model_validator(mode="after")
    def validate_payload(self) -> SharedGpuSwitchRequestEnvelopeV1:
        if (self.active_request is None) == (self.terminal_tombstone is None):
            raise ValueError("exactly one request-envelope payload must be present")
        payload_switch_id = (
            self.active_request.switch_id
            if self.active_request is not None
            else self.terminal_tombstone.switch_id
        )
        if payload_switch_id != self.switch_id:
            raise ValueError("request-envelope switch identity mismatch")
        if self.active_request is not None:
            if self.active_request.state in {"denied", "expired", "cancelled", "completed"}:
                raise ValueError("terminal request state requires a tombstone payload")
            if self.active_request.requester.session_id != self.requester_session_id:
                raise ValueError("request-envelope requester session mismatch")
        if self.terminal_tombstone is not None and (
            self.terminal_tombstone.principal_binding_id != self.principal_binding_id
            or self.terminal_tombstone.requester_user_id != self.requester_user_id
        ):
            raise ValueError("request-envelope tombstone identity mismatch")
        if self.terminal_tombstone is not None:
            expected_state = {
                "peer_denied": "denied",
                "response_timeout": "expired",
                "requester_expired": "expired",
                "replacement_completed": "completed",
            }.get(self.terminal_tombstone.terminal_reason, "cancelled")
            if self.terminal_tombstone.terminal_state != expected_state:
                raise ValueError("request-envelope tombstone state and reason disagree")
        if self.updated_at < self.created_at:
            raise ValueError("request-envelope timestamps are out of order")
        return self


class SharedGpuSwitchMarkerV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    switch_id: str
    finalization_id: str
    principal_binding_id: str
    requester_user_id: str = Field(min_length=1, max_length=64)
    requester_display_name: str = Field(min_length=1, max_length=80)
    old_pod_id: str
    old_gpu_id: str
    initial_target_gpu_id: str
    initial_replacement_attempt_id: str
    batch_id: str | None
    batch_owner_user_id: str | None
    batch_state_at_finalization: Literal["running", "paused", "interrupted"] | None
    phase: Literal["pausing", "ready_to_delete", "delete_intent", "replacement_ready"]
    replacement_attempt_id: str | None
    replacement_attempt_revision: int | None = Field(default=None, ge=1, le=MAX_SAFE_REVISION)
    replacement_pod_id: str | None
    actual_target_gpu_id: str | None
    create_contract_revision: Literal[1]
    create_marker_sha256: str | None
    create_intent_sha256: str | None
    create_wire_body_sha256: str | None
    expected_volume_id: str = Field(min_length=1, max_length=128)
    expected_data_center_id: Literal["EU-RO-1"]
    expected_image_digest: str
    expected_model_id: Literal[MODEL_ID]
    expected_model_revision: Literal[MODEL_REVISION]
    requested_at: str
    updated_at: str

    @field_validator("requester_user_id", "batch_owner_user_id")
    @classmethod
    def validate_optional_user_id(cls, value: str | None) -> str | None:
        return None if value is None else require_user_id(value)

    @field_validator(
        "switch_id", "finalization_id", "principal_binding_id", "initial_replacement_attempt_id"
    )
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        return require_canonical_uuid4(value)

    @field_validator("batch_id", "replacement_attempt_id")
    @classmethod
    def validate_optional_uuid(cls, value: str | None) -> str | None:
        return None if value is None else require_canonical_uuid4(value)

    @field_validator("old_pod_id", "replacement_pod_id")
    @classmethod
    def validate_optional_pod(cls, value: str | None) -> str | None:
        return None if value is None else require_pod_id(value)

    @field_validator("old_gpu_id", "initial_target_gpu_id", "actual_target_gpu_id")
    @classmethod
    def validate_optional_gpu(cls, value: str | None) -> str | None:
        return None if value is None else require_gpu_identity(value)

    @field_validator("create_marker_sha256", "create_intent_sha256", "create_wire_body_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return None if value is None else require_sha256(value)

    @field_validator("expected_image_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return require_image_digest(value)

    @field_validator("requested_at", "updated_at")
    @classmethod
    def validate_times(cls, value: str) -> str:
        return require_timestamp(value)

    @model_validator(mode="after")
    def validate_bindings(self) -> SharedGpuSwitchMarkerV1:
        if (self.batch_id is None) != (self.batch_owner_user_id is None):
            raise ValueError("batch identity fields must both be null or populated")
        if self.batch_id is None and self.batch_state_at_finalization is not None:
            raise ValueError("batch state requires a batch identity")
        replacement_values = (
            self.replacement_attempt_id,
            self.replacement_attempt_revision,
            self.replacement_pod_id,
            self.actual_target_gpu_id,
            self.create_marker_sha256,
            self.create_intent_sha256,
            self.create_wire_body_sha256,
        )
        populated = [value is not None for value in replacement_values]
        if self.phase == "replacement_ready":
            if not all(populated):
                raise ValueError("replacement-ready marker requires exact replacement identity")
        elif any(populated):
            raise ValueError("replacement identity is allowed only after adoption")
        if self.updated_at < self.requested_at:
            raise ValueError("marker timestamps are out of order")
        return self


class GpuRuntimeMinimumComputeCapabilityV1(StrictModel):
    major: int = Field(ge=1, le=99)
    minor: int = Field(ge=0, le=99)


class GpuRuntimeIdentityRecordV1(StrictModel):
    providerGpuId: str
    cudaNames: list[str] = Field(min_length=1, max_length=16)
    pciDeviceIds: list[str] = Field(min_length=1, max_length=16)
    minimumMemoryBytes: int = Field(ge=1, le=MAX_SAFE_REVISION)
    minimumComputeCapability: GpuRuntimeMinimumComputeCapabilityV1

    @field_validator("providerGpuId")
    @classmethod
    def validate_provider_gpu(cls, value: str) -> str:
        return require_gpu_identity(value)

    @field_validator("cudaNames")
    @classmethod
    def validate_cuda_names(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("CUDA identity names must be unique")
        return [require_gpu_identity(value) for value in values]

    @field_validator("pciDeviceIds")
    @classmethod
    def validate_pci_ids(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values) or any(
            PCI_DEVICE_ID_PATTERN.fullmatch(value) is None for value in values
        ):
            raise ValueError("PCI device identities must be unique lowercase values")
        return values


class GpuRuntimeIdentityContractV1(StrictModel):
    schemaVersion: Literal[1]
    identities: list[GpuRuntimeIdentityRecordV1] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_unique_provider_ids(self) -> GpuRuntimeIdentityContractV1:
        provider_ids = [item.providerGpuId for item in self.identities]
        if len(set(provider_ids)) != len(provider_ids):
            raise ValueError("runtime identity provider GPU IDs must be unique")
        return self


NativeGpuSwitchBlockedPhaseV1 = Literal[
    "planned",
    "consent_pending",
    "pausing",
    "ready_to_delete",
    "delete_intent",
    "delete_uncertain",
    "old_absent",
    "create_intent",
    "create_uncertain",
    "replacement_identified",
    "provisioning",
    "replacement_failed",
    "replacement_delete_intent",
    "replacement_delete_uncertain",
    "ready_paused",
]


class GpuSwitchCodeEntryV1(StrictModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    scope: Literal["native_issue", "native_attention", "worker_block", "worker_action"]
    retryable: bool
    httpStatus: int | None = Field(ge=400, le=599)
    permittedBlockedPhases: list[NativeGpuSwitchBlockedPhaseV1] = Field(max_length=15)

    @model_validator(mode="after")
    def validate_scope_fields(self) -> GpuSwitchCodeEntryV1:
        if self.scope == "worker_action":
            if self.httpStatus is None or self.permittedBlockedPhases:
                raise ValueError("worker actions require HTTP status and no native blocked phases")
        elif self.scope == "native_attention":
            if self.httpStatus is not None or not self.permittedBlockedPhases:
                raise ValueError("native attention requires one or more blocked phases")
        elif self.httpStatus is not None or self.permittedBlockedPhases:
            raise ValueError("non-action/non-attention code has incompatible metadata")
        return self


class GpuSwitchBlockActionMappingV1(StrictModel):
    blockCode: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    actionCode: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")


class GpuSwitchCodeContractV1(StrictModel):
    schemaVersion: Literal[1]
    precedence: list[str] = Field(min_length=1, max_length=32)
    codes: list[GpuSwitchCodeEntryV1] = Field(min_length=1, max_length=256)
    blockActionMappings: list[GpuSwitchBlockActionMappingV1] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_registry_relations(self) -> GpuSwitchCodeContractV1:
        keyed = [(entry.scope, entry.code) for entry in self.codes]
        if len(set(keyed)) != len(keyed):
            raise ValueError("GPU switch registry scope/code rows must be unique")
        blocks = {entry.code for entry in self.codes if entry.scope == "worker_block"}
        actions = {entry.code for entry in self.codes if entry.scope == "worker_action"}
        issues = {entry.code for entry in self.codes if entry.scope == "native_issue"}
        attention = {entry.code for entry in self.codes if entry.scope == "native_attention"}
        if not attention.issubset(issues):
            raise ValueError("every native attention code must also be a native issue")
        if len(set(self.precedence)) != len(self.precedence) or set(self.precedence) != blocks:
            raise ValueError("worker block precedence must contain every block exactly once")
        mapping_blocks = [mapping.blockCode for mapping in self.blockActionMappings]
        if len(set(mapping_blocks)) != len(mapping_blocks) or set(mapping_blocks) != blocks:
            raise ValueError("every worker block requires one action mapping")
        if any(mapping.actionCode not in actions for mapping in self.blockActionMappings):
            raise ValueError("worker block mapping references an unknown action")
        return self


class WorkerCudaDeviceIdentityV1(StrictModel):
    deviceIndex: Literal[0]
    nvmlUuid: str = Field(pattern=r"^GPU-[0-9A-Fa-f-]{36}$")
    pciDeviceId: str = Field(pattern=r"^0x[0-9a-f]{4}$")
    cudaName: str
    totalMemoryBytes: int = Field(ge=1, le=MAX_SAFE_REVISION)
    computeCapabilityMajor: int = Field(ge=1, le=99)
    computeCapabilityMinor: int = Field(ge=0, le=99)

    @field_validator("cudaName")
    @classmethod
    def validate_cuda_name(cls, value: str) -> str:
        return require_gpu_identity(value)


class WorkerGpuSwitchRuntimeIdentityV1(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION]
    switch_id: str
    principal_binding_id: str
    server_instance_id: str
    runtime_pod_id: str
    runtime_volume_id: str = Field(min_length=1, max_length=128)
    runtime_data_center_id: Literal["EU-RO-1"]
    data_root_binding_sha256: str
    expected_provider_gpu_id: str
    device_count: Literal[1]
    cuda_device: WorkerCudaDeviceIdentityV1
    image_digest: str
    model_id: Literal[MODEL_ID]
    model_revision: Literal[MODEL_REVISION]
    create_contract_revision: Literal[1]
    create_marker_sha256: str
    replacement_attempt_id: str
    replacement_attempt_revision: int = Field(ge=1, le=MAX_SAFE_REVISION)

    @field_validator(
        "switch_id", "principal_binding_id", "server_instance_id", "replacement_attempt_id"
    )
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        return require_canonical_uuid4(value)

    @field_validator("runtime_pod_id")
    @classmethod
    def validate_pod(cls, value: str) -> str:
        return require_pod_id(value)

    @field_validator("data_root_binding_sha256", "create_marker_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return require_sha256(value)

    @field_validator("expected_provider_gpu_id")
    @classmethod
    def validate_gpu(cls, value: str) -> str:
        return require_gpu_identity(value)

    @field_validator("image_digest")
    @classmethod
    def validate_image(cls, value: str) -> str:
        return require_image_digest(value)
