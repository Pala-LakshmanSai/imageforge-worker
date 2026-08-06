from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

from pydantic import Field, field_validator

from .auth import Principal
from .constants import API_SCHEMA_VERSION
from .domain import BatchManifest, BatchSummary, StrictModel
from .errors import WorkerError
from .gpu_switch_models import GpuSwitchRequestViewV1, require_gpu_identity

PRESENCE_TTL_SECONDS = 15
STOP_RESPONSE_TTL_SECONDS = 30
FINALIZATION_TTL_SECONDS = 60
MAX_STUDIO_SESSIONS = 16
MAX_PRINCIPAL_SESSIONS = 8
MAX_SAFE_REVISION = 9_007_199_254_740_991

UUID4_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
POD_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,56}[A-Za-z0-9])?$")

Availability = Literal["foreground", "background"]
StopDecision = Literal["approve", "deny"]
StopState = Literal["pending", "approved", "denied", "expired", "cancelled", "finalizing"]
StopReason = Literal[
    "peer_denied",
    "response_timeout",
    "requester_cancelled",
    "requester_expired",
    "generation_started",
    "finalization_expired",
]


def _require_uuid4(value: str) -> str:
    if UUID4_PATTERN.fullmatch(value) is None:
        raise ValueError("identifier must be a canonical UUIDv4")
    parsed = uuid.UUID(value)
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("identifier must be a canonical UUIDv4")
    return value


class HeartbeatRequest(StrictModel):
    availability: Availability


class CreateStopRequest(StrictModel):
    request_id: str
    session_id: str
    pod_id: str = Field(min_length=1, max_length=58)
    gpu_display_name: str = Field(min_length=1, max_length=128)

    @field_validator("request_id", "session_id")
    @classmethod
    def validate_ids(cls, value: str) -> str:
        return _require_uuid4(value)

    @field_validator("pod_id")
    @classmethod
    def validate_pod_id(cls, value: str) -> str:
        if POD_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("pod_id is invalid")
        return value

    @field_validator("gpu_display_name")
    @classmethod
    def validate_gpu_display_name(cls, value: str) -> str:
        return require_gpu_identity(value)


class StopResponseRequest(StrictModel):
    session_id: str
    decision: StopDecision

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return _require_uuid4(value)


class FinalizeStopRequest(StrictModel):
    session_id: str
    finalization_id: str

    @field_validator("session_id", "finalization_id")
    @classmethod
    def validate_ids(cls, value: str) -> str:
        return _require_uuid4(value)


class CancelStopRequest(StrictModel):
    session_id: str
    finalization_id: str | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return _require_uuid4(value)

    @field_validator("finalization_id")
    @classmethod
    def validate_finalization_id(cls, value: str | None) -> str | None:
        return None if value is None else _require_uuid4(value)


class CoordinationIdentity(StrictModel):
    session_id: str
    display_name: str = Field(min_length=1, max_length=80)


class StudioSessionView(CoordinationIdentity):
    availability: Availability
    expires_at: str


class StopRequestView(StrictModel):
    request_id: str
    pod_id: str
    gpu_display_name: str
    requester: CoordinationIdentity
    state: StopState
    reason: StopReason | None
    requested_at: str
    response_deadline: str
    finalization_expires_at: str | None
    waiting_for: list[CoordinationIdentity] = Field(max_length=MAX_STUDIO_SESSIONS)
    approved_by: list[CoordinationIdentity] = Field(max_length=MAX_STUDIO_SESSIONS)
    denied_by: list[CoordinationIdentity] = Field(max_length=MAX_STUDIO_SESSIONS)
    finalization_id: str | None


class StudioStateResponse(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION] = API_SCHEMA_VERSION
    server_instance_id: str
    coordination_revision: int = Field(ge=0, le=MAX_SAFE_REVISION)
    server_time: str
    presence_ttl_seconds: int = Field(ge=1, le=300)
    response_ttl_seconds: int = Field(ge=1, le=300)
    finalization_ttl_seconds: int = Field(ge=1, le=300)
    current_session: StudioSessionView
    sessions: list[StudioSessionView] = Field(max_length=MAX_STUDIO_SESSIONS)
    active_batch: BatchSummary | None
    stop_request: StopRequestView | None
    gpu_switch_request: GpuSwitchRequestViewV1 | None = None
    gpu_switch_can_respond: bool = False


class CoordinationClock(Protocol):
    def monotonic(self) -> float: ...

    def utcnow(self) -> datetime: ...


class SystemCoordinationClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def utcnow(self) -> datetime:
        return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(slots=True)
class _Session:
    session_id: str
    user_id: str
    display_name: str
    availability: Availability
    expires_monotonic: float
    expires_at: datetime


@dataclass(slots=True)
class _StopRequest:
    request_id: str
    pod_id: str
    gpu_display_name: str
    requester_session_id: str
    requester_user_id: str
    requester_display_name: str
    state: StopState
    reason: StopReason | None
    requested_at: datetime
    response_deadline_monotonic: float
    response_deadline: datetime
    participants: dict[str, CoordinationIdentity] = field(default_factory=dict)
    approvals: set[str] = field(default_factory=set)
    denied: set[str] = field(default_factory=set)
    finalization_id: str | None = None
    finalization_deadline_monotonic: float | None = None
    finalization_expires_at: datetime | None = None


class StudioCoordinator:
    """Ephemeral, authenticated studio presence and coordinated Stop state.

    GenerationController serializes every method under its app-level admission
    lock. Nothing here starts or stops compute; a successful finalization only
    creates the short guard that the desktop must hold while it performs the
    exact-Pod RunPod DELETE.
    """

    def __init__(
        self,
        *,
        clock: CoordinationClock | None = None,
        presence_ttl_seconds: int = PRESENCE_TTL_SECONDS,
        response_ttl_seconds: int = STOP_RESPONSE_TTL_SECONDS,
        finalization_ttl_seconds: int = FINALIZATION_TTL_SECONDS,
    ) -> None:
        for value in (presence_ttl_seconds, response_ttl_seconds, finalization_ttl_seconds):
            if not 1 <= value <= 300:
                raise ValueError("coordination TTLs must be between 1 and 300 seconds")
        self.clock = clock or SystemCoordinationClock()
        self.presence_ttl_seconds = presence_ttl_seconds
        self.response_ttl_seconds = response_ttl_seconds
        self.finalization_ttl_seconds = finalization_ttl_seconds
        self.server_instance_id = str(uuid.uuid4())
        self.revision = 0
        self.sessions: dict[str, _Session] = {}
        self.stop_request: _StopRequest | None = None

    def heartbeat(
        self,
        principal: Principal,
        session_id: str,
        request: HeartbeatRequest,
        active: BatchManifest | None,
    ) -> StudioStateResponse:
        _require_uuid4(session_id)
        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        existing = self.sessions.get(session_id)
        if existing is not None and existing.user_id != principal.user_id:
            raise self._session_not_found()
        if existing is None:
            principal_count = sum(
                session.user_id == principal.user_id for session in self.sessions.values()
            )
            if (
                len(self.sessions) >= MAX_STUDIO_SESSIONS
                or principal_count >= MAX_PRINCIPAL_SESSIONS
            ):
                raise WorkerError(
                    status_code=429,
                    code="studio_session_limit",
                    message="Too many ImageForge studio sessions are active.",
                )
        self.sessions[session_id] = _Session(
            session_id=session_id,
            user_id=principal.user_id,
            display_name=principal.display_name,
            availability=request.availability,
            expires_monotonic=now_mono + self.presence_ttl_seconds,
            expires_at=now_utc + timedelta(seconds=self.presence_ttl_seconds),
        )
        self._touch()
        self._maintain(now_mono, now_utc)
        return self._state(principal, session_id, active, now_utc)

    def state(
        self,
        principal: Principal,
        session_id: str,
        active: BatchManifest | None,
    ) -> StudioStateResponse:
        _require_uuid4(session_id)
        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        self._require_session(principal, session_id)
        return self._state(principal, session_id, active, now_utc)

    def require_foreground_session(
        self, principal: Principal, session_id: str
    ) -> CoordinationIdentity:
        """Return one authenticated live foreground identity for Switch.

        GenerationController already serializes this call with heartbeat and
        every coordination mutation, so the returned snapshot cannot race a
        concurrent presence update inside this process.
        """

        _require_uuid4(session_id)
        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        session = self._require_session(principal, session_id)
        if session.availability != "foreground":
            raise WorkerError(
                status_code=423,
                code="gpu_switch_requester_not_foreground",
                message="The GPU switch requester must remain foreground.",
            )
        return self._identity(session)

    def foreground_principals(
        self,
    ) -> dict[str, CoordinationIdentity]:
        """Deduplicate foreground sessions by principal for Switch consent."""

        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        return {
            user_id: self._identity(session)
            for user_id, session in self._foreground_representatives().items()
        }

    def principal_has_foreground_session(self, user_id: str) -> bool:
        return user_id in self.foreground_principals()

    def create_stop_request(
        self,
        principal: Principal,
        request: CreateStopRequest,
        active: BatchManifest | None,
    ) -> StudioStateResponse:
        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        self._require_session(principal, request.session_id)
        if active is not None:
            self._raise_active_batch_veto(active)
        current = self.stop_request
        if current is not None and current.request_id == request.request_id:
            if (
                current.requester_session_id != request.session_id
                or current.requester_user_id != principal.user_id
                or current.pod_id != request.pod_id
                or current.gpu_display_name != request.gpu_display_name
            ):
                raise WorkerError(
                    status_code=409,
                    code="stop_request_identity_mismatch",
                    message="The Stop request identifier belongs to different request details.",
                )
            return self._state(principal, request.session_id, active, now_utc)
        if current is not None and current.state in {"pending", "approved", "finalizing"}:
            raise WorkerError(
                status_code=409,
                code="stop_request_in_progress",
                message="Another coordinated GPU Stop request is already in progress.",
                details={
                    "request_id": current.request_id,
                    "requester": current.requester_display_name,
                    "state": current.state,
                },
            )
        self.stop_request = _StopRequest(
            request_id=request.request_id,
            pod_id=request.pod_id,
            gpu_display_name=request.gpu_display_name,
            requester_session_id=request.session_id,
            requester_user_id=principal.user_id,
            requester_display_name=principal.display_name,
            state="pending",
            reason=None,
            requested_at=now_utc,
            response_deadline_monotonic=now_mono + self.response_ttl_seconds,
            response_deadline=now_utc + timedelta(seconds=self.response_ttl_seconds),
        )
        self._touch()
        self._maintain(now_mono, now_utc)
        return self._state(principal, request.session_id, active, now_utc)

    def respond(
        self,
        principal: Principal,
        request_id: str,
        request: StopResponseRequest,
        active: BatchManifest | None,
    ) -> StudioStateResponse:
        _require_uuid4(request_id)
        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        session = self._require_session(principal, request.session_id)
        stop = self._require_stop_request(request_id)
        prior: StopDecision | None = None
        if principal.user_id in stop.approvals:
            prior = "approve"
        elif principal.user_id in stop.denied:
            prior = "deny"
        if prior is not None:
            if prior != request.decision:
                raise WorkerError(
                    status_code=409,
                    code="stop_response_conflict",
                    message="This user already sent a different response to the GPU Stop request.",
                )
            return self._state(principal, session.session_id, active, now_utc)
        if stop.state not in {"pending", "approved"}:
            raise WorkerError(
                status_code=409,
                code="stop_response_not_allowed",
                message="This GPU Stop request is no longer accepting responses.",
            )
        if (
            principal.user_id == stop.requester_user_id
            or principal.user_id not in stop.participants
        ):
            raise WorkerError(
                status_code=409,
                code="stop_response_not_allowed",
                message="This session is not a required approver for the GPU Stop request.",
            )
        stop.participants[principal.user_id] = self._identity(session)
        if request.decision == "deny":
            stop.denied.add(principal.user_id)
            stop.state = "denied"
            stop.reason = "peer_denied"
        else:
            stop.approvals.add(principal.user_id)
        self._touch()
        self._maintain(now_mono, now_utc)
        return self._state(principal, session.session_id, active, now_utc)

    def finalize(
        self,
        principal: Principal,
        request_id: str,
        request: FinalizeStopRequest,
        active: BatchManifest | None,
    ) -> StudioStateResponse:
        _require_uuid4(request_id)
        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        self._require_session(principal, request.session_id)
        stop = self._require_requester(principal, request.session_id, request_id)
        if active is not None:
            self._raise_active_batch_veto(active)
        if stop.state == "finalizing":
            if stop.finalization_id != request.finalization_id:
                raise WorkerError(
                    status_code=409,
                    code="finalization_mismatch",
                    message="A different finalization already owns this GPU Stop guard.",
                )
            return self._state(principal, request.session_id, active, now_utc)
        if stop.state == "pending":
            raise WorkerError(
                status_code=409,
                code="stop_approval_pending",
                message="Every active foreground editor must approve before GPU Stop finalization.",
                details={
                    "waiting_for": [item.display_name for item in self._waiting_identities(stop)]
                },
            )
        if stop.state != "approved":
            raise WorkerError(
                status_code=409,
                code="stop_request_not_approved",
                message="The GPU Stop request is not approved for finalization.",
                details={"state": stop.state},
            )
        stop.state = "finalizing"
        stop.finalization_id = request.finalization_id
        stop.finalization_deadline_monotonic = now_mono + self.finalization_ttl_seconds
        stop.finalization_expires_at = now_utc + timedelta(seconds=self.finalization_ttl_seconds)
        self._touch()
        return self._state(principal, request.session_id, active, now_utc)

    def cancel(
        self,
        principal: Principal,
        request_id: str,
        request: CancelStopRequest,
        active: BatchManifest | None,
    ) -> StudioStateResponse:
        _require_uuid4(request_id)
        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        self._require_session(principal, request.session_id)
        stop = self._require_requester(principal, request.session_id, request_id)
        if stop.state in {"denied", "expired", "cancelled"}:
            return self._state(principal, request.session_id, active, now_utc)
        if stop.state == "finalizing" and request.finalization_id != stop.finalization_id:
            raise WorkerError(
                status_code=409,
                code="finalization_mismatch",
                message="The cancellation did not match the active GPU Stop finalization.",
            )
        if stop.state != "finalizing" and request.finalization_id is not None:
            raise WorkerError(
                status_code=409,
                code="finalization_mismatch",
                message="No matching GPU Stop finalization exists.",
            )
        stop.state = "cancelled"
        stop.reason = "requester_cancelled"
        stop.finalization_id = None
        stop.finalization_deadline_monotonic = None
        stop.finalization_expires_at = None
        self._touch()
        return self._state(principal, request.session_id, active, now_utc)

    def admit_generation(self) -> None:
        """Atomically cancel approval-only Stop or reject the final guard."""

        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        stop = self.stop_request
        if stop is None:
            return
        if stop.state == "finalizing":
            assert stop.finalization_expires_at is not None
            raise WorkerError(
                status_code=423,
                code="gpu_stop_pending",
                message="GPU Stop is finalizing; new generation is temporarily blocked.",
                details={
                    "request_id": stop.request_id,
                    "requester": stop.requester_display_name,
                    "expires_at": _timestamp(stop.finalization_expires_at),
                },
            )
        if stop.state in {"pending", "approved"}:
            stop.state = "cancelled"
            stop.reason = "generation_started"
            self._touch()

    def admit_queue_generation(self) -> None:
        """Admit a locally staged successor without cancelling peer Stop consent.

        The caller already holds the same controller lock used by foreground
        admission. This closes the race where a peer requests Stop between a
        desktop preflight and its queue POST: foreground Generate retains the
        historic cancellation rule, while queue admission parks locally.
        """

        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        stop = self.stop_request
        if stop is None:
            return
        if stop.state == "finalizing":
            assert stop.finalization_expires_at is not None
            raise WorkerError(
                status_code=423,
                code="gpu_stop_pending",
                message="GPU Stop is finalizing; new generation is temporarily blocked.",
                details={
                    "request_id": stop.request_id,
                    "requester": stop.requester_display_name,
                    "expires_at": _timestamp(stop.finalization_expires_at),
                },
            )
        if stop.state in {"pending", "approved"}:
            raise WorkerError(
                status_code=423,
                code="queue_stop_pending",
                message="GPU Stop consent is pending; the local queue is paused.",
                details={
                    "request_id": stop.request_id,
                    "requester": stop.requester_display_name,
                    "state": stop.state,
                    "expires_at": _timestamp(stop.response_deadline),
                },
            )

    def rollback_finalization(
        self,
        request_id: str,
        finalization_id: str,
        *,
        generation_started: bool = False,
    ) -> None:
        """Undo an unpublished grant when the shared-volume lease loses the race."""

        _require_uuid4(request_id)
        _require_uuid4(finalization_id)
        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        stop = self.stop_request
        if (
            stop is None
            or stop.request_id != request_id
            or stop.state != "finalizing"
            or stop.finalization_id != finalization_id
        ):
            return
        stop.state = "cancelled" if generation_started else "approved"
        stop.reason = "generation_started" if generation_started else None
        stop.finalization_id = None
        stop.finalization_deadline_monotonic = None
        stop.finalization_expires_at = None
        self._touch()
        self._maintain(now_mono, now_utc)

    def finalization_remaining_seconds(self, request_id: str, finalization_id: str) -> float | None:
        """Maintain expiry and return the remaining lifetime of one exact guard."""

        _require_uuid4(request_id)
        _require_uuid4(finalization_id)
        now_mono, now_utc = self._now()
        self._maintain(now_mono, now_utc)
        stop = self.stop_request
        if (
            stop is None
            or stop.request_id != request_id
            or stop.state != "finalizing"
            or stop.finalization_id != finalization_id
            or stop.finalization_deadline_monotonic is None
        ):
            return None
        return max(0.0, stop.finalization_deadline_monotonic - now_mono)

    def _maintain(self, now_mono: float, now_utc: datetime) -> None:
        changed = False
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if session.expires_monotonic <= now_mono
        ]
        for session_id in expired_sessions:
            del self.sessions[session_id]
            changed = True

        stop = self.stop_request
        if stop is None:
            if changed:
                self._touch()
            return
        if stop.state in {"denied", "expired", "cancelled"}:
            if changed:
                self._touch()
            return
        if stop.state == "finalizing":
            # Once granted, the guard must outlive presence loss. Releasing it
            # when the requester's heartbeat expires could admit generation
            # while an exact-Pod DELETE is still in flight or ambiguous.
            if (
                stop.finalization_deadline_monotonic is not None
                and stop.finalization_deadline_monotonic <= now_mono
            ):
                stop.state = "expired"
                stop.reason = "finalization_expired"
                stop.finalization_id = None
                stop.finalization_deadline_monotonic = None
                stop.finalization_expires_at = None
                changed = True
        elif stop.requester_session_id not in self.sessions:
            stop.state = "cancelled"
            stop.reason = "requester_expired"
            stop.finalization_id = None
            stop.finalization_deadline_monotonic = None
            stop.finalization_expires_at = None
            changed = True
        elif stop.response_deadline_monotonic <= now_mono:
            stop.state = "expired"
            stop.reason = "response_timeout"
            changed = True
        else:
            representatives = self._foreground_representatives(stop.requester_user_id)
            if stop.participants != representatives:
                vanished = set(stop.participants) - set(representatives)
                stop.approvals.difference_update(vanished)
                stop.denied.difference_update(vanished)
                stop.participants = representatives
                changed = True
            next_state: StopState = (
                "approved" if set(stop.participants).issubset(stop.approvals) else "pending"
            )
            if stop.state != next_state:
                stop.state = next_state
                changed = True
        if changed:
            self._touch()

    def _foreground_representatives(
        self, requester_user_id: str | None = None
    ) -> dict[str, CoordinationIdentity]:
        by_user: dict[str, list[_Session]] = {}
        for session in self.sessions.values():
            if (
                requester_user_id is not None and session.user_id == requester_user_id
            ) or session.availability != "foreground":
                continue
            by_user.setdefault(session.user_id, []).append(session)
        return {
            user_id: self._identity(sorted(sessions, key=lambda item: item.session_id)[0])
            for user_id, sessions in by_user.items()
        }

    def _state(
        self,
        principal: Principal,
        session_id: str,
        active: BatchManifest | None,
        now_utc: datetime,
    ) -> StudioStateResponse:
        current = self._require_session(principal, session_id)
        sessions = [
            self._session_view(item)
            for item in sorted(self.sessions.values(), key=lambda item: item.session_id)
        ]
        stop_view = self._stop_view(self.stop_request, principal, session_id)
        return StudioStateResponse(
            server_instance_id=self.server_instance_id,
            coordination_revision=self.revision,
            server_time=_timestamp(now_utc),
            presence_ttl_seconds=self.presence_ttl_seconds,
            response_ttl_seconds=self.response_ttl_seconds,
            finalization_ttl_seconds=self.finalization_ttl_seconds,
            current_session=self._session_view(current),
            sessions=sessions,
            active_batch=self._batch_summary(active),
            stop_request=stop_view,
        )

    def _stop_view(
        self,
        stop: _StopRequest | None,
        principal: Principal,
        session_id: str,
    ) -> StopRequestView | None:
        if stop is None:
            return None
        finalization_id = (
            stop.finalization_id
            if principal.user_id == stop.requester_user_id
            and session_id == stop.requester_session_id
            else None
        )
        return StopRequestView(
            request_id=stop.request_id,
            pod_id=stop.pod_id,
            gpu_display_name=stop.gpu_display_name,
            requester=CoordinationIdentity(
                session_id=stop.requester_session_id,
                display_name=stop.requester_display_name,
            ),
            state=stop.state,
            reason=stop.reason,
            requested_at=_timestamp(stop.requested_at),
            response_deadline=_timestamp(stop.response_deadline),
            finalization_expires_at=(
                None
                if stop.finalization_expires_at is None
                else _timestamp(stop.finalization_expires_at)
            ),
            waiting_for=self._waiting_identities(stop),
            approved_by=self._decision_identities(stop, stop.approvals),
            denied_by=self._decision_identities(stop, stop.denied),
            finalization_id=finalization_id,
        )

    def _waiting_identities(self, stop: _StopRequest) -> list[CoordinationIdentity]:
        waiting_principals = set(stop.participants) - stop.approvals - stop.denied
        return [
            self._identity(session)
            for session in sorted(self.sessions.values(), key=lambda item: item.session_id)
            if session.user_id in waiting_principals
            and session.user_id != stop.requester_user_id
            and session.availability == "foreground"
        ]

    @staticmethod
    def _decision_identities(stop: _StopRequest, user_ids: set[str]) -> list[CoordinationIdentity]:
        return [
            stop.participants[user_id]
            for user_id in sorted(user_ids)
            if user_id in stop.participants
        ]

    @staticmethod
    def _session_view(session: _Session) -> StudioSessionView:
        return StudioSessionView(
            session_id=session.session_id,
            display_name=session.display_name,
            availability=session.availability,
            expires_at=_timestamp(session.expires_at),
        )

    @staticmethod
    def _identity(session: _Session) -> CoordinationIdentity:
        return CoordinationIdentity(
            session_id=session.session_id, display_name=session.display_name
        )

    @staticmethod
    def _batch_summary(active: BatchManifest | None) -> BatchSummary | None:
        if active is None:
            return None
        return BatchSummary(
            batch_id=active.batch_id,
            owner=active.owner,
            state=active.state,
            progress=active.progress,
            pause_requested=active.pause_requested,
            cancel_requested=active.cancel_requested,
        )

    def _require_session(self, principal: Principal, session_id: str) -> _Session:
        session = self.sessions.get(session_id)
        if session is None or session.user_id != principal.user_id:
            raise self._session_not_found()
        return session

    def _require_stop_request(self, request_id: str) -> _StopRequest:
        stop = self.stop_request
        if stop is None or stop.request_id != request_id:
            raise WorkerError(
                status_code=404,
                code="stop_request_not_found",
                message="The coordinated GPU Stop request does not exist on this worker instance.",
            )
        return stop

    def _require_requester(
        self, principal: Principal, session_id: str, request_id: str
    ) -> _StopRequest:
        stop = self._require_stop_request(request_id)
        if stop.requester_user_id != principal.user_id or stop.requester_session_id != session_id:
            raise WorkerError(
                status_code=404,
                code="stop_request_not_found",
                message="The coordinated GPU Stop request does not exist for this session.",
            )
        return stop

    @staticmethod
    def _session_not_found() -> WorkerError:
        return WorkerError(
            status_code=404,
            code="studio_session_not_found",
            message="Send a fresh authenticated studio heartbeat before this operation.",
        )

    @staticmethod
    def _raise_active_batch_veto(active: BatchManifest) -> None:
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

    def _now(self) -> tuple[float, datetime]:
        return self.clock.monotonic(), self.clock.utcnow().astimezone(UTC)

    def _touch(self) -> None:
        if self.revision >= MAX_SAFE_REVISION:
            # A new epoch is safer than emitting a non-monotonic JSON integer.
            self.server_instance_id = str(uuid.uuid4())
            self.sessions.clear()
            self.stop_request = None
            self.revision = 0
            return
        self.revision += 1
