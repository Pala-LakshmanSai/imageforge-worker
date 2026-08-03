from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import TOKEN_A, TOKEN_B, auth, wait_for_batch, worker_client

from imageforge_worker.config import Credential
from imageforge_worker.domain import BatchState, utc_now
from imageforge_worker.inference import FakeInferenceAdapter

A_SESSION = "00000000-0000-4000-8000-000000000001"
A_SESSION_2 = "00000000-0000-4000-8000-000000000002"
B_SESSION = "00000000-0000-4000-8000-000000000003"
B_SESSION_2 = "00000000-0000-4000-8000-000000000004"
NEW_SESSION = "00000000-0000-4000-8000-000000000005"
REQUEST_A = "10000000-0000-4000-8000-000000000001"
REQUEST_B = "10000000-0000-4000-8000-000000000002"
FINALIZATION_A = "20000000-0000-4000-8000-000000000001"
FINALIZATION_B = "20000000-0000-4000-8000-000000000002"

STATE_KEYS = {
    "schema_version",
    "server_instance_id",
    "coordination_revision",
    "server_time",
    "presence_ttl_seconds",
    "response_ttl_seconds",
    "finalization_ttl_seconds",
    "current_session",
    "sessions",
    "active_batch",
    "stop_request",
}


class FakeCoordinationClock:
    def __init__(self) -> None:
        self._monotonic = 100.0
        self._utc = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self._monotonic

    def utcnow(self) -> datetime:
        return self._utc

    def advance(self, seconds: float) -> None:
        self._monotonic += seconds
        self._utc += timedelta(seconds=seconds)


class AdjustableSystemClock:
    def __init__(self) -> None:
        self._monotonic = time.monotonic()
        self._utc = datetime.now(UTC)

    def monotonic(self) -> float:
        return self._monotonic

    def utcnow(self) -> datetime:
        return self._utc

    def advance(self, seconds: float) -> None:
        self._monotonic += seconds
        self._utc += timedelta(seconds=seconds)


def _install_clock(app: object) -> FakeCoordinationClock:
    clock = FakeCoordinationClock()
    app.state.runtime.controller.coordination.clock = clock
    return clock


@pytest.mark.parametrize("display_name", [" Lakshman", "Lakshman ", "Lak\nshman", "\x00"])
def test_credential_display_names_are_safe_for_shared_session_projection(
    display_name: str,
) -> None:
    with pytest.raises(ValueError, match="trimmed printable"):
        Credential("lakshman", display_name, TOKEN_A)


async def _heartbeat(
    client,
    session_id: str,
    *,
    token: str = TOKEN_A,
    availability: str = "foreground",
) -> dict:
    response = await client.put(
        f"/v1/studio/sessions/{session_id}",
        headers=auth(token),
        json={"availability": availability},
    )
    assert response.status_code == 200, response.text
    return response.json()


async def _state(client, session_id: str, *, token: str = TOKEN_A) -> dict:
    response = await client.get(
        f"/v1/studio/sessions/{session_id}", headers=auth(token)
    )
    assert response.status_code == 200, response.text
    return response.json()


async def _request_stop(
    client,
    session_id: str,
    *,
    request_id: str = REQUEST_A,
    token: str = TOKEN_A,
) -> object:
    return await client.post(
        "/v1/studio/stop-requests",
        headers=auth(token),
        json={
            "request_id": request_id,
            "session_id": session_id,
            "pod_id": "pod-123",
            "gpu_display_name": "NVIDIA RTX 4090",
        },
    )


async def _finalize(
    client,
    session_id: str,
    *,
    request_id: str = REQUEST_A,
    finalization_id: str = FINALIZATION_A,
    token: str = TOKEN_A,
) -> object:
    return await client.post(
        f"/v1/studio/stop-requests/{request_id}/finalize",
        headers=auth(token),
        json={"session_id": session_id, "finalization_id": finalization_id},
    )


@pytest.mark.anyio
async def test_heartbeat_is_authenticated_strict_safe_and_get_does_not_extend_ttl(
    tmp_path: Path,
) -> None:
    async with worker_client(tmp_path / "volume") as (client, app, _):
        clock = _install_clock(app)
        unauthenticated = await client.put(
            f"/v1/studio/sessions/{A_SESSION}",
            json={"availability": "foreground"},
        )
        assert unauthenticated.status_code == 401
        assert unauthenticated.json()["error"]["code"] == "authentication_required"

        first = await _heartbeat(client, A_SESSION)
        assert set(first) == STATE_KEYS
        assert first["schema_version"] == 1
        assert first["current_session"] == first["sessions"][0]
        assert first["current_session"]["display_name"] == "Lakshman"
        assert first["current_session"]["availability"] == "foreground"
        assert first["active_batch"] is None
        assert first["stop_request"] is None
        assert re.fullmatch(r".*\.\d{3}Z", first["server_time"])
        assert TOKEN_A not in str(first)
        assert TOKEN_B not in str(first)

        foreign = await client.get(
            f"/v1/studio/sessions/{A_SESSION}", headers=auth(TOKEN_B)
        )
        assert foreign.status_code == 404
        assert foreign.json()["error"]["code"] == "studio_session_not_found"

        invalid = await client.put(
            "/v1/studio/sessions/00000000-0000-4000-8000-00000000000A",
            headers=auth(),
            json={"availability": "foreground"},
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "validation_error"
        extra = await client.put(
            f"/v1/studio/sessions/{A_SESSION}",
            headers=auth(),
            json={"availability": "foreground", "credential": TOKEN_A},
        )
        assert extra.status_code == 422
        assert TOKEN_A not in extra.text

        clock.advance(14)
        observed = await _state(client, A_SESSION)
        assert observed["coordination_revision"] == first["coordination_revision"]
        clock.advance(2)
        expired = await client.get(
            f"/v1/studio/sessions/{A_SESSION}", headers=auth()
        )
        assert expired.status_code == 404
        assert expired.json()["error"]["code"] == "studio_session_not_found"


@pytest.mark.anyio
async def test_duplicate_windows_count_once_and_any_session_for_peer_can_decide(
    tmp_path: Path,
) -> None:
    async with worker_client(tmp_path / "volume") as (client, app, _):
        _install_clock(app)
        await _heartbeat(client, A_SESSION)
        await _heartbeat(client, A_SESSION_2)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        await _heartbeat(client, B_SESSION_2, token=TOKEN_B)

        created = await _request_stop(client, A_SESSION_2)
        assert created.status_code == 201
        stop = created.json()["stop_request"]
        assert stop["state"] == "pending"
        assert stop["waiting_for"] == [
            {"session_id": B_SESSION, "display_name": "Sujal"},
            {"session_id": B_SESSION_2, "display_name": "Sujal"},
        ]
        assert all(item["display_name"] != "Lakshman" for item in stop["waiting_for"])
        for peer_session in (B_SESSION, B_SESSION_2):
            peer = await _state(client, peer_session, token=TOKEN_B)
            assert any(
                item["session_id"] == peer_session
                for item in peer["stop_request"]["waiting_for"]
            )

        approved = await client.post(
            f"/v1/studio/stop-requests/{REQUEST_A}/responses",
            headers=auth(TOKEN_B),
            json={"session_id": B_SESSION_2, "decision": "approve"},
        )
        assert approved.status_code == 200
        stop = approved.json()["stop_request"]
        assert stop["state"] == "approved"
        assert stop["approved_by"] == [
            {"session_id": B_SESSION, "display_name": "Sujal"}
        ]

        duplicate = await client.post(
            f"/v1/studio/stop-requests/{REQUEST_A}/responses",
            headers=auth(TOKEN_B),
            json={"session_id": B_SESSION, "decision": "approve"},
        )
        assert duplicate.status_code == 200
        assert duplicate.json()["stop_request"]["state"] == "approved"
        conflict = await client.post(
            f"/v1/studio/stop-requests/{REQUEST_A}/responses",
            headers=auth(TOKEN_B),
            json={"session_id": B_SESSION, "decision": "deny"},
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "stop_response_conflict"

        wrong_requester_window = await _finalize(client, A_SESSION)
        assert wrong_requester_window.status_code == 404
        finalizing = await _finalize(client, A_SESSION_2)
        assert finalizing.status_code == 200
        assert finalizing.json()["stop_request"]["finalization_id"] == FINALIZATION_A
        peer_view = await _state(client, B_SESSION_2, token=TOKEN_B)
        assert peer_view["stop_request"]["finalization_id"] is None


@pytest.mark.anyio
async def test_finalization_guard_blocks_create_until_exact_cancel(tmp_path: Path) -> None:
    async with worker_client(tmp_path / "volume") as (client, app, _):
        _install_clock(app)
        await _heartbeat(client, A_SESSION)
        requested = await _request_stop(client, A_SESSION)
        assert requested.status_code == 201
        assert requested.json()["stop_request"]["state"] == "approved"

        finalizing = await _finalize(client, A_SESSION)
        assert finalizing.status_code == 200
        assert finalizing.json()["stop_request"]["state"] == "finalizing"
        duplicate = await _finalize(client, A_SESSION)
        assert duplicate.status_code == 200
        mismatch = await _finalize(
            client, A_SESSION, finalization_id=FINALIZATION_B
        )
        assert mismatch.status_code == 409
        assert mismatch.json()["error"]["code"] == "finalization_mismatch"

        blocked = await client.post(
            "/v1/batches", headers=auth(), json={"prompts": ["must wait"]}
        )
        assert blocked.status_code == 423
        assert blocked.json()["error"]["code"] == "gpu_stop_pending"
        assert blocked.json()["error"]["details"]["request_id"] == REQUEST_A
        assert blocked.json()["error"]["details"]["requester"] == "Lakshman"
        assert app.state.runtime.store.active_lease_held is True
        assert app.state.runtime.store.read_gpu_stop_guard() is not None

        wrong_cancel = await client.post(
            f"/v1/studio/stop-requests/{REQUEST_A}/cancel",
            headers=auth(),
            json={"session_id": A_SESSION, "finalization_id": FINALIZATION_B},
        )
        assert wrong_cancel.status_code == 409
        assert app.state.runtime.store.active_lease_held is True
        assert app.state.runtime.store.read_gpu_stop_guard() is not None
        cancelled = await client.post(
            f"/v1/studio/stop-requests/{REQUEST_A}/cancel",
            headers=auth(),
            json={"session_id": A_SESSION, "finalization_id": FINALIZATION_A},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["stop_request"]["state"] == "cancelled"
        assert cancelled.json()["stop_request"]["reason"] == "requester_cancelled"

        created = await client.post(
            "/v1/batches", headers=auth(), json={"prompts": ["safe now"]}
        )
        assert created.status_code == 201
        await wait_for_batch(client, created.json()["batch_id"], state="completed")


@pytest.mark.anyio
async def test_generation_cancels_pending_stop_without_queueing(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    adapter = FakeInferenceAdapter(
        first_generation_started=started, release_first_generation=release
    )
    async with worker_client(tmp_path / "volume", adapter) as (client, app, _):
        _install_clock(app)
        await _heartbeat(client, A_SESSION)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        requested = await _request_stop(client, A_SESSION)
        assert requested.status_code == 201
        assert requested.json()["stop_request"]["state"] == "pending"

        try:
            created = await client.post(
                "/v1/batches",
                headers=auth(TOKEN_B),
                json={"prompts": ["new work wins before finalization"]},
            )
            assert created.status_code == 201
            await asyncio.wait_for(started.wait(), timeout=2)
            shared = await _state(client, A_SESSION)
            assert shared["active_batch"]["owner"]["display_name"] == "Sujal"
            assert shared["stop_request"]["state"] == "cancelled"
            assert shared["stop_request"]["reason"] == "generation_started"
        finally:
            release.set()


@pytest.mark.anyio
async def test_active_batch_is_an_unconditional_stop_veto(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    adapter = FakeInferenceAdapter(
        first_generation_started=started, release_first_generation=release
    )
    async with worker_client(tmp_path / "volume", adapter) as (client, app, _):
        _install_clock(app)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        try:
            created = await client.post(
                "/v1/batches",
                headers=auth(),
                json={"prompts": ["still generating", "next"]},
            )
            assert created.status_code == 201
            await asyncio.wait_for(started.wait(), timeout=2)

            vetoed = await _request_stop(client, B_SESSION, token=TOKEN_B)
            assert vetoed.status_code == 423
            error = vetoed.json()["error"]
            assert error["code"] == "stop_blocked_by_active_batch"
            assert error["details"] == {"owner": "Lakshman", "completed": 0, "total": 2}
            state = await _state(client, B_SESSION, token=TOKEN_B)
            assert state["stop_request"] is None
            assert state["active_batch"]["batch_id"] == created.json()["batch_id"]
        finally:
            release.set()


@pytest.mark.anyio
async def test_paused_and_interrupted_batches_are_unconditional_stop_vetoes(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    adapter = FakeInferenceAdapter(
        first_generation_started=started, release_first_generation=release
    )
    async with worker_client(tmp_path / "volume", adapter) as (client, app, _):
        _install_clock(app)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        created = await client.post(
            "/v1/batches",
            headers=auth(),
            json={"prompts": ["finish current image", "leave pending"]},
        )
        assert created.status_code == 201
        batch_id = created.json()["batch_id"]
        await asyncio.wait_for(started.wait(), timeout=2)

        paused = await client.post(f"/v1/batches/{batch_id}/pause", headers=auth())
        assert paused.status_code == 200
        release.set()
        await wait_for_batch(client, batch_id, state="paused")

        paused_veto = await _request_stop(client, B_SESSION, token=TOKEN_B)
        assert paused_veto.status_code == 423
        assert paused_veto.json()["error"]["code"] == "stop_blocked_by_active_batch"

        manifest = app.state.runtime.store.load(batch_id)
        manifest.state = BatchState.INTERRUPTED
        manifest.interrupted_at = utc_now()
        app.state.runtime.store.save(manifest)

        interrupted_veto = await _request_stop(client, B_SESSION, token=TOKEN_B)
        assert interrupted_veto.status_code == 423
        assert interrupted_veto.json()["error"]["code"] == (
            "stop_blocked_by_active_batch"
        )
        await client.post(f"/v1/batches/{batch_id}/cancel", headers=auth())


@pytest.mark.anyio
async def test_dynamic_peers_background_and_expiry_recompute_approval(tmp_path: Path) -> None:
    async with worker_client(tmp_path / "volume") as (client, app, _):
        clock = _install_clock(app)
        await _heartbeat(client, A_SESSION)
        requested = await _request_stop(client, A_SESSION)
        assert requested.json()["stop_request"]["state"] == "approved"

        joined = await _heartbeat(client, B_SESSION, token=TOKEN_B)
        assert joined["stop_request"]["state"] == "pending"
        assert joined["stop_request"]["waiting_for"][0]["display_name"] == "Sujal"
        background = await _heartbeat(
            client, B_SESSION, token=TOKEN_B, availability="background"
        )
        assert background["stop_request"]["state"] == "approved"
        foreground = await _heartbeat(client, B_SESSION, token=TOKEN_B)
        assert foreground["stop_request"]["state"] == "pending"

        clock.advance(10)
        refreshed_requester = await _heartbeat(client, A_SESSION)
        assert refreshed_requester["stop_request"]["state"] == "pending"
        clock.advance(6)
        expired_peer = await _state(client, A_SESSION)
        assert expired_peer["stop_request"]["state"] == "approved"
        assert expired_peer["stop_request"]["waiting_for"] == []
        assert [item["display_name"] for item in expired_peer["sessions"]] == ["Lakshman"]


@pytest.mark.anyio
async def test_denial_timeout_and_requester_expiry_are_safe_terminal_states(
    tmp_path: Path,
) -> None:
    async with worker_client(tmp_path / "denial") as (client, app, _):
        _install_clock(app)
        await _heartbeat(client, A_SESSION)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        await _request_stop(client, A_SESSION)
        denied = await client.post(
            f"/v1/studio/stop-requests/{REQUEST_A}/responses",
            headers=auth(TOKEN_B),
            json={"session_id": B_SESSION, "decision": "deny"},
        )
        assert denied.status_code == 200
        assert denied.json()["stop_request"]["state"] == "denied"
        assert denied.json()["stop_request"]["reason"] == "peer_denied"
        duplicate = await client.post(
            f"/v1/studio/stop-requests/{REQUEST_A}/responses",
            headers=auth(TOKEN_B),
            json={"session_id": B_SESSION, "decision": "deny"},
        )
        assert duplicate.status_code == 200

    async with worker_client(tmp_path / "timeout") as (client, app, _):
        clock = _install_clock(app)
        await _heartbeat(client, A_SESSION)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        await _request_stop(client, A_SESSION)
        clock.advance(14)
        await _heartbeat(client, A_SESSION)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        clock.advance(14)
        await _heartbeat(client, A_SESSION)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        clock.advance(3)
        timed_out = await _state(client, A_SESSION)
        assert timed_out["stop_request"]["state"] == "expired"
        assert timed_out["stop_request"]["reason"] == "response_timeout"

    async with worker_client(tmp_path / "requester-expiry") as (client, app, _):
        clock = _install_clock(app)
        await _heartbeat(client, A_SESSION)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        await _request_stop(client, A_SESSION)
        clock.advance(10)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        clock.advance(6)
        requester_expired = await _state(client, B_SESSION, token=TOKEN_B)
        assert requester_expired["stop_request"]["state"] == "cancelled"
        assert requester_expired["stop_request"]["reason"] == "requester_expired"


@pytest.mark.anyio
async def test_simultaneous_stop_requests_have_one_winner_and_retries_are_idempotent(
    tmp_path: Path,
) -> None:
    async with worker_client(tmp_path / "volume") as (client, app, _):
        _install_clock(app)
        await _heartbeat(client, A_SESSION)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        first, second = await asyncio.gather(
            _request_stop(client, A_SESSION, request_id=REQUEST_A),
            _request_stop(
                client,
                B_SESSION,
                request_id=REQUEST_B,
                token=TOKEN_B,
            ),
        )
        assert sorted([first.status_code, second.status_code]) == [201, 409]
        winner = first if first.status_code == 201 else second
        loser = second if first.status_code == 201 else first
        assert loser.json()["error"]["code"] == "stop_request_in_progress"

        winner_is_a = winner is first
        winner_session = A_SESSION if winner_is_a else B_SESSION
        winner_token = TOKEN_A if winner_is_a else TOKEN_B
        winner_request = REQUEST_A if winner_is_a else REQUEST_B
        retry = await _request_stop(
            client,
            winner_session,
            request_id=winner_request,
            token=winner_token,
        )
        assert retry.status_code == 201
        assert retry.json()["stop_request"]["request_id"] == winner_request

        mismatched = await client.post(
            "/v1/studio/stop-requests",
            headers=auth(winner_token),
            json={
                "request_id": winner_request,
                "session_id": winner_session,
                "pod_id": "different-pod",
                "gpu_display_name": "NVIDIA RTX 4090",
            },
        )
        assert mismatched.status_code == 409
        assert mismatched.json()["error"]["code"] == "stop_request_identity_mismatch"


@pytest.mark.anyio
async def test_finalize_and_create_race_is_serialized_at_worker_boundary(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    adapter = FakeInferenceAdapter(
        first_generation_started=started, release_first_generation=release
    )
    async with worker_client(tmp_path / "volume", adapter) as (client, app, _):
        _install_clock(app)
        await _heartbeat(client, A_SESSION)
        await _request_stop(client, A_SESSION)

        finalize_response, create_response = await asyncio.gather(
            _finalize(client, A_SESSION),
            client.post(
                "/v1/batches",
                headers=auth(),
                json={"prompts": ["atomic race"]},
            ),
        )
        outcomes = (finalize_response.status_code, create_response.status_code)
        assert outcomes in {(200, 423), (423, 201)}
        if finalize_response.status_code == 200:
            assert finalize_response.json()["stop_request"]["state"] == "finalizing"
            assert create_response.json()["error"]["code"] == "gpu_stop_pending"
            await client.post(
                f"/v1/studio/stop-requests/{REQUEST_A}/cancel",
                headers=auth(),
                json={"session_id": A_SESSION, "finalization_id": FINALIZATION_A},
            )
        else:
            assert finalize_response.json()["error"]["code"] == (
                "stop_blocked_by_active_batch"
            )
            assert create_response.status_code == 201
            release.set()
            await wait_for_batch(
                client, create_response.json()["batch_id"], state="completed"
            )
        release.set()


@pytest.mark.anyio
async def test_finalization_guard_blocks_retry_and_expires_safely(tmp_path: Path) -> None:
    adapter = FakeInferenceAdapter(failures_before_success={1: 99})
    async with worker_client(tmp_path / "volume", adapter) as (client, app, _):
        clock = _install_clock(app)
        failed = await client.post(
            "/v1/batches", headers=auth(), json={"prompts": ["fail safely"]}
        )
        manifest = await wait_for_batch(
            client, failed.json()["batch_id"], state="completed"
        )
        assert manifest["images"][0]["status"] == "failed"

        await _heartbeat(client, A_SESSION)
        await _request_stop(client, A_SESSION)
        finalized = await _finalize(client, A_SESSION)
        assert finalized.json()["finalization_ttl_seconds"] == 60
        retry = await client.post(
            f"/v1/batches/{manifest['batch_id']}/retry-failed", headers=auth()
        )
        assert retry.status_code == 423
        assert retry.json()["error"]["code"] == "gpu_stop_pending"

        for _ in range(4):
            clock.advance(14)
            await _heartbeat(client, A_SESSION)
        clock.advance(5)
        expired = await _state(client, A_SESSION)
        assert expired["stop_request"]["state"] == "expired"
        assert expired["stop_request"]["reason"] == "finalization_expired"
        adapter.failures_before_success[1] = 3
        retried = await client.post(
            f"/v1/batches/{manifest['batch_id']}/retry-failed", headers=auth()
        )
        assert retried.status_code == 200
        await wait_for_batch(client, manifest["batch_id"], state="completed")


@pytest.mark.anyio
async def test_finalization_guard_outlives_requester_presence_during_ambiguous_delete(
    tmp_path: Path,
) -> None:
    async with worker_client(tmp_path / "volume") as (client, app, _):
        clock = _install_clock(app)
        await _heartbeat(client, A_SESSION)
        await _request_stop(client, A_SESSION)
        clock.advance(10)
        finalized = await _finalize(client, A_SESSION)
        assert finalized.json()["stop_request"]["state"] == "finalizing"
        await _heartbeat(
            client, B_SESSION, token=TOKEN_B, availability="background"
        )

        clock.advance(14)
        await _heartbeat(
            client, B_SESSION, token=TOKEN_B, availability="background"
        )
        clock.advance(14)
        after_requester_expiry = await _state(client, B_SESSION, token=TOKEN_B)
        assert after_requester_expiry["stop_request"]["state"] == "finalizing"
        assert after_requester_expiry["stop_request"]["finalization_id"] is None
        blocked = await client.post(
            "/v1/batches", headers=auth(TOKEN_B), json={"prompts": ["must stay blocked"]}
        )
        assert blocked.status_code == 423
        assert blocked.json()["error"]["code"] == "gpu_stop_pending"

        clock.advance(14)
        await _heartbeat(
            client, B_SESSION, token=TOKEN_B, availability="background"
        )
        still_blocked = await client.post(
            "/v1/batches",
            headers=auth(TOKEN_B),
            json={"prompts": ["delete may still be in flight after 30 seconds"]},
        )
        assert still_blocked.status_code == 423
        assert still_blocked.json()["error"]["code"] == "gpu_stop_pending"

        clock.advance(14)
        await _heartbeat(
            client, B_SESSION, token=TOKEN_B, availability="background"
        )
        clock.advance(5)
        expired = await _state(client, B_SESSION, token=TOKEN_B)
        assert expired["stop_request"]["state"] == "expired"
        assert expired["stop_request"]["reason"] == "finalization_expired"


@pytest.mark.anyio
async def test_finalization_guard_blocks_resume_even_if_manifest_changes_after_grant(
    tmp_path: Path,
) -> None:
    async with worker_client(tmp_path / "volume") as (client, app, _):
        _install_clock(app)
        created = await client.post(
            "/v1/batches", headers=auth(), json={"prompts": ["completed"]}
        )
        batch_id = created.json()["batch_id"]
        await wait_for_batch(client, batch_id, state="completed")
        await _heartbeat(client, A_SESSION)
        await _request_stop(client, A_SESSION)
        await _finalize(client, A_SESSION)

        manifest = app.state.runtime.store.load(batch_id)
        manifest.state = BatchState.INTERRUPTED
        manifest.completed_at = None
        manifest.interrupted_at = utc_now()
        assert app.state.runtime.store.try_acquire_active_lease()
        app.state.runtime.store.save(manifest)

        resume = await client.post(f"/v1/batches/{batch_id}/resume", headers=auth())
        assert resume.status_code == 423
        assert resume.json()["error"]["code"] == "gpu_stop_pending"
        unchanged = await client.get(f"/v1/batches/{batch_id}", headers=auth())
        assert unchanged.json()["state"] == "interrupted"


@pytest.mark.anyio
async def test_worker_restart_invalidates_sessions_requests_and_finalization_grants(
    tmp_path: Path,
) -> None:
    volume = tmp_path / "volume"
    async with worker_client(volume) as (client, app, _):
        first = await _heartbeat(client, A_SESSION)
        first_instance = first["server_instance_id"]
        await _request_stop(client, A_SESSION)
        await _finalize(client, A_SESSION)

    async with worker_client(volume) as (client, app, _):
        clock = AdjustableSystemClock()
        app.state.runtime.controller.coordination.clock = clock
        stale_session = await client.get(
            f"/v1/studio/sessions/{A_SESSION}", headers=auth()
        )
        assert stale_session.status_code == 404
        fresh = await _heartbeat(client, A_SESSION)
        assert fresh["server_instance_id"] != first_instance
        orphan = fresh["stop_request"]
        assert orphan["state"] == "finalizing"
        assert orphan["request_id"] == REQUEST_A
        assert orphan["pod_id"] == "pod-123"
        assert orphan["gpu_display_name"] == "NVIDIA RTX 4090"
        assert orphan["requester"]["display_name"] == "Lakshman"
        assert orphan["requester"]["session_id"] != A_SESSION
        assert orphan["requester"]["session_id"] not in {
            session["session_id"] for session in fresh["sessions"]
        }
        assert orphan["finalization_id"] is None
        assert orphan["finalization_expires_at"] is not None
        assert orphan["waiting_for"] == []
        assert orphan["approved_by"] == []
        assert orphan["denied_by"] == []
        stale_request = await client.post(
            f"/v1/studio/stop-requests/{REQUEST_A}/cancel",
            headers=auth(),
            json={"session_id": A_SESSION, "finalization_id": FINALIZATION_A},
        )
        assert stale_request.status_code == 404
        assert stale_request.json()["error"]["code"] == "stop_request_not_found"
        stale_response = await client.post(
            f"/v1/studio/stop-requests/{REQUEST_A}/responses",
            headers=auth(),
            json={"session_id": A_SESSION, "decision": "approve"},
        )
        assert stale_response.status_code == 404
        stale_finalize = await _finalize(client, A_SESSION)
        assert stale_finalize.status_code == 404
        status = await client.get("/v1/status", headers=auth())
        assert status.json()["permissions"]["can_create"] is False
        blocked = await client.post(
            "/v1/batches",
            headers=auth(),
            json={"prompts": ["restart cannot bypass an in-flight delete"]},
        )
        assert blocked.status_code == 423
        assert blocked.json()["error"]["code"] == "gpu_stop_pending"
        assert blocked.json()["error"]["details"]["request_id"] == REQUEST_A

        clock.advance(61)
        recovered = await _heartbeat(client, A_SESSION)
        assert recovered["stop_request"] is None
        status = await client.get("/v1/status", headers=auth())
        assert status.json()["permissions"]["can_create"] is True


@pytest.mark.anyio
async def test_studio_session_limits_are_bounded_per_authenticated_principal(
    tmp_path: Path,
) -> None:
    async with worker_client(tmp_path / "volume") as (client, app, _):
        _install_clock(app)
        for index in range(1, 9):
            session_id = f"30000000-0000-4000-8000-{index:012d}"
            await _heartbeat(client, session_id)
        ninth = await client.put(
            "/v1/studio/sessions/30000000-0000-4000-8000-000000000009",
            headers=auth(),
            json={"availability": "foreground"},
        )
        assert ninth.status_code == 429
        assert ninth.json()["error"]["code"] == "studio_session_limit"
