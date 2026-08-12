from __future__ import annotations

import asyncio
import hashlib
import io
import json
import uuid
from pathlib import Path
from typing import Literal

import httpx
import pytest
from conftest import TOKEN_A, TOKEN_B, auth, wait_for_batch, worker_client
from PIL import Image

from imageforge_worker import controller as controller_module
from imageforge_worker.constants import MAX_SEED
from imageforge_worker.inference import FakeInferenceAdapter
from imageforge_worker.model_profiles import FLUX2_KLEIN
from imageforge_worker.persistence import FileManifestStore, SubmissionStoreCorruptError


def _submission_id() -> str:
    return str(uuid.uuid4())


def _batch_payload(
    prompts: list[str] | None = None,
    *,
    submission_id: str | None = None,
    admission_mode: Literal["foreground", "queue"] | None = None,
    base_seed: int = 0,
    aspect_ratio: str = "16:9",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "client_submission_id": submission_id or _submission_id(),
        "prompts": prompts or ["idempotent editorial frame"],
        "base_seed": base_seed,
        "aspect_ratio": aspect_ratio,
    }
    if admission_mode is not None:
        payload["admission_mode"] = admission_mode
    return payload


def test_submission_history_enumeration_io_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileManifestStore(tmp_path / "volume")

    def fail_enumeration() -> list[str]:
        raise OSError("injected unreadable directory")

    monkeypatch.setattr(store, "list_batch_ids", fail_enumeration)
    assert store.submission_store_corrupt() is True
    assert store.try_acquire_submission_lease() is True
    try:
        with pytest.raises(SubmissionStoreCorruptError):
            store.find_submission(_submission_id())
    finally:
        store.release_submission_lease()


async def _heartbeat(client, session_id: str, *, token: str = TOKEN_A) -> None:  # type: ignore[no-untyped-def]
    response = await client.put(
        f"/v1/studio/sessions/{session_id}",
        headers=auth(token),
        json={"availability": "foreground"},
    )
    assert response.status_code == 200, response.text


async def _request_stop(client, session_id: str) -> dict:  # type: ignore[no-untyped-def]
    response = await client.post(
        "/v1/studio/stop-requests",
        headers=auth(),
        json={
            "request_id": _submission_id(),
            "session_id": session_id,
            "pod_id": "pod-queue-test",
            "gpu_display_name": "NVIDIA RTX 4090",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def reference_capable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the reference plumbing against a model that accepts references.

    The shipped model is text-to-image only, but the reference path still ships
    for the FLUX backend, so it keeps its coverage rather than being deleted.
    """

    monkeypatch.setattr(controller_module, "ACTIVE_PROFILE", FLUX2_KLEIN)


@pytest.mark.anyio
async def test_submission_id_is_required_canonical_and_admission_mode_is_strict(
    tmp_path: Path,
) -> None:
    async with worker_client(tmp_path / "volume", inject_submission_ids=False) as (client, _, _):
        missing = await client.post("/v1/batches", headers=auth(), json={"prompts": ["safe"]})
        assert missing.status_code == 422
        assert missing.json()["error"]["code"] == "validation_error"

        uppercase = await client.post(
            "/v1/batches",
            headers=auth(),
            json=_batch_payload(submission_id=_submission_id().upper()),
        )
        assert uppercase.status_code == 422
        assert uppercase.json()["error"]["code"] == "validation_error"

        wrong_version = await client.post(
            "/v1/batches",
            headers=auth(),
            json=_batch_payload(submission_id="00000000-0000-1000-8000-000000000001"),
        )
        assert wrong_version.status_code == 422

        invalid_mode = await client.post(
            "/v1/batches",
            headers=auth(),
            json={**_batch_payload(), "admission_mode": "later"},
        )
        assert invalid_mode.status_code == 422

        missing_lookup = await client.get(f"/v1/submissions/{_submission_id()}", headers=auth())
        assert missing_lookup.status_code == 404
        assert missing_lookup.json()["error"]["details"] is None
        malformed_lookup = await client.get(
            f"/v1/submissions/{_submission_id().upper()}", headers=auth()
        )
        assert malformed_lookup.status_code == 422


@pytest.mark.anyio
async def test_first_replay_fingerprint_and_owner_only_lookup_are_exact(tmp_path: Path) -> None:
    submission_id = _submission_id()
    request = _batch_payload(
        ["first resolved prompt", "second resolved prompt"],
        submission_id=submission_id,
        base_seed=42,
        aspect_ratio="1:1",
    )
    async with worker_client(tmp_path / "volume", inject_submission_ids=False) as (client, _, _):
        first = await client.post("/v1/batches", headers=auth(), json=request)
        assert first.status_code == 201, first.text
        replay = await client.post("/v1/batches", headers=auth(), json=request)
        assert replay.status_code == 201, replay.text
        assert replay.json() == first.json()
        manifest = first.json()
        assert manifest["client_submission_id"] == submission_id
        assert "request_fingerprint" not in first.text

        owner_lookup = await client.get(f"/v1/submissions/{submission_id}", headers=auth())
        assert owner_lookup.status_code == 200
        assert owner_lookup.json() == manifest

        foreign_lookup = await client.get(f"/v1/submissions/{submission_id}", headers=auth(TOKEN_B))
        assert foreign_lookup.status_code == 404
        assert foreign_lookup.json()["error"] == {
            "code": "submission_not_found",
            "message": "The requested submission does not exist.",
            "details": None,
        }

        foreign_status = await client.get("/v1/status", headers=auth(TOKEN_B))
        assert foreign_status.status_code == 200
        assert "client_submission_id" not in json.dumps(foreign_status.json())
        assert "request_fingerprint" not in json.dumps(foreign_status.json())

        raw = json.loads(
            (tmp_path / "volume" / "batches" / manifest["batch_id"] / "manifest.json").read_text()
        )
        assert set(raw) == {"schema_version", "manifest", "submission"}
        assert raw["schema_version"] == 2
        assert raw["manifest"]["client_submission_id"] == submission_id
        assert raw["submission"]["client_submission_id"] == submission_id
        expected_value = {
            "owner_user_id": "lakshman",
            "admission_mode": "foreground",
            "prompts": request["prompts"],
            "base_seed": 42,
            "aspect_ratio": "1:1",
            "references": [],
            "settings": manifest["settings"],
        }
        expected = hashlib.sha256(
            b"imageforge-batch-submission-v1\n"
            + json.dumps(
                expected_value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assert raw["submission"]["request_fingerprint"] == expected

        changed = await client.post(
            "/v1/batches",
            headers=auth(),
            json={**request, "prompts": ["changed after response loss"]},
        )
        assert changed.status_code == 409
        assert changed.json()["error"]["code"] == "submission_conflict"
        assert changed.json()["error"]["details"] is None

        foreign_reuse = await client.post("/v1/batches", headers=auth(TOKEN_B), json=request)
        assert foreign_reuse.status_code == 409
        assert foreign_reuse.json()["error"]["code"] == "submission_conflict"
        assert foreign_reuse.json()["error"]["details"] is None


@pytest.mark.anyio
async def test_fingerprint_binds_decoded_reference_hashes_and_all_settings(
    tmp_path: Path, reference_capable_model: None
) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), "teal").save(buffer, format="PNG")
    raw_reference = buffer.getvalue()
    submission_id = _submission_id()
    request = _batch_payload(
        ["reference-bound frame"],
        submission_id=submission_id,
        base_seed=7,
        aspect_ratio="3:4",
    )
    request["references"] = [
        {
            "name": "subject.png",
            "mime_type": "image/png",
            "data_hex": raw_reference.hex(),
        }
    ]
    async with worker_client(tmp_path / "volume", inject_submission_ids=False) as (client, _, _):
        created = await client.post("/v1/batches", headers=auth(), json=request)
        assert created.status_code == 201, created.text
        manifest = created.json()
        raw = json.loads(
            (tmp_path / "volume" / "batches" / manifest["batch_id"] / "manifest.json").read_text()
        )
        expected_value = {
            "owner_user_id": "lakshman",
            "admission_mode": "foreground",
            "prompts": ["reference-bound frame"],
            "base_seed": 7,
            "aspect_ratio": "3:4",
            "references": [
                {
                    "name": "subject.png",
                    "mime_type": "image/png",
                    "size_bytes": len(raw_reference),
                    "sha256": hashlib.sha256(raw_reference).hexdigest(),
                }
            ],
            "settings": manifest["settings"],
        }
        expected = hashlib.sha256(
            b"imageforge-batch-submission-v1\n"
            + json.dumps(
                expected_value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assert raw["submission"]["request_fingerprint"] == expected
        assert raw_reference.hex() not in json.dumps(raw)


@pytest.mark.anyio
async def test_replay_wins_over_busy_and_simultaneous_same_key_never_duplicates(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    adapter = FakeInferenceAdapter(
        first_generation_started=started,
        release_first_generation=release,
    )
    submission_id = _submission_id()
    request = _batch_payload(submission_id=submission_id)
    async with worker_client(tmp_path / "volume", adapter, inject_submission_ids=False) as (
        client,
        app,
        _,
    ):
        duplicate_transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=duplicate_transport, base_url="http://worker.test"
        ) as second_client:
            first, second = await asyncio.gather(
                client.post("/v1/batches", headers=auth(), json=request),
                second_client.post("/v1/batches", headers=auth(), json=request),
            )
        assert first.status_code == second.status_code == 201
        assert first.json()["batch_id"] == second.json()["batch_id"]
        assert len(app.state.runtime.store.list_batch_ids()) == 1
        await asyncio.wait_for(started.wait(), timeout=2)

        replay = await client.post("/v1/batches", headers=auth(), json=request)
        assert replay.status_code == 201
        assert replay.json()["batch_id"] == first.json()["batch_id"]

        busy = await client.post(
            "/v1/batches",
            headers=auth(TOKEN_B),
            json=_batch_payload(["another real batch"]),
        )
        assert busy.status_code == 423
        assert busy.json()["error"]["code"] == "batch_busy"
        release.set()


@pytest.mark.anyio
async def test_queue_stop_admission_parks_without_cancelling_peer_consent(tmp_path: Path) -> None:
    session_a = "10000000-0000-4000-8000-000000000001"
    session_b = "10000000-0000-4000-8000-000000000002"
    async with worker_client(tmp_path / "volume", inject_submission_ids=False) as (client, _, _):
        await _heartbeat(client, session_a)
        await _heartbeat(client, session_b, token=TOKEN_B)
        stop = await _request_stop(client, session_a)
        assert stop["stop_request"]["state"] == "pending"

        queue = await client.post(
            "/v1/batches",
            headers=auth(TOKEN_B),
            json=_batch_payload(admission_mode="queue"),
        )
        assert queue.status_code == 423
        error = queue.json()["error"]
        assert error["code"] == "queue_stop_pending"
        assert error["details"] == {
            "request_id": stop["stop_request"]["request_id"],
            "requester": "Lakshman",
            "state": "pending",
            "expires_at": stop["stop_request"]["response_deadline"],
        }

        approved = await client.post(
            f"/v1/studio/stop-requests/{stop['stop_request']['request_id']}/responses",
            headers=auth(TOKEN_B),
            json={"session_id": session_b, "decision": "approve"},
        )
        assert approved.status_code == 200
        assert approved.json()["stop_request"]["state"] == "approved"
        approved_queue = await client.post(
            "/v1/batches",
            headers=auth(TOKEN_B),
            json=_batch_payload(admission_mode="queue"),
        )
        assert approved_queue.status_code == 423
        assert approved_queue.json()["error"]["code"] == "queue_stop_pending"
        assert approved_queue.json()["error"]["details"]["state"] == "approved"

        state = await client.get(f"/v1/studio/sessions/{session_a}", headers=auth())
        assert state.json()["stop_request"]["state"] == "approved"

        foreground = await client.post(
            "/v1/batches",
            headers=auth(TOKEN_B),
            json=_batch_payload(["explicit foreground work"]),
        )
        assert foreground.status_code == 201
        state = await client.get(f"/v1/studio/sessions/{session_a}", headers=auth())
        assert state.json()["stop_request"]["state"] == "cancelled"
        assert state.json()["stop_request"]["reason"] == "generation_started"


@pytest.mark.anyio
async def test_replay_precedes_finalizing_stop_but_new_queue_is_blocked(tmp_path: Path) -> None:
    session = "10000000-0000-4000-8000-000000000003"
    submission_id = _submission_id()
    request = _batch_payload(submission_id=submission_id)
    finalization_id = _submission_id()
    async with worker_client(tmp_path / "volume", inject_submission_ids=False) as (client, _, _):
        created = await client.post("/v1/batches", headers=auth(), json=request)
        assert created.status_code == 201
        await wait_for_batch(client, created.json()["batch_id"], state="completed")

        await _heartbeat(client, session)
        stop = await _request_stop(client, session)
        assert stop["stop_request"]["state"] == "approved"
        finalizing = await client.post(
            f"/v1/studio/stop-requests/{stop['stop_request']['request_id']}/finalize",
            headers=auth(),
            json={"session_id": session, "finalization_id": finalization_id},
        )
        assert finalizing.status_code == 200
        assert finalizing.json()["stop_request"]["state"] == "finalizing"

        replay = await client.post("/v1/batches", headers=auth(), json=request)
        assert replay.status_code == 201
        assert replay.json()["batch_id"] == created.json()["batch_id"]

        blocked = await client.post(
            "/v1/batches",
            headers=auth(),
            json=_batch_payload(["new queue candidate"], admission_mode="queue"),
        )
        assert blocked.status_code == 423
        assert blocked.json()["error"]["code"] == "gpu_stop_pending"


@pytest.mark.anyio
async def test_js_safe_seed_limit_is_uniform_at_http_boundary(tmp_path: Path) -> None:
    async with worker_client(tmp_path / "volume", inject_submission_ids=False) as (client, _, _):
        at_limit = await client.post(
            "/v1/batches",
            headers=auth(),
            json=_batch_payload(["one"], base_seed=MAX_SEED),
        )
        assert at_limit.status_code == 201
        await wait_for_batch(client, at_limit.json()["batch_id"], state="completed")

        overflow = await client.post(
            "/v1/batches",
            headers=auth(),
            json=_batch_payload(["overflow"], base_seed=MAX_SEED + 1),
        )
        assert overflow.status_code == 422

        range_overflow = await client.post(
            "/v1/batches",
            headers=auth(),
            json=_batch_payload(["first", "second"], base_seed=MAX_SEED),
        )
        assert range_overflow.status_code == 422


@pytest.mark.anyio
async def test_corrupt_v2_history_fails_closed_but_valid_owner_batch_read_survives(
    tmp_path: Path,
) -> None:
    first_id = _submission_id()
    second_id = _submission_id()
    async with worker_client(tmp_path / "volume", inject_submission_ids=False) as (client, _, _):
        first = await client.post(
            "/v1/batches", headers=auth(), json=_batch_payload(submission_id=first_id)
        )
        assert first.status_code == 201
        await wait_for_batch(client, first.json()["batch_id"], state="completed")
        second = await client.post(
            "/v1/batches", headers=auth(), json=_batch_payload(submission_id=second_id)
        )
        assert second.status_code == 201
        await wait_for_batch(client, second.json()["batch_id"], state="completed")

        corrupt_path = tmp_path / "volume" / "batches" / second.json()["batch_id"] / "manifest.json"
        corrupt_path.write_text('{"schema_version":2}')

        status = await client.get("/v1/status", headers=auth())
        assert status.status_code == 200
        assert status.json()["permissions"]["can_create"] is False
        assert status.json()["permissions"]["create_block_reason"] == "submission_store_corrupt"

        blocked = await client.post("/v1/batches", headers=auth(), json=_batch_payload())
        lookup = await client.get(f"/v1/submissions/{first_id}", headers=auth())
        for response in (blocked, lookup):
            assert response.status_code == 503
            assert response.json()["error"] == {
                "code": "submission_store_corrupt",
                "message": (
                    "Worker submission history is unavailable. Repair the shared volume before "
                    "starting generation."
                ),
                "details": None,
            }

        valid_read = await client.get(f"/v1/batches/{first.json()['batch_id']}", headers=auth())
        assert valid_read.status_code == 200
        assert valid_read.json()["client_submission_id"] == first_id


@pytest.mark.anyio
async def test_duplicate_submission_association_blocks_status_create_and_other_lookup(
    tmp_path: Path,
) -> None:
    volume = tmp_path / "volume"
    first_id = _submission_id()
    second_id = _submission_id()
    async with worker_client(volume, inject_submission_ids=False) as (client, _, _):
        first = await client.post(
            "/v1/batches", headers=auth(), json=_batch_payload(submission_id=first_id)
        )
        assert first.status_code == 201
        await wait_for_batch(client, first.json()["batch_id"], state="completed")
        second = await client.post(
            "/v1/batches", headers=auth(), json=_batch_payload(submission_id=second_id)
        )
        assert second.status_code == 201
        await wait_for_batch(client, second.json()["batch_id"], state="completed")

        second_envelope = volume / "batches" / second.json()["batch_id"] / "manifest.json"
        payload = json.loads(second_envelope.read_text())
        payload["manifest"]["client_submission_id"] = first_id
        payload["submission"]["client_submission_id"] = first_id
        second_envelope.write_text(json.dumps(payload, separators=(",", ":")))

        status = await client.get("/v1/status", headers=auth())
        assert status.status_code == 200
        assert status.json()["permissions"] == {
            "can_create": False,
            "can_manage_active": False,
            "is_owner": False,
            "create_block_reason": "submission_store_corrupt",
            "can_switch": False,
            "switch_block_code": "runtime_identity_unavailable",
        }
        create = await client.post("/v1/batches", headers=auth(), json=_batch_payload())
        unrelated_lookup = await client.get(f"/v1/submissions/{second_id}", headers=auth())
        for response in (create, unrelated_lookup):
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "submission_store_corrupt"
            assert response.json()["error"]["details"] is None


@pytest.mark.anyio
async def test_corrupt_history_precedes_worker_not_ready_during_boot(tmp_path: Path) -> None:
    volume = tmp_path / "volume"
    original_id = _submission_id()
    async with worker_client(volume, inject_submission_ids=False) as (client, _, _):
        created = await client.post(
            "/v1/batches", headers=auth(), json=_batch_payload(submission_id=original_id)
        )
        assert created.status_code == 201
        await wait_for_batch(client, created.json()["batch_id"], state="completed")

    envelope = volume / "batches" / created.json()["batch_id"] / "manifest.json"
    envelope.write_text('{"schema_version":2}')
    slow_boot = FakeInferenceAdapter(startup_delay_seconds=0.2)
    async with worker_client(
        volume,
        slow_boot,
        wait_until_ready=False,
        inject_submission_ids=False,
    ) as (client, _, _):
        response = await client.post("/v1/batches", headers=auth(), json=_batch_payload())
        assert response.status_code == 503
        assert response.json()["error"] == {
            "code": "submission_store_corrupt",
            "message": (
                "Worker submission history is unavailable. Repair the shared volume before "
                "starting generation."
            ),
            "details": None,
        }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("seam", "committed"),
    [
        ("after_reference_fsync", False),
        ("after_envelope_fsync", False),
        ("after_envelope_rename", True),
        ("after_active_memory_assignment", True),
        ("after_runner_launch", True),
        ("before_create_response", True),
    ],
)
async def test_each_submission_commit_crash_seam_has_one_safe_recovery(
    tmp_path: Path,
    seam: str,
    committed: bool,
) -> None:
    volume = tmp_path / seam
    fired = False

    def crash(point: str) -> None:
        nonlocal fired
        if point == seam and not fired:
            fired = True
            raise RuntimeError(f"injected crash at {point}")

    store = FileManifestStore(
        volume,
        fsync_writes=True,
        crash_hook=crash if seam.startswith("after_") and "memory" not in seam else None,
    )
    submission_id = _submission_id()
    request = _batch_payload(submission_id=submission_id)
    first_batch_id: str | None = None
    async with worker_client(
        volume,
        FakeInferenceAdapter(delay_seconds=0.1),
        inject_submission_ids=False,
        store=store,
    ) as (client, app, _):
        if seam in {
            "after_active_memory_assignment",
            "after_runner_launch",
            "before_create_response",
        }:
            app.state.runtime.controller._crash_hook = crash
        failed = await client.post("/v1/batches", headers=auth(), json=request)
        assert failed.status_code == 500
        assert fired is True
        # Disable the test hook before the simulated process exits. The durable
        # state, not an in-memory hook, must decide the replacement behavior.
        store._crash_hook = None
        app.state.runtime.controller._crash_hook = None
        for batch_id in app.state.runtime.store.list_batch_ids():
            envelope_path = volume / "batches" / batch_id / "manifest.json"
            if not envelope_path.is_file():
                continue
            raw = json.loads(envelope_path.read_text())
            if raw.get("schema_version") == 2:
                first_batch_id = raw["manifest"]["batch_id"]
                break

    async with worker_client(volume, inject_submission_ids=False) as (client, app, _):
        recovered = await client.post("/v1/batches", headers=auth(), json=request)
        assert recovered.status_code == 201, recovered.text
        assert len(app.state.runtime.store.list_batch_ids()) == 1
        if committed:
            assert first_batch_id is not None
            assert recovered.json()["batch_id"] == first_batch_id
        else:
            assert first_batch_id is None
        # Response-loss recovery is always lookup-first and never makes a
        # second durable record for the same key.
        lookup = await client.get(f"/v1/submissions/{submission_id}", headers=auth())
        assert lookup.status_code == 200
        assert lookup.json()["batch_id"] == recovered.json()["batch_id"]
