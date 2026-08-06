from __future__ import annotations

import asyncio
import json
import multiprocessing
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import TOKEN_A, TOKEN_B, auth, wait_for_batch, worker_client
from pydantic import ValidationError

import imageforge_worker.gpu_switch as gpu_switch_module
from imageforge_worker.auth import Principal
from imageforge_worker.constants import MODEL_ID, MODEL_REVISION
from imageforge_worker.controller import GenerationController
from imageforge_worker.gpu_switch import (
    GpuControlGuardConflictError,
    GpuSwitchStore,
    GpuSwitchStoreCorruptError,
    NvidiaRuntimeDeviceInspector,
    load_gpu_switch_code_contract,
    load_runtime_identity_contract,
)
from imageforge_worker.gpu_switch_models import (
    AdoptGpuSwitchRequestV1,
    CancelGpuSwitchRequestV1,
    CompleteGpuSwitchRequestV1,
    CreateGpuSwitchRequestV1,
    DeleteIntentGpuSwitchRequestV1,
    FinalizeGpuSwitchRequestV1,
    GpuRuntimeIdentityContractV1,
    GpuSwitchBatchOwnerV1,
    GpuSwitchCodeContractV1,
    GpuSwitchLookupResponseV1,
    GpuSwitchParticipantV1,
    GpuSwitchRequestViewV1,
    GpuSwitchResponseRequestV1,
    NativeWorkerGpuSwitchOwnerLookupV1,
    SettleGpuSwitchCreateRequestV1,
    SharedGpuSwitchMarkerV1,
    SharedGpuSwitchRequestEnvelopeV1,
    SharedGpuSwitchTombstoneV1,
    require_gpu_identity,
)
from imageforge_worker.inference import FakeInferenceAdapter
from imageforge_worker.persistence import FileManifestStore

A_SESSION = "40000000-0000-4000-8000-000000000001"
A_SESSION_2 = "40000000-0000-4000-8000-000000000002"
B_SESSION = "40000000-0000-4000-8000-000000000003"
B_SESSION_2 = "40000000-0000-4000-8000-000000000004"
SWITCH_A = "50000000-0000-4000-8000-000000000001"
SWITCH_B = "50000000-0000-4000-8000-000000000002"
ATTEMPT_A = "60000000-0000-4000-8000-000000000001"
FINALIZATION_A = "70000000-0000-4000-8000-000000000001"
BINDING_A = "80000000-0000-4000-8000-000000000001"
BATCH_A = "90000000-0000-4000-8000-000000000001"
OLD_GPU = "NVIDIA GeForce RTX 4090"
TARGET_GPU = "NVIDIA L4"
IMAGE_DIGEST = "ghcr.io/imageforge/worker@sha256:" + "a" * 64


def _runtime_metadata(*, pod_id: str = "old-pod", gpu_id: str = OLD_GPU) -> dict[str, str]:
    return {
        "RUNPOD_POD_ID": pod_id,
        "RUNPOD_VOLUME_ID": "volume-123",
        "RUNPOD_DC_ID": "EU-RO-1",
        "RUNPOD_GPU_COUNT": "1",
        "IMAGEFORGE_IMAGE_DIGEST": IMAGE_DIGEST,
        "IMAGEFORGE_EXPECTED_GPU_TYPE_ID": gpu_id,
    }


def _create_request(
    *,
    switch_id: str = SWITCH_A,
    session_id: str = A_SESSION,
    expected_batch_id: str | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "switch_id": switch_id,
        "session_id": session_id,
        "old_pod_id": "old-pod",
        "old_gpu_id": OLD_GPU,
        "old_gpu_display_name": "RTX 4090",
        "initial_target_gpu_id": TARGET_GPU,
        "initial_target_gpu_display_name": "L4",
        "initial_replacement_attempt_id": ATTEMPT_A,
        "expected_batch_id": expected_batch_id,
        "inventory_observed_at": "2026-08-04T00:00:00.000Z",
    }


async def _heartbeat(client, session_id: str, *, token: str = TOKEN_A) -> dict:
    response = await client.put(
        f"/v1/studio/sessions/{session_id}",
        headers=auth(token),
        json={"availability": "foreground"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _view(*, state: str = "approved", session_id: str = A_SESSION) -> GpuSwitchRequestViewV1:
    return GpuSwitchRequestViewV1(
        schema_version=1,
        switch_id=SWITCH_A,
        old_pod_id="old-pod",
        old_gpu_id=OLD_GPU,
        old_gpu_display_name="RTX 4090",
        initial_target_gpu_id=TARGET_GPU,
        initial_target_gpu_display_name="L4",
        initial_replacement_attempt_id=ATTEMPT_A,
        requester=GpuSwitchParticipantV1(session_id=session_id, display_name="Lakshman"),
        state=state,
        reason=None,
        requested_at="2026-08-04T00:00:00.000Z",
        response_deadline="2026-08-04T00:00:30.000Z",
        ready_to_delete_at=None,
        waiting_for=[],
        approved_by=[],
        denied_by=[],
        batch_id=None,
        batch_owner=None,
        batch_state_at_finalization=None,
        replacement_attempt_id=None,
        replacement_attempt_revision=None,
        replacement_pod_id=None,
        actual_target_gpu_id=None,
    )


def _envelope(*, view: GpuSwitchRequestViewV1 | None = None) -> SharedGpuSwitchRequestEnvelopeV1:
    selected = view or _view()
    return SharedGpuSwitchRequestEnvelopeV1(
        schema_version=1,
        envelope_revision=1,
        switch_id=selected.switch_id,
        request_fingerprint_sha256="b" * 64,
        requester_user_id="lakshman",
        requester_session_id=selected.requester.session_id,
        principal_binding_id=BINDING_A,
        active_request=selected,
        terminal_tombstone=None,
        created_at="2026-08-04T00:00:00.000Z",
        updated_at="2026-08-04T00:00:00.000Z",
    )


def _tombstone(
    *,
    reason: str = "requester_cancelled",
    terminal_state: str = "cancelled",
    finalization_id: str | None = None,
) -> SharedGpuSwitchTombstoneV1:
    return SharedGpuSwitchTombstoneV1(
        schema_version=1,
        switch_id=SWITCH_A,
        principal_binding_id=BINDING_A,
        requester_user_id="lakshman",
        finalization_id=finalization_id,
        terminal_state=terminal_state,
        terminal_reason=reason,
        replacement_attempt_id=None,
        replacement_attempt_revision=None,
        replacement_pod_id=None,
        actual_target_gpu_id=None,
        terminal_at="2026-08-04T00:00:01.000Z",
    )


def _marker(*, phase: str = "pausing") -> SharedGpuSwitchMarkerV1:
    replacement = phase == "replacement_ready"
    return SharedGpuSwitchMarkerV1(
        schema_version=1,
        switch_id=SWITCH_A,
        finalization_id=FINALIZATION_A,
        principal_binding_id=BINDING_A,
        requester_user_id="lakshman",
        requester_display_name="Lakshman",
        old_pod_id="old-pod",
        old_gpu_id=OLD_GPU,
        initial_target_gpu_id=TARGET_GPU,
        initial_replacement_attempt_id=ATTEMPT_A,
        batch_id=None,
        batch_owner_user_id=None,
        batch_state_at_finalization=None,
        phase=phase,
        replacement_attempt_id=ATTEMPT_A if replacement else None,
        replacement_attempt_revision=1 if replacement else None,
        replacement_pod_id="replacement-pod" if replacement else None,
        actual_target_gpu_id=TARGET_GPU if replacement else None,
        create_contract_revision=1,
        create_marker_sha256="1" * 64 if replacement else None,
        create_intent_sha256="2" * 64 if replacement else None,
        create_wire_body_sha256="3" * 64 if replacement else None,
        expected_volume_id="volume-123",
        expected_data_center_id="EU-RO-1",
        expected_image_digest=IMAGE_DIGEST,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        requested_at="2026-08-04T00:00:00.000Z",
        updated_at="2026-08-04T00:00:01.000Z",
    )


def _batch_bound_view() -> GpuSwitchRequestViewV1:
    payload = _view(state="pausing").model_dump(mode="python")
    payload.update(
        {
            "batch_id": BATCH_A,
            "batch_owner": GpuSwitchBatchOwnerV1(display_name="Lakshman"),
            "batch_state_at_finalization": "running",
        }
    )
    return GpuSwitchRequestViewV1.model_validate(payload)


def _batch_bound_marker() -> SharedGpuSwitchMarkerV1:
    payload = _marker().model_dump(mode="python")
    payload.update(
        {
            "batch_id": BATCH_A,
            "batch_owner_user_id": "lakshman",
            "batch_state_at_finalization": "running",
        }
    )
    return SharedGpuSwitchMarkerV1.model_validate(payload)


def _view_for_marker_phase(phase: str) -> GpuSwitchRequestViewV1:
    payload = _view().model_dump(mode="python")
    payload["state"] = phase
    if phase in {"ready_to_delete", "delete_intent", "replacement_ready"}:
        payload["ready_to_delete_at"] = "2026-08-04T00:00:01.000Z"
    if phase == "replacement_ready":
        payload.update(
            {
                "replacement_attempt_id": ATTEMPT_A,
                "replacement_attempt_revision": 1,
                "replacement_pod_id": "replacement-pod",
                "actual_target_gpu_id": TARGET_GPU,
            }
        )
    return GpuSwitchRequestViewV1.model_validate(payload)


def _open_switch_store(root: Path) -> tuple[FileManifestStore, GpuSwitchStore]:
    manifest_store = FileManifestStore(root, fsync_writes=False)
    manifest_store.initialize()
    assert manifest_store.try_acquire_active_lease()
    assert manifest_store.try_acquire_gpu_control_lock()
    switch_store = GpuSwitchStore(root, manifest_store, fsync_writes=False)
    switch_store.initialize()
    return manifest_store, switch_store


def _close_switch_store(store: FileManifestStore) -> None:
    store.release_gpu_control_lock()
    store.release_active_lease()


def _switch_store_writer_child(
    root: Path,
    fingerprint_character: str,
    ready,
    start,
    release,
    results,
) -> None:
    manifest_store = FileManifestStore(root, fsync_writes=False)
    manifest_store.initialize()
    ready.put(fingerprint_character)
    if not start.wait(10):
        results.put({"writer": fingerprint_character, "result": "start_timeout"})
        return
    if not manifest_store.try_acquire_active_lease():
        results.put({"writer": fingerprint_character, "result": "lease_busy"})
        return
    try:
        if not manifest_store.try_acquire_gpu_control_lock():
            results.put({"writer": fingerprint_character, "result": "gpu_lock_busy"})
            return
        try:
            switch_store = GpuSwitchStore(root, manifest_store, fsync_writes=False)
            switch_store.initialize()
            if any(item.active_request is not None for item in switch_store.list_envelopes()):
                results.put({"writer": fingerprint_character, "result": "request_in_progress"})
                return
            envelope = _envelope().model_copy(
                update={"request_fingerprint_sha256": fingerprint_character * 64}
            )
            switch_store.write_envelope(envelope, previous=None)
            results.put({"writer": fingerprint_character, "result": "committed"})
            release.wait(10)
        finally:
            manifest_store.release_gpu_control_lock()
    finally:
        manifest_store.release_active_lease()


def test_shared_gpu_identity_vectors_and_task012_migration() -> None:
    vectors_path = Path(__file__).parents[2] / "contracts" / "gpu-identity-v1.vectors.json"
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
    for item in vectors["accepted"]:
        assert require_gpu_identity(item["value"]) == item["value"]
    for item in vectors["rejected"]:
        with pytest.raises(ValueError, match="GPU identity"):
            require_gpu_identity(item["value"])


def test_runtime_identity_contract_is_packaged_strict_complete_and_not_overridable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = Path(__file__).parents[2] / "contracts" / "gpu-runtime-identities-v1.vectors.json"
    root_contract = GpuRuntimeIdentityContractV1.model_validate_json(root_path.read_bytes())
    packaged = load_runtime_identity_contract()
    assert packaged == root_contract
    provider_ids = {item.providerGpuId for item in packaged.identities}
    assert {
        "NVIDIA A100 80GB PCIe",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "NVIDIA RTX PRO 4500 Blackwell",
        "NVIDIA RTX PRO 4000 Blackwell",
    }.issubset(provider_ids)

    poisoned = tmp_path / "poisoned-runtime-map.json"
    payload = root_contract.model_dump(mode="json")
    payload["unexpected"] = True
    poisoned.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("IMAGEFORGE_GPU_RUNTIME_IDENTITIES_PATH", str(poisoned))
    assert load_runtime_identity_contract() == root_contract
    with pytest.raises(ValueError, match="test-only"):
        load_runtime_identity_contract(poisoned)
    with pytest.raises(ValidationError):
        load_runtime_identity_contract(poisoned, allow_test_override=True)


def test_runtime_inspector_canonicalizes_nvml_device_vendor_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_switch_module.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu_switch_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                "GPU-00000000-0000-0000-0000-000000000001, "
                "0x268410DE, NVIDIA GeForce RTX 4090, 24564\n"
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                device_count=lambda: 1,
                get_device_capability=lambda _index: (8, 9),
            )
        ),
    )

    observed = NvidiaRuntimeDeviceInspector().inspect()
    assert observed.device_count == 1
    assert observed.device.pciDeviceId == "0x2684"
    assert observed.device.totalMemoryBytes == 24564 * 1024 * 1024


def test_gpu_switch_code_registry_is_exhaustive_and_packaged_without_drift() -> None:
    root_path = Path(__file__).parents[2] / "contracts" / "gpu-switch-codes-v1.json"
    root_contract = GpuSwitchCodeContractV1.model_validate_json(root_path.read_bytes())
    packaged = load_gpu_switch_code_contract()
    assert packaged == root_contract

    expected_issues = set(
        """gpu_switch_store_recovered gpu_switch_store_unrecoverable
        gpu_switch_active gpu_switch_not_found gpu_switch_revision_conflict
        gpu_switch_revision_exhausted gpu_switch_lease_busy
        gpu_switch_lease_required gpu_switch_transition_invalid
        gpu_switch_foreground_grant_required gpu_switch_foreground_grant_invalid
        gpu_switch_foreground_grant_expired gpu_switch_foreground_grant_consumed
        queue_gpu_switch_pending gpu_switch_queue_reservation_conflict
        gpu_switch_queue_reservation_corrupt gpu_switch_queue_dispatch_uncertain
        gpu_switch_local_receipts_pending gpu_switch_inventory_unavailable
        gpu_switch_inventory_stale gpu_switch_inventory_receipt_invalid
        gpu_switch_price_changed gpu_actual_price_changed
        gpu_actual_price_unavailable gpu_identity_invalid
        gpu_switch_target_unapproved gpu_switch_target_unavailable
        gpu_switch_current_pod_unverified gpu_switch_requester_not_foreground
        gpu_switch_old_pod_changed gpu_switch_old_pod_disappeared_early
        gpu_switch_profile_locked gpu_switch_worker_create_uncertain
        gpu_switch_worker_response_invalid gpu_switch_worker_guard_missing
        gpu_switch_delete_uncertain gpu_switch_create_uncertain
        gpu_switch_replacement_ambiguous gpu_switch_replacement_mismatch
        gpu_switch_provider_response_mismatch gpu_switch_zero_match_unproven
        gpu_switch_replacement_cleanup_required
        gpu_switch_replacement_delete_uncertain gpu_switch_peer_pod_present
        gpu_switch_peer_pod_overflow gpu_switch_quote_invalid
        gpu_switch_quote_expired gpu_switch_quote_consumed
        gpu_switch_pause_failed gpu_switch_completion_failed
        gpu_switch_cancel_not_allowed stop_request_in_progress gpu_stop_pending
        gpu_switch_request_in_progress gpu_switch_pending
        gpu_control_guard_conflict gpu_switch_store_corrupt
        gpu_switch_runtime_identity_unavailable""".split()
    )
    actual_issues = {entry.code for entry in packaged.codes if entry.scope == "native_issue"}
    assert actual_issues == expected_issues

    expected_attention = set(
        """gpu_switch_revision_exhausted gpu_actual_price_changed
        gpu_actual_price_unavailable gpu_switch_target_unavailable
        gpu_switch_old_pod_changed gpu_switch_old_pod_disappeared_early
        gpu_switch_profile_locked gpu_switch_worker_create_uncertain
        gpu_switch_worker_response_invalid gpu_switch_worker_guard_missing
        gpu_switch_replacement_ambiguous gpu_switch_replacement_mismatch
        gpu_switch_provider_response_mismatch gpu_switch_zero_match_unproven
        gpu_switch_peer_pod_present gpu_switch_peer_pod_overflow
        gpu_switch_pause_failed gpu_switch_completion_failed
        gpu_switch_runtime_identity_unavailable""".split()
    )
    actual_attention = {entry.code for entry in packaged.codes if entry.scope == "native_attention"}
    assert actual_attention == expected_attention
    retryable_native = {
        entry.code for entry in packaged.codes if entry.scope == "native_issue" and entry.retryable
    }
    assert retryable_native == {
        "gpu_switch_revision_conflict",
        "gpu_switch_lease_busy",
        "gpu_switch_inventory_unavailable",
    }
    pause_failure = next(
        entry
        for entry in packaged.codes
        if entry.scope == "native_attention" and entry.code == "gpu_switch_pause_failed"
    )
    assert pause_failure.permittedBlockedPhases == ["pausing"]


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (CreateGpuSwitchRequestV1, _create_request()),
        (
            GpuSwitchResponseRequestV1,
            {"schema_version": 1, "session_id": B_SESSION, "decision": "approve"},
        ),
        (
            FinalizeGpuSwitchRequestV1,
            {
                "schema_version": 1,
                "session_id": A_SESSION,
                "finalization_id": FINALIZATION_A,
            },
        ),
        (
            DeleteIntentGpuSwitchRequestV1,
            {
                "schema_version": 1,
                "session_id": A_SESSION,
                "finalization_id": FINALIZATION_A,
            },
        ),
        (
            AdoptGpuSwitchRequestV1,
            {
                "schema_version": 1,
                "session_id": A_SESSION,
                "finalization_id": FINALIZATION_A,
                "replacement_attempt_id": ATTEMPT_A,
                "replacement_attempt_revision": 1,
                "replacement_pod_id": "replacement-pod",
                "target_gpu_id": TARGET_GPU,
                "create_contract_revision": 1,
                "create_marker_sha256": "1" * 64,
                "create_intent_sha256": "2" * 64,
                "create_wire_body_sha256": "3" * 64,
            },
        ),
        (
            CompleteGpuSwitchRequestV1,
            {
                "schema_version": 1,
                "session_id": A_SESSION,
                "finalization_id": FINALIZATION_A,
                "replacement_attempt_id": ATTEMPT_A,
                "replacement_attempt_revision": 1,
                "replacement_pod_id": "replacement-pod",
            },
        ),
        (
            CancelGpuSwitchRequestV1,
            {
                "schema_version": 1,
                "session_id": A_SESSION,
                "finalization_id": None,
            },
        ),
        (
            SettleGpuSwitchCreateRequestV1,
            {
                "schema_version": 1,
                "action": "cancel",
                "create_request": _create_request(),
            },
        ),
    ],
)
def test_gpu_switch_request_models_require_schema_and_reject_unknown_fields(
    model,
    payload: dict,
) -> None:
    without_schema = {key: value for key, value in payload.items() if key != "schema_version"}
    with pytest.raises(ValidationError):
        model.model_validate(without_schema)
    assert model.model_validate(payload).schema_version == 1
    with pytest.raises(ValidationError):
        model.model_validate({**payload, "unexpected": True})


def test_models_reject_forged_relations_and_user_ids() -> None:
    forged = _envelope().model_dump(mode="python")
    forged["requester_session_id"] = A_SESSION_2
    with pytest.raises(ValidationError, match="requester session"):
        SharedGpuSwitchRequestEnvelopeV1.model_validate(forged)

    forged = _envelope().model_dump(mode="python")
    forged["requester_user_id"] = "bad/user"
    with pytest.raises(ValidationError, match="user identity"):
        SharedGpuSwitchRequestEnvelopeV1.model_validate(forged)

    with pytest.raises(ValidationError, match="tombstone hash"):
        NativeWorkerGpuSwitchOwnerLookupV1(
            schema_version=1,
            switch_id=SWITCH_A,
            state="cancelled",
            requester_user_id="lakshman",
            principal_binding_id=BINDING_A,
            finalization_id=None,
            terminal_tombstone_sha256=None,
            replacement_attempt_id=None,
            replacement_attempt_revision=None,
            replacement_pod_id=None,
            actual_target_gpu_id=None,
        )

    with pytest.raises(ValidationError, match="all null or all populated"):
        GpuSwitchLookupResponseV1(
            schema_version=1,
            switch_id=SWITCH_A,
            state="replacement_ready",
            replacement_attempt_id=ATTEMPT_A,
            replacement_attempt_revision=None,
            replacement_pod_id="replacement-pod",
            actual_target_gpu_id=TARGET_GPU,
        )

    completed = _tombstone().model_dump(mode="python")
    completed.update({"terminal_state": "completed", "terminal_reason": "replacement_completed"})
    with pytest.raises(ValidationError, match="completion identity"):
        SharedGpuSwitchTombstoneV1.model_validate(completed)

    ready = _view().model_dump(mode="python")
    ready["state"] = "ready_to_delete"
    with pytest.raises(ValidationError, match="pause fixed point"):
        GpuSwitchRequestViewV1.model_validate(ready)


def test_attention_view_relation_is_closed_and_store_rejects_forged_payload(
    tmp_path: Path,
) -> None:
    valid = _batch_bound_view().model_dump(mode="python")
    valid.update({"state": "needs_attention", "reason": "pause_failed"})
    assert GpuSwitchRequestViewV1.model_validate(valid).reason == "pause_failed"

    invalid_payloads = [
        {**valid, "reason": None},
        {**valid, "reason": "replacement_mismatch"},
        {
            **valid,
            "batch_id": None,
            "batch_owner": None,
            "batch_state_at_finalization": None,
        },
        {**valid, "batch_state_at_finalization": "paused"},
        {**valid, "ready_to_delete_at": "2026-08-04T00:00:02.000Z"},
        {**valid, "state": "pausing"},
        {**valid, "state": "approved", "reason": "pause_failed"},
        {**valid, "state": "completed", "reason": None},
        {
            **_view().model_dump(mode="python"),
            "replacement_attempt_id": ATTEMPT_A,
            "replacement_attempt_revision": 1,
            "replacement_pod_id": "replacement-pod",
            "actual_target_gpu_id": TARGET_GPU,
        },
    ]
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            GpuSwitchRequestViewV1.model_validate(payload)

    manifest_store, switch_store = _open_switch_store(tmp_path / "forged-attention")
    forged_envelope = _envelope().model_dump(mode="json")
    assert forged_envelope["active_request"] is not None
    forged_envelope["active_request"]["state"] = "needs_attention"
    forged_envelope["active_request"]["reason"] = None
    switch_store._write_immutable(
        switch_store._request_path(SWITCH_A),
        json.dumps(forged_envelope, sort_keys=True, separators=(",", ":")).encode(),
        "gpu_switch_envelope",
    )
    with pytest.raises(GpuSwitchStoreCorruptError, match="record is invalid"):
        switch_store.initialize()
    _close_switch_store(manifest_store)


@pytest.mark.parametrize(
    ("marker_phase", "view_state"),
    [
        ("pausing", "pending"),
        ("ready_to_delete", "approved"),
        ("delete_intent", "pausing"),
        ("replacement_ready", "pending"),
    ],
)
def test_marker_phase_cannot_overwrite_incompatible_active_view(
    tmp_path: Path,
    marker_phase: str,
    view_state: str,
) -> None:
    manifest_store, switch_store = _open_switch_store(tmp_path / marker_phase)
    view_payload = _view(state=view_state).model_dump(mode="python")
    if view_state == "pausing":
        view_payload["batch_state_at_finalization"] = None
    envelope = _envelope(view=GpuSwitchRequestViewV1.model_validate(view_payload))
    switch_store.write_envelope(envelope, previous=None)

    marker = _marker()
    switch_store.write_marker(marker, previous=None)
    if marker_phase in {"ready_to_delete", "delete_intent", "replacement_ready"}:
        ready = marker.model_copy(update={"phase": "ready_to_delete"})
        switch_store.write_marker(ready, previous=marker)
        marker = ready
    if marker_phase in {"delete_intent", "replacement_ready"}:
        delete_intent = marker.model_copy(update={"phase": "delete_intent"})
        switch_store.write_marker(delete_intent, previous=marker)
        marker = delete_intent
    if marker_phase == "replacement_ready":
        replacement = _marker(phase="replacement_ready")
        switch_store.write_marker(replacement, previous=marker)
        marker = replacement

    with pytest.raises(GpuSwitchStoreCorruptError, match="phases disagree"):
        switch_store.initialize()
    assert switch_store.read_marker() == marker
    assert switch_store.read_envelope(SWITCH_A) == envelope
    _close_switch_store(manifest_store)


@pytest.mark.parametrize("view_state", ["delete_intent", "replacement_ready"])
def test_replacement_marker_accepts_only_authored_immediate_or_settled_view(
    tmp_path: Path,
    view_state: str,
) -> None:
    manifest_store, switch_store = _open_switch_store(tmp_path / view_state)
    view = _view_for_marker_phase(view_state)
    envelope = _envelope(view=view)
    switch_store.write_envelope(envelope, previous=None)
    pausing = _marker()
    switch_store.write_marker(pausing, previous=None)
    ready = pausing.model_copy(update={"phase": "ready_to_delete"})
    switch_store.write_marker(ready, previous=pausing)
    delete_intent = ready.model_copy(update={"phase": "delete_intent"})
    switch_store.write_marker(delete_intent, previous=ready)
    replacement = _marker(phase="replacement_ready")
    switch_store.write_marker(replacement, previous=delete_intent)

    switch_store.initialize()
    projected = gpu_switch_module.GpuSwitchCoordinator._view_from_marker(view, replacement)
    assert projected.state == "replacement_ready"
    assert projected.replacement_attempt_id == ATTEMPT_A
    assert projected.replacement_pod_id == "replacement-pod"
    _close_switch_store(manifest_store)


@pytest.mark.anyio
async def test_controller_startup_persists_authored_marker_view_crash_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_switch_module,
        "utc_now",
        lambda: "2026-08-04T00:00:02.000Z",
    )
    root = tmp_path / "volume"
    manifest_store, switch_store = _open_switch_store(root)
    envelope = _envelope(view=_view_for_marker_phase("delete_intent"))
    switch_store.write_envelope(envelope, previous=None)
    pausing = _marker()
    switch_store.write_marker(pausing, previous=None)
    ready = pausing.model_copy(update={"phase": "ready_to_delete"})
    switch_store.write_marker(ready, previous=pausing)
    delete_intent = ready.model_copy(update={"phase": "delete_intent"})
    switch_store.write_marker(delete_intent, previous=ready)
    replacement = _marker(phase="replacement_ready")
    switch_store.write_marker(replacement, previous=delete_intent)
    _close_switch_store(manifest_store)

    controller = GenerationController(
        FileManifestStore(root, fsync_writes=False),
        FakeInferenceAdapter(),
        max_attempts=3,
        retry_delay_seconds=0,
        runtime_metadata=_runtime_metadata(),
        data_root=root,
    )
    await controller.initialize()
    try:
        persisted = controller.gpu_switch.store.read_envelope(SWITCH_A)
        assert persisted is not None and persisted.active_request is not None
        assert persisted.envelope_revision == envelope.envelope_revision + 1
        assert persisted.active_request.state == "replacement_ready"
        assert persisted.active_request.replacement_attempt_id == ATTEMPT_A
        assert controller.gpu_switch.store.read_marker() == replacement
    finally:
        await controller.shutdown()


def test_tombstone_first_crash_reconciles_exact_terminal_envelope(tmp_path: Path) -> None:
    manifest_store, switch_store = _open_switch_store(tmp_path / "volume")
    envelope = _envelope()
    switch_store.write_envelope(envelope, previous=None)

    def crash(point: str) -> None:
        if point == "gpu_switch_tombstone_directory_fsync":
            raise RuntimeError("simulated process death")

    switch_store.crash_hook = crash
    with pytest.raises(RuntimeError, match="simulated"):
        switch_store.terminalize(envelope, _tombstone())
    assert switch_store.read_envelope(SWITCH_A).active_request is not None
    assert switch_store.read_tombstone(SWITCH_A) == _tombstone()
    _close_switch_store(manifest_store)

    restarted_manifest, restarted = _open_switch_store(tmp_path / "volume")
    restarted.reconcile_terminal_commits()
    recovered = restarted.read_envelope(SWITCH_A)
    assert recovered is not None
    assert recovered.envelope_revision == 2
    assert recovered.active_request is None
    assert recovered.terminal_tombstone == _tombstone()
    _close_switch_store(restarted_manifest)


def test_orphan_tombstone_and_marker_envelope_mismatch_fail_closed(tmp_path: Path) -> None:
    manifest_store, switch_store = _open_switch_store(tmp_path / "orphan")
    switch_store._write_immutable(
        switch_store._tombstone_path(SWITCH_A),
        json.dumps(
            _tombstone().model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode(),
        "gpu_switch_tombstone",
    )
    with pytest.raises(GpuSwitchStoreCorruptError, match="orphan"):
        switch_store.initialize()
    _close_switch_store(manifest_store)

    manifest_store, switch_store = _open_switch_store(tmp_path / "mismatch")
    envelope = _envelope()
    switch_store.write_envelope(envelope, previous=None)
    marker = _marker()
    switch_store.write_marker(marker, previous=None)
    payload = marker.model_dump(mode="json")
    payload["old_gpu_id"] = "NVIDIA GeForce RTX 5090"
    switch_store.marker_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(GpuSwitchStoreCorruptError, match="binding"):
        switch_store.initialize()
    _close_switch_store(manifest_store)


@pytest.mark.parametrize("terminal_marker_phase", ["delete_intent", "replacement_ready"])
def test_post_delete_cancellation_tombstone_never_clears_forward_guard(
    tmp_path: Path,
    terminal_marker_phase: str,
) -> None:
    manifest_store, switch_store = _open_switch_store(tmp_path / terminal_marker_phase)
    envelope = _envelope(view=_view_for_marker_phase(terminal_marker_phase))
    switch_store.write_envelope(envelope, previous=None)
    marker = _marker()
    switch_store.write_marker(marker, previous=None)
    ready = marker.model_copy(update={"phase": "ready_to_delete"})
    switch_store.write_marker(ready, previous=marker)
    delete_intent = ready.model_copy(update={"phase": "delete_intent"})
    switch_store.write_marker(delete_intent, previous=ready)
    expected_marker = delete_intent
    if terminal_marker_phase == "replacement_ready":
        expected_marker = _marker(phase="replacement_ready")
        switch_store.write_marker(expected_marker, previous=delete_intent)

    cancellation = _tombstone(finalization_id=FINALIZATION_A)
    switch_store._write_immutable(
        switch_store._tombstone_path(SWITCH_A),
        json.dumps(
            cancellation.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        "gpu_switch_tombstone",
    )

    with pytest.raises(GpuControlGuardConflictError, match="forward-only"):
        switch_store.initialize()
    with pytest.raises(GpuControlGuardConflictError, match="forward-only"):
        switch_store.reconcile_terminal_commits()
    assert switch_store.read_marker() == expected_marker
    assert switch_store.read_envelope(SWITCH_A) == envelope
    assert switch_store.read_tombstone(SWITCH_A) == cancellation
    _close_switch_store(manifest_store)


@pytest.mark.anyio
async def test_live_post_delete_cancellation_returns_guard_conflict_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "volume"
    manifest_store, switch_store = _open_switch_store(root)
    envelope = _envelope(view=_view_for_marker_phase("delete_intent"))
    switch_store.write_envelope(envelope, previous=None)
    marker = _marker()
    switch_store.write_marker(marker, previous=None)
    ready = marker.model_copy(update={"phase": "ready_to_delete"})
    switch_store.write_marker(ready, previous=marker)
    delete_intent = ready.model_copy(update={"phase": "delete_intent"})
    switch_store.write_marker(delete_intent, previous=ready)
    _close_switch_store(manifest_store)

    async with worker_client(root, runtime_metadata=_runtime_metadata()) as (
        client,
        app,
        _,
    ):
        controller = app.state.runtime.controller
        assert controller.store.try_acquire_gpu_control_lock()
        try:
            cancellation = _tombstone(finalization_id=FINALIZATION_A)
            controller.gpu_switch.store._write_immutable(
                controller.gpu_switch.store._tombstone_path(SWITCH_A),
                json.dumps(
                    cancellation.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
                "gpu_switch_tombstone",
            )
        finally:
            controller.store.release_gpu_control_lock()

        response = await client.post(
            f"/v1/studio/gpu-switches/{SWITCH_A}/finalize",
            headers=auth(),
            json={
                "schema_version": 1,
                "session_id": A_SESSION,
                "finalization_id": FINALIZATION_A,
            },
        )
        assert response.status_code == 503
        assert response.json()["error"] == {
            "code": "gpu_control_guard_conflict",
            "message": (
                "Worker GPU control history conflicts. Repair the shared volume before "
                "changing compute."
            ),
            "details": None,
        }
        assert controller.gpu_switch.store.read_marker() == delete_intent
        assert controller.gpu_switch.store.read_envelope(SWITCH_A) == envelope
        assert controller.gpu_switch.store.read_tombstone(SWITCH_A) == cancellation


def test_simultaneous_cross_process_switch_writers_publish_one_active_envelope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "volume"
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    results = context.Queue()
    start = context.Event()
    release = context.Event()
    writers = [
        context.Process(
            target=_switch_store_writer_child,
            args=(root, character, ready, start, release, results),
        )
        for character in ("a", "b")
    ]
    for writer in writers:
        writer.start()
    assert {ready.get(timeout=20), ready.get(timeout=20)} == {"a", "b"}
    start.set()
    observed = [results.get(timeout=20), results.get(timeout=20)]
    release.set()
    for writer in writers:
        writer.join(20)
        if writer.is_alive():
            writer.terminate()
            writer.join(5)
        assert writer.exitcode == 0

    assert sum(item["result"] == "committed" for item in observed) == 1
    assert {item["result"] for item in observed}.issubset(
        {"committed", "lease_busy", "request_in_progress"}
    )
    manifest_store, switch_store = _open_switch_store(root)
    active = [item for item in switch_store.list_envelopes() if item.active_request is not None]
    assert len(active) == 1
    assert active[0].request_fingerprint_sha256 in {"a" * 64, "b" * 64}
    _close_switch_store(manifest_store)


@pytest.mark.anyio
@pytest.mark.parametrize("corrupt_manifest", [False, True])
@pytest.mark.parametrize("terminalized", [False, True])
async def test_marker_bound_to_missing_or_corrupt_manifest_fails_closed(
    tmp_path: Path,
    corrupt_manifest: bool,
    terminalized: bool,
) -> None:
    root = tmp_path / (
        f"{'corrupt' if corrupt_manifest else 'missing'}-{'terminal' if terminalized else 'active'}"
    )
    manifest_store, switch_store = _open_switch_store(root)
    envelope = _envelope(view=_batch_bound_view())
    switch_store.write_envelope(envelope, previous=None)
    marker = _batch_bound_marker()
    switch_store.write_marker(marker, previous=None)
    if terminalized:
        switch_store.terminalize(
            envelope,
            _tombstone(finalization_id=FINALIZATION_A),
        )
    _close_switch_store(manifest_store)
    if corrupt_manifest:
        batch_root = root / "batches" / BATCH_A
        batch_root.mkdir(parents=True)
        (batch_root / "manifest.json").write_text("{not-json", encoding="utf-8")

    controller = GenerationController(
        FileManifestStore(root, fsync_writes=False),
        FakeInferenceAdapter(),
        max_attempts=3,
        retry_delay_seconds=0,
        runtime_metadata=_runtime_metadata(),
        data_root=root,
    )
    await controller.initialize()
    try:
        status = await controller.status(Principal("lakshman", "Lakshman"), ready=True)
        assert status.permissions.can_create is False
        assert status.permissions.can_switch is False
        assert status.permissions.switch_block_code == "gpu_switch_store_corrupt"
        assert controller.gpu_switch.store.read_marker() == marker
        persisted = controller.gpu_switch.store.read_envelope(SWITCH_A)
        assert persisted is not None
        assert (persisted.terminal_tombstone is not None) is terminalized
        assert controller.store.active_lease_held is True
    finally:
        await controller.shutdown()


@pytest.mark.anyio
async def test_multi_session_consent_is_principal_deduplicated_and_replay_is_noop(
    tmp_path: Path,
) -> None:
    async with worker_client(tmp_path / "volume", runtime_metadata=_runtime_metadata()) as (
        client,
        app,
        _,
    ):
        await _heartbeat(client, A_SESSION)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        await _heartbeat(client, B_SESSION_2, token=TOKEN_B)
        created = await client.post(
            "/v1/studio/gpu-switches", headers=auth(), json=_create_request()
        )
        assert created.status_code == 201, created.text
        assert created.json()["requester_user_id"] == "lakshman"
        assert created.json()["request"]["state"] == "pending"
        assert len(created.json()["request"]["waiting_for"]) == 1

        alternate_peer = await client.get(
            f"/v1/studio/sessions/{B_SESSION_2}", headers=auth(TOKEN_B)
        )
        assert alternate_peer.status_code == 200, alternate_peer.text
        assert alternate_peer.json()["gpu_switch_request"]["waiting_for"] == [
            {"session_id": B_SESSION, "display_name": "Sujal"}
        ]
        assert alternate_peer.json()["gpu_switch_can_respond"] is True

        requester_view = await client.get(
            f"/v1/studio/sessions/{A_SESSION}", headers=auth()
        )
        assert requester_view.status_code == 200, requester_view.text
        assert requester_view.json()["gpu_switch_can_respond"] is False

        approved = await client.post(
            f"/v1/studio/gpu-switches/{SWITCH_A}/responses",
            headers=auth(TOKEN_B),
            json={"schema_version": 1, "session_id": B_SESSION_2, "decision": "approve"},
        )
        assert approved.status_code == 200, approved.text
        switch = approved.json()["gpu_switch_request"]
        assert switch["state"] == "approved"
        assert approved.json()["gpu_switch_can_respond"] is False
        assert switch["approved_by"] == [{"session_id": B_SESSION_2, "display_name": "Sujal"}]
        envelope = app.state.runtime.controller.gpu_switch.store.read_envelope(SWITCH_A)
        assert envelope is not None
        revision = envelope.envelope_revision

        replay = await client.post(
            f"/v1/studio/gpu-switches/{SWITCH_A}/responses",
            headers=auth(TOKEN_B),
            json={"schema_version": 1, "session_id": B_SESSION, "decision": "approve"},
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["gpu_switch_request"]["approved_by"][0]["session_id"] == (B_SESSION_2)
        persisted = app.state.runtime.controller.gpu_switch.store.read_envelope(SWITCH_A)
        assert persisted.envelope_revision == revision

        conflict = await client.post(
            f"/v1/studio/gpu-switches/{SWITCH_A}/responses",
            headers=auth(TOKEN_B),
            json={"schema_version": 1, "session_id": B_SESSION, "decision": "deny"},
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "gpu_switch_response_conflict"
        assert conflict.json()["error"]["details"] is None


@pytest.mark.anyio
async def test_pending_switch_parks_queue_but_foreground_generation_terminalizes(
    tmp_path: Path,
) -> None:
    async with worker_client(tmp_path / "volume", runtime_metadata=_runtime_metadata()) as (
        client,
        _,
        _,
    ):
        await _heartbeat(client, A_SESSION)
        await _heartbeat(client, B_SESSION, token=TOKEN_B)
        created = await client.post(
            "/v1/studio/gpu-switches", headers=auth(), json=_create_request()
        )
        assert created.status_code == 201

        queued = await client.post(
            "/v1/batches",
            headers=auth(),
            json={
                "client_submission_id": "90000000-0000-4000-8000-000000000001",
                "admission_mode": "queue",
                "prompts": ["must remain local"],
            },
        )
        assert queued.status_code == 423
        assert queued.json()["error"] == {
            "code": "queue_switch_pending",
            "message": "A GPU switch is awaiting consent; the local queue remains parked.",
            "details": None,
        }

        foreground = await client.post(
            "/v1/batches",
            headers=auth(),
            json={
                "client_submission_id": "90000000-0000-4000-8000-000000000002",
                "admission_mode": "foreground",
                "prompts": ["foreground wins before finalization"],
            },
        )
        assert foreground.status_code == 201, foreground.text
        await wait_for_batch(client, foreground.json()["batch_id"], state="completed")
        owner = await client.get(
            f"/v1/internal/gpu-switches/{SWITCH_A}/owner",
            headers=auth(),
            params={"session_id": A_SESSION},
        )
        assert owner.status_code == 200
        assert owner.json()["requester_user_id"] == "lakshman"
        assert owner.json()["state"] == "cancelled"
        assert owner.json()["terminal_tombstone_sha256"] is not None


@pytest.mark.anyio
async def test_finalize_finishes_only_current_image_and_persists_pause_fixed_point(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    adapter = FakeInferenceAdapter(
        first_generation_started=started, release_first_generation=release
    )
    async with worker_client(
        tmp_path / "volume",
        adapter,
        runtime_metadata=_runtime_metadata(),
    ) as (client, _, _):
        await _heartbeat(client, A_SESSION)
        batch = await client.post(
            "/v1/batches", headers=auth(), json={"prompts": ["one", "two", "three"]}
        )
        assert batch.status_code == 201
        await asyncio.wait_for(started.wait(), timeout=2)

        switch = await client.post(
            "/v1/studio/gpu-switches",
            headers=auth(),
            json=_create_request(expected_batch_id=batch.json()["batch_id"]),
        )
        assert switch.status_code == 201
        assert switch.json()["request"]["state"] == "approved"
        finalized = await client.post(
            f"/v1/studio/gpu-switches/{SWITCH_A}/finalize",
            headers=auth(),
            json={
                "schema_version": 1,
                "session_id": A_SESSION,
                "finalization_id": FINALIZATION_A,
            },
        )
        assert finalized.status_code == 200, finalized.text
        assert finalized.json()["gpu_switch_request"]["state"] == "pausing"
        release.set()
        paused = await wait_for_batch(client, batch.json()["batch_id"], state="paused")
        assert paused["progress"]["completed"] == 1
        assert adapter.generated_indices == [1]
        state = await _heartbeat(client, A_SESSION)
        assert state["gpu_switch_request"]["state"] == "ready_to_delete"
        assert state["gpu_switch_request"]["ready_to_delete_at"] is not None


@pytest.mark.anyio
async def test_finalize_during_retry_delay_finishes_exact_retry_before_pause(
    tmp_path: Path,
) -> None:
    adapter = FakeInferenceAdapter(failures_before_success={1: 1})
    async with worker_client(
        tmp_path / "volume",
        adapter,
        runtime_metadata=_runtime_metadata(),
    ) as (client, app, _):
        app.state.runtime.controller.retry_delay_seconds = 0.5
        await _heartbeat(client, A_SESSION)
        batch = await client.post("/v1/batches", headers=auth(), json={"prompts": ["one", "two"]})
        assert batch.status_code == 201
        batch_id = batch.json()["batch_id"]
        deadline = asyncio.get_running_loop().time() + 2
        while True:
            manifest = app.state.runtime.controller.store.load(batch_id)
            if manifest.images[0].status.value == "retrying":
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("first image never entered retry delay")
            await asyncio.sleep(0.005)

        switch = await client.post(
            "/v1/studio/gpu-switches",
            headers=auth(),
            json=_create_request(expected_batch_id=batch_id),
        )
        assert switch.status_code == 201
        finalized = await client.post(
            f"/v1/studio/gpu-switches/{SWITCH_A}/finalize",
            headers=auth(),
            json={
                "schema_version": 1,
                "session_id": A_SESSION,
                "finalization_id": FINALIZATION_A,
            },
        )
        assert finalized.status_code == 200, finalized.text
        assert finalized.json()["gpu_switch_request"]["state"] == "pausing"
        during_delay = app.state.runtime.controller.store.load(batch_id)
        assert during_delay.state.value == "running"
        assert during_delay.images[0].status.value == "retrying"

        paused = await wait_for_batch(client, batch_id, state="paused")
        assert paused["images"][0]["status"] == "ready"
        assert paused["images"][1]["status"] == "pending"
        assert adapter.calls_by_index == {1: 2}
        state = await _heartbeat(client, A_SESSION)
        assert state["gpu_switch_request"]["state"] == "ready_to_delete"


@pytest.mark.anyio
async def test_pause_timeout_is_durable_attention_until_explicit_finalize_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert gpu_switch_module.GPU_SWITCH_PAUSE_TTL_SECONDS == 15 * 60
    started = asyncio.Event()
    release = asyncio.Event()
    adapter = FakeInferenceAdapter(
        first_generation_started=started,
        release_first_generation=release,
    )
    async with worker_client(
        tmp_path / "volume",
        adapter,
        runtime_metadata=_runtime_metadata(),
    ) as (client, app, _):
        await _heartbeat(client, A_SESSION)
        batch = await client.post("/v1/batches", headers=auth(), json={"prompts": ["one", "two"]})
        assert batch.status_code == 201
        await asyncio.wait_for(started.wait(), timeout=2)
        switch = await client.post(
            "/v1/studio/gpu-switches",
            headers=auth(),
            json=_create_request(expected_batch_id=batch.json()["batch_id"]),
        )
        assert switch.status_code == 201
        finalize_body = {
            "schema_version": 1,
            "session_id": A_SESSION,
            "finalization_id": FINALIZATION_A,
        }
        finalized = await client.post(
            f"/v1/studio/gpu-switches/{SWITCH_A}/finalize",
            headers=auth(),
            json=finalize_body,
        )
        assert finalized.status_code == 200

        monkeypatch.setattr(gpu_switch_module, "GPU_SWITCH_PAUSE_TTL_SECONDS", 0)
        attention = await _heartbeat(client, A_SESSION)
        assert attention["gpu_switch_request"]["state"] == "needs_attention"
        assert attention["gpu_switch_request"]["reason"] == "pause_failed"
        marker = app.state.runtime.controller.gpu_switch.store.read_marker()
        assert marker is not None and marker.phase == "pausing"
        revision = app.state.runtime.controller.gpu_switch.store.read_envelope(
            SWITCH_A
        ).envelope_revision
        repeated = await _heartbeat(client, A_SESSION)
        assert repeated["gpu_switch_request"]["state"] == "needs_attention"
        assert (
            app.state.runtime.controller.gpu_switch.store.read_envelope(SWITCH_A).envelope_revision
            == revision
        )

        monkeypatch.setattr(
            gpu_switch_module,
            "GPU_SWITCH_PAUSE_TTL_SECONDS",
            15 * 60,
        )
        resumed = await client.post(
            f"/v1/studio/gpu-switches/{SWITCH_A}/finalize",
            headers=auth(),
            json=finalize_body,
        )
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["gpu_switch_request"]["state"] == "pausing"
        release.set()
        await wait_for_batch(client, batch.json()["batch_id"], state="paused")
        ready = await _heartbeat(client, A_SESSION)
        assert ready["gpu_switch_request"]["state"] == "ready_to_delete"


@pytest.mark.anyio
async def test_standby_takeover_runs_full_marker_and_manifest_adoption(
    tmp_path: Path,
) -> None:
    volume = tmp_path / "volume"
    started = asyncio.Event()
    adapter = FakeInferenceAdapter(
        first_generation_started=started,
        release_first_generation=asyncio.Event(),
    )
    async with worker_client(
        volume,
        adapter,
        runtime_metadata=_runtime_metadata(),
    ) as (client_a, app_a, _):
        await _heartbeat(client_a, A_SESSION)
        batch = await client_a.post("/v1/batches", headers=auth(), json={"prompts": ["one", "two"]})
        assert batch.status_code == 201
        await asyncio.wait_for(started.wait(), timeout=2)
        created = await client_a.post(
            "/v1/studio/gpu-switches",
            headers=auth(),
            json=_create_request(expected_batch_id=batch.json()["batch_id"]),
        )
        assert created.status_code == 201
        finalized = await client_a.post(
            f"/v1/studio/gpu-switches/{SWITCH_A}/finalize",
            headers=auth(),
            json={
                "schema_version": 1,
                "session_id": A_SESSION,
                "finalization_id": FINALIZATION_A,
            },
        )
        assert finalized.status_code == 200

        async with worker_client(
            volume,
            runtime_metadata=_runtime_metadata(),
        ) as (client_b, app_b, _):
            assert app_b.state.runtime.controller.store.active_lease_held is False
            await app_a.state.runtime.controller.shutdown()
            recovered = await _heartbeat(client_b, A_SESSION_2)
            assert recovered["active_batch"]["state"] == "interrupted"
            assert recovered["gpu_switch_request"]["state"] == "pausing"
            controller_b = app_b.state.runtime.controller
            assert controller_b.store.active_lease_held is True
            assert controller_b.gpu_switch.store.read_marker() is not None
            persisted = controller_b.gpu_switch.store.read_envelope(SWITCH_A)
            assert persisted is not None and persisted.active_request is not None


@pytest.mark.anyio
async def test_standby_takeover_expires_pre_marker_consent_and_writes_tombstone(
    tmp_path: Path,
) -> None:
    volume = tmp_path / "volume"
    started = asyncio.Event()
    adapter = FakeInferenceAdapter(
        first_generation_started=started,
        release_first_generation=asyncio.Event(),
    )
    async with worker_client(
        volume,
        adapter,
        runtime_metadata=_runtime_metadata(),
    ) as (client_a, app_a, _):
        await _heartbeat(client_a, A_SESSION)
        batch = await client_a.post("/v1/batches", headers=auth(), json={"prompts": ["one", "two"]})
        assert batch.status_code == 201
        await asyncio.wait_for(started.wait(), timeout=2)
        created = await client_a.post(
            "/v1/studio/gpu-switches",
            headers=auth(),
            json=_create_request(expected_batch_id=batch.json()["batch_id"]),
        )
        assert created.status_code == 201

        async with worker_client(
            volume,
            runtime_metadata=_runtime_metadata(),
        ) as (client_b, app_b, _):
            await app_a.state.runtime.controller.shutdown()
            recovered = await _heartbeat(client_b, A_SESSION_2)
            assert recovered["active_batch"]["state"] == "interrupted"
            assert recovered["gpu_switch_request"] is None
            owner = await client_b.get(
                f"/v1/internal/gpu-switches/{SWITCH_A}/owner",
                headers=auth(),
                params={"session_id": A_SESSION_2},
            )
            assert owner.status_code == 200, owner.text
            assert owner.json()["state"] == "expired"
            assert owner.json()["terminal_tombstone_sha256"] is not None
            persisted = app_b.state.runtime.controller.gpu_switch.store.read_envelope(SWITCH_A)
            assert persisted is not None
            assert persisted.terminal_tombstone is not None
            assert persisted.terminal_tombstone.terminal_reason == "requester_expired"


@pytest.mark.anyio
async def test_idle_pre_marker_envelope_expires_on_mutation_lease_takeover(
    tmp_path: Path,
) -> None:
    volume = tmp_path / "volume"
    async with worker_client(volume, runtime_metadata=_runtime_metadata()) as (
        client_a,
        app_a,
        _,
    ):
        async with worker_client(volume, runtime_metadata=_runtime_metadata()) as (
            client_b,
            app_b,
            _,
        ):
            await _heartbeat(client_a, A_SESSION)
            created = await client_a.post(
                "/v1/studio/gpu-switches",
                headers=auth(),
                json=_create_request(),
            )
            assert created.status_code == 201, created.text
            assert created.json()["request"]["state"] == "approved"
            assert app_a.state.runtime.controller.store.active_lease_held is False
            before = app_b.state.runtime.controller.gpu_switch.store.read_envelope(SWITCH_A)
            assert before is not None and before.active_request is not None

            takeover = await _heartbeat(client_b, A_SESSION_2)
            assert takeover["gpu_switch_request"] is None
            persisted = app_b.state.runtime.controller.gpu_switch.store.read_envelope(SWITCH_A)
            assert persisted is not None and persisted.active_request is None
            assert persisted.terminal_tombstone is not None
            assert persisted.terminal_tombstone.terminal_state == "expired"
            assert persisted.terminal_tombstone.terminal_reason == "requester_expired"
            assert (
                app_b.state.runtime.controller.gpu_switch.store.read_tombstone(SWITCH_A)
                == persisted.terminal_tombstone
            )


@pytest.mark.anyio
async def test_direct_settle_envelope_crash_recovers_without_uuid_reuse(tmp_path: Path) -> None:
    volume = tmp_path / "volume"
    async with worker_client(volume, runtime_metadata=_runtime_metadata()) as (client, app, _):
        await _heartbeat(client, A_SESSION)

        def crash(point: str) -> None:
            if point == "gpu_switch_envelope_directory_fsync":
                raise RuntimeError("simulated settle response loss")

        app.state.runtime.controller.gpu_switch.store.crash_hook = crash
        settled = await client.post(
            f"/v1/internal/gpu-switches/{SWITCH_A}/settle-create",
            headers=auth(),
            json={
                "schema_version": 1,
                "action": "cancel",
                "create_request": _create_request(),
            },
        )
        assert settled.status_code == 500

    async with worker_client(volume, runtime_metadata=_runtime_metadata()) as (client, _, _):
        await _heartbeat(client, A_SESSION)
        owner = await client.get(
            f"/v1/internal/gpu-switches/{SWITCH_A}/owner",
            headers=auth(),
            params={"session_id": A_SESSION},
        )
        assert owner.status_code == 200, owner.text
        assert owner.json()["state"] == "cancelled"
        replayed_create = await client.post(
            "/v1/studio/gpu-switches", headers=auth(), json=_create_request()
        )
        assert replayed_create.status_code == 404
        assert replayed_create.json()["error"]["code"] == "gpu_switch_request_not_found"
