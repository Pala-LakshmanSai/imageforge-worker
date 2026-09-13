from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from imageforge_worker.auth import Principal
from imageforge_worker.controller import GenerationController
from imageforge_worker.domain import (
    BatchManifest,
    BatchOwner,
    BatchProgress,
    BatchState,
    ImageRecord,
    ImageState,
    utc_now,
)
from imageforge_worker.inference import FakeInferenceAdapter
from imageforge_worker.persistence import FileManifestStore


def _manifest(state: BatchState, *, batch_id: str | None = None) -> BatchManifest:
    now = utc_now()
    if state == BatchState.CANCELLED:
        image_state = ImageState.CANCELLED
        progress = BatchProgress(total=1, cancelled=1, processed=1)
    elif state == BatchState.COMPLETED:
        image_state = ImageState.FAILED
        progress = BatchProgress(total=1, failed=1, processed=1)
    else:
        image_state = ImageState.PENDING
        progress = BatchProgress(total=1)
    return BatchManifest(
        batch_id=batch_id or str(uuid.uuid4()),
        owner=BatchOwner(user_id="lakshman", display_name="Lakshman"),
        state=state,
        created_at=now,
        updated_at=now,
        images=[ImageRecord(index=1, prompt="frame", seed=1, status=image_state)],
        progress=progress,
    )


def _seed_history(root: Path, count: int) -> list[BatchManifest]:
    store = FileManifestStore(root, fsync_writes=False)
    store.initialize()
    assert store.try_acquire_active_lease()
    manifests = [_manifest(BatchState.CANCELLED) for _ in range(count)]
    for manifest in manifests:
        store.create(manifest)
    store.release_active_lease()
    return manifests


def _controller(store: FileManifestStore) -> GenerationController:
    return GenerationController(
        store,
        FakeInferenceAdapter(),
        max_attempts=3,
        retry_delay_seconds=0,
    )


@pytest.mark.anyio
async def test_marker_migration_makes_85_manifest_boot_and_status_o1(tmp_path: Path) -> None:
    root = tmp_path / "volume"
    _seed_history(root, 85)

    migrating_store = FileManifestStore(root, fsync_writes=False)
    migrating = _controller(migrating_store)
    await migrating.initialize()
    assert migrating_store.volume_manifest_reads == 85
    await migrating.shutdown()

    warm_store = FileManifestStore(root, fsync_writes=False)
    warm = _controller(warm_store)
    await warm.initialize()
    assert warm_store.volume_manifest_reads == 0

    principal = Principal(user_id="lakshman", display_name="Lakshman")
    status = await warm.status(principal, ready=True)
    assert status.active_batch is None
    assert warm_store.volume_manifest_reads == 0
    await warm.shutdown()


@pytest.mark.anyio
async def test_active_pointer_reads_only_referenced_manifest_and_recovers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "volume"
    _seed_history(root, 85)
    seeded = FileManifestStore(root, fsync_writes=False)
    seeded.initialize()
    assert seeded.try_acquire_active_lease()
    active = _manifest(BatchState.RUNNING)
    seeded.write_active_batch_index(active.batch_id)
    seeded.create(active)
    seeded.release_active_lease()

    store = FileManifestStore(root, fsync_writes=False)
    controller = _controller(store)
    await controller.initialize()
    assert store.volume_manifest_reads == 1
    recovered = store.load(active.batch_id)
    assert recovered.state == BatchState.INTERRUPTED
    assert store.read_active_batch_index() == active.batch_id
    await controller.shutdown()


@pytest.mark.anyio
@pytest.mark.parametrize("damage", ["missing", "corrupt", "stale"])
async def test_untrusted_active_pointer_scans_and_repairs_safely(
    tmp_path: Path, damage: str
) -> None:
    root = tmp_path / "volume"
    history = _seed_history(root, 3)
    marker = root / ".active-batch.json"
    if damage == "corrupt":
        marker.write_bytes(b"{not-json")
    elif damage == "stale":
        store = FileManifestStore(root, fsync_writes=False)
        store.initialize()
        assert store.try_acquire_active_lease()
        store.write_active_batch_index(str(uuid.uuid4()))
        store.release_active_lease()

    store = FileManifestStore(root, fsync_writes=False)
    controller = _controller(store)
    await controller.initialize()
    assert store.volume_manifest_reads == len(history)
    assert store.read_active_batch_index() is None
    await controller.shutdown()


@pytest.mark.anyio
async def test_stale_terminal_pointer_cannot_hide_another_active_batch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "volume"
    store = FileManifestStore(root, fsync_writes=False)
    store.initialize()
    assert store.try_acquire_active_lease()
    terminal = _manifest(BatchState.CANCELLED)
    active = _manifest(BatchState.PAUSED)
    store.create(terminal)
    store.create(active)
    store.write_active_batch_index(terminal.batch_id)
    store.release_active_lease()

    restarted = FileManifestStore(root, fsync_writes=False)
    controller = _controller(restarted)
    await controller.initialize()
    assert restarted.read_active_batch_index() == active.batch_id
    principal = Principal(user_id="lakshman", display_name="Lakshman")
    status = await controller.status(principal, ready=True)
    assert status.active_batch is not None
    assert status.active_batch.batch_id == active.batch_id
    await controller.shutdown()


@pytest.mark.anyio
async def test_markerless_duplicate_active_history_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "volume"
    store = FileManifestStore(root, fsync_writes=False)
    store.initialize()
    assert store.try_acquire_active_lease()
    store.create(_manifest(BatchState.PAUSED))
    store.create(_manifest(BatchState.INTERRUPTED))
    store.release_active_lease()

    restarted = FileManifestStore(root, fsync_writes=False)
    controller = _controller(restarted)
    with pytest.raises(RuntimeError, match="multiple active batch leases"):
        await controller.initialize()
    assert not restarted.active_lease_held


@pytest.mark.anyio
async def test_pointer_written_before_manifest_is_repaired_after_crash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "volume"
    store = FileManifestStore(root, fsync_writes=False)
    store.initialize()
    assert store.try_acquire_active_lease()
    store.write_active_batch_index(str(uuid.uuid4()))
    store.release_active_lease()

    restarted = FileManifestStore(root, fsync_writes=False)
    controller = _controller(restarted)
    await controller.initialize()
    assert restarted.read_active_batch_index() is None
    await controller.shutdown()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "crash_point", ["after_active_index_fsync", "after_active_index_rename"]
)
async def test_active_pointer_atomic_write_crash_seams_are_safe(
    tmp_path: Path, crash_point: str
) -> None:
    root = tmp_path / "volume"
    baseline = FileManifestStore(root, fsync_writes=False)
    baseline.initialize()
    assert baseline.try_acquire_active_lease()
    baseline.write_active_batch_index(None)
    baseline.release_active_lease()

    def crash(point: str) -> None:
        if point == crash_point:
            raise OSError(f"injected {point}")

    faulted = FileManifestStore(root, fsync_writes=False, crash_hook=crash)
    faulted.initialize()
    assert faulted.try_acquire_active_lease()
    with pytest.raises(OSError, match="injected"):
        faulted.write_active_batch_index(str(uuid.uuid4()))
    faulted.release_active_lease()

    restarted = FileManifestStore(root, fsync_writes=False)
    controller = _controller(restarted)
    await controller.initialize()
    assert restarted.read_active_batch_index() is None
    await controller.shutdown()


@pytest.mark.anyio
async def test_opening_terminal_history_cannot_displace_active_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "volume"
    seeded = FileManifestStore(root, fsync_writes=False)
    seeded.initialize()
    assert seeded.try_acquire_active_lease()
    historical = _manifest(BatchState.COMPLETED)
    active = _manifest(BatchState.PAUSED)
    seeded.create(historical)
    seeded.create(active)
    seeded.write_active_batch_index(active.batch_id)
    seeded.release_active_lease()

    store = FileManifestStore(root, fsync_writes=False)
    controller = _controller(store)
    await controller.initialize()

    def forbidden_recovery(_: BatchManifest) -> bool:
        raise AssertionError("terminal recovery ran while another batch was active")

    monkeypatch.setattr(controller, "_recover_manifest", forbidden_recovery)
    principal = Principal(user_id="lakshman", display_name="Lakshman")
    opened = await controller.get_batch(principal, historical.batch_id)
    assert opened.batch_id == historical.batch_id
    assert store.read_active_batch_index() == active.batch_id
    await controller.shutdown()
