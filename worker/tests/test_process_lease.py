from __future__ import annotations

import asyncio
import multiprocessing
import queue
import time
from pathlib import Path

from imageforge_worker.auth import Principal
from imageforge_worker.cleanup_retention import cleanup_retained_artifacts
from imageforge_worker.controller import GenerationController
from imageforge_worker.domain import CreateBatchRequest
from imageforge_worker.errors import WorkerError
from imageforge_worker.inference import FakeInferenceAdapter, GenerationJob
from imageforge_worker.persistence import FileManifestStore


def _controller(root: Path, adapter: FakeInferenceAdapter) -> GenerationController:
    return GenerationController(
        FileManifestStore(root, fsync_writes=False),
        adapter,
        max_attempts=3,
        retry_delay_seconds=0,
    )


def _contention_child(root, user_id, display_name, barrier, release, results) -> None:
    async def run() -> None:
        controller = _controller(Path(root), FakeInferenceAdapter())
        await controller.initialize()
        barrier.wait()
        try:
            manifest = await controller.create_batch(
                Principal(user_id, display_name),
                CreateBatchRequest(prompts=[f"from {display_name}"]),
            )
        except WorkerError as exc:
            results.put(
                {
                    "user_id": user_id,
                    "status": exc.status_code,
                    "code": exc.code,
                    "details": dict(exc.details or {}),
                }
            )
        else:
            results.put(
                {
                    "user_id": user_id,
                    "status": 201,
                    "batch_id": manifest.batch_id,
                }
            )
            release.wait()
        finally:
            await controller.shutdown()

    asyncio.run(run())


class _BlockSecondGeneration(FakeInferenceAdapter):
    async def generate(self, job: GenerationJob):
        if job.index == 2:
            await asyncio.Event().wait()
        return await super().generate(job)


def _active_owner_child(root, results) -> None:
    async def run() -> None:
        controller = _controller(Path(root), _BlockSecondGeneration())
        await controller.initialize()
        manifest = await controller.create_batch(
            Principal("lakshman", "Lakshman"),
            CreateBatchRequest(prompts=["durable first", "kill during second"]),
        )
        results.put({"batch_id": manifest.batch_id})
        await asyncio.Event().wait()

    asyncio.run(run())


def _lease_holder_child(root, ready, release) -> None:
    store = FileManifestStore(Path(root), fsync_writes=False)
    store.initialize()
    if not store.try_acquire_active_lease():
        raise RuntimeError("test lease holder could not acquire the guard")
    ready.set()
    try:
        release.wait()
    finally:
        store.release_active_lease()


def _idle_worker_child(root, ready, release, results) -> None:
    async def run() -> None:
        controller = _controller(Path(root), FakeInferenceAdapter())
        await controller.initialize()
        results.put({"active_lease_held": controller.store.active_lease_held})
        ready.set()
        try:
            release.wait()
        finally:
            await controller.shutdown()

    asyncio.run(run())


def _duplicate_observer_child(root, batch_id, results) -> None:
    async def run() -> None:
        controller = _controller(Path(root), FakeInferenceAdapter())
        await controller.initialize()
        principal = Principal("lakshman", "Lakshman")
        manifest = await controller.get_batch(principal, batch_id)
        status = await controller.status(principal, ready=True)
        try:
            await controller.pause(principal, batch_id)
        except WorkerError as exc:
            mutation_status = exc.status_code
            mutation_code = exc.code
        else:
            mutation_status = 200
            mutation_code = "unexpected_success"
        results.put(
            {
                "state": manifest.state.value,
                "images": [image.status.value for image in manifest.images],
                "can_manage": status.permissions.can_manage_active,
                "mutation_status": mutation_status,
                "mutation_code": mutation_code,
            }
        )
        await controller.shutdown()

    asyncio.run(run())


def _recovery_child(root, batch_id, results) -> None:
    async def run() -> None:
        adapter = FakeInferenceAdapter()
        controller = _controller(Path(root), adapter)
        await controller.initialize()
        principal = Principal("lakshman", "Lakshman")
        interrupted = await controller.get_batch(principal, batch_id)
        results.put(
            {
                "phase": "interrupted",
                "state": interrupted.state.value,
                "images": [image.status.value for image in interrupted.images],
            }
        )
        await controller.resume(principal, batch_id)
        deadline = asyncio.get_running_loop().time() + 20
        while True:
            manifest = await controller.get_batch(principal, batch_id)
            if manifest.state.value == "completed":
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("recovered batch did not complete")
            await asyncio.sleep(0.01)
        results.put(
            {
                "phase": "completed",
                "state": manifest.state.value,
                "images": [image.status.value for image in manifest.images],
                "generated_indices": adapter.generated_indices,
            }
        )
        await controller.shutdown()

    asyncio.run(run())


def _persistent_observer_child(root, batch_id, ready, release, results) -> None:
    async def run() -> None:
        controller = _controller(Path(root), FakeInferenceAdapter())
        await controller.initialize()
        principal = Principal("lakshman", "Lakshman")
        initial = await controller.status(principal, ready=True)
        results.put({
            "phase": "observing",
            "state": initial.active_batch.state.value if initial.active_batch else None,
            "can_manage": initial.permissions.can_manage_active,
        })
        ready.set()
        deadline = asyncio.get_running_loop().time() + 20
        while True:
            status = await controller.status(principal, ready=True)
            if status.active_batch is not None and status.active_batch.state.value == "interrupted":
                results.put({
                    "phase": "recovered",
                    "state": status.active_batch.state.value,
                    "can_manage": status.permissions.can_manage_active,
                })
                await controller.resume(principal, batch_id)
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("already-running standby did not adopt the interrupted batch")
            await asyncio.sleep(0.01)
        while True:
            manifest = await controller.get_batch(principal, batch_id)
            if manifest.state.value == "completed":
                results.put({"phase": "completed", "state": manifest.state.value})
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("standby-recovered batch did not complete")
            await asyncio.sleep(0.01)
        release.wait()
        await controller.shutdown()

    asyncio.run(run())


def _queue_result(results, *, timeout: float = 20) -> dict:
    try:
        return results.get(timeout=timeout)
    except queue.Empty as exc:
        raise AssertionError("subprocess did not report a result") from exc


def _join_or_terminate(process, *, timeout: float = 10) -> None:
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise AssertionError("subprocess did not exit")


def test_two_processes_sharing_a_volume_get_one_atomic_batch_winner(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    release = context.Event()
    results = context.Queue()
    root = tmp_path / "shared-volume"
    processes = [
        context.Process(
            target=_contention_child,
            args=(root, "lakshman", "Lakshman", barrier, release, results),
        ),
        context.Process(
            target=_contention_child,
            args=(root, "sujal", "Sujal", barrier, release, results),
        ),
    ]
    for process in processes:
        process.start()
    try:
        outcomes = [_queue_result(results), _queue_result(results)]
        assert sorted(outcome["status"] for outcome in outcomes) == [201, 423]
        winner = next(outcome for outcome in outcomes if outcome["status"] == 201)
        loser = next(outcome for outcome in outcomes if outcome["status"] == 423)
        winning_name = "Lakshman" if winner["user_id"] == "lakshman" else "Sujal"
        assert loser["code"] == "batch_busy"
        assert loser["details"] == {"owner": winning_name, "completed": 0, "total": 1}
        batch_dirs = [path for path in (root / "batches").iterdir() if path.is_dir()]
        assert [path.name for path in batch_dirs] == [winner["batch_id"]]
    finally:
        release.set()
        for process in processes:
            _join_or_terminate(process)


def test_cleanup_maintenance_guard_refuses_while_worker_lease_is_live(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    root = tmp_path / "shared-volume"
    holder = context.Process(target=_lease_holder_child, args=(root, ready, release))
    holder.start()
    try:
        assert ready.wait(20), "lease holder did not become ready"
        try:
            cleanup_retained_artifacts(root)
        except RuntimeError as exc:
            assert "active-batch lease" in str(exc)
        else:
            raise AssertionError("cleanup acquired the guard while a worker lease was live")
    finally:
        release.set()
        _join_or_terminate(holder)


def test_idle_initialized_worker_presence_blocks_cleanup_until_process_exit(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    results = context.Queue()
    root = tmp_path / "shared-volume"
    worker = context.Process(
        target=_idle_worker_child,
        args=(root, ready, release, results),
    )
    worker.start()
    try:
        assert ready.wait(20), "idle worker did not become ready"
        assert _queue_result(results) == {"active_lease_held": False}
        try:
            cleanup_retained_artifacts(root)
        except RuntimeError as exc:
            assert "worker process is alive" in str(exc)
        else:
            raise AssertionError("cleanup ran while an initialized idle worker was alive")
    finally:
        release.set()
        _join_or_terminate(worker)

    cleaned = cleanup_retained_artifacts(root)
    assert cleaned.images_deleted == 0
    assert cleaned.files_deleted == 0
    assert cleaned.bytes_deleted == 0


def test_duplicate_boot_is_read_only_and_forced_kill_recovers_once(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    root = tmp_path / "shared-volume"
    owner = context.Process(target=_active_owner_child, args=(root, results))
    owner.start()
    batch_id = _queue_result(results)["batch_id"]
    store = FileManifestStore(root, fsync_writes=False)
    deadline = time.monotonic() + 20
    while True:
        active = store.load(batch_id)
        if [image.status.value for image in active.images] == ["ready", "generating"]:
            break
        if time.monotonic() >= deadline:
            owner.terminate()
            raise AssertionError("owner did not reach the forced-kill point")
        time.sleep(0.01)

    observer = context.Process(target=_duplicate_observer_child, args=(root, batch_id, results))
    observer.start()
    observed = _queue_result(results)
    _join_or_terminate(observer)
    assert observed == {
        "state": "running",
        "images": ["ready", "generating"],
        "can_manage": False,
        "mutation_status": 423,
        "mutation_code": "worker_standby",
    }
    unchanged = store.load(batch_id)
    assert unchanged.state.value == "running"
    assert [image.status.value for image in unchanged.images] == ["ready", "generating"]

    owner.kill()
    owner.join(10)
    assert not owner.is_alive()
    assert owner.exitcode is not None and owner.exitcode != 0

    recovery = context.Process(target=_recovery_child, args=(root, batch_id, results))
    recovery.start()
    interrupted = _queue_result(results)
    completed = _queue_result(results)
    _join_or_terminate(recovery)
    assert interrupted == {
        "phase": "interrupted",
        "state": "interrupted",
        "images": ["ready", "pending"],
    }
    assert completed == {
        "phase": "completed",
        "state": "completed",
        "images": ["ready", "ready"],
        "generated_indices": [2],
    }


def test_already_running_standby_adopts_after_owner_kill_and_can_resume(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    observer_ready = context.Event()
    release = context.Event()
    root = tmp_path / "shared-volume"
    owner = context.Process(target=_active_owner_child, args=(root, results))
    owner.start()
    batch_id = _queue_result(results)["batch_id"]
    store = FileManifestStore(root, fsync_writes=False)
    deadline = time.monotonic() + 20
    while True:
        active = store.load(batch_id)
        if [image.status.value for image in active.images] == ["ready", "generating"]:
            break
        if time.monotonic() >= deadline:
            owner.terminate()
            raise AssertionError("owner did not reach the forced-kill point")
        time.sleep(0.01)

    observer = context.Process(
        target=_persistent_observer_child,
        args=(root, batch_id, observer_ready, release, results),
    )
    observer.start()
    assert observer_ready.wait(20), "standby did not initialize"
    assert _queue_result(results) == {"phase": "observing", "state": "running", "can_manage": False}
    owner.kill()
    owner.join(10)
    assert not owner.is_alive()
    try:
        assert _queue_result(results) == {
            "phase": "recovered",
            "state": "interrupted",
            "can_manage": True,
        }
        assert _queue_result(results) == {"phase": "completed", "state": "completed"}
    finally:
        release.set()
        _join_or_terminate(observer)
