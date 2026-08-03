from __future__ import annotations

import asyncio
import json
import multiprocessing
import queue
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from imageforge_worker.auth import Principal
from imageforge_worker.cleanup_retention import cleanup_retained_artifacts
from imageforge_worker.controller import GenerationController
from imageforge_worker.coordination import (
    CancelStopRequest,
    CoordinationClock,
    CreateStopRequest,
    FinalizeStopRequest,
    HeartbeatRequest,
    StopResponseRequest,
)
from imageforge_worker.domain import CreateBatchRequest
from imageforge_worker.errors import WorkerError
from imageforge_worker.inference import FakeInferenceAdapter, GenerationJob
from imageforge_worker.persistence import FileManifestStore, SharedGpuStopGuard

SESSION_A = "30000000-0000-4000-8000-000000000001"
RESTART_SESSION = "30000000-0000-4000-8000-000000000002"
STANDBY_SESSION = "30000000-0000-4000-8000-000000000003"
STOP_REQUEST = "40000000-0000-4000-8000-000000000001"
FINALIZATION = "50000000-0000-4000-8000-000000000001"
OTHER_FINALIZATION = "50000000-0000-4000-8000-000000000002"


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _AdjustableCoordinationClock:
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


def _controller(
    root: Path,
    adapter: FakeInferenceAdapter,
    *,
    coordination_clock: CoordinationClock | None = None,
    finalization_ttl_seconds: int = 60,
) -> GenerationController:
    return GenerationController(
        FileManifestStore(root, fsync_writes=False),
        adapter,
        max_attempts=3,
        retry_delay_seconds=0,
        coordination_clock=coordination_clock,
        coordination_finalization_ttl_seconds=finalization_ttl_seconds,
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


def _stop_guard_holder_child(root, commands, results) -> None:
    async def run() -> None:
        clock = _AdjustableCoordinationClock()
        controller = _controller(
            Path(root),
            FakeInferenceAdapter(),
            coordination_clock=clock,
        )
        principal = Principal("lakshman", "Lakshman")
        try:
            await controller.initialize()
            await controller.studio_heartbeat(
                principal,
                SESSION_A,
                HeartbeatRequest(availability="foreground"),
            )
            await controller.request_gpu_stop(
                principal,
                CreateStopRequest(
                    request_id=STOP_REQUEST,
                    session_id=SESSION_A,
                    pod_id="pod-123",
                    gpu_display_name="NVIDIA RTX 4090",
                ),
            )
            finalized = await controller.finalize_gpu_stop(
                principal,
                STOP_REQUEST,
                FinalizeStopRequest(
                    session_id=SESSION_A,
                    finalization_id=FINALIZATION,
                ),
            )
            marker = controller.store.read_gpu_stop_guard()
            results.put(
                {
                    "phase": "ready",
                    "server_instance_id": finalized.server_instance_id,
                    "ttl": finalized.finalization_ttl_seconds,
                    "expires_at": marker.expires_at if marker is not None else None,
                    "lease": controller.store.active_lease_held,
                    "marker": marker is not None,
                }
            )

            while True:
                command = await asyncio.to_thread(commands.get)
                if command == "advance_16":
                    clock.advance(16)
                    await controller.studio_heartbeat(
                        principal,
                        SESSION_A,
                        HeartbeatRequest(availability="foreground"),
                    )
                    remaining = controller.coordination.finalization_remaining_seconds(
                        STOP_REQUEST, FINALIZATION
                    )
                    results.put(
                        {
                            "phase": "after_16",
                            "remaining": remaining,
                            "lease": controller.store.active_lease_held,
                            "marker": controller.store.read_gpu_stop_guard() is not None,
                        }
                    )
                elif command == "mismatch":
                    try:
                        await controller.cancel_gpu_stop(
                            principal,
                            STOP_REQUEST,
                            CancelStopRequest(
                                session_id=SESSION_A,
                                finalization_id=OTHER_FINALIZATION,
                            ),
                        )
                    except WorkerError as exc:
                        code = exc.code
                    else:
                        code = "unexpected_success"
                    results.put(
                        {
                            "phase": "mismatch",
                            "code": code,
                            "lease": controller.store.active_lease_held,
                            "marker": controller.store.read_gpu_stop_guard() is not None,
                        }
                    )
                elif command == "cancel":
                    response = await controller.cancel_gpu_stop(
                        principal,
                        STOP_REQUEST,
                        CancelStopRequest(
                            session_id=SESSION_A,
                            finalization_id=FINALIZATION,
                        ),
                    )
                    results.put(
                        {
                            "phase": "cancelled",
                            "state": response.stop_request.state
                            if response.stop_request is not None
                            else None,
                            "lease": controller.store.active_lease_held,
                            "marker": controller.store.read_gpu_stop_guard() is not None,
                        }
                    )
                elif command == "expire":
                    clock.advance(61)
                    status = await controller.status(principal, ready=True)
                    results.put(
                        {
                            "phase": "expired",
                            "can_create": status.permissions.can_create,
                            "lease": controller.store.active_lease_held,
                            "marker": controller.store.read_gpu_stop_guard() is not None,
                        }
                    )
                elif command == "exit":
                    return
                else:
                    raise RuntimeError(f"unknown child command: {command}")
        except BaseException as exc:
            results.put(
                {
                    "phase": "error",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            raise
        finally:
            if controller.initialized:
                await controller.shutdown()

    asyncio.run(run())


def _stop_guard_standby_child(root, ready, release, results) -> None:
    async def run() -> None:
        controller = _controller(Path(root), FakeInferenceAdapter())
        principal = Principal("sujal", "Sujal")
        await controller.initialize()
        try:
            studio = await controller.studio_heartbeat(
                principal,
                STANDBY_SESSION,
                HeartbeatRequest(availability="foreground"),
            )
            initial = await controller.status(principal, ready=True)
            stop = studio.stop_request
            results.put(
                {
                    "phase": "observing",
                    "can_create": initial.permissions.can_create,
                    "lease": controller.store.active_lease_held,
                    "stop": stop.model_dump(mode="json") if stop is not None else None,
                    "current_session_id": studio.current_session.session_id,
                }
            )
            ready.set()
            deadline = asyncio.get_running_loop().time() + 20
            while True:
                status = await controller.status(principal, ready=True)
                if controller.store.active_lease_held:
                    marker = controller.store.read_gpu_stop_guard()
                    studio = await controller.studio_state(principal, STANDBY_SESSION)
                    stop = studio.stop_request
                    results.put(
                        {
                            "phase": "adopted",
                            "can_create": status.permissions.can_create,
                            "lease": True,
                            "marker": marker is not None,
                            "stop": stop.model_dump(mode="json")
                            if stop is not None
                            else None,
                            "current_session_id": studio.current_session.session_id,
                        }
                    )
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError("standby did not adopt the released Stop guard")
                await asyncio.sleep(0.01)
            await asyncio.to_thread(release.wait)
        finally:
            await controller.shutdown()

    asyncio.run(run())


async def _wait_for_controller_batch(
    controller: GenerationController,
    principal: Principal,
    batch_id: str,
) -> object:
    deadline = asyncio.get_running_loop().time() + 20
    while True:
        manifest = await controller.get_batch(principal, batch_id)
        if manifest.state.value == "completed":
            return manifest
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("controller batch did not complete")
        await asyncio.sleep(0.01)


async def _seed_failed_batch(root: Path) -> str:
    controller = _controller(
        root,
        FakeInferenceAdapter(failures_before_success={1: 99}),
    )
    principal = Principal("lakshman", "Lakshman")
    await controller.initialize()
    try:
        created = await controller.create_batch(
            principal,
            CreateBatchRequest(prompts=["durable failed image"]),
        )
        completed = await _wait_for_controller_batch(
            controller, principal, created.batch_id
        )
        assert completed.images[0].status.value == "failed"
        return completed.batch_id
    finally:
        await controller.shutdown()


async def _attempt_guarded_generation(root: Path, batch_id: str) -> dict:
    controller = _controller(root, FakeInferenceAdapter())
    principal = Principal("lakshman", "Lakshman")
    await controller.initialize()
    try:
        status = await controller.status(principal, ready=True)
        try:
            await controller.create_batch(
                principal,
                CreateBatchRequest(prompts=["cross-process create must wait"]),
            )
        except WorkerError as exc:
            create = {"code": exc.code, "details": dict(exc.details or {})}
        else:
            create = {"code": "unexpected_success", "details": {}}
        try:
            await controller.retry_failed(principal, batch_id)
        except WorkerError as exc:
            retry = {"code": exc.code, "details": dict(exc.details or {})}
        else:
            retry = {"code": "unexpected_success", "details": {}}
        return {
            "can_create": status.permissions.can_create,
            "create": create,
            "retry": retry,
        }
    finally:
        await controller.shutdown()


async def _run_after_guard_release(root: Path, failed_batch_id: str | None = None) -> dict:
    controller = _controller(root, FakeInferenceAdapter())
    principal = Principal("lakshman", "Lakshman")
    await controller.initialize()
    try:
        retried_state = None
        if failed_batch_id is not None:
            retried = await controller.retry_failed(principal, failed_batch_id)
            retried = await _wait_for_controller_batch(
                controller, principal, retried.batch_id
            )
            retried_state = retried.state.value
        created = await controller.create_batch(
            principal,
            CreateBatchRequest(prompts=["generation after guard release"]),
        )
        created = await _wait_for_controller_batch(
            controller, principal, created.batch_id
        )
        return {"retried": retried_state, "created": created.state.value}
    finally:
        await controller.shutdown()


async def _inspect_recovered_guard(root: Path) -> dict:
    controller = _controller(root, FakeInferenceAdapter())
    principal = Principal("lakshman", "Lakshman")
    await controller.initialize()
    try:
        state = await controller.studio_heartbeat(
            principal,
            RESTART_SESSION,
            HeartbeatRequest(availability="foreground"),
        )
        status = await controller.status(principal, ready=True)
        try:
            await controller.create_batch(
                principal,
                CreateBatchRequest(prompts=["restart must retain stop guard"]),
            )
        except WorkerError as exc:
            create = {"code": exc.code, "details": dict(exc.details or {})}
        else:
            create = {"code": "unexpected_success", "details": {}}
        try:
            await controller.cancel_gpu_stop(
                principal,
                STOP_REQUEST,
                CancelStopRequest(
                    session_id=RESTART_SESSION,
                    finalization_id=FINALIZATION,
                ),
            )
        except WorkerError as exc:
            stale_cancel = exc.code
        else:
            stale_cancel = "unexpected_success"
        try:
            await controller.respond_to_gpu_stop(
                principal,
                STOP_REQUEST,
                StopResponseRequest(
                    session_id=RESTART_SESSION,
                    decision="approve",
                ),
            )
        except WorkerError as exc:
            stale_response = exc.code
        else:
            stale_response = "unexpected_success"
        try:
            await controller.finalize_gpu_stop(
                principal,
                STOP_REQUEST,
                FinalizeStopRequest(
                    session_id=RESTART_SESSION,
                    finalization_id=FINALIZATION,
                ),
            )
        except WorkerError as exc:
            stale_finalize = exc.code
        else:
            stale_finalize = "unexpected_success"
        return {
            "server_instance_id": state.server_instance_id,
            "current_session_id": state.current_session.session_id,
            "stop_request": state.stop_request.model_dump(mode="json")
            if state.stop_request is not None
            else None,
            "can_create": status.permissions.can_create,
            "lease": controller.store.active_lease_held,
            "create": create,
            "stale_cancel": stale_cancel,
            "stale_response": stale_response,
            "stale_finalize": stale_finalize,
        }
    finally:
        await controller.shutdown()


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


def test_shared_stop_guard_blocks_cross_process_create_and_retry_until_exact_cancel(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared-volume"
    failed_batch_id = asyncio.run(_seed_failed_batch(root))
    context = multiprocessing.get_context("spawn")
    commands = context.Queue()
    results = context.Queue()
    holder = context.Process(
        target=_stop_guard_holder_child,
        args=(root, commands, results),
    )
    holder.start()
    try:
        ready = _queue_result(results)
        assert ready["phase"] == "ready"
        assert ready["ttl"] == 60
        assert ready["lease"] is True
        assert ready["marker"] is True

        commands.put("advance_16")
        after_16 = _queue_result(results)
        assert after_16 == {
            "phase": "after_16",
            "remaining": 44.0,
            "lease": True,
            "marker": True,
        }

        blocked = asyncio.run(_attempt_guarded_generation(root, failed_batch_id))
        assert blocked["can_create"] is False
        for operation in ("create", "retry"):
            assert blocked[operation]["code"] == "gpu_stop_pending"
            assert blocked[operation]["details"] == {
                "request_id": STOP_REQUEST,
                "requester": "Lakshman",
                "expires_at": ready["expires_at"],
            }

        commands.put("mismatch")
        mismatch = _queue_result(results)
        assert mismatch == {
            "phase": "mismatch",
            "code": "finalization_mismatch",
            "lease": True,
            "marker": True,
        }
        still_blocked = asyncio.run(_attempt_guarded_generation(root, failed_batch_id))
        assert still_blocked["create"]["code"] == "gpu_stop_pending"
        assert still_blocked["retry"]["code"] == "gpu_stop_pending"

        commands.put("cancel")
        cancelled = _queue_result(results)
        assert cancelled == {
            "phase": "cancelled",
            "state": "cancelled",
            "lease": False,
            "marker": False,
        }
        assert asyncio.run(_run_after_guard_release(root, failed_batch_id)) == {
            "retried": "completed",
            "created": "completed",
        }
    finally:
        if holder.is_alive():
            commands.put("exit")
        _join_or_terminate(holder)


def test_shared_stop_guard_expiry_releases_marker_and_lease(tmp_path: Path) -> None:
    root = tmp_path / "shared-volume"
    context = multiprocessing.get_context("spawn")
    commands = context.Queue()
    results = context.Queue()
    holder = context.Process(
        target=_stop_guard_holder_child,
        args=(root, commands, results),
    )
    holder.start()
    try:
        ready = _queue_result(results)
        assert ready["phase"] == "ready"
        commands.put("expire")
        assert _queue_result(results) == {
            "phase": "expired",
            "can_create": True,
            "lease": False,
            "marker": False,
        }
        assert asyncio.run(_run_after_guard_release(root)) == {
            "retried": None,
            "created": "completed",
        }
    finally:
        if holder.is_alive():
            commands.put("exit")
        _join_or_terminate(holder)


def test_restart_adopts_unexpired_shared_stop_guard_and_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared-volume"
    context = multiprocessing.get_context("spawn")
    commands = context.Queue()
    results = context.Queue()
    holder = context.Process(
        target=_stop_guard_holder_child,
        args=(root, commands, results),
    )
    holder.start()
    ready = _queue_result(results)
    assert ready["phase"] == "ready"
    holder.kill()
    holder.join(10)
    assert not holder.is_alive()
    assert holder.exitcode is not None and holder.exitcode != 0

    store = FileManifestStore(root, fsync_writes=False)
    crashed_guard = store.read_gpu_stop_guard()
    assert crashed_guard is not None
    recovered = asyncio.run(_inspect_recovered_guard(root))
    assert recovered["server_instance_id"] != ready["server_instance_id"]
    orphan = recovered["stop_request"]
    assert orphan["state"] == "finalizing"
    assert orphan["request_id"] == STOP_REQUEST
    assert orphan["pod_id"] == "pod-123"
    assert orphan["gpu_display_name"] == "NVIDIA RTX 4090"
    assert orphan["requester"]["display_name"] == "Lakshman"
    assert orphan["requester"]["session_id"] != recovered["current_session_id"]
    assert orphan["finalization_expires_at"] == ready["expires_at"]
    assert orphan["finalization_id"] is None
    assert orphan["waiting_for"] == []
    assert orphan["approved_by"] == []
    assert orphan["denied_by"] == []
    assert recovered["can_create"] is False
    assert recovered["lease"] is True
    assert recovered["create"]["code"] == "gpu_stop_pending"
    assert recovered["create"]["details"] == {
        "request_id": STOP_REQUEST,
        "requester": "Lakshman",
        "expires_at": ready["expires_at"],
    }
    assert recovered["stale_cancel"] == "stop_request_not_found"
    assert recovered["stale_response"] == "stop_request_not_found"
    assert recovered["stale_finalize"] == "stop_request_not_found"
    assert store.read_gpu_stop_guard() == crashed_guard

    assert store.try_acquire_active_lease()
    try:
        store.clear_gpu_stop_guard(crashed_guard)
    finally:
        store.release_active_lease()


def test_persisted_stop_guard_rejects_unbounded_timestamp_envelope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared-volume"
    root.mkdir(parents=True)
    requested_at = datetime.now(UTC)
    response_deadline = requested_at + timedelta(seconds=30)
    expires_at = requested_at + timedelta(days=365)
    (root / ".gpu-stop-finalization.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "server_instance_id": "10000000-0000-4000-8000-000000000001",
                "request_id": STOP_REQUEST,
                "finalization_id": FINALIZATION,
                "pod_id": "pod-123",
                "gpu_display_name": "NVIDIA RTX 4090",
                "requester": "Lakshman",
                "requested_at": _timestamp(requested_at),
                "response_deadline": _timestamp(response_deadline),
                "expires_at": _timestamp(expires_at),
            }
        )
    )

    assert FileManifestStore(root, fsync_writes=False).read_gpu_stop_guard() is None


def test_restart_clears_bounded_stop_guard_shifted_beyond_safety_ttl(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared-volume"
    store = FileManifestStore(root, fsync_writes=False)
    requested_at = datetime.now(UTC) + timedelta(days=365)
    response_deadline = requested_at + timedelta(seconds=30)
    expires_at = response_deadline + timedelta(seconds=60)
    guard = SharedGpuStopGuard(
        server_instance_id="10000000-0000-4000-8000-000000000001",
        request_id=STOP_REQUEST,
        finalization_id=FINALIZATION,
        pod_id="pod-123",
        gpu_display_name="NVIDIA RTX 4090",
        requester="Lakshman",
        requested_at=_timestamp(requested_at),
        response_deadline=_timestamp(response_deadline),
        expires_at=_timestamp(expires_at),
    )
    assert store.try_acquire_active_lease()
    try:
        store.write_gpu_stop_guard(guard)
    finally:
        store.release_active_lease()

    async def inspect() -> None:
        controller = _controller(root, FakeInferenceAdapter())
        await controller.initialize()
        try:
            status = await controller.status(Principal("sujal", "Sujal"), ready=True)
            assert status.permissions.can_create is True
            assert controller.store.active_lease_held is False
            assert controller.store.read_gpu_stop_guard() is None
        finally:
            await controller.shutdown()

    asyncio.run(inspect())


def test_already_running_standby_adopts_guard_after_finalizer_crash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared-volume"
    context = multiprocessing.get_context("spawn")
    commands = context.Queue()
    results = context.Queue()
    standby_ready = context.Event()
    standby_release = context.Event()
    holder = context.Process(
        target=_stop_guard_holder_child,
        args=(root, commands, results),
    )
    holder.start()
    holder_ready = _queue_result(results)
    assert holder_ready["phase"] == "ready"

    standby = context.Process(
        target=_stop_guard_standby_child,
        args=(root, standby_ready, standby_release, results),
    )
    standby.start()
    assert standby_ready.wait(20), "standby did not initialize"
    observing = _queue_result(results)
    assert observing["phase"] == "observing"
    assert observing["can_create"] is False
    assert observing["lease"] is False
    assert observing["stop"]["state"] == "finalizing"
    assert observing["stop"]["finalization_id"] is None
    assert observing["stop"]["waiting_for"] == []
    assert observing["stop"]["requester"]["session_id"] != observing[
        "current_session_id"
    ]

    holder.kill()
    holder.join(10)
    assert not holder.is_alive()
    try:
        adopted = _queue_result(results)
        assert adopted["phase"] == "adopted"
        assert adopted["can_create"] is False
        assert adopted["lease"] is True
        assert adopted["marker"] is True
        assert adopted["stop"]["state"] == "finalizing"
        assert adopted["stop"]["finalization_id"] is None
        assert adopted["stop"]["waiting_for"] == []
        assert adopted["stop"]["requester"]["session_id"] != adopted[
            "current_session_id"
        ]
    finally:
        standby_release.set()
        _join_or_terminate(standby)

    store = FileManifestStore(root, fsync_writes=False)
    marker = store.read_gpu_stop_guard()
    assert marker is not None
    assert store.try_acquire_active_lease()
    try:
        store.clear_gpu_stop_guard(marker)
    finally:
        store.release_active_lease()
