"""The worker must not run network-volume I/O on the asyncio event loop.

`/workspace/imageforge` is a RunPod network volume where reads, writes and
fsyncs are slow. Every one of those calls was issued synchronously from a
coroutine, so a single acknowledgement or artifact read stalled the whole
process: `/v1/health` -- which touches no manifest, no disk and no lock --
was measured at a pinned 15-18 s against a live Pod while a batch ran, versus
0.42 s when idle. Generation shares that loop, so images sat in `ready` while
the serving path monopolised it.

These tests pin the invariant behaviourally: while a deliberately slow volume
operation is in flight, an endpoint that needs nothing from the volume must
still be answered promptly.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

import pytest
from conftest import TOKEN_A, auth, wait_for_batch, worker_client

from imageforge_worker.domain import BatchManifest
from imageforge_worker.persistence import FileManifestStore

# How long one simulated network-volume write blocks for.
VOLUME_DELAY_SECONDS = 0.4
# A responsive loop answers health in microseconds. Allow generous slack for CI
# scheduling noise while still failing decisively if the loop is blocked for the
# whole volume delay.
RESPONSIVE_BUDGET_SECONDS = VOLUME_DELAY_SECONDS / 2


class _SlowVolumeStore(FileManifestStore):
    """A manifest store whose writes block, like the RunPod network volume."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.slow = False
        self.saves = 0

    def save(self, manifest: BatchManifest) -> None:
        self.saves += 1
        if self.slow:
            time.sleep(VOLUME_DELAY_SECONDS)
        super().save(manifest)

    def write_artifacts(
        self, batch_id: str, index: int, jpeg: bytes, preview: bytes
    ) -> tuple[str, str]:
        if self.slow:
            time.sleep(VOLUME_DELAY_SECONDS)
        return super().write_artifacts(batch_id, index, jpeg, preview)


async def _worst_loop_stall(stop: asyncio.Event, tick: float = 0.01) -> float:
    """Measure the largest gap between scheduled ticks.

    A cooperative loop reschedules this within roughly `tick`. A synchronous
    volume call inside a coroutine cannot be preempted, so the gap grows to the
    full duration of that call. This measures the blockage itself rather than
    relying on a request happening to be in flight at the right moment.
    """
    loop = asyncio.get_running_loop()
    worst = 0.0
    previous = loop.time()
    while not stop.is_set():
        await asyncio.sleep(tick)
        now = loop.time()
        worst = max(worst, now - previous - tick)
        previous = now
    return worst


@pytest.mark.anyio
async def test_recording_a_generated_image_does_not_stall_the_event_loop(
    tmp_path: Path,
) -> None:
    """Persisting a finished image must not block the loop either.

    `_record_success` runs once per generated image and wrote the JPEG and
    preview to the volume, hashed both, and rewrote the manifest -- all inline
    under the controller lock. That is the generation path blocking artifact
    serving, which is why ready images could not be downloaded while the batch
    kept generating.
    """
    root = tmp_path / "volume"
    store = _SlowVolumeStore(root)
    async with worker_client(root, store=store) as (client, _, _):
        store.slow = True
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(_worst_loop_stall(stop))
        await asyncio.sleep(0.02)

        created = await client.post(
            "/v1/batches",
            json={"prompts": ["only frame"], "base_seed": 100},
            headers=auth(),
        )
        assert created.status_code == 201
        await wait_for_batch(client, created.json()["batch_id"], state="completed")
        stop.set()
        worst_stall = await heartbeat

        assert worst_stall < RESPONSIVE_BUDGET_SECONDS, (
            "the event loop was blocked while recording a generated image: it "
            f"could not be rescheduled for {worst_stall:.3f}s"
        )


@pytest.mark.anyio
async def test_repeated_artifact_reads_do_not_rehash_the_same_file(
    tmp_path: Path,
) -> None:
    """Serving an artifact must not re-read and re-hash it from the volume.

    `_file_matches` hashed the whole JPEG on every `GET /artifacts/{i}`, on the
    event loop and under the controller lock, and `FileResponse` then read the
    same bytes again. That is two full network-volume reads per download, paid
    again on every retry, while generation waited for the same lock.
    """
    root = tmp_path / "volume"
    async with worker_client(root) as (client, app, _):
        created = await client.post(
            "/v1/batches",
            json={"prompts": ["first frame"], "base_seed": 100},
            headers=auth(),
        )
        assert created.status_code == 201
        batch_id = created.json()["batch_id"]
        await wait_for_batch(client, batch_id, state="completed")

        controller = app.state.runtime.controller
        digests_before = controller.artifact_digest_computations

        for _ in range(4):
            response = await client.get(
                f"/v1/batches/{batch_id}/artifacts/1", headers=auth()
            )
            assert response.status_code == 200

        computed = controller.artifact_digest_computations - digests_before
        assert computed <= 1, (
            f"the artifact was hashed {computed} times for 4 unchanged reads; "
            "a verified artifact must not be re-hashed from the network volume"
        )


@pytest.mark.anyio
async def test_slow_volume_acknowledgement_does_not_stall_the_event_loop(
    tmp_path: Path,
) -> None:
    root = tmp_path / "volume"
    store = _SlowVolumeStore(root)
    async with worker_client(root, store=store) as (client, _, _):
        created = await client.post(
            "/v1/batches",
            json={"prompts": ["first frame", "second frame"], "base_seed": 100},
            headers=auth(),
        )
        assert created.status_code == 201
        batch_id = created.json()["batch_id"]
        await wait_for_batch(client, batch_id, state="completed")

        artifact = await client.get(f"/v1/batches/{batch_id}/artifacts/1", headers=auth())
        assert artifact.status_code == 200
        checksum = hashlib.sha256(artifact.content).hexdigest()

        # Only the acknowledgement below should pay the slow-volume cost.
        saves_before = store.saves
        store.slow = True
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(_worst_loop_stall(stop))
        await asyncio.sleep(0.02)

        receipt = await client.post(
            f"/v1/batches/{batch_id}/receipts",
            headers=auth(TOKEN_A),
            json={
                "receipts": [
                    {"index": 1, "sha256": checksum, "size_bytes": len(artifact.content)}
                ]
            },
        )
        stop.set()
        worst_stall = await heartbeat

        assert receipt.status_code == 200, receipt.text
        assert store.saves > saves_before, "the acknowledgement never wrote the manifest"
        assert worst_stall < RESPONSIVE_BUDGET_SECONDS, (
            "the event loop was blocked by network-volume I/O: it could not be "
            f"rescheduled for {worst_stall:.3f}s while a {VOLUME_DELAY_SECONDS}s "
            "manifest write ran inside a coroutine"
        )
