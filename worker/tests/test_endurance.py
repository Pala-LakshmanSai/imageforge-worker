from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest
from conftest import TOKEN_B, auth, wait_for_batch, worker_client

from imageforge_worker.inference import FakeInferenceAdapter


@pytest.mark.anyio
@pytest.mark.endurance
async def test_450_prompt_ordered_run_pauses_survives_new_pod_and_completes(
    tmp_path: Path,
) -> None:
    volume = tmp_path / "network-volume"
    prompts = [f"Editorial realism frame {index:03d}" for index in range(1, 451)]
    first_generation_started = asyncio.Event()
    release_first_generation = asyncio.Event()
    first_adapter = FakeInferenceAdapter(
        delay_seconds=0.001,
        first_generation_started=first_generation_started,
        release_first_generation=release_first_generation,
    )
    async with worker_client(volume, first_adapter, fsync_writes=False) as (client, _, _):
        created = await client.post(
            "/v1/batches",
            json={"prompts": prompts, "base_seed": 10_000},
            headers=auth(),
        )
        assert created.status_code == 201
        batch_id = created.json()["batch_id"]
        await asyncio.wait_for(first_generation_started.wait(), timeout=60)
        pause = await client.post(f"/v1/batches/{batch_id}/pause", headers=auth())
        assert pause.status_code == 200
        release_first_generation.set()
        paused = await wait_for_batch(
            client, batch_id, state="paused", timeout=60, poll_interval=0.05
        )
        assert paused["progress"]["completed"] >= 1
        busy = await client.post(
            "/v1/batches", json={"prompts": ["no queue"]}, headers=auth(TOKEN_B)
        )
        assert busy.status_code == 423
        await client.post(f"/v1/batches/{batch_id}/resume", headers=auth())
        await wait_for_batch(
            client, batch_id, processed_at_least=25, timeout=60, poll_interval=0.05
        )

        first = await client.get(f"/v1/batches/{batch_id}/artifacts/1", headers=auth())
        receipt = await client.post(
            f"/v1/batches/{batch_id}/receipts",
            headers=auth(),
            json={
                "receipts": [
                    {
                        "index": 1,
                        "sha256": hashlib.sha256(first.content).hexdigest(),
                        "size_bytes": len(first.content),
                    }
                ]
            },
        )
        assert receipt.status_code == 200

    replacement_adapter = FakeInferenceAdapter()
    async with worker_client(volume, replacement_adapter, fsync_writes=False) as (client, _, _):
        interrupted = await client.get(f"/v1/batches/{batch_id}", headers=auth())
        assert interrupted.json()["state"] == "interrupted"
        assert interrupted.json()["images"][0]["status"] == "downloaded"
        await client.post(f"/v1/batches/{batch_id}/resume", headers=auth())
        final = await wait_for_batch(
            client, batch_id, state="completed", timeout=600, poll_interval=0.05
        )

    assert final["progress"] == {
        "total": 450,
        "completed": 450,
        "downloaded": 1,
        "failed": 0,
        "cancelled": 0,
        "processed": 450,
        "current_index": None,
    }
    assert [image["index"] for image in final["images"]] == list(range(1, 451))
    assert [image["prompt"] for image in final["images"]] == prompts
    assert [image["seed"] for image in final["images"]] == list(range(10_000, 10_450))
    assert [image["filename"] for image in final["images"]] == [
        f"artifacts/{index:06d}.jpg" for index in range(1, 451)
    ]
    # The first worker is torn down abruptly, mid-run, on purpose -- that is what
    # produces the `interrupted` state asserted above. Whichever image is being
    # generated at that instant has no recorded success, so the replacement
    # correctly regenerates it. Requiring the two adapters to concatenate into a
    # gapless 1..450 therefore asserted that no work was ever in flight at
    # shutdown, which only held because a local SSD finishes a manifest write
    # faster than the scheduler can start the next image. On the RunPod network
    # volume that write is far slower, so the strict form encoded an assumption
    # that is false on the deployment target.
    #
    # Pin what actually matters instead: every prompt is generated, each worker
    # generates in order, and an abrupt stop costs at most the single in-flight
    # image -- never a silently dropped or wholesale-repeated range.
    combined = first_adapter.generated_indices + replacement_adapter.generated_indices
    assert set(combined) == set(range(1, 451))
    assert first_adapter.generated_indices == sorted(first_adapter.generated_indices)
    assert replacement_adapter.generated_indices == sorted(
        replacement_adapter.generated_indices
    )
    assert len(combined) - len(set(combined)) <= 1, (
        "an abrupt stop may only cost the one in-flight image; "
        f"{len(combined) - len(set(combined))} images were regenerated"
    )
    for image in final["images"]:
        path = volume / "batches" / batch_id / image["filename"]
        assert path.stat().st_size == image["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == image["sha256"]
