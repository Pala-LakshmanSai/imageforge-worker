from __future__ import annotations

import asyncio
import base64
import hashlib
import io
from pathlib import Path

import httpx
import pytest
from conftest import TOKEN_A, TOKEN_B, auth, wait_for_batch, worker_client
from PIL import Image

from imageforge_worker.inference import FakeInferenceAdapter


@pytest.mark.anyio
async def test_ordered_artifacts_checksums_receipts_and_owner_isolation(
    tmp_path: Path,
) -> None:
    async with worker_client(tmp_path / "volume") as (client, _, _):
        prompts = ["first frame", "second frame", "third frame"]
        created = await client.post(
            "/v1/batches",
            json={"prompts": prompts, "base_seed": 100},
            headers=auth(),
        )
        assert created.status_code == 201
        batch_id = created.json()["batch_id"]
        manifest = await wait_for_batch(client, batch_id, state="completed")

        assert [item["index"] for item in manifest["images"]] == [1, 2, 3]
        assert [item["prompt"] for item in manifest["images"]] == prompts
        assert [item["seed"] for item in manifest["images"]] == [100, 101, 102]
        assert [item["filename"] for item in manifest["images"]] == [
            "artifacts/000001.jpg",
            "artifacts/000002.jpg",
            "artifacts/000003.jpg",
        ]

        full = await client.get(f"/v1/batches/{batch_id}/artifacts/1", headers=auth())
        preview = await client.get(f"/v1/batches/{batch_id}/previews/1", headers=auth())
        assert full.status_code == preview.status_code == 200
        assert full.headers["content-type"] == "image/jpeg"
        assert preview.headers["content-type"] == "image/webp"
        checksum = hashlib.sha256(full.content).hexdigest()
        assert full.headers["x-imageforge-sha256"] == checksum
        assert full.headers["x-checksum-sha256"] == checksum
        assert full.headers["etag"] == f'"{checksum}"'
        assert full.headers["digest"] == (
            "sha-256=" + base64.b64encode(bytes.fromhex(checksum)).decode()
        )
        assert int(full.headers["content-length"]) == len(full.content)
        with Image.open(io.BytesIO(full.content)) as image:
            assert image.format == "JPEG"
            assert image.size == (1280, 720)
        with Image.open(io.BytesIO(preview.content)) as image:
            assert image.format == "WEBP"
            assert image.size == (320, 180)

        other_user = await client.get(f"/v1/batches/{batch_id}", headers=auth(TOKEN_B))
        assert other_user.status_code == 404
        assert other_user.json()["error"]["code"] == "batch_not_found"

        atomic_reject = await client.post(
            f"/v1/batches/{batch_id}/receipts",
            headers=auth(),
            json={
                "receipts": [
                    {"index": 1, "sha256": checksum, "size_bytes": len(full.content)},
                    {"index": 2, "sha256": "0" * 64, "size_bytes": 1},
                ]
            },
        )
        assert atomic_reject.status_code == 409
        unchanged = await client.get(f"/v1/batches/{batch_id}", headers=auth())
        assert unchanged.json()["images"][0]["status"] == "ready"

        receipt = await client.post(
            f"/v1/batches/{batch_id}/receipts",
            headers=auth(),
            json={"receipts": [{"index": 1, "sha256": checksum, "size_bytes": len(full.content)}]},
        )
        assert receipt.status_code == 200
        assert receipt.json()["accepted"] == [1]
        acknowledged = await client.get(f"/v1/batches/{batch_id}", headers=auth())
        assert acknowledged.json()["images"][0]["status"] == "downloaded"

        duplicate = await client.post(
            "/v1/batches",
            json={"prompts": prompts, "base_seed": 100},
            headers=auth(),
        )
        duplicate_manifest = await wait_for_batch(
            client, duplicate.json()["batch_id"], state="completed"
        )
        assert [item["sha256"] for item in duplicate_manifest["images"]] == [
            item["sha256"] for item in manifest["images"]
        ]


@pytest.mark.anyio
async def test_bounded_automatic_retries_then_retry_failed_only(tmp_path: Path) -> None:
    adapter = FakeInferenceAdapter(failures_before_success={1: 2, 2: 99})
    async with worker_client(tmp_path / "volume", adapter) as (client, _, _):
        created = await client.post(
            "/v1/batches", json={"prompts": ["flaky", "terminal"]}, headers=auth()
        )
        batch_id = created.json()["batch_id"]
        first = await wait_for_batch(client, batch_id, state="completed")
        assert first["images"][0]["status"] == "ready"
        assert first["images"][0]["attempts"] == 3
        assert first["images"][1]["status"] == "failed"
        assert first["images"][1]["attempts"] == 3
        assert len(first["images"][1]["attempt_history"]) == 3

        adapter.failures_before_success[2] = 3
        retried = await client.post(f"/v1/batches/{batch_id}/retry-failed", headers=auth())
        assert retried.status_code == 200
        final = await wait_for_batch(client, batch_id, state="completed")
        assert final["images"][0]["attempts"] == 3
        assert final["images"][0]["retry_rounds"] == 0
        assert final["images"][1]["status"] == "ready"
        assert final["images"][1]["attempts"] == 4
        assert final["images"][1]["retry_rounds"] == 1
        assert adapter.generated_indices == [1, 2]


@pytest.mark.anyio
async def test_pause_holds_lock_resume_and_cancel_releases_it(tmp_path: Path) -> None:
    adapter = FakeInferenceAdapter(delay_seconds=0.08)
    async with worker_client(tmp_path / "volume", adapter) as (client, _, _):
        created = await client.post(
            "/v1/batches",
            json={"prompts": [f"frame {index}" for index in range(10)]},
            headers=auth(),
        )
        batch_id = created.json()["batch_id"]
        await wait_for_batch(client, batch_id, current_index=1)
        pause = await client.post(f"/v1/batches/{batch_id}/pause", headers=auth())
        assert pause.status_code == 200
        paused = await wait_for_batch(client, batch_id, state="paused")
        processed_when_paused = paused["progress"]["processed"]
        await asyncio.sleep(0.12)
        still_paused = await client.get(f"/v1/batches/{batch_id}", headers=auth())
        assert still_paused.json()["progress"]["processed"] == processed_when_paused

        busy = await client.post(
            "/v1/batches", json={"prompts": ["must not queue"]}, headers=auth(TOKEN_B)
        )
        assert busy.status_code == 423
        assert busy.json()["error"]["code"] == "batch_busy"
        assert busy.json()["error"]["details"]["owner"] == "Lakshman"

        resumed = await client.post(f"/v1/batches/{batch_id}/resume", headers=auth())
        assert resumed.status_code == 200
        await wait_for_batch(client, batch_id, processed_at_least=processed_when_paused + 1)
        cancel = await client.post(f"/v1/batches/{batch_id}/cancel", headers=auth())
        assert cancel.status_code == 200
        cancelled = await wait_for_batch(client, batch_id, state="cancelled")
        assert cancelled["progress"]["cancelled"] > 0

        next_batch = await client.post(
            "/v1/batches", json={"prompts": ["new owner"]}, headers=auth(TOKEN_B)
        )
        assert next_batch.status_code == 201
        await wait_for_batch(
            client, next_batch.json()["batch_id"], state="completed", token=TOKEN_B
        )


@pytest.mark.anyio
async def test_two_independent_clients_get_one_atomic_winner(tmp_path: Path) -> None:
    adapter = FakeInferenceAdapter(delay_seconds=0.1)
    async with worker_client(tmp_path / "volume", adapter) as (client, app, _):
        second_transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=second_transport, base_url="http://worker.test"
        ) as other_client:
            first_request = client.post(
                "/v1/batches", json={"prompts": ["from Lakshman"]}, headers=auth(TOKEN_A)
            )
            second_request = other_client.post(
                "/v1/batches", json={"prompts": ["from Sujal"]}, headers=auth(TOKEN_B)
            )
            first, second = await asyncio.gather(first_request, second_request)
            assert sorted([first.status_code, second.status_code]) == [201, 423]
            winner, loser = (first, second) if first.status_code == 201 else (second, first)
            winning_name = "Lakshman" if winner is first else "Sujal"
            assert loser.json()["error"]["details"]["owner"] == winning_name
            assert loser.json()["error"]["details"]["total"] == 1
            winner_token = TOKEN_A if winner is first else TOKEN_B
            await wait_for_batch(
                client, winner.json()["batch_id"], state="completed", token=winner_token
            )
