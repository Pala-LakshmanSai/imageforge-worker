"""`/v1/status` must not re-read the whole submission namespace per request.

The desktop polls status every 1.5 seconds. When each poll re-read and
re-validated every manifest ever written to the shared volume, the worker's
event loop stalled for tens of seconds, and ready artifacts could not be
served while that scan held the controller lock.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from conftest import TOKEN_A, auth, worker_client

from imageforge_worker.persistence import FileManifestStore


class _CountingStore(FileManifestStore):
    """Count every manifest document actually read from the volume."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.manifest_reads = 0

    def _read_manifest_record(self, batch_id: str):  # type: ignore[no-untyped-def]
        self.manifest_reads += 1
        return super()._read_manifest_record(batch_id)


def _payload(prompt: str) -> dict[str, object]:
    return {
        "client_submission_id": str(uuid.uuid4()),
        "prompts": [prompt],
        "base_seed": 0,
        "aspect_ratio": "16:9",
    }


async def _drain(client, batch_id: str) -> None:
    from conftest import wait_for_batch

    await wait_for_batch(client, batch_id, state="completed")


@pytest.mark.anyio
async def test_status_manifest_reads_do_not_grow_with_history(tmp_path: Path) -> None:
    root = tmp_path / "volume"
    store = _CountingStore(root)

    async with worker_client(root, store=store) as (client, _app, _adapter):
        # Build up durable history the way repeated real batches do.
        for index in range(4):
            created = await client.post(
                "/v1/batches", json=_payload(f"history frame {index}"), headers=auth(TOKEN_A)
            )
            assert created.status_code == 201, created.text
            await _drain(client, created.json()["batch_id"])

        store.manifest_reads = 0
        first = await client.get("/v1/status", headers=auth(TOKEN_A))
        assert first.status_code == 200, first.text
        warm_reads = store.manifest_reads

        store.manifest_reads = 0
        for _ in range(5):
            polled = await client.get("/v1/status", headers=auth(TOKEN_A))
            assert polled.status_code == 200, polled.text
        steady_reads = store.manifest_reads

        # Five idle polls over four historical batches must not re-read the
        # namespace. Without a cache this is 5 * 4 = 20 manifest documents.
        assert steady_reads <= 5, (
            f"status re-read {steady_reads} manifests across 5 idle polls "
            f"(warm-up read {warm_reads}); history scanning is back"
        )


@pytest.mark.anyio
async def test_status_still_fails_closed_when_history_becomes_corrupt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "volume"
    store = _CountingStore(root)

    async with worker_client(root, store=store) as (client, _app, _adapter):
        created = await client.post(
            "/v1/batches", json=_payload("corruption probe"), headers=auth(TOKEN_A)
        )
        assert created.status_code == 201, created.text
        batch_id = created.json()["batch_id"]
        await _drain(client, batch_id)

        # Warm whatever cache the implementation keeps.
        warm = await client.get("/v1/status", headers=auth(TOKEN_A))
        assert warm.status_code == 200, warm.text
        assert warm.json()["permissions"]["create_block_reason"] is None

        # Damage the durable record after the cache is warm. A stale cached
        # verdict here would advertise a writable store over a corrupt one.
        manifest_path = root / "batches" / batch_id / "manifest.json"
        manifest_path.write_bytes(b"{ not a manifest")

        corrupted = await client.get("/v1/status", headers=auth(TOKEN_A))
        assert corrupted.status_code == 200, corrupted.text
        permissions = corrupted.json()["permissions"]
        assert permissions["can_create"] is False
        assert permissions["create_block_reason"] == "submission_store_corrupt"


@pytest.mark.anyio
async def test_status_does_not_rescan_history_while_a_batch_is_writing(
    tmp_path: Path,
) -> None:
    """The namespace cache must survive an in-progress batch.

    The verdict was cached against a fingerprint of the *whole* namespace, so
    any manifest write invalidated it. During a real run the live manifest is
    rewritten on every claim, success and receipt, which meant the cache only
    ever hit while the volume was idle -- exactly when it does not matter.
    Measured against a live Pod, `/v1/health` held 0.4 s idle and 12-14 s the
    moment a batch started writing.
    """
    root = tmp_path / "volume"
    store = _CountingStore(root)

    async with worker_client(root, store=store) as (client, _app, _adapter):
        for index in range(4):
            created = await client.post(
                "/v1/batches", json=_payload(f"history frame {index}"), headers=auth(TOKEN_A)
            )
            assert created.status_code == 201, created.text
            await _drain(client, created.json()["batch_id"])

        live = await client.post(
            "/v1/batches", json=_payload("live frame"), headers=auth(TOKEN_A)
        )
        assert live.status_code == 201, live.text
        live_id = live.json()["batch_id"]
        await _drain(client, live_id)

        assert (await client.get("/v1/status", headers=auth(TOKEN_A))).status_code == 200

        live_manifest = root / "batches" / live_id / "manifest.json"
        store.manifest_reads = 0
        for tick in range(5):
            # Stand in for the live batch publishing progress. A real write
            # replaces this file, so its identity changes exactly like this.
            stamp = os.stat(live_manifest).st_mtime_ns + (tick + 1) * 1_000_000
            os.utime(live_manifest, ns=(stamp, stamp))
            polled = await client.get("/v1/status", headers=auth(TOKEN_A))
            assert polled.status_code == 200, polled.text
        writing_reads = store.manifest_reads

        # Only the manifest that actually changed may be re-read. Rescanning the
        # namespace is 5 polls * 5 batches = 25 documents.
        assert writing_reads <= 10, (
            f"status re-read {writing_reads} manifests across 5 polls while one "
            "batch was writing; the namespace is being rescanned per write"
        )
