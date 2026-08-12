from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import uuid
from pathlib import Path

import httpx
import pytest
from conftest import TOKEN_A, TOKEN_B, auth, wait_for_batch, worker_client
from PIL import Image

from imageforge_worker import controller as controller_module
from imageforge_worker.inference import FakeInferenceAdapter
from imageforge_worker.model_profiles import FLUX2_KLEIN


def _encoded_image(image_format: str, size: tuple[int, int], color: str) -> str:
    payload = io.BytesIO()
    Image.new("RGB", size, color).save(payload, format=image_format)
    return payload.getvalue().hex()


@pytest.fixture
def reference_capable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the reference plumbing against a model that accepts references.

    The shipped model is text-to-image only, but the reference path still ships
    for the FLUX backend, so it keeps its coverage rather than being deleted.
    """

    monkeypatch.setattr(controller_module, "ACTIVE_PROFILE", FLUX2_KLEIN)


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
async def test_aspect_ratio_is_persisted_and_controls_encoded_dimensions(tmp_path: Path) -> None:
    expected = {
        "16:9": (1280, 720),
        "1:1": (1024, 1024),
        "9:16": (720, 1280),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
    }
    async with worker_client(tmp_path / "volume") as (client, _, _):
        for ratio, dimensions in expected.items():
            created = await client.post(
                "/v1/batches",
                json={"prompts": [f"ratio {ratio}"], "aspect_ratio": ratio},
                headers=auth(),
            )
            assert created.status_code == 201, created.text
            manifest = await wait_for_batch(client, created.json()["batch_id"], state="completed")
            artifact = await client.get(
                f"/v1/batches/{created.json()['batch_id']}/artifacts/1", headers=auth()
            )
            assert artifact.status_code == 200
            with Image.open(io.BytesIO(artifact.content)) as image:
                assert image.size == dimensions
            assert manifest["settings"]["width"] == dimensions[0]
            assert manifest["settings"]["height"] == dimensions[1]


@pytest.mark.anyio
async def test_batch_references_are_decoded_forwarded_and_persisted_without_raw_bytes(
    tmp_path: Path,
    reference_capable_model: None,
) -> None:
    adapter = FakeInferenceAdapter()
    first_data = _encoded_image("PNG", (32, 24), "red")
    second_data = _encoded_image("JPEG", (48, 36), "blue")
    async with worker_client(tmp_path / "volume", adapter) as (client, _, _):
        created = await client.post(
            "/v1/batches",
            headers=auth(),
            json={
                "prompts": ["reference guided frame"],
                "references": [
                    {"name": "subject.png", "mime_type": "image/png", "data_hex": first_data},
                    {"name": "palette.jpg", "mime_type": "image/jpeg", "data_hex": second_data},
                ],
            },
        )
        assert created.status_code == 201, created.text
        manifest = await wait_for_batch(client, created.json()["batch_id"], state="completed")
        assert manifest["references"] == [
            {
                "name": "subject.png",
                "mime_type": "image/png",
                "size_bytes": len(bytes.fromhex(first_data)),
                "sha256": hashlib.sha256(bytes.fromhex(first_data)).hexdigest(),
                "filename": "references/000001.png",
            },
            {
                "name": "palette.jpg",
                "mime_type": "image/jpeg",
                "size_bytes": len(bytes.fromhex(second_data)),
                "sha256": hashlib.sha256(bytes.fromhex(second_data)).hexdigest(),
                "filename": "references/000002.jpg",
            },
        ]
        assert first_data not in created.text
        assert second_data not in created.text
        assert adapter.reference_sizes_by_index[1] == ((32, 24), (48, 36))
        batch_id = created.json()["batch_id"]
        reference_path = tmp_path / "volume" / "batches" / batch_id / "references" / "000001.png"
        assert reference_path.read_bytes() == bytes.fromhex(first_data)


@pytest.mark.anyio
async def test_batch_reference_validation_rejects_mismatch_and_non_images(
    tmp_path: Path, reference_capable_model: None
) -> None:
    async with worker_client(tmp_path / "volume") as (client, _, _):
        malformed = await client.post(
            "/v1/batches",
            headers=auth(),
            json={
                "prompts": ["safe"],
                "references": [{"name": "bad.png", "mime_type": "image/png", "data_hex": "00"}],
            },
        )
        assert malformed.status_code == 422
        assert malformed.json()["error"]["code"] == "reference_invalid"
        assert "00" not in malformed.text

        mismatched = await client.post(
            "/v1/batches",
            headers=auth(),
            json={
                "prompts": ["safe"],
                "references": [
                    {
                        "name": "wrong-type.png",
                        "mime_type": "image/png",
                        "data_hex": _encoded_image("JPEG", (8, 8), "green"),
                    }
                ],
            },
        )
        assert mismatched.status_code == 422
        assert mismatched.json()["error"]["code"] == "reference_invalid"


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
                "/v1/batches",
                json={
                    "prompts": ["from Lakshman"],
                    "client_submission_id": str(uuid.uuid4()),
                },
                headers=auth(TOKEN_A),
            )
            second_request = other_client.post(
                "/v1/batches",
                json={
                    "prompts": ["from Sujal"],
                    "client_submission_id": str(uuid.uuid4()),
                },
                headers=auth(TOKEN_B),
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
