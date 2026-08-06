from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from conftest import auth, wait_for_health, worker_client

from imageforge_worker.config import Credential, WorkerSettings
from imageforge_worker.constants import MAX_REFERENCE_BYTES, MODEL_REVISION
from imageforge_worker.domain import CreateBatchRequest, ReceiptRequest
from imageforge_worker.inference import FakeInferenceAdapter


@pytest.mark.anyio
async def test_health_is_available_through_boot_phases_and_auth_is_required(
    tmp_path: Path,
) -> None:
    adapter = FakeInferenceAdapter(startup_delay_seconds=0.04)
    async with worker_client(tmp_path / "volume", adapter, wait_until_ready=False) as (
        client,
        _,
        _,
    ):
        first = await client.get("/v1/health")
        assert first.status_code == 200
        assert first.json()["schema_version"] == 1
        assert first.json()["phase"] in {
            "process",
            "storage",
            "weights",
            "gpu_load",
            "warmup",
            "ready",
        }

        early_create = await client.post(
            "/v1/batches", json={"prompts": ["never logged"]}, headers=auth()
        )
        assert early_create.status_code == 503
        assert early_create.json()["error"]["code"] == "worker_not_ready"

        no_auth = await client.get("/v1/status")
        wrong_auth = await client.get("/v1/status", headers={"Authorization": "Bearer " + "z" * 32})
        assert no_auth.status_code == wrong_auth.status_code == 401
        assert no_auth.headers["www-authenticate"] == "Bearer"
        assert no_auth.json()["error"]["code"] == "authentication_required"

        health = await wait_for_health(client, "ready")
        assert health["model"]["revision"] == MODEL_REVISION
        assert health["gpu"]["device_count"] == 1
        assert health["gpu"]["total_memory_bytes"] == 24 * 1024**3
        assert adapter.phase_history == ["weights", "gpu_load", "warmup", "ready"]
        assert set(health["phase_timings_ms"]) >= {
            "process",
            "storage",
            "weights",
            "gpu_load",
            "warmup",
            "ready",
        }


def test_runtime_secret_parsing_and_repr_redaction(tmp_path: Path) -> None:
    secret = "this-is-a-runtime-only-secret-0001"
    env = {
        "IMAGEFORGE_DATA_ROOT": str(tmp_path / "volume"),
        "IMAGEFORGE_MODEL_CACHE_DIR": str(tmp_path / "models"),
        "IMAGEFORGE_INFERENCE_BACKEND": "fake",
        "IMAGEFORGE_ALLOW_FAKE_INFERENCE": "1",
        "IMAGEFORGE_AUTH_TOKENS_JSON": json.dumps(
            [{"user_id": "lakshman", "display_name": "Lakshman", "token": secret}]
        ),
    }
    settings = WorkerSettings.from_env(env)
    assert settings.credentials[0].token == secret
    assert secret not in repr(settings.credentials[0])
    assert secret not in repr(settings)


def test_bearer_credentials_require_ascii_rfc_token_characters() -> None:
    with pytest.raises(ValueError, match="ASCII bearer-token"):
        Credential("lakshman", "Lakshman", "é" * 16)
    with pytest.raises(ValueError, match="ASCII bearer-token"):
        Credential("lakshman", "Lakshman", "not-valid-token!!!")


@pytest.mark.anyio
async def test_non_ascii_authorization_is_always_a_safe_401(tmp_path: Path) -> None:
    async with worker_client(tmp_path / "volume") as (client, _, _):
        response = await client.get(
            "/v1/status",
            headers=[(b"authorization", b"Bearer " + b"\xff" * 16)],
        )
        assert response.status_code == 401
        assert response.json()["schema_version"] == 1
        assert response.json()["error"] == {
            "code": "authentication_required",
            "message": "A valid worker bearer credential is required.",
            "details": None,
        }


@pytest.mark.anyio
async def test_validation_is_strict_and_does_not_echo_prompts(tmp_path: Path) -> None:
    async with worker_client(tmp_path / "volume") as (client, _, _):
        sensitive_prompt = "DO-NOT-ECHO-THIS" + "x" * 4096
        response = await client.post(
            "/v1/batches",
            json={"prompts": [sensitive_prompt], "unexpected_path": "../../etc/passwd"},
            headers=auth(),
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"
        assert sensitive_prompt not in response.text
        assert "../../etc/passwd" not in response.text

        wrong_type = await client.post("/v1/batches", json={"prompts": [123]}, headers=auth())
        assert wrong_type.status_code == 422

        not_found = await client.get("/v1/does-not-exist", headers=auth())
        assert not_found.status_code == 404
        assert not_found.json()["error"] == {
            "code": "not_found",
            "message": "The endpoint does not exist.",
            "details": None,
        }


def test_large_prompt_and_receipt_models_have_no_product_count_or_text_cap() -> None:
    prompts = [f"prompt {index}" for index in range(501)]
    prompts[500] = "long prompt " + "x" * 5000
    request = CreateBatchRequest(
        prompts=prompts,
        client_submission_id="00000000-0000-4000-8000-000000000001",
    )
    assert request.prompts == prompts

    receipts = ReceiptRequest(
        receipts=[{"index": index, "sha256": "0" * 64, "size_bytes": 1} for index in range(1, 502)]
    )
    assert len(receipts.receipts) == 501


def test_reference_request_bounds_are_practical_and_typed() -> None:
    with pytest.raises(ValueError):
        CreateBatchRequest(
            prompts=["safe"],
            client_submission_id="00000000-0000-4000-8000-000000000002",
            references=[
                {
                    "name": "too-large.png",
                    "mime_type": "image/png",
                    "data_hex": "00" * (MAX_REFERENCE_BYTES + 1),
                }
            ],
        )
    with pytest.raises(ValueError):
        CreateBatchRequest(
            prompts=["safe"],
            client_submission_id="00000000-0000-4000-8000-000000000003",
            references=[
                {"name": f"ref-{index}.png", "mime_type": "image/png", "data_hex": "00"}
                for index in range(9)
            ],
        )


@pytest.mark.anyio
async def test_create_batch_accepts_more_than_500_prompts_at_http_boundary(tmp_path: Path) -> None:
    first_generation_started = asyncio.Event()
    release_first_generation = asyncio.Event()
    adapter = FakeInferenceAdapter(
        first_generation_started=first_generation_started,
        release_first_generation=release_first_generation,
    )
    async with worker_client(tmp_path / "volume", adapter) as (client, _, _):
        prompts = [f"prompt {index}" for index in range(501)]
        prompts[-1] = "long prompt " + "x" * 5000
        response = await client.post(
            "/v1/batches", json={"prompts": prompts, "base_seed": 123}, headers=auth()
        )
        assert response.status_code == 201
        manifest = response.json()
        assert manifest["progress"]["total"] == 501
        assert [image["index"] for image in manifest["images"]] == list(range(1, 502))
        assert manifest["images"][-1]["prompt"] == prompts[-1]


@pytest.mark.anyio
async def test_health_stays_public_but_undocumented_routes_are_disabled(tmp_path: Path) -> None:
    async with worker_client(tmp_path / "volume") as (client, _, _):
        assert (await client.get("/v1/health")).status_code == 200
        docs = await client.get("/docs")
        schema = await client.get("/openapi.json")
        assert docs.status_code == schema.status_code == 404
        assert docs.json()["schema_version"] == 1
