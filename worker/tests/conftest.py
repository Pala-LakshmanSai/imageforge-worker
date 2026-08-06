from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from imageforge_worker.app import create_app
from imageforge_worker.config import Credential, WorkerSettings
from imageforge_worker.inference import FakeInferenceAdapter, InferenceAdapter
from imageforge_worker.persistence import ManifestStore

TOKEN_A = "lakshman-worker-token-000000000001"
TOKEN_B = "sujal-worker-token-00000000000002"


class _LegacyBatchTestClient(httpx.AsyncClient):
    """Keep pre-queue lifecycle tests on valid unique submission requests.

    Those tests exercise batching, receipts, and leases rather than request
    construction. New Task 013 API tests opt out to prove that the real HTTP
    endpoint rejects a missing submission ID.
    """

    async def post(self, url: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        payload = kwargs.get("json")
        if (
            url == "/v1/batches"
            and isinstance(payload, dict)
            and "client_submission_id" not in payload
        ):
            kwargs["json"] = {
                **payload,
                "client_submission_id": str(uuid.uuid4()),
            }
        return await super().post(url, *args, **kwargs)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def auth(token: str = TOKEN_A) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def settings_for(
    root: Path,
    *,
    fsync_writes: bool = True,
    runtime_metadata: dict[str, str] | None = None,
) -> WorkerSettings:
    return WorkerSettings(
        data_root=root,
        model_cache_dir=root / "model-cache",
        credentials=(
            Credential("lakshman", "Lakshman", TOKEN_A),
            Credential("sujal", "Sujal", TOKEN_B),
        ),
        inference_backend="fake",
        allow_fake_inference=True,
        fsync_writes=fsync_writes,
        retry_delay_seconds=0,
        runtime_metadata=runtime_metadata or {},
    )


@asynccontextmanager
async def worker_client(
    root: Path,
    adapter: InferenceAdapter | None = None,
    *,
    wait_until_ready: bool = True,
    fsync_writes: bool = True,
    runtime_metadata: dict[str, str] | None = None,
    inject_submission_ids: bool = True,
    store: ManifestStore | None = None,
) -> AsyncIterator[tuple[httpx.AsyncClient, object, InferenceAdapter]]:
    selected = adapter or FakeInferenceAdapter()
    app = create_app(
        settings_for(root, fsync_writes=fsync_writes, runtime_metadata=runtime_metadata),
        inference=selected,
        store=store,
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        client_type = _LegacyBatchTestClient if inject_submission_ids else httpx.AsyncClient
        async with client_type(transport=transport, base_url="http://worker.test") as client:
            if wait_until_ready:
                await wait_for_health(client, "ready")
            yield client, app, selected


async def wait_for_health(client: httpx.AsyncClient, phase: str, *, timeout: float = 20) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        response = await client.get("/v1/health")
        payload = response.json()
        if payload["phase"] == phase:
            return payload
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"health did not reach {phase}: {payload}")
        await asyncio.sleep(0.005)


async def wait_for_batch(
    client: httpx.AsyncClient,
    batch_id: str,
    *,
    state: str | None = None,
    processed_at_least: int | None = None,
    current_index: int | None = None,
    token: str = TOKEN_A,
    timeout: float = 30,
    poll_interval: float = 0.005,
) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    last: dict | None = None
    while True:
        response = await client.get(f"/v1/batches/{batch_id}", headers=auth(token))
        assert response.status_code == 200, response.text
        last = response.json()
        state_matches = state is None or last["state"] == state
        processed_matches = (
            processed_at_least is None or last["progress"]["processed"] >= processed_at_least
        )
        current_matches = (
            current_index is None or last["progress"]["current_index"] == current_index
        )
        if state_matches and processed_matches and current_matches:
            return last
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"batch did not reach requested condition: {last}")
        await asyncio.sleep(poll_interval)
