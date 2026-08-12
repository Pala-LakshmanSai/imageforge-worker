#!/usr/bin/env python3
"""Opt-in EU-RO-1 two-worker network-volume qualification harness.

This intentionally does not create or terminate RunPod compute. The operator
must provision two identical Pods, point both at an isolated gate data root,
and explicitly stop the owner between the contention and recovery phases. The
harness records evidence and fails closed on any unexpected mutation/queue.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


def require_opt_in() -> None:
    if os.environ.get("IMAGEFORGE_REAL_VOLUME_TEST") != "1":
        raise SystemExit("Set IMAGEFORGE_REAL_VOLUME_TEST=1; this gate never runs accidentally.")


def env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required gate setting: {name}")
    return value


def endpoint(raw: str) -> str:
    value = raw.rstrip("/")
    if not value.startswith("https://"):
        raise SystemExit("Gate worker endpoints must use HTTPS.")
    return value


async def request(
    client: httpx.AsyncClient,
    base: str,
    token: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    response = await client.request(
        method,
        f"{base}{path}",
        headers={"authorization": f"Bearer {token}"},
        json=payload,
    )
    try:
        body: Any = response.json()
    except ValueError:
        body = None
    return response.status_code, body


async def contention(
    client: httpx.AsyncClient, a: tuple[str, str], b: tuple[str, str]
) -> dict[str, Any]:
    prompt_count_raw = os.environ.get("IMAGEFORGE_GATE_PROMPT_COUNT", "1").strip()
    try:
        prompt_count = int(prompt_count_raw)
    except ValueError as exc:
        raise RuntimeError("IMAGEFORGE_GATE_PROMPT_COUNT must be a positive integer") from exc
    if prompt_count < 1 or prompt_count > 2000:
        raise RuntimeError("IMAGEFORGE_GATE_PROMPT_COUNT must be between 1 and 2000")
    prompt = f"ImageForge isolated volume gate {uuid.uuid4()}"
    prompts = [f"{prompt} frame {index}" for index in range(1, prompt_count + 1)]
    results = await asyncio.gather(
        request(
            client, a[0], a[1], "POST", "/v1/batches", {"prompts": prompts, "base_seed": 41000}
        ),
        request(
            client, b[0], b[1], "POST", "/v1/batches", {"prompts": prompts, "base_seed": 41000}
        ),
    )
    statuses = sorted(item[0] for item in results)
    if statuses != [201, 423]:
        raise RuntimeError(f"Expected exactly one winner and one HTTP 423 loser, got {statuses}")
    winner = next(index for index, item in enumerate(results) if item[0] == 201)
    loser = 1 - winner
    winner_body = results[winner][1]
    batch_id = winner_body.get("batch_id") if isinstance(winner_body, dict) else None
    if not isinstance(batch_id, str):
        raise RuntimeError("Winner response did not expose a server batch_id")
    loser_body = results[loser][1]
    if not isinstance(loser_body, dict) or loser_body.get("error", {}).get("code") != "batch_busy":
        raise RuntimeError("The losing create response was not the typed batch_busy error")

    winner_client = (a, b)[winner]
    observer_client = (a, b)[loser]
    status_code, status_body = await request(
        client, observer_client[0], observer_client[1], "GET", "/v1/status"
    )
    observed_active = status_body.get("active_batch") if isinstance(status_body, dict) else None
    if (
        status_code != 200
        or not isinstance(observed_active, dict)
        or observed_active.get("batch_id") != batch_id
    ):
        raise RuntimeError("Observer did not see the one authoritative active batch")
    mutation_code, _ = await request(
        client, observer_client[0], observer_client[1], "POST", f"/v1/batches/{batch_id}/pause"
    )
    if mutation_code != 423:
        raise RuntimeError(f"Observer mutation was not blocked with HTTP 423: {mutation_code}")
    return {
        "winner": "A" if winner == 0 else "B",
        "observer": "A" if loser == 0 else "B",
        "batch_id": batch_id,
        "prompt_count": prompt_count,
        "winner_create_status": results[winner][0],
        "observer_create_status": results[loser][0],
        "observer_create_error_code": (
            loser_body.get("error", {}).get("code") if isinstance(loser_body, dict) else None
        ),
        "observer_status_code": status_code,
        "observer_mutation_status": mutation_code,
        "winner_endpoint": winner_client[0],
        "observer_endpoint": observer_client[0],
    }


async def verify_identity(
    client: httpx.AsyncClient,
    label: str,
    pair: tuple[str, str],
    expected_pod_id: str,
    expected_volume_id: str,
    expected_root: str,
    expected_digest: str,
    expected_region: str,
) -> None:
    status_code, health = await request(client, pair[0], pair[1], "GET", "/v1/health")
    if status_code != 200 or not isinstance(health, dict):
        raise RuntimeError(f"Worker {label} health did not pass: HTTP {status_code}")
    runtime = health.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError(f"Worker {label} did not report pinned runtime identity")
    expected = {
        "RUNPOD_POD_ID": expected_pod_id,
        "RUNPOD_VOLUME_ID": expected_volume_id,
        "RUNPOD_DC_ID": expected_region,
        "IMAGEFORGE_DATA_ROOT": expected_root,
        "IMAGEFORGE_IMAGE_DIGEST": expected_digest,
    }
    mismatches = {
        key: {"expected": value, "actual": runtime.get(key)}
        for key, value in expected.items()
        if runtime.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Worker {label} identity mismatch: {mismatches}")
    model = health.get("model")
    if not isinstance(model, dict) or model.get("id") != "Comfy-Org/Mage-Flow":
        raise RuntimeError(f"Worker {label} reported an unexpected model")


async def observe_recovery(
    client: httpx.AsyncClient, endpoint_pair: tuple[str, str], batch_id: str
) -> dict[str, Any]:
    deadline = time.monotonic() + float(
        os.environ.get("IMAGEFORGE_GATE_RECOVERY_TIMEOUT_SECONDS", "600")
    )
    samples: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        status_code, body = await request(
            client, endpoint_pair[0], endpoint_pair[1], "GET", "/v1/status"
        )
        active_batch = body.get("active_batch") if isinstance(body, dict) else None
        samples.append({"status": status_code, "active_batch": active_batch})
        if status_code == 200 and isinstance(body, dict):
            manifest_code, manifest = await request(
                client, endpoint_pair[0], endpoint_pair[1], "GET", f"/v1/batches/{batch_id}"
            )
            if (
                manifest_code == 200
                and isinstance(manifest, dict)
                and manifest.get("state") in {"interrupted", "completed", "failed", "cancelled"}
            ):
                return {"batch_id": batch_id, "manifest": manifest, "samples": samples}
        await asyncio.sleep(2)
    raise TimeoutError("Survivor did not recover the gate batch before the deadline")


async def main() -> None:
    require_opt_in()
    run_id = os.environ.get("IMAGEFORGE_GATE_RUN_ID", uuid.uuid4().hex)
    gate_root = env("IMAGEFORGE_GATE_ROOT")
    expected_root = f"/workspace/imageforge-gates/{run_id}"
    if gate_root != expected_root:
        raise SystemExit(f"IMAGEFORGE_GATE_ROOT must be the isolated path {expected_root!r}")
    a = (endpoint(env("IMAGEFORGE_GATE_ENDPOINT_A")), env("IMAGEFORGE_GATE_TOKEN_A"))
    b = (endpoint(env("IMAGEFORGE_GATE_ENDPOINT_B")), env("IMAGEFORGE_GATE_TOKEN_B"))
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "region": env("IMAGEFORGE_GATE_REGION"),
        "volume_id": env("IMAGEFORGE_GATE_VOLUME_ID"),
        "pod_ids": {"A": env("IMAGEFORGE_GATE_POD_A_ID"), "B": env("IMAGEFORGE_GATE_POD_B_ID")},
        "image_digest": env("IMAGEFORGE_GATE_IMAGE_DIGEST"),
        "mount_root": gate_root,
        "host_kernel": platform.release(),
        "started_at_unix": time.time(),
    }
    if evidence["region"] != "EU-RO-1":
        raise SystemExit("This qualification harness is intentionally restricted to EU-RO-1.")

    async with httpx.AsyncClient(timeout=15.0) as client:
        survivor = os.environ.get("IMAGEFORGE_GATE_SURVIVOR", "").strip().upper()
        expected_root = evidence["mount_root"]
        if survivor:
            if survivor not in {"A", "B"}:
                raise SystemExit("IMAGEFORGE_GATE_SURVIVOR must be A or B")
            owner = env("IMAGEFORGE_GATE_OWNER").upper()
            if owner not in {"A", "B"}:
                raise SystemExit(
                    "IMAGEFORGE_GATE_OWNER must be A or B and identify the original winner"
                )
            survivor_pair = (a, b)[0 if survivor == "A" else 1]
            owner_pair = (a, b)[0 if owner == "A" else 1]
            await verify_identity(
                client,
                survivor,
                survivor_pair,
                evidence["pod_ids"][survivor],
                evidence["volume_id"],
                expected_root,
                evidence["image_digest"],
                evidence["region"],
            )
            batch_id = env("IMAGEFORGE_GATE_BATCH_ID")
            recovery_endpoint = (survivor_pair[0], owner_pair[1])
            evidence["recovery"] = await observe_recovery(client, recovery_endpoint, batch_id)
        else:
            await asyncio.gather(
                verify_identity(
                    client,
                    "A",
                    a,
                    evidence["pod_ids"]["A"],
                    evidence["volume_id"],
                    expected_root,
                    evidence["image_digest"],
                    evidence["region"],
                ),
                verify_identity(
                    client,
                    "B",
                    b,
                    evidence["pod_ids"]["B"],
                    evidence["volume_id"],
                    expected_root,
                    evidence["image_digest"],
                    evidence["region"],
                ),
            )
            evidence["contention"] = await contention(client, a, b)
            evidence["operator_instruction"] = (
                "Stop the recorded winner Pod, then rerun with "
                "IMAGEFORGE_GATE_SURVIVOR=<live role>, "
                "IMAGEFORGE_GATE_OWNER=<original winner role>, and "
                "IMAGEFORGE_GATE_BATCH_ID=<recorded batch_id>."
            )
    evidence["finished_at_unix"] = time.time()
    output = Path(
        os.environ.get("IMAGEFORGE_GATE_EVIDENCE", f"imageforge-volume-gate-{run_id}.json")
    )
    await asyncio.to_thread(
        output.write_text,
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "evidence": str(output), "run_id": run_id}, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (RuntimeError, TimeoutError, httpx.HTTPError) as error:
        print(f"ImageForge volume gate FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
