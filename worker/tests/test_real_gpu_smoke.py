from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest
from PIL import Image

from imageforge_worker.domain import GenerationSettings
from imageforge_worker.inference import FluxInferenceAdapter, GenerationJob

REAL_GPU_AUTHORIZED = os.environ.get("IMAGEFORGE_REAL_GPU_TEST") == "1"


@pytest.mark.real_gpu
@pytest.mark.anyio
@pytest.mark.skipif(
    not REAL_GPU_AUTHORIZED,
    reason="requires explicit IMAGEFORGE_REAL_GPU_TEST=1 authorization on an existing GPU",
)
async def test_authorized_approved_architecture_flux_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run on representative Ampere, Ada, and Blackwell Pods; never create a Pod."""

    expected_family = os.environ.get("IMAGEFORGE_REAL_GPU_FAMILY")
    if expected_family not in {"ampere", "ada", "blackwell"}:
        pytest.fail("set IMAGEFORGE_REAL_GPU_FAMILY=ampere, ada, or blackwell")
    cache_raw = os.environ.get("IMAGEFORGE_MODEL_CACHE_DIR")
    if not cache_raw:
        pytest.fail("IMAGEFORGE_MODEL_CACHE_DIR must point at the prepared pinned snapshot")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("DIFFUSERS_OFFLINE", "1")

    import torch

    capability = torch.cuda.get_device_capability(0)
    if expected_family == "ampere":
        assert capability == (8, 6), torch.cuda.get_device_name(0)
    elif expected_family == "ada":
        assert capability == (8, 9), torch.cuda.get_device_name(0)
    else:
        assert capability[0] >= 10, torch.cuda.get_device_name(0)

    phases: list[str] = []

    async def record_phase(phase, _progress: float) -> None:
        phases.append(phase.value)

    adapter = FluxInferenceAdapter(Path(cache_raw))
    try:
        await adapter.startup(record_phase)
        result = await adapter.generate(
            GenerationJob(
                index=1,
                prompt="Editorial photograph of a ceramic cup in soft window light",
                seed=20260801,
                settings=GenerationSettings(),
            )
        )
        with Image.open(io.BytesIO(result.jpeg)) as image:
            assert image.format == "JPEG"
            assert image.size == (1280, 720)
        with Image.open(io.BytesIO(result.preview)) as image:
            assert image.format == "WEBP"
            assert image.size == (320, 180)
        assert len(hashlib.sha256(result.jpeg).hexdigest()) == 64
        assert phases == ["weights", "gpu_load", "warmup", "ready"]
        snapshot = adapter.gpu_snapshot()
        assert snapshot["approved"] is True
        assert snapshot["device_count"] == 1
    finally:
        await adapter.shutdown()
