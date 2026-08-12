import asyncio
import io
from pathlib import Path

import pytest
from PIL import Image

from imageforge_worker.domain import GenerationSettings
from imageforge_worker.inference import GenerationJob, MageFlowInferenceAdapter
from imageforge_worker.inference.mageflow import build_workflow
from imageforge_worker.model_profiles import MAGE_FLOW_TURBO_BF16, MAGE_FLOW_TURBO_INT8
from tests.conftest import auth, worker_client


def _job(**overrides) -> GenerationJob:
    defaults = {
        "index": 1,
        "prompt": "a red bicycle against a blue door",
        "seed": 7,
        "settings": GenerationSettings(),
    }
    return GenerationJob(**{**defaults, **overrides})


def test_the_text_encoder_supplies_the_latent():
    """TextEncodeMageFlowEdit emits the latent; a generic empty latent is the wrong space.

    Guessing this wrong produced garbled text and broken compositions during the
    staging spike, and the output still looked plausible enough to mislead.
    """

    workflow = build_workflow(MAGE_FLOW_TURBO_INT8, _job())

    sampler = workflow["6"]["inputs"]
    assert sampler["latent_image"] == ["5", 2]
    assert sampler["positive"] == ["5", 0]
    assert sampler["negative"] == ["5", 1]
    class_types = {node["class_type"] for node in workflow.values()}
    assert "EmptySD3LatentImage" not in class_types
    assert "EmptyLatentImage" not in class_types


def test_the_clip_loader_uses_the_mage_type():
    """Any other CLIPLoader type silently degrades text rendering."""

    workflow = build_workflow(MAGE_FLOW_TURBO_INT8, _job())
    assert workflow["3"]["inputs"]["type"] == "mage"


def test_the_workflow_carries_the_profile_sampling_settings():
    workflow = build_workflow(MAGE_FLOW_TURBO_INT8, _job(seed=1234))
    sampler = workflow["6"]["inputs"]
    assert sampler["seed"] == 1234
    assert sampler["steps"] == MAGE_FLOW_TURBO_INT8.steps
    assert sampler["cfg"] == MAGE_FLOW_TURBO_INT8.guidance
    assert sampler["sampler_name"] == "euler"
    assert sampler["scheduler"] == "simple"
    assert sampler["denoise"] == 1.0


def test_the_workflow_requests_the_job_resolution():
    settings = GenerationSettings(width=1280, height=720)
    encode = build_workflow(MAGE_FLOW_TURBO_INT8, _job(settings=settings))["5"]["inputs"]
    assert encode["width"] == 1280
    assert encode["height"] == 720
    assert encode["batch_size"] == 1
    assert encode["prompt"] == "a red bicycle against a blue door"


def test_each_profile_loads_its_own_transformer_by_bare_filename():
    """ComfyUI addresses models by filename inside its own models directory."""

    int8 = build_workflow(MAGE_FLOW_TURBO_INT8, _job())
    bf16 = build_workflow(MAGE_FLOW_TURBO_BF16, _job())
    assert int8["1"]["inputs"]["unet_name"] == "mage_flow_turbo_int8_convrot.safetensors"
    assert bf16["1"]["inputs"]["unet_name"] == "mage_flow_turbo_bf16.safetensors"
    for workflow in (int8, bf16):
        assert workflow["3"]["inputs"]["clip_name"] == "qwen3vl_4b_bf16.safetensors"
        assert workflow["4"]["inputs"]["vae_name"] == "mage_flow_vae_bf16.safetensors"
        assert "/" not in workflow["1"]["inputs"]["unet_name"]


def test_reference_images_are_rejected(tmp_path: Path):
    """Mage-Flow Turbo is text-to-image only, so a reference must fail loudly."""

    adapter = MageFlowInferenceAdapter(tmp_path / "cache", tmp_path / "comfy")
    adapter._process = object()  # the check must precede any ComfyUI traffic
    job = _job(references=(Image.new("RGB", (8, 8)),))

    with pytest.raises(RuntimeError, match="text-to-image only"):
        asyncio.run(adapter.generate(job))


def test_generation_without_a_running_comfyui_fails_loudly(tmp_path: Path):
    adapter = MageFlowInferenceAdapter(tmp_path / "cache", tmp_path / "comfy")
    with pytest.raises(RuntimeError, match="ComfyUI is not running"):
        asyncio.run(adapter.generate(_job()))


def test_startup_refuses_a_missing_comfyui_install(tmp_path: Path):
    adapter = MageFlowInferenceAdapter(tmp_path / "cache", tmp_path / "comfy")
    with pytest.raises(RuntimeError, match="ComfyUI is not installed"):
        adapter._spawn_comfyui()


def test_the_gpu_snapshot_reports_loading_before_startup(tmp_path: Path):
    adapter = MageFlowInferenceAdapter(tmp_path / "cache", tmp_path / "comfy")
    snapshot = adapter.gpu_snapshot()
    assert snapshot["state"] == "loading"
    assert snapshot["available"] is False
    assert snapshot["approved"] is False


def test_model_links_replace_a_stale_directory(tmp_path: Path):
    """A previous boot can leave real directories where the symlinks belong."""

    snapshot = tmp_path / "snapshot"
    for directory in ("diffusion_models", "text_encoders", "vae"):
        (snapshot / directory).mkdir(parents=True)
    comfy = tmp_path / "comfy"
    stale = comfy / "models" / "diffusion_models"
    stale.mkdir(parents=True)
    (stale / "leftover.safetensors").write_bytes(b"stale")

    MageFlowInferenceAdapter(tmp_path / "cache", comfy)._link_models(snapshot)

    for directory in ("diffusion_models", "text_encoders", "vae"):
        link = comfy / "models" / directory
        assert link.is_symlink()
        assert link.resolve() == (snapshot / directory).resolve()
    assert not (stale / "leftover.safetensors").exists()


@pytest.mark.anyio
async def test_the_api_refuses_reference_images_for_a_text_to_image_model(tmp_path: Path):
    """A batch that would fail on every image must be refused at submission."""

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "teal").save(buffer, format="PNG")

    async with worker_client(tmp_path / "volume") as (client, _, _):
        response = await client.post(
            "/v1/batches",
            headers=auth(),
            json={
                "prompts": ["a red bicycle"],
                "references": [
                    {
                        "name": "ref.png",
                        "mime_type": "image/png",
                        "data_hex": buffer.getvalue().hex(),
                    }
                ],
            },
        )

    assert response.status_code == 422
    assert "references_unsupported" in response.text
