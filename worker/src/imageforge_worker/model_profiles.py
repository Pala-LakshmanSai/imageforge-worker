from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Everything that identifies one pinned image model to the worker.

    Model identity used to be spread across constants, the Dockerfile, the
    TypeScript contracts and the Rust native layer, which is what made changing
    models expensive. New models are described here once.
    """

    backend: str
    model_id: str
    revision: str
    precision: str
    steps: int
    guidance: float
    supports_references: bool
    min_gpu_memory_mib: int
    required_files: tuple[str, ...]


FLUX2_KLEIN: Final = ModelProfile(
    backend="flux",
    model_id="black-forest-labs/FLUX.2-klein-4B",
    revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
    precision="bfloat16",
    steps=4,
    guidance=1.0,
    supports_references=True,
    min_gpu_memory_mib=16_380,
    required_files=(
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/tokenizer.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ),
)

# Mage-Flow is not loadable through Diffusers: release 0.39.0 is the latest and
# exports no Mage-Flow pipeline. The INT8 ConvRot checkpoint is a ComfyUI-format
# single file, and ComfyUI core is its only public loader, so this profile names
# the public Comfy-Org mirror rather than the gated microsoft/* repositories.
# See docs/MAGEFLOW_STAGING.md for the spike evidence.
MAGE_FLOW_TURBO_INT8: Final = ModelProfile(
    backend="mageflow",
    model_id="Comfy-Org/Mage-Flow",
    revision="d8c99241f6fa80fbd453014234af2bf337ea21e6",
    precision="int8-convrot",
    steps=4,
    guidance=1.0,
    # The Turbo checkpoint is text-to-image only; the editing variant is a
    # separate model and deliberately out of scope.
    supports_references=False,
    min_gpu_memory_mib=16_380,
    required_files=(
        "diffusion_models/mage_flow_turbo_int8_convrot.safetensors",
        "text_encoders/qwen3vl_4b_bf16.safetensors",
        "vae/mage_flow_vae_bf16.safetensors",
    ),
)

# Same model, unquantized. Kept as the reference the INT8 checkpoint is scored
# against, and as the fallback if INT8 fails the quality gate.
MAGE_FLOW_TURBO_BF16: Final = ModelProfile(
    backend="mageflow-bf16",
    model_id=MAGE_FLOW_TURBO_INT8.model_id,
    revision=MAGE_FLOW_TURBO_INT8.revision,
    precision="bfloat16",
    steps=MAGE_FLOW_TURBO_INT8.steps,
    guidance=MAGE_FLOW_TURBO_INT8.guidance,
    supports_references=False,
    min_gpu_memory_mib=MAGE_FLOW_TURBO_INT8.min_gpu_memory_mib,
    required_files=(
        "diffusion_models/mage_flow_turbo_bf16.safetensors",
        "text_encoders/qwen3vl_4b_bf16.safetensors",
        "vae/mage_flow_vae_bf16.safetensors",
    ),
)

_PROFILES: Final = {
    profile.backend: profile
    for profile in (FLUX2_KLEIN, MAGE_FLOW_TURBO_INT8, MAGE_FLOW_TURBO_BF16)
}


# The profile the shipped constants describe, and therefore what the worker
# reports and enforces regardless of which adapter is instantiated.
ACTIVE_PROFILE: Final = MAGE_FLOW_TURBO_INT8


def profile_for_backend(backend: str) -> ModelProfile:
    try:
        return _PROFILES[backend]
    except KeyError as error:
        raise ValueError(f"unknown inference backend: {backend}") from error


def supported_backends() -> tuple[str, ...]:
    return tuple(_PROFILES)
