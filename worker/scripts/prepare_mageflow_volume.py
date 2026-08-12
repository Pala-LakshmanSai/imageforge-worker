"""One-time Mage-Flow Turbo download for an ImageForge network volume.

Normal worker boot never downloads weights. This command is the only place that
turns Hugging Face networking back on, and it requires an explicit confirmation
flag exactly like `imageforge_worker.prepare_model` does for FLUX.

The INT8 transformer, its Qwen3-VL text encoder, and the Mage-VAE come from the
ComfyUI-format `Comfy-Org/Mage-Flow` repository as single files. The Diffusers
scaffolding (`model_index.json`, `scheduler/`, `tokenizer/`, config files) comes
from `microsoft/Mage-Flow-Turbo`, which is what a Diffusers pipeline needs
around the single-file transformer.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

COMFY_REPO = "Comfy-Org/Mage-Flow"
INT8_TRANSFORMER_FILE = "diffusion_models/mage_flow_turbo_int8_convrot.safetensors"
BF16_TRANSFORMER_FILE = "diffusion_models/mage_flow_turbo_bf16.safetensors"
COMFY_FILES = (
    INT8_TRANSFORMER_FILE,
    "text_encoders/qwen3vl_4b_bf16.safetensors",
    "vae/mage_flow_vae_bf16.safetensors",
)

DIFFUSERS_REPO = "microsoft/Mage-Flow-Turbo"
# The transformer is deliberately excluded: the INT8 single file above replaces
# it, and the BF16 folder would otherwise duplicate 8.23 GB on the volume.
DIFFUSERS_ALLOW_PATTERNS = (
    "model_index.json",
    "scheduler/*",
    "text_encoder/config.json",
    "tokenizer/*",
    "vae/config.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time Mage-Flow Turbo preparation for an ImageForge network volume"
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="required acknowledgement that this one-time command downloads model weights",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="pin both repositories to an explicit revision once the spike has recorded one",
    )
    parser.add_argument(
        "--include-bf16-fallback",
        action="store_true",
        help="also fetch the 8.23 GB BF16 transformer used by the Path B fallback",
    )
    arguments = parser.parse_args()
    if not arguments.confirm_download:
        parser.error("--confirm-download is required; normal worker boot never downloads weights")

    arguments.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    wanted = list(COMFY_FILES)
    if arguments.include_bf16_fallback:
        wanted.append(BF16_TRANSFORMER_FILE)

    with _hub_download_enabled():
        from huggingface_hub import hf_hub_download, snapshot_download

        for relative_name in wanted:
            path = hf_hub_download(
                repo_id=COMFY_REPO,
                filename=relative_name,
                revision=arguments.revision,
                cache_dir=str(arguments.cache_dir),
            )
            print(f"prepared {relative_name} at {path}")

        snapshot_path = snapshot_download(
            repo_id=DIFFUSERS_REPO,
            revision=arguments.revision,
            cache_dir=str(arguments.cache_dir),
            allow_patterns=list(DIFFUSERS_ALLOW_PATTERNS),
        )
        print(f"prepared the Diffusers scaffolding at {snapshot_path}")

    missing = [name for name in ("model_index.json",) if not (Path(snapshot_path) / name).is_file()]
    if missing:
        raise SystemExit("the Mage-Flow Diffusers scaffolding is incomplete")
    print("Mage-Flow Turbo preparation complete")


@contextmanager
def _hub_download_enabled() -> Iterator[None]:
    """Override only Hub offline mode for the explicitly confirmed preparation call."""

    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


if __name__ == "__main__":
    main()
