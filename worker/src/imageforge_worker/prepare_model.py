from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .constants import MODEL_ALLOW_PATTERNS, MODEL_ID, MODEL_REVISION, REQUIRED_MODEL_FILES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time selective FLUX cache preparation for an ImageForge network volume"
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="required acknowledgement that this one-time command downloads model weights",
    )
    arguments = parser.parse_args()
    if not arguments.confirm_download:
        parser.error("--confirm-download is required; normal worker boot never downloads weights")

    with _hub_download_enabled():
        from huggingface_hub import snapshot_download

        snapshot_path = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(arguments.cache_dir),
                allow_patterns=list(MODEL_ALLOW_PATTERNS),
            )
        )
    missing = [name for name in REQUIRED_MODEL_FILES if not (snapshot_path / name).is_file()]
    if missing:
        raise SystemExit("selective model snapshot is incomplete")
    print(f"Prepared pinned ImageForge Diffusers snapshot at {snapshot_path}")


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
