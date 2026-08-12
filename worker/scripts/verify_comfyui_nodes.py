"""Fail the image build unless ComfyUI provides the nodes the worker uses.

The Mage-Flow workflow names seven node classes. If a ComfyUI revision renames
or drops any of them, the failure would otherwise surface as a runtime workflow
validation error on a live Pod, after the model had already loaded.

It runs on a GPU-less build machine, so it forces ComfyUI onto CPU. That is
enough to populate the node registry, which is all this verifies.
"""

from __future__ import annotations

import asyncio
import sys

COMFYUI_ROOT = "/opt/comfyui"
REQUIRED_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "TextEncodeMageFlowEdit",
    "KSampler",
    "VAEDecode",
    "SaveImage",
)


def main() -> None:
    sys.path.insert(0, COMFYUI_ROOT)

    # Image builders have no GPU, and ComfyUI picks its device while importing
    # comfy.model_management. It ignores argv unless argument parsing is enabled
    # first, which main.py normally does and a direct import does not, so both
    # steps have to happen before `nodes` is imported.
    import comfy.options

    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--cpu"]

    import nodes

    import_failed = asyncio.run(
        nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    )
    available = set(nodes.NODE_CLASS_MAPPINGS)
    missing = [name for name in REQUIRED_NODES if name not in available]
    if missing:
        raise SystemExit(
            f"ComfyUI is missing node class(es) the Mage-Flow workflow needs: {missing}. "
            f"Modules that failed to import: {import_failed}"
        )
    print(f"verified {len(REQUIRED_NODES)} required ComfyUI nodes out of {len(available)}")
    if import_failed:
        print(f"extra-node modules skipped (expected for audio): {import_failed}")


if __name__ == "__main__":
    main()
