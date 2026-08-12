# syntax=docker/dockerfile:1.7
#
# RunPod's CUDA 13 driver is supplied by the host.  Keep the userspace CUDA
# runtime inside the pinned PyTorch wheel instead of inheriting the PyTorch
# image: the current image tag contains Python 3.12, while the worker contract
# is Python 3.11.  This also makes the build reproducible on GitHub Actions.
FROM python:3.11-slim-bookworm@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941

LABEL org.opencontainers.image.title="ImageForge Worker" \
      org.opencontainers.image.version="0.2.0" \
      ai.imageforge.api-schema="1" \
      ai.imageforge.cuda-min="13.0" \
      ai.imageforge.python="3.11" \
      ai.imageforge.torch-wheel="torch-2.11.0+cu130-cp311-cp311-manylinux_2_28_x86_64" \
      ai.imageforge.model="Comfy-Org/Mage-Flow" \
      ai.imageforge.model-revision="d8c99241f6fa80fbd453014234af2bf337ea21e6" \
      ai.imageforge.model-file="mage_flow_turbo_int8_convrot.safetensors" \
      ai.imageforge.comfyui-revision="26d7f8556822d9d08c2d3e1878636ac3b4969af9"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DIFFUSERS_OFFLINE=1 \
    HF_HOME=/workspace/models/huggingface \
    IMAGEFORGE_DATA_ROOT=/workspace/imageforge \
    IMAGEFORGE_MODEL_CACHE_DIR=/workspace/models/huggingface \
    IMAGEFORGE_INFERENCE_BACKEND=mageflow \
    IMAGEFORGE_COMFYUI_ROOT=/opt/comfyui

WORKDIR /opt/imageforge-worker

COPY scripts/verify_comfyui_nodes.py ./verify_comfyui_nodes.py

# Keep the OS and CUDA layers independent of worker source changes so a normal
# release rebuild reuses the expensive CUDA wheel layer. Installation happens
# while the image is built, never during normal Pod boot.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
      libgomp1 \
      libglib2.0-0 \
      libgl1 \
      libnuma1 \
      gcc \
      libc6-dev \
      git \
    && rm -rf /var/lib/apt/lists/* \
    && python --version | grep -Eq '^Python 3\.11\.' \
    && python -m pip install --no-cache-dir --no-deps \
      'https://download.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=225b22e0a4e36ea573d3a68796e6816a160616f67e8b8c55683a88bf7777f4cd'


# ComfyUI is the only public loader for the Mage-Flow INT8 ConvRot checkpoint:
# Diffusers 0.39.0 is the latest release and exports no Mage-Flow pipeline.
# It is baked in at a pinned commit so a Pod boot never clones anything.
#
# ComfyUI's requirements leave torch, torchvision and torchaudio unpinned, which
# would silently replace the CUDA 13 wheel installed above with a default-index
# build. The constraint file below pins all three to one matched cu130 set.
#
# That set is torch 2.11 rather than 2.13: ComfyUI core imports torchaudio (via
# comfy/sd.py), and torchaudio has no build for torch 2.12 or 2.13 on any index.
# Mixing an older torchaudio against a newer torch risks a C++ ABI mismatch, so
# the whole set moves together.
ARG COMFYUI_REVISION=26d7f8556822d9d08c2d3e1878636ac3b4969af9
RUN git clone --filter=blob:none https://github.com/comfyanonymous/ComfyUI.git /opt/comfyui \
    && git -C /opt/comfyui checkout --detach "${COMFYUI_REVISION}" \
    && rm -rf /opt/comfyui/.git \
    && python -m pip install --no-cache-dir \
      'https://download.pytorch.org/whl/cu130/torchvision-0.26.0%2Bcu130-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=3b53e3b611561e03ac261d06cb3f38782120ad9e0b4cd9f01549799097c713a6' \
      'https://download.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=f6e1e3ab31abfa81f5c8801a0eab769d5308514dc644bb39dcf0a9057f364f16' \
    && printf 'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\ntorchaudio==2.11.0+cu130\n' \
      > /tmp/comfy-constraints.txt \
    && python -m pip install --no-cache-dir --constraint /tmp/comfy-constraints.txt \
      --requirement /opt/comfyui/requirements.txt \
    && rm /tmp/comfy-constraints.txt \
    && python -c "import torch; assert torch.__version__.startswith('2.11.0+cu130'), torch.__version__" \
    && python /opt/imageforge-worker/verify_comfyui_nodes.py

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir '.[gpu]' \
    && python -m pip check

EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=300s --retries=60 \
  CMD python -c "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=2)); assert r['process']['status']=='ok' and r['phase']=='ready' and r['model']['status']=='ready'" || exit 1

# One Uvicorn process is used per Pod; cross-Pod flock requires the documented
# paid validation gate for the selected RunPod network-volume configuration.
CMD ["python", "-m", "imageforge_worker"]
