# syntax=docker/dockerfile:1.7
#
# RunPod's CUDA 13 driver is supplied by the host.  Keep the userspace CUDA
# runtime inside the pinned PyTorch wheel instead of inheriting the PyTorch
# image: the current image tag contains Python 3.12, while the worker contract
# is Python 3.11.  This also makes the build reproducible on GitHub Actions.
FROM python:3.11-slim-bookworm@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941

LABEL org.opencontainers.image.title="ImageForge Worker" \
      org.opencontainers.image.version="0.1.2" \
      ai.imageforge.api-schema="1" \
      ai.imageforge.cuda-min="13.0" \
      ai.imageforge.python="3.11" \
      ai.imageforge.torch-wheel="torch-2.13.0+cu130-cp311-cp311-manylinux_2_28_x86_64" \
      ai.imageforge.model="black-forest-labs/FLUX.2-klein-4B" \
      ai.imageforge.model-revision="e7b7dc27f91deacad38e78976d1f2b499d76a294"

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
    IMAGEFORGE_INFERENCE_BACKEND=flux

WORKDIR /opt/imageforge-worker

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
    && rm -rf /var/lib/apt/lists/* \
    && python --version | grep -Eq '^Python 3\.11\.' \
    && python -m pip install --no-cache-dir --no-deps \
      'https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=794664c05a3470a5e738447b54495cc6bbf1efca48bf2794270a897d9f4356c4'

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
