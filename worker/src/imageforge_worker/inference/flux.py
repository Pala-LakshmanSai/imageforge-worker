from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from ..constants import (
    MIN_CUDA_VERSION,
    MIN_GPU_MEMORY_BYTES,
    MIN_GPU_MEMORY_MIB,
    MODEL_ALLOW_PATTERNS,
    MODEL_ID,
    MODEL_REVISION,
    REQUIRED_MODEL_FILES,
)
from ..domain import GenerationSettings, HealthPhase
from .base import GenerationJob, InferenceResult, PhaseReporter
from .encoding import encode_result


class FluxInferenceAdapter:
    """Pinned, offline-only BF16 FLUX.2 Klein adapter for one approved CUDA GPU."""

    def __init__(self, model_cache_dir: Path) -> None:
        self.model_cache_dir = model_cache_dir
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._gpu_name: str | None = None
        self._gpu_total_memory_bytes = 0

    async def startup(self, report_phase: PhaseReporter) -> None:
        await report_phase(HealthPhase.WEIGHTS, 0.1)
        snapshot_path = await asyncio.to_thread(self._resolve_local_snapshot)
        await report_phase(HealthPhase.WEIGHTS, 1.0)
        await report_phase(HealthPhase.GPU_LOAD, 0.05)
        await asyncio.to_thread(self._load_pipeline, snapshot_path)
        await report_phase(HealthPhase.GPU_LOAD, 1.0)
        await report_phase(HealthPhase.WARMUP, 0.1)
        warmup = GenerationJob(
            index=1,
            prompt="A neutral studio lighting calibration chart",
            seed=0,
            settings=GenerationSettings(),
        )
        await self.generate(warmup)
        if self._torch is not None:
            self._torch.cuda.synchronize()
            self._torch.cuda.reset_peak_memory_stats(0)
        await report_phase(HealthPhase.WARMUP, 1.0)
        await report_phase(HealthPhase.READY, 1.0)

    def _resolve_local_snapshot(self) -> str:
        self.model_cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download

        # local_files_only is deliberate: normal Pod starts never download model weights.
        snapshot_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(self.model_cache_dir),
            local_files_only=True,
            allow_patterns=list(MODEL_ALLOW_PATTERNS),
        )
        missing = [
            relative_name
            for relative_name in REQUIRED_MODEL_FILES
            if not (Path(snapshot_path) / relative_name).is_file()
        ]
        if missing:
            raise RuntimeError("the pinned local FLUX Diffusers snapshot is incomplete")
        return snapshot_path

    def _load_pipeline(self, snapshot_path: str) -> None:
        import torch
        from diffusers import Flux2KleinPipeline

        if not torch.cuda.is_available():
            self._torch = torch
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            self._torch = torch
            raise RuntimeError("ImageForge requires exactly one visible CUDA GPU")
        cuda_version = _parse_cuda_version(torch.version.cuda)
        if cuda_version < MIN_CUDA_VERSION:
            self._torch = torch
            raise RuntimeError("ImageForge requires a PyTorch CUDA runtime of at least 13.0")
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        self._torch = torch
        self._gpu_name = gpu_name
        self._gpu_total_memory_bytes = total_memory
        if "NVIDIA" not in gpu_name.upper():
            raise RuntimeError("the visible CUDA device is not an NVIDIA GPU")
        if total_memory < MIN_GPU_MEMORY_BYTES:
            raise RuntimeError(f"the visible GPU has less than {MIN_GPU_MEMORY_MIB} MiB of VRAM")

        pipeline = Flux2KleinPipeline.from_pretrained(
            snapshot_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
        )
        pipeline.set_progress_bar_config(disable=True)
        transformer_parameter = next(pipeline.transformer.parameters())
        if transformer_parameter.device.type != "cuda":
            raise RuntimeError("FLUX transformer was not loaded directly on CUDA")
        if transformer_parameter.dtype != torch.bfloat16:
            raise RuntimeError("FLUX transformer is not using BF16")
        self._pipeline = pipeline

    async def generate(self, job: GenerationJob) -> InferenceResult:
        if self._pipeline is None or self._torch is None:
            raise RuntimeError("FLUX pipeline is not loaded")
        return await asyncio.to_thread(self._generate_sync, job)

    def _generate_sync(self, job: GenerationJob) -> InferenceResult:
        torch = self._torch
        pipeline = self._pipeline
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        pipeline_kwargs: dict[str, Any] = {
            "prompt": job.prompt,
            "height": job.settings.height,
            "width": job.settings.width,
            "guidance_scale": job.settings.guidance,
            "num_inference_steps": job.settings.steps,
            "generator": torch.Generator(device="cuda").manual_seed(job.seed),
        }
        if job.references:
            pipeline_kwargs["image"] = list(job.references)
        with torch.inference_mode():
            image = pipeline(**pipeline_kwargs).images[0]
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - inference_started) * 1000

        return encode_result(image, job.settings, inference_ms)

    async def shutdown(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            del pipeline
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def gpu_snapshot(self) -> dict[str, object]:
        if self._torch is None or not self._torch.cuda.is_available():
            return {
                "state": "loading",
                "available": False,
                "approved": False,
                "name": self._gpu_name,
                "device_count": 0,
                "total_memory_bytes": self._gpu_total_memory_bytes,
                "memory_allocated_bytes": 0,
                "memory_reserved_bytes": 0,
                "peak_memory_allocated_bytes": 0,
                "peak_memory_reserved_bytes": 0,
            }
        torch = self._torch
        return {
            "state": "ready" if self._pipeline is not None else "loading",
            "available": True,
            "approved": self._gpu_total_memory_bytes >= MIN_GPU_MEMORY_BYTES,
            "name": self._gpu_name,
            "device_count": torch.cuda.device_count(),
            "total_memory_bytes": self._gpu_total_memory_bytes,
            "memory_allocated_bytes": torch.cuda.memory_allocated(0),
            "memory_reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }


def _parse_cuda_version(raw: object) -> tuple[int, int]:
    if not isinstance(raw, str):
        raise RuntimeError("PyTorch did not report its compiled CUDA runtime")
    parts = raw.split(".", maxsplit=2)
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("PyTorch reported an invalid compiled CUDA runtime") from exc
