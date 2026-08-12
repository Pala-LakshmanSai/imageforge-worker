from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from ..constants import MIN_GPU_MEMORY_BYTES, MIN_GPU_MEMORY_MIB
from ..domain import GenerationSettings, HealthPhase
from ..model_profiles import MAGE_FLOW_TURBO_INT8, ModelProfile
from .base import GenerationJob, InferenceResult, PhaseReporter
from .encoding import encode_result

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_STARTUP_TIMEOUT_SECONDS = 300.0
COMFY_POLL_INTERVAL_SECONDS = 0.25
GENERATION_TIMEOUT_SECONDS = 600.0

# ComfyUI reads models from directory names it owns, so the pinned snapshot is
# linked into a private tree rather than copied.
COMFY_MODEL_DIRECTORIES = ("diffusion_models", "text_encoders", "vae")


class MageFlowInferenceAdapter:
    """Mage-Flow Turbo INT8 driven through a private headless ComfyUI process.

    Diffusers cannot load Mage-Flow: release 0.39.0 is the latest and exports no
    Mage-Flow pipeline, and the INT8 ConvRot checkpoint is a ComfyUI-format
    single file whose only public loader is ComfyUI core. The worker therefore
    owns a ComfyUI child process bound to loopback, and speaks to it over its
    HTTP API. See docs/MAGEFLOW_STAGING.md for the spike evidence.
    """

    def __init__(
        self,
        model_cache_dir: Path,
        comfy_root: Path,
        profile: ModelProfile = MAGE_FLOW_TURBO_INT8,
    ) -> None:
        self.model_cache_dir = model_cache_dir
        self.comfy_root = comfy_root
        self.profile = profile
        self._process: subprocess.Popen[bytes] | None = None
        self._torch: Any | None = None
        self._gpu_name: str | None = None
        self._gpu_total_memory_bytes = 0
        self._ready = False

    async def startup(self, report_phase: PhaseReporter) -> None:
        await report_phase(HealthPhase.WEIGHTS, 0.1)
        snapshot_path = await asyncio.to_thread(self._resolve_local_snapshot)
        await asyncio.to_thread(self._link_models, Path(snapshot_path))
        await report_phase(HealthPhase.WEIGHTS, 1.0)

        await report_phase(HealthPhase.GPU_LOAD, 0.05)
        await asyncio.to_thread(self._verify_gpu)
        await asyncio.to_thread(self._spawn_comfyui)
        await self._await_comfyui(report_phase)
        await report_phase(HealthPhase.GPU_LOAD, 1.0)

        # ComfyUI loads weights lazily on the first prompt, so an unwarmed
        # worker would charge the first real request roughly a minute.
        await report_phase(HealthPhase.WARMUP, 0.1)
        await self.generate(
            GenerationJob(
                index=1,
                prompt="A neutral studio lighting calibration chart",
                seed=0,
                settings=GenerationSettings(),
            )
        )
        if self._torch is not None:
            self._torch.cuda.reset_peak_memory_stats(0)
        await report_phase(HealthPhase.WARMUP, 1.0)
        self._ready = True
        await report_phase(HealthPhase.READY, 1.0)

    def _resolve_local_snapshot(self) -> str:
        self.model_cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download

        # local_files_only is deliberate: normal Pod starts never download weights.
        snapshot_path = snapshot_download(
            repo_id=self.profile.model_id,
            revision=self.profile.revision,
            cache_dir=str(self.model_cache_dir),
            local_files_only=True,
            allow_patterns=list(self.profile.required_files),
        )
        missing = [
            relative_name
            for relative_name in self.profile.required_files
            if not (Path(snapshot_path) / relative_name).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"the pinned local Mage-Flow snapshot is missing {len(missing)} file(s)"
            )
        return snapshot_path

    def _link_models(self, snapshot_path: Path) -> None:
        models_root = self.comfy_root / "models"
        models_root.mkdir(parents=True, exist_ok=True)
        for directory in COMFY_MODEL_DIRECTORIES:
            link = models_root / directory
            if link.is_symlink() or link.exists():
                if link.is_symlink() or link.is_file():
                    link.unlink()
                else:
                    shutil.rmtree(link)
            link.symlink_to(snapshot_path / directory, target_is_directory=True)

    def _verify_gpu(self) -> None:
        import torch

        self._torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("ImageForge requires exactly one visible CUDA GPU")
        self._gpu_name = torch.cuda.get_device_name(0)
        self._gpu_total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        if "NVIDIA" not in self._gpu_name.upper():
            raise RuntimeError("the visible CUDA device is not an NVIDIA GPU")
        if self._gpu_total_memory_bytes < MIN_GPU_MEMORY_BYTES:
            raise RuntimeError(f"the visible GPU has less than {MIN_GPU_MEMORY_MIB} MiB of VRAM")

    def _spawn_comfyui(self) -> None:
        main = self.comfy_root / "main.py"
        if not main.is_file():
            raise RuntimeError(f"ComfyUI is not installed at {self.comfy_root}")
        environment = dict(os.environ)
        # ComfyUI must never reach the network; the weights are already local.
        environment["HF_HUB_OFFLINE"] = "1"
        self._process = subprocess.Popen(  # noqa: S603 - a fixed argv, no shell
            [
                sys.executable,
                str(main),
                "--listen",
                COMFY_HOST,
                "--port",
                str(COMFY_PORT),
                "--disable-auto-launch",
                "--disable-metadata",
            ],
            cwd=str(self.comfy_root),
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    async def _await_comfyui(self, report_phase: PhaseReporter) -> None:
        deadline = time.monotonic() + COMFY_STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI exited during startup with code {self._process.returncode}"
                )
            try:
                await asyncio.to_thread(self._request, "GET", "/system_stats", None)
                return
            except (urllib.error.URLError, OSError, TimeoutError):
                elapsed = COMFY_STARTUP_TIMEOUT_SECONDS - (deadline - time.monotonic())
                await report_phase(
                    HealthPhase.GPU_LOAD, min(0.95, 0.05 + elapsed / COMFY_STARTUP_TIMEOUT_SECONDS)
                )
                await asyncio.sleep(1.0)
        raise RuntimeError("ComfyUI did not become ready within the startup timeout")

    async def generate(self, job: GenerationJob) -> InferenceResult:
        if self._process is None:
            raise RuntimeError("ComfyUI is not running")
        if job.references:
            raise RuntimeError("Mage-Flow Turbo is text-to-image only and rejects reference images")
        return await asyncio.to_thread(self._generate_sync, job)

    def _generate_sync(self, job: GenerationJob) -> InferenceResult:
        workflow = build_workflow(self.profile, job)
        inference_started = time.perf_counter()
        queued = self._request("POST", "/prompt", {"prompt": workflow})
        prompt_id = queued["prompt_id"]

        deadline = time.monotonic() + GENERATION_TIMEOUT_SECONDS
        while True:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError("ComfyUI exited while a generation was in flight")
            history = self._request("GET", f"/history/{prompt_id}", None)
            if prompt_id in history:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("ComfyUI did not finish the generation within the timeout")
            time.sleep(COMFY_POLL_INTERVAL_SECONDS)

        entry = history[prompt_id]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError("ComfyUI reported a workflow error")
        image = self._fetch_single_image(entry)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        return encode_result(image, job.settings, inference_ms)

    def _fetch_single_image(self, entry: dict[str, Any]) -> Image.Image:
        descriptors = [
            descriptor
            for node_output in entry.get("outputs", {}).values()
            for descriptor in node_output.get("images", [])
        ]
        if len(descriptors) != 1:
            raise RuntimeError(f"ComfyUI returned {len(descriptors)} images, expected exactly one")
        descriptor = descriptors[0]
        query = urllib.parse.urlencode(
            {
                "filename": descriptor["filename"],
                "subfolder": descriptor.get("subfolder", ""),
                "type": descriptor.get("type", "output"),
            }
        )
        request = urllib.request.Request(f"{COMFY_BASE}/view?{query}")  # noqa: S310
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            return Image.open(io.BytesIO(response.read()))

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> Any:
        data = None if payload is None else json.dumps(payload).encode()
        headers = {} if payload is None else {"Content-Type": "application/json"}
        # COMFY_BASE is a fixed loopback http:// constant, not caller-supplied.
        request = urllib.request.Request(  # noqa: S310
            f"{COMFY_BASE}{path}", data=data, headers=headers, method=method
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            return json.load(response)

    async def shutdown(self) -> None:
        self._ready = False
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            await asyncio.to_thread(process.wait, 20)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            await asyncio.to_thread(process.wait, 10)

    def gpu_snapshot(self) -> dict[str, object]:
        torch = self._torch
        if torch is None or not torch.cuda.is_available():
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
        return {
            "state": "ready" if self._ready else "loading",
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


def build_workflow(profile: ModelProfile, job: GenerationJob) -> dict[str, Any]:
    """The official Comfy-Org `image_mage_flow_turbo_t2i_int8` graph.

    Two details are load-bearing and cost a wasted staging run when guessed:
    `CLIPLoader.type` must be `mage`, and `TextEncodeMageFlowEdit` emits the
    latent itself, so no `EmptySD3LatentImage` belongs in this graph.
    """

    transformer = Path(profile.required_files[0]).name
    text_encoder = Path(profile.required_files[1]).name
    vae = Path(profile.required_files[2]).name
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": transformer, "weight_dtype": "default"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder, "type": "mage", "device": "default"},
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "5": {
            "class_type": "TextEncodeMageFlowEdit",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "prompt": job.prompt,
                "negative_prompt": "",
                "width": job.settings.width,
                "height": job.settings.height,
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["5", 0],
                "negative": ["5", 1],
                "latent_image": ["5", 2],
                "seed": job.seed,
                "steps": profile.steps,
                "cfg": profile.guidance,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["4", 0]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "imageforge"},
        },
    }
