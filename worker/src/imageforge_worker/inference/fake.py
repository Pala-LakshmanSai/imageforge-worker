from __future__ import annotations

import asyncio
import hashlib
import io
import time
from collections.abc import Mapping

from PIL import Image, ImageDraw

from ..domain import HealthPhase
from ..errors import InferenceFailure
from .base import GenerationJob, InferenceResult, PhaseReporter


class FakeInferenceAdapter:
    """Fast, deterministic, valid-image inference for local and CI tests."""

    def __init__(
        self,
        *,
        delay_seconds: float = 0.0,
        startup_delay_seconds: float = 0.0,
        failures_before_success: Mapping[int, int] | None = None,
        first_generation_started: asyncio.Event | None = None,
        release_first_generation: asyncio.Event | None = None,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.startup_delay_seconds = startup_delay_seconds
        self.failures_before_success = dict(failures_before_success or {})
        self.first_generation_started = first_generation_started
        self.release_first_generation = release_first_generation
        self.calls_by_index: dict[int, int] = {}
        self.generated_indices: list[int] = []
        self.phase_history: list[str] = []

    async def startup(self, report_phase: PhaseReporter) -> None:
        for phase in (
            HealthPhase.WEIGHTS,
            HealthPhase.GPU_LOAD,
            HealthPhase.WARMUP,
            HealthPhase.READY,
        ):
            self.phase_history.append(phase.value)
            await report_phase(phase, 1.0 if phase == HealthPhase.READY else 0.5)
            if self.startup_delay_seconds:
                await asyncio.sleep(self.startup_delay_seconds)

    async def generate(self, job: GenerationJob) -> InferenceResult:
        if job.index == 1 and self.first_generation_started is not None:
            self.first_generation_started.set()
            if self.release_first_generation is not None:
                await self.release_first_generation.wait()
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        call_count = self.calls_by_index.get(job.index, 0) + 1
        self.calls_by_index[job.index] = call_count
        if call_count <= self.failures_before_success.get(job.index, 0):
            raise InferenceFailure("fake_inference_failure")

        started = time.perf_counter()
        digest = hashlib.sha256(f"{job.index}\0{job.seed}\0{job.prompt}".encode()).digest()
        color = tuple(24 + byte % 192 for byte in digest[:3])
        image = Image.new("RGB", (job.settings.width, job.settings.height), color)
        draw = ImageDraw.Draw(image)
        # Hash-derived geometry makes prompt/seed changes observable without rendering prompt text.
        for offset in range(3):
            left = int(digest[3 + offset] / 255 * (job.settings.width - 180))
            top = int(digest[6 + offset] / 255 * (job.settings.height - 100))
            fill = tuple(digest[9 + offset * 3 + channel] for channel in range(3))
            draw.rectangle((left, top, left + 180, top + 100), fill=fill)
        inference_ms = (time.perf_counter() - started) * 1000

        jpeg_started = time.perf_counter()
        jpeg_buffer = io.BytesIO()
        image.save(
            jpeg_buffer,
            format="JPEG",
            quality=job.settings.jpeg_quality,
            optimize=False,
            progressive=False,
            subsampling=0,
        )
        jpeg_encode_ms = (time.perf_counter() - jpeg_started) * 1000

        preview_started = time.perf_counter()
        preview = image.resize(
            (job.settings.preview_width, job.settings.preview_height), Image.Resampling.LANCZOS
        )
        preview_buffer = io.BytesIO()
        preview.save(preview_buffer, format="WEBP", quality=85, method=4, exact=True)
        preview_encode_ms = (time.perf_counter() - preview_started) * 1000
        self.generated_indices.append(job.index)
        return InferenceResult(
            jpeg=jpeg_buffer.getvalue(),
            preview=preview_buffer.getvalue(),
            inference_ms=inference_ms,
            jpeg_encode_ms=jpeg_encode_ms,
            preview_encode_ms=preview_encode_ms,
        )

    async def shutdown(self) -> None:
        return None

    def gpu_snapshot(self) -> dict[str, object]:
        return {
            "state": "ready",
            "available": True,
            "approved": True,
            "name": "ImageForge deterministic fake GPU",
            "device_count": 1,
            "total_memory_bytes": 24 * 1024**3,
            "memory_allocated_bytes": 0,
            "memory_reserved_bytes": 0,
            "peak_memory_allocated_bytes": 0,
            "peak_memory_reserved_bytes": 0,
        }
