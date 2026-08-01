from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from ..domain import GenerationSettings, HealthPhase

PhaseReporter = Callable[[HealthPhase, float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class GenerationJob:
    index: int
    prompt: str
    seed: int
    settings: GenerationSettings


@dataclass(frozen=True, slots=True)
class InferenceResult:
    jpeg: bytes
    preview: bytes
    inference_ms: float
    jpeg_encode_ms: float
    preview_encode_ms: float


class InferenceAdapter(Protocol):
    async def startup(self, report_phase: PhaseReporter) -> None: ...

    async def generate(self, job: GenerationJob) -> InferenceResult: ...

    async def shutdown(self) -> None: ...

    def gpu_snapshot(self) -> dict[str, object]: ...
