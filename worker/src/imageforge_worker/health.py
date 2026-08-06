from __future__ import annotations

import asyncio
import time
from typing import Any

from .constants import API_SCHEMA_VERSION, MODEL_ID, MODEL_PRECISION, MODEL_REVISION, WORKER_VERSION
from .domain import HealthPhase


class HealthTracker:
    def __init__(self, runtime_metadata: dict[str, str] | None = None) -> None:
        self._lock = asyncio.Lock()
        self._started = time.monotonic()
        self._phase_started = self._started
        self._phase = HealthPhase.PROCESS
        self._progress = 0.0
        self._phase_timings_ms: dict[str, float] = {}
        self._error_id: str | None = None
        self._runtime_metadata = dict(runtime_metadata or {})

    @property
    def ready(self) -> bool:
        return self._phase == HealthPhase.READY

    @property
    def phase(self) -> HealthPhase:
        return self._phase

    async def transition(self, phase: HealthPhase, progress: float) -> None:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("health progress must be between zero and one")
        async with self._lock:
            now = time.monotonic()
            if phase != self._phase:
                elapsed = (now - self._phase_started) * 1000
                self._phase_timings_ms[self._phase.value] = round(elapsed, 3)
                self._phase = phase
                self._phase_started = now
            self._progress = progress
            if phase == HealthPhase.READY:
                self._phase_timings_ms.setdefault(phase.value, 0.0)

    async def fail(self, error_id: str) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = (now - self._phase_started) * 1000
            self._phase_timings_ms[self._phase.value] = round(elapsed, 3)
            self._phase = HealthPhase.ERROR
            self._phase_started = now
            self._progress = 0.0
            self._error_id = error_id

    async def snapshot(self, gpu: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            phase_timings = dict(self._phase_timings_ms)
            if self._phase not in {HealthPhase.READY, HealthPhase.ERROR}:
                phase_timings[self._phase.value] = round(
                    (time.monotonic() - self._phase_started) * 1000, 3
                )
            model_status = (
                "ready"
                if self._phase == HealthPhase.READY
                else "error"
                if self._phase == HealthPhase.ERROR
                else "loading"
            )
            payload: dict[str, Any] = {
                "schema_version": API_SCHEMA_VERSION,
                "service": "imageforge-worker",
                "version": WORKER_VERSION,
                "process": {
                    "status": "ok",
                    "uptime_ms": round((time.monotonic() - self._started) * 1000, 3),
                },
                "model": {
                    "id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "precision": MODEL_PRECISION,
                    "status": model_status,
                },
                "gpu": gpu,
                "phase": self._phase.value,
                "phase_progress": self._progress,
                "phase_timings_ms": phase_timings,
                "runtime": self._runtime_metadata,
            }
            if self._error_id:
                payload["error"] = {
                    "code": "worker_boot_failed",
                    "message": "The worker could not become ready.",
                    "error_id": self._error_id,
                }
            return payload
