from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class WorkerError(Exception):
    status_code: int
    code: str
    message: str
    details: Mapping[str, Any] | None = None
    headers: Mapping[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.code


class InferenceFailure(Exception):
    def __init__(self, code: str = "inference_failed") -> None:
        self.code = code
        super().__init__(code)
