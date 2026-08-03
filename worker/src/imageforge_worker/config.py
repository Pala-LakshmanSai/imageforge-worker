from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import MAX_GENERATION_ATTEMPTS, MODEL_ID, MODEL_REVISION

_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
BEARER_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._~+/\-]+=*$")


@dataclass(frozen=True, slots=True)
class Credential:
    user_id: str
    display_name: str
    token: str = field(repr=False)

    def __post_init__(self) -> None:
        if not _USER_ID_PATTERN.fullmatch(self.user_id):
            raise ValueError("credential user_id must be a safe 1-64 character identifier")
        if (
            self.display_name != self.display_name.strip()
            or not self.display_name
            or len(self.display_name) > 80
            or not all(character.isprintable() for character in self.display_name)
        ):
            raise ValueError(
                "credential display_name must contain 1-80 trimmed printable characters"
            )
        if not 16 <= len(self.token) <= 512 or not BEARER_TOKEN_PATTERN.fullmatch(self.token):
            raise ValueError("worker bearer credentials must be 16-512 ASCII bearer-token chars")


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    data_root: Path
    model_cache_dir: Path
    credentials: tuple[Credential, ...]
    inference_backend: str = "flux"
    allow_fake_inference: bool = False
    fsync_writes: bool = True
    retry_delay_seconds: float = 0.25
    max_generation_attempts: int = MAX_GENERATION_ATTEMPTS
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    runtime_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.credentials:
            raise ValueError("at least one worker bearer credential is required")
        if len({item.user_id for item in self.credentials}) != len(self.credentials):
            raise ValueError("worker credential user IDs must be unique")
        if len({item.token for item in self.credentials}) != len(self.credentials):
            raise ValueError("worker bearer credentials must be unique")
        if self.inference_backend not in {"flux", "fake"}:
            raise ValueError("inference backend must be 'flux' or 'fake'")
        if self.inference_backend == "fake" and not self.allow_fake_inference:
            raise ValueError("fake inference requires IMAGEFORGE_ALLOW_FAKE_INFERENCE=1")
        if self.max_generation_attempts != MAX_GENERATION_ATTEMPTS:
            raise ValueError("ImageForge requires one attempt plus exactly two automatic retries")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry delay cannot be negative")
        if self.model_id != MODEL_ID or self.model_revision != MODEL_REVISION:
            raise ValueError("the production model ID and revision are pinned")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> WorkerSettings:
        env = os.environ if environ is None else environ
        credentials = _parse_credentials(env.get("IMAGEFORGE_AUTH_TOKENS_JSON", ""))
        backend = env.get("IMAGEFORGE_INFERENCE_BACKEND", "flux").strip().lower()
        metadata_keys = (
            "RUNPOD_POD_ID",
            "RUNPOD_DC_ID",
            "RUNPOD_GPU_COUNT",
            "RUNPOD_VOLUME_ID",
            "IMAGEFORGE_DATA_ROOT",
            "IMAGEFORGE_IMAGE_DIGEST",
            "CUDA_VERSION",
            "PYTORCH_VERSION",
        )
        runtime_metadata = {key: env[key] for key in metadata_keys if env.get(key)}
        return cls(
            data_root=Path(env.get("IMAGEFORGE_DATA_ROOT", "/workspace/imageforge")),
            model_cache_dir=Path(
                env.get("IMAGEFORGE_MODEL_CACHE_DIR", "/workspace/models/huggingface")
            ),
            credentials=credentials,
            inference_backend=backend,
            allow_fake_inference=_parse_bool(env.get("IMAGEFORGE_ALLOW_FAKE_INFERENCE", "0")),
            fsync_writes=_parse_bool(env.get("IMAGEFORGE_FSYNC_WRITES", "1")),
            retry_delay_seconds=float(env.get("IMAGEFORGE_RETRY_DELAY_SECONDS", "0.25")),
            runtime_metadata=runtime_metadata,
        )


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean setting: {raw!r}")


def _parse_credentials(raw: str) -> tuple[Credential, ...]:
    if not raw:
        raise ValueError("IMAGEFORGE_AUTH_TOKENS_JSON must be injected as a runtime secret")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("IMAGEFORGE_AUTH_TOKENS_JSON is not valid JSON") from exc

    records: list[dict[str, Any]] = []
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("credential JSON list entries must be objects")
        records = payload
    elif isinstance(payload, dict):
        for token, identity in payload.items():
            if isinstance(identity, str):
                records.append(
                    {"token": token, "user_id": identity.lower(), "display_name": identity}
                )
            elif isinstance(identity, dict):
                records.append({"token": token, **identity})
            else:
                raise ValueError("credential JSON mapping values must be strings or objects")
    else:
        raise ValueError("credential JSON must be a list or object")

    try:
        return tuple(
            Credential(
                user_id=str(record["user_id"]),
                display_name=str(record["display_name"]),
                token=str(record["token"]),
            )
            for record in records
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("each credential requires token, user_id, and display_name") from exc
