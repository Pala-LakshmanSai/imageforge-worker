from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    API_SCHEMA_VERSION,
    ASPECT_RATIO_DIMENSIONS,
    GUIDANCE_SCALE,
    INFERENCE_STEPS,
    JPEG_QUALITY,
    MAX_REFERENCE_BYTES,
    MAX_REFERENCE_NAME_BYTES,
    MAX_REFERENCE_TOTAL_BYTES,
    MAX_REFERENCES,
    MAX_SEED,
    MODEL_ID,
    MODEL_PRECISION,
    MODEL_REVISION,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class HealthPhase(StrEnum):
    PROCESS = "process"
    STORAGE = "storage"
    WEIGHTS = "weights"
    GPU_LOAD = "gpu_load"
    WARMUP = "warmup"
    READY = "ready"
    ERROR = "error"


class BatchState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ImageState(StrEnum):
    PENDING = "pending"
    GENERATING = "generating"
    RETRYING = "retrying"
    READY = "ready"
    DOWNLOADED = "downloaded"
    FAILED = "failed"
    CANCELLED = "cancelled"


LOCK_HOLDING_STATES = {BatchState.RUNNING, BatchState.PAUSED, BatchState.INTERRUPTED}
SUCCESS_STATES = {ImageState.READY, ImageState.DOWNLOADED}
NONTERMINAL_IMAGE_STATES = {
    ImageState.PENDING,
    ImageState.GENERATING,
    ImageState.RETRYING,
}


ReferenceMime = Literal["image/jpeg", "image/png", "image/webp"]
AspectRatio = Literal["16:9", "1:1", "9:16", "4:3", "3:4"]


class ReferenceInput(StrictModel):
    name: str = Field(min_length=1)
    mime_type: ReferenceMime
    data_hex: str = Field(min_length=2)

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        if (
            len(name.encode("utf-8")) > MAX_REFERENCE_NAME_BYTES
            or not name.strip()
            or any(separator in name for separator in ("/", "\\", "\x00"))
        ):
            raise ValueError("reference name is invalid")
        return name

    @field_validator("data_hex")
    @classmethod
    def validate_data_hex(cls, data_hex: str) -> str:
        if (
            len(data_hex) % 2
            or len(data_hex) > MAX_REFERENCE_BYTES * 2
            or re.fullmatch(r"[0-9a-f]+", data_hex) is None
        ):
            raise ValueError("reference data must be lowercase hexadecimal within the byte limit")
        return data_hex

    @property
    def size_bytes(self) -> int:
        return len(self.data_hex) // 2


class StoredReference(StrictModel):
    name: str = Field(min_length=1)
    mime_type: ReferenceMime
    size_bytes: int = Field(ge=1, le=MAX_REFERENCE_BYTES)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filename: str = Field(pattern=r"^references/[0-9]{6}\.(?:jpg|png|webp)$")

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        if (
            len(name.encode("utf-8")) > MAX_REFERENCE_NAME_BYTES
            or not name.strip()
            or any(separator in name for separator in ("/", "\\", "\x00"))
        ):
            raise ValueError("reference name is invalid")
        return name


class CreateBatchRequest(StrictModel):
    prompts: list[str] = Field(min_length=1)
    base_seed: int = Field(default=0, ge=0, le=MAX_SEED)
    aspect_ratio: AspectRatio = "16:9"
    references: list[ReferenceInput] = Field(default_factory=list, max_length=MAX_REFERENCES)

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, prompts: list[str]) -> list[str]:
        for prompt in prompts:
            if not prompt.strip():
                raise ValueError("prompts cannot be blank")
        return prompts

    @model_validator(mode="after")
    def validate_seed_range(self) -> CreateBatchRequest:
        if self.base_seed + len(self.prompts) - 1 > MAX_SEED:
            raise ValueError("base_seed plus prompt count exceeds the supported seed range")
        if sum(reference.size_bytes for reference in self.references) > MAX_REFERENCE_TOTAL_BYTES:
            raise ValueError("reference images exceed the total byte limit")
        return self


class GenerationSettings(StrictModel):
    model: Literal[MODEL_ID] = MODEL_ID
    revision: Literal[MODEL_REVISION] = MODEL_REVISION
    precision: Literal[MODEL_PRECISION] = MODEL_PRECISION
    width: int = Field(default=OUTPUT_WIDTH, ge=64, le=2048, multiple_of=8)
    height: int = Field(default=OUTPUT_HEIGHT, ge=64, le=2048, multiple_of=8)
    steps: Literal[INFERENCE_STEPS] = INFERENCE_STEPS
    guidance: Literal[GUIDANCE_SCALE] = GUIDANCE_SCALE
    jpeg_quality: Literal[JPEG_QUALITY] = JPEG_QUALITY
    preview_width: int = Field(default=PREVIEW_WIDTH, ge=32, le=PREVIEW_WIDTH)
    preview_height: int = Field(default=PREVIEW_HEIGHT, ge=32, le=PREVIEW_HEIGHT)

    @classmethod
    def for_aspect_ratio(cls, aspect_ratio: AspectRatio) -> GenerationSettings:
        width, height = ASPECT_RATIO_DIMENSIONS[aspect_ratio]
        # Keep previews proportional while bounding their long edge for cheap
        # polling and download previews.
        preview_scale = min(320 / width, 180 / height)
        preview_width = max(64, min(PREVIEW_WIDTH, int(round(width * preview_scale))))
        preview_height = max(64, min(PREVIEW_HEIGHT, int(round(height * preview_scale))))
        return cls(
            width=width,
            height=height,
            preview_width=preview_width,
            preview_height=preview_height,
        )


class BatchOwner(StrictModel):
    user_id: str
    display_name: str


class SafeImageError(StrictModel):
    code: str
    message: str


class AttemptRecord(StrictModel):
    attempt: int = Field(ge=1)
    started_at: str
    finished_at: str
    status: Literal["succeeded", "failed"]
    inference_ms: float | None = Field(default=None, ge=0)
    jpeg_encode_ms: float | None = Field(default=None, ge=0)
    preview_encode_ms: float | None = Field(default=None, ge=0)
    error_code: str | None = None


class StoredReceipt(StrictModel):
    user_id: str
    sha256: str
    size_bytes: int = Field(ge=1)
    acknowledged_at: str


class ImageRecord(StrictModel):
    index: int = Field(ge=1)
    prompt: str
    seed: int = Field(ge=0, le=MAX_SEED)
    status: ImageState = ImageState.PENDING
    attempts: int = Field(default=0, ge=0)
    attempts_in_cycle: int = Field(default=0, ge=0)
    retry_rounds: int = Field(default=0, ge=0)
    attempt_history: list[AttemptRecord] = Field(default_factory=list)
    filename: str | None = None
    preview_filename: str | None = None
    sha256: str | None = None
    preview_sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=1)
    preview_size_bytes: int | None = Field(default=None, ge=1)
    started_at: str | None = None
    finished_at: str | None = None
    generation_ms: float | None = Field(default=None, ge=0)
    error: SafeImageError | None = None
    receipt: StoredReceipt | None = None
    artifacts_cleanup_started_at: str | None = None
    artifacts_deleted_at: str | None = None

    @property
    def artifacts_expired(self) -> bool:
        return (
            self.artifacts_cleanup_started_at is not None or self.artifacts_deleted_at is not None
        )

    def clear_artifact(self) -> None:
        self.filename = None
        self.preview_filename = None
        self.sha256 = None
        self.preview_sha256 = None
        self.size_bytes = None
        self.preview_size_bytes = None
        self.finished_at = None
        self.generation_ms = None
        self.receipt = None
        self.artifacts_cleanup_started_at = None
        self.artifacts_deleted_at = None


class BatchProgress(StrictModel):
    total: int = Field(ge=1)
    completed: int = Field(default=0, ge=0)
    downloaded: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    cancelled: int = Field(default=0, ge=0)
    processed: int = Field(default=0, ge=0)
    current_index: int | None = Field(default=None, ge=1)


class BatchManifest(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION] = API_SCHEMA_VERSION
    batch_id: str
    owner: BatchOwner
    state: BatchState
    created_at: str
    updated_at: str
    completed_at: str | None = None
    interrupted_at: str | None = None
    pause_requested: bool = False
    cancel_requested: bool = False
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    references: list[StoredReference] = Field(default_factory=list, max_length=MAX_REFERENCES)
    images: list[ImageRecord]
    progress: BatchProgress

    @model_validator(mode="after")
    def validate_order(self) -> BatchManifest:
        expected = list(range(1, len(self.images) + 1))
        if [image.index for image in self.images] != expected:
            raise ValueError("manifest image indices must be contiguous and ordered")
        if self.progress.total != len(self.images):
            raise ValueError("manifest total must match image count")
        return self

    def recalculate_progress(self) -> None:
        completed = sum(image.status in SUCCESS_STATES for image in self.images)
        downloaded = sum(image.status == ImageState.DOWNLOADED for image in self.images)
        failed = sum(image.status == ImageState.FAILED for image in self.images)
        cancelled = sum(image.status == ImageState.CANCELLED for image in self.images)
        current = next(
            (image.index for image in self.images if image.status == ImageState.GENERATING),
            None,
        )
        self.progress = BatchProgress(
            total=len(self.images),
            completed=completed,
            downloaded=downloaded,
            failed=failed,
            cancelled=cancelled,
            processed=completed + failed + cancelled,
            current_index=current,
        )
        self.updated_at = utc_now()


class BatchSummary(StrictModel):
    batch_id: str
    owner: BatchOwner
    state: BatchState
    progress: BatchProgress
    pause_requested: bool
    cancel_requested: bool


class StatusPermissions(StrictModel):
    can_create: bool
    can_manage_active: bool
    is_owner: bool


class StatusResponse(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION] = API_SCHEMA_VERSION
    ready: bool
    active_batch: BatchSummary | None
    permissions: StatusPermissions


class ReceiptItem(StrictModel):
    index: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)


class ReceiptRequest(StrictModel):
    receipts: list[ReceiptItem] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_indices(self) -> ReceiptRequest:
        indices = [receipt.index for receipt in self.receipts]
        if len(indices) != len(set(indices)):
            raise ValueError("receipt indices must be unique")
        return self


class ReceiptResponse(StrictModel):
    schema_version: Literal[API_SCHEMA_VERSION] = API_SCHEMA_VERSION
    batch_id: str
    accepted: list[int]
    progress: BatchProgress
