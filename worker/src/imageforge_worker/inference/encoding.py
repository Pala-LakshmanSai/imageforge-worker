from __future__ import annotations

import io
import time

from PIL import Image

from ..domain import GenerationSettings
from .base import InferenceResult


def encode_result(
    image: Image.Image, settings: GenerationSettings, inference_ms: float
) -> InferenceResult:
    """Turn a generated frame into the JPEG and WEBP preview the contract promises.

    Every backend produces the same bytes for the same pixels, so the encoding
    lives here rather than in each adapter.
    """

    image = image.convert("RGB")
    if image.size != (settings.width, settings.height):
        raise RuntimeError("the backend returned an unexpected image size")

    jpeg_started = time.perf_counter()
    jpeg_buffer = io.BytesIO()
    image.save(
        jpeg_buffer,
        format="JPEG",
        quality=settings.jpeg_quality,
        optimize=False,
        progressive=False,
        subsampling=0,
    )
    jpeg_encode_ms = (time.perf_counter() - jpeg_started) * 1000

    preview_started = time.perf_counter()
    preview = image.resize(
        (settings.preview_width, settings.preview_height), Image.Resampling.LANCZOS
    )
    preview_buffer = io.BytesIO()
    preview.save(preview_buffer, format="WEBP", quality=85, method=4, exact=True)
    preview_encode_ms = (time.perf_counter() - preview_started) * 1000

    return InferenceResult(
        jpeg=jpeg_buffer.getvalue(),
        preview=preview_buffer.getvalue(),
        inference_ms=inference_ms,
        jpeg_encode_ms=jpeg_encode_ms,
        preview_encode_ms=preview_encode_ms,
    )
