from typing import Final

API_SCHEMA_VERSION: Final = 1
WORKER_VERSION: Final = "0.1.3"

MODEL_ID: Final = "black-forest-labs/FLUX.2-klein-4B"
MODEL_REVISION: Final = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
MODEL_PRECISION: Final = "bfloat16"

OUTPUT_WIDTH: Final = 1280
OUTPUT_HEIGHT: Final = 720
ASPECT_RATIO_DIMENSIONS: Final = {
    "16:9": (1280, 720),
    "1:1": (1024, 1024),
    "9:16": (720, 1280),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
}
INFERENCE_STEPS: Final = 4
GUIDANCE_SCALE: Final = 1.0
JPEG_QUALITY: Final = 95
PREVIEW_WIDTH: Final = 320
PREVIEW_HEIGHT: Final = 180

# Seeds are passed to the inference backend as signed 64-bit integers. Prompt
# lists themselves are intentionally unbounded; practical request, storage,
# and GPU limits remain outside the domain model.
MAX_SEED: Final = 2**63 - 1
MAX_GENERATION_ATTEMPTS: Final = 3

# Batch-level reference images are deliberately bounded for predictable request
# and GPU memory behavior. These are practical transport limits, not prompt
# count limits.
MAX_REFERENCES: Final = 8
MAX_REFERENCE_BYTES: Final = 8 * 1024 * 1024
MAX_REFERENCE_TOTAL_BYTES: Final = 32 * 1024 * 1024
MAX_REFERENCE_NAME_BYTES: Final = 255
MAX_REFERENCE_PIXELS: Final = 64_000_000

# RunPod's approved emergency RTX 2000 Ada reports 16,380 MiB rather than the
# nominal 16 GiB. Keep the allowance byte-exact and limited to that 4 MiB
# reporting margin so materially smaller devices still fail readiness.
MIN_GPU_MEMORY_MIB: Final = 16_380
MIN_GPU_MEMORY_BYTES: Final = MIN_GPU_MEMORY_MIB * 1024**2
MIN_CUDA_VERSION: Final = (13, 0)

# The root single-file checkpoint duplicates transformer/ and is deliberately excluded.
MODEL_ALLOW_PATTERNS: Final = (
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
)
REQUIRED_MODEL_FILES: Final = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/tokenizer.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
