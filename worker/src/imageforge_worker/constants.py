from typing import Final

API_SCHEMA_VERSION: Final = 1
WORKER_VERSION: Final = "0.1.0"

MODEL_ID: Final = "black-forest-labs/FLUX.2-klein-4B"
MODEL_REVISION: Final = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
MODEL_PRECISION: Final = "bfloat16"

OUTPUT_WIDTH: Final = 1280
OUTPUT_HEIGHT: Final = 720
INFERENCE_STEPS: Final = 4
GUIDANCE_SCALE: Final = 1.0
JPEG_QUALITY: Final = 95
PREVIEW_WIDTH: Final = 320
PREVIEW_HEIGHT: Final = 180

MAX_PROMPTS: Final = 500
MAX_PROMPT_UTF8_BYTES: Final = 4096
MAX_GENERATION_ATTEMPTS: Final = 3

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
