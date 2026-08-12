"""Phase 0 loader spike for Mage-Flow Turbo INT8 convrot.

Answers one question with evidence instead of assumption: can the Comfy-Org
`mage_flow_turbo_int8_convrot.safetensors` checkpoint be loaded and driven from
Diffusers, the way the ImageForge worker already drives FLUX?

This script never runs in production. It runs by hand on a throwaway spike Pod
attached to the staging network volume, and it prints a report that is pasted
into docs/MAGEFLOW_STAGING.md.

Usage on the spike Pod:

    python mageflow_spike.py --cache-dir /workspace/models/huggingface \
        --output-dir /workspace/spike

Add --bf16 to measure the Path B fallback (the 8.23 GB BF16 transformer) in the
same run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INT8_TRANSFORMER_FILENAME = "mage_flow_turbo_int8_convrot.safetensors"
BF16_TRANSFORMER_FILENAME = "mage_flow_turbo_bf16.safetensors"

# The pipeline class name is not documented consistently across the model card,
# the paper repository, and the ComfyUI packaging, so the spike probes every
# plausible home rather than hard-coding one and reporting a misleading failure.
PIPELINE_CANDIDATES = (
    ("diffusers", "MageFlowPipeline"),
    ("mage_flow", "MageFlowPipeline"),
    ("diffusers", "MageFlowImagePipeline"),
)
TRANSFORMER_CANDIDATES = (
    ("diffusers", "MageFlowTransformer2DModel"),
    ("diffusers", "MageFlowTransformer3DModel"),
    ("mage_flow", "MageFlowTransformer2DModel"),
)

# One prompt per weakness that motivated the migration, plus a control.
SPIKE_PROMPTS = (
    (
        "text",
        "A tall glass storefront at dawn with the words OPEN LATE etched in gold "
        "leaf across the window, a chalkboard below reading Espresso 3.50",
    ),
    (
        "face",
        "Close-up portrait of a woman in her fifties laughing, weathered skin, "
        "grey curls, both hands cupped around a ceramic mug, soft window light",
    ),
    (
        "spatial",
        "A red bicycle leaning against the left side of a blue door, a tabby cat "
        "sitting on the doormat to the right of the bicycle, potted fern above",
    ),
    (
        "control",
        "A neutral studio lighting calibration chart",
    ),
)


@dataclass
class SpikeReport:
    resolved: dict[str, Any] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        lines = ["", "=" * 72, "MAGE-FLOW SPIKE REPORT", "=" * 72]
        for key, value in self.resolved.items():
            lines.append(f"{key}: {value}")
        for attempt in self.attempts:
            lines.append("-" * 72)
            for key, value in attempt.items():
                lines.append(f"{key}: {value}")
        lines.append("=" * 72)
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="also load the BF16 transformer so Path B has measured numbers",
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    report = SpikeReport()
    import torch

    report.resolved["torch"] = torch.__version__
    report.resolved["torch_cuda"] = torch.version.cuda
    report.resolved["gpu"] = torch.cuda.get_device_name(0)
    report.resolved["gpu_total_gib"] = round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    )
    try:
        import diffusers

        report.resolved["diffusers"] = diffusers.__version__
    except ImportError as error:  # pragma: no cover - spike-only diagnostics
        report.resolved["diffusers"] = f"IMPORT FAILED: {error}"

    pipeline_class, pipeline_origin = _first_available(PIPELINE_CANDIDATES)
    transformer_class, transformer_origin = _first_available(TRANSFORMER_CANDIDATES)
    report.resolved["pipeline_class"] = pipeline_origin
    report.resolved["transformer_class"] = transformer_origin

    scaffolding = _find_scaffolding(arguments.cache_dir)
    report.resolved["scaffolding"] = scaffolding or "NOT FOUND"

    if pipeline_class is None or scaffolding is None:
        report.attempts.append(
            {
                "attempt": "int8",
                "result": "BLOCKED",
                "detail": "no importable Mage-Flow pipeline class or no Diffusers scaffolding; "
                "this is Path B evidence",
            }
        )
        print(report.render())
        _write_report(arguments.output_dir, report)
        raise SystemExit(1)

    wanted = [("int8", INT8_TRANSFORMER_FILENAME)]
    if arguments.bf16:
        wanted.append(("bf16", BF16_TRANSFORMER_FILENAME))

    for label, filename in wanted:
        transformer_path = _find_file(arguments.cache_dir, filename)
        attempt: dict[str, Any] = {"attempt": label, "transformer_file": transformer_path}
        if transformer_path is None:
            attempt["result"] = "BLOCKED"
            attempt["detail"] = f"{filename} is not on the volume; run prepare_mageflow_volume.py"
            report.attempts.append(attempt)
            continue
        try:
            _run_attempt(
                attempt=attempt,
                torch=torch,
                pipeline_class=pipeline_class,
                transformer_class=transformer_class,
                scaffolding=scaffolding,
                transformer_path=transformer_path,
                arguments=arguments,
                label=label,
            )
        except Exception as error:  # noqa: BLE001 - the exception type is the finding
            attempt["result"] = "FAILED"
            attempt["error_type"] = type(error).__name__
            attempt["error"] = str(error)[:2000]
        report.attempts.append(attempt)

    print(report.render())
    _write_report(arguments.output_dir, report)


def _run_attempt(
    *,
    attempt: dict[str, Any],
    torch: Any,
    pipeline_class: Any,
    transformer_class: Any,
    scaffolding: str,
    transformer_path: str,
    arguments: argparse.Namespace,
    label: str,
) -> None:
    load_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)

    if transformer_class is not None:
        transformer = transformer_class.from_single_file(
            transformer_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipeline = pipeline_class.from_pretrained(
            scaffolding,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    else:
        # No separate transformer class: the pipeline itself has to accept the
        # single file. If it does not, the raised error is the finding.
        pipeline = pipeline_class.from_single_file(
            transformer_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    pipeline.to("cuda")
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    attempt["load_seconds"] = round(time.perf_counter() - load_started, 1)
    attempt["load_peak_gib"] = round(torch.cuda.max_memory_allocated(0) / 1024**3, 2)

    per_image_seconds: list[float] = []
    torch.cuda.reset_peak_memory_stats(0)
    for category, prompt in SPIKE_PROMPTS:
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            image = pipeline(
                prompt=prompt,
                height=arguments.height,
                width=arguments.width,
                num_inference_steps=arguments.steps,
                guidance_scale=arguments.guidance,
                generator=torch.Generator(device="cuda").manual_seed(arguments.seed),
            ).images[0]
        torch.cuda.synchronize()
        per_image_seconds.append(time.perf_counter() - started)
        destination = arguments.output_dir / f"spike-{label}-{category}.jpg"
        image.convert("RGB").save(destination, quality=95, subsampling=0)
        attempt[f"image_{category}"] = str(destination)
        attempt[f"size_{category}"] = f"{image.size[0]}x{image.size[1]}"

    ordered = sorted(per_image_seconds)
    attempt["result"] = "OK"
    attempt["median_seconds_per_image"] = round(ordered[len(ordered) // 2], 2)
    attempt["generate_peak_gib"] = round(torch.cuda.max_memory_allocated(0) / 1024**3, 2)

    del pipeline
    torch.cuda.empty_cache()


def _first_available(candidates: tuple[tuple[str, str], ...]) -> tuple[Any, str]:
    errors: list[str] = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            errors.append(f"{module_name}: {error}")
            continue
        found = getattr(module, class_name, None)
        if found is not None:
            return found, f"{module_name}.{class_name}"
        errors.append(f"{module_name}.{class_name}: not exported")
    return None, "NONE (" + "; ".join(errors) + ")"


def _find_file(cache_dir: Path, filename: str) -> str | None:
    matches = sorted(cache_dir.rglob(filename))
    return str(matches[0]) if matches else None


def _find_scaffolding(cache_dir: Path) -> str | None:
    matches = sorted(cache_dir.rglob("model_index.json"))
    for match in matches:
        if "Mage-Flow" in str(match):
            return str(match.parent)
    return str(matches[0].parent) if matches else None


def _write_report(output_dir: Path, report: SpikeReport) -> None:
    destination = output_dir / "spike-report.json"
    destination.write_text(
        json.dumps({"resolved": report.resolved, "attempts": report.attempts}, indent=2)
    )
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
