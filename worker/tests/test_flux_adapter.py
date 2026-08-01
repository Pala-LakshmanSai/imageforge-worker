from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from imageforge_worker.constants import (
    MIN_CUDA_VERSION,
    MIN_GPU_MEMORY_BYTES,
    MIN_GPU_MEMORY_MIB,
    MODEL_ALLOW_PATTERNS,
    MODEL_ID,
    MODEL_REVISION,
    REQUIRED_MODEL_FILES,
)
from imageforge_worker.inference.flux import FluxInferenceAdapter
from imageforge_worker.prepare_model import main as prepare_model_main


class FakeCuda:
    def __init__(self, *, name: str, total_memory: int, count: int = 1) -> None:
        self.name = name
        self.total_memory = total_memory
        self.count = count

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return self.count

    def get_device_name(self, _: int) -> str:
        return self.name

    def get_device_properties(self, _: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=self.total_memory)

    def memory_allocated(self, _: int) -> int:
        return 1

    def memory_reserved(self, _: int) -> int:
        return 2

    def max_memory_allocated(self, _: int) -> int:
        return 3

    def max_memory_reserved(self, _: int) -> int:
        return 4


def test_flux_adapter_rejects_insufficient_vram_but_surfaces_actual_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_torch = SimpleNamespace(
        cuda=FakeCuda(name="NVIDIA T4", total_memory=15 * 1024**3),
        version=SimpleNamespace(cuda="13.0"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(Flux2KleinPipeline=object()))
    adapter = FluxInferenceAdapter(tmp_path)
    with pytest.raises(RuntimeError, match="less than 16380 MiB"):
        adapter._load_pipeline(str(tmp_path))
    snapshot = adapter.gpu_snapshot()
    assert snapshot["name"] == "NVIDIA T4"
    assert snapshot["total_memory_bytes"] == 15 * 1024**3
    assert snapshot["approved"] is False


@pytest.mark.parametrize(
    ("total_memory", "accepted"),
    [
        (MIN_GPU_MEMORY_BYTES - 1, False),
        (MIN_GPU_MEMORY_BYTES, True),
        (MIN_GPU_MEMORY_BYTES + 1, True),
    ],
    ids=["one-byte-below", "exact-floor", "one-byte-above"],
)
def test_flux_adapter_enforces_byte_exact_emergency_vram_floor_without_cpu_offload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    total_memory: int,
    accepted: bool,
) -> None:
    bfloat16 = object()
    fake_torch = SimpleNamespace(
        cuda=FakeCuda(name="NVIDIA RTX 2000 Ada Generation", total_memory=total_memory),
        bfloat16=bfloat16,
        version=SimpleNamespace(cuda="13.0"),
    )
    captured: dict = {}

    class Parameter:
        device = SimpleNamespace(type="cuda")
        dtype = bfloat16

    class Pipeline:
        transformer = SimpleNamespace(parameters=lambda: iter([Parameter()]))

        def set_progress_bar_config(self, *, disable: bool) -> None:
            captured["progress_disabled"] = disable

    class FakePipelineType:
        @classmethod
        def from_pretrained(cls, _path: str, **kwargs):
            captured.update(kwargs)
            return Pipeline()

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakePipelineType),
    )
    adapter = FluxInferenceAdapter(tmp_path)

    if accepted:
        adapter._load_pipeline(str(tmp_path / "snapshot"))
        assert captured["device_map"] == "cuda"
        assert captured["torch_dtype"] is bfloat16
        assert adapter.gpu_snapshot()["approved"] is True
    else:
        with pytest.raises(RuntimeError, match="less than 16380 MiB"):
            adapter._load_pipeline(str(tmp_path / "snapshot"))
        assert captured == {}
        assert adapter.gpu_snapshot()["approved"] is False


def test_flux_adapter_is_family_generic_and_loads_bf16_directly_on_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bfloat16 = object()
    cuda = FakeCuda(name="NVIDIA RTX A5000", total_memory=24 * 1024**3)
    fake_torch = SimpleNamespace(
        cuda=cuda,
        bfloat16=bfloat16,
        version=SimpleNamespace(cuda="13.0"),
    )
    captured: dict = {}

    class Parameter:
        device = SimpleNamespace(type="cuda")
        dtype = bfloat16

    class Pipeline:
        transformer = SimpleNamespace(parameters=lambda: iter([Parameter()]))

        def set_progress_bar_config(self, *, disable: bool) -> None:
            captured["progress_disabled"] = disable

    class FakePipelineType:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            captured["path"] = path
            captured.update(kwargs)
            return Pipeline()

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakePipelineType),
    )
    adapter = FluxInferenceAdapter(tmp_path)
    adapter._load_pipeline(str(tmp_path / "snapshot"))
    assert captured["device_map"] == "cuda"
    assert captured["torch_dtype"] is bfloat16
    assert captured["local_files_only"] is True
    assert captured["progress_disabled"] is True
    assert adapter.gpu_snapshot()["approved"] is True


def test_flux_adapter_rejects_pre_cuda_13_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_torch = SimpleNamespace(
        cuda=FakeCuda(name="NVIDIA RTX 4090", total_memory=24 * 1024**3),
        version=SimpleNamespace(cuda="12.8"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(Flux2KleinPipeline=object()))
    adapter = FluxInferenceAdapter(tmp_path)
    with pytest.raises(RuntimeError, match="at least 13.0"):
        adapter._load_pipeline(str(tmp_path))


def test_model_snapshot_resolution_is_pinned_and_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}
    snapshot = tmp_path / "snapshot"
    for relative_name in REQUIRED_MODEL_FILES:
        path = snapshot / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def snapshot_download(**kwargs) -> str:
        captured.update(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    adapter = FluxInferenceAdapter(tmp_path / "cache")
    assert adapter._resolve_local_snapshot() == str(tmp_path / "snapshot")
    assert captured == {
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "cache_dir": str(tmp_path / "cache"),
        "local_files_only": True,
        "allow_patterns": list(MODEL_ALLOW_PATTERNS),
    }
    assert MIN_GPU_MEMORY_MIB == 16_380
    assert MIN_GPU_MEMORY_BYTES == 16_380 * 1024**2
    assert 16 * 1024**3 - MIN_GPU_MEMORY_BYTES == 4 * 1024**2
    assert MIN_CUDA_VERSION == (13, 0)


def test_confirmed_model_preparation_narrowly_enables_hub_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    for relative_name in REQUIRED_MODEL_FILES:
        path = snapshot / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    captured: dict = {}

    def snapshot_download(**kwargs) -> str:
        assert os.environ["HF_HUB_OFFLINE"] == "0"
        captured.update(kwargs)
        return str(snapshot)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "imageforge-prepare-model",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--confirm-download",
        ],
    )

    prepare_model_main()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert captured == {
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "cache_dir": str(tmp_path / "cache"),
        "allow_patterns": list(MODEL_ALLOW_PATTERNS),
    }
