import pytest

from imageforge_worker import constants
from imageforge_worker.model_profiles import (
    FLUX2_KLEIN,
    MAGE_FLOW_TURBO_BF16,
    MAGE_FLOW_TURBO_INT8,
    profile_for_backend,
    supported_backends,
)


def test_the_shipped_constants_describe_the_active_model():
    """The constants are the identity the worker reports over the wire.

    They must track whichever profile is actually serving, or the health
    endpoint claims one model while the GPU runs another.
    """

    assert MAGE_FLOW_TURBO_INT8.model_id == constants.MODEL_ID
    assert MAGE_FLOW_TURBO_INT8.revision == constants.MODEL_REVISION
    assert MAGE_FLOW_TURBO_INT8.precision == constants.MODEL_PRECISION
    assert MAGE_FLOW_TURBO_INT8.steps == constants.INFERENCE_STEPS
    assert MAGE_FLOW_TURBO_INT8.guidance == constants.GUIDANCE_SCALE
    assert MAGE_FLOW_TURBO_INT8.min_gpu_memory_mib == constants.MIN_GPU_MEMORY_MIB


def test_the_retired_flux_profile_keeps_its_own_identity():
    """FLUX stays described accurately so a rollback does not need archaeology."""

    assert FLUX2_KLEIN.model_id == "black-forest-labs/FLUX.2-klein-4B"
    assert FLUX2_KLEIN.revision == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    assert FLUX2_KLEIN.supports_references is True


def test_mage_flow_turbo_is_text_to_image_only():
    assert MAGE_FLOW_TURBO_INT8.supports_references is False
    assert MAGE_FLOW_TURBO_BF16.supports_references is False


def test_mage_flow_names_the_public_mirror_not_the_gated_repository():
    """microsoft/Mage-Flow-Turbo returns 401; only the Comfy-Org mirror is public."""

    for profile in (MAGE_FLOW_TURBO_INT8, MAGE_FLOW_TURBO_BF16):
        assert profile.model_id == "Comfy-Org/Mage-Flow"
        assert not profile.model_id.startswith("microsoft/")


def test_the_int8_and_bf16_profiles_differ_only_in_the_transformer():
    """The comparison is only meaningful if everything else is held constant."""

    assert MAGE_FLOW_TURBO_INT8.revision == MAGE_FLOW_TURBO_BF16.revision
    assert MAGE_FLOW_TURBO_INT8.steps == MAGE_FLOW_TURBO_BF16.steps
    assert MAGE_FLOW_TURBO_INT8.guidance == MAGE_FLOW_TURBO_BF16.guidance
    int8_files = set(MAGE_FLOW_TURBO_INT8.required_files)
    bf16_files = set(MAGE_FLOW_TURBO_BF16.required_files)
    assert int8_files - bf16_files == {"diffusion_models/mage_flow_turbo_int8_convrot.safetensors"}
    assert bf16_files - int8_files == {"diffusion_models/mage_flow_turbo_bf16.safetensors"}


def test_every_profile_pins_an_explicit_revision():
    """An unpinned revision would let a mirror update change production output."""

    for backend in supported_backends():
        revision = profile_for_backend(backend).revision
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)


def test_profile_lookup_rejects_an_unknown_backend():
    assert profile_for_backend("flux") is FLUX2_KLEIN
    assert profile_for_backend("mageflow") is MAGE_FLOW_TURBO_INT8
    with pytest.raises(ValueError, match="unknown inference backend"):
        profile_for_backend("sdxl")


def test_the_default_backend_matches_the_reported_model():
    """A missing env var must not silently load a different model than reported."""

    from imageforge_worker import constants
    from imageforge_worker.config import WorkerSettings

    default_backend = WorkerSettings.__dataclass_fields__["inference_backend"].default
    assert profile_for_backend(default_backend).model_id == constants.MODEL_ID
    assert profile_for_backend(default_backend).revision == constants.MODEL_REVISION
