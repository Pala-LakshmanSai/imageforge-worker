from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from imageforge_worker.domain import (
    BatchManifest,
    BatchOwner,
    BatchProgress,
    BatchState,
    ImageRecord,
    ImageState,
    utc_now,
)


def _manifest() -> BatchManifest:
    now = utc_now()
    return BatchManifest(
        batch_id="00000000-0000-4000-8000-000000000099",
        owner=BatchOwner(user_id="lakshman", display_name="Lakshman"),
        state=BatchState.RUNNING,
        created_at=now,
        updated_at=now,
        images=[ImageRecord(index=1, prompt="test prompt", seed=0)],
        progress=BatchProgress(total=1),
    )


def test_progress_counters_must_reconcile() -> None:
    with pytest.raises(ValidationError, match="inconsistent"):
        BatchProgress(total=2, completed=1, failed=1, processed=1)


def test_manifest_rejects_stale_progress_and_duplicate_generating_images() -> None:
    payload = _manifest().model_dump(mode="json")
    payload["progress"]["completed"] = 1
    payload["progress"]["processed"] = 1
    with pytest.raises(ValidationError, match="progress does not match"):
        BatchManifest.model_validate_json(json.dumps(payload))

    duplicate = _manifest().model_dump(mode="json")
    duplicate["images"][0]["status"] = ImageState.GENERATING.value
    duplicate["images"].append({**duplicate["images"][0], "index": 2})
    duplicate["progress"] = {
        "total": 2,
        "completed": 0,
        "downloaded": 0,
        "failed": 0,
        "cancelled": 0,
        "processed": 0,
        "current_index": None,
    }
    with pytest.raises(ValidationError, match="progress does not match"):
        BatchManifest.model_validate_json(json.dumps(duplicate))
