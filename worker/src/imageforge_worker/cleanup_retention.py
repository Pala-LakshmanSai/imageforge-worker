from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .domain import LOCK_HOLDING_STATES
from .persistence import MINIMUM_RETENTION, CleanupResult, FileManifestStore


def cleanup_retained_artifacts(
    data_root: Path,
    *,
    now: datetime | None = None,
    minimum_age: timedelta = MINIMUM_RETENTION,
) -> CleanupResult:
    """Run explicit conservative cleanup; ImageForge never schedules this automatically."""

    store = FileManifestStore(data_root)
    if not store.try_acquire_maintenance_presence():
        raise RuntimeError("cleanup is disabled while any worker process is alive")
    try:
        if not store.try_acquire_active_lease():
            raise RuntimeError("another process owns the active-batch lease")
        try:
            # Exclusive presence is held before even the storage write probe.
            store.initialize()
            for batch_id in store.list_batch_ids():
                if store.load(batch_id).state in LOCK_HOLDING_STATES:
                    raise RuntimeError("cleanup is disabled while an active batch is retained")
            return store.cleanup_acknowledged_artifacts(
                now=now or datetime.now(UTC),
                minimum_age=minimum_age,
            )
        finally:
            store.release_active_lease()
    finally:
        store.release_maintenance_presence()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly clean verified, acknowledged ImageForge artifacts after 24 hours"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--minimum-age-hours", type=int, default=24)
    parser.add_argument(
        "--confirm-cleanup",
        action="store_true",
        help="required acknowledgement that eligible retained artifacts will be deleted",
    )
    arguments = parser.parse_args()
    if not arguments.confirm_cleanup:
        parser.error("--confirm-cleanup is required; cleanup is never automatic")
    if arguments.minimum_age_hours < 24:
        parser.error("--minimum-age-hours cannot be less than 24")

    result = cleanup_retained_artifacts(
        arguments.data_root,
        minimum_age=timedelta(hours=arguments.minimum_age_hours),
    )
    print(
        "Cleaned acknowledged ImageForge artifacts: "
        f"images={result.images_deleted} files={result.files_deleted} "
        f"bytes={result.bytes_deleted}"
    )


if __name__ == "__main__":
    main()
