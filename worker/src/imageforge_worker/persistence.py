from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from .domain import (
    LOCK_HOLDING_STATES,
    SUCCESS_STATES,
    BatchManifest,
    BatchState,
    ImageRecord,
    ImageState,
)

MINIMUM_RETENTION = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class CleanupResult:
    images_deleted: int = 0
    files_deleted: int = 0
    bytes_deleted: int = 0


class ManifestStore(Protocol):
    def initialize(self) -> None: ...

    def try_acquire_worker_presence(self) -> bool: ...

    def release_worker_presence(self) -> None: ...

    def try_acquire_maintenance_presence(self) -> bool: ...

    def release_maintenance_presence(self) -> None: ...

    @property
    def active_lease_held(self) -> bool: ...

    def try_acquire_active_lease(self) -> bool: ...

    def release_active_lease(self) -> None: ...

    def list_batch_ids(self) -> list[str]: ...

    def create(
        self, manifest: BatchManifest, reference_payloads: list[tuple[str, bytes]] | None = None
    ) -> None: ...

    def load(self, batch_id: str) -> BatchManifest: ...

    def save(self, manifest: BatchManifest) -> None: ...

    def write_artifacts(
        self, batch_id: str, index: int, jpeg: bytes, preview: bytes
    ) -> tuple[str, str]: ...

    def artifact_path(self, batch_id: str, relative_name: str) -> Path: ...

    def read_reference(self, batch_id: str, relative_name: str) -> bytes: ...

    def verify_record_artifacts(self, batch_id: str, record: ImageRecord) -> bool: ...

    def quarantine_artifacts(self, batch_id: str, index: int) -> None: ...

    def cleanup_acknowledged_artifacts(
        self, *, now: datetime, minimum_age: timedelta = MINIMUM_RETENTION
    ) -> CleanupResult: ...


class FileManifestStore:
    """Crash-safe JSON manifests and server-named immutable artifacts."""

    def __init__(self, root: Path, *, fsync_writes: bool = True) -> None:
        self.root = root
        self.batches_root = root / "batches"
        self.fsync_writes = fsync_writes
        self._worker_presence_descriptor: int | None = None
        self._maintenance_presence_descriptor: int | None = None
        self._active_lease_descriptor: int | None = None

    @property
    def active_lease_held(self) -> bool:
        return self._active_lease_descriptor is not None

    def initialize(self) -> None:
        self.batches_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        probe = self.root / ".write-probe"
        self._atomic_write(probe, b"imageforge-storage-probe\n")
        probe.unlink(missing_ok=True)
        self._fsync_directory(self.root)

    def try_acquire_worker_presence(self) -> bool:
        """Hold shared presence for the lifetime of an initialized worker process."""

        if self._worker_presence_descriptor is not None:
            return True
        if self._maintenance_presence_descriptor is not None:
            return False
        descriptor = self._try_acquire_lock(".worker-presence.lock", fcntl.LOCK_SH)
        if descriptor is None:
            return False
        self._worker_presence_descriptor = descriptor
        return True

    def release_worker_presence(self) -> None:
        descriptor = self._worker_presence_descriptor
        self._worker_presence_descriptor = None
        self._release_lock(descriptor)

    def try_acquire_maintenance_presence(self) -> bool:
        """Take exclusive presence only when no worker process is alive."""

        if self._maintenance_presence_descriptor is not None:
            return True
        if self._worker_presence_descriptor is not None:
            return False
        descriptor = self._try_acquire_lock(".worker-presence.lock", fcntl.LOCK_EX)
        if descriptor is None:
            return False
        self._maintenance_presence_descriptor = descriptor
        return True

    def release_maintenance_presence(self) -> None:
        descriptor = self._maintenance_presence_descriptor
        self._maintenance_presence_descriptor = None
        self._release_lock(descriptor)

    def try_acquire_active_lease(self) -> bool:
        if self._active_lease_descriptor is not None:
            return True
        descriptor = self._try_acquire_lock(".active-batch.lock", fcntl.LOCK_EX)
        if descriptor is None:
            return False
        self._active_lease_descriptor = descriptor
        return True

    def release_active_lease(self) -> None:
        descriptor = self._active_lease_descriptor
        self._active_lease_descriptor = None
        self._release_lock(descriptor)

    def _try_acquire_lock(self, name: str, operation: int) -> int | None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(self.root / name, flags, 0o600)
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(descriptor)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return None
            raise
        return descriptor

    @staticmethod
    def _release_lock(descriptor: int | None) -> None:
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def list_batch_ids(self) -> list[str]:
        if not self.batches_root.exists():
            return []
        result: list[str] = []
        for child in self.batches_root.iterdir():
            if not child.is_dir() or not (child / "manifest.json").is_file():
                continue
            try:
                result.append(str(uuid.UUID(child.name)))
            except ValueError:
                continue
        return sorted(result)

    def create(
        self, manifest: BatchManifest, reference_payloads: list[tuple[str, bytes]] | None = None
    ) -> None:
        self._require_active_lease()
        batch_dir = self._batch_dir(manifest.batch_id)
        batch_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        (batch_dir / "artifacts").mkdir(mode=0o700)
        (batch_dir / "previews").mkdir(mode=0o700)
        references_dir = batch_dir / "references"
        references_dir.mkdir(mode=0o700)
        (batch_dir / "quarantine").mkdir(mode=0o700)
        for relative_name, payload in reference_payloads or []:
            path = self.artifact_path(manifest.batch_id, relative_name)
            if not relative_name.startswith("references/"):
                raise ValueError("reference path must remain under the references directory")
            self._write_immutable(path, payload)
        self._fsync_directory(references_dir)
        self._fsync_directory(self.batches_root)
        self.save(manifest)

    def load(self, batch_id: str) -> BatchManifest:
        path = self._batch_dir(batch_id) / "manifest.json"
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            raise
        return BatchManifest.model_validate_json(payload)

    def save(self, manifest: BatchManifest) -> None:
        self._require_active_lease()
        manifest.recalculate_progress()
        payload = json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self._batch_dir(manifest.batch_id) / "manifest.json"
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
        self._atomic_write(path, payload)

    def write_artifacts(
        self, batch_id: str, index: int, jpeg: bytes, preview: bytes
    ) -> tuple[str, str]:
        self._require_active_lease()
        full_name = f"artifacts/{index:06d}.jpg"
        preview_name = f"previews/{index:06d}.webp"
        self._write_immutable(self.artifact_path(batch_id, full_name), jpeg)
        self._write_immutable(self.artifact_path(batch_id, preview_name), preview)
        return full_name, preview_name

    def artifact_path(self, batch_id: str, relative_name: str) -> Path:
        batch_dir = self._batch_dir(batch_id).resolve()
        candidate = (batch_dir / relative_name).resolve()
        if not candidate.is_relative_to(batch_dir):
            raise ValueError("artifact path escaped its batch directory")
        return candidate

    def read_reference(self, batch_id: str, relative_name: str) -> bytes:
        if not relative_name.startswith("references/"):
            raise ValueError("reference path must remain under the references directory")
        path = self.artifact_path(batch_id, relative_name)
        return path.read_bytes()

    def verify_record_artifacts(self, batch_id: str, record: ImageRecord) -> bool:
        if not all(
            (
                record.filename,
                record.preview_filename,
                record.sha256,
                record.preview_sha256,
                record.size_bytes,
                record.preview_size_bytes,
            )
        ):
            return False
        full = self.artifact_path(batch_id, record.filename)
        preview = self.artifact_path(batch_id, record.preview_filename)
        return self._matches(full, record.size_bytes, record.sha256) and self._matches(
            preview, record.preview_size_bytes, record.preview_sha256
        )

    def quarantine_artifacts(self, batch_id: str, index: int) -> None:
        self._require_active_lease()
        batch_dir = self._batch_dir(batch_id)
        quarantine_dir = batch_dir / "quarantine"
        quarantine_dir.mkdir(mode=0o700, exist_ok=True)
        nonce = uuid.uuid4().hex
        for source in (
            batch_dir / "artifacts" / f"{index:06d}.jpg",
            batch_dir / "previews" / f"{index:06d}.webp",
        ):
            if source.exists():
                target = quarantine_dir / f"{source.name}.{nonce}.corrupt-or-orphan"
                os.replace(source, target)
        self._fsync_directory(quarantine_dir)
        self._fsync_directory(batch_dir / "artifacts")
        self._fsync_directory(batch_dir / "previews")

    def cleanup_acknowledged_artifacts(
        self, *, now: datetime, minimum_age: timedelta = MINIMUM_RETENTION
    ) -> CleanupResult:
        """Explicitly remove verified artifacts acknowledged at least 24 hours ago.

        ImageForge never calls this automatically. A per-image cleanup intent is saved before
        either file is unlinked. The manifest, checksums, and receipt therefore remain durable
        at every crash point, and a later explicit cleanup safely resumes partial deletion.
        """

        self._require_active_lease()
        if now.tzinfo is None:
            raise ValueError("cleanup time must be timezone-aware")
        if minimum_age < MINIMUM_RETENTION:
            raise ValueError("acknowledged artifacts must be retained for at least 24 hours")

        images_deleted = 0
        files_deleted = 0
        bytes_deleted = 0
        cleanup_timestamp = (
            now.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        )
        for batch_id in self.list_batch_ids():
            manifest = self.load(batch_id)
            if manifest.state in LOCK_HOLDING_STATES:
                continue
            for image in manifest.images:
                if image.artifacts_deleted_at is not None:
                    continue

                if image.artifacts_cleanup_started_at is None:
                    receipt = image.receipt
                    if (
                        image.status != ImageState.DOWNLOADED
                        or receipt is None
                        or receipt.user_id != manifest.owner.user_id
                    ):
                        continue
                    acknowledged_at = _parse_timestamp(receipt.acknowledged_at)
                    if (
                        acknowledged_at is None
                        or now.astimezone(UTC) - acknowledged_at < minimum_age
                    ):
                        continue
                    if not self.verify_record_artifacts(batch_id, image):
                        continue
                    image.artifacts_cleanup_started_at = cleanup_timestamp
                    # This tombstone must be durable before the first destructive mutation.
                    self.save(manifest)

                assert image.filename is not None
                assert image.preview_filename is not None
                full = self.artifact_path(batch_id, image.filename)
                preview = self.artifact_path(batch_id, image.preview_filename)
                for path in (full, preview):
                    try:
                        size = path.stat().st_size
                    except FileNotFoundError:
                        continue
                    path.unlink()
                    self._fsync_directory(path.parent)
                    files_deleted += 1
                    bytes_deleted += size

                image.artifacts_deleted_at = cleanup_timestamp
                # Finalization is idempotent: after a failed save the durable intent makes a
                # restart preserve metadata, and the next explicit run saves completion.
                self.save(manifest)
                images_deleted += 1
            # Reference inputs are batch-scoped working data. Once a batch is
            # fully completed they are no longer needed for resume/retry, so
            # remove the raw files while retaining checksums and names in the
            # manifest for reproducibility. Interrupted/failed batches keep
            # references because they may still be resumed or retried.
            if (
                manifest.state == BatchState.COMPLETED
                and manifest.references
                and all(image.status in SUCCESS_STATES for image in manifest.images)
            ):
                for reference in manifest.references:
                    path = self.artifact_path(batch_id, reference.filename)
                    try:
                        size = path.stat().st_size
                    except FileNotFoundError:
                        continue
                    path.unlink()
                    self._fsync_directory(path.parent)
                    files_deleted += 1
                    bytes_deleted += size
        return CleanupResult(
            images_deleted=images_deleted,
            files_deleted=files_deleted,
            bytes_deleted=bytes_deleted,
        )

    def _batch_dir(self, batch_id: str) -> Path:
        normalized = str(uuid.UUID(str(batch_id)))
        return self.batches_root / normalized

    def _require_active_lease(self) -> None:
        if self._active_lease_descriptor is None:
            raise RuntimeError("manifest and artifact mutations require the active-volume lease")

    def _write_immutable(self, path: Path, payload: bytes) -> None:
        if path.exists():
            if self._matches(path, len(payload), hashlib.sha256(payload).hexdigest()):
                return
            raise FileExistsError(f"immutable artifact already exists: {path.name}")
        self._atomic_write(path, payload)

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                if self.fsync_writes:
                    os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            if self.fsync_writes:
                self._fsync_directory(path.parent)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _fsync_directory(self, path: Path) -> None:
        if not self.fsync_writes or not path.exists():
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
        finally:
            os.close(descriptor)

    @staticmethod
    def _matches(path: Path, expected_size: int, expected_sha256: str) -> bool:
        try:
            if path.stat().st_size != expected_size:
                return False
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest() == expected_sha256
        except FileNotFoundError:
            return False


def clone_store(source: Path, target: Path) -> None:
    """Test helper for simulating a persistent volume attached to a new Pod."""
    shutil.copytree(source, target, dirs_exist_ok=True)


def _parse_timestamp(raw: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)
