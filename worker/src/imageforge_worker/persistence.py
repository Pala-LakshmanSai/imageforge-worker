from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, ValidationError, field_validator, model_validator

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from .coordination import FINALIZATION_TTL_SECONDS, STOP_RESPONSE_TTL_SECONDS
from .domain import (
    LOCK_HOLDING_STATES,
    SUCCESS_STATES,
    BatchManifest,
    BatchState,
    ImageRecord,
    ImageState,
    StrictModel,
    require_canonical_uuid4,
)
from .gpu_switch_models import require_gpu_identity

MINIMUM_RETENTION = timedelta(hours=24)
_WINDOWS_PRESENCE_SLOTS = 256
_GPU_STOP_GUARD_FILENAME = ".gpu-stop-finalization.json"
_MAX_GPU_STOP_GUARD_BYTES = 2048
_POD_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,56}[A-Za-z0-9])?$")
_MANIFEST_ENVELOPE_SCHEMA_VERSION = 2
_FINGERPRINT_PATTERN = r"^[0-9a-f]{64}$"


class SubmissionStoreCorruptError(RuntimeError):
    """A v2 envelope cannot safely participate in an idempotent admission.

    The controller deliberately maps this to one opaque public 503 response.
    Keeping the storage exception distinct avoids accidentally treating a bad
    envelope as a missing submission and creating another remote batch.
    """


class SubmissionRecord(StrictModel):
    client_submission_id: str
    owner_user_id: str = Field(min_length=1)
    request_fingerprint: str = Field(pattern=_FINGERPRINT_PATTERN)

    @field_validator("client_submission_id")
    @classmethod
    def validate_client_submission_id(cls, value: str) -> str:
        return require_canonical_uuid4(value)


class ManifestEnvelopeV2(StrictModel):
    """Private durable v2 wrapper around the public API manifest."""

    schema_version: Literal[_MANIFEST_ENVELOPE_SCHEMA_VERSION] = _MANIFEST_ENVELOPE_SCHEMA_VERSION
    manifest: BatchManifest
    submission: SubmissionRecord

    @model_validator(mode="after")
    def validate_submission_binding(self) -> ManifestEnvelopeV2:
        if self.manifest.client_submission_id != self.submission.client_submission_id:
            raise ValueError("manifest submission ID does not match its private record")
        if self.manifest.owner.user_id != self.submission.owner_user_id:
            raise ValueError("manifest owner does not match its private record")
        return self


@dataclass(frozen=True, slots=True)
class SubmissionMatch:
    manifest: BatchManifest
    record: SubmissionRecord


@dataclass(frozen=True, slots=True)
class _HeldLock:
    descriptor: int
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class CleanupResult:
    images_deleted: int = 0
    files_deleted: int = 0
    bytes_deleted: int = 0


@dataclass(frozen=True, slots=True)
class SharedGpuStopGuard:
    server_instance_id: str
    request_id: str
    finalization_id: str
    pod_id: str
    gpu_display_name: str
    requester: str
    requested_at: str
    response_deadline: str
    expires_at: str

    def __post_init__(self) -> None:
        for value in (self.server_instance_id, self.request_id, self.finalization_id):
            try:
                parsed = uuid.UUID(value)
            except ValueError as exc:
                raise ValueError("shared GPU Stop guard IDs must be UUIDv4") from exc
            if parsed.version != 4 or parsed.variant != uuid.RFC_4122 or str(parsed) != value:
                raise ValueError("shared GPU Stop guard IDs must be canonical UUIDv4")
        if (
            self.requester != self.requester.strip()
            or not self.requester
            or len(self.requester) > 80
            or not all(character.isprintable() for character in self.requester)
        ):
            raise ValueError("shared GPU Stop requester must be a safe display name")
        if _POD_ID_PATTERN.fullmatch(self.pod_id) is None:
            raise ValueError("shared GPU Stop pod ID is invalid")
        require_gpu_identity(self.gpu_display_name)
        timestamps = [
            _parse_timestamp(value)
            for value in (self.requested_at, self.response_deadline, self.expires_at)
        ]
        raw_timestamps = (
            self.requested_at,
            self.response_deadline,
            self.expires_at,
        )
        if any(value is None for value in timestamps) or any(
            _format_timestamp(parsed) != raw
            for parsed, raw in zip(timestamps, raw_timestamps, strict=True)
            if parsed is not None
        ):
            raise ValueError("shared GPU Stop timestamps must be RFC3339 milliseconds")
        requested_at, response_deadline, expires_at = timestamps
        assert requested_at is not None
        assert response_deadline is not None
        assert expires_at is not None
        if not requested_at <= response_deadline < expires_at:
            raise ValueError("shared GPU Stop timestamps are out of order")
        if response_deadline - requested_at > timedelta(
            seconds=STOP_RESPONSE_TTL_SECONDS
        ) or expires_at - response_deadline > timedelta(seconds=FINALIZATION_TTL_SECONDS):
            raise ValueError("shared GPU Stop timestamps exceed the safety envelope")


class ManifestStore(Protocol):
    def initialize(self) -> None: ...

    def try_acquire_worker_presence(self) -> bool: ...

    def release_worker_presence(self) -> None: ...

    def try_acquire_maintenance_presence(self) -> bool: ...

    def release_maintenance_presence(self) -> None: ...

    @property
    def active_lease_held(self) -> bool: ...

    @property
    def submission_lease_held(self) -> bool: ...

    @property
    def gpu_control_lock_held(self) -> bool: ...

    def try_acquire_active_lease(self) -> bool: ...

    def release_active_lease(self) -> None: ...

    def try_acquire_submission_lease(self) -> bool: ...

    def release_submission_lease(self) -> None: ...

    def try_acquire_gpu_control_lock(self) -> bool: ...

    def release_gpu_control_lock(self) -> None: ...

    def write_gpu_stop_guard(self, guard: SharedGpuStopGuard) -> None: ...

    def read_gpu_stop_guard(self) -> SharedGpuStopGuard | None: ...

    def clear_gpu_stop_guard(self, expected: SharedGpuStopGuard) -> None: ...

    def clear_stale_gpu_stop_guard(self) -> None: ...

    def list_batch_ids(self) -> list[str]: ...

    def create(
        self,
        manifest: BatchManifest,
        reference_payloads: list[tuple[str, bytes]] | None = None,
        submission: SubmissionRecord | None = None,
    ) -> None: ...

    def load(self, batch_id: str) -> BatchManifest: ...

    def find_submission(self, client_submission_id: str) -> SubmissionMatch | None: ...

    def submission_store_corrupt(self) -> bool: ...

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

    def __init__(
        self,
        root: Path,
        *,
        fsync_writes: bool = True,
        crash_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root
        self.batches_root = root / "batches"
        self.fsync_writes = fsync_writes
        # Test-only deterministic seam injection. Production never configures
        # it, and every hook point is after the named durable boundary.
        self._crash_hook = crash_hook
        self._worker_presence_descriptor: _HeldLock | None = None
        self._maintenance_presence_descriptor: _HeldLock | None = None
        self._active_lease_descriptor: _HeldLock | None = None
        self._submission_lease_descriptor: _HeldLock | None = None
        self._gpu_control_descriptor: _HeldLock | None = None

    @property
    def active_lease_held(self) -> bool:
        return self._active_lease_descriptor is not None

    @property
    def submission_lease_held(self) -> bool:
        return self._submission_lease_descriptor is not None

    @property
    def gpu_control_lock_held(self) -> bool:
        return self._gpu_control_descriptor is not None

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
        descriptor = self._try_acquire_lock(".worker-presence.lock", shared=True)
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
        descriptor = self._try_acquire_lock(".worker-presence.lock", shared=False)
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
        descriptor = self._try_acquire_lock(".active-batch.lock", shared=False)
        if descriptor is None:
            return False
        self._active_lease_descriptor = descriptor
        return True

    def release_active_lease(self) -> None:
        descriptor = self._active_lease_descriptor
        self._active_lease_descriptor = None
        self._release_lock(descriptor)

    def try_acquire_submission_lease(self) -> bool:
        """Serialize envelope lookup and first admission across worker processes.

        This short-lived volume lock is independent of the active generation
        lease: it lets a second worker safely replay an already-admitted key
        while the first batch is still generating, without ever admitting a
        second remote batch.
        """

        if self._submission_lease_descriptor is not None:
            return True
        descriptor = self._try_acquire_lock(".submission-history.lock", shared=False)
        if descriptor is None:
            return False
        self._submission_lease_descriptor = descriptor
        return True

    def release_submission_lease(self) -> None:
        descriptor = self._submission_lease_descriptor
        self._submission_lease_descriptor = None
        self._release_lock(descriptor)

    def try_acquire_gpu_control_lock(self) -> bool:
        self._require_active_lease()
        if self._gpu_control_descriptor is not None:
            return True
        descriptor = self._try_acquire_lock(".gpu-control-v1.lock", shared=False)
        if descriptor is None:
            return False
        self._gpu_control_descriptor = descriptor
        return True

    def release_gpu_control_lock(self) -> None:
        descriptor = self._gpu_control_descriptor
        self._gpu_control_descriptor = None
        self._release_lock(descriptor)

    def write_gpu_stop_guard(self, guard: SharedGpuStopGuard) -> None:
        """Publish a strict cross-process finalization marker under the active lease."""

        self._require_active_lease()
        self._require_gpu_control_lock()
        payload = json.dumps(
            {
                "schema_version": 2,
                "server_instance_id": guard.server_instance_id,
                "request_id": guard.request_id,
                "finalization_id": guard.finalization_id,
                "pod_id": guard.pod_id,
                "gpu_display_name": guard.gpu_display_name,
                "requester": guard.requester,
                "requested_at": guard.requested_at,
                "response_deadline": guard.response_deadline,
                "expires_at": guard.expires_at,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._atomic_write(self.root / _GPU_STOP_GUARD_FILENAME, payload)

    def read_gpu_stop_guard(self) -> SharedGpuStopGuard | None:
        path = self.root / _GPU_STOP_GUARD_FILENAME
        try:
            if path.stat().st_size > _MAX_GPU_STOP_GUARD_BYTES:
                return None
            payload = json.loads(path.read_bytes())
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "server_instance_id",
            "request_id",
            "finalization_id",
            "pod_id",
            "gpu_display_name",
            "requester",
            "requested_at",
            "response_deadline",
            "expires_at",
        }:
            return None
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 2:
            return None
        values = [
            payload["server_instance_id"],
            payload["request_id"],
            payload["finalization_id"],
            payload["pod_id"],
            payload["gpu_display_name"],
            payload["requester"],
            payload["requested_at"],
            payload["response_deadline"],
            payload["expires_at"],
        ]
        if not all(isinstance(value, str) for value in values):
            return None
        try:
            return SharedGpuStopGuard(
                server_instance_id=payload["server_instance_id"],
                request_id=payload["request_id"],
                finalization_id=payload["finalization_id"],
                pod_id=payload["pod_id"],
                gpu_display_name=payload["gpu_display_name"],
                requester=payload["requester"],
                requested_at=payload["requested_at"],
                response_deadline=payload["response_deadline"],
                expires_at=payload["expires_at"],
            )
        except ValueError:
            return None

    def clear_gpu_stop_guard(self, expected: SharedGpuStopGuard) -> None:
        """Idempotently clear only the marker owned by the exact local guard."""

        self._require_active_lease()
        self._require_gpu_control_lock()
        path = self.root / _GPU_STOP_GUARD_FILENAME
        current = self.read_gpu_stop_guard()
        if current is None:
            if path.exists():
                raise RuntimeError("the shared GPU Stop guard marker is invalid")
            return
        if current != expected:
            raise RuntimeError("the shared GPU Stop guard marker belongs to another grant")
        path.unlink(missing_ok=True)
        self._fsync_directory(self.root)

    def clear_stale_gpu_stop_guard(self) -> None:
        """Clear a crashed owner's marker only after acquiring its released lease."""

        self._require_active_lease()
        self._require_gpu_control_lock()
        path = self.root / _GPU_STOP_GUARD_FILENAME
        path.unlink(missing_ok=True)
        self._fsync_directory(self.root)

    def _try_acquire_lock(self, name: str, *, shared: bool) -> _HeldLock | None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(self.root / name, flags, 0o600)
        if os.name == "nt":
            return self._try_acquire_windows_lock(
                descriptor,
                shared=shared,
                presence=name == ".worker-presence.lock",
            )
        try:
            fcntl.flock(
                descriptor,
                (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB,
            )
        except OSError as exc:
            os.close(descriptor)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return None
            raise
        return _HeldLock(descriptor, 0, 0)

    @staticmethod
    def _try_acquire_windows_lock(
        descriptor: int,
        *,
        shared: bool,
        presence: bool,
    ) -> _HeldLock | None:
        """Use byte-range locks because Windows has no shared flock operation.

        Each worker reserves one byte in the presence file. Maintenance takes the
        whole 256-byte range, so it cannot start while any worker process is alive.
        The active-batch file uses the same exclusive range mechanism with one byte.
        """

        required_size = _WINDOWS_PRESENCE_SLOTS if presence else 1
        try:
            if os.fstat(descriptor).st_size < required_size:
                os.ftruncate(descriptor, required_size)
            offsets = range(_WINDOWS_PRESENCE_SLOTS) if presence and shared else (0,)
            length = 1 if presence and shared else required_size
            for offset in offsets:
                os.lseek(descriptor, offset, os.SEEK_SET)
                try:
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, length)
                except OSError as exc:
                    if exc.errno in {errno.EACCES, errno.EAGAIN}:
                        continue
                    raise
                return _HeldLock(descriptor, offset, length)
        except BaseException:
            os.close(descriptor)
            raise
        os.close(descriptor)
        return None

    @staticmethod
    def _release_lock(lock: _HeldLock | None) -> None:
        if lock is None:
            return
        try:
            if os.name == "nt":
                os.lseek(lock.descriptor, lock.offset, os.SEEK_SET)
                msvcrt.locking(lock.descriptor, msvcrt.LK_UNLCK, lock.length)
            else:
                fcntl.flock(lock.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock.descriptor)

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
        self,
        manifest: BatchManifest,
        reference_payloads: list[tuple[str, bytes]] | None = None,
        submission: SubmissionRecord | None = None,
    ) -> None:
        """Create an immutable batch directory and publish one admission envelope.

        For newly admitted batches the v2 envelope rename is the only commit
        point. A crash before it leaves an ignored incomplete directory; a
        crash after it leaves a replayable owner/fingerprint association. The
        optional legacy path is retained solely so already-persisted v1
        manifests and explicit migration fixtures remain readable.
        """

        self._require_active_lease()
        if submission is not None:
            self._require_submission_lease()
            self._validate_submission_for_manifest(manifest, submission)
        elif manifest.client_submission_id is not None:
            raise ValueError("a manifest submission ID requires a private submission record")

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
        # The reference data and all newly created directories must be durable
        # before an envelope can make this submission discoverable.
        self._fsync_directory(references_dir)
        self._fsync_directory(batch_dir)
        self._fsync_directory(self.batches_root)
        self._crash("after_reference_fsync")

        manifest.recalculate_progress()
        if submission is None:
            self._write_legacy_manifest(manifest)
            return
        self._write_envelope(manifest, submission, creation=True)

    def load(self, batch_id: str) -> BatchManifest:
        manifest, _ = self._read_manifest_record(batch_id)
        return manifest

    def find_submission(self, client_submission_id: str) -> SubmissionMatch | None:
        """Return one exact durable submission association or fail closed.

        We deliberately scan every v2 envelope instead of trusting an index.
        The manifest file is atomically replaced, so a concurrent reader sees
        either the prior complete file or the next complete file. Scanning all
        records also lets one malformed v2 envelope disable new admission
        rather than turning a potentially admitted key into a false miss.
        """

        require_canonical_uuid4(client_submission_id)
        self._require_submission_lease()
        match: SubmissionMatch | None = None
        for manifest, submission in self._scan_submission_history():
            if submission is None or submission.client_submission_id != client_submission_id:
                continue
            candidate = SubmissionMatch(manifest=manifest, record=submission)
            if match is not None:
                # A duplicate key is impossible under the admission lease and
                # must never be resolved by picking an arbitrary manifest.
                raise SubmissionStoreCorruptError("duplicate submission association")
            match = candidate
        return match

    def submission_store_corrupt(self) -> bool:
        """Report any malformed persisted manifest/envelope without exposing it."""

        try:
            self._scan_submission_history()
            return False
        except SubmissionStoreCorruptError:
            return True

    def _scan_submission_history(
        self,
    ) -> list[tuple[BatchManifest, SubmissionRecord | None]]:
        """Validate the complete admission namespace as one closed set.

        A duplicate key anywhere makes every create/lookup ambiguous, even if
        the caller asked about another key. Directory enumeration and record
        I/O failures are likewise fail-closed instead of accidentally
        advertising a writable store through `/v1/status`.
        """

        records: list[tuple[BatchManifest, SubmissionRecord | None]] = []
        seen_submission_ids: set[str] = set()
        try:
            batch_ids = self.list_batch_ids()
            for batch_id in batch_ids:
                manifest, submission = self._read_manifest_record(batch_id)
                if submission is not None:
                    if submission.client_submission_id in seen_submission_ids:
                        raise SubmissionStoreCorruptError("duplicate submission association")
                    seen_submission_ids.add(submission.client_submission_id)
                records.append((manifest, submission))
        except SubmissionStoreCorruptError:
            raise
        except OSError as exc:
            raise SubmissionStoreCorruptError(
                "submission history could not be enumerated or read"
            ) from exc
        return records

    def save(self, manifest: BatchManifest) -> None:
        self._require_active_lease()
        # Preserve the private record across every progress/receipt mutation.
        # A v2 manifest cannot silently downgrade to a legacy raw file.
        _, submission = self._read_manifest_record(manifest.batch_id)
        manifest.recalculate_progress()
        if submission is None:
            self._write_legacy_manifest(manifest)
            return
        self._validate_submission_for_manifest(manifest, submission)
        self._write_envelope(manifest, submission, creation=False)

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

    def _read_manifest_record(self, batch_id: str) -> tuple[BatchManifest, SubmissionRecord | None]:
        path = self._batch_dir(batch_id) / "manifest.json"
        try:
            raw_payload = path.read_bytes()
        except FileNotFoundError:
            raise
        try:
            payload = json.loads(raw_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SubmissionStoreCorruptError("manifest JSON is not valid") from exc
        if not isinstance(payload, dict):
            raise SubmissionStoreCorruptError("manifest root is not an object")

        # API manifests are schema version 1. New durable storage deliberately
        # wraps that public document in a private version-2 envelope.
        if payload.get("schema_version") == _MANIFEST_ENVELOPE_SCHEMA_VERSION:
            try:
                envelope = ManifestEnvelopeV2.model_validate_json(raw_payload)
            except (ValidationError, ValueError) as exc:
                raise SubmissionStoreCorruptError("v2 manifest envelope is invalid") from exc
            return envelope.manifest, envelope.submission
        try:
            legacy = BatchManifest.model_validate_json(raw_payload)
        except (ValidationError, ValueError) as exc:
            # A malformed/unknown document cannot be safely distinguished from
            # damaged v2 history, so fail closed for new admissions.
            raise SubmissionStoreCorruptError("legacy manifest document is invalid") from exc
        if legacy.client_submission_id is not None:
            raise SubmissionStoreCorruptError("legacy manifest has an unbound submission ID")
        return legacy, None

    @staticmethod
    def _validate_submission_for_manifest(
        manifest: BatchManifest, submission: SubmissionRecord
    ) -> None:
        if manifest.client_submission_id != submission.client_submission_id:
            raise ValueError("manifest submission ID does not match its private record")
        if manifest.owner.user_id != submission.owner_user_id:
            raise ValueError("manifest owner does not match its private record")

    def _write_legacy_manifest(self, manifest: BatchManifest) -> None:
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

    def _write_envelope(
        self,
        manifest: BatchManifest,
        submission: SubmissionRecord,
        *,
        creation: bool,
    ) -> None:
        envelope = ManifestEnvelopeV2(manifest=manifest, submission=submission)
        payload = json.dumps(
            envelope.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self._batch_dir(manifest.batch_id) / "manifest.json"
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
        self._atomic_write(
            path,
            payload,
            after_file_sync="after_envelope_fsync" if creation else None,
            after_rename="after_envelope_rename" if creation else None,
            extra_fsync_directories=(self.batches_root,) if creation else (),
        )

    def _require_active_lease(self) -> None:
        if self._active_lease_descriptor is None:
            raise RuntimeError("manifest and artifact mutations require the active-volume lease")

    def _require_submission_lease(self) -> None:
        if self._submission_lease_descriptor is None:
            raise RuntimeError(
                "submission lookup and admission require the submission-volume lease"
            )

    def _require_gpu_control_lock(self) -> None:
        if self._gpu_control_descriptor is None:
            raise RuntimeError("GPU-control mutation requires the shared GPU-control lock")

    def _write_immutable(self, path: Path, payload: bytes) -> None:
        if path.exists():
            if self._matches(path, len(payload), hashlib.sha256(payload).hexdigest()):
                return
            raise FileExistsError(f"immutable artifact already exists: {path.name}")
        self._atomic_write(path, payload)

    def _atomic_write(
        self,
        path: Path,
        payload: bytes,
        *,
        after_file_sync: str | None = None,
        after_rename: str | None = None,
        extra_fsync_directories: tuple[Path, ...] = (),
    ) -> None:
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
            if after_file_sync is not None:
                self._crash(after_file_sync)
            os.replace(temporary_path, path)
            if self.fsync_writes:
                self._fsync_directory(path.parent)
                for directory in extra_fsync_directories:
                    self._fsync_directory(directory)
            if after_rename is not None:
                self._crash(after_rename)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _crash(self, point: str) -> None:
        if self._crash_hook is not None:
            self._crash_hook(point)

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


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
