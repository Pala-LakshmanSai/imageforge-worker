# ImageForge worker

This directory contains the Python 3.11/FastAPI worker for one ImageForge GPU.
It owns exactly one shared-volume POSIX batch lease, generates images sequentially
on one approved GPU, and stores crash-safe manifests on the RunPod network volume.
It also hosts ephemeral authenticated studio presence and Stop-consent state. It
never creates, stops, or terminates a Pod.

## Runtime contract

- API schema: `1`
- model: `Comfy-Org/Mage-Flow`
- model revision: `d8c99241f6fa80fbd453014234af2bf337ea21e6`
- precision: BF16, loaded directly onto one NVIDIA CUDA 13.0+ device with at least 16,380 MiB VRAM
- output: 1280x720 JPEG quality 95 and 320x180 WebP preview
- inference: four steps, guidance 1.0
- retries: one initial attempt and two automatic retries
- prompts: any non-empty finite list; prompt text is preserved without a product
  count or per-prompt byte cap
- aspect_ratio: one of `16:9`, `1:1`, `9:16`, `4:3`, or `3:4`; the worker
  derives a fixed, model-safe output size for the selected delivery shape.
- references: optional batch-level JPEG, PNG, or WebP images (up to 8 files,
  8 MiB each and 32 MiB total); manifests retain only safe metadata and checksums

The pinned Python 3.11 slim image plus the SHA-256-pinned `torch==2.11.0+cu130`
wheel supports the approved Ampere, Ada, and Blackwell families; RunPod supplies
the host NVIDIA driver. The Docker base image, Python minor version, direct
dependencies, PyTorch wheel, ComfyUI revision, model revision, and schema are
pinned.
The production adapter uses
`local_files_only=True` while all Hugging Face offline flags are set. Model
weights must be provisioned once onto `/workspace/models/huggingface` before a
normal start. Boot never installs packages and never downloads weights.

For the separately authorized one-time volume preparation, run:

```sh
python worker/scripts/prepare_mageflow_volume.py \
  --cache-dir /workspace/models/huggingface \
  --confirm-download
```

`--confirm-download` temporarily disables only `HF_HUB_OFFLINE` inside that
preparation process and restores it afterward. Normal worker boot always remains
offline. The command pins the model revision and downloads the three files the
worker needs: the INT8 ConvRot transformer (4.16 GB), the Qwen3-VL-4B text
encoder (8.88 GB), and the Mage-VAE (0.35 GB), roughly 13.4 GB in total on the
50 GB EU-RO-1 network volume. Pass `--include-bf16-fallback` to add the 8.23 GB
BF16 transformer, which the `mageflow-bf16` backend uses.

Weights come from the public `Comfy-Org/Mage-Flow` mirror. The upstream
`microsoft/Mage-Flow*` repositories are gated and return HTTP 401 without an
accepted licence, and Diffusers cannot load Mage-Flow in any case, so the worker
drives a pinned, loopback-only ComfyUI child process instead. See
`docs/MAGEFLOW_STAGING.md`.

This preparation command is never invoked by the Docker entrypoint.

GPU family selection remains the desktop/RunPod provider's responsibility. The
worker is hardware-generic across the approved NVIDIA fallback pool: it checks
that exactly one CUDA device is visible and that the actual device has at least
16,380 MiB VRAM, then reports its real name, capacity, allocation, reservation,
and peak memory through health. The byte-exact floor is only 4 MiB below nominal
16 GiB, accommodating the approved emergency RTX 2000 Ada's reported capacity
without admitting materially smaller cards. An undersized device fails measured
readiness instead of attempting CPU offload.

RunPod network-volume Pods are terminated and recreated rather than stopped.
The new Pod gets a new ID, but `/workspace` persists. Every initialized worker
attempts to hold shared presence at `/workspace/imageforge/.worker-presence.lock`
for its full process lifetime. Separately, the active batch owner holds an
exclusive advisory lock at `/workspace/imageforge/.active-batch.lock` for the
entire running, paused, or interrupted lifetime; a duplicate process that
observes that lock remains read-only. Local multi-process tests verify the worker
logic, but
they do **not** establish that a selected RunPod network-volume driver propagates
`flock` correctly between Pods. Before approving a deployment region/volume type,
run the paid two-Pod gate below. Do not rely on this lease in production until
that exact storage configuration passes.
After the prior process exits, recovery scans
`/workspace/imageforge/batches`, validates every ready artifact against its
size and SHA-256, changes a previously running batch to `interrupted`, and
retains that batch's lease until its owner resumes or cancels it. Manifests do
not contain or depend on a Pod ID.

## Authentication

Every route except `GET /v1/health` requires a bearer credential. Inject
`IMAGEFORGE_AUTH_TOKENS_JSON` from a RunPod secret; never commit it. The value is
a JSON list shaped as follows (values below are descriptive, not credentials):

```json
[
  {"user_id": "user-id", "display_name": "Display name", "token": "runtime-secret"}
]
```

Tokens use the RFC bearer-token ASCII alphabet and are compared in constant time.
Display names must contain 1-80 trimmed printable characters because they are
projected into authenticated studio coordination responses.
Unknown batch IDs and batches owned by a
different user intentionally return the same `batch_not_found` response. The
worker does not log authorization headers or prompt content. OpenAPI and docs
routes are disabled in the production app.

## Local verification

Keep the environment and caches on the removable disk:

```sh
cd /Volumes/ESD-USB/ImageForge-worktrees/worker/worker
python3.11 -m venv .venv
PIP_CACHE_DIR=/Volumes/ESD-USB/ImageForge-caches/pip .venv/bin/pip install -e '.[test]'
TMPDIR=$PWD/.tmp .venv/bin/pytest
.venv/bin/ruff check src tests
```

Tests inject the deterministic fake adapter directly and make no RunPod calls.
The fake emits valid JPEG/WebP files derived from prompt, index, and seed. A
real-GPU test is excluded unless `IMAGEFORGE_REAL_GPU_TEST=1` is set explicitly;
running that paid gate also requires the pinned weights and an already-authorized
Pod. The harness never creates or terminates compute. Run it once on representative
Ampere, Ada, and Blackwell Pods (the current pool includes RTX A4500, RTX 2000
Ada/RTX 4090, and RTX 5090 architectures):

```sh
IMAGEFORGE_REAL_GPU_TEST=1 IMAGEFORGE_REAL_GPU_FAMILY=ampere \
  IMAGEFORGE_MODEL_CACHE_DIR=/workspace/models/huggingface \
  .venv/bin/pytest -m real_gpu tests/test_real_gpu_smoke.py
IMAGEFORGE_REAL_GPU_TEST=1 IMAGEFORGE_REAL_GPU_FAMILY=ada \
  IMAGEFORGE_MODEL_CACHE_DIR=/workspace/models/huggingface \
  .venv/bin/pytest -m real_gpu tests/test_real_gpu_smoke.py
IMAGEFORGE_REAL_GPU_TEST=1 IMAGEFORGE_REAL_GPU_FAMILY=blackwell \
  IMAGEFORGE_MODEL_CACHE_DIR=/workspace/models/huggingface \
  .venv/bin/pytest -m real_gpu tests/test_real_gpu_smoke.py
```

The real shared-volume lease is a separate paid/deployment gate. Attach the exact
candidate network volume to two Pods. Hold a running batch on Pod A, then verify
Pod B can read status but every mutation returns HTTP 423 and the retention
cleanup command refuses to acquire its exclusive maintenance guard. Force-stop
Pod A, verify Pod B recovers the manifest exactly once, then repeat the contention
check in both directions. Record the region, volume type, mount options, image
digest, and results; any failure disqualifies that storage configuration.

## API behavior

`POST /v1/batches` accepts `{"prompts":[...],"base_seed":0,"aspect_ratio":"16:9","references":[{"name":"anchor.png","mime_type":"image/png","data_hex":"..."}]}`.
References are optional, batch-level, and apply to every prompt; raw bytes are
stored only as temporary worker files and never copied into prompt text or logs.
Batch IDs and all paths are server generated. If a lease is already held, the server returns
HTTP 423 with stable code `batch_busy`, the owner's display name, and progress;
there is no queue. Pause stops before the next image while retaining the lease.
Cancel allows the current image to finish, cancels the remainder, and releases
the lease. `retry-failed` reopens only terminally failed images when no other
batch owns the GPU.
A duplicate Pod that does not hold the volume lease returns typed `worker_standby`
for mutation attempts while continuing to expose authorized read-only status.

### Studio coordination

Clients heartbeat `PUT /v1/studio/sessions/{session_id}` with only
`foreground`/`background` availability and read shared state with `GET` on the
same path. Presence expires after 15 seconds and is never durable across worker
restart. Duplicate windows for one authenticated user count as one approval
participant.

`POST /v1/studio/stop-requests` binds consent to an exact Pod ID. Any active
batch returns `stop_blocked_by_active_batch`. Otherwise every other live
foreground user must approve through the response endpoint. Denial, timeout,
requester expiry, or cancellation keeps compute running. Pending approval never
blocks generation: valid create/resume/retry wins atomically and cancels the
request. After approval, the requester may acquire a 60-second finalization
guard; only that state rejects new generation with `gpu_stop_pending`. The
desktop cancels the exact guard after a definite provider failure, while expiry
fails safely. The guard outlives requester heartbeat loss until that expiry so
an in-flight or ambiguous provider call cannot release generation early. It is
backed by both the shared-volume active lease and an atomic marker, so every
worker process observes the same admission block. A replacement process adopts
an unexpired crash or shutdown marker for its remaining TTL while resetting all
client sessions and approvals. Full shapes and error codes are in
`docs/API_CONTRACT.md`.

After that adoption, studio heartbeats expose a read-only synthetic
`finalizing` request with the exact Pod/GPU/requester/expiry and a null
`finalization_id`. It informs every client and keeps Generate disabled without
authorizing any new-epoch session to respond, cancel, finalize, or repeat the
provider deletion. The synthetic state disappears when the shared guard expires.

## Retention cleanup

Receipts are durable, but the current manifest represents one acknowledgement
rather than an all-device acknowledgement set. ImageForge therefore never deletes
artifacts automatically. An operator may explicitly remove only checksum-verified,
acknowledged artifacts after a minimum 24-hour safety window. Terminate all worker
Pods before running this maintenance command. Before any storage probe or cleanup,
the command must acquire exclusive worker presence and therefore refuses while
even an idle initialized worker is alive. It then also acquires the active-batch
lease. Both locks are defense in depth and depend on the real cross-Pod validation
gate above:

```sh
python -m imageforge_worker.cleanup_retention \
  --data-root /workspace/imageforge \
  --minimum-age-hours 24 \
  --confirm-cleanup
```

Unacknowledged, recent, corrupt, or active-batch artifacts are never eligible.
For each eligible image, cleanup durably records intent before unlinking either
file and records completion afterward. Interrupted cleanup is resumable, and the
durable receipt, filenames, sizes, and checksums remain at every crash boundary;
subsequent artifact requests return the typed `artifact_expired` response as soon
as cleanup intent is durable.

Artifact responses include `Content-Type`, `Content-Length`, `Digest`, `ETag`,
`X-ImageForge-SHA256`, and `X-Checksum-SHA256`. A receipt must repeat the full
JPEG's verified SHA-256 and size. Receipts are durable acknowledgements; the
desktop remains responsible for its per-computer local receipt ledger.
