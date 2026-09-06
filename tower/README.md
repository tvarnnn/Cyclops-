# Glasses Tower

The Windows tower transport layer for the Glasses platform. Current
milestone: V0.9.3 — World Builder Foundations, Experiments 1–2. The tower
exposes a persistent runtime with a module container that supports stateful,
model-backed computer vision experiments. The core transport layer (WebSocket
health check, frame receive, JPEG validation, per-session measurements) is
fully decoupled from module implementations; modules register CV experiments
and handle frame processing. The `baseline` experiment runs deterministic
OpenCV (grayscale + mean intensity), while the `depth` experiment uses
MiDaS-small monocular depth estimation on GPU/CPU. Both log per-session
streaming measurements (FPS, bandwidth, sequence-gap count, Tower-side drops,
processing latency, CPU/RSS), plus per-stage timing for module operations.
Frames are processed in memory only and never written to disk. Frame-level
and module-level failures are reported to the connected client via a
`frame_error` message rather than silently dropping frames.

Two offline research harnesses (`scripts/depth_temporal_consistency.py`,
`scripts/feature_trackability.py`) analyse World Builder foundations against
recorded footage; their measured results and validity limits are in
`guidelines/docs/reports/V0.9.3-world-builder-experiments-1-2-report.md`.

## Environment Setup

Requires Python 3.12. One command, from the tower root:

```powershell
powershell -NoProfile -File scripts\setup_tower.ps1
```

It is idempotent — safe to re-run any time as a health check. It creates
`.venv` only if missing, installs the package and its `dev` extra, verifies
that the install actually landed and that `tower.main:app` imports, writes a
`.env` if there isn't one (never overwriting yours), reports the firewall
rule and your LAN address, and prints an actionable fix for anything it
can't do itself. It never deletes a venv, never installs the `ml`/`ocr`
extras, and never touches the firewall.

Two things it pins on purpose:

- **`py -3.12`, not `py`.** On this machine the bare launcher resolves to
  Python 3.14, which builds a venv that installs cleanly enough to look
  fine and then fails later on a wheel with no 3.14 build.
- **`.venv\Scripts\python.exe -m pip`, not `pip`.** Nothing in this repo's
  scripts depends on `Activate.ps1` having been run. A bare `pip` in an
  unactivated shell installs into the system Python and leaves the venv
  untouched, which looks exactly like the install having failed for no
  reason.

If you'd rather do it by hand, that is the same as:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

This installs FastAPI, Uvicorn, Pillow, OpenCV (headless), NumPy, psutil,
and the test dependencies (pytest, **httpx2**, websockets). It is `httpx2`,
not `httpx`, deliberately — see the comment on that line in
`pyproject.toml` before "correcting" it.

## Model-Backed Experiments (Optional)

The `depth` and `object_detection` experiments require the `ml` extra,
which includes PyTorch, torchvision, and timm (needed by MiDaS's hubconf
backbone chain). `object_detection` additionally downloads ~13.4 MB of
torchvision COCO weights on first run. Every other experiment needs only
OpenCV and runs by default.

**Install order matters here.** `pyproject.toml`'s `ml` extra declares an
unconstrained `"torch"`/`"torchvision"` requirement (no CUDA-specific index
pin), so if you run `pip install -e ".[dev,ml]"` first, pip will resolve
torch/torchvision from plain PyPI — which is very likely a **CPU-only**
wheel. Once that's installed, it already satisfies the unconstrained
requirement, so a later `pip install -e ".[dev,ml]"` will **not**
force-replace it, and even running the `--index-url` command below
afterward can report "already satisfied" without actually giving you a CUDA
build. The practical symptom: `TOWER_CV_DEVICE=cuda` fails with a
`RuntimeError` from `resolve_device()` (`tower/experiments/depth.py`)
because CUDA isn't actually available to torch, and `TOWER_CV_DEVICE=auto`
silently falls back to CPU with no error at all.

If you plan to use `TOWER_CV_DEVICE=cuda` (i.e. you want the GPU baseline,
not just a CPU-only `depth` experiment), install the CUDA-indexed
torch/torchvision build **first**, before the extras:

```powershell
.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -e ".[dev,ml]"
```

The second command then only needs to add `timm` and the `dev` extras —
torch/torchvision are already installed and already satisfy the
unconstrained requirement, so pip leaves them alone.

If you only need the `depth` experiment on CPU (no CUDA), plain
`pip install -e ".[dev,ml]"` on its own is fine.

### PyTorch/CUDA Installation

On this Windows machine with CUDA 13.2 support, the verified working install
command is:

```powershell
.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

This uses the `cu132` index, which is the newest stable PyTorch build matching
the driver-reported CUDA 13.2 support. (Other indexes like `cu128` or `cu129`
may be stale.) Run this **before** `pip install -e ".[dev,ml]"` — see the
install-order warning above; it is not automatically part of that workflow.

### MAX_PATH Warning (Windows)

A plain `pip install` of torch/torchvision from a path this deep (e.g., a git
worktree in `C:\Users\tvllo\Projects\...`) can fail with:

```
OSError: [WinError 206] The filename or extension is too long
```

This is Windows' MAX_PATH (260 character) limit, tripped by torch's
deeply-nested bundled third-party license file tree. **Do not attempt to
enable Windows' global `LongPathsEnabled` registry setting** as an ad-hoc fix
(it's a machine-wide policy change outside a normal install's scope). Instead,
extract the wheel manually to the target site-packages directory using the
`\\?\` Win32 extended-length path prefix, which bypasses MAX_PATH for file
APIs without requiring a registry change (always supported by NTFS/Win32).

### MiDaS-small Weight Downloads

The first time the `depth` experiment loads, it downloads MiDaS-small's weights
(and a nested `rwightman/gen-efficientnet-pytorch` backbone) via `torch.hub`
— this requires internet access once. Results are cached afterward
(`~/.cache/torch/hub`) and no further downloads occur on subsequent loads.

## Running the Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

## Starting the Server

```powershell
powershell -NoProfile -File scripts\start_tower.ps1
```

That is the whole normal flow — one command, one terminal. The script
preflights the venv, diagnoses whatever owns the port instead of failing
with `[WinError 10048]`, prints the configuration that will actually be in
effect, and runs uvicorn from the tower root.

| Flag | Default | Purpose |
|---|---|---|
| `-Port` | `8000` | Port to listen on |
| `-BindHost` | `0.0.0.0` | Interface to bind. Named `-BindHost` because `$Host` is an automatic PowerShell variable |
| `-Reload` | off | uvicorn auto-reload |
| `-Force` | off | If the port is held, try to stop the owning process. Off by default; the script always prints the `Stop-Process` command whether or not you pass it |

The equivalent by hand:

```powershell
.venv\Scripts\python.exe -m uvicorn tower.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

Three things about that line are load-bearing:

- **Run it from the tower root.** `tower/world_builder/redaction.py` resolves
  the YuNet weights relative to the process CWD. Start the server from
  anywhere else and face redaction is silently disabled — nothing fails,
  keyframes just honestly record their redaction as `none`.
- **`--host` is not optional.** uvicorn's own default host is `127.0.0.1`,
  not `0.0.0.0`. Omit `--host` and every check from this machine passes
  while the phone cannot connect at all.
- **Never `--factory`.** `tower/main.py` calls `create_app()` at import, and
  `create_app()` starts the module container. The factory form would build a
  second one.

Configuration is read from environment variables, all optional. Put them in
`.env` (gitignored; `scripts\setup_tower.ps1` writes a starting one) and
uvicorn's `--env-file` loads them in `Config.__init__`, before the app is
imported, so they reach `get_settings()`.

| Variable          | Default   | Purpose                                   |
|-------------------|-----------|--------------------------------------------|
| `TOWER_HOST`       | `0.0.0.0` | **Not wired to anything.** `tower/config.py` reads it into `Settings` and nothing reads it back; setting it binds nothing. Use `-BindHost` / `--host` |
| `TOWER_PORT`       | `8000`    | **Not wired to anything**, same as `TOWER_HOST`. Use `-Port` / `--port` |
| `TOWER_DEV_MODE`   | `true`    | Enables debug-level logging. Does **not** control the per-frame `[Tower][Frame]` lines, which are INFO and always on |
| `TOWER_CV_EXPERIMENT` | `baseline` | **Startup default only.** Which CV experiment this Tower arms at boot: `baseline`, `edge_detection`, `frame_quality`, `feature_detection`, `optical_flow`, `redaction_impact`, `object_detection`, `depth`. Since 2026-08-27 a client selects an experiment at runtime with `cv_lab_start` and no restart — see `docs/contracts/EXPERIMENTAL-CV-LAB.md`. This variable is what runs before anyone asks, so that a client which knows nothing about the CV Lab still gets a `frame_result` for every frame |
| `TOWER_CV_DEVICE`   | `auto`    | Device for model-backed experiments (`auto`, `cpu`, or `cuda`) |
| `TOWER_CV_TORCH_THREADS` | `auto` | torch's intra-op thread budget while a model-backed CV experiment is armed. `auto` is **2 on CUDA, 4 on CPU**, from a measured sweep (`tower/experiments/depth.py`); a positive integer is applied as given; `0` leaves torch's default of one thread per logical CPU — which on a 20-core host is twenty OpenMP workers spin-waiting between kernel launches, and is what "object detection pegs the machine at 99% CPU" was (12.6 cores at 20 threads against 1.8 at 2, same latency). Process-wide, like torch's own setting; if `TOWER_SCENE_TORCH_THREADS` is also set, whichever model loaded last decides |
| `TOWER_CAPTURE_ROOT` | *(unset)* | Arms the raw dataset recorder at this path. **Unset means no recording, ever.** Arming is not recording: nothing is written until a `stream_start` arrives, and `GET /health` reports the state. Use `data` — `tower/capture.py` appends `captures/<id>` itself |
| `TOWER_WORLD_ROOT` | *(unset)* | Where World Builder worlds are stored. **Unset means iOS sees World Builder as unsupported.** Use `data/world_builder` — `tower/world_builder/store.py` appends `worlds/<id>` itself, and the value must equal `DEFAULT_ROOT` in `scripts/world_build_session.py` or the result channel reads a different tree than the builder writes |
| `TOWER_WORLD_AUTOBUILD` | `true` | Whether each capture automatically gets a World Builder follower attached. Only has an effect when `TOWER_WORLD_ROOT` is set. Turn it off to keep reporting existing worlds while building no new ones — useful when reprocessing a recorded capture offline, and the escape hatch if auto-attach misbehaves |
| `TOWER_WORLD_REBUILD_EVERY` | `4` | Keyframes between mid-walk rebuilds in the attached follower. **Deliberately not the script's own default of `0`**, which means "build once, at the end" — correct for a batch reprocess and wrong for a live walk. `0` here is why the 2026-08-24 test showed a climbing keyframe count and no geometry at all until the capture closed |
| `TOWER_WORLD_REGISTER` | `true` | Whether the follower tries to place the walk's segments in one coordinate frame when the walk ends. Off is what shipped before 2026-08-29, and off is why every walk arrived on the phone as disconnected fragments: the registration pass existed, was tested, persisted and served, and nothing invoked it. It runs **once, after the final build, in the follower subprocess** — never on the frame path — and costs ~20 s on a 36-segment walk. Refusal is still the default answer for any pair, so this makes a world no more confident, only less silent. Turn it off if a walk must finalise the instant it stops; the reconstruction is unaffected |

The server does **not** bind `0.0.0.0` unless you say so: `TOWER_HOST` is
inert, and uvicorn's own default is `127.0.0.1`. `scripts\start_tower.ps1`
defaults `-BindHost` to `0.0.0.0` so it is reachable from other devices on
the LAN, and by-hand invocations must pass `--host 0.0.0.0` themselves.

### When the port is already in use

`scripts\start_tower.ps1` resolves the owning PID, its process name, and its
command line, and says whether it looks like our own stale uvicorn (its
command line names `tower.main:app`). It then prints `Stop-Process -Id <pid>`
and stops. Pass `-Force` to have it try the kill itself; if that comes back
access-denied — which happens when the stale server was started from a shell
with different privileges — it prints the elevation instruction rather than a
raw exception.

### World Builder while the tower runs

The Tower attaches a World Builder follower to each capture itself. You do
not need a second terminal and you do not need to run
`scripts/world_build_session.py` by hand in the normal flow. That script
remains available as a fallback and as an offline diagnostic — for rebuilding
a world from an already-recorded capture — and when you do run it manually,
run it from the tower root for the redaction reason above.

## A physical World Builder session, start to finish

The whole flow, once `scripts\setup_tower.ps1` has been run on this
machine:

```powershell
powershell -NoProfile -File scripts\start_tower.ps1
```

Then, on the phone: open World Builder, press Start, walk, press Stop.

That is all of it. There is no second terminal, no capture directory to
inspect, and no UUID to copy. The Tower mints a capture id at
`stream_start` and attaches a follower to it in the same breath
(`tower/capture_workers.py`); the follower rebuilds every four keyframes
so the world grows during the walk; and when the capture closes the
follower finalises, persists, and exits, and the Tower reaps it.

### What you should see in the Tower's console

The follower's output is inherited, deliberately, so one terminal shows
the whole story:

```
[Tower][Capture] recording started: 6bf1c84c92f94fb68db62d5ba24c3ad2
[Tower][Worker]  started pid 49784 for capture 6bf1c84c... : ... --follow-capture ...
[Tower][WorldBuilder] session 29da45bf... in world b1abcdb8...: source=live-capture
                      capture=6bf1c84c... backend=auto intrinsics=unknown rebuild_every=4
[Tower][WorldBuilder] rebuild 1: 2 keyframes -> 0 positioned poses, 0 points, 1 segments
...
[Tower][Capture] recording stopped (stop): 24 frames, 6521938 bytes
[Tower][Worker]  capture 6bf1c84c... closed; worker pid 49784 continues until it
                 observes completion
[Tower][WorldBuilder] session ... finished: backend=unposed (downgraded_from=classical),
                      0 solved poses, 0 points, scale=unknown
[Tower][Worker]  worker pid 49784 finished after 0.8s
```

`GET /health` answers the same question remotely, which matters because
this Tower is normally operated from another machine:

```json
{"capture": {"armed": true, "recording": true, "capture_id": "6bf1c84c..."},
 "capture_workers": {"enabled": true, "workers": [{"capture_id": "6bf1c84c...", "pid": 49784}]}}
```

### Zero poses and zero points is the CORRECT result today

Until the camera is calibrated you should expect exactly this, and it is
not a fault:

```
calibration  uncalibrated      scale   unknown
poses        0                 points  0
```

No intrinsics exist for the Ray-Ban camera, so `BACKEND_AUTO` selects the
backend that withholds every pose rather than inventing a focal length,
and it now says so loudly in the log. Keyframes, tracking, segments and
the persisted world are all real. See `docs/CALIBRATION.md` for the
physical procedure that changes this.

### Troubleshooting

| Symptom | Where to look |
|---|---|
| iOS shows World Builder unsupported | `TOWER_WORLD_ROOT` unset. The startup banner warns about this |
| Frames arrive, nothing is recorded | `TOWER_CAPTURE_ROOT` unset. `/health` shows `"capture": null` |
| A capture exists, no world appears | `/health` → `capture_workers`. `enabled: false` means autobuild is off; an empty `workers` list during a walk means the follower died, and it logs its exit code and argv |
| Keyframes climb, geometry stays absent | Expected mid-walk before the first rebuild. If it never appears, check `TOWER_WORLD_REBUILD_EVERY` is not `0` |
| Poses and points are 0 | Uncalibrated. See above |
| The world stops growing while the camera is live | Check the capture manifests under `<capture root>/captures/`. A gap longer than 90 s between captures is a new walk by design, and gets its own world |
| The phone shows many fragments and none are connected | Read the `registration` block the follower prints when the walk ends, or re-run `scripts/world_registration.py --world <id>`. Every candidate pair is reported with the reason it was refused and how much evidence it had. `no keyframe of either segment matched` means retrieval — the walk never showed the same place twice well enough; anything else means the pairs were found and the estimate was not trusted |

### Re-measuring a walk without walking it again

`scripts/world_replay.py` rebuilds a recorded walk from its raw frames,
deterministically, so a registration or tracking change can be scored against
real physical input instead of another walk. It reproduces the two 2026-08-29
sessions figure for figure and says so in its own output:

```powershell
.venv\Scripts\python.exe scripts\world_replay.py --case worldB `
    --root C:\Users\<you>\Projects\Glasses-scratch\wb-replay --register
```

`--case worldA` is the plain room walk; `--case worldB` is the drawer walk with
deliberate lateral motion and a return. `--captures <id> <id> ...` replays any
other recorded walk, in time order. One walk is usually several captures — a
transport disconnect ends one and the reconnect begins the next, while the
builder keeps a single session across the gap.

## LAN Access

For the iPhone to reach the tower, it needs the Windows machine's LAN IP
address.

Find it with:

```powershell
ipconfig
```

Look for the `IPv4 Address` under your active network adapter (Wi-Fi or
Ethernet), e.g. `192.168.1.42`.

The iPhone app should then address the tower as:

- Health check: `http://192.168.1.42:8000/health`
- WebSocket: `ws://192.168.1.42:8000/ws`

Both devices must be on the same LAN for this to work. This service is
**LAN-only for this milestone** — it must not be exposed to the public
internet, and no authentication is implemented yet.

### Firewall

Windows Firewall will likely block inbound connections to the tower by
default. This project does **not** modify firewall rules automatically.

If the iPhone cannot reach the tower, add an inbound rule allowing TCP
traffic on the configured port (default `8000`) for the Private network
profile:

```powershell
New-NetFirewallRule -DisplayName "Glasses Tower Dev" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```

Run this yourself after reviewing it — do not run firewall changes as part
of an automated script without understanding what it opens.

## Testing the Health Endpoint

With the server running:

```powershell
curl http://127.0.0.1:8000/health
```

Expected output:

```json
{"status":"ok","service":"glasses-tower","version":"0.1.0"}
```

## Testing the WebSocket

`websockets` is already part of the `dev` extra, so
`scripts\setup_tower.ps1` has installed it. Run a small test script with the
venv interpreter:

```python
import asyncio
import json
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        await ws.send(json.dumps({"type": "ping"}))
        print(await ws.recv())

asyncio.run(main())
```

Expected output:

```text
{"type":"pong"}
```

## Testing a Camera Frame

Send a `frame` message with base64-encoded JPEG data and matching
`width`/`height`:

```python
import asyncio
import base64
import json
import websockets

async def main():
    with open("test.jpg", "rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")

    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        await ws.send(json.dumps({
            "type": "frame",
            "seq": 1,
            "width": 480,   # must match the actual JPEG dimensions
            "height": 640,
            "format": "jpeg",
            "data": payload,
        }))
        print(await ws.recv())

asyncio.run(main())
```

Expected output (values vary by image and by whichever experiment is
armed; `mean_intensity` is only present for the `baseline` experiment —
see `tower/routes/ws.py`). The `cv_lab` block names the run and the
experiment that produced the number, so a result can never be read as
belonging to a different experiment:

```text
{"type":"frame_result","seq":1,"processing_ms":4.1,"result_value":130.0,
 "result_label":"mean_intensity","stage_ms":{"total":4.1},"mean_intensity":130.0,
 "cv_lab":{"contract":"experimental_cv.frame_result/2026-08-27",
           "tower_instance_id":"2a5b04b1b77c","run_id":"2a5b04b1b77c-1",
           "result_seq":1,"experiment_id":"baseline","experiment_name":"Baseline",
           "provenance":"measured","backend":"opencv","device":null,
           "device_requested":"auto","result_label":"mean_intensity",
           "processing_ms":4.1,"tower_received_at":1787810180.83,
           "time_basis":"tower-receipt"}}
```

To see what this Tower can run, and what it is running:

```powershell
curl http://localhost:8000/cv-lab
```

To change experiment without restarting anything, on the same socket:

```json
{"type": "cv_lab_start", "experiment_id": "edge_detection"}
```

The tower's own log output during this should show:

```text
[Tower][Frame] #1 received: <N> bytes
[Tower][Frame] #1 decoded: 480x640
[Tower][Frame] #1 verified
[Tower][Frame] #1 processed: mean_intensity=130.00
```

## Stream Lifecycle Control Messages (V0.7)

The WebSocket connection is persistent — it stays open across the phone's
own "Start Camera Session" / "Stop Camera Session" actions, and normally
across many such cycles, so the tower needs an explicit signal for when a
streaming *measurement window* begins and ends, separate from the
connection's own lifetime. Two control messages exist for this:

```json
{"type": "stream_start"}
```
Send once, right before the app begins forwarding `frame` messages for a
streaming session.

```json
{"type": "stream_stop"}
```
Send once, when the camera session/stream stops — no more `frame`
messages should follow until the next `stream_start`.

Neither message gets a reply (unlike `ping`→`pong`, or `frame`→
`frame_result`/`frame_error`) — don't wait on a response for either.

### `frame_error` (Tower → app)

A `frame` message is answered by **either** `frame_result` **or**
`frame_error` — never both, never neither. Clients must handle both.

```json
{"type": "frame_error", "seq": 30, "reason": "frame_skipped",
 "message": "module experimental-cv could not process this frame"}
```

`reason` is one of:

| `reason` | Meaning | Module state after |
|---|---|---|
| `invalid_frame` | The message failed transport-level validation (missing field, bad base64, unsupported format, undecodable JPEG). The module was never invoked. | unchanged |
| `frame_skipped` | The module rejected this one frame but is still healthy and will accept the next. | still `active` |
| `module_unavailable` | The module is not `active` — it failed while processing this frame, or was already failed/unloaded. Subsequent frames will also fail until the Tower is restarted. | `failed` |

`seq` is `null` when the message failed validation before `seq` could be
read. `message` is a human-readable diagnostic — log it, don't parse it.

Behavior:
- Every `frame` message gets exactly one reply, `frame_result` or
  `frame_error`, regardless of whether a stream measurement window is
  currently open — `stream_start` is not required for frame processing.
- Frames received before any `stream_start` (or after a `stream_stop`)
  are processed and answered normally, but are not counted in any
  measurement window.
- A malformed message that isn't frame-shaped at all (invalid JSON, or a
  JSON value that isn't an object) is logged and ignored; the connection
  stays open and no reply is sent.
- `stream_stop` with no window open logs a warning and is otherwise a
  no-op — the connection stays fully usable.
- A `stream_start` received while a window is already open finalizes the
  existing window first (`end_reason: "superseded_by_stream_start"`),
  then opens a fresh one — no measurement data is silently dropped.
- Multiple `stream_start` → frames → `stream_stop` cycles work on one
  persistent connection.

## Measuring a Sustained Session (V0.7)

The tower logs a `[Tower][Session] summary: {...}` line periodically
within an open streaming window (every 150 frames by default) and a
`[Tower][Session] final summary: {...}` line whenever that window closes
— on `stream_stop`, on a `stream_start` that supersedes it, or on
disconnect while it's still open (see Stream Lifecycle Control Messages
above). **A summary is only produced for frames sent between a
`stream_start` and its corresponding close** — frames sent outside any
window are still processed and acknowledged, but don't contribute to any
summary. Each final summary includes an `end_reason` (`"stream_stop"`,
`"superseded_by_stream_start"`, or `"disconnect"`) alongside effective
FPS, bandwidth, `seq_gap_total`, Tower-side backpressure drops, Tower
processing latency, and process CPU/RSS for that window — never raw
frame data.

**`process_cpu_percent` is a session average, as percent of one core.** It
is cumulative CPU time over `session_duration_s`, so it is directly
comparable between a periodic summary and the final summary, and is not
clamped to 100 (a multi-core process legitimately exceeds it). It is
deliberately *not* `psutil.cpu_percent(interval=None)`, which measures
only since its own previous call — that made the final summary describe
the sliver of time since the last periodic summary rather than the
session, and could read `0.0` with a core pegged.

**Per-frame logging is at INFO, and `TOWER_DEV_MODE` does not control
it.** `TOWER_DEV_MODE` only switches the root level between DEBUG and
INFO, so setting it to `false` does **not** quiet the four
`[Tower][Frame]` lines emitted per frame. Measured 2026-08-21 at 360x640:
suppressing those lines raises the Tower's saturation ceiling from ~736 to
~1065 fps (about 45% of peak throughput, ~0.4 ms of receive-to-result) and
avoids ~24 MB of log per 30 s at saturation. They are kept on by default
anyway: at the 10–15 fps target the Tower uses ~2.3% of one core, and
these lines are the primary diagnostic surface for physical runs. Revisit
only if the Tower is ever actually throughput-bound.

**`bandwidth_bps` is BYTES per second, not bits.** It is computed as
`bytes_received / elapsed_s`. The name is kept for backward compatibility
with existing report templates and log consumers, but read it as B/s —
mistaking it for bits/s understates throughput by 8x.

**`seq_gap_total` is a raw, causally-neutral count — not a "frames lost"
figure.** The iOS sender currently assigns `seq` from the DAT/source
capture-frame index and only forwards roughly 1-in-30 of them by design
(a throttled capture -> transmit branch), so the tower normally receives
`seq` like 1, 30, 60, 90, ... under completely normal operation. Under
the current wire protocol there is no separate transmission-attempt
counter, so a gap in `seq` cannot currently be attributed to intentional
sender-side sampling, a sender-side drop, or genuine network/transit
loss — they look identical on the wire. Do not interpret `seq_gap_total`
as network loss. See `guidelines/docs/07-PLATFORM-CONSTRAINTS.md`
Limitation 9 for the future `source_seq`/`tx_seq` protocol split that
would be needed to actually distinguish these causes (not implemented).

**`source_fps_estimate` and `sampling_stride_avg` name the sender's
throttle directly**, so the situation above no longer has to be inferred
from a large `seq_gap_total`. Both are derived from `source_seq` (the
DAT/capture frame index, which falls back to `seq` for every sender that
exists today), alongside the raw `source_seq_span` they are computed from:

| Field | Meaning |
|---|---|
| `source_seq_span` | Capture-index range observed (last minus first). |
| `source_frame_span_s` | Seconds between the first and last frame received. |
| `sampling_stride_avg` | Capture frames elapsed per frame recorded — ~30 for the current sender, 1.0 for a sender forwarding everything. Equals `source_seq_span / (frames_received - 1)`. |
| `source_fps_estimate` | The upstream capture rate that stride is sampling. Equals `source_seq_span / source_frame_span_s`. |

Read together with `effective_fps`, these state the pipeline's behavior
outright: the 2026-08-21 first physical-glasses remote run would have
reported a `source_fps_estimate` of ~23.5 and a `sampling_stride_avg` of
~29.98 behind its `effective_fps` of 0.8 — a ~24 fps capture stream
decimated 1-in-30. See
`guidelines/docs/reports/2026-08-21-first-physical-glasses-remote-baseline.md`.

These are **estimates**, labeled as such. Two specific traps:

- **`source_fps_estimate` and `effective_fps` do not share a
  denominator.** The former divides by `source_frame_span_s` (first to
  last frame *received*); the latter divides by `session_duration_s` (the
  whole `stream_start`-bounded window). A burst of frames followed by a
  long silence yields a high `source_fps_estimate` and a low
  `effective_fps` with no contradiction. Both spans are reported so the
  arithmetic is checkable rather than taken on trust.
- **`sampling_stride_avg` is only trustworthy when `frames_rejected` is
  0.** Its denominator counts frames actually recorded, so intermittent
  Tower-side rejection inflates it — a sender forwarding *every* frame,
  with every other frame answered `invalid_frame`, reports a stride of
  ~2.0. That is a Tower-side loss masquerading as a sender throttle, and
  it cannot be corrected from inside the metric (a rejected frame has no
  trustworthy capture index), so check `frames_rejected` before believing
  the stride.

Each reports `null` rather than `0` when fewer than two frames have
arrived, or when the capture index did not advance (a sender restart can
make it regress); an unmeasurable rate is not a rate of zero.

**`frames_rejected`** counts every frame that arrived and was answered
with a `frame_error` — `invalid_frame`, `frame_skipped`, or
`module_unavailable` — and so contributed to no other figure in the
summary. It overlaps `frame_processing_errors` (which counts only the
`frame_skipped` subset) deliberately: the two answer different questions,
namely "how many frames did the module fail on" versus "how many arriving
frames are missing from these numbers".

For local validation without the iPhone, run the soak-test script against
a running tower:

```powershell
python scripts/soak_test_stream.py --source synthetic-script --duration-s 60
```

Use `--duration-s 1200` (20 minutes) to `1800` (30 minutes) for the full
roadmap target session length, and `--source mock-device-kit` /
`--source iphone-camera` / `--source physical-glasses` when the frames
actually come from those sources via the iOS app, so the resulting report
is never mislabeled. See `guidelines/docs/reports/V0.7-sustained-streaming-report.md`
for the report template — fill it in using this script's printed output
combined with the Tower's own `[Tower][Session] final summary` log line.

A `synthetic-script` or `mock-device-kit` run validates the Tower's
instrumentation and connection stability only; it does not establish
real-world V0.7 figures — see `07-PLATFORM-CONSTRAINTS.md` Limitation 12.

## Project Structure

```text
tower/
  world_builder/          World Builder V1 engine (see
                          guidelines/docs/reports/2026-08-22-world-builder-v1-report.md).
                          Calibration-gated: no intrinsics means no poses,
                          and it says so rather than guessing.
  cv_lab/                 The Experimental CV Lab as a product surface:
                          the catalog, one run's measurements, and the
                          lifecycle that selects an experiment at runtime.
                          See docs/contracts/EXPERIMENTAL-CV-LAB.md.
  main.py                 FastAPI app factory + ASGI entrypoint; builds
                          the one Lab module and arms the startup default
                          named by TOWER_CV_EXPERIMENT
  config.py               Environment-based settings (host/port/dev
                          mode/CV startup default/CV device)
  logging_config.py       Structured logging setup
  session.py              Minimal single-client connection tracking
  frames.py               Frame message validation/decoding
                          (transport-level)
  frame_processing.py     Minimal deterministic OpenCV operation
                          (grayscale + mean intensity); used by the
                          `baseline` experiment below
  metrics.py              Per-connection sustained-streaming
                          measurements
  instrumentation.py      StageTimer: per-stage timing shared by
                          module experiments
  experiments/
    __init__.py           Experiment protocol, ExperimentResult (with
                          the `metrics` measurement channel), and the
                          EXPERIMENTS registry of factories
    baseline.py           Grayscale + mean-intensity OpenCV experiment
    edge_detection.py     Canny edge-detection OpenCV experiment
    frame_quality.py      Sharpness, gradient energy, entropy, contrast,
                          edge density, exposure clipping
    feature_detection.py  ORB keypoint yield and spatial coverage
    optical_flow.py       Sparse LK flow: magnitude, coherence,
                          forward-backward error (holds one frame)
    redaction_impact.py   What blurring a region costs downstream
                          feature tracking
    object_detection.py   torchvision SSDLite320 COCO detection
                          (holds a model)
    depth.py              Stateful MiDaS-small monocular depth
                          experiment (holds a loaded model)
  modules/
    base.py               Module ABC, lifecycle states,
                          FrameProcessingError/FrameSkippedError/
                          ModuleUnavailableError
    container.py          ModuleContainer: lifecycle orchestration for
                          the one active module slot
    experimental_cv.py    The one Lab module slot. Hosts ANY registered
                          experiment, stateful or not -- there is no
                          longer a second Module class for depth
  document_memory/
    records.py            DocumentObservation / PageObservation. Named
                          "observed", never "read": the camera cannot
                          establish attention
    store.py              Append-only JSONL, retention window, real purge
                          that reports what it could not delete
    detect.py             Page-quad detection PLUS a text-likeness gate --
                          a laptop lid is page-shaped too
    dwell.py              Is this page held in view long enough to be
                          worth 1.2s of OCR?
    ocr.py                TextRecogniser seam: EasyOCR, plus a fast fake
                          so the default suite neither downloads nor waits
    engine.py             Cheap per frame, expensive per dwell
    retrieval.py          BM25 by content, window by time, and an explicit
                          refusal when there is no record
  object_memory/
    records.py            ObjectObservation: a CATEGORY was visible at a
                          time, never "my keys" and never a position
    relevance.py          Which detections are worth remembering.
                          PERSISTED_CLASSES is a closed whitelist --
                          laptop and cell phone, the only two classes the
                          real corpus supports above 0.8. `person` is
                          excluded and there is no flag to re-admit it
    detector.py           Detector seam: torchvision SSDLite320 + a fast
                          fake. Does NOT import the Lab
    engine.py             Detect, filter, persist -- and count everything
                          it declined, and why
    store.py              Append-only JSONL, real purge, and a retention
                          cutoff applied on READ as well as on prune
  scene/
    records.py            Detection / Track / Relation / FacingEstimate.
                          A track_id is "the same blob one frame later",
                          never an identity
    detect.py             Detector seam: torchvision SSDLite320 + a fast
                          fake. Does NOT import the Lab
    tracking.py           IoU-only association. Counts come from
                          CONFIRMED TRACKS, never from detections
    orientation.py        Coarse facing from keypoint VISIBILITY. Off by
                          default: 798ms per call. Never gaze
    state.py              The live scene, and the relationships it
                          refuses to assert, with reasons
    engine.py             Frames in, a scene out. Stores nothing
    query.py              The brief's questions, and honest refusals
  routes/
    health.py             GET /health
    ws.py                 WebSocket /ws (ping/pong, frame receive +
                          module dispatch, frame_error reporting)
scripts/
  soak_test_stream.py     Local sustained-load soak-test client (V0.7)
  verify_cuda.py          One-shot PyTorch/CUDA verification spike
                          (V0.9.1)
  depth_benchmark.py      CPU vs GPU depth-experiment benchmark client
                          (V0.9.1)
  depth_temporal_consistency.py
                          World Builder Experiment 1: offline
                          frame-to-frame depth-flicker analysis and
                          EMA/median smoothing comparison (V0.9.3)
  feature_trackability.py World Builder Experiment 2: offline ORB
                          keypoint/match/RANSAC-inlier analysis across
                          frame gaps; intrinsics-free (V0.9.3)
  world_build_session.py  Drive a World Builder mapping session over
                          frames on disk (or a synthetic walk), then build
                          and persist the world
  world_inspect.py        Reload a saved world cold and report it;
                          --trajectory for the recorded camera path,
                          --verify for journal/image integrity
  calibrate_charuco.py    Recover camera intrinsics from ChArUco board
                          views; --generate-board writes a printable board
  cv_lab_benchmark.py     Every Experimental CV Lab experiment across
                          three resolutions, plus the sparse-vs-dense
                          optical-flow comparison
  document_memory_session.py
                          Observe documents in a frame stream (a capture,
                          live or recorded, or a synthetic one). Runs in a
                          separate process: OCR costs ~1.2s per page
  document_query.py       Ask Document Memory what it observed --
                          --recent, --minutes-ago, --text, --coverage,
                          --purge. Independent of any voice path
  document_memory_benchmark.py
                          Detection cost per frame, retrieval latency and
                          storage growth, plus read quality swept over
                          frame size and tilt
  object_memory_session.py
                          Remember which object categories were visible,
                          from a capture (live or recorded). Runs in a
                          separate process, so no module lifecycle has to
                          bound a model load
  object_query.py         Ask Object Memory when it last saw something.
                          "Where" is a FRAME reference, not a place --
                          this slice stores no spatial position
  scene_session.py        What is around the wearer, from a frame stream.
                          Answers the questions in the run that observed
                          the frames -- nothing is persisted, so there is
                          deliberately no separate query script
  scene_benchmark.py      Per-frame cost with and without orientation,
                          and count stability under detector dropout
  world_builder_benchmark.py
                          Stage timings for the World Builder pipeline
                          (synthetic input; timings real, imagery rendered)
  world_builder_env_check.py
                          Read-only World Builder readiness diagnostic:
                          GPU visibility vs. torch's ability to reach it,
                          OpenCV geometry/calibration coverage, optional
                          library inventory. Installs nothing, writes
                          nothing, exits 0 unless --strict
tests/
  test_health.py
  test_config.py
  test_ws.py
  test_ws_frames.py
  test_ws_disconnect_race.py
  test_ws_stream_lifecycle.py
  test_ws_sustained.py
  test_ws_module_unavailable.py
  test_ws_frame_skipped.py
  test_ws_experiment_fields.py
  test_frame_processing.py
  test_metrics.py
  test_instrumentation.py
  test_experiment_result.py
  test_experiments_registry.py
  test_experiments_baseline.py
  test_experiments_edge_detection.py
  test_experiments_depth.py
  test_module_base.py
  test_module_container.py
  test_module_container_wiring.py
  test_main_module_factory.py
  test_experimental_cv_module.py
  test_experimental_cv_stateful.py      the Lab module hosting a stateful
                                        experiment
  test_experiments_measure_truth.py     each measurement checked against
                                        independent truth
  test_ws_metrics_channel.py            the additive `metrics` wire field
  test_cv_lab_benchmark_cli.py
  test_depth_experiment_integration.py  (opt-in, real model — see below)
  test_object_detection_integration.py  (opt-in, real model)
  test_soak_script_cli.py
  test_depth_benchmark_cli.py
  test_ws_malformed_message.py
  test_frames_seq_split.py
  test_metrics_upstream_rate.py
  test_metrics_rejected_frames.py
  test_metrics_cpu_average.py
  test_ws_upstream_rate.py
  test_ws_finalize_robustness.py
  test_frames_field_types.py
  test_world_builder_experiment_clis.py
```

The module system (`tower/modules/`) owns the module lifecycle
(UNLOADED -> LOADING -> READY -> ACTIVE -> STOPPING/FAILED) and dispatches
each decoded frame to the one Lab module. WHICH experiment that module is
holding is `tower/cv_lab/`'s business and changes at runtime: the module
is the slot, the Lab is what is in it, and the container still holds
exactly one module constructed once with no discovery and no swap path. `frame_processing.py` no longer contains the only
OpenCV usage in the codebase — `tower/experiments/edge_detection.py` and
`tower/experiments/depth.py` also call into OpenCV (the latter only for
JPEG decode/color conversion ahead of model inference); `frame_processing.py`
remains the one OpenCV call site the `baseline` experiment itself uses.
`frames.py`/`routes/ws.py` remain transport infrastructure — frame
transport/decoding is fully decoupled from whichever module/experiment
processes the resulting pixels.
