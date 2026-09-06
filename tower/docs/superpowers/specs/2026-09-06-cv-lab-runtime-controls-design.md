# CV Lab runtime controls: design

**Date:** 2026-09-06
**Lane:** `feature/cv-lab-runtime-controls`, worktree
`C:\Users\tvllo\Projects\Glasses-worktrees\cv-lab-runtime`
**Reconciles:** `tower/docs/contracts/EXPERIMENTAL-CV-LAB.md` (the wire
contract this design extends additively), `2026-08-19-v0.9-experimental-cv-lab-design.md`
(the Lab's original shape), `docs/agent-handoffs/CV-LAB-LIVE-VIEW-MAC-HANDOFF.md`
(the last Mac handoff, whose iOS files this touches).

## Context

The mission brief describes a developer workflow -- set
`TOWER_CV_EXPERIMENT`, restart Tower, reconnect, restart the camera -- and
asks for a runtime-controlled product experience. Reading the code first
changed the shape of the work:

- **Runtime switching already exists.** Since 2026-08-27 a client selects
  an experiment with the `cv_lab_start` socket message; the Tower releases
  the old experiment and arms the new one without a restart
  (`tower/cv_lab/lab.py:685`). iOS already renders every experiment as a
  tappable row that sends that message
  (`ExperimentalCVWorkspaceView.swift:527`). The environment variable is
  only the startup default (`tower/main.py:62-91`).
- **What still forces a restart** is not selection. It is that two
  failure paths are *terminal for the life of the process*: an experiment
  that raises anything other than `FrameProcessingError` mid-frame
  (`ModuleContainer.process`, `tower/modules/container.py:161-177`), and a
  startup default that cannot load (`CVLab.load_initial`,
  `lab.py:298-358`). Both call `mark_failed()`, from which there is no
  path back. With object detection driving the machine to 99% CPU, a
  single frame exception is plausible, and after it every experiment is
  dead until Tower restarts.
- **What still forces a trip to Home** is that the CV Lab workspace is not
  handed the camera service (`ContentView.swift:150-197`), so Start/Stop
  live only on Home and World Builder. There is no in-cartridge Tower
  connection row either; only the shell status bar shows it.
- **The resource picture** measured on this host (RTX 5070, 20 logical
  CPUs, torch 2.13.0+cu132) is in the table below. The Lab itself owns no
  child process. The `python.exe` processes that survive a Tower shutdown
  are World Builder followers spawned by the *shared capture supervisor*
  on every `stream_start` when `TOWER_CAPTURE_ROOT`, `TOWER_WORLD_ROOT`
  and the default `TOWER_WORLD_AUTOBUILD=true` are all in force -- which
  is the operator's `.env`. See §8.

| Measurement (this host, 2026-09-06) | Value |
|---|---|
| Warm arm, `depth`, CUDA, hub cache warm | 220-260 ms |
| Cold arm, `depth`, first load in process | 2,261 ms |
| Warm arm, `object_detection`, CUDA | 125-155 ms |
| Arm, every other experiment | < 1 ms |
| Release, any experiment | < 4 ms |
| `object_detection` CUDA, torch threads 20 / 4 / 2 / 1 | 12.6 / 3.6 / 1.8 / 0.9 cores, p50 43.6 / 40.0 / 40.1 / 40.0 ms |
| `depth` CUDA, torch threads 20 / 2 / 1 | 6.2 / 1.1 / 0.9 cores, p50 11.8 / 11.7 / 16.3 ms |
| `object_detection` CPU, torch threads 20 / 4 | 13.1 / 3.8 cores, p50 46.6 / 35.9 ms |
| `depth` CPU, torch threads 20 / 8 / 4 | 12.5 / 7.7 / 3.9 cores, p50 42.3 / 34.1 / 48.2 ms |
| Tower process RSS with CUDA torch resident | ~1.6-1.75 GB, flat across 3 switch cycles |

Two conclusions follow. A model cache is not worth its memory: a quarter
second is the whole cost of a cold switch to a heavy experiment. And the
CPU figure the user saw is mostly OpenMP: torch's intra-op pool defaults
to 20 threads that spin-wait between kernel launches, and on CUDA the cap
of 2 costs nothing in latency while returning ten cores.

## Non-goals

- No new "select without arm" message. The contract argues against it
  (EXPERIMENTAL-CV-LAB.md §"cv_lab_start") and nothing here needs it.
- No warm swap (loading the new model while the old one still answers).
  Measured downtime is 0.25 s; the deterministic order *release, then
  arm* is worth more than that quarter second, and it keeps peak memory
  at one model.
- No HTTP control surface. iOS uses the socket; the soak harness uses the
  socket. The reason the contract gives (an arm can take two minutes and a
  request/response route would have to block or lie) still holds.
- No change to World Builder, Object Memory, Scene or Document Memory
  code, and no change to the shared capture supervisor. §8 records what
  was found there and what the operator can do with an environment
  variable today.
- No remote dead-start of Tower.
- No move of inference off the event loop. It is a real limitation (a
  40 ms object-detection frame blocks every socket for 40 ms) but it is
  not what breaks the target workflow, and doing it right means carrying
  the provenance handshake across an await in `ws.py`, which is shared
  transport. Recorded as a known limit.

## Chosen approach

### Tower

**T1. Mid-run crash containment.** `CVLab.process()` catches any
non-`FrameProcessingError` exception from `experiment.run()`, moves the run
to `failed` with a client-safe reason, drops and releases the experiment,
and re-raises as `FrameProcessingError(reason="cv_lab_failed")`. The
container therefore sees a *skipped frame*, the module stays `ACTIVE`,
and the next `cv_lab_start` arms a fresh instance. `failed` was already a
recoverable state on the wire for arm failures; this makes it the state
for run failures too, which the contract's own wording ("failed is
recoverable") already implies. Logged with `logger.exception` so a real
bug stays loud.

**T2. Startup default failure is loud but not terminal.** `load_initial()`
catches `Exception` (never `CancelledError`, so the container's load
timeout keeps its abandon semantics) from both the registry lookup and
the load, logs at ERROR naming the variable, and leaves the Lab in
`failed` with the reason. The module reaches `ACTIVE`; clients see
`failed` in the status document and may start any experiment. A Tower
with a typo in `TOWER_CV_EXPERIMENT` is now a Tower whose default did not
arm, not a dead Tower.

**T3. One loader thread per Lab, not one per arm.** Every arm today runs
`load()` on a fresh daemon thread (`tower/loading.py:101`). A thread that
runs a torch model load leaves torch's intra-op team behind when it
exits, which the Scene lane already measured at +19 OS threads and ~8 MB
per master thread (`tower/live_session.py:106-143`). The Lab gets a small
`ExperimentLoader` (`tower/cv_lab/loader.py`): one long-lived daemon
worker, a queue of jobs, `run(fn, *args)` returning an awaitable, and
*retire on abandon*: when an arm times out or is cancelled the current
worker is marked retired and a fresh one is started, so an abandoned load
can still finish and discard its model (the `LoadInvalidation` latch is
unchanged) without blocking the next arm. `run_abandonable` stays for the
container. Verified by a psutil thread-count test over repeated
object-detection/depth arms (skipped without torch; marked `slow`).

*Decision gate, resolved 2026-09-06:* measured directly on this host with
six consecutive arms of each heavy experiment, `load()` on a fresh thread
per arm grows the process by exactly **19 OS threads and ~8-9 MB RSS per
arm** (object_detection 48→143 threads, depth 163→258, linear); with the
intra-op cap at 2 it still grows by 1 thread per arm; on one persistent
loader thread it is **flat at ±0 threads** over the same twelve arms.
T3 is implemented.

**T4. Torch thread cap, per arm.** New `Settings.cv_torch_threads` from
`TOWER_CV_TORCH_THREADS` (default `auto`), carried as
`ExperimentSettings.torch_threads`. `resolve_torch_threads(device,
requested)` in `tower/experiments/depth.py` beside `resolve_device`:
`auto` → 2 on CUDA, 4 on CPU; an explicit positive integer is honoured;
`0` leaves torch's default alone. Applied in `DepthEstimation.load()` and
`ObjectDetectionExperiment.load()` after the device is resolved, and
reported by `describe()` as `torch_threads`. The setting is process-global
(so is torch's), which the config comment says plainly; Scene's own cap
(`TOWER_SCENE_TORCH_THREADS`) is untouched and, if both are set, the last
load wins -- also stated. Defaults come from the table above.

**T5. Additive status fields.**
- `run.arm_ms`: milliseconds from `start` accepted to `running`, `null`
  until then. It is the number tomorrow's checklist asks for.
- `run.runtime.torch_threads`: from `describe()`.
- `process`: `{pid, threads, rss_mb}` from psutil, built inside
  `status()`. It is what lets the live soak and a phone read the Tower's
  own resource state without a shell on the Windows box.
All three are additive; no contract identifier changes. The contract
document gains the fields and a changelog entry; the protocol test that
reads the document keeps passing because no new message type, state or
reason is introduced (`cv_lab_failed` already exists as a frame-refusal
reason).

**T6. Soak harness** `tower/scripts/cv_lab_switch_soak.py`.
- `--in-process` (default): builds a real `CVLab` with the real registry
  and a real event loop, walks the eight experiments for `--cycles`
  rounds, feeds `--frames` textured JPEGs per arm, and records per
  transition: arm ms, release ms, frames processed/refused, psutil
  threads, RSS, handle count, child-process count, CUDA allocated and
  reserved. Prints a table and a verdict: for threads, RSS and CUDA
  reserved it compares the mean of the last third of cycles against the
  first third and flags growth above a tolerance.
- `--live --host H --port P [--tower-pid PID]`: the same walk against a
  running Tower over `/ws`, reusing `cv_lab_smoke.py`'s message shapes;
  threads and RSS come from the new `process` block; with `--tower-pid`
  it also counts the uvicorn process's children via psutil, which is how
  a World Builder follower would show up.
- `--json PATH` writes the report; without it nothing is written to disk.
  No `--root` flag, so the artifact-root guard does not apply; the
  default writes nothing.

**T7. Documentation.** Contract doc (fields, semantics of `failed` after a
run crash, changelog). README (`TOWER_CV_TORCH_THREADS`, the soak
command). A findings report under `tower/docs/superpowers/reports/`
(process ownership, the follower spawn path, the venv launcher pair, the
measurements). A Mac/tomorrow handoff under `docs/agent-handoffs/`.

### iOS

**I1. Pending-command state.** `TowerExperimentalCVClient` records
`pendingCommand: CVLabPendingCommand?` (`command`, `experimentID?`,
`requestID`, `sentAt`) when it sends a command, and clears it when a
`cv_lab_status` or `cv_lab_error` event arrives whose `requestID` matches,
when the connection leaves `.online`, or when a 10 s reply bound passes
(which then surfaces as a request failure). The protocol gains the
property with a default of `nil` so the unavailable client and test fakes
keep compiling; the view model republishes it. The experiment row for the
pending experiment shows a spinner and the caption "Asking the Tower…";
other rows and the run controls are disabled while pending. The Tower's
own `starting` remains a distinct phase drawn from the status document,
so the user sees two truthful phases: *asked* and *loading*.

**I2. Camera controls inside CV Lab.** `ContentView` passes
`project.glassesConnection` into the workspace. A `CVCameraCard`
(DEBUG-gated exactly like Home, because the capture surface is
`#if DEBUG` in the model) offers Start / Pause / Resume / Stop:
- Start and Stop call `GlassesConnection.startCameraSession()` /
  `stopCameraSession()`, the same service Home and World Builder use.
- Pause and Resume are a **frame-sending gate on `TowerClient`**
  (`isFrameSendingPaused`, `pauseFrameSending()`, `resumeFrameSending()`).
  DAT offers no app-initiated camera pause, and sending `stream_stop`
  would end the capture lineage on the Tower (and, in the operator's
  configuration, spawn a new follower on the next start). The gate holds
  frames on the phone: the glasses camera and the socket stay up, the
  Tower keeps the experiment armed, and its `source.receiving_frames`
  turns false after 5 s -- which is what the Lab's LIVE indicator already
  reads. The gate is cleared on `disconnect()` and on camera stop, so it
  cannot strand a later session.
- Release builds show the existing "Capture is not available in this
  build." helper text.
The Lab's own Pause / Resume / Stop (Tower-side, model stays loaded) stay
where they are, labelled as experiment controls.

**I3. Tower connection row.** A leaf `TowerConnectionRow` (observes
`TowerClient` only, so the rest of the workspace does not re-render at the
Tower's reply rate) shows Disconnected / Connecting / Connected / Error
with the shell's colour mapping, the failure detail, the compiled-in
endpoint, and one action mirroring `ConnectionSheet`: Connect (offline or
failed), Cancel (connecting), Disconnect (online). The shared
`StateDisplay.tower` labels are left alone; the row carries its own
four-word mapping so `testTowerStatusLabels` keeps pinning the shell.

**I4. Tests.** Appended to existing test files (the test target is a
classic `PBXGroup`, so a new file would need a pbxproj edit):
`CVLabContractTests.swift` for the pending-state model and its clearing
rules; `TowerClientTests.swift` for the frame gate (paused: no `frame`
message reaches the mock server; resumed: frames flow; `disconnect()`
clears the gate). All iOS code is authored on Windows and statically
checked with `ios/scripts/swift-structure-check.py`; it is **not
compiled** here and the handoff says so.

## Error handling

| Situation | Before | After |
|---|---|---|
| Experiment raises on a frame | module FAILED, restart required | run `failed`, experiment released, frame refused `cv_lab_failed`, Lab usable |
| `TOWER_CV_EXPERIMENT` unknown or its load fails | module FAILED at boot | Lab `failed` with reason, module ACTIVE, any `cv_lab_start` works |
| Arm times out (120 s) | run `failed`, loader thread abandoned | same; loader worker retired and replaced, abandoned load discards its model |
| Stop during arm | wins, installs nothing | unchanged |
| Second start during arm | `lab_busy` | unchanged |
| iOS command gets no reply in 10 s | row looks untouched forever | pending cleared, `lastRequestFailure` set, row re-enabled |
| Socket drops while a command is pending | pending never cleared | cleared on leaving `.online` |
| Camera paused, then app disconnects | n/a | gate cleared; a fresh connect streams again |

## Cleanup semantics (the contract this design commits to)

```
switch requested (cv_lab_start)
  → refuse if malformed / unknown / missing extra / lab_busy
  → _switching = True; new run minted
  → OLD experiment dropped and release()d          (before the new run id is visible)
  → state = starting
  → loader worker runs new_experiment.load()       (one persistent worker; retired on abandon)
  → install under _guard iff still this run        (else release quietly)
  → state = running; arm_ms recorded
frame raises non-FrameProcessingError
  → run.record_failed; state = failed (reason); experiment dropped and release()d
  → FrameProcessingError(cv_lab_failed) to the container; module stays ACTIVE
```

Nothing in the CV Lab spawns a process. The only threads it owns are the
loader worker (one, reused) and, transiently, a retired worker finishing
an abandoned load.

## Testing strategy

- Unit (fast, no torch): T1 crash containment; T2 startup failure leaves
  the module ACTIVE and a later start works; T5 fields present and
  numerically sane; loader retire-on-abandon with a blocking fake;
  repeated A→B→C→A switching with fakes keeps exactly one experiment
  alive (weakrefs) and one loader worker.
- Property (torch, `slow`): 8 object_detection↔depth arms keep
  `psutil.Process().num_threads()` flat after the first cycle;
  `resolve_torch_threads` defaults.
- Protocol: the contract document names the new fields; the status
  document is identical on all three surfaces with them.
- Soak: `cv_lab_switch_soak.py --cycles 6` in-process (48 transitions,
  real models) run here, with the report attached to the handoff. The
  live mode is exercised against a local uvicorn Tower here too.
- iOS: appended XCTest cases as above; structure check only.

## Files

Tower: `tower/tower/cv_lab/lab.py`, new `tower/tower/cv_lab/loader.py`,
`tower/tower/experiments/__init__.py` (settings field),
`tower/tower/experiments/depth.py`, `tower/tower/experiments/object_detection.py`,
`tower/tower/config.py`, `tower/tower/main.py` (settings → ExperimentSettings),
new `tower/scripts/cv_lab_switch_soak.py`, tests under `tower/tests/`,
`tower/docs/contracts/EXPERIMENTAL-CV-LAB.md`, `tower/README.md`,
new report and handoff documents.

iOS: `ios/Glasses/ContentView.swift`, `ios/Glasses/TowerClient.swift`
(frame gate), `ios/Glasses/Workspaces/ExperimentalCV/ExperimentalCVClient.swift`,
`ExperimentalCVModel.swift` (pending type), `ExperimentalCVWorkspaceView.swift`,
new `ios/Glasses/Views/Components/TowerConnectionRow.swift`,
new `ios/Glasses/Workspaces/ExperimentalCV/CVCameraCard.swift`,
`ios/GlassesTests/CVLabContractTests.swift`, `ios/GlassesTests/TowerClientTests.swift`.

## §8. What the process investigation found (summary; full report separate)

1. The CV Lab owns no child processes. Its frame path is synchronous on
   the event loop; its only thread is the loader.
2. The shutdown warning "worker pid … did not exit within 10.0s … anything
   it had not finished writing is lost" is `tower/capture_workers.py:909`
   and refers to a World Builder follower (`scripts/world_build_session.py`),
   which the supervisor never asks to stop (its spec has no stdin stop
   channel) and which normally exits ~20 s after a capture closes. On
   every clean shutdown mid-walk the grace is spent waiting and the child
   is terminated: expected, not a leak.
3. Followers live in their own process group with no Job object, so any
   hard kill of uvicorn (second Ctrl-C, `Stop-Process`, closing the
   terminal) orphans them; they then poll for up to 900 s. That is the
   accumulation the user saw, multiplied by two because a venv
   `python.exe` on Windows is a launcher that spawns the real
   `Python312\python.exe` as its child, so every worker is two rows in
   Task Manager.
4. A follower is spawned on every `stream_start` that opens a new capture
   lineage. In CV Lab use that is every camera Start more than 90 s after
   the previous stop. For CV Lab sessions the operator can set
   `TOWER_WORLD_AUTOBUILD=false` (captures still record; nothing builds),
   which removes the followers entirely. That is an operator decision and
   is documented, not changed.
5. The Tower-owned CPU during object detection was OpenMP spin (T4 fixes
   it); the RAM was the CUDA-resident Tower (~1.7 GB) plus followers.
