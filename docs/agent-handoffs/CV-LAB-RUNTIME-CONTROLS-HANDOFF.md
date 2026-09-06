# CV Lab runtime controls — handoff for the live validation session

**Written:** 2026-09-06, on Windows, by the CV Lab lane.
**Branch:** `feature/cv-lab-runtime-controls` in worktree
`C:\Users\tvllo\Projects\Glasses-worktrees\cv-lab-runtime`, based on
`integration/all-current-v1` (a7b1c2a). Oldest lane commit `75d09b0`
(the spec); see `git log --oneline 75d09b0^..HEAD`.
**Design:** `tower/docs/superpowers/specs/2026-09-06-cv-lab-runtime-controls-design.md`.
**Findings:** `tower/docs/superpowers/reports/2026-09-06-cv-lab-process-lifecycle.md`.
**Contract:** `tower/docs/contracts/EXPERIMENTAL-CV-LAB.md` (§13 changelog, 2026-09-06).

Nothing on the World Builder branches or worktrees was touched. The four
`python.exe` processes alive at the start of this lane belong to the World
Builder lane's replay and were left alone.

## 0. The one-paragraph version

Runtime experiment switching already existed (`cv_lab_start`, since
2026-08-27); what still forced a Tower restart was two failure paths that
were terminal for the life of the process, and what forced a trip to
Home was that the CV Lab screen had no camera or connection controls.
This lane makes a frame crash and a failed startup default recoverable,
runs every model load on one reusable thread (the previous fresh thread
per arm leaked 19 OS threads a switch), caps torch's intra-op threads
(object detection on CUDA: 12.6 cores → 1.8 at identical latency), adds
`arm_ms` and a `process` block for diagnosis, ships a switch soak, and
gives the iOS CV Lab a Tower connection row, a camera card with
Start / Pause / Resume / Stop, and truthful "asked the Tower" pending state
on the experiment rows. The leftover Python processes the user saw are
World Builder followers spawned by the shared capture supervisor on every
camera start, not the Lab; `TOWER_WORLD_AUTOBUILD=false` removes them for
CV Lab sessions.

## 1. What is VERIFIED ON WINDOWS

| Claim | Evidence |
|---|---|
| A frame crash fails the run, releases the experiment, keeps the module ACTIVE, and the next start works | `tests/test_cv_lab_runtime_controls.py` (4 tests), incl. via `ModuleContainer` |
| A startup default that is unknown or fails to load is logged at ERROR, leaves the Lab `failed`, module ACTIVE, and any start recovers | same file (3 tests, one over HTTP+socket) |
| Every arm runs on one loader thread; abandoned arms retire it; it exits on release or GC | `tests/test_cv_lab_loader.py` (7), runtime-controls (3) |
| Eight real `object_detection`/`depth` arms through a real `CVLab` keep the OS thread count flat (±2) | `tests/test_cv_lab_torch_threads.py::test_repeated_heavy_arms_do_not_grow_the_thread_count` (slow, real torch, CUDA) |
| The torch thread budget resolves per device, is applied on the loader AND the inference thread, and is reported in `describe()` | `tests/test_cv_lab_torch_threads.py` (11) |
| 48-transition in-process soak (6 cycles × 8 experiments × 15 frames): threads 98 flat, RSS flat, handles flat, CUDA 0/0 after every release, all 48 arms `running`, 720/720 frames processed, exit 0 | `scripts/cv_lab_switch_soak.py --cycles 6 --frames 15`; same run against the previous `lab.py`: threads 115 → 344, exit 1 |
| `arm_ms` and `process` on the wire; contract document names every payload key | `tests/test_cv_lab_protocol.py::test_every_payload_key_is_documented` and siblings |
| CV Lab test subset (23 files) | 412 passed |
| Full Tower suite from the worktree | **2532 passed, 75 skipped, 1 xfailed, 0 failed** in 7 min 33 s (`python -m pytest -q --basetemp=...\gf-cvlab`) |
| Live soak against a uvicorn Tower over `/ws` (port 8017, no `.env`, 4 cycles × 8 + warm-up, 15 frames per arm, `--tower-pid`) | 40/40 arms `running`, 600/600 frames processed, **threads 62 flat, RSS +7 MB, handles flat, children 0**, `arm_ms` 255-265 ms for `depth` and `object_detection` (warm), exit 0; the Tower was one process (plus its venv launcher) throughout and left nothing behind when stopped |

Measured numbers (this host: 20 logical CPUs, RTX 5070, torch 2.13.0+cu132):

| | |
|---|---|
| Warm arm `depth` / `object_detection` (CUDA) | 220-300 ms / 125-160 ms |
| First arm in a process `depth` / `object_detection` | 2.3-3.1 s / 0.3 s |
| Arm, any cheap experiment; release, any | < 1 ms; < 4 ms |
| `object_detection` CUDA cores at torch threads 20 / 2 | 12.6 / 1.8 (p50 43.6 / 40.1 ms) |
| Tower RSS with CUDA torch resident | 1.6-1.75 GB, flat across cycles |

## 2. What REQUIRES MAC / XCODE (uncompiled Swift)

Everything under `ios/` in this lane was authored on Windows and checked
only with `ios/scripts/swift-structure-check.py` (bracket/string balance)
and a static review. **Nothing was compiled or run.** Files:

- new `ios/Glasses/Workspaces/ExperimentalCV/CVCameraCard.swift`
- new `ios/Glasses/Views/Components/TowerConnectionRow.swift`
- `ios/Glasses/Workspaces/ExperimentalCV/ExperimentalCVClient.swift`
  (pending-command state, 10 s reply bound, `commandFailures`)
- `ios/Glasses/Workspaces/ExperimentalCV/ExperimentalCVModel.swift`
  (`CVLabPendingCommand`)
- `ios/Glasses/Workspaces/ExperimentalCV/ExperimentalCVWorkspaceView.swift`
  (row spinner + disabled rows while pending; the two new cards; `glasses`)
- `ios/Glasses/TowerClient.swift` (`isFrameSendingPaused`,
  `pauseFrameSending()`, `resumeFrameSending()`; `sendFrame` holds frames
  while paused; reset on `disconnect()` and `sendStreamStop()`)
- `ios/Glasses/ContentView.swift` (passes `glasses`), `ios/Glasses/SenderMetrics.swift` (one doc line)
- tests appended to `ios/GlassesTests/CVLabContractTests.swift` and
  `ios/GlassesTests/TowerClientTests.swift` (no new test files, so no
  `.pbxproj` change; the app target is a synchronized group)

Static-review outcome (an independent adversarial pass over every changed
Swift file against the real symbols in the codebase): **no certain
compile errors found**; two possible warnings (`nonisolated static let`
as an init default on a `@MainActor` class; `URL.host` deprecation, since
replaced with `host(percentEncoded:)`); two behavioural bugs, both fixed
before commit: the frame gate was reset only when a `stream_stop` actually
left, so a camera stopped during a socket drop kept the hold into the
next session (now cleared before the bracket guard, with a test); and the
run header blamed the glasses during a phone-side hold (now names the
hold). One wording fix for offline advice while the camera is running.

Specific compile risks the Mac should expect (from this repo's history
and this lane's review): `import Combine`/`Foundation`/`MWDATCamera`/
`MWDATCore` visibility under member-import visibility; `nonisolated`
value types vs the app target's MainActor default; the
`nonisolated static let replyBoundSeconds` used as an initializer default
on a `@MainActor` class; `@unknown default` arms on
`MWDATCamera.StreamState`; `Just<CVLabPendingCommand?>(nil)` /
`Empty<CartridgeFailure, Never>()` defaults in a protocol extension;
SwiftUI builder validity of the new `if let` / `ProgressView` arms.

Mac steps, in order: `python ios/scripts/contract-drift-check.py` against
the running Tower; `xcodebuild` per `docs/agent-handoffs/MAC-BUILD-VERIFICATION.md`;
run `CVLabContractTests` and `TowerClientTests`; fix what does not
compile and commit on this branch with `fix(ios):`.

## 3. What REQUIRES THE PHYSICAL SESSION (Tower + iPhone + glasses)

Set-up, once:

```powershell
# on the Windows box, from the worktree's tower/ (or the canonical checkout after merge)
cd C:\Users\tvllo\Projects\Glasses-worktrees\cv-lab-runtime\tower
$py = "C:\Users\tvllo\Projects\Glasses\tower\.venv\Scripts\python.exe"
# .env: keep TOWER_CAPTURE_ROOT / TOWER_WORLD_ROOT if you want captures; ADD
#   TOWER_WORLD_AUTOBUILD=false      # no World Builder follower per camera start
# and leave TOWER_CV_TORCH_THREADS unset (auto = 2 on CUDA).
& $py -m uvicorn tower.main:app --host 0.0.0.0 --port 8000 --env-file .env
# in a second terminal, note the pid:
Get-CimInstance Win32_Process -Filter "Name like 'python%'" | ? { $_.CommandLine -match 'tower\.main:app' } | select ProcessId, ParentProcessId
```

The log line at boot should read
`CV Lab startup default is 'baseline' on device 'auto' with torch threads 'auto'`.

Checklist. Each step has a pass criterion; record the reading.

1. **Start Tower once.** It is not restarted again in this list.
2. **Open CV Lab** on the phone. The new Tower row at the top shows
   *Connecting…* then *Connected*; the pill is green. Pass: no trip to
   the shell status bar needed to know the state.
3. **Connect / reconnect.** Tap *Disconnect* on the row, then *Connect*.
   Pass: the row goes Disconnected → Connecting → Connected; the
   experiment list re-appears (the client re-subscribes on `.online`).
4. **Start the camera from inside CV Lab.** Camera card → *Start*. Pass:
   the card reads *Streaming*; the run header shows LIVE; on the Tower,
   `(Invoke-RestMethod http://<tower>:8000/cv-lab).status.source.receiving_frames`
   is `true`. You did not visit Home.
5. **Select Edge Detection.** Pass: the row shows the spinner and
   "Asking the Tower…" for well under a second, then the run header
   names `edge_detection` and frames flow. `status.run.arm_ms` under 5.
6. **Tap Optical Flow, then Depth, then Feature Detection.** Pass for
   each: no Tower restart, camera card still *Streaming*, the socket did
   not drop (the Tower row never left *Connected*). For Depth, the row
   shows *asked* then Tower-side *starting* (spinner from the status
   document), then live; note `arm_ms` (expect 0.2-0.3 s warm, 2-3 s the
   first time in the process).
7. **Measure switch latency** across all eight from the phone's
   perspective (tap → first new `frame_result`) and from the Tower's
   (`run.arm_ms`). Record both for `depth` and `object_detection`.
8. **Object detection under load.** Select it; leave it running 2 min.
   On the Windows box: Task Manager CPU should be far below the previous
   99% (expect a few cores, not twenty); `process.threads` from
   `GET /cv-lab` steady; `run.runtime.torch_threads` = 2 and
   `run.runtime.device` = `cuda:0`. If it says `cpu`, the venv's torch is
   the CPU wheel (see `tower/pyproject.toml`) — report it, do not
   continue timing.
9. **Switch away from object detection** (to Baseline). Pass:
   `process.rss_mb` drops back toward its pre-detection value within a
   few seconds; `process.threads` unchanged; Task Manager CPU idles.
10. **Repeated switching.** Cycle all eight experiments three times by
    tapping. Pass: `process.threads` identical at the end of each cycle
    (±2); `rss_mb` within a few tens of MB; no `frame_error` other than
    `cv_lab_starting` during arms. Then run the live soak from the box
    while the phone keeps streaming:
    `& $py scripts\cv_lab_switch_soak.py --live --host 127.0.0.1 --port 8000 --tower-pid <pid> --cycles 4`
    Pass: verdict all `flat`, `children 0` (with autobuild off), exit 0.
11. **Pause the camera** (camera card → *Pause frames*). Pass: card reads
    *Paused (held on phone)*; the glasses camera is still running (the
    viewfinder on Home would still update); within ~5 s the Tower says
    `receiving_frames: false` and the run header drops LIVE; the
    experiment stays armed (`lifecycle.state` still `running`,
    `frames_offered` stops climbing).
12. **Resume.** Pass: frames flow again within a second; no re-arm
    (`run_id` unchanged, `arm_ms` unchanged).
13. **Stop the camera** from the card. Pass: card reads *Stopped*;
    Tower `receiving_frames: false`; experiment still armed. (With
    autobuild ON this is where a follower would be spawned per start —
    with it off, none.)
14. **Disconnect and reconnect** from the row while the camera is
    stopped, then start the camera again. Pass: streaming resumes;
    `isFrameSendingPaused` did not survive the disconnect (the card does
    not say Paused).
15. **Inspect Tower-owned Python processes** (commands in §4). Pass:
    with autobuild off, the Tower tree is exactly the uvicorn pair (venv
    launcher + real interpreter) and nothing else, at every point in this
    list.
16. **Error handling, three ways.**
    a. Send a bad start from the box:
       `python scripts\cv_lab_smoke.py --experiment not_real` — the Lab
       answers `unknown_experiment`; the phone's state is unchanged.
    b. Kill the Tower process while the phone is connected: the row goes
       *Error* / *Disconnected*, auto-reconnect tries five times, then
       the row's *Connect* button is the way back; start the Tower again
       and tap it.
    c. (Optional, destructive to nothing) `TOWER_CV_EXPERIMENT=not_real`
       in the env for one boot: `/health` says `module_state: active`,
       `GET /cv-lab` says `lifecycle.state: failed` with the reason, and
       the phone can still pick any experiment. Restore the variable.
17. **Shutdown.** Ctrl-C the Tower once. Pass: with autobuild off, no
    `worker pid ... did not exit` warning, and the process list is clean
    after exit. With autobuild on, expect the warning ~10 s after Ctrl-C
    if a walk was being followed — that is the follower, not the Lab.

Report back: the readings for 5, 7, 8, 9, 10; a pass/fail per step; the
live soak table; the process listing from 15; anything the phone showed
that this document did not predict.

## 4. Process inspection commands (Windows)

All Python processes with parent and command line:

```powershell
Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
  Select-Object ProcessId, ParentProcessId, CreationDate,
    @{n='WS_MB';e={[math]::Round($_.WorkingSetSize/1MB)}},
    @{n='Cmd';e={ if ($_.CommandLine) { $_.CommandLine.Substring(0, [Math]::Min(140, $_.CommandLine.Length)) } }} |
  Sort-Object CreationDate | Format-Table -AutoSize -Wrap | Out-String -Width 240
```

The Tower's own tree only:

```powershell
function Get-Tree($pid) {
  Get-CimInstance Win32_Process -Filter "ParentProcessId = $pid" | ForEach-Object { $_; Get-Tree $_.ProcessId }
}
$tower = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
  Where-Object { $_.CommandLine -match 'tower\.main:app' } | Select-Object -First 1
$tower, (Get-Tree $tower.ProcessId) |
  Select-Object ProcessId, ParentProcessId, @{n='WS_MB';e={[math]::Round($_.WorkingSetSize/1MB)}},
    @{n='Cmd';e={$_.CommandLine.Substring(0,[Math]::Min(120,$_.CommandLine.Length))}} | Format-Table -AutoSize
```

Orphaned followers (a parent that no longer exists):

```powershell
$alive = (Get-CimInstance Win32_Process).ProcessId
Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
  Where-Object { $_.CommandLine -match 'world_build_session|object_memory_session' -and $alive -notcontains $_.ParentProcessId } |
  Select-Object ProcessId, ParentProcessId, CreationDate, @{n='Cmd';e={$_.CommandLine.Substring(0,100)}}
```

Remember the pairing: every venv `python.exe` is a launcher whose child
is the real `...\Python312\python.exe`. One worker = two rows.

## 5. Known risks and unresolved items

- **iOS is uncompiled.** See §2. The behaviour is designed and statically
  reviewed; the Mac build is the first compile.
- **Inference still runs on the event loop.** A 40 ms object-detection
  frame holds every socket for 40 ms, including command replies. Bounded
  (one frame at a time per connection; the phone drops rather than
  queues) but real. Moving it off the loop with a single-slot
  newest-wins policy is the next Tower-side improvement and touches
  shared transport (`routes/ws.py`), so it was left out of this lane.
- **The startup load timeout is still terminal** (container's 120 s
  bound). A warm load is two orders of magnitude inside it; a Tower with
  no network on its first `depth` boot would hit it. Use a cheap default
  and let the phone pick `depth`.
- **World Builder followers** are spawned per camera start whenever
  `TOWER_WORLD_ROOT` is set and autobuild is on, are never asked to stop,
  and survive hard kills. Documented in the findings report with three
  possible fixes in shared infrastructure; deliberately not changed by a
  CV Lab lane.
- **OpenCV's thread pool** spins on cheap experiments (3-5 cores for a
  1.5 ms edge pass). Cosmetic in Task Manager; `cv2.setNumThreads` is not
  set anywhere and was not added here.
- **The torch thread cap is process-wide.** If Scene Understanding is
  ever turned on in the same process with `TOWER_SCENE_TORCH_THREADS`,
  whichever model loaded last decides.
- **`process` on `GET /cv-lab` is HTTP-only** by design; the phone does
  not show it. If tomorrow wants it on the phone, that is a deliberate
  contract addition, not a field to slip into the document.

## 6. Temporary resources created by this lane

- Worktree `C:\Users\tvllo\Projects\Glasses-worktrees\cv-lab-runtime`
  (branch `feature/cv-lab-runtime-controls`). Persistent.
- pytest basetemp `C:\Users\tvllo\AppData\Local\Temp\gf-cvlab`; session
  scratchpad under `C:\Users\tvllo\AppData\Local\Temp\claude\...`
  (measurement scripts, soak JSON). OS temp.
- Nothing under `C:\`, `C:\Users\tvllo\`, or the canonical checkout.
