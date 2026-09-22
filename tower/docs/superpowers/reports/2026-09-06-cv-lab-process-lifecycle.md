# CV Lab process, thread and memory lifecycle: what was found

**Date:** 2026-09-06
**Lane:** `feature/cv-lab-runtime-controls`
**Host:** Windows 11, 20 logical CPUs, RTX 5070 (sm_120), Python 3.12.5,
torch 2.13.0+cu132 (CUDA available), uvicorn 0.52.4.
**Prompted by:** a live test in which `object_detection` drove the machine
to ~99% CPU and ~96% RAM, many `python.exe` processes survived a Tower
shutdown, and shutdown logged
`worker pid ... did not exit within 10.0s ... anything it had not finished writing is lost`.

Every number below was measured on this host on this date unless it says
otherwise. Commands are at the end.

## 1. What the CV Lab owns

**No child processes.** The Lab's frame path is synchronous on the event
loop (`routes/ws.py` → `ModuleContainer.process` → `CVLab.process` →
`experiment.run`). Nothing in `tower/cv_lab/` or `tower/experiments/`
uses `multiprocessing`, `subprocess`, executors or per-frame threads.

**One thread, now.** Before this lane, every arm ran `load()` on a fresh
daemon thread (`tower/loading.py::run_abandonable`). After it, a Lab has
one `tower-cv-lab-loader-N` thread, reused across arms, retired and
replaced only when an arm is abandoned (timeout or `cv_lab_stop`
mid-load), and exited when the Lab is released or garbage-collected.

## 2. The thread leak (fixed)

A thread that runs a torch model load creates torch's intra-op OpenMP
team on itself. The team does not leave when the thread does.

| Mode | `object_detection` ×6 arms | `depth` ×6 arms |
|---|---|---|
| fresh thread per arm, torch default threads | 48 → 143 OS threads (+19/arm), RSS +8-9 MB/arm | 163 → 258 (+19/arm) |
| fresh thread per arm, torch threads = 2 | +1/arm | +1/arm |
| one persistent loader thread | **±0** | **±0** |

The 48-transition in-process soak (6 cycles × 8 experiments, 15 frames
per arm) confirms it end to end:

| | before (fresh thread per arm) | after (one loader) |
|---|---|---|
| threads, cycle 0 → cycle 5 | 115 → 344 (+38 per cycle) | 98 → 98 |
| RSS, cycle 0 → 5 | 1,645 → 1,727 MB | 1,620 → 1,623 MB (object_detection rows), 1,746-1,749 (depth rows) |
| handles | 675 → 1,134 | 654 flat |
| CUDA allocated / reserved after release | 0 / 0 | 0 / 0 |
| verdict | threads, rss, handles GROWING | all flat, exit 0 |

The same walk against a real uvicorn Tower over `/ws` (4 cycles × 8
experiments plus warm-up, 15 frames per arm, no `.env`): 40/40 arms
`running`, 600/600 frames processed, threads 62 in every cycle, RSS
+7 MB across the run, handles flat, zero child processes, `arm_ms`
255-265 ms for both model-backed experiments, exit 0. The Tower was one
process (plus its venv launcher) throughout and left nothing behind when
stopped.

The remaining steady state is what a process that has used torch on two
threads (the loop thread that runs inference and the loader) holds:
about 98 threads and 1.6-1.75 GB RSS with the CUDA runtime resident.
That is the price of CUDA torch in the process, paid once, and it does
not move with switching.

## 3. The CPU (fixed by a cap; the rest is documented)

torch's intra-op pool defaults to one thread per logical CPU (20). Intel
OpenMP workers spin-wait (`KMP_BLOCKTIME`, 200 ms default) after every
parallel region, so at a 40-80 ms frame interval they never sleep.

| Experiment / device | threads | cores busy | p50 ms/frame |
|---|---|---|---|
| `object_detection` CUDA | 20 / 8 / 4 / 2 / 1 | 12.6 / 7.5 / 3.6 / 1.8 / 0.9 | 43.6 / 41.9 / 40.0 / 40.1 / 40.0 |
| `depth` CUDA | 20 / 8 / 4 / 2 / 1 | 6.2 / 3.3 / 1.8 / 1.1 / 0.9 | 11.8 / 11.8 / 11.9 / 11.7 / 16.3 |
| `object_detection` CPU | 20 / 8 / 4 / 2 / 1 | 13.1 / 7.1 / 3.8 / 1.8 / 1.0 | 46.6 / 40.8 / 35.9 / 46.6 / 45.4 |
| `depth` CPU | 20 / 8 / 4 / 2 / 1 | 12.5 / 7.7 / 3.9 / 1.9 / 1.0 | 42.3 / 34.1 / 48.2 / 58.9 / 79.3 |

`TOWER_CV_TORCH_THREADS` now defaults to 2 on CUDA and 4 on CPU. On CUDA
that returns ten cores at identical latency. The budget is per calling
thread in OpenMP, so it is applied both on the loader thread (in
`load()`) and on the inference thread (on `run()`).

Two CPU consumers remain and are outside this lane:

- **OpenCV's own pool.** `edge_detection` reads 3.5-5.3 cores for a
  1.5 ms operation: cv2's parallel backend spinning. `cv2.setNumThreads`
  is not set anywhere in the Tower. Harmless per frame, visible in Task
  Manager.
- **World Builder followers** (§4), each a CPU-bound numpy/cv2 process
  with its own 20-thread cv2 pool, rebuilding every 4 keyframes.

## 4. The leftover processes (World Builder followers; documented, not changed)

The warning text is `tower/capture_workers.py` `_stop_worker`, and with
the operator's `.env` (`TOWER_CAPTURE_ROOT` and `TOWER_WORLD_ROOT` set,
`TOWER_WORLD_AUTOBUILD` defaulting to true) the pid it names is a World
Builder follower: `scripts/world_build_session.py --follow-capture ...`.

- **Spawned per capture lineage.** Every `stream_start` that opens a new
  lineage spawns one (`routes/ws.py` → `supervisor.capture_opened`). A
  reconnect within 90 s chains into the same child; a camera Start more
  than 90 s after the previous Stop is a new child. In a CV Lab session,
  that is every camera start.
- **Never asked to stop.** The builder's spec has no stdin stop channel
  (`main.py::_world_build_spec` sets no `stop_via_stdin`), so on shutdown
  the supervisor waits the full 10 s grace and then terminates it. Its
  normal exit is ~20 s after the capture closes (final build +
  registration), or 90 s waiting for a successor, or up to 900 s idle if
  the Tower died without closing the manifest. **The warning on a clean
  shutdown mid-walk is therefore expected.**
- **Survive a hard kill.** Children run in their own process group with
  no Job object. A second Ctrl-C, `Stop-Process`, closing the terminal,
  or `start_tower.ps1 -Force` orphans them; they then poll for up to
  900 s.
- **Two rows each.** A venv `python.exe` on Windows is a launcher that
  spawns the real `Python312\python.exe` as its child, so every worker
  is two processes in Task Manager (observed on this host: the World
  Builder lane's replay showed exactly that pairing).
- **Object Memory's producer** is spawned only while a
  `POST /cartridges/object_memory/session/start` session is active and
  *is* asked to stop (stdin close + `CTRL_BREAK_EVENT`).

What the operator can do today, without a code change: set
`TOWER_WORLD_AUTOBUILD=false` in `.env` for CV Lab sessions. Captures
still record; no follower is spawned; the shutdown warning disappears.
Recommended for tomorrow's CV Lab validation so that CPU and process
counts describe the Lab alone.

What a later lane could do (shared infrastructure, deliberately not
touched here): give the builder a stdin stop channel so the grace is
spent finishing rather than waiting; put children in a Windows Job
Object with `KILL_ON_JOB_CLOSE` so a hard-killed Tower takes them along;
have `start_tower.ps1` list stray followers the way it lists a stale
uvicorn.

## 5. The RAM

The Tower web process with CUDA torch resident sits at 1.6-1.75 GB and
does not grow with switching (§2). The 96% figure was that plus the
followers (each holding a growing SfM state plus numpy/cv2, one per
lineage) plus whatever else the machine was running. Per-PID RSS with
children is the reading to take before attributing memory to the Lab;
the command is below.

CUDA: `depth` holds 83 MB allocated / 120 MB reserved while armed,
`object_detection` 13 / 78; both return to 0 / 0 after `release()`, every
time, across 12 cycles.

## 6. Switch latency

| Transition | arm (warm) | arm (first in process) | release |
|---|---|---|---|
| any → `depth` (CUDA) | 220-300 ms | 2.3-3.1 s | 2-3 ms |
| any → `object_detection` (CUDA) | 125-160 ms | 310-360 ms | 2 ms |
| any → cheap experiment | < 1 ms | < 1 ms | < 1 ms |

Frames sent during an arm are refused with `cv_lab_starting`; the socket
and the camera stay up. A model cache was considered and rejected on
these numbers: a quarter second is the whole downtime, and a cache would
keep a second model resident for it.

## 7. Terminal states (fixed)

Before: an experiment raising anything but `FrameProcessingError` on a
frame, or a startup default failing to load, marked the module FAILED
for the life of the process. After: both leave the Lab `failed` with the
reason, the module ACTIVE, and the next `cv_lab_start` works. What
remains terminal is a startup load overrunning the container's 120 s
bound. See `tests/test_cv_lab_runtime_controls.py`.

## 8. Commands

Tower-owned Python processes, with parent and command line (CIM, not
`Get-Process`, because only the command line tells a follower from a
stray shell):

```powershell
Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
  Select-Object ProcessId, ParentProcessId, CreationDate,
    @{n='WS_MB';e={[math]::Round($_.WorkingSetSize/1MB)}},
    @{n='Cmd';e={ if ($_.CommandLine) { $_.CommandLine.Substring(0, [Math]::Min(140, $_.CommandLine.Length)) } }} |
  Sort-Object CreationDate | Format-Table -AutoSize -Wrap | Out-String -Width 240
```

Only the Tower's tree (replace the pid with uvicorn's; the tree includes
the venv launcher pairs):

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

Followers that outlived a Tower (no live parent):

```powershell
$alive = (Get-CimInstance Win32_Process).ProcessId
Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
  Where-Object { $_.CommandLine -match 'world_build_session|object_memory_session' -and $alive -notcontains $_.ParentProcessId } |
  Select-Object ProcessId, ParentProcessId, CreationDate, @{n='Cmd';e={$_.CommandLine.Substring(0,100)}}
```

The Tower's own reading, from anywhere on the tailnet:

```powershell
(Invoke-RestMethod http://<tower>:8000/cv-lab).process      # pid, threads, rss_mb
(Invoke-RestMethod http://<tower>:8000/cv-lab).status.run   # arm_ms, runtime.torch_threads, frames_*
(Invoke-RestMethod http://<tower>:8000/health).capture_workers
```

The soak, in-process and live (from `tower/`):

```powershell
$py = "C:\Users\tvllo\Projects\Glasses\tower\.venv\Scripts\python.exe"
& $py scripts\cv_lab_switch_soak.py --cycles 6 --frames 15
& $py scripts\cv_lab_switch_soak.py --live --host 127.0.0.1 --port 8000 --tower-pid <pid> --cycles 4
```

Thread-team and thread-cap measurements were made with two throwaway
scripts in the session scratchpad (`measure_arm.py`,
`measure_thread_leak.py`); their method is reproduced by
`tests/test_cv_lab_torch_threads.py` and `tests/test_cv_lab_loader.py`.

## 9. Temporary resources created by this lane

- Worktree `C:\Users\tvllo\Projects\Glasses-worktrees\cv-lab-runtime`
  (branch `feature/cv-lab-runtime-controls`). Persistent; not to be
  removed without the human's say.
- pytest basetemp `C:\Users\tvllo\AppData\Local\Temp\gf-cvlab` (OS temp;
  allowed).
- Session scratchpad under
  `C:\Users\tvllo\AppData\Local\Temp\claude\...\scratchpad` (measurement
  scripts and soak JSON; OS temp).
- Nothing under the drive root, the home directory, or the canonical
  checkout.
