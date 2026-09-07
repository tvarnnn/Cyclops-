# World Builder + CV Lab — Windows live validation of the combined branch (2026-09-06)

**Lane:** Windows live-integration validation.
**Branch:** `integration/wb-cv-ios-validation-v1`, tested at `4ba531d` (the Mac
integration head) plus one commit made here (§7). Checked out in the
**canonical** checkout `C:\Users\tvllo\Projects\Glasses`, tracking
`origin/integration/wb-cv-ios-validation-v1`, as the mission required.
**Host:** Windows 11, 20 logical CPUs, RTX 5070, Python 3.12.5 in
`tower\.venv` (torch 2.13.0+cu132, pycolmap 4.2.0, OpenCV 5.0.0).
**Tailscale:** this box is `100.110.156.55`; the phone's Release build hard-codes
`100.110.156.55:8000` (`TowerConfiguration.defaultAuthority`), so every
phone-facing Tower here ran on port 8000.

Section §8 says exactly which hardware steps ran and which did not. Nothing
below claims a phone step that did not happen.

---

## 1. Git

| Check | Result |
|---|---|
| Canonical checkout before | `integration/all-current-v1 @ a7b1c2a`, clean except the pre-existing untracked `orb_vocab_stella.fbow` (preserved) |
| `origin/integration/wb-cv-ios-validation-v1` | exists, head `4ba531d`; `a7b1c2a` is its ancestor |
| Switch | `git switch -c integration/wb-cv-ios-validation-v1 --track origin/...` (no reset, no history rewrite) |
| Contains World Builder `ddff465` | yes |
| Contains CV Lab `bb47638` | yes |
| Contains Mac iOS/viewer commits `5d5fef6 … 9967420` | yes |
| Old source worktrees (`wb-recon`, `cv-lab-runtime`, …) | untouched; `git worktree list` unchanged at the end |

The three handoffs (`WB-CV-IOS-INTEGRATION-VALIDATION.md`,
`WORLD-BUILDER-GLOBAL-SOLVER.md`, `CV-LAB-RUNTIME-CONTROLS-HANDOFF.md`) were
read in full before testing.

## 2. Real World Builder data: what exists, and what the Tower now sees

**Runtime configuration** (`tower\.env`, unchanged): `TOWER_CAPTURE_ROOT=data`,
`TOWER_WORLD_ROOT=data/world_builder` — both relative, resolved from `tower\`,
which is where every Tower in this session was started. Boot log:
`[Tower][Config] world root data/world_builder`.

**What was on disk before this session:**

| Location | Content |
|---|---|
| `tower\data\world_builder\worlds\` (canonical root) | 155 worlds; the 2026-09-06 live walk is `678fe396b5de41e3b8659ec56500dc1c` (session `b6b47fdd…`, capture `ddcf9426…`, 438 keyframes) with the **old chain geometry** (194 poses, 2 registered segments) |
| `Glasses-scratch\wbrecon\live\0906\` | the same walk replayed through the product path with the global solver: world `7d31e8d7acde46808b7a31f1b7bc211e`, 425/438 posed, 29 of 34 segments in one frame |
| `Glasses-scratch\wbrecon\final\{0901,worldA,worldB,dense,long0827}\` | five more globally-solved replays, one world each |
| `Glasses-scratch\wbrecon\smoke\root0906\` | the solver run on a *copy* of `678fe396` (same world id; not migrated, see below) |

`GET /worlds` scans `<world root>/worlds/*/world.json`, so the replay roots
were invisible to the canonical Tower. They are in the store's own layout
already; no manifest conversion was needed, only placement.

**Migration performed** (`Glasses-scratch\wb-validate\migrate_worlds.py`,
record `Glasses-scratch\wb-validate\MIGRATION-RECORD.json`, every file
listed with source, destination and size). For each of the six replay
worlds, copied into `tower\data\world_builder\worlds\<world_id>\`:

- `world.json` — with `display_name` set **on the copy only**, e.g.
  `2026-09-06 walk (global solve, replay)`, so the phone's picker (which
  shows `display_name ?? worldID`) names them
- `sessions/<sid>/{session.json, keyframes.jsonl, events.jsonl, edges.jsonl}`
- `derived/**` (manifest + per-session poses/points/placements/support)
- `solve/<sid>/{solution.json, solution.npz, camera.json, sources.json, solve.log}`
  (so `engine.build()` would still find the persisted solution)

Skipped: `sessions/<sid>/images/` (keyframe JPEGs; nothing on the phone
path reads them) and `solve/<sid>/{database.db, images/, sparse/}` (the
matcher's workspace, 50–130 MB per world). Total copied: **18 MB, 90 files,
six worlds**. Originals untouched. Nothing was added to Git. Intrinsics were
not copied (the canonical root has its own). The canonical live world
`678fe396` was **left exactly as it was** so the phone can compare the old
chain geometry with the replayed global solve of the same walk; the
`smoke\root0906` copy shares its id and was therefore not migrated.

| World (display name) | id | session | keyframes | geometry |
|---|---|---|---|---|
| **2026-09-06 walk (global solve, replay)** — primary target | `7d31e8d7acde46808b7a31f1b7bc211e` | `2934e5b5…` | 438 | 419 cameras in the world frame, 29 registered segments; a 6-camera second component; 3 refused fragments |
| 2026-09-01 loop (global solve, replay) | `c2e3cb8ad5d74be9beddb6cd454cb0e5` | `13464d7a…` | 434 | yes |
| 2026-08-29 worldA normal (global solve, replay) | `1adc5e356cf14b4fa1cacf3c8cbc4ed8` | `eb9b168d…` | 229 | yes |
| 2026-08-29 worldB drawer (global solve, replay) | `a599b6f4e62b4115a63e101d9348bc56` | `0b693676…` | 218 | yes |
| 2026-08-29 dense (global solve, replay) | `ecc02df10a504fe69a44e56e434bb65f` | `812905f1…` | 77 | yes |
| 2026-08-27 long (global solve, replay) | `9a68430a44384ae595d874a4571f5b50` | `7ba2395c…` | 339 | yes |
| 2026-09-06 live walk, old chain geometry (unnamed) | `678fe396b5de41e3b8659ec56500dc1c` | `b6b47fdd…` | 438 | 33 frames on the page: two registered, the rest apart |

## 3. `GET /worlds` and `GET /worlds/{id}/render` against the real data

Tower from the canonical checkout, `127.0.0.1:8010`, `.env`,
`TOWER_WORLD_AUTOBUILD=false`; script `Glasses-scratch\wb-validate\check_routes.py`,
log `logs\check_routes-8010.log`.

| Request | Result |
|---|---|
| `GET /worlds` | 200, 110 ms, `world_builder.worlds/2026-09-06`, **161 worlds**, the six named worlds present with `has_geometry: true`, no path string anywhere in the body |
| `GET /worlds/7d31e8d7…/render` | **200, 545 KB, 67 ms**, `text/html`, `Cache-Control: no-store`, CSP `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'` |
| its `FRAMES` payload | 5 frames: `world (ref segment 0, 29 registered segments)` 419 cameras; `world_ref28_rev1 (ref segment 28, 2 registered segments)` 6 cameras; `UNREGISTERED segment 18/27/30 [refused] -- own frame, own scale` |
| its caption (visible text) | *Sparse structure-from-motion output: triangulated feature points and camera poses. Not a surface, not a mesh, not metric scale.* No BEHIND line (derived is current: manifest `current: true`) |
| path strings in the page (`C:\`, `Glasses-scratch`, `wbrecon`, `/Users/`) | none |
| `?session_id=2934e5b5…` | 200, identical page |
| `?max_points=5000` | 200, 323 KB |
| `GET /worlds/678fe396…/render` (old chain geometry) | 200, 809 KB, 113 ms, 33 frames |
| `GET /worlds/{id}/geometry/manifest?session_id=…` for `7d31e8d7` | 200, `current: true`, 34 segments, coverage 23 confident / 8 partial / 3 unresolved |
| `GET /worlds/doesnotexist/render` | 404 `{"detail":"no world 'doesnotexist'"}` |
| `?session_id=bogus` | 404 `world '…' has no session 'bogus'` |
| `GET /worlds/..%5C..%5Cintrinsics/render` | 404 `no world '..\\..\\intrinsics'` — containment holds, id echoed as text, no path |
| `?max_points=999999` / `=0` | 422 (`le 200000` / `ge 1`) |

No fixture world exists on this box; every reading above is against the
canonical root. The "no geometry" 404 wording was exercised through the
route tests (12 in `test_world_builder_render_route.py`) because every
session in the canonical root that has a session record also has a derived
tree; see §6 for what an *abandoned* session renders as.

## 4. Environment

| Check | Result |
|---|---|
| `pycolmap` import | 4.2.0, `solver_available() -> (True, None)`; nothing was installed |
| pycolmap vocabulary tree (loop detection at finalisation) | already cached at `~\.cache\colmap\…vocab_tree_faiss_flickr100K_words256K.bin` — no download during a live final solve |
| `scripts/world_builder_env_check.py` | all verdicts OK: RTX 5070 visible, torch CUDA usable (`sm_120`), OpenCV geometry complete, ChArUco available |
| CV Lab boot line | `CV Lab startup default is 'baseline' on device 'auto' with torch threads 'auto'` |
| `python -m pip show tower` | not installed as a package; the Tower runs from the checkout (as every earlier lane did) |

## 5. Tower test results (Windows, canonical checkout, this branch)

| Run | Result |
|---|---|
| Targeted subset (render route, world_render, library, geometry + transport, global solve, solve cadence, store, CV Lab runtime controls / loader / torch threads / protocol / lifecycle / shutdown / catalog, capture workers + wiring) | **308 passed** in 19 s |
| **Full suite** `python -m pytest -q` | **2613 passed, 34 skipped, 1 xfailed, 0 failed** in 9 min 57 s (`logs\pytest-full.log`) |
| After the §7 commit: library + render route + geometry transport | 62 passed |
| `ios/scripts/contract-drift-check.py --tower http://127.0.0.1:8000` | AGREEMENT — every contract the Tower states is implemented by the Swift on this branch (the additive `abandoned` key does not change a contract id) |

The Mac reported 89 failures that were all Mac-environment; on Windows the
same suite has none. The 34 skips are the designed ones (no model on some
tests, platform-specific).

## 6. Data finding: 28 abandoned sessions, and what the phone made of them

The listing showed 28 worlds (8 of them from 03:12–03:47 on 2026-09-06, the
morning's reconnect cycle) with `ended_at: null`, `end_reason: null`,
`keyframes_accepted: 0`, while their `keyframes.jsonl` held 6–292 rows,
their derived manifests reported real keyframes and poses, and each `LOCK`
named a pid that no longer exists. Two read-only subagent investigations
established (file:line evidence in their reports, summarised here):

- `session.json` is written exactly twice: at `start_session` (all counts
  zero) and at `stop_session` (`engine.py` ~433–444). `observe()` counts in
  memory only. So a builder that dies before the follower returns leaves a
  t=0 record forever. The eight 09-06 sessions all end on a
  `keyframe_accepted` event with the manifest `built_at` matching the last
  keyframe to the second: killed mid-walk right after an interim rebuild.
- The builder worker is never *asked* to stop: its `WorkerSpec` sets no
  `stop_via_stdin`, so `CaptureWorkerSupervisor.shutdown` waits the 10 s
  grace and then `TerminateProcess`es it (`capture_workers.py` ~896–918);
  the script has no signal handler or `try/finally`. That is the same
  shared-infrastructure gap the Mac handoff §11.1 names.
- `stop_session` runs **before** the final solve and final build, so a kill
  during the 30–135 s final solve loses only the final solution/build, never
  the session record or the lock release. An abandoned record therefore
  means "killed during the walk", not "killed during the final solve".
- The iOS picker does not decode `keyframes_accepted`; it shows
  **"still open"** whenever `ended_at` is null (`WorldPickerView.swift` ~79–83)
  — so those 28 sessions read as open on the phone, while the world row is
  not badged live. The render route serves them (the derived tree exists);
  five are BEHIND their journal and say so, three are exactly current.

**Fix made (§7):** one additive per-session boolean on `GET /worlds`,
`abandoned` = `ended_at is null and the world is not live`, computed from
the liveness answer the route already has. On the canonical root it is
true for exactly those 28 sessions and no other session is open. The iOS
side is a two-line follow-up for a Mac lane (`WorldLibrary.swift` decode,
`WorldPickerView.swift` label "unfinished" instead of "still open"); it was
not made here because Swift cannot be compiled on this box.

## 7. Commits on this branch from this session

| Commit | What |
|---|---|
| `1647c4b` | `feat(world-builder): the listing says when an open session was abandoned` — `world_builder_library.py`, one test (finished → false; open + dead lock → true; open + no lock → true; open + running lock → false), contract row + rule in `WORLD-BUILDER-WORLDS.md` |
| `b10ab36` | this document, first version |
| `12e4f1e` | `fix(world-builder): a rebuild that cannot write no longer ends the session` — `storage._replace_with_retry` budget 60 ms → 2 s with backoff; `world_build_session.py` logs and retries a failed interim rebuild; `tests/test_storage_replace_retry.py` (2), `tests/test_world_builder_rebuild_failure.py` (1). Related suites: 226 passed. Root cause and evidence in §8.4 |

Each made through a throw-away linked worktree
(`Glasses-worktrees\wb-cv-ios-validation-win`, branch `tmp/…-fix`) and a
`--ff-only` merge in the canonical checkout, because the lane guard refuses
agent commits in the canonical checkout and `--no-verify` is not an option.
The worktree and the temporary branch were removed afterwards; the canonical
checkout is on `integration/wb-cv-ios-validation-v1` at `1647c4b`, clean.

## 8. What ran under real load

The session had two halves. Until 20:12 no phone connected and this box
measured what it could on its own (§8.1–8.2). At 20:12:43 the iPhone
(`100.75.17.33`) connected; the person at the keyboard then ran their own
Towers on port 8000 from this checkout (three instances between 20:13 and
20:31; their console output was not captured to a file) and drove the
phone through a live World Builder capture and the CV Lab list. Those runs
were observed from here through the Tower's HTTP surface and the world
directory, not through the app's screen (§8.4–8.5). The iOS build's commit
still cannot be confirmed from this side: the app carries no build identity
on the wire (`CURRENT_PROJECT_VERSION = 1`, no hello payload from the
phone); the Picture button beside *Saved worlds* and the Tower row at the
top of CV Lab exist only on this branch, and the person holding the phone
is the only one who can say they were there.

What this box could measure on its own, against the branch's Tower:

### 8.1 CV Lab under real inference load (`scripts/cv_lab_switch_soak.py --live`)

Second Tower from the canonical checkout, `127.0.0.1:8017`, no `.env`,
`TOWER_WORLD_AUTOBUILD=false` (so no follower and no world written), pid
30920. Four cycles of all eight experiments, 15 frames per arm, over `/ws`
(`logs\soak-8017.{log,json}`):

| Reading | Result |
|---|---|
| Arms | 40/40 `running` (+8 warm-up), 600/600 frames processed, 0 refused, 0 errors |
| `arm_ms` warm: `depth` / `object_detection` | 260–262 / 259–268 ms (first in process: 2994 / 519 ms) |
| `arm_ms` cheap experiments | 0.9–1.6 ms; `redaction_impact` 3.3 ms |
| Threads (OS) | **64 → 64, flat** across cycles; no loader thread per switch |
| RSS | 1634 → 1640 MB, **+6 MB** over 32 switches (torch+CUDA resident) |
| Handles / children | 610 flat / **0** throughout |
| Stop latency | 1.0 ms per cycle; verdict `flat` on every axis, exit 0 |

Then 600 frames of `object_detection` alone (`logs\soak-8017-objdet.*`,
`logs\cpu-objdet-8017.log`), sampled every second with `psutil` on the
interpreter pid:

| Reading | Result |
|---|---|
| `run.runtime.device` / `torch_threads` | `cuda` / **2** (`TOWER_CV_TORCH_THREADS` unset → auto) |
| Throughput | 20.0–20.5 fps at 42–43 ms per frame (inference stage 40.3 ms) |
| Process CPU while processing | **p50 1.66 cores, max 2.02 cores** of 20 (the Tower's own session summary: `process_cpu_percent` 168.5) — the pre-fix 12.6-core / "99 %" behaviour does not reproduce |
| Threads / RSS / children during and after | 61–62 flat / 1619 MB flat / 0 |
| Shutdown of that Tower | clean; no `did not exit` warning; process table back to the one Tower on 8000 |

CUDA memory per process is reported as `[N/A]` by `nvidia-smi` under WDDM
on this box; the soak's `cuda a/r` column is `n/a` for the same reason.
Device-wide GPU memory was 1437 MiB idle before the soak.

### 8.2 The solver path on this branch, end to end, on the real 09-06 capture

`scripts/world_replay.py --captures ddcf9426… --solve --register` from the
canonical checkout and venv into `Glasses-scratch\wb-validate\replay0906\`
(`logs\replay0906.log`): the builder exactly as the Tower supervises it,
three background solve children during the walk, then the final solve.

| Reading | Result |
|---|---|
| Keyframes / segments | 438 / 34, 110 interim rebuilds |
| Global solve | `glomap`, **424 of 438 posed**, 14 790 points; components 429 images (424 supported) + 7 |
| **Final solve wall time** | **102.9 s** (match 28.9 s, map 71.4 s, 94 images undistorted incrementally) |
| Registrar | stood down: "placements come from the global solve" |
| Wall time, whole replay | 184 s |

That final-solve figure is the number the shutdown risk (§6, Mac handoff
§11.1) is about: after Stop, a Tower shutdown inside the next ~103 s here
would `TerminateProcess` the builder at the 10 s grace and lose the final
solution, keeping the last background one. The session record and lock
are already final by then (§6).

### 8.4 Live World Builder capture from the glasses (20:14–20:17)

Tower instance `8f42e16dc7ac` (the person's own, from this checkout at
`b10ab36`, `.env`, autobuild **on**, solve on). Observed through `/health`,
`/worlds`, the render route and the world directory
(`logs\live-world-fcbca9e9.jsonl`, `logs\cvlab-phone-8000.jsonl`):

| Time | What |
|---|---|
| 20:12:43 | phone `/ws` accepted on the Tower this lane had started; that Tower was ended at 20:12:56 and the person's own took port 8000 at 20:13:03 |
| 20:14:16 | capture `7febdae8…` starts; builder follower spawned (launcher 26560, interpreter **19604**); frames arriving |
| 20:14:17 | world `fcbca9e90b244785bdb671530b33c6a5`, session `158ef0ef…` created |
| 20:14:58, 20:15:14, 20:15:37, 20:16:07, 20:16:37, 20:17:06 | background solves land every ~50 keyframes; each is merged by the next rebuild (`manifest.global_solve.solved_at` tracks `solution.solved_at`) — **the phone saw segments snap together mid-walk, as designed** |
| ~20:15:25 | `GET /worlds/fcbca9e9…/render` from here: 200, 292 KB, 115 ms, 14 frames, world frame 133 cameras / 7 registered segments, BEHIND caption; `GET /worlds` lists it `live: true` |
| **20:17:12** | last keyframe accepted (467, segment 79); `poses.json` rewritten 20:17:12.798, `points.json` **not** (still 20:17:11.7), no `.tmp` left |
| 20:17:13 | `/health` `capture_workers.workers` empty — the supervisor reaped the builder. **It died 20 s before Stop, not at shutdown; the Tower kept running.** |
| 20:17:33 | phone Stop; `capture.json` `ended_at`, `end_reason: stop`, 2270 frames |
| 20:17:44 | the last background solve child (launched 20:17:10) finishes and writes its solution; nobody is left to merge it |
| afterwards | `session.json` `ended_at: null`, `keyframes_accepted: 0`; `LOCK` names dead pid 19604; no `session_stopped`; no final solve; derived tree torn (`poses.json` from build N+1, `points.json`/manifest from build N). `GET /worlds` lists it **`abandoned: true`** (§7), `live: false`; the render still serves the last merged solve: 39 frames, 257 cameras in the world frame, 23 registered segments, BEHIND |

**Root cause** (evidence: the file timestamps above; a read-only code
investigation; a reproduction in isolation). The builder died inside
`write_derived` on the atomic replace of `points.json`:
`storage._replace_with_retry` gave up after 12 × 5 ms because a reader
held the destination open. On Windows `os.replace` fails with
`WinError 5` while *any* handle is open on the target, and the reader
here is the Tower's own web thread reading the 2.2 MB `points.json` for
the phone's geometry pull — descheduled under the background solve child,
which uses 18 of 20 cores. Reproduced in isolation: a reader holding the
file for 120–150 ms a few times a second fails **7 of 30** atomic writes
under the 60 ms budget (`logs\` and `tests/test_storage_replace_retry.py`).
The same capture replayed offline through the identical builder path
(`--solve --register`, no concurrent reader) built **519 keyframes, 88
segments, final solve 93 s, exit 0**, so frames, merge and solver are not
the trigger. The morning's eight abandoned worlds have a different
signature (complete last build, killed at Tower shutdown).

**Fix** (`12e4f1e`, §7): the replace budget is now 2 s with 5→50 ms
backoff (0 of 30 failures against the same reader; a parked reader still
fails within the budget), and an `OSError` from an *interim* rebuild is
logged and retried at the next rebuild instead of ending the session —
so the stop, the final solve and the final build still happen. Builder
followers are spawned fresh per capture and import `storage` on start,
so the person's running Tower picks the fix up on its next capture with
autobuild on, without a restart. Not re-run on hardware in this session.

**Final-solve timing** could not be measured on the live walk because the
builder never reached it; the offline figure for this capture is 93 s
(match 32 s, map 57 s) and for the 09-06 capture 103 s (§8.2).

### 8.5 Live CV Lab from the phone (20:30–20:37)

Tower instance `36ed7ec381c0` (the person's own, this checkout, autobuild
**off** — `capture_workers.configured` was `object-memory-session` only),
observed at 1 Hz through `GET /cv-lab` with `psutil` on the interpreter
pid (`logs\cvlab-phone-8000.jsonl`):

| Time | run | experiment | `arm_ms` | device / torch threads | socket | camera |
|---|---|---|---|---|---|---|
| 20:30:49 | -1 | baseline (startup default) | 0.6 | | connected | |
| 20:31:09 | -1 | | | | | **frames arriving** (`receiving_frames: true`) |
| 20:31:20→:25 | -2 | depth | **4322** (first in process) | cuda / 2 | same | alive |
| 20:31:31 | -3 | edge_detection | 3.4 | | same | alive |
| 20:31:48 | -4 | feature_detection | 0.8 | | same | alive |
| 20:31:58 | -5 | frame_quality | 0.6 | | same | alive |
| 20:32:12→:13 | -6 | object_detection | **417** (first in process) | cuda / 2 | same | alive |
| 20:33:44 | -7 | optical_flow | 3.2 | | same | alive |
| 20:34:12 | -8 | redaction_impact | 0.6 | | same | alive |
| 20:35:21 | -8 | | | | connected | frames stop (`receiving_frames: false`), run still `running`, `run_id` unchanged |
| 20:36:19 → 20:36:32 | -8 | | | | **disconnected → reconnected** | |
| 20:36:42 | -8 | | | | connected | frames resume; a capture with an object-memory worker starts |

Earlier, on instance `8f42e16dc7ac` at 20:18:24–20:18:38 the phone armed
`depth` (5122 ms cold), then **paused** (`lifecycle.state: paused`), then
**stopped** the run, then disconnected — the run-level Pause/Resume/Stop
controls, on the wire.

What that pass proves: all eight experiments switched from the phone on
**one socket and one Tower process** (instance id and `clients_connected`
never changed through the eight arms), the camera stayed alive across
every switch, warm arms are single-digit milliseconds, the heavy models
land on `cuda` with the two-thread budget, and a frame stop with the run
left armed keeps `run_id` and `lifecycle.state`. Whether the 20:35:21
frame stop was the camera card's *Pause frames* or *Stop* cannot be told
from the Tower side; both look the same there by design.

Resource readings in that pass, from the OS: threads 26 idle → 50 after
`depth` → 63 after `edge_detection` → 74 at `object_detection` → 81 at
`redaction_impact` → **95** after the disconnect/reconnect and the new
capture; RSS 70 → 1201 MB (depth) → 1708 MB (object detection) → 1798 MB.
The soak in §8.1 holds 64 threads flat across four full cycles once warm,
so the climb from 63 to 95 here is **not the experiment switching**; the
pass also included two capture recordings, a websocket disconnect and
reconnect, and an object-memory producer worker (`children: 2`), none of
which the soak exercises. One pass cannot separate those; it is recorded
as an observation, not a leak.

Frame rate from the glasses was ~12 fps; `object_detection` processed at
up to 55 ms per frame, `depth` 45 ms.

### 8.3 How to run the phone steps against this box (as it was set up)

The Tower this lane started for the World Builder list ran on
`0.0.0.0:8000` from the canonical checkout at `1647c4b` (`.env`, autobuild
and solve on) until the person replaced it with their own at 20:13. To
repeat: open World Builder → Saved worlds → **2026-09-06 walk (global
solve, replay)** → Picture.

For the CV Lab list, run with the follower disabled:

```powershell
cd C:\Users\tvllo\Projects\Glasses\tower
$env:TOWER_WORLD_AUTOBUILD = "false"
& .venv\Scripts\python.exe -m uvicorn tower.main:app --host 0.0.0.0 --port 8000 --env-file .env
# in a second terminal, per-second readings while switching on the phone:
& .venv\Scripts\python.exe C:\Users\tvllo\Projects\Glasses-scratch\wb-validate\cv_lab_watch.py --port 8000 --out C:\Users\tvllo\Projects\Glasses-scratch\wb-validate\logs\cvlab-phone.jsonl
```

After a live World Builder capture, reconstruct the Stop timeline from
disk (T0 `capture.json.ended_at`; T1 `events.jsonl` last `session_stopped`
= `session.json.ended_at`, LOCK gone; T2 `solve/<sid>/solution.json.solved_at`
with `timing.final: true`; T3 `derived/manifest.json.built_at` with
`global_solve.solved_at == T2`; T4 the Tower log's `worker pid … finished
after …s`). A kill shows as `manifest.global_solve.solved_at <
solution.solved_at`, or `solution.timing.final == false`, or the Tower's
`did not exit within 10.0s of being asked; terminating` line.

## 9. Process hygiene

Before anything ran: **zero** `python`/`uvicorn` processes on the box,
nothing listening on 8000–8020 — no stale process from an earlier session
existed to clean up. Towers started by this lane: 8010 (route checks,
stopped), 8000 (WB mode, restarted once after the §7 commit; ended at
20:12:56 when the person took the port), 8017 (soak, stopped). Each was
stopped through its own task handle and the process table re-read
afterwards. From 20:13 the Towers on 8000 were the person's own (three
instances, the last stopped at 20:38:03). This lane's 1 Hz samplers
(`cv_lab_watch.py`, the state and world monitors) were stopped at the end.
The full pytest run left nothing behind. The builder that died at 20:17:12
left no process; its LOCK is reclaimed by the next writer of that world.

## 10. Remaining defects and open items

1. **Not observed from this side**: whether the in-app viewer (Picture) and
   Saved Worlds navigation were exercised on the phone. The phone was
   connected and drove a live capture and the CV Lab list (§8.4–8.5), but
   the HTTP access log of the person's Tower went to their console, so a
   `GET /worlds/{id}/render` from the phone is neither confirmed nor
   denied here. The route is proven against the real data (§3); the
   in-app rendering is the Mac Simulator's proof plus whatever the person
   saw.
2. **Live final solve not yet observed on hardware**: the one live walk
   died before Stop (§8.4, fixed in `12e4f1e`). The next live capture with
   autobuild on is the test of the fix and of the 30–135 s final-solve
   window; the timeline recipe is in §8.3.
3. **Thread count after a phone session** (§8.5): 63 → 95 across a pass
   that mixed experiment switching with capture recording, a socket
   disconnect/reconnect and an object-memory worker. The soak shows
   switching alone is flat; the other three need their own soak.
4. **Reader side of the replace race** (pre-existing): a route reading a
   derived file while the builder replaces it can get `PermissionError`
   on `open`; readers treat that as absent geometry for that request. Not
   changed here; the writer side is what killed a session.
5. **Builder is never asked to stop** (shared infrastructure, pre-existing):
   no stdin/CTRL_BREAK channel, so Tower shutdown terminates it after 10 s
   and a mid-walk kill leaves an abandoned session (28 on this box). The
   listing now says `abandoned`; the phone still says "still open" until
   the two-line iOS follow-up lands. The root fix (a stop channel plus
   `stop_session` in a `finally`, and a longer grace while a solve runs)
   belongs to a shared-infrastructure lane.
6. **Interim geometry on abandoned sessions renders as if finished**: three
   of the 28 are exactly current, so the page shows no BEHIND line and no
   other hint that the walk ended by a kill. Only the listing flag says so.
7. **One derived manifest per world** (Mac handoff §9, unchanged).
8. **Per-process CUDA memory is unobservable** on this box (`nvidia-smi`
   `[N/A]` under WDDM); CUDA release must be judged by the soak's in-process
   counters or device-wide memory.

## 11. Verdict

The branch checks out and tests clean on Windows (2613/0), the real
reconstructions are visible to and rendered by the combined Tower from the
canonical world root, the solver path solves the real walk on this venv,
and the CV Lab runtime fixes hold under real CUDA inference load in the
soak and under the phone: eight experiments on one socket, camera alive,
no restart. The live World Builder walk found a real defect — a builder
that dies when the Tower reads the file it is replacing — with the root
cause pinned from disk, reproduced in isolation, fixed and tested
(`12e4f1e`), but **not yet re-run on hardware**. The listing gap the
abandoned sessions exposed is fixed additively (`1647c4b`).

**Recommendation:** merge this branch into `integration/all-current-v1`
as the new baseline after one more live World Builder capture with
autobuild on reaches its final solve and its saved world reopens on the
phone (§8.3's timeline recipe). Everything else the mission asked for is
either proven here or proven on the Mac; nothing found argues against the
merge, and the one thing found is fixed on this branch.

## 12. Temporary resources (filesystem policy rule 9)

- `Glasses-scratch\wb-validate\` — `migrate_worlds.py`, `MIGRATION-RECORD.json`,
  `check_routes.py`, `cv_lab_watch.py`, `abandoned-flag.patch`, saved render
  pages, `logs\` (pytest, Tower, soak, samplers), pytest basetemps, and
  `replay0906\` (the §8.2 replay root, 130 MB). Disposable.
- Six world directories under `tower\data\world_builder\worlds\` (§2), 18 MB.
  Persistent by design: they are the runtime's copies of the reconstructions.
- Linked worktree `Glasses-worktrees\wb-cv-ios-validation-win` — created and
  removed within the session.
- Nothing under `C:\`, `C:\Users\tvllo\`, or the repository other than the
  commit in §7 and this document.
