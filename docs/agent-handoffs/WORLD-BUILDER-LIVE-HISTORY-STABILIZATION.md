# World Builder — live / history / lifecycle stabilization (2026-09-06 → 07)

**Lane:** World Builder stabilization after the first physical iPhone + glasses
test of the combined stack.
**Branch:** `fix/world-builder-live-history-lifecycle-v1`
**Worktree:** `C:\Users\tvllo\Projects\Glasses-worktrees\wb-live-history-fix`
**Base:** `integration/wb-cv-ios-validation-v1 @ 12e4f1e` (`b10ab36` was the
tested head when the lane opened; a concurrent session landed `12e4f1e` — the
atomic-replace retry — while this lane was reading, and this lane is based on it
rather than duplicating it).
**Design record:** `tower/docs/superpowers/specs/2026-09-06-world-builder-live-history-lifecycle-design.md`
**Next phase:** `docs/agent-handoffs/WORLD-BUILDER-ROADMAP.md` (recognizable
room reconstruction — documented, deliberately not started here).

Nothing below claims a phone step: no iPhone was connected to this lane. §12
is the Mac checklist and §13 the next physical test.

---

## 1. The physical test, as the disk recorded it

World `fcbca9e90b244785bdb671530b33c6a5`, session `158ef0efb5e5416b87d6faa8f5c28e55`,
capture `7febdae82af84fccaf6ab94e71aa8b65`. **Preserved untouched** in the
canonical world root; nothing in this lane reads it except read-only scripts.
Sources: the world's own files, the live monitor
`Glasses-scratch\wb-validate\logs\live-world-fcbca9e9.jsonl`, the CV Lab
watcher `cvlab-phone-8000.jsonl` (1 Hz Tower samples incl. child count),
`capture.json`. Local time.

| When | What | Evidence |
|---|---|---|
| 20:14:02 | Tower restarted by the operator (CV Lab run id `8f42e16dc7ac`) | watcher: refused 20:13:52–20:14:01, running 20:14:03 |
| 20:14:16.05 | `stream_start`; capture opens; builder attached (Tower children 0 → 2: a venv launcher pair) | `capture.json`, watcher |
| 20:14:16.2 / 17.46 | world created; session started; `LOCK {"pid": 19604}` | `world.json`, `session.json`, `LOCK` |
| 20:14:38 → 20:17:06 | five background solves, each ~30 s (children 2 ⇄ 4); solutions merged into interim rebuilds | watcher; `manifest.global_solve.solved_at` |
| 20:17:10.4 | background solve #6 launched | `sources.json`, `camera.json` mtimes |
| 20:17:11.66 | interim rebuild persisted: 463 keyframes, 20,258 points, the 20:17:06 solution merged | `derived/manifest.json` |
| 20:17:12.32 | keyframe 467 accepted — last journal event | `events.jsonl` |
| 20:17:12.34 → 12.80 | the next rebuild wrote `session.json`, `edges.jsonl`, `world.json`, `poses.json` (467 poses) | mtimes |
| **20:17:12.8–13.9** | **`points.json` never rewritten; builder gone** (children 4 → 0 at the 20:17:14 sample) | mtimes, watcher |
| 20:17:33.04 | the wearer's Stop; capture `end_reason: stop`; nothing left to finalize | `capture.json` |
| 20:17:44.8 | the **orphaned** solve child wrote `solution.json` (`final: false`, 17,674 points) — never merged | `solve.log`, mtimes |
| after | `LOCK` names a dead pid, `ended_at: null`, no `session_stopped` → lifecycle `failed` → phone: **"World building failed"** | `results/world_builder.py::_lifecycle` |

Transport was healthy (2,270 frames, 0 drops, 0 errors) and every background
solve landed. The builder died **21 s before Stop**, not during finalization.

## 2. Root causes

**Immediate cause of the death** (confirmed by reproduction in this lane):
`write_json_atomic` → `os.replace` onto `points.json` raised
`PermissionError(13, 'Access is denied')` because a reader held the destination
past the 60 ms retry budget. With a reader holding the file 150 ms, the old
budget failed **28 of 30** replaces. `12e4f1e` (concurrent session) raised the
budget to 2 s with backoff and made an interim rebuild's `OSError` non-fatal;
this lane keeps that and builds on it.

**Cause of the state the user saw**: the builder had no exception safety,
no stop channel and no finalization record. Any death between `observe()` and
`stop_session()` left a lock with a dead pid, a `t=0` record and no
`session_stopped`, which the Tower could only report as `failed`, and which
iOS rendered with no world at all — although 463 keyframes of geometry and a
working render existed.

**Adjacent causes, proven but not the trigger this time:**
- the 10 s shutdown grace against a 30–135 s final solve (the builder was
  never *asked* to stop);
- `sys.executable` is a venv launcher: every worker was a process pair, the
  supervisor held the launcher's pid, and the launcher's job object has
  `SILENT_BREAKAWAY_OK`, so grandchildren (solve children) survived every
  terminate — the orphan at 20:17:44;
- the builder attached to **every** capture regardless of cartridge, so CV Lab
  sessions needed `TOWER_WORLD_AUTOBUILD=false` and a Tower restart.

## 3. What changed (Tower)

| Commit | Change |
|---|---|
| `61e0c4a` | **Builder lifecycle.** `StopRequest` with two levels (stdin EOF = soft: stop observing, close as `interrupted`, skip the final solve, final build; SIGBREAK/SIGTERM/SIGINT = hard: kill solve child, close if open, final build). `try/except/finally` around the walk: exceptions close the session as `error`, record finalization interrupted, still build, exit 1; `finally` reaps solve children and releases the lock. Lock held through finalization (`stop_session(hold_lock=True)` / `release_world()`), `finalization` block on the session record, LOCK carries process `created_at`, one `store.lock_holder()`. Final solve runs as a child; `BackgroundSolver.wait()` terminates instead of abandoning; children spawned with `stdin=DEVNULL` (a child inheriting the supervisor's stop pipe stalled 90 s at 0 CPU behind the builder's blocked ReadFile — measured); the stop-pipe watcher reads the raw fd so a clean exit no longer aborts in interpreter shutdown. |
| `f49d73d` | The same stdin-watcher abort fixed in `object_memory_session.py` (latent: a producer finishing normally with the pipe open exited non-zero). |
| `03fa165` | **Process ownership + gate.** `tower/process_ownership.py`: one-process spawn (`sys._base_executable` + `__PYVENV_LAUNCHER__`), per-worker Job Object (`KILL_ON_JOB_CLOSE`, no breakaway), `terminate_tree`. Supervisor: `request_stop(name)` (soft, returns at once, worker stays registered), `_ask_to_stop(send_signal=)`, `WorkerSpec.stop_grace_seconds` (builder 30 s). `CartridgeSession.stop_policy` (`terminate` / `request`). `main.py`: builder spec gated on a `world_builder` `CartridgeSession` (`POST /cartridges/world_builder/session/{start,stop}`), `stop_via_stdin`, `--stop-on-stdin-close`; boot log names the gate and logs the world root absolute with a warning when it holds no `worlds/`. |
| `abb7831` | **Status channel + listing.** Contract `world_builder.status/2026-09-06`. Lifecycle `finalizing` (live lock after `session_stopped`, `build_in_progress: true`), `interrupted` (dead lock; `error`/`interrupted`; finalization left pending), `ready` (finalization complete); `model_state` gains `interrupted`; `lifecycle.finalization`; top-level `selection` (`live` / `finalizing` / `latest` / `pinned` / `none`, excluded from the revision). `GET /worlds` sessions gain `state`, `keyframes_journaled`, `finalization` (id unchanged, additive). Contract docs updated (`CARTRIDGE-RESULTS.md`, `WORLD-BUILDER-WORLDS.md`, `WORLD-BUILDER-IOS.md`, `TOWER-UNIFIED-CARTRIDGES.md` §4). |
| `e0c7a1c` | `scripts/cartridge_switch_soak.py` — WB ⇄ CV Lab on one Tower, repeatedly (§8). |
| `42f7bf5`, `2440ec3` | Design record; roadmap for the next phase. |

Tower files changed: `scripts/world_build_session.py`, `scripts/object_memory_session.py`,
`scripts/cartridge_switch_soak.py` (new), `tower/process_ownership.py` (new),
`tower/capture_workers.py`, `tower/cartridge_session.py`, `tower/main.py`,
`tower/results/{world_builder,world_builder_library,contracts}.py`,
`tower/world_builder/{engine,records,store}.py`, plus tests (§9) and docs.

## 4. Lifecycle after this lane

    walk ──► Stop ──► capture closed ──► builder: stop_session(hold_lock)  [record: ended_at, finalization pending]
                                          ├─ background solve terminated if still running
                                          ├─ final solve as a CHILD (hard stop kills it; soft stop skips it)
                                          ├─ final build (merges the newest solution)
                                          ├─ mark_finalization(complete | interrupted, final_solve=…)
                                          └─ release lock → exit 0 → supervisor reaps
    on the wire: receiving → finalizing (live pid, build_in_progress: true) → finalized (ready)

    leaving World Builder (soft stop): observing builder → interrupted + final build (seconds);
                                       finalizing builder → allowed to finish
    Tower shutdown (hard stop):        builder wraps up within one build (grace 30 s); Tower death kills the job
    crash / kill:                      lock with dead pid + record → `interrupted`, geometry still served

## 5. History root cause and fix

The read-only trace (`Glasses-scratch\wb-live-history\history-trace\REPORT.md`)
generated the real 162-world payload through the route's own code: **49 KB,
67 ms, zero decoder violations** against the Swift structs. The data and the
decoder were not the reason the phone could not browse. The most likely reason
is the Tower's HTTP surface being **unreachable or stalled when tried**: port
8000 changed hands three times during the session (Tower restarts at 20:13:52
and 20:30:45), and during the CV Lab half the 1 Hz watcher saw `GET /cv-lab`
time out for 36 s, 21 s and 13 s stretches (inference on the event loop, a
known CV Lab item, out of scope here). The picker's single 10 s request then
fails closed with one grey footnote and no retry.

Real usability defects found and fixed on the listing/picker path: 96 of 162
worlds are session-less shells offered as primary rows; 29 abandoned sessions
read "still open"; titles were hex ids; no loading state, no retry, no Release
logging. Tower side: `state`, `keyframes_journaled`, `finalization`; iOS side:
§6. A mis-resolved `TOWER_WORLD_ROOT` (relative `.env` path, wrong cwd) used to
be indistinguishable from "no worlds yet"; the boot log now says so.

## 6. iOS (Swift, written on Windows, NOT compiled)

Commit `0ade6a5`. Files: `WorldSelection.swift` (new: `WorldSelection`,
`WorldSelectionMode`, `WorldFinalizationReport`, `WorldRecentReference`),
`WorldBuilderSessionController.swift` (new), `WorldListingPresentation.swift`
(new), `TowerWorldBuilderClient.swift`, `WorldBuilderClient.swift`,
`WorldModel.swift`, `WorldSession.swift`, `WorldCanvasView.swift`,
`WorldBuilderWorkspaceView.swift`, `WorldLibrary.swift`, `WorldPickerView.swift`,
`CartridgeClient.swift` (one comment); tests in `WorldBuilderIntegrationTests.swift`,
`WorldGeometryTests.swift`, `ProductShellTests.swift`, `TowerClientTests.swift`.
The Xcode project uses file-system-synchronized groups, so the new files need
no `pbxproj` edit.

| Requirement | What the phone now does |
|---|---|
| Contract | `WorldBuilderResultContract.identifier` = `world_builder.status/2026-09-06`; `TowerCapabilities.supported` follows it; an older Tower shows the existing "contract this app does not understand" state |
| Live vs History | `TowerWorldBuilderClient.publishLastReport()`: following live, a `latest` selection presents `.idle` and publishes `recentWorld`; the canvas draws "Last saved world: <title> · <state>" with **Open** (pins it). Geometry coordinates are emitted only for a report presented with a snapshot. `WorldBuilderViewModel.geometryOwner` clears the gallery and picture target on world/session change, inspection change, and any snapshot-less state |
| Interrupted | `WorldModelState.interrupted(WorldSnapshot, reason:)`: headline "Interrupted", the Tower's reason, `WorldSummaryView`, fragments, Picture enabled when coordinates exist. `.settled` phase; gated like `.finalized` |
| Finalizing | `.finalizing(snapshot, buildInProgress:)`: spinner + "The Tower is finishing this world." only when `lifecycle.build_in_progress == true`; the old guarded sentence otherwise |
| Cartridge session | `WorldBuilderSessionController` (view-owned `@StateObject`): `start` on appear and when the Tower comes back while on screen, `stop` on disappear, through `CartridgeSessionHTTPClient(cartridge: "world_builder")`, 10 s bound; footnote under the capture control; generation-checked so a late reply cannot report `active` for a screen that is gone |
| Race C3 | `pendingSubscribeAcks`: an ack that arrives while a newer subscribe is pending is unsubscribed and retired on the spot |
| Saved worlds | `WorldListingPresentation`: dated titles ("Walk · 6 Sep 2026, 20:14") with the id as a caption; badges Building / Finishing / Complete / Interrupted / No geometry from the Tower's `state` (fallback "unfinished" from `abandoned`); keyframe count from the journal when the record says 0; session-less shells behind one `DisclosureGroup`; loading indicator; Retry; `os.Logger` failure lines |


## 7. Stale live canvas — root cause and fix

Tower: an unpinned subscription is answered with a live world if any, else the
most recently updated world on disk, and nothing said which. iOS: unpinned =
`.live`, and `WorldSessionGate` passes any snapshot through when no capture
bracket is open (permanently so in Release). So the newest saved world
hydrated into the Live canvas as `finalized`/`finalizing`, its geometry was
fetched, and the view model kept fragments and the Picture target keyed only on
`geometry.revision` — never on `world_id` — so they survived a bracket opening
and a subscription drifting to the new live world on the same subscription id
(audit races C1, C5, C7, C12, C20).

Fix: the Tower states `selection.mode`; iOS presents a `latest` selection in
live mode as `.idle` with a "last saved world … Open" reference (History),
emits geometry coordinates only for a report presented with a snapshot, and
clears the gallery on world-identity change, inspection change, and loss of
snapshot. Audit race C3 (a pin changed before its ack left two subscriptions)
is closed by attempt-tracking the ack.

## 8. Cartridge switching on one Tower (§34 of the mission)

`scripts/cartridge_switch_soak.py`, this box, own uvicorn from this worktree:

| Run | Threads | RSS | Handles | Builder during WB walk | Builder during CV Lab walk | Finalize after Stop | Sessions | Dead locks / strays | Shutdown |
|---|---|---|---|---|---|---|---|---|---|
| 3 cycles × 16 frames | 32 → 32 | +1.9 MB | flat | 1 (one process) | 0 | 1.5 s | 3 complete | 0 / 0 | clean |
| 10 cycles × 24 frames | 38 → 38 | −0.4 MB | flat | 1 | 0 | 1.8–2.0 s | 10 complete | 0 / 0 | clean |

Transition latencies (10-cycle run, `soak10-report.json`):
`POST …/session/start` 2–3 ms (23 ms on the first call), `POST …/session/stop`
2–23 ms, `cv_lab_start` arm 0.9–2.7 ms (baseline experiment), builder gone
1.8–3.2 s after Stop, Tower shutdown 0.22 s, whole run 31 s. Frames were synthetic and the world root had no calibration, so
the builder ran the unposed backend and the final solve child returned at
once; the soak measures process ownership and lifecycle, not solver time.
`--intrinsics-from` runs real solves.

## 9. Tests

New Tower tests: `test_world_builder_finalization.py` (10),
`test_world_builder_lifecycle.py` (9, real processes: normal stop, soft stop
mid-walk ×2, hard stop during a final solve, solver unavailable, solver crash,
injected exception, child start while the stop pipe is held, stop-request
semantics), `test_result_channel_lifecycle_states.py` (13),
`test_world_builder_interrupted_world.py` (3, the 09-06 state as a fixture),
`test_process_ownership.py`, `test_capture_workers_stop_request.py`, three new
cases in `test_world_builder_solve_cadence.py`, two gate tests in
`test_world_builder_autostart_e2e.py`, one in `test_object_memory_lifecycle.py`.

Changed assertions, each deliberate: `wait()` terminates rather than abandons
a slow solve child; a dead builder is `interrupted`, not `failed`; the builder
attaches only with an active World Builder session (three lifecycle/e2e tests
now open it first); `IOS_MODEL_STATES` includes `interrupted`.

**Full suite (Windows, this worktree, canonical venv):** **2641 passed, 75 skipped, 1 xfailed, 0 failed** (8 min 57 s; run twice,
identical). The previous validation's baseline was 2613 / 34 / 1 / 0 from the
*canonical* checkout. The 41 extra skips are all environmental and all
explained by where this run happened: the worktree has no machine-local
`tower/data`, so every test that reads the real world corpus (`world
3dd986b1… is not on this host`, `no world corpus at data\world_builder`,
`real capture corpus absent`) skips here and ran there; the remaining skips
are the same opt-in model-download tests (`TOWER_RUN_MODEL_TESTS=1`) as the
baseline. Passed count rose by 28 with the new tests.

## 10. Reviewers and audits

Three read-only audits ran before implementation (reports under
`Glasses-scratch\wb-live-history\`): the history trace (§5), the iOS
state/race audit (20 races enumerated; C1, C3, C5, C7, C12, C20 unsafe and
now closed; the rest confirmed safe), and the process-ownership audit
(launcher pair, silent breakaway, missing stop channel, ungated builder —
all confirmed by experiment). One implementation agent (supervisor track) was
cut off by a rate limit after landing `process_ownership.py`, the supervisor
changes and their tests; the remaining wiring was finished in this session.

Two implementation agents and one reviewer were terminated by the session's
API rate limit mid-run (7 am reset). What each left was inspected in full
before being kept: the supervisor track had landed `process_ownership.py`,
the supervisor changes and their tests (all passing; the gate, session policy
and wiring were finished in-session); the iOS track had landed every file
and test listed in §6 (read line by line in-session: every `switch` over
`WorldModelState` handles `.interrupted`; single-argument `.finalizing(...)`
constructions use the defaulted associated value; the only remaining
`2026-08-25` literals are the unchanged geometry contract and tests asserting
the old status id is no longer claimed). The Tower reviewer did not report
before it was cut off; a second independent Tower review is therefore still
owed and is listed in §12 as work for the merge, not as done.

Findings consciously not acted on: the CV Lab event-loop stall (out of scope,
§11); persisting cartridge intent across Tower restart (contradicts the
session contract's own rule); closing the World Builder gate on socket
disconnect (a generic `CartridgeSession` change, §11).


## 11. Known limitations

- A phone that crashes while World Builder is on screen leaves the Tower's
  `world_builder` session `active` until the app next enters and leaves the
  workspace; the next camera session from another cartridge would attach a
  builder once. Closing the gate on socket disconnect is a generic
  `CartridgeSession` change for a later lane.
- Soft stop while finalizing lets the final solve finish; soft stop while
  observing skips it. A wearer who leaves World Builder mid-walk gets the last
  background solution, and the record says so.
- `finalizing` on records older than this lane still means only "figures not
  final" (`stopped_unbuilt`); the payload distinguishes the two.
- One derived manifest per world (unchanged).
- CV Lab inference on the event loop can stall HTTP for tens of seconds
  (observed 20:32–20:37); `GET /worlds` and the render route ride that surface.
  Out of scope, named here because it is the most plausible reason the picker
  timed out during the CV Lab half of the session.
- The soak's default walk exercises lifecycle, not solver time.

## 12. Mac compile / test checklist

Build Debug and Release for the Simulator and `generic/platform=iOS`, run
`GlassesTests`, then the UI smoke against a Tower from this branch. Constructs
written without a compiler that deserve the first look:

1. `WorldModel.swift` — `case finalizing(WorldSnapshot, buildInProgress: Bool? = nil)`:
   an enum case with a defaulted associated value; call sites construct it
   with one argument and match it with two (`case .finalizing(let s, _)`).
2. `WorldBuilderSessionController.swift` — `@MainActor final class` holding a
   `Task { [weak self] in … }` that calls a `nonisolated struct` client
   (`CartridgeSessionHTTPClient.apply`) and then mutates `status`; the
   `Logger` interpolations with `privacy: .public`; `Status` enum with
   associated `String`s conforming to `Equatable, Sendable`.
3. `WorldBuilderWorkspaceView.swift` — `@StateObject private var session`
   initialised from an optional injected controller in `init`; `.onChange(of:)`
   two-parameter form (target 26.5, already used in CV Lab); `HelperText`
   given a computed `String`.
4. `WorldCanvasView.swift` — optional closure property `openRecent` used from a
   `@ViewBuilder`; `recentWorldLine` returning `some View` from a private func.
5. `WorldPickerView.swift` — `DisclosureGroup` inside a `List` `Section` with
   `.font`/`.foregroundStyle` applied to the group; `badgeStyle` returning
   `HierarchicalShapeStyle`.
6. `WorldListingPresentation.swift` — `nonisolated enum` with a nested
   `nonisolated struct Grouped`; `DateFormatter` built per call.
7. `TowerWorldBuilderClient.swift` — nested `struct StatusReport: Equatable,
   Sendable` inside a `@MainActor` class (must not capture actor state);
   `private(set) var recentWorld` with `didSet` publishing through a subject.
8. `WorldLibrary.swift` — memberwise inits kept in extensions after five new
   `let` properties; test fixtures that construct `WorldListingSession` /
   `WorldListingEntry` directly must pass the new fields.
9. Tests: `WorldBuilderIntegrationTests` uses a `URLProtocol` stub for the
   session controller and `MockTowerServer` payloads carrying `selection` —
   check the mock's ordering assumptions for the C3 test
   (`testAPinChangedBeforeItsAckIsUnsubscribedAndItsEnvelopesAreDropped`).

Then: `ios/scripts/contract-drift-check.py --tower http://<tower>:8000` must
report agreement on `world_builder.status/2026-09-06`.


## 13. Next physical test (pass / fail)

Set-up: Tower from this branch on the Windows box, port 8000, `.env` as before
(`TOWER_WORLD_AUTOBUILD` may stay `true` for both cartridges now). Boot log
must read the absolute world root with no "holds no worlds/" warning, and
"a builder will be attached to a capture WHILE World Builder is active".
Phone: a build of this branch (both sides speak `world_builder.status/2026-09-06`;
an older app shows "the Tower offers a contract this app does not understand",
which is the intended signal to update).

**History**
1. Open World Builder fresh. PASS: the canvas shows the ready/empty state; if a
   "Last saved world" line appears it names the newest world and says it is
   saved, not live. FAIL: fragments or figures drawn as "What the Tower builds".
2. Saved worlds. PASS: a list loads (spinner first), newest first, dated titles
   for unnamed worlds, state badges, session-less shells folded under one
   disclosure. FAIL: one grey sentence with no Retry.
3. Confirm the six migrated global-solve worlds are visible by name.
4. Open `7d31e8d7…` → metadata → Picture → orbit/pinch → Close → Back to live.
   PASS: header clears, gallery empties, ready state returns.
5. Find `fcbca9e9…`. PASS: badge **Interrupted**, keyframes ~467 (journal),
   opening it shows figures + fragments + an enabled Picture; headline says
   interrupted with the Tower's reason. FAIL: "World building failed", or a
   finished-looking world.

**New live world**
6. Start capture in World Builder. PASS: `/health` shows one `world-build`
   worker (one pid, no launcher pair); canvas shows only the new world.
7. Walk; observe rebuilds. Stop. PASS: canvas shows **finalizing** with
   "The Tower is finishing this world." and no failure for the whole final
   solve (expect 30–135 s); then the session closes, `LOCK` disappears,
   `session.json.finalization.state == complete`, `final_solve == solved`.
8. Saved worlds lists it as Complete; open it; Picture works.
9. `Get-Process` shows no `world_build_session`/`world_solve` process.

**Cartridge switching**
10. Leave World Builder for CV Lab (Tower not restarted). Start the camera in
    CV Lab. PASS: `/health` shows no `world-build` worker; CV Lab arms and
    processes frames. Return to World Builder; start capture; PASS: a builder
    attaches. Repeat three times; PASS: no Tower restart, no stale processes,
    `/health` thread/RSS not climbing.
11. Leave World Builder **during** a final solve. PASS: the builder finishes
    on its own (`/health` worker disappears when done), the world is Complete.
12. Stop the Tower (Ctrl-Break) during a walk. PASS: the builder exits within
    ~30 s with the session `interrupted`, no orphaned `world_solve` process.

## 14. Merge recommendation

**Not yet — two gates first, then yes.**

1. A Mac compiles this branch and runs `GlassesTests` and the UI smoke (§12).
   3,000 lines of Swift were written without a compiler; the previous lanes'
   Windows-written Swift compiled first time, but that is a precedent, not a
   proof.
2. The physical test in §13, at least steps 1–9. The lane changed the
   contract, the builder's process model and the cartridge gate, and every
   one of those is on the phone's path.

The Tower side is ready on its own evidence: the full suite is green, the
real 09-06 world is served as `interrupted` with its geometry on every
surface, the soak holds over ten switches, and no Tower behaviour outside
World Builder changed except the object-memory producer's exit code on a
clean finish (a fix). Merge back into `integration/wb-cv-ios-validation-v1`
with `--no-ff` once the two gates pass; nothing here needs to be squashed.


## 15. Temporary resources (filesystem policy rule 9)

- Worktree `Glasses-worktrees\wb-live-history-fix` (this branch). Persistent.
- `Glasses-scratch\wb-live-history\` — audit reports (`history-trace`,
  `ios-state-audit`, `process-audit`), pytest basetemps, soak roots and
  reports (`soak1*`, `soak10*`), reproduction roots (`repro-*`). Disposable.
- No process left running; the canonical checkout, `tower/data`, `C:\` and
  `~` were not written.
