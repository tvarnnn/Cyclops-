# World Builder live / history / lifecycle stabilization — design

**Date:** 2026-09-06 (evening, after the first physical test of the combined stack)
**Branch:** `fix/world-builder-live-history-lifecycle-v1` from `integration/wb-cv-ios-validation-v1 @ 12e4f1e`
**Worktree:** `C:\Users\tvllo\Projects\Glasses-worktrees\wb-live-history-fix`
**Evidence:** world `fcbca9e90b244785bdb671530b33c6a5`, session `158ef0efb5e5416b87d6faa8f5c28e55`
(preserved, untouched, in the canonical world root), the monitor log
`Glasses-scratch\wb-validate\logs\live-world-fcbca9e9.jsonl`, the CV Lab watcher
`cvlab-phone-8000.jsonl`, and three read-only audits under
`Glasses-scratch\wb-live-history\{history-trace,ios-state-audit,process-audit}`.

## 1. What actually happened (proved from disk, local time)

| When | Fact | Source |
|---|---|---|
| 20:14:02 | Tower restarted by the operator (run id `8f42e16dc7ac`) | CV Lab watcher |
| 20:14:16.05 | `stream_start`; capture `7febdae8…` opens; builder attached (Tower children 0 → 2) | `capture.json`, watcher |
| 20:14:16.2 / 20:14:17.46 | world created, session started, `LOCK {"pid": 19604}` | `world.json`, `session.json`, `LOCK` |
| 20:14:38 → 20:17:06 | background solve children every ~30 s (children 2 ⇄ 4), five solutions written | watcher, `solution.json` history |
| 20:17:10 | background solve #6 launched | `sources.json`, `camera.json` mtimes |
| 20:17:11.66 | interim rebuild persisted a manifest for 463 keyframes with the 20:17:06 solution merged | `derived/manifest.json` |
| 20:17:12.32 | keyframe 467 accepted (last journal event) → rebuild starts | `events.jsonl` |
| 20:17:12.34 → 12.80 | rebuild wrote `session.json`, `edges.jsonl`, `world.json`, `poses.json` (467 poses) | mtimes |
| **20:17:12.8 – 13.9** | **`points.json` never rewritten; builder gone** (children 4 → 0 at the 20:17:14 sample) | mtimes, watcher |
| 20:17:33.04 | the wearer's Stop; capture closed `end_reason: stop`; nothing left to finalize | `capture.json` |
| 20:17:44.8 | the orphaned solve child wrote `solution.json` (`final: false`, 17,674 points), never merged | `solve.log`, mtimes |
| after | `LOCK` names a dead pid, `ended_at: null`, no `session_stopped` → Tower lifecycle `failed` → phone: "World building failed" | `results/world_builder.py::_lifecycle` |

**Root cause of the death:** `write_json_atomic` → `os.replace` onto `points.json`
raised `PermissionError(13)` because a reader (the Tower web thread serving the
phone, descheduled under the solver's 18 cores) held the destination longer than
the 60 ms retry budget. Reproduced in this lane: with a reader holding the file
150 ms, the old budget failed 28 of 30 replaces. The concurrent commit `12e4f1e`
(another session) raised the budget to 2 s with backoff and made an interim
rebuild's `OSError` non-fatal. This lane builds on that commit.

**Root cause of the *state* the user saw:** the builder has no exception
safety, no stop channel and no finalization record. Any death between
`observe()` and `stop_session()` leaves a lock with a dead pid, a `t=0` session
record and no `session_stopped` event, which the Tower can only report as
`failed` — even though 463 keyframes of derived geometry and a working render
exist. The transport was healthy (2,270 frames, 0 drops); the reconstruction
algorithm was healthy (every background solve landed); the *lifecycle* failed.

**Not the cause this time, but real:** the 10 s shutdown grace against a
30–135 s final solve, and the venv-launcher process pair that lets solve
children outlive the builder (audit F2, verified: `TerminateProcess` on the
launcher kills the interpreter through the launcher's own job object, but that
job has `SILENT_BREAKAWAY_OK`, so grandchildren survive).

## 2. Decisions

### 2.1 Builder lifecycle (`scripts/world_build_session.py`, engine, records)

1. **Two-level stop request, both installed by the builder.**
   - *Soft* = stdin EOF (`--stop-on-stdin-close`): "you are no longer wanted
     for new frames". While observing: stop observing, `stop_session("interrupted")`,
     **skip the final solve**, final build, exit. While finalizing: carry on
     (finalization is bounded and owns no camera).
   - *Hard* = `SIGBREAK`/`SIGTERM`/`SIGINT`: "wrap up now". Terminate any solve
     child, stop the session if still open (`"interrupted"`), final build,
     exit. Bounded by one build.
   - The supervisor's `shutdown()` sends both (hard). The World Builder
     cartridge session's Stop sends only the soft one (see 2.4).
2. **Exception safety.** The frame loop and finalization sit in
   `try/except/finally`: an unexpected exception stops the session with
   `end_reason: "error"`, records the finalization as interrupted with the
   error text, still attempts the final build, exits non-zero. `finally`
   terminates any live solve child and releases the lock.
3. **The lock is held through finalization.** `engine.stop_session(reason,
   hold_lock=True)` keeps `LOCK`; `engine.release_world()` drops it after the
   final build. Liveness of the lock holder is therefore the truth for
   "finalizing" as it already is for "receiving".
4. **A finalization record on the session.** `session.json` gains
   `finalization: {state: pending|complete|interrupted, final_solve:
   pending|solved|skipped|failed|unavailable|null, started_at, updated_at,
   detail}`. Absent on records written before this change (parsed as `None`).
   Schema version unchanged: the field is additive and optional.
5. **The final solve runs as a child of the builder**, the same
   `scripts/world_solve.py --final --loop-detection` the background solves use,
   waited on with the stop request polled. Every solve child is tracked and
   killed in `finally`; a still-running background solve is terminated before
   the final solve touches the shared workspace.
6. **The lock names the process, not just a pid.** `LOCK` becomes
   `{"pid": …, "created_at": <process create time>}`; liveness checks compare
   both when `created_at` is present (pid reuse can no longer resurrect a dead
   builder). One helper in `store.py` answers the question for the store, the
   status producer and the listing.

### 2.2 Truthful lifecycle states on the wire

Tower `lifecycle.state` (from disk facts only):

| state | facts |
|---|---|
| `receiving` | lock held by a live process, no `session_stopped` |
| `finalizing` | lock held by a live process **and** `session_stopped` written |
| `interrupted` | lock names a dead process, or `end_reason ∈ {error, interrupted}`, or `session_stopped` with `finalization.state != complete` and no live holder |
| `ready` | stopped by `stop`, finalization complete (or a pre-finalization record whose geometry is current), no lock |
| `stopped_unbuilt` | legacy: stopped, no finalization record, geometry absent or behind |
| `idle`, `unavailable` | unchanged |

`lifecycle.finalization` carries the record; `geometry.available` says whether
a reconstruction exists regardless of state. `model_state` gains the word
`interrupted`; `finalizing` now means a live process is finishing.

A new top-level `selection` block says **why this world is on the wire**:
`{"mode": "pinned" | "live" | "finalizing" | "latest" | "none", "world_id",
"session_id", "reason"}`. `latest` is the case that produced the stale live
canvas: nothing is live and the Tower answered with the most recently updated
world. The client owns what to do with that.

**Contract bump:** `world_builder.status/2026-08-25` → `world_builder.status/2026-09-06`
on both sides. A new `model_state` word is exactly the kind of change the
dated identifier exists for: an older app must be told to update, not shown an
undecodable failure.

`GET /worlds` (additive, contract id unchanged): per session `state`
(`receiving|finalizing|complete|interrupted|unbuilt`), `finalization`,
`keyframes_journaled` (journal length, so an interrupted record does not read
as zero keyframes); `abandoned` kept.

### 2.3 Live versus History on the phone

- Decode `selection`. Following live (`inspection == .live`), a `latest`
  selection is **not** the live world: the canvas shows the ready/empty state
  plus a "last saved world" reference (name, time, state) with an Open action
  that pins it (= History). Geometry for a `latest` selection is never drawn in
  Live. `live`/`finalizing` selections are the current world; `pinned` is
  History, labelled as such.
- Geometry is keyed by world identity: the view model clears the gallery when
  the incoming coordinates name a different world than the one drawn, and on
  every inspection change. The retired-subscription filter stays.
- `interrupted` renders as its own state — headline, the Tower's reason, the
  summary rows, the fragments and the Picture button when geometry exists —
  never as "failed" and never as a finished world.
- The picker becomes usable at 160+ worlds: dated titles when unnamed, state
  badges, journal keyframe counts, empty (session-less) worlds collapsed behind
  one disclosure, a loading state, a Retry, and failures logged in Release.

### 2.4 Cartridge ownership on a long-lived Tower

- The builder spec gets a **gate**: it attaches to a capture only while the
  `world_builder` cartridge session is `active`. The generic
  `cartridge_session.control/2026-08-27` surface already exists and is
  cartridge-blind; `POST /cartridges/world_builder/session/{start,stop}` is
  registered exactly like Object Memory's. `TOWER_WORLD_AUTOBUILD` keeps its
  meaning ("a builder may run at all"); it is no longer the way to switch
  cartridges.
- The phone sends `start` when the World Builder workspace appears (and again
  when the socket comes back while it is on screen) and `stop` when it
  disappears. Nothing else on the phone starts or stops a builder.
- `CartridgeSession` gains a stop policy. Object Memory keeps `terminate`
  (ask, wait 3 s, terminate). World Builder uses `request`: close the worker's
  stdin and return; the worker stays registered until it exits on its own and
  is reaped. A builder that is finalizing is therefore allowed to finish; one
  that is still observing interrupts within a build. `shutdown()` is unchanged
  in shape and remains the hard path.
- **Process ownership on Windows.** Workers and solve children are spawned as
  one process each (`sys._base_executable` with `__PYVENV_LAUNCHER__`, verified
  to give a venv-aware interpreter), so the pid the supervisor holds, the pid
  in `/health`, and the pid in `LOCK` are the same process. Each worker is
  placed in a supervisor-owned Job Object with `KILL_ON_JOB_CLOSE` and without
  silent breakaway, so a Tower that dies takes its builders and their solve
  children with it, and a terminate kills the tree. On POSIX the tree is
  killed through `psutil`.
- `WorkerSpec.stop_grace_seconds` per spec; the builder gets 30 s at shutdown
  (it wraps up in one build once asked; the grace is the bound, not the wait).

### 2.5 What is deliberately not done here

- No dense reconstruction, no new solver work, no change to the fast path,
  acceptance thresholds, redaction or the sparse pipeline.
- No change to Object Memory's session semantics or CV Lab's runtime.
- No persisted cartridge intent across a Tower restart (unchanged rule).
- Inference on the CV Lab event loop (a known CV Lab item) is out of scope; it
  is named in the handoff because it is the most plausible reason `GET /worlds`
  timed out during the CV Lab half of the physical session.

## 3. Verification

- Tower: unit tests for every new state and record; a real-process builder
  test that closes stdin mid-walk and asserts `session_stopped`, no lock, a
  manifest and `finalization.state == interrupted`; a real-process test that
  a hard stop during a (stubbed, long) final solve child kills the child and
  still writes a manifest; a process-tree test on Windows that a terminated
  worker's grandchild is gone; the gate test (no builder without an active
  World Builder session; one with); the full suite.
- A fixture derived from `fcbca9e9…`'s *state* (not its data) pins
  `interrupted` + `geometry.available` + render 200.
- A cartridge-switch soak against a real Tower process (World Builder ⇄ CV Lab,
  repeated), sampling descendants, threads and RSS, asserting no stale
  builders or solvers and no lock naming a dead pid after the last Stop.
- iOS: decoder, view-model and picker tests written on Windows; compile and
  run on a Mac before the next physical test (checklist in the handoff).
