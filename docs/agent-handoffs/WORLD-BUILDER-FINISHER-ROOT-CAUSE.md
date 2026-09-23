# The finisher's stall, reproduced on demand, and what three reviewers broke

The second Windows/Tower pass on the Mac/iOS validation of the live-world
visualization (`WORLD-BUILDER-LIVE-WORLD-VISUALIZATION-MAC-VALIDATION.md`,
T1–T5). It follows `WORLD-BUILDER-FINISHER-REMEDIATION.md` (`ceaf72b`) and
**corrects its root cause**: the stall was not a race, and the fix that
shipped with it was partial.

| | |
|---|---|
| **Branch found on origin** | `world-builder/live-world-visualization-v1` @ `ceaf72b` — not `85f31c2` as the brief expected. A first Windows pass had already integrated the Mac commits and pushed five commits of its own (`12730a2` … `ceaf72b`). This pass started from there rather than redoing it. |
| **Mac commits integrated** | `0885cfa` (four iOS fixes) and `85f31c2` (the validation handoff), fast-forwarded onto the same branch by the first pass; untouched here. |
| **Worktree** | `C:\Users\tvllo\Projects\Glasses-worktrees\wb-live-visualization-v1` (pre-existing). The canonical checkout was not used for lane work; its three untracked files (`WORK-SAMPLE.pdf`, `WORK-SAMPLE.tex`, `orb_vocab_stella.fbow`) were not touched. |
| **Scratch** | `C:\Users\tvllo\Projects\Glasses-scratch\wb-finisher-remediation-2026-09-23\` (§10) |
| **Verdict** | §12 |

---

## 1. The stall, exactly

World `2f44716237544569b5f2faf782d9f877`, session
`cb30880107eb4e4bae7c53f821399549`, 385 keyframes. The live process was
dumped with py-spy by the first pass before it was stopped
(`Glasses-scratch\wb-finisher-forensics-2026-09-22\pyspy-finisher-27264.txt`):

```
MainThread                               world-builder-stop-watch
  RtlEnterCriticalSection  (ntdll)         NtReadFile / ReadFile
  "perror"                 (ucrtbase)      "fread_nolock_s" / read  (ucrtbase)
  libscipy_openblas-197ee2fc….dll          Py_read
  LdrLoadDll / LoadLibraryExW              wait_for_close (world_build_session.py:529)
  <module> scipy/linalg/blas.py:247
  … moge/model/v2.py → dense.py:509 _load → surfacify → finish
```

**The mechanism.** The Tower stops the finisher by closing its stdin, and the
finisher noticed with a daemon thread parked in `os.read(0, 1)` for the whole
run. `os.read` is the UCRT's `_read`, which holds **descriptor 0's lock** for
as long as its `ReadFile` blocks. The OpenBLAS that numpy and scipy ship
statically links libgfortran, whose load-time constructor (`init_units`)
calls a UCRT function on descriptor 0 — and waits for that lock. Loading
scipy's OpenBLAS for the first time while the watcher is parked therefore
waits until stdin closes: 0% CPU, no traceback, which is exactly what was
seen for 95 minutes. The Tower only closes the pipe to stop the finisher, so
the hang ended when something stopped it, and the finisher then gave its
attempt back ("attempt given back: stopped (stdin-closed)").

A second path, found here: pycolmap's bundled `libgfortran-5.dll` links
`msvcrt.dll`, not the UCRT. Its `init_units` reaches `fstat64(0)` →
`PeekNamedPipe` on the stdin pipe, and Windows serialises calls on a
synchronous file object behind a pending synchronous `ReadFile` — blocked in
the kernel (`ZwFsControlFile`).

**It is deterministic, not a race.** Every case was a fresh interpreter
spawned as the Tower spawns one (base interpreter + `__PYVENV_LAUNCHER__`,
`stdin=PIPE` held open, `CREATE_NEW_PROCESS_GROUP`), with the old watcher
armed (`…\repro\child.py`, `parent.py`, `out\pyspy-*.txt`):

| After the old watcher parks | Result |
|---|---|
| `import numpy` | **hangs** until the pipe closes; identical stack in `libscipy_openblas64_` |
| `import scipy.linalg` | **hangs** until the pipe closes |
| numpy + scipy + cv2 + torch + CUDA preloaded (the `ceaf72b` prewarm), then `import pycolmap` | **hangs** — the kernel path above |
| same prewarm, then moge / open3d / skimage / a CUDA conv | completes |
| no watcher, `import scipy.linalg` | 0.2 s |
| **new watcher**, `import numpy` / `scipy.linalg` / `pycolmap` | 0.05 / 0.25 s / completes; a stop is seen in ≤ 0.26 s |

**And on the real capture.** A copy of the world rebuilt from the forensic
snapshot's pre-remediation records (§7), run through the unfixed code path
(old watcher, no prewarm), froze ~10 s in at **1.61 CPU-seconds** — the stuck
process read 1.64 — with the same native stack and the same DLL hash
(`libscipy_openblas-197ee2fc…`). Closing stdin released it, and it left the
same ledger line as the incident.

**Why the first pass could not reproduce it.** Its minimal repro read a
stdin that was not a live pipe, so `os.read` returned at once and nothing was
ever held. Its gated "real spawned worker" test uses an empty root, where the
finisher exits before it arms a watcher. Both are corrected in the test
docstrings.

**What that means for `ceaf72b`'s fix.** `tower/native_prewarm.py` loaded the
two offenders that had been seen before arming the watcher. That is correct
for those two and does nothing for the rest: pycolmap, in the same venv, still
hangs after the whole prewarm list.

## 2. The root-cause fix

`tower/stdin_stop.py`, used by `world_build_session.py` (and so by the
finisher) and `object_memory_session.py`:

- **Windows, stdin a pipe:** no read is ever left pending. A daemon thread
  calls `PeekNamedPipe` every 0.2 s. Each call returns at once, so nothing
  waits behind it. It fails with `ERROR_BROKEN_PIPE` when the parent closes
  the write end.
- **Windows, stdin not a pipe** (a manual run): `_winapi.ReadFile` on the OS
  handle, which bypasses the UCRT descriptor lock.
- **POSIX:** unchanged.

The meaning is unchanged: the close is the request, a written byte is the
request, and an unreadable stdin is the request.

The prewarm is kept as belt and braces, and its docstring now says it is not
the fix. In the finisher it moved **after `record_attempt`**, so a hang inside
it is a counted attempt (§4, H2).

Proof on the real capture: the exact 2026-09-22 shape (watcher armed, pipe
held open, **no prewarm**, moge → scipy loaded after arming) completed.
Surface `ok`, appearance `ok`, photographic `complete`, in 502.6 s.

## 3. Containment: nothing can hold "Improving" for ever

`ceaf72b` bounded forgiveness across Tower restarts. Within one Tower uptime,
two gaps remained:

- A hung finisher had a live pid, so it read `running` until the next
  restart.
- The chore ran once per Tower **start**. Owed work that a walk interrupted,
  the second owed world past `--max-worlds 1`, and a dead run all waited for
  the next restart, and read "Improving" throughout.

`tower/main.py` `_BackgroundChore` now does three things:

- **Watches the run.** Every 15 s it samples the whole process tree's CPU.
  - A tree that used **< 1 CPU-s in 600 s** is stalled. The deadlock used ~0;
    a working build uses thousands. This is not a timeout: a long build that
    keeps working is never touched, however long it takes.
  - A stalled run is killed **hard**, with no polite pipe close first and no
    SIGTERM on POSIX. Both of those let the finisher forgive the attempt, so
    a stall **counts** against the ledger's bound.
  - A tick that arrives more than 4× late (machine sleep) resets the window.
  - A grandchild exiting resets the baseline.
- **Runs owed work at the next idle moment.** Idle means no stream held, no
  capture worker alive, and 120 s since the last yield.
  - `ws.py` holds the chore from `stream_start` until `stream_stop` or
    disconnect. A connection that never streamed cannot postpone it.
  - The finisher exits **3** when more worlds are still owed (run again now).
  - It exits **4** when another writer holds a world (retry after a backoff).
  - Any other failure retries after a backoff of 60 / 300 / 900 / 3600 s.
  - A child that survives its own kill blocks further runs until it is gone.
- **Reports itself.** `/health.background_chore` shows state, pid,
  `seconds_since_progress`, the backoff, and the last outcome (including
  `stalled`).

The Tower passes `--max-forgiven 25`. The finisher's own default of 5 was
sized for once-per-start runs; with idle re-runs, a reviewer showed ordinary
walks retiring a recoverable world to `failed` after eight preemptions.

**On the real capture, through the real chore (90 s window for the test):**

1. The first run deadlocked (old watcher).
2. At 100 s it was found stalled (1.59 CPU-s, 90.5 s quiet) and killed. The
   ledger kept the attempt: `attempts 1, forgiven 0`.
3. After a 15 s backoff the chore re-ran the finisher, which completed.
   Photographic `complete` at 590 s.

## 4. Lifecycle, viewer and recovery: what the reviews found and what changed

Three independent reviewers were given the full diff since `85f31c2` and the
three adversarial questions. Every finding was reproduced by a probe test, and
every fix below has a test that fails without it.

| # | Finding | Verdict | Fix |
|---|---|---|---|
| F1/A2 | Between the surface's `ok` record and the appearance's `running` record, the session read `{surface: ok}`: "Saved". A builder dying there left a grey mesh saying "Saved" for ever, and the finisher would never select it. The `ceaf72b` tests faked a state the builder never writes. | CONFIRMED by two reviewers | The appearance `running` is recorded **before** the surface `ok`. A test drives the real `final_surface_stages` and snapshots both surfaces after every write. |
| F2 | A `surface_pipeline` that failed to import turned every **historical** world `unobservable` ("Finishing"), silently. | CONFIRMED | Imported only when a `running` status actually needs judging. |
| F4 | `unobservable` was never logged. | CONFIRMED | Rate-limited warning with the traceback. |
| F3 | The finisher's world lock relabelled a finished sibling session "Improving" for the whole 6–16 min run. It also made the panel say `interrupted` where the row said `complete`. | CONFIRMED | The lock is evidence only for a session that is open or has its finalization `pending` — the rule `session_build_running` already used. |
| A3/F5 | A momentary appearance refusal (its lock busy, the redaction label moving under the build) was recorded as a **failed** photographic build: terminal, never retried. | CONFIRMED | Those refusals carry `retryable` and are recorded `stopped`, so they are owed and bounded by the ledger. Refusals about the session (open3d missing) still fail. |
| B3 | A Tower with the appearance switched off rebuilt the surface again and again under a stale `appearance: stopped`. | CONFIRMED | `final_surface_stages` records the appearance as "not requested" itself, so the finisher does too. |
| B2 | One malformed ledger entry raised out of `survey` and stalled every owed world on the Tower. An unreadable ledger made the world "Improving" for ever. | CONFIRMED | Entries are validated and `survey` isolates each session. An unreadable ledger is moved aside (not deleted) and counting restarts from the next attempt. |
| H1 | Forgiveness bound vs. idle re-runs (§3). | CONFIRMED | `--max-forgiven 25` from the Tower. |
| H2 | A hang in the prewarm came before `record_attempt`, so it was never counted and retried for ever. | CONFIRMED (mechanism) | The warm moved after the count. |
| M3 | On POSIX the stall kill sent SIGTERM, and the finisher's handler forgave the attempt. | CONFIRMED | `terminate_tree(hard=True)`. |
| M4 | A finisher that survived its kill was forgotten. | PLAUSIBLE | Tracked, and it blocks new runs. |
| L5/B4 | Exit 0 while a foreign writer held a world, or a retire that could not take the lock, left the work waiting for the next walk. | CONFIRMED | Exit 4 (`EXIT_WAITING`), retried after a backoff. |
| C2 | A WebGL context restore whose data fetch failed ended the viewer's only poller on an unchanged revision, so it never upgraded. | CONFIRMED (trace) | Once `buildGL()` succeeds, the failure is followable: `bootFailed` is set and the revision is forgotten. A GL rebuild failure stays terminal. |
| (own) | Every WebSocket disconnect restarted the quiet period, so a client reconnecting every minute would have postponed owed work for ever. | found in self-review | `release` is a no-op for an owner that held nothing. |

**Accepted, not fixed, with reasons (see §11):**

- a hung **builder** (not the finisher) has no watchdog;
- a hang that holds the GIL while spinning in native code may use more than
  the stall threshold;
- a lock file whose pid is not an integer;
- the `{surface: failed, appearance: stopped}` shape, where the two judges
  disagree harmlessly (the ledger is already exhausted, and retire converges
  it);
- `world_finalize.py` rescuing a mid-walk crash without setting `ended_at`
  (the finisher then refuses it as `never-stopped`);
- the iOS items (§11).

## 5. T2–T5 against the brief

- **T1:** root cause §1, fix §2, containment §3.
- **T2 (Saved flicker):** closed at both boundaries. The pre-release mark is
  from `ceaf72b`; the surface → appearance gap is F1, fixed here. The real
  sequence never reads "Saved" before the appearance, on either surface.
- **T3 (failed claims success):** on the Tower the `photographic` block says
  `failed`, and momentary refusals are no longer failures (A3). **On the
  phone, a failed build still reads "Saved"**: iOS does not read the block
  (§11).
- **T4 (probe fails open):** `unobservable` → `finalizing`, now logged (F4),
  and a broken import no longer relabels history (F2).
- **T5 (startup 404):** the `ceaf72b` boot fix is kept, and the context-restore
  path is fixed too (C2). A real 404 → 200 on the manifest was observed across
  a rebuild (§7). The page itself can only be rendered through the iOS scheme
  handler (CSP `connect-src glasses-world:`).

## 6. Tests

- New:
  - `tests/test_stdin_stop.py` (9). On Windows it forms the deadlock with the
    old watcher (a control that skips, rather than passes, if the host stops
    reproducing it) and refuses it with the new one: numpy, scipy.linalg and
    pycolmap.
  - `tests/test_world_builder_finisher_chore.py` (26).
  - `tests/test_world_builder_photographic_boundaries.py` (14).
  - New cases in `test_world_builder_finish_pending.py`.
- Updated deliberately, reasoning in each docstring:
  - the record-order and "unwanted appearance" tests in
    `test_world_builder_appearance.py`;
  - the finisher warm-order tests;
  - the dead-lock honesty test;
  - the context-restore page test;
  - the one-world exit code.
- **Mutation-checked:** 36 mutants across the watcher, the chore, the
  finisher, the lifecycle and the page. All killed. Three survived their first
  test and the tests were strengthened until they died:
  - the sleep rule: a respawn masked the stall;
  - the `stream_stop` release: the disconnect release masked it;
  - the pipeline's `retryable` flag: the boundary tests fake the pipeline.
- **Full Tower suite on the final tree:** see §12. It ran in a junction-free
  mirror of the worktree, because an SSH session cannot traverse the
  worktree's `.venv`/`data` junctions ("untrusted mount point").
- **Mac (POSIX) run of the new suites:** 77 passed, 4 skipped (Windows-only).

## 7. Real-data evidence

The fixture is world 2f447162 rebuilt in scratch exactly as it stood before
the first remediation. Inputs (keyframes, solve, derived) were copied from
the live world per the snapshot's manifest; the records (`session.json`,
surface `status.json`, ledger) came from the forensic snapshot. Nothing under
the live data tree was written (`realdata\make_fixture.py`).

| Run | Code | Outcome |
|---|---|---|
| A0b control | old watcher, no prewarm | froze at 1.61 CPU-s within ~10 s; py-spy stack identical to the incident; closing stdin → `stopped`, attempt forgiven |
| A1 | new watcher, no prewarm, pipe held open | **complete, 502.6 s** |
| B | full Tower on :8011 (pre-review code), a walk simulated 90 s in | `finalizing` held throughout. Yield → re-run 130 s after the walk ended → manifest 404 → **200** → representation `appearance` → `finalized`/`complete`. **0 moments** of Saved-while-unfinished over 678 s |
| B2 | same, **final code** | **0 moments** of Saved-while-unfinished over 664 s. Yield at 94 s → re-run at 243 s → surface published at 560 s while still `finalizing` → at 655 s `finalized`/`complete`, manifest **200**, `appearance`. The chore re-ran in 420.7 s and went idle |
| C | real `_BackgroundChore`, old-watcher first run | stall found at 100 s → killed, **counted** → retried → **complete**, 595 s |
| Census | all 70 real sessions / 166 worlds, deployed vs branch code | **0 differences** on the row or the status channel; photographic 1 complete / 2 never_recorded / 67 unattempted; finisher owes 0; listing 0.15–0.20 s |

The live Tower (still on `main` @ `83534e2` when this pass began) serves
2f447162 photographically: row `complete`, manifest 200,
`/render/revision` → `appearance`. The first pass finished it on 2026-09-22.

## 8. Timing (for the optimisation campaign — observed, not acted on)

A1 on the real 385-keyframe walk, 5-s samples, cold model load included:

| Stage | Wall | Notes |
|---|---|---|
| depth | ~70 s | GPU to 99%, 3.9 GB VRAM |
| transients | ~157 s | the largest stage; ~16 cores busy, GPU 74–99% |
| consistency | ~31 s | |
| fuse | ~10 s | |
| mesh | ~46 s | RSS peaks ~5.0 GB |
| pack | ~61 s | **GPU idle (1%)**; RSS ~5.7 GB — a CPU-only stage |
| appearance | ~97 s | GPU to 98% |
| **total** | **~503 s** | ~4,000 CPU-s; ≤ 4.9 GB VRAM; ≤ 155 threads |

The same walk took ~400 s in the first pass and 436 s for the second chore run
in B. The spread is load on the host (a test suite ran beside A1 and B).
Obvious targets for later: transients (~31%), and pack as a CPU-bound tail
with the GPU idle.

## 9. Repository changes

Four commits on `world-builder/live-world-visualization-v1`, on top of
`ceaf72b`:

| Commit | What |
|---|---|
| `598d8a2` | `fix(world-builder): the stop watcher no longer holds descriptor 0` — §1, §2 |
| `c788c45` | `fix(world-builder): a stalled finisher is killed and counted, and owed work runs when the Tower is idle` — §3, plus H1, H2, B2 and B4 |
| `87023b8` | `fix(world-builder): what three reviewers found in the photographic lifecycle` — §4 |
| this commit | this handoff, and a correction note at the top of `WORLD-BUILDER-FINISHER-REMEDIATION.md` |

`origin/main` was not touched, and neither was local `main` in the canonical
checkout.

## 10. Temporary resources

All under `C:\Users\tvllo\Projects\Glasses-scratch\wb-finisher-remediation-2026-09-23\`:

- `repro\` — the deadlock harness and its py-spy dumps.
- `realdata\`:
  - the fixture builder, the harnesses and the census script;
  - `root0`, `root0b`, `rootA1`, `rootB`, `rootB2`, `rootC` — scratch copies
    of the world, ~80 MB each;
  - `runs\` — samples, timelines, logs, py-spy dumps, census JSON.
- `mutation\` — the mutation runner and its sets.
- `suite\` — a junction-free mirror of the worktree for the suite.
- `full-suite-*.log`, `data-dependent-tests.log`.
- `deploy\` — the launcher for the live Tower (§12) and the corpus-test
  runner.
- Two one-shot scheduled tasks, `GlassesTowerBranchStart` and
  `GlassesDataTests`, were created to run those in the desktop session and
  deleted straight after they started. Nothing is left scheduled.

On the Mac, `/private/tmp/claude-501/…/scratchpad/`: exports, diffs, reviewer
probe copies (`advA`, `agentC`, `rev4`, `review-lifecycle`).

Nothing was created at `C:\` or in the home directory, and nothing was
deleted. The first pass's forensic directory is untouched.

## 11. Known limitations, and what iOS must do

- **A failed photographic build still reads "Saved" on the phone.** iOS does
  not decode `lifecycle.photographic` or the row's `photographic` block, and
  no existing contract word can say it without misreporting the capture or the
  solve (all three reviewers agree). The smallest iOS change:
  - in `modelState` `finalized`, read `lifecycle.photographic.{state,detail}`;
  - in `WorldPresentation.stage`, map `failed` + geometry to the existing
    `.partial` stage with "Saved — the photographic version could not be
    built";
  - on the Saved Worlds row, `complete` + `photographic.state == failed` →
    "Partial";
  - `never_recorded`, `unattempted`, `complete` and an absent block keep
    today's words.
- **Other iOS items from the adversarial review (not Tower-fixable):**
  - `/render` can fall back a rung while `/render/revision` says appearance,
    and `handledRevision` is set before the rung is checked (C1). Don't
    record it when the page's rung is below `latest.representation`.
  - `didFinish` is taken as "drawn" (C3).
  - The native follower is started once and exits on a nil revision (C4).
  - Saved Worlds opens the newest session with geometry, not the photographic
    one (C5, known).
- **A hung builder** (post-Stop) has no watchdog. It holds its world
  "Improving" and keeps the chore from running until it exits or the Tower
  restarts; a restart then recovers the work. The known hang class is
  removed from the builder by §2.
- **The stall rule's blind spots.** A hang that spins the CPU (e.g. a wedged
  GPU spin-sync) is never "stalled". A hang holding the GIL may use more than
  1 CPU-s per 10 min from GIL-wait wakeups; measured ~2.4 on macOS, not yet
  measured on Windows.
- A lock file with a non-integer pid still refuses the world while the
  serving path says `owed`.
- `TOWER_WORLD_FINISH_PENDING=false` (or surface/solve off) leaves owed
  worlds "Finalizing" by configuration; the reason text says so, and iOS does
  not show it.

## 12. Verdict

**Tests on the final tree (`87023b8`), Windows:**

- **Full suite: `4222 passed, 0 failed, 80 skipped, 1 xfailed`** (19 min 15 s),
  run in the junction-free mirror.
- The skips are the opt-in model tests (`TOWER_RUN_MODEL_TESTS`) and the
  tests that read the real `data/` corpus.
- Those corpus tests (`test_world_registration`,
  `test_world_builder_registration_wiring`, `test_document_detect_corpus`),
  plus the finisher-startup file, were run separately in the owner's desktop
  session, where the junctions work: **all passed, 2 skipped**. pytest then
  crashed in its own temp-directory cleanup (`pytest-current`: access denied),
  after the results were printed.
- The previous full run on the tree before one last fix: `4221 passed,
  1 failed`. The failure was the lock-evidence rule for a session with no
  finalization record; it was fixed (§4, F3) and is covered by that run.

**The live Tower:**

- **Now running.** `:8000` runs `87023b8` from the branch worktree. It was
  started with the worktree's own `scripts\start_tower.ps1` in the owner's
  desktop session, through a one-shot scheduled task that was deleted
  immediately; the window is titled "Glasses Tower (branch
  world-builder/live-world-visualization-v1)". It replaced the Tower started
  from the canonical `main` @ `83534e2` at 2026-09-22 19:52. Nothing was
  connected to it when it was stopped.
- **Verified after the start:**
  - `/health.background_chore`: the startup run found nothing owed and
    exited 0 in 15 s; the chore is now idle.
  - `/worlds`: 166 worlds; rows 29 complete / 30 interrupted / 11 unbuilt,
    the same as before.
  - 2f447162: row and status `complete`/`finalized`, photographic
    `complete`, manifest 200, `/render/revision` → `appearance`.
- **Canonical `main` is still `83534e2` and was not modified.** A Tower
  restarted the usual way, from the canonical checkout, runs the OLD code.
  Until the branch is merged, restart from the worktree (the launcher is in
  `…\deploy\start_tower_branch.ps1`), or fast-forward local `main` to the
  branch. The fast-forward was not done here, because another Claude session
  was active in the canonical checkout: it wrote an untracked
  `README.md` there at 04:44, which was left untouched.

**Verdict: READY FOR MAC REVALIDATION.** Not the physical test yet, for three
reasons:

1. **What the phone sees changed.**
   - Lifecycle words are more honest, and a stalled world now converges
     instead of reading "Improving" for ever.
   - Momentary appearance refusals now read "Finishing" and are retried,
     where they used to read "Saved".
   - A sibling session is no longer relabelled.
   - The viewer's context-restore path now recovers.
   - All of this needs to be seen on a device, and the page renders only
     through the iOS scheme handler.
2. **T3 is still open on the phone.** A photographic build that failed reads
   "Saved" until iOS reads the `photographic` block (§11). The Tower side is
   done; the product side is not.
3. **The Mac-side viewer items from the adversarial review** (§11, C1/C3/C4)
   are for the Mac lane to judge and fix.

What this pass establishes without a Mac:

- The stall is understood from the live native stack, reproduced on demand
  on the host and on the real capture, and fixed where it lived.
- The fixed path completes the real walk (~8 min 23 s, no prewarm).
- A deadlocked finisher is now found, killed, counted and retried to
  completion through the real chore.
- A walk interrupting recovery leaves the world "Improving" and it finishes
  by itself, with 0 false Saveds across two Tower-level runs.
- The 70 real sessions read exactly as before.
- 2f447162 is served photographically by the running Tower.

The physical acceptance test is still the pass, and it has not been run:
glasses → Start → walk → Stop → Improving → photographic room → Saved →
open → recognisable bedroom → reopen from Saved Worlds.
