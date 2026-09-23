# The finisher that never finished, and the three false "Saved"s

> **CORRECTED 2026-09-23 — read `WORLD-BUILDER-FINISHER-ROOT-CAUSE.md` first.**
>
> §1's mechanism is wrong. The stall was not a loader-lock race: the stop
> watcher's parked `os.read(0, 1)` held descriptor 0, and any library that
> touches descriptor 0 at load waited for it. It reproduces every time, on
> this host and on the real capture.
>
> §2's prewarm covered only the libraries it names; pycolmap still hangs
> after it. The fix is now `tower/stdin_stop.py`.
>
> Two other statements here were superseded by the review in that document:
>
> - §3's "stage-boundary gap" was closed at the first boundary only.
> - §4's "no wall-clock timeout" still holds, but a stall is now killed and
>   counted by the Tower.

The Windows/Tower answer to the Mac/iOS validation of
`WORLD-BUILDER-LIVE-WORLD-VISUALIZATION-MAC-VALIDATION.md` (§7, T1–T5).

| | |
|---|---|
| **Started from** | `83534e2`, the branch as the Mac found it |
| **Mac commits integrated** | `0885cfa` (four iOS defects fixed), `85f31c2` (the validation handoff) — fast-forwarded, not merged, onto the same branch |
| **Branch** | `world-builder/live-world-visualization-v1` — no new branch, no new lane |
| **Worktree** | `C:\Users\tvllo\Projects\Glasses-worktrees\wb-live-visualization-v1` (pre-existing; the canonical checkout was never used for lane work and its three untracked files are untouched) |
| **Scratch** | `C:\Users\tvllo\Projects\Glasses-scratch\wb-finisher-forensics-2026-09-22\` |
| **Verdict** | §12 |

---

## 1. The stall: what it actually was

World `2f44716237544569b5f2faf782d9f877`, session `cb30880107eb4e4bae7c53f821399549`
— the owner's ~02:37 physical walk, 385 keyframes — was held by
`scripts/world_finish_pending.py` for **more than 95 minutes** against a job
that this campaign has now measured at **6 minutes 40 seconds**.

It was not slow. **It was deadlocked, and it had made no progress at all.**

Evidence captured from the live process before anything was restarted
(`pyspy-finisher-27264.txt`, and the CPU samples beside it):

- CPU total frozen at **1.64 s** across repeated 20-second samples — a delta
  of exactly zero while the process was alive and holding the lock.
- GPU 1%, VRAM idle, 47 threads, 558 MB resident, nothing growing.
- **No log line and no artifact touched** after the lock and the first
  `status.json`. The world's last real write was `dense/…/align.json` at
  02:42; the finisher added nothing in 95 minutes.

`py-spy dump --native` named it exactly — two threads, one lock:

```
MainThread (idle)                     world-builder-stop-watch (idle)
  ZwWaitForAlertByThreadId              NtReadFile
  RtlEnterCriticalSection               ReadFile
  libscipy_openblas-*.dll               Py_read
  LdrLoadDll / LoadLibraryExW           wait_for_close
  <module> (scipy/linalg/blas.py:247)     (world_build_session.py:529)
  moge/model/v2.py
  _load (world_builder/dense.py:509)
  run_depth_stage -> ensure_depth_stage
  surfacify -> final_surface_stages
  finish -> main (world_finish_pending.py:889)
```

**The mechanism.** The Tower spawns this worker with `stdin=subprocess.PIPE`
and `--stop-on-stdin-close`, which arms a daemon thread that blocks in a
synchronous `ReadFile` on that pipe. The main thread then calls
`LoadLibraryExW` on `libscipy_openblas*.dll`, which builds its thread pool
inside `DllMain`, under the Windows loader lock. A thread parked in a
blocking kernel pipe read while the loader brings up a thread-spawning DLL is
a hard hang: 0% CPU, forever, no traceback.

**This is the second time.** Object Memory hit the identical deadlock on
2026-09-06 (`frames_observed: 0` on a healthy 244-frame walk) and fixed it in
`scripts/object_memory_session.py` by warming `scipy.linalg` on the main
thread before arming the watcher. The fix was written where the bug was
found, so World Builder inherited the bug instead of the fix.

### It is a race, and this is stated plainly

Once the DLLs are warm in the OS page cache the deadlock does **not**
reproduce. A minimal faithful repro (watcher armed, parked, then
torch/scipy/cv2 imported) completed in ~2 s in *both* orderings across four
park delays, and an **unfixed** finisher run against the real world also
completed and reached the depth stage. The three observed hangs were all on
the first Tower start of the day, with the DLLs cold.

So the root cause is established from the live native stack and matching
prior art, **not** from an on-demand reproduction. That is why the fix ships
with containment beside it (§4) rather than on its own.

### Two things ruled out, with evidence

- **Not the wrong interpreter.** The venv `python.exe` is a launcher that
  spawns the base interpreter; `tower/process_ownership.py` rewrites `argv[0]`
  to `sys._base_executable` and passes `__PYVENV_LAUNCHER__` deliberately, so
  the child is one process and still sees the venv. Verified: the venv
  reports `sys.prefix` in the venv and imports torch 2.13.0+cu132 and
  pycolmap 4.2.0.
- **Not the finisher crashing the parent.** The immediate uvicorn
  `Shutting down` after `Application startup complete` is an OS console
  control event (SIGINT/SIGBREAK). In uvicorn 0.52.4 the only assignment to
  `should_exit` is `Server.handle_exit`, installed solely as a signal handler
  for `SIGINT`/`SIGTERM`/`SIGBREAK`; the startup-failure path exits before
  "Uvicorn running on" is logged, and no traceback was produced. The child is
  spawned **with** `CREATE_NEW_PROCESS_GROUP` and is stopped by closing its
  stdin and `terminate_tree`, never by a console event, so it cannot signal
  the parent. `scripts/start_tower.ps1` runs uvicorn in the *calling shell's*
  process group, so any Ctrl-C/Ctrl-Break aimed at that shell reaches it.
  A detached launch of the identical command stayed up and served for the
  rest of the session. **No Tower code requests that shutdown.**

---

## 2. What happened to the world

It was finished, on the fixed code, on the real data. Nothing was deleted and
nothing was rebuilt from scratch.

| | |
|---|---|
| surface | `ok` — depth 66.9 s, transients 155.1 s, consistency 21.6 s, fuse 7.9 s, mesh 30.9 s, snap 5.5 s, pack 35.2 s = **323 s**; 283 of 379 frames used; 1,176,110 vertices, 2,225,865 faces |
| appearance | `ok` — detector 39.5 s, provenance 2.6 s, occluders 4.4 s, exposure 8.5 s, transients 8.2 s, selection 0.9 s, encode 12.4 s = **77 s**; 355 keyframes, 128 phone-budget, 31 chunks, 23.85 MB |
| **total** | **~400 s (6 min 40 s)** against the 95 minutes it was held |

Served, verified against the running Tower:

- `GET …/appearance/…/manifest` → **200** (it was 404 before)
- `GET …/render/revision?…&viewer=appearance-1` → `representation: "appearance"`, `appearance.state: "served"`, `live: false`
- `GET …/render?…&viewer=appearance-1` → the appearance page, `wb-representation: appearance`
- listing row `complete`; status channel `ready` + `photographic: complete`

This is the **first photographic world on this Tower** — the Mac census found
zero, on 39 sessions with geometry.

---

## 3. The three false "Saved"s (T2, T3, T4) — one cause, one fix

The serving path asked one present-tense question — *is a photographic stage
running right now?* — and answered the wearer's question from it. That is
false in the gaps between stages (T2), false over a failed build (T3), and
false when the probe itself breaks (T4).

**`tower/world_builder/photographic.py`** answers the settled question
instead: *does this world still owe a photographic room?* One closed
vocabulary, consumed by **both** surfaces:

| word | meaning | effect on the reported state |
|---|---|---|
| `complete` | the appearance stage finished | unchanged (`ready` / row `complete`) |
| `running` | a stage is running under a live pid | `finalizing` |
| `owed` | unfinished, nothing working on it | `finalizing` |
| `failed` | a stage ran and failed — **terminal** | unchanged; the block carries the truth |
| `unattempted` | the stages declined (no solve, appearance off) | unchanged |
| `never_recorded` | a world from before the stages existed | unchanged |
| `unobservable` | the probe could not answer | `finalizing`, `build_in_progress: null` |

Three design points that are load-bearing:

1. **The probe is asked last, and only where the record is ambiguous.** An
   earlier draft asked liveness first, for every session — which would have
   turned all 165 historical worlds `unobservable` at once if the render
   module ever failed to import, and parked every one of them on "Improving"
   forever. A world with no stage record is judged without the probe.
2. **`failed` stays settled.** Reporting it as "Improving" would strand the
   wearer waiting for a build that is not coming — the same failure as the
   false Saved, from the other side. The world *is* saved at whatever rung it
   reached; the block says the photographic room failed, and why.
3. **`owed` is a promise, and only one thing keeps it.** A defect found and
   fixed during this campaign: the first version could say `owed` about a
   session `world_finish_pending.assess()` refuses (`not-finalized`,
   `no-final-solve`) — a world that would say "Improving" **forever** with
   nothing ever finishing it. `photographic_state` now applies the same
   precondition, so the two judgements agree by construction rather than by
   both being edited the same way on some later day.

**The stage-boundary gap before the surface starts** was closed in the
builder: `scripts/world_build_session.py` now records the surface stage
`running` **before** `engine.release_world()`, guarded on the surface being
enabled *and* the global solve having succeeded. A builder that dies in that
window now leaves the exact signature the recovery finisher selects on, where
before it left nothing and the work was lost silently.

---

## 4. Containment: a hung finisher can no longer imply progress forever

The attempt ledger already bounded retries at three — and **could never reach
the bound in the one case it existed for.** Every stop request gave the
attempt back, including the stop that ends this process at every ordinary
Tower start, so the deadlocked run left `{"attempts": 0, "detail": "attempt
given back: stopped (stdin-closed)"}` and repeated forever.

Forgiveness is now itself bounded (`DEFAULT_MAX_FORGIVEN = 5`, `--max-forgiven`).
Five free passes covers a full day of ordinary Tower starts; past that,
attempts accumulate, the existing bound is reached, and `_retire()` records
the stage `failed` — so an unfinishable world retires after nine starts
instead of never, and then reports honestly as a failed photographic build
rather than a perpetual "Improving".

**No wall-clock timeout was added.** Nothing times or kills a running build:
a long walk may legitimately take twenty minutes or more, and a timer around
a working build is the wrong instrument. The bound is on forgiven attempts —
a fact this tool writes itself.

---

## 5. The viewer (T5)

`appearance_viewer.html`'s boot path was the one code path in this subsystem
that treated a 404 as terminal — contrary to its own contract, its own
follower, and its own context-restore handler, which already did the right
thing. A manifest 404 at boot (the documented `rebuilding` window after Stop,
or any transient) threw out of `loadRevision` into the boot `catch`, called
`fail()`, and `follow()` — the page's only revision poll, started at two
sites *after* the manifest load — never started. The page then reported
`didFinish` to WebKit like a healthy page, so iOS did not show its "Try
again" control either.

A boot 404 now drops to a neutral message ("it may still be being finished")
and starts `follow()`, so the page upgrades itself when the Tower serves the
build. The wording deliberately differs from the restore handler's "no longer
served", which reads as a privacy withdrawal; the usual cause here is an
artifact still being finished.

**A second stranding was found by rendering the real page**, and it is wider
than the 404. A boot fault that is NOT an absence — a transport error, a
500, an exhausted retry — also reached `fail()` and also took the poller
down with it. It now still calls `fail()`, loudly, because the wearer should
see it; what changed is that a `bootFailed` flag lets the existing follower
keep asking. It is set in exactly one place, behind a `glReady` witness
raised only after the GL programs, the transcoder and the navigation path
are all built — so a genuinely un-followable fault (no WebGL 2, a template
this build cannot read) stays terminal and never gets a poller — and it is
cleared the instant a build opens, so it cannot resurrect a page that failed
later for a reason nothing can follow.

No new timer in either fix; the existing interval and its backoff to the
120 s ceiling are reused. The GONE rule is unchanged. Sixteen mutation
checks across the two rounds, each killed by a test.

---

## 6. Independent review, and what it changed

Two reviewers that did not write the code, briefed to falsify rather than
approve -- one on process lifecycle / locking / crash-restart / staleness /
concurrency, one on world lifecycle semantics and old-world compatibility,
carrying the three adversarial questions. They were worth more than the rest
of the campaign's testing put together, and **six of their findings changed
the code.**

| Finding | Verdict | What was done |
|---|---|---|
| The warm left a thread-spawning DLL in the very frame it was fixing: `import torch` does not attach the NVIDIA driver, the first CUDA *query* does, and that query is one line after the moge import the py-spy stack blamed. 17 DLLs including `nvcuda64.dll`. | **Confirmed by measurement.** | Fixed. `_warm_cuda()`, measured at **0.02 s**. There was no argument for leaving it out. |
| The warm ran before the survey, so every Tower start paid ~1.4 s and ~650 MB of resident torch even with nothing owed -- against a documented promise in `tower/main.py` that it would not. | **Confirmed.** Survey measured at 0.04 s, zero owed. | Fixed. Survey first, exit before warming when there is no work. |
| `unavailable` was filed under "nothing is wrong", but the pipelines write it both for a stage nobody asked for AND for one that ran and broke (ASTC/WebP encode failure, open3d missing). A broken photographic build still read **Saved**. | **Confirmed.** T3 re-entering through an un-enumerated word. | Fixed. `attempted` is the discriminator and was already on disk. |
| A crash between the journal's `session_stopped` and the record write (`ended_at: null`) read `owed` on the panel, `interrupted` on the picker, and was refused by the finisher as `never-stopped` -- Improving for ever. | **Confirmed.** | Fixed. |
| "The Tower finishes owed photographic work at its next start" is false on a Tower with the finish-pending, surface or solve settings off. | **Confirmed.** | Fixed: the word stays (the world really is unfinished), the promise does not. |
| Two numbers in my own comments were wrong: "165 reach `never_recorded`" (it is 67 `unattempted` / 2 `never_recorded`) and the appearance gap "tens of seconds" (it is sub-millisecond). | **Confirmed.** | Corrected rather than dropped. Neither changes behaviour; both would have misled the next editor. |

A seventh finding -- that `photographic_state` said `owed` where
`assess()` refuses for want of a solve -- had already been found and fixed
independently before the review landed; the reviewer's snapshot predated the
fix, and their suggested change is the one that was made. It is why the
**agreement test** now exists: the two judges drifted apart once inside a
single afternoon, so prose claiming they agree is worth nothing. The test
builds each shape on disk and asks both.

Accepted rather than fixed, with reasons:

- **A corrupt `finish_attempts.json`, or a lock file whose `pid` is valid
  JSON but not an integer**, makes `assess()` refuse for ever while this
  module says `owed`. Both are corrupt-file states that already stop the
  finisher independently of this change, and modelling them would pull the
  attempt ledger and the lock protocol -- which live in `scripts/` -- into
  the serving path. Named in the agreement test's docstring and in §10.
- **`photographic: complete` does not guarantee the ladder serves
  appearance** -- a purged or relabelled world keeps `appearance: ok` on
  record while the manifest is refused. This is now an explicit documented
  boundary in both contracts: `photographic.state` is about the BUILD,
  `appearance.state` on `/render/revision` is about what can be DRAWN, and
  the viewer already steps down correctly and captions it.
- **Nothing on the phone reads `lifecycle.photographic` yet**, so a `failed`
  photographic build still renders as plain "Saved". The Tower is now
  honest and the product is not. This is the single reason §12 asks for Mac
  revalidation rather than going straight to the physical test -- see §11.

---

## 7. Real-data evidence

Nothing here is a fixture.

- **The stuck world was finished** on the real 385-keyframe capture (§2) and
  now serves photographically.
- **Interrupted recovery was exercised for real, twice**: a run killed mid
  depth stage left `state: "stopped"`, and the next run recovered it and went
  on to complete. That is the process-death/restart path, on real data.
- **A real 404 → 200 transition** on the same manifest URL, across the
  rebuild — the exact transient the viewer fix is about.
- **All 166 worlds / 70 sessions** were evaluated through both surfaces with
  the final code. Listing distribution is **byte-identical** to before the
  change (29 complete / 30 interrupted / 11 unbuilt), **zero** row-vs-panel
  disagreements, and zero exceptions. World `2f447162` is the one session
  that moved, and it moved to `complete`.
- **The recovery pass on the fixed Tower**: 70 sessions seen, **0 owed**,
  nothing started, nothing retired — and its `assess()` agrees with the new
  `photographic_state()` about every one of them.

---

## 7a. Tests

Every confirmed defect got regression coverage, and **every new test was
mutation-checked**: the fix line was broken, the test was confirmed to fail,
the line was restored, and the test was confirmed to pass again. Two mutants
survived their first test and the tests were strengthened until they died —
both are recorded below, because a surviving mutant is the only interesting
row in a mutation table.

**The full Tower suite on the final tree: `4209 passed, 38 skipped,
1 xfailed, 0 failed` in 16 m 34 s.** Plus the cross-stack gates against the
iOS build: `contract-drift-check.py` AGREEMENT, `cross-stack-constants-check.py`
agreement, `swift-structure-check.py` clean. The status contract id
(`world_builder.status/2026-09-10`) is deliberately unmoved — every payload
change here is additive, and iOS equality-tests that id.

| Suite | Result |
|---|---|
| `test_world_builder_finisher_startup.py` (new, 11) | the warm precedes the arm in the finisher, the builder and Object Memory; an empty root neither warms nor arms; `--dry-run` never warms; the shared module reports a missing library instead of raising; plus two gated tests that spawn a real interpreter |
| `test_world_builder_photographic_lifecycle.py` (new, 29) | the settled vocabulary shape by shape; the stage boundaries never say Saved (including a whole-sequence test that walks every intermediate disk state in order); a failed build does not claim success and is not left saying Improving; a broken probe does not say Saved **and** does not relabel a historical world; the **agreement test** against `assess()` |
| `test_world_builder_library_photographic.py` (new, 8) | the same rules on the Saved Worlds row, plus row-and-panel agreement across all six states |
| `test_world_builder_appearance_page.py` (+8) | the boot 404 recovery, the boot-fault follower, the `glReady` terminal line, and a GONE-shaped 404 still terminal for the imagery |
| `test_world_builder_finish_pending.py` (+5, 51 total) | the bound on forgiveness, old ledgers still parse, the ordinary stop still forgives |
| `test_world_builder_photographic_build_honesty.py` (3 updated) | three assertions that encoded the pre-recovery-finisher belief, changed deliberately and with the reasoning written into each docstring |

Mutation checks: **9** on the viewer's second round (one survived first: a
test asserted the `glReady` witness existed but not that the follower start
was gated on it), **7** re-run from its first round, **5** on the
forgiveness bound (two survived first — an old-ledger `KeyError` that
`forgive_attempt` would have swallowed, and a `MAX_FORGIVEN = 0` that a
`range()`-driven test sailed through), **5** on the listing, and **4** on
the lifecycle module (`is_unsettled` always false; `never_recorded` → `owed`;
the `unavailable`/`attempted` split; the `never-stopped` guard). The
lifecycle mutations are the ones worth noting: they kill tests in **both**
directions — one set catches a false "Saved", the other catches the
historical worlds being relabelled into a false "Improving".

The three failures in the first full-suite run were snapshots of a tree that
was being edited while it ran; all three pass on the final tree, verified
directly.

---

## 8. Performance observation (collected, not acted on)

For the later optimisation campaign only. No speculative rewrites were done.

- Whole photographic build on this walk: **~400 s** (surface 323 s,
  appearance 77 s), plus cold model load.
- The single largest stage is **transients at 155 s** — 48% of the surface
  budget — against depth at 67 s.
- GPU reached 65% with ~3.9 GB VRAM; RSS peaked ~2.5 GB, 94 threads.
- Obvious idle gap: cold DLL/model load at process start, paid once per
  finisher run. The prewarm now moves it earlier but does not remove it.
- The prewarm itself costs **1.3–1.7 s** warm, measured over three fresh
  processes. It is paid by the builder too, at `stream_start` — and it costs
  no keyframes, because `CaptureFollower` defaults to `start_at_end=False`,
  so the builder reads the capture journal from the beginning and simply
  catches up a second behind rather than starting from "now".
- `GET /worlds` over the real 166-world root: **0.12 s before** the change,
  **0.16 s after** — the per-row settled judgement reads the record first and
  only touches disk for a session that has no stage record.
- The recovery pass on the final code surveys 70 sessions and exits
  **without warming anything** when nothing is owed, which is the common
  case; before the reviewer's finding it paid a full torch import first.
- `depth_work_pruned_bytes: 484,710,766` — half a gigabyte of per-frame depth
  work written and then deleted, which a `--keep-depth-work` run would reuse.

---

## 9. Temporary resources

All under `C:\Users\tvllo\Projects\Glasses-scratch\wb-finisher-forensics-2026-09-22\`:
the pre-remediation copy of the world's records and logs
(`world-2f447162-preremediation/`, **forensic evidence — do not delete**),
the py-spy dump, Tower run logs, the finisher report, the repro harness, the
full-suite log, and the headless screenshot. Nothing was created at `C:\` or
in the home directory, and nothing was deleted.

The worktree gained three ignored dev conveniences so it can run a Tower
against the real data: junctions `tower/.venv` and `tower/data` to the
canonical checkout, and a copy of `tower/.env`. All three are in
`tower/.gitignore` and none is committed.

---

## 10. Known limitations

- **The deadlock could not be reproduced on demand** once the DLLs were warm
  (§1). The fix is the proven prior-art ordering fix and removes the hazard
  window entirely; the containment in §4 exists because that argument, on its
  own, is not proof.
- **No physical walk was performed.** Everything here is stored capture data
  and replay.
- **The photographic page cannot be rendered outside iOS.** Its CSP is
  `connect-src glasses-world:`, so only the app's scheme handler can fetch
  the manifest. A headless-browser screenshot is therefore not available as
  evidence, by design, not by defect.
- **Saved Worlds still opens a world's newest session with geometry, not its
  best-represented one** — a known iOS item from the Mac lane, untouched here.
- `_read_ledger` treats a corrupt ledger as unreadable for the whole world,
  so neither counting nor forgiving advances. `assess()` already refuses on
  it (`ledger-unreadable`), so it is contained, but it is the one path where
  no bound advances.
- **A corrupt `finish_attempts.json`, or a lock file whose `pid` is valid
  JSON but not an integer**, makes the finisher refuse a world for ever
  while the serving path calls it `owed` -- so it reads "Improving" with
  nothing to end it. Both are corrupt-file states that no writer heals and
  that already stopped the finisher before this change. The ledger is
  per-WORLD, so one bad file traps every session of that world. Not fixed:
  modelling them would pull the attempt ledger and the lock protocol into
  the serving path. The better structural answer, recorded for whoever picks
  this up, is to have `assess()` publish its verdict and have
  `photographic_state` consume it, rather than keeping two hand-synchronised
  copies of a five-clause predicate.

  Reproduced directly rather than taken on the reviewer's word: with a
  `surface: stopped` record on an otherwise healthy session, a truncated
  `finish_attempts.json` gives `photographic: owed` / `assess: ledger-
  unreadable`, and a `LOCK` carrying `{"pid": "x"}` gives `photographic:
  owed` / `assess: locked`. Both are therefore **Improving with nothing to
  end it** — but neither is silent: the ledger case logs
  `the finish ledger for <world> is unreadable`, and both carry `state:
  owed` with a reason in the `photographic` block, so the condition is
  diagnosable rather than invisible.
- **`--max-worlds 1` per Tower start**: N owed worlds need N quiet restarts.

---

## 11. What iOS must do next, and why it is not optional

The Tower now tells the truth about a failed photographic build. **Nothing
on the phone reads it yet.**

`lifecycle.photographic` and the listing row's `photographic` block are
additive, and `WORLD-BUILDER-IOS.md` §3a is explicit that an app which
ignores them is still correct -- it simply hears `finalizing` where it used
to hear a false `finalized`, which is the T2/T4 fix and lands with no iOS
change at all. That part is done.

The exception is `failed`. A world whose photographic build broke is settled
-- it is saved, it opens at whatever rung it reached, and nothing more is
coming -- so the Tower deliberately keeps `ready`/`complete` and puts the
truth in the block. Reporting `finalizing` instead would strand the wearer
waiting for a build that will never arrive. But until the phone reads the
block, that world renders as plain **"Saved"**, which is the T3 defect the
Tower half of this campaign was meant to close.

A reviewer named this precisely, and they are right: putting a truth in a
field nobody reads is the same mistake `build_in_progress_unavailable_reason`
made, and this document should not pretend otherwise. **The Tower is honest;
the product is not yet.**

The iOS work is small and the contract already expresses it
(`WORLD-BUILDER-IOS.md` §3a): read `photographic.state`, and where it is
`failed`, say something like *"Saved — the photographic version could not be
built"* with `detail` behind it. `never_recorded` and `unattempted` must
keep rendering exactly as today; `owed` may say "still to be finished"
rather than implying work is happening this instant.

---

## 12. Verdict

**READY FOR MAC REVALIDATION.**

Not ready for the physical test, and the reason is specific rather than
cautious. Two things changed that the phone can see:

1. **Lifecycle words moved.** A world with an interrupted photographic stage
   now reports `finalizing` where it reported `ready`, on both the status
   channel and the Saved Worlds row. That is the intended fix, it is the
   state the owner's own world was in for this entire campaign, and it
   changes what the installed build draws. It needs to be seen on a phone.
2. **The photographic page changed how it boots.** It no longer dies on a
   manifest 404 or a transport fault, and it can now rescue itself. Every
   assertion behind that is a source assertion or a node-level follower
   test, because the page's CSP (`connect-src glasses-world:`) means it can
   only be fetched through the iOS scheme handler — **no browser on this
   machine can render it, by design.** Only a Mac can prove the recovery
   draws.

Add to those the iOS work in §11, without which a failed photographic build
still says "Saved", and Mac revalidation is clearly the right next gate.

What this campaign can state without a Mac:

- the stall is understood, from a live native stack, and fixed at its cause
- the world it happened to is **finished and serving photographically** --
  the first such world on this Tower
- ~6 min 40 s, against the 95 minutes it was held
- all 166 worlds / 70 sessions read exactly as they did before, on both
  surfaces, with zero disagreements
- no permanent false Improving on any shape the agreement test can build,
  and the two corrupt-file exceptions are named in §10
- the recovery finisher, run against the real root on the fixed code, found
  nothing owed and started nothing

The physical acceptance test — glasses, walk, Stop, Improving, photographic
reconstruction, Saved, open, recognisable bedroom, reopen — remains the pass,
and it remains unrun.
