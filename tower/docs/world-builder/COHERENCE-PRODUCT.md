# World Builder coherence — the switches, the re-finish, and a validation Tower

This is the operator's guide to the coherence product: the masked, seeded, gated final
solve with its consensus, components and areas, and the look-back relocalizer. The
contract is
[`docs/contracts/WORLD-BUILDER-COMPONENTS.md`](../../../docs/contracts/WORLD-BUILDER-COMPONENTS.md)
(v7). This file links to it and does not restate it.

- **Code described:** the product lane `world-builder/coherence-product-v1` at `e5d7ccb`
  (2026-09-24).
- **Status:** every behaviour here is **off by default**, and off is the Tower as it
  was before (`tower/tower/config.py`).
  - Review V8 found it safe to keep merged and not ready to switch on
    (`RUN\baseline\review\V8\V8-REVIEW.md`). Its fixes are in `e5d7ccb`. The fresh review
    V9 and the acceptance run have not reported yet (`RUN\status.md`).
- **Evidence:** `RUN` is `C:\Users\<you>\Projects\Glasses-scratch\wb-coherence-run-2026-09-23`.
  A path that starts with `RUN\` is there, not in this repository.
  - Some figures were measured on earlier code. Where that matters, the text names the
    code.
- **Machine:** every figure was measured on the development Tower: 20 cores and 32 GB
  (`RUN\lead\PHASE2-RULES.md`), and one 12 GB RTX 5070
  (`docs/world-builder-dense/04-OPERATIONS.md`).
- **Where this file lives:** `tower/docs` had no directory for World Builder operator
  docs, so this file starts `tower/docs/world-builder/`. The nearest precedent is
  `docs/world-builder-dense/04-OPERATIONS.md`.

---

## 1. The six switches

Set them in `tower\.env` (see `tower/.env.example`) or in the Tower's environment.

| setting | default | what it does | what it needs | cost measured in the run |
| --- | --- | --- | --- | --- |
| `TOWER_WORLD_SOLVE_MASKS` | `false` | Masks the wearer's hands, arms and held phone in every image of the session's **final** solve. With a walk database, the solve maps from a filtered copy of it (`walk-database-filtered`). Without one, it re-extracts features under the masks (`re-extracted`). The masks are cached per image (§6). The background solves during a walk are unchanged | `TOWER_WORLD_SOLVE` (on by default) and pycolmap. The GPU detectors (Grounding DINO + SAM 2.1, and OneFormer). If they cannot run, the solve runs **unmasked**, `transients.state` in `solution.json` says so, and a gated solve writes a notice (contract §3.1) | 132–451 s a world for 218–795 keyframes, or 0.54–0.60 s a keyframe, with an empty cache (`RUN\experiments\P3-VAL\TABLE.md`, code `e2e7582`). GPU peak 2,678 MB on 383 images (`RUN\experiments\P3-PM\out-masks2.log`). A later solve of the same images computes none: 690 of 690 from the cache (`RUN\experiments\P3-H2\real\B.solution.json`) |
| `TOWER_WORLD_SOLVE_SEED` | unset | An integer ≥ 0 seeds every random generator of the final solve's mapper, and the mapper then runs on **one** thread. A seeded final solve also **freezes its matching** (§6). Unset, blank, `off`, or anything that is not a non-negative integer, means today's multi-threaded, unseeded solve. Recorded as `solve.seed` and `solve.threads` | `TOWER_WORLD_SOLVE`. The consensus needs it | Mapping is 3.3× slower: 111 s against 33 s at 383 keyframes (`RUN\research\D1-sfm-slam-posegraph.md` §0, §2.4). The seeded mapping took 23–181 s a world (`P3-VAL\TABLE.md`) |
| `TOWER_WORLD_SOLVE_GATE` | `false` | Runs on the final solve only. It computes depth before publishing (MoGe-2 ViT-L) and a metric scale per camera, then applies the evidence gate. It relabels the room and the unplaced pieces, and writes `solve\<session>\components.json`. The depth predictions are cached (§6). Off publishes the solve as the solver returned it | **Masks:** if `transients.state` is not `applied`, the gate attaches nothing outside the room's anchor block (reason `masks-unavailable`). **Metric scale:** if fewer than half the supported cameras have a ratio, it attaches nothing (reason `scale-unavailable`). If depth failed, the finisher owes a re-gate in place. An exception in the gate publishes the solve **ungated**, with `components: null` (`coherence_publish.py`). Every fail-safe writes `finalization.notice` (contract §3.1) | 40–164 s a world, of which depth is 39–160 s over 203–712 frames (`P3-VAL\TABLE.md`). With the predictions cached, the gate took 60 s against 190 s on 678 frames (`P3-H2\real\B.solution.json`, `C.solution.json`). The room's final surface reuses the depth (`coherence_publish.py`, step 6). Disk: the prediction cache holds 300 MB for 678 frames (`RUN\experiments\P3-H2\PROGRESS.md`) |
| `TOWER_WORLD_SOLVE_CONSENSUS` | `1` | With N ≥ 2, a gated, seeded final solve maps N draws, with mapper seeds s … s+N−1, on its one frozen database, with the same masks and depth predictions. It gates each draw, and each keyframe votes: attached to the room or not. It publishes the draw that agrees most with the strict majority; ties go to the lowest seed. If fewer than a strict majority of draws attached a group of that draw's room, the group is withheld as its own piece, reason `seed-unstable`. The room's anchor group is never withheld. Unset, blank, garbage, 0 or negative means 1: a single draw, exactly as before (`config.world_solve_consensus_setting`) | `TOWER_WORLD_SOLVE_GATE` and `TOWER_WORLD_SOLVE_SEED`. Without the seed, the consensus records `not-run`. If the first draw took a fail-safe, there is nothing to vote on: `not-needed`, or `deferred` to the re-gate that the fail-safe owes (`coherence_publish.py`) | About 250–260 s per extra draw on 678 posed keyframes. That is ~190 s of single-thread mapping plus ~60 s of gate, of which ~55 s is the depth stage re-fitting cached predictions; no GPU. N = 3 adds about 8.5 min (`RUN\experiments\P3-H2\PROGRESS.md`) |
| `TOWER_WORLD_AREA_BUILDS` | `false` | The idle finisher builds a surface and an appearance for each `shown_as: "area"` component, in `<world>\areas\<area_id>\`. Off: those areas are recorded as declined, which the phone shows as *could not be built*. `world_refinish.py` builds them either way | A components record, so the gate must be on. The finisher: `TOWER_WORLD_FINISH_PENDING`, `TOWER_WORLD_SURFACE` and `TOWER_WORLD_SOLVE` all on (`components.py`, `areas_nobody_will_build_reason`) | Per area: 8–17 s to prepare, plus 43–201 s of stages. That is 9 areas on 4 worlds; a walk had 0–5 areas (`P3-VAL\TABLE.md`) |
| `TOWER_WORLD_RELOCALIZER` | `off` | `prompt`: after a tracking loss, the builder matches incoming frames at 2 Hz against the 10 keyframes before the loss. When it relocalizes, it journals a verified revisit link. When it does not, it asks the wearer to look back: at most 2 prompts in any 60 s, with a 30 s cooldown (`relocalizer.py`). `silent`: it runs and records, but never prompts (the test's prompt-off arm). `on`, `true`, `yes` and `1` mean `prompt`. Any other word means `off`, and is logged | The **builder** reads it at each session start, from the environment it inherits from the Tower. It needs the session's calibration (`engine.py`). A final solve imports its revisit links only when the masks are `applied` and the solve is gated, and only at ≥ 50 inliers on every leg (contract §2.5, `solve.revisit_pairs`; `relocalizer.REVISIT_MIN_INLIERS`). The phone speaks prompts only in a DEBUG build (contract §6.5) | 0.09–0.28 of one core at 2 Hz, only while an episode is open. It asked 0.81–1.29 times a minute when replayed on 7 walks (`RUN\experiments\P3-PR\out\PRODUCT.md`, measured before `2dcf680`) |

**Spelling.** The masks, gate and area-build settings go through `config._flag`:

- `1`, `true`, `yes` and `on` mean true.
- A blank value means the default.
- Anything else, including a typo, means **false**.

The seed, the consensus and the relocalizer have their own readers, described in the
table. In every case a typo turns the behaviour off, never on.

**Who reads them.**

- **Masks, seed, gate and consensus:** the final solve's own process (`world_solve.py`
  or `world_finalize.py`).
- **Area builds:** the finisher.
- **The relocalizer:** the builder.

Each process inherits the environment the Tower started with. To change a setting,
restart the Tower.

**What the re-finish decides for itself.** `scripts/world_refinish.py` always runs
masks, the gate, its `--seed` and the area builds, whatever the Tower's settings say.
The consensus it takes from **its own** environment: the child solve inherits it
(`run_final_solve`). The run's acceptance chain exports
`TOWER_WORLD_SOLVE_CONSENSUS=3` before every re-finish
(`RUN\experiments\P3-ACC\chain.sh`).

**Old worlds do not change.** The finisher never computes components for a world that
has none (contract §7 rule 5). V8 found the pages of old worlds byte-identical.

## 2. Re-finish a saved world

`scripts/world_refinish.py` rebuilds one saved session with the product pipeline. It is
the only way an old world gets `components` (contract §7 rule 4). An owner runs it by
hand; nothing else does.

### 2.1 Stop the Tower first

**Stop every Tower that serves this world root**, or make sure the one that serves it
runs with `TOWER_WORLD_FINISH_PENDING=false`, as the validation Tower in §3 does
(contract §7 rule 4).

**Why (V8 M3; `RUN\baseline\review\V8\rvx-report.txt`):**

- The re-finish runs in steps, and holds the world's lock only inside each step.
- After step 1, it records the room's stages as `stopped` ("re-finish in progress"), and
  a `stopped` stage is owed work.
- An idle Tower's finisher could therefore take the room in the gaps between steps 1→2
  and 2→3. It would build the room from a solve that was set aside or not yet rebuilt,
  and hold the lock that the next step needs.

**What `e5d7ccb` does about it** (`world_finish_pending.refinish_in_progress`):

- The ledger records the re-finish's process: its pid and start time.
- While that process is alive, the finisher stays out of the world.
- A re-finish whose process is gone no longer holds the world. Its `stopped` room is
  owed like any other, and the finisher logs that the re-finish died.
- A ledger in a final state (`done`, `stopped`, rolled back or restored) holds nothing.

**Why you still stop the Tower:**

- **Older code does not do this.** The integration branch at `1111bb9`, which the
  validation Tower on :8020 runs, merged this lane at `d87aa5c`, before any of it.
- **A ledger written before `3abe763` has no process.** If that re-finish died between
  its steps, it still looks live, and the finisher waits until you run the re-finish
  again.
- **Windows cannot rename a directory while a file under it is open,** and a serving
  Tower may hold an area chunk or a surface level open. The set-aside retries each move
  5 times with backoff. If a move still fails, it undoes what it moved and refuses
  (`world_refinish.py`, `MOVE_ATTEMPTS`).

### 2.2 The command

```
cd tower
.venv\Scripts\python.exe scripts\world_refinish.py --root <world root> --world <world_id> [--session <session_id>] [--seed 0]
```

- **`--root`** is the directory that holds `worlds\`: a Tower's `TOWER_WORLD_ROOT`.
  - Give an absolute path.
  - It refuses a drive root, your home directory, and a direct child of either
    (`tower/artifact_paths.py`).
- **The interpreter** must have pycolmap, or the solve does not run
  (`global_solve.solver_available`).
  - The run used the canonical checkout's `tower\.venv\Scripts\python.exe`, with
    `PYTHONPATH=<code>\tower` (`RUN\experiments\P3-ACC\chain.sh`).
- **The environment** it needs:
  - `TOWER_CAPTURE_ROOT` to find the raw captures (§2.5);
  - `TOWER_SOURCES_ROOT` when the code runs from anywhere but the Tower that recorded
    the walk (§2.5);
  - `TOWER_WORLD_SOLVE_CONSENSUS` if you want a consensus (§1).

| flag | default | what it does |
| --- | --- | --- |
| `--root` | required | the world root |
| `--world` | required | the world to rebuild |
| `--session` | the world's latest session | the session to rebuild |
| `--seed` | `0` | the final solve's `TOWER_WORLD_SOLVE_SEED`; the first consensus draw's seed |
| `--threads` | `-1` | passed to `world_finalize.py` |
| `--appearance` / `--no-appearance` | on | build the room's and the areas' appearance |
| `--keep-depth-work` | off | keep the per-frame depth work afterwards |
| `--capture-dir DIR` | the session's capture chain | a capture directory to search for each keyframe's own raw frame (repeatable). It must hold its journal, `frames.jsonl` (§2.5) |
| `--no-capture` | off | search no capture: the walk's own `sources.json`, else the redacted keyframes. Not with `--capture-dir` |
| `--dry-run` | off | print what would be set aside, the solve's settings and the planned `solver_frames`; write nothing |
| `--format` | `json` | `json` or `text` |

**Exit status:**

- `0`: done.
- `1`: a step failed. The report says which one.
- `2`: refused, and the world is as it was. The one exception is a set-aside that
  failed part-way: it leaves its ledger and copies under `refinish\<stamp>\`, and a
  `rollback-incomplete` ledger names what could not be moved back.

**It refuses when:**

- the world or the session does not exist;
- the world's imagery was purged;
- the session never stopped;
- a live writer holds the world's lock;
- a stage of the session (the room's or an area's) is running under a live process
  (`world_refinish.build_in_progress`);
- the set-aside could not complete, and was rolled back.

### 2.3 What it does

1. **Sets the previous result aside (§2.4)** under the world's lock, and marks the
   room's surface and appearance as `stopped`. Under the same lock, it finds each
   keyframe's raw frame and writes the fresh solve's `sources.json` (§2.5).
2. **Runs `world_finalize.py` in a child process,** with `TOWER_WORLD_SOLVE_MASKS=1`,
   `TOWER_WORLD_SOLVE_GATE=1` and `TOWER_WORLD_SOLVE_SEED=<--seed>`. The child inherits
   the rest of the environment, `TOWER_WORLD_SOLVE_CONSENSUS` included.
3. **Builds the room:** its final surface and appearance.
4. **Builds the areas:** every `shown_as: "area"` component.

**When something goes wrong** (`3abe763`):

- If the final solve publishes nothing, it puts the previous result back, and keeps
  what the failed solve left under `refinish\<stamp>\failed-*`.
- If **any** exception happens between the set-aside and a published solve, it does the
  same. For example, a read-only `session.json` (§2.6). The ledger then says
  `restored-after-an-error`, and `error_after_set_aside` names the step and the error.
- If the previous result cannot all go back, the ledger says `restore-incomplete`, and
  lists what could not be done.

If the gate wrote no record, the report's `components` is `null`.

### 2.4 What it sets aside, and how to roll back by hand

Everything goes under `<world>\refinish\<stamp>\`, where `<stamp>` is local time as
`YYYYMMDD-HHMMSS`. **Nothing is deleted.**

| what | how | to |
| --- | --- | --- |
| `solve\<session>` | moved | `refinish\<stamp>\solve\<session>` |
| every `areas\<area_id>` of the session | moved | `refinish\<stamp>\areas\<area_id>` |
| `surface\<session>`, `appearance\<session>`, `dense\<session>` | copied | `refinish\<stamp>\<kind>\<session>` |
| `derived\` | copied | `refinish\<stamp>\derived` |
| the session record | copied | `refinish\<stamp>\session.json` |
| — | written first | `refinish\<stamp>\refinish.json`: the ledger |

**The ledger** (`refinish.json`) records:

- what moved or was copied, and from where;
- the session's previous finalization and stage records;
- the finisher's previous attempt counters, which the re-finish resets;
- `process`: the pid and start time of the re-finish;
- `solver_frames`: where every solver frame came from (§2.5).

**The fresh `solve\<session>`** gets **copies** of the walk's inputs
(`SOLVE_COPY_BACK`):

- `database.db`, with its `-wal` and `-shm` files;
- `database.matching.json`, the record of its frozen matching (§6);
- the mask cache, `transients\`;
- `images\`, `sources.json` and `camera.json`.

`dense\<session>` is copied, not moved, so the cached depth predictions stay in place
(`dense_pipeline.py`).

The copy keeps the masked solve on the walk's own database (`walk-database-filtered`).
Without the copy, the solve re-extracts features, and on the target walk that splits
the room (`world_refinish.py`, RV1 M1-1).

**The ledger's `state`:**

| state | when |
| --- | --- |
| `setting-aside` | while step 1 runs |
| `set-aside` | after step 1, until the solve publishes |
| `published` | the solve published; the room and areas are being rebuilt |
| `done` | finished |
| `stopped` | a stop was asked for, or the room could not be rebuilt (`detail` says which) |
| `rolled-back`, `rollback-incomplete` | step 1 failed |
| `restored-after-a-failed-solve` | the solve published nothing, and the previous result was put back |
| `restored-after-an-error` | an exception after the set-aside, and the previous result was put back |
| `restore-incomplete` | the previous result could not all be put back; see `restore_report` and `restore_errors` |

**Rolling back a re-finish that succeeded.** There is no rollback command. The ledger's
own `restore` field gives the procedure:

1. Stop the Tower (§2.1).
2. Move the rebuild's `solve\<session>`, and its new `areas\<area_id>` directories, aside
   into that stamp's `refinish\<stamp>\`. Delete nothing.
3. Move each `moved` entry in the ledger back, from its `to` path to its `from` path.
4. The `copied` entries are snapshots of what the rebuild replaced in place: the room's
   surface, appearance and dense trees, `derived\`, and `session.json`. Move the rebuilt
   ones aside, and put the snapshots back.

Deleting anything under `refinish\` needs a human's approval (filesystem policy,
rule 14). Nothing prunes these directories.

### 2.5 Where the solver's frames come from

Each keyframe's solver image is the first of these that exists (`plan_solver_frames`,
`global_solve.prepare_images`):

1. the walk's own undistorted solver image, already in `solve\<session>\images\` (copied
   back, §2.4);
2. the raw frame that the walk's builder recorded in `sources.json`, if that file is on
   disk;
3. the keyframe's **own** raw frame, found by its capture identity (below);
4. the session's stored keyframe copy, which is **face-redacted**. This is the fallback
   for each keyframe on its own: when its frame is not found, when more than one frame
   matches it, or when its keyframe id or image name occurs twice in the session.

**Found by capture identity, never by name** (`fe15656`).

- A keyframe's frame is the one record in a capture's journal, `frames.jsonl`, with the
  keyframe's `source_seq` **and** its `received_at`, to the microsecond. Where both
  records carry `wire_seq` and `time_basis`, those must agree too (`map_raw_frames`).
- The re-finish writes the frames it finds into the fresh solve's `sources.json`, as the
  builder records them. It never hands the solve a `--capture-dir`, because that lookup
  goes by frame name, and a walk's captures reuse names (`run_final_solve`).
- **Why:** the earlier by-name lookup gave 67 of the 353 keyframes of the live walk
  adc75972 another capture's frame (`V8-REVIEW.md`, fix log). By identity, PF checked
  the chosen frames against the stored keyframes' unfilled pixels and found no wrong
  assignment on adc75972, 3dd986b1, 991e5a15 or af47007c
  (`RUN\experiments\P3-PF\PROGRESS.md`, 02:40).

**Which captures are searched:**

- every `--capture-dir DIR` you name (repeatable). A directory with no `frames.jsonl` is
  not searched: a bare directory of frames has names only.
- Otherwise, the capture the session records, `session.capture_id`, under
  `TOWER_CAPTURE_ROOT`, as `<capture root>\captures\<capture_id>`. The search also
  covers every capture that continues it after a reconnect (`continues_capture`,
  followed through every descendant). A walk that reconnected lives in 2–3 captures:
  991e5a15 and af47007c each span 3 (`_capture_chain`).
- A relative `TOWER_CAPTURE_ROOT` is anchored like a `sources.json` path (below).
- `--no-capture` searches none.

**`solver_frames`**, in the report and the ledger, says what happened. The dry run
prints it too.

- It counts `already_undistorted`, `raw_from_sources_json`, `raw_from_capture_dir` and
  `redacted_session_copies`.
- It splits the redacted ones by why: `redacted_ambiguous_in_session`,
  `redacted_no_capture_frame` and `redacted_several_capture_frames`.
- `source` sums it up: `walk-solver-images`, `raw-capture`, `redacted-session-keyframes`
  or `mixed`.
- It also records `from` and `why` (which captures), `ambiguous_keyframe_ids`,
  `capture_notes`, and whether `sources.json` was rewritten (`sources_json_written`).

**Without raw captures,** a keyframe's solver image and its masks come from its
redacted copy, fill boxes and all. That costs keyframes: af47007c, re-finished from its
redacted copies, posed 168 of its 218 (`RUN\experiments\P3-VAL\TABLE.md`, finding 6, code
`e2e7582`). On that walk the redactor filled 66 of 218 frames by 2 % or more, with a
median fill of 22 % (`RUN\mailbox\to-lead\021-h1-decision-p2.md`). `solver_frames` now
says when this happens.

**The raw frames are solver input only.**

- They never leave the machine, and the only things derived from them that are
  published are points and poses (`global_solve._source_frame`).
- Point colours are withheld (`global_solve.WITHHELD_POINT_RGB`).
- The room's surface and appearance read the redacted keyframes (`world_refinish.py`).
  A redacted build never draws on a solver image, not even for a mask
  (`solve_masks.py`).

**`TOWER_SOURCES_ROOT`.**

- `sources.json` records paths as the builder was given them, usually relative:
  `data\captures\<capture>\frames\<n>.jpg`.
- A relative path is resolved against the `tower\` directory of the code that runs, or
  against `TOWER_SOURCES_ROOT` when that is set (`global_solve.sources_root`). It is
  never resolved against the working directory.
- So when you run code from anywhere but the Tower that recorded the walk, point
  `TOWER_SOURCES_ROOT` at a directory that holds `data\captures\...`:
  - the recording Tower's `tower\`; or
  - a copy laid out the same way. The run used copies (`RUN\v8020\refinish_all.sh`,
    `RUN\experiments\P3-ACC\chain.sh`).
- The frames are only read.

**The builder's own lookup.** `world_finalize.py --capture-dir` still looks up frames
by name. Since `6a442bb`, a name that is found in more than one capture directory takes
the stored keyframe instead of a guess, and `solve.frames_ambiguous_by_name` counts them
(contract §2.5). The re-finish never reaches this lookup.

### 2.6 Copied worlds: clear Windows' read-only attribute

A world copied from a read-only source, such as the run's frozen evidence, keeps the
read-only attribute on every file. The re-finish then cannot write `session.json`, and
fails with WinError 5 just after the set-aside (`P3-VAL\TABLE.md`, finding 8). Since
`3abe763`, it puts the previous result back and says `restored-after-an-error` (§2.3),
but the re-finish still does not happen.

Clear the attribute **on the copy only**:

```
attrib -R "<copy root>\worlds\<world_id>\*" /S /D
```

The run did the same in Python, adding `stat.S_IWRITE` to every entry
(`RUN\experiments\P2-G0\tools\setup_copies.py`). Never clear it on frozen evidence or on
the live store.

### 2.7 What it costs

**A whole re-finish** (masks, gate, seed 0, no consensus; the room and its areas):

| code | worlds | locked GPU time per world | source |
| --- | --- | --- | --- |
| `e2e7582` | 7, with 218–795 keyframes | 6.5–27.1 min | `RUN\experiments\P3-VAL\TABLE.md` |
| `1111bb9` | 3, with 383–398 keyframes | 13.7–16.0 min | `RUN\experiments\GPU-JOBS.log` |

Where the time went on 6839fb8f (690 keyframes, 25.0 min in all; `P3-VAL\TABLE.md`):

| stage | time |
| --- | --- |
| masks | 384 s |
| matching | 47 s |
| mapping | 181 s |
| gate (mostly depth) | 164 s |
| the room's surface | 495 s |
| the room's appearance | 127 s |
| one area | 83 s |

**The final solve alone, with consensus 3,** on a copy of 6839fb8f. The code was
`2dcf680` plus H2's files (`RUN\experiments\P3-H2\real\run.sh`):

| run | caches | locked time | source |
| --- | --- | --- | --- |
| B | masks cached, matching frozen, depth predicted | 15.4 min | `GPU-JOBS.log` |
| C | everything warm | 13.5 min | `GPU-JOBS.log` |

In run C, the consensus took 597 s: the three draws' gates and the two extra draws'
mapping (`P3-H2\real\C.solution.json`, `P3-H2\PROGRESS.md`).

**Disk and GPU:**

- **Disk.** The set-aside keeps full copies of the room's surface, appearance, dense and
  derived trees. The run's seven re-finished copies, with their set-asides, took 3.1 GB
  (`RUN\experiments\P3-VAL\PROGRESS.md`).
- **GPU.** Run one re-finish at a time. The run serialised GPU jobs with its own lock
  (`RUN\lead\gpulock.py`), which is not part of the product.

## 3. Stand up a validation Tower (serve only)

A validation Tower serves re-finished **copies** to a phone. It does not touch the live
store or port 8000.

**The worked example** is the Tower that the Mac gate used, on 2026-09-23/24:
`RUN\v8020\start_tower.ps1` and `RUN\v8020\refinish_all.sh`, on code `1111bb9`. Its
record is `RUN\mailbox\to-manager\20260924-0025-015-delivered.md`.

- Its re-finish predates capture identity, so the environment in step 4 follows the
  run's acceptance chain instead (`RUN\experiments\P3-ACC\chain.sh`).

**Its rules:**

- its own world root and its own port;
- never the live store, never port 8000, and never a checkout's `.env`;
- serve only: `TOWER_WORLD_FINISH_PENDING=false` and `TOWER_WORLD_AUTOBUILD=false`.

**1. Copy the worlds** into `<V>\worlds\<world_id>`. `<V>` is a directory under
`Glasses-scratch\`, never under the live store.

**2. Clear the read-only attribute** on the copies (§2.6).

**3. Make the raw captures reachable** (§2.5). Copy whole capture directories, with
their `capture.json` and `frames.jsonl`, not only `frames\`:

- as `<C>\captures\<capture_id>\` for `TOWER_CAPTURE_ROOT=<C>`;
- as `<S>\data\captures\<capture_id>\` for the relative paths in a walk's
  `sources.json`, with `TOWER_SOURCES_ROOT=<S>`.

**4. Re-finish each world, one at a time, before the Tower starts.** In Git Bash:

```bash
V='C:\Users\<you>\Projects\Glasses-scratch\<run>\v<port>'       # the validation root
CODE='C:\Users\<you>\Projects\Glasses-worktrees\<worktree>'      # the code under test
PY='C:\Users\<you>\Projects\Glasses\tower\.venv\Scripts\python.exe'
export PYTHONPATH="$CODE\\tower"
export PYTHONDONTWRITEBYTECODE=1
export TOWER_CAPTURE_ROOT='<C>'                                  # holds captures\<capture_id>\
export TOWER_SOURCES_ROOT='<S>'                                  # holds data\captures\...
export TOWER_WORLD_SOLVE_CONSENSUS=3                             # 1 for a single draw
mkdir -p "$(cygpath -u "$V")/refinish" "$(cygpath -u "$V")/logs"
for W in <world_id> <world_id>; do
  "$PY" "$CODE\\tower\\scripts\\world_refinish.py" --root "$V" --world "$W" --seed 0 \
    > "$(cygpath -u "$V")/refinish/${W:0:8}.json" 2> "$(cygpath -u "$V")/logs/${W:0:8}.stderr.log"
done
```

Read `solver_frames` in each report before serving the world. The run wrapped each call
in `RUN\lead\gpulock.py run ...`.

**5. Start the Tower.** In PowerShell:

```powershell
$V    = 'C:\Users\<you>\Projects\Glasses-scratch\<run>\v<port>'
$CODE = 'C:\Users\<you>\Projects\Glasses-worktrees\<worktree>\tower'
$PY   = 'C:\Users\<you>\Projects\Glasses\tower\.venv\Scripts\python.exe'
$PORT = 8020                                    # anything but 8000
New-Item -ItemType Directory -Force "$V\captures", "$V\logs", "$V\cwd" | Out-Null

$env:PYTHONPATH                 = $CODE
$env:PYTHONDONTWRITEBYTECODE    = '1'
$env:TOWER_HOST                 = '0.0.0.0'
$env:TOWER_PORT                 = "$PORT"
$env:TOWER_WORLD_ROOT           = $V
$env:TOWER_CAPTURE_ROOT         = "$V\captures"
$env:TOWER_WORLD_FINISH_PENDING = 'false'   # serve only: never finish owed work on this root
$env:TOWER_WORLD_AUTOBUILD      = 'false'   # no live world builds
$env:TOWER_OBSERVATION_ENABLED  = 'false'
$env:TOWER_DOCUMENT_ENABLED     = 'false'
$env:TOWER_SCENE_UNDERSTANDING  = 'off'
$env:TOWER_SCENE_AUTOSTART      = 'false'
$env:TOWER_CV_DEVICE            = 'cpu'     # leave the GPU to everything else

$p = Start-Process -FilePath $PY -ArgumentList @('-m', 'uvicorn', 'tower.main:app', '--host', '0.0.0.0', '--port', "$PORT") `
      -WorkingDirectory "$V\cwd" -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput "$V\logs\tower-$PORT.out.log" -RedirectStandardError "$V\logs\tower-$PORT.err.log"
"started pid $($p.Id) at $(Get-Date -Format s)" | Tee-Object -FilePath "$V\logs\tower-$PORT.pid.log" -Append
```

**How this differs from `scripts\start_tower.ps1`:**

- **No `.env`.** It starts uvicorn directly, with no `--env-file`, so no checkout's
  `.env` can leak in.
- **Two uvicorn options are left out:** `--loop tower.serve_loop:resilient_loop_factory`
  and `--timeout-graceful-shutdown 10`.
- **An empty working directory,** `cwd\`. The face-redaction model path resolves against
  the working directory (`tower/.env.example`, Privacy). So do not reuse this recipe for
  a Tower that records.

**6. Check it** the way the example was checked (the 015 note, §2; saved in
`RUN\v8020\logs\http-checks\`):

| check | expect |
| --- | --- |
| `/health` | 200 |
| `/worlds` | every row carries `components` |
| `/worlds/{w}/render/revision` and `/render` | 200 for each world |
| `/worlds/{w}/areas/{s}/{a}/render/revision`, `/render`, `/appearance/manifest` | 200 for each area |
| a counted-only id, or the room's id, used as an area | 404 `no such area in this session` |

Checks made from the Tower's own machine do not test the firewall.

**7. Stop it.** The process ID that `Start-Process` returns is the venv launcher's; the
server is its child. For the example, the launcher was 42644 and the server 38620 (the
015 note). Find the server by its port:
`Get-NetTCPConnection -LocalPort <port> -State Listen`.

## 4. Read `components.json`

**Where it is.** `<world>\solve\<session>\components.json`, written once for each
published solve, by the gate.

- The same array reaches the phone on the `GET /worlds` session rows, and on
  `GET /worlds/{w}/render/revision` (contract §3).
- The record is `wb-components-record/1`. Its keys are `record`, `contract`,
  `session_id`, `input_digest`, `solved_at`, `solve_identity`, `gate` and `components`.
- Each entry holds the gate's fields (`id`, `state`, `reason`, `reasons`, `shown_as`,
  `keyframes`, `capture_spans_s`) and `keyframe_ids`. When the Tower serves a row, it
  adds `has_geometry`, `keyframes_phone` and `photographic`, and drops `keyframe_ids`,
  which never leaves the Tower (`components.py`).

**No record, or `components: null`, means "not computed".** That covers:

- every world saved before the gate;
- a gate that failed;
- a record whose `solve_identity` is not the published solve's (contract §2.5).

It never means "no areas".

**`shown_as`** (contract §2.3):

- `room`: the one `placed` entry, always first.
- `area`: unplaced, with at least 30 keyframes or at least 5 s of capture span. It is
  built on its own, in `<world>\areas\<area_id>\` (`solve\`, `surface\`, `appearance\`,
  `dense\`, and `record.json`), and served on `/worlds/{w}/areas/{s}/{a}/…`
  (contract §5).
- `none`, which is **counted only**: unplaced and below both floors. It is never built
  and never drawable. The phone only counts it, in the footer below the room
  (contract §8). The floors are `AREA_MIN_KEYFRAMES` and `AREA_MIN_SPAN_S` in
  `coherence_publish.py`. They come from one case (OPEN T2).

**`reason` and `reasons`** are the gate's decisions, in the precedence of contract §2.2:

1. `masks-unavailable`
2. `scale-unavailable`
3. `solved-separately`
4. `no-verified-link`
5. `single-unconfirmed-link`
6. `link-contradicted`
7. `scale-mismatch`
8. `seed-unstable` (contract v7). The consensus withheld the piece: the published draw
   attached it, but fewer than a strict majority of draws did. For such a piece it is
   the **only** reason. The phone never shows reason text (Mac tip `8009854`,
   `RUN\mailbox\to-lead\022-mac-tip-8009854.md`).

Read §2.2 for what each one means.

**What a fail-safe looks like.** Every piece outside the room carries only
`masks-unavailable`, or only `scale-unavailable`. Then:

- read `transients.state` and the `gate` block in `solve\<session>\solution.json`:
  `gate.masks_applied`, `gate.metric_available`, `gate.retryable` and `gate.cause`
  (contract §2.5);
- read the row's `finalization.notice`, which says what the Tower could not do and who
  can fix it (contract §3.1).

**Where the consensus votes are** (contract §2.5):

- **`gate.consensus`, in `solution.json`:**
  - `requested` (N), `seeds`, and `state`: `applied`, `not-needed`, `deferred`,
    `not-run`, or `not-applied` (withholding would have changed which piece is the room,
    so the chosen draw was published as gated), with `why`;
  - a summary per draw: its seed, time, `solve_identity`, room keyframes and agreement;
  - the `chosen` draw;
  - per group, its `votes`, whether it is `ambiguous` (not unanimous), and its
    `decision`: `anchor`, `attached`, `seed-unstable` or `unplaced`;
  - `pieces`: a minority piece that sits inside the published anchor block. It cannot be
    withheld, so it is only reported.
  - It is absent when N = 1.
- **`solve\<session>\consensus.json`:** each draw's per-round gate decisions. Written
  only when N ≥ 2.

**`components.superseded.json`** is a record set aside when a solution was published
without the gate (`coherence_publish.py`).

## 5. The physical A/B test — the Tower side

The protocol is a run draft, `RUN\lead\PHYSICAL-TEST-PROTOCOL.md`, and is not in this
repository. This section covers only the Tower side. **The phone side is OPEN for the
Mac:** the DEBUG build, speech, and the device measurements M-a, M-b and M-c.

**Which Tower.**

- The candidate SHA, on a Tower that **records**: `TOWER_CAPTURE_ROOT` and
  `TOWER_WORLD_ROOT` set, and `TOWER_WORLD_AUTOBUILD` on (the default).
- Not the serve-only validation Tower in §3, which records nothing.
- Which port and which world root: **OPEN**. The protocol does not name them.

**Flags to turn on:**

| setting | value | why |
| --- | --- | --- |
| `TOWER_WORLD_SOLVE_MASKS` | `true` | the gate's hard dependency |
| `TOWER_WORLD_SOLVE_SEED` | `0` | an integer; the consensus and the frozen matching need it |
| `TOWER_WORLD_SOLVE_GATE` | `true` | components, areas and the notice |
| `TOWER_WORLD_SOLVE_CONSENSUS` | `3` | attachment decided by 3 mapper seeds, as in the acceptance run (`P3-ACC\chain.sh`) |
| `TOWER_WORLD_AREA_BUILDS` | `true` | so the Tower builds the walk's areas; otherwise they read *could not be built*. The acceptance run sets it; the protocol draft does not say |
| `TOWER_WORLD_RELOCALIZER` | `prompt` for walks W-A, W-C and W-G; `silent` for W-B | the A/B arm |

Leave `TOWER_WORLD_FINISH_PENDING` on (the default): the areas and any owed re-gate are
built by the finisher.

The builder reads the relocalizer setting when each session starts, from the Tower's
environment. So to switch between arms, restart the Tower with the new value.

**The phone build.** A DEBUG build (contract §6.5), from the Mac lane at `8009854` or
later. That build shows `finalization.notice` word for word under the room caption,
never on area screens (contract §8, v6; `RUN\mailbox\to-lead\022-mac-tip-8009854.md`).

**What to look at, for each walk:**

- **The journal,** `<world>\sessions\<session>\events.jsonl`:
  - `relocalizer_started`, with the `acceptance`, the `limiter` and `prompts_enabled`
    that the walk ran with;
  - `recovery_prompted`, `recovery_withheld`, `recovery_accepted`, `recovery_timed_out`,
    `recovery_anchored` and `relocalizer_stopped` (`relocalizer.py`).
  - Prompts per minute come from `recovery_prompted`. The pass bar is 2 or fewer.
- **The status channel's `tracking.recovery`,** including `prompt.issued_at` for
  measurement M-c (contract §6.2).
- **`finalization.notice`** on the row, and in the status channel's
  `lifecycle.finalization` (`d084627`). It is absent when nothing is owed.
- **`solve\<session>\solution.json`:**
  - `transients`, `solve.seed`, `solve.threads`, `solve.matching` and
    `gate.params_digest`. These show that the walk ran with the settings above;
  - `solve.revisit_pairs`: the relocalizer's links that the solve imported;
  - `gate.consensus`, read as in §4.
- **`solve\<session>\components.json`,** read as in §4.
- **The run harness,** `tower\scripts\world_coherence_eval.py`, with `--renders`. It lives
  on the lane branch `world-builder/coherence-v1`, not on this one.
- **Measurement M-b** needs the Tower's received-frame rate and resolution in the 5 s
  either side of each prompt, taken from the walk's capture.

**Prior evidence** (a replay, not a walk, before `2dcf680`): of 53 anchored revisit
links that could be checked, 2 were wrong by more than 8°, both on the control. The
protocol's bar is 0 (`RUN\experiments\P3-PR\out\PRODUCT.md`).

Keep the world directory, the journal and the capture. Raw imagery stays on this machine.

## 6. What is and is not deterministic

**The rule, in one line.** A seeded, gated finish of a walk that already has its caches
reproduces the same world. The **first** finish of a walk is one draw of its matching,
its masks and its depth, and that draw is then kept.

Each source of variation, and what `e5d7ccb` does with it:

**Masks: cached.**

- They are kept per solver image, keyed by the image's name and the SHA-1 of its bytes,
  in `solve\<session>\transients\` (`solve_masks.py`). A re-finish carries the cache
  back (§2.4).
- A later solve of the same images calls no detector. H2's real runs took 690 of 690
  from the cache (`P3-H2\real\B.solution.json`, `C.solution.json`), and a unit test pins
  it for a re-finish
  (`test_world_builder_frozen_matching.py::test_a_refinish_maps_the_frozen_matching_and_hits_every_cached_mask`).
- The first masked solve of a walk computes them on the GPU.

**Matching: frozen after the first seeded final solve.**

- **Why it has to be frozen.** Matching is not deterministic even on one thread. From one
  pristine walk database, two one-thread runs differed by 2 + 2 verified pairs, and two
  default-thread runs by 5 + 7 (`P3-PF\PROGRESS.md`, 00:57, step 2).
- **What freezes it.** After a seeded final solve matches, it writes
  `database.matching.json` beside the walk database. That record holds the key (pycolmap
  version, camera, features, overlap, loop detection, verification seed, revisit list),
  the image names with their SHA-1, and the database's content digest.
- **What happens next time.** A later seeded final solve with the same key, images and
  content skips extraction and matching: `solve.matching: "frozen"`. Anything else
  matches again and re-freezes: `"matched"`, with `solve.matching_detail`
  (`global_solve.py`).
- **So the first finish of a new walk is still one matching draw.** So is the first
  seeded finish of any walk with no record yet: a world finished before `e4d35ee`, or
  one finished unseeded. It matches once more, then it is frozen.
- **Two cases are never frozen:**
  - an unseeded solve (the default);
  - a walk with no walk database, whose masked solve re-extracts features. That
    database is the solve's own, so every re-finish matches afresh.

**Depth predictions: cached per input pixels.**

- The gate's MoGe predictions are kept as float16 in
  `dense\<session>\predictions\<token>\`. Each is keyed by the SHA-1 of the exact pixels
  the network was shown; `token` names the network and its parameters
  (`dense_pipeline.py`).
- `dense\<session>` is copied, not moved, by a re-finish, so the cache stays.
- The fit to the solve is always recomputed. A fresh prediction is rounded through
  float16 before it is fitted, so fresh and cached fits are identical.
- `gate.depth.predictions` says how many were `cached` and how many `predicted`. Run B
  predicted 678; run C found all 678 in the cache (`P3-H2\real\B.solution.json`,
  `C.solution.json`).

**The mapper: deterministic per seed, and consensus decides attachment.**

- **Per seed, it is deterministic.** On a fixed database, two seeded single-thread GLOMAP
  runs were bit-identical, and two unseeded runs were not (the `solve()` docstring in
  `global_solve.py`, run P3-PM; `RUN\research\D1-sfm-slam-posegraph.md` §2.4).
- **But the seed matters.** On 6839fb8f, 5 mapper seeds on one database gave rooms of
  526, 526, 361, 525 and 526 keyframes. The closet detached under seed 2 alone
  (`P3-PF\PROGRESS.md`, 00:57).
- **Consensus answers that** (§1). A majority of 3, simulated over all 10 triples of
  those 5 seeds, flipped no group of 30 or more keyframes in any of 45 triple pairs. The
  control's rooms did not differ by a single keyframe (`P3-PF\PROGRESS.md`, 00:57).
- **In the product, run B** mapped seeds 0, 1 and 2, with rooms of 526, 525 and 349
  keyframes. The consensus published seed 0's draw, with a room of 526 and one area of
  57, and withheld nothing (`P3-H2\PROGRESS.md`).

**The database digest: at a stated precision** (`d06ebf5`).

- `solve.database_digest` is the SHA-1 of the database that was mapped, at the precision
  named in `solve.database_digest_rule`:
  - F, E, H and qvec up to scale and sign;
  - tvec up to scale;
  - each at 10 decimals of the matrix's largest entry.
- **Why.** The mask filter's re-verification left float noise in F on 3 of 16,793 planar
  pairs, where F is not used, and one F came back as −F.
- The exact digest (`content`) is still what the frozen matching compares
  (`global_solve.py`, `DATABASE_DIGEST_RULE`).

**What this adds up to:**

- **Two warm, same-seed final solves** (runs B and C, consensus 3, on one copy of
  6839fb8f) were compared on 11,975 values of `solution.json` and `components.json`.
  Timestamps, timings and per-solve file names were left out. Exactly one value
  differed: the database digest (`P3-H2\PROGRESS.md`).
- **At the stated precision,** that digest is equal too
  (`RUN\mailbox\to-manager\20260924-0256-024-recovered.md`).
- **A unit test** makes three re-finishes of one walk and compares them field for field
  (`test_world_builder_reproducible_finish.py`, `ae734c0`).
- **The real two-re-finish check** at the candidate SHA is phase R of the acceptance run
  (`P3-ACC\chain.sh`), and it has not reported yet.

## 7. OPEN

**Backlog, outside this round:**

- **Redactor precision.** The face redactor fills wallpaper, screens and furniture. On
  af47007c it filled 66 of 218 frames by 2 % or more, with a median fill of 22 %. That
  damages any solve without raw capture, and the appearance texture everywhere
  (`RUN\mailbox\to-lead\021-h1-decision-p2.md`).
- **Keyframe ids collide after a reconnect.** The builder derives keyframe ids from
  `source_seq`, so a reconnect that restarts numbering duplicates ids and overwrites
  stored keyframe images: 20 ids on adc75972 (`V8-REVIEW.md`, backlog). A re-finish
  gives those keyframes their stored redacted copy (40 on adc75972;
  `P3-PF\PROGRESS.md`, 02:40).
- **A minority piece inside the anchor block cannot be withheld.** The consensus never
  withholds the room's anchor group, so such a piece is only reported, in
  `gate.consensus.pieces` (`P3-H2\PROGRESS.md`).

**Also still open:**

- **No rollback command** exists for a re-finish that succeeded (§2.4).
- **The area floor** (30 keyframes or 5 s) comes from one case (contract T2).
- **The physical test** has no named Tower, port or world root (§5). The phone side
  belongs to the Mac.
- **The first computation.** Whether two fresh mask computations, or two fresh depth
  predictions, of the same images agree bit for bit has not been measured. The caches
  make that matter only for a walk's first finish (§6).
- **Per-round gate decisions** are persisted only when N ≥ 2, in `consensus.json`
  (`P3-H2\PROGRESS.md`).
