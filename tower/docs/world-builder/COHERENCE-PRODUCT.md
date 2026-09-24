# World Builder coherence — the switches, the re-finish, and a validation Tower

This is the operator's guide to the coherence product: the masked, seeded, gated final
solve with its consensus, components and areas, and the look-back relocalizer. The
contract is
[`docs/contracts/WORLD-BUILDER-COMPONENTS.md`](../../../docs/contracts/WORLD-BUILDER-COMPONENTS.md)
(v8). This file links to it and does not restate it.

- **Code described:** the product lane `world-builder/coherence-product-v1` at `e5f7151`,
  the P3.6 candidate (2026-09-24), and contract v8 (`60030cf`, docs only).
- **Status:** every behaviour here is **off by default**, and off is the Tower as it
  was before (`tower/tower/config.py`).
  - Review V9 found the P3.5 code safe to keep merged and not ready to switch on
    (`RUN\baseline\review\V9\rv9-report.txt`).
  - P3.6 (`0f9cc83`, `30c28c0`, `99420c4`, `c952be4`, `e5f7151`) fixes its findings.
  - The acceptance run, the full suite and review V10 are running at `e5f7151`, and have
    not reported yet (`RUN\status.md`).
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
| `TOWER_WORLD_SOLVE_MASKS` | `false` | Masks the wearer's hands, arms and held phone in every image of the session's **final** solve. With a walk database, the solve maps from a filtered copy of it (`walk-database-filtered`). Without one, it re-extracts features under the masks (`re-extracted`). The masks are cached per image (§6). An image that cannot be masked is left out of the solve (§4, "fail-safes"). The background solves during a walk are unchanged | `TOWER_WORLD_SOLVE` (on by default) and pycolmap. The GPU detectors (Grounding DINO + SAM 2.1, and OneFormer). If they cannot run, the solve runs **unmasked**, `transients.state` in `solution.json` says so, and a gated solve writes a notice (contract §3.1) | 132–451 s a world for 218–795 keyframes, or 0.54–0.60 s a keyframe, with an empty cache (`RUN\experiments\P3-VAL\TABLE.md`, code `e2e7582`). GPU peak 2,678 MB on 383 images (`RUN\experiments\P3-PM\out-masks2.log`). A later solve of the same images computes none: 690 of 690 came from the cache (`RUN\experiments\P3-H2\real\B.solution.json`) |
| `TOWER_WORLD_SOLVE_SEED` | unset | An integer ≥ 0 seeds every random generator of the final solve's mapper, and the mapper then runs on **one** thread. A seeded final solve also **freezes its matching** (§6). Unset, blank, `off`, or anything that is not a non-negative integer, means today's multi-threaded, unseeded solve. Recorded as `solve.seed` and `solve.threads` | `TOWER_WORLD_SOLVE`. The consensus needs it | Mapping is 3.3× slower: 111 s against 33 s at 383 keyframes (`RUN\research\D1-sfm-slam-posegraph.md` §0, §2.4). The seeded mapping took 23–181 s a world (`P3-VAL\TABLE.md`) |
| `TOWER_WORLD_SOLVE_GATE` | `false` | Runs on the final solve only. It computes depth before publishing (MoGe-2 ViT-L) and a metric scale per camera, then applies the evidence gate. It relabels the room and the unplaced pieces, and writes `solve\<session>\components.json`. The depth predictions are cached (§6). Off publishes the solve as the solver returned it | **Masks:** if `transients.state` is not `applied`, the gate attaches nothing outside the room's anchor block (reason `masks-unavailable`). **Metric scale:** if fewer than half the supported cameras have a ratio, it attaches nothing (reason `scale-unavailable`). If depth failed, the finisher owes a re-gate in place, except when the walk has no camera intrinsics or the solve has no camera: a re-gate would fail the same way, so those are not `retryable` and their notice names the owner who can fix them (V11 LOW-15). An exception in the gate publishes the solve **ungated**, with `components: null` (`coherence_publish.py`). Every fail-safe writes `finalization.notice`, from a closed set of sentences (contract §3.1, v8) | 40–164 s a world, of which depth is 39–160 s over 203–712 frames (`P3-VAL\TABLE.md`). With the predictions cached, the gate took 60 s against 190 s on 678 frames (`P3-H2\real\B.solution.json`, `C.solution.json`). The room's final surface reuses the depth (`coherence_publish.py`, step 6). Disk: one set of predictions per walk, 311 MB for 678 frames (`dense_pipeline.py`) |
| `TOWER_WORLD_SOLVE_CONSENSUS` | `1` | With N = 3, 5 or 7, a gated, seeded final solve maps N draws, with mapper seeds s … s+N−1, on its one frozen database, with the same masks and depth predictions. It gates each draw. **Only a draw whose gate attached votes.** Each keyframe votes attached to the room or not. It publishes the draw that agrees most with the strict majority; ties go to the lowest seed. A group of that draw's room that fewer than a strict majority attached is withheld as its own piece, reason `seed-unstable`, whatever its keyframes (V10 MED-4 removed the `kept` exception; how many of its keyframes every voting draw attached is recorded as `unanimous_keyframes`, for audit only). The room's anchor group is never withheld. The withhold is re-gated and checked keyframe by keyframe, and if the check fails the chosen draw is published unchanged (`not-applied`). **Accepted values** are 1, 3, 5 and 7. Unset or blank means 1; anything else (even, above 7, garbage) means 1 and is logged (`config.world_solve_consensus_setting`) | `TOWER_WORLD_SOLVE_GATE` and `TOWER_WORLD_SOLVE_SEED`. Without the seed, the consensus records `not-run`. If the first draw took a fail-safe, there is nothing to vote on: `not-needed`, or `deferred` to the re-gate that the fail-safe owes (`coherence_publish.py`). The states are listed in §4 | About 250–260 s per extra draw on 678 posed keyframes. That is ~190 s of single-thread mapping plus ~60 s of gate, of which ~55 s is the depth stage re-fitting cached predictions; no GPU. N = 3 adds about 8.5 min (`RUN\experiments\P3-H2\PROGRESS.md`). The cap exists to bound the cost: at N = 30, a typo, it would have been about 2 h under the writer lock (`config.py`; V9 M-3) |
| `TOWER_WORLD_AREA_BUILDS` | `false` | The idle finisher builds a surface and an appearance for each `shown_as: "area"` component, in `<world>\areas\<area_id>\`. Off: those areas are recorded as declined, which the phone shows as *could not be built*. `world_refinish.py` builds them either way | A components record, so the gate must be on. The finisher: `TOWER_WORLD_FINISH_PENDING`, `TOWER_WORLD_SURFACE` and `TOWER_WORLD_SOLVE` all on (`components.py`, `areas_nobody_will_build_reason`) | Per area: 8–17 s to prepare, plus 43–201 s of stages. That is 9 areas on 4 worlds; a walk had 0–5 areas (`P3-VAL\TABLE.md`) |
| `TOWER_WORLD_RELOCALIZER` | `off` | `prompt`: after a tracking loss, the builder matches incoming frames at 2 Hz against the 10 keyframes before the loss. When it relocalizes, it journals a verified revisit link. When it does not, it asks the wearer to look back: at most 2 prompts in any 60 s, with a 30 s cooldown (`relocalizer.py`). `silent`: it runs and records, but never prompts (the test's prompt-off arm). `on`, `true`, `yes` and `1` mean `prompt`. Any other word means `off`, and is logged | The **builder** reads it at each session start, from the environment it inherits from the Tower. It needs the session's calibration (`engine.py`). A final solve imports the revisit links only when the masks are `applied` and the solve is gated. It imports them only into the database it maps, never the walk database. Of those, it drops only the pairs the import itself created that have fewer than 50 verified inliers there; a pair that loop detection or an earlier solve verified stays (contract §2.5, `solve.revisit_pairs`; V9 M-9, M-10, `99420c4`). The relocalizer writes a link only when every live leg had 50 inliers or more (`relocalizer.REVISIT_MIN_INLIERS`). The phone speaks prompts only in a DEBUG build (contract §6.5) | 0.09–0.28 of one core at 2 Hz, only while an episode is open. It asked 0.81–1.29 times a minute when replayed on 7 walks (`RUN\experiments\P3-PR\out\PRODUCT.md`, measured before `2dcf680`) |

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
has none (contract §7 rule 5). V9 found old worlds unchanged (`rv9-report.txt`).

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

**What `e5f7151` does about it** (`world_finish_pending.refinish_in_progress`,
`world_refinish.recover_dead_refinish`):

- **A live re-finish.** The ledger records the re-finish's process, its pid and start
  time. While that process is alive, the finisher stays out of the world.
- **A re-finish that died before its solve published** (a closed console, a reboot).
  Its ledger is still at `setting-aside` or `set-aside`, and the finisher owes that
  session one thing first: `owed-refinish-restore`. Under the lock, it puts back what
  the re-finish set aside, and the ledger says `restored-after-the-refinish-ended`.
  - The next run of `world_refinish.py` does the same first.
  - If the solve child had already published, the new solve stands: the ledger says
    `published`, and the room is owed to the finisher.
  - If the put-back cannot complete, the session is **parked**. Its room is never built
    from what was left, and its notice says *a re-finish of this walk stopped part-way
    and the Tower could not put the previous result back; an owner can re-run the
    re-finish* (`REFINISH_PARKED_NOTICE`; V9 M-5, `c952be4`).
- **A ledger in a final state** holds nothing (`LEDGER_TERMINAL_STATES`).

**Why you still stop the Tower:**

- **Older code does not do this.** The validation Tower on :8020 still runs the
  integration branch at `1111bb9` (`RUN\status.md`), which merged this lane at `d87aa5c`,
  before any of it.
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
| `--capture-dir DIR` | the session's capture chain | a capture directory to search for each keyframe's own raw frame (repeatable; a relative path is resolved). It must hold its journal, `frames.jsonl` (§2.5) |
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
   room's surface and appearance as `stopped`. Under the same lock, it plans each
   keyframe's solver frame and writes the fresh solve's `sources.json` (§2.5).
2. **Runs `world_finalize.py` in a child process,** with `TOWER_WORLD_SOLVE_MASKS=1`,
   `TOWER_WORLD_SOLVE_GATE=1` and `TOWER_WORLD_SOLVE_SEED=<--seed>`. The child inherits
   the rest of the environment, `TOWER_WORLD_SOLVE_CONSENSUS` included.
3. **Builds the room:** its final surface and appearance.
4. **Builds the areas:** every `shown_as: "area"` component.

**When something goes wrong:**

- **In step 1** (V9 M-6, `c952be4`): one rollback covers everything from the first move
  to the last ledger write. A failure anywhere in it moves back what had moved, and
  the command refuses.
- **Between the set-aside and a published solve** (`3abe763`): if the final solve
  publishes nothing, the previous result goes back, and what the failed solve left is
  kept under `refinish\<stamp>\failed-*`. **Any** exception there, such as a read-only
  `session.json` (§2.6), does the same: the ledger says `restored-after-an-error`, and
  `error_after_set_aside` names the step and the error.
- **If the previous result cannot all go back,** the ledger says `restore-incomplete`,
  and lists what could not be done.
- **If the process is killed** before the solve publishes, the finisher, or the next
  run of the command, puts it back (§2.1).
- **In steps 3 and 4:** an exception ends the ledger at `stopped`, and the room is owed
  to the finisher (`c952be4`).
- **The finisher's attempt counters** restart only once the new solve is published, and
  a failure to restart them is not fatal. A put-back leaves them as they were.

If the gate wrote no record, the report's `components` is `null`.

### 2.4 What it sets aside, and how to roll back by hand

Everything goes under `<world>\refinish\<stamp>\`, where `<stamp>` is local time as
`YYYYMMDD-HHMMSS`. **Nothing is deleted.**

| what | how | to |
| --- | --- | --- |
| `solve\<session>` | moved | `refinish\<stamp>\solve\<session>` |
| every `areas\<area_id>` of the session | moved | `refinish\<stamp>\areas\<area_id>` |
| `surface\<session>`, `appearance\<session>` | copied | `refinish\<stamp>\<kind>\<session>` |
| `dense\<session>`, **without `predictions\`** | copied | `refinish\<stamp>\dense\<session>` |
| `derived\` | copied | `refinish\<stamp>\derived` |
| the session record | copied | `refinish\<stamp>\session.json` |
| — | written first | `refinish\<stamp>\refinish.json`: the ledger |

**The depth-prediction cache is left out of the copy** (V9 M-12, `c952be4`). It can be
re-derived, it is about 20 times the size of the keyframe images, and every re-finish
would otherwise keep another full copy. The live cache stays in place for the rebuild to
reuse, and the ledger's copy entry records `skipped: ["predictions"]`
(`world_refinish.py`).

**The ledger** (`refinish.json`) records:

- what moved or was copied, and from where;
- the session's previous finalization and stage records;
- the finisher's previous attempt counters;
- `process`: the pid and start time of the re-finish;
- `solver_frames`: where every solver frame comes from, and which of the walk's solver
  images were carried back (§2.5).

**The fresh `solve\<session>`** gets **copies** of the walk's inputs
(`SOLVE_COPY_BACK`):

- `database.db`, with its `-wal` and `-shm` files;
- `database.matching.json`, the record of its frozen matching (§6);
- the mask cache, `transients\`;
- `sources.json` and `camera.json`;
- **only those solver images that are proven to be what this re-finish plans** (§2.5).
  The features of every other image are cleared from the copied database, and
  `images.provenance.json` records where each image came from.

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
| `stopped` | a stop was asked for, the room could not be rebuilt, or steps 3–4 raised (`detail` says which) |
| `rolled-back`, `rollback-incomplete` | step 1 failed |
| `restored-after-a-failed-solve` | the solve published nothing, and the previous result was put back |
| `restored-after-an-error` | an exception after the set-aside, and the previous result was put back |
| `restored-after-the-refinish-ended` | the re-finish died before its solve published, and the finisher or a re-run put the previous result back |
| `restore-incomplete` | the previous result could not all be put back; see `restore_report` and `restore_errors`. A session whose re-finish ended here is parked (§2.1) |

**Rolling back a re-finish that succeeded.** There is no rollback command. The ledger's
own `restore` field gives the procedure:

1. Stop the Tower (§2.1).
2. Move the rebuild's `solve\<session>`, and its new `areas\<area_id>` directories, aside
   into that stamp's `refinish\<stamp>\`. Delete nothing.
3. Move each `moved` entry in the ledger back, from its `to` path to its `from` path.
4. The `copied` entries are snapshots of what the rebuild replaced in place: the room's
   surface, appearance and dense trees, `derived\`, and `session.json`. Move the rebuilt
   ones aside, and put the snapshots back. The snapshot of `dense\` has no
   `predictions\`; the live one can stay.

Deleting anything under `refinish\` needs a human's approval (filesystem policy,
rule 14). Nothing prunes these directories.

### 2.5 Where the solver's frames come from

**Each keyframe's planned frame** is the first of these that exists
(`plan_solver_frames`, in `global_solve.prepare_images`' own order):

1. the raw frame that the walk's builder recorded in `sources.json`, if that file is on
   disk;
2. the keyframe's **own** raw frame, found by its capture identity (below);
3. the session's stored keyframe copy, which is **face-redacted**. This is the fallback
   for each keyframe on its own: when its frame is not found, when more than one frame
   matches it, or when its keyframe id or image name occurs twice in the session.

**Duplicate keyframe ids** come from a reconnect that restarted numbering (§7). Their
recorded `sources.json` entry is dropped, and they use the stored redacted copy
(`dropped_builder_entries`, `ambiguous_keyframe_ids`).

**The walk's own solver images are reused only when proven** (V9 M-7, `c952be4`,
`99420c4`).

- The re-finish carries a walk image back only when its provenance record names the
  planned frame and matches the file's SHA-1 (`record`), or when undistorting the
  planned frame gives the very same bytes (`reproduced`).
- Every other image is **withheld**: `no-keyframe`, `planned-frame-unreadable`,
  `not-reproducible` or `differs-from-plan`. The solve writes it afresh from its planned
  frame, and its old features are cleared.
- So a carried-back raw image is never used for a duplicate-id keyframe, and a walk first
  solved from redacted copies is re-solved from raw frames when they are found.
- The solve itself rewrites any image whose `images.provenance.json` entry differs from
  its plan, or is missing when a raw frame is planned. It counts them in
  `solve.solver_images_rewritten`. A rewritten image misses the mask cache and the frozen
  matching, as it must (`global_solve.prepare_images`).
- On the frozen walks, 6839fb8f carried back 689 of its 690 images, and the other three
  carried back all of theirs. The one withheld, kf `00000936`, had been made from a
  redacted copy (`c952be4`).

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

- It counts `raw_from_sources_json`, `raw_from_capture_dir` and
  `redacted_session_copies`. It splits the redacted ones by why:
  `redacted_ambiguous_in_session`, `redacted_no_capture_frame` and
  `redacted_several_capture_frames`.
- `already_undistorted` counts the keyframes whose walk image was carried back.
  `walk_images` says what became of each: `carried_back`, `by_record`,
  `by_reproduction`, `withheld` (by reason) and `features_to_clear`.
- `source` sums it up: `raw-capture`, `redacted-session-keyframes` or `mixed`.
  `walk-solver-images` appears only in ledgers written before `c952be4`.
- It also records `from` and `why` (which captures), `ambiguous_keyframe_ids`,
  `dropped_builder_entries`, `capture_notes`, and whether `sources.json` was rewritten
  (`sources_json_written`).

**Without raw captures,** a keyframe's solver image and its masks come from its
redacted copy, fill boxes and all. That costs keyframes: af47007c, re-finished from its
redacted copies, posed 168 of its 218 (`RUN\experiments\P3-VAL\TABLE.md`, finding 6, code
`e2e7582`). On that walk the redactor filled 66 of 218 frames by 2 % or more, with a
median fill of 22 % (`RUN\mailbox\to-lead\021-h1-decision-p2.md`). `solver_frames` says
when this happens.

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

- **Disk.** The set-aside keeps full copies of the room's surface, appearance and dense
  trees, apart from `predictions\`, and of `derived\`. The run's seven re-finished copies,
  with their set-asides, took 3.1 GB (`RUN\experiments\P3-VAL\PROGRESS.md`, code
  `e2e7582`, before the prediction cache existed).
- **The prediction cache is bounded** (V9 M-12, `0f9cc83`). After each depth stage that
  completes, the cache keeps one set: the current token's, with one prediction per
  keyframe. Other tokens' sets, predictions no keyframe names any more, and a dead
  writer's staging files are pruned, and `prediction_cache.pruned` records it. The
  bound is the walk's keyframes, once: 311 MB for 678 frames (`dense_pipeline.py`).
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
   attached it, but fewer than a strict majority of the voting draws did. For such a
   piece it is the **only** reason; the withhold re-gate checks this per keyframe
   (`30c28c0`). The phone never shows reason text (Mac tip `8009854`,
   `RUN\mailbox\to-lead\022-mac-tip-8009854.md`).

Read §2.2 for what each one means.

**Fail-safes, and the notice.**

- A fail-safe shows as every piece outside the room carrying only `masks-unavailable`,
  or only `scale-unavailable`. Then read `transients.state` and the `gate` block in
  `solve\<session>\solution.json`: `gate.masks_applied`, `gate.metric_available`,
  `gate.retryable` and `gate.cause` (contract §2.5).
- **The row's `finalization.notice`** says what the Tower could not do and who can fix
  it. It is a **closed set** of fixed sentences (contract §3.1, v8): 26 causes from
  `coherence_publish.NOTICE_SENTENCES`, plus the finisher's given-up, refused and parked
  sentences, joined by `"; "`. Its only variable text is integer image counts and a gate
  clause from the same table. It never carries exception text, a path or a measured
  figure (V9 M-4, `30c28c0`).
- **The diagnostics are in `finalization.detail`,** and they reach clients, so they are
  client-safe: one line, no path, no traceback frame and no user name; the exception class
  and its message stay (`coherence_publish.client_safe_detail`; V10 MED-5, V11 MED-B).
  - The `GET /worlds` row and `/ws` `lifecycle.finalization` both carry it. The row's copy
    is made client-safe as it is served (`world_builder_library._client_safe_finalization`),
    `notice` included, and each text is at most 700 characters, the phone guard's own bound.
  - On an interrupted session, `lifecycle.reason` quotes it owner-facing, with no class name
    and no square brackets ("a path", "a user name"), and the phone shows that reason
    (`coherence_publish.owner_facing_detail`).
  - The full text stays in the Tower's log and in `solution.json`'s own records
    (`gate.detail`, `gate.depth.detail`, `transients.detail`). The session record keeps the
    writer's text; only what is sent is scrubbed.
- **Images that cannot be masked are excluded, with a bound** (V9 M-8, `0f9cc83`;
  `solve_masks.py`):
  - A failed hash or read is tried once more.
  - Each excluded image gets an all-0 mask, and the walk-database filter drops every
    match that touches it. `transients.images_excluded` counts them, and
    `excluded_reasons` says why: `hash-failed`, `read-failed`, `undecodable`,
    `wrong-size` or `no-mask`.
  - At max(3, 2 % of the images), `exclusion_notice_due` is set, and the notice says
    *{excluded} of {images} images could not be masked and were left out of the solve*.
  - If **no** image could be masked, the state is `unavailable`, with cause
    `images-unmaskable` and `none_masked`. The notice names the images, not the
    detector.
  - **Excluded means left out of a masked solve** (V10 L-11b, `315b6bf`). An image is kept
    out only by its all-0 mask and the filter, and both act only when the masks reach the
    solve (`extraction_masked: true`). So `images_excluded`, `excluded_examples`,
    `excluded_reasons` and `exclusion_notice_due` count only then. A record that is
    `unavailable` (`images-unmaskable` included) belongs to a solve that ran unmasked with
    every image in it: it reads 0, `[]`, `{}` and `false`, owes no exclusion notice, and
    `images_unmasked` and `detail` say what could not be masked. Such a record no longer
    carries per-reason counts (RV11-D LOW-3, backlog).
  - **`cache_write_failed`** (V10 L-12b): a mask the detector computed but could not write
    to the mask cache (a full disk, a held file, MAX_PATH) is used for this solve, kept in
    memory, and counted here. The key is present only when the count is above 0, in the
    solve's `transients` record and in the surface stage's transient record. The next solve
    finds no cache entry and computes the mask again. It is not a detector failure, and it
    does not make the masks `unavailable`.

**Where the consensus votes are** (contract §2.5, v7 and v8):

- **`gate.consensus`, in `solution.json`:**
  - `requested` (N), `seeds`, and `state`, with `why` where it applies:

    | state | meaning |
    | --- | --- |
    | `applied` | every requested draw voted |
    | `partial` | fewer draws voted than were requested (`votes.draws`). Only draws whose gate attached vote; with fewer than 2, draw 0 is published as one draw would be |
    | `not-applied` | the withhold re-gate failed its per-keyframe check; the chosen draw is published unchanged |
    | `deferred` | owed to the re-gate in place: draw 0's fail-safe is retryable, **or a stop** came before the further draws voted. On a stop, draw 0 is published with `retryable` and cause `consensus-deferred` |
    | `not-needed` | the first draw attached nothing (a fail-safe that owes nothing) |
    | `not-run` | it could not run, for example without a seed |

  - a summary per draw: its own seed and mapping time, `solve_identity`, room keyframes
    and agreement;
  - the `chosen` draw;
  - **`groups`:** the chosen draw's groups, with their `keyframe_ids`, `votes`,
    `ambiguous` (not unanimous), and `decision`: `anchor`, `attached`, `seed-unstable`,
    `unplaced`, or `not-withheld`. A `seed-unstable` group of the room is withheld whole,
    and carries `unanimous_keyframes`, the count of its keyframes every voting draw attached,
    for audit only (V10 MED-4 removed `kept`). `not-withheld` is a group the vote would have
    withheld when the withhold re-gate failed its check (`not-applied`): it stays in the
    published room (V10 L-4);
  - **`pieces`:** what the chosen draw's groups cannot show (`30c28c0`, `e5f7151`). It
    lists every other draw's unplaced piece on which the draws disagree, with:
    - its `keyframe_ids`, `from_draw`, `votes` and `majority_attached`;
    - `in_published_room`: whether the published room holds it;
    - `against_majority`: whether the published room disagrees with the majority about
      it.

    Pieces are reporting only. They are also listed in `ambiguous`. The 6839fb8f closet
    is the case they exist for: it was its own piece under mapper seed 2, and part of the
    room's anchor block under seeds 0 and 1 (`coherence_publish.decide_consensus`);
  - **`held_against_majority`:** how many keyframes the published room holds that a
    strict majority of the voting draws did not attach, anchor included. It is the
    visible size of a **named residual risk**. Marginal pieces absorbed into the anchor
    block are never withheld, so the vote does not re-verify them
    (`RUN\mailbox\to-lead\025-v9-and-option-a.md`, decision 2).
  - It is absent when N = 1.
- **`solve\<session>\consensus.json`:** each draw's per-round gate decisions. Written
  only when N ≥ 2 and further draws were mapped. A publish that writes no such record moves
  an older one aside to `consensus.superseded.json` (never deletes it), so it cannot sit
  beside a solve it does not describe (V11 LOW-4). That covers N = 1, the gate off, a
  consensus that is `not-needed` or `not-run` or was `deferred` before any further draw was
  mapped, and the solve's early publish of draw 0.

**While a consensus finish is finalizing, the row serves draw 0's `components`** (V11 LOW-3;
documented, not changed).

- A consensus solve publishes draw 0 first (`global_solve._publish_draw_0_first`), with its
  `components.json`, and then maps and gates the further draws. On acc2's timings that took
  168–281 s (RV11-B). During that window the session is `finalizing`, and the
  `GET /worlds` row's `components` are draw 0's.
- When the consensus publishes, `components` becomes the consensus's. A group that only draw
  0 attached can then leave the room and appear as its own piece, reason `seed-unstable`.
- Draw 0 is gated once. The consensus is handed the early publish's own gate result for it,
  so what it publishes is what one pass over the N draws publishes (V11 LOW-1).
- Once the room is built from the consensus, the phone sees a new room revision of the same
  walk and handles it under `WORLD-BUILDER-IOS.md` §10's existing rules: the page is swapped
  in when the Tower reports nothing building (`live: false`); otherwise the phone offers
  *"A newer reconstruction is ready. Show it"*. Nothing new is needed on the phone.

**`components.superseded.json`** is a record set aside when a solution was published
without the gate (`coherence_publish.py`).

## 5. The physical A/B test — the Tower side

The protocol is a run draft, `RUN\lead\PHYSICAL-TEST-PROTOCOL.md`, and is not in this
repository. This section covers only the Tower side. **The phone side is OPEN for the
Mac:** the DEBUG build, speech, and the device measurements M-a, M-b and M-c.

**Which Tower.**

- The integrated SHA, on a Tower that **records**: `TOWER_CAPTURE_ROOT` and
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

Leave `TOWER_WORLD_FINISH_PENDING` on (the default): the areas, any owed re-gate and any
deferred consensus are finished by the finisher.

The builder reads the relocalizer setting when each session starts, from the Tower's
environment. So to switch between arms, restart the Tower with the new value.

**Do not stop the Tower while a walk is finalizing.** With consensus 3, the final solve
ran about 8.5 min longer on a walk of 678 posed keyframes (§1), and a hard stop then
loses that final solve (§7, the first item).

**The phone build.** A DEBUG build (contract §6.5), from the Mac lane at `8009854` or
later. That build shows `finalization.notice` word for word under the room caption,
never on area screens (contract §8, v6; `RUN\mailbox\to-lead\022-mac-tip-8009854.md`).
The next integration takes the Mac tip `de1b045`, which also replaces anything that
looks like machine output (contract §3.1; `RUN\status.md`).

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
  `lifecycle.finalization` (`tower/docs/contracts/CARTRIDGE-RESULTS.md`). It is absent
  when nothing is owed.
- **`solve\<session>\solution.json`:**
  - `transients`, `solve.seed`, `solve.threads`, `solve.matching` and
    `gate.params_digest`. These show that the walk ran with the settings above;
  - `solve.revisit_pairs`: the relocalizer's links that the solve imported;
  - `gate.consensus`, read as in §4, with `held_against_majority` next to the harness's
    placement check.
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

Each source of variation, and what `e5f7151` does with it:

**Masks: cached.**

- They are kept per solver image, keyed by the image's name and the SHA-1 of its bytes,
  in `solve\<session>\transients\` (`solve_masks.py`). A re-finish carries the cache
  back (§2.4).
- A later solve of the same images calls no detector. H2's real runs took 690 of 690
  from the cache (`P3-H2\real\B.solution.json`, `C.solution.json`), and a unit test pins
  it for a re-finish
  (`test_world_builder_frozen_matching.py::test_a_refinish_maps_the_frozen_matching_and_hits_every_cached_mask`).
- The first masked solve of a walk computes them on the GPU. So does any solve of an
  image that was rewritten because its provenance did not match (§2.5).
- A cache file that is empty, truncated or the wrong shape is a miss, and is rewritten
  (`0f9cc83`).

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
  - The verification seed is part of the key, so a re-finish with a different `--seed`
    matches again (`99420c4`).
  - A walk database that changed after the freeze check is not called frozen: it is
    re-digested before the filter (V9 Q1, `99420c4`).
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
  the network was shown. `token` names the network, its parameters, the weights
  revision, the library versions, the device and fp16 (cache schema 2, `0f9cc83`).
- The fit to the solve is always recomputed. A fresh prediction is rounded through
  float16 before it is fitted, so fresh and cached fits are identical.
- A cache write that fails (a full disk, a sharing violation, MAX_PATH) is counted in
  `prediction_cache.write_failed`, and the stage goes on with the prediction it has. It
  no longer loses the depth (V9 M-11, `0f9cc83`).
- The cache is pruned to one set per walk (§2.7), and it is left out of the re-finish's
  set-aside copy (§2.4).
- **`prediction_cache.pruned.index_rewritten`** (V10 L-14, in `dense\<session>\align.json`):
  a set's index that was read but is torn (not JSON, not UTF-8, or no `keyframes` map) is
  rewritten from the stage that just completed, and nothing is pruned on the strength of it
  that run (`kept_because` says why). The prune runs again from the next completed stage. An
  index that cannot be opened at all is left alone, as before.
- **A prediction with no finite value is a miss** (V10 L-15): a cached one is predicted
  again (and logged), a fresh one is not kept, and a fit is never recorded `ok` with a
  non-finite scale or offset.
- **One lock for every writer of `dense\<session>\`.** A hand-run densify now also takes
  the session's surface lock, which the gate's and the surface's depth stages hold. If
  it cannot, it returns `unavailable` and writes nothing (V9 Q8, `dense_pipeline.py`).
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
- **In the product, run B** (before P3.6) mapped seeds 0, 1 and 2, with rooms of 526,
  525 and 349 keyframes. The consensus published seed 0's draw, with a room of 526 and
  one area of 57, and withheld nothing (`P3-H2\PROGRESS.md`).
- **What the vote cannot reach.** A marginal piece inside the room's anchor block is
  never withheld. `held_against_majority` counts it (§4, and §7).

**The database digest: at a stated precision** (`d06ebf5`, `99420c4`).

- `solve.database_digest` is the SHA-1 of the database that was mapped, at the precision
  named in `solve.database_digest_rule`:
  - F, E, H and qvec up to scale and sign;
  - tvec up to scale;
  - each at 10 decimals of the matrix's largest entry, with nothing appended. The pivot
    is chosen tie-safe.
- **Why.** The mask filter's re-verification left float noise in F on 3 of 16,793 planar
  pairs, where F is not used, and one F came back as −F.
- The exact digest (`content`) is still what the frozen matching compares
  (`global_solve.py`, `DATABASE_DIGEST_RULE`).

**What this adds up to:**

- **Two warm, same-seed final solves** (runs B and C, consensus 3, on one copy of
  6839fb8f, before P3.6) were compared on 11,975 values of `solution.json` and
  `components.json`. Timestamps, timings and per-solve file names were left out.
  Exactly one value differed: the database digest (`P3-H2\PROGRESS.md`).
- **At the stated precision,** that digest is equal too
  (`RUN\mailbox\to-manager\20260924-0256-024-recovered.md`).
- **A unit test** makes three re-finishes of one walk and compares them field for field
  (`test_world_builder_reproducible_finish.py`, `ae734c0`). It fakes the mapper and
  pycolmap, so it checks plumbing, not the real variance (V9, Tests).
- **The real two-re-finish check** at `e5f7151` is phase R of the acceptance run. Phase R
  runs the same seed twice on its own copy, so it exercises the frozen path, and the
  `database_digest` must be equal (`RUN\mailbox\to-manager\20260924-0448-p36-cut.md`). It
  has not reported yet.

## 7. OPEN

**Found in P3.6, not fixed:**

- **A stop during the builder's consensus: what survives (fixed in P3.7 and P3.8).**
  - The builder runs its final solve in a `world_solve.py` child. Neither that child nor
    `world_finalize.py` hands `should_stop` to the solve, and on a hard stop the builder
    terminates the child (`world_build_session.run_final`).
  - Since `b7450f3`, a consensus solve publishes draw 0 FIRST, as `deferred`, before it
    maps further draws. Since `2febf3a` (V11 MED-A), the builder records
    `final_solve: solved` when the child already published such a solve: gated, consensus
    requested, loadable, and `solved_at` no earlier than the launch. The row carries the
    CONSENSUS-DEFERRED notice, and the idle finisher re-runs the owed consensus and builds
    the surface. Rows written before `2febf3a` are healed by the finisher.
  - A stop before draw 0 was published still loses the final solve. The last background
    solution then stands, and an owner's re-finish redoes it, as for N = 1 today.
  - A stop reaches the consensus directly, publishing draw 0 as `deferred`, only for
    callers that hand it on: in-process solves, and the finisher's re-gate in place
    (`99420c4`, `30c28c0`).
  - A legacy row healed only on the finisher's first run can read "nothing will retry it"
    until then (`photographic._appearance_is_expected`).
- **The area levelling floor is below chance.** Isotropic normals gave `levelled: true`
  in 20 of 20 cases, and the *may look tilted* caption was suppressed with the up
  direction 34° off (`rv9-report.txt`, LOW "Areas and masks"; `area_build.py`).

**A named residual risk** (manager 025, decision 2):

- **A minority piece inside the anchor block cannot be withheld.** Marginal pieces
  absorbed into the room's anchor block are not re-verified by the vote. The gate's τ
  (16.8°) is about twice the image-only pair noise.
- `gate.consensus.held_against_majority` makes the risk visible, and the acceptance
  reports it per world. Per-group image-only verification inside the anchor is a
  next-run item (`RUN\mailbox\to-lead\025-v9-and-option-a.md`).

**Backlog, outside this round:**

- **Redactor precision.** The face redactor fills wallpaper, screens and furniture. On
  af47007c it filled 66 of 218 frames by 2 % or more, with a median fill of 22 %. That
  damages any solve without raw capture, and the appearance texture everywhere
  (`RUN\mailbox\to-lead\021-h1-decision-p2.md`).
- **Keyframe ids collide after a reconnect.** The builder derives keyframe ids from
  `source_seq`, so a reconnect that restarts numbering duplicates ids and overwrites
  stored keyframe images: 20 ids on adc75972 (`V8-REVIEW.md`, backlog). A re-finish
  gives those keyframes their stored redacted copy (§2.5): 40 on adc75972
  (`P3-PF\PROGRESS.md`, 02:40).

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
