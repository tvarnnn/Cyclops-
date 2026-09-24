# World Builder coherence — the switches, the re-finish, and a validation Tower

This is the operator's guide to the coherence product: the masked, seeded, gated final
solve, components and areas, and the look-back relocalizer. The contract is
[`docs/contracts/WORLD-BUILDER-COMPONENTS.md`](../../../docs/contracts/WORLD-BUILDER-COMPONENTS.md).
This file links to it and does not restate it.

- **Code described:** the product lane `world-builder/coherence-product-v1` at `4211876`
  (2026-09-24).
  - Other agents are changing `world_refinish.py`, `global_solve.py`, `relocalizer.py`,
    `coherence_publish.py`, `solve_masks.py` and `world_finish_pending.py` right now.
    Where that matters, this file says **(changing)** and describes `4211876`.
- **Status:** every behaviour here is **off by default**, and off is the Tower as it
  was before (`tower/tower/config.py`).
  - Review V8 found it safe to keep merged and **not ready to switch on**; its fix
    round is open (`RUN\baseline\review\V8\V8-REVIEW.md`).
- **Evidence:** `RUN` is `C:\Users\<you>\Projects\Glasses-scratch\wb-coherence-run-2026-09-23`.
  A path that starts with `RUN\` is there, not in this repository.
- **Machine:** every figure was measured on the development Tower: 20 cores and 32 GB
  (`RUN\lead\PHASE2-RULES.md`), and one 12 GB RTX 5070
  (`docs/world-builder-dense/04-OPERATIONS.md`).
- **Where this file lives:** `tower/docs` had no directory for World Builder operator
  docs, so this file starts `tower/docs/world-builder/`. The nearest precedent is
  `docs/world-builder-dense/04-OPERATIONS.md`.

---

## 1. The five switches

Set them in `tower\.env` (see `tower/.env.example`) or in the Tower's environment.

| setting | default | what it does | what it needs | cost measured in the run |
| --- | --- | --- | --- | --- |
| `TOWER_WORLD_SOLVE_MASKS` | `false` | Masks the wearer's hands, arms and held phone in every image of the session's **final** solve. With a walk database, the solve maps from a filtered copy of it (`walk-database-filtered`). Without one, it re-extracts features under the masks (`re-extracted`). The background solves during a walk are unchanged | `TOWER_WORLD_SOLVE` (on by default) and pycolmap. The GPU detectors (Grounding DINO + SAM 2.1, and OneFormer). If they cannot run, the solve runs **unmasked**, and `transients.state` in `solution.json` says so | 132–451 s a world for 218–795 keyframes, or 0.54–0.60 s a keyframe, with an empty cache (`RUN\experiments\P3-VAL\TABLE.md`). GPU peak 2,678 MB on 383 images (`RUN\experiments\P3-PM\out-masks2.log`) |
| `TOWER_WORLD_SOLVE_SEED` | unset | An integer ≥ 0 seeds every random generator of the final solve's mapper, and the mapper then runs on **one** thread. Unset, blank, `off`, or anything that is not a non-negative integer, means today's multi-threaded, unseeded solve. Recorded as `solve.seed` and `solve.threads` | `TOWER_WORLD_SOLVE`. It makes the **mapping** repeatable, on a fixed database only (§6) | Mapping is 3.3× slower: 111 s against 33 s at 383 keyframes (`RUN\research\D1-sfm-slam-posegraph.md` §0, §2.4). The seeded mapping took 23–181 s a world (`P3-VAL\TABLE.md`) |
| `TOWER_WORLD_SOLVE_GATE` | `false` | Runs on the final solve only. It computes depth before publishing (MoGe-2 ViT-L) and a metric scale per camera, then applies the evidence gate. It relabels the room and the unplaced pieces, and writes `solve\<session>\components.json`. Off publishes the solve as the solver returned it | **Masks:** if `transients.state` is not `applied`, the gate attaches nothing outside the room's anchor block (reason `masks-unavailable`). **Metric scale:** if fewer than half the supported cameras have a ratio, it attaches nothing (reason `scale-unavailable`). If depth failed, the finisher owes a re-gate in place. An exception in the gate publishes the solve **ungated**, with `components: null` (`coherence_publish.py`) | 40–164 s a world, of which depth is 39–160 s over 203–712 frames (`P3-VAL\TABLE.md`). The room's final surface reuses that depth (`coherence_publish.py`, step 6) |
| `TOWER_WORLD_AREA_BUILDS` | `false` | The idle finisher builds a surface and an appearance for each `shown_as: "area"` component, in `<world>\areas\<area_id>\`. Off: those areas are recorded as declined, which the phone shows as *could not be built*. `world_refinish.py` builds them either way | A components record, so the gate must be on. The finisher: `TOWER_WORLD_FINISH_PENDING`, `TOWER_WORLD_SURFACE` and `TOWER_WORLD_SOLVE` all on (`components.py`, `areas_nobody_will_build_reason`) | Per area: 8–17 s to prepare, plus 43–201 s of stages. That is 9 areas on 4 worlds; a walk had 0–5 areas (`P3-VAL\TABLE.md`) |
| `TOWER_WORLD_RELOCALIZER` | `off` | `prompt`: after a tracking loss, the builder matches incoming frames at 2 Hz against the 10 keyframes before the loss. When it relocalizes, it journals a verified revisit link. When it does not, it asks the wearer to look back: at most 2 prompts in any 60 s, with a 30 s cooldown (`relocalizer.py`). `silent`: it runs and records, but never prompts (the test's prompt-off arm). `on`, `true`, `yes` and `1` mean `prompt`. Any other word means `off`, and is logged | The **builder** reads it at each session start, from the environment it inherits from the Tower. It needs the session's calibration (`engine.py`). The phone speaks prompts only in a DEBUG build (contract §6.5) | 0.09–0.28 of one core at 2 Hz, only while an episode is open. It asked 0.81–1.29 times a minute when replayed on 7 walks (`RUN\experiments\P3-PR\out\PRODUCT.md`). **(changing: REL)** V8 M4: its revisit links are imported into every final solve at 15 inliers, and that has never been measured |

**Spelling.** The masks, gate and area-build settings go through `config._flag`:

- `1`, `true`, `yes` and `on` mean true.
- A blank value means the default.
- Anything else, including a typo, means **false**.

The seed and the relocalizer have their own readers, described in the table. In every
case a typo turns the behaviour off, never on.

**Who reads them.**

- **Masks, seed and gate:** the final solve's own process (`world_solve.py` or
  `world_finalize.py`).
- **Area builds:** the finisher.
- **The relocalizer:** the builder.

Each process inherits the environment the Tower started with. To change a setting,
restart the Tower.

**What the re-finish ignores.** `scripts/world_refinish.py` always runs with masks, the
gate, `--seed`, and area builds, whatever the Tower's settings are (§2).

**Old worlds do not change.** The finisher never computes components for a world that
has none (contract §7 rule 5). V8 found the pages of old worlds byte-identical.

## 2. Re-finish a saved world

`scripts/world_refinish.py` rebuilds one saved session with the product pipeline. It is
the only way an old world gets `components` (contract §7 rule 4). An owner runs it by
hand; nothing else does.

### 2.1 Stop the Tower first

**Stop every Tower that serves this world root**, or make sure the one that serves it
runs with `TOWER_WORLD_FINISH_PENDING=false`, as the validation Tower in §3 does.

**Why (V8 M3; `RUN\baseline\review\V8\rvx-report.txt`):**

- The re-finish runs in steps, and holds the world's lock only inside each step.
- After step 1, it records the room's stages as `stopped` ("re-finish in progress"), and
  a `stopped` stage is owed work.
- An idle Tower's finisher could therefore take the room in the gaps between steps 1→2
  and 2→3. It would build the room from a solve that was set aside or not yet rebuilt,
  and hold the lock that the next step needs.

**What `4211876` fixes (V8 M3b).** The finisher now stays out of a world while that
world's re-finish is live (`world_finish_pending.refinish_in_progress`).

**Why you still stop the Tower:**

- **Older code has no such check.** The integration branch at `1111bb9` merged this lane
  at `d87aa5c`, which is before the fix.
- **A re-finish that died between steps looks live.** Its ledger records no process ID,
  so the finisher leaves that world alone until you run the re-finish again (the
  "THE COST, stated" comment in `world_finish_pending.py`). **(changing: PF)** A process
  ID and a final state in the ledger have been asked for.
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
    `PYTHONPATH=<code>\tower` (`RUN\v8020\refinish_all.sh`).

| flag | default | what it does |
| --- | --- | --- |
| `--root` | required | the world root |
| `--world` | required | the world to rebuild |
| `--session` | the world's latest session | the session to rebuild |
| `--seed` | `0` | the final solve's `TOWER_WORLD_SOLVE_SEED` |
| `--threads` | `-1` | passed to `world_finalize.py` |
| `--appearance` / `--no-appearance` | on | build the room's and the areas' appearance |
| `--keep-depth-work` | off | keep the per-frame depth work afterwards |
| `--dry-run` | off | print what would be set aside, and the solve's settings; write nothing |
| `--format` | `json` | `json` or `text` |
| `--capture-dir DIR` (repeatable), `--no-capture` | — | **(changing: PF; not in `4211876`)** where the raw frames are (§2.5) |

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
   room's surface and appearance as `stopped`.
2. **Runs `world_finalize.py` in a child process,** with `TOWER_WORLD_SOLVE_MASKS=1`,
   `TOWER_WORLD_SOLVE_GATE=1` and `TOWER_WORLD_SOLVE_SEED=<--seed>`.
3. **Builds the room:** its final surface and appearance.
4. **Builds the areas:** every `shown_as: "area"` component.

If the final solve publishes nothing:

- it puts the previous result back;
- it keeps what the failed solve left under `refinish\<stamp>\failed-*`.

If the gate wrote no record, the report's `components` is `null`.

**A known gap at `4211876` (changing: PF).** An exception between the set-aside and the
solve leaves the world set aside. A read-only `session.json` is one such exception. The
live `solve\<session>` then holds only the copied-back inputs, while the session still
says `complete` (`RUN\experiments\P3-VAL\TABLE.md`, finding 8). PF's uncommitted change
puts the previous result back after any such exception, and records
`restored-after-an-error`.

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
- the finisher's previous attempt counters, which the re-finish resets.

**The fresh `solve\<session>`** then gets **copies** of the walk's inputs:

- `database.db`, with its `-wal` and `-shm` files;
- `images\`, `sources.json` and `camera.json`;
- the mask cache, `transients\`.

The copy keeps the masked solve on the walk's own database (`walk-database-filtered`).
Without the copy, the solve re-extracts features, and on the target walk that splits
the room (`world_refinish.py`, RV1 M1-1).

**The ledger's `state`:**

| state | when |
| --- | --- |
| `setting-aside` | while step 1 runs |
| `set-aside` | after step 1. **It stays `set-aside` after a successful re-finish**, because nothing writes a final state |
| `rolled-back` or `rollback-incomplete` | step 1 failed |
| `restored-after-a-failed-solve` | the solve published nothing, and the previous result was put back |

**Rolling back by hand.** There is no rollback command. The ledger's own `restore` field
gives the procedure:

1. Stop the Tower (§2.1).
2. Move the rebuild's `solve\<session>`, and its new `areas\<area_id>` directories, aside
   into that stamp's `refinish\<stamp>\`. Delete nothing.
3. Move each `moved` entry in the ledger back, from its `to` path to its `from` path.
4. The `copied` entries are snapshots of what the rebuild replaced in place: the room's
   surface, appearance and dense trees, `derived\`, and `session.json`. Move the rebuilt
   ones aside, and put the snapshots back.

Deleting anything under `refinish\` needs a human's approval (filesystem policy,
rule 14). Nothing prunes these directories.

### 2.5 Where the solver's frames come from (changing: PF, PB)

At `4211876`, each keyframe's solver image is the first of these that exists
(`global_solve._source_frame`, `prepare_images`):

1. the walk's own undistorted solver image, already in `solve\<session>\images\` (copied
   back, §2.4);
2. the raw frame named in `solve\<session>\sources.json`, if that file is on disk;
3. a raw frame in a `--capture-dir` handed to `world_finalize.py`. **The re-finish at
   `4211876` passes none;**
4. the session's own keyframe copy, which is **face-redacted**.

**The raw frames are solver input only.**

- They never leave the machine, and the only things derived from them that are
  published are points and poses (`_source_frame`).
- Point colours are withheld (`global_solve.WITHHELD_POINT_RGB`).
- A redacted build never draws on a solver image, not even for a mask
  (`solve_masks.py`).

**Without the raw captures,** the solve and its masks run on the redacted copies, fill
boxes and all. At `4211876` nothing in the report says so (V8 M3). On the run's worlds:

- 991e5a15 and af47007c had no walk solve directory, so they were re-finished from the
  redacted copies. af47007c posed 168 of its 218 keyframes (`P3-VAL\TABLE.md`,
  finding 6).
- Each of those two walks spans three chained captures, and the session names only the
  first. That first capture holds 132 of 229 keyframes and 164 of 218
  (`RUN\experiments\P3-PF\PROGRESS.md`, 23:55).

**The uncommitted change (changing: PF, PB).** The working copy adds `--capture-dir DIR`
(repeatable) and `--no-capture`.

- By default, it takes the capture that the session records under `TOWER_CAPTURE_ROOT`.
- It follows the captures that continue that one after a reconnect.
- It records where each solver frame came from, as `solver_frames` in the ledger and in
  the report.
- In PF's dry run, this found all 229 and all 218 raw frames (`P3-PF\PROGRESS.md`,
  23:55).

**`TOWER_SOURCES_ROOT`.**

- `sources.json` records paths as the builder was given them, usually relative:
  `data\captures\<capture>\frames\<n>.jpg`.
- A relative path is resolved against the `tower\` directory of the code that runs, or
  against `TOWER_SOURCES_ROOT` when that is set (`global_solve.sources_root`). It is
  never resolved against the working directory.
- So when you run code from anywhere but the Tower that recorded the walk, point
  `TOWER_SOURCES_ROOT` at a directory that holds `data\captures\...`:
  - the recording Tower's `tower\`; or
  - a copy laid out the same way. The run used
    `RUN\experiments\P3-VAL\src\data\captures\<capture>\frames\` (`RUN\v8020\refinish_all.sh`).
- The frames are only read.

### 2.6 Copied worlds: clear Windows' read-only attribute

A world copied from a read-only source, such as the run's frozen evidence, keeps the
read-only attribute on every file. The re-finish then fails writing `session.json` with
WinError 5, **after** it has set the solve aside (`P3-VAL\TABLE.md`, finding 8).

Clear the attribute **on the copy only**:

```
attrib -R "<copy root>\worlds\<world_id>\*" /S /D
```

The run did the same in Python, adding `stat.S_IWRITE` to every entry
(`RUN\experiments\P2-G0\tools\setup_copies.py`). Never clear it on frozen evidence or on
the live store.

### 2.7 What it costs

| code | worlds | locked GPU time per world | source |
| --- | --- | --- | --- |
| `e2e7582` | 7, with 218–795 keyframes | 6.5–27.1 min | `RUN\experiments\P3-VAL\TABLE.md` |
| `1111bb9` | 3, with 383–398 keyframes | 13.7–16.0 min | `RUN\experiments\GPU-JOBS.log` |

Where the time goes, on 6839fb8f (690 keyframes, 25.0 min in all; `P3-VAL\TABLE.md`):

| stage | time |
| --- | --- |
| masks | 384 s |
| matching | 47 s |
| mapping | 181 s |
| gate (mostly depth) | 164 s |
| the room's surface | 495 s |
| the room's appearance | 127 s |
| one area | 83 s |

- **Disk.** The set-aside keeps full copies of the room's surface, appearance, dense and
  derived trees. The run's seven re-finished copies, with their set-asides, took 3.1 GB
  (`RUN\experiments\P3-VAL\PROGRESS.md`).
- **GPU.** Run one re-finish at a time. The run serialised GPU jobs with its own lock
  (`RUN\lead\gpulock.py`), which is not part of the product.

## 3. Stand up a validation Tower (serve only)

A validation Tower serves re-finished **copies** to a phone. It does not touch the live
store or port 8000.

**The worked example** is the Tower that the Mac gate used, on 2026-09-23/24:
`RUN\v8020\start_tower.ps1` and `RUN\v8020\refinish_all.sh`. Its record is
`RUN\mailbox\to-manager\20260924-0025-015-delivered.md`.

**Its rules:**

- its own world root and its own port;
- never the live store, never port 8000, and never a checkout's `.env`;
- serve only: `TOWER_WORLD_FINISH_PENDING=false` and `TOWER_WORLD_AUTOBUILD=false`.

**1. Copy the worlds** into `<V>\worlds\<world_id>`. `<V>` is a directory under
`Glasses-scratch\`, never under the live store.

**2. Clear the read-only attribute** on the copies (§2.6).

**3. Lay out the raw captures** as `<S>\data\captures\<capture>\frames\` (§2.5).

**4. Re-finish each world, one at a time, before the Tower starts.** In Git Bash:

```bash
V='C:\Users\<you>\Projects\Glasses-scratch\<run>\v<port>'       # the validation root
CODE='C:\Users\<you>\Projects\Glasses-worktrees\<worktree>'      # the code under test
PY='C:\Users\<you>\Projects\Glasses\tower\.venv\Scripts\python.exe'
export PYTHONPATH="$CODE\\tower"
export PYTHONDONTWRITEBYTECODE=1
export TOWER_SOURCES_ROOT='<S>'                                  # holds data\captures\...
mkdir -p "$(cygpath -u "$V")/refinish" "$(cygpath -u "$V")/logs"
for W in <world_id> <world_id>; do
  "$PY" "$CODE\\tower\\scripts\\world_refinish.py" --root "$V" --world "$W" --seed 0 \
    > "$(cygpath -u "$V")/refinish/${W:0:8}.json" 2> "$(cygpath -u "$V")/logs/${W:0:8}.stderr.log"
done
```

The run wrapped each call in `RUN\lead\gpulock.py run ...`, and logged each start and
end to `logs\timing.log`.

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

Read §2.2 for what each one means.

**What a fail-safe looks like.** Every piece outside the room carries only
`masks-unavailable`, or only `scale-unavailable`. Then read `transients.state` and the
`gate` block in `solve\<session>\solution.json`: `gate.masks_applied`,
`gate.metric_available`, `gate.retryable` and `gate.cause` (contract §2.5).

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

**Flags:**

| setting | value |
| --- | --- |
| `TOWER_WORLD_SOLVE_MASKS` | `true` |
| `TOWER_WORLD_SOLVE_SEED` | an integer |
| `TOWER_WORLD_SOLVE_GATE` | `true` |
| `TOWER_WORLD_RELOCALIZER` | `prompt` for walks W-A, W-C and W-G; `silent` for W-B |
| `TOWER_WORLD_AREA_BUILDS` | **OPEN.** The protocol does not say. Without it, the Tower does not build the walk's areas, and they read *could not be built* until somebody re-finishes the walk |

The builder reads the relocalizer setting when each session starts, from the Tower's
environment. So to switch between arms, restart the Tower with the new value.

**What to look at, for each walk:**

- **The journal,** `<world>\sessions\<session>\events.jsonl`:
  - `relocalizer_started`, with the `acceptance`, the `limiter` and `prompts_enabled`
    that the walk ran with;
  - `recovery_prompted`, `recovery_withheld`, `recovery_accepted`, `recovery_timed_out`,
    `recovery_anchored` and `relocalizer_stopped` (`relocalizer.py`).
  - Prompts per minute come from `recovery_prompted`. The pass bar is 2 or fewer.
- **The status channel's `tracking.recovery`,** including `prompt.issued_at` for
  measurement M-c (contract §6.2).
- **`solve\<session>\solution.json`:** `transients`, `solve.seed`, `solve.threads`, and
  `gate.params_digest`. These show that the walk ran with the settings above.
- **`solve\<session>\components.json`,** read as in §4.
- **The run harness,** `tower\scripts\world_coherence_eval.py`, with `--renders`. It lives
  on the lane branch `world-builder/coherence-v1`, not on this one.
- **Measurement M-b** needs the Tower's received-frame rate and resolution in the 5 s
  either side of each prompt, taken from the walk's capture.

**Prior evidence** (a replay, not a walk): of 53 anchored revisit links that could be
checked, 2 were wrong by more than 8°, both on the control. The protocol's bar is 0
(`RUN\experiments\P3-PR\out\PRODUCT.md`).

Keep the world directory, the journal and the capture. Raw imagery stays on this machine.

## 6. What is not deterministic

**`--seed` makes the mapping repeatable. It does not make a re-finish repeatable.**

**Repeatable: the mapper, on a fixed feature database.**

- Two seeded single-thread GLOMAP runs on one database were bit-identical; two unseeded
  runs were not (`global_solve.py`, the `solve()` docstring, run P3-PM).
- Mapper seed 0 on PF's frozen database and depth reproduced PF's seed-0 solve exactly:
  29,538 points, and a room of 526 keyframes (`P3-PF\PROGRESS.md`, 00:40).

**Not repeatable:**

- **Matching.** Every final solve matches into the walk database again, with threaded
  sequential matching and loop detection.
  - Two threaded runs matched 62 of 7,450 sequential pairs differently. One thread
    matched all 7,450 identically, but took 115 s against 20 s
    (`RUN\experiments\P3-PM\out-match-repeat.log`).
  - On 6839fb8f, the walk database went from 16,740 matched and 7,947 verified pairs, to
    16,779 and 7,950 after one run, and to 16,773 and 7,952 after another
    (`P3-PF\PROGRESS.md`, 00:10).
- **Masks,** where they must be computed. They are computed on the GPU, and review V8
  reports that they are not bit-reproducible (`rvx-report.txt`). No run measured two
  fresh computations against each other: OPEN.
  - They are cached per image, keyed by the image's name and SHA-1, in
    `solve\<session>\transients\`. A re-finish copies the cache back (§2.4).
  - So a walk re-finished a second time takes its masks from the cache. PF's 6839fb8f
    seed runs had 690 cache hits out of 690 (`P3-PF\seeds\s0`–`s2`, `solution.json`),
    and still differed.
- **Depth,** which the gate uses for metric scale. The lead's H2 plan says it is
  recomputed on the GPU on every run
  (`RUN\mailbox\to-manager\20260924-0030-019-ack-h2-design.md`). The gate reuses the
  walk's own predictions only where they were made the same way
  (`coherence_publish.py`). Whether depth is bit-identical from run to run has not been
  measured: OPEN.

**What that does to a result:**

- **Same seed, same walk, two code versions (`e2e7582` and `d87aa5c`),** with mapping code
  that did not change between them (`P3-PF\PROGRESS.md`, 00:10):
  - the room's membership was identical: 526 keyframes each time;
  - the gate placed 542 solver images against 550;
  - the harness counted 3 failing islands against 2.
- **2f447162 at `1111bb9`, against P3-VAL:** 3,291 pairs against 3,289, a room of 202
  keyframes against 200, and 10 components against 9 (`V8-REVIEW.md`, H2).
- **The mapper seed alone** flips the closet of 6839fb8f. Seeds 0–3 gave rooms of 526,
  526, 361 and 525 keyframes (`P3-PF\PROGRESS.md`, 00:40).

**(changing: H2 agent)** Manager decision 019 makes reproducibility a product property:
cache what is not deterministic, and decide attachment by a consensus of seeds
(`RUN\mailbox\to-lead\019-h2-variance-position.md`).

## 7. OPEN

- **Reproducibility.** Two re-finishes with the same seed are not identical (§6). The
  fix is in progress: the H2 agent, manager 019.
- **A re-finish that died between steps** parks the world for the finisher, until the
  owner runs it again. The ledger records no process ID (PF).
- **No rollback command** exists for a re-finish that succeeded (§2.4).
- **The area floor** (30 keyframes or 5 s) comes from one case (contract T2).
- **The physical test's Tower,** port and world root, and whether it has
  `TOWER_WORLD_AREA_BUILDS` on (§5). The phone side belongs to the Mac.
- **Mask and depth reproducibility** are unmeasured (§6).
- **Relocalizer revisit links** are imported into the final solve at 15 inliers (V8 M4;
  changing: REL).
- **`relocalizer.py:48`** still says it is not wired into `global_solve`. It is wired in
  (V8 M5; REL owns that file).
