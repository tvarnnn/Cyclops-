# World Builder coherence — handoff and physical-test guide

**Run `wb-coherence-run-2026-09-23`, handed over 2026-09-24.** This is the operational
companion to the manager's report to Tristan. It serves as the physical-test guide and as
the next run's starting point.

| | |
|---|---|
| Integration | `world-builder/live-world-visualization-v1` at `<SHA>` (the test runs this) |
| Tower lane | `world-builder/coherence-product-v1` (worktree `Glasses-worktrees\wb-coherence-product`) |
| iOS lane | `ios/wb-coherence-areas-v1`, at the Mac tip merged into `<SHA>` (`0e1c78d`, manager 035 §5) |
| Contract | `docs/contracts/WORLD-BUILDER-COMPONENTS.md` (v9) |
| Operator detail | `tower/docs/world-builder/COHERENCE-PRODUCT.md` (switches, re-finish, validation Tower) |
| Run root (`RUN`) | `C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23` |

Nothing is on `main`. Every new behaviour is **off by default**, and off is the Tower as
it was before (`tower/tower/config.py`).

**Where each figure comes from.**

- **[acc]** means acceptance at the tested code. That is `RUN\acc2\TABLE.md`, run on
  `e5f7151`, and carried forward to `<SHA>` under manager 029's conditions.
  - One of those conditions is the same-seed check (d). At `315b6bf` it re-finished the
    control and the target **warm, on identical inputs** (acc2's w0).
    - `components.json` and `solution.npz` were identical apart from `solved_at` and
      `solve_identity`.
    - `solution.json` differed only in fields listed as by design
      (`RUN\dchk\logs\*.compare-final.txt`).
  - P3.9 then changed the draw-0 gate path, so (d) is being re-run at `a5001ab`
    (`RUN\dchk2`): <D-AT-SHA>.
- **[earlier]** means an earlier phase of the run. The source is named each time.

---

## 1. What changed, in plain words

**The problem.** The target walk (world `6e6d3fc3`, Tristan's 06:01 walk of 2026-09-23)
had its bathroom and a bed block glued into the bedroom at a wrong tilt, scale and
position. The glue was feature matches on the phone in his hand.
- The two islands were held together by 7 verified pairs with 136 inliers. 85 % of those
  inliers, about 116, lay on the held phone [earlier: `RUN\baseline\FORENSICS.md` H-A].
- Masking hands, arms and held phones removed 117 of the 136 and left 0 cross-island pairs
  [earlier: `RUN\mailbox\to-manager\20260923-1813-candidate-architecture.md` §1].

**The answer.** Stop gluing on false evidence, and stop inventing placement. A piece of
the walk joins the room only on independent, consistent evidence. Anything else is shown
**on its own, as an area**, never inside the room at a guessed pose.

| change | in plain words | modules (`tower/tower/world_builder/` unless noted) |
|---|---|---|
| **Masks before SfM** | Hands, arms and the held phone are masked in every image of the final solve before its features are used. The walk's own feature database is filtered, not rebuilt | `solve_masks.py`, `transients.py`, `global_solve.py` |
| **Evidence gate, with honoured links** | After mapping and before publishing, a group of cameras joins the room only with ≥ 2 independent verified links (or a closed triangle) that the solve itself agrees with (within τ = 16.8°), **and** a matching metric level (MoGe depth, within ×1.25). Depth now runs before publishing | `coherence_gate.py`, `coherence_scale.py`, `coherence_publish.py`, `dense_pipeline.py` |
| **Components and areas** | Every piece of the walk is listed with the reason it was not placed. A piece of ≥ 30 keyframes or ≥ 5 s is built as its own levelled **area**, shown on its own routes and never positioned into the room. Smaller pieces are only counted | `components.py`, `area_build.py`, `tower/tower/routes/geometry.py`, `tower/tower/results/world_builder_library.py`, `tower/tower/results/world_builder_render.py` |
| **Consensus, N = 3** | The final solve is mapped with 3 mapper seeds on one database. A group of the room joins only if a strict majority of the draws attach it; otherwise it is withheld (`seed-unstable`). The chosen draw's room is never enlarged by the vote | `coherence_publish.py` (`gate_by_consensus`), `global_solve.py` |
| **Frozen matching and caches** | A seeded final solve freezes its matched feature database (`database.matching.json`). Masks are cached per image, and depth predictions per input pixels. A second finish of the same walk with the same seed maps the same database | `global_solve.py`, `solve_masks.py`, `dense_pipeline.py` |
| **Relocalizer and look-back prompt** | While live, after a tracking loss, the builder looks for the view it lost. If it cannot find it within 5 s, the phone says *"Look back the way you came."* (at most 2 prompts a minute). A found view becomes a verified revisit link that the final solve may import | `relocalizer.py`, `engine.py`, `tower/tower/results/world_builder.py` (`tracking.recovery`); iOS speaks it |
| **Fail-safes and notices** | If masks or metric depth are missing, the gate attaches nothing beyond the room's core, and says why. Each walk's row carries `finalization.notice`: a fixed sentence from a closed set saying what the Tower could not do and who can fix it. The idle finisher re-runs a gate or a consensus that is owed | `coherence_publish.py` (`NOTICE_SENTENCES`), `tower/scripts/world_finish_pending.py` |
| **The re-finish command** | `tower/scripts/world_refinish.py` rebuilds one saved walk the new way. It sets the previous result aside under `refinish\<stamp>\`, deletes nothing, finds raw frames by capture identity, and rolls back on failure | `tower/scripts/world_refinish.py` |
| **G0** | The surface stage refuses depth fits that are not physical. This removed the flying wall sheets (voxel coarsening on the target 2.03× → 1.00×) [earlier: 1813 §1] | `surface_pipeline.py` (integrated at `17e6d3d`) |

**How it was checked:**

- **Reviews V5 to V12**, all fresh and adversarial, ended APPROVE or READY WITH CHANGES.
  - Every must-fix was fixed. V11's MED-B is fixed at `6ae08c1` and `a5001ab`.
  - V12 found the P3.9 code sound, and asked for doc changes only
    (`RUN\baseline\review\V8\V8-REVIEW.md` … `V12\V12-REVIEW.md`).
- **The full suite** at `a5001ab`, the final product-lane code: 5454 passed, 87 skipped,
  1 xfailed, 0 failed (`RUN\lead\suite-a5001ab.log`).
  - The run at `2febf3a` had one known Windows timing flake, the finisher-chore kill test
    (`RUN\status.md`).
- **The acceptance run:** `RUN\acc2\TABLE.md`, with the control and the target complete.
  - At 11:44 on 2026-09-24 the chain's early-stop gate stopped it on 2f447162, another walk of
    the same bedroom. The room differs by 1 to 5 keyframes between seeds (at most 2.51 %), with
    GT 0 misplaced in all five runs (`RUN\acc2\STOP`;
    `RUN\mailbox\to-manager\20260924-1207-acc2-gate-stop-2f447162.md`).
  - It waits for a manager ruling. 6839fb8f and phases R, B and H have not run.

## 2. The switches

Set them in `tower\.env`. The test builds its own copy (§3.1), so Tristan's `.env` is
never touched.

| setting | default | physical test | what it does |
|---|---|---|---|
| `TOWER_WORLD_SOLVE_MASKS` | `false` | `true` | masks on the final solve; the gate's hard dependency |
| `TOWER_WORLD_SOLVE_SEED` | unset | `0` | seeded, single-thread mapper; freezes matching; the consensus needs it |
| `TOWER_WORLD_SOLVE_GATE` | `false` | `true` | depth before publish, the evidence gate, `components.json`, the notice |
| `TOWER_WORLD_SOLVE_CONSENSUS` | `1` | `3` | mapper-seed draws; accepts 1, 3, 5 or 7 only, and anything else reads as 1 and is logged |
| `TOWER_WORLD_AREA_BUILDS` | `false` | `true` | the idle finisher builds each area; off shows *could not be built* |
| `TOWER_WORLD_RELOCALIZER` | `off` | `prompt`, then `silent` | the A/B arm; the builder reads it at each session start |
| `TOWER_WORLD_FINISH_PENDING` | `true` | `true` | finishes areas, owed re-gates and deferred consensus |
| `TOWER_WORLD_SOLVE`, `_SURFACE` | `true` | `true` | the finisher runs only with these two and `TOWER_WORLD_FINISH_PENDING` on (`tower/tower/main.py`, `_world_finish_spec`) |
| `TOWER_WORLD_APPEARANCE` | `true` | `true` | builds the room's and the areas' appearance; for the finisher it only switches `--appearance` |
| `TOWER_WORLD_DENSIFY` | `false` | `false` | the gate runs its own depth |
| `TOWER_WORLD_AUTOBUILD`, `_REGISTER`; `_REBUILD_EVERY` | `true`; `4` | same | unchanged |
| `TOWER_CAPTURE_ROOT`, `TOWER_WORLD_ROOT` | unset | `data`, `data/world_builder` | Tristan's values; they reach the live store through the worktree's junction |
| `TOWER_SOURCES_ROOT` | the code's `tower\` | `C:\Users\tvllo\Projects\Glasses\tower` | where relative raw-frame paths resolve |
| `TOWER_WORLD_RAW_IMAGERY`, `TOWER_WORLD_TRANSIENTS` | unset | unset | the appearance stays redacted; the detector stays on |

Sources: `tower/tower/config.py` and `tower/.env.example` at the product lane head; the test
values are in `RUN\lead\deploy\make-test-env.ps1` and `RUN\lead\deploy\DEPLOY-PLAN.md` §2.

**What the re-finish decides for itself.** `world_refinish.py` always runs with masks, the
gate and its own `--seed`, and it builds areas whatever the Tower says. It takes the
consensus from its own environment; `refinish-target.ps1` sets 3.

## 3. The physical test

The protocol is `RUN\lead\PHYSICAL-TEST-PROTOCOL.md`: arms, pass criteria and device
measurements. Its source is 1813 §7.3, as amended by manager 010 decision 4.

### 3.1 Setup (the lead runs these; `$D` is `RUN\lead\deploy`)

```powershell
$D = 'C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\lead\deploy'
```

1. **Pause the acceptance run, for GPU exclusivity.** Create `RUN\acc2\STOP`, then wait until
   `RUN\acc2\logs\chain.log` shows that the chain stopped after its current run. It
   resumes where it left off, after the checks in §4.1.
2. **Re-finish the 06:01 walk in place,** with no Tower running:
   `powershell -NoProfile -ExecutionPolicy Bypass -File "$D\refinish-target.ps1" -Sha <SHA>`.
   - It holds the GPU for about 15 min: the acceptance's cold re-finishes of this walk took
     a median of 14.5 min [acc].
   - It checks its own dry run first: every solver frame must come from the raw capture.
   - It writes the set-aside to
     `C:\Users\tvllo\Projects\Glasses\tower\data\world_builder\worlds\6e6d3fc30e7b45f3a7521e618386b649\refinish\<STAMP>\`.
3. **Build the test `.env`:** `powershell -NoProfile -ExecutionPolicy Bypass -File "$D\make-test-env.ps1" -Sha <SHA>`.
   - It is Tristan's `.env`, byte for byte, plus the switches of §2. It is written to the
     integration worktree's gitignored `tower\.env`.
   - The canonical `tower\.env` is never written.
4. **Start the test Tower, prompt arm:**
   `powershell -NoProfile -ExecutionPolicy Bypass -File "$D\start-test-tower.ps1" -Sha <SHA> -Relocalizer prompt`.
   - It runs the integration worktree on port 8000, against Tristan's live store, detached,
     so it survives SSH.
   - It refuses when :8000 is taken, HEAD is not `<SHA>` or the tree is dirty, the `.env`
     lacks a switch, `import tower` does not load the worktree, the run's GPU lock is held,
     or the finisher finds owed work on his worlds.
5. **Confirm that the Tower is on the right SHA and switches:**
   - `git -C C:\Users\tvllo\Projects\Glasses-worktrees\wb-live-visualization-v1 rev-parse HEAD`
     prints `<SHA>`.
   - The last line of `RUN\physical-test\tower-starts.jsonl` shows `sha`, `relocalizer`,
     `server_pid` and `health`.
   - The end of `...\wb-live-visualization-v1\tower\.env` holds the appended switch block.
   - After the first walk, its `solve\<session>\solution.json` shows `transients.state:
     "applied"` with `device: "cuda"`, `solve.seed: 0` and `gate.consensus.requested: 3`.
     Its `sessions\<session>\events.jsonl` shows `relocalizer_started` with
     `prompts_enabled: true` in the prompt arm, and `false` in the silent arm.

**What was checked in advance.**

- **The dry runs** (`DEPLOY-PLAN.md` §8), and **CUDA under the detached (WMI) launch**: it
  passed on the RTX 5070, with masks `applied` on cuda (`RUN\physical-test\preflight\cuda-wmi.json`).
  Both were checked at earlier SHAs, and both are **re-run at `<SHA>`** before the hand-over.
- **A launch from an SSH session is not proven.** If the Tower will be started over SSH,
  run the pre-flight over that same SSH session first:
  1. Move the old result aside, with a move, never a delete. Otherwise an old PASS
     survives a probe that died:
     `$P = 'C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\physical-test\preflight'; Move-Item "$P\cuda-wmi.json" "$P\cuda-wmi.$(Get-Date -Format yyyyMMdd-HHmmss).json"`.
  2. Run `powershell -NoProfile -ExecutionPolicy Bypass -File "$D\preflight-cuda-wmi.ps1" -WaitMinutes 30 -EnvFile C:\Users\tvllo\Projects\Glasses-worktrees\wb-live-visualization-v1\tower\.env`.
     The default `-WaitMinutes 0` returns right after the launch, without the result.
  3. Check that a new `cuda-wmi.json` exists, says PASS, and that its `env.at` is today's
     run.
- **The validation Tower on :8020 can stay up.** It is serve-only, on the CPU
  (`TOWER_CV_DEVICE=cpu`), with no finisher (`TOWER_WORLD_FINISH_PENDING=false`;
  `RUN\v8020\start_tower.ps1`).
- **His 167 saved worlds:** the finisher's dry run finds 0 owed, so nothing is rebuilt on
  start or when idle. Old worlds look exactly as today (`DEPLOY-PLAN.md` §3).

**The phone:**

- the **DEBUG build** from Xcode, at the Mac tip merged into `<SHA>`. A Release build cannot
  capture or speak (contract §6.5);
- start each walk **on the World Builder screen**, and lock the phone there;
- do **not** open Saved Worlds during a walk: a pinned view never speaks.

### 3.2 What Tristan sees on the re-finished 06:01 walk

- **The room** is one compact room: the desk, the bed and, in one outcome, the closet. The wrong-scale bathroom, the
  flying wall sheets and the flung cameras are gone [earlier: `RUN\experiments\P3-VAL\TABLE.md`,
  renders]. An open viewer offers *A newer reconstruction is ready*.
- **The areas row:** *2 more areas — captured on this walk but not placed in this room.*
  - One is the bed pass, 66 photos.
  - The other is the bathroom, 55 photos, clean and levelled [acc: every run shows the 66
    and 55 areas].
  - With the closet in the room, the first reads *Area 1 · 0:29–0:33 and 1:49–2:12 · 66
    photos* [earlier: `DEPLOY-PLAN.md` §4].
- **The footer:** *3 short stretches (21 photos) could not be placed or shown* (10 + 6 + 5)
  [acc].
- **The notice:** none is expected. On this walk the masks and the metric depth both
  worked [acc: attach, masks_applied and metric_available are true in every run].
- **The closet: one of two outcomes.**
  - The room holds **226** keyframes with the closet in it,
  - or **148**, with the closet as a **third area of 78 photos**, listed first because it
    is the largest.
  - Across 5 acceptance runs it was 226 / 226 / 148 / 226 / 148 [acc]. There was no
    false placement in any run (GT: 0 misplaced), and 90.6 % of the walk was shown
    somewhere in every run [acc].

**The closet, in the manager's words (035 §4, verbatim).** Tristan is told in plain words
that:

> - after the in-place re-finish, his closet may appear either in the room or as a separate area;
> - both are honest outcomes;
> - once finished, re-opening shows the same result;
> - walking the closet doorway slowly, or looking back when prompted, is what makes it attach reliably.

### 3.3 The walks, grouped by arm (one restart)

**Route A.** Desk → closet → bed → bathroom doorway → **20 s inside the bathroom** → back to
the desk, with **one deliberate whip pan** at each of the doorway, the closet corner and the
bed corner.

**Group 1, prompt ON** (the Tower as started in §3.1):

| walk | route | count |
|---|---|---|
| W-A | route A, with the whip pans | 3 |
| W-C | route A with **no** whip pans | 1 |
| W-G | **generalization:** other rooms of the apartment, the hallway plus the living room or kitchen, 150–250 s | 1 |

**Between walks,** wait until the phone shows the room **saved**. Its area builds may still
be *Improving*.

- A walk with consensus 3 takes about 13 min of final solve plus about 4 min of room
  surface, and **about 20–25 min to settle fully**. The finisher builds its areas once no
  builder runs and 120 s have passed since the stream ended [acc: control cold,
  `DEPLOY-PLAN.md` §4].
- For 3–5 min while a walk finalizes, the row can show draw 0's pieces before the vote
  settles them (V11 LOW-3).
- Two walks finalizing at once would share the GPU. That has not been measured, so do not
  overlap them.

**Restart for the second arm.** Wait until Group 1's last walk has fully settled: the room
saved, and **no area saying *Improving***. Then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$D\stop-test-tower.ps1" -WaitMinutes 30
powershell -NoProfile -ExecutionPolicy Bypass -File "$D\start-test-tower.ps1" -Sha <SHA> -Relocalizer silent
```

- **The stop script** refuses while a walk is recording or finalizing, or while the
  finisher runs. `-WaitMinutes 30` polls until the Tower is idle.
  - DEPLOY is changing it to count owed area work as busy too (V12, RV12-A LOW-1). Until
    that lands, check the phone for *Improving* yourself.
- **If `start-test-tower.ps1` refuses** because the finisher's dry run finds a test walk's
  owed areas, re-run it with `-AllowOwed`. The new Tower then builds those areas.
- **Never use `-Force` while a walk is finalizing.**

**Group 2, prompt OFF** (the relocalizer only logs): **W-B ×3**, route A with the whip pans.

**Why grouped:** the manager chose one restart instead of five (031 item 5). Order effects,
such as tiredness and light, are a known limitation.

### 3.4 Pass criteria (prompt ON against prompt OFF; protocol §2)

1. A doorway relocalization is accepted within 10 s in **at least 2 of 3** W-A walks.
   `tracking.recovery` must reach `recovered`, by the triangle rule (2 links of 50
   inliers or more) or by one link of 100 inliers or more.
2. **0** accepted links are more than 8° off the final solve.
3. **0** islands fail placement plausibility at the doorways (tilt ≤ 10°, scale ×1.25,
   eye height 0.3 m).
4. The bathroom is in the room and clean in W-A **at least as often** as in W-B.
5. **At most 2 prompts per minute** in every walk, counted from the journal.
   - On replay this was already 0.81–1.29 prompts a minute [earlier:
     `RUN\experiments\P3-PR\out\PRODUCT.md`].
   - The same replay found 2 of 53 checkable links wrong by more than 8°, both on the
     control, against a bar of 0.
6. **W-G:** no island fails plausibility; every unplaced piece is shown as an area or
   counted, never drawn inside the room; and the room is level.

### 3.5 Device-only measurements (the Mac's C1 review, E6; on every prompt-ON walk)

| # | measure | how |
|---|---|---|
| M-a | the prompt is **audible on the glasses with the phone locked** | Tristan says "heard" aloud at each prompt. The phone's DEBUG log records `didStart`/`didFinish` and the output route, which must be A2DP or LE |
| M-b | the **camera's frame rate and resolution while speech plays** | the Tower's received-frame rate and resolution in the ±5 s around each prompt, against the walk's median, from the capture |
| M-c | the **latency from issue to audible** | the Tower's `prompt.issued_at`, then the envelope's `tower_sent_at`, then the phone's receipt and `didStart` (DEBUG log). Report the median and the maximum |

### 3.6 What to report back, and where the logs land

**Per walk, Tristan notes** the label (W-A1 … W-B3), the clock time at start and end, how
many prompts he heard and whether each was audible (M-a), any stutter while one played, and
anything odd.

**After each walk settles,** he takes a screenshot of the world page (the room caption,
the areas row and any notice) and opens each area once.

**Where the logs land:**

| what | where |
|---|---|
| Tower logs | `RUN\physical-test\logs\tower-8000-<stamp>.out.log` and `.err.log` |
| Each start | `RUN\physical-test\tower-starts.jsonl`: SHA, arm, PIDs |
| Walks | the live store, `C:\Users\tvllo\Projects\Glasses\tower\data\world_builder\worlds\<id>\`: `sessions\<s>\events.jsonl` (the recovery events), `solve\<s>\solution.json`, `solve\<s>\components.json`, `areas\` |
| Raw captures | `C:\Users\tvllo\Projects\Glasses\tower\data\captures\<id>\`. They stay on this machine |
| Phone | the DEBUG log is kept by the Mac side |

The lead scores every walk with the run harness
(`world_coherence_eval.py`, lane `world-builder/coherence-v1`), with renders. Nothing is
deleted without Tristan's approval.

## 4. Rollback

```powershell
$D = 'C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\lead\deploy'
```

### 4.1 Stop the test Tower, and return to today's Tower

1. Run `powershell -NoProfile -ExecutionPolicy Bypass -File "$D\stop-test-tower.ps1" -WaitMinutes 30`.
   It stops only the test Tower, and only when it is idle.
2. Start today's Tower as usual:
   - at the desk: `cd C:\Users\tvllo\Projects\Glasses\tower; .\scripts\start_tower.ps1`;
   - over SSH: `powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\tvllo\tower_start.ps1`.
3. **There is nothing to restore.** The canonical code (`main`) and its `.env` were never
   changed.
   - The test walks stay; they are Tristan's data.
   - Today's code serves re-finished worlds as rooms without areas. This was checked on
     v8020's three, this walk among them [earlier:
     `RUN\lead\deploy\probe-canonical-83534e2-on-v8020.json`].
4. **The lead resumes the acceptance run.** First read `RUN\acc2\STOP`, and the last `STOP`
   or `GATE-STOP` line of `RUN\acc2\logs\chain.log`.
   - If the chain's own gate stopped a world during the pause, do **not** force-continue:
     that needs a manager ruling.
   - Otherwise:
     `powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\lead\launch-acc2-detached.ps1 -ForceContinue`.

### 4.2 Restore the original `6e6d3fc3`: only with Tristan's explicit go

Stop every Tower on the live store first. Both routes **move** data; nothing is deleted.

```powershell
$W = 'C:\Users\tvllo\Projects\Glasses\tower\data\world_builder\worlds\6e6d3fc30e7b45f3a7521e618386b649'
$A = "$W\refinish\<STAMP>"
```

**Route 1: from the set-aside.** Follow the `restore` field in `$A\refinish.json`; read it
at `<SHA>`.

1. Make the destination folders:
   `New-Item -ItemType Directory "$A\rebuild\solve", "$A\rebuild\surface", "$A\rebuild\appearance", "$A\rebuild\dense", "$A\rebuild\sessions\<s>"`.
2. Move each of the rebuild's trees to its own destination:

   | from | to |
   |---|---|
   | `$W\solve\<s>` | `$A\rebuild\solve\<s>` |
   | `$W\surface\<s>` | `$A\rebuild\surface\<s>` |
   | `$W\appearance\<s>` | `$A\rebuild\appearance\<s>` |
   | `$W\dense\<s>` | `$A\rebuild\dense\<s>` |
   | `$W\areas` | `$A\rebuild\areas` |
   | `$W\derived` | `$A\rebuild\derived` |
   | `$W\sessions\<s>\session.json` | `$A\rebuild\sessions\<s>\session.json` |
3. Move `$A\solve\<s>` back to `$W\solve\<s>`, and every other `moved` entry of the ledger
   from its `to` to its `from`.
4. Move the snapshots back: `surface`, `appearance`, `dense` (which has no `predictions\`)
   and `derived`, then `$A\session.json` back to `sessions\<s>\session.json`.

`world.json` then differs only in `updated_at` (`DEPLOY-PLAN.md` §5).

**Route 2: exact, from the frozen evidence.** The live world was byte-identical to it
before the test: 861 files (`RUN\lead\deploy\target-live-vs-frozen.json`).

```powershell
$ErrorActionPreference = 'Stop'
$D = 'C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\lead\deploy'
$W = 'C:\Users\tvllo\Projects\Glasses\tower\data\world_builder\worlds\6e6d3fc30e7b45f3a7521e618386b649'
$F = 'C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\baseline\frozen\worlds\6e6d3fc30e7b45f3a7521e618386b649'
Move-Item $W 'C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\physical-test\aside-6e6d3fc3-<STAMP>'
if (Test-Path $W) { throw "$W still exists: the move did not complete; stop here" }
Copy-Item -Recurse $F $W
attrib -R "$W\*" /S /D      # the copy only; never the frozen evidence
C:\Users\tvllo\Projects\Glasses\tower\.venv\Scripts\python.exe "$D\compare_tree.py" $W $F   # expect: all identical
```

## 5. Known limitations and residual risks

1. **The doorway capture ceiling.** Between the bathroom and the bedroom, the images do not
   hold the yaw and horizontal position. No matcher tried (SIFT, EfficientLoFTR, ALIKED +
   LightGlue, VGGT) finds a corroborated bridge through the doorway, and no generic prior
   supplies one [earlier: `RUN\experiments\P2-LM\doorway\PROGRESS.md`; 1813 §0.4]. Only
   capture-time evidence can link it, and the look-back prompt is what this test measures.
2. **The thin-closet flip.**
   - The closet region (77–78 keyframes) hangs on 6 image pairs, all to **one** room
     keyframe. Their rotation error is 0.55° (median), so the placement is consistent, but
     thin [acc: `RUN\lead\acc2-target-77kf-groupcheck.md`].
   - It follows the **verification draw** of the matching, not the mapper seed. So
     consensus over mapper seeds cannot stabilise it [acc:
     `RUN\mailbox\to-manager\20260924-1047-acc2-target-early-stop.md`].
   - It flips only between a correct attachment and an honest area: GT shows 0 misplaced in
     all 5 runs.
   - Manager 035 accepted it as **"EXCEPTION (safe direction)"**. The follow-ups are in §6.
   - Once a walk is finished, re-opening shows the saved result, and the same seed maps
     the same frozen database.
     - The (d) check re-finished this target warm, on identical inputs.
       `components.json` and `solution.npz` were identical apart from `solved_at` and
       `solve_identity` [acc: `RUN\dchk\logs`].
     - Its re-run at `a5001ab`: <D-AT-SHA>.
3. **Marginal pieces inside the anchor block** (V9 M-2). The vote never withholds the
   room's anchor group, so a marginal piece absorbed into it is not re-verified. This is a
   named residual risk (manager 025, decision 2). `gate.consensus.held_against_majority`
   counts such keyframes: 0 in all 10 control and target runs [acc].
4. **τ against image noise.** The gate honours a link within τ = 16.8°, the control's p90 of
   link-against-solve disagreement. That is about twice the control's image-only
   pair-rotation noise (p95 7.83°). GT still counted 40 misplaced keyframes attached across
   its 24 arms, not yet broken down by group size [earlier:
   `RUN\mailbox\to-manager\20260924-0118-p2-dropped.md`].
5. **The coverage cost.** What is placed in the room falls where pieces become areas.
   - **Target:** 59.0 % placed, or 38.6 % when the closet flips, against 91.4 % in-main in
     the frozen, wrongly glued world. 90.6 % is shown somewhere [acc for the product;
     earlier for the frozen figure, `P3-VAL\TABLE.md`].
   - **Control:** 96.2–97.7 % [acc].
   - **Other worlds** [earlier, e2e7582]: 2f447162 94 → 52 %, and 52ed8e0a 85.5 → 37 %
     (`P3-VAL\TABLE.md`; V8 M6). The 85.5 is agent R's recipe run, not the frozen world.
6. **Face-redactor false positives.** The redactor blacks out wallpaper, screens and
   furniture. On af47007c, 66 of 218 frames were ≥ 2 % filled (median 22 %), and filled
   frames were lost at 42 % against 14 % [earlier:
   `RUN\mailbox\to-manager\20260924-0105-017-abc-h1-decision.md` B].
   It damages any solve without raw frames, and the redacted appearance everywhere.
7. **Release builds cannot capture.** Only a DEBUG build records walks and speaks the
   prompt (contract §6.5; C1 M14), and the Release build hardcodes its Tower address
   (`DEPLOY-PLAN.md` §6).
8. **The cost per walk.** A re-finish holds the GPU for a median of 20 min cold and 15.6 min
   warm on the control (a 398-keyframe walk; its room is 383), and 14.5 min cold on the
   target [acc]. Consensus 3 adds
   about 8.5 min at 678 keyframes [earlier: `RUN\experiments\P3-H2\PROGRESS.md`]. A live
   walk needs about 20–25 min after Stop to settle fully (`DEPLOY-PLAN.md` §4).
9. **Two cold computations are one draw.** Whether two fresh mask computations, or two fresh
   depth predictions, agree bit for bit is unmeasured. The caches make that matter only
   for a walk's first finish.

## 6. Next-run backlog, ranked

1. **Thin but correct attachments** (the closet; manager 035 §3). Options (ii) and (iii) are
   GT-scored on all 24 cases before either is adopted:
   - **(i) capture-time evidence through the look-back prompt.** It is the right lever,
     and this test measures it.
   - **(ii) consensus over the verification seed** as well as the mapper seed. It is
     generic, but with p ≈ 5/9 it does not stabilise this case.
   - **(iii) ≥ 2 distinct room cameras** before a group attaches. It trades coverage for
     determinism, so decide it with numbers.
2. **Anchor-absorbed marginal pieces (M-2):** per-group image-only verification inside the
   anchor block (manager 025).
3. **Redactor precision:** measure it on the frozen worlds and fix the false positives
   (manager 021).
4. **Builder keyframe-id collision on reconnect.** Ids come from `source_seq`, so a
   reconnect that restarts numbering duplicates 20 ids on adc75972 and overwrites their
   stored images (`V8-REVIEW.md`, backlog). A re-finish gives those keyframes their
   stored redacted copy.
5. **τ against noise:** break GT's 40 attached misplaced keyframes down by group size. Bring
   any τ change to the manager with numbers (P2-drop note).
6. **Coverage on hand-covered frames:** 13 af47007c keyframes stay under the
   30-observation floor even on raw frames. Two floor rules were tried and rejected (017
   B).
7. **V11's backlog LOWs** (`RUN\baseline\review\V11\rv11-report.txt`): LOW-2 (a hand-run
   `world_finalize.py` interrupted after the early publish), LOW-5 (a latent drive-root
   read), LOW-6 to LOW-10 (re-finish parking and old ledgers), LOW-11 to LOW-14 (a NaN anchor, re-noticed old records,
   per-reason counts, a full disk while writing a mask), LOW-18 and LOW-19 (photographic
   text), and its open questions.
8. **The area levelling floor is below chance:** isotropic normals gave `levelled: true` in
   20 of 20 cases (`RUN\baseline\review\V9\rv9-report.txt`, LOW).
9. **6839fb8f's g5 group** (15 kf, ×1.27) is still OPEN; about half its excess is regional
   (017 A).
10. **Run infrastructure:** a gpulock child died at spawn with 0xC0000142, twice, at a lock
    handoff (`RUN\status.md`; the record is in `RUN\acc2\aside\`).
11. **A learned matcher** for revisit links. It is deferred, since the doorway gap is
    capture-limited (1813 §7.2).
