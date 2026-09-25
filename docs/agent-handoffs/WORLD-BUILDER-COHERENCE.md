# World Builder coherence — handoff and physical-test guide

**Run `wb-coherence-run-2026-09-23`, handed over 2026-09-24.** This is the operational
companion to the manager's report to Tristan. It serves as the physical-test guide and as
the next run's starting point.

| | |
|---|---|
| Integration | `world-builder/live-world-visualization-v1` at `<SHA>`, its tip at hand-over (the test runs this). A commit cannot name its own SHA, so the lead records it in `RUN\physical-test\TEST-SHA.txt`; every `<SHA>` below means that value |
| Tower lane | `world-builder/coherence-product-v1` (worktree `Glasses-worktrees\wb-coherence-product`) |
| iOS lane | `ios/wb-coherence-areas-v1`, at the Mac tip merged into `<SHA>`: `a72e366` (the banner-and-haptic prompt plus LB2, manager 060); walk 1 ran on `0e1c78d`, walk 2 on the `f33bdfe` DEBUG build |
| Contract | `docs/contracts/WORLD-BUILDER-COMPONENTS.md` (v10: the prompt is shown, not spoken) |
| Operator detail | `tower/docs/world-builder/COHERENCE-PRODUCT.md` (switches, re-finish, validation Tower) |
| Run root (`RUN`) | `C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23` |

Nothing is on `main`. Every new behaviour is **off by default**, and off is the Tower as
it was before (`tower/tower/config.py`).

**Where each figure comes from.**

- **[acc]** means acceptance at the tested code. That is `RUN\acc2\TABLE.md`, run on
  `e5f7151`, and carried forward to `<SHA>` under manager 029's conditions.
  - One of those conditions is the same-seed check (d). At `315b6bf` it re-finished the
    control and the target **warm, on identical inputs** (acc2's w0). Each file was
    compared as `RUN\lead\dchk\compare_dchk.py` compares it:
    - `components.json` and `solution.json` key by key, leaving out, wherever it occurs,
      the script's `ALWAYS` set (its line 10): `solved_at`, `solve_identity`, `timing`, `seconds`, `gate_seconds`, `map_s`, `gate_s`, `database`, `swept`, `frozen_at`, `workspace`, `started_at`, `updated_at`, `at`, `path` and `detail_path`.
      `components.json` then matched; `solution.json` differed only in fields listed as by
      design.
    - `solution.npz` array by array, with nothing left out: fully identical.
    - Source: `RUN\dchk\logs\*.compare-final.txt`.
  - P3.9 then changed the draw-0 gate path, so (d) was re-run at `a5001ab` (`RUN\dchk2`),
    compared the same way. Both worlds gave the same result as before:
    - `components.json` matched with the `ALWAYS` set left out, and `solution.npz` was fully
      identical;
    - `solution.json` differed only in the by-design fields: the prediction and mask cache
      counts (with the mask device and GPU peak, which are empty when every mask came from
      the cache), the frozen-matching record and the digest-rule text
      (`RUN\dchk2\logs\*.compare.txt`, and `*.compare-final.txt` with those fields left out;
      verdicts in `RUN\dchk2\logs\dchk.log`, 12:13 on 2026-09-24).
  - P3.10 (`60b0704`) and P3.11 (`8eb2cc3`) changed only the text scrubber. Its output on
    every text under 4000 characters, the 138 real finalization texts among them, is
    unchanged (`RUN\lead\p310\diff_scrub.out`, `diff_scrub_p311.out`: 0 differences in
    62,366).
- **[earlier]** means an earlier phase of the run. The source is named each time.

---

## 1. What changed, in plain words

**The problem.** The target walk (world `6e6d3fc3`, Tristan's 06:01 walk of 2026-09-23)
had its bathroom and a bed block glued into the bedroom at a wrong tilt, scale and
position. The glue was feature matches on the phone in his hand.
- The two islands were held together by 7 verified pairs with 136 inliers. 85 % of those
  inliers, about 116, lay on the held phone [earlier: `RUN\baseline\FORENSICS.md` §0 item 2].
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
| **Relocalizer and look-back prompt** | While live, after a tracking loss, the builder looks for the view it lost. If it cannot find it within 5 s, the phone shows *"Look back the way you came."* on screen with a haptic (at most 2 prompts a minute). It was spoken until walk 1 of the physical test, where speaking over A2DP made the glasses end the camera session (manager 044). A found view becomes a verified revisit link that the final solve may import | `relocalizer.py`, `engine.py`, `tower/tower/results/world_builder.py` (`tracking.recovery`); iOS shows it (the Mac lane's banner-and-haptic build) |
| **Fail-safes and notices** | If masks or metric depth are missing, the gate attaches nothing beyond the room's core, and says why. Each walk's row carries `finalization.notice`: a fixed sentence from a closed set saying what the Tower could not do and who can fix it. The idle finisher re-runs a gate or a consensus that is owed | `coherence_publish.py` (`NOTICE_SENTENCES`), `tower/scripts/world_finish_pending.py` |
| **The re-finish command** | `tower/scripts/world_refinish.py` rebuilds one saved walk the new way. It sets the previous result aside under `refinish\<stamp>\`, deletes nothing, finds raw frames by capture identity, and rolls back on failure | `tower/scripts/world_refinish.py` |
| **G0** | The surface stage refuses depth fits that are not physical. This removed the flying wall sheets (voxel coarsening on the target 2.03× → 1.00×) [earlier: 1813 §1] | `surface_pipeline.py` (integrated at `17e6d3d`) |

**How it was checked:**

- **Reviews V5 to V14**, all fresh and adversarial, ended APPROVE or READY WITH CHANGES
  (`RUN\baseline\review\V8\V8-REVIEW.md` … `V13\V13-REVIEW.md`, and `V14\`).
  - Every must-fix was fixed. V11's MED-B is fixed at `6ae08c1` and `a5001ab`.
  - V12 found the P3.9 code sound. Its sub-reviewer RV12-B led to P3.10 (`60b0704`): the
    client-safe scrubber runs in bounded time.
  - V13 led to P3.11 (`8eb2cc3`) and to guide fixes.
  - V14 left no HIGH or MED open. The manager's convergence rule (038) sends its LOWs and
    NOTEs to §6, first among them V14 LOW-1.
- **The full suite** at the final product-lane code, `8cca192`: 5487 passed, 80 skipped,
  1 xfailed and 1 failed (`RUN\lead\suite-8cca192.log`). The failure is
  `test_capture_continuity::test_finding_a_successor_does_not_read_every_capture_on_the_disk`,
  a timing flake in code the P3.11 delta does not touch: it failed 1 of 8 in isolated re-runs
  of that file on the same code (§6).
  - At `60b0704` the suite gave 5479 passed and 0 failed.
  - The run at `2febf3a` had another known Windows timing flake, the finisher-chore kill
    test.
  - The integration suite at `<SHA>` is recorded beside the SHA in
    `RUN\physical-test\TEST-SHA.txt`.
- **The acceptance run:** `RUN\acc2\TABLE.md`, with the control and the target complete.
  - At 11:44 on 2026-09-24 the chain's early-stop gate stopped it on 2f447162, another walk of
    the same bedroom. The room differs by 1 to 5 keyframes between seeds (at most 2.51 %), with
    GT 0 misplaced in all five runs (`RUN\mailbox\to-manager\20260924-1207-*` and `-1226-*`).
  - Manager 038 ruled on both findings:
    - kf 306 is an **EXCEPTION (safe direction)**, of the same class as the target's closet;
    - kf 682–690 lead to a **rule clarification**: keyframes the solver left below the
      `min_obs` publication floor in some runs are reported separately, not as room
      differences, in every world.
  - Both rulings are in `RUN\acc2\RULINGS.json`, and the table was re-scored. The chain then
    went on with `-ForceContinue` through 6839fb8f and phases R, B and H
    (`RUN\acc2\logs\chain.log`; progress in `RUN\status.md`).

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
| `TOWER_WORLD_RELOCALIZER_HISTORY` | `0` | `0` through walk 3; `20` for the arm after it (manager 064, 067) | 4 to 20 older keyframes added to each episode's 10 references. They are consulted only when the recent ones decide nothing, as spare-cycle work that yields to a newer frame. A link through one carries `historical: true` in the journal. Out-of-range values read as `0` and are logged |
| `TOWER_WORLD_RELOCALIZER_SUMMARY` | `off` | `off` through walk 3; `on` with HISTORY | one `recovery_summary` line per episode, **in the journal only**; it never reaches the phone |
| `TOWER_WORLD_RELOCALIZER_WINDOW` | `loss` | `loss` | `prompt` would start a prompted episode's timeout at the prompt. It was measured worse in the replay (`RUN\lead\reloc2\`) and is not deployed |
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

1. **Pause the acceptance run, for GPU exclusivity.**
   - Create the pause note only if there is no `STOP`. This command refuses when one
     exists, so it can never overwrite a gate stop that waits for a ruling:
     `New-Item -ItemType File 'C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\acc2\STOP' -Value "lead $(Get-Date -Format s): pause for the physical test (GPU exclusivity)"`.
     If it refuses, the `STOP` already there is the chain's own. Read it: a gate stop needs
     a ruling before the chain goes on (§4.1, step 4). The chain is stopped either way.
   - Then wait until `RUN\acc2\logs\chain.log` ends with a `CHAIN-END` line. The chain stops
     after its current run. A relaunch refused because of `STOP` ends with `REFUSED` lines
     instead, and does not run either.
   - It resumes only after the checks in §4.1, step 4.
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
4. **Only if the Tower will be started over SSH,** run the CUDA pre-flight over that same
   SSH session first. A launch from an SSH session is not proven (`DEPLOY-PLAN.md` §8).
   1. Move the old result aside, with a move, never a delete. Otherwise an old PASS
      survives a probe that died:
      `$P = 'C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\physical-test\preflight'; Move-Item "$P\cuda-wmi.json" "$P\cuda-wmi.$(Get-Date -Format yyyyMMdd-HHmmss).json"`.
   2. Run `powershell -NoProfile -ExecutionPolicy Bypass -File "$D\preflight-cuda-wmi.ps1" -WaitMinutes 30 -EnvFile C:\Users\tvllo\Projects\Glasses-worktrees\wb-live-visualization-v1\tower\.env`.
      It needs the `.env` of step 3. The default `-WaitMinutes 0` returns right after the
      launch, without the result.
   3. Check that a new `cuda-wmi.json` exists, says PASS, and that its `env.at` is today's
      run.
5. **Start the test Tower, prompt arm:**
   `powershell -NoProfile -ExecutionPolicy Bypass -File "$D\start-test-tower.ps1" -Sha <SHA> -Relocalizer prompt`.
   - It runs the integration worktree on port 8000, against Tristan's live store, detached,
     so it survives SSH.
   - It refuses when :8000 is taken, HEAD is not `<SHA>` or the tree is dirty, the `.env`
     lacks a switch, `import tower` does not load the worktree, the run's GPU lock is held,
     or the finisher finds owed work on his worlds.
   - **A pre-approved restart loop must check the mailbox for a hold** (manager 070 §4). Such a
     loop waits for an idle window and then stops and starts the Tower. It must re-read
     `RUN\mailbox\to-lead` for a hold before it acts, or expire 30 minutes after launch. It is
     never left armed across a manager message.
     - **Why:** on 2026-09-25 such a loop reached `stop-test-tower` at 00:06:50, during walk
       3's finalization, 10 minutes after a hold (066) it had not read.
     - `stop-test-tower` refused on the live walk worker, and nothing was stopped. The
       refusal is the last line of defence, not the plan.
6. **Confirm that the Tower is on the right SHA and switches:**
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
- **A launch from an SSH session is not proven.** Step 4 is the check for that case.
- **The validation Tower on :8020 can stay up.** It is serve-only, on the CPU
  (`TOWER_CV_DEVICE=cpu`), with no finisher (`TOWER_WORLD_FINISH_PENDING=false`;
  `RUN\v8020\start_tower.ps1`).
- **His 167 saved worlds:** the finisher's dry run finds 0 owed, so nothing is rebuilt on
  start or when idle. Old worlds look exactly as today (`DEPLOY-PLAN.md` §3).

**The phone:**

- the **DEBUG build** from Xcode. A Release build cannot capture or prompt (contract §6.5).
  - Walk 1 ran on the build at the Mac tip merged into `<SHA>`, which spoke the prompt. The
    prompt-ON walks after it run on the **Mac lane's banner-and-haptic DEBUG build**
    (manager 044), which is not yet merged into `<SHA>`.
  - The Tower is the same `<SHA>` for both builds: nothing Tower-side changed.
- start each walk **on the World Builder screen**, and **hold the phone unlocked on it**. At
  each doorway, glance at it for the banner. It no longer speaks: speaking over A2DP while
  the glasses stream made them end the camera session (walk 1; manager 044);
- do **not** open Saved Worlds during a walk: a pinned view never prompts.

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

- **The stop script** refuses while a walk is recording or finalizing, while the finisher
  runs, and while the finisher still has **owed** work, such as the last walk's areas
  (`stop-test-tower.ps1`, lines 17–19 and 95–105; `RUN\lead\deploy\dryrun-stop-test-tower-v12.txt`
  shows `busy=1` for `owed`).
  - `-WaitMinutes 30` polls until the Tower is idle, and so waits through `owed` as well.
  - `-Force` is the way out of a long backoff of owed work. Use it only when the one reason
    it prints is `areas still owed`: it also cuts off a running area build, which then
    counts as one of the session's 3 attempts (`stop-test-tower.ps1`, lines 15–16), and it
    stops a Tower on :8000 that is not the test Tower. **Never use it while a walk is
    recording or finalizing.**
- **If `start-test-tower.ps1` refuses** because the finisher's dry run finds owed work:
  - first read what the refusal lists: world-id prefixes, each with a stage and a code. The
    session ids are in the `logs\finisher-dryrun-<stamp>.json` it names. `-AllowOwed` lets
    **every** owed session in the store through, not only the test walks';
  - if they are all test walks, re-run it with `-AllowOwed`. The new Tower then builds
    that work.

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
| M-a | the **banner is visible and the haptic is felt** (manager 044; the prompt is no longer spoken) | Tristan says "seen" aloud at each prompt. The phone's DEBUG log records the banner shown and the haptic fired |
| M-b | the **camera's frame rate and resolution around each prompt**, which should now show no change | the Tower's received-frame rate and resolution in the ±5 s around each prompt, against the walk's median, from the capture. Walk 1's spoken prompt stalled the stream 0.47 s after issue and ended the session (`RUN\mailbox\to-manager\20260924-2019-walk1-prompt-timing.md`) |
| M-c | the **latency from issue to banner** | the Tower's `prompt.issued_at`, then the envelope's `tower_sent_at`, then the phone's receipt and the banner shown (DEBUG log). Report the median and the maximum |

### 3.6 What to report back, and where the logs land

**Per walk, Tristan notes** the label (W-A1 … W-B3), the clock time at start and end, how
many prompt banners he saw, and whether he felt each haptic (M-a), any stutter around one,
and anything odd.

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
4. **The lead resumes the acceptance run, but only when every stop has a ruling.**
   1. Read `RUN\acc2\STOP`, and every stop line the chain logged after the last
      `FORCE-CONTINUE` line of `RUN\acc2\logs\chain.log`: `GATE-STOP`, `STOP-CHECK`,
      `STOP-INVALID` and `GATE-ERROR`.
   2. A `GATE-STOP` is ruled only when `RUN\acc2\RULINGS.json` has an entry for that world
      whose `findings.gate_rules` are the rules now in `RUN\acc2\gates\<w8>.json`, and whose
      ruling lets the chain go on. Any other stop, whether it came before the pause or
      during it, has no ruling: **do not force-continue.** Tell the manager instead. (The
      11:44:55 gate stop for 2f447162 was ruled by manager 038, and both findings are
      recorded.)
   3. `-ForceContinue` is not selective. It moves whatever `STOP` is there aside, and
      records the current findings of **every** stopping gate as overridden (`chain.sh`,
      lines 76–97). Use it only when `STOP` is the lead's own pause note or a ruled gate
      stop. When that holds:
      `powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\lead\launch-acc2-detached.ps1 -ForceContinue`.

### 4.2 Restore the original `6e6d3fc3`: only with Tristan's explicit go

Stop every Tower on the live store first. Both routes **move** data; nothing is deleted.

**Route 1: from the set-aside.** Follow the `restore` field in `$A\refinish.json`; read it
at `<SHA>`. `<s>` below is the target's only session, `a37f675a52e0479086e31cc62a9aeec2`.
Paste this block whole. It sets `$A` only when `refinish\` holds exactly one set-aside with
the original solve still in it: the re-finish of §3.1 step 2, whose stamp is
`set_aside.stamp` in `RUN\physical-test\refinish-6e6d3fc3-<time>.json`.

```powershell
. {
  $W = 'C:\Users\tvllo\Projects\Glasses\tower\data\world_builder\worlds\6e6d3fc30e7b45f3a7521e618386b649'
  $s = 'a37f675a52e0479086e31cc62a9aeec2'
  $A = $null
  $all = @(Get-ChildItem -LiteralPath "$W\refinish" -Directory -ErrorAction Stop | Sort-Object Name)
  $L = Get-Content -LiteralPath "$($all[-1].FullName)\refinish.json" -Raw -ErrorAction Stop | ConvertFrom-Json
  if ($all.Count -ne 1 -or $L.state -notin @('done', 'stopped') -or -not (Test-Path -LiteralPath "$($all[-1].FullName)\solve\$s")) {
    throw "refinish\ holds $($all.Count) set-aside(s); the newest, $($all[-1].Name), is '$($L.state)'. Stop: match it to the hand-over stamp by hand."
  }
  $A = $all[-1].FullName; "A = $A"
}
```

`. { }`, not `& { }`, keeps `$W`, `$s` and `$A` for steps 1–4. `done` and `stopped` are the
two ledger states after the publish, and both leave the original in `$A\solve\<s>`. Every
path below that starts from `$A` is built with `Join-Path $A …`, so with `$A` unset it
refuses and nothing moves. Steps 2 to 4 each run as one block that stops at the first
failure.

1. Make the destination folders:
   `New-Item -ItemType Directory -Path (Join-Path $A 'rebuild\solve'), (Join-Path $A 'rebuild\surface'), (Join-Path $A 'rebuild\appearance'), (Join-Path $A 'rebuild\dense'), (Join-Path $A "rebuild\sessions\$s")`.
   With `$A` unset, `Join-Path` refuses and nothing is created.
2. Move each of the rebuild's trees to its own destination:

   ```powershell
   & {
     $ErrorActionPreference = 'Stop'
     Move-Item -LiteralPath "$W\solve\$s"      -Destination (Join-Path $A "rebuild\solve\$s")
     Move-Item -LiteralPath "$W\surface\$s"    -Destination (Join-Path $A "rebuild\surface\$s")
     Move-Item -LiteralPath "$W\appearance\$s" -Destination (Join-Path $A "rebuild\appearance\$s")
     Move-Item -LiteralPath "$W\dense\$s"      -Destination (Join-Path $A "rebuild\dense\$s")
     Move-Item -LiteralPath "$W\areas"         -Destination (Join-Path $A 'rebuild\areas')
     Move-Item -LiteralPath "$W\derived"       -Destination (Join-Path $A 'rebuild\derived')
     Move-Item -LiteralPath "$W\sessions\$s\session.json" -Destination (Join-Path $A "rebuild\sessions\$s\session.json")
   }
   ```
3. Move the original solve back:
   `& { $ErrorActionPreference = 'Stop'; Move-Item -LiteralPath (Join-Path $A "solve\$s") -Destination "$W\solve\$s" }`.
   If the ledger's `moved` list names anything else, move each entry back the same way,
   from its `to` to its `from`.
4. Move the snapshots back (`dense` has no `predictions\`):

   ```powershell
   & {
     $ErrorActionPreference = 'Stop'
     Move-Item -LiteralPath (Join-Path $A "surface\$s")    -Destination "$W\surface\$s"
     Move-Item -LiteralPath (Join-Path $A "appearance\$s") -Destination "$W\appearance\$s"
     Move-Item -LiteralPath (Join-Path $A "dense\$s")      -Destination "$W\dense\$s"
     Move-Item -LiteralPath (Join-Path $A 'derived')       -Destination "$W\derived"
     Move-Item -LiteralPath (Join-Path $A 'session.json')  -Destination "$W\sessions\$s\session.json"
   }
   ```

`world.json` then differs only in `updated_at` (`DEPLOY-PLAN.md` §5). `$W\refinish\` stays
behind, holding the set-aside and `rebuild\`: moved, never deleted. Also left:
`$W\finish_attempts.json`, if a finisher wrote one; the counters from before the re-finish,
if any, are in the ledger's `previous.finish_attempts`.

**Route 2: exact, from the frozen evidence.** The live world was byte-identical to it
before the test: 861 files (`RUN\lead\deploy\target-live-vs-frozen.json`).

Paste the whole block at once. Inside `& { … }`, a failed step or the `throw` ends
everything, so nothing runs after a failure:

```powershell
& {
  $ErrorActionPreference = 'Stop'
  $D = 'C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\lead\deploy'
  $W = 'C:\Users\tvllo\Projects\Glasses\tower\data\world_builder\worlds\6e6d3fc30e7b45f3a7521e618386b649'
  $F = 'C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\baseline\frozen\worlds\6e6d3fc30e7b45f3a7521e618386b649'
  $aside = 'C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\physical-test\aside-6e6d3fc3-' + (Get-Date -Format yyyyMMdd-HHmmss)
  Move-Item $W $aside
  "moved aside to $aside"
  if (Test-Path $W) { throw "$W still exists: the move did not complete; stop here" }
  Copy-Item -Recurse $F $W
  attrib -R "$W\*" /S /D      # the copy only; never the frozen evidence
  & C:\Users\tvllo\Projects\Glasses\tower\.venv\Scripts\python.exe "$D\compare_tree.py" $W $F
}
```

`compare_tree.py` prints JSON and no verdict line. The restore is exact when
`files_left`, `files_right` and `identical` are all 861, and `only_left`, `only_right` and
`differ` are all empty.

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
       `solution.npz` was fully identical, array by array, and `components.json` matched
       with the comparison's `ALWAYS` set of timestamps, identities, timings, file names and
       paths left out
       [acc: `RUN\dchk\logs`; `RUN\lead\dchk\compare_dchk.py`].
     - Its re-run at `a5001ab` gave the same result [acc: `RUN\dchk2\logs`].
3. **Marginal pieces inside the anchor block (V9 M-2): OBSERVED. This is acceptance's FAIL.**
   - **The mechanism.** The vote never withholds the room's anchor group, so a stretch
     absorbed into it is never re-verified.
   - **Where it happened.** On 6839fb8f, a walk of the same bedroom, a revisit stretch of
     the bed can be drawn **inside the room, about 15° off**: 14 keyframes in run w10, and
     the whole 54-keyframe bed block in c10.
     - It happened in **2 of 5 runs**, both on the seed-10 matching database, and in 0 of 9
       draws on the seed-0 and seed-20 databases. The verification draw decides it.
     - The independent evidence says it is misplaced: the GT atom is MIS (14.5°, ×0.72), and
       two-hop image chains show a median of 32° against the control's 2.8°.
     - The solver's own database honours homography links that the harness contradicts by
       up to 31° (room-side cameras 3745–3835).
     - `held_against_majority` does not see it: every draw held these keyframes.
   - **Recorded** as **FAIL (attach direction)** by managers 042 and 053
     [acc: `RUN\lead\acc2-6839fb8f-flipcheck.md`; `RUN\acc2\RULINGS.json`].
   - **A second finding on the same world.** 14 closet-edge keyframes detach in one run
     only, and their attached placement cannot be verified. They are labelled
     **UNVERIFIED**, which is not a pass.
   - **The scorer now closes this hole:**
     - it reports "hidden MIS", an MIS atom whose majority lies outside the room;
     - an ambiguous report no longer excuses an attachment that independent evidence
       contradicts (manager 053).
4. **τ against image noise.** The gate honours a link within τ = 16.8°, the control's p90 of
   link-against-solve disagreement. That is about twice the control's image-only
   pair-rotation noise (p95 7.83°). GT still counted 40 misplaced keyframes attached across
   its 24 arms, not yet broken down by group size [earlier:
   `RUN\mailbox\to-manager\20260924-0118-p2-dropped.md`].
5. **The coverage cost.** What is placed in the room falls where pieces become areas.
   - **Target:** 59.0 % placed, or 38.6 % when the closet flips, against 91.4 % in-main in
     the frozen, wrongly glued world. 90.6 % is shown somewhere [acc for the product;
     earlier for the frozen figure, `RUN\experiments\P3-VAL\TABLE.md`].
   - **Control:** 96.2–97.7 % [acc].
   - **Other worlds** [earlier, e2e7582]: 2f447162 94 → 52 %, and 52ed8e0a 85.5 → 37 %
     (`RUN\experiments\P3-VAL\TABLE.md`; V8 M6). The 85.5 is agent R's recipe run, not the
     frozen world.
6. **Face-redactor false positives.** The redactor blacks out wallpaper, screens and
   furniture. On af47007c, 66 of 218 frames were ≥ 2 % filled (median 22 %), and filled
   frames were lost at 42 % against 14 % [earlier:
   `RUN\mailbox\to-manager\20260924-0105-017-abc-h1-decision.md` B].
   It damages any solve without raw frames, and the redacted appearance everywhere.
7. **Release builds cannot capture.** Only a DEBUG build records walks and shows the
   prompt (contract §6.5; C1 M14), and the Release build hardcodes its Tower address
   (`DEPLOY-PLAN.md` §6).
   - **No audio to the glasses while they stream.** Speaking the prompt over A2DP made the
     glasses end the camera session ("Session ended by device") about 3 s into the speech.
     Walk 1 of the physical test was aborted that way.
   - The prompt is now a banner and a haptic on the phone (manager 044; Tristan's decision).
     The Tower-side events (`recovery_prompted`, `tracking.recovery`) are unchanged.
8. **The cost per walk.** A re-finish holds the GPU for a median of 20 min cold and 15.6 min
   warm on the control (a 398-keyframe walk; its room is 383), and 14.5 min cold on the
   target [acc]. Consensus 3 adds
   about 8.5 min at 678 keyframes [earlier: `RUN\experiments\P3-H2\PROGRESS.md`]. A live
   walk needs about 20–25 min after Stop to settle fully (`DEPLOY-PLAN.md` §4).
9. **Two cold computations are one draw.** Whether two fresh mask computations, or two fresh
   depth predictions, agree bit for bit is unmeasured. The caches make that matter only
   for a walk's first finish.
10. **What the physical test's first walks found (2026-09-24)** (`RUN\physical-test\TEST-LOG.md`):
    - **Walk 1 (PARTIAL): speech ends the stream.** A spoken prompt over A2DP made the
      glasses end the camera session: the frame stream stalled 0.47 s after the prompt was
      issued. The prompt is now a banner and a haptic (§5.7). Walk 2's frame rate held at
      about 12 fps through both prompts.
    - **The viewer's lengths were absolute.** Walk 1's solve came out 27× smaller than other
      rooms, and the photographic page pushed the camera out of the room ("Nothing was
      photographed this way"). **Fixed at `57f56b5`:** the lengths are multiples of the
      scene's unit, and the room page walks the room's cameras only.
    - **The look-back did not re-link in walk 2:** verdict (d), no look-back returned to a
      reference view. A view from 2–4 m does not match close-up references, and the window
      runs 20 s from the loss, so about 15 s after the prompt (§6 #17).
    - **Acceptance on 6839fb8f: FAIL on thin evidence** (§5.3).

## 6. Next-run backlog, ranked

0. **Land P3.12, V14 LOW-1, with a V15. It is a privacy regression in over-long text only.**
   - **The fault.** `8eb2cc3` scrubs this machine's user names before it cuts a
     finalization line over 4000 characters. The `[user]` it leaves inside a POSIX, `~` or
     relative path stops the path patterns at its bracket. So the file and folder names
     under that user's home survive, for example `[path][user]/secret_plan.txt`. The user
     name itself does not.
   - **Why it is not urgent.** The longest real finalization text is 522 characters, so
     nothing the Tower writes today reaches this.
   - **The fix is ready.** It moves the cut back before a name instead of replacing the
     name, and puts the `...` on after the scrub. It is
     `RUN\lead\p310\P3.12-bounded-no-name-prepass.patch`, which applies cleanly to
     `8cca192`. It is checked against RV14's probes: 68 of 68 clean, fuzz 0 of 3000
     leaked, RV13's 24 of 24 (`RUN\lead\p310\p1_on_p312.txt`, `p6_on_p312.txt`,
     `d2_on_p312.txt`).
   - **Land it with** V14 LOW-2's tests, which pin each part of the cut
     (`RUN\baseline\review\V14\` has the shapes).
1. **Thin but correct attachments** (the closet; manager 035 §3). Options (ii) and (iii) are
   GT-scored on all 24 cases before either is adopted:
   - **(i) capture-time evidence through the look-back prompt.** It is the right lever,
     and this test measures it.
   - **(ii) consensus over the verification seed** as well as the mapper seed. It is
     generic, but with p ≈ 5/9 it does not stabilise this case.
   - **(iii) ≥ 2 distinct room cameras** before a group attaches. It trades coverage for
     determinism, so decide it with numbers.
2. **THE TOP ITEM (manager 042): per-group image-only verification inside the anchor block.**
   - **The goal:** detach a stretch whose independent image evidence contradicts the solve,
     even when the solver's own links honour it.
   - **The motivating evidence:** 6839fb8f (§5.3). It is a FAIL in 2 of 5 runs, with up to
     54 keyframes drawn about 15° off. The solver database and the harness disagree by up to
     31° on the same camera pairs.
   - **How to build it:** calibrate from the control only, and make a failing group an
     honest separate area.
   - **A Codex (gpt-5.6-sol) design note is in** `RUN\review\codex\design-anchor-verification-*.md`.
     Treat it as leads to verify.
   - **Decide how the gate counts a relocalizer import (manager 064).** One accepted
     triangle imports two pairs through one anchor image whose other ends are consecutive
     keyframes already linked to each other. That is exactly the gate's "two pairs through
     one image closing a triangle" (`coherence_gate.py:285-297`), so **one live decision
     counts as two links**. Decide whether one import counts as one link for
     independence.
3. **Redactor precision:** measure it on the frozen worlds and fix the false positives
   (manager 021).
   - **And recall:** in walk 3 (`4f5d0b15`), keyframes 780–786 show the wearer's reflection
     in the bathroom mirror, with the head not blotched in some frames. These frames stay
     local (manager 081 §5).
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
12. **Record per-keyframe publication in the consensus record** (manager 038). On
    2f447162 the record showed a unanimous room while three desk keyframes had 2 of 3
    votes at the `min_obs` floor. The tally was never saved
    (`RUN\lead\acc2-2f447162-flipcheck.md`).
13. **The scrubber's remaining LOWs and NOTEs:**
    - RV12-B LOW-C: other users' names in forward-slash UNC, `file://`, colon-glued and
      spaced relative paths;
    - RV12-B NOTE-1 to NOTE-4, and V12 NOTE 4: over-scrubbed dates and `e.g.`, a frame
      path holding an apostrophe, re-scrub growth, and non-string values sent raw;
    - V14 NOTE-1: a file name at the cut, glued to its `...`;
    - V13 NOTE-5: `refinish-target.ps1` header nits, already fixed.

    Record: `RUN\baseline\review\V12\` to `V14\`.
14. **Test flakes on Windows:**
    - `test_capture_continuity::test_finding_a_successor_does_not_read_every_capture_on_the_disk`:
      1 of 8 isolated runs, and once in the `8cca192` suite;
    - the finisher-chore survives-its-kill test;
    - `test_world_builder_coherence_publish::test_the_components_reader_refuses_a_record_of_another_solve`:
      2 of 6 isolated runs, **already at `4b4b444`**. `solve_identity` is exact, so the
      likely cause is a reader keyed on the solution's mtime, with two writes inside one
      Windows clock tick. Worth a real look: a stale components record should never survive
      a newer solve;
    - `test_result_channel_truthfulness::test_a_dead_builder_is_reported_as_interrupted_not_as_receiving`:
      only under heavy CPU load.

    All are timing-dependent, in code this run did not change.
15. **Publish every solve at metric scale** (manager 052). An SfM gauge is arbitrary: walk
    1 of the physical test (`ee48aae3`) came out 27× smaller than other rooms. The viewer
    now measures its lengths in the scene's own unit (`57f56b5`).
    - Normalising the published solve with the gate's metric estimate, where it is available
      and flagged, would make every downstream constant mean metres: viewer, voxel size and
      bounds.
    - The viewer's remaining absolute shader thresholds and its standoff on genuinely deep
      scenes (RV-VIEW F2–F4; the Codex failure-mode review) belong here too.
16. **Latent artefact-currency risks** (Codex's map of the viewer):
    - appearance and surface "currency" is reported but not enforced;
    - `input_digest` does not see a different draw over the same keyframe set;
    - an open page does not refresh `CONFIG.cameras` when a new revision arrives;
    - an area id stays stable across parent draws.

    None of these caused tonight's bug.
17. **Relocalizer observability and look-back design** (walk 2; manager 056;
    `RUN\mailbox\to-manager\20260924-2203-walk2-relocalizer-diagnosis.md`).
    - **What the Tower records:** no scan-level detail. Add a per-episode summary event
      (reference ids, attempts, best legs and closure, losses joined).
    - **The window** runs 20 s from the loss, so only about 15 s after the prompt. A timeout
      discards the episode's references.
    - **Walk 2's look-backs** never revisited a close-up reference view from nearby; a view
      from 2–4 m does not re-link.
    - **For the next run:** reference diversity and distance, and the prompt's wording.
      Measure a lower inlier floor's wrong-match rate before considering it.
    - **The candidates** (manager 058), measured offline with an exact replay of the
      relocalizer (`RUN\lead\reloc1\`, `reloc2\`), acceptance rule unchanged:
      - the window runs from the prompt: **worse; not deployed** (WINDOW);
      - references include a time- and viewpoint-spread sample of earlier keyframes:
        **HISTORY**, landed off by default at `08b1e09`;
      - the per-episode summary event: **SUMMARY**, landed off by default.
    - **HISTORY's replay** (9 walks, base against HISTORY=20; `RUN\lead\reloc2\TABLE_out_f1b*.md`):
      - recoveries went from 35 to 43 at normal cost, and 34 to 40 at double cost;
      - **0 extra wrong links** at either cost;
      - **0 of 19 and 0 of 11 historical links wrong.**

      Recent references decide first, and history runs only when they yield nothing, as
      preemptible spare-cycle work (review F1). With nothing set, the journal is
      byte-identical to `9f4766a` (the golden test).
    - **Aliasing on repeated structure must be measured before HISTORY can become a
      default** (review F2). A long-range link to a look-alike place can survive COLMAP's
      re-verification just as it fooled SIFT live. Score every `historical: true` link of
      the HISTORY arm against the final solve, with an image-only check.
    - **Walk 3 (held out; `RUN\lead\reloc3\DIAGNOSIS.md`): 0 of 9 prompted episodes recovered
      across walks 2 and 3.**
      - In 7 of 9 the user did not return to a reference view. The whip-pan was a turn to walk
        somewhere else, and the prompt found him there.
      - In 2 the references were unusable (the turn's own blurred frames; "the last 10" span
        only 0.6–4.1 s before the loss).
      - Reading the banner puts the phone in the camera's view, and its screen matches screen
        to screen.
      - The final solve links the stretch anyway in 7 of 9.
      - Neither a lower floor nor a longer window is supported.
      - **So the prompt is OFF by default: walk 4 runs the relocalizer `silent`**, with
        HISTORY=20 and SUMMARY=on (manager 081 §2).
    - **RELOC3's candidates move to the UX phase's capture-guidance design** (081 §2). Each is
      measured on the pre-walk-3 corpus first:
      - references drawn from sharp, stable keyframes;
      - the phone screen kept out of the relocalizer's matches;
      - withholding a prompt the reference set cannot satisfy;
      - **showing which view to return to**;
      - starting the episode clock at the loss frame's receipt.
    - **HISTORY=20 out of sample on walk 3** (`RUN\lead\reloc2\TABLE_w3.md`):
      - 4 prompted episodes recovered, against base's 1;
      - **1 wrong historical leg** (9.6°, 52 inliers, 68 kf older).
    - **RV-RELOC's remaining nits** (re-check of `08b1e09`, clean):
      - `events.py:78-81`'s summary-key comment omits `history_preempted`;
      - `recovery_anchored` links carry no historical mark;
      - the preemption unit test fakes the mailbox (RV-RELOC's real-thread check,
        `RUN\lead\rvreloc\async_preempt.py`, covers the gap);
      - under heavy load history may get no spare time. That is safe, and the summary's
        `history_preempted` count shows it.
18. **Speed.**
    - Compute masks and depth *during* the walk (per keyframe, as frames arrive), so the
      Stop-time solve does not start from zero.
    - Run the 3-draw vote in parallel where the GPU allows.
    - Walks took 8.8–11.2 min from Stop to settled tonight.
19. **Incremental world building** (Tristan's idea): extend the map live and correct
    earlier poses when later data links back, instead of re-solving at Stop. A staged
    design is owed; Codex's design note is deferred by its quota.
20. **A resolution test.** Measure what the 640×360 live frames cost the look-back matcher
    and the final solve, against higher resolutions.
21. **MockDevice duplicate classes** (Mac lane): the duplicate test-double classes in
    GlassesTests, reported by the Mac.
22. **The pocketed-prompt notification** (Tristan's decision): a locked phone cannot show
    the banner or fire the haptic. Any notification must not route audio to the glasses
    (§5.7).

## 7. Resources this run created or stopped

- **Worktrees:**
  - `Glasses-worktrees\wb-coherence-product` (`world-builder/coherence-product-v1`, the
    Tower lane);
  - `Glasses-worktrees\wb-live-visualization-v1` (`world-builder/live-world-visualization-v1`,
    the integration; the test Tower reads its code and its `tower\.env`);
  - `Glasses-worktrees\wb-coherence-v1` (the experiment lane).
- **Scratch:**
  - `RUN` itself;
  - `Glasses-scratch\wbcpt\` holds short pytest base-temps (`s<tag>`, `tf<n>`) and
    RV-RELOC's synthetic stores (`rvreloc30\`, `rvreloc31\`).
  - Disposable; nothing outside `Glasses-scratch` was created.
- **Processes stopped** (manager 063 §3), after checking that their parents were gone and
  that no other process named their files:
  - the GPU sampler loop `sh.exe` 8764 and its outer `bash` 37008, which exited with it.
    Its `gpu_samples.csv` (2.38 MB) is kept;
  - two `tail.exe` from 2026-09-21 (31608, 31848), and 17 run tails before them. No
    `tail.exe` is left.
- **Storage cleanup** (manager 062–064; authority `RUN\storage-audit\AUTHORITY.md`,
  Tristan's delegation):
  - **1,174 paths, 166.13 GiB deleted**, only through `RUN\bin\safe-delete.ps1` after a
    dry run;
  - every path is in `RUN\storage-audit\DELETED.md`, the guard's ledger is
    `deletion-ledger.jsonl`, and the manifests and transcripts are in `manifests\`;
  - the GT functional checks are byte-identical before and after (`functional\`);
  - `RUN\experiments\REGISTRY.md` lists every deleted or split experiment folder.
- **`Glasses-scratch\wb-dense` (the dense lane's scratch) was split, not removed** (S7):
  - **Deleted:** 449 paths, 16.39 GiB. These were the per-world solver databases,
    `solution.npz` and `sparse\`, plus the `_frames` and `sessions\*\images` directories.
  - **Kept, 1.015 GiB:** the scripts, logs, per-world JSON records and rendered images
    (listed in `RUN\storage-audit\split1\S7.keep.tsv`). `walk1_views`, `walk2_views`,
    `baseline-render` and `viewer-pages` are kept whole.
  - **The `_frames` were hard links** to the live capture frames, so deleting them freed
    no capture data. A census of the 12 source captures (15,601 frames) is identical
    before and after (`RUN\storage-audit\functional\s7_capture_census.*.json`).
  - **Each deleted `images` directory's source frames were verified present in the live
    store first** (`S7.report.json`, `source_verification`).
  - **To rebuild a wb-dense world**, re-stage it from the live captures.
