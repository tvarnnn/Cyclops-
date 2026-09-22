# Fragment registration — handoff, 2026-08-29

**Branch:** `world-builder/fragment-registration-v1`, pushed to `origin`.
**Base:** `768cecf` on `main`. **Commit:** `fc6eeaf`.
**Research record:**
`docs/superpowers/research/2026-08-29-the-drawer-walk-and-the-cameras-that-were-not-there.md`

---

## 1. The one-paragraph version

Two real walks came back to the phone as disconnected fragments — 7 and 22 of
them. Registration was never invoked from any production path, so the fragments
were never even considered for joining; and once invoked, the Sim3 estimator
was placing target cameras that share no view with the source, which collapses
the fitted constellation and reports a scale ~2.5x wrong. Both are fixed. The
drawer walk goes from **0 placed** to **6 of 23 segments and 59.9% of its
points in one coherent space**. Nothing in the saved corpus regressed: 3
sessions gained a pair, **0 lost one**.

---

## 2. What changed, by file

| file | change |
|---|---|
| `scripts/world_build_session.py` | `--register`: run registration once after the final build, persist placements, report what was placed. `register_session()` swallows every failure, persisting included — a walk that reconstructed is worth keeping even if it cannot be placed. |
| `tower/main.py` | The follower argv carries `--register` when the setting is on. Still an argv, not an import: the web process must keep knowing the builder as a command line. |
| `tower/config.py` | `world_register` / `TOWER_WORLD_REGISTER`, default true. |
| `scripts/world_registration.py` | `_camera_scale` + `_consensus_observations` — the estimator fix. `_placed_span_over_depth` — the gate now reads the baseline the fit had. `NO_VISUAL_LINK`, per-pair evidence counts, `admitted_components` — instrumentation. Two docstring corrections. |
| `scripts/world_replay.py` | New. Deterministic replay of a recorded walk from raw frames. |
| `tests/…camera_consensus.py` | New, 15 tests, host-independent: the filter's mechanism, its call site, and the tolerance's bounds. |
| `tests/…registration_wiring.py` | New, 9 tests: the wiring, its blast radius, and the digest binding. |
| `tests/test_capture_worker_wiring.py` | Two tests: the follower is told to register, and can be told not to. |
| `tests/test_world_registration.py` | One assertion that was RED on a clean tree becomes an honest skip. |

**No threshold in `Thresholds` moved. `admit()` is untouched.**

---

## 3. Numbers

### Before / after, same real physical input

| world | before | after |
|---|---|---|
| drawer walk `af47007c…` | 0 placed (never invoked) | **6 of 23 segments, 7,821 of 13,050 points (59.9%)** |
| …with registration invoked but unfixed | 5 segments, 4,704 (36.0%) | 6 segments, 7,821 (59.9%) |
| plain walk `991e5a15…` | 0 placed | 0 placed — no pair survives, honestly |
| canonical `3dd986b1…` | 3 segments, 3,739 (31.1%) | 3 segments, 3,739 — **unchanged** |
| **whole saved corpus, 16 sessions** | 12 segments / 15,571 points | **17 segments / 20,453 points** |
| | | **3 sessions gained a pair, 0 lost one, 0 pairs lost** |

Served through the real `usable_placements()`: 6 registered rows sharing
reference segment 29, 30 refused. iOS would draw one coherent six-segment space
instead of 22 islands.

### Cost

Registration is **+0.4%** slower with the filter across the corpus (139.4 s →
140.0 s over 16 sessions). It runs **once, at the end, in the follower
subprocess**. Nothing was added to the frame path. On the drawer walk it is
~20–40 s depending on host load. Deterministic: byte-identical reports over
three fresh processes.

---

## 4. The instrument you should use next

```powershell
.venv\Scripts\python.exe scripts\world_replay.py --case worldB `
    --root C:\Users\<you>\Projects\Glasses-scratch\wb-replay --register
```

It rebuilds the drawer walk from its raw frames and reproduces the recorded
session **figure for figure** — 1074 frames, 218 keyframes, identical rejection
histogram, 36 segments, 108 solved poses, 13,050 points — and says so in
`replay.reproduces_recorded_session`. `--case worldA` is the plain walk.

That equality is the whole point: a registration or tracking change can now be
scored against real physical input without another walk. Every number in §3 was
produced this way.

---

## 5. What is still wrong, in priority order

1. **Fragmentation is the largest remaining lever, and it is upstream.** The
   drawer walk breaks into 36 segments in 134 s; **23 of those 36 boundaries
   are `solve_chain_broken`, not `tracking_lost`**. The median segment is 3
   keyframes. On the plain walk, 15 of 23 segments have zero geometry and hold
   110 of its 229 keyframes — 48% of the walk reconstructs nothing, and 89 of
   its 106 pose refusals are cascaded from just 17 root ones.

   Measured against the persisted record: losses fire after the tracker's
   reference has been frozen 2.7–3.8x longer than normal (median 6.7–7.5
   observed frames stale), and the last keyframe before a loss is
   statistically indistinguishable from an ordinary one. That points at
   **advancing the reference frame on rejected frames** — not at holding
   frames. `loss_grace_frames = 3` was already measured and rejected (segments
   114 → 96 but poses_solved 265 → 178, points 42,100 → 27,262).

   This is a live-path change with a recorded history of going wrong twice. It
   now has a replay harness to be measured against. **Do that before touching
   it.**

2. **The new edge has no cycle to check it.** (14,29) joins a cluster with zero
   closures before and after. Segment 29 also becomes the reference — the frame
   everything else is drawn in. The corroboration is indirect: three estimators
   put the scale at 1.045, 0.993 and 1.023. A walk that produces a genuine loop
   would give `cycle_refusal_for` something to act on for the first time.

3. **The plain walk gains nothing** and no registration change reaches it. Its
   one strong visual revisit (segments 23↔24, 164 verified inliers) is unusable
   because segment 23 reconstructed nothing.

4. **Two walks, one room, one afternoon.** `MAX_CAMERA_SCALE_DEVIATION` is
   chosen from three worlds and a corpus-wide sweep, and the interval is NOT
   flat — see §7. The choice is defended by a mechanism (a tighter value
   starves the baseline until agreement is meaningless), not by a plateau.

5. **`test_an_unconfigured_tower_still_serves_its_own_memory` fails on this
   host**, and did before this branch — verified by stashing the change and
   reproducing it. It reads the host's real `data/object_memory`
   (116 observations) where it expects an empty store. Unrelated, pre-existing,
   environmental. Not fixed here because it is not this lane's file.

---

## 6. What was measured and deliberately NOT shipped

- **A 3-D/3-D (Umeyama) second opinion.** Built and scored. On known-answer
  splits it returns **0.3027 for a truth of 1.0**, and RANSAC beats the truth
  on inlier count doing it (28 inliers to 2); four of eight known-answer cases
  came back wrong. The cause is structural: a similarity over landmarks is
  driven by depth, and `landmark_gate` admits ~100% depth uncertainty. On the
  pairs the gate already admits its scale is off by up to 1.23x and its
  rotation by up to 37°. It reaches the same +1 segment as the shipped fix by a
  route with a 50% catastrophic-failure rate on known answers. As a *veto* it
  changes no verdict on either walk, so it would be an inert guard.
- **Raising `MAX_KEYFRAMES_PER_SEGMENT_FOR_MATCHING` from 8.** Every pair's
  recovered scale is flat in the sample size (k = 4, 6, 8, 12, 16, 23). Buys
  nothing, costs 3x the time.
- **Loosening `max_reciprocity_error`.** The honest population already sits at
  4.7–7.7% against a 10% bar, and measured intra-segment drift alone reaches
  7.1%. Nothing loosens it into admitting (14,29) at 2.40x without also
  admitting fits proven 3x wrong on ground truth.

---

## 7. Independent review

An adversarial reviewer who did not write the change reproduced every headline
number exactly — the drawer walk's 5→6 segments and 4,704→7,821 points, the
plain walk's 0 both ways, the canonical world's unchanged 3/3,739, the
ground-truth split returning 0.3046 unfiltered — and confirmed the mechanism:
for (14,29) the forward cameras split cleanly into frames 0–11 at scale
0.95–1.10 (kept) and frames 13–22 at 0.295–0.417 (dropped).

It also found real defects. All were fixed before this was pushed:

| finding | what was done |
|---|---|
| **SEV-1** `register_session`'s try/except covered only `register()`; `write_placements` and `placements_from_report` were outside it. Forcing a write failure gave **exit code 1 and zero bytes of report** — a walk that reconstructed fine, lost. `SegmentPlacement.__post_init__` raises on a NaN scale, which is precisely what this guard exists for. | Persisting moved inside the guard. |
| **SEV-2** Deleting the `_consensus_observations` call from `fit_direction` passed the whole suite — the entire behavioural change reverted silently. | `TestFitDirectionActuallyUsesIt` pins the call site; verified to fail under the mutation. |
| **SEV-2** The docstring's "it can only remove cameras, so it can only refuse more" is true literally and false as a safety argument — the filter materially improves four of `admit()`'s six clauses. | Rewritten to say what actually holds the line (reciprocity, which the filter cannot forge) with the measurement that shows it. |
| **SEV-2** "[0.30, 0.75] is flat on every world" is false: world `2f076449` admits (0,15) at 0.30 and loses it at 0.50, and the drawer walk's upper cliff is 0.90, not 1.00. | Re-measured and rewritten — see below; the finding **strengthened** the choice of 0.50. |
| **SEV-2** The digest binding, which the whole design rests on, was asserted by no test: forcing `input_digest=None` left the file green while `usable_placements` dropped to zero. | `TestTheTransformIsBoundToTheBuildItWasSolvedAgainst`. |
| **SEV-3** Three wiring tests were vacuous — `--synthetic` yields one segment, so `candidate_pairs == 0` and the assertions never ran. | Moved onto a `linked_world` fixture that requires ≥2 segments with geometry, skipping where the corpus is absent. |
| **SEV-3** The report never said cameras had been dropped. | `cameras_considered` on `DirectedFit` and in the verdict clauses. |
| **SEV-3** The skip comment named `TestGateClauses` as the surviving coverage; it has no scale-disagreement case. | Corrected to the three tests in `TestFitQualityCannotAdmit` that mutation testing shows actually kill the rule. |
| **SEV-3** `MIN_CORRESPONDENCES_FOR_CAMERA_SCALE` drops a camera from the *fit*, not just the *vote*; the vote reads pre-RANSAC correspondences. | Both documented precisely at the constants, including why voting on the pre-RANSAC set is deliberate. |
| **SEV-3** The tolerance test imported the constant it tested, so it auto-adapted. | `TestTheToleranceIsNotArbitrary` bounds it at [0.30, 0.90). |

**The tolerance finding is worth its own paragraph, because chasing it improved
the answer.** World `2f076449` pair (0,15) is admitted at tolerance 0.30 and
refused at 0.50 — which reads as lost recall until the two fits are compared:

| tol | cameras | placed span/depth | reciprocity | verdict |
|---|---|---|---|---|
| 0.30 | 3 | **0.0915** | 0.9567 | admitted, "agree to 4.3%" |
| 0.50 | 4 | **0.2036** | 0.8282 | refused, "disagree 1.21x" |

The tighter tolerance drops a camera and collapses the fit's baseline to a hair
over `MIN_SPAN_OVER_DEPTH`. The directions then agree because at that baseline
there is almost no scale left to disagree about. **Tightening this constant
manufactures agreement by starving the quantity scale is measured from.** That
is visible only because `target_span_over_depth` now reports the placed
cameras' span rather than the segment's.

Attacks the reviewer ran that did **not** break it: spurious agreement from a
self-consistent fabricated subset (0 of 16 reverse-camera subsets reach
reciprocity within 10%); cross-world negatives (325 pairs, 6 worlds, 0 admitted
with the filter on or off, and no pair anywhere lost admission); evidence
honesty (every `DirectedFit` field describes the filtered fit, and the
relabelled-copy attack is still refused); determinism (identical over 3 fresh
processes, and over 6 concurrent ones); performance (no measurable cost — the
filter pays for itself by shrinking `_refine`).

### 7a. What the false-merge harness found

Built over 12 saved worlds / 61 qualifying segments / 1,866 pairs. Its findings
about **this change**: nothing moved. Its findings about **the gate**, which
matter more:

- **The gate refuses 18 of 18 geometrically impossible partners** built from
  real imagery with perfect descriptor matches. Four of those are caught by
  `max_rotation_disagreement_deg` **and nothing else**, at reciprocity
  0.96–1.02 and 0.57–1.12 px. Its own comment calls that clause inert. It is
  inert on real pairs and it is the only thing between this gate and a
  sub-pixel geometrically impossible merge. **Do not delete it.**
- **No clause is dead weight.** Ablating each in turn: rotation +4 uniquely
  refused negatives, reciprocity +3, reprojection +3, span +3, cameras +2,
  ambiguity +1.
- **`max_reciprocity_error = 0.10` is ~2x the measured self-pair noise floor**,
  not comfortably clear of it, and the four cross-world admissions sit inside
  the honest band at 0.0063–0.0859. No reciprocity threshold separates cleanly.
- **The harness cannot measure a false-merge rate**, and says so: "different
  world ⇒ different place" is false on this corpus — two of its four
  "false merges" are between worlds captured **67 seconds apart in the same
  home**. The only unambiguous negatives are the 18 synthetic ones.

---

## 8. A note on the checkout

Partway through this work another session created
`feature/cv-lab-live-visualization-v1` in the canonical checkout at
`C:\Users\tvllo\Projects\Glasses` and began editing `tower/tower/cv_lab/` and
`tower/tower/experiments/` there. The branch switch took the checkout off this
lane's branch mid-flight.

Nothing was lost — both branches were at `768cecf` and this lane's work was all
uncommitted working-tree state, which survives a switch — but the tree then
held two lanes' uncommitted changes at once, and whoever committed with `-a`
would have captured the other's files.

Resolved by moving this lane into its own worktree rather than taking the
checkout back:

```
C:\Users\tvllo\Projects\Glasses-worktrees\wb-registration
```

with `data` and `.venv` junctioned in from the canonical checkout so the real
corpus and the venv are reachable. **This lane's files were copied, not moved**,
so identical copies remain in the canonical checkout's working tree on the
cv-lab branch. They are already committed here, so they can be discarded there
— but that is the cv-lab lane's call to make, not this one's, and nothing here
touched their files.

Temporary resources created by this run, per rule 9:

| path | what |
|---|---|
| `Glasses-worktrees\wb-registration` | this lane's worktree, with `data` and `.venv` junctions |
| `Glasses-scratch\wb-replay\` | staged replay frames (hard links) and replayed world roots |
| `Glasses-scratch\pt`, `pt2`…`pt7` | pytest basetemps — the default temp path plus long test names exceeds MAX_PATH |
| `Glasses-scratch\det_*.json`, `replayB_final.json` | determinism and end-to-end checks |

---

## 9. The next physical test

**One walk. Deliberately close a loop, and make the return leg long.**

The drawer walk proved the wearer can generate parallax and that a revisit can
be recognised. What it could not produce is a **cycle** — the only check this
module has that judges a placement by something other than the pair's own
evidence. Every admitted edge on that walk was judged alone.

Walk a closed circuit — out along one path, back along a different one, ending
where you started — so that three or more segments each overlap two others.
Keep moving laterally, keep a textured surface in view at a roughly constant
distance, and re-enter the start area at least 60 s after leaving it.

**What it settles:** whether the placements this branch now produces are
*consistent*, not merely self-consistent. `cycles_checked` becomes non-zero for
the first time on a real walk, and `cycle_residuals` reports how far round-trip
composition drifts. A cluster whose loops close tightly is much better evidence
than one with no loops at all — and if a loop does **not** close, the whole
cluster is refused rather than drawn with a fold in it, which is exactly the
protection we want exercised before anyone trusts a drawn room.

**Falsified if:** loops close but the drawn room still does not look like the
room. That would move the problem from registration to the reconstruction
inside each segment, and item 1 of §5 becomes the whole job.
