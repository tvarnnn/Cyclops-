# The drawer walk, and the cameras that were not there

**Date:** 2026-08-29. **Branch:** `world-builder/fragment-registration-v1`.
**Status:** MEASURED, and implemented. Two production changes, both on real
physical captures with a ground-truth control.

---

## 0. What this is about

Two real glasses walks were captured on 2026-08-29 and both arrived on the
phone as a heap:

| | walk A (normal room) | walk B (deliberate overlap round a drawer) |
|---|---|---|
| world / session | `991e5a15…` / `815c88ba…` | `af47007c…` / `7864d3b3…` |
| frames observed | 1008 | 1074 |
| keyframes accepted | 229 | 218 |
| segments | 23 | 36 |
| poses solved / refused | 100 / 106 | 108 / 74 |
| points | 9,145 | 13,050 |
| **iOS showed** | **7 fragments, not connected** | **22 fragments, not connected** |

Walk B is the important one. It was deliberately designed to make registration
easy: the wearer kept a textured drawer in view, moved laterally around it to
generate parallax, and returned toward the starting viewpoint. It still came
back in 22 pieces.

Two separate things were wrong, and only one of them was where everyone
including this repository's own prior research was looking.

---

## 1. The first finding: PT-1 passed, and it moved the constraint

Every earlier conclusion about cross-segment registration in this repository
rests on one number: **`span_over_depth` between 0.02 and 0.06**, i.e. the
wearer's camera baseline was 2–6% of scene depth, so scale was unrecoverable
at any quality of match. 135 of 141 refusals on the canonical world were that.
Two independent implementations converged on it, and the standing conclusion
was: *"the binding constraint is the capture, not the estimator."*

`docs/agent-handoffs/2026-08-27-PHYSICAL-TESTS-QUEUED.md` queued **PT-1** to
test exactly that: a walk with deliberate lateral translation, re-entering the
same area. Walk B is that walk. Its measured span/depth:

```
seg 29  0.734   (23 cameras, 3117 points)   <- the return leg
seg 14  0.720   (12 cameras, 1719 points)   <- the outbound leg
seg 30  0.515       seg 34  0.459
seg 11  0.266       seg 28  0.257       seg 3  0.207
```

**PT-1 passed.** The wearer translated. The two segments that carry the revisit
are the two best-conditioned in the corpus, at roughly a 1.4:1 depth-to-baseline
ratio. The old ceiling is gone on this footage, and with it the reason to
believe the estimator was innocent.

The long tail below 0.09 is still there — 187 of 253 candidate pairs are still
pruned as "the wearer stood still" — but it is entirely 2–3-keyframe fragments
that never got past the unit bootstrap baseline. They hold 1,808 of 13,050
points. **11,242 points (86%) live in segments that clear both the camera bar
and the span bar.** The registration ceiling on this walk is 86%, and we were
getting 0%.

---

## 2. The second finding: nothing ever called registration

`scripts/world_registration.py` — 1,876 lines implementing Sim3 estimation,
the `MutualEvidence` independence rule, reciprocity, cycle consistency,
digest-bound persistence — was complete and tested. `store.write_placements()`
existed. `results/world_builder_geometry.py::usable_placements()` read
placements, checked their digest, and refused stale ones.

**No module under `tower/tower/` imported any of it.** Neither walk has a
`placements.json`. The 22 fragments were not refused; the question was never
asked. This was written down on 2026-08-27
(`2026-08-27-registration-exists-and-is-never-called.md`) and not acted on.

Asking the question, with no other change, places **5 of walk B's 36 segments
and 4,704 of its 13,050 points (36.0%)**. On walk A it places nothing, honestly.

---

## 3. The third finding, and the real defect: fabricated cameras

With registration running, walk B still refused the pair that mattered.
Segments 14 and 29 are the outbound and return legs of the same physical
circuit. Between them: **270 verified keyframe pairs, 20,267 essential-matrix
inliers**, and 108 mutually-triangulated 3D–3D correspondences on the single
best view pair. The gate refused them because the two directions disagreed on
scale by **2.40x**.

That refusal was correct given its evidence and wrong about the room.

### 3.1 The mechanism

`_pnp_observations` places *any* target keyframe for which twelve of the source
segment's landmarks survive PnP RANSAC. On repetitive indoor texture, a
keyframe sharing **no physical view** with the source still clears that bar on
aliased matches. And the fabricated cameras are *mutually consistent* — they
all collapse toward the origin together — so nothing downstream notices.

A Sim3's scale **is** the ratio of the placed constellation's span to the
target's own span. Collapse part of the constellation and you report a smaller
scale. For (14,29), forward:

```
frame inliers |C_pnp| |C_target|      frame inliers |C_pnp| |C_target|
    0     169   5.530     0.000  ...     13      71   5.860     4.733
    3     240  10.931     5.190          15      35   4.759     6.944
    7     154   9.336     3.823          18      19   3.594     8.992
   11     132   7.990     2.641          22      27   3.021    14.040
```

For frames 13–22 the placed centre moves *inward* (5.86 → 3.02) while the
camera's true distance moves *outward* (4.73 → 14.04). Span of placed centres
9.244 against 20.208 in segment 29's own frame — **ratio 0.4574, which is
exactly the scale the fit reported.**

Restricting the forward solve to segment 29's frames 0–11, the ones that
actually overlap segment 14:

```
 all 0..22          scale 0.3944  cams 23  corr 2304  reproj 4.62  amb 3.47
 0..11 (near half)  scale 1.0031  cams 12  corr 1880  reproj 2.45  amb 1.19
 12..22 (far half)  scale 0.3206  cams 11  corr  424  reproj 3.23  amb 2.43
```

Reverse is stable at 0.93–0.97 everywhere. **The disagreement was never a
statement about the room. It was ten cameras that had no business in the fit.**

### 3.2 The ground-truth control

Split ONE segment into two halves. They share a coordinate frame and a unit by
construction, so the true scale is **exactly 1.0**. Segment 29:

| | forward | reverse | reciprocity | rotation | reproj | ambiguity |
|---|---|---|---|---|---|---|
| truth | 1.0 | 1.0 | 1.0 | 0° | — | — |
| measured | **0.3046** | 1.0813 | **0.3294** | 0.62° | 2.48 px | 2.04 |

**Every clause in `admit()` except reciprocity passed a fit that is provably
3.04x wrong.** Seven of that split's eight target cameras were fabricated,
placed with centre errors of 4.8 to 12.3 against a scene depth of 27.6. Only
28.8% of its landmark-carrying matches reproject within 3 px under the target
frame's own true pose.

This is the same signature as (14,29), reproduced where the answer is known.

### 3.3 What the estimator's noise floor actually is

With genuine dense overlap and correct correspondences, reciprocity error is
**0.01%–1.8%** (interleaved splits). With genuine but temporally offset overlap
it is **4.6%** (segment 14's halves) and **6.7–7.1%** (segment 29's thirds,
correspondence-cleaned) — and that residual is real intra-segment scale drift,
not estimator noise. Rotation disagreement on every honest case: **0.04–0.71°**
against a 15° bar.

**`max_reciprocity_error = 0.10` must not be loosened.** The four pairs walk B
already admits sit at errors of 4.7%, 5.4%, 7.0% and 7.7%; measured drift alone
reaches 7.1%. The honest population already consumes 77% of the budget. And
nothing loosens 0.10 into admitting (14,29) at 2.40x without also admitting
fits proven 3x wrong on ground truth. **Reciprocity was the only clause that
caught the ground-truth failure.** It is the asset here, not the obstacle.

---

## 4. The fix: make each camera state the scale itself

For a correspondence that is genuinely the same physical point, the depth of
the source's landmark in the PnP-placed camera and the depth of the target's
landmark in the target's own camera differ by exactly the Sim3 scale. So every
placed camera can state the scale **independently**, from the two
reconstructions already on disk and no ground truth at all.

`_consensus_observations` computes that per-camera scale, keeps the largest
agreeing group weighted by how much evidence each camera brought, and hands
only those cameras to the fit. On the ground-truth split of segment 29 the
per-camera scales read:

```
genuine  camera : 0.9979   (inter-quartile spread 0.0016)
fabricated (x7) : 0.31 - 0.48   (spreads 0.24 - 0.46)
```

**It is a narrowing, not a loosening.** It can only remove cameras, so it can
only make `min_cameras` harder to satisfy. No threshold in `Thresholds` moved.
`admit()` is untouched. On the ground-truth split the forward direction stops
returning 0.30 and returns nothing at all.

### 4.1 Choosing the tolerance

`MAX_CAMERA_SCALE_DEVIATION` is bounded from both sides. Too tight and honest
cameras at the far end of a segment are dropped for ordinary intra-segment
drift until a thin pair falls under `min_cameras`. Too loose and the fabricated
cameras outvote the genuine ones. Swept over three real worlds and eight
ground-truth splits:

| tol | drawer walk B | canonical `3dd986b1` | ground truth |
|---|---|---|---|
| 0.15 | 3 segs / 2,490 | 2 segs / 2,328 | nothing wrong admitted |
| 0.20 | 6 segs / 7,821 | 2 segs / 2,328 | nothing wrong admitted |
| 0.25 | 6 segs / 7,821 | 2 segs / 2,328 | nothing wrong admitted |
| **0.30** | **6 segs / 7,821** | **3 segs / 3,739** | nothing wrong admitted |
| 0.40 | 6 segs / 7,821 | 3 segs / 3,739 | nothing wrong admitted |
| **0.50** | **6 segs / 7,821** | **3 segs / 3,739** | nothing wrong admitted |
| 0.75 | 6 segs / 7,821 | 3 segs / 3,739 | nothing wrong admitted |
| 1.00 | 5 segs / 4,704 | 3 segs / 3,739 | nothing wrong admitted |

Below 0.30 the canonical world loses the pair (5,32) it registers today — its
three forward cameras read 4.23, 4.12 and 5.36, a 28% spread that is drift, not
fabrication. At 1.00 walk B loses (14,29) again. **[0.30, 0.75] is flat on
every world**, so 0.50 is the centre of a measured plateau rather than the edge
of a cliff, and it sits above the largest honest intra-segment drift measured
anywhere here.

Worth stating plainly: **the ground-truth splits do not discriminate between
these values.** Every one of them refuses the wrong answers. They bound the
change's safety; the two real worlds choose within it.

### 4.2 The whole saved corpus, before and after

Three worlds is not enough to know a constant was not fitted to them. Every
saved session with two or more segments carrying geometry was registered twice
in one process — once with `_consensus_observations` made the identity, once
at HEAD — so the only difference is the filter:

| | before | after |
|---|---|---|
| sessions scored | 16 | 16 |
| segments registered | 12 | **17** |
| points registered | 15,571 | **20,453** (of 134,323) |
| sessions that gained a pair | — | **3** |
| sessions that lost a pair | — | **0** |
| pairs gained | — | 3 |
| **pairs lost** | — | **0** |

The gains are `af47007c` (14,29), `3d49a771` (5,10) and `6502da15` (6,7); the
last two registered nothing at all before. **Nothing anywhere in the corpus
regressed.** The change is monotone on every session available.

---

## 5. Before and after, on the same real physical input

Measured by replaying the raw frames, not by re-walking. All three worlds:

| world | before | after |
|---|---|---|
| **walk B (drawer)** `af47007c…` | 0 placed *(never invoked)* | **6 of 23 segments, 7,821 of 13,050 points (59.9%)** |
| walk B, registration invoked but unfixed | 5 segments, 4,704 points (36.0%) | 6 segments, 7,821 points (59.9%) |
| walk A (normal room) `991e5a15…` | 0 placed | 0 placed — no pair survives, honestly |
| canonical `3dd986b1…` | 3 segments, 3,739 points (31.1%) | 3 segments, 3,739 points (31.1%) — **unchanged** |

Walk B's admitted pairs go from `[(3,30), (14,28), (14,30), (15,30)]` to
`[(3,30), (14,28), (14,29), (14,30), (15,30)]`, a single connected component
`[3, 14, 15, 28, 29, 30]`, and the reference segment moves from 14 to 29 — the
largest reconstruction in the walk. **The return leg is now joined to the
outbound leg.**

Three other pairs moved toward refusal, which is the point:

| pair | before | after |
|---|---|---|
| (14,29) | disagree 2.40x | **admitted**, agree to 6.1%, rotation 0.6° |
| (15,29) | disagree 2.10x | disagree 1.12x — still refused |
| (28,29) | disagree 1.29x | only 2 cameras placed — refused earlier and more honestly |
| (29,30) | disagree 2.32x | only 2 cameras placed — refused earlier and more honestly |

---

## 6. What was NOT done, and why

**Segmentation was not touched.** Walk B breaks into 36 segments in 134 s, and
23 of those 36 boundaries are `solve_chain_broken` rather than `tracking_lost`.
Measured against the persisted record, losses fire after the tracker's
reference has been frozen 2.7–3.8x longer than normal (median 6.7–7.5 observed
frames stale), and the last keyframe before a loss is statistically
indistinguishable from an ordinary one. That points at **advancing the
reference frame on rejected frames**, not at holding frames — `loss_grace_frames
= 3` was already measured and rejected (segments 114 → 96, but poses_solved
265 → 178 and points 42,100 → 27,262).

This is the largest remaining lever and it is a live-path change with a
recorded history of going wrong. It now has a replay harness to be measured
against. It is not in this pass.

**`MAX_KEYFRAMES_PER_SEGMENT_FOR_MATCHING = 8` was not raised.** Measured: the
scale of every pair on walk B is flat in the sample size (k = 4, 6, 8, 12, 16,
23 all give the same forward/reverse scales to within noise). Raising it buys
nothing and costs time. The full cross-product is 42.7 s against 13.9 s.

**Thresholds were not loosened.** None moved.

**A 3-D/3-D similarity estimator was measured and refused.** The obvious second
opinion — Umeyama over the landmarks both segments triangulated, RANSAC'd — was
built and scored, because `_pnp_observations`' own docstring justified rejecting
it on a density argument that is **false on this data**: segments 14 and 29
share 1,552 distinct mutually-triangulated landmark pairs, not "fewer than six",
and 43 of the walk's 253 pairs clear ten. The conclusion survives anyway, for a
better reason:

- On known-answer splits it returns **0.3027 where the truth is exactly 1.0**,
  and RANSAC *beats the truth on inlier count doing it* — 28 inliers against 2.
  Four of eight known-answer cases came back wrong, three catastrophically.
- The cause is structural, not fixable with more data: a 3-D/3-D similarity is
  driven by landmark DEPTH, and `landmark_gate` admits landmarks at
  `min_parallax_deg`, i.e. ~100% depth uncertainty. The landmarks two halves of
  segment 29 share land **1.05x scene depth apart**. PnP is driven by bearings,
  which are accurate.
- On the four pairs the gate already admits — the nearest thing to known-good —
  its scale is off by up to **1.23x** and its rotation by up to **37°**, while
  the two PnP directions agree to 8% and 3.5°. Its error envelope on good data
  is wider than the 10% question it would be asked to arbitrate.

It reached the same +1 segment / +3,117 points on walk B as the fix that
shipped, by a route with a measured 50% catastrophic-failure rate on known
answers. It is good at *refusing* — it refused every cross-world pair, every
permuted correspondence set and every collapse — and that is the only role it
could safely have. As a veto it changes no verdict on either walk, so it would
be an inert guard, which is worth saying rather than shipping.

Its one useful corroboration is recorded: on (14,29) it independently returns
**1.023**, agreeing with the PnP reverse direction (1.045) and with the filtered
forward (0.993), and disagreeing with the unfiltered forward (0.435). A third
estimator, built on different information, says the same thing the fix says.

---

## 7. Limitations

- **The new edge has no cycle to check it.** (14,29) joins a cluster that has
  zero closures before and after; `cycles_checked` is 0 either way. Every
  clause that judged it judged the pair from its own evidence, and the one
  independent check this module can run had nothing to run on. Segment 29 also
  becomes the *reference* — the frame everything else is drawn in — so if that
  edge is wrong, it is wrong loudly. The corroboration available is indirect:
  three estimators (PnP reverse, PnP forward after filtering, and an
  independent 3-D/3-D fit) put the scale at 1.045, 0.993 and 1.023.
- **No ground truth for accuracy.** "Registered" still means two independent
  solves in opposite directions agreed. The ground-truth splits used here prove
  the estimator's *self-consistency* under a known answer; they do not prove
  that a cross-segment placement is metrically right.
- **Two walks, one room.** Every number above comes from one apartment on one
  afternoon. The tolerance plateau is wide on three worlds; that is not the
  same as being wide everywhere.
- **Absolute scale is still unknown** and always will be, monocular. Nothing
  here changes that and nothing should be read as claiming metres.
- **Walk A gains nothing.** 15 of its 23 segments have zero geometry, holding
  110 of its 229 keyframes, and its one strong visual revisit (segments 23↔24,
  164 verified inliers) is unusable because segment 23 reconstructed nothing.
  That is an upstream reconstruction failure — 89 cascaded pose refusals from
  17 root ones — and no registration change reaches it.
- **The false-merge proxy is a proxy.** Cross-world negatives may be too easy
  and self-pair positives are optimistic. See §8.

---

## 8. What was checked on the safety side

The filter only ever removes cameras, so no pair is admitted on MORE cameras
than before. That is the whole of the structural guarantee and it is narrower
than it sounds: **removing outliers also tightens the surviving fit**, so
`scale_ambiguity`, `reprojection_px` and `rotation_disagreement_deg` can all
improve. They do — on world `6502da15` pair (6,7) the filter takes ambiguity
from 207.38 to 1.00 and reprojection from 29.88 px to 1.90 px, turning a pair
refused by four clauses into an admission at exactly `cameras = 3`. So "it can
only refuse more" is **false**, and the first version of this note said it.

What actually holds the line is **reciprocity, which the filter cannot forge**.
The two directions PnP different segments' landmarks into different images, so
a fabricated group in one has no counterpart in the other. Tested directly:
force the filter onto the *fabricated* group of (14,29) and try every
reverse-camera subset of size ≥ 3 — **none of the 16 reaches reciprocity within
10%** (best 0.4172).

Three independent checks:

- **Cross-world negatives.** 1,657 pairs of segments from different worlds;
  464 produce verified matches, 47 reach `admit()`. **4 admitted, with the
  filter on and with it off — the filter changed none of them.** But the
  premise is weaker than it looks: six of the twelve usable worlds were
  captured within 36 minutes of each other in the same home, and two of the
  four "false merges" are between worlds **67 seconds apart**, so they may be
  genuine place matches. The two cross-day ones are unresolved. **This
  measures nothing about the change; it measures the gate, and the gate did
  not move.**
- **Synthetic impossible partners.** 18 pairs built from real imagery with
  perfect descriptor matches, where the points were moved by one Sim3 and the
  cameras by a different one — geometrically impossible by construction. **The
  gate refuses all 18.** Four of them are caught by
  `max_rotation_disagreement_deg` **and nothing else**, at reciprocity
  0.96–1.02 and 0.57–1.12 px reprojection. That clause's own comment calls it
  inert ("changes no verdict on the corpus available today"). It is inert on
  real pairs and it is the only thing standing between this gate and a
  geometrically impossible merge that reprojects sub-pixel. **Do not delete
  it.**
- **Corpus-wide before/after.** 3 pairs gained, **0 lost**, across every saved
  session (§4.2). Every verdict that moved, moved toward admission of a pair
  that survives every unchanged clause.

Two things the safety work surfaced that are **not** about this change and
should not be read as reassurance:

- **`max_reciprocity_error = 0.10` is about 2× the measured self-pair noise
  floor, not comfortably clear of it.** The worst honest self-pair sits at
  0.0576. And the four cross-world admissions sit at 0.0063–0.0859 — *inside*
  the honest band. There is no reciprocity threshold that separates cleanly on
  this corpus.
- **`max_reprojection_px = 2.0` would remove all four cross-world admissions**,
  at a cost of 4 true ones — because the false merges reproject at 2.24–2.38 px
  while honest self-pairs sit at 0.35–0.84 px. Not changed here: three of the
  four "false" merges are unresolved and may be real, and tightening a
  threshold on that basis is the mistake this repository has already recorded
  twice.

**Determinism.** Registration on the drawer walk is byte-identical over three
fresh processes and over six run concurrently. A separate harness reported 11
of 1,657 verdicts moving between builds run under concurrent load, including
four admit/refuse flips on marginal cross-world pairs; that could not be
reproduced on the walks that matter, and remains an open question about
`solvePnPRansac` under OpenCV's parallel scheduling rather than a property of
this change.

---

## 9. Instruments added

- `scripts/world_replay.py` — rebuilds a recorded walk from raw frames,
  deterministically. Reproduces both 2026-08-29 sessions figure for figure
  (frames, keyframes, rejection histogram, segments, solved poses, points) and
  reports whether it still does. One walk is several captures; `CASES` pins
  which, in time order.
- Every candidate pair now produces a verdict. The branch for "the matcher
  could not link them" used to `continue`, so walk B's report covered 228 of
  253 pairs and the missing 25 were indistinguishable from pairs never
  enumerated. That distinction is the one that says whether a walk's problem is
  **retrieval** or **estimation**.
- Every pair row now carries `verified_frame_pairs` and `inliers`, so a refusal
  with 4,449 inliers behind it reads differently from one with 16.
- `admitted_components` — only the largest is ever served, and nothing used to
  say whether a second existed.
