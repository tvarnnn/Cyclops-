# Physical tests queued by the overnight SLAM run — 2026-08-27

Nothing here was performed. Each entry states a hypothesis, the exact
capture to make, what would confirm it, and **what would falsify it**, so
a walk that disproves the hypothesis is as useful as one that confirms it.

Ordered by value. **PT-1 is worth more than the rest combined.**

---

## PT-1 — A walk with deliberate lateral translation

> ### PERFORMED 2026-08-29. HYPOTHESIS REFUTED, USEFULLY.
>
> World `af47007c56924e568b096cfc0eaf2b24`, session
> `7864d3b370ed42f8b292408090a205ff`: the wearer circled a textured drawer
> laterally and returned. 1074 frames, 218 keyframes, 36 segments, 13,050
> points.
>
> **The wearer translated, and it did not fix registration by itself.**
> Span/depth on the segments that matter is now 0.72–0.79, not 0.02–0.06 —
> the outbound and return legs are the two best-conditioned segments in the
> corpus. The capture is no longer the binding constraint. But the pair
> carrying the revisit, segments 14 and 29 with 20,267 verified inliers
> between them, was still refused.
>
> So the hypothesis's *premise* held (behaviour changes span/depth) and its
> *conclusion* did not (registration did not follow). Chasing the
> difference found two real defects: registration was never invoked from
> any production path at all, and the Sim3 estimator was placing target
> cameras that share no view with the source, which collapses the fitted
> constellation and reports a scale ~2.5x wrong. Both are fixed on
> `world-builder/fragment-registration-v1`; walk B goes from 0 placed to
> 6 of 23 segments and 59.9% of its points.
>
> Full account:
> `docs/superpowers/research/2026-08-29-the-drawer-walk-and-the-cameras-that-were-not-there.md`.
> The walk is replayable without the glasses:
> `scripts/world_replay.py --case worldB`.

**Why this is first.** Registration already works on this branch and is
already refusing almost everything, and the refusals are unanimous about
the reason. Read off a real run of `world_registration.py` against the
canonical world (MEASURED, read-only, no `--write`):

| refusal reason | pairs |
|---|---|
| "the wearer stood still: one segment's cameras span only 0.02–0.06 of the scene depth" | **135** |
| "neither direction could be solved" | 6 |
| **admitted** | **2** of 143 |

135 of 141 refusals say the camera did not move enough for scale to be
recoverable *at any quality of match*. No algorithm recovers scale from a
camera that did not translate. Two independent lines of work — this
branch's `span_over_depth` pre-refusal, and the modern-SLAM research
programme's search for a validity statistic — converged on the same
quantity and the same limit.

**Hypothesis.** The dominant constraint on World Builder today is the
capture, not the estimator. Footage with genuine lateral translation will
register a materially larger share of segments.

**The capture.** One walk, 2–4 minutes, in a room already walked so the
content is comparable:
- Move **sideways** past furniture rather than pivoting on the spot.
  Strafing is what creates baseline; turning your head creates none.
- Keep a roughly constant distance from what you are looking at, so
  baseline grows relative to scene depth.
- Deliberately re-enter the same area **twice**, at least 60 s apart, to
  create a genuine revisit.
- Avoid: standing still and panning, walking straight toward a wall
  (the epipole sits in the image and parallax collapses), fast head turns.

**Expected if the hypothesis holds.** `segments_registered` rises well
above the current 3 of 51; the `span_over_depth` refusal share falls from
96% of refusals; median span/depth clears 0.05 on most segment pairs.

**FALSIFIED if.** Span/depth stays in the 0.02–0.06 band despite a
deliberate strafing walk. That would mean the limit is the camera
geometry or the room scale rather than wearer behaviour — a much more
serious finding, and one that would justify reopening whether a
monocular-only map can ever register on this hardware.

**Metrics to capture.** The capture id, then run read-only:
`world_registration.py --root <data>/world_builder --world <id> --format json`
and keep `segments_registered`, `points_registered`, `candidate_pairs`,
`admitted_pairs`, and the full refusal-reason histogram.

---

## PT-2 — Does the feature-starvation gate cost anything real?

**Context.** A gate now refuses a keyframe carrying fewer ORB features
than `MIN_INLIERS` (15). The threshold is arithmetic, not taste: matching
two frames yields at most `min(features)` correspondences, so such a
frame can never reach 15 inliers against anything.

MEASURED on the canonical capture: 10 keyframes carry under 20 features,
8 under 15, minimum **zero**. Every one is `unavailable` or `anchor`, not
one contributes a single support row, and two are segment anchors. Median
black fraction among them is **0.825** — they are face-redaction fill
(`FILL_VALUE = 0`, solid by design because blur is partially invertible).

**Hypothesis.** Refusing them is free or better, because accepting one
installs it as the tracking reference (`engine.py` `set_reference` runs on
every accept) and nothing can track against an 80%-black frame, so the
next loss is arranged in advance.

**The capture.** No new walk needed *if* the existing corpus exercises
it — but the corpus contains only ~8 moving captures from one apartment,
and the gate fired on none of `4fea31e2`'s keyframes. A walk **with a
person in frame at close range**, so the redactor actually fills a large
area, is what would test it.

**Expected.** Segment count does not rise; `poses_solved` does not fall;
the `feature_starved` rejection counter is non-zero.

**FALSIFIED if.** Segment count RISES. That would mean the rejected frame
was holding the tracker's reference chain together despite carrying no
usable features — counter-intuitive, and precisely the shape of result
this codebase has already recorded once for the blur gate, where
reordering made segmentation *worse* and the standing hypothesis was
refuted in-file with "Do not retry it."

---

## PT-3 — Does the multi-reference change help or hurt a real walk?

**Context.** `EXTEND_REFERENCE_DEPTH` makes `_extend` look three
keyframes back for further sightings of landmarks already in the map.
Measured effects conflict by harness (see
`2026-08-27-stage1-results.md`): per-segment it is neutral on poses and
clearly positive on multiplicity, but a full engine replay of capture
`4fea31e2` **halved solved poses, 10 → 5**, while the ≥3-view share rose
15.8% → 38.0%.

**Hypothesis.** The regression is specific to small, sparse captures
where suppressing new-landmark creation starves the map, and does not
appear on longer walks with denser structure.

**The capture.** A long walk (3+ minutes) through a well-textured room,
plus a short sparse one (under a minute, plain walls) for contrast.

**Expected if the hypothesis holds.** `poses_solved` holds or rises on
the long walk while the ≥3-view share improves; the regression reproduces
only on the sparse one.

**FALSIFIED if.** `poses_solved` falls on the long walk too. Then the
mechanism costs reconstruction generally and should be reverted or gated
far tighter (the first thing to try is tightening the admission
reprojection bar from `PNP_REPROJECTION_ERROR_PX` 3.0 px to
`RANSAC_THRESHOLD_PX` 1.0 px).

---

## PT-4 — The product bar itself

Every number in this programme is **self-consistency**. There is no
ground-truth room geometry, no reference trajectory, no metric scale.
"Registered" means two independent solves in opposite directions agreed;
it does not mean correct.

**The only test that addresses the actual product goal** is a person
looking at the reconstruction and saying whether they can tell which
geometry is which part of the room.

**The capture.** Walk PT-1. Build the world. Render the registered
geometry. Then, without being told which is which, identify at least
three distinct pieces of furniture or architecture from the point cloud
alone.

**Expected.** Recognisable, or not. Both are informative.

**FALSIFIED if.** Points appear and nothing is recognisable — which is
the failure the standing bar was written against: *"points appeared"* is
not the bar; *"the wearer can recognise which geometry corresponds to
which part of the room"* is.

---

## What is blocked and why

- Everything above needs the glasses on a head in a room. None of it can
  be simulated: the synthetic scenes render a perfect pinhole and, per
  `test_world_builder_incremental.py`'s own docstring, say nothing about
  the Ray-Ban camera. Synthetic footage in particular does not exhibit
  segmentation at all, which is the thing being fixed.
- `test_world_registration.py`'s entire real-corpus class (10 tests)
  **silently skips** in the world-builder worktree, because the corpus
  lives only in the main repo checkout. Those are the only end-to-end
  checks that the registration gate does anything on real data. Fixing
  that is a repo-layout question, not a physical test, but it belongs in
  the same conversation.
