# World Builder global solver — design

**Date:** 2026-09-06
**Branch:** `world-builder/global-reconstruction-v1`
**Evidence:** `tower/docs/world-builder-reconstruction-experiments.md` (E1–E9)

## 1. The problem, restated from evidence

The World Builder reconstructs each tracking segment as a forward PnP chain
and then tries to glue segments together with a Sim3 fitted between two
independent, already-drifted reconstructions. Three independent audits of the
2026-09-06 walk (438 keyframes, 34 segments) agree on why that does not make
a room:

1. **The chain refuses or strands half the walk.** 210 of 438 poses are
   refused, 185 of them cascaded from 25 root refusals; segment 4 holds 74
   keyframes and zero geometry although its images link strongly to
   segments 13/14/16 (541/433/412 inliers).
2. **Segments that do reconstruct are not rigid.** Segment 16's scale drops
   ~10× along its own length; segment 5's rises 7–15×. No Sim3 fits a piece
   whose unit changes, so the registrar's refusal of (16,17) — 226 verified
   frame pairs, 32,101 inliers, plainly the same desk — is *correct*.
3. **The features are there.** Sequential SIFT matching over the same 438
   frames yields 8,550 verified pairs, 5,559 with ≥15 inliers.

A global solver over pairwise constraints across every keyframe (E8) places
**428 of 438 keyframes in one frame at 0.84 px** where the baseline had 58.
Where the local chain is sound (segment 9) the two solvers agree to 0.3% of
the segment's extent; where it is known to have diverged (segment 16) they
disagree. The engine below is that solver, made incremental and wired into
the derived tree the phone already reads.

## 2. Engine recipe (measured, E6/E8/E9)

```
keyframe images (raw capture frame when on disk, else the redacted keyframe copy)
  -> undistort ONCE with the session calibration (alpha 0, crop to valid ROI)  ~1 ms/frame
  -> SIFT (pycolmap, CPU, <=4096 features)                                      ~15 ms/frame
  -> sequential matching, overlap 20 (+ loop closure, finalisation only)        ~75 ms/frame
  -> GLOMAP global mapping (rotation averaging -> positioning -> BA)             43 s @ 438
       fallback: incremental mapping when GLOMAP yields no model
  -> support floor: an image with < MIN_IMAGE_OBSERVATIONS 3D observations is UNPLACED
  -> components: each COLMAP model with >= MIN_MODEL_IMAGES images is one coherent frame
```

Intrinsics are held fixed (E2: refining them is worse and 3× slower). Feature
database and matches persist per session; a re-solve only pays for new
keyframes (E9).

## 3. Where it sits in the architecture

```
FAST PATH (unchanged)         BACKGROUND PATH (new)              FINALISATION (new)
frame -> observe()            world_solve.py subprocess          world_solve.py --final
  keyframe decision             every N new keyframes              after stop_session()
  local segment solve           extract/match new frames           loop closure widened
  engine.build() (rebuild)      GLOMAP -> solution.json            GLOMAP -> solution.json
                              engine.build() MERGES solution      engine.build() merges
                                -> poses/points/placements          -> final derived tree
```

`scripts/world_build_session.py` (the follower the Tower supervises) owns the
cadence: after each rebuild it launches `scripts/world_solve.py` if none is
running and enough keyframes are new; at stop it runs the final solve
synchronously and then builds. `engine.build()` stays the single writer of the
derived tree; it calls `global_solve.merge()` when a solution exists for the
session. The web process never learns any of this — it still only knows the
builder as a command line.

## 4. Representation in the derived tree (contract-compatible)

The geometry contract (`docs/contracts/WORLD-BUILDER-GEOMETRY.md`) is kept as
is. Tracker segments remain the unit; the global solution is expressed through
the layer the contract already has for it, **placements**:

- For every tracker segment with ≥ 1 globally-posed keyframe: the segment's
  poses and points are rewritten in the segment's own frame, anchored at its
  first posed keyframe (identity), and its placement is `registered` with the
  real Sim3 (scale 1, the anchor pose) into the component's reference segment
  (the lowest segment index in that component). Every segment in a component
  therefore shares a `reference_segment`, which is exactly the contract's
  composition rule.
- Points are assigned to the segment of the keyframe that first observed
  them, so per-segment `content_hash` still names real content.
- Keyframes the solver did not pose keep a refused pose row
  (`status: unavailable`, degeneracy `unregistered`).
- Segments with no posed keyframe keep their LOCAL geometry (if any) and a
  `refused` placement whose reason says the global solve did not place them.
- Segments not yet covered by the last solve (newer than its keyframe
  horizon) keep their local geometry and an `unplaced` state — the live
  "this is still building" case the contract already describes.

The support table (`support.json`) is written for solved segments as
`[segment, frame_index, feature_index, point_index]` where `feature_index`
indexes the solver's own keypoints; `world_registration.py` (the Sim3
registrar) is not run on a globally-solved session.

### Coverage / confidence (additive fields)

Per pose row: `observations` (int, 3D points this keyframe observes; 0 for
unposed). Per segment in the manifest: `coverage` ∈
`confident | partial | unresolved`, derived from the median observations of
its posed keyframes and its point count; `unresolved` is "keyframes exist,
no geometry". Unseen space is what has no keyframe at all and is never drawn.
Nothing is invented for gaps.

## 5. Truthfulness rules the solver enforces

- A camera with fewer than `MIN_IMAGE_OBSERVATIONS` 3D observations is not a
  measurement and is unplaced (E3: GLOMAP's zero-support cameras sat 200
  units from the room).
- Two COLMAP models are two frames. They are never composited; they carry
  different reference segments.
- Scale stays `unknown` (monocular). Placement scale between segments of one
  component is exactly 1 because they are one reconstruction.
- A solution is bound to the keyframe digest it was solved from; a stale
  solution is merged only for the keyframes it covers and reported as such.

## 6. Dependencies and licensing

`pycolmap 4.2.0` (BSD-3, `Requires-Dist: numpy` only; Windows cp312 wheel;
CPU-only on Windows). The wheel bundles Ceres with SuiteSparse (CHOLMOD,
which has GPL modules) — recorded for product licensing review, not a
technical blocker. The solver is optional: when `pycolmap` is missing the
World Builder behaves exactly as before and the manifest says why.

## 7. What is deliberately not done

- No learned features or feed-forward 3D models in the product path.
  LightGlue/ALIKED run on the GPU here and MapAnything-apache is permissive,
  but neither was needed to reach one coherent frame on the corpus.
- No dense geometry. Sparse points with colour are what the contract carries.
- No metric scale.
