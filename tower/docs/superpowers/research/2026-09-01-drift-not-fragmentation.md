# Drift, not fragmentation

**Date:** 2026-09-01
**Branch:** `world-builder/tracking-recovery-v1`, from
`world-builder/fragment-registration-v1` @ `e847339`
**Corpus:** five replayed physical walks, 6,548 frames, plus the eight-capture
pinned A/B corpus and ground-truth synthetic scenes
**Status:** measured.

---

## 0. The result in one paragraph

The World Builder was believed to fail because tracking restarts too often: a
two-minute room walk arrived on the phone as 30 segments and 18 disconnected
fragments, and the obvious reading was that the tracker gives up too easily. It
does give up too easily, and that is fixed here. But it is not why the walk does
not become a world. **The forward-only pose chain drifts, badly and silently,
and everything downstream is a consequence of that.** On a perfect synthetic
strafe with nothing refused and nothing blurred, rotation error grows from
0.95° at six keyframes to 33.98° at forty, and the reconstruction contracts by a
factor of three. On the real 2026-09-01 walk the one segment that did survive
170 keyframes has camera centres reaching 8 × 10¹⁰ and points reaching
2.9 × 10¹³ — the largest coherent piece of geometry in the corpus is not
coherent. Cross-segment registration then refuses to place it, **correctly**,
because it solves a Sim3 and two pieces whose internal geometry disagrees should
not be glued together. Fragmentation was the symptom everybody could see; drift
was the thing making the fragments unmergeable.

---

## 1. Physical dataset provenance and baseline correction

This section exists because the experimental provenance was corrected
mid-investigation and the correction changes what "before" means.

**The physical walks were captured while the Windows Tower ran `main @
768cecf`.** `world-builder/fragment-registration-v1` was *not* active during any
physical run. Current `origin/main` has since advanced to `b5c9089` through the
CV Lab and Object Memory validation merges; that advance contains no World
Builder change and this branch does not descend from it.

Three baselines must therefore be distinguished, and are, throughout:

| | what it is | tip |
|---|---|---|
| **A — physical era** | the code that actually processed the walks | `main @ 768cecf` |
| **B — registration** | the branch this work descends from | `world-builder/fragment-registration-v1 @ e847339` |
| **C — this branch** | tracking recovery + bundle adjustment | `world-builder/tracking-recovery-v1` |

**A and B produce identical reconstructions, and that is provable by
inspection rather than by measurement.** `git diff 768cecf e847339 --
tower/tower/world_builder/` is **empty**: the engine, the backend, the keyframe
policy and the frame tracker are byte-identical. The registration branch changed
only `scripts/world_registration.py`, `scripts/world_build_session.py` (the
`--register` wiring), `scripts/world_replay.py` (new), and a `world_register`
setting in `config.py` / `main.py`.

So the A → B delta is entirely **whether registration is invoked at all**. The
physical-era world on disk confirms it: `derived/96e0344d…/` contains
`points.json`, `poses.json` and `support.json` and **no `placements.json`** —
registration never ran, which is exactly why the phone drew 18 unconnected
fragments.

Empirical confirmation of the same fact: replaying the 2026-09-01 frames through
B reproduces the physical session figure for figure — 2,613 frames observed, 434
keyframes, 30 segments, 323 solved poses, 81 refused, 30,382 points, and the
identical four-way rejection histogram (`insufficient_motion` 1782, `blurred`
289, `tracking_lost` 16, `tracking_degraded` 92). The `worldA` and `worldB`
cases likewise report `reproduces_recorded_session: true`.

**Where replay is not exact, this report says so.** `world_replay.py --frames`
reads a directory rather than the capture journal, so `source_seq`, `tx_seq` and
receipt time are enumeration indices rather than the recorder's values. Nothing
in keyframe selection or geometry reads them — which is why the replay comes out
exact — but no transport claim is made from a replay anywhere in this document.

---

## 2. What the physical-era run actually did

2026-09-01, world `b5feee12718c460eabc571074563d669`, session
`96e0344d89ba46299bbe2b783eca797e`. Two captures,
`1ac63b51619f45e18ae8f8cce1440617` then `60bf1b02daf94c0ab713cd0355a5de58`,
2,613 frames, 64.7 MB.

Segment boundaries, read from `events.jsonl`, are 16 `tracking_lost` and 16
`solve_chain_broken` — 32 segment indices minted, 30 of which received a
keyframe. The manifest splits the 81 refused poses into **21 root and 60
cascaded**.

The distribution is the interesting part:

```
boundaries at keyframe:  1 12 13 14 15 19 20 21 22 39 40 47 57 59 62 65 70
                        76 97 102 111 145 146 155 156 190 191 217 246 249 264
                        ... then nothing until 433
```

The first 264 keyframes produced 31 segments. **The last 170 keyframes produced
one** — segment 31, with 168 solved poses, 39% of the walk in a single
coordinate frame. That segment is the subject of §4.

Corpus-wide, over every calibrated session on disk: 1,949 refused poses, of
which **1,812 (93%) are keyframes at which no solver ever ran**, and only 137
were an attempt that failed. Of those 137 root attempts, 118 had **≥100 ORB
matches with a median of 176 epipolar inliers** — the images plainly overlapped
and the pose was refused anyway.

---

## 3. The tracker: what the code actually did

`_Chain.broken` was a one-way latch. Set by the first keyframe whose pose
refused, and thereafter `extend()` returned early with `unavailable` **without
running ORB detection at all**. The engine's only available response was to cut
a segment. Where the restart budget (`MAX_BARREN_SEGMENTS = 1`) was already
spent, not even that happened, and the stranding continued silently with no
journal record — segment 25 of the physical walk stranded 27 consecutive
keyframes that way, segment 9 stranded 15.

Two further facts from reading the code:

- **The pose was solved against exactly one reference**, `observed[(N-1, f)]` —
  landmarks the immediately preceding keyframe had seen. A landmark
  triangulated four keyframes earlier and still in plain view was invisible to
  the solve.
- **A refused keyframe still became the next reference.** It has no entry in
  `absolute`, so it can supply no 3-D correspondence at all, which made the next
  refusal a certainty rather than a risk.

Every mature monocular system holds the opposite invariant, and the survey found
no exception: ORB-SLAM2/3 set `mpReferenceKF` only in `CreateNewKeyFrame()` and
`UpdateLocalKeyFrames()`, both reachable only after a successful track; DSO sets
its coarse tracking reference only from `makeKeyFrame()`, after the window
optimisation; stella_vslam inherits the previous reference unchanged and
advances it only via `local_map_updater`; SVO explicitly restores the last good
pose on failure. **No mature system re-roots on a frame whose pose was not
estimated.**

### 3.1 What was implemented

`references` is now the last `EXTEND_REFERENCE_DEPTH` keyframes that **have**
poses. A refusal contributes no reference, so the next keyframe is solved
against the last view that still has coordinates. `_Chain.failures` counts
consecutive refusals and only `MAX_RECOVERY_KEYFRAMES` of them in a row set
`broken`. **No acceptance threshold moved** — `MIN_PNP_CORRESPONDENCES = 12`,
`PNP_REPROJECTION_ERROR_PX = 3.0`, `MIN_INLIERS = 15`, `MIN_INLIER_RATIO = 0.05`
and `MIN_TRIANGULATION_ANGLE_DEG = 0.5` are the values they were, and a test
pins them.

With the bundle adjustment switched OFF, setting
`MAX_RECOVERY_KEYFRAMES = 1` reproduces the parent branch exactly — `worldB`
returns 36 segments, 108 solved poses, 13,050 points and a 7,821-point dominant
component, figure for figure. That is what makes the recovery budget an isolated
variable, and it is why every recovery comparison in §6.3 is trustworthy: the
restructured references, the failure counter and the seed-anchor retry are all
present at 1 and all inert.

**1 is also the shipped value**, for reasons that are the subject of §3.3.

### 3.2 What was implemented, measured, and REJECTED

Feeding all `EXTEND_REFERENCE_DEPTH` references' correspondences into a single
PnP is the obvious widening and is what ORB-SLAM's `TrackLocalMap` looks like
from a distance. Measured on the 2026-09-01 walk against identical recovery
behaviour:

| references into the PnP | solved | reprojection p99 | rows over 3 px | dominant component |
|---|---|---|---|---|
| 1 | 329 | 4.37 px | 2.21% | 38.8% |
| 3 | 327 | **8.87 px** | **5.19%** | **19.8%** |

`TrackLocalMap` is not "match more keyframes". It projects landmarks through a
**predicted** pose and searches a small radius around the prediction, so
appearance only ever chooses among candidates that are already geometrically
plausible. At the point where our pose is being solved there is no prediction,
so a wider reference set is pure descriptor matching over a wider baseline —
where ORB is weakest. The older references keep their post-solve
re-observation job, gated on reprojecting through the pose that was solved.

A separate prototype of the *proper* form — solve, then project the whole
segment's landmarks through that pose and re-solve within a 12 px radius —
halved the rotation error but did not remove the drift (20.7° → 9.8° median on a
40-keyframe strafe). Long-lived data association is necessary and not
sufficient; the landmarks themselves were triangulated from drifted poses, and
re-observing a wrong landmark constrains you to a wrong pose. That result is
what pointed at §4.

---

### 3.3 Why the shipped budget is 1

`MAX_RECOVERY_KEYFRAMES` is exactly the largest reference gap a solve may be
admitted at, and **no acceptance gate in this backend is a function of that
gap**. An adversarial pass measured what that costs, against exact synthetic
ground truth, holding the target keyframe and its image FIXED and moving only
the reference. Median relative rotation error of that one solve, six scene
seeds:

```
gap      1     2     3     4     6     8    10    12
forward  0.78  0.86  1.33  2.48  3.04  3.38  4.47  6.09   deg
strafe   1.57  2.71  4.55  5.86  9.10 11.89 14.91 19.41   deg
```

Zero refusals at any gap, 22–250 PnP inliers throughout.

Over REPEATING texture — an ordinary room — it is not drift at all. A room
tiled at 1.5 m, a continuous walk at 0.1875 m per keyframe, no teleport, no
occlusion, no adversary, nine samples per gap:

```
gap            1      2      3      4      6      8
walked      0.188  0.375  0.562  0.750  1.125  1.500   m
median err  0.007  0.207  0.241  0.411  0.766  1.499   m
over 10 cm    0/9    6/9    9/9    6/6    9/9    9/9
```

Read the two rows together. Whatever the gap, the solve publishes roughly ONE
keyframe of motion: it matches the wrong repetition, and the keyframes the
camera crossed cease to have happened. **At gap 8 the camera moved 1.500 m and
the pose reports 0.001 m — with 169 PnP inliers, 0.14° of rotation error, and
published support reprojecting at 0.22 px median.** Every instrument this
pipeline owns says that pose is excellent.

The physical corpus agrees the bound is not free coherence either; see §6.3.

That is not an argument for restoring the latch, which was wrong for 1,812
keyframes. It is an argument that the mechanism is right, the bound is the only
thing carrying the risk, and the value the evidence supports is 1. What would
earn a larger value is not a better matcher — appearance is precisely what lies
here — but an instrument that checks the displacement a recovered pose implies
against the number of keyframes it skipped. That instrument does not exist yet.

The adversarial pass found one more thing worth recording: with the
threshold-equality assertion deselected, mutating `PNP_REPROJECTION_ERROR_PX`
from 3.0 to 30.0 broke **no test in the pre-existing suite**, nor did loosening
`MIN_INLIERS`, `MIN_TRIANGULATION_ANGLE_DEG` or `MIN_INLIER_RATIO`. Every
acceptance threshold was guarded only by an assertion that the constant equals a
literal. `tests/test_world_builder_recovery_safety.py` fixes that.

## 4. The finding: the chain drifts

Measured on `tests/synthetic_scene.py`'s furnished room, with **no refusals, no
blur and no injected damage**. The generator is recorded exactly, because an
adversarial review reconstructed it wrongly and got numbers up to 2× apart — and
because guessing a slightly larger step walks the camera into a wall, which is
the confound `poses_outside_room` exists to prevent and which produced a wrong
diagnosis in this project once already:

```python
poses = ss.strafe(n, step=0.10, start=(-2.0, -1.6, 0.6))
assert ss.poses_outside_room(poses) == []       # holds for every n below
```

| keyframes | rotation error median / max | max drift ÷ path | per-step scale ratio, first third → last |
|---|---|---|---|
| 6 | 0.95° / 1.69° | 4.7% | 7.63 → 7.87 |
| 12 | 1.69° / 3.46° | 2.7% | 7.63 → 8.47 |
| 20 | 3.19° / 9.17° | 9.8% | 7.63 → 8.85 |
| 30 | 5.71° / 18.88° | 11.3% | 7.63 → **3.46** |
| 40 | 9.21° / 33.98° | 18.2% | 7.63 → **2.43** |

The last column is the ratio of recovered to true camera motion per step. It is
flat for twenty keyframes and then moves by a factor of three. The pipeline is
trustworthy for roughly 15–20 keyframes and progressively falsifies beyond that.

**The DIRECTION of that scale error is walk-dependent and an earlier draft of
this report overgeneralised it.** On this walk the reconstruction contracts; the
review measured a step of 0.08 where the same chain EXPANDS by 15× over the same
forty keyframes. What is invariant is that the scale drifts without bound, not
that it shrinks. The review's independent reconstruction at that step measured
36.1% drift and 40.7° of rotation error at forty keyframes — worse than the row
below, on a walk fully inside the room — so the finding is not a wall artifact
and is if anything understated here.

That number matters because the physical walk's segments averaged 14 keyframes.
The fragmentation everybody was trying to remove was, accidentally, holding
segments at about the length this backend can still be trusted over — and
cross-segment Sim3 registration was the only thing in the system absorbing the
scale error between them.

**So reducing segment count is not, by itself, an improvement.** Measured: with
recovery alone the drawer walk goes from 36 segments to 20 and from 108 solved
poses to 120, and its dominant connected component falls from 59.9% of the
geometry to 26.5%. The world got tidier and less true.

### 4.1 The 170-keyframe segment

Independently confirmed on the real data. Segment 31 of the 2026-09-01 walk —
168 solved poses, `source_seq` 1214–2598, the last 21 seconds of the walk:

```
first |camera centre| > 1e3 at pose index 77 of 169
|C|   p50 9,293   p90 6.38e9   p99 5.26e10   max 8.04e10
|xyz| p50 3.14e5  p95 4.78e10  max 2.89e13
```

`span_over_depth` reports 68.67 for that segment and `pair_is_hopeless` lets it
straight through, because the measure has no upper bound and cannot tell
"plenty of parallax" from "the solve diverged".

### 4.2 Why the record said bundle adjustment would not help, and why that is no longer true

The standing conclusion was that a bundle adjuster measured 0.00% improvement at
16, 32 and 104 keyframes because the observation graph is a chain whose median
covisibility span is 1 — 66% of landmarks seen by exactly two views. That was
true of the engine it was measured on. Measured at HEAD on the same fixture:

```
landmarks 5,876   mean views 4.67   >= 3 views 67.2%
observation span: median 3, p90 9, p99 24; 28% span >= 5 frames
```

`EXTEND_REFERENCE_DEPTH`'s guided re-observation built the graph the earlier
attempt was missing.

And the drift is **reachable by optimisation** rather than baked in. Reprojecting
the very same observations through **ground-truth** poses gives RMS 0.49 px
against 0.95 px for the solved poses: the solved reconstruction is demonstrably
not at the minimum. (A first attempt with `scipy.least_squares` reported no
improvement at all; that was a solver failure — a rank-deficient Jacobian under
`x_scale="jac"` terminating on a zero step — not a geometry result, and it is
recorded here because it nearly ended the investigation in the wrong place.)

---

## 5. `world_builder/bundle.py`

Levenberg-Marquardt over camera poses and landmark positions with the Schur
complement eliminating the landmarks. NumPy and OpenCV only — scipy is not a
Tower dependency. Run over a sliding window on the **keyframe** path, never the
frame path.

Structure, and the parts that are load-bearing:

- **Window** `BUNDLE_WINDOW = 12` cameras, the oldest
  `BUNDLE_ANCHOR_CAMERAS = 2` held fixed so an adjustment cannot slide geometry
  that has already been published, and so the seven free parameters of a
  monocular reconstruction stay pinned.
- **Cadence** `BUNDLE_EVERY = 3` solved keyframes, `BUNDLE_ITERATIONS = 4`.
  Bounded rather than run to convergence: the window is re-adjusted a few
  keyframes later anyway, and an unbounded optimiser on a live path is how a
  walk ends mid-room.
- **Landmarks are compacted to the window** before the solve, so an
  adjustment's cost grows with the window rather than with the length of the
  segment.
- **`fixed_points`** for landmarks first seen before the window: they must not
  move, because their older observers cannot follow, but their observations
  stay in the problem as the anchor tying this window to geometry already on
  disk.
- The reduced camera system is formed over observation **pairs** sharing a
  landmark, built once per call rather than per damping attempt, and
  accumulated with `bincount` rather than `np.add.at`. Both changes were forced
  by profiling: the first version spent 3.0 s of a 9.6 s walk inside Python-level
  `tile()` calls, 140,000 of them.

### 5.1 The invariant that had to be learned twice

> **Every observation of a camera that is allowed to move must participate in
> the optimisation.**

Drop an observation for any reason — too few views on its landmark, a per-
landmark view cap, a sampling shortcut — and its landmark stays put while its
camera moves out from under it. The **trajectory** improves and the **support
table**, which cross-segment registration solves PnP against and the viewer
draws, gets worse. It is invisible unless you measure the support table.

This shipped twice on this branch and was caught twice by the corpus:

| constant | value | published reprojection median / p99 | rows over the 3 px gate |
|---|---|---|---|
| *no adjustment at all* | — | 0.543 / 3.974 px | 1.79% |
| `MIN_VIEWS_FOR_ADJUSTMENT` | 3 | 0.723 / **13.698** px | **9.55%** |
| `MIN_VIEWS_FOR_ADJUSTMENT` | 2 | 0.522 / **2.761** px | **0.57%** |

(drawer walk, adjustment on, nothing else changed)

| constant | value | published reprojection median / p99 | over 3 px | dominant component |
|---|---|---|---|---|
| *no adjustment at all* | — | 0.587 / 4.732 px | 2.54% | 8,285 pts (27.3%) |
| `MAX_VIEWS_PER_LANDMARK` | 8 | 0.726 / **15.800** px | 9.51% | 7,521 pts (24.1%) |
| `MAX_VIEWS_PER_LANDMARK` | 16 | 0.546 / **2.781** px | **0.69%** | **16,340 pts (53.5%)** |

(2026-09-01 walk, adjustment on, nothing else changed. **These rows use
the PARENT's per-side-8 sampler**, not the shipped product budget, which
is why the dominant-component figures here differ from §6's — review
found the two tables reporting different numbers for nominally the same
build, and this is the reconciliation. The cap-8 row is the only figure
in this report not recoverable from a retained sweep record; it is the
build that motivated the change and was overwritten by it.)

The argument for `min_views = 3` is about **information** and is correct as far
as it goes — a two-view point is exactly determined and constrains nothing. It
is not an argument for removing its rows. A third of the map is two-view.

---

## 6. Corpus results

Five replayed physical walks. **A** is the physical era (`main @ 768cecf`,
what actually processed the walks, registration never invoked). **B** is
`world-builder/fragment-registration-v1 @ e847339`. **C** is this branch.
`dom` is the largest connected component after registration; for A, where
no `placements.json` exists at all, it is necessarily the largest single
segment.

| walk | stage | seg | solved | points | frags | dom pts | dom % | dom kf | dom kf % | admitted |
|---|---|---|---|---|---|---|---|---|---|---|
| **2026-09-01 loop** | A physical era | 30 | 323 | 30,382 | 18 | 6,190 | 20.4% | 170 | 39.2% | 0 |
| | B registration-v1 | 30 | 323 | 30,382 | 13 | 8,285 | 27.3% | 56 | 12.9% | 6 |
| | **C this branch** | 30 | 323 | 25,131 | **9** | **18,817** | **74.9%** | **156** | **35.9%** | **15** |
| **drawer walk** | A physical era | 36 | 108 | 13,050 | 23 | 3,117 | 23.9% | 23 | 10.5% | 0 |
| | B registration-v1 | 36 | 108 | 13,050 | 18 | 7,821 | 59.9% | 61 | 28.0% | 5 |
| | **C this branch** | 33 | **137** | 12,686 | 17 | 7,371 | 58.1% | 61 | 28.0% | **6** |
| **normal walk** | A physical era | 23 | 100 | 9,145 | 8 | 4,311 | 47.1% | 51 | 22.3% | 0 |
| | B registration-v1 | 23 | 100 | 9,145 | 8 | 4,311 | 47.1% | 51 | 22.3% | 0 |
| | C this branch | 23 | 100 | 6,762 | 8 | 2,790 | 41.3% | 51 | 22.3% | 0 |
| **dense 08-29** | B sampler | 6 | 68 | 11,009 | 6 | 6,386 | 58.0% | 42 | 54.5% | 0 |
| | **C this branch** | 6 | 68 | 10,266 | **5** | **8,638** | **84.1%** | **59** | **76.6%** | **1** |
| **long 08-27** | B sampler | 40 | 129 | 18,977 | 20 | 3,739 | 19.7% | 27 | 8.0% | 3 |
| | **C this branch** | 40 | 129 | 18,708 | **19** | 3,682 | 19.7% | 27 | 8.0% | **4** |

The headline row is the 2026-09-01 walk: the geometry sitting in ONE
coordinate frame goes from 8,285 points across 56 keyframes to **18,817
points across 156 keyframes** — from an eighth of the walk to better than
a third of it, and from a quarter of the geometry to three quarters.
Fragments 13 to 9, admitted pairs 6 to 15.

And the quantity this whole line of work is really about — does the
published geometry still reproject. **Percentiles alone would hide the
failure this branch had to find**, so the worst row and the count of rows
whose landmark ended up BEHIND its camera are reported beside them:

| walk | | median | p99 | max | over 3 px | behind camera |
|---|---|---|---|---|---|---|
| 2026-09-01 loop | B | 0.587 | 4.732 | 774.3 | 2.54% | 5 |
| | **C** | **0.541** | **2.509** | **58.1** | **0.24%** | **0** |
| drawer walk | B | 0.543 | 3.974 | 26.0 | 1.79% | 0 |
| | **C** | **0.510** | **2.565** | **13.5** | **0.25%** | 0 |
| normal walk | B | 0.693 | 5.316 | **33.4** | 3.27% | 0 |
| | **C** | **0.558** | **2.457** | 99.2 | **0.16%** | 0 |
| dense 08-29 | B | 0.648 | 3.100 | 23.2 | 1.16% | 0 |
| | **C** | **0.537** | **2.476** | **19.0** | **0.25%** | 0 |
| long 08-27 | B | 0.572 | 4.269 | 42.7 | 2.10% | 0 |
| | **C** | **0.555** | **2.672** | **35.0** | **0.52%** | 0 |

Median, p99 and the over-gate fraction improve on **every walk**, the
over-gate fraction by between 4× and 20×. `behind_camera` is zero
everywhere; an intermediate build of this branch had it rising 5 → 36 on
this walk, which is what §7.5 is about. The single regression is the
normal walk's worst row, 33.4 → 99.2 px — one row of 18,415.

Where C does NOT improve, stated in the terms it actually loses: the
normal walk's dominant component falls from 4,311 points to 2,790 and
the drawer walk's from 7,821 to 7,371. **Those are real point losses,
not a re-normalisation**, and an earlier draft of this section claimed
"on neither walk does the coherent PART of the world get smaller", which
is false in points and was corrected by review.

What IS true, and is the reason to accept them: on both walks the
dominant component covers the IDENTICAL set of keyframes as before — 51
and 61 — so no part of the room left the coherent piece. What left is
landmarks the adjustment moved into the noise-dominated regime, counted
in the manifest under `low_parallax`. The normal walk is the worst case
in the corpus for that trade and is discussed as such in §7.5.

### 6.1 Isolating the registration sampler

Registration is non-destructive and, verified over six repeated runs with
and without reseeding OpenCV, exactly deterministic given a built world.
So the sampler change can be isolated perfectly: hold the reconstruction
fixed and vary only which `world_registration.py` is asked about it.

| world (parent's own reconstruction) | sampler | candidates | admitted | cycles checked | dom pts | dom kf | seconds |
|---|---|---|---|---|---|---|---|
| 2026-09-01 loop | per-side 8 | 153 | 6 | **0** | 8,285 (27.3%) | 56 | 34.1 |
| | product ≤256 | 153 | **10** | **5** | **10,372 (34.1%)** | **70** | 65.7 |
| drawer walk | per-side 8 | 253 | 5 | 0 | 7,821 (59.9%) | 61 | 13.6 |
| | product ≤256 | 253 | 4 | 0 | 7,326 (56.1%) | 56 | 25.8 |
| normal walk | per-side 8 | 28 | 0 | — | 4,311 (47.1%) | 51 | 1.6 |
| | product ≤256 | 28 | 0 | — | 4,311 (47.1%) | 51 | 4.9 |

`cycles_checked: 0` on the 2026-09-01 walk is the line that matters. The
parent's admitted graph there is a **tree**, so not one placement in it
has any independent verification. The wider sample closes five cycles.

It is **not** monotone, and the drawer walk is the counterexample: a
wider sample changes which frame pairs `fit_direction` solves from, so a
pair admitted before can be refused after. The claim the change supports
is that no verdict is reached on less evidence than before, not that no
verdict changes.

### 6.2 What each change is worth — as a 2x2, not a ladder

An earlier draft presented this as an additive ladder, each step adding
to the one above. **That was wrong and review caught it**: the two
changes are not independent, and on an intermediate build the wider
sampler measured NEGATIVE on the adjusted reconstruction while positive
on the parent's, so the decomposition did not survive reordering.

Run as an actual 2x2 on the 2026-09-01 walk — reconstruction on one
axis, registration sampler on the other, dominant component as a share
of published geometry:

| | parent sampler (per-side 8) | product budget ≤256 |
|---|---|---|
| **parent reconstruction** | 8,285 pts / **27.3%** | 10,372 pts / **34.1%** |
| **this branch's reconstruction** | 14,842 pts / **59.1%** | 18,817 pts / **74.9%** |

Both factors are positive on the shipped build and they reinforce:
+6.8 pp from the sampler on the parent's geometry, +15.8 pp on the
adjusted geometry, and +31.8 / +40.8 pp from the reconstruction at either
sampler. The interaction has a mechanism rather than being a curiosity —
a drifted segment presents a warped shape to `fit_direction` no amount of
extra retrieval can reconcile, so widening retrieval pays more once the
geometry it retrieves is true.

The physical-era row is outside this table because registration never
ran at all: 6,190 points, 20.4%, which is a single un-placed segment and
not a component in any meaningful sense.

**The cells above are not directly comparable in absolute points** — the
branch publishes 25,131 where the parent publishes 30,382, for the
reasons in §7.5 — which is why the share is the honest column and why
the keyframe counts in §6 are the better one still.

### 6.3 Recovery, measured across the corpus

With the registration sampler fixed, raising `MAX_RECOVERY_KEYFRAMES`
does not produce a consistent result:

| walk | budget 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 2026-09-01 loop | 41.1% | 37.9% | 39.1% | **67.9%** |
| drawer walk | 55.8% | 55.1% | 28.5% | **27.2%** |

It doubles coherence on one walk and halves it on another, which is the
signature of a mechanism that sometimes glues the right things together
and sometimes glues the wrong ones with nothing able to tell which. §3.1
and `MAX_RECOVERY_KEYFRAMES` carry the ground-truth measurement that says
which: above a gap of 1, over repeating texture, a recovered pose
publishes roughly one keyframe of motion however far the camera walked.
The shipped budget is **1**.

---

## 7. What was not changed

- Object Memory, CV Lab, Document Memory, Translator, Scene Understanding, the
  iOS product shell, the camera profile system: untouched.
- Capture resolution, frame rate, calibration: untouched. The 360×640
  self-calibration is the only one on the machine and every capture in the
  corpus is 360×640.
- No acceptance threshold in the geometry path was loosened. `admit()`'s
  registration gates are byte-identical.
- No metric-scale claim is made anywhere; `scale_state` remains `unknown`.
- No user recording, world, capture or research asset was deleted or moved.

---

## 7.5 The anti-overfitting corpus, and what it caught

`scripts/world_builder_corpus_benchmark.py` runs a PINNED eight-capture
set through the journal path. It is deliberately not the five replay
walks, and running it is what stopped this branch shipping a defect the
replay corpus was structurally blind to.

It reported `bbox_blowup_max` -- the full point bounding box over its
p2-p98 core -- rising **11.0 to 35.2** while every reprojection statistic
improved.

The first thing to rule out was the metric itself, whose docstring warns
that it is computed over all segments at once and that between-segment
offsets contribute to the numerator. A bundle adjustment changes each
segment's arbitrary scale, so the metric's own "both runs see the same
offsets" caveat no longer held. Recomputing it strictly WITHIN each
segment:

| capture | | all-segment | within-segment median | within-segment max |
|---|---|---|---|---|
| 22e9d428 | parent | 2.74 | 1.63 | 6.16 |
| | branch | 35.15 | 1.69 | **219.10** |
| e1c52b9f | parent | 11.01 | 2.86 | 10.97 |
| | branch | 25.42 | 3.35 | **86.39** |

The median is untouched and the maximum explodes. It is real, it is
inside a segment, and it is a handful of landmarks.

**The mechanism, and it is one this project should have expected.** An
under-constrained landmark's degenerate direction is along its own
viewing ray, and moving along that ray costs almost nothing in
reprojection -- which is exactly why every reprojection statistic
improved while the geometry got worse in a way reprojection cannot see. A
two-view point has precisely that freedom, and two-view points are a
third of the map.

The publication gate that would have refused those points -- `landmark_gate`,
with its parallax floor and its 3 px reprojection bar -- runs at landmark
CREATION and was never re-run after an adjustment moved the landmark. So
`bundle.optimise` now returns `point_ok` and `_local_adjust` uses it to
DEMOTE. It can never promote: a landmark the creation gate already
refused stays refused.

Getting the test right took three attempts, and the two that failed are
worth recording because both looked correct.

**Reprojection alone** -- every observation in front of the camera and
within the same `huber_delta` the pose solve admitted it by. This fixed
22e9d428 completely (within-segment blowup 219.1 to 5.08, below the
parent's 6.16) and left e1c52b9f at 87.4, because a point sliding along
its ray keeps its pixel error and is invisible to a pixel test by
construction.

**Absolute parallax** -- demote any adjusted landmark whose observing
rays subtend less than `geometry.min_parallax_deg`, the same
focal-length-derived bound `landmark_gate` uses. It fixes the blowup
(9.23, below parent) and **removes 29% of the pinned corpus's points and
45% of the 2026-09-01 walk's**. Plenty of landmarks sit below that bound
honestly; the creation gate already ruled on them with the same number,
and an adjustment has no standing to re-litigate that ruling.

**Parallax DELTA** -- demote only a landmark whose rays subtended a
usable angle BEFORE this adjustment and do not after. That is exactly
"the adjustment slid this point along its ray", and it is the shipped
form.

A fourth attempt was needed even then. Demoting on the parallax
CROSSING alone still retires a landmark whose angle went 0.200° to
0.198° exactly as readily as one that was slid along its ray, and a map
this full of marginal two-view points has a great many of the first
kind. `PARALLAX_COLLAPSE_RATIO = 0.5` asks that the adjustment at least
HALVE the angle -- that it be the cause rather than the occasion.

| pinned eight-capture corpus | worst bbox blowup | points | solved | mean largest share |
|---|---|---|---|---|
| parent | 11.01 | 71,122 | 620 | 0.3758 |
| adjustment, no re-check | **35.15** | 70,662 | 603 | 0.3919 |
| + reprojection | 25.59 | 69,972 | 603 | 0.3921 |
| + absolute parallax | 9.23 | **50,422** | 603 | 0.3740 |
| + parallax crossing | 10.52 | 62,843 | 603 | 0.3936 |
| **+ collapse ratio (SHIPPED)** | **11.67** | 65,174 | 603 | **0.3964** |

The shipped row has the blowup back at parity with the parent and **the
best largest-share of all six**, which is the coherence number this whole
line of work is about.

**It costs 8.4% of the corpus's published points, and that is not a
rounding error.** Those are landmarks the adjustment moved into the
regime where, in `min_parallax_deg`'s own words, "a landmark is not a
measurement, and its distance is set by pixel noise". The alternative is
publishing a coordinate the pipeline's own gate would refuse. On the
2026-08-29 normal walk -- 23 segments, only three of which reconstruct
anything -- the cost is worst, 9,145 points to 6,762, and that walk is
the honest worst case for this trade rather than an outlier to be
explained away.

That has a visible consequence and it is worth stating plainly rather
than hiding: the sum of `Extension.new_points` deltas no longer equals
`snapshot()`, because a delta is what the backend believed when it
emitted it. The shortfall is exactly the number of retired landmarks,
which `test_extend_reports_only_the_structure_that_keyframe_added` now
asserts. No production code reads `Extension.new_points` today; a live
viewer that appends deltas forever would have to re-read `snapshot()`.

---

## 7.9 Independent review

An adversarial reviewer was asked to disprove the claim, with the built
worlds, the sweep records and both checkouts. It could not dent the
causal result and did dent the reporting. What it verified independently:

- **No acceptance threshold moved.** Not taken on trust from this
  branch's own pinning test: an AST comparison across `e847339..HEAD`
  found `geometry.py` with zero changed definitions, `world_registration.py`
  changed in exactly `cross_matches` plus a new `match_budget`, and
  `Thresholds`, `admit()` and `pair_is_hopeless` untouched.
- **The parent and `BUNDLE_WINDOW = 0` are identical to three decimals on
  every metric across eight configurations**, so the drift A/B is clean
  and the mutation really does reproduce the parent.
- **The drift result is larger than reported**: 16.46° → 0.43° median
  rotation error and 36.1% → 1.1% drift at forty keyframes on a walk
  fully inside the room.
- **The bundle's invariant holds on real data.** Instrumented replay of
  the drawer walk, 108,510 observations: `MIN_VIEWS_FOR_ADJUSTMENT`
  dropped **zero** rows belonging to a free camera, and
  `MAX_VIEWS_PER_LANDMARK` dropped **25 of 108,510 (0.023%)** — the
  residual risk the constant documents, now quantified.
- **Cost is bounded by the window, not by segment length.** Median
  adjustment cost is flat across segment-length buckets from 0–9 up to
  50–59 keyframes, with `window_landmarks` steady at 1,200–1,650.
- **The shutdown claim is exact**: `register()` contains no write at all.
- **The mechanism behind the new admissions is real**, not a correlation:
  the adjustment raises one segment's own `span_over_depth` from 0.057 to
  0.139, giving it a camera baseline the drifted chain had collapsed.
- **The branch REFUSES a pair the parent admitted.** `(29,30)`, admitted
  by the parent at reciprocity 0.967, measures reciprocity 2.80, scale
  ambiguity 4.15 and 13.5 px once the wider sample is used. That looks
  like a false merge the parent shipped and this branch catches.

What it found wrong, all of it fixed here or above:

| finding | what was wrong | disposition |
|---|---|---|
| the §6 numbers were stale | measured before the publishability re-check, on a build with points behind the wall | **re-measured**; `behind_camera` is now 0 on all five walks |
| the additive ladder | the two changes interact; the decomposition did not survive reordering | **replaced by the 2x2 in §6.2** |
| "cycles verified 0 → 5" | a cycle verifies only the segments it passes through, and two newly placed segments sat on unverified tree edges | **claim corrected** here and in `world_registration.py` |
| `match_budget`'s "SUPERSET" | `sampled_frames` spreads evenly, so a larger sample is not a superset | **docstring corrected** |
| `optimise` "re-anchors" | it does not; with no fixed camera, camera 0 moves | **claim removed** |
| the delta/snapshot comment | poses in the delta are stale too, not just points | **comment corrected**; no production consumer |
| the multi-reference table | confounded reference depth with the recovery budget | **numbers withdrawn**, argument kept |
| "the coherent PART never gets smaller" | false in points on two walks | **corrected in §6** |
| the drift generator | unrecorded, and the reviewer's guess walks into a wall | **recorded exactly in §4** |
| "contracts by a factor of three" | direction is walk-dependent; another step expands 15× | **corrected in §4** |
| a stale comment in `classical.py` | said there is no bundle adjustment | **corrected** |

Two of its cautions are not fixable here and are carried into §11
instead: reciprocity is self-consistency between two fits over one
reconstruction, so removing internal warp improves it whether or not a
pair is the same place; and `cycle_refusal_for` computes a translation
residual it never tests, so a loop can close in rotation and scale while
sitting a room's width out of position.

## 8. Determinism

The reconstruction is deterministic. Two runs of the shipped
configuration over the 2026-09-01 walk agree on every reported figure:
30 segments, 323 solved poses, 30,538 points, 10 fragments, a
12,558-point dominant component, reprojection median 0.5458 / p99 2.781
and 0.69% of rows over the gate. `cv2.setRNGSeed(0)` is set once at the
start of each run; `findEssentialMat(USAC_MAGSAC)` and
`solvePnPRansac(SQPNP)` are not individually seeded, so this is an
empirical result on this OpenCV build rather than a guarantee.

Registration is deterministic given a built world — six repeated runs,
three with a reseed and three without, produce the identical admitted
set. That is what made §6.1's isolation possible.

One caveat, found by the adversarial pass and worth carrying: OpenCV's
ORB/USAC output is **not bit-stable across processes** for every input.
One draft assertion in the safety suite passed in file order and failed
in isolation. Assertions in that file were rebuilt on quantities that are
stable cold and warm.

---

## 9. Performance

The bundle adjustment runs on the KEYFRAME path, never the frame path,
and its cost is bounded by the window rather than by the length of the
segment — `_local_adjust` compacts the landmark set to the ones the
window can see before calling the optimiser, which is what stops a
170-keyframe segment paying for its whole map on every adjustment.

Measured over the 2026-09-01 walk (2,613 frames, 434 keyframes), per
`observe()` call, replay on this host:

| | median | p95 | p99 | walk total |
|---|---|---|---|---|
| bundle off | 4.14 ms | 43.7 ms | 51.8 ms | 45.3 s |
| bundle on | **4.09 ms** | 47.8 ms | **128.3 ms** | 53.9 s |

The median is UNCHANGED -- the common path does not run the adjustment.
The p99 is the adjustment firing: it lands on one solved keyframe in
three, which is about one frame in twenty. Across the whole walk it costs
8.7 s over 2,613 frames, i.e. **3.3 ms per frame amortised and about
60 ms per adjustment**.

That is the shape the mission asked for -- an expensive step that is
bounded, exceptional, and off the frame path -- but the p99 is real and
the Tower's own per-frame budget is ~92 ms at the delivered 10.9 fps. On
a host where the common path already costs 96 ms, a 60 ms adjustment
every twentieth frame is the number to watch, and `BUNDLE_EVERY` is the
knob that trades it against drift.

Two honest notes. This host is not the Tower — the same walk reports
96.4 ms per frame in the physical record against ~4 ms here, so these
numbers compare configurations, not deployments. And the largest single
cost this change adds is not the bundle at all: registration went from
34.1 s to 65.7 s on that walk, which is §10.

---

## 10. Shutdown and finalisation

The risk flagged before this work is real, is now larger, and is not
fixed here.

`CaptureWorkerSupervisor.shutdown` gives each worker
`DEFAULT_GRACE_SECONDS = 10.0`. Registration measured 1.6–34 s on the
corpus before this change and 4.9–66 s after. Every substantial walk
exceeds the grace.

What a kill costs, precisely: `register()` writes nothing while it runs.
`write_placements` is its only write and happens once, at the end, after
`placements_from_report` has validated every transform. So a worker
killed mid-registration loses `placements.json` and **never** the
reconstruction — the world is left exactly as the physical era left it,
built and unplaced. That is a degraded outcome, not data loss.

Why it is not fixed here rather than merely deferred: on Windows
`Popen.terminate()` is `TerminateProcess`, which the child cannot catch,
so no handler or checkpoint inside `world_build_session.py` can help. The
available fixes are a longer grace for this worker specifically, or
writing placements incrementally. The first is a product decision about
how long a Stop may take, and the second is a change to a persisted
contract the iOS side reads. Neither belongs in a tracking change.

---

## 10.5 Tests, and what the mutations prove

Full Tower suite, three runs on this branch: **2,273 passed, 72 skipped,
0 failed** on the last and cleanest, and one failure on each of two
earlier runs —
`test_result_channel_hostile.py::test_the_channel_survives_the_world_vanishing_mid_subscription`,
which also passes in isolation and in a 324-test selection.

So the suite is INTERMITTENT under a full run rather than broken, and the
intermittency is not this branch's. The failing test is in the result
channel, a subsystem untouched here, and **the parent branch's full suite
is not clean either** — it failed
`test_object_memory_lifecycle.py::test_an_unconfigured_tower_still_serves_its_own_memory`,
a different unrelated test, on the same machine in the same session. The
honest statement is that a full run passes and occasionally does not, for
reasons that pre-date this work and sit outside the World Builder.

World-builder selection: **628 passed, 14 skipped, 0 failed**, on every
run.

New test files:

- `test_world_builder_bundle.py` (12) — the optimiser against geometry it
  cannot see. Nothing checks that the cost went down; every test compares
  against a rig whose answer is known independently.
- `test_world_builder_tracking_recovery.py` (7) — the recovery mechanism,
  exercised at a raised budget so the tests pin the MECHANISM and not the
  policy, plus one test pinning the shipped policy at 1 with its reason.
- `test_world_builder_recovery_safety.py` (14) — the adversarial suite,
  which is the evidence for the bound.

**Mutation testing.** Every protection was removed and a test had to
notice:

| mutation | caught by |
|---|---|
| `BUNDLE_WINDOW = 0` (drift control off) | 3 drift tests, including both shape assertions |
| `MAX_RECOVERY_KEYFRAMES = 1` applied to the mechanism tests | 3 recovery tests |
| `MIN_VIEWS_FOR_ADJUSTMENT = 3` | `test_two_view_landmarks_move_with_their_cameras` |
| `fixed_points` ignored | `test_a_landmark_the_window_cannot_hold_is_not_moved` |
| refused keyframes promoted to references | recovery collapses 8 solved to 0 |

And the mutation that mattered most, because it was a finding rather than
a confirmation: with the threshold-equality assertion deselected,
**mutating `PNP_REPROJECTION_ERROR_PX` from 3.0 to 30.0 broke no test in
the pre-existing suite**, nor did loosening `MIN_INLIERS`,
`MIN_TRIANGULATION_ANGLE_DEG` or `MIN_INLIER_RATIO`. Every acceptance
threshold was guarded only by an assertion that the constant equals a
literal — including the test whose own docstring called itself the guard
against "recovery implemented by lowering a threshold", whose frames
never seed a map so no threshold could matter. The new safety suite fails
5 tests under the 30 px mutation.


## 11. Remaining limitations

1. **`MAX_RECOVERY_KEYFRAMES` is 1 because nothing can check a wider
   gap.** The mechanism is built, tested and shipped inert. What unlocks
   it is an instrument that compares the displacement a recovered pose
   implies against the number of keyframes it skipped — not a better
   matcher, because appearance is precisely what lies over repeating
   texture.

2. **The dominant component is 74.9% of the 2026-09-01 walk, not 100%.**
   That walk is now one joined component plus eight orphan fragments.

3. **THE 12 UNRECONSTRUCTED AREAS ARE UNCHANGED.** The phone reported 12
   areas seen but not reconstructed. That is exactly the 12 of 30
   segments which received keyframes and triangulated nothing, and it is
   **still 12** on this branch: 30 segments, 18 with geometry, before and
   after. Bundle adjustment cannot help a segment that produced no
   geometry to adjust.

   What would help is the seed-anchor retry — a segment whose seed pair
   refuses currently dies, where holding the anchor and retrying on the
   next keyframe gives the pair a WIDER baseline, which is the direction
   that fixes a parallax refusal. That mechanism is built and tested. It
   is inert at `MAX_RECOVERY_KEYFRAMES = 1`, and at 8 it takes the same
   walk from 12 unreconstructed segments to 8 — which is exactly the
   trade §3.3 refuses, because the same budget is what lets a solve
   publish a metre and a half of walking as a millimetre.

   **This is the clearest thing the missing instrument would buy.** A
   displacement-consistency check would let the seed retry run without
   the risk that currently rides with it.

4. **Registration's second wall is untouched.** Only 2–12% of verified
   cross-segment correspondences name a feature the source segment
   triangulated into a point, so even with the full cross-product the
   biggest segments cannot reach `MIN_PNP_CORRESPONDENCES` on enough
   target frames. Widening retrieval does not reach this.

5. **`span_over_depth` has no upper bound**, so a diverged segment
   reports "plenty of parallax" rather than "this solve blew up". The
   drift control makes that far less reachable; it does not make it
   detectable.

6. **`span_over_depth` also mismeasures its most common refusal.** All 27
   two-pose segments in the corpus have a span numerator of exactly
   1.0000 — the seed pair's normalised baseline — so the ratio reports
   scene depth, not wearer motion, and "the wearer stood still" is
   factually wrong for 18 of the 32 segments it is printed about.

7. **No loop closure inside a segment.** A revisit within one segment
   becomes drift the bundle window cannot see across; only a revisit that
   crosses a segment boundary can be closed, and only by registration.

8. **Reciprocity is not identity evidence.** `admit()`'s strongest clause
   is the agreement of two independent Sim3 fits, but both are computed
   over ONE reconstruction, so removing that reconstruction's internal
   warp improves the agreement whether or not the two segments are the
   same place. The independent evidence is the cycle check, and a cycle
   verifies only the segments it passes through — on the 2026-09-01 walk
   some newly placed segments sit on tree edges no cycle touches.

9. **`cycle_refusal_for` computes a translation residual and never tests
   it.** It gates rotation and scale only. On this corpus the
   translation residuals are 0.6–1.7% of the placed cloud's diagonal so
   it does not fire, but a loop can close in rotation and scale while
   sitting a room's width out of place and nothing would notice.

10. **Every drift and safety number here is SYNTHETIC.** Rendered rooms
   with perfect optics, no rolling shutter and no compression say nothing
   about the Ray-Ban camera. Their value is that the poses are inputs, so
   the answers are exact rather than plausible. The corpus numbers are
   physical; the causal ones are not.
