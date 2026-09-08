# World Builder dense reconstruction — established facts

- Lane branch: `world-builder/dense-reconstruction-v1`
- Worktree: `C:\Users\tvllo\Projects\Glasses-worktrees\wb-dense`
- Scratch: `C:\Users\tvllo\Projects\Glasses-scratch\wb-dense`
- Branched from `integration/all-cartridges-v1` @ `9e939a3`
- Date: 2026-09-08

Everything here was measured on this machine. Where a figure came from a
subagent, the report it came from is named.

> **Sections 1-10 are a build log and their numbers are dated.** They record
> what was measured while the stage was being designed, under whatever
> configuration was current at the time -- including a different depth network.
> **Section 11 is the shipped configuration, re-measured on all seven worlds
> after the branch stopped moving.** Quote section 11. Where an earlier section
> conflicts with it, section 11 is right and the earlier one is kept because the
> reasoning it supports is still the reasoning that was used.

---

## 1. The baseline, and the two separate reasons it is not recognizable

`scripts/world_inspect.py` on the reference world `7d31e8d7acde46808b7a31f1b7bc211e`:

| quantity | value |
| --- | --- |
| frames observed | 1371 |
| keyframes accepted | 438 |
| pose status | 395 solved, 34 anchor, 9 unavailable |
| sparse points | 14,953 |
| segments | 34 |
| scale state | `unknown` — no unit at all |

`scripts/world_render.py` draws those points **coloured by segment index**. Two
distinct problems are visible in that render and they should not be conflated:

1. **Density.** 14.9k points recovered from 436 posed 359x639 views. Those views
   hold roughly 98 million pixel observations, so the sparse reconstruction
   retains about **0.015%** of the available visual evidence. That is the
   headroom the mission asked about.
2. **Colour.** `solution.npz` already carries a per-point `rgb` array. The
   renderer ignores it, and the contract admits it: `WORLD-BUILDER-GEOMETRY.md`
   says colour is *"not yet carried on the chunk"*. Re-rendering the same 14.9k
   points with their true RGB already reveals planar structure that the
   segment-coloured render destroys.

So the sparse appearance was never evidence that the input data is thin.

## 2. The global solve is a sound backbone — keep it

`solve/<session>/solution.json` reports `solver: "glomap"`, 438 keyframes, 436
poses, **2 components** (429 and 7). The 34-segment mosaic under `derived/` is
the older fragment-registration path; the global solve is a single
reconstruction, and it is good:

| quantity | value |
| --- | --- |
| per-frame median reprojection error | **0.753 px** (p90 1.006, worst 1.859) |
| observations | 184,982 |
| observations per frame, median | 405 |
| track length | median 5, p90 24, max 298 |
| camera-centre extent | 10.2 x 8.4 x 13.9 world units |

There is no reason to replace this. The dense work builds on top of it.

## 3. Pose convention — verified, and NOT what `world.json` declares

`world.json` declares `pose_type: "T_world_camera"` and says the stored
translation is the camera centre. That describes the **derived** tree.
`solve/solution.json` uses the opposite convention, which matters enormously to
anything that back-projects pixels.

Tested by projecting the solved 3-D points into each frame and comparing against
the stored `observation_xy`:

| candidate | median error |
| --- | --- |
| `R @ X + t` | **0.615 px** |
| `R @ (X - t)` | 110.7 px |
| `R.T @ X + t` | 169.9 px |
| `R.T @ (X - t)` | 262.3 px |

So in `solution.json`, `rotation` is `R_camera_world` and `translation` is
COLMAP's `tvec` — exactly COLMAP's convention. Camera centre is `C = -R.T @ t`.
This is codified in `proto/solveio.py`, whose `unproject` then `project` round
trip is exact to 0.00000 px.

**This discrepancy between the two artifacts is a documentation bug worth
fixing**, independent of the dense work.

## 4. Calibration and resolution — the hard ceiling

There is a genuine self-calibration, not an assumption:

```
model pinhole_radtan
fx 438.225  fy 437.778  cx 174.877  cy 323.380
dist [0.14395, -0.92780, 0.00151, 0.00233, 1.29980]
calibrated 360x640, reprojection RMS 0.289 px over 511 views
```

`cv2.getOptimalNewCameraMatrix(alpha=0)` plus the ROI crop reproduces the
solver's camera **exactly**: PINHOLE 359x639, fx 465.71872, fy 465.05395,
cx 176.43702, cy 322.13759, ROI offset (0, 0).

**Every frame in the corpus is 360x640 portrait — 0.23 MP.** All 97 captures,
45,594 frames, one JPEG quantization table (report 01). This is the native
capture, not a downscaled working copy. It fixes the honest resolution ceiling
at roughly 6.4 mm per pixel at 3 m, with 1–3 cm depth noise (report 06).
Therefore **2–3 cm voxels are truthful and 5 mm voxels would be fabricated**.

## 5. Environment — three blockers found, two fixed

| finding | status |
| --- | --- |
| `import cv2` raised `missing configuration file: config.py` | **FIXED.** `config.py` and `config-3.py` had been deleted from `site-packages/cv2/`. Their exact source was recovered from the surviving `__pycache__` bytecode and rewritten. cv2 5.0.0 now imports. 27 modules depend on it, including the World Builder pipeline. |
| seven declared dependencies missing: `certifi`, `anyio`, `click`, `colorama`, `contourpy`, `annotated-types`, `annotated-doc` | **FIXED** by `pip install`. `pip check` is now clean. Tower could not have started before this: `fastapi` needs `annotated-doc` and `uvicorn` needs `click`. |
| `pycolmap.has_cuda` is `False` | **REAL.** Calling `patch_match_stereo` raises *"Dense stereo reconstruction requires CUDA or HIP"*. The pycolmap Windows wheel has no CUDA and upstream says CUDA wheels are Linux-only. The fix is the prebuilt `colmap-x64-windows-cuda.zip` 4.2.0 binary, which statically links cudart and needs no toolkit (report 06). |

GPU works: RTX 5070, compute capability **(12, 0)**, and this torch build lists
`sm_120` in `get_arch_list()`. But there is **no MSVC and no CUDA toolkit newer
than 11.8**, so *compiling* a CUDA extension is impossible today (report 05).
That constraint, rather than model quality, eliminates several strong options.

## 6. Licence constraints — several obvious choices are unusable

This is a product, so licences are load-bearing:

- **Depth Anything V2 Large and Base are CC-BY-NC-4.0.** The first prototype was
  built on Large and must be replaced before shipping. V2 **Small is Apache-2.0**.
- Original INRIA 3DGS and the forks Mip-Splatting, 2DGS, GOF, RaDe-GS,
  3DGS-MCMC, Scaffold-GS, Octree-GS, AbsGS and Pixel-GS are **all
  non-commercial**. `gsplat` (Apache-2.0) reimplements most of them as flags.
- OpenMVS is **AGPL-3.0**, which matters for a networked Tower.
- VGGT-Omega is FAIR Noncommercial; Pi3 weights are CC-BY-NC; Depth Anything 3's
  *default* checkpoint is CC-BY-NC while BASE and SMALL are permissive.
- Clean: `gsplat`, Brush (Apache-2.0), MoGe (MIT), Metric3D v2 (BSD-2),
  LingBot-Map (Apache-2.0), COLMAP (BSD).

## 7. What the corpus actually contains

From report 01 (all 97 captures) and report 03 (all 162 worlds):

- 97 captures, 45,594 frames, 0.95 GiB, 80.6 minutes of wall clock over 7 days.
- Frame integrity is perfect: journal record count equals JPEG count for all 97.
- **No intrinsics, no IMU, no orientation, no depth and no GPS in the captures.**
  Self-calibration is mandatory. Timestamps are Tower receipt time, not sensor
  time.
- Of 162 worlds, **only 7 hold a real global solve**. 96 are empty shells and 41
  are incremental-only builds with `placed: 0`.
- The corpus is dominated by a **seated first-person view of a phone and a
  laptop screen**, across two environments. Only about 8,000–12,000 frames in
  8–10 captures carry real translation, and none of those captures had been
  solved before this lane started.
- Scale is arbitrary, and **the gauge is not normalised at all**. An early
  reading attributed the reference world's tidy 10 x 8 x 14 extent to COLMAP's
  `Normalize()`. That was wrong: `global_solve.py` never calls it. Solving eight
  further captures produced gauges from a ten-unit extent to a **340 x 70 x 175**
  one for the same kind of walk, so the reference world's tidy numbers were luck.
  Two consequences: **use component 0 only**, since components are solved
  independently and share no unit; and express **every** length as a fraction of
  that component's own median scene depth, never in absolute world units.

## 8. iOS reality — the viewer is ours to change

From report 07:

- There is **no `Codable` conformance anywhere** in the iOS target. Every World
  Builder payload is parsed with `JSONSerialization` into `[String: Any]` and
  read key by key, so unknown keys are ignored. Checked-in fixtures carrying
  three unread keys prove it. **Additive `world.json` fields are safe.**
- What does break it: bumping the `contract` identifier; re-typing an existing
  field, since `json["points"] as? [[Double]]` is a whole-array cast so a single
  `null` refuses the entire chunk, meaning colour must arrive as a **sibling**
  key; or removing any guarded field.
- **iOS renders 2-D dots.** No SceneKit, RealityKit, Metal or ARKit anywhere.
  The native gallery draws 2x2-pt ellipses at top-down `(x, z)`, discarding `y`.
- **The saved-world viewer is a `WKWebView` over HTML returned by
  `GET /worlds/{id}/render`.** Tower therefore owns the viewer, and a real 3-D
  renderer needs no Swift change. The route's CSP forbids every external
  resource, so the page must be entirely inline.
- `up_axis: "unknown"` is the stated reason the viewer is 2-D. Changing its
  value is safe; removing the key is not.

## 9. Architecture facts that constrain where a dense stage can live

From report 02:

- World Builder is **not an in-process cartridge**. It is a supervised child
  process (`scripts/world_build_session.py`) that tails a capture directory and
  itself spawns a grandchild for the global solve. `cartridge_runtime.py`
  contains zero references to `world_builder`.
- **`build()` is not Stop-only — it runs every 4 keyframes.** A dense stage must
  not hook there.
- The hook point is `scripts/world_build_session.py`, after the final build.
  **This bullet used to say the writer lock is held and `finalization` is still
  `pending`, and that describes a design that was not built.** What shipped
  runs densify AFTER the `finally:` block that marks finalization complete and
  releases the world -- deliberately, because holding the writer lock for the
  minutes this takes would block a new capture on the same world. For the whole
  dense run the world reports `ready` and the session `complete`, and
  `dense/status.json` is the only record that anything is still running.
- **`stop_grace_seconds = 30.0`** plus a Windows Job Object kills the whole
  process tree. A minutes-long dense job needs a raised grace, checkpointing, or
  an explicit skip-on-hard-stop.
- **There is no migration machinery**: `require_schema` refuses any version but
  1. A dense artifact must therefore be **additive and optional**, following the
  existing `support.json` and `placements.json` precedent.

---

## 10. Corrections and later findings

Recorded separately because each overturned something believed earlier in this
lane, and in every case the wrong version was the more tempting one.

### 10.1 The gauge is not normalised

See section 7. Believed normalised to extent 10; it is not normalised at all.

### 10.2 Most of the apparent "fusion error" was the evaluation renderer

Held-out rendered depth from the fused cloud missed the sparse points by 6.3%,
against 3.0% for the per-frame depth feeding it. That looked like fusion
doubling the error. It was not:

| what was measured | signed relative error |
| --- | --- |
| per-frame aligned depth, 345 frames | **+0.03% median** (mean -0.45%, 45% near) |
| fused cloud rendered with 0.5 px splats | -2.9% |
| fused cloud rendered with 1.0 px splats | -5.6% |
| fused cloud rendered with 1.6 px splats | -6.0% |
| fused cloud rendered with 2.5 px splats | -6.4% |

The per-frame depth is **unbiased**. The bias is a function of splat radius, and
its sign is toward the camera in 100% of frames, because a fat splat spills onto
neighbouring pixels and a z-buffer keeps the nearest of them, which on any
slanted surface is nearer than the truth. The geometry is about as good as the
depth maps feeding it. The evaluator now measures depth from a separate
near-point render, while appearance keeps the fat splat it needs for coverage.

The lesson worth keeping: the evaluation renderer is part of the measurement
apparatus, and it can be the thing that is wrong.

### 10.3 Averaging agreeing cameras genuinely helps

On identical held-out views:

| | reference camera only | averaged over agreeing cameras |
| --- | --- | --- |
| rendered depth error, median | 7.42% | **6.32%** |
| rendered depth error, p90 | 10.24% | **8.23%** |
| PSNR | 14.87 | **15.22** |
| voxels from the same 35.2 M points | 10.63 M | **8.73 M** |

18% of the points were redundant surface thickness. Fewer points AND better
depth is the signature of a real improvement rather than a trade.

### 10.4 The face redactor fires on hands, and it is expensive here

`redaction.py` runs YuNet at confidence 0.30, a documented compromise: below 0.2
it fires on face-free frames, above 0.4 it misses small faces and faces on
screens. On this corpus that compromise is costly. Over 77 frames of one
capture, the filled area was:

| | fraction of frame filled |
| --- | --- |
| median | 0.4% |
| mean | 8.6% |
| p90 | 33.0% |
| worst | 57.8% |

22 of 77 frames lose more than a tenth of the image, and inspection shows the
boxes sitting on **the wearer's hands and on carpet**, not on faces.

This is pre-existing rather than caused by the dense work: the same fill is
baked into the keyframe images the sparse solve already used, and an earlier
handoff measured it costing about 100 solved poses. But it hurts a dense stage
far more, and the damage is not confined to the filled pixels:

| redaction fill | frames | held-out residual, median | pass the 8% gate |
| --- | --- | --- | --- |
| under 1% | 37 | 6.9% | 65% |
| 1-10% | 16 | 6.1% | 62% |
| 10-30% | 13 | 10.3% | 46% |
| over 30% | 7 | 34.8% | **0%** |

A solid black rectangle drags the network's depth estimate for the whole frame,
not only its own region. So the dense stage does two separate things: it
**inpaints** the fill before the network sees it, to protect the rest of the
frame, and it still **masks the fill out** of the reconstruction afterwards,
because whatever the network puts there is invention. The region stays a hole.

Worth doing properly later: have the redactor persist its boxes alongside the
keyframe, so consumers read what was removed instead of re-deriving it from the
pixels.

### 10.5 The solver's own convention documentation is correct

Section 3 says `world.json`'s `pose_convention` block does not describe
`solve/solution.json`. That stands, but `global_solve.py` itself is explicit and
right: it annotates the field as `R_cw` and computes `centre = -r_wc @ t_cw`.
The mismatch is between `world.json`, which describes the derived tree, and the
solve artifact. It is a documentation gap, not a solver bug.


---

## 11. The shipped configuration, measured on all seven worlds

**Read this section before quoting any number from this lane.** Everything above
it was measured while the stage was being built, on whatever configuration was
current that hour, and the default depth network changed late. An adversarial
review made that its blocking finding, and it was right: a table describing a
model the product no longer runs is worse than no table.

So every world in the corpus was densified again, from scratch, with the
configuration that is actually on the branch -- MoGe-2 ViT-L, `tau` 0.05, the
exact redaction fill mask, bounded extrapolation, `gate_rel` 0.08 -- and scored
the same way each time. These are those numbers.

### 11.1 Per world

| world | what it is | posed | used | held-out residual | L0 points | L0 size | wall clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `7d31e8d7` | desk and shelf | 429 | 316 | 2.6% | 8.22 M | 132 MB | 230 s |
| `1b8812b1` | widest traverse | 438 | 303 | 2.9% | 8.65 M | 139 MB | 206 s |
| `37e497f8` | bedroom | 196 | 134 | 3.4% | 4.85 M | 78 MB | 96 s |
| `672578d0` | bedroom, closet, desk | 425 | 298 | 3.8% | 14.75 M | 236 MB | 210 s |
| `a378331a` | closet | 201 | 117 | 4.9% | 5.05 M | 81 MB | 92 s |
| `ecc02df1` | dresser | 77 | 50 | 5.3% | 2.09 M | 33 MB | 51 s |
| `6427900d` | bathroom, tight | 266 | 132 | 5.4% | 3.57 M | 57 MB | 108 s |

Peak VRAM for the depth stage is **2.4 GB**, up from 0.85 GB, and that is the
price of the model change. Everything else is CPU and RAM.

### 11.2 The fused cloud, rendered at held-out cameras

Not the per-frame residual above. This is the whole cloud re-rendered from
cameras the fusion did not privilege, and depth read out of the render.

**Every row is an `eval.json` beside the artifact it describes**, written by
`proto/score_cloud.py --out`. An adversarial review found that this table's
first version cited nothing on disk: the numbers were real but had only ever
been printed to a terminal, and a number nobody can re-derive is not evidence.

| world | points | depth error, median | depth error, p90 | pixel coverage, median |
| --- | --- | --- | --- | --- |
| `7d31e8d7` (desk and shelf) | 8.32 M | 2.3% | 12.7% | 97.9% |
| `ecc02df1` (dresser) | 2.09 M | 3.3% | 5.4% | 97.0% |
| `1b8812b1` (widest traverse) | 8.76 M | 3.3% | 6.6% | 96.8% |
| `fc58a64d` (end-to-end replay) | 8.43 M | 4.2% | 6.0% | 82.1% |
| `6427900d` (bathroom, tight) | 3.62 M | 4.4% | **57.6%** | 67.5% |
| `a378331a` (closet) | 5.11 M | 4.6% | 5.6% | 60.1% |
| `672578d0` (bedroom, closet, desk) | 14.93 M | 5.3% | **40.3%** | 97.9% |
| `37e497f8` (bedroom) | 4.91 M | 5.4% | 7.2% | 96.2% |

**The p90 column is the honest part of this table.** Two worlds carry a tail an
order of magnitude worse than their own median. `6427900d` is the tight
bathroom, where the walk never gets far enough from a surface for two cameras
to disagree usefully, and `672578d0` is the three-room chain, where the far end
of a long room is reconstructed from a handful of distant frames. In both, the
median says the reconstruction is good and the p90 says part of it is not, and
the confidence channel is what a viewer has to separate them with.

Coverage tells the same story from the other side: `a378331a` (closet) and
`6427900d` (bathroom) cover 60-67% of the held-out frame where every other
world covers 82-98%. Tight spaces are this pipeline's weakest case, and they
are weakest for a structural reason rather than a tuning one -- multi-view
consensus needs baseline, and a closet does not offer any.

### 11.3 What the gate does to the number

The held-out residual quoted in 11.1 is the median **of the frames that passed
the 8% gate**. It is therefore a property of the gate as much as of the
reconstruction, and it improves as the gate tightens while the reconstruction
gets worse. Both ends, from `proto/gate_sensitivity.py` over the shipped
artifacts (bracketed count is frames surviving that gate):

| world | posed | all frames | gate 0.16 | **gate 0.08 (shipped)** | gate 0.04 | gate 0.02 |
|---|---|---|---|---|---|---|
| `7d31e8d7` | 429 | 2.6% (399) | 2.3% (342) | **2.2% (317)** | 1.9% (262) | 1.3% (139) |
| `1b8812b1` | 438 | 2.9% (373) | 2.5% (330) | **2.4% (303)** | 2.0% (227) | 1.4% (114) |
| `37e497f8` | 196 | 3.4% (170) | 3.2% (154) | **3.0% (134)** | 2.6% (99) | 1.7% (29) |
| `672578d0` | 425 | 3.8% (374) | 3.5% (337) | **3.2% (300)** | 2.8% (201) | 1.7% (28) |
| `a378331a` | 201 | 4.9% (153) | 4.7% (139) | **4.4% (117)** | 3.3% (49) | 1.6% (1) |
| `ecc02df1` | 77 | 5.3% (74) | 4.2% (61) | **3.7% (50)** | 2.8% (27) | 1.8% (2) |
| `6427900d` | 266 | 5.4% (215) | 4.8% (179) | **3.8% (132)** | 2.5% (69) | 1.2% (22) |

Tighten to 2% and the "accuracy" becomes 1.2-1.8% while `a378331a` keeps ONE
frame. So: quote the all-frames column when describing the pipeline, quote a
gate column only when comparing two configurations at the same gate, and never
quote the best world as the pipeline's figure. Under the shipped model the gate
moves the number by 0.2-1.6 points depending on the world; under the previous
one it moved it by up to 3.4, which is most of what the earlier tables were
reporting as quality.


### 11.5 Standing where the wearer stood does not always show what they saw

An independent visual reviewer, given only the comparison sheets and no
engineering context, found columns where the reconstruction rendered from a
capture pose shows **a different part of the dwelling** than the photograph
taken from that pose. That is the most serious thing anyone has said about this
artifact, so it was measured rather than argued about.

`proto/seethrough.py` separates the two possible causes on 45 sampled poses
across six worlds. The discriminator is the sparse cloud: it comes from the same
bundle adjustment as the poses and is independent of the depth network.

| | median | p90 | worst |
| --- | --- | --- | --- |
| reprojection error of this frame's own observed sparse points | **0.76 px** | — | **1.66 px** |
| pixel coverage of the render | 93.5% | — | 26.5% (min) |
| sparse points the render places at least 25% too far away | 1.5% | 11.1% | 95.4% |
| relative depth error where the render shows anything | 5.6% | 83.6% | — |

**The poses are right.** Sub-pixel reprojection on every frame sampled, worst
1.66 px. Nothing is misplaced and nothing is in the wrong room.

**What happens instead is that you see through a hole.** A point cloud occludes
only where it has points. Where the near surface was dropped — a blank wall
carries almost no sparse points, so the affine fit there is unanchored and the
gate rejects the frame, or the validity mask removes it — there is nothing in
front, and the render shows the geometry BEHIND it. On the worst frame sampled,
95.4% of that frame's own sparse points are rendered at least a quarter too far
away: the wall is simply absent and the room behind it is what appears.

**6 of 45 sampled poses (13%) do this badly enough to be misleading.** That is
the honest number. It is not fabrication — every point shown is a real
observation of a real surface, just not the surface that should be in front of
it — but a wearer cannot tell the difference, and "an empty region means the
observations did not support geometry there" is a weaker promise than it sounds
when the empty region is a hole in a wall you are looking through.

**Nothing available fixes this within the artifact's own rules.** Filling the
hole is exactly the fabrication the format forbids. Closing it honestly needs
either more frames through the gate or a surface representation, and D13 records
why the surface options were rejected. What can be done is to say so, which is
what this section is for, and the diagnostic is in the tree so the next person
can re-measure rather than re-argue.


### 11.6 The walls are there, and they are not invented

The same independent visual reviewer made two claims that cannot both be true
of the same surfaces, and measuring them settles both.

> *"No world has a ceiling, a complete wall, a corner, or a closed floor plan.
> Not one."*

> *"The giant smooth sheets ... huge, smoothly-curved cream and white planes
> that are several times larger than any wall in any photograph, and they curve
> -- real walls do not. This looks like depth-map extrapolation off a blown-out
> overexposed wall, spraying a plausible-looking smooth surface into space that
> was never observed."*

`proto/planarity.py` RANSACs the largest planar structures out of the fused
cloud and reports, for each, how thick it really is and what confidence its
points carry. Confidence is the number of independent cameras that agreed on a
point's depth, which is the discriminator: **a fabricated surface cannot carry a
high one, because the mechanism that would have to invent it is the same one
that counts agreements.**

The six largest planes, on the two worlds the claims were made about:

| world | plane share of cloud | thickness, RMS / extent | thickness, p95 / extent | width / extent | confidence, median | share at 5+ cameras |
| --- | --- | --- | --- | --- | --- | --- |
| `672578d0` (three rooms) | 30.0% in six planes | 0.0011-0.0012 | 0.0035-0.0038 | 0.18-0.50 | 6-8 | 72-92% |
| `7d31e8d7` (desk) | 32.1% in six planes | 0.0011-0.0012 | 0.0037-0.0038 | 0.23-0.36 | 6-8 | 67-87% |

Against a whole-cloud confidence median of 6 with 16% of points sitting at the
floor of 3.

**So: about a third of each cloud lies in six structures that are between a
fifth and a half of the room across, flat to about one part in a thousand of
the scene, and supported by more cameras than the average point in the same
cloud.** Those are walls, floors and ceilings. They are not curved, they are not
extrapolated, and they are better evidenced than the furniture.

The first claim is therefore wrong, and the second claim is wrong about the same
surfaces the first claim says are missing. What is true is the thing underneath
both: **the enclosure is incomplete.** Planes exist but do not close; corners
are frequently absent; and a room with three of its four walls reads to a viewer
as no walls at all, especially from outside, where you see the backs of them.

A separate measurement rules out the obvious explanation. On 30 frames of the
desk world, wall-facing pixels survive the validity mask at **78.8%**, against
80.0% for all pixels -- the mask is not what removes walls. Floor and ceiling
pixels do worse, at 70.9%, losing 12.8% to the grazing-angle test, which is
exactly what a floor seen from standing height should lose. The gate refused 4
of 429 frames at alignment. Neither mechanism explains an incomplete enclosure;
what does is that a wearer walking through a room does not point the camera at
every wall from two angles, and consensus needs two angles.

### 11.4 What changed against the previous default, and what did not

The seven-world run above is the same seven worlds the stage had already
produced with Depth Anything V2 Small, so the comparison is like for like.

| | V2-Small | MoGe-2 ViT-L |
| --- | --- | --- |
| held-out residual, best world | 3.0% | 2.6% |
| held-out residual, worst world | 8.7% | 5.4% |
| frames passing the gate, `1b8812b1` | 247 / 438 | 303 / 438 |
| frames passing the gate, `672578d0` | 218 / 425 | 298 / 425 |
| L0 points, `1b8812b1` | 5.80 M | 8.65 M |
| peak VRAM | 0.85 GB | 2.4 GB |
| wall clock, `1b8812b1` | 7 min | 3.4 min |

More frames survive, so more of the room is reconstructed, and the frames that
survive are better aligned. The wall clock fell despite a 45x slower network
because the earlier figure was measured on a contended GPU; treat both as
upper bounds.

**What did not change: the gate still drops 30-50% of posed frames.** 303 of
438, 132 of 266, 50 of 77. That is disclosed in `frames_dropped`, in the
manifest and in the CLI's own output, and it is the number to quote when
someone asks how much of the walk the reconstruction uses. It is better than
the 43-64% the previous model dropped, and it is still most of a third.
