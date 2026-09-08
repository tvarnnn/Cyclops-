# World Builder dense reconstruction — established facts

- Lane branch: `world-builder/dense-reconstruction-v1`
- Worktree: `C:\Users\tvllo\Projects\Glasses-worktrees\wb-dense`
- Scratch: `C:\Users\tvllo\Projects\Glasses-scratch\wb-dense`
- Branched from `integration/all-cartridges-v1` @ `9e939a3`
- Date: 2026-09-08

Everything here was measured on this machine. Where a figure came from a
subagent, the report it came from is named.

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
- The hook point is `scripts/world_build_session.py`, between the final solve and
  the final build: the writer lock is held and `finalization` is already
  `pending` on disk.
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
