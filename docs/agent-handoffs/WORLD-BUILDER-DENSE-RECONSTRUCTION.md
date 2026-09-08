# World Builder — dense reconstruction handoff

**Autonomous development record, 2026-09-08.**
Companion documents: `docs/world-builder-dense/01-EVIDENCE.md` (what was
measured), `02-ARCHITECTURE.md` (the design), `03-DECISIONS.md` (what was
decided and against what), `04-OPERATIONS.md` (how to run it), and
`docs/contracts/WORLD-BUILDER-DENSE.md` (the artifact and its wire format).

---

## 1. Starting state

| | |
|---|---|
| Branch | `world-builder/dense-reconstruction-v1`, off `integration/all-cartridges-v1` @ `9e939a3` |
| Worktree | `C:\Users\tvllo\Projects\Glasses-worktrees\wb-dense` |
| Scratch | `C:\Users\tvllo\Projects\Glasses-scratch\wb-dense` |

The sparse reconstruction worked and was not the product. `world_inspect.py` on
the reference world reported 1371 frames observed, 438 keyframes, 395 solved
poses, **14,953 sparse points**, scale `unknown`. Rendering it drew those points
**coloured by segment index**, which is why nothing was recognizable.

Two separate problems were hiding in that one picture, and it matters that they
are separate:

- **Density.** 436 posed 359x639 views hold roughly 98 million pixel
  observations. The sparse cloud retains about **0.015%** of them.
- **Colour.** `solution.npz` already carried per-point RGB. The renderer ignored
  it and the contract admitted as much. Re-rendering the same 14,953 points in
  their true colours already showed planar structure the segment colouring
  destroyed.

So the sparse appearance was never evidence that the data was thin.

## 2. Three environment blockers, found before anything could run

| finding | status |
|---|---|
| `import cv2` raised *"missing configuration file: config.py"* | **Fixed.** `config.py` and `config-3.py` had been deleted from `site-packages/cv2/`. Their exact source was recovered from the surviving `__pycache__` bytecode and rewritten. 27 modules import cv2, including the World Builder pipeline. |
| Seven declared dependencies absent (`certifi`, `anyio`, `click`, `colorama`, `contourpy`, `annotated-types`, `annotated-doc`) | **Fixed** by install; `pip check` is clean. Tower could not have started: `fastapi` needs `annotated-doc`, `uvicorn` needs `click`. |
| `pycolmap.has_cuda` is `False` | **Real, and shaped the architecture.** `patch_match_stereo` raises *"Dense stereo reconstruction requires CUDA or HIP"*. Note that `hasattr(pycolmap, "patch_match_stereo")` is `True`, so the obvious probe gives the wrong answer. |

The GPU itself is fine: RTX 5070, compute capability (12, 0), and this torch
build lists `sm_120`. But there is **no MSVC and no CUDA toolkit past 11.8**, so
compiling any CUDA extension is impossible on this machine today. That
constraint, not model quality, eliminated several otherwise-strong options.

## 3. Datasets

**The corpus.** 97 captures, 45,594 frames, 0.95 GiB, 80.6 minutes over seven
days. Frame integrity is perfect. **Every frame is 360x640 portrait — 0.23 MP,
natively.** There are no intrinsics, no IMU, no orientation, no depth and no GPS
in the captures; a real ChArUco self-calibration exists separately
(`pinhole_radtan`, 0.289 px RMS over 511 views).

Of 162 saved worlds, **only 7 held a real global solve**, and they were mostly
seated desk scenes. So twelve further solves were produced from historical
captures — no new physical capture was needed at any point in this work.

**Densified and evaluated (8 worlds, materially different environments):**

| world | environment | posed | frames used | held-out align residual | L0 points |
|---|---|---|---|---|---|
| `7d31e8d7` | desk and shelf | 429 | 345 | 3.0% | 11.9 M |
| `1b8812b1` | widest traverse | 438 | 247 | 5.0% | 5.8 M |
| `672578d0` | bedroom, closet, desk | 425 | 218 | 6.7% | 5.7 M |
| `a378331a` | closet walk | 201 | 91 | 6.9% | 2.3 M |
| `37e497f8` | bedroom walk | 196 | 92 | 7.0% | 2.1 M |
| `6427900d` | bathroom, tight | 266 | 97 | 8.3% | 1.0 M |
| `ecc02df1` | dresser | 77 | 40 | 7.6% | 1.0 M |
| `be36bd70` | end-to-end replay | 58 | 26 | 8.7% | 0.7 M |

## 4. Data-quality findings that constrain the result

- **0.23 MP is the ceiling.** About 6.4 mm per pixel at 3 m, 1-3 cm of depth
  noise. Voxels finer than ~2 cm store noise rather than detail.
- **The gauge is not normalised.** `global_solve.py` never calls COLMAP's
  `normalize()`. Solves range from a ten-unit extent to **340 x 70 x 175** for
  the same kind of walk, so every threshold must be relative to the scene's own
  median depth.
- **The corpus is mostly seated.** Only ~8,000-12,000 frames across 8-10
  captures carry real translation. Most captures are a first-person view of a
  phone or laptop screen.
- **The face redactor fires on hands and carpet.** Over 77 frames of one
  capture it filled a median 0.4% of the frame but 33% at p90 and 57.8% at
  worst. This is pre-existing — the same fill is baked into the keyframe images
  the sparse solve used — but far more damaging to a dense stage.

## 5. The architecture that was built

```
capture → keyframes → GLOMAP global solve            (unchanged)
                            │  poses, K, sparse points, observations, RGB
                            ▼
              ┌──────────────────────────────┐
              │  DENSE FINALIZATION (new)    │
              │  1 redacted keyframe → undistort with the SOLVE's own maps
              │  2 monocular depth per keyframe
              │  3 fit disparity = a/z + b to the sparse points (IRLS, Huber)
              │  4 gate on the HELD-OUT residual
              │  5 mask depth edges, grazing angles, redaction fill
              │  6 keep only where ≥3 other cameras agree within 3%
              │  7 average the agreeing positions, voxel-reduce, LOD ladder
              └──────────────────────────────┘
                            ▼
              <world>/dense/<session>/  (additive, optional)
                            ▼
              GET /worlds/{id}/render → inline WebGL page → WKWebView
```

**Where scale comes from.** A monocular network knows shape and nothing about
size, so it is never asked for either. Every metre of every depth map is
anchored to the multi-view triangulated points the solve already placed. The
network only interpolates between points the solve earned.

**Why nothing is hallucinated.** Four refusals, in order of how much they
remove: the per-frame gate, the validity mask, multi-view consensus, and no hole
filling of any kind. Poisson and Delaunay are closure methods — watertight by
construction — and would turn "never observed" into "surface here". Every
surviving point carries how many cameras agreed with it, and the viewer exposes
that as a live filter.

## 6. Quantitative results

**Geometric accuracy**, measured against the sparse points, which come from
triangulating SIFT correspondences and are therefore independent of the depth
network:

| | |
|---|---|
| per-frame aligned depth vs SfM, held out | **3.0% median** |
| per-frame aligned depth, signed bias | **+0.03%** (unbiased) |
| fused cloud rendered at held-out cameras (closet walk) | **3.74% median, 5.83% p90** |
| pixel coverage at a viewer's splat radius | **47.7%** |

**Appearance, at genuinely held-out cameras** (those frames excluded from the
fusion): PSNR 15.2 median, SSIM 0.485, completeness 98.4%. PSNR is depressed by
auto-exposure drift between frames and should not be read as a geometry figure —
that is what the depth numbers above are for.

**Cost.** Peak VRAM for the depth stage is **0.85 GB**; everything else is CPU
and RAM. A 438-keyframe world takes about seven minutes end to end. The
end-to-end replay measured 327 frames staged, 105 s of dense work inside a 144 s
total.

## 7. Visual inspection

Reconstructions were rendered from real camera poses beside the real
photographs, and from viewpoints no camera occupied.

**What is recognizable.** On the bedroom/closet/desk walk: a closet with
individual garments and hangers, including the specific striped shirt; a wall
with window blinds, two framed pictures and a cubby shelf; a desk with a monitor
showing code, a cup and a keyboard with purple lighting. On the widest traverse:
a shelving unit with bottles, a games console and controller, an RGB keyboard, a
cyan-lit PC tower, carpet, and a bed with pillows.

**What is honestly empty.** Blank white walls, a ceiling fan, and a doorway with
no texture reconstruct as holes. That is the correct behaviour and it is visible
in the same comparison sheets.

**Novel views.** Offsetting the camera to roughly a quarter of the scene's
median depth off the capture path, the shelf, bed, pictures and carpet stay
coherent. Streaking appears in low-coverage regions.

**Artifacts seen and their causes.** Flying pixels at depth edges (fixed by the
validity mask); surface thickness from independent per-frame scale error (fixed
by averaging agreeing cameras); black holes where the face detector fired on
hands; noise on near-textureless desk surfaces.

## 8. The viewer

`GET /worlds/{id}/render` returns HTML that iOS shows in a `WKWebView`, so Tower
owns the saved-world viewer and a real 3-D renderer needed **no Swift change**.

The route's CSP is `default-src 'none'` with no `connect-src`, which blocks
`fetch` and `XMLHttpRequest` outright, so the points travel base64 **inside** the
page and go straight into a GPU buffer. The page reaches for nothing, and a test
asserts that.

**Verified in Chrome** on two real worlds: 393,216 points decoded, 74-108 fps,
orbit, pan, a confidence slider that visibly thins the cloud, colour-by-confidence,
point size, reset, and stepping through 240 capture positions. It opens **where
the wearer stood**, because a room reconstructed from the inside looks its worst
from the outside.

**Not verified:** wheel zoom and WASD flight, because the automation harness
does not deliver a wheel event the canvas sees, nor a key press longer than one
frame of a per-frame integrator.

## 9. Tests

`tower/tests/test_world_builder_dense.py` — 49 tests, no GPU, no network, no
depth model. They build a synthetic scene with a known camera and a known
surface so every geometric claim is checked against an answer that exists
independently of the code.

The refusals are load-bearing and are tested as such: that a depth
discontinuity's pixels are dropped, that a grazing surface is dropped **while a
face-on one is kept** (asserting only the first would also pass if the mask
rejected everything), that no registered depth backend carries a non-commercial
licence, that the dense stage prefers the world's redacted keyframe and
re-redacts when it must and refuses when it cannot, that the `GET /worlds`
contract identifier has not moved, and that a broken dense artifact costs a
world its dense page but never its sparse one.

**Two real bugs were caught by tests before any of it ran on real data:** the
point record was 17 bytes while the manifest advertised a 16-byte stride, which
would have sheared every point in the viewer; and a grazing-angle test that
passed for the wrong reason.

**One bug was caught only by opening the page in a browser:** the viewer's MVP
matrix multiplied view by projection instead of projection by view. That
compiles, links, uploads 393,216 points, reports 69 fps, and paints an entirely
black canvas.

**And one was caught by a test of the fix rather than of the code:** an escaping
change written through a shell heredoc had its escape sequences collapsed, so it
became a no-op that *also* replaced every space in the JSON config with a
JavaScript line terminator. Worse than doing nothing, and invisible on
inspection.

### Regression evidence

Every test outside World Builder, on this branch:

```
2163 passed, 59 skipped, 771 deselected, 1 xfailed in 537.76s
```

That covers CV Lab, Object Memory, Document Memory, Scene Understanding, the
shared camera transport, session lifecycle, listener resilience and process
cleanup. **No failures.**

That is the result that matters most for a change like this, because the only
shared surface it touches is one additive key in the `GET /worlds` listing. The
dense stage itself is a new module, a new script, and two opt-in flags.

## 10. Known weaknesses

1. **Coverage.** Roughly half a typical view is filled. Textureless walls and
   ceilings are genuinely unreconstructable at this resolution and stay empty.
2. **Frames used.** 45-56% on walk data. `align.json` records why each frame was
   dropped; the dominant causes are blur and heavy redaction fill.
3. **Points, not surfaces.** There is no mesh, so occlusion is imperfect and you
   can see through a wall's holes. Meshing was deliberately not done: the
   available closure methods fabricate unobserved geometry, which is the one
   thing this artifact promises not to do.
4. **Confidence has a short dynamic range** on this data, mostly 3-6, so the
   honesty slider saturates quickly.
5. **Scale remains unknown**, unchanged and explicitly so.
6. **The corpus limits the claim.** Only a handful of captures are true walks.

## 11. What still requires physical validation

- **Nothing on iOS was compiled or run.** There is no Mac in this environment.
  No Swift file was changed, and the integration is deliberately arranged so
  that none needed to be — but "the phone shows the dense viewer" is
  **unverified**, and the WKWebView's tolerance for an 8 MB inline page is
  **unmeasured**.
- **A live capture with `--densify` has not been run on the glasses.** The
  equivalent path was exercised end to end by replaying a stored capture through
  the same code.
- The hard-stop skip path is implemented and reasoned about but was not
  exercised by actually killing a live session mid-finalization.
