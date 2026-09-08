# World Builder dense reconstruction — decisions and the evidence for them

Each entry records what was decided, what it was decided against, and the
measurement that decided it. Where a decision was reversed, the reversal is
recorded rather than the record being tidied.

---

## D1. Keep the sparse GLOMAP solve as the registration backbone

**Decided against** replacing it with a feed-forward reconstructor (VGGT-Omega,
Pi3, DUSt3R/MASt3R, CUT3R, Fast3R) or re-solving with a different SfM.

**Evidence.** The existing solve is already sub-pixel: 0.753 px median
per-frame reprojection error over 436 posed frames, 405 observations per frame,
two components with 429 images in the first. Nothing in the survey improves on
that for our case, and three things count against replacing it:

- Only Depth Anything 3 accepts existing poses as a first-class input at all;
  VGGT-Omega cannot take them, so it would discard a better answer than it
  produces.
- The licensing is hostile. VGGT-Omega is FAIR Noncommercial with gated
  checkpoints, Pi3's weights are CC-BY-NC, and Depth Anything 3's *default*
  checkpoint is CC-BY-NC.
- Confidence heads do not deliver honest geometry. VGGT's depth branch is
  reported 48.5% accurate when confident. The mechanism that works is
  multi-view geometric consensus, which needs poses — which we already have.

**Reconsider if** a permissively licensed pose-conditioned model measurably
beats the current depth path on the same metric.

---

## D2. Anchor monocular depth to the sparse points rather than trusting a model's scale

**Decided against** using a metric-depth model's absolute output, and against
using the network's own confidence as the filter.

**Evidence.** RGB-only metric priors carry roughly 15.7% single-frame error and
the bias does not average out. Fitting `disp = a/z + b` per frame against the
sparse points the solve already placed gives, held out on points the fit never
saw, a **3.0% median relative depth residual** — and the per-frame depth is
**unbiased** (+0.03% signed median over 345 frames). Scale, position and
orientation come entirely from triangulated geometry; the network only
interpolates between points the solve earned.

---

## D3. Multi-view consensus, not a confidence head, decides what survives

**Evidence.** Independently wrong depths do not agree. Requiring three other
cameras to agree within 3% removes 38-40% of candidates. This is also what
makes the artifact defensible as spatial memory: an empty region means the
observations did not support geometry there.

**A limit worth stating.** Consensus does not catch a *correlated* error. The
face detector fires on the same object from several nearby cameras, so several
cameras agree on a black box. That is why redaction fill is masked explicitly
rather than left to consensus.

---

## D4. Average the agreeing cameras' positions, do not keep the reference camera's

**Evidence**, on identical held-out views:

| | reference only | averaged |
| --- | --- | --- |
| rendered depth error, median | 7.42% | **6.32%** |
| rendered depth error, p90 | 10.24% | **8.23%** |
| PSNR | 14.87 | **15.22** |
| voxels from the same 35.2 M points | 10.63 M | **8.73 M** |

Fewer points *and* better depth: 18% of the points were redundant surface
thickness rather than information.

---

## D5. Voxel sizes are fractions of the scene's median depth, never absolute

**Reversal.** This was originally absolute (0.02 world units), on the belief —
inherited from an early audit — that COLMAP normalises every model to extent 10.

**Evidence that overturned it.** `global_solve.py` never calls `normalize()`.
Solving eight further captures produced gauges from a ten-unit extent to
**340 x 70 x 175** for the same kind of walk. An absolute voxel would shatter
one world into hundreds of millions of points and collapse another into a blob.

The chosen fractions (0.003 / 0.0065 / 0.013 of median scene depth) reproduce
the previous sizes on the reference world.

---

## D6. The canonical level is L1, not the finest level

**Evidence.** The corpus is 360x640 — 0.23 MP, natively, for all 97 captures
and 45,594 frames. That gives about 6.4 mm per pixel at 3 m and 1-3 cm of depth
noise. L0 is finer than the evidence supports and mostly stores noise. L0 is
kept as an engineering export, not as the product default.

---

## D7. Read redacted pixels, and exclude the fill from geometry

**Decided against** reading the raw capture frames, which is what the first
implementation did and which would have rebuilt the room out of exactly the
pixels the privacy transformation removes, at far higher density than the sparse
cloud ever exposed.

**And decided against** trusting the fill as scene content. Measured over 77
frames of one capture, the detector filled a median 0.4% of the frame but 33% at
p90 and 57.8% at worst, with the boxes sitting on the wearer's hands and on
carpet. The damage is not confined to the filled pixels:

| redaction fill | frames | held-out residual | pass the 8% gate |
| --- | --- | --- | --- |
| under 1% | 37 | 6.9% | 65% |
| 1-10% | 16 | 6.1% | 62% |
| 10-30% | 13 | 10.3% | 46% |
| over 30% | 7 | 34.8% | **0%** |

So the fill is **inpainted before the network sees it** — which protects the
rest of the frame — and still **masked out of the reconstruction**, because
whatever the network puts inside the hole is invention. Inpainting moved the
>30% band from 34.8% to 20.3% and the 10-30% band from 10.3% to 8.3%, and
changed nothing where there was no fill, which is the right shape for the
change to have.

Honest limitation: on that dataset it converted no additional frame past the
gate.

---

## D8. The dense artifact is additive and optional

**Forced by** `require_schema`, which refuses any version but 1, and by the
absence of migration machinery. It lives at `<world>/dense/<session>` beside
`solve/`, following the `support.json` / `placements.json` precedent.

**And the `GET /worlds` contract identifier does not move**, because iOS
equality-tests it on the first line of every guard, and a mismatch there
produces an empty gallery with no user-visible message.

---

## D9. Serve the phone an inline payload, not a fetch

**Forced by** the render route's CSP: `default-src 'none'` with no
`connect-src` blocks `fetch` and `XMLHttpRequest` outright. The documented
promise is a page that loads nothing from anywhere, and that is worth keeping,
so the mobile level is embedded in the page.

---

## D10. COLMAP dense MVS was not available, and the reason was not the obvious one

`hasattr(pycolmap, "patch_match_stereo")` is **True**, which invites the
conclusion that dense stereo works. Calling it raises *"Dense stereo
reconstruction requires CUDA or HIP"*. The correct probe is
`pycolmap.has_cuda`, which is `False`; the Windows wheel ships no CUDA and
upstream says CUDA wheels are Linux-only.

This is recoverable — a prebuilt `colmap-x64-windows-cuda` binary statically
links cudart and needs no toolkit — and that path was measured separately. It
is recorded here because the failure mode is a silent one.

---

## Open, being decided by measurement

| decision | options | metric |
| --- | --- | --- |
| depth model | Depth Anything V2 Small (Apache-2.0) vs MoGe-2 (MIT) vs Metric3D v2 (BSD-2), against the CC-BY-NC V2 Large as a labelled ceiling | held-out relative residual, and how many frames pass the gate |
| depth source | SfM-anchored monocular vs COLMAP CUDA PatchMatch vs Depth Anything 3 pose-conditioned | same relative-depth metric, plus completeness, runtime, VRAM |
| appearance layer | point cloud alone vs adding a Gaussian splat | whether a splat trained with no CUDA compiler produces a recognizable room, and whether it invents geometry off the capture path |
