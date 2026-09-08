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

## D11. Keep the alignment gate strict at 8%, and expose it

**Decided against** loosening it to buy coverage, and against removing it on the
theory that multi-view consensus already does the job.

**Evidence**, sweeping the gate on the closet walk (201 posed frames), scored
against the sparse points — which are independent of the depth pipeline — with
near-point splats:

| gate | frames used | points | depth error, median | pixel coverage |
| --- | --- | --- | --- | --- |
| **0.08** | 91 / 201 | 2.32 M | **3.74%** | 47.7% |
| 0.15 | 133 / 201 | 3.36 M | 5.01% | 56.3% |
| 0.25 | 136 / 201 | 3.34 M | 4.51% | 56.2% |

Two things to read out of that. Loosening buys about **nine points of coverage
for roughly one point of extra depth error**, so it is a real trade rather than
a free lunch. And past 0.15 almost nothing changes — 0.25 admits three more
frames and produces slightly *fewer* points — which says the consensus filter is
already rejecting what the looser gate lets in.

Strict wins on principle rather than on the margin. The gate exists to catch the
one failure consensus cannot: a frame whose entire depth field is mis-scaled.
When several such frames are adjacent — a blurry stretch of a walk — they agree
with **each other**, and three agreeing cameras is exactly the evidence
consensus is looking for. Only the gate sees that, because only the gate
compares against something outside the depth pipeline.

The cost is stated rather than hidden: roughly nine points of coverage, and
`--gate-rel` is on the CLI for an operator who wants the other end of the trade.

---

## D12. The consensus tolerance is 0.05, and tightening it makes things worse

**Reversal.** The obvious value, and the one this shipped with for most of the
lane, was 0.03: agree within 3% of depth or you are not evidence. Measured on
two independent walks, that is the wrong end of a curve with a real minimum.

Bedroom walk, 196 posed frames, 92 used, scored against the sparse points:

| tau | points | depth error, median | depth error, p90 | pixel coverage |
| --- | --- | --- | --- | --- |
| 0.03 | 2.14 M | 5.58% | 9.52% | 75.4% |
| **0.05** | 2.74 M | **5.37%** | **8.14%** | 83.5% |
| 0.08 | 3.03 M | 6.56% | 8.98% | 86.0% |

Closet walk, 201 posed frames:

| tau | points | depth error, median | pixel coverage |
| --- | --- | --- | --- |
| 0.03 | 2.32 M | 3.74% | 47.7% |
| **0.05** | 3.17 M | 3.88% | **54.8%** |

So 0.05 adds 28-36% more points and 7-8 points of coverage while accuracy holds
on one walk and **improves on both median and p90** on the other. And 0.08 is
clearly worse, so this is a minimum rather than a "looser is always better"
slope — which is what makes it a defensible default rather than a preference.

**Why tightening hurts.** Each surviving point's position is the mean of the
positions all AGREEING cameras assign it. A looser threshold therefore admits
more agreeing cameras per point, and the extra averaging cancels more per-frame
error than the looser threshold lets in. Tighten it and you keep fewer, noisier,
less-averaged points. Past 0.05 the threshold starts admitting genuinely
disagreeing cameras and the averaging no longer pays for it.

**Worth carrying:** the accuracy of a consensus filter is not monotone in its
strictness when the filter also decides how much averaging each point gets.
Two parameters were entangled and only measurement separated them.

---

## D13. Gaussian splatting is rejected, and the reason is not quality

**Decided against** adopting 3D Gaussian Splatting as the appearance layer,
after actually training it on our data rather than reasoning about it.

**It worked, and it worked well.** Brush v0.3.0 (Apache-2.0) runs from a
prebuilt Windows binary on wgpu — no Rust, no MSVC, no CUDA toolkit — read our
COLMAP workspaces directly, and reached **23.9 dB / 0.867 SSIM** on held-out
views of the 77-image world and **24.3 dB** on the 429-image one. Dresser grain,
record sleeves, monitors: plainly the room. Off-path translation to about a
quarter of the trajectory extent holds with correct parallax.

**It is rejected for what it does to objects that moved.** The splat *silently
deletes* every object that moved during the capture and paints in a sharp,
correctly-lit reconstruction of what it believes was behind:

- An open laptop, in use, occupying a third of the frame, becomes a hoodie and a
  metal flask on the dresser — clean, consistent with its neighbours, and not
  what was there.
- A hand holding a phone becomes tidy carpet.
- A PC case lit green in the real frame renders blue, because the lighting
  changed mid-capture and the model committed to one colour.

**71% of all held-out squared error lives in the worst 5% of pixels**, and those
pixels are precisely the objects a person was interacting with — which is to say
precisely what a memory product exists to remember. Nothing in the output marks
them.

**And opacity cannot be used to gate it.** The obvious mitigation is to treat low
accumulated alpha as "I do not know here". Measured, it inverts: the view that
looks *worst*, an unreadable smear at 30 degrees off the captured cone, is
rendered **most** opaquely (alpha 0.982, zero under-covered pixels), while a
clean but entirely unsupported fabrication from half a trajectory away sits at
0.908. Alpha tracks "are there Gaussians along this ray", which is trivially
true once splats have been stretched across the scene. **Any gate built on splat
opacity passes the worst renders through.**

This is intrinsic to fitting a static radiance field to a moving scene, not a
coverage problem more capture would fix. The point cloud's confidence channel is
the opposite: it counts independent cameras that agreed, so an object that moved
fails consensus and leaves a hole.

The one salvageable idea, recorded for later: splats as *texture*, clipped to
regions the point cloud already supports.

---

## D14. COLMAP CUDA MVS is a measuring instrument, not the product path

**Decided against** putting real multi-view stereo on the critical path, and
**for** keeping it as an independent accuracy reference.

**It runs, and the Blackwell trap was real but is fixed.** `colmap-x64-windows-cuda`
4.2.0 needs no CUDA toolkit and no MSVC — it statically links cudart. Four checks
ruled out the documented silent-garbage failure on compute >= 10.0: normal maps
are unit length rather than zero, depth maps show scene structure aligned to the
photo, `gpu_mat_test` passes, and depths agree with the SfM points to 0.69%.

| | median relative depth error | pixel coverage |
| --- | --- | --- |
| **MVS, geometric (filtered)** | **0.69%** | **31.9%** |
| MVS, photometric (unfiltered) | 1.09% | 99.9% |
| Monocular, after its own validity mask | 2.58% | 88.5% |

**MVS is three to four times more accurate and covers a third of the pixels.**
Its filtered depth keeps furniture, boxes and record sleeves, and deletes
carpet, painted wall and moving hands entirely — which is the correct behaviour
for a photometric matcher on a corpus that is 83–93% untextured, and exactly why
it cannot be the product path on its own.

Cost settles it independently: **46.4 s per image**, about **5.5 hours** for a
429-frame world, against ten minutes for the monocular path.

So: **monocular produces the better reconstruction; MVS produces the better
measurement.** MVS is now the independent yardstick this lane's accuracy claims
are checked against.

Two things found while establishing that, worth carrying: `colmap.exe
image_undistorter` crashes with `STATUS_STACK_BUFFER_OVERRUN` on models of 400+
images (use `pycolmap.undistort_images`), and COLMAP runs every photometric
problem before any geometric one, so an interrupted dense pass yields no
filtered maps at all.

**The hybrid worth building later:** MVS geometric depth on a subset of frames
as high-confidence anchors conditioning the per-frame fit, attacking exactly the
frames where that fit is worst conditioned.

---

## Open, being decided by measurement

| decision | options | metric |
| --- | --- | --- |
| depth model | Depth Anything V2 Small (Apache-2.0) vs MoGe-2 (MIT) vs Metric3D v2 (BSD-2), against the CC-BY-NC V2 Large as a labelled ceiling | held-out relative residual, and how many frames pass the gate |
| depth source | SfM-anchored monocular vs COLMAP CUDA PatchMatch vs Depth Anything 3 pose-conditioned | same relative-depth metric, plus completeness, runtime, VRAM |
| appearance layer | point cloud alone vs adding a Gaussian splat | whether a splat trained with no CUDA compiler produces a recognizable room, and whether it invents geometry off the capture path |
