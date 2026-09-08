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

### The gate also determines the number this lane reports as accuracy

An adversarial review found this and it is the most important sentence in the
document. The held-out alignment residual quoted everywhere in this lane is the
median **of the frames that passed the gate** -- the median of a distribution
truncated at the threshold. So it improves as the gate tightens, and the gate
tightening makes the reconstruction WORSE: fewer frames, more holes.

`proto/gate_sensitivity.py` prints both ends for every shipped artifact, and the
table lives in `01-EVIDENCE.md` §"What the gate does to the number". Read it
before quoting any residual. A metric that improves as the product degrades is
not measuring the product, and this one does that; what it is actually good for
is comparing two configurations at the SAME gate, which is the only way it is
used to make a decision here.

Two consequences follow, both stated rather than fixed:

* Every accuracy figure in this lane is gate-conditioned unless it says
  otherwise, and the pre-gate figure is always the larger one.
* The reference is not independent. Every residual is `|z_pred - z_sfm|/z_sfm`
  at SIFT keypoints of the same triangulation the affine was fitted to, so it
  cannot see SfM error, and it is evaluated only where the sparse cloud is --
  textured corners, which are the network's easiest pixels. There is no external
  metric ground truth anywhere in this lane. That is a defensible position for a
  system with no depth sensor; it is not "geometric accuracy", and calling it
  that was wrong.

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

---

## D15. The depth network is MoGe-2 ViT-L, chosen by a 24-model bake-off, and the permissive licence cost nothing

The stage shipped its first working version on Depth Anything V2 **Small**,
which was picked for a bad reason: it was the first permissively licensed
checkpoint that ran. So 24 checkpoints were scored on the same 214 frames of
`7d31e8d7`, against the same held-out sparse points the pipeline itself fits to,
with each model's output tried in BOTH affine families (`p ~= a/z + b` and
`p ~= a*z + b`) and the better one recorded.

| model | licence | held-out median | frames passing the 8% gate |
| --- | --- | --- | --- |
| **`moge2-vitl`** (shipped) | **MIT** | **2.38%** | **84.6%** |
| `distill-any-depth-large` | MIT | 2.98% | 80.4% |
| `dav2-large` | CC-BY-NC-4.0 | 3.05% | 80.8% |
| `da3mono-large` | Apache-2.0 | 3.16% | 83.2% |
| `dav2-small` (previous default) | Apache-2.0 | 5.87% | 62.1% |
| `da3-small` | Apache-2.0 | 10.92% | 35.0% |
| `zoedepth-nyu-kitti` | MIT | 20.58% | 12.6% |

Three things came out of it, and only the first is about accuracy.

**The permissive/non-permissive trade did not exist.** The bake-off was framed
as "how much do we lose by refusing CC-BY-NC weights?", with `dav2-large` as the
labelled ceiling. The answer is that we lose nothing: the MIT model beats the
non-commercial one by 0.67 points of residual and 3.8 points of coverage. The
CC-BY-NC checkpoints are registered in `dense.py` and deliberately unreachable
from the default path, and after this they are not even the tempting choice.

**It wins where the pipeline actually fails.** On the textureless quartile --
blank shelving and painted wall, which is the single largest source of dropped
frames in this corpus -- MoGe scores 5.9% with 61% passing where V2-Small
manages 9.9% with 31%. It is better on 89.7% of individual frames, so this is a
distribution shift and not two outliers moving a median.

**It forced a change in the fit.** MoGe emits a point map, not disparity, so the
alignment fits `p ~= a*z + b` and inverts as `z = (p - b)/a`. Keeping the
disparity form costs MoGe 0.63 points -- 3.01% instead of 2.38% -- and throws
away a third of its advantage. `dense.py` therefore carries the model's output
KIND alongside its name, and `depth_from_prediction` branches on it. A backend
registered with the wrong kind is a silent 25% accuracy loss, which is why the
kind is a property of the registration and not a flag.

**What it costs.** 761 ms/frame against V2-Small's 17 ms, and 2.4 GB peak VRAM
against 0.13 GB. On a 429-keyframe world that is minutes, not hours, and the
stage already runs last and off the interactive path -- so the trade is bought
with wall clock the wearer never waits on. The 761 ms was measured on a
contended GPU and is an upper bound.

**The ranking rested on one room, and the cross-check narrows the margin.** The
table above is one solve of one room -- white shelving, a desk, two monitors --
so a model that happened to suit that room would look better than it is. Two
held-out solves were prepped at the time and not scored; they have been scored
since, on the five models that mattered:

| model | licence | `7d31e8d7` (214 fr) | `ecc02df1` (76 fr) | `c2e3cb8a` (106 fr) |
| --- | --- | --- | --- | --- |
| **`moge2-vitl`** | MIT | **2.38% / 84.6%** | **4.24% / 71.1%** | 1.98% / **90.6%** |
| `da3mono-large` | Apache-2.0 | 3.16% / 83.2% | 4.87% / **73.7%** | **1.89%** / 89.6% |
| `distill-any-depth-large` | MIT | 2.98% / 80.4% | 4.70% / 69.7% | 2.25% / 66.0% |
| `dav2-large` | CC-BY-NC-4.0 | 3.05% / 80.8% | 4.71% / 72.4% | 2.26% / 65.1% |
| `dav2-small` (previous default) | Apache-2.0 | 5.87% / 62.1% | 6.12% / 64.5% | 3.18% / 82.1% |

Read honestly, this says three things.

The choice survives: MoGe wins the median on two of three solves and is second
by 0.09 points on the third, wins or ties the pass rate on two of three, and
beats the CC-BY-NC ceiling on all three. The permissive-licence conclusion is
the most robust part of the result.

**The margin over `da3mono-large` is not real.** 2.38 vs 3.16, 4.24 vs 4.87,
1.98 vs 1.89 -- it leads on the room it was selected on and the two of them are
indistinguishable elsewhere. And `da3mono-large` is **2.7x faster** (53 ms
against 145 ms) in **1.6 GB against 2.4 GB**. MoGe stays the default because it
is never worse and the stage runs off the interactive path, where 90 ms a frame
buys nothing the wearer waits on -- but on a smaller card, or if this stage ever
moves onto a path someone waits on, `da3mono-large` is the switch to make and
this table is the evidence for making it. It is registered and reachable today.

The 2.5x headline in the table above shrinks to about 1.4x off its own room.
That is what one solve of one room is worth as evidence, and it is why the
seven-world re-measurement in `01-EVIDENCE.md` §11 exists.

**Nothing here validates the reference.** Every number in this table is
`|z_pred - z_sfm| / z_sfm` at SIFT keypoints of the same triangulation the
affine was fitted to. A model that fails the way COLMAP fails would score too
well, and no monocular bake-off against SfM points can detect that. It needs a
depth sensor. See D16.

---

## D16. Depth Anything 3 is registered, is not the default, and the reason is instructive

DA3 is pose-conditioned: given several views and their poses it predicts depth
that is already consistent between them, which is exactly the property this
pipeline builds by hand out of a per-frame affine fit and a consensus test. It
should have won.

It did not. The `da3-*` checkpoints score 8.30% (base) and 10.92% (small) in the
table above, worse than the incumbent V2-Small; only `da3mono-large` -- the
MONOCULAR variant, the one that does not use the poses -- is competitive at
3.16%. `da3-large` is CC-BY-NC and scores 5.36%.

The windowed backend interface in `dense.py` exists because of this: DA3 needs a
window of neighbouring frames and their poses rather than one image, and that is
a different shape of backend. It was built, it is tested, and it is not on the
default path. Keeping it is cheap and the interface is the part worth keeping:
the next pose-conditioned model that does win will need exactly it.

The lesson recorded rather than the result: a model whose architecture matches
the problem is not thereby better at the problem, and the only way to find that
out was to measure it.

---

## D17. The phone's byte budget is met by a coarser grid, not by a confidence cut

**Reversal, found by opening the phone's own page and looking at it.** Nothing
in the test suite caught this, because every test asserted the behaviour that
was wrong.

The rule was: when a level exceeds the byte budget, sort by confidence and keep
the best N. It reads as the honest choice — the geometry several cameras agreed
on survives, the weakest goes — and the caption said exactly that.

It is not honest, because **confidence is not distributed evenly through a
room.** It is high where the wearer stood still and low at the far end of every
space they walked past once. A global threshold therefore does not thin a room;
it deletes the parts of it that were seen from fewer angles.

Measured on `672578d0`, the three-room chain, whose mobile level is 2.60 M
points against a 393 k budget:

| | old rule | first fix | shipped |
| --- | --- | --- | --- |
| points shipped | 393,216 | 295,480 | **373,252** |
| share of the byte budget | 100% | 75% | **94.9%** |
| confidence floor | **9** | 3 | 3 |
| what the phone showed | isolated slabs of wall and ceiling; two of three rooms absent | the whole dwelling, coarser | the whole dwelling, coarser |

The fix is to meet the budget the way the LOD ladder meets it: a coarser voxel
over the whole extent, keeping the best confidence in each cell.

**The middle column is the more interesting failure.** The first fix solved for
the cell size from the scaling law and then took the first count under the
budget. That cannot correct an UNDERSHOOT, and the first step is always an
undershoot, because it feeds the count at NO reduction into a law about the
count at the current cell. With a size hint it spent 75% of the budget. Without
one, which is what all four of its new tests did, it spent **2.7%** and every
test passed, because they asserted only that the result was not LARGER than the
budget. 21 points out of 8,000 satisfies that.

So the loop now probes until the count is inside 85-100% of the budget, keeping
the best result seen, and a test asserts the budget is SPENT. Probing is cheap
because a helper predicts the count of a reduction without building it, so only
the winning cell is materialised. Measured across budgets from 100 to 50,000
points on a spatially split cloud: 88-98% of budget, both halves equally
represented.

**What this cost.** About 5% of the budget, because a voxel grid cannot hit an
exact count. That is the entire cost.

**What it says about the rest of this lane.** The original defect shipped, was
covered by four passing tests, survived one adversarial review, and was found in
about thirty seconds by building the page the phone actually gets and looking at
it. Its replacement then shipped a 37x undershoot with four NEW tests passing,
and was found by a reviewer measuring the function rather than reading it. Both
failures have the same shape: an assertion that bounds the answer on one side
only. If a value has a target, test the target.

---

## D18. The LOD gauge is a DEPTH statistic used as a SPATIAL one, and on a tight room the two diverge

D5 fixed the right bug: absolute voxel sizes are meaningless in a gauge-free
solve, so the ladder is expressed as a fraction of the scene's own
`median_scene_depth`. That is still right. What it does not account for is that
median depth and scene EXTENT are different quantities, and their ratio is not
stable across this corpus.

| world | median depth | L0 voxel | camera extent | L0 points | L1 points |
| --- | --- | --- | --- | --- | --- |
| `7d31e8d7` (desk) | 7.02 | 0.0211 | 15 x 17 x 11 | 8.22 M | 1.48 M |
| `6427900d` (bathroom) | **1.12** | **0.0034** | 30 x 36 x 29 | 3.57 M | 1.86 M |

The bathroom walk is a tight room the wearer never gets far from, so its median
depth is a sixth of the desk world's while its extent is twice as large. The
ladder therefore lays a cell six times finer over a scene twice as big, and the
levels stop separating: the desk world's ladder thins by 5.5x per step, the
bathroom's by 1.9x. Its mobile level lands at 1.22 M points, three times the
phone's budget, where the desk world's lands under it.

**Not changed, and the reason matters.** The obvious fix -- gauge on the extent,
or on the larger of the two -- would change the voxel size of every level of
every world, and therefore every artifact, every point count and every number in
`01-EVIDENCE.md`, to fix a symptom that the serve-time budget already handles
correctly now that it spends its budget (D17). A finer L0 stores more noise but
loses nothing; it costs disk, and disk is the cheapest thing here.

What is wrong is the CLAIM, not the ladder: `dense.py` says L0 is "already at
the edge of what the evidence supports", and on a tight room it is well past
that edge. Recorded here rather than papered over. The fix, when someone takes
it, is to gauge on `max(median_depth * k, extent * j)` and to re-measure the
whole corpus rather than assume the ratio holds -- which is the same mistake in
a new place if it is not measured.

---

## D19. Live progressive reconstruction: the sparse world goes live, the dense world stays at Stop

**The question.** Could the reconstruction appear while the wearer is still
walking, so that Stop is not the first moment a 3-D world exists? Four
independent investigations answered it: a technique survey of the online
reconstruction field, an audit of which stages of this pipeline are actually
incremental, an analysis of what the iOS client and the transport can carry,
and an adversarial reviewer briefed to argue against. Reports 16-19.

**The answer in one line: most of it already ships, the expensive half should
not, and the missing piece is a rendering and a criterion rather than a
technique.**

### What was already true before anyone started

The audit found this and it reframes the question. A global solve **already runs
during the walk**, every 50 accepted keyframes; `engine.build()` rewrites the
derived tree **every 4 keyframes**; the status snapshot carries
`geometry.revision` on a 0.5 s poll; and the phone already refetches only the
segments whose `content_hash` or `placement_hash` moved. Segments visibly snap
together mid-walk today. What the wearer sees while this happens is a **2-D
top-down canvas**. The 3-D viewer is a one-shot page for saved worlds.

So "Stop is the first moment a meaningful world exists" was not accurate. Stop
is the first moment a **3-D** world exists.

### Why the dense stage does not move into the walk

Not because it is too slow. Measured against real capture durations rather than
replay clocks -- `proto/live_budget.py`, and this correction mattered, because
one investigation divided by the replay clock and overstated the load by 2.3 to
3.6x:

| world | keyframes/s of walking | depth | fuse | together | together with `da3mono-large` |
| --- | --- | --- | --- | --- | --- |
| `7d31e8d7` | 2.18 | 0.50x | 0.59x | **1.09x** | **0.77x** |
| `1b8812b1` | 2.22 | 0.51x | 0.51x | **1.02x** | **0.70x** |
| `a378331a` | 1.27 | 0.28x | 0.25x | **0.53x** | **0.35x** |
| `37e497f8` | 1.26 | 0.29x | 0.28x | **0.57x** | **0.38x** |
| `fc58a64d` | 1.25 | 0.28x | 0.38x | **0.66x** | **0.48x** |

Seconds of work per second of walking. **Both expensive stages could keep up**,
and with the cheaper backend D15 keeps registered they keep up comfortably. The
consensus rule survives causality too: restricted to cameras that already
existed, 47-59% of today's points survive at the capturing instant, and choosing
the ten nearest EARLIER cameras recovers 69-88% of the final yield, with a
median wait of 2 keyframes -- about 0.17 s. The affine fit's support is already
94% causal, and the depth network takes an image and nothing else, so its output
is final the moment a keyframe is accepted.

Three things stop it anyway, and only the third is fatal.

**The GPU is shared and the contention penalty is measured.** MoGe-2 costs 145 ms
on a quiet card and **761 ms median with a 5,251 ms p90** when another job is on
the same device -- 5x on the median, 36x on the tail, from this lane's own
bake-off. Four other cartridges use that card. A live loop with a GPU deadline
is a promise this machine cannot keep, and the failure would be blamed on
whichever cartridge noticed first.

**The gauge is the genuinely batch part.** `median_scene_depth` on a prefix of a
three-room walk is **+352% wrong at 10% of the walk** and still +104% at half of
it; the far clip swings -95% to +132% between consecutive checkpoints. Every
voxel size derives from it, and `run_pack_stage` discards points outside a
percentile box that a later gauge would keep. A live ladder means repeated
rebuilds, and it is only affordable if the per-point accumulator is retained --
which is exactly what `prune_intermediates` deletes.

**And the pre-solve geometry is not a room.** This is the fatal one, and this
system has already measured it on itself. `bundle.py` records the live
incremental backend drifting **18.2% of path length by 40 keyframes** on
synthetic data with nothing going wrong, with the recovered scale collapsing
3.1x over the same span. On the real 438-keyframe walk the forward chain
produced **34 segments**, one of which loses a factor of ten of scale along its
own length, and gluing them pairwise placed 58 keyframes in a common frame. The
same keyframes through GLOMAP: 428 of 438, one component, 0.84 px.

Walks average **34 segments and the largest holds 13.6% of one**. Assembling
them without a solve would not invent a chair. It would invent the floor plan,
which is the thing every honest chair hangs on -- and the rule this lane is held
to says an honest hole beats a convincing invention.

### What the field offers, and why almost none of it is available

The survey is worth reading for the disqualifications, because they are not
about quality. This machine has **no Visual Studio and no MSVC**, and a CUDA
11.8 toolkit against an `sm_120` card. Nothing that compiles a CUDA or C++
extension can be installed, and no published wheel matches this
Python/torch/CUDA combination. That removes DROID-SLAM, DPVO/DPV-SLAM, every
online Gaussian-splatting system (the only permissive rasteriser JIT-compiles
and disables itself without a toolkit; the newest Windows wheel is Python 3.10
and CUDA 12.4), nvblox, which does not support Windows at all, and Open3D's GPU
path, whose Windows CUDA wheel has not shipped. Of what remains, the classical
SLAM systems are GPLv3, the good fusion systems are non-commercial, and every
confirmed streaming pointmap model but one is CC-BY-NC -- including several
whose permissive top-level licence sits on a non-commercial CUDA core, which a
repository-level scan does not catch.

There is also no IMU and no shutter-accurate timestamp, which removes every
visual-inertial system and with it the entire published route to metric scale.
And the camera has a rolling shutter, which the direct-VO family explicitly
warns against. **The shipped design sidesteps that last one for free**: global
SfM over parallax-selected keyframes never assumes a rigid instantaneous
projection, it just refuses the bad frames. That is an unremarked advantage of
what is already here over every live alternative.

### The decision

**Nothing about the dense stage changes.** It stays at Stop, unchanged, and the
saved world keeps the quality this lane measured. That is the load-bearing half
of the answer: the reconstruction people will look at is not made worse.

**The live half is a cadence and a rendering, not a technique.** The pieces are
already built: frozen segments in local frames, mutable Sim3 placements, a
manifest diff the phone already honours, a `--solve-every` flag that already
exists, and a contract that already says *"loop closure moves placements rather
than points"*. Re-running the existing solve on the existing persistent feature
database costs **zero VRAM and under 2 GB of RAM**, entirely on the CPU, and
gives a genuinely bundle-adjusted world every 15-60 s of walking -- three to
eight real updates per walk, each one a solve rather than an estimate.

**The one thing that must be added is honesty about motion.** A re-placed
segment is geometrically correct and perceptually a teleport. Before any of this
reaches a wearer, measure how far segments actually move between successive
solves, and gate what is shown on a *settled* threshold derived from that
measurement rather than guessed. Apple's answer to the identical problem is
instructive: while tracking is relocalising, ARKit withholds plane anchors and
hit-test results entirely, and its guidance is to show content only once
tracking returns to normal. Showing nothing beats showing something about to
jump.

That measurement is `reports/20-segment-settling.md`, and the criterion is not
committed to until its numbers are in.

### What is explicitly NOT decided here

Live depth precompute is the one tempting middle path: the network is
future-independent, its output is final at capture, and caching it would remove
0.21-0.30 s per keyframe from the wait at Stop -- about 95 s of the desk world's
230. It is not adopted, because it buys latency the wearer does not experience
(the dense stage already runs after the world is released and reports `ready`
throughout) at the cost of the one contended resource the survey says to keep
clear. It is written down here so the next person does not have to rediscover
that it is arithmetically fine and strategically wrong.

---

## D20. `neighbours` stays at 10: more witnesses buy coverage and cost the tail

`neighbours` -- how many nearby cameras the consensus test even consults -- was
the only parameter in the consensus block with no measurement behind it. It was
also the most promising lever for the one limitation an independent visual
review left standing, which is that the enclosure does not close: raising it
does not lower the bar the way a smaller `min_views` or a looser `tau` would.
It just looks harder for witnesses that already meet it. A wall the wearer
walked past may well have three cameras that saw it, none of them among the ten
nearest.

Measured by re-fusing the desk world from its cached depth maps, so the only
thing that changes between rows is the neighbour search:

| neighbours | L0 points | depth error, median | depth error, p90 | coverage |
| --- | --- | --- | --- | --- |
| **10 (shipped)** | 8.22 M | **2.31%** | **12.7%** | 97.9% |
| 16 | 8.72 M (+6%) | 2.35% | 17.2% | 98.5% |
| 24 | 9.04 M (+10%) | 2.43% | 17.7% | 98.6% |
| 32 | 9.12 M (+11%) | 2.32% | 17.4% | 98.8% |

**It works, and it is not worth it here.** Ten to eleven percent more points and
about one point of coverage, against a **p90 depth error that rises 37%**, from
12.7% to 17.4%. The mechanism is visible in the numbers: a more distant camera
is a weaker witness, so the points that only a wider search can rescue are
systematically the worse ones, and the median -- which they do not reach --
barely moves while the tail they land in gets heavier.

That is a bad trade for a world already at 98% coverage. **The world that
matters is the one with the gap**, so the same sweep was run on the closet walk,
which covers 60%:

| neighbours | L0 points | depth error, median | depth error, p90 | coverage |
| --- | --- | --- | --- | --- |
| **10 (shipped)** | 5.11 M | **4.57%** | **5.62%** | 60.1% |
| 16 | 4.96 M (**-3%**) | 4.96% | 5.71% | 60.9% |

**On the world with the coverage gap it is worse in every column but one.**
Fewer points, not more; accuracy down four tenths of a point; and eight tenths
of a point of coverage bought with all of it. The desk world's extra points came
from a scene that already had witnesses everywhere; the closet has a coverage
gap because nothing looked at those surfaces twice, and no width of search finds
a camera that does not exist.

(The 24- and 32-neighbour rows were still fusing when this was written. The
trend across six measured configurations on two worlds is one-directional and
the decision does not wait on them; `sweep-closet/` holds them when they land.)

**What this rules out, and it is the useful part.** It rules out the cheap fix
for the incomplete enclosure. The walls that are missing are not missing because
the search was too narrow: on the world that has them, widening the search
returns *fewer* points. Closing the enclosure honestly needs more observations
of the surfaces that have none -- which is a capture problem, and the product
answer is to tell the wearer while they are still in the room, not to tune a
fusion parameter afterwards. That is the one place where the live investigation
(D19) and this measurement point at the same feature: a live coverage cue is
worth more than any offline parameter in this block.

## Open, being decided by measurement

Two of the three questions this section opened with have been answered by
measurement and moved into decisions of their own. What remains open is listed
after them.

| decision | options | metric | outcome |
| --- | --- | --- | --- |
| depth model | 24 checkpoints, permissive and not | held-out relative residual, and how many frames pass the gate | **closed: MoGe-2 ViT-L, D15** |
| depth source | SfM-anchored monocular vs COLMAP CUDA PatchMatch vs DA3 pose-conditioned | same, plus completeness, runtime, VRAM | **closed: monocular, D14 and D16** |
| appearance layer | point cloud alone vs adding a Gaussian splat | whether a splat invents geometry off the capture path | **closed: point cloud alone, D13** |

Still open, and honestly open:

| question | why it is not answered here | what would answer it |
| --- | --- | --- |
| Is the reconstruction accurate, as opposed to self-consistent? | Every number in this lane is measured against the same triangulation the pipeline fits to. It cannot see SfM error. | A depth sensor, or a survey-grade scan of one of these rooms. |
| Does the gate drop frames that carry the geometry nothing else covers? | `frames_dropped` counts frames, not the coverage they would have added. | Fuse with the gate disabled and diff the coverage, not the accuracy. |
| Does it hold on a room this corpus does not contain? | Seven worlds, one dwelling. | A capture somewhere else. |
