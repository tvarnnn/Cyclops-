# World Builder roadmap — after stabilization, the next R&D phase

**Living document.** Written 2026-09-07 at the end of the
live/history/lifecycle stabilization lane
(`docs/agent-handoffs/WORLD-BUILDER-LIVE-HISTORY-STABILIZATION.md`).

## 1. Where World Builder stands

The product path today produces three things from a walk:

    original frames (the capture)
      -> solved camera poses (GLOMAP over SIFT, per session)
      -> sparse SfM landmarks (tens of thousands of points, colour, per segment)

On the 2026-09-06 physical walk that meant 467 keyframes, 350 solved poses
and ~20,000 points — visible and orbitable in the in-app picture, and
truthfully labelled as *sparse structure-from-motion output: not a surface,
not a mesh, not metric scale*. A person who walked that room does **not**
recognise it from the picture. That is the gap the next phase closes.

The stabilization lane made the lifecycle around that output deterministic
(Stop → finalize → persist → close → inspectable; interrupted worlds stay
inspectable; Live and History are separate; cartridges switch on one
Tower). It deliberately changed **nothing** in the solver, the fast path,
the thresholds or the sparse pipeline. The next phase builds on top of
those outputs; it must not destabilise them.

## 2. The next milestone: RECOGNIZABLE ROOM RECONSTRUCTION

**Success criterion.** Not "more points". The criterion is:

> A person who scanned the room can recognise the reconstructed room.

Concretely, for a walk like 2026-09-06's: the wall the desk is against, the
desk, the doorway, the window — recognisable as those things, in the right
places relative to each other, from the same in-app viewer, without a
caption that has to explain what the dots are.

**Inputs the phase is allowed to use.** Exactly what the product path already
persists per session, plus the Tower's GPU:

- the solved camera poses (`derived/<session>/poses.json`, `solution.json`)
- the original frames (`captures/<id>/frames/*.jpg`, 360×640, calibrated
  intrinsics in `intrinsics/`)
- the sparse landmarks (`points.json`, `solution.npz` with observations)
- the RTX-class GPU on the Tower host (CUDA torch is already resident)

**Product vision the representation must serve** (and the truthfulness rules
that come with it):

- reconstructed regions fill in progressively as the walk proceeds
- unknown / unobserved regions stay **grey** — visibly not claimed
- **no hallucinated geometry**: nothing is drawn that no frame observed;
  a learned method that completes unseen space is not admissible unless
  its completions are rendered as guesses and can be switched off
- the sparse output stays the fallback and the ground truth for cameras;
  the dense representation is *added*, never substituted silently

## 3. Candidate representation families (to be benchmarked, not chosen now)

| Family | What it would give | Known costs / risks on this stack |
|---|---|---|
| Multi-view stereo (patch-match / plane-sweep) over posed frames | dense depth per keyframe, fusable into a point cloud | 360×640 frames, motion blur, redaction fills; per-frame depth on 400+ frames is minutes on GPU |
| Monocular / multi-view learned depth + fusion (scale-aligned to sparse landmarks) | fast per-frame depth; fills texture-poor walls | scale drift; hallucinated planes are exactly the forbidden failure; needs sparse-anchored scale per frame |
| TSDF / voxel fusion of per-frame depth | a watertight-ish surface, natural grey-for-unknown semantics, progressive fill | memory at room scale × voxel size; depends entirely on depth quality |
| Meshing + texturing of the fused surface | the most "recognisable" output per byte | texture seams from redacted frames; a mesh over-claims where depth was thin |
| 3D Gaussian splatting (3DGS) from posed frames + sparse init | photorealistic novel views quickly; progressive by construction | needs many well-distributed views; "floaters" in unobserved space are hallucination-shaped; a web viewer for splats is new work |
| Learned multi-view feed-forward (e.g. DUSt3R/MASt3R-class) | dense geometry with few views, robust to low texture | scale and consistency with the GLOMAP frame; licensing; GPU memory; must be *aligned to*, not *replace*, solved poses |
| Hybrids (sparse → depth prior → TSDF; or 3DGS initialised from MVS) | best of two | twice the pipeline to keep truthful |

## 4. How the phase should be run

1. **Benchmark on the real captured walks first**, not on synthetic scenes:
   `7d31e8d7…` (09-06 replay, 438 keyframes, 425 posed), `678fe396…`
   (the same walk's live world), `fcbca9e9…` (09-06 evening, 467 keyframes,
   interrupted mid-walk — a good stress case), and the 08-29 / 09-01 replay
   worlds listed in `WB-CV-IOS-WINDOWS-LIVE-VALIDATION.md` §2. All are in the
   canonical world root on the Windows box; none may be modified.
2. **Measure the criterion, not a proxy**: a blind side-by-side where the
   wearer (or a reviewer with the frames) names what they see in each
   candidate's render of the same walk, plus the objective checks that
   protect truthfulness (coverage of observed space, fraction of rendered
   surface with no supporting observation, reprojection against held-out
   frames).
3. **Keep it off the frame path and inside the builder's ownership model**:
   dense work runs as a child of the builder (like `world_solve.py`), after
   the sparse finalization, tracked and killed with it; progressive results
   land in the derived tree under a new, additive geometry kind the contract
   gains explicitly (`WORLD-BUILDER-GEOMETRY.md` is versioned for this).
4. **Ship the viewer change with the representation**: the in-app picture
   must draw the dense output with grey-for-unknown, keep the sparse view
   selectable, and keep its caption truthful for whichever is on screen.
5. **Hardware discipline**: measured on the Tower's GPU with CV Lab idle,
   then with CV Lab running, because both cartridges now share one
   long-lived Tower.

## 5. What is out of scope for that phase

Metric scale (still monocular), learned features in the fast path, changes to
keyframe selection or redaction, and anything that requires a second
camera. Those are separate roadmap items and none is a prerequisite for a
recognisable room.
