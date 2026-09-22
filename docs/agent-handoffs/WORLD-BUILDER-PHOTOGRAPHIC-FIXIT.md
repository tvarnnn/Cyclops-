# World Builder: the photographic fix-it

**The saved world shows the wearer's own photographs, placed in space by the
reconstruction. Geometry is scaffolding; it is not what the wearer looks at.**

| | |
|---|---|
| Continues | `docs/agent-handoffs/WORLD-BUILDER-RECONSTRUCTION-FINAL.md`, whose READY verdict was **rejected** |
| Branch | `world-builder/reconstruction-fixit-v1` |
| Starting SHA | `f383015` (the rejected surface candidate) |
| Final SHA | the commit that sets this line; the campaign's last content commit is `c8d8456390f2` |
| Worktree | `C:\Users\tvllo\Projects\Glasses-worktrees\wb-fixit` |
| Canonical dataset | world `b2a75ab40d2d415d8d6ef5e4d5f0fb3d`, session `a8c6817e14a74e3c977fccfcdacad595` |
| Scratch (local only, not in git) | `C:\Users\tvllo\Projects\Glasses-scratch\wb-final-recon\fixit\` |
| Experiment log | `Glasses-scratch\wb-final-recon\EXPERIMENT-LOG.md`, sections **F0–F43** |
| Verdict | §14 |

---

## 1. Why this campaign existed

The previous campaign delivered a technically correct triangle mesh and called
it ready. The owner looked at it and rejected it: *"it looks like shit."*
Faceted geometry, crumpled walls, stretched triangles, floaters, blob objects
on the phone.

His correction defined the product:

> The captured RGB imagery is the primary **appearance truth**. Geometry, poses,
> intrinsics and depth are **spatial truth** — they decide where an observation
> belongs, what occludes what. *I should not be looking at the triangles.*

Everything below serves one acceptance test: **open the saved world, navigate
it, and perceive the captured room.**

## 2. What the wearer sees now

The Saved World page is a WebGL2 view the iPhone app hosts. It draws the
wearer's own keyframes, blended per view over a hidden proxy mesh:

- for each pixel, the page picks the best few keyframes by angle, distance,
  visibility and sharpness, tests each against that source's own depth, and
  blends them with exposure correction;
- the mesh decides position and occlusion, and is never drawn;
- where no kept frame looked, the page shows a flat grey haze, and says so;
- the wearer's hands, arms and held phone are detected and removed;
- movement is limited to a tube around the walk; **looking is free in every
  direction**, and a compass ring shows which directions were photographed;
- **Best view** and **Face the room** move the camera to a good vantage.

Measured against the wearer's own frames at matched pose and field of view:
median structural correlation **0.817** (0.92–0.95 on the best five), and the
render carries **43.5% picture-class pixels against 39.7% in his own
photography** (`fixit\visual-review4\`). The three worst pairs are worst
because the build correctly removed his hands and leg while keeping the phone
and laptop they held.

## 3. Root cause of the rejected result

`fixit\diagnose\DIAGNOSIS.md`, with controlled fusion experiments:

| Cause | Share of the damage |
|---|---|
| **Cross-frame depth disagreement** (frames place the same wall 6–12 voxels apart) | the crumpled walls, the folds, the holes |
| **Vertex colour** (many frames averaged into one colour per vertex) | correlation with keyframe detail 0.14–0.47, versus 0.61–0.96 for a single frame |
| **Phone decimation** | the blob objects |
| Viewer shading | 5–12% |

At desk range the mesh had **one vertex per 7–15 keyframe pixels**, so no
vertex colouring could ever show screen text or keycaps. That is why the answer
was imagery, not a better mesh.

## 4. What was built

| Area | Change |
|---|---|
| **Appearance stage** (`appearance.py`, `appearance_pipeline.py`) | selects keyframes by coverage, solves per-keyframe exposure, marks unusable pixels, encodes ASTC/WebP chunks, publishes atomically with a manifest |
| **Phone viewer** (`appearance_viewer.html`) | view-dependent blending, unshaded, crack fill, void haze, navigation envelope, free look, compass cue, Best view, Face the room, honest caption |
| **iOS transport** (`WorldAssetTransport.swift`) | a private scheme that feeds the web view from the Tower through an ephemeral session, memory only, whitelisted routes |
| **Geometry** (`depth_consistency.py`, `surface.py`) | a jointly solved per-frame depth correction so frames agree; conservative plane snap; low-weight evidence rules; **rim smoothing along the boundary rather than across it** |
| **Transients** (`transients.py`) | hand, arm and held-phone masks (97.7% / 99.7% recall) |
| **Confidence** | a per-vertex geometry-confidence byte shipped in the proxy; the page fades where geometry is untrustworthy |
| **Privacy** (`reredaction.py`, consensus in `appearance.py`) | preserved, defaults off — see §7 |

## 5. What was rejected, with numbers

Negative results are the campaign's most reusable output.

| Approach | Why it was rejected |
|---|---|
| Baked texture atlas | 30–45% single-triangle charts on imperfect geometry: grainy shards |
| RGB-D patches / surfels | ghosting (−4 dB); snapped to the mesh they only tie it |
| The 1,166 discarded capture frames | −0.14 to 0.00 dB. The selector had rejected them for insufficient motion (866) and blur (256) |
| MVS as the shipped proxy | depth genuinely better (AbsRel 0.98% → 0.68%) but only +0.15 dB at the product, ~46 GPU-min per walk, and it roughens boundaries. Kept as a documented option |
| Re-admitting back-facing faces to close holes | **three lanes measured it**; held-out frames contradict those faces 3–5× too often |
| Strict cross-frame privacy union | coverage 0.977 → 0.419: the room disappears |
| Anisotropy weighting, winner-take-all blending, tighter truncation, near-boost clamp | measured, no gain or worse |

## 6. Evidence

Curated, in `Glasses-scratch\wb-final-recon\fixit\`:

- `candidate\fig\ev_01…ev_14` — the final build: openings, room poses, a desk
  close-up, **render versus the wearer's own frames**, the 360° turn with the
  cue, Best view, rim before/after at 3×, two honest failures, the label.
- `baseline\fig\ev_14_render_vs_source.jpg` — five of his frames above, the
  render at the same pose below.
- `visual-review4\` — 1,285 stills and four sequences by an independent
  reviewer who drove the page through its real input path.
- Before/after against the rejected build: `final3\fig\`, `phone-viewer\fig\`.

## 7. Privacy: deferred by instruction, preserved in code

On 2026-09-21 the owner deferred privacy preprocessing so reconstruction could
be judged on its own: bypass it, **do not delete it**, keep raw imagery local.

What that means in the code:

- `params.imagery_source` = `redacted` (default) or `raw-local-research`
  (`TOWER_WORLD_RAW_IMAGERY`, `--imagery-source`). Raw mode reads the original
  capture frames, writes an empty fill mask, and forces the consensus off.
- Raw builds are labelled in six places — both manifests, the provenance, the
  cache keys, the served headers, the page itself and the world listing — and
  the serving gate **refuses in both directions**, proven over the wire.
- `redaction_consensus` now defaults to **off**. It works (it closed a measured
  leak) but costs a third of the shelf, posters and bed on this capture.
- The redactor, the re-redaction step and the consensus code are untouched and
  one value away from returning.

**A real leak, measured and mitigated** (`fixit\privleak\PRIVLEAK.md`): a face
redacted in one frame is published by other frames that saw the same surface.
On this capture 99.4% of covered surface is published opaque by another
keyframe; the one real face (a printed wall portrait) had 86 publishers. The
consensus rule closes it. **It is off in this candidate by instruction**, and
must be reconsidered before any world is shared or captured anywhere but home.

A sweep of this build found 91 detections over 83 frames — the wearer's hand,
a printed poster face, a lamp, a cup. **No live bystander.**

## 8. Performance, on the RTX 5070

| Stage | Canonical world |
|---|---|
| Surface (masks, consistency, fuse, mesh, snap, pack) | ~390 s, 4.6 GB peak |
| Appearance | ~55 s, 4.3 GB peak |
| Phone payload | 12 requests, ~13.7 MiB, 33 MB GPU |
| Page boot (software renderer; a phone will differ) | ~6.5 s |

Live during a walk: hand masks and a warm-started depth solve run per solve;
the appearance build follows each live surface.

## 9. What is still wrong

1. **Half the room was never photographed.** Turning spends ~144° on honest
   black. No algorithm fixes this; a second walk facing every wall would.
2. The bed side, one drag from the opening, is a torn sheet (~45% of that frame
   has no proxy).
3. A hard black slab beside the door leaf: the door was open in one pass and
   closed in the other.
4. Ragged rims remain at 1×; the rim fix is a real gain at 2× only.
5. Fan-blade cut-outs in the ceiling; grazing views are torn.
6. Sources are 640×360, so everything is soft; raw imagery buys no resolution.
7. The grey void fill still reads as a slab at its edges.

## 10. Tests

`pytest tests -q` in `tower/` at the final SHA: **3,968 passed, 77 skipped, 1 xfailed, 0 failed** (726 s).
The campaign selection (`-k "world_builder or surface or appearance or dense or
geometry or render or result_channel or contract or redaction or artifact_paths
or consistency or transient"`): **1,951 passed, 16 skipped, 2,079 deselected, 0 failed** (502 s).

Swift: `WorldAssetTransportTests`, `WorldRenderRevisionTests`,
`WorldRenderViewerTests`, `WorldRenderRepresentationTests`, plus the UI smoke
test — **written, never compiled**. There is no Mac in this campaign.

Every fix in this campaign carries a test that fails without it; lanes proved
that by reverting each fix in a scratch copy.

## 11. Reviews

Four independent adversarial reviews, each by an agent that did not write the
code: `fixit\review-code\`, `fixit\review-ios2\`, `fixit\visual-review3\`,
`fixit\visual-review4\`. They found, among others: the page dying at every
Stop, three privacy defects, a 24° look window, a caption that promised what
the page no longer did, and an opening pose with 150° of dead horizon. All were
fixed or answered with measurements.

One correction went the other way: a "fixed" opening pose whose numbers all
improved while the picture became a top-down view of the desk underside. The
final lane caught it by opening the PNG.

## 12. Mac validation

`docs/agent-handoffs/WORLD-BUILDER-VIEWER-MAC-VALIDATION.md` is the checklist;
it is current to this branch. In short:

1. Build the app. Compile risk is rated moderate-low; the two likely spots are
   named in the checklist, both in `WorldAssetTransport.swift`.
2. Run the Swift unit tests and the UI smoke test.
3. Serve the canonical world from the Tower on this branch and open it on a
   phone. The app requests the appearance rung by name; an older build is
   served the surface instead, by design.
4. Measure what Windows cannot: frame time, WebContent memory, the compressed
   texture path, touch feel.

## 13. Physical retest

Not yet. When the app is built: walk the room **facing every wall**, moving
rather than turning on the spot, with the lights on. Three quarters of the
current capture's frames are dark and 866 were rejected for insufficient
motion. Then compare the live surface during the walk with the final world
after Stop.

## 14. Verdict

**NOT READY FOR MAC/PHYSICAL RETEST — for one external reason: no Swift in this
campaign has ever been compiled.** That is the only blocker, and it cannot be
resolved on Windows.

The reconstruction itself is ready to be judged. Where the glasses looked, the
saved world is the wearer's own photography placed in space — richer in
picture-class pixels than his own frames, with his hands removed and nothing
invented anywhere, verified pose by pose by an independent reviewer. Where the
glasses did not look, it is honestly empty, and the page says so rather than
inventing a room.

The likeliest first reaction is no longer "this looks like shit" but "wait, is
that all of it?" — which is a statement about the capture, not the renderer,
and the answer is a second walk facing every wall.

Hand this to the Mac: build the app, run the Swift tests, and open the world on
a phone. If it builds and draws, the campaign's remaining questions are all
about capture coverage, not reconstruction.

## 15. Local-only data, preserved, not committed

`Glasses-scratch\wb-final-recon\` (tens of GB) holds every lane's dataset copy,
renders, model caches and browser profiles. It contains **unredacted
first-person imagery** and must not be pushed. It is outside the repository and
nothing from it is tracked. Nothing has been deleted; a cleanup pass is owed
once the Mac and the owner have validated this candidate.
