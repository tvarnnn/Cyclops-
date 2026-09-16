# World Builder: the final reconstruction campaign

| | |
|---|---|
| Starting SHA | `869d715` (the Mac-validated field-test candidate) |
| Final SHA | `bb15a77` |
| Branch | `world-builder/reconstruction-final-v1` |
| Worktree | `C:\Users\tvllo\Projects\Glasses-worktrees\wb-final-recon` |
| Review worktrees (merged, kept) | `wb-final-recon-r1` … `wb-final-recon-r4`, branches `world-builder/reconstruction-final-review1` … `review4` |
| Scratch | `C:\Users\tvllo\Projects\Glasses-scratch\wb-final-recon` (about 21 GB) |
| Canonical dataset | world `b2a75ab40d2d415d8d6ef5e4d5f0fb3d`, session `a8c6817e14a74e3c977fccfcdacad595` |
| Pushed | No |
| Experiment log | `Glasses-scratch\wb-final-recon\EXPERIMENT-LOG.md` (E0–E11) |

---

## 0. The answer in one screen

The saved world is no longer a cloud of dots. It is a **surface**: a
vertex-coloured triangle mesh. The pipeline:

1. Runs MoGe-2 monocular depth on every redacted keyframe.
2. Anchors each depth map to the global SfM solve by a robust affine fit, and
   gates it on held-out error.
3. Fuses them in a block-sparse truncated signed distance field on the RTX 5070.
4. Keeps a triangle only where **at least two frames measured it, no 2:1
   majority of frames saw through it, and a supporting camera saw its front**.

No hole is ever closed.

- **What Tristan sees.** Saved Worlds opens the surface by default, in the same
  WKWebView as before, as a self-contained WebGL2 page: 229k triangles, 6.17 MB
  on the canonical world. The page opens on the camera pose whose view takes in
  the most surface, with the horizon levelled to the surface's own floors and
  walls.
  - On the canonical walk, a stranger recognises a bedroom corner: the white
    desk hutch with shelves, cups and a box; the monitor and its purple-lit
    keyboard with a red mouse; the desk chair's legs on a wood floor; the two
    framed pictures on the back wall; the ceiling edge; and the open white door
    (§12).
- **During the walk.** Each time a background global solve lands, a coarser
  live surface is built. In the real-time replay, the first appeared about 65 s
  into the walk, and the median latency from solve to surface was 38 s under
  heavy CPU contention (27 s in an earlier, less contended run).
  - An open picture on the phone upgrades itself from dots to surface.
  - A newer live surface is *offered* with a button rather than swapped, so the
    camera is not reset mid-look.
- **After Stop.** The full-resolution surface is built from the final solve, and
  it replaces the live one on the phone without a tap. In the replay it landed
  185 s after Stop; that includes waiting for the final solve.
- **Sparse SfM** stays one tap away as the solver's diagnostics view. The ladder
  is surface → dense points → sparse.
- **Still missing, honestly.** The canonical capture is one side of a room. From
  the desk looking back into the room there is almost nothing, because the
  wearer never looked there. About a quarter of measured depth yields no
  surface, because frames disagreed about it. The phone level loses fine detail:
  the monitor becomes a glowing blob and the keys vanish.

**Verdict: READY FOR MAC/PHYSICAL RETEST** (§24).

---

## 1. Commits

`git log --first-parent 869d715..bb15a77` on
`world-builder/reconstruction-final-v1`, oldest first.

| SHA | What |
|---|---|
| `12b754a` | Merge the unmerged dense lane. Add the volumetric surface stage, live and final wiring, and the representation ladder |
| `7566def` | The WebGL2 viewer page and the surface contract. **Its template was committed as NUL bytes after a session crash; fixed in `3a232e5`** |
| `3a232e5` | A reconstructed world is not "no geometry yet". End-to-end pipeline tests. Template restored |
| `a6a24f7` | iOS: the native caption follows the rung |
| `f8eb842` | The face redactor stops blacking out hands and walls |
| `2df9a64` | An open picture follows `/render/revision` (Tower route + iOS follower) |
| `9f5f9a3` | Phone viewer, reviewed and measured (memory, touch, context loss, exposure, CSP) |
| `327c54c` | The final surface is built from the final solve (stale depth cache). Prediction reuse, live cadence, `TOWER_WORLD_SURFACE` on |
| `a9f8600` | Weld tile seams before component pruning |
| `17dfa50` | The live surface ends when the walk does |
| `2e3ee5d` | A triangle index is not a face identity |
| `421c8c4` | Extract only occupied tiles (795-keyframe mesh stage 520 s → 9.5 s) |
| `97e75eb` | Block budget, host-memory frames, LOD cascade |
| **Review round 1** (systems, iOS, truthfulness) | |
| `53be0c3` | Redaction fails closed when it cannot judge (sticky `none`, unjudgeable fills) |
| `e5f72f3` | Builds publish as a unit (per-build level names). Caches name their solve, backend and image hash |
| `1e1cdc8` | Contract: the unit publish |
| `f4eac09` | The revision says whether a build is live. Every page carries its CSP |
| `fe84ee3` | iOS: a failed refresh keeps the drawn world. A same-rung rebuild is offered, not forced |
| `701e8f6` | iOS: the finished surface after Stop swaps in without a tap |
| `888ac69` | Merge surface-r2: `687d47a` (keep measured far walls; drop what frames contradict) and `e2aa002` (caption, contract §2, opening view) |
| **Review round 2** | |
| `843f2dc`, `bdf49e6` | A live build's fill mask sees the engine's fill. Versioned fill rule |
| `54a37f6` | iOS caption no longer says a gap is where nothing looked |
| `9121367` | The contract states the exact 2:1 rule. Back-facing removal is pinned by a test |
| `b1accdb` | A frame-sized face box must be re-found at another resolution (`plausibility2`) |
| `fc99d20` | Prune grace counts from supersession. Stopped packs leave nothing. The budget refuses loudly |
| `5b91a38` | iOS per-rung refusal, no downgrade offers. The Tower never says `surface:None` |
| `f8aecba` | The phone budget bounds the page. Pack makes the phone level fit |
| **Final canonical rebuild and review round 3** | |
| `3b5ea60` | The page levels its horizon to the surface's floors and walls |
| `7ac1baa` | A build that built nothing says so |
| `defc9c1` | The solver's view is the sparse page on every rung; a published build is not live |
| `7b95d6e` | A depth model that cannot be had says so (permanent for the walk); a session's refusal is not the machine's |
| `7b35fcf` | iOS: a finished world is not hidden by failed rebuilds; the page decides what is swapped in; Try again after a revert |
| `b4444aa` | A close face cut by the frame edge is filled on its landmarks (`plausibility3`) |
| `481da3a` | Mac validation doc targets this branch with a built surface; contracts say what the code does |
| `bb15a77` | Merge of review 4 (the last code commit) |
| (this commit) | This handoff |

---

## 2. Dataset inventory

The canonical world:
`C:\Users\tvllo\Projects\Glasses\tower\data\world_builder\worlds\b2a75ab40d2d415d8d6ef5e4d5f0fb3d\`.
It was **read-only throughout**, and every build ran on a copy. The final
evidence copy was hashed before and after: 821 files, 0 changed content, 0
changed mtimes (`final-evidence\hashes_original_*.json`). The full inventory is
in `forensics\INVENTORY.md`, with the loader `forensics\wb_dataset.py`.

| Item | Where | Authoritative? |
|---|---|---|
| 398 keyframes, 360×640 JPEG, face-redacted | `sessions/<sid>/images/` | **Yes.** It cannot be regenerated except from the raw captures |
| Raw capture frames (1,564 across two captures, unredacted) | `tower/data/captures/159372d0…`, `0892c308…` | **Yes** |
| Session record, keyframe/edge/event journals | `sessions/<sid>/*.json(l)` | **Yes** |
| Self-calibrated intrinsics (0.289 px RMS over 511 views) | session + `intrinsics/360x640.json` | **Yes** |
| Global solve: world-to-camera poses, 14,415 points, 131,240 observations | `solve/<sid>/solution.json`, `.npz` | Derived, but expensive (~96 s) |
| COLMAP database | `solve/<sid>/database.db` | Derived |
| Undistorted images (**not redacted**, see §22) | `solve/<sid>/images/` | Derived |
| Segment-local sparse tree | `derived/<sid>/` | Derived; **different frame and convention** from the solve |
| Depth predictions, fill masks, per-solve fits | `dense/<sid>/align.json`, `work/` | Derived, rebuildable, prunable |
| Surface mesh ladder | `surface/<sid>/manifest.json`, `mesh_l{0,1,2}.<build>.bin` | Derived, rebuildable |

**The canonical checkout has no `dense/` or `surface/` for this world yet.**
Building them is step 1 of the retest procedure (§21).

Two traps were verified by measurement (E2):

- `world.json`'s `T_world_camera` describes the segment-local tree. By contrast,
  `solution.json` is world-to-camera (0.700 px reprojection, against 528 px for
  the inverse).
- `edges.jsonl` is not a co-visibility graph.

The capture is hostile to reconstruction:

- 0.23 MP frames;
- 79% of frames have mean brightness under 80/255;
- the median image is 57% texture-free;
- the median depth-to-baseline ratio is 30.

---

## 3. Authoritative versus derived

Only captures, keyframes, session journals and intrinsics are authoritative.
Everything under `solve/`, `derived/`, `dense/` and `surface/` is rebuildable
from them (§11), and no build writes outside those derived directories.

- **Unit publishing.** A surface build publishes as a unit. Level files carry
  the build id, `manifest.json` is written last and names them with byte counts,
  and readers resolve levels only through the manifest.
- **Pruning.** Superseded levels are pruned 120 s after they are superseded.
  Orphans from stopped builds are swept by age.
- **Cache identity.** Caches name what they were computed from: the solve digest,
  the depth backend, the keyframe image sha1, and the fill-rule version.

---

## 4. Approaches researched and prototyped

Every candidate was rendered through one deterministic judging instrument from
the same fixed cameras, including the wearer's own recorded poses. The
instrument is `viewkit/`, a torch/CUDA software rasteriser with a self-test. Two
bugs in viewkit itself were found and fixed first (E0).

| Approach | Tool | Result | Decision |
|---|---|---|---|
| Sparse SfM (baseline) | existing | 14,415 dots; nothing recognisable | Diagnostics only |
| Dense monocular-depth points | prior lane: MoGe-2 + IRLS + 3-camera consensus | 12.7M points, 272 s. Photographic from the capture's own vantage, torn slabs from anywhere else. Consensus deletes 40–55% of plain wall | Kept as the depth stage and a fallback rung |
| **TSDF surface on that depth** | `surface.py` | Occluding surfaces, readable from arbitrary viewpoints | **Adopted** |
| CUDA PatchMatch MVS, as anchors or extra depth | COLMAP 4.2.0 CUDA | Better depth where it exists (r 0.816) but **no floor or plain wall**; ~2.5 GPU-h per walk; needs unredacted images | Rejected (`mvs/MVS-HYBRID.md`) |
| Gaussian splatting | Brush (prior lane), gsplat | Paints plausible surfaces over everything that moved. gsplat is not buildable here (no MSVC; CUDA 11.8 vs sm_120) | Rejected |
| Poisson / Delaunay / ball pivoting | open3d | Closes unobserved space by construction | Rejected on principle |
| Enclosed-hole fill in the TSDF | opt-in | +0.25% faces, no visible change; the evidence filter removes all of it | Off |
| Learned multi-view depth (VGGT, MASt3R, DA3 pose-conditioned) | toolchain survey | VGGT 9.8 GB at 16 images; DA3 worse than monocular here | Rejected |

Depth backbone bake-off, scored against CUDA MVS:

- **MoGe-2 ViT-L (chosen):** r 0.835, AbsRel 0.033, δ<1.25 0.968, 131 ms/frame, MIT licence.
- **UniDepthV2:** 3.1× faster but CC BY-NC.

## 5. Benchmark table

Same cameras, same instrument. Canonical world, uncontended unless marked.

| Candidate | Faces / points | Build | Recognisable from novel views (9 non-campaign cameras + desk-into-room) |
|---|---|---|---|
| Sparse SfM | 14k pts | 96 s solve | 0/10 |
| Dense points (prior lane) | 12.7M pts | 272 s | only near the capture poses; slabs elsewhere |
| TSDF, first production (`97e75eb`) | 2.21M faces | 49–52 s (cached depth) | desk in 3 views, room in 1 (review-recon) |
| TSDF + evidence (`888ac69`) | 2.67M faces | 62.4 s (cached depth) | desk/room corner/door in N1, N6, N7, N9; walls continuous |
| **Final (`f8aecba`+)**, from scratch incl. depth network | 2.70M faces L0 / 229k phone | **155 s** under contention (depth 76.5, fuse 13.6, mesh 19.7, pack 44.3) | same as `888ac69`, within 1.7 points of coverage per view |

Detailed measurement of each evidence change (`surface-r2\SURFACE-R2.md`):

| Metric | Shipped `1e1cdc8` | Committed `888ac69` |
|---|---|---|
| L0 area (units²) | 369 | 456 |
| Supported by ≥2 frames | 99.3% | 98.8% |
| Contradicted 2:1 by see-through frames | 9.5% | 2.2% |
| Back-facing area | 17.1% | 0.6% |
| Faces within 0.5 units of the walked path (hand, lap) | 2,330 | 0 |
| Vertices on edges with a never-observed end | 8,513 | 0 |
| Measured depth with no surface, in range | 24.8% | 25.2% |

## 6. Rejected approaches, and why they stay rejected

- **MVS:** it adds no floor or plain wall, costs GPU-hours, and needs unredacted
  imagery.
- **Splatting:** it invents surfaces where things moved, and it can't be built
  on this machine.
- **Watertight reconstruction:** it closes unobserved space, which violates
  contract §3.
- **A near-depth weight clamp of 1:** it cost 16% of the area, all of it
  supported and uncontradicted.
- **An uncapped depth-proportional truncation band:** it goes over the block
  budget and coarsens every voxel ×1.2.
- **A global 2.6× median depth clip:** it deleted the far wall and door that
  45+ frames saw.

---

## 7. Final architecture

```
keyframes (redacted) ──► depth stage (dense_pipeline.run_depth_stage)
                          MoGe-2 per keyframe, cached <ki>_pred.npy,
                            reused only for the same keyframe image (sha1), backend, fill rule
                          fill mask: exact against the raw frame when readable, else
                            the diff unioned with the shape-gated guess (FILL_RULE 2)
                          robust affine fit to THIS solve's sparse points, held-out gate 8%
                               │  align.json names the solve
                               ▼
global solve poses ───► surface stage (surface_pipeline.surfacify)
                          scale = median depth of the solve's sparse observations
                          validity: depth edges, grazing > 80°, redaction fill (+8 px),
                            depth ≤ 1.5 × that frame's farthest sparse anchor
                          weight: incidence × gate score × depth falloff (near boost ≤ 4)
                          block-sparse TSDF, voxel 0.0051 × scale; band 3–12 voxels, grows with depth
                          free-space carving; block budget 360k (coarsen ≥10%/round, refuse at the end)
                          marching cubes: a cube only if all 8 corners reach min_weight
                          weld → evidence filter per face (≥2 frames, not 2:1 contradicted,
                            front seen) → drop tiny components → Taubin ×6
                          LOD cascade: L0 archive, L1 600k, L2 phone (decimated until the PAGE fits 6 MiB)
                               │  per-build level files, manifest last
                               ▼
<world>/surface/<sid>/ ─► render route ladder surface → dense → sparse
                          page: WebGL2, mesh inline, `up` refined on the surface's floors/walls,
                          opens on the camera pose that sees the most surface
                          /render/revision {revision, representation, live}
```

## 8. Live architecture

- **When live builds run.** `world_build_session.py` runs a background global
  solve every 50 keyframes. `BackgroundSurface` launches
  `world_surface.py --live --force` when a solve lands. It never builds on a
  keyframe count, because pre-solve segment chains place one wall in several
  places. It builds again only if `solution.json` actually changed.
- **Resource isolation.**
  - One live build at a time, at BELOW_NORMAL priority.
  - It is closed at the end of the walk.
  - A missing depth network is a permanent state (exit 4) that disables the
    worker.
- **Live preset.** Voxel ×2, `min_weight` 1.5, lighter smoothing. The evidence
  rules are the same as the final build.
- **After Stop.** The final surface is built BEFORE densify, from the final
  solve.
- **Phone.**
  - Polls `/render/revision` every 10 s while `live` is true, backing off to
    120 s when it is false.
  - Swaps by itself only to a better rung, or to a finished (`live:false`) build.
  - Offers a live rebuild with "A newer reconstruction is ready. Show it".
  - Reverts to the last drawn page if a new one fails to draw.

## 9. Live replay evidence

Real-time re-recording of the canonical walk's two raw captures through
`CaptureRecorder`. It is followed by the production builder, run with
`--solve --solve-every 50 --surface --rebuild-every 4`.

**Run D** (`replay\runD\LIVE-REPLAY-D.md`, on `888ac69`, under a concurrent full
test suite):

| # | quality | solve (poses) | landed s | surface ready s | latency | frames | faces |
|---|---|---|---:|---:|---:|---|---:|
| 1 | live | 51 | 34.3 | 64.6 | 30.3 | 51/51 | 92,319 |
| 2 | live | 101 | 67.3 | 105.5 | 38.2 | 98/101 | 177,050 |
| 3 | live | 170 | 97.7 | 142.5 | 44.8 | 147/151 | 451,234 |
| final | final | 395 | 215.8 | 331.7 | — | 368/395 | 2,373,560 |

- **The walk:** 6.2–146.2 s. Stop to final surface: 185.5 s (run C: 258.5 s).
- **Whole files:** all 1,118 samples of level files during builds were whole
  and named by their manifest.
- **The `live` flag:** true from the first geometry through the final stage, and
  false in the same 1.26 s sample in which the final revision appeared. No gap
  was seen.
- **Progression strip** (`replay\runD\surface_strip.jpg`): each step is visibly
  better: back wall and pictures, then door panel and ceiling edge, then a
  continuous wall and full door.
- **Comparison with run C** (`a9f8600`, less contention, `replay\LIVE-REPLAY.md`):
  - run C had 4 live builds and a 27 s median latency;
  - run D has 3 and 38 s. Part of that is contention (CPU median 73% vs 48%).
    The rest is the per-face evidence filter: the live mesh stage went from
    0.9 s to 5.1 s at ~100 frames.
  - The run D final looks clearly more like the room (`replay\runD\finals_C_vs_D.jpg`).
- **Not exercised:** closing a live child that is still running at Stop was not
  exercised in run D (none was running). Run C exercised it before `17dfa50`,
  and its unit test pins it.

## 10. GPU, runtime and storage (RTX 5070, 12 GB)

| World | Keyframes | Build | Peak VRAM | L0 faces | Notes |
|---|---:|---|---|---:|---|
| canonical, final code, from scratch | 398 | 155 s (depth 76.5 / fuse 13.6 / mesh 19.7 / pack 44.3) | +6.3 GiB device-wide (depth 3.6 GiB; fuse/mesh ~6.6 GiB incl. allocator cache) | 2.70M | contended by the pytest suite |
| canonical, cached depth (`888ac69`) | 398 | 62.4 s | 3.00 GiB | 2.67M | |
| canonical rerun, nothing changed | 398 | 0.17 s | — | — | "already built" |
| `1b8812b1` over budget | 438 | 36.9 s | 2.63 GiB | 1.51M | voxel coarsened ×1.18 (`97e75eb`) |
| `72bb4b9b` largest usable | 583 | 181 s | — | 1.61M | depth 123 s |
| `52ed8e0a` field walk | 795 | 28.2 s | — | — | mesh 519.6 s → 9.5 s after `421c8c4` |

**Storage per world** (canonical):

| Artifact | Size |
|---|---|
| L0 mesh | 50.5 MB |
| L1 mesh | 11.7 MB |
| L2 mesh | 4.6 MB |
| Depth work directory (undistorted JPEGs, float16 predictions, fill masks) | ~270 MB, prunable and regenerated on demand |
| Phone page | 6.17 MB |
| Diagnostics (sparse) page | 0.60 MB |

**GPU sharing.** The surface stage runs inside the World Builder builder's
process tree, which the Tower's Job Object owns. It runs only while a walk is
active, plus the final build after Stop. Live children run at below-normal
priority, and one live build runs at a time.

## 11. Rebuild procedure

From `tower/` in a checkout of the final SHA, with the Tower venv:

```
python scripts/world_surface.py --root <world root> --list
python scripts/world_surface.py --root <world root> --world <wid> --session <sid>          # builds if absent or stale
python scripts/world_surface.py --root <world root> --world <wid> --session <sid> --force  # rebuild
python scripts/world_surface.py --root <world root> --world <wid> --session <sid> --inspect
python scripts/world_surface.py --root <world root> --all
python scripts/world_surface.py --root <world root> --world <wid> --session <sid> --export-obj out.obj
```

- `--root` goes through `artifact_root_arg`, so it must be absolute or under an
  approved location.
- The build adds `dense/<sid>/` and `surface/<sid>/` beside the world's data,
  and never modifies other files.
- Exit codes: 0 ok, 4 = cannot run on this machine (e.g. no depth network).
- On a fresh machine the first build downloads MoGe-2 ViT-L (~1.3 GB) from the
  Hugging Face hub; pre-seed the cache before an offline retest.

## 12. Visual evidence and the canonical result

All paths are under `Glasses-scratch\wb-final-recon\`. I looked at every image
listed here myself.

| What | Image |
|---|---|
| BEFORE: sparse SfM | `before/before_sparse.png`; `final-evidence/fig/diagnostics_1200x900.png` |
| First shipped surface, novel views | `review-recon/novel_l0.jpg` |
| After the evidence changes, novel views | `surface-r2/fig/novel_final_l0.jpg` |
| **Final**, novel views L0 / clay / phone level | `final-evidence/fig/novel_final.jpg`, `novel_final_clay.jpg`, `novel_final_l2.jpg` |
| **The actual phone page**, opening view, levelled | `final-evidence/fig/page_r4_up_500x900.png`, `page_r4_up_1200x900.png` |
| Phone page before the horizon fix (rolled ~20°) | `final-evidence/fig/page_open_500x900.png` |
| Phone page walk poses and orbit | `final-evidence/fig/page_walk_*.png`, `page_orbit_*.png` |
| Live progression during a real-time replay | `replay/runD/surface_strip.jpg`, `replay/runD/faces_over_time.png` |
| Largest world | `scaling/render/w583_72bb4b9b_l0.jpg` |

**Novel views on the final build**, from cameras the campaign did not choose:

- **N1, room centre to desk.** Clearly a desk. Hutch with three shelves, cups and
  a box; a monitor with UI rows; a purple keyboard; a red mouse; a glowing PC
  case. The right-hand wall is continuous.
- **N6, high corner; N7, bedside; N9, doorway.** A room corner with the desk,
  monitor, pictures, ceiling strip, open white door, planked floor and bed.
  N9 is the most room-like.
- **N3, grazing along the desk wall.** Still a crumpled, faceted sheet.
- **N4, behind the wall.** The wall has no back, so you see through to the room.
- **N5, floor level.** Floaters and a dark ridge across the floor.
- **N8, over the bed.** The bed is a sheet with holes.
- **N2 and N10, desk into the room.** Almost nothing, because that side was
  essentially never captured.
- **Clay render.** The shape alone reads as the desk: crisp shelves, cylindrical
  cups, a flat monitor. The pictures vanish, because they are only colour.
- **Phone level.** Same framing. The monitor becomes a white-blue blob, the
  keyboard a plain purple bar, and shelf objects soft lumps.

## 13. Scaling

`scaling/SCALING.md`, synthetic walks up to 8×:

- Block count follows surface area, not frame count.
- The per-frame frustum cull is negligible (1.03 s at 1.1M blocks).
- Tile iteration over the bounding grid was cubic; fixed by iterating occupied
  tiles only.
- Decimation grows as faces^1.37; mitigated by the LOD cascade.
- Before the block budget, a 20–30 minute walk projected to exhaust VRAM past
  ~900 frames. Now the voxel coarsens by ≥10% per round. If the budget can't be
  met, the build refuses loudly and the previous surface stands.

The evidence filter is per face × supporting frames. It was checked for memory
on the canonical world, but it has not been measured on the 795-keyframe walk
at the final SHA.

## 14. Mobile considerations

- **Budget.** The page, not the mesh, is bounded at 6 MiB. The canonical page is
  5.89 MiB, and the phone level is decimated until the composed page fits.
- **What the page does on a phone.**
  - Dequantises in the vertex shader.
  - Caps DPR at 2.5.
  - Recovers from WebGL context loss.
  - Uses pointer events with `touch-action:none`.
  - Carries a CSP meta tag (all three rung pages now do).
  - Measures exposure and applies a tone curve.
- **Memory at swap time.** The app briefly holds the old page, the new page and
  the IPC copy (~4 × 6 MB), plus GL buffers. Measuring this is part of the Mac
  validation.
- **Content-process kills.** They count within a 60 s window, so kills spread
  over a long walk don't exhaust the reload budget.
- **Dense rung.** The dense page (8.03 MB) is not bounded by the budget. It is
  not the default rung (documented as a known limitation in `WORLD-BUILDER-WORLDS.md`; densify is off by default).

## 15. Data integrity and the artifact format

`docs/contracts/WORLD-BUILDER-SURFACE.md`.

**Format `wb-surface-mesh/1`:**

- a 48-byte header: magic `WBSURF01`, counts, flags, schema, bbox;
- uint16 quantised positions, uint8 colour, int8 normals, u16/u32 indices.

The reader refuses:

- a wrong magic or an unknown schema;
- any length other than what the header implies;
- an index count that isn't whole triangles;
- an index out of range.

An empty mesh is legal.

**Publishing and locking:**

- Everything is published atomically, as per-build level files with the
  manifest written last.
- One build per session: `.surface.lock`, O_EXCL, with liveness checked against
  pid AND written-at time so a recycled pid can't hold it.
- A stop is honoured through the whole pack.

**What the manifest records:** `input_digest` (the solve), `params_digest`
(including the backend and phone page budget), `detail` (evidence filter counts,
`mobile_page_fit`, weld, components), and `scale` (state `unknown`: nothing
claims metres).

## 16. Tests

- **Tower, new:**
  - `test_world_builder_surface.py`: format, field, carving, truncation, gauge
    invariance, weld, occupied tiles, allocation;
  - `test_world_builder_surface_pipeline.py`: end-to-end `surfacify` on a
    ray-cast synthetic world, unit publish, stop, locks, cache identity, image
    hash, fill rule, prune grace, budget refusal, live cadence, "already built";
  - `test_world_builder_surface_evidence.py`: phantom wall, far wall kept,
    stable scale, single-frame ghost, see-through ghost, back-facing, thin board;
  - `test_world_builder_surface_up.py`;
  - `test_world_builder_surface_fill.py`;
  - `test_world_builder_render_revision_live.py`;
  - redaction tests: plausibility gates, fail-closed, sticky label, re-find
    scales.
- **Mutation checks.** Every review-round fix was checked by reverting it in a
  scratch copy: surface-r2 5/5, round-2 fixes 14/14, and redaction 4 and 8
  tests. Each reverted copy fails its test.
- **Swift** (`WorldRenderRepresentationTests`, `WorldRenderRevisionTests`, the
  UI smoke test's rung-caption wait): **written, never compiled or run**.

## 17. Regression results

| Run | Result |
|---|---|
| Full Tower suite at `17dfa50` | 3402 passed, 1 failed (fixed `2e3ee5d`) |
| Full Tower suite at `888ac69` | **3458 passed, 76 skipped, 1 xfailed, 0 failed** |
| Full Tower suite at `f8aecba` | **3486 passed, 76 skipped, 1 xfailed, 0 failed** |
| Full Tower suite at final SHA | **3501 passed, 76 skipped, 1 xfailed, 0 failed** (`bb15a77`, `fullsuite-final.log`) |
| Reviewer repros, round 1 (17) | 14 pass; 3 explained (they now correctly attempt a rebuild their own fixture can't complete, or are perf-only) |
| Reviewer repros, round 2 (12) at `f8aecba` | 10 pass; I4 and I5 remain (§19) |

## 18. Independent reviews

Each round was a fresh adversarial reviewer that had not written the code.

| Round | Scope | Found | Disposition |
|---|---|---|---|
| 1 systems (`review-systems/REVIEW.md`) | integrity, caches, locks, live wiring | torn level sets, stale/foreign caches, pid reuse, stop in pack, revision races, diagnostics | fixed (`e5f72f3` etc.) |
| 1 iOS (`review-ios/REVIEW.md`) | Swift, contracts | 5 MAJOR: diagnostics unreachable, failed draw lost the world, camera reset every solve, 404 ended following, lifetime reload budget | fixed (`f4eac09`, `fe84ee3`, `701e8f6`) |
| 1 truthfulness (`review-recon/REVIEW.md`) | geometry vs evidence | BLOCKING: depth clip deleted measured walls; unstable scale. MAJOR: phantom faces, ghosts, band, back-folds; caption false | fixed in surface-r2 (`687d47a`, `e2aa002`) |
| 1 redaction (`redaction/REDACTION.md` §12) | fail-open paths | a failed frame laundered by the next; unjudgeable landmarks skipped | fixed (`53be0c3`); the >25% rule revisited twice (§20) |
| 2 (`review2/REVIEW.md`) | everything since 97e75eb | MAJOR: live fill mask lost engine fill; iOS caption; §2.3 overclaim. 20 minor | fixed (`843f2dc`…`f8aecba`) |
| Live replay D | end to end | page 7.89 MB over budget; redaction blacked out walls | fixed (`f8aecba`, `b1accdb`) |
| Canonical rebuild | visual | opening view rolled ~20° | fixed (`3b5ea60`) |
| 3 (`review3/REVIEW.md`) | everything since 888ac69 + e2e default path | MAJOR: off-frame large faces unfilled; iOS refusal could hide the final; diagnostics served dense; Mac doc stale | fixed (`defc9c1`, `7b95d6e`, `7b35fcf`, `b4444aa`, `481da3a`); I4, I5, R6 and iOS m1/m3/m7 left as limitations (§19) |

## 19. Remaining limitations

1. **Coverage is capture-limited.**
   - The canonical world is one side of a room, and the view from the desk back
     into the room is almost empty.
   - The floor ceiling reachable by any parameter is 33.4%: 87% of floor rays
     are blocked by the furniture the wearer sat among (`coverage/COVERAGE-FORENSICS.md`).
   - Only walking the room changes this. A live coverage cue is the natural
     next product step.
2. **About a quarter of measured depth yields no surface,** because frames
   disagreed about it (monocular depth inconsistency). The caption says a gap
   is not proof that nothing is there.
3. **Thin structures.** They can be deleted when distant frames' depth misses
   them (S3), and a hand seen by more frames than saw past it survives (S2).
   Both are stated in contract §2.
4. **The phone level loses fine detail** (monitor contents, keys).
5. **Live latency.** Median 27–38 s after a solve lands. The first live surface
   appears ~60 s into a walk.
6. **Scale is unknown,** and the vertical is an estimate.
7. **Redaction false positives.** Keyframe 2054 (a wall and a PC tower) is
   79% black, and keyframe 188 is 93% black.
   - Under `plausibility3` about 6% of captured pixels are filled, most of it
     false large boxes at frame edges, kept on purpose so faces cut by the
     edge are not missed.
   - The canonical world's stored keyframes carry the redaction of the Tower
     that captured them. New captures get the new gate.
8. **Open review items:**
   - I4: a stopped depth stage drops the reuse map (costs recomputation only).
   - I5: two processes can both reclaim a stale lock (narrowed; production
     builders run one after another).
   - R6: I5 combined with a pack over 120 s could sweep the other build's
     unpublished files.
   - iOS m1: a failed "Show it" loses the offer.
   - iOS m3: the kill window lets a page dying every 35 s reload forever.
   - iOS m7: repeated failed page fetches defeat the backoff.
9. **Nothing iOS was compiled or run,** and WKWebView behaviour is unmeasured
   on a device.
10. **First run on a new machine** downloads the depth model (~1.3 GB).

## 20. iOS awaiting Mac, and the decisions made

- **Changed:**
  - `ios/Glasses/Workspaces/WorldBuilder/WorldRenderViewer.swift`: caption by
    rung, the revision follower, fallback, offers, per-rung refusal,
    termination window;
  - `ios/GlassesTests/WorldBuilderIntegrationTests.swift`;
  - `ios/GlassesTests/WorldPresentationTests.swift`;
  - `ios/GlassesUITests/TowerSmokeUITests.swift`.
- **Compile risk:** two independent readings rated it LOW. Nothing was built.
- **Decision: swap versus offer.** A better rung swaps by itself. A finished
  build swaps by itself. A live same-rung rebuild is offered, because a swap
  reloads the page and resets the camera.
- **Decision: redaction of frame-sized boxes** (Tower, but privacy-visible):
  - the landmark "facelike" test was shown to accept 40/40 false large boxes,
    so it is not evidence at that size;
  - large boxes are filled when re-found at another resolution;
  - measured on 398 raw frames: fill 2.72% (the rejected OR rule gave 7.51%),
    65/65 close faces filled;
  -   - review round 3 then showed that close faces cut by the frame edge
    were lost (on 3,232 held-out off-frame composites: 32 lost under
    `plausibility2`);
  - `plausibility3` fills a facelike box within 5% of an edge on its
    landmarks: 1/3,232 lost (its box is misplaced away from the cut edge),
    65/65 close faces, but raw fill back up to **5.99%** (13 frames refused,
    worst frame 93.2%), because 25 of the 40 false large boxes touch an edge;
  - privacy was chosen over the pixels; a variant with low-confidence
    re-detection loses 0/3,232 at 6.58% fill and more over-filled frames.
    Both are in `redaction\REDACTION.md` §14.

## 21. Mac validation and physical retest procedure

The full Mac checklist: `docs/agent-handoffs/WORLD-BUILDER-VIEWER-MAC-VALIDATION.md`.

**Before the phone test (Windows Tower):**

1. Check out the final SHA of `world-builder/reconstruction-final-v1`, in a
   worktree or by integration into the branch the Tower runs from. Nothing is
   pushed.
2. Pre-seed the depth model on the Tower machine, so the first build doesn't
   download it:
   `python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Ruicheng/moge-2-vitl', 'model.pt'))"`
   (the command in the Mac checklist; or run step 3 once while online).
3. Build the saved world's surface. This is additive and writes only `dense/`
   and `surface/`:
   `python scripts/world_surface.py --root C:\Users\tvllo\Projects\Glasses\tower\data\world_builder --world b2a75ab40d2d415d8d6ef5e4d5f0fb3d --session a8c6817e14a74e3c977fccfcdacad595`
   Expect ~155 s and "L2: ~229k faces". Rerun it: it prints "already built".
4. Confirm the Tower serves the page: `GET /worlds/b2a75ab4…/render` has
   `<meta name="wb-representation" content="surface">` near the top, and
   `/render/revision` answers `{"representation":"surface","live":false}`.
   `TOWER_WORLD_SURFACE` defaults on.

**Mac:**

5. Build the app and run `GlassesTests` (`WorldRenderRepresentationTests`,
   `WorldRenderRevisionTests`) and the UI smoke test, per the checklist.

**Phone, with the saved world:**

6. Saved Worlds → b2a75ab4. Expect:
   - a levelled bedroom corner with the door;
   - the caption "Surfaces the Tower reconstructed…";
   - 229k triangles in the page caption.
7. Orbit, Walk, and step poses. Record time to first draw and WebContent memory.
8. World screen → Diagnostics → "Open the solver's 3D view": segment-coloured
   dots, not the surface.

**Phone, physical walk** (glasses on; `TOWER_WORLD_AUTOBUILD=true`):

9. Walk the room for 3–5 minutes, **turning to face every wall, not only the
   desk**. Open the Picture early and keep it open.
   - Within ~60–90 s the dots are replaced by a coarse surface without a tap.
   - Later rebuilds show "A newer reconstruction is ready. Show it".
   - Tap it at least once.
10. Stop. Keep the Picture open. Within ~3 minutes the full surface replaces the
    live one **without a tap**. The poll interval then backs off to 120 s.
11. Reopen from Saved Worlds: the final surface opens.
12. Record on the Tower:
    - `surface/<sid>/manifest.json` (`frames_used`, `detail.mobile_page_fit`);
    - the builder log's live build times;
    - any `unavailable` status.

**Physical live validation has not happened.** It is claimed nowhere in this
document.

## 22. Temporary resources and privacy notes

- **Worktrees:** `Glasses-worktrees\wb-final-recon` (lane) and
  `wb-final-recon-r1` … `r4` (review lanes, all merged), with their branches.
  Nothing was removed.
- **Scratch:** `Glasses-scratch\wb-final-recon\` (~21 GB). **It contains
  unredacted first-person imagery**:
  - `mvs\dense\images\`;
  - world copies under `worlds\`, `scaling\worlds\`, `surface-r2\world\`,
    `final-evidence\world\`, `review3\e2e\`, `replay\run*\root\`;
  - synthetic composited faces in `redaction\` and `review3\repro\lost_*.jpg`.

  Two large files: `review-recon\field.pt` (960 MB) and
  `fix-r3\runD-page\root\` (166 MB). Nothing was deleted; deletion needs
  Tristan's approval.
- **Pre-existing, found by forensics, not changed:** every world's
  `solve/<sid>/images/` is **unredacted**, built from raw capture frames.
- **Read-side effects:** zero-byte and 32 KB `database.db-wal`/`-shm` sidecars
  were created beside canonical `database.db` files by a WAL-mode read. The
  databases are unchanged.
- **Tooling:** the COLMAP CUDA binary is in `Glasses-scratch\wb-final-recon\toolchain\bin\`.
- **Processes:** all temporary http.server processes and Chrome profiles used
  for page renders were stopped. The Chrome profiles remain under scratch.
- **Auto-memory notes:** `wb-reconstruction-final-lane.md`,
  `webgl-verification-on-windows.md` and `wb-surface-r2-lane.md`.

## 23. "If Tristan puts on the glasses tomorrow, what is the most likely reason he will still say *This isn't a 3D world*?"

**He'll turn the phone toward the side of the room he didn't look at, and find
black.**

- The surface is real where he looked: walls, desk, shelves, door, from any
  angle. But the pipeline refuses to invent anything, and a seated walk covers
  one side of a room.
- The canonical walk shows exactly this: the view from the desk back into the
  room is 9–17% drawn.
- On the phone, even the good side has ragged dark edges. At 229k triangles the
  monitor is a glow and the keyboard has no keys.

The retest procedure tells him to walk the room and face every wall. If he does,
the gap closes; if he sits at the desk again, it won't. The second most likely
complaint is the monocular-depth disagreement holes.

## 24. Final verdict

READY FOR MAC/PHYSICAL RETEST
