# World Builder — the global solver (2026-09-06)

**Branch:** `world-builder/global-reconstruction-v1` (from `integration/all-current-v1 @ a7b1c2a`)
**Worktree:** `C:\Users\tvllo\Projects\Glasses-worktrees\wb-recon`
**Ledger (the evidence):** `tower/docs/world-builder-reconstruction-experiments.md`
**Design record:** `tower/docs/superpowers/specs/2026-09-06-world-builder-global-solver-design.md`
**Contracts touched:** `docs/contracts/WORLD-BUILDER-GEOMETRY.md` §8 (additive), `docs/contracts/WORLD-BUILDER-WORLDS.md` (new)
**Status:** Tower implemented, tested, replayed on six walks. iOS written, **not compiled** (no toolchain here).

---

## 1. What was wrong

The 2026-09-06 live test (438 keyframes, 34 segments) put 58 keyframes into
one coordinate frame. Three independent audits of that walk agree on why
(`Glasses-scratch\wbrecon\{tracking,registration}\REPORT.md`):

1. **The per-segment chain refuses or strands half the walk.** 210 of 438
   poses refused, 185 of them cascades from 25 root refusals through a
   barren latch; segment 4 has 74 keyframes and no geometry although its
   images link strongly to its neighbours. Face redaction is the largest
   single upstream factor: YuNet at 0.30 blacks out desks and floors in an
   empty room, and the identical code with the redactor off solves 294
   poses instead of 194.
2. **The segments that do reconstruct are not rigid.** One segment's scale
   drops ~10× along its own length. No Sim3 fits that, so the registrar's
   refusal of (16,17) — 226 verified frame pairs, 32,101 inliers, the same
   desk — is *correct*. Registration was never the bottleneck.
3. **The features were there.** Sequential SIFT matching over the same 438
   frames: 8,550 verified pairs, 5,559 with ≥ 15 inliers.

## 2. What was built

`tower/tower/world_builder/global_solve.py` — a global structure-from-motion
solve over every keyframe of a session:

```
raw capture frame (else the redacted keyframe copy)
  -> undistort once with the session calibration (pinhole, valid-ROI crop)
  -> SIFT (pycolmap, CPU)
  -> sequential matching, overlap 20; + vocabulary-tree loop detection at finalisation
  -> GLOMAP (rotation averaging -> global positioning -> BA), intrinsics fixed;
     incremental mapping if GLOMAP yields nothing
  -> support floor: a camera with < 30 3-D observations is unplaced
```

It reaches the phone through the geometry contract's existing layers:
`merge()` rewrites each solved segment's poses/points in the segment's own
frame and writes a `registered` placement (scale exactly 1) into the
component's reference segment. `engine.build()` is still the single writer of
the derived tree; it merges a persisted solution when one exists.

Live: `scripts/world_build_session.py --solve` launches
`scripts/world_solve.py` as a child every 50 accepted keyframes (features and
matches persist, so each solve pays only for new keyframes) and runs a final
in-process solve after Stop. A finished background solve triggers a rebuild,
so the phone sees segments snap together mid-walk. The Tower's builder worker
gets `--solve` from `config.world_solve` / `TOWER_WORLD_SOLVE` (default on).
The Sim3 registrar (`--register`) stands down whenever a solution exists.
`pycolmap` is optional (`pip install .[sfm]`); without it the World Builder
behaves exactly as before and the report says why.

Also: `GET /worlds` (saved-world listing), per-segment `coverage` on the
manifest (`confident | partial | unresolved | null`), `scripts/world_render.py`
(PLY / PNG / interactive HTML of a world, unregistered fragments kept apart),
the coherence report's `global_solve` block (the solver's own reprojection).

## 3. Results (product path, replayed; ledger E14)

| walk | before: keyframes in one frame / points share | after |
|---|---|---|
| **2026-09-06** | 58 / 36.7% | **425 of 438 posed, 29 of 34 segments registered, 99.1%** |
| 2026-09-01 loop | 156 / 74.9% | 424 of 434, 100% (the loop closed) |
| worldB drawer | 61 / 58.1% | 207 of 218, 97.1% |
| worldA normal | 51 / 41.3% | 203 of 229, 90.3% |
| dense | 59 / 84.1% | 76 of 77, 100% |
| long 08-27 | 27 / 19.7% | 322 of 339, 97.0% |

Solver reprojection medians 0.59–0.75 px, p99 ≤ 4.2 px, on every walk.
Renders: `Glasses-scratch\wbrecon\final\ARTIFACTS.md`.

## 4. How to replay

```sh
cd C:/Users/tvllo/Projects/Glasses-worktrees/wb-recon/tower
python scripts/world_replay.py --captures ddcf9426636042aaa37c7e7d9d0ed074 \
  --capture-root C:/Users/tvllo/Projects/Glasses/tower/data/captures \
  --intrinsics-from C:/Users/tvllo/Projects/Glasses/tower/data/world_builder/intrinsics \
  --root C:/Users/tvllo/Projects/Glasses-scratch/<fresh dir> --solve --register --format json
python scripts/world_coherence_report.py --root <that root> --format json
python scripts/world_render.py --world-root <that root> --world <world id> --out <dir>
# the solver alone, on an existing world (writes only solve/, never derived/):
python scripts/world_solve.py --root <root> --world <id> --session <id> --capture-dir <capture dir> --final --loop-detection
```

## 5. What is deliberately unchanged

The fast path (frame ingest, keyframe selection, the local chain and its
bundle adjustment) and every acceptance threshold in it. The redactor and
its 0.30 confidence. Capture resolution. No metric scale. No learned
features or feed-forward 3-D in the product path.

## 6. Limitations, precisely

- Cameras posed on far, low-parallax structure can sit far outside the room
  (long0827: a handful at 100 units). They pass the support floor; a
  per-camera triangulation-angle floor is the next instrument.
- The 09-01 walk's start/end and middle join only with loop detection; a
  walk with no revisits and a transport gap will stay two frames, reported
  as two.
- GLOMAP will not take the 8-parameter distortion model; undistorting first
  is the fix and costs the image border (335×595 of 360×640).
- The `--solve` final pass takes 30–135 s on this host for a two-minute
  walk. `CaptureWorkerSupervisor.shutdown` still gives the builder 10 s;
  a Tower stopped inside that window keeps the last merged solution but
  loses the final one. Unchanged from the registrar's own risk.
- `world_replay.py`'s pinned CASES figures (worldA/worldB) were already
  stale on the parent branch; not re-recorded here.
- Support rows (`support.json`) are not written for solved segments; the
  solver's observations live in `solve/<session>/solution.npz` and are
  checked by `reprojection_summary`.
- The pycolmap wheel bundles SuiteSparse (CHOLMOD has GPL modules): for
  product licensing review.

## 7. iOS

See `Glasses-scratch\wbrecon\ios\MAC-VALIDATION.md` and the final report for
what was written and how to validate it on a Mac. Nothing here has been
compiled.

## 8. Temporary resources (filesystem policy rule 9)

Worktree `Glasses-worktrees\wb-recon` (this branch). Scratch, all under
`Glasses-scratch\wbrecon\`: `inventory`, `tracking` (19 replay roots,
455 MB), `registration`, `research`, `install` (trial venv + VGGT weights,
14 GB — disposable), `baseline` (replay roots), `exp` (solver experiments),
`smoke`, `live`, `final`, `render`, `pytest-tmp`, `pytest-full`. Nothing
deleted; nothing written under `tower/data`. pycolmap 4.2.0 was installed into
`tower/.venv`.
