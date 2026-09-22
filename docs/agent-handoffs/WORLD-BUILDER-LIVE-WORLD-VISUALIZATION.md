# World Builder — Live World Visualization (V1 integration/validation)

Campaign date: 2026-09-22. Lane: `world-builder/live-world-visualization-v1`.
Worktree: `C:\Users\tvllo\Projects\Glasses-worktrees\wb-live-visualization-v1`.

## Mission

A live glasses capture must end in a Saved World that the iPhone opens as a
recognizable **photographic** reconstruction, reached through the normal
product path — no scripts, no hand-copied assets, no debug viewer.

Today the phone opens Saved Worlds into the legacy sparse point-cloud
presentation.

## Verified starting state (2026-09-22)

- Canonical checkout `C:\Users\tvllo\Projects\Glasses`, branch `main`,
  HEAD `8d0f1af`, equal to `origin/main`. Clean but for three untracked
  files (`WORK-SAMPLE.pdf`, `WORK-SAMPLE.tex`, `orb_vocab_stella.fbow`).
- Lane worktree created from `main` at `8d0f1af`; verified WORKTREE, not
  CANONICAL, per the repository lane-isolation policy.
- Tower venv `tower/.venv` (Python 3.12.5) has every reconstruction
  dependency installed: pycolmap 4.2.0, torch 2.13.0+cu132 on an
  RTX 5070 (CUDA available), MoGe, open3d 0.19, trimesh, xatlas,
  astc-encoder-py 0.1.12, transformers 5.16.1. Nothing needs installing
  for the rich pipelines to run.

## Evidence gathered before any change

### The 2026-09-22 live world

World `6839fb8f490a49e6b4ecfac6eb4f32b7`, session
`060d7223d0634603a0e1e0e104ddca28`, 179 MB on disk.

Its `derived/manifest.json` records a healthy reconstruction:
690 keyframes, 607 poses solved, 31,127 points, 70 segments, GLOMAP global
solve completed at 1790055645, 64 segments replaced, 6 refused, 0 pending,
four components with component 0 holding 617 keyframes and 29,020 points.

Its `derived/<session>/` contains exactly five files:
`manifest.json`, `placements.json`, `points.json`, `poses.json`,
`support.json`. **No dense, surface, or appearance artifact of any kind.**

### The comparison world

`b2a75ab40d2d415d8d6ef5e4d5f0fb3d` — the world the September surface
campaign worked on — has the *same* five sparse files and nothing else.
Whatever rich renders prior campaigns produced did not land in the world
store; they were written elsewhere by offline tooling.

This is the first strong signal for root-cause category **A** (the rich
representation is never generated in the live path), but it is not yet
proof, and the investigation below tests it rather than assuming it.

## Root cause — PROVEN, and it is not the one the brief assumed

The brief asked which of categories A–J explains the sparse viewer. The
answer is a version-deployment cause that none of them names cleanly, and
it is provable from the reflog rather than inferred.

### The serving path was never broken

The API investigator's census of the canonical world root is decisive:

| artifact directory | worlds having it (of 165) |
|---|---|
| `derived/` (sparse) | 51 |
| `solve/` | 10 |
| `dense/` | **0** |
| `surface/` | **0** |
| `appearance/` | **0** |

Every mechanism above the artifact is already correct and already shipped:

- `tower/tower/results/world_builder_render.py:212` defines the ladder
  `appearance → surface → dense → sparse`, walked in `build_world_render`.
- iOS sends `viewer=appearance-1` on the page request and on both revision
  polls — `ios/Glasses/Workspaces/WorldBuilder/WorldRenderViewer.swift:287`.
- iOS registers the `glasses-world:` scheme handler the appearance page
  needs — `WorldAssetTransport.swift:31`.
- iOS has captions and upgrade/downgrade ranking for all four rungs —
  `WorldRenderViewer.swift:421-435`, `:485-529`.
- `GET /worlds/{id}/render/revision` reports `representation` and `live` so
  a phone can poll and upgrade — `routes/geometry.py:192`.

The sparse caption the user saw is `WorldRenderViewer.swift:524`, selected
because the page Tower served was stamped
`<meta name="wb-representation" content="sparse">`. The ladder fell to its
floor because the three rungs above it had no artifact on disk.

### Why the artifact was absent

The canonical checkout's reflog:

```
869d715 HEAD@{2026-09-16 00:51:33} pull --ff-only origin integration/all-cartridges-v1
b5c9089 HEAD@{2026-09-22 01:48:09} checkout: moving from integration/all-cartridges-v1 to main
8c75943 HEAD@{2026-09-22 01:48:38} merge origin/world-builder/reconstruction-fixit-v1
```

The live walk ran 2026-09-22 01:33:18 → 01:40:47 (session `started_at`
1790055198.9, finalization `updated_at` 1790055647.2). Throughout that
window the checkout was `869d715`.

```
$ git merge-base --is-ancestor 327c54c 869d715   # the --surface wiring
NO  - wiring ABSENT during capture
$ git ls-tree 869d715 tower/tower/world_builder/ | grep -E "surface|appearance|dense"
(no surface/appearance/dense modules at all)
```

**The Tower that ran the walk did not contain the surface, appearance or
dense stages in any form.** The consolidation that put them on `main`
finished eight minutes after the capture ended. The phone showed sparse
points because sparse was the only rung that build could produce.

### What this actually means for the mission

It means the diagnosis "reconstruction is not connected to visualization"
is wrong as stated — the connection exists in code on `main`. But the
stronger and more uncomfortable fact is this:

> The surface and appearance stages on `main` have never been exercised by
> a live capture. Not once. Zero of 165 worlds carry their output.

So the remaining engineering risk is not "wire it up", it is "prove the
newly-merged wiring actually runs, on real capture data, end to end, and
fix what it turns out is broken". Untested-in-production code is the
hazard, and replay of the 2026-09-22 capture is the instrument.

## What the system is actually supposed to show

Settled by contract, not by opinion. `WORLD-BUILDER-WORLDS.md` §4 defines the
render ladder `appearance → surface → dense → sparse`, and
`WORLD-BUILDER-APPEARANCE.md` §1 defines the top rung:

> **The wearer's own redacted keyframes, prepared to be projected back onto
> the room's proxy surface.** It is the appearance half of the saved world;
> the surface artifact is the geometry half.

Appearance is not a successor to surface — it *consumes* it. The dependency
is a chain, not three rival generations:

```
solve/  →  depth stage (MoGe-2 ViT-L, GPU, writes dense/<sid>/work/)
              ├─→ dense/  points_lN.bin        (a LEAF; nothing consumes it)
              └─→ surface/ mesh_lN.bin         (TSDF fuse → triangle mesh)
                      └─→ appearance/ p.<digest>.bin (proxy = surface's phone level, verbatim)
                                      c.<digest>.bin (ASTC 6x6 / WebP keyframe chunks)
```

Both contracts forbid inventing pixels: no inpainting, no hole closure, no
generative completion, and any future change that fills unobserved space
must break the format identifier rather than quietly relax the promise.
That constraint is settled and must not be re-litigated.

## Regression baseline on main, before any change

```
$ python -m pytest tests/ -k "world" -q       (from the lane worktree)
1715 passed, 18 skipped, 2313 deselected in 544.83s
```

## The wiring is real; only the deployment was stale

`tower/tower/main.py:148-159` builds the builder child's argv:

| flag | setting | default |
|---|---|---|
| `--register` | `world_register` | on |
| `--solve` | `world_solve` | on |
| `--surface` | `world_surface and world_solve` | **on** |
| `--appearance` | `surface and world_appearance` | **on** |
| `--densify` | `world_densify and world_solve` | off (deliberate) |

`tower/.env` sets only `TOWER_CAPTURE_ROOT`, `TOWER_WORLD_ROOT` and
`TOWER_DEV_MODE`; nothing overrides the surface or appearance defaults. So a
Tower restarted from `8d0f1af` will pass `--surface --appearance` on the very
next capture. That is the first thing that has never happened.

## Second defect, found while tracing — and it is a live-path defect

`world_build_session.py` runs `mark_finalization(state=COMPLETE)` (`:2196`)
and `release_world()` (`:2204`) inside the `finally` block, which executes
*before* `--register` (`:2260`), `--surface` (`:2281`) and `--densify`
(`:2299`).

Two consequences on the real product path:

1. A world is stamped `finalization: complete` — which iOS renders as the
   stage word **"Saved"** — while the photographic representation has not
   begun and is six to eleven minutes away on a 690-keyframe walk.
2. **Nothing on disk records whether the surface and appearance stages ran,
   were skipped, or failed.** Their outcome lives only in the builder
   child's stdout report, which nobody persists. If either stage raises, the
   world stays sparse forever behind a session record that says "complete",
   and neither an operator nor the phone can distinguish "never attempted"
   from "failed".

The lock release before the surface stage is deliberate and documented
(`results/world_builder_render.py:552-559`) — the stages are designed to run
unlocked so the phone can read the world meanwhile. The sequencing is not the
bug. The silence is. A fix is in progress that records each stage's outcome
on the session record, additively and backward-compatibly.

## Instruments available (found, not built)

- `tower/scripts/world_replay.py` — deterministic replay of a recorded walk
  through the real builder. It reproduced two 2026-08-29 physical sessions
  figure for figure, which is the warrant for trusting an offline number.
  It supported `--surface` but **not** `--appearance`, so the top rung could
  not be replayed at all; this lane adds the flag.
- `Glasses-scratch\wb-final-recon\fixit\visual-review4\tools\` — a headless
  Chrome + SwiftShader harness that serves a world through the real
  `tower.routes.geometry` routes and drives the page through its own input
  path. Claude's browser tooling has no WebGL on this host; this does.
- The bedroom walk's raw frames survive: captures `27a416ae…` (319 frames,
  disconnect) + `e45108bb…` (1839 frames, stop) = 2158, exactly the
  session's `frames_observed`. The live walk can be replayed end to end.

