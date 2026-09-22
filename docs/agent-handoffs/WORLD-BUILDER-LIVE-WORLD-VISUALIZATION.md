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

## The second live capture — 2026-09-22 02:37, on current main

While this investigation was running, the owner performed a second physical
walk, this time against `main` @ `8d0f1af`. World
`2f44716237544569b5f2faf782d9f877`, session `cb30880107eb4e4bae7c53f821399549`.

```
02:36:21  builder child spawned (World Builder workspace opened)
02:37:40  LIVE surface child spawned  (pid 7964)
02:37:43  session started
02:39:22  Stop          385 keyframes of 1160 frames, end_reason "stop"
02:40:47  finalization  state "complete", final_solve "solved"
~02:42    owner manually stopped the Tower
```

### What it proves — the wiring works

This capture created `dense/` and `surface/` directories. The first world in
this project's history to do so. `surface/<sid>/surface.log` shows the live
child running `depth: 0/49 → 25/49 → 49/49`, then loading OneFormer and
starting `transients-provenance: 0/37`. `--surface` and `--appearance` were
on the builder's argv, the live surface child launched on a landing solve,
and the depth stage completed. **The photographic path starts automatically
on current main.** That question is now closed.

### What it does not prove

The owner stopped the Tower manually about two minutes after Stop, before
the build could finish. **The process death was external and is not a
product defect**; it is not treated as one here. The surface stage's
`state: stopped` at `stage: depth` is the honest record of an externally
interrupted run, not of a failure.

Note also that this session's live surface child was competing for the GPU
with a replay build this campaign had started at 02:34 (95% utilisation).
That contention was this campaign's fault and is not a property of the
product. The replay was killed at 02:42 to return the card.

### The real defect the capture exposed

The owner did not know the photographic build was still running. Nothing
told them, and everything available told them the opposite:

- `session.json` said `finalization: {state: "complete", final_solve:
  "solved"}` at 02:40:47 — one minute and twenty-five seconds after Stop.
- iOS maps a finalized session with `final_solve: solved` to the stage word
  **"Saved"** (`WorldPresentation.swift:502-535`).
- The render ladder correctly served `sparse`, because sparse was the only
  rung with an artifact. The phone captioned it honestly.

So the product's own account of itself was: *finished, saved, here are your
points*. The truth was: *the surface stage is four minutes into a build that
needs six to sixteen, and the thing you actually want does not exist yet.*

`mark_finalization(COMPLETE)` (`world_build_session.py:2196`) and
`release_world()` (`:2204`) run in the `finally` block, **before**
`--register` (`:2260`), `--surface` (`:2281`) and `--densify` (`:2299`).
"Complete" means the sparse derived tree is on disk. It has never meant the
photographic world exists.

### The durability gap behind it

Independent of who stopped the Tower, an interrupted photographic build is
lost permanently:

- `main.py` never references `scripts/world_finalize.py`, and that script
  rebuilds the *sparse* derived tree — it does not run surface or
  appearance.
- There is no startup reconciliation, no resume across process death, and no
  retry for a world that owes a surface. The only resume in the codebase is
  per-frame checkpointing *inside* one dense run.
- The serving path is strictly read-only and never generates on demand
  (proven: zero writes, zero spawns in the serving modules).

So any interruption in the six-to-sixteen-minute window after Stop — a
shutdown, a sleep, a crash, a lost cartridge session — leaves the world
sparse forever, behind a record that says "complete". That window is
unsupervised, unrecorded and unrepeatable, and it is the window in which the
entire photographic product lives.

## The visual bar, confirmed by inspection

Before changing anything, this campaign verified that the appearance
renderer actually clears the product bar, using the previous campaign's
headless Chrome + SwiftShader harness rebuilt under
`Glasses-scratch\wb-live-viz-v1\viz\`.

Rendered at ten of the wearer's own camera poses, at matched pose and field
of view, beside the wearer's own keyframe
(`viz/out/known-good/atposes/ab_sidebyside.png`, world `b2a75ab4…`):

- the backlit keycaps, the red mouse and the monitor's page layout are
  reproduced legibly;
- the shelf's bottles, cups and soundbar, the framed pictures, the white
  door and its hinges are all present and recognisable;
- the wearer's own hand, present in three of the captures, is correctly
  **absent** from the render — the transient masking works;
- the honest failures are visible too, as black torn regions where no
  geometry was measured.

This is not a visualisation of a reconstruction. It is the room. The
renderer is not the problem and must not be rewritten.

## The fix, and the constraint that shapes it

**Constraint: the iPhone app cannot be rebuilt as part of this fix.** This
is a Windows host with no Xcode. Any change that the next physical test
depends on must be Tower-side and must work with the build already on the
wearer's phone. (Swift changes can be written and validated separately on a
Mac session, but the primary fix must not require that.)

That constraint turns out to be a gift, because the installed app already
contains exactly the right behaviour and the Tower simply was not
triggering it:

| Tower lifecycle | iOS stage word | iOS note |
|---|---|---|
| `finalizing` + `build_in_progress: true` + geometry | **"Improving"** | "This world is still being finished… the finished world is very different from this one — **it is worth waiting for Saved**." |
| finalized + `final_solve: solved` | **"Saved"** | — |

`_lifecycle()` (`tower/tower/results/world_builder.py:1163`) decides that
purely from the writer lock: `alive = holder is not None and
holder["alive"]`. The builder releases the lock before the photographic
stages by design, so for the entire six-to-sixteen minutes in which the
photographic world is actually being built, the Tower reports the world
finished and the phone says "Saved".

**Fix A — the lifecycle tells the truth.** While a surface, appearance or
dense stage is verifiably running for the session, report `finalizing` with
`build_in_progress: true`. The detection already exists and is reused rather
than reinvented: `results/world_builder_render.py::session_build_running()`,
which checks each stage's `status.json` for a `running` state with a live
pid and rejects a status left behind by a dead process. Its own docstring
already names this gap. The lock ordering is NOT changed.

Effect on the installed app, with no reinstall: the wearer sees "Improving"
and is told to wait for "Saved", and the viewer's revision poller keeps
`live: true` so it does not back off to its 120-second ceiling. When the
appearance lands, the existing upgrade ladder swaps the page up
automatically.

**Fix B — the record says what the stages did.** Each of surface,
appearance and dense records its outcome on the session record: attempted or
not, its terminal state, and a detail on failure — including when the stage
raises. Backward compatible in the way `finalization` already was: a record
written before the field existed still parses, and absent means never
recorded. Without this a failed stage is indistinguishable from one that was
never attempted, which is the state every world on this machine is in.

**Fix C — owed work is finished, not forgotten.** An interrupted
photographic build is currently lost permanently: `main.py` never invokes
`scripts/world_finalize.py`, that script does not run surface or appearance
anyway, the serving path never generates on demand, and there is no
reconciliation at startup. A world that owes a photographic representation
must be able to acquire one without a developer command.

## The decisive experiment: the interrupted build, finished

The owner's 02:37 walk was interrupted by their Tower shutdown before the
photographic stages could finish. This campaign copied that world to
scratch and ran the same stages to completion, uninterrupted, with the
product's own defaults.

```
$ world_surface.py --root <scratch>/live2-root \
      --world 2f44716237544569b5f2faf782d9f877 \
      --session cb30880107eb4e4bae7c53f821399549 --appearance --force
02:48:48 → 02:56:28, exit 0
```

**It works.** Both artifacts built cleanly from a real live capture — the
first time in this project's history that a world produced by the glasses
has carried a photographic representation.

| surface stage | s | appearance stage | s |
|---|---:|---|---:|
| depth | 75.0 | detector | 41.2 |
| transients | 167.1 | provenance | 3.8 |
| consistency | 25.4 | occluders | 4.5 |
| fuse | 10.9 | exposure | 8.9 |
| mesh | 38.0 | transients | 9.7 |
| snap | 5.5 | selection | 1.2 |
| pack | 52.0 | encode | 15.5 |
| **total** | **373.9** | **total** | **85.3** |

**459 s — seven minutes forty — for a 385-keyframe, 100-second walk, from
a cold cache.** Mesh 1,198,535 verts / 2,265,728 faces at L0, decimated to
153,480 / 229,927 for the phone. 282 of 379 offered frames used. Appearance
352 keyframes, 128 in the phone tier.

### The privacy decision, settled by measurement rather than by precedent

Every render any human has praised in this project was built with
`imagery_source: raw-local-research` — the 2026-09-21 bypass that skips
redaction. The product default is `redacted`, and a normally-started Tower
**404s a raw artifact by design**. So the obvious worry was that the
shippable path produces a materially worse picture, and that the baseline
would need the bypass — which the mission forbids, because it would mean
environment-variable gymnastics during normal operation.

It does not. This build used the default:

```
imagery_source : redacted
privacy_safe   : true
```

and its selection coverage **beats** the raw reference:

| | seen by ≥1 keyframe | seen by ≥2 |
|---|---:|---:|
| this world, phone tier, **redacted** | **0.9841** | **0.9740** |
| canonical `b2a75ab4`, phone tier, **raw** | 0.971 | 0.9605 |

Two reasons, both measured. This session's redaction gate is
`yunet-2023mar@0.30+plausibility3`, which is on the appearance stage's
trusted allowlist and whose false-positive rate is far lower than the
ungated redactor: sampling 129 of its 385 stored keyframes gives **7.69% of
all pixels filled, median 0.20%**, against the 12.6% measured on the
canonical capture. And 352 keyframes over a small room give the selector
more to choose from for the same 128-frame phone budget.

**Decision: the live path keeps the default redacted imagery.** No flag, no
environment variable, no bypass, and `privacy_safe` stays true. The deferred
privacy research on `world-builder/reconstruction-fixit-precision` stays
unmerged; nothing here required it.

## The serving contract, proven on the live world

Driven through the real `tower.routes.geometry` router against the finished
world, with the exact query shapes `WorldRenderViewer.swift` builds:

| request | answer |
|---|---|
| `/render/revision?session_id=…&viewer=appearance-1` | `representation: "appearance"`, `appearance.state: "served"`, `current: true` |
| `/render/revision?session_id=…` (no capability) | `representation: "surface"` — an older app degrades to the rung it can draw |
| `/render?session_id=…&viewer=appearance-1` | 200, `text/html`, 282,029 B, `<meta name="wb-representation" content="appearance">` inside the first 4096 characters, which is the window iOS scans |
| `/worlds/{w}/appearance/{s}/manifest` | 200, `X-World-Imagery: redacted`, `X-World-Redaction: faces-detected-and-filled/yunet-2023mar@0.30+plausibility3` |
| `/worlds` | `contract: world_builder.worlds/2026-09-10` (the exact string iOS equality-tests), session `state: complete`, `has_geometry: true`, `appearance: served` |

Every link in GLASSES → IPHONE → TOWER → … → SAVED WORLD → IPHONE VIEWER is
now demonstrated on data from a real walk, with one exception: the wearer
has not yet seen it on the phone, because the build finishes about eight
minutes after Stop and nothing told them to wait.

