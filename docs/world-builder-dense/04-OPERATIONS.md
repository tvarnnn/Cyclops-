# World Builder dense reconstruction — how to run it

Everything here was run on the development Tower (Windows 11, Intel i9, RTX 5070
12 GB, 32 GB RAM) with `tower/.venv`.

---

## 1. Densify a world that already exists

This is the development loop, and it needs no glasses and no new capture. Seven
solved worlds are on disk; twelve more were solved from historical captures
during this work.

```
cd tower
.venv\Scripts\python.exe scripts/world_densify.py --list
.venv\Scripts\python.exe scripts/world_densify.py --world <world_id>
.venv\Scripts\python.exe scripts/world_densify.py --world <world_id> --force
.venv\Scripts\python.exe scripts/world_densify.py --all
```

`--list` prints every session that holds a global solve and whether it has been
densified. `--force` ignores cached stages; without it, a re-run resumes.

Useful knobs, all recorded into the manifest:

| flag | default | what it changes |
| --- | --- | --- |
| `--backend` | `depth-anything-v2-small` | the depth model. Only permissively licensed checkpoints are registered |
| `--gate-rel` | `0.08` | reject a frame whose held-out alignment residual exceeds this |
| `--tau` | `0.03` | how closely a neighbouring camera must agree |
| `--min-views` | `3` | how many other cameras must agree |
| `--neighbours` | `10` | how many nearby cameras are consulted |
| `--stride` | `1` | pixel stride when sampling a reference view |
| `--component` | `0` | which solve component to densify. Do not change without reading §5 |

To densify a world under a scratch root rather than the real store:

```
.venv\Scripts\python.exe scripts/world_densify.py ^
    --world-root C:\Users\<you>\Projects\Glasses-scratch\wb-dense\worlds\<name> ^
    --world <world_id>
```

## 2. Replay a historical capture and densify it in one go

This exercises the whole lifecycle — observe, keyframe, solve, build, densify —
exactly as a live capture would, from frames already on disk.

```
cd tower
.venv\Scripts\python.exe scripts/world_replay.py ^
    --captures <capture_id> ^
    --capture-root data\captures ^
    --root C:\Users\<you>\Projects\Glasses-scratch\wb-dense\e2e ^
    --intrinsics-from <a directory holding 360x640.json> ^
    --solve --densify --format json
```

`--densify` requires `--solve`; the dense stage is anchored to the global
solution and there is nothing to anchor to without one.

Measured on capture `0bbc2b7e…`: 327 frames staged, 58 keyframes, 26 used,
730,426 points, dense stage 105 s inside a 144 s total.

## 3. After a live capture

`scripts/world_build_session.py --densify` runs the dense stage after the final
build. It is skipped outright on a hard stop, and it is skipped when there is no
global solution. A skip is recorded in the report and in `dense/status.json`; it
is never reported as a failure.

## 4. Look at the result

```
cd tower
:: orthographic quad view, coloured by true RGB
.venv\Scripts\python.exe ..\..\Glasses-scratch\wb-dense\proto\view.py ^
    --npz <world>\dense\<session>\fused.npz --solve <world>\solve\<session> ^
    --out quad.png --min-conf 3

:: reconstruction beside the real photograph, from real camera poses
.venv\Scripts\python.exe ..\..\Glasses-scratch\wb-dense\proto\persp.py ^
    --npz <world>\dense\<session>\fused.npz --solve <world>\solve\<session> ^
    --undist <world>\dense\<session>\work\undist ^
    --out views --mode compare --n 8

:: viewpoints no camera ever occupied
.venv\Scripts\python.exe ..\..\Glasses-scratch\wb-dense\proto\persp.py ^
    --npz ... --solve ... --out views --mode orbit --n 6 --offset 0.9
```

`--offset` is in world units. Divide by the manifest's `median_scene_depth` to
read it as a fraction of the scene.

## 5. Things that will bite

**Do not change `--component` casually.** `global_solve.py` never calls COLMAP's
`normalize()`, so components are solved independently and share no unit.
Densifying component 1 and component 0 into one cloud would compose two
different scales into one room.

**Every length is a fraction of `median_scene_depth`.** The gauge varies by more
than an order of magnitude between worlds in this corpus — from a ten-unit
extent to a 340-unit one. An absolute threshold that works on one world is
meaningless on the next.

**A low frames-used ratio is usually the data, not a bug.** The alignment gate
drops frames whose predicted depth cannot be reconciled with the sparse points
the solve already placed. On the corpus's blurriest walk that is nearly half the
frames. `align.json` records why each frame was dropped.

**Heavy face-redaction fill wrecks a frame.** The detector fires on hands and on
carpet. `align.json` records `redaction_fill_fraction` per frame; frames above
30% almost never pass the gate even after inpainting.

**The dense stage takes minutes and Stop kills the tree on a 30-second grace.**
That is why it runs last, is skipped on a hard stop, and checkpoints each stage.
An interrupted run resumes with `scripts/world_densify.py`.

**GPU contention makes everything look hung.** With several jobs sharing the
RTX 5070, a stage that takes 90 s alone can take many minutes. Check
`nvidia-smi` before concluding something is stuck.

## 6. What the artifact costs

Measured, per world, on the datasets used here:

| world | frames used / posed | held-out residual | L0 points | L0 size | wall clock |
| --- | --- | --- | --- | --- | --- |
| `7d31e8d7` (desk and shelf) | 345 / 429 | 3.0% | 11.9 M | 190 MB | ~4 min |
| `1b8812b1` (widest traverse) | 247 / 438 | 5.0% | 5.8 M | 93 MB | 7 min |
| `672578d0` (bedroom, closet, desk) | 218 / 425 | 6.7% | 5.7 M | 92 MB | 8 min |
| `ecc02df1` (dresser) | 40 / 77 | 7.6% | 1.0 M | 16 MB | ~1.5 min |
| `be36bd70` (replay, end to end) | 26 / 58 | 8.7% | 0.7 M | 12 MB | 105 s |

Peak VRAM for the depth stage is **0.85 GB**; the rest is CPU and RAM. The
intermediate `work/` and `fused.npz` are regenerable and can be removed to
reclaim most of the footprint — but note the project's filesystem policy:
**move to `Glasses-scratch\`, never delete without explicit approval.**
