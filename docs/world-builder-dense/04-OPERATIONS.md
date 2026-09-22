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
| `--backend` | `moge2-vitl` | the depth model (MIT). Chosen by a 24-model bake-off, D15. Only permissively licensed checkpoints are reachable |
| `--gate-rel` | `0.08` | reject a frame whose held-out alignment residual exceeds this |
| `--tau` | `0.05` | how closely a neighbouring camera must agree. Measured optimum; 0.03 and 0.08 are both worse |
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

Measured on capture `20ce3c23…`, replayed at the shipped configuration:
1709 frames staged, 198 posed keyframes, 169 used, 8.43 M points,
45 s of depth and 67 s of fusion inside a 188 s total. The intermediates
were pruned on success and the artifact left is 167 MB.

## 3. After a live capture

`scripts/world_build_session.py --densify` runs the dense stage after the final
build. It is skipped outright on a hard stop, and it is skipped when there is no
global solution. A skip is recorded in the report and in `dense/status.json`; it
is never reported as a failure.

## 3b. Turning it on for every capture

`TOWER_WORLD_DENSIFY=true` makes the Tower pass `--densify` to every builder it
spawns, so a capture produces a dense world by itself. **It is off by default
and that default is the decision**, not an oversight: the stage costs about
2.4 GB of VRAM and two to four minutes of GPU per world, on a card four other
cartridges share. It runs after the world lock is released, so it blocks no
capture -- but it competes for the GPU with whatever the wearer does next.

It is ignored unless `TOWER_WORLD_SOLVE` is also on, because the dense stage is
anchored to the global solution's cameras and sparse points and there is
nothing to anchor to without one.

Before this setting existed the only way to get a dense artifact was to run
`scripts/world_densify.py` by hand, and the CLI's own docstring wrongly said
otherwise.

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
the solve already placed. Across the seven worlds it drops **30-50% of posed
frames**, and on the tight bathroom walk it drops half. `align.json` records
why each frame was dropped. This is the number to quote when someone asks how
much of the walk the reconstruction actually uses; "spends the evidence" spends
between half and two thirds of it.

**Tight rooms are the weakest case, and it is structural.** The closet and the
bathroom cover 60-67% of a held-out frame where every other world covers 96-98%,
and the bathroom's depth error p90 is 58% against a 4.4% median. Consensus needs
baseline between cameras and a small room does not offer any. The confidence
channel is what separates the good part from the bad; a viewer that ignores it
will show the tail as if it were the median.

**Heavy face-redaction fill wrecks a frame.** The detector fires on hands and on
carpet. `align.json` records `redaction_fill_fraction` per frame; frames above
30% almost never pass the gate even after inpainting.

**The dense stage takes minutes and Stop kills the tree on a 30-second grace.**
That is why it runs last, is skipped on a hard stop, and checkpoints each stage.
An interrupted run resumes with `scripts/world_densify.py`.

**GPU contention makes everything look hung.** With several jobs sharing the
RTX 5070, a stage that takes 90 s alone can take many minutes. Check
`nvidia-smi` before concluding something is stuck.

## 5b. Two things the manifest now records, and why you want them

`input_digest` names the solve the cloud was fused from, so a world re-solved
since it was densified is served with a BEHIND caption instead of silently.
Artifacts packed before that key existed fall back to `status.json`.

`dropped_outside_pack_box` counts the real observations the packing stage's
percentile trim removed -- 98,652 of 8.2 M on the desk world, about 1.2%. It
exists because a handful of points at extreme depth survive consensus (several
nearby frames making the same error) and otherwise stretch the bounding box and
with it the viewer's opening framing. It is the one mechanism in this pipeline
that discards real observations, so it is counted where anyone can see it.

## 6. What the artifact costs

Measured, per world, on the datasets used here:

All seven worlds, densified with the shipped configuration. `01-EVIDENCE.md`
§11 carries the same run with its accuracy tails and its gate sensitivity;
this is the operator's view of it.

| world | frames used / posed | held-out residual | L0 points | L0 size |
| --- | --- | --- | --- | --- |
| `7d31e8d7` (desk and shelf) | 344 / 429 | 2.4% | 8.5 M | 136 MB |
| `1b8812b1` (widest traverse) | 314 / 438 | 2.7% | 9.4 M | 150 MB |
| `672578d0` (bedroom, closet, desk) | 306 / 425 | 3.5% | 15.0 M | 239 MB |
| `37e497f8` (bedroom) | 139 / 196 | 3.5% | 5.2 M | 84 MB |
| `fc58a64d` (end-to-end replay) | 175 / 198 | 4.0% | 8.8 M | 141 MB |
| `ecc02df1` (dresser) | 55 / 77 | 4.3% | 2.2 M | 35 MB |
| `a378331a` (closet) | 114 / 201 | 4.9% | 5.3 M | 85 MB |
| `6427900d` (bathroom, tight) | 139 / 266 | 5.0% | 3.9 M | 62 MB |

Peak VRAM for the depth stage is **2.4 GB**; everything else is CPU and RAM.
A 12 GB card runs this comfortably; a 4 GB one will not.

**The residual column is gate-conditioned and the gate is 8%.** It is the median
over the frames that PASSED, so it improves as the gate tightens and the
reconstruction gets worse. `01-EVIDENCE.md` §11.3 prints both ends. Do not
quote a single figure from this column as the pipeline's accuracy.

### Footprint, and why a successful run cleans up after itself

A 438-keyframe world leaves **601 MB** if nothing is pruned, and only about
130 MB of that is the artifact. Every artifact produced before the pruning
landed still has the intermediates beside it -- 318-703 MB per session across
the corpus, of which 220-474 MB is `work/`. `--force` on an old artifact
re-runs and prunes; nothing sweeps them otherwise, and nothing in this stage
deletes anything a human has not asked it to.

| | |
| --- | --- |
| `points_l0/1/2.bin` + manifest + align + status | ~132 MB — **the artifact** |
| `fused.npz` | ~76 MB — the same points `points_l0.bin` already holds |
| `work/` (per-frame depth maps, undistorted frames) | ~393 MB |

So after `pack` succeeds, `work/` and `fused.npz` are removed. That is the stage
deleting its own intermediates, inside its own subtree, after the output they
produced is complete — not a cleanup of anyone's artifacts, and the project's
filesystem policy on deletion is about the latter.

**`--keep-intermediates` turns it off, and that is the flag for development.**
Re-fusing with different parameters off cached depth maps is the whole iteration
loop; it takes about two minutes instead of eight. The depth maps are stored as
float16, because the values run 0.2–40 world units and the pipeline's own error
is a few percent, so three significant digits is already more than the evidence
supports.

The analysis scripts under `Glasses-scratch\wb-dense\proto\` accept either
`fused.npz` or a `points_l0.bin` path, so a pruned artifact is still
inspectable.
