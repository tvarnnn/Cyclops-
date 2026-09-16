# World Builder — the surface artifact

Contract identifier: `wb-surface-mesh/1`.

This describes what `<world>/surface/<session>/` contains, what it claims, and
what it promises never to claim. It is additive: a reader that does not know
about `surface/` must be able to ignore it exactly as it ignores `solve/` and
`dense/`, and a world that has never been reconstructed keeps working.

---

## 1. What this artifact IS

A triangle mesh with per-vertex colour, extracted from a truncated signed
distance field that was fused from posed depth maps.

The depth maps are the dense stage's: monocular depth per keyframe, fitted by
a robust affine to the sparse points the global solve triangulated, gated on a
held-out residual. This stage adds the fusion, the surface, and nothing else
about where geometry comes from.

## 2. What this artifact CLAIMS

1. **Every triangle stands where at least two frames measured something, and
   was not contradicted.** Three tests, all of which a triangle passes:
   - *Field evidence at every corner.* A marching cube emits only if all
     eight corner voxels reached `min_weight`. Weight is accumulated per
     observation and scaled by the incidence cosine, the frame's own gate
     score, and an inverse-square depth falloff capped at `max_near_boost`
     (so one close frame CAN reach `min_weight` alone; weight is not a frame
     count).
   - *Distinct frames.* At least `min_support_frames` (2) distinct frames
     measured depth within their truncation of the face.
   - *Not seen through, not seen from behind.* Frames that measured depth
     beyond the face number fewer than `contradiction_ratio` (2) times the
     supporting frames, and at least one supporting frame is on the face's
     front side.
2. **Space that was never measured is absent, not closed.** Unobserved cells
   keep zero weight, and no cube with an unobserved corner emits. The surface
   stops at the edge of what was seen.

   **The converse is NOT claimed: a hole does not mean "nobody looked".** A
   hole is either space no frame measured, or space the frames disagreed
   about. Measured on the canonical capture (`b2a75ab4...`, 349 frames): of
   the depth samples the cameras measured (every valid pixel, 6-pixel
   stride), 26.4% have no surface within 0.18 units. Of those in the review's
   in-range band, 49% lie where other frames saw past them (the field was
   carved), 24% lack `min_weight` of agreeing evidence, 24% sit at a zero
   crossing whose faces were removed by the tests in claim 1 or pruned as
   small islands, 2% lie behind another frame's surface, and 0.4% were never
   integrated. A reader must not present
   a gap as proof of empty space.
3. **Free space was counted along the whole ray.** Where a camera measured
   depth `d`, everything nearer than `d` was recorded as empty: blended into
   the field within `max_carve_voxels` in front of that surface, and counted
   per face along the entire ray (claim 1). A thing near the camera that
   other frames saw straight through -- a hand, the wearer's lap -- does not
   survive. On the canonical capture the faces within 0.5 units of the walked
   path went from 2,330 (all contradicted) to 0. A thing that moved but that
   no other frame looked through is not detected.
4. **The coordinate frame is the solve's**, identical to
   `solve/<session>/solution.json`: world-to-camera poses, `x_cam = R·X + t`,
   OpenCV axes with y down. It is NOT the frame `world.json`'s
   `pose_convention` block describes, which belongs to the derived tree's
   segment-local poses.
5. **Scale is inherited, never invented.** `manifest.scale` is copied verbatim
   from the world's `ScaleState`. When the world's scale is `unknown` so is
   the artifact's, and no length in the viewer is labelled in metres.

## 3. What this artifact PROMISES NOT to claim

**No closure.** Poisson reconstruction, Delaunay/Ball-pivoting closure,
generative completion, and plane extrapolation into unobserved space are all
forbidden. They are watertight by construction, which means they answer "was
there a wall here?" with "yes" whether or not anyone looked.

This is a property of the format, not of the current implementation. **Any
future change that fills unobserved space must break the format identifier
rather than quietly relax this.** A reader that sees `wb-surface-mesh/1` is
entitled to believe every triangle was measured.

Two things that are permitted and are not closure, because both are bounded
by observation:

- **Averaging inside the truncation band.** A voxel's value is the weighted
  mean of what several cameras said about it. That is interpolation between
  measurements, not invention beyond them.
- **Feature-preserving smoothing.** Taubin smoothing moves vertices to remove
  the marching-cubes staircase. `manifest.detail.median_vertex_move_voxels`
  records how far, in voxels, the median vertex moved; a reader may treat that
  as the artifact's geometric slack.

## 4. Layout

```
<world>/surface/<session>/
    manifest.json        what is here, how it was built, what it claims
    status.json          the stage's state, with the pid that wrote it
    mesh_l0.<build>.bin  level 0, the archive
    mesh_l1.<build>.bin  level 1
    mesh_l2.<build>.bin  level 2, what a phone is sent
    .surface.lock        held while a build runs
    surface.log          the live child's output, when the builder ran one
```

Every file is published atomically: the bytes are written to a staging path
and renamed only once whole. A reader therefore sees either the previous
artifact or the new one, never a torn one.

A build is published as a unit. Each build writes its levels under its own
`<build>` id, and `manifest.json` -- written last -- names them in
`levels[].file` with their byte counts. Readers resolve a level only through
the manifest and refuse a file whose size disagrees. A build killed during
packing therefore leaves orphan level files that nothing names, and the
previous manifest keeps pointing at the previous build's whole set. Level
files no manifest names are pruned once they are two minutes old. A manifest
written before per-build names (no `file` key) still resolves to
`mesh_lN.bin`. This is not decoration — a torn
`solution.npz` read across processes ended a session holding 795 keyframes
and 26,634 points on 2026-09-09, and `mesh_l*.bin` has exactly the same
writer-in-one-process, reader-in-another shape.

## 5. `mesh_lN.<build>.bin` — the wire format

Little-endian throughout. One self-describing buffer.

| offset | type | field |
|---:|---|---|
| 0 | `char[8]` | magic, `WBSURF01` |
| 8 | `uint32` | vertex count |
| 12 | `uint32` | index count (three per triangle) |
| 16 | `uint32` | flags |
| 20 | `uint32` | schema version |
| 24 | `float32[3]` | bounding box minimum |
| 36 | `float32[3]` | bounding box maximum |
| 48 | … | payload |

Flags: bit 0 `HAS_NORMALS`, bit 1 `INDEX_U16`.

Payload, in order:

1. `uint16[3]` per vertex — position, quantised linearly across the bounding
   box. Not lossy in any way that matters: a 14-unit scene quantises to 0.0002
   units, four orders below the voxel the geometry was built at, and it halves
   what reaches a phone.
2. `uint8[3]` per vertex — colour.
3. `int8[3]` per vertex — normal, scaled by 127. Present only when
   `HAS_NORMALS`.
4. `uint16` or `uint32` per index, depending on `INDEX_U16`.

**The reader checks its own input.** A buffer whose length does not equal the
length its header implies is refused; so is a wrong magic and an unknown
schema version. An empty mesh — zero vertices, zero indices — is a legal
artifact and round-trips, because "nothing was reconstructed" has to be
distinguishable from corruption.

## 6. `manifest.json`

| key | meaning |
|---|---|
| `format` | `wb-surface-mesh/1` |
| `schema_version` | 1 |
| `input_digest` | the solve this was built from; compare with the live solve to detect staleness |
| `params_digest` | input digest plus every parameter that affects the result |
| `params` | the full parameter set, including `quality` |
| `median_scene_depth` | the scene scale: the median camera-frame depth of the solve's sparse observations by gated frames (`scene_scale_source` says `sparse-observation-depth`), or the dense-depth median over every frame when the solve carries too few observations (`dense-depth-median`). Voxel size is a fraction of it |
| `detail.truncation_floor`, `detail.truncation_rel` | the truncation band of a sample measured at depth `d` is `min(max(floor, rel * d), trunc_max_voxels * voxel)`; `truncation` is that band at the scene scale |
| `detail.evidence_filter` | faces removed by claim 1's frame tests, by reason |
| `voxel`, `truncation` | in scene units |
| `frames_used`, `frames_offered` | how much of the walk contributed |
| `vertices`, `faces` | of level 0 |
| `levels` | per level: level, vertices, faces, bytes |
| `canonical_level`, `mobile_level` | which rung is the archive and which the phone gets |
| `seconds` | per stage |
| `scale` | inherited verbatim, with a note saying so |
| `closure` | the sentence stating that unobserved space is absent, and that a hole may also be space the frames disagreed about |

`params.quality` is `live` or `final`. A live artifact is coarser — twice the
voxel, one level of detail, a lighter smoothing pass, `min_weight` 1.5 rather
than 2.0 — because it is built during the walk against the gap between global
solves. The two-frame and contradiction tests are the same for live.

**Where depth is used does not depend on the scene scale.** Each frame's depth
is used out to `anchor_depth_multiple` (1.5) times that frame's own farthest
fitted sparse anchor, so a live build and the final build bound depth the same
way. The scene scale still sets the voxel, and it still moves with what the
walk has seen: on the canonical capture a build over the first 100 / 150 / 264
keyframes had 1.09x / 1.14x / 1.09x the final scale (it was 1.27x / 1.39x /
1.20x when it was a sampled dense-depth median).
The final build replaces the live one in the same directory, because the
product question is always "the best available reconstruction of this
session".

## 7. Staleness

`surface_currency()` compares the manifest's `input_digest` with the solve now
on disk and reports `{present, current, reason}`.

Staleness is **reported, never enforced**. A surface built from an earlier
solve is still the best picture of the world that exists, and hiding it is
what left the gallery empty for a whole capture the last time a reader refused
non-current geometry.

## 8. How it is served

`GET /worlds/{world_id}/render?representation=…` walks a ladder:

    surface → dense points → sparse points

`auto`, the default, serves the best rung the session actually has. A named
value starts the walk at its own rung. A rung whose module fails to import, or
whose artifact is unreadable, falls through to the next one — the viewer never
loses the world to a bug in a better renderer. The rung actually served is
stated in the page and in `GET /worlds`.

`representation=surface` on a session that has no surface at all returns
**404** rather than silently serving points: a caller that pinned a
representation is comparing, and a silent substitution would corrupt the
comparison.

## 9. Rebuilding

Derived, and rebuildable from authoritative data alone:

```
.venv\Scripts\python.exe scripts/world_surface.py --world <id> --force
```

The authoritative inputs are the session's keyframes and the global solve.
Deleting `surface/` loses nothing that cannot be rebuilt. The depth maps under
`dense/<session>/work/` are shared with the point stage and are themselves
derived; if they have been pruned, the stage recomputes them.
