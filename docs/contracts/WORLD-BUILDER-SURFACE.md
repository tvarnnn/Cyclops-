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
held-out residual. Before fusion, each frame's affine depth is passed through
the **depth consistency field** (`WORLD-BUILDER-DENSE.md` §11): a smooth
per-keyframe scale-and-offset correction solved jointly over every gated frame
against the sparse points and against the other frames' depth, and applied
only when held-out checks say it helps. This stage adds that correction, the
fusion, the surface, the plane snap (§3), and nothing else about where
geometry comes from.

## 2. What this artifact CLAIMS

1. **Every triangle stands where at least two frames measured something, and
   was not contradicted.** Three tests, all of which a triangle passes:
   - *Field evidence at every corner.* A marching cube emits only if all
     eight corner voxels were observed (weight above 0). Weight is
     accumulated per observation and scaled by the incidence cosine, the
     frame's own gate score, and an inverse-square depth falloff capped at
     `max_near_boost` (so one close frame CAN reach `min_weight` alone;
     weight is not a frame count). A cube whose corners all reached
     `min_weight` emits a full-weight face. With `low_weight_evidence` (on by
     default, since 2026-09-17) a cube observed at every corner that did NOT
     all reach `min_weight` emits a **low-weight** face, which the frame tests
     below must admit under two more conditions: **no** frame saw through it
     (not a ratio), and its supporting cameras span `low_weight_min_parallax`
     (0.05: the diagonal of their centres' bounding box over the face's
     distance to the box's centre). With it off, only full-weight cubes emit.
   - *Distinct frames.* At least `min_support_frames` (2) distinct frames
     measured depth within their truncation of the face.
   - *Not seen through, not seen from behind.* Frames that measured depth
     beyond the face number fewer than `contradiction_ratio` (2) times the
     supporting frames, and at least one supporting frame is on the face's
     front side.

   Why low-weight faces exist, measured (`Glasses-scratch/wb-final-recon/fixit/holes/HOLES.md`):
   after the consistency field, the weight gate was the rule behind 56% of
   the black pixels at the phone proxy's walk poses and 38% at the novel views
   on the canonical capture -- far and oblique walls, the ceiling, the floor
   in front of the desk, which two or more frames had measured. With 10% of
   keyframes held out of fusion, the low-weight faces admitted were seen
   through by held-out frames 5.3% of the time against 4.4% for the faces the
   full-weight rules keep; relaxing `min_weight` without the two extra
   conditions admitted faces seen through 17% of the time. Relaxing the
   contradiction ratio, the back-facing test, the component prune, the
   grazing limit or the far bound gained almost no pixels and added
   contradicted geometry, and none of them changed.
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
   per face along the entire ray (claim 1).

   **What that removes is a ratio, not every hand.** A face is removed when
   the frames that saw past it number at least `contradiction_ratio` (2)
   times the frames that measured it: `through >= 2 x support`. A thing near
   the camera -- a hand, the wearer's lap -- is removed only when that holds.
   Consecutive keyframes often hold the same hand, so a hand that 3 frames
   measured and 5 saw past (5 < 6) **survives**. So does a hand that no other
   frame looked through. The rule also cannot tell "saw past it" from "could
   not resolve it": a thin structure that close frames measured and distant
   frames smoothed into the background can be removed.

   One measurement, not a property: on the canonical capture, the faces within
   0.5 units of the walked path went from 2,330 (all contradicted) to 0.
4. **The frames were made to agree before they were fused, or the manifest
   says why not.** `detail.depth_consistency.state` is `applied` when every
   gated frame's depth was fused through the correction field, and then the
   field beat the plain affine on held-out sparse points AND on held-out frame
   pairs (`detail.depth_consistency.heldout`). Any other state -- `refused`,
   `failed`, `skipped`, or the key absent (built before 2026-09-17) -- means
   the plain affine was fused. The evidence tests of claim 1 count frames by
   the same corrected depth that was fused.

   This is a claim about agreement, not about truth. Frames can agree on a
   wall that every one of them bends the same way: on the canonical capture
   the corrected walls bow by about 5-8 voxels over 8-10 units, where the solve
   has almost no sparse points to say otherwise.
5. **The coordinate frame is the solve's**, identical to
   `solve/<session>/solution.json`: world-to-camera poses, `x_cam = R·X + t`,
   OpenCV axes with y down. It is NOT the frame `world.json`'s
   `pose_convention` block describes, which belongs to the derived tree's
   segment-local poses.
6. **The wearer's detected hands, arms and held phone have zero weight.**
   When `transients.state` is `ok`, every pixel of a keyframe's transient
   detector mask (`WORLD-BUILDER-APPEARANCE.md` §5.3a; `union` for a final
   build, `oneformer` for a live one) is removed from that frame's valid depth
   before fusion: it neither measures, carves, nor counts as support or
   contradiction in claim 1. A hand lying on the desk is within depth noise of
   the desk, so claim 3 cannot remove it; this can. Any other state means no
   mask was applied — never read "unavailable" as "masked". The rule id is part
   of `params_digest`. Measured on the canonical world: L0 went from 2,698,323 to
   2,696,847 faces; rendered at the recorded poses ki 308/330/351/391, the vertex
   colour changed by a mean 6.9–9.9 (of 255) inside those keyframes' masks and
   1.1–2.3 outside, concentrated on the desk under the handled phone. The
   ghost it removes is faint, because each vertex colour already averages many
   keyframes; the phone itself stays (it rests there in other frames).
7. **Scale is inherited, never invented.** `manifest.scale` is copied verbatim
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

Three things that are permitted and are not closure, because all are bounded
by observation:

- **Averaging inside the truncation band.** A voxel's value is the weighted
  mean of what several cameras said about it. That is interpolation between
  measurements, not invention beyond them.
- **Feature-preserving smoothing.** Taubin smoothing moves vertices to remove
  the marching-cubes staircase. `manifest.detail.median_vertex_move_voxels`
  records how far, in voxels, the median vertex moved; a reader may treat that
  as the artifact's geometric slack.
- **Plane snap** (`params.plane_snap`, on by default; `surface.snap_planes`).

  *What it is.* After smoothing, large planes are found in the mesh itself.
  A plane is snapped only if all three hold:
  - its connected area is at least `snap_min_area_frac` x (scene scale)^2
    (6 square units on the canonical capture);
  - its own least-squares fit has RMS at most `snap_tol_voxels` (2.5 voxels);
  - at least `snap_min_frames` (8) keyframes' fused depth measures it.

  A vertex of an accepted plane moves only along the plane's normal, onto the
  plane: fully within the tolerance, smoothly less out to twice it, and not at
  all beyond. Only vertices within twice the tolerance whose (neighbourhood-
  smoothed) normal is within 20 degrees of the plane's are candidates.

  *What it is not.* It adds no vertex and no face, removes none, closes no
  hole, and never moves a vertex by more than twice the tolerance or in any
  direction but the plane's normal. A plane therefore never extends past the
  surface that was reconstructed, and a hole in a wall stays a hole. Structure
  that stands off a plane by more than twice the tolerance (a picture frame, a
  shelf), or that is turned away from it, does not move. It is not a claim that
  the room is made of planes: a surface the gates refuse keeps its measured
  shape. It does not fix a bend the frames share -- separate segments of one
  bowed wall can snap to slightly different planes.

  `detail.plane_snap` records every accepted plane (area, fit RMS in voxels,
  measuring frames, normal, point, moved p50/p99 in voxels), the count of
  refused candidates, `vertices_moved`, `area_snapped` and `max_move`.

## 4. Layout

```
<world>/surface/<session>/
    manifest.json        what is here, how it was built, what it claims
    status.json          the stage's state, with the pid that wrote it
    mesh_l0.<build>.bin  level 0, the archive
    mesh_l1.<build>.bin  level 1
    mesh_l2.<build>.bin  level 2, the final build's phone level
    .surface.lock        held while a build runs
    surface.log          the live child's output, when the builder ran one
```

A final build has three levels and `mobile_level` 2. A live build has two
(`lod_face_targets` `(0, 120000)`, `mobile_level` 1): `mesh_l0` and a 120k-face
`mesh_l1`. The phone is sent the largest level whose page fits the budget (§8),
which is not necessarily `mobile_level`.

Every file is published atomically: the bytes are written to a staging path
and renamed only once whole. A reader therefore sees either the previous
artifact or the new one, never a torn one.

A build is published as a unit. Each build writes its levels under its own
`<build>` id, and `manifest.json` -- written last -- names them in
`levels[].file` with their byte counts. Readers resolve a level only through
the manifest and refuse a file whose size disagrees. A build killed during
packing therefore leaves orphan level files that nothing names, and the
previous manifest keeps pointing at the previous build's whole set.

**Pruning.** A reader that has read a manifest may still read that manifest's
levels for at least two minutes after a newer manifest **replaces** it. The
grace counts from supersession, not from when the files were written: the
builder stamps the replaced manifest's level files just before it writes the
new manifest. Level files no current manifest names are pruned once that grace
has passed. Pruning runs after each successful publish, and also at the start
of every build of the session, including one that finds the session already
built. A stopped pack removes the levels it wrote, since no manifest names
them. Staging files left by a killed write are pruned once they are ten
minutes old. A manifest that cannot be read prunes nothing. A manifest
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
| `params_digest` | input digest plus every parameter that affects the result, then `|` and the depth backend's name (a different network changes every triangle), then `|set:<name>@<digest>` when the session reads a re-redacted keyframe set (`WORLD-BUILDER-APPEARANCE.md` §6.5), then `|transients:` and the transient detector's rule id. It includes the consistency solver's version and parameter digest and the plane snap's parameters and version, so every surface built before them is rebuilt. Recomputing it from `params` alone does not reproduce it |
| `params` | the full parameter set, including `quality` |
| `median_scene_depth` | the scene scale: the median camera-frame depth of the solve's sparse observations by gated frames (`scene_scale_source` says `sparse-observation-depth`), or the dense-depth median over every frame when the solve carries too few observations (`dense-depth-median`). Voxel size is a fraction of it |
| `detail` | the build's record (added 2026-09-16; absent from manifests written before). Artifacts without it carry the same record in `status.json` `result.detail` until the next status write |
| `detail.truncation_floor`, `detail.truncation_rel` | the truncation band of a sample measured at depth `d` is `max(floor, min(rel * d, trunc_max_voxels * voxel))` when `rel > 0` (the floor wins over the cap), and `floor` alone when `rel` is 0 |
| `detail.evidence_filter` | faces removed by claim 1's frame tests, by reason (`dropped_support`, `dropped_contradicted`, `dropped_back_facing`, and for low-weight faces `dropped_weak_seen_through`, `dropped_weak_parallax`), with `faces_in`, `faces_kept`, and `weak_in` / `weak_kept` (low-weight faces offered and kept; absent when `low_weight_evidence` is off or the enclosed fill is on, which does not use it); `null` when the filter did not run |
| `detail.depth_consistency` | the consistency field this surface was fused through (added 2026-09-17): `state` (`applied`, `refused`, `failed`, `skipped`), `reason`, `frames`, `cells`, `heldout.before` / `heldout.after` (held-out sparse-point `sfm` and held-out frame-pair `cross` median relative error, the plain affine vs the field), `warm_start`, `seconds`, `gpu_peak_mb`, `key`, `reused`. Only `applied` means corrected depth was fused. A surface built while the solve `failed` is never reported "already built": the next build tries the solve again |
| `detail.plane_snap` | the plane snap's record (§3): `plane_count`, `plane_areas`, `planes[]`, `rejected`, `vertices_moved`, `area_snapped`, `max_move`, `tol`, `min_area`, `min_frames`, `seconds`. `null` when `params.plane_snap` is off |
| `detail.voxel_coarsened_by` | how far the block budget (`params.max_blocks`) coarsened the voxel. A walk the budget cannot hold after 12 coarsening attempts is refused (`unavailable`, naming the budget) rather than built over it |
| `voxel`, `truncation` | in scene units. `truncation` is the band a sample at the scene scale actually got, cap included (manifests written before 2026-09-16 recorded the uncapped request) |
| `frames_used`, `frames_offered` | how much of the walk contributed |
| `vertices`, `faces` | of level 0 |
| `levels` | per level: level, vertices, faces, bytes |
| `canonical_level`, `mobile_level` | which rung is the archive and which the phone gets |
| `seconds` | per stage: `depth`, `transients` (ensuring the detector masks), `consistency`, `fuse`, `mesh`, `snap`, `pack` |
| `transients` | claim 6: the detector report (`state`, `detail`, `mode`, `rule`, `models`, `frames_masked`, `computed`, `cached`, `seconds`, `gpu_peak_mb`) plus `frames_fused_with_mask`. Absent from manifests written before 2026-09-17 |
| `scale` | inherited verbatim, with a note saying so |
| `closure` | the sentence stating that unobserved space is absent, and that a hole may also be space the frames disagreed about |

`params.quality` is `live` or `final`. A live artifact is coarser — twice the
voxel, two levels of detail rather than three, a lighter smoothing pass,
`min_weight` 1.5 rather than 2.0 — because it is built during the walk against
the gap between global solves. The two-frame, contradiction and low-weight tests are the same for live.

**Where depth is used does not depend on the scene scale.** Each frame's depth
is used out to `anchor_depth_multiple` (1.5) times that frame's own farthest
fitted sparse anchor, so a live build and the final build bound depth the same
way. The scene scale still sets the voxel, and it still moves with what the
walk has seen: on the canonical capture a build over the first 100 / 150 / 264
keyframes had 1.09x / 1.14x / 1.09x the final scale (it was 1.27x / 1.39x /
1.20x when it was a sampled dense-depth median).
The final build replaces the live one in the same directory, because the
product question is always "the best available reconstruction of this
session". **Exception:** the final surface needs the final global solve. When
that solve fails or solves nothing, no final surface is built, and the saved
world keeps the last live artifact (`params.quality: live`) from an earlier
background solve; the finalization report says why.

**States.** `status.json` `state` is `running`, `ok`, `stopped`, `failed` or
`unavailable`. `unavailable` with `permanent: true` means **this machine** cannot
run the depth network: its package or torch is not installed, the backend is
unknown, or its weights are neither in the Hugging Face cache nor downloadable
(offline on a machine that never fetched them; the default `moge2-vitl` is about
1.3 GB). The detail names the model and the cache. `world_surface.py` exits 4
for it, and the builder stops launching live surfaces **for the rest of that
walk**; the final build after Stop, `world_surface.py`, and the next walk all
try again, so a network that returns is used then. A refusal about one
session's inputs (for example a camera the poses were not solved in) is
`unavailable` without `permanent`. A first download of the weights is logged
with its size and time.

**`live` in the revision route turns false once the build has published.** A
build writes its manifest, prunes superseded levels, and only then writes `ok`.
A stage whose `manifest.json` is newer than its `running` `status.json` is not
reported live (`WORLD-BUILDER-WORLDS.md` §4a), so a poll in that gap sees the
finished build with `live: false`.

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
stated in the page (`<meta name="wb-representation">`) and by
`GET /worlds/{id}/render/revision`; the `GET /worlds` listing does **not** state
it (its per-session `dense` object is the only artifact summary there).

`representation=surface` on a session that has no surface at all returns
**404** rather than silently serving points: a caller that pinned a
representation is comparing, and a silent substitution would corrupt the
comparison.

**The phone budget bounds the page, not the mesh.** The page inlines the
chosen level base64-encoded (4/3 of its bytes) beside the viewer and its
configuration. The served level is the largest whose **page** fits
`MOBILE_BYTE_BUDGET` (6 MiB). The estimate is the template's size plus a
64 KiB configuration allowance; if the composed page still exceeds the budget,
the level is chosen once more with the measured overhead. The pack stage makes
`mobile_level` fit that page. `lod_face_targets[mobile_level]` is a ceiling:
the level is decimated again from its parent, scaled by the byte overshoot,
until its page fits `params.mobile_page_bytes`. `detail.mobile_page_fit`
records the result.

**Decimation keeps rims.** Every level is decimated with quadric boundary
weight `lod_boundary_weight` (100; it is in the params digest). A surface that
stops where the evidence stops is mostly rim, and at weight 1 the phone level
opened 0.13% / 0.20% / 0.18% of the pixels level 0 covered at the canonical
capture's walk poses / novel views / look-around views, as cracks along shelf
edges and silhouettes; at 100 it is 0.07% / 0.16% / 0.11% on a surface with 21%
more faces. A rim keeps more vertices, so the level that fits the page carries
about 7% fewer faces (217,306 against 235,368 on the canonical capture).

Before this, the budget was applied to the mesh bytes. Live replay D's
299,999-face phone level (5.88 MB) was served as a 7.89 MB page. Re-running
the phone-level step on that walk's final surface gives 232,086 faces and a
6.16 MB (5.87 MiB) page. A surface packed before this has no fitting level; it
is served its smallest level, over budget, until it is rebuilt.
`mobile_page_bytes` is in the params digest, so such a surface is not "already
built". Rebuilding it re-predicts depth for every frame as well, because the
dense fill-mask rule (`FILL_RULE`) also moved and invalidates the depth cache:
about 50-70 s of GPU for 400 keyframes.

**`max_points` floors at the smallest level.** It lowers the page budget to
`max_points x 16` bytes, but when no level fits, the smallest level is served,
over that budget. The phone level is normally already the smallest, so
`max_points` cannot make a surface page smaller than the default one; it can
only keep a larger level from being chosen. `level=` requests are not bounded.

**The dense rung is not held to the page budget.** Its 6 MB budget is on the
binary point buffer, which the page inlines base64-encoded (4/3 of its size): the
canonical densified world's dense page measured 8,028,516 bytes. Known and
accepted, because the dense stage is off by default (`TOWER_WORLD_DENSIFY=false`)
and `auto` serves it only on a densified world with no surface.

**The vertical is an estimate, stated as one.** The solve declares no up
(`up_axis: unknown`). The page's configuration carries `up`, seeded by the
mean camera up and refined on the served level's own faces: the area-weighted
normal of near-horizontal faces (within 30 degrees), iterated. The refinement is
kept only when near-horizontal faces are at least 10% of the area, it moves the
seed by at most 45 degrees, and the area-weighted mean wall tilt does not get
worse. Otherwise `up` is the camera estimate. On the canonical world the camera
estimate leaned 27 degrees toward the desk the wearer looked down at, and the
page showed the room rolled; the refined vertical takes the median wall tilt
from 13.1 to 4.6 degrees. Nothing claims the vertical is gravity.

## 9. Rebuilding

Derived, and rebuildable from authoritative data alone:

```
.venv\Scripts\python.exe scripts/world_surface.py --world <id> --force
```

The authoritative inputs are the session's keyframes and the global solve.
Deleting `surface/` loses nothing that cannot be rebuilt. The depth maps under
`dense/<session>/work/` are shared with the point stage and are themselves
derived; if they have been pruned, the stage recomputes them. So are
`dense/<session>/consistency.json` and `consistency_field.npz`: deleting them
costs one cold solve (about 35 s on the canonical capture).
