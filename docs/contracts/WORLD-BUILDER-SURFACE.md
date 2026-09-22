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
     below must admit under two more conditions: no frame saw through it --
     or, since 2026-09-17, when at least `low_weight_through_min_support` (8)
     distinct frames supported it, at most `low_weight_through_frac` (0.34) of
     that many did -- and its supporting cameras span `low_weight_min_parallax`
     (0.05: the diagonal of their centres' bounding box over the face's
     distance to the box's centre). With it off, only full-weight cubes emit.
     With `low_weight_hidden_test` (on by default, since 2026-09-17) a kept
     low-weight face is then removed when the kept surface hides it from
     **every** frame that supported it: the ray from each supporting frame's
     centre to the face meets kept surface nearer than the face by more than
     the fusion band. Such support is depth that passed through that surface,
     not evidence for the face.
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

   Why hidden low-weight faces are removed, measured
   (`Glasses-scratch/wb-final-recon/fixit/moved/MOVED.md`): the admitted
   low-weight faces included sheets behind walls that some frame's depth
   reached through the wall and that no frame could ever see through, because
   the wall blocks every view; they were most of the surface no kept keyframe
   image covers. With 10% of keyframes held out of fusion the test removed 26%
   of the kept low-weight faces (5% of the surface area); held-out frames
   supported and could see 0.5% of them (44% of the low-weight faces kept),
   98% of the removed faces a held-out frame measured were hidden from every
   such frame too, and held-out frames saw through them 11.5% of the time
   (kept low-weight faces 4.3%). The appearance's phone-tier coverage of the
   proxy rose from 95.1% to 97.7%.

   Why a few see-through frames are tolerated, measured
   (`Glasses-scratch/wb-final-recon/fixit/geom2/GEOM2.md`): the black tears
   in the ceiling around the fan and the cracks along the hutch's top board
   were low-weight faces that many frames of one pass supported and one or two
   frames of another pass "saw through", because the passes placed that far
   surface a band apart (the ceiling: 15 frames from 14 units away against 2
   from under the fan). With 10% of keyframes held out of fusion, the faces the
   tolerance admits were contradicted by held-out frames 5.6% of the time (all
   kept faces 4.4%); where they are the nearest surface the held-out depth
   agrees on 67% of pixels and the mesh stands in front of it on 6.5% (whole
   surface 79% / 8.7%). A floor of 5 supporters instead of 8 admitted faces
   contradicted 9.6% of the time, and was not taken.

   Measured and NOT adopted in the same round: keeping the near side of a
   depth edge (the rim of a shelf board the edge test rejects) added faces
   held-out frames contradicted 30% of the time and pixels in front of the
   held-out depth 17% of the time, for 0.1 point of black; lowering
   `contradiction_ratio` to 1 removed faces held-out frames contradicted 48% of
   the time but also removed half-right ceiling around the fan and the laptop
   on the bed, adding 0.3-0.6 points of black. Neither changed.

   **Moved objects are not detected.** No rule keys on WHEN frames measured a
   face. Measured on the same capture: kept faces whose supporting and
   see-through frames fall in disjoint time windows are real (the split
   predicts 97-99% of held-out votes), but they are split by VIEWPOINT as
   much as by time: the mean supporting and see-through camera positions stand
   0.7-1.0 face distances apart (median), within a quarter of the distance for
   fewer than 5% of them -- and the ones inspected were static walls
   whose monocular depth disagrees from a different distance. Removing the
   earlier state would cut holes in real walls. A thin moved object (a phone on
   a desk) is inside the fusion band and invisible to any depth test; a door
   swung between passes is removed by the contradiction test where it is.
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
8. **Every vertex carries how well it was measured, and it is a ranking, not
   a probability** (added 2026-09-18; `SurfaceParams.confidence`, on by
   default). Beside each level, `conf_lN.<build>.bin` holds one byte a vertex
   (§5a). 255 does not mean the triangle is correct and 0 does not mean it is
   absent — every triangle in the file already passed claim 1. It ranks the
   surface by how much of a photograph it deserves, and it was validated as a
   ranking on keyframes **held out of the fusion**:

   | measured on 36 held-out keyframes of the canonical capture | |
   |---|---|
   | pixels where the held-out frame disagrees with the mesh, ranked by confidence | AUC **0.759** |
   | pixels where the mesh stands in FRONT of what the frame measured | AUC **0.764** |
   | faces the held-out frames contradicted (saw through more than they supported) | AUC **0.903** |
   | faces under 0.30 that held-out frames contradict | **37%**, against 4.4% of the surface at large |
   | faces under 0.20 | **50%** |

   Half the answer is evidence the fusion counted (how many frames supported
   the face, its accumulated weight, how many saw through it, and how far the
   supporting frames' depths spread within the band **after** the consistency
   solve); half is what the level's own triangles say (aspect, distance to a
   boundary, normal agreement in the neighbourhood), computed on each level
   separately so a decimated level is graded on the triangles it has. What is
   NOT in it, having been measured and found not to predict, is in §5a.

   The number is comparable **within one artifact**, not across captures: it
   is not calibrated, and it says nothing about appearance — a well-measured
   wall no keyframe photographed scores high and is still drawn dark.

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

  **A rim is smoothed along the rim** (`params.smooth_boundary_curve`, on by
  default). A surface that stops where the evidence stops is mostly boundary —
  48% of the canonical capture's phone-level faces touch one — and a boundary
  vertex's neighbours all lie on one side of it, so the surface umbrella both
  dragged the rim across the surface, away from where the evidence ended, and
  left the staircase *along* the rim untouched, since none of the neighbours it
  averaged was itself on the rim. A boundary vertex with exactly two boundary
  neighbours now takes the same lambda/mu passes over its own boundary
  polyline; a junction (any other count) is held still; every interior vertex
  keeps the surface umbrella. No face and no vertex is added or removed, and
  the only interior vertices that move differently are a rim's own neighbours
  (63% of interior vertices do not move at all; p99 0.13 voxels).

  Measured on the canonical capture
  (`Glasses-scratch/wb-final-recon/fixit/rims/RIMS.md`): at level 0 the rim
  roughness — a boundary vertex's distance from the midpoint of its two
  boundary neighbours — falls from 0.31 to 0.17 voxels (p90 0.71 to 0.32), the
  boundary from 192,064 to 157,481 voxels long, and the rim's drift across the
  surface from −0.29 to −0.18 voxels, so 2.8% more area survives. At the phone
  level, 8,045 boundary loops become 4,203 and 7,167 pinholes become 3,574.
  Against 10% of keyframes held out of fusion, the pixels it gains and the
  pixels it loses agree with the held-out depth equally often (45.2% against
  44.1%), it gains 17x more than it loses, and the whole surface's agreement is
  unchanged (75.7 / 6.9 / 17.4%). `detail.rims` records which operator ran and
  what rim it left. It is in the params digest, so a surface built before it
  rebuilds.
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
    conf_l0.<build>.bin  level 0's per-vertex geometry confidence (§5a)
    conf_l1.<build>.bin  level 1's
    conf_l2.<build>.bin  level 2's
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
`levels[].file` with their byte counts, and its confidence sidecars in
`levels[].confidence.file` with theirs. Readers resolve a level only through
the manifest and refuse a file whose size disagrees. A build killed during
packing therefore leaves orphan level files that nothing names, and the
previous manifest keeps pointing at the previous build's whole set.

**Pruning.** A reader that has read a manifest may still read that manifest's
levels for at least two minutes after a newer manifest **replaces** it. The
grace counts from supersession, not from when the files were written: the
builder stamps the replaced manifest's level files just before it writes the
new manifest. Level files and confidence sidecars no current manifest names are
pruned once that grace has passed, under the same rule and the same grace. Pruning runs after each successful publish, and also at the start
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

## 5a. `conf_lN.<build>.bin` — the per-vertex geometry confidence

Little-endian. One byte a vertex, in the level's own vertex order.

| offset | type | field |
|---:|---|---|
| 0 | `char[8]` | magic, `WBCONF01` |
| 8 | `uint32` | vertex count |
| 12 | `uint32` | version (1) |
| 16 | `uint32` | flags (reserved, 0) |
| 20 | `uint8[count]` | confidence, 0–255 |

`format` is `wb-surface-confidence/1`. The reader refuses a wrong magic, an
unknown version, a length that is not exactly `20 + count`, and a count that
does not match the level it is being read against.

**Why it is not inside `wb-surface-mesh/1`.** Every reader of that format —
`read_mesh_bytes`, `store._looks_like_mesh`, and both viewer pages' own
`decodeMesh` — refuses a buffer whose length is not exactly what its header
implies. A channel appended to the mesh, behind a new flag or not, would break
all of them at once, including two pages. A sidecar the manifest names costs
the same bytes, is versioned on its own, and leaves every existing reader
untouched. **On the wire to a phone** the channel travels differently again:
the appearance proxy's colour bytes, which have been zero since review 1,
carry it at no cost at all (`WORLD-BUILDER-APPEARANCE.md` §4.2a).

**How it is computed.** Each component is a score in [0, 1] where 1 is good,
floored at 0.05 so no single one can answer zero alone, and they are combined
as a weighted **geometric mean** — not a sum, because the components are not
interchangeable: a face forty frames measured that no two of them agree about
is not "mostly fine".

| half | component | what it is | weight |
|---|---|---|---:|
| evidence (0.7) | `agreement` | 1 − see-through frames / supporting frames | 2.0 |
| | `spread` | 1 − RMS of (measured − face) / band over the supporting frames, the cross-frame disagreement local to that face after the consistency solve | 1.0 |
| | `support` | supporting frames / 40 | 0.5 |
| | `weight` | the face's smallest corner weight / (4 × `min_weight`) — under 1 exactly where the face came through the low-weight exception | 0.5 |
| geometry (0.3) | `rim` | edge hops to the surface's boundary / 3 | 1.0 |
| | `normal` | how far the incident face normals agree | 1.0 |
| | `shape` | worst incident triangle aspect (circumradius / twice inradius), 1 equilateral, 0 at 6 | 0.5 |

The evidence half is computed once, on level 0, where the fusion counted; it
reaches a decimated level through the nearest level 0 vertex. The geometry half
is computed on **each level's own triangles**, so the phone level is graded on
the slivers decimation gave it.

**Measured and deliberately not used.** Three things were asked for, computed,
and put to the held-out test, and none is in the score:

| | held-out AUC | why not |
|---|---|---|
| parallax of the supporting cameras | 0.56 on the fit half, **0.48** (below chance) on the validation half; adding it takes the combination 0.759 → 0.755 | it is already an admission rule for low-weight faces; as a grade it is noise. Both halves agree |
| the consistency field's local correction magnitude | 0.61 alone. **The halves disagree**: the fit half says removing it helps by 0.0016, the validation half says adding it helps by 0.0056 (0.759 → 0.765) | the pass that counts every other component does not carry the correction field, and plumbing a per-frame map through fusion costs more than a gain smaller than the disagreement between the halves |
| plane-snap membership | 0.517 | lifting snapped vertices toward 1 by 0.35 **cost** 0.002 of AUC. A snapped vertex is not measurably better geometry; it is smoother geometry |

**Two components that ship did not survive validation.** On the half of the
held-out frames nothing was fitted on, dropping `support` raises the AUC from
0.759 to **0.767** and dropping `weight` raises it to **0.764**; the fit half
said the opposite of both. They ship at the lowest weight in the table because
removing them on the strength of the validation half would be fitting to the
validation half, which is the only unfitted evidence this score has. Only
`agreement` (−0.032 if dropped) and `spread` (−0.030) earn their weight on
both halves.

A fitted logistic combination of all ten components, fitted on half the
held-out frames and scored on the other half, reaches AUC 0.751 — below the
0.759 of the rule above, because a linear model cannot express "one very bad
component spoils the vertex". A plain weighted **arithmetic** mean of the
shipped components, however, reaches 0.769 at pixel level and 0.908 at face
level, both above the geometric mean. The geometric mean ships because a page
reads only the bottom of the ranking, and at the worst 2% of faces — the
budget a fade would spend — it is the better of the two (48.5% of those faces
contradicted by held-out frames, against 47.0%).

**Honestly, about the geometry half:** on the held-out depth test it adds
nothing and very slightly costs. On the validation half evidence alone scores
0.7591 (0.7666 against the in-front label) and 0.7/0.3 scores 0.7589 (0.7642);
geometry alone scores 0.698. It is kept, at less than half the weight, because
the held-out frames stand where the wearer stood and never look at the surface
edge-on from a novel view, and because the level a phone is sent is decimated —
which damages exactly these three components and nothing the evidence half
counts.

**The number is not comparable between levels.** Because the geometry half is
recomputed on each level's own triangles, a decimated level scores
systematically lower: on the canonical capture level 0's vertices have median
0.71 and the phone level's 0.34. A reader that thresholds must use a threshold
for the level it read, or carry one across by matching surface area. Level 0's
validated 0.20–0.30 band is 0.157–0.243 (bytes 40–62) on that capture's phone
level.

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
| `detail.evidence_filter` | faces removed by claim 1's frame tests, by reason (`dropped_support`, `dropped_contradicted`, `dropped_back_facing`, and for low-weight faces `dropped_weak_seen_through`, `dropped_weak_parallax`, `dropped_weak_hidden`), with `faces_in`, `faces_kept`, and `weak_in` / `weak_kept` (low-weight faces offered and kept, after the hidden test; absent when `low_weight_evidence` is off or the enclosed fill is on, which does not use it); `weak_tested` and `hidden_rays` (the low-weight faces the hidden test examined and the rays it cast; absent when `low_weight_hidden_test` is off); `null` when the filter did not run |
| `detail.depth_consistency` | the consistency field this surface was fused through (added 2026-09-17): `state` (`applied`, `refused`, `failed`, `skipped`), `reason`, `frames`, `cells`, `heldout.before` / `heldout.after` (held-out sparse-point `sfm` and held-out frame-pair `cross` median relative error, the plain affine vs the field), `warm_start`, `seconds`, `gpu_peak_mb`, `key`, `reused`. Only `applied` means corrected depth was fused. A surface built while the solve `failed` is never reported "already built": the next build tries the solve again |
| `detail.rims` | the level-0 boundary the smoothing left (added 2026-09-21; absent from manifests written before): `smoothing` (`curve` or `umbrella`, per `params.smooth_boundary_curve`), `boundary_edges`, `length_voxels`, `junction_vertices`, and `roughness_voxels` / `roughness_voxels_p90` — the median and p90 distance of a boundary vertex from the midpoint of its two boundary neighbours, in voxels. It is how ragged the rims are, and the number §3's rim smoothing exists to move. Measured after the smoothing and before the plane snap, which moves vertices again (about +2% of boundary length on the canonical capture). `roughness_voxels` is `null` for a closed surface |
| `detail.plane_snap` | the plane snap's record (§3): `plane_count`, `plane_areas`, `planes[]`, `rejected`, `vertices_moved`, `area_snapped`, `max_move`, `tol`, `min_area`, `min_frames`, `seconds`. `null` when `params.plane_snap` is off |
| `detail.voxel_coarsened_by` | how far the block budget (`params.max_blocks`) coarsened the voxel. A walk the budget cannot hold after 12 coarsening attempts is refused (`unavailable`, naming the budget) rather than built over it |
| `voxel`, `truncation` | in scene units. `truncation` is the band a sample at the scene scale actually got, cap included (manifests written before 2026-09-16 recorded the uncapped request) |
| `frames_used`, `frames_offered` | how much of the walk contributed |
| `vertices`, `faces` | of level 0 |
| `levels` | per level: level, vertices, faces, bytes, `file`, and `confidence` (§5a): `{file, bytes, vertices, format, version}`, absent when none was built |
| `confidence` | the channel as a whole: `{format, version, levels, bytes, components, evidence_share}`, or `null`. Absent from manifests written before 2026-09-18, and `null` when `params.confidence` is off or the enclosed fill is on (that path does not produce the corner weight the score is graded on, and it already breaks the format identifier) |
| `canonical_level`, `mobile_level` | which rung is the archive and which the phone gets |
| `seconds` | per stage: `depth`, `transients` (ensuring the detector masks), `consistency`, `fuse`, `mesh`, `snap`, `pack` |
| `transients` | claim 6: the detector report (`state`, `detail`, `mode`, `rule`, `models`, `frames_masked`, `computed`, `cached`, `seconds`, `gpu_peak_mb`) plus `frames_fused_with_mask`. Absent from manifests written before 2026-09-17 |
| `outliers` | what this build did about the solve's own outliers (added 2026-09-22; absent from manifests written before, which fused whatever the solver produced). `poses_offered`, `poses_gated`, `pose_radius_multiple`, `pose_radius_median`, `pose_radius_gated_max`, `pose_detail` — a pose whose centre lies beyond `params.outlier_radius_multiple` times the median pose radius of this solve is not fused, and a build that used fewer frames than the solve offered says so here rather than quietly reporting the smaller coverage. `fusion_bound_lo` / `fusion_bound_hi` / `fusion_bound_multiple` / `fusion_bound_detail` — the box depth was fused inside, built from the inlier sparse points and from every kept camera centre grown by the scene-relative far bound, then scaled about its own centre; `frames_clipped_by_bound` and `frames_emptied_by_bound` count what it cost. `voxel_coarsened_for_key_range` — the block grid could not be keyed at the requested voxel and was coarsened until it could, rather than the build being refused; `voxel_coarsened_for_block_budget` is the same number for `params.max_blocks`. Every criterion is a multiple of the solve's own median, never a distance in units, because `global_solve` does not normalise. Excluding solver noise is not inventing geometry, and nothing here fills what it excludes (§3) |
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
