# World Builder dense reconstruction — the artifact and its wire format

**Living document.** Added 2026-09-08.

| | |
|---|---|
| Format | `wb-dense-points/1` |
| On disk | `<world>/dense/<session_id>/` |
| Tower producer | `tower/tower/world_builder/dense.py`, `dense_pipeline.py` |
| Offline entry point | `tower/scripts/world_densify.py` |
| Advertised in | `GET /worlds` — the per-session `dense` object (additive) |
| iOS consumer | none yet; the phone reaches it only through `GET /worlds/{id}/render` |

## 1. Why it exists, and what it is not

The sparse solve places a few hundred cameras to sub-pixel accuracy and keeps
about fifteen thousand points. Those same cameras hold roughly a hundred million
pixel observations. This artifact is what is recovered from the rest.

It is **not** a replacement for anything. The sparse reconstruction, the derived
tree, the placements and the existing viewer are unchanged, and a world without
a dense artifact behaves today exactly as it did before this document existed.

## 2. Additive and optional, by necessity

`require_schema` refuses any `schema_version` but 1 and there is no migration
machinery, so a new representation cannot be folded into the existing records.
The dense artifact therefore lives in its own subtree beside `solve/`, following
the `support.json` / `placements.json` precedent: **a reader that does not know
about `dense/` must ignore it, exactly as it ignores `solve/`.**

Three rules follow, and they are the whole compatibility story:

1. **The `contract` identifier of `GET /worlds` does not move.** iOS parses
   these payloads with `JSONSerialization` into `[String: Any]` and reads them
   key by key, so an unknown key is never looked at — but it equality-tests
   `contract` on the first line of every guard, and bumping it empties the
   gallery on every build that predates the change.
2. **No existing field changes type.** `json["points"] as? [[Double]]` is a
   whole-array bridged cast, so a single `null` refuses the entire chunk.
3. **No guarded field is removed**, including all nine `pose_convention` keys.

## 3. On-disk layout

```
<world>/dense/<session_id>/
    manifest.json     format, params, counts, bbox, scale, the LOD table
    points_l0.bin     finest
    points_l1.bin     canonical
    points_l2.bin     mobile
    align.json        per-frame alignment record and gate decisions
    status.json       ok | running | stopped | failed | unavailable
    fused.npz         intermediate; regenerable, not part of the contract
    work/             intermediate; regenerable, not part of the contract
```

Everything outside `manifest.json` and `points_l*.bin` is diagnostic or
intermediate. A consumer reads only those.

## 4. `points_lN.bin`

A flat interleaved buffer with **no header**, so a browser can hand it straight
to the GPU without touching a single point in JavaScript.

| offset | type | field |
|---|---|---|
| 0 | float32 LE | x |
| 4 | float32 LE | y |
| 8 | float32 LE | z |
| 12 | uint8 | red |
| 13 | uint8 | green |
| 14 | uint8 | blue |
| 15 | uint8 | confidence |

**Stride is exactly 16 bytes.** There is no padding field: the three colour
bytes plus the confidence byte fill the word. `byteLength` is always a multiple
of 16 and always equals `points * 16`; a consumer should assert both, because a
stride mismatch does not fail loudly — it shears the scene, which reads as a
geometry bug rather than a parsing one.

`confidence` is **the number of independent cameras whose depth agreed with this
point**, clamped to 255. It is the honesty channel: raising a threshold on it
removes weakly supported geometry, and a viewer should expose that.

Two properties of the number worth stating exactly, because both make it read
slightly HIGHER than a naive reading would suggest:

- It is bounded above by `params.neighbours` — the number of nearby cameras
  consulted — not by how many cameras could in principle have seen the point.
- After voxel reduction it is the **maximum** over the points merged into that
  voxel, not their mean. So it reads "at least one point in this voxel was
  agreed by N cameras", not "every point here was".

## 5. `manifest.json`

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | 1 |
| `format` | string | `wb-dense-points/1`. Compare for equality; ignore the subtree on a mismatch |
| `record`, `endian`, `stride_bytes` | string, string, int | The layout above, stated so a reader need not assume it |
| `confidence_meaning` | string | Prose, for a person |
| `min_confidence` | int | Points below this were never written. **Inert at its default**: the consensus filter already requires `min_views` other cameras, so the lowest confidence any point can carry is `min_views` (3, measured as the minimum on every artifact). It is a floor a stricter operator can raise, not a filter doing work today |
| `dropped_outside_pack_box` | int | How many real observations §6 rule 5 removed |
| `canonical_level`, `mobile_level` | int | Indices into `levels` |
| `median_scene_depth` | number | In world units. Every length in this artifact is a fraction of it |
| `bbox_min`, `bbox_max` | [3] number | Robust bounds (0.2 / 99.8 percentile), not extrema |
| `levels` | array | `{level, voxel, points, bytes}` |
| `scale` | object | **Repeated from the world, never re-derived** |
| `params` | object | Everything that changed the output |

### Scale semantics

`scale` is copied from the world's own `ScaleState`. If the world's scale is
`unknown`, the dense artifact's is `unknown`. **The dense stage introduces no
new scale claim**, and a consumer must not infer metric distance from the
presence of dense geometry.

Only **component 0** is densified. `global_solve.py` never calls COLMAP's
`normalize()`, so the gauge is whatever the first baseline happened to be, and
separate components are solved independently and share no unit. Any consumer
comparing lengths must do so within one component, against
`median_scene_depth`.

### Coordinate frame

Identical to `solve/<session>/solution.json`, which is COLMAP's convention and
**not** the convention `world.json`'s `pose_convention` block describes — that
block describes the derived tree. Verified by projecting the solved points
against the stored observations:

```
x_cam = R @ X_world + t          median reprojection error 0.615 px
camera centre C = -R.T @ t
```

Camera axes are OpenCV's: x right, **y down**, z forward. `up_axis` is
`unknown`, so a viewer must not assume +y or +z is up; on screen, up is `-y`.

## 6. What the artifact promises about honesty

Every point in this file is supported by at least `params.min_views`
independent cameras that agreed on its depth to within `params.tau`. Four
mechanisms enforce that, and each is a refusal:

1. **Per-frame gate.** A keyframe whose depth cannot be reconciled with the
   sparse points the solve already placed — scored on points the fit never saw
   — is dropped whole. Its part of the room stays empty.
2. **Validity mask.** Pixels strung across a depth edge, and surfaces seen at a
   grazing angle, are discarded.
3. **Redaction fill is excluded.** A filled rectangle is not an observation, and
   is left as a hole. See §7.
4. **No hole filling of any kind.** No Poisson, no TSDF closure, no generative
   completion. Poisson and Delaunay are closure methods, watertight by
   construction, and would turn "never observed" into "surface here".

There is a fifth mechanism and it is the one that removes REAL observations, so
it is stated separately rather than counted among the refusals:

5. **The packed ladder drops points outside a central percentile box**
   (`params.pack_percentile`, 0.2% per axis by default). A handful of points at
   extreme depth survive consensus because several nearby frames made the same
   error, and they stretch the bounding box and with it the viewer's opening
   framing. The manifest records `dropped_outside_pack_box`, so the count is
   visible in every artifact. This was hard-coded and undeclared until an
   adversarial review found it.

**An empty region means the observations did not support geometry there.** That
is a load-bearing guarantee for spatial memory, and any future change that fills
holes must break this format identifier rather than quietly relax it.

**Two things that guarantee does NOT cover, and a reader must know both.**
First, the page a phone is served is COARSER than the artifact and may be
thinned to fit a byte budget; §9 says how, and the page says so itself. Second,
a point cloud occludes only where it has points, so a region whose near surface
was dropped is not merely empty — you see the geometry BEHIND it. Measured on
45 sampled capture poses, 13% show that badly enough to mislead, with the poses
themselves correct to under 1.7 px. `01-EVIDENCE.md` §11.5 has the measurement.

## 7. Privacy

`engine.py` redacts faces **before** persisting a keyframe image, so the bytes
any later reconstruction reads are the redacted ones rather than raw frames
behind a display filter.

**That is a property of the session, not of the directory, and the dense stage
checks it.** `FaceRedactor.redact` returns the ORIGINAL bytes when the redactor
is unavailable or throws, labelled `none`, and `engine._persist_keyframe`
persists whatever comes back — so `<world>/sessions/<s>/images/` can legitimately
hold raw frames, and `session.redaction` is the only record that says which. An
earlier version of this stage asserted the boundary in a docstring and read the
directory unconditionally.

1. It reads `session.redaction` first. Anything absent, unreadable or `none`
   means the stored keyframes are **not** trusted as redacted.
2. When the session says they were redacted, it reads the world's own keyframe
   image.
3. When the session says they were not, it **applies the redaction itself**
   before any pixel is read, and refuses the frame when no redactor is
   available.
4. When a world has lost its `images/` directory, it reads the raw capture frame
   and **re-applies the same redaction** before anything looks at it; again it
   refuses rather than using it raw.
5. It never reads the solve workspace's `images/`, which on some workspaces are
   the raw frames COLMAP was fed.

`align.json` records, per frame, which of those paths was taken, and per run it
records what the **session** said about redaction rather than the label of the
redactor that happened to be loaded at densify time.

**The fill mask (§4 refusal 3) is the same whichever path was taken.** It marks
every pixel that is fill rather than what the camera saw. It is the difference
against the raw capture frame when that frame is readable. When the raw frame is
not readable, a build that trusts the stored keyframe uses the shape-gated guess
on that keyframe. A build that re-redacted the keyframe (path 3) uses that guess
united with the exact difference its own re-redaction made.

Path 3 is every live build during a walk, because the session records its
applied label only at Stop. It used to difference against the *stored*
keyframe, which already carries the engine's fill, so its mask was empty
wherever the engine had filled a face. The final build after Stop then reused
that empty mask by image hash. The rule is versioned as `fill_rule` on each
record and on the stage, and it is part of the depth cache key. A prediction or
a cached stage made under another rule is not reused.

## 8. `GET /worlds` — the `dense` object

Additive, per session, `null` on every world built before the dense stage.

| Field | Type | Meaning |
|---|---|---|
| `format` | string | `wb-dense-points/1` |
| `levels` | int | How many LODs exist |
| `canonical_points` | int \| null | Point count at the canonical level |
| `mobile_points` | int \| null | Point count at the mobile level |
| `scale` | object | As in the manifest |
| `solve_current` | bool or null | Whether this cloud was fused against the solve now on disk. `false` after a re-solve or a second session: the points are real observations of a superseded pose graph. **`null` means unknowable, not stale**, and must not be shown as staleness. Additive; §10 has the same fact as the page's caption |

`null` means this session has no dense reconstruction — which is not an error,
and is the state of every world that predates this work.

## 9. Serving it to a phone

`GET /worlds/{id}/render` returns HTML that iOS shows in a `WKWebView`, under

```
Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'
```

With no `connect-src`, **`fetch` and `XMLHttpRequest` are blocked**. That is
deliberate — the documented promise is a page that loads nothing from anywhere —
so the product path **embeds the mobile level inline** in the page rather than
fetching it. A separate desktop-only route may serve the larger levels as binary
under a relaxed `connect-src 'self'`; the phone never uses it.

The route grew one optional query parameter, and nothing else:

| parameter | values | meaning |
| --- | --- | --- |
| `representation` | `auto` (default), `sparse`, `dense` | `auto` serves the dense viewer when the session has a dense artifact and the existing sparse page otherwise. `sparse` forces the old page. `dense` refuses with 404 rather than falling back |

Two behaviours matter more than the parameter:

- **A world with no dense artifact is served exactly the page it was served
  before.** That is every world built before this stage, and the dense work is
  invisible to them.
- **A dense artifact that fails to load costs nothing.** The failure is logged
  and the sparse page is served, because the sparse reconstruction is complete
  and correct either way, and a dense bug must not turn a working world into a
  404. Only `representation=dense` opts out of that.

### Payload budget

The LOD ladder is a fraction of each scene's median depth, so its point count
follows the size of the room rather than the size of the payload — across the
corpus the mobile level ranges from 39 k points to 2.6 M, which is 0.6 MB to
42 MB. The page therefore enforces a byte budget.

**It meets that budget by making the picture coarser, never by keeping a
subset of the room.** A coarser voxel grid is applied over the whole extent and
the best confidence in each cell survives, which is the same operation the LOD
ladder itself performs — the phone gets a lower-resolution room, not a fraction
of one.

This reverses the previous rule, and the reversal is the point. Thinning by a
global confidence threshold sounds like the honest choice and is not:
confidence is high where the wearer stood still and low at the far end of any
space walked past once, so a threshold does not thin a room, it deletes the
parts of it seen from fewer angles. On the three-room chain the old rule shipped
`confidence >= 9`, kept 15% of 2.6 M points, and what reached the phone was a
scatter of isolated wall and ceiling slabs with two of the three rooms gone.

`thinned_to_confidence` in the page's config still records that a cut happened,
and `null` still means none was needed. The caption says the picture is coarser
than the artifact and that nothing was dropped from one part of the room and
kept in another.

## 10. Currency, and what the page must say

`WORLD-BUILDER-WORLDS.md` requires the render page to carry a caption saying
what it is, and a BEHIND line when what it draws is not current. The dense page
is what that route serves when a dense artifact exists, so both obligations are
this format's.

**The caption.** The dense page carries its own sentence, because the sparse
page's would be false here: *"Dense reconstruction: per-pixel depth from a
neural network, anchored to the structure-from-motion solve and kept only where
several cameras agreed. Not a surface, not a mesh, not metric scale."* It is
printed first, before every qualification, and rule 2 of the worlds contract
applies unchanged: the page never claims more than it says.

**Two things can be behind, and they are reported separately.**

| condition | how it is decided | caption |
| --- | --- | --- |
| the dense cloud was fused against an older solve | `manifest.input_digest` (or, for artifacts packed before that key existed, `status.json`'s) against `solve/solution.json`'s | *"This reconstruction is BEHIND the world: it was built from an earlier solve, and the world has been solved again since."* |
| the derived tree is behind the newest keyframes | `render.derived_current`, unchanged | the sparse page's BEHIND sentence |

Either may be **unknowable** — a missing digest, an unreadable solve — and
unknowable is not stale: no BEHIND claim is made from it. A stale reconstruction
is still served rather than suppressed, because the points in it are real
observations of a superseded pose graph, and hiding them would be a different
dishonesty from mislabelling them.

`manifest.input_digest` is additive; a reader that does not know the key behaves
exactly as before, and the format identifier does not move for it.

