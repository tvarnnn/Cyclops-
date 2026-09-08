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

## 5. `manifest.json`

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | 1 |
| `format` | string | `wb-dense-points/1`. Compare for equality; ignore the subtree on a mismatch |
| `record`, `endian`, `stride_bytes` | string, string, int | The layout above, stated so a reader need not assume it |
| `confidence_meaning` | string | Prose, for a person |
| `min_confidence` | int | Points below this were never written |
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

**An empty region means the observations did not support geometry there.** That
is a load-bearing guarantee for spatial memory, and any future change that fills
holes must break this format identifier rather than quietly relax it.

## 7. Privacy

`engine.py` redacts faces **before** persisting a keyframe image, so the bytes
any later reconstruction reads are the redacted ones rather than raw frames
behind a display filter. The dense stage keeps that boundary:

1. It reads the world's own redacted keyframe image when it exists.
2. When a world has lost its `images/` directory, it reads the raw capture frame
   and **re-applies the same redaction** before anything looks at it.
3. When no redactor is available it **refuses the frame** rather than using it
   raw.
4. It never reads the solve workspace's `images/`, which on some workspaces are
   the raw frames COLMAP was fed.

`align.json` records, per frame, which of those paths was taken.

## 8. `GET /worlds` — the `dense` object

Additive, per session, `null` on every world built before the dense stage.

| Field | Type | Meaning |
|---|---|---|
| `format` | string | `wb-dense-points/1` |
| `levels` | int | How many LODs exist |
| `canonical_points` | int \| null | Point count at the canonical level |
| `mobile_points` | int \| null | Point count at the mobile level |
| `scale` | object | As in the manifest |

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
