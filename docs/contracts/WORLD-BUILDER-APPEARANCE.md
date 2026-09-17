# World Builder — the appearance artifact

Contract identifier: `wb-appearance-keyframes/1`.

**Living document.** Added 2026-09-17 (fix-it campaign, appearance stage). §6.5 (re-redaction) added 2026-09-17.

| | |
|---|---|
| Tower producer | `tower/tower/world_builder/appearance_pipeline.py` (build), `tower/tower/world_builder/appearance.py` (provenance, masks, gains, selection, encoding) |
| Tower routes | `tower/tower/routes/geometry.py`, through the adapter `tower/tower/results/world_builder_appearance.py` |
| CLI | `tower/scripts/world_appearance.py` |
| Consumer | the appearance page, `tower/tower/world_builder/appearance_viewer.html` (`WORLD-BUILDER-WORLDS.md` §4, the top rung of the render ladder), reached on the phone through `ios/Glasses/Workspaces/WorldBuilder/WorldAssetTransport.swift` (`WORLD-BUILDER-IOS.md` §10) |

This describes what `<world>/appearance/<session>/` contains, what it claims,
and what it promises never to claim. It is additive in the way `surface/` is:
a reader that does not know about `appearance/` ignores it, and a world that
has none keeps working exactly as before.

---

## 1. What this artifact IS

**The wearer's own redacted keyframes, prepared to be projected back onto the
room's proxy surface.** It is the appearance half of the saved world; the
surface artifact (`WORLD-BUILDER-SURFACE.md`) is the geometry half.

Per usable keyframe it holds:

- the keyframe's pixels, undistorted into the solve's pinhole camera, as RGBA
  where **alpha 0 means "this pixel is not appearance"** (redaction fill, a
  near-black box, the wearer's hand or phone, and a safety ring around them);
- that camera's intrinsics and world-to-camera pose, copied from the solve;
- a per-channel exposure gain;
- why it was selected for the phone, or why not.

Plus one **proxy mesh**: a byte-exact copy of the surface artifact's phone
level at build time, named by its content digest, so the geometry every source's
visibility was computed against travels with the images.

A renderer draws the proxy and, per fragment, blends the keyframes that saw
that point (view-dependent, unstructured-lumigraph style), dividing each
sample by its keyframe's gain and giving alpha-0 samples zero weight. The
Tower does not render; it prepares.

## 2. What this artifact CLAIMS

1. **Every non-transparent texel is a pixel a camera recorded, after face
   redaction.** Pixels come only from the session's keyframe set
   (`sessions/<sid>/images/<seq>.jpg`, or the re-redacted set it was switched
   to, §6.5), through one provenance function (`appearance.keyframe_source`),
   under an exact redaction-label allowlist (§6). Nothing is inpainted, hallucinated, borrowed
   from a neighbour, or synthesised.
2. **Unobserved stays unobserved.** A texel whose pixel was redaction fill, a
   solid near-black block, or an occluder is transparent. A surface point that
   no keyframe saw through a non-transparent texel has no appearance; the
   renderer shows it as unobserved (dark, hatched, or absent), never as a
   colour taken from somewhere else.
3. **Poses and intrinsics are the solve's, verbatim.** `rotation` and
   `translation` are `solve/<sid>/solution.json`'s, world-to-camera,
   `x_cam = R·X + t`, OpenCV axes (y down). The camera is `solution.camera`,
   the undistorted ROI the solve was run in.
4. **The proxy is the surface artifact's geometry**, not a new reconstruction.
   `proxy.source` names the surface build it was copied from. A better surface
   plugs in by rebuilding this artifact; nothing in this stage changes geometry.
5. **Occluder masks are a depth disagreement, not a detector.** A pixel is an
   occluder in a keyframe when that keyframe's own aligned monocular depth is
   much nearer than the proxy at that pixel (§5.3). It removes the wearer's
   hands and phone where the proxy is behind them. It does not recognise hands,
   and it also removes any real object the proxy lacks (which would otherwise
   be pasted onto the surface behind it).

## 3. What this artifact PROMISES NOT to claim

- **No invented pixels.** No inpaint, no hole filling, no mip or downsample
  that averages transparent texels into opaque ones (a mip chain, if a client
  builds one, must be premultiplied: `sum(rgb·a) / sum(a)`), no generative
  completion. Any future change that fills unobserved texels must break the
  format identifier rather than quietly relax this.
- **Not anonymised.** Redaction is best-effort face detection with measured
  false negatives (§6). This artifact is first-person imagery of a private
  space and inherits the session's `privacy_tags` (`raw-imagery`,
  `first-person`). Screens, documents and bodies other than faces are **not**
  redacted.
- **Not a texture atlas, not a baked colour.** Appearance is view-dependent by
  construction; two keyframes may legitimately disagree about a point.
- **Not metric.** Scale is inherited from the solve; the artifact adds no scale
  claim.
- **Not complete.** The phone tier is a coverage-greedy subset (§5.5); the
  Tower tier keeps every usable keyframe.

## 4. Layout

```
<world>/appearance/<session>/
    manifest.json            what is here; written LAST, atomically
    status.json              the stage's state, with the pid that wrote it
    p.<digest>.bin           the proxy mesh, `WBSURF01` (WORLD-BUILDER-SURFACE.md §5)
    c.<digest>.bin           keyframe bundles (§4.1)
    .appearance.lock         held while a build runs
```

- **Content addressed.** A file's digest is the first 128 bits of the SHA-256 of
  its bytes, 32 lower-hex characters (not 64: a Windows path is capped at 260
  characters, and the first canonical build's staging path came to 263).
  Readers resolve files only through the manifest and refuse a file whose size
  or digest disagrees with it.
- **Published as a unit.** Chunks and proxy are written first under their final
  content names (atomic write each); the manifest is written last. A build that
  is stopped or killed before its manifest leaves files nothing names, and the
  previous manifest keeps naming the previous build's whole set.
- **Pruning** follows the surface artifact exactly (`WORLD-BUILDER-SURFACE.md`
  §4): a file the replaced manifest named is stamped at supersession and removed
  two minutes later; a file no manifest ever named is removed once two minutes
  old; staging files after ten minutes; an unreadable manifest prunes nothing.
- **Purge.** A world purge removes the world directory and this with it.

### 4.1 `c.<digest>.bin` — a chunk

Little-endian.

| offset | type | field |
|---:|---|---|
| 0 | `char[8]` | magic `WBAPCK01` |
| 8 | `uint32` | schema version (1) |
| 12 | `uint32` | encoding (1 `astc-6x6-rgba`, 2 `webp-rgba`) |
| 16 | `uint32` | slot count `n` |
| 20 | `uint32` | image width |
| 24 | `uint32` | image height |
| 28 | `uint32` | reserved (0) |
| 32 | `uint32[2]` × `n` | per slot: offset from the start of the payload, byte length |
| 32 + 8n | … | payload |

- `astc-6x6-rgba`: raw ASTC LDR blocks, `ceil(w/6) × ceil(h/6) × 16` bytes per
  slot, no `.astc` file header — exactly what `compressedTexSubImage3D` with
  `COMPRESSED_RGBA_ASTC_6x6_KHR` takes. The phone format (§5.6).
- `webp-rgba`: a complete WebP file per slot, lossy RGB, **lossless alpha**.
  The portable fallback for desktop and debugging. A frame with no transparent
  texel is written without an alpha plane (libwebp drops an all-255 one): read it
  as fully opaque.

A slot's keyframe is named by the manifest, never by the chunk. Up to 16 slots
per chunk.

### 4.2 Texel convention

A texel `(i, j)` covers `[i, i+1) × [j, j+1)` in pixel units and its centre is
`(i + 0.5, j + 0.5)` — the COLMAP convention the solve's observations use. A
world point `X` projects to `u = fx·x/z + cx`, `v = fy·y/z + cy` with
`(x, y, z) = R·X + t`; the texture coordinate is `(u / width, v / height)`.
Row 0 is the top of the image.

## 5. How it is built

Stage order, all over the gated, posed keyframes of one session:

1. **Provenance** (§6): bytes, effective label, unobserved mask, mask origin.
   Refused frames contribute nothing and are recorded.
2. **Proxy depth.** The proxy is ray-cast from every usable keyframe's pose at
   full resolution (Embree through Open3D, CPU): nearest hit, either facing.
3. **Near occluders** (§5.3, first test).
4. **Exposure gains** (§5.4), over what is still opaque.
5. **Transients** (§5.3, second test), which needs the gains.
6. **Selection** (§5.5).
7. **Encoding** (§5.6), chunks, proxy copy, manifest.

### 5.1 Which keyframes are usable

A keyframe is a candidate when it is posed in the solve and has a depth-stage
record in `dense/<sid>/align.json` whose fit is `ok` (the occluder test needs
its depth). Every other keyframe is listed in `excluded` with its reason:
`no-pose`, `no-depth-fit`, `refused-*` (§6, including
`refused-camera-mismatch`), `no-proxy-in-view`, `no-depth-prediction`.

### 5.2 Unobserved mask

`unobserved = dilate(stored_or_rerun_fill ∪ nearblack, 2 px)` in the
undistorted camera, where

- `stored_or_rerun_fill` is §6's fill mask (already dilated 3 px by the depth
  stage, remapped nearest);
- `nearblack` is every pixel whose three channels are ≤ 6 in the undistorted
  redacted image, morphologically opened with a 16×16 box. It catches the fill
  boxes a stored mask misses where they touch dark scene (seven frames on the
  canonical world), at the price of some genuinely black scene.

Rule string in the manifest: `fill2|nearblack6open16|dilate2`.

### 5.3 Occluders: the wearer's hands and phone

Two tests. A pixel that fails either is an occluder in that keyframe: RGB
zeroed, alpha 0 (§5.6). Neither recognises a hand.

**Near: the keyframe's own depth is far nearer than the proxy.** For a keyframe
with proxy depth `Zp` and aligned source depth
`Zs = depth_from_prediction(pred, a, b, kind)` (the raw prediction
`work/depth/<ki>_pred.npy` with the fit's affine; `kind` from `align.json`):

- over pixels where both are finite and positive and not unobserved,
  `r = Zs / Zp`, `s = median(r)`, `mad = median(|r − s|) / s`;
- threshold `τ = s × clamp(1 − 4·mad, 0.3, 0.6)`: never looser than 0.6 of the
  proxy depth, and stricter where the keyframe's depth disagrees with the proxy
  more (a noisy frame flags less rather than more);
- `near = r < τ`, opened 5×5, components under 0.2% of the frame dropped,
  dilated 4 px. Pixels with no proxy behind them are not tested.

This removes what the proxy does not contain: a hand held up in front of the
camera, the chair back or the wearer's own shoulder at the frame edge, a shelf
edge where the proxy has a hole. **It cannot remove a hand lying on the desk**:
measured on the canonical world, the proxy point under the hand in ki 330 is
1.90 from the camera and is confirmed within 5% by 146 other keyframes' own
depth. Geometry cannot tell a hand from the desk it rests on.

**Transient: most other keyframes saw something else there.** On an 8-pixel
grid of each keyframe's proxy points, colour (15×15 box, divided by the gain) is
sampled in every other keyframe where the point is the nearest proxy surface
(3%), on an opaque texel, 8 px from the border, and neither near-black (≤ 0.03)
nor clipped (≥ 0.92) before the gain. The reference is the per-channel median
over those keyframes. After one scale per keyframe (the median ratio of its own
colours to the references, so an exposure error is not a transient), a grid
point is transient when:

- at least 6 other keyframes saw it, and at least 70% of them agree with the
  median (their largest channel difference within the tolerance), and
- this keyframe's largest channel difference exceeds `max(0.12, 0.5 × median
  luma)` **at every one of nine offsets** (0 or ±6 px in each axis, opaque
  texels only). A keyframe misregistered by a pixel or two otherwise disagrees
  in strips along every high-contrast edge; along an edge some offset matches,
  inside a hand none does.

Measured on the canonical world, the offset is a trade: at ±3 px the test
removed more of the hand in ki 330 (4.2% of the frame against 1.9%) and also
striped the shelf boards of ki 255 (11.4% against 4.7%). ±6 px was chosen: hands
and phones on the desk are **partly** removed (fully in ki 366 and 340, the
phone and part of a hand in 330 and 351, little in 308), and residual hand
pixels remain possible in any keyframe.

A keyframe that disagrees on more than 20% of its tested points is treated as
**misregistered** (a pose or proxy error), not as full of transients: it is left
unmasked, listed in `occluders.misregistered_frames_unmasked`, and its
selection quality is multiplied by `max(0.15, 1 − fraction)`. Otherwise its hits
are lifted to full resolution (each covers its 8×8 cell; a cell with no hit
neighbour is dropped; components under 0.2% of the frame dropped; dilated
4 px).

Its cost: it also removes view-dependent appearance a keyframe disagrees with
the rest about — a clipped door edge, a monitor showing other content, the
glow of an LED — from that keyframe only. Other keyframes still supply those
points.

The per-keyframe record keeps `near_fraction`, `transient_fraction`,
`occluder: {median_ratio, mad, threshold}` and
`transient: {tested, disagreeing_fraction, applied}`.

### 5.4 Exposure gains

Per keyframe, per channel, `g` such that the recorded 8-bit value is
`g × radiance`, gauge `mean(log g) = 0` over usable keyframes. Solved by
alternating Huber-IRLS means (δ = 0.15 in log, 30 iterations) over co-visible
samples: a 16-pixel grid of each keyframe's proxy depth, back-projected and
re-projected into every keyframe; a sample is kept when it is inside that frame
8 px from the border, within 3% of that frame's proxy depth, on an opaque texel
there, and neither dark (< 0.02) nor saturated (> 0.97) in a 7×7 box blur;
points seen by fewer than three keyframes are dropped. A keyframe with no
kept observation has gain `[1, 1, 1]` and `gain_observations: 0`.

**A renderer divides the decoded channel (0…1) by `gain`.**

### 5.5 Selection

Greedy set cover over proxy visibility, with a quality weight.

- 60,000 area-weighted points on the proxy (fixed seed).
- A keyframe scores a point `|cos(normal, ray)| × min(1, z_ref / z) × q` when the
  point is in frame, within 3% of that keyframe's proxy depth, and on an opaque
  texel; otherwise 0. `z_ref` is the median proxy depth over usable keyframes.
- `q = clamp(sharpness / median sharpness, 0.15, 1) × (1 − 0.5 × transparent
  fraction)`, sharpness the variance of the Laplacian over opaque pixels, and
  lowered further for a misregistered keyframe (§5.3).
- Objective `Σ best + 0.5 × Σ second-best`, so most surface keeps two views to
  blend. Keyframes are added in order of marginal gain until the phone budget
  (`phone_budget`, 128) is reached or the gain falls below `1e-4` of the
  objective so far.
- `tier: "phone"` for the chosen keyframes (with `rank`), `tier: "tower"` for
  every other usable keyframe. `selection.coverage` reports the fraction of
  points seen at least once and at least twice by each tier.

### 5.6 Encoding

- **Phone: ASTC 6×6 LDR, RGBA**, encoded with `astc-encoder-py` (a pip wheel of
  ARM's `astcenc`; MIT wrapper, Apache-2.0 encoder) at quality 60 ("medium").
  iOS exposes `WEBGL_compressed_texture_astc` on essentially every device.
- **Everywhere: WebP RGBA**, lossy RGB quality 90, lossless alpha (OpenCV).
- **Transparent texels.** RGB is zeroed inside `unobserved ∪ occluder`; alpha is
  0 over that set dilated by 7 px (one 6×6 block plus a bilinear tap), so no
  block that holds a zeroed texel also holds an opaque one, and no bilinear
  fetch of an opaque texel reaches a zeroed one. A renderer treats
  `alpha < 0.5` as zero weight.
- **Chunks.** Phone keyframes in rank order, 16 per chunk, in both encodings;
  Tower-tier keyframes in keyframe order, 16 per chunk, WebP only.

Evidence (16 canonical keyframes, opaque texels, against the undistorted
redacted source):

| encoding | PSNR mean / min | bytes per keyframe | encode |
|---|---|---|---|
| **ASTC 6×6 (astc-encoder-py, q60)** | **43.8 / 40.8 dB** | **102,720** | 45 ms (1 thread) |
| ASTC 4×4 (same) | 49.8 / 46.0 dB | 230,400 | 87 ms |
| ETC2 RGBA8 (etcpak 0.9.15) | 39.5 / 37.2 dB | 230,400 | 4 ms |
| WebP q90, lossless alpha | 42.9 / 40.8 dB | 16,539 | 14 ms |

ETC2 was rejected: 4 dB worse than ASTC 6×6 at 2.24× the GPU memory. Alpha
survived exactly in every encoding measured.

## 6. Provenance and privacy

### 6.1 The one source

Pixels are read only from the session's keyframe set, only inside
`appearance.keyframe_source`. Which set is `WorldStore.keyframe_image_set`'s
answer, read once per build with its label: `sessions/<sid>/images/` under
`session.json`'s `redaction`, or the re-redacted set a session was explicitly
switched to, under that set's label (§6.5). This stage never opens `solve/<sid>/images`
(unredacted), raw captures, `sources.json`, `dense/<sid>/work/undist`
(TELEA-inpainted inside the fill), `solution.rgb` or `points.json` colours.
The undistortion is the solve's own (`global_solve._undistort_maps`).

### 6.2 The label

The keyframe set's label -- `session.json` `redaction`, or the re-redacted
set's (§6.5) -- is read once per build with the set, recorded, and read again
before the manifest is written; a change of either in between aborts the
publish.

| session label | treatment |
|---|---|
| `faces-detected-and-filled/yunet-2023mar@0.30` (ungated; a superset of every gate) | trusted |
| `…@0.30+plausibility1` (0 / 3,232 held-out close faces lost) | trusted |
| `…@0.30+plausibility3` (1 / 3,232) | trusted |
| `…@0.30+plausibility2` (32 / 3,232; weak at the frame edge) | re-redacted |
| `none`, empty, absent, unreadable, **any other string** | re-redacted |

- **Trusted.** The stored bytes are used. The fill mask is the depth stage's
  stored `_fill.npy`, and only when its `align.json` record has this
  `fill_rule` and an `image_sha1` equal to the SHA-1 of the bytes read now.
  **A missing, mismatched or wrong-shaped mask refuses the frame**
  (`refused-no-fill-mask`); it is never read as "nothing filled".
- **Re-redacted.** The current `FaceRedactor` runs over the stored bytes
  (a fill only adds pixels). The fill mask is the exact difference against the
  stored image, united with the shape-gated guess, united with the stored mask
  when its record matches the re-redacted bytes. A result labelled `none`
  refuses that frame (`refused-redaction-failed`). **If the redactor is
  unavailable the whole build is refused** (`unavailable`), because an
  appearance layer silently missing the room is worse than a clear refusal.
  The effective label is `<stored>&<current>`.
- **Purged.** `world.images_purged` refuses the build and every route.

During a walk the session label is `none` until Stop, so live builds are
re-redacted; the final build after Stop sees the real label.

### 6.3 What the manifest records

`appearance_provenance`:

| key | meaning |
|---|---|
| `session_redaction` | the label of the keyframe set read: `session.json`'s, or the re-redacted set's (§6.5); `null` when absent or unreadable |
| `keyframe_image_set` | `null` for `sessions/<sid>/images/`; `images.redacted-<gate>@<set digest>` when the session was switched to a re-redacted set (§6.5) |
| `redaction_effective` | the label actually applied |
| `redactor_applied_here` | `FaceRedactor.label` when this build re-redacted, else `null` |
| `label_trusted` | whether the stored label was on the allowlist |
| `fill_rule` | the depth stage's `FILL_RULE` the stored masks were required to carry |
| `unobserved_rule` | `fill2|nearblack6open16|dilate2` |
| `source` | `session-keyframes` |
| `frames` | `{used, refused: {reason: count}}` |
| `per_frame_sha1_digest` | SHA-1 over the sorted `(keyframe id, SHA-1 of the stored JPEG)` pairs |
| `privacy_tags`, `retains_raw_imagery` | inherited from the session |
| `note` | "best-effort redaction with measured false negatives; not anonymised" |

Per keyframe: `source_sha1`, `image_sha1` (of the bytes the pixels came from),
`origin`, `mask_origin` (`stored-fill` or `rerun-difference+guess`, each
`+nearblack`), unobserved and occluder fractions.

### 6.4 Cache key

`params_digest` is the SHA-256 of: the schema version, the solve's
`input_digest`, the proxy's SHA-256 and its source surface build, the
depth stage's `cache_key`, `session_redaction`, `keyframe_image_set`,
`redactor_applied_here`,
`fill_rule`, `unobserved_rule`, `per_frame_sha1_digest`, the occluder, exposure
and selection parameters, and the encoder names and versions. Any change to any
of them rebuilds; an equal digest with every named file whole is
"already built" (`--force` rebuilds anyway).

### 6.5 Re-redaction: an explicit switch to a lighter, current redaction

Worlds captured under an **older rule of the same family** -- the ungated
`faces-detected-and-filled/yunet-2023mar@0.30`, `…+plausibility1`,
`…+plausibility2` -- carry that rule's false positives. On the canonical capture
the ungated rule filled 12.61% of all pixels (box masks), almost all of it the
wearer's hands, a lit PC case and bare wall; `plausibility3` on the same raw
frames fills 5.99%, and the page's opening pose (ki 148) goes from 52.0% to 0%.
`scripts/world_reredact.py` recovers that, by hand, never during a walk:

```
.venv\Scripts\python.exe scripts/world_reredact.py --root <root> --world <id> [--session <sid>] --dry-run
.venv\Scripts\python.exe scripts/world_reredact.py --root <root> --world <id> [--session <sid>] --apply
.venv\Scripts\python.exe scripts/world_reredact.py --root <root> --world <id> [--session <sid>] --revert
```

**What it writes.** `sessions/<sid>/images.redacted-<gate>/` (for today's gate,
`images.redacted-plausibility3/`): one `<seq>.jpg` per keyframe and a
`record.json` (`wb-reredaction-record/1`: tool version, stored label, set label,
redactor label and model, the verification rule, the invariant, per-origin
counts, fill totals, the stored and set digests, and per frame its `origin`,
`detail`, `stored_sha1`, `sha1`, `raw_sha1`, fill pixel counts and recovered
pixels). Then, **last and atomically**, `sessions/<sid>/redaction_set.json`
(`wb-redaction-set/1`): `{active, redaction, stored_redaction, set_digest,
record, tool_version, switched_at, history}`. `sessions/<sid>/images/` and
`session.json` are never written. `--revert` rewrites the pointer with
`active: null`; the set stays on disk, and a later `--apply` of the same result
re-points to it.

**Who honours it: one accessor.** `WorldStore.keyframe_image_set(world,
session)` returns the directory and the label every build reads: the dense
depth stage (`keyframe_image_bytes`), the surface (through the depth stage), this
stage (`keyframe_source`, the label policy, the pack-time and serve-time label
checks) and so the render revision's `appearance` object. A pointer is honoured
only when `active` names an existing `images.redacted-*` directory beside
`images/`, it carries a label and a digest, and its `stored_redaction` still
equals `session.json`'s `redaction`; anything else reads as no switch.

**Per frame.** A frame's set image is the current redactor's output on its raw
capture frame only if all of these hold, else the set holds the **stored
keyframe's bytes** and the record says why:

| check | origin when it fails |
|---|---|
| a `sources.json` entry exists and reads, resolved against the Tower root (`TOWER_SOURCES_ROOT`, else `tower/`), never the cwd | `stored:raw-missing` |
| the raw frame decodes at the stored keyframe's exact shape, every difference over 40 lies within 2 px of stored near-black (≤ 12) with at most 0.1% unexplained, and the mean absolute difference elsewhere is ≤ 3 (canonical: 398/398 true pairs, 0/397 adjacent keyframes) | `stored:raw-unverified` |
| the redactor's result is not labelled `none` | `stored:redaction-none` |
| **the new fill lies inside the stored fill** (fill = output ≤ 12 where the raw frame differs by > 25; stored fill dilated 3 px): re-redaction may only un-fill | `stored:fill-outside-stored` (logged) |
| the output is a fresh image, the raw frame with nothing changed by > 40 further than 4 px from its fill | `stored:output-not-raw-plus-fill` |
| at least 64 px were recovered | `stored:nothing-recovered` |

Raw bytes are read in exactly one function (`reredaction.reredact_frame`) and
never leave it; a set image is the redactor's own encode, a fresh q90 encode of
the decoded frame when nothing was filled, or the stored bytes.

**The whole step refuses, and nothing is switched,** when: the label is `none`,
empty, absent or unknown (those take §6.2's re-redaction of the stored bytes); it
is already the current label; the session has not ended; the world is purged or
held by a live writer; no redactor is available, or its label is not the one
this step writes (`reredaction.TARGET_LABEL`, versioned in code); any exception
is raised; a stored keyframe is unreadable or changes during the run; no frame
recovers anything; or a pointer is already active.

**The set's label.** A switched session reads under the current label, so the
allowlist in §6.2 applies unchanged. That is honest only if every frame's fill
contains what the current rule fills on its raw frame: a re-redacted frame is
that rule's output; a frame kept as `stored:nothing-recovered` has the same
fill; a frame kept for any other reason holds the stored rule's fill, which
contains the current rule's only for the ungated rule (every gate only removes
boxes from the same detection pass). **So from `plausibility1` or
`plausibility2`, an apply that would keep any frame for another reason is
refused whole.**

**Caches.** A switch changes the pixels without changing the solve, so the
set's identity (`images.redacted-<gate>@<set digest>`, absent for `images/`)
is part of the depth stage's cache key and `align.json`
(`keyframe_image_set`), the fuse key, the dense points manifest, the surface
`params_digest`, and this artifact's `params_digest` and provenance. It is
appended only when a set is active, so no cache made before this existed is
invalidated. Depth predictions are still reused per frame by image SHA-1, so an
apply re-predicts only the frames whose bytes changed. After `--apply` (or
`--revert`): `world_surface.py`, then `world_appearance.py`.

**Measured on a copy of the canonical world** (b2a75ab4…, 398 keyframes,
ungated label): 89 frames re-redacted, 309 `stored:nothing-recovered`, no
invariant failure, no unverified raw frame; box-mask fill 12.61% → 5.99%
(difference masks 9.83% → 4.86%), 22 frames over half filled (was 38), 7 over
70% (was 16); ki 148 52.0% → 0%, ki 120 63.7% unchanged. 24 s on the CPU.

**What it does not change.** Face recall on a real bystander is not measured by
that capture (it has none); the case for `plausibility3` rests on its composites
(1 of 3,232 held-out close faces lost against the ungated rule). Recovered pixels
are first-person imagery (the wearer's hands and legs, screens, room contents)
and inherit the session's privacy tags. When a world's raw frames are gone the
step cannot run; a set already written is unaffected.

## 7. `manifest.json`

| key | meaning |
|---|---|
| `format` | `wb-appearance-keyframes/1` |
| `schema_version` | 1 |
| `build_id` | opaque, unique per publish |
| `built_at` | epoch seconds |
| `quality` | `live` or `final` |
| `input_digest` | the solve this was built from |
| `params_digest` | §6.4 |
| `params` | every parameter |
| `appearance_provenance` | §6.3 |
| `camera` | `{fx, fy, cx, cy, width, height}`, the solve camera |
| `proxy` | `{digest, bytes, vertices, faces, format: "wb-surface-mesh/1", source: {surface_built_at, surface_input_digest, surface_params_digest, surface_quality, level}}` |
| `encodings` | per encoding: `{name, texel_format, block, encoder, version, quality, available, ...}`. ASTC is absent when its encoder is not installed, with the reason in `encoding_notes`; the WebP fallback is always present |
| `chunks` | `[{digest, bytes, encoding, slots, tier}]` |
| `keyframes` | per usable keyframe, below |
| `excluded` | `[{ki, reason}]` |
| `selection` | `{phone_budget, phone, tower, coverage: {phone: {seen1, seen2}, all: {…}}, objective}` |
| `exposure` | `{observations, points, abs_log_residual_before, abs_log_residual_after, gain_range}`; observations are a seeded sample of at most 6,000,000 |
| `occluders` | `{frames_with_occluders, total_pixels, mean_fraction, near_mean_fraction, transient_mean_fraction, frames_with_near, frames_with_transient, misregistered_frames_unmasked, rule}` |
| `seconds` | per stage |
| `scale` | inherited, with a note |

Per keyframe:

| key | meaning |
|---|---|
| `id` | opaque, `sha256(session_id + ":" + keyframe_id)[:16]` |
| `ki` | index into the solve's `keyframe_ids` (not a capture sequence number) |
| `tier`, `rank` | `phone` with its greedy rank, or `tower` with `rank: null` |
| `width`, `height` | image size |
| `intrinsics` | `{fx, fy, cx, cy}` |
| `rotation` | 9 floats, row-major, world-to-camera |
| `translation` | 3 floats |
| `gain` | 3 floats, divide by it |
| `gain_observations` | samples that constrained it |
| `quality` | the selection weight `q` |
| `sharpness` | variance of the Laplacian over opaque pixels |
| `unobserved_fraction`, `occluder_fraction`, `near_fraction`, `transient_fraction` | of the frame, before the alpha ring |
| `transparent_fraction` | alpha 0, ring included |
| `occluder` | the near test's `{median_ratio, mad, threshold}`, or `null` when there was too little overlap to measure |
| `transient` | `{tested, disagreeing_fraction, applied}`, or `null` |
| `source_sha1`, `image_sha1`, `origin`, `mask_origin` | §6.3 |
| `chunks` | `{<encoding>: {digest, slot}}` |

## 8. Staleness

`appearance_currency()` reports `{present, current, reason}`:
`current` is false when the solve now on disk has another `input_digest`
("built from an earlier solve") or the surface manifest now on disk is a
different build from `proxy.source` ("built on an earlier surface").
Reported, never enforced: an earlier appearance is still the best one there is,
**except** for the redaction label, which is enforced (§9).

## 9. Routes

All `GET`, same origin and containment rules as `WORLD-BUILDER-WORLDS.md` §4
(`contained_world_id`; `session_id` must be one of the world's sessions).

| route | body |
|---|---|
| `/worlds/{world_id}/appearance/{session_id}/manifest` | the manifest, plus `currency` (§8) |
| `/worlds/{world_id}/appearance/{session_id}/chunk/{digest}` | chunk bytes, `application/octet-stream` |
| `/worlds/{world_id}/appearance/{session_id}/proxy/{digest}` | proxy mesh bytes, `application/octet-stream` |

`digest` must be 32 lower-hex characters **and** named by the current manifest;
anything else is 404. URLs carry no path, file name or sequence number.

**Headers on every response, 200 or 404:** `Cache-Control: no-store`,
`Pragma: no-cache`, `X-Content-Type-Options: nosniff`, and no `ETag` or
`Last-Modified`. A 200 also carries `X-World-Redaction: <redaction_effective>`.

**Served only while all of these hold, else 404** with a `detail`:

- a world root is configured, the world and session exist, the world is not
  `images_purged` ("appearance imagery was purged");
- a manifest exists and reads ("no appearance for this session");
- the manifest's `session_redaction` and `keyframe_image_set` equal the
  session's keyframe set's label and identity **now** (§6.5) ("appearance is
  stale against the session's redaction record");
- for a file, the manifest names it and its bytes on disk have the recorded
  size ("no such appearance file").

The label is re-checked on every request, so a relabelled session stops
serving its old textures immediately.

`GET /worlds/{world_id}/render/revision` (`WORLD-BUILDER-WORLDS.md` §4a) carries
an additive `appearance` object: `{"revision": str | null, "current": bool}`.
`revision` is `<session_id>/appearance:<build_id>` exactly when the manifest
route would answer 200, else `null`. It is opaque and compared for equality. It
is deliberately **not** folded into the page `revision`: the page follows
appearance builds itself, and a page swap for every appearance build would
reset the wearer's camera.

**Who calls these.** Only the appearance page of `WORLD-BUILDER-WORLDS.md` §4,
through the transport it names: on the phone the app's `glasses-world:` scheme
handler, which proxies exactly these three routes and the revision route for the
world and session on screen, through an ephemeral session with no URL cache, and
drops its in-memory copy of bundles whenever a manifest or revision request stops
answering with a served appearance (`WORLD-BUILDER-IOS.md` §10).

## 10. Live, final, rebuilding

- **Live.** During a walk the live surface child (`world_surface.py --live
  --appearance`) builds the appearance right after each live surface, against
  that surface's phone level, with `quality: live` (30,000 selection samples,
  a 24-pixel exposure grid, a 12-pixel transient grid; the privacy and occluder
  rules do not move). A live appearance is re-redacted (the label is `none`
  until Stop): 27.5 ms a keyframe with the current redactor. Measured on the
  canonical world (374 keyframes, trusted label so no re-redaction): 45 s on a
  quiet card, 90 s with another lane holding it at 98%.
- **Final.** After Stop the builder runs the final surface, then the final
  appearance, then prunes the depth stage's work — in that order, because the
  appearance needs `_fill.npy` and `_pred.npy`.
- **Rebuild** from authoritative data alone:

  ```
  .venv\Scripts\python.exe scripts/world_appearance.py --root <root> --world <id> [--session <sid>] [--force]
  .venv\Scripts\python.exe scripts/world_appearance.py --root <root> --world <id> --inspect
  ```

  If the depth work has been pruned, run `world_surface.py --force` first
  (it recomputes it); the appearance stage never runs the depth network.

**States.** `status.json` `state` is `running`, `ok`, `stopped`, `failed` or
`unavailable` (with `detail`), like the surface stage.
