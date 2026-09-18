# World Builder saved worlds — the listing a viewer opens old worlds from

**Living document.** Added 2026-09-06.

| | |
|---|---|
| Contract | `world_builder.worlds/2026-09-10` |
| Transport | HTTP `GET /worlds`, same origin and same rules as the geometry routes (`WORLD-BUILDER-GEOMETRY.md` §1) |
| Tower producer | `tower/tower/results/world_builder_library.py` |
| Tower route | `tower/tower/routes/geometry.py` |
| iOS consumer | `ios/Glasses/Workspaces/WorldBuilder/WorldLibrary.swift` (compiled and tested on the Mac, 2026-09-06; the 2026-09-10 id bump has **not** been compiled — Windows lead) |

## 1. Why it exists

The Tower already serves any world by id (`/worlds/{id}/geometry/…`) and the
status subscription already pins a stored world (`result_subscribe` with
`world_id` / `session_id`, `WORLD-BUILDER-IOS.md` §6). What no client could do
was *find* an id. This is the index, and nothing more: it carries no geometry,
no imagery, no per-keyframe data.

## 2. `GET /worlds`

**404** when no world root is configured. **200** otherwise, including when
there are no worlds (`worlds: []`).

| Field | Type | Meaning |
|---|---|---|
| `contract` | string | `world_builder.worlds/2026-09-10`. Compared for equality; a mismatch refuses the payload |
| `world_count` | int | Length of `worlds` |
| `worlds` | array | Newest `updated_at` first |

Per world:

| Field | Type | Meaning |
|---|---|---|
| `world_id` | string | The id the geometry routes and the status pin take |
| `display_name` | string \| null | As stored. `null` means unnamed, never `""` |
| `created_at`, `updated_at` | number | Tower-receipt epoch seconds |
| `live` | bool | A builder process holds this world's writer lock **right now** (its pid is running). A live world is still changing; open it as such |
| `session_count` | int | Length of `sessions` |
| `sessions` | array | Oldest first |

Per session:

| Field | Type | Meaning |
|---|---|---|
| `session_id` | string | The id the geometry routes take |
| `started_at` | number | |
| `ended_at` | number \| null | `null` means still open, **not** zero |
| `end_reason` | string \| null | `stop`, a disconnect reason, or `null` while open |
| `frame_source` | string | `live-capture`, `recorded-capture`, `synthetic` |
| `capture_id` | string \| null | The capture this session followed, when it followed one |
| `keyframes_accepted` | int | As recorded on the session |
| `has_geometry` | bool | **Opening this session would show the wearer something.** The geometry manifest may still answer `current: false`. **Amended 2026-09-10 — this is why the contract id moved.** It used to mean only that `poses.json` and `points.json` exist, and `engine.build` writes both unconditionally: a walk that solved nothing leaves a `points.json` holding `{"points": []}`, 14 bytes. **Eleven sessions on the real 163-world root were reported `has_geometry: true` over exactly that**, while the status channel for the same session projected `needsRetry` — the picker promising a wearer something to look at and the panel behind it saying nothing came of the walk. It is now the same question the status channel asks, of the same figures: `points > 0` or `poses_positioned > 0`, from the session's manifest, or counted from `points.json` when no manifest describes it |
| `abandoned` | bool | `ended_at` is `null` **and** the world is not `live`: the builder ended without finalising the record (killed inside the Tower's shutdown grace, or by hand). Its `keyframes_accepted` is then the start-of-session value, not the journal length, and its geometry is whatever the last interim build wrote. Additive (2026-09-06, Windows validation) |
| `state` | string | One word, the status channel's lifecycle vocabulary: `receiving` (a live builder, session open), `finalizing` (a live builder, session stopped — the final solve and build are running), `complete` (finished; `finalization.state == "complete"` **and** geometry on disk, or an older record that stopped and built), `interrupted` (killed mid-walk, killed mid-finalization, stopped by a request or an error, **or a session whose manifest records real figures and whose derived tree is gone** — something happened to this session, which is what the word claims), `unbuilt` (**there is nothing to open**: a record that stopped and never built, or one whose build ran and found nothing — a dark corridor, a blank wall, a calibration that never arrived. iOS renders it “No geometry”, which is the whole claim. It is deliberately NOT `interrupted`: nothing was interrupted, and saying so tells a wearer the walk failed when it merely found nothing). Additive (2026-09-06, live-history lane); **amended 2026-09-10**, which is why the contract id moved: `complete` gained the `has_geometry` requirement and `interrupted` gained the manifest-without-a-tree case, so a word a phone already implements now arrives in states it did not before. An older phone does **not** decode this payload at all: `WorldListingDecoder.listing` compares the id for equality and returns `nil` for the whole body on a mismatch, so Saved Worlds is empty rather than wrong. That is the intended behaviour of a dated identifier and it is why the app must be rebuilt from this branch. |
| `keyframes_journaled` | int | Lines in the keyframe journal. On a record that never stopped `keyframes_accepted` is still the zero written at start; this is what actually landed (467 on the 09-06 walk against a recorded 0). Show this when the two disagree. Additive (2026-09-06) |
| `finalization` | object \| null | The builder's own record of what happened after the session stopped: `{state: pending\|complete\|interrupted, final_solve: pending\|solved\|skipped\|failed\|unavailable\|null, started_at, updated_at, detail}`. `null` on records written before 2026-09-06 and on sessions that never stopped. Additive (2026-09-06) |
| `appearance` | object \| null | **The appearance artifact, reported as the imagery it is** (additive, 2026-09-17, review 1 m6): `{format, state: served\|rebuilding\|withdrawn, quality, keyframes, keyframes_phone, bytes, redaction, redaction_effective, label_trusted, keyframe_image_set, privacy_tags, retains_raw_imagery, imagery, retention}`. `state` is `APPEARANCE.md` §9's: whether the routes serve it now. `redaction` is the label it was built under and `redaction_effective` what was applied; `imagery` says *first-person keyframe imagery of a private space; best-effort face redaction with measured false negatives; not anonymised*; `retention` says it is kept with the world until rebuilt or purged. No URL, id or path: the page reaches it through §4b only. `null` when the session has none |

## 3. Rules

1. **Absent is never zero** (`WORLD-BUILDER-GEOMETRY.md` §6 applies): `ended_at: null` is an open session; `display_name: null` is an unnamed world. An open session with `abandoned: true` is not still open: nobody is writing it, and a client should say so rather than "still open". Prefer `state` when it is present; `abandoned` is kept for clients that predate it.
5. **An interrupted session is not hidden and is not presented as finished.** `state: "interrupted"` with `has_geometry: true` is a world a person can open and look at (the render route serves it); the row must say interrupted and must not say complete. A live Tower has no reason to hide the 2026-09-06 walk, and every reason not to call it finished.
6. **Worlds with no sessions are shells**, left by a builder that opened a world and never received a frame (96 of 162 on the Windows box). A picker may fold them away; it must not offer them as the primary rows.
2. **A world that cannot be read is omitted, never invented.** A session that cannot be read is omitted from its world.
3. **No imagery, no paths.** `capture_id` is an opaque id, not a location. This listing still carries none. Imagery is served by exactly one route family, §4b, deliberately and under its own contract; nothing here or in §4 or §4a carries it.
4. Opening a world means: subscribe with `world_id` (+ `session_id`) on the status channel, then pull geometry exactly as for the live world. There is no second geometry path.

## 4. `GET /worlds/{world_id}/render` — the interactive viewer

Added 2026-09-06 on the Mac integration branch, so a saved world can be
*looked at* from inside the app. Same origin and rules as §2.

| | |
|---|---|
| Tower producer | `tower/tower/results/world_builder_render.py`, composing with `tower/tower/world_builder/render.py` — the same code `scripts/world_render.py` writes `world.html` with |
| iOS consumer | `ios/Glasses/Workspaces/WorldBuilder/WorldRenderViewer.swift` (a `WKWebView` that loads nothing else) |

| Query | Type | Meaning |
|---|---|---|
| `session_id` | string, optional | The session to draw. Absent: the newest session of the world that has geometry (`has_geometry` in §2) |
| `max_points` | int 1…200000, optional | Point budget, and only a point budget. **Sparse:** default 40,000 for a phone (`MOBILE_MAX_POINTS` in `tower/results/world_builder_render.py`, set from a measured canvas-fill cliff — 80,000 was the cliff, not a margin). Fractional-stride sampling over every segment, never a prefix. **Dense:** default 393,216 (a 6 MB binary buffer), met by a coarser voxel grid over the whole extent rather than by sampling; an explicit value is honoured as a cap. **Surface:** the mesh is not points, so this selects the level-of-detail rung rather than thinning vertices: it lowers the page budget to `max_points x 16` bytes, and the largest rung whose composed page (mesh base64-encoded, plus the viewer) fits it is served; when none fits, the smallest is served over budget, so it cannot go below the default phone page. The default page budget is 6 MiB (`WORLD-BUILDER-SURFACE.md` §8). The dense page's budget is on its binary buffer, not the page, which can therefore exceed 6 MiB (8.0 MB measured) |
| `representation` | `auto` \| `sparse` \| `dense` \| `surface` \| `appearance`, optional | Which reconstruction to serve, as a ladder: `appearance` → `surface` → `dense` → `sparse` (`appearance` added 2026-09-17). `auto` (default) serves the best rung the session actually has **that the client can draw**: the `appearance` rung only when the request declares `viewer=appearance-1` (below), otherwise the walk starts at `surface`. A named value starts the walk at its own rung and falls through to worse ones, **except** that naming a rung the session does not have at all returns **404** rather than silently serving a different one — a caller that pinned a representation is comparing, and a silent substitution would corrupt the comparison. **422** outside this set. The rung actually served is stated in the page (`wb-representation`) and by `GET /worlds/{id}/render/revision` (§4a); the `GET /worlds` listing does not state it |
| `transport` | `app` \| `tower`, optional | Where the **appearance** page fetches its imagery from; every other page fetches nothing and ignores it. `app` (default, and the only value the phone sends): `glasses-world://tower/…`, the app's private scheme, CSP `connect-src glasses-world:`. `tower`: a desktop debug mode, only when named, fetching from the Tower's own origin, CSP `connect-src 'self'`. **422** outside this set. See "The appearance page" below |
| `viewer` | comma-separated capability tokens, optional | What the client can draw (added 2026-09-17). **`appearance-1`** declares a client that can load the appearance page — one with the `glasses-world:` scheme handler of `WORLD-BUILDER-IOS.md` §10. `representation=auto` offers the `appearance` rung **only** when it is declared; without it `auto` starts at `surface`, byte-for-byte the page served before the rung existed. The reason is old apps: an iOS build older than the appearance page loads every page with `loadHTMLString`, has no scheme handler, cannot fetch the imagery, and would show a broken page. A pinned `representation=appearance` is served regardless (a caller naming the rung is asking for it). Unknown tokens are ignored and no value is ever a 422, so a newer app talking to an older Tower still gets a picture (FastAPI ignores the parameter on a Tower older than it). The iOS app sends `viewer=appearance-1` on the page request and on both revision polls (§4a) |
| `view` | `product` \| `diagnostics`, optional | Which rendering the **sparse** page opens in: the product view (default) or the solver's segment-coloured diagnostics view. Honoured server-side, because a `loadHTMLString` client has no `location.search`. An unrecognised value opens the product view, never a 422. **`view=diagnostics` with `representation=auto` serves the sparse page**, whatever other rungs the session has — only the sparse page has the diagnostics rendering, so starting the ladder at the surface (or the dense rung) would answer "open the solver's view" with a surface or dense points. A pinned `representation` still wins |

**200** `text/html`, `Cache-Control: no-store`. Every page but the appearance
page is self-contained: no external script, stylesheet, image or fetch, so a
web view that refuses every navigation but the initial one shows it whole. The
appearance page fetches its imagery through exactly one transport, below. On the sparse,
dense and surface pages one finger orbits, two fingers pinch to zoom and drag to
pan; on a desktop, drag / wheel / shift-drag. The appearance page navigates
differently (its "Navigation" below).

**The appearance page** (2026-09-17, `tower/tower/world_builder/appearance_render.py`
+ `appearance_viewer.html`). The top rung, served whenever
`GET /worlds/{id}/appearance/{session}/manifest` would answer 200 for the chosen
session (label still matching, world not purged, manifest readable, proxy
whole) — **not only when the artifact is `current`**. During a walk every new
surface makes the appearance "built on an earlier surface" for the minute its
rebuild takes; demoting the rung for that minute would swap the page down and
back up on every solve, resetting the wearer's camera twice. The page says it is
behind instead (`WORLD-BUILDER-APPEARANCE.md` §8: currency is reported, never
enforced).

- **A shell, not a data page.** About 70 KB on the canonical world: the WebGL2
  renderer, the recorded camera path (`surface_render._camera_path`), the
  vertical (`surface_render.surface_up` measured on the proxy, seeded by the
  cameras), and the four addresses it fetches, relative to `CONFIG.base`:
  `/worlds/{w}/appearance/{s}/manifest`, `…/chunk/{digest}`, `…/proxy/{digest}`
  and `/worlds/{w}/render/revision?session_id={s}`. Every fetch goes through
  one helper that prefixes `CONFIG.base`; nothing else is requested.
- **The transport decision.** `transport=app` (default): `CONFIG.base` is
  `glasses-world://tower` and the CSP (header and `<meta>`) is
  `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src glasses-world:`.
  The web content cannot reach the network or the Tower directly; the app's
  scheme handler serves the page and proxies only this world's four routes
  (`WORLD-BUILDER-IOS.md` §10). A desktop browser opening the route this way
  fetches nothing and says so. `transport=tower` (named explicitly): `CONFIG.base`
  is `""` and the policy is `… connect-src 'self'`, which reaches only routes any
  client on the Tower's network can already call. No other value is accepted,
  and no policy ever allows `*`, `http:`, `https:`, `img-src` or `script-src`
  beyond inline.
- **What it draws** (revised 2026-09-17, fix-it viewer-polish lane). The proxy
  mesh, depth-prepassed; eight candidate keyframes chosen per frame on the CPU
  from a 32×24 probe of the proxy (a keyframe already on screen scores ×1.12, so
  the choice changes less while the camera moves), plus up to four still fading
  out of the previous choice (12 bound at most). **The choice covers first**
  (revised 2026-09-17, fix-it nav lane): a probe point counts for a keyframe only
  when it is in that keyframe's frame and **visible in its depth** (a CPU copy
  of the device-rendered source depth, min-pooled to 8 source px a texel,
  2 bytes × 45×80 a layer, 0.9 MB for 128), its score fades smoothly with the
  angle to the viewing ray out to 85° (not a hard 60° cut), and the greedy pick
  adds 0.6 for every probe point no chosen keyframe covers yet — so a surface
  that only an oblique keyframe saw is bound before a better angle on surface
  already covered. The first rule bound, beside the canonical world's door, eight
  keyframes that held 1–6% of the wall and none of the ones that walked toward
  it (97%, at 75–85°), and drew a real, imaged wall as "not captured". **A changed choice crossfades**:
  every bound source carries a presence that moves 0 ↔ 1 over 280 ms.
  Per fragment, each source gets a **confidence**, the product of soft tests —
  in its frame (weight rises over 80 source px from its image border), visible
  from it (its depth, rendered **on the device** from the proxy at full
  resolution and min-pooled 4×4 into an R16UI array; the test fades from 1.5% to
  4.5% beyond it, and where the nearest depth block is in doubt it is evaluated
  on the 2×2 neighbouring blocks and interpolated), within 60° of the viewing ray
  (fading from 46°) **or, with a low tail weight 0.08, within 85°** for a
  source that did not see the surface edge-on (judged by the proxy's **smooth
  vertex normal**, its facet's where it has none, used for that alone and never
  as light: a keyframe grazing a floor paints a stretched streak), its **alpha
  as a weight** (APPEARANCE §5.6), and its
  presence — so no test switches a source on or off between neighbouring pixels.
  Penalty = angle + 0.15·distance ratio (capped) + 0.10·(1 − quality) +
  2·(1 − confidence)³. The **k = 4** best blend with weight
  confidence × exp(−(penalty − best)/0.07) × a smooth cut below the 5th
  penalty (×0.3 for a sample clipped in its source). Then a **consensus** over the
  k + 2 best: a soft medoid of their colours, and every weight × 1/(1 + (d/0.35)²)
  for its brightness-relative distance d to it, so a colour a minority of good
  sources saw (a door standing open in a few frames) is voted down. Each sample
  is divided by its keyframe's photometric model (APPEARANCE §5.4: gain, tilt,
  falloff). **Unshaded**: no lighting term; one display mapping for every pixel,
  **the keyframes' own brightness** (exposure 1.0, γ 1.0, a hue-preserving
  roll-off above 0.8 that approaches 0.97 and never reaches white). Revised
  2026-09-17 (fix-it blotch lane, after the visual review): the first mapping
  (exposure 1.8, γ 1.15, knee 0.6) drew the render 1.19–2.04× as bright as the
  keyframe at the keyframe's own pose and field of view (median 1.66, NCC 0.85);
  now median 1.01×, NCC 0.94 on the same 10 poses.
  Measured and **not** adopted (same lane, 16 views): down-weighting and
  blurring a source by the anisotropy of its projection (no change in speckle,
  blotch, seams or the oblique smears), sharpening the weights toward the
  strongest source where sources disagree (the doubled phone stayed doubled;
  seam excess +1.4), a visibility tolerance scaled by incidence (no change in
  speckle; seam excess +0.5), and pulling the weights toward the keyframes
  nearest in the walk to the strongest one where sources disagree (the chair
  legs painted from two passes stayed two; seam excess +0.7 on 9 of the visual
  review's views).
  **The display field of view is the capture's frame plus a margin** (same lane,
  visual review item 1; widened 2026-09-17 by the fix-it framing lane): the
  viewport is no taller than `CONFIG.view_margin_v` × a keyframe's vertical
  tangent and no wider than `CONFIG.view_margin` × its horizontal one (the
  keyframes are 72° × 45° on the canonical world, from the manifest's `camera`),
  and both tangents are capped so that no canvas shape and no margin can make
  the page look like a fisheye (85° vertical, 100° horizontal). A portrait phone
  is decided by the vertical (about 77° × 40° at 1.15); a 900×700 canvas by the
  width (46° × 57° at 1.40). The first page drew 72° vertical in landscape (~85°
  horizontal) and 97° in portrait — twice what any photo framed — and that
  margin was where the black, the speckle and the oblique smears lived. Cutting
  it to the keyframe exactly took the damage out and left a viewfinder so tight
  that **61% of the reachable views were a clean, well-lit, empty wall**, so the
  margin came back with the damage measured. Landscape, over the 40 walk poses
  and 20 look-arounds, margin 1.0 / 1.15 / 1.25 / 1.4 / 1.6: drawn 98.2 / 97.5 /
  96.9 / 95.9 / 94.3% at the walk poses, speckle 0.093 / 0.124 / 0.144 / 0.168 /
  0.207%, blotch 1.25 / 1.45 / 1.55 / 1.70 / 1.88%, detail 5.2 / 5.8 / 6.4 /
  7.2 / 8.0, featureless 30 / 20 / 20 / 17.5 / 12.5%. **1.40** is the largest
  margin that breaks no more than two of the 61 shots by over 5 points of drawn;
  1.6 breaks thirteen. Portrait, `view_margin_v` 1.0 / 1.15 / 1.3 / 1.45: drawn
  95.1 / 94.4 / 93.5 / 93.3%, speckle 0.27 / 0.32 / 0.35 / 0.36%, detail 9.3 /
  10.3 / 11.1 / 11.3; portrait has no featureless views at any margin, and
  **1.15** is taken because it is what makes the portrait Overview a view of the
  room rather than a bed-corner close-up.
  **Thin cracks in the proxy are closed on the screen** (same lane): the phone
  proxy is decimated and filtered and 35% of its edges are open, so a wall seen
  at a grazing angle showed its pinholes and cracks as black speckle (on the
  canonical world the largest single cause of the speckle on the wall right of
  the door). After the proxy's own fragments, a full-screen pass reads the
  layer's depth: a pixel with no surface whose row or column has surface on
  both sides within 4 CSS px, two pixels deep, **with each side's slope
  predicting the other to 3% of the distance** (one plane, not an occluding
  edge), is placed on that plane and shaded by the same function as the proxy —
  from source pixels that see that point, or not at all. A void, an edge or
  anything wider stays empty.
  The layer's alpha is the **evidence** 1 − Π(1 − confidence′), where
  confidence′ uses a 20 px border feather and only the last 5° before 85° (the
  angle lowers a source's weight, not the evidence that it saw the point), so a
  place every source masked fades out instead of cutting. The frame is that
  layer composited over the **background, a dark smooth vignette, never a
  texture**, with a soft edge wherever nothing is drawn: where the evidence
  exceeds 0.02 is blurred over 8 CSS px (two separable passes) and alpha rises
  from 0 at that border to 1 inside it. **Proxy that no kept image covers is
  drawn exactly as "nothing"** — the same background, the same soft edge — not
  as a tint (revised 2026-09-17: on wide views the tint read as dark angular
  shards floating in the room); it still writes depth in the prepass, so it
  still hides what is behind it. Fading is only ever inward; a pixel with no
  evidence never gains colour. **A void is an unlit fog, not a cut-out** (same
  lane): the background is lifted toward 0.35 × the mostly grey (10% hue) mean of
  what is drawn nearby, read from a mip level about 24 texels across the frame's
  short side, fading to the plain background where nothing is drawn within that
  footprint; and the drawn room's alpha is multiplied by
  1 − 0.2 × (1 − smoothstep(0.3, 0.9, coarse coverage)), so a large void dims
  its surroundings over tens of pixels while a pinhole changes nothing. The fog
  has no texture and no detail and never raises a pixel's alpha.
  **Nothing is drawn until the opening pose is chosen**; the room then fades in
  from the background over 450 ms (the first page drew a pose-1 close-up while
  the images landed and cut to the opening). (A fade by distance over the mesh to its
  open edges was built and rejected: 17% of the canonical proxy's edges are open,
  and its sliver triangles with three rim vertices went fully transparent,
  shredding walls into dark triangles.)
- **Formats and memory.** ASTC 6×6 (`WEBGL_compressed_texture_astc`) into one
  `TEXTURE_2D_ARRAY` allocated once at the phone budget (128 layers, capped at
  192): 13.1 MB colour + 3.7 MB depth + 3.7 MB proxy buffers = 20.5 MB on the
  canonical world, reported in `window.__wbAppearance.gpu` / `gpuBytes`. Beside
  it, not in it, both scaling with the canvas: the drawing buffer (about 7 bytes
  a pixel) and the surface layer with its mip levels, its depth texture and the
  blurred coverage (`gpu.layers`, about 10.3 bytes a pixel: 6.5 MB at 900×700,
  13.6 MB for a 390×844 portrait at the capped device pixel ratio 2). The
  opening's and Overview's 64-px renders keep one more small set.
  Without ASTC the WebP chunks are decoded to RGBA8, **capped at 48 layers**
  (42 MB colour) and the caption says the set is reduced.
- **Navigation** (revised 2026-09-17, fix-it nav lane). This representation is
  photographic where the wearer stood and looked and dark from anywhere else:
  the walkthrough found every orbit behind or outside the room 99–100% dark, and
  orbiting was the first thing a person tried. So there is **no free orbit**. One
  camera `{position, yaw, pitch}`, levelled to `CONFIG.up`, moves freely **inside
  a capture-supported envelope** (`NAV` in the page, pure, run under node by the
  Tower's tests):
  - **Where.** A tube around the recorded walk: 1.0 scene unit across, 0.6 along
    the vertical, soft from 0.55 of it. Consecutive recorded poses more than 2.0
    apart are not joined (a jump in the record is not a corridor).
  - **Which way.** A **support field** over position and look direction, built
    on the page after the images land (and again for a new build), in slices so
    the page stays interactive: 6,000 area-weighted points on the proxy; a point
    is *seen* by a bound phone-tier keyframe when it is in its frame and visible
    in its depth (the CPU copy above); on a grid 0.5 apart inside the tube (2,033
    points on the canonical world, 1.56 MB) each of 384 cube-map directions keeps
    how well its nearest proxy point is coloured from there — 1 when a keyframe
    that saw it is within 60° of that ray, ½ at 80°, 0 past 85° (the blend's
    tail). A view's support is the mean over a 12×9 grid of its rays,
    trilinear between grid points; outside the tube it is 0. Measured on the
    canonical world over 200 reachable views: drawn ≈ 1.19 × support − 0.24
    (r = 0.95), so the thresholds are **0.92** (resistance starts) and **0.80**
    (it stops).
  - **How it feels.** A step that makes things worse inside the soft band is
    taken in small sub-steps, each scaled by (1 − b)², b the smoothstep through
    the band, so motion slows toward the edge and never passes it: no wall, no
    snap. A released camera drifts back inside (position toward 0.55 of the tube
    with a 450 ms time constant; look along the support gradient at up to
    0.5 rad/s × b). A resisted push shows *Not captured beyond here* for about a
    second. **A recorded pose is never "beyond"**: arriving at one sets its own
    support as the ceiling of both thresholds, so a partly dark recorded view is
    neither pushed nor locked.
  - **Quality, not only coverage** (revised 2026-09-17, fix-it blotch lane,
    visual review item 3: the old envelope stopped ON the ugly frame). Each
    keyframe's contribution to a field point is its angle weight × how squarely
    it saw the surface (0.2 + 0.8 · smoothstep(0.1, 0.4, cos incidence), from the
    sampled face's normal); each proxy sample is × 0.7 when one keyframe alone
    saw it and × 0.3 within 0.35 of a **large hole** (a loop of open edges at
    least 0.8 around; thin cracks are closed on the screen and do not count).
    Thresholds: resisted below **0.80**, stopped at **0.68**; a worsening step is
    scaled by (1 − smoothstep(0.35, 1, b))², so the first third of the band is
    free and a push in a good place is not sluggish. Measured over 250 random
    views in the tube at the capture's field of view, a view is good (drawn ≥ 0.9,
    blotch ≤ 2.5%, seam excess ≤ 8) for 96% at support ≥ 0.80, 75% at 0.70–0.80
    and 64–68% at 0.60–0.70. Below a recorded pose's own support the band is
    0.06 wide: from a weak recorded view the camera may not wander into anything
    weaker. The support field is honest but coarse: its correlation with the
    rendered drawn fraction is 0.71 over all views and 0.41 within the recorded
    pitch range.
  - **Content, not only quality** (added 2026-09-17, fix-it framing lane). A
    view can be 100% drawn, clean, and hold nothing — a plain wall, a plain
    ceiling, a blank door panel — and support rated it exactly as highly as the
    desk. Every proxy sample now also carries **how much there is to see** there:
    when a keyframe's depth pass runs, a second pass measures that keyframe's
    own local contrast (the standard deviation of luma over each 4×4 block of
    its pixels, over the pixels its alpha keeps, so a redaction box is not
    content), and a sample takes the best of the sources that saw it, discounted
    for a grazing source. The field keeps that per direction beside the
    coverage, and a view's content is the mean over the rays that land on
    something. Calibrated on the canonical world against the review's own
    measure (mean |Laplacian| over the drawn region of the 900×700 frame) over
    260 views: the field's content correlates 0.58 with it and a cut at 0.19
    reproduces its "clean but empty" call on 76.5%; the same Laplacian measured
    on the page's own 64-px scoring render correlates **0.876** and its cut
    (0.0574) agrees on **90.4%**, so the scores that can afford a render use
    that one and the envelope uses the field.
    Content is a **nudge, never a wall**. Its bound is
    0.55 × (1 − smoothstep(0.13, 0.28, content)), capped well below the 0.999 at
    which a step is refused, so a deliberate look at a blank wall always goes
    through — it is at most 40% slower, and only while the view is getting
    worse, so panning along a wall is not resisted at all. The edge hint (*Not
    captured beyond here*) is shown for the support edge only, never for a dull
    view. On **release**, a camera left on a featureless view eases toward
    content along the content gradient (asked over a wide 0.30 rad, since a
    blank wall is featureless for tens of degrees), at most 0.30 rad in total,
    stopping as soon as the view has something in it; any look of the person's
    own gives the budget back, and a finger down stops it entirely. Measured
    over 200 reachable views, released for two seconds: featureless 44.5% →
    29.0% (31 rescued, 0 lost), mean turn 4.7°, largest 18.2°, 80 of 200 did not
    move, drawn 93.5% → 94.8%.
  - **Pitch** stays within what the walk looked at (the recorded range plus
    0.2 rad each way, resisted over its last 0.2 rad): the review's torn ceiling
    was one vertical drag from the opening.
  - **Distance**: the camera keeps `CONFIG.standoff` scene units from the
    nearest proxy sample (resisted from 0.30 beyond it): nearer, a 360×640
    keyframe is magnified past its resolution and the frame fills with one
    blurred patch that the field scores as fully supported. Raised from 0.55 to
    **1.0** by the fix-it framing lane, which also found that the standoff is
    close to inert on the canonical world — the tube and the support threshold
    already hold the camera 2.2 units off the proxy on average, and a hard push
    forward from 17 recorded poses reached 0.86 at a standoff of 0.55 and 1.19
    at 1.15 — while 1.5 starts to fight the recorded walk, which itself passes
    within 0.3 of the proxy in places. The **reachable sampler** applies the
    standoff too (it did not, so the measured distribution used to contain views
    the camera could not reach), and so does the drift back inside: a push
    stopped by the standoff must not drift to a spot within it.
  - **Looking across a gap**: a look held against the edge that keeps pushing
    (0.3 rad of resisted input) glides to the first direction within half a
    turn whose support is at least 0.80, instead of parking on the edge frame.
  - **Before the field is built the camera does not move** (review item 5:
    early input escaped the envelope and the page then took the escaped
    camera's support as its floor). When the field completes, the floor is the
    support of the recorded pose the camera is at, never of wherever it is.
  - **The envelope only limits the camera; it paints nothing.** Dark places
    inside it stay dark.
  - **Controls.** Touch: one finger drags the look (the room follows the
    finger), two fingers drag to move sideways and up/down, pinch moves forward
    and back (log of the spread × 2 units). Desktop (debug): left drag looks,
    right or shift drag moves, the wheel moves forward/back, W A S D Q E fly,
    ←/→ step the walk, O is Overview. Motion coasts after a flick (220 ms time
    constant). Buttons: **Overview**, **←**, **→** (the recorded walk), **Reset**
    (the opening).
  - **Stepping the walk glides**: eased position and the short way round in
    yaw, 350–1100 ms by distance and turn, instead of jumping (the walkthrough's
    worst flicker steps were path jumps). Any touch interrupts a glide.
  - **Overview** replaces Orbit: a **raised** vantage in the envelope that
    looks at the room. The field proposes (points at least 0.15 above the walk,
    within 0.85 of the tube and at least 1.2 from the opening pose, 12 yaws × 2
    downward pitches, support ≥ 0.68 and mean supported distance ≥ 0.5 × the
    scene's median depth), scored support × (0.5 + 0.5 × distance) × (0.5 + 0.5
    × facing what the walk looked at, the mean of the recorded look targets);
    the real blend decides between the top twelve (spread by position and yaw)
    by drawn × (0.4 + 0.6 × rendered distance) × facing × (0.15 + 0.85 × rendered
    **detail**) × (0.5 + 0.5 × its **depth range**, the 10–90 spread of the
    distance to what it draws, over the scene's median depth); the camera glides
    there. Revised 2026-09-17 (fix-it blotch lane): at the capture's field of
    view the old score chose a frame-filling close-up of the door edge (100%
    drawn, nothing to see), and in landscape it landed about 0.6 from the
    opening. Revised again the same day (fix-it framing lane): the candidates
    are now filtered by the field's **content** (≥ 0.19, relaxed in steps until
    something passes, so the button is never dead) and by a raised depth floor
    (0.8 × the median), colour spread is replaced by the rendered detail, and the
    depth range breaks ties. In portrait this makes the Overview the room in one
    frame. **In landscape it does not**: on the canonical world all twelve
    candidates are the same desk area, because the recorded walk never stood
    back from it, so no vantage inside the envelope frames the whole room.
  - **Prev/next skip poses that render badly, and poses with nothing in them**:
    once the field is built every recorded pose is rendered in the background at
    the opening's 64-px size, and ←/→ step to the next pose whose drawn fraction
    is at least 0.8 **and whose rendered detail is at least `EMPTY_DETAIL`** (an
    unscored pose counts as good; with none ahead the camera stays and the hint
    shows). **Skipping is capped**: at most 4 poses are passed over in one press,
    and a longer bad run lands on the best pose in it rather than being jumped
    whole, so the walk that is shown is shorter than the one recorded but never
    misses a stretch of it. The caption says so. On the canonical world 18 of
    198 poses are skipped (17 for drawn, 7 for detail).
  - Before the field is built (a second or so after the images land) the camera
    does not move and Overview is disabled.

  It opens at the recorded pose whose **rendered frame is most drawn and has the
  most in it** (drawn × (0.85 + 0.15 × wide) × (0.35 + 0.65 × rendered detail),
  the detail term added by the fix-it framing lane), with a mild
  preference for wider content: every ⌈n/32⌉-th recorded pose is rendered at 64
  px on the long side **in the canvas's own aspect** through the real blend,
  scored `drawn × (0.85 + 0.15 × min(1, mean distance / (2.5 × z_ref)))` (drawn =
  mean evidence alpha), then refined around the best at a third of the step
  (35 renders on the canonical world). Revised 2026-09-17: the first rule,
  observed pixels × distance², preferred far half-empty views and opened the
  canonical world at pose 71 (46% observed, 52% black). The horizon is levelled
  to `CONFIG.up`. The caption is one line (*Captured images on reconstructed
  geometry* and an **About** button); what the images are, what the dark
  areas mean, and that the arrows pass over poses that render badly or show
  nothing (at most 4 in a row, so the walk shown is shorter than the one
  recorded but never misses a stretch of it) open on tap (the four-line
  disclaimer covered 11–13% of the frame).
- **Live.** The page polls the revision route every 10 s (backing off to 120 s
  while `live` is false and nothing changed). What a poll means is one pure unit
  in the page, `FOLLOW` (`decide`, `mustReplace`, `nextDelay`), run under node by
  the Tower's tests (review 1, B1). A new `appearance.revision` fetches the
  manifest and only the chunks whose digest it lacks, overwrites layers in place
  (a keyframe that left the phone tier frees its layer), renders source depth
  for moved or new layers, and keeps the camera: no navigation, no reload —
  **unless the new manifest's `epoch` differs from the one on screen**, when every
  texture is dropped first. `appearance.state: rebuilding` (the ordinary Stop)
  keeps the textures, keeps polling at 10 s and captions *finishing the world*;
  after 20 minutes without a served build the page drops them and says so.
  `withdrawn`, `absent`, an old Tower's bare `null`, or a revision 404 naming a
  world or session that is gone (m3) deletes every texture at once and says why —
  **and keeps polling**, so a rebuild that is served again is drawn again without
  closing the viewer. A chunk or proxy that 404s while loading (the next build
  published and the file left its grace) refetches the manifest and tries once
  more (M4).
- **Context loss.** Textures and the manifest are not kept across it: on
  restore the page fetches the manifest again, then the proxy and chunks,
  through the same transport (so the Tower's label check runs again, and a build
  or withdrawal that happened meanwhile is honoured) and keeps the camera. If
  WebKit has not restored the context after 3 s the page asks for it
  (`WEBGL_lose_context.restoreContext`); after 10 s it reloads itself, which the
  app allows for exactly this page (`WORLD-BUILDER-IOS.md` §10; review 1 m8).
- **Caption.** *Captured images on reconstructed geometry · N of M keyframes
  shown* (· *still building as you walk*, · the currency reason when behind),
  then: *These are the camera's own frames, with faces redacted, placed on the
  reconstructed room. Dark areas are places no kept frame saw, or that were
  masked as unreliable (redaction, hands, views that disagreed); nothing there
  is filled in. Where the geometry underneath is wrong, images smear or double.
  Scale is unknown, so distances are relative.*
- `window.__wbAppearance` also exposes `walk`, `setView`, `coverage`,
  `snapshot`, `camera`, `shotMode`, `clock`, and for navigation `pose`,
  `setPose`, `support`, `navStats`, `navReady`, `sampleReachable`, `input`,
  `step`, `overview`, `frame`, `poseQuality`, `poseContent`, `navConst` and
  `detailOf`, for verification; they read the page and move
  its camera, nothing else. `orbitView` remains only to measure the views the
  removed Orbit mode reached.

**The route serves one of two pages and they differ, deliberately.** This
section described only the sparse one until the dense viewer shipped; what
follows is both.

*The sparse page.* A `<select>` switches between the drawable spaces: the
shared world frame(s) first, then every unregistered segment in its own frame
and own scale, named as such. Its caption is *sparse structure-from-motion
output: triangulated feature points and camera poses; not a surface, not a
mesh, not metric scale.*

*The dense page.* One cloud in the world frame, so there is no space selector:
a dense artifact is built for a single solve component and the format refuses
to composite two, because nothing normalises them and they share no unit. Its
caption is *dense reconstruction: per-pixel depth from a neural network,
anchored to the structure-from-motion solve and kept only where several
cameras agreed; not a surface, not a mesh, not metric scale* — a different
sentence because the sparse one would be false of it. It adds a line when the
page is coarser than the artifact on disk, and it names face redaction as a
cause of emptiness.

Both carry a BEHIND line when what they draw is not current — the dense page
distinguishes *behind the newest keyframes* from *built against an earlier
solve*; see `WORLD-BUILDER-DENSE.md` §10.

**404** with a `detail` the phone shows verbatim, for exactly these:
no world root configured; no such world; no such session in that world;
the named session has no geometry yet; no session of the world has
geometry yet. The last two are what a world still being built or solved
answers: the client's response is to say so and offer to try again, not to
treat it as an error in the world.

**422** for a `max_points` outside its range, or a `representation` outside its
set.

### Rules

1. What is drawn follows `WORLD-BUILDER-GEOMETRY.md` §5 rule 3 and §7 exactly: a segment is placed in the world frame only when its placement is `registered`, complete, and bound to the current build. Everything else is drawn apart, labelled, never overlaid.
2. The page never claims more than the caption says. A viewer that shows it must not either.
3. Composed on request, cached nowhere: a world under construction changes with every build, and the client bypasses its own cache for the same reason the geometry routes ask it to.
4. `world_id` passes the same containment guard as the geometry routes (`contained_world_id`): an id that resolves outside the world root is "no world", and a non-canonical spelling is answered as the world it names.
5. **One derived manifest per world, not per session.** `derived/manifest.json` records the digest of the *last* build, so in a world with two built sessions the older one's placements no longer bind to it: that session renders with every segment apart, labelled `unbound`, and the BEHIND caption — which is the truthful reading of a tree the current build did not produce, not a defect in the session. `GET /worlds` still answers `has_geometry: true` for it. A per-session manifest is the fix and belongs to the store, not to this route.
6. The response carries `Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'` — the promise above, enforced by the browser as well as kept by the producer — **and every page carries the same policy as a `<meta http-equiv="Content-Security-Policy">` right after `<meta charset>`**, which is the only form a `loadHTMLString` client enforces: iOS drops the response headers, and its navigation policy sees navigations, not an `<img>` or a `fetch`. All three pages (sparse, dense, surface) carry it, and it never pushes `wb-representation` out of the first 4096 characters. **The appearance page is the one exception to the policy's content, not to the rule:** its header and its `<meta>` both read `… connect-src glasses-world:` (or `connect-src 'self'` under `transport=tower`), and the two are always equal.

## 4a. `GET /worlds/{world_id}/render/revision` — has a better picture been built?

Additive. A client may keep a picture from §4 open while the Tower is still
building the session — during a walk the surface is rebuilt each time a global
solve lands — and ask this, a few hundred bytes, instead of re-downloading the
page to find out.

| param | type | meaning |
|---|---|---|
| `session_id` | string, optional | as §4; resolved the same way |
| `view` | string, optional | as §4. `view=diagnostics` reports the **sparse** rung, because that is the page §4 serves for it. A client showing the diagnostics rendering has nothing to follow and should not ask (the iOS app does not) |
| `viewer` | string, optional | as §4, and it must match the page request's: without `appearance-1` the rung reported is the one §4 would serve a client that cannot draw the appearance page (`surface` or lower). A poll that dropped it while an appearance page is on screen would be told `surface` and swap the page down. The iOS app sends it on its native poll, and its scheme handler adds it to the page's own proxied poll (`WORLD-BUILDER-IOS.md` §10). `appearance` in the body is reported either way |

**200** `{"session_id": str, "representation": "appearance"|"surface"|"dense"|"sparse", "revision": str, "live": bool, "appearance": {"revision": str|null, "current": bool, "state": "served"|"rebuilding"|"withdrawn"|"absent", "epoch": str|null}}`,
`Cache-Control: no-store`. **404** exactly when §4 would 404.

`appearance` (additive, 2026-09-17) follows the session's appearance artifact
(`WORLD-BUILDER-APPEARANCE.md`): `revision` is
`<session_id>/appearance:<build_id>` exactly when
`GET /worlds/{id}/appearance/{session_id}/manifest` would answer 200, and `null`
otherwise — no artifact, a purged world, or **a session whose redaction label no
longer matches the one the artifact was built under**. `state` says which, and
what a page holding textures does (`WORLD-BUILDER-APPEARANCE.md` §9): keep them
through `rebuilding` (the ordinary Stop), drop them on `withdrawn` or `absent`,
and in every case keep asking. `epoch` changes exactly when an open page must
drop its textures before drawing the served build. `current` is false when the artifact
was built from an earlier solve or on an earlier surface. It is opaque, compared
for equality, and **deliberately not part of `revision`**: a page that blends
keyframes follows appearance builds itself, and folding them into the page
revision would swap the page, and reset the wearer's camera, on every one.

`live` (additive, 2026-09-16) is `true` while something is building **this
session**: the world's writer lock is held by a live builder and this session is
the one it is writing (record still open, or finalization still `pending`), or
this session's surface, appearance (added 2026-09-17) or dense stage reports
`running` from a live process **and has not yet published**: a stage whose `manifest.json` is newer than its
`running` `status.json` has written its result and is not live, although `ok`
lands a moment later. Otherwise a poll in that gap reported the finished build as
live, and a client that offers live builds rather than swapping them in missed
the finished one. See rule 6 for what `false` does and does not promise.

Every page §4 serves carries the same two values in its head, within its first
4096 characters:

    <meta name="wb-representation" content="surface">
    <meta name="wb-revision" content="<session_id>/surface:1789551234.56">

### Rules

1. `revision` is **opaque**. Compare for equality only. A client that needs the
   rung reads `representation`, never a prefix of `revision`.
2. It changes when the page §4 would serve changes rung, or when the surface or
   dense artifact behind the served rung is rebuilt, or when the session §4
   would choose changes: every revision is prefixed with the chosen session id.
3. The appearance rung's revision is `<session_id>/appearance:1@<epoch>`: the
   page PROGRAM's version (`appearance_render.PAGE_REVISION`) and the served
   manifest's `epoch` (APPEARANCE §9), not the appearance build: the page follows
   builds itself (`appearance.revision`), so an ordinary build, and the Stop
   transition, never change `revision` and never make a client reload the page.
   Stepping onto the rung, off it (a relabelled or purged session steps down to
   the surface), a rebuild that starts a new epoch (a relabel, a re-redaction
   switch or revert, a purge and rebuild), or a Tower update that bumps the page
   version does change it. Until 2026-09-17 it was the constant
   `<session_id>/appearance:1`, so a page that had dropped its textures was
   stamped exactly like its rebuild and was never replaced (review 1, B1). A
   manifest built before epochs existed answers `appearance:1`.
   The sparse rung's revision is the constant `<session_id>/sparse`. The derived
   tree is rewritten every few keyframes during a walk, and a picture that
   reloaded on each of those would be unusable to look at; the step the wearer
   is waiting for — up the ladder — still changes it.
4. The endpoint decides the rung by the same artifact checks as
   `has_geometry` in §2 rather than by composing the page. The two can disagree
   only when an artifact passes its header check and then fails to parse (the
   Tower logs that loudly at `ERROR`); a client that remembers the revision it
   acted on, as well as the one stamped into the page it got, pays one extra
   page fetch for that, not a loop.
5. **A 404 whose `detail` is FastAPI's `Not Found`** — no such route, which is
   what a Tower older than this route answers — means **nothing to follow**:
   stop asking and keep the picture on screen. **A 404 with one of §4's own
   sentences may be transient** ("no geometry yet" is what a world mid-build
   can answer while its derived tree is being replaced): keep the picture and
   ask again at the next interval.
6. **`live: false` means "slow down", not "stop".** The builder releases the
   world lock at the end of finalization and only then starts the final surface
   and dense stages, so there is a short window — up to the length of a
   registration — in which nothing is marked running and a better picture is
   still coming. A client backs off while `live` is `false` (the iOS app: 10 s
   doubling to a 120 s ceiling, back to 10 s whenever the revision changes or
   `live` is `true`) and keeps polling at its base interval while it is `true`.
   A finished world therefore costs one request every two minutes, not every
   ten seconds.
7. **A surface whose manifest cannot be read at that moment is answered as the
   next rung, never as a surface with no revision.** A read that races the
   replace of a landing build is retried; if it still fails, the rung falls to
   dense or sparse for that one request, and a page composed in that moment
   carries no `wb-revision` rather than a false one. A client that does not
   step down the ladder by itself (the iOS app neither swaps nor offers a worse rung,
   `WORLD-BUILDER-IOS.md` §10) sees nothing change.

## 4b. `GET /worlds/{world_id}/appearance/{session_id}/…` — the keyframes a phone blends

Added 2026-09-17. **The first route family that serves imagery**, deliberately:
the wearer's redacted keyframes, masked and bundled for view-dependent blending
over the surface. The whole contract is `WORLD-BUILDER-APPEARANCE.md` §9; in
short:

| route | body |
|---|---|
| `…/manifest` | the appearance manifest, plus `currency` |
| `…/chunk/{digest}` | a keyframe bundle (ASTC 6×6 for the phone, WebP everywhere) |
| `…/proxy/{digest}` | the proxy mesh the keyframes were prepared against (`WBSURF01`) |

- `Cache-Control: no-store`, `Pragma: no-cache`, `X-Content-Type-Options: nosniff`
  on every response including 404s; no `ETag` or `Last-Modified`; a 200 carries
  `X-World-Redaction` with the label that was actually applied.
- **The redaction label is re-checked on every request**: a session whose
  keyframe set's label or identity differs from the one the artifact was built
  under (including a re-redaction switch or its revert,
  `WORLD-BUILDER-APPEARANCE.md` §6.5) answers 404
  ("appearance is stale against the session's redaction record"), as does a
  world with `images_purged`.
- `digest` is 32 lower-hex and must be named by the current manifest, or by one
  it superseded less than 120 s ago under the same label and keyframe set
  (APPEARANCE §9, review 1 M4). URLs carry no path, file name or capture
  sequence number.
- `world_id` passes `contained_world_id`; `session_id` must be one of the
  world's sessions.
- The only page that reaches these routes is §4's appearance page, through the
  transport §4 names (`glasses-world:` on the phone, the Tower's origin only
  under `transport=tower`). Every other page still loads nothing from anywhere.
