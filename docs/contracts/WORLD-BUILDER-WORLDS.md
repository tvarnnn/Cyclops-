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
| `photographic` | object | **Where this session's photographic representation — the image-based room, which is the final user-facing output — has got to** (additive, 2026-09-22): `{state, stage: "surface"\|"appearance"\|null, detail}`. `state` is one word from a closed vocabulary: `complete`, `running`, `owed`, `failed`, `unattempted`, `never_recorded`, `unobservable`. The same block the status payload carries (`CARTRIDGE-RESULTS.md`, `lifecycle.photographic`), computed from the same helper, so a phone that reads the row and then opens the panel reads one fact and not two. §2a says what each word means, which of them move `state` and which do not, and why `failed` does not. **Always present on this listing**, unlike the status channel's `lifecycle.photographic`, which is null when the lifecycle was computed from the record alone: a row is built for every session, and a row with no answer is a row that keeps saying `complete`, which is the failure being fixed. A probe that cannot answer yields the `unobservable` WORD, never a missing block |
| `components` | array \| null | **PROPOSED 2026-09-23 — awaiting Mac review; nothing implemented.** The pieces of this walk's final solve after the evidence gate: exactly one `placed` (the room) first, then every piece the gate could not place, with `reason`, keyframes, capture spans, `shown_as` (`room` / `area` / `none`) and its own §2a block. No metric figure, no name. **`null` means not computed** — every world today — never "no areas". Additive; the contract identifier does not move. Specified in [`WORLD-BUILDER-COMPONENTS.md`](WORLD-BUILDER-COMPONENTS.md) §2–§3 |

## 2a. `photographic` — whether the room the wearer walked actually exists

Added 2026-09-22. `state` (above) is a word about the **session**: what
happened to the capture and the solve. It cannot hold what happened
**after** them, and the picker used to decide that from a present-tense
probe — *is a photographic stage running this millisecond*. The Mac/iOS
validation of 2026-09-22 caught that being the wrong question (finding T3):
a stage that **failed** is not running, a stage that is **owed** is not
running, and a probe that **broke** reported not running, so all three read
`complete` in the picker. The one thing that knew better, `session.stages`,
was read by `scripts/world_finish_pending.py` — the tool that builds the
missing work — and by nothing on the wire. The picker is the surface a
person chooses a walk from and then shuts the Tower down.

This block answers the settled question instead: *does this world still owe
a photographic room*. It is computed from the session's own stage record
first and the stages' own artifacts second — the same two signals, in the
same order, that `world_finish_pending.assess()` uses — so the tool that
BUILDS the missing work and the channel that REPORTS it cannot disagree
about which worlds are missing it, and so the answer is true across the gaps
between stages rather than false in them.

| Field | Type | Meaning |
|---|---|---|
| `state` | one of the seven words below | never null when the block is present |
| `stage` | `"surface"`, `"appearance"`, or null | the stage the word is about |
| `detail` | string | prose; safe to show, names no path |
| `scope` | `"room"` \| `"area"`, optional | **PROPOSED 2026-09-23 (C1 E3), additive:** `"area"` exactly when the word is an area's (`WORLD-BUILDER-COMPONENTS.md` §3.4); absent means `"room"`. The phone chooses its Improving copy from it and never parses `detail` |

| `state` | Meaning | the session's `state` |
|---|---|---|
| `complete` | the appearance stage finished; the photographic room exists | unchanged (`complete` on an ordinary walk) |
| `running` | a stage is running under a live process | `finalizing` |
| `owed` | unfinished, nothing working on it; the Tower finishes owed work the next time it is idle (no stream open, no capture worker alive), and at every start | `finalizing` |
| `failed` | a stage ran and failed. **Terminal** — nothing more is coming | unchanged (`complete`): the world is saved, at whatever rung it reached |
| `unattempted` | the stages declined (no global solve, or the Tower's appearance setting is off) | unchanged |
| `never_recorded` | a world from before the photographic stages existed | unchanged |
| `unobservable` | the probe could not answer | `finalizing` |

**Why three of these change the row's word and four do not.** The three that
do (`running`, `owed`, `unobservable`) all mean "this world is not finished
being made", and a world in any of them must never be reported finished —
that was the false **Saved** the 2026-09-22 validation found at the stage
boundaries (T2), over a failed build (T3), and after a broken probe (T4).
The four that do not are settled: `complete` is finished, `failed` is
finished badly, and `unattempted` / `never_recorded` are worlds that were
never going to have one. Reporting a settled world as `finalizing` would be
the opposite failure — a wearer left waiting forever for a build that is not
coming.

**Why `failed` keeps `complete`, which is the one judgement call here.** The
world **is** saved and opens at whatever rung it reached: `finalization.state`
is `complete`, `final_solve` is `solved`, the derived tree is on disk and the
render route (§4) serves it. What failed is the photographic room on top of
it. `finalizing` would tell the wearer to wait, and waiting does not fix a
stage that ran and raised — `world_finish_pending.py` picks up the
INTERRUPTED stages the next time the Tower is idle, not the failed ones.
`interrupted` was the other tempting answer and it is worse: it is defined
above as a claim about the CAPTURE and the SOLVE, both of which succeeded
here, so it would tell the wearer their walk was lost. The word never claims
photographic success, because the word was never about the photographic
room. This block is, and it says `failed`, with `detail` carrying the reason.

**`never_recorded` is load-bearing for compatibility.** Worlds built by a
Tower with no photographic stages at all (corrected 2026-09-23, C1 E10: the
Tower's own code measured 67 `unattempted` and 2 `never_recorded` on the
development root, `tower/tower/world_builder/photographic.py`, not "165 of
166"). They are finished, they are owed nothing, and their rows are
unchanged. It is reached only when there is NO stage record AND no stage
artifact on disk, and it is decided **without** consulting the liveness
probe — so a probe that breaks cannot relabel all of them at once. The
distinction is sound in both directions because `world_build_session.py`
records the surface stage BEFORE it releases the world lock: absent means
old, not "new and early".

**The row and the panel are one answer to one question.** Both are computed
from `tower/tower/world_builder/photographic.py`, and
`test_the_row_and_the_panel_agree_about_every_photographic_state` pins that
for every word in the vocabulary. A row saying `interrupted` over a panel
saying `ready` is exactly the picker/panel disagreement that test exists to
stop, and it is the second reason `failed` falls through to the settled arms.

**Additive, and the contract identifier deliberately does not move.** iOS
reads these rows key by key out of a `[String: Any]`, so a key it does not
know is a key it never looks at — but it equality-tests `contract`, and a
bump would empty the gallery (see `state` above). An app that ignores this
block still behaves correctly; it simply hears `finalizing` where it used to
hear a false `complete`. What the block adds is the ability to be specific,
and the one case worth new copy is `failed`: *Saved — the photographic
version could not be built*. `WORLD-BUILDER-IOS.md` §3a is the phone's side
of the same block.

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

- **A shell, not a data page.** About 180 KB on the canonical world (175,436 B
  at `f6d5520`, and it has grown with every lane; "about 70 KB" was the first
  build's figure and stood uncorrected until review 2). It carries no imagery
  and no geometry: the WebGL2
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
  what is INKED nearby, read from a mip level about 24 texels across the frame's
  short side, fading to the plain background where nothing is drawn within that
  footprint; and the drawn room's alpha is multiplied by
  1 − 0.2 × (1 − smoothstep(0.02, 0.35, that footprint's mean alpha)), so a
  large void dims its surroundings over tens of pixels while a pinhole changes
  nothing. The fog has no texture and no detail and never raises a pixel's
  alpha. **It is always darker than the room beside it**, which is what makes a
  void read as "nothing was seen here" rather than as something: the layer is
  written PREMULTIPLIED, so the mip level is the mean of what is actually on the
  screen, and 0.35 of that cannot reach it. Corrected 2026-09-17 (review 2,
  P-2): the mean was taken over an unpremultiplied layer and divided by the mean
  ALPHA, which is evidence and not coverage — at colour 0.97 with evidence 0.5,
  routine where few sources saw a wall, that is 1.94, and the fog came out at
  about 0.68 grey, brighter than the wall it was a hole beside. The same
  confusion dimmed a fully covered but 40%-confident region as though it were
  the edge of a hole (P-2b), which is why the wide fade's knee is on 0.02–0.35
  now.
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
  The opening's 64-px renders and the Best view's 128-px ones keep one more
  small set.
  Without ASTC the WebP chunks are decoded to RGBA8, **capped at 48 layers**
  (42 MB colour) and the caption says the set is reduced.
- **Navigation** (revised 2026-09-17, fix-it nav lane; **revised again
  2026-09-18, fix-it interaction lane — looking is now free**). This
  representation is photographic where the wearer stood and looked and dark from
  anywhere else: the walkthrough found every orbit behind or outside the room
  99–100% dark, and orbiting was the first thing a person tried. So there is
  **no free orbit**. One camera `{position, yaw, pitch}`, levelled to
  `CONFIG.up` (`NAV` in the page, pure, run under node by the Tower's tests).

  **The two halves are not the same thing, and the page must not treat them as
  one.** *Which way it faces* is free: a full turn of yaw from any position it
  can be in, and pitch limited only by the neck. Turning to a wall the glasses
  never photographed shows honest darkness, which is true and expected; a
  control that answers a deliberate two-inch drag with four degrees is not a
  boundary, it is a broken page. *Where it may be* is limited, by the two
  limits that are physically true: the tube around the walk that was actually
  recorded, and how close a 360×640 keyframe may be magnified.

  This replaces the rule of 2026-09-17, under which a look was bounded by the
  support field exactly as a move was. An independent review measured what that
  cost, through the page's own input path: **24.1° of reachable yaw at the
  opening pose** (a left drag moved −3.8° and then nothing for the next 57
  frames of held drag), a median leftward look of **−7.3°** and a median upward
  look of **+5.7°** across 16 recorded poses, nine of sixteen under +7° upward,
  fifteen of sixteen under 19° one way while giving 67–100° the other, and a
  deliberate 4.48-unit forward push travelling a median **0.94 units**. The
  designed escape from the edge (`lookAcross`) fired **zero times** in ten
  60-frame sweeps. Roughly 100 of 147 frames of a portrait interaction sequence
  were static under continuous input.
  - **Where.** A tube around the recorded walk: **2.2** scene units across,
    **1.4** along the vertical, soft from **half** of it (it was 1.0 × 0.6,
    soft from 0.55, which left the first half of every sideways push resisted;
    then 1.5 × 0.8, soft from 0.75).
    Consecutive recorded poses more than 2.0 apart are not joined (a jump in the
    record is not a corridor).
    **Widened again 2026-09-21** (fix-it ux lane, `fixit\ux\UX.md` §4). An
    independent review measured what a finger reaches: forward 100% and right
    100% of what it asks — those run ALONG the walk, where the tube does not
    bind — against **backward 31%** and **up and down 16% each**, with the
    eight up frames and the eight down frames indistinguishable from one
    another. The limit had never been measured against the imagery, only
    asserted. It has been now: the opening view rendered at 0.25-unit steps
    along the world vertical out to ±3 and at 0.5-unit steps straight back out
    to 5 (through `S.setPose`, which the envelope does not bound) stays
    **97–99% drawn and 0.3–1.7% black at every one of them**, with the same
    detail as the opening. The tube was three times tighter than the pictures
    require. At 2.2 × 1.4 a finger reaches **back 2.10, up 1.25, down 1.25** of
    a 4.48-unit push (from 1.41 / 0.71 / 0.70), and each of those directions
    still stops, visibly, with *Not captured beyond here*. The price is the
    support field's volume, and it is paid in **lattice spacing** (below), not
    in waiting.
  - **Which way.** A **support field** over position and look direction, built
    on the page after the images land (and again for a new build), in slices so
    the page stays interactive: 6,000 area-weighted points on the proxy; a point
    is *seen* by a bound phone-tier keyframe when it is in its frame and visible
    in its depth (the CPU copy above); on a grid **0.7** apart inside the tube
    (0.5 until 2026-09-21, when the tube grew: at 0.5 the new tube is 2.6× the
    voxels, at 0.6 it is 5,914 and 1,272 ms of field work, at 0.7 it is 4,259
    and 974 ms — the 3,851 / 940 ms the 0.5 lattice cost over the old tube. The
    fine pass is what *Best view* waits for, so the wider envelope is paid for
    in resolution rather than in waiting; the page already ships a coarser
    field still and uses it for the first second) (2,033
    points on the canonical world, 1.56 MB) each of 384 cube-map directions keeps
    how well its nearest proxy point is coloured from there — 1 when a keyframe
    that saw it is within 60° of that ray, ½ at 80°, 0 past 85° (the blend's
    tail). A view's support is the mean over a 12×9 grid of its rays,
    trilinear between grid points; outside the tube it is 0. Measured on the
    canonical world over 200 reachable views: drawn ≈ 1.19 × support − 0.24
    (r = 0.95), so the thresholds are **0.92** (resistance starts) and **0.80**
    (it stops).
  - **How it feels.** A MOVE that makes things worse inside the soft band is
    taken in small sub-steps, each scaled by (1 − smoothstep(0.55, 1, b))², b
    the worse of the tube's bound and the standoff's, so motion slows toward the
    edge and never passes it: no wall, no snap, and no resistance at all over
    the first part of the band. A released camera drifts back inside (position
    toward 0.75 of the tube with a 450 ms time constant). A resisted push shows
    *Not captured beyond here* for about a second. **A look is never resisted
    by the capture** — not by support, not by content, not by a recorded pose's
    own quality. Yaw is therefore never scaled at all and never raises
    `resisted`. The single exception is the *neck*: the last 0.2 rad before
    ±80° of pitch is eased, which does raise `resisted` (and so spends a
    flick's momentum) but leaves `hard` at 0, so the edge hint is never shown
    for a look. **A recorded pose is never "beyond"**: arriving at one records its own
    support as the floor (`S.pose().floor`). The floor exists to stop a *limit*
    pushing a weak recorded view away, so with no limit left on a look it now
    constrains nothing, and in particular it does **not** enter what the page
    *says*: `dark` is measured against a fixed band (DARK_LO/DARK_HI, below),
    so the dimmest
    recorded pose in a world reports its view exactly as the brightest one does.
    Reading it through the floor would have made the darkest place on a world
    the one place the page went quiet about the dark.
  - **Quality, not only coverage** (revised 2026-09-17, fix-it blotch lane,
    visual review item 3: the old envelope stopped ON the ugly frame). Each
    keyframe's contribution to a field point is its angle weight × how squarely
    it saw the surface (0.2 + 0.8 · smoothstep(0.1, 0.4, cos incidence), from the
    sampled face's normal); each proxy sample is × 0.7 when one keyframe alone
    saw it and × 0.3 within 0.35 of a **large hole** (a loop of open edges at
    least 0.8 around; thin cracks are closed on the screen and do not count).
    Measured over 250 random views in the tube at the capture's field of view, a
    view is good (drawn ≥ 0.9, blotch ≤ 2.5%, seam excess ≤ 8) for 96% at
    support ≥ 0.80, 75% at 0.70–0.80 and 64–68% at 0.60–0.70. The support field
    is honest but coarse: its correlation with the rendered drawn fraction is
    0.71 over all views and 0.41 within the recorded pitch range.
    **Since 2026-09-18 support resists nothing.** T_HI (0.80) and T_LO (0.68)
    are what the page *says*, not what it enforces, and the turn happens in
    full either way. Support still filters the Best view's candidates and is
    still reported by `S.support`.
    **The darkness the page reports is a different band** (2026-09-20, fix-it
    orient lane). *Nothing was photographed this way* is a claim about whether
    a view is on **anything**, and it was being read off T_LO, which is the
    band at which three views in four render *well*. On the canonical world
    that made `dark` exactly 1 at eight of nine sampled recorded poses —
    including poses the page draws 99.4% of — so thirty frames of drag at the
    opening, over 97% drawn, raised the sentence five times. A page that says
    *nothing was photographed this way* over a photograph of the room is
    crying wolf, and the sentence has stopped meaning anything by the time it
    is true. `dark` is now `1 − smoothstep(DARK_LO, DARK_HI, support)` with
    **DARK_LO = 0.07, DARK_HI = 0.25**, swept against the page's own
    `coverage()` over 200 reachable views (`fixit\orient\ORIENT.md` §2.3).
    Those 200 separate cleanly — every view that is at least 95% background
    has support ≤ 0.088, every view that draws at least half the frame has
    support ≥ 0.183 — and the band puts the hint's own 0.9 crossing in the
    middle of that empty gap: over the 200 the sentence is now said over
    **none** of the 116 views that draw half the frame (the old band said it
    over **97** of them) and over **all 24** that are essentially black.
    Both bands are reported by `S.navConst()`.
    **It is a STATE, not a toast** (2026-09-21, fix-it ux lane, `UX.md` §3).
    It used to be `hint()`: 1,100 ms of life, throttled to one showing per
    4,000 ms. Measured over a 24-step turn from the opening it was visible at
    **3 of 24 steps and at NONE of the six that render a 100% black frame** —
    the deepest black was the least explained, which is the one thing the
    sentence exists for. A page cannot explain a condition that persists with
    an event that does not. `#dark` is now held for as long as the view is on
    nothing: it appears when `dark > 0.90` has held for **320 ms** and goes
    when `dark < 0.45`, so sweeping through a dark patch on the way somewhere
    does not flash it and stopping in one always explains it. It reads
    *Nothing was photographed this way / the glasses never looked this way —
    tap to turn back*, and tapping it does what *Face the room* does. Measured
    again over the same 24 steps: visible at **10 of 24 and at 6 of 6** of the
    100%-background steps, and at none of the 14 that show the room.
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
    **Content resists nothing either** (2026-09-18). The framing lane made it a
    nudge rather than a wall — at worst 40% slower; the interaction lane took
    the nudge out, because from the other end of a finger a nudge and a wall are
    the same gesture failing, and the honest place to say *there is nothing
    here* is the picture.
    What content still does is **the settle**, and it is the only thing that may
    move the camera on its own. A camera **the page placed** (the opening, a
    walk step, the Best view) that lands on a featureless view eases along the
    content gradient (asked over a wide 0.30 rad, since a blank wall is
    featureless for tens of degrees), at most 0.30 rad in total, stopping as
    soon as the view has something in it. **A look of the person's own is never
    settled**: any look input marks the camera `aimed`, which only the page
    placing the camera clears, so a deliberate look at a blank wall stays on the
    blank wall for as long as it is left there. A finger down stops the settle
    entirely. (Before 2026-09-18 the settle applied to a released look as well,
    and a released look was also eased back along the *support* gradient; both
    are gone.)
  - **Pitch** is limited by the neck and not by the capture: **80° each way**
    (`NAV.PITCH_MAX` = 1.396 rad), eased over its last 0.2 rad so the limit is a
    stop and not a wall. It used to be the recorded walk's own pitch range plus
    0.2 rad each way, which on the canonical world left nine of sixteen poses
    with under 7° of upward look. The recorded range is still computed and
    reported as `S.pitchRange`, beside `S.pitchLimit`; it is not enforced. What
    a wearer finds when they look up is whatever the capture holds there, which
    on this world is a torn ceiling — that is a reconstruction problem and the
    page does not hide it by refusing to look.
  - **Distance**: the camera keeps `CONFIG.standoff` scene units from the
    nearest proxy sample (resisted from 0.30 beyond it): nearer, a 360×640
    keyframe is magnified past its resolution and the frame fills with one
    blurred patch that the field scores as fully supported. Raised from 0.55 to
    1.0 by the fix-it framing lane, which found it close to inert on the
    canonical world *because the view-quality bound was stopping the push
    first*. With that bound gone this is the limit a forward push actually
    meets, and 1.0 on a world whose median scene depth is 4.7 is a fifth of the
    room, so it is **0.6** (2026-09-18). The blurred close-up it guards against
    is a MINOR finding in review 2; being unable to cross the room is a blocking
    one. The **reachable sampler** applies the standoff too, and so does the
    drift back inside: a push stopped by the standoff must not drift to a spot
    within it.
  - **Looking across a gap** is gone (2026-09-18) — `NAV.lookAcross` is removed,
    not left unfired. It existed to carry a look held against the support edge
    across to the next well-supported heading; there is no support edge to be
    held against any more, and it never fired even when there was.
  - **Before the field is built the finger is still answered** (2026-09-18).
    The field says where the capture *covers*; it is not needed to know where
    the camera may *stand*, which is the recorded walk. So a look turns and a
    move is held inside the tube from the first frame, and early input still
    cannot leave the envelope — which is what the rule of 2026-09-17 (*before
    the field is built the camera does not move*) was protecting. That rule
    shipped as a fix and read, from the outside, as a page that draws the room,
    enables its buttons and ignores the finger for several seconds with nothing
    on screen to say why (review 2, item 7: 40 frames of push and 20 of drag
    moved the camera exactly zero at `phase: "ready"`). `S.waiting` still
    reports whether the field is ready; nothing waits on it. When the field
    completes, the floor is the support of the recorded pose the camera is at,
    never of wherever it is.
  - **The envelope only limits the camera; it paints nothing.** Dark places
    inside it stay dark.
  - **Where the room is, while you are pointed away from it** (added
    2026-09-20, fix-it orient lane). Free looking is only honest if the page
    can say what the darkness means: on the canonical capture a full turn at
    the opening is 60% black, five consecutive 30° steps are over 95% black,
    and with nothing on screen that reads as a crash rather than as the truth.
    Three things, all of them reporting **coverage** and none of them implying
    content:
    - an **orientation ring** at the top right, on from the **first drawn
      frame** and labelled *photographed / from here* underneath, with **YOU**
      at its centre (2026-09-21, fix-it ux lane: it used to ramp its opacity
      in only once the field existed, a second or more after the first
      picture — exactly when a first-time viewer looks at it — and it carried
      no words at all, so its meaning lived only in the About panel. Without
      a field the arcs are drawn blank, which is the truth: the page does not
      know yet). `RING_BINS` = 36 headings on the horizon from where the camera
      stands, each one the same `support` the rest of the page uses at the
      same field of view, drawn as an arc that is lit where a frame exists and
      dim where none does, rotated so the current heading is always at the top
      (so the room being behind you *looks* like the room being behind you).
      The view's own horizontal spread is shaded inside it. The profile is
      recomputed only when the camera has moved more than 0.22 units — turning
      on the spot only rotates it — and costs 0.5–1.1 ms on SwiftShader; the
      per-frame cost is one `support` for the view in front of you and a 58 px
      2D draw. The sense of rotation is derived from `dirFrom` and the world's
      vertical, never assumed.
    - an **edge arrow** toward the shorter turn to the nearest covered
      heading, shown only while the view really is on nothing (`dark` > 0.60,
      released again below 0.25 — well below, because a frame that is 89%
      background can still hold a torn fragment and read 0.36 — and never for
      a turn under 0.35 rad), drawn with CSS borders rather than a glyph.
      **It does not change its mind at the antipode**: within `BACK_FLIP` =
      0.35 rad of the point where the room is exactly behind you the two ways
      home are the same length, so the side it already shows is kept
      (2026-09-21). (VISUAL-REVIEW-3 §8 reported "two contradictory arrows at
      once"; that is an artifact of that lane's own `crop.py`, which tiles
      crops from different frames edge to edge with no gutter, so one frame's
      right-edge chevron abuts the next frame's left-edge one. Every raw frame
      in its `raw_seq4` carries exactly one arrow — checked in
      `fixit\ux\tools\arrowside.py`. The flip that IS real is the one fixed
      here.)
    - a **dark line** above the bar, held for as long as the view is on
      nothing (see the darkness band above), which is also a control.
    - **Face the room** — a button, and the ring, the arrow and the dark line
      are the same control. It turns the camera, from where it stands, to the heading
      `NAV.bestHeading` names: the best-supported one discounted by how far
      you would have to turn (`RING_TURN_PENALTY` = 0.25 over half a turn), at
      the best of four pitches, as a **glide**, so a `pointerdown` cancels it
      like any other. It never translates — that is what Best view does — and
      when nothing at all is covered from where you stand it returns null and
      offers nothing rather than inventing a direction.

    The **settle** gained one fallback for the same reason: where the glasses
    never looked there is no content anywhere near, so the content gradient is
    exactly zero and a camera *the page placed* on nothing sat on nothing. It
    now falls back to the **coverage** gradient over a wider 0.60 rad. It is a
    nudge at the edge of the capture and not a way home — the whole budget is
    still `C_DRIFT_MAX`, about 17°, and half a turn of dark is 180° — and it
    is still refused for a look of the person's own and while a finger is
    down.
  - **A place nobody photographed is drawn as a flat grey haze** (added
    2026-09-21, fix-it ux lane, `UX.md` §8; `WORLD-BUILDER-APPEARANCE.md`
    §4.2b). Proxy geometry that no kept frame saw used to write alpha 0 and
    come out as the background. The caption had always called that *a grey
    haze … always darker than the room around it*, and the last independent
    review measured it: in the shipped render the "haze" is luminance
    **10–18** against a background of **11–28**, so in the opening view it was
    **darker than the emptiness it is supposed to be distinguishable from**,
    and at +90° **13.4%** of the frame was real geometry nobody ever
    photographed, indistinguishable from nothing at all. Those fragments now
    write **one flat colour** (display RGB 0.160 / 0.168 / 0.190, alpha
    `CONFIG.unseen_haze` = 0.9) with no texture, no hue of its own and no
    detail at any scale. It is the only thing on the page that is painted
    rather than photographed, and that is exactly what it says: *there is a
    surface here and there is no picture of it*. It invents nothing — a flat
    patch cannot be mistaken for imagery — and **no measurement sees it**: the
    observed/unobserved mask (mode 2), the drawn fraction the opening and the
    Best view are scored on (mode 3) and the fade's own cost (mode 11) all
    return before it is drawn, and the uniform is zero for every mode but the
    display one. `unseen_haze: 0` restores the previous behaviour exactly.
    Measured over the review's nineteen poses: the haze reads **27–42**
    against a background of 11–28, it is darker than the background in **1 of
    19** views (14 of 19 before), and the median separation is **+18.9** where
    it was −1.8.
  - **Confidence fades appearance toward the haze** (added 2026-09-20, fix-it
    orient lane; `WORLD-BUILDER-APPEARANCE.md` §4.2a). The proxy's R colour
    byte is the surface's per-vertex geometry confidence. Per fragment,
    `k = smoothstep(CONFIG.confidence_lo, CONFIG.confidence_hi, conf)` and the
    colour is `mix(haze(), colour, k)`: below the band the photograph
    gives way entirely to the mark the page draws where it knows
    nothing, above it nothing changes, and between them it crosses over
    smoothly. Never a hard cut.
    **It faded toward the background until 2026-09-21, and that is why nobody
    could see it** (fix-it ux lane, `UX.md` §9). A/B'd at `?clo=0&chi=0` over
    nineteen poses, the fade changed a median **0.22 of an 8-bit level** over
    the whole frame and 0.60% of the frame visibly — because it faded dark
    geometry toward a near-black background, so where it acted hardest there
    was least to see. Against the haze it changes **1.56%** of the frame
    (median) by ~20 levels, in exactly the places the page will not vouch for.
    **The band itself did not move**, and raising it was measured and
    rejected: 110/255 doubles the cost (2.53% of drawn against 0.74%) for no
    visible gain, and 140/255 (6.85% of drawn) begins fogging real
    photographs. **The alpha is not touched**: alpha is the
    evidence that a frame saw the place, the depth still hides what is behind,
    and no pixel becomes see-through. `proxy.confidence.present === false` is
    *unknown*, not zero, and disables the fade entirely (the attribute is then
    the constant 1). The band is a **phone-level** band and is served as
    `confidence_lo` / `confidence_hi` (24/255 and 78/255); reusing a level-0
    number here fades 19% of the opening view. The thin cracks `FS_FILL`
    closes are not faded — they have no vertex to read a confidence from —
    which is at most 4 CSS px anywhere. Debug mode 11 renders what the fade
    removes (R) over what is drawn (G), which is what `S.fadeCost()` reads.
  - **Controls.** Touch: one finger drags the look (the room follows the
    finger), two fingers drag to move sideways and up/down, pinch moves forward
    and back (log of the spread × 2 units). Desktop (debug): left drag looks,
    right or shift drag moves, the wheel moves forward/back, W A S D Q E fly,
    ←/→ step the walk, O is the Best view. Motion coasts after a flick (220 ms
    time constant). Buttons: **Best view**, **Face the room**, **←**, **→**
    (the recorded walk), **Reset** (the opening).
  - **Stepping the walk glides**: eased position and the short way round in
    yaw, 400–2600 ms by distance and turn, instead of jumping (the walkthrough's
    worst flicker steps were path jumps). Any touch interrupts a glide. The cap
    was 1600 ms until 2026-09-18; the Best view is several units away and at
    that cap the crossing read as a teleport.
  - **Best view** (called **Overview** until 2026-09-18, which was a promise
    this capture cannot keep — from any starting pose it landed on the same
    close-up of the monitor, about 5 units away, and all twelve candidates the
    field proposes are the same desk area because the walk never stood back from
    it; the button is now named for what it does, says so in the About text, and
    flies rather than jumps). A **raised** vantage in the envelope that
    looks at the room. The field proposes (points at least 0.15 above the walk,
    within **half** of the tube — the part a released camera does not drift
    out of; 0.85 until 2026-09-21, which let the destination land at envelope
    0.52–0.78, reachable but inside the drift band, so the page could fly you
    somewhere it would then quietly pull you away from. It is a preference,
    relaxed in 0.15 steps rather than allowed to leave the button dead
    (`S.overviewStats.maxE` says which) — and at least 1.2 from the opening pose, 12 yaws × 2
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
    depth range breaks ties. In portrait this makes it the room in one
    frame. **In landscape it does not**: on the canonical world all twelve
    candidates are the same desk area, because the recorded walk never stood
    back from it, so no vantage inside the envelope frames the whole room.
    2026-09-18 (fix-it interaction lane): **the score is unchanged.** Weighting
    the range hardest of all in landscape (0.25 + 0.75 × range) was tried and
    measured to be a *no-op* on the canonical world — with and without it the
    same candidate wins — so the framing lane's swept weights stand.
    What was wrong with the button was its **name** and the **speed** it
    arrived at: it now says *Best view*, the About text says what that is and
    why there is no overview to give, and the crossing from the opening —
    5.3 units — takes **2394 ms** instead of being clipped to 1600.
    **The destination moved when the envelope did, and it has been re-swept**
    (2026-09-20, fix-it orient lane). A wider tube and a smaller standoff took
    the candidate set from 205 to 406 and handed the score a vantage half a
    unit further back: more of the room in frame (depth range 4.92 against
    3.69) and **88.7% drawn against 99.8%**, with a shredded region in the
    lower right. Nothing the score measured could see the difference, because
    *drawn, deep, detailed and with range* is as true of a photograph smeared
    over wrong geometry as of a photograph of the room — and on this candidate
    set the **detail** term is saturated for all twelve and decides nothing.
    The confidence channel can see it, and the page now draws it, so the score
    reads it off the same render: one more factor,
    `soundTerm(faded) = 0.45 + 0.55 × (1 − min(1, faded / 0.05))`, where
    `faded` is the share of the drawn area the confidence fade removes.
    Measured on the twelve candidates, `faded` splits them cleanly — ten at
    0.16–0.93%, two at 2.34% and 3.07%, and those two are exactly the two
    shredded frames. Every rule that reads the channel (a graded penalty, the
    range-dominant variant, or a hard veto at 2%) then picks the same
    candidate: the 99.8%-drawn vantage the score chose on the 1.0-unit tube,
    by 4.6% where the old rule preferred the shredded one by 1.4%. **The
    button is still not an overview** and the About text still says so: the
    re-sweep buys a frame that is not wrecked, not a view of the room.
    (`fixit\orient\ORIENT.md` §4, `fig\sheet_ov_candidates.png`.)

    **The destination is stable, and it is judged on how torn the frame is**
    (2026-09-21, fix-it bestview lane; `fixit\bestview\BESTVIEW.md`).

    - *Stable.* Forty cold boots — twenty of each of two builds — chose one
      destination each, and the whole twelve-row candidate table came back
      identical to the last decimal in all forty, under two and three headless
      browsers at once. The chooser is not random and the field is not raced.
      What it was sensitive to was **the camera the reader left behind**:
      `navView()` returns the verification override's field of view whenever
      one is set, so a press after `setView(…fovYDeg 70)` filtered the field
      through a frame the page never draws — 97 candidates against a fresh
      page's 126 on one build, 111 against 138 on the other — and landed
      somewhere else, and the orbit hook's 54° gave a third place again. Two
      lanes measured the same two builds and disagreed about where Best view
      goes for exactly that reason. `findOverview` now clears the override
      **before** it takes its view and restores it at every exit, so the
      destination is a function of the field, the walk, the opening and the
      canvas and of nothing the reader did first; and the ranking has an
      explicit tie-break (score, then yaw, then position) so an exact tie does
      not depend on the order the lattice was walked in. Measured after: the
      same destination from a fresh page, after a walk step, after `setView`,
      after `orbitView` and on a second press, on both builds. Portrait still
      differs from landscape, which is by design — the frame is a different
      shape, so the best frame is a different frame.
    - *Judged on how torn it is.* The confidence term catches a candidate
      whose geometry the fade eats; it did not catch the one the owner would
      have seen, whose `faded` is 0.40% — cleaner than half the shortlist —
      and which carries a stair-stepped black shape and a column of bright
      cream fragments down its right quarter. One more factor,
      `cleanTerm(torn) = 0.45 + 0.55 × (1 − min(1, torn / 0.02))`, where
      `torn` is the length of the boundary between what is drawn and what is
      not, per frame pixel, measured on the candidate's own probe render with
      no extra pass. `drawn` cannot tell one clean black rectangle from a
      thousand slivers of the same area; this can. Over both builds' twelve
      candidates the measure splits them the way looking at them does —
      0.0022–0.0088 for the frames with nothing shredded in them, 0.0107–0.0231
      for the torn ones — and every reference from 0.010 to 0.030 at every
      floor from 0.15 to 0.45 picks the same candidate on both builds, by
      8–42%. The shortlist's probe rises from **64 px to 128 px**, because at
      64 a torn edge and a straight one are the same handful of pixels; the
      press costs 993 → 1833 ms and 1153 → 1675 ms on SwiftShader.
      **`speck` is measured, reported and deliberately not scored**: it is the
      campaign's speckle rule with a box mean for the median, and at the probe
      size it does not reproduce what it stands for — the fragments in that
      column are one pixel wide at 900×700 and gone at 128 — so penalising it
      picks a frame that measures cleaner and looks worse. Blotch is not
      scored either: on this world it counts the dark desk, and the cleanest
      frame in the shortlist has the highest blotch of all.
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
    **A press covers ground first** (2026-09-21, fix-it ux lane, `UX.md` §7).
    The walk was recorded at whatever rate the wearer moved, and on this
    capture that is as little as **0.049 scene units a pose** through the desk
    stretch — fifteen presses moved the camera 0.73 units, which is not
    stepping through a walk but watching yourself sit at a desk. A press now
    advances along the path until it has gone `STEP_UNITS` (0.32 × the scene's
    median depth ÷ 4.7, so 0.28 here) or has passed `STEP_MAX_POSES` = 12,
    whichever comes first, and the quality rule above then decides where in
    that run it lands. Both ends of the walk stay inert. Measured from the
    review's own starting index: **0.165 units a press before, 0.308 after**
    (index 4 → 19 against 4 → 25 over fifteen presses).
  - Before the field is built (a second or so after the images land) the look
    and the walk work, the status line reads **“Preparing the view…”**, and
    **Best view** is disabled. Both of the things the field is needed for are
    things the page *says*, not things it does: which vantage is best, and why
    a direction is dark. A first drag in that window turns, and on a world like
    the canonical one it turns straight into the part of the room nobody
    photographed — so without the line the page would answer the finger with a
    black screen and no account of it. The line is cleared the moment the field
    completes, and only if nothing else has since written to the status.
    **The field is built twice** (2026-09-20, fix-it orient lane): a coarse
    pass at twice the spacing, which is an eighth of the work and answers the
    same question — over 24 headings the two fields never disagree about
    whether a direction is covered, and nowhere by more than 0.2 of support —
    and then the real one, which replaces it silently. The ring, the dark
    hint, *Face the room* and `S.navReady()` land with the coarse pass; **Best
    view** waits for the fine one, because its candidates are the field's own
    lattice points and a coarse lattice may hold too few. On the canonical
    world, quiet: the *Preparing the view…* window falls from 1007 to 790 ms
    and the Best view arrives at 1.86 s instead of 1.01 s. Under six-way
    contention, where the field work dominates: 2964–3629 ms → 1316–2312 ms
    for the window, and 3.0–3.6 s → 5.4–7.6 s for the button. The coarse pass
    costs 136 ms of extra work (935 against 799 ms in total).
    **The field starts at the first drawn frame, beside the rest of the
    opening scan** (2026-09-21, fix-it ux lane), not after it: both are sliced
    on a 12–14 ms budget, so they interleave instead of queueing. What must
    NOT run beside the fine pass is `scorePoses` (198 renders), which pushed
    it from 1.0 s to 3.7–4.7 s when that was tried; it still waits.

  **The cold open shows its state before it shows its chrome** (2026-09-21,
  fix-it ux lane, `UX.md` §1–2). Until then, every cold open put the finished
  caption and the whole button bar on screen over a pure black canvas with
  the status line cleared: 1.45–1.71 s on the reviewer's machine and
  **2.8–3.1 s** on the lane's, ending in a first lit frame at 6.1–6.5 s. The
  natural reading of those frames is that the page is broken. Three rules now
  hold, and each is a unit:

  - **The chrome does not arrive before the picture.** `<body class="booting">`
    hides `#caption` and `#bar`; the class comes off at the first drawn frame,
    so the room and the controls arrive together. The buttons exist in the
    markup from the first byte — something has to enable them — they are just
    not shown.
  - **The page never falls silent while it is still working.** The status runs
    *Loading the images…* → *Loading the room's surface…* → *Placing images
    n / N* → **“Choosing where to open…”** (the opening scan, about two
    seconds of its own, which said nothing at all before) → *Preparing the
    view…* → nothing.
  - **A pose is placed and drawn long before the scan finishes.** The coarse
    scan runs in a bit-reversed order, so its first six candidates are spread
    over the whole recorded walk rather than over its first few poses, and the
    best of those six is placed and drawn (`S.provisional`). When the scan
    finishes the page corrects to the real opening — a cut, not a glide, so
    `phase: ready` never describes a camera in flight — **unless the reader has
    touched the glass, in which case the view is theirs and the page leaves it
    alone** (`S.openingKept`).

  Measured on the lane's machine, three quiet loads each, polling only: the
  longest stretch that is black with nothing said falls from **2.83–3.13 s to
  0.00 s**; the first lit frame from **5.65–6.04 s to 4.20–4.30 s**; *Reset*
  becomes usable at the first drawn frame (it was never disabled and never
  useful before it). *Best view* moves the other way, **7.84 s → 8.73 s**,
  which is the cost of drawing the room 1.45 s sooner on a renderer where one
  frame is ~400 ms rather than a phone's ~16.

  **Input before the first drawn frame is refused out loud, never banked.**
  `frame()` draws nothing until the opening is placed, so anything pushed into
  `ctl` before that just accumulated — and then ran, all of it, in the first
  frame after the handover. An independent review put a thumb on the glass
  from the first instant and landed at **yaw 2.90 rad** in the empty half of
  the room, where it stayed for the nineteen seconds they watched, because a
  look the person made is deliberately never taken back. You cannot aim at a
  picture you have not seen: `deferInput` clears `ctl` and `vel`, says *One
  moment — you can look around as soon as it draws*, counts the event in
  `S.deferredInputs`, and every way in goes through it (the pointer handlers,
  the wheel, the fly keys and `S.input`). Re-run: **0 of 44 sampled frames
  after the handover are dark**, against 41 of 41.

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
  areas mean, that the arrows pass over poses that render badly or show
  nothing (at most 4 in a row, so the walk shown is shorter than the one
  recorded but never misses a stretch of it), **that turning is free and
  moving is not, what the ring at the top right is and what *Face the room*
  does, and what the Best view is**, open on tap (the four-line
  disclaimer covered 11–13% of the frame). The expanded text carries its own
  dark plate rather than relying on the caption's gradient scrim: the scrim
  fades out over the top of the frame and the paragraph is longer than it, so
  over a bright frame the tail of the sentence that keeps the page honest was
  grey on cream (2026-09-18, review 2 item 9).
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
  `withdrawn`, `absent`, `unavailable`, an old Tower's bare `null`, or a
  revision 404 naming a world or session that is gone (m3) deletes every texture
  at once and says why —
  **and keeps polling**, so a rebuild that is served again is drawn again without
  closing the viewer. The reason differs: `unavailable` is the Tower's own
  appearance code having raised, and the page says *the Tower could not answer
  for this world's images just now* rather than telling the wearer their
  redaction record changed (2026-09-17, review 2 m-15).
  **A boot that places no imagery takes the same path**: it says so, keeps
  polling, and finishes its opening when a build it can draw arrives. It used to
  `fail()`, which stops the script above the follower, so the page never asked
  again — and the app could not tell, because the page reported `didFinish` like
  any other (2026-09-17, review 2 M-0). A chunk or proxy that 404s while loading (the next build
  published and the file left its grace) refetches the manifest and tries once
  more (M4).
- **Context loss.** Textures and the manifest are not kept across it: on
  restore the page fetches the manifest again, then the proxy and chunks,
  through the same transport (so the Tower's label check runs again, and a build
  or withdrawal that happened meanwhile is honoured) and keeps the camera. If
  WebKit has not restored the context after 3 s the page asks for it
  (`WEBGL_lose_context.restoreContext`, re-fetched per context); after 10 s it
  reloads itself, which the app allows for exactly this page
  (`WORLD-BUILDER-IOS.md` §10; review 1 m8). The reload is a **watchdog, not a
  one-shot**: `webglcontextrestored` re-arms it (to 90 s) instead of cancelling
  it, and only a restore that finished — drawn, or saying truthfully that there
  is nothing to draw — clears it. It used to be cancelled by the first statement
  of the restore handler, before the manifest and every chunk were fetched
  again, so the slowest and least reliable part of the restore ran with no
  escape at all (2026-09-17, review 2 P-1). Every fetch the page makes is
  bounded by an `AbortController` at 45 s for the same reason: on
  `glasses-world:` a request settles only if the app's handler answers it, and a
  handler that loses a task would otherwise leave the page waiting for ever.
- **Caption.** *Captured images on reconstructed geometry · N of M keyframes
  shown* — N and M are both the PHONE tier, so the figure is "of what this
  device was offered" and not "of what the Tower keeps" — (· *still building as
  you walk*, · the currency reason when behind, · a reduced set and **which
  side** lacks compressed textures: this device, or a Tower that built none),
  then **five short titled sections**, not one paragraph (2026-09-21, fix-it
  ux lane): *What you are looking at*, *The flat grey patches*, *Looking and
  moving*, *Finding your way*, *The walk*, and finally *Scale is unknown, so
  distances are relative.* The panel scrolls at `max-height: 44vh`.
  The words are all but unchanged; the shape is not. The last review called
  the text "correct, honest and well written — and one 250-word paragraph in
  small type filling 28% of a phone screen, with no headings and no breaks".
  Measured: **335 words in a single block, 50.3% of the frame's height**;
  after, 357 words in five sections whose longest block is **89 words**, at
  **44%** and scrollable.
  Two sentences changed, both because a measurement contradicted them:
  - *A grey haze … it is always darker than the room around it* became **“A
    flat grey patch is a place no kept frame saw, or one that was masked as
    unreliable (redaction, hands, views that disagreed). It has no texture and
    no detail at any scale, and it is never an image of anything: it is there
    so that ‘nobody photographed this’ does not look like ‘there is nothing
    here’.”** The old claim was false as written — the haze measured luminance
    10–18 against a background of 11–28 — and the page now draws a patch that
    makes the new one true and checkable.
  - *Moving is not free* gained **“so you can step back, stand and crouch a
    little”**, which is what the widened tube bought.
  The *Cracks a few pixels wide …* sentence is **unchanged, deliberately**: a
  parallel lane is checking that claim against multi-view fill of redacted
  regions, and nothing in this lane alters what the crack fill does.
  It said *"Dark areas … nothing is filled in there"* until 2026-09-17: written
  before the crack fill and the void fog and not revisited, so the page's own
  caption denied two things the page does (review 2, M-5). The replacement
  states the BOUND rather than a denial, which is the part a wearer can act on.
  The native caption above the web view says the same in one line
  (`WORLD-BUILDER-IOS.md` §10).
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

**200** `{"session_id": str, "representation": "appearance"|"surface"|"dense"|"sparse", "revision": str, "live": bool, "appearance": {"revision": str|null, "current": bool, "state": "served"|"rebuilding"|"withdrawn"|"absent"|"unavailable", "epoch": str|null}}`,
`Cache-Control: no-store`. **404** exactly when §4 would 404.

`appearance` (additive, 2026-09-17) follows the session's appearance artifact
(`WORLD-BUILDER-APPEARANCE.md`): `revision` is
`<session_id>/appearance:<build_id>` exactly when
`GET /worlds/{id}/appearance/{session_id}/manifest` would answer 200, and `null`
otherwise — no artifact, a purged world, **a session whose redaction label no
longer matches the one the artifact was built under**, or the Tower's own
appearance code having raised (`unavailable`, 2026-09-17: a crash is not a
privacy event and must not be reported as one). `state` says which, and
what a page holding textures does (`WORLD-BUILDER-APPEARANCE.md` §9): keep them
through `rebuilding` (the ordinary Stop), drop them on `withdrawn`, `absent` or
`unavailable`, and in every case keep asking. `epoch` changes exactly when an open page must
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

`components` (**PROPOSED 2026-09-23 — awaiting Mac review; nothing
implemented**): additive, the chosen session's `components` array exactly as on
its §2 row, or `null` when not computed. **Not part of `revision`**: an area
finishing never swaps the room page. Areas are not a rung and not a query
parameter of §4; they have their own routes, `GET /worlds/{w}/areas/{s}/{a}/…`
(`WORLD-BUILDER-COMPONENTS.md` §3.2 and §5).

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
