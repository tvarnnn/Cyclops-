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
| `representation` | `auto` \| `sparse` \| `dense` \| `surface`, optional | Which reconstruction to serve, as a ladder: `surface` → `dense` → `sparse`. `auto` (default) serves the best rung the session actually has. A named value starts the walk at its own rung and falls through to worse ones, **except** that naming a rung the session does not have at all returns **404** rather than silently serving a different one — a caller that pinned a representation is comparing, and a silent substitution would corrupt the comparison. **422** outside this set. The rung actually served is stated in the page (`wb-representation`) and by `GET /worlds/{id}/render/revision` (§4a); the `GET /worlds` listing does not state it |
| `view` | `product` \| `diagnostics`, optional | Which rendering the **sparse** page opens in: the product view (default) or the solver's segment-coloured diagnostics view. Honoured server-side, because a `loadHTMLString` client has no `location.search`. An unrecognised value opens the product view, never a 422. **`view=diagnostics` with `representation=auto` serves the sparse page**, whatever other rungs the session has — only the sparse page has the diagnostics rendering, so starting the ladder at the surface (or the dense rung) would answer "open the solver's view" with a surface or dense points. A pinned `representation` still wins |

**200** `text/html`, `Cache-Control: no-store`. A self-contained page: no
external script, stylesheet, image or fetch, so a web view that refuses
every navigation but the initial one shows it whole. One finger orbits,
two fingers pinch to zoom and drag to pan; on a desktop, drag / wheel /
shift-drag.

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
6. The response carries `Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'` — the promise above, enforced by the browser as well as kept by the producer — **and every page carries the same policy as a `<meta http-equiv="Content-Security-Policy">` right after `<meta charset>`**, which is the only form a `loadHTMLString` client enforces: iOS drops the response headers, and its navigation policy sees navigations, not an `<img>` or a `fetch`. All three pages (sparse, dense, surface) carry it, and it never pushes `wb-representation` out of the first 4096 characters.

## 4a. `GET /worlds/{world_id}/render/revision` — has a better picture been built?

Additive. A client may keep a picture from §4 open while the Tower is still
building the session — during a walk the surface is rebuilt each time a global
solve lands — and ask this, a few hundred bytes, instead of re-downloading the
page to find out.

| param | type | meaning |
|---|---|---|
| `session_id` | string, optional | as §4; resolved the same way |
| `view` | string, optional | as §4. `view=diagnostics` reports the **sparse** rung, because that is the page §4 serves for it. A client showing the diagnostics rendering has nothing to follow and should not ask (the iOS app does not) |

**200** `{"session_id": str, "representation": "surface"|"dense"|"sparse", "revision": str, "live": bool, "appearance": {"revision": str|null, "current": bool}}`,
`Cache-Control: no-store`. **404** exactly when §4 would 404.

`appearance` (additive, 2026-09-17) follows the session's appearance artifact
(`WORLD-BUILDER-APPEARANCE.md`): `revision` is
`<session_id>/appearance:<build_id>` exactly when
`GET /worlds/{id}/appearance/{session_id}/manifest` would answer 200, and `null`
otherwise — no artifact, a purged world, or **a session whose redaction label no
longer matches the one the artifact was built under**, so a page holding
textures drops them when it sees `null`. `current` is false when the artifact
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
3. The sparse rung's revision is the constant `<session_id>/sparse`. The derived
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
  `redaction` differs from the one the artifact was built under answers 404
  ("appearance is stale against the session's redaction record"), as does a
  world with `images_purged`.
- `digest` is 32 lower-hex and must be named by the current manifest. URLs carry
  no path, file name or capture sequence number.
- `world_id` passes `contained_world_id`; `session_id` must be one of the
  world's sessions.
- §4's page is unchanged by this: it still loads nothing from anywhere and its
  CSP is still `default-src 'none'`. How a page reaches these routes is the
  phone lane's design, not this route's.
