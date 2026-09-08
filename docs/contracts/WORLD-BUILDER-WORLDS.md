# World Builder saved worlds — the listing a viewer opens old worlds from

**Living document.** Added 2026-09-06.

| | |
|---|---|
| Contract | `world_builder.worlds/2026-09-06` |
| Transport | HTTP `GET /worlds`, same origin and same rules as the geometry routes (`WORLD-BUILDER-GEOMETRY.md` §1) |
| Tower producer | `tower/tower/results/world_builder_library.py` |
| Tower route | `tower/tower/routes/geometry.py` |
| iOS consumer | `ios/Glasses/Workspaces/WorldBuilder/WorldLibrary.swift` (compiled and tested on the Mac, 2026-09-06) |

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
| `contract` | string | `world_builder.worlds/2026-09-06`. Compared for equality; a mismatch refuses the payload |
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
| `has_geometry` | bool | A derived tree exists for this session (`poses.json` and `points.json`). The geometry manifest may still answer `current: false` |
| `abandoned` | bool | `ended_at` is `null` **and** the world is not `live`: the builder ended without finalising the record (killed inside the Tower's shutdown grace, or by hand). Its `keyframes_accepted` is then the start-of-session value, not the journal length, and its geometry is whatever the last interim build wrote. Additive (2026-09-06, Windows validation) |
| `state` | string | One word, the status channel's lifecycle vocabulary: `receiving` (a live builder, session open), `finalizing` (a live builder, session stopped — the final solve and build are running), `complete` (finished; `finalization.state == "complete"`, or an older record that stopped and built), `interrupted` (killed mid-walk, killed mid-finalization, stopped by a request or an error — **`has_geometry` still says whether a reconstruction exists**), `unbuilt` (an older record that stopped and never built). Additive (2026-09-06, live-history lane) |
| `keyframes_journaled` | int | Lines in the keyframe journal. On a record that never stopped `keyframes_accepted` is still the zero written at start; this is what actually landed (467 on the 09-06 walk against a recorded 0). Show this when the two disagree. Additive (2026-09-06) |
| `finalization` | object \| null | The builder's own record of what happened after the session stopped: `{state: pending\|complete\|interrupted, final_solve: pending\|solved\|skipped\|failed\|unavailable\|null, started_at, updated_at, detail}`. `null` on records written before 2026-09-06 and on sessions that never stopped. Additive (2026-09-06) |

## 3. Rules

1. **Absent is never zero** (`WORLD-BUILDER-GEOMETRY.md` §6 applies): `ended_at: null` is an open session; `display_name: null` is an unnamed world. An open session with `abandoned: true` is not still open: nobody is writing it, and a client should say so rather than "still open". Prefer `state` when it is present; `abandoned` is kept for clients that predate it.
5. **An interrupted session is not hidden and is not presented as finished.** `state: "interrupted"` with `has_geometry: true` is a world a person can open and look at (the render route serves it); the row must say interrupted and must not say complete. A live Tower has no reason to hide the 2026-09-06 walk, and every reason not to call it finished.
6. **Worlds with no sessions are shells**, left by a builder that opened a world and never received a frame (96 of 162 on the Windows box). A picker may fold them away; it must not offer them as the primary rows.
2. **A world that cannot be read is omitted, never invented.** A session that cannot be read is omitted from its world.
3. **No imagery, no paths.** `capture_id` is an opaque id, not a location.
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
| `max_points` | int 1…200000, optional | Point budget. **Sparse:** default 80,000 for a phone, fractional-stride sampling over every segment, never a prefix. **Dense:** default 393,216 (a 6 MB binary buffer), met by a coarser voxel grid over the whole extent rather than by sampling; an explicit value is honoured as a cap |
| `representation` | `auto` \| `sparse` \| `dense`, optional | Which page to serve. `auto` (default) serves the dense viewer when the session has a dense artifact and the sparse page otherwise; `sparse` forces the old page; `dense` returns **404** rather than falling back. **422** outside this set |

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
6. The response carries `Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'` — the promise above, enforced by the browser as well as kept by the producer.
