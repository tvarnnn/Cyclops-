# World Builder saved worlds — the listing a viewer opens old worlds from

**Living document.** Added 2026-09-06.

| | |
|---|---|
| Contract | `world_builder.worlds/2026-09-06` |
| Transport | HTTP `GET /worlds`, same origin and same rules as the geometry routes (`WORLD-BUILDER-GEOMETRY.md` §1) |
| Tower producer | `tower/tower/results/world_builder_library.py` |
| Tower route | `tower/tower/routes/geometry.py` |
| iOS consumer | `ios/Glasses/Workspaces/WorldBuilder/WorldLibrary.swift` (written 2026-09-06, **not yet compiled**) |

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

## 3. Rules

1. **Absent is never zero** (`WORLD-BUILDER-GEOMETRY.md` §6 applies): `ended_at: null` is an open session; `display_name: null` is an unnamed world.
2. **A world that cannot be read is omitted, never invented.** A session that cannot be read is omitted from its world.
3. **No imagery, no paths.** `capture_id` is an opaque id, not a location.
4. Opening a world means: subscribe with `world_id` (+ `session_id`) on the status channel, then pull geometry exactly as for the live world. There is no second geometry path.
