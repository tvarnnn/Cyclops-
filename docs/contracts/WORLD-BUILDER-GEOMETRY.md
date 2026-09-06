# World Builder geometry — the Tower↔iOS boundary

**Living document.** It describes the boundary as it exists now.

| | |
|---|---|
| Contract | `world_builder.geometry/2026-08-25` |
| Transport | HTTP `GET`, two endpoints. **Not the WebSocket** — see §1 |
| Tower producer | `tower/tower/results/world_builder_geometry.py` |
| Tower routes | `tower/tower/routes/geometry.py` |
| iOS consumer | `ios/Glasses/Workspaces/WorldBuilder/WorldGeometry{,Client}.swift` |
| Design record | `tower/docs/superpowers/specs/2026-08-25-world-builder-geometry-transport-design.md` |

**Why this is a separate file from `WORLD-BUILDER-IOS.md`.** That document
reconciles **one** contract (`world_builder.status/2026-08-25`) over **one**
transport, and its whole thesis is *"no second socket, no second connection, no
view-owned transport"*. Geometry is a different contract with a different
identifier, on a different transport, versioned independently — a client may
implement status alone and many will. Folding a second transport's field tables
into a document built around a single seam would blur the exact distinction
that document exists to make. `docs/contracts/` already keeps one file per
contract; this follows it. `WORLD-BUILDER-IOS.md` §1 links here.

**Status:** implemented on both sides. iOS has not been compiled in this
environment (no Swift toolchain); the Tower half is exercised by
`tower/tests/test_world_builder_geometry_transport.py`.

---

## 1. Why geometry is HTTP and status is the WebSocket

Not taste — a measurement and a lock.

**The lock.** `tower/tower/routes/ws.py:38` gives the result sender and the
frame path a single `asyncio.Lock` on one WebSocket. Anything sent on the
result channel holds that lock while it serialises and writes.

**The size.** One real session's `points.json` is **1.07 MB**. The status
snapshot beside it is **3,884 bytes** — a factor of ~275. Pushing geometry down
the socket would hold the shared lock for the duration of a megabyte write and
starve `frame_result`, which is precisely what `CARTRIDGE-RESULTS.md` forbids
in Tower responsibility #3.

**The consequences of the split, stated so neither side is surprised:**

- Geometry is **pulled**, never pushed. The status payload's existing
  `geometry.revision` is the "refetch the manifest" signal; no field was added
  to the status contract for this.
- Both Tower handlers are declared `def`, not `async def`, so FastAPI runs them
  in its threadpool and the 1 MB read plus the content hash stay off the event
  loop. This is pinned by a test, because it is the whole mechanism.
- The HTTP endpoints are unauthenticated on the same LAN-local origin the
  socket uses. They read; they never write.

```
 Tower web process ──ws://…/ws──────────┐  status: small, pushed, shares
                                        │  the frame path's send lock
                                        ▼
                                  TowerClient
                                        │ cartridgeResults
                                        ▼
 Tower web process          TowerWorldBuilderClient ──┐
        ▲                               │             │ geometry.revision moved
        │  http://…/worlds/…/geometry/* │             ▼
        └───────────────────────────────┴──── WorldGeometryClient  (HTTP GET)
           manifest, then one GET per segment          │
           bulk, pulled, its own connection            ▼
                                             WorldGeometryStore (cache, by
                                             content_hash) → WorldFragmentsView
```

---

## 2. `GET /worlds/{world_id}/geometry/manifest`

| Query parameter | Required | Meaning |
|---|---|---|
| `session_id` | yes | Which session's derived tree to describe. A world may hold several |

**404** when there is no derived tree for that world and session, or when the
world is unknown. **200** otherwise — including when the tree is *behind* the
journal. See §4.

### 2.1 Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `contract` | string | `world_builder.geometry/2026-08-25`. Opaque; compared for equality only. A mismatch refuses the whole payload |
| `world_id` | string | Echoed |
| `session_id` | string | Echoed |
| `current` | bool | Whether this geometry reflects **every keyframe accepted so far**. `false` is normal during a live walk — see §4 |
| `geometry_revision` | string | Opaque rollup over every segment's `content_hash` AND `placement_hash`, so a placement-only change moves it. See §7. Equality only |
| `pose_convention` | object | Nine keys; see §5 rule 1 |
| `scale` | object | `{state, meters_per_unit}`. `meters_per_unit` is `null` unless the state is metric, which is unreachable on this hardware |
| `segment_count` | int | Length of `segments` |
| `segments` | array | One row per segment, ascending by index — **including segments that resolved to nothing** |

### 2.2 Per-segment fields

| Field | Type | Meaning |
|---|---|---|
| `segment_index` | int | Identity within the session |
| `content_hash` | string | 16 hex chars over the segment's *whole* poses and points. The client's cache key. Opaque |
| `frame_id` | string | `"segment:{index}"`. Two segments never share a frame today |
| `registered` | bool | `true` once a segment is placed. See §5 rule 3 |
| `registration_state` | string | `unplaced` \| `registered` \| `refused`. Distinguishes "we tried and the solves disagreed" from "nobody looked" — `registered: false` alone could not |
| `registration_refusal_reason` | string\|null | why this segment is not placed. Usually "the wearer stood still", which is a message to the wearer, not a fault |
| `placement_hash` | string | 16 hex over WHERE the segment sits. **A cache key** — see §7 |
| `transform_to_world` | object\|null | `null` unless registered. `{rotation_wxyz[4], translation[3], scale, reference_segment, frame_revision}` — maps this segment's frame into `reference_segment`'s |
| `resolution_state` | `resolved`\|`unresolved` | `resolved` iff `point_count > 0` |
| `dominant_degeneracy` | string\|null | Most frequent non-empty `degeneracy` among this segment's refused rows; `null` when none was refused. Present on resolved segments too — a segment can solve some poses and refuse others |
| `keyframe_count` | int | Pose rows for this segment. Exactly one per keyframe, so also the length of the chunk's `poses` |
| `solved_count` | int | Rows with `status == "solved"`. The **only** count that is evidence of measured camera motion |
| `point_count` | int | Points in this segment |
| `bounds` | object\|null | 3D `{min:[x,y,z], max:[x,y,z]}` over this segment's points, in the segment's own frame. `null` when `point_count == 0` — see §6 |

`pose_count` and `anchor_count` are deliberately **not** carried: the first
would duplicate `keyframe_count`, and the second is always exactly 1 per
segment.

---

## 3. `GET /worlds/{world_id}/geometry/segment/{segment_index}`

| Query parameter | Required | Meaning |
|---|---|---|
| `session_id` | yes | As above |
| `max_points` | no | Cap on points returned, `>= 1` (a value below 1 is **422** at the edge, never a 500). Default: unlimited |

**404** when the world, the session's derived tree, or that segment index is
absent. **200** otherwise, behind-the-journal included.

| Field | Type | Meaning |
|---|---|---|
| `contract` | string | As above |
| `current` | bool | Same meaning as on the manifest, **repeated here on purpose** — see §4 |
| `segment_index` | int | Echoed |
| `content_hash` | string | Identical to the manifest's for this segment, and identical whether or not the cloud was sampled: the hash identifies the **segment**, not the transfer |
| `frame_id` | string | `"segment:{index}"` |
| `registered` | bool | as the manifest |
| `registration_state` | string | as the manifest |
| `registration_refusal_reason` | string\|null | as the manifest |
| `transform_to_world` | object\|null | as the manifest |
| `placement_hash` | string | as the manifest |
| `poses` | array | `{keyframe_id, status, degeneracy, rotation, translation}`, in `poses.json` order, which is index-aligned to `keyframes.jsonl` (457/457 on the real world) |
| `points` | array | Bare `[x,y,z]` triples. Not tagged rows: the chunk already names its segment |
| `points_sent` | int | Points in `points` |
| `points_total` | int | Points the segment holds |
| `point_sampling` | `none`\|`stride` | How `points` was reduced, if at all |

**Sampling spans the whole cloud and is never a prefix.** A prefix is one
corner of the room and would read as a *smaller* world rather than a coarser
one. The stride is fractional (`total / max_points`) for exactly this reason:
an integer stride collapses to 1 whenever `max_points > total/2`, which turned
"cap 3,033 points at 2,000" into `points[0:2000]` — the truncation the design
set out to avoid.

---

## 4. `current` — behind is not absent

The first implementation read the derived tree with the store's default digest
verification, which treats a tree that no longer matches the journal as
**absent**. During a walk that digest moves with **every keyframe**, so a build
finished, the next keyframe put it behind, and the manifest answered `404` for
the rest of the capture — while real geometry sat on disk. The fragment gallery
stayed empty until the session ended.

The Tower had already settled this on the status channel
(`tower/tower/results/world_builder.py:1058`, `_geometry_block`), and geometry
mirrors that decision rather than inventing a second policy. A build over the
first N keyframes is not wrong; it is a **correct answer to an older question**.
So it is served, with `current: false`. Hiding it discarded true information;
serving it unflagged would let a viewer read a partial world as the finished
one. **The flag is the whole difference.**

Three states stay distinct, and holding them apart is the constraint:

| On disk | Answer |
|---|---|
| no derived tree at all | `404` — absent |
| a tree behind the journal | `200`, `current: false` |
| a tree matching the journal | `200`, `current: true` |

`current` rides on **both** payloads. A client that holds a cached chunk and
never re-reads the manifest would otherwise have no way to know the geometry in
its hand is behind.

`current` is **additive**. An older decoder that ignores it reads exactly the
payload it read before, so per `tower/docs/contracts/CARTRIDGE-RESULTS.md` §12
this is not grounds for a contract bump, and the identifier did not move.

**What iOS does with it.** `WorldFragmentsModel` says so in one sentence — the
world is still building, so what is shown may be behind the newest frames. It
does not refuse to draw: the geometry is real.

---

## 5. Five rules that are not negotiable

1. **`pose_convention` is compared key-by-key and any mismatch refuses the
   render.** Inverting `T_world_camera` still produces a plausible-looking map,
   and that was a real shipped bug
   (`tower/tests/test_world_builder_engine.py:472`). iOS compares the five keys
   that change how a pose is interpreted — `pose_type`, `quaternion_order`,
   `handedness`, `camera_axes`, `translation_units` — and deliberately excludes
   `up_axis`, because the 2D top-down view does not depend on it.
2. **`translation: null` survives to the viewer**, which draws a break rather
   than a line through a gap. `null` means refused, **never zero**.
3. **`registered: false` forbids placing two segments in one space.** They
   share no coordinate frame and their scales disagree by up to ~87x on a real
   walk. A renderer that ignores this fabricates geometry.
4. **`points_sent` / `points_total` are always present**, so a sampled cloud
   can never be read as the whole one.
5. **No imagery.** `image_relpath` and every keyframe byte stay Tower-side.
   `retains_raw_imagery` remains permanently `true`; redaction is a process
   claim, not an outcome claim.

---

## 6. Absent is never zero

Every nullable field here means **"there is no such fact"**, and never "the
fact is zero". The distinction is the whole reason the transport exists at all.

| Field | `null` means | It does **not** mean |
|---|---|---|
| `bounds` | the segment resolved to nothing, so it has no extent | a zero-size box at the origin |
| `translation` / `rotation` | the pose was **refused**; the degeneracy says why | the camera was at the origin |
| `dominant_degeneracy` | nothing in this segment was refused | no reason is known |
| `transform_to_world` | this segment is not registered into any shared frame | an identity transform, which would silently place it at the reference origin |
| `scale.meters_per_unit` | no metric scale was ever established | 1.0 |

Two consequences that a renderer must honour:

- A segment with `resolution_state: "unresolved"` is **counted, never placed**.
  We know reconstruction failed; we do not know *where* it failed, and drawing
  it as a region would invent a location. On the real walk this was 32 of 51
  segments — dropping those rows would erase the difference between "we did not
  look" and "we looked and failed".
- A `resolved` segment with `bounds: null` is incoherent and is **refused**
  rather than framed by guess.

---

## 7. What the wire already carries and V1 does not draw

The contract is **fully 3D**: quaternion `rotation`, 3D `translation`, 3D
`points`, 3D `bounds`, and the full `pose_convention` including `camera_axes`
and `handedness`. The 2D top-down renderer is a V1 **presentation** choice, not
a property of the wire — `up_axis` is `"unknown"` today, so a 3D view would
have to guess which way is up. A future 3D renderer consumes the same payload
with **no wire change**.

Registration is a layer **above** frozen segments: geometry immutable,
placements mutable. `registered` flips to `true` and `transform_to_world`
carries a Sim3; segment-local geometry is unchanged, so every cached
`content_hash` stays valid across a registration pass, and loop closure moves
placements rather than points.

**That is safe only because `placement_hash` changes instead.** A client MUST
key its per-segment cache on `(content_hash, placement_hash)`. Keyed on
`content_hash` alone it will never refetch a re-placed segment, and will draw an
unplaced version of a segment the world knows how to place — a failure that
looks like nothing at all, because the fragment simply sits in the wrong place
forever. `geometry_revision` rolls up both hashes, so the change signal fires;
the risk is entirely in the per-chunk cache.

**Composition rule.** `transform_to_world` maps a segment's own frame into the
frame of its `reference_segment`. Segments sharing a `reference_segment` are in
one space and may be drawn together. Segments with **different** reference
segments are not, and must not be composited. `frame_revision` stamps the gauge
the Sim3 is expressed in; a coordinate stamped with one revision may not be
reinterpreted under another, and a mismatch is a refuse-to-draw condition rather
than something to guess past.

---

## 8. Addendum 2026-09-06 — the global solve (additive, no identifier bump)

The Tower now runs a global structure-from-motion solve over EVERY keyframe of
a session (`tower/tower/world_builder/global_solve.py`; design record
`tower/docs/superpowers/specs/2026-09-06-world-builder-global-solver-design.md`).
It reaches the wire **through the layers this contract already has** — per-
segment geometry plus placements — so the identifier is unchanged and an
older decoder reads exactly what it read before. What an older decoder sees
differently is only *how many* segments are `registered` and that they share
one `reference_segment`: on the 2026-09-06 walk, 29 of 34 instead of 2 of 17.

What the solve guarantees about rows it wrote:

- A `registered` segment's `transform_to_world` is exact, not fitted: every
  segment of one solve component is one reconstruction, `scale` is exactly
  `1.0`, and the transform is the segment's anchor pose in the component's
  frame relative to the reference segment's anchor. Segments in different
  components carry different `reference_segment`s and MUST NOT be composited
  (§7's rule, unchanged).
- A segment the solve could not place keeps whatever the local chain built
  and a `refused` placement whose `registration_refusal_reason` says so; a
  segment newer than the last solve is `unplaced` (the live "still building"
  case in §4).
- A pose row the solve did not support has `status: "unavailable"` and
  degeneracy `"unregistered"`, with `translation: null` (§6, unchanged).

Additive fields (an older decoder ignores them; per `CARTRIDGE-RESULTS.md`
§12 not grounds for a bump):

| Where | Field | Type | Meaning |
|---|---|---|---|
| manifest, per segment | `coverage` | `confident` \| `partial` \| `unresolved` \| null | How much to trust the segment's geometry. `unresolved` = keyframes exist, no geometry (the "observed but unresolved" state); `partial` = placed with thin support or few points; `confident` = placed, ≥ 60 3-D observations per keyframe (median), ≥ 50 points, at least half its keyframes posed. `null` = nothing has judged it (no solve ran). **Unseen space has no segment and is never represented.** |
| derived `poses.json` row | `observations` | int, optional | 3-D points this keyframe observes in the solve. Written only by the solve. Not yet carried on the chunk. |
| derived `points.json` row | `rgb` | `[r,g,b]`, optional | Colour sampled from the frame that first observed the point. Written only by the solve. Not yet carried on the chunk. |
| derived `manifest.json` | `global_solve` | object \| null | The solve's summary: `solver` (`glomap` \| `incremental`), `solved_at`, `horizon_keyframes`, `components[]` (`reference_segment`, `segments`, `keyframes`, `points`), counts, and `segments{index → {state, coverage, keyframes_posed, …}}`. Tower-side; not on the wire. |

A renderer that wants the progressive-coverage model draws `confident`
segments normally, `partial` muted, and `unresolved` as a count or a marker
without a position (they have no `bounds`). Nothing is invented for gaps.
