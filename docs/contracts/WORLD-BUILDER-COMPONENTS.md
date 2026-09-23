# World Builder components — the parts of a walk the Tower could place, and the areas it could not

**Living document.** Added 2026-09-23.

> **Status: PROPOSED 2026-09-23 — awaiting Mac review; nothing implemented.**
> No Tower code and no iOS code implements anything in this document. It is
> the contract half of manager decision 010, P3.1 ("contract first"): the Mac
> Validation Lead reviews it for Apple feasibility **before** any iOS or Tower
> product code is written. Every "OPEN" reference in the text is a question
> the drafter could not settle: M-numbers are addressed to the Mac (§10),
> T-numbers to the Tower lane, P3.2 (§11).

| | |
|---|---|
| Contract | `world_builder.components/2026-09-23` (names this agreement; not sent on the wire, §9) |
| Adds to | `GET /worlds` session rows (`WORLD-BUILDER-WORLDS.md` §2), `GET /worlds/{id}/render/revision` (WORLDS §4a), the status payload's `tracking` block (`tower/docs/contracts/CARTRIDGE-RESULTS.md` §10.1). All additive; no identifier moves |
| New routes | `GET /worlds/{w}/areas/{s}/{a}/…` (§5). New paths, which no existing app requests |
| Tower producer (proposed, P3.2 lane `world-builder/coherence-product-v1`) | a product `world_builder/coherence_gate.py` (ported from `coherence_exp/gate.py`, `rule="evidence"`), `global_solve.merge`, `results/world_builder_library.py`, `results/world_builder_render.py`, `results/world_builder.py::_tracking_block`, `engine.py::observe` (the `decision.lost` branch) |
| iOS consumer (proposed, Mac lane) | `WorldLibrary.swift`, `WorldRenderViewer.swift`, `WorldAssetTransport.swift`, `TowerWorldBuilderClient.swift`, `WorldModel.swift`, `WorldCanvasView.swift`, `WorldBuilderWorkspaceView.swift`, a new speech helper |
| Evidence | run `wb-coherence-run-2026-09-23`: `mailbox/to-manager/20260923-1813-candidate-architecture.md`, `experiments/P2-PX/PHONE-EXPERIENCE.md`, `baseline/review/V4a/V4a-REVIEW.md`, `experiments/P2-LOOKBACK/`, `experiments/P2-LM/lookback/` |

## 1. Why it exists

The target walk's damage (6e6d3fc3) was false geometry: a bathroom and a bed
block glued into the room by feature matches on the phone in the wearer's
hand, at a wrong tilt, scale and position. The approved fix stops gluing on
false evidence: hands, arms and the held phone are masked before feature
extraction, and an **evidence gate** attaches a group of cameras to the room
only on independent, consistent evidence (two independent verified links or a
closed triangle, **and** a matching metric level).

That leaves pieces of a walk that are real, internally sound, and not placed.
Today there is nowhere honest to show them: every existing page draws a
session's reconstruction in one frame, so an unplaced piece is either drawn
inside the room (an invented placement — the old bathroom box at 3× scale) or
not drawn at all. Drawn **on its own**, the target's bathroom reaches coverage
0.94 and agreement 0.82, against 0.62 and 0.52 inside the frozen room.

This document gives those pieces a name on the wire (`components[]`), a place
to be shown (the area routes, §5), and gives the live walk a way to avoid
making them in the first place (the look-back prompt, §6).

**What it deliberately does not do.** It never says where an area is relative
to the room. Between the bathroom and the bedroom, yaw and horizontal position
are not in the images (E3: no matcher finds a link through the doorway; R5:
every generic prior fails). The old placement was invented; the new screens
say so instead.

## 2. `components[]` — the pieces of one session's final solve

One entry per component of the session's **published** final solve, after the
evidence gate.

### 2.1 Fields

| Field | Type | Meaning |
|---|---|---|
| `id` | string | **16 lower-hex characters.** Opaque; compare for equality. Stable while the session's published solve is unchanged (rule 3). The alphabet is fixed because the id is a path segment of the area routes (§5) and of the app's scheme URLs, where it must need no encoding |
| `state` | `"placed"` \| `"unplaced"` | `placed`: the room — the component every other one was tested against. **Exactly one** entry is `placed`, and it is first. `unplaced`: the gate did not attach it; its position, direction and size relative to the room are unknown |
| `reason` | string \| null | `null` exactly when `placed`. Otherwise the first entry of `reasons` |
| `reasons` | array of string | Every §2.2 reason that applied, in §2.2's order. `[]` when `placed`; never empty when `unplaced`. A list, because the gate usually refuses a group for more than one reason and a single value mislabels it (V4a: the bathroom has one link, not none, **and** a ×4.27 level) |
| `shown_as` | `"room"` \| `"area"` \| `"none"` | §2.3 |
| `keyframes` | int | Published keyframes in this component (the solve's publish floor of 30 observations applied). What the phone calls **photos** |
| `keyframes_phone` | int \| null | Phone-tier keyframes of this component's appearance build, as the row's `appearance.keyframes_phone`. `null` when it has no appearance build |
| `capture_spans_s` | array of `[start, end]` | When this component was captured: seconds since the session's `started_at`, from the keyframes' `received_at` — **the Tower's receipt clock**, the only clock this system has (`CARTRIDGE-RESULTS.md` §4). Ascending, disjoint, at most 8 (rule 5). `[[86.7, 97.1], [106.4, 109.6]]` is the target's bathroom |
| `has_geometry` | bool | **Opening this component would show the wearer something.** `room`: identical to the row's `has_geometry`. `area`: the area render route (§5.1) would answer 200 now, decided by the same artifact checks as WORLDS §4a rule 4. `none`: always `false` |
| `photographic` | object \| null | WORLDS §2a's block, `{state, stage, detail}` with its seven-word vocabulary unchanged, for **this component's own** build. `room`: the room's build. `area`: the area's build. `null` for `shown_as: "none"`: nothing is built, by rule, and `shown_as` already says so. §3.4 says how the row's own `photographic` relates |

No other field. In particular **no footprint, extent, distance, height,
metric scale or scale factor** (rule 6) and **no name** (rule 7).

### 2.2 `reason` — which gate decision produced it

In precedence order; `reason` is the first that applies. The order follows the
evidence from the solve outward: a piece with no link at all could not be
placed whatever its level, so the link reasons come first.

| `reason` | The gate decision it reports (`coherence_exp/gate.py`, `rule="evidence"`) | Seen on |
|---|---|---|
| `solved-separately` | The solver itself returned this piece as a separate model, so the gate never tested it against the room. For such a piece this is the **only** reason: refusals inside another solver model are relative to that model, not to the room, and are not reported | target, arm A1: the solver separates the closet (V4a) |
| `no-verified-link` | In the room's solver model, but no verified image pair (≥ `min_link_inliers`, 15 inliers, COLMAP's floor) links it to the room — "not coupled", or `cross_links == 0` | no case checked by name in this run |
| `single-unconfirmed-link` | It has verified links to the room, but they are **one point of failure**: a single pair, or several pairs through one image whose other ends are not themselves linked (no closed triangle) — `redundant_links()` false with ≥ 1 link | target bathroom: one non-redundant link; 991e5a15 kf 13–25: one floor-level pair, 55° off in image-only roll |
| `scale-mismatch` | Its metric level (per-camera MoGe TRI ratio, median over ≥ 10 cameras on each side) differs from the room's by more than ×1.25 — including a part the gate split off its own group at an internal scale step | target bathroom: ×4.27 |
| `link-contradicted` | It has verified links to the room, but the solve **contradicts** them: each pair's own two-view rotation disagrees with the solve's relative rotation by more than the control-measured bound (`max_link_disagreement_deg`, the control's p90 of link-vs-solve disagreement, 16.8°), so they are not evidence that it was placed right. Reported when setting those links aside is what left the piece without redundant links; otherwise the link reason above applies | 6839fb8f kf 92–98: four UNCALIBRATED 15–18-inlier links, contradicted by 34–96° (precondition b, lane `9ffe043`) |

**Not reasons, deliberately.** `seed-unstable`: the seed-stability test is
dropped from the product gate (its 0.075 threshold was set by the target alone,
V4a; one seeded single-thread solve replaces the ensemble). `too-small-to-build`
(proposed by P2-PX): size is not a placement decision, and is `shown_as`'s
business (§2.3), so a small piece still says *why* it was not placed.

### 2.3 `shown_as` — the room, an area, or counted only

| `shown_as` | Which components | Drawn by |
|---|---|---|
| `room` | the `placed` one | WORLDS §4 (`/worlds/{w}/render`), exactly as today (§4) |
| `area` | `unplaced`, and **at least 30 keyframes or at least 5 s** of total span | its own surface and appearance, in its own frame, on its own routes (§5) |
| `none` | `unplaced`, below both floors | nothing. Counted in the footer (§8), never built, never drawable |

The floor exists because a small piece is worse than nothing: 2f447162's
20-keyframe, 3-second fragment built at coverage 0.54 with most orbit views
empty and a TRI level (9.7 units/m against the room's 0.44) no 3-second
stretch can support. **The floor is unvalidated** — it comes from that one case
(OPEN T2) — and it is the Tower's: the phone reads `shown_as` and never
recomputes it from `keyframes` or `capture_spans_s`.

### 2.4 Rules

1. **Absent is never zero.** `components: null` (or no key, from an older
   Tower) means **not computed** — every session built before this change, and
   any session with no final solve. It never means "no areas". When computed,
   the list is never empty: it holds at least the `placed` entry.
2. **Order.** `placed` first; then `area` entries by `keyframes` descending;
   then `none` entries by `keyframes` descending; ties by earliest span start.
   The phone numbers areas by their position among `area` entries (Area 1 is
   the first). A number is not a name and is not stable across a re-finish.
3. **`id` stability.** Unchanged while the published solve is unchanged. The
   Tower derives it from membership (proposed: the first 16 hex characters of
   sha256 over the session id and the sorted keyframe ids), so a rebuild that
   reproduces a component keeps its id; a client must not rely on that. A new
   final solve or a re-finish (§7) may change every id.
4. **Keyframes outside every component are not reported here.** Keyframes the
   solve did not publish (unposed, or under the 30-observation floor: 15 on the
   target) belong to no component. The row's `keyframes_accepted` minus the sum
   of `keyframes` is **not** a count of them to show: nothing on this contract
   says it is (OPEN T7).
5. **Spans.** Consecutive member keyframes at most 2.0 s apart are one span.
   More than 8 spans are merged across the shortest gaps until 8 remain
   (both values proposed, OPEN T4). A
   span is therefore an envelope — "captured between A and B" — not a claim of
   continuous presence. For a `recorded-capture` session replayed at other
   than real time the seconds are replay seconds (`frame_source` says which;
   OPEN T10).
6. **No metric figure.** `WORLD-BUILDER-SURFACE.md` §2 claim 7 ("scale is
   inherited, never invented") and every page's *Scale is unknown* caption
   forbid a metre figure. The MoGe level that refused a component stays in the
   Tower's gate report: it is a prior that R5 found reliable on only 2 of 4
   worlds, with the bathroom mirror a known confound (candidate report, risk 4).
   P2-PX's `extent_m_approx` is withdrawn for this reason (V4a).
7. **No names.** Region labels ("bathroom", "closet") are evaluation-only and
   never leave the Tower; nothing on this contract carries a name, a region or
   a label. Areas are numbered by the phone.
8. **Unknown values** (a Tower newer than the app). An unknown `reason` gets
   the generic unplaced copy. An unknown `shown_as` is treated as `none`. A
   list in which the phone finds no `placed` entry it understands is treated
   as `null` — the phone behaves as today.

## 3. Where `components` appears

### 3.1 On the `GET /worlds` session row — additive

One key, `components`: the §2 array, or `null` (WORLDS §2 carries the
cross-reference). The listing's contract identifier
**`world_builder.worlds/2026-09-10` does not move**, for WORLDS §2a's reason:
iOS reads rows key by key out of a `[String: Any]`, so a key it does not know
is a key it never looks at, but it **equality-tests `contract`** and a bump
empties Saved Worlds on every installed app until it is rebuilt. An app that
ignores `components` shows the room as today, and that is correct: the room it
draws **is** the placed component; it simply offers no areas.

The listing already carries unbounded arrays (`worlds`, `sessions`); each entry
here is a few hundred bytes, and the regression set's worst case is 21
components in one session (52ed8e0a, history arm A1h, GT's offline evaluation;
V4a). No cap is proposed (OPEN M9).

### 3.2 On `GET /worlds/{w}/render/revision` — additive

One key, `components`: the same array as the §2 row's for the session §4a
resolved, or `null`. **Not part of `revision`**: an area finishing, or its
build state moving, does not change the room page and must not swap it. A
phone with the room on screen uses it to keep the areas row current, comparing
the array by value (ids, `shown_as`, `has_geometry`, `photographic.state`).
§4a's other fields, rules and cadence are unchanged.

### 3.3 Not on the status channel

`components` is **not** carried by `world_builder.status`:

- the channel's payload is fixed-arity and under 8 KB, and a test asserts it
  holds no unbounded list (`CARTRIDGE-RESULTS.md` §8);
- components exist only after the final solve's gate runs, in finalization,
  when the listing and the revision route already answer; a live walk has
  none;
- what the world screen needs to know about unfinished areas already reaches
  it through `lifecycle.photographic` (§3.4).

OPEN M6: if the world screen wants "2 more areas" without an HTTP request, a
fixed-arity count (`{areas, short_stretches, short_stretch_keyframes}`) could be
added to `lifecycle`; it is not proposed.

### 3.4 The row's `photographic` covers the room **and** its areas

The row's `photographic` block (and the status channel's
`lifecycle.photographic`, computed by the same helper) answers WORLDS §2a's
question — *does this world still owe photographic work* — for the room and
its areas together:

- the room's word, when the room is `running`, `owed` or `unobservable`;
- otherwise, if any area's word is `running`, `owed` or `unobservable`, that
  word, with the area's `stage` and a `detail` that says *an area of this walk*
  (no name, no number);
- otherwise the room's word. An area's `failed` never reaches the row: the room
  is saved and shown, and the area's own entry says `failed`.

So the row keeps `state: finalizing` until every area is settled, and
`world_finish_pending.assess()` — which must agree with the row — sees owed
area work. A component's own `photographic` is always its own build's. Area
builds take 45–90 s each, typically 1–3 per walk, so "Improving" lasts 1–4 min
longer on a walk that has areas (OPEN M13). On a session with `components:
null` nothing changes. When implemented, WORLDS §2a and CARTRIDGE-RESULTS
`lifecycle.photographic` gain one sentence each; they are not edited by this
proposal.

## 4. The room page, for a session with components

Unchanged route, queries, ladder, revision and follower rules (WORLDS §4, §4a;
IOS §10). What is drawn is the **placed** component, as today's builds already
draw solve component 0; what changes is that the evidence gate, not the
solver's grouping alone, decides which cameras are in it. Two things change on
screen:

- the page's caption gains ` · N more areas shown separately` when at least
  one entry is `shown_as: "area"` (P2-PX: *Captured images on reconstructed
  geometry · 128 of 128 keyframes shown · 2 more areas shown separately*), and
  the native caption line the same;
- nothing else. Areas are **not** a rung of the room's ladder, never a query
  parameter of §4, and never drawn on §4's pages.

The sparse page and the geometry routes are unchanged: an unplaced component
reaches them as segments with their own `reference_segment`, exactly as a
second solve component does today, which `WORLD-BUILDER-GEOMETRY.md` §8 already
forbids compositing. The sparse page's "segments in their own frames" is the
precedent the area routes follow (V4a).

## 5. Area routes

**Their own routes, not a query parameter.** P2-PX first proposed
`/worlds/{id}/render?session_id=…&component=<id>` and an optional path segment
on the appearance routes. V4a caught that this is not additive for the app: the
scheme handler (`WorldAssetRequest.parse`) answers exactly a whitelist of paths
and queries and 404s everything else, and the navigation policy loads exactly
one page URL. A new path family is something the handler can whitelist
precisely, with no change to what it answers for the room.

All `GET`, same origin, containment and rules as WORLDS §4 (`contained_world_id`;
`session_id` must be one of the world's sessions), plus: `area_id` must be
16 lower-hex **and** the id of a current component of that session with
`shown_as: "area"`. The room's id and a `none` entry's id are never served here.

### 5.1 `GET /worlds/{world_id}/areas/{session_id}/{area_id}/render`

| Query | Type | Meaning |
|---|---|---|
| `representation` | `auto` \| `appearance` \| `surface`, optional | WORLDS §4's ladder, shortened: an area is built as a surface and an appearance only. `auto` (default) serves the best rung the area has. A named rung the area does not have is 404, as in §4. `sparse` and `dense` are **404** ("an area has no sparse or dense rung"); any other value is **422** |
| `max_points` | int 1…200000, optional | as §4 for the surface rung (it selects the level-of-detail rung by page budget) |
| `transport` | `app` \| `tower`, optional | as §4 |
| `viewer` | string, optional | **accepted and ignored**, never a 422. `auto` offers the appearance rung unconditionally: the only clients that can name this route postdate the `glasses-world:` handler (IOS §10, 2026-09-17), so the old-app reason for the declaration cannot arise (OPEN M12) |

`session_id` is in the path, never a query. There is no `view=diagnostics`.

**200** `text/html`, `Cache-Control: no-store`, the same page programs as the
room (§5.4). CSP header and `<meta http-equiv>` exactly as §4 rule 6 for the
corresponding room page: the appearance page `… connect-src glasses-world:` (or
`'self'` under `transport=tower`), the surface page `default-src 'none';
script-src 'unsafe-inline'; style-src 'unsafe-inline'`. Within the first 4096
characters, as §4a:

    <meta name="wb-representation" content="appearance">
    <meta name="wb-revision" content="<session_id>/area:<area_id>/appearance:1@<epoch>">
    <meta name="wb-area" content="<area_id>">

`wb-area` marks the page as an area page: the native caption switches on it.

**404**, with a `detail` the phone shows verbatim, for exactly: no world root
configured; no such world; no such session in that world;
**`no such area in this session`** (a malformed id, an id from an earlier solve,
the room's id, a `none` entry's id — terminal for that id);
**`this session has no areas`** (`components` is `null`: every older world, §7);
**`this area has not been built yet`** (transient: its build is owed or running
and nothing is drawable); **`this area could not be built`** (terminal: its build
failed or was declined and nothing is drawable). **422** as in the table.

### 5.2 `GET /worlds/{world_id}/areas/{session_id}/{area_id}/render/revision`

No query. **200** `{"session_id": str, "area_id": str, "representation":
"appearance"|"surface", "revision": str, "live": bool, "appearance": {…}}`,
`Cache-Control: no-store`. **404** exactly when §5.1 would.

- `appearance` is WORLDS §4a's object, unchanged in shape and meaning, for the
  area's artifact.
- `live` is `true` while this area's surface or appearance stage is running
  under a live process and has not yet published (§4a's rule, per area).
- `revision` is `<session_id>/area:<area_id>/surface:<built_at>` or
  `<session_id>/area:<area_id>/appearance:1@<epoch>`: opaque, compared for
  equality, and prefixed by the session id so IOS §10's "same walk" test reads
  it unchanged. It can never equal a room revision.
- WORLDS §4a rules 1, 3, 4, 5, 6 and 7 apply unchanged to the area's artifacts.
- **`no such area in this session` is terminal for that id** (the session was
  re-finished or re-solved and the membership changed): keep the picture, stop
  following, and offer the way back to the room, where the areas row is
  current. `this area has not been built yet` is transient: ask again at the
  next interval, as §4a rule 5.

### 5.3 `GET /worlds/{world_id}/areas/{session_id}/{area_id}/appearance/…`

| route | body |
|---|---|
| `…/appearance/manifest` | the area's appearance manifest, plus `currency` |
| `…/appearance/chunk/{digest}` | a keyframe bundle |
| `…/appearance/proxy/{digest}` | the area's proxy mesh (`WBSURF01`) |

`WORLD-BUILDER-APPEARANCE.md` §9 applies **in every respect**, to the area's own
artifact: the redaction label and keyframe-set identity re-checked against the
**session's** keyframe set on every request, `images_purged` refused, a digest
32 lower-hex and named by the current manifest or one superseded less than
120 s ago, `Cache-Control: no-store`, `Pragma: no-cache`,
`X-Content-Type-Options: nosniff`, `X-World-Redaction`, `X-World-Imagery`,
compression, and no `ETag` or `Last-Modified`. The manifest format stays
`wb-appearance-keyframes/1`; the area's appearance and surface manifests each
gain one additive key, `area: {id, levelled}`.

The area's artifacts live beside the session's (proposed:
`<world>/areas/<session>/<area_id>/{surface,appearance}/`, with the same files,
atomic publish and prune rules as SURFACE §4 and APPEARANCE §4). Tower-internal;
OPEN T9.

### 5.4 What the area page draws

The **same page programs** as the room — the appearance page, or the surface
page — fed the area's artifacts. No new renderer. What differs:

- **Its four addresses.** The appearance page fetches, relative to `CONFIG.base`,
  exactly `/worlds/{w}/areas/{s}/{a}/appearance/manifest`, `…/appearance/chunk/{digest}`,
  `…/appearance/proxy/{digest}` and `/worlds/{w}/areas/{s}/{a}/render/revision`
  (no query) — the §5.5 whitelist, and nothing else.

- **Its own frame, never the room's.** The area's cameras and surface are the
  component's own rigid reconstruction, rotated so that the vertical estimated
  from the area's own images is up (**levelled**), in the area's own unit. No
  route, page or field carries a transform between an area and the room, the
  room's geometry, or the room's cameras. An area is never composited with the
  room or with another area.
- **Levelling may fail.** When the vertical cannot be estimated (support below
  the Tower's floor), the area is drawn in the solve's own orientation,
  `area.levelled` is `false`, and the caption adds *Its vertical could not be
  estimated, so it may look tilted.* The target's Area 2 was levelled on low
  normal support (0.058) and rendered with a level floor (OPEN T3).
- **Its own walk.** The recorded path, the opening pose, *Best view*, the
  orientation ring and the tube are the area's keyframes only. The *Face the
  room* button reads *Face the area* (OPEN M11).
- **Captions** (P2-PX's wording, in the existing style). Header: *Area k of N —
  not placed in the room*, numbered by §2.4 rule 2. Appearance rung: *The
  camera's own images, faces redacted, from 1:26 to 1:49 of this walk. This
  area could not be placed relative to the room, so it is shown on its own: its
  position, direction and size are not comparable with the room's. Not to
  scale.* Surface rung: the same with *Surfaces the Tower reconstructed from the
  walk* in place of the first clause. The page's *Scale is unknown, so distances
  are relative* section stays. The native caption line (IOS §10) says the same
  in one line when the page carries `wb-area`.
- **Scale.** The area's surface manifest records `scale.state: "unknown"`
  against the world's unit: an area inherits no scale from the room, so it
  claims none (SURFACE §2 claim 7).

### 5.5 The scheme-handler whitelist — **requires Mac validation**

In the style of IOS §10. **Not implemented; the handler answers none of these
today.** An area viewer is opened for exactly one triple (`<w>`, `<s>`, `<a>`),
taken from the listing row, with `<a>` exactly 16 lower-hex (checked like
`isDigest`). Its handler answers exactly:

| Path under `glasses-world://tower` | Query | Answered by |
|---|---|---|
| `/worlds/<w>/areas/<s>/<a>/render` | none | the page string the app already fetched, from memory |
| `/worlds/<w>/areas/<s>/<a>/render/revision` | none | proxied to the same Tower path with no query (the Tower ignores `viewer` here; adding it is harmless) |
| `/worlds/<w>/areas/<s>/<a>/appearance/manifest` | none | proxied |
| `/worlds/<w>/areas/<s>/<a>/appearance/chunk/<digest>` | none | proxied; `<digest>` 32 lower-hex |
| `/worlds/<w>/areas/<s>/<a>/appearance/proxy/<digest>` | none | proxied; `<digest>` 32 lower-hex |

- **Everything else is a 404 from the handler** and never reaches the Tower —
  including **every room route** (`/worlds/<w>/render`, `/worlds/<w>/appearance/…`)
  and every other area or session. Conversely a **room** viewer's handler
  answers **no** `/areas/` path. Each viewer's whitelist is exactly its own
  family.
- `<s>` is fixed when the viewer opens. Unlike the room page, the area page never
  teaches the handler its session through `wb-revision`.
- The page itself is fetched natively over HTTP, with whatever query
  `WorldRenderClient` sends (§5.1), and loaded into the web view at the query-free
  scheme URL — exactly as the room page is today.
- **Navigation policy:** the initial load of exactly the area page URL, and the
  page putting itself back (the same counted reload the room page has).
- **Memory and authorisation** per (`<s>`, `<a>`): an area's manifest answered
  200 within the last 20 s authorises only that area's bundles; a copy made
  under one area's manifest is never answered for another area or for the room.
  The copy is dropped under the same conditions as the room's (IOS §10).

### 5.6 Rules

1. Composed on request, cached nowhere, like §4.
2. An area page never fetches, draws or names the room, and the room page never
   fetches or draws an area.
3. An id that stops being a current area is `no such area in this session`,
   never another area's content.
4. A world with `images_purged`, or a session whose redaction label no longer
   matches, refuses the area imagery exactly as the room's (APPEARANCE §9).

## 6. `tracking.recovery` and the look-back prompt

Walks almost never look back unprompted: over 7 replayed worlds only 14 of 107
unbridged tracking losses relocalized within 10 s, and the target's doorway
never did in 47 s. Views that **are** seen again relocalize at once. So when
tracking is lost and does not come back by itself, the phone asks the wearer
to look back, once. This section is what the phone needs to say it exactly once
and to show what happened.

### 6.1 The relocalizer, in one paragraph

On a `tracking_lost` with no episode open, the builder opens a **recovery
episode** and matches incoming frames (at `acceptance.scan_hz`) against the last
`acceptance.reference_keyframes` keyframes accepted before the loss.
`searching` → `recovered` if a match is accepted (§6.3) before
`limiter.prompt_after_s` — silently. Otherwise, at `prompt_after_s`, → `prompting`
if prompts are enabled **and** the limiter allows (§6.4); if not, the episode
stays `searching` and the withheld prompt is counted. `searching` or
`prompting` → `recovered` on an acceptance before `limiter.timeout_s` from the
loss, else → `timed_out`. **A `tracking_lost` inside an open episode joins it**:
no new episode, no second prompt, the reference set unchanged. After it
resolves, the block keeps the last episode's outcome until the next loss opens
a new one. This is P2-LOOKBACK's replayed state machine (`scripts/prompts.py`).

### 6.2 Fields

`tracking.recovery` is an object, or **`null`** when the session's journal
records no relocalizer — every session today, any Tower running without it,
and every older world. `null` means *not recorded*, never *no losses*.

| Field | Type | Meaning |
|---|---|---|
| `state` | `none` \| `searching` \| `prompting` \| `recovered` \| `timed_out` | The current or last episode. `none`: no episode yet this session |
| `episode` | int | 1-based count of episodes this session; `0` with `none` |
| `lost_at` | number \| null | Tower clock: the journaled `tracking_lost` that opened the current or last episode |
| `resolved_at` | number \| null | Tower clock: when it became `recovered` or `timed_out`; `null` while open |
| `recovered_by` | `triangle` \| `strong-link` \| null | Which acceptance (§6.3); `null` unless `recovered` |
| `prompts_enabled` | bool | `false` in the physical test's prompt-off arm or when the owner turns prompts off. The relocalizer still runs and records; `state` is then never `prompting` |
| `prompt` | object \| null | The **latest** prompt issued this session; `null` when none has been |
| `prompt.id` | int | 1-based, **strictly increasing within the session, never reused**. The phone's speak-once key (§6.5) |
| `prompt.episode` | int | The episode it was issued in |
| `prompt.kind` | `"look-back"` | The only kind. The phone owns the words (§8) |
| `prompt.issued_at` | number | Tower clock |
| `prompt.speak_until` | number | Tower clock: `issued_at + limiter.speak_window_s`. After it the prompt is stale and is never spoken |
| `counts` | object | This session: `{episodes, recovered, recovered_after_prompt, timed_out, prompts, withheld_by_limiter, withheld_disabled}`, all int |
| `limiter` | object | Read-only (§6.4): `{max_prompts, window_s, mechanism, cooldown_s, prompt_after_s, timeout_s, speak_window_s}` |
| `acceptance` | object | Read-only (§6.3): `{matcher, reference_keyframes, scan_hz, triangle: {min_links, min_link_inliers, max_closure_deg}, strong_link: {min_inliers}}` |

- **Every time is the Tower's clock** (`CARTRIDGE-RESULTS.md` §4). The phone
  compares them only with each other and with the envelope's `tower_sent_at`,
  never with its own clock.
- `limiter` and `acceptance` are what **the builder ran with**, read from its own
  journal record at session start, never from the web process's configuration
  (the two processes can be configured differently).
- The block is fixed-arity. It holds **no volatile field** (no "seconds since"),
  so an unchanged episode never looks like a change; every field is part of the
  payload's `revision`, so a new prompt changes it and is pushed at the next
  0.5 s poll.
- The status contract identifier `world_builder.status/2026-09-10` does not move:
  iOS decodes the payload as dictionaries and ignores unknown keys.

### 6.3 Acceptance — what `recovered` means

`recovered` means the builder accepted a relocalization under **exactly one**
of two rules, and nothing weaker:

| `recovered_by` | Rule (P2-LOOKBACK name) | Wrong, replayed on 7 worlds |
|---|---|---|
| `triangle` | In one scanned frame, ≥ 2 verified links, each ≥ 50 RANSAC inliers, to two reference keyframes that are themselves verified-linked, and the triangle (reference 1, reference 2, frame) closes within **8°** (`tri2_50`; 8° is the control-derived consensus bound) | 4 of 63 |
| `strong-link` | One verified link of ≥ 100 inliers (`single_100`) | 0 of 38 (0 of 47 counted without de-duplication) |

A single link of any size is wrong 20 of 99 times, which is why it is not
`recovered`. The thresholds were measured with SIFT; `acceptance.matcher`
reports the matcher, and another matcher needs its own measurement (OPEN T6).

**`recovered` does not mean "placed".** It means a verified revisit edge was
written for the final solve. Whether the part walked after the loss joins the
room is decided at finalization by the evidence gate (§2); a recovered episode
can still end as an area. Nor is it the tracker re-acquiring: after a loss the
tracker starts a new segment at once, so `tracking.state: "good"` beside
`recovery.state: "searching"` is normal.

### 6.4 The rate limiter — at most 2 prompts a minute, by design

Replayed under the `tri2_50` rule, the prompt fired 1.2–5.1 times a minute per
world across the settings tried (the candidate report summarises 1.7–4;
`experiments/P2-LOOKBACK/out/prompts_tri2_50_*.json`); even the gentlest
setting (`prompt_after_s` 5, `timeout_s` 20) gave 2.1–2.6 on the control and
2.2–2.5 on the bedroom walks. The manager requires ≤ 2 **by design**. Two
layers:

1. **The cap, invariant.** The builder never issues a prompt when `max_prompts`
   (2) prompts were issued at or after `t − window_s` (60 s). Any closed 60-second
   window therefore holds at most 2 prompts, whatever the mechanism below does.
   A prompt the cap refuses is counted in `counts.withheld_by_limiter` and never
   issued later: a late "look back" is wrong advice.
2. **The mechanism, chosen in P3.2** so that the cap rarely binds: a cooldown,
   hysteresis, or suppression during continuous motion — one of them, measured
   on replay to ≤ 2/min **before** the cap. `mechanism` names it; `cooldown_s` is
   its parameter when it is a cooldown, else `null`. Values of `prompt_after_s`,
   `timeout_s`, `cooldown_s` and `speak_window_s` are P3.2's (OPEN T5); the replay
   tried `prompt_after_s` 1–5 s and `timeout_s` 10–20 s.

Because the parameters and the counts are on the wire, the guarantee is visible:
`counts.prompts` over a session can be checked against `max_prompts` per
`window_s`. The phone does not rate-limit or delay a prompt; it may refuse a
third within 60 s of its own clock as a defence in depth, never as the design.

### 6.5 Speaking each prompt exactly once — the phone's rule

Speak `prompt` if and only if **all** hold:

1. the session is **live and is the one this phone is streaming to**: IOS §9's
   `WorldSessionBinding` is `.bound` (camera bracket open; the Tower says
   `receiving`, `ended_at: null`, `frame_source: "live-capture"`), whether the
   subscription is pinned or following. Never under `.none`, `.awaiting` or
   `.foreign`, so never for a stopped, finalizing or historical session;
2. `recovery.state == "prompting"`, `prompt.episode == recovery.episode`, and
   `prompt.kind` is one the phone knows (an unknown kind is never spoken);
3. `prompt.id` is **greater than** `lastSpoken[(world_id, session_id)]`;
4. the envelope's `tower_sent_at ≤ prompt.speak_until`.

Then set `lastSpoken` to `prompt.id` **before** speech starts. Heartbeats
(`revision_changed: false`) and coalesced snapshots carry the same id, so they
never speak twice. `lastSpoken` only ever rises: an id at or below it is never
spoken.

### 6.6 Reconnect, relaunch, stale ids

- **Reconnect** is re-subscribe: a complete snapshot with `seq: 1`
  (`CARTRIDGE-RESULTS.md` §6). §6.5 applies unchanged, and `lastSpoken` survives,
  because it is keyed by world and session, not by subscription. A prompt issued
  while the socket was down is spoken only if it is still before `speak_until`.
- **Relaunch.** A `lastSpoken` held only in memory is lost; rule 4 then bounds
  a repeat to a relaunch within `speak_window_s` of the issue. Persisting it per
  session (one integer, no imagery) makes it exact (OPEN M4).
- **Another session** is another key; nothing carries over.
- **Coalescing** cannot hide a prompt during a connected walk: prompts are
  spaced by the mechanism and by the cap, and the channel polls every 0.5 s.
- `recovery: null`, `prompts_enabled: false`, or any state but `prompting`:
  nothing is spoken.

### 6.7 Producer notes (Tower-internal, for P3.2)

The builder journals `relocalizer_started {acceptance, limiter, prompts_enabled}`
at session start, and `recovery_prompted {prompt_id, episode}`,
`recovery_withheld {episode, why: limiter|disabled}`,
`recovery_accepted {episode, by, links}` and `recovery_timed_out {episode}`;
`tracking_lost` itself is unchanged and opens an episode. `_tracking_block`
summarises them into the fixed-arity block, cached like the rest of the journal
summary. The prompt event is written at issue, so it reaches the phone within
one poll (0.5 s) plus the send. Transitions are evaluated as frames arrive; a
stalled stream leaves an episode open until the next frame (OPEN T8). The live
relocalizer costs 0.3–0.8 of one core, only while an episode is open.

## 7. Saved-world compatibility

1. **Every world on every Tower today has no components.** Its rows and §4a
   bodies carry `components: null`, and the phone shows exactly today's
   screens: no areas row, no footer, no caption suffix. The room route keeps
   serving the whole session's rung ladder for it, byte for byte as today.
2. `null` is **not computed**, never "zero areas". A phone must never render
   "0 more areas" or an empty footer.
3. `tracking.recovery` is `null` on every older session; nothing is spoken.
4. **The re-finish path — referenced, specified by P3.2.** One documented
   command rebuilds a saved session with the product pipeline (masks, the
   seeded solve, depth before publishing, the evidence gate, components, room
   and area builds), from authoritative data (the stored, redacted keyframes;
   refused for `images_purged`). Proposed name:
   `.venv\Scripts\python.exe scripts/world_refinish.py --root <root> --world <id> [--session <sid>]`.
   An owner runs it; nothing else does. Its result: `components` appears, and
   the room may change (the gate may move pieces out of it), so the room's
   revision changes and an open viewer offers *A newer reconstruction is ready*
   under IOS §10's existing rules.
5. **The finisher never computes components.** `world_finish_pending.py` and
   the Tower's idle and start-up finishing treat `components: null` as owing
   **nothing**: no world is rebuilt into components on Tower start or when idle.
   Owed area work exists only for a session whose components record already
   names an area whose build is unsettled, and it is finished like any owed
   stage (§3.4).
6. The listing's and the status channel's identifiers, and every existing
   route, are unchanged.

## 8. What the phone shows

**Below the room** (a session with `components` not `null`):

- **The areas row**, when N ≥ 1 entries are `shown_as: "area"`: *N more areas —
  captured on this walk but not placed in this room.* One entry per area, in
  list order: *Area k · 0:29–0:33 and 1:49–2:12 · 66 photos* — the spans from
  `capture_spans_s` as m:ss of this walk, the photos from `keyframes`. An entry
  with `has_geometry: true` opens the area viewer (§5). One whose `photographic`
  is `running` or `owed` and has nothing to draw yet says *Improving*. One with
  nothing drawable and a settled word says *could not be built* and does not
  open.
- **The footer**, when K ≥ 1 entries are `shown_as: "none"`: *K short stretches
  (P photos) could not be placed or shown*, P the sum of their `keyframes`.
- **The room caption** gains *· N more areas shown separately*.
- With N = 0 and K = 0 nothing new appears: a walk the gate kept whole looks
  exactly as today.

**The area viewer:** the header and captions of §5.4, and *Back to the room*.
No arrow toward the room, no distance, no size, no name, no position — none of
them is known.

**The spoken prompt** (`prompt.kind: "look-back"`): the phone owns the words.
Proposed: *Look back at what you were just looking at.* (OPEN M11)

**`tracking.recovery` on the world screen** (proposed copy, OPEN M11):
`searching` or `prompting` — *Finding where you are…*; `recovered` — *Linked
back to what you saw before* (not "placed", §6.3); `timed_out` — *Could not link
back; this part may be shown as a separate area.*

## 9. Versioning — why no identifier moves

- `world_builder.worlds/2026-09-10`, `world_builder.status/2026-09-10`,
  `wb-appearance-keyframes/1` and `wb-surface-mesh/1` are unchanged. Every field
  here is an additive key in a payload whose own identifier governs it, and
  every iOS reader of those payloads ignores keys it does not know; the area
  routes are new paths an older app never requests.
- `world_builder.components/2026-09-23` names this document's agreement. It is
  not on the wire in this proposal. An incompatible change later adds new keys
  or paths rather than repurposing these (OPEN M10: whether the Mac wants the id
  on the wire, e.g. on the area revision body).
- Status of this document moves from PROPOSED to implemented only when the Mac
  has reviewed it and both halves exist.

## 10. OPEN questions for the Mac Validation Lead

- **M1 — handler shape.** One `WKURLSchemeHandler` per viewer bound to its triple,
  or one handler with two whitelists? Can `WorldAssetRequest.parse` take the area
  family (§5.5) without weakening what it answers for the room, e.g. as a
  separate case set? Is the area revision route proxied with no query
  acceptable?
- **M2 — navigation and memory.** Is the area viewer pushed over the room viewer
  (two `WKWebView`s and two WebGL contexts alive; up to 22.6 + 10.1 + 9.4 MB of
  imagery on the target against the 64 MB `WorldAssetMemory` cap) or does it
  replace it? `.onDisappear` → `tearDown()` drops the room's copy when the room
  is covered, so *Back to the room* re-downloads ~22 MB. The Tower requires
  neither; which is feasible and safe on the target devices?
- **M3 — speech.** Can the app speak (e.g. `AVSpeechSynthesizer`) to the glasses'
  audio route during a DAT streaming session, with the phone locked in a pocket?
  Does it need the `audio` background mode, and does speech interrupt or duck
  anything the session depends on? What is the speech latency?
- **M4 — `lastSpoken` persistence.** Persist it per session, or accept one
  repeat on a relaunch within `speak_window_s`?
- **M5 — where `recovery` is read.** It sits on the payload's `tracking` block,
  which IOS §2.4 lists as not consumed. Keep it there (a third §2.4 exception) or
  project it into `world_snapshot`?
- **M6 — a fixed-arity component count on the status channel** for the world
  screen (§3.3)? Not proposed unless asked.
- **M7 — numbering.** Areas numbered by list position, ids possibly changing on
  a re-finish: acceptable for the areas row and the viewer's header?
- **M8 — unknown enum values** (§2.4 rule 8): feasible in the Swift decoders,
  e.g. raw strings with an `.unknown` case?
- **M9 — row size.** Every component is listed (worst measured: 21). Does the
  Mac want a cap?
- **M10 — contract id on the wire** (§9)?
- **M11 — copy.** The prompt sentence, the recovery display copy, *Face the
  area*, and the time format for spans.
- **M12 — `viewer` ignored on area routes.** Acceptable? Should
  `WorldRenderClient` still send it?
- **M13 — "Improving" lasts 1–4 min longer** on a walk with areas (§3.4).
  Acceptable, or should area work stay out of the row's word?
- **M14 — which builds can hear a prompt.** IOS §9 ("Release builds") says
  Release has no capture path, so `isStreamingToTower` is always `false` there.
  If that still holds, the prompt (§6.5 condition 1) can be exercised only on a
  DEBUG build, which is what the physical A/B test would run. Is that right?

## 11. OPEN for the Tower lane (P3.2)

- **T1** Settled by preconditions (a) and (b) of decision 010 (note
  `20260923-1905-preconditions-ab.md`): the product gate counts only links the
  solve honours (`max_link_disagreement_deg`), which makes `link-contradicted`
  reachable. `require_group_scale` and `exclude_uncalibrated_links` were
  measured and rejected, so no `scale-unmeasurable` reason exists. The producer
  must report `link-contradicted` exactly when the honoured-links filter is what
  removed the piece's redundancy (compare the decision with and without it).
- **T2** The 30-keyframe / 5-second area floor is unvalidated (one case).
- **T3** The levelling support floor below which `levelled` is `false`.
- **T4** The 2.0 s span join and the 8-span cap are proposals.
- **T5** The limiter mechanism and values, measured on replay to ≤ 2/min
  before the cap binds.
- **T6** The relocalizer's matcher: SIFT today; decision 2 re-tests ALIKED +
  LightGlue. The 50/100-inlier thresholds are SIFT's.
- **T7** Whether to report keyframes in no component (the solve could not pose
  them) as a count.
- **T8** Episode timeouts when frames stop arriving.
- **T9** Where the components record and the area artifacts live
  (proposed: `solve/<session>/components.json`, written with the published
  solve; `<world>/areas/<session>/<area_id>/…`; per-area stage records in
  `session.json` `stages`).
- **T10** The span timebase for recorded captures replayed at other than real
  time.
- **T11** Whether a re-finish keeps the previous solve and builds for rollback
  (deletion requires human approval).
