# Object Memory observations — the Tower↔client boundary

**Living document.** It describes the boundary as it exists now.

| | |
|---|---|
| Contract | `object_memory.observations/2026-08-26` — **unchanged**, see §8 |
| Also offered | `cartridge_session.control/2026-08-27` (§9), `object_memory.imagery/2026-08-27` (§10) |
| Transport | HTTP. The two observation endpoints are `GET` and stay read-only (§2); session control is `POST` (§9) |
| Tower adapter | `tower/tower/results/object_memory.py` |
| Tower routes | `tower/tower/routes/observations.py` |
| Store | `tower/tower/object_memory/store.py` |
| CLI (same questions, plus deletion) | `tower/scripts/object_query.py` |
| Tests | `tower/tests/test_object_memory_transport.py` |
| iOS consumer | `ios/Glasses/Workspaces/ObjectMemory/` (4 files) |
| iOS tests | `ios/GlassesTests/ObjectMemoryTests.swift` |

**Status:** Tower half implemented and exercised against the real 55-observation
corpus in `tower/data/object_memory`. The iOS half is **written and
uncompiled** — there is no Swift toolchain on the machine it was authored on;
see `docs/agent-handoffs/OBJECT-MEMORY-MAC-HANDOFF.md`.

**One correction the client found by reading the route rather than this
document.** §4.4 below lists `confidence` as `low | medium | high`.
`tower/confidence.py` defines a fourth value, `unknown`, and
`Confidence.from_score(None)` returns it — so a record written before scores
were tracked carries it, and a decoder built from the table alone refuses a real
record. The Swift decoder accepts all four. The table is corrected in place.

The contract identifier is **opaque and dated**. It is compared for equality and
nothing else — never parsed, never split on `/`, never ordered. A newer date is
not "greater than" an older one; it is a different contract.

---

## 1. What this cartridge actually knows, and what it does not

This section is the contract, not a preamble to it. A client that implements the
field tables in §4 while ignoring this section will make claims the sensor cannot
support.

**A record means a category was visible once.** Not that the object is still
there. Not that it will be there later. Every observation carries
`claim: "category-was-visible-once"` so this is a value a decoder can switch on
rather than a sentence in a document.

**A record is about a category, not an instance.** `laptop` means *a* laptop was
in view. It is not "your laptop", and two records of `laptop` are not evidence
about the same object. Every observation carries
`identity: "category-not-instance"`. Nothing here re-identifies an object across
sightings, and a client must not present it as if it does.

*A correction to this paragraph, 2026-08-27.* It used to say persistent identity
was "forbidden outright by the cartridge brief (`07-PLATFORM-CONSTRAINTS.md`
Core Principle 3)". That citation is wrong twice over. Core Principle 3 is
"Absence of Observation ≠ Observation of Absence" and is about something else;
and Limitation 6 of that same document, *Object Identity vs. Object Class*,
explicitly lists "visual embeddings / re-identification" and
"confidence-scored identity association rather than binary identity claims" as
**mitigations**, names "usefully probable identity association for
retrieval-style queries" as what they buy, and says "persistent identity should
be represented probabilistically unless strongly established". The module brief
is equally clear and equally not a prohibition: *"Do not claim unique-object
identity unless the implementation actually supports it."*

So the rule is **not** that identity may never be claimed. It is that this
implementation does not support it, and the evidence says it should not try yet:
the best frozen embeddings get 26.4% Recall@1 on small mass-produced objects,
tracking IDF1 collapses from ~100% to ~40% from identical same-class distractors
alone, and zero-shot egocentric object re-ID tops out at 45.3% mAP. See
`docs/superpowers/research/2026-08-27-object-memory-vision-model-landscape.md`
§7. `category-not-instance` is a measured position, not an inherited
prohibition — which matters, because a measured position can be revisited when
the measurement changes, and it will need a **new contract identifier** when it
is.

**Absence of a record is not evidence of absence of the object.** Every payload
carries `absence_means: "not-observed-by-this-cartridge"`. A missing record means
the camera was not pointed at it, or the detector scored it below threshold, or
its class is not one this cartridge ever writes, or its retention window closed.
None of those is "the object is not there". This is why
`GET /object-memory/last-seen/{object_class}` answers **200 with
`observed: false`** and never 404 (§3.2) — a 404 reads as "there is no laptop",
which is a claim about the world.

**"Where" is a frame reference, not a place.** Every observation carries
`spatial_ref: null`. The field is reserved, never populated, and actively nulled
on read: nothing in this cartridge knows where anything is in a room. The `where`
object answers the question as a pointer back into a recording — which capture,
which frame sequence number, which camera — and says so in
`where.kind == "frame-reference"`.

`spatial_ref` travels as an **explicit null** rather than being omitted, at both
the envelope and the record level. An absent key looks like version skew and
invites a client to go looking for the value elsewhere; a null is an answer.

**The only position offered is inside a frame.** `bounding_box_normalized` is
nested **inside** `where`, not beside it. At the top level a box reads as a
position; under `kind: "frame-reference"` it reads as what it is — where in a
picture, not where in a room. A client must not project it into any world frame.

---

## 2. Read-only, and what that protects

Every endpoint on the observations router is `GET` — the two in §3 and the three
in §10. There is no write, no delete, and no path by which one can be added
without a test failing: `test_the_router_exposes_only_reads` asserts the
router's method set is exactly `{"GET"}`.

The session control surface in §9 is a **different router**, deliberately. It
mutates a session; it cannot touch a store. Keeping the two apart is what lets
the rule above stay absolute rather than becoming "read-only except for the bits
that are not".

`ObservationStore` does have `purge()` and `prune_expired()`. **Neither is on the
wire, and neither may go on it.** An unauthenticated HTTP endpoint on a LAN-local
origin that erases a wearer's memory is not a feature. `06-PRIVACY-DATA.md`
requires real deletion, and real deletion stays with
`scripts/object_query.py --purge-all`, where a human types it against a store
they can name.

Two tests hold this:

- `test_the_router_exposes_only_reads` — the router's method set is exactly
  `{"GET"}`.
- `test_no_wire_module_can_reach_purge_or_prune` — an AST scan of both wire
  modules for a call to `purge` or `prune_expired`. An AST scan rather than a
  grep, so a comment explaining why they are absent does not trip the rule.

**What deletion would not reach anyway.** A record's `session_id` + `frame_seq`
resolves into `data/captures/`, where the JPEG it was derived from is still
sitting. Object Memory's retention governs *this store and nothing else*.
Since 2026-08-29 a record may also own one small filtered crop under
`<root>/keyframes/`, and `--purge-all` and `prune_expired` both delete it
with the record — so *that* picture is reached, and the purge reports any
file the filesystem refused to remove rather than printing a count over
it. What is still out of reach is the full frame in `data/captures/`.
Every `where` object therefore carries `imagery_retention: "capture-side"`:
purging every observation here leaves the imagery exactly where it is, and
capture-side retention is not this cartridge's to promise or to give away.

This is also why a record with a frame pointer carries `frame-referenced`
alongside `derived-only` in `privacy_tags`. `derived-only` is a true claim about
the record's *content* — a class label, a score and a box, no pixels — and was
read as a claim about its *reach*, where it was false.

---

## 3. Endpoints

Both handlers are declared `def`, not `async def`. FastAPI runs a sync endpoint
in its threadpool, which keeps the JSONL read off the event loop with no executor
of our own — the same reason `tower/tower/routes/geometry.py` gives for the same
choice, and pinned by
`test_the_handlers_are_sync_so_disk_reads_stay_off_the_event_loop` because it is
the whole mechanism.

Both answer **404** when this Tower has the cartridge switched off
(`TOWER_OBSERVATION_ENABLED=false`). That 404 is a statement about
*configuration* — "this Tower serves no object memory" — and is never the answer
to a query about a class.

**That state used to be the default and no longer is** — §13 gives the reason.
An unconfigured Tower now answers 200 with an empty listing, which is a
different and weaker claim: "this Tower serves object memory and has none" is
not "this Tower serves no object memory". A client must keep rendering the two
differently.

### 3.1 `GET /object-memory/observations`

Every observation still within the retention window, **newest first** by
`observed_at`.

| Query parameter | Type | Default | Meaning |
|---|---|---|---|
| `object_class` | string | *(none)* | Narrow to one class. Absent means all classes. |
| `since` | float ≥ 0 | *(none)* | Only records the Tower **wrote** at or after this moment (`recorded_at >= since`). See below. |
| `retention_days` | float ≥ 0 | *(none)* | Narrow the window this read may see. See §5. |

`retention_days` below 0 is refused with **422** by the route rather than
reaching the store, which raises `ValueError` on a negative window and would
surface as a 500. `since` below 0 is refused the same way.

**`since` is how a client tells THIS recording's memories from history, and
it exists because a physical run once recorded zero observations while the
app kept showing 116 old ones — "nothing happened" and "everything is fine"
were indistinguishable.** It is a Tower-receipt epoch second, the same clock
a session's `started_at` is on (`GET /cartridges/object_memory/session`), so
a client stamps the session start and passes it here to get exactly the
records that recording created; an empty answer is then an honest empty
answer. Omitting it returns the whole store, which is the historical view a
wearer opts into separately. It filters on `recorded_at` (when the memory
was written), never `observed_at` (capture-journal receipt time), because
the former shares the session's clock and the latter would answer a subtly
different question at the edge. The chosen value is **echoed** as `since` in
the payload (`null` when absent), so the answer states which claim it is.
This is an additive parameter and an additive field: the contract identifier
is **unchanged**.

### 3.2 `GET /object-memory/last-seen/{object_class}`

When a category was last in view, or an honest silence. Accepts the same
`retention_days` parameter.

`{object_class}` is URL-encoded; classes with spaces (`cell phone`) work.

**Always 200 when a root is configured.** There is no 404 for an unobserved
class — see §1.

---

## 4. Fields

### 4.1 Envelope (both endpoints)

| Field | Type | Notes |
|---|---|---|
| `contract` | string | `object_memory.observations/2026-08-26`. Equality only. |
| `claim` | string | `category-was-visible-once` |
| `identity` | string | `category-not-instance` |
| `absence_means` | string | `not-observed-by-this-cartridge` |
| `spatial_ref` | **always `null`** | Reserved, never populated. Carried so a consumer sees the field exists and is empty. |
| `recorded_classes` | array of string | The universe of what could ever appear. **Fourteen entries on a default Tower since 2026-08-29**, when the verifier default became `owlv2` (§13); `["laptop", "cell phone"]` on a Tower with `TOWER_OBSERVATION_VERIFIER=none`. Derived from the **configured** verifier, which is what was asked for rather than what loaded — see §13. A class outside this list has never been looked for, a weaker silence than "looked for and not seen". **A client must read this list rather than hard-coding it.** |
| `imagery` | object | Where the pictures are: `contract`, `claim`, `filter_means`, and URL **templates** for `view`, `frame` and `crop` (§10). A descriptor, not an availability claim. |
| `retention` | object | See §5. |
| `object_class` | string \| **null** | The class the request narrowed to. `null` on an unfiltered listing. |

### 4.2 `GET /object-memory/observations` adds

| Field | Type | Notes |
|---|---|---|
| `since` | float \| **null** | The `since` the request was filtered to, echoed. `null` means the whole store (history); a value means "records written at or after this", which is how a recording's own memories are told from history. §3.1. |
| `observation_count` | integer | Length of `observations`. Counts what is **served**, which is what is within the clamped window and at or after `since` — not what is on disk. |
| `observations` | array of observation | Newest first by `observed_at`. |

### 4.3 `GET /object-memory/last-seen/{object_class}` adds

| Field | Type | Notes |
|---|---|---|
| `recordable` | **boolean** | Whether this class is one the cartridge ever writes. `false` means its absence carries no information at all. |
| `observed` | **boolean** | `observed`, never `present`. True means a record exists, not that the object is there. |
| `observation` | observation \| **null** | `null` when `observed` is false. |
| `where` | frame reference \| **null** | Hoisted from `observation` because "where did I leave it" is the question a client will bind to this response. `null` when `observed` is false. |

### 4.4 Observation

| Field | Type | Notes |
|---|---|---|
| `object_class` | string | One of `recorded_classes`. |
| `claim` | string | `category-was-visible-once`, repeated per record so a client holding one record out of context still has it. |
| `identity` | string | `category-not-instance`, likewise. |
| `confidence` | string | `unknown` \| `low` \| `medium` \| `high`. **The field a consumer should read.** Derived from `best_score` — the claim is "this was in view", and the best look is the best evidence for it. `unknown` is what `Confidence.from_score(None)` returns, so it reaches the wire on any record whose `best_score` is `null`; a decoder that accepts only the other three refuses a real record. |
| `detector_score` | float \| **null** | **Provenance.** How confident the detector was in the frame that *first* brought the class into view — the one frame `observed_at`, `frame_seq` and the box all describe. Never revised. |
| `best_score` | float \| **null** | **Evidence.** Strongest score while that sighting stayed in view. `null` means *not tracked* — a record written before this field existed. Never `0.0`, which would be a claim of no evidence. |
| `observed_at` | float (epoch seconds) | When the category came into view. Qualified by `time_basis`. |
| `time_basis` | string | `tower-receipt`. This slice can only know Tower-receipt time, never on-glasses capture time; `Rule 16` forbids conflating them. A client must not present this as a shutter time. |
| `recorded_at` | float (epoch seconds) | When *we* stored it. This is the privacy-relevant clock and the one retention filters on (§5). |
| `module_id` | string | `object-memory`. |
| `retention_tag` | string | The record's own retention class. |
| `privacy_tags` | array of string | `derived-only`, plus `frame-referenced` when a frame pointer is present. See §2. |
| `where` | frame reference | See §4.5. |
| `observation_id` | string | 16 hex characters. The handle the imagery routes take (§10). **Derived** from `session_id`, `object_class` and `observed_at`, never minted — so records written before this field existed have one, permanently, with no migration. |
| `last_seen_at` | float \| **null** | When the sighting stopped. `observed_at` says when it started and never moves; this accumulates. `null` on a record written before sightings existed — never `0`, which would claim a sighting of no duration. |
| `frame_count` | integer \| **null** | How many frames the sighting spanned. Same `null` rule. |
| `tier` | string \| **null** | Which policy tier admitted the record: `remembered` (written on the detector's word) or `verify` (written only because something agreed). See §11. |
| `verification` | object \| **null** | `null` means nothing was asked — the ordinary case for a `remembered` class. Otherwise `{agrees, proposed, label, score, model, reason}`: **which** model agreed and **how strongly**, so the record stays auditable and can be re-evaluated when the model changes. `label` is what the verifier would have called it and is **not** used to relabel the record. |

None of the three strength fields is a calibrated probability. They are detector
output, and an interpretation of detector output. `verification.score` is not one
either: it is a similarity, from a model whose threshold was fitted to 94 crops
from one home.

### 4.5 Frame reference (`where`)

| Field | Type | Notes |
|---|---|---|
| `kind` | string | Always `frame-reference`. **Not a place.** |
| `spatial_ref` | **always `null`** | Reserved, never populated, actively nulled on read. |
| `session_id` | string \| **null** | The capture this frame belongs to. `null` when the record carries no capture provenance. |
| `frame_seq` | integer \| **null** | Sequence number within that capture. |
| `camera` | string | e.g. `glasses-camera`. Renamed from the store's `source`, because "source" invites being read as a provenance system. |
| `bounding_box_normalized` | array of 4 floats \| **null** | Where in the *picture*, in normalised frame coordinates. **Not a world position.** Nested here, never at the top level. |
| `imagery_retention` | string | `capture-side`. This pointer resolves into `data/captures/`, whose lifetime this cartridge neither sets nor enforces. |

---

## 5. Retention — narrowable, never widenable

**This is the sharpest constraint in the contract.**

`retention_days` is a **request, not an authority**. The store records the window
it was *written* under in a manifest beside the data, and clamps every read to
`min(persisted, requested)`. A caller may narrow the window it sees and **nothing
a caller passes can widen it**.

This hole has been found here before. At the CLI layer,
`--retention-days 3650` against a store written under the 30-day default served a
40-day-old record in full, and retention under `06-PRIVACY-DATA.md` stopped being
a promise at all. HTTP is a new layer and therefore a new chance to reintroduce
it. The route adds nothing of its own that could weaken the clamp — it converts
days to seconds and hands them to the store — and
`test_a_reader_cannot_widen_retention_over_the_route` attempts the widening over
the wire against a store holding a genuinely out-of-window record.

`retention_days = 0` means **"no limit of my own"**, matching the CLI. It does
*not* mean "keep forever": it is still clamped to the persisted window.

The clamp is **reported**, not merely applied. A client that asked for 3650 days
and silently received 30 would have no way to learn its question was refused.

| `retention.*` | Type | Notes |
|---|---|---|
| `requested_days` | float \| **null** | What the caller asked for. `null` when the caller asked for nothing. |
| `effective_days` | float \| **null** | What will actually be honoured. `null` is *unbounded*, reachable only when the store itself was written unbounded. **Never `0`** — 0 days would mean "nothing is visible", the opposite claim. |
| `clamped` | **boolean** | `true` only when the caller asked for **more** than it received. A caller that asked for nothing has been refused nothing; a caller that asked for less got what it wanted. |
| `policy` | string | `min(persisted, requested): a reader may narrow this window and can never widen it` |

Filtering is on `recorded_at`, not `observed_at`: retention is about how long
*we* have held the data. The two are equal today and diverge the moment a real
capture timestamp is threaded through.

The store's `include_expired` opt-out is **never** passed from the wire. It
exists for maintenance paths — purge counting what it deletes, an operator
auditing the file — and is never the right answer for anything a wearer will be
shown, which is everything on this transport.

### Observed behaviour against the real corpus

`tower/data/object_memory` holds 55 real observations under a persisted 30-day
window. With one record backdated 40 days in a scratch copy:

| Request | `observation_count` | `effective_days` | `clamped` |
|---|---|---|---|
| *(no parameter)* | 54 | 30.0 | `false` |
| `retention_days=3650` | **54** | **30.0** | `true` |
| `retention_days=0` | **54** | **30.0** | `true` |
| `retention_days=0.5` | 0 | 0.5 | `false` |

The expired record cannot be recovered by asking.

---

## 6. Encoding disciplines

- **`null` means absent — never `0`, never `""`.** A missing `detector_score` or
  `best_score` is `null`. Substituting a number would invent evidence.
- **Booleans are booleans.** `bool` subclasses `int` in Python, and a `1` where a
  client expects `true` fails every Swift `as? Bool` decode. `observed`,
  `recordable` and `retention.clamped` are pinned as real booleans by
  `test_booleans_are_booleans_and_not_ints`.
- **The contract identifier is opaque.** Equality only.
- **Additive fields are not a contract bump.** A field an older decoder ignores
  does not break it. Removing a field, renaming one, or changing what an existing
  one *claims* does — and a change to any of the §1 claim values is the most
  breaking change this contract can carry, because it changes what the data
  means rather than what it contains.

---

## 7. Configuration

Moved to **§13**, which is the full table — the cartridge grew five more
settings and one of them reverses a default this document used to state.

The property that did not change: the observation root is **read-only to the
web process**, which never observes and never deletes. The producer is
`scripts/object_memory_session.py`, in its own process; deletion is
`scripts/object_query.py --purge-all`, typed by a human. What did change is
that both sides are now handed the **same** value from the same `Settings`
object, so they cannot disagree about where the store is — which they could,
and did, on 2026-08-26.

---

## 8. What did NOT change, and why that matters

`object_memory.observations/2026-08-26` is **unchanged**. The shipped iOS
decoder (`ios/Glasses/Workspaces/ObjectMemory/`) refuses a populated
`spatial_ref` and refuses any changed value of `claim`, `identity` or
`absence_means`, and every one of those still travels exactly as before:

- `claim` is still `category-was-visible-once`;
- `identity` is still `category-not-instance`;
- `absence_means` is still `not-observed-by-this-cartridge`;
- `spatial_ref` is still an explicit `null` at the envelope, the record
  and the `where` object.

The fields added in §4.4 are **additive**. A Swift `Codable` decoder
ignores unknown keys, so a client built against the previous payload
decodes the new one unchanged and simply does not see them. That is why
the identifier does not move: adding a field an older decoder ignores is
not a contract break; changing what an existing field *means* is.

**One thing to watch, and it changed on 2026-08-29.** `recorded_classes`
is configuration-dependent (§11), and a default Tower now returns
**fourteen** entries rather than two, because the verifier default became
`owlv2`. A client that hard-coded `["laptop", "cell phone"]` rather than
reading the list was correct against every Tower before that date and is
wrong against every one after it. This is exactly the "changing what an
existing field *means*" case the paragraph above calls a break — the
field's *type* did not move, its *contents* did, and the mitigation is
the one already stated: read the list.

---

## 9. Session control — `cartridge_session.control/2026-08-27`

The first **mutating** surface in this Tower, and the reason it exists is
measured. The 2026-08-26 physical run remembered 64 real observations,
and every one of them required a human to find a capture directory and
start a producer against it in a second terminal. There was no Start.

Deliberately **generic**: the route is keyed by cartridge id, knows no
cartridge, and the next producer that needs a button gets it for free.

| Method | Path |
|---|---|
| `GET` | `/cartridges/{cartridge}/session` |
| `POST` | `/cartridges/{cartridge}/session/{action}` |

`{cartridge}` is `object_memory`. `{action}` is one of `start`, `pause`,
`resume`, `stop`.

### 9.1 State, and what it does *not* claim

Three states: `stopped`, `active`, `paused`. There is no `starting` —
attaching is a process spawn and a dict update, and a transient state no
client can observe is a state that only exists to be got wrong.

**`state` is INTENT. `following` is FACT.** The payload says so in
`state_means: "intent-not-liveness"`. An `active` session whose producer
died in the first ten seconds of a walk is the "looks successful but does
nothing" failure this whole surface exists to make visible, and a client
that draws "remembering" from `state` alone will draw it for the rest of
that walk. **Render liveness from `following`.**

| Field | Type | Notes |
|---|---|---|
| `contract` | string | `cartridge_session.control/2026-08-27`. Equality only. |
| `state` | string | `stopped` \| `active` \| `paused`. |
| `state_means` | string | `intent-not-liveness`. |
| `states` / `actions` | array of string | The vocabulary, so a client can render controls without hard-coding it. |
| `supported` | boolean | Whether this Tower has a producer to start at all. `false` on a Tower with the cartridge switched off — a Start button that silently does nothing is worse than one that says why it cannot. |
| `session_id` | string \| **null** | Minted at Start, kept across Pause, cleared at Stop. **Not** a capture id. |
| `started_at` / `changed_at` | float \| **null** | Tower-receipt epoch seconds. |
| `following` | array of string | **Every** capture a producer for this cartridge is alive on right now, including one left over from a session that could not kill it. Supervisor-scoped, deliberately — see below. |
| `following_this_session` | array of string \| **null** | The subset of `following` that **this session** started. **This is the field a liveness claim is drawn from.** Added 2026-08-29. Three values, see below. |
| `captures` | array of string | Every capture **this session's** producer has been seen following, in order first seen — accumulated from `following_this_session`. When that is `null` this falls back to `following`, because a history that goes permanently empty on a Tower that cannot scope is a second thing broken by one thing being unanswerable. |

> ✅ **The supervisor-scoping defect is fixed, and the fix is additive.**
> This table said "this session's producer" until 2026-08-27; that was
> wrong, a reviewer reproduced it, and the warning that stood here from
> 2026-08-27 to 2026-08-29 described a live false positive.
>
> The defect: a producer that ignored `terminate()` on a previous Pause or
> Stop stays registered with the supervisor — the same condition that
> makes a Pause able to report `changed: true` without stopping anything.
> Start a **new** session and `following` reported the OLD session's
> capture under the new `session_id`, having attached nothing. Against the
> rule stated everywhere else here — render liveness from `following`,
> never from `state` — that produced a **false positive**: a brand-new
> session that attached nothing rendered as recording.
>
> **`following` was NOT narrowed to fix it.** Narrowing it would hide the
> un-killable producer, and an un-killable producer is exactly what "the
> Stop button failed open" looks like. That is the worst thing this
> cartridge can do to a person and the one thing that must never become
> silent. `following` therefore keeps its full breadth, and the
> intent-contradicts-liveness alarm is still drawn from it.
>
> **`following_this_session` is the new field**, scoped by the
> supervisor's own clock reading at the moment this session last went
> active. A client should:
>
> - draw **"remembering"** from `following_this_session`;
> - draw the **alarm** (`state` is `paused` or `stopped` and something is
>   still recording) from `following`;
> - say something separate and specific about `following` minus
>   `following_this_session` — a producer that is recording and that this
>   session's Stop will not reach.
>
> **Three values, and the third is why this is nullable.**
>
> - a **list** — these captures, and no others, have a producer this
>   session started still alive on them;
> - the **empty list** — this session has started nothing that is still
>   running. A positive claim, and safe to draw a warning from;
> - **`null`** — this session cannot scope the question at all. Fall back
>   to `following`, and draw no warning.
>
> A Tower that omits the field entirely means the same as `null`.
>
> The distinction is not pedantry. The first implementation answered `[]`
> for the unanswerable case, reasoning that a claim which cannot be
> scoped must not be made. The reasoning was right and the encoding was
> wrong: a client reads `[]` as a positive claim and draws "a producer
> you did NOT start is recording, and Stop will not reach it" from the
> difference with `following` — so a Tower that merely could not answer
> would have raised that alarm about a producer the person had started
> themselves.
>
> Scoped by **start time**, not by a list of ids the session kept, and the
> difference is not cosmetic: a capture that opens while the gate is open
> is spawned by the supervisor without consulting the session at all, and
> that is the *normal* path — Start before the camera is the documented
> order. A session that counted only its own attach returns would report
> nothing for almost every real walk.
>
> Both clocks are the supervisor's. `CaptureWorkerSupervisor._clock`
> defaults to `time.monotonic` and `CartridgeSession._clock` to
> `time.time`; the first implementation of this compared one against the
> other, every worker looked older than every session, and the correct
> looking code reported nothing at all. `CaptureWorkerSupervisor.mark()`
> exists so that a caller cannot get that wrong.
| `accepted` | boolean | On `POST` only. |
| `changed` | boolean | On `POST` only. `false` means the action was honoured and nothing moved — a double tap. |
| `attached_capture_id` | string \| **null** | On `POST` only. The capture a producer was just started against, if any. |

### 9.2 Status codes

| Code | When |
|---|---|
| **200** | The action was honoured, including an idempotent no-op (`changed: false`). |
| **404** | No such cartridge session on this Tower. A configuration answer. |
| **409** | The action cannot be honoured *from this state*. Body carries `reason`, `message`, and the state actually reached. |

Refusals are **409, not 200 with a flag**, and that is a deliberate
departure from §1's reasoning about `last-seen`. There, a 404 would be a
claim about the world. Here, an action that could not be honoured is not
a fact about anything, and a client that ignored the body would read 200
as "paused". A status code a client cannot ignore is the right shape for
a control surface.

`reason` values: `unsupported`, `not-active`, `not-paused`,
`unknown-action`.

### 9.3 Semantics a client should not have to guess

- **Start from `stopped` or `paused`.** It means "be running", whatever
  the app thought the state was. `resume` is stricter — it is a claim
  about continuing something, and from `stopped` it is refused.
- **Pause detaches the producer**; it does not signal it to idle. The
  process stops, which is observable in the process table and cannot go
  stale. Resuming costs one model load.
- **Resume attaches to whatever is recording NOW**, not to the capture
  Start found. A pause long enough to matter is long enough for the phone
  to have reconnected.
- **Stop is never refused**, from any state. Refusing it would leave the
  only way out of a bad state being a Tower restart.
- **Start before the camera is normal.** The session goes `active` with
  `attached_capture_id: null` and the next capture to open finds the gate
  open.
- **Nothing is persisted.** A Tower that restarts comes back `stopped`.
  Resuming a memory of what a camera sees without anybody asking again is
  the wrong direction to fail in.
- **Stopping ends the producing, not the memory.** Observations already
  written are untouched and stay queryable.

---

## 10. Imagery — `object_memory.imagery/2026-08-27`

Until now this cartridge exposed a **pointer** to a frame and never the
frame, and iOS renders that pointer as `Frame reference: capture
22e9d428…, frame 3410` inside a disclosure. That is a correct thing to
show a developer and close to useless to a wearer.

The evidence for serving the picture instead is unusually direct. MemPal
(15 adults aged 62–96, in their own homes, objects retrieved after a
40-minute delay) measured its own last-seen images as showing the true
location only **53%** of the time — and users still went from 0.81 to
0.95 retrieval accuracy, searching 1.1 rooms instead of 1.9. A
wrong-but-plausible cue plus a human closes the gap; a confident sentence
does not.

| Method | Path | Returns |
|---|---|---|
| `GET` | `/object-memory/observations/{observation_id}/imagery` | JSON — whether there is a picture and what may be said about it |
| `GET` | `/object-memory/observations/{observation_id}/frame` | `image/jpeg` — the whole frame, filtered |
| `GET` | `/object-memory/observations/{observation_id}/crop` | `image/jpeg` — the object, padded 35%, filtered |

`observation_id` comes from the listing. Both binary routes send
`Cache-Control: no-store`: this is sensitive first-person imagery on a
LAN-local origin, and a proxy or a browser holding a copy is a second
store nobody chose and nobody's retention governs.

### 10.0 Two sources, and only `/crop` has two — added 2026-08-29

`/crop` prefers a small crop **this cartridge owns**, written under
`<observation_root>/keyframes/<observation_id>.jpg` when the sighting
closed, and falls back to cropping the capture frame when there is no
owned one. `/frame` has only ever one source, because a keyframe *is* a
crop and there is nothing to synthesise a context view out of.

`imagery_source` on the payload says which served the bytes, and
`imagery_retention` says whose retention therefore governs them. **Both
are additive and `object_memory.imagery/2026-08-27` does not move**: a
shipped iOS build compares that identifier for equality and refuses a
payload that does not carry it, and a decoder that has never heard of a
keyframe reads `imagery_retention` exactly as it always did.

**Why this exists.** A record is kept for 30 days; the frame it points at
lives in `data/captures/`, which this cartridge does not own. Until
2026-08-29 that was survivable only by accident — this repository has no
capture pruner at all, `CaptureRecorder.purge()` has zero production
callers — so the imagery had never actually expired. Any pruner, or any
human reclaiming the ~2.1 GB an hour of walking costs, would have taken
the picture off every memory at once. A durable record pointing into an
ephemeral store is the defect; owning one small crop fixes it without
extending raw capture retention by a day.

**What it costs.** One JPEG per observation, long side capped at 384 px
at quality 80. The context view still comes from the recording and is
still governed by capture-side retention.

**What it does not change.** It is written through the same YuNet filter
`/frame` uses, and it is fail-closed: no weights, a filter that raises,
or a filter that returns nothing all write **no file at all** rather than
an unfiltered crop. There is no path in `KeyframeStore.write` by which
the input crop reaches the disk; everything written derives from the
filter's output. An image whose sidecar is missing is **ignored rather
than served**, so the sidecar's presence is what makes the claim
checkable.

### 10.1 The `imagery` payload

| Field | Type | Notes |
|---|---|---|
| `contract` | string | `object_memory.imagery/2026-08-27`. |
| `claim` | string | `frame-from-the-recording-this-record-was-derived-from`. It is a frame from a recording, filtered on the way out. It is **not** evidence of where anything is. |
| `available` | boolean | **Whether a picture can be served at all**, not whether `/frame` holds one. Built from a crop render since 2026-08-29, so it is `true` for a record whose recording has been deleted and whose owned keyframe survives — see the warning below. |
| `reason` | string \| **null** | `null` when available. Otherwise a value to switch on, never a sentence to display. |
| `memory_retained` | boolean | **The field this shape exists for.** `true` with `available: false` means the record is still here and its picture is not. |
| `filter` | string | `display-filter/yunet-2023mar@0.30`. See §10.3. |
| `filter_means` | string | `applied-on-read-the-stored-frame-is-unchanged` for a capture frame; `applied-before-this-file-was-written` for an owned keyframe. |
| `imagery_source` | string \| **null** | `capture-frame` or `object-memory-keyframe`. Which store served these bytes; `null` on a refusal, because nothing served them. Added 2026-08-29. |
| `frame_available` | boolean \| **null** | Whether `/frame` could serve the wider context view. `null` where it was not computed — on a refusal detail from a binary route, where the client already knows which it asked for. Added 2026-08-29. |
| `regions_filled` | integer | How many regions the filter filled. Zero means **nothing was detected**, not that there was nothing there. |
| `subject_obscured` | float 0.0–1.0 | How much of the picture the filter covered. **The denominator follows `imagery_source`**: for a capture frame it is the record's own box, measured on the full frame; for an owned keyframe the full frame is gone by serving time, so it is the whole padded crop — which is the box plus 35% on every side, and therefore a systematically *smaller* number for the same fill. Read it with `imagery_source`. See §10.4. |
| `bounding_box_normalized` | array of 4 floats \| **null** | Where in the picture. Same caveat as §4.5. |
| `imagery_retention` | string | `capture-side` **or** `object-memory`, depending on `imagery_source`. Not a constant since 2026-08-29: a crop served from this cartridge's own keyframe expires with the record and is deleted by `--purge-all`, while a frame served out of `data/captures/` may vanish on a schedule this cartridge does not set. |
| `imagery_retention_means` | string | The same fact as a value to switch on rather than a sentence to display. |

> ⚠️ **`available` is not `/frame`'s availability, and it was until
> 2026-08-29.**
>
> The view route rendered a frame and reported that. Once a record could
> own a crop that outlives its recording, that answer was wrong in the
> one case the crop exists for: `available: false` for a record whose
> picture was sitting on disk. A client that gates on it — and the
> shipped iOS loader does, with `guard description.available else {
> .noPicture }` — would never have asked for the crop, and would have
> told a wearer the picture was gone while it was being held for them.
>
> So the view is built from a **crop** render, and `frame_available`
> answers the separate question about the context view. A client
> choosing between the object and its context must read both: they no
> longer stand or fall together.

### 10.2 Status codes, and the one that carries meaning

| Code | `reason` | Meaning |
|---|---|---|
| 200 | `null` | A picture. |
| **410** | `imagery-no-longer-available` | **The pointer is intact and the picture is gone**, and there is no owned keyframe either. `memory_retained: true`. Usually capture-side retention; the Tower distinguishes that (INFO) from a pointer that does not resolve into a recording still on disk (WARNING, a defect) in its **log**, and deliberately not on the wire — there is nothing a wearer can do differently about the two. |
| 404 | `no-such-observation` | The handle matched nothing **within retention**. |
| 404 | `record-has-no-frame-reference` | The record never had a pointer. |
| 503 | `display-filter-unavailable` | No face-detection weights, or the filter failed. **No capture frame is served.** An owned keyframe still is: it was filtered before it was written, so the check has already been passed — §10.3. |
| 503 | `no-capture-root-configured` | This Tower has nowhere to look. |
| 503 | `frame-unreadable` | The file is there and could not be decoded. |

`/imagery` answers **200 even when there is no picture** — the resource,
what this cartridge knows about the imagery behind a record, exists
either way. Only an unknown handle is a real 404 there.

**Retention is not bypassed by knowing an id.** The handle resolves
through the same clamped read the listing uses, so an expired record is
unreachable by its own id.

### 10.3 The filter is a display filter, and the wording is load-bearing

`tower/world_builder/redaction.py` runs **before persistence**, at the one
choke point every persisted pixel passes through, and earns the name
privacy transformation. This runs on **read**. The raw frame stays exactly
where it is — the capture manifests record `redaction: "none"` and this
cartridge does not own that tree.

So the label says `display-filter/…` and never "redacted", "anonymised"
or "privacy-safe", and it names the detector and its threshold rather
than asserting an outcome. YuNet has measured blind spots: a face
occluded past about 60%, a face rotated about 90° in plane, and profile
and rear views are a known weakness of this detector class. **Bodies,
clothing, room contents, screens and any undetected face are all still in
the picture.**

A Tower whose weights are missing **serves no capture frame**. There is
no lenient default, because the lenient default there is a raw
first-person frame.

**An owned keyframe is the one exception, and it is not a weakening.**
Those bytes were filtered *before* they were written — which is the
`world_builder/redaction.py` posture this section contrasts itself with,
not the read-time one — so `filter_means` for a keyframe reads
`applied-before-this-file-was-written`. Refusing to serve one because the
ONNX weights went missing *afterwards* would withhold a picture on the
strength of a check that has already been passed, more thoroughly, at
write time. The corollary is the safe one: if the weights were missing at
write time, **no keyframe exists**, and the request falls through to the
capture path and is refused there.

The wording rule is unchanged and applies to both. A keyframe's label is
still `display-filter/yunet-2023mar@0.30` and still names the detector
and its threshold; it is never "redacted", "anonymised" or
"privacy-safe", and YuNet's blind spots are exactly what they were.
`regions_filled: 0` on a keyframe still means nothing was detected.

### 10.4 `subject_obscured`, and the defect it exists for

On frame 2708 of the physically validated capture — a desk with a
monitor, a lit keyboard and a red gaming mouse, and no person anywhere in
the picture — the filter fired **twice**, and one fill landed squarely on
the mouse the record was about. Measured across 1,845 corpus frames it
fires on **40.2%** of them, median region 12.5% of the frame; of 36
firings inspected by eye, 4 were a real face (the wearer in a mirror) and
32 were hands on a keyboard, a screen, a door or a sink.

The filter was **not** weakened: a face-detection threshold is not a
picture-quality knob. Instead the overlap is measured and reported, and a
client with `subject_obscured > 0` should say the subject is behind a
fill, or fall back to `/frame` — rather than showing a black rectangle
without comment.

---

## 11. What is remembered, and why it is two lists

The old policy was `("laptop", "cell phone")`, chosen from a score
histogram. A score histogram describes the detector's opinion of itself.
Reading the crops found a ceiling fan detected as `airplane` at 0.99 and
as `scissors` at 0.93, a white door as `refrigerator` at 0.95, and the
three highest-scoring `remote` sightings in 18,821 frames all laptop
keyboards. Full evidence:
`docs/superpowers/research/2026-08-27-object-memory-corpus-precision.md`.

So the class list became a policy with tiers:

| tier | what it means | classes |
|---|---|---|
| `remembered` | written on the detector's word | `laptop`, `cell phone` |
| `verify` | written **only** if a second opinion agrees | `remote`, `mouse`, `cup`, `bottle`, `keyboard`, `backpack`, `handbag`, `suitcase`, `book`, `umbrella`, `scissors`, `toothbrush` |
| `context` | detected, often correctly, and not a memory — furniture is a live question and belongs to Scene Understanding | `bed`, `tv`, `couch`, `chair`, `sink`, `toilet`, `refrigerator`, `microwave`, `oven`, `dining table`, `potted plant` |
| excluded | never, by any configuration | `person` |

`person` is not a tier. It is a separate constant, checked first, and no
model can reach past it. Whether this cartridge may persist a record per
detected bystander remains an **open ruling for a human**; leaving
`person` out is what lets the cartridge ship without one.

**A verifier can only refuse or confirm.** It is consulted for classes
the deterministic tables have already admitted, and a verdict naming
anything else is recorded as evidence and changes nothing.

With no verifier — `TOWER_OBSERVATION_VERIFIER=none`, which was the
default until 2026-08-29 — the `verify` tier is never written, and
`recorded_classes` is `["laptop", "cell phone"]`: the same answer the old
whitelist gave, reached from evidence rather than asserted. On a default
Tower today the verifier is `owlv2` and all fourteen are recordable.

---

## 12. `spatial_ref` — the shape it will take, and why it is still null

**Nothing in this cartridge knows where anything is in a room, and
nothing here is going to change that.** Object Memory does not run SLAM
and must not.

The field stays an explicit `null` at every level, in every payload, on
this contract identifier. The shipped iOS decoder **refuses a populated
value**, so filling it would not be a widening — it would be a break.

When a trustworthy world exists to anchor against, the shape is already
determined by `CARTRIDGE-GROUNDWORK.md` §4 and by World Builder's own
schema, and it is **not** a bare coordinate:

```jsonc
"spatial_ref": {
  "kind": "world-builder-anchor",
  "world_id": "…",
  "world_schema": "world_builder.geometry/…",
  "anchor_keyframe_id": "…",          // REQUIRED. Without it the first
  "position_in_anchor_frame": [x,y,z],// loop closure permanently and
  "frame_revision": 12,               // undetectably invalidates every
                                      // earlier anchor: a submap
                                      // re-anchor is not a global
                                      // similarity and cannot be
                                      // composed forward.
  "units": "world",                   // NEVER metres. Monocular SLAM
                                      // has no scale.
  "observed_at": 1787806912.4,
  "time_basis": "tower-receipt",
  "confidence": "low"                 // a LABEL, not a number
}
```

Three rules travel with it whenever it lands:

1. It requires a **new dated contract identifier**, because a client that
   refuses a populated `spatial_ref` today is correct to.
2. It is **optional forever**. Object Memory must keep working with no
   map at all, and a record without an anchor is a complete record.
3. `where.kind` stays `frame-reference` for records that have no anchor.
   A client must never project `bounding_box_normalized` into a world
   frame.

Object Memory will not build this. It will consume it if World Builder
offers it.

---

## 13. Configuration

| Variable | Default | Effect |
|---|---|---|
| `TOWER_OBSERVATION_ENABLED` | `true` | `false` switches the cartridge off entirely: nothing is produced, and both `GET`s answer **404**. |
| `TOWER_OBSERVATION_ROOT` | `<tower>/data/object_memory` | Where the producer writes **and** where the routes read. One value, handed to both. |
| `TOWER_OBSERVATION_DEVICE` | `auto` | Where the detector runs. `auto` resolves in the **producer**, which is the process that imports torch. CPU measured 69.3 ms/frame against CUDA's 100.4 on this host, so auto picking the GPU costs nothing measurable either way. |
| `TOWER_OBSERVATION_VERIFIER` | **`owlv2`** | Changed from `none` on 2026-08-29 — see below. `owlv2` loads `google/owlv2-base-patch16-ensemble` (Apache-2.0, ~600 MB, downloaded once) and unlocks the `verify` tier, taking `recorded_classes` from two to fourteen. An unrecognised name falls back to `none` — the **narrowing** direction — and is logged loudly. An **empty** value means `none`: a person switching it off is obeyed rather than defaulted over. |
| `TOWER_OBSERVATION_VERIFIER_DEVICE` | `auto` | Where the verifier runs. The measurement that made this CUDA still holds — 126 ms a crop on the GPU against 2,473 on the CPU, a factor of nineteen, for 620 MB of VRAM — but `cuda` as a literal default is a value a second machine has to un-set by hand, which is the edit-the-environment-before-every-run problem this whole surface exists to remove. `auto` picks the GPU where there is one and says which it got. |
| `TOWER_OBSERVATION_RETENTION_DAYS` | `30` | The window the producer writes under, recorded in the store manifest. |
| `TOWER_OBSERVATION_KEEP_IMAGERY` | `true` | **Whether this cartridge persists pixels at all.** `true` writes one small filtered crop per observation under `<observation_root>/keyframes/` (§10.0), deleted with the record. `false` writes no new crop; a record's picture is then a capture frame for as long as capture-side retention keeps one. Turning it off does **not** delete crops already on disk — they are still served and still pruned with their records, because a config change is not a deletion request; `scripts/object_query.py --purge-all` is. |
| `TOWER_CAPTURE_ROOT` | *(unset)* | Required for §10. Without it the imagery routes answer 503. |
| `TOWER_FACE_REDACTION_MODEL` | *(vendored)* | The YuNet weights. Without them **no capture frame is served, and no keyframe is written**. A keyframe already on disk is still served, because it was filtered before it was written — see §10.3. |

**The verifier default was reversed on 2026-08-29, and this closes a
decision that was explicitly reserved.**
`docs/agent-handoffs/OBJECT-MEMORY-HANDOFF.md` section 7.4 recorded
turning the verifier on by default as an open question **for a human, not
for an agent**: *"the default stays `none` because 94 crops from one home
justify building it and not switching it on for everybody."* A human
closed it. The 2026-08-29 product pass was instructed that OWLv2 is this
project's intended standard configuration and that setting the variable
by hand before every launch is not acceptable for ordinary use.

What makes it safe to default rather than merely asked for: a host that
cannot load the weights is **not broken by it**. `_build_verifier`
catches the failure, says so on stderr, and the run continues with no
verifier — which narrows `recorded_classes` back to the two the detector
is trusted on. The producer's report then carries both `verifier` (what
ran) and `verifier_requested` (what was asked for), so a run can never
claim a verification tier it did not have. Set the variable to `none` to
get the old behaviour deliberately.

**The unset-means-404 default was reversed, and the reason is the point.**
It used to default to `None` on the grounds that "a memory of what a
wearer's camera saw does not go on the network because a process happened
to start in a directory that has one". The physical test showed what that
bought: the producer wrote 64 observations to `data/object_memory`
regardless, and the only thing the unset default prevented was the
**wearer** reading their own memory back. Data existed, nothing served
it, and no log line said why. A default that hides data from its owner
while still storing it protects nobody. The switch moved to
`TOWER_OBSERVATION_ENABLED`, where the decision actually is.
