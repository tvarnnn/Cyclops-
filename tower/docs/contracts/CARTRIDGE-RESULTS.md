# Cartridge Results — the Tower → iOS structured result channel

**Status: IMPLEMENTED**, on branch `integration/cartridge-result-channel-v1`.
Everything in this document exists in code and is covered by tests. Where
something does not exist, it says so and says why.

**Envelope contract:** `cartridge_results.envelope/2026-08-23`
**Producers offered:** World Builder `status`, contract
`world_builder.status/2026-09-10`. Nothing else. See §9.

**Audience.** Whoever implements the iOS consumer. You should be able to
write it from this document without reading Tower's Python. If you find
yourself guessing, that is a defect in this document — say so rather than
guessing.

---

## 1. What this channel is, and what it is not

It is a **read-only reporting surface**. The Tower web process reports what
another process has already persisted. It does not run World Builder, does
not start builds, does not touch frames, and never writes.

That matters for what you can expect from it:

- Results are **snapshots**, never deltas. Each one is complete and
  supersedes every earlier one.
- Results are **coalesced**. If you read slowly you get the newest state,
  not a backlog.
- Nothing here is **live in the sensor sense**. World Builder runs in its
  own process, writing to disk; this channel polls those files. Latency is
  a poll interval, not a frame time.

`TOWER-TO-IOS.md` §6.1 previously declared "any World Builder transport at
all" BLOCKED. That entry conflated two separable things — a World Builder
**transport**, and World Builder as a **live in-process module**. All four
of its blockers (a `bytes`-only synchronous `process()`, a scalar
`ExperimentResult`, a registry of one module, a lifecycle timeout too short
for a build) are properties of the second. None of them bear on a reader
that never joins the frame path. §6.1 is now resolved for the transport;
the live-module half remains blocked and is not what this is.

---

## 2. Discovery — do this first

### On the socket: `{"type": "cartridges"}`

**This is the path for the phone.** iOS owns exactly one WebSocket and has
no HTTP client, no REST layer and no second transport — so discovery
happens on the socket already open. Send `{"type":"cartridges"}`, receive
the declaration below. **Cache it**; it changes only when the Tower build
changes.

Send it only **after** the ping/pong handshake has completed. The Tower
never speaks first, so nothing here can arrive before the pong — but a
client that asked for capabilities before validating would be reading its
own reply into the handshake.

### `GET /cartridges` — the same object, for operators

An HTTP route returning byte-identical output, for curl, dashboards and
anything that is not the phone. Both surfaces call one function and a test
asserts they cannot drift. **The phone does not need it.**

```json
{
  "type": "cartridges",
  "envelope_contract": "cartridge_results.envelope/2026-08-23",
  "cartridges": [
    {
      "cartridge": "world_builder",
      "result_type": "status",
      "contract": "world_builder.status/2026-09-10",
      "available": true,
      "unavailable_reason": null,
      "snapshot_only": true
    }
  ],
  "not_offered": [
    {"cartridge": "experimental_cv", "reason": "..."},
    {"cartridge": "document_memory", "reason": "..."},
    {"cartridge": "scene_understanding", "reason": "..."}
  ]
}
```

Both surfaces call one function; a test asserts they are byte-identical.

### The three states, and how to tell them apart

`IOS-to-Tower.md` §0.1 requires these to stay distinct because they call
for opposite user responses.

| You observe | State | Show |
|---|---|---|
| the cartridge appears in `not_offered`, or in neither list | the Tower says nothing about it | "not built yet" |
| it appears in `cartridges` with a `contract` your build does not implement | contract you do not speak | "update the app" |
| it appears in `cartridges`, contract matches, `available: false` | offered but unreachable | the `unavailable_reason`, and an action to connect/configure |
| it appears in `cartridges`, contract matches, `available: true` | ready | subscribe |

**`not_offered` is for humans.** Do not decode against it, and never read
presence there as an offer. It exists so an operator can tell "the Tower
does not know what `document_memory` is" from "the Tower knows and is not
serving it yet".

### Contract identifiers are opaque

Compare for **equality only**. Do not parse them, do not order them, do not
infer compatibility. They are dated rather than numbered precisely so that
nobody is tempted to compute which is greater. A mismatch means "we are not
talking about the same agreement" — nothing more, nothing less.

This is deliberately different from `schema_version` inside the World
Builder payload, which is an integer describing Tower's own on-disk format.
That one is Tower's business; ignore it unless you are debugging.

---

## 3. Subscribing

All result messages travel on the **existing `/ws` socket**. No new
connection, no new port. `ping`/`pong`, `frame`/`frame_result`,
`stream_start`/`stream_stop` are entirely unaffected and unchanged — a test
asserts `frame_result` is field-for-field identical with and without a
subscription open.

### Request

```json
{
  "type": "result_subscribe",
  "cartridge": "world_builder",
  "result_type": "status",
  "contract": "world_builder.status/2026-09-10",
  "world_id": null,
  "session_id": null,
  "since_revision": null
}
```

| Field | Required | Meaning |
|---|---|---|
| `cartridge` | yes | string, from the declaration |
| `result_type` | yes | string, from the declaration |
| `contract` | no | if present, must equal the offered contract, or the subscribe is refused |
| `world_id` | no | inspection mode: report on this specific world. Omit to follow the live/most-recent one |
| `session_id` | no | a specific session within that world. Omit for the latest |
| `since_revision` | no | the `revision` you last held. Affects only `cursor_status` — see §6 |

Omitting `world_id` selects a world holding a live writer lock if one
exists, otherwise the most recently updated. Supplying it is
`WorldInspectionMode.inspecting(worldID:)`: it pins the target, and a
counter that moved would be a bug.

### Reply

```json
{
  "type": "result_subscribed",
  "envelope_contract": "cartridge_results.envelope/2026-08-23",
  "subscription_id": "sub-1",
  "cartridge": "world_builder",
  "result_type": "status",
  "contract": "world_builder.status/2026-09-10",
  "snapshot_only": true,
  "world_id": null,
  "session_id": null,
  "cursor_status": "absent"
}
```

**Immediately followed by a `cartridge_result` with `seq: 1`.** A
subscription always begins with a complete snapshot, unconditionally,
whatever `since_revision` you sent.

`subscription_id` is unique **per connection** and starts at `sub-1` on
each new socket. Two connections will both see `sub-1`; that is correct and
they are entirely independent.

**`world_id` and `session_id` on the reply are the request's, verbatim** —
`null` where the request omitted them — never the world or session the
Tower *resolved* (that is the payload's `selection` block, §10). A client
with more than one subscribe outstanding — a pin change while an earlier,
slower subscribe is still being answered — matches each ack to the subscribe
it answers by these two fields, and closes an ack whose pin is not the one
it holds. The Tower has echoed them since the channel existed; stated here
since 2026-09-14 because the phone now depends on it.

### Unsubscribing

```json
{"type": "result_unsubscribe", "subscription_id": "sub-1"}
```
→ `{"type": "result_unsubscribed", "subscription_id": "sub-1"}`

Unsubscribing an id that is not open returns `result_error` with reason
`unknown_subscription`. **Disconnecting is sufficient cleanup** — you do not
need to unsubscribe before closing the socket.

---

## 4. The envelope

Every result arrives in this shape. The `payload` is the only
cartridge-specific part.

```json
{
  "type": "cartridge_result",
  "envelope_contract": "cartridge_results.envelope/2026-08-23",
  "subscription_id": "sub-1",
  "cartridge": "world_builder",
  "result_type": "status",
  "contract": "world_builder.status/2026-09-10",
  "seq": 4,
  "revision": "e252f739c1cdedab",
  "revision_changed": true,
  "coalesced": 0,
  "cursor_status": null,
  "snapshot": true,
  "tower_sent_at": 1787463092.958,
  "time_basis": "tower-receipt",
  "payload": { }
}
```

| Field | Type | Meaning |
|---|---|---|
| `seq` | int | **Dense per subscription**, starting at 1, assigned at send time. Ordering guarantee. A gap means corruption, never a drop |
| `revision` | string | **Opaque.** Compare for equality only. Changes if and only if the reported state changed |
| `revision_changed` | bool | Whether `revision` differs from the one you were last sent on this subscription |
| `coalesced` | int | How many snapshots were superseded in your slot since your last delivery. `0` normally. `>0` means you read slowly and skipped intermediate **states** — you did not miss information, because this snapshot is complete |
| `cursor_status` | string or null | Only on the first result of a subscription. Null afterwards |
| `snapshot` | bool | Always `true` today. Present so a future delta mode cannot be mistaken for this one |
| `tower_sent_at` | float | Unix epoch seconds, **Tower's clock** |
| `time_basis` | string | Always `"tower-receipt"` |

### `seq` and `coalesced`, and why they are two fields

A single gappy sequence number would conflate "you missed data" with "you
were sent less data because you did not need it", and only the first
deserves alarm. `seq` stays dense so a gap is unambiguously a bug;
`coalesced` reports slowness separately.

### Every timestamp is Tower-receipt time

There is **no capture timestamp anywhere in this system** — the frame
protocol carries no time field. So `tower_sent_at`, `started_at`,
`ended_at`, `built_at` and `mapping_seconds` are all Tower's clock.

`IOS-to-Tower.md` §0.3 holds `observedAt` and `receivedAt` separately and
will never substitute one for the other. Nothing here is an observation
time in the sensor sense. Render capture time as **unknown**.

---

## 5. Ordering and revision semantics

- **`seq` is monotonic and dense within one subscription.** It is not
  global, not per cartridge, and not comparable across subscriptions or
  connections.
- **`revision` is an opaque change identity**, computed by hashing the
  payload with volatile fields removed. It is *not* a counter and carries
  no ordering. Two different revisions mean the state differs; equal
  revisions mean it does not.
- **`progress.mapping_seconds` is excluded from `revision`.** It advances
  continuously while a session is live, and including it would make every
  poll look like a change — the exact failure `IOS-to-Tower.md` §1.2
  describes.
- Because that field is excluded, it is refreshed by a **heartbeat**: an
  unchanged snapshot is re-sent about every 2 s while you are subscribed,
  carrying `revision_changed: false`. Use that flag to skip redraws.
- `geometry.revision` and `trajectory.revision` are separate opaque
  identities for those sections, so you can tell *what* changed without
  diffing.

**`geometry.revision` includes the build timestamp deliberately.** The
build's own input digest covers only which keyframes went in, so a rebuild
with a different backend or code version can produce different geometry
under an identical digest. Between a revision that occasionally reports a
change when nothing changed, and one that can stay silent while geometry
moves under a viewer, only the first is safe.

---

## 6. Reconnect and cursor semantics

**There is no delta stream, therefore there is no gap, therefore a cursor
cannot lose data.** This is the single most important property of the
design and it is why reconnection is simple:

> To reconnect, open the socket and subscribe again. You will receive a
> complete snapshot with `seq: 1`. You are fully caught up.

`since_revision` is **advisory only**. It never changes what you are sent —
only `cursor_status`, which tells you whether a redraw is needed:

| `cursor_status` | Meaning | Suggested action |
|---|---|---|
| `absent` | you sent no cursor | render the snapshot |
| `matched` | your revision is the current one | nothing changed while you were away; you may skip the redraw |
| `stale` | a real-looking revision that is not current | the world moved; redraw |
| `unrecognised` | not a usable revision (empty, wrong type) | treat your cache as worthless; redraw |

A stale or nonsense cursor is **never an error**. Refusing would deny a
client correct data over a field that cannot affect correctness.

Note that `revision` values do not survive a Tower restart in any
meaningful sense beyond content equality — they are content hashes, so the
same state produces the same revision. That is a feature: a cached
revision remains comparable across restarts.

---

## 7. Errors

All are `{"type": "result_error", "reason": <code>, "message": <prose>, ...}`
plus the fields noted. **None of them close the socket** — the frame path
keeps working through every one.

| `reason` | When | Extra fields |
|---|---|---|
| `malformed_request` | missing or wrong-typed `cartridge`/`result_type`/`world_id`/`session_id`/`subscription_id` | — |
| `unknown_cartridge` | this Tower offers no contract for that cartridge | `offered` (array of names) |
| `unknown_result_type` | the cartridge is offered but not that result type | `cartridge`, `result_type` |
| `contract_mismatch` | you sent a `contract` that is not the one offered | `offered_contract`, `requested_contract` |
| `cartridge_unavailable` | offered, but nothing to serve (e.g. no world root configured) | `contract` |
| `too_many_subscriptions` | more than 8 open on one connection | — |
| `unknown_subscription` | unsubscribing an id that is not open | `subscription_id` |
| `snapshot_failed` | the first snapshot could not be built | `cartridge`, `result_type`, `contract` |
| `consumer_too_slow` | a result was not accepted within the send timeout; **this subscription is now closed** — subscribe again to resume | `subscription_id`, `cartridge`, `result_type` |
| `channel_failed` | the Tower's shared reader died; **this subscription is now closed** | `subscription_id`, `cartridge`, `result_type` |

Every `result_error` sent in reply to a request also carries
`envelope_contract`. The two unsolicited ones, `channel_failed` and
`consumer_too_slow`, do not: they are built by the publisher's send
loop, which names the subscription and its cartridge/result type and
nothing more (`tower/results/publisher.py`). A client must not require
the field on a `result_error`.

`channel_failed` and `consumer_too_slow` are the two that arrive
unsolicited. On receiving it,
that subscription is gone — re-subscribe to resume. It exists because a
dead channel that still looks alive is worse than a crash: nothing anywhere
reports it.

There is also a **non-result** error you will now see on this socket:

```json
{"type": "protocol_error", "reason": "unknown_message_type",
 "message_type": "...", "message": "..."}
```

Previously an unrecognised message produced only a server-side log line, so
"not implemented" and "lost in flight" were indistinguishable from the
phone. This is additive; the six existing message types never trigger it.

---

## 8. Limits, bounds and backpressure

| Bound | Value | Why |
|---|---|---|
| Subscriptions per connection | **8** | a remote party must not grow a server-side dict at will |
| Pending snapshots per subscription | **1** | there is no queue. A new snapshot replaces the pending one |
| Snapshot size | fixed arity, measured **< 8 KB** | a test asserts the payload contains no unbounded list |
| Send timeout | **1 s** (`SEND_TIMEOUT_S`) | a consumer that does not accept a result within this has its subscription closed. It is also the longest a `frame_result` can queue behind a result send, because both take the connection's send lock |
| Lock wait | **1 s** (`LOCK_TIMEOUT_S`) | bounded separately from the send, so a slow frame send cannot consume a result's budget and cause a spurious drop |
| Lock wait **+** send, together | **2 s** (`TOTAL_SEND_TIMEOUT_S`) | the ceiling a client should budget against |

> These three read **2 s / 2 s / 5 s** in this table until 2026-08-27,
> along with a justification ("which is why it is 2 and not 5") for a
> value that had already changed. A client budgeting against 5 s gets
> `consumer_too_slow` and a closed subscription at 1 s, re-subscribes,
> and loops. Read the constants in `tower/results/publisher.py`.

| Poll interval | **0.5 s** | how often the Tower re-reads disk |
| Heartbeat | **2 s** | how often an unchanged snapshot is re-sent |

### Measured cost

All on this host (Windows 11, CPU-only), so treat them as shape rather
than as a spec.

| What | Figure |
|---|---|
| One snapshot, 7-event journal | **0.75 ms** |
| One snapshot, 50,000-event journal | **0.73 ms** — flat, because the journal is summarised once per change and the summary is three scalars |
| The same, without caching | 0.98 ms → **117 ms**, i.e. linear in session length |
| Frame reply, no subscription | median **3.224 ms**, p95 3.873 ms |
| Frame reply, one subscription open | median **3.220 ms**, p95 3.817 ms |
| Cost of the channel to the frame path | **−0.004 ms (−0.1%)** — noise, over 400 frames per condition in 5 alternating reps |

Over a simulated 30-minute session (3,600 polls, journal growing
throughout, one subscriber): traced memory went 5.7 KiB → **34.3 KiB and
plateaued**, and the payload stayed **byte-constant at 3,173 bytes**. Your
memory cost per subscriber does not grow with how long the Tower has been
mapping.

Snapshot computation runs in a worker thread, never on the event loop.

**Client responsibilities**

1. **Read the socket.** One WebSocket is one TCP stream: if you stop
   reading, you block your own `frame_result`s too, and after 5 s your
   subscription is dropped.
2. **Throttle rendering yourself.** `IOS-to-Tower.md` §4.8 is right that
   the receiving side needs its own throttle. Tower coalesces, but you
   should still not put a list diff on the main actor per message.
3. **Unsubscribe when a workspace closes** if the connection outlives it.
4. **Treat `revision` as opaque** and `contract` as opaque.

**Tower responsibilities**

1. Never send a partial payload; every result is complete.
2. Never let a slow consumer accumulate memory: one slot, newest wins.
3. Never let the result channel affect the frame path, capture, or any
   cartridge's state.
4. Stop all polling when the last subscriber goes.
5. Tell you when the channel dies rather than going quiet.

---

## 9. Cross-cartridge extension

The envelope is generic; payloads are not. To add a cartridge:

1. Write `tower/results/<cartridge>.py` producing a `Snapshot`. It is the
   **only** file allowed to import that cartridge — enforced by
   `test_the_result_channel_core_is_cartridge_blind`.
2. Wire it in `tower/results/__init__.py: make_snapshot_for`.
3. Add a `CartridgeOffer` in `tower/results/registry.py` and move the entry
   out of `NOT_OFFERED`.
4. Mint a new contract identifier. Do not reuse another cartridge's.

Nothing in the envelope, publisher, registry or routes changes. The
subscription, ordering, coalescing, reconnect and error machinery is
already generic and already tested — the Experimental CV Lab was added
without touching any of it.

**Two lists, because there are two transports.** `cartridges` is what
can be SUBSCRIBED to. `http_contracts` is what can be FETCHED, and it
exists because iOS caches a declaration — a contract discoverable only by
making a call is one a phone cannot plan around. Each entry carries
`cartridge`, `contract`, `entry_route`, `available`, `unavailable_reason`
and `why_not_a_subscription`.

Only Document Memory's library is listed today. World Builder's geometry
and Object Memory's observations are the same shape and are not declared;
their identifiers live in adapter modules rather than in `contracts.py`,
and `registry.py` must stay cartridge-blind, so those two lanes own that
move.

**What is offered, as of the 2026-08-27 unification**

| Cartridge | Result type | Contract | Section |
|---|---|---|---|
| World Builder | `status` | `world_builder.status/2026-09-10` | §10 |
| Experimental CV Lab | `status` | `experimental_cv.status/2026-08-27` | `EXPERIMENTAL-CV-LAB.md` |
| Scene Understanding | `live` | `scene_understanding.live/2026-08-27` | §14 |
| Document Memory | `status` | `document_memory.status/2026-09-07` | §15 |

**`not_offered` is now EMPTY, and that is a claim.** Three cartridges
left that list on 2026-08-27 — Scene Understanding, Document Memory and
the Experimental CV Lab — and none of them left it because they got
better. Document Memory's detector still fires on essentially nothing at
the geometry the glasses deliver. They left because their limits became
things the PAYLOAD STATES, which is what an offer is for. A cartridge
belongs in `not_offered` only while it can say nothing at all; one that
can say "I have observed nothing, and here is precisely why" belongs in
`cartridges`, available or not.

**Object Memory is in NEITHER list, and that is the one deliberate gap.**
It has a store, read routes, imagery routes and a live control surface at
`/cartridges/{cartridge}/session` — but no entry in `registry.declare()`.
Declaring it breaks a pinned iOS test
(`testTheTowerDeclaresOnlyTheWorldBuilderContract`), so the socket
declaration waits for the iOS lane to take both halves in one change. The
Tower side is about four lines. This is a decision for a human and must
not be closed by an integrator noticing the asymmetry; a client that
needs Object Memory today reaches it over HTTP and learns nothing about
it from the declaration.

**One shared signature, reconciled.** `declare()`, `find_offer()` and
`known_cartridges()` take `world_root` positionally and everything else
KEYWORD-ONLY with a default: `document_root`, `scene_enabled`, `cv_lab`.
The defaults are the safety property — a caller that has not been taught
about a cartridge gets it declared UNAVAILABLE rather than silently
offered as working, so forgetting to thread a value through under-promises
(iOS renders "connect") instead of promising a channel the Tower cannot
serve. `cv_lab` is **duck-typed and never imported** — anything with an
`availability()` returning `(available, reason)` — because
`test_the_result_channel_core_is_cartridge_blind` forbids the registry
from importing a cartridge, and this time the surface is a wire contract.

Callers do not assemble those arguments by hand. `registry.declaration_inputs(app_state)`
is the single reader of declaration state off an app, used by `GET
/cartridges` and by `{"type": "cartridges"}` alike. Two call sites each
reaching for their own subset of `app.state` is exactly how the two
surfaces would come to disagree, and the disagreement would be invisible
until a phone hit the one that was wrong.

**A live cartridge does not fit steps 1–4 above, and Scene Understanding
is the case.** It persists nothing by design — enforced, not intended —
so there is no file for a reader to read. The resolution was to keep the
separation that matters rather than the mechanism: the live half is
`tower/scene/live.py`, which owns a worker thread and a model and is
constructed by `tower/cartridge_runtime.py`; the results package is handed
the session object and may only call `status()` and `latest()` on it.
`test_the_result_channel_never_writes` forbids a call named `observe` or
`build` anywhere under `tower/results/`, which mechanically prevents that
handle from becoming a second frame path.

So step 1 gains a clause: a cartridge that produces live state passes a
SESSION OBJECT into `make_snapshot_for`, and the adapter projects it. A
cartridge with a store still passes a root. The CV Lab is the third
shape: a live object that is neither a store nor a frame consumer, passed
as `cv_lab` and read the same read-only way.

**Experimental CV Lab is now offered** — `experimental_cv` / `status`,
contract `experimental_cv.status/2026-08-27`, documented separately in
`EXPERIMENTAL-CV-LAB.md`. It is the first producer in this package that
reads **live in-process state** rather than files another process wrote,
and that is worth knowing before adding a third:

- Step 1 above still holds, and `tower/results/experimental_cv.py` is
  still the only file allowed to know what a CV Lab is.
- The channel is still **read-only**. It reports the Lab; it cannot
  start, stop or configure one. Commands travel on their own messages
  (`cv_lab_start` and friends), deliberately not here — a mutation on a
  reporting surface is a place the next cartridge's author would look for
  one.
- What is new is **concurrency**. A World Builder snapshot is read from
  disk by `asyncio.to_thread` while nothing in this process is writing
  it. A CV Lab snapshot is read from a worker thread while the event loop
  is mutating the object, so the LAB takes a lock around every state
  transition and around building its document. A future live producer
  must do the same; the hub offers no such guarantee and should not,
  because it holds no state of its own.

---

## 10. World Builder `status` payload

Contract: `world_builder.status/2026-09-10`.

### 10.0 If you implement nothing else, implement this

The payload has two halves. **`model_state` and `world_snapshot` are the
half you decode.** Everything else is Tower-native evidence for those
values — useful when something looks wrong, not required to render.

They are shaped to drop straight into the iOS types that already exist:
`world_snapshot` maps 1:1 onto `WorldSnapshot`, and `model_state` names a
`WorldModelState` case. Tower does that mapping deliberately — the
alternative would put the translation table on the phone, where changing
it is an App Store release rather than a Tower restart.

```json
"model_state": "finalized",
"model_state_reason": null,
"world_snapshot": {
  "name": "Probe Room",
  "world_id": "be5076514e0d4727ab06f2ad1df5a5bf",
  "keyframe_count": 4,
  "revision": "b00bfe85819804da",
  "tracking": "good",
  "scale": "relative",
  "mapping_seconds": 0.0789,
  "calibration": "calibrated",
  "geometry": {
    "representation": "sparse point cloud",
    "element_count": 1360,
    "is_incremental": false
  },
  "trajectory": {
    "pose_count": 4,
    "path_length": 2.853251890377782,
    "path_length_unit": "world units",
    "scale": "relative"
  },
  "persistence": {"state": "saved", "revision": "67ccaee79f212c8d"}
}
```

**`model_state`** — one of `unsupported`, `idle`, `receiving`,
`finalizing`, `finalized`, `interrupted`, `failed`. `model_state_reason` is
prose for a person, or null.

| Tower sends | Meaning | Note |
|---|---|---|
| `unsupported` | this Tower cannot serve World Builder at all | e.g. no world root configured. Do not invite the user to wait |
| `idle` | Tower is fine, there is nothing to show yet | no worlds, or a world with no sessions |
| `receiving` | a mapping session is live | a process holds the world's writer lock |
| `finalizing` | capture ended; a builder is finishing, **or** (on a record older than 2026-09-06) the stored figures are not the final figures | `lifecycle.state` says which: `finalizing` is a live process that still holds the lock; `stopped_unbuilt` is the old caveat, and since 2026-09-10 only when the FIGURES say there is something to wait for — `geometry.element_count > 0` or `trajectory.pose_count > 0`, the same test as iOS `WorldEvidence.hasGeometry`; not `geometry.available`, which is true for a build that solved nothing — see below |
| `finalized` | capture ended and the stored geometry matches the keyframes | |
| `interrupted` | the session did not end the way a walk ends: the builder died, was asked to stop mid-walk, recorded an error, or finalization was left unfinished | **`world_snapshot` and `geometry` still describe whatever was built.** `model_state_reason` says what happened; `lifecycle.finalization` says how far finalization got. Added at `/2026-09-06` (the reason the identifier moved) |
| `failed` | reserved; nothing on disk maps to it since `/2026-09-06` | a client must still decode it |

**`selection`** — why *this* world is on the wire. Added at `/2026-09-06`.
Always present, including on an unavailable payload. **Not part of the
revision:** two subscriptions describing the same bytes agree on `revision`
whatever their `selection`.

| Field | Type | Notes |
|---|---|---|
| `mode` | `"pinned"`, `"live"`, `"finalizing"`, `"latest"`, `"none"` | `pinned`: the client named the world. `live`: a running builder holds its lock and its session is open. `finalizing`: a running builder holds its lock and its session has stopped. `latest`: **nothing is live; this is merely the most recently updated world on disk** — a client following the live world must not draw it as the live world. `none`: nothing to report |
| `world_id`, `session_id` | string or null | what was resolved |
| `reason` | string | prose |

**`awaiting_first_update` is never sent**, deliberately. It means "frames
are going out and the Tower has said nothing yet" — a fact about the
phone's own situation, which only the phone can know, and which it reaches
by not having received a snapshot.

**`world_snapshot` field notes**

- `revision` is the **same opaque string** the envelope carries. Compare
  for equality; never parse or order it.
- `tracking` — `good` / `lost` / `unavailable`. **`limited` is never
  sent**: it would need a threshold nobody has defined.
- `calibration` — `unknown` / `uncalibrated` / `calibrated`.
  **`calibrating` is never sent**: there is no in-session calibration
  procedure to be in the middle of.
- `scale` — `relative` or `unknown` in V1. **`inferredMetric` and
  `measuredMetric` are unreachable** on monocular hardware and will not
  arrive. Never render any figure from this channel in metres.
- `mapping_seconds` — the **Tower's** clock. Null if the Tower's wall
  clock stepped backwards.
- `geometry.is_incremental` is always `false` and never null: a build
  replaces the whole derived tree, so every snapshot is a whole world.
- `trajectory.scale` is carried **separately** from the snapshot's scale,
  because a spatial figure travels with its own. When `path_length` is
  null so is `path_length_unit`.
- `persistence.state` is `saved` whenever a world exists — World Builder
  persists by construction, so `session` is unreachable.
- Any field may be `null`, and null means **absent**, never zero.

**Nothing else in the payload is required.** Read on only if you want the
evidence behind these values.

### 10.1 Field reference

**`world`** — or `null` when nothing could be resolved.

| Field | Type | Notes |
|---|---|---|
| `world_id` | string | opaque, stable across sessions |
| `display_name` | string or **null** | operator-supplied. **Null means unnamed.** Do not derive a name |
| `schema_version` | int | Tower's on-disk format. Not the contract |
| `created_at`, `updated_at` | float | Tower clock |

**`session`** — or `null` when the world has no sessions.

| Field | Type | Notes |
|---|---|---|
| `session_id` | string | opaque |
| `started_at` | float | Tower clock |
| `ended_at` | float or null | null while live |
| `end_reason` | string or null | `"stop"`, `"disconnect"`, `"bounded_limit"` |
| `frame_source` | string | `"synthetic"`, `"recorded-capture"`, `"live-capture"`, `"unknown"` |
| `capture_id` | string or null | links to a capture directory, if any |
| `retains_raw_imagery` | bool | always `true` for World Builder, and stated on the record |

**`lifecycle`**

| `state` | Means | Evidence |
|---|---|---|
| `receiving` | a mapping session is live | a live pid holds the writer lock, no `session_stopped` |
| `finalizing` | the session stopped and the builder is finishing it (final solve, final build) | a live pid holds the writer lock **and** `session_stopped` was written. Since 2026-09-06 the live builder keeps its lock through finalization |
| `ready` | the session finished; stored geometry is what it produced | `finalization.state == "complete"` and the lock released; or, on an older record, manifest current. **Since 2026-09-10 it also covers a derived tree that no manifest describes** — either none was written beside it (a world built before `write_derived` wrote one per session) or the one that was cannot be read. Currency is then genuinely unjudgeable, `geometry.current` is `false`, and `lifecycle.reason` is non-null saying so; the figures are recounted from `poses.json`/`points.json` so `element_count` and `pose_count` are real, which is what lets the world open |
| `interrupted` | the session did not end the way a walk ends | a lock held by a dead pid (mid-walk or mid-finalization); `end_reason` `error` or `interrupted`; or a `finalization` left `pending`/`interrupted` with no live holder. **`geometry.available` says whether a reconstruction exists regardless** |
| `stopped_unbuilt` | (records older than 2026-09-06) capture ended; stored geometry is not current with the keyframes, **or** nothing was built at all | a manifest that is present and stale, and no `finalization` record. (Not "no manifest": since 2026-09-10 a tree with no manifest is `ready`, and a manifest with no tree is `interrupted`.) The `model_state` projection tells the two meanings apart on the FIGURES — `finalizing` when `element_count` or `pose_count` is above zero, `interrupted` when neither is. Not on `geometry.available`, which is true as soon as a build ran: `engine.build` writes a derived tree even when it solved nothing, so gating on it told a wearer to keep waiting for a world with zero points in it |
| `failed` | reserved; no longer emitted | — |
| `idle` | a world with no live session and no stop event | — |
| `unavailable` | nothing could be read | see `reason` |

`evidence` is prose naming what was observed. `reason` is prose for a
person, or null.

**`lifecycle.finalization`** — the builder's own account of what happened
after `session_stopped`, copied from the session record, or **null** on a
record written before 2026-09-06 or a session that never stopped.

| Field | Type | Notes |
|---|---|---|
| `state` | `"pending"`, `"complete"`, `"interrupted"` | `pending` with a live lock is `finalizing`; `pending` with no live lock is `interrupted` |
| `final_solve` | `"pending"`, `"solved"`, `"skipped"`, `"failed"`, `"unavailable"`, or null | null when no final solve was configured. `skipped` names why in `detail` (a stop request mid-walk, a hard stop during finalization) |
| `started_at`, `updated_at` | float | Tower clock |
| `detail` | string or null | prose: the error, or why the solve was skipped |

> **`finalizing` is now a live claim.** The builder keeps the writer lock
> until its final build is written, so a lock held by a running pid after
> `session_stopped` means a process is working, and
> `lifecycle.build_in_progress` is **`true`** in that state on the evidence
> of the lock. On records older than that change nothing on disk says so:
> there `stopped_unbuilt` still means only "the stored figures are not the
> final figures", `build_in_progress` is **`null`** (not `false`, which
> would be a claim), and `build_in_progress_unavailable_reason` explains.
> It is `false` while `receiving` and in every `interrupted` state, where
> the lock's holder is known to be gone.

**`lifecycle.photographic`** — added `2026-09-22`; **`null`** when the
lifecycle was computed from the record alone (no world store in hand), and
on the arms that read a live writer lock.

Where the world's PHOTOGRAPHIC representation -- the image-based room, which
is the final user-facing output -- has got to. `build_in_progress` answers
"is a process working this millisecond"; this answers "does this world still
owe a photographic room", which is the question the wearer is asking and the
one that survives the gaps between stages.

| Field | Type | Notes |
|---|---|---|
| `state` | one of the seven words below | never null when the block is present |
| `stage` | `"surface"`, `"appearance"`, or null | the stage the word is about |
| `detail` | string | prose; safe to show, names no path |

| `state` | Meaning | `lifecycle.state` |
|---|---|---|
| `complete` | the appearance stage finished; the photographic room exists | unchanged (`ready` on an ordinary walk) |
| `running` | a stage is running under a live process | `finalizing` |
| `owed` | unfinished, nothing working on it; the Tower finishes owed work the next time it is idle (no stream open, no capture worker alive), and at every start | `finalizing` |
| `failed` | a stage ran and failed. **Terminal** -- nothing more is coming | unchanged (`ready`): the world is saved, at whatever rung it reached |
| `unattempted` | the stages declined (no global solve, or the Tower's appearance setting is off) | unchanged |
| `never_recorded` | a world from before the photographic stages existed | unchanged |
| `unobservable` | the probe could not answer | `finalizing`, with `build_in_progress: null` |

> **Why three of these change `lifecycle.state` and four do not.** The three
> that do (`running`, `owed`, `unobservable`) all mean "this world is not
> finished being made", and a world in any of them must never read `ready` --
> that was the false **Saved** the 2026-09-22 Mac/iOS validation found at the
> stage boundaries (T2), over a failed build (T3), and after a broken probe
> (T4). The four that do not are settled: `complete` is finished, `failed` is
> finished badly, and `unattempted` / `never_recorded` are worlds that were
> never going to have one. Reporting a settled world as `finalizing` would be
> the opposite failure -- a wearer left waiting forever for a build that is
> not coming -- so `failed` keeps its `ready` and tells the truth in this
> block instead.

> **`never_recorded` is load-bearing for compatibility.** 165 of the 166
> worlds on the machine this was written for were built by a Tower with no
> photographic stages at all. They are finished, they are owed nothing, and
> they keep reading `ready`/`finalized` exactly as before. It is reached only
> when there is NO stage record AND no stage artifact on disk, and it is
> decided **without** consulting the liveness probe -- so a probe that breaks
> cannot relabel all of them at once.

> **`complete` is about the BUILD, not about what is being SERVED right now.**
> The two are almost always the same and deliberately are not the same fact.
> A world whose imagery was purged, or whose redaction label changed, keeps
> `stages.appearance: ok` -- the build did happen -- while
> `_servable_manifest` refuses it and the render revision reports
> `appearance.state: withdrawn`. That is the honest split: the lifecycle says
> the world is finished and nothing more is coming (which is true, and
> "Improving" would be a lie), the revision route says the imagery is not
> being served (which is also true), and the viewer falls to the best rung
> that exists and captions it. **Read `appearance.state` on
> `/worlds/{id}/render/revision`, never `photographic.state`, to decide what
> can be drawn.**

> **Additive.** iOS decodes this payload with `JSONSerialization`
> dictionaries and ignores unknown keys, so a client that does not read
> `photographic` behaves exactly as it did before -- except that it now hears
> `finalizing` where it used to hear a false `finalized`.

**`progress`** — or `null` with no session.

| Field | Type | Notes |
|---|---|---|
| `keyframes_accepted` | int | **the count the Tower actually keeps.** While live it is counted from the event journal, not from the session record, which still holds the zero written at session start |
| `keyframes_accepted_provenance` | `"measured"` | counted, not inferred |
| `keyframes_accepted_source` | string | `"event journal"` or `"session record"`. The journal is used until the session is stopped AND its record finalised — including when it `failed`, because a crashed session's record still holds the zeros written at start |
| `frames_observed` | int or **null** | **null while live.** An ordinary rejected frame writes no journal event, so this is genuinely not knowable until the session stops. `null ≠ 0` |
| `frames_observed_unavailable_reason` | string or null | why, when null |
| `rejected_by_reason` | object or null | histogram, available only after stop |
| `frames_rejected_wrong_size` | int | frames rejected for being a different size from the session's first frame, counted LIVE from the journal (the engine journals this rejection, unlike an ordinary one). The calibration is exact per resolution, so such a frame could never have produced a pose; before this was counted, a whole walk at the wrong rung read "Mapping" with a frozen keyframe count and then "Saved" with a truncated world. Additive (2026-09-14) |
| `frames_rejected_malformed` | int | frames rejected because they could not be decoded, counted live from the journal. Additive (2026-09-14) |
| `journal_corrupt_lines` | int | Unparseable lines in the event journal. **1 at the tail is routine** — a journal is appended without fsync, so a reader can arrive mid-write. More than that, or a count that does not clear, means corruption, and the keyframe count beside it is correspondingly low |
| `mapping_seconds` | float or **null** | **on the Tower's clock.** Do not derive this from a phone timer. **Null** if the Tower's wall clock moved backwards during the session — reported as unknown rather than clamped to `0.0`, because a plausible zero is worse than an absent value |
| `mapping_seconds_unavailable_reason` | string or null | why, when null |
| `mapping_clock` | `"tower"` | |

**`tracking`**

| `state` | Means |
|---|---|
| `good` | the most recent tracking event was a keyframe acceptance |
| `lost` | the most recent tracking event was a tracking loss |
| `unknown` | no keyframe accepted and no loss recorded |

**`limited` is never sent**, and `limited_ever_reported` is permanently
`false`. The nearest candidate threshold in Tower is documented in its own
source as an untuned placeholder and is not emitted as an event at all.
Emitting `limited` from it would present an unmeasured placeholder as a
calibrated judgment.

**`calibration`** — scope is **`"session"`**, always.

| `state` | Condition |
|---|---|
| `calibrated` | intrinsics present, finite, and physically possible |
| `uncalibrated` | no intrinsics were supplied |
| `unknown` | a source is declared but the numbers do not survive validation |

**`calibrating` is never sent** (`calibrating_ever_reported: false`).
Calibration is an offline procedure run before a session; there is no
in-session state to be in the middle of. There is deliberately **no
percentage** — a denominator nobody has defined.

Calibration is per **session**, not per world: intrinsics are
resolution-keyed, and two sessions of one world can legitimately differ. A
world-level calibration state would be a fabrication.

**`scale`**

| Field | Notes |
|---|---|
| `state` | Tower's vocabulary: `unknown` or `relative` in V1 |
| `unavailable_reason` | Set when this **session** has no build of its own. Scale lives on the world record and is earned by a build, so a session that was never built reports `unknown` rather than inheriting a scale another session earned |
| `semantics` | iOS's vocabulary: `"relative"`, or **`null`** when state is `unknown` |
| `unit` | `"world units"`, or **null** when there is no unit at all |
| `meters_per_unit` | always `null` in V1 |
| `allows_metres` | always `false` in V1 |

**`inferredMetric` and `measuredMetric` are unreachable in V1** and will
never arrive. `estimated` and `measured` have no code path that produces
them. **Never render any figure from this channel as metres.**

`state: "unknown"` means the reconstruction has **no unit at all** — a
strictly weaker claim than "internally consistent with an arbitrary unit".
Do not map it to `.relative`. When it is `unknown`, no distance figure is
sent at all.

**`geometry`**

| Field | Notes |
|---|---|
| `available` | true once **any** build has produced output for this session — including one that is now behind |
| `current` | whether that output reflects **every** keyframe accepted so far. **`false` is normal during a session that rebuilds as it goes** |
| `built_from_keyframes` / `keyframes_now` | how many keyframes the build consumed, and how many exist now. When they differ, the figures are correct for the first number |
| `stale_reason` | prose, when `current` is false; null otherwise |
| `representation` | **`"sparse point cloud"`** — Tower's own word. Display verbatim; never parse or match it |
| `element_count` | number of points, or **null**. Never `0` for "we did not build" |
| `element_name` | `"point"` |
| `is_incremental` | **`false`.** A build replaces the whole derived tree; it never emits a delta |
| `revision` | opaque; see §5 |
| `provenance` | `"inferred"` |
| `confidence` | **`null`.** Tower keeps per-keyframe and per-edge confidence but has never defined an aggregate for a whole reconstruction. A number here would be one nobody specified |
| `backend_id`, `built_at` | diagnostic |
| `unavailable_reason` | prose, when unavailable |

A manifest missing any of `input_digest`, `session_id`, `keyframes`,
`points`, `poses_solved`, `poses_refused` or `segments` is treated as
**absent**, not as geometry with null figures. "We have geometry" is not
asserted without the figures to show for it.

**`trajectory`** — a summary. **No pose array is sent**, per
`IOS-to-Tower.md` §1.4.

| Field | Notes |
|---|---|
| `current`, `built_from_keyframes`, `keyframes_now`, `stale_reason` | as for geometry |
| `pose_count` | poses carrying a position that is **evidence**. Read from the manifest's `poses_positioned`, which the build counts per segment: every solved pose, plus the anchor of each segment that solved something. **Not** `poses_solved` (that drops the origin of every segment) and **not** `keyframes - poses_refused` (that promotes the bare anchor of a segment which resolved nothing). See the changelog for why this changed |
| `poses_anchor` | how many poses were anchors. Reported beside the count, never folded into it, so an uncalibrated walk reads as "N segment origins, no trajectory" rather than as a trajectory |
| `poses_solved`, `poses_refused`, `keyframes`, `segments` | the underlying figures |
| `tracking_restarts` | how many times the tracker actually lost the world, **counted** from `tracking_lost` events. Never `segments - 1`: two independent causes open a segment, and only one of them is a tracking loss. On the 2026-09-09 walk 122 segments were 64 tracking losses, 57 solve-chain breaks and the one the session started with, while the phone rendered `segments - 1` under the words "Tracking restarted 121 times". `null` from a build with no journal to count |
| `chain_breaks` | how many times the solver could not extend its chain and opened a segment while tracking was healthy. `events.py` is explicit that a consumer must not read one as the wearer having lost the world; it does not move `last_tracking`. Diagnostics, not a headline figure |
| `path_length` | see below |
| `provenance` | `"inferred"` |

**`path_length`** is either

```json
{"available": true, "value": 2.853, "unit": "world units",
 "scale_semantics": "relative", "display": "2.853 world units",
 "provenance": "inferred"}
```

or `{"available": false, "reason": "..."}`. It is **refused** when:

- **any pose was refused** — a refused pose is a hole in the path, not a
  shorter path. Summing across it draws a straight line between the
  keyframes either side and calls that distance walked;
- **the session has more than one segment** — a segment break means
  tracking was lost, and poses either side share no coordinate frame. A
  total would sum incomparable distances;
- **the scale state is `unknown`** — the figure could not be labelled, so
  it is not renderable;
- the derived poses are unreadable.

Always render `display`, or `value` **with** `unit`. Never bare.

**`persistence`**

`state` is always `"saved"` — World Builder persists everything by
construction. `revision` is an opaque world identity. `location_disclosed`
is `false`: **no filesystem path is ever sent.** A Tower path is useless to
a phone and names a machine's layout to a remote consumer.

**`artifacts`**

`present` is **tri-state**: `true`, `false`, or **`null`** for "not
established" (the images directory could not be read). It is never `false`
merely because `images_purged_declared` is true — that flag makes rebuilds
refuse and deletes nothing, so it cannot support a claim that the imagery
is absent.

```json
{"keyframe_images": {"present": true, "count": 4, "redaction": "none",
                     "reredacted_set": null,
                     "fetchable": false, "reason": "..."},
 "images_purged_declared": false,
 "images_purged_verified": null,
 "images_purged_meaning": "..."}
```

**No imagery is offered, and none will be under this contract.** Since
2026-08-23 keyframes are **face-redacted before they are written**, and
`redaction` reports what the session recorded — e.g.
`faces-detected-and-filled/yunet-2023mar@0.30+plausibility3`, naming the
detector, its threshold, and the version of the plausibility gate its
detections pass before being filled (older sessions record `+plausibility2`,
`+plausibility1`, or no suffix at all; `docs/contracts/WORLD-BUILDER-IOS.md`
describes each gate). That is a **process** claim ("this detector's hits were
filled"), never an outcome claim: the detector has measured false
negatives on heavily occluded and ~90°-rotated faces. Sessions captured
before that date keep `none` forever.

`reredacted_set` is `null` unless the session was explicitly re-redacted
(`scripts/world_reredact.py --apply`,
`docs/contracts/WORLD-BUILDER-APPEARANCE.md` §6.5): then it is
`{"redaction": <the set's label>, "count": <images in it>}`, and builds read
that set. `redaction` and `count` still describe the stored `images/`, which
is never modified. Only an older label of the same family (the ungated rule,
`+plausibility1`, `+plausibility2`) can be re-redacted, only from verified raw
frames, and only so that fill is removed, never moved; `none` never is. It is
the same kind of process claim, and it does not make the imagery fetchable.

They stay unfetchable regardless. A best-effort filter is not grounds to
start shipping first-person imagery. `IOS-to-Tower.md` §5 requires an image whose treatment is not
`redacted` to be **withheld**, with no lenient default. So the channel
declares that the images exist and that they are not fetchable, and mints
**no id and no URL** — inventing a fetch scheme would be exactly the
fabricated contract that document refuses.

`images_purged_declared` is a **flag, reported as a flag**. It makes
rebuilds refuse; it does **not** delete anything, and a world carrying it
was verified to still have every JPEG on disk. `images_purged_verified` is
`null` because Tower cannot verify it. Never render this as "the imagery is
gone".

### 10.2 Real examples

These are actual payloads, captured from the implementation.

**Live session, `receiving`:**

```json
{
  "world": {"world_id": "eb9fe5739a2941ef95336696fc352e76",
            "display_name": "Live Room", "schema_version": 1,
            "created_at": 1787463092.883, "updated_at": 1787463092.890},
  "session": {"session_id": "45e21a3a6f6042adaced831b22eee530",
              "started_at": 1787463092.889, "ended_at": null,
              "end_reason": null, "frame_source": "synthetic",
              "capture_id": null, "retains_raw_imagery": true},
  "lifecycle": {"state": "receiving",
                "evidence": "a live process (pid 21280) holds the writer lock",
                "reason": null, "build_in_progress": false,
                "build_in_progress_unavailable_reason": null},
  "progress": {"keyframes_accepted": 3,
               "keyframes_accepted_provenance": "measured",
               "frames_observed": null,
               "frames_observed_unavailable_reason": "no event is written for an ordinary rejected frame, so this count is only known once the session stops",
               "rejected_by_reason": null,
               "mapping_seconds": 0.069, "mapping_clock": "tower",
               "time_basis": "tower-receipt"},
  "tracking": {"state": "good",
               "evidence": "the most recent tracking event was keyframe_accepted",
               "limited_ever_reported": false},
  "calibration": {"state": "calibrated", "source": "self_calibrated",
                  "calibrated_width": 480, "calibrated_height": 360,
                  "reprojection_rms_px": null, "view_count": null,
                  "calibrating_ever_reported": false, "scope": "session"},
  "scale": {"state": "unknown", "semantics": null, "meters_per_unit": null,
            "method": null, "confidence": "unknown", "unit": null,
            "allows_metres": false},
  "geometry": {"available": false, "current": false,
               "built_from_keyframes": null, "keyframes_now": null,
               "stale_reason": null, "representation": null,
               "element_count": null, "element_name": null,
               "is_incremental": false, "revision": null, "provenance": null,
               "confidence": null, "backend_id": null, "built_at": null,
               "time_basis": "tower-receipt",
               "unavailable_reason": "no build has run for this session, so no geometry exists"},
  "trajectory": {"available": false, "current": false,
                 "built_from_keyframes": null, "keyframes_now": null,
                 "stale_reason": null, "pose_count": null, "poses_solved": null,
                 "poses_refused": null, "poses_anchor": null,
                 "keyframes": null, "segments": null,
                 "path_length": null, "revision": null, "provenance": null,
                 "confidence": null,
                 "unavailable_reason": "no build has run for this session, so no poses exist"},
  "persistence": {"state": "saved", "revision": "c1e7c6a8857045d0",
                  "images_purged": false, "location_disclosed": false},
  "artifacts": {"keyframe_images": {"present": true, "count": 3,
                                    "redaction": "none", "fetchable": false,
                                    "reason": "these are unredacted first-person frames and no artifact transfer contract exists; a consumer must withhold imagery whose treatment is not stated"},
                "images_purged_declared": false,
                "images_purged_verified": null,
                "images_purged_meaning": "a declaration that rebuilds are refused for this world, not a verified deletion of the imagery"},
  "time_basis": "tower-receipt"
}
```

Note what a live session does **not** know: `frames_observed` is null, and
`scale` is `unknown` because no build has run.

**After stop, before build — `stopped_unbuilt`:** identical except

```json
"session": {"ended_at": 1787463092.958, "end_reason": "stop", "...": "..."},
"lifecycle": {"state": "stopped_unbuilt",
              "evidence": "capture ended and no build output exists for this session",
              "reason": "no geometry has been built for this session yet",
              "build_in_progress": null,
              "build_in_progress_unavailable_reason": "the writer lock is released before build() is called, and build() emits no event and writes nothing until it finishes, so a build in progress is indistinguishable on disk from one that never started and from one that crashed"},
"progress": {"frames_observed": 8,
             "frames_observed_unavailable_reason": null,
             "rejected_by_reason": {"insufficient_motion": 5}, "...": "..."}
```

`frames_observed` and the rejection histogram appear here and only here.

**After build — `ready`:**

```json
"lifecycle": {"state": "ready",
              "evidence": "capture ended and the stored geometry matches the keyframes",
              "reason": null, "build_in_progress": null,
              "build_in_progress_unavailable_reason": "..."},
"scale": {"state": "relative", "semantics": "relative",
          "meters_per_unit": null, "method": null, "confidence": "unknown",
          "unit": "world units", "allows_metres": false},
"geometry": {"available": true, "current": true,
             "built_from_keyframes": 4, "keyframes_now": 4,
             "representation": "sparse point cloud",
             "element_count": 1360, "element_name": "point",
             "is_incremental": false, "revision": "ecb09a6f0c3f3cdf",
             "provenance": "inferred", "confidence": null,
             "backend_id": "classical-sfm", "built_at": 1787464196.020,
             "time_basis": "tower-receipt", "unavailable_reason": null,
             "stale_reason": null},
"trajectory": {"available": true, "current": true,
               "built_from_keyframes": 4, "keyframes_now": 4,
               "stale_reason": null,
               "pose_count": 4, "poses_solved": 3,
               "poses_refused": 0, "poses_anchor": 1,
               "keyframes": 4, "segments": 1,
               "path_length": {"available": true, "value": 2.853251890377782,
                               "unit": "world units",
                               "scale_semantics": "relative",
                               "display": "2.853 world units",
                               "provenance": "inferred"},
               "revision": "e0787672c4faee47", "provenance": "inferred",
               "confidence": null, "unavailable_reason": null}
```

`pose_count: 4` with `poses_solved: 3` is not an inconsistency — the fourth
is the anchor, which has a position and is counted as neither solved nor
refused. That segment resolved, so its origin is a real point on the path.

The opposite case is the one that matters. A build with `poses_solved: 0`
reports **`pose_count: 0`**, however many anchors it produced, because a
segment that resolved nothing contributes an origin marker for an empty
coordinate frame rather than a camera position.

### 10.3 What becomes available only after Stop, and after build

| Available | Live | After stop | After build |
|---|:--:|:--:|:--:|
| world id, session id, display name | ✅ | ✅ | ✅ |
| lifecycle, tracking, calibration | ✅ | ✅ | ✅ |
| `keyframes_accepted`, `mapping_seconds` | ✅ | ✅ | ✅ |
| `frames_observed`, `rejected_by_reason` | ❌ null | ✅ | ✅ |
| `frames_rejected_wrong_size`, `frames_rejected_malformed` | ✅ (from the journal, live) | ✅ | ✅ |
| `scale.state` beyond `unknown` | ❌ | ❌ | ✅ |
| geometry: representation, element count | ⚠️ | ❌ | ✅ |
| trajectory: pose count | ⚠️ | ❌ | ✅ |
| trajectory: path length | ❌ | ❌ | ✅ |

⚠️ = available **only** with `world_build_session.py --rebuild-every N`,
and then always with `current: false` — see below.

### "Watch it build", concretely

With `--rebuild-every N`, geometry appears *during* the walk and grows.
A real run, 28 synthetic frames, `--rebuild-every 2`, sampled through this
channel:

| lifecycle | `keyframes_now` | `element_count` | `current` | `built_from_keyframes` |
|---|---|---|---|---|
| `unavailable` | — | — | — | — |
| `receiving` | 2 | null | false | null |
| `receiving` | 6 | **1360** | false | 4 |
| `receiving` | 10 | **2336** | false | 8 |
| `ready` | 10 | **2712** | true | 10 |

Point count grows monotonically and lands on the final figure. **Every
mid-walk row carries `current: false`**, because the next keyframe lands
before the viewer sees the build — that is not a defect, it is the normal
and permanent condition of a world being built, and the two keyframe
counts say exactly how far behind the figures are.

`path_length` stays unavailable throughout a live session even with
`--rebuild-every`: it reads the actual poses, and the store refuses a
derived tree that no longer matches its inputs. The counts come from the
build's own manifest and are correct for that build; the path length would
have to trust pose files that may be mid-rewrite, so it honestly refuses.

**Without `--rebuild-every`, geometry does not exist until after Stop**,
and `Start → Walk → Stop → the world appears` is the honest description.

---

## 14. Scene Understanding `live` payload

Contract: `scene_understanding.live/2026-08-27`.
Subscription pair: `("scene_understanding", "live")`.

### 14.0 What this cartridge is, in one paragraph

It answers **"what is around me, right now"**. It stores nothing, it
retains nothing, and it needs no purge because there is nothing to purge.
Every payload describes a moment; none of them describes a record. If you
want history, that is Object Memory, and it is a separate privacy review.

The result type is `live`, not `status`, and the difference is the payload
rather than the cadence. World Builder's `status` describes a BUILD — how
far it has got. This payload **is** the answer.

### 14.1 Availability

`available` is about **configuration**, never about current activity.

| `TOWER_SCENE_UNDERSTANDING` | `available` | `lifecycle.state` |
|---|---|---|
| unset / off | `false`, reason names the variable | — (no payload; the subscription is refused) |
| on, nobody has pressed Start | `true` | `"stopped"` |
| on, Start pressed, model loading | `true` | `"starting"` |
| on, observing | `true` | `"running"` |

A Tower that is enabled but stopped is **available**. Folding "not running
right now" into "unavailable" would tell a person their Tower cannot do
this, when in fact nobody has started it — and those call for opposite
responses.

### 14.2 Starting and stopping

The phone sends **nothing** to start a session; §6.2 of `IOS-to-Tower.md`
is explicit that opening a cartridge on the phone is silent, and a test
asserts the wire stays silent. Sessions are driven over HTTP:

| Route | Effect |
|---|---|
| `GET /scene` | the same payload the socket carries, plus `contract` |
| `POST /scene/start` | begin observing. Idempotent; resumes a paused session |
| `POST /scene/pause` | stop observing, KEEP the last scene, mark it not current |
| `POST /scene/resume` | observe again without reloading the model |
| `POST /scene/stop` | end the session and **discard** the scene |

**A phone does not call any of them.** `IOS-to-Tower.md` §6.2 is
explicit that opening a cartridge on the phone sends no verb. What it
does send is a `result_subscribe` for `scene_understanding/live`, and
since 2026-09-07 the session runs while **somebody is streaming AND
somebody is watching**:

- the **stream** (`stream_start` … `stream_stop` or a disconnect) is the
  feed. The last open stream closing stops the session **whoever started
  it** — frames come from nowhere else.
- a **watcher** (a live subscription) is the demand. A stream with no
  watcher — a phone using World Builder's or the CV Lab's camera — leaves
  the session `stopped`. The last watcher leaving (`result_unsubscribe`
  or its socket closing) stops the session and releases the detector.
- `POST /scene/start` is the **operator** path: it runs at once with or
  without a stream, survives watchers leaving, and ends on
  `POST /scene/stop` or on its last stream closing.

`lifecycle.follows_stream` reports whether the stream-and-watcher rule is
on; `TOWER_SCENE_AUTOSTART=false` leaves only the operator path.
`lifecycle.demand` reports the rule's inputs as counts. No demand event
resumes a Pause. A connection that never sent `stream_start` cannot end
anything.

All five answer `404` when the cartridge is not enabled. A `POST` that
returned `200` and did nothing is how an operator comes to believe a
physical test is running when it is not.

**Stop discards the scene, and this is load-bearing.** A scene held past
the end of a session is a claim about a room the wearer has left. No
staleness number makes that safe, because a client that renders counts
above staleness shows the room first. After Stop:
`scene_available: false`, `counts: null`, `people: null`, `where: null`,
and `scene_unavailable_reason` says the last scene was discarded.

**Pause is the deliberately different case.** The scene survives with its
age, and `lifecycle.scene_is_current` is `false`. That is iOS's
`lastKnown`, kept apart from `observing` as §4.7 asks.

### 14.3 Field reference

Every key is present in every state. A key that appeared and disappeared
would force a decoder to treat the field as optional and lose the ability
to tell "zero of these" from "this Tower did not say".

**Self-description — constant, and safe to assert against**

| Field | Value | Meaning |
|---|---|---|
| `claim` | `"visible-now-not-a-record"` | this is the present, not history |
| `identity` | `"anonymous-and-unpublished"` | nothing here identifies anyone, and no handle is published |
| `absence_means` | `"not-visible-to-this-cartridge"` | a zero is about this camera's forward cone, never about the room |
| `persistence` | `"none"` | nothing is written; there is nothing to purge |
| `frame_of_reference` | `"camera"` | everything positional is camera-relative and changes when the wearer turns their head |
| `time_basis` | `"tower-receipt"` | see §4 |

**`lifecycle`**

| Field | Type | Meaning |
|---|---|---|
| `state` | string | one of `states` |
| `states` | list of string | the closed vocabulary: `stopped`, `starting`, `running`, `paused`, `failed` |
| `session_id` | int | session-scoped, increments per Start. Two payloads with different `session_id` came from different tracking sessions and **must not be compared** |
| `scene_is_current` | bool | `true` only while `running` |
| `failure_reason` | string or null | non-null only in `failed` |
| `started_at` | float or null | Tower-receipt |
| `ready_at` | float or null | when the model finished loading |
| `loading_seconds` | float or null | non-null only while still loading |
| `load_overdue` | bool | the load has exceeded `load_overdue_after_seconds`. **Not a failure** — nothing can interrupt a blocking model load, and a first-run weight download is slow and still correct |
| `load_overdue_after_seconds` | float | `120.0` |
| `follows_stream` | bool | whether the stream-and-watcher rule is on (§14.2) |
| `demand` | object | why the session is or is not running: `demand.streams` (int, connections streaming frames), `demand.watchers` (int, live subscriptions), `demand.operator_hold` (bool, started by `POST /scene/start`), `demand.runs_when` (`"stream-and-watcher-or-operator"`). Counts only, never a token; volatile, so a subscriber joining does not mint an envelope |

**Freshness and flow**

| Field | Type | Meaning |
|---|---|---|
| `observed_at` | float or null | when the Tower **received the frame** this scene came from — not when the detector finished with it |
| `staleness_seconds` | float or null | now minus `observed_at` |
| `frames_offered` | int | frames handed to the session this session |
| `frames_observed` | int | frames that produced a scene |
| `frames_skipped` | int | frames displaced from the single slot because the worker was busy. **On the wire deliberately: a silently dropped frame is indistinguishable from a quiet room** |
| `frames_dropped_not_running` | int | frames arriving while stopped or paused |
| `decode_failures` | int | frames that would not decode |

**The scene**

| Field | Type | Meaning |
|---|---|---|
| `scene_available` | bool | false in four different situations, told apart by the next field |
| `scene_unavailable_reason` | string or null | stopped / still loading / failed / running-but-no-frame-yet |
| `detector` | string or null | the model that produced the counts |
| `score_threshold` | float or null | the detector's own floor |
| `reported_classes` | list of 13 strings | the **universe** of labels that can appear in `counts`. A label outside it has never been looked for, which is a weaker silence than "looked for and not seen" |

The thirteen, in full, so a consumer can look one up rather than infer it:
`person`, `chair`, `couch`, `bed`, `dining table`, `tv`, `laptop`, `book`,
`bottle`, `cup`, `keyboard`, `mouse`, `cell phone`. They are COCO class
names and carry COCO's meanings; `mouse` is the pointing device, `tv` is
any large display. The list is fixed at build time and is what keeps
`counts` and `where` bounded without a truncation rule.

| `counts` | `{label: int}` or null | one entry per `reported_classes`, present at `0` rather than omitted |
| `count_basis` | `"confirmed-tracks"` | counts come from the tracker, never from raw detections |
| `count_is_lower_bound` | bool, always `true` | see §14.4 |
| `count_limitations` | list of `{limitation, detail}` | `size-floor`, `recall`, `people-count-accuracy`, `departure-lag`, `field-of-view` |

> **All five ship, unconditionally.** This row listed three until
> 2026-08-27, omitted the noise-class disclosure entirely, and described
> the departure-lag one as conditional -- there is no branch in
> `tower/results/scene_understanding.py`, the tuple is a constant. A
> client rendering a fixed set of disclosures silently dropped the two
> it was never told about, and these are the fields that stop an
> undercount looking like a quiet room.

**`people`** — a count and an aggregate, never a list

| Field | Type | Meaning |
|---|---|---|
| `count` | int | confirmed person tracks in the camera's forward cone |
| `may_include_wearer` | bool, `true` | every `person` box in this platform's only real corpus is the wearer's own torso |
| `validated` | bool, `false` | no bystander footage exists on this host; nothing here has been checked against ground truth |
| `facing_wearer` | int or **null** | **never 0 when unmeasured.** Null means orientation never produced an estimate |
| `facing_answered` | bool | whether `facing_wearer` is a measurement |
| `facing_unavailable_reason` | string or null | why it is null |
| `facing_unknown` | int or null | people whose orientation is unknown |
| `oldest_estimate_seconds` | float or null | age of the stalest orientation estimate |
| `facing_note` | string | the wording, carried as data |

**`where`** — per-label side counts, **non-person labels only**

`{label: {left, centre, right, unknown}}`, integers, summing to that
label's count. Three fixed buckets per label, so the block is bounded by
the class list rather than by what is in the room.

Side counts rather than one side per label, because one side cannot
describe a chair on the left and a chair on the right, and picking one
would be a wrong answer where a refusal was available.

`where_excludes: ["person"]` and `where_excludes_reason` say why: a
per-person position, sampled repeatedly, is a movement trace.

**`tracks`** — always `null`, and unexpressibly so

`tracks_absent_reason` says why; `refused_entity_fields` is a list of
`{field, reason}` naming `track_id`, `box`, `facing`, `visible_eyes` and
`confidence`. There is no key anywhere in this payload that could hold an
entity.

This refuses `IOS-to-Tower.md` §4.1's session-scoped anonymous track
handle. The refusal is delivered as a value rather than as silence
because those are different instructions: "refused" means build a
different screen, and "not implemented yet" means wait for the next
Tower, and a client finding nothing to decode cannot tell them apart.

**`side_convention`** — the one convention iOS declares rather than
leaves open, answered

§4.3: *"a bearing has to be signed somehow to be usable and a silent
presumption is the dangerous version — a Tower signing the other way
would put every person on the wrong side of the wearer."* This payload
publishes no bearing, but `where` is a coarse signed bearing under
another name, so it states its convention: the wearer's own left and
right as the camera sees them, thresholds at 0.45 and 0.55 of frame
width, stream assumed unmirrored and nothing verifies that.

**`confidence`** — always `null`, with `confidence_absent_reason`

§4.1 asks for a confidence on every track. There are no tracks, and a
confidence attached to a COUNT would be an average of scores that did not
decide anything. `score_threshold` is the floor those scores had to
clear and is published instead. Stated rather than omitted, because
§0.2's rule is that an unprovenanced value is worse than an absent one.

**`observed_at_note`** — what `observed_at` actually is

Tower-receipt time. §0.3 holds `observedAt` and `receivedAt` separately
and never substitutes one for the other; a decoder mapping by field name
would make exactly that substitution, so the payload says it in words.

**`count_measurement`** — when the limitations above were measured

`measured_at`, `corpus_frames`, `corpus_captures`, `is_current`, `note`.
`is_current` is `false`: this platform's corpus grows continuously, and a
rate asserted in the present tense would read as current state.

**`people.facing_states_reported` / `facing_states_withheld`**

Only `facing_wearer` and `unknown` have buckets, so
`count − facing_wearer − facing_unknown` is an undifferentiated
remainder rather than a fifth category. `away_from_wearer` and `profile`
are withheld with a reason (`facing_states_withheld_reason`): a
per-person facing state narrows to one person's orientation the moment
only one person is in view.

**People, since 2026-09-07.** `people` also carries
`partial_bottom_edge` (int) with `partial_bottom_edge_note` — person
boxes cut off by the bottom edge with no head region, most often the
wearer's own body, kept out of `count`; `by_apparent_size` with buckets
`large`, `medium`, `small`, `unknown` and `apparent_size_note` — sizes
in the picture, never distances; and `orientation_method` (string or
null), `orientation_status` (`"experimental"`) and
`orientation_validation` (prose) — the facing stage is a face detector
on each tracked person's box, validated only on COCO stills.
`where` now includes `person` as side counts, and `where_excludes` is
empty. `single_person_note` (top level, constant prose) states the one
thing the aggregates cannot hide: with one person in view they describe
that person while they are in view; a client must not store a sequence
of payloads.

**`lifecycle.follows_stream`** — whether a stream together with a
watcher starts this session. See §14.2. **`lifecycle.demand`** — the
counts that rule is evaluated over, so a client can tell "nobody is
streaming" from "nobody is watching".

**`relations`** — always `null`

`relations_absent_reason` says why; `refused_relations` is a list of
`{relation, reason, reason_source}` covering `in_front_of`, `behind`,
`on`, `inside`, `near`, `nearer_than_same_class`. The `reason` is the
first sentence of each refusal and `reason_source` points at the full
measurement in the repository — the `in_front_of` entry alone runs to
1,500 characters of flip rates and sample sizes, and it is constant, so
publishing it whole put 1.5 KB of unchanging prose into every 2 s
heartbeat.

`withheld_relations` is the separate and weaker list: `left_of`,
`right_of` and `higher_in_view` are computable from 2-D boxes and are
TRUE. They are withheld rather than refused because they are
camera-relative and stop being true the moment the wearer turns their
head. A client must be able to tell "we can and will not" from "we
cannot, and here is the measurement". **There is no schema slot anywhere in
this payload that could hold one.** A refusal that depends on remembering
not to fill a field is not a refusal.

### 14.4 Every count is an undercount, and the payload says so

Measured against a `fasterrcnn_resnet50_fpn_v2` oracle over 14,128 real
frames: recall **0.306** for `person`, 0.497 for `cell phone`, 0.209 for
`tv`, and effectively **blind below ~2% of frame area** (0.000 under 1%).
The oracle shares COCO training data with the shipped model, so 0.306 is
an **upper bound**.

`SCORE_THRESHOLD` is not the lever: the F1 sweep is a plateau and
lowering it buys +0.05 recall for 1.7× the false boxes. The floor is the
model's.

**An undercount published without disclosure looks exactly like a quiet
room.** Render `count_is_lower_bound` somewhere a person will see it.

### 14.5 What is NOT on this wire, and why

`track_id`, bounding boxes, `normalised_x`, `view_offset`, per-person
facing state, `visible_eyes`, `visible_ears`.

The reasoning is not "minimise disclosure". Tower → phone is inside the
local-first boundary: **the phone sent the pixels**, so a count discloses
strictly less than the frame the phone already holds, and withholding it
while shipping frames would be theatre.

What is genuinely new is **joinability**. A stable `track_id` plus a
timestamp lets a recipient assemble the per-person dwell timeline this
cartridge refuses to keep — persists-nothing laundered onto the consumer.

**This refuses something iOS asked for.** `IOS-to-Tower.md` §4.1 requests
a session-scoped anonymous track handle, and §4.3 a signed bearing. V1
serves neither. The consequence is concrete and is not hidden: iOS cannot
render per-entity rows for people, only a count and an aggregate. If that
proves to be the wrong trade, it is a contract change made deliberately,
with a new identifier — not a field quietly populated later.

### 14.6 Cadence

Published at the standard 0.5 s poll with the 2 s heartbeat, **not at
frame rate**. `IOS-to-Tower.md` §4.8 asks the Tower to coalesce, and this
is that: the result sender shares a lock with the frame path, and
starving frame delivery is what forced World Builder's geometry onto HTTP.

`observed_at`, `staleness_seconds` and every `frames_*` counter are
excluded from `revision`, so an unchanged scene coalesces. `frames_observed`
is among them deliberately: a frame having been processed is not the same
event as the scene having changed.

### 14.7 Known limitations

1. **Orientation is off by default.** The pose model costs 956 ms per call
   on CPU — 11.5× the delivered frame interval — against 43 ms on CUDA,
   and `facing_from_keypoints` is unvalidated against ground truth.
   `TOWER_SCENE_ORIENTATION` enables it; `facing_wearer` stays `null`
   until it has succeeded once.
2. **The tracker's constants assume every delivered frame is seen.**
   `max_misses` is a frame count derived from a 1.0 s absence at 12 fps.
   A session that is skipping frames stretches what that means. Watch
   `frames_skipped`: at the measured cost (33 ms of work per 83.5 ms
   frame) it should stay near zero, and a sustained non-zero value means
   this Tower is overloaded and the counts are less stable than they look.
3. **`person` is the wearer's own torso** in every real capture on this
   host. `may_include_wearer` is `true` and `validated` is `false` for
   that reason. Nobody has yet worn these glasses in a room with a
   bystander and checked.
4. **No bearing, no distance, no world frame.** §14.5.
5. **No imagery.** This cartridge serves no image and has no artifact
   fetch contract. Nothing it produces may be displayed as a picture.

---

## 15. Document Memory `status` payload, and the library on HTTP

Contracts: `document_memory.status/2026-09-07` (this channel) and
`document_memory.library/2026-09-07` (HTTP).
Subscription pair: `("document_memory", "status")`.

### 15.0 Read this before building anything against it

**The premise is untested, not proven.** On 9,199 frames of real
first-person footage from these glasses, the page detector fired **six
times and every one was a false positive** — a venetian blind and a
backlit laptop keyboard. After `MIN_ROW_TRANSITIONS` was re-derived
against those same frames it fires **zero** times. And no capture on this
platform has ever contained a sheet of paper, so the detector has never
been shown a positive it was built for.

Separately, at the 360×640 the glasses deliver, EasyOCR returned **zero
dictionary words** across 919 sampled real frames that were dense with
screen text, at median confidence 0.056.

So: **an empty library is the expected result today.** Every response
carries `recording_limitations` saying so. A client that renders an empty
library as "no documents yet" is inviting a person to wait for something
that is not coming.

The measured fix is a **high-resolution still**, not a higher stream:
504×896 buys 0.886–1.000 word recall against 0.343–1.000 at the delivered
rung, and raising the stream would break World Builder's tracking. That
needs iOS/DAT work and is not in this contract.

### 15.1 The split: status here, documents on HTTP

The documents do **not** travel on this channel. Two reasons, and they
agree:

- **Size.** `tower/routes/ws.py` gives the result sender and the frame
  path one shared lock. Document text is the largest thing this platform
  could put on it.
- **Privacy.** `IOS-to-Tower.md` §3.2: "The list carries a character
  count, not the text, so a list of documents is not also a bulk transfer
  of every document's contents onto the phone."

| Route | Answers |
|---|---|
| `GET /documents?limit=&retention_days=` | recent documents, newest first, **no text** |
| `GET /documents/search?text=&limit=` | literal term matching (BM25), with snippets |
| `GET /documents/around?at=&window_seconds=` | documents observed within a window of an instant |
| `GET /documents/{document_id}` | one document **with** its pages and their text |
| `GET /documents-session` | the capture session's status |
| `POST /documents-session/{start,pause,resume,stop}` | control it |

All answer `404` when `TOWER_DOCUMENT_ROOT` is unset. That `404` is about
**configuration** and is never the answer to a query about a document,
which is answered with `answer: "no_observation"`.

### 15.2 The three answers

Every library response carries `answer`, one of:

| `answer` | Meaning | Render as |
|---|---|---|
| `matched` | documents were found | the list |
| `not_found` | the memory was searched and nothing matched | "Nothing matched" |
| `no_observation` | the memory holds nothing that could have matched | "Never observed" — **and say explicitly that this is not the same as the document not existing** |

Collapsing the third into the second lets a gap in what the glasses
happened to see read as a statement about the world. On this platform
that gap is the normal case, which is why `no_observation_note` is
carried beside it.

### 15.3 Library field reference

**Envelope**

| Field | Type | Meaning |
|---|---|---|
| `contract` | string | `document_memory.library/2026-09-07` |
| `claim` | `"a-page-was-in-view-and-was-ocred"` | The wire value, pinned as `DOCUMENT_CLAIM` in `tower/results/document_memory.py`. Until the 2026-08-27 unification this table quoted an older spelling that said a document "was read" — a claim the camera cannot establish, five keys above the note saying so. The code had been corrected and the doc had not. `test_the_contract_quotes_the_values_the_wire_actually_carries` now pins the two together, and asserts the retired spelling appears nowhere in this file |
| `identity` | `"same-page-by-text-and-look-within-library"` | since 2026-09-07 a later dwell whose first readable page carries the same words AND looks the same as a page already on record becomes a **sighting** of that record (`sighting_count`, `last_observed_at`, `total_observed_seconds`) rather than a new one. Anything weaker — same template different numbers, half-shared text, an unreadable re-view — stays a separate record. The earlier value said no such join was made; it was true then |
| `absence_means` | `"not-recorded-by-this-cartridge"` | |
| `time_basis` | `"tower-receipt"` | |
| `spatial_ref` | always `null` | this cartridge does not know where anything is |
| `answers` | list | the closed vocabulary above |
| `retrieval_kinds` | list | `recent`, `text`, `observed_within` |
| `semantic_retrieval` | bool, `false` | with `semantic_retrieval_unavailable_reason` |
| `recording_limitations` | list of `{limitation, detail}` | §15.0 |
| `imagery_treatment` | `"none-retained"` | no image is served; nothing here may be displayed as a picture. `"raw-ephemeral-not-served"` was the spelling here until 2026-08-27 and the code names it retired; the wire has never carried it |
| `retention` | object | `requested_days`, `writer_window_days` (**always null**), `writer_window_unavailable_reason`, `policy` |
| `answer`, `no_observation_note`, `documents_in_memory`, `document_count`, `documents` | | |

`retention.writer_window_days` is honestly `null`: unlike Object Memory's
store, `DocumentStore` persists no retention manifest, so a reader cannot
learn the window its writer used. A `retention_days` query parameter
**narrows this read and can never widen what was kept**.

**Imagery, on every response and every record**

| Field | Meaning |
|---|---|
| `imagery_treatment` | `none-retained` or `raw-persisted`. It varies with the fact; a constant here said the same thing whether or not an image existed |
| `imagery_ios_state` | `rawEphemeral` — the strictest of §5's three states that applies, named in iOS's vocabulary so the mapping is the Tower's decision and not the phone's guess. Never `redacted`: this platform performs no redaction |
| `imagery_served` | always `false`. No route serves an image |
| `imagery_note` | says the above in words |
| `privacy_tag_vocabulary` | the closed set `privacy_tags` draws from, published so a client can pin it |
| `recording_measurement` | `measured_at`, `corpus_frames`, `corpus_captures`, `is_current`, `note` — when the limitations were measured, and on what |
| `semantic_retrieval_alternative` | what to do instead. iOS routes typed free text to `.semantic`, so its primary input path has no Tower route; saying only "no semantic retrieval" leaves it to guess |
| `snippet_max_chars` | on a search result: how much verbatim text a match may carry |

**Per document, in a list**

`document_id`, `claim`, `identity`, `title`, `title_is_derived`,
`summary_available`, `confidence`, `confidence_basis`, `observed_at`,
`recorded_at`, `observed_seconds`, `sighting_count`, `last_observed_at`,
`total_observed_seconds`, `pages_observed`, `pages_readable`,
`text_availability`, `end_reason`, `timing`, `provenance`,
`retains_raw_imagery`, `redaction`, `imagery_treatment`, `privacy_tags`,
`schema_version`.

**Sightings.** `observed_at`, `observed_seconds`, `provenance` and every
page's `source_seq` describe the FIRST observation and are never
rewritten. `sighting_count` is how many times the record was observed
in total (one for a record seen once), `last_observed_at` the most
recent of those, `total_observed_seconds` their sum. `pages_readable`
counts the pages whose text cleared the readability floor; a record can
hold pages that OCR looked at and could not read.

> **The prose fields are NOT here. They are on the envelope, once.**
> `record_notes` carries five entries — `summary_withheld`,
> `observed_seconds`, `timing`, `imagery_retention`, `joinable` — and a
> record no longer repeats any of them. That hoist is described a few
> paragraphs down; this list, and three other places in this file,
> continued to name the per-record spellings (`summary_withheld_reason`,
> `observed_seconds_note`, `imagery_retention_note`, `joinable_note`)
> until 2026-08-27. **None of those keys has ever shipped since the
> hoist**, so a non-optional Swift decode of any of them throws
> `keyNotFound` on every document.
>
> The hoist was measured, not tidying: 68% of each record was constant
> prose, 2,351 bytes repeated per record, and a 200-document listing was
> 488 KB with ~313 KB of it the same five sentences two hundred times.
> After: 249 KB, 1,218 bytes per record, **nothing dropped**. Caveats were
> hoisted, never deleted — deleting one to save bytes is the one saving
> this contract may not make.

Three of those deserve a sentence:

- **`observed_seconds`** is how long the region was in view. It is **not**
  a claim that the wearer looked at it, noticed it, or read it — the
  camera cannot establish any of those. Render it as "In view 45 s".
  `record_notes.observed_seconds` carries that qualification as data,
  on the envelope rather than on every record.
- **`title`** is lifted from the document's own first text region.
  `title_is_derived: true`, and it is CLIPPED to `title_max_chars` (60)
  with an ellipsis. iOS asks for a title in the list knowing it comes
  from the document, and one line is a label — but 90 characters times a
  200-document listing is 18 KB of verbatim first lines for a caller that
  asked no question. A null title must render as a description of the
  RECORD — "Untitled document" — never as an invented name.
**`record_notes` carries the caveats once, at the envelope.** Five
sentences used to be repeated on every document — the "in view, not
read" qualification, the summary's provenance, the clock, the
capture-side imagery lifetime, and the joinability of the frame
reference. A 200-document listing was 488 KB with two thirds of it the
same sentences two hundred times. They are keyed by the field they
qualify: `observed_seconds`, `summary_withheld`, `timing`,
`imagery_retention`, `joinable`. Render a document with `record_notes`
beside it; none of them may be dropped, because each is a caveat and
deleting a caveat to save bytes is the one saving this contract may not
make.

- **`summary` is NOT in the list.** The stored summary is the document's
  first forty words **verbatim** — an excerpt, not a paraphrase — and
  forty words per document across a list is exactly the bulk transfer
  §3.2 exists to prevent. `summary_available` says it exists;
  `GET /documents/{document_id}` serves it beside the pages it came from,
  with `summary_is_verbatim_excerpt: true`.

**`text_availability`** — the typed form of iOS's
`unknown` / `notReadable` / `extracted(characterCount:)`:

| `state` | `character_count` | Meaning |
|---|---|---|
| `unknown` | null | the record has no pages |
| `not_readable` | 0 | **a real answer**: we looked and found no readable text |
| `extracted` | int > 0 | text was captured; fetch the document to read it |

A **search** result additionally carries `score`, `matched_terms` and a
`snippet` — a bounded window around the matched term, capped at
`snippet_max_chars` (48). It is evidence, not an excerpt: a match with no
evidence is a number a client has to trust, and `DocumentQueryEvidence`
exists on the iOS side for exactly this. The cap is published beside it,
and it is what keeps a 50-result search from becoming a bulk transfer.

**`provenance`** — a pointer into a recording, not a place

`kind: "frame-reference"`, `spatial_ref: null`, `capture_id`,
`capture_id_validated` (**always `false`** — nothing checks that the
capture still exists), `page_source_seqs` (the sequence number of each
frame actually read, at most two per document), `pages_without_source_seq`,
`frames_considered`, `frames_ocred`, `world_id`, `world_session_id`,
`imagery_retention: "capture-side"` and `joinable: true`, both
defined by `record_notes` on the envelope.

**That last pair is said out loud rather than left to be noticed.** This
block IS joinable: a capture id, frame sequence numbers and a timestamp
locate this reading in a recording on disk, and the link is durable
across sessions — which is precisely what Scene Understanding refuses to
hand anyone. The two cartridges differ here on purpose. A document is a
record; a scene is not.

The capture id is stamped when the DWELL STARTED, not when the record was
written. A `stream_start` arriving mid-reading does not move that reading
onto the new capture, and a `stream_stop` does not erase it — both used
to happen, and both produced a `page_source_seqs` pointer that resolved
into the wrong recording.

`frames_considered` will be much larger than `frames_ocred`. That is the
architecture: OCR costs ~1.19 s a page against a 0.771 ms per-frame
detection, and at most two frames per dwell are ever read.

**`timing`** — `time_basis`, `source` (`capture-journal` /
`assumed-interval` / `mixed`), `assumed_frame_interval_s`, `note`. A
duration derived from an assumed interval is a reconstruction and must
not be rendered identically to a measured one.

**A search result carries four more fields**

| Field | Type | Meaning |
|---|---|---|
| `match_kind` | `"lexical"` | literal term matching. Never `"semantic"`; see `semantic_retrieval` |
| `searched_documents` | int | how many records were scored. Compare with `documents_in_memory`: a difference means the retention window narrowed this read |
| `searched_pages` | int | how many PAGES were scored. Since 2026-09-07 the page is the unit of scoring and a document's score is its best page's |
| `min_score` | float | the BM25 floor a page had to clear. Default `0.1` |
| `sufficient_evidence` | bool | whether the memory held enough to answer at all. `false` with `answer: "no_observation"` is an empty memory; `false` with `not_found` is a query whose terms nothing contained |
| `match_tolerance` | string | says in words what `fuzzy` below means: a query term of five or more characters also matches a token within one edit of it or one it begins, weighted below an exact match. OCR reads "Kubernetes" as "Kubemetes" often enough to matter |

Each matched document additionally carries `score` (rounded to 4 places),
`matched_terms`, `snippet` — `snippet_max_chars` characters around the
first matched term, **so an answer is always traceable back to text that
was actually captured** rather than to a number a client has to trust —
`page_index`, the page the score and snippet came from, and `fuzzy`,
true when any matched term was a near-miss rather than the word itself.

`snippet_max_chars` is **48**, and it is on the envelope so a client reads
the bound rather than assuming one. This paragraph said 160 until the
2026-08-27 unification, which contradicted both the value in
`tower/results/document_memory.py` (`SNIPPET_MAX_CHARS = 48`) and §15.3
four hundred lines above it. Read the field, not the prose.

**One document carries pages, and pages carry these**

| Field | Type | Meaning |
|---|---|---|
| `page_index` | int | position within this document |
| `text` | string | what OCR read. Empty string is `not_readable`, not "no page" |
| `text_source` | `"ocr"` | an enum of one today, stated so a second source is a visible change |
| `region_count` | int | text regions the recogniser returned. `0` with an empty `text` is the readable-nothing case |
| `mean_region_confidence` | float or null | null means no region had a score |
| `min_region_confidence` | float or null | the worst region on the page |
| `confidence` | string | derived from the MEAN — one hard word should not condemn a page |
| `sharpness` | float or null | the frame-quality measure that chose this keyframe |
| `squareness` | float or null | how square-on the page was |
| `source_seq` | int or null | the frame this page was read from. Null on a record written before provenance existed |
| `observed_at` | float or null | Tower-receipt time of that frame |
| `observation_count` | int | how many separate views of this page were merged into it. Two readings of one page during one dwell is one page with a count of two, not two pages |
| `readable` | bool | whether `text` cleared the readability floor (at least one word at a mean confidence above the recogniser's noise floor). `false` with `region_count > 0` is a page OCR looked at and could not read — a different fact from never looking |
| `box_count` | int | text boxes the detector found in the region this page was read from. Evidence that text was there even when it was not readable |
| `image_kept` | bool | whether a page image exists on this Tower's disk. **False unless page images were explicitly enabled**, which is off by default, must stay off, and has no configuration path from a web process: this platform has no redaction, so a stored page image is an unredacted photograph of what the wearer was reading |
| `image_served` | bool | always `false`. A BOOLEAN and not the path, which told a reader where in the store to find that photograph — disclosure with no consumer, since no route resolves it and none may |

The document object also carries `word_count`, and the two summary
qualifiers `summary_is_model_output` and `summary_is_verbatim_excerpt`.

**`coverage`** — how much of the document was captured, not how much
exists:

| Field | Type | Meaning |
|---|---|---|
| `pages_observed` | int | pages this record holds |
| `pages_total` | **always null** | the camera cannot know how many pages a document has |
| `pages_total_note` | string | says exactly that |
| `words_captured` | int | words across all pages |
| `low_confidence_pages` | list of int | page indices whose confidence is `low` or `unknown` |

### 15.4 The session `status` payload, on this channel

| Block | Meaning |
|---|---|
| `contract_note` | a string pointing at the HTTP routes, carried IN the payload so a client that reads only this channel still learns the documents are elsewhere |
| `library` | what is on disk, **regardless of whether anything is running**: `available`, `document_count_unfiltered`, `retention_applied: false`, `unavailable_reason`, `newest_observed_at`, `revision`, `bytes`, `location_disclosed: false` |
| `session` | the live capture, or `{state: "unavailable", reason: ...}` when `TOWER_DOCUMENT_CAPTURE` is off or the session could not be constructed (the OCR extra is not installed, most likely; `reason` says which) |

**`library.revision` is the live-update trigger.** It changes whenever
the journal changes — an append, a sighting merged onto an existing
record, a prune — and never otherwise. A client holding a listing
re-fetches `/documents` (or re-runs its standing search) when this
moves; the count alone would miss a sighting, and a prune plus an append
in one tick. It is an opaque integer (the journal's modification stamp),
not a clock, and `newest_observed_at` now reflects the most recent
SIGHTING of any record, not only first observations. Polling
`/documents` on a timer remains the wrong design: it re-parses the
journal for nothing.

`library.document_count_unfiltered` and `session.library_count` are
DIFFERENT QUANTITIES and are named apart for that reason. The first
counts every parseable record on disk; the second is the same count
through the session's retention window and is refreshed only by a prune.
They diverge whenever records exist past the window.

`session.follows_stream` reports whether `stream_start` starts this
session. It defaults to FALSE for Document Memory — the opposite of Scene
Understanding's — and the asymmetry is the difference between the two
cartridges: this one writes, and a session that persists what a wearer
read gets an explicit start. `TOWER_DOCUMENT_AUTOSTART=true` changes it.

**The session block keeps its shape.** When no session exists every field
is present and `null`, `state` is `"unavailable"`, and `states` carries
the full vocabulary including that value. A block that changed shape
forced a decoder to make thirty fields optional and to handle a `state`
its own enum denied existed — in the one shape that did not carry the
enum.

**The five session routes carry the full envelope**, not a bare status:
`contract`, `claim`, `identity`, `absence_means`, `time_basis`,
`recording_limitations`, `recording_measurement` and the imagery fields,
with the status under `session`. A client that polls the session and
never calls a listing would otherwise never learn that an empty library
is the expected result here.

`session.engine` and `session.recogniser` both name the text recogniser
in use; `engine` is the generic lifecycle field every live session
carries, and `recogniser` is this cartridge's name for the same thing.
They agree by construction.

A Tower with a library and no session is a **normal** configuration — it
serves documents recorded elsewhere and records nothing itself.

`session` carries the same lifecycle block Scene Understanding uses
(`state`, `states`, `session_id`, `failure_reason`, `started_at`,
`ready_at`, `loading_seconds`, `load_overdue`, `frames_offered`,
`frames_observed`, `frames_skipped`, `frames_dropped_not_running`) plus:
`recogniser`, `ocr_device`, `capture_id`, `capture_id_validated`,
`in_dwell`, `dwells_started`, `pages_detected`, `documents_recorded`,
`documents_resighted`, `dwells_unreadable`, `last_document_id`,
`last_document_at`, `flushed_document_id`, `keeps_page_images`,
`follows_stream`, `idle_stop_seconds`, `idle_stop_pending`,
`retention_days`, `documents_pruned`, `retention_incomplete`,
`library_count`, `library_soft_limit`, `library_over_soft_limit`,
`library_soft_limit_note`.

**`ocr_device`** is where the reader actually loaded — `"cuda"` or
`"cpu"` — once the session is running, null before. `TOWER_DOCUMENT_DEVICE`
asks; this reports. **`documents_recorded`** counts NEW records this
session wrote; **`documents_resighted`** counts dwells this session
merged onto a record that already existed (see `identity`);
`last_document_id` names whichever happened last. **`dwells_unreadable`**
counts dwells whose every page OCR looked at and could not read — they
are persisted, with `readable: false` on each page, and a session where
this climbs while `documents_recorded` does not is a session pointed at
something that is not a page.

**`idle_stop_seconds` / `idle_stop_pending`.** A running session whose
stream closed starts a timer; if no stream reopens within
`idle_stop_seconds` the session stops itself, so a phone that
disconnected for good does not leave ~1.4 GB of OCR model resident on a
Tower whose next cartridge wants it. Unlike a Stop a person presses, an
idle stop does NOT read a dwell that was still open: that dwell ended
when the stream did, and the Tower logs that it was dropped.
`idle_stop_pending` is true while that timer runs; a reconnect or a
frame cancels it. A Stop or Pause a person presses reads at most four
frames of an open dwell, newest first.

**`retention_incomplete`** is reported rather than logged: a deletion that
quietly failed looks exactly like one that was kept.

**`library_soft_limit` is reported, never enforced.** This session evicts
by AGE only. Deleting a wearer's memories because a count grew is a
policy decision, not a cleanup, and this lane declined to make it.

**Stop KEEPS what was recorded**, unlike Scene Understanding's Stop. A
record of what was read is exactly as true after the session ends. A dwell
in progress is **flushed**, not dropped: a wearer still reading when a
session stops has read something.

### 15.5 Known limitations

1. **The premise is untested on a physical page.** §15.0. The pipeline is
   validated on rendered pages with known text; someone has to wear the
   glasses and read a page.
2. **Identity is one rule, and it is conservative.** A later dwell joins
   a record only when its first readable page has the same words (token
   overlap ≥ 0.70, numbers agreeing) AND the same look (perceptual hash
   within 6 bits). Two invoices on one template, two pages of one book,
   or an unreadable re-view stay separate records. A multi-page dwell
   that matches on its first page adds a sighting and keeps the record's
   pages; its later pages are not merged in. Records are not grouped
   into "documents" across dwells: a dwell is the record.
3. **No pagination.** `limit` is the only bound, capped at 200.
4. **No semantic retrieval.** BM25 over literal terms with bounded
   near-miss tolerance. Calling it semantic would be an overclaim, and a
   client routing a description here will get a lexical answer.
5. **No redaction exists.** `redaction` is an enum of one, `"none"`, and
   that is the honest value for imagery this platform cannot redact. Page
   images are OFF by default and must stay off.
6. **`retention.writer_window_days` is null.** §15.3.
7. **The search corpus is rebuilt when the journal changes**, and cached
   on the journal's stamp between changes. There is no persistent index;
   the JSONL journal is the only truth and a newly persisted page is
   searchable on the next query.
8. **No capture timestamp.** Everything is `tower-receipt`. This is a
   cross-boundary blocker, not a Tower gap.
9. **A text-detector false positive reaches OCR.** A keyboard or a
   screen held steady long enough is a dwell; what OCR makes of it is
   kept as a page with `readable: false`, and the session counts these
   as `dwells_unreadable`.

---

### 15.6 Key index — the nested names, spelled out

A key that this document mentions only in prose is a key a consumer has
to guess at, and a test that matched prose would not notice. These are
the nested names on both cartridges' wires that are otherwise only
implied by the block that contains them.

**Inside every `count_limitations` and `recording_limitations` entry**

| Key | Meaning |
|---|---|
| `limitation` | a short slug naming the class of limit — `size-floor`, `recall`, `field-of-view`, `noise-classes`, `departure-lag`, `detection-rate`, `no-validated-positive`, `resolution`, `resolution-remedy-is-not-a-fix` |
| `detail` | the measurement, in a sentence a person can be shown |

**Inside every `refused_relations` entry:** `relation` — the name of the
relation that is refused. Inside every `refused_entity_fields` entry:
`field` — the name of the entity field that is refused.

**Inside `where`:** one entry per non-person reported class, each with
`left`, `centre`, `right` and `unknown` — integer counts of confirmed
tracks of that label on each side. `where_excludes` is the list of labels
that never appear here, and is `["person"]`.

**Inside a document `provenance` block**

| Key | Meaning |
|---|---|
| `kind` | always `"frame-reference"`. A pointer into a recording, not a place |
| `imagery_retention` | always `"capture-side"`, defined by `record_notes.imagery_retention` |
| `joinable` | always `true`, defined by `record_notes.joinable`. This block locates a reading in a recording; the link is durable across sessions, unlike anything Scene Understanding publishes |

**Inside `query`**, which echoes what was asked so a response is
self-describing:

| Key | Meaning |
|---|---|
| `kind` | one of `retrieval_kinds`, or `"document"` on the single-document route |
| `limit` | the effective cap, after the route's own bound |
| `text` | the search terms, on a `text` query |
| `centre` | the instant an `observed_within` window is centred on |
| `window_seconds` | the half-width of that window |
| `document_id` | the id asked for, on the single-document route |

**Inside `pagination`**

| Key | Meaning |
|---|---|
| `supported` | always `false`. There is no cursor |
| `bound` | `"limit"` — the name of the thing that does bound a listing |
| `reason` | how to detect truncation on each query kind |

**Inside `library.bytes`:** `journal`, `images` and `total`, in bytes.
`journal` is the JSONL; `images` is the page-image directory, which is
empty unless page images were explicitly enabled; `total` is their sum.
`retention_applied` on the `library` block is always `false` — that count
is unfiltered, which is why it is named `document_count_unfiltered`.

**On the single-document response:** `document` is the record itself, and
`pages` is the list of `PageObservation` views inside it. It is the only
place either carries text.


---

## 16. World Builder: known limitations

1. **Polling latency.** Up to one poll interval (0.5 s) plus the builder's
   own write. Not a frame-rate signal.
2. **No `finalizing`** — see §10.1. Not observable.
3. **No error reason for a crashed session.** `end_reason: "error"` exists
   in Tower's vocabulary but no code path writes it. A crashed builder is
   detected via the stale lock instead.
4. **`frames_observed` has no live source.** Fixing it would mean World
   Builder emitting a per-frame event, which would turn a bounded journal
   into a per-frame one.
5. **No imagery, and no artifact fetch contract.** §10.1.
6. **Most figures here came from synthetic renders.** One physical walk
   has now been recorded and analysed (2026-08-24); where a number in
   this document came from it, it says so.
7. **`build_completed` is unreachable** in Tower's event vocabulary and is
   not exposed. Do not wait for it.

---

## 17. Changelog

Identifiers are opaque and compared for equality only (§2). This section
exists so a mismatch can be *understood*, not so it can be computed.

### `scene_understanding.live/2026-08-27` — new

First contract for this cartridge. Nothing preceded it on any wire.

Dated 2026-08-27 rather than the 2026-08-26 of the design document it
implements, because the payload that shipped is not the payload that was
designed: `where` carries per-label SIDE COUNTS rather than one side per
label — one side cannot describe a chair on the left and a chair on the
right — and a `lifecycle` block was added, because this cartridge has a
Start and a Stop that World Builder's file-reading status did not.

The design was explicitly "designed, not implemented", so no consumer was
broken. Minting the date the agreement actually reached a wire is the
whole discipline these identifiers exist for.

### `document_memory.status/2026-09-07` and `document_memory.library/2026-09-07` — identity, sightings, live updates

Both identifiers moved because `identity` changed meaning. The
2026-08-27 value, `"no-document-identity-across-sightings"`, described a
cartridge that made no join across dwells; this one joins a later dwell
onto an existing record when the words and the look both agree, and a
decoder written against the old value would render a merged record as a
single observation. Additive on top of that, all with defaults a
2026-08-27 decoder would have ignored:

- per document: `sighting_count`, `last_observed_at`,
  `total_observed_seconds`, `pages_readable`;
- per page: `readable`, `box_count`;
- per search match: `page_index`, `fuzzy`; on the search envelope:
  `searched_pages`, `match_tolerance`;
- on the status `library` block: `revision`, the live-update trigger;
- on the status `session` block: `ocr_device`, `documents_resighted`,
  `dwells_unreadable`, `idle_stop_seconds`, `idle_stop_pending`.

The 404 bodies still name `TOWER_DOCUMENT_ROOT` and
`TOWER_DOCUMENT_CAPTURE`, which the phone string-matches, even though the
root now has a managed default and "unconfigured" is reachable only by
`TOWER_DOCUMENT_ENABLED=false`. The session verbs still answer `200`
with the full envelope; the phone renders any other status as a
transport failure.

### `document_memory.status/2026-08-27` and `document_memory.library/2026-08-27` — new

First contracts for this cartridge. Two identifiers rather than one,
because they govern different surfaces with different failure modes: the
`status` payload rides this channel and is small and pushed; the
`library` payload is bulk text on HTTP and is pulled. A change to one is
not a change to the other.

### `world_builder.status/2026-09-10`

Supersedes `/2026-09-06`. No word was added and nothing an older phone
decodes was removed; what changed is that a `model_state` it already
implements now arrives in a state it did not before. A
`lifecycle.state: "stopped_unbuilt"` with no drawable figures --
`geometry.element_count` and `trajectory.pose_count` both zero or absent
-- projects to `interrupted` rather than `finalizing`, because there is
nothing to wait for -- a phone told to wait there waits forever. The gate
is the figures and not `geometry.available`: `engine.build` writes a
derived tree even when it solved nothing, so `available` is true for a
world with zero points in it, and gating on it told a wearer to keep
waiting for exactly that (§10.1). A client that switches on
`model_state` needs no change; one that switched on `lifecycle.state`
and assumed the old projection should read the two counts beside it,
the way iOS `WorldEvidence.hasGeometry` does.

Supersedes `world_builder.status/2026-08-23`. **One field changed
meaning**, which is why the identifier moved rather than staying put for
an additive change.

**`trajectory.pose_count` counts something different.** It was
`keyframes - poses_refused`. The build counts a segment ANCHOR as
neither solved nor refused, so that subtraction promoted every anchor to
a camera position — and an anchor is definitional, not measured: an
identity rotation at the origin.

The first physical walk (2026-08-24) made the consequence concrete. Its
manifest reads `backend_id: "unposed", keyframes: 155, poses_solved: 0,
poses_refused: 119, points: 0, segments: 36`. Nothing was reconstructed:
no intrinsics exist for this camera, so the backend that solves poses
could not run and withheld every one. The 36 remaining rows were segment
anchors, all at the same point. The channel reported **"pose_count: 36"**
and a phone displayed *"Camera poses: 36"*.

`pose_count` is now read from the manifest's `poses_positioned`, which
the build counts per segment: every solved pose, plus the anchor of each
segment that solved something. For a build with `poses_solved: 0` it is
**0**, whatever the anchor count.

**`trajectory.poses_anchor` is new**, reported beside the count and never
folded into it, so the same walk reads as "36 segment origins, no
trajectory".

**Worlds built before this change** carry no `poses_positioned`. The
producer falls back to the old arithmetic for them rather than blanking
the trajectory of every world already on disk; for the classical builds
those worlds are, the two agree except across a segment that resolved
nothing.

**What a consumer must do.** Nothing, if it does not pin the contract
identifier — the key set is unchanged apart from the added
`poses_anchor`. A consumer that *does* pin it will be refused with
`contract_mismatch` and must adopt the new identifier deliberately,
having read the paragraph above. That refusal is the point: this build
serves a figure that means something different from what the old
identifier promised.

---

## 18. Where the code is

| Concern | File |
|---|---|
| Contract identifiers | `tower/results/contracts.py` |
| Envelope, revision hashing | `tower/results/envelope.py` |
| Capability declaration | `tower/results/registry.py` |
| Fan-out, coalescing, bounds | `tower/results/publisher.py` |
| World Builder producer | `tower/results/world_builder.py` |
| WebSocket protocol | `tower/routes/results_ws.py` |
| `GET /cartridges` | `tower/routes/cartridges.py` |
| Scene Understanding producer | `tower/results/scene_understanding.py` |
| Scene Understanding session | `tower/scene/live.py`, `tower/live_session.py` |
| Scene Understanding control | `tower/routes/scene.py` |
| Document Memory producer | `tower/results/document_memory.py` |
| Document Memory session | `tower/document_memory/live.py`, `tower/live_session.py` |
| Document Memory library and control | `tower/routes/documents.py` |
| Experimental CV Lab producer | `tower/results/experimental_cv.py` |
| Experimental CV Lab control | `tower/routes/cv_lab.py`, `tower/routes/cv_lab_ws.py` |
| Object Memory session control | `tower/routes/sessions.py`, `tower/cartridge_session.py` |
| Object Memory store and imagery | `tower/object_memory/`, `tower/routes/observations.py` |
| Live cartridge construction | `tower/cartridge_runtime.py` |
| Wiring | `tower/main.py`, `tower/config.py` (`TOWER_WORLD_ROOT`, `TOWER_SCENE_UNDERSTANDING`, `TOWER_SCENE_DEVICE`, `TOWER_SCENE_ORIENTATION`, `TOWER_DOCUMENT_ROOT`, `TOWER_DOCUMENT_CAPTURE`) |

Tests: `tests/test_result_channel_{protocol,bounds,truthfulness,isolation}.py`
and `tests/result_channel_fixtures.py`. The two newer cartridges are
driven end to end through the real app in `tests/test_scene_wire_e2e.py`
and `tests/test_documents_wire_e2e.py`; their sessions are tested in
isolation in `tests/test_scene_live_session.py`.
