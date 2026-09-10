# Tower → Mac: the unified cartridge contract

**Branch:** `integration/tower-unified-cartridges-v1`
**Written:** 2026-08-27, against the merged tree, by reading the wire rather than the lanes' documents.

This is the **one authoritative Mac-facing summary** of every cartridge the
Tower now serves. Four lanes were developed independently and merged here;
each wrote its own contract document, and three of them describe a Tower
that no longer exists in isolation. Where this file and a lane's own
document disagree, **this file is the one that was checked against a
running process.**

It is a synthesis, not a replacement. Each cartridge's full field-by-field
contract still lives in its own document, named per section. What is here
is what a Mac client needs to hold in its head at once: what is offered,
how to reach it, which transport carries what, what a refusal means, and
what a client is obliged NOT to do with the answer.

Every identifier below was read out of the running Tower on the merged
tree. Nothing is quoted from a lane document without being checked.

---

## 0. The one-paragraph version

The Tower serves **five** cartridges and declares **four**. World Builder,
the Experimental CV Lab, Scene Understanding and Document Memory each have
a typed contract and appear in `GET /cartridges`. Object Memory has a
store, read routes, imagery routes and a live Start/Stop control surface —
and is **deliberately absent from the declaration** (§9). Commands never
travel on the result channel. Bulk and imagery never travel on the socket.
Every timestamp is Tower-receipt time; there is no capture clock anywhere
on this wire.

---

## 1. Contract identifiers

Opaque strings. **Compare for equality; never parse the date.** A change
of identifier is a change of contract and is the only signal a client gets.

| Identifier | Surface | Section |
|---|---|---|
| `cartridge_results.envelope/2026-08-23` | the result socket's envelope, shared by every subscription | §3 |
| `world_builder.status/2026-09-10` | World Builder status, subscription | §5 |
| `world_builder.geometry/2026-08-25` | World Builder geometry, HTTP | §5 |
| `experimental_cv.status/2026-08-27` | CV Lab status — HTTP, socket and subscription | §6 |
| `experimental_cv.control/2026-08-27` | CV Lab command vocabulary | §6 |
| `experimental_cv.frame_result/2026-08-27` | the `cv_lab` block on every `frame_result` | §6 |
| `scene_understanding.live/2026-08-27` | Scene Understanding live state | §7 |
| `document_memory.status/2026-09-07` | Document Memory session status, subscription | §8 |
| `document_memory.library/2026-09-07` | Document Memory library, HTTP | §8 |
| `object_memory.observations/2026-08-26` | Object Memory query results, HTTP | §9 |
| `object_memory.imagery/2026-08-27` | Object Memory frame/crop retrieval, HTTP | §9 |
| `cartridge_session.control/2026-08-27` | the generic Start/Pause/Resume/Stop surface | §4 |

---

## 2. `GET /cartridges` — the capability declaration

The thing a Mac client **caches**. Available over HTTP *and* as
`{"type": "cartridges"}` on the socket, and a test asserts the two are
**byte-identical** — they are one function (`registry.declare`) reached
through one reader (`registry.declaration_inputs`), not two functions that
agree today.

HTTP exists as well as the socket because of the third state below: a
client that can only learn the contract set by opening the socket cannot
tell "unreachable" from "not built".

```json
{
  "type": "cartridges",
  "envelope_contract": "cartridge_results.envelope/2026-08-23",
  "cartridges":     [ /* what can be SUBSCRIBED to */ ],
  "http_contracts": [ /* what can be FETCHED */ ],
  "not_offered":    [ /* cartridges with no wire contract at all */ ]
}
```

**Three states a client must keep apart**, because they call for opposite
instructions to a person:

| The declaration says | Means | Show |
|---|---|---|
| the cartridge is absent from every list | this Tower has never heard of it | "not built yet" |
| present, `contract` a string this build does not know | the Tower speaks a contract this app does not | "update the app" |
| present, `"available": false` | this build implements it and cannot be started here | "connect" / the reason |

`available` is about **whether a session could be started, never about
current activity.** A Scene Understanding that is enabled but stopped is
`available: true` — it can be started — and its payload says
`lifecycle.state: "stopped"`. Folding "not running right now" into
"unavailable" would tell a person their Tower cannot do this when in fact
nobody has pressed Start.

**Configuration is the usual reason but not the only one.** A cartridge
whose mandatory runtime dependency is missing on this host — Scene
Understanding needs `torch`, which ships in the optional `[ml]` extra —
is `available: false` with a reason that says so. It was previously
`available: true` on such a host, and `POST /scene/start` then answered
`200` and failed ~50 ms later; a client had no way to learn that from the
declaration, because the declaration never re-derived it. **Read
`available: false` as "do not offer the control", and
`unavailable_reason` as prose for a person — never parse it.**

`available: true` still is not a promise that Start will succeed. It
means this build implements the contract and this host is configured and
equipped to run it. Weights can still be missing and a device can still
vanish; those arrive as `lifecycle.state: "failed"` with a reason, which
is the designated answer for a failure only a load can discover.

**As of this branch:**

| Cartridge | `result_type` | Contract | Available when |
|---|---|---|---|
| `world_builder` | `status` | `world_builder.status/2026-09-10` | `TOWER_WORLD_ROOT` is set |
| `experimental_cv` | `status` | `experimental_cv.status/2026-08-27` | a CV Lab module exists (normally always) |
| `scene_understanding` | `live` | `scene_understanding.live/2026-08-27` | the session constructs (needs `torch`/`torchvision`, the `[ml]` extra). `TOWER_SCENE_UNDERSTANDING` is `auto` when unset (since 2026-09-07); `off` withdraws it |
| `document_memory` | `status` | `document_memory.status/2026-09-07` | `TOWER_DOCUMENT_ENABLED` is on (the default since 2026-09-07; the root has a managed default under `tower/data/document_memory`, `TOWER_DOCUMENT_ROOT` overrides it) |

`http_contracts` carries one entry — `document_memory.library/2026-09-07`
at `entry_route: /documents` — with an `available`, an
`unavailable_reason` and a `why_not_a_subscription`. World Builder's
geometry and Object Memory's observations are the same shape and are
**not** declared; see §11.

**`not_offered` is EMPTY, and empty is a claim.** Every cartridge in this
build that has a wire contract now offers it. A cartridge belongs in
`not_offered` only while it can say nothing at all; one that can say "I
have observed nothing, and here is precisely why" belongs in `cartridges`,
available or not.

---

## 3. Socket versus HTTP — the rule, and why

One WebSocket at `/ws`. Everything else is HTTP.

| Travels on the socket | Travels on HTTP |
|---|---|
| frames, and `frame_result` / `frame_error` | the declaration (also on the socket) |
| the declaration | World Builder geometry (manifests, segment chunks) |
| result **subscriptions** and their snapshots | the Document Memory library and its text |
| CV Lab **commands** and status | Object Memory queries, frames and crops |
| Scene Understanding live state | every Start/Pause/Resume/Stop except the CV Lab's |

Three rules generate that table:

1. **Commands never travel on the result channel.** `tower/results/` is a
   read-only reporting surface; a test forbids a call named `observe` or
   `build` anywhere inside it.
2. **Bulk and imagery never travel on the socket.** The result sender
   shares its send lock with the frame path, so a large payload stalls
   frames. Document text and Object Memory imagery are pulled on demand.
3. **A cacheable thing wants a plain GET.** Hence the declaration on both.

The CV Lab is the deliberate exception to (1): its commands are plain
socket messages on `/ws`, not on the result channel, and there is **no
HTTP surface for CV Lab start, pause or stop.**

---

## 4. The shared lifecycle

Four cartridges have a Start and a Stop, and they do **not** share one
implementation. A Mac client must not assume one screen's controls behave
like another's.

| Cartridge | Controlled by | States |
|---|---|---|
| Object Memory | `POST /cartridges/object_memory/session/{action}` | `stopped` / `active` / `paused` |
| World Builder (since 2026-09-06) | `POST /cartridges/world_builder/session/{action}` — **intent to build**, sent by the phone when the World Builder workspace appears (`start`) and leaves (`stop`). A builder attaches to a camera capture only while `active`. `stop_policy: "request"`: Stop closes the builder's stdin and returns; a builder still observing ends its session (`interrupted`) and writes its final build; one already finalizing is allowed to finish. `following` stays truthful until it exits | `stopped` / `active` / `paused` |
| Scene Understanding | `POST /scene/{start,pause,resume,stop}` *and* `stream_start`/`stream_stop` | `stopped` / `starting` / `running` / `paused` / `failed` |
| Document Memory | `POST /documents-session/{start,pause,resume,stop}` | as Scene, plus `unavailable` |
| Experimental CV Lab | socket `cv_lab_start` / `pause` / `resume` / `stop` | `unavailable` / `idle` / `starting` / `running` / `paused` / `stopped` / `failed` |

**The generic session surface** — `cartridge_session.control/2026-08-27` —
is keyed by cartridge id and knows no cartridge, so the next producer that
needs a button gets one for free. `object_memory` and, since 2026-09-06,
`world_builder` answer it; any other name is a **404**, which is a
configuration answer. The payload now also carries `stop_policy`
(`"terminate"` for Object Memory, `"request"` for World Builder) so a client
knows whether "stopped" meant the process is gone or was asked to finish.

```
GET  /cartridges/{cartridge}/session
POST /cartridges/{cartridge}/session/{start|pause|resume|stop}
```

Response carries `contract`, `state`, `state_means`, `states`, `actions`,
`supported`, `session_id`, `started_at`, `changed_at`, `following`,
`captures`, and on POST also `accepted`, `changed`, `attached_capture_id`.

**`state_means: "intent-not-liveness"`, and this is the field a client
most often gets wrong.**

> **Render liveness from `following`, never from `state`.** An `active`
> session with an empty `following` *while a capture is recording* is a
> producer that died. `state` says what a person asked for; `following`
> says what is actually running.

This is not advice, it is load-bearing, and a reviewer reproduced why. A
Pause whose producer ignores `SIGTERM` answers **200** with `state:
"paused"` and `changed: true` — a positive claim that the action took
effect — while `following` still names the capture and the process is
still alive and still recording. `state_means: "intent-not-liveness"` is
the Tower saying so in the payload. **A Pause button keyed on `state` will
tell a person they stopped being recorded when they did not.**

> ⚠️ **But `following` has a false positive of its own.** It is
> **supervisor-scoped, not session-scoped**: a producer that survived an
> earlier Pause stays registered, so a **new** session reports the OLD
> session's capture under the new `session_id`, having attached nothing.
>
> So the two fields fail in opposite directions — `state` can claim a stop
> that did not happen, `following` can claim a start that did not happen.
> Cross-check `attached_capture_id` on the POST reply, which is honest,
> and treat a `following` entry that does not match a capture this
> session opened as unproven.

Transition rules worth memorising:

- **`start` is idempotent** and works from `stopped` *and* `paused`. It
  means "be running, whatever the app thought". A second start answers
  **200** with `"changed": false` — honoured, nothing moved. That is not
  an error and must not be shown as one.
- **`resume` is stricter** — it claims to continue something. From
  `stopped` it is refused **409** with `reason: "not-paused"`. (`pause`
  from `stopped` is the neighbouring refusal and reads `"not-active"`;
  the two are not interchangeable and were verified on the wire.)
- **`stop` is never refused, from any state.** Refusing it would make a
  Tower restart the only way out of a bad state.
- **Nothing is persisted.** A Tower that restarts comes back with every
  cartridge **stopped**, deliberately: resuming a memory of what a camera
  sees without anybody asking again is the wrong direction to fail in.
- Start before the camera is running is legal: the session goes `active`
  with `attached_capture_id: null`, and the next capture to open finds the
  gate open.

**Stop does not mean the same thing to every cartridge, and the
differences are intentional:**

| Cartridge | What Stop does to what was produced |
|---|---|
| Document Memory | **KEEPS it.** A record of what was read is exactly as true after the session ends. A dwell in progress is **flushed, not dropped.** |
| Scene Understanding | **DISCARDS it.** `scene_available` goes false, `counts`/`where`/`people` go null. A scene held past the end of a session is a claim about a room the wearer has left. |
| Object Memory | keeps the store; detaches the producer. |
| CV Lab | ends the run and **keeps the figures** until the next start. |

**Pause is the deliberately different case for Scene:** the scene survives
with its age and `lifecycle.scene_is_current` goes `false`. That is the
"last known" state, and it must be kept visually apart from "observing".

### 4.1 The four cartridges answer the same verb three different ways

This is **measured on the merged tree**, not designed. Each lane chose its
own refusal policy and no two agree. A Mac client must not write one
lifecycle component and point it at four cartridges.

| Verb, from a state that cannot honour it | Object Memory | Scene / Document | CV Lab |
|---|---|---|---|
| `pause` when already paused | **200**, `changed:false` | **200**, silent no-op | **refused** `invalid_state` |
| `pause` when stopped | **409** `not-active` | **200**, silent no-op | **refused** `invalid_state` |
| `resume` when stopped | **409** `not-paused` | **200**, silent no-op | **refused** `invalid_state` |
| `stop` when already stopped | **200**, `changed:false` | **200**, silent no-op | **refused** `invalid_state` |

Two consequences a client must code around today:

1. **Scene and Document silently no-op.** `POST /scene/resume` on a
   stopped scene returns **200 with `state: "stopped"`** and no refusal
   field. **Read the returned `state`; never treat 200 as "it worked".**
2. **The CV Lab refuses Stop** from `stopped`, `idle`, `failed` and
   `unavailable`. A client whose recovery path is "Stop, then Start" is
   refused on step one — send `cv_lab_start` directly instead.

Neither is a merge regression; both are lane behaviour, preserved
deliberately rather than harmonised during integration. Unifying them is a
contract change and belongs to a human. Until then, **this table is the
contract.**

**Demand-bound lifecycle (2026-09-07).** A Scene session runs while
**somebody is streaming AND somebody is watching**, or while an operator
holds it open by hand. The two facts are tracked apart:

- the **stream** is the feed. `stream_start` opens it, `stream_stop` or a
  disconnect closes it. The **last open stream closing stops the session
  whoever started it** — frames come from nowhere else, and a scene kept
  past its feed is a claim about a room the wearer has left. This closes
  the "owned by nobody after Stop → Start" defect the earlier draft of this
  paragraph described.
- a **watcher** is a `result_subscribe` to `scene_understanding/live`. It is
  the phone saying "a person is looking at this". A phone streaming for
  World Builder or the CV Lab is *not* a request to detect people in the
  room, so a stream with no watcher leaves the session `stopped` and its
  frames counted in `frames_dropped_not_running`. When the last watcher
  leaves (`result_unsubscribe`, or its socket closing) the detector is
  released and the GPU handed back.
- the **operator** path is `POST /scene/start`: it runs at once, with or
  without a stream, so a physical test can be driven from a Mac or curl
  without a phone build. `POST /scene/stop` ends that hold. The operator's
  session survives watchers leaving; it does not survive its last stream
  closing.

Two things the earlier stream-bound wording said that are still true, and
are not about Scene Understanding. **The phone sends nothing on the socket
to open a cartridge**, and a test asserts the wire stays silent: World
Builder's activation is the HTTP session above, which is intent to build
rather than a stream boundary. And stream ownership is a **set of
connection tokens**, so with two phones streaming the first to drop does
not close the stream out from under the second.

No demand event ever resumes a Pause. `lifecycle.follows_stream` reports
whether the stream-and-watcher rule is on (`TOWER_SCENE_AUTOSTART`, default
on); off leaves only the operator path. `lifecycle.demand` reports the
rule's inputs as counts — `streams`, `watchers`, `operator_hold` — and the
rule itself as `runs_when: "stream-and-watcher-or-operator"`, so a client
can tell "stopped because nobody is streaming" from "stopped because nobody
is watching". The counts are volatile (excluded from the revision) and
never carry a token or a subscription id.

> A phone build that subscribes at connection time rather than when the
> Scene screen is open will keep the detector running for as long as it is
> connected and streaming. That is the 2026-08-27 behaviour, not a
> regression, and the fix is on the phone: subscribe on appear, unsubscribe
> on disappear.

Document Memory's
`follows_stream` defaults **false** — that cartridge writes, and a session
that persists what a wearer read gets an explicit start.

---

## 5. World Builder

Full contract: `docs/contracts/WORLD-BUILDER-GEOMETRY.md` and
`docs/contracts/WORLD-BUILDER-IOS.md`.

**Status** arrives on the result channel (`world_builder`/`status`). Four
top-level keys, and these are the ones iOS actually decodes:

| Key | Means |
|---|---|
| `lifecycle` | `state`, `evidence`, `reason`, `build_in_progress`, `build_in_progress_unavailable_reason` |
| `model_state` | what to render. `"unsupported"` is the Tower saying it cannot do this at all, not that it is empty |
| `model_state_reason` | why, as a sentence |
| `world_snapshot` | the world, or `null`. **`null` is a real answer** and is not an empty world |

`docs/contracts/WORLD-BUILDER-IOS.md` has the field-by-field detail;
what is here is the shape a decoder needs to get right first.

**Geometry** is HTTP, because it is bulk:

```
GET /worlds/{world_id}/geometry/manifest
GET /worlds/{world_id}/geometry/segment/{segment_index}
```

### 5.1 `transform_to_world`

A **Sim3** — rotation, translation, uniform scale — mapping a segment's
own frame into the frame of its `reference_segment`. **It is not a
world-absolute pose. There is no global world frame.**

```json
"transform_to_world": {
  "rotation_wxyz":     [w, x, y, z],
  "translation":       [x, y, z],
  "scale":             0.3533,
  "reference_segment": 4,
  "frame_revision":    1
}
```

Decode as `p_ref = scale * (R(q) · p_segment) + t`.

- `rotation_wxyz` is a **quaternion, w first** — not a matrix, so
  row/column-major does not arise. It is validated as a **unit**
  quaternion at write time, because a non-unit quaternion scales the
  geometry it rotates, silently, on top of `scale`.
- `translation` is in the **reference segment's** units.
  `pose_convention.translation_units` is `"world"` — **never metres.**
- **`transform_to_world: null` means the segment is not registered into
  any shared frame.** It does **not** mean identity, which would silently
  place it at the reference origin.
- **Segments sharing a `reference_segment` may be drawn together.
  Segments with different reference segments may NOT be composited.**
- `registered: false` **forbids** placing two segments in one space:
  their scales disagree by up to ~87× on a real walk. A renderer that
  ignores this fabricates geometry.
- `pose_convention` is compared **key by key** and a mismatch **refuses
  the render** — inverting `T_world_camera` still produces a
  plausible-looking map, and that was a real shipped bug. Compare
  `pose_type`, `quaternion_order`, `handedness`, `camera_axes`,
  `translation_units`; `up_axis` is deliberately excluded for a 2-D
  top-down view.

### 5.2 Caching — the trap

**Key the per-segment geometry cache on the tuple `(content_hash,
placement_hash)`.** Both are 16 hex chars.

`content_hash` covers **poses and points only**, deliberately, so a
segment that gains a placement keeps its content hash. `placement_hash`
covers where it sits: `state`, `rotation_wxyz`, `translation`, `scale`,
`reference_segment`, `frame_revision` **and `refusal_reason``.

> Keyed on `content_hash` alone, the day a segment gains a placement the
> client keeps its cached chunk forever and draws an **unplaced** version
> of a segment the world now knows how to place. Nothing looks broken. The
> fragment simply sits in the wrong place, permanently.

`refusal_reason` is inside the placement hash because it was once left
out, and on the real corpus **26 of 29 segments are refused and share one
`placement_hash`** — so a re-registration that changed every refusal
reason moved no hash, and a conforming client showed stale text forever.

**The deliberate regression test:** change only a placement, same points —
the client must refetch and re-place.

### 5.3 Scale — the truthfulness rule

Two states are reachable, and only two:

- **`relative`** — internally consistent with an arbitrary unit. **Not
  metric.**
- **`unknown`** — **no unit at all**, a strictly weaker claim. **Never map
  it to `relative`.**

`estimated` and `measured` are unreachable on this hardware and are
mapped rather than discarded so a future arrival is not silently
downgraded. `meters_per_unit` is `null` unless the state is metric —
**`null` does not mean 1.0.** A world with more than one segment stays
`unknown`, because segments do not share a coordinate frame. Calibration
does not change this: intrinsics unlock *poses*, not size.

**Two separate display gates:**

| Gate | Asks | Tower answer today |
|---|---|---|
| `distanceDisplayable` | may this be shown as a physical distance? | **always false** |
| `labelledFigureDisplayable` | may this be shown as the labelled figure it is? | true when a unit is present |

"6.6 world units", with the unit attached and "Shape and layout only. No
real-world distances are claimed." beneath it, is not a distance claim.
**The gate is the unit, not the scale** — a bare number is what a reader
silently reads as metres.

### 5.4 Imagery

**No imagery, ever.** `image_relpath` and every keyframe byte stay
Tower-side. `retains_raw_imagery` remains permanently `true`; redaction is
a **process claim, not an outcome claim.**

---

## 6. Experimental CV Lab

Full contract: `tower/docs/contracts/EXPERIMENTAL-CV-LAB.md`.

**Advertised identifier: `experimental_cv`.** (On the phone side the
catalog key is `experimental-cv` with a hyphen and must be mapped —
without that mapping nothing else works.)

**Three read surfaces, one builder**, and a test asserts they agree
structurally:

```
GET /cv-lab                                   -> {contract, control_contract, status}
socket {"type": "cv_lab_status"}              -> {..., status}
result_subscribe experimental_cv/status       -> the payload IS that status
```

They are **not** byte-identical across time — wall-clock fields advance
between reads. The claim that holds is structural: same keys, same types,
same meanings, one builder.

### 6.1 Control — socket only

| Message | Fields | Effect |
|---|---|---|
| `cv_lab_status` | `request_id?` | read the document |
| `cv_lab_start` | `experiment_id`, `request_id?` | select **and** arm; replaces whatever ran |
| `cv_lab_pause` | `run_id?`, `request_id?` | stop processing, keep the experiment loaded |
| `cv_lab_resume` | `run_id?`, `request_id?` | resume |
| `cv_lab_stop` | `run_id?`, `request_id?` | end the run, release the experiment, **keep the figures** |

There is **no separate select message** — selection without arming is a
state nobody needs on the wire. `request_id` is opaque, max 64 chars,
echoed back; longer is **dropped, not refused**. A stale `run_id` is
refused `stale_run` rather than applied to whichever run is current.

Replies carry `accepted_command`, which is how a pushed status is told
from an answer to a command.

**Eight refusal reasons on `cv_lab_error`, and every one of them means the
request did not take effect. There is no partial application:**
`malformed_request`, `unknown_experiment` (carries `available`),
`experiment_unavailable`, `lab_busy`, `invalid_state`, `stale_run`
(carries `current_run_id`), `lab_unavailable` (**terminal** → show
unsupported), `internal_error` (**transient, retryable**).

### 6.2 Provenance and staleness-by-structure

Every `frame_result` gains a `cv_lab` block: `contract`,
`tower_instance_id`, `run_id`, `result_seq`, `experiment_id`,
`experiment_name`, `provenance`, `backend`, `device`, `device_requested`,
`result_label`, `processing_ms`, `tower_received_at`, `time_basis`.

> **Discard any `frame_result` whose `cv_lab.run_id` is not the run you
> are watching.**

The Tower makes this structural rather than checked: a new experiment is a
new run, and the old experiment is released **before** the new run id is
published, so no result computed by one experiment can carry another's
name. Frames arriving in that window are refused `cv_lab_starting` rather
than answered by the experiment being replaced.

`run_id` is `"<tower_instance_id>-<n>"`, so comparing `run_id` alone also
covers a reconnect to a **restarted** Tower. Clear the held run id on
`sendStreamStop()` and on teardown.

`result_seq` is dense within a run, from 1. **The wire `seq` is the
phone's capture index and skips by design** — the sender forwards one
frame in thirty — so it cannot order results.

Per metric: `provenance` is **required and never omitted** (`measured` or
`inferred`); `confidence`, `baseline` and `higher_is_better` are
**always null**, so no better/worse verdict is ever rendered.

### 6.3 The `cv_lab_starting` refusal window

While the Lab is not `running`, a frame is answered with `frame_error`
rather than silence:

```json
{"type": "frame_error", "seq": 30, "reason": "cv_lab_starting",
 "message": "the CV Lab is arming an experiment; frames are refused until it is ready..."}
```

Six mapped reasons — `cv_lab_idle`, `cv_lab_starting`, `cv_lab_paused`,
`cv_lab_stopped`, `cv_lab_failed`, `cv_lab_unavailable` — alongside the
transport's `invalid_frame`, `frame_skipped`, `module_unavailable`. The
window lasts as long as the arm, bounded by the **120 s arm timeout**
(the same bound and reason as the module container's load timeout: 119 MB
of MiDaS weights does not fit a 10 s bound on any ordinary link). There is
**no progress reporting** — `torch.hub` does not offer any.

**A refusal is not a processing error.** These count under
`frames_rejected`, never under `frame_processing_errors`: a Lab paused for
five minutes has not failed hundreds of times.

### 6.4 Terminal FAILED, truthfully

**There is no `start_failed` message.** An arm is asynchronous — that is
the whole reason a start returns immediately — so by the time a load
fails, the command has already been answered `accepted`.

> **The outcome arrives as state.** `lifecycle.state` becomes `failed`
> with a `reason`. A client that sends commands and does not also read
> status will never learn that a start failed.

A failed **interactive** start is recoverable. A failed **startup**
experiment is **terminal** until restart — a typo in configuration should
be loud. An experiment raising anything other than `FrameProcessingError`
while processing a frame is **terminal** and takes the Lab with it.

### 6.5 Debug versus Release

The iOS camera path is `#if DEBUG`. **A Release build has no camera, sends
no frame, and therefore receives no `frame_result`.** That predates this
work and is not fixed by it.

> **`.running` may be shown as LIVE only when this build is itself
> streaming AND `source.receiving_frames` is true.** Both halves are
> needed: `source` is **Tower-wide**, so it is `true` for a Release build
> with no camera whenever a second phone is attached.

Never render a Start control in a configuration with no
`startCameraSession` to call. On a build that never processed a frame the
run publishes **`null`, not zero** — a rate over a zero-length window is
undefined, not zero.

---

## 7. Scene Understanding

Full contract: `tower/docs/contracts/CARTRIDGE-RESULTS.md` §14.

Result type is **`live`, not `status`** — the payload *is* the answer, not
progress toward one.

**Every key is present in every state.** A key that appeared and
disappeared would force a decoder to treat it as optional and lose the
ability to tell "zero of these" from "this Tower did not say".

Constant self-description, safe to assert against: `claim:
"visible-now-not-a-record"`, `identity: "anonymous-and-unpublished"`,
`absence_means: "not-visible-to-this-cartridge"`, `persistence: "none"`,
`frame_of_reference: "camera"`, `time_basis: "tower-receipt"`.

`lifecycle` carries `state`, `states`, `session_id` (int, increments per
Start), `scene_is_current`, `failure_reason`, `started_at`, `ready_at`,
`loading_seconds`, `load_overdue`, `load_overdue_after_seconds` (120.0),
`follows_stream`, and `demand` (`streams`, `watchers`, `operator_hold`,
`runs_when` — see *Demand-bound lifecycle*).

> **Two payloads with different `session_id` came from different tracking
> sessions and must not be compared.**

`load_overdue` is **not a failure** — nothing can interrupt a blocking
model load, and a first-run weight download is slow and still correct.

### 7.1 The scene

`reported_classes` is **13** COCO names, fixed at build time, carrying
COCO's meanings (`mouse` is the pointing device, `tv` is any large
display). `counts` has **one entry per reported class, present at `0`
rather than omitted** — a class silently absent would be
indistinguishable from one looked for and not seen.

`where` carries **per-label side counts** (`left`/`centre`/`right`/
`unknown`) for every reported label **including `person`** (since
2026-09-07), because one side cannot describe a chair on the left and a
chair on the right, and "is there a person on my left" is a count of
people on the left. `where_excludes` is empty. `side_convention` is
declared on the payload: the wearer's own left and right as the camera
sees them, thresholds at 0.35 and 0.65 of frame width (14° of the
camera's 45°) with 0.03 of hysteresis, stream assumed unmirrored and
nothing verifies that.

`people` is a count and an aggregate, **never a list**:
`may_include_wearer` is **true**, `validated` is **false**, and
`facing_wearer` is **null, never 0, when unmeasured.** Since 2026-09-07 it
also carries `partial_bottom_edge` — person boxes cut off by the bottom
edge with no head region, kept out of `count` because from a head-worn
camera they are most often the wearer's own body (`partial_bottom_edge_
note` says it can be somebody's legs too); `by_apparent_size`
(`large`/`medium`/`small`/`unknown`, box height as a fraction of the
frame — **sizes in the picture, never distances**, per
`apparent_size_note`); and `orientation_method`, `orientation_status:
"experimental"` and `orientation_validation`.

### 7.2 Truthfulness — the fields you are obliged to render

- **`count_is_lower_bound` is `true` on every payload.** Measured on 700
  human-labelled COCO images at this camera's 640 px long side (the
  corpus has no bystander): the CUDA detector (RT-DETRv2-R18 at 0.5)
  reaches person AP50 **0.86**, chair **0.54**, laptop **0.83**, tv
  **0.82**, book **0.35**, with recall **0.40** under 1% of frame area
  rising to 0.88 above 20%; the CPU detector (SSDLite320 at 0.4) reaches
  0.60 / 0.20 / 0.55 / 0.56 / 0.09 and is blind under 2%. `detector` on
  the payload says which one counted. Those are stills of other people's
  photographs, never a person seen through these glasses.

  > **An undercount published without disclosure looks exactly like a
  > quiet room.** Render this somewhere a person will see it.

- `count_limitations` — slugs `size-floor`, `recall`,
  `people-count-accuracy`, `departure-lag`, `field-of-view`.
- `count_measurement` carries `measured_at`, the labelled-image count and
  a note naming what was and was not measured.
- **`scene_available: false` in four distinct situations**, told apart by
  `scene_unavailable_reason`: stopped / still loading / failed /
  running-but-no-frame-yet. All zero counts with `scene_available: true`
  is the fifth and different case — "I looked and saw nothing".
- `tracks`, `relations` and `confidence` are **always null**, each with an
  `*_absent_reason` and a refusal list. **There is no key anywhere in this
  payload that could hold an entity or a relation.**

### 7.3 Privacy

`persistence: "none"` — enforced, not intended. **No face recognition:**
the facing stage runs a face *detector* (the vendored YuNet) on each
tracked person's box and keeps one boolean per box — no crop, no
landmark, no score and no descriptor survives the call, and nothing
supports matching. **No identity persistence:** track ids are
session-scoped integers, never published, and named explicitly in
`refused_entity_fields`.

The reasoning is **not** "minimise disclosure" — the phone sent the
pixels, so a count discloses strictly less than the frame the phone
already holds. What is genuinely new is **joinability**: a stable
`track_id` plus a timestamp would let a recipient assemble the per-person
dwell timeline this cartridge refuses to keep — persists-nothing laundered
onto the consumer.

**This cartridge serves no image and has no artifact fetch contract.**

---

## 8. Document Memory

Full contract: `tower/docs/contracts/CARTRIDGE-RESULTS.md` §15.

**Two identifiers, deliberately**, because they govern different
transports with different failure modes: `status` is small and pushed on
the socket; `library` is bulk text and pulled over HTTP. A change to one
is not a change to the other.

```
GET  /documents?limit=&retention_days=      recent, newest first, no text
GET  /documents/search?text=&limit=         BM25, bounded snippets
GET  /documents/around?at=&window_seconds=  a window around an instant
GET  /documents/{document_id}               one document, with its pages
GET  /documents-session                     the capture session
POST /documents-session/{start,pause,resume,stop}
```

`/documents*` answer **404** when `TOWER_DOCUMENT_ENABLED` is off, which
since 2026-09-07 is the only way to have no root: the default is
`tower/data/document_memory`, and `TOWER_DOCUMENT_ROOT` overrides it.
`/documents-session*` answer **404** when `TOWER_DOCUMENT_CAPTURE` is off
(it is on by default) or the session could not be constructed — the
`[ocr]` extra not installed, most likely, and the body says so. A root
with capture off is a Tower that serves a library recorded elsewhere and
records nothing itself. Both 404 bodies still name `TOWER_DOCUMENT_ROOT`
and `TOWER_DOCUMENT_CAPTURE` respectively, because the phone
string-matches them, and **neither is ever the answer to a query about a
document**, which is answered with `answer: "no_observation"`.

The session verbs answer **200 with the full envelope** from any state;
a verb that does not apply leaves `session.state` unchanged. The phone
renders any other status as a transport failure.

### 8.1 The three answers — a closed vocabulary on every response

| `answer` | Means | Render |
|---|---|---|
| `matched` | documents were found | the list |
| `not_found` | the memory was searched and nothing matched | "Nothing matched" |
| `no_observation` | the memory holds nothing that could have matched | "Never observed" — **and say explicitly this is not the same as the document not existing** |

`no_observation_note` carries the sentence.

### 8.2 Fields a client must read rather than assume

- `claim: "a-page-was-in-view-and-was-ocred"`. Not "was read" — the
  camera cannot establish that.
- `identity: "same-page-by-text-and-look-within-library"` — a later
  dwell whose first readable page has the same words AND the same look
  as a record becomes a **sighting** of it (`sighting_count`,
  `last_observed_at`, `total_observed_seconds`); anything weaker is a
  separate record. The first observation's fields are never rewritten.
- `library.revision` on the status payload is the **live-update
  trigger**: re-fetch the listing when it changes, never on a timer.
- `text_availability.state` — `unknown` (no pages), `not_readable`
  (**a real answer**: we looked and found no readable text), `extracted`.
- `title_is_derived`; a null title renders as "Untitled document", **never
  as an invented name**. Titles clip at 60 chars.
- `snippet_max_chars` is **48** — read the field, do not hard-code.
- `observed_seconds` is **not** a claim that the wearer looked at it,
  noticed it, or read it.
- `summary` is the first forty words **verbatim**
  (`summary_is_verbatim_excerpt: true`), served only by
  `GET /documents/{document_id}`.
- `retention.writer_window_days` is **always null** — the store persists
  no retention manifest, so a reader cannot learn the window its writer
  used. `?retention_days=` **narrows a read and can never widen it.**
- `imagery_served` is **always false** — a boolean, not a path.
  `image_kept` is false unless explicitly enabled and **must stay off**.
- `record_notes` is hoisted to the envelope, keyed by field name. It was
  2,351 bytes per record repeated; hoisting cut a 200-document listing
  from 488 KB to 249 KB **with nothing dropped.** Caveats were hoisted,
  never deleted.

### 8.3 The limitation you must not hide

> **The premise is untested on a physical page.** No capture on this
> platform has ever contained a sheet of paper: a visual review of one
> sharp frame from each of 71 captures and a text-detector sweep of every
> tenth frame of 97 captures (4,599 frames) found none, and found zero
> confident text boxes. The contour-quad detector that shipped on
> 2026-08-27 fired six times on 9,199 of those frames, all false; it was
> replaced on 2026-09-07 by a steadiness gate plus the OCR engine's own
> text detector, validated on rendered pages with known text.
>
> Separately, at the 360×640 the glasses deliver, EasyOCR returned **zero
> dictionary words** across 919 sampled real frames dense with screen
> text, at median confidence 0.056. On rendered pages at that geometry
> word recall is 0.98 with the page filling the frame, 0.74 at 60% of
> the frame height and 0.02 at 40%.
>
> **An empty library on existing footage is the expected result.** Every
> response carries `recording_limitations` saying so. A client that
> renders an empty library as "no documents yet" is inviting a person to
> wait for something that is not coming; what is coming requires a page
> held close.

Resolution remains a cross-cartridge choice the phone makes: raising
the stream to 720×1280 breaks World Builder's tracking (**73.3%** of
frames rejected as blurred), 504×896 is the unmeasured middle rung, and
at 504×896 recall on rendered pages is 1.00 / 0.98 / 0.66 at the three
fills above. The DEBUG-only Capture Resolution picker on the phone is
the lever for a physical test. That is iOS/DAT work and is not in this
contract.

### 8.4 Provenance — and the deliberate contrast

`provenance` carries `capture_id`, `capture_id_validated` (**always
false** — nothing checks the capture still exists), `page_source_seqs`,
`world_id`, `world_session_id`, `imagery_retention: "capture-side"`, and
**`joinable: true` with a `joinable_note`.**

> That last pair is said out loud rather than left to be noticed. This
> block **is** joinable: a capture id, frame sequence numbers and a
> timestamp locate this reading in a recording on disk, durably across
> sessions — which is precisely what Scene Understanding refuses to hand
> anyone. **The two cartridges differ here on purpose. A document is a
> record; a scene is not.**

---

## 9. Object Memory

Full contract: `docs/contracts/OBJECT-MEMORY.md`.

**Object Memory is NOT in `GET /cartridges`, and that is deliberate.**

Its identifier exists (`CARTRIDGE_OBJECT_MEMORY`), its control surface is
live at `/cartridges/object_memory/session`, and its query and imagery
routes answer — but declaring it on the socket breaks a pinned iOS test
(`testTheTowerDeclaresOnlyTheWorldBuilderContract`), so the declaration
waits for the iOS lane to take both halves in one change. The Tower side
is about four lines in `tower/results/registry.py`.

**This is a decision for a human. Do not close it by noticing the gap.**
A `result_subscribe` for `object_memory` is refused `unknown_cartridge`,
and a test pins that so an accidental declaration is caught.

**Reach Object Memory over HTTP. Learn nothing about it from the
declaration.**

### 9.1 Query

```
GET /object-memory/observations?object_class=&retention_days=&limit=
GET /object-memory/last-seen/{object_class}
```

Envelope: `contract: "object_memory.observations/2026-08-26"`, `claim:
"category-was-visible-once"`, `identity: "category-not-instance"`,
`absence_means: "not-observed-by-this-cartridge"`, `spatial_ref: null`,
`recorded_classes`, `imagery`, `retention`.

**Read `recorded_classes` from the payload.** Its value is
configuration-dependent: `["laptop", "cell phone"]` by default, twelve
more when the verifier is on.

`retention` reports `requested_days`, `effective_days`, `clamped`, and a
`policy` of `min(persisted, requested)` — **a reader may narrow this
window and can never widen it.** `effective_days` is never 0. Filtering is
on `recorded_at`, not `observed_at`.

### 9.2 Imagery

```
GET /object-memory/observations/{observation_id}/imagery   -> JSON
GET /object-memory/observations/{observation_id}/frame     -> image/jpeg
GET /object-memory/observations/{observation_id}/crop      -> image/jpeg, 35% padding
```

Both binary routes send **`Cache-Control: no-store`** — a proxy or browser
holding a copy is a second store nobody chose and nobody's retention
governs. **Do not cache these bytes.**

> ### ⚠️ EVERY NON-200 BODY IS NESTED UNDER `detail`
>
> These routes raise FastAPI's `HTTPException(detail=<envelope>)` and no
> custom handler unwraps it, so on **404, 409, 410 and 503** the envelope
> below is one level down:
>
> ```json
> { "detail": { "contract": "object_memory.imagery/2026-08-27",
>               "reason": "imagery-no-longer-available",
>               "memory_retained": true, "...": "..." } }
> ```
>
> This applies to the session-control refusals in §4 as well. It matters
> most here: **`memory_retained` is unreachable at the top level**, so a
> client that reads it where this document used to say it was gets
> `nil`, falls through to its error branch, and renders the broken image
> that §12 forbids. Decode a non-200 body as `{"detail": <envelope>}`.

| Code | `reason` | Means |
|---|---|---|
| 200 | `null` | a picture — and the **only** case where the envelope is at the top level |
| **410** | `imagery-no-longer-available` | **the pointer is intact and the picture is gone** |
| 404 | `no-such-observation` | nothing matched **within retention** |
| 404 | `record-has-no-frame-reference` | the record never had a pointer |
| 503 | `display-filter-unavailable` | no face weights, or the filter failed — **nothing is served** |
| 503 | `no-capture-root-configured` | nowhere to look |
| 503 | `frame-unreadable` | present, undecodable |

`/imagery` answers **200 even when there is no picture**; only an unknown
handle is a real 404 there.

> **410 means "the memory is kept and the picture is not."**
> `memory_retained: true` is in the body. **This must not render as a
> broken image or an empty row.** It is a true and useful sentence about
> capture-side retention.

**Retention is not bypassed by knowing an id** — a handle resolves through
the same clamped read, so an expired record is unreachable by its own id.

### 9.3 The filter, and what it is not called

`filter: "display-filter/yunet-2023mar@0.30"`, `filter_means:
"applied-on-read-the-stored-frame-is-unchanged"`.

This runs **on read**. The raw frame stays where it is; the capture
manifests record `redaction: "none"` and this cartridge does not own that
tree. Hence "display filter" and **never "redacted", "anonymised" or
"privacy-safe".**

**A Tower whose weights are missing serves nothing** (503). There is no
lenient default, because the lenient default here is a raw first-person
frame. `regions_filled: 0` means **nothing was detected**, not that
nothing was there.

**`subject_obscured` (0.0–1.0) exists for a measured defect.** The filter
fires on **40.2%** of real corpus frames; of 36 firings inspected by eye,
**4 were a real face** and 32 were not — hands on a keyboard, screens, a
door, a sink. One fill landed squarely on the mouse the record was about.
The filter was **not weakened** — a face-detection threshold is not a
picture-quality knob. With `subject_obscured > 0`, say the subject is
behind a fill, or fall back to `/frame`.

### 9.4 Identity and language — what is refused

- **A record is about a category, not an instance.** `laptop` means *a*
  laptop was in view. Two records of `laptop` are not evidence about the
  same object. Cross-session instance identity is deliberately not built.
- **`person` is refused absolutely** — not a tier, a separate constant
  checked first, that no model can reach past.
- `spatial_ref` is **always null** and actively nulled on read.
  `where.kind` is `frame-reference` — **not a place.**
  `bounding_box_normalized` is where in the **picture**, nested under
  `where`, never at top level.
- `time_basis: "tower-receipt"` — **must not be presented as a shutter
  time.**
- **Say `observed`, never `present`.** True means a record exists, not
  that the object is there. `recordable: false` means its absence carries
  no information at all.

> A picture is a much stronger location cue than a sentence, and no string
> test can catch it. The caption carries the whole burden. Suggested, and
> **to be tested on a person rather than accepted from here**: "A laptop
> was visible. This is the frame the Tower kept the record against — a
> picture from the recording, not a place. It does not say anything about
> now."

---

## 10. Errors and refusals, in one place

**Result channel** (`result_error`): `malformed_request`,
`unknown_cartridge`, `unknown_result_type`, `contract_mismatch`,
`cartridge_unavailable`, `too_many_subscriptions`, `unknown_subscription`,
`snapshot_failed`.

`unknown_cartridge` and `cartridge_unavailable` are **different
instructions to a person** — "not built yet" against "connect" — and
collapsing them tells someone to give up on a cartridge one environment
variable would turn on. A `cartridge_unavailable` message names the
`TOWER_` variable that would fix it **where a variable is what is
missing** — which is the common case but not the only one. When the
cartridge is switched on and could not be constructed, the reason names
what actually failed instead, because naming a variable that is already
set sends an operator to check the one thing that is already correct.
Either way it is prose for a person and must not be parsed.

**Frame path**: `invalid_frame`, `frame_skipped`, `module_unavailable`,
plus the six `cv_lab_*` states in §6.3.

**CV Lab control**: the eight in §6.1.

**Session control**: **200** honoured (including an idempotent no-op),
**404** no such cartridge session (a configuration answer), **409** cannot
be honoured from this state. Four `reason` values, verified on the wire:

| `reason` | Raised by |
|---|---|
| `not-active` | `pause` from `stopped` |
| `not-paused` | `resume` from `stopped` |
| `unsupported` | any action on a Tower with no producer to start |
| `unknown-action` | an action outside `start`/`pause`/`resume`/`stop` |

Refusals are **409, not 200 with a flag** — a client ignoring the body
would read 200 as success. `stop` never appears here: it is never refused.

**HTTP 404 on this Tower usually means "not configured", not "no such
route"**, and every such 404 names the variable. Treat a 404 with no
`TOWER_` in it as a genuine routing bug.

---

## 11. Timestamps and provenance

**There is no capture timestamp anywhere on this wire.** `tower/frames.py`
carries no time field. Every timestamp on every cartridge is
`time_basis: "tower-receipt"` — when the **Tower** received the frame,
never when the glasses captured it.

Consequences a client must respect:

- **No end-to-end latency field exists**, deliberately: a number derived
  from two unrelated clocks is not a latency.
- Scene's `observed_at` is when the Tower received the frame the scene
  came from, not when the detector finished with it.
- Object Memory's times must not be presented as shutter times.

**Two HTTP contracts are declared nowhere.** `world_builder.geometry/2026-08-25`
and `object_memory.observations/2026-08-26` are the same shape as
Document Memory's library entry and are not in `http_contracts`.
Declaring them means moving their identifiers out of adapter modules into
`tower/results/contracts.py`, and `registry.py` must stay cartridge-blind
so it cannot import an adapter. Those two lanes own the move. **A Mac
client must hard-code those two identifiers today.**

---

## 12. Privacy and display obligations — the short list

A client that gets everything else right and these wrong has shipped the
wrong product.

1. **Never render a World Builder figure as metres.** `distanceDisplayable`
   is always false. Attach the Tower's own unit or show nothing.
2. **Never composite segments with different `reference_segment`s**, and
   never treat `transform_to_world: null` as identity.
3. **Render Scene's `count_is_lower_bound` where a person sees it.** An
   undercount without disclosure looks exactly like a quiet room.
4. **Never cache a Scene across a Stop.** Keep `paused`'s last-known state
   visually distinct from `observing`.
5. **Render Document's empty library as "never observed", not "not yet".**
6. **Render Object Memory's 410 as "the memory is kept, the picture is
   gone"** — never a broken image.
7. **Never call the display filter redaction.** Never serve an Object
   Memory frame when the filter is unavailable — the Tower already
   refuses; do not work around it.
8. **Say "observed", never "present"; "a laptop", never "your laptop".**
9. **Do not cache Object Memory imagery bytes.** `Cache-Control: no-store`
   is on the response for a reason.
10. **Read `recorded_classes` and `snippet_max_chars` off the payload.**
    Both are configuration-dependent.
11. **Never persist, cache or re-serve a CV Lab preview.** It arrives
    `raw_ephemeral`, which permits a live view and nothing else. The Tower
    sends `Cache-Control: no-store`, writes nothing to disk and keeps only
    the newest frame; a client that saved one would have created the
    persistence that treatment says does not exist.
12. **Never call a CV Lab preview redacted, anonymised or privacy-safe.**
    No face detector runs on that path and the contract says so
    (`face_filter: "none"`). An edge map keeps a jawline and a depth map
    keeps a silhouette — derived is not unrecognisable.
13. **Never render a CV Lab depth preview as distance.** It is relative
    inverse depth on a scale that differs per frame. Same rule as item 1,
    and the same reason.

---

## 13. Where the code is

| Concern | File |
|---|---|
| Contract identifiers | `tower/results/contracts.py` |
| Capability declaration | `tower/results/registry.py` (`declare`, `declaration_inputs`) |
| `GET /cartridges` | `tower/routes/cartridges.py` |
| Result socket protocol | `tower/routes/results_ws.py` |
| Generic session control | `tower/routes/sessions.py`, `tower/cartridge_session.py` |
| Shared capture workers | `tower/capture_workers.py` |
| World Builder geometry | `tower/results/world_builder_geometry.py`, `tower/routes/geometry.py` |
| CV Lab | `tower/cv_lab/`, `tower/routes/cv_lab.py`, `tower/routes/cv_lab_ws.py` |
| Scene Understanding | `tower/scene/`, `tower/results/scene_understanding.py`, `tower/routes/scene.py` |
| Document Memory | `tower/document_memory/`, `tower/results/document_memory.py`, `tower/routes/documents.py` |
| Object Memory | `tower/object_memory/`, `tower/results/object_memory.py`, `tower/routes/observations.py` |
| Live cartridge construction | `tower/cartridge_runtime.py`, `tower/live_session.py` |
| Wiring | `tower/main.py`, `tower/config.py` |
| The end-to-end smoke | `tower/scripts/unified_cartridge_smoke.py` |
