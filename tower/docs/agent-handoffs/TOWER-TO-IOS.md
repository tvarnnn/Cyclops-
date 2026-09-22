# Tower → iOS Integration Contract

> **PARTLY SUPERSEDED — 2026-08-26.** For anything you intend to *act* on,
> start at `docs/agent-handoffs/IOS-EXECUTION-PLAN.md`. This document
> predates both the result channel and the geometry transport; the
> surfaces it describes that still exist are unchanged, but it is no
> longer a to-do list.

> **2026-08-23 — a structured result channel now exists.** The implemented
> wire contract lives in **`docs/contracts/CARTRIDGE-RESULTS.md`**, and
> every requirement in the repo-root `IOS-to-Tower.md` is classified in
> **`docs/contracts/IOS-TO-TOWER-RECONCILIATION.md`**. Sections 6.1, 6.2,
> 6.4 and 6.9 below are updated in place; the rest of this document
> describes surfaces that predate it and is unchanged.

**Audience:** whoever implements or changes the iOS side of the Glasses
platform (the wearable gateway / capture-and-control plane), and any
future Tower agent who must not break what iOS already depends on.

**Written:** 2026-08-22, from `world-builder/v1` @ `019cd1c`, during the
World Builder V1 closeout. Updated the same day with the `metrics` wire
field (§1.3) and Document Memory's measured resolution requirement
(§6.8).

## What this document is

A description of the surfaces that **actually exist on the Tower today**,
copied from source, plus an explicit list of the things a future iOS
feature would need that **Tower does not provide**.

It is deliberately not a design. Where a capability is missing, this
document names the missing capability rather than inventing an API for
it. `plan.md` §7 and the cartridge brief both forbid inventing an iOS
contract because one would be convenient, and a fabricated endpoint here
would be indistinguishable from a real one to the next reader.

**Status legend used throughout:**

| Tag | Meaning |
|---|---|
| **EXISTS** | Implemented on Tower, covered by tests, callable today |
| **EXISTS (OFF BY DEFAULT)** | Implemented, but nothing arms it in a normal process |
| **MISSING** | Does not exist. Named so nobody assumes it does |
| **BLOCKED** | Cannot be built without unblocking a named architecture decision |

---

## 1. The only network surface that exists

Tower is a FastAPI app (`tower/main.py::create_app`). It mounts exactly
two routers: `tower/routes/health.py` and `tower/routes/ws.py`. There are
no other routes. There is no authentication and the default bind is
`0.0.0.0:8000` (`tower/config.py`) — that is a known, deliberately
deferred Phase 1.5 item, not an oversight this document resolves.

### 1.1 `GET /health` — **EXISTS**

```json
{
  "status": "ok",
  "service": "glasses-tower",
  "version": "0.1.0",
  "module_state": "active",
  "module_id": "experimental-cv"
}
```

`module_state` is one of `unloaded | loading | ready | active | stopping |
failed` (`tower/modules/base.py::ModuleState`). `module_id` is the
descriptor id of the single loaded module. Both are read live from
`app.state.module_container`; neither is cached or faked.

### 1.2 `WS /ws` — **EXISTS**

One WebSocket, JSON text messages only. A binary frame is logged and
ignored, not treated as a protocol error. A non-object JSON payload is
logged and ignored. An unknown `type` is logged and ignored.

**Client → Tower**

| `type` | Required fields | Optional fields |
|---|---|---|
| `ping` | — | — |
| `stream_start` | — | — |
| `stream_stop` | — | — |
| `frame` | `seq`, `width`, `height`, `format`, `data` | `source_seq`, `tx_seq` |

`frame` field semantics (`tower/frames.py::parse_and_decode_frame`):

- `seq` — integer. **Rejected if not a JSON integer**; `true` is rejected
  explicitly because `bool` is an `int` subclass in Python.
- `width` / `height` — the sender's *declared* dimensions. Tower decodes
  the JPEG and compares; a mismatch is logged as a warning and the frame
  is still processed. It is not an error.
- `format` — must be the exact string `"jpeg"`. Nothing else is accepted.
- `data` — strict base64 (`validate=True`) of the JPEG bytes.
- `source_seq` — the DAT/capture frame index. **Falls back to `seq`** when
  absent or explicitly `null`. This is the field that must carry the
  capture-side index once the sender stops conflating the two.
- `tx_seq` — a dense per-transmitted-message counter. `null`/absent means
  "this sender does not provide it", which is **not** the same as "no
  gaps". Tower's gap arithmetic distinguishes the two.

**Tower → Client**

| `type` | Fields |
|---|---|
| `pong` | `type` |
| `frame_result` | `seq`, `processing_ms`, `result_value`, `result_label`, `stage_ms`; plus `mean_intensity` and `metrics` **only when the module produced them** |
| `frame_error` | `seq` (may be `null`), `reason`, `message` |

`frame_error.reason` is one of exactly: `invalid_frame` (failed parsing or
decoding), `frame_skipped` (one recoverable bad frame; the module is
still ACTIVE and will accept the next), `module_unavailable` (the module
cannot accept observations at all).

`seq` is `null` on an `invalid_frame` that failed before `seq` could be
read. That is deliberate — Rule 3 forbids inventing a sequence number.

**Ordering guarantee that matters to iOS:** `frame_result` is sent
*before* any recording side-effect runs. Recording does an fsync'd disk
write on the event loop, so putting it first would add disk latency to
every reply (`ws.py::_record_capture`).

### 1.3 `metrics` — a measurement channel (added 2026-08-22)

`frame_result` may carry a `metrics` object: **`name -> number`, nothing
richer.**

```json
{
  "type": "frame_result", "seq": 41,
  "processing_ms": 5.4,
  "result_value": 357.9, "result_label": "sharpness_laplacian_var",
  "stage_ms": {"decode": 0.7, "sharpness": 1.8, "...": 0.0},
  "metrics": {
    "sharpness_laplacian_var": 357.9,
    "entropy_bits": 4.93,
    "edge_density": 0.0537,
    "overexposed_fraction": 0.0
  }
}
```

Rules a client can rely on:

- **The field is omitted entirely when there is nothing to report.** The
  default `baseline` experiment does not emit it. A client that has never
  heard of `metrics` is unaffected — this is an additive change under §7
  rule 3.
- **`result_label` always appears as a key in `metrics`** when `metrics`
  is present, with the same value as `result_value`. A client may read
  either; they can never disagree.
- **Every value is a JSON number.** Never a string, never a bool, never
  `null`, and never `NaN`/`Infinity` — a test pins that, because
  `json.dumps` writes bare `NaN` and a strict parser would reject the
  whole message.
- **The key set depends on the selected experiment and is not frozen.**
  Treat it as a bag to display or log, not a schema to destructure. New
  keys may appear; do not fail on an unknown one.
- **These are measurements, not facts.** Anything model-derived
  (`detections`, `mean_relative_depth`) is inference, and must never be
  presented as a sensor measurement (Rule 16 / Core Principle 2).

### 1.3b `GET /cartridges` and the result channel — **EXISTS (2026-08-23)**

`GET /cartridges` returns the capability declaration, and the `/ws` socket
gained `cartridges`, `result_subscribe` and `result_unsubscribe`, replying
with `result_subscribed`, `cartridge_result`, `result_unsubscribed` and
`result_error`. An unrecognised message type now also returns
`protocol_error` instead of being silently logged.

All additive. The six pre-existing message types are byte-identical, and a
test asserts `frame_result` is unchanged field-for-field with a
subscription open. Full contract: `docs/contracts/CARTRIDGE-RESULTS.md`.

### 1.4 What `frame_result` still cannot carry — **the binding constraint**

`metrics` widened the channel; it did not open it. There is still **no
way to return a structured object** — no detection list with boxes, no
geometry, no world delta, no event. `dict[str, float]` was chosen
deliberately over something richer, because a general result type needs
the module-contract work that is blocked at V1.0/V1.1, and widening this
type would have been a quiet way of pretending otherwise.

Every "why can't iOS see X" question below still ends here.

---

## 2. Session and capture lifecycle

### 2.1 Connection lifecycle — **EXISTS**

`ConnectionTracker` (`tower/session.py`) records connect/disconnect.
`stream_start` opens a measurement window (`SessionMetrics`); a second
`stream_start` finalises the previous window with end reason
`superseded_by_stream_start` rather than silently discarding it.
`stream_stop` finalises with `stream_stop`. An abrupt disconnect finalises
with `disconnect`.

There is **no session identifier on the wire.** iOS cannot name a session,
and Tower cannot tell iOS which session a result belongs to. A World
Builder `session_id` exists only on disk and is never transmitted.

### 2.2 Capture (raw dataset recording) — **EXISTS (OFF BY DEFAULT)**

`tower/capture.py::CaptureRecorder` writes raw frames to disk. It is
armed by `stream_start` and stopped by `stream_stop` **or by any exit
from the WebSocket handler**, including a crash or a walk-out-of-range
disconnect. That `finally` is a privacy guarantee, not tidiness: without
it a recorder armed by one connection would keep recording the next
connection's frames with no `stream_start` behind them.

Bounds (`CaptureLimits`): `max_seconds = 900.0`, `max_bytes =
1_073_741_824`. Reaching either stops the recording with end reason
`bounded_limit` and returns `False` — never an exception, because
session teardown must not be at the mercy of a disk quota.

**How it is armed:** `app.state.frame_observers` is a **list** of
observers, read defensively (`getattr(..., None) or []`). Each observer
must expose `is_recording`, `write_frame(raw_bytes, *, source_seq,
wire_seq, tx_seq, width, height)` and `stop(reason)`. Observers are
isolated from each other and from the client — an exception in one is
logged and swallowed.

### 2.3 What iOS is told about recording — **MISSING**

Nothing. There is no message type announcing that recording started, no
field on `frame_result`, and no `/health` field. `06-PRIVACY-DATA.md`
requires recording state to be clearly indicated; today that indication
exists only in Tower's server-side log and in the local process that
armed it.

---

## 3. World Builder: the persisted artifact

> **Heading corrected 2026-08-23.** This section was titled "there is no
> network surface". There is one now -- `docs/contracts/CARTRIDGE-RESULTS.md`
> -- carrying STATES, COUNTS and SUMMARIES. Everything below still
> describes the persisted artifact itself: the records, the pose
> convention, the scale semantics. None of that crosses the wire, and the
> conventions in 3.6 and 3.7 remain what any future geometry transport
> would have to honour.

**Stated plainly: no HTTP route and no WebSocket message exposes any
World Builder data.** World Builder is not registered as a module, has no
descriptor, and appears nowhere in `tower/main.py`. Nothing outside
`tower/world_builder/` imports it — a test enforces that
(`tests/test_architecture_boundaries.py`).

An iOS World Builder feature therefore has **no transport at all** today,
not a partial one. The sections below describe the artifact a future
transport would have to carry, so that whoever builds it does not invent
a different representation.

### 3.1 How World Builder is driven today — **EXISTS**

Three separate processes, never concurrent on one world:

```
glasses -> iOS -> WS /ws -> CaptureRecorder -> <root>/captures/<id>/frames/*.jpg
                                                     |
                             scripts/world_build_session.py --frames <dir>
                                                     |
                                      <root>/worlds/<world_id>/...
                                                     |
                                       scripts/world_inspect.py --world <id>
```

The engine's own docstring is explicit that live-versus-offline is a
*driver* choice: `scripts/world_build_session.py` calls exactly the
`engine.observe()` an in-process module adapter would call.

### 3.2 Lifecycle — **EXISTS** (`tower/world_builder/engine.py`)

| Call | Signature | Effect |
|---|---|---|
| `create_world` | `(display_name=None) -> str` | Mints a `world_id`, writes `world.json` |
| `start_session` | `(world_id, *, intrinsics=None, frame_source="unknown", declared_size=None, capture_id=None) -> str` | Acquires the world writer lock, mints a `session_id` |
| `observe` | `(raw_bytes, *, received_at=None, source_seq, wire_seq=None, tx_seq=None) -> ObserveResult` | The cheap per-frame path. Raises `SessionNotActiveError` with no session |
| `stop_session` | `(reason="stop") -> SessionSummary` | Releases the writer lock |
| `build` | `(world_id, session_id) -> BuildResult` | The expensive offline reconstruction |

`ObserveResult`: `outcome`, `reason`, `keyframe_id`, `frames_observed`,
`keyframes_accepted`.
`SessionSummary`: `session_id`, `frames_observed`, `keyframes_accepted`,
`rejected_by_reason`, `segments`, `end_reason`.
`BuildResult`: `world_id`, `session_id`, `backend_id`, `keyframes`,
`poses_solved`, `poses_refused`, `points`, `segments`, `scale_state`,
`downgraded_from`, `diagnostics`.

Session end reasons: `stop`, `disconnect`, `interrupted`, `error`,
`bounded_limit`. `interrupted` is stamped by recovery on a session whose
process died, and its `ended_at` stays `null` — we genuinely do not know
when it stopped, and unknown stays unknown.

### 3.3 Identifiers — **EXISTS**

| Id | Format | Stability |
|---|---|---|
| `world_id` | `uuid4().hex` (32 lowercase hex) | Opaque and permanent. Survives rename and rebuild |
| `session_id` | `uuid4().hex` | Opaque and permanent |
| `keyframe_id` | `"{session_id}:{source_seq:08d}"` | Derived, stable. `source_seq` alone is only monotonic *within* a session |
| `capture_id` | `uuid4().hex` | Separate subsystem; a session may reference one via `Session.capture_id` |

Never parse a `world_id` or `session_id`. `keyframe_id` is documented as
composite only so a consumer can recognise it; construct it with
`records.make_keyframe_id`.

### 3.4 The persisted world — **EXISTS**

```
<root>/worlds/<world_id>/world.json
<root>/worlds/<world_id>/LOCK
<root>/worlds/<world_id>/sessions/<session_id>/session.json
<root>/worlds/<world_id>/sessions/<session_id>/keyframes.jsonl
<root>/worlds/<world_id>/sessions/<session_id>/edges.jsonl
<root>/worlds/<world_id>/sessions/<session_id>/events.jsonl
<root>/worlds/<world_id>/sessions/<session_id>/images/<source_seq:08d>.jpg
<root>/worlds/<world_id>/derived/manifest.json
<root>/worlds/<world_id>/derived/<session_id>/poses.json
<root>/worlds/<world_id>/derived/<session_id>/points.json
```

Default root `data/world_builder`.

**Crash safety.** Every whole-file JSON write is
write-to-`.tmp` → flush → fsync → atomic replace
(`tower/storage.py::write_json_atomic`). Keyframe images and capture
frames use the same pattern by hand. Journals are append-only JSONL, read
by `read_raw_jsonl`, which tolerates and logs a torn final line rather
than failing the whole read.

**The ordering rule a consumer may rely on:** the image is written **and
fsynced before** its journal line is appended. A journal line therefore
always points at a complete image. An orphan image is possible and
harmless. This is what makes tailing safe.

**Schema.** `SCHEMA_VERSION = 1`, an integer, on every record. A reader
that meets a version it does not know **refuses** — `require_schema`
raises `UnsupportedSchemaError`. There are no compatibility ranges. Any
iOS decoder must adopt the same refusal, because guessing produces a
plausible wrong answer that stays wrong forever.

### 3.5 Records a viewer needs

**`World`** — `world_id`, `created_at`, `updated_at`, `display_name`,
`schema_version`, `frame_revision`, `pose_convention`, `scale`,
`session_ids`, `images_purged`.

**`Session`** — `session_id`, `world_id`, `started_at`, `ended_at`,
`end_reason`, `time_basis`, `frame_source`, `capture_id`,
`declared_width`/`declared_height`, `intrinsics`, `backend_id`,
`backend_requires_intrinsics`, `backend_downgraded_from`,
`backend_downgrade_reason`, `frames_observed`, `keyframes_accepted`,
`rejected_by_reason`, `retains_raw_imagery`, `privacy_tags`, `redaction`.

**`Keyframe`** — `keyframe_id`, `session_id`, `source_seq`, `received_at`,
`image_relpath`, `width`, `height`, `byte_count`, `time_basis`,
`wire_seq`, `tx_seq`, `segment_index`, `sharpness`, `feature_count`,
`selection_reason`, `median_parallax_px`, `overlap_ratio`,
`survival_ratio`, `tracked_count`, `homography_residual_px`, `quality`,
`frame_revision`, `spatial_ref` (reserved, always `null`),
`external_refs` (reserved, always empty).

**A `Keyframe` carries no pose.** Poses live only in the *derived*
artifact, because a pose is a rebuild output and a keyframe is an
observation. A consumer that caches a pose against a keyframe must
re-read it after any rebuild.

**`KeyframeEdge`** — `from_keyframe_id`, `to_keyframe_id`, `matches`,
`inliers`, `inlier_ratio`, `median_parallax_px`, `median_parallax_deg`,
`cheirality_fraction`, `r_h`, `rotation_dominant`, `pose_status`,
`degeneracy`, `quality`, `frame_revision`.

### 3.6 Coordinate conventions — **EXISTS, and self-describing**

`world.json` carries `pose_convention` verbatim so the artifact does not
depend on any code still existing in the form the writer had:

```json
{
  "pose_type": "T_world_camera",
  "quaternion_order": "wxyz",
  "handedness": "right",
  "camera_axes": "opencv_x_right_y_down_z_forward",
  "translation_units": "world",
  "world_axes_origin": "first_keyframe_camera",
  "up_axis": "unknown",
  "pose_dtype": "float64",
  "point_dtype": "float32"
}
```

`require_pose_convention` refuses a world whose convention differs in
**any** key.

Three consequences an iOS renderer must respect:

1. **`T_world_camera` means the translation IS the camera position in
   world coordinates.** Do not compute `-R^T t`. Doing so mirrors every
   camera through the origin and still looks like a plausible map. That
   exact bug was found and fixed once already in this codebase.
2. **`up_axis` is `"unknown"` and that is a fact, not a placeholder.** DAT
   exposes no IMU, so there is no gravity observation. A viewer must not
   assume Y-up; it should offer an orientation control or wait for a
   future floor-plane estimate. Declaring an up axis would be a
   fabricated fact under Rule 3.
3. The world origin is the **first keyframe's camera pose**, whose pose
   row has `status: "anchor"`, identity rotation `[1,0,0,0]` and zero
   translation.

**Derived pose row** (`derived/<session_id>/poses.json`, `{"poses": [...]}`):

```json
{"keyframe_id": "...", "segment_index": 0, "status": "solved",
 "degeneracy": "", "rotation": [1.0, 0.0, 0.0, 0.0],
 "translation": [0.0, 0.0, 0.0]}
```

`status` is one of `unavailable | solved | rotation_only | anchor`.
`rotation` and `translation` are `null` when the backend refused. A
`rotation_only` pose keeps its rotation and has `translation: null` —
render it as an orientation at an unknown position, never at the origin.

`degeneracy` is one of `"" | pure_rotation | low_parallax |
no_correspondence | no_intrinsics`. This is what makes a thin
reconstruction *explainable* rather than merely empty, and a viewer
should surface it.

**Derived points** (`derived/<session_id>/points.json`,
`{"points": [...]}`): each entry is
`{"segment_index": int, "xyz": [x, y, z]}`.

**Segments.** Tracking loss starts a new segment.
**Poses in different segments are not in a common frame.** A viewer must
not draw one continuous polyline across a segment boundary, and must not
compute a distance between points of different segments.

### 3.7 Units and scale — **EXISTS**

`scale.state` is one of `unknown | relative | estimated | measured`.

- `relative` means internally consistent with an **arbitrary** unit, fixed
  by whatever baseline the first solved pair happened to have. It is not
  metric.
- `measured` is the **only** state that licenses printing metres.
  `records.format_distance()` is the single choke point and a test pins
  it. An iOS viewer must implement the same rule.
- `build()` yields `unknown` when no pose solved, and `unknown` when a
  session has more than one segment — two segments of one session were
  measured 4x apart in unit length, so a single number across them would
  be a lie.
- An existing `measured` scale is never clobbered by a rebuild.

`ScaleState` also carries `meters_per_unit`, `method`, `confidence` and a
`history` tuple. **As of V1 the state is never `measured`** — nothing in
the shipped code can establish metric scale.

The UI states the brief asked for (`Scale: Relative` / `Acquiring` /
`Metric / Locked`, confidence Low/Medium/High) map onto `scale.state`
plus `scale.confidence`. `Acquiring` has no corresponding implemented
state and must not be shown.

### 3.8 Timestamps — **EXISTS, and deliberately limited**

Every persisted timestamp is a **float Unix-epoch wall-clock second** and
every record says so via `time_basis: "tower-receipt"`.

**There is no capture timestamp anywhere in the system.** The wire carries
`seq`/`width`/`height`/`format`/`data` plus optional
`source_seq`/`tx_seq` — and no time field at all. Tower receipt time is
therefore the only clock, and Rule 16 forbids presenting it as capture
time. A viewer must label these as "received", never "recorded at".

Timestamps are wall-clock, not monotonic; they can move backwards across
an NTP correction. Do not derive durations from them where precision
matters — the `SessionSummary` counters are the trustworthy quantities.

### 3.9 Confidence — **EXISTS**

`tower/confidence.py`, `CONFIDENCE_VOCABULARY_VERSION = 1`, values
`unknown | low | medium | high`, thresholds `< 0.5 -> low`, `< 0.8 ->
medium`, else `high`.

Stored as a **label, never a score**, so a later threshold change cannot
silently relabel history. An iOS client must render the stored label and
must not recompute it from any numeric field.

`Keyframe.quality` derives from track survival ratio; `KeyframeEdge.quality`
from pose inlier ratio.

### 3.10 Calibration — **EXISTS**

`CameraIntrinsics`: `source`, `model`, `fx`, `fy`, `cx`, `cy`,
`dist_coeffs`, `calibrated_width`, `calibrated_height`,
`reprojection_rms_px`, `view_count`, `calibrated_at`,
`scales_linearly_across_resolutions`.

`source` is one of `unknown | self_calibrated | declared`. There is
deliberately **no value meaning "guessed"**: the published 100-degree
Ray-Ban FOV describes a 3:4 still while the stream is 9:16 through an
undocumented crop, so no legitimate conversion exists.

Intrinsics are **resolution-keyed** and refuse to rescale:
`scaled_to()` raises unless `scales_linearly_across_resolutions is True`,
and that flag is currently `null` because nobody has established it.
`build()` refuses outright (`IntrinsicsResolutionMismatchError`) if the
keyframes' resolution differs from the calibrated resolution. DAT's
adaptive ladder changes resolution mid-stream, so this matters in
practice.

Intrinsics are produced only by `scripts/calibrate_charuco.py` and are
stored **per `Session`**, plus whatever JSON file the operator passes.
There is no device-keyed calibration profile store — see §6.4.

### 3.11 The incremental update stream — **EXISTS as a file**

`events.jsonl` is append-only JSONL; each line is
`{schema_version, event_id, kind, at, time_basis, payload}`.

`event_id` is **dense within a session on purpose**: a gap means an event
was genuinely dropped, so a consumer can always tell it missed something.

The kind vocabulary is a closed set of ten. Six are actually emitted
today:

| Kind | Emitted? | Payload |
|---|---|---|
| `session_started` | yes | `{frame_source}` |
| `frame_rejected` | yes | `{reason}` |
| `tracking_lost` | yes | `{segment_index}` |
| `solve_chain_broken` | yes | `{segment_index}` |
| `keyframe_accepted` | yes | `{keyframe_id, reason, segment_index}` |
| `session_stopped` | yes | `{end_reason}`, plus `capture_end_reason` when a live capture was being followed. `end_reason` is what happened to the WORLD; `capture_end_reason` is what happened to the LINK, and they differ — a capture that ended `disconnect` still counts as a finished walk. Absent, not null, when the caller had no follower to ask. |
| `segment_started` | reserved | — |
| `backend_downgraded` | reserved | — |
| `mapping_stalled` | reserved | — |
| `build_completed` | reserved | — |

`solve_chain_broken` announces that the SOLVE chain failed and a new
segment was opened. It is deliberately **not** `tracking_lost` and must
not be treated as one: tracking is healthy at that moment and keeps its
reference. Reading it as a tracking loss would tell a wearer they had
lost the world when they had not. It does not participate in
`last_tracking`.

Like `tracking_lost` it carries the index of the segment it OPENS, and it
is journalled after the keyframe that preceded it, so a consumer
rebuilding segmentation by streaming the journal should attribute
keyframes by their own `segment_index` rather than by position relative
to this event.

A consumer must tolerate a kind it does not know by ignoring it, and must
not assume the reserved kinds will never appear.

There is **no pub/sub bus and no subscription API**, deliberately —
nothing can subscribe in-process while the wire contract is scalar-only, so
a bus would be machinery with no consumer. The journal *is* the stream.

### 3.12 Errors a consumer must handle

| Exception | Raised when |
|---|---|
| `UnsupportedSchemaError` | Persisted `schema_version` differs from the reader's |
| `UnknownPoseConventionError` | `world.json` pose convention differs in any key |
| `WorldLockedError` | Another **live** process holds the world writer lock |
| `SessionNotActiveError` | `observe()` / `stop_session()` with no session |
| `ImagesPurgedError` | `build()` on a world whose imagery was purged |
| `IntrinsicsResolutionMismatchError` | Intrinsics resolution differs from keyframe resolution |

`WorldLockedError` checks whether the recorded pid is actually running, so
a crashed process does not lock a world forever.

---

## 4. Privacy posture iOS must reflect

- World Builder is the **first** module that retains raw imagery.
  `Session.retains_raw_imagery = true` and `privacy_tags` carry
  `raw-imagery`, `first-person`. Both shipped CV modules truthfully
  declare `false`; that flip is the visible signal the posture changed.
- `Session.redaction` is currently the literal string `"none"`. It records
  provenance so that the day redaction ships, an older session's imagery
  can still be identified as unredacted. **A viewer must not assume any
  persisted keyframe is redacted.** Faces are not blurred.
- Every persisted pixel passes through exactly one function
  (`store.write_keyframe_image`, `capture.write_frame`). Whatever
  redaction policy is eventually chosen is a change to one place.
- `purge_world()` and `CaptureRecorder.purge()` perform real deletion and
  **report what they could not delete** (`PurgeReport.retained`,
  `PurgeReport.complete`). A purge that cannot delete everything must
  never be presented to the user as success.
- Bystander policy is **unresolved and deliberately not decided** by
  Tower. iOS must not imply one.

---

## 5. Coordinate/units summary card

| Question | Answer |
|---|---|
| Handedness | Right |
| Camera axes | OpenCV: +X right, +Y **down**, +Z forward |
| Pose meaning | `T_world_camera`; translation **is** camera position |
| Quaternion order | `wxyz` |
| Up axis | **Unknown.** No IMU exists |
| Origin | First keyframe camera |
| Units | "world units". Metres only when `scale.state == "measured"`, which never happens in V1 |
| Point type | float32 `[x,y,z]` |
| Timestamps | float Unix epoch seconds, wall clock, `tower-receipt` basis |
| Cross-segment comparison | **Invalid.** Segments do not share a frame |

---

## 6. What iOS needs that Tower does NOT provide

Each item names the missing capability and where it is blocked. None of
these are implemented; do not code against them.

### 6.1 Any World Builder transport at all — **RESOLVED 2026-08-23**

> **Superseded.** A structured Tower→iOS result channel now exists and
> World Builder is its first producer. See
> **`docs/contracts/CARTRIDGE-RESULTS.md`** for the implemented contract
> and **`docs/contracts/IOS-TO-TOWER-RECONCILIATION.md`** for how every
> iOS requirement was classified against it.
>
> **What this section got wrong**, and it is worth stating because the
> same conflation is easy to repeat: it treated "a World Builder
> transport" and "World Builder as a live in-process module" as one thing.
> They are separable. Every blocker in the table below is a property of
> the *second*. None of them bear on a READER that never joins the frame
> path — one that polls what `world_build_session.py` has already
> persisted and reports it.
>
> Still blocked, and correctly so: World Builder (or any cartridge)
> running as a registered production module, publishing results as it
> computes them. That is what the four rows below still describe, and it
> is what Scene Understanding needs, because it persists nothing and so
> has no file for a reader to read.
>
> Also **not** provided, deliberately: keyframes, poses, points, raw
> events, and imagery. The channel carries states, counts and summaries.
> `IOS-to-Tower.md` §1.4 marks a pose array NOT REQUESTED, and §5 requires
> unredacted imagery to be withheld.

The four blockers, which still describe the LIVE MODULE path:

| Location | Gap |
|---|---|
| `ws.py` | `process()` is synchronous on the event loop and takes only `bytes`; `observe()` needs `received_at`, `source_seq`, `tx_seq` |
| `ExperimentResult` | Scalars plus a `name -> number` bag — cannot carry a keyframe decision or a world delta |
| `main.py::_build_cv_module` | A registry of one. A second module id **is** the V1.0 trigger, and V1.0 is untriggered |
| `container.LIFECYCLE_TIMEOUT_S` | A stop-time `build()` would exceed it. That bound is V1.1 hardening, and V1.1 is **BLOCKED** on an unrecorded user ruling |

The engine is complete behind a clean interface. The future Tower-side
diff is a small `Module` subclass, one branch in `_build_cv_module`, and
passing frame metadata through `process()`.

### 6.2 Live progressive world rendering on the phone — **PARTLY RESOLVED 2026-08-23**

The transport now exists (§6.1), so *live progressive **state*** is
available: lifecycle, keyframe count, mapping seconds on the Tower's
clock, tracking, calibration, and — with
`world_build_session.py --rebuild-every N` — geometry and trajectory
figures that update during the walk.

**Live progressive GEOMETRY RENDERING remains blocked**, and on two
counts rather than one:

1. No geometry data is sent. `element_count` is a count; there are no
   points and no poses on the wire, because `IOS-to-Tower.md` §1.4 marked
   a pose array NOT REQUESTED and no representation contract has been
   agreed.
2. Without `--rebuild-every`, geometry does not exist until after Stop.
   `Start → Walk → Stop → the world appears` remains the honest default;
   `--rebuild-every` is what turns it into *watch it build*, and that is a
   choice made when the build session is launched, not by the phone.

### 6.3 A device-keyed calibration profile store — **MISSING**

Intrinsics exist per `Session` and per operator-supplied JSON file. There
is no store keyed by device/source, so iOS cannot ask "do we have
calibration for this pair of glasses at this resolution?". The record
already carries everything such a store would need
(`calibrated_width`/`calibrated_height`, `source`, `calibrated_at`,
`reprojection_rms_px`, `view_count`); only the keyed lookup is absent.

### 6.4 Recording-state indication on the wire — **MISSING on the socket,
AVAILABLE over HTTP**

`GET /health` reports `capture.armed`, `capture.recording`,
`capture.capture_id` and `capture.frames_written`, so a client can display
the truth. There is still no push notification of a recording state
change, and no way to arm it from iOS (§6.11).

### 6.5 A session identifier on the wire — **MISSING**

See §2.1. iOS and Tower cannot name the same session.

### 6.6 A capture timestamp — **REQUIRES iOS WORK**

Tower cannot invent one. `CMSampleBuffer` carries a presentation
timestamp, but whether it reflects on-glasses capture time is
unconfirmed (`07-PLATFORM-CONSTRAINTS.md` Limitation 9). Until iOS sends
a time field with documented semantics, every timestamp in the system
stays `tower-receipt` and must be labelled that way.

### 6.7 Multi-consumer frame distribution — **BLOCKED**

`ModuleContainer` is a registry of one; two cartridges cannot receive
frames simultaneously. `frame_observers` is a genuine list, but it is a
*side-errand* channel — an observer returns no result to the client — so
it is not a substitute for a second active module.

### 6.8 Resolution negotiation — **MISSING, and now measured**

DAT's adaptive ladder drops resolution first under bandwidth pressure and
cannot be overridden. A Document/OCR cartridge wanting high-resolution
stills has no way to ask for them.

**This stopped being a hypothetical on 2026-08-22.** Document Memory
measured word recall against known rendered text — the fraction of a
page's words OCR actually captured, which is what makes a document
findable at all:

| Frame size | Word recall |
|---|---|
| 1280×720 | 0.957 – 1.000 |
| 640×480 | 0.905 – 1.000 |
| **640×360 — what iOS delivers today** | **0.429 – 0.810** |

A page inside a 640×360 frame warps to roughly 500×320, putting ordinary
body text at about 10 px. **Tilt barely matters; resolution dominates** —
the spread across a full range of viewing angles at 640×480 is only
0.905–1.000, so perspective is handled and pixels are not.

**Page DETECTION still works at 640×360.** Only recognition is starved,
which means the cheap per-frame machinery is fine and the requirement is
narrowly about the frames that get read.

**What iOS would need to provide**, stated as a requirement rather than a
design:

1. A way for a Tower-side consumer to request a **higher-resolution
   frame**, at least occasionally — this cartridge needs one or two per
   document, not a sustained high-rate stream. A ~1280×720 still on
   demand would move recall from ~0.5 to ~0.96.
2. Failing that, a way to learn **which rung of the adaptive ladder is
   currently active**, so a consumer can record that a reading was taken
   at a resolution too low to trust rather than storing a bad one
   silently.

Rule 4 still forbids designing a generalised negotiation protocol before
the real DAT configuration model is known via `search_dat_docs`. This
section states the requirement and the measurement; the mechanism is a
Mac-side question.

### 6.9 A structured result channel — **RESOLVED for persisted state,
STILL BLOCKED for live per-frame results**

> The envelope, subscription, ordering, coalescing, reconnect and error
> machinery now exist and are generic — see
> `docs/contracts/CARTRIDGE-RESULTS.md` §9 for what adding a cartridge
> costs (one file, one registry entry, one contract id).
>
> **The Scene Understanding case below is still unmet**, and the reason is
> not the wire any more. Scene Understanding is a live in-process state
> that **persists nothing** — deliberately, and enforced by test. This
> channel reads persisted state, so there is nothing for it to read, and
> giving that cartridge a store to make it publishable would pre-empt
> Environmental Memory's entire reason to exist. It needs the live-module
> path in §6.1, not this one.
>
> What the channel does prove is that the *refusal* requirement below is
> satisfiable: World Builder's payload carries `available: false` with a
> prose reason, `null` counts that are never `0`, and
> `build_in_progress: null` distinguishing "not running" from "cannot
> tell". A wire that can carry a refusal now demonstrably exists.

The original case, still accurate:

§1.4 says `frame_result` cannot carry a structured object. Scene
Understanding is what that costs, stated concretely rather than in the
abstract.

It produces, per frame: a list of tracks (each with an id, a class, a
box, an age and a facing estimate), a list of relations between them, and
a set of counts. **None of it can reach iOS.** `metrics` is
`name -> number`, so a count could cross — `{"person": 2}` — but a box, a
track id, a relation or a refusal could not, and a count without the
refusal beside it is the more dangerous half.

Specifically: *"how many people appear to be facing my direction"* has
**three** possible answers — a number, "none", and **"we never
measured"** — and only the first two fit in a number. A wire that can
carry `0` but not `answered: false` would turn a refusal into a
confident, wrong zero at the boundary. Whatever eventually carries this
must be able to carry a refusal.

### 6.10 Resolution: what Scene Understanding does NOT need

Worth recording next to §6.8 so the two are not confused, because they
pull in opposite directions.

`ssdlite320_mobilenet_v3_large` resizes internally to 320, and its cost
is essentially flat across input resolution:

| Resolution | Detection |
|---|---|
| 640×360 | 33.1 ms |
| 896×504 | 36.8 ms |
| 1280×720 | 34.4 ms |

So **object detection gains nothing from a higher-resolution frame** —
the extra pixels are discarded by the model and cost only decode time.
Document Memory's request in §6.8 is for the frames it OCRs, one or two
per document, and not for the stream in general. Raising the whole
stream's resolution would pay Document Memory's cost for every cartridge
and buy this one nothing.

### 6.11 A way to arm capture from iOS — **MISSING**

`TOWER_CAPTURE_ROOT` arms the recorder at Tower start-up, and
`stream_start`/`stream_stop` bound each recording. There is no way for
iOS to arm or disarm it, and no way for the wearer to. `GET /health`
reports the state, so a client can at least display it truthfully.

---

## 7. Standing obligations for whoever changes either side

1. **A cross-machine feature is not complete because one side shipped its
   half.** Update this document in the same change.
2. **Do not invent an endpoint here.** If iOS needs something Tower lacks,
   add it to §6 as a missing requirement.
3. **Additive wire changes only** while `frame_result` remains
   scalar-shaped. A sender that omits `source_seq`/`tx_seq` must keep
   working; both fall back honestly. A receiver that has never heard of
   `metrics` must keep working; it is omitted when empty. This is the
   rule `metrics` itself was added under, and it is the rule the next
   field must follow.
4. **Never present `tower-receipt` time as capture time** (Rule 16).
5. **Never render world units as metres** unless `scale.state ==
   "measured"` (§3.7).
6. **Never invert a `T_world_camera` pose** (§3.6).
7. **Refuse an unknown `schema_version`**; never guess (§3.4).
8. **Contract identifiers on the result channel are opaque strings
   compared for equality** — never parsed, never ordered, never assumed
   compatible. Changing one is a deliberate wire break that tells iOS to
   update the app. See `docs/contracts/CARTRIDGE-RESULTS.md` §2.
9. **The result channel is a READER.** Nothing in `tower/results/` may
   write, build, or join the frame path; two boundary tests enforce it.
10. **No imagery crosses the wire while `redaction` is `"none"`.** An
    unstated or absent treatment means withhold, with no lenient default
    (`IOS-to-Tower.md` §5).
