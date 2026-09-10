# World Builder on iOS — what to change, and why

> **SUPERSEDED — 2026-08-26.** iOS ownership moved to the Mac lane and the
> current instructions live in `docs/agent-handoffs/IOS-EXECUTION-PLAN.md`,
> with wire truth in `docs/contracts/WORLD-BUILDER-GEOMETRY.md`,
> `docs/contracts/OBJECT-MEMORY.md` and `docs/contracts/WORLD-BUILDER-IOS.md`.
> Kept for its reasoning; do not execute from it.

**Written:** 2026-08-25
**Tower branch:** `integration/world-builder-lifecycle-v1`
**Audience:** whoever holds the Mac.

This document exists because the iOS half of World Builder **could not be
written on the machine that did tonight's Tower work**. There is no Swift
toolchain there (`xcodebuild`, `swift`, `swiftc` all absent — it is
Windows), and more importantly the Tower-backed `WorldBuilderClient` that
displayed "Building / Keyframes 143" on 2026-08-24 **is in no branch of
this repository**. Every ref — `main`, `ios-origin/main`,
`ios/cartridge-integration`, `ios/integration-candidate`,
`ios/fix-camera-start-regression` — contains only
`UnavailableWorldBuilderClient`, whose `state` is a constant
`.unsupported`. `ios-origin`'s newest commit is 2026-08-23 00:01, before
the test.

So that client lives on a Mac and is unpushed. Writing iOS code against
this tree would have been unbuildable, unverifiable, and likely to
collide with it. **Push it first.** Everything below is written to be
applied on top of whatever it actually says.

---

## 1. The one thing that must change: bind the world to the capture

### The defect

On 2026-08-24 the phone showed **camera LIVE** and **"Capture has
ended."** at the same time, with frozen figures. Both were telling the
truth about different machines.

`cameraStreamState` is DAT's opinion of the phone-to-glasses link.
`WorldModelState` is the Tower's opinion of *a world directory on the
Tower's disk*. Nothing in the iOS tree — no view, no view model, no test
— asserts any relation between them, and `WorldCanvasView` renders
`WorldModelState` "and nothing else" by design.

What actually happened, from the capture manifests:

```
2e6cff  t=0.0   -> 121.9  1395 frames  disconnect   <- the one being followed
                 ...105 s with no capture at all...
341b0f  t=226.8 -> 256.9   259 frames  disconnect
...eight more, none followed by anything...
```

The WiFi dropped at t=121.9 s. The successor arrived 105 s later, past
the 90 s resume grace, so the follower correctly finalised and exited.
The wearer kept walking. Nine further captures were recorded that
**nothing was reading**, because on that build nothing ever started a
follower. The result channel, asked for "the world" with no `world_id`,
answered with *the most recently updated* one — a finished world from
earlier — which iOS faithfully rendered as `.finalized`, frozen, beside
a LIVE badge.

### What the Tower now does about it

The Tower attaches a builder to every capture automatically, at the
moment the capture id is minted (`tower/capture_workers.py`). One worker
per capture *lineage*, so a reconnect does not fork the walk. That
removes the cause. It does **not** remove the class of bug: the result
channel can still legitimately report a world that is not this session's
— that is what inspection mode is for.

### The invariant iOS must add

> **No `WorldModelState` carrying a snapshot may be rendered as this
> session's result unless iOS can establish that the snapshot belongs to
> the capture iOS currently has open.**

The payload already carries what is needed: `payload.session.capture_id`
and `payload.session.ended_at` are in the contract today.

What is missing is the phone's half — iOS does not know its own capture
id, because `stream_start` gets no reply. **This is the one wire
addition needed**, and it is small:

```json
{"type": "stream_started", "capture_id": "6bf1c84c92f94fb68db62d5ba24c3ad2"}
```

Sent by the Tower in response to `stream_start`. Note what this is
**not**: it is not a field in `stream_start`. That payload stays exactly
`{"type":"stream_start"}`, pinned by
`TowerClientTests.testStreamStartSendsExactPayloadOnce`, and nothing
here asks iOS to supply an identity. The Tower mints it and reports it,
which is the direction `handoff.md` §9.2 already specifies for world
ids.

It arrives as an ordinary inbound message in `handleInboundMessage`,
alongside `frame_result`. `TowerClient` awaits no reply for anything but
the pong, so this is not request/response correlation — it is a fact the
Tower volunteers. **This message is NOT yet implemented on the Tower**;
see §5.

### The gate

```swift
enum WorldSessionBinding: Equatable, Sendable {
    case none                          // no bracket open
    case awaiting(captureID: String?)  // bracket open, no matching snapshot yet
    case bound(captureID: String)      // snapshot's capture_id is ours
    case foreign(captureID: String?)   // snapshot describes some other capture
}
```

| Camera bracket | Tower `model_state` | Binding | iOS state |
|---|---|---|---|
| open | nothing received yet | `.awaiting` | `.awaitingFirstUpdate` |
| open | matches ours, `receiving` | `.bound` | `.receiving(snapshot)` |
| open | **foreign**, any state | `.foreign` | **`.awaitingFirstUpdate`** |
| closed | matches, `stopped_unbuilt`, geometry available | `.bound` | `.finalizing(snapshot)` |
| closed | matches, `stopped_unbuilt`, no geometry | `.bound` | `.interrupted(snapshot)` (since 2026-09-10) |
| closed | matches, `ready` | `.bound` | `.finalized(snapshot)` |
| closed | foreign, `ready` | `.foreign` | `.finalized` **only** under `.inspecting(worldID:)` |

One rule does the work: **a foreign snapshot while a bracket is open
renders as "waiting", never as a result.**

Note the Tower never sends `awaiting_first_update` and never will —
§10.0 of the contract calls it "a fact about the phone's own situation,
which only the phone can know". iOS must synthesise it, and today
nothing does.

### The sentence that was missing

When the camera is streaming and the world has been `.awaitingFirstUpdate`
for more than a bounded interval, say so:

> *"Frames are reaching the Tower. Nothing is building a world from them
> yet."*

True, actionable, and exactly what nobody could see on 2026-08-24.

---

## 2. Contract identifier: adopt `world_builder.status/2026-08-25`

The identifier moved from `.../2026-08-23`. **A field changed meaning**,
which is why it moved rather than staying put for an additive change.

`trajectory.pose_count` was `keyframes - poses_refused`. `build()` counts
a segment ANCHOR as neither solved nor refused, so that arithmetic
promoted every anchor to a camera position — and an anchor is
definitional, not measured: identity rotation, zero translation.

**That is where "Camera poses: 36" came from.** The manifest behind it
reads `backend_id: "unposed", poses_solved: 0, points: 0, segments: 36`.
Nothing was reconstructed. 36 was the segment count, rendered as a
trajectory.

`pose_count` is now counted per segment: every solved pose, plus the
anchor of each segment that solved something. A build with
`poses_solved: 0` reports **0**. A new sibling field `poses_anchor`
reports the anchors separately, so an uncalibrated walk reads as "36
segment origins, no trajectory".

**What iOS must do:** if it pins the contract identifier, adopt the new
one — deliberately, having read the paragraph above. If it does not pin
it, nothing breaks; the key set is unchanged apart from the added
`poses_anchor`, which iOS may ignore.

Expect `ProductShellTests.testTheTowerDeclaresNoCartridgeContracts` to
fail the moment World Builder is added to `TowerCapabilities.declared`.
That failure is the designed review trigger, not a nuisance.

---

## 3. What iOS will actually see now, and what it still will not

Set `TOWER_CAPTURE_ROOT` and `TOWER_WORLD_ROOT` (both done for you by
`scripts/setup_tower.ps1`) and the flow is:

| | During the walk | After Stop |
|---|:--:|:--:|
| `keyframe_count` | climbs live | final |
| `tracking` | live | final |
| `mapping_seconds` | live | final |
| `world_id`, `persistence` | live | live |
| `geometry`, `trajectory` | **now appears mid-walk** | final |
| `frames_observed` | `null`, with a reason | final |

Geometry appearing mid-walk is new: the Tower attaches the follower with
`--rebuild-every 4` rather than the script's own default of `0`, which
means "build once, at the end" and is why nothing appeared until the
capture closed on 2026-08-24.

**What will still be zero, on real hardware, until a calibration exists:**

```
calibration: "uncalibrated"    scale: "unknown"
pose_count:  0                 geometry.element_count: 0
```

This is correct and iOS should render it as such. No intrinsics exist for
the Ray-Ban camera, so the backend that solves poses cannot run and
withholds every pose. See `docs/CALIBRATION.md` for the physical
procedure. **Do not treat zeroes here as a bug to work around.**

---

## 4. A trajectory viewer — the decision, not the implementation

`docs/agent-handoffs/WORLD-BUILDER.md` §8 says no poses, points or paths
cross the wire, on the grounds that "building a transport for a consumer
that does not exist is the fabricated contract this project refuses".
That rule is sound and it is why nothing was added tonight.

But one of its five stated grounds has been falsified and should be
recorded. It says iOS "has no pose schema", and `IOS-to-Tower.md` §1.4
elaborates: a pose schema needs position, rotation convention,
handedness, coordinate frame and units — "five Tower decisions, each of
which renders plausibly and wrongly if guessed".

**All five are decided, written down, and already in every world
artifact** (`TOWER-TO-IOS.md` §3.6):

```json
{"pose_type": "T_world_camera", "quaternion_order": "wxyz",
 "handedness": "right", "camera_axes": "opencv_x_right_y_down_z_forward",
 "translation_units": "world", "world_axes_origin": "first_keyframe_camera",
 "up_axis": "unknown", "pose_dtype": "float64", "point_dtype": "float32"}
```

Sending that object verbatim turns the objection into a decode. The
remaining ground — "the consumer does not exist" — is a statement about
scheduling, and it stops being true the day someone writes the viewer.

If and when that happens, the minimal honest payload is:

1. **`pose_convention`** — the nine keys above, verbatim. iOS compares
   every key against what it implements and **refuses to render on any
   mismatch**. Non-negotiable: inverting `T_world_camera` still produces
   a plausible-looking map, and that was a real bug once.
2. **`trajectory.poses`** — aligned with keyframe order, each
   `{segment_index, status, translation | null, rotation_wxyz | null,
   degeneracy}`. **`translation: null` must survive** so the viewer draws
   a break rather than a line through a gap.
3. **`geometry.points`** plus `points_sent`, `points_total`,
   `point_sampling` — so a truncated cloud can never be read as the whole
   one.
4. **`bounds`** — so the viewer can frame the scene without scanning.

Two things to hold onto:

- **Points must be opt-in and budgeted.** The payload is byte-constant at
  3,173 B today and the whole resource table depends on that. Poses are
  ~4 KB at 64 keyframes and bounded by a count already reported; points
  are 60–100 KB and re-sent on every revision change at up to 2 Hz.
- **`up_axis` is `"unknown"`.** A 2D top-down `(x, z)` `Canvas` with
  pinch and drag needs no 3D framework and is the *more* honest first
  view. SceneKit only earns its weight once a floor plane exists.

**`image_relpath` and every byte of keyframe imagery stays Tower-side.**
Redaction is a *process* claim, not an outcome claim;
`retains_raw_imagery` is permanently `true`. A trajectory and a point
cloud are geometry. The frames are not.

---

## 5. Not implemented on the Tower — deliberately

- **`stream_started` with `capture_id`** (§1). It is one message and the
  Tower already has the id at hand, but it is a wire change whose only
  consumer is the iOS gate, and that gate is not written. Ask for it and
  it is a small change.
- **Pose and point arrays** (§4). Same reason, larger.

Both are blocked on the same thing: a consumer. Neither is blocked on
Tower work.

---

## 6. Order of work

1. **Push the Mac's `WorldBuilderClient`.** Nothing here can be reviewed
   against a tree that does not contain it.
2. Adopt `world_builder.status/2026-08-25`.
3. Add the session binding and the foreign-snapshot gate (§1). This is
   the correctness fix and it needs no new Tower work except
   `stream_started` — until that lands, `.foreign` can only be detected
   after Stop, which is strictly better than today but not complete.
4. Add the "frames are arriving, nothing is building" sentence.
5. Only then consider the viewer (§4).

## 7. Tests worth writing first

The existing harness needs no extension — `MockTowerServer` takes a
`server.send(text:)` of any JSON, and
`testUnknownInboundMessageDoesNotKillConnection` already proves the
client survives an unrecognised inbound type.

- **A world from a previous run is not shown as this session's.** Open a
  bracket, feed a `cartridge_result` whose `session.capture_id` is not
  ours with `model_state: "finalized"`, assert the state is
  `.awaitingFirstUpdate`. This is the load-bearing negative and it is the
  2026-08-24 bug.
- A matching snapshot while the bracket is open reaches `.receiving`.
- A matching snapshot after Stop reaches `.finalizing` then `.finalized`.
- A malformed payload produces `CartridgeFailure(kind: .undecodableResponse)`,
  not a partly-populated snapshot.
- `pose_count: 0` with `keyframe_count: 155` renders as no trajectory and
  **not** as a missing value — `nil` and `0` are different claims all the
  way to the screen.
- The wire stays byte-silent when the World Builder workspace is opened
  and closed (`testCartridgeViewModelsSendNothingToTheTower` already
  asserts this; keep it true).
