# World Builder — the Tower↔iOS boundary, reconciled

> **iOS ENTRY POINT IS `docs/agent-handoffs/IOS-EXECUTION-PLAN.md`.**
> This file is **REFERENCE**: deeper detail, still accurate except where
> the plan says otherwise. Read the plan first — it says what to do now,
> what is already settled, and what the Tower has measured and refused.
> Where the two disagree, the plan wins.

**Living document.** It describes the boundary as it exists now.

**Status:** implemented on both sides and exercised end to end over a real
socket. The Tower half has met the Ray-Ban camera; the iOS half of
`world_builder.status/2026-09-10` has not yet been walked. See
`docs/agent-handoffs/WORLD-BUILDER-INTEGRATION.md` for exactly what has and
has not met hardware.

**The wire itself** is `tower/docs/contracts/CARTRIDGE-RESULTS.md`, which is
the Tower's document and the authority on message shapes. This document is the
part neither side owns alone: **which Tower field becomes which Swift value,
and what happens where the two vocabularies do not line up.**

Contracts in play:

| | |
|---|---|
| Envelope | `cartridge_results.envelope/2026-08-23` |
| World Builder payload | `world_builder.status/2026-09-10` (supersedes `/2026-08-25`: `model_state` gained `interrupted`; the payload gained `selection` and `lifecycle.finalization`) |
| World Builder geometry | `world_builder.geometry/2026-08-25` — **its own document: [`WORLD-BUILDER-GEOMETRY.md`](WORLD-BUILDER-GEOMETRY.md)**. Different transport (HTTP), versioned independently |
| Tower cartridge name | `world_builder` |
| iOS catalog id | `world-build` |

Those last two are **different strings for the same cartridge**, and the
mapping lives in exactly one place: `TowerCapabilities.towerCartridgeNames`.

---

## 1. The shape of the seam

```
 Tower web process ──ws://…/ws──┐
                                │  one socket, six inbound message types
                                ▼
                        TowerClient            decodes the envelope.
                                │              Knows no cartridge.
                                │ cartridgeResults / $cartridgeDeclaration
                                ▼
                    TowerWorldBuilderClient    owns the contract, the
                                │              subscription, and the mapping.
                                │              Built by ProjectManager, so it
                                │              outlives every workspace switch.
                                │              ALSO drives the geometry pull
                                │              below, off geometry.revision.
                                │ stateUpdates
                                ▼
                    WorldBuilderViewModel      republishes into SwiftUI.
                                │
                                ▼
                      WorldCanvasView          renders facts.

 Tower web process ──http://…/worlds/{id}/geometry/{manifest,segment/{i}}──┐
                                │  world_builder.geometry/2026-08-25       │
                                │  PULLED, not pushed. Contract, tables    │
                                │  and rules: WORLD-BUILDER-GEOMETRY.md    │
                                ▼                                          │
                    WorldGeometryClient ── WorldGeometryStore ── WorldFragmentsView
```

**No second socket, no second connection, no view-owned transport** for status.
Discovery, subscription and results share the socket the camera already streams
over. `frame_result` is field-for-field unchanged and pinned by a test.

**Geometry is the one thing that does not travel that way**, and the reason is
in the code: `tower/tower/routes/ws.py:38` gives the result sender and the
frame path a single `asyncio.Lock`, and one session's `points.json` is 1.07 MB
against a 3,884-byte status snapshot. Sending it there would hold that lock and
starve `frame_result`. It is an ordinary HTTP `GET` instead, pulled when the
status payload's `geometry.revision` moves. See
[`WORLD-BUILDER-GEOMETRY.md`](WORLD-BUILDER-GEOMETRY.md).

---

## 2. Reconciliation matrix

Verdicts: **CLEAN** (1:1), **ADAPTER** (small mapping, no duplicated state),
**IOS-ONLY** (no Tower field; the phone is the only machine that can know),
**UNREACHABLE** (modelled on one side, no code path on the other),
**HARDWARE** (correctness depends on something only a physical run settles).

### 2.1 Lifecycle

| Tower `model_state` | Swift `WorldModelState` | Verdict | Notes |
|---|---|---|---|
| `unsupported` | `.unsupported(reason:)` | CLEAN | `model_state_reason` verbatim; a fallback sentence when null |
| `idle` | `.idle` | CLEAN | Tower's reason is dropped — `.idle` carries none. Its prose ("no worlds exist under this Tower's world root") is diagnostic, not actionable |
| `receiving` | `.receiving(snapshot)` | ADAPTER | With `world_snapshot: null` → `.awaitingFirstUpdate`. Unreachable in practice; a live session implies a world |
| `finalizing` | `.finalizing(snapshot)` | ADAPTER | **Read the caveat in §3.** Since 2026-09-06 `lifecycle.build_in_progress == true` means a live process is finishing and the phone may say so |
| `finalized` | `.finalized(snapshot)` | CLEAN | |
| `interrupted` | `.interrupted(snapshot, reason:)` | CLEAN | New at `/2026-09-06`. The session ended abnormally (builder died, was asked to stop mid-walk, recorded an error, or finalization was left unfinished); the snapshot and the geometry still describe what exists. Rendered with its own headline, the Tower's reason, the figures, the fragments and the Picture — never as `.failed`, never as `.finalized` |
| `failed` | `.failed(.towerReportedFailure)` | CLEAN | Attribution matters: the Tower reported it, so it is not `.transport` and not `.notSupported`. No disk state maps to it since `/2026-09-06`; still decoded |
| *(never sent)* | `.awaitingFirstUpdate` | IOS-ONLY | Subscribed, not yet answered. Only the phone can know this, which is why Tower does not send it |
| unknown word | `.failed(.undecodableResponse)` | ADAPTER | A disagreement discovered on arrival, not an empty world |

### 2.2 `world_snapshot` → `WorldSnapshot`

Every field below is optional on both sides, and **null means absent, never
zero**.

| Tower | Swift | Verdict |
|---|---|---|
| `name` | `name` | CLEAN — null means unnamed; no name is derived |
| `world_id` | `worldID` | CLEAN |
| `keyframe_count` | `keyframeCount` | CLEAN |
| `revision` | `revision` | CLEAN — opaque, equality only. Same string as the envelope's |
| `tracking` (`good`/`lost`/`unavailable`) | `WorldTrackingQuality` | CLEAN. `limited` is mapped but **never sent** — it needs a threshold nobody has defined |
| `scale` | `WorldScaleSemantics` | CLEAN — **Tower sends iOS's vocabulary**, not its own. See §4 |
| `mapping_seconds` | `mappingSeconds` | CLEAN — **the Tower's clock.** Never derived from a phone timer. Nothing renders it today |
| `calibration` (`unknown`/`uncalibrated`/`calibrated`) | `WorldCalibrationState` | CLEAN. `calibrating` mapped but **never sent** — calibration is an offline procedure |
| `geometry.representation` | `geometry.representation` | CLEAN — opaque label, displayed verbatim, never parsed |
| `geometry.element_count` | `geometry.elementCount` | CLEAN — shown only beside the label, never alone |
| `geometry.is_incremental` | `geometry.isIncremental` | CLEAN — always `false`; a build replaces the whole tree |
| `trajectory.pose_count` | `trajectory.poseCount` | CLEAN — **meaning changed at `/2026-08-25`.** See §8 |
| `trajectory.poses_anchor` *(payload block, not the snapshot)* | `trajectory.posesAnchor` | ADAPTER — see §8 |
| `trajectory.poses_solved` / `poses_refused` / `segments` *(payload block)* | `trajectory.posesSolved` / `posesRefused` / `segments` | ADAPTER — see §8 |
| `trajectory.path_length` / `_unit` / `scale` | `WorldTrajectoryReport` | ADAPTER — see §4 |
| `persistence.state` / `revision` | `WorldPersistenceState` | CLEAN — `saved` is the only reachable state; `session` is unreachable by construction |

### 2.3 Availability

| Tower | Swift | Verdict |
|---|---|---|
| cartridge absent, or in `not_offered` | `.noContract` | CLEAN. `not_offered` is **not decoded** — both readings mean "the Tower has declared nothing" |
| declared, contract we do not implement | `.unsupportedContract` | CLEAN — tells a person to update the app, not to reconnect |
| declared, `available: false` | `.available` + `.unsupported(reason:)` | ADAPTER — see §5 |
| declared, `available: true`, socket up | `.available` | CLEAN |
| declared, socket down | `.towerUnreachable` | ADAPTER — the cached declaration deliberately survives a drop |

### 2.4 Not consumed, and deliberately

The payload's other blocks — `world`, `lifecycle`, `progress`, `tracking`,
`calibration`, `scale`, `geometry`, `persistence`, `artifacts` — are
**Tower-native evidence** for the two halves above. iOS reads none of them. A
reader looking for where they are consumed will correctly find that they are
not.

**Two exceptions, added at `/2026-08-25`,** each for something the projection
cannot carry:

| Read | Keys | Why the projection cannot answer |
|---|---|---|
| `trajectory` | `poses_anchor`, `poses_solved`, `poses_refused`, `segments` | `world_snapshot.trajectory` keeps `pose_count` and drops these. They are what separates a walk that positioned 36 cameras from one that produced 36 segment origins and positioned none. §8 |
| `session` | `capture_id`, `ended_at`, `frame_source` | Not a figure and never drawn as one. A `WorldSnapshot` describes a directory on the Tower's disk and cannot say *whose capture built it*. §9 |

| Not consumed | Why |
|---|---|
| pose arrays, point clouds | Not on **this** channel, and deliberately: they are bulk, and this socket shares its send lock with the frame path. They travel over HTTP under `world_builder.geometry/2026-08-25` instead — see [`WORLD-BUILDER-GEOMETRY.md`](WORLD-BUILDER-GEOMETRY.md) |
| keyframe images | **The Tower does not send them, on any channel.** `retains_raw_imagery` stays true Tower-side; no byte of imagery crosses to iOS, redacted or not |
| `progress.frames_observed` | Null while live and genuinely unknowable — an ordinary rejected frame writes no journal event |
| `lifecycle.build_in_progress` | `null` in every stopped state, and null means **unobservable**, not `false` |
| `artifacts.*`, `session.retains_raw_imagery` | Real and honest, but no iOS surface asks the question yet. See §6 |

---

## 3. `finalizing` means two things, and the payload says which

Since 2026-09-06 the live builder **keeps the writer lock** through the final
solve and the final build and records a `finalization` block on the session.
`lifecycle.state: "finalizing"` therefore means a live process is finishing
(`build_in_progress: true`, evidence names the pid), and the phone may say
"The Tower is finishing this world." with a progress indicator.

On records older than that change the old projection remains:
`model_state: "finalizing"` from `lifecycle: stopped_unbuilt` means only **"the
stored figures are not the final figures"**, `build_in_progress` is `null`, and
the phone keeps the guarded sentence. `WorldModelState.finalizing` carries
`buildInProgress: Bool?` so the two copies are chosen from the payload, not
guessed.

**Since 2026-09-10, `stopped_unbuilt` does not always project to
`finalizing`.** It carries two states and only one of them means wait:

| `lifecycle.state` | the figures | `model_state` | means |
|---|---|---|---|
| `stopped_unbuilt` | `element_count > 0` or `pose_count > 0` | `finalizing` | built, and BEHIND its keyframes. A rebuild is outstanding; the world is intact. |
| `stopped_unbuilt` | neither above zero | `interrupted` | nothing drawable came of this walk. There is nothing to wait for. |

The second used to project to `finalizing` too, and the phone therefore
showed a **permanent "Finalizing"** over a walk that had produced no
geometry — a state nothing would ever change. Four separate reviews found
that by four separate routes. The distinction is made in the projection
rather than in `lifecycle.state`, deliberately: the state name is on the
wire and iOS decodes it, so it did not move.

The predicate is the FIGURES, not `geometry.available`. `available` is true
as soon as a build ran and left a tree behind, and `engine.build` writes one
unconditionally — a walk down a dark corridor produces `poses.json`,
`points.json` and a manifest reading `points: 0, poses_solved: 0`. The first
version of this rule gated on `available` and therefore kept saying
`finalizing` over a world with nothing in it. It is deliberately the same
question `WorldEvidence.hasGeometry` asks on iOS — `(elements ?? 0) > 0 ||
(poses ?? 0) > 0` — because if the two disagree the Tower tells the wearer to
wait for a screen the phone will never have anything to put on.

**Also since 2026-09-10: `lifecycle.state: "ready"` covers a derived tree that
no manifest describes** — a world built before the Tower wrote one per session,
or one whose manifest cannot be read. `geometry.current` is `false`,
`lifecycle.reason` is non-null and says currency cannot be judged, and
`element_count`/`pose_count` are **recounted from `poses.json` and
`points.json`** rather than left null. That last part is what makes the world
open: the phone decides what to draw from those two numbers, so reporting them
as null over a real reconstruction rendered *"Needs retry — nothing usable came
of this session"* on top of 1,347 points and 4 camera poses.

A client that switches on `model_state` needs no change. A client that
switches on `lifecycle.state` and assumes the old projection should read
`geometry.element_count` and `trajectory.pose_count` beside it.

## 3.1 `selection`: whose world is on the wire

An unpinned subscription is answered with a live world if one exists, else
the most recently updated world on disk. Until 2026-09-06 nothing said which,
and a phone opening World Builder with nothing live drew the newest saved
world — fragments and all — as the live world. The payload now carries
`selection.mode`: `pinned` (the client named it), `live`, `finalizing`,
`latest` (**nothing is live; this is history**), `none`.

The phone's rule: following live, a `latest` selection is **not** the live
world. `TowerWorldBuilderClient` presents it as `.idle` and publishes it as
`recentWorld` — a reference the canvas shows as "Last saved world … Open",
which pins it (History). Geometry coordinates are emitted only for a report
presented with a snapshot, and the view model clears its gallery whenever the
world identity changes, so no earlier world's fragments survive a switch.
`WorldSessionGate` (§9) still applies on top for the DEBUG capture bracket;
an older Tower with no `selection` block keeps the gate-only behaviour.

---

## 4. Scale — the load-bearing refusal

**Reachable states in V1 are `unknown` and `relative`. That is all.**
`inferredMetric` and `measuredMetric` have no code path that produces them on
monocular hardware. Both are still mapped rather than discarded, so that one
arriving later is not silently downgraded.

- `relative` = internally consistent with an arbitrary unit fixed by whatever
  baseline the first solved pair happened to have. **Not metric.**
- `unknown` = **no unit at all**, a strictly weaker claim. Never mapped to
  `.relative`.

Two gates in `WorldTrajectoryReport`, and they answer different questions:

| | Asks | Tower answer today |
|---|---|---|
| `distanceDisplayable` | may this be shown as a **physical distance**? | always false |
| `labelledFigureDisplayable` | may this be shown as **the labelled figure it is**? | true when a unit is present |

The second is new. Refusing on scale alone meant the panel would never show a
path length at all, because `relative` is the best this hardware reaches — and
"6.6 world units", with the Tower's own unit attached and
"Shape and layout only. No real-world distances are claimed." beneath it, is
not a distance claim. **The gate is the unit, not the scale**: a bare number is
what a reader silently reads as metres.

---

## 5. `available: false` is an offer, not silence

A Tower that declares World Builder and cannot serve it right now — no world
root configured, typically — is saying something quite different from a Tower
that has never heard of it, and the two call for opposite responses.

iOS therefore resolves the contract to `.available` (this Tower speaks an
agreement we implement, and we can reach it) and carries the unserveability in
the **domain state**, as `.unsupported(reason:)` with the Tower's prose
verbatim. It does not subscribe. Collapsing the offer to `.noContract` would
render "no world root is configured" as "this Tower will never do this".

The cached declaration also **survives a disconnect**, for the same reason: what
a Tower can do is a property of its build, not of the socket. Clearing it would
turn every WiFi blip into "this will never work" when the truthful reading is
"not reachable".

---

## 6. What iOS does not implement yet

| Missing | What exists Tower-side | What iOS needs |
|---|---|---|
| ~~World picker / reopen a saved world~~ | **Done 2026-09-06.** `GET /worlds` lists worlds (`WORLD-BUILDER-WORLDS.md`), `WorldPickerView` chooses one, `TowerWorldBuilderClient.inspect(worldID:sessionID:)` pins the channel, `WorldInspectionMode.inspecting(worldID:)` is set, and `GET /worlds/{id}/render` (§4 of that contract) shows the picture inside the app | — |
| **Replay** | `WorldView.trajectory(session_id)` returns per keyframe: pose, segment, pose status, and `image_relpath` — a recorded camera path with a real first-person view at each point. `world_inspect.py --trajectory` renders it today | The poses and points now cross the wire (geometry contract) and iOS draws them 2D, per segment. What is still missing is the **first-person view**: `image_relpath` and every keyframe byte stay Tower-side, and a 3D view needs a floor plane that does not exist (`up_axis: "unknown"`) |
| **Privacy disclosure** | `retains_raw_imagery: true` and the redaction process claim are on every session record | A surface that states what the Tower keeps. See §7 |
| **Calibration status/action** | `calibration` state is on the wire and rendered | Nothing invites or explains calibration, and without it there is no geometry at all (§7) |

---

## 7. Two facts a reader will otherwise get wrong

**Starting capture is not starting a build.** The Tower web process writes
frames to a capture and answers `frame_result`. Reconstruction runs in a
**separate process** (`scripts/world_build_session.py --follow-capture`) reading
that capture from disk. Since 2026-09-06 that process is attached to a capture
**only while the `world_builder` cartridge session is active** — the phone
posts `start` when the World Builder workspace appears and `stop` when it
leaves (`TOWER-UNIFIED-CARTRIDGES.md` §4). That is intent, not liveness: the
status channel's `lifecycle` remains the only evidence that a builder is
running, and every string in the workspace still says so.

**Without intrinsics there is no geometry.** `select_backend` downgrades to
`UnposedBackend` when intrinsics are unknown, and no flag invents a focal
length. A real walk with no calibration produces keyframes, `tracking: good`,
`scale: unknown`, `poses_solved: 0` and `points: 0` — truthful, and empty.
Calibration is `scripts/calibrate_charuco.py` against a **printed, physically
measured** board, and the intrinsics must be calibrated at the **delivered
resolution**: `_require_matching_resolution` refuses to apply a 480×360
calibration to 640×360 frames rather than silently scaling the world by the
ratio.

**Redaction is a process claim, never an outcome claim.** The recorded value is
`faces-detected-and-filled/yunet-2023mar@0.30`. Never "redacted", "anonymised"
or "privacy-safe" — YuNet has measured false negatives on faces occluded past
~60% and rotated ~90°, and `retains_raw_imagery` stays **true**: bodies,
clothing, room contents and any undetected face are still in the image. No
imagery crosses to iOS, before or after redaction.

---

## 8. Positioned poses and segment anchors are different figures

**This is why the identifier moved from `/2026-08-23` to `/2026-08-25`.** A
field changed *meaning*; nothing was merely added.

`trajectory.pose_count` was `keyframes - poses_refused`. The Tower's build
counts a segment ANCHOR as neither solved nor refused, so that subtraction
promoted every anchor to a camera position — and an anchor is definitional, not
measured: identity rotation, zero translation.

On the 2026-08-24 physical walk the manifest read `backend_id: "unposed",
keyframes: 155, poses_solved: 0, poses_refused: 119, points: 0, segments: 36`.
Nothing was reconstructed. The channel reported `pose_count: 36` and the phone
displayed **"Camera poses: 36"**.

| Tower now sends | Means |
|---|---|
| `pose_count` | poses carrying a position that is **evidence**: every solved pose, plus the anchor of each segment that solved something. `0` for a build with `poses_solved: 0`, whatever the anchor count |
| `poses_anchor` | how many poses were anchors. **Reported beside the count, never folded into it** |
| `segments` | tracking segments. A break means tracking was lost, and poses either side share no coordinate frame — which is why a path length is refused across more than one |

**What iOS does with them.** `WorldTrajectoryReport` keeps all three separate
and never adds any two. `WorldSummaryView` draws "Camera poses", "Segment
origins" and "Segments" as three rows, and `isAnchorsOnly` — `poseCount == 0`
with anchors present — adds the sentence *"No camera position was
reconstructed. Each origin marks where tracking restarted, not where the camera
was."* An uncalibrated walk therefore reads as **36 segment origins, no
trajectory**, which is what happened.

`0` and `null` stay different claims all the way to the screen: zero is "the
Tower counted none", absent is "the Tower did not say", and only the second one
omits the row.

---

## 9. Which capture the world on screen belongs to

### The defect

On 2026-08-24 the phone showed camera **LIVE** and **"Capture has ended."** at
the same time, with frozen figures. Both halves were telling the truth about
different machines: `cameraStreamState` is DAT's opinion of the phone-to-glasses
link, and `WorldModelState` is the Tower's opinion of *a world directory on the
Tower's disk*. Nothing asserted any relation between them.

The WiFi had dropped past the resume grace, the follower had finalised and
exited, and nine further captures were recorded that nothing was reading. Asked
for "the world" with no `world_id`, the result channel answered with the most
recently updated one — a finished world from earlier — which iOS faithfully
rendered as `.finalized`.

The Tower now attaches a builder to every capture automatically, one per capture
lineage, which removes the *cause*. It does not remove the *class*: the result
channel can still legitimately report a world that is not this session's.

### The invariant iOS now enforces

> No `WorldModelState` carrying a snapshot may be rendered as this session's
> result unless iOS can establish that the snapshot belongs to the capture iOS
> currently has open.

`WorldSessionGate` (`ios/Glasses/Workspaces/WorldBuilder/WorldSession.swift`)
is the one place that decides it, from two inputs: `TowerClient.isStreamingToTower`
— true between a sent `stream_start` and its `stream_stop` — and the payload's
`session` block.

| Camera bracket | Tower says | `WorldSessionBinding` | iOS state |
|---|---|---|---|
| closed | anything | `.none` | the Tower's own state, unchanged |
| open | no `session` at all | `.awaiting` | `.awaitingFirstUpdate` |
| open | `receiving`, `ended_at: null`, `frame_source: "live-capture"`, a `capture_id` | `.bound(captureID:)` | `.receiving(snapshot)` |
| open | anything else | `.foreign(captureID:)` | **`.awaitingFirstUpdate`** |

One rule does the work: **a foreign snapshot while a bracket is open renders as
"waiting", never as a result.**

`.unsupported` and `.failed` pass through under every binding. They are reports
about the *Tower* — "no world root is configured", "the builder died" — and the
phone cannot establish whose builder died. Swallowing either into a spinner
would hide a real fault behind an animation.

### What this is not

**It is not a capture-id comparison, because the phone does not know its own
capture id.** The Tower mints it when `stream_start` arrives and does not report
it; the `stream_started` message that would carry it is deliberately
unimplemented Tower-side (`tower/docs/agent-handoffs/IOS-WORLD-BUILDER-INTEGRATION.md`
§5), and iOS does not ask for a field it has no consumer for.

What the phone *can* establish is **liveness**, and the three facts above are
jointly sufficient for it: the builder is attached at the moment the id is
minted, the result channel prefers a world whose writer lock is held by a
running process, and `world_build_session.py --follow-capture` is the only path
that records `frame_source: "live-capture"` with the capture directory's name.
A snapshot missing any of them describes a capture that is over, replayed, or
synthetic — none of which this phone opened.

That is strictly weaker than an id comparison and strictly stronger than what
shipped before, which was nothing. When `stream_started` lands, the equality
check drops into `WorldSessionGate.binding` and nothing else moves.

### What the wearer sees

`.awaiting` and `.foreign` replace *"Waiting for the Tower's first world
update…"* with **"Frames are reaching the Tower."** plus one of:

- `.awaiting` — *"Nothing is building a world from them yet."*
- `.foreign` — *"The world the Tower is reporting was built from a different
  capture, so nothing here describes this session yet."*

Both are claims the phone can support. That sentence is exactly what nobody
could see on 2026-08-24.

### Release builds

`isStreamingToTower` is permanently `false` in Release — the two functions that
set it are on the DEBUG-only frame path, and Release has no capture control on
any screen. The binding is therefore permanently `.none` there, and the Tower's
own state is the whole answer, which is correct: a build with no capture cannot
be looking at the wrong one.
