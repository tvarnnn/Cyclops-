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
| `lifecycle.photographic` | Added `2026-09-22`. **Optional to read, and the app is correct without it** — the states that matter already move `lifecycle.state`. Read it to say *why* a world is still Improving, or to distinguish a Saved world whose photographic build FAILED from one that has the room. §3a |
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


---

## 3a. `photographic` — whether the room the wearer walked actually exists

`lifecycle.photographic` (added `2026-09-22`, full table in
[`CARTRIDGE-RESULTS.md`](CARTRIDGE-RESULTS.md)) answers a different question
from `build_in_progress`. The boolean is present tense — *is a process
working this millisecond* — and it is **false in the gaps between the
photographic stages**, which is how a wearer could be told **Saved** twice: once
at the surface/appearance boundary, and once for good at the end.
`photographic.state` is settled: *does this world still owe a photographic
room*.

The rule the Tower now applies, so the phone does not have to:

- `running`, `owed`, `unobservable` → `lifecycle.state` is **`finalizing`**.
  The world is not finished being made. `model_state` is `finalizing`, and
  the existing Improving copy is correct for all three.
- `complete`, `failed`, `unattempted`, `never_recorded` → `lifecycle.state`
  is whatever it always was. These are settled; a world in one of them is
  not waiting for anything.

**So an app that ignores this block still behaves correctly** — it simply
hears `finalizing` where it used to hear a false `finalized`. What the block
adds is the ability to be specific:

- `owed` — "still to be finished; the Tower will pick it up" rather than a
  spinner implying work is happening right now (`build_in_progress` is
  `false`, honestly, in this state).
- `failed` — the one case worth new copy. The world **is** saved and opens at
  whatever rung it reached, but the photographic room will not arrive.
  Presenting it as plain "Saved" is the T3 defect; presenting it as
  "Improving" would strand the wearer waiting forever. Something like
  "Saved — the photographic version could not be built" is the honest
  middle, and `detail` carries the reason.
- `never_recorded` — a world from before the photographic stages existed.
  **Must keep rendering exactly as today**: 165 of the 166 worlds on the
  development Tower are these, they are finished, and nothing about them
  changed.

`stage` names `surface` or `appearance` when the word is about one.

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
`faces-detected-and-filled/yunet-2023mar@0.30+plausibility3` for sessions
captured with the current Tower; older sessions may record `+plausibility2`,
`+plausibility1` or no suffix (`faces-detected-and-filled/yunet-2023mar@0.30`).
The suffix names a gate that runs on each detection before it is filled. Under
`plausibility3`: a box under 2% of the frame is always filled; a box of 2–25%
must have facelike landmarks, and from 5% must also be found again at native
resolution; a box over 25% must be found again at native, 1/2 or 1/4
resolution (its landmarks carry no evidence at that size), **except** that one
touching or within 5% (of the frame's short side) of the frame edge is also
filled on facelike landmarks alone, because a close face cut by the edge is
found at no reduced scale; landmarks too broken to judge always fill.
`plausibility2` lacked the edge exception. `plausibility1` differed above 25%
everywhere, where facelike landmarks alone filled the box. An older session can
be explicitly re-redacted on the Tower under the current rule
(`WORLD-BUILDER-APPEARANCE.md` §6.5); its `session.json` label does not change,
and the appearance it builds carries the current label in `X-World-Redaction`. The gate exists because on real
captures 220 of 240 detections were not faces (hands, a cup, bare wall) and
blacked out 12.6% of every frame; it changes what is filled, not the
detector or its threshold. Nothing parses any of these values. Never "redacted", "anonymised"
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

## 10. The saved-world picture (`WorldRenderViewer.swift`)

Consumes `WORLD-BUILDER-WORLDS.md` §4 (the page) and §4a (its revision). Added
2026-09-16 after an adversarial review of the viewer. Nothing below has been
compiled or run on a device yet; `docs/agent-handoffs/` names the Mac checks.

**The capability declaration** (2026-09-17). Every rung-deciding request
carries `viewer=appearance-1` (`WorldAssetScheme.viewerCapability`): the page
URL (`WorldRenderClient.url`), the native revision poll
(`WorldRenderClient.revisionURL`) and the page's own revision poll as the scheme
handler proxies it. The Tower offers the appearance rung to `auto` only on that
declaration (WORLDS §4 `viewer`), so an app built before this transport keeps
getting the surface page it can draw.

**The page.** `WorldRenderClient.page(for:)` fetches the HTML with an
**ephemeral `URLSession` with no URL cache** (`WorldAssetClient.sharedUncachedSession`;
requests `.reloadIgnoringLocalAndRemoteCacheData`) and hands the string to a
`WKWebView`. Changed 2026-09-17: the web view no longer uses
`loadHTMLString(_:baseURL: nil)`. It loads `glasses-world://tower/worlds/<world>/render`,
and the app's scheme handler answers that URL with the string it fetched, so
the typed fetch errors, the 30 s bound, the render watchdog and the
`wb-representation` / `wb-revision` scan are all unchanged. The response headers
still never reach WebKit, so the Tower's CSP reaches the phone only as the
`<meta http-equiv>` tag every page carries (§4 rule 6).

**An appearance page that names no session is refetched once, then reported.**
With no pinned session the handler's session comes only from the page's own
`wb-revision` stamp, and the Tower composes the page unstamped when
`_appearance_revision` raced away between choosing the rung and stamping it
(§4a rule 7). Such a page can fetch nothing at all: every manifest, chunk and
proxy request is refused locally, **the Tower log shows nothing**, and the field
report reads "the world opened empty". `load()` refetches once — the race is
narrow and the refetch wins it — and then fails with a retryable sentence rather
than presenting a page that cannot work (2026-09-17, review 2 m-16).

**The transport** (`WorldAssetTransport.swift`), for the appearance page of
WORLDS §4, which fetches its imagery:

- **One scheme, one host.** `glasses-world://tower/…`, whose paths mirror the
  Tower's. The handler (`WorldAssetSchemeHandler`, a `WKURLSchemeHandler`)
  answers exactly: the page (`/worlds/<w>/render`, no query, from memory);
  `/worlds/<w>/appearance/<s>/manifest`; `…/chunk/<digest>` and
  `…/proxy/<digest>` with a 32 lower-hex digest; and
  `/worlds/<w>/render/revision?session_id=<s>` (that query and no other; the
  handler proxies it as `?session_id=<s>&viewer=appearance-1`, WORLDS §4a).
  `<w>` is the world the viewer was opened for; `<s>` is the session the page
  draws — the target's session, or the one the page's own `wb-revision` names.
  **Everything else is a 404 from the handler and never reaches the Tower**:
  another world or session, any other route, any method but GET, a query
  elsewhere, a user, port or fragment, an empty, `.` or `..` segment. The
  whitelist is the pure `WorldAssetRequest.parse`.
- **Proxying.** Whitelisted requests go to the same path on
  `TowerConfiguration.httpBaseURL` through `WorldAssetClient`: an ephemeral
  session, `urlCache = nil`, no cookies, `.reloadIgnoringLocalAndRemoteCacheData`.
  Status and MIME type pass through; the handler adds `Cache-Control: no-store`
  and `nosniff`. Bodies go to WebKit in 1 MB pieces. The Tower gzips appearance
  bodies (APPEARANCE §9); `URLSession` negotiates that itself (the client never
  sets `Accept-Encoding`) and returns decoded bytes, so WebKit is given the
  decoded body, its decoded `Content-Length`, and **no** `Content-Encoding`
  (`WorldAssetSchemeHandler.responseHeaders`).
- **Isolation.** The handler is `@MainActor` and the two `WKURLSchemeHandler`
  requirements are `nonisolated`, hopping with `MainActor.assumeIsolated`
  (2026-09-17, review 2 C-1). Nonisolated witnesses, because an isolated
  conformance to an `@objc` protocol is not expressible and
  `SWIFT_APPROACHABLE_CONCURRENCY` would try to infer one; `assumeIsolated`
  and not a `Task`, because a `Task` hop would make `stop` asynchronous and
  let a `start` for a task WebKit had already stopped be processed first.
- **Stop, cancel and teardown.** A task WebKit stops is removed from the live
  set and its fetch cancelled; a completion for a stopped task says nothing
  (answering one raises an Objective-C exception, which Swift cannot catch).
  `WorldRenderWebView.dismantleUIView` stops the load, clears the navigation
  delegate and calls `detach()`, which empties the live set and cancels every
  outstanding fetch — so a task whose web view is gone is answered by nobody,
  not even if WebKit never delivered its `stop` (2026-09-17, review 2 M-3).
  The page's fetches are bounded from its own side too (§4 "Following"), so a
  dropped task cannot leave it waiting for ever.
- **Memory only.** Chunks and the proxy (content-addressed, immutable) are kept
  in the handler's memory (`WorldAssetMemory`), capped at 64 MB, so a WebContent
  kill does not download 14 MB again. **A copy is answered only under a fresh
  authorisation** (2026-09-17, review 1 M5): the Tower answered this session's
  manifest with 200 within the last 20 s (`authorizationWindow`). The page fetches
  the manifest before any bundle — at boot, for a new build, on a restored
  context — so a page load is authorised by its own manifest request. A hit
  outside the window is revalidated: the handler fetches the manifest first and
  answers from memory only if that is 200; otherwise the copy is dropped and the
  bundle request goes to the Tower, which applies the label check. Concurrent
  revalidations share ONE manifest fetch (2026-09-17, review 2 m-3): a restored
  context re-asks for every bundle at once, and eight independent manifest
  fetches were ~2 MB of redundant transfer and eight redundant label checks in
  one burst. The copy and its authorisation are **dropped** whenever a manifest
  request answers anything but 200, or a revision request stops naming a served
  appearance, or the page on screen changes to a different SESSION, and
  **`tearDown()` drops them when the viewer closes** — called from the
  screen's `.onDisappear`, and the first thing that actually implemented the
  last of those: `dropCache()` had no caller at all and up to 64 MB of
  first-person room imagery was left to ARC (2026-09-17, review 2 M-4 and m-4).
  `tearDown()` does not detach the handler, because `.onDisappear` also fires
  for a screen that is merely covered; `dismantleUIView` is what says the web
  view is gone. Dropping the imagery early costs a refetch, and only if the
  page asks again.
  The copy belongs to the VIEWER, not to the web view: it is owned by
  `WorldRenderViewerModel`, survives `dismantleUIView`, and a "Try again" that
  builds a second web view does not re-download it. Before this a hit was
  answered with no Tower check at all, so a restore or reload after a relabel
  or purge redrew withdrawn imagery for up to a poll interval. Tested on the
  handler (`answer(_:sessionID:)`) against a stubbed Tower, not only on a JSON
  predicate. Nothing is written to disk.
- **The web view.** `WKWebsiteDataStore.nonPersistent()`, the scheme handler
  registered before the view exists (`WorldRenderWebView.makeConfiguration`),
  no data detectors, no inline media.
- **Navigation.** `WorldRenderNavigationPolicy.allows(_:isInitialLoad:pageURL:isReload:)`:
  the initial load of exactly the page URL, nothing else — not `about:blank`,
  not the page with a query, not a second load of it — **except the page
  putting itself back**: a main-frame navigation to exactly the page URL after
  the initial load (2026-09-17, review 1 m8), which the appearance page asks
  for when WebKit never restores a lost WebGL context. The reload is served
  the same string from memory and costs the camera, and the screen goes back
  to a bounded "Drawing the world…" while it happens rather than claiming to
  be ready over a black rectangle.
  Both `.reload` and `.other` are accepted for it, and it is **counted**: at
  most three inside the same 60 s window as the kill budget (2026-09-17,
  review 2 m-8 and m-9). WebKit does not promise which `WKNavigationType` a
  script-initiated reload arrives as, and cancelling it because it came as the
  other one leaves the wearer on "Restoring…" for good — which is the exact
  failure the reload exists to prevent. The count is what makes accepting
  `.other` safe.
- **CSP.** The appearance page's `<meta>` allows `connect-src glasses-world:`
  and nothing else; every other page keeps `default-src 'none'`.

**The caption follows the page.** The native caption above the web view reads
the page's `<meta name="wb-representation">` from its first 4096 characters:
*Surfaces the Tower reconstructed from the walk, only where the cameras measured
them. A gap is not proof that nothing is there. Not to scale.* (surface -- quoted
whole, because the sentence it replaced, "Gaps are places nothing looked", is
the claim `WORLD-BUILDER-SURFACE.md` §2 claim 2 retracts. It does not say "two
views": the app cannot see the manifest, and a surface built before the
per-face filter made no such test. The page's own caption says "at least two
camera views" only when the manifest shows the filter ran), *Points the Tower measured
densely…* (dense), *Points the Tower measured from the walk…* (sparse),
*The camera's own images, faces redacted, placed on the reconstructed room.
Grey haze is where no kept image looked; only cracks a few pixels wide are
filled, from the images beside them. Not to scale.* (appearance), and a
rung-neutral *What the Tower reconstructed from the walk. Not to scale.* before
the page arrives or for a page that declares nothing. Details says the page's
own Diagnostics button switches views **only** on a sparse (or undeclared) page;
on a surface or dense page it points at the world screen's Diagnostics instead.

The appearance sentence was rewritten on 2026-09-17 (review 2, M-5). It read
*"Dark gaps were not seen or were masked as unreliable; nothing is filled in"*,
written before `21d6f1a`, and both halves stopped being true in opposite
directions. **Nothing is filled in** is false: a pixel with no surface whose two
sides within a few device pixels lie on one plane is placed on that plane and
shaded (§4 "what it draws"). The PIXELS are still camera pixels — the artifact's
claims in `WORLD-BUILDER-APPEARANCE.md` §2 and §3 are untouched — but the
geometry under them is synthesised there, and a wearer told "nothing is filled
in" would read a closed crack as measured. **Dark gaps** is false in the other
direction: a void is painted as an unlit grey fog lifted from the mean of what
is inked around it, never as black. The replacement states the bound rather than
a denial, which is what a wearer can act on: haze means distrust this, a seam a
few pixels wide may have been closed from its neighbours. A caption that denies
what is on screen is the same failure as one that overclaims.

**There is always a way to ask again.** The screen's toolbar carries a plain
**Reload** (`world-render-reload`), which is `load()` — the same thing "Try
again" calls — and is reachable whatever the screen is showing (2026-09-17,
review 2 M-0 and m-7). Every other control here is conditional on the model's
state, and the state that needed one most had none: a page that placed no
imagery, or whose fetches were all refused, still reports `didFinish`, so the
screen is `.ready`, the failure view is not shown, and the wearer was left with
the page's own sentence and a Close.

**Following.** While a page is on screen (`.ready`), the screen's `.task` asks
`GET /worlds/{id}/render/revision` and compares the answer with the revision
stamped into the page (`wb-revision`) and with the last revision it acted on:

| The Tower says | The app does |
|---|---|
| same revision | nothing; no page is fetched |
| same revision, **new `appearance.revision`** (decoded as `WorldRenderRevision.appearance`) | nothing. The appearance page polls the same route through the scheme and overwrites its texture layers in place, keeping the camera. The follower never reloads the page for an appearance build |
| while an **appearance** page is on screen: `appearance.state` `rebuilding` (decoded as `appearanceState`; the ordinary Stop), then the appearance served again | nothing. The page kept its textures and loads the final build in place; the page revision (`…@<epoch>`) did not move |
| while an **appearance** page is on screen: the appearance **withdrawn** (`state` `withdrawn`/`absent`/`unavailable`, or no `state` from an older Tower), then served again on the appearance rung, of the same walk | **waits one poll for the page, then replaces it.** The page's own follower refetches the manifest and redraws IN PLACE, keeping the camera; the app can see that it did, because that manifest fetch goes through the scheme handler (`WorldAssetSchemeHandler.servedAppearanceToPageAt`). If it did, the app does nothing at all. If it did not — the script died, or the page never started one — the app fetches the page and swaps it in **by itself, even when the page revision is unchanged** (review 1, B1), which on a pre-epoch world is the only path there is. Exactly one of the two acts: before 2026-09-17 both did, and the slower one won — a camera reset to the opening pose and ~13 MB of chunks fetched again for a page that had already fixed itself (review 2, M-2). The flag is cleared **only after a refresh that actually put a page on screen**; a failed fetch leaves it armed and the next poll tries again, where before one unreachable moment disabled the recovery for the life of the screen (review 2, M-1). `WorldRenderViewerModel.appearanceFollow` is the whole rule, and is pure |
| a new revision of a **better rung** (sparse → dense → surface → appearance, read from `representation`, never from the opaque revision), of the **same walk** | fetches the page and swaps it in |
| a new revision of the **same rung** with `live: false`, of the **same walk** | fetches the page and swaps it in. This is any same-rung revision seen while the Tower reports nothing building, which is normally the finished build after Stop (a live build polled in the gap before the final starts also qualifies). A world is not rebuilt after its final build, and a wearer who stopped walking is shown the finished world |
| a new revision of the **same rung** otherwise (live, or the Tower did not say), or **any** new revision of a better or same rung from **another walk** | shows *"A newer reconstruction is ready. Show it"*; the swap happens only on tap, because a swap reloads the page and resets the reader's camera. The button goes away if the Tower goes back to reporting the revision on screen |
| a new revision of a **worse rung** | nothing: not swapped, and not offered as "newer" |
| a revision whose page could not be drawn on this phone | never swaps it in or offers it again, except once: a revision refused while the Tower said its build was **live** is tried again when the Tower reports **the same revision** finished (`live: false`). The appearance page revision (`…@<epoch>`) survives the ordinary Stop, so without this one walk-time appearance page that could not be drawn refused the finished world's page too, and once the Stop gap's surface drew, nothing on screen said so (2026-09-22 Mac validation). A revision refused when it was already finished is not retried |
| any revision of a rung that **two refreshes up to it** failed to draw on this screen | not fetched, swapped or offered, except that a **finished** build (`live: false`) of that rung is tried **once** per screen. A failure of the rung already on screen does not count (that rung drew here, so its rebuild failing is memory pressure), and a refresh of the rung that draws forgets its failures |

"The same walk" means the screen was opened on a named session, or the
revision's session (the part before `/`, §4a) is the session of the page on
screen. With no session named the Tower answers its newest session with
geometry, which can be another walk; nothing from it replaces the world the
reader opened without a tap.

**What was fetched decides, not what was polled.** A refresh (automatic or
tapped) that fetches a page of a **worse rung** than the one on screen does not
swap it in: a tapped offer whose build has since become undrawable on the
Tower, or a fetch racing a manifest replace, is served points. A fetched page
that differs from the one on screen **only in its `wb-revision` stamp** is the
same picture and is not swapped in either: a page composed while the Tower
could not read the manifest carries no stamp (WORLDS §4a rule 7), and the same
page a moment later does.

**After a refresh could not be drawn** the previous page is back and the screen
is ready, so the caption shows *"A newer reconstruction could not be drawn on
this phone. Try again"*. It calls `load()`, which fetches the page again and
forgets every refused revision and rung. A later refresh that draws removes it.
| `404` with FastAPI's `{"detail": "Not Found"}` | stops following for this screen (a Tower older than the route) |
| any other `404`, another status, or a transport error | keeps the picture and asks again next interval |

The interval is 10 s while the payload says `live: true`, and doubles to a
120 s ceiling while it says `false` (or says nothing), back to 10 s on any
change. It never stops on `live: false`, because the Tower starts the final
surface a few seconds after it releases the world lock (§4a rule 6).
A diagnostics target (`view=diagnostics`) is not followed at all: its page is
always the sparse one.

**A refresh does not take the world away while it is being drawn.** A failed
fetch leaves the page. A fetched page that fails to draw before it reports
finished — the 20 s render watchdog, `didFail`, or the content process being
killed past its budget — puts the previous page back (with a fresh kill budget,
since it already drew once) and refuses that revision, and the rung after its
second such failure going up to it. A refresh reaches the failure view only if the restored
page then fails to draw as well. **Not covered:** a page that reports finished
(`didFinish`) and is killed afterwards, for example on the first frames of a
large mesh; the fallback is released at `didFinish`, so that kill goes to the
failure view (review 2, iOS m2).

**Content-process kills** are counted within a sliding 60 s window, not for
the life of the screen: up to two reloads, and the third kill inside a minute
reports "too large to draw on this phone". Kills spread over a long walk, such
as iOS reclaiming a backgrounded app's WebContent process, do not add up.

**Known and accepted.** When a surface or dense artifact passes its header
check but its page cannot be built, the revision route reports the better rung
while the page is stamped lower; the app pays one page download per rebuild
for that, not one per poll, and the Tower logs it at `ERROR`. A swap briefly
holds the old and new page strings plus the old document's heap; for the
default phone surface page (sized to the 6 MiB page budget,
`WORLD-BUILDER-SURFACE.md` §8) that is fine, and for a Tower configured to serve
larger pages it is the likeliest moment for a WebContent kill, which the
fallback above then absorbs.
