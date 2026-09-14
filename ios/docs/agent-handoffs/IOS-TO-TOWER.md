# iOS → Tower — what the product shell needs

**Audience:** Tower Claude, and whoever owns the Tower roadmap.

**What this document is.** A statement of what the iOS product shell needs in
order to display each cartridge truthfully. Every item below is an **iOS
requirement**, not a description of something that exists.

**What this document is not.** It is not a protocol, not a schema, and not a
proposal for message names. Where it names a *quantity* ("keyframes accepted"),
that is a question — "do you keep this, and in what unit?" — not an instruction.
Where a decision is genuinely the Tower's (geometry representation, coordinate
convention, experiment vocabulary, relation predicates), iOS has deliberately
built an opaque hole rather than a guess, and says so.

**Nothing in the iOS app decodes anything that does not exist.** The four
cartridge clients all report "no contract", and a test asserts that
(`ProductShellTests.testTheTowerDeclaresNoCartridgeContracts`).

---

## Legend

Every item is labelled:

| Label | Meaning |
|---|---|
| **AVAILABLE** | The Tower demonstrably provides this today, and iOS consumes it. |
| **EXPECTED FROM EXISTING CONTRACT** | Derivable from what the Tower already sends, without a new contract. |
| **MISSING — TOWER NEEDED** | iOS needs it; the Tower does not provide it; nothing in iOS fakes it. |
| **UNKNOWN** | Neither side has decided. iOS has left a hole shaped so any answer fits. |

---

## 0. What actually exists today, so the rest is readable

**AVAILABLE.** The entire Tower vocabulary, as iOS implements it:

| Direction | Message | Payload |
|---|---|---|
| iOS → Tower | `ping` | — |
| Tower → iOS | `pong` | — |
| iOS → Tower | `stream_start` | (exact payload pinned by `TowerClientTests.testStreamStartSendsExactPayloadOnce`) |
| iOS → Tower | `frame` | encoded frame, `seq`, `width`, `height` |
| Tower → iOS | `frame_result` | `seq`, `mean_intensity`, `processing_ms` |
| iOS → Tower | `stream_stop` | — |

That is all of it. There is no module container, no module, no selection
message, no capability declaration, and no per-cartridge channel.

`mean_intensity` is currently the **only** thing the Tower says about a frame's
content, and it is surfaced on the Home workspace precisely so the app can show
something real rather than needing a "coming soon" panel — in **Debug builds**.
The whole capture-and-forward path is `#if DEBUG`, so a Release build never
sends a frame and therefore never receives a `frame_result` to display.

### Cross-cutting, before the per-cartridge sections

These four apply to every cartridge, and getting them wrong once costs more than
any single field below.

**0.1 — A capability declaration. MISSING — TOWER NEEDED.**
iOS needs to know, per cartridge, whether the Tower can serve it and which
revision of the agreement it is offering. iOS models this as an **opaque
identifier compared for equality** (`CartridgeContract`) — it deliberately does
*not* assume integer versions, ordering, or backward compatibility, because
those are the Tower's to decide. The three states iOS distinguishes and must be
able to keep distinguishing:

- the Tower says nothing about this cartridge → "not built yet";
- the Tower offers a contract this build does not implement → "update the app";
- the Tower offers a contract this build implements, but is unreachable → "connect".

Those call for opposite user responses, which is why they cannot be one state.
Note this is **not** a request for dynamic module discovery —
`04-MODULE-SYSTEM.md` forbids that before V1.0, and iOS caches a declaration
rather than fetching a registry.

**0.2 — Provenance on every derived value. MISSING — TOWER NEEDED.**
Rule 16 and Core Principle 2/4: for any value a model produced, iOS needs to know
that it was inferred and, where the Tower has one, the confidence. A value that
arrives with no provenance is displayed with an explicit "the Tower did not say
whether this was measured or estimated" caveat, which is worse for everyone than
the Tower saying.

**0.3 — Observation time, distinct from arrival time. MISSING — TOWER NEEDED.**
Core Principle 5. iOS holds `observedAt` and `receivedAt` separately and will
**never** substitute one for the other; a report with no observation time renders
as "time unknown". Note the open question in `07-PLATFORM-CONSTRAINTS.md` about
whether DAT's `CMSampleBuffer` presentation timestamp is capture time or
phone-arrival time — until that is resolved empirically, the Tower's own
observation clock is the only usable one.

**0.4 — Redaction state on every image. MISSING — TOWER NEEDED.**
See §5. iOS will not display persisted imagery whose treatment was not stated.

**0.5 — A unit string beside every figure. MISSING — TOWER NEEDED.**
`WorldScaleSemantics.inferredMetric` says a figure is *metric in kind*. It does
not say what unit it counts in, and "metric" is not "metres". iOS renders a
figure with the unit the Tower names and **bare** when it names none — a bare
number being the honest rendering of an unlabelled quantity. This is the
position `CVMetric.unit` already took ("never assumed"); it now applies to every
spatial figure in the app.

---

## 1. World Builder

iOS surface: `Glasses/Workspaces/WorldBuilder/`.
Types: `WorldModelState`, `WorldSnapshot`, `WorldGeometryReport`,
`WorldTrajectoryReport`, `WorldCalibrationState`, `WorldPersistenceState`.

### 1.1 Lifecycle

| Need | Status | Detail |
|---|---|---|
| Is a world being built at all? | **MISSING — TOWER NEEDED** | A start/stop/failed signal **distinct from "frames are arriving"**. iOS currently cannot tell "the Tower received a frame" from "the Tower did something spatial with it", and must not conflate them. |
| Has capture ended but work continues? | **MISSING — TOWER NEEDED** | iOS models this as `.finalizing`, separate from both `.receiving` and `.finalized`. It matters because during finalisation the figures on screen are not yet the figures that will be stored, and the camera may already be off — so "live" would be a lie about the sensor even while the Tower is genuinely working. |
| Failure | **MISSING — TOWER NEEDED** | A reason string a person can read. `04-MODULE-SYSTEM.md` already requires a FAILED state that does not "return stale or fabricated results". |

### 1.2 Identity

| Need | Status | Detail |
|---|---|---|
| Stable world id | **UNKNOWN** | Does a world have an identity that survives a session, or is it per-session? iOS holds `worldID: String?` and treats `nil` as "not named", which is honest either way. |
| Human-readable name | **UNKNOWN** | If the Tower does not name worlds, iOS shows no name rather than deriving one. |
| Change detection | **MISSING — TOWER NEEDED** | A monotonic revision or counter, so the UI can tell **new data from repeated data** without diffing geometry. Without it, every update looks like a change and any "updated Xs ago" is guesswork. |

### 1.3 Geometry — the decision iOS has deliberately not made

**UNKNOWN, and iOS wants it to stay the Tower's decision.**

iOS does **not** assume point cloud, mesh, sparse landmarks, pose graph,
trajectory-only, or anything else. `WorldGeometryReport` carries:

- `representation: String?` — **the Tower's own name for what it built**, stored,
  displayed verbatim, never parsed and never matched against a known set;
- `elementCount: Int?` — a count in whatever unit that representation implies.
  Always displayed next to the label, never alone, because iOS does not know what
  one element is;
- `isIncremental: Bool?` — **MISSING — TOWER NEEDED.** Whether updates are deltas
  or whole snapshots. A UI that assumes incremental updates will draw a partial
  world as a complete one.

This lets the workspace say something true today — "the Tower reports a
representation of this kind, and this build cannot draw it" — without any
rendering path that quietly expects the wrong shape. **When the Tower chooses a
representation, that is the point at which iOS adds a renderer**, and the choice
should be made on the Tower's terms, not to fit a guess made here.

No 3D framework has been introduced on iOS (no SceneKit, RealityKit, or Metal).

### 1.4 Trajectory

| Need | Status | Detail |
|---|---|---|
| Pose count | **MISSING — TOWER NEEDED** | How many camera poses the Tower retained. |
| Path length | **MISSING — TOWER NEEDED** | Must arrive **with** its scale semantics (§1.5). iOS refuses to render a `.relative` path length as metres. |
| Path-length unit | **MISSING — TOWER NEEDED** | See §0.5. Without it the figure renders bare. |
| Pose array | **NOT REQUESTED** | iOS deliberately holds **no poses**. A pose schema needs position, rotation convention, handedness, coordinate frame and units — five Tower decisions, each of which renders plausibly and wrongly if guessed. Send a summary until the representation is settled. |

### 1.5 Calibration and scale — a hard requirement, not a preference

**MISSING — TOWER NEEDED, and mandatory with any spatial figure.**

`docs/modules/WORLD-BUILD.md`:

> World Build must never represent monocularly inferred depth as ground-truth
> physical distance. Any distance figure derived from monocular inference must be
> identifiable as an estimate wherever it is stored, **displayed**, or consumed by
> another module.

"Displayed" is the iOS layer. So **every** spatial figure must arrive labelled as
one of: `relative` / `inferredMetric` / `measuredMetric`. iOS encodes this in the
type (`WorldScaleSemantics`), so a figure cannot reach a view without having said
where it came from — and a figure that arrives unlabelled is simply not shown as
a distance.

Separately, **calibration state** (`unknown` / `uncalibrated` / `calibrating` /
`calibrated`) — coarse, because the calibration *procedure* is the Tower's to
design and iOS must not presume which. Deliberately **no** calibration
percentage: "62% calibrated" implies a denominator nobody has defined.

iOS shows a metric distance only when calibration is established **and** the
scale claims metric provenance. On current monocular hardware the honest outcome
is usually that it does not.

### 1.6 Tracking

**MISSING — TOWER NEEDED.** Coarse — good / limited / lost — is enough. iOS
deliberately does **not** want a percentage: that would imply a calibrated
confidence model neither side has defined.

### 1.7 Persistence and inspection

| Need | Status | Detail |
|---|---|---|
| Did the world survive the session? | **MISSING — TOWER NEEDED** | `session` / `saved(revision)` / `reloading`, plus "did not say" as distinct from "session only" — silence is not a promise that a world was discarded. |
| Reload a saved world | **MISSING — TOWER NEEDED** | iOS has an inspection mode (`WorldInspectionMode.inspecting(worldID:)`) that changes what every control on screen means: there is no capture to start, and a counter that moved would be a bug. |
| Where is it stored? | **NOT REQUESTED** | The Tower owns persistence entirely (`04-MODULE-SYSTEM.md`). iOS stores no world data and must not imply it could. |

### 1.8 Progress

**MISSING — TOWER NEEDED.** Whatever the Tower actually counts — keyframes
accepted, observations fused, mapping seconds on its own clock. **A count the
Tower does not keep should not be invented for the UI's benefit.** iOS renders an
absent count as absent, not as "—" and not as zero; `nil` and `0` are different
claims and are kept different all the way to the screen.

Note `mappingSeconds` is deliberately **not** derived from an iOS timer: the
iPhone's idea of elapsed time is not the Tower's idea of mapping time.

### 1.x The render route is now the product surface, and iOS asks it two questions

**AVAILABLE, with one addition MISSING — TOWER NEEDED.**

As of 2026-09-09 the iOS side treats `GET /worlds/{id}/render` as **the primary
representation of a saved world**. Opening a world from Saved Worlds pushes that
page straight onto the screen; the sparse per-segment gallery this app draws from
`geometry/manifest` moved behind a Diagnostics disclosure. Nothing was deleted —
the manifest, the chunks, the placement words and the refusal prose are all still
fetched and all still shown — but they are no longer what a person sees first.

Two consequences for the Tower:

1. **The render page carries the product claim now.** Its composition, its
   colouring and its caption are what a person reads as "my world". A page that
   colours by segment index reads as a debugger however good the geometry is.
2. **iOS asks for a diagnostics variant of the same page: `?view=diagnostics`.**
   `WorldRenderView.diagnostics` appends that query parameter and nothing else;
   the product view sends no `view` parameter at all, so its URL is byte-for-byte
   the one every previous build asked for.

   **AVAILABLE as of 2026-09-09, and server-side, which is the only form that
   works.** `world_render` declares `view`, `build_world_render` threads it, and
   `render.py` substitutes the opening mode into the page's script.

   That mattered because **a page that switches modes from `location.search`
   does not reach the phone.** iOS loads the render page with
   `loadHTMLString(_:baseURL: nil)`: no origin, exactly one navigation
   (`about:blank`), every other navigation refused by
   `WorldRenderNavigationPolicy`. That is deliberate — it is what lets the app
   show the Tower's own 404 prose instead of a blank web view, and what makes
   refusing navigation a one-line policy instead of an allow-list — and it will
   not be loosened. Inside it `location.search` is empty, so an in-page
   `?view=diagnostics` link would have gone nowhere. Selecting at compose time
   is what made it reachable.

   **iOS asks for it at open time only, never as a toggle.** "Open the solver's
   3D view" in the world screen's Diagnostics section fetches the page already
   in diagnostics mode. There is deliberately no in-app switch between the two
   renderings: switching would refetch several megabytes to change one `let
   mode` in a script already on screen, and the Tower's page carries its own
   `<button id="mode">` and `d` key that switch in place for nothing. Keep that
   in-page control — on the phone it is the switch.

   A Tower too old to declare `view` still answers with the ordinary world page
   and no error, because FastAPI ignores query parameters a route does not
   declare. That is the right failure: the wearer asked to see something and
   sees their world.

**Also still MISSING: any route that would let a person act on a world that did
not finish.** Everything under `/worlds` is a read. `WorldRecoverability` therefore
says what survived and states plainly that finishing the world is the Tower's to
do — iOS deliberately offers no Retry, Rebuild or Re-solve button, because there
is nothing behind one.

**And now read: `lifecycle.finalization.final_solve`.** It has been on the wire
since `world_builder.status/2026-09-06` and was decoded by iOS and displayed by
nothing. `pending`, `solved`, `skipped`, `failed` and `unavailable` now each
produce a different sentence and, in two cases, a different stage word (`Saved`
versus `Partial`). A sixth word will survive to the screen verbatim. **`null`
produces nothing** — it conflates "no report has arrived yet", "this Tower
predates the field" and "the record kept no account", and the phone will not
report on the Tower from silence.

**One request about how you send it.** iOS republishes `lifecycle.finalization`
as an input in its own right, because the client dedupes `model_state` and a
report that moves `final_solve` while the snapshot stands still would otherwise
change nothing on screen for the whole length of a final solve. So: **keep
sending the finalization block on the heartbeat**, even when nothing else in the
payload has moved. iOS drops the repeats itself.

**`model_state` is the single source of the lifecycle word.** iOS decodes
`end_reason` (`WorldSessionReport`, `WorldListingSession`) and derives **nothing**
from it — no "interrupted", no failure, no badge. So the classifier reporting a
repaired world as `ready` with an `error` `end_reason` arrives on the phone as
`Saved`, and nothing on this side contradicts it. (An earlier version of this
sentence said the Tower's reason string is shown beside it; it is not —
`modelState(from:)` drops `reason` for `finalized`, and the `.finalized` arm of
the canvas draws none. Corrected at the 2026-09-14 Mac gate.)

---

## 2. Experimental CV Lab

iOS surface: `Glasses/Workspaces/ExperimentalCV/`.
Types: `ExperimentalCVState`, `CVExperiment`, `CVMetric`, `CVAnnotationReport`,
`CVTimings`, `CVExperimentRun`.

This is Module #1 on the roadmap (V0.9) and therefore the cartridge most likely
to land first. It is also the one where iOS has been most careful to decide
nothing.

### 2.1 The experiment registry

**MISSING — TOWER NEEDED.** The Tower declares which experiments exist; iOS holds
**no list**. `docs/modules/EXPERIMENTAL-CV.md` lists nineteen candidates and calls
the list "intentionally broad", so any subset hardcoded on the phone would be the
app asserting that those specific experiments exist.

Per experiment, iOS needs: an opaque `id`, a `name`, and optionally a `summary`.
Nothing else, and nothing is parsed.

### 2.2 Lifecycle

**MISSING — TOWER NEEDED.** iOS models: `idle(available:)` → `starting` →
`running` → `completed`, plus `failed`. Two specific asks:

- **a request may be refused, and the refusal must be legible.** iOS never lets a
  request silently no-op; `04-MODULE-SYSTEM.md` already requires an unsupported
  request to "produce a clear degraded/failed state rather than silently
  pretending" it applied.
- **partial results while running** are welcome; iOS distinguishes "running, these
  figures may still change" from "completed".

### 2.3 Results

| Need | Status | Detail |
|---|---|---|
| Metrics | **MISSING — TOWER NEEDED** | Each: `label` (Tower's word, displayed verbatim), `value`, optional `unit` (Tower's word), **and provenance**. |
| Measured vs inferred | **MISSING — TOWER NEEDED, and required** | EXPERIMENTAL-CV.md: experiment output "is model inference, not a measured sensor fact, unless the experiment specifically validates against a ground-truth reference. Results/logs must distinguish the two." iOS makes provenance a **required** field, not an optional one, so the party that decodes the reply has to answer it. |
| Baseline | **MISSING — TOWER NEEDED** | The same document's success criteria require one, and end with "avoid declaring an approach 'better' without a measurement". iOS renders a better/worse verdict **only** when both a baseline and a direction (`higherIsBetter`) arrive. With either missing there is no code path to a verdict. |
| Frames processed | **MISSING — TOWER NEEDED** | A count the Tower keeps. |

### 2.4 Timings

| Need | Status | Detail |
|---|---|---|
| Tower processing time | **EXPECTED FROM EXISTING CONTRACT** | `frame_result.processing_ms` already exists and is exactly this shape: the Tower measuring itself. |
| End-to-end latency | **NOT REQUESTED** | iOS deliberately has **no** end-to-end field. It would be computed across two clocks whose relationship is an open question in `07-PLATFORM-CONSTRAINTS.md`. A number derived from two unrelated clocks is not a latency. |

### 2.5 Annotations

| Need | Status | Detail |
|---|---|---|
| Annotation count | **MISSING — TOWER NEEDED** | `0` is a real result ("found nothing") and must not merge with "did not say". |
| Rendered annotated frame | **MISSING — TOWER NEEDED** | As an artifact with a redaction state (§5). **An experiment gets no privacy exemption for being a debug surface.** |
| Annotation geometry | **NOT REQUESTED** | Boxes, masks, keypoints and flow fields each need a schema and a coordinate convention. A wrong convention renders confidently in the wrong place. Send a rendered frame or send a schema, not a guess. |

### 2.6 Cancellation

**MISSING — TOWER NEEDED, and iOS has no shape for it yet.** `run(_:)` returns
nothing and yields no handle, so there is currently no way to stop a bounded
Tower job once started. Rule 15 requires lifecycle operations to be bounded with
a defined failure transition; a long experiment with no cancel is the case that
rule exists for. When the contract lands, this is one of the few places the iOS
seam will need a genuine addition rather than a mapping.

### 2.7 Dataset recording

**MISSING — TOWER NEEDED, and iOS has built nothing for it yet.**
`06-PRIVACY-DATA.md` allows explicit dataset-recording sessions but requires them
to be manually started/stopped with recording state clearly indicated. iOS has no
surface for this and will not add one before the Tower can honour it — a
recording indicator the Tower cannot drive would be the worst kind of privacy
control.

---

## 3. Document Memory

iOS surface: `Glasses/Workspaces/DocumentMemory/`.
Types: `DocumentMemoryState`, `RememberedDocument`, `DocumentQuery`,
`DocumentQueryResult`, `DocumentQueryEvidence`, `DocumentTextAvailability`.

> **Scope note, read this first.** "Document Memory" is **not an adopted Tower
> module.** It is an iOS-side concept seed written during this pass
> (`docs/modules/DOCUMENT-MEMORY.md`) narrowing `ENVIRONMENTAL-MEMORY.md`'s own
> stated first version — "searchable OCR history" — down to documents
> specifically, with the reading path from `VISUAL-QA.md` behind it. Its catalog
> status is `.future`. **The Tower has not agreed to build it.** If the roadmap
> prefers to implement Environmental Memory whole, this iOS surface should be
> folded into it rather than kept alongside.

### 3.1 Document list

| Need | Status | Detail |
|---|---|---|
| Recent documents | **MISSING — TOWER NEEDED** | A list with a limit. |
| Opaque document id | **MISSING — TOWER NEEDED** | |
| Title | **MISSING — TOWER NEEDED** | Optional. iOS renders an absent title as "Untitled document" — a description of the *record*, never an invented name for the thing. |
| Summary | **MISSING — TOWER NEEDED** | Optional, and **it is model output**, so it arrives with provenance and is displayed with its caveat. |
| Confidence | **MISSING — TOWER NEEDED** | Core Principle 4: it must survive to display, not be dropped at the last hop. |

### 3.2 Text

**MISSING — TOWER NEEDED**, and note what iOS asks for:
`unknown` / `notReadable` / `extracted(characterCount:)`.

- **iOS runs no OCR and holds no document text in a list.** Rule 5 keeps heavy CV
  on the Tower; `06-PRIVACY-DATA.md` makes document contents among the most
  sensitive data the platform handles.
- The list carries a **character count, not the text**, so a list of documents is
  not also a bulk transfer of every document's contents onto the phone. Full text
  is fetched when a person opens one.
- `notReadable` is a **real answer**, not a failure. `VISUAL-QA.md` requires
  "insufficient visual evidence" to be first-class.

### 3.3 Time and duration

| Need | Status | Detail |
|---|---|---|
| Observation time | **MISSING — TOWER NEEDED** | When the glasses observed it. iOS will **never** substitute arrival time; an absent value renders as "time unknown". |
| Time in view | **MISSING — TOWER NEEDED** | iOS calls this `ObservedDuration` and labels it "In view 45s", never "viewed for" or "read for". Limitation 8: appearing in the camera does not establish that the wearer looked at it, noticed it, or read it, and the mitigation is classified REQUIRES FUTURE HARDWARE/API. **Please do not send a field named `viewing_duration`** — the name is the failure. |

### 3.4 Retrieval

**MISSING — TOWER NEEDED.** iOS models four query kinds and needs the Tower to
say which it can serve:

- `recent(limit:)`
- `text(String)` — literal matching
- `observedWithin(DateInterval)` — **a range, not an instant.** "This morning"
  and "around lunch" are approximate, and answering them exactly answers a
  different question.
- `semantic(String)` — resolved **entirely on the Tower**. iOS computes no
  embedding and runs no model.

Typed input is routed as `.semantic` because a person asking "the parking notice
from this morning" is *describing* a document, not quoting it.

**Input source is carried separately from the query** (`DocumentQueryOrigin`:
`appText` / `externalIntent`). No voice input is implemented, required, or
assumed — the separation exists so a future Siri intent, shortcut, or wake-word
layer submits through the same path without this cartridge growing a dependency
on speech. **The Tower does not need to know or care which.**

### 3.5 Result semantics — the part most likely to be got wrong

**MISSING — TOWER NEEDED, and these are three answers, not two.**

| Answer | Meaning | How iOS renders it |
|---|---|---|
| `matched(confidence:)` | Documents found. | The list, plus the confidence. |
| `notFound` | The memory was searched and nothing matched. | "Nothing matched." |
| `noObservation` | The memory holds nothing covering what was asked. | "Never observed" — plus an explicit statement that this is *not* the same as the document not existing. |

Core Principle 3 and `ENVIRONMENTAL-MEMORY.md`'s failure behaviour: absence of
observation is not observation of absence, and the module must "never create a
memory event retroactively to satisfy a query". Collapsing `noObservation` into
"no results" lets a gap in what the glasses happened to see read as a statement
about the world.

iOS additionally refuses to construct a result that claims `matched` while
carrying no documents — it coerces to `notFound` — so a decoder cannot produce
that combination even by accident.

### 3.6 Pagination

**UNKNOWN, and iOS currently has no room for it.** `DocumentQueryResult` carries
no cursor, and `DocumentMemoryState.results` holds exactly one result, so a
second page has nowhere to accumulate. `DocumentQuery.recent(limit:)` is the
only bound that exists. If the Tower expects to page a long history, say so —
this is the second place the iOS seam needs an addition rather than a mapping.

### 3.7 Thumbnails

**MISSING — TOWER NEEDED**, as an artifact with redaction state (§5). A
photographed page routinely contains a bystander, a screen, or a second document.

### 3.8 Source context

**UNKNOWN.** `sessionID` and, optionally, `worldID`. The world link is optional
and nothing depends on it: `ENVIRONMENTAL-MEMORY.md` requires any dependency on
World Build's spatial service to be "an explicit architecture evolution, not an
assumed dependency", so **Document Memory does not depend on World Builder** and
degrades in no way without it.

---

## 4. Scene Understanding

iOS surface: `Glasses/Workspaces/SceneUnderstanding/`.
Types: `SceneUnderstandingState`, `SceneSnapshot`, `SceneEntity`, `SceneTrackID`,
`SceneEntityKind`, `SceneFacing`, `ScenePosition`, `SceneRelationship`.

> **Scope note.** Also **not an adopted Tower module.** An iOS-side concept seed
> (`docs/modules/SCENE-UNDERSTANDING.md`) surfacing the *live* half of
> `OBJECT-MEMORY.md`'s detector/tracker pipeline without its persistence layer.
> Status `.future`. If Object Memory is implemented, this should become a live
> view onto its tracker rather than a second pipeline.

### 4.1 Tracks

| Need | Status | Detail |
|---|---|---|
| Anonymous track handle | **MISSING — TOWER NEEDED** | **Session-scoped and opaque.** It distinguishes the person on the left from the person on the right within one tracking session and is meaningless afterwards. |
| Person vs object | **MISSING — TOWER NEEDED** | iOS's `.person` case carries **no payload at all** — no label, no attribute, no descriptor — so there is nowhere for identity to be added without changing the type. |
| Object class label | **MISSING — TOWER NEEDED** | A **category** ("chair"), never an identity ("your chair"). Limitation 6's distinction, preserved: "black backpack detected" vs "likely the same black backpack previously observed". |
| Confidence | **MISSING — TOWER NEEDED** | Required on every track. |

**Do not send a durable person identifier.** A handle that survived sessions
would be a re-identification key by function, whatever it is made of, and
`ENVIRONMENTAL-MEMORY.md` requires avoiding biometric identity features absent an
explicitly justified use case. iOS will not persist one and does not display the
Tower's handle at all — rows are labelled positionally ("Person 1") so nothing on
screen can be mistaken for an identity.

### 4.2 Orientation — the wording is part of the contract

**MISSING — TOWER NEEDED**, as body/head orientation relative to the camera:
`unknown` / `towardCamera` / `awayFromCamera` / `acrossView`.

**This is not gaze, and iOS will not present it as gaze.** `.towardCamera` reads
**"Facing your direction"**. It does not read "Looking at you", "Watching you",
or "Making eye contact", at any confidence, in any phrasing — a test asserts
those words cannot appear. Limitation 8 is classified REQUIRES FUTURE
HARDWARE/API: there is no eye tracking on the target glasses, so there is no gaze
to report.

**Please do not send a field named `gaze`, `looking_at`, or `attention`.** If the
Tower ever computes something in that space, it needs to arrive named for what it
actually measures.

### 4.3 Positions

| Need | Status | Detail |
|---|---|---|
| Frame of reference | **MISSING — TOWER NEEDED** | Camera-relative or world-relative (with the world id). These are different claims: a camera-relative bearing changes when the wearer turns their head. |
| Bearing | **MISSING — TOWER NEEDED, and the sign convention must be stated** | iOS holds degrees from straight ahead, **positive to the right**, and the decode site is required to convert into that. This is the one convention iOS declares rather than leaves open, because a bearing has to be signed *somehow* to be usable and a silent presumption is the dangerous version — a Tower signing the other way would put every person on the wrong side of the wearer, rendering confidently and wrongly. **Please state yours.** Not gated by scale: a bearing is an angle and needs no depth. Rendered coarsely ("to your left"), because a bounding-box centre does not support "37.4° right". |
| Distance unit | **MISSING — TOWER NEEDED** | See §0.5. |
| Distance | **MISSING — TOWER NEEDED** | Must arrive with `WorldScaleSemantics`, exactly as World Builder's figures do. iOS will not render a `.relative` or unlabelled distance as metres. |

### 4.4 Counts

**EXPECTED FROM EXISTING CONTRACT** once tracks exist — iOS derives counts from
the track list rather than taking them as separate numbers, so a header can never
disagree with the rows beneath it.

What the **Tower** must not do is present a count as a statement about the room.
Zero means zero currently tracked **within the camera's field of view**. iOS
attaches that qualification to every count; the Tower should not compute or store
one that means anything else.

### 4.5 What iOS will not say about a bearing

**No "behind you", and no "beside you".** The glasses observe a forward cone, so
an entity at 150° cannot be a camera observation, and a phrase like "Behind you,
right" would tell the wearer the system had detected someone behind them.
`docs/modules/SCENE-UNDERSTANDING.md` says it plainly: *"Field of view is
narrower than human awareness, and a wearer looking at a desk has most of the
room behind the camera."*

iOS caps the vocabulary at `Ahead` / `To your left` / `To your right` / `At the
edge of view`. A bearing beyond the plausible field of view is reported as being
at its edge rather than given a direction the sensor cannot justify. If the
Tower ever reports a wide-angle or stitched field, that is a change worth
discussing rather than assuming.

### 4.6 Relationships

**UNKNOWN, and iOS wants the vocabulary to stay the Tower's.**

`SceneRelationship` carries `subject`, an **opaque predicate string** ("next to",
"holding", "seated at"), `object`, and a **required** confidence. iOS displays
the predicate verbatim and never matches on it — a fixed enum here would make the
phone the place the Tower's semantics are decided, and any term missing from it
would have to be dropped or mangled.

Confidence is required because a relation is an inference about two inferences
and is the least certain thing on the screen.

### 4.7 Staleness

**MISSING — TOWER NEEDED.** iOS distinguishes `observing` (current) from
`lastKnown` (the last scene reported), and announces the age of a stale scene
before its contents. Limitation 7: a stale observation must never be presented as
current state. A lost track should **disappear** rather than be extrapolated to a
guessed position.

### 4.8 Update rate

**Please coalesce before publishing.** This is the cartridge whose real client
will emit fastest. A scene replaced at frame rate and republished straight into
a `@Published` property with a `ForEach` under it would put a list diff on the
main actor at frame rate — and the main actor is where the sender releases its
send-window slots, so a scene view would be paid for out of the sender's
throughput budget. Whatever rate the Tower can produce, iOS will need to throttle
on receipt; a Tower-side rate that is already sensible saves both ends the work.

### 4.9 Persistence

**NOT REQUESTED.** This cartridge stores nothing, on either side, and its
data-behavior declaration should say so: persists nothing, retains nothing,
needs no purge because there is nothing to purge. If history is wanted, that is
Object Memory and a separate privacy review.

---

## 5. Imagery and redaction — applies to every cartridge

**MISSING — TOWER NEEDED, and this is the item iOS is least able to work around.**

The pipeline the platform requires:

```text
raw sensor data → ephemeral perception → derived structured state
                → redaction → persistence / display
```

**iOS applies no redaction and must not pretend to.** Doing it on the phone would
mean the raw pixels had already arrived, at which point the control is theatre.
`06-PRIVACY-DATA.md` requires redaction where the data is derived — the Tower.

So every image the Tower offers must arrive stating how it was treated:

| State | iOS behaviour |
|---|---|
| `redacted` | May be displayed on a persisted surface. |
| `rawEphemeral` | Live view only; never for anything stored or re-served. |
| `unknown` (not stated) | **Handled exactly as strictly as raw — withheld.** An unstated treatment is not a treatment. |

There is deliberately no `.probablySafe` and no lenient default.
`06-PRIVACY-DATA.md` is explicit that a crop is not inherently safe: "a cropped
image can still contain a bystander's face, a private room, or a document".

**Artifact fetching itself is UNKNOWN.** iOS models the *state machine* around a
fetch (`absent` / `notFetched` / `fetching` / `available` / `failed`) but holds
**no URL, no id format, and no bytes**, because inventing a fetch scheme would be
exactly the fabricated contract this work refuses to produce. When a real fetch
contract lands, one case gains a payload and every other call site is already
written.

**iOS ships no privacy toggle backed by any of this**, because a switch the Tower
cannot honour converts a limitation into a false assurance.

---

## 6. What iOS deliberately did NOT assume — the short list

For the Tower's benefit, everything below is an open decision that iOS has left
open rather than guessed:

1. Any message name, route, or JSON key beyond the six that exist.
2. A module-selection or module-status message. Opening a cartridge on the phone
   sends **nothing**, and a test asserts the wire stays silent.
3. A geometry representation, element unit, or update mode.
4. A pose schema: coordinate frame, rotation convention, handedness, units.
5. A world identity or persistence scheme.
6. Contract versioning semantics — identifiers are compared for equality only, so
   ordering and compatibility remain the Tower's to define.
7. Experiment names, algorithms, metric names, metric units, or metric directions.
8. Annotation geometry or coordinate conventions.
9. Document retrieval routes or query syntax.
10. Scene relation predicates or object class vocabularies.
11. Artifact URLs, ids, or transfer mechanics.
12. A sensor-profile negotiation protocol — Rule 4 forbids designing one before
    the real DAT camera/stream configuration model is known.
13. Any tracking-quality or confidence *scale* beyond "a number 0-1 if you have
    one".
14. The **unit** of any spatial figure. Metric is not metres.
15. What redaction *does*. iOS states that the producer claimed an image was
    redacted; it does not tell a person what was removed, because no contract
    defines it.
16. What the Tower **stores**. iOS has no channel through which it could know,
    so none of its copy claims anything about Tower-side retention — not even
    the reassuring direction.
17. A monotonic integer for world revisions. `revision` is an opaque string
    compared for equality, because inequality is the entire requirement.

---

## 7. If you can only build one thing first

In iOS's order of usefulness, and with reasons:

1. **A capability declaration (§0.1).** Without it, every cartridge is stuck at
   "the Tower says nothing", and no other work on this list can be shown at all.
   It is also the smallest.
2. **Experimental CV Lab (§2).** It is Module #1 on the roadmap, its results
   shape is the least speculative, and it is the natural place to validate the
   module lifecycle/descriptor contract against real implementation experience
   before generalising — which is exactly what `03-ROADMAP.md` V0.9 says it is
   for.
3. **Redaction state (§5)**, as soon as *any* cartridge is about to return an
   image. Retro-fitting it is much worse than starting with it, because in the
   interval the app must withhold everything.
4. **World Builder lifecycle + progress (§1.1, §1.8)** before geometry. "A world
   is being built and here is what the Tower counts" is a complete, honest screen
   on its own, and it does not require the representation decision to have been
   made.

---

## 8. How to reconcile a real contract into iOS

When a contract lands, the iOS change is bounded and is the same shape every
time:

1. Add the contract identifier to `TowerCapabilities.declared` and to
   `TowerCapabilities.supported`. `ProductShellTests.testTheTowerDeclaresNoCartridgeContracts`
   will fail — that failure is the intended signal to review every consumer, not
   a nuisance to delete.
2. Write a Tower-backed client conforming to the cartridge's existing client
   protocol (`WorldBuilderClient`, `ExperimentalCVClient`,
   `DocumentMemoryClient`, `SceneUnderstandingClient`), mapping the wire payload
   onto the existing domain types. Publish every change through `stateUpdates` —
   the `state` property alone can be read but cannot announce, and the view
   models subscribe to that publisher in `init`.
3. Construct it in `CartridgeClients`, which `ProjectManager` owns. **Not in the
   workspace view**: that `@StateObject` is destroyed on every cartridge switch,
   and a client holding a subscription and a partly-built world must outlive it.
   This is why the view model initialisers have no default `client:` argument.
4. **No view changes.** Every view already renders the full lifecycle, including
   the states no client can currently reach.
5. Add the decode tests, including the negative ones: a malformed payload must
   produce `CartridgeFailure(kind: .undecodableResponse)`, not a partially
   populated snapshot and not a stronger claim than the Tower made. Document
   Memory already works this way — `DocumentQueryResult.init` throws rather than
   quietly downgrading an incoherent answer to "nothing matched".

Anything that does not fit that shape is a signal that this document guessed
wrong somewhere — which is worth saying out loud rather than working around.
