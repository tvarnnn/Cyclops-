# Reconciliation of `IOS-to-Tower.md` — 2026-08-23

Every requirement in the iOS product shell's document, classified against
what Tower actually does as of branch
`integration/cartridge-result-channel-v1`.

**Labels**

| Label | Meaning |
|---|---|
| **IMPLEMENTED NOW** | Built in this change, on the wire, tested |
| **AVAILABLE** | Already existed before this change |
| **DEFERRED** | Tower could do it; no contract is offered yet, and why |
| **BLOCKED** | Something concrete prevents it; the blocker is named |
| **NOT APPLICABLE** | Correctly absent — either iOS marked it NOT REQUESTED, or Tower cannot say it truthfully |

Nothing is silently ignored. Where a requirement is only partly met, it is
split into rows rather than rounded up.

---

## 0. Cross-cutting

| § | Requirement | Status | Detail |
|---|---|---|---|
| 0.1 | A capability declaration | **IMPLEMENTED NOW** | `GET /cartridges` and WS `{"type":"cartridges"}`, byte-identical (tested). Contract ids are opaque strings compared for equality, exactly as asked. All three of your states are distinguishable — `available` is a separate field from `contract` precisely so "offered but unreachable" cannot collapse into "not built yet" |
| 0.2 | Provenance on every derived value | **IMPLEMENTED NOW** (provenance) | `provenance: "inferred"` on geometry, trajectory and path length; `keyframes_accepted_provenance: "measured"` |
| 0.2 | …and the confidence where Tower has one | **NOT APPLICABLE** | `confidence: null`, deliberately. Tower keeps per-keyframe and per-edge confidence labels but has **never defined an aggregate for a whole reconstruction**. A number here would be one nobody specified. Render your "the Tower did not say" caveat |
| 0.3 | Observation time distinct from arrival time | **BLOCKED** | There is no capture timestamp anywhere in this system — the frame protocol carries no time field. Every timestamp is labelled `time_basis: "tower-receipt"`. This is `TOWER-TO-IOS.md` §6.6 and it needs **iOS work**: until `CMSampleBuffer`'s presentation timestamp is empirically established as capture time, Tower cannot invent one. Render capture time as unknown |
| 0.4 | Redaction state on every image | **IMPLEMENTED NOW** (the state) | Every artifact block carries `redaction: "none"` and `fetchable: false` |
| 0.4 | …redaction itself | **BLOCKED** | No redaction is implemented anywhere in Tower. World Builder keyframes are raw first-person frames. Consequence, and it is the right one: **no imagery is offered at all**, and none will be under this contract. Your "unstated treatment is withheld" rule has nothing to catch because nothing is offered |
| 0.5 | A unit string beside every figure | **IMPLEMENTED NOW** | `scale.unit` is `"world units"` or **null** when there is no unit at all. `path_length` carries `unit`, `scale_semantics` and a pre-rendered `display` |

---

## 1. World Builder — the first producer

### 1.1 Lifecycle

| Requirement | Status | Detail |
|---|---|---|
| Is a world being built at all, **distinct from "frames are arriving"** | **IMPLEMENTED NOW** | `lifecycle.state: "receiving"`, evidenced by a live process holding the world's **writer lock**. That lock is held for the lifetime of a mapping session and by nothing else, so it answers your question exactly. It is completely independent of frame traffic — a Release build that sends no frames still sees it |
| Has capture ended but work continues (`.finalizing`) | **NOT APPLICABLE**, with a substitute | Tower **cannot observe this**. While a build runs, the files on disk are byte-identical to "stopped and never built" and to "stopped and the build crashed": the build writes nothing until it finishes, emits no event, and the writer lock is already released. A state named `finalizing` would assert what cannot be seen. Instead: `state: "stopped_unbuilt"` means *the stored figures are not the final figures*, which is the half of `.finalizing` that is true, and `build_in_progress` is **`null`** — not `false`, which would claim no build is running. Map `.finalizing` from it if you wish; the caveat travels with it. **Amended 2026-09-10:** the Tower CAN observe a live build now -- `stop_session(hold_lock=True)` holds the writer lock through the final solve and the final build -- so `lifecycle.state: "finalizing"` with `build_in_progress: true` is a real observation. `stopped_unbuilt` keeps the old caveat, and projects to `finalizing` only when `geometry.available` is true; with no geometry there is nothing to wait for and it projects to `interrupted` |
| Failure, with a reason a person can read | **IMPLEMENTED NOW** (one case) | `state: "failed"` when the writer lock is held by a pid that is no longer running — a builder that died mid-session — with the pid in `evidence` and prose in `reason` |
| Failure, general case | **BLOCKED** | `end_reason: "error"` exists in Tower's vocabulary but **no code path writes it**, and `"interrupted"` has no recovery code to stamp it. A session whose process crashed leaves `ended_at: null`, which is byte-identical to a healthy live session — the stale lock is the only discriminator, and it is what the row above uses |

### 1.2 Identity

| Requirement | Status | Detail |
|---|---|---|
| Stable world id | **IMPLEMENTED NOW** | It **survives sessions**. A world is a long-lived artifact; sessions are windows within it. Your `worldID: String?` is never nil from this channel unless nothing could be resolved |
| Human-readable name | **IMPLEMENTED NOW** | `display_name`, operator-supplied, and **null when unnamed**. Never derived. (Tower's own text report renders an unnamed world as the string `"unknown"`; that is a CLI display choice and is deliberately not propagated to the wire) |
| Change detection | **IMPLEMENTED NOW** | `revision`, an **opaque string compared for equality**, exactly as §6.17 asked. Computed by hashing the payload rather than incrementing a counter, so it cannot drift from the content. Continuously-advancing figures are excluded, so a live session does not make every update look like a change |

### 1.3 Geometry

| Requirement | Status | Detail |
|---|---|---|
| `representation` — the Tower's own name | **IMPLEMENTED NOW** | `"sparse point cloud"`. Prose, not an identifier, because you display it verbatim and never parse it |
| `elementCount` | **IMPLEMENTED NOW** | `element_count` (points), with `element_name: "point"`. **Null, never 0**, when no build has run — "we never built" and "the build found nothing" must not render alike |
| `isIncremental` | **IMPLEMENTED NOW** | `is_incremental: false`. A build replaces the whole derived tree; it never emits a delta. Your warning that a UI assuming incremental updates would draw a partial world as complete is exactly why this is stated rather than implied |
| The representation decision itself | **Tower has made it, provisionally** | Sparse triangulated landmarks plus camera poses. When you add a renderer, that is the shape. It is a monocular classical-SfM output and will change if the backend does — which is why `geometry.revision` includes the build timestamp |

### 1.4 Trajectory

| Requirement | Status | Detail |
|---|---|---|
| Pose count | **IMPLEMENTED NOW** | `pose_count` = poses carrying a **position**, which is *not* `poses_solved`: an anchor has a position and Tower counts it as neither solved nor refused. Both underlying figures are also sent |
| Path length | **IMPLEMENTED NOW**, with honest refusals | Sent with `unit`, `scale_semantics`, `provenance` and a rendered `display`. **Refused** — `{"available": false, "reason": ...}` — when any pose was refused (a hole in the path is not a shorter path), when the session has more than one segment (segments share no coordinate frame, so a total would sum incomparable distances), or when the scale state is unknown |
| Path-length unit | **IMPLEMENTED NOW** | `"world units"`. Never metres |
| Pose array | **NOT APPLICABLE** | You marked it NOT REQUESTED and you were right. No poses, no translations, no rotations, no quaternions cross this wire — a test greps the payload for all of them. The five decisions you listed remain unmade because nothing forces them |

### 1.5 Calibration and scale

| Requirement | Status | Detail |
|---|---|---|
| Every spatial figure labelled `relative`/`inferredMetric`/`measuredMetric` | **IMPLEMENTED NOW** | `scale.semantics` uses **your** vocabulary, alongside `scale.state` in Tower's. When Tower's state is `unknown`, `semantics` is **null and no distance figure is sent at all** — so your "an unlabelled figure is not shown as a distance" rule has nothing to catch |
| `inferredMetric` / `measuredMetric` | **BLOCKED** | Unreachable in V1 and verified so: `SCALE_ESTIMATED` is defined in Tower and referenced nowhere; `measured` is only ever *defended*, never written; both backends declare `produces_metric_scale=False`. On monocular hardware with no IMU this is the honest outcome, and it is the one you predicted |
| Calibration state, coarse | **IMPLEMENTED NOW** | `unknown` / `uncalibrated` / `calibrated`, mapped on whether the intrinsics are **physically possible**, not merely present — intrinsics with `fx=0` or `fx=NaN` report `uncalibrated` even though a source is declared |
| `calibrating` | **NOT APPLICABLE** | No code path can produce it. Calibration is an offline procedure run before a session; there is no in-session state to be in the middle of. `calibrating_ever_reported: false` says so on the wire |
| No calibration percentage | **Honoured** | None is sent, and none exists |
| **Scope** | **Correction you should absorb** | Calibration is a property of the **session**, not the world — `scope: "session"`. Intrinsics are resolution-keyed because DAT's ladder changes resolution mid-stream, and two sessions of one world can legitimately differ. A world-level calibration state would be a fabrication |

### 1.6 Tracking

| Requirement | Status | Detail |
|---|---|---|
| Coarse good / limited / lost | **IMPLEMENTED NOW** for `good`, `lost`, `unknown` | Both are real events with names in Tower's code |
| `limited` | **NOT APPLICABLE** | It would require inventing a threshold. The nearest candidate is documented in Tower's own source as an untuned placeholder and is not emitted as an event at all, so it is not even available live. `limited_ever_reported: false` is on the wire so you never wait for a case that cannot arrive. This is the same reasoning that made you refuse a percentage |

### 1.7 Persistence and inspection

| Requirement | Status | Detail |
|---|---|---|
| Did the world survive the session | **IMPLEMENTED NOW** | `persistence.state: "saved"` with a `revision`. World Builder persists everything by construction, so `session`-only never occurs |
| Reload a saved world | **IMPLEMENTED NOW** | Pass `world_id` (and optionally `session_id`) on subscribe. That pins the target — no live session is followed, and a counter that moved would indeed be a bug |
| Where is it stored | **NOT APPLICABLE**, and enforced | `location_disclosed: false`. **No filesystem path ever crosses this wire** — a test asserts it. A Tower path is useless to a phone and names a machine's layout to a remote consumer |

### 1.8 Progress

| Requirement | Status | Detail |
|---|---|---|
| Whatever the Tower actually counts | **IMPLEMENTED NOW** | `keyframes_accepted` — and it is counted from the event journal while a session is live, **not** from the session record, which still holds the zero written at session start. Reading the obvious field would have reported a confident wrong **0** |
| A count the Tower does not keep must not be invented | **Honoured, and it bites immediately** | `frames_observed` is **null while live**, with a reason field explaining that an ordinary rejected frame writes no journal event. It appears, with the full rejection histogram, once the session stops. This is your `nil ≠ 0` rule producing a real null |
| `mappingSeconds` on the Tower's clock | **IMPLEMENTED NOW** | `mapping_seconds` with `mapping_clock: "tower"`, clamped so a backward NTP step cannot produce a negative |

---

## 2. Experimental CV Lab

**Overall: DEFERRED.** The channel is built to carry it and no contract is
offered. The blocker is a design decision, not transport.

| § | Requirement | Status | Detail |
|---|---|---|---|
| 2.1 | Experiment registry | **DEFERRED** | Tower has a real registry (`tower/experiments/EXPERIMENTS`) with ids and names, so this is genuinely cheap. It is not offered because a partial CV Lab contract would put you in your own "the Tower offers a contract this build does not implement" state for a cartridge whose results already reach you |
| 2.2 | Lifecycle, refusals legible | **DEFERRED** | The refusal half is now true generally: an unrecognised WS message returns `protocol_error: unknown_message_type` instead of being silently logged |
| 2.3 | Metrics with label/value/unit | **AVAILABLE, partially** | `frame_result.metrics` is a `name -> number` bag added 2026-08-22. It carries no unit and no provenance |
| 2.3 | Measured vs inferred, **required** | **DEFERRED** | This is the substantive gap. The Lab's `ExperimentResult` has no provenance field. Adding it is a cartridge change, not a channel change |
| 2.3 | Baseline + `higherIsBetter` | **DEFERRED** | Tower's benchmark scripts hold baselines; the module does not |
| 2.3 | Frames processed | **AVAILABLE** | Counted in session metrics, not currently on the wire |
| 2.4 | Tower processing time | **AVAILABLE** | `frame_result.processing_ms`, already exactly this shape |
| 2.4 | End-to-end latency | **NOT APPLICABLE** | You correctly declined it; two unrelated clocks do not make a latency |
| 2.5 | Annotation count | **DEFERRED** | |
| 2.5 | Rendered annotated frame | **BLOCKED** | It is an image, so §5 applies: no redaction exists, so no imagery is offered |
| 2.5 | Annotation geometry | **NOT APPLICABLE** | NOT REQUESTED, and Tower has made no coordinate-convention decision to send |
| 2.6 | Cancellation | **BLOCKED** | There is no handle to cancel. `ModuleContainer.process()` is synchronous on the event loop; a long job has nothing to stop. This is V1.1 lifecycle work and V1.1 is blocked on an unrecorded ruling |
| 2.7 | Dataset recording, state clearly indicated | **AVAILABLE** | `GET /health` reports `capture.armed` / `capture.recording` / `frames_written`, added 2026-08-22 |
| 2.7 | …arming it from iOS | **BLOCKED** | `TOWER_CAPTURE_ROOT` arms the recorder at Tower start-up and `stream_start`/`stream_stop` bound each recording. There is no way for iOS or the wearer to arm it. You are right not to ship an indicator the Tower cannot drive — but you *can* now display the real state truthfully |

---

## 3. Document Memory

**Overall: DEFERRED**, and one item **BLOCKED by physics**.

Note your scope caveat is now out of date in one direction: Document
Memory **is implemented on Tower** (`tower/document_memory/`) as of
2026-08-22 — records, store, detection, dwell, OCR, retrieval and CLIs. It
is still not an adopted *module*, and no contract is offered.

| § | Requirement | Status | Detail |
|---|---|---|---|
| 3.1 | Recent documents, ids, titles, summaries, confidence | **DEFERRED** | All exist in Tower's records. No summary is generated — an abstractive summary of partial capture was rejected as fabrication |
| 3.2 | `unknown` / `notReadable` / `extracted(characterCount:)` | **DEFERRED**, and the design already matches | Tower stores derived text, not frames, and a character count is trivially derivable. `notReadable` as a first-class answer is already how the cartridge behaves |
| 3.2 | …usable text at all | **BLOCKED** | Measured: word recall is 0.957–1.000 at 1280×720 and **0.429–0.810 at the 640×360 iOS delivers**. Page *detection* is unaffected; only recognition is starved. This is `TOWER-TO-IOS.md` §6.8 and it is **iOS/DAT work** — a way to request an occasional higher-resolution still, or failing that a way to learn which rung of the adaptive ladder is active so a reading can be recorded as untrustworthy rather than stored silently |
| 3.3 | Observation time | **BLOCKED** | Same as §0.3: tower-receipt only |
| 3.3 | Time in view, **not** `viewing_duration` | **Honoured** | Tower independently reached the same conclusion: the record is `DocumentObservation`, the CLI prints **OBSERVED, NOT READ**, and a test bans `looking_at`, `gaze_direction`, `is_looking`, `face_id`, `person_id` across every cartridge — now including this channel |
| 3.4 | `recent` / `text` / `observedWithin` / `semantic` | **DEFERRED**, three of four exist | Recent, literal text (BM25) and time-range are implemented. **`semantic` is BLOCKED**: embedding retrieval was rejected — no corpus justifies it, and BM25's explainability matters more while an answer must be traceable |
| 3.5 | `matched` / `notFound` / **`noObservation`** | **DEFERRED**, and the distinction already exists | Tower's retrieval already separates "searched and found nothing" from "holds nothing covering what was asked". You are right that collapsing them is the dangerous error |
| 3.6 | Pagination | **NOT APPLICABLE yet** | Tower expects `recent(limit:)` to be sufficient at V1 scale. If a long history appears, this is the seam to revisit |
| 3.7 | Thumbnails | **BLOCKED** | §5. Page images are opt-in and **off**, and unredacted when on |
| 3.8 | `sessionID`, optional `worldID` | **DEFERRED** | And the independence you require holds: Document Memory does not depend on World Builder |

---

## 4. Scene Understanding

**Overall: BLOCKED for *this* channel, specifically and for a good reason.**

Scene Understanding is implemented on Tower (`tower/scene/`) and
**persists nothing** — no store, no journal, no imagery, enforced by test.
This channel reads persisted state. There is therefore **nothing on disk
for it to read**, and giving the cartridge a store to make it publishable
would pre-empt Environmental Memory's entire reason to exist.

It needs the **live in-process module path**, which is the half of
`TOWER-TO-IOS.md` §6.1 that remains blocked at V1.0/V1.1.

| § | Requirement | Status | Detail |
|---|---|---|---|
| 4.1 | Anonymous, session-scoped track handle | **BLOCKED (transport)** | Exists and is exactly what you asked: restarts at 1 each session, IoU association only, never appearance. A person who leaves and returns is deliberately a **new track** |
| 4.1 | Person vs object; class label; confidence | **BLOCKED (transport)** | All exist. Labels are COCO categories — "chair", never "your chair" |
| 4.1 | Do not send a durable person identifier | **Honoured permanently** | No such identifier exists anywhere in Tower, and an AST test bans `person_id` and `face_id` outright |
| 4.2 | Orientation `unknown`/`towardCamera`/`awayFromCamera`/`acrossView` | **BLOCKED (transport)**, and **off by default** | Implemented from COCO eye/ear keypoint **visibility**, which is genuine coarse evidence. It costs a measured **43.4 ms** per call on CUDA and **956.4 ms** on CPU, and CPU is the default device — 1.43× the detector on CUDA, 29.1× on CPU, against a measured 83.5 ms delivered frame interval. It runs at a bounded cadence (3 frames, ~250 ms) and every estimate carries its age, which the CPU path and an NTP-step guard both still require. The CUDA build that was named as the unblocker exists and was measured on 2026-08-26; the earlier 798 ms / 24× / 2.5× figures were CPU-with-synthetic-input and are withdrawn |
| 4.2 | It is not gaze; do not send `gaze`/`looking_at`/`attention` | **Honoured** | The state is `toward_wearer`, the property is `appears_facing_wearer`, confidence never reaches HIGH, and the vocabulary ban is enforced by test across every cartridge and now this channel |
| 4.3 | Frame of reference | **Camera-relative, and only that** | World-anchored positions are **BLOCKED**: World Builder produces poses offline, after a session, so there is no live pose to anchor to. No world ids are invented |
| 4.3 | Bearing, with the sign convention stated | **NOT YET STATED — and this is a live question for Tower** | You declared yours: degrees from straight ahead, **positive to the right**. Tower currently asserts only `left_of` / `right_of` from box centroid x and computes no bearing at all. When it does, it will adopt your convention and say so. Recorded here so it is not decided by accident |
| 4.3 | Distance, with scale semantics | **BLOCKED, now on a real measurement** | Tower refuses depth-dependent relations entirely. Measured 2026-08-26 on 9,199 real frames: MiDaS relative inverse depth orders two detector boxes stably while the scene is still (3.8% reversals overall, 0.0% above 0.40 separation) and degrades to **11.5% reversals at matched separation** once the wearer moves — on a corpus whose 99th-percentile inter-frame motion is 56 px and which contains no walking. There is also no metric scale: MiDaS-small emits relative inverse depth and nothing converts it to distance. See `docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md` |
| 4.4 | Counts derived from the track list | **Agreed, and Tower already matches** | Counts come from **confirmed tracks**, and hold exactly at 0/10/20% detector dropout. Tower attaches no meaning beyond "currently tracked within the field of view" |
| 4.5 | No "behind you", no "beside you" | **Honoured** | Tower asserts nothing outside the forward cone |
| 4.6 | Opaque relation predicates | **Compatible** | Tower asserts two (`left_of`/`right_of`, `higher_in_view`) and **refuses six**, each with the evidence that would settle it. One, `nearer_than_same_class`, shipped and was **withdrawn** on a counterexample: two chairs at the same distance, one face-on and one edge-on, differ 2.5× in box area |
| 4.7 | Staleness; a lost track disappears | **Honoured** | Tracks expire rather than being extrapolated, and facing estimates expire to `unknown` rather than being deleted — a missing field would read as "not facing" |
| 4.8 | Coalesce before publishing | **Already the design** | This channel coalesces to newest-wins with one slot per subscriber. When Scene Understanding publishes, it inherits that |
| 4.9 | Persists nothing | **Honoured, and enforced** | Which is precisely why it cannot use this channel |

---

## 5. Imagery and redaction

| Requirement | Status | Detail |
|---|---|---|
| Every image states its treatment | **IMPLEMENTED NOW** | `redaction: "none"` on the World Builder artifact block |
| `redacted` imagery | **BLOCKED** | No redaction exists anywhere in Tower. `Session.redaction` was carried from day one as the one un-retrofittable field, and its honest V1 value is `"none"` |
| Therefore | **No imagery is offered** | `fetchable: false`, and **no id or URL is minted**. Your rule handles an unstated treatment by withholding; Tower withholds first, so the rule never has to fire |
| Artifact fetching | **DEFERRED, deliberately** | You hold no URL, no id format and no bytes, and inventing a fetch scheme would be the fabricated contract this work refuses. When redaction exists, that is when a fetch contract is worth designing |
| `images_purged` | **Correction you should absorb** | The flag **deletes nothing**. It makes rebuilds refuse. A world carrying it was verified to still have every JPEG on disk. The wire reports `images_purged_declared` with `images_purged_verified: null` and prose saying so. Never render it as "the imagery is gone" |

---

## 6. Your "deliberately did not assume" list

Every one of these was Tower's to decide. Where a decision has now been
made, it is here; where it has not, it stays open.

| # | Open decision | Now |
|---|---|---|
| 1 | Message names, routes, JSON keys | **Decided**: see `CARTRIDGE-RESULTS.md`. Additive; the six existing messages are unchanged |
| 2 | A module-selection/status message | **Still none.** Opening a cartridge sends nothing. Subscribing is an explicit act |
| 3 | Geometry representation, element unit, update mode | **Decided**: sparse point cloud, `point`, snapshots not deltas |
| 4 | A pose schema | **Still open**, and deliberately — no poses are sent |
| 5 | World identity and persistence scheme | **Decided**: opaque world id surviving sessions; always saved |
| 6 | Contract versioning semantics | **Decided your way**: opaque, equality only, dated not numbered |
| 7 | Experiment names, metric names/units/directions | **Still open** — CV Lab is deferred |
| 8 | Annotation geometry | **Still open** |
| 9 | Document retrieval routes/syntax | **Still open** — deferred |
| 10 | Scene relation predicates, class vocabularies | **Partly decided** on Tower (two asserted, six refused), not yet on a wire |
| 11 | Artifact URLs, ids, transfer | **Still open**, deliberately |
| 12 | Sensor-profile negotiation | **Still open.** Rule 4 forbids designing one before DAT's real configuration model is known. But §6.8's *requirement* is now measured, not hypothetical |
| 13 | A confidence scale | **Still open.** Tower sends `confidence: null` rather than a number nobody defined |
| 14 | The unit of any spatial figure | **Decided**: `"world units"`, never metres |
| 15 | What redaction does | **Still open**, and moot while nothing is redacted |
| 16 | What the Tower stores | **Partly answered**: `persistence.state`, `retains_raw_imagery`, and the artifact block say what exists — without saying where |
| 17 | A monotonic integer for world revisions | **Decided your way**: an opaque string. Note the envelope's `seq` *is* a monotonic integer, but it is an ordering guarantee for the message stream, not a world revision. The two are separate on purpose |

---

## 7. Your "if you can only build one thing first"

| Your priority | Outcome |
|---|---|
| 1. **A capability declaration** | **DONE.** Built first, for exactly the reason you gave |
| 2. **Experimental CV Lab** | **Deferred.** World Builder went first because the brief named it, and because its state is already persisted and therefore readable without touching the frame path. CV Lab needs provenance and a baseline on `ExperimentResult` — a cartridge change |
| 3. **Redaction state** | **Done as a state**; redaction itself is blocked, so the honest outcome is that no imagery is offered at all |
| 4. **World Builder lifecycle + progress before geometry** | **DONE — and geometry too.** You were right that lifecycle plus "here is what the Tower counts" is a complete, honest screen on its own; it is also everything a live session can truthfully show, because geometry does not exist until a build runs |

---

## 8. What Tower now needs from iOS

Stated as requirements, not designs, mirroring your own format.

1. **A capture timestamp with documented semantics** (§0.3). Until then
   every time in the system is tower-receipt and must be labelled so.
2. **A way to request an occasional higher-resolution still**, or failing
   that **a way to learn which rung of the adaptive ladder is active**
   (§3.2). The measurement is in `TOWER-TO-IOS.md` §6.8. Without it,
   Document Memory cannot be trusted to read a page.
3. **Confirmation of the bearing sign convention** when Scene
   Understanding eventually publishes (§4.3). You declared yours; Tower
   will adopt it and say so, and this note exists so it is not decided by
   accident.
4. **Nothing else.** The result channel needs no iOS-side protocol
   decisions — the message names, keys and semantics are all Tower's, as
   you asked.
