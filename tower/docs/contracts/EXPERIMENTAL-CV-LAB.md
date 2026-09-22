# Experimental CV Lab — the Tower ↔ iOS contract

**Status:** implemented on Tower, offered on the wire.
**Contracts:**

| What | Identifier |
|---|---|
| Status document | `experimental_cv.status/2026-08-27` |
| Control vocabulary | `experimental_cv.control/2026-08-27` |
| `frame_result` provenance block | `experimental_cv.frame_result/2026-08-27` |
| Envelope (shared) | `cartridge_results.envelope/2026-08-23` |

Cartridge name on the wire: `experimental_cv`. Result type: `status`.
Every timestamp: `tower-receipt`.

All four identifiers are **opaque and compared for equality only**. Do not
parse them, order them, or range over them. Three of them because the CV
Lab has three surfaces that version independently — a client may implement
the read-only half and never send a command, which is exactly what a
Release iOS build with no camera should do.

---

## 0. What this replaces

Choosing an experiment used to mean: edit `TOWER_CV_EXPERIMENT`, restart
the Tower, go to the Home screen, press a generic Start, and read an
unlabelled number off a debug panel. Every step of that follows from one
fact — the experiment was decided at process start and nothing on the wire
said which one it was.

Now:

- the Tower **enumerates** its experiments, with stable ids and metadata;
- a client **selects and starts** one with no restart;
- **pause, resume and stop** are requests with legible refusals;
- **every result names the run, the experiment and the configuration that
  produced it**, so a result from a previous experiment cannot be read as
  a result from the current one;
- the Tower reports **whether frames are actually arriving**, which is
  what "I pressed Start and nothing happened" is really asking.

`TOWER_CV_EXPERIMENT` still exists. It is now the **startup default** and
nothing more: what this Tower arms at boot so that a client which knows
nothing about the CV Lab still gets a `frame_result` for every frame,
exactly as before.

---

## 1. The three surfaces

| Surface | Direction | Carries |
|---|---|---|
| `frame_result` (existing) | Tower → client | the per-frame answer, plus a `cv_lab` provenance block |
| `cv_lab_*` messages | both | the control plane: status, start, pause, resume, stop |
| `cartridge_result` on the result channel | Tower → client | the same status document, pushed |

**The per-frame path is unchanged and was deliberately not replaced.** It
is the live path, it is already latency-measured, and duplicating it would
have meant two answers to "what did the Tower see in that frame". What
changed is that the answer now says who produced it.

**Commands never travel on the result channel.** `tower/results/` is a
read-only reporting surface; putting a mutation on it would make the next
cartridge's producer a place where somebody looks for one.

---

## 2. Discovery

### 2.1 Is the cartridge offered?

`experimental_cv` now appears in the capability declaration
(`GET /cartridges`, and `{"type": "cartridges"}` on the socket) alongside
`world_builder`:

```json
{
  "cartridge": "experimental_cv",
  "result_type": "status",
  "contract": "experimental_cv.status/2026-08-27",
  "available": true,
  "unavailable_reason": null,
  "snapshot_only": true
}
```

`available: false` with a reason means this build speaks the contract but
cannot serve it — the Lab module failed, or this Tower runs without one.
That is iOS's third state ("offered, implemented, unreachable → connect"),
never its first ("the Tower says nothing → not built yet").

### 2.2 What experiments exist?

Three surfaces, **one document from one function**:

```
GET /cv-lab                          → {"contract", "control_contract", "status", "process"}
{"type": "cv_lab_status"}            → {"type": "cv_lab_status", ..., "status"}
result_subscribe experimental_cv/status → cartridge_result.payload IS that "status"
```

`GET /cv-lab` alone also carries `process`, **beside** the document and
never inside it:

```json
"process": {"pid": 24036, "threads": 98, "rss_mb": 1622.0}
```

It is the Tower's reading of its own process (psutil), for the operator
asking "did the last switch give its memory back" from a machine they
have no shell on, and for the switch soak (`scripts/cv_lab_switch_soak.py
--live`). It is not in the document because the document is one object on
three surfaces and a test holds them byte-equal; a thread count differs
between two reads a millisecond apart, and the result channel would
republish the whole document on every poll for a number that moved by
0.1 MB. Each field is `null` when psutil cannot answer.

A test asserts the three agree. **Not byte-identical across time**, and do
not build anything on that: `elapsed_s`, the three `throughput` figures,
`last_frame_at`, `receiving_frames` and `clients_connected` are clock- or
connection-derived, so two reads a second apart differ for reasons that
are not the contract. Two reads at the same instant would be identical;
you cannot take two reads at the same instant. The claim that holds is
structural — same keys, same types, same meanings, one builder — and the
channel's `revision` is not derivable from `GET /cv-lab`.

`GET /cv-lab` exists for the operator with a terminal; the Tower is
normally driven over Tailscale where a server-side log line is invisible.

There is **no HTTP surface for start, pause or stop**. A command needs the
connection it was issued on to still be there when the outcome arrives — a
start may take two minutes to arm a model — and a request/response route
would have to either block for that or lie about having finished.

---

## 3. The status document

One document answers every question about the Lab. It is a **complete
snapshot**; there are no deltas to merge.

```json
{
  "contract": "experimental_cv.status/2026-08-27",
  "control_contract": "experimental_cv.control/2026-08-27",
  "frame_result_contract": "experimental_cv.frame_result/2026-08-27",
  "tower_instance_id": "2a5b04b1b77c",
  "time_basis": "tower-receipt",
  "lifecycle": {"state": "running", "reason": null,
                "since": 1787810078.29, "run_id": "2a5b04b1b77c-1"},
  "available": [ /* the catalog, §3.2 */ ],
  "selected": "edge_detection",
  "default_experiment": "baseline",
  "device_requested": "auto",
  "run": { /* §3.3 */ },
  "source": { /* §3.5 */ }
}
```

### 3.1 `lifecycle.state` — seven values, and how iOS renders them

| Tower state | Means | iOS `ExperimentalCVState` |
|---|---|---|
| `unavailable` | this Tower cannot run experiments at all | `.unsupported(reason:)` |
| `idle` | nothing armed; a start would be accepted | `.idle(available:)` |
| `starting` | a start was accepted, the experiment is loading | `.starting(experiment)` |
| `running` | processing frames | `.running(run)` |
| `paused` | armed and deliberately not processing | **`.paused(run)` — a new case; see `CV-LAB-IOS-HANDOFF.md` §4** |
| `stopped` | the last run ended; its figures are final | `.completed(run)` |
| `failed` | the last start failed, **or the running experiment raised on a frame**; the experiment is released and another start may be sent | `.failed(CartridgeFailure)` |

`stopped` rather than `completed` on the wire, deliberately. A bench run
does not complete; it is stopped by a person. The Tower says what happened
and iOS renders it with the case its state machine has.

`paused` and `stopped` are different states because the difference is
real, and it is two differences:

- a paused run keeps the experiment **loaded**, so resuming a `depth` run
  costs nothing while a stopped one pays the model load again;
- a paused run is **not over**, so it keeps counting: `frames_processed`
  and the metrics stop moving, but `frames_refused` climbs with every
  frame that arrives. A stopped run freezes everything. If you want to
  see whether the phone is still sending while paused, that counter is
  where it shows.

`lifecycle.reason` is prose for a person, present only when the state
needs explaining. `null` is not "no reason" — it is "the state speaks for
itself".

**`failed` is always recoverable, since 2026-09-06.** Three things put a
Lab there and the same `cv_lab_start` brings it back from all of them: an
interactive start whose load raised; the startup default
(`TOWER_CV_EXPERIMENT`) naming something unknown or failing to load at
boot — logged at ERROR, and the module is still ACTIVE; and the running
experiment raising anything other than a `FrameProcessingError` on a
frame — the run ends, `frames_failed` counts that frame, the experiment
is released at once, and `reason` names the exception. Nothing a frame
does can reach the module's terminal FAILED state any more. What still
can: a startup load that overruns the container's 120 s bound, which is
an abandoned thread rather than a failed one.

`lifecycle.since` is when the Lab entered this state.

`lifecycle.run_id` is the current run, or `null`.

### 3.2 `available` — the catalog

One entry per registered experiment, sorted by `id`. **iOS holds no list
of its own**: `docs/modules/EXPERIMENTAL-CV.md` calls its candidate list
"intentionally broad", so any subset hard-coded on the phone would be the
app asserting that those experiments exist.

```json
{
  "id": "edge_detection",
  "name": "Edge detection",
  "summary": "Fraction of pixels Canny calls an edge, after a Gaussian blur. ...",
  "provenance": "measured",
  "headline_label": "edge_density",
  "headline_unit": "fraction",
  "stateful": false,
  "requires_model": false,
  "backend": "opencv",
  "annotation_metric": null,
  "available": true,
  "unavailable_reason": null
}
```

`id`, `name` and `summary` are the three iOS reads. Everything else is
additive; a client that ignores all of it still has a working picker.

- `provenance` — `measured` or `inferred`. Not a hint. See §4.
- `headline_label` / `headline_unit` — what this will measure, and in
  what. A `null` unit means the quantity genuinely has none and is
  rendered **bare** (`IOS-to-Tower.md` 0.5: metric is not metres). Depth
  is the case: MiDaS-small emits relative inverse depth on an arbitrary
  scale.
- `stateful` — carries state across frames, so its first frame is not
  like its hundredth.
- `requires_model` — needs the optional `[ml]` extra. A start may take a
  hundred times longer than a cheap experiment's.
- `backend` — `opencv` or `torch`.
- `annotation_metric` — the metric that is a count of things found in a
  frame, or `null`. Only `object_detection` has one.
- `available` / `unavailable_reason` — false when this Tower is missing
  a module the experiment needs, checked **per experiment**: `depth` needs
  `torch` and `timm`, `object_detection` needs `torch` and `torchvision`.
  Starting such an experiment is refused in advance, with a reason.
  **What this cannot check is the network**: `depth` fetches MiDaS weights
  through `torch.hub` on first use, so an offline Tower reports it
  `available: true`, accepts the start, and then goes `failed` with the
  reason. That is why `failed` is recoverable.

The eight registered today: `baseline`, `depth`, `edge_detection`,
`feature_detection`, `frame_quality`, `object_detection`, `optical_flow`,
`redaction_impact`.

### 3.3 `run` — what the current or last run measured

`null` when no run exists. Otherwise:

```json
{
  "run_id": "2a5b04b1b77c-2",
  "experiment": { /* the same shape as an `available` entry */ },
  "origin": "client_request",
  "started_at": 1787810180.1, "ended_at": null, "elapsed_s": 14.2,
  "arm_ms": 231.4,
  "runtime": {"backend": "torch", "device": "cuda:0",
              "device_requested": "auto", "model": "MiDaS_small",
              "torch_threads": 2},
  "frames_offered": 12, "frames_processed": 12,
  "frames_refused": 0, "frames_failed": 0,
  "metrics": [ /* §4 */ ], "metrics_omitted": 0,
  "unclassified_metrics": [],
  "annotation": { /* §5 */ },
  "timings": { /* §6 */ },
  "throughput": { /* §6 */ }
}
```

`origin` is `client_request` or `startup_default`. **`startup_default`
means nobody asked for this run** — the Tower armed it at boot. Reported
so that "the Lab is running" never reads as "somebody chose this".

**On a run that has processed no frame — which is every Release build,
and every Tower nobody has streamed to yet — these are `null`, not zero:**

```json
"metrics": [], "annotation": {"count": null, ...},
"timings":    {"processing_ms": null, "processing_ms_max": null,
               "stage_ms": {}, "observed_at": null,
               "time_basis": "tower-receipt"},
"throughput": {"processed_fps": null, "offered_fps": null,
               "capacity_fps": null}
```

**Every number under `timings` and `throughput` is nullable, and the
reply to your very first `cv_lab_start` will exercise it.** `null` is
"nothing has been measured"; it is never a zero you can render as one.

The two `fps` figures are `null` specifically while `elapsed_s` is `0.0`,
because a rate over a zero-length window is undefined rather than zero.
That is not a rare edge: `time.time()` on Windows has ~15.6 ms
granularity, so a run started and read in the same tick reports
`elapsed_s: 0.0` — measured at 11 of 12 `cv_lab_start` replies. They
become numbers a few milliseconds later. `capacity_fps` is `null` until
one frame has been processed, because it is derived from measured
per-frame cost.

`arm_ms` is how long this run took to arm: from the `cv_lab_start` being
accepted (or the boot for a `startup_default` run) to the moment its
experiment was installed and could answer a frame. `null` while `starting`
and `null` forever for a run whose arm failed. Measured on a monotonic
clock, so it is meaningful below the wall clock's 15.6 ms Windows tick.
Warm figures on the reference host: ~230 ms for `depth` on CUDA, ~150 ms
for `object_detection`, under 1 ms for everything else; the very first
`depth` arm in a process is ~2-3 s (torch import, CUDA context, hub
cache).

`runtime` is what the experiment says it actually loaded, and is empty for
an experiment that holds nothing. Its keys are the experiment's own; do
not switch on them. (The two model-backed experiments report
`torch_threads`, the intra-op budget they applied — see
`TOWER_CV_TORCH_THREADS` in the README — beside `device`.) It exists because `TOWER_CV_DEVICE=auto` is a
**request** and the Tower decides the answer — a run labelled "auto" has
not said whether it used the GPU, and a CPU figure with a GPU label on it
is a real failure this closes.

**`frames_offered` is derived**: it is exactly
`frames_processed + frames_refused + frames_failed`, so the sum holds at
every read rather than only between them. (A frame currently being
processed is in none of them yet, because its outcome is not known.) That
invariant is what makes a dead start diagnosable:

| Reading | Means |
|---|---|
| `frames_offered == 0`, `source.frames_rejected_before_lab == 0` | nothing is reaching the Tower at all. The stream is not running |
| `frames_offered == 0`, `source.frames_rejected_before_lab > 0` | frames ARE arriving and the transport cannot decode them. A sender problem, not a Lab one |
| `frames_offered > 0`, `frames_processed == 0` | frames are arriving and the Lab is refusing them. Check `lifecycle.state` |
| `frames_failed > 0` | the experiment raised on a frame. It stays armed; those frames produced nothing |

Frames rejected by wire validation before they ever reach the Lab are
**not** counted here — they never reached it. They appear in
`source.frames_rejected_before_lab` (§3.5) and in the Tower's own session
summary as `frames_rejected`.

`metrics_omitted` is how many aggregate metrics did not fit the 16-row
bound. Reported rather than silently truncated.

`unclassified_metrics` names any metric an experiment emitted without
declaring how it combines across frames. Empty is the only correct value
and a test enforces it for every registered experiment; this is what the
wire says if one ever reaches production anyway.

### 3.4 Switching discards the previous run; stopping freezes it

Starting a different experiment mints a **new** run and the previous run's
figures leave the document. That is the point: a run is the unit of
provenance, and keeping an old summary beside a new one is how a number
from the wrong experiment ends up on a screen. Press **Stop** to keep a
run's figures readable; they stay until the next start.

**A stopped run stops counting.** Every field under `run` is frozen at the
moment of the stop, including the frame counters — frames that keep
arriving afterwards are refused and counted by `source`, which is a
property of the Tower rather than of a run that ended. An earlier build
kept adding refused frames to the stopped run's `frames_offered` while
`elapsed_s` stayed frozen, so `offered_fps` climbed without bound: 8
frames over 9 s read 0.89, and 400 refused frames later the same nine
seconds read 45.3.

### 3.5 `source` — is anything feeding this Lab

```json
{"clients_connected": 1, "receiving_frames": true,
 "last_frame_at": 1787810180.83, "frames_offered_total": 7,
 "frames_rejected_before_lab": 0, "idle_after_s": 5.0}
```

`frames_rejected_before_lab` counts frames that arrived and the transport
could not decode — a truncated JPEG, a bad base64, a missing field. They
never reach the Lab, so `frames_offered_total` does not count them, and
without this field a phone sending garbage reads exactly like a phone
sending nothing. Those need opposite fixes.

`receiving_frames` is `last_frame_at` within `idle_after_s`. Five seconds,
because the current iOS sender forwards roughly one frame in thirty of a
~24 fps capture — about 0.8 frames per second observed — so five seconds
is about four missed frames: long enough never to flicker during normal
streaming, short enough to show up while a person is still standing there.

`clients_connected` is `null` when this Tower cannot report it.

**Every figure in this block is TOWER-WIDE, not per connection.** One
Tower has one Lab and one run, so `receiving_frames: true` means *somebody*
is feeding it — possibly the other phone. There is no per-connection frame
counter anywhere in this contract, because the Lab is handed bytes and not
a connection identity.

That matters for §7: `receiving_frames` alone cannot tell a client that
**its own** frames are arriving. Combine it with what the client already
knows — whether this build is streaming at all — and read
`clients_connected > 1` as "somebody else is on this Tower too".

---

## 4. Metrics, and provenance on every one

```json
{"label": "edge_density", "value": 0.0413, "unit": "fraction",
 "aggregation": "rate", "frames": 6, "provenance": "measured",
 "confidence": null, "headline": true, "varied": false,
 "baseline": null, "higher_is_better": null}
```

- `label` — the Tower's word, displayed verbatim. **iOS matches on no
  metric name, ever.**
- `value` — `null` is a real answer meaning "this metric has no meaningful
  aggregate", never "zero". See `aggregation`.
- `unit` — `null` means the quantity has no unit and is rendered bare.
- `provenance` — **required, never omitted.** `measured` or `inferred`.
  `EXPERIMENTAL-CV.md` requires that experiment output be distinguished
  from measured sensor fact, and iOS makes this a non-optional field so
  that whoever decodes the reply has to answer it. A `constant`-kind
  metric is always `measured` even when the experiment as a whole infers:
  a configured threshold or an image dimension is a fact about how this
  Tower is set up, not a model's opinion.
- `confidence` — always `null`. The Tower has no calibrated confidence for
  any of these, and a number here would be invented.
- `headline` — the experiment's single most important number, always
  first in the list.
- `aggregation` — how this number was combined across frames:

  | Value | Combined by | Because |
  |---|---|---|
  | `rate` | mean | summing a fraction is nonsense |
  | `count` | sum | including a 0/1 flag, whose sum is "how many frames it fired on" |
  | `constant` | the value observed | neither a sum nor a mean of an image width means anything |
  | `unaggregated` | **nothing** — `value` is `null` | `dominant_direction_deg` is circular: the mean of 179° and −179° is 0°, the one direction neither frame was moving in |

- `varied` — a `constant` that was not constant. Its `value` is `null` and
  this says why, so `null` is not read as "never observed".
- `baseline` and `higher_is_better` — **always `null`**, and this is not an
  omission. iOS renders a better/worse verdict only when **both** arrive,
  and the Lab holds no reference run to compare against. A comparison
  against nothing is the "declaring an approach 'better' without a
  measurement" that `EXPERIMENTAL-CV.md` rules out. Offline corpus
  comparison is `scripts/cv_lab_benchmark.py`, and it is not this channel.

### 4.1 `latest`, `observed_min`, `observed_max` — placing a number without inventing a threshold

Three fields on every metric row. Non-`null` only for `rate` metrics: a
`count`'s running total has no "range this run", a `constant` has no range
by definition, and an `unaggregated` one has no meaningful anything.

`value` is the run's aggregate. `latest` is the most recent frame's value,
and `observed_min` / `observed_max` are the range **this run** has seen.

They exist because `sharpness_laplacian_var: 483.068` says nothing to
anybody, and the same number beside *"this run has seen 79 to 1309"* says
the frame is middling — from **this** camera in **this** room, rather than
from a constant somebody picked once. The standard reference for
variance-of-Laplacian is explicit that its threshold must be tuned per
dataset, and this Lab has one physical run to tune against, which is none.

**What a client may conclude from them:** where a value sits inside this
run. **What it may not:** anything about whether that is good. There is no
calibration behind these numbers and a UI that renders "Sharpness: Good"
from them has invented the threshold this contract refuses to invent. A
run that has seen four frames has a range of four frames.

---

## 5. Annotations and imagery

```json
{"count": 3, "count_unavailable_reason": null,
 "artifact": {"contract": "experimental_cv.preview/2026-08-29",
              "kind": "live_preview", "visual_kind": "detections",
              "description": "Every box the detector produced, ...",
              "treatment": "raw_ephemeral", "face_filter": "none",
              "persistence": "none",
              "derived_from": "one frame, transiently, in memory",
              "path": "/cv-lab/preview", "media_type": "image/png",
              "run_id": "a1b2c3d4e5f6-3", "max_age_s": 2.0,
              "poll_interval_s": 0.1, "max_edge_px": 320},
 "artifact_unavailable_reason": null}
```

`count` is `null` when the experiment produces no annotation count and a
**number** when it does, **including zero**. `0` is a real result meaning
"found nothing" and must not merge with "did not say".

### 5.1 `artifact` used to be `null`, always. Here is what changed.

The previous version of this section said `artifact` is *"always `null` in
this contract"*, and gave three conditions that would have to be met
first:

1. a redaction-state vocabulary shared with §5 of `IOS-to-Tower.md`;
2. an artifact fetch contract;
3. per-experiment declaration of whether the visual output contains source
   pixels — *"an edge map or a depth map is derived, a detection overlay is
   the original frame with boxes on it, and those are not the same privacy
   object."*

All three are met, and the third is met by removing the case rather than
by declaring it: **no CV Lab preview contains source pixels.** A detection
overlay is not the original frame with boxes on it; it is boxes drawn over
a Canny line drawing. See §5.3.

### 5.2 The artifact block

Present when the running experiment declares a `preview_kind` and this
Tower has previews on. `null` otherwise, with
`artifact_unavailable_reason` carrying the sentence — the two are mutually
exclusive and never both `null`.

| Field | Meaning |
|---|---|
| `contract` | `experimental_cv.preview/2026-08-29`. Versioned separately from the status document: a client may read the whole document and never fetch an image, which is exactly what a Release iOS build with no camera does. |
| `kind` | `live_preview`. What the artifact *is*, so a later kind is a new value rather than a client guessing from the media type. |
| `visual_kind` | How to READ the picture, never what it means: `edge_map`, `relative_depth`, `keypoints`, `detections`, `flow_tracks`, `redaction_regions`, `frame_quality`. A client must not infer semantics from it — a `relative_depth` preview is **not** metres, and §4's rules about provenance and units are unchanged by there being a picture. |
| `description` | The Tower's own sentence about what is drawn. Displayed verbatim. It is also where a client that renders nothing else learns the one thing it must not get wrong — the depth preview's says the values are not metres. |
| `treatment` | `raw_ephemeral`, always, on every preview. See §5.3. |
| `face_filter` | `none`, always. A **process** claim: no face detector runs on this path. It never says the result is safe. |
| `persistence` | `none`. Nothing reaches a disk and only the newest frame exists. |
| `derived_from` | `one frame, transiently, in memory`. |
| `path` | `/cv-lab/preview`. A path, not a URL — the Tower does not know what address it was reached on, and a client that resolved a base URL for this document can resolve this against the same one. |
| `media_type` | `image/png` for every kind except `relative_depth`, which is `image/jpeg`. |
| `run_id` | The run these previews belong to. Send it back; see §5.4. |
| `max_age_s` | Past this age the Tower refuses rather than serving. `2.0`. |
| `poll_interval_s` | What the Tower suggests. Advisory: it cannot make a phone poll at any rate and does not try. |
| `max_edge_px` | Longest side of the served image. |

`available[].preview_kind` carries the same vocabulary in the **catalog**,
so a client can say "this one has a live view" in the picker without
starting a run to find out. `null` there is a real answer — `baseline`
will never have a picture, deliberately, because it is the control every
other experiment's cost is measured against and drawing one would roughly
double what it costs.

### 5.3 Privacy — `raw_ephemeral`, and why that is the strict answer

`IOS-to-Tower.md` §5 defines three treatments and this Tower emits exactly
one of them:

- **`redacted`** — a redaction step ran. **Not true here.** No face
  detector runs on the preview path, and claiming otherwise would be the
  *"switch the Tower cannot honour"* that `VisualArtifact.swift` says is
  worse than no switch at all.
- **`raw_ephemeral`** — *"Untreated imagery. Permitted only for the live,
  in-memory view of what the wearer currently sees — never for anything
  persisted, and never for anything a cartridge stored and re-served
  later."* **Every clause is true here**, and it is the strict answer
  rather than a lenient one.
- **`unknown`** — the producer did not say. Withheld by iOS, correctly.

A fourth, gentler value was considered and rejected. `IOS-to-Tower.md` §5
says *"There is deliberately no `.probablySafe` and no lenient default"*,
and "it is only a Canny map" is precisely the argument a `.probablySafe`
would encode. It is also wrong: **an edge map of a face keeps the jawline,
the hairline and the frames of a pair of glasses, and a depth map keeps
the silhouette.** Derived is not unrecognisable.

**No preview is photographic.** Every overlay — keypoints, boxes, flow
arrows, the redaction rectangle — is drawn over an edge map derived from
the frame, never over the frame. The alternative was the real frame with
the display filter from `object_memory/imagery.py` applied and failing
closed; it was rejected because it needs vendored YuNet weights (so a
Tower without them shows a blank debug viewer), because a display filter
is not a redaction, and because an edge map is enough to tell a chair from
a doorway, which is all a box needs to be placed against. The
implementation is `tower/experiments/scene_structure`, and its docstring
carries the full argument.

**Obligations on a client**, all of which follow from `raw_ephemeral`:

- draw it live and nowhere else;
- never write it to disk, a URL cache, a photo library or a log;
- drop the bytes when the view goes away, when the run changes, and when
  the run pauses or stops;
- never re-serve it.

The Tower does its half structurally: `Cache-Control: no-store` on every
response, nothing written to disk, one image in memory at a time, and the
slot emptied the moment a run leaves `running`.

### 5.4 Fetching one

```
GET /cv-lab/preview?run_id=<the run you are watching>
    If-None-Match: <the ETag you last received>
```

**HTTP, not the socket, and that is load-bearing.** `ws.py` gives the
frame path and the result sender one shared lock, and every bulky thing in
this Tower is an HTTP route for that reason — `geometry.py` says so about
a megabyte of points and `observations.py` about a JPEG. A 5–40 KB image
several times a second on the frame socket would queue against
`frame_result`. It is also the shape that cannot build a backlog: a GET is
a pull, so a client that falls behind asks again and gets the **newest**
picture, and the ones it missed were dropped when they were replaced
rather than queued against its return. There is no per-client state on the
Tower at all.

Responses:

| Status | Meaning |
|---|---|
| `200` | The image. Headers carry `ETag`, `X-CV-Preview-Run`, `X-CV-Preview-Seq`, `X-CV-Preview-Kind`, `X-CV-Preview-Age` (seconds), `X-CV-Preview-Treatment` and `X-CV-Preview-Contract`, plus `Cache-Control: no-store`. |
| `304` | You already have this frame. No body, no encode on the Tower. |
| `404` | `experiment_has_no_visual_output`. Asking again will not help. |
| `409` | `preview_run_changed`. You named a run that is not current; `current_run_id` on the body says which one is. |
| `503` | `preview_disabled`, `no_preview_yet`, `preview_stale` or `preview_render_failed`. The Tower is willing and has nothing right now. |

The **reason value is on the body in every case** and a client should
switch on that rather than on the status code — the same rule
`observations.py` states for its own imagery routes.

`GET /cv-lab/preview/status` returns the artifact block without the bytes,
for a client that wants to know whether to draw a viewer at all. Same
split as `/object-memory/.../imagery` beside `/frame`.

**Staleness is answered three ways**, and the point of three is that no
one of them is trusted alone:

1. **Run identity.** Send `run_id`. Stop Edge Detection, start Depth, and
   a request still naming Edge's run is refused at the Tower rather than
   answered with a picture the phone would draw under Depth's name.
2. **An epoch.** Every stop, pause, failure and release bumps a counter,
   and a render that began before one and finished after it is discarded.
3. **Age.** Past `max_age_s` the Tower refuses. A phone showing a
   four-second-old edge map while its wearer turns their head is showing a
   lie about where they are looking.

The **descriptor** rides the status document and the **identity** rides
the bytes, deliberately. `result_seq` changes every frame; putting it in
the status document would make `revision_changed` — which the result
channel defines as *"news, not a heartbeat"* — fire on every poll of a Lab
that had merely gone on running.

### 5.5 `run.preview` — what the picture cost

A diagnostics block on the run document. It exists to answer one question
with evidence rather than opinion: **did adding a live view slow the CV
pipeline down?**

| Field | Meaning |
|---|---|
| `enabled` | Whether this Tower serves previews at all. |
| `live` | Whether a capture right now would be kept: previews on, the experiment has a picture, and the run is `running`. |
| `visual_kind` | The kind being captured, or `null`. |
| `frames_offered` | Frames the Lab told the preview about. |
| `captured` | …of which this many became the newest preview. |
| `skipped_by_throttle` | …and this many were never asked for, because the last capture was too recent. **This is the figure that says visualisation runs at its own rate.** |
| `empty_takes` | The experiment had nothing to hand over. Normally zero. |
| `replaced_unread` | Captures overwritten before anything rendered them. The intended behaviour, counted rather than hidden: `captured - replaced_unread` is roughly what the phone actually saw. |
| `encoded` | Encodes actually performed — far fewer than `captured`, and the gap **is** the design: nothing is encoded until somebody asks. |
| `encode_failures` | Renders that raised. The run is unaffected; see §5.6. |
| `served` | Responses carrying bytes. |
| `not_modified` | Answered `304`. A high figure means the phone is polling faster than the Tower produces, which costs a round trip and no encode. |
| `refused` | Responses carrying a reason instead of a picture. |
| `render_ms`, `render_ms_max` | Mean and worst encode, **on a worker thread**. Not part of `timings.processing_ms`. |
| `payload_bytes`, `payload_bytes_max`, `payload_bytes_last` | What a preview weighs. |
| `max_edge_px`, `min_interval_s`, `max_age_s` | The policy in force. |

Two numbers are deliberately kept apart. `timings.processing_ms` is what
the **experiment** costs and stays comparable against every figure
recorded before previews existed; `preview.render_ms` is what a
**picture** costs, on a different thread, at a different rate. Where an
experiment does pay something on the frame path to derive what will be
drawn, it appears as a **`preview` stage inside `timings.stage_ms`** —
visible, subtractable, and only on the frames the throttle allowed.

### 5.6 The picture may fail; the run may not

Rendering never raises. A failed encode produces a `503` with
`preview_render_failed`, increments `encode_failures`, and leaves the
experiment untouched — it has already returned its result and its numbers
are already on the wire. Capturing is wrapped for the same reason:
`ModuleContainer` treats anything that is not a `FrameProcessingError` as
a **terminal** module failure, so a bug in a picture would otherwise end
CV processing for the life of the process. A preview is a convenience, and
no convenience gets to end a run.

---

## 6. Timing and throughput

```json
"timings": {"processing_ms": 1.12, "processing_ms_max": 5.19,
            "stage_ms": {"blur": 0.80, "canny": 0.15, "decode": 0.16,
                         "summarize": 0.01},
            "observed_at": 1787810180.83, "time_basis": "tower-receipt"}
"throughput": {"processed_fps": 0.79, "offered_fps": 0.79,
               "capacity_fps": 892.83}
```

`processing_ms` is the **mean** over the run and `processing_ms_max` the
worst frame — the Tower measuring itself, the same quantity
`frame_result.processing_ms` carries per frame.

`stage_ms` is an **open map**: its keys are the experiment's own stage
names (`decode`, `blur`, `canny`, `summarize`, …) and a client must not
switch on them. Same for `runtime` in §3.3. Both are bounded — 16 stage
names and 8 runtime facts per run — because "open" must not mean
"unbounded" in a run that stays open for the life of the Tower.

`observed_at` is when the Tower last produced a result for this run.
**There is no capture timestamp anywhere on the wire** — `tower/frames.py`
carries no time field — so this is when the Tower saw a frame, never when
the glasses did. That is why `time_basis` is stated on every timing block.

**There is deliberately no end-to-end latency field.** It would be
computed across two clocks whose relationship is an open question in
`07-PLATFORM-CONSTRAINTS.md`, and a number derived from two unrelated
clocks is not a latency. iOS asked for none.

`processed_fps` and `offered_fps` are per second of run wall-clock.
`capacity_fps` is `1000 / processing_ms` — how fast the Lab could go if
frames never stopped arriving. Read together they say whether the Lab or
the link is the limit: the current sender forwards roughly one frame in
thirty, so `processed_fps` is normally bounded by what arrives, not by
what the Lab can do.

---

## 7. Debug, Release, and not claiming what cannot arrive

The iOS camera path is `#if DEBUG` — the DAT capture, the JPEG encode, the
frames→Tower bridge and both stream-lifecycle bridges. **A Release build
has no camera, sends no frame, and therefore receives no `frame_result`.**
That predates this work and is not fixed by it.

What this contract does about it:

1. **The read-only half is reachable in Release.** The capability
   declaration, the result channel and the status document are not gated
   on iOS and do not depend on frames. A Release build can enumerate
   experiments and read the Lab's state truthfully.
2. **The Tower reports whether frames are arriving** (`source`, §3.5). A
   Release build that showed "running" with no frames would be claiming
   something it cannot receive; `receiving_frames: false` and
   `frames_offered: 0` are what stop it.
3. **The Tower never claims a client's stream is running.** It reports
   what it observed. `lifecycle.state: running` means *the Lab is armed
   and processing whatever arrives*, not *frames are arriving*. Those are
   two facts and the document keeps them apart.

The rule for iOS: **`.running` may be shown as LIVE only when this build
is itself streaming AND `source.receiving_frames` is true.** Both halves
are needed. `receiving_frames` is Tower-wide (§3.5), so on a Tower with a
second phone attached it is `true` for a Release build that has no camera
at all — and the client's own streaming state is the half that catches
that. Otherwise show the run as armed and waiting for a stream, and in
Release say the build has no camera. Never render a Start control in a
configuration that has no `startCameraSession` to call.

---

## 8. Control messages

All are plain WebSocket JSON on the existing `/ws` socket. Every reply
carries the **whole status document**, so a client is never left guessing
what state it is now in.

### Client → Tower

| Type | Fields | Effect |
|---|---|---|
| `cv_lab_status` | `request_id?` | read the document |
| `cv_lab_start` | `experiment_id`, `request_id?` | select **and** arm. Replaces whatever was running |
| `cv_lab_pause` | `run_id?`, `request_id?` | stop processing; keep the experiment loaded |
| `cv_lab_resume` | `run_id?`, `request_id?` | resume processing |
| `cv_lab_stop` | `run_id?`, `request_id?` | end the run, release the experiment, keep the figures |

There is **no separate select message**. Selection without arming is a
state nobody needs on the wire — the phone holds the choice until Start,
and iOS's `run(_ experiment:)` is already one call.

`run_id`, when sent, is checked: a command naming a run that is no longer
current is refused with `stale_run` rather than applied to whichever run
is. Send it whenever the button was drawn against a run you have seen.

`request_id` is an opaque token of at most 64 characters, echoed back. A
client that sends none still gets a complete reply; a client that sends
one can match a reply to the button that was pressed, which matters when
two commands are in flight and one is refused.

### Tower → client

**`cv_lab_status`** — the document. `accepted_command` is present only
when this is the reply to a command, which is how a pushed status is told
from an answer.

```json
{"type": "cv_lab_status",
 "control_contract": "experimental_cv.control/2026-08-27",
 "contract": "experimental_cv.status/2026-08-27",
 "request_id": "req-1", "accepted_command": "cv_lab_start",
 "status": { ... }}
```

**`cv_lab_error`** — a refusal. **Every one of them means the request did
not take effect.** There is no partial application.

| `reason` | When | Extra fields |
|---|---|---|
| `malformed_request` | `experiment_id` or `run_id` missing or wrong-typed | `command` |
| `unknown_experiment` | no experiment with that id on this Tower | `available` (array of ids) |
| `experiment_unavailable` | the experiment exists but a module it needs is not installed on this Tower — `depth` needs `torch` and `timm`, `object_detection` needs `torch` and `torchvision`. `message` names the missing one | `experiment_id` |
| `lab_busy` | a start is already in flight (only a start can be; stop, pause and resume are immediate) | — |
| `invalid_state` | the command does not apply from the current state (resume when idle, pause when stopped) | — |
| `stale_run` | the `run_id` named is not the current one | `current_run_id` |
| `lab_unavailable` | this Tower runs no CV Lab, or its module failed. **Terminal** — treat as `.unsupported` | — |
| `internal_error` | the Tower failed while answering, and the request did not take effect. **Transient and retryable** — deliberately not `lab_unavailable`, because telling a person to give up on a working Tower is worse than telling them to try again | — |

Every `cv_lab_error` also carries `control_contract`, `message` (prose for
a person), and `status` (the document, unchanged) — including the
`lab_unavailable` refusal from a Tower with no Lab at all, which carries a
hollow document with the real contract identifiers in it.

**There is no `start_failed` refusal, and that is not an omission.** An arm
is asynchronous — the whole reason a start returns immediately — so by the
time a load fails, the command has already been answered `accepted`. A
second reply to a reply is not a thing this wire has. The outcome arrives
as **state**: `lifecycle.state` becomes `failed` with a `reason`, pushed on
the result channel or read with `cv_lab_status`. That is the shape iOS's
own `run(_:)` already has. **A client that sends commands and does not
also read status will never learn that a start failed.**

A `request_id` longer than 64 characters is **dropped, not refused**: the
command still applies and the reply simply carries no `request_id`. Keep
them short — matching a reply to a pressed button is the entire purpose.

**Refused, never queued.** `lab_busy` exists because the Lab holds one
experiment: queueing a second request behind the first would let two
clients each believe they chose what is running.

### The frame path during a switch

While `lifecycle.state` is not `running`, a frame is answered with
`frame_error` rather than silence:

| `reason` | State |
|---|---|
| `cv_lab_idle` | `idle` |
| `cv_lab_starting` | `starting` |
| `cv_lab_paused` | `paused` |
| `cv_lab_stopped` | `stopped` |
| `cv_lab_failed` | `failed` |
| `cv_lab_unavailable` | `unavailable` |

These sit alongside the transport's existing `invalid_frame`,
`frame_skipped` and `module_unavailable`. They are refusals, not failures:
the module stays ACTIVE and the next frame is accepted the moment the Lab
is running again. The `message` beside each one says what to send.

`cv_lab_failed` is also the answer to the frame on which the experiment
itself raised: that frame's `frame_error` carries the exception's
client-safe text and the run has already ended by the time it is sent.
Frames after it get the same reason with `lifecycle.reason` as the
message, until the next `cv_lab_start`.

`cv_lab_unavailable` is a **defensive default** rather than a state you
will normally see — when the Lab is `unavailable` the module behind it is
FAILED or UNLOADED, so the transport answers `module_unavailable` before
the Lab is reached.

**A refusal is not counted as a frame processing error.** The Tower's own
session summary counts these under `frames_rejected` (they are missing
from the measured numbers, which is what that field means) but **not**
under `frame_processing_errors` — a Lab paused for five minutes has not
failed hundreds of times.

---

## 9. `frame_result` provenance

Additive. Everything that was on `frame_result` before is still there,
unchanged, in the same place.

```json
{"type": "frame_result", "seq": 30, "processing_ms": 1.4,
 "result_value": 0.041, "result_label": "edge_density",
 "stage_ms": {...},
 "cv_lab": {
   "contract": "experimental_cv.frame_result/2026-08-27",
   "tower_instance_id": "2a5b04b1b77c",
   "run_id": "2a5b04b1b77c-2",
   "result_seq": 41,
   "experiment_id": "edge_detection",
   "experiment_name": "Edge detection",
   "provenance": "measured",
   "backend": "opencv",
   "device": null,
   "device_requested": "auto",
   "result_label": "edge_density",
   "processing_ms": 1.4,
   "tower_received_at": 1787810180.83,
   "time_basis": "tower-receipt"
 }}
```

`result_seq` is dense within the run, starting at 1. **The wire `seq` is
the phone's capture index and skips by design** — the current sender
forwards one frame in thirty — so it cannot be used to order results.

### The staleness rule, and why it is one line

> **Discard any `frame_result` whose `cv_lab.run_id` is not the run you
> are watching.**

The Tower makes this structural rather than checked: a new experiment is a
new run, and the old experiment is released **before** the new run id is
published, so there is no window in which a result computed by one
experiment can carry another's name. The client-side rule exists for the
case the Tower cannot cover — a reconnect to a **restarted** Tower, which
starts counting runs again. `tower_instance_id` is part of every run id
for that reason, so comparing `run_id` alone is sufficient.

---

## 10. Limits and bounds

| Bound | Value | Why |
|---|---|---|
| Metrics reported per run | **16** | the largest registered experiment (`optical_flow`) emits **14**, so the real headroom is two; this bounds a future one, and says how many it dropped |
| Metric names tracked per run | the experiment's own declared set — **14** at its largest | a name is filed only if the experiment declared it. There is no separate cap, because one would never fire |
| Stage names tracked per run | **16** | unlike a metric, a stage name is whatever the experiment passed to `StageTimer` with nothing declaring it in advance. The most any registered experiment uses is four |
| Unclassified metric names reported | **8** | a name list a producer controls must not grow without limit |
| `request_id` length | **64** characters | it is echoed onto the wire; longer is dropped, not refused |
| Echoed `experiment_id` / `run_id` / failure reason | **120** characters in a message | a remote party must not choose the size of a message this Tower sends |
| Status document size | bounded at **16 KB**; measured worst case **8 852 B** (`optical_flow`, 14 metrics + the 8-experiment catalog) | a fixed arity with no unbounded list. The bound is deliberately looser than the measurement, so that adding one honest field does not fail a test |
| Stream-idle threshold | **5.0 s** | ~4 missed frames at the sender's observed 0.8 fps |
| Arm timeout | **120 s** | the same bound and the same reason as the module container's load timeout: 119 MB of MiDaS weights does not fit a 10 s bound on any ordinary link |

**Everything a run accumulates is O(1) in frames.** A mean is a running
total and a count; a maximum is a maximum. There is no frame list, no
metric history and no sample buffer — `handoff.md` 9.3 says a
`stream_stop` may never arrive, so "for the length of a run" means "for as
long as the Tower is up".

**Cost on the wire.** The document is re-sent when its **revision**
changes or every 2 s otherwise. The revision deliberately excludes
`run.elapsed_s` and the two throughput rates derived from it — they
advance with the clock and with nothing else, and hashing them made
`revision_changed` fire on every poll for a Lab that had seen no frame at
all. So a subscriber sees a document per processed frame, plus a 2 s
heartbeat: at the sender's observed ~0.8 fps that is roughly **1.3
documents per second, about 11 KB/s**, against ~16 KB/s for the frame
stream itself. A Lab that is idle, paused or stopped costs the 2 s
heartbeat and nothing more. Unsubscribe when the CV Lab screen is not
visible.

---

## 11. Known limitations

1. **One Lab, shared by every connection.** The module container holds one
   module and the Lab holds one experiment. Two phones streaming to one
   Tower feed the *same* run, and for a stateful experiment
   (`optical_flow`) that means frames from two sources are diffed against
   each other. `optical_flow` has a 2 s staleness guard that catches the
   common case; the residual needs a session-boundary hook on the module
   contract, which is blocked V1.0/V1.1 work. `source.clients_connected`
   is reported so the condition is at least visible.
2. **Last start wins.** Two clients starting different experiments in
   sequence both succeed; the second replaces the first, and both see it
   in the pushed status. There is no ownership model, because a bench with
   one slot and two operators has a social problem, not a protocol one.
3. **One failure is still terminal.** Until 2026-09-06 there were three:
   a failed startup default and an experiment raising on a frame both
   reached the module's terminal FAILED state (§3.1 says what they do
   now). What remains is a startup load that overruns the container's
   120 s bound: the load is abandoned on its thread, the module is
   marked FAILED and the Lab reports `unavailable` until the Tower
   restarts. That is the shared module lifecycle's bound, not the Lab's,
   and a warm load on the reference host is two orders of magnitude
   inside it. Every registered experiment still routes its recoverable
   per-frame failures through `FrameProcessingError`, which keeps the run
   going; anything else now ends the run, not the process.
6. **Inference runs on the event loop.** A 40 ms `object_detection` frame
   holds every socket for 40 ms, including the reply to a command sent
   during it. It is bounded (one frame at a time per connection; the
   phone drops rather than queues) and it is why the torch thread budget
   matters, but it is not a queue you can drain from the client.
4. **No artifact, no baseline, no direction.** See §4 and §5. All three
   are `null` with a stated reason rather than omitted.
5. **No cancellation of an in-flight arm from the client's point of
   view**, beyond `cv_lab_stop`, which does cancel it. There is no way to
   ask "how far through the download are you"; torch.hub does not report
   it.

---

## 12. Where the code is

| Concern | File |
|---|---|
| Identifiers, states, refusal reasons, bounds | `tower/cv_lab/contracts.py` |
| The catalog | `tower/cv_lab/catalog.py` |
| One run's identity and measurements | `tower/cv_lab/run.py` |
| Lifecycle, selection, the frame path, the document | `tower/cv_lab/lab.py` |
| The one loader thread every arm runs on | `tower/cv_lab/loader.py` |
| The torch thread budget and its measured defaults | `tower/experiments/depth.py` (`resolve_torch_threads`) |
| The switch soak | `scripts/cv_lab_switch_soak.py` |
| The module that holds the Lab | `tower/modules/experimental_cv.py` |
| Experiment registration and metadata | `tower/experiments/__init__.py` |
| Control messages | `tower/routes/cv_lab_ws.py` |
| `GET /cv-lab` | `tower/routes/cv_lab.py` |
| The result-channel producer | `tower/results/experimental_cv.py` |
| The capability declaration | `tower/results/registry.py` |
| `frame_result` provenance on the wire | `tower/routes/ws.py` |

Adding an experiment: `guidelines/docs/modules/EXPERIMENTAL-CV.md`, "Adding
a new experiment".

---

## 13. Changelog

### 2026-09-06 — additive; no identifier moved

- `run.arm_ms` (§3.3) and `run.runtime.torch_threads` on the two
  model-backed experiments.
- `process` beside `status` on `GET /cv-lab` only (§2.2).
- `failed` now also follows an experiment raising on a frame, and a
  startup default that could not arm; both are recoverable with
  `cv_lab_start` (§3.1, §8, §11). Nothing changed shape; a client that
  already rendered `failed` as "another start may be sent" was right all
  along and is now right in every case.
- Every arm runs on one reusable loader thread per Lab. Invisible on the
  wire; it is why a Tower that switches experiments for an afternoon
  holds a flat thread count instead of gaining nineteen per switch.

### `experimental_cv.status/2026-08-27`, `.control/2026-08-27`, `.frame_result/2026-08-27`

First offer. Before this, `experimental_cv` was listed in the declaration's
`not_offered` with the reason *"results already reach the client on
`frame_result`; a typed contract awaits the experiment-registry and
provenance work described in IOS-to-Tower.md 2.1–2.3"*. That is the work
this contract is.
