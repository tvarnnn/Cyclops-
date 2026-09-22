# Module Concept — Scene Understanding / Environmental Intelligence

## Status

**CURRENTLY IMPLEMENTED** as of 2026-08-22 (`tower/scene/`), re-architected
2026-09-07 on measured evidence
(`docs/superpowers/research/2026-09-07-scene-understanding-architecture.md`).

| Part | Status |
|---|---|
| Capability | **PRODUCT-MANAGED** — offered when the `[ml]` extra imports (`TOWER_SCENE_UNDERSTANDING` unset = auto) |
| Activation | **STREAM AND WATCHER** — runs while a phone streams AND a client subscribes to the live scene; last of either out stops it and releases the models |
| Object detection | **CURRENTLY IMPLEMENTED** — RT-DETRv2-R18 on CUDA (mAP50 0.669 on 700 labelled images), SSDLite320 on CPU (0.367) |
| Anonymous tracking, counts from tracks | **CURRENTLY IMPLEMENTED** — cardinality-first Hungarian, kept 1.0 s, **counted while seen within 0.5 s** |
| Camera-relative positions, incl. people as side counts | **CURRENTLY IMPLEMENTED** — bands 0.35/0.65 with hysteresis |
| Apparent size of people (large/medium/small), partial figures at the bottom edge | **CURRENTLY IMPLEMENTED** — sizes in the picture, never distances |
| Coarse facing ("appears to be facing your direction") | **EXPERIMENTAL, ON BY DEFAULT** — a face detector on each tracked person's box, two states only (toward / not established), validated on COCO stills at 0.83 precision, never on this camera |
| World-anchored positions | **BLOCKED** — no live world pose exists. Camera-relative is the honest alternative and is what ships |
| Depth-dependent relationships (`in_front_of`, `on`, `inside`) | **REFUSED**, each with the evidence it would need |
| Validation on real people | **BLOCKED** — no bystander in 45,594 corpus frames (2026-09-07 audit); COCO stills stand in and the wire says so |

Plan: `docs/superpowers/plans/2026-08-22-scene-understanding-v1.md`.
Report: `reports/2026-08-22-scene-understanding-v1-report.md`.

## Goal

Maintain a structured understanding of what exists around the wearer
**right now**, so questions like these can be answered from perception
rather than by asking a model to re-read a raw frame each time:

- How many people are in this room?
- Where is the desk? Where is a chair?
- How many people appear to be facing my direction?

## The distinction that settles the design

**Scene Understanding is a live state. Environmental Memory is a
history.** One answers "what is around me now"; the other answers "what
did I encounter, and when".

From that, two consequences that are not negotiable:

- **Nothing is persisted.** No store, no journal, no imagery. A cartridge
  answering "how many people are in this room" has no reason to write to
  disk, and writing would import all of Environmental Memory's retention,
  purge and privacy surface for no gain. A test asserts that no write
  primitive is ever called. If a durable record is wanted, it belongs in
  Environmental Memory and should be built there.
- **There is no query CLI.** With nothing persisted there would be
  nothing for a separate process to read, so the run that observes the
  frames answers the questions.

## Counting uses tracking, and that is the point

Summing detections is wrong in two directions at once: a detector that
misses someone on one frame in five reports a count flickering between 2
and 3 while nothing in the room changed, and one that fires twice on a
person reports two people.

Counts therefore come from **confirmed tracks** — associated across
frames, with a minimum hit streak before they count and a maximum miss
budget before they expire. Measured, with a correct answer of 2
throughout:

| Detector dropout | Modal count | Counts seen | Fraction correct |
|---|---|---|---|
| 0% | 2 | [2] | 1.000 |
| 10% | 2 | [2] | 1.000 |
| 20% | 2 | [2] | 1.000 |
| 40% | 2 | [1, 2] | **0.965** |
| 60% | 2 | [0, 1, 2] | **0.783** |

A count taken from raw detections would follow the dropout column
exactly.

The 40% row was **0.974 before the confirmation fix in §8.1**, 0.939
after it, and 0.965 since the miss budget was retuned; the 60% row is
new and is where that retune shows, at 0.783 against **0.252** on the
old constant. Requiring a consecutive streak means a track dropped at
extreme dropout takes longer to re-confirm, and it is what stops a
detection present one frame in six from becoming a permanent phantom
person. A count that is occasionally conservative under a detector
losing 40% of frames is a better failure than one that is permanently
wrong under a reflection.

**The miss budget is a duration, and it was written as a frame count.**
`max_misses = 5` was justified as "roughly 1.5 seconds of absence"
against an assumed ~3.3 fps. At the measured 12.0 fps it bought 0.42 s,
so a person occluded for half a second was dropped, returned with a new
`track_id`, and was **counted as somebody new** — the exact failure
counting-from-tracks exists to prevent. It is now derived:
`MAX_ABSENCE_S = 1.0` divided by the measured frame interval, which is
12 frames. The sweep behind that number, on 9,145 real corpus frames,
is in `docs/superpowers/research/2026-08-26-tracker-retune.md`. The other
two thresholds were swept in the same pass and both survived, with
`min_iou = 0.25` now derived from the measured 1st percentile of
same-object consecutive-frame IoU and `min_hits = 3` from a two-sided
sweep that rejects 4 and 2.

The cost is named rather than hidden: a track whose object has genuinely
gone stays confirmed for up to one second, so the count can be one too
high for that long. That is a real claim about the room, and 1.0 s is
where it was put because count stability at 18 and 24 frames is
identical to 12 — a longer window buys nothing measurable and asserts
more.


**Association is by IoU only, never appearance.** Matching by how
something looks is the first step toward recognising it again. A
`track_id` means "the same blob one frame later", restarts at 1 every
session, and a person who leaves and returns is deliberately a **new
track**.

## Orientation: a face, visible, in a person's box

*"How many people appear to be facing my direction?"* needs evidence. A
person box carries none — inferring facing from box shape would be
exactly the weak evidence the brief forbids.

**What shipped first, and why it is gone.** The 2026-08-22 design inferred
facing from which COCO keypoints a `keypointrcnn_resnet50_fpn` reported as
visible: both eyes and an ear meant "toward", both ears and no eye meant
"away", one ear meant "profile". On 2026-09-07 that rule was checked
against 966 human-labelled persons in COCO val2017, with the ground truth
derived from the annotators' own visibility flags. It called **220 of 228
true-profile people "toward"**: the keypoint model reports a confident
score and a plausible coordinate for an occluded eye, so a score threshold
does not track human visibility exactly where it matters. `toward`
precision was 0.56–0.62 across every threshold, and `away` / `profile`
were 0.02–0.34 precise — wrong more often than right. It also cost 43 ms
on CUDA and 956 ms on CPU per frame.

**What ships now.** `cv2.FaceDetectorYN` — the vendored YuNet model World
Builder already uses for redaction — runs on the upper part of each
tracked person's box. A face found with a score ≥ 0.9 means the front of
that head is toward the camera. On the same 966 persons that is **0.83
precision / 0.64 recall** for `toward` (0.6: 0.63 / 0.94; 0.8: 0.69 /
0.89); a landmark-yaw refinement and an AND with the keypoint model added
nothing over raising the threshold. It costs **~9 ms per face on CPU**,
no VRAM.

**Two states only.** `toward_wearer`, or `unknown` meaning *not
established* — which covers facing away, side-on, too small to tell, and
never measured alike. "Facing away" and "side-on" are never produced,
because nothing measured on this platform produces them with usable
precision, and the wire says so (`facing_states_withheld_reason`).

**Voting and ageing.** One frame's detection is one vote; a track is
reported `toward` only when two of its last three estimates agree, at the
tracker's ~250 ms cadence — about half a second of evidence to make the
claim and about half a second to drop it. Every estimate carries its age
and expires to `unknown` after 6 s. Confidence never exceeds MEDIUM.

**Status: EXPERIMENTAL.** COCO stills are third-party photographs — front
lit, in focus, and 47% of the people in them face the camera because
photographers point cameras at faces. That base rate is an upper bound
for a glasses camera in a room where most people are not looking at the
wearer, and no person has been measured through these glasses. The wire
carries `orientation_status: "experimental"` and `orientation_validation`
so a client can say so.

### It is never gaze

`07-PLATFORM-CONSTRAINTS.md` Limitation 8: the camera cannot establish
that anyone looked at, noticed or read anything, and there is no eye
tracking on this hardware. Head orientation is not eye direction — a
person squarely facing the wearer may be reading over their shoulder.

The state is `toward_wearer`, the property is `appears_facing_wearer`,
confidence never reaches HIGH, and a boundary test bans the identifiers
`looking_at`, `gaze_direction`, `is_looking`, `face_id` and `person_id`
across **every** cartridge.

**Asking the question with orientation disabled returns a refusal, not
zero.** Zero would be an observation gap reported as an observation of
absence — Core Principle 3's exact error, and the one most likely to be
mistaken for data because zero looks like an answer.

## Relationships: what is asserted, and what is refused

Everything is **camera-relative**, and every relation says so. World
Builder produces poses offline, after a session, so there is no live pose
to anchor to. No world ids are invented.

**Asserted:**

| Relationship | Basis |
|---|---|
| `left_of` / `right_of` | Box centroid x, with a minimum separation so a one-pixel difference asserts nothing |
| `higher_in_view` | Box centroid y. Named for the **image**, not the room: something further away sits higher in frame without being higher in the room |

**Refused, each with the evidence it would need:**

| Refused | Why, and what would settle it |
|---|---|
| `in_front_of` / `behind` | **Measured and still refused** (2026-08-26, 9,199 real frames — `docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md`). Ordering two boxes by MiDaS relative inverse depth reverses on **3.8%** of consecutive-frame transitions over 2,700 object pairs, and separation predicts it strongly (15.7% below 0.02 separation, 0.0% above 0.40) — but only while the scene is still. At matched separation the flip rate goes from **0.0% (n=124) to 11.5% (n=52)** between the most static frames and the top motion decile, and the corpus's 99th-percentile inter-frame box motion is 56 px, so it contains no walking. Cost is not the obstacle: **5.73 ms CUDA / 18.29 ms CPU** against an 83.4 ms interval. The earlier 6–8% flicker figure was about right in magnitude (4.8% here) but the ordering conclusion drawn from it did not follow. To settle it: corpus footage with sustained wearer locomotion |
| `on` | Needs support-surface reasoning and depth. Box containment is not it — a laptop *in front of* a desk overlaps its box identically to one *on* it |
| `inside` | Same: 2-D containment cannot distinguish it |
| `near` | Image proximity is not world proximity. Two things at opposite ends of a room can be adjacent in a frame |
| `nearer_than_same_class` | **Shipped, then withdrawn.** Box area within one class looked like safe evidence for relative distance; an adversarial review produced two chairs at the *same* distance, one face-on and one edge-on, whose areas differ 2.5x — a wrong relation, not a weak one. Nothing in a 2-D box separates shape from distance |

`why_not(relationship)` returns those reasons, so the next cartridge does
not re-derive them from scratch. **A relationship nobody can support is
worse than a missing one**, because a consumer cannot tell a wrong answer
from a right one.

## Privacy

The strongest posture of any cartridge so far, and it is free here
because the purpose is a live answer:

- **Nothing persisted.** No store, no imagery, no history.
- **No identity.** Anonymous, session-scoped track ids, meaningless
  across processes. No appearance matching, so no re-identification.
- **No face processing.** Keypoints locate eyes and ears as anonymous
  landmarks; they produce no descriptor and support no matching.

  This bullet used to add "no face detector exists on this platform
  anyway", and that justification was **wrong**. `cv2.FaceDetectorYN` is
  compiled into our OpenCV and needed only a 227 KB weights file, which
  is now vendored at `models/face_detection_yunet_2023mar.onnx` and used
  by World Builder to redact faces before a keyframe is written. The
  original search was scoped to `cv2/` and missed it; the same error was
  corrected in `reports/2026-08-22-cartridge-run-report.md` on 2026-08-23
  and missed here.

  **The posture is unchanged and does not depend on that claim.** This
  cartridge does no face processing because it has no need to, not
  because it could not. A capability being available is exactly when
  "we don't do this" has to be a decision rather than a limitation.
- **Raw pixels are ephemeral**, held only for the frame being processed.

## Relationship to other cartridges

- **Environmental Memory** — the history to this cartridge's present. If
  a durable record of "what was in this room" is wanted, it belongs
  there. Do not add a store here.
- **Experimental CV Lab** — measured the detector this cartridge uses
  (35.3 ms, and notably resolution-independent), which is exactly what
  the promotion path is for. It is **not imported**: the Lab's
  `ExperimentResult` cannot carry a box, and the two want different
  things from the same weights.
- **World Builder** — provides no live pose, so no anchoring today. The
  contract for a future upgrade is already written in
  `CARTRIDGE-GROUNDWORK.md` §4 and is not pre-empted here.
- **Object Memory** — tracks *objects over time*; this tracks them
  *across frames*. Do not merge: one needs identity across sessions, and
  this must never have it.

## Limitations

- **Detection accuracy on real people is unvalidated on this camera.**
  There is no bystander in any of the 45,594 corpus frames, so accuracy
  was measured on 700 human-labelled COCO images (per-image people count
  exact on 73% of them, mean error 0.48) and the detector's behaviour
  through these glasses is not measured.
- **The wearer's own body is in frame in most captures.** A person box cut
  off by the bottom edge with no head region is reported apart
  (`partial_bottom_edge`) rather than counted; a box that includes an arm
  raised into the head region is still counted, and that is the
  remaining source of "1 person" in an empty room.
- **Camera-relative only.** "Left of" means left in the current view and
  means something else the moment the wearer turns.
- **No depth**, hence the refusals above.
- **Orientation is coarse, experimental, and two-valued**, and is not gaze.
- **A count is of what is *visible*.** An occluded or out-of-frame person
  is not counted, and absence of a detection is never evidence of
  absence.
