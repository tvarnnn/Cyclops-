# Scene Understanding v1: the architecture, and the measurements behind it

**Date:** 2026-09-07. **Lane:** `feature/scene-understanding-v1`, worktree
`C:\Users\tvllo\Projects\Glasses-worktrees\scene-understanding-v1`, base
`6beaf57`. **Host:** RTX 5070 12 GB (driver 596.21), 20-core CPU, 32 GB,
`torch 2.13.0+cu132`, `torchvision 0.28.0`, `transformers 5.16.1`,
`opencv-python-headless 5.0.0`.

This document is the evidence for every constant and every model the
cartridge now ships with. Everything measured on this host is reproducible
from the scripts named; everything that is not is marked as such. All
scratch material lives under
`C:\Users\tvllo\Projects\Glasses-scratch\scene-understanding-v1\`
(`detector/`, `tracker/`, `orientation/`, `corpus-audit/`, `coco/`,
`results/`, `corpus3/`). Nothing under `tower/data/` was modified; the
replay corpus `corpus3/` is a copy of three captures.

## 0. What existed, and what was wrong with it

On 2026-09-06 the cartridge was a complete, carefully reasoned pipeline that
nobody could switch on without an environment variable, and whose
orientation stage was measurably wrong:

| Part | Shipped | Finding |
|---|---|---|
| Capability | `TOWER_SCENE_UNDERSTANDING` default off | product required an env var; unset read as "off" |
| Activation | any `stream_start` started it | a people detector ran behind World Builder's camera; after HTTP Stop→Start the session was owned by nobody (finding 15, strict xfail) |
| Detector | SSDLite320, CPU default | mAP50 0.367 on labelled images; chair AP50 0.199; blind under 2% of frame |
| Tracker | Kuhn max-cardinality IoU, kept 1.0 s, counted 1.0 s | departure lag dominated count error on labelled sequences |
| Orientation | KeypointRCNN keypoint visibility, off by default | called 220 of 228 true-profile people "toward" (96.5%); away/profile precision 0.02–0.34 |
| Position | left <0.45, centre, right >0.55 | 4.7° centre band in a 44.7° field |
| Wire | no position for people; no size; person torso counted as a person | "is there a person on my left" unanswerable |
| Real data | 45,594 corpus frames | no bystander in any of them (audit below) |

## 1. Real data audit (`corpus-audit/`)

All 97 captures were contact-sheeted and inspected (by an AI agent, not a
human). Findings: **no bystander appears in any frame** — the only human
figure in 45,594 frames is the wearer's own bathroom-mirror reflection
(`0fc400bb…/00002234.jpg`). The wearer's own hands, arms, lap and legs are
in frame in most desk captures. Object presence across captures: laptop 38,
tv/monitor 33, cell phone 33, keyboard 30, mouse 20, chair 13, bottle 8, bed
6, dining table 5, cup 4, couch 3, book 0 (of the 85-frame labelled fixture).
Intrinsics (`world_builder/intrinsics/360x640.json`, fx 438 px) give
**HFOV 44.7°, VFOV 72.3°**. Head-motion proxy over 6 captures: median
inter-frame phase shift ~1 px full-res, p95 3–20 px, occasional bursts to
70 px.

Consequence: **people counting and orientation cannot be validated on this
camera from the corpus.** They were validated on COCO val2017 (human
labels) and the report says so everywhere the numbers appear.

## 2. Detector (`detector/`, `bench_detectors.py`)

Labelled set: 700 COCO val2017 images, indoor-biased, resized to a 640 px
long side (this camera's), scored on the 13 classes with crowd regions
ignored. Latency on 300 real corpus frames (360×640), CUDA, quiet GPU
(`*quiet*` results). All candidates are importable without new packages
(torchvision or `transformers`); ultralytics/YOLO not evaluated (AGPL, not
installed); DEIMv2 excluded (custom licence, broken conversion).

| model | mAP50 | mAP50-95 | person AP50 | chair | laptop | tv | CUDA ms med / p95 | VRAM res | licence |
|---|---|---|---|---|---|---|---|---|---|
| ssdlite320 (old) | 0.367 | 0.229 | 0.604 | 0.199 | 0.553 | 0.560 | 29.5 / 31.4 | 56 MB | BSD |
| frcnn_mbv3_320 | 0.406 | 0.244 | 0.602 | 0.233 | 0.619 | 0.602 | 16.7 / 37* | 206 | BSD |
| frcnn_r50_v2 @640 | 0.664 | 0.462 | 0.854 | 0.533 | 0.813 | 0.782 | 25.7 / 27.3 | 628 | BSD |
| **rtdetr_v2_r18** (fp32) | **0.669** | 0.497 | **0.856** | 0.540 | 0.833 | 0.824 | **17.3 / 19.2** | 298 | Apache-2.0 |
| rtdetr_v2_r34 | 0.693 | 0.517 | 0.874 | 0.576 | 0.863 | 0.847 | 54* | 344 | Apache-2.0 |
| lwdetr_small (fp16) | 0.689 | 0.510 | 0.872 | 0.606 | 0.847 | 0.851 | 18.2 / 19.7 | 138 | Apache (community conversion) |
| dfine_s_obj2coco (fp16) | 0.702 | 0.536 | 0.876 | 0.596 | 0.877 | 0.891 | 28.3 / 30.2 (fp32: 121*) | 152 | Apache-2.0 |
| dfine_m_obj2coco (fp16) | 0.737 | 0.569 | 0.895 | 0.640 | 0.904 | 0.888 | 35.4 / 38.4 | 210 | Apache-2.0 |

`*` measured under GPU contention (the first batch overlapped other
research jobs); quiet re-measurements are the unstarred numbers.

CPU (60 frames): ssdlite320 43.5 ms; frcnn_mbv3_320 61; dfine_n 66;
rtdetr_v2_r18 226; dfine_s 148.

**Choice: RT-DETRv2-R18 in fp32 on CUDA; SSDLite320 on CPU.** Nearly twice
the accuracy of the baseline at 60% of its latency, in plain fp32, from the
authors' own hub organisation, through a package the repository already
carries. D-FINE-S is 3 points better but only at speed under an fp16
autocast path (121 ms in fp32) and its hub checkpoints carry a documented
conversion caveat (transformers issue #40253); it is selectable
(`TOWER_SCENE_DETECTOR=dfine_s|dfine_m`). LW-DETR's weights are a
community conversion whose provenance was not verified.

**Threshold: 0.5 for hub detectors, 0.4 for SSDLite.** RT-DETRv2-R18 per-
image person count (all labelled persons): exact 63.9% / MAE 1.04 at 0.4;
**73.3% / 0.48 at 0.5**; 73.1% / 0.55 at 0.6. Micro precision/recall over
13 classes: 0.553/0.698 at 0.4, 0.698/0.614 at 0.5. Recall by GT area at
0.5: <1% 0.40, 1–2% 0.64, 2–5% 0.74, 5–20% 0.81, >20% 0.88 (SSDLite at 0.4:
0.00, 0.04, 0.36, 0.62, 0.81).

## 3. Tracker (`tracker/`, `bench_trackers.py`, `fresh_count_experiment.py`, `bonus_hungarian_experiment.py`)

Labelled fixture: 40 COCO scenes with 2–5 people, each turned into a
120-frame 9:16 sequence at 12 fps by a random-walk pan/zoom with
head-turn bursts (median speed 2% of width per frame, p99 8%), motion blur
on fast frames, GT identity known; detections from two real detectors
(ssdlite320, fasterrcnn_v2); five conditions (base, 20%/40% dropout, 0.5 s
and 1.5 s occlusions); 10,800 frames.

Pooled over five conditions, fasterrcnn_v2 detections:

| tracker | id consistency | id switches | ids per person | phantoms | count exact | count MAE | flicker |
|---|---|---|---|---|---|---|---|
| Kuhn max-cardinality (old) | 0.522 | 5,488 | 3.35 | 413 | 0.315 | 1.457 | 0.080 |
| Hungarian max-weight | 0.587 | 1,739 | 3.66 | 628 | 0.325 | 1.401 | 0.092 |
| **Hungarian, cardinality first (1+IoU)** | 0.577 | **2,018** | 3.62 | 554 | 0.314 | 1.423 | 0.089 |
| SORT (Kalman) Hungarian | 0.609 | 922 | — | 408 | 0.342 | 1.280 | 0.093 |
| ByteTrack-style | 0.613 | 984 | — | 426 | 0.311 | 1.410 | 0.096 |
| OC-SORT-lite | 0.559 | 1,178 | — | 303 | 0.330 | 1.349 | 0.093 |

Count window (old tracker, pooled): kept 1.0 s / counted 1.0 s → exact
0.315, MAE 1.457, flicker 0.080; **kept 1.0 s / counted 0.5 s → 0.370,
1.187, 0.117**; counted 0.25 s → 0.415, 0.987, 0.184; kept 0.5 s → 0.401,
1.037, 0.097 (but ids per person 3.35 → 3.74).

**Choices:** cardinality-first Hungarian (keeps the no-starvation
guarantee the existing adversarial tests pin, 63% fewer switches than
Kuhn, equal counts); no Kalman (fewer switches but more phantoms and 3×
the cost, no count gain); tracks kept 1.0 s (corpus-derived, unchanged)
and **counted only while seen within 0.5 s**. Absolute count accuracy on
this fixture is low for every tracker because GT counts persons at ≥30%
visibility that no detector sees; the relative comparisons are what the
fixture is for. Real footage (unlabelled, 5 captures × 400 frames): the
new tracker costs 9–37 µs per frame.

## 4. Orientation (`orientation/`, `bench_orientation.py`)

Ground truth: 966 COCO val2017 persons (≥1% of image, not crowd) labelled
from the annotators' own keypoint visibility flags: toward 457, profile
228, away 36, unknown 245. Images resized to a 640 px long side.

| method | toward precision | toward recall | away/profile precision |
|---|---|---|---|
| KeypointRCNN + shipped rule (t=3.0) | 0.579 | 0.989 | 0.02–0.15 |
| same, strictest sweep (t=5.0, nose) | 0.616 | 0.987 | 0.08–0.34 |
| YuNet face in person box, t=0.6 | 0.630 | 0.941 | — |
| YuNet t=0.8 | 0.690 | 0.888 | — |
| **YuNet t=0.9** | **0.830** | **0.641** | — |
| YuNet + landmark yaw ≤0.15 | 0.853 | 0.431 | — |
| KeypointRCNN AND YuNet t=0.8 | 0.703 | 0.880 | — |

The keypoint rule called 220 of 228 true-profile people "toward": the
model reports a confident score for an occluded eye. Cost: KeypointRCNN
48 ms CUDA / 1,112 ms CPU per frame, 754 MB VRAM; YuNet 9.3 ms per face
on CPU, no VRAM.

**Choice:** YuNet on the head region of each tracked person's box, score
≥0.9, **two states only** (toward / not established), 2-of-3 temporal vote
at the ~250 ms cadence, 6 s expiry, confidence capped at MEDIUM, status
**EXPERIMENTAL**. The COCO base rate (47% of people face the camera) is an
upper bound for a glasses camera; no person has been measured through
these glasses.

## 5. Position and size (`corpus-audit/fov_analysis.json`)

HFOV 44.7°. Bands: left <0.35, right >0.65 (centre 14°), hysteresis 0.03
(11 px, above the 4 px median inter-frame motion). A standing adult
(0.45 m shoulders) is 0.27 of the frame wide at 2 m, so the old 0.45/0.55
band could not hold one. Apparent size buckets for people: box height
≥0.6 of frame "large", <0.3 "small" — image sizes, never distances (a
1.7 m adult projects to 0.58 at 2 m, 0.39 at 3 m; a seated person breaks
it). Depth relations remain refused (2026-08-26 MiDaS measurement stands).

Bottom-edge rule: a person box with bottom ≥0.97 h and top ≥0.45 h (no
head region) is reported as `partial_bottom_edge`, not counted. From a
head-worn camera that is the wearer's own body in the overwhelming
majority of corpus frames; the wire says it can also be someone's legs.

## 6. Lifecycle (`tower/scene/live.py`, `tests/test_scene_activation.py`)

A session runs while **a stream is open AND a watcher (live-result
subscription) exists**, or while an operator holds it via `POST
/scene/start`. Last watcher out → stop, release models. Last stream out →
stop, whoever started it (closes finding 15). No demand event resumes a
Pause. `lifecycle.demand` publishes the counts. The phone subscribes on
the Scene screen's appear and unsubscribes on disappear (iOS change, not
compiled on this host).

Capability: `TOWER_SCENE_UNDERSTANDING` is `auto` when unset — offered
when the `[ml]` extra imports; `on`/`off` remain. Device `auto`.

## 7. Measured performance of the new pipeline

Baseline (old pipeline, quiet, `results/baseline_quiet_*.json`, 600 real
frames paced at 12 fps): CUDA 11.9 fps observed, 0.5% skipped, 16 CPU
cores (OpenMP spin at torch's default thread count), RSS +623 MB (CUDA
context), load 2.4 s; CPU 12.0 fps, 0 skipped, 15.7 cores, RSS +20 MB.

New pipeline, replay smoke (`results/replay_smoke/`, 180 real frames,
CUDA, RT-DETRv2-R18 + YuNet): decode 0.44 ms, **detector 14.9 ms median /
16.2 p95**, orientation 4.3 ms per call (median) / 9.0 p95, **observe
15.4 ms median / 23.8 p95**, 49.6 fps unpaced, load 5.2 s, VRAM 46 MB
reserved after the run (peak 217 MB allocated), 16.3 CPU cores (same
OpenMP spin as the baseline; `TOWER_SCENE_TORCH_THREADS` still applies).

Full-corpus replay and soak: see the handoff
(`docs/agent-handoffs/SCENE-UNDERSTANDING-V1-RESEARCH-IMPLEMENTATION.md`),
which carries the final numbers.

## 8. What was rejected, and why

- **YOLO (ultralytics):** AGPL, not installed; not evaluated.
- **D-FINE as default:** fp16-only speed, checkpoint caveat.
- **Kalman / ByteTrack / OC-SORT:** more phantoms, no count gain, 3× cost.
- **KeypointRCNN orientation:** measurably wrong on profiles.
- **Head-pose regressors (6DRepNet, WHENet, JointBDOE):** need new weights
  or packages; noted for a later lane.
- **Metric distance / depth ordering:** unchanged refusal.
- **Per-entity rows / track handles on the wire:** unchanged refusal; side
  counts and size buckets answer the product questions without a handle.
- **A subprocess worker:** the in-process reused worker thread is
  sufficient; the models release on stop.
