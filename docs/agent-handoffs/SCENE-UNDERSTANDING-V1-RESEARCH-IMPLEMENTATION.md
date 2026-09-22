# Scene Understanding V1 — research, implementation and handoff

**Date:** 2026-09-07. **Branch:** `feature/scene-understanding-v1`.
**Worktree:** `C:\Users\tvllo\Projects\Glasses-worktrees\scene-understanding-v1`.
**Base:** `6beaf57` (`integration/wb-cv-ios-validation-v1`, the common base
of the concurrent Document Memory and Object Memory runtime lanes).
**Final HEAD:** see §12.
**Evidence:** `tower/docs/superpowers/research/2026-09-07-scene-understanding-architecture.md`
and `C:\Users\tvllo\Projects\Glasses-scratch\scene-understanding-v1\`.

---

## 1. Executive summary

**What existed.** A complete, carefully reasoned cartridge (`tower/scene/`,
`tower/results/scene_understanding.py`, `routes/scene.py`, an iOS screen
implementing `scene_understanding.live/2026-08-27`) that nobody could use
without setting `TOWER_SCENE_UNDERSTANDING=1`, that started a people
detector behind any camera stream, whose detector missed most furniture
(mAP50 0.37), whose orientation stage was off by default and measurably
wrong when on, and whose wire refused to say which side a person was on.

**What was implemented.**

- **Capability is product-managed.** `TOWER_SCENE_UNDERSTANDING` is a
  tri-state (`auto` when unset): a Tower with the `[ml]` extra offers Scene
  Understanding with no configuration; the unavailable reason names what
  is actually missing. Device defaults to `auto`.
- **Activation is demand-driven.** A session runs while a phone streams
  **and** a client is subscribed to the live scene, or while an operator
  holds it via `POST /scene/start`. Last watcher out → stop and release the
  GPU; last stream out → stop, whoever started it (closes the "owned by
  nobody" defect that was a strict xfail). iOS subscribes when the Scene
  screen appears and unsubscribes when it disappears.
- **Detector:** RT-DETRv2-R18 (Apache-2.0, via `transformers`, fp32) on
  CUDA — mAP50 0.669 vs 0.367, person AP50 0.856 vs 0.604 on 700
  human-labelled images, at 17.8 ms median on real frames; SSDLite320 stays
  on CPU. Thresholds 0.5 / 0.4.
- **Tracker:** cardinality-first Hungarian association (ID switches
  5,488 → 2,018 on 10,800 labelled frames), tracks kept 1.0 s but
  **counted only while seen within 0.5 s** (exact-count 0.315 → 0.370).
- **Orientation:** the keypoint rule (96.5% of profile faces called
  "toward") is gone; facing is a face detector on each tracked person's
  box, two states only (toward / not established), 2-of-3 vote, 0.83
  precision on labelled stills, ~9 ms a face on CPU. Ships **EXPERIMENTAL**
  and the wire says so.
- **Wire (same contract id, additive):** people get side counts,
  apparent-size buckets, a `partial_bottom_edge` bucket for figures with no
  head region (most often the wearer's own body), orientation
  method/status/validation, re-measured limitations, and a stated
  single-person limitation.
- **Tooling:** `scripts/scene_replay.py` (deterministic replay over
  recorded captures, optional labelled fixture) and `scripts/scene_soak.py`
  (real-app start/stop and cartridge-switch soak with resource sampling).
- **Tests:** 326 scene-related tests (was 263), full suite in §10.

**Readiness:** READY WITH KNOWN LIMITATIONS — see §14. Everything that
needs a person in front of the glasses is a physical step (§13); the
corpus on this host contains no bystander.

---

## 2. Original implementation findings

- `TOWER_SCENE_UNDERSTANDING` (default off) gated whether
  `cartridge_runtime._scene_session` was constructed at all; "unset" read
  as "off" and the phone showed the variable's name. `available: true`
  was derived from construction succeeding (an eager `import torch`),
  which was honest but unreachable for a product user.
- Backend: `SceneEngine` (detect → IoU tracker → optional keypoint
  orientation → camera-relative state), `SceneLive` on the shared
  `LiveSession` (in-process reused worker thread, one-slot newest-wins
  frame path, discard-on-stop), `results/scene_understanding.py`
  (counts, per-label side counts excluding people, aggregate facing,
  every refusal as a value), five HTTP routes the phone never calls, a
  0.5 s poll / 2 s heartbeat result channel.
- iOS: a complete decoder and screen for the contract; subscribes at
  connection time regardless of which screen is open; renders "2
  persons"; a cold launch showed "no contract" (a `knownToThisBuild`
  omission).
- Placeholders vs functional: nothing was a placeholder. The detector,
  tracker, wire, lifecycle and tests were all real. What was wrong was
  measured: SSDLite's recall, the keypoint facing rule, the 4.7° centre
  band, the stream-only activation, and the env-var gate.

---

## 3. Research findings

Full tables in the research document. Summary:

| Track | Finding | Consequence |
|---|---|---|
| Real data | 97 captures / 45,594 frames audited: **no bystander anywhere**; wearer's own body in most desk captures; HFOV 44.7°, VFOV 72.3° | accuracy validated on COCO val2017 (human labels); a bottom-edge/no-head bucket for the wearer's body |
| Detector | 20 candidates on 700 labelled images + 300 real frames; RT-DETRv2-R18 0.669 mAP50 @ 17.3 ms; D-FINE-S 0.702 @ 28.3 ms but fp16-only speed and a checkpoint caveat; SSDLite 0.367 @ 29.5 ms | RT-DETRv2-R18 default on CUDA, D-FINE selectable, SSDLite on CPU; threshold 0.5 (per-image people count exact 73%, MAE 0.48) |
| Tracker | 10 candidates on 40 labelled synthetic-motion sequences; Kalman/Byte/OC-SORT: fewer switches, more phantoms, no count gain, 3× cost; departure lag dominated count error | cardinality-first Hungarian; 1.0 s kept / 0.5 s counted |
| Orientation | keypoint rule 0.58 precision, 96.5% of profiles "toward"; YuNet face-in-box 0.83 / 0.64 at t=0.9; yaw and AND-combos add nothing | YuNet, binary, voted, EXPERIMENTAL |
| Position | 0.45/0.55 = 4.7° centre band; person at 2 m spans 0.27 of the frame | 0.35/0.65 (14°) with 0.03 hysteresis |
| Depth | 2026-08-26 MiDaS ordering measurement stands (11.5% flips under motion) | still refused; apparent size published as size, never distance |
| Live state | 0.5 s poll / 2 s heartbeat, revision-coalesced | kept; demand counts added, volatile |
| Privacy | reviewer: no direct or persisted identity; single-person aggregates describe that person | accepted and stated on the wire (`single_person_note`) |
| Performance | old pipeline 11.9 fps observed at 12 fps paced, 16 CPU cores of OpenMP spin; new pipeline 17.8 ms/frame, 217 MB VRAM peak | fits with headroom; the spin is torch's default thread pool, unchanged |

Rejected: YOLO/ultralytics (AGPL, not installed); D-FINE as default;
LW-DETR (unverified conversion); Kalman trackers; keypoint orientation;
head-pose regressors needing new weights; metric distance; per-entity
rows; a subprocess worker.

---

## 4. Selected architecture

```
/ws frame (JPEG, ~12 fps)
    -> LiveSession one-slot newest-wins (event loop, ~0.01 ms)
    -> worker thread: decode (0.6 ms)
    -> detector_for(device): RT-DETRv2-R18 fp32 @0.5 on CUDA (17.8 ms) | SSDLite320 @0.4 on CPU (30 ms)
    -> Tracker: per-class Hungarian over 1+IoU; kept 12 frames, counted 6, confirmed after 3
    -> side assignment with hysteresis (0.35/0.65 ± 0.03)
    -> every 3rd frame: YuNet on each tracked person's head region (4-9 ms/face, CPU) -> 2-of-3 vote -> aged, expires 6 s
    -> SceneState (counted tracks; partial figures apart)
    -> results/scene_understanding.live_payload (counts, where incl. people, sizes, partial, facing aggregate)
    -> result channel 0.5 s poll / 2 s heartbeat -> iOS (also GET /scene)
```

Activation: `SceneLive` tracks open streams and watchers apart; runs iff
both (or operator hold). Models load on start (≈0.5 s warm, ≈5 s cold with
a first `transformers` import) and release on stop (VRAM back to the CUDA
context floor).

---

## 5. Scene data model

Internal (`tower/scene/records.py`, never on the wire): `Track` with a
session-scoped integer id, label, box, score, hit/miss/streak counters,
`side` (hysteresis), `facing: FacingEstimate(state, confidence,
age_seconds, evidence)`, `facing_history` (last 3 raw states).
`SceneState`: counted tracks, `partial_people`, relations, counts, frame
size, detector, threshold, orientation flags.

Wire (`scene_understanding.live/2026-08-27`, additive since 08-27):
`counts{13 classes}`, `where{label: {left,centre,right,unknown}}` now
including `person`, `people{count, may_include_wearer, validated,
partial_bottom_edge(+note), by_apparent_size{large,medium,small,unknown}
(+note), facing_wearer|null, facing_answered, facing_unknown,
facing_states_reported/withheld(+reason), oldest_estimate_seconds,
orientation_method, orientation_status:"experimental",
orientation_validation, facing_note}`, `lifecycle{state, session_id,
scene_is_current, …, demand{streams, watchers, operator_hold,
runs_when}}`, `count_limitations`, `count_measurement`,
`single_person_note`, and every prior refusal field unchanged
(`tracks: null`, `relations: null`, `confidence: null`).

---

## 6. Privacy

- **Ephemeral:** every track, estimate and scene lives in the worker's
  memory and is discarded on stop; `test_scene_understanding_persists_nothing`
  AST-walks the wire path and `test_a_full_run_leaves_the_filesystem_untouched`
  watches the disk across a real run.
- **Retained:** nothing. No imagery, no crops, no landmarks, no scores.
- **Face pixels:** YuNet reads the head region of a person box in memory,
  returns boxes/scores/landmarks; the estimator keeps the single best
  score long enough to compare it with 0.9 and discards everything. The
  score never reaches a track or the wire (`evidence` strings stay
  internal).
- **Track ids:** session-scoped integers, restart at 1 per session, never
  published (`refused_entity_fields`, substring-scanned by a test).
- **Why it cannot identify people:** no descriptor is computed, nothing
  is matched against anything, nothing outlives the session, and the wire
  carries only counts and aggregate buckets with no handle to join two
  payloads by.
- **Known limitation (privacy reviewer, accepted):** with exactly one
  person in view, the side, size and facing aggregates describe that
  person for as long as they are in view. That is the product question,
  it is what the wearer's own preview shows, and it is stated on the wire
  (`single_person_note`); clients are obliged not to store payload
  sequences (contract §12).
- **Orientation limits:** never gaze; two states; capped at MEDIUM;
  validated only on stills of other people's photographs.

---

## 7. Count, position and orientation semantics

- **"2 people visible"** means: two confirmed person tracks whose head
  region is in view were detected within the last 0.5 s in the camera's
  45° forward cone. It does not mean two people are in the room, that
  there are exactly two, or that either is looking at the wearer. Every
  count is a floor (`count_is_lower_bound`), and the wire lists why.
- **`partial_bottom_edge: 1`** means: one "person" box with no head region
  in view (or spanning the width to the bottom edge) — most often the
  wearer's own body, possibly somebody's legs — and it is not in `count`.
- **"1 person on your left"** means: that track's box centre is below
  0.35 of frame width, in the frame as received, with hysteresis; it
  changes when the wearer turns.
- **"1 large"** means: that person's box is at least 0.6 of the frame's
  height — a size in the picture, not a distance.
- **"1 appears to be facing your direction"** means: a face was found in
  the upper part of that person's box at score ≥ 0.9 in at least two of
  the last three estimates within the last 6 s. It does not mean they are
  looking at, noticed, or are attending to the wearer. `unknown` means not
  established, and covers facing away, side-on, too small and unmeasured.
- **`facing_wearer: null`** means orientation was not measured or nothing
  is established; it is never 0.

---

## 8. Real project data evaluation

- **Used, read-only:** all 97 captures under `tower/data/captures`
  (45,594 frames, 360×640) for the audit and the full replay; three
  captures copied to `Glasses-scratch/…/corpus3` (6,868 frames) for the
  baseline and the soak; `world_builder/intrinsics/360x640.json` for the
  FOV. Nothing under `tower/data/` was written.
- **Labelled evaluation:** 700 COCO val2017 images (human annotation)
  for detection, counting and orientation; 40 COCO scenes turned into
  10,800 synthetic-motion frames with known identity for tracking; an
  85-frame corpus fixture labelled by an AI agent (not a human) for
  object counts and the wearer-body rule.
- **Exploratory (unlabelled):** the full-corpus replay for cost and
  stability; the soak for lifecycle.
- **Measured on the 85-frame fixture:** object exact-count rates chair
  0.85, tv 0.79, laptop 0.69 (overcounts: monitors and duplicates),
  keyboard 0.68, mouse 0.80, cell phone 0.73, bottle 0.94, cup 0.91,
  bed 0.93. Person, first rule: labelled 3, engine 28 — the wearer's
  body; of 49 wearer-only frames, 40 produced a person box, 30 were
  routed to the partial bucket and 17 still counted a person. After the
  rule was tightened (no head region, bottom edge or not; or full width
  to the bottom): engine 13, **7 of 49 still counted**, 35 routed to the
  partial bucket, person exact rate 0.88. Over the whole corpus the
  frames with a counted person fell from 15,957 to 7,408 of 45,594 and
  the mean person count from 0.47 to 0.20 — in footage that contains no
  person. The remaining 7 are boxes whose top reaches above the middle
  of the frame (an arm raised into it, a torso seen looking down at a
  laptop) and cannot be separated from a close bystander by geometry.
- **Limitations:** no bystander footage; AI labels; COCO stills are not
  this camera; no ground truth for facing on this camera.

---

## 9. Performance

Quiet RTX 5070, real 360×640 frames.

| Measure | Old pipeline (SSDLite, CPU default) | New pipeline (RT-DETRv2-R18 CUDA + YuNet) |
|---|---|---|
| Detector latency | 29.5 ms med / 31.4 p95 (CUDA), 43.5 CPU | **17.8 ms med / 20.6 p95** (45,594 frames) |
| Orientation | 48 ms/frame CUDA, 1,112 CPU (KeypointRCNN) | **4.4 ms/face med, 9.9 p95** (CPU) |
| Observe total | — | **18.0 ms med / 21.1 p95**, max 516 (first call) |
| Throughput | 11.9 fps observed at 12 fps paced, 0.5% skipped | 50.8 fps unpaced |
| VRAM | 56 MB (SSDLite); 754 MB (KeypointRCNN) | **217 MB peak allocated, 298 MB reserved** while running; 46 MB after stop |
| RSS | +623 MB (CUDA context) | +1.85 GB (CUDA context + `transformers`) |
| CPU | ~16 cores (OpenMP spin, torch default threads) | ~15 cores (same cause; `TOWER_SCENE_TORCH_THREADS` applies) |
| Model load | 2.4 s | 5.0 s cold (first `transformers` import), **0.49 s** warm restart |
| Stop | — | **≤ 2.7 ms** to `stopped` on the wire (soak) |
| Facing claims on the corpus | — | **0** in 45,594 frames (335 estimator calls; no bystander exists) |

Soak, 20 start/stop cycles through the real app and result channel
(`results/soak_20`, verdict **STABLE**): scene never survived a stop, no
watcher left over, health 200 throughout, VRAM allocated flat at ~115 MB
while running and 34 MB after every stop (reserved 298 MB running, 46
MB after), RSS 1,898 → 1,918 MB over the run and **+0.5 MB** between the
first and last thirds, OS threads 94 → 104 (+4.0 between thirds, i.e.
the asyncio default executor filling from 0 to 6 workers plus one
transient teardown thread per cycle; Python-visible threads 15 at the
end: 6 executor, 1 scene worker, 1 CV Lab loader pair, 1 tqdm monitor,
the rest are torch's native pool created at first load and never
reclaimed), handles +20 between thirds, **stop ≤ 2.2 ms**, warm start
median 488 ms, cold first start 3.3 s. **Cartridge switch (Scene → CV
Lab → Scene) was NOT completed by the soak harness:** its `--switch`
branch hangs waiting for the Lab's reply to `cv_lab_start` over the
shared socket (a harness defect in the Lab handshake, twice retried,
not a product finding). What covers the switch instead: the e2e test
that a stream with no watcher leaves the scene `stopped` while frames
flow (`test_a_stream_alone_does_not_start_it`), the e2e tests that
unsubscribing and resubscribing on one stream start a fresh session,
the 20-cycle soak above, and steps 32–34 of the physical plan.

Replay with the tightened partial-figure rule (`results/replay_full_cuda_v2`,
45,594 frames, 74.1 min of footage): detector 18.0 ms median / 20.5 p95,
observe 18.2 / 20.9, 51.7 fps unpaced, 14 CPU cores, VRAM 217 MB peak;
frames with a counted person 7,408 (was 15,957), with a partial figure
25,453; **count changes 100 per minute and tracks created 46 per
minute** over footage of hand-held phones, laptops and the wearer's own
body — the object counts on this footage flicker, and the physical test
should expect a laptop or phone count to move when the wearer's hands
move. Facing claims: 0.

---

## 10. Testing

- **Added:** `tests/test_scene_activation.py` (24), `tests/test_scene_capability.py` (16),
  `tests/test_scene_geometry_and_policy.py` (28), rewritten
  `tests/test_scene_orientation_staleness.py` (26), new cases in
  `test_scene_wire_e2e.py` (activation, positions for people, sizes,
  partial), Swift `SceneUnderstandingAdditionsTests` (5, not compiled here).
- **Migrated:** every keypoint-orientation consumer; `test_live_session_lifecycle_races.py`'s
  strict xfail is now a passing test; stream-driven e2e tests subscribe.
- **Scene-related suites:** 326 passed, 1 skipped (`-k "scene or documented"`).
- **Full Tower suite:** 2,668 passed, 75 skipped, 1 failed in 11 min 44 s
  (`results/pytest-full.log`); the failure (`test_ws_finalize_robustness`,
  the connection tracker read before a two-hop teardown finished) was
  fixed by updating the tracker before the awaits, the ws/capture/session
  suites (523) then passed, and the final full run is **2,669 passed,
  75 skipped, 0 failed in 7 min 44 s** (`results/pytest-full-final.log`).
- **Three tests changed meaning with the default flip** and now set
  `TOWER_SCENE_UNDERSTANDING=off` explicitly: the two subprocess import
  probes in `test_architecture_boundaries.py`, `test_no_other_cartridge_can_be_subscribed_to`,
  and `test_cartridges_without_a_contract_are_not_offered`. The first
  full run of the suite stalled in the isolation test, which waited for
  a refusal that a now-available cartridge never sends.
- **Replay:** smoke on corpus3 and full corpus (§9), labelled fixture (§8).
- **Soak:** §9.
- **Could not run:** the Swift tests and any iOS build (no Xcode on this
  host); anything requiring a person in front of the glasses.

---

## 11. Independent review

Three reviewers (privacy/orientation, lifecycle/code, perception/architecture).

- **Privacy:** PASS WITH LIMITATIONS. SEV-2 single-person aggregate →
  accepted, stated on the wire and here. SEV-3 expiry wording asserted an
  expiry that had not happened → fixed (`_not_established_reason`). SEV-3
  per-entity query layer unguarded → boundary test added. Verified fine:
  transient face data, no away/profile, MEDIUM ceiling, base-rate caveat,
  discard on stop, no per-frame logging.
- **Lifecycle:** PASS WITH LIMITATIONS. SEV-1 demand lock held across a
  5 s stop join could freeze the event loop → fixed (lock released before
  `stop()`, reconciled after; a test proves a demand event returns in
  <0.5 s while a stop is joining). SEV-2 per-frame exception log flood on
  a poisoned detector → rate-limited (1st, 10th, 100th…). SEV-2 engine
  release racing an abandoned worker inside `detect()` after a join
  timeout → **accepted residual** in the shared base (needs a >5 s single
  forward pass); documented here, not changed cross-lane. Verified fine:
  lock order, Hungarian (brute-force checked), config/boot cost,
  `counted()` everywhere it matters.
- **Perception/architecture:** LIMITED on counting, objects and
  position; EXPERIMENTAL on orientation (all as reported in §14). Every
  headline number reproduced from the JSONs. Findings acted on: the
  D-FINE fp32 "121 ms" in `detect.py` named the wrong checkpoint →
  corrected with both; `COUNT_LIMITATIONS` quoted the CPU detector at
  0.5 instead of its 0.4 floor (64%/1.25 → 66%/1.15) → corrected;
  LW-DETR-small beats the default on the count metric (0.783 / 0.42 vs
  0.733 / 0.48) and mAP50 at equal latency and was set aside on an
  unverified licence → licence checked (card Apache-2.0, upstream
  Apache-2.0, lineage to the authors' checkpoints unstated), the model
  made selectable at its own threshold, and the trade stated in
  `detect.py` and the research doc; the Bayes arithmetic on the
  orientation base rate (0.38–0.58 precision at a 10–20% base rate) →
  put on the wire; plain max-weight Hungarian beat cardinality-first on
  4 of 6 tracker metrics → the trade (13% fewer phantoms, the
  no-starvation guarantee) stated in `tracking.py`; the ablation script
  compared production with itself after production changed → corrected;
  threshold chosen on the same 700 images as the accuracy, the
  engineered 57/43 subset, the fixture's lack of independent motion, the
  ~5% rerun variance, and the size buckets' zero real-bystander basis →
  disclosed in the research doc. Accepted: "no count gain" for Kalman
  was an overstatement (+0.01–0.03 exact) and is now worded so.

---

## 12. Git handoff

- Branch `feature/scene-understanding-v1`, worktree above, base `6beaf57`.
- Commits (oldest first): `4283220` demand-driven activation; `2e05923`
  capability, detector, tracker, orientation, wire, replay; `c431969` iOS,
  soak, geometry tests; `3ef6680` docs; `3e1906c` privacy review fixes;
  `b3bce0e` lifecycle review fixes; `5f1dcef` partial-figure rule;
  `a4ed055` soak thread attribution; `3e1906c`… `c2d0993` review fixes
  (privacy, lifecycle, perception); `5f1dcef` partial-figure rule v2;
  `339d508` default-flip test fixes; `2024d93` teardown bookkeeping;
  final HEAD: the `docs(handoff)` commit that carries this file (`git
  log -1` on the branch).
- Major files: `tower/tower/scene/{live,engine,detect,orientation,tracking,state,records}.py`,
  `tower/tower/results/scene_understanding.py`, `tower/tower/cartridge_runtime.py`,
  `tower/tower/config.py`, `tower/tower/routes/{results_ws,ws}.py`,
  `tower/tower/results/registry.py`, `tower/tower/main.py`, `tower/pyproject.toml`
  (`scene` extra), `tower/scripts/{scene_replay,scene_soak,scene_session,cartridge_live_benchmark,scene_benchmark}.py`,
  `tower/docs/contracts/CARTRIDGE-RESULTS.md`, `docs/contracts/TOWER-UNIFIED-CARTRIDGES.md`,
  `tower/guidelines/docs/modules/SCENE-UNDERSTANDING.md`,
  `ios/Glasses/Workspaces/SceneUnderstanding/*.swift`, `ios/GlassesTests/SceneUnderstandingTests.swift`.
- Scratch (not committed, keep): `C:\Users\tvllo\Projects\Glasses-scratch\scene-understanding-v1\`
  — `coco/` (700 labelled images + annotations), `detector/`, `tracker/`,
  `orientation/`, `corpus-audit/` (sheets, survey, labelled fixture),
  `results/` (baselines, replays, soaks), `corpus3/` (copied captures).
- Weights fetched into the user's Hugging Face cache on first load
  (`PekingU/rtdetr_v2_r18vd`, ~80 MB); the vendored YuNet at
  `tower/models/` is reused.
- **Integration with concurrent lanes.** Shared files touched:
  `cartridge_runtime.py` (`LiveCartridges.watcher_*`, `_scene_session`,
  `build_live_cartridges`, two new constants), `config.py` (scene fields
  and three parsers), `routes/ws.py` (`ChannelHolder` construction, the
  teardown `try/finally`, `_close_cartridge_streams`), `routes/results_ws.py`
  (`ChannelHolder`, subscribe/unsubscribe hooks), `results/registry.py`
  (`SCENE_DISABLED_REASON` wording), `main.py` (log lines),
  `tests/test_architecture_boundaries.py` (Lab import probe sets scene
  off). `live_session.py` is untouched. The Document Memory lane has
  commits touching `document_memory/*` and its guidelines doc — no
  overlap; the Object Memory runtime lane is clean at the base as of this
  writing. Expect line-level, not semantic, conflicts in `ws.py` and
  `cartridge_runtime.py` if either lane edits the same regions.
- `TOWER_SCENE_UNDERSTANDING` unset now means **auto**: after
  integration, every Tower with the `[ml]` extra offers the cartridge.

---

## 13. Physical device test plan (Windows Tower + iPhone + Ray-Ban Meta)

Build the phone from this branch (the Swift changes are uncompiled here;
if the build fails, the 2026-08-27 app still works with one difference:
it subscribes at connection time, so the detector runs whenever the app
streams). Start the Tower from the worktree with the `.venv`:

```
cd C:\Users\tvllo\Projects\Glasses-worktrees\scene-understanding-v1\tower
& ..\..\Glasses\tower\.venv\Scripts\python.exe -m uvicorn tower.main:app --host 0.0.0.0 --port 8000
```

(No `TOWER_SCENE_*` variable needed. First start on a fresh cache fetches
~80 MB from huggingface.co; `state: starting` may last ~10 s once.)

1. Start Tower once. Note the log line `Scene Understanding is enabled (mode auto)`.
2. Connect iOS; confirm Tower: Connected.
3. Open Scene Understanding.
4. Verify the screen no longer says "not enabled"; `curl http://<tower>:8000/cartridges` shows `scene_understanding … available: true`.
5. Start the camera (Home/World Builder controls as today); the Scene screen goes "Still loading" then live. `GET /scene` shows `lifecycle.state: running`, `demand: {streams: 1, watchers: 1}`.
6. Point at a blank wall.
7. Verify "Nothing in view" (`scene_available: true`, all counts 0).
8. Have one person stand 2 m in front.
9. Verify "1 person observed" within ~0.5 s; `people.partial_bottom_edge` 0; `by_apparent_size` medium.
10. Add a second person.
11. Verify "2 people observed" holds for 30 s without flicker (count changes ≤ 2).
12. Move a person to the left / centre / right of view.
13. Verify `where.person` follows; centre needs the person within ~7° of straight ahead.
14. Have a person face the wearer, then turn away.
15. Verify "1 appears to be facing your direction" appears within ~1 s and disappears within ~1 s of turning; the label says Experimental.
16. Place a chair, a laptop, a cup, a phone in view.
17. Verify chair/laptop/phone counts; cup and bottle may need to be within ~1.5 m.
18. Turn the head naturally for 20 s.
19. Verify counts do not flicker more than a couple of times.
20. Occlude a person behind a door for 1 s.
21. Verify the count drops after ~0.5 s and returns within ~0.3 s of reappearance.
22. Pause (`curl -X POST http://<tower>:8000/scene/pause`).
23. Verify the screen shows "Last reading" and `scene_is_current: false`.
24. Resume (`POST /scene/resume`).
25. Verify counts update again.
26. Leave the Scene screen (or `POST /scene/stop`).
27. Verify `GET /scene` shows `stopped`, `scene_available: false`, `demand.watchers: 0`.
28. Verify nothing was written: `tower/data/` unchanged (`git status` in the canonical checkout, and no new directory under the worktree's `tower/data`).
29. `curl http://<tower>:8000/health` → 200.
30. Reopen Scene Understanding.
31. Verify a new `session_id` and live counts within ~1 s (warm start ≈ 0.5 s).
32. Switch to World Builder or CV Lab; verify `GET /scene` shows `stopped` while the stream continues (`demand.streams: 1, watchers: 0`).
33. Return to Scene Understanding.
34. Verify it runs again without a Tower restart.
35. `Get-Process python | Select Threads, WorkingSet64` before and after 10 open/close cycles; expect threads within ~+10 and RSS within ~+50 MB after the first cycle.
36. `curl http://<tower>:8000/health` → 200; the phone still streams.

**If something fails, capture:** the Tower console (all `[Tower][Scene]`,
`[Tower][Cartridge]`, `[Tower][Results]` lines and any traceback);
`curl http://<tower>:8000/scene` and `/cartridges` at the moment of
failure; `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv`;
`Get-Process python | Select Threads, Handles, WorkingSet64`; and the
phone's Xcode console filtered on `result_subscribe`, `result_error`,
`cartridge_result`. For a wrong count, also note where the person stood,
whether their head was in view, and whether a hand or arm was raised.

---

## 14. Verdict

**SCENE UNDERSTANDING V1: READY WITH KNOWN LIMITATIONS.** The cartridge
starts with no configuration, runs only while watched, counts with a
detector nearly twice as accurate as before at lower latency, publishes
where people are and how big they appear, releases its models on stop,
and survived a start/stop soak; nothing about people has been checked on
this camera because the corpus contains no bystander.

- **PEOPLE COUNTING: LIMITED.** Person AP50 0.856 and per-image exact
  count 73% / MAE 0.48 on labelled stills; the wearer's own body still
  counts as a person in 7 of 49 labelled wearer-only frames (was 17);
  LW-DETR-small would count better on stills and is one setting away;
  unvalidated through these glasses.
- **OBJECT SCENE PERCEPTION: LIMITED.** mAP50 0.669 on the 13 classes;
  chair 0.54, laptop 0.83, tv 0.82; exact-count rates 0.68–0.95 on the
  85-frame corpus fixture; but counts of hand-held objects flicker on
  real footage (100 count changes a minute over the corpus) and book and
  dining table remain weak and say so.
- **COARSE POSITION: LIMITED.** Side bands sized to the measured 44.7°
  field with hysteresis and unit-tested geometry, people included, no
  metric claim; the band widths and the size buckets are projections
  from calibrated intrinsics with no real bystander box to check them
  against.
- **PERSON ORIENTATION: EXPERIMENTAL.** 0.83 precision / 0.64 recall on
  966 labelled persons in stills, binary, voted, worded conservatively;
  zero claims across 45,594 corpus frames; not validated on this camera.
- **PRIVACY MODEL: PASS WITH LIMITATIONS.** Nothing persisted, nothing
  identifying, transient face pixels only; the single-person aggregate
  limitation is stated on the wire.
- **LONG-LIVED TOWER COMPATIBILITY: PASS WITH LIMITATIONS.** Demand-driven
  start/stop, models released, VRAM flat across cycles, stop ≤ 3 ms,
  health stays 200; residual: an abandoned worker inside a >5 s forward
  pass could see its model released (shared base, unchanged), and torch's
  default thread pool still spins ~15 cores while a session runs.
