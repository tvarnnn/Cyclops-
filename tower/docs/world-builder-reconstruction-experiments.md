# World Builder reconstruction — experiment ledger

**Started:** 2026-09-06
**Branch:** `world-builder/global-reconstruction-v1` (worktree `C:\Users\tvllo\Projects\Glasses-worktrees\wb-recon`)
**Scratch root for all runs:** `C:\Users\tvllo\Projects\Glasses-scratch\wbrecon\`
**Mission:** turn World Builder capture data into ONE coherent room, through a
generic program, with visual evidence, on a path that can run incrementally live.

Every entry records: dataset, hypothesis, change, result, artifact, decision.
Numbers are measured unless marked *inferred*.

---

## 0. Corpus and baseline (2026-09-06 03:15–03:45)

Inventory: `Glasses-scratch\wbrecon\inventory\CATALOG.md` (79 captures, all
360x640, all with raw frames; 55 sessions, 37 with derived geometry, 2 with
placements). Benchmark walks (all replayable from raw frames):

| walk | captures | frames | live world / session | keyframes / segments / solved / points |
|---|---|---|---|---|
| **0906** (the failed live test) | `ddcf9426` | 1371 | `678fe396` / `b6b47fdd` | 438 / 34 / 194 / 29,860 |
| 0901 loop | `1ac63b51`+`60bf1b02` | 2613 | `b5feee12` / `96e0344d` | 434 / 30 / 323 / 30,382 |
| worldB drawer | `023b5d84`+`95831c34`+`e3e8fd2e` | 1074 | `af47007c` / `7864d3b3` | 218 / 36 / 108 / 13,050 |
| worldA normal | `69a4e59d`+`28f544af`+`01e4c64f` | 1008 | `991e5a15` / `815c88ba` | 229 / 23 / 100 / 9,145 |
| long0827 | `0fc400bb` | 2203 | `f80e88a5` / `5578241c` | 339 / 24 / 90 / 18,899 |
| dense | `0bbc2b7e`+`74f7e34c` | 377 | `f4252374` / `5d478211` | 77 / 6 / 68 / 11,009 |

`2789a227` (same day as 0906) is a separate 27 s false start, not part of the walk.

**Baseline** (`Glasses-scratch\wbrecon\baseline\BASELINE.md`, HEAD `a7b1c2a`,
replay via `scripts/world_replay.py --captures … --register`):

| walk | kf | segs (w/ geom) | solved | refused | points | candidates | admitted | registered segs | registered pts | dominant share | reproj med / p99 | reg s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0906 | 438 | 34 (17) | 194 | 210 (25 root + 185 cascaded) | 29,860 | 136 | 2 | 2/17 | 36.7% | 0.367 | 0.54 / 2.34 | 88 |
| 0901 | 434 | 30 (18) | 323 | 81 | 25,131 | 153 | 15 | 10/18 | 74.9% | 0.749 | 0.54 / 2.51 | 129 |
| worldA | 229 | 23 (8) | 100 | 106 | 6,762 | 28 | 0 | 0/8 | 0 | 0.413 | 0.56 / 2.46 | 7 |
| worldB | 218 | 33 (22) | 137 | 48 | 12,686 | 231 | 6 | 6/22 | 58.1% | 0.581 | 0.51 / 2.56 | 52 |

The 0906 replay reproduces the live session figure for figure (438 / 34 / 194 /
210 / 29,860; 136 candidates, 2 admitted, pairs (9,13) and (14,17)) and is
deterministic across two runs. Full test suite at baseline: **2487 passed, 75
skipped, 1 xfailed, 0 failed** (603 s).

Baseline render of the live 0906 world: `Glasses-scratch\wbrecon\render\678fe396\`
(`overview.png`: two registered segments, 10,958 points, 58 cameras;
`unregistered_tiles.png`: 15 fragments in their own frames, several visibly
diverged — segment 16 reaches −500 units).

---

## E1. Global SfM over the session keyframes (pycolmap 4.2, incremental) — 0906

- **Hypothesis.** The features exist and the sequential chain is what fails
  (drift report §4, registration census). A global solver over pairwise
  constraints across ALL keyframes — not a chain, not segment-to-segment Sim3
  — should place most of the walk in one frame.
- **Change.** No repository change; driver `Glasses-scratch\wbrecon\exp\colmap_exp.py`.
  Inputs: the session's 438 keyframe JPEGs (face-redacted copies), the
  session's own intrinsics (`FULL_OPENCV`, fx fy cx cy k1 k2 p1 p2 k3, k4–k6 = 0)
  held FIXED. SIFT (CPU, ≤4096 features), sequential matching overlap 20, no
  loop detection, `incremental_mapping` with `max_extra_param = 1000`.
- **Gotcha that cost one run.** COLMAP's default `max_extra_param = 1` treats a
  camera with |k3| = 1.30 as degenerate and refuses every initial pair
  ("Discarding reconstruction due to bad initial pair", 0 models). The
  calibration is fine; the filter is not for calibrated cameras.
- **Result** (`exp\runs\0906_seq20_inc\`): extract 3.7 s, match 28 s, map 72 s.
  8,550 verified pairs (5,559 with ≥15 inliers). **408/438 keyframes registered
  in 4 models: 307 + 41 + 47 + 13.** Model 0: 307 images, 19,771 points, mean
  reprojection **0.87 px**, mean track length 7.4, min 44 3D observations per
  image, spanning keyframes 1–2139 (the desk/shelf/door part of the room);
  model 1 (2297–2485) is the bed/laptop corner; model 2 (2502–2735) the end.
- **Cross-solver check** (`exp\crosscheck.py`, Sim3-align each local tracker
  segment's camera centres to model 0): segment 9 (20 common keyframes)
  **0.4% of extent RMS**; segments 17, 20 5–6%; segments 5, 13, 14 14–17%;
  segment 16 30% — 16 and the tail of 13 are exactly the segments the baseline
  render shows diverged. Two independent solvers agree where the local one is
  sound and disagree where it is known to be wrong.
- **Against baseline:** largest coherent piece 58 keyframes / 10,958 points →
  **307 keyframes / 19,771 points**.
- **Artifact:** `exp\runs\0906_seq20_inc\model_0.png` (+ `model_0.ply` with
  colours). Keyframe montage `exp\montage_0906.jpg`.
- **Decision:** RETAIN as the direction. Global SfM is the engine.

## E2. Control: plain `OPENCV` model (k3 dropped) — 0906

- **Hypothesis.** The self-calibration's odd k1/k2/k3 might be hurting.
- **Result** (`exp\runs\0906_seq20_inc_opencv\`): 402 registered but split
  191 + 113 + 41 + 46 + 11, model reprojection 1.02 px, map 236 s (3.3× slower).
- **Decision:** REVERT. The calibrated k3 is real; keep the full model, fixed.

## E3. GLOMAP (`global_mapping`) with `OPENCV`, intrinsics refined (its default) — 0906

- **Result** (`exp\runs\0906_seq20_glob_opencv\`): map 124 s, **423/438 in ONE
  model**, 19,446 points, 0.77 px, mean track 8.1 — but 14 images have <20 3D
  observations and 3 have **zero**, sitting 30–214 units from the room
  (`model_0.png`). Rotation averaging poses every image; positions with no
  track support are not measurements.
- **Cross-solver check:** identical pattern to E1 (segment 9: 0.3%).
- **Decision:** PROMISING with a support floor (≥30 3D observations, which
  E1's incremental model satisfies everywhere with min 44). GLOMAP with
  `FULL_OPENCV` failed with "no 3D points to optimize" — its default bundle
  adjustment refines focal length and extra params; E4 pins them.

## E4. GLOMAP, `FULL_OPENCV`, intrinsics fixed — 0906

- **Result** (`exp
uns\0906_seq20_glob_fixed\`): still 0 models. GLOMAP's
  track filter ("Filtering tracks by reprojection") empties every component
  under the 8-parameter model even with intrinsics fixed; the same database
  maps fine incrementally. Not worth chasing inside GLOMAP — E8 removes the
  distortion model from the problem instead.
- **Decision:** REVERT (do not feed FULL_OPENCV to GLOMAP).

## E6. Unredacted raw capture frames — 0906

- **Hypothesis.** The session's keyframe copies are face-redacted and, on
  this walk, the redactor blacks out large phone-screen/hand regions
  (`exp\montage_0906.jpg`), destroying features. The raw capture frames
  (retained, never redacted) should reconstruct better.
- **Change.** Hard-linked the 438 raw frames by name into `exp
aw_0906\`
  (`frames/<source_seq>.jpg` == the keyframe file name); E1 settings.
- **Result** (`exp
uns\0906raw_seq20_inc\`): 427 registered; model 0
  **337 images / 21,928 points / 0.88 px** (E1: 307 / 19,771). Extraction and
  matching ~2× slower (more features survive).
- **Decision:** RETAIN. The solver reads raw frames when the capture is on
  disk, and falls back to the redacted keyframe copy per frame otherwise.
  Nothing from the frames reaches the wire: the derived output is points and
  poses, exactly as before.

## E7. 0901 loop walk, E1 settings (generality)

- **Result** (`exp
uns\0901_seq20_inc\`): 420/434 registered, models
  183 (kf 584–1895, 12,063 pts, 0.82 px) + 170 (2998–5861, 2,782 pts) + 51 +
  13 + 3. The two large models sit either side of the transport-disconnect
  gap between the walk's two captures, where sequential matching has no
  window; the walk is a loop so a loop-closure pass should merge them.
- **Against baseline** (dominant component 156 keyframes / 18,817 pts after
  registration): the largest coherent piece is 183 keyframes before any loop
  closure. Modest here; the 0906 walk is the one the chain failed on.
- **Artifact:** `exp
uns\0901_seq20_inc\model_1.png`.

## E8. Undistort once, then PINHOLE; GLOMAP vs incremental — 0906

- **Hypothesis.** Undistorting the frames with the session calibration
  (`cv2.initUndistortRectifyMap`, `getOptimalNewCameraMatrix` alpha 0, crop
  to the valid ROI → 335×595) turns every solver into a pinhole problem,
  which GLOMAP handles and which removes the extra-parameter filter entirely.
- **Result, GLOMAP** (`exp
uns\0906und_seq20_glob\`): extract 7 s, match
  33 s, **map 43 s**. **428/438 keyframes in ONE model, 22,042 points, 0.84 px,
  mean track 8.3**, p5 of 3D observations per image 91, only 3 images below
  30 observations, no camera further than 13 units from the room (E3 had
  three at 30–214). Cross-solver check identical to E1 (segment 9: 0.3%).
- **Result, incremental** (`exp
uns\0906und_seq20_inc\`): map 85 s,
  320 + 48 + 42 + 15 + 25. Better than E1 but still split.
- **Against baseline:** 58 keyframes / 10,958 pts in one frame →
  **428 keyframes / 22,042 pts**, i.e. 13% → 98% of the walk's keyframes.
- **Artifact:** `exp
uns\0906und_seq20_glob\model_0.png`, `model_0.ply`.
- **Decision:** RETAIN. Engine recipe = raw frames → undistort → SIFT →
  sequential matching (+ loop closure, E5) → GLOMAP → support floor per
  image. Incremental mapping stays as the fallback when GLOMAP yields nothing.

## E5. Exhaustive matching (loop closure upper bound) — 0906 — running
## E9. Live cadence: extend a database and continue a model — running
## E10. E8 recipe on 0901 / worldA / worldB / dense / long0827 — running

### E5 result. Exhaustive matching + incremental — 0906 (redacted frames, FULL_OPENCV)

- **Result** (`exp\runs\0906_exh_inc\`): match **230 s**, map **304 s**;
  **407/438 in ONE model**, 21,111 points, 1.07 px, mean track 10.9. Loop
  closures do merge the walk for the incremental mapper, at 9 minutes.
- **Decision:** not the live recipe. E8 (undistort + GLOMAP, sequential only)
  reaches 428/438 in 84 s. Exhaustive or vocabulary-tree loop closure stays a
  finalisation option to evaluate on loop walks (E10's 0901 result decides).

### E9 result. Live cadence: extend a database, continue a model — 0906 undistorted

- **Change.** `exp\continue_exp.py`: extract+match the first 300 keyframes,
  map; then extract+match ALL 438 against the same database; continue the
  model with `incremental_mapping(input_path=…)`.
- **Result:** first 300: extract 4.6 s, match 22 s, map 52 s (299 registered).
  Extending to 438: extract **2.2 s**, match **12.3 s** — pycolmap skips
  images and pairs already in the database, so per-update cost is
  proportional to the NEW keyframes. Continuing the model: **29 s**
  (`fix_existing_frames=True`, 325 + 48 + 42 + 15) vs 31 s free.
- **Decision:** the background path can re-solve every ~30–60 s of walk.
  Database and features persist across solves; the model may be rebuilt from
  scratch with GLOMAP (43 s at 438 images) or continued incrementally.

## E10. The E8 recipe on the other five walks (undistort → SIFT → sequential 20 → GLOMAP)

`exp\prep_undist.py` stages each session's keyframes from the raw captures
(all six sessions had every keyframe's raw frame on disk); `exp\batch_walks.sh`.

| walk | keyframes | registered | models (images) | largest model: kf / points / px | baseline largest coherent piece (kf / pts) | seconds (extract+match+map) |
|---|---|---|---|---|---|---|
| 0906 | 438 | 435 | 428 + 7 | **428 / 22,042 / 0.84** | 58 / 10,958 | 7 + 33 + 43 |
| 0901 loop | 434 | 432 | 257 + 170 + 5 | **257 / 16,960 / 0.78** | 156 / 18,817 | 7 + 42 + 59 |
| worldB drawer | 218 | 216 | 207 + 5 + 4 | **207 / 7,335 / 0.84** | 61 / 7,371 | 3 + 13 + 11 |
| worldA normal | 229 | 226 | 108 + 49 + 46 + 18 + 5 | **108 / 3,472 / 0.73** | 51 / 2,790 | 3 + 12 + 14 |
| dense 08-29 | 77 | 77 | 77 | **77 / 3,238 / 0.87** | 59 / 8,638 | 1 + 4 + 6 |
| long 08-27 | 339 | 339 | 144 + 139 + 52 + 4 | 139 / 4,605 / 0.80 (the 144-image model has 258 points: a rotation-dominated stretch, posed but nearly point-free) | 27 / 3,682 | 4 + 18 + 17 |

- The largest coherent piece grows in keyframe coverage on **every** walk.
  Points are fewer than the chain's on some walks (the chain publishes
  two-view points; COLMAP keeps tracks of ≥3 views with sub-pixel error):
  fewer, better-supported points in one frame beats more points in fragments.
- 0901, worldA and long0827 stay split at capture-disconnect gaps and
  revisits that a sequential window cannot bridge; E11 tests loop detection.
- Renders: `exp\runs\<walk>und_seq20_glob\model_0.png`.
- **Decision:** RETAIN the recipe as the engine. Build it into the product.

## E11. Sequential matching + loop detection (vocabulary tree) — 0901 / worldA / long0827 — running

### E11 result. Sequential + vocabulary-tree loop detection (GLOMAP, undistorted)

`SequentialPairingOptions.loop_detection = True`; pycolmap downloads COLMAP's
own `vocab_tree_faiss_flickr100K_words256K.bin` (BSD, from the COLMAP GitHub
release) once into `~/.cache/colmap/` and reuses it.

| walk | without loop detection (largest model) | with loop detection | match s: without → with |
|---|---|---|---|
| worldA | 108 / 229 kf, 3,472 pts | **210 / 229 kf, 5,457 pts, 0.81 px** (+19) | 12 → 24 |
| long0827 | 139 kf, 4,605 pts | **335 / 339 kf, 9,191 pts, 0.79 px** (+4) | 18 → 36 |
| 0901 loop | 257 + 170 + 5 | 243 (kf 573–2337) + 184 (kf 1–5861, 1,692 pts): the loop closed start-to-end, the middle stays its own model | 42 → 65 |

- **Decision:** RETAIN. Loop detection is ON for the finalisation solve and
  available to the background solve. 0901's two remaining models are the
  walk's textured middle and its start/end; both are coherent, they are
  simply not yet joined, and they are reported as two frames.

## E12. Integration: the solver in the product path — 0906 (smoke on a copy of the live world)

- **Change (repository).** `tower/world_builder/global_solve.py` (engine),
  `scripts/world_solve.py` (CLI), `engine.build()` merges a persisted
  solution, `scripts/world_build_session.py --solve` (background child every
  `--solve-every` keyframes + final solve after Stop; the Sim3 registrar is
  skipped when a solution exists), `config.world_solve` / `TOWER_WORLD_SOLVE`
  (default on) → the Tower's builder worker gets `--solve`,
  `scripts/world_replay.py --solve`, coherence report `global_solve` block,
  manifest `coverage` per segment, `GET /worlds`.
- **Result** (`Glasses-scratch\wbrecon\smoke\root0906\`, solve CLI with
  `--final --loop-detection`, 2 min 1 s; `engine.build()` 8 s):
  424/438 keyframes posed, **29 of 34 segments registered into ONE reference
  (segment 0)**, 5 refused (lone-anchor segments), 15,054 points, coverage
  22 confident / 7 partial / 5 unresolved. Served manifest: 29 `registered`
  rows with real Sim3s (scale 1) into reference 0, `current: true`.
  Coherence report: dominant component **99.4%** of points (baseline 36.7%);
  solver self-check over 187,357 observations: median 0.76 px, p95 3.00,
  p99 3.74, max 4.62, 0 behind camera.
- **Artifact:** `Glasses-scratch\wbrecon\smoke\render0906\overview.png` (+
  `world.ply`, `world.html`).
- **Tests:** `tests/test_world_builder_global_solve.py` (11),
  `tests/test_world_builder_solve_cadence.py` (4),
  `tests/test_world_builder_library.py` (5); World Builder subset
  648 passed / 0 failed.
