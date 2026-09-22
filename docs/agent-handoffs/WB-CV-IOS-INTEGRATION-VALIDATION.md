# World Builder + CV Lab — Mac integration and iOS validation (2026-09-06)

**Lane:** Mac integration / iOS validation.
**Branch:** `integration/wb-cv-ios-validation-v1`, worktree
`/Users/tristan/Projects/Glasses-worktrees/wb-cv-ios-validation-v1`.
**Base:** `integration/all-current-v1 @ a7b1c2a`.
**Toolchain:** Xcode 26.6 (17F113), Swift 6.3.3, iOS SDK 26.5, Simulator
iPhone 17 Pro (iOS 26.5). Python 3.12.5 in a venv at
`Glasses-scratch/wb-cv-venv` (`.[dev]` + matplotlib; no torch, no pycolmap).
**No physical iPhone and no glasses were attached during this session.**
Nothing below claims otherwise.

---

## 1. The headline

Both Windows lanes' Swift — 3,241 lines written without a compiler —
**compiled on the first Mac build with zero errors**, and every test they
wrote executed and passed. The merge had no textual conflicts and no
semantic ones. On top of that this lane added the one product piece that
was missing — the World Builder's interactive HTML picture viewable inside
the app — and proved both cartridges end to end in the Simulator against a
Tower running this branch.

| Gate | Result |
|---|---|
| iOS Debug build (Simulator) | clean; 8 warnings, all pre-existing on `a7b1c2a` (DocumentMemory ×4, ObjectMemory ×2, WorldGeometry ×2), none in either lane's code |
| iOS Release build (Simulator) | clean, same 8 warnings |
| iOS Debug build, `generic/platform=iOS` (arm64, signed) | clean |
| iOS unit tests | **812 passed, 0 failed** (794 after the merges, +18 from this lane) |
| iOS UI smoke (Simulator, real Tower on this branch) | **3 passed** — see §6 |
| Tower suite, merged branch | 2434 passed, 89 failed, 93 skipped, 1 xfailed |
| Tower suite, unmodified base, same Mac | 2373 passed, **the same 89 failed**, 88 skipped |
| Tower suite delta | **0 new failures**; +61 passing at the first full run (both lanes' new tests + 8 from this lane); 4 more route tests added with the review fixes, all passing |

The 89 Tower failures are Mac-environment only and identical on the base:
PowerShell startup-script tests, NTFS junction tests (`cmd /c mklink`),
Document Memory OCR/page-detection (OpenCV rendering differs on macOS),
Object Memory OWL-V2 / imagery (no model, no torch), World Builder pose-accuracy
numerics, `test_scene_dependency_truthfulness` (no torch on this host). The
Windows lanes reported these suites green on Windows; this Mac cannot confirm
GPU or Windows-specific behaviour and does not claim to. Skips are skips.

## 2. Source branches and what was verified before merging

| Lane | Branch | Commits | Verified |
|---|---|---|---|
| World Builder | `origin/world-builder/global-reconstruction-v1` | `d8af694`, `2a98b9a`, `ddff465` | all three present, base `a7b1c2a` |
| CV Lab | `origin/feature/cv-lab-runtime-controls` | `75d09b0`…`bb47638` (7) | all seven present, base `a7b1c2a` |

Neither source branch was modified. Both were merged with `--no-ff`, so
their full histories are on this branch.

## 3. Merge

`git merge` reported no conflicts. Three files were touched by both lanes
and were reviewed hunk by hunk:

| File | World Builder | CV Lab | Interaction |
|---|---|---|---|
| `ios/Glasses/TowerClient.swift` | `subscribeToResults` gains optional `worldID`/`sessionID` (line ~1623) | `isFrameSendingPaused` gate, `pauseFrameSending()`/`resumeFrameSending()`, reset in `disconnect()` and `sendStreamStop()` (lines ~785, ~1224, ~1313, ~1528) | none — different members, different paths |
| `tower/tower/config.py` | `world_solve` + `TOWER_WORLD_SOLVE` | `cv_torch_threads` + `TOWER_CV_TORCH_THREADS` + `_torch_threads()` | none — independent settings |
| `tower/tower/main.py` | `--solve` appended to the builder's argv | `torch_threads` into `ExperimentSettings`; log line | none — different functions |

No merge markers, no duplicated symbols, no file from either lane missing
(`git diff --stat a7b1c2a <lane>` ⊆ `git diff --stat a7b1c2a HEAD`).

## 4. Commits on this branch (after the two merges)

| Commit | What |
|---|---|
| `f403cb7` | merge of World Builder |
| `ed7ae45` | merge of CV Lab |
| `5d5fef6` | **Tower:** `GET /worlds/{id}/render` — the interactive viewer served to the phone |
| `20ab1b5` | **iOS:** DEBUG-only `GLASSES_TOWER_AUTHORITY` override (Simulator → Mac-local Tower) |
| `ba6f1e1` | **iOS:** the picture of a saved world inside the app (`WorldRenderViewer.swift`, `renderTarget`, Picture button) |
| `60459db` | **iOS:** `GlassesUITests` smoke target; one line of CV Lab copy |
| `3d542d0` | **Tower:** review fixes — world-id containment, CSP, `<` escaping, no paths in details |
| `9967420` | **iOS:** review fixes — stale-envelope filter, WebKit recovery, override hardening, frame-hold truthfulness |
| (this doc) | handoff |

21 files, +2,277 / −421 after the merges (before this document). Nothing in either lane's files was
changed except one sentence in `ExperimentalCVWorkspaceView.swift`
(an armed experiment waiting for frames now points at the camera card
above it instead of "Start a session").

## 5. World Builder — the HTML viewer (the product requirement)

### What existed

The World Builder lane wrote `scripts/world_render.py`, an offline tool that
turns a persisted derived tree into PLY, PNG and an interactive `world.html`
(plotly if installed, otherwise a self-contained 2-D canvas orbit viewer).
It lived outside the `tower` package, imported `scripts.world_registration`
for the Sim3, and its canvas viewer answered only mouse events. No route
served it. The iOS lane had no viewer.

### What was built

**Tower (`5d5fef6`).** The composition — reading the derived tree, applying
each registered placement exactly as the store defines it
(`X_ref = s·R·X + t`), grouping segments that share a reference into one
space, keeping everything else apart — and the canvas viewer moved into
`tower/tower/world_builder/render.py`. `scripts/world_render.py` imports and
re-exports those names; its eight tests pass unchanged; it keeps the
PLY/PNG/plotly writers and the CLI. A fourth cartridge-named adapter,
`tower/tower/results/world_builder_render.py`, resolves the session (an
explicit id, else the newest session with geometry) and composes the page.
`routes/geometry.py` gains:

```
GET /worlds/{world_id}/render?session_id=<optional>&max_points=<1..200000, default 80000>
200 text/html, Cache-Control: no-store          self-contained page
404 {"detail": "..."}                            no root / no world / no session / no geometry yet
422                                              max_points out of range
```

The page gained touch (one finger orbits, two fingers pinch-zoom and pan,
`touch-action: none`), a `viewport` meta, an empty-frame message, and a
caption that says what it is — *sparse structure-from-motion output:
triangulated feature points and camera poses; not a surface, not a mesh,
not metric scale* — plus a second line when the derived tree is behind the
newest keyframes. The title and every string that reaches the page are
HTML-escaped and `</` in the JSON payload is escaped, so a hostile id
stays text (tested). Contract: `docs/contracts/WORLD-BUILDER-WORLDS.md` §4.

**iOS (`ba6f1e1`).** `WorldRenderViewer.swift`:

- `WorldRenderClient` fetches the page with `URLSession` (30 s bound,
  `.reloadIgnoringLocalCacheData`), not the web view — so a 404 arrives as
  `WorldRenderFetchError.absent(detail:)` carrying the Tower's own words,
  and every other failure has a typed case with a truthful sentence and a
  retryable flag. The id is percent-encoded as **one** path component by
  hand: a test showed `appendingPathComponent` letting `a/b` through as two
  segments.
- `WorldRenderWebView` is a `WKWebView` that receives the page as a string
  with **no base URL**. `WorldRenderNavigationPolicy` allows exactly the
  initial `about:blank` navigation and cancels everything else — links,
  `window.location`, `file:`, `about:srcdoc`. Data detectors off, link
  preview off, back/forward gestures off, scroll view disabled so the
  canvas owns every touch.
- `WorldRenderViewerView` is a sheet: loading → page, or the Tower's
  sentence with *Try again*. Its caption repeats what the picture is.
- `WorldBuilderViewModel.renderTarget` names the world the picture can be
  opened for: set by `open(worldID:sessionID:)` (the person chose it), set
  by geometry coordinates naming the live world (guarded on equality so the
  2 s heartbeat does not republish), cleared by `returnToLive()`. The
  **Picture** button beside *Saved worlds* is disabled until then, not
  hidden. The sheet captures its target at the tap.

The native connected-world canvas (`WorldFragmentsView` /
`InteractiveWorldCanvas`, the WB lane's work) is untouched and remains the
in-workspace, per-cluster, truthful-per-tile view; the HTML picture is the
whole-world orbit view of the same derived tree. They read the same store
through different routes and neither depends on the other.

### Verified

- Tower: 12 route tests (`tests/test_world_builder_render_route.py`) —
  self-contained page, correct segments drawn, stale caption, session
  default, every 404 wording, point-budget bounds, script-block and `<`
  escaping, CSP, world-id containment, no paths in details.
- iOS: 13 unit tests (`WorldRenderViewerTests`) — address building and
  encoding, every fetch outcome, model state machine, navigation policy,
  `renderTarget` lifecycle.
- Simulator, real Tower (§6): the picture loads, draws two composed
  segments with 19 camera frusta, orbits on a swipe and zooms on a pinch,
  closes, and *Back to live* returns.

## 6. Simulator validation against a Tower on this branch

A Tower from this worktree ran on the Mac (`127.0.0.1:8010`,
`TOWER_WORLD_ROOT` = a synthetic fixture world under `Glasses-scratch/wb-cv-sim/`,
`TOWER_WORLD_AUTOBUILD=false`). The app was launched in the Simulator with
`GLASSES_TOWER_AUTHORITY=127.0.0.1:8010` (DEBUG-only override, `20ab1b5`).
`GlassesUITests` (`60459db`) drove it; screenshots are attached to the
result bundle at every step.

| Test | Steps proven | Result |
|---|---|---|
| `testASavedWorldsPictureOpensInsideTheApp` | Cartridges → World Builder → Picture disabled → Saved worlds lists the Tower's world with 2 sessions → tap world → "Looking at saved world …" → Picture enabled → sheet → page title and caption rendered inside the `WKWebView` → swipe orbits, pinch zooms, sheet stays → Close → Back to live → header clears | passed |
| `testASessionWithoutGeometrySaysSo` | open the session tagged *no geometry* → Picture → "The Tower has no picture for this world yet: session … has no geometry yet." with *Try again* | passed |
| `testTheLabConnectsAndSwitchesExperimentsFromItsOwnScreen` | Cartridges → Experimental CV Lab → Tower row *Connected* at `127.0.0.1:8010` → camera card *Stopped*, Start disabled, "Waiting for the glasses to become active." → catalog listed → tap Edge detection → "Edge detection · armed" → tap Baseline → "Baseline · armed" → Pause → "Paused." → Resume → Stop → "Stopped." → Disconnect → *Disconnected* → Connect → *Connected* → catalog back | passed |

The Tower's own record after the CV Lab test: `GET /cv-lab` showed
`run_id …-3` (startup baseline, edge, baseline), `origin: client_request`,
`lifecycle.state: stopped`, `arm_ms: 1.2`, and the new `process` block.
The Tower process was never restarted and its log shows one `/ws` accept
per connect. The 404 for the no-geometry session and the 200 for the
render were also checked with curl (109 KB, 22 ms).

To rerun:

```sh
# Tower, from this worktree's tower/ (any free port; 8000 is taken on this Mac)
TOWER_WORLD_ROOT=<a world root> TOWER_CAPTURE_ROOT=<dir> TOWER_WORLD_AUTOBUILD=false \
  python -m uvicorn tower.main:app --host 127.0.0.1 --port 8010
# UI smoke
cd ios && TEST_RUNNER_GLASSES_UITEST_TOWER_AUTHORITY=127.0.0.1:8010 xcodebuild \
  -project Glasses.xcodeproj -scheme Glasses \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -only-testing:GlassesUITests test
```

Without the variable the three tests report **skipped**, never passed.

## 7. CV Lab — what the Mac can and cannot say

**Compiles and behaves (Mac):** every file in the lane's §2 list; the
handoff's predicted compile risks (`import Combine` visibility,
`nonisolated static let` as a `@MainActor` init default, `@unknown default`
on `StreamState`, `Just`/`Empty` protocol-extension defaults) did not
materialise — the lane's static review was accurate. 15 new CV Lab unit
tests in `CVLabContractTests` / `TowerClientTests` pass, including the
frame-gate reset on a socket drop. The connection row, the camera card, the
experiment rows' *Asking the Tower…* state and the run panel's
Pause/Resume/Stop are all real in the Simulator against a real Tower.

**Not provable here:** anything after *Start camera* — the Simulator has no
DAT device, so the card is truthfully disabled. Frame pause/resume on the
wire, `receiving_frames` flipping, `arm_ms` under load, torch thread
budgets, the live soak: **Windows + glasses only** (§10).

## 8. Contract audit

- `GET /worlds` — `WorldLibrary.swift` decodes exactly the fields
  `world_builder_library.py` writes; `null` display name and `null`
  `ended_at` are kept as absent, never zero (7 decoder tests).
- `GET /worlds/{id}/render` — new, additive, documented (§5).
- Geometry manifest `coverage` (`WORLD-BUILDER-GEOMETRY.md` §8) — optional;
  the iOS decoder keeps unknown words verbatim and acts on none.
- `result_subscribe` `world_id`/`session_id` — keys inserted only when
  non-nil, so an unpinned subscribe is byte-identical to before.
- CV Lab `run.arm_ms`, `run.runtime.torch_threads`, `process` — additive;
  `process` is HTTP-only by design and the phone does not read it;
  `CVLabContractTests` decode payloads with and without them.
- `failed` lifecycle now also covers a startup default that failed to load;
  the phone renders the Tower's reason verbatim.
- No identifier changed. No route or field collides between the lanes.
- Independent contract review: §9.

## 9. Independent review

Three independent reviewers were given permission to challenge the work:
(A) the viewer and every Swift change on this branch, (B) the merge and
the CV Lab lane's iOS state machine, (C) the Tower render route and the
cross-lane contract audit. Every finding was verified against the code
before it was acted on. Fixes are in `3d542d0` (Tower) and `9967420`
(iOS); the suites were rerun afterwards (812 unit, 3 UI, Tower subset).

**Fixed**

| # | From | Finding | Fix |
|---|---|---|---|
| 1 | A | A result envelope for a subscription just left (a queued heartbeat) could name the old world for the Picture button — permanently for a live world with no geometry | `TowerWorldBuilderClient` drops envelopes for retired subscription ids (per socket). Mock server now numbers subscriptions like the Tower; new test |
| 2 | A | WebKit content-process death left a blank view with no way back but Close | `webViewWebContentProcessDidTerminate` reloads the string |
| 3 | A | `%`-encoded override values passed the checks and would crash the force-unwrapped `static let`s | `%` refused; both URLs built and checked before acceptance; tests |
| 4 | A | "no picture … yet" claimed a *yet* for "no world" and for a Tower without the route (FastAPI `Not Found`) | Neutral wording; `Not Found` named as a Tower predating §4 and not retryable; test |
| 5 | A | `isInitialLoad` was not one-shot | One allowed navigation per load |
| 6 | A | A cancelled fetch (sheet dismissed) reported as failure | `Task.isCancelled` guard |
| 7 | A | UI test asserted "disabled" without asserting it | `XCTAssertFalse(picture.isEnabled)` |
| 8 | A | UI tests would steer the wrong Tower in a Release build | `setUp` skips unless the DEBUG-only control is present |
| 9 | B | Pause frames could be set during *Starting*; a start that then failed never sends `stream_stop`, so the hold outlived the session into Home, which has no control that shows it | Pause offered only while `cameraStreamState == .streaming` |
| 10 | B | Run header said LIVE for up to 5 s above a card reading "held on phone" (the Tower's `receiving_frames` idles late) | `isLive` reads the hold; Lab republishes on `$isFrameSendingPaused` |
| 11 | B | The hold was invisible on every other screen | Shell camera pill reads "On · held" |
| 12 | B | Doc drift on `commandCounter` ("per connection"; it is not reset) | Comment corrected: monotonic for the client's life, deliberately |
| 13 | C | **The render route had no world-id containment guard** (backslash ids escape the root on Windows; the geometry routes guard it) | `contained_world_id` applied first; test with both spellings |
| 14 | C | `</` escaped but not `<!--`; no CSP | Every `<` written as `\u003c`; `Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'`; tests |
| 15 | C | A `FileNotFoundError` string (with a filesystem path) could reach a 404 detail the phone shows | Fixed sentence; ids in details clipped; test |
| 16 | C | Operator's `world.html` never got the BEHIND caption the phone's page gets | `current` threaded through `write_html` |
| 17 | C | Render resolved sessions from `world.json` while the listing scans directories; the engine writes `session.json` first | Union of both |
| 18 | C | `WORLD-BUILDER-IOS.md` §6 still listed the picker as unimplemented | Updated |

**Documented, not changed (out of this lane's scope or by design)**

- (C) **One derived manifest per world.** A world with two built sessions renders the older one with every placement `unbound` and the BEHIND caption — truthful, and pre-existing in the script and the geometry route alike, but `GET /worlds` advertises `has_geometry: true` for both and the phone can now open either. Recorded in `WORLD-BUILDER-WORLDS.md` §4 rule 5; the fix is a per-session manifest in the store.
- (A) The navigation policy governs navigations, not subresource loads; the CSP on the response (fix 14) now covers those in the browser, and the doc comment says which is which.
- (B) `Disconnect` on the Tower row ends a hold (by design, `TowerClient` docs) while `connect()`-while-online keeps it; noted, not changed.
- (B) Home's "Sent to Tower" rate is a session-cumulative average, so during a hold it decays rather than reading 0. The shell pill now says "held"; making the rate windowed is a `SenderMetrics` change for a later lane.
- (B) Two "Pause" controls on the CV Lab screen (frames vs. run). Labels differ; left as is.
- (C) `Sim3`/quaternion decode exist in both `tower/world_builder/render.py` and `scripts/world_registration.py`, byte-identical and unpinned; the script should import the package's. Not done here to keep the WB lane's registrar untouched.
- (C) CV Lab wording "the CV Lab's last start failed" is now also used for a frame crash (`lab.py`, mirrored on iOS); the reason string disambiguates. Both lanes' files; left for the CV Lab lane.
- (C) `max_points` floors each segment at one point, so it is a budget, not a strict cap. As the contract says.
- (A) `WorldRenderTarget.id` is not injective for ids containing `/`; ids never do.

**Confirmed correct by the reviewers** (verbatim where it matters): the merge is textually and semantically clean, both feature trees are subsets of HEAD; the CV Lab pending-command machine cannot wedge (every clear path enumerated); `request_id` handling matches the Tower on every reply path; the frame gate sits after the online/bracket guards and before the stall test and is reset correctly across a socket gap; the composition in `render.py` is logic-identical to the WB lane's script; no XSS reachable; the render cost is 0.39 s for a 300k-point world at the mobile budget, off the event loop; all World Builder and CV Lab contract identifiers are identical on both sides and every additive field decodes or is ignored as intended; no coupling between the cartridges.

## 10. What requires the physical session (Windows Tower + iPhone + glasses)

Exactly the two checklists from the mission, with the readings to record.
Set-up once, on the Windows box, from a checkout of **this branch**:

```powershell
cd <checkout>\tower
# .env: TOWER_CAPTURE_ROOT and TOWER_WORLD_ROOT as before.
#   For the CV Lab list:  TOWER_WORLD_AUTOBUILD=false   (no follower per camera start)
#   For the World Builder list: TOWER_WORLD_AUTOBUILD=true, TOWER_WORLD_SOLVE=true (default),
#   pycolmap installed in the venv (pip install .[sfm]) or the solver reports itself absent.
& .venv\Scripts\python.exe -m uvicorn tower.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

Boot log should read `CV Lab startup default is 'baseline' on device 'auto' with torch threads 'auto'`.
The phone build is a Release or Debug build of this branch pointed at the
Tower's Tailscale address (`TowerConfiguration.defaultAuthority`); the
override is not needed and Release never reads it.

**World Builder**

1. Start Tower. 2. Connect the iPhone; shell pill *Connected*.
3. Open World Builder; header shows no "Looking at saved world"; Picture
   disabled until the Tower names geometry.
4. Confirm live state: capture control reads *Start capture*.
5. Saved worlds → the list shows every world under `TOWER_WORLD_ROOT`,
   newest first, `live` badge only on a world a builder holds.
6. Select a reconstructed world (the 2026-09-06 walk).
7. The header names it; the native canvas draws its clusters with coverage
   captions; the status subscription is pinned (no live counters move).
8. Picture → the sheet loads (expect a few MB for a 400-keyframe walk;
   30 s bound) and draws the composed world frame first.
9. Confirm the page's frame selector lists the world frame and each
   unregistered segment separately, named as such.
10. Orbit with one finger, pinch to zoom, two-finger pan; select a
    different frame; nothing navigates away.
11. Confirm the caption reads sparse SfM points/poses, not a surface; if
    the world is mid-solve, the "BEHIND the newest keyframes" line appears.
12. Close → back on the workspace, still inspecting.
13. Back to live → header clears, Picture disabled again until the next
    geometry report.
14. Start capture; walk; confirm Picture enables once the Tower reports
    geometry, and the picture of the *live* world opens.
15. Stop; wait for the final solve (30–135 s on the Windows host); Saved
    worlds lists the new world; open it; the picture shows one frame for
    the registered majority.

**CV Lab** — the 20 steps of the mission, which are the CV Lab handoff's
own §3 list with these additions: at step 4 note that the card reads
*Streaming* and `receiving_frames` is true; at step 15 confirm the card
reads *Paused (held on phone)*, the Home viewfinder still updates, and
within ~5 s `receiving_frames` is false while `lifecycle.state` stays
`running` and `run_id` is unchanged after Resume. Record `arm_ms` for each
switch and the process counts from the handoff's §4 commands.

## 11. Known issues and risks

1. **World Builder followers vs. shutdown grace (shared infrastructure,
   unchanged by all three lanes).** `capture_workers.py` gives a worker
   10 s at shutdown before terminating it, and the builder has no stdin
   stop channel. The World Builder lane's `--solve` (now on by default,
   `TOWER_WORLD_SOLVE`) adds a final in-process solve of 30–135 s after
   Stop. The merged branch therefore widens the window in which a Tower
   shutdown kills the builder mid-work from ~20 s to up to ~155 s after a
   capture closes; the merged solution from the last background solve
   survives, the final one is lost. This is the same risk the WB handoff
   names in §6, not a new mechanism, and it is **not** attributable to the
   CV Lab. Mitigation for CV Lab sessions: `TOWER_WORLD_AUTOBUILD=false`.
   Fix belongs to a shared-infrastructure lane (stdin stop channel for the
   builder, or a longer grace when a solve is in progress).
2. **Inference on the event loop** (CV Lab handoff §5) — unchanged.
3. **Startup load timeout is terminal** (CV Lab handoff §5) — unchanged.
4. **Eight pre-existing Swift warnings** (main-actor isolation in
   DocumentMemory/ObjectMemory/WorldGeometry) — not touched; they predate
   both lanes and are outside this branch's scope.
5. **Port 8000 on this Mac is held by a VS Code helper**, and a uvicorn
   Tower from 2026-08-27 (pid 33655, port 8765, another session's) is still
   running; both were left alone. Use another port for Mac Towers.
6. **The render is composed per request.** ~22 ms for the fixture; a
   400-keyframe walk at the 80 k point budget is expected well under a
   second, but it runs on the web process's threadpool. If a world is
   opened repeatedly during a live walk, that is one composition per tap,
   never per heartbeat.
7. `pycolmap` is not installed on this Mac; the global solver's tests that
   need it skip here, as they are designed to.

## 12. Temporary resources (filesystem policy rule 9)

- Worktree `Glasses-worktrees/wb-cv-ios-validation-v1` (this branch). Persistent.
- `Glasses-scratch/wb-cv-venv` — the Tower venv. Disposable.
- `Glasses-scratch/wb-cv-sim/` — `make_fixture_world.py`, the fixture
  world root, an empty captures dir. Disposable.
- Session scratchpad under `/private/tmp/claude-501/…/scratchpad` — derived
  data, logs, result bundles, screenshots. OS temp.
- The Tower on `127.0.0.1:8010` was stopped at the end of the session.
- Nothing written under the canonical checkout, `~`, or `tower/data`.

## 13. Recommendation

Merge `integration/wb-cv-ios-validation-v1` into `integration/all-current-v1`
**after** the physical session in §10 has run at least the World Builder
steps 5–13 and the CV Lab steps 1–7 on hardware. Everything a Mac can
prove is proven; what remains is the camera path, which only the glasses
can exercise.
