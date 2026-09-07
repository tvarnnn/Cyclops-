# All cartridges — Mac integration and validation

**Date:** 2026-09-07.
**Lane:** integration and validation. Not feature development.
**Branch:** `integration/all-cartridges-v1`.
**Worktree:** `/Users/tristan/Projects/Glasses-worktrees/all-cartridges-v1`.
**Toolchain:** macOS (Darwin 25.5.0); Xcode 26.6 (17F113), Swift 6.3.3, iOS SDK
26.5, Simulator iPhone 17 Pro; Python 3.12.5.
**Plan:** `docs/superpowers/plans/2026-09-07-all-cartridges-mvp-integration.md`.

**No iPhone and no glasses were attached at any point in this lane.** Nothing
below claims a physical step. §22 gives the verdict and §23 the physical test
that is now owed.

---

## 1. Starting point and source lanes

Base: `origin/integration/wb-cv-ios-validation-v1 @ b10ab36`, which already
carried CV Lab, World Builder, Object Memory, the product shell and the shared
Tower/iOS infrastructure — and carried Document Memory and Scene Understanding
as complete but effectively unreachable cartridges, which is what two of the
four lanes rewrote.

| Lane | Branch | Expected HEAD | Actual HEAD | Commits |
|---|---|---|---|---|
| World Builder | `fix/world-builder-live-history-lifecycle-v1` | `087dcaa` | `087dcaa` ✔ | 10 |
| Object Memory | `fix/object-memory-runtime-v1` | `70fade7` | `70fade7` ✔ | 9 |
| Document Memory | `feature/document-memory-v1` | `beadb56` | `beadb56` ✔ | 10 |
| Scene Understanding | `feature/scene-understanding-v1` | `6cfe1d5` | `6cfe1d5` ✔ | 14 |

All four matched. The worktree was clean at the start and the branch was
`integration/all-cartridges-v1`.

## 2. Ancestry

The lanes are **not** independent, and the shape is not what any one handoff
describes from its own side.

```
        ┌── 087dcaa  WB
        │
  12e4f1e ── 6beaf57 ─┬── 70fade7  OM
        │             ├── beadb56  DM
        │             └── 6cfe1d5  SU
        │
  b10ab36  HEAD
```

| Pair | Merge base |
|---|---|
| OM × WB, DM × WB, SU × WB | `12e4f1e` |
| DM × OM, SU × OM, DM × SU | `6beaf57` |
| every lane × HEAD | `b10ab36` |

`b10ab36` is the merge base for all four, so each is an ordinary three-way
merge and none is an ancestor of another.

**The trap this avoided.** `12e4f1e` (the World Builder atomic-replace retry)
is an ancestor of **all four** lanes, and `6beaf57` (a docs-only handoff) of
three. So `tower/tower/storage.py`, `tower/scripts/world_build_session.py` and
two test files appear in all four lane diffs against `b10ab36` — which reads
as four lanes editing one file and is nothing of the kind. Netting the shared
history out first is what left the real overlap in §5.

**Nothing was omitted.** The only remote branch not already merged into HEAD,
besides the four lanes, is `origin/rescue/canonical-worktree-snapshot-2026-08-29`,
a rescue snapshot rather than an MVP lane.

## 3. Merge order, and why

Simulated first with `git merge-tree --write-tree`, which creates dangling
objects and touches no ref and no working tree:

| Order | Conflicts per step | Total |
|---|---|---|
| **WB → OM → DM → SU** | 0, 0, 0, **4** | **4** |
| WB → OM → SU → DM | 0, 0, 1, 4 | 5 |
| OM → WB → DM → SU | 0, 0, 0, 4 | 4 |

**Chosen: WB, OM, DM, SU**, each with `git merge --no-ff`. All four lane
histories are preserved; nothing was squashed and no lane was rebased.

1. **WB first** — the deepest change to shared infrastructure: a new
   `tower/tower/process_ownership.py`, one owned process per worker, per-worker
   Job Objects, `request_stop`/`stop_policy` on the supervisor and the cartridge
   session, and the World Builder gate in `main.py`. Everything else sits on
   top of that process model rather than being merged under it.
2. **OM second** — the smallest lane, and the other half of the
   `object_memory_session.py` collision, so the two competing producer fixes met
   while that context was fresh. Its listener and WebSocket changes land before
   two more cartridges start subscribing.
3. **DM third, SU last** — Scene Understanding edits `_document_session` and
   `build_live_cartridges`, which Document Memory owns. Landing DM first
   reconciles SU *against* a finished DM rather than the reverse, and
   concentrates every conflict in the final merge.

## 4. Merge conflicts encountered

Four, all at the Scene Understanding merge, exactly as forecast. All four were
unions.

| File | The collision | Resolution |
|---|---|---|
| `tower/tower/config.py` | Both lanes add a local to `get_settings` at the same anchor line (`document_enabled` / `scene_mode`) | Both kept |
| `tower/tests/test_result_channel_isolation.py` | Each lane switched off **its own** cartridge so "declared and unavailable" stayed reachable after its default flipped | **Both** switches — see below |
| `tower/tests/test_result_channel_protocol.py` | Same | **Both** switches |
| `docs/contracts/TOWER-UNIFIED-CARTRIDGES.md` | Two cartridges' declaration rows, and Scene's lifecycle paragraph | Scene's row (tri-state) + Document's row (2026-09-07 id); Scene's demand-bound paragraph replaces the stream-bound one |

The two test conflicts are the clearest example of what a merge cannot see.
Each test covers **both** cartridges, and **both** defaults flipped in this
release, but each lane could only see its own — so each deleted half the fix
the other needed. Taking either side alone would have left a test asserting
"unavailable" against a cartridge that is now available. Both switches are
required and neither lane could have known it.

For the contract document, two sentences of the replaced paragraph were never
about Scene Understanding at all — that the phone opens no cartridge over the
socket, and that stream ownership is a set of connection tokens — and are
restated after the new bullets rather than lost with the paragraph.

## 5. Semantic conflicts found where Git merged cleanly

These are the ones that matter, because nothing flagged them.

**1. The producer carried two fixes and one import twice.** World Builder
(`f49d73d`) and Object Memory (`c85effc`) both fixed
`tower/scripts/object_memory_session.py`, differently and complementarily: WB
made the stop watcher read the raw descriptor, OM added the OpenBLAS pre-warm
and a hard exit on every path. Git merged both cleanly and correctly — the
pre-warm still runs *before* the watcher is armed, which is the ordering the
deadlock fix depends on — but each lane had added `import os` at a different
place in the block and both survived. Removed.

**2. Document Memory's new default root had no test isolation.** DM gave the
cartridge a default root, which is the right product decision and the same one
Object Memory made a fortnight earlier. Object Memory's came with an autouse
conftest fixture repointing the default at a temp directory, because a test
that unsets the variable otherwise reads whatever the developer's own checkout
has accumulated — a privacy defect before it is a flake, and invisible in CI
because `data/` is gitignored. That fixture was not carried across. The hazard
is larger for this cartridge: what Document Memory persists is the *text of
pages a person held in front of their face*. Added.

**3. The World Builder gate broke a harness that predated it.**
`scripts/unified_cartridge_smoke.py` opened a stream and waited for a builder.
Since 2026-09-06 a builder attaches only while the World Builder cartridge
session is active. The check now asserts the gate **in both directions** — a
stream on its own attaches nothing, and a builder appears once the session the
phone opens is active — because the old check could not tell a working gate
from a broken one.

**4. A process-wide thread assertion that only the full suite could break.**
`test_scene_live_session.py` proved "a second start begins no second worker"
with `threading.active_count() < 50`. It passed alone and failed at 54 inside
the full suite on a host with torch, counting threads no session started. It
measures the delta across the second start now, which is both what it claims
and a tighter bound.

**5. A Swift test that could never have passed.** Document Memory's
`testSightingsDecodeOnADocument` unwrapped the first row of a listing built
from the **empty** library fixture, and the sibling one-document fixture puts
its record under the singular `document` key, which the decoder reads
elsewhere entirely. Written on Windows, never executed; this lane was its first
run.

**6. The Scene screen's subscription never actually closed.** §17.

**7. Document Memory's camera claim died with its screen.** §17.

**8. A cartridge session could outlive every client.** §17.

## 6. Integration fixes made

Each is a separate commit with its reasoning in the message.

| Commit | What |
|---|---|
| `a10950a` | The World Builder soft-stop tests close stdin themselves, so `communicate()` must not flush it. Two tests failed on macOS with `ValueError: I/O operation on closed file` and buried it behind an `AttributeError` on the retry. Nine lifecycle tests now pass. |
| `a5f187a` | One `import os` in the merged producer; the same closed-pipe fix in `test_object_memory_graceful_stop.py`, which failed the same way on the base as after the merge. Twelve tests now pass, so the merged producer's stop path is exercised on macOS rather than skipped. |
| `9ef62db` | The Document Memory root isolation fixture, and the font fallback (§9). |
| `5589f42` | Two Swift compile errors, the impossible Swift test, and the Scene dispatch defect. |
| `c07ca8e` | Document Memory's camera claim now outlives its screen. |
| `381f98b` | The unified smoke opens the World Builder session and proves the gate both ways. |
| `2e78e8e` | `scripts/all_cartridge_switch_soak.py` — five cartridges on one Tower (§14). |
| `322e7ac` | The single-person disclosure reaches the wearer, not just the wire. |
| `c07ce06` | A cartridge session does not outlive every client. |

## 7. The combined architecture

**Five cartridges, four activation disciplines.** This is the integration risk
in one sentence, and it is deliberate rather than untidy — each discipline
matches what its cartridge costs.

| Cartridge | Activated by | Holds | Released when |
|---|---|---|---|
| CV Lab | `cv_lab_start` on the socket; processes frames inline | the armed experiment | Tower shutdown |
| World Builder | a `world_builder` `CartridgeSession` the phone opens with its workspace | one owned subprocess per worker, in its own job, plus solve children | the session stops (`stop_policy: "request"` — asks, and a finalizing builder is allowed to finish) |
| Object Memory | an `object_memory` `CartridgeSession` | one owned subprocess | the session stops (`terminate`, bounded by a 3 s grace after a flush) |
| Document Memory | a person starting a session; **never** the stream | an EasyOCR reader on a parked worker thread | Stop, or ten minutes after its last stream closes |
| Scene Understanding | demand: somebody streams **and** somebody watches | a detector and its model | the last watcher leaves, or the last stream closes |

Two things were reconciled rather than left duplicated: the producer's two
stop fixes are now one coherent path (§5.1), and cartridge sessions now end
when the last client goes, which closes the same hole for both gated
cartridges with one change instead of two (§17).

Two abstractions remain deliberately separate — `CartridgeSession` for
subprocess workers, `LiveSession` for in-process ones. They are not
duplicates: one supervises a process and one supervises a thread and a model,
and merging them was judged out of scope for an integration lane. §18 records
it as a known limitation rather than pretending it is finished.

## 8. Debug build

`xcodebuild -project Glasses.xcodeproj -scheme Glasses -configuration Debug
-destination 'platform=iOS Simulator,name=iPhone 17 Pro' build`

**Clean.** One compile error had to be fixed first: `WorldBuilderSessionController.swift`
declared an `ObservableObject` with `@Published` while importing only
`Foundation` and `os`, and this project's settings require `Combine`
explicitly, as every other client in the app does.

**A predicted error that was not real.** Static review rated a default value
on an enum case's associated value (`WorldModel.swift:476`) a high-confidence
compile error, with four dependent call-site errors. It compiles in this
toolchain. It was left alone. The compiler is the authority, which is why it
was run before anything was changed on that account.

## 9. Release build

Same command with `-configuration Release`. **Clean**, 8 warnings — the same 8
the previous Mac lane recorded on the base (DocumentMemory ×4, ObjectMemory
×2, WorldGeometry ×2). **No new warnings.** `SWIFT_COMPILATION_MODE =
wholemodule` on Release, so neither configuration hides an error from the
other.

## 10. Swift unit tests

`-only-testing:GlassesTests`, Simulator iPhone 17 Pro.

Two failures had to be fixed first, and both were defects rather than noise:
`SceneUnderstandingAdditionsTests` carried no `@MainActor` unlike every
sibling class in its file, so it could not call the decoder, the view or the
client it drives; and Document Memory's sightings test unwrapped a row from an
empty fixture (§5.5).

**Final: 873 executed, 0 failures.**

## 11. UI smoke

`GlassesUITests`, driven against a real Tower built from this branch.

The CV Lab test passed unchanged. The two World Builder tests needed updating,
and the reason is a deliberate behaviour change rather than a regression.

**What changed.** Opening a world *without naming a session* asks the Tower
for its `latest` selection, which is the most recently updated session. In the
fixture that session has no geometry, so there is no picture and the control
is correctly disabled. The old test tapped the world row and demanded a
picture — which the old build produced by drawing a *different* session's
geometry under this session's name. That is exactly the "geometry that is not
this world's, presented as if it were" defect the World Builder lane set out to
close, so the test was pinning the bug. It now opens a session that has
geometry, which is what a person does.

The second test expected to open the picture on a session with no geometry and
read the Tower's 404 prose. That sheet is now unreachable for such a session,
deliberately: the Tower says `geometry.available: false`, the app names no
render target, and the control is disabled — "a disabled one says 'not yet'
truthfully". Offering a button whose only outcome is a 404 was the weaker
behaviour. The test asserts the better one.

**The fixture also had to be corrected, and that is a finding in itself.** The
world root the previous Mac lane generated has no events journal and no
finalization record, so the Tower reports its "complete" session as `idle` —
correctly, since `ready` is derived from a `session_stopped` event plus a
complete finalization. A corrected copy was built under
`Glasses-scratch/ac-tmp/uifixture-world` (the original was not modified) and
the same session then reports `finalized` / `ready`. **Any fixture world built
before 2026-09-06 will read as `idle` to this Tower**, which the physical test
should not mistake for a defect.

**Final: 3 passed, 0 failed**, against a Tower from this branch with the
corrected fixture world.

## 12. Tower full suite

Run from this worktree in two environments, because which one you use changes
what the numbers mean.

| Environment | Result |
|---|---|
| **Baseline, unmerged `b10ab36`**, no torch | 2446 passed, **90 failed**, 93 skipped, 1 xfailed, 2 errors |
| Merged, no torch | 2796 passed, 25 failed, 99 skipped |
| **Merged, with CPU torch + torchvision + transformers + EasyOCR** | **2836 passed, 15 failed, 84 skipped** |

The baseline was run **first**, on the unmerged base, precisely so that every
later failure could be attributed rather than argued about. The 90 baseline
failures are all macOS-environment: no PowerShell, no NTFS junctions, no
torch, no EasyOCR, no world corpus, and World Builder pose numerics. The
previous Mac lane recorded 89 of the same on the same base.

**Against that baseline, the merged branch with torch introduced exactly one
new failure at any point** — the thread-count assertion in §5.4 — which is fixed. In the
torchless environment two further failures appear
(`test_scene_capability::test_a_host_with_the_ml_extra_offers_it_by_default`
and one Document Memory privacy test) and both are the absence of torch
saying so; neither appears in the torch environment.

**What macOS honestly cannot exercise, and is reported as skipped, not passed:**

- `test_serve_loop.py` — the resilient Proactor loop for CPython gh-93821 is
  Windows-only and `skipif`s on `sys.platform != "win32"`. **The Object Memory
  lane's listener fix is therefore unverified on this host.** It is a Windows
  bug with a Windows fix, and only the Windows box can retest it.
- The Job Object cases in `test_process_ownership.py` — `skipif(os.name != "nt")`.
- `test_startup_scripts.py` — needs PowerShell.
- The NTFS junction case in `test_world_builder_geometry_transport.py`.
- The real-corpus tests, which need `tower/data` from the canonical checkout.

## 13. Per-cartridge targeted results

| Cartridge | Suites | Result |
|---|---|---|
| World Builder | finalization, lifecycle, interrupted world, solve cadence, autostart e2e, process ownership, stop request, result-channel lifecycle states | 132 passed, 3 skipped after the harness fix; the 9 lifecycle tests all pass |
| Object Memory | worker startup, transport, graceful stop, lifecycle, result-channel disconnect | pass; 12 graceful-stop tests now execute on macOS for the first time |
| Document Memory | config, gate, identity, segments, live, search, retrieval, wire e2e, privacy, engine, hostile, CLI, detect | 74 failed → **1** after the font fix; the 1 is a load-sensitive flake that passes alone |
| Scene Understanding | `-k "scene or documented"` | **326 passed, 2 skipped** — the same count the lane reported on Windows |
| CV Lab | covered by the unified smoke and the switch soaks | pass |

**The Document Memory result deserves its own sentence, because it looked like
a product failure and was not.** 29 tests failed on macOS because
`tests/document_fixtures.py` tried three font files, none of which exists on
macOS, and fell back to `ImageFont.load_default()` — a 10-pixel bitmap face
that silently ignores the requested 34 px. Every "printed page" it rendered
came out at a mean luminance of 249 of 255. The gate found no text, the
detector returned nothing, no dwell ever opened. A missing font is not an
error, it is a smaller font, which is why it read as a broken cartridge for a
whole platform.

## 14. Cross-cartridge lifecycle and soak

This is the validation none of the four lanes could perform, and the reason
this lane exists.

**`scripts/unified_cartridge_smoke.py`** (extended): **59 checks pass** without
models, **71** with them — Scene and Document really load and release their
models on this Mac, Scene's Stop discards its scene, Document's Stop keeps its
library.

**`scripts/all_cartridge_switch_soak.py`** (new): one cycle walks CV Lab →
World Builder → Object Memory → Document Memory → Scene Understanding → CV Lab
again, on one long-lived Tower. After **every** stream it asserts the rule that
matters most and that no single lane could check: **a camera streaming for one
cartridge starts nothing else.** Scene Understanding gets all three of its
cases — a stream with no watcher stays stopped, a stream with a watcher runs,
the last watcher leaving stops it and releases the model.

Six cycles, torch and EasyOCR installed:

| | first third | last third | delta |
|---|---|---|---|
| Python threads | 20.5 | 21.0 | +0.5 |
| RSS | 579.8 MB | 570.6 MB | −9.2 MB |

0 workers left, 0 strays, no lock naming a dead pid, 6 World Builder sessions
all `complete`. **VERDICT: STABLE.** Re-run after the last fix at three
cycles: threads flat at 20.0, RSS +4.8 MB, same verdict.

Frames are synthetic noise. This measures lifecycle — ownership, gating,
release, leak — and says nothing about perception.

## 15. Contract drift

`ios/scripts/contract-drift-check.py` against a live Tower from this branch:
**AGREEMENT — every contract the Tower stated is implemented by this build**,
across all eight the Tower serves, including the two identifiers that moved
(`world_builder.status/2026-09-06`, `document_memory.status|library/2026-09-07`).

`ios/scripts/cross-stack-constants-check.py`: **agreement**. It covers only
Object Memory by construction, which is a coverage gap rather than a pass for
the other four.

`ios/scripts/swift-structure-check.py`: clean. It is explicitly not a compiler
and would not have caught any of §8's errors.

## 16. Process, thread and resource observations

- Tower baseline 8 threads / 332 MB; after a first full cycle 20 threads /
  593 MB, then flat across five more cycles (§14). The step is torch's own
  pools and the CUDA-less model load, created once and never reclaimed —
  the same behaviour the Scene lane measured on Windows.
- No child process outlives the Tower, in either soak or the unified smoke,
  including when the Tower is stopped with a capture open and the socket
  still connected.
- A builder appears only while the World Builder cartridge session is active,
  and is gone within the finalize window after Stop.
- No world lock naming a dead pid, and no session left open, after any run.

## 17. Reviewer findings and disposition

Two adversarial read-only reviewers ran against the merged tree (Tower
lifecycle; privacy and truthfulness), plus a static Swift compile review before
the first build.

**Fixed.**

1. **Scene Understanding's screen never closed its subscription.** (HIGH.)
   `workspaceVisibilityChanged` was declared only in the protocol *extension*,
   not in the requirement list, while the view model holds its client as `any
   SceneUnderstandingClient`. A method that is not a requirement dispatches
   statically through an existential, so every call ran the no-op default and
   never reached the Tower-backed client. It compiles, every other test passes,
   and it silently removes the entire point of the lane: the Tower runs a scene
   session only while somebody streams **and** somebody watches, so with the
   screen never opening or closing its subscription the detector runs for as
   long as the socket lives. Now a requirement, with a regression test that
   calls it through an existential.
2. **Document Memory's camera claim died with its screen.** (HIGH.)
   `startedTheCamera` lived on a `@StateObject` that SwiftUI destroys on every
   cartridge switch. Start the camera, switch cartridge, come back: the capture
   is still running, the panel says it belongs to another screen, and Stop no
   longer stops it because the branch that would is guarded on the flag that
   was lost. Only Home could then end it. The fact now lives on a
   `CartridgeCameraClaim` that `ProjectManager` owns — which is where the
   project's own doc comment already said it belongs, and where Object Memory
   already keeps the identical fact.
3. **A cartridge session could outlive every client.** (HIGH.) A phone that
   crashed mid-session left the gate open for as long as the Tower ran, and the
   next capture from **any** cartridge attached that worker with nobody having
   asked — for Object Memory, a recorder starting itself on somebody else's
   later walk. The World Builder handoff names this hole for its own session
   and defers it as "a generic `CartridgeSession` change for a later lane";
   both gated cartridges have it and this is where the cartridges meet. Now the
   connection teardown stops every session **once `live_connections` reaches
   zero** — so a superseded connection's teardown during an iOS reconnect,
   which is what `ConnectionTracker` counts rather than flags for, stops
   nothing. A WiFi hiccup does not end a walk.
4. **The single-person disclosure never reached the phone.** (MEDIUM.) The
   privacy reviewer of the Scene lane accepted exactly one limitation — with
   one person in view the aggregates describe that person — and the acceptance
   rested on the Tower saying so in `single_person_note`. Nothing on the phone
   read it. Now decoded and shown, at a count of one, verbatim.
5. **Two Swift compile errors** (§8) and **three Swift tests that had never
   passed**, all of them written on Windows without a compiler and run here for
   the first time: Document Memory's sightings test unwrapped a row from the
   *empty* library fixture (§5.5); the Scene additions class had no
   `@MainActor` and could not call anything it drove; and World Builder's
   session-switch guard demanded an empty gallery after a switch that
   legitimately refetches, so it asserted that changing session leaves the
   screen blank. The last of these is worth its own note, because the
   production code was right and the test was wrong about its own stub — what
   "another gallery" means is that the old owner's segments were dropped and
   the new owner's fetched, and the second manifest request is that evidence.

**Accepted, not fixed, with reasons.**

- **World Builder's Stop returns while its builder is still finishing** (up to
  ~135 s). That is `stop_policy: "request"` working as designed — a builder
  finalizing a walk the wearer just finished must be allowed to — and the wire
  says so. Recorded as a limitation, not a defect.
- **Stop→Start on one continuous stream inside the finalization window** leaves
  the session `active` but unattached until the next `stream_start`.
  `following` reports `[]` truthfully, so a client can see it. Changing the
  attach path during an integration lane is riskier than the defect.
- **Two lifecycle abstractions coexist** (`CartridgeSession` / `LiveSession`).
  Real duplication of *vocabulary*, not of mechanism. Out of scope; §18.
- **`SceneLive` does not call `super()` for stream ownership.** Correct today,
  fragile later. Recorded.
- **Document Memory's idle timer does not consult multi-owner stream state**,
  and its retention prune is event-triggered rather than on a clock. Both
  self-correct on the next frame or write; neither is reachable with one phone.
- **Object Memory's `since` is server-side only.** The phone still fetches the
  whole store. The reviewer judged the labelling honest (absolute timestamps,
  "everything recorded in the window"), so this is a gap rather than a
  falsehood — but it means §23's step asking "did *this* recording capture
  anything" must be answered with `curl`, not the phone.
- **Document Memory has no in-app deletion** and does not say so. Real, and a
  product decision rather than an integration one.

**Rejected.** The static review's high-confidence enum-default compile error
(§8) — the compiler disagreed.

## 18. Known remaining limitations

Carried forward unchanged, and none of them was weakened by this lane.

- **World Builder** produces **sparse structure-from-motion**, not a dense or
  recognizable room reconstruction. It is not a mesh, a surface, a metric map
  or a complete 3D room model. A sweep of the code, the contracts and the
  Swift found no such claim; every occurrence of those words is a negation.
  Dense reconstruction is the next R&D phase and is not in this lane.
- **Object Memory** does not ship user-taught instance identity. The prototype
  and its benchmark exist only in `tower/docs/superpowers/research/`; a search
  of every `.py` and `.swift` for enrolment, embeddings or DINOv2 found
  nothing reachable from a route, a config flag or the UI. Cross-session
  re-identification measured AUC ~0.80 and the corpus has one instance per
  category, so the core promise is untestable on it. That is a decision on the
  evidence, not a deferral for want of it.
- **Document Memory** has never read a physical page through the glasses.
  Every threshold was calibrated on replay and synthetic positives, and real
  paper is lower-contrast than the renderer. No accuracy on real paper is
  claimed anywhere.
- **Scene Understanding**: people counting and object perception are LIMITED;
  orientation is EXPERIMENTAL. "Appears to be facing your direction" is not
  gaze, attention or intent, and the code, the wire and the UI all say so.
  Nothing about people has been checked on this camera, because the corpus
  contains no bystander.
- Two lifecycle vocabularies (§17); Object Memory's `since` unused by the phone
  (§17); no in-app document deletion (§17).

## 19. What could not be validated on macOS

Stated plainly, because a test that did not run is not evidence.

1. **The Windows listener fix.** `ResilientProactorEventLoop` and its
   regression test are Windows-only and skip here. CPython gh-93821 is the
   whole reason Object Memory's lane exists on that side, and this host cannot
   retest it.
2. **Job Object process ownership.** The launcher-pair and
   `KILL_ON_JOB_CLOSE` behaviour is Windows-specific. What macOS did exercise
   is the *cross-platform* half: one owned process per worker, the stop
   request, the bounded reap, and no child outliving the Tower.
3. **CUDA.** Every GPU figure in the four handoffs — OCR at 0.27 s a page,
   RT-DETRv2 at 17.8 ms, VRAM release on Stop — is a Windows/RTX 5070 number.
   This Mac ran the CPU paths: SSDLite for Scene, CPU EasyOCR for Document.
   **The detector Scene ships on CUDA was never loaded here.**
4. **The PowerShell startup scripts**, including the two new uvicorn flags.
5. **The real capture corpus** in the canonical checkout's `tower/data`.
6. **Anything involving a phone, glasses, a real page, a real person, or a real
   network.** No device was attached.

## 20. Final Git HEAD

`b6bb37a` on `integration/all-cartridges-v1` is the last commit that changes
code or tests; the commits after it add and correct this file. Every number in
§21 was re-verified at the final HEAD after the handoff landed, and none moved.

Four `--no-ff` merge commits preserving all four lane histories, and fourteen
commits of this lane's own: one plan, nine integration fixes and tests, and
the handoff. 151 files changed against `b10ab36`.

Nothing squashed, no lane rebased, no published history rewritten. All four
lane tips are ancestors of this HEAD and all four source branches still point
where they did (`087dcaa`, `70fade7`, `beadb56`, `6cfe1d5`). Nothing was
pushed and nothing was merged onward: **this branch is the candidate**.

No model weights, dataset, replay corpus, cache, virtual environment,
machine-local log or user data is committed — checked by extension and by
path.

## 21. Final validation at the final HEAD

Every number in this section was produced at the final commit.

| Gate | Result |
|---|---|
| iOS **Debug** build (Simulator) | **clean**, 0 errors |
| iOS **Release** build (Simulator) | **clean**, 0 errors, 8 warnings — all pre-existing on the base, none new |
| **Swift unit tests** (`GlassesTests`) | ****873 passed, 0 failed**** |
| **UI smoke** (`GlassesUITests`, live Tower from this branch) | **3 passed, 0 failed** |
| **Tower full suite**, with torch | **2836 passed, 15 failed, 84 skipped** |
| — new failures vs the unmerged baseline | **0** |
| `unified_cartridge_smoke.py` | **71 of 71 checks passed** (with models) |
| `all_cartridge_switch_soak.py`, 4 cycles | **STABLE**, 0 findings, 0 leftovers |
| `contract-drift-check.py` | **AGREEMENT** |
| `cross-stack-constants-check.py` | **agreement** |
| `swift-structure-check.py` | clean |

**All 15 Tower failures are in the baseline set** taken on the unmerged
`b10ab36` before any merge, and every one is macOS-environmental: World
Builder pose-accuracy and point-quality numerics (7 + 1), the PowerShell
startup scripts (3), the NTFS junction and one geometry-transport case (2),
one recovery-safety numeric, and one Document Memory provenance test. **The
merge introduced none.**

The count moved between two runs of the same tree — 14 then 15 — and the
difference is one baseline test that fails intermittently under load
(`test_live_cartridge_regressions.py::…::test_a_reading_that_spans_a_switch_is_split_rather_than_mislabelled`).
It is reported rather than smoothed over, because a number that moves is a
fact about this suite that the next person should know before they chase it.
The set-difference against the baseline is **empty in both runs**, which is
the claim that matters and the one that does not move.

**The final soak, per cartridge, over four cycles — the assertions no lane
could make:**

| Assertion | Result |
|---|---|
| a builder attaches only while the World Builder session is active | True ×4 |
| the builder finalizes and deregisters after Stop | True ×4 |
| the Object Memory producer is gone after its session stops | True ×4 |
| a stream with **no watcher** leaves Scene Understanding `stopped` | `stopped` ×4 |
| a stream **with** a watcher runs it | True ×4 |
| the last watcher leaving stops it and releases the model | True ×4 |
| Document Memory reaches `running` and its library survives Stop | True ×4 |
| **no unrelated cartridge started behind any stream** | 0 findings |
| workers left, strays, dead locks, sessions left open | 0 / 0 / 0 / 0 |

Threads 20.0 → 20.0, RSS 292 → 299 MB across the run.

## 22. Verdict

# READY FOR PHYSICAL VALIDATION

**With the limitations in §18 and the gaps in §19 stated as part of the
verdict, not as footnotes to it.**

What that means, and what it does not.

**What is established.** The four lanes are integrated deliberately rather
than merged blindly: the ancestry was established before the first merge, the
order was chosen from a simulation rather than a guess, and every conflict was
resolved as a union with the reason recorded. Three lanes' Swift met a
compiler for the first time and both configurations build clean. The combined
Tower suite introduces **zero** new failures against a baseline taken on the
unmerged base on this same host. The five cartridges coexist on one long-lived
Tower through four full rotations with nothing leaked, and — the assertion no
lane could make for itself — **a camera streaming for one cartridge starts
nothing else**, checked after every stream in every cycle. The contracts
agree, including the two identifiers that moved. Nine defects were found and
fixed here, three of which no compiler and no existing test would ever have
caught: a subscription that never closed, a camera claim that died with its
screen, and a cartridge session that outlived every client.

**Why it is READY rather than validated.** Nothing in this lane touched a
phone, a pair of glasses, a printed page, a person, or a real network. Every
claim above rests on a Simulator, synthetic frames, and a Mac. Three things
that matter most are structurally unavailable here: the Windows listener fix
(§19.1), Job Object process ownership (§19.2), and every CUDA path — the
detector Scene Understanding actually ships on was never loaded (§19.3).

**Why it is READY rather than NOT READY.** Every gate this host *can* close is
closed, and the things it cannot are Windows-and-hardware questions that only
the physical test can answer. There is no known defect being carried forward
into that test, and no result in this lane that a device run would be expected
to contradict. The three lifecycle defects that would have made a physical
test misleading — a detector that never stops, a camera that cannot be
released, a recorder that re-arms itself — were found and fixed before it.

**The verdict would be NOT READY** if any of these were still true, and none
is: a builder attaching to a CV Lab capture; the app failing to build; a
contract disagreement; a cartridge that cannot be left; a new Tower failure
the merge introduced.

§23 is the test that turns this verdict into a validated one.

## 23. The physical test

One campaign, combining the four lane plans. Windows Tower + iPhone + Ray-Ban
Meta glasses. **Build and install the phone from this branch first** — three
lanes' Swift is compiled but has never run on a device, and two contract
identifiers moved, so an older app will correctly say it does not understand
this Tower.

### Set-up

1. Start the Tower **once**, from the canonical checkout, with **no
   `TOWER_*` variables set beyond the existing `.env`:
   `cd tower; .\scripts\start_tower.ps1`. **Do not restart it again until
   step 45**, which is the deliberate shutdown. Every step between here and
   there runs on this one process: that is the product claim under test.
2. Confirm in the boot log: the uvicorn line carries
   `--loop tower.serve_loop:resilient_loop_factory` and
   `--timeout-graceful-shutdown 10`; the world root prints **absolute** with no
   "holds no worlds/" warning; `a builder will be attached to a capture WHILE
   World Builder is active`; `document root ...\data\document_memory`;
   `Scene Understanding is enabled (mode auto)`.
3. `curl http://<tower>:8000/cartridges` — `world_builder`,
   `experimental_cv`, `document_memory` and `scene_understanding` all
   `available: true`, `not_offered` empty.
4. Connect the phone. Home shows Glasses Registered, Tower Connected.

### CV Lab sanity

5. Open the CV Lab. Start the camera from its own screen. Confirm frames
   process and the run panel shows a run id.
6. **`curl /health` — no `world-build` worker.** A builder here is the gate
   failing and is a blocker.
7. Stop the Lab.

### World Builder — new world and History

8. Open World Builder. **PASS:** the canvas shows ready/empty; if a "Last
   saved world" line appears it names the newest world and says it is *saved*.
   **FAIL:** fragments drawn as "What the Tower builds".
9. Saved worlds: a list loads with a spinner first, newest first, dated
   titles, state badges, session-less shells folded under one disclosure.
10. Find `fcbca9e9…`. **PASS:** badge **Interrupted**, ~467 keyframes, opening
    it shows figures and an enabled Picture, headline says interrupted with the
    Tower's reason. **FAIL:** "World building failed", or a finished-looking
    world.
11. Open a **Complete** session → Picture → orbit/pinch → Close → Back to live.
    Header clears, gallery empties.
12. Start capture in World Builder. `/health` shows **one** `world-build`
    worker with one pid — no launcher pair.
13. Walk 2–3 minutes. Stop. **PASS:** the canvas shows **finalizing** with
    "The Tower is finishing this world." and no failure for the whole final
    solve (expect 30–135 s); then `LOCK` disappears and
    `session.json.finalization.state == complete`.
14. Saved worlds lists it as Complete; open it; Picture works.
15. `Get-Process` shows no `world_build_session` or `world_solve`.

### Object Memory

16. Open Object Memory. Press **Start remembering** once. Do not visit Home
    first.
17. **PASS within ~10 s:** the console prints the worker start, then
    `detector=ssdlite320 device=cuda verifier=owlv2`, and the panel reaches
    "remembering". **This is the zero-frame deadlock under test.** If it sits
    on "asked to remember, and not observed", capture the worker's stderr.
18. Confirm the verifier line names **fourteen** classes. Two means it did not
    load.
19. Walk ~2 minutes with a laptop, a phone and a bottle in view, ≥3 s each.
20. **This recording's own memories, over `curl`, not the phone:**
    `GET /object-memory/observations?since=<started_at as epoch seconds>`
    returns only this walk's records; the unscoped query still returns all
    history. The phone does not yet send `since` (§17), so this step is the
    one that answers "did this recording capture anything".
21. Pause — the producer leaves the process table. Resume — new observations
    continue. Stop.
22. **PASS:** the producer's report prints `stopped_because stdin-closed`,
    `observations_recorded > 0`, and the supervisor logs it *finished*, not
    `EXITED 3221225477`.

### Document Memory — a real printed page

23. Open Document Memory. **PASS:** a recent listing ("Never observed" on a
    fresh Tower is correct) and a Recording panel — **not** "has not declared a
    contract".
24. Start. **PASS:** `starting` then `running` within ~2 s (~6 s cold), panel
    shows `OCR on cuda`.
25. Hold a **clearly printed page** (12–14 pt body text, one distinctive word)
    so it fills most of the frame height, steady for 2–3 s. Small hand motion
    is fine; walking is not.
26. Look away for 2 s. **PASS within ~3 s:** the panel shows `1 recorded` and
    the listing gains a row with no tap — that is the live library revision.
27. Search the distinctive word: one result, `page_index 0`, a snippet
    containing it. Search a word not on the page: "Nothing matched".
28. Show the **same** page again for 2–3 s, look away. **PASS:** `1 recorded,
    1 seen again`; one row, "Seen 2 times"; `documents.jsonl` has one line.
    **This is the dedup claim, and it has never been tested on real paper.**
29. Show a **second, different** page. **PASS:** `2 recorded`, two rows, and a
    word unique to page two returns only it.
30. Stop. **PASS:** `stopped` within ~5 s; if a page was in view the panel
    reports `flushed_document_id`; `nvidia-smi` shows GPU memory drop.

### Scene Understanding — a real person

31. Open Scene Understanding. **PASS:** not "not enabled"; `GET /scene` shows
    `lifecycle.state: running` and `demand: {streams: 1, watchers: 1}`.
32. Point at a blank wall. **PASS:** "Nothing in view", all counts 0.
33. One person 2 m in front. **PASS within ~0.5 s:** "1 person observed",
    `partial_bottom_edge` 0, size medium. **PASS:** the single-person
    disclosure is on screen — that is this lane's fix and it has never been
    seen on a device.
34. Add a second person. **PASS:** "2 people observed" holds 30 s with ≤2 count
    changes.
35. Move a person left / centre / right. `where.person` follows; centre needs
    them within ~7° of straight ahead.
36. Have a person face the wearer, then turn away. **PASS:** "appears to be
    facing your direction" appears within ~1 s and goes within ~1 s of turning,
    and the label says **Experimental**.
37. **Leave the Scene screen.** **PASS:** `GET /scene` shows `stopped`,
    `scene_available: false`, `demand.watchers: 0`, and `nvidia-smi` shows the
    model released. **This is the dispatch fix under test (§17.1) and it is the
    single most important step for that lane** — before it, the detector kept
    running for as long as the socket lived.
38. Confirm nothing was written: no new directory under `tower/data` for Scene.

### Switching, interruption, and the end

39. Move among all five cartridges without restarting the Tower, three times
    round: CV Lab → World Builder → Object Memory → Document Memory → Scene →
    CV Lab. **PASS:** each works; `/health` shows only the worker for the
    cartridge in use; threads and RSS do not climb.
40. **Leave World Builder during a final solve.** **PASS:** the builder
    finishes on its own and the world is Complete.
41. **Walk out of Wi-Fi range and back.** **PASS:** the Tower stays reachable,
    `Get-NetTCPConnection -LocalPort 8000 -State Listen` still shows a
    listener, `/health` answers. **This is the Windows listener fix under test,
    and macOS could not check it (§19.1).**
42. **PASS:** after the reconnect the cartridge you were in is still usable.
    A brief drop must not have ended your session; a session ends only when no
    phone is connected at all (§17.3). If a short reconnect *did* end a
    recording, that is this lane's change and is a blocker.
43. **Force-quit the app while a cartridge session is active.** **PASS:**
    within a few seconds `GET /cartridges/object_memory/session` reads
    `stopped`, and starting a camera session from another cartridge attaches
    **no** object-memory producer. **This is §17.3 under test.**
44. **The Tower is still healthy after all of it, and this is checked BEFORE
    it is stopped.** `/health` 200; port 8000 still listening; `Get-Process
    python` shows no leftover worker; `nvidia-smi` back at the CUDA-context
    baseline; threads and working set not climbed across the whole campaign.
    This is the "Tower starts once" claim, and it is the last thing that can
    be asked of a running process.
45. **Only now**, stop the Tower with Ctrl-Break, during a walk, so the
    shutdown path is exercised rather than a quiet one. **PASS:** the builder
    exits within ~30 s with its session `interrupted`, no `world_solve`
    process is orphaned, and the shutdown does not hang — the bounded
    graceful-shutdown timeout is under test here, and a hang is the failure it
    was added to prevent.

### Telling a blocker from a V1 limitation

**Blockers:** a builder attached to a CV Lab capture; a producer that observes
zero frames; the Tower's listener gone after a reconnect; a session that
survives a force-quit and re-arms; a screen that leaves its detector running; a
camera that keeps streaming after a cartridge switch and cannot be stopped from
its own screen; history presented as live; an interrupted world shown as
complete or failed.

**Known V1 limitations, not failures:** flickering object counts on hand-held
things; a person count that misses people outside the 45° cone; the wearer's
own body counted as a person in a small fraction of frames; a page that must
be held close at 360×640; no in-app document deletion; the phone showing all
history rather than this recording's memories; World Builder output being
sparse points rather than a room.

## 24. Temporary resources (filesystem policy rule 9)

- Worktree `Glasses-worktrees/all-cartridges-v1` (this branch). Persistent.
- `Glasses-scratch/all-cartridges-venv` — Python 3.12 venv, `.[dev]`, no torch.
  Disposable.
- `Glasses-scratch/all-cartridges-venv-ml` — the same plus CPU torch,
  torchvision, transformers and EasyOCR. Disposable.
- `Glasses-scratch/ac-tmp/` — pytest basetemps, soak roots, Tower logs, and
  `uifixture-world` (the corrected UI-smoke fixture). Disposable.
- Session scratchpad under `/private/tmp/claude-501/…/scratchpad` — logs and
  derived reports. OS temp.
- **Nothing was written** to the canonical checkout, to `tower/data`, to
  `Glasses-scratch/wb-cv-sim` (the previous lane's fixture was copied, not
  modified), or anywhere outside `Projects/`. No process was left running.
