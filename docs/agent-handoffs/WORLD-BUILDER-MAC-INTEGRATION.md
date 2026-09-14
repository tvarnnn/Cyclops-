# World Builder: the Mac integration gate

The Windows campaign (`WORLD-BUILDER-FIELD-REMEDIATION-AND-LIVE-3D.md`) ended
**READY FOR PHYSICAL RETEST — gated on the Mac build**. This is that gate: the
Windows candidate compiled, tested, reviewed and repaired on macOS/Xcode, and
handed back as a validated candidate for the next iPhone + glasses walk.

| | |
|---|---|
| **Starting branch** | `integration/all-cartridges-v1` (canonical checkout `~/Projects/Glasses`, no new worktree) |
| **Starting SHA** | `55bc24c` (the campaign's handoff commit; code SHA `dc52b2d`) |
| **Windows candidate verified present** | `git cat-file -t dc52b2d` → `commit`; `git cat-file -t 55bc24c` → `commit`; HEAD **was** `55bc24c` on the intended branch, up to date with `origin/integration/all-cartridges-v1`. Nothing to integrate, nothing stale. |
| **Final Mac SHA** | `319172a` — §1 lists the commits, all local, nothing pushed |
| **Xcode** | 26.6 (17F113) |
| **macOS** | 26.5.2 (25F84) |
| **Simulator destinations** | `platform=iOS Simulator,name=iPhone 17 Pro` (lead); `iPhone 17` (a fix agent, to avoid contention) |
| **Tower on the Mac** | `~/Projects/Glasses-scratch/all-cartridges-venv-ml/bin/python` 3.12.5, CPU torch, **no pycolmap** — the Mac is not the Tower host; see §6 |
| **Verdict** | **§10** |

---

## 0. The one-paragraph version

**The candidate did not compile.** Two campaign commits (`f970788`) had put a
nonexistent WebKit API in the app target and a main-actor default argument in
the test target, so `xcodebuild` failed on both, and the three `tx_seq` tests
written on Windows recorded nothing when they finally ran. All of that is
fixed. Beyond compilation, seven independent reviewers (build, tests,
concurrency, UX/state, contract, reconnect, cross-cartridge) and then two more
on the repaired tree (adversarial "what breaks on the day", fresh review of
every change) found **no blocking software-side defect** in the final tree.
They did find, and this gate fixed, a family of phone-side truthfulness and
lifecycle defects the Windows host could never have seen — most importantly
that a Tower-side status-read timeout (`snapshot_failed`, which the campaign's
own result hub was rebuilt to emit) was rendered as **"World building failed
… Needs retry … walking the space again is what produces another one"** over a
walk that was fine, with no retry. The numbers are in §3. The verdict is in
§10, and it is the same word the Windows campaign used, now with the gate it
was gated on behind it.

---

## 1. Commits this gate made

All local, on `integration/all-cartridges-v1`, **nothing pushed**.

| SHA | |
|---|---|
| `339a9c1` | fix(ios): the branch that did not compile, the tests that recorded nothing, and the tasks that outlived their screen |
| `a01f864` | fix(world-builder): a Tower read timeout is not a failed walk, and the walk you just finished is not history |
| `71092a9` | fix(tower): re-entering World Builder attaches a builder, and one bad row does not empty Saved Worlds |
| `fa8861b` | docs(contracts): four claims the code did not make |
| `10ef2b9` | docs(handoff): this document, first version |
| `22ebea5` | fix(world-builder): an ack is matched by its pin, not by counting, and a foreign walk is never remembered as followed |
| `319172a` | docs(contracts): the ack echoes the request's pin, said out loud and pinned |
| _(the next SHA)_ | docs(handoff): this document, final |

`git log 55bc24c..HEAD` is the authoritative list. **Final Mac code SHA:
`319172a`** (Swift: a DEBUG log line and two comments past `22ebea5`; the
968/0 unit run and the 3/3 UI smoke below are on `22ebea5`'s code, and the
unit suite was run again on `319172a`: 968/0). Committed from the canonical checkout as the brief asked; no
commit hooks are configured in this clone, so nothing was bypassed.

---

## 2. Exact commands used

All from `~/Projects/Glasses`. `$S` was a per-session scratch directory; logs
and `.xcresult` bundles live there and are not part of the tree.

```sh
# Repository state
git status; git branch --show-current; git rev-parse --short HEAD
git log --oneline --decorate -25
git cat-file -t dc52b2d; git cat-file -t 55bc24c

# Xcode matrix (the project's documented form; no shared .xcscheme is committed,
# so `-scheme Glasses` relies on the auto-created one)
xcodebuild -list -project ios/Glasses.xcodeproj
xcodebuild -project ios/Glasses.xcodeproj -scheme Glasses -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath $DD clean build
xcodebuild ... -configuration Release ... clean build
xcodebuild ... build-for-testing
xcodebuild ... -only-testing:GlassesTests -resultBundlePath $S/unit-N.xcresult test-without-building

# Tower for the UI smoke and contract checks. Port 8000 is held by a VS Code
# helper on this machine, so 8001. No tower/.env exists; World Builder needs
# the two roots passed explicitly.
cd tower && TOWER_CAPTURE_ROOT=~/Projects/Glasses-scratch/mac-gate-2026-09-14/capture \
  TOWER_WORLD_ROOT=~/Projects/Glasses-scratch/mac-gate-2026-09-14/world_builder \
  ~/Projects/Glasses-scratch/all-cartridges-venv-ml/bin/python -m uvicorn tower.main:app \
  --host 127.0.0.1 --port 8001 --timeout-graceful-shutdown 10

# UI smoke. The TEST_RUNNER_ prefix is load-bearing: without it all three
# tests XCTSkip and xcodebuild still prints TEST SUCCEEDED.
TEST_RUNNER_GLASSES_UITEST_TOWER_AUTHORITY=127.0.0.1:8001 xcodebuild ... \
  -only-testing:GlassesUITests -resultBundlePath $S/ui.xcresult test-without-building

# Contract checks
python ios/scripts/contract-drift-check.py --tower http://127.0.0.1:8001
python ios/scripts/cross-stack-constants-check.py
python ios/scripts/swift-structure-check.py

# Tower suites (from tower/, venv python, roots under Glasses-scratch)
python -m pytest -q -p no:randomly --timeout=900 --ignore-glob="tests/*world*"   # everything else
python -m pytest -q -p no:randomly --timeout=900 tests/test_world_builder*.py tests/test_world_*.py \
  tests/test_result_channel*.py tests/test_capture_continuity.py tests/test_capture_workers*.py \
  tests/test_cartridge_session*.py tests/test_ws_*.py tests/test_metrics_rejected_frames.py \
  tests/test_ios_test_target_membership.py tests/test_storage_replace_retry.py       # World Builder + shared
```

The UI smoke fixture (one `complete` session with geometry whose record says
`final_solve: skipped`, one `unbuilt` session) was copied from the previous
lane's `~/Projects/Glasses-scratch/ac-tmp/uifixture-world/worlds` into the
world root above.

---

## 3. Results

| Gate | At `55bc24c` (as received) | Final tree |
|---|---|---|
| Debug build (Simulator) | **FAILED** — `WorldRenderViewer.swift:634: type 'WKError.Code' has no member 'frameLoadInterrupted'` | **clean**, 0 errors |
| Release build (Simulator) | **FAILED**, same error | **clean**, 0 errors, **7 warnings = the recorded baseline** (DocumentMemory ×4, ObjectMemory ×2, WorldGeometry ×1 + its `<unknown>` echo), 0 new |
| Test-target build | **FAILED** — `WorldPresentationTests.swift:319: main actor-isolated static property 'evidence' can not be referenced from a nonisolated context` | clean; test-target warnings are the pre-existing `subscribeCount` / `WorldScaleSemantics` set (blamed to commits before `223a506`), 0 new |
| Swift unit tests (`GlassesTests`) | could not run | **§3.1** |
| UI smoke (`GlassesUITests`, live Tower, fixture) | could not run | **§3.2** |
| `contract-drift-check.py` | **AGREEMENT** (13 ids implemented, 8 served, all agree) | same |
| `cross-stack-constants-check.py` | agreement (10 constants, 6 wire keys) | same |
| `swift-structure-check.py` | clean | same |
| Tower, everything but World Builder (`--ignore-glob="tests/*world*"`) | 2278 passed / 4 failed / 66 skipped | same tree for those files; the 4 = 3 PowerShell launcher tests (no `pwsh`) + 1 caused by the reviewer's own exported env var (passes unset) — **0 real** |
| Tower, World Builder + shared infrastructure | 1094 passed / 13 failed / 19 skipped | **§3.3** |
| Cross-cartridge smoke (iOS) | — | the UI smoke's CV Lab test (connect, disconnect, reconnect, switch experiments from its own screen) passed every run; the World Builder tests exercise Saved Worlds → 3D world → Details → back → Close → Back to live; no shared-infrastructure regression found by the cross-cartridge reviewer (§5) |

### 3.1 Swift unit tests

| Run | Executed | Failed | Note |
|---|---|---|---|
| first run after the two compile fixes | 942 | 4 (8 assertions) | all four were campaign-written tests running for the first time: the three `tx_seq` tests had installed a recorder hook that `respondToPing` then silently replaced, so they recorded nothing; `testAnUnchangedFinalizationIsNotRepublished` was **right and the code was wrong** — the view model republished an unchanged `finalization` on every heartbeat |
| after this gate's fixes, run 1 | 964 | 1 | `testAnUnmatchedRouteIsNamedAsSuchAndNotRetried` pinned a wearer-facing sentence this gate reworded (it cited a contract section); the assertion was updated to the new sentence |
| run 2 (same tree) | 964 | 1 | same test — stable, not load-sensitive |
| run 3 (final tree) | 965 | 0 | |
| run 4 (`fa8861b`) | 967 | 0 | |
| run 5 (`22ebea5`, the pin-aware ack fix) | 968 | 8 (4 tests) | **the three mock Towers were unfaithful**: they acknowledged every subscribe with `world_id: null, session_id: null`, where the real Tower echoes the request's pin (`routes/results_ws.py`), so the new pin rule refused the pinned tests' acks. The mocks echo the request now; the product code did not change |
| run 6 (`22ebea5`, faithful mocks) | **968** | **0** | |

No `Restarting after unexpected exit, crash, or test timeout` in any run. The
previous lane's 877 became 965: the campaign's `WorldPresentationTests` (48
tests) and the tests this gate added (§7).

### 3.2 UI smoke

| Run | Result | Note |
|---|---|---|
| first (under simulator load from a concurrent test agent) | 1 passed / 1 failed / 1 skipped | the failure was `open(cartridge: "World Builder")` on an app that took 85 s to become idle; the skip was the saved-world test finding no "Complete" row — because this gate's badge fix truthfully labels the fixture's `final_solve: skipped` session **"Partial"** |
| second (alone) | 2 passed / 1 failed | the saved-world test failed at `reveal(Back to live)`: the campaign moved Diagnostics below the fold and "Back to live" is in the top banner, and the test helper only ever swiped **up** — the same helper trap the previous lane had already documented, in the other direction. Fixed in the helper (swipe up for half the budget, then down); the assertion is unchanged |
| third (`fa8861b`, alone) | 3 passed / 0 failed / 0 skipped (160 s) | Tower healthy after; no traceback in its log |
| fourth (`22ebea5`, alone) | **3 passed / 0 failed / 0 skipped** (139 s) | same |

Screenshots from the second run are what found the duplicated "final pass was
skipped" sentence (§7, fix 14).

### 3.3 Tower World Builder suite on macOS

13 failures at `55bc24c`, all environmental, all in the recorded macOS baseline
or explained by the same cause:

| tests | why |
|---|---|
| `test_world_builder_pose_accuracy.py` ×7, `test_world_builder_point_quality.py` ×1, `test_world_builder_recovery_safety.py` ×1 | **no pycolmap in the Mac venv** — the reconstruction backend is not installed here (the previous lane's baseline lists the same nine) |
| `test_world_builder_env_check_cli.py::test_the_preflight_passes_the_backend_check_where_it_can_solve` | new in the campaign, same cause: it asserts the interpreter *can* solve, and this one cannot. On the Windows Tower host it is the check that matters |
| `test_world_builder_geometry_transport.py` ×2 | Windows junctions (`cmd /c mklink /J`) and backslash traversal — no `cmd` on macOS |
| `test_world_builder_experiment_clis.py::test_depth_temporal_consistency_requires_video_argument` | the script is run with `sys.executable` from a venv that has no editable `tower` install — recorded in the previous lane's baseline too |

**Zero World Builder failures attributable to the campaign or to this gate.**
The final rerun of the World Builder **and** shared-infrastructure set
(`test_capture_workers*`, `test_cartridge_session*`, `test_ws_*` added) on
`fa8861b` (no Tower file changed after it): **1238 passed, 13 failed, 19
skipped** — the same 13, by name.

---

## 4. Warnings reviewed

Debug and Release at `55bc24c` carried the 7-warning baseline plus **one new**
(`WorldPickerView.swift:106: value 'opened' was defined but never used`,
cosmetic, campaign commit `f970788`) — removed. No Swift 6 concurrency,
Sendable, actor-isolation, unreachable-code, deprecated-API or optional
warning was introduced by the campaign in the app target. Two warnings this
gate briefly introduced (main-actor-isolated constants read from a
`nonisolated` predicate) were fixed before commit. Four test-target warnings
of the pre-existing `subscribeCount`-in-a-`@Sendable`-closure shape that this
gate's first test helper copied were replaced with a boxed counter.

---

## 5. Reviewer findings

Nine review passes ran on this Mac, each by an agent that had not written what
it reviewed. What follows is what they found that was **real**; each item is
either fixed in §7 or recorded in §8/§9.

**Build agent.** The compile error (§0). The public `WKError` enum has no
frame-load case; WebKit reports a policy refusal under the legacy
`WebKitErrorDomain` with code 102, and neither is exported to Swift on iOS.

**Test agent.** The test-target compile error: a `@MainActor` fixture used as a
default argument, which Swift 5 mode evaluates outside the actor. Also the
`TEST_RUNNER_` prefix trap, port 8000 in use, and that the project's lack of a
shared scheme makes `-only-testing:GlassesUITests` still compile
`GlassesTests`.

**Concurrency / lifecycle reviewer** (every finding pre-existing before
`223a506` unless marked):
- `.onDisappear` sends `session/stop` while the phone's capture bracket is
  still open — §14.22 of the campaign handoff, documented, **not changed here**
  (§8).
- `start`/`stop` HTTP requests were fired concurrently and never serialized,
  so the Tower could apply them out of order across a fast leave-and-return.
  **Fixed** (chained).
- Geometry fetches: no request timeout (60 s default), no staleness check
  inside the per-segment loop, an uncancellable unstructured Task per
  coordinates emission. **Fixed.**
- `snapshot_failed` and the 10 s ack timeout terminal with no retry. **Fixed.**
- A fresh view model after a cartridge switch seeded `state` but not
  `geometryStatus` (campaign-introduced). **Fixed.**
- A navigation push cancels the picker's `.task` and a cancelled `loadWorlds`
  wiped the list. **Fixed** (single-flight, uncancellable request).

**World Builder UX/state reviewer:**
- After a normal Stop the live panel dropped to **"No world yet / Nothing is
  being built right now / Last saved world … finished [Open]"** (campaign
  §14.23, which a dress-rehearsal reviewer read as the walk vanishing).
  **Fixed**: the walk this screen just followed is presented as itself —
  "Saved", geometry addressed — when the Tower's `latest` names it.
- The `.finalizing` note said "Nothing here can see a build running" under a
  spinner saying "The Tower is finishing this world" (the campaign's one
  uncompiled structural Swift change). **Fixed.**
- Row badge "Complete · no final pass" over a canvas reading "Partial"
  (§14.34), and "Interrupted" over "Needs retry" — the invariant in
  `WorldListingPresentation`'s own docstring, broken twice. **Fixed** on the
  phone; no contract change.
- The picker caption said "No geometry was built for this walk" under a row
  whose badge said a build had run and its output was gone. **Fixed.**
- `hasGeometry || solve == .solved ? .saved : .needsRetry` (§14.18) made a
  `solved` record over counted **zeros** read "Saved". **Fixed**: the solve
  word stands in only when the figures are absent.
- No refresh in Saved Worlds. **Fixed** (`.refreshable`).
- A wearer-facing sentence cited "WORLD-BUILDER-WORLDS.md §4". **Reworded.**
- The stage table for every Tower emission was otherwise truthful; normal-UI
  wording never claims a surface, mesh or metric model (§9 of this file).

**Contract reviewer.** No blocking decode mismatch: every `model_state`,
`lifecycle.state`, `selection`, `finalization`, listing `state` and error
`reason` the Tower emits is accepted; all 13 contract ids agree on both sides;
numeric bridging is safe. Real findings: the wearer-visible wording of
`snapshot_failed` (fixed); a listing row with a non-numeric timestamp — served
raw by the Tower since round 16's sort hardening — makes the phone refuse the
**whole** listing (fixed at the producer, which the contract already made
responsible); and four stale doc claims (fixed). It also recorded what
`contract-drift-check.py` cannot see: everything below the identifier.

**Reconnect / freshness reviewer.** Every reconnect, subscribe, stall and
session-control line is pre-campaign; the campaign touched only `tx_seq` on
the phone. The phone compares neither `seq` nor `revision` and never sends
`since_revision` — so the "older result over newer" race cannot occur on the
phone at all; freshness is the Tower's guarantee and the phone correctly
trusts it. Timers and tasks are all generation-guarded. Real findings: the
give-up point of the reconnect schedule was invisible on the World Builder
screen (fixed: `reconnectGaveUp` and a sentence naming Connect); the reconnect
budget is ~16 s against a refusal and up to ~75 s against a host that accepts
TCP and never upgrades (the comment said ~45 s; corrected; §9); the
`session/start` re-ask on reconnect correctly moves `requested_at`, which is
what the Tower's 105 s follow-up keys on.

**Cross-cartridge reviewer.** Shared Tower changes verified: the grace
follow-up stops only the named World Builder walk; disconnect teardown order
is unchanged; `tx_seq` is additive; the staging sweeper cannot touch Object
Memory, Document Memory or recorder temp files. One phone holds at most 4
result targets against a cap of 8, so cartridge switching cannot hit
`too_many_subscriptions`. Real finding: **re-entering World Builder while the
previous builder is still finalizing attached nothing** — the session came up
`active` with no builder, and frames were recorded that nobody built (the
campaign's §14.22 described a second world; that is true only once the old
builder has exited). **Fixed** on the Tower. Also the `#if DEBUG` assessment
(§8).

**Last fresh reviewer, on the round of fixes above.** One HIGH in the
ack-timeout retry's second version: the superseded-ack rule counted
outstanding acks, and a pin change while a timed-out (slow, not lost)
subscribe was still answerable adopted the *live* retry's ack under the pin
and unsubscribed the pinned one — the saved-world screen would have drawn
the live world. **Fixed** in `22ebea5`: the Tower's ack has always echoed the
request's `world_id`/`session_id`; the phone now decodes them and an ack
whose pin is not the one held is closed, whatever the count says. One
MEDIUM: a `snapshot_failed` for a superseded attempt cleared a held
subscription without closing it (an orphan heartbeating for the socket's
life). **Fixed.** One LOW: the followed-walk memory was recorded before the
gate judged the report, so another phone's walk seen as "waiting" could come
back as "Saved" under Live. **Fixed.** It verified the picker's session
choice matches the render route's predicate exactly, that no writer leaves
`frame_source` null, and that each new test fails for its stated reason.

**Final reviewer, on `22ebea5` alone.** **No blocking or high defect.** One
MEDIUM that is a documentation gap with a misleading failure behind it: the
phone now depends on the ack echoing the request's pin, and the contract
never said whose values those were (a Tower echoing the *resolved* session
would satisfy the doc and be refused by the phone on every attempt, which
the client would then report as "the Tower did not acknowledge"). Verified
unreachable with any Tower in this repository's history; **fixed** by
stating it in `CARTRIDGE-RESULTS.md` §3, a protocol test that pins the echo
unpinned, by world, and by world-and-session, and a DEBUG log where an ack
is refused. Two stale comments fixed. It traced the pin rule through every
ordering of unpinned → pinned → live acks, the superseded-reply guard
against every subscribe-reply reason the Tower emits, the followed-walk
gate ordering, and the new test against the pre-commit code (fails for its
stated reason).

**Fresh reviewer of this gate's own diff.** One HIGH in this gate's first
version of the ack-timeout retry: the superseded-ack rule is count-based, so
the retry's own ack could be discarded as the superseded one and the retry
loop could never succeed for a subscribe that never reached the wire.
**Fixed** (the timed-out attempt is written off; a late ack for it is
unsubscribed by a separate rule; a test where the Tower ignores the first
subscribe pins it). One MEDIUM: the cancelled geometry task ran its first
side effects before its first cancellation check. **Fixed.** Confirmed the
WebKit predicate against the iOS 26.5 SDK headers and the simulator's WebKit
binary.

**Adversarial reviewer ("assume the next physical test fails").** Ranked list
in §9. Two fixable items were fixed: the world-row tap in Saved Worlds passed
`sessionID: nil`, so the render route (newest session **with** geometry) and
the status pin (latest session regardless) could describe different sessions
on a world whose newest walk found nothing (fixed: the row names the newest
session with geometry); and the followed-walk memory was cleared on every
socket drop, so a drop during finalization whose reconnect landed after the
lock was released fell back to the idle-with-Open shape (fixed: bounded by a
30-minute lifetime instead). It also found that a null `frame_source` on one
record would empty the picker the same way a string timestamp did (fixed at
the producer).

---

## 6. What this Mac could and could not prove

The Mac venv has no pycolmap, so nothing here reconstructed anything; every
World Builder figure this gate touched came from fixtures or the Tower's own
producers over synthetic worlds. The simulator has no DAT device, so no test
here ever sent a frame or a `stream_start` — the entire streaming path
(`#if DEBUG`, `ProjectManager`, `sendFrame`) is exercised only by unit tests
against `MockTowerServer`. What the Mac **did** prove is everything the Windows
host could not: the Swift compiles in both configurations, 965+ unit tests
pass repeatedly, the UI flow Saved Worlds → 3D world → Details → back → Close
→ Back to live works against a real Tower over a real socket, and the
contracts agree.

---

## 7. Fixes made (all with regression tests unless marked)

| # | Where | What |
|---|---|---|
| 1 | `WorldRenderViewer.swift` | the nonexistent `WKError.Code.frameLoadInterrupted` → legacy `"WebKitErrorDomain"` / 102, extracted as a testable predicate |
| 2 | `WorldPresentationTests.swift` | the main-actor default argument → an overload |
| 3 | `TowerClientTests.swift` | the three `tx_seq` tests record through `attachRecorder` (they recorded nothing) |
| 4 | `WorldBuilderClient.swift` | `finalization` is deduped before publishing (the campaign's own test was right) |
| 5 | `TowerWorldBuilderClient.swift`, `WorldCanvasView.swift` | `snapshot_failed` and the ack timeout are retried under the existing 3-resubscribe budget with a short pause; the terminal message is about the **channel**, headlined "World Builder is not reporting", never "World building failed"; the timed-out attempt is written off and a late ack for it is unsubscribed |
| 6 | `TowerWorldBuilderClient.swift` | `followedWalk`: a `latest` naming the walk this screen just followed is presented as that walk, finished — "Saved", geometry addressed — not as "No world yet"; any other `latest` is history as before; bounded by a 30-minute lifetime, session id required on both sides |
| 7 | `WorldPresentation.swift` | `.finalizing` note no longer claims nothing is running; `.finalized` + `solved` over counted zeros is "Needs retry", not "Saved"; the `.partial` note no longer repeats the record's own sentence |
| 8 | `WorldListingPresentation.swift`, `WorldPickerView.swift` | row badge "Partial" for a finished walk whose final solve was denied, "Needs retry" for an interrupted walk with nothing to open; the no-geometry caption names which kind of nothing; `.refreshable`; the world row names the newest session with geometry; the unused-`opened` warning |
| 9 | `WorldRenderViewer.swift` | wearer-facing sentence for a Tower without the render route |
| 10 | `WorldGeometryClient.swift`, `WorldBuilderClient.swift` | 30 s request timeout; staleness and cancellation checked inside the segment loop and before the first side effect; one fetch task per coordinates, cancelled on forget/pin/unpin; a fresh view model seeds `geometryStatus`; `loadWorlds` single-flight and uncancellable |
| 11 | `WorldBuilderSessionController.swift` | `start`/`stop` chained so the Tower receives them in the order the phone decided |
| 12 | `TowerClient.swift`, `WorldBuilderWorkspaceView.swift` | `reconnectGaveUp`; the capture control says "The phone has stopped trying to reconnect. Use Connect under Connections to retry."; the stale ~45 s comment; a test seam for the backoff schedule |
| 13 | `TowerSmokeUITests.swift` | the saved-world test accepts a "Partial" row; `reveal` swipes back down for the second half of its budget (a helper fix, assertions unchanged) |
| 14 | `WorldBuilderIntegrationTests.swift` | the reworded-sentence assertion; a boxed counter instead of a captured `var` |
| 15 | `ios/docs/agent-handoffs/IOS-TO-TOWER.md` | the false "reason shown beside Saved" claim |
| 16 | `tower/tower/capture_workers.py` | `attach()` on a capture whose owner is alive and `stop_requested` releases that lineage and starts a fresh builder (the finishing worker stays registered, reaped and shut down under an opaque key) |
| 17 | `tower/tower/results/world_builder_library.py`, `tower/tower/routes/geometry.py` | a session whose `started_at`/`ended_at` are not finite numbers or whose `frame_source` is not a string, and a world whose `created_at`/`updated_at` are not, are omitted with a warning instead of served; the route returns `json_safe(...)` (measured: on this FastAPI/pydantic a NaN was already `null`, not a 500; the wrap makes the guarantee the route's own) |
| 18 | `CARTRIDGE-RESULTS.md`, `IOS-TO-TOWER-RECONCILIATION.md`, `WORLD-BUILDER-WORLDS.md` | `envelope_contract` on closing errors; the `stopped_unbuilt` projection is on the figures; `mapping_seconds` is null not clamped; `max_points` default 40,000 |
| 19 | `CartridgeResultChannel.swift`, `TowerWorldBuilderClient.swift`, `WorldSession.swift` | the ack carries its pin and is matched by it; a reply to a superseded attempt while a subscription is held changes nothing; the followed-walk memory is recorded only for a report the gate let through; the three integration mocks echo the request's pin like the Tower |

Tests added: `WorldRenderNavigationErrorTests` (2), `WorldListingBadgeAgreementTests` (4),
`WorldStageMacGateTests` (3), in `TowerWorldBuilderClientTests`: a lost first
subscribe, `snapshot_failed` retried then answered, `snapshot_failed` that never
clears; in `TowerWorldBuilderLiveHistoryTests`: the followed walk presented as
itself, another session of the same world is history, an expired memory is
history; `WorldGeometryFetchLifecycleTests` (5), `WorldBuilderViewModelSeedingTests` (2),
`WorldListLoadLifecycleTests` (2), `WorldBuilderSessionOrderingTests` (1),
`TowerReconnectGiveUpTests` (2); in `TowerWorldBuilderClientTests`: a pin
change during a timed-out retry keeps the pinned subscription; Tower:
`TestAttachAfterRequestStop` (3), six listing-robustness tests. Every one was run against the code it replaced
and failed there, except the two `json_safe` pins, which are documented as pins.

---

## 8. Unresolved limitations (software-side, deliberately not changed here)

1. **A Release build streams nothing.** `sendFrame`, `sendStreamStart`,
   `sendStreamStop` and the `ProjectManager` wiring that sends them are
   inside `#if DEBUG`; `session/start` is not, so a Release phone says "World
   Builder is active on the Tower" while no capture ever opens and no world is
   created. Introduced with the frame path itself (`645e57d`, 2026-08-21),
   documented as a deliberate staging decision, an ungating was attempted and
   backed out before, untouched by the Windows campaign and by this gate.
   **Install the Debug build.** It is the first item in §11 because it is
   binary and invisible until afterwards.
2. **Leaving the World Builder screen mid-walk ends the walk** (`.onDisappear`
   → `session/stop`; campaign §14.13/§14.22). With fix 16, returning while the
   old builder finalizes now starts a **fresh** builder on the same capture (a
   second world holding the rest of the walk) instead of attaching nothing.
   Redesigning what a screen change means is a product decision, not a gate
   fix. Stay on the screen.
3. **A Stop tapped while offline never reaches the Tower** (§14.27). The naive
   resend is a no-op (a `stream_stop` from a token that does not own the
   capture is ignored); the correct fix (`stream_start` then `stream_stop` on
   the next socket) exercises a Tower path no rehearsal drove. Recorded.
4. **No `scenePhase` handling.** Backgrounding kills the socket; whether the
   DAT camera stream survives the app's suspension is unknown from this
   codebase. Do not pocket the phone mid-walk (§11).
5. **Stop, then leave the screen within the same sub-second**: `stream_stop`
   rides the websocket behind queued frames, `session/stop` is HTTP; if the
   HTTP stop lands first the builder reads an unclosed capture and records
   `interrupted`. The procedure's "wait three minutes after Stop" avoids it.
6. **`following()` lists a capture id twice** after fix 16 (the released
   worker and its replacement). No World Builder client reads the pair; noted
   for the contract.
7. **A failed `session/start` on reconnect is not retried** until the next
   reachability change; with the 105 s follow-up already fired this leaves a
   capture with no builder and a "waiting for the first update" panel.
   Recovery: leave and re-enter the screen. Two faults are needed.
8. The `.improving` note's "very different … worth waiting for" is a promise
   measured on one walk (§14.12 of the campaign). Kept.
9. The saved world is a coloured sparse point cloud, not a surface (§14.1 of
   the campaign). Every normal-surface string says so ("Not a surface, and
   not to scale"); nothing here relabelled it.

---

## 9. Known physical-only uncertainties

In the adversarial reviewer's order of likelihood × severity on the day:

1. A Release build installed → nothing recorded (§8.1). Binary. Check first.
2. The app's resolution rung (default `.low`, 360×640) vs the calibration on
   the Tower host: the pre-flight checks the rung of the **last capture on
   disk**, not the one the app will send. A mismatch is `final_solve:
   unavailable` → "Needs retry" over a walk that streamed perfectly. Not
   observable from the Mac (no frames are ever sent here).
3. Backgrounding / pocketing the phone (§8.4). 30 s away → one walk; 120 s
   away → a coin flip against the 90 s grace → two half-worlds (§14.28 of the
   campaign). Any suspension over ~16 s spends the reconnect budget; the
   screen now says so and names Connect.
4. A WiFi drop longer than the grace (§14.28 / §17.5 of the campaign) — two
   `Complete` half-worlds. Documented, unchanged.
5. Coming back on another screen inside the grace (§14.31 / §17.6). Unchanged.
6. The 179 s finalization wait and what the phone shows during it — now
   "Improving…" with the note, then **"Saved"** with "Open the 3D world"
   (fix 6), rather than the drop to "No world yet". This is the one
   procedure step this gate changed; §11 says what to expect.
7. The real-iPhone `WKWebView` render at `MOBILE_MAX_POINTS = 40,000`
   (measured on desktop only), the GLOMAP solve, the Windows junction/BOM/
   launcher paths, the Tailscale authority compiled in as
   `100.110.156.55:8000`, uvicorn's real ping timing — all Windows/phone-only.

---

## 10. Verdict

**READY FOR PHYSICAL RETEST.**

Every item of the completion bar is met on this Mac: the `dc52b2d`/`55bc24c`
candidate is the code tested (HEAD was that commit, on that branch, clean);
Debug and Release compile cleanly with only the recorded baseline warnings;
the Swift unit suite is green on repeated runs; the UI smoke passes 3/3
against a real Tower; the contract checks agree; Saved Worlds compiles and
behaves coherently in the flows the simulator can drive; a normal Stop is
represented as "Improving" and then "Saved" with no phone-side path to
"Interrupted"; the reconnect/resume Swift matches the server contract and
duplicates nothing; no supported stale-response race remains (the phone holds
no ordering state to get wrong, and the Tower's ordering was verified by the
campaign); no lifecycle/task leak the reviewers could support remains; no
shared cross-cartridge regression was found; and the last two independent
reviewers on the repaired tree found no blocking software-side defect. What
remains (§9) genuinely needs the phone, the glasses and the Windows Tower.

**Read the verdict with §8.1 in front of it.** A Release build makes every
other line of this document unobservable.

---

## 11. Physical retest procedure

**Before walking:**

1. On the Mac, build and install the **Debug** configuration from this branch
   (`git log -1` must be at or after the last SHA in §1). An older build
   refuses the Tower with *"The Tower offers a World Builder contract this
   version of the app does not understand"* and an empty Saved Worlds — both
   loud, both deliberate.
2. On the Tower host: `tailscale ip -4` must equal `TowerConfiguration.defaultAuthority`
   (`100.110.156.55:8000`). Run `scripts\world_builder_env_check.py` on the
   venv Python; every verdict `[OK ]`, and read `calibration_for_the_camera`
   knowing it checks the last capture's rung (§9.2). Start the Tower once with
   `start_tower.ps1`. It is not restarted for the rest of this procedure.
3. Connect the glasses; the World Builder screen's capture control must not
   say "The Tower is not connected".

**One walk:**

4. Open World Builder, start capture, walk 3–5 minutes through a loop (leave a
   room and come back). Watch: the panel reads "Mapping", then "Building";
   keyframes climb; the fragment count stops climbing and starts falling.
   **Stay on the World Builder screen. Do not background the phone.**
5. *Optional, only if safe:* one brief WiFi interruption of **under 15 s**,
   staying on the screen. Expected: the shell pill flickers to Disconnected
   and back; the panel resumes on the same world; one walk. If the screen
   instead says "The phone has stopped trying to reconnect", tap Connect under
   Connections within ~90 s of the drop — that is the documented budget, not a
   defect.
6. Tap **Stop capture** and **stay on the screen**. Expected, in order:
   "Improving…" with the spinner and the note that the finished world is worth
   waiting for; nothing on screen changes for about three minutes (measured
   179 s post-Stop on the field walk); then the panel reads **"Saved"** with
   "Open the 3D world" and the world's summary — not "Interrupted", not
   "No world yet". "World Builder is not reporting" during this wait is a
   channel message, not a walk failure; it clears on its own. Ten minutes
   with no change is a hang; capture `solve/<session>/solve.log`.
7. Open **Saved worlds**. The new walk's world is first; its session row reads
   **"Complete"** (or **"Partial"** if the record says the final solve was
   skipped or failed — the same word the canvas will show). Pull to refresh
   if it still reads "Finishing". Tap the session row.
8. Expected: a coloured 3D point cloud you can orbit and pinch, captioned
   "Not a surface, and not to scale" — one coherent frame, not a grid of grey
   tiles. Tap Details, then back, then Close: the workspace says "Looking at
   saved world …" with the same headline word; Diagnostics is a disclosure at
   the bottom, secondary. "Back to live" returns to the live panel.
9. Open Diagnostics → "Open the solver's 3D view": segment colours and camera
   frustums — the engineering view, still there, still labelled as such.
10. Leave World Builder, open CV Lab, then Object Memory, then Scene
    Understanding, then World Builder again. Expected: each starts without a
    Tower restart; `/health` on the Tower still `ok`; the World Builder
    session comes back `active` with a builder only once capture is started
    again.

**Expected behaviour by case:**

| Case | Phone | Saved Worlds row |
|---|---|---|
| successful final solve | "Improving…" → "Saved"; "Open the 3D world" | "Complete" |
| partial (final solve skipped/failed/unavailable) | "Partial", with the record's own sentence once | "Partial" |
| reconnect within the grace, screen kept | same world resumes; one walk; one world | one session |
| reconnect beyond the grace, or Connect tapped too late | the first half finalizes on its own; a new Start begins a second world | two sessions or two worlds, each "Complete" |
| Stop tapped while offline | "Building (live)" for up to ~2 minutes after Stop, then finishes | "Complete" |
| a Tower-side status read stalls | "World Builder is not reporting" after up to ~40 s, self-clearing on the next socket | unaffected |

**If anything fails**, capture before anything else: the world directory,
every capture in the lineage (`continues_capture` links them), the Tower
console (there is no log file), `solve/<session>/solve.log`, and the phone's
Console.app output for `category=WorldBuilder`. Note which step.

---

## 12. Temporary resources this gate created

All under `~/Projects/Glasses-scratch/mac-gate-2026-09-14/` (capture and
world roots for the Tower runs, the UI fixture, reviewers' artifact roots) and
the session scratchpad (logs, `.xcresult` bundles, derived data). Nothing was
deleted; nothing was created at `~` or `/`. The `all-cartridges-venv-ml` venv
was reused, not modified.

---

## 13. Final numbers, on `319172a`

| | |
|---|---|
| Debug build | clean, 0 errors, 0 new warnings |
| Release build | clean, 0 errors, 7 warnings (the recorded baseline), 0 new |
| Swift unit tests | **968 executed, 0 failures**, no restarts |
| UI smoke, live Tower | **3 passed, 0 failed, 0 skipped** |
| Contract checks | AGREEMENT / agreement / clean |
| Tower, World Builder + shared infrastructure | 1238 passed, 13 environmental failures (unchanged by name from the starting SHA), 19 skipped |
| Tower, everything else | 2278 passed, 0 real failures (3 PowerShell launcher tests, no `pwsh`) |

One honest note about the machine: this is an 8 GB Mac, and a background
`xcodebuild test` with a simulator, the Tower and a second simulator up was
killed for memory three times. The numbers above come from runs made one at a
time with the other simulator shut down; none of the kills was a test failure,
and every run that completed is the one reported.
