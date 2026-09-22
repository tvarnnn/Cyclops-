# All cartridges — Mac post-integration validation

**Date:** 2026-09-08.
**Lane:** independent re-validation of the integration candidate, plus bounded
UX and correctness polish. Not feature development.
**Branch:** `integration/all-cartridges-v1`.
**Starting HEAD:** `9e939a3`.
**Working tree:** `/Users/tristan/Projects/Glasses` — the canonical checkout.
**Toolchain:** macOS (Darwin 25.5.0); Xcode 26.6 (17F113), Swift 6.3.3, iOS SDK
26.5, Simulator iPhone 17 Pro; Python 3.12.5.

**No iPhone and no glasses were attached at any point.** Nothing below claims a
physical result. §15 gives the verdict, §12 what this host structurally could
not establish, and §13 the physical test that is owed.

---

## 0. A note on where this ran, and one thing that had moved

The task named the worktree
`/Users/tristan/Projects/Glasses-worktrees/all-cartridges-v1`. **That worktree
no longer exists** — it had been removed before this lane started. The task
also said, explicitly, to work directly in `~/Projects/Glasses` on the current
integration branch and to create no new worktrees. That checkout was on
`integration/all-cartridges-v1` at exactly the expected `9e939a3`, with a clean
tree, so this lane ran there.

That is a deliberate exception to the lane-isolation policy in `CLAUDE.md`,
taken on the instruction that named it. Two things make it safe here: no other
agent was working in the tree (it was clean at start and every change in it is
this lane's), and the `pre-commit` guard that would normally refuse an agent
commit in the canonical checkout is **not installed on this machine** —
`git config core.hooksPath` is empty and `.git/hooks/pre-commit` does not
exist. Nothing was bypassed and `--no-verify` was never passed. The next agent
should not read this as precedent.

---

## 1. What this lane did, and did not, take on trust

The previous lane's handoff
(`ALL-CARTRIDGES-MAC-INTEGRATION-VALIDATION.md`) is unusually careful and
almost everything in it reproduced. This lane re-ran every gate from scratch
rather than confirming numbers, and dispatched eight independent read-only
reviewers across build health, contracts, the five cartridges, cross-cartridge
coherence and lifecycle.

**One of its claims did not reproduce** (§4.1), and one of its unstated
preconditions turns out not to hold on a fresh checkout (§4.2). Both are
recorded rather than smoothed over, because both would otherwise be discovered
on the day of the physical run.

---

## 2. Validation commands run

All from `~/Projects/Glasses`, against the venv the previous lane left at
`~/Projects/Glasses-scratch/all-cartridges-venv-ml` (Python 3.12.5, CPU torch
2.14.0, torchvision, transformers, EasyOCR).

```sh
# iOS
xcodebuild -project ios/Glasses.xcodeproj -scheme Glasses -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build
xcodebuild ... -configuration Release ... build
xcodebuild ... -only-testing:GlassesTests test
TEST_RUNNER_GLASSES_UITEST_TOWER_AUTHORITY=127.0.0.1:8000 \
  xcodebuild ... -only-testing:GlassesUITests test

# Tower, started once with product-managed defaults and no .env
python -m uvicorn tower.main:app --host 127.0.0.1 --port 8000 \
  --timeout-graceful-shutdown 10

# checks
python ios/scripts/contract-drift-check.py --tower http://127.0.0.1:8000
python ios/scripts/cross-stack-constants-check.py
python ios/scripts/swift-structure-check.py
python tower/scripts/unified_cartridge_smoke.py --with-models
python -m pytest -q -p no:randomly --timeout=600      # from tower/
```

---

## 3. Results

| Gate | Result | vs. previous lane |
|---|---|---|
| iOS **Debug** build (Simulator) | **clean**, 0 errors | same |
| iOS **Release** build (Simulator) | **clean**, 0 errors, **8** warnings | same 8, none new |
| **Swift unit tests** (`GlassesTests`) | **877 passed, 0 failed** | was 873 / 0; +4 new here |
| **UI smoke** (`GlassesUITests`, live Tower) | **3 passed, 0 failed** — after §5.1e | 2 passed / 1 failed before it — §4.1 |
| Tower still healthy **before** it was stopped | `/health` ok, 0 workers, both sessions `stopped` | — |
| Tower shutdown | clean; port 8000 released, 0 strays | — |
| **Tower full suite**, CPU torch | **2836 passed, 15 failed, 84 skipped** | identical totals |
| — new failures vs. the recorded baseline | **0** | same |
| `unified_cartridge_smoke.py --with-models` | **71 / 71 checks passed** | same |
| `contract-drift-check.py` | **AGREEMENT** | same, and now honest — §5.2 |
| `cross-stack-constants-check.py` | **agreement** | same |
| `swift-structure-check.py` | clean | same |

The 8 Release warnings are the recorded set (DocumentMemory ×4, ObjectMemory
×2, WorldGeometry ×2). **Three of them say "this is an error in the Swift 6
language mode"**; the project builds at `SWIFT_VERSION = 5.0`, so they are a
future migration cost rather than a defect, and they are unchanged by this
lane.

**One thing that moved, reported rather than smoothed over.** One
`GlassesTests` run out of four aborted mid-suite — `Restarting after unexpected
exit, crash, or test timeout` — during `TowerClientTests`, which drives real
local sockets. It restarted and finished with 0 failures, but the summary then
counts only the final launch, so that run's "279 tests" is not a suite total
and must not be read as one. The immediately preceding and following runs of
the same tree completed with no restart and 877/0. It is recorded because a
count that moves is a fact about this suite the next person should know before
chasing it, and because a restart plus 0 failures still prints a total that
looks like a pass.

**One Tower served the whole campaign.** The same process started at the top
of §2 answered the contract checks, the UI smoke, and the connect/disconnect
churn of 877 unit tests, and was still healthy at the end: `/health` 200, zero
capture workers, and both cartridge sessions `stopped` with `following: []`.
That is the "Tower starts once" claim exercised incidentally rather than
asserted, and it held.

The 15 Tower failures are the macOS-environment baseline: PowerShell startup
scripts (3), World Builder pose accuracy (7), point quality (1), recovery
safety (1), NTFS junction and geometry transport (2), and one experiment-CLI
argument test. The composition differs by one test from the previous lane's
list (it recorded a Document Memory provenance test where this run has
`test_depth_temporal_consistency_requires_video_argument`), which is consistent
with its own note that this suite has load-sensitive members and that the count
moved between two runs of the same tree. **No failure is attributable to the
merge or to this lane**, and no Tower code was changed here.

---

## 4. What did not hold

### 4.1 The UI smoke does not pass three — it passed two and failed one, deterministically

The previous lane recorded "3 passed, 0 failed".

**First, a harness trap worth its own sentence.** Passing
`GLASSES_UITEST_TOWER_AUTHORITY` to `xcodebuild` does **not** reach the test
runner process; it needs the `TEST_RUNNER_` prefix. Without it all three tests
`XCTSkip`, and `xcodebuild` still prints `** TEST SUCCEEDED **` and exits **0**.
A CI job wired the obvious way gets a green light with zero UI tests executed.
That is a "a test that did not run is not evidence" hazard sitting directly in
front of the physical campaign, and it is why this section exists at all.

Run properly, `testASavedWorldsPictureOpensInsideTheApp` **failed**, twice,
including in isolation. It is not flaky.

**It is not a product defect.** The failing line is `reveal(picture)`, *after*
the assertion that the saved world opened had already passed. Instrumenting the
same step showed the Picture control `exists`, `isEnabled` **and** `isHittable`
once the picker's dismissal animation had finished. The Tower was healthy for
that fixture session throughout (`/worlds` reports `state: complete`,
`has_geometry: true`; the geometry manifest and the render page both answer
200).

The defect is in the test helper. `reveal()` swiped the moment its first
hittability check failed. A sheet dismissing over a control reads as
not-hittable for a few hundred milliseconds; one `swipeUp` then scrolls a
control that was **already on screen** up past the top of the scroll view, and
because the helper only ever swipes *up* it can never bring it back — so it
spends its whole ten-second timeout scrolling away from the thing it is looking
for. Fixed by waiting for hittability before touching the scroll view (§5.1e);
the suite is **3 passed, 0 failed** with that change, and the fix was made to
the helper, not to the assertion it was failing.

### 4.2 World Builder is unavailable on a stock Tower — a precondition, not a contradiction

To be fair to the previous lane: it never claimed otherwise. Its setup step
says to start the Tower "with no `TOWER_*` variables set beyond the existing
`.env`", and its `unified_cartridge_smoke.py` configures every root explicitly
so the declaration under test is the full one. The gap is that **there is no
`.env` in the repository** — only `.env.example` — so "the existing `.env`" is
an unstated precondition rather than something a fresh checkout provides.

Starting the Tower with **no `.env` and no `TOWER_*` variables** — which is what
a fresh checkout gives you — produces:

```
[Tower][Config] TOWER_WORLD_ROOT is unset: World Builder is declared but
reported unavailable, and iOS will show it as unsupported
```

`GET /cartridges` then reports `world_builder` `available: false` while the
other four are `available: true`. Document Memory has had a product-managed
absolute default since 2026-09-07 (`DEFAULT_DOCUMENT_ROOT`, `config.py:40`),
and its config comment says the default exists precisely so that "forgetting to
set a path" no longer disables the cartridge. World Builder never got the same
treatment.

This is not a regression and not a merge artifact — but it is a live
cross-cartridge inconsistency of exactly the kind operating rule 6 is about
(product-managed behaviour where environment-variable-driven behaviour used to
be). **Not fixed here**: giving World Builder a default root chooses a new
on-disk location for generated worlds, which is a storage decision for the
Tower lane rather than a Mac validation lane, and `.env.example` does ship
`TOWER_WORLD_ROOT=data/world_builder`. Recorded as §11.1.

**Consequence for the physical test:** step 2 of the existing plan already
expects the world root to print absolute. Confirm a `.env` exists before
starting, or World Builder will read as unsupported on the phone and the whole
World Builder half of the campaign will look broken for a configuration reason.

---

## 5. Changes made

Every change is iOS-side or documentation. **No Tower code was modified.**

### 5.1 Defects fixed

**a. Document Memory could stop a capture it did not start.** *(HIGH — found
independently by three reviewers and confirmed by hand.)*

`ObjectMemoryRecordingCoordinator.cameraClaimChanged` drops ownership whenever
the capture ends, whoever ended it, with a comment naming the exact hazard.
Document Memory is documented as "Modelled on
`ObjectMemoryRecordingCoordinator`" and copied the branch structure of
`send(_:)` without that half: its `captureClaimUpdates` sink only refreshed a
string.

The integration lane's own fix made this worse rather than causing it. Moving
`startedTheCamera` onto a `ProjectManager`-owned `CartridgeCameraClaim` was
right — it stopped a cartridge switch stranding a running capture — but a fact
that outlives the screen also outlives the capture it describes. So: Start in
Document Memory, Stop the camera from Home or the CV Lab, Start it again there,
come back. The claim still said this screen owned the capture. Its panel then
went silent where it should have said "the camera is streaming from another
screen", and its Stop called `stopCameraSession()` on somebody else's session.

Fixed by giving Document Memory the invalidation Object Memory already has.
Two regression tests: one proves a capture ended elsewhere is no longer this
screen's to stop *and* that the panel says so; the other proves this screen's
own Stop still works, which is what stops the fix from being "never stop
anything".

**b. Document Memory's page text could never have reached the app.** *(Wire
correctness.)*

The Tower nests pages inside the document —
`payload["document"] = dict(_summary_view(...), pages=[...])`
(`tower/tower/results/document_memory.py`), and its own wire test asserts
`payload["document"]["pages"][0]["text"]`. `DocumentMemoryDecoder` read
`json["pages"]`, one level up, where no Tower has ever put it. Against a real
Tower `pages` was always empty, and page text is the only thing
`GET /documents/{id}` exists to carry.

It survived because the iOS fixture had been written to match the decoder
rather than the Tower: both were wrong in the same direction, so the test
agreed with the bug. The fixture is now the real wire shape, which turns the
existing `testAPageThatReadNothingIsStillAPage` into a real test, and a new
test fails in both directions — a top-level `pages` must be ignored and a
nested one must be read.

Latent today, because nothing renders pages yet (a library row is a navigation
dead end — §11.3). Fixed anyway: the next lane to build that screen would
otherwise have "verified" it against a fixture the Tower never sends.

**c. Scene Understanding could leave a watcher nothing could retract.**
*(HIGH — privacy-shaped.)*

`workspaceVisibilityChanged(false)` can only unsubscribe an id it holds, and
between a `result_subscribe` and its ack there is no id yet — so leaving in
that window sent nothing. The ack handler then adopted the subscription
unconditionally, opening one for a screen that no longer existed. Nothing could
close it afterwards: the visibility call is guarded on a *change*, so it will
not fire twice, and `subscribeIfPossible` is the only other path.

On this cartridge a watcher is half of the Tower's "somebody streams **and**
somebody watches" gate, so the result is a people detector running for as long
as the socket lives — the precise failure the lane's `workspaceVisibilityChanged`
fix was written to remove, reached through a different door. An ack arriving
after the screen has gone is now unsubscribed rather than adopted. The
regression test drives a real socket against `MockTowerServer`, because the
defect lives in the gap between two wire messages and nothing smaller than the
wire reproduces it.

**d. Scene Understanding kept watching after the phone was pocketed.**
*(HIGH — privacy-shaped.)*

`Info.plist` declares `bluetooth-central`, `bluetooth-peripheral` and
`external-accessory`, so with glasses connected the app is not suspended when
backgrounded: the socket stays up and the camera keeps streaming.
`.onDisappear` does not fire on backgrounding, so the screen stayed "visible"
for the rest of the walk and the Tower kept its detector running, with no
control on screen to stop it.

`scenePhase` now drives visibility. `.background` rather than `.inactive` —
deliberately different from `CVLivePreviewPanel` beside it, and the reasoning
is recorded in the code: that panel is hiding a *picture* from the app-switcher
snapshot, which is taken during `.inactive`, so it must act early; this screen
holds no imagery and what it releases is a Tower-side model, so dropping the
watcher on `.inactive` would unload and reload a detector every time a
notification banner passed over the screen the wearer is still on.

**e. The UI test helper scrolled past its target.** §4.1.

### 5.2 Truthfulness and accuracy

**f. The cartridge drawer told the wearer something false, in the direction its
own comment forbids.** The footer read "Opening one does not start anything on
the Tower by itself". Verified by hand: World Builder POSTs its cartridge
session `start` from `.onAppear` (`WorldBuilderSessionController.workspaceDidAppear`),
and an active session is exactly what lets a builder attach to a capture — this
lane's own smoke run prints `a world-build worker attached once the cartridge
was active`. Scene Understanding subscribes from `.onAppear`, and a
subscription *is* the watcher half of the gate. Both make the Tower begin work
with no control touched.

The comment above that string already recorded a previous correction of the
same sentence in the same direction, and said it is "the one direction this
claim must never be wrong in". The copy now names the two cartridges that reach
the Tower as they open, and keeps the true and useful half — that neither keeps
anything, and that recording is always a deliberate act.

**g. A stale doc comment was corrupting the contract drift gate.**
`CartridgeAvailability.swift` claimed `TowerCapabilities.supported` had "one
element today" and quoted `"world_builder.status/2026-08-25"`. Both halves were
stale — the set holds five, and that identifier was superseded by
`/2026-09-06`. It was not merely wrong: `contract-drift-check.py` decides what
the build implements by sweeping *quoted* identifiers out of Swift source, so a
dead identifier in prose was reported as implemented. The gate's previous
AGREEMENT was therefore reached while listing a superseded contract as
supported — the one thing that check exists to catch. The comment now names
the constant instead of quoting literals, and the gate's output is 13 current
identifiers with no stale entry.

**h. Stale current-state identifiers in four contract documents.**
`docs/contracts/TOWER-UNIFIED-CARTRIDGES.md` still listed
`world_builder.status/2026-08-25` in both its index and its "As of this branch"
table — the file whose own header says it "is the one that was checked against
a running process". `tower/docs/contracts/CARTRIDGE-RESULTS.md` still named
`document_memory.status|library/2026-08-27` in §15's header and the offers
table, while its changelog three sections below documents the 2026-09-07 bump
correctly. Two World Builder cross-references were stale the same way.
Corrected. **No changelog or history entry was rewritten** — those record what
happened at the time and are correct as history.

### 5.3 Build health

**i. The app claimed platforms it cannot run on.** The `Glasses` target had
`SDKROOT = auto` and
`SUPPORTED_PLATFORMS = "iphoneos iphonesimulator macosx xros xrsimulator"`,
with `TARGETED_DEVICE_FAMILY = "1,2,7"`. The Meta DAT xcframeworks ship iOS
slices only, so `xcodebuild -scheme Glasses build` with no `-destination`
resolved `SDKROOT` to the **macOS** SDK and failed with six
`no library for this platform was found` errors naming MWDATCore, MWDATCamera
and MWDATMockDevice — a confusing failure for a real cause.

This directly answers the campaign's "confirm Meta DAT packages are linked only
for supported platforms": they were not restricted. The `GlassesUITests` target
already had the right answer (`iphoneos iphonesimulator`, family `1,2`); it had
simply never been applied to the app or the unit-test target. Now all three
agree, and a destination-less build resolves to the iPhoneOS SDK. Both
configurations still build clean and both test targets still run.

---

## 6. Contract and wire findings

`contract-drift-check.py` reports **AGREEMENT**, and after §5.2g that verdict is
now honest. All 14 current identifiers agree between `tower/` and
`ios/Glasses/`, including the three that moved
(`world_builder.status/2026-09-06`, `document_memory.status|library/2026-09-07`).
Every remaining older-dated string in Swift is either historical prose or a
deliberate negative test — `WorldBuilderIntegrationTests` asserts the old
identifiers are *not* in `TowerCapabilities.supported`. `world_builder.geometry/2026-08-25`
is current, not stale: it is a different contract on a different surface.

**The drift gate is narrower than its name suggests, and this should be said
plainly.** It compares identifier *strings* in one direction — everything the
Tower serves is implemented by the build. It cannot see envelope shape, and it
scrapes Swift source with a regex, which is how §5.2g happened.
`cross-stack-constants-check.py` covers Object Memory only, by construction.
So the payload-level agreement of the other four cartridges rests on
hand-written fixtures, and §5.1b is what that costs: a decoder and its fixture
wrong in the same direction, green for as long as nobody looked at the Tower.

One envelope defect was found and fixed (§5.1b). Others were reported by the
contract reviewer and are **not** fixed here, because none is reachable today
and each needs a decision rather than a patch — §11.4.

---

## 7. Cartridge-by-cartridge

**World Builder.** Contract negotiation, the `latest`-selection fix, the
live/finalizing/history state machine, and the truthfulness of the wording all
hold up. A sweep of every user-facing string found no claim of a mesh, surface,
scan, metric map or room model; the viewer caption is explicit that the output
is sparse structure-from-motion. The saved-worlds picker renders correctly
against the fixture (screenshot captured): world title, session rows, `Complete`
and `No geometry` badges. Two real UX problems are recorded rather than fixed
(§11.2): leaving the screen mid-walk silently ends the session as *interrupted*
and skips the final solve, and the geometry surface has essentially no
accessibility representation.

**Object Memory.** Opening the screen cannot allocate a Tower worker — traced
end to end: `onAppear` starts only a `GET` poll, and spawning is reachable only
from `POST`. Start/Stop/Pause semantics, liveness read from `following` rather
than intent, and the image-degradation states are all sound, and no string
implies instance identity or re-identification. The known `since` gap is
unchanged and remains the sharpest product issue in the cartridge (§11.5).

**Document Memory.** Two defects fixed (§5.1a, §5.1b). Confirmed: no remaining
dependency on `TOWER_DOCUMENT_ROOT` for normal behaviour — the managed default
is absolute and the Tower boots to
`document root .../tower/data/document_memory` with no environment set. The
variable name does survive as a *protocol token*: iOS classifies a 404 by
string-matching `TOWER_DOCUMENT_ROOT` in the Tower's prose, and the Tower keeps
the name deliberately for that reason. Correct today, brittle, and recorded.

**Scene Understanding.** Two lifecycle defects fixed (§5.1c, §5.1d). **The
privacy invariants hold.** Verified independently: the wire carries a count and
four aggregate buckets and refuses `track_id`, `box`, `facing`, `visible_eyes`
and `confidence` as explicit values rather than silences; association is IoU
only, with the refusal of appearance matching written into the method; the
YuNet face *detector* used for orientation indexes past its landmarks and
yields a single boolean, computing no embedding and matching nothing across
frames or sessions; and the app has no persistence layer at all — a sweep for
`UserDefaults`, `FileManager`, `write(to:`, CoreData, SwiftData and Keychain
across `ios/` returns zero hits. Orientation is labelled Experimental from the
wire rather than hard-coded, and nothing implies gaze, attention, intent or
emotion. The single-person disclosure is decoded, gated at a count of one, and
rendered verbatim — but it has **no test**, and it is nested under an unrelated
flag (§11.6).

**CV Lab.** **INTACT.** Its own source is byte-identical to the pre-integration
base, and the only files in its dependency closure that the integration touched
changed comment text or added Document Memory properties. The screen opens, the
controls stay enabled, experiment selection still reaches the wire, and it
holds no `CartridgeSession`, so the new "stop sessions when the last client
goes" teardown does not touch it. The integration in fact improved it: the
World Builder gate stopped an ungated builder attaching to CV Lab captures.

---

## 8. Cross-cartridge coherence

Reviewed as one product rather than five lanes. The findings are real but are
overwhelmingly *consistency* rather than *correctness*, and fixing them well is
a product pass, not a validation lane. Recorded in §11.7 with the evidence so
the next lane does not have to rediscover them. The headline items: five
different names for the one control that starts the glasses camera; the Tower's
connection state rendered by three different word-tables, two of which are on
the CV Lab screen simultaneously saying "Offline" and "Disconnected"; Object
Memory rendering every error as unstyled grey prose indistinguishable from its
empty state; and — the one with a privacy edge — **nothing outside a cartridge
says that cartridge is still recording**, with the camera pill compiled out
entirely in Release.

---

## 9. External research

None was needed. Every question this lane had to settle was answerable from
primary sources already in the repository or from the toolchain itself: the
Tower's own contract documents and wire tests for §5.1b and §6, the project's
existing `CVLivePreviewPanel` for the `scenePhase` pattern in §5.1d, the
existing `GlassesUITests` target for the correct platform settings in §5.3, and
`xcodebuild -showBuildSettings` plus a deliberately destination-less build for
the diagnosis in §5.3. Where a UIKit/SwiftUI behaviour was load-bearing —
`.onDisappear` not firing on backgrounding, `scenePhase` transitions, and
`TEST_RUNNER_`-prefixed environment variables reaching an XCUITest runner —
each was confirmed by observation on this machine rather than from memory: the
`TEST_RUNNER_` behaviour by running the suite both ways (§4.1), and the
hittability behaviour by instrumenting the app under test.

---

## 10. Reviewer findings and disposition

Eight independent read-only reviewers ran against `9e939a3` (build/contracts,
the five cartridges, cross-cartridge coherence, lifecycle), and one independent
adversarial reviewer against this lane's diff.

**Acted on:** the Document Memory camera claim (raised by three reviewers
independently, and confirmed by hand before any change); the Document Memory
`pages` nesting; the Scene subscribe/leave race; Scene backgrounding; the
drawer's false claim; the drift-gate doc comment; the stale contract documents.

**Recorded, not fixed** — every item in §11. The rule applied was that a change
earns its place in a validation lane only if it is bounded, testable, and
reduces risk for the physical campaign. Rewording five cartridges' Start
controls does not meet that bar three days before a device run; a detector that
never stops does.

**Reviewer disposition on this lane's own diff.** One adversarial reviewer,
instructed to assume this lane was overconfident, went through all seven changes
against the code and both sides of the wire. Six came back SOUND or SOUND WITH
NITS. **One came back WRONG, and it was right to.**

**The defect this lane introduced, and fixed.** The corrected drawer copy in
§5.2f originally ended "Neither keeps anything." That is false. World Builder's
cartridge session is the gate on the world-build worker, and that worker is
launched with `--root <world_root>`, opens a `WorldStore` on it, and writes
sources, placements and a solve log — the very worlds "Saved worlds" later
lists. Verified by hand before changing anything back
(`tower/tower/main.py`, `tower/scripts/world_build_session.py`).

The error is worth naming precisely because it is the same class this lane was
fixing: generalising from *"a `CartridgeSession` is not persisted"* — true — to
*"nothing is written down"* — false. It replaced a stale-but-conservative
sentence with an active reassurance that was wrong in the one direction the
surrounding comment says this claim must never be wrong in. The copy now
separates the two cartridges: Scene Understanding keeps nothing, and World
Builder's session is what lets the Tower build and save a world from a capture
already running.

**Three comment overclaims, also fixed.** The `reveal()` doc said the settle
wait "costs nothing" when the element is genuinely below the fold — it costs a
flat two seconds there, and raises the helper's worst case from `timeout` to
`timeout + settle`. The Scene ack comment said "nothing could ever close it" —
in fact the next appear-and-leave cycle would have. And the rewritten
`supported` bullet called all five identifiers subscription contracts, when
`document_memory.library` is HTTP — which contradicted the comment immediately
beside it. In a codebase that treats prose as load-bearing these are defects,
not tidying.

**One finding recorded rather than fixed** — §11.13.

**Accepted as correct by the reviewer, with evidence:** the Document Memory
camera-claim fix including its Stop ordering (twice over — the branch is
synchronous, and `captureClaimUpdates` hops to the next main-queue turn); the
`pages` nesting fix and that the fixture was moved to the real wire shape rather
than bent to pass; the Scene ack guard, including that the resulting
`result_unsubscribed` is correctly ignored and no double-unsubscribe occurs;
`.background` rather than `.inactive`, and that `onAppear`/`onChange` cannot
reach a wrong state because `ContentView` destroys the workspace on a cartridge
switch; that the `reveal()` change weakens no assertion and therefore cannot
mask a product defect; and that the pbxproj edit lints clean, matches the
UITests target, and breaks neither device nor archive builds. The reviewer also
independently re-ran the drift check's own regex over the Swift tree and
confirmed it now yields 13 identifiers, all from `static let` declarations, with
no comment pollution left.

---

## 11. Known limitations carried forward

Nothing here was weakened by this lane. Items 1–7 are new to this document;
the rest are carried from §18 of the previous handoff and remain accurate.

1. **World Builder is unavailable on a stock Tower** (§4.2). Configuration, not
   a defect, but it will read as a broken cartridge on the phone.
2. **World Builder: leaving the screen mid-walk ends the session as
   `interrupted` and skips the final solve.** The session is bound to SwiftUI
   view lifetime, `.onDisappear` sends `stop` unconditionally, and on the Tower
   that is a soft stop that closes the session interrupted. The camera is not
   stopped, so the glasses look like they are still mapping, and returning
   attaches a *second* builder to the same capture. The skipped final solve is
   what produces the registered cluster, so the result silently degrades from
   one connected world to loose fragments — and it is indistinguishable from a
   genuine builder crash in both the picker badge and the canvas headline. The
   existing physical plan does not enter this window (step 40 leaves during
   *finalization*, which is safe by design); §13 adds a step that does.
   **The highest-value product fix in this document.**
3. **Document Memory library rows are a navigation dead end.** `DocumentPage`,
   `pageIndex`, `DocumentCoverage`, `wordCount` and `summary` are all decoded
   and unreachable, and `document(id:)` — "the only route that carries text" —
   has no caller. Also: no in-app deletion, and the screen does not say so, on
   a cartridge that persists the text of pages held in front of a person's
   face. `retentionDays` and `documentsPruned` are already decoded and unused.
4. **Envelope-level gaps the drift gate cannot see.** Reported with evidence
   and left for a decision: Document Memory renders "Seen 5 times · 12.5 s"
   where the duration is the *first* sighting only (`total_observed_seconds` is
   not modelled); `match_tolerance` and per-match `fuzzy` are not decoded, so a
   one-edit fuzzy match renders identically to an exact one under copy that
   says "Matched word for word"; per-document OCR `confidence` is decoded and
   never rendered; `stop_policy` is never read, so a client cannot tell "the
   process is gone" from "the process was asked and is finishing"; World
   Builder's `artifacts` block and `retains_raw_imagery` are unread. Several
   latent decode-refusal paths were also identified.
5. **Object Memory's `since` is still server-side only.** The phone fetches the
   whole store, and Stop auto-refreshes into that unscoped list — so the
   canonical first run ends by rendering 30 days of history as the visible
   consequence of stopping a two-minute walk. The labelling names a *window*
   and never says the list spans previous sessions. Everything needed to fix it
   (`started_at`) is already decoded on the phone. §13 still answers "did this
   recording capture anything" with `curl`.
6. **Scene Understanding's single-person disclosure has no test**, and renders
   only inside the `countIsLowerBound` branch — a privacy note gated on an
   unrelated truthfulness flag that the Tower currently hard-codes `true`. Also
   unmodelled: the Tower's own `orientation_validation` text, which is more
   candid than the app's hand-written substitute, and `where.person`, which the
   Tower has emitted since 2026-09-07 while three iOS comments and a test still
   say it cannot.
7. **Cross-cartridge inconsistency** (§8), and no global indication that a
   cartridge is still recording after you leave it.
13. **A residual double-subscribe race in Scene Understanding**, narrowed but
   not closed by §5.1c, and given a new trigger by §5.1d. Appear, leave, and
   return *before* the first ack, and two subscribes are outstanding: both acks
   now arrive while visible, both are adopted, and the second overwrites
   `subscriptionID`, orphaning the first. `isSubscribing` is a boolean, so it
   cannot tell one outstanding subscribe from two; the durable answer is a
   generation token, which is a wider lifecycle change than this lane should
   make days before a device run. The window is a socket round trip plus the
   Tower's snapshot build — tens of milliseconds on a LAN. Recorded in the code
   at the guard, and §13 adds the Tower-side check that would reveal it.
8. **World Builder** produces sparse structure-from-motion, not a dense or
   metric room reconstruction. Re-verified: every occurrence of those words in
   code, contracts and Swift is a negation.
9. **Object Memory** ships no user-taught instance identity and no
   cross-session re-identification.
10. **Document Memory** has never read a physical page through the glasses.
11. **Scene Understanding**: counting and object perception LIMITED,
    orientation EXPERIMENTAL, nothing about people ever checked on this camera.
12. Two lifecycle vocabularies (`CartridgeSession` / `LiveSession`) still
    coexist.

---

## 12. What macOS could not establish

Unchanged from the previous lane, and restated because a test that did not run
is not evidence.

1. **The Windows resilient listener fix.** `test_serve_loop.py` skips on
   `sys.platform != "win32"`. CPython gh-93821 is the reason the Object Memory
   lane exists on that side and this host cannot retest it.
2. **Job Object process ownership.** Windows-only. What macOS did exercise is
   the cross-platform half: one owned process per worker, the stop request, the
   bounded reap, and no child outliving the Tower.
3. **Every CUDA path.** This Mac ran CPU EasyOCR and SSDLite. **The detector
   Scene Understanding ships on was never loaded here**, so §5.1c and §5.1d —
   both of which are about releasing that detector — are verified as *client
   behaviour on the wire*, not as VRAM actually coming back.
4. **The PowerShell startup scripts.**
5. **The real capture corpus.**
6. **Anything involving a phone, glasses, a printed page, a person, or a real
   network.** No device was attached. In particular the four iOS fixes in §5.1
   have compiled and passed unit tests but have never run on a device.

---

## 13. Physical test recommendations

Run the existing 45-step campaign in
`ALL-CARTRIDGES-MAC-INTEGRATION-VALIDATION.md` §23 unchanged. **Build and
install the phone from this branch**, not an older build. Add and change the
following.

**Before step 1 — new.** Confirm `tower/.env` exists and sets
`TOWER_WORLD_ROOT`. On a stock checkout there is no `.env` and World Builder
declares itself unavailable (§4.2). Then confirm at step 3 that all four
cartridges report `available: true`, rather than assuming it.

**Step 37 is now the most important step in the campaign, and it has a second
half.** After leaving the Scene screen and confirming `GET /scene` reads
`stopped` with `demand.watchers: 0`, repeat it by **backgrounding the app**
(swipe to the home screen, do not leave the screen first) while a camera is
streaming. `GET /scene` must reach `stopped` and `nvidia-smi` must show the
model released. That is §5.1d, it has never run on a device, and before it the
detector ran for the whole walk with the phone in a pocket.

**New, after step 37.** Open Scene Understanding and leave it again
*immediately*, within a second, several times. Each time, `GET /scene` must end
at `watchers: 0`. This is §5.1c — the ack arriving after the screen has gone —
and a slow or throttled phone is exactly where the window is wide.

**New, replacing part of step 30.** In Document Memory: Start, then go to the
CV Lab and press **Stop camera**, then **Start camera** there, then return to
Document Memory. **PASS:** the panel says the camera is streaming from another
screen, and Document Memory's Stop does **not** kill the CV Lab's capture.
Before §5.1a it claimed the capture as its own and stopped it.

**New, after step 14.** Start a World Builder capture, walk 30 seconds, then
**leave the World Builder screen while still walking** — do not press Stop.
**Expected today:** the session ends as `interrupted`, the final solve is
skipped, and the camera keeps streaming. Record what actually happens; this is
§11.2 and it decides whether that becomes a blocker or a documented V1
limitation.

**New — a caution, not a step.** Backgrounding the app now drops the Scene
Understanding subscription (§5.1d). On return the counts start from zero and
the payload reports loading while the Tower reloads its detector. That is the
fix working, not a defect. A tester who pockets the phone mid-walk and comes
back to an empty scene must not file it as a regression.

**New — watch the Tower, not only the phone, during the Scene steps.** Follow
`watcher_joined` / `watcher_left` on the Tower across steps 31–37. §11.13 is a
residual race that the Simulator cannot reveal and that only shows up as a
watcher count that does not return to zero. If `demand.watchers` ever settles
above zero with the Scene screen closed, that is §11.13 and it should be
reported rather than worked around.

**Step 20 stands as written** — `since` is still unimplemented on the phone, so
"did this recording capture anything" must be answered with `curl`.

**A caution for whoever reads the results.** Any fixture world built before
2026-09-06 reads as `idle` to this Tower, because `ready` is derived from a
`session_stopped` event plus a complete finalization. That is correct
behaviour, not a defect.

---

## 14. Final state

**Branch:** `integration/all-cartridges-v1`.
**Starting HEAD:** `9e939a3`. **Final HEAD:** `4e152c3` plus this file.
**Repository cleanliness:** clean apart from this document at the moment it
was written, and clean after it. Nine commits, each one change with its
reasoning in the message: four defect fixes, one test-harness fix, one
build-settings fix, one user-facing copy correction, one documentation pass,
and this file. No Tower code was
modified -- the only Tower-side edit is two stale identifiers in a contract
document.

No model weights, dataset, replay corpus, cache, virtual environment,
machine-local log, DerivedData, `.env` or user data is committed. No branch was
created, deleted, renamed or force-pushed; no history was rewritten; nothing
was pushed. All four source lane tips remain where they were.

---

## 15. Verdict

# READY FOR PHYSICAL VALIDATION

**With §11 and §12 read as part of the verdict rather than as footnotes to it.**

**What this lane establishes.** Every gate the previous lane claimed was re-run
from scratch rather than confirmed, and the substantive ones reproduced exactly:
both builds clean with no new warnings, the Tower suite at the same totals with
zero new failures against its baseline, 71 of 71 unified smoke checks including
the gate assertions in both directions, and contract agreement across all five
cartridges. Two claims did not reproduce and both are documented with evidence
rather than smoothed over — one a deterministic UI test failure that turned out
to be a test-harness defect and not a product one, the other a stock-Tower
configuration gap that would have read as a broken cartridge on the phone.

Four defects were found and fixed that no compiler and no existing test would
have caught, and two of them are the same class the integration lane fixed —
a subscription that could not be retracted, and a camera claim that outlived
the capture it described. Two of the four are privacy-shaped: before this lane,
pocketing the phone on the Scene screen left a people detector running for the
rest of the walk.

**Why READY rather than validated.** Nothing here touched a phone, glasses, a
printed page, a person, or a real network. The four iOS fixes have compiled and
passed unit tests and have never executed on a device. The Windows listener
fix, Job Object ownership and every CUDA path remain structurally unavailable
on this host — including, pointedly, the VRAM release that §5.1c and §5.1d
exist to cause.

**Why READY rather than NOT READY.** No known defect is being carried into the
physical test that would make its results misleading. The three carried
limitations that could confuse a reader — World Builder's stock-Tower
unavailability, the interrupted-on-leave behaviour, and Object Memory's
unscoped list — are each named, each explained, and each given an explicit step
or caution in §13 so that a configuration problem is not mistaken for a
regression.

**The verdict would be NOT READY** if any of these were true, and none is: a
builder attaching to a CV Lab capture; the app failing to build; a contract
disagreement; a cartridge that cannot be left; a new Tower failure introduced
here; or a screen that leaves its detector running — which was true when this
lane started, twice over, and is not now.

---

## 16. Temporary resources (filesystem policy rule 9)

- **No worktree was created.** Work ran in the canonical checkout, on
  instruction — §0.
- Reused, not created: `Glasses-scratch/all-cartridges-venv-ml` (the previous
  lane's venv) and `Glasses-scratch/ac-tmp/uifixture-world` (its corrected UI
  fixture, read-only).
- **Nothing was written into the canonical checkout.** The Tower was started
  on product defaults, which resolve the document root to
  `tower/data/document_memory` — but that directory is created lazily on the
  first recording and no document session was ever started, so `tower/data`
  does not exist. Checked, rather than assumed.
- Session scratchpad under `/private/tmp/claude-501/.../scratchpad` — build and
  test logs, DerivedData, an extracted screenshot, and backups of the files
  edited by script. OS temp.
- One Tower on port 8000 ran for the length of the lane and was **stopped at
  the end**; no worker, child process or lock outlived it. Nothing was written
  outside `Projects/` and the OS temp directory.
