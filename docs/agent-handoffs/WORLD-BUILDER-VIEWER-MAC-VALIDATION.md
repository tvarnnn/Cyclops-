# Mac validation: the saved-world viewer after the iOS review fixes

Branch `world-builder/reconstruction-final-review1` (worktree
`Glasses-worktrees/wb-final-recon-r1` on Windows). This supersedes the scratch
checklist `Glasses-scratch/wb-final-recon/review-ios/MAC-VALIDATION.md`, which
was written against 97e75eb before the review's fixes; that file is kept as it
was. The review itself is `Glasses-scratch/wb-final-recon/review-ios/REVIEW.md`.

Run these steps in order. Stop at the first failure and record it. **Nothing
Swift on this branch has been compiled.** It was written and read on Windows.
Each "Record" line is a number or a verdict that goes into the handoff.

Conventions:
- Repository on the Mac: `~/Projects/Glasses`.
- Worktree: `~/Projects/Glasses-worktrees/wb-final-recon`.
- Scratch: `~/Projects/Glasses-scratch/wb-final-recon-mac/`.

Do not work in the canonical checkout.

## What changed since the first checklist (the review finding each check covers)

| Finding | Change | Where it is checked |
|---|---|---|
| M1 | Tower serves the sparse page for `view=diagnostics`; the phone does not follow a diagnostics target; Details text depends on the rung | §3, §5.9 |
| M2 | A refreshed page that fails to draw (watchdog, `didFail`, kill budget) puts the previous page back and that revision is never retried | §2, §5.5 |
| M3 | Only a better rung, or a same-rung build the Tower says is finished (`live: false`), swaps by itself; a live same-rung rebuild shows *"A newer reconstruction is ready. Show it"* | §2, §5.4 |
| M4 | Only `{"detail":"Not Found"}` ends following; other 404s are retried | §2, §3, §5.10 |
| M5 | WebContent kills count within a 60 s window | §2, §5.6 |
| m1 | Every revision is `<session>/<rung…>` | §3 |
| m2, m9 | The page's stamped revision is recorded, plus the polled one acted on: one extra download per rebuild, not per poll; the Tower logs the disagreement at ERROR | §2, §3 |
| m3, m4 | Tests bounded, followers awaited, cancellation test added | §2 |
| m5 | UI smoke waits for the rung caption | §4 |
| m6 | CSP `<meta http-equiv>` in the sparse and dense pages (surface already had it) | §3 |
| m7 | `WORLD-BUILDER-IOS.md` §10 describes the viewer | read it |
| m8 | §4a payload carries `live`; the phone polls every 10 s while live and backs off to 120 s otherwise (never stops) | §3, §5.4, §5.7 |

## 0. Check out the exact commit

```sh
cd ~/Projects/Glasses && git fetch --all
git worktree add ~/Projects/Glasses-worktrees/wb-final-recon world-builder/reconstruction-final-review1
cd ~/Projects/Glasses-worktrees/wb-final-recon && git rev-parse HEAD      # the commit named in the handoff
python3 -c "import pathlib,sys; bad=[p for p in pathlib.Path('.').rglob('*') if p.is_file() and p.suffix in {'.html','.swift','.py','.md'} and b'\x00' in p.read_bytes()]; print(bad); sys.exit(bool(bad))"
```

Record: the HEAD hash, and that the NUL-byte scan printed `[]`.

## 1. Build

```sh
cd ios
xcodebuild -version
xcodebuild -project Glasses.xcodeproj -scheme Glasses \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -derivedDataPath ~/Projects/Glasses-scratch/wb-final-recon-mac/dd \
  build-for-testing 2>&1 | tee ~/Projects/Glasses-scratch/wb-final-recon-mac/build.log
grep -E "error:|warning:" ~/Projects/Glasses-scratch/wb-final-recon-mac/build.log | sort | uniq -c
```

Record `BUILD SUCCEEDED` or the errors, and the warning count against the
baseline of **8**. Any warning in `WorldRenderViewer.swift`,
`WorldBuilderIntegrationTests.swift` or `TowerSmokeUITests.swift` is new and is a
finding.

Most likely spots if the build fails, most likely first:
- `WorldRenderViewerModel.nextPollInterval`: `min(max(current * 2, base), ceiling)` on `Duration` (`Duration * Int`, `Comparable`).
- `WorldRenderViewerModel.fallback`, an optional labelled tuple `(html: String, revision: String?)?` stored on a `@MainActor` class.
- `followRevisions()`: `client.revision(for:)` now returns `WorldRenderRevision?` (a `nonisolated` `Sendable` struct) across the call from the main actor.
- `WorldRenderWebView.Coordinator`: `nonisolated static let terminationWindow` / `nonisolated static func recentTerminations` on a main-actor class.
- Tests: the nested `@MainActor private final class Flag` inside `WorldRenderRevisionTests`, and `Task { @MainActor in await follow.value; ended.value = true }`.
- Tests: `XCTAssertEqual(next(.seconds(10), live: true), .seconds(10), …)` — implicit-member `Duration` in the second argument.

## 2. Unit tests (no Tower)

```sh
xcodebuild -project Glasses.xcodeproj -scheme Glasses \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -derivedDataPath ~/Projects/Glasses-scratch/wb-final-recon-mac/dd \
  -test-timeouts-enabled YES -maximum-test-execution-time-allowance 60 \
  -only-testing:GlassesTests/WorldRenderRepresentationTests \
  -only-testing:GlassesTests/WorldRenderRevisionTests \
  -only-testing:GlassesTests/WorldRenderViewerTests \
  -resultBundlePath ~/Projects/Glasses-scratch/wb-final-recon-mac/unit-render.xcresult \
  test-without-building
```

Expect:
- 8 tests in `WorldRenderRepresentationTests` (unchanged);
- **19** in `WorldRenderRevisionTests`: 7 pure (address, page stamp, body decode, upgrade order, finished-build swap, poll interval, termination window) and 12 that drive the follower (better rung swaps; same rung offered then swapped on `showNewerPicture`; unchanged revision fetches no page; a revision the page does not carry costs one fetch; failed fetch keeps the page; kill-budget failure reverts; watchdog failure reverts; a first page that cannot draw still fails; unmatched-route 404 ends following after one request; contract-worded 404 is retried; diagnostics target not followed; cancellation ends the loop);
- `WorldRenderViewerTests` unchanged, all green.

The follower tests no longer hang on a regression: every wait is bounded at
2–3 s and fails an assertion instead.

Flakiness (review m4) — run the pair **5 times**:

```sh
for i in 1 2 3 4 5; do xcodebuild … -only-testing:GlassesTests/WorldRenderRevisionTests -only-testing:GlassesTests/WorldRenderViewerTests test-without-building 2>&1 | grep -E "Executed|failed"; done
```

The timing-sensitive ones, if any flake: `testARefreshThatNeverFinishesDrawingPutsTheOldPictureBack`
(150 ms render bound against 10 ms polling) and
`testARevisionThePageDoesNotCarryCostsOneFetchNotALoop` (6 polls at 20 ms).

Then the whole unit suite once:

```sh
xcodebuild … -only-testing:GlassesTests test-without-building 2>&1 | tee ~/Projects/Glasses-scratch/wb-final-recon-mac/unit-all.log | grep -E "Executed|error:|failed"
```

Record pass/fail per class, the time `WorldRenderRevisionTests` took (expect a
few seconds), any failure in the 5 repeats (name the test), and the full-suite
totals against base 869d715.

## 3. Tower on the Mac, with a world that has all three rungs

A Mac cannot build surfaces. Copy world `b2a75ab40d2d415d8d6ef5e4d5f0fb3d`
(session `a8c6817e14a74e3c977fccfcdacad595`, surface L2 ≈3.7 MB page) as whole
directories from the Windows canonical checkout's
`tower\data\world_builder\worlds\` to
`~/Projects/Glasses-scratch/wb-final-recon-mac/worlds/`, plus the synthetic
sparse-only fixture from `Glasses-scratch/wb-cv-sim/`.

```sh
cd ~/Projects/Glasses-worktrees/wb-final-recon/tower
python3 -m venv ~/Projects/Glasses-scratch/wb-final-recon-mac/venv && source ~/Projects/Glasses-scratch/wb-final-recon-mac/venv/bin/activate
pip install -e .
TOWER_WORLD_ROOT=~/Projects/Glasses-scratch/wb-final-recon-mac/worlds \
TOWER_CAPTURE_ROOT=~/Projects/Glasses-scratch/wb-final-recon-mac/captures \
TOWER_WORLD_AUTOBUILD=false \
  python -m uvicorn tower.main:app --host 127.0.0.1 --port 8010
```

```sh
W=b2a75ab40d2d415d8d6ef5e4d5f0fb3d; S=a8c6817e14a74e3c977fccfcdacad595; T=http://127.0.0.1:8010
curl -s -D - "$T/worlds/$W/render/revision?session_id=$S"
#   200, Cache-Control: no-store, {"session_id": S, "representation": "surface", "revision": "S/surface:…", "live": false}
curl -s "$T/worlds/$W/render/revision?session_id=$S&view=diagnostics"     # representation "sparse", revision "S/sparse"
for q in "" "&representation=dense" "&representation=sparse"; do
  curl -s "$T/worlds/$W/render?session_id=$S$q" -o /tmp/p.html; wc -c < /tmp/p.html
  head -c 4096 /tmp/p.html | grep -c 'http-equiv="Content-Security-Policy"'          # 1 for EVERY rung (m6)
  head -c 4096 /tmp/p.html | grep -o '<meta name="wb-[a-z]*" content="[^"]*">'        # both tags inside 4096
done
curl -s "$T/worlds/$W/render?session_id=$S&view=diagnostics" | head -c 800 | grep -o 'wb-representation" content="[a-z]*"'   # sparse (M1)
curl -s "$T/worlds/$W/render/nope"; echo                                   # {"detail":"Not Found"}  -> the phone stops
curl -s "$T/worlds/nope/render/revision"; echo                             # {"detail":"no world 'nope'"} -> the phone retries
```

Record: page size per rung; the two meta values; that the revision JSON's
`revision` equals the surface page's `wb-revision` (**must be equal**); that the
CSP meta count is 1 on all three rungs; the rung for `view=diagnostics` (must be
`sparse`); `live` (must be `false` with autobuild off).

## 4. UI smoke in the Simulator against that Tower

```sh
cd ~/Projects/Glasses-worktrees/wb-final-recon/ios
TEST_RUNNER_GLASSES_UITEST_TOWER_AUTHORITY=127.0.0.1:8010 xcodebuild \
  -project Glasses.xcodeproj -scheme Glasses \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -derivedDataPath ~/Projects/Glasses-scratch/wb-final-recon-mac/dd \
  -only-testing:GlassesUITests/TowerSmokeUITests \
  -resultBundlePath ~/Projects/Glasses-scratch/wb-final-recon-mac/ui.xcresult test
```

Record each test's pass/skip/fail. **Skipped is not passed.** The picture test
now fails with *"the caption read the page's rung"* if the app never read
`wb-representation` (review m5). In the `3d-world` screenshot:

- [ ] a shaded surface for the b2a75ab4 session, captioned *"Surfaces the Tower reconstructed from the walk…"*;
- [ ] for the sparse fixture, *"Points the Tower measured from the walk, with the camera path through them…"*.

## 5. Manual checks in the Simulator, then on a physical iPhone

Debug build with `GLASSES_TOWER_AUTHORITY=127.0.0.1:8010` for the Simulator; the
Tower's Tailscale address on the phone (a Windows Tower on port 8000 for §5.4).
Console.app streaming the device, filtered for `WebContent`, `jetsam`, `Glasses`.

1. **Caption per rung.** Open each world. Record each caption's first four words and the page's own caption; they must not contradict. Open Details: on the surface world it must read *"The solver's own view of this session is under Diagnostics on the world screen."*; on the sparse fixture it must mention *"The page's own Diagnostics button, above"*.
2. **Gestures.** Orbit, pinch, pan, Walk/Orbit, Prev/Next. The sheet never scrolls or bounces; back swipe works from the left edge.
3. **Time and memory to first draw** (physical phone, b2a75ab4). Tap → overlay gone, three times; `window.__wbSurface.timing` (`b64Ms`, `decodeMs`, `uploadMs`, `bootMs`); app MB; WebContent MB.
4. **Following during a live walk** (phone + glasses + Windows Tower, `TOWER_WORLD_AUTOBUILD=true`, surface on). Open the Picture early in the walk, keep it open ≥5 min, orbit into a corner and keep looking. Record:
   - Tower log times of each `GET …/render/revision`: about one every **10 s** while walking (`live: true`);
   - when the rung first **improves** (sparse → surface), the picture swaps by itself, once;
   - after that, each surface rebuild shows the button *"A newer reconstruction is ready. Show it"* and does **not** move the camera (review M3). Count the rebuilds offered, and tap the button at least twice: each tap swaps the page and resets the camera, and the button disappears;
   - count of full `GET …/render`: must equal 1 + rung improvements + taps (+ at most one per rebuild if the Tower logged the m9 ERROR);
   - after Stop: polls continue through finalization, the final surface (built after Stop, `live: false`) is **swapped in by itself** without a tap, once, and once finalization is done the poll gaps grow 10 → 20 → 40 → 80 → 120 s and stay at 120 s. Record the gaps.
5. **A swap that cannot be drawn** (review M2). During §5.4, at each swap or tap, watch Console for WebContent termination. If a swap is killed past its budget or takes over 20 s, the **previous picture must come back** (a brief "Drawing the world…" then the old picture) and that revision must not be offered again. Record the WebContent footprint before and after each swap, and any revert. If none happens naturally, optionally serve L1 pages from the Tower (larger) to provoke one, and record the result.
6. **Backgrounding** (physical phone, surface world open, not walking). Lock 2 min, unlock; repeat 4 times, **at least 60 s apart**. After each unlock record: picture back without reload / context-restored message then picture / full reload / *"…ran out of memory N times…"*. The failure message must not appear from kills spread over minutes (review M5). If it does, record the times.
7. **Retry path.** Tower stopped, open a world: failure with Try again. Start the Tower, tap Try again; the picture draws. Leave it 3 min. In the Tower log, revision polls come from **one** loop, with gaps 10, 20, 40, 80, 120 s (a finished world, `live: false`).
8. **Dismissal.** Open a world, wait 15 s, close. No `/render/revision` request after the close.
9. **Solver view** (review M1). World screen → Diagnostics → "Open the solver's 3D view" on b2a75ab4: segment-coloured sparse points with the page's Diagnostics toggle, **not** the surface. Leave it open 60 s: **no** `/render/revision` requests in the Tower log for it.
10. **Older Tower.** Point the app at a Tower on 869d715 and open a saved world: caption *"What the Tower reconstructed from the walk. Not to scale."*, and exactly **one** `/render/revision` request (404 `Not Found`) in its log, then none.

## 6. Clean up and record resources

Stop uvicorn. `git worktree list` shows only what you added. Leave
`~/Projects/Glasses-scratch/wb-final-recon-mac/` in place and list it in the
handoff; do not delete it without explicit approval.

## What to paste into the handoff

| Item | Value |
|---|---|
| HEAD validated | |
| Xcode / Simulator / device iOS | |
| Build result, warnings (new vs 8) | |
| `WorldRenderRepresentationTests` 8 / `WorldRenderRevisionTests` 18 / `WorldRenderViewerTests` | |
| 5× repeat flakes | |
| Full `GlassesTests` totals vs base | |
| revision JSON == page `wb-revision`; `live` value | |
| CSP meta count per rung (sparse/dense/surface) | |
| `view=diagnostics` rung served | |
| TowerSmokeUITests per test (rung caption assertion) | |
| Details text per rung | |
| Tap-to-drawn ×3, `__wbSurface.timing`, app MB, WebContent MB | |
| Live walk: poll gaps while live, auto swaps, offers, taps, camera moved without a tap (must be 0), full fetches | |
| Post-finalization poll gaps | |
| Reverts after a failed swap | |
| Backgrounding ×4 outcomes | |
| Retry: single loop, back-off gaps | |
| Dismiss stops polling | |
| Solver view: sparse, no polling | |
| Old Tower: one 404 then silence | |
