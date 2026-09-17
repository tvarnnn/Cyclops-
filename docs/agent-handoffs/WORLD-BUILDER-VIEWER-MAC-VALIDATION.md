# Mac validation: the saved-world viewer after the iOS review fixes

## Fix-it addendum (2026-09-17): the appearance rung and the `glasses-world:` transport

Branch `world-builder/reconstruction-fixit-v1`. **Run this addendum first, on the
fix-it SHA named in the fix-it handoff; the checklist below it still applies
unchanged except where this addendum says otherwise.** Nothing Swift in this
change has been compiled; the page and the Tower routes were verified on Windows
in headless Chrome (SwiftShader), which says nothing about WebKit, Metal ASTC or
phone frame times.

What changed, and what each check proves:

| Change | Files | Check |
|---|---|---|
| New top rung `appearance`: a ~70 KB page shell that fetches the manifest, the proxy and ASTC bundles, blends the redacted keyframes on the proxy (k = 4 ULR, on-device source depth, alpha 0 = no weight, unshaded), and follows new appearance builds in place | `tower/tower/world_builder/appearance_render.py`, `appearance_viewer.html`, `tower/tower/results/world_builder_render.py`, `tower/tower/routes/geometry.py` | A3, A5, A6 |
| The page is served from `glasses-world://tower/worlds/<w>/render` by a `WKURLSchemeHandler` that proxies only this world's appearance and revision routes | `ios/Glasses/Workspaces/WorldBuilder/WorldAssetTransport.swift` (new; the app target uses a synchronized group, so no project edit) | A1, A2, A4 |
| Non-persistent `WKWebsiteDataStore`; navigation policy allows exactly the page URL; `WorldRenderClient` moved off `URLSession.shared` to an ephemeral session with no URL cache | `WorldRenderViewer.swift` | A1, A2, A7 |
| `WorldRenderRepresentation.appearance` (rank above surface) and its caption; `WorldRenderRevision.appearance` decoded but never acted on | `WorldRenderViewer.swift` | A2, A6 |

### A1. Build

As §1 below, then:

```sh
grep -E "WorldAssetTransport.swift|WorldRenderViewer.swift" ~/Projects/Glasses-scratch/wb-final-recon-mac/build.log | grep -E "error:|warning:"
```

Record every error and warning in either file. Most likely spots, most likely first:

- `WorldAssetSchemeHandler.webView(_:start:)`: the `Task { [weak self] in … }` captures `urlSchemeTask` (`any WKURLSchemeTask`, not `Sendable`) and `asset` in a main-actor closure; `ObjectIdentifier(urlSchemeTask as AnyObject)`.
- The `WKURLSchemeHandler` conformance on a main-actor class (default isolation) — if the 26.5 SDK's protocol is not `@MainActor`, mark the class `@MainActor` explicitly or the two requirements `nonisolated` + `MainActor.assumeIsolated`.
- `nonisolated struct WorldAssetClient: Sendable` with `static let sharedUncachedSession = uncachedSession()`.
- `url.host` on a custom scheme (deprecated spelling; a warning, not an error) — replace with `url.host(percentEncoded: false)` if it warns.
- `WorldRenderWebView.makeConfiguration(assets:)`, a `static func` on the representable; `Coordinator.init(target:)` calling `super.init()` after `let` properties.
- Tests: `import WebKit` added to `WorldBuilderIntegrationTests.swift`; `configuration.urlSchemeHandler(forURLScheme:) === assets` (existential identity comparison).

### A2. Unit tests (no Tower)

```sh
xcodebuild -project Glasses.xcodeproj -scheme Glasses \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -derivedDataPath ~/Projects/Glasses-scratch/wb-final-recon-mac/dd \
  -only-testing:GlassesTests/WorldAssetTransportTests \
  -only-testing:GlassesTests/WorldRenderRevisionTests \
  -only-testing:GlassesTests/WorldRenderViewerTests \
  test-without-building
```

Expect:
- **8** in `WorldAssetTransportTests`: the whitelisted routes are recognised; 23 other URLs, a POST, a session-less manifest and an empty world are refused; each whitelisted request maps to the Tower path of the same name; no URL cache anywhere (`urlCache == nil`, both cache policies, no cookies, and `WorldRenderClient().session` too); the web view configuration's `websiteDataStore.isPersistent == false` and the scheme handler is registered; the handler's session is the pinned one or the page's own; a withdrawn appearance in a revision body is recognised; the appearance caption.
- `WorldRenderRevisionTests` **31** (29 + the appearance decode test + `testAnAppearanceOnlyChangeNeverReloadsThePage`), `WorldRenderViewerTests` **20** (the navigation test is now `testOnlyTheInitialLoadOfTheSchemePageIsAllowed`).

Record pass/fail per class. `testAnAppearanceOnlyChangeNeverReloadsThePage` is timing-sensitive (7 polls at 20 ms, with a back-off after the last change); if it flakes, name it.

### A3. Tower routes on the Mac

Use the Tower of §3, but copy the world from the Windows lane copy that HAS an
appearance artifact: `Glasses-scratch\wb-final-recon\fixit\phone-viewer\dataset\worlds\b2a75ab40d2d415d8d6ef5e4d5f0fb3d\`
(whole directory; it is imagery — keep it private, do not commit it). The Mac
Tower does not need the ASTC encoder to SERVE it.

```sh
W=b2a75ab40d2d415d8d6ef5e4d5f0fb3d; S=a8c6817e14a74e3c977fccfcdacad595; T=http://127.0.0.1:8010
curl -s "$T/worlds/$W/render/revision?session_id=$S"
#   {"session_id": S, "representation": "appearance", "revision": "S/appearance:1", "live": false,
#    "appearance": {"revision": "S/appearance:18d60c554d5e8e7089d4", "current": true}}
curl -s -D /tmp/h.txt "$T/worlds/$W/render?session_id=$S" -o /tmp/a.html; wc -c < /tmp/a.html      # about 70 KB
grep -i content-security-policy /tmp/h.txt        # ... connect-src glasses-world:
head -c 4096 /tmp/a.html | grep -o '<meta [^>]*>'  # wb-representation appearance, wb-revision S/appearance:1, the same CSP
curl -s -D - "$T/worlds/$W/render?session_id=$S&transport=tower" -o /dev/null | grep -i content-security   # connect-src 'self'
curl -s -o /dev/null -w "%{http_code}\n" "$T/worlds/$W/render?transport=https://x"                        # 422
curl -s "$T/worlds/$W/render?session_id=$S&representation=surface" | head -c 800 | grep -o 'wb-representation" content="[a-z]*"'   # surface
curl -s "$T/worlds/$W/render?session_id=$S&view=diagnostics" | head -c 800 | grep -o 'wb-representation" content="[a-z]*"'          # sparse
```

Record: the revision JSON; page bytes; the header CSP equals the `<meta>` CSP;
`wb-revision` equals the revision JSON's `revision`; the 422; surface and sparse
still served on request.

### A4. The transport, in the Simulator (Safari Web Inspector attached to the app's web view)

Open b2a75ab4 from Saved Worlds. In Web Inspector → Network:

- [ ] The document is `glasses-world://tower/worlds/<W>/render`; every other request is `glasses-world://tower/…` — the manifest once, the proxy once, **8** chunk requests — and **no** `http://` request from the page.
- [ ] Console: `fetch("https://example.com")` and `fetch("http://127.0.0.1:8010/worlds")` are refused by CSP; `fetch("glasses-world://tower/worlds")` and `fetch("glasses-world://tower/worlds/<W>/appearance/<S>/chunk/" + "0".repeat(32))` answer **404 from the handler** and the Tower log shows **no** request for either.
- [ ] Tower log during load: only `GET /worlds/<W>/render`, `/render/revision`, `/appearance/<S>/manifest`, `/proxy/…`, `/chunk/…`.
- [ ] `window.__wbAppearance` in the console: `phase: "ready"`, `encoding: "astc-6x6-rgba"`, `layers: 128`, `gpuBytes` about 20.5 MB, `errors: []`.
- [ ] After closing the viewer: `~/Library/Developer/CoreSimulator/Devices/<udid>/data/Containers/Data/Application/<app>/` contains no file with a chunk's bytes (search for the 8-byte magic `WBAPCK01`: `grep -rl WBAPCK01 .` prints nothing) and no WebKit website-data directory created at the time of the test.

### A5. What the page shows (physical iPhone, Windows Tower on port 8000)

Record a screenshot for each.

1. **Opening.** It opens in Walk at pose 99 / 198 (the desk hutch, monitor, door at right). Compare with `fixit\phone-viewer\fig\after_open_phone390.png`. The caption reads *Captured images on reconstructed geometry · 128 of 374 keyframes shown* and the native caption *The camera's own images, faces redacted, placed on the reconstructed room…*.
2. **The extension.** `__wbAppearance.gl.astc` must be `true` on the phone. If it is `false`, record it: the page falls back to 48 WebP layers (about 42 MB colour) and says so in its caption.
3. **Walk.** Step ← → through at least 10 poses and look around with one finger (left, right, up, down). Record whether it looks like the room or like triangles, and name seams, ghosting (a hand on the desk is expected around poses 120–180), swimming while turning, black cracks (proxy holes), smears.
4. **Orbit, pinch, pan.** As §5.2.
5. **Frame time.** Web Inspector → Timelines → Rendering Frames while dragging in Walk for 10 s: median and worst frame. Repeat in Orbit. Record both, and whether a frame over 100 ms happens when the camera stops (the async probe readback).
6. **Memory.** Xcode's WebContent memory gauge (or Instruments → VM Tracker) at the opening view and after a minute of dragging. Record peak MB. Expected order: 20.5 MB of GL textures and buffers plus the drawing buffer (about 7 bytes a pixel at DPR 2) plus WebKit's own baseline.
7. **Boot time.** Tap → first picture, three times: `__wbAppearance.timing.bootMs`, `openingMs`, `firstFrameMs`.
8. **Context loss.** Background the app for 2 min, return: either the picture is still there, or *The graphics context was taken away… Restoring…* then the picture at the **same camera**. Record which, and `__wbAppearance.contextLosses`.

### A6. Live append during a walk (phone + glasses + Windows Tower, `TOWER_WORLD_AUTOBUILD=true`)

Open the picture early. Once the rung reaches appearance:

- [ ] Each new appearance build (Tower log: `[appearance]` publish; revision JSON `appearance.revision` changes) is taken **without a page reload**: Web Inspector shows no new document request, `__wbAppearance.appends` increments, the camera stays where you left it, and the native *"A newer reconstruction is ready"* button does **not** appear for it.
- [ ] Count full `GET …/render` in the Tower log during the walk: 1 + rung improvements (sparse → surface → appearance) + taps on the button + at most 1 after Stop. Appearance builds add **zero**.
- [ ] After Stop, the final appearance arrives the same way (appends +1), within one poll interval (10–120 s) of its publish.

### A7. Privacy spot checks

- [ ] Relabel test (Mac Tower, copy only): edit the copied `session.json` `redaction` to `…@0.30+plausibility2` while the page is open. Within one poll the page replaces the picture with *These images are no longer served for this world…*; Web Inspector shows the handler answered the next manifest request 404; `curl …/render/revision` reports `representation: "surface"`. Restore the label afterwards.
- [ ] No texture, page HTML or manifest in the app container, `Caches/` or `tmp/` after a session (A4's grep, on the device via Xcode → Devices → Download Container).

---


Branch `world-builder/reconstruction-final-v1`, at **the final SHA named in
`docs/agent-handoffs/WORLD-BUILDER-RECONSTRUCTION-FINAL.md`**. Validate that
commit and no other: the review branches (`…-review1` to `…-review4`) are
intermediate states, and `-review1` in particular lacks every review-2 and
review-3 fix this checklist expects. This supersedes the scratch
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
| r2 S1 | The native surface caption no longer says gaps are where nothing looked; it reads *"…only where the cameras measured them. A gap is not proof that nothing is there. Not to scale."* (prefix unchanged for the UI smoke test) | §2, §5.1 |
| r2 m4 | A rung whose pages failed to draw twice is not fetched, swapped or offered again until "Try again" | §2, §5.5 |
| r2 m5 | A finished same-rung build swaps by itself only when the screen names a session; the Tower never answers `surface:None` | §2, §3 |
| r2 m6 | A worse rung is neither swapped nor offered; an offer is withdrawn when the Tower reports the shown revision again | §2 |
| r2 t1 | The refusal test now fails if the refusal is deleted (a new revision whose page carries the refused stamp) | §2 |
| r3 R1 | A face box over 25% of the frame that touches the frame edge (within 5% of the short side) is filled on facelike landmarks, as well as on re-detection; the session label is `+plausibility3` | Tower tests only |
| r3 R2 | Only refreshes UP to a rung count towards refusing it; a refresh of the rung that draws forgets its failures; a finished (`live: false`) build of a refused rung is tried once; after a revert the caption offers *"A newer reconstruction could not be drawn on this phone. Try again"* | §2, §5.5 |
| r3 R3 | `view=diagnostics` serves the sparse page even when a dense artifact exists | §3, §5.9 |
| r3 R5 | `live` turns false as soon as a stage's manifest is newer than its `running` status, so the finished world after Stop is swapped in, not offered | §5.4 |
| r3 R7 | A depth model that is neither cached nor downloadable is `unavailable` and permanent for the walk, naming the model and the cache; a first download is logged with its size and time | §3 |
| r3 iOS | A fetched page of a worse rung is never swapped in (tap or race); a page differing only in its `wb-revision` stamp is not swapped over itself; with no session named, nothing from another walk swaps by itself | §2 |

## 0. Check out the exact commit

The branch has to be pushed before a Mac can fetch it; if `git fetch` does not
deliver `origin/world-builder/reconstruction-final-v1`, stop and ask for it.

```sh
cd ~/Projects/Glasses && git fetch --all
FINAL=<the final SHA named in docs/agent-handoffs/WORLD-BUILDER-RECONSTRUCTION-FINAL.md>
git worktree add ~/Projects/Glasses-worktrees/wb-final-recon origin/world-builder/reconstruction-final-v1
cd ~/Projects/Glasses-worktrees/wb-final-recon && git rev-parse HEAD      # must equal $FINAL
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
- `WorldRenderRepresentation.withoutRevisionStamp`: `html.prefix(4096).range(of:)` (a `Substring` range) passed to `String.removeSubrange`.
- `pageEvent(.rendered)`: `if fallback != nil, let rung = state.representation` — `!= nil` on an optional labelled tuple.
- `revertRefresh()`: `rung != WorldRenderRepresentation.declared(in:)`, a non-optional compared with an optional.

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
- **9** tests in `WorldRenderRepresentationTests` (review 2 added the caption-retraction test);
- **29** in `WorldRenderRevisionTests`: 9 pure (address, page stamp, body decode, upgrade order, finished-build swap, nothing from another walk swaps by itself, a revision's session and a page without its stamp, poll interval, termination window) and 20 that drive the follower (better rung swaps; same rung offered then swapped on `showNewerPicture`; unchanged revision fetches no page; a revision the page does not carry costs one fetch; failed fetch keeps the page; kill-budget failure reverts and a new revision carrying the refused stamp is not swapped in again; a rung that failed twice is not fetched again; a worse rung is neither swapped nor offered; failures of the rung on screen do not refuse it; the finished build of a refused rung is tried once and Try again lifts the refusal; tapping an offer never swaps in a worse rung; a page that only gained its stamp is not swapped in; with no session named only the walk on screen swaps by itself; a stale offer is withdrawn; watchdog failure reverts; a first page that cannot draw still fails; unmatched-route 404 ends following after one request; contract-worded 404 is retried; diagnostics target not followed; cancellation ends the loop). The refusal tests each wait up to 3 s on a bounded poll that, on a regression, times out rather than hangs;
- **20** in `WorldRenderViewerTests`, unchanged, all green.

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

## 3. Tower on the Mac, with a world that has a built surface

A Mac cannot build surfaces, and **the canonical world on Windows does not have
one yet**: `tower\data\world_builder\worlds\b2a75ab40d2d415d8d6ef5e4d5f0fb3d\`
holds only `derived`, `sessions`, `solve` and `world.json`. A copy of that
directory as it stands has no surface rung, so §3 would answer `sparse` and §4's
surface caption check would fail against the data, not the app.

**First, on the Windows Tower, on the final SHA, before the phone test**, build
the surface of session `a8c6817e14a74e3c977fccfcdacad595` (from `tower\`, with the
Tower's venv interpreter; `<root>` is the world root the Tower is configured
with, e.g. the canonical checkout's absolute `tower\data\world_builder`):

```powershell
.venv\Scripts\python.exe scripts\world_surface.py --root <root> --world b2a75ab40d2d415d8d6ef5e4d5f0fb3d --session a8c6817e14a74e3c977fccfcdacad595
```

It adds `dense\a8c6817e…\` (the depth stage's alignment and cache, not a dense
artifact) and `surface\a8c6817e…\` beside the world's existing data, and does
not modify any existing file. It needs the MoGe-2 depth model
(`Ruicheng/moge-2-vitl`, about 1.3 GB) in the Hugging Face cache. The Windows
Tower machine has it. On a fresh machine, **pre-seed the cache while online**
before any walk or build:

```sh
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Ruicheng/moge-2-vitl', 'model.pt'))"
```

Without it, an offline build reports `unavailable` naming the model and the
cache (and a walk stops launching live surfaces for the rest of that walk);
online, the first build downloads it and logs `downloaded depth model … MB in …
s`. Record the build's printed levels: expect about **229k triangles** at the
phone level (L2) and a phone page of about **6.2 MB** (6,174,107 bytes measured
on this world, under the 6 MiB budget).

Then copy world `b2a75ab40d2d415d8d6ef5e4d5f0fb3d` as whole directories,
**including the new `surface\` and `dense\`**, to
`~/Projects/Glasses-scratch/wb-final-recon-mac/worlds/worlds/`, plus the
synthetic sparse-only fixture from `Glasses-scratch/wb-cv-sim/`. This world has
no dense rung (only `world_densify.py` builds one, and it is off by default), so
`&representation=dense` below answers 404; that is expected.

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
for q in "" "&representation=dense" "&representation=sparse"; do            # dense: 404 on this world
  curl -s "$T/worlds/$W/render?session_id=$S$q" -o /tmp/p.html; wc -c < /tmp/p.html
  head -c 4096 /tmp/p.html | grep -c 'http-equiv="Content-Security-Policy"'          # 1 for EVERY rung (m6)
  head -c 4096 /tmp/p.html | grep -o '<meta name="wb-[a-z]*" content="[^"]*">'        # both tags inside 4096
done
curl -s "$T/worlds/$W/render?session_id=$S&view=diagnostics" | head -c 800 | grep -o 'wb-representation" content="[a-z]*"'   # sparse (M1)
curl -s "$T/worlds/$W/render/nope"; echo                                   # {"detail":"Not Found"}  -> the phone stops
curl -s "$T/worlds/nope/render/revision"; echo                             # {"detail":"no world 'nope'"} -> the phone retries
```

Record: page size per rung (surface about 6.2 MB); the two meta values; that
the revision JSON's `revision` equals the surface page's `wb-revision` (**must
be equal**); that the CSP meta count is 1 on the surface and sparse rungs; the
rung for `view=diagnostics` (must be `sparse`; since review 3 R3 it is sparse
even on a world that also has a dense artifact); `live` (must be `false` with
autobuild off).

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
   - count of full `GET …/render`: must equal 1 + rung improvements + taps + 1 for the finished surface swapped in after Stop (+ at most one per rebuild if the Tower logged the m9 ERROR);
   - after Stop: polls continue through finalization, the final surface (built after Stop, `live: false`) is **swapped in by itself** without a tap, once, and once finalization is done the poll gaps grow 10 → 20 → 40 → 80 → 120 s and stay at 120 s. Record the gaps.
5. **A swap that cannot be drawn** (review M2). During §5.4, at each swap or tap, watch Console for WebContent termination. If a swap is killed past its budget or takes over 20 s, the **previous picture must come back** (a brief "Drawing the world…" then the old picture) and that revision must not be offered again. Record the WebContent footprint before and after each swap, and any revert. If none happens naturally, optionally serve L1 pages from the Tower (larger) to provoke one, and record the result. After a revert the caption shows *"A newer reconstruction could not be drawn on this phone. Try again"*; tap it once and record whether the page is fetched again and draws. A same-rung rebuild that fails does not stop the finished surface after Stop from being swapped in (review 3, R2).
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
| `WorldRenderRepresentationTests` 9 / `WorldRenderRevisionTests` 29 / `WorldRenderViewerTests` 20 | |
| Windows `world_surface.py` levels (L2 triangles), surface page bytes | |
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
