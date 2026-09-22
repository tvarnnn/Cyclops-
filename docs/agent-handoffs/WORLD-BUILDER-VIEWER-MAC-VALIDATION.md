# Mac validation: the saved-world viewer after the iOS review fixes

## Read this first: which part of this file wins

This file has two layers, written at two times. **§A wins wherever they
disagree**, and where the older layer's expectation was simply wrong it has been
corrected in place rather than contradicted further down. The review-1 addendum
that used to sit between them has been folded into §A, because it had grown
three different test counts for one class and an A3 expectation its own dataset
could not meet.

1. **§A (2026-09-17, review 2)** — the appearance rung, the `glasses-world:`
   transport, and everything the nine commits after `70d1681` changed. Run it
   first.
2. **§0–§6** — the surface-rung checklist from `world-builder/reconstruction-final-v1`.
   Still valid; still run it.

## §A. Fix-it addendum: the appearance rung and the `glasses-world:` transport

Branch `world-builder/reconstruction-fixit-ios2`, at the SHA named in the fix-it
handoff. Nothing Swift in this change has been compiled: the page and the Tower
routes were verified on Windows in headless Chrome (SwiftShader), which says
nothing about WebKit, Metal, ASTC or phone frame times.

### A0. What changed, and what each check proves

Everything below `70d1681`, in the order it landed. The first table was written
at `70d1681` and stopped there; the second is what nine commits added after it,
which is everything the wearer actually looks at.

| Change | Files | Check |
|---|---|---|
| New top rung `appearance`: a page shell that fetches the manifest, the proxy and ASTC bundles, blends the redacted keyframes on the proxy (k = 4 ULR, on-device source depth, alpha 0 = no weight, unshaded), and follows new appearance builds in place | `tower/tower/world_builder/appearance_render.py`, `appearance_viewer.html`, `tower/tower/results/world_builder_render.py`, `tower/tower/routes/geometry.py` | A3, A5, A6 |
| The page is served from `glasses-world://tower/worlds/<w>/render` by a `WKURLSchemeHandler` that proxies only this world's appearance and revision routes | `ios/Glasses/Workspaces/WorldBuilder/WorldAssetTransport.swift` (new; the app target uses a synchronized group, so no project edit) | A1, A2, A4 |
| Non-persistent `WKWebsiteDataStore`; navigation policy allows exactly the page URL; `WorldRenderClient` moved off `URLSession.shared` to an ephemeral session with no URL cache | `WorldRenderViewer.swift` | A1, A2, A7 |
| `WorldRenderRepresentation.appearance` (rank above surface) and its caption; `WorldRenderRevision.appearance` decoded but never acted on | `WorldRenderViewer.swift` | A2, A6, A9 |
| **Stop no longer kills the open page.** The revision's `appearance` gains `state` and `epoch`; `FOLLOW` keeps textures through `rebuilding`, drops on `withdrawn` but keeps polling, draws a rebuild without a reload | `appearance_viewer.html`, `appearance_pipeline.py`, `results/world_builder_appearance.py`, `results/world_builder_render.py` | A3, A6, A7 |
| The handler answers a memory copy only within 20 s of a manifest 200, else revalidates | `WorldAssetTransport.swift` | A1, A2, A7 |
| The page may reload itself when WebKit never restores a lost context | `WorldRenderViewer.swift`, `appearance_viewer.html` | A2, A5.8 |

**Added after `70d1681`, none of which the old checklist mentioned at all:**

| Commit | What it changed | Check |
|---|---|---|
| `7757860`, `3291c81`, `91e4fd9` | low-weight and hidden-face surface removal; a low-weight face many frames measured survives a few that saw past it | A5.3 (look for holes that were not there before) |
| `e838793`, `573cd53` | **the capture-supported navigation envelope** — the camera moves freely only inside a tube around the walk with a support field over position and look direction — and coverage-aware source choice | **A9** (new) |
| `21d6f1a`, `0b56122` | **the crack fill, the void fog, the edge fade, and "stop before the ugly view"** | **A10** (new), A5.3 |
| `35f1eab`, `f6d5520` | **the 1.4/1.15 frame margin, a per-keyframe on-device detail pass, pose skipping, standoff 0.55 → 1.0** | A5.1, A5.7, **A11** (new) |
| review 2 (this lane) | the caption's two retractions; `dismantleUIView`; `dropCache()` on close; the withdrawal recovery; the boot with nothing placed; the fog's arithmetic; the boot-time slicing; the smoke test's rung predicate | A1, A2, A5, A7, A8, A10, A11 |

`f6d5520`'s own message says it costs **+1.04 s of boot on SwiftShader** for the
new per-keyframe pass. Nobody has seen that on Metal. **A5.7 is a NEW BASELINE,
not a regression check** — the pre-`f6d5520` boot numbers in §5.3 below are for
a different page and must not be compared with it.

### A1. Build

As §1 below, then:

```sh
grep -E "WorldAssetTransport.swift|WorldRenderViewer.swift" ~/Projects/Glasses-scratch/wb-final-recon-mac/build.log | grep -E "error:|warning:"
```

Record every error and warning in either file. Most likely spots, most likely
first:

- `WorldAssetSchemeHandler`'s `WKURLSchemeHandler` conformance. The class is now
  explicitly `@MainActor` with both requirements `nonisolated` +
  `MainActor.assumeIsolated`, which is the fix the previous checklist told you
  to apply if it failed — so if it fails now, record the exact diagnostic,
  because the remedy already in the file did not work.
- `WorldAssetClient` no longer declares `: Sendable` (it matches its four
  sibling HTTP clients). If `static let sharedUncachedSession = uncachedSession()`
  warns as a non-concurrency-safe global, record it; it is a warning in Swift 5
  mode, and `URLSession` is `NS_SWIFT_SENDABLE` in recent Foundation.
- `WorldRenderWebView.dismantleUIView(_:coordinator:)` — a `static func` on the
  representable, new in this lane.
- `MainActor.assumeIsolated { self.start(urlSchemeTask) }` capturing a
  non-`Sendable` `any WKURLSchemeTask` in a `@MainActor` closure from a
  `nonisolated` method.
- `WorldAssetSchemeHandler.revalidate`'s `Task { [weak self] in … }` with a
  `defer` that writes `self.revalidating`.
- `WorldAppearanceFollow`, a file-scope `nonisolated enum` with an associated
  value, returned from a `nonisolated static func` on a `@MainActor` class.
- `url.host` on a custom scheme (deprecated spelling; a warning, not an error).
- Tests: `import WebKit` in `WorldBuilderIntegrationTests.swift`;
  `configuration.urlSchemeHandler(forURLScheme:) === assets`; the new
  `var asking: [Task<Void, Never>]` loop in
  `testConcurrentRevalidationsShareOneManifestFetch`.

### A2. Unit tests (no Tower)

```sh
xcodebuild -project Glasses.xcodeproj -scheme Glasses \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -derivedDataPath ~/Projects/Glasses-scratch/wb-final-recon-mac/dd \
  -only-testing:GlassesTests/WorldAssetTransportTests \
  -only-testing:GlassesTests/WorldRenderRevisionTests \
  -only-testing:GlassesTests/WorldRenderViewerTests \
  -only-testing:GlassesTests/WorldRenderRepresentationTests \
  test-without-building
```

**Counts, stated once and measured, not remembered.** Earlier versions of this
file gave three different numbers for one class; do not trust a remembered
count, count it:

```sh
cd ~/Projects/Glasses-worktrees/wb-final-recon
count() { awk -v c="$1" '$0 ~ "^final class " c {f=1; next} /^final class /{f=0}   f && /^    func test/{n++} END{print n+0}' "$2"; }
for c in WorldAssetTransportTests WorldRenderRevisionTests WorldRenderViewerTests; do
  echo "$c $(count $c ios/GlassesTests/WorldBuilderIntegrationTests.swift)"
done
echo "WorldRenderRepresentationTests $(count WorldRenderRepresentationTests ios/GlassesTests/WorldPresentationTests.swift)"
```

At this lane's head that prints **18, 40, 21, 9** (at `f6d5520`: 15, 35, 21, 9).
Record what it prints on the Mac and what `xcodebuild` executed; they must
agree. Note the range has to STOP at the next `final class`: an
`awk '/^final class X/,0'` runs to end of file and counts every later class
too, which is one of the ways this file grew three different numbers for one
class.

Timing-sensitive, name them if they flake:
`testAnAppearanceOnlyChangeNeverReloadsThePage`,
`testAWithdrawnAppearanceThatComesBackReplacesThePage`,
`testAFailedRecoveryFetchIsTriedAgainRatherThanDisarmingTheRecovery`,
`testAPageThatTookItsImageryBackIsNotReloadedUnderTheReader`,
`testARefreshThatNeverFinishesDrawingPutsTheOldPictureBack`.

**`StubbedGeometryProtocol` is process-global static state** shared by
`WorldRenderRevisionTests` and `WorldAssetTransportTests` (`reset(routes:)`
clears one table). That is fine while XCTest runs classes serially in one
process and breaks the moment parallel testing is enabled for the target. If you
see a flake that looks like one class's routes answering another's, this is it —
report it as a test-infrastructure finding, not as a product bug.

### A2b. UI tests — **run these, and expect the first run to prove a fix**

The old checklist never ran `GlassesUITests` at all, which is how a certain
regression sat in it: `TowerSmokeUITests.testOpeningASavedWorldShowsThe3DWorld`
asserted the rung caption against three prefixes and the app had gained a fourth
rung. Against the canonical validation world — which HAS an appearance artifact,
and which `WorldRenderClient.url` asks for with `viewer=appearance-1`
unconditionally — that assertion failed on the primary happy path.

```sh
TEST_RUNNER_GLASSES_UITEST_TOWER_AUTHORITY=127.0.0.1:8010 xcodebuild \
  -project Glasses.xcodeproj -scheme Glasses \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -derivedDataPath ~/Projects/Glasses-scratch/wb-final-recon-mac/dd \
  -only-testing:GlassesUITests/TowerSmokeUITests \
  -resultBundlePath ~/Projects/Glasses-scratch/wb-final-recon-mac/ui-appearance.xcresult test
```

- [ ] `testOpeningASavedWorldShowsThe3DWorld` passes against the appearance
      world. The rung-caption predicate now includes *"The camera's own
      images"*.
- [ ] The same test finds the toolbar control `world-render-reload`.
- [ ] In the `3d-world` screenshot, the caption reads
      *The camera's own images, faces redacted, placed on the reconstructed
      room. Grey haze is where no kept image looked; only cracks a few pixels
      wide are filled, from the images beside them. Not to scale.*
      **Word for word.** If it still says *"nothing is filled in"*, the build is
      older than this lane.
- [ ] The gestures after it (`swipeLeft`, `pinch`) were written for the
      sparse/orbit page and now run against the appearance page's
      capture-supported envelope. They assert only that the sheet does not move
      under them, which still holds — but watch the screenshot: a swipe should
      turn the head, not orbit.

### A3. Tower routes on the Mac

Copy the world from the Windows lane copy that HAS an appearance artifact:
`Glasses-scratch\wb-final-recon\fixit\phone-viewer\dataset\worlds\b2a75ab40d2d415d8d6ef5e4d5f0fb3d\`
(whole directory; it is imagery — keep it private, do not commit it). The Mac
Tower does not need the ASTC encoder to SERVE it.

**This world is PRE-EPOCH.** Measured on its manifest: `"epoch": null`. So
`render_revision` returns the bare `PAGE_REVISION` and its page revision is the
constant `a8c6817e…/appearance:1` for the life of that world — there is no
`@<epoch>` suffix, and there never will be unless you rebuild the appearance on
the Mac. An earlier addendum told you to expect `"revision": "S/appearance:1@<epoch>"`
and `"epoch": "<epoch>"`; that is right for a NEWLY BUILT world and wrong for
the one you were told to copy, and following it records a false FAIL.

```sh
W=b2a75ab40d2d415d8d6ef5e4d5f0fb3d; S=a8c6817e14a74e3c977fccfcdacad595; T=http://127.0.0.1:8010
# `viewer=appearance-1` is what the app declares; without it `auto` serves the
# surface, which is what an app built before the appearance page gets (WORLDS §4).
curl -s "$T/worlds/$W/render/revision?session_id=$S&viewer=appearance-1"
#   {"session_id": S, "representation": "appearance", "revision": "S/appearance:1", "live": false,
#    "appearance": {"revision": "S/appearance:<build id>", "current": true, "state": "served", "epoch": null}}
#   NOTE: "epoch": null and NO "@<epoch>" on the page revision, on THIS world.
curl -s "$T/worlds/$W/render/revision?session_id=$S" | grep -o '"representation": *"[a-z]*"'   # surface (old app)
curl -s "$T/worlds/$W/render?session_id=$S" | head -c 800 | grep -o 'wb-representation" content="[a-z]*"'   # surface (old app)
curl -s -D /tmp/h.txt "$T/worlds/$W/render?session_id=$S&viewer=appearance-1" -o /tmp/a.html; wc -c < /tmp/a.html
#   about 190 KB. NOT "about 70 KB": that was the first build's figure and stood
#   uncorrected through six lanes. 175,436 B at f6d5520; this lane adds ~8 KB.
# Compression: wire bytes gzip vs identity for one chunk (digest from the manifest)
curl -s "$T/worlds/$W/appearance/$S/manifest" | python3 -c 'import json,sys; print(json.load(sys.stdin)["chunks"][0]["digest"])' > /tmp/d.txt
curl -s -D - -H 'Accept-Encoding: gzip' "$T/worlds/$W/appearance/$S/chunk/$(cat /tmp/d.txt)" -o /tmp/c.gz | grep -i 'content-encoding\|vary\|cache-control\|etag'   # gzip, Accept-Encoding, no-store, no etag
wc -c < /tmp/c.gz; curl -s "$T/worlds/$W/appearance/$S/chunk/$(cat /tmp/d.txt)" | wc -c    # about 0.9x of the plain body
grep -i content-security-policy /tmp/h.txt        # ... connect-src glasses-world:
head -c 4096 /tmp/a.html | grep -o '<meta [^>]*>'  # wb-representation appearance, wb-revision S/appearance:1, the same CSP
curl -s -D - "$T/worlds/$W/render?session_id=$S&transport=tower" -o /dev/null | grep -i content-security   # connect-src 'self'
curl -s -o /dev/null -w "%{http_code}\n" "$T/worlds/$W/render?transport=https://x"                        # 422
curl -s "$T/worlds/$W/render?session_id=$S&representation=surface" | head -c 800 | grep -o 'wb-representation" content="[a-z]*"'   # surface
curl -s "$T/worlds/$W/render?session_id=$S&view=diagnostics" | head -c 800 | grep -o 'wb-representation" content="[a-z]*"'          # sparse
```

Record: the revision JSON verbatim; page bytes; the header CSP equals the
`<meta>` CSP; `wb-revision` equals the revision JSON's `revision`; the 422;
surface and sparse still served on request.

**Then read A8 before A7**: on a pre-epoch world the page revision can never
change, so `appearanceWithdrawnWhileShown` is the app's ONLY path to replacing
that page, and A7 is the only check that exercises it.

### A4. The transport, in the Simulator (Safari Web Inspector attached to the app's web view)

Open b2a75ab4 from Saved Worlds. In Web Inspector → Network:

- [ ] The document is `glasses-world://tower/worlds/<W>/render`; every other
      request is `glasses-world://tower/…` — the manifest once, the proxy once,
      **8** chunk requests — and **no** `http://` request from the page.
- [ ] Console: `fetch("https://example.com")` and
      `fetch("http://127.0.0.1:8010/worlds")` are refused by CSP;
      `fetch("glasses-world://tower/worlds")` and
      `fetch("glasses-world://tower/worlds/<W>/appearance/<S>/chunk/" + "z".repeat(32))`
      answer **404 from the handler** and the Tower log shows **no** request for
      either.
      **Use `"z".repeat(32)`, not `"0".repeat(32)`.** Thirty-two `0`s is a
      *valid* digest — `isDigest` accepts any 32 characters in `0-9a-f` — so it
      parses, is proxied, and the Tower log DOES show it. An operator following
      the old text records a failure that is the code working as designed.
      `…/appearance/<some other session>/manifest` is a second good probe.
- [ ] Tower log during load: only `GET /worlds/<W>/render`, `/render/revision`,
      `/appearance/<S>/manifest`, `/proxy/…`, `/chunk/…`.
- [ ] `window.__wbAppearance` in the console: `phase: "ready"`,
      `encoding: "astc-6x6-rgba"`, `layers: 128`, `errors: []`.
      For `gpuBytes` see A5.6 — it means something different now.
- [ ] **Compression through the scheme handler.** The Tower log (or Proxyman)
      shows the chunk/proxy/manifest requests arriving with `Accept-Encoding`
      containing `gzip` and answered `Content-Encoding: gzip`; the page still
      reaches `phase: "ready"` with 128 layers (so WebKit received decoded
      bytes). Record the total wire bytes for a full load (expect about
      **14.0 MB** against **18.3 MB** decoded, measured on Windows).
- [ ] **Old-app safety.** The Tower log shows `viewer=appearance-1` on
      `GET /worlds/<W>/render` and on BOTH `render/revision` polls (the app's
      and the page's proxied one). If an app build from before this branch is
      available, open the same world with it: it must show the **surface** page.

### A5. What the page shows (physical iPhone, Windows Tower on port 8000)

Record a screenshot for each. **Every expectation here was re-derived against
`f6d5520` plus this lane; the figures in §5 below are for the pre-`f6d5520`
page.**

1. **Opening.** It opens in Walk, at the pose the page scores best on *drawn
   fraction × rendered detail* at THIS canvas aspect — on the canonical world
   and a 390×844 canvas that is index **97** of 198, at 99.7% drawn. It is not
   pose 99 any more and it is aspect-dependent, so do not treat a different
   index as a failure; record the index and `__wbAppearance.opening`.
   The page's caption reads *Captured images on reconstructed geometry* with an
   **About** button. Tap About: the long text must include the walk-shortening
   sentence (*"the arrows … pass over poses that render badly or show nothing,
   at most 4 in a row"*) and the two claims in A10. The native caption above it
   is the one quoted in A2b.
   **`N of M keyframes shown` counts the PHONE tier on both sides now** (128 of
   ~128 on this world), not 128 of 374 — M used to count the Tower tier the
   phone was never going to draw.
2. **The extension.** `__wbAppearance.gl.astc` must be `true` on the phone. Also
   record `__wbAppearance.encoding` **and the manifest's `encodings` and
   `encoding_notes`**: the page falls back to WebP both when the device has no
   ASTC and when the Tower built none, and only the manifest tells you which. If
   the caption says *"the Tower built no compressed textures for this world"*,
   that is a Tower finding, not a phone finding.
3. **Walk.** Step ← → through at least 10 poses and look around with one finger.
   Record whether it looks like the room or like triangles, and name seams,
   ghosting (a hand on the desk is expected around poses 120–180), swimming
   while turning, black cracks, smears. **The arrows now SKIP poses** that
   render badly or show nothing, at most 4 in a row — so ← → moves further than
   one recorded pose sometimes. Record `__wbAppearance.lastStep` for a skip.
4. **The envelope, not an orbit.** See A9; §5.2's orbit/pinch/pan expectations
   are for the page that had a free orbit, which this one does not.
5. **Frame time.** Web Inspector → Timelines → Rendering Frames while dragging
   for 10 s: median and worst frame. Record whether a frame over 100 ms happens
   when the camera stops (the async probe readback).
6. **Memory.** Xcode's WebContent memory gauge (or Instruments → VM Tracker) at
   the opening view and after a minute of dragging. Record peak MB.
   `__wbAppearance.gpuBytes` **now includes the render targets and costs the
   drawing buffer at 8 B/px**, so it is roughly twice the number the old
   checklist expected and is the honest one. Expected order on a DPR-2 phone:
   ~13 MB ASTC colour + ~3.7 MB depth + ~29 MB of render targets and drawing
   buffer. The breakdown is in `__wbAppearance.gpu`.
7. **Boot time — A NEW BASELINE.** Tap → first picture, three times:
   `__wbAppearance.timing.bootMs`, `openingMs`, `sourceDepthMs`,
   `firstFrameMs`. Record them as the baseline for this page; do NOT compare
   with §5.3's numbers, which are the surface page's. Measured on Windows with
   SwiftShader for scale only: boot ≈ 7.3 s before this lane's slicing and
   ≈ 8.2 s after, and the longest main-thread block fell from **4.6 s to
   0.55 s** (see A11).
8. **Context loss.** Background the app for 2 min, return: either the picture is
   still there, or *The graphics context was taken away… Restoring…* then the
   picture at the **same camera**. Record which, and
   `__wbAppearance.contextLosses`. `__wbAppearance.reloadRequested` is set just
   before a self-reload; after one, `contextLosses` is back to 0 and the camera
   is at the opening pose. It must never sit on *Restoring…* for more than
   ~10 s at first, or ~90 s if the restore is fetching.
9. **Landscape, once.** Rotate on a notched iPhone. Photograph the bar and the
   status line: the Overview button and `#status` must clear the sensor housing
   (the page had no horizontal safe-area insets until this lane). Then press ←→
   a few times: the pose scores are re-measured about 300 ms after a rotation,
   so the skipping should suit the new shape rather than the old one.

### A6. Live append during a walk (phone + glasses + Windows Tower, `TOWER_WORLD_AUTOBUILD=true`)

Open the picture early. Once the rung reaches appearance:

- [ ] Each new appearance build is taken **without a page reload**: no new
      document request, `__wbAppearance.appends` increments, the camera stays,
      and the native *"A newer reconstruction is ready"* button does **not**
      appear for it.
- [ ] Count full `GET …/render` during the walk: 1 + rung improvements + taps on
      the button + at most 1 after Stop. Appearance builds add **zero**.
- [ ] After Stop, the final appearance arrives the same way (appends +1), within
      one poll interval (10–120 s) of its publish.
- [ ] **On this pre-epoch world, `FOLLOW.mustReplace` is always true** ("unknown
      is never the same"), so the first new build CLEARS and re-uploads all 128
      layers rather than appending in place. Expect a visible reload of the
      textures without a document reload, and record how long the page shows
      *Placing images N / M*. That contradicts this item's "in place"
      expectation on this dataset and is correct behaviour; a world built with
      an epoch appends.
- [ ] **The Tower is paying for two followers.** One
      `GET /render/revision?viewer=appearance-1` on the canonical world costs
      **37.9 ms** and does 2× `load_solution` + 2× `read_manifest_at`
      (161,787 B + 454,733 B of JSON), because `build_render_revision` and
      `render_revision` each compute the appearance revision. The phone polls it
      **twice every 10 s**: once natively and once from the page through the
      scheme handler. During a live walk that is ~7.6 ms/s of JSON parsing on
      the machine also running the solve. Record the count and the Tower's CPU;
      this is a known cost, not a defect to fix on the Mac.

### A7. Privacy spot checks, and the withdrawal round trip

- [ ] **Relabel (Mac Tower, copy only).** Edit the copied `session.json`
      `redaction` to `…@0.30+plausibility2` while the page is open. Within one
      poll the page replaces the picture with *These images are no longer served
      for this world…* (not "Close and reopen"); Web Inspector shows the handler
      answered the next manifest request 404; `curl …/render/revision` reports
      `representation: "surface"`.
- [ ] **Then, before restoring the label**, background the app for 2 minutes and
      return: the picture must NOT come back (the memory copy is not answered
      without a manifest 200; the Tower log shows the manifest answered 404).
- [ ] **Restore the label** and run `world_appearance.py --force` on the copy.
      Record, in this order and separately:
      (a) whether the picture returns **in place with the camera kept** — that
      is the page's own follower, and it is what should happen;
      (b) whether the app *also* replaces the document a few seconds later.
      **It must not.** The app now waits one poll and checks whether the page
      fetched a served manifest through the handler; if it did, the app does
      nothing. A camera reset to the opening pose here is review 2's M-2 coming
      back, and it costs ~13 MB of chunks;
      (c) the chunk request count in the Tower log for that window.
- [ ] **Then repeat with the Tower stopped for the ten seconds after the label
      is restored.** The picture must still come back once the Tower is up. This
      is the one check for review 2's M-1: the recovery used to disarm itself on
      a single failed fetch, and on this pre-epoch world it is the only path the
      app has.
- [ ] **A crash is not a relabel.** If you can make the Tower's appearance code
      raise (e.g. `chmod 000` the appearance directory mid-session), the page
      must say *The Tower could not answer for this world's images just now* and
      NOT *its redaction record changed*. `curl …/render/revision` shows
      `"state": "unavailable"`.
- [ ] No texture, page HTML or manifest in the app container, `Caches/` or
      `tmp/` after a session: `grep -rl WBAPCK01 .` prints nothing (Xcode →
      Devices → Download Container). **Note what this cannot see:** the iOS
      app-switcher snapshot is a rendered image of the room, written to
      `…/Library/SplashBoard/Snapshots/…` when the app is backgrounded, and no
      grep for chunk bytes will find it. Nothing in this app blurs or covers on
      `scenePhase` change. Record it as a finding; it is not a regression of
      this lane (the surface page had the same exposure) but this is the first
      rung whose pixels are the room.

### A8. Teardown and the memory copy

- [ ] Close the viewer with chunks in flight (throttle the link to make that
      easy). No `NSInternalInconsistencyException`, no crash. This is review 2's
      M-3: there was no `dismantleUIView` at all, so nothing stopped the load,
      cleared the delegate or cancelled the scheme tasks.
- [ ] Open and close the viewer **ten times**, recording the app process's
      footprint after each. It must not climb by ~14 MB a time. `tearDown()` on
      `.onDisappear` drops the copy; before this lane `dropCache()` had no
      caller and up to 64 MB was left to ARC.
- [ ] Tap **Reload** in the toolbar with the picture on screen. The page is
      fetched again and drawn; the Tower log shows ONE `GET …/render` and the
      chunks are **not** re-downloaded (the copy belongs to the viewer, not to
      the web view).

### A9. The navigation envelope (new: `e838793`, `573cd53`, `f6d5520`)

Nothing in the old checklist covers this, and it is the largest change to how
the page feels.

- [ ] **There is no free orbit.** One finger turns the head; two fingers move
      sideways and up/down; pinch moves forward and back. Push outward until it
      resists: the resistance must be smooth and must never hard-stop, and the
      hint *Not captured beyond here* appears at the edge. Record
      `__wbAppearance.support()` at the edge.
- [ ] Release while pushing out: the camera drifts back inside without a jump.
- [ ] **Overview** (the button): a raised vantage that shows the room in one
      frame. Record whether it does, in portrait and in landscape — on the
      canonical world portrait works and landscape is a known open question.
- [ ] `__wbAppearance.navStats()` after boot: record `voxels`, `samples`,
      `sources`, `inputMs`, `jobMs`, `ms`. On Windows: 2031 / 6000 / 128 /
      ~230 ms / ~87 ms / ~730 ms.

### A10. The two claims the caption makes (new: `21d6f1a`)

The page fills thin cracks and paints voids with fog. Both are visible and both
are now named in the caption, and this is the check that they match.

- [ ] Find a wall seen at a grazing angle (right of the door on the canonical
      world). The black speckle of the proxy's pinholes should be closed. Zoom
      in: a closed crack must carry the wall's own texture, never a smear of a
      different colour. Anything wider than a few pixels must stay empty.
- [ ] Look into a real void (turn toward the unwalked half of the room). It must
      be a **grey haze that is clearly darker than the room beside it**, with no
      texture and no detail, fading to the plain background as you look further
      in. **If any part of a void is brighter than the drawn room next to it,
      that is review 2's P-2 not fixed** — photograph it and record the pose.
- [ ] Read the native caption and the page's About side by side. Neither may say
      "nothing is filled in". Both must bound the fill ("cracks a few pixels
      wide") and describe a void as haze rather than as a dark gap.

### A11. Boot and rebuild stalls on Metal (new; review 2 P-3 and P-4)

`f6d5520` added a per-keyframe render pass to boot and nobody has measured it on
a phone.

- [ ] Web Inspector → Timelines, record the **longest main-thread block** during
      boot, and again when a live rebuild is accepted during A6.
      `__wbAppearance.timing.sourceDepthMs` and `__wbAppearance.nav.inputMs`
      name the two suspects.
- [ ] Measured on Windows/SwiftShader, before and after this lane's slicing:
      longest boot block **4.6 s → 0.55 s**, boot **7.3 s → 8.2 s** (yielding
      ~40 times is not free), `navInput` **300 ms in one block → ~230 ms spread
      over slices of ≤ 130 ms**. A phone's GPU is roughly ten times faster, so
      expect the blocks to be far smaller — but they are time-budgeted (12 ms
      for the depth pass and the opening scan, 14 ms for the envelope), so the
      SHAPE should hold: many small blocks, no large one.
- [ ] While the page boots, keep a finger moving on the screen. It must stay
      responsive. That is the whole point of the change.

### A12. What to paste into the handoff for §A

| Item | Value |
|---|---|
| HEAD validated | |
| Build: errors/warnings in `WorldAssetTransport.swift`, `WorldRenderViewer.swift` | |
| Test counts printed by the `awk` above vs executed | |
| `GlassesUITests/TowerSmokeUITests` per test | |
| Native caption, word for word, from the `3d-world` screenshot | |
| A3 revision JSON verbatim, and page bytes | |
| A4: chunk requests, `"z".repeat(32)` refused, wire bytes | |
| A5.1 opening index and `__wbAppearance.opening` | |
| A5.2 `gl.astc`, `encoding`, manifest `encodings` / `encoding_notes` | |
| A5.6 WebContent MB, `__wbAppearance.gpu` breakdown | |
| A5.7 bootMs / openingMs / sourceDepthMs ×3 (NEW BASELINE) | |
| A5.9 landscape photographs | |
| A6 appends, full `/render` count, re-upload on the first new build | |
| A7 (a)/(b)/(c), the Tower-stopped repeat, the `unavailable` check | |
| A8 teardown: exception? footprint over ten opens? Reload cost | |
| A9 navStats, Overview in both orientations | |
| A10 crack fill and void fog photographs | |
| A11 longest main-thread block, boot and rebuild | |
| App-switcher snapshot: recorded as a finding? | |

---
## §0-§6. The surface-rung checklist (2026-09-16)

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
`wb-representation` (review m5). Against a world that HAS an appearance
artifact, run §A2b instead: the rung is `appearance` and the caption is a
fourth sentence. In the `3d-world` screenshot, for a world without one:

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
