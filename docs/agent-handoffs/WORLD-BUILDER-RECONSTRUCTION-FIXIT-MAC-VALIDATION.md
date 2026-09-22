# World Builder reconstruction fix-it: Mac validation

The Windows campaign (`WORLD-BUILDER-PHOTOGRAPHIC-FIXIT.md`) ended **NOT READY
FOR MAC/PHYSICAL RETEST — for one external reason: no Swift in this campaign
has ever been compiled.** This is the Mac's answer to that one reason, plus
everything else a Mac can establish about the candidate without a phone, the
glasses or the Windows Tower.

| | |
|---|---|
| **SHA tested** | **`5da13c838192b8491614f053e2fca388dc88d809`** (`origin/world-builder/reconstruction-fixit-v1`, tip at 2026-09-21 23:17:37 -0400, unchanged through the whole run; the campaign's last content commit `c8d8456` is its parent) |
| **Handoff docs verified present at that SHA** | `docs/agent-handoffs/WORLD-BUILDER-PHOTOGRAPHIC-FIXIT.md`, `WORLD-BUILDER-VIEWER-MAC-VALIDATION.md`, `WORLD-BUILDER-RECONSTRUCTION-FINAL.md`; contracts `WORLD-BUILDER-APPEARANCE.md`, `WORLD-BUILDER-DENSE.md`, `WORLD-BUILDER-SURFACE.md`, `WORLD-BUILDER-IOS.md`, `WORLD-BUILDER-WORLDS.md` |
| **Checkout** | detached worktree `~/Projects/Glasses-worktrees/wb-reconstruction-fixit-mac-v1`, `git rev-parse HEAD` == the SHA above, tree clean; the canonical checkout was not touched |
| **Xcode / macOS / Simulator** | Xcode 26.6 (17F113); macOS 26.5.2 (25F84); iPhone 17 Pro, iOS 26.5 (23F77) |
| **Tower on the Mac** | fresh venv `~/Projects/Glasses-scratch/mac-gate-2026-09-21/venv` (Python 3.12.5) from this branch's `pyproject.toml` with `[dev,appearance,transients]` + CPU torch + scikit-image; **no open3d (dlopen fails: Homebrew libusb absent), no moge (GitHub-only), pycolmap removed (duplicate-OpenMP crash against torch)** — see §6 |
| **Product code modified** | **none.** Nothing in `ios/` or `tower/` was changed to make anything pass. The only commit this gate adds is this document |
| **Verdict** | **§9** |

---

## 0. The one-paragraph version

**The Swift compiles, and it works.** Debug, Release and the test target build
clean with zero errors and zero new warnings; every one of the campaign's
"written, never compiled" test classes runs and passes at exactly the counts
the checklist predicts (18/40/21/9), six times over with no flake; the full
unit suite is 1036/0. Against a Mac Tower serving a fabricated appearance
world, the app fetched the appearance page through its new
`glasses-world:` scheme handler, the page booted inside `WKWebView`, drew the
keyframe imagery, and showed the checklist's caption **word for word**; the
Tower log shows the manifest, proxy and chunks arriving only through the
handler. Two independent reviewers found **no HIGH** defect; their MEDIUMs
(§5) are recorded, not fixed. The Tower suite's 174 macOS failures are, by
individual traceback, the missing native reconstruction stack (open3d) and a
pre-existing nine-test numeric baseline — plus one real packaging finding for
the Tower lane (§6.1).

---

## 1. Exact commands used

All from the worktree; `$G` = `~/Projects/Glasses-scratch/mac-gate-2026-09-21`.

```sh
# 0. Checkout
git fetch origin; git ls-remote --heads origin refs/heads/world-builder/reconstruction-fixit-v1
git worktree add --detach ~/Projects/Glasses-worktrees/wb-reconstruction-fixit-mac-v1 5da13c838192b8491614f053e2fca388dc88d809
git rev-parse HEAD                    # 5da13c838192b8491614f053e2fca388dc88d809
python3 -c "…NUL-byte scan over *.html *.swift *.py *.md…"   # []

# 1. Builds (ios/)
xcodebuild -project Glasses.xcodeproj -scheme Glasses -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath $G/dd build-for-testing
xcodebuild … -configuration Debug   -derivedDataPath $G/dd-Debug   clean build
xcodebuild … -configuration Release -derivedDataPath $G/dd-Release clean build

# 2. Unit tests
xcodebuild … -derivedDataPath $G/dd -test-timeouts-enabled YES -maximum-test-execution-time-allowance 60 \
  -only-testing:GlassesTests/WorldAssetTransportTests -only-testing:GlassesTests/WorldRenderRevisionTests \
  -only-testing:GlassesTests/WorldRenderViewerTests -only-testing:GlassesTests/WorldRenderRepresentationTests \
  test-without-building                                   # ×6
xcodebuild … -only-testing:GlassesTests test-without-building        # full suite

# 3. Tower (tower/, port 8010; 8000 is held by VS Code on this machine)
TOWER_CAPTURE_ROOT=$G/capture TOWER_WORLD_ROOT=<root> TOWER_WORLD_AUTOBUILD=false \
  $G/venv/bin/python -m uvicorn tower.main:app --host 127.0.0.1 --port 8010
#   <root> = $G/world_builder            (the sparse fixture from the 2026-09-14 gate)
#   <root> = $G/world_builder-appearance (the fabricated appearance world w1/s1, §3.4, plus the sparse fixture)

# 4. UI smoke (the TEST_RUNNER_ prefix is load-bearing)
TEST_RUNNER_GLASSES_UITEST_TOWER_AUTHORITY=127.0.0.1:8010 xcodebuild … \
  -only-testing:GlassesUITests/TowerSmokeUITests test-without-building

# 5. Contract checks
$G/venv/bin/python ios/scripts/contract-drift-check.py --tower http://127.0.0.1:8010
$G/venv/bin/python ios/scripts/cross-stack-constants-check.py
$G/venv/bin/python ios/scripts/swift-structure-check.py

# 6. Tower suites (tower/, venv python, roots under $G)
python -m pytest -q -p no:randomly --timeout=900 tests -k "world_builder or surface or appearance or dense \
  or geometry or render or result_channel or contract or redaction or artifact_paths or consistency or transient"
python -m pytest -q -p no:randomly --timeout=900 --ignore-glob="tests/*world*" tests
```

---

## 2. Results at a glance

| Gate | Result |
|---|---|
| NUL-byte scan | `[]` |
| Test-target build (`build-for-testing`) | **TEST BUILD SUCCEEDED**, 0 errors |
| Debug `clean build` | **BUILD SUCCEEDED**, 0 errors, 8 warnings = the recorded baseline (7 + the `<unknown>` echo), **0 new** |
| Release `clean build` | **BUILD SUCCEEDED**, 0 errors, the identical 8 warnings, **0 new** |
| Warnings in `WorldAssetTransport.swift`, `WorldRenderViewer.swift`, `TowerSmokeUITests.swift` | **none** |
| Warnings in `WorldBuilderIntegrationTests.swift` | only the pre-existing `subscribeCount` set (lines 507–527, 3151–3169, blamed before this branch); 0 new |
| The four campaign test classes | **88 executed, 0 failures** — 18 / 40 / 21 / 9, equal to the source counts the checklist's `awk` prints |
| Same four, repeated | **6 completed runs, 88/0 every time**; one attempt executed nothing (simulator failed to install the bundle — `containermanagerd/Dead/temp…` — while pip was writing wheels; re-run clean) |
| Full `GlassesTests` | **1036 executed, 0 failures**, 106 classes, no `Restarting after…` |
| `contract-drift-check.py` (live Tower) | **AGREEMENT** — every contract the Tower stated is implemented |
| `cross-stack-constants-check.py` | agreement — the Swift names every constant and key the Tower sends |
| `swift-structure-check.py` | clean |
| UI smoke, sparse fixture | **3 passed / 0 failed / 0 skipped** (171 s) |
| UI smoke, appearance world | §3.5: the appearance checks (`:177`–`:334`) pass; the final `Back to live` step fails on a test-driver hit-test, not the product |
| Tower, campaign selection (macOS) | 1879 selected: 1679 passed / 200 failed / 23 skipped; with scikit-image: 174 failed — **every one classified, none attributable to the campaign** (§6) |
| Tower, everything else (`--ignore-glob="tests/*world*"`) | §6.4 |
| Reviews (two independent adversarial passes) | **no HIGH**; 2 + 3 MEDIUM recorded (§5) |

---

## 3. Detail

### 3.1 Build

The checklist's "most likely spots" (`WKURLSchemeHandler` conformance with
`nonisolated` + `MainActor.assumeIsolated`, `dismantleUIView`, the
`revalidate` Task with `defer`, `WorldAppearanceFollow`, `url.host` on a
custom scheme, `import WebKit` in the test target) all compile without a
diagnostic. Debug and Release warning lists are byte-identical to each other
and to the 2026-09-14 baseline: `DocumentMemoryDecoder.swift:268`,
`DocumentMemoryLibrary.swift:731/757/786`, `ObjectMemoryCopy.swift:1503/1504`,
`WorldGeometry.swift:318` (+ its `<unknown>:0` echo).

### 3.2 Unit tests

| Run | Executed | Failed |
|---|---|---|
| four classes, run 1 | 88 | 0 |
| repeats 1–4, 5b, 6 | 88 each | 0 each |
| full suite | **1036** | **0** |

Per class in the full suite: `WorldAssetTransportTests` 18, `WorldRenderRevisionTests`
40 (4.2 s), `WorldRenderViewerTests` 21, `WorldRenderRepresentationTests` 9. None of
the five timing-sensitive tests the checklist names flaked in seven runs. The
previous gate's 968 became 1036: the 68 are the campaign's tests, all of which
ran here for the first time.

### 3.3 Tower routes on the Mac (checklist §3 / §A3)

Sparse fixture (`a1b2c3d4…/1111aaaa…`, no appearance artifact):

```
render/revision?session_id=S                  → 200, Cache-Control: no-store,
  {"session_id":S,"representation":"sparse","revision":"S/sparse","live":false,
   "appearance":{"revision":null,"current":false,"state":"absent","epoch":null}}
render?session_id=S&viewer=appearance-1       → 118,851 B; CSP <meta> count 1;
  <meta name="wb-representation" content="sparse">, <meta name="wb-revision" content="S/sparse">
  header CSP: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'
render?transport=https://x                    → 422
render/nope                                   → {"detail":"Not Found"}          (the phone stops)
/worlds/nope/render/revision                  → {"detail":"no world 'nope'"}    (the phone retries)
representation=dense / surface                → 404 / 404 on this world (expected)
```

Fabricated appearance world (`w1/s1`, §3.4) — the A3 block, on an
**epoch-bearing** world (the canonical Windows world is pre-epoch; this one is
not, so the `@<epoch>` suffix is present, as the checklist says it is for a
newly built world):

```
render/revision?session_id=s1&viewer=appearance-1 →
  {"session_id":"s1","representation":"appearance","revision":"s1/appearance:1@18d787f9ac3bcc208891",
   "live":false,"appearance":{"revision":"s1/appearance:18d787f9ac3bcc208891","current":true,
   "state":"served","epoch":"18d787f9ac3bcc208891"}}
render/revision?session_id=s1 (no viewer)     → representation "surface"      (old-app safety)
render?session_id=s1&viewer=appearance-1      → 200, 271,761 B
  header CSP: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src glasses-world:
  <meta name="wb-representation" content="appearance">
  <meta name="wb-revision" content="s1/appearance:1@18d787f9ac3bcc208891">   == the revision JSON's revision
appearance/s1/manifest                        → 200, X-World-Redaction=<label>, X-World-Imagery=redacted, currency current
chunk ×4 (astc 16 / webp 16 / astc 8 / webp 8), proxy 13,782 B → 200, bytes == manifest
proxy with Accept-Encoding: gzip              → content-encoding: gzip, 4,574 B on the wire, Vary: Accept-Encoding
representation=surface → surface meta; view=diagnostics → sparse meta
```

### 3.4 The appearance world this Mac served

The canonical dataset lives only in Windows scratch, the Windows Tower was not
reachable over Tailscale, and the Mac cannot run `build_appearance` (open3d's
`ProxyCaster`). Serving needs only numpy, so a world was **fabricated** by
script from the Tower's own writers and the shapes in `appearance_pipeline.py`:
`$G/appearance-fixture/make_world.py` → `$G/appearance-fixture/worldroot/worlds/w1/`
(24 phone-tier keyframes of a synthetic wall pattern plus a per-keyframe
colour band, 2 chunk groups × {ASTC, WebP}, a 16-face proxy with a zeroed
colour block, trusted redaction label, `imagery_source: redacted`,
`input_digest` and `surface_built_at` consistent so `current: true`). It is
**not the wearer's room** and proves transport and rendering, not
reconstruction quality.

First attempt lacked `sessions/s1/events.jsonl`; without a `session_stopped`
event the Tower's `_lifecycle` (`results/world_builder.py:1336`) reports
`idle`, the phone renders "No world yet" with a disabled Picture button, and
the smoke test failed at `:304`. A real engine-written world cannot lack the
journal (`engine.py:557-587` writes `end_reason` and `session_stopped` in the
same method), so this was a fixture defect, fixed in the fixture. Recorded
here because the *symptom* is worth knowing: a world whose journal is lost
presents as "No world yet" even with a complete finalization on disk
(`_lifecycle` deliberately does not fall back to `ended_at`/`finalization`).

### 3.5 UI smoke against the appearance world

`testASessionWithoutGeometrySaysSoOnTheRow` passed; `testTheLabConnectsAndSwitchesExperimentsFromItsOwnScreen`
passed; `testOpeningASavedWorldShowsThe3DWorld` passed every assertion from
`:177` to `:334` and failed at `:339`.

What passed, with the Tower log as witness:

- Saved worlds listed the world; the `Complete` row opened the 3D world with
  one tap; the `WKWebView` rendered; the rung caption was read from
  `wb-representation` (`:249`).
- **Native caption, word for word** (from the `3d-world` screenshot):
  *The camera's own images, faces redacted, placed on the reconstructed room.
  Grey haze is where no kept image looked; only cracks a few pixels wide are
  filled, from the images beside them. Not to scale.*
- The page's own caption *Captured images on reconstructed geometry* with
  **About**; the compass cue ("photographed from here"); **Best view**,
  **Face the room**, ← →, **Reset**; "20 / 24"; the envelope hint *Not captured
  beyond here* after the swipe; the imagery drawn (green banded wall on the
  left, honest dark beyond it).
- `world-render-reload` present; swipe and pinch did not move the sheet.
- Details: *World w1 / Session s1 / The solver's own view of this session is
  under Diagnostics on the world screen.*
- Back → Saved worlds → Close → workspace: *Looking at saved world Appearance
  fixture (Mac).*, **Saved**, **Open the 3D world**, Keyframes 24, Geometry
  "16 · sparse point cloud", Camera poses 24, Diagnostics disclosure.
- Tower log for the load: `GET /worlds/w1/render?…&viewer=appearance-1`,
  `/render/revision?…&viewer=appearance-1` (the app's poll and the page's
  proxied one), `/appearance/s1/manifest`, `/proxy/<digest>`, two
  `/chunk/<digest>` — and nothing else. The page's CSP is
  `connect-src glasses-world:`, so those chunk/proxy/manifest requests could
  only have arrived through the scheme handler. **A4's transport claim holds
  in the Simulator.**

What failed: `XCTAssertTrue(tap(app.buttons["Back to live"], until: !looking.exists))`.
Reproduced three times. `returnToLive()` → `TowerWorldBuilderClient.followLive()`
sets `inspection = .live` **synchronously** (`TowerWorldBuilderClient.swift:960-964`),
which removes the banner with no Tower round trip — so a tap that reached the
button would have satisfied the predicate at once. It did not reach it: in one
run the simulator's log shows SpringBoard's `_UISystemGestureWindow`
recognising a system gesture during the helper's swipe and the app dropping to
the background (the frame capture shows the home screen and the Tower saw the
socket close); in another the app stayed in the foreground and three taps on
the banner button — which sits under the navigation bar once the helper has
scrolled Diagnostics into view — had no effect. The same test passes 3/3 on
the sparse fixture, whose workspace is shorter. This is the third documented
instance of `reveal`/`tap` fragility in this file (the 2026-09-14 gate fixed
one in each direction). **A test-helper finding for the iOS lane, not a
product defect; the helper was not changed here** because the brief forbade
changing code to make validation pass and the point of the run was the
appearance rung, which it proved.

### 3.6 Contract checks

`contract-drift-check.py`: 13 ids implemented, 8 served, **AGREEMENT**;
`cross-stack-constants-check.py`: agreement (includes the campaign's new wire
keys `imagery_source`, `imagery_retention`); `swift-structure-check.py`: clean.

---

## 4. What this Mac could and could not prove

Proved: the Swift compiles in both configurations with no new warning; 1036
unit tests pass and the 88 campaign tests pass repeatedly; the
`glasses-world:` handler proxies exactly the routes the page fetches, and the
appearance page boots, draws and reports its rung inside `WKWebView` on iOS
26.5; the contracts agree; the sparse and appearance rungs are both served and
captioned truthfully; the Tower's render/revision/manifest/chunk/proxy routes
behave as A3 specifies on both a pre-epoch-shaped and an epoch-bearing world.

Not proved, and not provable here: everything in the checklist's A4–A11 that
needs a phone, Metal, ASTC on real hardware, Web Inspector, frame times,
WebContent memory, context loss, landscape safe areas, live append during a
walk, the privacy relabel round trip, teardown footprint over ten opens, the
navigation envelope's feel, crack fill and void fog on the real room. The
simulator has no DAT device, so no frame was ever streamed; the Mac built no
surface and no appearance, so nothing here says anything about reconstruction
quality. The WebGL page ran on the simulator's software/Metal path, which
says nothing about a phone's GPU.

---

## 5. Reviewer findings (recorded, not fixed)

Two independent adversarial reviews of `git diff 869d715 5da13c8 -- ios`,
each by an agent that had not written the code, each verified against the
source by the gate before being recorded. **Neither found a HIGH.**

### 5.1 Concurrency / lifecycle

- **MEDIUM — a transient non-200 on the page's proxied revision poll wipes
  the memory copy.** `WorldAssetTransport.swift:363-366`: `record(.renderRevision)`
  calls `drop()` on any `status != 200`, including the Tower's documented
  mid-build "no geometry yet" 404 and any 5xx — answers the page itself treats
  as "keep drawing" (`appearance_viewer.html` `FOLLOW.decide`) and the native
  follower treats as "ask again" (`WorldRenderViewer.swift:862-867`). Cost in
  the common case: the next reload re-downloads every bundle (~13 MB). In the
  narrow case — the error lands in the ≤10 s between the page's own served
  manifest fetch and the app's next poll during a withdrawal recovery —
  `servedAppearanceToPageAt` is gone, `pageRecovered` is false, and the app
  reloads the document under the reader (review 2's M-2, back). The test
  `testAnyAnswerThatWithdrawsTheAppearanceDropsTheCopyAndItsAuthorisation`
  asserts the current behaviour. Fix shape: drop only on a 200 body naming no
  served appearance, plus the GONE 404s.
- **MEDIUM — the withdrawal recovery has no bound when `refresh` returns false
  for a non-transport reason** (`WorldRenderViewer.swift:890-908` with
  `:1115-1118`): a lower-rung page or a refused revision keeps the flag armed
  and the interval at 10 s, so a Tower serving "a lower rung while the revision
  route reports appearance" (`world_builder_render.py:437-441`) is asked for the
  multi-MB page every 10 s for the life of the screen.
- LOW: no eviction in `WorldAssetMemory` below the 64 MB cap (stale bundles
  held for the viewer's life); late responses recorded under a switched
  session for unpinned viewers (bounded by the 30 s timeout);
  `refusedRevisions` is never cleared by `load()` although its comment says
  every refusal is forgotten; `load()` does not reset the follower's back-off
  (up to 120 s before the first poll after Reload); a one-run-loop window in
  which a reverted page's `didFinish` is attributed to the previous page.
- Checked and sound: answer-after-stop (`isAttached && liveTasks.remove(id)`
  in the same synchronous section as `respond`); `ObjectIdentifier` reuse;
  never-answered tasks bounded by the page's 45 s abort; the `nonisolated`
  witnesses; `revalidate` dedup; one follow loop; attach/detach ordering;
  termination and self-reload budgets; no `@Published` mutation inside a view
  update; no retain cycles; the privacy posture (non-persistent store,
  ephemeral session with no cache, `no-store`, whitelist equal to the Tower's
  `routes_for`, `tearDown()` on `.onDisappear`).

### 5.2 Contract (iOS ↔ Tower)

- **MEDIUM — after a withdrawal on an epoch-bearing world, "the app does
  nothing" is not what the code does.** `WorldRenderViewer.swift:884-885`:
  `.leaveItToThePage` has no `continue`, so control falls into the generic
  rung rules at `:922-961`. A withdrawal guarantees a new epoch
  (`appearance_pipeline.py:908-916`: textures never carry over), the page's
  `CONFIG` embeds `build_id`/`appearance_revision` (`appearance_render.py:260-261`),
  so `withoutRevisionStamp` differs and `refresh` swaps (`:1103-1105`): a
  document reload and a camera reset — exactly what `WORLD-BUILDER-IOS.md`
  §10 says must not happen. The single test of the path uses a constant
  pre-epoch stamp, which is why it passes; the canonical Windows world is
  pre-epoch, which is why A7 on it would not show this. Any world the
  campaign's pipeline builds today carries an epoch.
- **MEDIUM — a crash inside the serving gate is reported as `withdrawn`**
  (`results/world_builder_appearance.py:240-249`, `except Exception → WITHDRAWN`),
  which the page renders as "its redaction record changed, or its imagery was
  removed" — contradicting `WORLD-BUILDER-APPEARANCE.md` ("a crash is not a
  privacy event"); the `unavailable` mapping only sees exceptions that escape
  `appearance_revision`. Tower-side.
- **MEDIUM — the page does not "keep polling" when its manifest 404s at
  boot** (`appearance_viewer.html:4819-4841`: `loadRevision` throws, `fail()`,
  `follow()` never starts). The app's `replaceThePage` fallback recovers it one
  poll later, so a page-claim mismatch, not a dead end.
- LOW: the research marker is read from `CONFIG` at compose time, not from
  the manifest drawn; `absent` is also the answer for an imagery-source
  mismatch; a transient `OSError` in `read_world` becomes "This world is no
  longer on the Tower" until the next poll; two doc nits (CSP `<meta>`
  position, which `epoch` the page compares).
- Checked and agree: no decoder refusal is reachable (`decodeRevision` needs
  only a non-empty `revision`); every `appearance.state` and representation
  the Tower emits is handled; query parameters match on both sides; the two
  404 shapes are told apart exactly as the contract says; the handler never
  forwards `Content-Encoding`, so WebKit is never asked to decode plain bytes;
  header and `<meta>` CSP come from one function; every page fetch is
  absolute `glasses-world://tower/…`; the handler whitelist equals
  `routes_for` including the digest rule; captions are verbatim the IOS §10
  sentences; kill and self-reload budgets as documented.

---

## 6. Tower suite on macOS

### 6.1 A packaging finding for the Tower lane

`tower/world_builder/surface.py:1140` (`from skimage import measure`, the
marching-cubes step) needs **scikit-image**, which `pyproject.toml` declares
nowhere — it reaches the Windows venv only transitively through the `ocr`
extra (easyocr). A fresh venv built from the extras the campaign names for
these stages (`[dense,appearance,transients]`) cannot extract a surface; on
this Mac 111 tests hit `No module named 'skimage'` before it was added by
hand. Add it to the `dense` or a `surface` extra.

### 6.2 The campaign selection, classified

First run 1679 / 200 / 23; with scikit-image, the failing files re-run:
**174 failed, 579 passed, 3 skipped**. Every one of the 174 was matched to
its own traceback and captured log (`$G/logs/pytest-classify-buckets.txt`):

| cause | count |
|---|---|
| open3d — `import open3d` raises `ImportError: dlopen(… libusb-1.0.0.dylib …)` (Homebrew libusb not installed on this Mac); asserted directly | 53 |
| cascades of the same — a surface/appearance stage records `unavailable`/`failed` because `surface.py:1961` or `:2206 decimate` imported open3d; the captured log carries the dlopen line in every section | 109 |
| pycolmap not importable — the env-check test's own guard that it runs on the Tower venv | 1 |
| Windows-only — `cmd /c mklink /J`, and a backslash traversal that is a literal filename on POSIX | 2 |
| **the nine below** | 9 |

The nine: `test_world_builder_pose_accuracy.py` ×7 (strafe seeds 1000/1002/
1003/1004: the classical pair gate publishes a pose 87° from the truth as
`solved`), `test_world_builder_point_quality.py` ×1 (a point at infinity from
parallel rays passes the cheirality gate on the sign of floating-point
noise), `test_world_builder_recovery_safety.py` ×1 (a ratio bound in a
thermometer-style measurement). **These are, by file and by count, exactly the
nine the 2026-09-14 Mac gate recorded at `55bc24c`, before this campaign
existed** (`WORLD-BUILDER-MAC-INTEGRATION.md` §3.3). That gate blamed them on
pycolmap; this run's tracebacks show pycolmap is not on their code path — they
run the pure-OpenCV classical backend (`ORB` + `USAC_MAGSAC`, unseeded), whose
keypoints and RANSAC samples differ on arm64/NEON/Accelerate from x86-64.
The Mac baseline stands; but the pose-accuracy seven are the one group whose
failure mode is the one the test exists to forbid (a confidently wrong pose
reported `solved`), and they deserve a Windows re-check under varied RANSAC
seeds rather than a permanent "Mac-only" label. Not introduced by this
campaign either way.

No failure involved model weights, network, torch, path case, timestamps or
fork/spawn.

### 6.3 Why open3d and moge were not installed

open3d's macOS wheel links `/opt/homebrew/opt/libusb`; installing libusb is a
system change this gate declined to make on the owner's machine without
asking. moge is a GitHub-only package. pycolmap 4.2.0 does install on macOS
now, but importing it beside torch in one process aborts with "multiple
copies of the OpenMP runtime", so it was removed rather than run under
`KMP_DUPLICATE_LIB_OK` ("may silently produce incorrect results"). None of
this affects serving; the Mac is not the Tower host.

### 6.4 Everything else

`--ignore-glob="tests/*world*"`: **2277 passed / 7 failed / 68 skipped**
(221 s). The 7: three PowerShell launcher tests (no `pwsh`, the recorded
baseline); `test_capture_is_off_by_default`, caused by this gate's own
exported `TOWER_CAPTURE_ROOT` (passes with it unset — the 2026-09-14 gate
tripped the same wire); and three scene/document capability tests that assume
a host with torch also has the `ml` extra (`torchvision`, `timm`), which this
venv deliberately does not. **0 real**, the same shape as the previous gate's
2278 / 0 real.

---

## 7. Temporary resources this gate created

- Worktree `~/Projects/Glasses-worktrees/wb-reconstruction-fixit-mac-v1` (detached at the SHA; this document committed on branch `validation/wb-reconstruction-fixit-mac-v1`, **not pushed**).
- `~/Projects/Glasses-scratch/mac-gate-2026-09-21/` (~3.8 GB): `venv/`, `dd/`, `dd-Debug/`, `dd-Release/`, `logs/`, `*.xcresult`, `ui-shots*/`, `frames/`, `world_builder/` (sparse fixture copy), `world_builder-appearance/`, `appearance-fixture/` (script + world root), `pindiag/` (websocket subscribe client and frame dumps), `pytest-world/`, `pytest-capture/`, `RECORDED-SHA.txt`.
- The session scratchpad (task outputs).
- Nothing was deleted; nothing was created at `~` or `/`; the 2026-09-14 gate's venv and fixture were read, not modified; the Tower on 8010 was stopped at the end.

---

## 8. What the next step needs

1. **Install the Debug build** from this SHA (or later) on the phone — §8.1 of
   the 2026-09-14 gate still applies: a Release build streams nothing.
2. Run the checklist's A3–A11 against the Windows Tower serving the canonical
   world `b2a75ab4…/a8c6817e…` — the pre-epoch caveats in A3/A6/A7 are
   correct and matter.
3. Before A7 on any world the current pipeline built, read §5.2's first MEDIUM:
   on an epoch-bearing world the relabel → restore round trip will reload the
   page and reset the camera, and that is the code, not the phone.
4. Tower lane: §6.1 (scikit-image), §5.2's `withdrawn`-for-a-crash, and the
   nine-test numeric baseline.
5. iOS lane: §3.5's helper, §5.1's two MEDIUMs.

---

## 9. Verdict

**PASS — at `5da13c838192b8491614f053e2fca388dc88d809`, for everything the Mac
can decide.**

The campaign's one stated blocker — "no Swift in this campaign has ever been
compiled" — is cleared: it compiles clean in Debug, Release and the test
target, its 88 new tests pass at the predicted counts and keep passing, the
full unit suite is green, the contracts agree, and the new transport and the
appearance page work together inside `WKWebView` against a real Tower over a
real socket. No product code was modified. No reviewer found a HIGH.

Read the verdict with three things beside it: the appearance world this Mac
served was fabricated, so nothing here judges the reconstruction; the
`Back to live` step of the smoke test on that world failed on the test driver
(§3.5) and should be fixed in the helper before the next gate leans on it; and
the two MEDIUMs in §5 describe behaviour a phone will show on the day
(a page reload after a privacy relabel on any epoch-bearing world; a copy
dropped by a transient poll error) — neither blocks a physical retest, both
should be on the list before a world is shared.

**READY FOR PHYSICAL RETEST**, in the sense the previous gates used those
words: what remains needs the phone, the glasses and the Windows Tower.
