# World Builder live-world visualization: Mac/iOS validation

The Mac's answer to `WORLD-BUILDER-LIVE-WORLD-VISUALIZATION.md`: does the iOS
app show the **photographic** world when one exists, report the lifecycle
honestly, and do the same from Saved Worlds — against the branch the Windows
Tower is actually running.

| | |
|---|---|
| **Validated** | `world-builder/live-world-visualization-v1` at `83534e2` (as published), then the fix commit `0885cfa` on top of it; this document is the commit after it |
| **Checkout** | the canonical checkout was **not** switched (the auto-mode classifier refused `git switch` there; the brief forbade a worktree). `83534e2` was exported with `git archive` to `~/Projects/Glasses-scratch/mac-ios-validation-2026-09-22/src-83534e2`; builds and tests ran from exports; the fix was committed with plumbing (`read-tree`/`commit-tree`) so the shared checkout, its index and HEAD were never touched |
| **Xcode / macOS / Simulator** | Xcode 26.6 (17F113); macOS 26.5.2; iPhone 17 Pro, iOS 26.5 |
| **Tower** | the live Windows Tower `100.110.156.55:8000` (the app's default authority), **read-only**: GETs and `result_subscribe` only, and the app was driven through a local proxy that forwards GETs and read-only socket messages and refuses every write (`tower-observe/readonly_proxy.py`) — the app POSTs `session/start` when the World Builder workspace appears |
| **Verdict** | §9 |

## 1. What the live Tower holds (read-only census, 13:26–14:40)

- Provenance: the Tower runs branch code (c2fc4fa or later). Nothing on the
  wire separates the branch from `main` in today's states; the evidence is
  behavioural — a live pid (28572) holds the writer lock of world
  `2f44716237544569b5f2faf782d9f877` / session `cb308801…` (the interrupted
  02:37 walk), `/health` shows no capture worker and both cartridge sessions
  stopped, and only `world_finish_pending.py` takes a world lock and runs the
  photographic stages. That is the finisher.
- 166 worlds, 70 sessions, 39 with geometry. **Every one serves the sparse
  rung**: revision `representation: sparse` 39/39, `appearance.state: absent`
  39/39, and a pinned `representation=surface|dense|appearance` is 404 for
  117/117. **No photographic world exists on this Tower.**
- Case A (historical sparse only): `6839fb8f…/060d7223…` (690 kf),
  `b2a75ab4…/a8c6817e…`. Case B (recovering): `2f447162…/cb308801…`.
  Case C (photographic): none on the Tower — proven on a Mac Tower (§3).
  Case D (photographic build failed): none on disk; the nearest is
  `52ed8e0a…` (a failed sparse finalization), reported `interrupted`
  consistently.
- Status channel, `/worlds` row and `/render/revision` agree for every case
  examined; no contradiction. `contract-drift-check.py` against the live
  Tower: AGREEMENT. `cross-stack-constants-check.py`: agreement.
- **The finisher has not finished.** World `2f447162` read `finalizing`,
  `build_in_progress: true`, `live: true`, sparse, from 13:19 until the end
  of this run (**>95 minutes**), for stages measured at ~8 minutes on this
  walk. See §7, T1.

## 2. Builds and tests

| Gate | 83534e2 | with the fix |
|---|---|---|
| Debug `build-for-testing` (clean derived data; the fix column is built from `git archive 0885cfa`) | SUCCEEDED, 0 errors | SUCCEEDED, 0 errors |
| Release `clean build` | SUCCEEDED, 0 errors | SUCCEEDED, 0 errors |
| Warnings | the recorded baseline: 7 app + the `<unknown>` echo; test-target `subscribeCount` and `WorldScaleSemantics` sets | byte-identical to 83534e2 |
| `GlassesTests` | **1036 / 0 failures**, 106 classes | **1044 / 0 failures** (+8 new), no runner restarts |
| `swift-structure-check.py`, `cross-stack-constants-check.py` | — | clean / agreement |
| New tests on 83534e2 production code | the two follower tests **fail** at the final swap assertion | pass; three repeats 48/0; each fails when its own fix line is removed (mutation check) |
| Targeted classes (WorldRenderRevision/Viewer/AssetTransport/Representation, ladder, stages, listing, picker note) | — | 122 / 0 |
| `TowerSmokeUITests` vs a Mac Tower on the fixture root | — | 2 pass; `testOpeningASavedWorldShowsThe3DWorld` fails at `:339` ("Back to live"), **identically on 83534e2** — the helper hit-test the 2026-09-21 gate recorded, every 3D-world assertion before it passes |

## 3. What the app actually showed (Simulator, real app, screenshots in the xcresults)

Through the read-only proxy to the **live Windows Tower**:

- **Live screen, unpinned**: the Tower selects `2f447162` (`selection.mode:
  finalizing`). Stage word **Improving**, spinner "The Tower is finishing this
  world.", note "…it is worth waiting for Saved." — Case B honest.
  "Open the 3D world" requests `/render?session_id=cb308801…&viewer=appearance-1`
  and draws the **sparse** rung (the only one that exists), captioned "Points
  the Tower measured… Not a surface". The follower polled
  `/render/revision?…&viewer=appearance-1` every 10 s (`live: true`) for the
  whole run.
- **Saved Worlds**: the row reads **Finishing** (world badge `live`); tapping
  it opens the same viewer, same URL, sparse. **On 83534e2 the viewer said
  nothing about the build** (defect I2; `ui-saved-2f44-1.xcresult`,
  attachment `saved-viewer-periodic-t66`); with
  the fix it shows the Improving note above the caption.

Through a **Mac Tower running 83534e2's own Tower code** (`PYTHONPATH` to the
export; autobuild and the finisher off) serving the 2026-09-21 gate's
fabricated appearance world `w1/s1` (synthetic imagery — it proves the
renderer path, **not** reconstruction quality) and the sparse fixture:

- `render/revision?…&viewer=appearance-1` → `representation: appearance`,
  `state: served`; the page carries `wb-representation: appearance` and the
  branch's `HOLD_MAX_MS = 45 * 60 * 1000`.
- Saved Worlds → `w1` → **the appearance renderer**: "Captured images on
  reconstructed geometry", compass cue, Best view / Face the room, "7 / 24",
  keyframe imagery drawn; native caption "The camera's own images, faces
  redacted…". The Tower log shows manifest, proxy and chunks fetched only
  through the `glasses-world:` handler. Reopened from the list: appearance
  again. **Terminated and relaunched, reopened from Saved Worlds: appearance
  again.**
- The sparse-only fixture opens as sparse, captioned as points — nothing
  fabricated.

## 4. Defects found in the iOS app, and fixed (`0885cfa`)

| # | Severity | Defect | Fix |
|---|---|---|---|
| I1 | **HIGH for the "open it while it builds" promise** | A refused appearance page could strand the **bare surface** on screen over the finished photographic world, with no control saying so. The appearance page revision `<s>/appearance:1@<epoch>` survives the ordinary Stop, so one walk-time appearance page that failed to draw (watchdog, `didFail`, three WebContent kills) refused the finished world's page too; the Stop gap's surface then drew and removed the "Try again" button; the photographic world under the same revision was never tried. Also: `load()` did not clear `refusedRevisions`, contrary to `WORLD-BUILDER-IOS.md` §10. | A revision refused while its build was live gets one try when the Tower reports the same revision finished (`refusedWhileLive`, the per-revision form of `finishedBuildRetried`; the liveness travels in `fallback`, not a shared flag); `load()` clears refusals. Bounded: a revision refused when already finished is not retried, and the retry is spent once. `WORLD-BUILDER-IOS.md` §10 states the rule. Two new tests, both failing on 83534e2 |
| I2 | MEDIUM | Saved Worlds: a **Finishing** row opened onto sparse points with no word that the photographic world was still being made, while the workspace's route into the same viewer said so (the Windows lane's review round 2 found it; it needed a Mac). Reproduced on `2f447162`. | `WorldPickerView.note(forOpened:…)`: a `finalizing`/`receiving` row gets the ladder's note — live from the pin once its report names this world, so it leaves when the build lands. Settled rows unchanged. Six tests |
| I3 | LOW (copy, consequential) | The Improving note said "It usually takes a few minutes"; the stage now spans the photographic stages (459–756 s measured, twenty planned). | "After a long walk, finishing can take twenty minutes or more" |
| I4 | LOW (copy, consequential) | The Storage row read **Saved** directly under "Improving … worth waiting for Saved" for the whole build. | "Stored on the Tower" |

Each was reviewed by an independent agent that did not write it, twice.
The first review found that the first version of I1 (only `load()`
clearing) did not cover the no-tap path. A "forget a refusal when the Tower
reports another revision" version then broke
`testARefreshThePhoneCannotDrawPutsTheOldPictureBack` (§4a rule 4: the route
and the page may disagree), and was replaced by the finished-revision
retry. The second review called that sound, and its three requests were
taken: the Try-again test reports `live: true` so it pins `load()` alone,
the liveness moved into `fallback` (a shared flag could be raced by a tap on
"Show it"), and §10 of the contract states the rule.

## 5. Adversarial questions, answered

1. **Photographic on the Tower, SfM on the phone?** Not by selection: every
   rung-deciding request carries `viewer=appearance-1` (page, native poll,
   the page's proxied poll), iOS never names a representation, and the
   Tower's `auto` ladder serves appearance first. The one iOS path that
   could keep a lower rung over it was I1 (fixed).
2. **Saved Worlds worse than the live viewer?** No: the same
   `WorldRenderScene`, model, URL and follower. It differed in honesty (I2,
   fixed) and a world row opens the newest session with geometry, not the
   best-represented one (noted, §8).
3. **Cached old sparse viewer?** No: ephemeral URLSession with no cache,
   `no-store` both ways, non-persistent WebKit store, one model per pushed
   screen, the handler's memory keyed by digest and dropped on close.
4. **Saved while photographic is incomplete?** Yes, Tower-side, briefly: T2.
5. **Appearance on disk but not advertised?** Only by design, when the
   serving gate refuses it (a redaction relabel, raw imagery on a normal
   Tower) — and then the revision says `withdrawn`/`absent` honestly. Not
   observed on the live Tower.
6. **Photographic requested, sparse returned silently?** The ladder falls to
   the best rung that exists; the page states its rung and iOS captions it.
   Observed: sparse served and captioned sparse.
7. **Photographic initialises, then downgrades?** The app never swaps a worse
   rung in; a page that fails before `didFinish` is reverted to the page that
   drew (with a "could not be drawn — Try again" control). The page itself
   has no geometry fallback. See T5 for a page that dies on a boot-time 404.
8. **Relaunch changes selection?** No; nothing is persisted. Proven by relaunch.
9. **Interrupted/recovered world, stale metadata?** Consistent across the
   three surfaces on `2f447162` for 95 minutes.
10. **New lifecycle fields break decoding?** No: status and listing decode
    with `JSONSerialization` dictionaries, unknown keys ignored, unknown
    listing/selection words survive. An unknown `model_state` is refused on
    purpose (contract §2.1).
11. **Older sparse-only world breaks?** No — opens and is captioned sparse.
12. **Mesh/triangle renderer as the final view?** Only when the surface is the
    best rung that exists (appearance failed or absent), captioned as
    surfaces — and, before I1, when stranded (fixed). No native renderer:
    the point galleries are behind Diagnostics.
13. **Does the app expose what the Tower built for the user?** Yes for the
    render ladder. It cannot show *why* a photographic build is missing:
    `stages` reaches no payload (T3).

## 6. Camera on upgrade

Confirmed from source and the page: every app-driven swap loads a new
document at the opening pose; the page restores a saved pose only for the
identical page. Expect the view to jump at each rung upgrade.

## 7. For the Tower lane (not changed from the Mac)

- **T1 — the finisher on the live Tower has not finished `2f447162` in >95
  minutes** (measured ~8 min on Windows). While it holds the lock the world
  reads `finalizing` / Improving, and the lock arm has no staleness ceiling,
  so a hung finisher is "Improving" indefinitely. Check the Tower log and
  `surface/cb308801…/status.json`. `/health` shows nothing of the chore.
- **T2 — "Saved" flickers mid-build.** Between `release_world()` and the
  surface's first `running` status, and between the surface's manifest/`ok`
  and the appearance's first `running` status (label policy, keyframe
  hashing), `_lifecycle`, the listing and `live` all read ready/complete —
  and the phone says **Saved** for those polls, then Improving again. The
  `stages.appearance = running` record is on disk for most of the second gap
  and nothing reads it. A wearer told to wait for "Saved" can see it at the
  surface/appearance boundary, over the surface rung.
- **T3 — a failed or owed photographic build reads "Saved"/"Complete".**
  `stages` is served nowhere, so the phone cannot tell "photographic failed"
  from "photographic done". The viewer's caption stays honest about the rung;
  the stage word does not.
- **T4 — a probe failure fails into the old bug**: `_still_building` keeps
  `ready` and only nulls `build_in_progress`, which iOS does not read for a
  finalized world.
- **T5 — `appearance_viewer.html`**: a manifest 404 at boot (the `rebuilding`
  window, or a transient) goes to `fail()` and `follow()` never starts; the
  app does not rescue a `rebuilding` page. Recovery is Reload.
- **T6 — per-world `live`**: every stopped session of a world whose lock is
  held is listed `finalizing`. Docs still describe `finalizing` as "a live
  builder" and `build_in_progress` as lock-only (`WORLD-BUILDER-WORLDS.md`
  row states, `WORLD-BUILDER-IOS.md` §2.4/§3).

## 8. Still open on iOS (recorded, not fixed)

- A world's own row opens its newest session with geometry, not its
  best-represented one; the listing's per-session `appearance` block is not
  read.
- After a withdrawal and a rebuild in a new epoch, both the page and the app
  may act (a reload with a camera reset, or an offer) — the 2026-09-21 gate's
  MEDIUM, unchanged.
- The follower polls only in `.ready`: a viewer whose first fetch failed
  waits for "Try again".
- `TowerSmokeUITests` `:339` helper (§2).

## 9. Verdict

**READY FOR PHYSICAL VALIDATION**, with the fixes in `0885cfa` on the phone
(a DEBUG build from that commit), and with the Tower items read first — T1
above all: confirm the Windows Tower's finisher is not wedged before walking,
because the next walk's world will be reported by the same machinery.

What the Mac could not prove: that a **real** photographic room draws on a
**phone** (none exists on the Tower; the Simulator drew a synthetic one
through the same path), frame time, WebContent memory with ASTC on Apple
GPU, touch feel, and the whole capture → Stop → Improving → Saved sequence
on glasses. The acceptance test remains the physical one.

## 10. Temporary resources

All under `~/Projects/Glasses-scratch/mac-ios-validation-2026-09-22/`
(~3.5 GB): `src-83534e2/` (the export), `harness-83534e2/` (export + a
validation-only UI test, never committed), `fix-83534e2/` (the fixed tree),
`review-copy/` (a reviewer's overlay), `dd*/` (derived data), `*.xcresult`,
`logs/`, `shots/`, `tower-observe/` (read-only observer, proxy, logs),
`tower-api/` (census captures), `fixture-root/` + `fixture-capture/` (a copy
of the 2026-09-21 fixtures served by the Mac Tower on 127.0.0.1:8011),
`commit-msg-fix.txt`, `fix*.diff`. The Simulator is `iPhone 17 Pro`
`9F377BE7…`. The Mac Tower (8011), the proxy (8010) and the observer were
stopped at the end. Nothing was deleted; nothing was created at `~` or `/`;
the canonical checkout's working tree, index and HEAD were not touched
(`git fetch --prune` did drop five remote-tracking refs whose branches no
longer exist on origin).
