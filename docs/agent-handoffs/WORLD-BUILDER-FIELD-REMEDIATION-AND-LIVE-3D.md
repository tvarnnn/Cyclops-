# World Builder: field remediation and live 3D

A campaign that began with one failed physical walk and a `BadZipFile`, and
found that the reconstruction had largely worked and the product had thrown
it away three separate times.

| | |
|---|---|
| **Starting SHA** | `223a506` |
| **Branch** | `integration/all-cartridges-v1` (canonical checkout) |
| **Field artifact** | world `52ed8e0abce54ca186fc49fd8631f048`, session `1e0d7ab80889478a977ce471d4d0a973` |
| **Preserved copy** | `C:\Users\tvllo\Projects\Glasses-scratch\wb-field-forensics\` (115 MB, 1,542 files) |
| **Verdict** | see the last section |

---

## 1. What the field test actually produced

The 2026-09-09 walk was real: Ray-Ban Meta glasses, iPhone, Windows Tower.
It ran 246 seconds, received 2,759 frames and accepted 795 keyframes. It
ended `Interrupted`, showed `Nothing mapped yet`, and drew 87 disconnected
fragments.

None of that described the reconstruction. On disk, under the failure, was
a world that had solved. **Three independent defects stood between the
reconstruction and the wearer**, and only the first was the one anybody had
noticed.

---

## 2. The BadZipFile root cause

`solution.npz` is a numpy archive, and a numpy archive is a zip. A zip is
only a zip once its central directory is written **last**.

`write_solution` handed the FINAL path to `np.savez_compressed`, so the
archive was built up in place — `open(..., "wb")` truncating the published
path to zero at the start of a write measured in hundreds of milliseconds —
while `solution.json` beside it went through the atomic helper.

The reader is in a **different OS process**. `scripts/world_build_session.py`
rebuilds its derived tree every 4 accepted keyframes (`--rebuild-every 4`,
`config.py:494`), and `BackgroundSolver.maybe_launch` spawns
`scripts/world_solve.py` as a child that writes that same file. No lock in
either process could have closed that window; only an atomic publication
could.

`load_solution` promised in its own docstring that it "never raises: an
unreadable or half-written solution is absent", and caught
`(OSError, KeyError, ValueError, json.JSONDecodeError)`. Measured over 15
seconds of the real race:

| exception | count | caught by the old guard? |
|---|---|---|
| `EOFError` | 48,854 | no |
| `zipfile.BadZipFile` | 10,295 | no |
| clean `None` returns | **0** | — |

`EOFError` descends from `Exception`; `BadZipFile` from `Exception` alone.
Neither is an `OSError`. The guard was an allowlist of exception types for a
best-effort read of a foreign binary format, which is incomplete by
construction. One escaped into the builder's `BaseException` handler and
ended a session holding 795 keyframes and 26,634 points.

The exact timeline, from file mtimes and record timestamps:

```
20:24:31.228  the 9th background solve finishes solving
20:24:31.390  the reader catches the write mid-zip; session ends "error"
20:24:31.558  the writer finishes.  The file on disk is VALID from here on
20:24:32.245  the best-effort recovery build succeeds
20:24:32.539  finalization recorded: interrupted / BadZipFile
```

The archive on disk today opens fine. That is what a torn read followed by a
completed write looks like, and it is why the artifact appears healthy.

**Fixed** by `storage.write_bytes_atomic` (temp file, fsync, retried
replace) and by widening the read guard to `except Exception` — deliberately,
with the reasoning in the code: a solution is DERIVED, re-solving rebuilds
it, and it is never worth ending a capture over. `BaseException` still
propagates, so a stop still stops.

The same fix closes a worse variant the field run happened to dodge:
`BackgroundSolver.wait` **terminates** a solve child that outstays a stop,
and `TerminateProcess` mid-zip left a torn archive at the published path
*permanently*, poisoning every later read of that world.

---

## 3. The defect that was actually costing the world

`derived/manifest.json` and `derived/<sid>/placements.json` in the field
artifact **contradict each other, under the same `input_digest`**:

| | manifest (`global_solve`) | placements.json |
|---|---|---|
| segments registered | **72**, across 14 components | **2** |
| segment 0 | `"state": "registered", "component": 2` | `"state": "refused"` |
| segment 0's reason | — | *"triangulated no points at all"* — with `points: 431` |

Two producers, diverged. `merge()` writes a registered placement for every
segment the global solve placed; `scripts/world_registration.py` writes the
weaker pairwise Sim3 answer. `placements.json` is what iOS reads.

The registrar was meant to stand down when a global solve had already
answered, and it asked:

```python
if args.register and not (solve_report or {}).get("solved"):
```

`solve_report` is the **final** solve's report — `None` whenever a session
does not finalise normally. The session died in the observe loop, so it was
`None`, so the guard concluded no solution existed. `placements.json` mtime
is **20:24:48.3**, sixteen seconds after the last build wrote its placements
at 20:24:32.5. Nine good background solves were discarded on the strength of
a question about a tenth that never happened.

**That single substitution is why a walk that reconstructed most of a room
was drawn as 87 disconnected postage stamps.**

Fixed: `engine.build()` reports `diagnostics["placements_source"]`, and
`should_register()` asks that. A second line of defence in `register_session`
refuses to replace placements that are registered AND digest-current, so a
silent read failure cannot re-open the same hole.

---

## 4. The third defect: the final solve is the world

Re-running the final solve the crash prevented — same images, same camera,
nothing in the artifact edited:

| | shipped | recovered |
|---|---|---|
| components | 16 | **6** (4 after merge) |
| largest component | 156 kf / 5,478 pts | **652 kf / 19,382 pts / 81 segments** |
| segments registered | 2 / 122 | **88 / 122** |
| keyframes posed | 561 | **668 / 795** |
| walk coverage | 24.8% | **82%** |
| mean reprojection | — | 0.82 px |

A subagent had concluded from boundary analysis that the capture was
unrecoverable: at each of the 15 component boundaries, all 210 attempted
image pairs produced **zero** verified correspondences. That analysis was
sound and its conclusion was wrong, because it was run against a
**sequential-only** database, which by construction contains no pairs beyond
±20 frames. Loop detection finds the revisits that bridge them. Recorded
because the same trap is easy to fall into again: *a measurement of a graph
that was never built is not a measurement of the scene.*

### Loop detection belongs on the live path

Sequential matching reaches 20 keyframes either side, so a live solve can
only chain forwards — it cannot notice the wearer walking back into a room
it already mapped. Measured on the field capture, at the field run's own
solve horizons, in one workspace, the way a live session accumulates:

| horizon | components (seq → loop) | largest component's share of posed | loop total | seq total |
|---:|---|---:|---:|---:|
| 156 | 6 → **3** | 0.75 | 17.1 s | 9.1 s |
| 311 | 11 → **6** | 0.77 | 26.0 s | 16.1 s |
| 526 | 14 → **5** | 0.90 | 45.3 s | 32.7 s |
| 646 | 16 → **6** | 0.91 | 50.8 s | 42.6 s |
| 795 | — → **5** | **0.95** | 70.3 s | — |

With loop detection the live world **converges**; without it, it fragments.
Cost 1.2–1.9×, paid in matching, which is incremental.

**A measured caveat, from an independent audit:** at 84 keyframes, on a walk
sequential already solved into one component, loop detection gave 22% *fewer*
points and slightly worse reprojection. So it is not free on short walks. The
crossover was not established. See "Unresolved limitations".

---

## 5. Normal Stop was still recorded as an interruption

Making the soft stop RUN the final solve did not change what the stop was
CALLED. `end_reason` was `interrupted` for any stop request, and iOS posts
`session/stop` from `.onDisappear` — so the wearer tapping Stop and then
leaving the World Builder screen sends the capture's end and a soft stop
within milliseconds. `results/world_builder.py` tests `end_reason` **before**
it looks at `finalization`.

Measured by an audit on a real 12 fps capture, on identical geometry:

| wearer lingers after Stop | `end_reason` | phone shows |
|---|---|---|
| 0.0 s | `interrupted` | **Interrupted** |
| 1.0 s | `stop` | Saved |

The builder now asks whether the **capture** finished — `CaptureFollower.is_closed()`,
asked of the follower rather than the directory, because a reconnect
retargets it. A stop that arrives while frames are still being written is
still an interruption, and there is a test for each direction.

---

## 6. Artifact durability model

| artifact | class | notes |
|---|---|---|
| `sessions/<sid>/keyframes.jsonl`, `events.jsonl`, `session.json`, `images/` | **AUTHORITATIVE** | the capture. Everything else is rebuildable from these. |
| `world.json` | **AUTHORITATIVE** (identity, convention, measured scale) | a rebuild never clobbers a measured scale |
| `derived/**` | DERIVED | `engine.build()` rebuilds it; proven by an existing test that deletes it |
| `solve/**` (`database.db`, `sparse/`, `solution.*`, undistorted images) | DERIVED | 49 of 163 worlds have `derived/` and only 8 have `solve/` |
| `solve/<sid>/sources.json` | **misfiled** — see limitations | raw-frame provenance nothing can recreate, stored in the tree `prepare_images` rmtree's |

Writes are atomic through `storage.write_json_atomic` / `write_bytes_atomic`:
temp file → fsync → `os.replace` retried for 2 s against a reader (Windows
refuses a replace onto an open destination, and `FILE_SHARE_DELETE` does not
lift it — measured, 223/400 bare replaces fail under a spinning reader, 0/400
with the retry).

**Temp names are now unique** (`<name>.<pid>.<uuid8>.tmp`). The shared
`<name>.tmp` was measured by an adversarial reviewer to reintroduce the very
bug it fixed: two writers of one destination produced **656 torn reads** in
12 seconds, plus a writer killed by `FileNotFoundError` when its peer's
`finally` unlinked the temp underneath it. Nothing serialises writers of
`solution.npz` — a hand-run `world_solve.py` against a live world is two
writers, and that is how this artifact was recovered.

---

## 7. Recovery: `scripts/world_finalize.py`

`mark_finalization` had exactly one caller in the entire system — the
builder's own `finally`. A session whose builder died between the last
keyframe and the finished world could not be repaired by any supported
operation: re-running the builder opens a **new** session, and
`world_solve.py` writes a solution that nothing merges. The field artifact
sat there with 795 good keyframes and no operation able to act on it.

Run against a pristine copy of the failed world:

```
was   interrupted / "BadZipFile: File is not a zip file"
now   complete / final_solve: solved
      668 of 795 keyframes posed, 6 components, 20,085 points
      88 of 122 segments registered  (the shipped world had 2)
      largest component: 81 segments, 652 keyframes, 19,382 points
      placements_source: global_solve
      67 s, nothing in the artifact edited by hand
```

**These figures vary run to run.** GLOMAP is not bit-deterministic. A second
recovery of the same pristine copy, after every later fix in this campaign,
gave **86** of 122 registered, 664 of 795 posed, and a largest component of
81 segments / 650 keyframes / 19,417 points. Read every number in this
document as one sample of a distribution a couple of percent wide, not as a
constant — including the 88 above, which is also quoted in the commit
messages.

Idempotent (every step rewrites), refuses a world a live builder holds
(tested against a genuinely live other process — the same-process case
self-reclaims by design and would have passed for the wrong reason), and
reads nothing but the session's own journals and images.

---

## 8. Tracking and segmentation: the UI was wrong, the algorithm was not

The UI said **121 tracking restarts**. It was rendering `segments - 1`.

The journal says: 64 `tracking_lost` events and 64 `solve_chain_broken`
events, 128 segment-index increments, 122 populated segments (7 indices
never received a keyframe). Two independent causes open a segment
(`engine.py:314` on a tracking loss, `engine.py:400` on a solve-chain break),
and the engine is explicit that the second must not be read as the wearer
losing the world — `solve_chain_broken` deliberately does not move
`last_tracking`.

**121 = 64 tracking losses + 57 chain breaks that produced a populated
segment.** 47% of the number on screen had nothing to do with tracking.

`Tracking: Good` was truthful under its own definition — a one-bit latch on
the most recent event — and that definition cannot express "64 losses in 246
seconds". All 64 losses were followed by an accepted keyframe within a median
of 37 ms, so the payload read `good` for 96.9% of the walk.

The thresholds behind the fragmentation are documented, measured decisions
with recorded rejected alternatives (`keyframes.py:121-206`, `engine.py:116-169`);
the blur gate was affirmatively refuted as a cause. **Nothing was retuned.**
The Tower now emits counted `tracking_restarts` and `chain_breaks`; iOS shows
the counted value and says nothing about restarts against a Tower that does
not send them, rather than falling back to arithmetic.

---

## 9. Saved Worlds

**Tower.** The renderer coloured every point by **segment index** — a
debugger's palette — while `points.json` carried real photometric `rgb` for
18,954 of 26,634 points. It now draws the measured colour, composes the
largest connected component as one frame, and keeps a neutral (never
palette) tone for points with no measurement, counted in the caption. The
diagnostic view — segment colours, camera frustums, unregistered fragments —
is unchanged and reachable.

`?view=diagnostics` is honoured **server-side**. It was read from
`location.search`, and iOS loads the page with `loadHTMLString(_:baseURL: nil)`
— no URL to read, and its navigation policy cancels the page's own links —
so the affordance did nothing on the only device it was for. Found by one
agent reviewing another's.

Served through the real route, the recovered world:

```
19,382 points from 81 segments placed in one frame.
4 of those segments produced no points to draw.
Every point carries a measured colour.
3 other groups of segments could not be placed relative to this one.
34 segments could not be placed at all.
```

Colour coverage improves with the recovery, for a reason worth naming: the
shipped world's points were a mix of the global solve's (which carry `rgb`)
and the local chain's (which do not), at **18,954 of 26,634 = 71%**. The
recovered world's points come from one solve, at **19,710 of 19,963 = 99%**.
Both figures measured directly on the two `points.json` files.

**Visually inspected** in Chrome (screenshots in the session transcript). It
is one coherent frame with corridor-into-room topology — a dense structured
cluster and a linear trail — not noise. It is also unmistakably a sparse
point cloud; there are no surfaces, and the caption says so. See limitations.

**iOS.** Opening a saved world now leads with the 3D reconstruction. The
fragment gallery, segment counts, registered/refused figures, refusal prose,
coverage tokens and raw identifiers moved behind a Diagnostics disclosure and
all remain reachable.

`Nothing mapped yet` previously covered five unrelated situations — never
fetched, fetch suppressed by the session gate, manifest fetch failed, one
manifest row failed to decode, pose-convention mismatch — with one sentence
and no Release-visible logging. It is now produced in exactly one place,
reachable only when the Tower's own geometry route says it has none AND its
snapshot claims nothing. A 404 against a snapshot claiming 17,674 points
reads *"The Tower disagrees with itself"*.

---

## 10. Instrumentation added

- **`tx_seq` now ships from iOS.** `tower/metrics.py` has carried the
  receiving half since 2026-08-19 and says in its own docstring that a gap in
  `seq` "cannot be attributed to any single cause" without it. The sender
  never sent it. On the field walk roughly **half the captured frames never
  reached the Tower** — 474 of 953 source indices on one capture, 2,391 of
  4,801 on the next — and one resulting gap ran 410 source frames, **17
  seconds**, which split the reconstruction. Whether the phone declined to
  send them or the link lost them is not recoverable from the artifacts.
  Proven against the Tower's own decoder: three scenarios with identical
  `seq_gap_total` of 116 separate into `tx_seq_gap_total` of `None`, `1`, `0`.
- **`world_builder_env_check.py`** now reports the COLMAP vocabulary tree and
  which `tower` package the install points at.

---

## 11. Windows-specific validation

Measured on this host (Windows 11 26200, NTFS, Python 3.12):

| behaviour | result | exposed? |
|---|---|---|
| `os.replace` onto a destination a reader holds | fails, WinError 5; `FILE_SHARE_DELETE` does not lift it | handled by the retry |
| reader vs concurrent `.jsonl` append | 0 partial lines in 47,732 + 76,089 tail samples | not exposed |
| a torn line + `append_jsonl`'s newline heal | one crash damages one record, not two | correct |
| directory rename with an open file inside | fails | no code renames directories |
| `shutil.rmtree(ignore_errors=True)` with an open file | **fails silently, leaves the tree** | exposed — see limitations |
| `database.db` | WAL; a hard-killed writer replays cleanly, `integrity_check=ok` | safe |
| a killed writer's temp file | survives `TerminateProcess`; only `purge_world` sweeps | see limitations |

Cross-cartridge soak (`scripts/cartridge_switch_soak.py --cycles 10`, World
Builder ↔ CV Lab): threads 41→41, RSS 651.0→652.3 MB, handles 612→612,
verdict **flat**, 0 dead locks, 10 of 10 worlds `complete`.

---

## 11b. Cross-cartridge: what this campaign did to everything else

The campaign modified SHARED infrastructure — `capture.py`, `metrics.py`,
`routes/ws.py`, `storage.py`, and the iOS sender every cartridge uses — so a
reviewer was asked the one question nobody else had: did any of it break CV
Lab, Object Memory, Document Memory, Scene Understanding or Tower
networking?

**Two regressions, both mine, both fixed.**

*The metrics fix went the wrong way.* Making a refused frame advance
`last_tx_seq` stopped a refusal being counted as transit loss — and
swallowed any real gap that sat immediately before one. Measured across five
scenarios: three frames genuinely lost then a refusal reported as **zero**,
and a session where every frame was refused reported `0` where the file's
own Rule 3 requires `None` ("we cannot tell" must not be reported as "no
loss occurred"). It lands on CV Lab, which refuses every frame while its
module is stopped, arming or paused. A refused frame now does the same
arithmetic an accepted one does — it is a frame this Tower SAW, so it closes
the interval without being missing itself. All five scenarios report the
truth:

| scenario | truth | first fix | now |
|---|---:|---:|---:|
| module active, 3 lost in transit | 3 | 3 | **3** |
| module paused, refuses 3 frames, nothing lost | 0 | 3 | **0** |
| 3 lost immediately before a refusal | 3 | **0** | **3** |
| every frame refused, 6 lost | 6 | **0** | **6** |
| alternating refuse/accept, 10 lost | 10 | 13 | **10** |

*The sweeper could delete a live writer's file.* It read "the last all-digit
component" as the pid, and `uuid4().hex[:8]` is all decimal digits **2.33%**
of the time — so a live writer's staging file was deleted at that rate. It
also deleted a user's `2024.tmp` and the capture recorder's own
`<source_seq>.jpg.tmp`, reading a frame number as a process. Staging names
now carry the pid as a `p`-prefixed component, which neither a hex uuid nor
a frame number can produce, and all seven cases are pinned.

Also: the `_replace_with_retry` → `replace_with_retry` rename left one caller
broken in `scripts/research/native_eval/` — outside the suite, so nothing
caught it.

**Proved clean:** `tx_seq` is additive on the wire (`REQUIRED_FIELDS` is a
missing-field check; no cartridge sees the raw dict); `is_closed()` is
provably identical for its two callers; object and document memory define
their own `TEMP_SUFFIX` and never touch `staging_path`; the lifecycle change
is sealed inside a private function with one call site.
`--ignore-glob="tests/*world*"` → 2,227 passed; the four other cartridges →
1,130 passed.

**And one blocker that was not this campaign's, which is now fixed.**
Document Memory could not start on this Windows Tower at all:
`ModuleNotFoundError: no module named 'bidi'`, from a **corrupted**
`python-bidi 0.6.11` install — recorded as installed, module absent. Not a
version conflict; a force-reinstall of the same version repaired it.
Pre-existing (the dependency file was last touched before this campaign's
baseline), and it had never shown up because the all-cartridge soak had only
ever been run on macOS. So the "five cartridges, one Tower" property was
**unproven on Windows** for the whole campaign and is now proven:

```
all_cartridge_switch_soak.py --cycles 8
  threads 82 -> 81   rss_mb 949.1 -> 943.0   handles 803 -> 801
  workers left: 0    strays: 0    locks: complete 8
  VERDICT: STABLE -- every cycle held, nothing leaked
```

**One intermittent, honestly:** of three all-cartridge soak runs (3, 8, 8
cycles), one 8-cycle run ended `the Tower did not stop when asked` against
the harness's 120 s budget — on a Tower whose own log showed a clean
`Application shutdown complete`. Measured in isolation, shutdown takes
**0.4 s** with or without Document Memory loaded, and the re-run of the same
8 cycles was STABLE with RSS *falling* 6.1 MB. Cause not established. It is
recorded rather than explained.

---

## 12. Approaches considered and rejected

- **Merging the dense-reconstruction lane** (`world-builder/dense-reconstruction-v1`,
  75 commits, +8,258 lines, unmerged). Its own verdict is *"PARTIALLY
  ACHIEVED"*: the room is recognisable, the enclosure does not close, 13% of
  sampled poses see through a dropped surface, and nothing was verified on
  iOS. Its measured cost is ~389 ms/keyframe with the fastest MIT-licensed
  backend against a 308 ms live budget, and it is gated on a single-component
  solve the field walk did not have until this campaign produced one.
  **Deferred, not refused** — the gate it needs now exists.
- **Retuning the tracking or blur thresholds.** The blur gate was measured
  and affirmatively refuted as a cause of fragmentation (loosening it made
  segments *worse*, 36 → 43 → 49). `loss_grace_frames` and
  `MAX_BARREN_SEGMENTS` are documented decisions with recorded measurements
  either side. Changing them without new evidence would have been guessing.
- **`try: ... except BadZipFile: pass`.** Named in the brief as unacceptable
  and it would have been: it would have left the non-atomic write, the
  poisoned-world variant, the `EOFError` majority, and the registrar clobber
  entirely untouched — and the walk would still have shipped 87 fragments.

---

## 13. Independent reviewer findings, resolved

Five review passes ran against work their authors did not write.

| finding | severity | resolution |
|---|---|---|
| `write_bytes_atomic` reused the known-unsafe shared temp name — 656 torn reads measured under two writers | HIGH | unique `<name>.<pid>.<uuid>.tmp` |
| the placements guard asked "did merge run", not "did merge place anything" — an all-refused merge suppressed the registrar | HIGH | requires ≥1 registered placement |
| the broad `except` could route a MemoryError or schema drift into the destructive registrar fallback | HIGH | `register_session` refuses to replace registered+current placements |
| the guard's test re-typed the condition and asserted the copy — a tautology | MEDIUM | extracted `should_register()`; the test calls it |
| two docstrings claimed things measurement contradicted (the `finally` sweeping a killed writer's temp; the eager read closing a hazard — 23 ms held against a 2000 ms budget) | MEDIUM | corrected in place |
| **a normal Stop still read as Interrupted** — the campaign's own headline symptom, unfixed | HIGH | `end_reason` asks the capture |
| a missing vocabulary tree **aborts** the solve process (glog CHECK → exit 3, uncatchable), shipping every segment refused | HIGH | `global_solve` checks the cache and degrades |
| `WorldTrajectoryReport.init` never declared the two properties the decoder passed — **the branch had not compiled** | P0 | initialiser fixed |
| `?view=diagnostics` was inert on the only client that exists | P1 | honoured server-side |
| iOS: `.loading` flicker every 2 s; final-solve sentence asserted from silence; shell worlds pushed into a guaranteed 404; geometry publish guarded on a content-derived revision that two empty sessions share; `unfetched` wrong in both directions; two identifiers became unreachable | P1/P2 | all six fixed |
| `tower/routes/geometry.py` imported a cartridge | — | caught by the existing architecture test; adapter owns the default |

**Final adversarial round**, two reviewers over the whole campaign:

| finding | severity | resolution |
|---|---|---|
| `acquire_writer_lock` is check-then-act — **two writers admitted in 8 of 8 trials**, and two concurrent `world_finalize.py` runs both succeeded on one world | BLOCKING | `O_CREAT \| O_EXCL`; 8 real two-process trials that fail 8/8 against the old code |
| `CaptureLimits.max_seconds = 900` — a 30-minute walk records 15, and since the `end_reason` fix that truncation read as a **successful stop** | BLOCKING | 40 min / 2 GiB from measured frame sizes; `bounded_limit` distinguished from the wearer |
| a repaired session still read `Interrupted` — the classifier tested `end_reason` before `finalization`, so the recovery tool could not produce a recovered world | BLOCKING | a completed finalization with a solved final solve outranks the capture's end; `end_reason` kept truthful |
| `import tower` loaded a **stale worktree** with none of these fixes, no longer even a git repo. The campaign's own pre-flight flagged it and had not acted | BLOCKING | editable install repointed; pre-flight green |
| `world_finalize.py` **downgraded a healthy record to `interrupted`** on any failure, permanently | SERIOUS | the record only moves if the run had something to improve on |
| the repair tool never ran the registrar, so a repair of a solve-less session left **no placements at all** | SERIOUS | it runs it, behind the same guard the builder uses |
| `write_derived` cannot keep a fixed 4-keyframe cadence past ~2,700 keyframes (~14 min) — 27% → 69% → 148% → 288% of wall clock | BLOCKING for a 20–30 min walk | interval doubles as the world doubles; 26–35% throughout |
| unique staging names leak one stray **per hard kill, forever** — 21 files / 11.4 MB after six kills, and `purge_world` has no production caller | SERIOUS | `sweep_abandoned_staging`, keyed on the writer's pid so a live write is never touched |
| `prepare_images` feeds COLMAP **two calibrations** under one `camera.json`, and deletes `sources.json` three lines before reading it | SERIOUS | the skip is conditional on recalibration; provenance restored; **five tests where there were none** — writing them surfaced a further bare `os.replace` that raised WinError 5 |
| `tx_seq_gap_total` scored every Tower-side refusal as transit loss (10 sent, 0 lost, reported 3) — the instrument this campaign enabled | SERIOUS | a refused frame advances the counter |
| iOS: no `didFinish`/`didFail`/render timeout, so a failed render was a black rectangle; unbounded reload on content-process death; session rows pushing into a guaranteed 404; the final-solve sentence stale forever | BLOCKING / SERIOUS | all fixed |
| two assertions in `test_storage_replace_retry.py` had become **vacuous** — asserting the absence of a name that is no longer created | COSMETIC | assert on any staging file |

**Exonerated by measurement**, against the lead's own stated doubts: the render page is fast (5.3 ms median for 19,329 points — the Chrome timeouts were screenshot artifacts); loop-detection-always is a net win at every *live* horizon (the 22%-fewer-points result did not reproduce); `load_solution`'s eager read projects to ~49 MB peak at 6,000 keyframes; a kill mid-`write_derived` or mid-`append_jsonl` loses no authoritative data; the field artifact is **byte-identical** to its preserved copy (1,542 files, 116,452,684 bytes, 0 mismatches), verified twice.

---

## 14. Unresolved limitations

1. **The saved world is a point cloud, not a surface.** Coloured, coherent,
   spatially readable — and still dots. The brief asks for "not a cloud of
   SfM debugging dots"; what ships is a cloud of *measured, photometric,
   single-frame* dots. The honest gap is a dense representation, and the
   lane that would close it is written, measured and unmerged.
2. **Render page weight — I called this the likeliest disappointment of the
   retest on weak evidence, then measured it and was wrong.** Two 30-second
   CDP screenshot timeouts were artifacts of capturing a large canvas, not
   the page. Timing the page's own `draw()` in desktop Chrome: the field
   world's 19,329 points draw in **6.2 ms (161 fps)**.

   The measurement did find a real problem one step further out. Draw cost
   is superlinear, and the phone's budget was set at a number nobody had
   timed:

   | points | desktop draw | fps |
   |---:|---:|---:|
   | 19,329 | 6.2 ms | 161 |
   | 38,658 | 13.6 ms | 74 |
   | 77,316 | 35.3 ms | **28** |
   | 115,974 | 65.0 ms | 15 |

   `MOBILE_MAX_POINTS` was **80,000** — 28 fps on a fast desktop, and a phone
   is slower. That was not a margin, it was the cliff. Now **40,000**
   (13.6 ms / 74 fps here), which is twice the total the field walk produced
   and a 2.6x margin on the measured cliff. The desktop-to-phone factor is
   **UNVERIFIED**; the retest is the first chance to measure it, and this
   number should then be set from that rather than argued about.
3. **Loop detection may be a net negative on short walks** (measured: 22%
   fewer points at 84 keyframes). The crossover is unestablished.
4. **Final-solve wall time projects to 3–10 minutes at 20–30 minutes of
   walking.** Unbounded by design on the happy path. What the wearer sees
   during it is a truthful `Finalizing`, but it is a long wait.
5. **~50% frame loss is diagnosable now, not fixed.** `tx_seq` will attribute
   it on the next walk; nothing in this campaign changes it.
6. **Nothing warns the wearer, live, that they are producing unusable
   frames.** 38.5% of the field capture was motion-blurred and 27.2% was more
   than 30% black. The dense lane's own recommended next step was exactly
   this cue, and it is not built.

   **This is the highest-value next lane, and it is small — which is why it
   is worth saying exactly why it was not done here.** The data already
   exists: `WorldBuilderEngine` counts `frames_observed` and
   `rejected_by_reason` in memory on every frame (`engine.py:292`,
   `_note_rejected`), and both reach disk only at `stop_session`
   (`engine.py:471`). So during a walk the phone cannot be told the
   acceptance ratio, and `_progress_block` says so in its own comment:
   *"`frames_observed` has no source at all"*. Publishing it needs a
   periodic write of those two fields — naturally placed in `build()`,
   which already writes — plus a ratio and dominant-reason line in the
   progress block, plus one row on the phone.

   It was not done in this campaign because **a new periodic write on the
   live path is the exact failure family this campaign spent six review
   rounds on**: a writer publishing to a file a cross-process reader polls
   is what `BadZipFile` was. Adding one in the last hour, after every
   recent change needed two rounds of fixes, is the wrong risk at the wrong
   time. The next lane should do it, with the atomic-publication discipline
   the rest of the store already has.

   It matters because the field walk's real limiting factor was capture
   quality, and this is the only change that would let the wearer correct
   it *while walking* rather than discover it afterwards.
7. **`shutil.rmtree(ignore_errors=True)` fails silently on Windows** at
   `store.py:343` and `global_solve.py:375`; `prepare_images` then *skips*
   surviving stale images, which could feed COLMAP two calibrations. Known,
   not fixed.
8. **Unique temp names mean orphans no longer collide — and nothing prunes
   them.** Only `purge_world` sweeps.
9. **`sources.json`** holds unrecreatable raw-frame provenance inside the
   workspace `prepare_images` deletes, and is already 523/643 stale.
10. **The venv's editable install points at a different worktree**
    (`Glasses-worktrees/all-cartridges-field-test-v1`). `start_tower.ps1`
    `Set-Location`s to the tower root first, so the supported launcher is
    safe; a hand-run `uvicorn tower.main:app` from elsewhere is not. The
    pre-flight now reports it.
11. **Nothing iOS was compiled.** Windows host.

---

## 15. The physical retest

**On the Mac, first — and this is a real step, not a formality.** The branch
did not compile for several commits and nothing said so (see the verdict),
and the iOS changes since are large.

```
xcodebuild build          # expect this to be where a problem shows up
xcodebuild test           # GlassesTests
```

**Install the DEBUG build.** `TowerClient.sendFrame` is inside `#if DEBUG`,
deliberately — this is an app that streams camera frames — so a Release
build transmits nothing at all and the walk produces an empty world. That
guard was left alone; it is a build instruction, not a bug.

**On the Tower, run the pre-flight.** If `vocabulary_tree_cached` or
`tower_package_is_this_checkout` is `[NO ]`, fix that before walking: the
first means every solve runs without loop detection and the world comes out
in pieces, the second means you are testing a different checkout.

```
cd tower
.venv\Scripts\python.exe scripts\world_builder_env_check.py
.\scripts\start_tower.ps1
```

Then, in one walk:

1. **Start World Builder on the phone and walk normally for 3–5 minutes.**
   Cover a loop — leave a room and come back into it. That is what loop
   detection needs.
   → *Watch: does the fragment count stop climbing and start falling?*
2. **Stop, and stay on the screen for a few seconds.** Then leave it.
   → *Watch: does it say Finalizing, then Saved — not Interrupted?*
3. **Open the world from Saved Worlds.**
   → *Watch: does it open a coloured 3D world you can orbit, rather than a
   grid of grey tiles?*
4. **Open Diagnostics, then "Open the solver's 3D view".**
   → *Watch: segment colours and camera frustums — the old view, still there.*
5. **Leave World Builder and start CV Lab.**
   → *Watch: it starts without restarting the Tower.*

**If anything fails, capture before doing anything else:**

- `tower/data/world_builder/worlds/<world id>/` — the whole directory
- `tower/data/captures/<capture id>/` — especially `capture.json` and `frames.jsonl`
- `tower/tower.log`
- `solve/<session>/solve.log` — the only place a crashed background solve leaves a trace
- the phone's Console.app output, `category=WorldBuilder`

And note **which of the five steps** it was, because they fail for different
reasons.

If the world comes out in pieces, the first thing to check is
`tx_seq_gap_total` in the session metrics: it now says whether the frames
were lost in the air or never sent.

---

## 15b. Temporary resources this campaign created

Filesystem policy rule 9. All under `C:\Users\tvllo\Projects\Glasses-scratch\`;
nothing at the drive root, nothing in the home directory, no new worktree.
**None of these were deleted** — rule 14 requires explicit human approval and
rule 15 says move rather than delete. They are already in the approved
location, so they are recorded here instead.

| directory | MB | what it is | keep? |
|---|---:|---|---|
| `wb-field-forensics\` | 115 | **The preserved field artifact.** Byte-identical to the live one, verified twice (1,542 files, 116,452,684 bytes, 0 mismatches). | **KEEP — evidence** |
| `wb-recover2\` | 138 | The most recent recovery of a pristine copy, after every fix. The best available reconstruction of the 2026-09-09 walk. | KEEP until the retest |
| `wb-replay\` | 244 | The first recovery, plus the rendered pages (`recovered-render.html`, `product-render.html`, `dpr-render.html`) that were visually inspected. | disposable |
| `wb-recover\` | 138 | Superseded by `wb-recover2`. | disposable |
| `wb-horizon\` | 134 | The loop-detection horizon sweep's workspaces. | disposable |
| `wb-production-walk\` | 7 | The production-argv walk's world and capture. | disposable |

The live world root under `tower/data/world_builder/` was never modified:
every recovery ran against a copy.

Scripts live in the session scratchpad under `%TEMP%\claude\...` and go with
the session: `production_walk.py` (the production-argv walk),
`horizon_sweep.py` (the loop-detection sweep), `swiftcheck.py` (a brace
balancer for a host that cannot compile Swift), and `swiftinit.py` — an
argument-label checker that produced 43 false positives and was **abandoned
rather than polished**, because a checker that cannot gate is worse than
none.

---

## 16. The end-to-end proof that is not the field artifact

The field artifact proves recovery. It does not prove a *walk*, because it
was recorded before any of this existed. So the production path was run
directly: `world_build_session.py` with the argv `tower/tower/main.py`
builds verbatim — `--follow-capture --rebuild-every 4 --register --solve
--solve-every 50 --max-idle-polls 3600 --stop-on-stdin-close` — the real
`world_solve.py`, a real 12 fps feed, ended the way a wearer ends it: the
capture closes and the workspace goes away in the same gesture, with no
wait between them. That combination is production's, and no test in the
suite had ever run it: every `--solve` test substitutes a stub solver, and
nothing passes `--register` and `--solve` together.

```
fed 260 frames in 22.8 s (11.4 fps); builder exited 0 after 25.2 s
keyframes 29  points 835  segments 4  posed 23
global_solve components 2  replaced 2  refused 2
placements: registered 2, refused 2

  [OK ] end_reason is stop          [OK ] lifecycle not interrupted
  [OK ] finalization complete       [OK ] placements from global solve
  [OK ] final solve solved          [OK ] something registered
  [OK ] geometry published          [OK ] lock released
  [OK ] no stray temp files
```

---

## 17. Verdict

**READY FOR PHYSICAL RETEST**, with one gate that is not a formality.

Everything on the Tower side is verified on this machine: the root causes
are fixed with tests that fail against the old code, the failed field walk
is recovered by a supported command, a production-argv walk passes end to
end, the suites are green, and two independent review rounds have been
answered.

**The gate is the Mac build.** Nothing iOS was compiled, and that is not a
theoretical risk: a reviewer found that `WorldTrajectoryReport.init` had
been missing two parameters its own decoder was passing, so **the branch
did not compile for several commits and nothing said so**. The iOS changes
since are substantial — a new 772-line state model, a restructured Saved
Worlds flow, a rewritten render viewer. Treat `xcodebuild build` as a real
step that may fail, not a checkbox.

What would most likely disappoint on the day, in order:

1. **The Swift does not build**, for something like the defect above.
2. **The world is a point cloud, not a surface.** It is coherent, coloured
   and navigable, and it is still dots. The dense lane that would change
   that is written, measured, and unmerged.
3. **The walk fragments anyway**, because the capture is poor rather than
   the solver is. 38.5% of the field capture was motion-blurred and 27.2%
   was more than 30% black, and nothing warns the wearer in the moment.
   `tx_seq` will at least say whether frames were lost or never sent.

None of the three can be reduced further from Windows. The first needs a
Mac, the second needs a decision about scope rather than more evidence, and
the third needs the glasses.
