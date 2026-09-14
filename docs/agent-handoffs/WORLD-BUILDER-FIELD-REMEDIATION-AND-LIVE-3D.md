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
| **Final SHA** | `3c163c2` (7 commits, local only) |
| **Verdict** | see the last section |

---

## 0. The commits

| SHA | |
|---|---|
| `26d0abc` | fix(world-builder): the solution a reader caught mid-write, and the placements a guard threw away |
| `a8f87e8` | fix(world-builder): the solve that makes the world, and a way to run it again |
| `f0deaed` | fix(world-builder): the walk that finished and was called Interrupted anyway |
| `f970788` | fix(world-builder): what the final adversarial team found, including in the fixes |
| `998785b` | fix(tower): the shared-code regressions this campaign put in, and a cartridge that could not start |
| `77409fd` | fix(ios): the 52 tests that would have reported success without running |
| `3c163c2` | fix(world-builder): six defects the fixes-to-fixes introduced, and the test that hid one |
| `e60d753` | fix(world-builder): the fourth root cause, and one manifest per SESSION |
| `f6e3443` | fix(world-builder): count the geometry, not the manifest that describes it |
| `60ec81c` | fix(world-builder): the defects in the fixes for the defects in the fixes |
| `feb08a6` | fix(world-builder): the ghost walk, and three ways one cartridge stopped the Tower |
| `9c52b68` | fix(world-builder): one reconnect is one walk, through the Tower this time |

Final code SHA **`9c52b68`**, 12 commits, all local — nothing was pushed.
This document is carried by one further commit on top, which is why it is
not in the table above.

**Committed with `--no-verify`, deliberately, and it should be revisited
before this branch merges.** CLAUDE.md's lane policy says agents commit from
a linked worktree and never pass `--no-verify`; this campaign's brief said
to work directly in the canonical checkout and not to create a worktree,
which cannot both be satisfied. The user was asked and chose the override.
The hazard the guard exists for — another lane's uncommitted files riding
along — was checked before each commit: the tree held only three untracked
files of the user's own, and every path was staged explicitly, never
`git add -A`.

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
| largest component, as a share of all 795 keyframes | 19.6% | **82.0%** |
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
solve horizons, in one workspace, the way a live session accumulates —
**both arms swept the same way on the same idle host**:

| horizon | components | largest component's share of posed | wall clock |
|---:|---|---|---:|
| | seq → **loop** | seq → **loop** | seq → **loop** |
| 84 | 4 → **3** | 0.46 → **0.76** | 6.4 s → 10.2 s |
| 156 | 8 → **3** | 0.27 → **0.75** | 11.4 s → 17.1 s |
| 311 | 13 → **6** | 0.23 → **0.77** | 21.3 s → 26.0 s |
| 526 | 16 → **5** | 0.33 → **0.90** | 35.0 s → 45.3 s |
| 646 | 18 → **6** | 0.27 → **0.91** | 39.2 s → 50.8 s |
| 795 | 19 → **5** | 0.22 → **0.96** | 52.4 s → 70.3 s |

With loop detection the live world **converges**; without it, its component
count rises at every horizon measured and never comes back down. Cost
**1.2–1.6×**, paid in matching, which is incremental.

The 84 row is its own fresh workspace in both arms rather than the first
step of the accumulating sweep, so its wall clock carries a full feature
extraction and is not incremental like the rows below it. It is there
because limitation 3 was about 84 keyframes specifically.

**The first version of this table was not a fair comparison and a document
audit caught it.** The `seq` column came from the field run's own
`solve.log` — solves that ran while the machine was receiving a live 12 fps
capture — and the `loop` column from an idle offline sweep. The component
counts survived that (they are counts, not timings) but the cost ratio did
not. The table above is a re-run: same script, same protocol, same host,
`loop_detection=False` as the only difference. The component counts moved
(the field run's live solves fragmented *less* than the idle control, at
6/11/14/16 against 8/13/16/18) and the conclusion did not.

Sequential does produce slightly *more* points at the end — 21,696 against
19,774 at 795 — spread across 19 components instead of 5. A raw point count
is not the product; a world you can orbit is.

---

## 4b. The fourth defect: a reconnect pointed the ledger at the wrong capture

Found by the lead while checking a claim in this document rather than by
reviewing code. Limitation 9 in §14 said `sources.json` was "already
523/643 stale"; the number was too specific to leave unexplained, and
nothing about a solve makes a path *stale*.

`_follow_capture` built each frame's `source_path` as
`directory / frame.relpath`. `relpath` is relative to the capture the frame
**came from**; `directory` is the one the generator was **called with**. A
reconnect retargets the follower onto a successor capture — the whole point
of `test_capture_continuity.py`, and a comment fifteen lines above this one
says so about `is_closed()` — and this line kept reading the closure.

The 2026-09-09 walk reconnected once, 55 seconds in. Measured on the
preserved artifact:

| | |
|---|---|
| captures the walk actually used | `6a1b544c` (474 frames, indices 1–953, ended `disconnect`) then `dd885cca` (2,391 frames, indices 1309–6109, ended `stop`) |
| captures `sources.json` names | `6a1b544c`, for **all 643** entries |
| entries that resolve to a file | **120** (indices 1–902 — genuinely the first capture) |
| entries that resolve to nothing | **523** (indices 1312–4811 — all in the second) |

So for **81% of the walk** the raw frame could not be found, and
`_source_frame` fell back to the session's face-redacted copies. That is not a
neutral substitution: `_source_frame`'s own docstring records why the raw
frames are preferred — "the redactor blacks out large regions (phone
screens, hands) that carry exactly the texture a solver needs: E6 measured
307 -> 337 keyframes in the main model from using the raw frames"
(`global_solve.py:352-361`). The walk this campaign exists for was
reconstructed the worse way, for 81% of its length.

**It could have been worse than missing, and only luck made it not.** The
phone's source index ran 1–953 in the first capture and 1309–6109 in the
second, so no wrong path happened to exist. Nothing guarantees that: a
counter that restarted at 1 would have resolved the successor's frame 17 to
the predecessor's frame 17 — a real photograph, from another moment, fed to
COLMAP under the right name, with nothing anywhere to notice. Both cases
have a test, and both tests assert on the **bytes** rather than the path,
because a path check cannot tell those two apart.

Fixed: `source_path=follower.directory / frame.relpath`.

**The same hazard was already known and already solved one cartridge
over.** `object_memory/imagery.py` resolves a record's picture by searching
the record's own capture *and then the reconnect lineage after it*
(`_successors`, `frame_path`), and its comment records the measurement that
made the second half necessary. World Builder had the same problem and none
of that.

**What is still lineage-blind here is the fallback, not the ledger.** When
`sources.json` has no entry for a keyframe, `_source_frame` searches
`capture_dirs` — which `main()` fills with `args.follow_capture` alone, so
it too names only the first capture. With the ledger correct that path is
secondary (every observed frame records its source), but a recovery of a
reconnected walk should pass every capture in the lineage explicitly:
`world_finalize.py --capture-dir` takes more than one. Adopting Object
Memory's `_successors` walk here is the real fix and is not in this
campaign.

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

**A capture that ended in `disconnect` counts as finished, deliberately, and
that is a real asymmetry worth stating.** A self-imposed `bounded_limit` is
reclassified as `interrupted` — nobody asked for a walk to end at forty
minutes — but a dropped socket is not. The reason is measurement, not
principle: **6 of the 14 most recent captures on this rig ended
`disconnect`**, including the 2026-09-09 field walk's own first capture,
which dropped mid-walk and was followed by a reconnect. iOS posts
`session/stop` from `.onDisappear` at the same moment it tears the socket
down, so an ordinary end can plausibly present as a disconnect, and
treating one as an interruption would put the campaign's headline symptom
straight back. The cost of the choice is the other direction: a wearer
whose link dies mid-walk, who then leaves the screen, gets a world labelled
**Saved** rather than Interrupted. The geometry is real and openable either
way; only the label is generous. The artifacts do record which it was —
`data/captures/<id>/capture.json` carries the capture's own `end_reason` —
but the session names only the FIRST capture it followed, so after a
reconnect a diagnostician has to find the successor by timestamp. Recording
the chain in the session would close that; it is a new write on the live
path, which is the family §14.6 declines to open in the last hour of a
campaign.

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

**These figures vary run to run.** GLOMAP is not bit-deterministic. Three
recoveries of the same pristine copy, at three points in this campaign:

| run | registered | largest component |
|---|---:|---|
| first | 88 / 122 | 81 segments, 652 keyframes, 19,382 points |
| after the final-review fixes | 86 / 122 | 81 segments, 650 keyframes, 19,417 points |
| after everything | 85 / 122 | 80 segments, 646 keyframes, 19,531 points |

A spread of about 3%. Read every number in this document as one sample, not
a constant — including the 88 quoted above and in the commit messages. All
three runs left **zero** stray staging files.

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
  Proven against the Tower's own decoder, in
  `tests/test_frames_seq_split.py`: a sender that omits `tx_seq` leaves the
  transit gap `None` rather than `0` (`seq_gap_total` 28, honestly
  unattributable); a sender that sends it turns a source gap of **118** into
  a transit gap of **2**; and deliberate 1-in-30 source sampling —
  `seq_gap_total` **86** — reports a transit gap of **0**. An earlier draft
  of this line quoted three scenarios sharing a `seq_gap_total` of 116
  separating into `None`/`1`/`0`. No such test exists; the numbers above are
  the ones that do. Caught by a document audit, and it is the reason this
  document now cites the file for every measurement of this kind.
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


**The fixes-to-fixes round.** A reviewer was pointed at the least-reviewed
work in the campaign — the two most recent commits — on the theory that
every round so far had found something in the newest code. It found six,
all the lead's, and one is this campaign's own pattern in miniature.

| finding | severity | resolution |
|---|---|---|
| **The last commit silently un-fixed the one before it.** `prepare_images` spelled the pid bare; tightening the sweeper to require `p<pid>` stopped it sweeping the writer that leaks MOST — the per-frame undistort write, where terminated solve children die | BLOCKING | `staging_path` is the one producer; the binding test runs `prepare_images`, spies on what it writes, and hands that to the real sweeper |
| **The sweeper's test wrote a name by hand**, so it pinned the fix and not the code — its docstring spelled the name one way and its assertion another | BLOCKING (it is why the above survived) | the test asks the producer; verified to fail against the bare-pid name |
| **The `O_EXCL` lock rewrite created a way to brick a world permanently.** The old code wrote through `write_json_atomic`, never partial; the rewrite creates then writes, so a kill in that window leaves a zero-byte lock naming nobody — refused after eight attempts in 1.3 ms, forever, including for the recovery tool | BLOCKING | an unreadable lock is waited out, then reclaimed: a lock that names nobody protects nobody |
| **Two processes could still both take it** — the reclaim unlinked whatever was at the path, not the file it read. Driven to both-acquiring 5 of 5 with a stall injected (0 of 120 naturally) | BLOCKING | the winner reads its own record back before returning; the loop sleeps between attempts, so a loser is told *who* holds it rather than "contending" (which it reported 104 of 104 times) |
| **A bound was still labelled `stop`** — the warning claimed the truncation "is not reported as a clean finish" beside the clean-finish label. The path had no test at all | SERIOUS | a self-imposed bound records `interrupted`; nobody asks for a capture to end at forty minutes |
| **`camera.json` was committed before the images matched it**, so an interrupted recalibration made stale frames permanent — measured: the second call re-undistorted zero frames with three of four from the old calibration | SERIOUS | the camera is committed last, so the file means "the images beside me were made with these parameters" and an interrupted pass self-heals |
| **A preserved record described artifacts the tool had just destroyed** — `complete / solved` on a world whose derived tree was gone, reading `ready` to the phone | SERIOUS | the record stands only if its manifest does; the classifier's READY branch requires one |

**The second flake, named.**
`test_result_channel_hostile.py::test_the_channel_survives_the_world_vanishing_mid_subscription`
fails with `PermissionError: [WinError 32]` on
`world_path(world_id).unlink()` — Windows refusing to unlink `world.json`
while the subscription under test still holds a read handle. Never in 11
unloaded runs; once in 4 concurrent ones, and once in a 20-run loop here:
about **1 in 20**, which is the profile of a test that looks solid until
the suite is busy. Its own docstring rejected `rmtree(ignore_errors=True)`
for partially succeeding and called the `unlink` "deterministic". It is
not: `unlink` does not partially succeed, it raises, which is worse only in
that it looks like a product failure. The unlink is retried for five
seconds now; the file still goes, the subscription is still live when it
does, and every assertion is unchanged.

It also corrected the flake characterisation used in an earlier commit
message: `test_object_memory_lifecycle`'s teardown race fails ~29% in
isolation on an idle machine, so "load-sensitive" was the wrong word, and
the test and its implementation were both added on this branch — not, as
claimed, code the campaign never touched. It is still not a regression from
any of these commits. The claim was wrong; the conclusion was not.

**And there is a FAMILY of these, not one.** Across the full-suite runs of
rounds 16-18, three different teardown tests failed on different runs and
never twice the same:
`test_object_memory_lifecycle::test_the_session_stops_when_the_last_connection_closes`,
`test_document_live::TestTheIdleWindDown::test_a_session_whose_stream_closed_stops_itself`,
and two in `test_world_builder_autostart_e2e.py` (§14.21). Every one of
them waits on a background teardown with a bounded `_wait(...)` and every
one passes in isolation — the Document Memory one 5/5, the autostart file
5/5 and 4/4 under four concurrent copies. None of them references World
Builder code (Document Memory's only mention of it is a comment about a
write pattern).

**Corrected after round 20.** An earlier version of this paragraph said
none of them was reachable from anything this campaign changed. Round 19
edited the disconnect `finally` the Object Memory test exercises — the new
`await` is skipped when nothing is recording, which is that test's case,
and a reviewer measured its rate unchanged (3/10 alone at HEAD, 5/10 on
the parent) — so the family is *unaffected* rather than *unreachable*, and
the distinction is worth the sentence. The same reviewer added two members:
`test_capture_continuity::test_finding_a_successor_does_not_read_every_capture_on_the_disk`
(1/15 alone; an mtime tie between directories created microseconds apart;
`capture.py` untouched by round 19) and, from the lead's own runs,
`test_result_channel_bounds::test_a_connection_cannot_open_unbounded_subscriptions`
(failed once beside a concurrent foreground run, 3/3 alone and 2/2 in-file
after).

The honest reading is that this suite has a class of teardown assertions
whose timeout is generous on an idle machine and marginal on a loaded one,
and that a full-suite run will usually show one of them. **It is not a
World Builder regression and it is not evidence of cross-cartridge
poisoning** — but "the suite is green" is only true run-to-run, and the
next person to see a red teardown test should check this list before
chasing it.

**The document round.** With the code quiet, a reviewer was pointed at
this file instead, and asked the question no code review asks: *is every
claim in here true, and does it match the code as it stands now?* It found
eleven that were not, in a document that had been through seven rounds of
review — and the pattern is worth naming, because six of the eleven were
**claims that went stale when a later fix landed**, not claims that were
wrong when written.

| finding | what it was | resolution |
|---|---|---|
| §14.7 said a defect was "known, not fixed" — **it was fixed**, in this campaign, by the `recalibrated` flag — and both its line citations had moved | FALSE | rewritten to the residue that genuinely remains (`clear_derived` can orphan a file); citations corrected and every remaining `file:line` in this document re-verified against the tree |
| §14.10 said the editable install pointed at a stale worktree; §13 said, four paragraphs earlier, that it had been repointed | CONTRADICTION | verified live (`import tower` resolves into this checkout) and withdrawn |
| §14.3 said loop detection may cost 22% of the points on short walks; §13 said that did not reproduce | CONTRADICTION | swept both arms at 84 keyframes: −0.2%, not −22%. Withdrawn with the measurement |
| §4's horizon table compared an **idle offline sweep** against the field run's **live** solves and drew a cost ratio from it | UNSOUND | the sequential arm was re-run under the sweep's own protocol; the ratio is now 1.2–1.6× and the conclusion is unchanged |
| §4's walk-coverage row put 24.8% beside 82% on different denominators | MISLEADING | one denominator, stated: largest component over all 795 keyframes |
| §10 quoted three `tx_seq` scenarios "with identical `seq_gap_total` of 116" separating into `None`/`1`/`0` | NO SUCH TEST | replaced with the figures the test actually asserts (28/118/86 → `None`/`2`/`0`), and the file is now cited |
| §15b's resource inventory under-reported by ~320 MB — six directories listed, eleven created, plus five loose files | INCOMPLETE | `du`-measured, complete, 1,264 MB total |
| the `end_reason == "stop"` READY branch had the same hole the `interrupted` branch had just been fixed for, and a code comment claimed it was "noted in the handoff" | **CODE, and the audit caught the claim rather than the hole** | one `and has_manifest`, and a parametrised test verified to fail against the old branch |
| the lock read-back — the fix for two processes both acquiring — had no test | UNCOVERED | driven deterministically at the read-back seam; fails against the code it replaced |
| §15 gave no expected Finalizing duration, no step to check the phone can still reach the Tower, and no execution-policy note | GAPS | all three added; the address is compiled in and was verified to match this host |
| "a Release build transmits nothing at all" | called an OVERREACH, and the correction was **worse than the claim** | see below |

Two more came out of the same sweep, both found by the lead reading this
document against the tree rather than reading code:

- §14.8 said "nothing prunes them" about staging orphans, four paragraphs
  after §13 recorded the sweeper that prunes them.
- §14.9 said `sources.json` was "already 523/643 stale". Nothing about a
  solve makes a path *stale*, and chasing the number found **the fourth
  defect of this campaign** — a reconnect pointing the raw-frame ledger at
  the capture the follow started in, which cost the field walk its raw
  frames for 81% of its length. It has its own section (§4b), a fix, two
  tests that assert on bytes rather than paths, and a production-argv walk
  driven across a real retarget.

**And one of the eleven was corrected in the wrong direction**, which is
worth recording rather than quietly repairing. The document said "a Release
build transmits nothing at all"; the audit called it an overreach on the
grounds that pings and lifecycle markers sit outside the `#if DEBUG`, and
the correction was written. A later reviewer read the guard boundaries
instead of the function names and found that `sendStreamStart` (1539) and
`sendStreamStop` (1571) are both inside the block opened at
`TowerClient.swift:1315`, and `sendLifecycleMarker` (1694) inside the one at
1683. Only the CV Lab commands (1787+) and the ping (1946) are outside.

The consequence is worse than "no frames": without `stream_start` the Tower
never opens a capture at all (`routes/ws.py:808`), so no builder attaches
and **no world is created**. A Release walk does not produce an empty
world; it produces nothing to open. §15 says so now. Two rounds of review
touched that sentence and neither checked where the `#endif` was.

**That is the lesson of this round.** Seven review passes read the code and
none read the document; when one finally did, it found eleven false claims
— and one of those claims, taken seriously enough to chase, was a defect in
the product that no amount of further code review had turned up. A wrong
number in a handoff is not only a documentation problem. It is sometimes
the only surviving trace of a bug.

**The two-reviewer round, on the question the brief ends with.** One
reviewer was pointed at the newest code in the campaign; the other was
asked only *"if he puts the glasses on right now, what is the most likely
thing that still goes wrong?"* and told to answer with things it had run.
Between them they found seven. Five are code fixes here; one is a
correction to a correction this document had already made; one is the
flake that had gone two rounds without a name.

| finding | severity | resolution |
|---|---|---|
| **The `has_manifest` guard refused READY and then said something worse.** The fall-through landed on `stopped_unbuilt`, whose reason is *"no geometry has been built for this session yet"* over a session whose record says `complete / solved` — and which iOS renders as a **permanent "Finalizing"**, a stage `WorldPresentation` reads as still-changing with the explaining sentence suppressed. It also hard-codes `finalization: None`, deleting the only evidence the solve ever finished | BLOCKING | the branch returns `interrupted` — "Needs retry" on the phone: settled, not-changing, carrying its reason and its record. Four tests, each verified against a different mutation |
| **The same hole, unfixed, on the surface a person CHOOSES a walk from.** `world_builder_library.session_state` short-circuits past its own `has_geometry` check, so the picker badge read **"Complete"** over a world with nothing behind it. Its docstring says it "Mirrors `_lifecycle`" | BLOCKING | it mirrors it now |
| **A reconnect that outlasts the socket's death silently truncates the walk.** `_await_successor` polls for 90 s without ever asking `should_stop`, and `routes/ws.py` stops every cartridge session when the last connection goes. The stop lands mid-wait; the loop then finds the successor, binds to it, and returns having read **zero** of its frames — moving `end_reason()` onto an open capture on the way out. Measured across this Tower's own **52 reconnect chains**: three at 31–35 s, against a socket uvicorn notices as dead in 20–40 s | BLOCKING | the wait ends when the stop does, the follower stays bound to the capture it actually read, and `stopped_awaiting_successor()` records that a reconnect was in flight — the one fact that separates an abandoned walk from a finished one |
| **The rebuild cadence was modelled on the wrong operation.** It costed `write_derived` (0.342 s at 795 keyframes) while the loop calls `engine.build()`. Measured over 204 rebuilds of a replay of the real field capture: **1.006 s at 601–800 keyframes, 81% of wall clock**, where the model said 27%. And the knee arrived a doubling late — `(accepted // 750).bit_length() - 1` is zero below 1,500 — so nothing widened until eight minutes in | BLOCKING for a 20–30 min walk | knee 600, no `- 1`: 8 from 601, 16 from 1,200, 32 from 2,400, 64 from 4,800. The cadence tests are re-anchored on measured `engine.build()` cost, with a control that fails for the schedule they replaced |
| **The pre-flight was all-green and never checked that a calibration exists.** A reviewer pointed a replay at a fresh root and watched **131 rebuilds produce zero poses and zero points**, announced only as a WARNING in a log nobody reads. The seven verdicts checked that OpenCV *can* calibrate; none that anything *has* | SERIOUS | an eighth verdict, `calibration_for_the_camera`, which reads the launcher's own `.env` and compares the last capture's resolution against `intrinsics/`. Green here: `360x640 is calibrated` |
| **"A Release build transmits nothing at all" was corrected in the wrong direction** by the document audit — see the paragraph below | SERIOUS | reinstated, with the guard boundaries quoted |
| the second flake, uncharacterised for two rounds | COSMETIC | named, reproduced and fixed — see below |

**Two further findings are documented rather than fixed** — the wearer's
three-minute wait after Stop (§14.12) and the World Builder screen having
to stay foregrounded for the whole walk (§14.13). The first is the biggest
thing still standing between this branch and a retest that reads as a
success, and it is a procedure step rather than a defect.

**The round that reviewed the previous round's fixes, and found two of them
worse than what they replaced.** This is the pattern the campaign has
produced every single time, and it produced it again at the very end.

| finding | severity | resolution |
|---|---|---|
| **`has_manifest` is the WORLD's, and the fix used it to answer a question about a SESSION.** `derived/manifest.json` names whichever session built last, so the new guard called **every older session of a world walked twice** unopenable — `interrupted`, over a reason saying its geometry was "no longer on disk", with `poses.json` sitting right there. Measured on two synthetic sessions in one world, both trees present | **CRITICAL — worse than the hole it closed.** That one lied about a world with nothing in it; this lied about an intact one and invited the wearer to redo the walk | `has_session_geometry`, the same two files (`derived/<sid>/poses.json`, `points.json`) the listing already asks about. A test builds the two-session state on real disk, because flags are exactly what got it wrong — plus one that fails if the two modules' copies of the predicate ever disagree |
| **`stopped_awaiting_successor` fired on any stop inside the grace window**, which is also the ordinary shape of a phone that disconnects for good — the socket dies, `routes/ws.py` stops the session 20–40 s later, comfortably inside 90 s. So it silently reversed the policy two files away that a `disconnect` capture counts as finished, and a permanent disconnect started reporting `interrupted` | **HIGH — the headline symptom, back through the door it was pushed out of.** Reproduced through the CLI, not inferred | the flag is set only when a successor was actually found and then discarded, which is the only shape that means "a reconnect was in flight". No successor and a stop is an ordinary end |
| **The pre-flight's hand-rolled `.env` parser disagreed with python-dotenv in 6 of 12 cases** — `export KEY=v`, both quote styles, a UTF-8 BOM (`str.strip()` does not remove `\ufeff`), an inline `# comment`, a quoted value with a space. Every disagreement pointed the same way: a RED verdict saying "every pose and every point of the walk will be missing" against a correctly configured Tower | MEDIUM — a lie in exactly the situation the check exists for | `dotenv_values`, which is what uvicorn's `--env-file` uses. A test asserts six legal spellings of one root resolve to one directory |
| a superseded comment block still stated the old cadence flatly (*"The knee is 750"*, with the old schedule), and `--solve-every`'s help text became false | LOW | the old block says what supersedes it; the help text names the coupling |

**The same reviewer also caught the lead destroying a source file.** A
scratch tool, `swiftcheck.py`, takes its OUTPUT path as `argv[1]`. Invoked
the way a thing called "swiftcheck" looks like it should be — with a
`.swift` path — it wrote its 3,728-line symbol index over
`WorldPresentation.swift`. Nothing in the Python suite touches Swift, so
nothing failed; the review noticed it in `git diff --stat` and reported it
as another agent's doing rather than acting on it, which is what the lane
policy asks. The file was tracked, so `git checkout --` restored it and
the one-line note change was re-applied; the tool now refuses to write over
an existing file or anything ending `.swift`, `.py` or `.md`. It cost
nothing this time because the file was committed. That is luck, not
process.

**And three findings are recorded rather than fixed**, with what is known
about each:

- **The picker badge and the canvas headline disagree for a session with
  no geometry.** `WorldListingPresentation.swift:85-91` states as an
  invariant that they must not: *"a row that says 'Interrupted' opens onto
  a canvas headlined 'Interrupted'."* But `model_state: interrupted` with
  no geometry renders `.needsRetry` — headline **"Needs retry"** — under a
  row reading **"Interrupted"**, and the phone's own sentence *"Nothing
  usable came of this session"* prints directly above the Tower's *"the
  capture is still there and it can be rebuilt"*. This is **pre-existing**
  (it fires for any `interrupted` session with no manifest) and the
  `has_session_geometry` fix removes the common route to it, but it is
  real, it is iOS, and nothing on this host can compile a fix. The Swift
  test that claims to protect the invariant checks badge strings against
  literals and never touches the canvas side.
- **The background-solve cadence is coupled to the rebuild cadence.**
  `maybe_launch` is checked inside the rebuild block, so `--solve-every 50`
  is really "50, rounded up to the rebuild interval" — 50–52 today, 64 past
  4,800 keyframes. The coupling is pre-existing; the retune tightened it
  2.5×. It is benign in effect (a solve at that size takes minutes anyway,
  so 50 was never achievable there) and the help text now says so. The
  same block's `OSError` retry costs up to 64 keyframes instead of 4, and
  `World.updated_at` — the library's sort key — moves every ~20 s instead
  of ~1.2 s. Nothing watchdogs a rebuild gap, and the result channel's 2 s
  heartbeat keeps the phone's figures moving independently.
- **`LIFECYCLE_STOPPED_UNBUILT` still projects to `finalizing`**, whose
  iOS stage reads as still-changing. The permanent-"Finalizing" this
  campaign fixed on one route is still reachable on another: a session
  with no finalization record at all. Pre-existing, untouched, and it wants
  a contract change rather than a guard.

**The round that reviewed the round that reviewed the previous round.** Same
brief, same question, same result: the newest code had defects, including in
the fix written an hour earlier for the same class of bug.

| finding | severity | resolution |
|---|---|---|
| **The identical `has_manifest` bug, one branch further down, untouched.** `if not has_manifest: return stopped_unbuilt` was never part of the fix, so a session with its geometry on disk and **no finalization record** read "no geometry has been built for this session yet" and projected to the phone as a permanent `finalizing` — verbatim the outcome the branch above spends a paragraph forbidding, reached by a different door. And `finalization is None` is the common case for anything offline: `stop_session` writes the block only when `hold_lock=True` | HIGH | that branch asks `has_session_geometry` too, and a session whose tree exists while the world's manifest names another reads `ready` with the currency claim withheld rather than asserted |
| **`has_session_geometry` stats; the serving path parses.** Empty, truncated and wrong-shaped derived trees all read `ready` here and 404 there — and the test written to prove the previous fix wrote `"{}"`, which is one of them, so it proved the classifier changed and not that anything could be opened | HIGH | the gap is kept, deliberately (this runs on the 0.5 s poll and `points.json` is megabytes) and now **stated** — "was there a build" here, "can it be read" at the route — with a test that pins both halves. The docstring's "the files the phone actually opens" is gone |
| **A corrupt derived tree was an HTTP 500, not a 404.** `read_derived` caught `JSONDecodeError` and `KeyError` but not `TypeError`, so a top-level list came out through `build_manifest` as a server error. Its two neighbours in the same file carry explicit comments about being hardened for exactly that shape | MEDIUM | `TypeError`, `ValueError` and `OSError` join the guard. A corrupt world and an absent one are both "nothing to serve" |
| **The reconnect race survived the fix that named it**, in a window that shrank from 90 s to a few statements: `_await_successor` samples `should_stop`, returns, `follow` rebinds, and only then checks again. A stop in between bound the follower to a capture it read zero frames of — the original defect exactly | MEDIUM | asked once more before the rebind |
| **`_find_successor` opened every `capture.json` on the disk, every poll.** 11 ms against 104 captures × 360 polls, so the "ninety second" grace window ran **94 s** — an overrun that grows with a directory that only grows | MEDIUM | `os.scandir`, newest first, nothing older than the capture being followed, stop at the first match. **10.16 ms → 0.17 ms**, and the overrun from 4.4 s to 0.06 s. A test asserts the scan opens exactly one manifest |
| **python-dotenv was never a declared dependency** — it arrived as `uvicorn[standard]`'s extra — and the pre-flight swallowed its absence into a verdict reading *"there is no `.env` to read it from"* about a file it had just confirmed exists. The same lie the hand-rolled parser was replaced for, reintroduced through the error path | MEDIUM | declared in `pyproject.toml`, and the failure now names the exception and says the check cannot answer rather than answering wrongly |
| the predicate has **three** copies, not two, and the test pinned two — leaving the render page free to drift; `--solve-every`'s help hardcoded 64, which is only true at the default `--rebuild-every` | LOW | all three pinned; the help states the multiplier |

**And the lead caught one in its own fix before it shipped.** The new
`ready` branch for an older session sits beside a geometry block that reads
the same world manifest — and with no manifest for this session, that block
said *"no build has run for this session, so no geometry exists"*, over a
derived tree on disk. `ready` beside "no build has run" is the Tower
disagreeing with itself: the failure this campaign is named after, pointing
the other way. The block now distinguishes "no build ran" from "a build ran
and the world's manifest is about another session"; the counts still cannot
be summarised from there, and the route that serves the geometry reads this
session's own tree.

**One ambiguity is resolved rather than fixed, and the reviewer was right
that it is a coin.** When a stop lands *before* a successor's directory
appears, a walk cut short is indistinguishable on disk from a walk that
ended — same `disconnect` capture, same absent successor — and the overlap
is not small (reconnect chains at 31–35 s against a socket noticed dead at
20–40 s). Neither answer is derivable, so it resolves toward **finished**,
which shows a real world as Saved rather than showing a real world as
broken. `stopped_awaiting_successor()` now documents that it means "a
successor had appeared and I left it unread", not "the walk was cut short".
Closing it needs the phone to say which it was.

**The round that found the root, after four rounds of fixing its
symptoms.** A reviewer enumerated `_lifecycle`'s entire input space, drove
each reachable state through the real producer and the real engine, and
took every answer end to end into the Swift. Its first finding killed the
previous round's fix; its last one explained all four.

**A world has ONE `derived/manifest.json` and it names whichever session
built last.** The status producer correctly refuses to attribute it to any
other session — and then had no figures at all. So an older session of a
world walked twice reported no geometry, no poses, no currency, and the
phone rendered a red **"Needs retry"** over a reconstruction sitting on
disk. Four branches were written to paper over that in four consecutive
rounds, three of them wrong, before anyone asked why there was only one
copy.

`write_derived` now writes the manifest **beside the poses and points it
describes**. One small JSON next to megabytes of geometry, atomic like
everything else, additive — a world built before it reads exactly as it
did. Every session built from now on answers for itself, and currency,
counts and the phone's saved-versus-needs-retry decision work for an older
session exactly as they do for the newest.

| finding | severity | resolution |
|---|---|---|
| **The previous round's `ready` branch reported `ready` over geometry behind the journal** — and destroyed the one truthful answer the producer had. Measured: a SECOND, unrelated session building is what flips a session from *"a rebuild is outstanding"* to `ready`, with nothing about that session changing | CRITICAL | the per-session manifest above; the legacy branch that remains says what it does *and does not* know, and its reason is not `None` the way a verified `ready`'s is |
| **And the phone rendered that `ready` as "Needs retry".** Traced into the Swift: `.finalized` + null counts + no finalization record → `hasGeometry == false` → `.needsRetry`, a red warning triangle over 1,119 points and 3 solved poses | CRITICAL | the counts are real now, so the same session reads `.saved` |
| **`_trajectory_block` still said *"no build has run for this session"*** — the exact sentence `_geometry_block` had been fixed for, twelve lines away, in the same file, left behind by the same change | MEDIUM | both blocks, one test that prints both for one session |
| **The `_find_successor` mtime prune could lose a successor permanently.** `write_json_atomic` on `capture.json` bumps its parent directory, so a predecessor's `stop()` moves its own mtime FORWARD — past a successor created before that `finally` ran. Measured: found on **0 of 360 polls**, not "a missed successor on that poll" as the comment claimed. Both mtimes are frozen for the window, so a prune that misses once misses every time — and the follower then reports no reconnect, so the walk finalises as an ordinary end with the successor's frames silently dropped | HIGH | the prune carries `RESUME_GRACE_SECONDS` of slack, which is the same bound the recorder uses to decide whether to link a successor at all. Still 0.14 ms per scan against 104 real captures |
| **Widening `read_derived` to catch `OSError` turned a disk fault into "no geometry".** EIO injected: a 404 on a route whose own comment says *"404 now means ABSENT only"*, and an empty reconstruction in `world_inspect` | MEDIUM | `OSError` is out again — a server that cannot read its own storage should say so, and 500 is how. The log line names the exception and the session now, where it named only the world |
| **The dotenv early return regressed the case `python-dotenv` was declared for.** An operator who exports `TOWER_WORLD_ROOT` needs no `.env`, and the early return skipped the env-var lookup, so a correctly configured machine went RED anyway — measured against the behaviour it replaced, which was right about that case | MEDIUM | the reader failure is remembered, not returned on; it reaches the verdict only if the environment could not answer either |
| a stale state table, one no-op `finalization` beside two siblings spelled differently, three redundant local imports, `json.JSONDecodeError` listed beside its own base class | LOW | all four |

**And then the lead found the same defect on the path that actually serves
the geometry**, by asking what the phone does *after* the status channel
says `ready`: it fetches, and got nothing.

`read_derived` gates on `derived_is_current`, which read the **world's**
manifest — the one naming whichever session built last. So on a world
walked twice it judged the older session's keyframes against the *newer*
session's digest, found a mismatch, logged *"stale; treating as absent"*
and refused to serve. **"Open an earlier walk from Saved Worlds" was a 404
for a reconstruction sitting on disk**, and the geometry route and the
render page gated the same way.

Four review rounds had fixed this bug on the reporting path, one branch at
a time, and nobody had followed the wearer's next tap. `derived_is_current`
takes a session id now and judges a session by its own manifest; a world
built before those existed cannot be judged at all, so it is served rather
than refused, and the status channel says in words that its currency is
unknown. The test that pins it was **verified to fail** against the old
gate — and had to be strengthened first, because with no keyframes both
sessions digest identically and the bug is invisible.

**Junction aliasing is recorded, not fixed.** On Windows a junction under
the captures root reports `is_symlink() == False`, so `_find_successor`
cannot tell it from a real directory and may bind to the alias. Pre-existing
— the `iterdir()` version behaved identically — and it needs a policy about
what a junction under an artifact root even means.

**The round that reviewed the root fix.** The per-session manifest removed
the cause; this round found what the fix for it had missed, and one thing it
had newly broken. Every finding is a place where two readers of the same two
files disagreed.

| finding | severity | resolution |
|---|---|---|
| **The route asserted `current: true` over geometry nothing had judged** — and kept asserting it after a reviewer appended a keyframe that made the build genuinely stale. One boolean was answering two questions: *is this current* (the wire contract's flag, "reflects every keyframe accepted so far") and *may this be served at all* | HIGH | `derived_currency` returns **True, False, or None** — "nothing here can judge it". `read_derived` refuses only on `False`, so a legacy world is served; the route and the render page report `current: false` for `None`, because `current` is a claim |
| **One walk's coverage classes served as another's.** `usable_placements` and the `global_solve.segments` lookup still read the WORLD's manifest after `derived_is_current` had been given a session id — so an earlier walk was told a segment was `confident` when its own build called it `unresolved`, and **every placement was refused**, which means an earlier walk could not be composited at all | HIGH | both read the session's own manifest; a test drives `usable_placements` with a real placement and was verified to fail against the old read |
| **`_lifecycle` asked "is there a tree" before "did a build run"**, so a session whose manifest is present and whose derived tree is gone said *"no geometry has been built for this session yet"* — the permanent `finalizing` again, now rendered by this campaign's own iOS note as "worth waiting for Saved", forever. Reached through the ordinary case: `finalization is None` is what every offline caller leaves | MEDIUM | a manifest proves a build ran; that state is `interrupted` with a reason saying the geometry is gone |
| **The function extracted to make both manifest copies behave alike did not.** `read_session_manifest` guards its return type; `read_derived_manifest` does not, and `_validate_manifest` called `.get()` on whatever it was handed — so a top-level list gave a clean refusal for one file and an **AttributeError out of the status channel** for the other | MEDIUM | `isinstance`, and the same corruption now gives the same answer for both |
| **The file cache's bound was a cliff.** `MAX_ENTRIES` cleared the dictionary *entirely*; a reviewer measured the hit rate falling from **95.8% to 14.0%** the moment the count crossed it, and the per-session read moved that crossing two worlds closer — on a host holding 163 worlds, with a producer that is a process-lifetime singleton | MEDIUM | oldest-out eviction, cap raised to 512. An eviction costs one re-read; clearing cost every reader one |
| the session-manifest failure logged the world id, so a malformed `derived/<sid>/manifest.json` was indistinguishable from a malformed `derived/manifest.json` | LOW | the source is named |

**Two things are recorded rather than fixed.** `read_derived(verify=True)`
now serves a legacy tree it cannot verify — deliberately, because refusing
is a guaranteed 404 for a good reconstruction and the `current` flag beside
it is false; its docstring says so. And `_find_successor`'s prune bounds
only "older than me", so **replaying** an old capture still scans every
younger directory (~9 ms against 104, against 12.5 ms before). The live
case — following the newest capture — is 0.17 ms and 0 survivors.

**The round that made every reader agree, and ended the permanent
"Finalizing" at the mapping rather than at the fourth branch.** A reviewer
enumerated the eleven states in which the two manifest copies can disagree
and drove each through all four readers.

| finding | severity | resolution |
|---|---|---|
| **The four readers disagreed about which copy wins.** The status producer preferred the WORLD's manifest; `derived_currency` and `_session_manifest` preferred the SESSION's. Where the copies differ they picked different files — reproducing **both** of this campaign's named failures at once: `ready` beside a 404 in one direction, and a route serving geometry the phone was told was still finalizing in the other | HIGH | one rule everywhere — **the copy beside the geometry decides** — and a test that tears the two apart and checks all four agree |
| **A manifest was reported as geometry after its poses and points were gone.** `has_session_geometry` had been threaded into only the `manifest is None` arm of the geometry and trajectory blocks, so in the exact state the lifecycle branch beside them was written for, one payload said *"its geometry is no longer on disk"* and `available: true, current: true, element_count: 26,634` | HIGH | a manifest describes a build; the poses and points **are** the build |
| **The permanent "Finalizing" came back for the fourth time**, by a route the three previous branches missed: an older session whose tree was deleted takes its manifest with it, so nothing proves a build ran | HIGH | fixed at the **mapping**, where it always belonged. `stopped_unbuilt` projected to `finalizing` because a pre-finalization record could not show whether a build was running — true when written, and **made false by this campaign**: `stop_session(hold_lock=True)` means a live build is a live lock, which `_lifecycle` answers three branches earlier. Every state reaching `stopped_unbuilt` has already been shown to have no live holder. It projects to `interrupted` now, and a second test holds a real lock to prove the "wait" answer still works when something is actually working |
| **The reader was fixed and the writers were not.** Both places that STAMP a placement's `input_digest`, plus `world_finalize.py`'s "does this world still have geometry", still read the world's manifest — so `world_registration.py --write --session <older>` stamps another session's digest and the reader then refuses every placement it just wrote | MEDIUM | all three read the session's |
| **The two manifest readers still failed differently.** `read_session_manifest` caught `ValueError`; `read_derived_manifest` caught only `json.JSONDecodeError` — and `UnicodeDecodeError` is a sibling of the first, not the second. The same three bad bytes gave a clean refusal in one file and an exception out of every reader for the other, **including out of `read_derived`'s verify gate, which sits above its own `try`** | MEDIUM | `ValueError`, in both, plus a test that uses invalid UTF-8 rather than invalid JSON — which is why the hole survived, since `{not json at all` is caught by both |
| **The cache was first-in-first-out, not least-recently-used**, so the hottest entry — the live session's journal, re-read every poll — kept its original slot and was evicted first. And 512 entries held **14.8 MiB** at field scale, because a parsed manifest is 63.5 KB, not the "small summary" the comment claimed | MEDIUM | move-to-end on a hit, cap 256 (~7 MiB), with a test that re-reads one entry through a full flood |
| four comments describing code that had moved | LOW | corrected, including `derived_is_current`'s claim to have production callers — it has none left |

**And my own fix for the last of those was caught by the suite within a
minute**: `world_finalize.py` reads `args.session`, which is `None` on
every ordinary invocation because the CLI resolves the latest session
itself. The check would have downgraded every healthy record it was written
to protect. That is what a load-bearing test is for, and it is worth saying
that the mistake was mine and the catch was automatic.

**The round that ran the CLI nobody tests, and caught me making one mistake
twice.**

| finding | severity | resolution |
|---|---|---|
| **`world_registration.py --write` crashed on its ordinary invocation.** It read `args.session`, which is `None` whenever the caller lets the CLI pick the world's only session — which is what its own `--session` help says it is for. `session_manifest_path` then did `derived_dir / None`, and the run died **after** `register()` had done the expensive Sim3 pass, throwing the walk away. That is the loss `register_session`'s try/except exists to prevent, reintroduced in a different file. **Nothing in the suite calls that CLI** | **BLOCKING** | the resolved `session_id`. I made this exact mistake in `world_finalize.py` an hour earlier, wrote a comment there explaining it, and then made it again here — which is the honest measure of how much a comment protects the next file |
| **The previous round's mapping fix was too broad.** `stopped_unbuilt` carries two states and pointing the whole thing at `interrupted` made a complete world that merely needed a rebuild render as a red *"Interrupted … what was built before it stopped is here"*. The same reviewer also showed the fix's stated premise to be false: `stop_session()` defaults to `hold_lock=False`, so every offline caller releases the lock and **then** builds | HIGH | the distinction moved to the **projection**, where both facts are in hand — geometry present → `finalizing` ("wait" is true), no geometry → `interrupted` (nothing to wait for). `lifecycle.state` did not move, because it is on the wire and iOS decodes it. Both halves are pinned so neither can be fixed at the other's expense again |
| **The LRU change made a shared cache thread-unsafe.** This producer is a process-lifetime singleton reached from `asyncio.to_thread` on both the publisher's poll loop and every websocket subscribe handler. A stress harness produced `KeyError` 15 times in 16 threads — which is in `snapshot()`'s except tuple, so the world blinks out of existence on the phone for a poll — and `RuntimeError: dictionary changed size during iteration`, which is **not**, and escapes to the publisher's consecutive-failure counter | HIGH | `pop(key, None)` and an eviction that does not iterate a live dict |
| **The contract said the opposite of what the code did, and the id had not moved** | HIGH | see below |
| a listing flake that was a real wart; four comments and one tautology assertion | MEDIUM/LOW | below, and corrected |

**The contract id had to move, and that changes the retest.** The status
projection's meaning changed, and `contracts.py`'s own rule is "equality
only; a mismatch means we are not talking about the same agreement". So
`world_builder.status/2026-09-06` → `/2026-09-10`, in the Tower, in four
contract documents, in the iOS constant and in 13 iOS test literals.

**The phone equality-checks that string and refuses a Tower it does not
know**, with a visible sentence — *"The Tower offers a World Builder
contract this version of the app does not understand"*. So a build from
before this branch will now show that and nothing else. That is deliberate
and it is loud, which is what a dated identifier is for; it also promotes
"rebuild the app" from a caution to a precondition, and §15 leads with it.

**The listing flake was a real wart, and then the test was wrong too.**
Two worlds created inside one clock tick (Windows: ~15.6 ms) share an
`updated_at`, and sorting on that alone left their order to whatever
`list_world_ids` yielded — so rows could swap places between polls, under
the wearer's finger. The sort is a total order now. But the failing test
asserted "newest first" of two worlds created back to back, which is not a
property anything can have when two things are the same age: it was
answering an unanswerable question. It establishes recency explicitly now,
and a second test pins what a tie actually guarantees.

### Round 16 — the round that caught the previous round's fix, twice

**The pattern this campaign kept producing, produced once more: a fix that
was right about the case it was written for and wrong one case over.**

Round 15 split `stopped_unbuilt` on `geometry.available`. Round 16 found
that predicate wrong, found the branch beside it lying, and then found the
first version of *its own* fix wrong — caught not by a reviewer but by the
hostile suite, which had a test for exactly the rule I had just broken.

| finding | severity | resolution |
|---|---|---|
| **`geometry.available` is true over a world with nothing in it.** `engine.build()` calls `write_derived` **unconditionally**, so a walk down a dark corridor writes `poses.json`, `points.json` and a manifest saying `points: 0, poses_solved: 0`. Round 15's split therefore still said `finalizing` over an empty world, and iOS — which decides what to draw from `WorldEvidence.hasGeometry`, `(elements ?? 0) > 0 \|\| (poses ?? 0) > 0` — drew a "Finalizing" nothing would ever change. **The permanent Finalizing, back through a different door one round after it was closed.** Found by the lead reading the two predicates side by side; reproduced with a featureless capture | **BLOCKING** | `_has_drawable_geometry(payload)` — deliberately the same question iOS asks, of the same two numbers. If the Tower and the phone disagree about whether there is anything to show, the Tower tells the wearer to wait for a screen the phone will never have anything to put on |
| **A derived tree with no manifest reported `available: false, element_count: null`, and the branch's own comment promised "the world itself opens normally".** It did not. Measured: **1,347 points and 4 camera poses on disk, served 200 by the geometry route**, and the phone drawing *"Needs retry — nothing usable came of this session. Walking the space again is what produces another one."* That is the failure this campaign is named for, reintroduced by a branch written to fix it. Found by a reviewer; reproduced by the lead in one test | **BLOCKING** | `_figures_from_the_tree`. **A manifest is a summary of poses.json and points.json.** When the summary is unusable those files are still there, so they are counted. `_summarise_pose_rows` reimplements `engine.build`'s arithmetic including the rule an anchor counts only in a segment that solved; a test compares the recount against a hand-written manifest built around exactly that case, and against the engine's own on a real build |
| **The same branch's evidence was false for four corrupt shapes.** It said the world's manifest "names another session and this session has no copy of its own" — for unreadable bytes, a top-level list, a schema from the future and a required key set to null, where **both files existed and both named this session**. `_validate_manifest` was extracted to give corruption a clean refusal; the refusal was then laundered into a confident wrong story | HIGH | two cases, said as themselves: "either none was written beside it or the one that was cannot be read" |
| **THEN MY OWN FIX WAS WRONG, AND THE SUITE CAUGHT IT.** Recounting a tree whose manifest declares `schema_version: 999` fabricates: that manifest is *positive evidence* the tree was written by a Tower this build does not understand, and a future format could keep the row shape and change what `status` means. `test_a_manifest_from_another_schema_is_refused` went red within a minute of the change | HIGH | `_declares_a_schema_we_cannot_read`. The three reasons `_validate_manifest` refuses are not equally informative and are now treated separately: unreadable bytes say nothing about the tree, missing keys leave the schema intact, a foreign `schema_version` is the one that must refuse |
| **AND THE FIX FOR THAT FIX DID NOT WORK, FOR A REASON WORTH THE WHOLE ROUND.** `_FileCache` keyed on the path alone. `_payload` reads the session manifest through `_validate_manifest`, which answers `None` for a foreign schema; my new check asked the *same file* what it CLAIMS, hit the cache, got the validated `None` back, and concluded the file made no claim. **One file, two questions, one entry — the second caller silently gets the first caller's answer.** The refusal I had just added passed, and the schema-999 tree was recounted and served | HIGH | the cache keys on `(kind, path)` and every call site names its read. A latent hazard in a shared cache, activated by being the first code to ask a second question of a file |
| **The picker said `unbuilt` — "stopped and never built" — over a session whose manifest proves a build ran** and whose tree was deleted. `session_state`'s docstring claims it "mirrors `_lifecycle`", which grew a whole `interrupted` branch to stop saying that. On the more damaging of the two surfaces | HIGH | `has_manifest`, through `manifest_for` — the same rule the status producer and the geometry route use. Four readers, one rule |
| **The pre-flight reports a correctly configured Tower as unset if `.env` has a byte-order mark.** python-dotenv 1.2.1 does not strip it: the key parses as `'\ufeffTOWER_WORLD_ROOT'`. **On Windows a BOM is the default** — Notepad and PowerShell's `Out-File` both write one, and this repository's own CLAUDE.md warns about it. Found by the test's own premise assertion, on the check the wearer runs immediately before a retest | HIGH | `dotenv_values(env_path, encoding="utf-8-sig")` |
| **The listing's new tiebreak could 500 `GET /worlds`.** A `null` or mistyped `created_at` raises `TypeError` *only* when a tie on `updated_at` sends Python to the second key — so the single-key sort it replaced could not reach it. The sort is outside the try/except and `routes/geometry.py` has no handler: one malformed row and the picker loses all 163 worlds | MEDIUM | `_sortable` — compare within a type, so the order between types is arbitrary but total |
| **`world_builder.worlds/2026-09-06` changed meaning and its id did not move.** `session_state` gained the `has_geometry` requirement and the manifest-without-a-tree case: a word a phone already implements now arrives in states it did not before, which is the exact argument used to justify bumping the status contract in the same batch. Asymmetric | MEDIUM | `/2026-09-10`, in the Tower, the route, the iOS constant, six test literals and the contract document |
| **The iOS "it is worth waiting for" note was attached to a state where nothing is known to be running.** `.improving` requires `buildInProgress == true` from a lock the Tower can see, and the measured 179 seconds are that state. `.finalizing` also covers `stopped_unbuilt` with geometry, where `build_in_progress` is `null` — a finalization that died, or an offline rebuild nobody will run. Same permanent-Finalizing shape, with a stronger promise bolted on | MEDIUM | the two cases split; `.finalizing` names what is missing and says the phone cannot see a build running. **Not compiled** |
| **The picker promised the wearer geometry that is not there, on eleven real sessions.** `has_geometry` on the listing meant "`poses.json` and `points.json` exist", and `engine.build` writes both unconditionally: a walk that solved nothing leaves a `points.json` of **14 bytes**, `{"points": []}`. Measured on the real 163-world root — 11 sessions listed `complete, has_geometry: true` while the status channel for the same session projected `needsRetry`. `WorldPickerView` branches on that field, so the row said "opening this shows something" and the panel behind it said nothing came of the walk. Found by the lead running the round's own changes against the real root | HIGH | `has_geometry` is now the same question the status channel asks, of the same figures — from the manifest the listing already reads, which every one of the 49 real sessions with a tree carries. Free: 82.5 ms → 90.0 ms. Parsing every `points.json` instead would have cost 389 ms for 340,549 points |
| **And the first word chosen for those eleven was a lie too.** Making `has_geometry` truthful sent them to `interrupted`, which iOS renders **"Interrupted"** — a claim that the walk FAILED, over one that finalized cleanly and merely found nothing to reconstruct | MEDIUM | `unbuilt`, which iOS renders **"No geometry"** — the true sentence, and the one the panel agrees with. `interrupted` is kept for the case that earns it: a manifest recording real figures whose tree is gone. `_nothing_to_open` makes that distinction once, from the figures |
| `world_registration.py --write` printed a success line for placements carrying no `input_digest`, every one of which the serving path refuses | MEDIUM | a stderr warning naming the fix |
| three comments that said what the code did not do — a cache bound argued against a number the file never had; "every offline caller" where the shipped offline driver passes `hold_lock=True`; and a `TypeError` narrated as committed history when the diff shows the superseded line never touched `args.session` | LOW | corrected, including the last one as a labelled correction rather than a silent edit |
| doc drift: the geometry contract's status id, `CARTRIDGE-RESULTS.md`'s `ready` and `stopped_unbuilt` rows, `WORLD-BUILDER-WORLDS.md`'s `state` vocabulary | LOW | all four amended |

**Every round-16 fix has a test proven load-bearing by mutation** —
**twelve** mutations, each reverting the fix to a specific earlier version
of itself (round 15's `available` gate, round 14's two mappings, the
uncounted tree, the old evidence sentence, the existence-only
`has_geometry`, the FIFO listing sort), each caught.

**Where the findings came from is the notable thing about this round.**
Three were caught by the existing suite rather than by review — the first
time in this campaign that has happened more than once in a round, and
the suite catching the schema-999 regression within a minute is the
clearest evidence yet that the earlier rounds' tests are doing work. Two
more came from running the round's own changes against the real 163-world
root rather than against fixtures, which is the only way the eleven
14-byte sessions could have been found.

**After the fixes, all 67 sessions on the real root agree between the two
surfaces** — every `complete` row is drawable, every `unbuilt` and every
no-geometry `interrupted` row projects `needsRetry`, and the one live
session reads `receiving` on both. No snapshot raised; the slowest was
19.7 ms.

### Round 17 — the round that reviewed the round that reviewed the round

**A reviewer who did not write round 16 was asked to break it, and broke
three things — two of which round 16 had introduced while fixing the same
class of defect.**

Round 16's central judgement survived: `_has_drawable_geometry` is the
right predicate, the recount agrees with the engine on every case the
reviewer could build, the cache genuinely prevents the re-parse, the sort
fix is real. What it had not done was ask **what else consumed the manifest
it replaced.**

| finding | severity | resolution |
|---|---|---|
| **A readable reconstruction reported as unreadable, and only by one of the two readers.** Round 16 refused to recount a tree whose manifest declares a `schema_version` this build does not know — reasoning that a future format could keep the row shape and change what `status` means. The reviewer built it: `element_count: null`, the phone drawing *"Needs retry — nothing usable came of this session"*, and `build_manifest()` **serving 1,347 points for the same tree in the same breath**, because `world_builder_geometry._read` uses `read_derived(verify=False)` and never looks at `schema_version` | **CRITICAL** | the gate removed. One reader refusing what the other serves is the Tower disagreeing with itself, which is the failure this campaign is named for — worse than either answer given consistently. The manifest's FIGURES are still refused, which is what a schema version is about; the files are counted, which is what both readers already do. `test_a_foreign_schema_does_not_split_the_two_readers` now pins the agreement directly |
| **`path_length.reason: "the derived poses are unreadable"` beside `pose_count: 4` counted from those poses.** `read_derived` returns None both for a tree it cannot read and for one it will not SERVE, and the second is far commoner — a build older than its keyframes is refused by design. Round 16 routed the no-manifest case into a block that had only ever had the first sentence | HIGH | the two are told apart by asking the currency gate, not by guessing from a `None`. And the underlying refusal was itself wrong: **`derived_currency` returned `False` — "a manifest exists and disagrees" — when there is no manifest at all**, so a legacy world was 404 on the serving path while the status channel promised it. The three-valued docstring had spent a paragraph rejecting exactly that answer for the one case it did handle |
| **"the world opens" asserted over a world with nothing in it.** `has_readable_figures` was `tree_figures is not None`, which asks only whether the files PARSED — and a featureless walk parses perfectly and counts zero. The same mistake `_has_drawable_geometry` was written to stop the projection making, made again in prose two branches away and shipped to the phone as `lifecycle.reason` | HIGH | `_figures_are_drawable`, the same question of the figures before they become a payload |
| **On Windows the recount fails a quarter of the time next to a live writer.** `write_json_atomic` ends in `os.replace`, and a replace onto a path a reader has open fails with WinError 5 — symmetrically, so does the reader's `open()` during the writer's window. Measured by the reviewer: **3,519 failures in 14,486 reads, 24%**, all `PermissionError` from the open rather than torn parses. Round 16 widened that window from tiny manifests to a **2.86 MB** file, and reported each failure as "could not be read" — at the publisher's 2 Hz, `geometry.available` flickering true/false while a session rebuilds | HIGH | two immediate retries (a replace is over in microseconds; no sleep in a poll path), and `ValueError` deliberately not retried — a file that parsed and was wrong will parse and be wrong again |
| **"Four readers, one rule" was written on a comment above a fifth reader with its own rule.** `manifest_for` checked `isinstance(dict)` and `session_id`; the status producer additionally ran `_validate_manifest`. Measured contradiction: for a foreign-schema manifest whose tree is gone, the picker said `interrupted` — "a build ran and its output is gone" — while the panel behind that row said "this walk produced no geometry" | MEDIUM | `validate_manifest` moved to `WorldStore`, where the manifest's shape is owned, and both readers call it. A no-op on real data: all 49 sessions with a derived tree pass both rules |
| **`_sortable` put garbage at the TOP of a "newest first" list, and still raised.** Ranking by type name meant `("str", …)` sorted above `("num", …)` under `reverse=True`, so `updated_at: "2020-01-01T00:00:00Z"` took the first row ahead of all 163 real worlds. And `float(10**400)` raises `OverflowError` — out of a function whose docstring opened *"A key that never raises"*, into the same 500 on the same trigger | MEDIUM | rank 1 for a real timestamp and 0 for everything else, so a malformed value sorts last; no `float()` call, so no overflow; NaN excluded, because it does not raise — it silently makes `sort` return an arbitrary permutation |
| **The render page chose an empty walk over a real one.** `resolve_session` picks the newest session "that has geometry", and that meant the files exist — so a second walk that solved nothing outranked the first walk's reconstruction and the world opened blank. Found by a two-round-old test going red when the listing's copy of the predicate was corrected | HIGH | two of the three copies moved into `WorldStore.session_has_drawable_geometry`; the third answers a genuinely different question and stays |
| **`--write` warned about placements the route would discard and then wrote them anyway** — and `write_placements` replaces the file whole, so on a world already carrying a good `placements.json` (the 2026-09-09 walk carries a 35 KB one from its global solve) a single run would destroy it | LOW | refuses and exits non-zero. The warning's own reasoning said not to write |
| two more comments stating false history — the sort tiebreak called a regression when `e60d753` already had three keys and the old key raised on the same input, and a `_FileCache` sizing comment still saying four reads per session where there are now six | LOW | both corrected, the first as a labelled correction |
| one new test could not fail for the reason it claimed (it asserted `"read" in reason`, which several different sentences satisfy), and one docstring claimed to check the anchor rule with a fixture that cannot exercise it — proven by mutation, and caught only by a sibling test | LOW | the first asserts the exact sentence; the second's docstring now says which test owns that rule |

#### The render page was drawing the wrong walk, and an old test found it

**The most wearer-visible defect in either round, and nobody was looking
for it.** `world_builder_render.resolve_session` picks which session to
draw — *"the newest session of the world that has geometry"* — and "has
geometry" meant the two files exist. So on a world walked twice where the
**second** walk solved nothing, the empty 14-byte `points.json` outranked
the older walk that holds a reconstruction, and opening the world gave a
blank page.

It surfaced the moment the listing's copy of that predicate was corrected:
`test_the_three_surfaces_ask_the_same_question_of_the_same_files` — written
two rounds earlier, after a reviewer counted three copies of the predicate
and observed they were free to drift — went red with a `TypeError`, and
chasing it found the render page asking a question it had no business
asking.

Two of the three copies are now one function in the store,
`session_has_drawable_geometry`. The third stays separate **on purpose**:
`_has_session_geometry` answers "was there a build", which is what decides
a lifecycle state, and a test already pinned that distinction. So the three
surfaces may differ in exactly one place — a tree that exists and is empty
— and the amended test asserts both that they agree everywhere else and
that they differ *there*, in the right direction.

**And it caught the unification being too broad, six tests at once.**
Holding `manifest_for` to `validate_manifest` refused a manifest carrying
a session id and a digest but no counts -- so `usable_placements` stopped
serving every registered segment of such a session, and a registered
transform silently became unplaced. The figure check exists because the
STATUS CHANNEL reports those figures; a reader asking "which build
produced this" needs only identity and provenance. `require_figures=False`
splits them, and the schema check stays shared, which is the half that
fixed the picker/panel contradiction. **Both of these appeared only in the
full suite** -- the targeted runs I had been doing did not include
`test_world_builder_placements.py` or `test_world_builder_render_route.py`,
and the second of those is what found the render page drawing the wrong
walk.

**And the suite caught me deleting a function.** Moving `_validate_manifest`
to the store, I replaced a span that also contained `_read_manifest`;
`NameError` inside a minute. That is the third time in two rounds the
existing tests have caught a change of mine before any reviewer saw it.

#### The interpreter, which I got wrong for a whole working session

**Rounds 16 and 17 were tested on a Python with no `pycolmap` in it.**
`python` on this host resolves to the system 3.12, and the Tower's own
environment is `tower/.venv`. I noticed only when a full run came back
with eight failures in Object Memory, Scene Understanding and the privacy
suite — none of them World Builder, all of them `ModuleNotFoundError` for
`skimage`, `torchvision`, `easyocr`. Chasing what I had broken in
unrelated cartridges turned up that I had broken nothing: I was running
the wrong Python.

That is not a harmless slip. `build_world` in the fixtures drives the real
engine, and on an interpreter with no `pycolmap` it drives a **different
backend** — so every reconstruction figure round 16 and 17 asserted was
about a solver the Tower does not use.

**Everything was re-run on `tower/.venv/Scripts/python.exe`:**

| re-taken on the real interpreter | result |
|---|---|
| the round-16/17 suites | **95 passed** |
| the mutation proofs | **13 of 13 still load-bearing** |
| the real 163-world root | identical classification; listing 86.9 ms, slowest snapshot 20.1 ms, nothing raised |
| the listing-cost measurement | 373 ms to parse all 340,549 points (was 389 ms) |
| the pre-flight | all eight verdicts `[OK ]` |

**And the pre-flight gained a ninth verdict, because it would not have
caught this.** Every other check passes on an interpreter that cannot
reconstruct anything: `import tower` works from the tower directory
whether or not the package is installed, torch and OpenCV are commonly
present system-wide, and the vocabulary tree lives in `~/.cache`. So a
green pre-flight was achievable on a Python that would produce a world
with zero poses and zero points and say so nowhere — the same failure
shape as the missing-calibration case an earlier round added a verdict
for. `sfm_backend_importable` imports `pycolmap`, and its detail names
`sys.executable`, because "why is it failing" is almost always "you are
not running the Python you think you are". Proven in both directions:
`[OK ] pycolmap 4.2.0` on the venv, `[NO ]` on the system Python.

**Measured again on the real 163-world root afterwards:** 163 worlds,
listing in 92.9 ms, slowest snapshot 24.0 ms, **no snapshot raised**, and
every one of the 67 sessions agrees between the picker and the panel.

**Not fixed, and recorded rather than patched:** the real root holds **two
sessions whose manifest and `poses.json` describe different builds** —
`fcbca9e9…`'s manifest counts 463 poses where the file holds 467. Neither
is reachable through the recount (both carry a usable manifest), and which
of the two is right is a question about `engine.build`. The docstring that
called a manifest "a summary of these files" now says "meant to be", and
says why.

### Round 18 — the round that fixed things one line above the line it fixed

**Three HIGHs, one of them a defect round 17 shipped, and one of them a
defect round 17 measured and then fixed on the wrong file.**

| finding | severity | resolution |
|---|---|---|
| **`GET /worlds` still 500s on one bad session record — one line above the sort that had just been hardened to stop exactly that.** `sessions.sort(key=lambda s: s["started_at"])` sits at the top of the per-world loop, OUTSIDE the inner try that skips an unreadable session, outside `build_world_listing`'s only handler, on a route with none. `session_from_json_dict` does not coerce. A reviewer wrote an ISO timestamp string into one `started_at` and the exception escaped **before any world was returned** — Saved Worlds empty, all 163 worlds gone, over one record. They found it by applying the world sort's own justification to the line above it | **HIGH** | `_sortable` on both session sorts, here and in `resolve_session`. Reproduced against the reviewer's script before and after |
| **The Windows-replace fix went on the two files that were not the problem.** Round 17 measured a WinError 5 collision and added retries to `poses.json` and `points.json`. `write_derived` replaces **five** files microseconds apart, and neither manifest reader caught `OSError` — `read_derived_manifest` catches only `ValueError`, `read_session_manifest` only `(JSONDecodeError, ValueError)`, `derived_currency` calls both with no `try`, and `_is_current` calls `derived_currency` outside its own. The `PermissionError` walked out to an HTTP **500** on 2.4% of requests beside a live writer, on a route whose comment promises *"404 now means ABSENT only"* and for which the phone has no branch | **HIGH** | one `_read_json_past_a_replace` in the store, used by both manifests, the support index, and — found by my own probe rather than the review — **`read_derived`'s poses and points too**, which are the largest files and hold the window open longest. `absent_on_failure=False` there keeps the deliberate design: a real fault still raises a 500, only a transient collision is ridden out. **Retry is not swallow** |
| **A FIFTH manifest reader, inside the function written to stop there being one.** `session_has_drawable_geometry`'s docstring says *"two surfaces need this answer and they must not compute it apart"* — and when its caller passed `None` it re-read the manifest with a LOOSE rule, re-admitting exactly what `validate_manifest` had just refused. A reviewer built a session whose manifest claims `points: 500` under an unknown schema over two empty files and got **three answers from one payload**: picker "complete", panel "Needs retry", render page blank. Shipped by round 17, hours old | **HIGH** | the shared reader. Mine, and the sharpest lesson of the round: a helper written to end a duplication became the sixth copy of it |
| **The schema gate was wired into placement serving, where it means nothing.** `SCHEMA_VERSION` versions the whole record family — `World`, `Session`, `Keyframe`, `KeyframeEdge` — the manifest inherits `world.schema_version` rather than the module constant, and the rows in `poses.json`/`points.json` carry no version at all (their shape is `world_builder.geometry/2026-08-25`, which has never moved under it). A reviewer bumped the constant against the real root: **408 placements across 10 sessions dropped to 0**, every segment a disconnected island — the picture the final solve exists to prevent — on a change with nothing to do with a Sim3 | MEDIUM-HIGH | `manifest_describing(..., purpose=)`. **Identity** (which build produced this: `input_digest`) takes neither the schema nor the figures; **figures** takes both. Re-measured after the fix: 408 → 408 |
| **The CLI guard and the reader it protects asked different questions.** `world_registration.main` guarded on the RAW manifest's digest while `usable_placements` judges with the validated one — and `world_build_session.session_manifest`, the function that STAMPS the digest, used a third rule. So the guard passed in exactly the case its own message describes, and the 35 KB `placements.json` on the 2026-09-09 walk was destroyed anyway | MEDIUM | all three on `manifest_describing(purpose="identity")`. "Four readers, one rule" is finally true, one round after it was written |
| **`float('inf')` sorted ahead of every real world** — `value == value` excludes NaN and admits the infinities, which is the outcome `_sortable` exists to prevent for a string. The Tower's own writer can emit it: `json.dumps` writes the bare `Infinity` token by default and `json.loads` reads it back | LOW | `math.isfinite` — **and then that raised `OverflowError` on `10**400`**, because `isfinite` converts to float first. The test written for the first version caught the second. Ints are finite by construction and are taken before the float check |
| the worlds-contract document claimed an older phone "decodes every field; it simply draws a different word" — it decodes nothing, because `WorldListingDecoder` returns `nil` on any mismatch | LOW | corrected |

**Two of the reviewer's measurements did not reproduce, and both are
recorded as theirs rather than folded in.** Round 17 cited **24%** read
failures beside a live writer, from the round-16 review; this reviewer
measured **2.6–2.8%** against a hotter writer and could not reproduce 24%.
And they measured the retry taking a 1.3 MB read from 2.77% to **0%**,
with **zero** additional writer failures — the reverse hazard round 17
worried about does not materialise, because `replace_with_retry` already
carries a 2,000 ms budget and the worst observed write used 6% of it.

My own probe — a FastAPI TestClient against the real router with an
unthrottled writer beside it, harsher than any real build — put all three
of round 17's immediate retries inside one collision window and the 500
still escaped. Five attempts over ~14 ms of backoff, paid only on failure:
**823 requests, 0 server errors, 831 concurrent writes.**

### Round 19 — two reviewers, one answer, and it was not in the code this campaign had been reading

**Two reviewers ran in parallel with deliberately different mandates: one
told to break round 18's store layer, one told nothing about what had
changed and asked only "what goes wrong when Tristan walks a room
tomorrow". They converged on the same finding, and it was neither a
regression from this campaign nor anything §14 covered.**

#### The ghost walk

`_holder_is_running` decides whether a lock file names a live builder. It
reads:

```python
if not psutil.pid_exists(pid):
    return False
if created_at is None:
    return True          # <- every legacy lock lands here
```

On the machine this retest will run on, **29 of 163 worlds hold a lock
file and not one of them carries `created_at`** — every one predates the
field. So all 29 are decided by the pid alone, and Windows recycles pids
freely. Both reviewers hit a live alias; one hit it on their first sample,
and the process it named was **one of their own audit shells**.

What that does: `_most_relevant` prefers a world with a live lock over
every saved world, so the phone's default subscription is hijacked. Both
reviewers saw the same payload — a fortnight-old empty world reported
`receiving`, `keyframes: 0`, `mapping_seconds: 1,407,085` (**16 days**).
One reproduced the whole consequence through a real uvicorn Tower and a
real 220-frame walk:

```
t+ 21.0s  model_state='finalizing'  elements=4219 poses=28   <- correct
t+ 28.1s  model_state='receiving'   elements=None poses=None <- lock released; the ghost takes over
t+ 29.1s  model_state='receiving'   elements=None poses=None
...unchanged for the remaining 90 s
```

The world built correctly and is correct in Saved Worlds. But **at the
exact moment finalization completes and releases its own lock** — which is
the moment §15 tells the wearer to watch for — the live screen reverts to
the ghost and says "Mapping", 0 keyframes, forever. It never says
"Finalizing → Saved". The wearer concludes the walk failed.

**The fix deletes nothing.** A process that started AFTER the lock file
was written cannot be the process that wrote it, and the filesystem keeps
that timestamp for free. That is the same argument the `created_at` check
makes, from a source every one of the 29 legacy locks already has. They
stay exactly where they are and simply read dead, which is what they are.
The tolerance runs one way on purpose: a real builder writes its lock
milliseconds after starting, so only a process that started **clearly**
later is called dead, and a live builder can never be judged dead by a
coarse timestamp.

Verified both directions end to end: with the fix the phone gets the real
world (`ready`/`finalized`); with it reverted, the same harness reports
`receiving`, 0 keyframes, on the ghost.

The pre-flight gained a tenth verdict, `no_world_claims_to_be_building`,
so the condition is visible before a walk rather than discovered during
one. It reads `29 lock file(s), none naming a live process` today.

#### The rest

| finding | severity | resolution |
|---|---|---|
| **The status channel had the uncoerced-timestamp defect the three HTTP surfaces had just been hardened against** — `world.updated_at > best_at` and `session.started_at > best_at` in `resolve_with_selection`, which runs BEFORE `snapshot()`'s try, and `snapshot()`'s except tuple carried neither `TypeError` nor `OverflowError`. The escape reaches the publisher's consecutive-failure counter and the panel **stops updating for the rest of the walk**. A reviewer built all 24 corruption shapes: every HTTP surface survived all of them, the live channel died on all of them | **HIGH** | `_sortable` on both comparisons, both exceptions in the tuple |
| **`_stop_capture` ran `supervisor.capture_closed()` on the event loop**, which holds the supervisor lock across a detach whose World Builder grace is **30 s**. Measured: **33.72 seconds** of dead event loop — no frames, no results, no `/health`, for every cartridge — reached by the ordinary "press Stop, lose WiFi". The sibling `capture_opened` had been moved off-thread for exactly this reason and this one was left behind | **HIGH** | `await asyncio.to_thread(...)`, and `_stop_capture` is now async. This is the brief's own non-negotiable: a World Builder failure must not degrade the other cartridges or Tower networking |
| **A wedged result producer silenced every cartridge.** Targets are polled sequentially with no deadline, so a World Builder read that never returned delivered **nothing at all, for anyone, across 12 poll windows** — no error, no `fail_target` — and a merely slow (2 s) producer made Document Memory and Scene Understanding **5× slower** | **HIGH** | `asyncio.wait_for` at 10 s, and a timeout now reaches the same consecutive-failure notification an exception does, so subscribers are told rather than left quiet |
| **A mid-stream frame-size change killed the walk.** `MotionTracker.measure` feeds the frame and a stored reference straight into `cv2.calcOpticalFlowPyrLK`, which asserts in **C** — a `cv2.error`, not a `ValueError`, so it walks past `observe`'s decode guard and past `world_build_session`'s `except OSError` into the outermost `except BaseException`. A reviewer drove 220 frames with a rung change at 120 through a real Tower: session `end_reason: error`, finalization `interrupted`, the remaining 100 frames discarded. **The wearer walked the whole room and gets "Interrupted"** | **HIGH** | the frame is rejected and counted, which is the correct answer and not merely the safe one — the calibration is per-resolution and exact, so such a frame could never have produced a usable pose. It also stops `IntrinsicsResolutionMismatchError`, which is raised in one place and **caught nowhere**, from ever being reached |
| **`purpose="identity"` widened an existing 500.** `build_manifest` reads `global_solve.segments` off the identity-path manifest with a `.get` chain, so a `global_solve` that is a string or a list is an `AttributeError` → HTTP 500. Unguarded at HEAD for a schema-1 manifest; round 18 removed the schema check that had been keeping unknown-schema manifests away from it | MEDIUM-HIGH | the chain is type-checked at every step, and the coverage verdicts come from a figures-validated manifest. The purpose split itself stays — refusing 408 real placements over a record-schema bump is what it was for |
| **A corrupt `world.json` was an HTTP 500 on an unauthenticated route.** `read_json_closed` raises `JSONDecodeError` on a truncated or non-UTF-8 file, `world_builder_geometry._read` catches only `WorldStoreError`, and `routes/geometry.py` has no handler | MEDIUM | a parse failure is a `WorldStoreError`, which every caller already turns into a 404. Probed after: truncated and non-UTF-8 both give `/worlds` 200, manifest 404, render 404. A genuine `OSError` still escapes, so "the disk is broken" stays a 500 |
| **`lock_holder` swallowed a transient read into "no lock at all"** — which every caller reads as "this world is idle", so one collision on the LOCK file reports a live walk as dead. The suite showed it before a reviewer named it: the one test asserting a live session reads `receiving` failed once under full-suite load and passes otherwise | MEDIUM | through the retry; an unreadable lock reports `unreadable: true`, which the method already had a word for |
| **`world.json` was the sixth file** `engine.build` replaces — `write_world` is called on the statement before `write_derived` — and it was the one still on a bare read. Round 18 fixed five and enumerated five | MEDIUM | through the retry, with `absent_on_failure=False` so a real fault still raises |

**The lead's own check of the two shared-code changes, done before any
reviewer reported on them.** Round 19 is the first round to touch code
every cartridge depends on, so the two riskiest changes were inspected
independently rather than left to review:

- *`_stop_capture` going async.* The worry was ordering: a `stream_start`
  can now interleave with a pending `capture_closed`, where before the
  blocked loop made that impossible. It does not matter, because
  `capture_closed` **does not stop or detach anything** — its own docstring
  says workers are deliberately not stopped there, and its body logs and
  calls `reap()`, which only notices processes that have already exited,
  under the supervisor's lock. The 33.7 s was a lock *wait* behind a
  concurrent `detach`, not work of its own. Moving a lock-wait off the
  loop reorders nothing a live worker can observe.
- *`wait_for` orphaning threads.* A failed subscription is sent one
  `result_error` and **then closed** (`_drain`), so a wedged target leaves
  the poll set after `MAX_CONSECUTIVE_TARGET_FAILURES` = 3 passes. That
  bounds the orphans at three per wedged target, each alive only as long
  as the wedge, against a default executor of at least twelve here. A
  cancelled `to_thread` result is discarded, so a late snapshot cannot
  reach a closed channel.

**Nine mutations across rounds 18-19, each reverting a fix to the exact
code it replaced, each caught.** The publisher one is worth naming: with
the deadline removed the two isolation tests take **121 seconds** instead
of 66, because the wedged producer really does block everything behind it.

### Round 20 — the dress rehearsal said NOT READY, and it was right

**Two reviewers again, one attacking round 19's shared-code changes, one
told nothing and asked to rehearse the retest on a real Tower.** The
first found a defect in round 19's own fix. The second stood up uvicorn,
drove it over the websocket the way the phone does, and returned a
**NOT READY** verdict with a reproduced blocker that no earlier round had
been in a position to see.

#### One reconnect became two worlds, and the campaign's own proof could not have caught it

§16's reconnect proof drove `world_build_session.py` directly. It never
crossed the websocket route. So the fourth root cause was fixed at the
builder and defeated one layer up — at every timing tried (clean close
0.5 s; link cut 12, 30, 60 s), a mid-walk reconnect produced **two worlds,
each "Complete", each half the walk.** Two mechanisms:

| mechanism | what happened | fix |
|---|---|---|
| **Supersession.** iOS reconnects in ~0.5 s; uvicorn takes 20–40 s to notice the old socket died, so the new connection's `stream_start` arrives while the old capture is still recording. `_start_capture` closed it as `stop` — and `resumable_capture()` offers a predecessor only for a capture that ended by `disconnect`. `continues` was never set; the old builder saw a polite close and finalised world 1 | it ends by `disconnect`, which is also the truth: the socket that opened it is gone |
| **The Tower notices first.** The last connection leaving stopped every cartridge session, World Builder's included — closing the builder's stdin while it sat in `_await_successor` inside its 90 s grace. World 1 finalised; the phone that came back 60 s later started world 2 | World Builder's session is left running while a disconnect-ended capture is inside its grace; the builder finishes on its own if nobody returns. Only World Builder is deferred — it is the one cartridge whose session follows a lineage |

**The reviewer's own harness, re-run against the fix at all four timings:
one session in one world every time**, both halves' keyframes (`kf=39`),
finalised `complete`/`solved`, picker Building → Finishing → Complete,
the same builder pid chaining into the second capture. At 60 s the Tower
had noticed the drop **25 s before** the phone returned — the case the
deferral exists for. Both mechanisms are pinned in the suite: the
supersession test holds both sockets open at once (the existing reconnect
test only ever covered the disconnect path), and the deferral test calls
the decision directly, because Starlette's TestClient cancels the handler
at the first `await` of the disconnect `finally` and never runs it.

#### The round-19 fix that was worse than its defect

| finding | severity | resolution |
|---|---|---|
| **The deadline leaked a thread per poll per wedged target.** `wait_for` cancels the await, not the thread — and the loop dispatched the same target on the next pass while the last was still running. Measured: one wedged target exhausted a 24-worker executor in ~252 s at defaults, **the executor the capture path shares**, after which every `asyncio.to_thread` in the process queued forever: `stream_start`, the disconnect cleanup, shutdown. Round 19's wedge had silenced the result channel and nothing else | **HIGH** | one snapshot in flight per target, by construction — a target with a running snapshot is never dispatched again. All free targets are dispatched together and the pass waits with one bounded `asyncio.wait`, which does not cancel. The pass is the slowest target capped at the deadline, not the sum, which also retires the "a 2 s producer makes the others 5× slower" finding |
| the subscribe-time snapshot — inline in the phone's own socket loop — had no deadline; a 3 s stall delayed that walk's frames 3 s on every reconnect | MEDIUM | the same `wait_for`. Proven by measurement, not by a suite test (§14.24) |
| a walk of wrong-sized frames was rejected with no signal anywhere a person looks | MEDIUM | counted from the journal into the live payload as two scalars, and the follower warns once per session |
| the picker said "Interrupted" over a session the panel called "Saved" — `end_reason` tested before a completed finalization | MEDIUM | a completed finalization with geometry outranks how the capture ended, the panel's own rule |
| a `world.json` whose top level is a list: `GET /worlds` 500 and **1,775 tracebacks** on the status channel, one per poll, picker and panel blind while the file existed | MEDIUM | a corrupt world, not a raise |
| `AccessDenied` on a lock holder's start time called a live builder dead — the one failure the lock exists to prevent | LOW | cannot-judge is not dead |
| `_stop_capture`'s docstring said it held the lock across a detach; it blocks *behind* someone else's | LOW | corrected |

**The suite is not the whole story, and this round is where that became
plain.** Nine mutations, each caught. But the blocker was found by a
real Tower and a real socket, and its regression tests exist only because
the harness said what to pin. One test was withdrawn rather than shipped:
a suite test for the subscribe deadline hung the harness, and a hang is
worse than a gap.

**Exonerated by measurement**, against the lead's own stated doubts: the render page is fast (5.3 ms median for 19,329 points, 6.2 ms on a second run — the Chrome timeouts were screenshot artifacts); loop-detection-always is a net win at every *live* horizon (the 22%-fewer-points result did not reproduce); `load_solution`'s eager read projects to ~49 MB peak at 6,000 keyframes; a kill mid-`write_derived` or mid-`append_jsonl` loses no authoritative data; the field artifact is **byte-identical** to its preserved copy (1,542 files, 116,452,684 bytes, 0 mismatches), verified twice.

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
   world's 19,329 points draw in **5.3–6.2 ms (160–190 fps)** across
   runs. Both ends of that range appear in this document — 5.3 ms in
   §13, 6.2 ms in the table below — because they are two runs of the
   same measurement, not two different claims. The table uses the
   slower one throughout, which is the conservative direction for a
   budget.

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
3. ~~**Loop detection may be a net negative on short walks.**~~
   **Withdrawn — measured, and it does not hold on this corpus.** The 22%
   fewer points at 84 keyframes came from a different walk and did not
   reproduce. Swept fresh, both arms, same host: at 84 keyframes loop
   detection gives **1,885 points against sequential's 1,888** (−0.2%, not
   −22%) and puts 0.76 of the posed keyframes in one component against
   0.46. It costs 1.6× the wall clock there, which is the real caveat and a
   small one at 10 s. Contradicted §13's own "did not reproduce" line for
   two rounds; a document audit caught the contradiction.
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
7. **`shutil.rmtree(ignore_errors=True)` still fails silently on Windows**,
   but the consequence this entry described is **fixed**, and the entry
   claimed otherwise for two rounds. `prepare_images`
   (`global_solve.py:408`) no longer trusts the delete: `recalibrated`
   (`:410`) makes the `if target.exists()` skip conditional (`:442`), so a
   frame that survived the rmtree is re-undistorted rather than fed to
   COLMAP under a `camera.json` it does not match. Both line citations here
   were also stale — the real sites are `store.py:369` and
   `global_solve.py:408`. What genuinely remains is smaller:
   `WorldStore.clear_derived` (`store.py:360`) can leave a file behind, and
   nothing prunes it. It is orphaned rather than served — every reader
   enters the derived tree through the manifest, which is rewritten
   atomically — so the cost is disk, not a wrong answer.
8. **The staging sweeper covers the solve workspace and nothing else.**
   `sweep_workspace` runs at the top of every solve
   (`global_solve.py:725`) over the workspace root and its images
   directory — which is where a hard-killed writer actually dies, since
   the builder terminates a solve child on every stop that outstays its
   budget. Every *other* `staging_path` writer has no sweeper:
   `write_keyframe_image` into `sessions/<sid>/images/`, and every
   `write_json_atomic` / `write_bytes_atomic` caller across object memory,
   document memory and the capture recorder. Those are short in-process
   writes, so an orphan needs a kill inside a few milliseconds; the
   exposure is real but small. **This entry read "nothing prunes them" for
   two rounds after the sweeper was added, contradicting §13 four
   paragraphs above.** Found by the lead re-reading its own limitations
   against its own findings, which is what an audit of §14.7 and §14.10
   should have prompted the first time.
9. **`sources.json`** holds unrecreatable raw-frame provenance inside the
   workspace `prepare_images` deletes. Its restore across the rmtree is in
   §13. ~~and is already 523/643 stale~~ — **that half was not a
   limitation, it was a defect, and it is now fixed. See §4b.**
10. ~~**The venv's editable install points at a different worktree.**~~
    **Fixed during the campaign; this entry was stale.** It pointed at
    `Glasses-worktrees/all-cartridges-field-test-v1` — a tree with none of
    these fixes, and no longer even a git repository — and §13 records the
    repoint two sections above, so the document contradicted itself.
    Verified now: `import tower` resolves to
    `C:\Users\tvllo\Projects\Glasses\tower\tower\__init__.py`, and the
    pre-flight's `tower_package_is_this_checkout` is what keeps it that
    way. Caught by a document audit.
11. **Nothing iOS was compiled.** Windows host. Every Swift change in this
    campaign was written and reviewed by reading, and **none of it has been
    through `xcodebuild`**:

    | file | change | risk if wrong |
    |---|---|---|
    | `TowerWorldBuilderClient.swift` | status contract id → `/2026-09-10` | a one-line string; the tests that pin it moved with it |
    | `WorldLibrary.swift` | worlds contract id → `/2026-09-10` | same |
    | `GlassesTests/*.swift` | 13 + 6 test literals | a stale literal fails the test, which is the point |
    | `WorldPresentation.swift` | the `.improving, .finalizing` case **split into two cases**, each with its own note | **the only structural change.** `WorldStage` declares both cases and the switch has no `default`, so an omission is a compile error rather than a silent fallthrough — but that is a reading, not a build |

    The split is the one to look at first if `xcodebuild` fails. Nothing
    else added, removed or renamed a declaration.
12. **The wearer waits about three minutes after Stop, and the phone will
    show them the wrong world for all of it.** Measured on a replay of the
    real field capture, post-Stop:

    | | |
    |---|---:|
    | waiting out the background solve (`--solve-wait-seconds`, default 120) | **82.1 s** |
    | the final solve itself | **96.0 s** |
    | the final build | 0.8 s |
    | **total** | **178.9 s** |

    Nothing moves during it. `mark_finalization` is written once, after
    both return; `global_solve` emits no journal events; and
    `mapping_seconds` freezes the moment `ended_at` is set, so the
    published payload is byte-identical at second 3 and at second 179.
    Meanwhile `WorldPresentation` keeps the Picture button live and opens
    the **pre-final-solve** world. That build is the shattered one — the
    final solve is what takes 16 components to 6.

    **So the single most likely way this retest reads as a failure is a
    wearer who opens the world ten seconds after Stop, sees fragments, and
    concludes it did not work — 170 seconds before it does.** §15 says so
    now, in the step where it matters.

    One thing was changed on the phone: the note under the offered world.
    It said *"This world is being finished, so it will change,"* which reads
    as polish, and this is not polish — the final solve is what takes the
    reconstruction from 16 components to 6. It now says the finished world
    is very different from this one and is worth waiting for. **Not
    compiled** (Windows host); it is a string literal inside an existing
    `case`, which is as safe as an unverified Swift edit gets, and it is
    still unverified.

    The button is deliberately *not* disabled — a wearer who wants to look
    is entitled to. The rest is not fixed because the two candidate fixes
    are a live-path progress write (the family limitation 6 declines to
    open) and retuning `--solve-wait-seconds`, which trades the wearer's
    wait against a final solve that has to redo matching a terminated child
    had already done. Neither should be guessed at; both are a measurement
    away.

13. **The World Builder screen must stay foregrounded for the whole walk.**
    `.onDisappear` sends the cartridge stop, and the re-arm on reconnect is
    guarded by `isOnScreen`. The footnote a wearer can see is intent, not
    liveness: the controller never polls, so if the Tower stopped the
    session at minute 3 the phone still reads "World Builder is active on
    the Tower" at minute 5. Pocketing the phone is not a neutral act, and
    there is no glasses-side HUD to say otherwise.

14. **`capture_dirs` names only the first capture**, so `_source_frame`'s
    fallback is lineage-blind even though the ledger beside it is not.
    See §4b.

15. **A world built before this campaign cannot say how current its
    geometry is**, once a second walk overwrites the world's manifest.
    `write_derived` writes a per-session copy from now on; existing worlds
    have none. Since round 16 the status channel reports such a session
    `ready` with its poses and points **counted from the files
    themselves**, `geometry.current` false, and a reason saying currency
    cannot be judged — so the world opens and nothing claims to know its
    age. Rebuilding it (`world_finalize.py`) gives it a manifest.

16. **`stopped_unbuilt` projects to `finalizing` when there is something to
    show**, which reads on the phone as a world the Tower may still
    change. Correct while a build might genuinely be running, wrong once
    nothing ever will — and on a session with no finalization record the
    Tower cannot tell the two apart. Round 16 narrowed the damage twice:
    a walk that produced no drawable geometry now projects `interrupted`
    instead, and the iOS note for this state no longer promises the world
    "is worth waiting for" (it says nothing here can see a build running).
    What remains is a world that IS intact, is behind its keyframes, and
    whose rebuild nobody will run: it reads "the final pass has not
    landed" forever. Rebuilding it is the fix and the note now says so.

17. **A junction under the captures root can alias a successor.** On
    Windows `is_symlink()` is False for one, so `_find_successor` cannot
    distinguish it from a real directory. Pre-existing; it needs a policy
    about what a junction under an artifact root means, not a patch.

18. **A `final_solve: "solved"` over an empty build would render as
    "Saved".** `WorldPresentation.stage`'s `.finalized` arm is
    `hasGeometry || solve == .solved ? .saved : .needsRetry`, and the
    `|| solve == .solved` clause exists for older records that carry no
    figures. If `global_solve` ever reported `solved: true` and the
    build that follows it produced zero poses and zero points, the phone
    would call that world Saved. **Found by the lead inspecting the
    branch next to round 16's; NOT reproduced.** Every path I can see
    makes it unreachable — the solve reports `solved` only when it
    produced a reconstruction, and `engine.build` merges that
    reconstruction — so changing iOS on the strength of the reading
    alone would be speculation, and this build cannot compile iOS to
    check. Recorded rather than patched. The Tower-side half is already
    truthful: such a session reports `element_count: 0`.

19. **The recount's cost is now measured, and it is a 34 ms parse in a
    2 Hz poll path.** A reviewer timed `_figures_from_the_tree` against
    the largest real world (`52ed8e0a…`, 26,634 points, 2.86 MB):
    **34.2 ms** for the points count, 5.2 ms for the pose summary, peak
    traced memory **13.91 MB** during the parse — and **512 bytes**
    retained afterwards, so the docstring's claim that only the length is
    kept is true. Across six polls with one producer, each loader ran
    **once**: the cache does what it is there for.

    What remains unmeasured is that shape at scale. The recount fires
    only for a session no manifest describes, of which the real root has
    **none** — every one of round 16's and 17's exercises of it is
    synthetic. A host whose worlds all predate per-session manifests
    would pay 34 ms per world per change, bounded by `_FileCache` and by
    how often the files actually move, and nobody has run that host.

20. **Two real sessions' manifests disagree with the files beside them.**
    `fcbca9e9…/158ef0ef…` — the flagship walk — has a manifest counting
    463 poses (`solved + refused + anchor`) where `poses.json` holds
    **467** rows, and `4b31766…/1c6e9b39…`'s manifest carries
    `poses_anchor: null` where the file has 36. Found by a reviewer
    scanning all 49 real trees with the recount and diffing it against
    each manifest; 47 agree exactly.

    **Not reachable through anything this campaign changed** — both
    sessions carry a usable manifest, so the recount never runs for them
    — and which of the two numbers is right is a question about
    `engine.build`'s counting, not about the readers. It is recorded
    because it falsifies, as a statement about this Tower's disk, the
    sentence "a manifest is a summary of these files", and because a
    463-vs-467 discrepancy on the walk this whole campaign is built
    around deserves to be looked at by someone with the solver in
    front of them.

21. **A reconnect can leave the WORKER REGISTRY's lineage incomplete,
    and it is load-sensitive.** Two full-suite runs failed two *different*
    tests in `test_world_builder_autostart_e2e.py`; the file passes 5/5 in
    isolation and 4/4 under four concurrent copies, so it takes real
    suite-level load to show.

    The one I chased asserts `second_capture in workers[0]["lineage"]`
    after a socket drop and reconnect. When it fails, `len(workers) == 1`
    still passes and the surviving worker's lineage holds only the FIRST
    capture — so no second builder was spawned and no append happened.
    That points at `CaptureWorkers.capture_opened`, which consults
    `_gate_open(spec)` before attaching and **fails closed by design**: if
    the World Builder gate is momentarily shut when the reconnect's
    capture opens, the registry never learns the new capture id.

    **The field impact is bounded, and this is why it is a limitation
    rather than a blocker.** The registry's lineage is bookkeeping for
    deciding whether to spawn; what actually carries a walk across a
    reconnect is the CAPTURE-side lineage — `continues_capture` plus
    `_await_successor` — which is this campaign's §4b work and is proven
    end to end by the production reconnect walk (29 sources, 29 resolved,
    0 wrong, both captures named). The comment in `_attach_to_registry`
    says the same thing: *"the existing follower will walk into this
    capture by itself"*. What an incomplete registry costs is recognition
    of a LATER successor, i.e. a second reconnect.

    **Not diagnosed to root and not fixed.** It is pre-existing —
    `capture_workers.py` and that test predate this campaign, nothing in
    this campaign touched either, and the test references none of the code
    these rounds changed. It needs someone to instrument the gate's state
    at `capture_opened` during a reconnect, which is a capture/worker
    question rather than a World Builder one.


22. **Leaving the World Builder screen mid-walk ends the walk, and coming
    back starts a second, duplicate world.** `.onDisappear` sends
    `session/stop` while the capture is open, so the first builder ends
    `end_reason: interrupted` (and, since round 20, is listed `complete`
    if it finalised with geometry). Returning sends `start`, which attaches
    a **new** builder to the same still-open capture — and that builder
    re-reads it from frame 0, producing a second world holding the whole
    walk beside the first one's fragment. Reproduced by a dress-rehearsal
    reviewer (`s3`, 20 s away). **Not fixed here:** the honest fix is on
    the phone (do not stop the session on a screen change while the
    capture is live), which cannot be compiled on this host, and the Tower
    side would mean changing what `session/stop` means. §15 forbids the
    action; this records what happens if it is taken.

23. **After Stop, the live panel drops to "No world yet / Last saved
    world … finished [Open]" rather than "Saved".** When the world becomes
    ready the unpinned payload flips to `selection: latest`, and
    `TowerWorldBuilderClient.publishLastReport` presents that as `.idle`
    with a recent world. The walk is fine — the picker says "Complete" and
    Open shows "Saved" — but §15's earlier wording ("Finalizing, then
    Saved") described a transition the live panel never shows, and a
    reviewer following it read the world vanishing as a failure. iOS
    presentation; recorded and written into §15 as "tap Open — that is
    success". Measured on the real 2026-09-09 capture through the Tower:
    "Improving…" at +0.4 s, frozen, then "finished" at **+144 s**.

24. **The subscribe-time snapshot deadline is proven by measurement, not
    by a suite test.** `routes/results_ws.py` now wraps the first snapshot
    in the same `wait_for` the poll loop uses. A suite test for it hung
    the harness: Starlette's TestClient never delivered the error reply
    even after the stalled thread finished, and the test was withdrawn
    rather than left as a hang. What stands is the primitive, measured on
    this Python (`wait_for` over a running `to_thread` raises
    `TimeoutError` at the deadline, 0.30 s), and the round-19 reviewer's
    real-uvicorn stall harness, which is what found the path. Someone
    with a real client should confirm the `result_error` arrives.


---

## 15. The physical retest

**On the Mac, first — and this is a real step, not a formality.** The branch
did not compile for several commits and nothing said so (see the verdict),
and the iOS changes since are large.

```
xcodebuild -project Glasses.xcodeproj -scheme Glasses   -destination 'platform=iOS Simulator,name=iPhone 16' build   # expect this to be where a problem shows up
xcodebuild -project Glasses.xcodeproj -scheme Glasses   -destination 'platform=iOS Simulator,name=iPhone 16' test    # GlassesTests
```

(Bare `xcodebuild build` / `test` do not work here: the repo's own handoffs
use the `-project`/`-scheme`/`-destination` form, and `test` needs a scheme.
A dress-rehearsal reviewer tried the bare form and it failed.)

**The app MUST be rebuilt from this branch.** **Two** contract ids moved,
and both are compared for equality:

| contract | old | new | what an older app does |
|---|---|---|---|
| status | `world_builder.status/2026-09-06` | `/2026-09-10` | refuses the Tower outright — *"The Tower offers a World Builder contract this version of the app does not understand"*, and nothing else |
| worlds (`GET /worlds`) | `world_builder.worlds/2026-09-06` | `/2026-09-10` | `WorldListingDecoder.listing` returns `nil` → `WorldListFetchError.undecodable`: **Saved Worlds does not open** |

Both refusals are deliberate and both are loud, which is what a dated
identifier is for. An older TestFlight or Simulator build shows the first
sentence and an empty picker, so there is no way to mistake a stale app
for a broken Tower. §13's rounds 15 and 16 say why each id had to move.

**Install the DEBUG build. This is the binary go/no-go for the whole
retest.** `TowerClient.swift` guards the entire streaming path behind
`#if DEBUG` — `sendFrame` (1325), `sendStreamStart` (**1539**) and
`sendStreamStop` (1571) all sit inside the block opened at 1315, and
`sendLifecycleMarker` (1694) inside the one at 1683. Only the CV Lab
commands (1787+) and the ping (1946) are outside.

Without `stream_start` the Tower never opens a capture (`routes/ws.py:808`),
so no builder attaches and **no world is created at all** — not an empty
one, nothing to open. `txSequence` and the `tx_seq` field are inside the
same guard, so this campaign's loss instrumentation is DEBUG-only too. The
guard was left alone; it is a build instruction, not a bug. Check it first,
because if it is wrong nothing else on this list can be observed.

**Check the phone can still find the Tower.** The address is compiled in —
`TowerConfiguration.defaultAuthority`, currently `100.110.156.55:8000`,
which is this Windows box's Tailscale address and matched it when this was
written. There is no discovery and no settings screen: if Tailscale hands
the machine a different address, the app must be rebuilt with it. Confirm
before walking, not after:

```
tailscale ip -4          # on the Tower host; must equal defaultAuthority
```

**Start the Tower with `start_tower.ps1`, or with
`tower/.venv/Scripts/python.exe` — not with a bare `python`.** The venv
is the only interpreter on this host with `pycolmap` in it, and a Tower
started on the system Python answers every request, records every
keyframe, and reconstructs **nothing**: zero poses, zero points, a world
that opens to "Needs retry". The lead of this campaign lost a working
session to exactly that mistake, which is why the pre-flight now refuses
it by name (`sfm_backend_importable`).

**Two pre-flight caveats a dress rehearsal established.**
`calibration_for_the_camera` checks the resolution of the **last capture on
disk**, not the rung the app will request next: a reviewer's scratch root
passed the pre-flight, walked to the end, and produced 68 keyframes and
**0 poses** (`final_solve: unavailable — cannot undistort: session has no
intrinsics`, phone "Needs retry"). If the app's resolution rung has changed
since the last walk, the pre-flight cannot know. And the Tower log names
`AppData\…\Python312\python.exe` for every builder: that is the
`process_ownership` launcher pair (venv-aware through `__PYVENV_LAUNCHER__`;
the field replay solved with GLOMAP under it), **not** the wrong-Python
failure — do not abort the retest on that line.

**Two things a dress rehearsal found about the walk itself, before the
pre-flight.**

- **Never let the socket drop, and never leave the World Builder screen
  mid-walk.** Before round 20 a mid-walk reconnect through the Tower split
  the walk into two worlds at every timing tried (§13, round 20). That is
  fixed on the Tower side, but the phone's reconnect budget is
  **5 tries over ~15.5 s** (`TowerClient.swift`), after which it logs
  "reconnect given up — use Connect to retry": more than ~16 s offline
  means a manual Connect, and the walk resumes only if that happens inside
  the Tower's 90 s resume grace. Leaving the screen sends `session/stop`
  while the capture is open — the first builder ends `interrupted`, and
  coming back attaches a **second** builder that re-reads the capture from
  frame 0 into a second, whole-walk world (§14.13).
- **After Stop, the live panel will drop to "No world yet / Nothing is
  being built right now / Last saved world: Walk · … · finished [Open]".
  That is success, not failure — tap Open.** When the world becomes ready
  the unpinned payload flips to `selection: latest`, and the app presents
  that as idle-with-a-recent-world rather than "Saved". "Saved" appears
  only on the pinned view after Open; the picker says "Complete". A
  reviewer following the previous version of this step literally read
  the world vanishing as a failure. Measured on the real 2026-09-09
  capture replayed through the Tower: "Improving…" at +0.4 s, the payload
  byte-frozen, then "finished" at **+144 s**.

**On the Tower, run the pre-flight.** It reads the launcher's `.env` with
`utf-8-sig` since round 16 — python-dotenv 1.2.1 does not strip a
byte-order mark, and on Windows a BOM is what Notepad and PowerShell's
`Out-File` write by default, so before that fix a `.env` edited in either
made the check report a correctly configured Tower as unset. If
`vocabulary_tree_cached` or
`tower_package_is_this_checkout` is `[NO ]`, fix that before walking: the
first means every solve runs without loop detection and the world comes out
in pieces, the second means you are testing a different checkout.

```
cd tower
.venv\Scripts\python.exe scripts\world_builder_env_check.py
.\scripts\start_tower.ps1          # -Port 8000 is the default
```

If PowerShell refuses the launcher with an execution-policy error, run it
for that one shell rather than changing the machine's policy:

```
powershell -ExecutionPolicy Bypass -File .\scripts\start_tower.ps1
```

Then, in one walk:

1. **Start World Builder on the phone and walk normally for 3–5 minutes.**
   Cover a loop — leave a room and come back into it. That is what loop
   detection needs.
   → *Watch: does the fragment count stop climbing and start falling?*
2. **Stop — and then wait about three minutes before opening anything.
   This step is the one most likely to make a working retest read as a
   failure.**
   → *Watch: does it say Finalizing, then Saved — not Interrupted?*

   Measured post-Stop on a replay of the real field capture: **82 s**
   waiting out the background solve, **96 s** for the final solve, 1 s for
   the final build — **179 s in total**, during which the payload is
   byte-identical from second 3 to second 179. Nothing is hung and nothing
   is wrong.

   **The phone will offer the world before it is ready**, with a note
   saying so. That offer is the *pre-final-solve* build, which is the
   shattered one — the final solve is what takes 16 components to 6.
   Opening it early shows fragments and proves nothing. Wait for **Saved**.

   Ten minutes with no change is a real hang, and
   `solve/<session>/solve.log` is where it shows itself.
3. **Open the world from Saved Worlds** — after "Saved", not before.
   → *Watch: does it open a coloured 3D world you can orbit, rather than a
   grid of grey tiles?*
4. **Open Diagnostics, then "Open the solver's 3D view".**
   → *Watch: segment colours and camera frustums — the old view, still there.*
5. **Leave World Builder and start CV Lab.**
   → *Watch: it starts without restarting the Tower.*

**If anything fails, capture before doing anything else:**

- `tower/data/world_builder/worlds/<world id>/` — the whole directory
- `tower/data/captures/<capture id>/` — especially `capture.json` and
  `frames.jsonl`. **Take every capture in the lineage, not just the one the
  session names.** A reconnect starts a new capture directory and the
  session record keeps naming the first; `solve/<session>/sources.json` now
  names them all, and `capture.json`'s `continues_capture` field links them
  (the 2026-09-09 successor carries
  `"continues_capture": "6a1b544c364c4469a1d75e3b58398fb0"`).
- the Tower's console output — there is **no** `tower/tower.log`: `logging_config.py` has no file handler and `start_tower.ps1` does not redirect, so capture the console (a reviewer following the previous wording went looking for a file that does not exist)
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

A document audit found the first version of this table under-reporting by
about 320 MB — it listed six directories and the campaign had created
eleven, plus five loose files. The full list, `du`-measured:

| directory | MB | what it is | keep? |
|---|---:|---|---|
| `wb-field-forensics\` | 115 | **The preserved field artifact.** Byte-identical to the live one, verified twice (1,542 files, 116,452,684 bytes, 0 mismatches). | **KEEP — evidence** |
| `wb-recover3\` | 138 | The most recent recovery of a pristine copy, after every fix. The best available reconstruction of the 2026-09-09 walk. | KEEP until the retest |
| `wb-recover2\` | 138 | The second recovery. Superseded by `wb-recover3`. | disposable |
| `wb-recover\` | 138 | The first recovery. Superseded. | disposable |
| `wb-replay\` | 244 | The first recovery's replay, plus the rendered pages (`recovered-render.html`, `product-render.html`, `dpr-render.html`) that were visually inspected. | disposable |
| `wb-horizon\` | 134 | The loop-detection horizon sweep (`loop_detection=True`). | disposable |
| `wb-horizon-seq\` | 121 | Its sequential control, run after a document audit found the first comparison unmatched. | disposable |
| `wb-h84-loop\`, `wb-h84-seq\` | 58 | The 84-keyframe pair that withdrew limitation 3. | disposable |
| `wb-review-2026-09-10\` | 288 | The two-reviewer round's replay of the real capture and its standalone finalize, plus their logs. **The source of the post-Stop timing and the rebuild duty-cycle table.** | KEEP until the retest |
| `wb-production-reconnect\` | 7 | The production-argv walk driven across a real retarget, which is what proves §4b end to end. | disposable |
| `review-q2\`, `review-q3\` | 6 | The last review round's two-session world and its reconnect-race runs. | disposable |
| `wb-allcart-soak8\` | 76 | The all-cartridge soak. | disposable |
| `wb-allcart-soak8b\` | 76 | Its rerun, after repairing the broken `python-bidi` install. | disposable |
| `wb-allcart-soak\` | 29 | The first, abandoned soak attempt. | disposable |
| `wb-production-walk\` | 7 | The production-argv walk's world and capture. | disposable |
| `wb-shutdown-time\` | 1 | Shutdown timing. | disposable |
| loose: `recover.json`, `recover2.json`, `recover3.json`, `recover.log`, `horizon-sweep.log` | <1 | The recovery runs' summaries. | disposable |

**1,563 MB in total, of which 403 MB is evidence** (the preserved field
artifact and the review replay the last round's measurements come from).
Nothing was deleted:
policy rule 14 requires explicit human approval and rule 15 says move
rather than delete, and these are already in the approved location.

The live world root under `tower/data/world_builder/` was never modified:
every recovery ran against a copy.

Scripts live in the session scratchpad under `%TEMP%\claude\...` and go with
the session: `production_walk.py` (the production-argv walk),
`horizon_sweep.py` (the loop-detection sweep) with `horizon_sweep_seq.py`,
`h84_loop.py` and `h84_seq.py` (its controls),
`production_walk_reconnect.py` (the reconnect walk), `swiftcheck.py` (a
brace balancer for a host that cannot compile Swift), and `swiftinit.py` — an
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

**Re-run at the end of round 17, on the Tower's own interpreter and
against the current tree** (the first run was on the wrong Python — see
§13). Same argv, same solver, same wearer's stop:

```
fed 260 frames in 22.4 s (11.6 fps); builder exited 0 after 25.7 s
keyframes 29  points 831  segments 4  posed 23
global_solve components 2  replaced 2  refused 2
placements: registered 2, refused 2      rebuilds 7

  [OK ] end_reason is stop            [OK ] lifecycle not interrupted
  [OK ] finalization complete         [OK ] placements from global solve
  [OK ] final solve solved            [OK ] something registered
  [OK ] geometry published            [OK ] lock released
  [OK ] no stray temp files
  [OK ] model_state is finalized      [OK ] geometry available on the wire
  [OK ] the phone has something to draw
  [OK ] the listing agrees
```

831 points against the first run's 835: the solver is not deterministic
across runs, and nothing here depends on the exact figure.

**The last four checks are new, and they are the point of re-running it.**
The harness used to call `_lifecycle` with hand-picked arguments, which
went stale twice as that function grew keyword arguments — and a check
that cannot run is not a check. It now takes the payload from
`WorldBuilderStatusProducer.snapshot`, which is what the phone actually
receives, so the projection, the drawable predicate and the recount are
all exercised by the production walk rather than only by fixtures. The
last one asks the saved-worlds listing about the same session and requires
it to agree — the picker row and the panel behind it, computed apart from
one disk.

### And the same argv across a reconnect

The field walk reconnected 55 seconds in, and no test in the suite runs the
production argv across a retarget. Re-run at the end of round 17 on the
Tower's own interpreter, this one passes seventeen checks:

```
sources: 29 entries, 29 resolved, 0 absent, 0 wrong
captures ['60a3fc1f...', '9d626ea0...']      keyframes 29  points 835

  [OK ] every source resolves         [OK ] no source is the wrong image
  [OK ] the ledger names both captures
  [OK ] the journal records the capture's own end
  ...and the thirteen above, including the four new ones
```

So the same walk was driven again with
the socket dropped at frame 130 and a successor brought up 0.3 s later:

```
-- disconnect after 130 frames
-- successor c6ffc2f922c34b1c8e01a4ac08bde965
fed 260 frames in 23.2s (11.2 fps); builder exited 0 after 26.7s
session_stopped payload: {'end_reason': 'stop', 'capture_end_reason': 'stop'}
sources: 29 entries, 29 resolved, 0 absent, 0 wrong; captures [both]

  [OK ] end_reason is stop            [OK ] lifecycle not interrupted
  [OK ] finalization complete         [OK ] placements from global solve
  [OK ] final solve solved            [OK ] something registered
  [OK ] geometry published            [OK ] lock released
  [OK ] no stray temp files           [OK ] every source resolves
  [OK ] no source is the wrong image  [OK ] the ledger names both captures
  [OK ] the journal records the capture's own end
```

Reverting the one line in §4b and running it again reproduces the field
defect exactly — `sources: 29 entries, 20 resolved, 9 absent`, one capture
named — so this walk is a control as well as a proof.

---

## 17. Verdict

**READY FOR PHYSICAL RETEST**, with one gate that is not a formality —
and with round 20's verdict on the record first. A dress-rehearsal
reviewer, driving a real Tower over the websocket, returned **NOT READY**:
a mid-walk reconnect split every walk into two worlds. That was correct
when it was written. Both mechanisms are fixed and re-verified with the
reviewer's own harness at all four timings (§13, round 20); the verdict
below is the one that stands after that, not instead of it.

Everything on the Tower side is verified on this machine: the root causes
are fixed with tests that fail against the old code, the failed field walk
is recovered by a supported command, a production-argv walk passes end to
end — twice, the second driven across a real reconnect — the suite is
green, and **twenty-two** independent review passes have been answered.

**Read that number as a warning, not a boast.** Every round found
something in the newest code, including **five in a row** that found defects
in the fixes written for the round before — three of them worse than the bug
they replaced.

What finally broke that cycle was not a better fix. It was a reviewer asking
why the same class of bug kept coming back, and finding that a world has one
manifest and it names whichever session built last. Four rounds had been
spent patching what that caused. **The lesson is the campaign's own, twice
over: a defect that survives repeated fixing is usually a symptom, and the
number of rounds it takes to ask "why does this keep happening" is the real
measure of a review process.**

**The gate is the Mac build.** Nothing iOS was compiled, and that is not a
theoretical risk: a reviewer found that `WorldTrajectoryReport.init` had
been missing two parameters its own decoder was passing, so **the branch
did not compile for several commits and nothing said so**. The iOS changes
since are substantial — a new 830-line state model (with 948 lines of tests beside it), a restructured Saved
Worlds flow, a rewritten render viewer. Treat `xcodebuild build` as a real
step that may fail, not a checkbox.

What would most likely disappoint on the day, in order:

1. **A Release build is installed, and the walk records nothing at all.**
   `sendStreamStart` is inside `#if DEBUG`, so no capture is ever opened
   and no world is created. This is first because it is binary, it is
   invisible until afterwards, and it cannot be checked from Windows.
2. **The wearer opens the world before the final solve finishes**, sees the
   shattered pre-solve build, and concludes it failed — 170 seconds before
   it succeeds. Measured at 179 s post-Stop on a 4-minute walk, with a
   frozen payload and a live "open it now" button throughout. This is the
   most likely way a *working* retest reads as a failed one, and it is a
   procedure step in §15 rather than a fix.
3. **The Swift does not build**, for something like the `init` defect
   above.
4. **The world is a point cloud, not a surface.** It is coherent, coloured
   and navigable, and it is still dots. The dense lane that would change
   that is written, measured, and unmerged.
5. **The phone stays offline for more than ~16 seconds.** Its reconnect
   budget is 5 tries over ~15.5 s (`TowerClient.swift`), then "reconnect
   given up — use Connect to retry". The Tower now holds the walk open for
   90 s; the wearer has to tap Connect inside that window or it finalises
   without them. Argued from the Swift, not run.
6. **The walk fragments anyway**, because the capture is poor rather than
   the solver is. 38.5% of the field capture was motion-blurred and 27.2%
   was more than 30% black, and nothing warns the wearer in the moment.
   `tx_seq` will at least say whether frames were lost or never sent.

The first and third need a Mac and a phone. The second is measured and
mitigated by the procedure, and reducing it further means either a new
live-path write or a retune that should be measured rather than guessed.
The fourth needs a decision about scope rather than more evidence. The
fifth is a phone-side budget read from the Swift and needs the phone to
confirm; the sixth needs the glasses.

**Two things that would have been on this list and are now refused instead
of suffered**, both found in the last two rounds and both silent before:

- **The app is not rebuilt from this branch.** Both World Builder contract
  ids moved to `/2026-09-10` and both are compared for equality, so an
  older build now shows a refusal sentence and an empty Saved Worlds
  rather than decoding the payload and drawing the wrong word. Loud, and
  §15 leads with it.
- **The Tower is started on the wrong Python.** A bare `python` on this
  host has no `pycolmap`; such a Tower answers every request, records
  every keyframe, and reconstructs nothing. Every other pre-flight verdict
  passes on it. `sfm_backend_importable` now fails by name and prints
  `sys.executable`. I lost a working session to this before adding it.
