# World Builder — tracking recovery, drift control, retrieval width

**Branch:** `world-builder/tracking-recovery-v1`
**Parent:** `world-builder/fragment-registration-v1` @ `e847339`
**Date:** 2026-09-01
**Full report:** `tower/docs/superpowers/research/2026-09-01-drift-not-fragmentation.md`

This is the current-state handoff. The report is the evidence; this is
what a person needs to know before touching the World Builder next.

---

## 1. The headline, and the correction to the working theory

The working theory was that the World Builder fragments because the
tracker restarts too often. It does restart too often, and that is
addressed. **It is not the reason a walk does not become a world.**

The forward-only pose chain drifts. On a perfect synthetic strafe with
nothing refused and nothing blurred, rotation error grows 0.95° at six
keyframes to 33.98° at forty and the reconstruction contracts by a factor
of three. On the real 2026-09-01 walk, the one segment that survived 170
keyframes has camera centres reaching 8 × 10¹⁰ and points reaching
2.9 × 10¹³. The largest coherent piece of geometry in the corpus was not
coherent, and cross-segment registration was refusing to place it
**correctly**.

So: fragmentation is the symptom; drift is what made the fragments
unmergeable. Anything that reduces segment count without controlling
drift makes the world tidier and less true.

---

## 2. What is now in the code

**`tower/world_builder/bundle.py`** (new). Sparse Levenberg-Marquardt over
poses and landmarks with the Schur complement, numpy and OpenCV only, run
over a sliding window on the **keyframe** path. `BUNDLE_WINDOW = 12`
cameras, oldest 2 fixed, every 3 solved keyframes, 4 iterations.

**`backends/classical.py`.** `_Chain.broken` is no longer a one-way latch;
references are the last keyframes that HAVE poses; a refused keyframe
never becomes a reference. `MAX_RECOVERY_KEYFRAMES` bounds how many
consecutive refusals are tolerated and **is 1** — see §4.

**`scripts/world_registration.py`.** `MAX_KEYFRAMES_PER_SEGMENT_FOR_MATCHING`
is now a floor rather than a cap; `match_budget` splits a
`MAX_MATCH_FRAME_PAIRS = 256` product budget in proportion to the two
segments' lengths. No gate was touched.

**`scripts/world_coherence_report.py`** (new). Reads a built world cold and
reports the dominant connected component AND the reprojection of the
published support rows. It refuses to report one without the other, on
purpose.

---

## 3. Results

2026-09-01 long-loop walk, physical era → this branch:

| | physical era | this branch |
|---|---|---|
| fragments | 18 (no placements written at all) | **9** |
| dominant component | 6,190 pts / 20.4% (one diverged segment) | **18,817 pts / 74.9%** |
| dominant component, in keyframes | 170 of 434 (a diverged segment) | **156 of 434, coherent** |
| admitted pairs | 0 | **15** |
| cycles verified | — | **6** |
| published reprojection p99 | 4.73 px | **2.51 px** |
| rows over the 3 px gate | 2.54% | **0.24%** |
| rows whose landmark is behind its camera | 5 | **0** |

**Reprojection improves on every walk in the five-walk corpus**, and the
over-gate fraction by between 4× and 20×. The dense 08-29 walk goes from
58.0% of its geometry in one frame to **84.1%**.

**What it costs**, because this is the number to argue about: the branch
publishes fewer points. 25,131 against 30,382 on this walk, and 6,762
against 9,145 on the 2026-08-29 normal walk, which is the worst case.
Those are landmarks the adjustment moved into the regime where their
depth is set by pixel noise, and they are counted in the manifest under
`low_parallax`. On every walk the dominant component still covers the
same keyframes or more.

**What it does NOT fix: the 12 unreconstructed areas.** The phone
reported 12 areas seen but not reconstructed; that is the 12 of 30
segments which got keyframes and triangulated nothing, and it is still
12. A bundle adjustment cannot help a segment with no geometry to
adjust. The mechanism that would — holding the anchor and retrying a
refused seed pair on the next keyframe, which gives it a WIDER baseline —
is built, tested, and inert at `MAX_RECOVERY_KEYFRAMES = 1`. At 8 it
takes that walk from 12 unreconstructed segments to 8, and §4 is why we
will not have it at 8.

---

## 4. The thing to be careful about: `MAX_RECOVERY_KEYFRAMES`

That constant is exactly the largest reference gap a solve may be
admitted at, and **no acceptance gate in the backend is a function of the
gap**. Over repeating texture — an ordinary room — a gap above 1
publishes roughly one keyframe of motion however far the camera walked.
Measured, continuous walk, no teleport: at gap 8 the camera moved
1.500 m and the pose reported 0.001 m, with 169 PnP inliers, 0.14° of
rotation error, and support reprojecting at 0.22 px median.

Every instrument this pipeline owns says that pose is excellent.

`tests/test_world_builder_recovery_safety.py` carries the full tables.
**Do not raise this constant without an instrument that checks the
displacement a recovered pose implies against the number of keyframes it
skipped.** A better matcher will not help: appearance is what lies here.

---

## 5. Shutdown risk — REAL, unfixed, and characterised

`CaptureWorkerSupervisor.shutdown` gives each worker
`DEFAULT_GRACE_SECONDS = 10.0`. Registration now takes **14–66 s** on the
corpus's substantial walks (it took 1.6–34 s before). A Tower shutdown
landing inside that window will kill registration.

What that costs, precisely: `register()` writes nothing until it
completes and `write_placements` is its only write, so a kill loses
`placements.json` and **never** the reconstruction. The world is left
exactly as the physical era left it — built, unplaced.

Why it is not fixed here: on Windows `Popen.terminate()` is
`TerminateProcess`, which is not catchable, so no handler in the child
can help. The fix has to be a longer grace for this worker specifically,
or a placement checkpoint, and the first is a product decision about how
long a Stop may take. **Left for a follow-up, deliberately.**

---

## 6. What the next physical test should be

Run `main` as usual for CV Lab and Object Memory. For World Builder,
check out `world-builder/tracking-recovery-v1` on the Tower.

1. **Repeat the 2026-09-01 walk**, same room, same route, ~2 minutes,
   deliberate overlap, return toward the start. That gives a like-for-like
   physical comparison against the best dataset we have.
2. **Then walk a long straight corridor** for 60–90 seconds without
   returning. Every walk in the corpus is a room; the drift measurement
   says the risk concentrates in long unlooped chains, and nothing in the
   corpus tests one.
3. **Do not stop the Tower for at least 90 seconds after ending a walk.**
   See §5.
4. Expect on the phone: fewer fragments, and a dominant fragment holding
   substantially more of the room. Report the fragment count and which
   fragment is largest — that is the number this branch moved.

---

## 7. Temporary resources this run created

Recorded per the repository's filesystem policy. **Nothing was deleted**,
and nothing was written to `C:\`, to the user home, or into
`tower/data/`.

**Git worktrees** (both created with an explicit destination):

- `C:\Users\tvllo\Projects\Glasses-worktrees\wb-track` — this branch. Keep
  while the branch is in review.
- `C:\Users\tvllo\Projects\Glasses-worktrees\wb-phys` — detached at
  `768cecf`, used to measure the physical-era baseline. Disposable;
  `git worktree remove` when the numbers in §3 are no longer being
  questioned.

**Scratch** — all under `C:\Users\tvllo\Projects\Glasses-scratch\wbt\`:
replayed worlds (`b_*`, `sw/`, `phys/`, `cb0`–`cb5`), benchmark result
files (`cb_*.json`), pytest basetemps (`pt*`), and the experiment drivers
(`sweep.py`, `grid*.sh`, `regcompare.py`, `physera.py`, `table.py`,
`drift.py`, `localmap_proto.py`, `ba_probe.py`, `ba2.py`). Roughly 3 GB,
all reproducible.

**One file in the canonical checkout that is NOT ours to keep or remove:**
`C:\Users\tvllo\Projects\Glasses\orb_vocab_stella.fbow`, untracked. It
came from this run's relocalization research, which downloaded
stella_vslam's ORB vocabulary to check its provenance. **Do not vendor
it.** That research established, by parsing the binaries rather than
reading the READMEs, that stella's `orb_vocab.fbow` is byte-identical to
fbow's, which is Mur-Artal's ORB-SLAM2 vocabulary data redistributed
under a third party's copyright — the same class of derivation that got
OpenVSLAM withdrawn in 2020. **No independently trained, permissively
licensed pretrained ORB vocabulary exists.** If retrieval is ever built
here, train a vocabulary from our own footage: measured, K=4096 from
60k of our own descriptors builds in 7.6 s, quantizes a keyframe in
18 ms, and recovers most of the full vocabulary's discriminative margin.
