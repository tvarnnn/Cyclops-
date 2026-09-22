# Post-MVP hardening, deferred optimization and red-test repair

**Branch:** `hardening/post-mvp-runtime-v1`
**Worktree:** `C:\Users\tvllo\Projects\Glasses-worktrees\post-mvp-hardening`
**Date:** 2026-08-28
**Status:** complete. Green, pushed, not merged. `ios/` untouched: 0 files.

---

## 1. Starting point

| | |
|---|---|
| Branch created from | `35a2418` (`main`, = `origin/main` at the time) |
| Rebased onto | `768cecf` (clean) |
| `ios/` files changed | **0** |

### 1.1 The branch-reconciliation run, and the collision

The brief said to wait for a concurrent branch-reconciliation run and not to
race it. That run had already merged and pushed `main` @ `35a2418` before this
lane started, and the working tree was clean.

**It had not finished.** A second Claude session (`tower-dc`) was still active
and, fifteen minutes in, began editing `main`'s working tree — specifically the
eight `--root` scripts that are this lane's item (A). Both sessions were also
running the full Tower suite concurrently on a 20-core host.

Rather than guess, the two sessions were put in direct contact and the scope
was split explicitly:

* `tower-dc` owns `main`, item (A), the `.gitignore` conflict markers, and
  branch deletion/archive tags.
* This lane owns items (B) and (C) and all deferred runtime work, in its own
  worktree, and does not touch `main`.

That exchange also produced three corrections in each direction; they are
recorded in §2 and §5 rather than being smoothed away, because each one
changed what the other session published.

### 1.2 A defect in the "canonical" main

`main` @ `35a2418` — the commit the brief names as the proven canonical base —
**contained committed merge-conflict markers** in `.gitignore` at lines 3, 22
and 28 (`<<<<<<< HEAD`, `=======`, `>>>>>>> 2b7fb43`). The unified merge
committed them. A repo-wide scan at that commit finds them in that file only.

Found by this lane, fixed by `tower-dc` in `32c96c9` as a union of both sides.
Recorded here because the brief's §17 asks for a conflict-marker scan at the
END of the run, and running it at the START is what caught this.

## 2. Environment verification

Verified from the worktree, with `tower.__file__` asserted to resolve there:

```
Python            3.12.5 AMD64
FastAPI           0.141.1
OpenCV            5.0.0
Torch             2.13.0+cu132
CUDA available    True
CUDA version      13.2
GPU               NVIDIA GeForce RTX 5070
NumPy             2.5.2
pip check         No broken requirements found.
```

Matches the environment the previous lane recorded, exactly. Independently
reproduced by `tower-dc` on the main checkout.

### 2.1 The code-resolution trap, form 4

The previous lane documented three forms. There is a fourth on this host, and
it is the most dangerous because it is silent and permanent:

**The venv contains an editable install that hard-codes the MAIN checkout.**
`__editable___glasses_tower_0_1_0_finder.py` maps `tower` ->
`C:\Users\tvllo\Projects\Glasses\tower\tower`. Any worktree that does not get
itself onto `sys.path` first resolves the *other* tree's code, with no error.

It is survivable only because setuptools *appends* the finder to
`sys.meta_path`, so the ordinary `sys.path` finder wins when the worktree is on
`sys.path`. Verified empirically rather than assumed: `tests/__init__.py`
exists, so pytest prepends the worktree root and a suite run from the worktree
tests the worktree. Confirmed by a throwaway test that printed
`tower.__file__`, `sys.path[0]` and the interpreter from inside a real pytest
run.

Every scratch script in this lane does
`sys.path.insert(0, <worktree>/tower)` and then asserts, because CPython puts
the *script's own* directory on `sys.path[0]` and never the cwd.

### 2.2 Two hosts, one basetemp — a cross-session hazard

Two concurrent pytest runs share `%TEMP%\pytest-of-tvllo` and its
`pytest-current` symlink. This lane's first full run died at 91% with

```
PermissionError: [WinError 5] Access is denied:
'C:\Users\tvllo\AppData\Local\Temp\pytest-of-tvllo\pytest-current'
```

which is not a test failure and reads exactly like one. `tower-dc`
independently lost a whole run (138 failed / 27 errors) to the *opposite*
mistake — a `--basetemp` deep enough to breach Windows MAX_PATH:

```
FileNotFoundError: [WinError 206] The filename or extension is too long
```

**Both failure modes produce large red suites that are entirely environmental.**
Every run in this lane therefore uses a dedicated, SHORT basetemp under
`Glasses-scratch\`.

## 3. Baseline

### 3.1 Inherited baseline (real tree, from `tower-dc` @ `768cecf`)

```
2 failed, 2249 passed, 34 skipped
```

against the brief's recorded `3 failed, 2248 passed, 34 skipped`. The delta is
exactly item (A) going green. The two remaining reds are (B) and (C).

### 3.2 This lane's hermetic baseline (worktree, no `data/`)

Zero test failures through 91% of the suite before the basetemp collision in
§2.2 ended the run.

**That green is not evidence, and the reason is the most useful thing in this
section.** `data/` is gitignored and therefore does not exist in a fresh
worktree, and both remaining reds are invisible without it:

* `tests/test_world_registration.py:598` has `REAL_ROOT = Path("data/world_builder")`
  — cwd-relative — and the fixture *skips* when the world is absent. Item (C)
  does not fail in a clean checkout; it silently disappears.
* `tower/config.py` derives `DEFAULT_OBSERVATION_ROOT` from `TOWER_ROOT`, which
  is derived from `__file__`. In a fresh worktree it points at a directory that
  does not exist, so item (B) sees 0 observations and passes.

A clean checkout reports these two defects as PASS and SKIP. That is worth
knowing before anyone trusts a green run from a fresh tree.

### 3.3 Controlled reproduction

Both reds were made to reproduce deterministically in the worktree from a
minimal, declared fixture:

| Fixture | Contents | Why |
|---|---|---|
| `data/world_builder/` | one world (`3dd986b1…`, 12 MB) + `intrinsics/` | item (C) needs the pinned real walk |
| `data/object_memory/` | **64 synthetic records**, generated | item (B) needs a non-empty store |

The observation fixture is **synthetic on purpose**. The defect is "an
unconfigured Tower under test reads whatever store sits at the repository-local
default root", so any non-empty store demonstrates it. It reproduces the
reported failure exactly —

```
assert response.json()["observation_count"] == 0
E   assert 64 == 0
```

— without copying a single real observation. A privacy defect that can be
worked on without touching the data it is about is a better-shaped defect, and
this corrected `tower-dc`'s published claim that (B) only reproduces where real
capture data has accumulated.

Item (C) reproduces in ~3 s:

```
FAILED tests/test_world_registration.py::TestTheRealWalk::
       test_pairs_whose_directions_disagree_are_never_admitted
E   AssertionError: expected some pair whose two solves disagree
E   assert []
```

## 4. The three red tests

### 4.1 Failure A — artifact root safety — **REASSIGNED, not solved here**

Owned and landed by `tower-dc` on `main` as `768cecf`. Not this lane's work and
not claimed as such.

This lane's analysis, produced before the split and passed to them, was that
the eight scripts do **not** share one `--root` meaning:

| Script | `--root` is | Evidence |
|---|---|---|
| `object_memory_session.py` | **written** | writes observations |
| `calibrate_charuco.py` | **written** | `IntrinsicsStore(args.root).save(...)` |
| `world_registration.py` | **written** under `--write` | `WorldStore(args.root)` |
| `object_query.py` | read; deletes only under `--purge-all` | `ObservationStore(args.root, …)` |
| `capture_corpus_benchmark.py` | **read-only input** | `benchmark_corpus(args.root, …)` prints a report |
| `research/native_eval/profile_split.py` | **read-only input** | profiles an existing world root |
| `research/native_eval/registration_scale.py` | **read-only input** | same |
| `research/native_eval/residuals_micro.py` | **read-only input** | same |

An input root that must already exist wants an "existing directory" converter,
not the drive-root refusal; and a test that demands every `--root` route
through *one of two named converters* is strictly stronger than today's rule,
because then neither kind can be forgotten silently.

`tower-dc` deliberately did not act on this, on the grounds that its brief was
minimal additive reconciliation with an explicit "no unrelated architectural
changes" constraint, and that an independent reviewer classed it the same way.
That is a defensible call and this document does not overturn it. **The
refinement is left open, owned by this lane**, to be taken as a separate,
explicitly-scoped change or explicitly declined — see §9.

Note for whoever takes it: the AST scanner in
`tests/test_artifact_paths.py` keys on the literal `"--root"`, so a second
converter must be taught to the test in the same commit or the test will
silently accept either.

### 4.2 Failure B — Object Memory default store isolation

*(pending — specialist agent running)*

Established mechanism, to be confirmed or refuted by the agent:

`tower/config.py` sets `TOWER_ROOT = Path(__file__).resolve().parent.parent`
and `DEFAULT_OBSERVATION_ROOT = str(TOWER_ROOT / "data" / "object_memory")`.
`_observation_root(enabled)` returns `TOWER_OBSERVATION_ROOT or
DEFAULT_OBSERVATION_ROOT` whenever the cartridge is enabled, which is the
default. So a test that *deletes* `TOWER_OBSERVATION_ROOT` does not get an
isolated store — it gets the developer's own, at a repository-local path.

This is a test-isolation defect and a privacy defect in the same line of code.

### 4.3 Failure C — World Registration real-walk assumption

*(pending — specialist agent running)*

The test's universal half — disagreeing pairs are never admitted — is never
reached, because its *existential* half (`assert disagreeing`) is what fails.

The question the agent must settle is whether the disagreeing pairs are
**absent** or merely **non-finite**: `scripts/world_registration.py` emits
`reciprocity: None` when `math.isfinite(...)` is false, and the test's filter
skips `None`. A pipeline that began producing NaN reciprocity for exactly those
pairs would empty the list *and* look like an improvement.

## 5. Deferred findings — reproduction status

### 5.1 Stub worlds — **CONFIRMED, and worse-shaped than reported**

The previous lane recorded 83 of 120 corpus worlds as empty stubs. Current
census of the real root: **123 worlds, 86 with `session_ids: []`, 37 with
sessions.** By creation date the stubs cluster 31 on 08-25, 51 on 08-26, 1 on
08-27, 3 on 08-28 — far more than the number of real walks.

**The suite is the writer, and it is not merely polluting an existing corpus —
it creates the root.** Two independent reproductions:

* `tower-dc`, one clean full-suite run against the real tree: 123 -> 124, count
  stable at 9%, 37% and 59% and incremented only at the end. The new directory
  holds `world.json` and no `sessions/`.
* This lane, in a worktree that had **no `data/` directory at all** when the run
  started: two stub worlds minted into a `data/world_builder/worlds` the run
  created itself.

So the defect is "the suite writes to the repository-local default world root",
of which the user's corpus is simply the instance that happens to be theirs.

The minting itself is **deliberate product behaviour** —
`scripts/world_build_session.py` creates the world *before* waiting for the
first frame, and says why: "a Tower whose phone has connected but not yet sent
a frame reports a world that exists and is empty rather than no world at all."

**But that is not why the stubs exist, and the real cause is a production
defect.** `follow_capture` is a **generator function**, so its precondition

```python
if not directory.exists():
    raise SystemExit(f"no capture directory at {directory}")
```

did not execute when `main()` *called* it — only when something first
*advanced* it, which happens after `engine.create_world(args.name)`. A session
pointed at a capture directory that does not exist therefore **minted a world,
then exited nonzero, and the world stayed.** Reproduced directly on the product
path, with no test involved:

```
$ world_build_session.py --root <tmp> --follow-capture <absent>
no capture directory at <absent>
$ find <tmp> -name world.json | wc -l
1
```

So every failed follow — a typo, a capture directory that vanished, a race
where the directory does not exist yet — left a permanent empty world in the
wearer's store, and nothing ever collected them.

**Fixed in `d796628`.** The check moved into a plain function that validates
and then *returns* the generator, which is the standard way to make a
generator's preconditions eager. The deliberate mint-before-first-frame
behaviour is preserved for sessions that can start.

The test-side leak was localised in the same pass, by per-file bisection:
`tests/test_world_builder_follow_cli.py::TestFollowCapture::test_a_missing_capture_directory_exits_nonzero`
ran the CLI with **no `--root`**, so it defaulted to a *relative*
`data/world_builder` resolved against pytest's cwd — one stub per full-suite
run, which is exactly the 123 -> 124 delta. It now passes `--root`.

Verification: 56 passed across the five world-builder CLI suites, and the world
count in the default root is **unchanged by the run** (6 -> 6) where it
previously incremented. Product path re-checked: 1 -> 0 worlds minted.

**Two mechanisms, not one — and they need different fixes.** Agent B's
correction to this lane's framing is worth recording: the observation-root
defect is an *absolute, file-anchored* default in `config.py`, while the
world-root defect is a *relative, cwd-anchored* default in a script's argparse.
They present as the same symptom ("tests inherit repository artifact roots")
and a single sweep would have got one of them wrong.

**No existing world was deleted or moved.** The 86 stubs already on disk remain
the user's to decide about; this change only stops new ones being created.

### 5.1a `OMP_WAIT_POLICY` and the thread leak are INDEPENDENT

Worth settling early, because the two most-promising deferred items both name
OpenMP and it would be easy to assume one fixes the other. Re-running the arm
A / arm B isolation under `OMP_WAIT_POLICY=passive` gives **identical** thread
growth:

```
arm A: +152 OS threads for 8 inferences (+19.00/inference, on 8 threads)
arm B:  +19 OS threads for 8 inferences (on 1 thread)
```

`passive` changes how OpenMP worker threads *wait* (spin vs sleep), not how
many are *created*. So it cannot help the leak, and the leak fix cannot be
credited to it. They are separate levers on separate problems and are judged
separately below.

### 5.2 Capture retention — **CONFIRMED**

No pruner exists for `capture_root`. Object Memory and Document Memory both
have retention windows and `prune_expired`; the capture recorder, whose copy is
the **unredacted** one, has neither.

The recorder is honest about it rather than silent — `capture.py:396` writes
`"redaction": "none"` into the manifest, so the claim on disk is accurate. The
gap is lifetime, not truthfulness.

Current corpus on this host: **35 captures, 407 MB, spanning 2026-08-24 01:22
to 2026-08-27 01:04** — three days. Largest single capture 50 MB. Nothing in
`tower/` deletes a capture: the four deletion sites are Document Memory's page
images, World Builder's derived trees, and one partial-directory cleanup in
`capture.py`. None of them is a retention policy over the photographs.

So the asymmetry the Mac lane flagged holds and is now quantified: **derived
records expire at 30 days; the imagery they were derived from expires never,
and the imagery is the unredacted copy.**

This is a data-lifetime decision for a human, not an optimization. This lane
has no deletion permission and proposes no deletion of existing captures.

### 5.3 Two JSONL stores — **one CONFIRMED, one NOT REPRODUCIBLE**

The previous report's §15 lists this as a single item: "Two JSONL stores rewrite
per record and re-parse per request", naming `document_memory/store.py` and
`object_memory/store.py`. **Half of that does not reproduce.**

**Object Memory — CONFIRMED by inspection.**

* `update_sighting` reads every raw record and calls
  `_rewrite_locked(raw_records)` — a full rewrite per sighting update.
* `all_observations` re-reads and re-parses the whole file per call; no cache.

**Document Memory — NOT REPRODUCIBLE as stated.** `DocumentStore.append` is

```python
def append(self, document: DocumentObservation) -> None:
    with ...:
        append_jsonl(self._path, document.to_json_dict())
```

and `tower/storage.py: append_jsonl` is a genuine append: it opens the file to
read the final byte, then opens in `"a"` mode and writes one line. Two opens
and a one-byte tail read — not a rewrite. Whole-file rewrites in that module
occur **only** in `_rewrite_keeping`, reached from `prune_expired` and `purge`,
which are deletion paths and are expected to be O(n).

So the "journal rewrite per record" characterisation applies to Object Memory's
*sighting update* path and not to Document Memory's *append* path. The two
stores were grouped as one finding and they are not the same finding.

At today's scale (64 records) neither is a product problem. Whether Object
Memory's rewrite threatens product behaviour at reachable scale is a
measurement, not an argument, and has not been taken yet.

### 5.4 Capture recorder I/O — **CONFIRMED shape**

`tower/capture.py` writes each image to a temp path, `flush()`, `os.fsync()`,
renames, then appends the journal line — "Image first with fsync, journal line
second". The ordering is a deliberate durability guarantee, so any batching
must preserve it rather than trade it away.

### 5.5 Live-session torch thread pool — **REPRODUCED on current main**

Measured in-process through the real `/scene/start` and `/scene/stop` routes
against a real `create_app()`, with `tower.__file__` asserted to resolve in the
worktree, and every cycle confirmed to reach `lifecycle.state == "running"`
before the Stop — so these are whole cycles, not abandoned loads.

```
baseline    threads=  3 os= 29 rss=   643.0MB handles=401
cycle  threads    os    rss_MB  handles   d_os    d_rss   state
    1        3    48     701.8      440     19     58.7   running
    2        3    67     709.7      478     19      8.0   running
    3        3    86     717.6      516     19      7.8   running
    4        3   105     726.0      554     19      8.4   running
    5        3   124     734.4      592     19      8.4   running
    6        3   143     742.3      630     19      7.9   running
    7        3   162     750.8      668     19      8.5   running
    8        3   181     758.8      706     19      8.0   running
    9        3   200     766.8      744     19      8.0   running
   10        3   219     774.8      782     19      8.0   running
   11        3   238     783.1      820     19      8.3   running
   12        3   257     790.6      858     19      7.5   running
```

**Exactly +19 OS threads per cycle, on all twelve cycles, with no plateau.**
That reproduces the previous lane's +19 threads and +8.1 MB per cycle
*precisely*. RSS is +8.0 MB/cycle in steady state; cycle 1's +58.7 MB is the
one-time model load, not the leak. Handles grow +38/cycle.

(The harness's own "first half / second half" summary line prints +22.17 and
+15.83, which looks like a decaying trend and is not one — it is an off-by-one
in how that line indexes the midpoint. The per-cycle `d_os` column is the
measurement, and it is 19 twelve times out of twelve.)

**Extrapolated, this is a real failure mode rather than untidiness.** Twelve
Start/Stop cycles cost 228 threads; fifty would cost ~950 on top of the
baseline, in one process, on a device a wearer is expected to start and stop
repeatedly through a day.

**The most useful number here is the one that does not move.**
`threading.active_count()` stays at **3** across every cycle while the OS
thread count goes 29 -> 80. The growth is entirely in **native** threads —
OpenMP/torch intra-op pools created per worker thread and never reclaimed when
that thread dies. It is therefore invisible to Python-level thread accounting,
which is why a `gc.get_objects()`-style audit (the previous lane ran one, and
found nothing) cannot see it either. Handles grow in lockstep (+34/cycle).

#### The mechanism, isolated

Inference alone, two arms, same process, same total work:

```
torch.get_num_threads() = 20      (20-core host)
baseline after one warm-up inference: 45 OS threads

ARM A -- one fresh thread per inference
  after 1 fresh threads:   64  (+19)
  after 2 fresh threads:   83  (+38)
  ...
  after 8 fresh threads:  197  (+152)

ARM B -- one reused thread, 8 inferences
  after 8 inferences on ONE thread:  216  (+19)
```

**One OpenMP team — 19 threads, exactly `cores - 1` — is created per NEW
thread that calls into ATen, and is not reclaimed when that thread exits.**
Every arm-A thread was `join()`ed and its 19 threads still persisted.

That settles three things at once:

* the growth is per-**thread**, not per-session, per-model or per-engine;
* reuse collapses N teams into one, so the fix is thread reuse and nothing
  else;
* it is not a torch *leak* in any repairable sense — it is OpenMP teams being
  thread-local and outliving their creator. Nothing in `torch` API terms will
  return them.

It also explains why the previous lane's `gc.get_objects()` audit found
nothing: there is no Python object to find.

#### The entangled race, reproduced

`LiveSession._begin_session_locked` creates a **new `threading.Thread` per
session**; `stop()` holds `_lifecycle` for all four steps while `start()` takes
only `_condition`; and step 4's `_release_engine` does
`engine, self._engine = self._engine, None` — reading the CURRENT engine rather
than the one `stop()` captured.

Driven deterministically with a Scene session whose `_on_pause` blocks
(Document Memory's real `_on_pause` is `engine.flush()`, an OCR pass measured
at 1.19 s a page, so this window is **seconds** wide in the product):

```
session 1 running, engine #1
stop() is inside its flush (state is already STOPPED)
session 2 running, engine #2

  engine1 (#1) released : False   <- LEAKED, it was the one being stopped
  engine2 (#2) released : True    <- the NEXT session's engine, torn down
  session.state         : running
  session._engine       : None
```

**Three failures in one window**, which is worse than the integration report's
description of it: the next session's engine is torn down, the stopping
session's engine leaks, and the session then reports `running` with no engine
behind it — so every frame it is offered is dropped, and no Stop will fix it
because `stop()` sees an engine of `None`.

This is integration finding 13, now reproduced rather than read.

#### Finding 15 also reproduced, and it is a privacy defect as well as a leak

Same file, same method's ordering. `stop()` clears `_stream_owners` ("a manual
stop disowns the stream") and `_begin_session_locked` clears it again ("a fresh
session owns nothing until a stream claims it"). Neither re-adopts a stream
that is **still open**:

```
phone connected      -> state=running owners={'connection-token-A'}
after HTTP stop      -> state=stopped owners=set()
after HTTP start     -> state=running owners=set()
phone disconnected   -> state=running owners=set()
```

After a Stop→Start on the HTTP routes the session is owned by nobody. It keeps
consuming frames and holding its model, and **the phone disconnecting can no
longer stop it.** Only an explicit HTTP Stop can — which is precisely the state
a wearer cannot reach by putting the glasses down.

For Scene Understanding, whose whole subject is detecting **people**, "keeps
observing after the wearer disconnects, and cannot be stopped by disconnecting"
is a privacy claim broken, not only a resource one.



`LiveSession._begin_session_locked` creates a **new `threading.Thread` per
session**, and torch inference runs on that thread (`SceneLive._consume` ->
`orientation.py`, which imports torch and runs `torch.inference_mode()`). A new
OpenMP team per worker thread, never reclaimed when the thread dies, is
consistent with the reported +19 threads per Start/Stop cycle.

**The entangled race is real and was read directly (not yet reproduced by test).** `stop()` holds
`_lifecycle` for all four steps; `start()` takes only `_condition`. A `start()`
landing between step 1 and step 4 stands up session 2 — and step 4 then calls
`self._release_engine`, which does `engine, self._engine = self._engine, None`,
reading the CURRENT engine rather than the one `stop()` captured. So the stop
releases **session 2's** engine. That is integration finding 13, confirmed by
reading, not yet by test.

## 6. Changes implemented

Rebased onto `origin/main` @ `768cecf` cleanly. Three commits, each with its
regression test proven RED against the defect first.

| Commit | What |
|---|---|
| `219df0b` | `fix(world-builder)`: a session that cannot start no longer mints a world |
| `d21c1de` | `fix(tests)`: the suite no longer reads the checkout's own object memory |
| `c6f6681` | `test(world-registration)`: pin the safety property, not the corpus accident |
| `a415562` | `fix(live-session)`: a Stop releases the engine it captured, not the current one |
| `a7546f7` | `perf(live-session)`: reuse the worker thread, because an OpenMP team outlives it |
| `1ded64d` | `fix(tests)`: the reviewer's findings — two guards proved less than they claimed |

Red-test gate after the rebase:

```
tests/test_artifact_paths.py  tests/test_object_memory_lifecycle.py
tests/test_config.py          tests/test_world_registration.py
tests/test_world_builder_follow_cli.py
tests/test_unified_lifecycle_regressions.py
    132 passed
```

All three originally-red tests are green, and failure (A) is green by way of
`tower-dc`'s work on `main` rather than anything done here.

## 6a. The reviewer's findings, and what they cost

An independent reviewer attacked the three red-test fixes **by mutation** rather
than by reading, and found two places where the suite had quietly stopped
proving what its commit messages claimed. Both are the kind of thing a reading
review does not catch.

**HIGH — the suite could no longer see the product's own default.** The autouse
fixture that stops every test reading a real wearer's store also made the thing
it protects unobservable. An `_observation_root()` mutated to ignore
`DEFAULT_OBSERVATION_ROOT` **entirely** passed the whole suite at
`2233 passed, 58 skipped` — byte-identical to baseline. Both new assertions
were negative ("the root is outside the checkout") and nothing asserted the
positive, so the 2026-08-26 reversal had become unprovable.

Two further weaknesses in the same tests, both demonstrated:
`TOWER_ROOT.resolve() not in root.parents` **admits `TOWER_ROOT` itself**
(`Path.parents` excludes the path), and the containment check was
**cwd-dependent for a relative default** — from `Projects\` or
`Projects\Glasses` a relative `data/object_memory` passed. That is the same
cwd-anchored trap that put a stub world in the real corpus.

**MEDIUM — and this one was against my own commit message.** `c6f6681` claimed
its new existential test was driven RED by weakening the gate. **It is not
reproducible for a reciprocity weakening.** `admit()` evaluates finite scale,
cameras, span/depth, *then* reciprocity — and all three disagreeing pairs are
refused before reciprocity is reached: (1,50) and (12,46) on `cameras` (2 < 3),
(5,6) on `span_over_depth` (0.043 < 0.09). With `max_reciprocity_error = 10.0`,
i.e. the clause switched **off**, every real-walk test stayed green and only
three synthetic ones reddened.

The fix turned out better than the thing it replaced. Relaxing `min_cameras`
and `min_span_over_depth` to their floor moves reciprocity to the front of the
queue — **and needs no monkeypatch at all**, because `pair_is_hopeless` reads
`thresholds.min_span_over_depth` itself, so dropping it to 0.0 neutralises the
pre-filter through the gate's own configuration. The walk then refuses those
pairs *on reciprocity*, in its own words:

```
(1, 50)  0.89440  "the two directions disagree on scale by 1.12x"
(5,  6)  0.70716  "... by 1.41x"
(12,46)  1.35588  "... by 1.36x"
```

Relaxing changes no verdict — admitted set, reference segment and
`points_registered` identical to the shipped run — which is now asserted rather
than assumed. **Re-running the mutation reddens five tests where it previously
reddened three, and the two new ones are the real-corpus tests.**

The vacuity is now stated rather than implied: on the shipped thresholds the
universal loop's body runs **zero** times, because the only two finite
reciprocities are 1.0389 and 0.9558.

Also fixed from the same report: `object_memory_session.py: loose_frames` is
the same generator shape as the `follow_capture` defect, latent only because
`ObservationStore.__init__` and `engine.load()` happen to create nothing; and
`test_object_memory_cli.py` spawned that CLI with **no `--root`** — a
subprocess, out of reach of any fixture, defaulting to the developer's real
store.

Attacks the reviewer tried and **could not** break, recorded because a failed
attack is evidence: the `follow_capture` startup race (a late-appearing
directory already failed before the fix; the live path has no race at all,
because `observer.start()` creates the directory before the child's argv is
built); import-time binding of the patched constant anywhere in `tower/`;
symlink and junction evasion of the containment check; and the
`_segment_with_span` removal (the surviving definition was already the one in
effect, and no call site reads `.points`).

## 6b. Judge 1 on the live-session worker

Driven to **1,200 cycles**, far past this lane's 12:

| generations | OS threads | RSS | handles |
|---|---|---|---|
| 0 | 45 | 552 MB | 335 |
| 100 | 1,945 | 1,245 MB | 4,155 |
| 500 | 9,545 | 3,987 MB | 19,355 |
| 1,200 | **22,845** | **8,778 MB** | 45,955 |

`+19.0` threads, `+6.85` MB, `+38` handles per cycle. No cliff, no exception,
no degradation of thread creation even at 22,845 threads.

**Two costs it looked for and could NOT substantiate — recorded because they
are exactly what a reader would assume:**

* **Idle CPU: zero.** Its first pass measured 2.188 CPU-seconds over a 3 s idle
  window at 805 threads and it nearly published "0.73 cores burned forever". A
  controlled re-measurement (1 s settle for `KMP_BLOCKTIME`, then a 4 s window)
  gives **0.000 s of CPU at 45, 140, 235, 425, 805 and 1,565 threads** — and
  the reuse arm showed the same 1.98 s spike, so the spike was the last team's
  200 ms block-time spin, not the orphans. Orphaned teams sleep permanently.
* **Per-frame latency: not reproducible.** The orphan arm's median ATen call
  went 0.31 -> 0.71 ms across 0 -> 1,946 threads, which looks damning until the
  control (zero orphans, identical loop) wanders 0.43 -> 0.63 ms. Bands
  overlap.

**So the case for fixing rests on unbounded memory, threads and handles, and on
nothing else.** A judge killing its own headline is the most useful thing in
this section.

**What makes it product-relevant is the cycle RATE.** `scene_autostart`
defaults **True** (`config.py:260`), and `ws.py` calls `stream_opened` /
`stream_closed` per websocket connection. **A cycle is not a wearer pressing a
button — it is a socket connecting and dropping.** A phone at the edge of
range, an app backgrounding, a Wi-Fi handoff: each reconnect is a full
Start/Stop with a fresh engine load and a fresh OpenMP team. At 6.85 MB/cycle a
4 GB headroom budget is gone in roughly **580 reconnects** — a week of flaky
link, not a day. Verified independently here.

**Correction to this document's earlier claim.** §5.5 said a session left with
`_engine is None` drops every frame. It does not: `_loop` holds `engine` as a
**local**, so with a tolerant engine `frames_observed` keeps climbing. With an
engine that fails once released the real symptom is worse —
`frames_offered: 1, frames_observed: 0, frames_dropped_not_running: 0,
state: running` — because every frame raises inside `_consume` and is swallowed
by "a frame failed; the session continues". The session runs forever, reports
healthy, and **no field in `status()` reveals it.** It also silently breaks
`DocumentLive.capture_started` / `capture_stopped`, which read `self._engine`
and no-op on None.

**Design verdict: FIX, as three commits in order** — (1) the finding-13 engine
capture, six lines, ships alone; (2) the reusable session worker; (3) findings
14/15.

Two structural facts settled the option space:

* **The session object is already a process-wide singleton** — `_scene_session`
  / `_document_session` are built once in `build_live_cartridges`, and the same
  instance serves every Start/Stop for the life of the process. So "a worker
  owned by the session" and "a process-wide worker" are the *same topology*
  here, and the former gets there without inventing a shared resource.
* **`start()` must NOT take `_lifecycle`.** That is the obvious way to close
  13/14/15 by construction and it is wrong: `_tell_cartridges_the_stream_opened`
  is called **inline on the event loop**, and its own docstring says so —
  "`stream_opened` stays inline: it starts a thread and returns". Making
  `start()` wait on `_lifecycle` would put a multi-second Document Memory flush
  on the event loop, trading a slow leak for a Tower that answers nothing
  during every reconnect. **Independently verified here** (`ws.py:769` inline;
  `stream_closed` goes through `asyncio.to_thread` at `ws.py:530`).

Prototype, 30 sessions each with a real ATen call:

```
reuse=False  30 cycles  threads 45 -> 615 (+570, 19.0/cycle)  workers=30
reuse=True   30 cycles  threads 45 ->  65 (+20,   0.7/cycle)  workers=1
reuse=True   30 cycles, 2 sessions wedged past the bound
                        threads 45 -> 103 (+58,   1.9/cycle)  workers=3  fallbacks=2
```

**The leak becomes proportional to abandonments rather than to cycles**, and on
the abandoned path the behaviour degrades to exactly today's: retire the wedged
worker, mint a fresh one, let the abandoned one release its own engine through
its own `LoadInvalidation`.

**The one design rule that decouples the two fixes:** step 3's
`thread.join(timeout=...)` must become a **per-session completion `Event`
captured in step 1**, never a wait on worker idleness — otherwise a `stop()`
for session 1 blocks on *session 2's* work, on an HTTP handler. That is the
single way this fix goes wrong.

Refusals it re-affirmed: no thread cap in any form; no `ThreadPoolExecutor`
(`concurrent.futures.thread` joins its workers at interpreter shutdown, the
exact hazard `tower/loading.py` already documents for `asyncio.to_thread`); no
worker shared between Scene and Document (1.19 s of OCR in front of a 33 ms
detection); no keeping the engine loaded across Stop/Start (that trades a
privacy-adjacent product guarantee for a memory fix); no reordering the four
steps.

Sibling site noted, not changed: `tower/object_memory/verification.py:283`
spawns a per-start thread that runs torch, in the same shape. Bounded in
practice — it lives in the object-memory producer subprocess and starts once
per run, not once per cycle.

## 6c. Judge 2, independently — and the measurement Judge 1 only predicted

Judge 2 reached the same verdict from the same evidence without seeing Judge 1,
and added three things.

**It measured that naive reuse is a hard regression.** Judge 1 reasoned that
step 3 must key on a per-session event rather than worker idleness; Judge 2
built both and drove a Start into a Stop's flush window:

```
UNSCOPED step-3 wait ("is the worker idle?"):
  stop() took 5.01s      -- the entire bound, on the stream-close path
  worker: ABANDONED      -- the one session 2 was actively loading on
  engines released: [False, True]   state=running  _engine=None

SCOPED step-3 wait (generation token):
  stop() took 0.00s      worker: reused
```

And the converse: with session 2's load made instant, the **scoped** variant
*still* reproduces finding 13 in full. So neither fix subsumes the other, and
the ordering — race first — is forced rather than stylistic.

**It dissolved the objection it was specifically assigned.** Asked whether
reuse introduces cross-thread engine teardown, CUDA affinity problems or
thread-local hazards, it probed which thread runs each hook *today*:

```
ordinary stop:   _create   on tower-probe-session-1
                 _consume  on tower-probe-session-1
                 _on_pause on http-handler-thread   <- easyocr flush, cross-thread
                 _teardown on http-handler-thread   <- engine.release(), cross-thread
```

**The shipped design already builds the engine on one thread and tears it down
on another, on every ordinary stop**, and `SceneEngine.release()` already
reaches `torch.cuda.empty_cache()` there. Reuse does not introduce cross-thread
teardown; it is the status quo. The genuinely new exposure is narrower —
session 2's `_create` running on a thread that previously ran session 1's
`_consume` — and is what the parity test buys. It declined to claim the CUDA
path was safe and asked for a manual check before merge; that is queued in §11.

**It corrected the product-exposure estimate downward.** `capture.py:303`
records that uvicorn takes 20–40 s to notice a dead socket while iOS reconnects
in 0.5 s, so the common Wi-Fi hiccup is *reconnect-before-close* — the owner
set is never empty and **no cycle occurs**. Cycles come from app backgrounding,
deliberate `stream_stop`, being out of range long enough for the old socket to
close first, and route Start/Stop. Realistically tens per day, so +150 to
+500 MB per day of walking, cleared only by a restart. Serious for a
long-running Tower; not a same-day outage. Both judges independently refused to
call it thread exhaustion.

**One finding neither the brief nor this lane had:** the `asyncio.to_thread`
path where `stop()`'s flush and teardown run leaks teams too — but it
*plateaus*, at `pool_size x torch_threads`, measured at +640 threads with
`max_workers=32`. Roughly 250 MB of one-time RSS nobody has booked. Bounded, so
not a blocker; recorded in §9.

## 7. The live-session leak, before and after

Same harness, same routes, same host, `tower.__file__` asserted into the
worktree, every cycle confirmed to reach `running`:

| | OS threads | per cycle | handles | RSS |
|---|---|---|---|---|
| before, 12 cycles | 29 -> **257** | **+19.00**, twelve for twelve | +38/cycle | +8.0 MB/cycle |
| after, 12 cycles | 29 -> **49** | **+0 from cycle 2 on** | flat at 450 | flat |
| after, 30 cycles | 29 -> **49** | **+0 from cycle 2 on** | flat at 450 | flat |
| after, 50 cycles | 29 -> **49** | **+0 from cycle 2 on** | flat at 450 | flat |

All of the residual +20 is cycle 1 — one OpenMP team and one worker — and that
cost is **constant**. Before the fix, 50 cycles would have reached ~979
threads; a judge's 1,200-cycle run reached 22,845 threads and 8.8 GB.

RSS is constant too, and the three run lengths are the evidence: it settles at
**~700 MB whether the run is 12, 30 or 50 cycles** (699.1 / 708.6 / 704.4),
oscillating a few MB either way with no trend. Before the fix it was climbing
+8.0 MB per cycle and had reached 790.6 MB by cycle 12 with no sign of
stopping.

`update_sighting`'s scaling table in §10 was taken the same way and is the
other half of the resource picture.

## 8. Changes explicitly refused

| Candidate | Why |
|---|---|
| **Capping torch threads** (`scene_torch_threads`, `OMP_NUM_THREADS`, `KMP_BLOCKTIME`) | Already measured by the previous lane at object_detection +83% and depth +179%, pushing 20% of depth frames past the whole 83.3 ms interval. Both judges re-affirmed the refusal, and both noted it is not a fix anyway — it divides the leak by 19 rather than removing it. |
| **`ThreadPoolExecutor`** | `concurrent.futures.thread` registers an atexit hook that JOINS its workers at interpreter shutdown, so a wedged OCR or a cold model load would hold process exit open. This repository has already paid for that bug once and documents it in `tower/loading.py`. Measured not to remove the leak either: it plateaus at `pool_size x torch_threads` (+640 threads at `max_workers=32`). |
| **One worker shared between Scene and Document** | Serialises the two cartridges: 1.19 s of OCR in front of a 33 ms detection. Breaks the module header's property 1 outright, and saves exactly one thread over the design taken. |
| **`start()` taking `_lifecycle`** | The obvious way to close findings 13/14/15 by construction, and wrong. `_tell_cartridges_the_stream_opened` runs **inline on the event loop** and its docstring says so. This would put a multi-second Document Memory flush on the loop — trading a slow leak for a Tower that answers nothing during every reconnect. Verified independently at `ws.py:769`. |
| **Keeping the engine loaded across Stop/Start** | Stop would stop meaning stopped, and `SceneEngine.release()` resets session-scoped track ids. Trades a privacy-adjacent product guarantee for a memory fix. |
| **A worker that exits when idle** | This bug with a timer on it: the thread would go and its OpenMP team would stay. Parking is the point. |
| **Reordering `stop()`'s four steps** | Nothing in the fix needs it, and the file records what both previous inversions cost. |
| **Deleting any existing stub world or capture** | This lane has no deletion permission and did not exercise one. The 86 stubs and 407 MB of captures already on disk remain the user's decision. |
| **Fixing integration finding 15 mechanically** | Reproduced and carried as a STRICT xfail instead. `stop()` clearing `_stream_owners` is itself a deliberate fix for the opposite defect, so choosing between them is a product decision about iOS-facing lifecycle semantics — see §9. |

## 9. Remaining deferred work

* **Integration finding 15**, reproduced and carried as a strict xfail in
  `tests/test_live_session_lifecycle_races.py`. After Stop -> Start on the HTTP
  routes the session is owned by nobody, so the stream closing cannot stop it —
  a privacy defect for a cartridge that detects people. Fixing it needs the set
  of OPEN streams tracked separately from the set of OWNING ones, and a
  decision about whether an HTTP `start()` while a stream is open should be
  owned by that stream. The two defects are opposite ends of one choice:
  adopt too eagerly and a passing connection stops a hand-started session;
  adopt not at all and a restarted session can never be stopped by the stream.
* **Integration finding 14** — a stopped session's scene publishing under the
  new session's identity. Same file, not attempted here.
* The `asyncio.to_thread` executor's own OpenMP teams: bounded at
  `pool_size x torch_threads` (~250 MB one-time at `max_workers=32`), currently
  unbooked. Found by Judge 2, not acted on.
* The two-converter refinement to the artifact-root guard (§4.1).
* `OMP_WAIT_POLICY=passive` — measured thoroughly and **refused** (§10a). One
  experiment would settle it: Scene and a CV Lab experiment driven
  CONCURRENTLY through a real websocket at 12 fps, both policies, comparing
  `frames_skipped` and Scene's p95.
* `process()` on the event loop, capture recorder I/O, capture retention, and
  Object Memory store scaling — see §10.

## 10. The other deferred candidates, and their disposition

| # | Candidate | Status |
|---|---|---|
| 1 | Live-session torch thread pool | **FIXED** (§5.5, §7) |
| 2 | `OMP_WAIT_POLICY=passive` | measured across four workloads; **DEFERRED, not shipped** (§10a) |
| 3 | Synchronous `process()` on the event loop | **STILL PRESENT**, characterised, not re-measured |
| 4 | Capture recorder I/O | **STILL PRESENT**, shape confirmed |
| 5 | Capture retention / unredacted data | **CONFIRMED and quantified** (§5.2) |
| 6 | Stub world accumulation | **FIXED** (§5.1) |
| 7 | Object Memory store scaling | **CONFIRMED**, not measured |
| 7 | Document Memory "journal rewrite per record" | **NOT REPRODUCIBLE** (§5.3) |

**Finding 3 — `process()` on the loop.** Still reachable and unchanged:
`tower/routes/ws.py:128` calls `module_container.process(frame.raw_bytes)`
synchronously in the websocket handler. The constraint the previous lane named
is real and visible in the code — the provenance handshake is read *between*
`process()` and the `await`, and `ws.py:210` says that placement "is the whole
guarantee: the frame path is synchronous up to this point, so nothing can have
changed the running experiment since the result was computed". Moving the work
off-loop moves that guarantee with it. Not attempted here; it is an
architecture change, not an optimization, and this lane spent its risk budget
on the leak.

**Finding 4 — recorder I/O.** Confirmed by reading rather than re-profiled:
`tower/capture.py:260-275` writes the image to a temp path, `flush()`,
`os.fsync()`, renames, then appends the journal line — "Image first with fsync,
journal line second." That ordering **is** the durability guarantee, so any
batching trades it away rather than improving it. `append_jsonl`'s heal-a-torn
-line read is per-append by design. **KEEP_AS_IS is the likely correct answer**
and nothing here contradicts it.

**Finding 7 — Object Memory. Confirmed AND measured.**

| records | `append` ms | `update_sighting` ms | `all_observations` ms | file KB |
|---|---|---|---|---|
| 64 (today) | 0.427 | 1.902 | 0.611 | 39 |
| 250 | 0.414 | 3.395 | 1.757 | 153 |
| 1,000 | 0.410 | 9.324 | 6.564 | 611 |
| 4,000 | 0.419 | 36.787 | 30.531 | 2,445 |
| 16,000 | 0.406 | **157.444** | **148.507** | 9,779 |

Three things fall out, and one contradicts the grouping in the previous report:

* **`append` is O(1)** — 0.41 ms flat across a 250x range. `append_jsonl` is a
  genuine append, exactly as §5.3 found for Document Memory. Nothing to fix.
* **`update_sighting` is O(n)**, 1.9 -> 157 ms: it reads every raw record and
  rewrites the file whole.
* **`all_observations` is O(n)**, 0.61 -> 149 ms, re-parsed per call.

**Not product-threatening today, and the trigger is now a number rather than a
feeling.** At today's 64 records both cost under 2 ms. The interesting
threshold is ~4,000 records, where `update_sighting` reaches 37 ms — a
meaningful fraction of the 83.3 ms frame interval — and 16,000, where it
exceeds the whole interval. It does NOT block the Tower's event loop, because
the producer is a separate process, so the failure mode is a producer falling
behind rather than a dropped frame.

Reaching 4,000 needs a great deal of walking inside a 30-day retention window,
against 64 accumulated so far. **KEEP_AS_IS is correct now**; the store's own
docstring already names the assumption ("the file is expected to stay small;
that assumption is the trigger to revisit") and this puts a figure on it.

**If it ever needs fixing, the fix is not SQLite.** Both costs are one full
read of a file that is 611 KB at 1,000 records; an in-memory cache keyed on
`(size, mtime_ns)` — the pattern `results/world_builder.py` already uses — or
an index over the three fields `update_sighting` matches on removes them with
no migration and no format change.

## 10a. `OMP_WAIT_POLICY=passive` — measured across four workloads

The previous lane recorded a Scene session going from 18.6 cores to 2.29 and
host CPU from 100% to 32% under `passive`, called it "a promising separate
lever, unvalidated for correctness", and did not ship it. This lane validated
it, and the answer is not the one the headline implies.

**Arm order alternated, medians of two runs each, same host, `tower.__file__`
asserted into the worktree.**

### The paced frame path — the number that flatters it

`cartridge_live_benchmark.py`, Scene, 1,845 real corpus frames at the measured
delivery rate of 12 fps:

| | default | passive | ratio |
|---|---|---|---|
| CPU cores | **17.93** | **1.74** | **0.10** |
| observed fps | 11.97 | 11.97 | 1.00 |
| frames observed / offered | 1845 / 1845 | 1845 / 1845 | 1.00 |
| **frames skipped** | **0** | **0** | — |
| `offer_frame` median | 0.012 ms | 0.014 ms | 1.15 |
| `offer_frame` p95 | 0.024 ms | 0.032 ms | 1.35 |
| RSS growth | 10.1 MB | 10.3 MB | 1.02 |

A **10.3x CPU reduction with byte-identical throughput and not one dropped
frame.** The event-loop cost rises 15-35% and is measured in *microseconds*.

**One number in that harness must not be quoted.**
`worker_service_ms_mean` reads 1497.7 -> 145.6 ms, which looks like a 10x
throughput win and is not one: it is computed as `cpu_seconds / frames`, so it
divides CPU spread across ~18 cores by the frame count. It is the CPU figure
restated, and the benchmark's own docstring mislabels it as "how long the
cartridge occupies its thread".

### The saturated path — the number that condemns it

Back-to-back `ssdlite320` inference with no idle gap, 60 inferences after 10
warm-ups:

| | default | passive |
|---|---|---|
| median | 28.2 / 29.1 ms | **44.4 / 47.4 ms** |
| p95 | 32.0 / 32.1 ms | 48.1 / 50.5 ms |
| throughput | 34.0 / 35.2 per s | 21.5 / 22.7 per s |
| CPU cores | 17.9 | 2.5 |

**+57% to +63% per inference**, reproducible across both pairs. Spin-waiting is
not free to remove: a sleeping team has to be woken for every parallel region,
and at 12 fps that cost is hidden by the idle gap while at saturation it is
not.

### Depth — the workload that decided the previous refusal

The runtime-fitness lane **refused** capping torch threads because depth went
19.73 -> 55.01 ms (+179%) and pushed **20% of frames past the entire 83.3 ms
delivered interval**. `passive` is held to the same bar:

| | default | passive |
|---|---|---|
| median | 28.7 / 29.7 ms | 49.0 / 48.2 ms (+65%) |
| max | 41.2 / 33.5 ms | 52.2 / 51.2 ms |
| **over the 83.3 ms interval** | **0/40** | **0/40** |
| CPU cores | 17.2 / 17.9 | 3.4 / 3.8 |

**`passive` is materially safer than the cap that was refused.** It costs about
the same proportion of latency but leaves 31 ms of headroom against the frame
interval where the cap left none.

### World Builder — the subsystem whose figures are pinned exactly

`world_builder_benchmark.py`, every stage, both policies:

```
build.estimate_window.mean_ms      65.70 -> 65.33   (1.00)
build.ms_per_keyframe               8.21 ->  8.17   (0.99)
preprocessing.*                    every ratio 0.87-1.10 on sub-0.1 ms filters
```

**Unaffected, on every stage.** That is the result this experiment most needed:
World Builder is the largest subsystem in the backend and its corpus figures
reproduce exactly, so a process-wide switch that perturbed it would have been
disqualifying. Its hot path is OpenCV and compiled numerics with their own
threading, not torch's OpenMP, which is why.

### Both cartridges — and the limitation of that run

`--cartridge both`, Scene and Document Memory, 1,280 frames each:

| | default | passive |
|---|---|---|
| Scene CPU cores | 17.897 | **1.755** |
| Scene frames observed / skipped | 1280 / 0 | 1280 / 0 |
| Document CPU cores | 0.264 | 0.218 |
| Document frames observed / skipped | 1280 / 0 | 1280 / 0 |
| both observed fps | 11.97 | 11.97 |

**This is NOT the concurrency test it looks like.** The harness drives the two
cartridges one after the other — two `runs` entries, 106.9 s of wall clock
each — so it measures both cartridges *in one process*, not both *at once*.
The genuinely concurrent case is still unmeasured, and it is the one the
integration report already flags: finding 10, "the CV Lab can cost Scene its
entire frame-time margin".

### Verdict: **DEFER. Do not ship in this lane.**

Not because the numbers are bad — they are the best in this report — but
because of what they do and do not cover.

**What decides it is that the product is not CPU-bound today.** Both arms
delivered 1,845 frames with **zero** skipped, at an identical 11.97 fps, in
every configuration measured. The 10.3x saving buys *headroom*, not product
behaviour. Headroom is worth having, and it is not worth a **process-wide
+65% frame latency** bought on measurements that never ran two torch cartridges
at the same time — on the one axis a previous report already names as able to
consume Scene's whole margin.

There is also no narrow scope available. `OMP_WAIT_POLICY` is read when the
OpenMP runtime initialises, before torch finishes importing, so it can only be
a process-wide startup setting. "Only for the Scene worker" is not reachable.

**The honest asymmetry, stated so a successor can overturn this cheaply:** the
theory says passive should *help* the concurrent case, not hurt it — under the
default, one Scene session already claims 17.9 of 20 cores, so two torch
cartridges oversubscribe badly, while under passive each wants under 4. If that
is measured and holds, the case flips and this should ship.

**The one experiment that would settle it:** drive Scene and a CV Lab
experiment *concurrently* through a real websocket at 12 fps under both
policies, and compare `frames_skipped` and Scene's p95. If passive does not
raise either, ship it as a startup default with the latency cost documented.

This is a stronger recommendation than the previous lane's, on far more
evidence, and it is still a refusal.

## 10b. Final verification

```
Full Tower suite     2241 passed, 58 skipped, 1 xfailed in 305.11s   exit 0
pip check            No broken requirements found.
tower.__file__       ...\Glasses-worktrees\post-mvp-hardening\tower\tower\__init__.py
torch                2.13.0+cu132
torch.cuda           True
GPU                  NVIDIA GeForce RTX 5070
conflict markers     none in any tracked file
git status           clean but for this document
```

**Zero failures.** The one xfail is integration finding 15, strict and
deliberate (§9).

The reviewer's independent baseline on this branch before the last three
commits was `2233 passed, 58 skipped`; the eight extra passes are the tests
added since.

**World count in the worktree's default root: 6 before the run, 6 after.** That
is the check that matters for §5.1 — before the fix, a full suite incremented
it every time.

## 11. Physical validation still required

Nothing in this lane was validated on glasses or a phone. Queued, with the
reason each needs hardware:

1. **A real Wi-Fi reconnect cycle.** The whole product case for the thread fix
   rests on `stream_opened`/`stream_closed` firing from real socket churn.
   Judge 2's reading of `capture.py:303` says the common hiccup is
   reconnect-*before*-close and therefore **not** a cycle. Walk out of range
   and back, repeatedly, and read `os.num_threads()` and the session's
   `session_id` to see how many cycles a real walk actually produces.
2. **Scene Understanding on CUDA across a Stop/Start.** Judge 2 declined to
   claim the CUDA path is safe under worker reuse and asked for a manual check
   rather than assert it. `TOWER_SCENE_DEVICE=cuda`, then Start/Stop several
   times and confirm the engine still loads and observes, and that VRAM
   returns.
3. **A Document Memory flush racing a reconnect.** The finding-13 window is
   seconds wide only because Document's `_on_pause` is OCR. Reproduce on
   hardware: begin a Stop while a page is being read, reconnect immediately,
   confirm the new session keeps its engine.

## 12. Temporary resources created by this lane

| Path | What |
|---|---|
| `C:\Users\tvllo\Projects\Glasses-worktrees\post-mvp-hardening` | the worktree (git-tracked, `git worktree add` with an explicit destination) |
| `C:\Users\tvllo\Projects\Glasses-scratch\post-mvp-hardening\` | fixtures, harnesses, agent scratch |
| `C:\Users\tvllo\Projects\Glasses-scratch\pt85`, `pt85b`, `pt85c` | pytest basetemps, one per agent |
| `<worktree>/tower/data/` | test fixtures — one copied world, one synthetic observation store; gitignored |

**Nothing was created under `C:\` or `C:\Users\tvllo\` directly. Nothing under
any `data/` directory was deleted or moved.** This lane has no deletion
permission and did not exercise one.

## 13. Agents used, and what each was worth

| Role | Verdict it produced |
|---|---|
| Agent B (object memory) | Root cause confirmed with evidence; **corrected this lane's framing** — the world-root defect is a *different* mechanism (relative, cwd-anchored) from the observation-root one (absolute, file-anchored), and a single sweep would have got one wrong. Also localised the stub minting by per-file bisection. |
| Agent C (world registration) | Traced the failure to `96f6e21` and **ruled out a real regression by experiment**, not argument. Found that `96f6e21`'s own commit message reads "449 passed, 10 skipped" while `TestTheRealWalk` had exactly 10 tests — **the author never saw the class run when they broke it**, because `REAL_ROOT` was cwd-relative. |
| Reviewer R | Attacked by MUTATION. Found that the suite could no longer prove the product default (a mutation passed 2233/58 unchanged) and that **my own commit message's RED claim was not reproducible**. Four attacks failed and are recorded as evidence. |
| Judge 1 | Drove the leak to 1,200 cycles; **killed its own headline** by disproving the CPU and latency costs it first measured; established that the session object is already a process-wide singleton, collapsing the option space. |
| Judge 2 | **Measured** what Judge 1 predicted (naive reuse costs `stop()` its whole bound and abandons a live worker); **dissolved its own assigned objection** by showing cross-thread teardown is already the status quo; corrected the product-exposure estimate downward. |

The pattern worth keeping: **every agent that mattered corrected something, and
three of them corrected the person who briefed them.** Two judges reaching the
same design independently is worth more than one judge agreeing with me.

## 14. Success, measured against the brief

* The three unexplained red tests are resolved — one by `tower-dc` on `main`,
  two here, each with the root cause identified and a regression test proven
  RED first.
* The most serious deferred resource issue is **fixed**, not re-characterised:
  +19 threads/cycle to +0, with the mechanism isolated and the fix designed by
  two independent judges.
* Major optimization candidates were measured honestly, including two whose
  headline claims **did not survive measurement** and were withdrawn.
* Unsafe and speculative changes were refused, with the measurement that
  refused each one (§8).
* Current MVP behaviour is preserved: `ios/` untouched, no contract changed, no
  cartridge semantics altered, no data deleted.
* No new unbounded resource growth was introduced; the one growth path that
  remains is bounded and booked (§9).
* Final suite: **0 failures**.
