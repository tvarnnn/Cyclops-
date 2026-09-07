# Object Memory stabilization, Tower resilience, and user-teachable objects

**Date:** 2026-09-07
**Branch:** `fix/object-memory-runtime-v1`
**Worktree:** `C:\Users\tvllo\Projects\Glasses-worktrees\object-memory-runtime-v1`
**Base:** `6beaf57` (integration/wb-cv-ios-validation-v1)
**Host:** Windows 11, 20 logical CPUs, RTX 5070 (Blackwell sm_120), Python
3.12.5 in `tower\.venv` (torch 2.13.0+cu132, uvicorn 0.52.4, transformers
5.16.1, scipy 1.18.1).

This is a Tower-lane mission. Nothing under `ios/` was compiled or changed;
the two client-facing changes (the composed lifecycle and the `since`
filter) are described so the iOS lane can consume them, and are called out
in §12 as the physical-test dependencies.

---

## 1. Executive summary

Two production failures from the physical tests were reproduced offline,
root-caused to the mechanism, fixed, and guarded by tests. Both are Windows
runtime bugs, and **neither is caused by the Object Memory ML pipeline** --
both reproduce with no worker, no torch, no GPU.

1. **Object Memory recorded zero frames.** The owlv2 producer deadlocked
   inside its own model load and never reached the frame loop. Cause: a
   Windows loader-lock hang between the stdin-stop watcher thread (blocked
   in a pipe `ReadFile`) and the OpenBLAS DLL that `transformers`/`scipy`
   load during the OWLv2 verifier init. Fixed by warming that native stack
   before the watcher is armed. A second, separate crash on the way out
   (`0xC0000005` at interpreter teardown) is fixed by a hard exit after the
   flush.

2. **The Tower stopped serving while its process stayed alive** -- port
   8000 gone, `/health` refused everywhere, shutdown hanging on Ctrl-C.
   Cause: CPython gh-93821, an unfixed `ProactorEventLoop` bug that closes
   the listening socket on a per-connection accept error and never re-arms;
   plus uvicorn's unbounded graceful-shutdown drain. Fixed with a resilient
   loop and a bounded shutdown timeout.

Also improved: a swallowed `WebSocketDisconnect` that surfaced as an ASGI
error, and a new `since` query parameter that lets a client tell a
recording's own memories from history (the truthfulness gap the zero-frame
run exposed).

Verified offline: the owlv2 worker now reaches its frame loop and records
observations (244 frames -> 4 observations on the real physical-test
capture), exits 0 on every path, and survives repeated Start/Stop with no
process or memory leak; the resilient loop keeps its listener through a
reset-in-backlog burst that kills the stock loop; the real uvicorn binds
and serves on the new flags. **Not yet done offline:** the user-teachable
object-recognition lane (research + prototype + benchmark) is in progress
(§5) and the physical device run (§12) is owed.

---

## 2. Tower connection / lifecycle findings

### 2.1 The listener that vanished while the process lived — root cause

Reproduced against the real `tower.main:app` under uvicorn on the default
Windows loop, with no ML worker and no GPU. This is **CPython gh-93821**,
unfixed in 3.12 and on `main`.

On Windows uvicorn runs on `asyncio.ProactorEventLoop`. Its accept path
(`asyncio/proactor_events.py`, `BaseProactorEventLoop._start_serving`,
the `loop` closure) does this: when the `AcceptEx` completion raises an
`OSError` -- which happens when a connection queued in the kernel backlog
is **reset by its peer before the loop accepts it** -- the handler
(`proactor_events.py:864-871`) logs `Accept failed on a socket`, calls
`sock.close()` on the **listening** socket, and never re-arms accept. The
loop keeps running, every existing connection keeps working, and the port
is simply gone. `Server.is_serving()` still returns `True`. The companion
`accept_coro` task produces the `Task exception was never retrieved ...
OSError: [WinError 64]` line.

**Why port 8000 disappeared while processes stayed alive:** the listening
socket was closed by the accept handler; the process, the event loop, and
the existing websocket/capture/worker all continued. That is exactly the
field observation -- `Get-NetTCPConnection -LocalPort 8000` empty, `/health`
refused on localhost AND Tailscale, process tree resident.

**The enabling condition is a busy loop.** While the event loop is occupied
(decoding a multi-MB frame, an fsync, a `reap()` under the supervisor lock,
DEBUG console logging with `TOWER_DEV_MODE=true`), only one `AcceptEx` is
armed, so connections pile up in the backlog and any one reset while queued
trips the bug. A shared, long-lived Tower that a phone reconnects to over a
flaky link (iOS retries on a backoff after a capture `disconnect`, Tailscale
re-paths, a URLSession attempt is cancelled) is precisely that workload.
Measured: pure racing RSTs without a loop stall survived 10 rounds; a 300 ms
stall killed the listener in round 0; a `SelectorEventLoop` survived every
round.

### 2.2 Why shutdown hung

With uvicorn's default `timeout_graceful_shutdown=None`, `Server.shutdown()`
(`uvicorn/server.py:288-291`) waits **forever** for a websocket transport
whose `WSASend` never completes -- a peer that is alive but not reading (a
phone suspended with the socket open, its TCP window at zero). A second
Ctrl-C does not help: after `force_exit`, uvicorn still awaits
`server.wait_closed()` on the same transport. That is the `INFO: Shutting
down` that then never finished. With `timeout_graceful_shutdown=5` the
identical scenario exits in ~6 s.

Tower's own lifespan shutdown (`tower/main.py:473-506`) is already bounded:
the hub cancel, the supervisor's `shutdown` (10 s grace + 2 s terminate),
the CV Lab loader cancellation, and the container `wait_for` are all
finite. The unbounded wait was uvicorn's, not Tower's.

### 2.3 Relationship to the Object Memory worker

**None.** Both failures reproduced against the bare Tower with
`TOWER_OBSERVATION_ENABLED=false`, no world root, the CPU baseline
experiment: no torch, no GPU, no worker. The worker being alive after the
capture closed (a field observation) is the *follower* correctly waiting
out its idle bound after a `disconnect`; it is not the cause of the
listener death. The zero-frame worker bug (§3) is a genuinely separate,
worker-local failure.

### 2.4 Changes made

- **`tower/serve_loop.py`** (new): `ResilientProactorEventLoop`, a
  `ProactorEventLoop` subclass whose accept closure differs from CPython's
  in exactly one place -- an `OSError` from the per-connection `f.result()`
  whose Winsock code names the *connection* (WinError 64, 1236, 10053,
  10054, 10057, 10058) is logged and skipped, and accept is re-armed,
  rather than closing the listener. Every other error keeps CPython's
  behaviour. `resilient_loop_factory()` returns a loop **instance** (the
  custom `--loop module:callable` path hands the imported name straight to
  `asyncio.run(loop_factory=...)`, which calls it for a loop; returning the
  class -- as an early draft did -- binds nothing and fails at startup).
- **`scripts/start_tower.ps1`**: the uvicorn argv gains `--loop
  tower.serve_loop:resilient_loop_factory` and `--timeout-graceful-shutdown
  10`. (`--loop asyncio` does NOT help: it still yields a Proactor loop on
  win32.)
- **`tower/routes/results_ws.py`**: `handle()` no longer swallows
  `WebSocketDisconnect` in its broad fault handler. A send on an
  already-dropped socket raised it; swallowed, the receive loop discovered
  the dead socket only on its next `receive_json`, which raised a bare
  `RuntimeError: WebSocket is not connected` that uvicorn logged as
  "Exception in ASGI application". It now propagates to the endpoint's own
  `except WebSocketDisconnect` and ends the connection cleanly. Result-
  channel faults are still swallowed.

### 2.5 Evidence and failure-injection

The investigation (full report and runnable repros in
`Glasses-scratch\om-runtime\tower-listener\`) reproduced both symptoms
line-for-line against the real app: the two WinErrors, `Accept failed on a
socket`, listener `fileno=-1`, `netstat` empty, `/health` refused, process
and loop alive. The resilient loop absorbed 116 accept failures over 6
rounds with the listener alive and `/health` 200; `--timeout-graceful-
shutdown 5` turned a >45 s hang into a 6.4 s exit. Both were then re-checked
here: the real uvicorn was launched on the two flags and served `/health`
200, and the loop-survival test (below) is now a regression guard.

**Remaining hypothesis, not fully closed:** the field shutdown hang's exact
stuck connection. The mechanism is proven (a stalled websocket write), but
whether the field instance was that or an h11 keep-alive from `/health`
probes depends on whether the operator's `Get-NetTCPConnection` was filtered
with `-State Listen`. Both take the same fix (`--timeout-graceful-shutdown`).

---

## 3. Object Memory zero-frame findings

### 3.1 Exactly why 244 frames received became 0 observed

The producer (`scripts/object_memory_session.py`) is spawned by
`CaptureWorkerSupervisor` with `stdin=PIPE`, `--stop-on-stdin-close`, and
`CREATE_NEW_PROCESS_GROUP`. On startup it installs `_StopRequest`, whose
`--stop-on-stdin-close` path starts a daemon thread blocked in a
synchronous `ReadFile` on that pipe (the stop channel that works under a
pseudoconsole where a console-control event does not). Then `engine.load()`
loads the detector and, when the verifier is `owlv2` (the production
default since 2026-08-29), the OWLv2 model -- which imports `transformers`,
which imports `scipy.linalg`, which loads `libscipy_openblas*.dll`, which
spawns its thread pool in `DllMain`.

Dissected with py-spy `--native`: the main thread sits in that DLL load
under the Windows loader lock (`LdrpDrainWorkQueue` /
`ZwWaitForAlertByThreadId`) while the `object-memory-stop-watch` thread is
blocked in the pipe `ReadFile`. A thread parked in a blocking pipe read
while the loader brings up a thread-spawning DLL is a hard hang: **0% CPU,
forever**, until Stop terminated the worker.

- **Attachment / launch ordering were correct.** `attach_mode=from-start`,
  the follower attached at capture open, `frames_observed:0` was the
  producer never leaving its load -- not a journal or attach fault. The
  capture journal, the follower, and `from-start` semantics all work; the
  data-inventory agent confirmed a CPU replay of the shipped producer over
  the four physical-test captures writes 11 records, so the zero was never
  a property of the footage.
- **Why every test missed it:** every subprocess-spawning test pins
  `--verifier none` (and usually `--detector none`) to avoid downloading
  weights, so nothing ever loaded transformers/scipy in a spawned worker
  while the stdin watcher was armed. Production defaults to `owlv2`.

### 3.2 The exit-time crash (separate, pre-existing)

When a walk ends by **frames-ended** (the capture closed, no successor,
idle bound) rather than by Stop, the parent still holds the pipe, so the
watcher is still blocked in `ReadFile`. Finalizing the interpreter with
that thread blocked races CUDA/torch teardown and access-violates
(`0xC0000005`) -- **after** the flush and the report, so no data is lost,
but `CaptureWorkerSupervisor.reap` logs a perfect walk as `EXITED
3221225477`. Reproduced with the verifier OFF as well as on, so it is the
blocked reader plus GPU teardown, generic and pre-existing; the zero-frame
fix merely made this path reachable for owlv2 workers.

### 3.3 The fixes

- `_prewarm_native_libraries()` imports `scipy.linalg` on the main thread
  **before** `_StopRequest.install(watch_stdin=...)`, so OpenBLAS comes up
  single-threaded with no pipe-reader parked behind the loader; the later
  `transformers` import is then a no-op and the watcher stays armed for the
  whole run (so a Stop during the ~8 s load is still honoured). Contained:
  a host without scipy cannot load owlv2 either, so it degrades to no
  verifier, and a warming failure is a warning, never a refusal to start.
- `_run_and_exit()` calls `os._exit(code)` after an explicit stdout/stderr
  flush, on the SUCCESS path only. A `SystemExit` from argument parsing or
  any exception still finalizes normally and keeps its traceback.

### 3.4 Regression test

`tests/test_object_memory_worker_startup.py`:
- Fast, no GPU: pins the pre-warm/watcher ORDER (the fix's mechanism) and
  the `os._exit` contract.
- Gated (`TOWER_RUN_MODEL_TESTS=1`): spawns the real owlv2 worker with a
  held stdin pipe -- the physical shape -- and asserts it observes frames
  and exits 0. **Proven red/green:** it times out (92 s deadlock) with the
  pre-warm disabled and passes in ~18 s with it.

---

## 4. Object Memory V1 findings (lifecycle, worker ownership, state)

- **Start / Pause / Resume / Stop** are the existing `CartridgeSession`
  state machine (`tower/cartridge_session.py`), unchanged by this lane and
  exercised green by `tests/test_object_memory_lifecycle.py` and the
  autostart e2e. Start opens the gate before the camera; Pause and Stop
  detach the producer; the producer's `finally: engine.release()` flushes
  open sightings, which is the graceful-stop mechanism the earlier product
  pass built.
- **Flush on Stop verified with the real worker:** on the physical-test
  capture, both the natural idle-end and the stdin-closed Stop path write
  4 records to disk and exit 0 (owlv2 and none). Stop is lossless -- the
  earlier "0 records on Stop" reading was a bug in an ad-hoc measurement
  harness, disproven by on-disk verification.
- **Worker ownership / cleanup verified by soak:** 4 repeated Start/Stop
  cycles of the real owlv2 worker through the real supervisor -- every
  cycle reaped cleanly, 0 leftover workers, 0 leftover RSS, final 0 alive.
  No process or memory leak. A worker detached mid-load (before the frame
  loop starts) is bounded-terminated at the 3 s grace, which is correct:
  no observations exist during load, so nothing is lost.
- **Current vs historical (§8 of the brief):** the store already stamps
  every record with `recorded_at` and `session_id`, and the session
  snapshot carries `started_at`, `captures`, and `following_this_session`.
  The gap the zero-frame run exposed was that telling a recording's own
  memories from history was left to client correlation, and that is what
  failed. Now `GET /object-memory/observations?since=<started_at>` returns
  only the records the Tower wrote at or after a moment -- a first-class,
  tested primitive. An empty recording gets an honestly empty answer
  instead of borrowing history's; the whole-store view is a separate
  request. Additive parameter and field; the contract identifier is
  unchanged. See `docs/contracts/OBJECT-MEMORY.md` §3.1/§4.2.

---

## 5. Custom / user-teachable object research

Full record: `tower/docs/superpowers/research/2026-09-07-teachable-object-
instance-recognition.md` (in-tree, carries the decision); runnable prototype
and raw artifacts in `Glasses-scratch\om-runtime\teachable\`.

**Architecture selected** (evidence-backed, on our footage): `existing
ssdlite320 detector → crop (face-filtered) → frozen DINOv2-small 384-d
embedding → multi-view prototype gallery (3–5 pose-diverse enrollment views)
→ nearest-prototype cosine match → calibrated distance threshold (~0.48) → a
confidence-scored, human-confirmed claim`. Enrollment encodes a few crops
(no retrain); a profile is embeddings + a mean prototype + optional owned
crops; deletion drops the vectors. Persistence is a new `identity/` sibling
to `object_memory/` under its own dated contract that does not touch
`object_memory.observations/2026-08-26`, so the observation stream stays
`category-not-instance` and identity is added only as a probabilistic
association. It is few-shot, incremental, and cheap: 1.5 KB/vector, 2.1
ms/crop GPU / 17.5 ms/crop CPU, 331 MB VRAM — it fits Tower's budget on CPU
alone.

**Rejected, with measured reasons:** OWLv2 image embeddings (the vision
pooler COLLAPSES — mean pairwise cosine distance 9.6e-6 over 531 crops — and
is 50× slower / 20× more VRAM; OWLv2 stays a category verifier); full
fine-tuning / per-object heads (breaks incremental add, too few examples);
pure local-feature matching as the recall engine (precise when it fires,
45-vs-4 median good matches, but misses 25% of same-object pairs and is ~0
on textureless objects — right role is an optional geometric verifier for an
auto-confirm band); CLIP backbone (semantic-category bias, the wrong axis);
open-vocab-only for identity.

**Benchmark on real crops (531 crops from the 116 records; labels assigned
by eye from montage sheets, tiered by identity-evidence and flagged):**

| Measurement (DINOv2-small) | Value |
|---|---|
| CLEAN same-sighting vs different-object AUC | 0.81 |
| Held-out enrolled view, 1 → 3 views AUC | 0.94 → 0.97 |
| Cross-session re-ID (same device, different capture) AUC | ~0.69–0.80 |
| macbook-laptop vs external monitor (genuine two-display) AUC | 0.89 |
| 3-class device leave-one-out 1-NN purity | 0.85 |
| Operating point at threshold ~0.48 (k=3) | TPR 0.80 @ FPR 0.34 |

Labelled-vs-exploratory is kept strict: the same-sighting tier is verified
(temporal continuity of one recording); the cross-session re-ID tier is
inferred from one-home/one-device continuity and flagged as such; negatives
are always a different true-object across categories, so the embedding
cannot key on "is this a glowing dark screen".

**Honest limits (the three that matter most):**
1. **Cross-session re-ID is only moderate** (AUC ~0.80; one in three
   different objects crosses the threshold). Viable as suggest-and-confirm /
   retrieval, NOT silent auto-identity. Small mass-produced look-alikes
   (a phone, AirPods, a black Yeti) are the worst case — the product's own
   examples.
2. **The corpus has one laptop and one phone**, so "tell two similar
   laptops apart" — the core promise — is untestable here; the 0.89
   macbook-vs-monitor figure is a proxy, not proof for same-model
   distractors. This is the production gate.
3. **Small, blurry, screen-dominated 360×640 crops:** the "object" is often
   a glowing screen whose content changes between sightings, so the
   embedding partly keys on transient screen content; ~30% of `laptop`
   detector boxes were not laptops, so a crop-quality/category gate is
   mandatory upstream.

Ground-truth discipline held throughout: no instance-identity labels exist
in the data; every identity claim states how it was established, and no
visually-similar crop was asserted to be the same physical object without
evidence.

---

## 7. World Builder / real-data evaluation

Read-only inventory of the real glasses footage (agent report:
`Glasses-scratch\om-runtime\data-inventory\INVENTORY.md`, `captures.json`):
97 capture directories (95 with frames), 48 walks after following
`continues_capture` lineage, **45,594 JPEGs, 1,013 MB, ~80.6 min, all
360x640**, 0 undecodable on a 3,643-frame sample. 116 Object Memory records
(laptop 61, cell phone 53, keyboard 1, mouse 1; 106 high / 10 medium; 2
carry real OWLv2 verdicts, both agreed). World Builder: 162 worlds, 8,591
keyframe lines, 6,832 keyframe JPEGs with `poses.json` (per-keyframe
`rotation[wxyz]`, `translation`, `T_world_camera`, scale unknown).

**Canonical data was never modified** -- every figure comes from reading it
in place. The scratch corpus and all extracted crops live under
`Glasses-scratch\`. **Ground truth:** no instance-identity labels exist
anywhere; the 116 records are detector category output; the only human
labels are 94 positional *category* crop verdicts in an older benchmark
(and the ranking file they referenced no longer exists in any checkout).

---

## 8. Performance (measured on this host)

| Quantity | Value |
|---|---|
| Worker cold start to first inference (owlv2, warm weights) | ~8.7 s |
| — of which OWLv2 model load | ~5 s |
| — SSDLite load | ~1.2 s |
| Producer throughput (ssdlite320, cuda) | ~40–47 ms/frame idle; higher under contention |
| Worker RSS (owlv2 on cuda, loaded) | ~0.8 GB |
| Detach/reap after Stop (past load) | clean flush, ~0.4 s |
| Detach/reap mid-load | bounded terminate at 3.0 s grace |
| Repeated Start/Stop (4 cycles) | 0 leftover processes, 0 leftover RSS |
| Resilient-loop shutdown (bounded) | ~6 s vs unbounded hang |

---

## 9. Testing

**Added:**
- `tests/test_object_memory_worker_startup.py` — pre-warm/watcher order,
  `os._exit` contract (fast); gated real-owlv2 deadlock+exit reproduction.
- `tests/test_serve_loop.py` — resilient loop keeps its listener through a
  reset-in-backlog burst (pass); stock Proactor loop loses it (xfail,
  gh-93821 as documentation). Windows-gated.
- `tests/test_result_channel_disconnect.py` — `WebSocketDisconnect`
  propagates from a handler; a result-channel fault is still swallowed.
- `tests/test_object_memory_transport.py` — 5 new `since` cases (this
  recording's memories, echoed field, empty recording ≠ history, over the
  route, negative refused 422).
- `tests/test_startup_scripts.py` — asserts the two new uvicorn flags.

**Results:** the added and neighbouring suites pass (worker-startup 3
passed + 1 gated; serve_loop 1 passed + 1 xfail; transport 34 passed;
lifecycle/capture-worker/graceful-stop 101 passed; result-channel + startup
103 passed). The gated owlv2 test passes in ~18 s and was proven to fail
without the fix. **The full `tower/` suite passes with all fixes and the
review changes: 2592 passed, 76 skipped, 2 xfailed, 0 failed in 487 s**
(the opt-in gated owlv2 test is among the skips unless enabled). It
must be run with a **short `--basetemp`** (`C:\Users\tvllo\AppData\Local\
Temp\gf`) or World Builder's nested UUID tmp paths trip Windows MAX_PATH and
manufacture phantom failures.

---

## 10. Independent review

**An independent adversarial review of the three production commits ran and
found four real issues, all fixed in `3435bdc`; the rest of the review
confirmed the changes hold** (the copied accept closure is faithful to
CPython 3.12.5, the `WebSocketDisconnect` fix matches Starlette's actual
send-raises behaviour, `since` cannot hit a null `recorded_at`, the prewarm
adds no BLAS/OpenMP risk, and the fast tests genuinely pin their contracts).

The four findings and their fixes:
1. **The exit-crash fix covered only the success path** — an exception from
   `engine.load()` (CUDA OOM, a failed weights download) or the frame loop
   finalized with the watcher still blocked, re-triggering the `0xC0000005`
   and masking the real error; the docstring's justification was factually
   wrong. `_run_and_exit` now hard-exits on every path (traceback + exit 1
   on a real error, SystemExit code preserved, success unchanged), with
   four new tests.
2. **The resilient-loop test could pass vacuously** if the reset never
   landed. It now also asserts the per-connection re-arm warning fired.
3. **The re-arm logged per reset** (plus asyncio's "Task exception was never
   retrieved" per reset), which a reconnect storm would flood. The warning
   is throttled and the specific asyncio noise is suppressed narrowly.
4. **A misleading `use_subprocess` docstring** was corrected.

The ML-review lens for the §5 research (label validity, enrollment leakage,
similar-instance negatives) was folded into synthesis rather than run as a
separate agent (budget): labels are inspection-based and tiered with the
inferred tier flagged, negatives are cross-category, and held-out-view AUC
is reported apart from cross-session re-ID AUC. The one unresolved review
point is that the hardest, most product-relevant case (similar same-category
instances) is unmeasured because the corpus has one instance per category —
which is stated as the production gate, §5.

**Not reviewed:** the iOS composed lifecycle (unbuilt source), which the
device run (§12) covers.

---

## 11. Git handoff

- **Branch:** `fix/object-memory-runtime-v1`
- **Worktree:** `C:\Users\tvllo\Projects\Glasses-worktrees\object-memory-runtime-v1`
- **Base:** `6beaf57`
- **Commits:**
  - `c85effc` fix(object-memory): the owlv2 worker reaches its frame loop, and leaves clean
  - `2975dd0` fix(tower): a reset connection no longer takes the Windows listener down
  - `269d0e7` feat(object-memory): `since` tells a recording's memories from history
  - (+ this handoff)
- **Major files:** `tower/serve_loop.py` (new), `scripts/object_memory_session.py`,
  `scripts/start_tower.ps1`, `tower/routes/results_ws.py`,
  `tower/routes/observations.py`, `tower/results/object_memory.py`, and the
  four test files above; `docs/contracts/OBJECT-MEMORY.md`.
- **Intentionally uncommitted:** the pre-existing untracked
  `orb_vocab_stella.fbow` in the canonical checkout (left alone).
- **Scratch / evaluation data (disposable, not committed):**
  `C:\Users\tvllo\Projects\Glasses-scratch\om-runtime\` — repros, timing
  harnesses, `data-inventory\`, `tower-listener\` (incl. ~325 MB of capture
  output the real recorder wrote during listener repros), `teachable\`.
- **Compatibility:** the `since` parameter and the OWLv2 pre-warm are
  additive; no store migration; the observations contract id is unchanged.

---

## 12. Physical device test plan

Windows Tower + iPhone + Ray-Ban Meta glasses. **iOS must be built and
installed from this branch first** — nothing under `ios/` here has been
compiled, and the composed Start/Stop lifecycle is unverified source.

1. Start Tower once, with no environment setup:
   `cd C:\Users\tvllo\Projects\Glasses\tower; .\scripts\start_tower.ps1`.
   Confirm the console prints `--loop tower.serve_loop:resilient_loop_factory`
   and `--timeout-graceful-shutdown 10` in the uvicorn line, and
   `verifier owlv2 ... recording laptop, cell phone, ...` (**fourteen**
   classes; two means the verifier did not load).
2. Connect iOS. Open Object Memory. Do NOT visit Home first.
3. Press Start remembering (once).
4. **Confirm live frames are consumed:** the Tower console shows
   `[Tower][Worker] started object-memory-session ...` then
   `[Tower][ObjectMemory] detector=ssdlite320 device=cuda verifier=owlv2`,
   and within ~10 s the panel reaches "remembering". **This is the
   zero-frame fix under test.** If it sits on "asked to remember, and not
   observed", capture the worker's stderr.
5. Walk ~2 min with a laptop, a cell phone, and a bottle/cup/book in view,
   ≥3 s each. Confirm new observations appear.
6. Query only this recording's memories:
   `GET /object-memory/observations?since=<the session's started_at>` — it
   returns just this walk's records; the un-scoped query still returns all
   history.
7. Pause; verify the producer leaves the process table. Resume; verify new
   observations continue. Stop.
8. **Confirm clean finalization:** the producer prints its report with
   `stopped_because stdin-closed`, `observations_recorded > 0`,
   `keyframes_refused {}`, and the worker exits (the supervisor logs it
   *finished*, not `EXITED 3221225477`).
9. Confirm the new memories from THIS run are visible, and history remains.
10. Confirm the worker exited cleanly and `/health` still answers.
11. Start a SECOND session without restarting Tower; confirm it also
    records. Then exercise disconnect/reconnect (walk out of WiFi range and
    back) and confirm Tower stays reachable — **this is the listener fix
    under test.** After it, `Get-NetTCPConnection -LocalPort 8000 -State
    Listen` must still show a listener and `/health` must answer.
12. Switch Object Memory -> CV Lab -> World Builder -> back to Object
    Memory, all without restarting Tower. Confirm each works and the second
    Object Memory session still records.
13. Check port 8000 is still listening, `/health` 200, and process/RAM are
    stable (no growth across the cycles).

**If something fails, capture:** the full Tower console (the `[Tower]…`
lines and any `asyncio`/traceback), the worker's stderr report block,
`Get-NetTCPConnection -LocalPort 8000 -State Listen`, `Test-NetConnection
127.0.0.1 -Port 8000`, and `Get-Process python | Select Id,WS,CPU`.

---

## 13. Verdict

- **OBJECT MEMORY V1: READY WITH KNOWN LIMITATIONS.** The zero-frame
  deadlock and the exit crash are fixed and guarded; Stop is lossless;
  repeated sessions leak nothing; current-vs-history is now truthful. The
  known limitation is that every claim above is offline — the composed iOS
  Start/Stop is unbuilt source, so the end-to-end product path is proven in
  Tower and owed a device run (§12).
- **LONG-LIVED TOWER: READY WITH KNOWN LIMITATIONS.** The listener survives
  the reset-in-backlog burst that killed it, shutdown is bounded, and a
  dropped socket no longer surfaces as an ASGI error — all reproduced and
  regression-tested. Known limitations: the resilient loop copies a private
  CPython 3.12 method (pinned to this host's version; delete when gh-93821
  ships upstream), and the field shutdown hang's exact stuck connection is
  proven by mechanism but not uniquely identified (both candidates take the
  same fix). The full multi-cartridge switch soak is owed to the device run.
- **USER-TEACHABLE OBJECT MEMORY: PROTOTYPE ONLY (productionization
  deferred, on the evidence).** The architecture is chosen and validated on
  real footage — DINOv2-small crop embeddings + multi-view prototype +
  calibrated threshold, with a working enroll/identify/delete prototype and
  a full persistence/privacy design (§5). It is not wired into V1 and adds
  no runtime dependency, deliberately: cross-session re-ID is only moderate
  (AUC ~0.80, one-in-three false crossings), and the single-instance corpus
  cannot test the core "tell two similar objects apart" promise at all. The
  gate to production is a multi-instance, multi-home, ground-truthed eval
  set plus a local-feature auto-confirm verifier; until then it would ship,
  if built, as suggest-and-confirm, never silent auto-identity. This is a
  decision on the evidence, not a deferral for want of one.
