# All-cartridges MVP integration — plan

**Date:** 2026-09-07.
**Lane:** Mac integration and validation.
**Worktree:** `/Users/tristan/Projects/Glasses-worktrees/all-cartridges-v1`.
**Branch:** `integration/all-cartridges-v1`, created from
`origin/integration/wb-cv-ios-validation-v1 @ b10ab36`.
**Toolchain:** macOS (Darwin 25.5.0), Xcode 26.6 (17F113), Python 3.12.5 in
`~/Projects/Glasses-scratch/all-cartridges-venv` (`.[dev]`; no torch, no
pycolmap, no CUDA on this host).

This plan is derived from the ancestry, the four lane diffs and the four lane
handoffs read in full, not from the mission brief alone. §2 and §3 are
measured, not predicted: the conflict forecast comes from `git merge-tree`
simulations that touched no ref.

---

## 1. Ancestry — the four lanes are NOT independent

The mission brief warned that the lanes might share lineage. They do, and the
shape is not what the handoffs describe from their own side.

```
        ┌── 087dcaa  WB  fix/world-builder-live-history-lifecycle-v1  (10 commits)
        │
  12e4f1e ── 6beaf57 ─┬── 70fade7  OM  fix/object-memory-runtime-v1        (9 commits)
        │             ├── beadb56  DM  feature/document-memory-v1          (10 commits)
        │             └── 6cfe1d5  SU  feature/scene-understanding-v1      (14 commits)
        │
  b10ab36  HEAD  integration/all-cartridges-v1 = origin/integration/wb-cv-ios-validation-v1
```

Established with `git merge-base`, `git log --graph` and branch containment:

| Pair | Merge base |
|---|---|
| OM × WB, DM × WB, SU × WB | `12e4f1e` |
| DM × OM, SU × OM, DM × SU | `6beaf57` |
| every lane × HEAD | `b10ab36` |

**`b10ab36` is the merge base for all four**, so all four are three-way merges
against the same base and none is an ancestor of another. Every lane is ahead
of HEAD and none is behind it.

Two commits sit between HEAD and the lanes and are therefore **common to
several lanes, not overlap**:

- **`12e4f1e`** `fix(world-builder): a rebuild that cannot write no longer ends
  the session` — touches `tower/tower/storage.py`,
  `tower/scripts/world_build_session.py` and two tests. It is an ancestor of
  **all four** lanes, so those four files appear in all four diffs against
  `b10ab36`. That is shared history arriving four times, not four lanes
  editing one file.
- **`6beaf57`** — a docs-only handoff commit, ancestor of OM, DM and SU.

Netting those out leaves the true overlap in §2.

**MVP inventory on the base.** `b10ab36` already carries CV Lab (runtime
controls, live visualization), World Builder (global solver, worlds, geometry
transport, the in-app render viewer), Object Memory (self-contained product
pass), the product shell and the shared Tower/iOS infrastructure; Document
Memory and Scene Understanding exist on it as complete-but-ungated cartridges,
which is exactly what the DM and SU lanes rewrite. The only remote branch not
already merged into HEAD, besides the four lanes, is
`origin/rescue/canonical-worktree-snapshot-2026-08-29`, which is a rescue
snapshot and not an MVP lane. **The four lanes plus this base are the intended
current MVP.**

---

## 2. True cross-lane overlap

Files touched by two or more lanes, after removing `12e4f1e` and `6beaf57`:

| File | Lanes | What collides |
|---|---|---|
| `tower/scripts/object_memory_session.py` | **WB, OM** | Both fix the stdin stop-watcher. WB (`f49d73d`) switches `wait_for_close` to a raw `os.read(fd, 1)`; OM (`c85effc`) adds `_prewarm_native_libraries()` and `_run_and_exit()`/`os._exit`. **Complementary, both must survive.** Only the `import os` line is a textual collision. |
| `tower/tower/cartridge_runtime.py` | **DM, SU** | Both edit `_document_session` and `build_live_cartridges`; SU also adds ~50 lines of watcher tracking to `LiveCartridges`. Highest semantic risk in the merge. |
| `tower/tower/config.py` | **DM, SU** | Both add `Settings` fields and `get_settings` parsing; both add module-level defaults. Adjacent hunks around `get_settings`. |
| `tower/tower/main.py` | WB, DM, SU | WB rewrites the worker spec and adds the World Builder gate; DM and SU add boot-log lines and route wiring. Hunks are close in `_log_effective_configuration` (WB @339, SU @343, DM @354). |
| `tower/tower/results/registry.py` | **DM, SU** | Adjacent: SU rewords `SCENE_DISABLED_REASON`, DM edits `DOCUMENT_DISABLED_REASON` four lines below. |
| `tower/tower/routes/results_ws.py` | **OM, SU** | OM stops swallowing `WebSocketDisconnect` in `handle()`; SU adds watcher tracking to `ChannelHolder` and the subscribe/unsubscribe hooks. Different regions, one adjacent import block. |
| `tower/tower/results/contracts.py` | WB, DM | Disjoint hunks (WB @86, DM @107). |
| `tower/tests/test_result_channel_protocol.py`, `test_result_channel_isolation.py` | DM, SU | Both make the same class of change: switch a cartridge off so an isolation test does not wait forever for a refusal an now-available cartridge never sends. |
| `docs/contracts/TOWER-UNIFIED-CARTRIDGES.md`, `tower/docs/contracts/CARTRIDGE-RESULTS.md` | WB, DM, SU | Three lanes documenting three cartridges in one file. |
| `ios/GlassesTests/TowerClientTests.swift` | WB, DM | Both add cases. |

**Object Memory touches no `ios/` file at all.** Its two client-facing changes
(the composed lifecycle and `since`) are additive and were left for this lane
to consume.

---

## 3. Merge order, and why

Simulated with `git merge-tree --write-tree` (dangling objects only; no refs
created, no worktree change):

| Order | Conflicts |
|---|---|
| **A. WB → OM → DM → SU** | 0, 0, 0, **4** — total 4 |
| B. WB → OM → SU → DM | 0, 0, 1, 4 — total 5 |
| C. OM → WB → DM → SU | 0, 0, 0, 4 — total 4 |

**Chosen: order A — WB, then OM, then DM, then SU.**

1. **WB first.** It is the deepest change to shared infrastructure: a new
   `tower/tower/process_ownership.py`, one-owned-process spawning, per-worker
   Job Objects, `request_stop`/`stop_policy` on the supervisor and the
   cartridge session, and the World Builder gate in `main.py`. Every later
   lane sits on top of that process model rather than being merged under it.
2. **OM second.** It is the smallest lane and it is the other half of the
   `object_memory_session.py` collision. Merging it directly after WB puts the
   two competing producer fixes in one place while that context is fresh, and
   its listener/WebSocket changes land before two more cartridges start
   subscribing.
3. **DM third**, **SU last.** SU's own handoff says it edits `_document_session`
   in `cartridge_runtime.py`; DM owns that function. Landing DM first and SU
   last means SU is reconciled *against* DM's finished version rather than the
   reverse, and it concentrates all four conflicts in the final merge, where
   the integrator has every other lane already in hand. Order C is equally
   clean textually but merges OM's listener fix before the process model it
   runs under, so A is preferred.

Every merge is `git merge --no-ff <branch>`, preserving all four lane
histories. Nothing is squashed and no lane is rebased.

---

## 4. Contract interactions

| Contract | Lane | Change | Compatibility |
|---|---|---|---|
| `world_builder.status/2026-09-06` | WB | **New identifier**, replacing `…/2026-08-25`. Adds lifecycle `finalizing`/`interrupted`, `model_state.interrupted`, `lifecycle.finalization`, top-level `selection`. | **Non-additive at the identifier.** An app built before `0ade6a5` shows "a contract this app does not understand". Both halves must ship together — they do, in this branch. |
| `GET /worlds` sessions | WB | `state`, `keyframes_journaled`, `finalization` | Additive |
| `object_memory.observations/2026-08-26` | OM | `since` query parameter and echoed field | Additive, identifier unchanged |
| `document_memory.*` | DM | Identifiers moved; `sightings[]`, page `visual_hash`/`box_count`/`readable`, `library.revision` | Identifier moved — the phone must be built from this branch |
| `scene_understanding.live/2026-08-27` | SU | Side counts for people, apparent-size buckets, `partial_bottom_edge`, orientation method/status/validation, `demand{}` | **Additive under an unchanged identifier** |

**The audit this lane owes:** three cartridges changed the wire and two moved
their identifier. `ios/scripts/contract-drift-check.py` must agree on all five
contracts against a Tower built from this branch, and every Swift decoder must
tolerate the additive fields. Enum decoding is the specific risk: a new Tower
value (`interrupted`, a new `selection.mode`, a new orientation status) must
not throw in an older decoder path.

## 5. Camera, stream and worker ownership — the interactions to prove

The product rule is: Tower starts once; the wearer moves between cartridges
without restarting anything; a generic camera stream must not start an
unrelated cartridge; leaving a cartridge relinquishes what it no longer needs.

Each lane changed a different part of that rule, and **no lane could test the
combination**:

- **WB** gates the builder behind a `world_builder` `CartridgeSession`, so the
  builder attaches only while that cartridge is active. It also makes each
  worker one owned process in its own Job Object.
- **OM** keeps the existing `CartridgeSession` state machine and fixes the
  producer that sat inside it, plus the listener that a reconnect storm could
  kill.
- **DM** gives Document Memory a managed root and a capability that is on by
  default when torch is present; sessions are still started by a person, and an
  idle session self-stops ten minutes after its stream closes.
- **SU** makes Scene Understanding demand-driven: it runs only while a phone
  streams **and** a client watches, and releases the GPU when the last watcher
  leaves.

**Four different activation disciplines now coexist on one Tower.** The
integration questions this lane must answer, none of which any lane could ask
alone:

1. With all four merged and a stock config, does a phone streaming for CV Lab
   start the World Builder builder, the Document Memory OCR, or the Scene
   detector? (Each lane says no for itself. The combination is untested.)
2. Do DM's and SU's capability defaults both flip to available on a stock
   Tower, and do the isolation tests that each lane patched for its own
   cartridge still hold once **both** are available?
3. Does WB's `request_stop` / `stop_policy` change alter how DM's and SU's
   in-process sessions are stopped, or only subprocess workers?
4. Does SU's `ChannelHolder` watcher tracking survive OM's change to the same
   file's disconnect handling — specifically, can a disconnect now propagate
   through a path that decrements a watcher count twice, or not at all?
5. Can leaving Document Memory or Scene Understanding leave a worker,
   a model, or a subscription behind that blocks the next cartridge?

## 6. iOS interactions

Three lanes wrote Swift on Windows that has **never been compiled**: WB's
`0ade6a5` (~3,000 lines), DM's `f728e88`, SU's `c431969`. This Mac lane is
their first Xcode validation, and the project's own trap is documented:
`SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor` silently makes value types —
including synthesized `Equatable` and memberwise inits — main-actor isolated,
so a pure wire type that crosses an actor boundary needs `nonisolated`.

The specific checks: each new screen's camera start/stop symmetry; each new
screen's subscribe on appear and unsubscribe on disappear; whether two screens
can hold the capture session at once; whether a `Task` can outlive its view
model; and whether the three lanes' cartridge registrations collide in the app
shell. The Xcode project uses file-system-synchronized groups, so new Swift
files need no `pbxproj` edit — which must be confirmed, not assumed.

## 7. Validation strategy

Ordered, and each gate must pass at the **final** HEAD, not an earlier one.

1. **Per-merge.** After each of the four merges: inspect the merge result,
   resolve conflicts, look for semantic conflicts even where git was silent,
   run that lane's targeted Tower tests, and do not proceed while knowingly
   broken. Integration-only fixes commit separately.
2. **Tower baseline attribution.** The suite was run on the unmerged base
   `b10ab36` on this Mac first, so every failure after merging can be
   attributed to the merge or to this host. macOS-environment failures (no
   torch, no CUDA, no PowerShell, no NTFS junctions, no `tower/data` corpus)
   are reported as skipped or environmental **honestly** — a Windows-only
   regression test that does not run here is not evidence of anything.
3. **iOS.** Debug build, Release build, `GlassesTests`, the `GlassesUITests`
   smoke against a Tower from this branch, and the three `ios/scripts`
   checkers. Compiler errors are integration defects and get fixed here.
4. **Cross-cartridge lifecycle.** One long-lived Tower driven through
   idle → CV Lab → idle → World Builder → finalizing/History → Object Memory →
   idle → Document Memory → idle → Scene Understanding → idle → CV Lab, plus
   the direct transitions the app permits, asserting at each step: Tower alive,
   `/health` answering, listener bound, no unrelated cartridge started, the
   previous cartridge's workers/models/subscriptions released, no duplicate
   watchers or camera owners, bounded Stop, and no growth in threads, RSS or
   child processes. The repository already has
   `scripts/cartridge_switch_soak.py` (WB), `scripts/scene_soak.py` (SU) and
   `scripts/unified_cartridge_smoke.py`; extend those rather than writing a
   brittle new synthetic soak.
5. **Adversarial review.** Independent read-only reviewers on the merged tree
   for Tower lifecycle, Swift concurrency, camera ownership, wire contracts,
   privacy and truthfulness, resource teardown, and test quality. Findings are
   reproduced from code before being fixed; speculative preferences without
   evidence are not implemented.
6. **Re-validate at the final HEAD** and report only those numbers.

## 8. Reviewer gates

- No merge proceeds while the previous one is knowingly broken.
- A clean textual merge is not evidence of a correct semantic merge; §5's five
  questions must be answered from code or a test, not from the handoffs.
- Truthfulness is a gate, not a preference: historical data must not be
  presentable as current-session output, interrupted must not look complete,
  unavailable capability must not look empty, stale geometry must not look live.
- The privacy constraints on Scene Understanding are hard requirements: no face
  recognition, no biometric identity, no person naming, no persistent person
  profiles, no cross-session re-identification, no inferred gaze, intent or
  emotion. Transient face processing for anonymous orientation is allowed only
  while the raw information is discarded and the claim stays conservative.
- Object Memory's teachable-instance prototype stays research. It is not
  productionized in this lane.
- The verdict at the end is **READY FOR PHYSICAL VALIDATION** or **NOT READY**.
  No physical device is attached in this mission and nothing may claim otherwise.

## 9. Product limitations to carry forward unchanged

- **World Builder** produces sparse structure-from-motion, not a dense or
  recognizable room reconstruction. Dense reconstruction is the next R&D phase
  and is not in this lane.
- **Object Memory** does not ship user-taught instance identity. The prototype
  exists and its evidence was judged insufficient; it stays documented research.
- **Document Memory** has never read a physical page through the glasses. Every
  threshold was calibrated on replay and synthetic positives.
- **Scene Understanding** perception is limited and orientation is experimental.
  "Appears to face the wearer's direction" is not gaze, attention or intent.

## 10. Temporary resources (filesystem policy rule 9)

- Worktree `Glasses-worktrees/all-cartridges-v1` (this branch). Persistent.
- `Glasses-scratch/all-cartridges-venv` — the Python 3.12 venv for this
  worktree. Disposable.
- `Glasses-scratch/ac-tmp/` — pytest basetemps and soak roots. Disposable.
- Session scratchpad under `/private/tmp/claude-501/…/scratchpad` — logs and
  derived reports. OS temp.
- Nothing is written to the canonical checkout, to `tower/data`, or outside
  `Projects/`.
