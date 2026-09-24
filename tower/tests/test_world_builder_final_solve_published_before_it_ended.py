"""Review V11, MED-A: the builder's final-solve child ended AFTER it had published.

A consensus final solve (`TOWER_WORLD_SOLVE_CONSENSUS` >= 2, seeded, gated) publishes its
draw 0 FIRST -- `consensus-deferred`, retryable, owed -- and only then maps its further
draws (`global_solve._publish_draw_0_first`). In the builder that solve is a CHILD
(`world_solve.py`), and a hard stop during those draws TERMINATES it
(`world_build_session.run_final`). Until P3.8 the builder then recorded `final_solve:
skipped`, "... the last background solution stands" -- false, draw 0 was published -- and
the finisher's `assess` answered `no-final-solve` for every final solve that is not
`solved`, before ever asking about the re-gate. So the owed consensus never ran, the room
was never built, and the row said something false. A child that crashed after the early
publish (exit 1, no summary) was the same hole through `failed`.

Pinned here:

* THE BUILDER, end to end through `main`, with a REAL child process (`--solve-script`)
  that publishes and is then terminated by a hard stop (or exits 1): the row says `solved`
  with the published record's notice and detail, and the finisher runs the consensus and
  builds the room. A child that published nothing, a solution solved before the child was
  launched, an ungated solve and a single draw (N = 1) keep today's row exactly.
* THE FINISHER, for sessions ALREADY in the old state (`stale_final_solve`): assessed as
  solved -- the owed consensus first -- and the row put right under the lock. Every other
  `skipped` / `failed` row (a `--skip-solve` repair, N = 1, ungated, unloadable) takes
  today's path.

The GPU stages and the gate are faked; the solutions are real (`write_solution`).
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_build_session as wbs  # noqa: E402
from scripts import world_finish_pending as wfp  # noqa: E402
from scripts.world_build_session import StopRequest  # noqa: E402
from tests.test_world_builder_finish_pending import _finalization, _stage, _world  # noqa: E402
from tests.test_world_builder_finish_pending_gate import _at_the_bound, _solution  # noqa: E402
from tower.world_builder import coherence_publish as CP  # noqa: E402
from tower.world_builder.engine import WorldBuilderEngine  # noqa: E402
from tower.world_builder.global_solve import (  # noqa: E402
    load_solution,
    workspace_for,
    write_solution,
)
from tower.world_builder.records import (  # noqa: E402
    FINAL_SOLVE_FAILED,
    FINAL_SOLVE_SKIPPED,
    FINAL_SOLVE_SOLVED,
    FINALIZATION_COMPLETE,
    STAGE_APPEARANCE,
    STAGE_STATE_OK,
    STAGE_STATE_STOPPED,
    STAGE_SURFACE,
)
from tower.world_builder.store import WorldStore  # noqa: E402

TOWER_ROOT = Path(wbs.__file__).resolve().parents[1]

# What the EARLY publish leaves (`gate_by_consensus`'s `stopped` branch with
# `why_deferred=WHY_PUBLISHED_FIRST`): an attached room, owed its consensus.
EARLY = {"state": CP.GATE_STATE_APPLIED, "attach": True, "masks_applied": True,
         "metric_available": True, "retryable": True, "cause": CP.CAUSE_CONSENSUS_DEFERRED,
         "depth": {"state": "ok"},
         "consensus": {"record": CP.CONSENSUS_RECORD, "state": CP.CONSENSUS_DEFERRED,
                       "requested": 3, "why": CP.WHY_PUBLISHED_FIRST}}
# The full consensus, published over it once the further draws voted.
FULL = dict(EARLY, retryable=False, cause=None,
            consensus=dict(EARLY["consensus"], state=CP.CONSENSUS_APPLIED, why=None))
# A single draw (N = 1): a gate record with no `consensus` block.
SINGLE = {"state": CP.GATE_STATE_APPLIED, "attach": True, "masks_applied": True,
          "metric_available": True, "retryable": False, "cause": None, "depth": {"state": "ok"}}
# A single draw whose depth stage could not finish: retryable, but not a consensus.
SINGLE_RETRYABLE = dict(SINGLE, retryable=True, cause=CP.CAUSE_DEPTH_UNAVAILABLE,
                        metric_available=False,
                        depth={"state": "unavailable", "detail": "CUDA out of memory"})

TERMINATED = ("final solve terminated: hard stop (stdin-closed) during finalization; "
              "the last background solution stands")
CRASHED = "final solve failed: world_solve.py exited 1"


@pytest.fixture(autouse=True)
def _no_native_warm(monkeypatch):
    monkeypatch.setattr(wfp, "prewarm_world_builder", lambda *a, **k: ())
    monkeypatch.setattr(wbs, "prewarm_world_builder", lambda *a, **k: ())


def _fin(store, world_id="w1", session_id="s1") -> dict:
    return store.read_session(world_id, session_id).finalization


# ---------------------------------------------------------------------------
# the builder: a real final-solve child that publishes, then is ended
# ---------------------------------------------------------------------------

# The child `run_final` spawns (`--solve-script`). It publishes what its plan says -- a
# real `write_solution` over the session's own keyframes -- says it is ready, and then
# either waits to be terminated, or exits 1 without a summary (a crash).
FAKE_CHILD = r'''
import json, sys, time
sys.path.insert(0, {tower!r})
import numpy as np
from tower.world_builder.global_solve import Solution, workspace_for, write_solution
from tower.world_builder.store import WorldStore

args = sys.argv[1:]
arg = lambda name: args[args.index(name) + 1]
plan = json.loads(open({plan!r}, encoding="utf-8").read())
store = WorldStore(arg("--root"))
world_id, session_id = arg("--world"), arg("--session")
workspace = workspace_for(store, world_id, session_id)
workspace.root.mkdir(parents=True, exist_ok=True)
workspace.database_path.write_bytes(b"the walk's features (not read: the gate is faked)")
if plan["publish"]:
    kids = [k.keyframe_id for k in store.read_keyframes(world_id, session_id)]
    n = len(kids)
    write_solution(workspace, Solution(
        solver="glomap", solved_at=time.time() if plan["fresh"] else 1.0,
        input_digest="digest", keyframe_ids=kids,
        poses={{kid: {{"component": 0, "rotation": np.eye(3).reshape(-1).tolist(),
                       "translation": [0.1 * i, 0.0, 0.0], "observations": 40}}
               for i, kid in enumerate(kids)}},
        components=[{{"index": 0, "images": n, "points": n}}],
        xyz=np.zeros((n, 3), np.float32), rgb=np.zeros((n, 3), np.uint8),
        component=np.zeros(n, np.int32), first_keyframe=np.arange(n, dtype=np.int32),
        track_length=np.full(n, 2, np.int32), error=np.full(n, 0.5, np.float32),
        observations=np.array([[i, 0, i] for i in range(n)], np.int32).reshape(-1, 3),
        timing={{"final": True}}, transients={{"state": "applied"}}, gate=plan["gate"]))
(workspace.root / "fake-child.ready").write_text("ready", encoding="utf-8")
if plan["then"] == "crash":
    sys.exit(1)
time.sleep(120)
'''


def _build(tmp_path, monkeypatch, *, publish: bool, gate=None, fresh: bool = True,
           then: str = "stop"):
    """A synthetic walk through the builder's real `main` and its real `run_final`, whose
    final-solve child is FAKE_CHILD. `then="stop"`: a hard stop arrives once the child is
    ready (after its publish), and the builder terminates it. `then="crash"`: the child
    exits 1 on its own. Returns (store, world_id, session_id, the printed report)."""
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"publish": publish, "gate": gate, "fresh": fresh,
                                "then": then}), encoding="utf-8")
    script = tmp_path / "fake_world_solve.py"
    script.write_text(FAKE_CHILD.format(tower=str(TOWER_ROOT), plan=str(plan)),
                      encoding="utf-8")
    root = tmp_path / "wb"
    monkeypatch.setattr(wbs.StopRequest, "install", lambda self, **k: None)
    real_hard = wbs.StopRequest.hard_asked_for

    def hard_asked_for(self):
        # The supervisor's hard stop, the moment the child has published: exactly the
        # window MED-A is about (draws 1..N-1 being mapped).
        if then == "stop" and self.level is None and any(root.rglob("fake-child.ready")):
            self.request(StopRequest.HARD, "stdin-closed")
        return real_hard(self)

    monkeypatch.setattr(wbs.StopRequest, "hard_asked_for", hard_asked_for)
    if then == "crash":
        # No stop, so the room would really be built (GPU): recorded as not built instead.
        # On a stop the builder's own `final_surface_stages` runs, and skips on the stop.
        def no_room(store, world_id, session_id, *, record, **kwargs):
            record(STAGE_SURFACE, state=STAGE_STATE_STOPPED, detail="not built in this test")
            return {}

        monkeypatch.setattr(wbs, "final_surface_stages", no_room)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = wbs.main(["--synthetic", "--synthetic-frames", "10", "--root", str(root),
                         "--solve", "--solve-every", "0", "--solve-script", str(script),
                         "--surface", "--format", "json"])
    assert code == 0
    store = WorldStore(root)
    (world_id,) = store.list_world_ids()
    (session_id,) = store.list_session_ids(world_id)
    return store, world_id, session_id, json.loads(out.getvalue())


def _published_summary(store, world_id, session_id) -> dict:
    solution = load_solution(store, world_id, session_id)
    return {"gate": solution.gate, "transients": solution.transients}


def test_a_hard_stop_after_the_early_publish_records_the_published_solve(tmp_path,
                                                                          monkeypatch):
    """RED at 315b6bf: `skipped`, "the last background solution stands", no notice, and
    `assess` -> `no-final-solve`."""
    store, w, s, report = _build(tmp_path, monkeypatch, publish=True, gate=EARLY)
    fin = _fin(store, w, s)
    summary = _published_summary(store, w, s)
    assert fin["state"] == FINALIZATION_COMPLETE
    assert fin["final_solve"] == FINAL_SOLVE_SOLVED, fin
    # The CONSENSUS-DEFERRED sentence: the phone's copy and its diagnostic twin.
    assert fin["notice"] == CP.publish_notice(summary) == \
        CP.NOTICE_SENTENCES[CP.CAUSE_CONSENSUS_DEFERRED]
    assert fin["detail"] == CP.publish_detail(summary)
    assert "background solution stands" not in fin["detail"]
    # The child really was terminated by the stop, after it had published.
    solve = report["global_solve"]
    assert solve["interrupted"] is True and solve["published_before_it_ended"] is True
    assert solve["final_solve"] == FINAL_SOLVE_SOLVED
    # The room is owed (a hard stop skips it), and the finisher owes the consensus first.
    session = store.read_session(w, s)
    assert session.stages[STAGE_SURFACE]["state"] == STAGE_STATE_STOPPED
    verdict = wfp.assess(store, w, s)
    assert verdict.owed and verdict.code == "owed-regate", verdict


def test_the_finisher_then_runs_the_consensus_and_builds_the_room(tmp_path, monkeypatch):
    store, w, s, _ = _build(tmp_path, monkeypatch, publish=True, gate=EARLY)
    rooms = []

    def regate(store_, world_id, session_id, should_stop=None):
        solution = load_solution(store_, world_id, session_id)
        write_solution(workspace_for(store_, world_id, session_id),
                       replace(solution, gate=FULL, solved_at=solution.solved_at + 1.0))
        return {"notice": None, "detail": None}

    def surface_stages(store_, world_id, session_id, **kwargs):
        # What the real one records with the appearance off.
        rooms.append((world_id, session_id, kwargs["solved"]))
        kwargs["record"](STAGE_APPEARANCE, state="unavailable", attempted=False,
                         detail="not requested")
        kwargs["record"](STAGE_SURFACE, state=STAGE_STATE_OK, detail=None)
        return {"surface": {"attempted": True, "state": STAGE_STATE_OK}}

    monkeypatch.setattr(WorldBuilderEngine, "build",
                        lambda self, world_id, session_id: SimpleNamespace(poses_solved=10))
    out = wfp.finish_regate(store, wfp.assess(store, w, s), appearance=False,
                            prune_depth_work=False, stop_request=StopRequest(), regate=regate,
                            surface_stages=surface_stages)
    assert out["finished"] is True, out
    assert rooms == [(w, s, True)]
    fin = _fin(store, w, s)
    assert fin["final_solve"] == FINAL_SOLVE_SOLVED and "notice" not in fin
    assert wfp.assess(store, w, s).code == "nothing-interrupted"


def test_a_child_that_crashes_after_the_early_publish_is_recorded_solved_too(tmp_path,
                                                                             monkeypatch):
    """The same hole through the error branch: exit 1 with no summary used to be
    `failed`, "world_solve.py exited 1", over a published draw 0 owed its consensus."""
    store, w, s, report = _build(tmp_path, monkeypatch, publish=True, gate=EARLY,
                                 then="crash")
    fin = _fin(store, w, s)
    assert fin["final_solve"] == FINAL_SOLVE_SOLVED, fin
    assert fin["notice"] == CP.NOTICE_SENTENCES[CP.CAUSE_CONSENSUS_DEFERRED]
    assert report["global_solve"]["error"] == "world_solve.py exited 1"
    assert report["global_solve"]["published_before_it_ended"] is True
    assert wfp.assess(store, w, s).code == "owed-regate"


@pytest.mark.parametrize("case", ["published-nothing", "solved-before-the-launch",
                                  "ungated", "single-draw"])
def test_a_stop_that_ended_a_child_with_nothing_of_its_own_published_is_todays_row(
        tmp_path, monkeypatch, case):
    """Nothing this child published, or nothing that is a consensus: `skipped`, today's
    sentence word for word, no notice, and the finisher leaves it alone."""
    kwargs = {"published-nothing": dict(publish=False),
              "solved-before-the-launch": dict(publish=True, gate=EARLY, fresh=False),
              "ungated": dict(publish=True, gate=None),
              "single-draw": dict(publish=True, gate=SINGLE)}[case]
    store, w, s, report = _build(tmp_path, monkeypatch, **kwargs)
    fin = _fin(store, w, s)
    assert fin["final_solve"] == FINAL_SOLVE_SKIPPED
    assert fin["detail"] == TERMINATED, fin
    assert "notice" not in fin
    assert report["global_solve"] == {"attempted": True, "solved": False, "interrupted": True,
                                      "seconds": report["global_solve"]["seconds"],
                                      "background_launches": 0,
                                      "final_solve": FINAL_SOLVE_SKIPPED}
    # Not asked of "solved-before-the-launch": a gated consensus solve from BEFORE the
    # child cannot exist for a builder session (its only final solve is this one), so the
    # finisher, which has no launch time to compare, takes one under this row as the
    # child's -- the backstop for a clock that stepped back.
    if case != "solved-before-the-launch":
        assert wfp.assess(store, w, s).code == "no-final-solve"


@pytest.mark.parametrize("gate", [None, SINGLE], ids=["ungated", "single-draw"])
def test_a_child_without_a_consensus_that_crashes_is_todays_failed_row(tmp_path, monkeypatch,
                                                                       gate):
    store, w, s, _ = _build(tmp_path, monkeypatch, publish=True, gate=gate, then="crash")
    fin = _fin(store, w, s)
    assert fin["final_solve"] == FINAL_SOLVE_FAILED
    assert fin["detail"] == CRASHED and "notice" not in fin


# ---------------------------------------------------------------------------
# `published_consensus_solve`: what identifies the child's publish
# ---------------------------------------------------------------------------


def _published(tmp_path, gate, *, solved_at=5.0, loadable=True):
    store = _world(tmp_path, stages={})
    workspace = workspace_for(store, "w1", "s1")
    if loadable:
        write_solution(workspace, _solution("s1", gate, solved_at=solved_at))
    else:
        workspace.root.mkdir(parents=True, exist_ok=True)
        workspace.solution_path.write_text(json.dumps({"gate": gate, "solved_at": solved_at}))
    return store


@pytest.mark.parametrize("gate,since,loadable,found", [
    (EARLY, None, True, True),
    (EARLY, 5.0, True, True),                  # solved AT the launch instant counts
    (EARLY, 5.5, True, False),                 # solved before the child was launched
    (FULL, 4.0, True, True),                   # the full consensus, ended before its summary
    (SINGLE, None, True, False),               # N = 1: never asked about
    (SINGLE_RETRYABLE, None, True, False),
    (None, None, True, False),                 # ungated: a background solve, or no gate
    (dict(EARLY, consensus=dict(EARLY["consensus"], requested=1)), None, True, False),
    (dict(EARLY, consensus=dict(EARLY["consensus"], requested=True)), None, True, False),
    (EARLY, None, False, False),               # torn: `solution.json` without its arrays
])
def test_published_consensus_solve(tmp_path, gate, since, loadable, found):
    store = _published(tmp_path, gate, loadable=loadable)
    got = wbs.published_consensus_solve(store, "w1", "s1", since=since)
    assert (got is not None) == found, got
    if found:
        assert got["gate"] == gate and got["solved_at"] == 5.0
        assert got["transients"] == {"state": "applied"}


def test_no_solution_at_all_is_nothing_published(tmp_path):
    assert wbs.published_consensus_solve(_world(tmp_path, stages={}), "w1", "s1") is None


# ---------------------------------------------------------------------------
# the finisher: sessions ALREADY in the old state heal (`stale_final_solve`)
# ---------------------------------------------------------------------------


def _legacy(tmp_path, *, final_solve=FINAL_SOLVE_SKIPPED, detail=TERMINATED, gate=EARLY,
            stages=None, loadable=True, database=True):
    """The on-disk state a 315b6bf builder left: its row, over the early publish."""
    store = _world(tmp_path, stages={} if stages is None else stages,
                   finalization=dict(_finalization(FINALIZATION_COMPLETE, final_solve),
                                     detail=detail))
    workspace = workspace_for(store, "w1", "s1")
    workspace.root.mkdir(parents=True, exist_ok=True)
    if loadable:
        write_solution(workspace, _solution("s1", gate))
    elif gate is not None:
        workspace.solution_path.write_text(json.dumps({"gate": gate}))
    if database:
        workspace.database_path.write_bytes(b"x")
    return store


@pytest.mark.parametrize("final_solve,detail", [
    (FINAL_SOLVE_SKIPPED, TERMINATED),
    (FINAL_SOLVE_SKIPPED, "final solve terminated: hard stop (signal) during finalization; "
                          "the last background solution stands"),
    (FINAL_SOLVE_FAILED, CRASHED),
    (FINAL_SOLVE_FAILED, "final solve failed: world_solve.py exited 3221225786"),
])
def test_a_legacy_row_over_the_early_publish_is_owed_its_consensus(tmp_path, final_solve,
                                                                   detail):
    """RED at 315b6bf (the RV11 probe's state): `no-final-solve`."""
    store = _legacy(tmp_path, final_solve=final_solve, detail=detail)
    v = wfp.assess(store, "w1", "s1")
    assert v.owed and v.code == "owed-regate", v
    assert wfp.stale_final_solve(store, "w1", "s1", _fin(store)) is not None


@pytest.mark.parametrize("case", ["skip-solve-repair", "solved-by-no-child", "single-draw",
                                  "single-draw-retryable", "ungated", "no-solution",
                                  "unloadable", "failed-for-another-reason",
                                  "a-sentence-with-more-after-it", "not-complete"])
def test_every_other_skipped_or_failed_row_is_todays_no_final_solve(tmp_path, case):
    kwargs = {
        "skip-solve-repair": dict(detail="final solve skipped: --skip-solve"),
        "solved-by-no-child": dict(detail="final solve skipped: hard stop (stdin-closed) "
                                          "while observing"),
        "single-draw": dict(gate=SINGLE),
        "single-draw-retryable": dict(gate=SINGLE_RETRYABLE),
        "ungated": dict(gate=None),
        "no-solution": dict(gate=None, loadable=False),
        "unloadable": dict(loadable=False),
        "failed-for-another-reason": dict(final_solve=FINAL_SOLVE_FAILED,
                                          detail="final solve failed: RuntimeError: x"),
        "a-sentence-with-more-after-it": dict(detail=TERMINATED + "; and more"),
        "not-complete": {},
    }[case]
    store = _legacy(tmp_path, **kwargs)
    if case == "not-complete":
        session = store.read_session("w1", "s1")
        store.write_session(replace(session, finalization=dict(session.finalization,
                                                               state="interrupted")))
    # The same answer as at 315b6bf (this test passes there too).
    v = wfp.assess(store, "w1", "s1")
    assert not v.owed and v.code == ("not-finalized" if case == "not-complete"
                                     else "no-final-solve"), v


def _regate_to(gate):
    calls = []

    def regate(store_, world_id, session_id, should_stop=None):
        calls.append(world_id)
        write_solution(workspace_for(store_, world_id, session_id),
                       _solution(session_id, gate, solved_at=6.0))
        return {"notice": CP.publish_notice({"gate": gate, "transients": {"state": "applied"}}),
                "detail": None}

    regate.calls = calls
    return regate


@pytest.fixture
def stage_runner(monkeypatch):
    calls = []

    def fake(store, world_id, session_id, **kwargs):
        calls.append(kwargs["solved"])
        kwargs["record"](STAGE_SURFACE, state=STAGE_STATE_OK, detail=None)
        if kwargs.get("appearance"):
            kwargs["record"](STAGE_APPEARANCE, state=STAGE_STATE_OK, detail=None)
        return {"surface": {"attempted": True, "state": STAGE_STATE_OK}}

    monkeypatch.setattr(wfp, "final_surface_stages", fake)
    return calls


def _run(root) -> int:
    return wfp.main(["--root", str(root), "--format", "json"], stop_request=StopRequest())


def test_the_finisher_heals_a_legacy_row_end_to_end(tmp_path, monkeypatch, stage_runner):
    """Through the finisher's own entry point: the consensus runs, the room is built from
    it, and the row says what the builder now writes -- then nothing is owed."""
    store = _legacy(tmp_path, stages={STAGE_SURFACE: _stage(STAGE_STATE_STOPPED)})
    regate = _regate_to(FULL)
    monkeypatch.setattr(CP, "regate_published", regate)
    monkeypatch.setattr(WorldBuilderEngine, "build",
                        lambda self, world_id, session_id: SimpleNamespace(poses_solved=4))
    assert _run(tmp_path) == 0
    assert regate.calls == ["w1"] and stage_runner == [True]
    fin = _fin(store)
    assert fin["final_solve"] == FINAL_SOLVE_SOLVED and "notice" not in fin
    assert fin["detail"] is None
    assert wfp.assess(store, "w1", "s1").code == "nothing-interrupted"


def test_the_row_is_put_right_before_the_re_gate(tmp_path, monkeypatch):
    """Healed under the lock first, so a re-gate that a stop reached at draw 0 (it writes
    nothing, MED-1b) leaves a TRUE row -- `solved`, owed its consensus -- still owed."""
    store = _legacy(tmp_path)

    def stopped(store_, world_id, session_id, should_stop=None):
        assert _fin(store_)["final_solve"] == FINAL_SOLVE_SOLVED, "healed before the re-gate"
        return {"stopped": True, "publish": {"written": False}}

    out = wfp.finish_regate(store, wfp.assess(store, "w1", "s1"), appearance=False,
                            prune_depth_work=False, stop_request=StopRequest(), regate=stopped)
    assert out["final_solve_healed"]["was"] == FINAL_SOLVE_SKIPPED
    fin = _fin(store)
    summary = {"gate": EARLY, "transients": {"state": "applied"}}
    assert fin["final_solve"] == FINAL_SOLVE_SOLVED
    assert fin["notice"] == CP.publish_notice(summary)
    assert fin["detail"] == CP.publish_detail(summary)
    assert wfp.assess(store, "w1", "s1").code == "owed-regate"


def test_a_heal_that_cannot_be_written_still_leaves_the_re_gate_solved(tmp_path, monkeypatch,
                                                                       stage_runner):
    """The heal is never fatal; the re-gate's own re-mark keeps `solved`, because the
    re-gate rewrites the detail the row was recognised by."""
    store = _legacy(tmp_path)

    def refuse(*a, **k):
        raise PermissionError(13, "session.json is read-only for a moment")

    monkeypatch.setattr(wfp, "_heal_final_solve", refuse)
    monkeypatch.setattr(WorldBuilderEngine, "build",
                        lambda self, world_id, session_id: SimpleNamespace(poses_solved=4))
    out = wfp.finish_regate(store, wfp.assess(store, "w1", "s1"), appearance=False,
                            prune_depth_work=False, stop_request=StopRequest(),
                            regate=_regate_to(FULL))
    assert out["finished"] is True and "final_solve_heal_error" in out, out
    assert stage_runner == [True]
    assert _fin(store)["final_solve"] == FINAL_SOLVE_SOLVED


def test_a_legacy_row_given_up_at_the_bound_is_solved_and_its_room_is_asked(tmp_path,
                                                                           stage_runner):
    """`_retire_regate` replaces the detail the row was recognised by; it writes `solved`
    too, so the room behind the given-up re-gate is still asked (M1b)."""
    store = _legacy(tmp_path, stages={STAGE_SURFACE: _stage(STAGE_STATE_STOPPED)})
    _at_the_bound(store)
    v = wfp.assess(store, "w1", "s1")
    assert v.exhausted and v.stage == wfp.REGATE_STAGE, v
    assert _run(tmp_path) == 0
    fin = _fin(store)
    assert fin["final_solve"] == FINAL_SOLVE_SOLVED
    assert "stopped trying" in fin["notice"]
    assert stage_runner == [True], "the room behind the given-up re-gate was built"


def test_a_row_the_builder_wrote_solved_is_not_touched(tmp_path):
    """The heal only ever rewrites a row it recognises."""
    store = _legacy(tmp_path, final_solve=FINAL_SOLVE_SOLVED, detail=None)
    before = store.session_path("w1", "s1").read_bytes()
    assert wfp._heal_final_solve(store, WorldBuilderEngine(store), "w1", "s1") is None
    assert store.session_path("w1", "s1").read_bytes() == before
