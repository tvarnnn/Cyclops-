"""The Tower producer of `finalization.notice` (contract WORLD-BUILDER-COMPONENTS.md v6,
§3.1 and §8; manager 020 G1-F2).

The gate's fail-safe and owed-work sentences used to land only in `finalization.detail`,
which also carries raw error strings, so the phone could not show it. v6 adds one additive
key inside the session record's `finalization` object: `notice`, a string or ABSENT --

* written only for a session whose final solve went through the evidence gate and took a
  fail-safe or owes work (`coherence_publish.publish_notice`, and the finisher's given-up
  sentence);
* removed when the owed work is done (a re-gate in place that succeeds, a re-finish whose
  new solve owes nothing);
* absent on every older session and whenever nothing is owed, so those records and rows
  are byte for byte as today.

`detail` keeps its meaning and still carries the sentence. Pinned here, writer by writer:
the engine (`mark_finalization`), the builder (`world_build_session.py`), the repair tool
(`world_finalize.py`), the finisher (`world_finish_pending.py`), and the row
(`results/world_builder_library.py`, which passes the block through as it is).

And one follow-up to review V8 M3b: a re-finish whose PROCESS is gone no longer parks a
world (its ledger's `process`, `world_refinish.refinish_process_alive`).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_build_session as wbs  # noqa: E402
from scripts import world_finalize  # noqa: E402
from scripts import world_finish_pending as wfp  # noqa: E402
from scripts.world_build_session import StopRequest  # noqa: E402
from tests.test_world_builder_finish_pending import _stage, _world  # noqa: E402
from tests.test_world_builder_finish_pending_gate import (  # noqa: E402
    DEPTH_LOST,
    ROOM_INTERRUPTED,
    ROOM_OK,
    _at_the_bound,
    _gated,
    _marked,
    _refinish_ledger,
    _republishing,
    _run,
)
from tower.world_builder import coherence_publish as CP  # noqa: E402
from tower.world_builder.engine import WorldBuilderEngine  # noqa: E402
from tower.world_builder.records import (  # noqa: E402
    FINAL_SOLVE_SOLVED,
    FINALIZATION_COMPLETE,
    STAGE_STATE_STOPPED,
    STAGE_SURFACE,
)
from tower.world_builder.store import WorldStore  # noqa: E402

MASKS_OFF_GATE = {"state": CP.GATE_STATE_APPLIED, "masks_applied": False,
                  "metric_available": True, "retryable": False, "cause": None}
CLEAN_GATE = {"state": CP.GATE_STATE_APPLIED, "masks_applied": True,
              "metric_available": True, "retryable": False, "cause": None}
OOM = {"state": "unavailable", "cause": "gpu-oom", "retryable": True}


@pytest.fixture(autouse=True)
def _no_native_warm(monkeypatch):
    monkeypatch.setattr(wfp, "prewarm_world_builder", lambda *a, **k: ())


def _fin(store, world_id="w1", session_id="s1") -> dict:
    return store.read_session(world_id, session_id).finalization


def _engine(store, now=10.0) -> WorldBuilderEngine:
    return WorldBuilderEngine(store, clock=lambda: now)


# ---------------------------------------------------------------------------
# the engine: `mark_finalization(notice=...)`
# ---------------------------------------------------------------------------


def test_mark_finalization_writes_a_notice_and_removes_it(tmp_path):
    store = _world(tmp_path, stages=ROOM_OK)
    engine = _engine(store)
    engine.mark_finalization("w1", "s1", state=FINALIZATION_COMPLETE,
                             final_solve=FINAL_SOLVE_SOLVED, detail="d", notice="owed")
    assert _fin(store)["notice"] == "owed" and _fin(store)["detail"] == "d"
    engine.mark_finalization("w1", "s1", state=FINALIZATION_COMPLETE,
                             final_solve=FINAL_SOLVE_SOLVED, detail=None, notice=None)
    assert "notice" not in _fin(store), "removed means ABSENT, never null"


def test_an_unrelated_re_mark_keeps_a_still_valid_notice(tmp_path):
    """The default is KEEP: a caller that has nothing to say about the published solve --
    a `--skip-solve` repair, a record kept as it was -- does not silently drop its notice."""
    store = _world(tmp_path, stages=ROOM_OK)
    engine = _engine(store)
    engine.mark_finalization("w1", "s1", state=FINALIZATION_COMPLETE,
                             final_solve=FINAL_SOLVE_SOLVED, detail="d", notice="owed")
    engine.mark_finalization("w1", "s1", state=FINALIZATION_COMPLETE,
                             final_solve=FINAL_SOLVE_SOLVED, detail="something else")
    assert _fin(store)["notice"] == "owed" and _fin(store)["detail"] == "something else"


def test_an_empty_notice_is_absent(tmp_path):
    store = _world(tmp_path, stages=ROOM_OK)
    _engine(store).mark_finalization("w1", "s1", state=FINALIZATION_COMPLETE,
                                     final_solve=FINAL_SOLVE_SOLVED, notice="")
    assert "notice" not in _fin(store)


def test_an_old_record_and_its_row_stay_byte_for_byte_as_today(tmp_path):
    """§7 rule 1. A record with no notice, re-marked by today's callers (keep, or nothing
    owed), is the same bytes -- and so is its row."""
    from tower.results.world_builder_library import build_world_listing

    store = _world(tmp_path, stages=ROOM_OK)
    fin = _fin(store)
    record = store.session_path("w1", "s1").read_bytes()
    row = json.dumps(build_world_listing(store)["worlds"][0]["sessions"][0], sort_keys=True)
    assert "notice" not in row
    engine = _engine(store, now=fin["updated_at"])
    for notice in ({}, {"notice": None}):
        engine.mark_finalization("w1", "s1", state=fin["state"],
                                 final_solve=fin["final_solve"], detail=fin["detail"], **notice)
        assert store.session_path("w1", "s1").read_bytes() == record
        assert json.dumps(build_world_listing(store)["worlds"][0]["sessions"][0],
                          sort_keys=True) == row


def test_the_status_channel_carries_the_notice_and_old_sessions_are_unchanged(tmp_path):
    """Manager 022: the phone also reads `lifecycle.finalization.notice` from the status
    channel. The producer passes the session's finalization block through as it is
    (`results/world_builder.py`, `finalization = session.finalization` on every arm); this
    pins that the key arrives when set, and that a record without one -- re-marked by
    today's callers -- is the same payload and the same revision."""
    from tests.result_channel_fixtures import start_live_world
    from tower.results.world_builder import WorldBuilderStatusProducer

    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=6)
    engine.stop_session(hold_lock=True)
    engine.build(world_id, session_id)
    fixed = WorldBuilderEngine(WorldStore(root), clock=lambda: 1000.0)
    fixed.mark_finalization(world_id, session_id, state=FINALIZATION_COMPLETE,
                            final_solve=FINAL_SOLVE_SOLVED)
    engine.release_world(world_id)
    producer = WorldBuilderStatusProducer(root, lambda: 2000.0)
    before = producer.snapshot(world_id, session_id)
    assert before.payload["lifecycle"]["finalization"] is not None
    assert "notice" not in before.payload["lifecycle"]["finalization"]
    for kwargs in ({}, {"notice": None}):
        fixed.mark_finalization(world_id, session_id, state=FINALIZATION_COMPLETE,
                                final_solve=FINAL_SOLVE_SOLVED, **kwargs)
        after = producer.snapshot(world_id, session_id)
        assert after.revision == before.revision
        assert (json.dumps(after.payload, sort_keys=True, default=str)
                == json.dumps(before.payload, sort_keys=True, default=str))
    fixed.mark_finalization(world_id, session_id, state=FINALIZATION_COMPLETE,
                            final_solve=FINAL_SOLVE_SOLVED, notice=CP.NOTICE_MASKS_OOM)
    now = producer.snapshot(world_id, session_id)
    assert now.payload["lifecycle"]["finalization"]["notice"] == CP.NOTICE_MASKS_OOM
    assert now.revision != before.revision, "a new notice is a new revision"


def test_the_row_carries_the_notice_when_set(tmp_path):
    from tower.results.world_builder_library import build_world_listing

    store = _world(tmp_path, stages=ROOM_OK)
    _engine(store).mark_finalization("w1", "s1", state=FINALIZATION_COMPLETE,
                                     final_solve=FINAL_SOLVE_SOLVED, detail="d",
                                     notice=CP.NOTICE_MASKS_OOM)
    row = build_world_listing(store)["worlds"][0]["sessions"][0]
    assert row["finalization"]["notice"] == CP.NOTICE_MASKS_OOM
    assert row["finalization"]["detail"] == "d"


# ---------------------------------------------------------------------------
# the builder: `world_build_session.py`'s finalization
# ---------------------------------------------------------------------------


def _build(tmp_path, monkeypatch, solve_report: dict) -> WorldStore:
    """A synthetic walk with a final solve whose report is `solve_report` (the child and
    its GPU work faked), through the builder's real finalization."""
    monkeypatch.setattr(wbs, "prewarm_world_builder", lambda *a, **k: ())
    monkeypatch.setattr(wbs.StopRequest, "install", lambda self, **k: None)
    monkeypatch.setattr(wbs.BackgroundSolver, "run_final",
                        lambda self, store, sources, should_stop=None: dict(solve_report))
    root = tmp_path / "wb"
    with contextlib.redirect_stdout(io.StringIO()):
        code = wbs.main(["--synthetic", "--synthetic-frames", "10", "--root", str(root),
                         "--solve", "--solve-every", "0", "--format", "json"])
    assert code == 0
    return WorldStore(root)


def _only_session(store):
    (world_id,) = store.list_world_ids()
    (session_id,) = store.list_session_ids(world_id)
    return store.read_session(world_id, session_id)


def test_the_builder_writes_the_gates_notice(tmp_path, monkeypatch):
    report = {"attempted": True, "solved": True, "solver": "glomap", "gate": DEPTH_LOST,
              "transients": {"state": "applied"}}
    fin = _only_session(_build(tmp_path, monkeypatch, report)).finalization
    sentence = CP.publish_notice(report)
    assert sentence and "re-runs the gate" in sentence
    assert fin["notice"] == sentence
    assert fin["detail"] == sentence, "detail keeps its meaning, and the sentence"


@pytest.mark.parametrize("report", [
    {"attempted": True, "solved": True, "solver": "glomap", "gate": CLEAN_GATE,
     "transients": {"state": "applied"}},
    {"attempted": True, "solved": True, "solver": "glomap", "gate": None,
     "transients": OOM},
    {"attempted": True, "solved": True, "solver": "glomap"},
    {"attempted": True, "solved": False, "error": "RuntimeError: no model"},
], ids=["gated-nothing-owed", "ungated-masks-oom", "ungated", "failed"])
def test_the_builder_writes_no_notice_when_the_gate_owes_nothing(tmp_path, monkeypatch,
                                                                 report):
    """Nothing owed, no gate, or no published solve: the key is absent. The ungated
    masked solve keeps today's `detail` sentence -- the notice is the GATE's (§3.1)."""
    fin = _only_session(_build(tmp_path, monkeypatch, report)).finalization
    assert "notice" not in fin, fin
    if report.get("transients") == OOM:
        assert fin["detail"] == CP.NOTICE_MASKS_OOM


# ---------------------------------------------------------------------------
# the repair tool: `world_finalize.py`
# ---------------------------------------------------------------------------


def _finalize(store, monkeypatch, summary: dict | None, *extra) -> dict:
    from tower.world_builder import global_solve as GS

    if summary is not None:
        monkeypatch.setattr(GS, "solve", lambda *a, **k: dict(summary))
    monkeypatch.setattr(WorldBuilderEngine, "build",
                        lambda self, w, s: SimpleNamespace(
                            keyframes=0, poses_solved=0, poses_refused=0, points=0,
                            segments=0, scale_state="relative", diagnostics={}))
    monkeypatch.setattr(wbs, "should_register", lambda result: False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = world_finalize.main(["--root", str(store.root), "--world", "w1",
                                    "--session", "s1", *extra])
    return {"exit": code, **json.loads(out.getvalue())}


def _with_notice(store, notice):
    session = store.read_session("w1", "s1")
    store.write_session(replace(session, finalization=dict(session.finalization,
                                                            detail=notice, notice=notice)))


def test_finalize_writes_the_notice_of_the_solve_it_publishes(tmp_path, monkeypatch):
    store = _world(tmp_path, stages=ROOM_OK)
    summary = {"solved": True, "gate": MASKS_OFF_GATE, "transients": {}}
    assert _finalize(store, monkeypatch, summary)["finalized"] is True
    assert _fin(store)["notice"] == CP.publish_notice(summary)
    assert "TOWER_WORLD_SOLVE_MASKS" in _fin(store)["notice"]


def test_a_new_solve_that_owes_nothing_removes_the_notice(tmp_path, monkeypatch):
    """A re-finish's final solve (this tool, in its child) that owes nothing: the key goes."""
    store = _world(tmp_path, stages=ROOM_OK)
    _with_notice(store, CP.NOTICE_MASKS_OOM)
    assert _finalize(store, monkeypatch, {"solved": True, "gate": CLEAN_GATE,
                                          "transients": {"state": "applied"}})["finalized"]
    assert "notice" not in _fin(store)


@pytest.mark.parametrize("how", ["--skip-solve", "a failed solve"])
def test_a_repair_that_publishes_nothing_keeps_the_notice(tmp_path, monkeypatch, how):
    """The notice describes the PUBLISHED solve; a run that published none leaves it."""
    store = _world(tmp_path, stages=ROOM_OK)
    _with_notice(store, CP.NOTICE_MASKS_OOM)
    if how == "--skip-solve":
        _finalize(store, monkeypatch, None, "--skip-solve")
    else:
        _finalize(store, monkeypatch, {"solved": False, "error": "RuntimeError: x"})
    assert _fin(store)["notice"] == CP.NOTICE_MASKS_OOM


def test_an_ungated_repair_writes_no_notice(tmp_path, monkeypatch):
    store = _world(tmp_path, stages=ROOM_OK)
    _finalize(store, monkeypatch, {"solved": True, "gate": None, "transients": OOM})
    assert "notice" not in _fin(store)
    assert _fin(store)["detail"] == CP.NOTICE_MASKS_OOM, "detail: as today"


# ---------------------------------------------------------------------------
# the finisher: the re-gate in place, and the give-up
# ---------------------------------------------------------------------------


def test_the_gated_fixture_carries_the_notice_its_publish_wrote(tmp_path):
    store = _gated(tmp_path, stages=ROOM_OK)
    assert _fin(store)["notice"] == CP.publish_notice({"gate": DEPTH_LOST})


def test_a_re_gate_that_succeeds_removes_the_notice(tmp_path, monkeypatch, stage_runner):
    store = _gated(tmp_path, stages=ROOM_OK)
    monkeypatch.setattr(WorldBuilderEngine, "build",
                        lambda self, w, s: SimpleNamespace(poses_solved=4))
    v = wfp.assess(store, "w1", "s1")
    out = wfp.finish_regate(store, v, appearance=True, prune_depth_work=False,
                            stop_request=StopRequest(), regate=_republishing(store))
    assert out["finished"] is True, out
    assert "notice" not in _fin(store)
    assert _fin(store)["detail"] is None


def test_a_re_gate_that_takes_a_new_fail_safe_says_the_new_one(tmp_path, monkeypatch):
    store = _gated(tmp_path, stages=ROOM_OK)
    stop = StopRequest()
    short = "the evidence gate had too little metric scale to place pieces by it (x)"

    def regate(store_, world_id, session_id, should_stop=None):
        stop.request(StopRequest.SOFT, "capture opened")
        return {"notice": short}

    v = wfp.assess(store, "w1", "s1")
    wfp.finish_regate(store, v, appearance=True, prune_depth_work=False, stop_request=stop,
                      regate=regate)
    assert _fin(store)["notice"] == short and _fin(store)["detail"] == short


def test_the_give_up_is_the_notice(tmp_path, stage_runner):
    store = _gated(tmp_path, stages=ROOM_INTERRUPTED)
    _at_the_bound(store)
    assert _run(tmp_path) == 0
    fin = _fin(store)
    assert "stopped trying" in fin["notice"] and fin["detail"] == fin["notice"]
    assert "re-runs" not in fin["notice"]


def test_a_detail_rewritten_by_an_unrelated_mark_does_not_undo_the_give_up(tmp_path,
                                                                          stage_runner):
    """The give-up is recorded in the notice, and a re-mark that keeps it (a `--skip-solve`
    repair writes its own `detail`) leaves it recorded: no second retirement."""
    store = _gated(tmp_path, stages=ROOM_OK)
    _at_the_bound(store)
    assert _run(tmp_path) == 0
    given_up = _fin(store)["notice"]
    fin = _fin(store)
    WorldBuilderEngine(store).mark_finalization(
        "w1", "s1", state=fin["state"], final_solve=fin["final_solve"],
        detail="final solve skipped: --skip-solve")
    v = wfp.assess(store, "w1", "s1")
    assert not v.exhausted and v.code == "nothing-interrupted", v
    assert _fin(store)["notice"] == given_up


def test_a_promise_put_back_is_given_up_again(tmp_path, stage_runner):
    """A new publish of the same retryable solve (a hand-run `world_finalize.py`) writes
    the promise back into the notice; at the bound it is corrected at the next run."""
    store = _gated(tmp_path, stages=ROOM_OK)
    _at_the_bound(store)
    assert _run(tmp_path) == 0
    given_up = _fin(store)["notice"]
    promise = CP.publish_notice({"gate": DEPTH_LOST})
    _with_notice(store, promise)
    assert wfp.assess(store, "w1", "s1").exhausted
    assert _run(tmp_path) == 0
    assert _fin(store)["notice"] == given_up


def test_a_give_up_recorded_before_the_notice_existed_is_written_again(tmp_path,
                                                                      stage_runner):
    """A session given up by the finisher before v6 has the sentence in `detail` only; the
    next run writes it into `notice` too, once."""
    store = _gated(tmp_path, stages=ROOM_OK)
    _at_the_bound(store)
    sentence = wfp.regate_given_up_notice(wfp._published_meta(store, "w1", "s1"), 3)
    session = store.read_session("w1", "s1")
    fin = {k: v for k, v in session.finalization.items() if k != "notice"}
    store.write_session(replace(session, finalization=dict(fin, detail=sentence)))
    assert wfp.assess(store, "w1", "s1").exhausted
    assert _run(tmp_path) == 0
    assert _fin(store)["notice"] == sentence


@pytest.fixture
def stage_runner(monkeypatch):
    calls = []

    def fake(store, world_id, session_id, **kwargs):
        calls.append(world_id)
        kwargs["record"](STAGE_SURFACE, state="ok", detail=None)
        return {"surface": {"attempted": True, "state": "ok"}}

    monkeypatch.setattr(wfp, "final_surface_stages", fake)
    return calls


# ---------------------------------------------------------------------------
# M3b follow-up: the ledger's process decides whether a re-finish is live
# ---------------------------------------------------------------------------


def _ledger_with_process(store, stamp, state, process):
    from scripts import world_refinish as wr

    _refinish_ledger(store, stamp, state)
    path = store.world_dir("w1") / wr.REFINISH_DIRNAME / stamp / wr.LEDGER_FILENAME
    ledger = json.loads(path.read_text(encoding="utf-8"))
    ledger["process"] = process
    path.write_text(json.dumps(ledger), encoding="utf-8")


@pytest.fixture
def dead_process():
    """The pid and start time of a process that has exited."""
    from tower.world_builder.store import _lock_record

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        record = _lock_record(child.pid)
    finally:
        child.kill()
        child.wait(timeout=30)
    assert "created_at" in record
    return record


def _live_process():
    from tower.world_builder.store import _lock_record

    return _lock_record(os.getpid())


@pytest.mark.parametrize("state", ["setting-aside", "set-aside", "published"])
def test_a_live_refinish_holds_the_world_whatever_its_room_says(tmp_path, state):
    store = _world(tmp_path, stages={STAGE_SURFACE: _stage(STAGE_STATE_STOPPED)})
    _ledger_with_process(store, "t1", state, _live_process())
    v = wfp.assess(store, "w1", "s1")
    assert v.code == "refinish-in-progress" and not v.owed, v
    assert f"pid {os.getpid()}" in v.reason


@pytest.mark.parametrize("state", ["setting-aside", "set-aside", "published"])
def test_a_refinish_whose_process_is_gone_no_longer_parks_the_world(tmp_path, dead_process,
                                                                    stage_runner, state):
    """Before: the room still carried the re-finish's marker, so the world was left alone
    for ever. Now the ledger says the process is gone, and the stopped room is owed --
    the rebuild the re-finish promised the finisher would do."""
    store = _world(tmp_path, stages=_marked("t1"))
    _ledger_with_process(store, "t1", state, dead_process)
    v = wfp.assess(store, "w1", "s1")
    assert v.owed and v.stage == STAGE_SURFACE, v
    assert _run(tmp_path) == 0
    assert stage_runner == ["w1"]


def test_a_recycled_pid_is_not_the_refinish(tmp_path):
    """The pid is alive, but it started after the ledger's process did: another process."""
    store = _world(tmp_path, stages=_marked("t1"))
    record = dict(_live_process(), created_at=_live_process()["created_at"] - 1000.0)
    _ledger_with_process(store, "t1", "set-aside", record)
    assert wfp.assess(store, "w1", "s1").owed


@pytest.mark.parametrize("state", ["done", "stopped", "restored-after-a-failed-solve"])
def test_a_finished_refinish_is_todays_path_even_over_a_marked_room(tmp_path, state):
    """A terminal ledger is finished whatever the room says -- `stopped` after the room
    could not be rebuilt leaves the marker behind, and that room is owed, not held."""
    store = _world(tmp_path, stages=_marked("t1"))
    _ledger_with_process(store, "t1", state, _live_process())
    v = wfp.assess(store, "w1", "s1")
    assert v.owed and v.code == "owed", v


def test_a_ledger_from_before_the_process_was_recorded_is_decided_by_the_room(tmp_path):
    """No `process` (a re-finish before 3abe763): the marker rule, as before."""
    store = _world(tmp_path, stages=_marked("t1"))
    _refinish_ledger(store, "t1", "set-aside")
    assert wfp.assess(store, "w1", "s1").code == "refinish-in-progress"
    store2 = _world(tmp_path / "b", stages={STAGE_SURFACE: _stage(STAGE_STATE_STOPPED)})
    _refinish_ledger(store2, "t1", "set-aside")
    assert wfp.assess(store2, "w1", "s1").owed
