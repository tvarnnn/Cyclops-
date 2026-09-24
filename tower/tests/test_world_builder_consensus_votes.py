"""Who votes in the consensus, and what a stop publishes (review V9, M-1 and M-3; RV9-A F5 and P4; V9 LOWs).

Module: `tower/world_builder/coherence_publish.py` (`gate_by_consensus`, `draw_votes`, `gate_and_publish`,
`regate_published`, `regate_refusal`), `tower/config.py` (`world_solve_consensus_setting`).

  * Only a draw whose gate ATTACHED votes. A draw whose gate took a fail-safe (depth unavailable or stopped)
    used to vote every non-anchor keyframe "detached", mark every group ambiguous, and could even be the
    published draw.
  * Fewer voting draws than requested is `partial`, with the count; fewer than 2 publishes draw 0 unchanged.
  * A stop -- the `stopped` keyword, or `should_stop` between draws -- publishes draw 0 as N = 1 would,
    `deferred`, and owes the re-gate in place, which re-runs the consensus.
  * A re-gate of a solve that asked for N >= 2 re-runs the consensus, whatever its record's state.
  * `TOWER_WORLD_SOLVE_CONSENSUS` accepts 1, 3, 5, 7 only.

The synthetic world is the consensus test module's (`test_world_builder_solve_consensus`).
"""

from __future__ import annotations

import json
import logging
import time
import types

import pytest

from tests.test_world_builder_solve_consensus import (  # noqa: F401 -- `world` is a fixture
    PIECES,
    SID,
    _candidate,
    _consensus,
    _piece_kids,
    _plan,
    _room,
    _Store,
    world,
)
from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP
from tower.world_builder import global_solve as GS

OOM = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"


@pytest.fixture(autouse=True)
def _still_clock(monkeypatch):
    """Records carry seconds; a still clock makes two gates of one draw byte-comparable."""
    monkeypatch.setattr(time, "perf_counter", lambda: 100.0)


def _depth(monkeypatch, *, fail_calls=(), detail=OOM, stop=None, stop_on_call=None):
    """The gate's depth stage: the n-th call (1-based) in `fail_calls` raises `DepthUnavailable(detail)`; with
    `stop`, the `stop_on_call`-th call asks for a stop and every call that sees it is stopped."""
    calls = {"n": 0}

    def depth(store, world_id, session_id, solution, intrinsics, should_stop=None):
        calls["n"] += 1
        if stop is not None and calls["n"] == stop_on_call:
            stop["asked"] = True
        if should_stop is not None and should_stop():
            raise CP.DepthUnavailable(f"{CP.DEPTH_STOPPED}: the depth stage was stopped after 12 frames")
        if calls["n"] in fail_calls:
            raise CP.DepthUnavailable(detail)
        return ({"backend": "moge2-vitl", "known_fov": 42.0, "targets": len(solution.poses), "records": []},
                "work", object())

    monkeypatch.setattr(CP, "run_gate_depth", depth)
    return calls


def _short_scale(monkeypatch, *, short_calls=()):
    """The metric scale: the n-th call (1-based) in `short_calls` measures no camera -- the scale fail-safe WITH
    DEPTH IN HAND, which is not retryable (a re-gate would reproduce it). Review V10, L-1 made a RETRYABLE
    non-voting draw defer the consensus, so a non-voting draw that leaves it `partial` is this one."""
    real = CP.measure_metric_scale
    calls = {"n": 0}

    def metric(solution, name_of, db, work):
        calls["n"] += 1
        out = real(solution, name_of, db, work)
        return dict(out, metric_log={}) if calls["n"] in short_calls else out

    monkeypatch.setattr(CP, "measure_metric_scale", metric)
    return calls


def _single(world):
    return CP.gate_final_solution(_Store(), "w1", SID, _candidate(world.pieces), database_path="db",
                                  keyframes=world.keyframes)


def _without(record, *keys):
    return {k: v for k, v in record.items() if k not in keys + ("consensus", "seconds", "gate_seconds")}


# ---------------------------------------------------------------------------
# M-1: only draws whose gate attached vote


def test_a_draw_whose_gate_took_a_fail_safe_does_not_vote(world, monkeypatch):
    """Three identical draws; draw 2's gate measures too little metric scale (depth in hand: not retryable), so it
    attaches nothing. It used to vote every piece "detached" -- every group ambiguous 2-1. It does not vote: two
    draws do, unanimously, and the consensus says `partial`, 2 of 3. (A RETRYABLE fail-safe -- a depth stage out of
    GPU memory -- defers the consensus instead: review V10, L-1, `test_world_builder_consensus_v10.py`.)"""
    _short_scale(monkeypatch, short_calls=(3,))
    result, calls = _consensus(world, [(), (), ()])
    c = result.record["consensus"]
    assert calls == [8, 9]
    assert [d["attach"] for d in c["draws"]] == [True, True, False]
    assert [d["votes"] for d in c["draws"]] == [True, True, False]
    assert c["state"] == CP.CONSENSUS_PARTIAL and c["votes"]["draws"] == 2 and "2 of 3" in c["why"]
    assert all(not g["ambiguous"] for g in c["groups"]) and c["ambiguous"] == []
    assert c["votes"]["unanimous"] == world.n and c["detached"] == []
    assert _room(result) == sorted(kid for kid in result.solution.poses)


def test_with_one_voting_draw_draw_0_is_published_unchanged(world, monkeypatch):
    """Draws 1 and 2 both take the (non-retryable) fail-safe. The fail-safe draw -- its room the anchor block --
    used to agree best with a "consensus" of fail-safes and be PUBLISHED. With one voting draw there is no vote:
    draw 0, as the single gate publishes it."""
    single = _single(world)
    _short_scale(monkeypatch, short_calls=(2, 3))
    result, _ = _consensus(world, [(), (), ()])
    c = result.record["consensus"]
    assert c["state"] == CP.CONSENSUS_PARTIAL and c["votes"]["draws"] == 1
    assert c["chosen"] == {"draw": 0, "seed": 7} and c["detached"] == [] and c["groups"] == []
    assert result.record["attach"] is True and result.record["retryable"] is False
    assert _without(result.record) == _without(single.record)
    assert result.components == single.components
    assert result.solution.poses == single.solution.poses


# ---------------------------------------------------------------------------
# a stop publishes draw 0, deferred (M-1, M-3)


def test_a_stop_during_the_draws_publishes_draw_0_deferred_and_owes_the_consensus(world, monkeypatch):
    """RV9 `probe_stop_mid_consensus.py`: a stop arrives in draw 1's depth stage. Draw 1 takes the stopped
    fail-safe, draw 2 is skipped. The fail-safe draw 1 used to be published (its mapping, the anchor block alone)
    as consensus `applied`. Now: draw 0 exactly as one draw publishes it, `deferred`, owed to the re-gate."""
    single = _single(world)
    stop = {"asked": False}
    _depth(monkeypatch, stop=stop, stop_on_call=2)
    result, calls = _consensus(world, [(), (), ()], should_stop=lambda: stop["asked"])
    c = result.record["consensus"]
    assert calls == [8]
    assert c["draws"][2]["skipped"] and c["draws"][1]["attach"] is False
    assert c["state"] == CP.CONSENSUS_DEFERRED and c["why"] == CP.WHY_STOPPED
    assert c["chosen"] == {"draw": 0, "seed": 7} and c["votes"]["draws"] == 1
    assert result.solution.poses == single.solution.poses and result.components == single.components
    assert result.record["retryable"] is True and result.record["cause"] == CP.CAUSE_CONSENSUS_DEFERRED
    assert _without(result.record, "retryable", "cause") == _without(single.record, "retryable", "cause")
    assert CP.publish_notice({"gate": result.record, "transients": {"state": "applied"}}) == \
        CP.NOTICE_SENTENCES["consensus-deferred"]


def test_the_stopped_keyword_maps_no_draw_and_publishes_draw_0_deferred(world):
    """The interface the solve's draw loop calls (SOL): `stopped=True` -- nothing further is mapped."""
    single = _single(world)
    result, calls = _consensus(world, [(), ("A", "X"), ("B", "X")], stopped=True)
    c = result.record["consensus"]
    assert calls == []
    assert c["state"] == CP.CONSENSUS_DEFERRED and c["requested"] == 3 and c["seeds"] == [7, 8, 9]
    assert result.solution.poses == single.solution.poses and result.components == single.components
    assert result.record["retryable"] is True and result.record["cause"] == CP.CAUSE_CONSENSUS_DEFERRED
    assert result.consensus_detail is None


def test_the_stopped_keyword_on_a_fail_safe_first_draw_keeps_its_own_state(world, monkeypatch):
    """A first draw that took a fail-safe has nothing to vote on, stopped or not: as before."""
    _depth(monkeypatch, fail_calls=(1,))
    result, calls = _consensus(world, [(), (), ()], stopped=True)
    assert calls == [] and result.record["consensus"]["state"] == CP.CONSENSUS_DEFERRED
    assert result.record["cause"] == CP.CAUSE_DEPTH_UNAVAILABLE, "the depth stage's own cause"


def test_gate_and_publish_passes_stopped_through(world, tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_SOLVE_GATE", "1")
    plan, calls = _plan([(), ("A", "X"), ("B", "X")])
    written = []
    out, record = CP.gate_and_publish(_Store(), "w1", SID, types.SimpleNamespace(root=tmp_path),
                                      _candidate(world.pieces), final=True, database_path="db",
                                      keyframes=world.keyframes, write=lambda ws, s: written.append(s),
                                      consensus=plan, stopped=True)
    assert calls == [] and written == [out]
    assert record["consensus"]["state"] == CP.CONSENSUS_DEFERRED
    assert record["cause"] == CP.CAUSE_CONSENSUS_DEFERRED


# ---------------------------------------------------------------------------
# the re-gate in place re-runs the consensus (M-1)


def _published_store(tmp_path, monkeypatch, world, gate: dict, *, camera=True):
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    monkeypatch.setattr(store, "read_session", _Store().read_session)
    monkeypatch.setattr(store, "read_keyframes", lambda w, s: world.keyframes)
    ws = GS.workspace_for(store, "w1", SID)
    published = _candidate(world.pieces)
    if not camera:
        published.camera = None
    published.gate = gate
    GS.write_solution(ws, published)
    ws.database_path.write_bytes(b"features")
    return store, ws


def _mapper(asked):
    def mapper(store_, world_id, session_id, database_path, base, keyframes=None):
        def map_draw(s):
            asked.append(s)
            return _candidate(tuple(PIECES), rotated={8: ("A", "X"), 9: ("B", "X")}[s])
        return map_draw
    return mapper


def test_a_consensus_deferred_by_a_stop_is_owed_and_the_re_gate_runs_it(world, tmp_path, monkeypatch):
    stopped, _ = _consensus(world, [(), (), ()], stopped=True)
    store, ws = _published_store(tmp_path, monkeypatch, world, stopped.record)
    assert CP.regate_owed(store, "w1", SID) == CP.CAUSE_CONSENSUS_DEFERRED
    asked = []
    monkeypatch.setattr(GS, "frozen_draw_mapper", _mapper(asked))
    out = CP.regate_published(store, "w1", SID)
    assert asked == [8, 9]
    c = GS.load_solution(store, "w1", SID).gate["consensus"]
    assert c["state"] == CP.CONSENSUS_APPLIED and c["detached"] == [_piece_kids("X")[0]]
    assert out["gate"]["retryable"] is False and out["notice"] is None
    assert CP.regate_owed(store, "w1", SID) is None


@pytest.mark.parametrize("state", ["applied", "partial"])
def test_a_re_gate_of_a_consensus_solve_runs_the_consensus_whatever_its_state(world, tmp_path, monkeypatch,
                                                                              state):
    """RV9-A P4: 6d4b567 published a fail-safe draw as consensus `applied` (retryable); the re-gate that followed
    was a SINGLE gate, because only `deferred` re-ran the consensus, and the world stayed one draw. A solve that
    asked for N >= 2 is re-gated by consensus."""
    gate = {"state": CP.GATE_STATE_APPLIED, "retryable": True, "cause": CP.CAUSE_DEPTH_UNAVAILABLE,
            "consensus": {"state": state, "requested": 3, "seeds": [7, 8, 9]}}
    store, _ = _published_store(tmp_path, monkeypatch, world, gate)
    asked = []
    monkeypatch.setattr(GS, "frozen_draw_mapper", _mapper(asked))
    CP.regate_published(store, "w1", SID)
    assert asked == [8, 9]
    assert GS.load_solution(store, "w1", SID).gate["consensus"]["seeds"] == [7, 8, 9]


def test_the_refusal_and_the_re_gate_agree_when_the_draws_have_no_camera(world, tmp_path, monkeypatch):
    """V9 LOW: a deferred consensus whose solve has no camera -- `regate_refusal` said it would start,
    `regate_published` then raised inside `frozen_draw_mapper` after the finisher had counted the attempt."""
    gate = {"state": CP.GATE_STATE_APPLIED, "retryable": True, "cause": CP.CAUSE_CONSENSUS_DEFERRED,
            "consensus": {"state": CP.CONSENSUS_DEFERRED, "requested": 3, "seeds": [7, 8, 9]}}
    store, ws = _published_store(tmp_path, monkeypatch, world, gate, camera=False)
    assert not ws.camera_path.exists()
    before = ws.solution_path.read_bytes()
    refusal = CP.regate_refusal(store, "w1", SID)
    assert refusal == CP.REFUSAL_NO_DRAW_CAMERA
    with pytest.raises(CP.RegateRefused) as err:
        CP.regate_published(store, "w1", SID)
    assert str(err.value) == refusal
    assert ws.solution_path.read_bytes() == before, "nothing was attempted"
    # the workspace's camera.json is the mapper's fallback: with it the re-gate would start
    ws.camera_path.write_text(json.dumps(_candidate(world.pieces).camera), encoding="utf-8")
    assert CP.regate_refusal(store, "w1", SID) is None


# ---------------------------------------------------------------------------
# V9 LOW: the one frozen database is read once per consensus


def test_the_links_and_rotations_are_read_once_for_every_draw(world, monkeypatch):
    reads = {"links": 0, "rotations": 0}
    real_links, real_rotations = CG.read_verified_links, CG.read_link_rotations

    def links(db, min_inliers=15):
        reads["links"] += 1
        return real_links(db, min_inliers=min_inliers)

    def rotations(db, cam, min_inliers=15):
        reads["rotations"] += 1
        return real_rotations(db, cam, min_inliers=min_inliers)

    monkeypatch.setattr(CG, "read_verified_links", links)
    monkeypatch.setattr(CG, "read_link_rotations", rotations)
    result, _ = _consensus(world, [(), ("A", "X"), ("B", "X")])     # three draws and the withhold re-gate
    assert result.record["consensus"]["detached"] == [_piece_kids("X")[0]]
    assert reads == {"links": 1, "rotations": 1}


# ---------------------------------------------------------------------------
# M-3: the cap


@pytest.mark.parametrize("value,expected,logged", [
    (None, 1, False), ("", 1, False), ("1", 1, False), ("3", 3, False), (" 5 ", 5, False), ("7", 7, False),
    ("2", 1, True), ("4", 1, True), ("9", 1, True), ("30", 1, True), ("0", 1, True), ("-3", 1, True),
    ("three", 1, True),
])
def test_the_setting_accepts_odd_values_up_to_seven_and_logs_the_rest(monkeypatch, caplog, value, expected,
                                                                       logged):
    from tower.config import world_solve_consensus_setting

    if value is None:
        monkeypatch.delenv("TOWER_WORLD_SOLVE_CONSENSUS", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS", value)
    with caplog.at_level(logging.WARNING, logger="tower.config"):
        assert world_solve_consensus_setting() == expected
    said = [r for r in caplog.records if "TOWER_WORLD_SOLVE_CONSENSUS" in r.getMessage()]
    assert bool(said) is logged


def test_draw_votes_needs_an_attaching_gate():
    applied = CP.GateResult(solution=None, record={"state": CP.GATE_STATE_APPLIED, "attach": True},
                            components=None, gated={"groups": []})
    assert CP.draw_votes(applied)
    for record in ({"state": CP.GATE_STATE_APPLIED, "attach": False}, {"state": CP.GATE_STATE_FAILED},
                   {"state": CP.GATE_STATE_APPLIED}):
        assert not CP.draw_votes(CP.GateResult(solution=None, record=record, components=None,
                                               gated={"groups": []}))
    assert not CP.draw_votes(CP.GateResult(solution=None, record={"state": CP.GATE_STATE_APPLIED,
                                                                  "attach": True}, components=None))
