"""Review V10's consensus and publish findings, fixed in P3.7 (CON2).

Module: `tower/world_builder/coherence_publish.py` (and, for MED-5, `tower/results/world_builder.py`).

  * MED-1(b): a stop never reaches draw 0 of a STOPPED consensus (`stopped=True`), and a writer that already
    published the solve (`regate_published`, `gate_and_publish(keep_on_stop=True)`) writes nothing when a stop
    reached draw 0's own depth stage. Every depth fake here HONOURS the stop, as the product depth stage does
    (`dense_pipeline.run_depth_stage` polls `should_stop` before its first frame): V9's tests used fakes that
    ignored it, and passed for the wrong reason.
  * MED-4: the group-level vote decides; one unanimous keyframe no longer keeps a minority group (`kept` is gone).
  * L-1: a further draw that could not vote for a RETRYABLE cause defers the consensus (owed), not `partial`.
  * L-4: `not-withheld` on `not-applied`; every consensus state carries the vote keys.
  * L-5: the re-gate's consensus is capped by the setting's rule, and the published draw is its draw 0.
  * L-17: the surface lock is held while the metric scale reads the depth stage's predictions.
  * L-18: `write_failed` reaches `gate.depth.predictions`.
  * L-19b: no notice promises an idle re-run that cannot fix it; the closed set passes the phone's guard.
  * L-19d: a malformed `gate.depth` does not make the notice writers raise.
  * MED-5: `finalization.detail` and `lifecycle.reason` carry no path, traceback or line break.

The synthetic worlds: the consensus test module's (`test_world_builder_solve_consensus`), and, for MED-4, RV10-A's
`probe_kept.py` world (a room anchor and pieces hung on it by closed triangles of links), ported below.
"""

from __future__ import annotations

import builtins
import json
import math
import re
import time
import types

import numpy as np
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
from tower.world_builder.records import Keyframe

# The product depth stage, taken before any fixture replaces it (L-17 runs it for real).
PRODUCT_RUN_GATE_DEPTH = CP.run_gate_depth
OOM = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
RAW_OOM_AT_A_PATH = (r"RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB at "
                     r"C:\Users\tvllo\Projects\Glasses\tower\.venv\Lib\moge\model.py:411")


@pytest.fixture(autouse=True)
def _still_clock(monkeypatch):
    monkeypatch.setattr(time, "perf_counter", lambda: 100.0)


def _depth(monkeypatch, *, stop=None, arm_on_call=None, fail_calls=(), detail=OOM, seen=None):
    """The gate's depth stage, HONOURING the stop as the product's does: it asks `should_stop` before its first
    frame and is then `stopped`. With `stop`, its `arm_on_call`-th call asks for the stop first (a capture starts
    while that depth stage runs). The n-th call in `fail_calls` raises `DepthUnavailable(detail)`. `seen` collects
    the `should_stop` each call was handed."""
    calls = {"n": 0}

    def depth(store, world_id, session_id, solution, intrinsics, should_stop=None):
        calls["n"] += 1
        if seen is not None:
            seen.append(should_stop)
        if stop is not None and calls["n"] == arm_on_call:
            stop["asked"] = True
        if should_stop is not None and should_stop():
            raise CP.DepthUnavailable(f"{CP.DEPTH_STOPPED}: the depth stage was stopped after 0 frames")
        if calls["n"] in fail_calls:
            raise CP.DepthUnavailable(detail)
        return ({"backend": "moge2-vitl", "known_fov": 42.0, "targets": len(solution.poses), "records": []},
                "work", object())

    monkeypatch.setattr(CP, "run_gate_depth", depth)
    return calls


def _single(world):
    return CP.gate_final_solution(_Store(), "w1", SID, _candidate(world.pieces), database_path="db",
                                  keyframes=world.keyframes)


def _published_store(tmp_path, monkeypatch, world, gate: dict, *, solve_seed=0, published=None):
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    monkeypatch.setattr(store, "read_session", _Store().read_session)
    monkeypatch.setattr(store, "read_keyframes", lambda w, s: world.keyframes)
    ws = GS.workspace_for(store, "w1", SID)
    published = published if published is not None else _candidate(world.pieces)
    published.gate = gate
    published.solve = dict(published.solve or {}, seed=solve_seed)
    GS.write_solution(ws, published)
    ws.database_path.write_bytes(b"features")
    return store, ws


def _mapper(asked, rotated=None):
    rotated = rotated or {}

    def mapper(store_, world_id, session_id, database_path, base, keyframes=None):
        def map_draw(s):
            asked.append(s)
            return _candidate(tuple(PIECES), rotated=rotated.get(s, ()))
        return map_draw
    return mapper


# ---------------------------------------------------------------------------
# MED-1(b): a stop never reaches draw 0 of a stopped consensus


def test_a_stopped_consensus_gates_draw_0_without_the_stop(world, monkeypatch):
    """SOL's call once a stop was asked (`stopped=True` WITH `should_stop`). At e5f7151 the stop reached draw 0's
    own depth stage, which took the scale fail-safe: the anchor block alone was published, and the `stopped`
    branch never ran. Draw 0 is now gated as N = 1 gates it -- without the stop -- and published attached,
    `deferred`, owed."""
    single = _single(world)
    seen = []
    _depth(monkeypatch, seen=seen)
    result, calls = _consensus(world, [(), (), ()], stopped=True, should_stop=lambda: True)
    c = result.record["consensus"]
    assert calls == [], "no further draw is mapped"
    assert seen == [None], "draw 0's depth stage is not handed the stop"
    assert result.record["attach"] is True and result.record["depth"]["state"] == CP.DEPTH_OK
    assert c["state"] == CP.CONSENSUS_DEFERRED and c["why"] == CP.WHY_STOPPED
    assert result.record["retryable"] is True and result.record["cause"] == CP.CAUSE_CONSENSUS_DEFERRED
    assert result.solution.poses == single.solution.poses and result.components == single.components


def test_gate_and_publish_stopped_publishes_draw_0_attached_and_says_why(world, tmp_path, monkeypatch):
    """The same through the publish step, with the early publish's own `why` (review V10, MED-1a)."""
    monkeypatch.setenv("TOWER_WORLD_SOLVE_GATE", "1")
    _depth(monkeypatch)
    plan, calls = _plan([(), (), ()])
    written = []
    out, record = CP.gate_and_publish(_Store(), "w1", SID, types.SimpleNamespace(root=tmp_path),
                                      _candidate(world.pieces), final=True, database_path="db",
                                      keyframes=world.keyframes, write=lambda ws, s: written.append(s),
                                      consensus=plan, should_stop=lambda: True, stopped=True,
                                      why_deferred=CP.WHY_PUBLISHED_FIRST)
    assert calls == [] and written == [out]
    assert record["attach"] is True and record["consensus"]["state"] == CP.CONSENSUS_DEFERRED
    assert record["consensus"]["why"] == CP.WHY_PUBLISHED_FIRST
    assert record["retryable"] is True and record["cause"] == CP.CAUSE_CONSENSUS_DEFERRED


def test_keep_on_stop_writes_nothing_when_the_stop_reached_draw_0(world, tmp_path, monkeypatch):
    """The solve's second pass (after its early publish): a stop in draw 0's depth stage publishes NOTHING."""
    monkeypatch.setenv("TOWER_WORLD_SOLVE_GATE", "1")
    _depth(monkeypatch)
    plan, calls = _plan([(), (), ()])
    written = []
    out, record = CP.gate_and_publish(_Store(), "w1", SID, types.SimpleNamespace(root=tmp_path),
                                      _candidate(world.pieces), final=True, database_path="db",
                                      keyframes=world.keyframes, write=lambda ws, s: written.append(s),
                                      consensus=plan, should_stop=lambda: True, keep_on_stop=True)
    assert out is None and written == [] and calls == []
    assert record["publish"] == {"written": False, "why": CP.WHY_KEPT_ON_STOP}
    assert CP.draw_0_stopped(types.SimpleNamespace(record=record))
    assert not list(tmp_path.iterdir()), "no record, no depth hand-off"


def test_without_keep_on_stop_the_stopped_draw_0_is_published_as_today(world, tmp_path, monkeypatch):
    """The default: nothing was published before, so a stop never loses the finish."""
    monkeypatch.setenv("TOWER_WORLD_SOLVE_GATE", "1")
    _depth(monkeypatch)
    plan, _ = _plan([(), (), ()])
    written = []
    out, record = CP.gate_and_publish(_Store(), "w1", SID, types.SimpleNamespace(root=tmp_path),
                                      _candidate(world.pieces), final=True, database_path="db",
                                      keyframes=world.keyframes, write=lambda ws, s: written.append(s),
                                      consensus=plan, should_stop=lambda: True)
    assert written == [out] and record["attach"] is False and record["depth"]["state"] == CP.DEPTH_STOPPED


def test_a_stop_in_the_re_gates_draw_0_depth_stage_writes_nothing(world, tmp_path, monkeypatch):
    """RV10-A `probe_stop_draw0.py`, part 2: a `consensus-deferred` world (draw 0 attached) is re-gated by the
    finisher; a capture starts during draw 0's depth stage. At e5f7151 the world was overwritten by the anchor
    block alone (attach False). Now nothing is written, and it is still owed."""
    deferred, _ = _consensus(world, [(), (), ()], stopped=True)
    store, ws = _published_store(tmp_path, monkeypatch, world, deferred.record)
    (ws.root / CP.COMPONENTS_FILENAME).write_text(json.dumps(deferred.components), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in ws.root.iterdir() if p.is_file()}
    assert GS.load_solution(store, "w1", SID).gate["attach"] is True
    stop = {"asked": False}
    _depth(monkeypatch, stop=stop, arm_on_call=1)
    asked = []
    monkeypatch.setattr(GS, "frozen_draw_mapper", _mapper(asked))
    out = CP.regate_published(store, "w1", SID, should_stop=lambda: stop["asked"])
    assert GS.load_solution(store, "w1", SID).gate["attach"] is True, "the attached room was not replaced"
    assert {p.name: p.read_bytes() for p in ws.root.iterdir() if p.is_file()} == before, "nothing written"
    assert out["stopped"] is True and out["publish"]["written"] is False
    assert asked == []
    assert out["gate"]["attach"] is True and out["gate"]["cause"] == CP.CAUSE_CONSENSUS_DEFERRED
    assert out["notice"] == CP.NOTICE_SENTENCES["consensus-deferred"] == out["detail"]
    assert CP.regate_owed(store, "w1", SID) == CP.CAUSE_CONSENSUS_DEFERRED


def test_a_stop_in_a_single_re_gate_writes_nothing_either(world, tmp_path, monkeypatch):
    """N = 1: a failed gate's solve (published as the solver returned it) is not replaced by a stopped fail-safe."""
    gate = {"state": CP.GATE_STATE_FAILED, "retryable": True, "cause": CP.CAUSE_GATE_FAILED,
            "detail": "ZeroDivisionError: division by zero"}
    store, ws = _published_store(tmp_path, monkeypatch, world, gate)
    before = ws.solution_path.read_bytes()
    _depth(monkeypatch)
    out = CP.regate_published(store, "w1", SID, should_stop=lambda: True)
    assert ws.solution_path.read_bytes() == before and out["stopped"] is True
    assert CP.regate_owed(store, "w1", SID) == CP.CAUSE_GATE_FAILED


def test_a_stop_after_draw_0_still_publishes_draw_0_deferred(world, tmp_path, monkeypatch):
    """A stop that reaches draw 1 in the re-gate: draw 0 attached, `deferred`, written (as at e5f7151)."""
    gate = {"state": CP.GATE_STATE_APPLIED, "retryable": True, "cause": CP.CAUSE_DEPTH_UNAVAILABLE, "attach": False,
            "consensus": {"state": CP.CONSENSUS_DEFERRED, "requested": 3, "seeds": [7, 8, 9]}}
    store, _ = _published_store(tmp_path, monkeypatch, world, gate, solve_seed=7)
    stop = {"asked": False}
    _depth(monkeypatch, stop=stop, arm_on_call=2)
    monkeypatch.setattr(GS, "frozen_draw_mapper", _mapper([]))
    out = CP.regate_published(store, "w1", SID, should_stop=lambda: stop["asked"])
    assert "stopped" not in out
    published = GS.load_solution(store, "w1", SID).gate
    assert published["attach"] is True and published["consensus"]["state"] == CP.CONSENSUS_DEFERRED
    assert published["consensus"]["why"] == CP.WHY_STOPPED


# ---------------------------------------------------------------------------
# MED-4: the group-level vote decides


def _pure_draw(kids, room, room_groups, outside):
    poses = {k: {"component": 0 if k in room else outside.get(k, 99), "observations": 40} for k in kids}
    sol = types.SimpleNamespace(poses=poses, keyframe_ids=kids)
    groups = [{"label": 0, "reference": ref, "first_camera": first.upper(), "members": [m.upper() for m in mem]}
              for first, mem, ref in room_groups]
    return CP.GateResult(solution=sol, record={"state": CP.GATE_STATE_APPLIED, "attach": True}, components=None,
                         gated={"groups": groups})


def _rng(prefix, n, start=0):
    return [f"{prefix}{i:04d}" for i in range(start, start + n)]


def test_one_unanimous_keyframe_no_longer_keeps_a_minority_group():
    """G (40 kf) is attached by draw 0 only; ONE of its keyframes is in every draw's anchor block. At e5f7151 the
    whole group was `kept`, and 39 keyframes 2 of 3 draws did not attach stayed in the room."""
    A, g_one, g_rest, P, Q = _rng("a", 100), _rng("g", 1), _rng("g", 39, 1), _rng("p", 50), _rng("q", 50)
    kids = A + g_one + g_rest + P + Q
    d0 = _pure_draw(kids, set(kids), [(A[0], A, True), (g_one[0], g_one + g_rest, False), (P[0], P, False),
                                      (Q[0], Q, False)], {})
    d1 = _pure_draw(kids, set(A + g_one + Q), [(A[0], A + g_one, True), (Q[0], Q, False)],
                    {**{k: 1 for k in g_rest}, **{k: 2 for k in P}})
    d2 = _pure_draw(kids, set(A + g_one + P), [(A[0], A + g_one, True), (P[0], P, False)],
                    {**{k: 1 for k in g_rest}, **{k: 2 for k in Q}})
    out = CP.decide_consensus([d0, d1, d2], kid_of_name={k.upper(): k for k in kids}, min_obs=30)
    g = next(x for x in out["groups"] if x["first_keyframe"] == g_one[0])
    assert g["votes"] == [True, False, False]
    assert g["decision"] == CP.DECISION_SEED_UNSTABLE and g["unanimous_keyframes"] == 1
    assert out["withhold"] == [g_one[0].upper()]


# RV10-A's `probe_kept.py` world, end to end through `gate_by_consensus`.


def _rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])


def _nm(i):
    return f"{i:08d}.jpg"


def _kd(i):
    return f"{SID}:{i:08d}"


class _HungWorld:
    """A room anchor and pieces hung on it (RV9-A's fixture): `hang=(target, offset)` puts k consistent links from
    the piece's first k cameras to one target camera; `direct` adds links whose database rotation is off by
    `offset` degrees."""

    def __init__(self, spec: dict):
        self.spec, self.start, at = spec, {}, 0
        for p, s in spec.items():
            self.start[p] = at
            at += s["size"]
        self.n = at
        self.piece_of = {i: p for p, s in spec.items() for i in range(self.start[p], self.start[p] + s["size"])}
        links = {}
        for p, s in spec.items():
            lo = self.start[p]
            for i in range(lo, lo + s["size"]):
                for j in (i + 1, i + 2):
                    if j < lo + s["size"]:
                        links[(_nm(i), _nm(j))] = (100, 0.0)
        for p, s in spec.items():
            if s.get("hang"):
                tp, toff = s["hang"]
                for q in range(s["links"]):
                    links[(_nm(self.start[tp] + toff), _nm(self.start[p] + q))] = (40, 0.0)
            for (tp, toff, k, off) in s.get("direct", ()):
                for q in range(k):
                    links[(_nm(self.start[tp] + toff), _nm(self.start[p] + s["size"] - 1 - q))] = (40, float(off))
        self.links_db = links
        self.keyframes = [Keyframe(keyframe_id=_kd(i), session_id=SID, source_seq=i, received_at=1000.0 + 0.5 * i,
                                   image_relpath=f"images/{i:08d}.jpg", width=320, height=240, byte_count=1)
                          for i in range(self.n)]

    def install(self, monkeypatch):
        links = {k: v[0] for k, v in self.links_db.items()}
        rots = {(a, b): _rz(off + 3.0 * int(b[:8]) - 3.0 * int(a[:8])) for (a, b), (_i, off) in self.links_db.items()}
        metric = {_nm(i): 0.0 for i in range(self.n)}
        monkeypatch.setattr(CG, "read_verified_links", lambda db, min_inliers=15: dict(links))
        monkeypatch.setattr(CG, "read_link_rotations", lambda db, cam, min_inliers=15: dict(rots))
        monkeypatch.setattr(CP, "run_gate_depth", lambda store, w, s, solution, intr, should_stop=None: (
            {"backend": "moge2-vitl", "known_fov": 42.0, "targets": len(solution.poses), "records": []},
            "work", object()))
        monkeypatch.setattr(CP, "measure_metric_scale", lambda solution, name_of, db, work: {
            "metric_log": dict(metric), "cameras_published": self.n, "cameras_measured": self.n})

    def candidate(self, rotated=None, solved_at=1234.5, seed=0):
        rotated = rotated or {}
        R = [_rz(rotated.get(self.piece_of[i], 0.0) + 3.0 * i) for i in range(self.n)]
        obs, p = [], 0
        for pc, s in self.spec.items():
            idx = list(range(self.start[pc], self.start[pc] + s["size"]))
            for j in range(len(idx)):
                for _ in range(30):
                    for c in idx[j:j + 3]:
                        obs.append([c, len(obs), p])
                    p += 1
        obs = np.asarray(obs, np.int32)
        counts = np.bincount(obs[:, 0], minlength=self.n)
        poses = {_kd(i): {"component": 0, "rotation": R[i].ravel().tolist(), "translation": [0.0, 0.0, 0.0],
                          "observations": int(counts[i])} for i in range(self.n)}
        first = np.zeros(p, np.int32)
        for row in obs[::-1]:
            first[row[2]] = row[0]
        return GS.Solution(
            solver="glomap", solved_at=solved_at, input_digest="digest-final",
            keyframe_ids=[_kd(i) for i in range(self.n)], poses=poses,
            components=[{"index": 0, "images": self.n, "points": p}],
            xyz=np.zeros((p, 3), np.float32), rgb=np.zeros((p, 3), np.uint8), component=np.zeros(p, np.int32),
            first_keyframe=first, track_length=np.full(p, 3, np.int32), error=np.full(p, 0.5, np.float32),
            observations=obs, observation_xy=np.zeros((len(obs), 2), np.float32),
            camera={"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240},
            transients={"state": "applied"}, solve={"seed": seed})

    def kids(self, p):
        return [_kd(i) for i in range(self.start[p], self.start[p] + self.spec[p]["size"])]


def test_probe_kept_the_minority_group_is_withheld_end_to_end(monkeypatch):
    """RV10-A `probe_kept.py`: G0 (1 camera) + G (39) attach together in draw 0; draws 1 and 2 put G0 in the anchor
    block and leave G out. At e5f7151 the group (40) was `kept` for G0's unanimous vote and published placed."""
    w = _HungWorld({
        "A": dict(size=60, hang=None),
        "G0": dict(size=1, hang=("A", 10), links=1, direct=[("A", 11, 1, 0.0)]),
        "G": dict(size=39, hang=("G0", 0), links=3, direct=[("A", 30, 4, 40.0)]),
        "P": dict(size=50, hang=("A", 40), links=3),
        "Q": dict(size=50, hang=("A", 50), links=3),
    })
    w.install(monkeypatch)
    draws = [{"G0": 40, "G": 40}, {"G": 70, "P": 40}, {"G": 70, "Q": 40}]
    cands = [w.candidate(rotated=draws[i], solved_at=2007.0 + i, seed=7 + i) for i in range(3)]
    plan = CP.ConsensusPlan(draws=3, seed=7, map_draw=lambda s: cands[s - 7])
    r = CP.gate_by_consensus(_Store(), "w1", SID, cands[0], plan=plan, database_path="db", keyframes=w.keyframes)
    c = r.record["consensus"]
    assert c["state"] == CP.CONSENSUS_APPLIED and c["chosen"]["draw"] == 0
    g = next(x for x in c["groups"] if x["first_keyframe"] in w.kids("G0") + w.kids("G"))
    assert g["keyframes"] == 40 and g["votes"] == [True, False, False]
    assert g["decision"] == CP.DECISION_SEED_UNSTABLE and g["unanimous_keyframes"] == 1
    room = {kid for kid, p in r.solution.poses.items() if p["component"] == 0}
    assert not room & set(w.kids("G")), "the minority group is out of the published room"
    entry = next(e for e in r.components["components"] if w.kids("G")[0] in e["keyframe_ids"])
    assert entry["reasons"] == [CG.REASON_SEED_UNSTABLE]


# ---------------------------------------------------------------------------
# L-1: a retryable non-voting draw defers the consensus


def test_a_further_draw_out_of_gpu_memory_defers_the_consensus(world, monkeypatch):
    """RV10-A `probe_partial2.py`: draw 2's depth stage runs out of GPU memory. At e5f7151 that was `partial` with
    2 voters -- a strict majority of 2 is unanimity -- never owed a re-run. Now: draw 0, `deferred`, owed."""
    single = _single(world)
    _depth(monkeypatch, fail_calls=(3,))
    result, calls = _consensus(world, [(), ("A",), ()])
    c = result.record["consensus"]
    assert calls == [8, 9]
    assert c["state"] == CP.CONSENSUS_DEFERRED and c["why"] == CP.WHY_DRAW_UNFINISHED
    assert c["chosen"] == {"draw": 0, "seed": 7} and c["votes"]["draws"] == 2
    assert result.record["retryable"] is True and result.record["cause"] == CP.CAUSE_CONSENSUS_DEFERRED
    assert result.solution.poses == single.solution.poses and result.components == single.components
    assert CP.publish_notice({"gate": result.record, "transients": {"state": "applied"}}) == \
        CP.NOTICE_SENTENCES["consensus-deferred"]


def test_a_further_draw_whose_gate_failed_defers_the_consensus(world, monkeypatch):
    real = CP.measure_metric_scale
    calls = {"n": 0}

    def metric(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite_error()
        return real(*a, **kw)

    def sqlite_error():
        import sqlite3

        return sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(CP, "measure_metric_scale", metric)
    result, _ = _consensus(world, [(), (), ()])
    c = result.record["consensus"]
    assert c["draws"][1]["gate_state"] == CP.GATE_STATE_FAILED
    assert c["state"] == CP.CONSENSUS_DEFERRED and c["why"] == CP.WHY_DRAW_UNFINISHED


def test_the_consensus_deferred_sentence_is_true_of_every_way_it_is_owed():
    assert "stopped" not in CP.NOTICE_SENTENCES["consensus-deferred"]
    assert CP.NOTICE_SENTENCES["consensus-deferred"] == ("the evidence gate's consensus of mapper seeds did not "
                                                          "finish; the Tower re-runs it when it is idle")


# ---------------------------------------------------------------------------
# L-4: the decision on `not-applied`, and the full key set in every state

VOTE_KEYS = ("groups", "pieces", "detached", "ambiguous", "held_against_majority")


def test_not_applied_says_the_group_was_not_withheld(world, monkeypatch):
    monkeypatch.setattr(CP, "withhold_refusal", lambda *a, **kw: "a forced refusal")
    result, _ = _consensus(world, [(), ("A", "X"), ("B", "X")])
    c = result.record["consensus"]
    assert c["state"] == CP.CONSENSUS_NOT_APPLIED and c["detached"] == []
    x = next(g for g in c["groups"] if g["first_keyframe"] == _piece_kids("X")[0])
    assert x["decision"] == CP.DECISION_NOT_WITHHELD and x["votes"] == [True, False, False]
    assert set(_piece_kids("X")) <= set(_room(result)), "published in the room, as the chosen draw gated it"


@pytest.mark.parametrize("shape", ["not-run", "not-needed", "deferred-first-draw", "deferred-stopped",
                                   "deferred-unfinished", "partial-one", "applied"])
def test_every_consensus_state_carries_the_vote_keys(world, monkeypatch, shape):
    kw = {}
    draws = [(), (), ()]
    if shape == "not-needed":
        monkeypatch.setattr(CP, "measure_metric_scale", lambda *a, **k: {"metric_log": {}})
    elif shape == "deferred-first-draw":
        _depth(monkeypatch, fail_calls=(1,))
    elif shape == "deferred-stopped":
        kw["stopped"] = True
    elif shape == "deferred-unfinished":
        _depth(monkeypatch, fail_calls=(2,))
    elif shape == "partial-one":
        real = CP.measure_metric_scale
        n = {"n": 0}

        def metric(*a, **k):
            n["n"] += 1
            return dict(real(*a, **k), metric_log={}) if n["n"] > 1 else real(*a, **k)

        monkeypatch.setattr(CP, "measure_metric_scale", metric)
    if shape == "not-run":
        plan, _ = _plan(draws)
        plan.refusal = "the solve was not seeded"
        result = CP.gate_by_consensus(_Store(), "w1", SID, _candidate(world.pieces), plan=plan,
                                      database_path="db", keyframes=world.keyframes)
    else:
        result, _ = _consensus(world, draws, **kw)
    c = result.record["consensus"]
    expected = {"not-run": CP.CONSENSUS_NOT_RUN, "not-needed": CP.CONSENSUS_NOT_NEEDED,
                "partial-one": CP.CONSENSUS_PARTIAL, "applied": CP.CONSENSUS_APPLIED}.get(shape,
                                                                                       CP.CONSENSUS_DEFERRED)
    assert c["state"] == expected
    assert all(k in c for k in VOTE_KEYS), sorted(set(VOTE_KEYS) - set(c))


# ---------------------------------------------------------------------------
# L-5: the re-gate's consensus is capped, and the published draw is its draw 0


def test_an_uncapped_old_record_is_re_gated_as_the_setting_would_treat_it(world, tmp_path, monkeypatch):
    gate = {"state": CP.GATE_STATE_APPLIED, "retryable": True, "cause": CP.CAUSE_CONSENSUS_DEFERRED,
            "consensus": {"state": CP.CONSENSUS_DEFERRED, "requested": 30, "seeds": list(range(7, 37))}}
    store, _ = _published_store(tmp_path, monkeypatch, world, gate, solve_seed=7)
    asked = []
    monkeypatch.setattr(GS, "frozen_draw_mapper", _mapper(asked))
    CP.regate_published(store, "w1", SID)
    assert asked == [], "30 is not one of 1, 3, 5, 7: one draw, as TOWER_WORLD_SOLVE_CONSENSUS=30 gives today"
    assert "consensus" not in GS.load_solution(store, "w1", SID).gate


def test_a_published_draw_k_is_the_re_gates_draw_0(world, tmp_path, monkeypatch):
    """The consensus published its draw 2 (seed 9). The re-gate maps the OTHER seeds (7, 8), never 9 again."""
    gate = {"state": CP.GATE_STATE_APPLIED, "retryable": True, "cause": CP.CAUSE_CONSENSUS_DEFERRED,
            "consensus": {"state": CP.CONSENSUS_DEFERRED, "requested": 3, "seeds": [7, 8, 9]}}
    store, _ = _published_store(tmp_path, monkeypatch, world, gate, solve_seed=9)
    asked = []
    monkeypatch.setattr(GS, "frozen_draw_mapper", _mapper(asked))
    CP.regate_published(store, "w1", SID)
    assert asked == [7, 8]
    c = GS.load_solution(store, "w1", SID).gate["consensus"]
    assert c["seeds"] == [9, 7, 8] and c["draws"][0]["seed"] == 9


def test_owed_consensus_reads_only_what_a_record_can_hold(world):
    sol = types.SimpleNamespace(gate={"consensus": {"requested": 3, "seeds": [7, 8, 9]}}, solve={"seed": 7})
    assert CP._owed_consensus(sol) == (3, [7, 8, 9])
    sol.solve = {"seed": 8}
    assert CP._owed_consensus(sol) == (3, [8, 7, 9])
    for bad in ({"requested": 2, "seeds": [7, 8]}, {"requested": "3", "seeds": [7, 8, 9]},
                {"requested": True, "seeds": [7]}):
        assert CP._owed_consensus(types.SimpleNamespace(gate={"consensus": bad}, solve={}))[0] == 1
    assert CP._owed_consensus(types.SimpleNamespace(gate={"consensus": {"requested": 3}}, solve={})) == (3, None)
    assert CP._owed_consensus(types.SimpleNamespace(gate="not a dict", solve=None)) == (1, None)


# ---------------------------------------------------------------------------
# L-17 and L-18: the depth stage's lock and record


def _real_depth_world(tmp_path, monkeypatch, world, align_extra=None, stopped_after=None):
    """The PRODUCT `run_gate_depth` on a real store, with `dense_pipeline.run_depth_stage` faked."""
    from tower.world_builder import dense_pipeline
    from tower.world_builder.store import WorldStore
    from tower.world_builder.surface_pipeline import surface_dir

    monkeypatch.setattr(CP, "run_gate_depth", PRODUCT_RUN_GATE_DEPTH)
    store = WorldStore(tmp_path)
    intrinsics = types.SimpleNamespace(fx=300.0, fy=300.0, cx=160.0, cy=120.0, width=320, height=240)
    monkeypatch.setattr(store, "read_session",
                        lambda w, s: types.SimpleNamespace(started_at=1000.0, intrinsics=intrinsics))

    def stage(store_, world_id, session_id, solution, intrinsics_, dparams, root, **kw):
        return {"backend": "moge2-vitl", "known_fov": 42.0, "targets": len(solution.poses), "records": [],
                "stopped_after": stopped_after, **(align_extra or {})}

    monkeypatch.setattr(dense_pipeline, "run_depth_stage", stage)
    return store, surface_dir(store, "w1", SID) / ".surface.lock"


def test_the_surface_lock_is_held_while_the_metric_scale_reads_the_predictions(world, tmp_path, monkeypatch):
    """RV10 L-17 (RV10-D D4): the lock was released before `metric_fn` read `work/`, so a densify in between could
    replace the predictions the gate then measured."""
    store, lock = _real_depth_world(tmp_path, monkeypatch, world)
    seen = []
    real = CP.measure_metric_scale

    def metric(*a, **kw):
        seen.append(lock.exists())
        return real(*a, **kw)

    monkeypatch.setattr(CP, "measure_metric_scale", metric)
    result = CP.gate_final_solution(store, "w1", SID, _candidate(world.pieces), database_path="db",
                                    keyframes=world.keyframes)
    assert result.record["attach"] is True and seen == [True]
    assert not lock.exists(), "released after the read"


def test_a_stopped_depth_stage_releases_the_lock(world, tmp_path, monkeypatch):
    store, lock = _real_depth_world(tmp_path, monkeypatch, world, stopped_after=3)
    result = CP.gate_final_solution(store, "w1", SID, _candidate(world.pieces), database_path="db",
                                    keyframes=world.keyframes)
    assert result.record["depth"]["state"] == CP.DEPTH_STOPPED and not lock.exists()


def test_write_failed_reaches_the_gate_record(world, tmp_path, monkeypatch):
    """RV10 L-18: a prediction the cache could not keep was counted by the depth stage and lost on the way."""
    store, _ = _real_depth_world(tmp_path, monkeypatch, world, align_extra={"prediction_cache": {
        "token": "t", "hits": 5, "predicted": 3, "write_failed": 2}})
    result = CP.gate_final_solution(store, "w1", SID, _candidate(world.pieces), database_path="db",
                                    keyframes=world.keyframes)
    assert result.record["depth"]["predictions"] == {"token": "t", "cached": 5, "predicted": 3, "write_failed": 2}


# ---------------------------------------------------------------------------
# L-19d: a malformed `gate.depth`


@pytest.mark.parametrize("depth", ["unavailable", ["x"], 3])
def test_a_malformed_depth_record_does_not_make_the_notice_writers_raise(depth):
    gate = {"state": CP.GATE_STATE_APPLIED, "retryable": True, "cause": CP.CAUSE_DEPTH_UNAVAILABLE,
            "masks_applied": True, "depth": depth}
    summary = {"gate": gate, "transients": {"state": "applied"}}
    assert CP.publish_notice(summary) == CP.NOTICE_SENTENCES["depth-unavailable"]
    assert CP.publish_detail(summary)
    assert CP.notice_cause_phrase(gate) == CP.NOTICE_PHRASES["depth-unavailable"]
    assert CP.regate_clause(gate) == CP.NOTICE_CLAUSES["depth-unavailable"]
    shortfall = {"state": CP.GATE_STATE_APPLIED, "metric_available": False, "depth": depth,
                 "evidence": "not a dict"}
    assert CP.publish_notice({"gate": shortfall, "transients": {"state": "applied"}}) is None


# ---------------------------------------------------------------------------
# L-19b: no idle re-run promised that cannot fix it; the phone's guard


def test_no_notice_promises_an_idle_re_run_that_cannot_fix_it():
    for cause in ("depth-no-intrinsics", "depth-no-camera"):
        sentence = CP.NOTICE_SENTENCES[cause]
        assert "re-runs" not in sentence and "an owner can re-" in sentence, sentence
    assert CP.NOTICE_SENTENCES["depth-no-intrinsics"].endswith("an owner can re-capture this walk")
    assert CP.NOTICE_SENTENCES["depth-no-camera"].endswith("an owner can re-finish this walk")
    model = CP.NOTICE_SENTENCES["depth-model-missing"]
    assert "an operator can install the depth model on this Tower, then" in model
    summary = {"gate": {"state": CP.GATE_STATE_APPLIED, "retryable": True, "masks_applied": True,
                        "cause": CP.CAUSE_DEPTH_UNAVAILABLE,
                        "depth": {"state": CP.DEPTH_UNAVAILABLE, "detail": "session has no intrinsics"}},
               "transients": {"state": "applied"}}
    assert "re-runs" not in CP.publish_detail(summary), "the diagnostic twin names the same owner"


# The Mac's `WorldTowerText` guard (manager 030, Mac tip 3bb4431) swaps a Tower sentence for a generic one when it
# holds a path, a traceback, an exception class name, a lower-case key=value pair, JSON, a line break or more than
# 700 characters. The rules are the Mac's; the patterns below are this test's reading of them, deliberately broad.
_BUILTIN_EXCEPTIONS = {n for n in dir(builtins) if isinstance(getattr(builtins, n), type)
                       and issubclass(getattr(builtins, n), BaseException)}
_GUARD = {
    "a path": re.compile(r"[\\/]|\b[A-Za-z]:|~[\w.-]*[\\/]|\.py\b"),
    "a traceback": re.compile(r"Traceback|File \"|line \d+, in "),
    "an exception class name": re.compile(r"\b[A-Z]\w*(?:Error|Exception|Interrupt|Warning)\b|\b(?:"
                                          + "|".join(sorted(_BUILTIN_EXCEPTIONS)) + r")\b"),
    "a key=value pair": re.compile(r"\b[a-z_][a-z0-9_.]*\s*="),
    "JSON": re.compile(r"[{}\[\]\"]"),
    "a line break": re.compile(r"[\r\n]"),
}


def _guard_trips(text: str) -> list[str]:
    tripped = [name for name, pattern in _GUARD.items() if pattern.search(text)]
    return tripped + (["over 700 characters"] if len(text) > 700 else [])


def _every_phone_sentence():
    from scripts import world_finish_pending as wfp

    out = {}
    for cause, sentence in CP.NOTICE_SENTENCES.items():
        out[f"NOTICE_SENTENCES[{cause}]"] = sentence.format(unmasked=12, images=400, excluded=12)
    for cause, clause in CP.NOTICE_CLAUSES.items():
        what = clause.format(unmasked=12, images=400, excluded=12)
        out[f"WFP.REGATE_GIVEN_UP[{cause}]"] = wfp.REGATE_GIVEN_UP.format(what=what, attempts=3)
        out[f"WFP.REGATE_REFUSED_NOTICE[{cause}]"] = wfp.REGATE_REFUSED_NOTICE.format(what=what)
    out["WFP.REFINISH_PARKED_NOTICE"] = wfp.REFINISH_PARKED_NOTICE
    return out


def test_every_phone_sentence_passes_the_macs_guard():
    trips = {name: _guard_trips(text) for name, text in _every_phone_sentence().items()}
    assert {name: t for name, t in trips.items() if t} == {}


def test_the_guard_reading_catches_what_it_must():
    for bad in ("RuntimeError: x", "masks state=partial", "C:\\Users\\x", "a\nb", '{"a": 1}', "x" * 701,
                "Traceback (most recent call last)", "MemoryError"):
        assert _guard_trips(bad), bad


# ---------------------------------------------------------------------------
# MED-5: the detail and the lifecycle reason carry no path, traceback or line break


RAW_GATE = {"state": CP.GATE_STATE_APPLIED, "masks_applied": True, "retryable": True,
            "cause": CP.CAUSE_DEPTH_UNAVAILABLE,
            "depth": {"state": CP.DEPTH_UNAVAILABLE, "detail": RAW_OOM_AT_A_PATH}}


def _no_leak(text):
    assert "tvllo" not in text and "Users" not in text and "\\" not in text and "\n" not in text
    assert not re.search(r"\b[A-Za-z]:[\\/]", text) and ".py" not in text and "Traceback" not in text


def test_publish_detail_keeps_the_class_and_message_and_drops_the_path():
    """RV10-E `probe_detail_to_phone.py`: e5f7151 wrote the raw text, `C:\\Users\\<user>\\...` and all."""
    detail = CP.publish_detail({"gate": RAW_GATE, "transients": {"state": "applied"}})
    _no_leak(detail)
    assert "(RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB at [path])" in detail
    assert RAW_GATE["depth"]["detail"] == RAW_OOM_AT_A_PATH, "the record keeps its text"


def _lifecycle(detail, *, end_reason="error", state="complete", geometry=False):
    from tower.results import world_builder as RWB

    session = types.SimpleNamespace(finalization={"state": state, "final_solve": "solved", "detail": detail},
                                    end_reason=end_reason, ended_at=1.0, stages=None)
    return RWB._lifecycle_from_the_record(holder=None, stopped=True, session=session, geometry_current=True,
                                          has_manifest=True, has_session_geometry=geometry,
                                          has_readable_figures=True)


@pytest.mark.parametrize("detail", [
    RAW_OOM_AT_A_PATH,
    r"PermissionError: [WinError 32] The process cannot access the file: "
    r"'C:\Users\tvllo\Projects\Glasses\tower\data\worlds\w\derived\s\points.json'",
    'Traceback (most recent call last):\n  File "C:\\Users\\tvllo\\x.py", line 3, in <module>\n'
    "RuntimeError: boom at /home/tvllo/a/b.py",
    "final solve failed: OSError: [Errno 28] No space left on device: '/home/tvllo/Glasses/tower/data/w'",
])
@pytest.mark.parametrize("branch", ["error-no-geometry", "finalization-interrupted"])
def test_the_lifecycle_reason_quotes_the_detail_owner_facing(detail, branch):
    lc = (_lifecycle(detail) if branch == "error-no-geometry"
          else _lifecycle(detail, end_reason="stopped", state="interrupted", geometry=True))
    assert lc["state"] == "interrupted"
    _no_leak(lc["reason"])
    assert not re.search(r"\b[A-Z]\w*(?:Error|Exception)\b", lc["reason"]), lc["reason"]


def test_the_finalization_on_the_socket_carries_a_client_safe_detail():
    """`lifecycle.finalization` goes out whole on the unauthenticated `/ws`: its `detail` is a client-safe copy;
    the record itself is not touched."""
    from tower.results import world_builder as RWB

    record = {"state": "interrupted", "final_solve": "solved", "detail": RAW_OOM_AT_A_PATH, "notice": "n"}
    session = types.SimpleNamespace(finalization=record, end_reason="stopped", ended_at=1.0, stages=None)
    lc = RWB._lifecycle_from_the_record(holder=None, stopped=True, session=session, geometry_current=True,
                                        has_manifest=True, has_session_geometry=True, has_readable_figures=True)
    _no_leak(lc["finalization"]["detail"])
    assert lc["finalization"]["detail"].startswith("RuntimeError: CUDA out of memory")
    assert {k: v for k, v in lc["finalization"].items() if k != "detail"} == \
        {k: v for k, v in record.items() if k != "detail"}
    assert record["detail"] == RAW_OOM_AT_A_PATH
    clean = dict(record, detail="final solve skipped: hard stop (stdin-closed) during finalization")
    assert RWB._client_safe_finalization(clean) is clean


@pytest.mark.parametrize("detail", [
    "final solve skipped: hard stop (stdin-closed) during finalization",
    "final solve terminated: hard stop (signal) during finalization; the last background solution stands",
    "final solve produced no solution: no keyframes were posed",
    CP.NOTICE_SENTENCES["consensus-deferred"],
    CP.NOTICE_SENTENCES["scale-short"],
])
def test_a_clean_detail_renders_byte_for_byte_as_before(detail):
    """e5f7151's formula, verbatim, for a detail with nothing to scrub."""
    assert _lifecycle(detail)["reason"] == f"the mapping session ended with 'error': {detail}"
    assert _lifecycle(detail, end_reason="stopped", state="interrupted", geometry=True)["reason"] == (
        "finalization did not complete; the geometry stored is the last build that finished" + f": {detail}")
    assert _lifecycle(None)["reason"] == "the mapping session ended with 'error'"


@pytest.mark.parametrize("raw,safe", [
    (r"C:\Users\tvllo\a b\c.py failed", "[path] failed"),
    ("C:/Program Files/x y/z.py failed", "[path] failed"),
    (r"\\server\share\file.txt missing", "[path] missing"),
    ("OSError: [Errno 2] No such file or directory: ~/models/x.bin",
     "OSError: [Errno 2] No such file or directory: [path]"),
    ("open '/home/tvllo/x y.db' failed", "open '[path]' failed"),
    ("line one\nline two", "line one"),
    ("worker failed: Traceback (most recent call last): File \"C:\\Users\\tvllo\\x.py\", line 3, in <module> "
     "RuntimeError: boom", "worker failed: RuntimeError: boom"),
    ("see https://pytorch.org/docs/stable/notes/cuda.html for more",
     "see https://pytorch.org/docs/stable/notes/cuda.html for more"),
    ("5 of 40 supported cameras with a ratio (12%; the gate needs 50%)",
     "5 of 40 supported cameras with a ratio (12%; the gate needs 50%)"),
])
def test_client_safe_detail(raw, safe):
    assert CP.client_safe_detail(raw) == safe


def test_a_user_name_outside_a_path_is_scrubbed_too(monkeypatch):
    monkeypatch.setattr(CP, "_user_names", lambda: {"jsmith"})
    assert CP.client_safe_detail("PermissionError: access denied for JSmith on the share") == \
        "PermissionError: access denied for [user] on the share"
    assert CP.client_safe_detail("jsmith-backup and xjsmith stay") == "jsmith-backup and xjsmith stay"


def test_this_machines_user_name_is_scrubbed():
    for name in CP._user_names():
        assert CP.client_safe_detail(f"the lock is held by {name} (pid 4)") == "the lock is held by [user] (pid 4)"


def test_owner_facing_detail_drops_class_names_only_when_there_are_some():
    assert CP.owner_facing_detail("DepthModelUnavailable: the MoGe-2 weights are not cached") == \
        "the MoGe-2 weights are not cached"
    assert CP.owner_facing_detail("the evidence gate failed (torch.OutOfMemoryError: CUDA out of memory); x") == \
        "the evidence gate failed (CUDA out of memory); x"
    assert CP.owner_facing_detail("KeyboardInterrupt") == ""
    clean = "final solve skipped: hard stop (stdin-closed) during finalization"
    assert CP.owner_facing_detail(clean) == clean
    assert len(CP.client_safe_detail("x " * 400, max_chars=CP.WHY_MAX_CHARS)) <= CP.WHY_MAX_CHARS
