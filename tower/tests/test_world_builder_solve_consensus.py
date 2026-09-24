"""Attachment decided by a consensus of mapper seeds, not one draw (review V8 H2, manager 019; part C).

`TOWER_WORLD_SOLVE_CONSENSUS` = N >= 2 on a gated, seeded final solve: N candidates mapped with seeds s, s+1,
... on ONE frozen database, masks and depth; each gated; per keyframe "attached to the room" is a vote; the
draw agreeing most with the strict majority is published; a group of its room fewer than a strict majority
attached is barred from the room (`coherence_gate.apply_gate(withhold=...)`) and published as its own piece,
reason `seed-unstable`; a group the majority attached but the published draw did not stays unplaced.

The synthetic solve: a room R (cameras 0..29) and pieces hung on R's camera 29 by a closed triangle of links
-- attachable when the draw's poses honour those links. A draw that rotates a piece 40 deg (the mapper put it
elsewhere) contradicts them, and the gate leaves that piece out.
"""

from __future__ import annotations

import json
import math
import time
import types

import numpy as np
import pytest

from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP
from tower.world_builder import global_solve as GS
from tower.world_builder.records import Keyframe

SID = "s1"
ROOM = 30
PIECES = {"A": 20, "X": 10, "B": 20, "Y": 10}


def _rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])


def _layout(pieces):
    """{name: (first camera, size)} after the room, in order."""
    out, at = {}, ROOM
    for name in pieces:
        out[name] = (at, PIECES[name])
        at += PIECES[name]
    return out, at


def _name(i):
    return f"{i:08d}.jpg"


def _kid(i):
    return f"{SID}:{i:08d}"


def _candidate(pieces, rotated=(), solved_at=1234.5):
    """One draw: the room and `pieces`; a piece in `rotated` is posed 40 deg away from where the links say."""
    layout, n = _layout(pieces)
    islands = [range(0, ROOM)] + [range(s, s + size) for s, size in layout.values()]
    R = [_rz(3.0 * i) for i in range(n)]
    for name in rotated:
        s, size = layout[name]
        for i in range(s, s + size):
            R[i] = _rz(40.0) @ R[i]
    obs, p = [], 0
    for island in islands:
        idx = list(island)
        for j in range(len(idx)):
            for _ in range(30):
                for c in idx[j:j + 3]:
                    obs.append([c, len(obs), p])
                p += 1
    for s, _size in layout.values():
        for _ in range(10):                     # the piece and the room share points
            obs.append([ROOM - 1, len(obs), p])
            obs.append([s, len(obs), p])
            p += 1
    obs = np.asarray(obs, np.int32)
    counts = np.bincount(obs[:, 0], minlength=n)
    poses = {_kid(i): {"component": 0, "rotation": R[i].ravel().tolist(), "translation": [0.0, 0.0, 0.0],
                       "observations": int(counts[i])} for i in range(n)}
    first = np.zeros(p, np.int32)
    for row in obs[::-1]:
        first[row[2]] = row[0]
    return GS.Solution(
        solver="glomap", solved_at=solved_at, input_digest="digest-final", keyframe_ids=[_kid(i) for i in range(n)],
        poses=poses, components=[{"index": 0, "images": n, "points": p}],
        xyz=np.zeros((p, 3), np.float32), rgb=np.zeros((p, 3), np.uint8), component=np.zeros(p, np.int32),
        first_keyframe=first, track_length=np.full(p, 3, np.int32), error=np.full(p, 0.5, np.float32),
        observations=obs, observation_xy=np.zeros((len(obs), 2), np.float32),
        camera={"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240},
        transients={"state": "applied"}, solve={"seed": 0})


def _links(pieces):
    """The database's verified links, and their two-view rotations -- as an unrotated draw poses them."""
    layout, n = _layout(pieces)
    links = {}
    for lo, size in [(0, ROOM)] + list(layout.values()):
        for i in range(lo, lo + size):
            for j in (i + 1, i + 2):
                if j < lo + size:
                    links[(_name(i), _name(j))] = 100
    for s, _size in layout.values():
        links[(_name(ROOM - 1), _name(s))] = 40
        links[(_name(ROOM - 1), _name(s + 1))] = 40
    rots = {(a, b): _rz(3.0 * int(b[:8])) @ _rz(3.0 * int(a[:8])).T for a, b in links}
    return links, rots, n


class _Store:
    def read_session(self, world_id, session_id):
        return types.SimpleNamespace(started_at=1000.0, intrinsics=None)


def _keyframes(n):
    return [Keyframe(keyframe_id=_kid(i), session_id=SID, source_seq=i, received_at=1000.0 + 0.5 * i,
                     image_relpath=f"images/{i:08d}.jpg", width=320, height=240, byte_count=1) for i in range(n)]


@pytest.fixture
def world(monkeypatch):
    """The gate's inputs faked from one database: links, rotations, one metric level, a depth stage."""
    pieces = tuple(PIECES)
    links, rots, n = _links(pieces)
    monkeypatch.setattr(CG, "read_verified_links", lambda db, min_inliers=15: dict(links))
    monkeypatch.setattr(CG, "read_link_rotations", lambda db, cam, min_inliers=15: dict(rots))
    depth_calls = []

    def depth(store, world_id, session_id, solution, intrinsics, should_stop=None):
        depth_calls.append(sorted(solution.poses))
        return ({"backend": "moge2-vitl", "known_fov": 42.0, "targets": len(solution.poses), "records": []},
                "work", object())

    monkeypatch.setattr(CP, "run_gate_depth", depth)
    monkeypatch.setattr(CP, "measure_metric_scale", lambda solution, name_of, db, work: {
        "metric_log": {_name(i): 0.0 for i in range(n)}, "cameras_published": n, "cameras_measured": n})
    return types.SimpleNamespace(pieces=pieces, n=n, keyframes=_keyframes(n), depth_calls=depth_calls)


def _plan(draws_rotated, seed=7):
    calls = []

    def map_draw(s):
        calls.append(s)
        return _candidate(tuple(PIECES), rotated=draws_rotated[s - seed])

    return CP.ConsensusPlan(draws=len(draws_rotated), seed=seed, map_draw=map_draw), calls


def _consensus(world, draws_rotated, seed=7, **kw):
    plan, calls = _plan(draws_rotated, seed)
    first = _candidate(world.pieces, rotated=draws_rotated[0])
    result = CP.gate_by_consensus(_Store(), "w1", SID, first, plan=plan, database_path="db",
                                  keyframes=world.keyframes, **kw)
    return result, calls


def _room(result):
    return sorted(kid for kid, p in result.solution.poses.items() if p["component"] == 0)


def _piece_kids(name):
    layout, _n = _layout(tuple(PIECES))
    s, size = layout[name]
    return [_kid(i) for i in range(s, s + size)]


def _entry(result, name):
    first = _piece_kids(name)[0]
    return next(e for e in result.components["components"] if first in e["keyframe_ids"])


# ---------------------------------------------------------------------------
# the setting


def test_the_setting_is_one_draw_unless_asked(monkeypatch):
    from tower.config import world_solve_consensus_setting

    monkeypatch.delenv("TOWER_WORLD_SOLVE_CONSENSUS", raising=False)
    assert world_solve_consensus_setting() == 1
    for value, expected in (("3", 3), (" 5 ", 5), ("1", 1), ("0", 1), ("-2", 1), ("three", 1), ("", 1)):
        monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS", value)
        assert world_solve_consensus_setting() == expected, value


def test_only_a_gated_final_solve_reads_it(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS", "3")
    assert GS.consensus_requested(final=True, gated=True) == 3
    assert GS.consensus_requested(final=True, gated=False) == 1
    assert GS.consensus_requested(final=False, gated=True) == 1
    assert GS.consensus_requested(final=False, gated=False, consensus=4) == 4


# ---------------------------------------------------------------------------
# the gate's hook


def test_withholding_bars_a_group_from_the_room_and_says_seed_unstable(world):
    from tower.world_builder.coherence_publish import _image_names, solve_model

    sol = _candidate(world.pieces)
    model = solve_model(sol, _image_names(world.keyframes))
    links, rots, _ = _links(world.pieces)
    metric = {_name(i): 0.0 for i in range(world.n)}
    plain = CG.apply_gate(model, links, metric, link_rotations=rots, masks_applied=True)
    assert CG.apply_gate(model, links, metric, link_rotations=rots, masks_applied=True, withhold=()) == plain
    assert set(plain["labels"].values()) == {0}, "every piece attached"
    room_groups = [g for g in plain["groups"] if g["label"] == 0]
    assert sum(len(g["members"]) for g in room_groups) == world.n
    assert [g["reference"] for g in room_groups].count(True) == 1
    x_first = _name(_layout(world.pieces)[0]["X"][0])
    held = CG.apply_gate(model, links, metric, link_rotations=rots, masks_applied=True, withhold={x_first})
    x = [n for n, lab in held["labels"].items() if lab != 0]
    assert sorted(x) == sorted(_name(int(k[-8:])) for k in _piece_kids("X"))
    (piece,) = [c for c in held["components"] if c["state"] == "unplaced"]
    assert piece["reasons"] == [CG.REASON_SEED_UNSTABLE], "the only reason"
    # the room's anchor group cannot be withheld: it is the reference, never "attached"
    anchor = next(g["first_camera"] for g in room_groups if g["reference"])
    assert CG.apply_gate(model, links, metric, link_rotations=rots, masks_applied=True,
                         withhold={anchor})["labels"] == plain["labels"]


# ---------------------------------------------------------------------------
# the consensus


def test_the_cycle_identical_draws_publish_the_single_draw(world, monkeypatch):
    monkeypatch.setattr(time, "perf_counter", lambda: 100.0)
    single = CP.gate_final_solution(_Store(), "w1", SID, _candidate(world.pieces), database_path="db",
                                    keyframes=world.keyframes)
    result, calls = _consensus(world, [(), (), ()])
    assert calls == [8, 9]
    record = dict(result.record)
    consensus = record.pop("consensus")
    assert record == single.record
    assert result.components == single.components
    assert result.solution.poses == single.solution.poses
    assert consensus["state"] == CP.CONSENSUS_APPLIED and consensus["detached"] == []
    assert consensus["chosen"] == {"draw": 0, "seed": 7}
    assert consensus["ambiguous"] == [] and all(not g["ambiguous"] for g in consensus["groups"])
    assert consensus["votes"]["unanimous"] == consensus["votes"]["keyframes"] == world.n


def test_a_piece_only_a_minority_attached_is_detached_seed_unstable(world):
    """Draw 0 attaches A, X and B; draw 1 leaves A and X out, draw 2 B and X. The majority attaches A and B,
    not X. Draw 0 disagrees on X alone (10 keyframes), the others on 20 or more: draw 0 is published, and X --
    which it attached -- leaves its room as `seed-unstable`."""
    result, calls = _consensus(world, [(), ("A", "X"), ("B", "X")])
    c = result.record["consensus"]
    assert c["chosen"] == {"draw": 0, "seed": 7}
    x = _entry(result, "X")
    assert x["state"] == "unplaced" and x["reasons"] == [CG.REASON_SEED_UNSTABLE]
    assert _room(result) == sorted(set(_kid(i) for i in range(world.n)) - set(_piece_kids("X")))
    groups = {g["first_keyframe"]: g for g in c["groups"]}
    gx, ga, gb = (groups[_piece_kids(n)[0]] for n in ("X", "A", "B"))
    assert gx["decision"] == CP.DECISION_SEED_UNSTABLE and gx["votes"] == [True, False, False]
    assert gx["ambiguous"] is True and gx["keyframes"] == 10
    assert ga["decision"] == gb["decision"] == CP.DECISION_ATTACHED and ga["ambiguous"] and gb["ambiguous"]
    assert groups[_kid(0)]["decision"] == CP.DECISION_ANCHOR
    assert c["detached"] == [_piece_kids("X")[0]]
    assert [d["room_keyframes"] for d in c["draws"]] == [world.n, world.n - 30, world.n - 30]
    assert [d["seed"] for d in c["draws"]] == [7, 8, 9] and all(d["solve_identity"] for d in c["draws"])
    assert c["unit"] == CP.DRAW_UNIT_MAPPER_SEED and c["seeds"] == [7, 8, 9]
    # the published depth is the chosen draw's; nothing was predicted again for the re-gate
    assert len(world.depth_calls) == 3
    assert result.consensus_detail["draws"][1]["rounds"], "each draw's per-round decisions are kept"


def test_a_piece_the_majority_attached_but_the_published_draw_did_not_stays_out(world):
    """Draw 0 leaves Y out; draws 1 and 2 attach Y but each leaves out a larger piece. Draw 0 agrees most and is
    published: Y is not invented into its room, and is recorded as ambiguous."""
    result, _ = _consensus(world, [("Y",), ("A",), ("B",)])
    c = result.record["consensus"]
    assert c["chosen"]["draw"] == 0
    y = _entry(result, "Y")
    assert y["state"] == "unplaced" and CG.REASON_SEED_UNSTABLE not in y["reasons"]
    g = next(g for g in c["groups"] if g["first_keyframe"] == _piece_kids("Y")[0])
    assert g["decision"] == CP.DECISION_UNPLACED and g["votes"] == [False, True, True] and g["ambiguous"]
    assert c["detached"] == []


def test_ties_go_to_the_lowest_seed(world):
    result, _ = _consensus(world, [("A",), ("B",), ()])
    # A 2/3 attached, B 2/3 attached: draw 2 agrees fully; with draws 0 and 1 tied behind it, draw 2 wins
    assert result.record["consensus"]["chosen"]["draw"] == 2
    tie, _ = _consensus(world, [("A",), ("A",), ()])
    # A 1/3: draws 0 and 1 agree fully (tied), draw 2 does not: the lowest seed, draw 0
    assert tie.record["consensus"]["chosen"] == {"draw": 0, "seed": 7}


def test_a_draw_that_cannot_be_mapped_does_not_vote(world):
    plan, _ = _plan([(), (), ()])
    real = plan.map_draw

    def flaky(s):
        if s == 8:
            raise RuntimeError("the mapper crashed")
        return real(s)

    plan.map_draw = flaky
    result = CP.gate_by_consensus(_Store(), "w1", SID, _candidate(world.pieces), plan=plan, database_path="db",
                                  keyframes=world.keyframes)
    c = result.record["consensus"]
    assert c["draws"][1]["failed"] == "RuntimeError: the mapper crashed"
    assert c["votes"]["draws"] == 2 and c["state"] == CP.CONSENSUS_APPLIED


@pytest.mark.parametrize("case", ["masks", "depth", "unseeded"])
def test_nothing_to_vote_on_maps_no_draw(world, monkeypatch, case):
    plan, calls = _plan([(), ("X",), ("X",)])
    first = _candidate(world.pieces)
    if case == "masks":
        first.transients = {"state": "partial"}
    elif case == "depth":
        def no_depth(*a, **k):
            raise CP.DepthUnavailable("CUDA out of memory")

        monkeypatch.setattr(CP, "run_gate_depth", no_depth)
    else:
        plan = CP.ConsensusPlan(draws=3, seed=None, refusal="the solve is not seeded")
    result = CP.gate_by_consensus(_Store(), "w1", SID, first, plan=plan, database_path="db",
                                  keyframes=world.keyframes)
    assert calls == []
    c = result.record["consensus"]
    assert c["state"] == {"masks": CP.CONSENSUS_NOT_NEEDED, "depth": CP.CONSENSUS_DEFERRED,
                          "unseeded": CP.CONSENSUS_NOT_RUN}[case]
    assert c["why"]
    assert result.consensus_detail is None


def test_the_decision_is_pure_and_reads_only_the_draws():
    """`decide_consensus` over hand-made draws: a failed gate does not vote."""
    def draw(room, state=CP.GATE_STATE_APPLIED):
        # outside the room, c and d are two pieces (labels 1 and 2)
        sol = types.SimpleNamespace(poses={k: {"component": 0 if k in room else (1 if k == "c" else 2),
                                               "observations": 40}
                                           for k in ("a", "b", "c", "d")}, keyframe_ids=["a", "b", "c", "d"])
        groups = [{"label": 0, "reference": True, "first_camera": "A", "members": ["A", "B"]}]
        if "c" in room:
            groups.append({"label": 0, "reference": False, "first_camera": "C", "members": ["C"]})
        return CP.GateResult(solution=sol, record={"state": state}, components=None, gated={"groups": groups})

    kid_of_name = {"A": "a", "B": "b", "C": "c", "D": "d"}
    out = CP.decide_consensus([draw({"a", "b", "c"}), draw({"a", "b"}), draw({"a", "b", "c"}, state="failed")],
                              kid_of_name=kid_of_name, min_obs=30)
    assert out["voting"] == [0, 1]
    # c: 1 of 2 -- not a strict majority; draw 1 agrees on every keyframe
    assert out["chosen"] == 1 and out["withhold"] == []
    c = next(g for g in out["groups"] if g["first_keyframe"] == "c")
    assert c["decision"] == CP.DECISION_UNPLACED and c["votes"] == [True, False] and c["ambiguous"]
    # with draw 0 alone agreeing best, c -- attached by it, by 1 of 3 -- is withheld
    out = CP.decide_consensus([draw({"a", "b", "c"}), draw({"a"}), draw({"a", "b"})],
                              kid_of_name=kid_of_name, min_obs=30)
    assert out["chosen"] == 2 and out["withhold"] == []
    out = CP.decide_consensus([draw({"a", "b", "c"}), draw({"a", "b", "c", "d"}), draw({"a", "b"})],
                              kid_of_name=kid_of_name, min_obs=30)
    # a, b 3/3; c 2/3 attached; d 1/3: draw 0 is right on all four, and d is not in its room
    assert out["chosen"] == 0 and out["withhold"] == []


def test_flips_inside_the_published_anchor_are_reported_as_pieces():
    """6839fb8f's closet: its own piece in one draw, inside the room's anchor block in the others -- no group of
    the published draw names it. Every other draw's disputed piece is listed with its votes; one the published
    anchor holds against the majority says so (the anchor is never withheld)."""
    kids = ["a", "b", "c", "x1", "x2", "y1", "y2"]

    def draw(room, outside):
        poses = {k: {"component": 0 if k in room else outside[k], "observations": 40} for k in kids}
        sol = types.SimpleNamespace(poses=poses, keyframe_ids=kids)
        groups = [{"label": 0, "reference": True, "first_camera": "A",
                   "members": [k.upper() for k in kids if k in room]}]
        return CP.GateResult(solution=sol, record={"state": CP.GATE_STATE_APPLIED}, components=None,
                             gated={"groups": groups})

    draws = [draw(set(kids), {}),
             draw({"a", "b", "y1", "y2"}, {"c": 1, "x1": 2, "x2": 2}),
             draw({"a", "b", "x1", "x2"}, {"c": 1, "y1": 2, "y2": 2})]
    out = CP.decide_consensus(draws, kid_of_name={k.upper(): k for k in kids}, min_obs=30)
    assert out["chosen"] == 0 and out["withhold"] == [], "c is inside the anchor: it cannot be withheld"
    pieces = {p["first_keyframe"]: p for p in out["pieces"]}
    assert pieces["c"]["votes"] == [True, False, False] and pieces["c"]["against_majority"] is True
    assert pieces["x1"]["votes"] == [True, False, True] and pieces["x1"]["against_majority"] is False
    assert pieces["x1"]["keyframes"] == 2 and pieces["x1"]["from_draw"] == 1
    assert pieces["y1"]["from_draw"] == 2


def test_a_disputed_piece_the_published_draw_names_is_reported_once_as_its_group(world):
    """A piece one draw leaves out and two attach, which is its own group in the published draw: placed, and
    reported once -- as the published draw's ambiguous group, not again as a piece."""
    result, _ = _consensus(world, [(), (), ("A",)])
    c = result.record["consensus"]
    assert c["detached"] == [] and _entry(result, "A")["state"] == "placed"
    g = next(g for g in c["groups"] if g["first_keyframe"] == _piece_kids("A")[0])
    assert g["votes"] == [True, True, False] and g["ambiguous"] and g["decision"] == CP.DECISION_ATTACHED
    assert c["pieces"] == []


# ---------------------------------------------------------------------------
# wired into the final solve


def test_the_final_solve_maps_each_draw_on_its_database_with_the_next_seed(walk, engines, colmap, monkeypatch):
    from tests.test_world_builder_solve_masks import StubDetector

    asked = []
    real = GS._map_candidate

    def recording(pycolmap, database_path, workspace, sparse_dir, keyframes, *, seed, threads, **kw):
        asked.append({"seed": seed, "threads": threads, "database": database_path.name,
                      "sparse": sparse_dir.relative_to(workspace.root).as_posix()})
        return real(pycolmap, database_path, workspace, sparse_dir, keyframes, seed=seed, threads=threads, **kw)

    monkeypatch.setattr(GS, "_map_candidate", recording)
    summary = GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True, seed=5, gate=True,
                       consensus=3, transient_backend_factory=StubDetector(), mask_device_probe=lambda: None,
                       input_digest="walk-digest")
    mapped = summary["solve"]["database"]
    assert asked == [{"seed": 5, "threads": 1, "database": mapped, "sparse": "sparse"},
                     {"seed": 6, "threads": 1, "database": mapped, "sparse": "sparse-draws/seed-6"},
                     {"seed": 7, "threads": 1, "database": mapped, "sparse": "sparse-draws/seed-7"}]
    c = summary["gate"]["consensus"]
    assert c["state"] == CP.CONSENSUS_APPLIED and c["seeds"] == [5, 6, 7]
    detail = json.loads((walk.workspace.root / CP.CONSENSUS_FILENAME).read_text(encoding="utf-8"))
    assert [d["seed"] for d in detail["draws"]] == [5, 6, 7]
    assert all(d["rounds"] for d in detail["draws"])


def test_an_unseeded_solve_does_not_run_a_consensus(walk, engines, colmap):
    from tests.test_world_builder_solve_masks import StubDetector

    summary = GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True, gate=True,
                       consensus=3, transient_backend_factory=StubDetector(), mask_device_probe=lambda: None)
    c = summary["gate"]["consensus"]
    assert c["state"] == CP.CONSENSUS_NOT_RUN and "not seeded" in c["why"]
    assert not (walk.workspace.root / CP.CONSENSUS_FILENAME).exists()


def test_the_cycle_through_the_final_solve_is_byte_identical_but_for_the_consensus(
        walk, engines, colmap, monkeypatch):
    """Draws that all agree (the fake mapper ignores the seed) publish exactly what one draw publishes:
    `solution.json` and `components.json` byte for byte, apart from `gate.consensus`."""
    from tower.world_builder import solve_masks as SM
    from tests.test_world_builder_solve_masks import StubDetector

    monkeypatch.setattr(time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)
    monkeypatch.setattr(SM, "new_masked_database_path",
                        lambda workspace: workspace.root / "database.masked.pX.fixed.db")

    def finish(n):
        GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True, seed=0, gate=True,
                 consensus=n, transient_backend_factory=StubDetector(), mask_device_probe=lambda: None,
                 input_digest="walk-digest")
        ws = walk.workspace
        return ws.solution_path.read_bytes(), (ws.root / CP.COMPONENTS_FILENAME).read_bytes()

    finish(1)                       # the world's first: it matches, then its matching is frozen
    one_solution, one_components = finish(1)
    three_solution, three_components = finish(3)
    assert three_components == one_components
    one, three = json.loads(one_solution), json.loads(three_solution)
    consensus = three["gate"].pop("consensus")
    assert consensus["state"] == CP.CONSENSUS_APPLIED and consensus["detached"] == []
    assert json.dumps(three, sort_keys=True) == json.dumps(one, sort_keys=True)
    assert "consensus" not in one["gate"]


def test_with_the_setting_unset_the_final_solve_is_todays(walk, engines, colmap, monkeypatch):
    from tests.test_world_builder_solve_masks import StubDetector

    monkeypatch.delenv("TOWER_WORLD_SOLVE_CONSENSUS", raising=False)
    calls = []
    real = CP.gate_by_consensus
    monkeypatch.setattr(CP, "gate_by_consensus", lambda *a, **k: calls.append(1) or real(*a, **k))
    summary = GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True, seed=0, gate=True,
                       transient_backend_factory=StubDetector(), mask_device_probe=lambda: None)
    assert calls == [] and "consensus" not in summary["gate"]
    assert not (walk.workspace.root / CP.CONSENSUS_FILENAME).exists()


# ---------------------------------------------------------------------------
# the re-gate in place runs a consensus the solve deferred


def test_a_deferred_consensus_is_run_by_the_re_gate(world, tmp_path, monkeypatch):
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    monkeypatch.setattr(store, "read_session", _Store().read_session)
    monkeypatch.setattr(store, "read_keyframes", lambda w, s: world.keyframes)
    ws = GS.workspace_for(store, "w1", SID)
    published = _candidate(world.pieces)
    published.gate = {"state": CP.GATE_STATE_APPLIED, "retryable": True, "cause": CP.CAUSE_DEPTH_UNAVAILABLE,
                      "consensus": {"state": CP.CONSENSUS_DEFERRED, "requested": 3, "seeds": [7, 8, 9]}}
    GS.write_solution(ws, published)
    ws.database_path.write_bytes(b"features")
    asked = []

    def mapper(store_, world_id, session_id, database_path, base, keyframes=None):
        def map_draw(s):
            asked.append((s, database_path.name))
            return _candidate(world.pieces, rotated={8: ("A", "X"), 9: ("B", "X")}[s])
        return map_draw

    monkeypatch.setattr(GS, "frozen_draw_mapper", mapper)
    out = CP.regate_published(store, "w1", SID)
    assert asked == [(8, "database.db"), (9, "database.db")]
    assert out["gate"]["state"] == CP.GATE_STATE_APPLIED
    regated = GS.load_solution(store, "w1", SID)
    c = regated.gate["consensus"]
    assert c["state"] == CP.CONSENSUS_APPLIED and c["detached"] == [_piece_kids("X")[0]]
    assert regated.gate["regate"]["previous"]["cause"] == CP.CAUSE_DEPTH_UNAVAILABLE


from tests.test_world_builder_reproducible_finish import engines, walk  # noqa: E402,F401 -- fixtures
from tests.test_world_builder_solve_masks import colmap  # noqa: E402,F401 -- fixture
