"""The consensus's withhold re-gate is constrained and checked (review V9, H-1), groups keep the keyframes every
draw attached (RV9-A P6), and the room says what it holds against the majority (review V9, M-2).

Module: `tower/world_builder/coherence_publish.py` (`gate_by_consensus`, `decide_consensus`, `withhold_refusal`,
`keep_outside_pieces`, `held_against_majority`) and `coherence_gate.apply_gate(withhold=..., room=...)`.

THE FIXTURE (RV9-A's `consensus/fx.py`, turned into a test fixture). The lane's own fixture cannot produce these
shapes -- every metric level is 0, every piece hangs on one camera, and the room is always its component's first
round -- so each piece here has its own metric level, hangs on any camera of any piece by a closed triangle of
links (`hang`), may carry further links the database puts some degrees off (`direct`), and a draw may pose any
piece rotated (the mapper put it elsewhere: its links to the rest are contradicted, the gate leaves it out).

Every rotation is about z, so a link's disagreement with a draw is |rotation offset| in degrees. Only the
database, the depth stage and the metric scale are faked; `gate_by_consensus`, `gate_final_solution` and
`apply_gate` are the product's.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP
from tower.world_builder import global_solve as GS
from tower.world_builder.records import Keyframe

SID = "s1"
SEED = 7


def _rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])


def _name(i):
    return f"{i:08d}.jpg"


def _kid(i):
    return f"{SID}:{i:08d}"


class World:
    """Pieces laid out in order; `spec[piece]` = dict(size, level, hang=(target, offset), links=k,
    direct=[(target, offset, k, degrees off)])."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.start, at = {}, 0
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
                        links[(_name(i), _name(j))] = (100, 0.0)
        for p, s in spec.items():
            if s.get("hang"):
                tp, toff = s["hang"]
                for q in range(s["links"]):
                    links[(_name(self.start[tp] + toff), _name(self.start[p] + q))] = (40, 0.0)
            for (tp, toff, k, off) in s.get("direct", ()):
                for q in range(k):
                    links[(_name(self.start[tp] + toff), _name(self.start[p] + s["size"] - 1 - q))] = (40, float(off))
        self.links_db = links
        self.keyframes = [Keyframe(keyframe_id=_kid(i), session_id=SID, source_seq=i, received_at=1000.0 + 0.5 * i,
                                   image_relpath=f"images/{i:08d}.jpg", width=320, height=240, byte_count=1)
                          for i in range(self.n)]

    def links(self):
        return {k: v[0] for k, v in self.links_db.items()}

    def rotations(self):
        return {(a, b): _rz(off + 3.0 * int(b[:8]) - 3.0 * int(a[:8])) for (a, b), (_i, off) in self.links_db.items()}

    def metric_log(self):
        return {_name(i): float(self.spec[self.piece_of[i]]["level"]) for i in range(self.n)}

    def candidate(self, rotated: dict | None = None, seed: int = SEED):
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
        poses = {_kid(i): {"component": 0, "rotation": R[i].ravel().tolist(), "translation": [0.0, 0.0, 0.0],
                           "observations": int(counts[i])} for i in range(self.n)}
        first = np.zeros(p, np.int32)
        for row in obs[::-1]:
            first[row[2]] = row[0]
        return GS.Solution(
            solver="glomap", solved_at=2000.0 + seed, input_digest="digest-final",
            keyframe_ids=[_kid(i) for i in range(self.n)], poses=poses,
            components=[{"index": 0, "images": self.n, "points": p}],
            xyz=np.zeros((p, 3), np.float32), rgb=np.zeros((p, 3), np.uint8), component=np.zeros(p, np.int32),
            first_keyframe=first, track_length=np.full(p, 3, np.int32), error=np.full(p, 0.5, np.float32),
            observations=obs, observation_xy=np.zeros((len(obs), 2), np.float32),
            camera={"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240},
            transients={"state": "applied"}, solve={"seed": seed})

    def kids(self, p):
        s = self.start[p]
        return [_kid(i) for i in range(s, s + self.spec[p]["size"])]


class _Store:
    def read_session(self, world_id, session_id):
        return types.SimpleNamespace(started_at=1000.0, intrinsics=None)


@pytest.fixture
def install(monkeypatch):
    """Fake the gate's IO with a world's database, one metric level per piece, and a depth stage."""
    def _install(world: World, metric: dict | None = None):
        links, rots = world.links(), world.rotations()
        metric = world.metric_log() if metric is None else metric
        monkeypatch.setattr(CG, "read_verified_links", lambda db, min_inliers=15: dict(links))
        monkeypatch.setattr(CG, "read_link_rotations", lambda db, cam, min_inliers=15: dict(rots))
        monkeypatch.setattr(CP, "run_gate_depth", lambda store, w, s, solution, intr, should_stop=None: (
            {"backend": "moge2-vitl", "known_fov": 42.0, "targets": len(solution.poses), "records": []},
            "work", types.SimpleNamespace(backend="moge2-vitl")))
        monkeypatch.setattr(CP, "measure_metric_scale", lambda solution, name_of, db, work: {
            "metric_log": dict(metric), "cameras_published": world.n, "cameras_measured": len(metric)})
        return world
    return _install


def _consensus(world: World, draws_rotated: list):
    plan = CP.ConsensusPlan(draws=len(draws_rotated), seed=SEED,
                            map_draw=lambda s: world.candidate(rotated=draws_rotated[s - SEED], seed=s))
    first = world.candidate(rotated=draws_rotated[0])
    return CP.gate_by_consensus(_Store(), "w1", SID, first, plan=plan, database_path="db",
                                keyframes=world.keyframes)


def _single(world: World, rotated=None):
    return CP.gate_final_solution(_Store(), "w1", SID, world.candidate(rotated=rotated), database_path="db",
                                  keyframes=world.keyframes)


def _room(result) -> set:
    return {kid for kid, p in result.solution.poses.items() if p["component"] == 0 and p["observations"] >= 30}


def _in_room(world, result, p) -> bool:
    ks = world.kids(p)
    return 2 * sum(1 for k in ks if k in _room(result)) > len(ks)


def _entries_of(result, kids) -> list:
    kids = set(kids)
    return [e for e in result.components["components"] if kids & set(e["keyframe_ids"])]


def _group(consensus, world, p):
    return next(g for g in consensus["groups"] if g["first_keyframe"] == world.kids(p)[0])


def _assert_consistent(world, result, chosen_room):
    """What every consensus must publish, whatever its state: nothing outside the chosen draw's room is in the
    published room; `detached` names exactly the pieces published `seed-unstable`, each with it alone."""
    c = result.record["consensus"]
    assert _room(result) <= chosen_room, "a keyframe no chosen room held was attached"
    unstable = [e for e in result.components["components"] if CG.REASON_SEED_UNSTABLE in e["reasons"]]
    assert all(e["reasons"] == [CG.REASON_SEED_UNSTABLE] for e in unstable), "seed-unstable is the only reason"
    assert sorted(min(e["keyframe_ids"]) for e in unstable) == sorted(c["detached"])


# ---------------------------------------------------------------------------
# H-1, the four shapes (RV9 `probe_withhold_attaches.py`; RV9-A `probe_regate.py`, `probe_room_not_first_round.py`)


def test_p1_the_withhold_never_attaches_a_piece_no_draw_attached(install):
    """G1 (level +0.2) pulls the room's median up; H (level -0.1) is refused for scale in every draw. Withheld,
    G1 no longer pulls the median, and the unconstrained re-gate attached H -- 0 of 3 votes -- as `placed`. The
    allow-list keeps H out: it stays unplaced, with the reason the published draw gave it."""
    w = install(World({
        "A": dict(size=60, level=0.0),
        "G1": dict(size=40, level=0.2, hang=("A", 10), links=5),
        "G2": dict(size=30, level=0.2, hang=("A", 20), links=5),
        "P": dict(size=50, level=0.2, hang=("A", 30), links=3),
        "Q": dict(size=50, level=0.2, hang=("A", 40), links=3),
        "H": dict(size=20, level=-0.1, hang=("A", 50), links=4),
    }))
    draws = [{}, {"G1": 40, "P": 40, "H": 40}, {"G1": 40, "Q": 40, "H": 40}]
    chosen = _single(w)
    assert not _in_room(w, chosen, "H") and _in_room(w, chosen, "G1")
    result = _consensus(w, draws)
    c = result.record["consensus"]
    assert _group(c, w, "H")["votes"] == [False, False, False]
    assert not _in_room(w, result, "H"), "H: attached by no draw, and not by the consensus either"
    assert c["state"] == CP.CONSENSUS_APPLIED and c["detached"] == [w.kids("G1")[0]]
    (h,) = _entries_of(result, w.kids("H"))
    (h0,) = _entries_of(chosen, w.kids("H"))
    assert h["state"] == "unplaced" and h["reasons"] == h0["reasons"] == [CG.REASON_SCALE_MISMATCH]
    _assert_consistent(w, result, _room(chosen))


def test_p1_on_the_lanes_layout_y_stays_where_the_published_draw_left_it(install):
    """RV9's own case (`probe_withhold_attaches.py`, case 2): every piece hangs on the room's last camera; A (a
    minority group) shifted the median Y was refused against. Y, attached by no draw, is not published placed,
    and keeps the published draw's reason, not the re-gate's fallback."""
    room, sizes = 30, {"A": 10, "B": 20, "D": 20, "E": 20, "Y": 10}
    levels = {"R": 0.0, "A": 0.2, "B": 0.2, "D": None, "E": None, "Y": -0.15}
    spec = {"R": dict(size=room, level=0.0)}
    for p, size in sizes.items():
        spec[p] = dict(size=size, level=levels[p], hang=("R", room - 1), links=2)
    w = World(spec)
    # D and E have no metric level at all
    install(w, metric={_name(i): levels[w.piece_of[i]] for i in range(w.n) if levels[w.piece_of[i]] is not None})
    result = _consensus(w, [{}, {"A": 40, "D": 40, "Y": 40}, {"A": 40, "E": 40, "Y": 40}])
    c = result.record["consensus"]
    assert _group(c, w, "Y")["votes"] == [False, False, False]
    assert not _in_room(w, result, "Y")
    (y,) = _entries_of(result, w.kids("Y"))
    assert y["state"] == "unplaced" and y["reasons"] == [CG.REASON_SCALE_MISMATCH]
    _assert_consistent(w, result, _room(_single(w)))


def test_p2_a_piece_every_draw_attached_is_never_detached(install):
    """H2's only honoured route to the room runs through G1 (its direct links to A are 25 deg off). Withholding
    G1 took H2 out with it, published `seed-unstable` although all three draws attached it. The check refuses
    that re-gate: `not-applied`, and the chosen draw is published exactly as it was gated."""
    w = install(World({
        "A": dict(size=60, level=0.0),
        "G1": dict(size=40, level=0.0, hang=("A", 10), links=5),
        "H2": dict(size=30, level=0.0, hang=("G1", 20), links=4, direct=[("A", 50, 4, 25.0)]),
        "P": dict(size=50, level=0.0, hang=("A", 30), links=3),
        "Q": dict(size=50, level=0.0, hang=("A", 40), links=3),
    }))
    result = _consensus(w, [{}, {"G1": 60, "H2": 25, "P": 40}, {"G1": 60, "H2": 25, "Q": 40}])
    c = result.record["consensus"]
    assert _group(c, w, "H2")["votes"] == [True, True, True]
    assert _in_room(w, result, "H2"), "a piece every draw attached stays in the room"
    assert c["state"] == CP.CONSENSUS_NOT_APPLIED and c["detached"] == []
    assert "left it" in c["why"]
    chosen = _single(w)
    assert result.components == chosen.components, "the chosen draw, published as it was gated"
    assert result.solution.poses == chosen.solution.poses
    assert c["held_against_majority"] == 40, "G1 (1 of 3) stays, and the record counts it"
    _assert_consistent(w, result, _room(chosen))


def test_p3_a_withheld_group_is_its_own_piece_with_seed_unstable_alone(install):
    """G1 (1 of 3) links on to H3, a larger piece every draw leaves out for scale. The unsealed re-gate let G1
    join H3's round and take H3's reasons. Sealed, G1 is a piece of its own, `seed-unstable` only; H3 keeps its
    own reasons."""
    w = install(World({
        "A": dict(size=60, level=0.0),
        "G1": dict(size=30, level=0.15, hang=("A", 10), links=5),
        "H3": dict(size=50, level=0.3, hang=("G1", 15), links=4),
        "P": dict(size=50, level=0.0, hang=("A", 30), links=3),
        "Q": dict(size=50, level=0.0, hang=("A", 40), links=3),
    }))
    result = _consensus(w, [{}, {"G1": 40, "P": 40}, {"G1": 40, "Q": 40}])
    c = result.record["consensus"]
    assert c["state"] == CP.CONSENSUS_APPLIED and c["detached"] == [w.kids("G1")[0]]
    g1 = _group(c, w, "G1")
    assert g1["decision"] == CP.DECISION_SEED_UNSTABLE
    chosen = _single(w)
    # G1's camera 15 carries H3's links: an articulation, it went to H3's larger block, out of the room in
    # every draw. The rest of G1 is the group the consensus withheld.
    withheld = (set(w.kids("G1")) & _room(chosen)) - _room(result)
    assert len(withheld) == g1["keyframes"] == 29
    for e in _entries_of(result, withheld):
        assert set(e["keyframe_ids"]) == withheld and e["reasons"] == [CG.REASON_SEED_UNSTABLE]
    h3_now = [e for e in _entries_of(result, w.kids("H3")) if not set(e["keyframe_ids"]) & withheld]
    h3_then = _entries_of(chosen, w.kids("H3"))
    assert [e["reasons"] for e in h3_now] == [e["reasons"] for e in h3_then]
    _assert_consistent(w, result, _room(chosen))


def test_p8_the_bar_holds_when_the_room_is_not_its_components_first_round(install):
    """B1, the component's largest group, attaches nothing; B2 and what hangs on it out-number it and are the
    room (label 0) after the remap. The first-round-only bar let the withheld C re-attach in the room's round
    while the record said `detached: [C]`."""
    w = install(World({
        "B1": dict(size=60, level=0.0),
        "B2": dict(size=55, level=0.0),
        "C": dict(size=30, level=0.0, hang=("B2", 10), links=5),
        "D": dict(size=30, level=0.0, hang=("B2", 20), links=5),
        "P": dict(size=40, level=0.0, hang=("B2", 30), links=3),
        "Q": dict(size=40, level=0.0, hang=("B2", 40), links=3),
    }))
    chosen = _single(w)
    rounds = chosen.gated["rounds"]
    assert rounds[0]["final_label"] != 0, "the fixture: the room is not the component's first round"
    result = _consensus(w, [{}, {"C": 40, "P": 40}, {"C": 40, "Q": 40}])
    c = result.record["consensus"]
    assert c["detached"] == [w.kids("C")[0]]
    assert not _in_room(w, result, "C"), "the record says C is detached; so does the room"
    (e,) = _entries_of(result, w.kids("C"))
    assert e["reasons"] == [CG.REASON_SEED_UNSTABLE]
    _assert_consistent(w, result, _room(chosen))


# ---------------------------------------------------------------------------
# the gate's hooks, directly


def test_the_allow_list_admits_only_the_rooms_groups_and_the_hooks_off_are_todays_gate(install):
    w = install(World({
        "A": dict(size=60, level=0.0),
        "G": dict(size=30, level=0.0, hang=("A", 10), links=5),
        "H": dict(size=30, level=0.0, hang=("A", 30), links=5),
    }))
    sol = w.candidate()
    model = CP.solve_model(sol, CP._image_names(w.keyframes))
    plain = CG.apply_gate(model, w.links(), w.metric_log(), link_rotations=w.rotations(), masks_applied=True)
    assert set(plain["labels"].values()) == {0}
    for off in ({"withhold": None, "room": None}, {"withhold": (), "room": ()}):
        assert CG.apply_gate(model, w.links(), w.metric_log(), link_rotations=w.rotations(), masks_applied=True,
                             **off) == plain
    a, g, h = (_name(w.start[p]) for p in ("A", "G", "H"))
    out = CG.apply_gate(model, w.links(), w.metric_log(), link_rotations=w.rotations(), masks_applied=True,
                        withhold={g}, room={a, g})
    labels = {nm: lab for nm, lab in out["labels"].items()}
    assert {labels[_name(i)] for i in range(w.start["A"], w.start["A"] + 60)} == {0}
    assert all(labels[_name(i)] != 0 for i in range(w.start["H"], w.start["H"] + 30)), "H: not in the list"
    assert all(labels[_name(i)] != 0 for i in range(w.start["G"], w.start["G"] + 30)), "G: withheld"
    reasons = {c["label"]: c["reasons"] for c in out["components"]}
    assert reasons[labels[g]] == [CG.REASON_SEED_UNSTABLE]
    assert CG.REASON_SEED_UNSTABLE not in reasons[labels[h]]


# ---------------------------------------------------------------------------
# RV9-A P6: a keyframe every draw attached is never withheld


def _pure_draw(kids, room, room_groups, outside):
    poses = {k: {"component": 0 if k in room else outside.get(k, 99), "observations": 40} for k in kids}
    sol = types.SimpleNamespace(poses=poses, keyframe_ids=kids)
    groups = [{"label": 0, "reference": ref, "first_camera": first.upper(), "members": [m.upper() for m in mem]}
              for first, mem, ref in room_groups]
    return CP.GateResult(solution=sol, record={"state": CP.GATE_STATE_APPLIED, "attach": True}, components=None,
                         gated={"groups": groups})


def _rng(prefix, n, start=0):
    return [f"{prefix}{i:04d}" for i in range(start, start + n)]


def test_p6_a_group_holding_keyframes_every_draw_attached_is_kept_not_withheld():
    """G (40 kf) in the published draw's room: 15 of its keyframes are in every draw's anchor block, 25 only in
    draw 0's room. Voted as one unit G gets [T, F, F], and it used to be withheld whole -- 15 unanimous keyframes
    published `seed-unstable`, and nothing said so. Decided: the gate's unit stays, decision `kept`, with the
    count of its unanimous keyframes; its 25 minority keyframes are held against the majority."""
    A, Ga, Gb, P, Q = _rng("a", 100), _rng("g", 25), _rng("g", 15, 25), _rng("p", 50), _rng("q", 50)
    kids = A + Ga + Gb + P + Q
    d0 = _pure_draw(kids, set(kids), [(A[0], A, True), (Ga[0], Ga + Gb, False), (P[0], P, False),
                                      (Q[0], Q, False)], {})
    d1 = _pure_draw(kids, set(A + Gb + Q), [(A[0], A + Gb, True), (Q[0], Q, False)],
                    {**{k: 1 for k in Ga}, **{k: 2 for k in P}})
    d2 = _pure_draw(kids, set(A + Gb + P), [(A[0], A + Gb, True), (P[0], P, False)],
                    {**{k: 1 for k in Ga}, **{k: 2 for k in Q}})
    out = CP.decide_consensus([d0, d1, d2], kid_of_name={k.upper(): k for k in kids}, min_obs=30)
    assert out["chosen"] == 0
    g = next(x for x in out["groups"] if x["first_keyframe"] == Ga[0])
    assert g["votes"] == [True, False, False] and g["keyframes"] == 40
    assert g["decision"] == CP.DECISION_KEPT and g["unanimous_keyframes"] == 15
    assert out["withhold"] == []
    room = set(kids)
    assert CP.held_against_majority(room, out["tally"], len(out["voting"])) == 25


def test_a_group_with_no_unanimous_keyframe_is_still_withheld():
    A, G, P, Q = _rng("a", 100), _rng("g", 20), _rng("p", 50), _rng("q", 50)
    kids = A + G + P + Q
    d0 = _pure_draw(kids, set(kids), [(A[0], A, True), (G[0], G, False), (P[0], P, False), (Q[0], Q, False)], {})
    d1 = _pure_draw(kids, set(A + Q), [(A[0], A, True), (Q[0], Q, False)], {**{k: 1 for k in G}, **{k: 2 for k in P}})
    d2 = _pure_draw(kids, set(A + P), [(A[0], A, True), (P[0], P, False)], {**{k: 1 for k in G}, **{k: 2 for k in Q}})
    out = CP.decide_consensus([d0, d1, d2], kid_of_name={k.upper(): k for k in kids}, min_obs=30)
    g = next(x for x in out["groups"] if x["first_keyframe"] == G[0])
    assert g["decision"] == CP.DECISION_SEED_UNSTABLE and "unanimous_keyframes" not in g
    assert out["withhold"] == [G[0].upper()]


# ---------------------------------------------------------------------------
# M-2: what the published room holds against the majority, and every disputed piece in `ambiguous`


def test_an_anchor_absorbed_piece_is_counted_and_listed_ambiguous(install):
    """X sits inside the anchor block in draw 0 (linked to A through two cameras) and is its own piece in draws 1
    and 2. Draw 0 is published (the others disagree on larger pieces); the anchor is never withheld, so X is
    published in the room against 2 of 3 draws: `held_against_majority` counts its keyframes, and X -- which
    no group of the published draw names -- is in `ambiguous` through `pieces`."""
    w = install(World({
        "A": dict(size=80, level=0.0),
        "X": dict(size=20, level=0.0, hang=("A", 10), links=3, direct=[("A", 60, 3, 0.0)]),
        "P": dict(size=40, level=0.0, hang=("A", 30), links=3),
        "Q": dict(size=40, level=0.0, hang=("A", 50), links=3),
    }))
    result = _consensus(w, [{}, {"X": 40, "P": 40}, {"X": 40, "Q": 40}])
    c = result.record["consensus"]
    assert c["chosen"]["draw"] == 0 and _in_room(w, result, "X")
    assert all(g["first_keyframe"] != w.kids("X")[0] for g in c["groups"]), "no group names X"
    (piece,) = [p for p in c["pieces"] if p["first_keyframe"] == w.kids("X")[0]]
    assert piece["against_majority"] is True
    assert w.kids("X")[0] in c["ambiguous"]
    assert c["held_against_majority"] == 20
    assert c["detached"] == []


def test_a_unanimous_consensus_holds_nothing_against_the_majority(install):
    w = install(World({
        "A": dict(size=60, level=0.0),
        "P": dict(size=40, level=0.0, hang=("A", 30), links=3),
    }))
    c = _consensus(w, [{}, {}, {}]).record["consensus"]
    assert c["state"] == CP.CONSENSUS_APPLIED and c["held_against_majority"] == 0 and c["ambiguous"] == []
