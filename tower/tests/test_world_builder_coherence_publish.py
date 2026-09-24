"""Depth before publish, the evidence gate on the final solve, and the components record
(world_builder/coherence_publish.py).

Pinned here:
  * defaults change nothing: the setting is off, and every key, digest and manifest the depth and surface
    stages write without `known_fov` is what they wrote before it existed;
  * the components record holds the contract's shape (checked with the READER's own parser,
    `components.parse_components`, so the writer and the reader cannot drift);
  * the fail-safes: masks `partial` / `unavailable` / absent -> `masks-unavailable`, depth unavailable ->
    `scale-unavailable`, a broken gate -> today's solve with no record;
  * the surface stage reuses the gate's depth (a cache hit: the network is never called);
  * saved worlds: no record is `components: null`, and a solve published without the gate retires an old
    record rather than leaving it to describe the new solve.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP
from tower.world_builder.components import parse_components

SID = "s1"


def _rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])


def _keyframes(n, t0=1000.0, dt=0.5):
    from tower.world_builder.records import Keyframe

    return [Keyframe(keyframe_id=f"{SID}:{i:08d}", session_id=SID, source_seq=i, received_at=t0 + dt * i,
                     image_relpath=f"images/{i:08d}.jpg", width=320, height=240, byte_count=1)
            for i in range(n)]


def _solution(n_a=30, n_b=20, transients=("state", "applied")):
    """Two islands of one solver component: A (the room, cameras 0..n_a-1) and B. Every camera sees 30
    tracks with each of its two neighbours, so every camera is supported; 60 points are seen by one A and
    one B camera, so the islands are coupled."""
    from tower.world_builder.global_solve import Solution

    n = n_a + n_b
    kids = [f"{SID}:{i:08d}" for i in range(n)]
    R = [_rz(3.0 * i) for i in range(n)]
    obs, p = [], 0
    for island in (range(0, n_a), range(n_a, n)):
        idx = list(island)
        for j in range(len(idx)):
            for _ in range(30):
                for c in idx[j:j + 3]:
                    obs.append([c, len(obs), p])
                p += 1
    for k in range(60):
        obs.append([k % n_a, len(obs), p])
        obs.append([n_a + k % n_b, len(obs), p])
        p += 1
    obs = np.asarray(obs, np.int32)
    counts = np.bincount(obs[:, 0], minlength=n)
    poses = {kid: {"component": 0, "rotation": R[i].ravel().tolist(), "translation": [0.0, 0.0, 0.0],
                   "observations": int(counts[i])} for i, kid in enumerate(kids)}
    first = np.zeros(p, np.int32)
    for row in obs[::-1]:
        first[row[2]] = row[0]
    return Solution(
        solver="glomap", solved_at=1234.5, input_digest="digest-final", keyframe_ids=kids, poses=poses,
        components=[{"index": 0, "images": n, "points": p}],
        xyz=np.zeros((p, 3), np.float32), rgb=np.zeros((p, 3), np.uint8), component=np.zeros(p, np.int32),
        first_keyframe=first, track_length=np.full(p, 3, np.int32), error=np.full(p, 0.5, np.float32),
        observations=obs, observation_xy=np.zeros((len(obs), 2), np.float32),
        camera={"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240},
        transients=dict([transients]) if transients else None)


def _name(i):
    return f"{i:08d}.jpg"


def _links(n_a, n_b, cross):
    n = n_a + n_b
    out = {}
    for island in (range(0, n_a), range(n_a, n)):
        idx = list(island)
        for j in range(len(idx)):
            for d in (1, 2):
                if j + d < len(idx):
                    out[(_name(idx[j]), _name(idx[j + d]))] = 100
    for a, b in cross:
        out[(_name(a), _name(b))] = 40
    return out


class _Store:
    def read_session(self, world_id, session_id):
        return SimpleNamespace(started_at=1000.0, intrinsics=None)


def _run(monkeypatch, *, n_a=30, n_b=20, cross=((29, 30),), level_b=0.0, transients=("state", "applied"),
         depth_runner=None, metric_fn=None):
    solution = _solution(n_a, n_b, transients)
    links = _links(n_a, n_b, cross)
    rots = {}
    for (a, b) in links:
        ia, ib = int(a[:8]), int(b[:8])
        rots[(a, b)] = _rz(3.0 * ib) @ _rz(3.0 * ia).T      # the solve's own relative rotation: honoured
    monkeypatch.setattr(CG, "read_verified_links", lambda db, min_inliers=15: dict(links))
    monkeypatch.setattr(CG, "read_link_rotations", lambda db, cam, min_inliers=15: dict(rots))

    def default_depth(store, world_id, session_id, solution, intrinsics, should_stop=None):
        return {"backend": "moge2-vitl", "known_fov": 42.0, "targets": n_a + n_b, "records": []}, \
            "work-dir", object()

    def default_metric(solution, name_of, database_path, work):
        ml = {_name(i): (0.0 if i < n_a else level_b) for i in range(n_a + n_b)}
        return {"metric_log": ml, "cameras_published": n_a + n_b, "cameras_measured": n_a + n_b}

    return solution, CP.gate_final_solution(
        _Store(), "w1", SID, solution, database_path="db", keyframes=_keyframes(n_a + n_b),
        depth_runner=depth_runner or default_depth, metric_fn=metric_fn or default_metric)


# Two links from B through one room image (29) whose other ends (30, 31) are linked: a closed triangle, so
# REDUNDANT evidence -- while image 29 is an articulation point, so B is still its own biconnected block and
# the attachment is a real decision of the gate.
TRIANGLE = ((29, 30), (29, 31))


def _entries(result):
    rec = parse_components(result.components)
    assert rec is not None, "the reader must accept what the writer wrote"
    return rec.entries


# ---------------------------------------------------------------------------
# the gate on the final solve


def test_a_piece_held_by_one_link_is_published_as_its_own_area(monkeypatch):
    before, result = _run(monkeypatch)
    assert result.record["state"] == CP.GATE_STATE_APPLIED
    room, area = _entries(result)
    assert (room["state"], room["shown_as"], room["reasons"], room["reason"]) == ("placed", "room", [], None)
    assert area["state"] == "unplaced" and area["reasons"] == [CG.REASON_SINGLE_UNCONFIRMED_LINK]
    assert area["shown_as"] == "area"          # 20 keyframes, but 9.5 s of capture >= 5 s
    assert area["keyframes"] == 20 and area["capture_spans_s"] == [[15.0, 24.5]]
    assert area["keyframe_ids"] == [f"{SID}:{i:08d}" for i in range(30, 50)]
    # RELABELLED: the piece is no longer the room's component in the published solve
    sol = result.solution
    assert {sol.poses[f"{SID}:{i:08d}"]["component"] for i in range(30)} == {0}
    assert {sol.poses[f"{SID}:{i:08d}"]["component"] for i in range(30, 50)} == {1}
    assert sorted(c["index"] for c in sol.components) == [0, 1]
    only_b = sol.observations[:, 0] >= 30
    b_points = np.setdiff1d(sol.observations[only_b, 2], sol.observations[~only_b, 2])
    assert set(sol.component[b_points].tolist()) == {1}
    # the candidate itself is not modified
    assert {p["component"] for p in before.poses.values()} == {0}
    # contract §2.5: the gate's parameters and digest, and what ran
    assert result.record["params"]["max_link_disagreement_deg"] == 16.8
    assert result.record["params_digest"] == CG.GateParams().digest()
    assert result.record["masks_applied"] is True and result.record["metric_available"] is True
    assert result.record["components"] == {"placed": 1, "area": 1, "none": 0}


def test_a_piece_with_redundant_links_at_the_rooms_level_stays_in_the_room(monkeypatch):
    _, result = _run(monkeypatch, cross=TRIANGLE)
    assert result.record["attach"] is True
    (room,) = _entries(result)
    assert room["state"] == "placed" and room["keyframes"] == 50
    assert {p["component"] for p in result.solution.poses.values()} == {0}


def test_a_piece_at_another_metric_level_is_refused(monkeypatch):
    _, result = _run(monkeypatch, cross=TRIANGLE, level_b=math.log(4.0))
    room, area = _entries(result)
    assert area["reasons"] == [CG.REASON_SCALE_MISMATCH]


@pytest.mark.parametrize("transients", [("state", "partial"), ("state", "unavailable"), None])
def test_masks_not_applied_attach_nothing(monkeypatch, transients):
    """Manager 011: masks are a hard dependency. `partial`, `unavailable`, or a solve that recorded none
    (masks off) all take the fail-safe, even for a piece with redundant links at the room's level."""
    _, result = _run(monkeypatch, cross=TRIANGLE, transients=transients)
    room, area = _entries(result)
    assert area["reasons"] == [CG.REASON_MASKS_UNAVAILABLE]
    assert result.record["masks_applied"] is False
    assert result.record["transients_state"] == (transients[1] if transients else None)


def test_depth_unavailable_attaches_nothing(monkeypatch):
    def no_depth(*a, **kw):
        raise CP.DepthUnavailable("DepthModelUnavailable: moge is not installed")

    _, result = _run(monkeypatch, cross=TRIANGLE, depth_runner=no_depth)
    room, area = _entries(result)
    assert area["reasons"] == [CG.REASON_SCALE_UNAVAILABLE]
    assert result.record["depth"]["state"] == CP.DEPTH_UNAVAILABLE
    assert "moge" in result.record["depth"]["detail"]
    assert result.record["metric_scale"]["state"] == "unavailable"
    assert result.depth is None


def test_a_stopped_depth_stage_is_no_depth(monkeypatch):
    def stopped(*a, **kw):
        raise CP.DepthUnavailable(f"{CP.DEPTH_STOPPED}: the depth stage was stopped after 3 frames")

    _, result = _run(monkeypatch, cross=TRIANGLE, depth_runner=stopped)
    assert result.record["depth"]["state"] == CP.DEPTH_STOPPED
    assert _entries(result)[1]["reasons"] == [CG.REASON_SCALE_UNAVAILABLE]


def test_a_broken_gate_publishes_todays_solve_and_no_record(monkeypatch):
    def broken(*a, **kw):
        raise ZeroDivisionError("a bug")

    before, result = _run(monkeypatch, metric_fn=broken)
    assert result.solution is before
    assert result.components is None
    assert result.record["state"] == CP.GATE_STATE_FAILED and "ZeroDivisionError" in result.record["detail"]


# ---------------------------------------------------------------------------
# the record


def test_ids_are_16_hex_of_the_session_and_sorted_members():
    a = CP.component_id(SID, ["s1:2", "s1:1"])
    assert a == CP.component_id(SID, ["s1:1", "s1:2"]) and len(a) == 16 and int(a, 16) >= 0
    assert a != CP.component_id("s2", ["s1:1", "s1:2"]) and a != CP.component_id(SID, ["s1:1"])


def test_spans_join_at_two_seconds_and_cap_at_eight():
    spans, total = CP.capture_spans([0.0, 1.0, 3.0, 10.0])
    assert spans == [[0.0, 3.0], [10.0, 10.0]] and total == pytest.approx(3.0)
    many = [10.0 * i + d for i in range(12) for d in (0.0, 0.5)]
    spans, total = CP.capture_spans(many)
    assert len(spans) == 8 and total == pytest.approx(6.0)
    assert all(a <= b for a, b in spans) and all(spans[i][1] < spans[i + 1][0] for i in range(7))


@pytest.mark.parametrize("state,kf,span,expected", [
    ("placed", 3, 0.0, "room"), ("unplaced", 30, 0.0, "area"), ("unplaced", 29, 5.0, "area"),
    ("unplaced", 29, 4.9, "none")])
def test_shown_as_follows_the_contracts_floor(state, kf, span, expected):
    assert CP.shown_as(state, kf, span) == expected


def test_a_small_short_piece_is_counted_only(monkeypatch):
    """A 10-camera piece captured in 4.5 s is `none`: listed and counted, never built."""
    _, result = _run(monkeypatch, n_a=30, n_b=10, cross=((29, 30),))
    room, small = _entries(result)
    assert small["shown_as"] == "none" and small["keyframes"] == 10


def test_the_document_carries_what_produced_it(monkeypatch):
    _, result = _run(monkeypatch)
    doc = result.components
    assert doc["contract"] == CP.CONTRACT and doc["session_id"] == SID
    assert doc["input_digest"] == "digest-final" and doc["solved_at"] == 1234.5
    assert doc["gate"]["params_digest"] == CG.GateParams().digest()
    # the wire fields the READER computes are not written by the gate
    assert not any(k in e for e in doc["components"] for k in ("has_geometry", "photographic",
                                                               "keyframes_phone"))


def test_after_publish_writes_the_record_once_and_a_gateless_solve_retires_it(tmp_path, monkeypatch):
    _, result = _run(monkeypatch)
    result.depth = None
    out = CP.after_publish(_Store(), "w1", SID, tmp_path, result.solution, result)
    assert out == {"components_written": True}
    written = (tmp_path / CP.COMPONENTS_FILENAME).read_bytes()
    assert parse_components(json.loads(written)) is not None
    # a later solve published WITHOUT the gate: the record is moved aside, never deleted
    assert CP.after_publish(_Store(), "w1", SID, tmp_path, result.solution, None) == {"components_retired": True}
    assert not (tmp_path / CP.COMPONENTS_FILENAME).exists()
    assert (tmp_path / CP.SUPERSEDED_FILENAME).read_bytes() == written


def test_a_world_without_a_record_is_not_computed(tmp_path):
    """Saved-world compatibility (contract §7): nothing is written for a solve without the gate, and the
    reader says `components: null` for a session with no record."""
    from tower.world_builder.components import read_components_record
    from tower.world_builder.store import WorldStore

    assert CP.after_publish(None, "w1", SID, tmp_path, None, None) == {"components_retired": False}
    assert list(tmp_path.iterdir()) == []
    assert read_components_record(WorldStore(tmp_path), "w1", SID) is None


# ---------------------------------------------------------------------------
# defaults change nothing


def test_the_gate_is_off_unless_asked_and_never_on_a_background_solve(monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_SOLVE_GATE", raising=False)
    assert CP.gate_setting_for(final=True) is False
    monkeypatch.setenv("TOWER_WORLD_SOLVE_GATE", "1")
    assert CP.gate_setting_for(final=True) is True
    assert CP.gate_setting_for(final=False) is False
    assert CP.gate_setting_for(final=True, gate=False) is False


def test_known_fov_is_invisible_when_off():
    from tower.world_builder.dense import DenseParams
    from tower.world_builder.dense_pipeline import _depth_cache_key
    from tower.world_builder.surface import SurfaceParams

    assert "known_fov" not in DenseParams().as_dict()
    assert DenseParams(known_fov=True).as_dict()["known_fov"] is True
    assert "fov" not in _depth_cache_key("d", DenseParams(), None, "t")
    assert _depth_cache_key("d", DenseParams(known_fov=True), None, "t").endswith("|fov:known")
    off = SurfaceParams().digest_fields()
    assert ("depth-fov", "known") not in off
    assert SurfaceParams(depth_known_fov=True).digest_fields() == off + (("depth-fov", "known"),)


def test_moge_is_told_the_fov_only_when_asked():
    from tower.world_builder.dense import MoGeBackend

    calls = []

    class Model:
        def infer(self, t, **kw):
            calls.append(kw)
            import torch

            return {"points": torch.ones((*t.shape[1:], 3))}

    torch = pytest.importorskip("torch")
    b = MoGeBackend("m", "moge2-vitl", "MIT")
    b._model, b._torch, b._device = Model(), torch, "cpu"
    b.predict(np.zeros((4, 5, 3), np.uint8))
    b.predict(np.zeros((4, 5, 3), np.uint8), fov_x=42.0)
    assert "fov_x" not in calls[0] and calls[1]["fov_x"] == 42.0


def test_predictions_never_cross_the_fov_mode(tmp_path):
    from tower.world_builder.dense_pipeline import FILL_RULE, reusable_predictions

    rec = {"ki": 0, "kid": "s1:0", "image_sha1": "x", "fill_rule": FILL_RULE, "ok": True}
    p = tmp_path / "align.json"
    p.write_text(json.dumps({"backend": "moge2-vitl", "records": [rec]}))
    assert reusable_predictions(p, "moge2-vitl") == {0: ("s1:0", "x")}
    assert reusable_predictions(p, "moge2-vitl", known_fov=True) == {}
    p.write_text(json.dumps({"backend": "moge2-vitl", "known_fov": 42.2, "records": [rec]}))
    assert reusable_predictions(p, "moge2-vitl") == {}
    assert reusable_predictions(p, "moge2-vitl", known_fov=True) == {0: ("s1:0", "x")}


# ---------------------------------------------------------------------------
# the surface reuses the gate's depth


def test_the_final_surface_reuses_the_gates_depth(tmp_path, monkeypatch):
    """Depth before publish -> relabel -> publish -> stamp: the surface stage's `ensure_depth_stage` finds
    the gate's depth current for the published solve and predicts nothing, and it sees the ROOM's frames
    only (the piece the gate left out is not fused into the room)."""
    from tests.test_world_builder_surface_pipeline import SESSION, WORLD, _synthetic_world
    from tower.world_builder import surface_pipeline as SP
    from tower.world_builder.dense import DenseParams
    from tower.world_builder.dense_pipeline import depth_trust_now
    from tower.world_builder.global_solve import load_solution, workspace_for, write_solution
    from tower.world_builder.surface import SurfaceParams

    store = _synthetic_world(tmp_path, digest="final-solve")
    dense = store.world_dir(WORLD) / "dense" / SESSION
    align = json.loads((dense / "align.json").read_text())
    # what the gate's depth stage leaves before publish: UNSTAMPED, told the FoV, every posed frame
    for k in ("input_digest", "digest", "cache_key"):
        align.pop(k, None)
    align["known_fov"] = 42.2
    align["redaction_trust"] = depth_trust_now(store, WORLD, SESSION)
    (dense / "align.json").write_text(json.dumps(align))

    candidate = load_solution(store, WORLD, SESSION)
    unstamped = json.loads((dense / "align.json").read_text())
    assert not SP._depth_cache_usable(unstamped, dense, candidate, DenseParams(known_fov=True),
                                      trust=unstamped["redaction_trust"]), "unstamped is no cache"

    # the gate leaves frames 6 and 7 out of the room
    labels = {kid: (1 if i >= 6 else 0) for i, kid in enumerate(candidate.keyframe_ids)}
    published = CP.relabel_solution(candidate, labels, [{"label": 0, "state": "placed", "reasons": []},
                                                        {"label": 1, "state": "unplaced",
                                                         "reasons": ["single-unconfirmed-link"]}])
    write_solution(workspace_for(store, WORLD, SESSION), published)
    dparams = CP._depth_params()
    out = CP.hand_depth_to_surface(store, WORLD, SESSION, published,
                                   {"align": unstamped, "work": dense / "work", "dparams": dparams})
    assert out["stamped"] and out["records_kept"] == 6 and out["records_outside_room"] == 2

    def must_not_predict(*a, **kw):
        raise AssertionError("the surface re-ran the depth stage")

    monkeypatch.setattr("tower.world_builder.dense_pipeline.run_depth_stage", must_not_predict)
    sp = SurfaceParams(depth_known_fov=True)
    got, work = SP.ensure_depth_stage(store, WORLD, SESSION, load_solution(store, WORLD, SESSION), None,
                                      gate_rel=sp.gate_rel, backend=None, imagery_source=sp.imagery_source,
                                      known_fov=sp.depth_known_fov)
    assert got["input_digest"] == "final-solve"
    assert sorted(r["ki"] for r in got["records"]) == list(range(6))
    # and a default (FoV-unknown) surface does NOT take it: that would mix two predictions
    assert not SP._depth_cache_usable(got, dense, load_solution(store, WORLD, SESSION), DenseParams(),
                                      trust=got["redaction_trust"])


@pytest.mark.parametrize("gated", [False, True])
def test_the_final_surface_asks_for_the_gates_depth_only_for_a_gated_solve(tmp_path, monkeypatch, gated):
    """Keyed on the published solve's gate record, not the environment (review V7, L-a): an ungated
    world gets today's call (no new keyword) even with the setting on, and a gated one the FoV-known
    depth -- the gate's -- even with it off."""
    import scripts.world_build_session as B
    from tower.world_builder import surface_pipeline
    from tower.world_builder.store import WorldStore

    monkeypatch.setenv("TOWER_WORLD_SOLVE_GATE", "0" if gated else "1")
    store = WorldStore(tmp_path)
    solve = store.world_dir("w1") / "solve" / SID
    solve.mkdir(parents=True)
    meta = {"poses": {}}
    if gated:
        meta["gate"] = {"state": CP.GATE_STATE_APPLIED}
    (solve / "solution.json").write_text(json.dumps(meta))
    seen = {}

    def surfacify(store_, world_id, session_id, **kw):
        seen.update(kw)
        return SimpleNamespace(state="unavailable", detail="stub", as_dict=lambda: {})

    monkeypatch.setattr(surface_pipeline, "surfacify", surfacify)
    B.final_surface_stages(store, "w1", SID, solved=True, appearance=False, prune_depth_work=False,
                           should_stop=lambda: False)
    assert seen["params"].depth_known_fov is False
    assert ("depth_known_fov" in seen) is gated
    if gated:
        assert seen["depth_known_fov"] is True


def test_surfacify_takes_the_fov_choice_into_its_params_digest(tmp_path):
    """`surfacify(depth_known_fov=True)` builds with `SurfaceParams(depth_known_fov=True)`: the digest says
    so, and a surface built without it is not "already built" for it."""
    from tests.test_world_builder_surface_pipeline import SESSION, WORLD, _params, _synthetic_world
    from tower.world_builder import surface_pipeline as SP

    store = _synthetic_world(tmp_path, digest="final-solve")
    SP.surfacify(store, WORLD, SESSION, params=_params(), force=True)
    before = SP.read_surface_manifest(store, WORLD, SESSION)["params_digest"]
    assert "depth-fov" not in before
    # the synthetic depth stage was not told the FoV, so the FoV-known surface must not reuse it
    dense = store.world_dir(WORLD) / "dense" / SESSION
    align = json.loads((dense / "align.json").read_text())
    from tower.world_builder.global_solve import load_solution
    from tower.world_builder.dense import DenseParams

    assert not SP._depth_cache_usable(align, dense, load_solution(store, WORLD, SESSION),
                                      DenseParams(known_fov=True))


def test_the_publish_step_with_the_gate_off_is_todays_write(tmp_path, monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_SOLVE_GATE", raising=False)
    solution = _solution()
    writes = []
    ws = SimpleNamespace(root=tmp_path)
    out, record = CP.gate_and_publish(_Store(), "w1", SID, ws, solution, final=True, database_path="db",
                                      keyframes=_keyframes(50), write=lambda w, s: writes.append(s))
    assert out is solution and record is None and writes == [solution]
    assert list(tmp_path.iterdir()) == []


def test_the_publish_step_with_the_gate_on_writes_the_gated_solve_then_the_record(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_SOLVE_GATE", "1")
    solution = _solution()
    links = _links(30, 20, ((29, 30),))
    rots = {(a, b): _rz(3.0 * int(b[:8])) @ _rz(3.0 * int(a[:8])).T for a, b in links}
    monkeypatch.setattr(CG, "read_verified_links", lambda db, min_inliers=15: dict(links))
    monkeypatch.setattr(CG, "read_link_rotations", lambda db, cam, min_inliers=15: dict(rots))

    def no_depth(*a, **kw):
        raise CP.DepthUnavailable("no GPU in this test")

    monkeypatch.setattr(CP, "run_gate_depth", no_depth)
    order = []
    ws = SimpleNamespace(root=tmp_path)

    def write(w, s):
        order.append(("write", (tmp_path / CP.COMPONENTS_FILENAME).exists()))

    out, record = CP.gate_and_publish(_Store(), "w1", SID, ws, solution, final=True, database_path="db",
                                      keyframes=_keyframes(50), write=write)
    assert order == [("write", False)], "the solution is published BEFORE its record"
    assert record["state"] == CP.GATE_STATE_APPLIED
    assert record["publish"] == {"components_written": True}
    assert out.timing["gate_s"] == record["seconds"]
    doc = json.loads((tmp_path / CP.COMPONENTS_FILENAME).read_text())
    assert [e["reasons"] for e in doc["components"]] == [[], [CG.REASON_SCALE_UNAVAILABLE]]
    # a background solve never runs the gate, whatever the setting
    out2, record2 = CP.gate_and_publish(_Store(), "w1", SID, ws, _solution(), final=False, database_path="db",
                                        keyframes=_keyframes(50), write=lambda w, s: None)
    assert record2 is None and (tmp_path / CP.SUPERSEDED_FILENAME).exists()


class _FovBackend:
    """A depth network that can be told the FoV, and records what it was told."""

    name = "moge2-vitl"
    licence = "test"
    windowed = False
    window_size = 0
    kind = "depth"
    accepts_fov = True

    def __init__(self):
        self.fov = []

    def predict(self, rgb, fov_x=None):
        self.fov.append(fov_x)
        return np.full(rgb.shape[:2], 2.0, np.float32)


def test_the_gates_depth_stage_covers_every_component_is_told_the_fov_and_is_not_a_cache(tmp_path, monkeypatch):
    """Depth before publish, for real (`run_gate_depth` -> `dense_pipeline.run_depth_stage`) on a stub
    network: every posed frame of EVERY solver component is predicted (the gate needs a level in each), the
    network is told the solve camera's FoV, and the record is left unstamped -- no stage may take it as the
    depth of a published solve until `hand_depth_to_surface` says which."""
    import dataclasses as dc

    import tower.world_builder.dense_pipeline as DP
    from tests.test_world_builder_surface_pipeline import SESSION, WORLD, _synthetic_world
    from tower.world_builder import global_solve as GS
    from tower.world_builder import surface_pipeline as SP
    from tower.world_builder.dense import DenseParams

    store = _synthetic_world(tmp_path, digest="final-solve")
    (store.world_dir(WORLD) / "dense" / SESSION / "align.json").unlink()   # no earlier predictions: predict all
    backend = _FovBackend()
    monkeypatch.setattr(DP, "make_backend", lambda name: backend)

    def identity_maps(_intrinsics, width, height):
        xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        return xs, ys, (0, 0, width, height), None

    monkeypatch.setattr(GS, "_undistort_maps", identity_maps)
    sol = GS.load_solution(store, WORLD, SESSION)
    poses = {k: dict(p, component=(1 if i >= 6 else 0)) for i, (k, p) in enumerate(sol.poses.items())}
    candidate = dc.replace(sol, poses=poses)
    align, work, dparams = CP.run_gate_depth(store, WORLD, SESSION, candidate,
                                             store.read_session(WORLD, SESSION).intrinsics)
    assert len(backend.fov) == 8, "both solver components' frames were predicted"
    cam = sol.camera
    fov = math.degrees(2 * math.atan(cam["width"] / (2 * cam["fx"])))
    assert all(f == pytest.approx(fov) for f in backend.fov)
    assert align["known_fov"] == pytest.approx(fov, abs=1e-5) and dparams.known_fov
    on_disk = json.loads((work.parent / "align.json").read_text())
    assert "input_digest" not in on_disk and "cache_key" not in on_disk
    assert not SP._depth_cache_usable(on_disk, work.parent, sol, DenseParams(known_fov=True))
    assert len(list((work / "depth").glob("*_pred.npy"))) == 8
    # the lock is released
    assert not (store.world_dir(WORLD) / "surface" / SESSION / ".surface.lock").exists()


def test_the_metric_scale_reads_the_depth_stages_frames_by_keyframe_index(tmp_path, monkeypatch):
    """The glue between the solve and the depth stage's files: `<ki>_pred.npy` is the keyframe at position
    ki of `solution.keyframe_ids`, whichever keyframes are posed. A solve at scale 2.5 reads log(2.5)."""
    from tests.test_world_builder_coherence_scale import CAM, scene
    from tower.world_builder import coherence_scale as CS
    from tower.world_builder.global_solve import Solution

    cams, pairs, depths = scene(scale=2.5)
    kids = [f"{SID}:{i:08d}" for i in range(len(cams.names) + 1)]     # keyframe 0 is not posed
    posed = kids[1:]
    poses = {kid: {"component": 0, "rotation": cams.R_cw[i].ravel().tolist(),
                   "translation": cams.t_cw[i].tolist(), "observations": 100} for i, kid in enumerate(posed)}
    sol = Solution(solver="glomap", solved_at=1.0, input_digest="d", keyframe_ids=kids, poses=poses,
                   components=[], xyz=np.zeros((0, 3), np.float32), rgb=np.zeros((0, 3), np.uint8),
                   component=np.zeros(0, np.int32), first_keyframe=np.zeros(0, np.int32),
                   track_length=np.zeros(0, np.int32), error=np.zeros(0, np.float32),
                   observations=np.zeros((0, 3), np.int32), camera=dict(CAM))
    (tmp_path / "depth").mkdir()
    name_of = {kid: f"{int(kid[-8:]) - 1:08d}.jpg" for kid in posed}          # scene() names cameras 0..5
    for ki, kid in enumerate(kids):
        if kid in name_of:
            np.save(tmp_path / "depth" / f"{ki:05d}_pred.npy", depths[name_of[kid]].astype(np.float16))
    monkeypatch.setattr(CS, "read_inlier_pairs", lambda db: pairs)
    out = CP.measure_metric_scale(sol, name_of, "db", tmp_path)
    assert out["cameras_measured"] == 6 and out["frames_without_prediction"] == 0
    for v in out["metric_log"].values():
        assert v == pytest.approx(math.log(2.5), abs=0.02)


# ---------------------------------------------------------------------------
# wired into the final solve (global_solve.solve -> coherence_publish.gate_and_publish)

from tests.test_world_builder_solve_masks import colmap, session  # noqa: E402,F401  (fixtures)


def test_the_gate_record_travels_with_the_solution_into_the_manifest(tmp_path):
    """Contract §2.5: `gate.params` and their digest are written with the published solve and reach the
    manifest's `global_solve`; a solution without the record (every older world) reads and merges as before."""
    from tests.test_world_builder_solve_masks import _bare_solution
    from tower.world_builder import global_solve as GS

    rec = {"state": CP.GATE_STATE_APPLIED, "params": CG.GateParams().to_json(),
           "params_digest": CG.GateParams().digest()}

    class _WS:
        def world_dir(self, world_id):
            return tmp_path / world_id

    for gate, name in ((rec, "w-gated"), (None, "w-old")):
        ws = GS.workspace_for(_WS(), name, "s")
        GS.write_solution(ws, _bare_solution(gate=gate))
        meta = json.loads(ws.solution_path.read_text(encoding="utf-8"))
        assert ("gate" in meta) is (gate is not None)
        loaded = GS.load_solution(_WS(), name, "s")
        assert loaded.gate == gate
        merged = GS.merge([], [], [], None, loaded, input_digest="d")
        assert merged.summary.get("gate") == gate
    assert merged.summary.keys() >= {"solver", "components"} and "gate" not in merged.summary


def _stub_gate(monkeypatch, calls):
    def fake(store, world_id, session_id, solution, *, database_path, keyframes, should_stop=None, **kw):
        calls.append({"database": Path(database_path).name, "keyframes": len(keyframes)})
        doc = {"components": [{"id": "0123456789abcdef", "state": "placed", "reason": None, "reasons": [],
                               "shown_as": "room", "keyframes": 0, "capture_spans_s": [],
                               "keyframe_ids": []}]}
        return CP.GateResult(solution=solution, record={"state": CP.GATE_STATE_APPLIED, "seconds": 0.1},
                             components=doc)

    monkeypatch.setattr(CP, "gate_final_solution", fake)


def test_the_final_solve_runs_the_gate_before_publishing_when_the_setting_is_on(session, colmap, monkeypatch):
    from tower.world_builder import global_solve as GS

    monkeypatch.setenv("TOWER_WORLD_SOLVE_GATE", "1")
    calls = []
    _stub_gate(monkeypatch, calls)
    summary = GS.solve(session.store, session.world_id, session.session_id, final=True)
    assert calls == [{"database": "database.db", "keyframes": 4}]
    assert summary["gate"]["state"] == CP.GATE_STATE_APPLIED
    meta = json.loads(session.workspace.solution_path.read_text(encoding="utf-8"))
    assert meta["gate"]["state"] == CP.GATE_STATE_APPLIED and "gate_s" in meta["timing"]
    assert parse_components(json.loads((session.workspace.root / CP.COMPONENTS_FILENAME).read_text())) \
        is not None
    # a background solve of the same session never runs it, and retires the record it would contradict
    calls.clear()
    summary = GS.solve(session.store, session.world_id, session.session_id, final=False)
    assert calls == [] and summary["gate"] is None
    assert not (session.workspace.root / CP.COMPONENTS_FILENAME).exists()


def test_with_the_setting_off_the_final_solve_publishes_as_today(session, colmap, monkeypatch):
    from tower.world_builder import global_solve as GS

    monkeypatch.delenv("TOWER_WORLD_SOLVE_GATE", raising=False)
    calls = []
    _stub_gate(monkeypatch, calls)
    summary = GS.solve(session.store, session.world_id, session.session_id, final=True)
    assert calls == [] and summary["gate"] is None
    meta = json.loads(session.workspace.solution_path.read_text(encoding="utf-8"))
    assert "gate" not in meta and "gate_s" not in meta["timing"]
    assert sorted(p.name for p in session.workspace.root.iterdir() if "components" in p.name) == []


def test_an_explicit_argument_wins_over_the_setting(session, colmap, monkeypatch):
    from tower.world_builder import global_solve as GS

    monkeypatch.delenv("TOWER_WORLD_SOLVE_GATE", raising=False)
    calls = []
    _stub_gate(monkeypatch, calls)
    GS.solve(session.store, session.world_id, session.session_id, final=True, gate=True)
    assert len(calls) == 1


def test_a_real_gate_that_cannot_read_its_database_publishes_todays_solve(session, colmap, monkeypatch):
    """The real gate on the fake solver's empty database: the gate fails, and the solve is still published,
    as the solver returned it, with `gate.state: failed` and no components record (`components: null`)."""
    from tower.world_builder import global_solve as GS

    monkeypatch.setenv("TOWER_WORLD_SOLVE_GATE", "1")
    summary = GS.solve(session.store, session.world_id, session.session_id, final=True)
    assert summary["solved"] is True
    assert summary["gate"]["state"] == CP.GATE_STATE_FAILED
    assert not (session.workspace.root / CP.COMPONENTS_FILENAME).exists()
    assert GS.load_solution(session.store, session.world_id, session.session_id).gate["state"] == "failed"


# ---------------------------------------------------------------------------
# merge honours the gate's labels: a label that splits a tracker segment keeps every keyframe it placed


def _split_world(gate):
    """Two tracker segments: 0 = keyframes 0..7, 1 = keyframes 8..11. The gate put keyframes 6 and 7 (the
    tail of segment 0) in component 1, with segment 1. Each keyframe first-observes 3 points of its own
    component."""
    from tower.world_builder.global_solve import Solution
    from tower.world_builder.records import Keyframe

    kfs = [Keyframe(keyframe_id=f"{SID}:{i:08d}", session_id=SID, source_seq=i, received_at=float(i),
                    image_relpath=f"images/{i:08d}.jpg", width=320, height=240, byte_count=1,
                    segment_index=0 if i < 8 else 1) for i in range(12)]
    comp = [0] * 6 + [1] * 6
    poses = {k.keyframe_id: {"component": comp[i], "rotation": _rz(2.0 * i).ravel().tolist(),
                             "translation": [0.1 * i, 0.0, 0.0], "observations": 50}
             for i, k in enumerate(kfs)}
    first = np.repeat(np.arange(12), 3).astype(np.int32)
    return kfs, Solution(
        solver="glomap", solved_at=1.0, input_digest="d", keyframe_ids=[k.keyframe_id for k in kfs],
        poses=poses, components=[], xyz=np.random.default_rng(0).normal(size=(36, 3)).astype(np.float32),
        rgb=np.zeros((36, 3), np.uint8), component=np.asarray([comp[i] for i in first], np.int32),
        first_keyframe=first, track_length=np.full(36, 2, np.int32), error=np.zeros(36, np.float32),
        observations=np.zeros((0, 3), np.int32), gate=gate)


def test_merge_splits_a_tracker_segment_the_gate_cut_and_keeps_every_placed_keyframe():
    from tower.world_builder import global_solve as GS

    kfs, sol = _split_world({"state": CP.GATE_STATE_APPLIED})
    merged = GS.merge(kfs, [], [], None, sol, input_digest="d")
    by_kid = {r["keyframe_id"]: r for r in merged.pose_rows}
    assert len(by_kid) == 12 and all(r["rotation"] is not None for r in by_kid.values()), \
        "every keyframe the gate placed is published"
    assert {by_kid[f"{SID}:{i:08d}"]["segment_index"] for i in range(6)} == {0}
    assert {by_kid[f"{SID}:{i:08d}"]["segment_index"] for i in (6, 7)} == {2}   # a new index past 0 and 1
    assert merged.segments[2]["split_from"] == 0 and merged.segments[2]["component"] == 1
    placement = {p.segment_index: p for p in merged.placements}
    assert placement[2].state == "registered" and placement[2].reference_segment == 1
    assert placement[2].evidence["split_from"] == 0
    assert by_kid[f"{SID}:{6:08d}"]["status"] == "anchor"
    # the split segment's points are the ones its keyframes first observed, in its own frame
    assert sum(1 for r in merged.point_rows if r["segment_index"] == 2) == 6
    comps = {c["component"]: c for c in merged.summary["components"]}
    assert comps[1]["segments"] == [1, 2] and comps[1]["keyframes"] == 6 and comps[0]["keyframes"] == 6


@pytest.mark.parametrize("gate", [None, {"state": CP.GATE_STATE_FAILED}])
def test_an_ungated_solution_merges_exactly_as_before(gate):
    from tower.world_builder import global_solve as GS

    kfs, sol = _split_world(gate)
    merged = GS.merge(kfs, [], [], None, sol, input_digest="d")
    by_kid = {r["keyframe_id"]: r for r in merged.pose_rows}
    assert {r["segment_index"] for r in merged.pose_rows} == {0, 1}
    assert by_kid[f"{SID}:{6:08d}"]["rotation"] is None and by_kid[f"{SID}:{6:08d}"]["degeneracy"] == "unregistered"
    assert set(merged.segments) == {0, 1} and not any("split_from" in v for v in merged.segments.values())


# ---------------------------------------------------------------------------
# review V7: H2 -- no unattended re-solve (masks lost to GPU memory are an owner's re-finish, said on the
# row); H1b / L-c -- a gate that could not finish is owed a re-gate IN PLACE


def _gated_session(tmp_path, *, transients, gate=None, record=True):
    from tests.test_world_builder_finish_pending import _stage, _world
    from tower.world_builder import global_solve as GS
    from tower.world_builder.records import STAGE_STATE_OK, STAGE_SURFACE

    store = _world(tmp_path, stages={STAGE_SURFACE: _stage(STAGE_STATE_OK),
                                     "appearance": _stage(STAGE_STATE_OK)})
    gate = gate if gate is not None else {"state": CP.GATE_STATE_APPLIED, "masks_applied": False}
    # A PUBLISHED solve a re-gate can start from: it loads, and the walk's database is there.
    # (Review V8, M1a: the finisher decides a refusal read-only, so a `solution.json` alone --
    # which `load_solution` reads as absent -- is `regate-refused`, not owed.)
    solution = _solution(transients=None)
    solution.transients, solution.gate = transients, gate
    workspace = GS.workspace_for(store, "w1", "s1")
    GS.write_solution(workspace, solution)
    workspace.database_path.write_bytes(b"not read: the re-gate is a stub in these tests")
    solve = workspace.root
    if record:
        (solve / CP.COMPONENTS_FILENAME).write_text(json.dumps({"components": []}))
    return store


OOM = {"state": "unavailable", "cause": "gpu-oom", "retryable": True}
DEPTH_LOST = {"state": CP.GATE_STATE_APPLIED, "masks_applied": True, "metric_available": False,
              "retryable": True, "cause": CP.CAUSE_DEPTH_UNAVAILABLE,
              "depth": {"state": "unavailable", "detail": "DepthModelUnavailable: moge is not installed"}}


def test_masks_lost_to_gpu_memory_owe_nothing_unattended_and_say_so_on_the_row(tmp_path):
    """V7 H2: no finisher re-solve. The record keeps `retryable`, and the row's finalization detail
    tells the owner to re-finish."""
    from scripts import world_finish_pending as wfp

    store = _gated_session(tmp_path, transients=OOM)
    v = wfp.assess(store, "w1", "s1")
    assert not v.owed and "masks" not in v.code
    assert not hasattr(wfp, "finish_masks_retry")
    notice = CP.publish_notice({"transients": OOM, "gate": {"state": CP.GATE_STATE_APPLIED}})
    assert notice == CP.NOTICE_MASKS_OOM
    assert "GPU out of memory" in notice and "re-finish" in notice
    assert CP.publish_notice({"transients": {"state": "applied"}, "gate": None}) is None
    assert CP.publish_notice(None) is None


@pytest.mark.parametrize("gate", [DEPTH_LOST, {"state": CP.GATE_STATE_FAILED, "retryable": True,
                                               "cause": CP.CAUSE_GATE_FAILED, "detail": "ZeroDivisionError"}])
def test_a_gate_that_could_not_finish_is_owed_a_re_gate_and_says_so(tmp_path, gate):
    from scripts import world_finish_pending as wfp

    store = _gated_session(tmp_path, transients={"state": "applied"}, gate=gate)
    v = wfp.assess(store, "w1", "s1")
    assert v.owed and v.code == "owed-regate" and v.stage == wfp.REGATE_STAGE
    notice = CP.publish_notice({"gate": gate, "transients": {"state": "applied"}})
    assert "re-runs" in notice


@pytest.mark.parametrize("case", ["ungated", "not-retryable", "masks-fail-safe"])
def test_nothing_else_is_owed_a_re_gate(tmp_path, case):
    from scripts import world_finish_pending as wfp

    if case == "ungated":
        store = _gated_session(tmp_path, transients={"state": "applied"}, record=False)
        path = store.world_dir("w1") / "solve" / "s1" / "solution.json"
        path.write_text(json.dumps({"transients": {"state": "applied"}}))
    elif case == "not-retryable":
        store = _gated_session(tmp_path, transients={"state": "applied"},
                               gate=dict(DEPTH_LOST, retryable=False))
    else:
        store = _gated_session(tmp_path, transients=OOM)
    v = wfp.assess(store, "w1", "s1")
    assert v.code != "owed-regate" and not v.owed


def test_the_gate_records_a_depth_failure_as_retryable_and_a_scale_shortfall_not(monkeypatch):
    def no_depth(*a, **kw):
        raise CP.DepthUnavailable("CUDA out of memory")

    _, result = _run(monkeypatch, cross=TRIANGLE, depth_runner=no_depth)
    assert result.record["retryable"] is True and result.record["cause"] == CP.CAUSE_DEPTH_UNAVAILABLE

    def few_levels(solution, name_of, database_path, work):
        return {"metric_log": {_name(i): 0.0 for i in range(10)}}   # 10 of 50 cameras: under half

    _, result = _run(monkeypatch, cross=TRIANGLE, metric_fn=few_levels)
    assert result.record["metric_available"] is False and result.record["retryable"] is False
    assert [e["reasons"] for e in _entries(result)][1] == [CG.REASON_SCALE_UNAVAILABLE]

    def broken(*a, **kw):
        raise ZeroDivisionError("a bug")

    _, result = _run(monkeypatch, metric_fn=broken)
    assert result.record["state"] == CP.GATE_STATE_FAILED and result.record["retryable"] is True


def test_the_re_gate_runs_in_place_from_the_solvers_partition(tmp_path, monkeypatch):
    """The published fail-safe solve is gated again IN PLACE: the solver's components come back as the
    candidate (kept per pose as `solver_component`), nothing is moved aside, the solution and its record
    are republished with the new partition and a matching solve identity."""
    from tower.world_builder import global_solve as GS
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    solution = _solution()
    # what the first publish did: the scale fail-safe split B off and recorded depth as lost
    labels = {kid: (1 if i >= 30 else 0) for i, kid in enumerate(solution.keyframe_ids)}
    published = CP.relabel_solution(solution, labels, [{"label": 0, "state": "placed", "reasons": []},
                                                       {"label": 1, "state": "unplaced",
                                                        "reasons": ["scale-unavailable"]}])
    published.gate = dict(DEPTH_LOST)
    ws = GS.workspace_for(store, "w1", SID)
    GS.write_solution(ws, published)
    (ws.root / "database.db").write_bytes(b"not read: the gate runner is a stub")
    before = sorted(p.name for p in ws.root.iterdir())
    seen = {}

    def gate_runner(store_, world_id, session_id, candidate, *, database_path, keyframes, should_stop=None):
        seen["components"] = {p["component"] for p in candidate.poses.values()}
        seen["database"] = Path(database_path).name
        relabelled = CP.relabel_solution(candidate, {k: 0 for k in candidate.keyframe_ids},
                                         [{"label": 0, "state": "placed", "reasons": []}])
        doc = CP.components_document(session_id, relabelled, {k: 0 for k in candidate.keyframe_ids},
                                     {"components": [{"label": 0, "state": "placed", "reasons": []}],
                                      "gate": CG.GATE_ID, "params": {}, "params_digest": "x",
                                      "masks_applied": True, "metric_available": True},
                                     _keyframes(50), 1000.0)
        return CP.GateResult(solution=relabelled, record={"state": CP.GATE_STATE_APPLIED,
                                                          "retryable": False, "metric_available": True},
                             components=doc)

    monkeypatch.setattr(store, "read_keyframes", lambda w, s: _keyframes(50), raising=False)
    out = CP.regate_published(store, "w1", SID, gate_runner=gate_runner)
    assert seen == {"components": {0}, "database": "database.db"}, "the solver's partition, the solve's DB"
    again = GS.load_solution(store, "w1", SID)
    assert {p["component"] for p in again.poses.values()} == {0}
    assert again.gate["regate"]["previous"]["cause"] == CP.CAUSE_DEPTH_UNAVAILABLE
    assert out["notice"] is None, "nothing owed any more"
    doc = json.loads((ws.root / CP.COMPONENTS_FILENAME).read_text())
    meta = json.loads(ws.solution_path.read_text())
    assert doc["solve_identity"] == meta["solve_identity"] == GS.solve_identity(again)
    assert sorted(p.name for p in ws.root.iterdir()) == sorted(set(before) | {CP.COMPONENTS_FILENAME})


def test_a_re_gate_that_cannot_start_is_waiting_and_spends_no_attempt(tmp_path):
    from scripts import world_finish_pending as wfp
    from scripts.world_build_session import StopRequest

    store = _gated_session(tmp_path, transients={"state": "applied"}, gate=DEPTH_LOST)

    def refusing(*a, **kw):
        raise CP.RegateRefused("the solve's database database.db is gone")

    v = wfp.assess(store, "w1", "s1")
    for _ in range(4):
        out = wfp.finish_regate(store, v, appearance=False, prune_depth_work=False,
                                stop_request=StopRequest(), regate=refusing)
        assert out["finished"] is False and out["waiting"] is True
    v = wfp.assess(store, "w1", "s1")
    assert v.owed and v.attempts == 0


def test_the_re_gate_is_bounded_and_rebuilds_the_room(tmp_path, monkeypatch):
    from scripts import world_finish_pending as wfp
    from scripts.world_build_session import StopRequest

    store = _gated_session(tmp_path, transients={"state": "applied"}, gate=DEPTH_LOST)
    calls = []
    monkeypatch.setattr(wfp.WorldBuilderEngine, "build",
                        lambda self, w, s: SimpleNamespace(poses_solved=1))

    def regate(store_, world_id, session_id, should_stop=None):
        calls.append("regate")
        return {"notice": None}

    def stages(*a, **kw):
        calls.append("room")
        return {"surface": {"state": "ok"}}

    v = wfp.assess(store, "w1", "s1")
    for _ in range(3):
        out = wfp.finish_regate(store, v, appearance=False, prune_depth_work=False,
                                stop_request=StopRequest(), regate=regate, surface_stages=stages)
        assert out["finished"] is True
    assert calls == ["regate", "room"] * 3
    v = wfp.assess(store, "w1", "s1")       # the stub never cleared the record: bounded
    assert not v.owed and v.code == "attempt-bound"


def test_a_split_segment_is_a_frame_of_its_own_for_scale_and_path_length(tmp_path):
    """V7 M1: the build counts the derived tree's segments, so a gated solve that split ONE tracker segment
    in two is two frames -- scale not `relative`, no path length summed across the split -- and the debug
    trajectory names the segment each pose is in."""
    from tower.results.world_builder import WorldBuilderStatusProducer
    from tower.world_builder import global_solve as GS
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.inspect import WorldView
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    wid = engine.create_world()
    sid = engine.start_session(wid, frame_source="synthetic", intrinsics=CameraIntrinsics(
        source="self_calibrated", model="pinhole", fx=300.0, fy=300.0, cx=160.0, cy=120.0,
        calibrated_width=320, calibrated_height=240))
    from tower.world_builder.records import Keyframe
    import cv2

    kfs = []
    for i in range(6):
        ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 40 * i, np.uint8))
        store.write_keyframe_image(wid, sid, f"{i:08d}.jpg", buf.tobytes())
        k = Keyframe(keyframe_id=f"{sid}:{i:08d}", session_id=sid, source_seq=i, received_at=float(i),
                     image_relpath=f"images/{i:08d}.jpg", width=320, height=240, byte_count=1,
                     segment_index=0)
        store.append_keyframe(wid, k)
        kfs.append(k)
    engine.stop_session("stopped")
    comp = [0, 0, 0, 1, 1, 1]
    poses = {k.keyframe_id: {"component": comp[i], "rotation": np.eye(3).ravel().tolist(),
                             "translation": [-float(i) if i < 3 else -100.0 - i, 0.0, 0.0],
                             "observations": 100} for i, k in enumerate(kfs)}
    first = np.arange(6, dtype=np.int32)
    sol = GS.Solution(solver="glomap", solved_at=1.0, input_digest="d",
                      keyframe_ids=[k.keyframe_id for k in kfs], poses=poses, components=[],
                      xyz=np.zeros((6, 3), np.float32), rgb=np.zeros((6, 3), np.uint8),
                      component=np.asarray(comp, np.int32), first_keyframe=first,
                      track_length=np.full(6, 2, np.int32), error=np.zeros(6, np.float32),
                      observations=np.zeros((0, 3), np.int32), gate={"state": CP.GATE_STATE_APPLIED})
    GS.write_solution(GS.workspace_for(store, wid, sid), sol)
    result = engine.build(wid, sid)
    manifest = store.read_derived_manifest(wid)
    assert manifest["segments"] == 2 and result.scale_state == "unknown"
    world = store.read_world(wid)
    from dataclasses import replace
    from tower.world_builder.records import ScaleState
    pl = WorldBuilderStatusProducer._compute_path_length(None, store, replace(world, scale=ScaleState(
        state="relative")), sid, manifest)
    assert pl["available"] is False
    traj = {t["keyframe_id"]: t for t in WorldView(store, wid).trajectory(sid)}
    assert traj[kfs[4].keyframe_id]["segment_index"] == 1 and traj[kfs[4].keyframe_id]["tracker_segment_index"] == 0


def test_metric_scale_needs_a_majority_of_the_supported_cameras(monkeypatch):
    """V7 H1a: a level on fewer than half the supported cameras is not a scale test: the gate takes its
    fail-safe; at half or more it tests as before. The fraction is a named, recorded parameter."""
    assert CG.GateParams().min_metric_fraction == 0.5 and "min_metric_fraction" in CG.GateParams().to_json()

    def levels(n):
        def metric(solution, name_of, database_path, work):
            return {"metric_log": {_name(i): 0.0 for i in range(n)}}
        return metric

    _, result = _run(monkeypatch, cross=TRIANGLE, metric_fn=levels(24))          # 24 of 50
    assert result.record["metric_available"] is False
    assert result.record["evidence"]["metric_fraction"] == pytest.approx(0.48)
    _, result = _run(monkeypatch, cross=TRIANGLE, metric_fn=levels(25))          # 25 of 50
    assert result.record["metric_available"] is True


def test_zero_links_is_no_verified_link_even_with_shared_points(monkeypatch):
    """V7 M2: points shared across the pieces are not a link."""
    _, result = _run(monkeypatch, cross=())
    assert _entries(result)[1]["reasons"] == [CG.REASON_NO_VERIFIED_LINK]
    assert CG.REASONS.index(CG.REASON_LINK_CONTRADICTED) < CG.REASONS.index(CG.REASON_SCALE_MISMATCH)


def test_a_depth_stamp_of_another_solve_of_the_same_keyframes_is_not_this_ones(tmp_path):
    """V7 M3: the gated solve's depth stamp names the solve; a re-solve of the same keyframes does not
    take it. An ungated solve is checked exactly as before."""
    import dataclasses as dc

    from tests.test_world_builder_surface_pipeline import SESSION, WORLD, _synthetic_world
    from tower.world_builder import surface_pipeline as SP
    from tower.world_builder.dense import DenseParams
    from tower.world_builder.global_solve import load_solution, solve_identity

    store = _synthetic_world(tmp_path, digest="final-solve")
    dense = store.world_dir(WORLD) / "dense" / SESSION
    align = json.loads((dense / "align.json").read_text())
    sol = load_solution(store, WORLD, SESSION)
    assert SP._depth_cache_usable(align, dense, sol, DenseParams()), "ungated: as before"
    gated_a = dc.replace(sol, gate={"state": "applied"})
    gated_b = dc.replace(gated_a, solved_at=gated_a.solved_at + 1.0)          # a re-solve, same keyframes
    stamped = dict(align, solve_identity=solve_identity(gated_a))
    assert SP._depth_cache_usable(stamped, dense, gated_a, DenseParams())
    assert not SP._depth_cache_usable(stamped, dense, gated_b, DenseParams())
    assert not SP._depth_cache_usable(align, dense, gated_a, DenseParams()), "unstamped is not a gated solve's"


def test_the_components_reader_refuses_a_record_of_another_solve(tmp_path, monkeypatch):
    """V7 L-d: a crash between publishing a solve and writing its record must not pair them."""
    from tower.world_builder import global_solve as GS
    from tower.world_builder.components import read_components_record
    from tower.world_builder.store import WorldStore

    _, result = _run(monkeypatch)
    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w1", SID)
    published = result.solution
    published.gate = result.record
    GS.write_solution(ws, published)
    CP.write_components(ws.root, result.components)
    assert read_components_record(store, "w1", SID) is not None
    import dataclasses as dc
    GS.write_solution(ws, dc.replace(published, solved_at=published.solved_at + 5.0))   # a newer solve
    assert read_components_record(store, "w1", SID) is None
