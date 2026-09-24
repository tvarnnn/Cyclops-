"""Area builds (WORLD-BUILDER-COMPONENTS.md §5.4) and the finisher's owed area work (§7
rule 5).

The GPU stages are faked: `final_surface_stages` is the builder's own path and
`test_world_builder_surface_pipeline.py` / `test_world_builder_appearance.py` own it.
What is pinned here is what is PREPARED for it -- the area's own solution, cut to its
keyframes, levelled from its own depth, never scaled -- and which sessions the finisher
touches: never one without a components record.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_finish_pending as wfp  # noqa: E402
from scripts.world_build_session import StopRequest  # noqa: E402
from tower.world_builder import area_build as AB  # noqa: E402
from tower.world_builder import components as C  # noqa: E402
from tower.world_builder.records import (  # noqa: E402
    STAGE_APPEARANCE,
    STAGE_STATE_OK,
    STAGE_SURFACE,
    CameraIntrinsics,
    Session,
    World,
)
from tower.world_builder.store import WorldStore  # noqa: E402

W1, S1 = "w1", "s1"
ROOM = "0123456789abcdef"
AREA1 = "a1a1a1a1a1a1a1a1"
AREA2 = "a2a2a2a2a2a2a2a2"
FX, WIDTH, HEIGHT = 200.0, 240, 180
N_KF = 12
# The true vertical of the synthetic world: tilted, so a solve-frame `-Y` is wrong.
TRUE_UP = np.array([0.30, -0.90, 0.31]) / np.linalg.norm([0.30, -0.90, 0.31])


def _kid(i):
    return f"{S1}:{i + 1:08d}"


def _look(forward, up):
    """R_wc whose camera z is `forward` and camera -y is `up` (OpenCV)."""
    z = forward / np.linalg.norm(forward)
    y = -(up - (up @ z) * z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)


def _poses():
    """Twelve level-headed cameras in the tilted world, pitched down at a floor."""
    e1 = np.cross(TRUE_UP, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(TRUE_UP, e1)
    poses, rwcs, centres = {}, {}, {}
    for i in range(N_KF):
        yaw = 2 * math.pi * i / N_KF
        horiz = math.cos(yaw) * e1 + math.sin(yaw) * e2
        forward = horiz - 0.6 * TRUE_UP
        r_wc = _look(forward, TRUE_UP)
        c = 0.4 * horiz + 1.5 * TRUE_UP
        r_cw = r_wc.T
        poses[_kid(i)] = {"component": 0, "rotation": r_cw.reshape(-1).tolist(),
                          "translation": (-r_cw @ c).tolist(), "observations": 60}
        rwcs[_kid(i)], centres[_kid(i)] = r_wc, c
    return poses, rwcs, centres


def _floor_depth(r_wc, c):
    """Depth of the floor plane (through the origin, normal TRUE_UP) from a camera."""
    vv, uu = np.mgrid[0:HEIGHT, 0:WIDTH].astype(float)
    d_c = np.stack([(uu + 0.5 - WIDTH / 2) / FX, (vv + 0.5 - HEIGHT / 2) / FX,
                    np.ones_like(uu)], -1)
    d_w = d_c @ r_wc.T
    denom = d_w @ TRUE_UP
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -(c @ TRUE_UP) / denom
    return np.where((denom < -1e-6) & (t > 0), t, np.nan).astype(np.float32)


def _store(root, *, finalized=True, stages=None):
    from tower.world_builder.global_solve import Solution, workspace_for, write_solution

    store = WorldStore(root)
    store.write_world(World(world_id=W1, created_at=1.0, updated_at=2.0, session_ids=(S1,)))
    intr = CameraIntrinsics(source="self_calibrated", model="pinhole", fx=FX, fy=FX,
                            cx=WIDTH / 2, cy=HEIGHT / 2, calibrated_width=WIDTH,
                            calibrated_height=HEIGHT)
    store.write_session(Session(
        session_id=S1, world_id=W1, started_at=10.0, ended_at=20.0, end_reason="stop",
        intrinsics=intr,
        finalization=({"state": "complete", "final_solve": "solved", "started_at": 9.0,
                       "updated_at": 10.0, "detail": None} if finalized else None),
        stages=stages if stages is not None else {
            STAGE_SURFACE: {"state": "ok", "attempted": True},
            STAGE_APPEARANCE: {"state": "ok", "attempted": True}}))
    poses, _rwcs, centres = _poses()
    kids = [_kid(i) for i in range(N_KF)]
    # One 3-D point per camera, seen by it and its neighbour.
    xyz = np.array([centres[k] - 1.5 * TRUE_UP for k in kids], np.float32)
    obs = []
    for p in range(N_KF):
        obs += [[p, 0, p], [(p + 1) % N_KF, 1, p]]
    sol = Solution(
        solver="glomap", solved_at=5.0, input_digest="room-digest", keyframe_ids=kids,
        poses=poses, components=[{"index": 0, "images": N_KF, "points": N_KF}],
        xyz=xyz, rgb=np.full((N_KF, 3), 128, np.uint8),
        component=np.zeros(N_KF, np.int32), first_keyframe=np.zeros(N_KF, np.int32),
        track_length=np.full(N_KF, 2, np.int32), error=np.full(N_KF, 0.5, np.float32),
        observations=np.array(obs, np.int32),
        observation_xy=np.zeros((len(obs), 2), np.float32),
        camera={"fx": FX, "fy": FX, "cx": WIDTH / 2, "cy": HEIGHT / 2,
                "width": WIDTH, "height": HEIGHT})
    write_solution(workspace_for(store, W1, S1), sol)
    return store, kids


def _components(store, kids):
    doc = {"components": [
        {"id": ROOM, "state": "placed", "reason": None, "reasons": [], "shown_as": "room",
         "keyframes": 4, "capture_spans_s": [[0.0, 4.0]], "keyframe_ids": kids[:4]},
        {"id": AREA1, "state": "unplaced", "reason": "scale-mismatch",
         "reasons": ["scale-mismatch"], "shown_as": "area", "keyframes": 5,
         "capture_spans_s": [[5.0, 12.0]], "keyframe_ids": kids[4:9]},
        {"id": AREA2, "state": "unplaced", "reason": "solved-separately",
         "reasons": ["solved-separately"], "shown_as": "area", "keyframes": 3,
         "capture_spans_s": [[13.0, 19.0]], "keyframe_ids": kids[9:]},
    ]}
    path = C.components_path(store, W1, S1)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return C.read_components_record(store, W1, S1)


@pytest.fixture
def fake_depth(monkeypatch):
    """`surface_pipeline.ensure_depth_stage`, replaced by the floor's exact depth."""
    import tower.world_builder.surface_pipeline as SP
    from tower.world_builder.dense_pipeline import dense_dir

    calls = []

    def fake(store, world_id, session_id, solution, intrinsics, **kwargs):
        calls.append(kwargs)
        root = dense_dir(store, world_id, session_id)
        work = root / "work"
        (work / "depth").mkdir(parents=True, exist_ok=True)
        records = []
        for ki, kid in enumerate(solution.keyframe_ids):
            pose = solution.poses.get(kid)
            if not pose:
                continue
            r_cw = np.array(pose["rotation"]).reshape(3, 3)
            r_wc = r_cw.T
            c = -r_wc @ np.array(pose["translation"])
            np.save(work / "depth" / f"{ki:05d}_pred.npy",
                    _floor_depth(r_wc, c).astype(np.float16))
            records.append({"ki": ki, "kid": kid, "ok": True, "a": 1.0, "b": 0.0})
        align = {"kind": "depth", "records": records, "input_digest": solution.input_digest}
        (root / "align.json").write_text(json.dumps(align))
        return align, work

    monkeypatch.setattr(SP, "ensure_depth_stage", fake)
    return calls


# ---------------------------------------------------------------------------
# the vertical
# ---------------------------------------------------------------------------


def test_group_up_finds_a_tilted_floor():
    _poses_, rwcs, centres = _poses()
    kids = list(rwcs)
    K = np.array([[FX, 0, WIDTH / 2], [0, FX, HEIGHT / 2], [0, 0, 1.0]])
    normals = np.concatenate([AB.camera_normals(_floor_depth(rwcs[k], centres[k]), K)
                              @ rwcs[k].T for k in kids])
    up = AB.group_up(np.array([rwcs[k] for k in kids]), normals)
    assert up["levelled"] is True
    assert math.degrees(math.acos(min(1.0, float(np.dot(up["up"], TRUE_UP))))) < 1.0
    assert up["support"] > 0.5


def test_too_few_normals_is_not_levelled():
    _poses_, rwcs, _c = _poses()
    up = AB.group_up(np.array(list(rwcs.values())), np.zeros((10, 3)))
    assert up["levelled"] is False and up["up"] is None


# ---------------------------------------------------------------------------
# the area's own solution
# ---------------------------------------------------------------------------


def test_the_area_solution_is_its_keyframes_only_as_component_zero(tmp_path):
    from tower.world_builder.global_solve import load_solution

    store, kids = _store(tmp_path)
    room = load_solution(store, W1, S1)
    sub, members = AB.area_solution(room, kids[4:9])
    assert members == kids[4:9]
    assert set(sub.poses) == set(kids[4:9])
    assert all(p["component"] == 0 for p in sub.poses.values())
    # `ki` is the room's: the full keyframe list is kept.
    assert sub.keyframe_ids == room.keyframe_ids
    # Only points the members observe, re-indexed; observations only by members.
    member_idx = {room.keyframe_ids.index(k) for k in members}
    assert set(sub.observations[:, 0].tolist()) <= member_idx
    assert sub.observations[:, 2].max() == len(sub.xyz) - 1
    assert sub.input_digest != room.input_digest
    with pytest.raises(AB.AreaNotBuildable):
        AB.area_solution(room, ["s1:nope"])


def test_prepare_levels_the_area_from_its_own_depth_and_never_scales_it(
        tmp_path, fake_depth):
    from tower.world_builder.global_solve import load_solution

    store, kids = _store(tmp_path)
    record = _components(store, kids)
    view, levelling = AB.prepare_area(store, W1, S1, AREA1, record)
    assert levelling["levelled"] is True
    area = load_solution(view, W1, S1)
    assert set(area.poses) == set(kids[4:9])
    # In the levelled frame the world's true up is -Y.
    R = np.array(levelling["rotation"])
    assert np.allclose(R @ TRUE_UP, [0, -1, 0], atol=0.02)
    # Its cameras keep their spacing: rotated and centred, never scaled.
    room = load_solution(store, W1, S1)

    def centres(sol):
        out = []
        for k in kids[4:9]:
            r_cw = np.array(sol.poses[k]["rotation"]).reshape(3, 3)
            out.append(-r_cw.T @ np.array(sol.poses[k]["translation"]))
        return np.array(out)

    a, b = centres(area), centres(room)
    assert np.allclose(np.linalg.norm(a[1:] - a[:-1], axis=1),
                       np.linalg.norm(b[1:] - b[:-1], axis=1), atol=1e-5)
    # Written under the area, never over the room's solve.
    assert (view.area_dir / "solve" / S1 / "solution.json").exists()
    assert load_solution(store, W1, S1).input_digest == "room-digest"
    rec = C.read_area_record(store, W1, S1, AREA1)
    assert rec["levelled"] is True and rec["components_sha1"] == record.sha1
    assert rec["levelling"]["scaled"] is False


def test_an_area_being_levelled_is_running_not_owed(tmp_path, monkeypatch, fake_depth):  # noqa: F811
    """C1 E13 across the preparation: the depth pass before `surfacify` writes its own
    status is live work, and the liveness probe must see it."""
    import tower.world_builder.surface_pipeline as SP

    store, kids = _store(tmp_path)
    record = _components(store, kids)
    seen = []
    inner = SP.ensure_depth_stage

    def watching(store_, world_id, session_id, solution, intrinsics, **kwargs):
        session = store.read_session(W1, S1)
        seen.append(C.area_photographic_state(store, W1, S1, session, AREA1,
                                              record.sha1)["state"])
        return inner(store_, world_id, session_id, solution, intrinsics, **kwargs)

    monkeypatch.setattr(SP, "ensure_depth_stage", watching)
    AB.build_area(store, W1, S1, AREA1, record, final_surface_stages=_fake_stages([]))
    assert seen == ["running"]


def test_an_area_stopped_while_levelling_is_owed_again(tmp_path, monkeypatch, fake_depth):  # noqa: F811
    store, kids = _store(tmp_path)
    record = _components(store, kids)
    report = AB.build_area(store, W1, S1, AREA1, record,
                           final_surface_stages=_fake_stages([]), should_stop=lambda: True)
    assert report["built"] is False
    session = store.read_session(W1, S1)
    assert C.area_photographic_state(store, W1, S1, session, AREA1,
                                     record.sha1)["state"] == "owed"


def test_an_area_whose_vertical_cannot_be_estimated_keeps_the_solves_orientation(
        tmp_path, monkeypatch):
    import tower.world_builder.surface_pipeline as SP
    from tower.world_builder.dense_pipeline import dense_dir

    def blind(store, world_id, session_id, solution, intrinsics, **kwargs):
        root = dense_dir(store, world_id, session_id)
        (root / "work" / "depth").mkdir(parents=True, exist_ok=True)
        return {"kind": "depth", "records": []}, root / "work"

    monkeypatch.setattr(SP, "ensure_depth_stage", blind)
    store, kids = _store(tmp_path)
    record = _components(store, kids)
    _view, levelling = AB.prepare_area(store, W1, S1, AREA1, record)
    assert levelling["levelled"] is False
    assert np.allclose(levelling["rotation"], np.eye(3))


def test_an_area_the_record_names_no_keyframes_for_is_not_buildable(tmp_path):
    store, kids = _store(tmp_path)
    _components(store, kids)
    raw = json.loads(C.components_path(store, W1, S1).read_text())
    for e in raw["components"]:
        e.pop("keyframe_ids")
    C.components_path(store, W1, S1).write_text(json.dumps(raw))
    record = C.read_components_record(store, W1, S1)
    with pytest.raises(AB.AreaNotBuildable, match="does not name"):
        AB.prepare_area(store, W1, S1, AREA1, record)


# ---------------------------------------------------------------------------
# build_area: the builder's stages on the area, recorded on the area
# ---------------------------------------------------------------------------


def _fake_stages(calls):
    def fake(store, world_id, session_id, **kwargs):
        calls.append({"store": store, **kwargs})
        root = store.world_dir(world_id)
        for stage in ("surface", "appearance"):
            (root / stage / session_id).mkdir(parents=True, exist_ok=True)
            (root / stage / session_id / "manifest.json").write_text(json.dumps({"x": 1}))
        kwargs["record"](STAGE_SURFACE, state=STAGE_STATE_OK, detail=None)
        if kwargs.get("appearance"):
            kwargs["record"](STAGE_APPEARANCE, state=STAGE_STATE_OK, detail=None)
        return {"surface": {"attempted": True, "state": "ok"}}
    return fake


def test_build_area_runs_the_builders_stages_on_the_area_view(tmp_path, fake_depth):
    store, kids = _store(tmp_path)
    record = _components(store, kids)
    calls = []
    report = AB.build_area(store, W1, S1, AREA1, record,
                           final_surface_stages=_fake_stages(calls))
    assert report["built"] is True and report["levelling"]["levelled"] is True
    assert len(calls) == 1
    view = calls[0]["store"]
    assert isinstance(view, C.AreaStore) and view.area_id == AREA1
    assert calls[0]["solved"] is True
    rec = C.read_area_record(store, W1, S1, AREA1)
    assert rec["stages"]["surface"]["state"] == "ok"
    assert rec["stages"]["appearance"]["state"] == "ok"
    # The additive `area` key on both manifests (§5.3).
    for stage in ("surface", "appearance"):
        man = json.loads((view.area_dir / stage / S1 / "manifest.json").read_text())
        assert man["area"] == {"id": AREA1, "levelled": True}
    # Nothing of the room's was written.
    assert not (store.world_dir(W1) / "surface" / S1).exists()


def test_an_unbuildable_area_is_settled_not_owed(tmp_path):
    store, kids = _store(tmp_path)
    raw_record = _components(store, kids)
    raw = json.loads(C.components_path(store, W1, S1).read_text())
    raw["components"][1]["keyframe_ids"] = ["s1:nope"]
    C.components_path(store, W1, S1).write_text(json.dumps(raw))
    record = C.read_components_record(store, W1, S1)
    assert record.sha1 != raw_record.sha1
    report = AB.build_area(store, W1, S1, AREA1, record,
                           final_surface_stages=_fake_stages([]))
    assert report["built"] is False
    session = store.read_session(W1, S1)
    word = C.area_photographic_state(store, W1, S1, session, AREA1, record.sha1)
    assert word["state"] == "failed"   # attempted and could not produce: terminal


# ---------------------------------------------------------------------------
# the finisher: owed area work, and never components
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_native_warm(monkeypatch):
    monkeypatch.setattr(wfp, "prewarm_world_builder", lambda *a, **k: ())


def _no_writes_beyond(root: Path, before: set) -> list:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file() and p not in before)


def test_area_builds_are_off_unless_asked_for(monkeypatch):
    """P3-RULES rule 4: a missing setting is today's behaviour."""
    from tower.config import world_area_builds_setting

    monkeypatch.delenv("TOWER_WORLD_AREA_BUILDS", raising=False)
    assert world_area_builds_setting() is False
    for word in ("", "0", "off", "banana"):
        monkeypatch.setenv("TOWER_WORLD_AREA_BUILDS", word)
        assert world_area_builds_setting() is False
    monkeypatch.setenv("TOWER_WORLD_AREA_BUILDS", "1")
    assert world_area_builds_setting() is True


def test_the_finisher_never_computes_components(tmp_path, monkeypatch):
    """§7 rule 5: `components: null` owes nothing. A finished world without a record is
    left exactly as it is -- nothing is written, not even an `areas/` directory."""
    store, _kids = _store(tmp_path)
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setenv("TOWER_WORLD_AREA_BUILDS", "1")
    verdict = wfp.assess(store, W1, S1)
    assert verdict.owed is False and verdict.code == "nothing-interrupted"
    assert wfp.main(["--root", str(tmp_path), "--format", "json"]) == 0
    assert _no_writes_beyond(tmp_path, before) == []
    assert not (store.world_dir(W1) / "areas").exists()
    assert not C.components_path(store, W1, S1).exists()


def test_a_historical_session_with_no_stage_record_owes_no_areas(tmp_path):
    store, _kids = _store(tmp_path, stages={})
    verdict = wfp.assess(store, W1, S1)
    assert verdict.owed is False and verdict.code == "no-stage-record"


def test_areas_a_record_names_are_owed_and_agree_with_the_row(tmp_path):
    from tower.results.world_builder_library import build_world_listing

    store, kids = _store(tmp_path)
    _components(store, kids)
    verdict = wfp.assess(store, W1, S1)
    assert verdict.owed is True and verdict.code == "owed-area"
    assert verdict.stage == wfp.AREAS_STAGE
    row = build_world_listing(store)["worlds"][0]["sessions"][0]
    assert row["photographic"]["scope"] == "area"
    assert row["photographic"]["state"] == "owed"


def test_with_area_builds_off_the_finisher_declines_them_and_the_row_settles(
        tmp_path, monkeypatch):
    from tower.results.world_builder_library import build_world_listing

    monkeypatch.delenv("TOWER_WORLD_AREA_BUILDS", raising=False)
    called = []
    monkeypatch.setattr(wfp, "final_surface_stages", _fake_stages(called))
    store, kids = _store(tmp_path)
    _components(store, kids)
    assert wfp.main(["--root", str(tmp_path), "--format", "json"]) == 0
    assert called == []                       # no GPU work at all
    row = build_world_listing(store)["worlds"][0]["sessions"][0]
    assert row["photographic"] == {"state": "complete", "stage": "appearance",
                                   "detail": "the appearance stage finished",
                                   "scope": "room"}
    # Settled: no longer held on "finalizing" (this fixture has no derived tree, so
    # its settled word is `unbuilt`; what matters is that the areas no longer hold it).
    assert row["state"] != "finalizing"
    areas =[c for c in row["components"] if c["shown_as"] == "area"]
    assert [c["photographic"]["state"] for c in areas] == ["unattempted", "unattempted"]
    assert all(c["has_geometry"] is False for c in areas)
    assert wfp.assess(store, W1, S1).code == "nothing-interrupted"


def test_with_area_builds_on_the_finisher_builds_each_area_under_the_lock(
        tmp_path, monkeypatch, fake_depth):
    monkeypatch.setenv("TOWER_WORLD_AREA_BUILDS", "1")
    called = []
    stages = _fake_stages(called)

    def locked(store, world_id, session_id, **kwargs):
        holder = store.lock_holder(world_id)
        kwargs_lock = None if holder is None else holder["pid"]
        out = stages(store, world_id, session_id, **kwargs)
        called[-1]["lock_pid"] = kwargs_lock
        return out

    monkeypatch.setattr(wfp, "final_surface_stages", locked)
    store, kids = _store(tmp_path)
    _components(store, kids)
    assert wfp.main(["--root", str(tmp_path), "--format", "json"]) == 0
    assert [c["store"].area_id for c in called] == [AREA1, AREA2]
    assert all(c["lock_pid"] == os.getpid() for c in called)
    assert store.lock_holder(W1) is None
    assert wfp.assess(store, W1, S1).code == "nothing-interrupted"
    ledger = json.loads((store.world_dir(W1) / wfp.ATTEMPTS_FILENAME).read_text())
    assert ledger["sessions"][wfp.area_ledger_key(S1)]["attempts"] == 1
    assert S1 not in ledger["sessions"]      # the room's counter is untouched


def test_a_running_area_is_building_now(tmp_path):
    store, kids = _store(tmp_path)
    record = _components(store, kids)
    C.write_area_record(store, W1, S1, AREA1, components_sha1=record.sha1,
                        stage="surface", state="running")
    status = C.area_root(store, W1, S1, AREA1) / "surface" / S1 / "status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({"state": "running", "pid": os.getpid(),
                                  "updated_at": time.time()}))
    assert wfp.assess(store, W1, S1).code == "building-now"


def test_exhausted_area_work_is_retired_as_failed(tmp_path, monkeypatch):
    store, kids = _store(tmp_path)
    record = _components(store, kids)
    for _ in range(3):
        wfp.record_attempt(store, W1, wfp.area_ledger_key(S1), detail="test")
    verdict = wfp.assess(store, W1, S1)
    assert verdict.exhausted and verdict.stage == wfp.AREAS_STAGE
    monkeypatch.setattr(wfp, "final_surface_stages", _fake_stages([]))
    assert wfp.main(["--root", str(tmp_path)]) == 0
    session = store.read_session(W1, S1)
    for area_id in (AREA1, AREA2):
        word = C.area_photographic_state(store, W1, S1, session, area_id, record.sha1)
        assert word["state"] == "failed"
        assert "world_refinish.py" in word["detail"]


def test_a_stop_between_areas_stops_there(tmp_path, monkeypatch, fake_depth):
    monkeypatch.setenv("TOWER_WORLD_AREA_BUILDS", "1")
    stop = StopRequest()
    called = []
    inner = _fake_stages(called)

    def stopping(store, world_id, session_id, **kwargs):
        out = inner(store, world_id, session_id, **kwargs)
        stop.request("hard", "capture-opened")
        return out

    monkeypatch.setattr(wfp, "final_surface_stages", stopping)
    store, kids = _store(tmp_path)
    _components(store, kids)
    wfp.main(["--root", str(tmp_path)], stop_request=stop)
    assert [c["store"].area_id for c in called] == [AREA1]
    # The attempt was given back: a walk starting is not a failure.
    ledger = json.loads((store.world_dir(W1) / wfp.ATTEMPTS_FILENAME).read_text())
    assert ledger["sessions"][wfp.area_ledger_key(S1)]["attempts"] == 0
    assert wfp.assess(store, W1, S1).code == "owed-area"
