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


def test_too_few_normals_falls_back_to_the_level_head():
    """P2-R5's fallback (review V6, L5): the roll-free up of the cameras, never upside down."""
    _poses_, rwcs, _c = _poses()
    R = np.array(list(rwcs.values()))
    up = AB.group_up(R, np.zeros((10, 3)))
    assert up["up"] is not None and "roll-free" in up["source"]
    assert float(np.dot(up["up"], TRUE_UP)) > 0, "never upside down"
    rf = AB.roll_free_up(R)
    assert up["levelled"] is bool(rf["well_conditioned"])


def test_a_level_headed_group_without_normals_is_levelled_by_its_head():
    """Cameras panning a level head around a known vertical: the fallback recovers it and levels."""
    up_true = np.array([0.3, -0.9, 0.2])
    up_true /= np.linalg.norm(up_true)
    base = AB.rot_between(np.array([0.0, -1.0, 0.0]), up_true)   # camera -Y -> up_true
    Rwc = []
    for yaw in np.radians(np.arange(0, 360, 20)):
        c, s_ = math.cos(yaw), math.sin(yaw)
        Ry = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])      # yaw about camera -Y (up)
        Rwc.append(base @ Ry)
    up = AB.group_up(np.array(Rwc), np.zeros((0, 3)))
    assert up["levelled"] is True
    assert math.degrees(math.acos(min(1.0, float(np.dot(up["up"], up_true))))) < 0.5


def test_too_few_cameras_and_normals_is_not_levelled():
    _poses_, rwcs, _c = _poses()
    up = AB.group_up(np.array(list(rwcs.values()))[:4], np.zeros((10, 3)))
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


def test_an_area_without_depth_normals_is_levelled_by_the_head_not_left_upside_down(
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
    # no depth normals at all: the level head levels it (review V6, L5), never upside down
    assert levelling["up"] is not None and "roll-free" in levelling["source"]
    R = np.asarray(levelling["rotation"])
    assert float(np.dot(R @ np.asarray(levelling["up"]), AB.LEVEL_UP)) > 0.999


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
    # The additive `area` key on both manifests (§5.3), in the area's own directory: the
    # stages ran in the short build directory and were moved into place.
    final = C.area_root(store, W1, S1, AREA1)
    assert view.area_dir == AB.area_build_dir(store, AREA1) and not view.area_dir.exists()
    assert sorted(report["published"]) == ["appearance", "dense", "solve", "surface"]
    for stage in ("surface", "appearance"):
        man = json.loads((final / stage / S1 / "manifest.json").read_text())
        assert man["area"] == {"id": AREA1, "levelled": True}
    assert (final / "solve" / S1 / "solution.json").exists()
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


# ---------------------------------------------------------------------------
# paths: an area build must not near Windows' MAX_PATH (lead, round 2, item 2)
# ---------------------------------------------------------------------------

# The live Tower's root on this machine, `C:\Users\tvllo\Projects\Glasses\tower\data\world_builder`
# (57 characters), and a worst case: a Tower under a longer user profile and project path.
LIVE_ROOT_CHARS = 57
WORST_ROOT_CHARS = 90
# The deepest names the real stages write, measured on a real area build (run P3-PG re-finish of the
# target): the appearance's chunks and proxy (`c.`/`p.` + a 32-hex digest + `.bin`), the transient
# detector's cache, a surface level. Written through the product's own atomic writer, so its staging
# names are measured too.
REAL_DEEPEST = [("appearance", "c." + "f" * 32 + ".bin"), ("appearance", "p." + "e" * 32 + ".bin"),
                ("dense", "work/depth/00206_transient.oneformer.npz"),
                ("surface", "mesh_l2." + "d" * 20 + ".bin")]


def test_no_path_an_area_build_writes_nears_max_path(tmp_path, monkeypatch, fake_depth):  # noqa: F811
    import tower.storage as ST

    wid, sid, aid = "a" * 32, "b" * 32, AREA1               # the real id lengths (32, 32, 16)
    monkeypatch.setitem(globals(), "W1", wid)
    monkeypatch.setitem(globals(), "S1", sid)
    store, kids = _store(tmp_path)
    record = _components(store, kids)
    written = []
    real_staging = ST.staging_path

    def spy(path):
        out = real_staging(path)
        written.append(Path(out))
        return out

    monkeypatch.setattr(ST, "staging_path", spy)

    def stages(store_, world_id, session_id, **kwargs):
        root = store_.world_dir(world_id)
        for stage, name in REAL_DEEPEST:
            path = root / stage / session_id / name
            path.parent.mkdir(parents=True, exist_ok=True)
            ST.write_bytes_atomic(path, lambda h: h.write(b"x"))
        for stage in ("surface", "appearance"):
            ST.write_json_atomic(root / stage / session_id / "manifest.json", {"x": 1})
        written.extend(p for p in Path(store_.root).rglob("*") if p.is_file())
        kwargs["record"](STAGE_SURFACE, state=STAGE_STATE_OK, detail=None)
        return {"surface": {"attempted": True, "state": "ok"}}

    report = AB.build_area(store, wid, sid, aid, record, final_surface_stages=stages)
    assert report["built"] is True
    placed = [p for p in Path(store.root).rglob("*") if p.is_file()]
    pid_pad = max(0, 10 - len(str(os.getpid())))      # a 10-digit pid in a staging name
    root_len = len(str(store.root))

    def longest(paths):
        return max(len(str(p)) - root_len + (pid_pad if ".p" in p.name and p.name.endswith(".tmp") else 0)
                   for p in paths)

    deepest_written = longest(written)
    deepest_placed = longest(placed)
    print("DEEPEST written", deepest_written, "placed", deepest_placed)
    assert any(".tmp" in p.name for p in written), "the staging names were measured"
    # what is WRITTEN stays well under 200 on the live root, and under 240 on the worst case
    assert LIVE_ROOT_CHARS + deepest_written < 200, deepest_written
    assert WORST_ROOT_CHARS + deepest_written < 240, deepest_written
    # what is left in place (read, never written again by the build) stays under 240 on the worst case
    assert WORST_ROOT_CHARS + deepest_placed < 240, deepest_placed
    assert all("areas" in str(p) or "worlds" in str(p) for p in placed if ".ab" not in str(p))
    assert not (Path(store.root) / AB.AREA_BUILD_DIRNAME).exists(), "the build directory is gone"


# ---------------------------------------------------------------------------
# the build directory holds imagery outside the world: owned, purged, never reused
# (lead, round 3 -- a privacy rule)
# ---------------------------------------------------------------------------


def _dead_pid():
    import subprocess

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    return dead.pid


def _killed_build(store, area_id=AREA1, world_id=None, pid=None):
    """What a build killed mid-way leaves: its marked build directory with keyframe imagery."""
    bdir = AB._fresh_build_dir(store, world_id or W1, S1, area_id)
    img = bdir / "dense" / S1 / "work" / "undist" / "00004.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\xff\xd8 keyframe pixels")
    chunk = bdir / "appearance" / S1 / ("c." + "f" * 32 + ".bin")
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(b"imagery")
    if pid is not None:
        owner = json.loads((bdir / AB.OWNER_FILENAME).read_text())
        owner["pid"] = pid
        (bdir / AB.OWNER_FILENAME).write_text(json.dumps(owner))
    return bdir


def test_purge_after_a_killed_area_build_leaves_no_imagery(tmp_path):
    store, kids = _store(tmp_path)
    _components(store, kids)
    bdir = _killed_build(store)
    owner = json.loads((bdir / AB.OWNER_FILENAME).read_text())
    assert owner["world_id"] == W1 and owner["area_id"] == AREA1
    # and one left WITHOUT a marker (killed before it was written) for an area of this world
    (C.area_root(store, W1, S1, AREA2)).mkdir(parents=True, exist_ok=True)
    unmarked = AB.area_build_dir(store, AREA2)
    (unmarked / "appearance").mkdir(parents=True)
    (unmarked / "appearance" / "c.bin").write_bytes(b"imagery")
    report = store.purge_world(W1)
    assert not report.retained
    assert not (Path(store.root) / AB.AREA_BUILD_DIRNAME).exists()
    assert not store.world_dir(W1).exists()
    assert any(str(bdir) in r for r in report.removed)


def test_purge_leaves_another_worlds_area_build_alone(tmp_path):
    store, kids = _store(tmp_path)
    other = _killed_build(store, area_id="b" * 16, world_id="w-other")
    store.purge_world(W1)
    assert other.exists() and (other / AB.OWNER_FILENAME).exists()


def test_a_rebuild_after_a_kill_starts_clean(tmp_path, fake_depth):
    store, kids = _store(tmp_path)
    record = _components(store, kids)
    stale = _killed_build(store, pid=_dead_pid())
    (stale / "surface" / S1).mkdir(parents=True)
    (stale / "surface" / S1 / "stale-from-the-killed-build.bin").write_bytes(b"x")
    report = AB.build_area(store, W1, S1, AREA1, record, final_surface_stages=_fake_stages([]))
    assert report["built"] is True
    final = C.area_root(store, W1, S1, AREA1)
    assert not list(final.rglob("stale-from-the-killed-build.bin"))
    assert not list(final.rglob("00004.jpg")), "nothing of the killed build is reused"
    assert not AB.area_build_dir(store, AREA1).exists()


def test_an_orphan_of_a_world_that_no_longer_exists_is_swept_never_reused(tmp_path, fake_depth):
    store, kids = _store(tmp_path)
    record = _components(store, kids)
    orphan = _killed_build(store, area_id="c" * 16, world_id="w-gone", pid=_dead_pid())
    AB.build_area(store, W1, S1, AREA1, record, final_surface_stages=_fake_stages([]))
    assert not orphan.exists()


def test_a_build_killed_while_moving_into_place_is_never_recorded_complete(tmp_path, monkeypatch,
                                                                          fake_depth):
    store, kids = _store(tmp_path)
    record = _components(store, kids)
    real = AB.publish_area_build

    def killed_half_way(store_, world_id, session_id, area_id):
        bdir = AB.area_build_dir(store_, area_id)
        final = C.area_root(store_, world_id, session_id, area_id)
        (bdir / "solve").replace(final / "solve")          # one stage in, then the process dies
        raise KeyboardInterrupt("killed")

    monkeypatch.setattr(AB, "publish_area_build", killed_half_way)
    with pytest.raises(KeyboardInterrupt):
        AB.build_area(store, W1, S1, AREA1, record, final_surface_stages=_fake_stages([]))
    rec = C.read_area_record(store, W1, S1, AREA1)
    assert rec["stages"]["surface"]["state"] == "running", "not recorded ok before it is in place"
    assert rec["stages"].get("appearance", {}).get("state") != "ok"
    monkeypatch.setattr(AB, "publish_area_build", real)
    # the next build replaces it whole, and only then says ok
    AB.build_area(store, W1, S1, AREA1, record, final_surface_stages=_fake_stages([]))
    rec = C.read_area_record(store, W1, S1, AREA1)
    assert rec["stages"]["surface"]["state"] == "ok"


def test_publishing_never_holds_two_builds_stages(tmp_path, fake_depth):
    store, kids = _store(tmp_path)
    record = _components(store, kids)
    AB.build_area(store, W1, S1, AREA1, record, final_surface_stages=_fake_stages([]))
    final = C.area_root(store, W1, S1, AREA1)
    (final / "appearance" / S1 / "from-the-first-build.bin").write_bytes(b"x")
    AB.build_area(store, W1, S1, AREA1, record, final_surface_stages=_fake_stages([]))
    assert not (final / "appearance" / S1 / "from-the-first-build.bin").exists()


def test_a_keyframe_set_switch_discards_a_dead_builds_imagery(tmp_path):
    from tower.world_builder import reredaction as RR

    store, kids = _store(tmp_path)
    _components(store, kids)
    dead = _killed_build(store, pid=_dead_pid())
    live = _killed_build(store, area_id=AREA2)                 # this process: alive
    RR._discard_area_builds(store, W1)
    assert not dead.exists() and live.exists()
