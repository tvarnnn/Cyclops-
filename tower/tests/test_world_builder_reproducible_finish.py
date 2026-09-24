"""Two re-finishes of one walk with the same seed publish the same world (review V8 H2,
manager 019; part R4 of the H2 brief).

The whole final solve runs -- masks from the cache, the frozen matching, the mask filter,
the gate, the relabel, the components record and the depth hand-off -- with only the
external engines faked: pycolmap (the recording fake), the transient detector (a stub),
the depth network and the mapper (a fixed candidate: GLOMAP seeded on one thread is
bit-identical on one database, RUN P3-PM). What is pinned is that nothing in between adds
a difference: `solution.json` and `components.json` are equal field for field, except
exactly these, which a reader must exclude when comparing two finishes:

  EVERY RUN (timestamps, durations, per-solve paths):
    solution.json   solved_at; solve_identity (it hashes solved_at); timing.*;
                    gate.seconds, gate.gate_seconds, gate.depth.seconds,
                    gate.metric_scale.seconds; transients.seconds.*;
                    transients.filter.seconds; transients.database and solve.database
                    (a per-solve masked database, `database.masked.p<pid>.<nonce>.db`);
                    transients.filter.swept (names of earlier solves' databases)
    components.json solved_at; solve_identity
  THE FIRST RE-FINISH OF A WORLD AGAINST A LATER ONE (cold caches, the same result):
                    solve.matching and solve.matching_detail ("matched" -> "frozen");
                    gate.depth.predictions (made -> cached); and, only when masks had to
                    be computed, transients.cache_hits / computed / device / gpu_peak_mb
"""

from __future__ import annotations

import copy
import io
import json
import math
import sqlite3
import time
import types

import cv2
import numpy as np
import pytest

from tests.test_world_builder_solve_masks import (  # noqa: F401 -- fixtures and helpers
    HEIGHT,
    WIDTH,
    StubDetector,
    _B,
    _KEYPOINTS,
    _blob,
    colmap,
)
from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP
from tower.world_builder import global_solve as GS
from tower.world_builder import relocalizer
from tower.world_builder.records import Keyframe

N_A, N_B = 30, 20
N_KF = N_A + N_B

EVERY_RUN = (
    ("solution", "solved_at"), ("solution", "solve_identity"), ("solution", "timing"),
    ("solution", "gate", "seconds"), ("solution", "gate", "gate_seconds"),
    ("solution", "gate", "depth", "seconds"), ("solution", "gate", "metric_scale", "seconds"),
    ("solution", "transients", "seconds"), ("solution", "transients", "filter", "seconds"),
    ("solution", "transients", "database"), ("solution", "solve", "database"),
    ("solution", "transients", "filter", "swept"),
    ("components", "solved_at"), ("components", "solve_identity"),
)
COLD_AGAINST_WARM = EVERY_RUN + (
    ("solution", "solve", "matching"), ("solution", "solve", "matching_detail"),
    ("solution", "gate", "depth", "predictions"),
)
# Only when the first re-finish had to compute masks (this fixture's walk was unmasked).
MASKS_COMPUTED = (
    ("solution", "transients", "cache_hits"), ("solution", "transients", "computed"),
    ("solution", "transients", "device"), ("solution", "transients", "gpu_peak_mb"),
)


def _rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def walk(tmp_path, colmap, monkeypatch):
    """A stopped 50-keyframe walk whose solve left undistorted images and a walk database."""
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda session_dir: [])
    store = WorldStore(tmp_path / "worlds")
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(
        world_id, frame_source="synthetic",
        intrinsics=CameraIntrinsics(source="self_calibrated", model="pinhole_radtan",
                                    fx=50.0, fy=50.0, cx=WIDTH / 2, cy=HEIGHT / 2,
                                    dist_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
                                    calibrated_width=WIDTH, calibrated_height=HEIGHT))
    rng = np.random.default_rng(0)
    started = store.read_session(world_id, session_id).started_at
    for i in range(N_KF):
        ok, buf = cv2.imencode(".jpg", rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8))
        store.write_keyframe_image(world_id, session_id, f"{i:08d}.jpg", buf.tobytes())
        store.append_keyframe(world_id, Keyframe(
            keyframe_id=f"{session_id}:{i:08d}", session_id=session_id, source_seq=i,
            received_at=started + 0.5 * i, image_relpath=f"images/{i:08d}.jpg",
            width=WIDTH, height=HEIGHT, byte_count=1, segment_index=0))
    engine.stop_session("stopped")
    s = types.SimpleNamespace(store=store, world_id=world_id, session_id=session_id,
                              workspace=GS.workspace_for(store, world_id, session_id))
    GS.solve(store, world_id, session_id, final=False)          # the walk's own solve
    db = s.workspace.database_path
    db.unlink()
    con = sqlite3.connect(str(db))
    con.executescript(
        "create table images (image_id integer primary key, name text, camera_id integer);"
        "create table keypoints (image_id integer primary key, rows integer, cols integer, data blob);"
        "create table matches (pair_id integer primary key, rows integer, cols integer, data blob);"
        "create table two_view_geometries (pair_id integer primary key, rows integer, cols integer,"
        " data blob, config integer);")
    for i in range(N_KF):
        con.execute("insert into images values (?, ?, 1)", (i + 1, f"{i:08d}.jpg"))
        con.execute("insert into keypoints values (?, 4, 6, ?)", (i + 1, _KEYPOINTS.tobytes()))
        if i:
            con.execute("insert into matches values (?, ?, ?, ?)", (i * _B + i + 1, *_blob([[0, 0], [2, 2], [3, 3]])))
            con.execute("insert into two_view_geometries values (?, ?, ?, ?, 2)",
                        (i * _B + i + 1, *_blob([[0, 0], [2, 2], [3, 3]])))
    con.commit()
    con.close()
    colmap.log.clear()
    return s


def _candidate(keyframes, camera):
    """The mapper's candidate: two islands of one solver component (the room, cameras
    0..29, and a piece, 30..49), coupled by 60 shared points."""
    kids = [k.keyframe_id for k in keyframes]
    R = [_rz(3.0 * i) for i in range(N_KF)]
    obs, p = [], 0
    for island in (range(0, N_A), range(N_A, N_KF)):
        idx = list(island)
        for j in range(len(idx)):
            for _ in range(30):
                for c in idx[j:j + 3]:
                    obs.append([c, len(obs), p])
                p += 1
    for k in range(60):
        obs.append([k % N_A, len(obs), p])
        obs.append([N_A + k % N_B, len(obs), p])
        p += 1
    obs = np.asarray(obs, np.int32)
    counts = np.bincount(obs[:, 0], minlength=N_KF)
    poses = {kid: {"component": 0, "rotation": R[i].ravel().tolist(),
                   "translation": [0.1 * i, 0.0, 0.0], "observations": int(counts[i])}
             for i, kid in enumerate(kids)}
    first = np.zeros(p, np.int32)
    for row in obs[::-1]:
        first[row[2]] = row[0]
    rng = np.random.default_rng(1)
    return GS.Solution(
        solver="glomap", solved_at=time.time(), input_digest=None, keyframe_ids=kids, poses=poses,
        components=[{"index": 0, "images": N_KF, "points": p}],
        xyz=rng.normal(size=(p, 3)).astype(np.float32), rgb=np.zeros((p, 3), np.uint8),
        component=np.zeros(p, np.int32), first_keyframe=first, track_length=np.full(p, 3, np.int32),
        error=np.full(p, 0.5, np.float32), observations=obs,
        observation_xy=rng.uniform(0, 40, size=(len(obs), 2)).astype(np.float32),
        camera=camera.to_json_dict())


@pytest.fixture
def engines(monkeypatch):
    """The mapper, the gate's links and the depth network, faked -- each a pure function
    of its input, as the real ones are once seeded and cached."""
    def map_candidate(pycolmap, database_path, workspace, sparse_dir, keyframes, *, seed, threads,
                      input_digest, min_image_observations, camera):
        sol = _candidate(keyframes, camera)
        sol.input_digest = input_digest
        return sol

    monkeypatch.setattr(GS, "_map_candidate", map_candidate)
    name = "{:08d}.jpg".format
    links = {}
    for island in (range(0, N_A), range(N_A, N_KF)):
        idx = list(island)
        for j in range(len(idx)):
            for d in (1, 2):
                if j + d < len(idx):
                    links[(name(idx[j]), name(idx[j + d]))] = 100
    links[(name(29), name(30))] = 40
    rots = {(a, b): _rz(3.0 * int(b[:8])) @ _rz(3.0 * int(a[:8])).T for a, b in links}
    monkeypatch.setattr(CG, "read_verified_links", lambda db, min_inliers=15: dict(links))
    monkeypatch.setattr(CG, "read_link_rotations", lambda db, cam, min_inliers=15: dict(rots))
    made = set()

    def depth(store, world_id, session_id, solution, intrinsics, should_stop=None):
        posed = sorted(solution.poses)
        new = [k for k in posed if k not in made]
        made.update(posed)
        align = {"backend": "moge2-vitl", "known_fov": 42.0, "targets": len(posed), "records": [],
                 "image_origins": {"stored-redacted": len(posed)},
                 "prediction_cache": {"token": "t", "hits": len(posed) - len(new), "predicted": len(new)}}
        return align, store.world_dir(world_id) / "dense" / session_id / "work", CP._depth_params()

    monkeypatch.setattr(CP, "run_gate_depth", depth)
    monkeypatch.setattr(CP, "measure_metric_scale", lambda solution, name_of, db, work: {
        "metric_log": {name(i): 0.0 for i in range(N_KF)}, "cameras_published": N_KF,
        "cameras_measured": N_KF})


def _refinish(s, stamp):
    """The re-finish's own step 1 (`world_refinish.set_aside`), then its final solve with
    the product settings -- in this process, the solve itself rather than its child."""
    from scripts import world_refinish as wr

    wr.set_aside(s.store, s.world_id, s.session_id, stamp)
    GS.solve(s.store, s.world_id, s.session_id, final=True, masks=True, seed=0, gate=True,
             loop_detection=True, transient_backend_factory=StubDetector(),
             mask_device_probe=lambda: None, input_digest="walk-digest")
    ws = s.workspace
    arrays = np.load(io.BytesIO(ws.arrays_path.read_bytes()))
    return {"solution": json.loads(ws.solution_path.read_text(encoding="utf-8")),
            "components": json.loads((ws.root / CP.COMPONENTS_FILENAME).read_text(encoding="utf-8")),
            "arrays": {k: arrays[k].tobytes() for k in arrays.files}}


def _without(published, paths):
    out = copy.deepcopy({k: published[k] for k in ("solution", "components")})
    for path in paths:
        node = out
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return out


@pytest.fixture
def no_attempt_ledger(monkeypatch):
    from scripts import world_refinish as wr

    monkeypatch.setattr(wr, "_restart_attempts", lambda *a, **k: {})


def test_two_refinishes_with_one_seed_publish_the_same_world(walk, engines, colmap, no_attempt_ledger):
    first = _refinish(walk, "r1")        # a world from before the freeze: matched, then frozen
    second = _refinish(walk, "r2")
    third = _refinish(walk, "r3")
    assert first["solution"]["solve"]["matching"] == GS.MATCHING_MATCHED
    assert second["solution"]["solve"]["matching"] == third["solution"]["solve"]["matching"] \
        == GS.MATCHING_FROZEN
    # the gate did decide something: a room and a piece outside it
    assert [e["shown_as"] for e in second["components"]["components"]] == ["room", "area"]
    assert second["solution"]["transients"]["cache_hits"] == N_KF
    # warm against warm: equal but for the timestamps, durations and per-solve paths
    assert _without(second, EVERY_RUN) == _without(third, EVERY_RUN)
    assert second["arrays"] == third["arrays"]
    # the first re-finish of the world against a later one: the same world
    assert first["solution"]["transients"]["computed"] == N_KF, "this walk's masks were not cached"
    cold = COLD_AGAINST_WARM + MASKS_COMPUTED
    assert _without(first, cold) == _without(second, cold)
    assert first["arrays"] == second["arrays"]
    # and the exclusions are real differences, not dead entries
    assert first["solution"]["solve_identity"] != second["solution"]["solve_identity"]
    assert first["solution"]["gate"]["depth"]["predictions"] != \
        second["solution"]["gate"]["depth"]["predictions"]


def test_nothing_else_differs(walk, engines, colmap, no_attempt_ledger):
    """Every field NOT in the exclusion lists is compared: a new timestamp or path added to
    either file later fails here until it is listed (and reported)."""
    _refinish(walk, "a")                 # the world's first: it matches, then it is frozen
    b, c = _refinish(walk, "b"), _refinish(walk, "c")

    def leaves(doc, prefix=()):
        if isinstance(doc, dict):
            for k, v in doc.items():
                yield from leaves(v, prefix + (k,))
        else:
            yield prefix, doc

    lb = dict(leaves(_without(b, EVERY_RUN)))
    lc = dict(leaves(_without(c, EVERY_RUN)))
    assert lb.keys() == lc.keys()
    assert [k for k in lb if lb[k] != lc[k]] == []
    assert len(lb) > 100, "the comparison covers the whole record"
