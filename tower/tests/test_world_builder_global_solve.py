"""The global solver's merge: how a solution becomes the derived tree.

These tests need no pycolmap. They build a synthetic `Solution` by hand,
with known camera poses in a known world frame, and check that `merge()`
expresses it through the contract's layers exactly: segment-local poses
anchored at the first posed keyframe, points in the segment's own frame,
and a Sim3 placement into the component's reference segment that maps a
segment-frame coordinate back onto the world-frame coordinate it came from.
"""

import inspect

import numpy as np
import pytest

from tower.world_builder import global_solve
from tower.world_builder.global_solve import (
    DEGENERACY_UNREGISTERED,
    MIN_IMAGE_OBSERVATIONS,
    Solution,
    merge,
    quaternion_wxyz_to_rotation,
    rotation_to_quaternion_wxyz,
)
from tower.world_builder.records import Keyframe


def _keyframe(i: int, segment: int) -> Keyframe:
    return Keyframe(
        keyframe_id=f"s:{i:08d}",
        session_id="s",
        source_seq=i,
        received_at=float(i),
        image_relpath=f"images/{i:08d}.jpg",
        width=360,
        height=640,
        byte_count=1,
        segment_index=segment,
    )


def _rot(axis, deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    a = np.deg2rad(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def _cam_from_world(r_wc, centre):
    """COLMAP's convention from a camera-to-world rotation and a centre."""
    r_cw = r_wc.T
    t_cw = -r_cw @ np.asarray(centre, dtype=np.float64)
    return r_cw, t_cw


def _entry(r_wc, centre, component=0, observations=100):
    r_cw, t_cw = _cam_from_world(r_wc, centre)
    return {
        "component": component,
        "rotation": [float(v) for v in r_cw.reshape(-1)],
        "translation": [float(v) for v in t_cw],
        "observations": observations,
    }


@pytest.fixture
def scene():
    # Segment 0: keyframes 0,1,2 all posed. Segment 1: keyframes 3,4,5 --
    # 3 and 5 posed, 4 below the support floor. Segment 2: keyframes 6,7
    # in the horizon, none posed. Segment 3: keyframe 8 beyond the horizon.
    keyframes = [
        _keyframe(0, 0), _keyframe(1, 0), _keyframe(2, 0),
        _keyframe(3, 1), _keyframe(4, 1), _keyframe(5, 1),
        _keyframe(6, 2), _keyframe(7, 2),
        _keyframe(8, 3),
    ]
    world_poses = {
        0: (_rot([0, 1, 0], 10), [0.0, 0.0, 0.0]),
        1: (_rot([0, 1, 0], 20), [1.0, 0.0, 0.0]),
        2: (_rot([1, 0, 0], 5), [2.0, 0.5, 0.0]),
        3: (_rot([0, 0, 1], 30), [5.0, 1.0, -1.0]),
        4: (_rot([0, 0, 1], 35), [5.5, 1.0, -1.0]),
        5: (_rot([0, 1, 1], 40), [6.0, 2.0, -2.0]),
    }
    poses = {}
    for i, (r, c) in world_poses.items():
        poses[keyframes[i].keyframe_id] = _entry(r, c, observations=(5 if i == 4 else 100))
    xyz = np.array([
        [0.0, 0.0, 10.0], [1.0, 1.0, 9.0], [2.0, -1.0, 8.0],   # first seen by segment 0
        [7.0, 2.0, 3.0], [6.0, 1.0, 4.0],                       # first seen by segment 1
    ], dtype=np.float32)
    first = np.array([0, 1, 0, 3, 5], dtype=np.int32)
    observations = np.array([
        [0, 10, 0], [1, 11, 0], [1, 12, 1], [2, 13, 2], [0, 14, 2],
        [3, 20, 3], [5, 21, 3], [5, 22, 4], [4, 23, 4],   # keyframe 4 is unposed
        [2, 15, 3],                                        # cross-segment observation
    ], dtype=np.int32)
    solution = Solution(
        solver="glomap", solved_at=1.0, input_digest="d",
        keyframe_ids=[k.keyframe_id for k in keyframes[:8]],
        poses=poses,
        components=[{"index": 0, "images": 6, "points": 5}],
        xyz=xyz, rgb=np.full((5, 3), 128, dtype=np.uint8),
        component=np.zeros(5, dtype=np.int32), first_keyframe=first,
        track_length=np.array([2, 1, 2, 2, 2], dtype=np.int32),
        error=np.full(5, 0.5, dtype=np.float32), observations=observations,
    )
    local_pose_rows = [
        {"keyframe_id": k.keyframe_id, "segment_index": k.segment_index,
         "status": "anchor" if j == 0 else "unavailable", "degeneracy": "" if j == 0 else "no_correspondence",
         "rotation": [1.0, 0, 0, 0] if j == 0 else None, "translation": [0.0, 0, 0] if j == 0 else None}
        for seg in (0, 1, 2, 3)
        for j, k in enumerate([kf for kf in keyframes if kf.segment_index == seg])
    ]
    local_point_rows = [{"segment_index": 2, "xyz": [1.0, 2.0, 3.0]}, {"segment_index": 3, "xyz": [4.0, 5.0, 6.0]}]
    local_support = [[2, 0, 7, 0], [3, 0, 8, 0]]
    return keyframes, world_poses, solution, local_pose_rows, local_point_rows, local_support


def _placement(result, segment):
    return next(p for p in result.placements if p.segment_index == segment)


def test_every_keyframe_keeps_exactly_one_pose_row(scene):
    keyframes, _, solution, poses, points, support = scene
    result = merge(keyframes, poses, points, support, solution, input_digest="digest")
    assert [r["keyframe_id"] for r in result.pose_rows] == [k.keyframe_id for k in keyframes]


def test_a_replaced_segment_is_anchored_at_its_first_posed_keyframe(scene):
    keyframes, world, solution, poses, points, support, = scene
    result = merge(keyframes, poses, points, support, solution, input_digest="digest")
    rows = {r["keyframe_id"]: r for r in result.pose_rows}
    anchor = rows[keyframes[0].keyframe_id]
    assert anchor["status"] == "anchor"
    assert anchor["rotation"] == [1.0, 0.0, 0.0, 0.0]
    assert anchor["translation"] == [0.0, 0.0, 0.0]
    assert anchor["observations"] == 100
    # Keyframe 1 relative to keyframe 0: T_a^-1 T_k, T_world_camera convention.
    r_a, c_a = world[0]
    r_k, c_k = world[1]
    row = rows[keyframes[1].keyframe_id]
    assert row["status"] == "solved"
    np.testing.assert_allclose(quaternion_wxyz_to_rotation(row["rotation"]), r_a.T @ r_k, atol=1e-9)
    np.testing.assert_allclose(row["translation"], r_a.T @ (np.array(c_k) - np.array(c_a)), atol=1e-9)


def test_the_support_floor_refuses_a_keyframe_honestly(scene):
    keyframes, _, solution, poses, points, support = scene
    result = merge(keyframes, poses, points, support, solution, input_digest="digest")
    rows = {r["keyframe_id"]: r for r in result.pose_rows}
    weak = rows[keyframes[4].keyframe_id]
    assert weak["status"] == "unavailable"
    assert weak["degeneracy"] == DEGENERACY_UNREGISTERED
    assert weak["rotation"] is None and weak["translation"] is None
    assert weak["observations"] == 5
    assert MIN_IMAGE_OBSERVATIONS > 5


def test_placement_maps_segment_frame_back_onto_the_world_point(scene):
    """Segment 1's local point, pushed through its placement, must land on
    the same world-frame coordinate as segment 0's frame gives it."""
    keyframes, world, solution, poses, points, support = scene
    result = merge(keyframes, poses, points, support, solution, input_digest="digest")
    ref = _placement(result, 0)
    assert ref.state == "registered" and ref.reference_segment == 0 and ref.scale == 1.0
    assert tuple(ref.rotation_wxyz) == (1.0, 0.0, 0.0, 0.0)
    p1 = _placement(result, 1)
    assert p1.state == "registered" and p1.reference_segment == 0 and p1.scale == 1.0
    # world point 3 ([7,2,3]) is first seen by keyframe 3, so it is segment 1's.
    seg1_points = [r for r in result.point_rows if r["segment_index"] == 1]
    assert len(seg1_points) == 2
    local = np.array(seg1_points[0]["xyz"])
    r_ref, c_ref = world[0]
    expected_in_ref = r_ref.T @ (np.array([7.0, 2.0, 3.0]) - np.array(c_ref))
    composed = p1.scale * quaternion_wxyz_to_rotation(p1.rotation_wxyz) @ local + np.array(p1.translation)
    np.testing.assert_allclose(composed, expected_in_ref, atol=1e-5)
    assert seg1_points[0]["rgb"] == [128, 128, 128]


def test_a_segment_the_solve_did_not_pose_keeps_its_local_rows_and_is_refused(scene):
    keyframes, _, solution, poses, points, support = scene
    result = merge(keyframes, poses, points, support, solution, input_digest="digest")
    seg2 = [r for r in result.pose_rows if r["segment_index"] == 2]
    assert [r["status"] for r in seg2] == ["anchor", "unavailable"]
    assert [r for r in result.point_rows if r["segment_index"] == 2] == [{"segment_index": 2, "xyz": [1.0, 2.0, 3.0]}]
    assert [2, 0, 7, 0] in result.support_rows
    refused = _placement(result, 2)
    assert refused.state == "refused"
    assert "posed none" in refused.refusal_reason
    assert refused.input_digest == "digest"


def test_a_segment_beyond_the_horizon_is_left_unplaced(scene):
    keyframes, _, solution, poses, points, support = scene
    result = merge(keyframes, poses, points, support, solution, input_digest="digest")
    assert all(p.segment_index != 3 for p in result.placements)
    assert result.segments[3]["state"] == "pending"
    assert [r for r in result.point_rows if r["segment_index"] == 3] == [{"segment_index": 3, "xyz": [4.0, 5.0, 6.0]}]


def test_a_solved_segment_writes_no_support_rows_and_a_kept_segment_keeps_its_own(scene):
    """support.json indexes the chain's ORB keypoints; the solver's SIFT
    observations must not be written under that index."""
    keyframes, _, solution, poses, points, support = scene
    result = merge(keyframes, poses, points, support, solution, input_digest="digest")
    assert [r for r in result.support_rows if r[0] in (0, 1)] == []
    assert sorted(result.support_rows) == sorted([[2, 0, 7, 0], [3, 0, 8, 0]])


def test_reprojection_summary_is_exact_on_consistent_observations(scene):
    keyframes, world, solution, poses, points, support = scene
    from tower.world_builder.global_solve import reprojection_summary
    cam = {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 320.0, "width": 360, "height": 640}
    xy = []
    for kf_i, _, p in solution.observations:
        entry = solution.poses[solution.keyframe_ids[kf_i]]
        r = np.asarray(entry["rotation"]).reshape(3, 3)
        t = np.asarray(entry["translation"])
        c = r @ solution.xyz[p].astype(np.float64) + t
        xy.append([cam["fx"] * c[0] / c[2] + cam["cx"], cam["fy"] * c[1] / c[2] + cam["cy"]])
    solution.observation_xy = np.asarray(xy, dtype=np.float32)
    solution.camera = cam
    summary = reprojection_summary(solution)
    assert summary["overall"]["count"] == len(xy)
    assert summary["overall"]["max"] < 1e-2
    assert summary["behind_camera"] == 0
    assert set(summary["per_component"]) == {"0"}


def test_summary_counts_what_was_placed(scene):
    keyframes, _, solution, poses, points, support = scene
    result = merge(keyframes, poses, points, support, solution, input_digest="digest")
    s = result.summary
    assert s["segments_replaced"] == 2
    assert s["segments_refused"] == 1
    assert s["segments_pending"] == 1
    assert s["poses_solved"] == 3   # 1, 2 and 5 (0 and 3 are anchors, 4 refused)
    assert s["points"] == 5
    assert s["components"] == [{
        "component": 0, "reference_segment": 0, "segments": [0, 1], "keyframes": 5, "points": 5,
    }]
    assert result.segments[0]["coverage"] in ("partial", "confident")


def test_quaternion_round_trip():
    r = _rot([1, 2, 3], 73)
    np.testing.assert_allclose(quaternion_wxyz_to_rotation(rotation_to_quaternion_wxyz(r)), r, atol=1e-12)


def test_solver_availability_is_reported_not_raised(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "pycolmap":
            raise ImportError("no pycolmap here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    available, reason = global_solve.solver_available()
    assert available is False
    assert "pycolmap" in reason


class TestLoopDetectionDegradesRatherThanAborting:
    """A missing vocabulary tree must cost convergence, not the walk.

    Loop detection runs on every solve now, and it needs a 72 MB vocabulary
    tree COLMAP downloads on first use. COLMAP's failure to fetch that file
    is a glog CHECK, so the process dies of `abort()` -- exit code 3, no
    Python exception, nothing to catch, because `match_sequential` is
    outside every try/except in `solve()`.

    Measured end to end with an empty cache and no network, before the
    guard existed: every solve died, the manifest carried no `global_solve`
    at all, and the world shipped with every segment refused. That is the
    "87 disconnected fragments" outcome again, from a machine that merely
    has no network.
    """

    def test_the_cache_directory_is_the_one_colmap_uses(self):
        # COLMAP caches in the USER's home, not the venv -- which is why a
        # fresh checkout on a warm machine works and a fresh machine does
        # not, and why this is worth a pre-flight line of its own.
        from pathlib import Path

        assert global_solve.vocabulary_tree_cache_dir() == (
            Path.home() / ".cache" / "colmap"
        )

    def test_an_unreadable_cache_directory_reads_as_absent(self, monkeypatch, tmp_path):
        """Being wrong in this direction is survivable; the other is not."""
        missing = tmp_path / "nowhere"
        monkeypatch.setattr(global_solve, "vocabulary_tree_cache_dir", lambda: missing)
        assert global_solve.vocabulary_tree_cached() is False

    def test_a_cached_tree_is_found_by_the_name_colmap_gives_it(
        self, monkeypatch, tmp_path
    ):
        cache = tmp_path / "colmap"
        cache.mkdir()
        (cache / "96ca8ec8-vocab_tree_faiss_flickr100K_words256K.bin").write_bytes(b"x")
        monkeypatch.setattr(global_solve, "vocabulary_tree_cache_dir", lambda: cache)
        assert global_solve.vocabulary_tree_cached() is True

    def test_the_env_check_and_the_solver_ask_the_same_question(self):
        """A pre-flight that disagreed with the thing it checks is worse
        than no pre-flight."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts import world_builder_env_check as env_check

        source = inspect.getsource(env_check.collect_vocabulary_tree)
        assert "vocabulary_tree_cache_dir" in source
