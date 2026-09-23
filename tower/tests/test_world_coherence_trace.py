"""Unit tests for the coherence trace's geometry helpers (no world on disk needed)."""

import numpy as np
import pandas as pd

from tower.world_builder.coherence_eval.trace import (
    _label_components,
    angle_between_deg,
    quat_wxyz_to_R,
    R_to_quat_wxyz,
    rigid_islands,
    roll_free_up,
    rotation_angle_deg,
    track_cut,
    umeyama,
)


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def test_umeyama_recovers_a_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    R = _rot([0.3, 1.0, 0.2], 37.0)
    dst = 2.5 * (R @ src.T).T + np.array([1.0, -2.0, 0.5])
    s, R_est, t = umeyama(src, dst)
    assert abs(s - 2.5) < 1e-9
    assert rotation_angle_deg(R_est.T @ R) < 1e-6
    assert np.allclose(t, [1.0, -2.0, 0.5])


def test_quaternion_round_trip():
    R = _rot([1, 2, 3], 123.0)
    assert rotation_angle_deg(quat_wxyz_to_R(R_to_quat_wxyz(R)).T @ R) < 1e-6


def test_roll_free_up_ignores_pitch_and_sees_tilt():
    # Cameras (OpenCV: y down) yawing around world up = -y, each pitched down 30 deg.
    Rs = []
    for yaw in np.linspace(0, 300, 12):
        Rs.append(_rot([0, 1, 0], yaw) @ _rot([1, 0, 0], -30.0))
    up = roll_free_up(Rs)
    assert up["well_conditioned"]
    assert angle_between_deg(up["up"], [0, -1, 0]) < 1e-6
    # The same rig tilted 50 deg as a rigid body: its up tilts with it.
    tilt = _rot([0, 0, 1], 50.0)
    up2 = roll_free_up([tilt @ R for R in Rs])
    assert abs(angle_between_deg(up2["up"], [0, -1, 0]) - 50.0) < 1e-6


def test_components_and_rigid_islands_split_on_zero_shared_points():
    # largest first, ties broken by the smallest member
    assert list(_label_components(5, [(0, 1), (3, 4)])) == [0, 0, 2, 1, 1]
    T = pd.DataFrame({"index": range(6), "g_supported": [True] * 6})
    cov = {"n": 6, "pair_shared_points": {(0, 1): 50, (1, 2): 40, (3, 4): 60, (4, 5): 5}}
    isl = rigid_islands(T, cov, thresholds=(1, 10))
    k1 = isl["rigid_island_k1"].tolist()
    assert k1[0] == k1[1] == k1[2] and k1[3] == k1[4] == k1[5] and k1[0] != k1[3]
    k10 = isl["rigid_island_k10"].tolist()
    assert k10[4] != k10[5]


def test_track_cut_counts_points_spanning_each_boundary():
    T = pd.DataFrame({"keyframe_id": [f"k{i}" for i in range(4)]})
    solution = {
        "keyframe_ids": [f"k{i}" for i in range(4)],
        "arrays": {
            # point 0 seen by 0,1 ; point 1 seen by 1,3 ; point 2 seen by 2,3
            "observations": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [3, 1, 1], [2, 2, 2], [3, 2, 2]]),
            "component": np.zeros(3, int),
        },
    }
    cut = track_cut(T, {"main_component": 0}, solution)
    # boundary before 0: none; before 1: point 0; before 2: point 1; before 3: points 1 and 2
    assert list(cut) == [0, 1, 1, 2]
