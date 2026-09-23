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


# --- added after the V2 harness review (M7, L12) ---------------------------

import sqlite3

import pytest

from tower.world_builder.coherence_eval.trace import (
    COLMAP_PAIR_BASE,
    StaleDatabaseError,
    compose_placement,
    decode_pair_id,
    is_verified,
    open_database_readonly,
    segment_table,
    wallclock_columns,
)


def test_pair_id_decodes_smaller_image_first():
    # COLMAP: pair_id = id1 * 2147483647 + id2 with id1 < id2
    for id1, id2 in [(1, 2), (7, 383), (2, 2147483646)]:
        assert decode_pair_id(id1 * COLMAP_PAIR_BASE + id2) == (id1, id2)
    # the first pair seen in the target database
    assert decode_pair_id(2147483649) == (1, 2)


def test_verified_rule_is_inliers_and_config():
    inl = [0, 14, 15, 40, 40, 40]
    cfg = [0, 2, 2, 1, 6, 7]
    assert list(is_verified(inl, cfg)) == [False, False, True, False, True, True]
    assert list(is_verified(inl, cfg, unverified_configs={0, 1, 7})) == [False, False, True, False, True, False]


def test_placement_composition_is_scale_rotate_translate():
    q = R_to_quat_wxyz(_rot([0, 0, 1], 90.0))
    placement = {"rotation_wxyz": q, "translation": [1.0, 2.0, 3.0], "scale": 2.0}
    x = compose_placement(placement, [1.0, 0.0, 0.0])
    assert np.allclose(x, [1.0, 4.0, 3.0])  # 2 * Rz90 @ x + t
    x2, R2 = compose_placement(placement, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(x2, [1.0, 2.0, 3.0])
    assert rotation_angle_deg(R2.T @ _rot([0, 0, 1], 90.0)) < 1e-9


def _synthetic_segment(scale=3.0, noise=0.0, seed=1):
    """One segment whose live chain is an exact similarity of the global poses."""
    rng = np.random.default_rng(seed)
    n = 8
    R0 = _rot([0.2, 1.0, 0.1], 25.0)
    t0 = np.array([0.5, -1.0, 2.0])
    Cg = rng.normal(size=(n, 3))
    Rg = [_rot(rng.normal(size=3), rng.uniform(5, 60)) for _ in range(n)]
    Cl = ((R0.T @ (Cg - t0).T).T) / scale + noise * rng.normal(size=(n, 3))
    Rl = [R0.T @ R for R in Rg]
    kid = [f"s:{i:08d}" for i in range(n)]
    T = pd.DataFrame({
        "index": range(n), "keyframe_id": kid, "segment_id": 0, "t_s": np.arange(n) * 0.3,
        "segment_cut_cause": "session_start", "segment_cut_detail": "", "breaks_chain": False,
        "live_Cx": Cl[:, 0], "live_Cy": Cl[:, 1], "live_Cz": Cl[:, 2],
        "g_Cx": Cg[:, 0], "g_Cy": Cg[:, 1], "g_Cz": Cg[:, 2],
        "g_supported": True, "g_component": 0.0,
    })
    q = np.array([R_to_quat_wxyz(R) for R in Rg])
    for j, a in enumerate("wxyz"):
        T[f"g_q{a}"] = q[:, j]
    ctx = {"manifest": {}, "_live_R_store": dict(zip(kid, Rl))}
    return T, ctx


def test_segment_sim3_recovers_scale_and_orientation_without_run_trace():
    T, ctx = _synthetic_segment(scale=3.0)
    S = segment_table(T, ctx)
    r = S.iloc[0]
    assert abs(r["sim3_scale"] - 3.0) < 1e-9
    assert r["sim3_rms_norm"] < 1e-9
    # M7: orientation columns exist when called directly, not only via run_trace
    assert r["orient_resid_med_deg"] < 1e-6
    assert r["position_vs_orientation_frame_deg"] < 1e-6
    assert abs(r["sim3_fixedR_scale"] - 3.0) < 1e-9


def test_segment_sim3_reports_position_disagreement():
    T, ctx = _synthetic_segment(scale=3.0, noise=0.2)
    r = segment_table(T, ctx).iloc[0]
    assert r["orient_resid_med_deg"] < 1e-6      # rotations still agree exactly
    assert r["sim3_fixedR_rms_norm"] > 0.02      # positions no longer do


def _wal_db(path):
    con = sqlite3.connect(path)
    con.execute("pragma journal_mode=wal")
    con.execute("pragma wal_autocheckpoint=0")
    con.execute("create table images (image_id integer primary key, name text)")
    con.execute("insert into images values (1, 'a.jpg')")
    con.commit()
    return con  # kept open: the WAL is not checkpointed away


def test_nonempty_wal_is_refused_not_silently_ignored(tmp_path):
    db = tmp_path / "database.db"
    writer = _wal_db(db)
    try:
        assert (tmp_path / "database.db-wal").stat().st_size > 0
        with pytest.raises(StaleDatabaseError):
            open_database_readonly(db)
        con = open_database_readonly(db, on_nonempty_wal="warn")
        con.close()
    finally:
        writer.close()


def test_readonly_open_creates_no_side_files(tmp_path):
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.execute("create table images (image_id integer primary key, name text)")
    con.execute("pragma journal_mode=wal")
    con.commit()
    con.execute("pragma wal_checkpoint(truncate)")
    con.close()
    for side in ("database.db-wal", "database.db-shm"):
        (tmp_path / side).unlink(missing_ok=True)
    ro = open_database_readonly(db)
    assert ro.execute("select count(*) from images").fetchone() == (0,)
    ro.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["database.db"]


def test_wallclock_is_labelled_utc_and_local_offset():
    utc, local = wallclock_columns([1790157708.805452, float("nan")])
    assert utc[0] == "2026-09-23T10:01:48.80Z"
    assert local[0][-5] in "+-" and local[0][:2].isdigit()
    assert utc[1] is None and local[1] is None
