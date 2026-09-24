"""The coherence gate's metric scale (world_builder/coherence_scale.py): per camera log(z_sfm / z_metric).

A port of the run's harness TRI estimator (coherence_eval/eval_placement.triangulated_depth_ratios) onto the
solve database's verified pairs and the product depth stage's MoGe-2 predictions. These tests pin it on
synthetic scenes whose true answer is known: a solve at scale s against metric depth reads log(s).
"""

from __future__ import annotations

import math
import sqlite3

import numpy as np
import pytest

from tower.world_builder import coherence_scale as CS

CAM = {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240}
K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])


def _ry(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]])


def scene(n_cams=6, scale=1.0, seed=0, component=None, n_obs=100):
    """Cameras on a line looking down +z at a textured wall 3-5 m away (METRIC). The solve is the same scene
    at `scale` (z_sfm = scale * z_metric). Returns (cameras, pairs, depth maps by name)."""
    rng = np.random.default_rng(seed)
    X = np.c_[rng.uniform(-2.0, 2.0, 400), rng.uniform(-1.5, 1.5, 400), rng.uniform(3.0, 5.0, 400)]
    names, Rs, ts, depths = [], [], [], {}
    for i in range(n_cams):
        C = np.array([0.25 * i, 0.0, 0.0])
        R = _ry(2.0 * i)
        t = -R @ C
        names.append(f"{i:08d}.jpg")
        Rs.append(R)
        ts.append(t * scale)
        Xc = (R @ X.T).T + t
        uv = (K @ (Xc / Xc[:, 2:3]).T).T[:, :2]
        d = np.full((CAM["height"], CAM["width"]), np.nan, np.float32)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < CAM["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < CAM["height"])
        d[np.floor(uv[ok, 1]).astype(int), np.floor(uv[ok, 0]).astype(int)] = Xc[ok, 2]
        depths[names[-1]] = (d, uv, ok)
    pairs = []
    for i in range(n_cams):
        for j in range(i + 1, n_cams):
            ok = depths[names[i]][2] & depths[names[j]][2]
            pairs.append((names[i], names[j], depths[names[i]][1][ok], depths[names[j]][1][ok]))
    cams = CS.ScaleCameras(names=names, R_cw=np.stack(Rs), t_cw=np.stack(ts),
                           component=np.asarray(component if component is not None else [0] * n_cams),
                           n_obs=np.full(n_cams, n_obs))
    return cams, pairs, {k: v[0] for k, v in depths.items()}


def sampler(depths):
    return lambda name, uv: CS.sample_nearest(depths.get(name), uv)


@pytest.mark.parametrize("scale", [1.0, 0.37, 4.27])
def test_a_solve_at_scale_s_reads_log_s(scale):
    cams, pairs, depths = scene(scale=scale)
    out = CS.metric_scale(cams, pairs, CAM, sampler(depths))
    assert out["cameras_measured"] == len(cams.names)
    for v in out["metric_log"].values():
        assert v == pytest.approx(math.log(scale), abs=0.02)
    assert out["pairs_used"] > 0 and out["params_digest"] == CS.ScaleParams().digest()


def test_unpublished_cameras_and_cross_component_pairs_are_not_measured():
    cams, pairs, depths = scene(component=[0, 0, 0, 1, 1, 1])
    cams.n_obs[0] = 29  # under the 30-observation publish floor
    out = CS.metric_scale(cams, pairs, CAM, sampler(depths))
    assert cams.names[0] not in out["metric_log"]
    assert out["cameras_published"] == 5
    # within each component the level is still measured; no pair crosses components
    assert set(out["metric_log"]) == set(cams.names[1:])


def test_a_camera_with_too_few_valid_depth_samples_has_no_ratio():
    cams, pairs, depths = scene()
    depths[cams.names[2]] = np.full_like(depths[cams.names[2]], np.nan)   # the network masked everything
    depths[cams.names[3]] = np.full_like(depths[cams.names[3]], 80.0)     # beyond depth_valid_m
    out = CS.metric_scale(cams, pairs, CAM, sampler(depths))
    assert cams.names[2] not in out["metric_log"] and cams.names[3] not in out["metric_log"]
    assert cams.names[0] in out["metric_log"]


def test_no_depth_at_all_is_an_empty_metric_log_not_an_error():
    cams, pairs, _ = scene()
    out = CS.metric_scale(cams, pairs, CAM, lambda name, uv: np.full(len(uv), np.nan))
    assert out["metric_log"] == {} and out["cameras_measured"] == 0


def test_matches_the_harness_estimator_on_the_same_inputs():
    """The port computes what eval_placement.triangulated_depth_ratios computes (same gates, same medians)."""
    cams, pairs, depths = scene(scale=2.0, seed=3)
    rng = np.random.default_rng(1)
    noisy = [(a, b, ua + rng.normal(0, 0.8, ua.shape), ub + rng.normal(0, 0.8, ub.shape)) for a, b, ua, ub in pairs]
    out = CS.metric_scale(cams, noisy, CAM, sampler(depths))
    # the reference: the harness formula written out independently
    idx = cams.index()
    C = np.stack([-cams.R_cw[i].T @ cams.t_cw[i] for i in range(len(cams.names))])
    per = {n: ([], []) for n in cams.names}
    Kinv = np.linalg.inv(K)
    for a, b, ua, ub in noisy:
        i, j = idx[a], idx[b]
        x1 = (Kinv @ np.c_[ua, np.ones(len(ua))].T).T[:, :2]
        x2 = (Kinv @ np.c_[ub, np.ones(len(ub))].T).T[:, :2]
        R = cams.R_cw[j] @ cams.R_cw[i].T
        t = cams.R_cw[j] @ (C[i] - C[j])
        s = CS.sampson_px(R, t, x1, x2, K[0, 0])
        z1, z2, ang = CS.triangulate_midpoint(x1, x2, R, t)
        ok = (s <= 2.0) & (ang >= 2.0) & (ang <= 90.0) & (z1 > 0) & (z2 > 0)
        per[a][0].append(ua[ok]); per[a][1].append(z1[ok])
        per[b][0].append(ub[ok]); per[b][1].append(z2[ok])
    for n, (uvs, zs) in per.items():
        uv, z = np.concatenate(uvs), np.concatenate(zs)
        zm = CS.sample_nearest(depths[n], uv)
        ok = np.isfinite(zm) & (zm > 0.05) & (zm < 50.0)
        if ok.sum() < 10:
            assert n not in out["metric_log"]
            continue
        assert out["metric_log"][n] == pytest.approx(float(np.median(np.log(z[ok] / zm[ok]))), abs=1e-9)


def test_read_inlier_pairs_reads_verified_geometries_only(tmp_path):
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.execute("create table images (image_id integer, name text)")
    con.execute("create table keypoints (image_id integer, rows integer, cols integer, data blob)")
    con.execute("create table two_view_geometries (pair_id integer, rows integer, cols integer, data blob, "
                "config integer, F blob, E blob, H blob)")
    for i in (1, 2, 3):
        con.execute("insert into images values (?, ?)", (i, f"{i:08d}.jpg"))
        kp = np.c_[np.arange(40, dtype=np.float32) + i, np.arange(40, dtype=np.float32),
                   np.ones((40, 4), np.float32)]
        con.execute("insert into keypoints values (?, ?, ?, ?)", (i, 40, 6, kp.astype(np.float32).tobytes()))
    base = 2147483647
    m = np.c_[np.arange(20), np.arange(20) + 1].astype(np.uint32)
    con.execute("insert into two_view_geometries values (?, ?, 2, ?, 2, NULL, NULL, NULL)",
                (1 * base + 2, 20, m.tobytes()))            # CALIBRATED: read
    con.execute("insert into two_view_geometries values (?, ?, 2, ?, 1, NULL, NULL, NULL)",
                (1 * base + 3, 20, m.tobytes()))            # DEGENERATE: not verified
    con.execute("insert into two_view_geometries values (?, ?, 2, ?, 2, NULL, NULL, NULL)",
                (2 * base + 3, 3, m[:3].tobytes()))         # 3 inliers: under the harness's 5
    con.commit()
    con.close()
    pairs = CS.read_inlier_pairs(db)
    assert [(a, b) for a, b, _, _ in pairs] == [("00000001.jpg", "00000002.jpg")]
    _, _, ua, ub = pairs[0]
    assert ua[0].tolist() == [1.0, 0.0] and ub[0].tolist() == [3.0, 1.0]
    # opened immutable: no side files next to a published database
    assert sorted(p.name for p in tmp_path.iterdir()) == ["database.db"]


def test_depth_stage_sampler_masks_the_redaction_fill_and_misses_absent_frames(tmp_path):
    work = tmp_path / "work"
    (work / "depth").mkdir(parents=True)
    pred = np.full((10, 12), 2.0, np.float16)
    fill = np.zeros((10, 12), bool)
    fill[:, :6] = True
    np.save(work / "depth" / "00003_pred.npy", pred)
    np.save(work / "depth" / "00003_fill.npy", fill)
    s = CS.DepthStageSampler(work, {"a.jpg": 3, "b.jpg": 4})
    got = s("a.jpg", np.array([[1.5, 1.5], [8.2, 3.9], [20.0, 1.0]]))
    assert np.isnan(got[0]) and got[1] == pytest.approx(2.0) and np.isnan(got[2])
    assert np.isnan(s("b.jpg", np.array([[1.0, 1.0]]))).all()
    assert s.missing == {"b.jpg"}
