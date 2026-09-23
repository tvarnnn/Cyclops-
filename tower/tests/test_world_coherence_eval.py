"""The coherence evaluation harness on synthetic scenes whose truth is known,
plus the adapters on synthetic world / COLMAP directories and a golden test on
the frozen target world (skipped where that evidence is not on disk).

Covered: a clean two-lap closed loop; an injected jump; a spike; two
components; a scale step; gauge invariance under a random Sim(3) per
component; revisit errors against a synthetic pair set; the fixed-denominator
coherence and compare's coverage gate (review V2 H1); the x1.5 / x2 / x2.2 level
steps (H2); harness triangulation for a poses-only variant (H3); world-defined
revisit buckets (H4); physical bounds across tracking losses and image evidence
(H5); image islands, "UNOBSERVED" placement and tilt (B1); `verify_pair` on
synthetic correspondences; the world and COLMAP adapters; world/session
mismatch refusal (M1); improper rotations refused (L1).
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from tower.world_builder.coherence_eval import metrics as M
from tower.world_builder.coherence_eval.eval_pairs import PAIR_PARAMS, PairSet, verify_pair, wahba_rotation
from tower.world_builder.coherence_eval.eval_variant import (
    POSED,
    PUBLISHED,
    Variant,
    VariantMismatch,
    load_any,
    load_variant,
    save_variant,
    variant_from_colmap,
    variant_from_world,
)

N_PER_LAP = 80
LAPS = 2
N = N_PER_LAP * LAPS
W, H = 360, 640
FX = FY = 400.0
CX, CY = 180.0, 320.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
CAM = {"model": "PINHOLE", "width": W, "height": H, "params": [FX, FY, CX, CY]}
WALL_R = 5.0


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_axis(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * Kx + (1 - math.cos(angle)) * Kx @ Kx


def truth_scene(seed=0):
    """Cameras on a circle (radius 2, then 2.3 on the second lap), level
    (OpenCV y down = world +y down), looking outward at a cylindrical wall of
    points (radius 5). Metric units: the depth truth is in the same units."""
    rng = np.random.default_rng(seed)
    poses = []
    for k in range(N):
        lap, j = divmod(k, N_PER_LAP)
        theta = 2 * math.pi * j / N_PER_LAP
        r = 2.0 + 0.3 * lap
        c = np.array([r * math.sin(theta), 0.1 * math.sin(3 * theta), r * math.cos(theta)])
        T = np.eye(4)
        T[:3, :3] = _rot_y(theta)
        T[:3, 3] = c
        poses.append(T)
    n_pts = 1500
    phi = rng.uniform(0, 2 * math.pi, n_pts)
    X = np.c_[WALL_R * np.sin(phi), rng.uniform(-1.5, 1.5, n_pts), WALL_R * np.cos(phi)]
    return poses, X


def _cam(T, X):
    return (T[:3, :3].T @ (X - T[:3, 3]).T).T


def observe(poses, X, rng, noise_px=0.2):
    ok_k, ok_p, uv, depth = [], [], [], []
    for k, T in enumerate(poses):
        Pc = _cam(T, X)
        z = Pc[:, 2]
        u = FX * Pc[:, 0] / np.where(z > 0, z, 1) + CX
        v = FY * Pc[:, 1] / np.where(z > 0, z, 1) + CY
        vis = (z > 0.2) & (u > 0) & (u < W) & (v > 0) & (v < H)
        idx = np.nonzero(vis)[0]
        ok_k.append(np.full(len(idx), k))
        ok_p.append(idx)
        uv.append(np.c_[u[idx], v[idx]] + rng.normal(0, noise_px, (len(idx), 2)))
        depth.append(z[idx])
    return (np.concatenate(ok_k), np.concatenate(ok_p), np.concatenate(uv), np.concatenate(depth))


def make_variant(poses, X, obs, components=None, status=None):
    ids = [f"s:{k:08d}" for k in range(len(poses))]
    v = Variant(name="synthetic", world_id="w", session_id="s")
    for k, T in enumerate(poses):
        if T is None:
            continue
        v.poses[ids[k]] = np.asarray(T, float)
        v.component[ids[k]] = components[k] if components else "0"
        v.status[ids[k]] = status[k] if status else PUBLISHED
        v.camera_of[ids[k]] = "default"
    v.cameras = {"default": CAM}
    if obs is not None:
        kf, pt, uv, _ = obs
        v.xyz = np.asarray(X, float)
        v.obs_keyframe_ids = ids
        v.obs_keyframe = kf.astype(np.int64)
        v.obs_point = pt.astype(np.int64)
        v.obs_uv = uv
    return v, ids


def truth_depth_fn(obs):
    kf, pt, uv, z = obs
    table = {}
    for k, (u, vv), zz in zip(kf, uv, z):
        table[(int(k), float(u), float(vv))] = float(zz)

    def fn(i, q, cam):
        return np.array([table.get((int(i), float(a), float(b)), np.nan) for a, b in np.asarray(q)])

    return fn


def wall_depth_sample(poses):
    """Exact metric depth at canonical pixels: ray cast against the wall."""

    def fn(i, uv):
        T = poses[i]
        uv = np.asarray(uv, float).reshape(-1, 2)
        d = np.c_[(uv[:, 0] - CX) / FX, (uv[:, 1] - CY) / FY, np.ones(len(uv))]
        dw = d @ T[:3, :3].T
        c = T[:3, 3]
        a = dw[:, 0] ** 2 + dw[:, 2] ** 2
        b = 2 * (c[0] * dw[:, 0] + c[2] * dw[:, 2])
        cc = c[0] ** 2 + c[2] ** 2 - WALL_R ** 2
        disc = b * b - 4 * a * cc
        t = (-b + np.sqrt(np.maximum(disc, 0))) / (2 * a)
        return np.where(disc > 0, t, np.nan)  # t along a ray with z = 1 is the depth

    return fn


def truth_pairs(poses, X=None, gaps=(1,), distant=True, exclude=None):
    """Exact relative poses; with X, up to 64 exact inlier correspondences."""
    I, J = [], []
    for g in gaps:
        for i in range(N - g):
            I.append(i)
            J.append(i + g)
    if distant:
        for i in range(N_PER_LAP):
            I.append(i)
            J.append(i + N_PER_LAP)
    if exclude is not None:
        keep = [not exclude(i, j) for i, j in zip(I, J)]
        I = [i for i, k in zip(I, keep) if k]
        J = [j for j, k in zip(J, keep) if k]
    Rs, ts, xi, xj, offs = [], [], [], [], [0]
    for i, j in zip(I, J):
        Ri, Rj = poses[i][:3, :3].T, poses[j][:3, :3].T
        Rs.append(Rj @ Ri.T)
        t = Rj @ (poses[i][:3, 3] - poses[j][:3, 3])
        ts.append(t / np.linalg.norm(t))
        if X is not None:
            Pi, Pj = _cam(poses[i], X), _cam(poses[j], X)
            both = np.nonzero((Pi[:, 2] > 0.2) & (Pj[:, 2] > 0.2)
                              & (np.abs(Pi[:, 0] / np.maximum(Pi[:, 2], 1e-9)) < 0.4)
                              & (np.abs(Pj[:, 0] / np.maximum(Pj[:, 2], 1e-9)) < 0.4))[0][:64]
            xi.append(Pi[both, :2] / Pi[both, 2:])
            xj.append(Pj[both, :2] / Pj[both, 2:])
            offs.append(offs[-1] + len(both))
    P = len(I)
    arrays = {"i": np.array(I, np.int32), "j": np.array(J, np.int32), "R": np.array(Rs), "t": np.array(ts),
              "t_reliable": np.ones(P, bool), "parallax_deg": np.full(P, 10.0, np.float32),
              "n_inliers": np.full(P, 100, np.int32), "n_matches": np.full(P, 120, np.int32)}
    if X is not None:
        arrays.update({"inlier_offsets": np.array(offs, np.int64),
                       "inlier_xy_i": np.concatenate(xi).astype(np.float32),
                       "inlier_xy_j": np.concatenate(xj).astype(np.float32)})
    return PairSet.from_arrays(arrays)


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(1)
    poses, X = truth_scene()
    obs = observe(poses, X, rng)
    return poses, X, obs


def run(v, ids, pairs=None, depth_fn=None, segment_of=None, regions=None, times=None, depth_sample=None):
    return M.evaluate(v, ids, segment_of=segment_of, times=times, pairs=pairs, depth_fn=depth_fn,
                      depth_sample=depth_sample, K=K if depth_sample is not None else None,
                      regions=regions, focal_px=FX)


def scaled_block(poses, start, s):
    pivot = poses[start - 1][:3, 3]
    out = []
    for k, T in enumerate(poses):
        T2 = T.copy()
        if k >= start:
            T2[:3, 3] = pivot + s * (T[:3, 3] - pivot)
        out.append(T2)
    return out, pivot


# ---------------------------------------------------------------------------
# the basics


def test_clean_closed_loop(scene):
    poses, X, obs = scene
    v, ids = make_variant(poses, X, obs)
    out = run(v, ids, pairs=truth_pairs(poses, X), depth_fn=truth_depth_fn(obs),
              depth_sample=wall_depth_sample(poses))
    reg = out["registration"]
    assert reg["published_fraction"] == 1.0 and reg["in_largest_fraction"] == 1.0
    assert out["components"]["count"] == 1 and out["components"]["main_runs"] == 1
    main = out["continuity"]["main"]
    assert main["jumps"] == 0 and main["spikes"] == 0
    assert main["max_ratio"] < 3.0  # the lap change (radius 2.0 -> 2.3) is the largest step
    assert out["sanity"]["main"]["outlier_cameras"] == 0
    rp = out["reprojection"]["overall"]
    assert rp["median"] < 0.5 and out["reprojection"]["behind_camera"] == 0
    for key in ("depth", "depth_tri"):
        d = out["scale"][key]
        assert d["main"]["spread_mad"] < 1e-3 and d["main"]["steps"] == 0
        assert d["main"]["trend_factor_over_span"] < 1.01
        assert d["fixed_denominator"]["bands"]["x1.25"]["coherent_all"] == pytest.approx(
            d["ratio_coverage"])
    # metric truth: units per metre = 1
    assert out["scale"]["depth"]["main"]["units_per_metre_informative"] == pytest.approx(1.0, rel=1e-3)
    assert out["scale"]["depth_tri"]["main"]["units_per_metre_informative"] == pytest.approx(1.0, rel=1e-3)
    r = out["revisits"]["adjacent"]
    assert r["joined_fraction"] == 1.0 and r["rot_err_deg"]["max"] < 1e-6 and r["t_err_deg"]["max"] < 1e-4
    rv = out["revisits"]["revisit"]
    assert rv["pairs"] == N_PER_LAP and rv["rot_err_deg"]["max"] < 1e-6


def test_injected_jump_is_found_at_its_index(scene):
    poses, X, obs = scene
    shifted = [T.copy() for T in poses]
    step = np.median([np.linalg.norm(poses[k + 1][:3, 3] - poses[k][:3, 3]) for k in range(N - 1)])
    for k in range(60, N):
        shifted[k][:3, 3] += np.array([12 * step, 0, 0])
    v, ids = make_variant(shifted, X, None)
    main = run(v, ids)["continuity"]["main"]
    assert main["translation_jumps"] == 1
    assert main["spikes"] == 0
    (jump,) = [j for j in main["jump_list"] if j["counted"]]
    assert (jump["from"], jump["to"]) == (59, 60)
    assert jump["ratio"] > M.PARAMS["jump_ratio"]


def test_single_displaced_keyframe_is_a_spike(scene):
    poses, X, obs = scene
    bad = [T.copy() for T in poses]
    bad[100][:3, 3] += np.array([0, 3.0, 0])
    v, ids = make_variant(bad, X, None)
    main = run(v, ids)["continuity"]["main"]
    assert main["spike_keyframes"] == [100]
    assert main["translation_jumps"] == 2


def test_two_components(scene):
    poses, X, obs = scene
    comps = ["0" if k < 100 else "1" for k in range(N)]
    R = _rot_axis([1, 2, 3], 0.7)
    moved = []
    for k, T in enumerate(poses):
        if k >= 100:
            T2 = np.eye(4)
            T2[:3, :3] = R @ T[:3, :3]
            T2[:3, 3] = 3.0 * (R @ T[:3, 3]) + np.array([10, -4, 2])
            moved.append(T2)
        else:
            moved.append(T)
    v, ids = make_variant(moved, X, None, components=comps)
    out = run(v, ids, pairs=truth_pairs(poses))
    assert out["components"]["count"] == 2
    assert out["components"]["sizes"] == [100, 60]
    assert out["registration"]["in_largest_fraction"] == pytest.approx(100 / N)
    assert out["continuity"]["all_components"]["jumps"] == 0
    rv = out["revisits"]["revisit"]
    assert rv["split_across_components"] == 60
    assert rv["joined_fraction"] == pytest.approx(20 / 80)
    assert rv["rot_err_deg"]["max"] < 1e-6
    assert out["revisits"]["adjacent"]["split_across_components"] == 1


def test_scale_step_between_halves(scene):
    poses, X, obs = scene
    kf, pt, uv, z = obs
    half = N // 2
    s = 2.0
    scaled, pivot = scaled_block(poses, half, s)
    X2 = np.concatenate([X, pivot + s * (X - pivot)])
    pt2 = np.where(kf >= half, pt + len(X), pt)
    v, ids = make_variant(scaled, X2, (kf, pt2, uv, z))
    segment_of = [0 if k < half else 1 for k in range(N)]
    out = run(v, ids, depth_fn=truth_depth_fn(obs), segment_of=segment_of)
    assert out["reprojection"]["overall"]["median"] < 0.5
    d = out["scale"]["depth"]["main"]
    assert d["steps"] == 1
    (step,) = d["step_list"]
    assert abs(step["at_index"] - half) <= 1
    assert step["factor"] == pytest.approx(s, rel=1e-3)
    assert d["trend_slope_per_100kf"] > 0
    seg = out["scale"]["depth"]["by_tracker_segment"]
    assert seg["max_over_min_factor"] == pytest.approx(s, rel=1e-3)
    assert out["continuity"]["main"]["jumps"] == 0


# ---------------------------------------------------------------------------
# gauge


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


GAUGE_DEPENDENT = ("units_per_metre", "d3_pooled_informative", "depth_log_ratio", "_per_keyframe")


def test_gauge_invariance_under_random_sim3_per_component(scene):
    poses, X, obs = scene
    comps = ["0" if k < 110 else "1" for k in range(N)]
    kf, pt, uv, z = obs
    pt_c = np.where(kf >= 110, pt + len(X), pt)
    X_c = np.concatenate([X, X])
    point_comp = np.array(["0"] * len(X) + ["1"] * len(X))
    v, ids = make_variant(poses, X_c, (kf, pt_c, uv, z), components=comps)
    v.point_component = point_comp
    pairs = truth_pairs(poses, X, gaps=(1, 2))
    seg = [k // 40 for k in range(N)]
    times = np.arange(N) * 0.4
    kw = dict(pairs=pairs, depth_fn=truth_depth_fn(obs), depth_sample=wall_depth_sample(poses),
              segment_of=seg, times=times)
    base = run(v, ids, **kw)

    rng = np.random.default_rng(7)
    moved_poses, moved_X = [], X_c.copy()
    sims = {}
    for c in ("0", "1"):
        R = _rot_axis(rng.normal(size=3), rng.uniform(0, math.pi))
        sims[c] = (R, rng.uniform(0.01, 100.0), rng.normal(size=3) * 50)
    for k, T in enumerate(poses):
        R, s, t = sims[comps[k]]
        T2 = np.eye(4)
        T2[:3, :3] = R @ T[:3, :3]
        T2[:3, 3] = s * (R @ T[:3, 3]) + t
        moved_poses.append(T2)
    for c, sl in (("0", slice(0, len(X))), ("1", slice(len(X), None))):
        R, s, t = sims[c]
        moved_X[sl] = s * (R @ X_c[sl].T).T + t
    v2, _ = make_variant(moved_poses, moved_X, (kf, pt_c, uv, z), components=comps)
    v2.point_component = point_comp
    moved = run(v2, ids, **kw)  # the depth truth belongs to the IMAGES, not the gauge

    a, b = _flatten(M.round_sig(base)), _flatten(M.round_sig(moved))
    assert set(a) == set(b)
    diffs = []
    for key in a:
        if any(g in key for g in GAUGE_DEPENDENT):
            continue
        x, y = a[key], b[key]
        if isinstance(x, float) and isinstance(y, float):
            if not math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-6):
                diffs.append((key, x, y))
        elif x != y:
            diffs.append((key, x, y))
    assert not diffs, diffs[:10]


# ---------------------------------------------------------------------------
# revisits


def test_revisit_errors_on_a_synthetic_pair_set(scene):
    poses, X, obs = scene
    Rz = _rot_axis([0, 1, 0], math.radians(5.0))
    moved = []
    for k, T in enumerate(poses):
        T2 = T.copy()
        if k >= N_PER_LAP:
            T2[:3, :3] = Rz @ T[:3, :3]
            T2[:3, 3] = Rz @ T[:3, 3]
        moved.append(T2)
    v, ids = make_variant(moved, X, None)
    out = run(v, ids, pairs=truth_pairs(poses))
    rv = out["revisits"]["revisit"]
    assert rv["evaluated"] == N_PER_LAP
    assert rv["rot_err_deg"]["median"] == pytest.approx(5.0, abs=1e-6)
    assert rv["rot_err_deg"]["over_2"] == 1.0
    adj = out["revisits"]["adjacent"]
    assert adj["rot_err_deg"]["max"] == pytest.approx(5.0, abs=1e-6)
    assert adj["rot_err_deg"]["median"] < 1e-6
    assert rv["t_err_deg"]["median"] > 0.5


def test_revisit_buckets_need_a_new_segment_and_time():
    # H4: a long index gap inside one tracker segment, or too soon, is lingering
    I = np.array([0, 0, 0, 0, 5])
    J = np.array([1, 10, 40, 40, 45])
    seg = [0] * 30 + [1] * 30
    t = np.arange(60) * 0.5          # 40 vs 0: 20 s apart
    b = M.pair_buckets(I, J, seg, t)
    assert list(b[:3]) == ["adjacent", "near", "revisit"]
    t2 = np.arange(60) * 0.1         # 40 vs 0: 4 s apart
    assert M.pair_buckets(I, J, seg, t2)[2] == "lingering"
    same = [0] * 60
    assert M.pair_buckets(I, J, same, t)[2] == "lingering"


def test_revisit_rotation_convention():
    # the stored pair convention is x_j = R x_i + t
    R_wc_i = _rot_axis([0, 1, 0], 0.3)
    R_wc_j = _rot_axis([1, 0, 0], -0.2) @ R_wc_i
    Ci, Cj = np.zeros(3), np.array([1, 0, 0.5])
    Rj = R_wc_j.T
    R_ji = Rj @ R_wc_i
    t_ji = Rj @ (Ci - Cj)
    Xw = np.array([0.3, -0.2, 4.0])
    assert np.allclose(R_ji @ (R_wc_i.T @ (Xw - Ci)) + t_ji, Rj @ (Xw - Cj))


def test_verify_pair_recovers_the_synthetic_two_view_geometry():
    rng = np.random.default_rng(11)
    Xw = np.c_[rng.uniform(-2, 2, 400), rng.uniform(-3, 3, 400), rng.uniform(4, 8, 400)]
    Ri = np.eye(3)
    Rj = _rot_axis([0.1, 1, 0.05], 0.15)
    Ci, Cj = np.zeros(3), np.array([0.6, 0.05, 0.1])
    Pi = (Ri.T @ (Xw - Ci).T).T
    Pj = (Rj.T @ (Xw - Cj).T).T
    ui = (K @ (Pi / Pi[:, 2:]).T).T[:, :2]
    uj = (K @ (Pj / Pj[:, 2:]).T).T[:, :2]
    ok = (ui[:, 0] > 0) & (ui[:, 0] < W) & (ui[:, 1] > 0) & (ui[:, 1] < H) & \
         (uj[:, 0] > 0) & (uj[:, 0] < W) & (uj[:, 1] > 0) & (uj[:, 1] < H)
    ui, uj = ui[ok], uj[ok]
    desc = np.abs(rng.normal(size=(len(ui), 128))).astype(np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    r = verify_pair(ui.astype(np.float32), desc, uj.astype(np.float32), desc, K, (W, H), PAIR_PARAMS, distant=False)
    assert r is not None and r["n_inliers"] >= 0.9 * len(ui)
    R_true = Rj.T @ Ri                     # x_j = R x_i + t
    t_true = Rj.T @ (Ci - Cj)
    assert M.rot_angle_deg(r["R"].T @ R_true) < 0.05
    assert M.vec_angle_deg(r["t"], t_true / np.linalg.norm(t_true))[0] < 0.5
    assert r["t_reliable"]


def test_wahba_rotation_recovers_a_pure_rotation():
    rng = np.random.default_rng(3)
    f1 = rng.normal(size=(50, 3))
    f1 /= np.linalg.norm(f1, axis=1, keepdims=True)
    R = _rot_axis([0.2, 1, 0.1], 0.4)
    assert np.allclose(wahba_rotation(f1, f1 @ R.T), R)


# ---------------------------------------------------------------------------
# H1: coverage cannot be gamed


def test_fixed_denominator_cannot_be_gamed_by_unpublishing(scene):
    poses, X, obs = scene
    kf, pt, uv, z = obs
    half = N // 2
    scaled, pivot = scaled_block(poses, half, 3.0)
    X2 = np.concatenate([X, pivot + 3.0 * (X - pivot)])
    pt2 = np.where(kf >= half, pt + len(X), pt)
    ob = (kf, pt2, uv, z)
    v, ids = make_variant(scaled, X2, ob)
    base = run(v, ids, depth_fn=truth_depth_fn(obs))
    status = [PUBLISHED if k < half else POSED for k in range(N)]
    g, _ = make_variant(scaled, X2, ob, status=status)
    gamed = run(g, ids, depth_fn=truth_depth_fn(obs))
    # the per-published number "improves"; the fixed-denominator one cannot
    assert gamed["scale"]["depth"]["main"]["levels"]["bands"]["x1.5"]["coherent_fraction"] > \
        base["scale"]["depth"]["main"]["levels"]["bands"]["x1.5"]["coherent_fraction"]
    fb = base["scale"]["depth"]["fixed_denominator"]["bands"]["x1.5"]
    fg = gamed["scale"]["depth"]["fixed_denominator"]["bands"]["x1.5"]
    assert fg["coherent_all"] <= fb["coherent_all"] + 1e-9
    assert fg["longest_bad_run"] >= fb["longest_bad_run"]

    def res(body, name):
        return {"harness": {"code_digest": "x"}, "variant": {"name": name},
                "world": {"world_id": "w", "caches": {}}, "metrics": body}

    text = M.compare_markdown(res(base, "a"), res(gamed, "b"))
    assert "COVERAGE CHANGED" in text and "no verdict" in text


def test_compare_counts_a_missing_metric_as_a_loss(scene):
    poses, X, obs = scene
    v, ids = make_variant(poses, X, obs)
    with_pts = run(v, ids, depth_fn=truth_depth_fn(obs))
    p, _ = make_variant(poses, X, None)
    poses_only = run(p, ids, depth_fn=truth_depth_fn(obs))

    def res(body, name):
        return {"harness": {"code_digest": "x"}, "variant": {"name": name},
                "world": {"world_id": "w", "caches": {}}, "metrics": body}

    text = M.compare_markdown(res(with_pts, "a"), res(poses_only, "b"))
    assert "COVERAGE CHANGED" not in text
    assert "| reprojection median (px) |" in text and "A (n/a loses)" in text


# ---------------------------------------------------------------------------
# H2: level bands


@pytest.mark.parametrize("step", [1.5, 2.0, 2.2])
def test_level_steps_below_x2_25_are_detected(step):
    y = np.r_[np.zeros(265), np.full(118, math.log(step))]
    lv = M.level_stats(y)
    assert lv["bands"]["x1.25"]["coherent_fraction"] == pytest.approx(265 / 383)
    assert lv["bands"]["x1.25"]["longest_off_level_run"] == 118
    if step >= 2.0:
        assert lv["bands"]["x1.5"]["coherent_fraction"] == pytest.approx(265 / 383)
    assert lv["max_step_15kf_factor"] == pytest.approx(step)
    lv2 = M.level_stats(y + 5.0)  # gauge: a constant log offset
    assert lv2["bands"]["x1.25"]["coherent_fraction"] == lv["bands"]["x1.25"]["coherent_fraction"]


def test_level_stats_last_split_counts():
    # the 15-kf step must be seen even when it sits at the last possible split
    y = np.r_[np.zeros(15), np.full(15, math.log(3.0))]
    assert M.level_stats(y)["max_step_15kf_factor"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# H3: harness triangulation for a variant without points


def test_triangulated_scale_for_a_poses_only_variant(scene):
    poses, X, obs = scene
    pairs = truth_pairs(poses, X, gaps=(1, 2), distant=False)
    half = N // 2
    scaled, _ = scaled_block(poses, half, 2.0)
    v, ids = make_variant(scaled, X, None)  # NO points
    out = run(v, ids, pairs=pairs, depth_sample=wall_depth_sample(poses), segment_of=[k // 40 for k in range(N)])
    assert out["scale"]["depth"]["available"] is False
    tri = out["scale"]["depth_tri"]
    assert tri["available"] and tri["ratio_coverage"] > 0.95
    assert tri["main"]["max_step_factor"] == pytest.approx(2.0, rel=0.02)
    assert tri["fixed_denominator"]["bands"]["x1.25"]["coherent_all"] < 0.6
    clean, _ = make_variant(poses, X, None)
    ok = run(clean, ids, pairs=pairs, depth_sample=wall_depth_sample(poses))["scale"]["depth_tri"]
    assert ok["main"]["spread_mad"] < 1e-3
    assert ok["fixed_denominator"]["bands"]["x1.25"]["coherent_all"] > 0.95


def test_triangulation_ignores_inliers_the_variant_disagrees_with(scene):
    poses, X, obs = scene
    pairs = truth_pairs(poses, X, gaps=(1,), distant=False)
    bad = [T.copy() for T in poses]
    for k in range(0, N, 2):  # every other camera rotated 3 degrees: epipolar violations
        bad[k][:3, :3] = _rot_axis([1, 0, 0], math.radians(3.0)) @ bad[k][:3, :3]
    v, ids = make_variant(bad, X, None)
    tri = run(v, ids, pairs=pairs, depth_sample=wall_depth_sample(poses))["scale"]["depth_tri"]
    assert tri["inliers_gated"] > 0.5 * tri["inliers_total"]


# ---------------------------------------------------------------------------
# H5: physical bounds across tracking losses


def test_physical_bound_flags_an_impossible_move_across_a_tracking_loss(scene):
    poses, X, obs = scene
    kf, pt, uv, z = obs
    moved = [T.copy() for T in poses]
    shift = np.array([3.0, 0, 0])  # metres (the depth truth is metric)
    for k in range(60, N):
        moved[k][:3, 3] += shift
    X2 = np.concatenate([X, X + shift])
    pt2 = np.where(kf >= 60, pt + len(X), pt)
    v, ids = make_variant(moved, X2, (kf, pt2, uv, z))
    seg = [0 if k < 60 else 1 for k in range(N)]
    no_link = truth_pairs(poses, X, gaps=(1, 2), distant=False,
                          exclude=lambda i, j: i < 60 <= j)
    times = np.arange(N) * 0.3
    times[60:] += 0.7  # a 1 s tracking loss
    fast = run(v, ids, pairs=no_link, depth_fn=truth_depth_fn(obs), segment_of=seg, times=times)
    m = fast["continuity"]["main"]
    assert m["physically_implausible"] == 1 and m["jumps"] >= 1
    assert m["tracking_loss_steps"] == 1 and m["tracking_loss_steps_without_image_evidence"] == 1
    j = [x for x in m["jump_list"] if x["from"] == 59][0]
    assert j["step_m"] == pytest.approx(3.0, rel=0.05) and j["physically_implausible"]
    # the same move after a 5 s loss is walkable: not implausible
    slow_t = np.arange(N) * 0.3
    slow_t[60:] += 5.0
    slow = run(v, ids, pairs=no_link, depth_fn=truth_depth_fn(obs), segment_of=seg, times=slow_t)
    assert slow["continuity"]["main"]["physically_implausible"] == 0
    # with image evidence across the loss the step is no longer "without evidence"
    linked = run(v, ids, pairs=truth_pairs(poses, X, gaps=(1, 2), distant=False), depth_fn=truth_depth_fn(obs),
                 segment_of=seg, times=slow_t)
    assert linked["continuity"]["main"]["tracking_loss_steps_without_image_evidence"] == 0


def test_time_aware_jumps_separate_a_long_move_from_a_pose_error(scene):
    poses, X, obs = scene
    step = np.median([np.linalg.norm(poses[k + 1][:3, 3] - poses[k][:3, 3]) for k in range(N - 1)])
    moved = [T.copy() for T in poses]
    for k in range(60, N):
        moved[k][:3, 3] += np.array([12 * step, 0, 0])
    times = np.arange(N) * 0.3
    v, ids = make_variant(moved, X, None)
    fast = run(v, ids, times=times)["continuity"]["main"]
    assert fast["jumps"] == 1 and fast["index_jumps"] == 1
    slow_t = times.copy()
    slow_t[60:] += 5.0
    slow = run(v, ids, times=slow_t)["continuity"]["main"]
    assert slow["index_jumps"] == 1 and slow["jumps"] == 0


# ---------------------------------------------------------------------------
# B1: islands, placement, tilt


def test_islands_report_unobserved_placement_and_tilt(scene):
    poses, X, obs = scene
    # two image islands: laps share no verified pair
    split = truth_pairs(poses, X, gaps=(1, 2), distant=False, exclude=lambda i, j: i < N_PER_LAP <= j)
    # lap 2 rolled 20 degrees about its (outward) viewing direction at its first
    # camera: a misrotation about a horizontal axis
    axis = poses[N_PER_LAP][:3, 2]
    Q = _rot_axis(axis, math.radians(20.0))
    c0 = poses[N_PER_LAP][:3, 3]
    tilted = []
    for k, T in enumerate(poses):
        T2 = T.copy()
        if k >= N_PER_LAP:
            T2[:3, :3] = Q @ T[:3, :3]
            T2[:3, 3] = c0 + Q @ (T[:3, 3] - c0)
        tilted.append(T2)
    v, ids = make_variant(poses, X, None)
    ok = run(v, ids, pairs=split)["islands"]
    assert ok["large_islands"] == 2 and ok["large_island_pairs_unobserved"] == 1
    assert ok["island_pairs"][0]["placement"] == "UNOBSERVED"
    assert ok["max_tilt_vs_main_island_deg"] < 1.0
    t, _ = make_variant(tilted, X, None)
    bad = run(t, ids, pairs=split)["islands"]
    # a full-yaw lap: the tilt shows as roll scatter (sinusoidal in yaw)
    assert bad["tilt"]["1"]["roll_under_reference_mad_deg"] > 8.0
    # a strict cross-island pair makes placement observed, and scores the variant
    x = truth_pairs(poses, X, gaps=(), distant=True)
    x.arrays["strict"] = np.ones(len(x.arrays["i"]), bool)
    both = PairSet.from_arrays(split.arrays, xarrays=x.arrays)
    obs_ok = run(v, ids, pairs=both)["islands"]
    assert obs_ok["large_island_pairs_observed"] == 1
    assert obs_ok["island_pairs"][0]["variant"]["rot_err_median_deg"] < 1e-6
    obs_bad = run(t, ids, pairs=both)["islands"]
    assert obs_bad["island_pairs"][0]["variant"]["rot_err_median_deg"] == pytest.approx(20.0, abs=0.01)


def test_concentrated_yaw_island_tilt_is_its_median_roll():
    from tower.world_builder.coherence_eval.eval_placement import roll_free_up

    rng = np.random.default_rng(5)
    Rs = []
    for yaw in rng.uniform(-0.6, 0.6, 40):  # a desk: yaw within +-35 deg, pitch varies
        Rs.append(_rot_y(yaw) @ _rot_axis([1, 0, 0], rng.uniform(-0.5, 0.2)))
    up = roll_free_up(np.array(Rs))
    assert up["well_conditioned"]
    assert M.vec_angle_deg(up["up"], np.array([0, -1.0, 0]))[0] < 1.0


# ---------------------------------------------------------------------------
# M1 / L1 / L2 and the adapters


def _write_world(root: Path, n=8):
    """A tiny world directory: session, keyframes, a global solution and the
    derived tree merge() would write for it (two segments, one placement)."""
    from tower.world_builder.global_solve import rotation_to_quaternion_wxyz

    wid, sid = "w" * 32, "s" * 32
    wd = root / "worlds" / wid
    (wd / "sessions" / sid).mkdir(parents=True)
    (wd / "world.json").write_text(json.dumps({"world_id": wid}))
    intr = {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "dist_coeffs": [0, 0, 0, 0, 0],
            "calibrated_width": W, "calibrated_height": H}
    (wd / "sessions" / sid / "session.json").write_text(json.dumps({"session_id": sid, "intrinsics": intr}))
    kfs = []
    for k in range(n):
        kfs.append({"keyframe_id": f"{sid}:{k + 1:08d}", "image_relpath": f"images/{k + 1:08d}.jpg",
                    "width": W, "height": H, "segment_index": 0 if k < 4 else 1, "received_at": 100.0 + k})
    (wd / "sessions" / sid / "keyframes.jsonl").write_text("\n".join(json.dumps(r) for r in kfs) + "\n")
    poses, X = truth_scene()
    poses = poses[:n]
    sol_poses = {}
    for k in range(n - 1):  # the last keyframe stays unposed
        T = poses[k]
        Rcw = T[:3, :3].T
        sol_poses[kfs[k]["keyframe_id"]] = {"component": 0, "rotation": list(Rcw.reshape(-1)),
                                            "translation": list(-Rcw @ T[:3, 3]), "observations": 100}
    sd = wd / "solve" / sid
    sd.mkdir(parents=True)
    cam = {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "width": W, "height": H}
    (sd / "solution.json").write_text(json.dumps({
        "schema_version": 1, "solver": "glomap", "solved_at": 0.0, "input_digest": "d",
        "keyframe_ids": [r["keyframe_id"] for r in kfs], "components": [], "poses": sol_poses,
        "camera": cam, "timing": {}}))
    obs, xy = [], []
    Xs = X[:50]
    for k in range(n - 1):
        Pc = _cam(poses[k], Xs)
        for q in np.nonzero(Pc[:, 2] > 0.2)[0]:
            obs.append([k, 0, q])
            xy.append([FX * Pc[q, 0] / Pc[q, 2] + CX, FY * Pc[q, 1] / Pc[q, 2] + CY])
    np.savez_compressed(sd / "solution.npz", xyz=Xs.astype(np.float32), rgb=np.zeros((50, 3), np.uint8),
                        component=np.zeros(50, np.int32), first_keyframe=np.zeros(50, np.int32),
                        track_length=np.ones(50, np.int32), error=np.zeros(50, np.float32),
                        observations=np.array(obs, np.int32), observation_xy=np.array(xy, np.float32))
    dd = wd / "derived" / sid
    dd.mkdir(parents=True)
    rows, pls = [], []
    anchors = {0: poses[0], 1: poses[4]}
    for k in range(n):
        seg = kfs[k]["segment_index"]
        if k == n - 1:
            rows.append({"keyframe_id": kfs[k]["keyframe_id"], "segment_index": seg, "status": "unavailable",
                         "rotation": None, "translation": None})
            continue
        A = anchors[seg]
        rel = np.linalg.inv(A) @ poses[k]
        rows.append({"keyframe_id": kfs[k]["keyframe_id"], "segment_index": seg,
                     "status": "anchor" if k in (0, 4) else "solved",
                     "rotation": rotation_to_quaternion_wxyz(rel[:3, :3]), "translation": list(rel[:3, 3])})
    for seg in (0, 1):
        rel = np.linalg.inv(anchors[0]) @ anchors[seg]
        pls.append({"segment_index": seg, "state": "registered",
                    "rotation_wxyz": rotation_to_quaternion_wxyz(rel[:3, :3]), "translation": list(rel[:3, 3]),
                    "scale": 1.0, "reference_segment": 0})
    (dd / "poses.json").write_text(json.dumps({"poses": rows}))
    (dd / "placements.json").write_text(json.dumps({"placements": pls}))
    return wd, poses, kfs


def test_world_adapter_composes_derived_through_placements(tmp_path):
    from tower.world_builder.coherence_eval.eval_world import open_world

    wd, poses, kfs = _write_world(tmp_path)
    w = open_world(wd)
    v = variant_from_world(w)
    assert len(v.poses) == 7 and all(s == PUBLISHED for s in v.status.values())
    for k in range(7):
        assert np.allclose(v.poses[kfs[k]["keyframe_id"]], poses[k], atol=1e-9)
    chk = v.meta["derived_vs_solution"]["0"]
    assert chk["max_centre_residual_rel_p90"] < 1e-9 and chk["max_rotation_residual_deg"] < 1e-4
    assert v.obs_uv is not None and len(v.obs_uv) > 0


def test_colmap_adapter_reads_a_text_model(tmp_path):
    from tower.world_builder.coherence_eval.eval_world import open_world

    wd, poses, kfs = _write_world(tmp_path)
    w = open_world(wd)
    model = tmp_path / "sparse" / "0"
    model.mkdir(parents=True)
    (model / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {FX} {FY} {CX} {CY}\n")
    lines = []
    for k in range(3):
        Rcw = poses[k][:3, :3].T
        t = -Rcw @ poses[k][:3, 3]
        from tower.world_builder.global_solve import rotation_to_quaternion_wxyz

        q = rotation_to_quaternion_wxyz(Rcw)
        lines.append(f"{k + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {k + 1:08d}.jpg")
        lines.append("")
    (model / "images.txt").write_text("\n".join(lines) + "\n")
    (model / "points3D.txt").write_text("")
    v = variant_from_colmap(tmp_path / "sparse", w, publish_min_observations=0, publish_min_model_images=0)
    assert len(v.poses) == 3 and v.meta["unmatched_images"] == 0
    for k in range(3):
        assert np.allclose(v.poses[kfs[k]["keyframe_id"]], poses[k], atol=1e-6)
    v2 = variant_from_colmap(tmp_path / "sparse", w)  # the product floor: 30 observations
    assert all(s == POSED for s in v2.status.values())


def test_a_variant_of_another_world_is_refused(tmp_path):
    from tower.world_builder.coherence_eval.eval_world import open_world

    wd, poses, kfs = _write_world(tmp_path)
    w = open_world(wd)
    v = variant_from_world(w)
    save_variant(v, tmp_path / "var")
    doc = json.loads((tmp_path / "var" / "reconstruction.json").read_text())
    assert doc["meta"]["adapted_from"] == "world" and "adapter" not in doc["meta"]
    assert load_variant(tmp_path / "var", w).meta["adapter"] == "interchange"
    doc["world_id"] = "x" * 32
    (tmp_path / "var" / "reconstruction.json").write_text(json.dumps(doc))
    with pytest.raises(VariantMismatch):
        load_any(tmp_path / "var", w)
    forced = load_any(tmp_path / "var", w, force=True)
    assert forced.meta["world_mismatch_forced"]


def test_an_improper_rotation_is_refused(tmp_path, scene):
    poses, X, obs = scene
    v, ids = make_variant(poses[:5], X, None)
    save_variant(v, tmp_path / "var")
    doc = json.loads((tmp_path / "var" / "reconstruction.json").read_text())
    T = np.array(doc["keyframes"][0]["T_world_camera"]).reshape(4, 4)
    T[:3, :3] *= 2.0  # a Sim(3) scale folded into R
    doc["keyframes"][0]["T_world_camera"] = list(T.reshape(-1))
    (tmp_path / "var" / "reconstruction.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="proper rotation"):
        load_variant(tmp_path / "var")


def test_unpublished_poses_count_only_in_registration(scene):
    poses, X, obs = scene
    status = [POSED if 30 <= k < 40 else PUBLISHED for k in range(N)]
    bad = [T.copy() for T in poses]
    bad[35][:3, 3] += 50.0
    v, ids = make_variant(bad, X, None, status=status)
    out = run(v, ids)
    assert out["registration"]["posed_fraction"] == 1.0
    assert out["registration"]["published_fraction"] == pytest.approx(1 - 10 / N)
    assert out["sanity"]["main"]["outlier_cameras"] == 0
    assert out["continuity"]["main"]["spikes"] == 0


def test_region_clean_requires_on_level_scale(scene):
    poses, X, obs = scene
    kf, pt, uv, z = obs
    half = 100  # the "closet" (100..159) is at twice the scale of the "desk"
    scaled, pivot = scaled_block(poses, half, 2.0)
    X2 = np.concatenate([X, pivot + 2.0 * (X - pivot)])
    pt2 = np.where(kf >= half, pt + len(X), pt)
    v, ids = make_variant(scaled, X2, (kf, pt2, uv, z))
    regions = {ids[k]: {"region": "desk" if k < 100 else "closet", "confidence": "high"} for k in range(N)}
    out = run(v, ids, regions=regions, depth_fn=truth_depth_fn(obs))
    reg = out["regions"]["regions"]
    assert reg["desk"]["in_main"] == 1.0 and reg["closet"]["in_main"] == 1.0
    assert reg["closet"]["in_main_clean"] == 0.0  # in main, but at twice the scale
    assert out["scale"]["depth"]["by_region"]["closet"]["factor_vs_main"] == pytest.approx(2.0, rel=0.02)


def test_interchange_roundtrip(tmp_path, scene):
    poses, X, obs = scene
    comps = ["0" if k < 100 else "1" for k in range(N)]
    v, ids = make_variant(poses, X, obs, components=comps)
    v.meta = {"variant": "synthetic", "code_commit": "abc", "params": {"k": 1}}
    save_variant(v, tmp_path / "var")
    doc = json.loads((tmp_path / "var" / "reconstruction.json").read_text())
    assert doc["format"] == "wb-coherence-variant/1"
    back = load_variant(tmp_path / "var")
    assert set(back.poses) == set(v.poses)
    for k in v.poses:
        assert np.allclose(back.poses[k], v.poses[k])
        assert back.component[k] == v.component[k]
    assert np.allclose(back.xyz, v.xyz)
    assert np.array_equal(back.obs_point, v.obs_point)
    a = run(v, ids)
    b = run(back, ids)
    a.pop("integrity"), b.pop("integrity")
    assert M.round_sig(a) == M.round_sig(b)


def test_outputs_are_deterministic_and_comparable(tmp_path, scene):
    poses, X, obs = scene
    v, ids = make_variant(poses, X, obs)
    body = run(v, ids, pairs=truth_pairs(poses), depth_fn=truth_depth_fn(obs))
    result = {"harness": {"version": M.HARNESS, "params": M.PARAMS, "code_digest": "x", "commit": None},
              "world": {"world_id": "w" * 32, "session_id": "s", "keyframes": N, "world_dir": "x",
                        "canonical_camera": None,
                        "caches": {"depth": {"available": True, "params": None, "digest": "e"},
                                   "pairs": {"available": True, "count": 1, "digest": "d"}}},
              "variant": {"name": "a", "image_space": "canonical", "meta": {}, "posed": N, "points": 1,
                          "observations": 1},
              "metrics": {k: v for k, v in body.items() if not k.startswith("_")}}
    M.write_outputs(result, tmp_path / "a")
    M.write_outputs(result, tmp_path / "b")
    ja = (tmp_path / "a" / "metrics.json").read_text()
    assert ja == (tmp_path / "b" / "metrics.json").read_text()
    text = M.compare_markdown(json.loads(ja), json.loads(ja))
    assert "| jumps (main; index+time or physical) | 0 | 0 | 0 |" in text
    assert "Overall: B better on 0 rows, A better on 0" in text


def test_camera_models_project_like_opencv():
    import cv2

    cam = {"model": "OPENCV", "width": 360, "height": 640, "params": [400, 401, 180, 320, 0.1, -0.05, 0.001, 0.002]}
    P = np.array([[0.3, -0.2, 2.0], [-0.5, 0.4, 3.0], [0.0, 0.0, 1.0]])
    uv = M.project(cam, P)
    Kc = np.array([[400, 0, 180], [0, 401, 320], [0, 0, 1.0]])
    ref = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kc,
                            np.array([0.1, -0.05, 0.001, 0.002, 0]))[0].reshape(-1, 2)
    assert np.allclose(uv, ref)
    back = M.unproject_normalized(cam, uv)
    assert np.allclose(back, P[:, :2] / P[:, 2:], atol=1e-6)
    assert M.project({"model": "OPENCV_FISHEYE", "params": [1, 1, 1, 1, 0, 0, 0, 0]}, P) is None


def test_measure_records_wall_time_and_rss(tmp_path):
    import sys

    rec = M.measure_command([sys.executable, "-c", "import time; x = bytearray(50_000_000); time.sleep(0.6)"],
                            poll_s=0.1)
    assert rec["returncode"] == 0
    assert rec["wall_s"] >= 0.5
    assert rec["peak_rss_mb"] >= 40
    assert rec["samples"] >= 2


def test_unstated_components_come_from_covisibility(scene):
    poses, X, obs = scene
    kf, pt, uv, z = obs
    pt2 = np.where(kf >= 100, pt + len(X), pt)
    v, ids = make_variant(poses, np.concatenate([X, X]), (kf, pt2, uv, z))
    v.component_stated = False
    out = run(v, ids)
    assert out["components"]["count"] == 2
    assert out["components"]["sizes"] == [100, 60]
    assert out["components"]["definition"].startswith("not stated: covisibility")
    v.component_stated = True
    assert run(v, ids)["components"]["count"] == 1


# ---------------------------------------------------------------------------
# golden: the frozen target world (skipped where the evidence is not on disk)

RUN = Path(os.environ.get("WB_COHERENCE_RUN",
                          Path.home() / "Projects" / "Glasses-scratch" / "wb-coherence-run-2026-09-23"))
TARGET = "6e6d3fc30e7b45f3a7521e618386b649"
GOLDEN_KEYS = [
    "registration.published_fraction", "registration.in_largest_fraction", "components.count",
    "islands.large_islands", "islands.large_island_pairs_unobserved", "islands.max_tilt_vs_main_island_deg",
    "continuity.main.jumps", "continuity.main.tracking_loss_steps_without_image_evidence",
    "scale.depth.fixed_denominator.bands.x1.25.coherent_all", "scale.depth.main.max_step_factor",
    "scale.depth_tri.fixed_denominator.bands.x1.25.coherent_all", "scale.depth_tri.main.max_step_factor",
    "revisits.revisit.pairs", "revisits.revisit.rot_err_deg.median", "reprojection.overall.median",
    "islands.placement_plausibility.groups_failing", "islands.placement_plausibility.max_scale_deviation_factor",
    "islands.placement_plausibility.max_height_offset_beyond_band_m", "rigid_groups.count",
    "rigid_groups.placement_plausibility.groups_failing",
]


@pytest.mark.skipif(not (RUN / "baseline" / "frozen" / "worlds" / TARGET).is_dir()
                    or not (RUN / "baseline" / "metrics" / "golden_target.json").is_file(),
                    reason="frozen target world or golden file not on this machine")
def test_golden_target_world():
    from tower.world_builder.coherence_eval.eval_world import open_world

    golden = json.loads((RUN / "baseline" / "metrics" / "golden_target.json").read_text())
    w = open_world(RUN / "baseline" / "frozen" / "worlds" / TARGET)
    res = M.evaluate_world_variant(w, variant_from_world(w), cache_root=RUN / "baseline" / "metrics" / "cache",
                                   regions_dir=RUN / "baseline" / "forensics" / "regions")
    m = M.round_sig(res["metrics"])
    got = {k: M.get_path(m, k) for k in GOLDEN_KEYS}
    for k in GOLDEN_KEYS:
        want = golden["values"][k]
        if isinstance(want, float):
            assert got[k] == pytest.approx(want, rel=1e-4, abs=1e-9), k
        else:
            assert got[k] == want, k
    assert res["world"]["caches"]["pairs"]["digest"] == golden["pairs_digest"]
    assert res["world"]["caches"]["depth"]["digest"] == golden["depth_digest"]


# ---------------------------------------------------------------------------
# placement plausibility between islands / rigid groups (tilt, scale, height)


def two_rooms():
    """Two rooms 20 m apart, 40 level cameras each (yaw -30..30 deg, small
    pitch), every camera at eye height 0 (world y down), each room facing its
    own wall at z = 5: no shared point, no image pair between the rooms."""
    rng = np.random.default_rng(4)
    poses = []
    for k in range(80):
        room, j = divmod(k, 40)
        yaw = math.radians(-30 + 60 * j / 39)
        pitch = math.radians(3.0 if j % 2 else -3.0)
        T = np.eye(4)
        T[:3, :3] = _rot_y(yaw) @ _rot_axis([1, 0, 0], pitch)
        T[:3, 3] = [20.0 * room - 3 + 6 * j / 39, 0.0, 0.0]  # 15 cm steps: triangulable
        poses.append(T)
    X = np.concatenate([np.c_[rng.uniform(-10, 9.5, 900), rng.uniform(-1.5, 1.5, 900), np.full(900, 5.0)],
                        np.c_[rng.uniform(10.5, 30, 900), rng.uniform(-1.5, 1.5, 900), np.full(900, 5.0)]])
    return poses, X


def _plane_depth(poses):
    def fn(i, uv):
        T = poses[i]
        uv = np.asarray(uv, float).reshape(-1, 2)
        d = np.c_[(uv[:, 0] - CX) / FX, (uv[:, 1] - CY) / FY, np.ones(len(uv))] @ T[:3, :3].T
        return (5.0 - T[2, 3]) / d[:, 2]
    return fn


def _room_pairs(poses, X):
    I, J, Rs, ts, xi, xj, offs = [], [], [], [], [], [], [0]
    for g in (1, 2):
        for i in range(80 - g):
            j = i + g
            if i < 40 <= j:
                continue
            Pi, Pj = _cam(poses[i], X), _cam(poses[j], X)
            ui = Pi[:, :2] / np.maximum(Pi[:, 2:], 1e-9)
            uj = Pj[:, :2] / np.maximum(Pj[:, 2:], 1e-9)
            vis = np.nonzero((Pi[:, 2] > 0.2) & (Pj[:, 2] > 0.2) & (np.abs(ui[:, 0]) < 0.4) & (np.abs(uj[:, 0]) < 0.4)
                             & (np.abs(ui[:, 1]) < 0.7) & (np.abs(uj[:, 1]) < 0.7))[0][:64]
            Ri, Rj = poses[i][:3, :3].T, poses[j][:3, :3].T
            t = Rj @ (poses[i][:3, 3] - poses[j][:3, 3])
            I.append(i), J.append(j), Rs.append(Rj @ Ri.T), ts.append(t / np.linalg.norm(t))
            xi.append(ui[vis]), xj.append(uj[vis]), offs.append(offs[-1] + len(vis))
    P = len(I)
    return PairSet.from_arrays({
        "i": np.array(I, np.int32), "j": np.array(J, np.int32), "R": np.array(Rs), "t": np.array(ts),
        "t_reliable": np.ones(P, bool), "parallax_deg": np.full(P, 10.0, np.float32),
        "n_inliers": np.full(P, 100, np.int32), "n_matches": np.full(P, 120, np.int32),
        "inlier_offsets": np.array(offs, np.int64), "inlier_xy_i": np.concatenate(xi).astype(np.float32),
        "inlier_xy_j": np.concatenate(xj).astype(np.float32)})


def _move_room_b(poses, X, M3, t):
    """Apply x -> M3 (x - pivot) + pivot + t to room B's cameras and points
    (pivot = room B's median camera centre); rotations only through M3's
    rotation part."""
    pivot = np.median([T[:3, 3] for T in poses[40:]], axis=0)
    s = abs(np.linalg.det(M3)) ** (1 / 3)
    R = M3 / s
    out = [T.copy() for T in poses]
    for T in out[40:]:
        T[:3, :3] = R @ T[:3, :3]
        T[:3, 3] = M3 @ (T[:3, 3] - pivot) + pivot + t
    X2 = X.copy()
    X2[900:] = (M3 @ (X[900:] - pivot).T).T + pivot + t
    return out, X2


def _room_obs(poses_true, X):
    kf, pt, uv = [], [], []
    for k, T in enumerate(poses_true):
        Pc = _cam(T, X)
        z = Pc[:, 2]
        u = FX * Pc[:, 0] / np.where(z > 0, z, 1) + CX
        v = FY * Pc[:, 1] / np.where(z > 0, z, 1) + CY
        idx = np.nonzero((z > 0.2) & (u > 0) & (u < W) & (v > 0) & (v < H))[0]
        kf.append(np.full(len(idx), k)), pt.append(idx), uv.append(np.c_[u[idx], v[idx]])
    return np.concatenate(kf), np.concatenate(pt), np.concatenate(uv), None


@pytest.mark.parametrize("case, fails", [
    ("clean", set()),
    ("tilted", {"tilt"}),
    ("scaled", {"scale"}),
    ("raised", {"height"}),
])
def test_placement_plausibility_fails_the_right_check(case, fails):
    poses, X = two_rooms()
    move = {"clean": (np.eye(3), np.zeros(3)),
            "tilted": (_rot_axis([0, 0, 1], math.radians(20.0)), np.zeros(3)),   # horizontal axis
            "scaled": (2.0 * np.eye(3), np.zeros(3)),
            "raised": (np.eye(3), np.array([0.0, -0.6, 0.0]))}[case]            # world y is DOWN
    vposes, vX = _move_room_b(poses, X, *move)
    obs = _room_obs(poses, X)  # the images: observations of the TRUE scene
    v, ids = make_variant(vposes, vX, obs)
    out = run(v, ids, pairs=_room_pairs(poses, X), depth_sample=_plane_depth(poses))
    for block in (out["islands"]["placement_plausibility"],
                  out["rigid_groups"]["placement_plausibility"]):
        assert block["available"] and block["groups_checked"] == 1
        (other,) = [g for g, r in block["groups"].items() if not r["reference"]]
        r = block["groups"][other]
        got = {name for name, key in (("tilt", "tilt_ok"), ("scale", "scale_ok"), ("height", "height_ok"))
               if r[key] is False}
        assert got == fails, (block["kind"], case, r["line"])
        assert r["plausible"] is (not fails)
        assert block["units_per_metre_reference"] == pytest.approx(1.0, rel=0.02)
    r = out["islands"]["placement_plausibility"]["groups"]
    other = [g for g in r if not r[g]["reference"]][0]
    if case == "scaled":
        assert r[other]["scale_factor"] == pytest.approx(2.0, rel=0.02)
    if case == "raised":
        assert r[other]["height_offset_beyond_band_m"] == pytest.approx(0.6, abs=0.02)
    if case == "tilted":
        assert r[other]["tilt_deg"] > 15.0
    assert out["islands"]["placement_plausibility"]["unobservable_without_image_links"]


def test_eloftr_float_matches_are_the_library_postprocess_without_truncation():
    """V (P2-E3) found transformers' EfficientLoFTR post-process truncates
    keypoints to int32 -- up to 1 px, the whole verify_points threshold. The
    harness helper must select the same matches and scale identically, and
    differ from the library only by that truncation."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from types import SimpleNamespace

    from tower.world_builder.coherence_eval.eval_pairs import eloftr_matches_float

    try:
        from transformers.models.efficientloftr.image_processing_efficientloftr import (
            EfficientLoFTRImageProcessor,
        )
    except ImportError:  # pragma: no cover
        pytest.skip("transformers without EfficientLoFTR")
    g = torch.Generator().manual_seed(0)
    kp = torch.rand(1, 2, 50, 2, generator=g)
    matches = torch.where(torch.rand(1, 2, 50, generator=g) > 0.3, torch.arange(50).repeat(1, 2, 1), -1)
    matches[0, 1] = matches[0, 0]            # symmetric validity, as the model emits
    scores = torch.rand(1, 2, 50, generator=g)
    scores[0, 1] = scores[0, 0]
    out = SimpleNamespace(keypoints=kp, matches=matches, matching_scores=scores)
    sizes = [((639, 359), (639, 359))]
    lib = EfficientLoFTRImageProcessor().post_process_keypoint_matching(out, sizes, threshold=0.2)[0]
    ours = eloftr_matches_float(out, sizes, threshold=0.2)[0]
    assert ours["keypoints0"].shape == lib["keypoints0"].shape
    assert torch.equal(ours["keypoints0"].to(torch.int32), lib["keypoints0"])
    assert torch.equal(ours["keypoints1"].to(torch.int32), lib["keypoints1"])
    assert torch.equal(ours["matching_scores"], lib["matching_scores"])
    frac = (ours["keypoints0"] - ours["keypoints0"].floor()).abs()
    assert float(frac.max()) > 0.1   # sub-pixel information survives


def _minimal_pair_arrays(i, j):
    n = len(i)
    return {"i": np.asarray(i, np.int32), "j": np.asarray(j, np.int32),
            "R": np.repeat(np.eye(3)[None], n, 0), "t": np.tile([0.0, 0.0, 1.0], (n, 1)),
            "t_reliable": np.ones(n, bool)}


def test_pairset_never_reads_the_grid_snapped_eloftr_tier(tmp_path):
    """V4b: the transformers EfficientLoFTR tier (versions 1-2) snapped every
    match to an 8-px grid. Its files stay on disk but must not feed metrics;
    only the ALIKED+LightGlue tier (version 3) is read."""
    from tower.world_builder.coherence_eval import eval_pairs as EP

    np.savez_compressed(tmp_path / EP.LOFTR_ARRAYS, **_minimal_pair_arrays([0], [5]))
    (tmp_path / EP.LOFTR_MANIFEST).write_text(json.dumps({"complete": True}), encoding="utf-8")
    assert PairSet(tmp_path).larrays is None
    np.savez_compressed(tmp_path / EP.LEARNED_ARRAYS, **_minimal_pair_arrays([1], [7]))
    (tmp_path / EP.LEARNED_MANIFEST).write_text(json.dumps({"complete": True}), encoding="utf-8")
    ps = PairSet(tmp_path)
    assert ps.larrays["i"].tolist() == [1] and ps.larrays["j"].tolist() == [7]


def test_learned_tier_verifies_the_matchers_points_across_islands(tmp_path):
    """The tier asks the matcher for every cross-island candidate and keeps
    exactly the pairs verify_points accepts, under the version-3 files."""
    from types import SimpleNamespace

    from tower.world_builder.coherence_eval import eval_pairs as EP

    rng = np.random.default_rng(5)
    Xw = np.c_[rng.uniform(-2, 2, 400), rng.uniform(-3, 3, 400), rng.uniform(4, 8, 400)]
    Rj = _rot_axis([0.1, 1, 0.05], 0.15)
    Cj = np.array([0.6, 0.05, 0.1])
    Pj = (Rj.T @ (Xw - Cj).T).T
    ui = (K @ (Xw / Xw[:, 2:]).T).T[:, :2]
    uj = (K @ (Pj / Pj[:, 2:]).T).T[:, :2]
    ok = (ui[:, 0] > 0) & (ui[:, 0] < W) & (ui[:, 1] > 0) & (ui[:, 1] < H) & \
         (uj[:, 0] > 0) & (uj[:, 0] < W) & (uj[:, 1] > 0) & (uj[:, 1] < H)
    ui, uj = ui[ok], uj[ok]

    class Matcher:
        calls = []

        def match(self, a, b, key_a, key_b):
            self.calls.append((key_a, key_b))
            if (key_a, key_b) == (3, 14):
                return ui, uj, np.ones(len(ui), np.float32)
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, np.float32)

    n = 20
    world = SimpleNamespace(world_id="w", session_id="s", n=n, canonical_K=lambda: K,
                            canonical_camera={"width": W, "height": H},
                            canonical_image=lambda i: (np.zeros((H, W, 3), np.uint8), "fake"))
    root = EP.pairs_dir(tmp_path, "w")
    root.mkdir(parents=True)
    chain = [(k, k + 1) for k in range(9)] + [(k, k + 1) for k in range(10, 19)]
    np.savez_compressed(root / "pairs.npz", **_minimal_pair_arrays(*zip(*chain)))
    (root / "manifest.json").write_text(json.dumps({"complete": True}), encoding="utf-8")
    desc = rng.normal(size=(n, 8)).astype(np.float32)
    np.save(root / "descriptors.npy", desc / np.linalg.norm(desc, axis=1, keepdims=True))
    matcher = Matcher()
    m = EP.build_learned_cross_island_tier(world, tmp_path, matcher=matcher, log=lambda s: None)
    assert len(matcher.calls) == 100          # every 10 x 10 cross-island candidate
    assert m["params"]["version"] == 3
    assert m["counts"] == {"candidates": 100, "verified_strict": 1}
    assert m["links_by_island_pair"] == {"0-1": 1}
    ps = PairSet(root)
    assert ps.larrays["i"].tolist() == [3] and ps.larrays["j"].tolist() == [14]
    assert not (root / EP.LOFTR_ARRAYS).exists()
