"""The coherence evaluation harness on synthetic trajectories.

Every metric is exercised on a scene whose truth is known:
a clean two-lap closed loop, the same loop with an injected jump and with a
spike, split into two components, with a scale step between halves, under a
random Sim(3) per component (gauge invariance), and against a synthetic
verified pair set with a known injected rotation error.
"""

import json
import math

import numpy as np
import pytest

from tower.world_builder.coherence_eval import metrics as M
from tower.world_builder.coherence_eval.eval_pairs import PairSet, wahba_rotation
from tower.world_builder.coherence_eval.eval_variant import (
    POSED,
    PUBLISHED,
    Variant,
    load_variant,
    save_variant,
)

N_PER_LAP = 80
LAPS = 2
N = N_PER_LAP * LAPS
W, H = 360, 640
FX = FY = 400.0
CX, CY = 180.0, 320.0
CAM = {"model": "PINHOLE", "width": W, "height": H, "params": [FX, FY, CX, CY]}


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_axis(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def truth_scene(seed=0):
    """Cameras on a circle (radius 2, y down is OpenCV 'down'), looking outward
    at a cylindrical wall of points (radius 5). Two laps; lap 2 is lap 1 with a
    small radial offset so revisits have parallax."""
    rng = np.random.default_rng(seed)
    poses = []
    for k in range(N):
        lap, j = divmod(k, N_PER_LAP)
        theta = 2 * math.pi * j / N_PER_LAP
        r = 2.0 + 0.3 * lap
        c = np.array([r * math.sin(theta), 0.1 * math.sin(3 * theta), r * math.cos(theta)])
        # camera z (forward) points outward along the radius: R_wc = rot_y(theta)
        R = _rot_y(theta)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = c
        poses.append(T)
    n_pts = 1500
    phi = rng.uniform(0, 2 * math.pi, n_pts)
    X = np.c_[5.0 * np.sin(phi), rng.uniform(-1.5, 1.5, n_pts), 5.0 * np.cos(phi)]
    return poses, X


def observe(poses, X, rng, noise_px=0.2):
    ok_k, ok_p, uv, depth = [], [], [], []
    for k, T in enumerate(poses):
        Rcw = T[:3, :3].T
        Pc = (Rcw @ (X - T[:3, 3]).T).T
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


def truth_pairs(poses, gaps=(1,), distant=True):
    """Exact relative poses: adjacent pairs plus lap-1 / lap-2 revisits."""
    I, J = [], []
    for g in gaps:
        for i in range(N - g):
            I.append(i)
            J.append(i + g)
    if distant:
        for i in range(N_PER_LAP):
            I.append(i)
            J.append(i + N_PER_LAP)
    Rs, ts = [], []
    for i, j in zip(I, J):
        Ri, Rj = poses[i][:3, :3].T, poses[j][:3, :3].T
        R = Rj @ Ri.T
        t = Rj @ (poses[i][:3, 3] - poses[j][:3, 3])
        Rs.append(R)
        ts.append(t / np.linalg.norm(t))
    P = len(I)
    arrays = {"i": np.array(I, np.int32), "j": np.array(J, np.int32), "R": np.array(Rs), "t": np.array(ts),
              "t_reliable": np.ones(P, bool), "parallax_deg": np.full(P, 10.0, np.float32),
              "n_inliers": np.full(P, 100, np.int32), "n_matches": np.full(P, 120, np.int32)}
    return PairSet.from_arrays(arrays)


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(1)
    poses, X = truth_scene()
    obs = observe(poses, X, rng)
    return poses, X, obs


def run(v, ids, pairs=None, depth_fn=None, segment_of=None, regions=None, times=None):
    return M.evaluate(v, ids, segment_of=segment_of, times=times, pairs=pairs, depth_fn=depth_fn,
                      regions=regions, focal_px=FX)


# ---------------------------------------------------------------------------


def test_clean_closed_loop(scene):
    poses, X, obs = scene
    v, ids = make_variant(poses, X, obs)
    out = run(v, ids, pairs=truth_pairs(poses), depth_fn=truth_depth_fn(obs))
    reg = out["registration"]
    assert reg["published_fraction"] == 1.0 and reg["in_largest_fraction"] == 1.0
    assert out["components"]["count"] == 1 and out["components"]["main_runs"] == 1
    main = out["continuity"]["main"]
    assert main["jumps"] == 0 and main["spikes"] == 0
    assert main["max_ratio"] < 3.0  # the lap change (radius 2.0 -> 2.3) is the largest step
    assert out["sanity"]["main"]["outlier_cameras"] == 0
    rp = out["reprojection"]["overall"]
    assert rp["median"] < 0.5 and out["reprojection"]["behind_camera"] == 0
    d = out["scale"]["depth"]["main"]
    assert d["spread_mad"] < 1e-3 and d["steps"] == 0 and d["drift_factor_over_span"] < 1.01
    for bucket in ("adjacent", "distant"):
        r = out["revisits"][bucket]
        assert r["joined_fraction"] == 1.0
        assert r["rot_err_deg"]["max"] < 1e-6
        assert r["t_err_deg"]["max"] < 1e-4


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
    (jump,) = main["jump_list"]
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
    # the second component in an unrelated gauge
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
    # continuity is measured inside each component: no jump at the split
    assert out["continuity"]["all_components"]["jumps"] == 0
    dist = out["revisits"]["distant"]
    # revisit pairs (i, i+80): i >= 20 lands in component 1 for j, i < 100 in 0
    assert dist["split_across_components"] == 60
    assert dist["joined_fraction"] == pytest.approx(20 / 80)
    assert dist["rot_err_deg"]["max"] < 1e-6  # the joined ones are exact
    assert out["revisits"]["adjacent"]["split_across_components"] == 1


def test_scale_step_between_halves(scene):
    poses, X, obs = scene
    kf, pt, uv, z = obs
    half = N // 2
    pivot = poses[half - 1][:3, 3]
    s = 2.0
    scaled = []
    for k, T in enumerate(poses):
        T2 = T.copy()
        if k >= half:
            T2[:3, 3] = pivot + s * (T[:3, 3] - pivot)
        scaled.append(T2)
    # second-half observations see scaled copies of the points (their own tracks)
    X2 = np.concatenate([X, pivot + s * (X - pivot)])
    pt2 = np.where(kf >= half, pt + len(X), pt)
    v, ids = make_variant(scaled, X2, (kf, pt2, uv, z))
    segment_of = [0 if k < half else 1 for k in range(N)]
    out = run(v, ids, depth_fn=truth_depth_fn(obs), segment_of=segment_of)
    assert out["reprojection"]["overall"]["median"] < 0.5  # still self-consistent
    d = out["scale"]["depth"]["main"]
    assert d["steps"] == 1
    (step,) = d["step_list"]
    assert abs(step["at_index"] - half) <= 1
    assert step["factor"] == pytest.approx(s, rel=1e-3)
    assert d["drift_slope_per_100kf"] > 0
    seg = out["scale"]["depth"]["by_tracker_segment"]
    assert seg["max_over_min_factor"] == pytest.approx(s, rel=1e-3)
    assert out["continuity"]["main"]["jumps"] == 0  # scaled about the junction: no jump


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


def test_gauge_invariance_under_random_sim3_per_component(scene):
    poses, X, obs = scene
    comps = ["0" if k < 110 else "1" for k in range(N)]
    kf, pt, uv, z = obs
    # give component 1 its own copies of the points it observes, so each
    # component can take its own Sim(3)
    pt_c = np.where(kf >= 110, pt + len(X), pt)
    X_c = np.concatenate([X, X])
    point_comp = np.array(["0"] * len(X) + ["1"] * len(X))
    v, ids = make_variant(poses, X_c, (kf, pt_c, uv, z), components=comps)
    v.point_component = point_comp
    pairs = truth_pairs(poses, gaps=(1, 2))
    base = run(v, ids, pairs=pairs, depth_fn=truth_depth_fn(obs), segment_of=[k // 40 for k in range(N)])

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
    # the monocular depth does not move with the gauge: re-express it so the
    # comparison is between two reconstructions of the SAME images
    moved = run(v2, ids, pairs=pairs, depth_fn=truth_depth_fn(obs), segment_of=[k // 40 for k in range(N)])

    a, b = _flatten(M.round_sig(base)), _flatten(M.round_sig(moved))
    assert set(a) == set(b)
    diffs = []
    for key in a:
        if ("units_per_metre_informative" in key or "d3_pooled_informative" in key
                or "depth_log_ratio" in key or key.startswith("_per_keyframe")):
            continue  # gauge-DEPENDENT by definition, and labelled so
        x, y = a[key], b[key]
        if isinstance(x, float) and isinstance(y, float):
            if not math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-6):
                diffs.append((key, x, y))
        elif x != y:
            diffs.append((key, x, y))
    assert not diffs, diffs[:10]


def test_revisit_errors_on_a_synthetic_pair_set(scene):
    poses, X, obs = scene
    # lap 2 rotated 5 degrees about the loop's vertical axis (positions AND
    # orientations): a rigid mis-registration of the revisit
    ang = math.radians(5.0)
    Rz = _rot_axis([0, 1, 0], ang)
    moved = []
    for k, T in enumerate(poses):
        T2 = T.copy()
        if k >= N_PER_LAP:
            T2[:3, :3] = Rz @ T[:3, :3]
            T2[:3, 3] = Rz @ T[:3, 3]
        moved.append(T2)
    v, ids = make_variant(moved, X, None)
    out = run(v, ids, pairs=truth_pairs(poses))
    dist = out["revisits"]["distant"]
    assert dist["evaluated"] == N_PER_LAP
    assert dist["rot_err_deg"]["median"] == pytest.approx(5.0, abs=1e-6)
    assert dist["rot_err_deg"]["over_2"] == 1.0
    adj = out["revisits"]["adjacent"]
    # only the lap boundary pair (79, 80) straddles the rotation
    assert adj["rot_err_deg"]["max"] == pytest.approx(5.0, abs=1e-6)
    assert adj["rot_err_deg"]["median"] < 1e-6
    # and the direction errors are real, computed in camera j's frame
    assert dist["t_err_deg"]["median"] > 0.5


def test_revisit_rotation_is_measured_against_the_pair_not_the_gauge():
    # the stored pair convention is x_j = R x_i + t; check the adapter math
    R_wc_i = _rot_axis([0, 1, 0], 0.3)
    R_wc_j = _rot_axis([1, 0, 0], -0.2) @ R_wc_i
    Ti, Tj = np.eye(4), np.eye(4)
    Ti[:3, :3], Tj[:3, :3] = R_wc_i, R_wc_j
    Ti[:3, 3], Tj[:3, 3] = [0, 0, 0], [1, 0, 0.5]
    Rj = R_wc_j.T
    R_ji = Rj @ R_wc_i
    t_ji = Rj @ (Ti[:3, 3] - Tj[:3, 3])
    X = np.array([0.3, -0.2, 4.0])
    x_i = R_wc_i.T @ (X - Ti[:3, 3])
    x_j = Rj @ (X - Tj[:3, 3])
    assert np.allclose(R_ji @ x_i + t_ji, x_j)


def test_unpublished_poses_count_only_in_registration(scene):
    poses, X, obs = scene
    status = [POSED if 30 <= k < 40 else PUBLISHED for k in range(N)]
    bad = [T.copy() for T in poses]
    bad[35][:3, 3] += 50.0  # a wild pose the variant itself did not publish
    v, ids = make_variant(bad, X, None, status=status)
    out = run(v, ids)
    assert out["registration"]["posed_fraction"] == 1.0
    assert out["registration"]["published_fraction"] == pytest.approx(1 - 10 / N)
    assert out["sanity"]["main"]["outlier_cameras"] == 0
    assert out["continuity"]["main"]["spikes"] == 0


def test_region_coverage_uses_labels_only_for_reporting(scene):
    poses, X, obs = scene
    v, ids = make_variant([T if k < 150 else None for k, T in enumerate(poses)], X, None)
    regions = {ids[k]: {"region": "desk" if k < 100 else "closet", "confidence": "high"} for k in range(N)}
    out = run(v, ids, regions=regions)
    reg = out["regions"]["regions"]
    assert reg["desk"]["in_main"] == 1.0
    assert reg["closet"]["in_main"] == pytest.approx(50 / 60)


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
    assert M.round_sig(a) == M.round_sig(b)


def test_outputs_are_deterministic_and_comparable(tmp_path, scene):
    poses, X, obs = scene
    v, ids = make_variant(poses, X, obs)
    body = run(v, ids, pairs=truth_pairs(poses), depth_fn=truth_depth_fn(obs))
    result = {"harness": {"version": M.HARNESS, "params": M.PARAMS, "code_digest": "x", "commit": None},
              "world": {"world_id": "w" * 32, "session_id": "s", "keyframes": N, "world_dir": "x",
                        "canonical_camera": None,
                        "caches": {"depth": {"available": True, "params": None},
                                   "pairs": {"available": True, "count": 1, "digest": "d"}}},
              "variant": {"name": "a", "image_space": "canonical", "meta": {}, "posed": N, "points": 1,
                          "observations": 1},
              "metrics": {k: v for k, v in body.items() if not k.startswith("_")}}
    M.write_outputs(result, tmp_path / "a")
    M.write_outputs(result, tmp_path / "b")
    ja = (tmp_path / "a" / "metrics.json").read_text()
    assert ja == (tmp_path / "b" / "metrics.json").read_text()
    text = M.compare_markdown(json.loads(ja), json.loads(ja))
    assert "| jumps (main; index+time) | 0 | 0 | 0 |" in text


def test_wahba_rotation_recovers_a_pure_rotation():
    rng = np.random.default_rng(3)
    f1 = rng.normal(size=(50, 3))
    f1 /= np.linalg.norm(f1, axis=1, keepdims=True)
    R = _rot_axis([0.2, 1, 0.1], 0.4)
    assert np.allclose(wahba_rotation(f1, f1 @ R.T), R)


def test_camera_models_project_like_opencv():
    import cv2

    cam = {"model": "OPENCV", "width": 360, "height": 640, "params": [400, 401, 180, 320, 0.1, -0.05, 0.001, 0.002]}
    P = np.array([[0.3, -0.2, 2.0], [-0.5, 0.4, 3.0], [0.0, 0.0, 1.0]])
    uv = M.project(cam, P)
    K = np.array([[400, 0, 180], [0, 401, 320], [0, 0, 1.0]])
    ref = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K,
                            np.array([0.1, -0.05, 0.001, 0.002, 0]))[0].reshape(-1, 2)
    assert np.allclose(uv, ref)
    back = M.unproject_normalized(cam, uv)
    assert np.allclose(back, P[:, :2] / P[:, 2:], atol=1e-6)
    assert M.project({"model": "OPENCV_FISHEYE", "params": [1, 1, 1, 1, 0, 0, 0, 0]}, P) is None


def test_level_stats_match_the_research_definition():
    # 60 keyframes at one level, a 3x step for 30, back to the level for 10
    y = np.r_[np.zeros(60), np.full(30, math.log(3.0)), np.zeros(10)]
    lv = M.level_stats(y)
    assert lv["coherent_fraction"] == pytest.approx(70 / 100)
    assert lv["longest_off_level_run"] == 30
    assert lv["max_step_15kf_factor"] == pytest.approx(3.0)
    # shift-invariant: the gauge's scale is a constant offset in log
    lv2 = M.level_stats(y + 5.0)
    assert lv2["coherent_fraction"] == lv["coherent_fraction"]
    assert lv2["max_step_15kf_factor"] == pytest.approx(lv["max_step_15kf_factor"])


def test_measure_records_wall_time_and_rss(tmp_path):
    import sys

    rec = M.measure_command([sys.executable, "-c", "import time; x = bytearray(50_000_000); time.sleep(0.6)"],
                            poll_s=0.1)
    assert rec["returncode"] == 0
    assert rec["wall_s"] >= 0.5
    assert rec["peak_rss_mb"] >= 40
    assert rec["samples"] >= 2


def test_time_aware_jumps_separate_a_long_move_from_a_pose_error(scene):
    poses, X, obs = scene
    step = np.median([np.linalg.norm(poses[k + 1][:3, 3] - poses[k][:3, 3]) for k in range(N - 1)])
    moved = [T.copy() for T in poses]
    for k in range(60, N):
        moved[k][:3, 3] += np.array([12 * step, 0, 0])
    times = np.arange(N) * 0.3
    v, ids = make_variant(moved, X, None)
    # a fast jump: 12 steps in 0.3 s -> counted by both rules
    fast = run(v, ids, times=times)["continuity"]["main"]
    assert fast["criterion"] == "index-and-time"
    assert fast["jumps"] == 1 and fast["index_jumps"] == 1
    # the same displacement over a 5 s gap (tracking lost while walking) is a
    # plausible move: the index rule still lists it, the count does not
    slow_t = times.copy()
    slow_t[60:] += 5.0
    slow = run(v, ids, times=slow_t)["continuity"]["main"]
    assert slow["index_jumps"] == 1 and slow["jumps"] == 0
    assert slow["jump_list"][0]["counted"] is False


def test_unstated_components_come_from_covisibility(scene):
    poses, X, obs = scene
    kf, pt, uv, z = obs
    # keyframes >= 100 observe their own copies of the points: no shared track
    pt2 = np.where(kf >= 100, pt + len(X), pt)
    v, ids = make_variant(poses, np.concatenate([X, X]), (kf, pt2, uv, z))
    v.component_stated = False
    out = run(v, ids)
    assert out["components"]["count"] == 2
    assert out["components"]["sizes"] == [100, 60]
    assert out["components"]["definition"].startswith("not stated: covisibility")
    v.component_stated = True  # stated as one frame: taken at its word
    assert run(v, ids)["components"]["count"] == 1
