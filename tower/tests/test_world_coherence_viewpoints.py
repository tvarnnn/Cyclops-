"""The coherence viewpoint rule (/2) is deterministic, stable and transferable."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tower.world_builder.coherence_eval import layer_renders as lr
from tower.world_builder.coherence_eval import viewpoints as vp


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = math.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def _circle_walk(n=60, radius=2.0):
    """A wearer walking an ellipse, world +Z up, looking outward, head level."""
    poses, order, comp = {}, [], {}
    for i in range(n):
        th = 2 * math.pi * i / n
        C = np.array([radius * math.cos(th), 1.5 * radius * math.sin(th), 1.6])
        fwd = np.array([math.cos(th), math.sin(th), -0.2])
        R = vp._look_at(C, C + fwd, [0, 0, 1])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, C
        kid = f"s:{i:08d}"
        order.append(kid)
        if i % 17 == 5:          # an unposed keyframe
            continue
        poses[kid] = T
        comp[kid] = 1 if i in (30, 31, 32) else 0   # a small second component
    return poses, order, comp


def _room_walk(n=240, seed=0):
    """A walk shaped like the target's: an elongated, wobbly path, horizontal
    PCA ratio ~0.40 (target 0.38-0.49 over its reruns), the wearer looking
    mostly one way (view resultant ~0.77; target ~0.66)."""
    rng = np.random.default_rng(seed)
    poses, order, obs = {}, [], {}
    for i in range(n):
        s = i / (n - 1)
        x = 6.0 * s + 0.4 * math.sin(9 * s)
        y = 1.7 * math.sin(4 * math.pi * s) + 0.3 * math.cos(13 * s)
        C = np.array([x, y, 1.6 + 0.05 * math.sin(7 * s)])
        yaw = 0.9 * math.sin(5 * s) + 0.3 * rng.standard_normal()
        fwd = np.array([math.cos(yaw), math.sin(yaw), -0.25])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = vp._look_at(C, C + fwd, [0, 0, 1]), C
        kid = f"w:{i:08d}"
        order.append(kid)
        poses[kid] = T
        obs[kid] = 200
    return poses, order, obs


INTR = {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 320.0, "width": 360, "height": 640}


def _sim3(poses, s, R, t):
    out = {}
    for k, T in poses.items():
        T2 = np.eye(4)
        T2[:3, :3] = R @ T[:3, :3]
        T2[:3, 3] = s * (R @ T[:3, 3]) + t
        out[k] = T2
    return out


def test_deterministic_and_json_roundtrip(tmp_path):
    poses, order, comp = _circle_walk()
    a = vp.viewpoint_set(poses, order, comp, INTR)
    b = vp.viewpoint_set(dict(reversed(list(poses.items()))), order, dict(comp), INTR)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["rule"] == "wb-coherence-viewpoints/2" and a["mode"] == "native"
    names = [v["name"] for v in a["views"]]
    assert names[0] == "TOP"
    assert names[1:9] == [f"ORBIT_{k}" for k in range(8)]
    assert len([n for n in names if n.startswith("TRAJ_")]) == 24
    assert set(a["anchors"]) == set(a["main_keyframe_ids"])
    p = vp.save_viewpoints(a, tmp_path / "v.json")
    assert vp.load_viewpoints(p) == json.loads(json.dumps(a))
    assert a["frame"]["component"] == 0
    # up recovered from the cameras, to within the bias a downward pitch on
    # an unevenly sampled loop leaves in a mean of camera up vectors
    assert float(np.dot(a["frame"]["up"], [0, 0, 1])) > math.cos(math.radians(3.0))


def test_traj_choice_substitutes_unposed_and_minor_component():
    poses, order, comp = _circle_walk()
    vs = vp.viewpoint_set(poses, order, comp, INTR)
    main = set(vs["main_keyframe_ids"])
    for v in vs["views"]:
        if v["family"].startswith("TRAJ"):
            assert v["keyframe_id"] in main
            idx = min(len(order) - 1, int(math.floor(v["quantile"] * len(order))))
            assert v["requested_capture_index"] == idx
            assert v["substituted"] == (order[idx] not in main)
    subs = [v for v in vs["views"] if v["family"] == "TRAJ_INSITU" and v["substituted"]]
    # capture index 32 is component 1 -> substituted (quantile 6.5/12 of 60 = 32)
    s = next(v for v in subs if v["requested_capture_index"] == 32)
    assert s["capture_index"] == 33  # 31, 30 are component 1; 33 is the nearest main


@pytest.mark.parametrize("s,axis,deg,t", [
    (3.7, [0.3, -1.0, 0.4], 71.0, [5.0, -2.0, 11.0]),
    (0.05, [1.0, 0.0, 0.0], 180.0, [0.0, 0.0, 0.0]),
])
def test_rule_is_sim3_equivariant(s, axis, deg, t):
    """TRAJ choice invariant; every view (ORBIT, TOP, TRAJ) maps with the Sim(3)."""
    poses, order, comp = _circle_walk()
    a = vp.viewpoint_set(poses, order, comp, INTR)
    R = _rot(axis, deg)
    t = np.asarray(t, float)
    b = vp.viewpoint_set(_sim3(poses, s, R, t), order, comp, INTR)
    pick = lambda vs: [(v["name"], v["keyframe_id"]) for v in vs["views"]
                       if v["family"].startswith("TRAJ")]
    assert pick(a) == pick(b)
    va, vb = vp.views_by_name(a), vp.views_by_name(b)
    for name in va:
        Ta, Tb = vp.view_T_world_camera(va[name]), vp.view_T_world_camera(vb[name])
        assert np.allclose(Tb[:3, :3], R @ Ta[:3, :3], atol=1e-6), name
        assert np.allclose(Tb[:3, 3], s * R @ Ta[:3, 3] + t, atol=1e-6 * max(1, s * 10)), name


def test_unsupported_cameras_take_no_part():
    """A zero-observation camera far from the walk (rule /1's worst swing)
    changes nothing when observation counts are given, and is never a TRAJ pick."""
    poses, order, obs = _room_walk()
    base = vp.viewpoint_set(poses, order, None, INTR, observations=obs)
    wild = dict(poses)
    T = np.eye(4)
    T[:3, 3] = [300.0, -200.0, 40.0]
    wild["w:99999999"] = T
    order2 = order[:120] + ["w:99999999"] + order[120:]
    obs2 = dict(obs, **{"w:99999999": 0})
    got = vp.viewpoint_set(wild, order2, None, INTR, observations=obs2)
    assert got["frame"]["supported_total"] == base["frame"]["supported_total"]
    for key in ("center", "up", "axis1", "radius"):
        assert np.allclose(got["frame"][key], base["frame"][key], atol=1e-9), key
    assert all(v.get("keyframe_id") != "w:99999999" for v in got["views"])


def test_frame_stable_under_noise_and_dropped_cameras():
    """3% (of r) centre noise, 2 deg orientation noise and 5% of cameras
    dropped must not flip or swing the frame."""
    poses, order, obs = _room_walk()
    ref = vp.viewpoint_set(poses, order, None, INTR, observations=obs)
    assert ref["frame"]["conditioning"]["stable"]
    assert 0.3 < ref["frame"]["conditioning"]["pca_ratio"] < 0.5  # as weak as the target's
    r = ref["frame"]["radius"]
    V0 = vp.views_by_name(ref)
    worst_e1 = worst_eye = 0.0
    for seed in range(25):
        rng = np.random.default_rng(100 + seed)
        noisy = {}
        for k, T in poses.items():
            if rng.random() < 0.05:
                continue
            T2 = T.copy()
            T2[:3, 3] = T[:3, 3] + rng.normal(0, 0.03 * r / math.sqrt(3), 3)
            T2[:3, :3] = _rot(rng.normal(size=3), rng.normal(0, 2.0)) @ T[:3, :3]
            noisy[k] = T2
        got = vp.viewpoint_set(noisy, order, None, INTR,
                               observations={k: obs[k] for k in noisy})
        worst_e1 = max(worst_e1, _angle(got["frame"]["axis1"], ref["frame"]["axis1"]))
        V1 = vp.views_by_name(got)
        for k in range(8):
            d = np.linalg.norm(vp.view_T_world_camera(V1[f"ORBIT_{k}"])[:3, 3]
                               - vp.view_T_world_camera(V0[f"ORBIT_{k}"])[:3, 3]) / r
            worst_eye = max(worst_eye, d)
    assert worst_e1 < 8.0, worst_e1        # measured 2.8 deg; a flip would be ~180
    assert worst_eye < 0.25, worst_eye     # measured 0.10 r


def test_transfer_carries_one_set_across_a_variant():
    poses, order, obs = _room_walk()
    ref = vp.viewpoint_set(poses, order, None, INTR, observations=obs)
    s, R, t = 2.7, _rot([0.2, 1.0, -0.4], 57.0), np.array([3.0, -1.0, 8.0])
    rng = np.random.default_rng(7)
    variant = _sim3(poses, s, R, t)
    kids = sorted(variant)
    for k in kids[::10]:  # 10% of cameras grossly wrong: the fit must trim them
        variant[k] = variant[k].copy()
        variant[k][:3, 3] += rng.normal(0, 5.0 * s, 3)
    del variant[ref["views"][9]["keyframe_id"]]  # TRAJ_00 keyframe unposed in the variant
    out = vp.transfer_viewpoints(ref, None, variant, observations_to={k: 200 for k in variant})
    tr = out["transfer"]
    assert out["mode"] == "transferred" and abs(tr["scale"] - s) / s < 1e-6
    assert tr["inliers"] < tr["shared"]
    Vr, Vo = vp.views_by_name(ref), vp.views_by_name(out)
    for name, v in Vr.items():
        T0, T1 = vp.view_T_world_camera(v), vp.view_T_world_camera(Vo[name])
        if Vo[name]["family"] == "TRAJ_INSITU" and Vo[name]["pose_source"] == "own":
            assert np.allclose(T1, variant[Vo[name]["keyframe_id"]])
            continue
        assert np.allclose(T1[:3, :3], R @ T0[:3, :3], atol=1e-6), name
        assert np.allclose(T1[:3, 3], s * R @ T0[:3, 3] + t, atol=1e-5), name
    assert Vo["TRAJ_00_insitu"]["pose_source"] == "transferred"
    assert tr["traj_keyframes_missing_in_target"] == [Vo["TRAJ_00_insitu"]["keyframe_id"]]
    assert np.isclose(Vo["TOP"]["half_width"], s * Vr["TOP"]["half_width"])
    # anchors are re-expressed in the target gauge, so transfers chain
    back = vp.transfer_viewpoints(out, None, poses, observations_to=obs)
    assert abs(back["transfer"]["scale"] - 1 / s) < 1e-6
    assert np.allclose(vp.view_T_world_camera(vp.views_by_name(back)["ORBIT_3"]),
                       vp.view_T_world_camera(Vr["ORBIT_3"]), atol=1e-5)


def test_transfer_refuses_too_few_shared():
    poses, order, obs = _room_walk()
    ref = vp.viewpoint_set(poses, order, None, INTR, observations=obs)
    few = {k: poses[k] for k in list(poses)[:4]}
    with pytest.raises(ValueError, match="shared"):
        vp.transfer_viewpoints(ref, None, few)


def test_project_centre_lands_mid_image():
    poses, order, comp = _circle_walk()
    vs = vp.viewpoint_set(poses, order, comp, INTR)
    c = np.asarray(vs["frame"]["center"])
    for v in vs["views"]:
        if v["family"] in ("ORBIT", "TOP"):
            uv, z = vp.project(v, c[None])
            assert z[0] > 0
            if v["family"] == "ORBIT":
                assert np.allclose(uv[0], [v["width"] / 2, v["height"] / 2], atol=1e-6)


def test_palettes_never_wrap_and_any_region_label_has_a_colour():
    cols = [lr.component_rgb(i) for i in range(64)]
    assert len(set(cols)) == 64
    assert lr.region_rgb("desk") == lr.REGION_RGB["desk"]
    assert lr.region_rgb("kitchen") == lr.region_rgb("kitchen")
    assert lr.region_rgb("kitchen") != lr.region_rgb("garage")
    assert lr.region_rgb("kitchen") != lr.REGION_RGB[None]


def test_cameras_render_main_component_only_by_default():
    poses, order, comp = _circle_walk()
    vs = vp.viewpoint_set(poses, order, comp, INTR)
    w = lr.WorldLayers(world_dir=None, session_id="s", solution={}, poses=poses, component_of=comp,
                       capture_order=order, keyframe_ids=order, xyz=np.zeros((0, 3)),
                       point_component=np.zeros(0, np.int32), point_first_kid=[],
                       observations=np.zeros((0, 3), np.int64), camera=INTR, session={},
                       surface_manifest=None)
    top = vp.views_by_name(vs)["TOP"]
    minor = np.array(lr.component_rgb(1), np.uint8)
    img = lr.render_cameras(top, w, vs["frame"]["radius"])
    assert not (img == minor).all(-1).any()
    img2 = lr.render_cameras(top, w, vs["frame"]["radius"], include_minor=True)
    assert (img2 == minor).all(-1).any()
