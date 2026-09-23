"""The coherence viewpoint rule is deterministic and trajectory-derived."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tower.world_builder.coherence_eval import viewpoints as vp


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = math.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _circle_walk(n=60, radius=2.0):
    """A wearer walking a circle, world +Z up, looking outward, head level."""
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
    names = [v["name"] for v in a["views"]]
    assert names[0] == "TOP"
    assert names[1:9] == [f"ORBIT_{k}" for k in range(8)]
    assert len([n for n in names if n.startswith("TRAJ_")]) == 24
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
    assert any(v["requested_capture_index"] == 32 for v in subs)
    s = next(v for v in subs if v["requested_capture_index"] == 32)
    assert s["capture_index"] == 33  # 31, 30 are component 1; 33 is the nearest main


@pytest.mark.parametrize("s,axis,deg,t", [
    (3.7, [0.3, -1.0, 0.4], 71.0, [5.0, -2.0, 11.0]),
    (0.05, [1.0, 0.0, 0.0], 180.0, [0.0, 0.0, 0.0]),
])
def test_traj_choice_invariant_under_sim3(s, axis, deg, t):
    poses, order, comp = _circle_walk()
    a = vp.viewpoint_set(poses, order, comp, INTR)
    R = _rot(axis, deg)
    b = vp.viewpoint_set(_sim3(poses, s, R, np.asarray(t, float)), order, comp, INTR)
    pick = lambda vs: [(v["name"], v["keyframe_id"]) for v in vs["views"]
                       if v["family"].startswith("TRAJ")]
    assert pick(a) == pick(b)
    # in-situ views are the keyframe's own pose, transformed with it
    va, vb = vp.views_by_name(a), vp.views_by_name(b)
    for name in va:
        if va[name]["family"] == "TRAJ_INSITU":
            Ta, Tb = vp.view_T_world_camera(va[name]), vp.view_T_world_camera(vb[name])
            assert np.allclose(Tb[:3, :3], R @ Ta[:3, :3], atol=1e-8)
            assert np.allclose(Tb[:3, 3], s * R @ Ta[:3, 3] + t, atol=1e-6 * max(1, s))


def test_orbit_equivariant_under_translation_and_scale():
    poses, order, comp = _circle_walk()
    a = vp.viewpoint_set(poses, order, comp, INTR)
    b = vp.viewpoint_set(_sim3(poses, 2.5, np.eye(3), np.array([1.0, 2.0, 3.0])),
                         order, comp, INTR)
    va, vb = vp.views_by_name(a), vp.views_by_name(b)
    for k in range(8):
        Ta, Tb = vp.view_T_world_camera(va[f"ORBIT_{k}"]), vp.view_T_world_camera(vb[f"ORBIT_{k}"])
        assert np.allclose(Tb[:3, :3], Ta[:3, :3], atol=1e-8)
        assert np.allclose(Tb[:3, 3], 2.5 * Ta[:3, 3] + [1, 2, 3], atol=1e-6)


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
