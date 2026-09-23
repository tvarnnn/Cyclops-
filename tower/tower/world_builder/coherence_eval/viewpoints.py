"""A deterministic, trajectory-derived viewpoint set for coherence renders.

Every forensic render of a world -- cameras, sparse points, the fused
surface, the appearance -- is taken from the SAME set of viewpoints, and that
set is derived from the walk alone by a fixed rule. Nobody picks a flattering
angle, and a variant of a world (a re-solve, a re-fusion) gets its viewpoints
from the same rule applied to its own trajectory, so two renders of the same
view name are directly comparable.

The rule (`RULE_VERSION`)
-------------------------
Inputs: posed keyframes as camera-to-world transforms ``T_world_camera``
(4x4, OpenCV camera axes: x right, y DOWN, z forward), the capture order of
ALL accepted keyframes, and optionally each posed keyframe's solver
component.

* Frame. Only the posed keyframes of the LARGEST component are used (ties:
  the lowest component id). Let C_i be their camera centres.
  - ``center`` c = coordinate-wise median of C_i. (Coordinate-wise, so it is
    equivariant under translation and scale but only approximately under
    rotation; the TRAJ keyframe choice does not depend on it at all.)
  - ``up`` u = trimmed mean of the camera up vectors, where a camera's up is
    the -Y column of R_world_camera (OpenCV y points down). This is the
    camera-only seed of ``surface_render._camera_up``; the surface stage then
    refines it on the mesh (``surface_render.surface_up``), which we
    deliberately do NOT do, because the viewpoints must not depend on the
    layer under evaluation. Trim: compute the plain mean, drop the ups whose
    angle to it is above the 90th percentile, and re-average.
  - ``radius`` r = 90th percentile of |C_i - c|.
  - ``axis`` e1 = the principal axis of the C_i projected onto the plane
    perpendicular to u (PCA of the centred, projected centres), sign fixed
    so that e1 . (C_first - c) >= 0 where C_first is the first posed
    main-component keyframe in capture order. e2 = u x e1, so (e1, e2, u)
    is right-handed.
* TOP. Orthographic, looking along -u (camera z = -u, camera x = e1). It
  frames the 2nd..98th percentile box of the C_i in (e1, e2), each side
  padded by 50% of that axis's extent (a box twice as wide and twice as deep
  as the percentile box, same centre), with a floor of 0.5 r on each
  half-extent so a straight-line walk does not produce a sliver. Camera
  placed at c + 4 r u. Rendered with back faces culled ("dollhouse"), so a
  ceiling -- whose normal faces down, toward the cameras that saw it -- does
  not hide the room.
* ORBIT_k, k = 0..7. Perspective, vertical FOV 60 deg, eye at
  c + 2 r (cos 35deg (cos a e1 + sin a e2) + sin 35deg u), a = 45deg k,
  looking at c with image-up along u. Back faces culled, as for TOP.
* TRAJ_jj, j = 0..11. Keyframe at capture-order index
  floor((j + 0.5) / 12 * N) among ALL N accepted keyframes. If that keyframe
  is unposed or outside the main component, the nearest posed
  main-component keyframe by capture-order index is used instead (ties go to
  the EARLIER keyframe) and the substitution is recorded. Two views each:
  - ``TRAJ_jj_insitu``: the keyframe's own pose, the given intrinsics and
    image size. No culling.
  - ``TRAJ_jj_pullback``: the same orientation, the centre moved 0.75 r
    backwards along the viewing axis (-z) and 0.25 r along u, vertical FOV
    75 deg, the in-situ aspect ratio. No culling.

The TRAJ keyframe choice depends only on the capture order and on which
keyframes are posed in which component, so it is invariant under any global
Sim(3) of the poses and is keyed by keyframe id -- the handle for comparing
the same view across variants.

API
---
``viewpoint_set(poses, capture_order, component_of, intrinsics) -> dict``
    The whole set, JSON-serialisable.
``viewpoints_for_world(world_dir, session_id=None, *, source="solve") -> dict``
    Convenience: reads ``solve/<sid>/solution.json`` (GLOMAP global solve;
    the poses the surface/appearance stages use) and
    ``sessions/<sid>/keyframes.jsonl`` of a world directory (or a variant
    directory of the same shape) and returns ``viewpoint_set(...)``.
``save_viewpoints(vs, path)`` / ``load_viewpoints(path)``
``view_T_world_camera(view) -> (4, 4) ndarray``;
``view_K(view) -> (3, 3) ndarray`` (perspective views only);
``project(view, X_world) -> (uv (N, 2), depth (N,))`` for either projection.
``poses_from_solution(meta) -> (poses, component_of)`` converts the
    solution's R_cw / t_cw to T_world_camera.

Every view dict carries: ``name``, ``family`` (TOP / ORBIT / TRAJ_INSITU /
TRAJ_PULLBACK), ``projection`` ("orthographic" | "perspective"),
``T_world_camera`` (4x4 list), ``width``, ``height``, ``cull_backfaces``;
perspective views add ``fx, fy, cx, cy, fov_y_deg``; orthographic views add
``half_width``, ``half_height`` (world units at the image edges); TRAJ views
add ``keyframe_id``, ``requested_keyframe_id``, ``capture_index``,
``requested_capture_index``, ``quantile``, ``substituted``,
``substitution_reason``.

CLI::

    python -m tower.world_builder.coherence_eval.viewpoints WORLD_DIR \
        [--session SID] --out VIEWPOINTS.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

RULE_VERSION = "wb-coherence-viewpoints/1"
SCHEMA = "glasses.wb.coherence.viewpoints/1"

N_ORBIT = 8
ORBIT_ELEVATION_DEG = 35.0
ORBIT_DISTANCE_R = 2.0
ORBIT_FOV_Y_DEG = 60.0
N_TRAJ = 12
PULLBACK_BACK_R = 0.75
PULLBACK_UP_R = 0.25
PULLBACK_FOV_Y_DEG = 75.0
TOP_PERCENTILES = (2.0, 98.0)
TOP_MARGIN = 0.5
TOP_MIN_HALF_R = 0.5
UP_TRIM_PERCENTILE = 90.0
RADIUS_PERCENTILE = 90.0

DEFAULT_ORBIT_SIZE = (1024, 768)
DEFAULT_TOP_LONG_SIDE = 1024
DEFAULT_PULLBACK_SCALE = 1.5


# ---------------------------------------------------------------------------
# small geometry helpers
# ---------------------------------------------------------------------------


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    if not n > 1e-12:
        raise ValueError("degenerate direction")
    return v / n


def _T(R_wc: np.ndarray, C: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_wc
    T[:3, 3] = C
    return T


def _look_at(eye, target, up) -> np.ndarray:
    """Camera-to-world rotation (OpenCV axes) looking from eye to target."""
    z = _unit(np.asarray(target, float) - np.asarray(eye, float))
    x = np.cross(-np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-9:  # looking along up: pick any horizontal x
        x = np.cross(z, [1.0, 0.0, 0.0])
        if np.linalg.norm(x) < 1e-9:
            x = np.cross(z, [0.0, 1.0, 0.0])
    x = _unit(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _list(a) -> list:
    return np.asarray(a, np.float64).round(9).tolist()


def _f_from_fov(size_px: int, fov_deg: float) -> float:
    return 0.5 * size_px / math.tan(math.radians(fov_deg) / 2.0)


# ---------------------------------------------------------------------------
# the frame
# ---------------------------------------------------------------------------


def main_component(posed: Iterable[str], component_of: dict | None) -> tuple[int | None, list[str]]:
    """(component id, posed keyframe ids in it). None when no components."""
    posed = list(posed)
    if not component_of:
        return None, posed
    counts: dict[int, int] = {}
    for k in posed:
        c = component_of.get(k)
        if c is not None:
            counts[int(c)] = counts.get(int(c), 0) + 1
    if not counts:
        return None, posed
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best, [k for k in posed if component_of.get(k) is not None
                  and int(component_of[k]) == best]


def robust_up(R_wc_list: list[np.ndarray]) -> np.ndarray:
    """Trimmed mean of the cameras' -Y axes (see module docstring)."""
    ups = np.array([np.asarray(R, float)[:, 1] * -1.0 for R in R_wc_list])
    m = _unit(ups.mean(axis=0))
    ang = np.degrees(np.arccos(np.clip(ups @ m, -1.0, 1.0)))
    keep = ang <= np.percentile(ang, UP_TRIM_PERCENTILE)
    if keep.sum() >= 3:
        m = _unit(ups[keep].mean(axis=0))
    return m


def walk_frame(poses: dict, capture_order: list[str], component_of: dict | None) -> dict:
    """The (c, u, r, e1, e2) frame of the rule, plus bookkeeping."""
    order_index = {k: i for i, k in enumerate(capture_order)}
    posed = [k for k in capture_order if k in poses]
    # Posed keyframes missing from capture_order go last, sorted by id.
    posed += sorted(k for k in poses if k not in order_index)
    comp, main = main_component(posed, component_of)
    if len(main) < 2:
        raise ValueError(f"need >= 2 posed keyframes in the main component, have {len(main)}")
    T = np.array([np.asarray(poses[k], float) for k in main])
    C = T[:, :3, 3]
    c = np.median(C, axis=0)
    u = robust_up([t[:3, :3] for t in T])
    d = np.linalg.norm(C - c, axis=1)
    r = float(np.percentile(d, RADIUS_PERCENTILE))
    if not r > 1e-12:
        r = float(d.max()) if d.max() > 1e-12 else 1.0
    # principal horizontal axis
    P = C - c
    P = P - np.outer(P @ u, u)
    P0 = P - P.mean(axis=0)
    cov = P0.T @ P0
    w, V = np.linalg.eigh(cov)
    e1 = V[:, int(np.argmax(w))]
    e1 = e1 - (e1 @ u) * u
    if np.linalg.norm(e1) < 1e-9:  # all centres on the up axis
        e1 = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.cross(u, [0.0, 1.0, 0.0])
    e1 = _unit(e1)
    first = main[0]
    if float(e1 @ (np.asarray(poses[first], float)[:3, 3] - c)) < 0:
        e1 = -e1
    e2 = np.cross(u, e1)
    return {
        "component": comp,
        "main_keyframes": main,
        "posed_total": len(posed),
        "center": c, "up": u, "radius": r, "axis1": e1, "axis2": e2,
        "first_keyframe_id": first,
        "centres": C,
        "order_index": order_index,
    }


# ---------------------------------------------------------------------------
# the views
# ---------------------------------------------------------------------------


def _perspective(name, family, T, width, height, fov_y_deg=None, *, fx=None, fy=None,
                 cx=None, cy=None, cull=False, **extra) -> dict:
    if fx is None:
        fy = fx = _f_from_fov(height, fov_y_deg)
        cx, cy = width / 2.0, height / 2.0
    if fov_y_deg is None:
        fov_y_deg = math.degrees(2.0 * math.atan(0.5 * height / fy))
    view = {"name": name, "family": family, "projection": "perspective",
            "T_world_camera": _list(T), "width": int(width), "height": int(height),
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "fov_y_deg": float(fov_y_deg), "cull_backfaces": bool(cull)}
    view.update(extra)
    return view


def _traj_choice(capture_order, main_set, j) -> dict:
    n = len(capture_order)
    q = (j + 0.5) / N_TRAJ
    idx = min(n - 1, int(math.floor(q * n)))
    req = capture_order[idx]
    if req in main_set:
        return {"quantile": q, "requested_capture_index": idx, "requested_keyframe_id": req,
                "capture_index": idx, "keyframe_id": req, "substituted": False,
                "substitution_reason": None}
    best = None
    for off in range(1, n):
        for cand in (idx - off, idx + off):  # earlier first: ties go early
            if 0 <= cand < n and capture_order[cand] in main_set:
                best = cand
                break
        if best is not None:
            break
    if best is None:
        raise ValueError("no posed main-component keyframe in capture order")
    return {"quantile": q, "requested_capture_index": idx, "requested_keyframe_id": req,
            "capture_index": best, "keyframe_id": capture_order[best], "substituted": True,
            "substitution_reason": "unposed-or-outside-main-component"}


def viewpoint_set(poses: dict, capture_order: list[str], component_of: dict | None,
                  intrinsics: dict, *, orbit_size=DEFAULT_ORBIT_SIZE,
                  top_long_side: int = DEFAULT_TOP_LONG_SIDE,
                  pullback_scale: float = DEFAULT_PULLBACK_SCALE,
                  meta: dict | None = None) -> dict:
    """The full viewpoint set of the rule in the module docstring.

    poses: keyframe id -> (4, 4) T_world_camera (camera-to-world, OpenCV axes).
    capture_order: ALL accepted keyframe ids in capture order (posed or not).
    component_of: keyframe id -> solver component, or None (one component).
    intrinsics: {fx, fy, cx, cy, width, height} of the in-situ camera -- the
        pinhole the poses are expressed in.
    Returns a JSON-serialisable dict (see module docstring for the fields).
    """
    for key in ("fx", "fy", "cx", "cy", "width", "height"):
        if key not in intrinsics:
            raise ValueError(f"intrinsics missing {key!r}")
    capture_order = list(capture_order)
    fr = walk_frame(poses, capture_order, component_of)
    c, u, r, e1, e2 = fr["center"], fr["up"], fr["radius"], fr["axis1"], fr["axis2"]
    views = []

    # TOP -------------------------------------------------------------------
    P = fr["centres"] - c
    a1, a2 = P @ e1, P @ e2
    lo1, hi1 = np.percentile(a1, TOP_PERCENTILES)
    lo2, hi2 = np.percentile(a2, TOP_PERCENTILES)
    mid1, mid2 = 0.5 * (lo1 + hi1), 0.5 * (lo2 + hi2)
    hw = max(0.5 * (hi1 - lo1) + TOP_MARGIN * (hi1 - lo1), TOP_MIN_HALF_R * r)
    hh = max(0.5 * (hi2 - lo2) + TOP_MARGIN * (hi2 - lo2), TOP_MIN_HALF_R * r)
    # image x = e1, image y = -e2 (camera y = z x x = (-u) x e1 = -e2)
    if hw >= hh:
        W = int(top_long_side)
        H = max(2, int(round(top_long_side * hh / hw)))
    else:
        H = int(top_long_side)
        W = max(2, int(round(top_long_side * hw / hh)))
    Rtop = np.stack([e1, np.cross(-u, e1), -u], axis=1)
    top_center = c + mid1 * e1 + mid2 * e2
    views.append({
        "name": "TOP", "family": "TOP", "projection": "orthographic",
        "T_world_camera": _list(_T(Rtop, top_center + 4.0 * r * u)),
        "width": W, "height": H, "half_width": float(hw), "half_height": float(hh),
        "cull_backfaces": True,
    })

    # ORBIT -----------------------------------------------------------------
    el = math.radians(ORBIT_ELEVATION_DEG)
    for k in range(N_ORBIT):
        az = math.radians(45.0 * k)
        d = math.cos(el) * (math.cos(az) * e1 + math.sin(az) * e2) + math.sin(el) * u
        eye = c + ORBIT_DISTANCE_R * r * d
        views.append(_perspective(
            f"ORBIT_{k}", "ORBIT", _T(_look_at(eye, c, u), eye),
            orbit_size[0], orbit_size[1], ORBIT_FOV_Y_DEG, cull=True,
            azimuth_deg=45.0 * k, elevation_deg=ORBIT_ELEVATION_DEG))

    # TRAJ ------------------------------------------------------------------
    main_set = set(fr["main_keyframes"])
    iw, ih = int(intrinsics["width"]), int(intrinsics["height"])
    pw, ph = int(round(iw * pullback_scale)), int(round(ih * pullback_scale))
    for j in range(N_TRAJ):
        ch = _traj_choice(capture_order, main_set, j)
        T = np.asarray(poses[ch["keyframe_id"]], float)
        R, C = T[:3, :3], T[:3, 3]
        views.append(_perspective(
            f"TRAJ_{j:02d}_insitu", "TRAJ_INSITU", T, iw, ih,
            fx=float(intrinsics["fx"]), fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]), cy=float(intrinsics["cy"]), cull=False, **ch))
        Cp = C - PULLBACK_BACK_R * r * R[:, 2] + PULLBACK_UP_R * r * u
        views.append(_perspective(
            f"TRAJ_{j:02d}_pullback", "TRAJ_PULLBACK", _T(R, Cp), pw, ph,
            PULLBACK_FOV_Y_DEG, cull=False, **ch))

    return {
        "schema": SCHEMA,
        "rule": RULE_VERSION,
        "rule_parameters": {
            "orbit": {"n": N_ORBIT, "elevation_deg": ORBIT_ELEVATION_DEG,
                      "distance_r": ORBIT_DISTANCE_R, "fov_y_deg": ORBIT_FOV_Y_DEG},
            "traj": {"n": N_TRAJ, "pullback_back_r": PULLBACK_BACK_R,
                     "pullback_up_r": PULLBACK_UP_R, "pullback_fov_y_deg": PULLBACK_FOV_Y_DEG,
                     "pullback_scale": pullback_scale},
            "top": {"percentiles": list(TOP_PERCENTILES), "margin": TOP_MARGIN,
                    "min_half_r": TOP_MIN_HALF_R, "camera_height_r": 4.0},
            "up": f"trimmed mean of camera -Y (keep angle <= p{UP_TRIM_PERCENTILE:g})",
            "radius": f"p{RADIUS_PERCENTILE:g} of |C_i - c|",
            "center": "coordinate-wise median of main-component camera centres",
        },
        "frame": {
            "component": fr["component"],
            "posed_in_component": len(fr["main_keyframes"]),
            "posed_total": fr["posed_total"],
            "accepted_total": len(capture_order),
            "center": _list(c), "up": _list(u), "radius": float(r),
            "axis1": _list(e1), "axis2": _list(e2),
            "first_keyframe_id": fr["first_keyframe_id"],
        },
        "intrinsics": {k: (float(intrinsics[k]) if k in ("fx", "fy", "cx", "cy")
                           else int(intrinsics[k]))
                       for k in ("fx", "fy", "cx", "cy", "width", "height")},
        "main_keyframe_ids": list(fr["main_keyframes"]),
        "views": views,
        "meta": dict(meta or {}),
    }


# ---------------------------------------------------------------------------
# helpers for renderers
# ---------------------------------------------------------------------------


def view_T_world_camera(view: dict) -> np.ndarray:
    return np.asarray(view["T_world_camera"], np.float64).reshape(4, 4)


def view_K(view: dict) -> np.ndarray:
    if view["projection"] != "perspective":
        raise ValueError(f"{view['name']} is {view['projection']}")
    return np.array([[view["fx"], 0, view["cx"]], [0, view["fy"], view["cy"]], [0, 0, 1.0]])


def project(view: dict, X_world) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (N, 2) and camera-z depth (N,) of world points.

    Perspective: the pinhole of the view (points behind the camera get
    depth <= 0; callers must drop them). Orthographic: x,y scaled so that
    +-half_width / +-half_height reach the image edges; depth is camera z.
    """
    X = np.asarray(X_world, np.float64).reshape(-1, 3)
    T = view_T_world_camera(view)
    R, C = T[:3, :3], T[:3, 3]
    Xc = (X - C) @ R  # == R^T (X - C)
    z = Xc[:, 2]
    W, H = view["width"], view["height"]
    if view["projection"] == "orthographic":
        uu = (Xc[:, 0] / view["half_width"] * 0.5 + 0.5) * W
        vv = (Xc[:, 1] / view["half_height"] * 0.5 + 0.5) * H
        return np.stack([uu, vv], axis=1), z
    with np.errstate(divide="ignore", invalid="ignore"):
        uu = view["fx"] * Xc[:, 0] / z + view["cx"]
        vv = view["fy"] * Xc[:, 1] / z + view["cy"]
    return np.stack([uu, vv], axis=1), z


def save_viewpoints(vs: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(vs, indent=1, sort_keys=False), encoding="utf-8")
    tmp.replace(path)
    return path


def load_viewpoints(path) -> dict:
    vs = json.loads(Path(path).read_text(encoding="utf-8"))
    if vs.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} file (schema={vs.get('schema')!r})")
    return vs


def views_by_name(vs: dict) -> dict:
    return {v["name"]: v for v in vs["views"]}


# ---------------------------------------------------------------------------
# reading a world directory (read-only)
# ---------------------------------------------------------------------------


def poses_from_solution(meta: dict) -> tuple[dict, dict]:
    """solution.json `poses` (R_cw row-major, t_cw) -> (T_world_camera, component)."""
    poses, comp = {}, {}
    for kid, p in (meta.get("poses") or {}).items():
        R_cw = np.asarray(p["rotation"], np.float64).reshape(3, 3)
        t_cw = np.asarray(p["translation"], np.float64).reshape(3)
        R_wc = R_cw.T
        poses[kid] = _T(R_wc, -R_wc @ t_cw)
        if p.get("component") is not None:
            comp[kid] = int(p["component"])
    return poses, comp


def capture_order_from_keyframes(path) -> list[str]:
    """Accepted keyframe ids in capture order (source_seq, then file order)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append((d.get("source_seq", i), i, d["keyframe_id"]))
    rows.sort()
    seen, out = set(), []
    for _s, _i, k in rows:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def single_session_id(world_dir) -> str:
    sessions = sorted(p.name for p in (Path(world_dir) / "sessions").iterdir() if p.is_dir())
    if len(sessions) != 1:
        raise ValueError(f"{world_dir}: expected one session, found {sessions}; pass session_id")
    return sessions[0]


def viewpoints_for_world(world_dir, session_id: str | None = None, *,
                         source: str = "solve", **kw) -> dict:
    """`viewpoint_set` for a world (or variant) directory, read-only.

    source="solve": solve/<sid>/solution.json (the global solve). The
    in-situ camera is the solution's pinhole camera -- the undistorted
    camera the poses, the depth and the fusion are all expressed in -- and
    the session's self-calibrated (distorted) intrinsics are recorded in
    `meta` for reference.
    """
    world_dir = Path(world_dir)
    sid = session_id or single_session_id(world_dir)
    if source != "solve":
        raise ValueError(f"unknown source {source!r}")
    meta = json.loads((world_dir / "solve" / sid / "solution.json").read_text(encoding="utf-8"))
    poses, comp = poses_from_solution(meta)
    order = capture_order_from_keyframes(world_dir / "sessions" / sid / "keyframes.jsonl")
    cam = dict(meta["camera"])
    session_intr = None
    sj = world_dir / "sessions" / sid / "session.json"
    if sj.exists():
        session_intr = json.loads(sj.read_text(encoding="utf-8")).get("intrinsics")
    return viewpoint_set(poses, order, comp, cam, meta={
        "world_dir": str(world_dir), "world_id": world_dir.name, "session_id": sid,
        "pose_source": f"solve/{sid}/solution.json",
        "solver": meta.get("solver"), "input_digest": meta.get("input_digest"),
        "session_intrinsics": session_intr,
        "insitu_camera": "solution.camera (undistorted pinhole of the solve)",
    }, **kw)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("world_dir")
    ap.add_argument("--session", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    vs = viewpoints_for_world(a.world_dir, a.session)
    save_viewpoints(vs, a.out)
    fr = vs["frame"]
    print(f"{len(vs['views'])} views; component {fr['component']} "
          f"({fr['posed_in_component']} posed of {fr['accepted_total']}); r={fr['radius']:.4g}")
    for v in vs["views"]:
        if v["family"] == "TRAJ_INSITU":
            print(f"  {v['name']}: {v['keyframe_id']}"
                  + (f" (for {v['requested_keyframe_id']})" if v["substituted"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
