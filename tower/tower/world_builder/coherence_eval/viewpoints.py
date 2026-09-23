"""A deterministic, trajectory-derived viewpoint set for coherence renders.

Every forensic render of a world -- cameras, sparse points, the fused
surface, the appearance -- is taken from the SAME set of viewpoints, and that
set is derived from the walk alone by a fixed rule. Nobody picks a flattering
angle.

TWO WAYS TO GET A VIEWPOINT SET FOR A VARIANT
---------------------------------------------
* **Transferred (recommended for A/B).** Derive the set ONCE, on the
  reference (usually the frozen world), and carry it onto each variant of
  the same world with `transfer_viewpoints`: a robust Sim(3) fitted on the
  camera centres of shared, supported keyframes maps every view into the
  variant's gauge. ORBIT_3 of the reference and ORBIT_3 of the variant are
  then the same physical viewpoint up to the variant's own error, which is
  what an A/B render needs.
* **Native.** Apply the rule to the variant's own poses
  (`viewpoint_set` / `viewpoints_for_world`). Two natively derived sets agree
  only as far as the frame (c, u, r, e1) is stable; the set records the
  frame's conditioning (`frame.conditioning`) so a reader can tell. Measured
  on C1's three GLOMAP reruns of the target (Sim(3) rms/r 0.03-0.05): e1
  after alignment moved 6.2 / 6.0 / 1.4 deg under rule /2 versus
  122 / 14 / 82 deg under rule /1.

The rule (`RULE_VERSION` = wb-coherence-viewpoints/2)
-----------------------------------------------------
Inputs: posed keyframes as camera-to-world transforms ``T_world_camera``
(4x4, OpenCV camera axes: x right, y DOWN, z forward), the capture order of
ALL accepted keyframes, optionally each posed keyframe's solver component,
and optionally each posed keyframe's observation count.

* Supported cameras. When observation counts are given, a keyframe with fewer
  than ``min_observations`` (default 30 = `global_solve.MIN_IMAGE_OBSERVATIONS`,
  the product's publish floor) is UNSUPPORTED and takes no part in anything:
  not the frame, not the TRAJ choice. Without counts the caller vouches that
  every pose given is supported (e.g. the harness passes published poses).
  Unsupported cameras were what drove rule /1's worst swings (a single
  zero-observation camera far from the walk moves an unweighted PCA).
* Main component: the component with the most supported keyframes (ties: the
  lowest id). Let C_i be the centres of its supported keyframes, in capture
  order.
* ``center`` c = geometric median of C_i (Weiszfeld; Sim(3)-equivariant,
  unlike rule /1's coordinate-wise median).
* ``up`` u = trimmed mean of the camera up vectors (-Y column of R_wc): plain
  mean, drop ups above the 90th-percentile angle to it, re-average. The
  camera-only seed of ``surface_render._camera_up``, deliberately NOT refined
  on the mesh: viewpoints must not depend on the layer under evaluation.
* ``radius`` r = 90th percentile of |C_i - c|.
* ``axis`` e1 = principal axis of the INNER centres (|C_i - c| <= r) projected
  onto the plane perpendicular to u. Sign: e1 . m >= 0, where m is the mean
  horizontal viewing direction (mean of the camera z axes projected
  perpendicular to u) -- a whole-walk quantity, where rule /1 used the first
  keyframe. e2 = u x e1.
  ``frame.conditioning`` records pca_ratio = lambda2/lambda1 (near 1: the
  axis is ill-defined) and view_resultant = |m| / mean |z_h| (near 0: the sign
  is ill-defined), and ``stable`` = pca_ratio <= 0.8 and view_resultant >= 0.15.
  When not stable, use transfer mode for cross-variant comparison.
* TOP: orthographic, looking along -u (camera z = -u, camera x = e1), framing
  the 2nd..98th percentile box of the C_i in (e1, e2), each side padded by
  50% of that axis's extent, a floor of 0.5 r per half-extent. Camera at
  c + 4 r u. Back faces culled ("dollhouse").
* ORBIT_k, k = 0..7: perspective, vertical FOV 60 deg, eye at
  c + 2 r (cos 35deg (cos a e1 + sin a e2) + sin 35deg u), a = 45deg k,
  looking at c with image-up along u. Back faces culled.
* TRAJ_jj, j = 0..11: the keyframe at capture-order index
  floor((j + 0.5) / 12 * N) among ALL N accepted keyframes; if it is not a
  supported main-component keyframe, the nearest one by capture-order index
  (ties: earlier), recorded as a substitution.
  - ``TRAJ_jj_insitu``: that keyframe's own pose and the given intrinsics.
  - ``TRAJ_jj_pullback``: same orientation, centre moved 0.75 r back along the
    viewing axis and 0.25 r along u, vertical FOV 75 deg.

The set also stores ``anchors`` (keyframe id -> camera centre of every
supported main-component keyframe), so `transfer_viewpoints` needs only the
set and the target poses.

API
---
``viewpoint_set(poses, capture_order, component_of, intrinsics, *,
observations=None, min_observations=30) -> dict``
``viewpoints_for_world(world_dir, session_id=None) -> dict``
``transfer_viewpoints(vs, poses_from, poses_to, *, observations_to=None,
component_of_to=None, intrinsics_to=None, insitu="own") -> dict``
``transfer_viewpoints_for_world(vs, world_dir, session_id=None) -> dict``
``save_viewpoints`` / ``load_viewpoints`` / ``views_by_name`` /
``view_T_world_camera`` / ``view_K`` / ``project(view, X) -> (uv, z)`` /
``poses_from_solution(meta) -> (poses, component_of)`` /
``observations_from_solution(meta) -> {kid: n}``.

CLI::

    python -m tower.world_builder.coherence_eval.viewpoints WORLD_DIR --out X.json
    python -m tower.world_builder.coherence_eval.viewpoints VARIANT_DIR \
        --transfer-from REFERENCE_VIEWPOINTS.json --out Y.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

RULE_VERSION = "wb-coherence-viewpoints/2"
SCHEMA = "glasses.wb.coherence.viewpoints/2"
READABLE_SCHEMAS = ("glasses.wb.coherence.viewpoints/1", SCHEMA)

MIN_OBSERVATIONS = 30          # == global_solve.MIN_IMAGE_OBSERVATIONS
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
STABLE_MAX_PCA_RATIO = 0.8
STABLE_MIN_VIEW_RESULTANT = 0.15

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


def geometric_median(X, iters: int = 500, tol: float = 1e-12) -> np.ndarray:
    """Weiszfeld's algorithm, started at the mean. Deterministic."""
    X = np.asarray(X, np.float64)
    y = X.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(X - y, axis=1)
        if (d < 1e-12).any():   # sitting on a sample: nudge off it
            d = np.maximum(d, 1e-12)
        w = 1.0 / d
        y2 = (X * w[:, None]).sum(0) / w.sum()
        if np.linalg.norm(y2 - y) <= tol * max(1.0, float(np.linalg.norm(y))):
            return y2
        y = y2
    return y


def umeyama(src, dst, with_scale: bool = True):
    """Least-squares similarity dst ~ s R src + t (Umeyama 1991)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs ** 2).sum() / len(src)
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s) if (with_scale and var_s > 0) else 1.0
    return s, R, mu_d - s * R @ mu_s


def robust_sim3(src, dst, *, trim: float = 3.0, iters: int = 5):
    """Umeyama with iterative trimming of residuals > trim * 1.4826 * MAD
    (floored at 1e-9 of the target spread). Returns (s, R, t, inlier mask)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    keep = np.ones(len(src), bool)
    spread = float(np.linalg.norm(dst - dst.mean(0), axis=1).mean()) or 1.0
    for _ in range(iters):
        s, R, t = umeyama(src[keep], dst[keep])
        res = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
        med = float(np.median(res[keep]))
        mad = float(np.median(np.abs(res[keep] - med)))
        thr = max(med + trim * 1.4826 * mad, 1e-9 * spread)
        new = res <= thr
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
    s, R, t = umeyama(src[keep], dst[keep])
    return s, R, t, keep


# ---------------------------------------------------------------------------
# the frame
# ---------------------------------------------------------------------------


def main_component(posed: Iterable[str], component_of: dict | None) -> tuple[int | None, list[str]]:
    """(component id, the given keyframe ids in it, order kept). None when
    there are no components. Size = count among `posed`."""
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


def supported_ids(poses: dict, observations: dict | None,
                  min_observations: int = MIN_OBSERVATIONS) -> set:
    if observations is None:
        return set(poses)
    return {k for k in poses if int(observations.get(k) or 0) >= int(min_observations)}


def walk_frame(poses: dict, capture_order: list[str], component_of: dict | None, *,
               observations: dict | None = None,
               min_observations: int = MIN_OBSERVATIONS) -> dict:
    """The (c, u, r, e1, e2) frame of the rule, plus bookkeeping."""
    order_index = {k: i for i, k in enumerate(capture_order)}
    sup = supported_ids(poses, observations, min_observations)
    posed = [k for k in capture_order if k in sup]
    posed += sorted(k for k in sup if k not in order_index)
    comp, main = main_component(posed, component_of)
    if len(main) < 3:
        raise ValueError(f"need >= 3 supported posed keyframes in the main component, have {len(main)}")
    T = np.array([np.asarray(poses[k], float) for k in main])
    C = T[:, :3, 3]
    c = geometric_median(C)
    u = robust_up([t[:3, :3] for t in T])
    d = np.linalg.norm(C - c, axis=1)
    r = float(np.percentile(d, RADIUS_PERCENTILE))
    if not r > 1e-12:
        r = float(d.max()) if d.max() > 1e-12 else 1.0
    inner = d <= r
    P = C[inner] - c
    P = P - np.outer(P @ u, u)
    P0 = P - P.mean(axis=0)
    w, V = np.linalg.eigh(P0.T @ P0)
    e1 = V[:, int(np.argmax(w))]
    e1 = e1 - (e1 @ u) * u
    if np.linalg.norm(e1) < 1e-9:  # all centres on the up axis
        e1 = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.cross(u, [0.0, 1.0, 0.0])
    e1 = _unit(e1)
    ws = np.sort(w)[::-1]
    pca_ratio = float(ws[1] / ws[0]) if ws[0] > 0 else 1.0
    Z = T[:, :3, 2]
    Zh = Z - np.outer(Z @ u, u)
    m = Zh.mean(axis=0)
    denom = float(np.linalg.norm(Zh, axis=1).mean()) or 1.0
    view_resultant = float(np.linalg.norm(m) / denom)
    if float(e1 @ m) < 0:
        e1 = -e1
    e2 = np.cross(u, e1)
    return {
        "component": comp,
        "main_keyframes": main,
        "posed_total": len(poses),
        "supported_total": len(sup),
        "center": c, "up": u, "radius": r, "axis1": e1, "axis2": e2,
        "centres": C,
        "conditioning": {
            "pca_ratio": pca_ratio, "view_resultant": view_resultant,
            "stable": bool(pca_ratio <= STABLE_MAX_PCA_RATIO
                           and view_resultant >= STABLE_MIN_VIEW_RESULTANT),
        },
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
        raise ValueError("no supported main-component keyframe in capture order")
    return {"quantile": q, "requested_capture_index": idx, "requested_keyframe_id": req,
            "capture_index": best, "keyframe_id": capture_order[best], "substituted": True,
            "substitution_reason": "unposed-unsupported-or-outside-main-component"}


def viewpoint_set(poses: dict, capture_order: list[str], component_of: dict | None,
                  intrinsics: dict, *, observations: dict | None = None,
                  min_observations: int = MIN_OBSERVATIONS,
                  orbit_size=DEFAULT_ORBIT_SIZE,
                  top_long_side: int = DEFAULT_TOP_LONG_SIDE,
                  pullback_scale: float = DEFAULT_PULLBACK_SCALE,
                  meta: dict | None = None) -> dict:
    """The full viewpoint set of the rule in the module docstring (native mode).

    poses: keyframe id -> (4, 4) T_world_camera (camera-to-world, OpenCV axes).
    capture_order: ALL accepted keyframe ids in capture order (posed or not).
    component_of: keyframe id -> solver component, or None (one component).
    intrinsics: {fx, fy, cx, cy, width, height} of the in-situ camera -- the
        pinhole the poses are expressed in.
    observations: keyframe id -> observation count; below `min_observations`
        a keyframe is unsupported and ignored. None: every pose is supported.
    Returns a JSON-serialisable dict.
    """
    for key in ("fx", "fy", "cx", "cy", "width", "height"):
        if key not in intrinsics:
            raise ValueError(f"intrinsics missing {key!r}")
    capture_order = list(capture_order)
    fr = walk_frame(poses, capture_order, component_of, observations=observations,
                    min_observations=min_observations)
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
            cx=float(intrinsics["cx"]), cy=float(intrinsics["cy"]), cull=False,
            pose_source="own", **ch))
        Cp = C - PULLBACK_BACK_R * r * R[:, 2] + PULLBACK_UP_R * r * u
        views.append(_perspective(
            f"TRAJ_{j:02d}_pullback", "TRAJ_PULLBACK", _T(R, Cp), pw, ph,
            PULLBACK_FOV_Y_DEG, cull=False, **ch))

    return {
        "schema": SCHEMA,
        "rule": RULE_VERSION,
        "mode": "native",
        "rule_parameters": {
            "orbit": {"n": N_ORBIT, "elevation_deg": ORBIT_ELEVATION_DEG,
                      "distance_r": ORBIT_DISTANCE_R, "fov_y_deg": ORBIT_FOV_Y_DEG},
            "traj": {"n": N_TRAJ, "pullback_back_r": PULLBACK_BACK_R,
                     "pullback_up_r": PULLBACK_UP_R, "pullback_fov_y_deg": PULLBACK_FOV_Y_DEG,
                     "pullback_scale": pullback_scale},
            "top": {"percentiles": list(TOP_PERCENTILES), "margin": TOP_MARGIN,
                    "min_half_r": TOP_MIN_HALF_R, "camera_height_r": 4.0},
            "support": ({"min_observations": int(min_observations)} if observations is not None
                        else "caller-vouched (no observation counts given)"),
            "up": f"trimmed mean of camera -Y (keep angle <= p{UP_TRIM_PERCENTILE:g})",
            "radius": f"p{RADIUS_PERCENTILE:g} of |C_i - c|",
            "center": "geometric median of supported main-component camera centres",
            "axis": "PCA of centres with |C_i - c| <= r, perpendicular to up; "
                    "sign by mean horizontal viewing direction",
        },
        "frame": {
            "component": fr["component"],
            "posed_in_component": len(fr["main_keyframes"]),
            "posed_total": fr["posed_total"],
            "supported_total": fr["supported_total"],
            "accepted_total": len(capture_order),
            "center": _list(c), "up": _list(u), "radius": float(r),
            "axis1": _list(e1), "axis2": _list(e2),
            "conditioning": fr["conditioning"],
        },
        "intrinsics": {k: (float(intrinsics[k]) if k in ("fx", "fy", "cx", "cy")
                           else int(intrinsics[k]))
                       for k in ("fx", "fy", "cx", "cy", "width", "height")},
        "main_keyframe_ids": list(fr["main_keyframes"]),
        "anchors": {k: _list(np.asarray(poses[k], float)[:3, 3]) for k in fr["main_keyframes"]},
        "views": views,
        "meta": dict(meta or {}),
    }


# ---------------------------------------------------------------------------
# transfer: one viewpoint set carried across variants of one world
# ---------------------------------------------------------------------------


def transfer_viewpoints(vs: dict, poses_from: dict | None, poses_to: dict, *,
                        observations_to: dict | None = None,
                        min_observations: int = MIN_OBSERVATIONS,
                        component_of_to: dict | None = None,
                        intrinsics_to: dict | None = None,
                        insitu: str = "own", min_shared: int = 6,
                        trim: float = 3.0, meta: dict | None = None) -> dict:
    """Carry viewpoint set `vs` into the gauge of `poses_to` (recommended for A/B).

    A Sim(3) X_to = s R X_from + t is fitted (Umeyama, trimmed at `trim` x
    1.4826 MAD) on the camera centres of keyframes that are in `vs`'s main
    component AND supported in `poses_to` (observation floor when
    `observations_to` is given) AND, when `component_of_to` is given, in the
    target's component holding the most of them. `poses_from` supplies the
    source centres; None uses the centres stored in `vs["anchors"]`.

    Every view is mapped by the Sim(3): R_wc' = R R_wc, C' = s R C + t, and
    orthographic half extents scale by s. TRAJ_INSITU views use the keyframe's
    OWN pose in `poses_to` when it is posed and supported there (`insitu="own"`,
    the default: in-situ means "from where this variant put that camera"), and
    the mapped pose otherwise (`pose_source` says which). Everything else,
    pullbacks included, is the mapped physical viewpoint. The returned set's
    anchors are expressed in the target gauge, so transfers chain.
    """
    if insitu not in ("own", "transferred"):
        raise ValueError(f"insitu must be 'own' or 'transferred', not {insitu!r}")
    if poses_from is None:
        anchors = vs.get("anchors")
        if not anchors:
            raise ValueError("viewpoint set has no anchors (rule /1?); pass poses_from")
        src_c = {k: np.asarray(v, float) for k, v in anchors.items()}
    else:
        src_c = {k: np.asarray(T, float)[:3, 3] for k, T in poses_from.items()}
    sup_to = supported_ids(poses_to, observations_to, min_observations)
    shared = [k for k in vs["main_keyframe_ids"] if k in src_c and k in sup_to]
    comp_used = None
    if component_of_to:
        comp_used, shared = main_component(shared, component_of_to)
    if len(shared) < max(3, int(min_shared)):
        raise ValueError(f"only {len(shared)} shared supported keyframes; need {max(3, min_shared)}")
    src = np.array([src_c[k] for k in shared])
    dst = np.array([np.asarray(poses_to[k], float)[:3, 3] for k in shared])
    s, R, t, inl = robust_sim3(src, dst, trim=trim)
    res = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
    r_from = float(vs["frame"]["radius"])
    r_to = s * r_from

    def mapT(Tm):
        Tm = np.asarray(Tm, float).reshape(4, 4)
        return _T(R @ Tm[:3, :3], s * R @ Tm[:3, 3] + t)

    out = json.loads(json.dumps(vs))  # deep copy
    for v in out["views"]:
        v["T_world_camera"] = _list(mapT(v["T_world_camera"]))
        if v["projection"] == "orthographic":
            v["half_width"] = float(v["half_width"] * s)
            v["half_height"] = float(v["half_height"] * s)
        if v["family"] == "TRAJ_INSITU":
            kid = v.get("keyframe_id")
            if insitu == "own" and kid in sup_to:
                v["T_world_camera"] = _list(np.asarray(poses_to[kid], float))
                v["pose_source"] = "own"
                if intrinsics_to:
                    for k2 in ("fx", "fy", "cx", "cy"):
                        v[k2] = float(intrinsics_to[k2])
                    v["width"], v["height"] = int(intrinsics_to["width"]), int(intrinsics_to["height"])
                    v["fov_y_deg"] = math.degrees(2.0 * math.atan(0.5 * v["height"] / v["fy"]))
            else:
                v["pose_source"] = "transferred"
            v["keyframe_in_target"] = bool(kid in sup_to)
    fr = out["frame"]
    fr["center"] = _list(s * R @ np.asarray(fr["center"], float) + t)
    for key in ("up", "axis1", "axis2"):
        fr[key] = _list(R @ np.asarray(fr[key], float))
    fr["radius"] = float(r_to)
    out["anchors"] = {k: _list(s * R @ np.asarray(c, float) + t) for k, c in src_c.items()
                      if k in set(vs["main_keyframe_ids"])}
    if intrinsics_to:
        out["intrinsics"] = {k: (float(intrinsics_to[k]) if k in ("fx", "fy", "cx", "cy")
                                 else int(intrinsics_to[k]))
                             for k in ("fx", "fy", "cx", "cy", "width", "height")}
    out["mode"] = "transferred"
    out["transfer"] = {
        "from_rule": vs.get("rule"), "from_mode": vs.get("mode", "native"),
        "from_meta": vs.get("meta", {}),
        "scale": float(s), "rotation": _list(R), "translation": _list(t),
        "shared": len(shared), "inliers": int(inl.sum()),
        "target_component": comp_used,
        "rms_over_r": float(np.sqrt((res[inl] ** 2).mean()) / r_to),
        "residual_p90_over_r_all": float(np.percentile(res, 90) / r_to),
        "insitu": insitu,
        "traj_keyframes_missing_in_target": sorted(
            v["keyframe_id"] for v in out["views"]
            if v["family"] == "TRAJ_INSITU" and v["keyframe_id"] not in sup_to),
    }
    out["meta"] = dict(meta or {})
    return out


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
    if vs.get("schema") not in READABLE_SCHEMAS:
        raise ValueError(f"{path}: not a viewpoints file (schema={vs.get('schema')!r})")
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


def observations_from_solution(meta: dict) -> dict:
    return {kid: int(p.get("observations") or 0) for kid, p in (meta.get("poses") or {}).items()}


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


def _read_world(world_dir, session_id):
    world_dir = Path(world_dir)
    sid = session_id or single_session_id(world_dir)
    meta = json.loads((world_dir / "solve" / sid / "solution.json").read_text(encoding="utf-8"))
    poses, comp = poses_from_solution(meta)
    return world_dir, sid, meta, poses, comp, observations_from_solution(meta)


def viewpoints_for_world(world_dir, session_id: str | None = None, *,
                         source: str = "solve",
                         min_observations: int = MIN_OBSERVATIONS, **kw) -> dict:
    """Native `viewpoint_set` for a world (or variant) directory, read-only.

    Poses: solve/<sid>/solution.json (the global solve), with its observation
    counts as the support rule. In-situ camera: the solution's pinhole -- the
    undistorted camera the poses, depth and fusion are expressed in; the
    session's self-calibrated intrinsics are recorded in `meta`.
    """
    if source != "solve":
        raise ValueError(f"unknown source {source!r}")
    world_dir, sid, meta, poses, comp, obs = _read_world(world_dir, session_id)
    order = capture_order_from_keyframes(world_dir / "sessions" / sid / "keyframes.jsonl")
    session_intr = None
    sj = world_dir / "sessions" / sid / "session.json"
    if sj.exists():
        session_intr = json.loads(sj.read_text(encoding="utf-8")).get("intrinsics")
    return viewpoint_set(poses, order, comp, dict(meta["camera"]), observations=obs,
                         min_observations=min_observations, meta={
        "world_dir": str(world_dir), "world_id": world_dir.name, "session_id": sid,
        "pose_source": f"solve/{sid}/solution.json",
        "solver": meta.get("solver"), "input_digest": meta.get("input_digest"),
        "session_intrinsics": session_intr,
        "insitu_camera": "solution.camera (undistorted pinhole of the solve)",
    }, **kw)


def transfer_viewpoints_for_world(vs: dict, world_dir, session_id: str | None = None, *,
                                  min_observations: int = MIN_OBSERVATIONS, **kw) -> dict:
    """`transfer_viewpoints` of `vs` onto a world (or variant) directory's solve."""
    world_dir, sid, meta, poses, comp, obs = _read_world(world_dir, session_id)
    return transfer_viewpoints(vs, None, poses, observations_to=obs,
                               min_observations=min_observations, component_of_to=comp,
                               intrinsics_to=dict(meta["camera"]), meta={
        "world_dir": str(world_dir), "world_id": world_dir.name, "session_id": sid,
        "pose_source": f"solve/{sid}/solution.json",
        "solver": meta.get("solver"), "input_digest": meta.get("input_digest"),
    }, **kw)


def main(argv=None) -> int:
    import argparse

    from tower.artifact_paths import artifact_root_arg

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("world_dir")
    ap.add_argument("--session", default=None)
    ap.add_argument("--out", required=True, type=artifact_root_arg,
                    help="output JSON path (refused at a drive root or directly in home)")
    ap.add_argument("--transfer-from", default=None,
                    help="a reference viewpoints JSON to carry onto WORLD_DIR (A/B mode)")
    a = ap.parse_args(argv)
    if a.transfer_from:
        vs = transfer_viewpoints_for_world(load_viewpoints(a.transfer_from), a.world_dir, a.session)
        tr = vs["transfer"]
        print(f"transferred: {tr['inliers']}/{tr['shared']} shared keyframes, scale {tr['scale']:.4g}, "
              f"rms/r {tr['rms_over_r']:.4f}; TRAJ keyframes missing in target: "
              f"{tr['traj_keyframes_missing_in_target']}")
    else:
        vs = viewpoints_for_world(a.world_dir, a.session)
    save_viewpoints(vs, a.out)
    fr = vs["frame"]
    print(f"{len(vs['views'])} views ({vs['mode']}); component {fr['component']} "
          f"({fr['posed_in_component']} supported of {fr['accepted_total']}); r={fr['radius']:.4g}; "
          f"conditioning {fr['conditioning']}")
    for v in vs["views"]:
        if v["family"] == "TRAJ_INSITU":
            print(f"  {v['name']}: {v['keyframe_id']}"
                  + (f" (for {v['requested_keyframe_id']})" if v["substituted"] else "")
                  + (f" [{v.get('pose_source')}]" if vs["mode"] == "transferred" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
