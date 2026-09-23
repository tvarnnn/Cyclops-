"""Coherence metrics for any reconstruction of a frozen world.

One function, `evaluate`, computes every metric for one variant (see
`eval_variant` for the interchange format) against per-world caches that
depend only on the images (`eval_depth`, `eval_pairs`). The saved world is the
baseline variant; every future experiment is scored by this same code.

GAUGE. A monocular reconstruction has an arbitrary frame and scale per
component. Every metric below is invariant under an independent Sim(3) of each
component: it uses only ratios of distances within one component, relative
rotations, camera-frame depths up to a per-component constant (log-ratios are
centred per component), and pixel errors. `tests/test_world_coherence_eval.py`
checks this under random Sim(3) transforms.

WHAT IS COUNTED. Pose metrics use the variant's PUBLISHED keyframes (what it
would show; `status` in the interchange format). Posed-but-unpublished
keyframes count only in registration. A variant therefore cannot improve the
geometric metrics by publishing less without registration showing the loss,
and cannot improve registration by publishing wrong poses without the
continuity, revisit and sanity metrics showing it. Region labels
(`regions`/`revisits`) are forensic annotations used ONLY in these reports;
nothing here feeds anything back into a reconstruction.

THE METRICS (keys in metrics.json; every threshold is in `PARAMS`)

a. registration  -- fractions of the world's accepted keyframes that are
   posed at all / published / published in the largest component.
b. components     -- components = the variant's stated frames (COLMAP models,
   solve components); a variant that states none gets the connected
   components of its covisibility graph (>= `covis_min_shared` shared
   points) when it has observations, else ONE frame. Sizes, count
   of joined (>= 2 published) and singleton components, the largest share,
   and `main_runs`: the maximal runs of consecutive capture indices inside the
   largest component (how often the walk leaves the main frame). With
   observations: `covisibility_components_in_main`, the connected components
   of the main frame's covisibility graph (edge = >= `covis_min_shared`
   shared points) -- a frame held together by a thin link shows up here.
c. continuity     -- per consecutive published pair (in capture order) of one
   component: ratio = |c_j+1 - c_j| / (local median per-keyframe step x index
   gap), the local median over `jump_window` steps each side (excluding the
   step itself), floored at `jump_floor_frac` x the component's median step
   (a pause must not make an ordinary step look like a jump). With keyframe
   times (the world's receipt times; identical for every variant) the same is
   computed for SPEED (step / max(dt, `dt_floor_s`)) and a translation JUMP
   needs BOTH ratios > `jump_ratio`; a rotation jump needs > `rot_jump_deg`
   per index gap AND > `rot_jump_deg_per_s`. The time test is what separates
   a pose error (a large displacement in a fraction of a second) from a
   legitimate long move across a tracking loss: on the known-good control
   world the index-only rule flags 14-16 steps, almost all at tracker-segment
   boundaries, the time-aware rule 2. `index_jumps` keeps the index-only count.
   A SPIKE keyframe has jumps on both sides. Each jump says whether a
   tracker-segment boundary (tracking lost in the frontend) lies inside it.
   Limitation: receipt times include network jitter (tens of ms); a solver
   error that moves a whole later block smoothly (a scale step) is NOT a jump
   -- that is what the scale metric is for.
d. scale          -- (i) `placement_scales`: spread of per-segment Sim(3)
   scales where the variant declares them (the saved world's are exactly 1 by
   construction: a GLOMAP model is one reconstruction). (ii) `depth`: per
   published keyframe, r_k = median over its observations of
   log(z_sfm / z_mono), z_sfm the camera-frame depth of the observed point,
   z_mono the cached monocular metric depth (MoGe-2) at the observation pixel
   (observations with reprojection error > `reproj_gate_px` excluded; >=
   `depth_min_samples` per keyframe). r is centred on its component median
   (gauge). Reported: robust spread (1.4826 MAD), Theil-Sen drift slope
   over capture index (per 100 keyframes) and the drift it implies across the
   component's span, step changes (difference of medians of
   `scale_step_window` keyframes either side, peaks > `scale_step_log`),
   spread per tracker segment, per region; and research lane D3's three
   level numbers (`levels`): the share of keyframes within x1.5 of the
   dominant scale level, the longest off-level run, and the largest step
   between 15-keyframe windows. `units_per_metre` (exp of the uncentred
   median) and `d3_pooled_informative` (all components pooled, as D3's pilot
   did) are gauge-DEPENDENT and informative only.
   Limitation: MoGe's per-image scale error (a few %) is noise in r_k; only
   multi-keyframe statistics are meaningful. Needs the variant's own
   observations; N/A otherwise.
e. revisits       -- against the world's cached verified pair set: per pair
   with both keyframes published in ONE component, the relative-rotation
   error (deg) and, where the pair's translation is reliable (essential-matrix
   model and parallax >= `t_min_parallax_deg`: below ~3 deg the measured
   direction itself scatters by tens of degrees on these images), the
   translation-direction error (deg) and the median Sampson epipolar error
   (px) of the pair's stored inliers under the variant's relative pose.
   Buckets by capture-order gap: adjacent (1), near (2..`revisit_window`),
   distant (> `revisit_window`, i.e. beyond any sequential matching window:
   true revisits). `joined_fraction` = pairs with both keyframes published in
   one component / all verified pairs of the bucket; `main_joined_fraction`
   likewise for the largest component. `annotated`: for C0's hand-annotated
   revisit ranges, the closest-approach and centroid distances between the
   two ranges' main-component camera centres over the main component's p90
   radius (evaluation only; two ranges can see one place from different
   positions, so a large distance is a flag, not a proof).
f. reprojection   -- the variant's own observations through its own poses and
   cameras: median, p90, p99, max, >3 px fraction, behind-camera count.
g. regions        -- per C0 region: keyframes, fraction posed, published,
   published in the main component, and "clean" (in main, not a spike, not an
   outlier camera); jump endpoints and scale per region.
h. sanity         -- per component: camera centres' distances from their
   geometric median; p90 radius R90, max/R90, outlier cameras (> `outlier_k`
   x R90); path length / R90; the same for the main component's points.
i. renders        -- the fixed trajectory-derived viewpoint set
   (`coherence_eval.viewpoints`, sibling lane C2) applied to the variant's own
   published poses; the number of TRAJ keyframes that had to be substituted
   (unposed or outside the main component) is itself a coverage number. With
   ``--renders``, the variant's cameras and sparse points are drawn from every
   view with C2's renderer (`layer_renders`) into ``<out>/renders/``. Visual
   evidence only; no number is derived from the images.
j. runtime        -- the world's recorded timings, or the variant's
   `meta.runtime` (see `measure_command`).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

HARNESS = "wb-coherence-metrics/1"

PARAMS = {
    "jump_window": 10,
    "jump_ratio": 5.0,
    "jump_floor_frac": 0.25,
    "rot_jump_deg": 30.0,
    "rot_jump_deg_per_s": 250.0,
    "dt_floor_s": 0.05,
    "outlier_k": 3.0,
    "revisit_window": 30,
    "t_min_parallax_deg": 3.0,
    "reproj_gate_px": 4.0,
    "depth_min_samples": 10,
    "depth_valid_m": [0.05, 50.0],
    "scale_step_window": 8,
    "scale_step_log": round(math.log(1.25), 6),
    "coherent_band_factor": 1.5,
    "coherent_grid": 400,
    "d3_step_window": 15,
    "covis_min_shared": 15,
    "min_component_for_stats": 5,
    "rot_err_thresholds_deg": [2.0, 5.0, 10.0],
    "t_err_thresholds_deg": [10.0, 30.0],
    "worst_list": 15,
}

# ---------------------------------------------------------------------------
# small math


def rot_angle_deg(R) -> np.ndarray:
    """Rotation angle(s) in degrees, numerically stable near 0 and 180."""
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    R = R.reshape(-1, 3, 3)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    ang = np.degrees(np.arctan2(np.linalg.norm(v, axis=1), tr - 1.0))
    return ang[0] if single else ang


def vec_angle_deg(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    dot = (a * b).sum(1)
    return np.degrees(np.arctan2(cross, dot))


def geometric_median(X, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Weiszfeld. Equivariant under rotation, translation and scale (unlike a
    coordinate-wise median), which keeps the sanity metrics gauge-invariant."""
    X = np.asarray(X, dtype=np.float64)
    y = X.mean(0)
    for _ in range(iters):
        d = np.linalg.norm(X - y, axis=1)
        d = np.maximum(d, 1e-12 * (np.abs(X).max() + 1.0))
        w = 1.0 / d
        y_new = (X * w[:, None]).sum(0) / w.sum()
        if np.linalg.norm(y_new - y) <= tol * (np.linalg.norm(y) + 1e-12):
            y = y_new
            break
        y = y_new
    return y


def stats(values, *, thresholds=None, prefix_over="over_") -> dict | None:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    out = {"count": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
           "p99": float(np.percentile(v, 99)), "max": float(v.max()), "mean": float(v.mean())}
    for t in thresholds or ():
        out[f"{prefix_over}{t:g}"] = float((v > t).mean())
    return out


def mad(v) -> float | None:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _ranges(indices) -> list[list[int]]:
    """[3,4,5,9,10] -> [[3,5],[9,10]]"""
    out = []
    for i in sorted(int(x) for x in indices):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def round_sig(obj, sig: int = 6):
    """Recursively round floats to `sig` significant digits (deterministic JSON)."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        if obj == 0.0:
            return 0.0
        return float(f"{obj:.{sig}g}")
    if isinstance(obj, (np.floating,)):
        return round_sig(float(obj), sig)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): round_sig(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_sig(v, sig) for v in obj]
    if isinstance(obj, np.ndarray):
        return round_sig(obj.tolist(), sig)
    return obj


# ---------------------------------------------------------------------------
# camera models (reprojection)

_SUPPORTED = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV",
              "PINHOLE_RADTAN"}


def cv_intrinsics(cam: dict):
    """(K, dist) in OpenCV form for the supported models, else None."""
    model = str(cam.get("model", "")).upper()
    p = [float(x) for x in cam.get("params") or []]
    if model not in _SUPPORTED:
        return None
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = p[:3]
        fx = fy = f
        rest = p[3:]
    else:
        fx, fy, cx, cy = p[:4]
        rest = p[4:]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if model in ("SIMPLE_PINHOLE", "PINHOLE"):
        dist = np.zeros(5)
    elif model == "SIMPLE_RADIAL":
        dist = np.array([rest[0], 0, 0, 0, 0.0])
    elif model == "RADIAL":
        dist = np.array([rest[0], rest[1], 0, 0, 0.0])
    elif model == "OPENCV":
        dist = np.array([*rest[:4], 0.0])
    elif model == "PINHOLE_RADTAN":
        dist = np.array((rest + [0.0] * 5)[:5])
    else:  # FULL_OPENCV: k1 k2 p1 p2 k3 k4 k5 k6 == OpenCV rational
        dist = np.array((rest + [0.0] * 8)[:8])
    return K, dist


def project(cam: dict, Pc: np.ndarray) -> np.ndarray | None:
    """Camera-frame points -> pixels (NaN for z <= 0)."""
    ki = cv_intrinsics(cam)
    if ki is None:
        return None
    K, dist = ki
    Pc = np.asarray(Pc, dtype=np.float64).reshape(-1, 3)
    out = np.full((len(Pc), 2), np.nan)
    ok = Pc[:, 2] > 1e-9
    if not ok.any():
        return out
    if not np.any(dist):
        uv = (K @ (Pc[ok] / Pc[ok, 2:3]).T).T[:, :2]
    else:
        import cv2

        uv = cv2.projectPoints(Pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    out[ok] = uv
    return out


def unproject_normalized(cam: dict, uv) -> np.ndarray | None:
    """Pixels -> normalised image coordinates (x/z, y/z) through the camera model."""
    ki = cv_intrinsics(cam)
    if ki is None:
        return None
    K, dist = ki
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if not len(uv):
        return uv
    if not np.any(dist):
        return (uv - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
    import cv2

    return cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist).reshape(-1, 2)


# ---------------------------------------------------------------------------
# prepared view of a variant over the world's keyframe order


class Prepared:
    def __init__(self, variant, keyframe_ids: list[str], segment_of: list[int] | None = None,
                 times=None) -> None:
        self.v = variant
        self.times = None if times is None else np.asarray(times, dtype=np.float64)
        self.ids = list(keyframe_ids)
        self.n = len(self.ids)
        self.index_of = {k: i for i, k in enumerate(self.ids)}
        self.segment_of = list(segment_of) if segment_of is not None else None
        n = self.n
        self.posed = np.zeros(n, bool)
        self.published = np.zeros(n, bool)
        self.comp = np.array([None] * n, dtype=object)
        self.C = np.full((n, 3), np.nan)
        self.Rwc = np.full((n, 3, 3), np.nan)
        for kid, T in variant.poses.items():
            i = self.index_of.get(kid)
            if i is None:
                continue
            T = np.asarray(T, dtype=np.float64)
            self.posed[i] = True
            self.published[i] = variant.status.get(kid, "published") == "published"
            self.comp[i] = str(variant.component.get(kid, "0"))
            self.C[i] = T[:3, 3]
            self.Rwc[i] = T[:3, :3]
        pub_comp = [c for c, p in zip(self.comp, self.published) if p]
        counts: dict[str, int] = {}
        for c in pub_comp:
            counts[c] = counts.get(c, 0) + 1
        # largest by published keyframes; ties -> the component seen first in capture order
        first_seen = {}
        for i in range(n):
            if self.published[i] and self.comp[i] not in first_seen:
                first_seen[self.comp[i]] = i
        self.comp_counts = counts
        self.order = sorted(counts, key=lambda c: (-counts[c], first_seen[c]))
        self.main = self.order[0] if self.order else None
        self.in_main = self.published & (self.comp == self.main) if self.main is not None else np.zeros(n, bool)

    def members(self, comp) -> np.ndarray:
        return np.nonzero(self.published & (self.comp == comp))[0]

    def Rcw(self, i):
        return self.Rwc[i].T

    def tcw(self, i):
        return -self.Rwc[i].T @ self.C[i]


# ---------------------------------------------------------------------------
# a. registration


def registration(p: Prepared) -> dict:
    n = max(p.n, 1)
    return {
        "keyframes": p.n,
        "posed": int(p.posed.sum()),
        "published": int(p.published.sum()),
        "in_largest": int(p.in_main.sum()),
        "posed_fraction": p.posed.sum() / n,
        "published_fraction": p.published.sum() / n,
        "in_largest_fraction": p.in_main.sum() / n,
        "unposed_ranges": _ranges(np.nonzero(~p.posed)[0]),
        "unpublished_posed_ranges": _ranges(np.nonzero(p.posed & ~p.published)[0]),
    }


# ---------------------------------------------------------------------------
# b. components


def _covisibility_components(p: Prepared, keyframes: np.ndarray, min_shared: int) -> list[int] | None:
    v = p.v
    if not v.has_observations:
        return None
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ids = v.obs_keyframe_ids
    world_idx = np.array([p.index_of.get(k, -1) for k in ids])
    kf = world_idx[v.obs_keyframe]
    keep = np.isin(kf, keyframes)
    if not keep.any():
        return []
    local = {int(k): j for j, k in enumerate(keyframes)}
    rows = np.array([local[int(k)] for k in kf[keep]])
    pts, inv = np.unique(v.obs_point[keep], return_inverse=True)
    M = coo_matrix((np.ones(len(rows)), (rows, inv)), shape=(len(keyframes), len(pts))).tocsr()
    M.data[:] = 1.0
    S = (M @ M.T).tocoo()
    mask = (S.data >= min_shared) & (S.row != S.col)
    A = coo_matrix((np.ones(mask.sum()), (S.row[mask], S.col[mask])), shape=S.shape)
    ncomp, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels, minlength=ncomp)
    return sorted((int(s) for s in sizes), reverse=True)


def components(p: Prepared) -> dict:
    comps = []
    for c in p.order:
        m = p.members(c)
        posed = np.nonzero(p.posed & (p.comp == c))[0]
        comps.append({"id": c, "published": int(len(m)), "posed": int(len(posed)),
                      "first_index": int(m.min()), "last_index": int(m.max()),
                      "ranges": _ranges(m)[:40]})
    total = int(p.published.sum())
    runs = _ranges(np.nonzero(p.in_main)[0])
    out = {
        "stated": bool(p.v.component_stated),
        "definition": ("stated by the variant (solver components / COLMAP models)" if p.v.component_stated
                       else p.v.meta.get("_component_definition", "not stated: all poses taken as one frame")),
        "count": len(comps),
        "joined": sum(1 for c in comps if c["published"] >= 2),
        "singletons": sum(1 for c in comps if c["published"] == 1),
        "largest_id": p.main,
        "largest_share_of_published": (comps[0]["published"] / total) if comps and total else None,
        "sizes": [c["published"] for c in comps],
        "components": comps[:20],
        "main_runs": len(runs),
        "main_run_ranges": runs[:60],
    }
    if p.main is not None:
        cov = _covisibility_components(p, p.members(p.main), PARAMS["covis_min_shared"])
        out["covisibility_components_in_main"] = (
            None if cov is None else {"count": len(cov), "sizes": cov[:20],
                                      "min_shared_points": PARAMS["covis_min_shared"]})
    return out


# ---------------------------------------------------------------------------
# c. continuity


def _local_median(x: np.ndarray, W: int, floor: float) -> np.ndarray:
    """Median of up to W values either side of each position (excluding it),
    floored."""
    g = float(np.median(x))
    out = np.empty(len(x))
    for j in range(len(x)):
        window = np.concatenate([x[max(0, j - W):j], x[j + 1:j + 1 + W]])
        out[j] = max(float(np.median(window)) if len(window) else g, floor)
    return out


def _continuity_for(p: Prepared, members: np.ndarray) -> dict:
    W = PARAMS["jump_window"]
    T = PARAMS["jump_ratio"]
    if len(members) < 3:
        return {"steps": max(0, len(members) - 1), "insufficient": True}
    C = p.C[members]
    R = p.Rwc[members]
    gap = np.diff(members).astype(np.float64)
    d = np.linalg.norm(np.diff(C, axis=0), axis=1)
    per = d / gap
    local = _local_median(per, W, PARAMS["jump_floor_frac"] * float(np.median(per)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(local > 0, d / (local * gap), np.where(d > 0, np.inf, 0.0))
    rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
    rot = rot_angle_deg(rel)
    tj_idx = ratio > T
    rj_idx = (rot / gap) > PARAMS["rot_jump_deg"]
    timed = p.times is not None and np.isfinite(p.times[members]).all()
    if timed:
        dt = np.maximum(np.diff(p.times[members]), PARAMS["dt_floor_s"])
        speed = d / dt
        lsp = _local_median(speed, W, PARAMS["jump_floor_frac"] * float(np.median(speed)))
        with np.errstate(divide="ignore", invalid="ignore"):
            sratio = np.where(lsp > 0, speed / lsp, np.where(speed > 0, np.inf, 0.0))
        tj = tj_idx & (sratio > T)
        rj = rj_idx & ((rot / dt) > PARAMS["rot_jump_deg_per_s"])
    else:
        dt = sratio = None
        tj, rj = tj_idx, rj_idx
    any_j = tj | rj
    any_idx = tj_idx | rj_idx
    spikes = [int(members[j]) for j in range(1, len(members) - 1) if any_j[j - 1] and any_j[j]]
    endpoints = sorted({int(members[j]) for j in np.nonzero(any_j)[0]}
                       | {int(members[j + 1]) for j in np.nonzero(any_j)[0]})

    def boundary(j):
        if p.segment_of is None:
            return None
        a, b = int(members[j]), int(members[j + 1])
        return len(set(p.segment_of[a:b + 1])) > 1

    jumps = []
    for j in np.nonzero(any_idx)[0]:
        a, b = int(members[j]), int(members[j + 1])
        jumps.append({"from": a, "to": b, "from_id": p.ids[a], "to_id": p.ids[b], "gap": int(b - a),
                      "ratio": float(ratio[j]),
                      "speed_ratio": None if sratio is None else float(sratio[j]),
                      "dt_s": None if dt is None else float(dt[j]),
                      "rot_deg": float(rot[j]),
                      "counted": bool(any_j[j]),
                      "kind": "+".join(k for k, f in (("translation", tj_idx[j]), ("rotation", rj_idx[j])) if f),
                      "segment_boundary_inside": boundary(j)})
    counted = [x for x in jumps if x["counted"]]
    r90 = float(np.percentile(np.linalg.norm(C - geometric_median(C), axis=1), 90))
    return {
        "criterion": "index-and-time" if timed else "index-only",
        "steps": int(len(d)),
        "jumps": int(any_j.sum()),
        "translation_jumps": int(tj.sum()),
        "rotation_jumps": int(rj.sum()),
        "jumps_within_segments": int(sum(1 for x in counted if x["segment_boundary_inside"] is False)),
        "jumps_at_segment_boundary": int(sum(1 for x in counted if x["segment_boundary_inside"])),
        "spikes": len(spikes),
        "index_jumps": int(any_idx.sum()),
        "index_jumps_at_segment_boundary": int(sum(1 for x in jumps if x["segment_boundary_inside"])),
        "max_ratio": float(np.max(ratio)) if len(ratio) else None,
        "ratio": stats(ratio),
        "speed_ratio": None if sratio is None else stats(sratio),
        "rot_step_deg": stats(rot),
        "path_length_over_r90": float(d.sum() / r90) if r90 > 0 else None,
        "jump_list": jumps[:80],
        "spike_keyframes": spikes,
        "_endpoints": endpoints,
    }


def continuity(p: Prepared) -> dict:
    per = {}
    for c in p.order:
        m = p.members(c)
        if len(m) >= 3:
            per[c] = _continuity_for(p, m)
    main = per.get(p.main) if p.main is not None else None
    out = {"main": main,
           "all_components": {
               "components_evaluated": len(per),
               "jumps": int(sum(x["jumps"] for x in per.values())),
               "spikes": int(sum(x["spikes"] for x in per.values())),
               "translation_jumps": int(sum(x["translation_jumps"] for x in per.values())),
               "rotation_jumps": int(sum(x["rotation_jumps"] for x in per.values())),
           },
           "per_component": {c: {k: x[k] for k in ("steps", "jumps", "spikes", "max_ratio")}
                             for c, x in per.items()}}
    return out


# ---------------------------------------------------------------------------
# h. sanity


def _radii(X):
    gm = geometric_median(X)
    r = np.linalg.norm(X - gm, axis=1)
    return r, float(np.percentile(r, 90))


def sanity(p: Prepared) -> dict:
    k = PARAMS["outlier_k"]
    out = {"outlier_k": k, "per_component": {}}
    outliers_main = []
    for c in p.order:
        m = p.members(c)
        if len(m) < PARAMS["min_component_for_stats"]:
            continue
        r, r90 = _radii(p.C[m])
        o = m[r > k * r90] if r90 > 0 else np.zeros(0, int)
        row = {"cameras": int(len(m)), "max_over_r90": float(r.max() / r90) if r90 > 0 else None,
               "outlier_cameras": int(len(o))}
        out["per_component"][c] = row
        if c == p.main:
            outliers_main = [int(x) for x in o]
            out["main"] = dict(row, outlier_camera_indices=outliers_main)
    v = p.v
    if v.has_observations and p.main is not None:
        pc = _point_components(p)
        sel = pc == p.main
        if sel.sum() >= 10:
            X = v.xyz[sel]
            r, r90 = _radii(X)
            cam_r90 = _radii(p.C[p.members(p.main)])[1] if len(p.members(p.main)) >= 3 else None
            out["main_points"] = {
                "points": int(len(X)),
                "max_over_r90": float(r.max() / r90) if r90 > 0 else None,
                "outlier_points": int((r > k * r90).sum()) if r90 > 0 else None,
                "outlier_fraction": float((r > k * r90).mean()) if r90 > 0 else None,
                "points_r90_over_cameras_r90": (r90 / cam_r90) if cam_r90 else None,
            }
    out["_outliers_main"] = outliers_main
    return out


def _point_components(p: Prepared) -> np.ndarray:
    v = p.v
    if v.point_component is not None:
        return np.asarray(v.point_component).astype(str)
    # component of the first published observer
    pc = np.array([None] * len(v.xyz), dtype=object)
    world_idx = np.array([p.index_of.get(k, -1) for k in v.obs_keyframe_ids])
    kf = world_idx[v.obs_keyframe]
    for o in np.argsort(kf, kind="stable"):
        i = kf[o]
        if i < 0 or not p.published[i]:
            continue
        pt = v.obs_point[o]
        if pc[pt] is None:
            pc[pt] = p.comp[i]
    return pc.astype(str)


# ---------------------------------------------------------------------------
# f. reprojection (and the per-observation table the depth metric reuses)


def observation_table(p: Prepared):
    """Per observation of a PUBLISHED keyframe: world index, point, uv, camera-frame
    depth, reprojection error (NaN if the camera model is unsupported)."""
    v = p.v
    if not v.has_observations:
        return None
    world_idx = np.array([p.index_of.get(k, -1) for k in v.obs_keyframe_ids])
    kf = world_idx[v.obs_keyframe]
    ok = (kf >= 0)
    ok[ok] = p.published[kf[ok]]
    idx = np.nonzero(ok)[0]
    kf = kf[idx]
    pts = v.obs_point[idx]
    uv = v.obs_uv[idx]
    z = np.full(len(idx), np.nan)
    err = np.full(len(idx), np.nan)
    unsupported = set()
    order = np.argsort(kf, kind="stable")
    bounds = np.searchsorted(kf[order], np.arange(p.n + 1))
    for i in range(p.n):
        rows = order[bounds[i]:bounds[i + 1]]
        if not len(rows):
            continue
        Rcw = p.Rcw(i)
        tcw = p.tcw(i)
        Pc = (Rcw @ v.xyz[pts[rows]].T).T + tcw
        z[rows] = Pc[:, 2]
        cam = v.cameras.get(v.camera_of.get(p.ids[i], "default")) or next(iter(v.cameras.values()), None)
        proj = project(cam, Pc) if cam else None
        if proj is None:
            unsupported.add(str((cam or {}).get("model")))
            continue
        err[rows] = np.hypot(proj[:, 0] - uv[rows, 0], proj[:, 1] - uv[rows, 1])
    return {"kf": kf, "point": pts, "uv": uv, "z": z, "err": err, "unsupported": sorted(unsupported)}


def reprojection(p: Prepared, table) -> dict:
    if table is None:
        return {"available": False, "why": "the variant has no observations (points.npz)"}
    behind = int((table["z"] <= 0).sum())
    valid = np.isfinite(table["err"]) & (table["z"] > 0)
    out = {"available": True, "observations": int(len(table["err"])), "behind_camera": behind,
           "unsupported_camera_models": table["unsupported"],
           "overall": stats(table["err"][valid], thresholds=[3.0])}
    per = {}
    comp = p.comp[table["kf"]]
    for c in p.order:
        sel = valid & (comp == c)
        if sel.any():
            s = stats(table["err"][sel], thresholds=[3.0])
            per[c] = {k: s[k] for k in ("count", "median", "p90", "p99")}
    out["per_component"] = per
    return out


# ---------------------------------------------------------------------------
# d. scale


def keyframe_depth_ratios(p: Prepared, table, depth_fn) -> dict | None:
    """Per published keyframe: median log(z_sfm / z_mono), sample count, and the
    within-keyframe spread (1.4826 MAD) of the log-ratio."""
    if table is None or depth_fn is None:
        return None
    lo, hi = PARAMS["depth_valid_m"]
    r = np.full(p.n, np.nan)
    n = np.zeros(p.n, int)
    spread = np.full(p.n, np.nan)
    kf = table["kf"]
    order = np.argsort(kf, kind="stable")
    bounds = np.searchsorted(kf[order], np.arange(p.n + 1))
    for i in range(p.n):
        rows = order[bounds[i]:bounds[i + 1]]
        if not len(rows):
            continue
        z = table["z"][rows]
        err = table["err"][rows]
        keep = z > 0
        keep &= ~(err > PARAMS["reproj_gate_px"])  # NaN error (unsupported model) is kept
        if keep.sum() < PARAMS["depth_min_samples"]:
            continue
        cam = p.v.cameras.get(p.v.camera_of.get(p.ids[i], "default"))
        zm = depth_fn(i, table["uv"][rows[keep]], cam)
        if zm is None:
            continue
        zs = z[keep]
        ok = np.isfinite(zm) & (zm > lo) & (zm < hi)
        if ok.sum() < PARAMS["depth_min_samples"]:
            continue
        lr = np.log(zs[ok] / zm[ok])
        r[i] = float(np.median(lr))
        n[i] = int(ok.sum())
        spread[i] = mad(lr)
    return {"r": r, "n": n, "spread": spread}


def _theil_sen(x, y):
    from scipy.stats import theilslopes

    res = theilslopes(y, x, 0.95)
    return float(res[0]), float(res[2]), float(res[3])


def _steps(x_idx, y, w, thr):
    """Step changes along the sequence.

    Detection is robust: at each split j, the difference of the MEDIANS of the
    w values after and the w values before; a split is a candidate when that
    exceeds `thr`. Localisation uses the difference of the MEANS (it peaks
    exactly at a clean step, where the median difference is flat over several
    splits). Candidates are taken in order of |mean difference| with
    non-maximum suppression over w; the reported factor is exp(|median
    difference|) at the chosen split."""
    n = len(y)
    if n < 2 * w:
        return []
    dmed = np.full(n, np.nan)
    dmean = np.full(n, np.nan)
    for j in range(w, n - w + 1):
        dmed[j] = np.median(y[j:j + w]) - np.median(y[j - w:j])
        dmean[j] = np.mean(y[j:j + w]) - np.mean(y[j - w:j])
    cand = [j for j in range(n) if np.isfinite(dmed[j]) and abs(dmed[j]) > thr]
    cand.sort(key=lambda j: (-abs(dmean[j]), j))
    taken: list[int] = []
    for j in cand:
        if all(abs(j - t) >= w for t in taken):
            taken.append(j)
    return [{"at_index": int(x_idx[j]), "between": [int(x_idx[j - 1]), int(x_idx[j])],
             "delta_log": float(dmed[j]), "factor": float(math.exp(abs(dmed[j])))}
            for j in sorted(taken)]


def level_stats(y_ordered) -> dict | None:
    """Research lane D3's three scale-coherence numbers on a log-scale sequence
    in capture order (sign convention irrelevant: every statistic is symmetric).

    * coherent_fraction: share within x`coherent_band_factor` of the DOMINANT
      level, the grid value (``coherent_grid`` points spanning min..max) whose
      band holds the most keyframes;
    * longest_off_level_run: longest run of consecutive off-level keyframes;
    * max_step_15kf: exp of the largest |median(next w) - median(previous w)|,
      w = `d3_step_window`, and where it happens.
    Shift-invariant, hence gauge-invariant on one component."""
    y = np.asarray(y_ordered, dtype=np.float64)
    if len(y) < 3:
        return None
    band = math.log(PARAMS["coherent_band_factor"])
    grid = np.linspace(y.min(), y.max(), PARAMS["coherent_grid"])
    cnt = [(np.abs(y - g) <= band).sum() for g in grid]
    g = grid[int(np.argmax(cnt))]
    ok = np.abs(y - g) <= band
    run = best = 0
    for o in ok:
        run = 0 if o else run + 1
        best = max(best, run)
    w = PARAMS["d3_step_window"]
    step, at = 0.0, None
    for i in range(w, len(y) - w):
        d = abs(float(np.median(y[i:i + w]) - np.median(y[i - w:i])))
        if d > step and d > 1e-9:  # float noise is not a step
            step, at = d, i
    return {"coherent_fraction": float(ok.mean()), "off_level": int((~ok).sum()),
            "longest_off_level_run": int(best), "max_step_15kf_factor": float(math.exp(step)),
            "_max_step_pos": at}


def scale(p: Prepared, ratios, regions: dict | None) -> dict:
    out: dict = {}
    segs = p.v.segments
    if segs:
        sc = np.array([float(s["scale"]) for s in segs
                       if s.get("scale") not in (None, 0) and s.get("state", "registered") == "registered"])
        if len(sc):
            ls = np.log(sc)
            out["placement_scales"] = {
                "segments": int(len(sc)), "log_std": float(ls.std()),
                "max_over_min": float(sc.max() / sc.min()),
                "note": ("all exactly 1: every registered segment is a rigid piece of one global model"
                         if np.allclose(sc, 1.0) else "Sim(3) placement scales differ between segments")}
    if ratios is None:
        out["depth"] = {"available": False,
                        "why": "no observations in the variant, or no depth cache for the world"}
        return out
    r = ratios["r"]
    depth: dict = {"available": True, "keyframes_with_ratio": int(np.isfinite(r).sum()),
                   "median_samples_per_keyframe": float(np.median(ratios["n"][np.isfinite(r)]))
                   if np.isfinite(r).any() else None,
                   "within_keyframe_spread_median": float(np.nanmedian(ratios["spread"]))
                   if np.isfinite(ratios["spread"]).any() else None,
                   "per_component": {}}
    centred = np.full(p.n, np.nan)
    for c in p.order:
        m = p.members(c)
        m = m[np.isfinite(r[m])]
        if len(m) < PARAMS["min_component_for_stats"]:
            continue
        med = float(np.median(r[m]))
        y = r[m] - med
        centred[m] = y
        row = {"keyframes": int(len(m)), "spread_mad": mad(y),
               "p05_p95": [float(np.percentile(y, 5)), float(np.percentile(y, 95))],
               "units_per_metre_informative": float(math.exp(med))}
        if len(m) >= 10:
            slope, lo, hi = _theil_sen(m.astype(float), y)
            span = float(m.max() - m.min())
            row["drift_slope_per_100kf"] = slope * 100.0
            row["drift_slope_ci95_per_100kf"] = [lo * 100.0, hi * 100.0]
            row["drift_factor_over_span"] = float(math.exp(abs(slope * span)))
            steps = _steps(m, y, PARAMS["scale_step_window"], PARAMS["scale_step_log"])
            row["steps"] = len(steps)
            row["max_step_factor"] = max((s["factor"] for s in steps), default=1.0)
            row["step_list"] = steps[:20]
        lv = level_stats(y)
        if lv is not None:
            pos = lv.pop("_max_step_pos")
            lv["max_step_15kf_at_index"] = int(m[pos]) if pos is not None else None
            row["levels"] = lv
        depth["per_component"][c] = row
    main = depth["per_component"].get(p.main)
    depth["main"] = main
    # Research lane D3's pilot pooled every fitted keyframe of the world,
    # whatever its component. Across components that is gauge-DEPENDENT
    # (each component has its own scale); kept only to reproduce D3's numbers.
    allp = np.nonzero(p.published & np.isfinite(r))[0]
    lv = level_stats(r[allp]) if len(allp) >= 3 else None
    if lv is not None:
        pos = lv.pop("_max_step_pos")
        lv["max_step_15kf_at_index"] = int(allp[pos]) if pos is not None else None
        lv["keyframes"] = int(len(allp))
        lv["note"] = "all published keyframes pooled across components (gauge-dependent; D3-comparable)"
    depth["d3_pooled_informative"] = lv
    if main is not None and p.segment_of is not None:
        by_seg: dict[int, list[float]] = {}
        for i in p.members(p.main):
            if np.isfinite(centred[i]):
                by_seg.setdefault(int(p.segment_of[i]), []).append(centred[i])
        seg_med = {s: float(np.median(v)) for s, v in by_seg.items() if len(v) >= 3}
        if seg_med:
            vals = np.array(list(seg_med.values()))
            depth["by_tracker_segment"] = {
                "segments": len(seg_med), "log_spread_mad": mad(vals),
                "max_over_min_factor": float(math.exp(vals.max() - vals.min())),
                "median_log_by_segment": {str(s): seg_med[s] for s in sorted(seg_med)}}
    if main is not None and regions:
        by_reg: dict[str, list[float]] = {}
        for i in p.members(p.main):
            reg = regions.get(p.ids[i])
            if reg and np.isfinite(centred[i]):
                by_reg.setdefault(reg["region"], []).append(centred[i])
        depth["by_region"] = {g: {"keyframes": len(v), "median_log": float(np.median(v)),
                                  "factor_vs_main": float(math.exp(np.median(v))), "spread_mad": mad(v)}
                              for g, v in sorted(by_reg.items())}
    out["depth"] = depth
    out["_centred"] = centred
    return out


# ---------------------------------------------------------------------------
# e. revisits


def _sampson_px(R, t, x1, x2, f):
    """Median Sampson distance (px) of normalised correspondences under E = [t]x R."""
    t = t / (np.linalg.norm(t) + 1e-15)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    a = np.c_[x1, np.ones(len(x1))]
    b = np.c_[x2, np.ones(len(x2))]
    Ea = (E @ a.T).T
    Etb = (E.T @ b.T).T
    num = (b * Ea).sum(1) ** 2
    den = Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2
    return float(np.median(np.sqrt(num / np.maximum(den, 1e-30)))) * f


def revisits(p: Prepared, pairs, regions: dict | None, focal_px: float | None) -> dict:
    if pairs is None or not getattr(pairs, "available", False):
        return {"available": False, "why": "no verified pair cache for this world"}
    A = pairs.arrays
    I, J = A["i"].astype(int), A["j"].astype(int)
    W = PARAMS["revisit_window"]
    gap = J - I
    bucket = np.where(gap == 1, "adjacent", np.where(gap <= W, "near", "distant"))
    both = p.published[I] & p.published[J]
    same = both & (p.comp[I] == p.comp[J])
    main = same & (p.comp[I] == p.main)
    rot_err = np.full(len(I), np.nan)
    t_err = np.full(len(I), np.nan)
    samp = np.full(len(I), np.nan)
    offs = A.get("inlier_offsets")
    for k in np.nonzero(same)[0]:
        i, j = I[k], J[k]
        Ri, Rj = p.Rcw(i), p.Rcw(j)
        R_ji = Rj @ Ri.T
        t_ji = Rj @ (p.C[i] - p.C[j])
        rot_err[k] = rot_angle_deg(A["R"][k].T @ R_ji)
        nt = np.linalg.norm(t_ji)
        if A["t_reliable"][k] and A["parallax_deg"][k] >= PARAMS["t_min_parallax_deg"] and nt > 0:
            t_err[k] = vec_angle_deg(t_ji / nt, A["t"][k])[0]
            if offs is not None and focal_px:
                a, b = int(offs[k]), int(offs[k + 1])
                if b - a >= 5:
                    samp[k] = _sampson_px(R_ji, t_ji, A["inlier_xy_i"][a:b].astype(np.float64),
                                          A["inlier_xy_j"][a:b].astype(np.float64), focal_px)
    out: dict = {"available": True, "pairs": int(len(I)), "window": W,
                 "pair_cache_digest": pairs.digest()}
    for b in ("adjacent", "near", "distant"):
        sel = bucket == b
        nb = int(sel.sum())
        row = {"pairs": nb, "evaluated": int((sel & same).sum()),
               "split_across_components": int((sel & both & ~same).sum()),
               "unposed": int((sel & ~both).sum()),
               "joined_fraction": float((sel & same).sum() / nb) if nb else None,
               "main_joined_fraction": float((sel & main).sum() / nb) if nb else None,
               "rot_err_deg": stats(rot_err[sel], thresholds=PARAMS["rot_err_thresholds_deg"]),
               "t_err_deg": stats(t_err[sel], thresholds=PARAMS["t_err_thresholds_deg"]),
               "sampson_px": stats(samp[sel])}
        out[b] = row
    # worst distant pairs
    dist = np.nonzero((bucket == "distant") & same)[0]
    # rounded so that numerically-equal errors order by index, not by float noise
    worst = sorted(dist, key=lambda k: (-round(float(rot_err[k]), 3), int(I[k]), int(J[k])))[: PARAMS["worst_list"]]
    out["worst_distant"] = [
        {"i": int(I[k]), "j": int(J[k]), "rot_err_deg": float(rot_err[k]),
         "t_err_deg": (float(t_err[k]) if np.isfinite(t_err[k]) else None),
         "inliers": int(A["n_inliers"][k]),
         "regions": ([regions.get(p.ids[I[k]], {}).get("region"), regions.get(p.ids[J[k]], {}).get("region")]
                     if regions else None)}
        for k in worst]
    if regions:
        by: dict[str, list[int]] = {}
        for k in np.nonzero(bucket == "distant")[0]:
            ri = (regions.get(p.ids[I[k]]) or {}).get("region")
            rj = (regions.get(p.ids[J[k]]) or {}).get("region")
            key = ri if ri == rj and ri else "cross-region"
            by.setdefault(key, []).append(k)
        out["distant_by_region"] = {
            g: {"pairs": len(ks), "joined_fraction": float(np.mean(same[ks])),
                "main_joined_fraction": float(np.mean(main[ks])),
                "rot_err_median_deg": (float(np.nanmedian(rot_err[ks])) if np.isfinite(rot_err[ks]).any() else None),
                "t_err_median_deg": (float(np.nanmedian(t_err[ks])) if np.isfinite(t_err[ks]).any() else None)}
            for g, ks in sorted(by.items())}
    out["_arrays"] = {"I": I, "J": J, "same": same, "rot_err": rot_err, "t_err": t_err}
    return out


def annotated_revisits(p: Prepared, rows: list[dict] | None, rev: dict | None) -> dict:
    if not rows:
        return {"available": False, "why": "no annotated revisit ranges for this world"}
    main = p.members(p.main) if p.main is not None else np.zeros(0, int)
    r90 = _radii(p.C[main])[1] if len(main) >= 3 else None
    out = []
    arr = (rev or {}).get("_arrays")
    for row in rows:
        a0, a1, b0, b1 = row["a"][0], row["a"][1], row["b"][0], row["b"][1]
        A = np.arange(a0, a1 + 1)
        B = np.arange(b0, b1 + 1)
        Am = A[p.in_main[A]] if len(A) else A
        Bm = B[p.in_main[B]] if len(B) else B
        rec = {"a": [int(a0), int(a1)], "b": [int(b0), int(b1)], "region": row.get("region"),
               "confidence": row.get("confidence"),
               "a_in_main": float(len(Am) / len(A)) if len(A) else None,
               "b_in_main": float(len(Bm) / len(B)) if len(B) else None}
        if len(Am) and len(Bm) and r90:
            D = np.linalg.norm(p.C[Am][:, None, :] - p.C[Bm][None, :, :], axis=2)
            rec["closest_over_r90"] = float(D.min() / r90)
            rec["centroid_over_r90"] = float(np.linalg.norm(p.C[Am].mean(0) - p.C[Bm].mean(0)) / r90)
        else:
            rec["closest_over_r90"] = None
            rec["centroid_over_r90"] = None
            rec["joined"] = False
        if arr is not None:
            inA = (arr["I"] >= a0) & (arr["I"] <= a1) & (arr["J"] >= b0) & (arr["J"] <= b1)
            inA |= (arr["J"] >= a0) & (arr["J"] <= a1) & (arr["I"] >= b0) & (arr["I"] <= b1)
            rec["verified_pairs_between"] = int(inA.sum())
            rec["verified_pairs_joined"] = int((inA & arr["same"]).sum())
            e = arr["rot_err"][inA & arr["same"]]
            rec["rot_err_median_deg"] = float(np.median(e)) if len(e) else None
        out.append(rec)
    closest = [r["closest_over_r90"] for r in out if r["closest_over_r90"] is not None]
    return {"available": True, "ranges": len(out), "joined": len(closest),
            "closest_over_r90_median": float(np.median(closest)) if closest else None,
            "closest_over_r90_max": float(max(closest)) if closest else None,
            "list": out}


# ---------------------------------------------------------------------------
# g. regions


def region_coverage(p: Prepared, regions: dict | None, spikes, outliers, endpoints) -> dict:
    if not regions:
        return {"available": False, "why": "no region labels for this world"}
    spikes, outliers, endpoints = set(spikes), set(outliers), set(endpoints)
    groups: dict[str, list[int]] = {}
    for i, kid in enumerate(p.ids):
        reg = regions.get(kid)
        if reg:
            groups.setdefault(reg["region"], []).append(i)
    out = {}
    for g, idx in sorted(groups.items()):
        idx = np.array(idx)
        n = len(idx)
        in_main = p.in_main[idx]
        clean = np.array([in_main[k] and idx[k] not in spikes and idx[k] not in outliers for k in range(n)])
        out[g] = {"keyframes": n, "posed": float(p.posed[idx].mean()),
                  "published": float(p.published[idx].mean()),
                  "in_main": float(in_main.mean()), "in_main_clean": float(clean.mean()),
                  "jump_endpoints": int(sum(1 for i in idx if i in endpoints)),
                  "spikes": int(sum(1 for i in idx if i in spikes)),
                  "outlier_cameras": int(sum(1 for i in idx if i in outliers))}
    return {"available": True, "regions": out}


# ---------------------------------------------------------------------------
# label files (evaluation-only)


def read_regions(path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kid = row.get("keyframe_id")
            if kid:
                out[kid] = {"region": (row.get("region") or "").strip() or "unclear",
                            "confidence": (row.get("confidence") or "").strip()}
    return out


def read_revisits(path) -> list[dict] | None:
    path = Path(path)
    if not path.is_file():
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                a = [int(row["range_a_start"]), int(row["range_a_end"])]
                b = [int(row["range_b_start"]), int(row["range_b_end"])]
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({"a": sorted(a), "b": sorted(b), "region": row.get("region"),
                         "confidence": row.get("confidence")})
    return rows


# ---------------------------------------------------------------------------
# evaluation


def covisibility_labels(variant, keyframe_ids, min_shared: int) -> dict[str, str] | None:
    """keyframe_id -> component label from the covisibility graph over ALL
    posed keyframes (edge = >= min_shared shared points); None without
    observations."""
    if not variant.has_observations:
        return None
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    posed = [k for k in keyframe_ids if k in variant.poses]
    local = {k: i for i, k in enumerate(posed)}
    rows = np.array([local.get(k, -1) for k in variant.obs_keyframe_ids])[variant.obs_keyframe]
    keep = rows >= 0
    if not keep.any():
        return {k: "0" for k in posed}
    pts, inv = np.unique(variant.obs_point[keep], return_inverse=True)
    M = coo_matrix((np.ones(int(keep.sum())), (rows[keep], inv)), shape=(len(posed), len(pts))).tocsr()
    M.data[:] = 1.0
    S = (M @ M.T).tocoo()
    mask = (S.data >= min_shared) & (S.row != S.col)
    A = coo_matrix((np.ones(int(mask.sum())), (S.row[mask], S.col[mask])), shape=S.shape)
    _, labels = connected_components(A, directed=False)
    return {k: f"covis{int(labels[i])}" for k, i in local.items()}


def evaluate(variant, keyframe_ids, *, segment_of=None, times=None, pairs=None, depth_fn=None, regions=None,
             revisit_ranges=None, focal_px=None) -> dict:
    """Every pose/structure metric for one variant. Pure: no IO.

    keyframe_ids / segment_of / times describe the WORLD (capture order, the
    frontend's tracker segments, keyframe receipt times); they are the same for
    every variant of that world.

    A variant that does not STATE its components gets them from its
    observations (connected components of the covisibility graph, >=
    `covis_min_shared` shared points), else all its poses are one frame. State
    components (even all "0") when the poses share one frame by construction
    (e.g. a pose graph whose odometry joins what no shared point does)."""
    if not variant.component_stated:
        labels = covisibility_labels(variant, keyframe_ids, PARAMS["covis_min_shared"])
        if labels is not None:
            import copy

            variant = copy.copy(variant)
            variant.component = dict(labels)
            variant.meta = dict(variant.meta, _component_definition=(
                f"not stated: covisibility components (>= {PARAMS['covis_min_shared']} shared points)"))
    p = Prepared(variant, keyframe_ids, segment_of, times)
    table = observation_table(p)
    ratios = keyframe_depth_ratios(p, table, depth_fn)
    cont = continuity(p)
    san = sanity(p)
    sc = scale(p, ratios, regions)
    rev = revisits(p, pairs, regions, focal_px)
    ann = annotated_revisits(p, revisit_ranges, rev)
    main_c = cont.get("main") or {}
    reg = region_coverage(p, regions, main_c.get("spike_keyframes", []), san.get("_outliers_main", []),
                          main_c.get("_endpoints", []))
    if isinstance(rev, dict):
        rev = {k: v for k, v in rev.items() if not k.startswith("_")}
        rev["annotated"] = ann
    for block in (cont.get("main") or {},):
        block.pop("_endpoints", None)
    san.pop("_outliers_main", None)
    sc.pop("_centred", None)
    per_kf = None
    if ratios is not None:
        per_kf = {"depth_log_ratio": [None if not np.isfinite(x) else float(x) for x in ratios["r"]]}
    return {
        "registration": registration(p),
        "components": components(p),
        "continuity": cont,
        "scale": sc,
        "revisits": rev,
        "reprojection": reprojection(p, table),
        "regions": reg,
        "sanity": san,
        "_per_keyframe": per_kf,
    }


def _git_head(path: Path) -> dict:
    import subprocess

    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--", "tower/tower/world_builder/coherence_eval",
                                "tower/scripts/world_coherence_eval.py"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return {"commit": head or None, "harness_files_uncommitted": bool(dirty)}
    except Exception:  # noqa: BLE001
        return {"commit": None}


def _harness_digest() -> str:
    h = hashlib.sha1()
    here = Path(__file__).resolve().parent
    for name in sorted(p.name for p in here.glob("*.py") if p.name == "metrics.py" or p.name.startswith("eval_")):
        h.update((here / name).read_bytes())
    return h.hexdigest()[:16]


def evaluate_world_variant(world, variant, *, cache_root, regions_dir=None, renders=False, out_dir=None) -> dict:
    """Glue: caches + labels + runtime around `evaluate`."""
    from tower.world_builder.coherence_eval.eval_depth import DepthCache, depth_dir
    from tower.world_builder.coherence_eval.eval_pairs import PairSet, pairs_dir
    from tower.world_builder.coherence_eval.eval_world import CANONICAL, RAW

    depth = DepthCache(depth_dir(cache_root, world.world_id))
    pairs = PairSet(pairs_dir(cache_root, world.world_id))
    depth_fn = None
    depth_note = None
    if depth.available:
        K_can = world.canonical_K()
        if variant.image_space == CANONICAL:
            def to_can(uv, cam):
                return uv
        elif variant.image_space == RAW:
            def to_can(uv, cam):
                return world.raw_to_canonical(uv)
        else:
            depth_note = "custom image space: pixels mapped through the variant's own camera (approximate)"

            def to_can(uv, cam):
                xy = unproject_normalized(cam, uv) if cam else None
                if xy is None:
                    return np.full((len(uv), 2), np.nan)
                return (K_can @ np.c_[xy, np.ones(len(xy))].T).T[:, :2]

        def depth_fn(i, uv, cam):
            return depth.sample(world.image_name(i), to_can(np.asarray(uv), cam))
    else:
        depth_note = "depth cache missing: run `world_coherence_eval.py cache --what depth`"
    regions = revisit_rows = None
    if regions_dir is not None:
        regions = read_regions(Path(regions_dir) / f"{world.world_id}_regions.csv")
        revisit_rows = read_revisits(Path(regions_dir) / f"{world.world_id}_revisits.csv")
    segment_of = [int(k.get("segment_index", -1)) for k in world.keyframes]
    times = [float(k["received_at"]) if k.get("received_at") is not None else float("nan")
             for k in world.keyframes]
    t0 = time.time()
    body = evaluate(variant, world.keyframe_ids, segment_of=segment_of, times=times,
                    pairs=pairs if pairs.available else None, depth_fn=depth_fn, regions=regions,
                    revisit_ranges=revisit_rows, focal_px=float(world.canonical_camera["fx"]))
    eval_seconds = time.time() - t0
    if depth_note:
        body["scale"].setdefault("depth", {})["note"] = depth_note
    body["renders"] = render_block(world, variant, out_dir, renders, regions=regions)
    body["runtime"] = {"world_recorded": world.runtime_evidence() if variant.meta.get("adapter") == "world" else None,
                       "variant_measured": variant.meta.get("runtime")}
    meta = {k: v for k, v in variant.meta.items() if k not in ("runtime",)}
    repo = Path(__file__).resolve().parents[4]
    result = {
        "harness": {"version": HARNESS, "params": PARAMS, "code_digest": _harness_digest(), **_git_head(repo)},
        "world": {"world_id": world.world_id, "session_id": world.session_id, "keyframes": world.n,
                  "world_dir": str(world.world_dir), "canonical_camera": world.canonical_camera,
                  "caches": {"depth": {"available": depth.available,
                                       "params": (depth.manifest or {}).get("params")},
                             "pairs": {"available": pairs.available, "count": len(pairs),
                                       "digest": pairs.digest(),
                                       "counts": (pairs.manifest or {}).get("counts"),
                                       "params": (pairs.manifest or {}).get("params")}},
                  "labels": {"regions": regions is not None, "revisit_ranges": revisit_rows is not None}},
        "variant": {"name": variant.name, "image_space": variant.image_space, "meta": meta,
                    "posed": len(variant.poses), "points": 0 if variant.xyz is None else int(len(variant.xyz)),
                    "observations": 0 if variant.obs_point is None else int(len(variant.obs_point))},
        "metrics": {k: v for k, v in body.items() if not k.startswith("_")},
        "per_keyframe": body.get("_per_keyframe"),
    }
    result["_eval_seconds"] = eval_seconds
    return result


# ---------------------------------------------------------------------------
# i. renders (sibling lane C2's tools)


def render_block(world, variant, out_dir, do_render: bool, regions=None) -> dict:
    """Viewpoints from C2's deterministic rule; renders when its renderer exists.

    `coherence_eval.viewpoints.viewpoint_set` is applied to THIS variant's
    published poses (the rule is keyed by keyframe id and Sim(3)-invariant in
    its keyframe choice). The TRAJ substitution count is itself a coverage
    signal: a TRAJ keyframe outside the main component is replaced by the
    nearest posed one."""
    block: dict = {}
    try:
        from tower.world_builder.coherence_eval import viewpoints as vp
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "why": f"viewpoints module unavailable: {type(exc).__name__}"}
    poses = {k: np.asarray(T) for k, T in variant.poses.items() if variant.status.get(k) == "published"}
    comp = {k: variant.component.get(k, "0") for k in poses}
    cam = world.canonical_camera
    try:
        vs = vp.viewpoint_set(poses, world.keyframe_ids, comp,
                              {k: cam[k] for k in ("fx", "fy", "cx", "cy", "width", "height")})
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "why": f"viewpoint_set failed: {type(exc).__name__}: {exc}"}
    views = vs.get("views") or []
    subs = sorted({str(v.get("requested_keyframe_id")) for v in views if v.get("substituted")})
    block["viewpoints"] = {"rule": vs.get("rule") or getattr(vp, "RULE_VERSION", None),
                           "views": len(views),
                           "traj_keyframes": len({v.get("requested_keyframe_id") for v in views
                                                  if str(v.get("family", "")).startswith("TRAJ")}),
                           "traj_substitutions": len(subs),
                           "traj_substituted_keyframes": subs}
    if out_dir is not None:
        try:
            vp.save_viewpoints(vs, Path(out_dir) / "viewpoints.json")
            block["viewpoints"]["file"] = "viewpoints.json"
        except Exception as exc:  # noqa: BLE001
            block["viewpoints"]["save_error"] = f"{type(exc).__name__}: {exc}"
    if not do_render:
        block["renders"] = {"available": False, "why": "not requested (--renders)"}
        return block
    try:
        block["renders"] = render_variant(world, variant, vs, Path(out_dir) / "renders", regions=regions)
    except Exception as exc:  # noqa: BLE001
        block["renders"] = {"available": False, "why": f"render failed: {type(exc).__name__}: {exc}"}
    return block


RENDER_LAYERS = ("cameras", "sparse", "sparse_time")


def render_variant(world, variant, vs, out_dir, *, regions=None, layers=RENDER_LAYERS) -> dict:
    """Render the variant's PUBLISHED cameras and points from the fixed viewpoint
    set with sibling lane C2's renderer (`coherence_eval.layer_renders`).

    The variant is wrapped in C2's `WorldLayers` shape, so any variant -- not
    only world-shaped directories -- renders identically. Components are
    renumbered by size (0 = largest) for colouring; minor components are drawn
    in their own gauge, exactly as C2 draws the saved world. With region labels
    the `_region` layers are added (evaluation-only colouring). Surface layers
    are not rendered here: a variant is a pose/point reconstruction, and the
    saved world's surface renders are C2's (`world_layer_renders.py`).
    """
    from tower.world_builder.coherence_eval import layer_renders as lr

    out_dir = Path(out_dir)
    pub = [k for k in world.keyframe_ids if variant.status.get(k) == "published" and k in variant.poses]
    counts: dict[str, int] = {}
    for k in pub:
        counts[variant.component.get(k, "0")] = counts.get(variant.component.get(k, "0"), 0) + 1
    rank = {c: i for i, c in enumerate(sorted(counts, key=lambda c: (-counts[c], c)))}
    poses = {k: np.asarray(variant.poses[k]) for k in pub}
    comp_of = {k: rank[variant.component.get(k, "0")] for k in pub}
    X = np.zeros((0, 3))
    pcomp = np.zeros(0, np.int32)
    first_kid: list = []
    if variant.has_observations:
        order_of = {k: i for i, k in enumerate(world.keyframe_ids)}
        obs_world = np.array([order_of.get(k, -1) for k in variant.obs_keyframe_ids])[variant.obs_keyframe]
        pubset = np.array([world.keyframe_ids[i] in poses if i >= 0 else False for i in range(world.n)] + [False])
        ok = (obs_world >= 0) & pubset[np.where(obs_world >= 0, obs_world, world.n)]
        first = np.full(len(variant.xyz), np.iinfo(np.int64).max)
        np.minimum.at(first, variant.obs_point[ok], obs_world[ok])
        keep = first < np.iinfo(np.int64).max
        X = variant.xyz[keep]
        first_kid = [world.keyframe_ids[i] for i in first[keep]]
        pcomp = np.array([comp_of[k] for k in first_kid], np.int32)
    wl = lr.WorldLayers(
        world_dir=world.world_dir, session_id=world.session_id, solution={}, poses=poses,
        component_of=comp_of, capture_order=list(world.keyframe_ids), keyframe_ids=list(world.keyframe_ids),
        xyz=X, point_component=pcomp, point_first_kid=first_kid,
        observations=np.zeros((0, 3), np.int64), camera=dict(world.canonical_camera), session=world.session,
        surface_manifest=None,
        regions={k: v["region"] for k, v in (regions or {}).items()})
    r = float(vs["frame"]["radius"])
    layers = list(layers)
    if regions:
        layers += ["cameras_region", "sparse_region"]
    kid_rank = {k: i / max(1, world.n - 1) for i, k in enumerate(world.keyframe_ids)}
    col_comp = np.array([lr.COMPONENT_RGB[int(c) % len(lr.COMPONENT_RGB)] for c in pcomp], np.uint8).reshape(-1, 3)
    col_time = lr.turbo(np.array([kid_rank.get(k, 0.0) for k in first_kid])) if len(first_kid) else col_comp
    col_reg = np.array([lr.REGION_RGB.get(wl.regions.get(k), lr.REGION_RGB[None]) for k in first_kid],
                       np.uint8).reshape(-1, 3)
    index = {"viewpoint_rule": vs.get("rule"), "layers": {}, "published_cameras": len(poses),
             "points_drawn": int(len(X)), "component_colour_rank": {c: i for c, i in rank.items()}}
    for layer in layers:
        tiles = []
        for view in vs["views"]:
            if layer == "cameras":
                img = lr.render_cameras(view, wl, r, "component")
            elif layer == "cameras_region":
                img = lr.render_cameras(view, wl, r, "region")
            elif layer == "sparse":
                img = lr.render_points(view, X, col_comp)
            elif layer == "sparse_time":
                img = lr.render_points(view, X, col_time)
            elif layer == "sparse_region":
                img = lr.render_points(view, X, col_reg)
            else:
                continue
            lr.save_png(out_dir / layer / f"{view['name']}.png", img)
            cap = view["name"]
            if "keyframe_id" in view:
                cap += "\n" + str(view["keyframe_id"]).split(":")[-1] + (" (subst)" if view.get("substituted") else "")
            tiles.append((cap, img))
        if tiles:
            sheet = lr.contact_sheet(tiles, cols=6, tile=(300, 300),
                                     title=f"{world.world_id[:8]} {variant.name} {layer}")
            lr.save_png(out_dir / "sheets" / f"{layer}.png", sheet)
        index["layers"][layer] = len(tiles)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return {"available": True, "dir": "renders", "layers": index["layers"],
            "published_cameras": len(poses), "points_drawn": int(len(X))}


# ---------------------------------------------------------------------------
# outputs


HEADLINE = [
    # (label, path, better)
    ("published / accepted", "registration.published_fraction", "higher"),
    ("in largest component / accepted", "registration.in_largest_fraction", "higher"),
    ("components (with published kf)", "components.count", "lower"),
    ("main-component runs in capture order", "components.main_runs", "lower"),
    ("covisibility comps in main", "components.covisibility_components_in_main.count", "lower"),
    ("jumps (main; index+time)", "continuity.main.jumps", "lower"),
    ("  of which within a tracker segment", "continuity.main.jumps_within_segments", "lower"),
    ("index-only jumps (main)", "continuity.main.index_jumps", "lower"),
    ("spikes (main)", "continuity.main.spikes", "lower"),
    ("max step ratio (main)", "continuity.main.max_ratio", "lower"),
    ("p99 step ratio (main)", "continuity.main.ratio.p99", "lower"),
    ("jumps (all components)", "continuity.all_components.jumps", "lower"),
    ("depth log-ratio spread MAD (main)", "scale.depth.main.spread_mad", "lower"),
    ("scale drift factor over span (main)", "scale.depth.main.drift_factor_over_span", "lower"),
    ("scale steps > 1.25x (main)", "scale.depth.main.steps", "lower"),
    ("max scale step factor (main)", "scale.depth.main.max_step_factor", "lower"),
    ("scale-coherent fraction x1.5 (main, D3)", "scale.depth.main.levels.coherent_fraction", "higher"),
    ("longest off-level run (main, D3)", "scale.depth.main.levels.longest_off_level_run", "lower"),
    ("max 15-kf scale step factor (main, D3)", "scale.depth.main.levels.max_step_15kf_factor", "lower"),
    ("segment scale max/min (main)", "scale.depth.by_tracker_segment.max_over_min_factor", "lower"),
    ("adjacent pairs joined", "revisits.adjacent.joined_fraction", "higher"),
    ("adjacent rot err median (deg)", "revisits.adjacent.rot_err_deg.median", "lower"),
    ("adjacent rot err p90 (deg)", "revisits.adjacent.rot_err_deg.p90", "lower"),
    ("near rot err p90 (deg)", "revisits.near.rot_err_deg.p90", "lower"),
    ("distant pairs joined (any comp)", "revisits.distant.joined_fraction", "higher"),
    ("distant pairs joined (main)", "revisits.distant.main_joined_fraction", "higher"),
    ("distant rot err median (deg)", "revisits.distant.rot_err_deg.median", "lower"),
    ("distant rot err p90 (deg)", "revisits.distant.rot_err_deg.p90", "lower"),
    ("distant rot err > 5 deg", "revisits.distant.rot_err_deg.over_5", "lower"),
    ("distant t-dir err median (deg)", "revisits.distant.t_err_deg.median", "lower"),
    ("distant Sampson median (px)", "revisits.distant.sampson_px.median", "lower"),
    ("annotated revisits joined", "revisits.annotated.joined", "higher"),
    ("annotated closest/R90 median", "revisits.annotated.closest_over_r90_median", "lower"),
    ("reprojection median (px)", "reprojection.overall.median", "lower"),
    ("reprojection p90 (px)", "reprojection.overall.p90", "lower"),
    ("reprojection p99 (px)", "reprojection.overall.p99", "lower"),
    ("outlier cameras (main)", "sanity.main.outlier_cameras", "lower"),
    ("camera max/R90 (main)", "sanity.main.max_over_r90", "lower"),
    ("outlier point fraction (main)", "sanity.main_points.outlier_fraction", "lower"),
]


def get_path(d, path):
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) >= 100:
            return f"{x:.0f}"
        if abs(x) >= 1:
            return f"{x:.3g}" if abs(x) < 10 else f"{x:.1f}"
        return f"{x:.3f}"
    return str(x)


def headline(result: dict) -> list[tuple[str, object, str]]:
    m = result["metrics"]
    rows = [(label, get_path(m, path), better) for label, path, better in HEADLINE]
    regs = get_path(m, "regions.regions") or {}
    for g, r in regs.items():
        rows.append((f"region {g}: in main & clean", r.get("in_main_clean"), "higher"))
    return rows


def markdown(result: dict) -> str:
    m = result["metrics"]
    w = result["world"]
    v = result["variant"]
    L = [f"# Coherence metrics: {v['name']} on world {w['world_id'][:8]}", ""]
    L.append(f"World `{w['world_id']}` session `{w['session_id']}`, {w['keyframes']} accepted keyframes. "
             f"Variant `{v['name']}` ({v['meta'].get('adapter', 'interchange')}), {v['posed']} posed, "
             f"{v['points']} points, {v['observations']} observations, image space `{v['image_space']}`.")
    h = result["harness"]
    L.append(f"Harness {h['version']} code {h['code_digest']} at commit {str(h.get('commit'))[:12]}"
             f"{' (+uncommitted harness files)' if h.get('harness_files_uncommitted') else ''}. "
             f"Pair cache {w['caches']['pairs'].get('digest')} ({w['caches']['pairs'].get('count')} pairs); "
             f"depth cache {'present' if w['caches']['depth']['available'] else 'MISSING'}.")
    L += ["", "| metric | value |", "|---|---|"]
    for label, val, better in headline(result):
        L.append(f"| {label} ({better} is better) | {_fmt(val)} |")
    main = get_path(m, "continuity.main") or {}
    if main.get("jump_list"):
        L += ["", "**Index-rule jump candidates in the main component** (capture index from->to; step ratio, "
              "speed ratio, dt, rotation; COUNTED = also fails the time test):", ""]
        for j in main["jump_list"][:30]:
            L.append(f"- {j['from']}->{j['to']} gap {j['gap']}: ratio {_fmt(j['ratio'])}, speed ratio "
                     f"{_fmt(j.get('speed_ratio'))}, dt {_fmt(j.get('dt_s'))} s, rot {_fmt(j['rot_deg'])} deg"
                     f" [{j['kind']}]{' (segment boundary)' if j.get('segment_boundary_inside') else ''}"
                     f"{' COUNTED' if j.get('counted') else ''}")
    steps = get_path(m, "scale.depth.main.step_list") or []
    if steps:
        L += ["", "**Scale steps (main component, depth log-ratio)**: " +
              ", ".join(f"at {s['at_index']} x{_fmt(s['factor'])}" for s in steps)]
    regs = get_path(m, "regions.regions")
    if regs:
        L += ["", "| region | kf | posed | published | in main | in main & clean | jump endpoints | scale vs main |",
              "|---|---|---|---|---|---|---|---|"]
        by_reg = get_path(m, "scale.depth.by_region") or {}
        for g, r in regs.items():
            f = (by_reg.get(g) or {}).get("factor_vs_main")
            L.append(f"| {g} | {r['keyframes']} | {_fmt(r['posed'])} | {_fmt(r['published'])} | {_fmt(r['in_main'])} | "
                     f"{_fmt(r['in_main_clean'])} | {r['jump_endpoints']} | {_fmt(f)} |")
    dbr = get_path(m, "revisits.distant_by_region")
    if dbr:
        L += ["", "| distant pairs by region | pairs | joined | joined (main) | rot err median | t err median |",
              "|---|---|---|---|---|---|"]
        for g, r in dbr.items():
            L.append(f"| {g} | {r['pairs']} | {_fmt(r['joined_fraction'])} | {_fmt(r['main_joined_fraction'])} | "
                     f"{_fmt(r['rot_err_median_deg'])} | {_fmt(r['t_err_median_deg'])} |")
    ann = get_path(m, "revisits.annotated")
    if ann and ann.get("available"):
        L += ["", "| annotated revisit (C0) | region | A in main | B in main | closest/R90 | centroid/R90 | verified pairs (joined) | rot err med |",
              "|---|---|---|---|---|---|---|---|"]
        for r in ann["list"]:
            L.append(f"| {r['a'][0]}-{r['a'][1]} vs {r['b'][0]}-{r['b'][1]} | {r.get('region')} | {_fmt(r['a_in_main'])} | "
                     f"{_fmt(r['b_in_main'])} | {_fmt(r['closest_over_r90'])} | {_fmt(r['centroid_over_r90'])} | "
                     f"{r.get('verified_pairs_between')} ({r.get('verified_pairs_joined')}) | {_fmt(r.get('rot_err_median_deg'))} |")
    rt = m.get("runtime") or {}
    wr = rt.get("world_recorded") or {}
    vm = rt.get("variant_measured") or {}
    L += ["", "**Runtime**: " + (
        f"final solve {_fmt(wr.get('final_solve_seconds'))} s, all logged solves {_fmt(wr.get('solve_log_total_seconds'))} s, "
        f"finalization {_fmt(wr.get('finalization_seconds'))} s; "
        + ", ".join(f"{k} {_fmt((s or {}).get('wall_seconds'))} s" for k, s in (wr.get("stages") or {}).items())
        + "; GPU/RAM not recorded by the world." if wr else
        (f"wall {_fmt(vm.get('wall_s'))} s, peak RSS {_fmt(vm.get('peak_rss_mb'))} MB, "
         f"peak VRAM delta {_fmt(vm.get('peak_vram_delta_mb'))} MB" if vm else "not recorded"))]
    rb = m.get("renders") or {}
    L += ["", f"**Renders**: viewpoints {get_path(rb, 'viewpoints.views')} "
              f"(TRAJ substitutions {get_path(rb, 'viewpoints.traj_substitutions')}); "
              f"renders: {get_path(rb, 'renders.why') or 'see renders/'}"]
    L += ["", "Definitions, thresholds and limitations: `tower/world_builder/coherence_eval/metrics.py` docstring; "
              "every threshold is in `harness.params` of metrics.json."]
    return "\n".join(L) + "\n"


def write_outputs(result: dict, out) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    clean = round_sig(clean)
    (out / "metrics.json").write_text(json.dumps(clean, indent=1, sort_keys=True), encoding="utf-8")
    (out / "metrics.md").write_text(markdown(clean), encoding="utf-8")


def table_markdown(results: list[dict], names: list[str] | None = None) -> str:
    """One row per headline metric, one column per metrics.json (any worlds or
    variants). Region rows appear for every region any result labels."""
    names = names or [f"{r['variant']['name']} @ {r['world']['world_id'][:8]}" for r in results]
    rows: dict[str, list] = {}
    better_of: dict[str, str] = {}
    for col, res in enumerate(results):
        for label, val, better in headline(res):
            rows.setdefault(label, [None] * len(results))[col] = val
            better_of[label] = better
    L = ["| metric | better | " + " | ".join(names) + " |",
         "|---|---|" + "---|" * len(results)]
    for label, vals in rows.items():
        L.append(f"| {label} | {better_of[label]} | " + " | ".join(_fmt(v) for v in vals) + " |")
    return "\n".join(L) + "\n"


def compare_markdown(a: dict, b: dict) -> str:
    na, nb = a["variant"]["name"], b["variant"]["name"]
    L = [f"# Compare: A = {na} ({a['world']['world_id'][:8]}) vs B = {nb} ({b['world']['world_id'][:8]})", ""]
    if a["world"]["world_id"] != b["world"]["world_id"]:
        L.append("**WARNING: different worlds; the numbers are not comparable pair by pair.**\n")
    for key in ("pairs", "depth"):
        da = get_path(a, f"world.caches.{key}.digest") if key == "pairs" else get_path(a, f"world.caches.{key}.params")
        db = get_path(b, f"world.caches.{key}.digest") if key == "pairs" else get_path(b, f"world.caches.{key}.params")
        if da != db:
            L.append(f"**WARNING: the {key} cache differs between A and B.**\n")
    if a["harness"].get("code_digest") != b["harness"].get("code_digest"):
        L.append("_Note: harness code digests differ._\n")
    L += ["| metric | A | B | B - A | better |", "|---|---|---|---|---|"]
    ra = {label: (val, better) for label, val, better in headline(a)}
    rb = {label: (val, better) for label, val, better in headline(b)}
    for label in list(ra) + [x for x in rb if x not in ra]:
        va, better = ra.get(label, (None, None))
        vb, better_b = rb.get(label, (None, None))
        better = better or better_b
        delta = None
        verdict = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            delta = vb - va
            # a verdict only for a difference above float noise (1e-3 relative)
            if abs(delta) > 1e-9 + 1e-3 * max(abs(va), abs(vb)):
                verdict = "B" if (delta > 0) == (better == "higher") else "A"
            else:
                delta = 0 if isinstance(delta, int) else 0.0
        L.append(f"| {label} | {_fmt(va)} | {_fmt(vb)} | {_fmt(delta)} | {verdict} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# j. measure


def _gpu_used_mb() -> float | None:
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        vals = [float(x) for x in out.split() if x.strip()]
        return sum(vals) if vals else None
    except Exception:  # noqa: BLE001
        return None


def _gpu_process_mb(pids: set[int]) -> float | None:
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return None
    total, known = 0.0, False
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid in pids:
            try:
                total += float(parts[1])
                known = True
            except ValueError:
                pass  # "[N/A]" under Windows WDDM
    return total if known else None


class _WddmGpuMemory:
    """Per-process dedicated GPU memory on Windows (WDDM), read through the PDH
    counter ``GPU Process Memory(*) / Dedicated Usage`` with ctypes -- the
    per-process figure nvidia-smi reports as [N/A] under WDDM. WDDM accounts
    residency its own way (the counter can exceed nvidia-smi's total across
    processes), so it is reported beside, never instead of, the device total.
    `sample` returns None when PDH is unavailable (any non-Windows host)."""

    def __init__(self) -> None:
        self.ok = False
        try:
            import ctypes
            from ctypes import wintypes

            self._ct, self._wt = ctypes, wintypes
            self._pdh = ctypes.WinDLL("pdh")

            class FMT(ctypes.Structure):
                _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]

            class ITEM(ctypes.Structure):
                _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", FMT)]

            self._ITEM = ITEM
            self._q = ctypes.c_void_p()
            self._c = ctypes.c_void_p()
            if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._q)) != 0:
                return
            if self._pdh.PdhAddEnglishCounterW(self._q, r"\GPU Process Memory(*)\Dedicated Usage", 0,
                                               ctypes.byref(self._c)) != 0:
                return
            self.ok = True
        except Exception:  # noqa: BLE001
            self.ok = False

    def sample(self, pids: set[int]) -> float | None:
        if not self.ok:
            return None
        ct, wt = self._ct, self._wt
        try:
            if self._pdh.PdhCollectQueryData(self._q) != 0:
                return None
            size, count = wt.DWORD(0), wt.DWORD(0)
            self._pdh.PdhGetFormattedCounterArrayW(self._c, 0x400, ct.byref(size), ct.byref(count), None)
            if not size.value:
                return None
            buf = (ct.c_byte * size.value)()
            if self._pdh.PdhGetFormattedCounterArrayW(self._c, 0x400, ct.byref(size), ct.byref(count), buf) != 0:
                return None
            items = ct.cast(buf, ct.POINTER(self._ITEM))
            total = 0
            for i in range(count.value):
                m = re.match(r"pid_(\d+)_", items[i].szName or "")
                if m and int(m.group(1)) in pids:
                    total += int(items[i].FmtValue.largeValue)
            return total / 2**20
        except Exception:  # noqa: BLE001
            return None


def measure_command(command: list[str], poll_s: float = 0.5) -> dict:
    """Run `command`; sample peak RSS of its process tree (psutil) and GPU memory.

    VRAM, three readings, because none is complete on a shared Windows GPU:
    `peak_vram_process_mb` (nvidia-smi per-process; [N/A] -> None under WDDM),
    `peak_vram_process_wddm_mb` (PDH per-process dedicated usage of the
    command's process tree; WDDM residency accounting), and the device total,
    whose peak minus the value at start is `peak_vram_delta_mb` -- an upper
    bound that includes every other process using the GPU meanwhile
    (`vram_other_users_possible`). A torch variant that wants an exact figure
    should also record `torch.cuda.max_memory_allocated()` in its own meta."""
    import subprocess

    import psutil

    start_used = _gpu_used_mb()
    t0 = time.time()
    proc = subprocess.Popen(command)
    ps = psutil.Process(proc.pid)
    peak_rss = 0
    peak_total = start_used
    peak_proc = None
    peak_wddm = None
    wddm = _WddmGpuMemory()
    samples = 0
    while True:
        try:
            tree = [ps] + ps.children(recursive=True)
        except psutil.Error:
            tree = []
        rss = 0
        for x in tree:
            try:
                rss += x.memory_info().rss
            except psutil.Error:
                pass
        peak_rss = max(peak_rss, rss)
        used = _gpu_used_mb()
        if used is not None:
            peak_total = used if peak_total is None else max(peak_total, used)
        pids = {x.pid for x in tree}
        pm = _gpu_process_mb(pids)
        if pm is not None:
            peak_proc = pm if peak_proc is None else max(peak_proc, pm)
        wm = wddm.sample(pids)
        if wm is not None:
            peak_wddm = wm if peak_wddm is None else max(peak_wddm, wm)
        samples += 1
        if proc.poll() is not None:
            break
        time.sleep(poll_s)
    wall = time.time() - t0
    return {"command": command, "returncode": proc.returncode, "wall_s": round(wall, 3),
            "peak_rss_mb": round(peak_rss / 2**20, 1),
            "peak_vram_process_mb": peak_proc,
            "peak_vram_process_wddm_mb": None if peak_wddm is None else round(peak_wddm, 1),
            "vram_total_at_start_mb": start_used, "peak_vram_total_mb": peak_total,
            "peak_vram_delta_mb": (None if start_used is None or peak_total is None else peak_total - start_used),
            "vram_other_users_possible": True, "poll_s": poll_s, "samples": samples,
            "started_at": round(t0, 3)}
