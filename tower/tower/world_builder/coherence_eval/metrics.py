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
checks this under random Sim(3) transforms. (Physical continuity, H5 below,
converts steps to metres with the component's own MoGe scale, which is a
ratio of two quantities in the same gauge -- also invariant.)

WHAT IS COUNTED, AND OVER WHAT. Pose metrics use the variant's PUBLISHED
keyframes (what it would show; `status` in the interchange format). Every
coherence FRACTION that could be improved by publishing less is ALSO reported
over a FIXED denominator -- all accepted keyframes of the world, with
unposed, unpublished, off-main and no-ratio keyframes counted as failures
(`fixed_denominator`) -- and `compare` withholds its verdict when coverage
changed. Region labels (`regions`/`revisits`) are forensic annotations used
ONLY in these reports; nothing here feeds anything back into a
reconstruction.

WHAT THE HARNESS CANNOT SEE. Where one image island (see b2) sits relative to
another when no verified image pair links them: those island pairs are
listed as "UNOBSERVED" and no number pretends otherwise. A misrotation of an
island about the vertical axis (tilt, b2, sees only non-vertical axes).

THE METRICS (keys in metrics.json; every threshold is in `PARAMS`)

a. registration  -- fractions of the world's accepted keyframes that are
   posed at all / published / published in the largest component.
b. components     -- components = the variant's stated frames (COLMAP models,
   solve components); a variant that states none gets the connected
   components of its covisibility graph (>= `covis_min_shared` shared
   points) when it has observations, else ONE frame. Sizes, joined /
   singleton counts, `main_runs` (how often capture order enters the main
   frame), and the covisibility components inside the main frame.
b2. islands       -- image-only: the connected components of the world's
   verified pair graph (base tier). For every pair of islands of >=
   `island_min_kf` keyframes: the verified pairs linking them by tier (the
   cross-island SIFT tier: strict = the base floors, relaxed-only; the
   optional EfficientLoFTR tier) and the variant's rotation / direction errors
   on the strict ones, or "UNOBSERVED". Tilt: the angle between each island's
   roll-free up and the main island's (`eval_placement`). See the
   `eval_placement` docstring for the level-head assumption.
   PLACEMENT PLAUSIBILITY per island (and per RIGID group of the variant:
   components of its own shared-3-D-point graph at >= 1 point, main
   component), against the island / group with the most main-component
   keyframes: tilt (|median roll| under the reference's roll-free up, <= 10
   deg), scale (triangulated log-ratio median vs the reference, within x1.25,
   with IQR), eye height (median camera height along the reference up, in
   metres via the reference's MoGe units-per-metre, <= 0.3 m beyond the
   reference's p10-p90 band; own spread reported). The assumptions (level
   head, level floor, one posture) and why each bound is physical are in
   `eval_placement.PLAUSIBILITY_BASIS`. Yaw about the vertical and horizontal
   position stay UNOBSERVABLE without image links.
c. continuity     -- per consecutive published pair (capture order) of one
   component: step ratio = |dc| / (local median per-keyframe step x index
   gap) over `jump_window` steps each side, floored at `jump_floor_frac` x the
   component's median; the same for speed with the world's keyframe receipt
   times. An index+time JUMP needs both ratios > `jump_ratio` (or a rotation >
   `rot_jump_deg` per gap and > `rot_jump_deg_per_s`). PHYSICAL (H5): the
   step in metres (the component's MoGe units_per_metre) above
   `walk_speed_max_mps` x dt + `walk_margin_m`, or a rotation above
   `head_turn_max_dps` x dt + `head_turn_margin_deg`, is physically
   implausible for a walking, head-turning wearer -- whatever the time gap.
   `jumps` = index+time OR physical. Steps ACROSS A TRACKING LOSS (a tracker
   segment boundary inside) are counted separately, with how many have no
   strict verified image pair linking the `evidence_window` keyframes before
   the step to those after it ("without image evidence": their relative
   placement is the solver's guess). SPIKE = jumps on both sides.
d. scale          -- per keyframe, a log-ratio r = log(z_variant / z_MoGe)
   from TWO sources, each centred per component:
   * `depth` (variant points): median over its own observations
     (reprojection <= `reproj_gate_px`, >= `depth_min_samples`); needs
     points.npz;
   * `depth_tri` (harness triangulation, H3): the cached image-only pair
     inliers triangulated with the variant's relative poses (Sampson <=
     `tri_sampson_gate_px`, triangulation angle >= `tri_min_angle_deg`);
     available for EVERY variant with poses, and not curatable by it.
   Per source: coverage (keyframes with a ratio / accepted), robust spread,
   Theil-Sen TREND (a step also produces a trend; `trend_factor_over_span`
   is not "drift"), two-window steps, levels, fixed-denominator coherence,
   per tracker segment (p10-p90 over segments with >= `segment_min_kf`),
   per region. LEVELS (H2): the level is the MODE of a Gaussian KDE of the
   main component's r (bandwidth `level_kde_bw_log`); on-level = within the
   band; two bands, x1.5 and x1.25. A x1.5 step is invisible to the x1.5 band
   by construction and is caught by the x1.25 band. FIXED DENOMINATOR (H1):
   coherent_all = keyframes published, in main, with a ratio and on-level /
   ALL accepted keyframes; longest_bad_run over all accepted keyframes.
   Limitation: MoGe's per-image metric error (a few %) is noise in r; only
   multi-keyframe statistics mean anything; mirrors and very near clutter can
   bias one region's level.
e. revisits       -- against the world's verified pair set (base tier), per
   pair with both keyframes published in ONE component: relative-rotation
   error, translation-direction error where the pair's own parallax (the
   image-only measure of baseline over scene depth) is >= `t_min_parallax_deg`,
   and Sampson error. Buckets (H4), all world-defined so every variant is
   scored on the same pairs: adjacent (gap 1); near (gap 2..`revisit_window`);
   revisit (DIFFERENT tracker segment AND >= `revisit_min_dt_s` apart AND gap
   > `revisit_window`); lingering (gap > `revisit_window` but not a revisit:
   same segment or too little time -- e.g. a slow final pass). `joined` =
   both published in one component / all pairs of the bucket. The travelled
   path is not a bucket criterion (it would make the pair set depend on the
   variant); the variant's metric path between revisit pairs is reported.
   `annotated`: C0's hand-annotated ranges (evaluation only).
f. reprojection   -- the variant's own observations: median, p90, p99, max,
   >3 px fraction, behind-camera count.
g. regions        -- per C0 region: posed / published / in main / clean (in
   main, not a spike, not an outlier camera, AND on the main scale level at
   x1.5, triangulated ratio) and scale factor vs main.
h. sanity         -- per component: camera distances from their geometric
   median; R90, max/R90, outlier cameras (> `outlier_k` x R90); points.
i. renders        -- C2's viewpoint rule on the variant's published poses,
   and (``--renders``) cameras / sparse points drawn with C2's renderer.
j. runtime        -- the world's recorded timings, or the variant's
   `meta.runtime` (see `measure_command`).
k. integrity      -- keyframe references the variant made that do not
   resolve in this world (unresolved rows, unmatched COLMAP images).
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

from tower.world_builder.coherence_eval.eval_placement import (
    PLACEMENT_PARAMS,
    PLAUSIBILITY,
    PLAUSIBILITY_BASIS,
    group_placement,
    island_report,
    rigid_groups,
    triangulated_depth_ratios,
)

HARNESS = "wb-coherence-metrics/2"

PARAMS = {
    "jump_window": 10,
    "jump_ratio": 5.0,
    "jump_floor_frac": 0.25,
    "rot_jump_deg": 30.0,
    "rot_jump_deg_per_s": 250.0,
    "dt_floor_s": 0.05,
    "walk_speed_max_mps": 2.0,
    "walk_margin_m": 0.3,
    "head_turn_max_dps": 300.0,
    "head_turn_margin_deg": 30.0,
    "evidence_window": 5,
    "outlier_k": 3.0,
    "revisit_window": 30,
    "revisit_min_dt_s": 10.0,
    "t_min_parallax_deg": 5.0,
    "reproj_gate_px": 4.0,
    "depth_min_samples": 10,
    "depth_valid_m": [0.05, 50.0],
    "scale_step_window": 8,
    "scale_step_log": round(math.log(1.25), 6),
    "level_bands": [1.5, 1.25],
    "level_kde_bw_log": 0.05,
    "level_kde_grid_log": 0.0025,
    "d3_step_window": 15,
    "segment_min_kf": 10,
    "covis_min_shared": 15,
    "min_component_for_stats": 5,
    "rot_err_thresholds_deg": [2.0, 5.0, 10.0],
    "t_err_thresholds_deg": [10.0, 30.0],
    "worst_list": 15,
    "compare_coverage_tol": 0.02,
    "compare_ratio_coverage_tol": 0.05,
    **{f"placement.{k}": v for k, v in PLACEMENT_PARAMS.items()},
    **{f"plausibility.{k}": v for k, v in PLAUSIBILITY.items()},
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
    coordinate-wise median), which keeps the sanity metrics gauge-invariant.
    Iterates on the centred cloud so the stopping tolerance does not depend
    on the absolute translation (review V2, L5)."""
    X = np.asarray(X, dtype=np.float64)
    c0 = X.mean(0)
    Xc = X - c0
    y = np.zeros(3)
    scale = np.abs(Xc).max() + 1e-300
    for _ in range(iters):
        d = np.linalg.norm(Xc - y, axis=1)
        d = np.maximum(d, 1e-12 * scale)
        w = 1.0 / d
        y_new = (Xc * w[:, None]).sum(0) / w.sum()
        if np.linalg.norm(y_new - y) <= tol * scale:
            y = y_new
            break
        y = y_new
    return y + c0


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


def _longest_run(bad) -> int:
    run = best = 0
    for b in bad:
        run = run + 1 if b else 0
        best = max(best, run)
    return int(best)


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


def evidence_matrix(n: int, arrays_list) -> np.ndarray:
    """n x n symmetric boolean matrix of STRICT verified image pairs (all tiers)."""
    M = np.zeros((n, n), bool)
    for A in arrays_list:
        if A is None or not len(A["i"]):
            continue
        sel = np.asarray(A["strict"], bool) if "strict" in A else np.ones(len(A["i"]), bool)
        I = np.asarray(A["i"])[sel].astype(int)
        J = np.asarray(A["j"])[sel].astype(int)
        M[I, J] = True
        M[J, I] = True
    return M


def _continuity_for(p: Prepared, members: np.ndarray, upm: float | None, evidence: np.ndarray | None) -> dict:
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
    # physical bound (H5): metres via the component's own MoGe scale
    d_m = None
    phys = np.zeros(len(d), bool)
    if timed and upm is not None and upm > 0:
        d_m = d / upm
        phys_t = d_m > PARAMS["walk_speed_max_mps"] * dt + PARAMS["walk_margin_m"]
        phys_r = rot > PARAMS["head_turn_max_dps"] * dt + PARAMS["head_turn_margin_deg"]
        phys = phys_t | phys_r
    any_j = tj | rj | phys
    any_idx = tj_idx | rj_idx
    spikes = [int(members[j]) for j in range(1, len(members) - 1) if any_j[j - 1] and any_j[j]]
    endpoints = sorted({int(members[j]) for j in np.nonzero(any_j)[0]}
                       | {int(members[j + 1]) for j in np.nonzero(any_j)[0]})
    E = PARAMS["evidence_window"]

    def boundary(j):
        if p.segment_of is None:
            return None
        a, b = int(members[j]), int(members[j + 1])
        return len(set(p.segment_of[a:b + 1])) > 1

    def has_evidence(j):
        if evidence is None:
            return None
        a, b = int(members[j]), int(members[j + 1])
        return bool(evidence[max(0, a - E):a + 1, b:b + E + 1].any())

    loss = [j for j in range(len(d)) if boundary(j)]
    loss_noev = [j for j in loss if has_evidence(j) is False]
    listed = sorted(set(np.nonzero(any_idx | any_j)[0].tolist()) | set(loss))
    jumps = []
    for j in listed:
        a, b = int(members[j]), int(members[j + 1])
        jumps.append({"from": a, "to": b, "from_id": p.ids[a], "to_id": p.ids[b], "gap": int(b - a),
                      "ratio": float(ratio[j]),
                      "speed_ratio": None if sratio is None else float(sratio[j]),
                      "dt_s": None if dt is None else float(dt[j]),
                      "step_m": None if d_m is None else float(d_m[j]),
                      "rot_deg": float(rot[j]),
                      "counted": bool(any_j[j]),
                      "physically_implausible": bool(phys[j]),
                      "kind": "+".join(k for k, f in (("translation", tj_idx[j]), ("rotation", rj_idx[j]),
                                                     ("physical", phys[j])) if f),
                      "segment_boundary_inside": boundary(j),
                      "image_evidence_across": has_evidence(j)})
    counted = [x for x in jumps if x["counted"]]
    r90 = float(np.percentile(np.linalg.norm(C - geometric_median(C), axis=1), 90))
    return {
        "criterion": ("index+time or physical" if timed and d_m is not None else
                      "index+time" if timed else "index-only"),
        "steps": int(len(d)),
        "jumps": int(any_j.sum()),
        "jumps_index_time": int((tj | rj).sum()),
        "translation_jumps": int(tj.sum()),
        "rotation_jumps": int(rj.sum()),
        "physically_implausible": None if d_m is None else int(phys.sum()),
        "jumps_within_segments": int(sum(1 for x in counted if x["segment_boundary_inside"] is False)),
        "jumps_at_segment_boundary": int(sum(1 for x in counted if x["segment_boundary_inside"])),
        "tracking_loss_steps": len(loss) if p.segment_of is not None else None,
        "tracking_loss_steps_without_image_evidence": (len(loss_noev) if evidence is not None
                                                       and p.segment_of is not None else None),
        "tracking_loss_steps_without_evidence_implausible": (
            int(sum(1 for j in loss_noev if phys[j])) if d_m is not None else None),
        "spikes": len(spikes),
        "index_jumps": int(any_idx.sum()),
        "max_ratio": float(np.max(ratio)) if len(ratio) else None,
        "ratio": stats(ratio),
        "speed_ratio": None if sratio is None else stats(sratio),
        "step_m": None if d_m is None else stats(d_m),
        "units_per_metre_used": upm,
        "rot_step_deg": stats(rot),
        "path_length_over_r90": float(d.sum() / r90) if r90 > 0 else None,
        "path_length_m": None if d_m is None else float(d_m.sum()),
        "jump_list": jumps[:100],
        "spike_keyframes": spikes,
        "_endpoints": endpoints,
    }


def continuity(p: Prepared, upm_by_comp: dict, evidence) -> dict:
    per = {}
    for c in p.order:
        m = p.members(c)
        if len(m) >= 3:
            per[c] = _continuity_for(p, m, upm_by_comp.get(c), evidence)
    main = per.get(p.main) if p.main is not None else None

    def total(key):
        vals = [x.get(key) for x in per.values()]
        return None if any(v is None for v in vals) else int(sum(vals))

    return {"main": main,
            "all_components": {
                "components_evaluated": len(per),
                "jumps": total("jumps"),
                "spikes": total("spikes"),
                "physically_implausible": total("physically_implausible"),
                "tracking_loss_steps": total("tracking_loss_steps"),
                "tracking_loss_steps_without_image_evidence": total("tracking_loss_steps_without_image_evidence"),
            },
            "per_component": {c: {k: x.get(k) for k in ("steps", "jumps", "spikes", "max_ratio",
                                                        "physically_implausible")}
                              for c, x in per.items()}}


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
    """Per published keyframe: median log(z_sfm / z_mono) over the variant's own
    observations, sample count, and the within-keyframe spread (1.4826 MAD)."""
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


def level_mode(y) -> float:
    """The scale LEVEL: the mode of a Gaussian KDE of y (bandwidth
    `level_kde_bw_log`, grid `level_kde_grid_log`; ties -> the lowest grid
    value). Unlike "the band placement that covers most keyframes", a mode
    cannot sit BETWEEN two levels, so a band around it separates them
    (review V2, H2)."""
    y = np.asarray(y, dtype=np.float64)
    h = PARAMS["level_kde_bw_log"]
    g = np.arange(y.min() - 2 * h, y.max() + 2 * h + PARAMS["level_kde_grid_log"], PARAMS["level_kde_grid_log"])
    dens = np.zeros(len(g))
    for k in range(0, len(y), 512):
        dens += np.exp(-0.5 * ((g[:, None] - y[None, k:k + 512]) / h) ** 2).sum(1)
    return float(g[int(np.argmax(dens))])


def level_stats(y_ordered) -> dict | None:
    """Scale-level numbers on a log-scale sequence in capture order (shift-
    invariant, hence gauge-invariant on one component).

    * level: `level_mode`;
    * per band B in `level_bands`: coherent_fraction = share within xB of the
      level; longest_off_level_run = longest run of consecutive off-level
      entries;
    * max_step_15kf: exp of the largest |median(next w) - median(previous w)|,
      w = `d3_step_window`, over every split (research lane D3's number).
    The fractions here are over the entries GIVEN; the fixed-denominator
    versions (over all accepted keyframes) are in `scale_block`."""
    y = np.asarray(y_ordered, dtype=np.float64)
    if len(y) < 3:
        return None
    mode = level_mode(y)
    out = {"level_log": mode, "bands": {}}
    for b in PARAMS["level_bands"]:
        ok = np.abs(y - mode) <= math.log(b)
        out["bands"][f"x{b:g}"] = {"coherent_fraction": float(ok.mean()), "off_level": int((~ok).sum()),
                                   "longest_off_level_run": _longest_run(~ok)}
    w = PARAMS["d3_step_window"]
    step, at = 0.0, None
    for i in range(w, len(y) - w + 1):
        d = abs(float(np.median(y[i:i + w]) - np.median(y[i - w:i])))
        if d > step and d > 1e-9:  # float noise is not a step
            step, at = d, i
    out["max_step_15kf_factor"] = float(math.exp(step))
    out["_max_step_pos"] = at
    return out


def scale_block(p: Prepared, ratios, regions: dict | None, source: str) -> dict:
    """Every scale statistic for one per-keyframe log-ratio source."""
    if ratios is None:
        return {"available": False, "source": source}
    r = ratios["r"]
    n_acc = max(p.n, 1)
    has = np.isfinite(r)
    blk: dict = {"available": True, "source": source,
                 "keyframes_with_ratio": int(has.sum()),
                 "ratio_coverage": float(has.sum() / n_acc),
                 "ratio_coverage_published": float((has & p.published).sum() / max(1, p.published.sum())),
                 "median_samples_per_keyframe": float(np.median(ratios["n"][has])) if has.any() else None,
                 "within_keyframe_spread_median": (float(np.nanmedian(ratios["spread"]))
                                                   if np.isfinite(ratios["spread"]).any() else None),
                 "per_component": {}}
    for k in ("pairs_used", "inliers_total", "inliers_gated"):
        if k in ratios:
            blk[k] = int(ratios[k])
    centred = np.full(p.n, np.nan)
    upm = {}
    for c in p.order:
        m = p.members(c)
        m = m[np.isfinite(r[m])]
        if len(m) < PARAMS["min_component_for_stats"]:
            continue
        med = float(np.median(r[m]))
        y = r[m] - med
        centred[m] = y
        upm[c] = float(math.exp(med))
        row = {"keyframes": int(len(m)), "spread_mad": mad(y),
               "p05_p95": [float(np.percentile(y, 5)), float(np.percentile(y, 95))],
               "units_per_metre_informative": float(math.exp(med))}
        if len(m) >= 10:
            slope, lo, hi = _theil_sen(m.astype(float), y)
            span = float(m.max() - m.min())
            row["trend_slope_per_100kf"] = slope * 100.0
            row["trend_slope_ci95_per_100kf"] = [lo * 100.0, hi * 100.0]
            row["trend_factor_over_span"] = float(math.exp(abs(slope * span)))
            steps = _steps(m, y, PARAMS["scale_step_window"], PARAMS["scale_step_log"])
            row["steps"] = len(steps)
            row["max_step_factor"] = max((s["factor"] for s in steps), default=1.0)
            row["step_list"] = steps[:20]
        lv = level_stats(y)
        if lv is not None:
            pos = lv.pop("_max_step_pos")
            lv["max_step_15kf_at_index"] = int(m[pos]) if pos is not None else None
            row["levels"] = lv
        blk["per_component"][c] = row
    main = blk["per_component"].get(p.main)
    blk["main"] = main
    # FIXED DENOMINATOR (H1): over ALL accepted keyframes; anything not
    # published-in-main-with-a-ratio-on-level counts as a failure.
    fixed = {"definition": "good = published & in main & has ratio & |r - level| <= log(band); "
                           "denominator = all accepted keyframes", "bands": {}}
    on = {}
    if main is not None and "levels" in main:
        mode = main["levels"]["level_log"]
        for b in PARAMS["level_bands"]:
            good = p.in_main & has & (np.abs(np.where(has, centred, np.inf) - mode) <= math.log(b))
            on[b] = good
            fixed["bands"][f"x{b:g}"] = {"coherent_all": float(good.sum() / n_acc),
                                         "longest_bad_run": _longest_run(~good)}
    else:
        for b in PARAMS["level_bands"]:
            fixed["bands"][f"x{b:g}"] = {"coherent_all": 0.0, "longest_bad_run": int(p.n)}
            on[b] = np.zeros(p.n, bool)
    blk["fixed_denominator"] = fixed
    # D3's pilot pooled every fitted keyframe of the world, whatever its
    # component: gauge-DEPENDENT across components; kept only for D3 parity.
    allp = np.nonzero(p.published & has)[0]
    lv = level_stats(r[allp]) if len(allp) >= 3 else None
    if lv is not None:
        pos = lv.pop("_max_step_pos")
        lv["max_step_15kf_at_index"] = int(allp[pos]) if pos is not None else None
        lv["keyframes"] = int(len(allp))
        lv["note"] = "all published keyframes pooled across components (gauge-dependent; D3-comparable)"
    blk["d3_pooled_informative"] = lv
    if main is not None and p.segment_of is not None:
        by_seg: dict[int, list[float]] = {}
        for i in p.members(p.main):
            if np.isfinite(centred[i]):
                by_seg.setdefault(int(p.segment_of[i]), []).append(centred[i])
        seg_med = {s: float(np.median(v)) for s, v in by_seg.items() if len(v) >= PARAMS["segment_min_kf"]}
        if seg_med:
            vals = np.array(list(seg_med.values()))
            blk["by_tracker_segment"] = {
                "segments": len(seg_med), "min_keyframes_per_segment": PARAMS["segment_min_kf"],
                "log_spread_mad": mad(vals),
                "p10_p90_factor": float(math.exp(np.percentile(vals, 90) - np.percentile(vals, 10))),
                "max_over_min_factor": float(math.exp(vals.max() - vals.min())),
                "median_log_by_segment": {str(s): seg_med[s] for s in sorted(seg_med)}}
    if main is not None and regions:
        by_reg: dict[str, list[int]] = {}
        for i in range(p.n):
            reg = regions.get(p.ids[i])
            if reg:
                by_reg.setdefault(reg["region"], []).append(i)
        rows = {}
        for g, idx in sorted(by_reg.items()):
            idx = np.array(idx)
            v = centred[idx][np.isfinite(centred[idx]) & p.in_main[idx]]
            rows[g] = {"keyframes": int(len(idx)), "with_ratio_in_main": int(len(v)),
                       "median_log": float(np.median(v)) if len(v) else None,
                       "factor_vs_main": float(math.exp(np.median(v))) if len(v) else None,
                       "spread_mad": mad(v) if len(v) else None,
                       **{f"on_level_x{b:g}_of_region": float(on[b][idx].mean()) for b in PARAMS["level_bands"]}}
        blk["by_region"] = rows
    blk["_centred"] = centred
    blk["_on_level"] = on
    blk["_upm"] = upm
    return blk


def placement_scales(p: Prepared) -> dict | None:
    segs = p.v.segments
    if not segs:
        return None
    sc = np.array([float(s["scale"]) for s in segs
                   if s.get("scale") not in (None, 0) and s.get("state", "registered") == "registered"])
    if not len(sc):
        return None
    ls = np.log(sc)
    return {"segments": int(len(sc)), "log_std": float(ls.std()), "max_over_min": float(sc.max() / sc.min()),
            "note": ("all exactly 1: every registered segment is a rigid piece of one global model"
                     if np.allclose(sc, 1.0) else "Sim(3) placement scales differ between segments")}


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


BUCKETS = ("adjacent", "near", "lingering", "revisit")


def pair_buckets(I, J, segment_of, times) -> np.ndarray:
    """World-defined buckets (H4): the same for every variant of a world."""
    I = np.asarray(I, int)
    J = np.asarray(J, int)
    gap = J - I
    W = PARAMS["revisit_window"]
    far = gap > W
    if segment_of is not None and times is not None:
        seg = np.asarray(segment_of)
        tt = np.asarray(times, dtype=np.float64)
        rev = far & (seg[I] != seg[J]) & (np.abs(tt[J] - tt[I]) >= PARAMS["revisit_min_dt_s"])
    else:
        rev = far
    return np.where(gap == 1, "adjacent", np.where(~far, "near", np.where(rev, "revisit", "lingering")))


def revisits(p: Prepared, pairs, regions: dict | None, focal_px: float | None, path_m=None) -> dict:
    if pairs is None or not getattr(pairs, "available", False):
        return {"available": False, "why": "no verified pair cache for this world"}
    A = pairs.arrays
    I, J = A["i"].astype(int), A["j"].astype(int)
    bucket = pair_buckets(I, J, p.segment_of, p.times)
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
    out: dict = {"available": True, "pairs": int(len(I)), "pair_cache_digest": pairs.digest(),
                 "bucket_definition": {
                     "adjacent": "capture-order gap 1",
                     "near": f"gap 2..{PARAMS['revisit_window']}",
                     "revisit": (f"gap > {PARAMS['revisit_window']} AND different tracker segment AND "
                                 f">= {PARAMS['revisit_min_dt_s']:g} s apart"),
                     "lingering": f"gap > {PARAMS['revisit_window']} but not a revisit"}}
    for b in BUCKETS:
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
    if path_m is not None:
        rv = np.nonzero((bucket == "revisit") & main)[0]
        pm = np.array([path_m(int(I[k]), int(J[k])) for k in rv], dtype=np.float64)
        pm = pm[np.isfinite(pm)] if len(pm) else pm
        out["revisit"]["variant_path_m"] = stats(pm)
        out["revisit"]["variant_path_under_1m"] = int((pm < 1.0).sum()) if len(pm) else 0
    rv = np.nonzero((bucket == "revisit") & same)[0]
    worst = sorted(rv, key=lambda k: (-round(float(rot_err[k]), 3), int(I[k]), int(J[k])))[: PARAMS["worst_list"]]
    out["worst_revisit"] = [
        {"i": int(I[k]), "j": int(J[k]), "rot_err_deg": float(rot_err[k]),
         "t_err_deg": (float(t_err[k]) if np.isfinite(t_err[k]) else None),
         "inliers": int(A["n_inliers"][k]), "parallax_deg": float(A["parallax_deg"][k]),
         "regions": ([regions.get(p.ids[I[k]], {}).get("region"), regions.get(p.ids[J[k]], {}).get("region")]
                     if regions else None)}
        for k in worst]
    if regions:
        by: dict[str, list[int]] = {}
        for k in np.nonzero(np.isin(bucket, ("revisit", "lingering")))[0]:
            ri = (regions.get(p.ids[I[k]]) or {}).get("region")
            rj = (regions.get(p.ids[J[k]]) or {}).get("region")
            key = (ri if ri == rj and ri else "cross-region") + f" ({bucket[k]})"
            by.setdefault(key, []).append(k)
        out["by_region"] = {
            g: {"pairs": len(ks), "joined_fraction": float(np.mean(same[ks])),
                "rot_err_median_deg": (float(np.nanmedian(rot_err[ks])) if np.isfinite(rot_err[ks]).any() else None),
                "t_err_median_deg": (float(np.nanmedian(t_err[ks])) if np.isfinite(t_err[ks]).any() else None),
                "t_err_count": int(np.isfinite(t_err[ks]).sum())}
            for g, ks in sorted(by.items())}
    out["_arrays"] = {"I": I, "J": J, "same": same, "rot_err": rot_err, "t_err": t_err, "bucket": bucket}
    return out


def annotated_revisits(p: Prepared, rows: list[dict] | None, rev: dict | None, island_lab=None) -> dict:
    if not rows:
        return {"available": False, "why": "no annotated revisit ranges for this world"}
    main = p.members(p.main) if p.main is not None else np.zeros(0, int)
    r90 = _radii(p.C[main])[1] if len(main) >= 3 else None
    out = []
    arr = (rev or {}).get("_arrays")
    for row in rows:
        a0, a1, b0, b1 = row["a"][0], row["a"][1], row["b"][0], row["b"][1]
        A = np.arange(a0, min(a1, p.n - 1) + 1)
        B = np.arange(b0, min(b1, p.n - 1) + 1)
        Am = A[p.in_main[A]] if len(A) else A
        Bm = B[p.in_main[B]] if len(B) else B
        rec = {"a": [int(a0), int(a1)], "b": [int(b0), int(b1)], "region": row.get("region"),
               "confidence": row.get("confidence"),
               "a_in_main": float(len(Am) / len(A)) if len(A) else None,
               "b_in_main": float(len(Bm) / len(B)) if len(B) else None}
        if island_lab is not None and len(A) and len(B):
            ia = sorted({int(x) for x in island_lab[A]})
            ib = sorted({int(x) for x in island_lab[B]})
            rec["same_image_island"] = bool(set(ia) & set(ib))
        if len(Am) and len(Bm) and r90:
            D = np.linalg.norm(p.C[Am][:, None, :] - p.C[Bm][None, :, :], axis=2)
            rec["closest_over_r90"] = float(D.min() / r90)
            rec["centroid_over_r90"] = float(np.linalg.norm(p.C[Am].mean(0) - p.C[Bm].mean(0)) / r90)
        else:
            rec["closest_over_r90"] = None
            rec["centroid_over_r90"] = None
        if arr is not None:
            inA = (arr["I"] >= a0) & (arr["I"] <= a1) & (arr["J"] >= b0) & (arr["J"] <= b1)
            inA |= (arr["J"] >= a0) & (arr["J"] <= a1) & (arr["I"] >= b0) & (arr["I"] <= b1)
            rec["verified_pairs_between"] = int(inA.sum())
            rec["verified_pairs_joined"] = int((inA & arr["same"]).sum())
            e = arr["rot_err"][inA & arr["same"]]
            rec["rot_err_median_deg"] = float(np.median(e)) if len(e) else None
        out.append(rec)
    closest = [r["closest_over_r90"] for r in out if r["closest_over_r90"] is not None]
    unobserved = sum(1 for r in out if r.get("same_image_island") is False)
    return {"available": True, "ranges": len(out), "both_in_main": len(closest),
            "ranges_in_different_image_islands": unobserved,
            "closest_over_r90_median": float(np.median(closest)) if closest else None,
            "closest_over_r90_max": float(max(closest)) if closest else None,
            "note": ("closest/centroid distances between ranges in DIFFERENT image islands are the solver's "
                     "guess, not a measurement; two ranges can see one place from different positions"),
            "list": out}


# ---------------------------------------------------------------------------
# g. regions


def region_coverage(p: Prepared, regions: dict | None, spikes, outliers, endpoints, on_level, level_src) -> dict:
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
        lvl = on_level[idx] if on_level is not None else np.zeros(n, bool)
        clean = np.array([in_main[k] and idx[k] not in spikes and idx[k] not in outliers and lvl[k]
                          for k in range(n)])
        out[g] = {"keyframes": n, "posed": float(p.posed[idx].mean()),
                  "published": float(p.published[idx].mean()),
                  "in_main": float(in_main.mean()), "in_main_clean": float(clean.mean()),
                  "on_level": float(lvl.mean()),
                  "jump_endpoints": int(sum(1 for i in idx if i in endpoints)),
                  "spikes": int(sum(1 for i in idx if i in spikes)),
                  "outlier_cameras": int(sum(1 for i in idx if i in outliers))}
    return {"available": True, "clean_definition": (
        f"in main, not a spike, not an outlier camera, and on the main scale level within x1.5 ({level_src})"),
            "regions": out}


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


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def evaluate(variant, keyframe_ids, *, segment_of=None, times=None, pairs=None, depth_fn=None,
             depth_sample=None, K=None, regions=None, revisit_ranges=None, focal_px=None) -> dict:
    """Every pose/structure metric for one variant. Pure: no IO.

    keyframe_ids / segment_of / times describe the WORLD (capture order, the
    frontend's tracker segments, keyframe receipt times); they are the same for
    every variant of that world. `depth_fn(i, uv_variant_pixels, cam)` samples
    the depth cache for the variant's own observations; `depth_sample(i,
    uv_canonical)` samples it at canonical pixels (triangulation); `K` is the
    canonical camera the pair inliers are normalised in.

    A variant that does not STATE its components gets them from its
    observations (connected components of the covisibility graph, >=
    `covis_min_shared` shared points), else all its poses are one frame."""
    if not variant.component_stated:
        labels = covisibility_labels(variant, keyframe_ids, PARAMS["covis_min_shared"])
        if labels is not None:
            import copy

            variant = copy.copy(variant)
            variant.component = dict(labels)
            variant.meta = dict(variant.meta, _component_definition=(
                f"not stated: covisibility components (>= {PARAMS['covis_min_shared']} shared points)"))
    p = Prepared(variant, keyframe_ids, segment_of, times)
    have_pairs = pairs is not None and getattr(pairs, "available", False)
    table = observation_table(p)
    r_obs = keyframe_depth_ratios(p, table, depth_fn)
    r_tri = None
    if have_pairs and depth_sample is not None and K is not None:
        r_tri = triangulated_depth_ratios(p, pairs.arrays, depth_sample, np.asarray(K, dtype=np.float64),
                                          min_samples=PARAMS["depth_min_samples"],
                                          valid_m=PARAMS["depth_valid_m"])
    sc_obs = scale_block(p, r_obs, regions, "variant observations")
    sc_tri = scale_block(p, r_tri, regions, "harness triangulation of cached pair inliers")
    # metric scale per component for the physical bound: the variant's own
    # observations when present, else the triangulation
    upm = dict(sc_tri.get("_upm") or {})
    upm.update(sc_obs.get("_upm") or {})
    evidence = None
    isl = None
    if have_pairs:
        evidence = evidence_matrix(p.n, [pairs.arrays, getattr(pairs, "xarrays", None),
                                         getattr(pairs, "larrays", None)])
        use = r_tri if r_tri is not None and np.isfinite(r_tri["r"]).any() else r_obs
        isl = island_report(p, pairs, rot_angle_deg, vec_angle_deg, PARAMS["t_min_parallax_deg"],
                            r_log=None if use is None else use["r"],
                            r_source=None if use is None else ("triangulated" if use is r_tri else "variant points"))
    cont = continuity(p, upm, evidence)
    san = sanity(p)
    rig = rigid_groups(p)
    use = r_tri if r_tri is not None and np.isfinite(r_tri["r"]).any() else r_obs
    if rig is None:
        rigid = {"available": False, "why": "the variant has no observations (rigid groups need shared 3-D points)"}
    else:
        rigid = {"count": len(rig), "sizes": [int(len(v)) for v in rig.values()][:20],
                 "min_shared_points": 1,
                 "placement_plausibility": group_placement(
                     p, rig, "rigid0", None if use is None else use["r"],
                     None if use is None else ("triangulated" if use is r_tri else "variant points"),
                     kind="rigid groups (shared 3-D points >= 1, main component)")}

    def path_m(i, j):
        c = p.comp[i]
        u = upm.get(c)
        if u is None or c != p.comp[j]:
            return float("nan")
        m = p.members(c)
        m = m[(m >= i) & (m <= j)]
        return float(np.linalg.norm(np.diff(p.C[m], axis=0), axis=1).sum() / u) if len(m) > 1 else float("nan")

    rev = revisits(p, pairs, regions, focal_px, path_m=path_m if upm else None)
    ann = annotated_revisits(p, revisit_ranges, rev, isl.get("_labels") if isl else None)
    main_c = cont.get("main") or {}
    lvl_src = sc_tri if sc_tri.get("available") and sc_tri.get("main") else sc_obs
    on_level = (lvl_src.get("_on_level") or {}).get(1.5)
    reg = region_coverage(p, regions, main_c.get("spike_keyframes", []), san.get("_outliers_main", []),
                          main_c.get("_endpoints", []), on_level, lvl_src.get("source"))
    if isinstance(rev, dict):
        rev = _strip(rev)
        rev["annotated"] = ann
    if cont.get("main"):
        cont["main"].pop("_endpoints", None)
    san.pop("_outliers_main", None)
    per_kf = {"depth_log_ratio_obs": (None if r_obs is None else
                                      [None if not np.isfinite(x) else float(x) for x in r_obs["r"]]),
              "depth_log_ratio_tri": (None if r_tri is None else
                                      [None if not np.isfinite(x) else float(x) for x in r_tri["r"]])}
    scale_out = {"placement_scales": placement_scales(p), "depth": _strip(sc_obs), "depth_tri": _strip(sc_tri)}
    return {
        "registration": registration(p),
        "components": components(p),
        "islands": _strip(isl) if isl else {"available": False, "why": "no verified pair cache for this world"},
        "rigid_groups": rigid,
        "continuity": cont,
        "scale": scale_out,
        "revisits": rev,
        "reprojection": reprojection(p, table),
        "regions": reg,
        "sanity": san,
        "integrity": {"unresolved_keyframes": int(variant.meta.get("unresolved_keyframes") or 0),
                      "unmatched_images": int(variant.meta.get("unmatched_images") or 0)},
        "_per_keyframe": per_kf,
    }


def _git_head(path: Path) -> dict:
    import subprocess

    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--",
                                "tower/tower/world_builder/coherence_eval", "tower/scripts/world_coherence_eval.py"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return {"commit": head or None, "harness_files_uncommitted": bool(dirty)}
    except Exception:  # noqa: BLE001
        return {"commit": None}


HARNESS_FILES = ("metrics.py", "viewpoints.py", "layer_renders.py")


def _harness_digest() -> str:
    """SHA-1 over every file the numbers and renders depend on: metrics.py,
    every eval_*.py, C2's viewpoints.py and layer_renders.py, and the CLI
    (review V2, L4)."""
    h = hashlib.sha1()
    here = Path(__file__).resolve().parent
    names = sorted(p.name for p in here.glob("*.py") if p.name in HARNESS_FILES or p.name.startswith("eval_"))
    for name in names:
        h.update(name.encode())
        h.update((here / name).read_bytes())
    cli = here.parents[2] / "scripts" / "world_coherence_eval.py"
    if cli.is_file():
        h.update(b"world_coherence_eval.py")
        h.update(cli.read_bytes())
    return h.hexdigest()[:16]


def evaluate_world_variant(world, variant, *, cache_root, regions_dir=None, renders=False, out_dir=None,
                           viewpoints_from=None) -> dict:
    """Glue: caches + labels + runtime around `evaluate`."""
    from tower.world_builder.coherence_eval.eval_depth import DepthCache, depth_dir
    from tower.world_builder.coherence_eval.eval_pairs import PairSet, pairs_dir
    from tower.world_builder.coherence_eval.eval_world import CANONICAL, RAW

    depth = DepthCache(depth_dir(cache_root, world.world_id))
    pairs = PairSet(pairs_dir(cache_root, world.world_id))
    depth_fn = None
    depth_sample = None
    depth_note = None
    K_can = world.canonical_K()
    if depth.available:
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

        def depth_sample(i, uv):
            return depth.sample(world.image_name(i), uv)
    else:
        depth_note = "depth cache missing or incomplete: run `world_coherence_eval.py cache --what depth`"
    regions = revisit_rows = None
    if regions_dir is not None:
        regions = read_regions(Path(regions_dir) / f"{world.world_id}_regions.csv")
        revisit_rows = read_revisits(Path(regions_dir) / f"{world.world_id}_revisits.csv")
    segment_of = [int(k.get("segment_index", -1)) for k in world.keyframes]
    times = [float(k["received_at"]) if k.get("received_at") is not None else float("nan")
             for k in world.keyframes]
    t0 = time.time()
    body = evaluate(variant, world.keyframe_ids, segment_of=segment_of, times=times,
                    pairs=pairs if pairs.available else None, depth_fn=depth_fn, depth_sample=depth_sample,
                    K=K_can, regions=regions, revisit_ranges=revisit_rows,
                    focal_px=float(world.canonical_camera["fx"]))
    eval_seconds = time.time() - t0
    if depth_note:
        body["scale"]["depth"]["note"] = depth_note
        body["scale"]["depth_tri"]["note"] = depth_note
    body["renders"] = render_block(world, variant, out_dir, renders, regions=regions,
                                   viewpoints_from=viewpoints_from)
    body["runtime"] = {"world_recorded": world.runtime_evidence() if variant.meta.get("adapter") == "world" else None,
                       "variant_measured": variant.meta.get("runtime")}
    meta = {k: v for k, v in variant.meta.items() if k not in ("runtime",) and not k.startswith("_")}
    repo = Path(__file__).resolve().parents[4]
    result = {
        "harness": {"version": HARNESS, "params": PARAMS, "code_digest": _harness_digest(), **_git_head(repo)},
        "world": {"world_id": world.world_id, "session_id": world.session_id, "keyframes": world.n,
                  "world_dir": str(world.world_dir), "canonical_camera": world.canonical_camera,
                  "caches": {"depth": {"available": depth.available,
                                       "params": (depth.manifest or {}).get("params"),
                                       "digest": depth.digest() if depth.available else None},
                             "pairs": {"available": pairs.available, "count": len(pairs),
                                       "digest": pairs.digest(),
                                       "counts": (pairs.manifest or {}).get("counts"),
                                       "params": (pairs.manifest or {}).get("params"),
                                       "cross_island_sift": (None if pairs.xmanifest is None else
                                                             {"digest": pairs.xdigest(),
                                                              "counts": pairs.xmanifest.get("counts"),
                                                              "params": pairs.xmanifest.get("params")}),
                                       "cross_island_learned": (None if pairs.lmanifest is None else
                                                                {"counts": pairs.lmanifest.get("counts"),
                                                                 "params": pairs.lmanifest.get("params")})}},
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


def render_block(world, variant, out_dir, do_render: bool, regions=None, viewpoints_from=None) -> dict:
    """Viewpoints from C2's deterministic rule (`viewpoints`, rule /2), and
    renders of the variant's main component with C2's renderer.

    Native mode applies the rule to THIS variant's published poses. For A/B
    renders pass `viewpoints_from` (a viewpoint-set JSON, e.g. the baseline's
    `viewpoints.json`): the reference set is carried into this variant's gauge
    by `viewpoints.transfer_viewpoints` (a Sim(3) on shared keyframes), so both
    variants are rendered from the same physical viewpoints. The TRAJ
    substitution count is itself a coverage signal."""
    block: dict = {}
    try:
        from tower.world_builder.coherence_eval import viewpoints as vp
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "why": f"viewpoints module unavailable: {type(exc).__name__}"}
    poses = {k: np.asarray(T) for k, T in variant.poses.items() if variant.status.get(k) == "published"}
    comp = {k: variant.component.get(k, "0") for k in poses}
    cam = world.canonical_camera
    intr = {k: cam[k] for k in ("fx", "fy", "cx", "cy", "width", "height")}
    try:
        if viewpoints_from:
            ref = vp.load_viewpoints(viewpoints_from)
            vs = vp.transfer_viewpoints(ref, None, poses, component_of_to=comp, intrinsics_to=intr)
            mode = "transferred"
        else:
            vs = vp.viewpoint_set(poses, world.keyframe_ids, comp, intr)
            mode = "native"
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "why": f"viewpoints failed: {type(exc).__name__}: {exc}"}
    views = vs.get("views") or []
    subs = sorted({str(v.get("requested_keyframe_id")) for v in views if v.get("substituted")})
    block["viewpoints"] = {"rule": vs.get("rule") or getattr(vp, "RULE_VERSION", None),
                           "rule_version": getattr(vp, "RULE_VERSION", None),
                           "mode": mode, "reference": str(viewpoints_from) if viewpoints_from else None,
                           "transfer": vs.get("transfer"),
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


def render_variant(world, variant, vs, out_dir, *, regions=None, layers=RENDER_LAYERS,
                   include_minor: bool = False) -> dict:
    """Render the variant's PUBLISHED cameras and points of its MAIN component
    (default) from viewpoint set `vs` with C2's renderer (`layer_renders`).

    Minor components live in their own gauge; drawing them over the main one
    places them arbitrarily (C2's MINOR_GAUGE_NOTE), so they are left out
    unless `include_minor`. Colours come from `layer_renders.component_rgb` /
    `region_rgb`, which never wrap. With region labels the `_region` layers are
    added (evaluation-only colouring). Surfaces are not rendered here."""
    from tower.world_builder.coherence_eval import layer_renders as lr

    out_dir = Path(out_dir)
    pub = [k for k in world.keyframe_ids if variant.status.get(k) == "published" and k in variant.poses]
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for n_, k in enumerate(pub):
        c = variant.component.get(k, "0")
        counts[c] = counts.get(c, 0) + 1
        first.setdefault(c, n_)
    rank = {c: i for i, c in enumerate(sorted(counts, key=lambda c: (-counts[c], first[c])))}
    keep_ids = [k for k in pub if include_minor or rank[variant.component.get(k, "0")] == 0]
    poses = {k: np.asarray(variant.poses[k]) for k in keep_ids}
    comp_of = {k: rank[variant.component.get(k, "0")] for k in keep_ids}
    X = np.zeros((0, 3))
    pcomp = np.zeros(0, np.int32)
    first_kid: list = []
    if variant.has_observations and poses:
        order_of = {k: i for i, k in enumerate(world.keyframe_ids)}
        obs_world = np.array([order_of.get(k, -1) for k in variant.obs_keyframe_ids])[variant.obs_keyframe]
        drawn = np.array([world.keyframe_ids[i] in poses for i in range(world.n)] + [False])
        ok = (obs_world >= 0) & drawn[np.where(obs_world >= 0, obs_world, world.n)]
        firsto = np.full(len(variant.xyz), np.iinfo(np.int64).max)
        np.minimum.at(firsto, variant.obs_point[ok], obs_world[ok])
        keep = firsto < np.iinfo(np.int64).max
        X = variant.xyz[keep]
        first_kid = [world.keyframe_ids[i] for i in firsto[keep]]
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
    col_comp = np.array([lr.component_rgb(int(c)) for c in pcomp], np.uint8).reshape(-1, 3)
    col_time = lr.turbo(np.array([kid_rank.get(k, 0.0) for k in first_kid])) if len(first_kid) else col_comp
    col_reg = np.array([lr.region_rgb(wl.regions.get(k)) for k in first_kid], np.uint8).reshape(-1, 3)
    index = {"viewpoint_rule": vs.get("rule"), "layers": {}, "published_cameras_drawn": len(poses),
             "points_drawn": int(len(X)), "include_minor": include_minor,
             "component_colour_rank": {c: i for c, i in rank.items()}}
    for layer in layers:
        tiles = []
        for view in vs["views"]:
            if layer == "cameras":
                img = lr.render_cameras(view, wl, r, "component", include_minor=include_minor)
            elif layer == "cameras_region":
                img = lr.render_cameras(view, wl, r, "region", include_minor=include_minor)
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
    return {"available": True, "dir": "renders", "layers": index["layers"], "main_component_only": not include_minor,
            "published_cameras_drawn": len(poses), "points_drawn": int(len(X))}


# ---------------------------------------------------------------------------
# outputs

# (label, path, better): better is "higher" / "lower" for a variant quality,
# "world" for a property of the world's image-only caches (identical for every
# variant of the world; never a verdict).
HEADLINE = [
    ("unresolved / unmatched keyframe refs", "integrity.total", "lower"),
    ("published / accepted", "registration.published_fraction", "higher"),
    ("in largest component / accepted", "registration.in_largest_fraction", "higher"),
    ("components (with published kf)", "components.count", "lower"),
    ("main-component runs in capture order", "components.main_runs", "lower"),
    ("covisibility comps in main", "components.covisibility_components_in_main.count", "lower"),
    ("image islands >= 10 kf (world)", "islands.large_islands", "world"),
    ("largest image island / accepted (world)", "islands.largest_island_share", "world"),
    ("island pairs with placement UNOBSERVED (world)", "islands.large_island_pairs_unobserved", "world"),
    ("islands split across variant components", "islands.islands_split_by_variant", "lower"),
    ("max island tilt (abs median roll under main-island up, deg)", "islands.max_tilt_vs_main_island_deg", "lower"),
    ("image islands checked for placement plausibility", "islands.placement_plausibility.groups_checked", "info"),
    ("  failing (tilt / scale / height)", "islands.placement_plausibility.groups_failing", "lower"),
    ("  failing tilt > 10 deg", "islands.placement_plausibility.failing_tilt", "lower"),
    ("  failing scale beyond x1.25 (TRI)", "islands.placement_plausibility.failing_scale", "lower"),
    ("  failing eye height > 0.3 m beyond main band", "islands.placement_plausibility.failing_height", "lower"),
    ("  max island scale deviation vs main island (x)", "islands.placement_plausibility.max_scale_deviation_factor", "lower"),
    ("  max island eye-height offset beyond main band (m)",
     "islands.placement_plausibility.max_height_offset_beyond_band_m", "lower"),
    ("rigid groups (shared points >= 1, main comp)", "rigid_groups.count", "info"),
    ("  rigid groups >= 10 kf failing plausibility", "rigid_groups.placement_plausibility.groups_failing", "lower"),
    ("jumps (main; index+time or physical)", "continuity.main.jumps", "lower"),
    ("  physically implausible steps (main)", "continuity.main.physically_implausible", "lower"),
    ("  of which within a tracker segment", "continuity.main.jumps_within_segments", "lower"),
    ("tracking-loss steps (main)", "continuity.main.tracking_loss_steps", "info"),
    ("  without image evidence across (main)", "continuity.main.tracking_loss_steps_without_image_evidence", "lower"),
    ("index-only jumps (main)", "continuity.main.index_jumps", "lower"),
    ("spikes (main)", "continuity.main.spikes", "lower"),
    ("max step ratio (main)", "continuity.main.max_ratio", "lower"),
    ("jumps (all components)", "continuity.all_components.jumps", "lower"),
    ("TRI: kf with scale ratio / accepted", "scale.depth_tri.ratio_coverage", "higher"),
    ("TRI: coherent_all x1.5 (all accepted)", "scale.depth_tri.fixed_denominator.bands.x1.5.coherent_all", "higher"),
    ("TRI: coherent_all x1.25 (all accepted)", "scale.depth_tri.fixed_denominator.bands.x1.25.coherent_all", "higher"),
    ("TRI: longest bad run x1.25 (all accepted)", "scale.depth_tri.fixed_denominator.bands.x1.25.longest_bad_run", "lower"),
    ("TRI: max scale step factor (main)", "scale.depth_tri.main.max_step_factor", "lower"),
    ("TRI: segment scale p10-p90 factor (main)", "scale.depth_tri.by_tracker_segment.p10_p90_factor", "lower"),
    ("OBS: kf with scale ratio / accepted", "scale.depth.ratio_coverage", "higher"),
    ("OBS: coherent_all x1.5 (all accepted)", "scale.depth.fixed_denominator.bands.x1.5.coherent_all", "higher"),
    ("OBS: coherent_all x1.25 (all accepted)", "scale.depth.fixed_denominator.bands.x1.25.coherent_all", "higher"),
    ("OBS: longest bad run x1.25 (all accepted)", "scale.depth.fixed_denominator.bands.x1.25.longest_bad_run", "lower"),
    ("OBS: coherent x1.5 among ratio kf (main)", "scale.depth.main.levels.bands.x1.5.coherent_fraction", "higher"),
    ("OBS: coherent x1.25 among ratio kf (main)", "scale.depth.main.levels.bands.x1.25.coherent_fraction", "higher"),
    ("OBS: max scale step factor (main)", "scale.depth.main.max_step_factor", "lower"),
    ("OBS: max 15-kf scale step factor (main, D3)", "scale.depth.main.levels.max_step_15kf_factor", "lower"),
    ("OBS: scale trend factor over span (main)", "scale.depth.main.trend_factor_over_span", "lower"),
    ("OBS: segment scale p10-p90 factor (main)", "scale.depth.by_tracker_segment.p10_p90_factor", "lower"),
    ("adjacent pairs joined", "revisits.adjacent.joined_fraction", "higher"),
    ("adjacent rot err p90 (deg)", "revisits.adjacent.rot_err_deg.p90", "lower"),
    ("near rot err p90 (deg)", "revisits.near.rot_err_deg.p90", "lower"),
    ("revisit pairs (world)", "revisits.revisit.pairs", "world"),
    ("revisit pairs joined (within islands only)", "revisits.revisit.joined_fraction", "higher"),
    ("revisit rot err median (deg)", "revisits.revisit.rot_err_deg.median", "lower"),
    ("revisit rot err p90 (deg)", "revisits.revisit.rot_err_deg.p90", "lower"),
    ("revisit t-dir err median (deg; parallax >= 5)", "revisits.revisit.t_err_deg.median", "lower"),
    ("lingering pairs rot err p90 (deg)", "revisits.lingering.rot_err_deg.p90", "lower"),
    ("annotated revisits in different image islands (world)",
     "revisits.annotated.ranges_in_different_image_islands", "world"),
    ("reprojection median (px)", "reprojection.overall.median", "lower"),
    ("reprojection p99 (px)", "reprojection.overall.p99", "lower"),
    ("outlier cameras (main)", "sanity.main.outlier_cameras", "lower"),
    ("camera max/R90 (main)", "sanity.main.max_over_r90", "lower"),
]


def get_path(d, path):
    cur = d
    parts = path.split(".")
    k = 0
    while k < len(parts):
        if not isinstance(cur, dict):
            return None
        # keys may contain dots ("x1.5"): try the longest key first
        for span in range(len(parts) - k, 0, -1):
            key = ".".join(parts[k:k + span])
            if key in cur:
                cur = cur[key]
                k += span
                break
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
    m = dict(result["metrics"])
    integ = m.get("integrity") or {}
    m["integrity"] = dict(integ, total=int(integ.get("unresolved_keyframes") or 0) + int(integ.get("unmatched_images") or 0))
    rows = [(label, get_path(m, path), better) for label, path, better in HEADLINE]
    regs = get_path(m, "regions.regions") or {}
    by_reg = get_path(m, "scale.depth_tri.by_region") or {}
    for g, r in regs.items():
        rows.append((f"region {g}: in main & clean (incl. on-level)", r.get("in_main_clean"), "higher"))
        rows.append((f"region {g}: TRI scale factor vs main", (by_reg.get(g) or {}).get("factor_vs_main"), "info"))
    return rows


def _plaus_lines(pl: dict, what: str, col: str) -> list[str]:
    if not pl.get("available"):
        return ["", f"**Placement plausibility of {what}**: n/a ({pl.get('why', 'not computed')})."] if pl else []
    L = ["", f"**Placement plausibility of {what}** (vs {col} {pl['reference']}; up: {pl['up_source']}; "
             f"scale: {pl['scale_source']}; bounds: tilt <= {pl['bounds']['tilt_max_deg']:g} deg, scale within "
             f"x{pl['bounds']['scale_max_factor']:g}, eye height <= {pl['bounds']['height_offset_max_m']:g} m "
             "beyond the reference band; see PLAUSIBILITY_BASIS):", "",
         f"| {col} | kf in main | tilt (deg) | scale vs ref (IQR) | eye height median / beyond band / own spread (m) | verdict |",
         "|---|---|---|---|---|---|"]
    for g, r in pl["groups"].items():
        if not r.get("checkable"):
            continue
        iqr = r.get("scale_iqr")
        L.append(f"| {g}{' (ref)' if r['reference'] else ''} | {r['keyframes_in_main']} | {_fmt(r['tilt_deg'])} | "
                 f"{_fmt(r.get('scale_factor'))}{' (' + _fmt(iqr[0]) + '-' + _fmt(iqr[1]) + ')' if iqr else ''} | "
                 f"{_fmt(r.get('height_median_m'))} / {_fmt(r.get('height_offset_beyond_band_m'))} / "
                 f"{_fmt(r.get('height_spread_p10_p90_m'))} | "
                 f"{'reference' if r['reference'] else r['line']} |")
    small = sum(1 for r in pl["groups"].values() if not r.get("checkable"))
    if small:
        L.append(f"\n({small} smaller {what} not checkable: fewer than {PLAUSIBILITY['min_group_kf']} keyframes in the main component.)")
    L += ["", "Unobservable without image links, whatever these checks say: "
          + "; ".join(pl["unobservable_without_image_links"]) + "."]
    return L


def _island_lines(m) -> list[str]:
    isl = m.get("islands") or {}
    if not isl or isl.get("available") is False:
        return ["", "**Image islands**: n/a (no pair cache)."]
    L = ["", f"**Image islands** (world property; {isl['definition']}): {isl['islands']} islands, "
             f"{isl['singletons']} singletons, {isl['large_islands']} with >= {isl['large_island_min_kf']} keyframes; "
             f"largest holds {_fmt(isl['largest_island_share'])} of the accepted keyframes. "
             f"Cross-island tiers present: {', '.join(isl.get('cross_island_tiers_available') or []) or 'none'}.", ""]
    L += ["| island | keyframes | ranges | in variant main | variant components | tilt: median roll (deg) | "
          "roll scatter MAD (deg) |",
          "|---|---|---|---|---|---|---|"]
    for r in isl["large_island_list"]:
        t = (isl.get("tilt") or {}).get(str(r["island"])) or {}
        ref = " (reference)" if r["island"] == isl["large_island_list"][0]["island"] else ""
        L.append(f"| {r['island']} | {r['keyframes']} | {r['ranges'][:4]}{'...' if len(r['ranges']) > 4 else ''} | "
                 f"{r['in_main']} | {r['variant_components']} | {_fmt(t.get('tilt_deg'))}{ref} | "
                 f"{_fmt(t.get('roll_under_reference_mad_deg'))} |")
    L += _plaus_lines(isl.get("placement_plausibility") or {}, "image islands", "island")
    L += ["", "| island pair | verified pairs by tier | placement | variant on strict pairs |", "|---|---|---|---|"]
    for r in isl["island_pairs"]:
        v = r.get("variant") or {}
        L.append(f"| {r['islands'][0]}-{r['islands'][1]} | {r['pairs'] or 'no cross-island tier'} | "
                 f"**{'placement between islands: UNOBSERVED' if r['placement'] == 'UNOBSERVED' else 'observed'}** | "
                 f"{('joined ' + _fmt(v.get('joined_fraction')) + ', rot err med ' + _fmt(v.get('rot_err_median_deg')) + ' deg, t err med ' + _fmt(v.get('t_err_median_deg'))) if v else '-'} |")
    return L


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
             f"depth cache {w['caches']['depth'].get('digest') or 'MISSING'}.")
    integ = m.get("integrity") or {}
    if integ.get("unresolved_keyframes") or integ.get("unmatched_images"):
        L.append(f"**WARNING: {integ.get('unresolved_keyframes')} unresolved keyframe rows, "
                 f"{integ.get('unmatched_images')} unmatched images.**")
    L += ["", "| metric | value |", "|---|---|"]
    for label, val, better in headline(result):
        tag = {"world": "world property", "info": "info"}.get(better, f"{better} is better")
        L.append(f"| {label} ({tag}) | {_fmt(val)} |")
    L += _island_lines(m)
    rg = m.get("rigid_groups") or {}
    if rg.get("available") is False:
        L += ["", f"**Rigid groups**: n/a ({rg.get('why')})."]
    elif rg:
        L += ["", f"**Rigid groups** of the variant (connected by >= 1 shared 3-D point, main component): {rg['count']} "
                  f"groups, sizes {rg['sizes'][:10]}."]
        L += _plaus_lines(rg.get("placement_plausibility") or {}, "rigid groups", "group")
    main = get_path(m, "continuity.main") or {}
    if main.get("jump_list"):
        L += ["", "**Step candidates in the main component** (index rule, physical rule, and every step across a "
              "tracking loss; COUNTED = a jump; step_m via the component's MoGe scale):", ""]
        for j in main["jump_list"][:40]:
            L.append(f"- {j['from']}->{j['to']} gap {j['gap']}: ratio {_fmt(j['ratio'])}, speed ratio "
                     f"{_fmt(j.get('speed_ratio'))}, dt {_fmt(j.get('dt_s'))} s, step {_fmt(j.get('step_m'))} m, "
                     f"rot {_fmt(j['rot_deg'])} deg [{j['kind'] or '-'}]"
                     f"{' (tracking loss' + ('; NO image evidence across' if j.get('image_evidence_across') is False else '') + ')' if j.get('segment_boundary_inside') else ''}"
                     f"{' COUNTED' if j.get('counted') else ''}")
    for src, key in (("TRI", "depth_tri"), ("OBS", "depth")):
        steps = get_path(m, f"scale.{key}.main.step_list") or []
        if steps:
            L += ["", f"**Scale steps ({src}, main component)**: " +
                  ", ".join(f"at {s['at_index']} x{_fmt(s['factor'])}" for s in steps)]
    regs = get_path(m, "regions.regions")
    if regs:
        by_t = get_path(m, "scale.depth_tri.by_region") or {}
        by_o = get_path(m, "scale.depth.by_region") or {}
        L += ["", f"Regions (C0 labels, evaluation only). Clean = {get_path(m, 'regions.clean_definition')}.", "",
              "| region | kf | posed | published | in main | on-level | in main & clean | jump endpoints | "
              "scale vs main TRI | scale vs main OBS |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for g, r in regs.items():
            L.append(f"| {g} | {r['keyframes']} | {_fmt(r['posed'])} | {_fmt(r['published'])} | {_fmt(r['in_main'])} | "
                     f"{_fmt(r.get('on_level'))} | {_fmt(r['in_main_clean'])} | {r['jump_endpoints']} | "
                     f"{_fmt((by_t.get(g) or {}).get('factor_vs_main'))} | {_fmt((by_o.get(g) or {}).get('factor_vs_main'))} |")
    dbr = get_path(m, "revisits.by_region")
    if dbr:
        L += ["", "| long-gap pairs by region (bucket) | pairs | joined | rot err median | t err median (n) |",
              "|---|---|---|---|---|"]
        for g, r in dbr.items():
            L.append(f"| {g} | {r['pairs']} | {_fmt(r['joined_fraction'])} | {_fmt(r['rot_err_median_deg'])} | "
                     f"{_fmt(r['t_err_median_deg'])} ({r['t_err_count']}) |")
    ann = get_path(m, "revisits.annotated")
    if ann and ann.get("available"):
        L += ["", f"Annotated revisits (C0): {ann['note']}.", "",
              "| annotated revisit (C0) | region | same image island | A in main | B in main | closest/R90 | "
              "verified pairs (joined) | rot err med |",
              "|---|---|---|---|---|---|---|---|"]
        for r in ann["list"]:
            L.append(f"| {r['a'][0]}-{r['a'][1]} vs {r['b'][0]}-{r['b'][1]} | {r.get('region')} | "
                     f"{r.get('same_image_island')} | {_fmt(r['a_in_main'])} | {_fmt(r['b_in_main'])} | "
                     f"{_fmt(r['closest_over_r90'])}{'' if r.get('same_image_island', True) else ' (unobserved)'} | "
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
    L += ["", "Definitions, thresholds and limitations: `tower/world_builder/coherence_eval/metrics.py` and "
              "`eval_placement.py` docstrings; every threshold is in `harness.params` of metrics.json."]
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


def coverage_changes(a: dict, b: dict) -> list[str]:
    """Coverage differences large enough to invalidate row-by-row verdicts (H1)."""
    out = []
    for path, tol in (("registration.published_fraction", PARAMS["compare_coverage_tol"]),
                      ("registration.in_largest_fraction", PARAMS["compare_coverage_tol"]),
                      ("scale.depth_tri.ratio_coverage", PARAMS["compare_ratio_coverage_tol"]),
                      ("scale.depth.ratio_coverage", PARAMS["compare_ratio_coverage_tol"])):
        va, vb = get_path(a["metrics"], path), get_path(b["metrics"], path)
        if va is None and vb is None:
            continue
        if path == "scale.depth.ratio_coverage" and (va is None or vb is None):
            continue  # a variant WITHOUT points: its OBS rows are n/a and count as losses instead
        if va is None or vb is None or abs(float(va) - float(vb)) > tol:
            out.append(f"{path}: {_fmt(va)} -> {_fmt(vb)} (tolerance {tol:g})")
    return out


def compare_markdown(a: dict, b: dict) -> str:
    """Row-by-row diff with a verdict per row and an overall line.

    * A row whose value is n/a on ONE side is a loss for that side (a variant
      cannot win by making a metric disappear, review V2 H3).
    * Rows marked "world" are properties of the world's caches: no verdict.
    * When coverage changed beyond tolerance (published / in-main fractions,
      scale-ratio coverage) every verdict is WITHHELD: a variant that
      publishes less is not comparable row by row (review V2 H1)."""
    na, nb = a["variant"]["name"], b["variant"]["name"]
    L = [f"# Compare: A = {na} ({a['world']['world_id'][:8]}) vs B = {nb} ({b['world']['world_id'][:8]})", ""]
    if a["world"]["world_id"] != b["world"]["world_id"]:
        L.append("**WARNING: different worlds; the numbers are not comparable pair by pair.**\n")
    for key, field in (("pairs", "digest"), ("depth", "digest")):
        da = get_path(a, f"world.caches.{key}.{field}")
        db = get_path(b, f"world.caches.{key}.{field}")
        if da != db:
            L.append(f"**WARNING: the {key} cache differs between A and B ({da} vs {db}).**\n")
    if a["harness"].get("code_digest") != b["harness"].get("code_digest"):
        L.append("_Note: harness code digests differ._\n")
    cov = coverage_changes(a, b)
    if cov:
        L.append("**COVERAGE CHANGED -- row verdicts withheld:** " + "; ".join(cov) + "\n")
    L += ["| metric | A | B | B - A | better |", "|---|---|---|---|---|"]
    ra = {label: (val, better) for label, val, better in headline(a)}
    rb = {label: (val, better) for label, val, better in headline(b)}
    wins = {"A": 0, "B": 0}
    na_losses = {"A": 0, "B": 0}
    ties = 0
    for label in list(ra) + [x for x in rb if x not in ra]:
        va, better = ra.get(label, (None, None))
        vb, better_b = rb.get(label, (None, None))
        better = better or better_b
        delta = None
        verdict = ""
        num_a = isinstance(va, (int, float)) and not isinstance(va, bool)
        num_b = isinstance(vb, (int, float)) and not isinstance(vb, bool)
        if num_a and num_b:
            delta = vb - va
            if abs(delta) <= 1e-9 + 1e-3 * max(abs(va), abs(vb)):
                delta = 0 if isinstance(delta, int) else 0.0
        if better in ("higher", "lower"):
            if num_a and num_b:
                if delta == 0:
                    ties += 1
                else:
                    verdict = "B" if (delta > 0) == (better == "higher") else "A"
            elif num_a != num_b:
                verdict = "A" if num_a else "B"
                na_losses["B" if num_a else "A"] += 1
                verdict += " (n/a loses)"
            if verdict and not cov:
                wins[verdict[0]] += 1
            elif verdict and cov:
                verdict = f"withheld ({verdict})"
        elif better == "world":
            verdict = "world" if va == vb else "WORLD DIFFERS"
        L.append(f"| {label} | {_fmt(va)} | {_fmt(vb)} | {_fmt(delta)} | {verdict} |")
    if cov:
        L.append("\n**Overall: no verdict -- coverage changed** (see above); compare the fixed-denominator rows "
                 "(coherent_all, longest bad run) and registration first.")
    else:
        L.append(f"\n**Overall: B better on {wins['B']} rows, A better on {wins['A']}, {ties} equal "
                 f"(n/a counted as a loss: A {na_losses['A']}, B {na_losses['B']}).** Rows are not equally "
                 "important; read the scale and placement rows first.")
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
