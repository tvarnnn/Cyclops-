"""The coherence gate's metric scale: per camera, log(z_sfm / z_metric), from the solve's own evidence.

WHY. The evidence gate (`coherence_gate.py`) refuses to attach a piece whose metric level differs from the
room's by more than x1.25, and splits a piece at an internal x1.25 step: one similarity cannot hold a x4
scale step (the target's bathroom sat at x4.27). It needs, per camera, the ratio between the solve's depth
of what the camera sees and an independent METRIC depth of the same pixels. Metric scale is a hard
dependency of the gate: with no ratio at all the gate attaches nothing (`scale-unavailable`).

WHAT IT MEASURES. A port of the run's harness "TRI" estimator
(`coherence_eval/eval_placement.triangulated_depth_ratios`, harness `wb-coherence-metrics/2`), which is the
measurement the gate rule was validated with (run wb-coherence-run-2026-09-23, P2-GT, P2-LM). Per verified
image pair of the SAME solver component whose two cameras are both published (>= `min_obs` observations):

  * the pair's inlier correspondences, normalised by the solve's pinhole camera;
  * the pair's relative pose FROM THE SOLVE (R_ji = R_j R_i^T, t_ji = R_j (C_i - C_j));
  * gated like the harness: Sampson error <= 2 px, ray angle 2..90 deg, both depths positive;
  * each surviving correspondence gives both cameras a triangulated depth (least-squares midpoint).

Per camera, the median over all its triangulated samples of log(z_triangulated / z_metric), where z_metric
is the metric depth sampled at the same pixel (nearest pixel, COLMAP convention), kept only inside
`valid_m` and only with >= `min_samples` samples. Cameras without enough samples have no ratio and are
simply absent from `metric_log` -- the gate then has no scale evidence for them.

WHAT DIFFERS FROM THE HARNESS, deliberately, because the product has neither harness cache:
  * PAIRS: the solve database's own verified inlier matches (`two_view_geometries.data` over the
    `keypoints`, the same rows `coherence_gate.read_link_rotations` reads), not the harness's cached
    image-only pairs. The pairs are the solver's evidence, so this is what the product can see.
  * DEPTH: whatever `depth_sample(name, uv)` returns. In the product that is the depth stage's own MoGe-2
    ViT-L (`moge2-vitl`, the harness's checkpoint) raw prediction for the keyframe (`<ki>_pred.npy`, metric
    z), with the redaction fill (`<ki>_fill.npy`) masked out: a depth predicted inside an inpainted face box
    is invention, and the depth stage refuses to anchor on it for the same reason (`_fit_record`).
The validation of both substitutions against the harness is in RUN experiments/P3-PG.

THRESHOLDS. Every one is the harness's own (`eval_placement.PLACEMENT_PARAMS` tri_*, `metrics.PARAMS`
depth_min_samples / depth_valid_m, and its 5-inlier pair floor); none was tuned here.

numpy only; `read_inlier_pairs` reads SQLite read-only and immutable. No GPU, no pycolmap.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SCALE_ID = "coherence-scale:tri(solve-db-pairs,moge2-vitl)/1"

NOT_VERIFIED_CONFIGS = (0, 1)  # COLMAP TwoViewGeometry UNDEFINED, DEGENERATE
_COLMAP_PAIR_BASE = 2147483647


@dataclass(frozen=True)
class ScaleParams:
    # global_solve.MIN_IMAGE_OBSERVATIONS: only published cameras are measured (the harness scores and the
    # gate tests published cameras only).
    min_obs: int = 30
    # eval_placement.triangulated_depth_ratios skips a pair with fewer than 5 inliers.
    min_pair_inliers: int = 5
    # eval_placement.PLACEMENT_PARAMS["tri_sampson_gate_px"]
    sampson_gate_px: float = 2.0
    # eval_placement.PLACEMENT_PARAMS["tri_min_angle_deg"] / ["tri_max_angle_deg"]
    min_angle_deg: float = 2.0
    max_angle_deg: float = 90.0
    # metrics.PARAMS["depth_min_samples"]
    min_samples: int = 10
    # metrics.PARAMS["depth_valid_m"], metres
    valid_min_m: float = 0.05
    valid_max_m: float = 50.0

    def to_json(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        doc = {"scale": SCALE_ID, **self.to_json()}
        return hashlib.sha1(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class ScaleCameras:
    """The solve's cameras, as the estimator needs them. `R_cw`/`t_cw` world->camera; `component` the
    solver model index; `n_obs` the camera's 3-D observations. Indexed like `names`."""

    names: list[str]
    R_cw: np.ndarray        # (n, 3, 3)
    t_cw: np.ndarray        # (n, 3)
    component: np.ndarray   # (n,) int
    n_obs: np.ndarray       # (n,) int

    def index(self) -> dict[str, int]:
        return {nm: i for i, nm in enumerate(self.names)}


# ---------------------------------------------------------------------------
# inputs


def read_inlier_pairs(database_path, *, min_inliers: int = 5,
                      exclude_configs=NOT_VERIFIED_CONFIGS) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """[(name_a, name_b, uv_a (k,2), uv_b (k,2))] for every verified two-view geometry of a COLMAP database:
    the pair's stored inlier matches over the two images' keypoints, in the database's pixel coordinates.
    Opened read-only and immutable, so a published database is never touched."""
    import sqlite3

    uri = Path(database_path).resolve().as_uri() + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        names = dict(con.execute("select image_id, name from images").fetchall())
        kp: dict = {}

        def keypoints(i):
            if i not in kp:
                row = con.execute("select rows, cols, data from keypoints where image_id=?", (i,)).fetchone()
                if row is None or row[2] is None:
                    kp[i] = np.zeros((0, 2), np.float64)
                else:
                    r, c, d = row
                    kp[i] = np.frombuffer(d, np.float32).reshape(r, c)[:, :2].astype(np.float64)
            return kp[i]

        out = []
        for pid, rows, config, data in con.execute(
                "select pair_id, rows, config, data from two_view_geometries"):
            if config in exclude_configs or rows < min_inliers or data is None:
                continue
            b = int(pid) % _COLMAP_PAIR_BASE
            a = (int(pid) - b) // _COLMAP_PAIR_BASE
            if a not in names or b not in names:
                continue
            m = np.frombuffer(data, np.uint32).reshape(rows, 2).astype(np.int64)
            ka, kb = keypoints(a), keypoints(b)
            ok = (m[:, 0] < len(ka)) & (m[:, 1] < len(kb))
            m = m[ok]
            if len(m) < min_inliers:
                continue
            out.append((names[a], names[b], ka[m[:, 0]], kb[m[:, 1]]))
        return out
    finally:
        con.close()


def cameras_from_solution(solution, image_name_of: Callable[[str], str]) -> ScaleCameras:
    """`ScaleCameras` for every posed keyframe of a `global_solve.Solution`. `image_name_of(keyframe_id)` is
    the name the solve database knows the keyframe by (`global_solve.keyframe_image_name`)."""
    names, R, t, comp, nobs = [], [], [], [], []
    for kid in solution.keyframe_ids:
        pose = (solution.poses or {}).get(kid)
        if not pose or pose.get("rotation") is None or pose.get("translation") is None:
            continue
        names.append(image_name_of(kid))
        R.append(np.asarray(pose["rotation"], np.float64).reshape(3, 3))
        t.append(np.asarray(pose["translation"], np.float64).reshape(3))
        comp.append(int(pose.get("component", 0)))
        nobs.append(int(pose.get("observations", 0)))
    return ScaleCameras(names=names, R_cw=np.asarray(R, np.float64).reshape(-1, 3, 3),
                        t_cw=np.asarray(t, np.float64).reshape(-1, 3),
                        component=np.asarray(comp, np.int64), n_obs=np.asarray(nobs, np.int64))


class DepthStageSampler:
    """`depth_sample(name, uv)` over the product depth stage's raw predictions.

    `work` is the depth stage's work directory (`dense/<session>/work`); `ki_of_name` maps a solve image
    name to the keyframe index the stage names its files by. Reads `depth/<ki>_pred.npy` (MoGe-2's metric z
    at the solve camera's resolution, NaN where the network masked the pixel) and, when present,
    `depth/<ki>_fill.npy` (the redaction fill, which becomes NaN). A missing prediction is all-NaN."""

    def __init__(self, work, ki_of_name: dict[str, int], *, memo: int = 8) -> None:
        self.work = Path(work)
        self.ki_of_name = dict(ki_of_name)
        self._memo: dict[str, np.ndarray | None] = {}
        self._memo_size = memo
        self.missing: set[str] = set()

    def _load(self, name: str) -> np.ndarray | None:
        if name in self._memo:
            return self._memo[name]
        ki = self.ki_of_name.get(name)
        d = None
        if ki is not None:
            p = self.work / "depth" / f"{int(ki):05d}_pred.npy"
            if p.is_file():
                try:
                    d = np.load(p).astype(np.float32)
                except (OSError, ValueError):
                    d = None
                if d is not None:
                    fp = self.work / "depth" / f"{int(ki):05d}_fill.npy"
                    if fp.is_file():
                        try:
                            fill = np.load(fp)
                            if fill.shape == d.shape:
                                d = np.where(fill.astype(bool), np.nan, d)
                            else:
                                d = None  # a mask for another image: trust neither
                        except (OSError, ValueError):
                            d = None
        if d is None:
            self.missing.add(name)
        if len(self._memo) >= self._memo_size:
            self._memo.clear()
        self._memo[name] = d
        return d

    def __call__(self, name: str, uv) -> np.ndarray:
        return sample_nearest(self._load(name), uv)


def sample_nearest(depth: np.ndarray | None, uv) -> np.ndarray:
    """Depth at pixels (nearest pixel, COLMAP convention: pixel i spans [i, i+1)); NaN outside or invalid.
    The harness's `DepthCache.sample`, verbatim in what it computes."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    out = np.full(len(uv), np.nan)
    if depth is None or not len(uv):
        return out
    h, w = depth.shape[:2]
    finite = np.isfinite(uv).all(1)
    uvs = np.where(np.isfinite(uv), uv, -1.0)
    iu = np.floor(uvs[:, 0]).astype(np.int64)
    iv = np.floor(uvs[:, 1]).astype(np.int64)
    ok = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h) & finite
    out[ok] = depth[iv[ok], iu[ok]]
    return out


# ---------------------------------------------------------------------------
# the harness's triangulation, verbatim in what it computes


def triangulate_midpoint(x1, x2, R, t):
    """Depths (z1, z2) of normalised correspondences under x2 ~ R x1 + t (least-squares midpoint along the two
    rays) and the angle between the rays (degrees). `eval_placement.triangulate_midpoint`."""
    f1 = np.c_[x1, np.ones(len(x1))]
    f2 = np.c_[x2, np.ones(len(x2))]
    a = f1 @ R.T
    b = f2
    aa = (a * a).sum(1)
    bb = (b * b).sum(1)
    ab = (a * b).sum(1)
    at = a @ t
    bt = b @ t
    det = aa * bb - ab * ab
    with np.errstate(divide="ignore", invalid="ignore"):
        l1 = (-at * bb + ab * bt) / det
        l2 = (aa * bt - ab * at) / det
        ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), ab))
    return l1, l2, ang


def sampson_px(R, t, x1, x2, f):
    """Per-correspondence Sampson error in pixels (focal `f`). `eval_placement.sampson_px_each`."""
    t = t / (np.linalg.norm(t) + 1e-15)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    a = np.c_[x1, np.ones(len(x1))]
    b = np.c_[x2, np.ones(len(x2))]
    Ea = (E @ a.T).T
    Etb = (E.T @ b.T).T
    num = (b * Ea).sum(1) ** 2
    den = Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-30)) * f


def metric_scale(cameras: ScaleCameras, pairs, camera: dict,
                 depth_sample: Callable[[str, np.ndarray], np.ndarray],
                 params: ScaleParams | None = None) -> dict:
    """Per camera log(z_sfm / z_metric) (module docstring).

    cameras: the solve's cameras; pairs: `read_inlier_pairs(...)` (pixel coordinates in `camera`);
    camera: the pinhole camera {fx, fy, cx, cy}; depth_sample(name, uv_pixels) -> metric depth (NaN unknown).

    Returns {"metric_log": {name: float}, "samples": {name: int}, "spread": {name: float}, "pairs_used",
    "inliers_total", "inliers_gated", "cameras_published", "cameras_measured", "params", "params_digest",
    "scale": SCALE_ID}. `metric_log` holds only finite ratios."""
    params = params or ScaleParams()
    fx, fy = float(camera["fx"]), float(camera["fy"])
    cx, cy = float(camera["cx"]), float(camera["cy"])
    idx = cameras.index()
    n = len(cameras.names)
    published = cameras.n_obs >= params.min_obs
    R = cameras.R_cw
    C = -np.einsum("nji,nj->ni", R, cameras.t_cw) if n else np.zeros((0, 3))
    per_xy: list[list[np.ndarray]] = [[] for _ in range(n)]
    per_z: list[list[np.ndarray]] = [[] for _ in range(n)]
    used = total = gated = 0
    for name_a, name_b, uv_a, uv_b in pairs or ():
        i, j = idx.get(name_a), idx.get(name_b)
        if i is None or j is None or i == j:
            continue
        if not (published[i] and published[j] and cameras.component[i] == cameras.component[j]):
            continue
        if len(uv_a) < params.min_pair_inliers:
            continue
        R_ji = R[j] @ R[i].T
        t_ji = R[j] @ (C[i] - C[j])
        if not np.isfinite(t_ji).all() or np.linalg.norm(t_ji) <= 0:
            continue
        x1 = np.c_[(uv_a[:, 0] - cx) / fx, (uv_a[:, 1] - cy) / fy]
        x2 = np.c_[(uv_b[:, 0] - cx) / fx, (uv_b[:, 1] - cy) / fy]
        total += len(x1)
        s = sampson_px(R_ji, t_ji, x1, x2, fx)
        z1, z2, ang = triangulate_midpoint(x1, x2, R_ji, t_ji)
        ok = (s <= params.sampson_gate_px) & (ang >= params.min_angle_deg) & (ang <= params.max_angle_deg)
        ok &= np.isfinite(z1) & np.isfinite(z2) & (z1 > 0) & (z2 > 0)
        gated += int((~ok).sum())
        if not ok.any():
            continue
        used += 1
        per_xy[i].append(x1[ok])
        per_z[i].append(z1[ok])
        per_xy[j].append(x2[ok])
        per_z[j].append(z2[ok])
    metric_log: dict[str, float] = {}
    samples: dict[str, int] = {}
    spread: dict[str, float] = {}
    for i in range(n):
        if not per_z[i]:
            continue
        xy = np.concatenate(per_xy[i])
        z = np.concatenate(per_z[i])
        uv = np.c_[xy[:, 0] * fx + cx, xy[:, 1] * fy + cy]
        zm = np.asarray(depth_sample(cameras.names[i], uv), dtype=np.float64)
        ok = np.isfinite(zm) & (zm > params.valid_min_m) & (zm < params.valid_max_m)
        if int(ok.sum()) < params.min_samples:
            continue
        lr = np.log(z[ok] / zm[ok])
        med = float(np.median(lr))
        if not math.isfinite(med):
            continue
        metric_log[cameras.names[i]] = med
        samples[cameras.names[i]] = int(ok.sum())
        spread[cameras.names[i]] = float(1.4826 * np.median(np.abs(lr - med)))
    return {"metric_log": metric_log, "samples": samples, "spread": spread,
            "pairs_used": used, "inliers_total": total, "inliers_gated": gated,
            "cameras_published": int(published.sum()), "cameras_measured": len(metric_log),
            "params": params.to_json(), "params_digest": params.digest(), "scale": SCALE_ID}
