"""Chronological trajectory, sparse-structure and segment forensics.

One question, asked of a persisted World Builder world: *walking the capture
in the order it happened, where does the reconstruction stop being one
coherent space, and which upstream layer made it so?*

Everything here READS. A world directory (ideally a frozen copy) and the raw
capture it came from go in; pandas tables and plots come out. Nothing writes
into the world, the COLMAP database is opened ``mode=ro&immutable=1``, and no
region label or other annotation is ever an input to anything that estimates
geometry -- labels are joined onto the output table at the very end, for
reading only.

The layers, and where each column comes from
--------------------------------------------

* **frontend** -- ``sessions/<sid>/keyframes.jsonl`` for accepted keyframes,
  plus an exact *replay* of ``FrameTracker`` + ``KeyframeSelector`` over the
  raw capture frames (the journal records only accepted keyframes and
  ``tracking_lost`` events; per-frame rejection reasons exist nowhere on
  disk). The replay is verified against the persisted keyframe set.
* **live chain** -- ``edges.jsonl`` and ``events.jsonl`` as persisted, plus a
  re-run of the classical backend's ``estimate_window`` over each segment's
  (redacted) keyframe images, which is what the live solve fed and which
  ``tests/test_world_builder_incremental.py`` pins bit-identical to it. The
  re-run is verified edge-by-edge against ``edges.jsonl``.
* **global solve** -- ``solve/<sid>/solution.json`` + ``solution.npz``
  (GLOMAP poses per component, observations) and ``database.db``
  (``two_view_geometries``: the verified pair graph GLOMAP solved).
* **published** -- ``derived/<sid>/poses.json`` + ``placements.json`` +
  ``manifest.json``: what a consumer actually receives, after
  ``global_solve.merge``.
* **joins** -- ``dense/<sid>/align.json`` (per-keyframe depth-alignment
  scale; owned by the dense stage) and a region-label CSV, when given.

Frames and conventions
----------------------

Persisted poses are ``T_world_camera`` (``schema.POSE_CONVENTION``): the
translation IS the camera centre. The solution stores ``cam_from_world``
(``R_cw``, ``t_cw``); centres are ``-R_cw.T @ t_cw``. Relative motion between
consecutive keyframes a -> b is ``R_ba = R_b R_a^T`` (angle in degrees) and
the unit direction of ``t_ba = t_b - R_ba t_a`` expressed in camera *b*;
this is also what ``cv2.recoverPose(E, pts_a, pts_b)`` returns, so two-view
and solved relative poses are directly comparable.
"""

from __future__ import annotations

import io
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

COLMAP_PAIR_BASE = 2147483647
# COLMAP TwoViewGeometry::ConfigurationType
TWO_VIEW_CONFIG = {
    0: "undefined", 1: "degenerate", 2: "calibrated", 3: "uncalibrated",
    4: "planar", 5: "panoramic", 6: "planar_or_panoramic", 7: "watermark",
    8: "multiple",
}
# A verified pair in the sense GLOMAP's view graph uses by default.
MIN_VERIFIED_INLIERS = 15
# global_solve.MIN_IMAGE_OBSERVATIONS, restated so this module can be read
# without importing the builder; `load_world` checks the two agree when the
# builder is importable.
MIN_IMAGE_OBSERVATIONS = 30
# Sequential matching window (global_solve.SEQUENTIAL_OVERLAP); pairs further
# apart than this in keyframe order can only have come from loop detection.
SEQUENTIAL_OVERLAP = 20


# ---------------------------------------------------------------------------
# Small geometry helpers (numpy only).


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat_wxyz(m) -> list[float]:
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        q = [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def rotation_angle_deg(R) -> float:
    c = (np.trace(np.asarray(R)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def angle_between_deg(u, v) -> float:
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0))))


def project_to_so3(M) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
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
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_z(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if ok.sum() < 3:
        return out
    med = np.median(v[ok])
    mad = np.median(np.abs(v[ok] - med)) * 1.4826
    if mad <= 0:
        mad = np.std(v[ok]) or 1.0
    out[ok] = (v[ok] - med) / mad
    return out


# ---------------------------------------------------------------------------
# Loading a world, read-only.


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class WorldPaths:
    world_dir: Path
    world_id: str
    session_id: str

    @property
    def session_dir(self) -> Path:
        return self.world_dir / "sessions" / self.session_id

    @property
    def solve_dir(self) -> Path:
        return self.world_dir / "solve" / self.session_id

    @property
    def derived_dir(self) -> Path:
        return self.world_dir / "derived" / self.session_id

    @property
    def dense_dir(self) -> Path:
        return self.world_dir / "dense" / self.session_id


def resolve_world(world_dir, session_id: str | None = None) -> WorldPaths:
    world_dir = Path(world_dir)
    world = _read_json(world_dir / "world.json") or {}
    sessions = list(world.get("session_ids") or [])
    if session_id is None:
        if not sessions:
            sessions = sorted(p.name for p in (world_dir / "sessions").iterdir() if p.is_dir())
        if not sessions:
            raise FileNotFoundError(f"no session in {world_dir}")
        session_id = sessions[-1]
    return WorldPaths(world_dir=world_dir, world_id=world.get("world_id", world_dir.name),
                      session_id=session_id)


def load_solution(paths: WorldPaths) -> dict | None:
    """solution.json + solution.npz, as plain dicts/arrays (no builder import)."""
    meta = _read_json(paths.solve_dir / "solution.json")
    npz = paths.solve_dir / "solution.npz"
    if meta is None or not npz.exists():
        return None
    with np.load(io.BytesIO(npz.read_bytes())) as arrays:
        arr = {k: arrays[k] for k in arrays.files}
    meta["arrays"] = arr
    return meta


def capture_chain(captures_root: Path, first_capture_id: str, session: dict) -> list[Path]:
    """The capture directories a session followed, in time order.

    A session names only the FIRST capture it followed; a reconnect starts a
    new capture whose ``capture.json`` says ``continues_capture``. Follow that
    chain forward while captures begin before the session ended.
    """
    captures_root = Path(captures_root)
    first = captures_root / first_capture_id
    if not first.exists():
        return []
    chain = [first]
    metas = {}
    for d in captures_root.iterdir():
        meta = _read_json(d / "capture.json") if d.is_dir() else None
        if meta:
            metas[d.name] = meta
    current = first_capture_id
    ended = session.get("ended_at") or float("inf")
    while True:
        nxt = [cid for cid, m in metas.items() if m.get("continues_capture") == current
               and (m.get("started_at") or 0) <= ended]
        if not nxt:
            break
        current = sorted(nxt, key=lambda c: metas[c].get("started_at") or 0)[0]
        chain.append(captures_root / current)
    return chain


def load_capture_frames(capture_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for cdir in capture_dirs:
        for r in _read_jsonl(Path(cdir) / "frames.jsonl"):
            rows.append({"capture_id": Path(cdir).name, "source_seq": r.get("source_seq"),
                         "wire_seq": r.get("wire_seq"), "tx_seq": r.get("tx_seq"),
                         "received_at": r.get("received_at"),
                         "path": str(Path(cdir) / r["relpath"])})
    df = pd.DataFrame(rows)
    if len(df):
        df["frame_index"] = np.arange(len(df))
    return df


# ---------------------------------------------------------------------------
# Frontend replay.


def replay_frontend(frames: pd.DataFrame) -> pd.DataFrame:
    """Re-run the per-frame keyframe decision exactly as `engine.observe` does.

    Order of operations mirrors engine.py (decode, size gate, analyse_frame,
    note_frame, measure, evaluate; on loss reset+note_lost; on accept
    set_reference(raw gray)+note_accepted). The live solve's chain breaks do
    not touch the tracker or selector (engine.py: "The tracker keeps its
    reference"), so they need not be simulated for the decisions.
    """
    import cv2  # noqa: F401  (imported by frontend; keep the import order explicit)

    from tower.world_builder.frontend import FrameTracker, analyse_frame, decode_gray
    from tower.world_builder.keyframes import KeyframePolicy, KeyframeSelector

    selector = KeyframeSelector(KeyframePolicy())
    tracker = FrameTracker()
    shape = None
    out = []
    for row in frames.itertuples(index=False):
        rec = {"frame_index": row.frame_index, "capture_id": row.capture_id,
               "source_seq": row.source_seq, "received_at": row.received_at}
        try:
            gray = decode_gray(Path(row.path).read_bytes())
        except (ValueError, OSError):
            rec.update(outcome="reject", reason="malformed_frame")
            out.append(rec)
            continue
        if shape is None:
            shape = gray.shape[:2]
        elif gray.shape[:2] != shape:
            rec.update(outcome="reject", reason="frame_size_changed")
            out.append(rec)
            continue
        quality = analyse_frame(gray)
        selector.note_frame(quality)
        motion = tracker.measure(gray)
        decision = selector.evaluate(quality, motion)
        rec.update(outcome=decision.outcome, reason=decision.reason, sharpness=quality.sharpness,
                   survival_ratio=getattr(motion, "survival_ratio", np.nan),
                   overlap_ratio=getattr(motion, "overlap_ratio", np.nan),
                   displacement_px=getattr(motion, "median_displacement_px", np.nan),
                   tracked=getattr(motion, "tracked_count", np.nan))
        if decision.lost:
            tracker.reset()
            selector.note_lost()
        elif decision.accepted:
            tracker.set_reference(gray)
            selector.note_accepted()
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Live chain re-run.


def rerun_live_chain(paths: WorldPaths, keyframes: list[dict], session: dict) -> tuple[pd.DataFrame, dict]:
    """Per-segment `estimate_window` over the session's keyframe images.

    Returns per-keyframe local poses (T_world_camera in the SEGMENT frame,
    exactly as engine._pose_row persists them) and per-segment point counts.
    """
    from tower.world_builder.backend import KeyframeInput
    from tower.world_builder.backends.classical import ClassicalTwoViewBackend
    from tower.world_builder.frontend import decode_gray
    from tower.world_builder.records import camera_intrinsics_from_json_dict

    intrinsics = camera_intrinsics_from_json_dict(session["intrinsics"])
    backend = ClassicalTwoViewBackend()
    backend.prepare(intrinsics)
    by_seg: dict[int, list[dict]] = defaultdict(list)
    for k in keyframes:
        by_seg[int(k["segment_index"])].append(k)
    rows = []
    seg_info = {}
    for seg, members in sorted(by_seg.items()):
        window = [KeyframeInput(keyframe_id=k["keyframe_id"],
                                image_gray=decode_gray((paths.session_dir / k["image_relpath"]).read_bytes()))
                  for k in members]
        est = backend.estimate_window(window)
        seg_info[seg] = {"live_points": 0 if est.points is None else int(len(est.points)),
                         "diagnostics": est.diagnostics}
        for k, pose in zip(members, est.poses):
            r = {"keyframe_id": k["keyframe_id"], "live_status": pose.status,
                 "live_degeneracy": pose.degeneracy, "live_matches": pose.matches,
                 "live_inliers": pose.inliers, "live_R_wc": None, "live_C": None}
            if pose.status == "anchor":
                r["live_R_wc"] = np.eye(3)
                r["live_C"] = np.zeros(3)
            elif pose.rotation is not None:
                R = np.asarray(pose.rotation, float)
                r["live_R_wc"] = R.T
                if pose.translation is not None:
                    r["live_C"] = -R.T @ np.asarray(pose.translation, float).reshape(3)
            rows.append(r)
    backend.release()
    return pd.DataFrame(rows), seg_info


# ---------------------------------------------------------------------------
# COLMAP database: the verified pair graph.


class PairGraph:
    """Read-only view of `database.db`, keyed by keyframe order index."""

    def __init__(self, db_path: Path, index_by_name: dict[str, int]):
        uri = Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1"
        self.con = sqlite3.connect(uri, uri=True)
        self.index_by_name = index_by_name
        self.image_index = {}
        for image_id, name in self.con.execute("select image_id, name from images"):
            if name in index_by_name:
                self.image_index[image_id] = index_by_name[name]
        self.kf_image = {v: k for k, v in self.image_index.items()}
        cam = self.con.execute("select model, width, height, params from cameras").fetchone()
        params = np.frombuffer(cam[3], np.float64)
        # PINHOLE: fx fy cx cy (global_solve.solve sets camera_model PINHOLE)
        self.K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1.0]])
        self._kp_cache = {}
        rows = []
        raw = {pid: r for pid, r in self.con.execute("select pair_id, rows from matches")}
        for pid, nrows, cfg in self.con.execute("select pair_id, rows, config from two_view_geometries"):
            id2 = pid % COLMAP_PAIR_BASE
            id1 = (pid - id2) // COLMAP_PAIR_BASE
            if id1 not in self.image_index or id2 not in self.image_index:
                continue
            a, b = self.image_index[id1], self.image_index[id2]
            rows.append({"pair_id": pid, "image_id1": id1, "image_id2": id2,
                         "a": min(a, b), "b": max(a, b), "swapped": a > b,
                         "raw_matches": int(raw.get(pid, 0)), "inliers": int(nrows),
                         "config": int(cfg)})
        self.pairs = pd.DataFrame(rows)
        if len(self.pairs):
            self.pairs["gap"] = self.pairs["b"] - self.pairs["a"]
            self.pairs["config_name"] = self.pairs["config"].map(TWO_VIEW_CONFIG)
            self.pairs["verified"] = self.pairs["inliers"] >= MIN_VERIFIED_INLIERS
        self._by_ab = {(int(r.a), int(r.b)): r for r in self.pairs.itertuples(index=False)}

    def keypoints(self, image_id: int) -> np.ndarray:
        if image_id not in self._kp_cache:
            row = self.con.execute("select rows, cols, data from keypoints where image_id=?",
                                   (image_id,)).fetchone()
            if row is None or row[2] is None:
                self._kp_cache[image_id] = np.zeros((0, 2), np.float32)
            else:
                kp = np.frombuffer(row[2], np.float32).reshape(row[0], row[1])
                self._kp_cache[image_id] = kp[:, :2].copy()
        return self._kp_cache[image_id]

    def pair(self, a: int, b: int):
        return self._by_ab.get((min(a, b), max(a, b)))

    def inlier_matches(self, a: int, b: int):
        """(feature idx in a, feature idx in b) inlier correspondences, a/b in keyframe order."""
        p = self.pair(a, b)
        if p is None or p.inliers == 0:
            return None
        row = self.con.execute("select rows, cols, data from two_view_geometries where pair_id=?",
                               (p.pair_id,)).fetchone()
        m = np.frombuffer(row[2], np.uint32).reshape(row[0], row[1]).astype(np.int64)
        # columns are (image_id1, image_id2); image_id1 < image_id2
        ia, ib = self.image_index[p.image_id1], self.image_index[p.image_id2]
        if ia == a:
            return m[:, 0], m[:, 1]
        return m[:, 1], m[:, 0]

    def relative_pose(self, a: int, b: int, config: int | None = None):
        """Two-view relative pose b_from_a from the verified inliers.

        E + recoverPose for calibrated/uncalibrated pairs; for
        planar/panoramic pairs a homography, whose rotation K^-1 H K is exact
        for a pure rotation (reported as model 'H', translation unknown).
        Returns dict(model, R, t (unit or None), n) or None.
        """
        import cv2

        mm = self.inlier_matches(a, b)
        if mm is None or len(mm[0]) < 8:
            return None
        pa = self.keypoints(self.kf_image[a])[mm[0]].astype(np.float64)
        pb = self.keypoints(self.kf_image[b])[mm[1]].astype(np.float64)
        p = self.pair(a, b)
        cfg = p.config if config is None else config
        K = self.K
        if cfg in (4, 5, 6):
            H, _ = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
            if H is None:
                return None
            R = project_to_so3(np.linalg.inv(K) @ H @ K / np.cbrt(np.linalg.det(np.linalg.inv(K) @ H @ K)))
            return {"model": "H", "R": R, "t": None, "n": len(pa)}
        E, mask = cv2.findEssentialMat(pa, pb, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            return None
        _, R, t, _ = cv2.recoverPose(E, pa, pb, K, mask=mask)
        return {"model": "E", "R": R, "t": np.asarray(t, float).reshape(3), "n": len(pa)}


# ---------------------------------------------------------------------------
# The per-keyframe table.


EDT = timezone(timedelta(hours=-4))


def _solution_poses(solution: dict, keyframe_ids: list[str]):
    """keyframe index -> (component, R_cw, t_cw, R_wc, C, observations)."""
    out = {}
    for i, kid in enumerate(keyframe_ids):
        p = solution["poses"].get(kid)
        if not p:
            continue
        R_cw = np.asarray(p["rotation"], float).reshape(3, 3)
        t_cw = np.asarray(p["translation"], float).reshape(3)
        out[i] = {"component": int(p["component"]), "R_cw": R_cw, "t_cw": t_cw,
                  "R_wc": R_cw.T, "C": -R_cw.T @ t_cw, "observations": int(p["observations"])}
    return out


def per_keyframe_reprojection(solution: dict, keyframe_ids: list[str], poses: dict) -> pd.DataFrame:
    arr = solution["arrays"]
    obs = arr["observations"].reshape(-1, 3)
    xy = arr["observation_xy"].reshape(-1, 2).astype(np.float64)
    cam = solution.get("camera")
    rows = []
    if cam is None or not len(xy):
        return pd.DataFrame(columns=["index"])
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    xyz = arr["xyz"].astype(np.float64)
    track = arr["track_length"]
    # per-point triangulation angle: max ray angle over its observing cameras
    centres = {i: p["C"] for i, p in poses.items()}
    point_rays: dict[int, list[np.ndarray]] = defaultdict(list)
    for kf, _feat, pt in obs:
        if kf in centres:
            v = xyz[pt] - centres[kf]
            n = np.linalg.norm(v)
            if n > 0:
                point_rays[int(pt)].append(v / n)
    tri = np.full(len(xyz), np.nan)
    for pt, rays in point_rays.items():
        if len(rays) >= 2:
            R = np.stack(rays)
            d = np.clip(R @ R.T, -1, 1)
            tri[pt] = float(np.degrees(np.arccos(d.min())))
    order = np.argsort(obs[:, 0], kind="stable")
    obs_s, xy_s = obs[order], xy[order]
    bounds = np.searchsorted(obs_s[:, 0], np.arange(len(keyframe_ids) + 1))
    for i in range(len(keyframe_ids)):
        lo, hi = bounds[i], bounds[i + 1]
        if i not in poses or hi <= lo:
            rows.append({"index": i})
            continue
        p = poses[i]
        pts = xyz[obs_s[lo:hi, 2]]
        cam_pts = (p["R_cw"] @ pts.T).T + p["t_cw"]
        z = cam_pts[:, 2]
        ok = z > 1e-9
        u = fx * cam_pts[:, 0] / np.where(ok, z, 1) + cx
        v = fy * cam_pts[:, 1] / np.where(ok, z, 1) + cy
        e = np.hypot(u - xy_s[lo:hi, 0], v - xy_s[lo:hi, 1])
        e[~ok] = np.inf
        t = tri[obs_s[lo:hi, 2]]
        rows.append({
            "index": i, "g_obs_rows": int(hi - lo),
            "g_reproj_median_px": float(np.median(e)),
            "g_reproj_p90_px": float(np.percentile(e[np.isfinite(e)], 90)) if np.isfinite(e).any() else np.nan,
            "g_behind_camera": int((~ok).sum()),
            "g_depth_median": float(np.median(z[ok])) if ok.any() else np.nan,
            "g_track_len_median": float(np.median(track[obs_s[lo:hi, 2]])),
            "g_tri_angle_median_deg": float(np.nanmedian(t)) if np.isfinite(t).any() else np.nan,
            "g_frac_points_tri_lt2deg": float(np.nanmean(t < 2.0)) if np.isfinite(t).any() else np.nan,
        })
    return pd.DataFrame(rows)


def estimate_up(R_wc_list: list[np.ndarray]) -> dict:
    """World 'up' from camera axes (OpenCV: x right, y DOWN, z forward).

    Primary: the direction most orthogonal to every camera's x axis (smallest
    eigenvector of sum x x^T) -- head roll is small, so camera x is nearly
    horizontal whatever the pitch. Sign from the robust mean of camera -y.
    Also reported: the robust (angle-trimmed, 3 passes) mean of camera -y,
    which is biased by mean pitch.
    """
    X = np.stack([R[:, 0] for R in R_wc_list])
    Y = np.stack([-R[:, 1] for R in R_wc_list])
    m = Y.mean(0)
    m /= np.linalg.norm(m)
    for _ in range(3):
        ang = np.degrees(np.arccos(np.clip(Y @ m, -1, 1)))
        keep = ang < max(30.0, np.percentile(ang, 80))
        m = Y[keep].mean(0)
        m /= np.linalg.norm(m)
    w, V = np.linalg.eigh(X.T @ X)
    up = V[:, 0]
    if up @ m < 0:
        up = -up
    return {"up": up, "up_minus_y_mean": m,
            "angle_between_deg": angle_between_deg(up, m),
            "x_axis_residual_deg_median": float(np.median(np.degrees(np.arcsin(np.clip(np.abs(X @ up), 0, 1))))),
            "minus_y_spread_deg_median": float(np.median(np.degrees(np.arccos(np.clip(Y @ up, -1, 1)))))}


def topdown_basis(up: np.ndarray, centres: np.ndarray):
    """Orthonormal (e1, e2) spanning the plane perpendicular to `up`,
    e1 along the principal horizontal spread of the camera centres."""
    P = centres - centres.mean(0)
    P = P - np.outer(P @ up, up)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    e1 = Vt[0] - (Vt[0] @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return e1, e2


def load_regions(regions_csv, keyframe_ids: list[str]) -> pd.DataFrame | None:
    if regions_csv is None or not Path(regions_csv).exists():
        return None
    reg = pd.read_csv(regions_csv)
    if "keyframe_id" in reg.columns:
        m = reg.set_index("keyframe_id")
        return pd.DataFrame({"keyframe_id": keyframe_ids,
                             "region": [m["region"].get(k, None) if k in m.index else None for k in keyframe_ids],
                             "region_confidence": [m["confidence"].get(k, None) if ("confidence" in m.columns and k in m.index) else None for k in keyframe_ids]})
    return None


def load_keyframe_table(world_dir, session_id: str | None = None, *, captures_root=None,
                        capture_dirs=None, replay: bool = True, live_chain: bool = True,
                        regions_csv=None, return_context: bool = False):
    """The per-keyframe forensic table, in capture order. See module docstring.

    `captures_root` (a directory holding `<capture_id>/`) or explicit
    `capture_dirs` enable the frontend replay and raw-frame gap accounting;
    without them those columns are absent. `live_chain=True` re-runs the
    classical backend per segment (tens of seconds). With
    `return_context=True` returns (table, context) where context carries the
    segment table, the pair graph table, the frame replay and summaries.
    """
    paths = resolve_world(world_dir, session_id)
    session = _read_json(paths.session_dir / "session.json")
    kfs = _read_jsonl(paths.session_dir / "keyframes.jsonl")
    events = _read_jsonl(paths.session_dir / "events.jsonl")
    edges = _read_jsonl(paths.session_dir / "edges.jsonl")
    manifest = _read_json(paths.derived_dir / "manifest.json") or {}
    derived_poses = (_read_json(paths.derived_dir / "poses.json") or {}).get("poses", [])
    placements = (_read_json(paths.derived_dir / "placements.json") or {}).get("placements", [])
    solution = load_solution(paths)
    align = _read_json(paths.dense_dir / "align.json")
    ctx: dict = {"paths": paths, "session": session, "manifest": manifest, "notes": []}

    n = len(kfs)
    kid = [k["keyframe_id"] for k in kfs]
    T = pd.DataFrame({"index": np.arange(n), "keyframe_id": kid})
    T["source_seq"] = [k["source_seq"] for k in kfs]
    T["image_name"] = [Path(k["image_relpath"]).name for k in kfs]
    T["received_at"] = [k["received_at"] for k in kfs]
    T["segment_id"] = [int(k["segment_index"]) for k in kfs]
    for src, dst in [("selection_reason", "fe_selection_reason"), ("sharpness", "fe_sharpness"),
                     ("median_parallax_px", "fe_median_parallax_px"), ("overlap_ratio", "fe_overlap_ratio"),
                     ("survival_ratio", "fe_survival_ratio"), ("tracked_count", "fe_tracked_count"),
                     ("homography_residual_px", "fe_homography_residual_px")]:
        T[dst] = [k.get(src) for k in kfs]

    # -- raw capture frames, replay, gaps ------------------------------------
    frames = None
    replayed = None
    if capture_dirs is None and captures_root is not None and session.get("capture_id"):
        capture_dirs = capture_chain(Path(captures_root), session["capture_id"], session)
    if capture_dirs:
        frames = load_capture_frames([Path(c) for c in capture_dirs])
    walk_start = float(frames["received_at"].iloc[0]) if frames is not None and len(frames) else float(T["received_at"].iloc[0])
    T["t_s"] = T["received_at"] - walk_start
    T["wallclock_edt"] = [datetime.fromtimestamp(v, EDT).strftime("%H:%M:%S.%f")[:-4] for v in T["received_at"]]
    ctx["walk_start"] = walk_start
    if frames is not None and len(frames):
        pos = {round(float(v), 6): int(i) for i, v in zip(frames["frame_index"], frames["received_at"])}
        T["capture_frame_index"] = [pos.get(round(float(v), 6), np.nan) for v in T["received_at"]]
        T["capture_id"] = [frames["capture_id"].iloc[int(i)] if np.isfinite(i) else None for i in T["capture_frame_index"]]
        if replay:
            replayed = replay_frontend(frames)
            ctx["replay"] = replayed
            acc = replayed.index[replayed["outcome"] == "accept"].to_numpy()
            ctx["replay_matches_journal"] = bool(len(acc) == n and np.array_equal(acc, T["capture_frame_index"].to_numpy()))
            ctx["replay_histogram"] = dict(Counter(replayed.loc[replayed["outcome"] != "accept", "reason"]))
        gap_total, gap_reasons, gap_sec, gap_drop, gap_lost = [], [], [], [], []
        prev = -1
        prev_seq = None
        prev_t = walk_start
        for i in range(n):
            ci = T["capture_frame_index"].iloc[i]
            ci = int(ci) if np.isfinite(ci) else prev + 1
            gap_total.append(ci - prev - 1)
            if replayed is not None:
                seg = replayed.iloc[prev + 1:ci]
                c = Counter(seg["reason"])
                gap_reasons.append(";".join(f"{k}:{v}" for k, v in sorted(c.items())))
                gap_lost.append(int(c.get("tracking_lost", 0)))
            seq = frames["source_seq"].iloc[ci]
            cap = frames["capture_id"].iloc[ci]
            if prev_seq is not None and prev >= 0 and frames["capture_id"].iloc[prev] == cap:
                gap_drop.append(int(seq - prev_seq - 1 - (ci - prev - 1)))
            else:
                gap_drop.append(np.nan)
            gap_sec.append(float(frames["received_at"].iloc[ci]) - prev_t)
            prev, prev_seq, prev_t = ci, seq, float(frames["received_at"].iloc[ci])
        T["gap_frames_rejected"] = gap_total
        T["gap_seconds"] = gap_sec
        T["gap_frames_never_delivered"] = gap_drop
        if replayed is not None:
            T["gap_reject_reasons"] = gap_reasons
            for r in ("blurred", "insufficient_motion", "tracking_degraded", "tracking_lost",
                      "no_motion_evidence", "tracking_held"):
                T[f"gap_{r}"] = [int(dict(x.split(":") for x in s.split(";") if x).get(r, 0)) for s in gap_reasons]
    else:
        T["gap_seconds"] = T["received_at"].diff().fillna(T["received_at"].iloc[0] - walk_start)

    # -- segments and why they were cut --------------------------------------
    seg_start = [True] + [T["segment_id"].iloc[i] != T["segment_id"].iloc[i - 1] for i in range(1, n)]
    T["segment_start"] = seg_start
    # Walk the event journal: tracking_lost precedes the first keyframe of
    # the segment it announces; solve_chain_broken precedes the keyframe_accepted
    # of the BREAKER, which keeps the old index (engine.py, observe()).
    idx_by_kid = {k: i for i, k in enumerate(kid)}
    cause = [""] * n
    cause_event = [np.nan] * n
    pending_lost = None
    breaker_of_next = None
    last_kf = None
    for e in events:
        kind = e["kind"]
        if kind == "tracking_lost":
            pending_lost = e["event_id"]
        elif kind == "solve_chain_broken":
            breaker_of_next = e["event_id"]
        elif kind == "keyframe_accepted":
            i = idx_by_kid.get(e["payload"]["keyframe_id"])
            if i is None:
                continue
            if breaker_of_next is not None:
                # this keyframe broke the chain; the NEXT keyframe starts a segment
                T.loc[i, "breaks_chain"] = True
                T.loc[i, "chain_break_event_id"] = breaker_of_next
                breaker_of_next = None
            if seg_start[i]:
                if i == 0:
                    cause[i] = "session_start"
                elif pending_lost is not None:
                    cause[i] = "tracking_lost"
                    cause_event[i] = pending_lost
                elif last_kf is not None and bool(T.get("breaks_chain", pd.Series(False, index=T.index)).fillna(False).iloc[last_kf]):
                    cause[i] = "solve_chain_broken"
                    cause_event[i] = T.loc[last_kf, "chain_break_event_id"]
                else:
                    cause[i] = "unknown"
            pending_lost = None
            last_kf = i
    if "breaks_chain" not in T:
        T["breaks_chain"] = False
    T["breaks_chain"] = T["breaks_chain"].fillna(False).astype(bool)
    T["segment_cut_cause"] = cause
    T["segment_cut_event_id"] = cause_event

    # -- live chain, as persisted (edges.jsonl) ------------------------------
    edge_by_to = {e["to_keyframe_id"]: e for e in edges}
    for src, dst in [("matches", "live_edge_matches"), ("inliers", "live_edge_inliers"),
                     ("inlier_ratio", "live_edge_inlier_ratio"), ("pose_status", "live_edge_status"),
                     ("degeneracy", "live_edge_degeneracy"), ("median_parallax_deg", "live_edge_tri_deg"),
                     ("median_parallax_px", "live_edge_parallax_px"),
                     ("cheirality_fraction", "live_edge_cheirality"), ("r_h", "live_edge_r_h")]:
        T[dst] = [edge_by_to.get(k, {}).get(src) for k in kid]
    detail = []
    for i in range(n):
        c = T["segment_cut_cause"].iloc[i]
        if c == "solve_chain_broken":
            b = i - 1
            e = edge_by_to.get(kid[b], {})
            detail.append(f"chain broke at {T['image_name'].iloc[b]}: {e.get('pose_status')}/{e.get('degeneracy')} "
                          f"matches {e.get('matches')} inliers {e.get('inliers')} tri {e.get('median_parallax_deg')}")
        elif c == "tracking_lost":
            detail.append(f"frontend survival < loss floor after {T['image_name'].iloc[i - 1]}")
        else:
            detail.append("")
    T["segment_cut_detail"] = detail

    # -- live chain re-run ----------------------------------------------------
    live = None
    if live_chain:
        live, seg_live = rerun_live_chain(paths, kfs, session)
        ctx["live_segments"] = seg_live
        live = live.set_index("keyframe_id").reindex(kid)
        T["live_status"] = live["live_status"].to_numpy()
        T["live_degeneracy"] = live["live_degeneracy"].to_numpy()
        agree = []
        for k, m, inl in zip(kid, live["live_matches"], live["live_inliers"]):
            e = edge_by_to.get(k)
            if e is not None:
                agree.append(e["matches"] == m and e["inliers"] == inl)
        ctx["live_rerun_edge_agreement"] = (int(sum(agree)), len(agree))
        ctx["_live_R_store"] = {k: R for k, R in zip(kid, live["live_R_wc"]) if isinstance(R, np.ndarray)}

    # -- global solution --------------------------------------------------------
    gposes = {}
    if solution is not None:
        index_of = {k: i for i, k in enumerate(solution["keyframe_ids"])}
        sol_ids = solution["keyframe_ids"]
        gp = _solution_poses(solution, sol_ids)
        # re-key to this table's order (identical when the horizon is complete)
        for si, p in gp.items():
            ti = idx_by_kid.get(sol_ids[si])
            if ti is not None:
                gposes[ti] = p
        T["g_in_horizon"] = [k in index_of for k in kid]
        T["g_component"] = [gposes[i]["component"] if i in gposes else np.nan for i in range(n)]
        T["g_observations"] = [gposes[i]["observations"] if i in gposes else 0 for i in range(n)]
        T["g_supported"] = T["g_observations"] >= MIN_IMAGE_OBSERVATIONS
        for j, ax in enumerate("xyz"):
            T[f"g_C{ax}"] = [gposes[i]["C"][j] if i in gposes else np.nan for i in range(n)]
        q = [R_to_quat_wxyz(gposes[i]["R_wc"]) if i in gposes else [np.nan] * 4 for i in range(n)]
        for j, ax in enumerate("wxyz"):
            T[f"g_q{ax}"] = [qq[j] for qq in q]
        rep = per_keyframe_reprojection(solution, sol_ids, gp)
        if "index" in rep and len(rep.columns) > 1:
            rep["keyframe_id"] = [sol_ids[i] for i in rep["index"]]
            rep = rep.drop(columns=["index"]).set_index("keyframe_id").reindex(kid)
            for c in rep.columns:
                T[c] = rep[c].to_numpy()

    # relative motion to the previous keyframe (same component / same segment)
    for c in ["g_rel_rot_deg", "g_step", "g_rel_tdir_x", "g_rel_tdir_y", "g_rel_tdir_z",
              "live_rel_rot_deg", "live_step", "live_rel_tdir_x", "live_rel_tdir_y", "live_rel_tdir_z",
              "live_vs_g_rel_rot_err_deg", "live_vs_g_tdir_err_deg"]:
        T[c] = np.nan
    prev_g = None
    for i in range(n):
        if i in gposes and prev_g is not None and gposes[prev_g]["component"] == gposes[i]["component"]:
            a, b = gposes[prev_g], gposes[i]
            R_ba = b["R_cw"] @ a["R_cw"].T
            t_ba = b["t_cw"] - R_ba @ a["t_cw"]
            T.loc[i, "g_rel_rot_deg"] = rotation_angle_deg(R_ba)
            T.loc[i, "g_step"] = float(np.linalg.norm(b["C"] - a["C"]))
            nt = np.linalg.norm(t_ba)
            if nt > 0:
                T.loc[i, ["g_rel_tdir_x", "g_rel_tdir_y", "g_rel_tdir_z"]] = t_ba / nt
            T.loc[i, "g_prev_index"] = prev_g
        if i in gposes:
            prev_g = i
    if live is not None:
        for i in range(1, n):
            if T["segment_id"].iloc[i] != T["segment_id"].iloc[i - 1]:
                continue
            Ra, Rb = live["live_R_wc"].iloc[i - 1], live["live_R_wc"].iloc[i]
            if Ra is None or Rb is None or (isinstance(Ra, float)) or (isinstance(Rb, float)):
                continue
            R_ba = Rb.T @ Ra  # R_cw_b R_wc_a
            T.loc[i, "live_rel_rot_deg"] = rotation_angle_deg(R_ba)
            Ca, Cb = live["live_C"].iloc[i - 1], live["live_C"].iloc[i]
            have_t = Ca is not None and Cb is not None and not isinstance(Ca, float) and not isinstance(Cb, float)
            if have_t:
                T.loc[i, "live_step"] = float(np.linalg.norm(Cb - Ca))
                # cam_from_world: t_cw = -R_cw C, so
                # t_ba = t_b - R_ba t_a = -R_cw_b C_b + R_cw_b C_a = R_cw_b (C_a - C_b)
                t_ba = Rb.T @ (Ca - Cb)
                nt = np.linalg.norm(t_ba)
                if nt > 0:
                    T.loc[i, ["live_rel_tdir_x", "live_rel_tdir_y", "live_rel_tdir_z"]] = t_ba / nt
            if i in gposes and (i - 1) in gposes and gposes[i]["component"] == gposes[i - 1]["component"]:
                Rg = gposes[i]["R_cw"] @ gposes[i - 1]["R_cw"].T
                T.loc[i, "live_vs_g_rel_rot_err_deg"] = rotation_angle_deg(Rg.T @ R_ba)
                if have_t:
                    tg = gposes[i]["t_cw"] - Rg @ gposes[i - 1]["t_cw"]
                    T.loc[i, "live_vs_g_tdir_err_deg"] = angle_between_deg(tg, t_ba)
        T["live_Cx"] = [c[0] if isinstance(c, np.ndarray) else np.nan for c in live["live_C"]]
        T["live_Cy"] = [c[1] if isinstance(c, np.ndarray) else np.nan for c in live["live_C"]]
        T["live_Cz"] = [c[2] if isinstance(c, np.ndarray) else np.nan for c in live["live_C"]]

    # -- the pair graph (database.db) ------------------------------------------
    graph = None
    dbp = paths.solve_dir / "database.db"
    if dbp.exists():
        graph = PairGraph(dbp, {name: i for i, name in enumerate(T["image_name"])})
        P = graph.pairs
        ctx["pairs"] = P
        cols = {"db_prev_raw_matches": [], "db_prev_inliers": [], "db_prev_config": [],
                "db_prev2_inliers": [], "db_max_inliers_back20": [], "db_verified_back": [],
                "db_loop_verified": [], "tv_model": [], "tv_rot_deg": [], "tv_vs_g_rot_err_deg": [],
                "tv_vs_g_tdir_err_deg": [], "tv_vs_live_rot_err_deg": []}
        verified = P[P["verified"]]
        vb = verified.groupby("b").size()
        va = verified.groupby("a").size()
        loop = verified[verified["gap"] > SEQUENTIAL_OVERLAP]
        loop_count = Counter(loop["a"].tolist() + loop["b"].tolist())
        back = {}
        for r in P.itertuples(index=False):
            if r.gap <= SEQUENTIAL_OVERLAP:
                back[r.b] = max(back.get(r.b, 0), r.inliers)
        # cut strength: verified pairs spanning the boundary (i-1 | i)
        diff = np.zeros(n + 1)
        for r in verified.itertuples(index=False):
            diff[r.a + 1] += 1
            diff[r.b + 1] -= 1
        cut = np.cumsum(diff)[:n]
        T["db_cut_strength"] = cut  # pairs (a < i <= b)
        for i in range(n):
            p = graph.pair(i - 1, i) if i > 0 else None
            p2 = graph.pair(i - 2, i) if i > 1 else None
            cols["db_prev_raw_matches"].append(p.raw_matches if p is not None else np.nan)
            cols["db_prev_inliers"].append(p.inliers if p is not None else np.nan)
            cols["db_prev_config"].append(p.config_name if p is not None else None)
            cols["db_prev2_inliers"].append(p2.inliers if p2 is not None else np.nan)
            cols["db_max_inliers_back20"].append(back.get(i, 0))
            cols["db_verified_back"].append(int(vb.get(i, 0)))
            cols["db_loop_verified"].append(int(loop_count.get(i, 0)))
            rel = graph.relative_pose(i - 1, i) if (p is not None and p.inliers >= MIN_VERIFIED_INLIERS) else None
            if rel is None:
                for c in ("tv_model", "tv_rot_deg", "tv_vs_g_rot_err_deg", "tv_vs_g_tdir_err_deg", "tv_vs_live_rot_err_deg"):
                    cols[c].append(None if c == "tv_model" else np.nan)
                continue
            cols["tv_model"].append(rel["model"])
            cols["tv_rot_deg"].append(rotation_angle_deg(rel["R"]))
            if i in gposes and (i - 1) in gposes and gposes[i]["component"] == gposes[i - 1]["component"]:
                Rg = gposes[i]["R_cw"] @ gposes[i - 1]["R_cw"].T
                cols["tv_vs_g_rot_err_deg"].append(rotation_angle_deg(Rg.T @ rel["R"]))
                tg = gposes[i]["t_cw"] - Rg @ gposes[i - 1]["t_cw"]
                cols["tv_vs_g_tdir_err_deg"].append(angle_between_deg(tg, rel["t"]) if rel["t"] is not None else np.nan)
            else:
                cols["tv_vs_g_rot_err_deg"].append(np.nan)
                cols["tv_vs_g_tdir_err_deg"].append(np.nan)
            if live is not None and T["segment_id"].iloc[i] == T["segment_id"].iloc[i - 1] \
                    and isinstance(live["live_R_wc"].iloc[i], np.ndarray) and isinstance(live["live_R_wc"].iloc[i - 1], np.ndarray):
                R_live = live["live_R_wc"].iloc[i].T @ live["live_R_wc"].iloc[i - 1]
                cols["tv_vs_live_rot_err_deg"].append(rotation_angle_deg(R_live.T @ rel["R"]))
            else:
                cols["tv_vs_live_rot_err_deg"].append(np.nan)
        for c, v in cols.items():
            T[c] = v
        # translation direction is only meaningful for E pairs; blank H ones
        ctx["graph"] = graph

    # -- published (derived) ----------------------------------------------------
    dp = {r["keyframe_id"]: r for r in derived_poses}
    plc = {int(p["segment_index"]): p for p in placements}
    gseg = (manifest.get("global_solve") or {}).get("segments") or {}
    T["pub_status"] = [dp.get(k, {}).get("status") for k in kid]
    T["pub_degeneracy"] = [dp.get(k, {}).get("degeneracy") for k in kid]
    T["pub_observations"] = [dp.get(k, {}).get("observations") for k in kid]
    T["placement_state"] = [plc.get(s, {}).get("state", "unplaced") for s in T["segment_id"]]
    T["placement_reference_segment"] = [plc.get(s, {}).get("reference_segment") for s in T["segment_id"]]
    T["seg_replaced"] = [bool(gseg.get(str(s), {}).get("replaced", False)) for s in T["segment_id"]]
    T["seg_coverage"] = [gseg.get(str(s), {}).get("coverage") for s in T["segment_id"]]
    T["seg_component"] = [gseg.get(str(s), {}).get("component") for s in T["segment_id"]]
    src = []
    for i, k in enumerate(kid):
        row = dp.get(k)
        if row is None:
            src.append("absent")
        elif T["seg_replaced"].iloc[i]:
            src.append("glomap" if row.get("translation") is not None else "glomap-unregistered")
        elif row.get("status") == "anchor":
            src.append("live-chain-anchor-only")
        else:
            src.append("live-chain" if row.get("translation") is not None else "live-chain-refused")
    T["pub_pose_source"] = src
    # world position of the published pose: placement o local pose
    pw = np.full((n, 3), np.nan)
    for i, k in enumerate(kid):
        row = dp.get(k)
        p = plc.get(int(T["segment_id"].iloc[i]))
        if row is None or row.get("translation") is None or p is None or p.get("state") != "registered":
            continue
        R = quat_wxyz_to_R(p["rotation_wxyz"])
        pw[i] = float(p["scale"]) * R @ np.asarray(row["translation"], float) + np.asarray(p["translation"], float)
    T["pub_ref_Cx"], T["pub_ref_Cy"], T["pub_ref_Cz"] = pw[:, 0], pw[:, 1], pw[:, 2]
    T["pub_drawn_in_main_frame"] = [
        (T["placement_state"].iloc[i] == "registered" and np.isfinite(pw[i]).all()
         and T["placement_reference_segment"].iloc[i] == _main_reference(placements, gseg))
        for i in range(n)]

    # -- dense align (join only) --------------------------------------------------
    if align:
        rec = {r["kid"]: r for r in align.get("records", [])}
        T["dense_align_ok"] = [rec.get(k, {}).get("ok") for k in kid]
        T["dense_align_a"] = [rec.get(k, {}).get("a") for k in kid]
        T["dense_align_b"] = [rec.get(k, {}).get("b") for k in kid]
        T["dense_align_held_out_rel"] = [rec.get(k, {}).get("held_out_rel") for k in kid]
        T["dense_align_n_points"] = [rec.get(k, {}).get("n_points") for k in kid]
        ctx["dense_align_kind"] = align.get("kind")
        ctx["dense_backend"] = align.get("backend")

    # -- regions (join only; never an input to anything above) --------------------
    reg = load_regions(regions_csv, kid)
    if reg is not None:
        T["region"] = reg["region"].to_numpy()
        T["region_confidence"] = reg["region_confidence"].to_numpy()

    # -- up vector, top-down, jump detector (main component) ------------------------
    if gposes:
        comps = Counter(p["component"] for p in gposes.values())
        main = comps.most_common(1)[0][0]
        ctx["main_component"] = main
        ctx["_gposes"] = gposes
        T["g_main"] = (T["g_component"] == main) & T["g_supported"]
        apply_topdown(T, ctx, T["g_main"].to_numpy())
    return (T, ctx) if return_context else T


def apply_topdown(T: pd.DataFrame, ctx: dict, reference_mask: np.ndarray, label: str = "main component") -> None:
    """(Re)compute the up vector from the poses in `reference_mask` and project
    every main-component camera onto the plane perpendicular to it.

    `run_trace` calls this a second time with the LARGEST RIGID ISLAND as the
    reference: an up vector averaged over two rigid bodies that are tilted
    against each other describes neither of them.
    """
    gposes = ctx["_gposes"]
    main = ctx["main_component"]
    n = len(T)
    sel = [i for i in range(n) if reference_mask[i] and i in gposes]
    up = estimate_up([gposes[i]["R_wc"] for i in sel])
    C = np.stack([gposes[i]["C"] for i in sel])
    med = np.median(C, 0)
    d = np.linalg.norm(C - med, axis=1)
    keep = d <= np.percentile(d, 95) * 3
    e1, e2 = topdown_basis(up["up"], C[keep])
    up["reference"] = label
    up["reference_keyframes"] = len(sel)
    ctx["up"] = up
    ctx["topdown_basis"] = (e1, e2)
    ctx["robust_extent"] = float(np.percentile(d, 95))
    allC = T[["g_Cx", "g_Cy", "g_Cz"]].to_numpy(dtype=float)
    T["td_x"] = (allC - med) @ e1
    T["td_y"] = (allC - med) @ e2
    T["td_h"] = (allC - med) @ up["up"]
    T.loc[T["g_component"] != main, ["td_x", "td_y", "td_h"]] = np.nan
    _jumps(T, main)


def roll_free_up(R_wc_list) -> dict:
    """Up from camera x axes alone, with its conditioning.

    `eig_mid / eig_min` large means the x axes span a plane well (the
    cameras looked in varied directions) and the estimate is trustworthy;
    near 1 means they were nearly parallel and it is not.
    """
    X = np.stack([R[:, 0] for R in R_wc_list])
    Y = np.stack([-R[:, 1] for R in R_wc_list])
    w, V = np.linalg.eigh(X.T @ X / len(X))
    u = V[:, 0]
    if u @ Y.mean(0) < 0:
        u = -u
    resid = np.degrees(np.arcsin(np.clip(np.abs(X @ u), 0, 1)))
    return {"up": u, "eig_min": float(w[0]), "eig_mid": float(w[1]),
            "well_conditioned": bool(w[1] > 10 * max(w[0], 1e-6) and w[1] > 0.03),
            "x_resid_med_deg": float(np.median(resid))}


def orientation_blocks(T: pd.DataFrame, ctx: dict, col: str) -> pd.DataFrame:
    """Roll-free up of every block in `col` (islands or annotations), and its
    angle to the up of the largest block. Blocks of one rigid, correct
    reconstruction share an up to within head-roll noise (a few degrees)."""
    gposes = ctx["_gposes"]
    M = T[T["g_main"]]
    ups = {}
    rows = []
    for key, g in M.groupby(col, sort=False):
        idx = [i for i in g["index"] if i in gposes]
        if len(idx) < 5:
            continue
        r = roll_free_up([gposes[i]["R_wc"] for i in idx])
        ups[key] = r
        rows.append({col: key, "n": len(idx), "index_runs": " ".join(_runs(sorted(idx))[:6]),
                     "up_x": r["up"][0], "up_y": r["up"][1], "up_z": r["up"][2],
                     "eig_min": r["eig_min"], "eig_mid": r["eig_mid"],
                     "well_conditioned": r["well_conditioned"], "x_resid_med_deg": r["x_resid_med_deg"]})
    out = pd.DataFrame(rows)
    if len(out):
        ref = out.sort_values("n", ascending=False)[out["well_conditioned"]].head(1)
        if len(ref):
            u0 = ups[ref[col].iloc[0]]["up"]
            out["angle_to_largest_deg"] = [angle_between_deg(ups[k]["up"], u0) for k in out[col]]
    return out


def track_cut(T: pd.DataFrame, ctx: dict, solution: dict) -> np.ndarray:
    """Per keyframe i: 3-D points of the main component observed both before
    i and at-or-after i -- the rigid coupling a boundary in capture order
    actually has in the bundle adjustment."""
    t_index = {k: i for i, k in enumerate(T["keyframe_id"])}
    kf_t = np.array([t_index.get(k, -1) for k in solution["keyframe_ids"]])
    arr = solution["arrays"]
    obs = arr["observations"].reshape(-1, 3)
    comp = arr["component"]
    main = ctx.get("main_component", 0)
    keep = comp[obs[:, 2]] == main
    kf = kf_t[obs[keep, 0]]
    pt = obs[keep, 2]
    lo = pd.Series(kf).groupby(pt).min().to_numpy()
    hi = pd.Series(kf).groupby(pt).max().to_numpy()
    n = len(T)
    diff = np.zeros(n + 1)
    np.add.at(diff, lo + 1, 1)
    np.add.at(diff, hi + 1, -1)
    return np.cumsum(diff)[:n]


def _main_reference(placements: list[dict], gseg: dict):
    counts = Counter(p.get("reference_segment") for p in placements if p.get("state") == "registered")
    return counts.most_common(1)[0][0] if counts else None


def _jumps(T: pd.DataFrame, main: int, window: int = 8) -> None:
    """Flag robust outliers in per-step translation and rotation (main component)."""
    step = T["g_step"].where(T["g_main"] & T["g_main"].shift(1, fill_value=False)).to_numpy()
    ratio = np.full(len(T), np.nan)
    for i in range(len(T)):
        if not np.isfinite(step[i]):
            continue
        lo, hi = max(0, i - window), min(len(T), i + window + 1)
        neigh = np.concatenate([step[lo:i], step[i + 1:hi]])
        neigh = neigh[np.isfinite(neigh)]
        if len(neigh) >= 3 and np.median(neigh) > 0:
            ratio[i] = step[i] / np.median(neigh)
    T["g_step_local_ratio"] = ratio
    T["g_step_z"] = robust_z(step)
    rot = T["g_rel_rot_deg"].where(T["g_main"] & T["g_main"].shift(1, fill_value=False)).to_numpy()
    T["g_rot_z"] = robust_z(rot)
    speed = step / np.maximum(T["gap_seconds"].to_numpy(dtype=float), 1e-3)
    T["g_speed_per_s"] = speed
    T["g_speed_z"] = robust_z(speed)
    T["jump_translation"] = (T["g_step_local_ratio"] > 4) & (T["g_step_z"] > 5)
    T["jump_rotation"] = (T["g_rot_z"] > 5) & (T["g_rel_rot_deg"] > 25)


# ---------------------------------------------------------------------------
# Per-segment table: chain vs global, and how each segment was published.


def segment_table(T: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    rows = []
    gseg = (ctx["manifest"].get("global_solve") or {}).get("segments") or {}
    live_seg = ctx.get("live_segments", {})
    for seg, g in T.groupby("segment_id", sort=True):
        r = {"segment_id": seg, "first_index": int(g["index"].min()), "last_index": int(g["index"].max()),
             "n_keyframes": len(g), "t_start_s": float(g["t_s"].min()), "t_end_s": float(g["t_s"].max()),
             "cut_cause": g["segment_cut_cause"].iloc[0], "cut_detail": g["segment_cut_detail"].iloc[0],
             "ends_by_chain_break": bool(g["breaks_chain"].iloc[-1]) if "breaks_chain" in g else None}
        if "live_status" in g:
            c = Counter(g["live_status"])
            r.update({f"live_{k}": int(c.get(k, 0)) for k in ("anchor", "solved", "rotation_only", "unavailable")})
            r["live_points"] = live_seg.get(seg, {}).get("live_points")
        if "g_component" in g:
            comps = Counter(int(x) for x in g["g_component"].dropna())
            r["g_components"] = ";".join(f"c{k}:{v}" for k, v in sorted(comps.items()))
            r["g_supported"] = int(g["g_supported"].sum())
        m = gseg.get(str(seg), {})
        r.update({"merge_state": m.get("state"), "replaced": m.get("replaced"), "coverage": m.get("coverage"),
                  "component": m.get("component"), "reference_segment": m.get("reference_segment"),
                  "keyframes_posed": m.get("keyframes_posed"), "points_published": m.get("points"),
                  "median_observations": m.get("median_observations")})
        if "region" in g:
            rc = Counter(x for x in g["region"] if isinstance(x, str))
            r["regions"] = ";".join(f"{k}:{v}" for k, v in rc.most_common())
        # Sim3: live chain centres -> global centres (same keyframes, main or any single component)
        if {"live_Cx", "g_Cx"} <= set(g.columns):
            ok = g[["live_Cx", "g_Cx"]].notna().all(axis=1) & g["g_supported"]
            if ok.sum() and g.loc[ok, "g_component"].nunique() == 1:
                src = g.loc[ok, ["live_Cx", "live_Cy", "live_Cz"]].to_numpy()
                dst = g.loc[ok, ["g_Cx", "g_Cy", "g_Cz"]].to_numpy()
                r["sim3_n"] = int(ok.sum())
                ext = float(np.linalg.norm(dst.max(0) - dst.min(0)))
                r["g_extent"] = ext
                if ok.sum() >= 3 and np.linalg.matrix_rank(src - src.mean(0), tol=1e-9) >= 2:
                    # Free Umeyama on centres. On a rotation-dominant segment the
                    # centres barely move and this fit is ill-conditioned, so the
                    # rotation is ALSO fitted from the orientations and the two
                    # are reported separately.
                    s, R, t = umeyama(src, dst)
                    res = dst - (s * (R @ src.T).T + t)
                    rms = float(np.sqrt((res ** 2).sum(1).mean()))
                    r.update({"sim3_scale": s, "sim3_rms": rms,
                              "sim3_rms_norm": rms / ext if ext > 0 else np.nan})
                    Rls, Rgs = [], []
                    for idx in g.index[ok]:
                        Rl = _live_R(ctx, T.loc[idx, "keyframe_id"])
                        if Rl is not None:
                            Rls.append(Rl)
                            Rgs.append(quat_wxyz_to_R(T.loc[idx, ["g_qw", "g_qx", "g_qy", "g_qz"]].to_numpy(float)))
                    if Rls:
                        Ro = project_to_so3(sum(Rg_ @ Rl_.T for Rg_, Rl_ in zip(Rgs, Rls)))
                        errs = [rotation_angle_deg(Rg_.T @ (Ro @ Rl_)) for Rg_, Rl_ in zip(Rgs, Rls)]
                        r["orient_resid_med_deg"] = float(np.median(errs))
                        r["orient_resid_max_deg"] = float(np.max(errs))
                        r["position_vs_orientation_frame_deg"] = rotation_angle_deg(R.T @ Ro)
                        # scale + translation with the orientation-fitted rotation
                        a = (Ro @ (src - src.mean(0)).T).T
                        b = dst - dst.mean(0)
                        s2 = float((a * b).sum() / max((a * a).sum(), 1e-18))
                        res2 = b - s2 * a
                        rms2 = float(np.sqrt((res2 ** 2).sum(1).mean()))
                        r.update({"sim3_fixedR_scale": s2, "sim3_fixedR_rms_norm": rms2 / ext if ext > 0 else np.nan})
                elif ok.sum() == 2:
                    dl = np.linalg.norm(src[1] - src[0])
                    dg = np.linalg.norm(dst[1] - dst[0])
                    r["sim3_scale"] = dg / dl if dl > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _live_R(ctx, keyframe_id):
    lr = ctx.get("_live_R")
    if lr is None:
        return None
    return lr.get(keyframe_id)


# ---------------------------------------------------------------------------
# Pair-level consistency: does the global solve honour the verified pairs?


def pair_consistency(T: pd.DataFrame, ctx: dict, solution: dict | None = None, max_pairs: int | None = None) -> pd.DataFrame:
    """For every verified pair in the database, compare the global relative
    rotation with the two-view one, and test for DOUBLED STRUCTURE.

    Doubling test: each verified inlier correspondence (f_a, f_b) whose two
    features both carry a 3-D point in the same component should name ONE
    point. Two different points P != Q for the same matched feature pair are
    the same physical point triangulated twice; |P - Q| divided by the depth
    of P from camera a is how far apart the solve put the two copies.
    """
    graph: PairGraph | None = ctx.get("graph")
    if graph is None:
        return pd.DataFrame()
    paths = ctx["paths"]
    if solution is None:
        solution = load_solution(paths)
    arr = solution["arrays"]
    obs = arr["observations"].reshape(-1, 3).astype(np.int64)
    xyz = arr["xyz"].astype(np.float64)
    comp = arr["component"]
    sol_ids = solution["keyframe_ids"]
    t_index = {k: i for i, k in enumerate(T["keyframe_id"])}
    kf_t = np.array([t_index.get(k, -1) for k in sol_ids])
    key = kf_t[obs[:, 0]] * 1_000_000 + obs[:, 1]
    order = np.argsort(key)
    key_s, pt_s = key[order], obs[order, 2]
    gp = _solution_poses(solution, sol_ids)
    gposes = {t_index[sol_ids[i]]: p for i, p in gp.items() if sol_ids[i] in t_index}

    def lookup(kf, feats):
        k = kf * 1_000_000 + feats
        pos = np.searchsorted(key_s, k)
        pos = np.clip(pos, 0, len(key_s) - 1)
        hit = key_s[pos] == k
        out = np.full(len(feats), -1, np.int64)
        out[hit] = pt_s[pos[hit]]
        return out

    rows = []
    P = graph.pairs[graph.pairs["verified"]]
    if max_pairs:
        P = P.head(max_pairs)
    for r in P.itertuples(index=False):
        a, b = int(r.a), int(r.b)
        row = {"a": a, "b": b, "gap": b - a, "inliers": r.inliers, "config": r.config_name}
        ga, gb = gposes.get(a), gposes.get(b)
        same_comp = ga is not None and gb is not None and ga["component"] == gb["component"]
        row["same_component"] = same_comp
        mm = graph.inlier_matches(a, b)
        if mm is not None:
            pa_ = lookup(a, mm[0])
            pb_ = lookup(b, mm[1])
            both = (pa_ >= 0) & (pb_ >= 0)
            row["n_both_3d"] = int(both.sum())
            if both.any():
                same = both & (pa_ == pb_)
                row["n_same_point"] = int(same.sum())
                diff = both & (pa_ != pb_) & (comp[np.maximum(pa_, 0)] == comp[np.maximum(pb_, 0)])
                row["n_diff_point"] = int(diff.sum())
                if diff.any() and ga is not None:
                    Pp, Qp = xyz[pa_[diff]], xyz[pb_[diff]]
                    depth = ((ga["R_cw"] @ Pp.T).T + ga["t_cw"])[:, 2]
                    dn = np.linalg.norm(Pp - Qp, axis=1) / np.maximum(np.abs(depth), 1e-9)
                    row["dup_dist_norm_median"] = float(np.median(dn))
                    row["dup_frac_gt_0p1"] = float(np.mean(dn > 0.1))
        if same_comp:
            rel = graph.relative_pose(a, b, r.config)
            if rel is not None:
                Rg = gb["R_cw"] @ ga["R_cw"].T
                row["tv_model"] = rel["model"]
                row["rot_err_deg"] = rotation_angle_deg(Rg.T @ rel["R"])
                if rel["t"] is not None:
                    tg = gb["t_cw"] - Rg @ ga["t_cw"]
                    row["tdir_err_deg"] = angle_between_deg(tg, rel["t"])
                row["g_rel_rot_deg"] = rotation_angle_deg(Rg)
                row["g_dist"] = float(np.linalg.norm(gb["C"] - ga["C"]))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rigidity: which keyframes the final bundle adjustment actually ties together.


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _label_components(n: int, edges) -> np.ndarray:
    uf = _UnionFind(n)
    for a, b in edges:
        uf.union(int(a), int(b))
    roots = np.array([uf.find(i) for i in range(n)])
    sizes = Counter(roots.tolist())
    order = {r: j for j, (r, _) in enumerate(sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])))}
    return np.array([order[r] for r in roots])


def covisibility(T: pd.DataFrame, ctx: dict, solution: dict | None = None) -> dict:
    """Shared-3-D-point counts between every pair of keyframes in the solution.

    GLOMAP reports a *component* as a connected VIEW GRAPH (verified pairs).
    The final bundle adjustment, however, only ties two cameras together
    through 3-D points both observe. A group of cameras that shares no point
    with the rest of its component is, in the BA, a free rigid body: its
    similarity transform relative to the rest is fixed only by whatever the
    rotation averaging / global positioning left it at. This counts those
    shared points so the rigid structure can be read off.
    """
    if solution is None:
        solution = load_solution(ctx["paths"])
    t_index = {k: i for i, k in enumerate(T["keyframe_id"])}
    sol_ids = solution["keyframe_ids"]
    kf_t = np.array([t_index.get(k, -1) for k in sol_ids])
    obs = solution["arrays"]["observations"].reshape(-1, 3)
    kf = kf_t[obs[:, 0]]
    pt = obs[:, 2]
    order = np.lexsort((kf, pt))
    kf, pt = kf[order], pt[order]
    counts: Counter = Counter()
    starts = np.flatnonzero(np.r_[True, pt[1:] != pt[:-1]])
    ends = np.r_[starts[1:], len(pt)]
    for s, e in zip(starts, ends):
        views = np.unique(kf[s:e])
        views = views[views >= 0]
        for x in range(len(views)):
            for y in range(x + 1, len(views)):
                counts[(int(views[x]), int(views[y]))] += 1
    return {"pair_shared_points": counts, "n": len(T)}


def rigid_islands(T: pd.DataFrame, cov: dict, thresholds=(1, 10, 30)) -> pd.DataFrame:
    """Per keyframe: island id at each shared-point threshold (0 = largest).

    Keyframes with fewer than MIN_IMAGE_OBSERVATIONS observations are left
    out of the edges (they are not published poses) and get their own id.
    """
    n = cov["n"]
    supported = T["g_supported"].to_numpy() if "g_supported" in T else np.ones(n, bool)
    out = pd.DataFrame({"index": np.arange(n)})
    for k in thresholds:
        edges = [(a, b) for (a, b), c in cov["pair_shared_points"].items()
                 if c >= k and supported[a] and supported[b]]
        lab = _label_components(n, edges)
        out[f"rigid_island_k{k}"] = lab
    return out


def island_links(T: pd.DataFrame, cov: dict, labels: np.ndarray, ctx: dict) -> pd.DataFrame:
    """Island-to-island coupling: shared points and verified pairs, with
    the GLOMAP-vs-two-view rotation disagreement on those pairs."""
    shared: Counter = Counter()
    for (a, b), c in cov["pair_shared_points"].items():
        la, lb = labels[a], labels[b]
        if la != lb:
            shared[(min(la, lb), max(la, lb))] += c
    pairs = ctx.get("pair_consistency")
    rows = []
    keys = set(shared)
    if pairs is not None and len(pairs):
        la = labels[pairs["a"].to_numpy()]
        lb = labels[pairs["b"].to_numpy()]
        for x, y in zip(la, lb):
            if x != y:
                keys.add((min(x, y), max(x, y)))
    for (x, y) in sorted(keys):
        r = {"island_a": int(x), "island_b": int(y), "shared_point_observation_pairs": int(shared.get((x, y), 0))}
        if pairs is not None and len(pairs):
            m = ((la == x) & (lb == y)) | ((la == y) & (lb == x))
            q = pairs[m]
            r.update({"verified_pairs": int(len(q)), "inliers_sum": int(q["inliers"].sum()),
                      "inliers_max": int(q["inliers"].max()) if len(q) else 0,
                      "rot_err_med_deg": float(q["rot_err_deg"].median()) if len(q) and "rot_err_deg" in q else np.nan,
                      "same_point_frac": (float(q["n_same_point"].sum() / max(1, q["n_both_3d"].sum()))
                                          if len(q) and "n_same_point" in q else np.nan)})
        rows.append(r)
    return pd.DataFrame(rows)


def sequential_islands(T: pd.DataFrame, ctx: dict) -> np.ndarray:
    """Connected components of the verified pair graph using only pairs
    within the sequential matching window: the walk as continuous footage
    links it, before loop detection glues anything."""
    P = ctx.get("pairs")
    n = len(T)
    if P is None or not len(P):
        return np.zeros(n, int)
    q = P[P["verified"] & (P["gap"] <= SEQUENTIAL_OVERLAP)]
    return _label_components(n, zip(q["a"], q["b"]))


def island_table(T: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for isl, g in T.groupby(col):
        r = {col: int(isl), "n_keyframes": len(g), "index_runs": " ".join(_runs(g["index"].tolist())),
             "t_start_s": float(g["t_s"].min()), "t_end_s": float(g["t_s"].max()),
             "supported": int(g["g_supported"].sum()) if "g_supported" in g else None}
        if "g_component" in g:
            r["g_components"] = ";".join(f"c{int(k)}:{v}" for k, v in Counter(g["g_component"].dropna()).items())
        if "region" in g:
            r["regions"] = ";".join(f"{k}:{v}" for k, v in Counter(x for x in g["region"] if isinstance(x, str)).most_common())
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation joins (region labels are annotations; nothing above reads them).


def label_runs(T: pd.DataFrame, col: str = "region") -> list[tuple[str, int, int]]:
    runs, s = [], 0
    for i in range(1, len(T) + 1):
        if i == len(T) or T[col].iloc[i] != T[col].iloc[s]:
            runs.append((T[col].iloc[s], int(T["index"].iloc[s]), int(T["index"].iloc[i - 1])))
            s = i
    return runs


def region_summary(T: pd.DataFrame, col: str = "region") -> pd.DataFrame:
    """Per contiguous label run: support, parallax, scale proxies, placement."""
    rows = []
    for reg, a, b in label_runs(T, col):
        g = T[(T["index"] >= a) & (T["index"] <= b)]
        m = g[g["g_main"]] if "g_main" in g else g
        r = {"region": reg, "kf": f"{a}-{b}", "n": len(g), "n_main_supported": len(m),
             "t": f"{g['t_s'].min():.1f}-{g['t_s'].max():.1f}",
             "components": ";".join(f"c{int(k)}:{v}" for k, v in Counter(g["g_component"].dropna()).items())
             if "g_component" in g else None}
        for c in ("td_x", "td_y", "td_h", "g_depth_median", "dense_align_a", "dense_align_b",
                  "dense_align_held_out_rel", "g_step", "g_speed_per_s", "g_tri_angle_median_deg",
                  "g_frac_points_tri_lt2deg", "g_reproj_median_px", "g_observations"):
            if c in m:
                r[c] = float(pd.to_numeric(m[c], errors="coerce").median()) if len(m) else np.nan
        for c in ("rigid_island_k1", "rigid_island_k10", "sequential_island"):
            if c in g:
                r[c] = ";".join(f"{k}:{v}" for k, v in Counter(g[c]).most_common(3))
        rows.append(r)
    return pd.DataFrame(rows)


def revisit_consistency(T: pd.DataFrame, ctx: dict, revisits: pd.DataFrame, solution: dict | None = None) -> pd.DataFrame:
    """For annotated revisits (range A and range B see the same things):
    are they tied by verified pairs and shared 3-D points, does the solve
    honour those pairs, and do the two ranges' points occupy the same space?

    `NN B->A` is the median distance from points only B observes to the
    nearest point A observes; `A self` is the same statistic between two
    interleaved halves of A's own points (the density baseline). A ratio far
    above 1 means B's copy of the scene is not where A's is -- or that B also
    sees much A does not; read it with the view description.
    """
    from scipy.spatial import cKDTree

    if solution is None:
        solution = load_solution(ctx["paths"])
    arr = solution["arrays"]
    t_index = {k: i for i, k in enumerate(T["keyframe_id"])}
    kf_t = np.array([t_index.get(k, -1) for k in solution["keyframe_ids"]])
    obs = arr["observations"].reshape(-1, 3)
    okf = kf_t[obs[:, 0]]
    xyz = arr["xyz"].astype(np.float64)
    comp = arr["component"]
    main = ctx.get("main_component", 0)
    graph: PairGraph | None = ctx.get("graph")
    PC = ctx.get("pair_consistency")

    def pts(a, b):
        ids = np.unique(obs[(okf >= a) & (okf <= b), 2])
        return ids[comp[ids] == main]

    rows = []
    for r in revisits.itertuples(index=False):
        a0, a1, b0, b1 = int(r.range_a_start), int(r.range_a_end), int(r.range_b_start), int(r.range_b_end)
        A, B = pts(a0, a1), pts(b0, b1)
        row = {"region": r.region, "A": f"{a0}-{a1}", "B": f"{b0}-{b1}",
               "confidence": getattr(r, "confidence", None), "pts_A": len(A), "pts_B": len(B),
               "shared_pts": len(np.intersect1d(A, B))}
        if graph is not None:
            att = graph.pairs[((graph.pairs["a"].between(a0, a1)) & (graph.pairs["b"].between(b0, b1))) |
                              ((graph.pairs["a"].between(b0, b1)) & (graph.pairs["b"].between(a0, a1)))]
            row["db_pairs_attempted"] = int(len(att))
            row["db_pairs_verified"] = int(att["verified"].sum())
            row["db_inliers_max"] = int(att["inliers"].max()) if len(att) else 0
        if PC is not None and len(PC):
            q = PC[((PC["a"].between(a0, a1)) & (PC["b"].between(b0, b1))) |
                   ((PC["a"].between(b0, b1)) & (PC["b"].between(a0, a1)))]
            row["rot_err_med_deg"] = float(q["rot_err_deg"].median()) if len(q) else np.nan
            row["same_point_frac"] = float(q["n_same_point"].sum() / max(1, q["n_both_3d"].sum())) if len(q) else np.nan
        Aonly, Bonly = np.setdiff1d(A, B), np.setdiff1d(B, A)
        if len(Aonly) > 10 and len(Bonly) > 10:
            d, _ = cKDTree(xyz[Aonly]).query(xyz[Bonly])
            d0, _ = cKDTree(xyz[Aonly[::2]]).query(xyz[Aonly[1::2]])
            row["nn_B_to_A_med"] = float(np.median(d))
            row["nn_A_self_med"] = float(np.median(d0))
            row["nn_ratio"] = row["nn_B_to_A_med"] / max(row["nn_A_self_med"], 1e-12)
            row["pts_centroid_dist"] = float(np.linalg.norm(np.median(xyz[Aonly], 0) - np.median(xyz[Bonly], 0)))
        for rng, tag in (((a0, a1), "A"), ((b0, b1), "B")):
            g = T[(T["index"] >= rng[0]) & (T["index"] <= rng[1]) & T.get("g_main", True)]
            for c in ("td_h", "dense_align_a"):
                if c in g:
                    row[f"{c}_{tag}"] = float(pd.to_numeric(g[c], errors="coerce").median()) if len(g) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def gap_matchability(T: pd.DataFrame, frames: pd.DataFrame, indices, max_features: int = 2000) -> pd.DataFrame:
    """Could the raw frames the frontend refused have carried the walk across?

    For each keyframe index i in `indices`, walk the raw capture frames from
    keyframe i-1 to keyframe i (inclusive) and count SIFT + ratio-test + F-RANSAC
    inliers between CONSECUTIVE raw frames, and between each raw frame and
    keyframe i-1. Reports the weakest consecutive link and how far into the
    gap keyframe i-1 is still matchable. Raw pixels, no undistortion: this
    counts correspondences, it does not estimate geometry.
    """
    import cv2

    sift = cv2.SIFT_create(nfeatures=max_features)
    cache = {}

    def feats(fi):
        if fi not in cache:
            img = cv2.imread(frames["path"].iloc[fi], cv2.IMREAD_GRAYSCALE)
            cache[fi] = sift.detectAndCompute(img, None)
            if len(cache) > 400:
                cache.pop(next(iter(cache)))
        return cache[fi]

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def inliers(f1, f2):
        k1, d1 = feats(f1)
        k2, d2 = feats(f2)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return 0
        m = matcher.knnMatch(d1, d2, k=2)
        good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]
        if len(good) < 8:
            return 0
        p1 = np.float32([k1[g.queryIdx].pt for g in good])
        p2 = np.float32([k2[g.trainIdx].pt for g in good])
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.5, 0.999)
        return int(mask.sum()) if mask is not None else 0

    rows = []
    for i in indices:
        if i <= 0 or not np.isfinite(T["capture_frame_index"].iloc[i]) or not np.isfinite(T["capture_frame_index"].iloc[i - 1]):
            continue
        f0, f1 = int(T["capture_frame_index"].iloc[i - 1]), int(T["capture_frame_index"].iloc[i])
        chain = [inliers(f, f + 1) for f in range(f0, f1)]
        to_prev = [inliers(f0, f) for f in range(f0 + 1, f1 + 1)]
        reach = next((j for j, v in enumerate(to_prev) if v < 30), len(to_prev))
        rows.append({"index": int(i), "raw_frames_in_gap": f1 - f0 - 1,
                     "direct_kf_inliers": to_prev[-1] if to_prev else np.nan,
                     "chain_min_consecutive_inliers": min(chain) if chain else np.nan,
                     "chain_median_consecutive_inliers": float(np.median(chain)) if chain else np.nan,
                     "chain_consecutive_inliers": " ".join(str(v) for v in chain),
                     "prev_kf_matchable_frames_ge30": reach})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots.


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"


def _style(ax, title=None):
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b4b3ae")
    ax.tick_params(colors=INK2, labelsize=8)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


def _seg_lines(ax, T, **kw):
    for i in T.index[T["segment_start"] & (T["index"] > 0)]:
        ax.axvline(T.loc[i, "t_s"], color="#c9c8c3", linewidth=0.6, zorder=0, **kw)


def _region_bands(ax, T):
    if "region" not in T:
        return {}
    regions = [r for r in pd.unique(T["region"]) if isinstance(r, str)]
    colours = {r: PALETTE[j % len(PALETTE)] for j, r in enumerate(sorted(regions))}
    run_start = 0
    for i in range(1, len(T) + 1):
        if i == len(T) or T["region"].iloc[i] != T["region"].iloc[run_start]:
            r = T["region"].iloc[run_start]
            if isinstance(r, str):
                ax.axvspan(T["t_s"].iloc[run_start], T["t_s"].iloc[i - 1] + 0.15, color=colours[r],
                           alpha=0.10, linewidth=0, zorder=0)
            run_start = i
    return colours


def write_plots(T: pd.DataFrame, S: pd.DataFrame, ctx: dict, out_dir: Path, tag: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    have_td = "td_x" in T and T["td_x"].notna().any()
    main = ctx.get("main_component")

    def save(fig, name):
        p = out_dir / f"{name}_{tag}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(p)

    def td_limits(ax):
        x, y = T["td_x"].to_numpy(float), T["td_y"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 5:
            lo = np.percentile(np.c_[x[ok], y[ok]], 1, axis=0)
            hi = np.percentile(np.c_[x[ok], y[ok]], 99, axis=0)
            pad = 0.15 * (hi - lo).max()
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("top-down e1 (solve units)", fontsize=8, color=INK2)
        ax.set_ylabel("top-down e2 (solve units)", fontsize=8, color=INK2)

    if have_td:
        M = T[T["g_main"]]
        # 1) by time, with the path drawn in capture order
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(M["td_x"], M["td_y"], color="#c9c8c3", linewidth=0.8, zorder=1)
        sc = ax.scatter(M["td_x"], M["td_y"], c=M["t_s"], cmap="viridis", s=14, zorder=2)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label("seconds since walk start", fontsize=8)
        for i in M.index[::max(1, len(M) // 25)]:
            ax.annotate(str(int(T.loc[i, "index"])), (T.loc[i, "td_x"], T.loc[i, "td_y"]), fontsize=6, color=INK2)
        J = M[M["jump_translation"] | M["jump_rotation"]]
        ax.scatter(J["td_x"], J["td_y"], s=70, facecolors="none", edgecolors="#e34948", linewidths=1.2, zorder=3,
                   label="flagged jump")
        if len(J):
            ax.legend(fontsize=8, frameon=False)
        _style(ax, f"Top-down trajectory, component {main} (supported poses), coloured by time")
        td_limits(ax)
        save(fig, "topdown_time")

        # 2) by segment: alternating two tones + labels for segments >= 5 kf
        fig, ax = plt.subplots(figsize=(8, 7))
        segs = sorted(M["segment_id"].unique())
        for j, s in enumerate(segs):
            g = M[M["segment_id"] == s]
            colour = PALETTE[j % 2 * 1] if len(g) >= 2 else MUTED
            ax.plot(g["td_x"], g["td_y"], color=colour, linewidth=1.0, marker="o", markersize=2.5)
            if len(g) >= 5:
                ax.annotate(f"s{s}", (g["td_x"].iloc[0], g["td_y"].iloc[0]), fontsize=7, color=INK)
        _style(ax, "Top-down, by live tracker segment (alternating tones; label at segment start)")
        td_limits(ax)
        save(fig, "topdown_segment")

        # 3) by component (all components, each in its own frame -> only main is plotted here)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(M["td_x"], M["td_y"], color=PALETTE[0], linewidth=0.8, marker="o", markersize=2.5,
                label=f"c{main}: {len(M)} supported kf")
        U = T[(T["g_component"] == main) & ~T["g_supported"]]
        ax.scatter(U["td_x"], U["td_y"], marker="x", color="#e34948", s=30,
                   label=f"c{main} with < {MIN_IMAGE_OBSERVATIONS} obs: {len(U)} (unregistered in derived)")
        others = T[T["g_component"].notna() & (T["g_component"] != main)]
        ax.text(0.01, 0.01, "other components (own frames, not drawable here): " +
                ", ".join(f"c{int(c)}: kf {int(g['index'].min())}-{int(g['index'].max())}"
                          for c, g in others.groupby("g_component")), transform=ax.transAxes, fontsize=7, color=INK2)
        ax.legend(fontsize=8, frameon=False)
        _style(ax, "Top-down, by GLOMAP component")
        td_limits(ax)
        save(fig, "topdown_component")

        # 4) by region
        if "region" in T:
            fig, ax = plt.subplots(figsize=(8, 7))
            regions = sorted(r for r in pd.unique(T["region"]) if isinstance(r, str))
            markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
            ax.plot(M["td_x"], M["td_y"], color="#dddcd7", linewidth=0.6, zorder=1)
            handles = []
            for j, r in enumerate(regions):
                g = M[M["region"] == r]
                ax.scatter(g["td_x"], g["td_y"], color=PALETTE[j % len(PALETTE)], marker=markers[j % len(markers)],
                           s=16, zorder=2)
                handles.append(Line2D([], [], color=PALETTE[j % len(PALETTE)], marker=markers[j % len(markers)],
                                      linestyle="", label=f"{r} ({len(g)})"))
            ax.legend(handles=handles, fontsize=8, frameon=False)
            _style(ax, "Top-down, by region label (C0; annotation only)")
            td_limits(ax)
            save(fig, "topdown_region")

            # 4b) small multiples: one panel per region, all others grey
            k = len(regions)
            cols = min(3, k)
            rws = int(math.ceil(k / cols))
            fig, axes = plt.subplots(rws, cols, figsize=(4.2 * cols, 4.0 * rws), squeeze=False)
            for j, r in enumerate(regions):
                ax = axes[j // cols][j % cols]
                ax.scatter(M["td_x"], M["td_y"], color="#dddcd7", s=5)
                g = M[M["region"] == r]
                ax.scatter(g["td_x"], g["td_y"], c=g["t_s"], cmap="viridis", s=12)
                # frustum direction (camera z projected)
                for idx in g.index[::2]:
                    Rw = quat_wxyz_to_R(T.loc[idx, ["g_qw", "g_qx", "g_qy", "g_qz"]].to_numpy(float))
                    e1, e2 = ctx["topdown_basis"]
                    z = Rw[:, 2]
                    L = 0.04 * ctx["robust_extent"] * 4
                    ax.plot([T.loc[idx, "td_x"], T.loc[idx, "td_x"] + L * (z @ e1)],
                            [T.loc[idx, "td_y"], T.loc[idx, "td_y"] + L * (z @ e2)], color=INK2, linewidth=0.4)
                _style(ax, f"{r}: kf {', '.join(_runs(g['index'].tolist()))}")
                td_limits(ax)
            for j in range(k, rws * cols):
                axes[j // cols][j % cols].axis("off")
            fig.tight_layout()
            save(fig, "topdown_region_panels")

    # 5) scale per segment
    if "sim3_scale" in S:
        fig, ax = plt.subplots(figsize=(10, 4))
        scol = "sim3_fixedR_scale" if "sim3_fixedR_scale" in S else "sim3_scale"
        s = S[S[scol].notna() & (S[scol] > 0)]
        ax.scatter(s["t_start_s"], s[scol], s=np.clip(s["n_keyframes"] * 3, 10, 200), color=PALETTE[0],
                   alpha=0.8)
        for r in s.itertuples():
            ax.annotate(f"s{r.segment_id}", (r.t_start_s, getattr(r, scol)), fontsize=6, color=INK2)
        ax.set_yscale("log")
        ax.set_xlabel("segment start (s)", fontsize=8)
        ax.set_ylabel("global / live-chain scale (log)", fontsize=8)
        _region_bands(ax, T)
        _style(ax, "Per-segment scale: Umeyama Sim3 of live-chain centres onto GLOMAP centres (marker size = keyframes)")
        save(fig, "scale_per_segment")

    # 6) correspondences over time
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    if "db_prev_inliers" in T:
        ax.plot(T["t_s"], T["db_prev_inliers"], color=PALETTE[0], linewidth=1, label="GLOMAP db inliers to previous kf")
    ax.plot(T["t_s"], T["live_edge_inliers"], color=PALETTE[1], linewidth=1, label="live-chain epipolar inliers (edges.jsonl)")
    ax.set_ylabel("inliers", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    _seg_lines(ax, T)
    _region_bands(ax, T)
    _style(ax, "Correspondence to the previous keyframe (vertical lines = live segment boundaries)")
    ax = axes[1]
    if "db_cut_strength" in T:
        ax.plot(T["t_s"], T["db_cut_strength"], color=PALETTE[2], linewidth=1, label="verified pairs spanning boundary")
        ax.plot(T["t_s"], T["db_loop_verified"], color=PALETTE[6], linewidth=0.8, label="verified loop pairs (gap > 20)")
        if "g_track_cut" in T:
            ax.plot(T["t_s"], T["g_track_cut"], color=PALETTE[7], linewidth=1.2,
                    label="3-D tracks spanning boundary (final model)")
        ax.set_yscale("symlog", linthresh=10)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    _seg_lines(ax, T)
    _style(ax, "View-graph connectivity across each boundary in capture order")
    ax = axes[2]
    if "g_observations" in T:
        ax.plot(T["t_s"], T["g_observations"], color=PALETTE[0], linewidth=1, label="3-D observations of the kf")
        ax.axhline(MIN_IMAGE_OBSERVATIONS, color="#e34948", linewidth=0.8, linestyle="--",
                   label=f"support floor {MIN_IMAGE_OBSERVATIONS}")
        ax.set_yscale("symlog", linthresh=30)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    _seg_lines(ax, T)
    _style(ax, "Sparse support per keyframe (GLOMAP)")
    ax.set_xlabel("seconds since walk start", fontsize=8)
    save(fig, "correspondence_over_time")

    # 7) jump detector
    if "g_step" in T:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        ax = axes[0]
        ax.plot(T["t_s"], T["g_step_local_ratio"], color=PALETTE[0], linewidth=1)
        J = T[T["jump_translation"]]
        ax.scatter(J["t_s"], J["g_step_local_ratio"], color="#e34948", s=25, zorder=3)
        for r in J.itertuples():
            ax.annotate(str(r.index), (r.t_s, r.g_step_local_ratio), fontsize=6, color=INK2)
        ax.set_yscale("log")
        ax.axhline(4, color="#e34948", linewidth=0.6, linestyle="--")
        _seg_lines(ax, T)
        _region_bands(ax, T)
        _style(ax, "Translation step / local median step (GLOMAP, main component)")
        ax = axes[1]
        ax.plot(T["t_s"], T["g_rel_rot_deg"], color=PALETTE[1], linewidth=1, label="GLOMAP rel. rotation")
        if "tv_rot_deg" in T:
            ax.plot(T["t_s"], T["tv_rot_deg"], color=MUTED, linewidth=0.8, label="two-view rel. rotation (db)")
        ax.set_ylabel("deg", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        _seg_lines(ax, T)
        _style(ax, "Rotation step between consecutive keyframes")
        ax = axes[2]
        if "tv_vs_g_rot_err_deg" in T:
            ax.plot(T["t_s"], T["tv_vs_g_rot_err_deg"], color=PALETTE[6], linewidth=1, label="GLOMAP vs two-view (db)")
        if "live_vs_g_rel_rot_err_deg" in T:
            ax.plot(T["t_s"], T["live_vs_g_rel_rot_err_deg"], color=PALETTE[2], linewidth=1, label="GLOMAP vs live chain")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel("deg", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        _seg_lines(ax, T)
        _style(ax, "Relative-rotation disagreement, consecutive keyframes")
        ax.set_xlabel("seconds since walk start", fontsize=8)
        save(fig, "jumps_over_time")

    # 8) published vs refused timeline
    fig, ax = plt.subplots(figsize=(12, 3.8))
    lanes = [
        ("frontend: raw frames rejected in gap", "gap_frames_rejected"),
        ("live chain status", "live_status"),
        ("GLOMAP component", "g_component"),
        ("published pose source", "pub_pose_source"),
        ("region", "region"),
    ]
    yticks, ylabels = [], []
    for li, (label, col) in enumerate(lanes):
        if col not in T:
            continue
        y = len(lanes) - li
        yticks.append(y)
        ylabels.append(label)
        vals = T[col]
        if col == "gap_frames_rejected":
            ax.bar(T["t_s"], np.clip(vals.to_numpy(float) / 20.0, 0, 0.8), bottom=y - 0.4, width=0.25,
                   color=MUTED)
            continue
        cats = [v for v in pd.unique(vals) if v is not None and not (isinstance(v, float) and np.isnan(v))]
        cmap = {c: PALETTE[j % len(PALETTE)] for j, c in enumerate(sorted(cats, key=str))}
        colours = [cmap.get(v, "#ffffff") if not (isinstance(v, float) and np.isnan(v)) else "#ffffff" for v in vals]
        ax.scatter(T["t_s"], np.full(len(T), y), c=colours, marker="|", s=160, linewidths=2)
        txt = "  ".join(f"{c}" for c in sorted(cats, key=str))
        for j, c in enumerate(sorted(cats, key=str)):
            ax.text(1.005, y - 0.25 + 0.5 * j / max(1, len(cats)), str(c), color=cmap[c], fontsize=6,
                    transform=ax.get_yaxis_transform())
    _seg_lines(ax, T)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("seconds since walk start", fontsize=8)
    _style(ax, "What happened to each keyframe, in capture order")
    save(fig, "timeline")

    # 9) scale drift proxies: dense align a (SfM units per depth-unit) and point depth
    if "dense_align_a" in T and T["dense_align_a"].notna().any():
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax = axes[0]
        ax.plot(T["t_s"], T["dense_align_a"].astype(float), color=PALETTE[0], linewidth=1, marker="o", markersize=2)
        ax.set_ylabel("align a", fontsize=8)
        _region_bands(ax, T)
        _seg_lines(ax, T)
        _style(ax, f"Depth-alignment scale per keyframe ({ctx.get('dense_backend')}; SfM depth ~ a * predicted + b)")
        ax = axes[1]
        if "g_depth_median" in T:
            dm = T["g_depth_median"].where(T["g_main"]) if "g_main" in T else T["g_depth_median"]
            ax.plot(T["t_s"], dm, color=PALETTE[1], linewidth=1, marker="o", markersize=2)
            ax.set_ylabel("median point depth", fontsize=8)
        _region_bands(ax, T)
        _seg_lines(ax, T)
        _style(ax, f"Median sparse point depth per keyframe (solve units; component {main} supported poses only)")
        ax.set_xlabel("seconds since walk start", fontsize=8)
        save(fig, "scale_proxies_over_time")

    # 10) rigid islands (shared-point connectivity) top-down + side view
    if have_td and "rigid_island_k1" in T:
        M = T[T["g_main"]]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        for ax, col, title in ((axes[0], "rigid_island_k1", "rigid islands, >= 1 shared 3-D point"),
                               (axes[1], "rigid_island_k10", "rigid islands, >= 10 shared 3-D points")):
            if col not in M:
                continue
            ax.scatter(M["td_x"], M["td_y"], color="#dddcd7", s=5)
            isl = Counter(M[col]).most_common()
            for j, (lab, cnt) in enumerate(isl[:7]):
                g = M[M[col] == lab]
                ax.scatter(g["td_x"], g["td_y"], color=PALETTE[j % len(PALETTE)], s=14,
                           label=f"island {lab}: {cnt} kf ({' '.join(_runs(g['index'].tolist())[:4])})")
            ax.legend(fontsize=7, frameon=False, loc="upper left")
            _style(ax, f"Top-down, {title}")
            td_limits(ax)
        fig.tight_layout()
        save(fig, "topdown_rigid_islands")

    if have_td:
        M = T[T["g_main"]]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        key = "region" if "region" in T else "rigid_island_k1" if "rigid_island_k1" in T else None
        if key is not None:
            cats = sorted({x for x in M[key] if isinstance(x, (str, int, np.integer))}, key=str)
            for j, cval in enumerate(cats):
                g = M[M[key] == cval]
                ax.scatter(g["td_x"], g["td_h"], color=PALETTE[j % len(PALETTE)], s=12, label=f"{cval} ({len(g)})")
            ax.legend(fontsize=7, frameon=False)
        else:
            ax.scatter(M["td_x"], M["td_h"], color=PALETTE[0], s=12)
        ax.set_xlabel("top-down e1 (solve units)", fontsize=8)
        ax.set_ylabel("height along estimated up (solve units)", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        _style(ax, "Side view: camera height vs e1 (a walking wearer's eye height should be nearly constant)")
        save(fig, "sideview")
    return written


def _runs(idx: list[int]) -> list[str]:
    out = []
    if not idx:
        return out
    start = prev = idx[0]
    for v in idx[1:]:
        if v != prev + 1:
            out.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = v
        prev = v
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return out


# ---------------------------------------------------------------------------
# One call: world in, tables + plots out.


def run_trace(world_dir, out_dir, *, session_id=None, captures_root=None, capture_dirs=None,
              regions_csv=None, revisits_csv=None, tag=None, replay=True, live_chain=True, pairs=True,
              gaps: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T, ctx = load_keyframe_table(world_dir, session_id, captures_root=captures_root, capture_dirs=capture_dirs,
                                 replay=replay, live_chain=live_chain, regions_csv=regions_csv,
                                 return_context=True)
    paths = ctx["paths"]
    tag = tag or paths.world_id[:8]
    # live-chain rotations for the Sim3 rotation residual
    if live_chain:
        ctx["_live_R"] = _live_rotations(T, ctx)
    S = segment_table(T, ctx)
    solution = load_solution(paths)
    if pairs and ctx.get("graph") is not None and solution is not None:
        ctx["pair_consistency"] = pair_consistency(T, ctx, solution)
        ctx["pair_consistency"].to_csv(out_dir / f"pairs_{tag}.csv", index=False)
        _pair_plot(ctx["pair_consistency"], T, out_dir / "plots", tag)
    if solution is not None:
        cov = covisibility(T, ctx, solution)
        isl = rigid_islands(T, cov)
        for c in isl.columns[1:]:
            T[c] = isl[c].to_numpy()
        if ctx.get("pairs") is not None:
            T["sequential_island"] = sequential_islands(T, ctx)
        tables = []
        for col in [c for c in T.columns if c.startswith("rigid_island_k") or c == "sequential_island"]:
            t = island_table(T, col)
            t.insert(0, "labelling", col)
            t = t.rename(columns={col: "island"})
            tables.append(t)
        pd.concat(tables).to_csv(out_dir / f"islands_{tag}.csv", index=False)
        links = []
        for col in [c for c in T.columns if c.startswith("rigid_island_k") or c == "sequential_island"]:
            lk = island_links(T, cov, T[col].to_numpy(), ctx)
            lk.insert(0, "labelling", col)
            links.append(lk)
        pd.concat(links).to_csv(out_dir / f"island_links_{tag}.csv", index=False)
        T["g_track_cut"] = track_cut(T, ctx, solution)
        if "g_main" in T and T["g_main"].any():
            largest = Counter(T.loc[T["g_main"], "rigid_island_k1"]).most_common(1)[0][0]
            apply_topdown(T, ctx, (T["g_main"] & (T["rigid_island_k1"] == largest)).to_numpy(),
                          label=f"largest rigid island (k1) {largest}")
            ob = [orientation_blocks(T, ctx, c).assign(labelling=c)
                  for c in ("rigid_island_k1", "rigid_island_k30", "sequential_island") if c in T]
            if "region" in T:
                T["_region_run"] = [f"{r}:{a}-{b}" for r, a, b in label_runs(T) for _ in range(b - a + 1)]
                ob.append(orientation_blocks(T, ctx, "_region_run").assign(labelling="region_run"))
                T.drop(columns=["_region_run"], inplace=True)
            pd.concat(ob).to_csv(out_dir / f"orientation_blocks_{tag}.csv", index=False)
    if "region" in T:
        region_summary(T).to_csv(out_dir / f"regions_summary_{tag}.csv", index=False)
    if revisits_csv is not None and Path(revisits_csv).exists() and solution is not None:
        revisit_consistency(T, ctx, pd.read_csv(revisits_csv), solution).to_csv(
            out_dir / f"revisits_{tag}.csv", index=False)
    if gaps and "capture_frame_index" in T and ctx.get("replay") is not None:
        frames = load_capture_frames(capture_dirs) if capture_dirs else None
        if frames is None and captures_root is not None:
            frames = load_capture_frames(capture_chain(Path(captures_root), ctx["session"]["capture_id"], ctx["session"]))
        idx = T.index[(T["segment_cut_cause"] == "tracking_lost")].tolist()
        if frames is not None and idx:
            gm = gap_matchability(T, frames, idx)
            gm.to_csv(out_dir / f"gaps_{tag}.csv", index=False)
            ctx["gaps"] = gm
    T.to_csv(out_dir / f"keyframes_{tag}.csv", index=False)
    S.to_csv(out_dir / f"segments_{tag}.csv", index=False)
    summary = {
        "world_id": paths.world_id, "session_id": paths.session_id, "keyframes": int(len(T)),
        "walk_start_epoch": ctx["walk_start"],
        "replay_matches_journal": ctx.get("replay_matches_journal"),
        "replay_rejection_histogram": ctx.get("replay_histogram"),
        "session_rejected_by_reason": ctx["session"].get("rejected_by_reason"),
        "live_rerun_edge_agreement": ctx.get("live_rerun_edge_agreement"),
        "main_component": ctx.get("main_component"),
        "up": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in (ctx.get("up") or {}).items()},
        "robust_extent_p95": ctx.get("robust_extent"),
    }
    if "replay" in ctx:
        ctx["replay"].to_csv(out_dir / f"frames_{tag}.csv", index=False)
    for col in ("rigid_island_k1", "rigid_island_k10", "rigid_island_k30", "sequential_island"):
        if col in T:
            main_isl = T.loc[T.get("g_main", pd.Series(True, index=T.index)), col]
            summary[f"{col}_sizes_top5"] = [int(v) for _, v in Counter(main_isl).most_common(5)]
    summary["plots"] = [str(p) for p in write_plots(T, S, ctx, out_dir / "plots", tag)]
    with open(out_dir / f"summary_{tag}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return {"table": T, "segments": S, "context": ctx, "summary": summary}


def _live_rotations(T, ctx):
    """Recover the live R_wc per keyframe from the re-run (kept on the context)."""
    return ctx.get("_live_R_store", {})


def _pair_plot(PC: pd.DataFrame, T: pd.DataFrame, out_dir: Path, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if PC.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, col, title, vmax in [
        (axes[0], "rot_err_deg", "GLOMAP vs two-view relative rotation error (deg), verified pairs", 10),
        (axes[1], "dup_dist_norm_median", "Doubled structure: |P-Q| / depth for matched features on 2 points", 0.5),
    ]:
        if col not in PC:
            continue
        d = PC[PC[col].notna()]
        sc = ax.scatter(d["a"], d["b"], c=np.clip(d[col], 0, vmax), cmap="magma_r", s=4, vmin=0, vmax=vmax)
        fig.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_xlabel("keyframe index a", fontsize=8)
        ax.set_ylabel("keyframe index b", fontsize=8)
        if "region" in T:
            for i in T.index[1:]:
                if T.loc[i, "region"] != T.loc[i - 1, "region"]:
                    ax.axvline(i, color="#dddcd7", linewidth=0.4)
                    ax.axhline(i, color="#dddcd7", linewidth=0.4)
        ax.set_aspect("equal")
        _style(ax, title)
    fig.tight_layout()
    fig.savefig(out_dir / f"pairs_matrix_{tag}.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
