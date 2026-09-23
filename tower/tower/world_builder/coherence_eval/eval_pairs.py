"""A verified image-pair set per frozen world, independent of any solver.

Revisit consistency asks: where the images themselves say two keyframes see
the same place, does a reconstruction agree about how the two cameras relate?
The answer must not come from the solver under test, so the pair set is built
ONCE per world from the images alone and every variant is scored against it.

CONSTRUCTION (``PAIR_PARAMS`` records every number)

1. Images: the canonical undistorted keyframes (`eval_world`), grayscale.
2. Candidates:
   * adjacent pairs, capture-order |di| in {1, 2};
   * retrieved pairs: for every keyframe, its ``retrieval_top_k`` most similar
     keyframes with |di| > ``retrieval_min_gap``, by cosine similarity of a
     global descriptor. The descriptor is DINOv2-small (``facebook/dinov2-small``,
     Apache-2.0, run on the CPU for determinism): the L2-normalised
     concatenation of the normalised CLS token and the normalised mean patch
     token of the image resized to 224x392 (portrait, multiple of 14). If
     DINOv2 cannot be loaded the build FAILS, unless a fallback to a 32x56
     zero-mean tiny-image descriptor is explicitly allowed (the manifest's
     ``retrieval_used`` then says so).
3. Matching: RootSIFT (OpenCV SIFT, ``sift_features`` per image),
   mutual nearest neighbours with Lowe's ratio test (``ratio``).
4. Verification: essential matrix by RANSAC in the canonical pinhole camera
   (``ransac_px`` threshold, confidence ``ransac_conf``). A pair is KEPT
   when its RANSAC inliers are at least
   ``min_inliers_local`` (|di| <= ``distant_gap``) or ``min_inliers_distant``
   (|di| > ``distant_gap``, stricter because a false revisit is the costly
   error), its inlier ratio (inliers / mutual matches) is at least
   ``min_inlier_ratio``, and the inliers' bounding box covers at least
   ``min_bbox_fraction`` of both images (a clustered patch is not a view).
5. Per kept pair: R, t (unit) with x_j = R x_i + t (camera i to camera j,
   OpenCV axes). With parallax (median derotated angle between inlier
   bearing rays >= ``min_parallax_deg``) and a cheirality-consistent
   decomposition (`recoverPose`, far threshold 1e4 baselines, keeping >= half
   the inliers), R and t come from the essential matrix and ``t_reliable`` is
   set. Otherwise the pair is rotation-dominant: the essential matrix is
   ill-conditioned there, so R is the rotation-only (Wahba/SVD) fit to the
   inlier bearings and the translation is stored but flagged unreliable (only
   the rotation is scored). Up to ``stored_inliers`` inlier correspondences
   are kept in normalised coordinates for epipolar checks.

The pair set is a MEASUREMENT with its own errors (repetitive texture can
produce a confidently wrong essential matrix; small-parallax translation
directions are poor). It is the same measurement for every variant, so it
ranks variants fairly, but a residual error of a few degrees on a pair is not
by itself proof that the reconstruction is wrong.

LAYOUT (``<cache>/<world_id>/pairs/``)::

    manifest.json   parameters, per-image source sha1, counts, timings
    pairs.npz       i, j (capture-order indices), R (P,3,3), t (P,3), t_reliable,
                    n_matches, n_inliers, parallax_deg, similarity, source,
                    inlier_offsets (P+1,), inlier_xy_i / inlier_xy_j (normalised)
    descriptors.npy (N, D) float32 global descriptors (retrieval provenance)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tower.world_builder.coherence_eval.eval_world import WorldInfo

PAIR_CACHE_VERSION = 1
PAIR_PARAMS = {
    "version": PAIR_CACHE_VERSION,
    "adjacent_gaps": [1, 2],
    "retrieval": "dinov2-small cls+meanpatch 224x392 cpu",
    "retrieval_top_k": 20,
    "retrieval_min_gap": 5,
    "sift_features": 4000,
    "rootsift": True,
    "ratio": 0.8,
    "mutual": True,
    "ransac_px": 1.0,
    "ransac_conf": 0.9999,
    "min_inliers_local": 30,
    "min_inliers_distant": 50,
    "distant_gap": 30,
    "min_inlier_ratio": 0.25,
    "min_bbox_fraction": 0.05,
    "min_parallax_deg": 1.0,
    "stored_inliers": 256,
}

SOURCE_ADJACENT = 0
SOURCE_RETRIEVAL = 1


def pairs_dir(cache_root, world_id: str) -> Path:
    return Path(cache_root) / world_id / "pairs"


class PairSet:
    """Read side of the cache."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.manifest = None
        self.arrays = None
        m = self.root / "manifest.json"
        z = self.root / "pairs.npz"
        if m.is_file() and z.is_file():
            self.manifest = json.loads(m.read_text(encoding="utf-8"))
            with np.load(z, allow_pickle=False) as f:
                self.arrays = {k: f[k] for k in f.files}

    @property
    def available(self) -> bool:
        return self.arrays is not None and bool(self.manifest.get("complete"))

    def __len__(self) -> int:
        return 0 if self.arrays is None else int(len(self.arrays["i"]))

    @classmethod
    def from_arrays(cls, arrays: dict, manifest: dict | None = None) -> "PairSet":
        obj = cls.__new__(cls)
        obj.root = None
        obj.manifest = dict(manifest or {"complete": True, "params": PAIR_PARAMS})
        obj.arrays = arrays
        return obj

    def digest(self) -> str | None:
        if self.arrays is None:
            return None
        h = hashlib.sha1()
        for k in ("i", "j", "R", "t", "t_reliable"):
            h.update(np.ascontiguousarray(self.arrays[k]).tobytes())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# build


def _global_descriptors(images: list[np.ndarray], log, allow_fallback: bool = False) -> tuple[np.ndarray, str]:
    try:
        import torch
        from transformers import AutoModel

        torch.manual_seed(0)
        model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        import cv2

        out = []
        with torch.no_grad():
            for k in range(0, len(images), 16):
                batch = []
                for im in images[k:k + 16]:
                    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else im[:, :, ::-1]
                    rgb = cv2.resize(rgb, (224, 392), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                    batch.append(((rgb - mean) / std).transpose(2, 0, 1))
                x = torch.from_numpy(np.stack(batch))
                h = model(pixel_values=x).last_hidden_state
                cls_ = torch.nn.functional.normalize(h[:, 0], dim=-1)
                pm = torch.nn.functional.normalize(h[:, 1:].mean(1), dim=-1)
                d = torch.nn.functional.normalize(torch.cat([cls_, pm], dim=-1), dim=-1)
                out.append(d.numpy().astype(np.float32))
        return np.concatenate(out), "dinov2-small"
    except Exception as exc:  # noqa: BLE001 -- refused unless asked for, then recorded
        # A different retrieval is a different pair set: two worlds (or two
        # rebuilds of one) would then not be scored against comparable pairs.
        # Observed 2026-09-23: DINOv2 failed to load under commit-charge
        # pressure (WinError 1455) while another process held the GPU stack.
        if not allow_fallback:
            raise RuntimeError(
                f"DINOv2 retrieval unavailable ({type(exc).__name__}: {exc}); rerun when memory is "
                "free, or pass allow_fallback (--allow-descriptor-fallback) to use tiny-image "
                "descriptors, which the manifest then records") from exc
        log(f"[pairs] DINOv2 unavailable ({type(exc).__name__}: {exc}); tiny-image descriptors")
        import cv2

        out = []
        for im in images:
            g = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            t = cv2.resize(g, (32, 56), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            t -= t.mean()
            out.append(t / (np.linalg.norm(t) + 1e-9))
        return np.asarray(out, np.float32), "tiny-image-32x56"


def _features(image: np.ndarray, n: int):
    import cv2

    sift = cv2.SIFT_create(nfeatures=n)
    kps, desc = sift.detectAndCompute(image, None)
    if desc is None or not len(kps):
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    xy = np.array([k.pt for k in kps], np.float32)
    desc = desc.astype(np.float32)
    desc /= np.abs(desc).sum(1, keepdims=True) + 1e-9
    desc = np.sqrt(desc)
    return xy, desc


def _match(d1: np.ndarray, d2: np.ndarray, ratio: float) -> np.ndarray:
    import cv2

    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), np.int64)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    fwd = bf.knnMatch(d1, d2, k=2)
    bwd = bf.knnMatch(d2, d1, k=1)
    back = {m[0].queryIdx: m[0].trainIdx for m in bwd if m}
    out = []
    for m in fwd:
        if len(m) < 2:
            continue
        a, b = m
        if a.distance < ratio * b.distance and back.get(a.trainIdx) == a.queryIdx:
            out.append((a.queryIdx, a.trainIdx))
    return np.asarray(out, np.int64).reshape(-1, 2)


def _bearings(xy_norm: np.ndarray) -> np.ndarray:
    f = np.concatenate([xy_norm, np.ones((len(xy_norm), 1))], 1)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def wahba_rotation(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """R minimising sum |f2 - R f1|^2 over unit bearings (rotation-only model)."""
    H = f1.T @ f2
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def verify_pair(xy1, d1, xy2, d2, K, size, params, distant: bool):
    """Two-view verification of one candidate. Returns a dict or None.

    Support is the essential-matrix RANSAC inlier set. The relative rotation
    comes from the essential matrix (cheirality-checked decomposition) when the
    pair has parallax; for a low-parallax pair (median derotated ray angle
    below ``min_parallax_deg``, or a decomposition whose cheirality check keeps
    under half of the inliers) the essential matrix is ill-conditioned, so the
    rotation is the rotation-only (Wahba) fit to the inlier bearings instead
    and the translation direction is flagged unreliable.
    """
    import cv2

    m = _match(d1, d2, params["ratio"])
    n_matches = len(m)
    floor = params["min_inliers_distant"] if distant else params["min_inliers_local"]
    if n_matches < floor:
        return None
    p1 = xy1[m[:, 0]].astype(np.float64)
    p2 = xy2[m[:, 1]].astype(np.float64)
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=params["ransac_conf"],
                                   threshold=params["ransac_px"])
    if E is None or mask is None or E.shape[0] < 3 or E.shape[1] != 3:
        return None
    E = E[:3]  # several stacked solutions only arise without RANSAC; take the first
    inl = mask.ravel().astype(bool)
    n_inl = int(inl.sum())
    if n_inl < floor or n_inl / max(n_matches, 1) < params["min_inlier_ratio"]:
        return None
    w, h = size
    for p in (p1[inl], p2[inl]):
        span = (p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min())
        if span / float(w * h) < params["min_bbox_fraction"]:
            return None
    Kinv = np.linalg.inv(K)
    n1 = (Kinv @ np.c_[p1[inl], np.ones(n_inl)].T).T[:, :2]
    n2 = (Kinv @ np.c_[p2[inl], np.ones(n_inl)].T).T[:, :2]
    b1, b2 = _bearings(n1), _bearings(n2)
    # cheirality with a generous far-point threshold: at low parallax the
    # default (50 baselines) rejects nearly every point as "at infinity"
    out = cv2.recoverPose(E, p1[inl], p2[inl], cameraMatrix=K, distanceThresh=1e4)
    n_cheir, R_E, t_E = int(out[0]), out[1], out[2]
    R_rot = wahba_rotation(b1, b2)

    def parallax_of(R):
        return float(np.median(np.degrees(np.arccos(np.clip(((b1 @ R.T) * b2).sum(1), -1, 1)))))

    par_E = parallax_of(R_E)
    if par_E >= params["min_parallax_deg"] and n_cheir >= 0.5 * n_inl:
        R, t, reliable, par, model = R_E, t_E.ravel(), True, par_E, "essential"
    else:
        R, t, reliable, par, model = R_rot, t_E.ravel(), False, parallax_of(R_rot), "rotation"
    return {"R": R, "t": t / (np.linalg.norm(t) + 1e-12), "n_matches": n_matches,
            "n_inliers": n_inl, "n_cheirality": int(n_cheir), "parallax_deg": par,
            "t_reliable": bool(reliable), "model": model,
            "xy1": n1[: params["stored_inliers"]].astype(np.float32),
            "xy2": n2[: params["stored_inliers"]].astype(np.float32)}


def candidate_pairs(n: int, desc: np.ndarray, params: dict) -> list[tuple[int, int, int, float]]:
    """(i, j, source, similarity) with i < j, deterministic order."""
    sim = desc @ desc.T
    cand: dict[tuple[int, int], tuple[int, float]] = {}
    for gap in params["adjacent_gaps"]:
        for i in range(n - gap):
            cand[(i, i + gap)] = (SOURCE_ADJACENT, float(sim[i, i + gap]))
    idx = np.arange(n)
    for i in range(n):
        s = sim[i].copy()
        s[np.abs(idx - i) <= params["retrieval_min_gap"]] = -np.inf
        # stable ordering: similarity desc, then index asc
        order = np.lexsort((idx, -s))[: params["retrieval_top_k"]]
        for j in order:
            if not np.isfinite(s[j]):
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            cand.setdefault((a, b), (SOURCE_RETRIEVAL, float(sim[a, b])))
    return [(a, b, src, s) for (a, b), (src, s) in sorted(cand.items())]


def build_pair_cache(world: WorldInfo, cache_root, *, workers: int | None = None, log=print,
                     params: dict | None = None, allow_descriptor_fallback: bool = False) -> dict:
    import cv2

    params = dict(params or PAIR_PARAMS)
    root = pairs_dir(cache_root, world.world_id)
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    images, sources = [], {}
    colour = []
    for i in range(world.n):
        img, kind = world.canonical_image(i)
        path, _ = world.canonical_image_source(i)
        if img is None:
            raise RuntimeError(f"keyframe {i} ({world.image_name(i)}): no canonical image ({kind})")
        colour.append(img)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        sources[world.image_name(i)] = {"kind": kind, "sha1": hashlib.sha1(path.read_bytes()).hexdigest()}
    t_load = time.time() - t0
    desc, desc_kind = _global_descriptors(colour, log, allow_descriptor_fallback)
    del colour
    params["retrieval_used"] = desc_kind
    np.save(root / "descriptors.npy", desc)
    t_desc = time.time() - t0 - t_load
    cv2.setNumThreads(1)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        feats = list(pool.map(lambda im: _features(im, params["sift_features"]), images))
    t_feat = time.time() - t0 - t_load - t_desc
    cands = candidate_pairs(world.n, desc, params)
    log(f"[pairs] {world.world_id[:8]}: {world.n} images, {len(cands)} candidates, "
        f"desc={desc_kind}, features {t_feat:.0f}s")
    K = world.canonical_K()
    size = (int(world.canonical_camera["width"]), int(world.canonical_camera["height"]))

    def work(c):
        i, j, src, s = c
        return verify_pair(feats[i][0], feats[i][1], feats[j][0], feats[j][1], K, size, params,
                           distant=(j - i) > params["distant_gap"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, cands))
    cv2.setNumThreads(-1)
    keep = [(c, r) for c, r in zip(cands, results) if r is not None]
    P = len(keep)
    offsets = np.zeros(P + 1, np.int64)
    for k, (_, r) in enumerate(keep):
        offsets[k + 1] = offsets[k] + len(r["xy1"])
    arrays = {
        "i": np.array([c[0] for c, _ in keep], np.int32),
        "j": np.array([c[1] for c, _ in keep], np.int32),
        "source": np.array([c[2] for c, _ in keep], np.int8),
        "similarity": np.array([c[3] for c, _ in keep], np.float32),
        "R": np.array([r["R"] for _, r in keep], np.float64).reshape(-1, 3, 3),
        "t": np.array([r["t"] for _, r in keep], np.float64).reshape(-1, 3),
        "t_reliable": np.array([r["t_reliable"] for _, r in keep], bool),
        "n_matches": np.array([r["n_matches"] for _, r in keep], np.int32),
        "n_inliers": np.array([r["n_inliers"] for _, r in keep], np.int32),
        "parallax_deg": np.array([r["parallax_deg"] for _, r in keep], np.float32),
        "n_cheirality": np.array([r["n_cheirality"] for _, r in keep], np.int32),
        "inlier_offsets": offsets,
        "inlier_xy_i": (np.concatenate([r["xy1"] for _, r in keep]) if keep else np.zeros((0, 2), np.float32)),
        "inlier_xy_j": (np.concatenate([r["xy2"] for _, r in keep]) if keep else np.zeros((0, 2), np.float32)),
    }
    np.savez_compressed(root / "pairs.npz", **arrays)
    gap = arrays["j"] - arrays["i"]
    counts = {
        "candidates": len(cands),
        "candidates_adjacent": sum(1 for c in cands if c[2] == SOURCE_ADJACENT),
        "candidates_retrieved": sum(1 for c in cands if c[2] == SOURCE_RETRIEVAL),
        "verified": P,
        "verified_gap1": int((gap == 1).sum()),
        "verified_gap2_to_distant": int(((gap > 1) & (gap <= params["distant_gap"])).sum()),
        "verified_distant": int((gap > params["distant_gap"]).sum()),
        "verified_t_reliable": int(arrays["t_reliable"].sum()),
        "adjacent_gap1_possible": world.n - 1,
    }
    manifest = {"world_id": world.world_id, "session_id": world.session_id, "params": params,
                "camera": world.canonical_camera, "sources": sources, "counts": counts,
                "complete": True,
                "seconds": {"load": round(t_load, 2), "descriptors": round(t_desc, 2),
                            "features": round(t_feat, 2), "total": round(time.time() - t0, 2)}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    log(f"[pairs] {world.world_id[:8]}: kept {P} ({counts['verified_distant']} distant) in {time.time() - t0:.0f}s")
    return manifest
