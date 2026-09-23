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
SOURCE_XISLAND_EXHAUSTIVE = 2
SOURCE_XISLAND_RETRIEVAL = 3

XISLAND_ARRAYS = "xisland.npz"
XISLAND_MANIFEST = "xisland_manifest.json"
XISLAND_PARAMS = {
    "version": 1,
    "islands_from": "base tier (pairs.npz)",
    "exhaustive_min_island": 3,
    "exhaustive_max_pairs": 250000,
    "retrieval_top_k_small": 50,
    "strict": "the base tier's own floors (min_inliers_local/distant, min_inlier_ratio, min_bbox_fraction)",
    "relaxed_min_inliers": 20,
    "relaxed_min_inlier_ratio": 0.35,
}


def pairs_dir(cache_root, world_id: str) -> Path:
    return Path(cache_root) / world_id / "pairs"


class PairSet:
    """Read side of the cache: the base tier (`pairs.npz`, the comparable one)
    and, when built, the flagged cross-island tier (`xisland.npz`)."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.manifest = None
        self.arrays = None
        self.xmanifest = None
        self.xarrays = None
        m = self.root / "manifest.json"
        z = self.root / "pairs.npz"
        if m.is_file() and z.is_file():
            self.manifest = json.loads(m.read_text(encoding="utf-8"))
            with np.load(z, allow_pickle=False) as f:
                self.arrays = {k: f[k] for k in f.files}
        xm = self.root / XISLAND_MANIFEST
        xz = self.root / XISLAND_ARRAYS
        if xm.is_file() and xz.is_file():
            self.xmanifest = json.loads(xm.read_text(encoding="utf-8"))
            with np.load(xz, allow_pickle=False) as f:
                self.xarrays = {k: f[k] for k in f.files}
        # the optional learned-matcher cross-island tier (ALIKED+LightGlue).
        # The EfficientLoFTR files of versions 1-2 (LOFTR_ARRAYS) are never
        # read: that matcher snapped every match to its 8-px grid (V4b).
        self.lmanifest = None
        self.larrays = None
        lm = self.root / LEARNED_MANIFEST
        lz = self.root / LEARNED_ARRAYS
        if lm.is_file() and lz.is_file():
            self.lmanifest = json.loads(lm.read_text(encoding="utf-8"))
            with np.load(lz, allow_pickle=False) as f:
                self.larrays = {k: f[k] for k in f.files}

    @property
    def available(self) -> bool:
        return self.arrays is not None and bool(self.manifest.get("complete"))

    @property
    def xavailable(self) -> bool:
        return self.xarrays is not None and bool((self.xmanifest or {}).get("complete"))

    def __len__(self) -> int:
        return 0 if self.arrays is None else int(len(self.arrays["i"]))

    @classmethod
    def from_arrays(cls, arrays: dict, manifest: dict | None = None, xarrays: dict | None = None) -> "PairSet":
        obj = cls.__new__(cls)
        obj.root = None
        obj.manifest = dict(manifest or {"complete": True, "params": PAIR_PARAMS})
        obj.arrays = arrays
        obj.xarrays = xarrays
        obj.xmanifest = {"complete": True} if xarrays is not None else None
        obj.lmanifest = None
        obj.larrays = None
        return obj

    def digest(self) -> str | None:
        return _digest(self.arrays)

    def xdigest(self) -> str | None:
        return _digest(self.xarrays)


def _digest(arrays) -> str | None:
    if arrays is None:
        return None
    h = hashlib.sha1()
    for k in ("i", "j", "R", "t", "t_reliable"):
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()[:16]


def islands(n: int, I, J) -> np.ndarray:
    """Connected components ("islands") of the verified pair graph over the n
    keyframes; labels ordered by size (0 = largest), ties by first keyframe.
    Image-only: the same for every variant of a world."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    I = np.asarray(I, dtype=np.int64)
    J = np.asarray(J, dtype=np.int64)
    A = coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab)
    first = np.full(len(sizes), n)
    for i in range(n - 1, -1, -1):
        first[lab[i]] = i
    order = sorted(range(len(sizes)), key=lambda c: (-sizes[c], first[c]))
    remap = np.empty(len(sizes), np.int64)
    remap[order] = np.arange(len(sizes))
    return remap[lab]


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
    m = _match(d1, d2, params["ratio"])
    floor = params["min_inliers_distant"] if distant else params["min_inliers_local"]
    if len(m) < floor:
        return None
    return verify_points(xy1[m[:, 0]], xy2[m[:, 1]], K, size, params, distant)


def verify_points(p1, p2, K, size, params, distant: bool):
    """`verify_pair` from already-matched pixel correspondences (any matcher)."""
    import cv2

    p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(p2, dtype=np.float64).reshape(-1, 2)
    n_matches = len(p1)
    floor = params["min_inliers_distant"] if distant else params["min_inliers_local"]
    if n_matches < max(floor, 5):
        return None
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


def _pack(keep) -> dict:
    P = len(keep)
    offsets = np.zeros(P + 1, np.int64)
    for k, (_, r) in enumerate(keep):
        offsets[k + 1] = offsets[k] + len(r["xy1"])
    return {
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


def build_cross_island_tier(world: WorldInfo, cache_root, *, workers: int | None = None, log=print) -> dict:
    """A SEPARATE, flagged tier that tries to link the islands of the base tier.

    The base tier's retrieval (top-20 per keyframe) can leave the verified pair
    graph split into islands -- on the target world the desk, the closet and
    the 265-382 block share NO verified pair -- and then where one island sits
    relative to another is unobservable. This tier re-matches ACROSS islands:

    * exhaustively: every keyframe pair between two different islands of at
      least ``exhaustive_min_island`` keyframes (capped at
      ``exhaustive_max_pairs``; above the cap it falls back to retrieval);
    * for keyframes in smaller islands: their ``retrieval_top_k_small`` most
      similar keyframes in OTHER islands (the base tier's DINOv2 descriptors).

    Every candidate goes through the same `verify_pair` (RootSIFT, mutual
    ratio test, essential RANSAC). A kept pair is `strict` when it passes the
    base tier's own floors -- the same bar as a base pair -- and relaxed-only
    when it passes just ``relaxed_min_inliers`` with an inlier ratio of at
    least ``relaxed_min_inlier_ratio``. Placement metrics use strict pairs;
    relaxed ones are reported separately. The base tier (`pairs.npz`) is not
    touched, so every number computed on it stays comparable.
    """
    import cv2

    params = dict(PAIR_PARAMS)
    xp = dict(XISLAND_PARAMS)
    root = pairs_dir(cache_root, world.world_id)
    base = PairSet(root)
    if not base.available:
        raise RuntimeError("build the base pair tier first (cache --what pairs)")
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()
    lab = islands(world.n, base.arrays["i"], base.arrays["j"])
    sizes = np.bincount(lab)
    desc = np.load(root / "descriptors.npy")
    sim = desc @ desc.T
    big = [int(c) for c in np.nonzero(sizes >= xp["exhaustive_min_island"])[0]]
    members = {c: np.nonzero(lab == c)[0] for c in big}
    n_exh = sum(len(members[a]) * len(members[b]) for k, a in enumerate(big) for b in big[k + 1:])
    cand: dict[tuple[int, int], tuple[int, float]] = {}
    if n_exh <= xp["exhaustive_max_pairs"]:
        for k, a in enumerate(big):
            for b in big[k + 1:]:
                for i in members[a]:
                    for j in members[b]:
                        x, y = (int(i), int(j)) if i < j else (int(j), int(i))
                        cand[(x, y)] = (SOURCE_XISLAND_EXHAUSTIVE, float(sim[x, y]))
        small = np.nonzero(sizes[lab] < xp["exhaustive_min_island"])[0]
        xp["mode"] = "exhaustive between islands + retrieval for small islands"
    else:
        small = np.arange(world.n)
        xp["mode"] = f"retrieval only ({n_exh} exhaustive pairs exceed the cap)"
    idx = np.arange(world.n)
    for i in small:
        srow = sim[i].copy()
        srow[lab == lab[i]] = -np.inf
        for j in np.lexsort((idx, -srow))[: xp["retrieval_top_k_small"]]:
            if not np.isfinite(srow[j]):
                continue
            x, y = (int(i), int(j)) if i < j else (int(j), int(i))
            cand.setdefault((x, y), (SOURCE_XISLAND_RETRIEVAL, float(sim[x, y])))
    cands = [(a, b, src, sv) for (a, b), (src, sv) in sorted(cand.items())]
    log(f"[xisland] {world.world_id[:8]}: {len(sizes)} islands ({len(big)} with >= "
        f"{xp['exhaustive_min_island']} kf), {len(cands)} cross-island candidates ({xp['mode']})")
    images = []
    for i in range(world.n):
        img, kind = world.canonical_image(i, gray=True)
        if img is None:
            raise RuntimeError(f"keyframe {i}: no canonical image ({kind})")
        images.append(img)
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        feats = list(pool.map(lambda im: _features(im, params["sift_features"]), images))
    del images
    relaxed = dict(params, min_inliers_local=xp["relaxed_min_inliers"],
                   min_inliers_distant=xp["relaxed_min_inliers"],
                   min_inlier_ratio=min(params["min_inlier_ratio"], xp["relaxed_min_inlier_ratio"]))
    K = world.canonical_K()
    size = (int(world.canonical_camera["width"]), int(world.canonical_camera["height"]))

    def work(c):
        i, j = c[0], c[1]
        return verify_pair(feats[i][0], feats[i][1], feats[j][0], feats[j][1], K, size, relaxed, distant=False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, cands))
    cv2.setNumThreads(-1)
    keep, strict_flags = [], []
    for c, r in zip(cands, results):
        if r is None:
            continue
        ratio = r["n_inliers"] / max(r["n_matches"], 1)
        floor = params["min_inliers_distant"] if (c[1] - c[0]) > params["distant_gap"] else params["min_inliers_local"]
        strict = r["n_inliers"] >= floor and ratio >= params["min_inlier_ratio"]
        if not strict and ratio < xp["relaxed_min_inlier_ratio"]:
            continue
        keep.append((c, r))
        strict_flags.append(bool(strict))
    arrays = _pack(keep)
    arrays["strict"] = np.array(strict_flags, bool)
    arrays["island_i"] = lab[arrays["i"]].astype(np.int32)
    arrays["island_j"] = lab[arrays["j"]].astype(np.int32)
    np.savez_compressed(root / XISLAND_ARRAYS, **arrays)
    link: dict[str, dict] = {}
    for (c, _), st in zip(keep, strict_flags):
        a, b = sorted((int(lab[c[0]]), int(lab[c[1]])))
        row = link.setdefault(f"{a}-{b}", {"strict": 0, "relaxed_only": 0})
        row["strict" if st else "relaxed_only"] += 1
    manifest = {"world_id": world.world_id, "session_id": world.session_id, "params": xp,
                "base_params": params, "base_digest": base.digest(),
                "islands": {"count": int(len(sizes)), "sizes": [int(x) for x in sizes],
                            "labels": [int(x) for x in lab]},
                "counts": {"candidates": len(cands),
                           "candidates_exhaustive": sum(1 for c in cands if c[2] == SOURCE_XISLAND_EXHAUSTIVE),
                           "candidates_retrieved": sum(1 for c in cands if c[2] == SOURCE_XISLAND_RETRIEVAL),
                           "verified_strict": int(sum(strict_flags)),
                           "verified_relaxed_only": int(len(keep) - sum(strict_flags))},
                "links_by_island_pair": dict(sorted(link.items())),
                "complete": True, "seconds": round(time.time() - t0, 2)}
    (root / XISLAND_MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    log(f"[xisland] {world.world_id[:8]}: {manifest['counts']} in {time.time() - t0:.0f}s")
    return manifest


# Versions 1-2 of the learned tier (EfficientLoFTR through transformers 5.16)
# wrote these files. That port returns its matches on the 8-px coarse grid,
# so they are left on disk and never read (`PairSet` loads LEARNED_*).
LOFTR_ARRAYS = "xisland_loftr.npz"
LOFTR_MANIFEST = "xisland_loftr_manifest.json"

LEARNED_ARRAYS = "xisland_learned.npz"
LEARNED_MANIFEST = "xisland_learned_manifest.json"
LEARNED_PARAMS = {
    "version": 3,
    "matcher": "aliked-n16+lightglue (learned_match.AlikedLightGlue; Apache-2.0 / BSD-3)",
    "matcher_params": None,  # filled from learned_match.ALIKED_LG_PARAMS at build time
    "input_size": "native (the canonical undistorted keyframe, no resize)",
    "min_island": 10,
    "top_k_per_island_pair": 200,
    "keypoints": "float, sub-pixel (validated: P2-LM subpixel_check, median 0.04-0.23 px on synthetic shifts)",
    "verification": "verify_points with the base tier's own floors (strict) -- same RANSAC, same thresholds",
}


def eloftr_matches_float(outputs, target_sizes, threshold: float = 0.0) -> list[dict]:
    """transformers' `post_process_keypoint_matching` for EfficientLoFTR,
    without its `keypoints.to(torch.int32)`.

    The library truncates every keypoint to a whole pixel -- up to 1 px of
    error, which is the whole RANSAC threshold `verify_points` applies -- so
    version 1 of this tier verified learned matches under a bias SIFT's
    sub-pixel keypoints never carry. Same scaling, same score/match filter,
    float coordinates. `target_sizes` is a list of ((h0, w0), (h1, w1)).
    """
    import torch

    sizes = torch.tensor(target_sizes, device=outputs.matches.device, dtype=torch.float32)
    keypoints = outputs.keypoints.float() * sizes.flip(-1).reshape(-1, 2, 1, 2)
    results = []
    for kp, matches, scores in zip(keypoints, outputs.matches, outputs.matching_scores):
        valid = torch.logical_and(scores > threshold, matches > -1)
        results.append({"keypoints0": kp[0][valid[0]], "keypoints1": kp[1][valid[1]],
                        "matching_scores": scores[0][valid[0]]})
    return results


def build_learned_cross_island_tier(world: WorldInfo, cache_root, *, device: str | None = None, log=print,
                                    top_k: int | None = None, matcher=None) -> dict:
    """Optional, flagged tier: a learned matcher (ALIKED+LightGlue,
    `learned_match.AlikedLightGlue`) on the most similar cross-island candidates.

    SIFT finds no strict cross-island pair on the target even exhaustively;
    this asks whether a stronger matcher does. Candidates: for every pair of
    base-tier islands with at least ``min_island`` keyframes, the
    ``top_k_per_island_pair`` most DINOv2-similar keyframe pairs across them.
    Matches are verified by `verify_points` with the base tier's floors, so a
    pair kept here meets the same geometric bar as a base pair; only the
    correspondence source differs. Needs ``lightglue`` on sys.path and its
    weights in the torch hub cache (TORCH_HOME). ``matcher``: anything with
    ``match(rgb_a, rgb_b, key_a, key_b) -> (kp_a, kp_b, score)`` (tests).
    """
    import cv2

    params = dict(PAIR_PARAMS)
    lp = dict(LEARNED_PARAMS)
    if top_k is not None:
        lp["top_k_per_island_pair"] = int(top_k)
    root = pairs_dir(cache_root, world.world_id)
    base = PairSet(root)
    if not base.available:
        raise RuntimeError("build the base pair tier first (cache --what pairs)")
    t0 = time.time()
    lab = islands(world.n, base.arrays["i"], base.arrays["j"])
    sizes = np.bincount(lab)
    desc = np.load(root / "descriptors.npy")
    sim = desc @ desc.T
    big = [int(c) for c in np.nonzero(sizes >= lp["min_island"])[0]]
    cands = []
    for k, a in enumerate(big):
        for b in big[k + 1:]:
            A = np.nonzero(lab == a)[0]
            B = np.nonzero(lab == b)[0]
            S = sim[np.ix_(A, B)]
            flat = np.lexsort((np.arange(S.size), -S.ravel()))[: lp["top_k_per_island_pair"]]
            for f in flat:
                i, j = int(A[f // len(B)]), int(B[f % len(B)])
                x, y = (i, j) if i < j else (j, i)
                cands.append((x, y, 4, float(sim[x, y])))
    cands = sorted(set(cands))
    torch = None
    if matcher is None:
        import torch

        from tower.world_builder.coherence_eval.learned_match import ALIKED_LG_PARAMS, AlikedLightGlue

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        matcher = AlikedLightGlue(device=device)
        lp["matcher_params"] = dict(ALIKED_LG_PARAMS)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
    log(f"[learned] {world.world_id[:8]}: {len(big)} islands >= {lp['min_island']} kf, {len(cands)} candidates")
    K = world.canonical_K()
    size = (int(world.canonical_camera["width"]), int(world.canonical_camera["height"]))
    cache: dict[int, np.ndarray] = {}

    def rgb(i):
        if i not in cache:
            img, kind = world.canonical_image(i)
            if img is None:
                raise RuntimeError(f"keyframe {i}: no canonical image ({kind})")
            cache[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if len(cache) > 256:
                cache.pop(next(iter(cache)))
        return cache[i]

    keep = []
    for n_done, (i, j, src, sv) in enumerate(cands):
        p0, p1, _ = matcher.match(rgb(i), rgb(j), i, j)
        r = verify_points(np.asarray(p0, np.float64), np.asarray(p1, np.float64), K, size, params,
                          distant=(j - i) > params["distant_gap"])
        if r is not None:
            keep.append(((i, j, src, sv), r))
        if (n_done + 1) % 200 == 0:
            log(f"[learned] {n_done + 1}/{len(cands)} ({len(keep)} verified) {time.time() - t0:.0f}s")
    arrays = _pack(keep)
    arrays["strict"] = np.ones(len(keep), bool)
    arrays["island_i"] = lab[arrays["i"]].astype(np.int32)
    arrays["island_j"] = lab[arrays["j"]].astype(np.int32)
    np.savez_compressed(root / LEARNED_ARRAYS, **arrays)
    link: dict[str, int] = {}
    for (c, _) in keep:
        x, y = sorted((int(lab[c[0]]), int(lab[c[1]])))
        link[f"{x}-{y}"] = link.get(f"{x}-{y}", 0) + 1
    cuda = torch is not None and device == "cuda"
    manifest = {"world_id": world.world_id, "session_id": world.session_id, "params": lp,
                "base_params": params, "base_digest": base.digest(),
                "counts": {"candidates": len(cands), "verified_strict": len(keep)},
                "links_by_island_pair": dict(sorted(link.items())),
                "complete": True, "seconds": round(time.time() - t0, 2),
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None}
    (root / LEARNED_MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    log(f"[learned] {world.world_id[:8]}: {manifest['counts']} links {manifest['links_by_island_pair']} "
        f"in {time.time() - t0:.0f}s")
    return manifest
