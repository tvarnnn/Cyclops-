"""The anchor verification: re-check the PUBLISHED ROOM of the chosen draw -- its anchor block included -- after the
gate and the consensus, and seal what the evidence says the solve misplaced (`TOWER_WORLD_ANCHOR_VERIFY`; OFF by
default, and off is today's gate byte for byte).

WHY. The gate never re-verifies the block it starts from. On 6839fb8f the anchor absorbed a bed stretch that
independent image evidence puts 15-32 deg off (manager 071); on walk 3 the bathroom sits INSIDE the anchor at x2.9
the room's scale (manager 081 section 1): `scale_split`'s binary segmentation cannot see a middle stretch shorter than
both of its flanks, and the attach-time level test is never applied to the reference.

THE RULE is RUN `experiments/P4-IV/RULE.md` (declared 2026-09-25; `VerifyParams` digest 8efd726d8a1548d8), with
part (a) replaced by `RULE-a2.md` (digest 8f5d494796345cd7), ported from their reference implementation
`scripts/iv_rule.py` and the harness's matcher (`coherence_eval/eval_pairs.py`,
PORTED, never imported). On the published room of the chosen draw (its kept set: the anchor group plus every
attached group, the gate's own groups with label 0):

  (c) `motion`: consecutive kept cameras at most 2 s apart that moved more than 2.0 m/s * dt + 0.3 m or turned more
      than 300 deg/s * dt + 30 deg are FLAGGED (human walking and head-turn bounds, the harness's metrics.PARAMS; the
      metres are the gate's own metric level). A flag only adds a cut point to (a) and (b); it never places.
  (a) `scale`: the anchor's cameras are cut into capture runs; each run is split by the product's own
      `coherence_gate.scale_split`; each segment with >= 10 ratios is compared with the REST of the anchor by
      RULE-a2.md's test (digest 8f5d494796345cd7; it replaces RULE.md's `scale_levels_differ`, which failed the
      control): the sign-test confidence intervals of the two median levels (alpha 0.05 each) must be separated by
      more than log 1.25. The segment with the largest gap is SEALED as `scale-mismatch`, repeatedly. GUARD: if
      what would be sealed holds half or more of the anchor's cameras with a ratio, nothing is (no majority level).
  (b) `images`: the kept set (less (a)'s seals) is cut into verification groups (capture runs, windows of 30, at
      least 5), and each group is judged against the rest of the kept set on MASKED image-only two-view evidence:
      the harness's base tier with every keypoint under the solve's transient mask dropped before matching. Only
      REVISIT pairs count. Direct pairs decide at >= 3 pairs over >= 3 group and 3 partner cameras (rotation
      median > 4.66 deg, or translation-direction median > 23.22 deg, contradicts); else two-hop chains through one
      revisit leg (median > 23.04 deg contradicts); else the group is unverifiable and left as today. The worst
      contradicted group is sealed (`IMAGE_SEAL_REASON`: `link-contradicted`, manager 086), and the rest are
      judged again (peeling).
  THE SEAL RE-GATE: `coherence_gate.apply_gate` again on the same candidate, links, levels and depth, with the
      consensus's `withhold` unchanged, the new `seal` hook and the room's cameras as the allow-list; published
      only if its post-checks hold (`seal_refusal`: the room shrinks to a subset of the chosen room minus the sealed
      cameras, every sealed camera has its one reason, every piece outside the chosen room is the chosen draw's).
      Collateral that changed the kept set runs (a) and (b) again, at most 3 rounds.
  (i) `imports`: in the gate itself (every draw): the pairs one relocalizer import created are ONE evidence unit
      in `coherence_gate.redundant_links` (`import_link_units`).

Every bound is RULE.md's (thresholds from the control b2a75ab4 only; see there). The record is `gate.anchor_verify`
(Tower-internal, additive, only when on). CPU only; the DINOv2-small retrieval descriptor runs on the CPU from the
local Hugging Face cache and is NEVER substituted: if it cannot load, the verification fails and says so.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from tower.world_builder import coherence_gate as CG

logger = logging.getLogger(__name__)

ANCHOR_VERIFY_ID = "anchor-verify:p4-iv-rule/1"
PART_SCALE = "scale"
PART_IMAGES = "images"
PART_MOTION = "motion"
PART_IMPORTS = "imports"

# THE ONE LINE: the reason a group the images contradict is sealed with. Manager 086 (walk 4): the contract's existing
# `link-contradicted` -- a new closed-set reason (RULE.md section 6's `image-contradicted`) would need the contract
# and a Mac G4 first, so it is never written. Which part sealed a piece is recorded Tower-side only
# (`gate.anchor_verify.sealed_groups[*].source`), never in the components record.
IMAGE_SEAL_REASON = CG.REASON_LINK_CONTRADICTED
SCALE_SEAL_REASON = CG.REASON_SCALE_MISMATCH
SOURCE_IMAGES = "anchor-image-verification"
SOURCE_SCALE = "anchor-scale-segmentation"

VERDICT_CONTRADICTED = "contradicted"
VERDICT_CONFIRMED = "confirmed"
VERDICT_UNVERIFIABLE = "unverifiable"

STATE_APPLIED = "applied"            # it ran; `sealed` may be empty
STATE_NOT_APPLIED = "not-applied"    # the seal re-gate failed its checks; the room is published as gated
STATE_NOT_RUN = "not-run"            # nothing to verify (`why`)
STATE_FAILED = "failed"              # an error (`detail`); the room is published as gated

MAX_ROUNDS = 3


# PART (a) IS RULE-a2.md (P4-IV, declared 02:12, digest 8f5d494796345cd7), which replaces RULE.md section 3.2's level
# test: the declared `scale_levels_differ` read per-camera level NOISE as a x3.73 step on the control (P4-IV
# PHASE2.md). A segment differs from the rest of the anchor only when the distribution-free (sign-test) confidence
# intervals of the two MEDIAN levels, coverage >= 1 - alpha each, are separated by more than log(scale_step_factor).
# alpha 0.05 is RULE-a2.md section 3's choice, on the control's 7 runs alone (0 keyframes sealed on each).
A2_ALPHA = 0.05
A2_SPEC = {"test": "median-ci-sign", "alpha": A2_ALPHA, "gap_bound": "log(scale_step_factor)=log(1.25)",
           "rest_interval": True, "segments": "capture runs (cuts i-v) x product scale_split, unchanged",
           "min_ratios": "scale_min_cameras=10, unchanged", "peeling": "largest CI gap first",
           "guard": "refuse when sealed ratios >= half the anchor ratios"}


def a2_digest(vp: "VerifyParams | None" = None) -> str:
    """RULE-a2.md section 5: sha1[:16] of the sorted JSON {"verify": VerifyParams, "a2": A2_SPEC}."""
    doc = {"verify": (vp or VerifyParams()).to_json(), "a2": A2_SPEC}
    return hashlib.sha1(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]



@dataclass(frozen=True)
class VerifyParams:
    """RULE.md section 4.6, digest 8efd726d8a1548d8 (sha1[:16] of the sorted JSON of these fields)."""

    # --- the verification groups ---
    window: int = 30              # longest run, in kept cameras (the harness's revisit_window)
    min_group: int = 5            # smallest group; smaller runs merge into the nearest same-gate-group group
    span_join_s: float = 2.0      # a capture pause longer than this cuts (coherence_publish.SPAN_JOIN_S)
    seq_gap: int = 2              # the image-only sequential chain = base pairs with capture gap <= 2
    # --- which image pairs are evidence (the harness's own "revisit" bucket, metrics.pair_buckets) ---
    revisit_gap: int = 30
    revisit_min_dt_s: float = 10.0
    revisit_other_segment: bool = True
    t_min_parallax_deg: float = 5.0
    # --- sufficiency ---
    min_pairs: int = 3
    min_distinct: int = 3
    max_middles: int = 10
    # --- thresholds (the control b2a75ab4 only; RULE.md section 4.5, calibrate_control.py) ---
    theta_rot_deg: float = 4.66
    theta_t_deg: float = 23.22
    theta_2hop_deg: float = 23.04

    def to_json(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha1(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class MotionParams:
    """RULE.md section 5: human bounds (the harness's metrics.PARAMS), never fitted to a walk."""

    v_max_mps: float = 2.0
    margin_m: float = 0.3
    omega_max_dps: float = 300.0
    margin_deg: float = 30.0
    dt_min_s: float = 0.05
    dt_max_s: float = 2.0

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------------------------------------
# part (b)'s evidence: the harness base tier (eval_pairs.PAIR_PARAMS, unchanged), MASKED. A PORT, never an import.

PAIR_PARAMS = {
    "version": 1,
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
MASK_RULE = "drop keypoints whose mask pixel (floor x, floor y, clipped) is 0; no mask file = no keypoint"
DESCRIPTOR_ID = "facebook/dinov2-small cls+meanpatch 224x392 cpu local-files-only"
PAIRS_DIRNAME = "verify_pairs"
SOURCE_ADJACENT = 0
SOURCE_RETRIEVAL = 1


class DescriptorUnavailable(RuntimeError):
    """DINOv2-small could not be loaded from the local cache: no pair set is built (never a substitute)."""


def global_descriptors(images: list) -> np.ndarray:
    """eval_pairs._global_descriptors, without its fallback: DINOv2-small on the CPU, from the local Hugging Face
    cache only. `images`: BGR uint8 arrays, or None (a missing image: a zero descriptor)."""
    try:
        import cv2  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from transformers import AutoModel  # noqa: PLC0415

        torch.manual_seed(0)
        model = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).eval()
    except Exception as exc:  # noqa: BLE001 -- a different retrieval is a different pair set: refused
        raise DescriptorUnavailable(f"DINOv2-small unavailable ({type(exc).__name__}: {exc})") from exc
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    present = [k for k, im in enumerate(images) if im is not None]
    out = np.zeros((len(images), 2 * int(model.config.hidden_size)), np.float32)
    with torch.no_grad():
        for s in range(0, len(present), 16):
            batch = []
            for k in present[s:s + 16]:
                im = images[k]
                rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else im[:, :, ::-1]
                rgb = cv2.resize(rgb, (224, 392), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                batch.append(((rgb - mean) / std).transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(batch))
            h = model(pixel_values=x).last_hidden_state
            cls_ = torch.nn.functional.normalize(h[:, 0], dim=-1)
            pm = torch.nn.functional.normalize(h[:, 1:].mean(1), dim=-1)
            d = torch.nn.functional.normalize(torch.cat([cls_, pm], dim=-1), dim=-1)
            out[present[s:s + 16]] = d.numpy().astype(np.float32)
    return out


def _features(image: np.ndarray, n: int):
    import cv2  # noqa: PLC0415

    sift = cv2.SIFT_create(nfeatures=n)
    kps, desc = sift.detectAndCompute(image, None)
    if desc is None or not len(kps):
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    xy = np.array([k.pt for k in kps], np.float32)
    desc = desc.astype(np.float32)
    desc /= np.abs(desc).sum(1, keepdims=True) + 1e-9
    desc = np.sqrt(desc)
    return xy, desc


def masked_features(image, mask, n: int):
    """RootSIFT keypoints of one image with every keypoint under the solve's transient mask dropped BEFORE matching
    (the product's own test: floor x, floor y, clipped, as in `global_solve._mask_imported_pairs`). No mask: no
    keypoint (the product's "excluded" rule). Returns (xy, desc, all, kept)."""
    if image is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), 0, 0
    xy, d = _features(image, n)
    if mask is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), len(xy), 0
    x = np.clip(np.floor(xy[:, 0]).astype(np.int64), 0, mask.shape[1] - 1)
    y = np.clip(np.floor(xy[:, 1]).astype(np.int64), 0, mask.shape[0] - 1)
    keep = mask[y, x] != 0
    return xy[keep], d[keep], len(xy), int(keep.sum())


def _match(d1: np.ndarray, d2: np.ndarray, ratio: float) -> np.ndarray:
    import cv2  # noqa: PLC0415

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
    H = f1.T @ f2
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def verify_pair(xy1, d1, xy2, d2, K, size, params, distant: bool):
    """eval_pairs.verify_pair: mutual ratio matching, then `verify_points`."""
    m = _match(d1, d2, params["ratio"])
    floor = params["min_inliers_distant"] if distant else params["min_inliers_local"]
    if len(m) < floor:
        return None
    return verify_points(xy1[m[:, 0]], xy2[m[:, 1]], K, size, params, distant)


def verify_points(p1, p2, K, size, params, distant: bool):
    """eval_pairs.verify_points: essential-matrix RANSAC, the floors, R and t (x_j = R x_i + t)."""
    import cv2  # noqa: PLC0415

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
    E = E[:3]
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
    out = cv2.recoverPose(E, p1[inl], p2[inl], cameraMatrix=K, distanceThresh=1e4)
    n_cheir, R_E, t_E = int(out[0]), out[1], out[2]
    R_rot = wahba_rotation(b1, b2)

    def parallax_of(R):
        return float(np.median(np.degrees(np.arccos(np.clip(((b1 @ R.T) * b2).sum(1), -1, 1)))))

    par_E = parallax_of(R_E)
    if par_E >= params["min_parallax_deg"] and n_cheir >= 0.5 * n_inl:
        R, t, reliable, par = R_E, t_E.ravel(), True, par_E
    else:
        R, t, reliable, par = R_rot, t_E.ravel(), False, parallax_of(R_rot)
    return {"R": R, "t": t / (np.linalg.norm(t) + 1e-12), "n_matches": n_matches, "n_inliers": n_inl,
            "n_cheirality": int(n_cheir), "parallax_deg": par, "t_reliable": bool(reliable),
            "xy1": n1[: params["stored_inliers"]].astype(np.float32),
            "xy2": n2[: params["stored_inliers"]].astype(np.float32)}


def candidate_pairs(n: int, desc: np.ndarray, params: dict) -> list:
    """eval_pairs.candidate_pairs: adjacent gaps, then retrieval top-k beyond the minimum gap; (i, j, source, sim)."""
    sim = desc @ desc.T
    cand: dict = {}
    for gap in params["adjacent_gaps"]:
        for i in range(n - gap):
            cand[(i, i + gap)] = (SOURCE_ADJACENT, float(sim[i, i + gap]))
    idx = np.arange(n)
    for i in range(n):
        s = sim[i].copy()
        s[np.abs(idx - i) <= params["retrieval_min_gap"]] = -np.inf
        order = np.lexsort((idx, -s))[: params["retrieval_top_k"]]
        for j in order:
            if not np.isfinite(s[j]):
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            cand.setdefault((a, b), (SOURCE_RETRIEVAL, float(sim[a, b])))
    return [(a, b, src, s) for (a, b), (src, s) in sorted(cand.items())]


def pack(keep) -> dict:
    """eval_pairs._pack."""
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


def pair_digest(arrays) -> str | None:
    """eval_pairs._digest (the harness PairSet.digest)."""
    if arrays is None:
        return None
    h = hashlib.sha1()
    for k in ("i", "j", "R", "t", "t_reliable"):
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()[:16]


def _sha1(path: Path) -> str | None:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except OSError:
        return None


def build_masked_pairs(workspace_root, names: list[str], camera: dict, *, params: dict | None = None,
                       workers: int | None = None, descriptor_fn: Callable | None = None, images_dir=None,
                       masks_dir=None, cache_root=None) -> tuple[dict, dict]:
    """The MASKED base tier of the session's keyframes (`names`, capture order = index), content-addressed and
    cached under `solve/<session>/verify_pairs/<key>/` (RULE.md section 9.3): key = sha1 of the pair parameters,
    the mask rule, the descriptor, the camera, and every image's and mask's sha1. Returns (arrays, record).

    Images are the solve's canonical undistorted frames (`solve/<session>/images/<name>`, the harness's
    `solve-images` source); masks are the solve's transient masks (`solve/<session>/masks/<name>.png`, the 12 px
    dilation already in them). A missing or wrongly sized image contributes no keypoint and a zero descriptor.
    `images_dir`, `masks_dir` and `cache_root` replace those three places (a measurement on a read-only copy); the
    key names content, never paths."""
    import cv2  # noqa: PLC0415

    from tower.storage import write_bytes_atomic, write_json_atomic  # noqa: PLC0415

    params = dict(params or PAIR_PARAMS)
    t0 = time.perf_counter()
    root = Path(workspace_root) if workspace_root is not None else None
    img_dir = Path(images_dir) if images_dir is not None else root / "images"
    msk_dir = Path(masks_dir) if masks_dir is not None else root / "masks"
    img_paths = [img_dir / nm for nm in names]
    mask_paths = [msk_dir / f"{nm}.png" for nm in names]
    img_sha = [_sha1(p) for p in img_paths]
    mask_sha = [_sha1(p) for p in mask_paths]
    cam = {k: camera[k] for k in ("fx", "fy", "cx", "cy", "width", "height")}
    key_doc = {"pair_params": params, "mask_rule": MASK_RULE, "descriptor": DESCRIPTOR_ID, "camera": cam,
               "images": img_sha, "masks": mask_sha}
    key = hashlib.sha1(json.dumps(key_doc, sort_keys=True).encode()).hexdigest()[:16]
    cache = (Path(cache_root) if cache_root is not None else root / PAIRS_DIRNAME) / key
    manifest_path, arrays_path = cache / "manifest.json", cache / "pairs.npz"
    if manifest_path.is_file() and arrays_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with np.load(arrays_path, allow_pickle=False) as f:
                arrays = {k: f[k] for k in f.files}
            if manifest.get("complete") and pair_digest(arrays) == manifest.get("digest"):
                return arrays, {"key": key, "digest": manifest["digest"], "pairs": int(len(arrays["i"])),
                                "source": "cached", "keypoints": manifest.get("keypoints"),
                                "images_missing": manifest.get("images_missing"),
                                "masks_missing": manifest.get("masks_missing"),
                                "seconds": round(time.perf_counter() - t0, 3)}
        except (OSError, ValueError, KeyError):
            logger.warning("[Tower][WorldBuilder] anchor verification: the pair cache %s is unreadable; it is "
                           "rebuilt", cache, exc_info=True)
    w, h = int(cam["width"]), int(cam["height"])
    colour, gray, masks = [], [], []
    for p, mp in zip(img_paths, mask_paths):
        im = None
        if p.is_file():
            im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is not None and (im.shape[1] != w or im.shape[0] != h):
                im = None
        colour.append(im)
        gray.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im is not None else None)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.is_file() else None
        masks.append(m)
    t_load = time.perf_counter() - t0
    desc = (descriptor_fn or global_descriptors)(colour)
    del colour
    t_desc = time.perf_counter() - t0 - t_load
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            F = list(pool.map(lambda k: masked_features(gray[k], masks[k], params["sift_features"]),
                              range(len(names))))
        t_feat = time.perf_counter() - t0 - t_load - t_desc
        cands = candidate_pairs(len(names), desc, params)
        K = np.array([[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]])

        def work(c):
            i, j, _src, _s = c
            return verify_pair(F[i][0], F[i][1], F[j][0], F[j][1], K, (w, h), params,
                               distant=(j - i) > params["distant_gap"])

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(work, cands))
    finally:
        cv2.setNumThreads(-1)
    keep = [(c, r) for c, r in zip(cands, results) if r is not None]
    arrays = pack(keep)
    kp_all, kp_kept = sum(f[2] for f in F), sum(f[3] for f in F)
    digest = pair_digest(arrays)
    keypoints = {"all": int(kp_all), "kept": int(kp_kept),
                 "dropped_fraction": round(1 - kp_kept / max(kp_all, 1), 4)}
    manifest = {"key": key, "params": params, "mask_rule": MASK_RULE, "descriptor": DESCRIPTOR_ID, "camera": cam,
                "images": dict(zip(names, img_sha)), "masks": dict(zip(names, mask_sha)),
                "images_missing": sum(1 for g in gray if g is None),
                "masks_missing": sum(1 for m in masks if m is None), "keypoints": keypoints,
                "candidates": len(cands), "verified": len(keep), "digest": digest, "complete": True,
                "seconds": {"load": round(t_load, 2), "descriptors": round(t_desc, 2),
                            "features": round(t_feat, 2), "total": round(time.perf_counter() - t0, 2)}}
    try:
        cache.mkdir(parents=True, exist_ok=True)
        write_bytes_atomic(cache / "descriptors.npy", lambda handle: np.save(handle, desc))
        write_bytes_atomic(arrays_path, lambda handle: np.savez_compressed(handle, **arrays))
        write_json_atomic(manifest_path, manifest)          # last: the manifest says the set is complete
    except OSError:
        logger.warning("[Tower][WorldBuilder] anchor verification: could not cache the pair set in %s", cache,
                       exc_info=True)
    return arrays, {"key": key, "digest": digest, "pairs": len(keep), "source": "built", "keypoints": keypoints,
                    "images_missing": manifest["images_missing"], "masks_missing": manifest["masks_missing"],
                    "candidates": len(cands), "seconds": round(time.perf_counter() - t0, 3)}


# ---------------------------------------------------------------------------------------------------------------
# the rule (iv_rule.py, ported). Indices are WORLD keyframe indices: the session's keyframes in capture order.
# Pair k links i < j with x_j = R_k x_i + t_k; a solve gives R_cw and centre C per camera, so the solve's relative
# rotation is R_j R_i^T and its translation direction R_j (C_i - C_j).


def rot_deg(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    return np.degrees(np.arctan2(np.linalg.norm(v, axis=1), tr - 1.0))


def vec_deg(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), float(a @ b))))


class Pairs:
    """The masked base tier, indexed for lookups."""

    def __init__(self, A: dict, n: int, t_min_parallax_deg: float = 5.0):
        self.A = A
        self.I = np.asarray(A["i"], np.int64)
        self.J = np.asarray(A["j"], np.int64)
        self.n = n
        self.key = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(self.I, self.J))}
        self.nbrs: list[list[int]] = [[] for _ in range(n)]
        for k, (i, j) in enumerate(zip(self.I, self.J)):
            self.nbrs[int(i)].append(k)
            self.nbrs[int(j)].append(k)
        self.t_ok = np.asarray(A["t_reliable"], bool) & (np.asarray(A["parallax_deg"]) >= t_min_parallax_deg)

    def R_from_to(self, k: int, a: int) -> np.ndarray:
        return self.A["R"][k] if int(self.I[k]) == a else self.A["R"][k].T

    def other(self, k: int, a: int) -> int:
        return int(self.J[k]) if int(self.I[k]) == a else int(self.I[k])


class World:
    """Capture order, times and tracker segments of the session's keyframes (index = capture order)."""

    def __init__(self, times, segments):
        self.t = np.asarray(times, np.float64)
        self.seg = np.asarray(segments)
        self.n = len(self.t)


class Solve:
    """Per world keyframe index: R_cw (3x3) and centre."""

    def __init__(self, R_cw: dict, C: dict):
        self.R = R_cw
        self.C = C

    def R_rel(self, a: int, b: int) -> np.ndarray:
        return self.R[b] @ self.R[a].T

    def t_rel(self, i: int, j: int) -> np.ndarray:
        return self.R[j] @ (self.C[i] - self.C[j])


def is_revisit(w: World, a: int, b: int, p: VerifyParams) -> bool:
    if abs(a - b) <= p.revisit_gap:
        return False
    if p.revisit_other_segment and w.seg[a] == w.seg[b]:
        return False
    return abs(w.t[a] - w.t[b]) >= p.revisit_min_dt_s


def capture_runs(kept, gate_group_of: dict, published: set, w: World, pairs: Pairs, p: VerifyParams,
                 extra_cuts=()) -> list[list[int]]:
    """RULE.md section 2 step 1: cut between consecutive kept a < b when (i) their gate groups differ, (ii) a
    published camera outside the kept set lies between them, (iii) the capture pauses more than span_join_s,
    (iv) the image-only sequential chain breaks, or (v) (a, b) is motion-flagged (`extra_cuts`)."""
    kept = sorted(int(k) for k in kept)
    kept_set = set(kept)
    flagged = {(int(a), int(b)) for a, b in extra_cuts}
    runs: list[list[int]] = []
    cur: list[int] = []
    for a, b in zip([None] + kept[:-1], kept):
        if a is None:
            cur = [b]
            continue
        cut = gate_group_of.get(a) != gate_group_of.get(b)
        cut = cut or any(x in published and x not in kept_set for x in range(a + 1, b))
        cut = cut or (w.t[b] - w.t[a]) > p.span_join_s
        cut = cut or (a, b) in flagged
        if not cut:
            linked = False
            for x in range(max(0, a - p.seq_gap), a + 1):
                for y in range(b, min(w.n, x + p.seq_gap + 1)):
                    if x < y and (x, y) in pairs.key:
                        linked = True
                        break
                if linked:
                    break
            cut = not linked
        if cut:
            runs.append(cur)
            cur = [b]
        else:
            cur.append(b)
    if cur:
        runs.append(cur)
    return runs


def verification_groups(kept, gate_group_of: dict, published: set, w: World, pairs: Pairs, p: VerifyParams,
                        extra_cuts=()) -> list[np.ndarray]:
    """RULE.md section 2: capture runs, windows of at most `window`, runs under `min_group` merged into the nearest
    group of the same gate group (capture-index distance; ties: the earlier)."""
    runs = capture_runs(kept, gate_group_of, published, w, pairs, p, extra_cuts)
    groups: list[list[int]] = []
    for r in runs:
        n = len(r)
        if n <= p.window:
            groups.append(r)
            continue
        k = math.ceil(n / p.window)
        base, extra = divmod(n, k)
        s = 0
        for q in range(k):
            e = s + base + (1 if q < extra else 0)
            groups.append(r[s:e])
            s = e
    changed = True
    while changed:
        changed = False
        small = [g for g in groups if len(g) < p.min_group]
        for g in sorted(small, key=lambda g: (len(g), g[0])):
            gg = gate_group_of.get(g[0])
            cands = [h for h in groups if h is not g and gate_group_of.get(h[0]) == gg]
            if not cands:
                continue

            def dist(h, g=g):
                d = min(abs(x - y) for x in g for y in h)
                return (d, 0 if h[0] < g[0] else 1, h[0])

            h = min(cands, key=dist)
            h.extend(g)
            h.sort()
            groups = [x for x in groups if x is not g]
            changed = True
            break
    out = [np.asarray(sorted(g), np.int64) for g in groups]
    out.sort(key=lambda g: int(g[0]))
    return out


def direct_evidence(G: set, P: set, w: World, pairs: Pairs, s: Solve, p: VerifyParams) -> dict:
    rot, tdir, ends_r, ends_t = [], [], [], []
    for a in sorted(G):
        for k in pairs.nbrs[a]:
            r = pairs.other(k, a)
            if r not in P or not is_revisit(w, a, r, p):
                continue
            i, j = int(pairs.I[k]), int(pairs.J[k])
            rot.append(float(rot_deg(pairs.A["R"][k].T @ s.R_rel(i, j))[0]))
            ends_r.append((a, r))
            if pairs.t_ok[k]:
                tm = s.t_rel(i, j)
                nt = np.linalg.norm(tm)
                if nt > 0:
                    tdir.append(vec_deg(tm / nt, pairs.A["t"][k]))
                    ends_t.append((a, r))
    return {"rot": rot, "t": tdir, "ends_rot": ends_r, "ends_t": ends_t}


def two_hop_evidence(G: set, P: set, w: World, pairs: Pairs, s: Solve, p: VerifyParams) -> dict:
    """Per endpoint pair (a in G, r in P, a revisit relation): the median over at most `max_middles` middles m (any
    keyframe; only the image-only legs a-m and m-r, never m's pose), strongest by min(leg inliers), ties by m. One
    leg must itself be a revisit pair."""
    inl = pairs.A["n_inliers"]
    per_end, ends = [], []
    legs: set = set()
    for a in sorted(G):
        legs1 = {pairs.other(k, a): k for k in pairs.nbrs[a]}
        by_r: dict = {}
        for m, k1 in legs1.items():
            for k2 in pairs.nbrs[m]:
                r = pairs.other(k2, m)
                if r == a or r not in P or not is_revisit(w, a, r, p):
                    continue
                rev_legs = [k for k, (x, y) in ((k1, (a, m)), (k2, (m, r))) if is_revisit(w, x, y, p)]
                if not rev_legs:
                    continue
                by_r.setdefault(r, []).append((min(int(inl[k1]), int(inl[k2])), m, k1, k2, rev_legs[0]))
        for r in sorted(by_r):
            chains = sorted(by_r[r], key=lambda c: (-c[0], c[1]))[: p.max_middles]
            errs = []
            Rsol = s.R_rel(a, r)
            for _, m, k1, k2, kl in chains:
                Rc = pairs.R_from_to(k2, m) @ pairs.R_from_to(k1, a)
                errs.append(float(rot_deg(Rc.T @ Rsol)[0]))
                legs.add(int(kl))
            per_end.append(float(np.median(errs)))
            ends.append((a, r))
    return {"ends": ends, "per_end": per_end, "legs": legs}


def _distinct_ok(ends, p: VerifyParams) -> bool:
    return (len(ends) >= p.min_pairs and len({a for a, _ in ends}) >= p.min_distinct
            and len({r for _, r in ends}) >= p.min_distinct)


def judge(G: set, P: set, w: World, pairs: Pairs, s: Solve, p: VerifyParams) -> dict:
    """RULE.md section 4.3-4.4: direct pairs decide when sufficient; else two-hop chains; else unverifiable."""
    d = direct_evidence(G, P, w, pairs, s, p)
    out = {"direct_pairs": len(d["rot"]), "direct_t_pairs": len(d["t"]),
           "direct_rot_med": float(np.median(d["rot"])) if d["rot"] else None,
           "direct_t_med": float(np.median(d["t"])) if d["t"] else None}
    if _distinct_ok(d["ends_rot"], p):
        rot_bad = out["direct_rot_med"] > p.theta_rot_deg
        t_ok = _distinct_ok(d["ends_t"], p)
        t_bad = t_ok and out["direct_t_med"] > p.theta_t_deg
        margins = [out["direct_rot_med"] / p.theta_rot_deg]
        if t_ok:
            margins.append(out["direct_t_med"] / p.theta_t_deg)
        out.update(evidence="direct", margin=max(margins),
                   verdict=VERDICT_CONTRADICTED if (rot_bad or t_bad) else VERDICT_CONFIRMED,
                   why=",".join(x for x, v in (("rotation", rot_bad), ("translation", t_bad)) if v) or None)
        return out
    h = two_hop_evidence(G, P, w, pairs, s, p)
    out["two_hop_ends"] = len(h["per_end"])
    out["two_hop_med"] = float(np.median(h["per_end"])) if h["per_end"] else None
    out["two_hop_legs"] = len(h["legs"])
    if _distinct_ok(h["ends"], p) and len(h["legs"]) >= p.min_distinct:
        bad = out["two_hop_med"] > p.theta_2hop_deg
        out.update(evidence="two-hop", margin=out["two_hop_med"] / p.theta_2hop_deg,
                   verdict=VERDICT_CONTRADICTED if bad else VERDICT_CONFIRMED,
                   why="two-hop rotation" if bad else None)
        return out
    out.update(evidence=None, margin=None, verdict=VERDICT_UNVERIFIABLE, why=None)
    return out


def verify(groups: list, w: World, pairs: Pairs, s: Solve, p: VerifyParams) -> dict:
    """Peeling (RULE.md section 4.4): judge every group against the rest of the CURRENT kept set; seal the one
    contradicted group with the largest margin (ties: the earliest first camera); repeat until none is."""
    kept = set(int(x) for g in groups for x in g)
    live = list(range(len(groups)))
    sealed: list[int] = []
    rounds = 0
    final: dict = {}
    while True:
        rounds += 1
        res = {}
        for gi in live:
            G = set(int(x) for x in groups[gi])
            res[gi] = judge(G, kept - G, w, pairs, s, p)
        bad = [gi for gi in live if res[gi]["verdict"] == VERDICT_CONTRADICTED]
        if not bad:
            final.update(res)
            break
        worst = max(bad, key=lambda gi: (res[gi]["margin"], -int(groups[gi][0])))
        final[worst] = dict(res[worst], sealed_round=rounds)
        sealed.append(worst)
        live.remove(worst)
        kept -= set(int(x) for x in groups[worst])
    return {"groups": final, "sealed": sealed, "rounds": rounds}


def median_ci(values, alpha: float):
    """iv_rule.median_ci: the distribution-free (sign-test / binomial order-statistic) confidence interval of the
    median of `values` at coverage >= 1 - alpha: [x_(k), x_(n-k+1)] with the largest k such that
    P(Bin(n, 1/2) <= k-1) <= alpha/2. None when n is too small for any k >= 1 to reach the coverage."""
    x = np.sort(np.asarray(values, np.float64))
    n = len(x)
    if n == 0:
        return None
    cdf, pk, k = 0.0, 0.5 ** n, 0
    for j in range(n):          # cdf = P(Bin <= j)
        cdf += pk
        if cdf <= alpha / 2.0:
            k = j + 1
        else:
            break
        pk = pk * (n - j) / (j + 1)
    if k < 1:
        return None
    return float(x[k - 1]), float(x[n - k])


def anchor_scale_ci(anchor_runs: list, r: np.ndarray, rank: np.ndarray, gp: CG.GateParams,
                    alpha: float = A2_ALPHA) -> dict:
    """RULE-a2.md section 2 (iv_rule.anchor_scale_ci): segments = each capture run split by the product's own
    `scale_split`; a segment with >= scale_min_cameras ratios DIFFERS from the rest of the anchor only when the two
    medians' sign-test confidence intervals are separated by more than log(scale_step_factor); peeling, largest gap
    first; the no-majority GUARD."""
    thr = math.log(gp.scale_step_factor)
    segs = []
    for run in anchor_runs:
        segs.extend(np.asarray(x, np.int64) for x in CG.scale_split(np.asarray(run, np.int64), r, rank, gp))
    alive = list(range(len(segs)))
    detached, tested = [], {}
    untestable = [i for i in alive if int(np.isfinite(r[segs[i]]).sum()) < gp.scale_min_cameras]
    while True:
        rest_all = np.concatenate([segs[i] for i in alive]) if alive else np.zeros(0, np.int64)
        best, gap_best = None, -1.0
        for i in alive:
            if i in untestable:
                continue
            rest = np.setdiff1d(rest_all, segs[i])
            vs = r[segs[i]]
            vs = vs[np.isfinite(vs)]
            vr = r[rest]
            vr = vr[np.isfinite(vr)]
            if len(vr) < gp.scale_min_cameras:
                continue
            cs, cr = median_ci(vs, alpha), median_ci(vr, alpha)
            if cs is None or cr is None:
                continue
            gap = max(cs[0] - cr[1], cr[0] - cs[1])
            tested[i] = {"n": int(len(vs)), "ci": [round(cs[0], 4), round(cs[1], 4)],
                         "rest_ci": [round(cr[0], 4), round(cr[1], 4)], "gap": round(float(gap), 4),
                         "median_factor": round(float(math.exp(np.median(vs) - np.median(vr))), 4)}
            if gap > thr and gap > gap_best:
                best, gap_best = i, gap
        if best is None:
            break
        detached.append((best, gap_best))
        alive.remove(best)
    n_ratio = int(sum(np.isfinite(r[s_]).sum() for s_ in segs))
    n_det = int(sum(np.isfinite(r[segs[i]]).sum() for i, _ in detached))
    refused = bool(detached) and 2 * n_det >= n_ratio
    return {"segments": [[int(x) for x in s_] for s_ in segs],
            "detached": [] if refused else [{"segment": i, "cameras": [int(x) for x in segs[i]],
                                             "level_gap": float(g), "factor": float(tested[i]["median_factor"]),
                                             "ci_gap": float(g)} for i, g in detached],
            "untestable": list(untestable), "refused_no_majority": refused, "tested": tested,
            "would_detach": [{"segment": i, "n": int(len(segs[i])), "ci_gap": float(g)} for i, g in detached]}


def motion_flags(kept, w: World, s: Solve, metres_per_unit: float, mp: MotionParams | None = None) -> list:
    """RULE.md section 5 (iv_rule.motion_flags): consecutive kept cameras at most dt_max apart that moved more than
    v_max * dt + margin_m metres or turned more than omega_max * dt + margin_deg degrees."""
    mp = mp or MotionParams()
    kept = sorted(int(k) for k in kept)
    out = []
    for a, b in zip(kept[:-1], kept[1:]):
        dt = max(float(w.t[b] - w.t[a]), mp.dt_min_s)
        if dt > mp.dt_max_s:
            continue
        d_m = float(np.linalg.norm(s.C[b] - s.C[a])) * metres_per_unit
        ang = float(rot_deg(s.R[b] @ s.R[a].T)[0])
        if d_m > mp.v_max_mps * dt + mp.margin_m or ang > mp.omega_max_dps * dt + mp.margin_deg:
            out.append({"a": a, "b": b, "dt": dt, "metres": d_m, "deg": ang})
    return out


# ---------------------------------------------------------------------------------------------------------------
# one round on a gated result: (c) -> (a) -> (b)


@dataclass
class Inputs:
    kids: list                 # world index -> keyframe id
    names: list                # world index -> solver image name
    w_of_kid: dict
    w_of_name: dict
    world: World
    solve: Solve
    published: set             # world indices with >= min_obs observations
    r: np.ndarray              # world index -> log(z_sfm / z_metric), NaN = none
    min_obs: int


def inputs_of(result, keyframes, min_obs: int = 30) -> Inputs:
    from tower.world_builder.global_solve import keyframe_image_name  # noqa: PLC0415

    kids = [k.keyframe_id for k in keyframes]
    names = [keyframe_image_name(k) for k in keyframes]
    w_of_kid = {kid: i for i, kid in enumerate(kids)}
    w_of_name = {nm: i for i, nm in enumerate(names)}
    R, C = {}, {}
    published = set()
    for kid, pose in (result.solution.poses or {}).items():
        w = w_of_kid.get(kid)
        if w is None:
            continue
        Rw = np.asarray(pose["rotation"], np.float64).reshape(3, 3)
        R[w] = Rw
        C[w] = -Rw.T @ np.asarray(pose["translation"], np.float64).reshape(3)
        if int(pose.get("observations", 0)) >= min_obs:
            published.add(w)
    r = np.full(len(kids), np.nan)
    for nm, v in ((result.scale or {}).get("metric_log") or {}).items():
        w = w_of_name.get(nm)
        if w is not None and v is not None and np.isfinite(v):
            r[w] = float(v)
    world = World([float(k.received_at) for k in keyframes], [int(getattr(k, "segment_index", 0) or 0)
                                                              for k in keyframes])
    return Inputs(kids, names, w_of_kid, w_of_name, world, Solve(R, C), published, r, min_obs)


def _room_w(result, x: Inputs) -> list[int]:
    return sorted(x.w_of_kid[kid] for kid, p in (result.solution.poses or {}).items()
                  if kid in x.w_of_kid and int(p.get("component", 0)) == 0
                  and int(p.get("observations", 0)) >= x.min_obs)


def analyse(result, x: Inputs, pairs: Pairs | None, parts, vp: VerifyParams, mp: MotionParams,
            gp: CG.GateParams) -> tuple[dict, dict]:
    """(c), (a) and (b) on the room of one gated result. Returns ({world index: reason} to seal, the round's audit)."""
    kept_w = _room_w(result, x)
    gg, anchor_first = {}, None
    for g in (result.gated or {}).get("groups") or []:
        if g.get("label") != 0 or g.get("sealed"):
            continue
        for nm in g["members"]:
            if nm in x.w_of_name:
                gg[x.w_of_name[nm]] = g["first_camera"]
        if g.get("reference"):
            anchor_first = g["first_camera"]
    rd: dict = {"kept": len(kept_w)}
    anchor_w = [w for w in kept_w if gg.get(w) == anchor_first]
    rd["anchor"] = len(anchor_w)
    flags: list = []
    if PART_MOTION in parts:
        lv = x.r[anchor_w]
        lv = lv[np.isfinite(lv)]
        if len(lv):
            mpu = float(math.exp(-float(np.median(lv))))
            flags = motion_flags(kept_w, x.world, x.solve, mpu, mp)
            rd["metres_per_unit"] = round(mpu, 6)
        else:
            rd["metres_per_unit"] = None
        rd["motion_flags"] = [{"a": x.kids[f["a"]], "b": x.kids[f["b"]], "dt": round(f["dt"], 3),
                               "metres": round(f["metres"], 3), "deg": round(f["deg"], 2)} for f in flags]
    cuts = [(f["a"], f["b"]) for f in flags]
    seal: dict = {}
    sealed_groups: list = []
    if PART_SCALE in parts:
        runs = capture_runs(anchor_w, gg, x.published, x.world, pairs, vp, extra_cuts=cuts)
        a = anchor_scale_ci(runs, x.r, np.arange(x.world.n), gp, A2_ALPHA)
        rd["anchor_scale"] = {
            "test": A2_SPEC["test"], "alpha": A2_ALPHA, "runs": len(runs), "segments": len(a["segments"]),
            "untestable": len(a["untestable"]),
            "refused": "no majority level" if a["refused_no_majority"] else None,
            "tested": [dict(t, first_keyframe=x.kids[a["segments"][i][0]], keyframes=len(a["segments"][i]))
                       for i, t in sorted(a["tested"].items())],
            "would_seal": [{"first_keyframe": x.kids[a["segments"][d["segment"]][0]], "keyframes": d["n"],
                            "ci_gap": round(d["ci_gap"], 4)} for d in a["would_detach"]],
            "sealed": [{"first_keyframe": x.kids[d["cameras"][0]], "keyframes": len(d["cameras"]),
                        "factor": round(d["factor"], 4), "keyframe_ids": [x.kids[c] for c in d["cameras"]]}
                       for d in a["detached"]]}
        for d in a["detached"]:
            for c in d["cameras"]:
                seal[int(c)] = SCALE_SEAL_REASON
            t = a["tested"][d["segment"]]
            sealed_groups.append({"source": SOURCE_SCALE, "reason": SCALE_SEAL_REASON,
                                  "first_keyframe": x.kids[d["cameras"][0]], "keyframes": len(d["cameras"]),
                                  "factor": round(d["factor"], 4), "ci_gap": round(d["ci_gap"], 4),
                                  "ci": t["ci"], "rest_ci": t["rest_ci"], "n": t["n"],
                                  "keyframe_ids": [x.kids[c] for c in d["cameras"]]})
    if PART_IMAGES in parts:
        kept_b = [w for w in kept_w if w not in seal]
        groups = verification_groups(kept_b, gg, x.published, x.world, pairs, vp, extra_cuts=cuts)
        vres = verify(groups, x.world, pairs, x.solve, vp)
        rows = []
        for gi, g in enumerate(groups):
            v = vres["groups"][gi]
            rows.append({"first_keyframe": x.kids[int(g[0])], "keyframes": int(len(g)), "evidence": v.get("evidence"),
                         "verdict": v["verdict"], "why": v.get("why"),
                         "n": v.get("direct_pairs") if v.get("evidence") == "direct" else v.get("two_hop_ends"),
                         "statistic": _r(v.get("direct_rot_med") if v.get("evidence") == "direct"
                                         else v.get("two_hop_med")),
                         "theta": (vp.theta_rot_deg if v.get("evidence") == "direct"
                                   else vp.theta_2hop_deg if v.get("evidence") == "two-hop" else None),
                         "direct_pairs": v.get("direct_pairs"), "direct_rot_med": _r(v.get("direct_rot_med")),
                         "direct_t_pairs": v.get("direct_t_pairs"), "direct_t_med": _r(v.get("direct_t_med")),
                         "two_hop_ends": v.get("two_hop_ends"), "two_hop_legs": v.get("two_hop_legs"),
                         "two_hop_med": _r(v.get("two_hop_med")), "margin": _r(v.get("margin"), 4),
                         "sealed_round": v.get("sealed_round"), "keyframe_ids": [x.kids[int(c)] for c in g]})
            if v["verdict"] == VERDICT_CONTRADICTED:
                for c in g:
                    seal[int(c)] = IMAGE_SEAL_REASON
                row = rows[-1]
                sealed_groups.append({"source": SOURCE_IMAGES, "reason": IMAGE_SEAL_REASON,
                                      **{k: row[k] for k in ("first_keyframe", "keyframes", "evidence", "why",
                                                             "statistic", "theta", "margin", "direct_pairs",
                                                             "direct_rot_med", "direct_t_pairs", "direct_t_med",
                                                             "two_hop_ends", "two_hop_legs", "two_hop_med",
                                                             "sealed_round", "keyframe_ids")}})
        rd["image_verification"] = {"groups": rows, "peeling_rounds": vres["rounds"], **_counts(rows)}
    rd["sealed_groups"] = sealed_groups
    return seal, rd


def _r(v, nd=3):
    return None if v is None else round(float(v), nd)


def _counts(rows) -> dict:
    out = {}
    for verdict in (VERDICT_CONFIRMED, VERDICT_UNVERIFIABLE, VERDICT_CONTRADICTED):
        sel = [r for r in rows if r["verdict"] == verdict]
        out[verdict] = {"groups": len(sel), "keyframes": sum(r["keyframes"] for r in sel),
                        "keyframe_ids": [k for r in sel for k in r["keyframe_ids"]]}
    return out


# ---------------------------------------------------------------------------------------------------------------
# the seal re-gate's post-checks (RULE.md section 1 step 3.5; mirrors coherence_publish.withhold_refusal and
# keep_outside_pieces)


def _room_kids(solution, min_obs: int) -> set:
    return {kid for kid, p in (getattr(solution, "poses", None) or {}).items()
            if int(p.get("component", 0)) == 0 and int(p.get("observations", 0)) >= min_obs}


def _published_kids(solution, min_obs: int) -> set:
    return {kid for kid, p in (getattr(solution, "poses", None) or {}).items()
            if int(p.get("observations", 0)) >= min_obs}


def seal_refusal(chosen, regated, sealed: dict, *, min_obs: int) -> str | None:
    """Why the seal re-gate may NOT be published, or None when it may. `sealed`: {keyframe id: reason}. Per
    published keyframe: the new room is a subset of the chosen room minus the sealed keyframes; every sealed
    keyframe is published with its one reason, in a piece of sealed keyframes only; every piece outside the chosen
    room is the chosen draw's, and is published with the reasons the chosen draw gave it (patched into `regated`'s
    record and its solution's `components[*].gate_reasons`, as `keep_outside_pieces` does)."""
    if regated.record.get("state") != "applied" or regated.components is None or chosen.components is None:
        return "the seal re-gate did not publish a components record"
    before = _room_kids(chosen.solution, min_obs)
    after = _room_kids(regated.solution, min_obs)
    if not after <= before - set(sealed):
        return (f"the re-gate's room holds {len(after - (before - set(sealed)))} keyframes outside the chosen room "
                "minus the sealed ones")
    entries = regated.components.get("components") or []
    reasons = {kid: list(e.get("reasons") or []) for e in entries for kid in e.get("keyframe_ids") or []}
    published = _published_kids(regated.solution, min_obs)
    wrong = [kid for kid, why in sealed.items() if kid in published and reasons.get(kid) != [why]]
    if wrong:
        return f"{len(wrong)} sealed keyframes were not published with their one reason"
    src_of = {kid: e for e in chosen.components.get("components") or [] for kid in e.get("keyframe_ids") or []}
    outside = {e.get("id"): set(e.get("keyframe_ids") or []) for e in chosen.components.get("components") or []
               if e.get("state") != "placed"}
    kept: dict = {}
    seen_outside: set = set()
    for e in entries:
        kids = set(e.get("keyframe_ids") or [])
        if e.get("state") == "placed" or not kids:
            continue
        if kids & set(sealed):
            if not kids <= set(sealed):
                return "a sealed piece was published in one piece with other keyframes"
            continue
        if kids <= before:
            continue                       # collateral: a piece of the chosen room, with the gate's own reasons
        src = src_of.get(next(iter(sorted(kids))))
        if src is None or set(src.get("keyframe_ids") or []) != kids:
            return "a piece outside the chosen room changed its keyframes"
        kept[e["id"]] = (list(src.get("reasons") or []), src.get("reason"))
        seen_outside.add(src.get("id"))
    if any(i not in seen_outside for i in outside):
        return "a piece outside the chosen room was not published as it was"
    for e in entries:
        if e.get("id") in kept:
            e["reasons"], e["reason"] = kept[e["id"]]
    reasons_of = {kid: kept[e["id"]][0] for e in entries if e.get("id") in kept for kid in e.get("keyframe_ids") or []}
    poses = regated.solution.poses or {}
    for comp in getattr(regated.solution, "components", None) or []:
        members = [kid for kid, p in poses.items() if int(p.get("component", 0)) == int(comp.get("index", -1))
                   and int(p.get("observations", 0)) >= min_obs]
        if members and members[0] in reasons_of:
            comp["gate_reasons"] = list(reasons_of[members[0]])
    return None


# ---------------------------------------------------------------------------------------------------------------
# the whole step


# The record keys the seal re-gate decides; every other key of the published record is the chosen draw's.
_GATE_KEYS = ("attach", "evidence", "labels", "components", "components_file", "params", "params_digest",
              "metric_available", "masks_applied")


def verify_published(result, *, keyframes, parts, regate: Callable, workspace_root, min_obs: int = 30,
                     pair_builder: Callable | None = None, vp: VerifyParams | None = None,
                     mp: MotionParams | None = None, gp: CG.GateParams | None = None):
    """RULE.md section 1 step 3 on a gated result (a `coherence_publish.GateResult`, the chosen draw as the
    consensus publishes it). Never raises: a failure publishes `result` as it was gated, and says so.

    `regate(seal={name: reason}, room=[names]) -> GateResult`: the seal re-gate (the caller's
    `gate_final_solution` with the same candidate, depth and scale, and the consensus's withhold). Returns the
    published result, its record carrying `anchor_verify`."""
    vp = vp or VerifyParams()
    mp = mp or MotionParams()
    gp = gp or CG.GateParams()
    parts = frozenset(parts)
    started = time.perf_counter()
    audit: dict = {"id": ANCHOR_VERIFY_ID, "parts": sorted(parts), "params": vp.to_json(),
                   "params_digest": vp.digest(), "motion_params": mp.to_json(),
                   "image_seal_reason": IMAGE_SEAL_REASON}
    if PART_SCALE in parts:
        audit["a2"] = {"spec": A2_SPEC, "digest": a2_digest(vp)}

    def done(published, state, **kw):
        audit.update(state=state, **kw)
        audit["seconds"] = round(time.perf_counter() - started, 3)
        record = dict(result.record)
        if published is not result:
            record.update({k: published.record.get(k) for k in _GATE_KEYS if k in published.record})
        record["seconds"] = round(float(result.record.get("seconds") or 0.0) + audit["seconds"], 3)
        record["anchor_verify"] = audit
        return dataclasses.replace(published, record=record, consensus_detail=result.consensus_detail,
                                   draw_0=result.draw_0, withheld=result.withheld, depth=result.depth,
                                   scale=result.scale)

    try:
        if not ({PART_SCALE, PART_IMAGES, PART_MOTION} & parts):
            return done(result, STATE_NOT_RUN, why="only the gate's own part (imports) is on")
        rec = result.record or {}
        if rec.get("state") != "applied" or not rec.get("attach") or result.gated is None \
                or result.components is None:
            return done(result, STATE_NOT_RUN, why="the gate attached nothing (a fail-safe) or did not publish a "
                                                   "components record: there is no room to verify")
        x = inputs_of(result, keyframes, min_obs)
        pairs = None
        if PART_IMAGES in parts or PART_SCALE in parts:
            arrays, pair_record = (pair_builder or _default_pair_builder)(workspace_root, x.names,
                                                                          result.solution.camera)
            audit["pair_set"] = pair_record
            pairs = Pairs(arrays, x.world.n, vp.t_min_parallax_deg)
        seal_w: dict = {}
        rounds: list = []
        cur, published = result, result
        collateral_all: set = set()
        cap_hit = False
        before = _room_kids(result.solution, min_obs)
        while True:
            new, rd = analyse(cur, x, pairs, parts, vp, mp, gp)
            new = {w: why for w, why in new.items() if w not in seal_w}
            rd["sealed_new"] = {why: sum(1 for v in new.values() if v == why) for why in sorted(set(new.values()))}
            rounds.append(rd)
            if not new:
                break
            seal_w.update(new)
            seal = {x.names[w]: why for w, why in seal_w.items()}
            room_names = sorted(x.names[x.w_of_kid[kid]] for kid in before if kid in x.w_of_kid)
            regated = regate(seal=seal, room=room_names)
            sealed_kids = {x.kids[w]: why for w, why in seal_w.items()}
            refusal = seal_refusal(result, regated, sealed_kids, min_obs=min_obs)
            if refusal is not None:
                audit.update(rounds=rounds, sealed=_sealed_summary(sealed_kids))
                logger.warning("[Tower][WorldBuilder] anchor verification: the seal re-gate failed its check (%s); "
                               "the room is published as it was gated", refusal)
                return done(result, STATE_NOT_APPLIED, why=f"the seal re-gate failed its check ({refusal}); the "
                                                            "chosen draw is published as it was gated", sealed_kf=0)
            published = regated
            after = _room_kids(regated.solution, min_obs)
            collateral = before - set(sealed_kids) - after
            grew = collateral - collateral_all
            collateral_all |= collateral
            rd["collateral"] = sorted(grew)
            if not grew:
                break
            if len(rounds) >= MAX_ROUNDS:
                cap_hit = True
                break
            cur = regated
        sealed_kids = {x.kids[w]: why for w, why in seal_w.items()}
        # Which part sealed which piece, with its statistic (manager 086): Tower-internal only, never in the
        # components record, whose reason is the contract's.
        sealed_groups = [dict(g, round=k + 1) for k, rd in enumerate(rounds) for g in rd.get("sealed_groups") or []
                         if any(kid in sealed_kids for kid in g["keyframe_ids"])]
        return done(published, STATE_APPLIED, rounds=rounds, cap_hit=cap_hit, sealed=_sealed_summary(sealed_kids),
                    sealed_groups=sealed_groups,
                    sealed_kf=len(sealed_kids), collateral={"keyframes": len(collateral_all),
                                                            "keyframe_ids": sorted(collateral_all)},
                    room_before=len(before), room_after=len(_room_kids(published.solution, min_obs)),
                    image_verification=_first(rounds, "image_verification"),
                    anchor_scale=_first(rounds, "anchor_scale"), motion_flags=_first(rounds, "motion_flags"))
    except Exception as exc:  # noqa: BLE001 -- the room is published as gated, and the record says why
        logger.exception("[Tower][WorldBuilder] anchor verification failed; the room is published as it was gated")
        return done(result, STATE_FAILED, detail=f"{type(exc).__name__}: {exc}")


def _first(rounds: list, key: str):
    """The first round's block (the verdicts on the room as the gate published it); every round is in `rounds`."""
    return next((rd[key] for rd in rounds if key in rd), None)


def _sealed_summary(sealed_kids: dict) -> dict:
    out: dict = {}
    for kid, why in sorted(sealed_kids.items()):
        out.setdefault(why, []).append(kid)
    return {why: {"keyframes": len(kids), "keyframe_ids": kids} for why, kids in out.items()}


def _default_pair_builder(workspace_root, names, camera):
    return build_masked_pairs(workspace_root, names, camera)


# ---------------------------------------------------------------------------------------------------------------
# part (i): one relocalizer import counts as ONE link (RULE.md section 7)


def revisit_pair_episodes(session_dir) -> dict:
    """{(name_a, name_b) sorted: episode} for every pair `relocalizer.revisit_pairs` returns: the same journal read
    (`recovery_accepted` and its `recovery_anchored` follow-ups, every live leg at `REVISIT_MIN_INLIERS`), keeping
    which episode made each pair. Absent journal: {} -- never an error."""
    from tower.world_builder import relocalizer as RL  # noqa: PLC0415

    session_dir = Path(session_dir)
    events_path, keyframes_path = session_dir / "events.jsonl", session_dir / "keyframes.jsonl"
    if not events_path.is_file() or not keyframes_path.is_file():
        return {}
    names: dict = {}
    for record in RL._read_jsonl(keyframes_path):
        kid, rel = record.get("keyframe_id"), record.get("image_relpath")
        if isinstance(kid, str) and isinstance(rel, str):
            names[kid] = Path(rel).name
    out: dict = {}
    accepted_refs: dict = {}

    def at_floor(inliers) -> bool:
        return (isinstance(inliers, (int, float)) and not isinstance(inliers, bool)
                and inliers >= RL.REVISIT_MIN_INLIERS)

    def add(refs, anchor, episode):
        anchor_name = names.get(anchor.get("keyframe_id")) if isinstance(anchor, dict) else None
        if anchor_name is None:
            return
        if anchor.get("identity") is not True and not at_floor(anchor.get("inliers")):
            return
        for ref in refs:
            ref_name = names.get(ref)
            if ref_name is None or ref_name == anchor_name:
                continue
            out.setdefault(tuple(sorted((ref_name, anchor_name))), episode)

    for event in RL._read_jsonl(events_path):
        kind = event.get("kind")
        if kind not in (RL.EVENT_ACCEPTED, RL.EVENT_ANCHORED):
            continue
        payload = RL._payload(event)
        episode = payload.get("episode")
        if kind == RL.EVENT_ACCEPTED:
            refs = [link.get("ref_keyframe_id") for link in payload.get("links") or ()
                    if isinstance(link, dict) and at_floor(link.get("inliers"))]
            accepted_refs[episode] = refs
            add(refs, payload.get("anchor"), episode)
        elif episode in accepted_refs:
            add(accepted_refs[episode], payload.get("anchor"), episode)
    return out


def imported_pairs(database_path) -> set | None:
    """The image-name pairs (sorted) of the mapped database's `wb_revisit_imported` table (the pairs the revisit
    import CREATED and kept), or None when the database has no such table. Read-only and immutable."""
    import sqlite3  # noqa: PLC0415

    from tower.world_builder.global_solve import REVISIT_IMPORTED_TABLE  # noqa: PLC0415

    path = Path(database_path)
    if not path.is_file():
        return None
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        if con.execute("select name from sqlite_master where type = 'table' and name = ?",
                       (REVISIT_IMPORTED_TABLE,)).fetchone() is None:
            return None
        names = dict(con.execute("select image_id, name from images").fetchall())
        out = set()
        for (pid,) in con.execute(f'select pair_id from "{REVISIT_IMPORTED_TABLE}"'):
            b = int(pid) % CG._COLMAP_PAIR_BASE
            a = (int(pid) - b) // CG._COLMAP_PAIR_BASE
            if a in names and b in names:
                out.add(tuple(sorted((names[a], names[b]))))
        return out
    finally:
        con.close()


def import_link_units(store, world_id: str, session_id: str, database_path, name_of: dict) -> tuple[dict, dict]:
    """RULE.md section 7: ({(name_a, name_b) sorted: "import:<episode>"}, audit) for the pairs the relocalizer import
    CREATED and kept (the mapped database's `wb_revisit_imported`), each in the unit of the episode that made it
    (`revisit_pair_episodes`). Every other link -- a listed pair the database already held (`kept_existing`), or a
    created pair no journaled episode names -- is a unit of its own. No table or no journal: ({}, audit), today's
    gate. Never raises: an unreadable input is today's gate too, and the audit says why."""
    audit: dict = {"part": PART_IMPORTS}
    try:
        created = imported_pairs(database_path)
        if not created:
            audit.update(imported_pairs=0, units=0, why="the database holds no imported revisit pair")
            return {}, audit
        episodes = revisit_pair_episodes(store.session_dir(world_id, session_id))
        units = {pair: f"import:{episodes[pair]}" for pair in sorted(created) if pair in episodes}
        audit.update(imported_pairs=len(created), in_episodes=len(units),
                     units=len(set(units.values())),
                     pairs_per_unit=max((list(units.values()).count(u) for u in set(units.values())), default=0))
        return units, audit
    except Exception as exc:  # noqa: BLE001 -- today's gate, and said so
        logger.warning("[Tower][WorldBuilder] anchor verification (imports): could not read the imported pairs "
                       "(%s: %s); each link counts on its own, as today", type(exc).__name__, exc)
        audit.update(state="failed", detail=f"{type(exc).__name__}: {exc}")
        return {}, audit
