"""Dense reconstruction of a solved world.

WHY THIS EXISTS

The global solve places a few hundred cameras to sub-pixel accuracy and then
keeps about fifteen thousand points. Those same cameras hold roughly a hundred
million pixel observations. The sparse cloud is not what the data contains; it
is what triangulating SIFT keypoints happens to retain. This module spends the
rest of the evidence.

WHAT MAKES IT HONEST

A monocular depth network knows the SHAPE of a scene and nothing about its size
or position. So it is never asked for either. For each keyframe we fit

    disparity(u, v)  ~=  a * (1 / z_sfm)  +  b

against the sparse points the global solve ALREADY placed in that frame, by
iteratively reweighted least squares with a Huber loss, and then read depth back
out as z = a / (disparity - b). Scale, position and orientation come entirely
from multi-view triangulated geometry. The network only interpolates between
points that the solve earned.

Four things then remove what is not evidence:

  1. a per-frame gate on the HELD-OUT alignment residual -- fit on half the
     sparse points, score on the half the fit never saw, and drop the frame
     entirely if it misses. A dropped frame leaves its part of the room empty.
  2. a geometric validity mask -- reject pixels strung across a depth edge
     ("flying pixels") and pixels on surfaces viewed at a grazing angle, where
     depth error explodes.
  3. multi-view consensus -- a point survives only if at least `min_views`
     OTHER cameras independently hold a depth that agrees within `tau`.
     Independently wrong depths do not agree, which is why this, and not a
     network's own confidence head, is the mechanism that separates evidence
     from invention.
  4. no hole filling of any kind. No Poisson, no TSDF closure, no generative
     completion. Poisson and Delaunay are closure methods -- watertight by
     construction -- and would silently turn "never observed" into "surface
     here". An unobserved region stays empty.

Every surviving point carries the number of cameras that agreed with it, so a
viewer can raise the bar and watch weak geometry disappear.

WHAT IT DOES NOT CLAIM

Scale semantics are inherited unchanged from the solve. If the world's scale is
`unknown`, so is the dense artifact's. Only component 0 is densified, because
COLMAP normalises each component separately and two components of one world are
not in the same unit.

LIFECYCLE

Four stages, each checkpointed to disk, so an interrupted run resumes instead of
restarting -- Stop kills the process tree on a 30-second grace and this work
takes minutes. `should_stop` is polled between frames and between stages, and a
stop leaves a legible `status.json` rather than a half-written artifact.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

logger = logging.getLogger(__name__)

# The artifact format identifier. A reader that does not recognise this string
# must ignore the dense subtree rather than guess at it.
DENSE_FORMAT = "wb-dense-points/1"

# Stage names, in order. Each writes its own output and is skipped when that
# output is already present and current.
STAGE_DEPTH = "depth"
STAGE_FUSE = "fuse"
STAGE_PACK = "pack"
STAGES = (STAGE_DEPTH, STAGE_FUSE, STAGE_PACK)

STATE_OK = "ok"
STATE_RUNNING = "running"
STATE_STOPPED = "stopped"
STATE_FAILED = "failed"
STATE_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DenseParams:
    """Everything that changes the output, in one place, recorded in the manifest.

    Defaults are the values that were measured, not guessed. See
    `docs/world-builder-dense/03-DECISIONS.md` for what each was compared
    against.
    """

    # -- depth ------------------------------------------------------------
    backend: str = "depth-anything-v2-small"
    # Reject a frame whose HELD-OUT relative depth residual exceeds this.
    # 0.08 keeps 346 of 425 frames on the reference world; a regularised refit
    # rescued only one of the rejects, which is the evidence that the rejects
    # are genuinely unreliable rather than merely ill-conditioned.
    gate_rel: float = 0.08
    min_sparse_points: int = 20

    # -- validity mask ----------------------------------------------------
    edge_rel: float = 0.03
    max_grazing_deg: float = 80.0
    erode_px: int = 1
    # Inpaint redaction fill before the depth network sees it. A solid black
    # rectangle does not merely lose its own pixels: it drags the network's
    # estimate for the WHOLE frame. Measured on one capture, frames with over
    # 30% filled had a 34.8% held-out residual and not one passed the 8% gate,
    # while frames under 10% filled sat near 6%. The inpainted pixels are still
    # masked out of the reconstruction afterwards -- this exists to protect the
    # rest of the frame, not to recover the hole.
    inpaint_redaction_fill: bool = True

    # -- consensus --------------------------------------------------------
    neighbours: int = 10
    tau: float = 0.03
    min_views: int = 3
    average_views: bool = True
    stride: int = 1
    max_depth_pct: float = 97.0

    # -- output -----------------------------------------------------------
    # Voxel sizes are FRACTIONS OF THE SCENE'S OWN MEDIAN DEPTH, never absolute
    # world units. `global_solve.py` never calls COLMAP's `normalize()`, so the
    # gauge is whatever the first baseline happened to be and it varies wildly
    # between worlds -- solves in this corpus range from a ten-unit extent to a
    # three-hundred-unit one for the same kind of walk. A fixed voxel would
    # therefore mean centimetres in one world and metres in another: it would
    # shatter one reconstruction into hundreds of millions of points and
    # collapse another into a blob.
    #
    # The chosen fractions reproduce 0.021 / 0.045 / 0.091 on the reference
    # world, whose median scene depth is 6.99. The corpus is 0.23 MP, giving
    # about 6.4 mm per pixel at 3 m and 1-3 cm of depth noise, so the finest
    # level is already at the edge of what the evidence supports and stores
    # mostly noise -- which is why L1, not L0, is canonical.
    lod_depth_fractions: tuple[float, ...] = (0.003, 0.0065, 0.013)
    canonical_level: int = 1
    mobile_level: int = 2
    min_confidence: int = 2

    # -- housekeeping ------------------------------------------------------
    # A 438-keyframe world leaves 601 MB behind, of which about 470 MB is
    # regenerable intermediate: the per-frame depth maps, the undistorted
    # frames, and fused.npz, which holds the same points points_l0.bin already
    # holds. Keeping all of that on every world forever is not a reasonable
    # default for a product. It IS the right default for the development loop,
    # where re-fusing with different parameters off cached depth is the whole
    # iteration, so world_densify.py exposes --keep-intermediates.
    keep_intermediates: bool = False

    # -- scope ------------------------------------------------------------
    component: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["lod_depth_fractions"] = list(self.lod_depth_fractions)
        return d

    def voxels_for(self, median_scene_depth: float) -> list[float]:
        """The LOD ladder in this world's own units."""
        return [f * float(median_scene_depth) for f in self.lod_depth_fractions]


@dataclass
class DenseResult:
    state: str
    detail: str | None = None
    frames_total: int = 0
    frames_aligned: int = 0
    frames_used: int = 0
    frames_dropped: int = 0
    points: int = 0
    levels: list = field(default_factory=list)
    seconds: dict = field(default_factory=dict)
    align_rel_median: float | None = None
    stopped_after: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class DenseUnavailable(RuntimeError):
    """The world cannot be densified, and the reason is not a bug."""


# ---------------------------------------------------------------------------
# depth backends
# ---------------------------------------------------------------------------


class DepthBackend:
    """Predicts affine-invariant inverse depth for one image.

    Kept behind an interface because the choice of network is an empirical
    decision that is expected to be revisited, and because licence terms differ
    between checkpoints of the same family. Only permissively licensed defaults
    are wired in.
    """

    name = "abstract"
    licence = "unknown"

    def predict(self, rgb: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class TransformersDepthBackend(DepthBackend):
    """Any `transformers` depth-estimation checkpoint that returns disparity."""

    def __init__(self, model_id: str, name: str, licence: str) -> None:
        self.model_id = model_id
        self.name = name
        self.licence = licence
        self._model = None
        self._proc = None

    def _load(self) -> None:
        if self._model is not None:
            return
        # Imported here, not at module import: Tower's web process must not
        # pull torch in just because World Builder is declared.
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._proc = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(self.model_id, dtype=dtype)
            .to(device)
            .eval()
        )
        self._device = device
        self._dtype = dtype
        logger.info(
            "[Tower][WorldBuilder][dense] depth backend %s (%s) on %s",
            self.name, self.licence, device,
        )

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        self._load()
        h, w = rgb.shape[:2]
        with torch.no_grad():
            inp = self._proc(images=rgb, return_tensors="pt").to(self._device, self._dtype)
            pred = self._model(**inp).predicted_depth
            pred = torch.nn.functional.interpolate(
                pred[:, None].float(), size=(h, w), mode="bicubic", align_corners=False
            )[0, 0]
        return pred.cpu().numpy().astype(np.float32)


# Only permissively licensed checkpoints are registered. Depth Anything V2
# Base and Large are CC-BY-NC-4.0 and are deliberately absent: this is a
# product, and a non-commercial weight cannot ship in one.
_BACKENDS: dict[str, Callable[[], DepthBackend]] = {
    "depth-anything-v2-small": lambda: TransformersDepthBackend(
        "depth-anything/Depth-Anything-V2-Small-hf",
        "depth-anything-v2-small",
        "Apache-2.0",
    ),
}


def register_backend(name: str, factory: Callable[[], DepthBackend]) -> None:
    _BACKENDS[name] = factory


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def make_backend(name: str) -> DepthBackend:
    try:
        return _BACKENDS[name]()
    except KeyError:
        raise DenseUnavailable(
            f"unknown depth backend {name!r}; available: {', '.join(available_backends())}"
        ) from None


# ---------------------------------------------------------------------------
# geometry -- the pose convention here is the SOLVE's, which is COLMAP's, and
# which is NOT the convention world.json declares for the derived tree.
# Verified by projecting the solved points against the stored observations:
# x_cam = R @ X_world + t lands at 0.615 px; every alternative at 110 px or worse.
# ---------------------------------------------------------------------------


def project(R: np.ndarray, t: np.ndarray, K: np.ndarray, X: np.ndarray):
    """World points -> (pixels, camera-frame depth)."""
    Xc = X @ R.T + t
    z = Xc[:, 2]
    uv = np.full((len(X), 2), np.nan)
    ok = z > 1e-9
    if ok.any():
        uv[ok] = Xc[ok, :2] / z[ok, None] @ np.diag([K[0, 0], K[1, 1]]) + K[[0, 1], 2]
    return uv, z


def unproject(R: np.ndarray, t: np.ndarray, K: np.ndarray, uv: np.ndarray, depth: np.ndarray):
    """Pixels + camera-frame depth -> world points."""
    x = (uv[:, 0] - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) / K[1, 1]
    Xc = np.stack([x * depth, y * depth, depth], 1)
    return (Xc - t) @ R


def camera_centre(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


def robust_affine(disp: np.ndarray, inv_z: np.ndarray, iters: int = 12):
    """Fit disp = a * inv_z + b by IRLS with a Huber weight.

    Returns (a, b). Least squares alone is not enough: a handful of sparse
    points land on a moving object or a reflection, and one of those drags an
    unweighted fit far enough to ruin a whole frame.
    """
    A = np.stack([inv_z, np.ones_like(inv_z)], 1)
    w = np.ones_like(inv_z)
    a = b = 0.0
    for _ in range(iters):
        M = A * w[:, None]
        try:
            sol, *_ = np.linalg.lstsq(M, disp * w, rcond=None)
        except np.linalg.LinAlgError:
            break
        a, b = float(sol[0]), float(sol[1])
        r = disp - (a * inv_z + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        d = 1.5 * s
        w = np.where(np.abs(r) <= d, 1.0, d / (np.abs(r) + 1e-12))
    return a, b


def _relative_residual(disp: np.ndarray, z_true: np.ndarray, a: float, b: float):
    """Relative depth error of an (a, b) fit, or None if it is infeasible."""
    if a <= 0:
        return None
    den = disp - b
    ok = den > 1e-6
    if ok.sum() < 5:
        return None
    z = a / den[ok]
    rel = np.abs(z - z_true[ok]) / z_true[ok]
    rel = rel[np.isfinite(rel)]
    return rel if len(rel) else None


def align_frame(disp_at_points: np.ndarray, z_sfm: np.ndarray):
    """Fit one frame, and score it on sparse points the fit never saw.

    The held-out split is the whole point. An in-sample residual measures how
    well two free parameters can chase the data; it says nothing about whether
    the depth between the sparse points can be trusted, which is exactly what
    the dense stage is about to rely on.
    """
    inv_z = 1.0 / z_sfm
    idx = np.arange(len(inv_z))
    fit_m, ho_m = idx % 2 == 0, idx % 2 == 1
    ho_med = None
    if fit_m.sum() >= 10 and ho_m.sum() >= 10:
        ha, hb = robust_affine(disp_at_points[fit_m], inv_z[fit_m])
        rel = _relative_residual(disp_at_points[ho_m], z_sfm[ho_m], ha, hb)
        if rel is not None:
            ho_med = float(np.median(rel))
    a, b = robust_affine(disp_at_points, inv_z)
    return a, b, ho_med


def validity_mask(
    depth: np.ndarray, K: np.ndarray, *, edge_rel: float, max_grazing_deg: float, erode_px: int
) -> np.ndarray:
    """Pixels whose depth is locally smooth and not seen edge-on.

    A pixel spanning a depth discontinuity gets a depth that belongs to neither
    surface and back-projects into empty space -- the classic "flying pixel"
    streak. A pixel on a surface viewed at a grazing angle has depth error that
    grows without bound. Neither is evidence.
    """
    import cv2

    H, W = depth.shape
    z = depth
    gy, gx = np.gradient(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        edge = (np.abs(gx) + np.abs(gy)) / np.maximum(z, 1e-6)
    ok = np.isfinite(z) & (z > 1e-3) & np.isfinite(edge) & (edge < edge_rel)

    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xn = (uu - K[0, 2]) / K[0, 0]
    yn = (vv - K[1, 2]) / K[1, 1]
    Px, Py = xn * z, yn * z
    t1 = np.stack([np.gradient(Px, axis=1), np.gradient(Py, axis=1), np.gradient(z, axis=1)], -1)
    t2 = np.stack([np.gradient(Px, axis=0), np.gradient(Py, axis=0), np.gradient(z, axis=0)], -1)
    nrm = np.cross(t1, t2)
    nl = np.linalg.norm(nrm, axis=-1)
    ray = np.stack([xn, yn, np.ones_like(xn)], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.abs((nrm * ray).sum(-1) / np.maximum(nl, 1e-12))
    ok &= np.isfinite(cosang) & (cosang > np.cos(np.deg2rad(max_grazing_deg)))

    if erode_px:
        ok = cv2.erode(ok.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=erode_px).astype(bool)
    return ok


# ---------------------------------------------------------------------------
# voxel reduction and packing
# ---------------------------------------------------------------------------


def voxel_reduce(X: np.ndarray, C: np.ndarray, F: np.ndarray, voxel: float):
    """Average positions and colours inside each voxel; keep the best confidence."""
    g = np.floor(X / voxel).astype(np.int64) + (1 << 20)
    key = (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]
    order = np.argsort(key, kind="stable")
    uniq, counts = np.unique(key[order], return_counts=True)
    grp = np.repeat(np.arange(len(uniq)), counts)
    P = np.zeros((len(uniq), 3))
    Cc = np.zeros((len(uniq), 3))
    FF = np.zeros(len(uniq), np.int64)
    np.add.at(P, grp, X[order].astype(np.float64))
    np.add.at(Cc, grp, C[order].astype(np.float64))
    np.maximum.at(FF, grp, F[order].astype(np.int64))
    return (
        (P / counts[:, None]).astype(np.float32),
        (Cc / counts[:, None]).astype(np.uint8),
        np.clip(FF, 0, 255).astype(np.uint8),
    )


# Exactly 16 bytes: 3 x float32 + 4 x uint8. The three colour bytes plus the
# confidence byte already fill the word, so there is no padding field -- adding
# one made the record 17 bytes while the manifest still advertised a stride of
# 16, which would have sheared every point in the viewer.
POINT_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
     ("r", "u1"), ("g", "u1"), ("b", "u1"), ("c", "u1")]
)
POINT_STRIDE_BYTES = 16
assert POINT_DTYPE.itemsize == POINT_STRIDE_BYTES


def write_points_bin(path: Path, X: np.ndarray, C: np.ndarray, F: np.ndarray) -> int:
    """Flat interleaved buffer a browser can hand straight to the GPU.

    Written to a temporary file and renamed, so a reader never sees a partial
    buffer. A half-written points file does not fail loudly -- it is a valid
    file of the wrong length, and the viewer would either refuse it or shear
    the scene.
    """
    a = np.zeros(len(X), dtype=POINT_DTYPE)
    a["x"], a["y"], a["z"] = X[:, 0], X[:, 1], X[:, 2]
    a["r"], a["g"], a["b"] = C[:, 0], C[:, 1], C[:, 2]
    a["c"] = F
    tmp = path.with_suffix(path.suffix + ".tmp")
    a.tofile(tmp)
    tmp.replace(path)
    return path.stat().st_size


def read_points_bin(path: Path):
    a = np.fromfile(path, dtype=POINT_DTYPE)
    X = np.stack([a["x"], a["y"], a["z"]], 1)
    C = np.stack([a["r"], a["g"], a["b"]], 1)
    return X, C, a["c"]
