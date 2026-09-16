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
out. Scale, position and orientation come entirely from multi-view triangulated
geometry.

The network interpolates between points the solve earned, and is allowed to
extrapolate past them by at most `max_extrapolation` -- beyond that the pixel is
refused. Without that bound the claim would be false: measured, 11-18% of
surviving pixels per frame sat outside the range their own frame's sparse points
bracketed, by up to 5.18x.

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

Three stages -- depth, fuse, pack -- checkpointed to disk so an interrupted run
resumes rather than restarting. Depth resumes per FRAME: a frame whose
prediction is already on disk is not predicted again, so a stop halfway through
a 429-frame world costs the frames not yet reached, not all of them. Fusion
resumes as a whole. `should_stop` is polled between frames and between stages,
and a stop leaves a legible `status.json` rather than a half-written artifact.
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
    # MoGe-2 ViT-L (MIT). Chosen from a 24-model bake-off on identical frames,
    # then re-measured through this pipeline on the reference world:
    #
    #     backend                held-out median   frames past the 8% gate
    #     depth-anything-v2-small     6.50%              235 / 395
    #     moge2-vitl                  2.60%              317 / 399
    #
    # It also beats the CC-BY-NC checkpoints this lane refused to ship, so
    # there is no accuracy being traded away for the licence -- only gained.
    #
    # The margin over `da3mono-large` (Apache-2.0) is NOT robust across rooms:
    # scored on two held-out solves the two are indistinguishable, and DA3-mono
    # is 2.7x faster in two thirds of the VRAM. MoGe stays the default because
    # it is never worse and this stage runs off the interactive path. On a
    # smaller card, switch. D15 carries the table.
    backend: str = "moge2-vitl"
    # Reject a frame whose HELD-OUT relative depth residual exceeds this.
    # Across the seven worlds 0.08 keeps 50-74% of posed frames; a regularised
    # refit rescued only one of the rejects, which is the evidence that the
    # rejects are genuinely unreliable rather than merely ill-conditioned.
    # NOTE the metric it gates is also the metric this lane reports: the
    # residual quoted anywhere is the median of the frames that PASSED, so it
    # improves as this number falls and the reconstruction gets worse.
    # 01-EVIDENCE.md section 11.3 prints both ends.
    gate_rel: float = 0.08
    min_sparse_points: int = 20

    # -- validity mask ----------------------------------------------------
    edge_rel: float = 0.03
    max_grazing_deg: float = 80.0
    erode_px: int = 1
    # How far past the depth range of a frame's OWN sparse points a pixel may
    # sit and still be used. "The network only interpolates" was an overstatement
    # before this existed: nothing clamped a pixel to the range the solve had
    # bracketed, and 11-18% of surviving pixels per frame fell outside it, up to
    # 5.18x past the farthest sparse point. Consensus removed about 92% of those,
    # but 4.4% of the shipped cloud was still geometry no solved point bracketed.
    # 1.5 keeps honest near/far margin around the sparse hull and refuses the rest.
    max_extrapolation: float = 1.5
    # Inpaint redaction fill before the depth network sees it. A solid black
    # rectangle does not merely lose its own pixels: it drags the network's
    # estimate for the WHOLE frame. Measured on one capture, frames with over
    # 30% filled had a 34.8% held-out residual and not one passed the 8% gate,
    # while frames under 10% filled sat near 6%. The inpainted pixels are still
    # masked out of the reconstruction afterwards -- this exists to protect the
    # rest of the frame, not to recover the hole.
    inpaint_redaction_fill: bool = True

    # -- consensus --------------------------------------------------------
    # How many nearby cameras the agreement test consults. Raising it does
    # NOT lower the bar -- it looks harder for witnesses that already meet
    # it -- so it was the obvious lever for coverage. Measured on the desk
    # world by re-fusing from cached depth: 16/24/32 buy 6-11% more points
    # and about a point of coverage, and cost 37% on the p90 depth error
    # (12.7% -> 17.4%). A more distant camera is a weaker witness, so the
    # points only a wider search rescues are the worse ones. D20.
    neighbours: int = 10
    # How closely a neighbouring camera must agree, as a fraction of depth.
    # 0.05 rather than the obvious 0.03, measured on two independent walks:
    # it adds 28-36% more points and 7-8 points of pixel coverage while depth
    # accuracy holds or IMPROVES. The mechanism is that a looser threshold
    # admits more AGREEING cameras per point, and each point's position is the
    # mean of them -- so the extra averaging cancels more per-frame error than
    # the looser threshold lets in. Tightening to 0.03 keeps fewer, noisier,
    # less-averaged points.
    tau: float = 0.05
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
    # Every point below this is dropped before the ladder is written. The
    # consensus filter already requires `min_views` OTHER cameras, so the
    # lowest confidence any surviving point can carry is `min_views`, which is
    # 3 -- measured as the minimum on all eight artifacts. This threshold is
    # therefore INERT at its default and kept as a floor a stricter operator
    # can raise, not as a filter that is doing work today. The contract used to
    # advertise it as one.
    min_confidence: int = 2
    # The packed ladder also drops points outside this central percentile box
    # on each axis. It exists because a handful of points at extreme depth,
    # surviving consensus because several nearby frames made the SAME error,
    # otherwise stretch the bounding box and with it the viewer's whole opening
    # framing. It was hard-coded and undeclared, which meant a filter that
    # removes real observations did not appear in the manifest, in the params,
    # or in the format's list of the four things it refuses.
    pack_percentile: float = 0.2

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
    # True when this run found a complete artifact from the same solve and the
    # same parameters and did nothing. Additive, and worth reporting: a caller
    # that sees zero seconds and thousands of points should be told why.
    reused: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


class DenseUnavailable(RuntimeError):
    """The world cannot be densified, and the reason is not a bug."""


class DepthModelUnavailable(DenseUnavailable):
    """This MACHINE cannot run the depth network right now: its package, torch,
    or its weights are missing, or the backend is not one this Tower knows.

    Separate from `DenseUnavailable` because the two call for different
    responses. A session-specific refusal (a camera the poses were not solved
    in, too few frames passing the gate) says nothing about the next session,
    while this says every later build on this machine will fail the same way
    until someone installs something or the network comes back -- which is
    what lets the surface stage mark it permanent and the live worker stop
    relaunching a child that cannot succeed (review 3, e2e m1 and m2).
    """


# The Hugging Face hub raises these when the weights are neither in the local
# cache nor downloadable: offline (HF_HUB_OFFLINE, or no route to the hub) on a
# machine that never fetched them. Resolved lazily so importing this module
# never needs huggingface_hub.
def _weights_missing_errors() -> tuple:
    errors = []
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: PLC0415

        errors.append(LocalEntryNotFoundError)
    except ImportError:
        pass
    try:
        from huggingface_hub.errors import OfflineModeIsEnabled  # noqa: PLC0415

        errors.append(OfflineModeIsEnabled)
    except ImportError:
        pass
    return tuple(errors)


def hub_model_cache(model_id: str) -> Path | None:
    """Where the hub keeps `model_id`'s files on this machine, or None."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415
    except ImportError:
        return None
    return Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--"))


def _tree_bytes(path: Path | None) -> int:
    if path is None:
        return 0
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                st = item.lstat()
            except OSError:
                continue
            if not item.is_symlink() and item.is_file():
                total += st.st_size
    except OSError:
        pass
    return total


def load_hub_weights(name: str, model_id: str, load: Callable[[], object]):
    """Run `load` (a `from_pretrained`), turning "the weights are not here and
    cannot be fetched" into `DepthModelUnavailable` naming the model and the
    cache, and logging what a first-run download cost.

    The default backend's weights are about 1.3 GB and are fetched on first
    use. Before this, a machine that was offline on its first walk raised the
    hub's own `LocalEntryNotFoundError`, which the surface stage recorded as an
    ordinary failure, so the live worker relaunched a child on every solve.
    """
    cache = hub_model_cache(model_id)
    before = _tree_bytes(cache)
    t0 = time.time()
    missing = _weights_missing_errors()
    try:
        model = load()
    except Exception as exc:  # noqa: BLE001 -- re-raised unless it is the one case
        if missing and isinstance(exc, missing):
            raise DepthModelUnavailable(
                f"depth model {model_id!r} (backend {name!r}) is not in the "
                f"Hugging Face cache ({cache}) and could not be downloaded: "
                f"{type(exc).__name__}. Connect this machine to the internet "
                "for its first build (about 1.3 GB for the default model), or "
                "pre-seed the cache, then build again"
            ) from None
        raise
    grown = _tree_bytes(cache) - before
    if grown > 1_000_000:
        logger.info("[Tower][WorldBuilder][dense] downloaded depth model %s: "
                    "%.0f MB in %.0f s into %s", model_id, grown / 1e6,
                    time.time() - t0, cache)
    return model


# ---------------------------------------------------------------------------
# depth backends
# ---------------------------------------------------------------------------


class DepthBackend:
    """Predicts affine-invariant inverse depth for one image.

    Kept behind an interface because the choice of network is an empirical
    decision that is expected to be revisited, and because licence terms differ
    between checkpoints of the same family. Only permissively licensed defaults
    are wired in.

    Some models are not per-image at all: they take a WINDOW of frames and,
    optionally, the poses the solve already recovered, and reason across them.
    Those override `predict_window` and declare `windowed = True`; the caller
    then feeds whole windows instead of single frames. What comes back is still
    per-frame inverse depth, so everything downstream -- the per-frame fit to
    the sparse points, the gate, the mask, the consensus -- is unchanged. In
    particular the per-frame fit absorbs any window-to-window scale
    disagreement, which is the failure mode windowed models are prone to.
    """

    name = "abstract"
    licence = "unknown"
    windowed = False
    window_size = 0
    # What `predict` returns, which decides the shape of the per-frame fit.
    #   "disparity" -- affine-invariant INVERSE depth (Depth Anything family)
    #   "depth"     -- affine-invariant depth, e.g. the z of a point map (MoGe)
    # Fitting a point map as though it were disparity costs real accuracy, so
    # this is not cosmetic.
    kind = "disparity"

    def predict(self, rgb: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict_window(self, images, R=None, t=None, K=None):  # pragma: no cover
        """A window of HxWx3 RGB images -> a list of inverse-depth maps."""
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


class MoGeBackend(DepthBackend):
    """MoGe, which predicts a point map rather than a disparity image.

    The z channel of that point map is affine-invariant DEPTH, so the per-frame
    fit is `pred ~= a*z + b` rather than `pred ~= a/z + b`. Fitting it as
    disparity anyway is not free: it costs about a third of the model's
    advantage, which is why `kind` exists at all.
    """

    kind = "depth"

    def __init__(self, model_id: str, name: str, licence: str,
                 resolution_level: int = 9) -> None:
        self.model_id = model_id
        self.name = name
        self.licence = licence
        self.resolution_level = resolution_level
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from moge.model.v2 import MoGeModel
        except ImportError as exc:
            raise DepthModelUnavailable(
                f"backend {self.name!r} needs the `moge` package and torch, "
                f"which are not installed ({exc}). Install them, or pass a "
                "different --backend"
            ) from None

        # The device is chosen, never assumed: a Tower without a GPU should fall
        # back rather than raise a CUDA error from inside a finalization step.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = load_hub_weights(
            self.name, self.model_id,
            lambda: MoGeModel.from_pretrained(self.model_id)).to(self._device).eval()
        self._torch = torch
        logger.info("[Tower][WorldBuilder][dense] depth backend %s (%s) on %s",
                    self.name, self.licence, self._device)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        self._load()
        t = self._torch.tensor(rgb / 255.0, dtype=self._torch.float32,
                               device=self._device).permute(2, 0, 1)
        with self._torch.no_grad():
            out = self._model.infer(t, resolution_level=self.resolution_level,
                                    apply_mask=False)
        pts = out["points"].float().cpu().numpy()
        z = np.ascontiguousarray(pts[..., 2]).astype(np.float32)
        mask = out.get("mask")
        if mask is not None:
            z = np.where(mask.cpu().numpy().astype(bool), z, np.nan).astype(np.float32)
        return z


class DepthAnything3Backend(DepthBackend):
    """Depth Anything 3, pose-conditioned, over a window of frames.

    Unlike a per-image model this one is told where the cameras are, so it
    reasons across the window and returns depth already in the solve's frame.
    We still run the per-frame fit against the sparse points on top of it: that
    is what absorbs the window-to-window scale disagreement DA3 is prone to,
    and it is the configuration that measured best.

    Licence care: DA3's own `DEFAULT_MODEL` is `DA3NESTED-GIANT-LARGE-1.1`,
    which is CC-BY-NC. It is never used here -- the checkpoint is always passed
    explicitly, and only Apache-2.0 ones are registered.
    """

    windowed = True

    def __init__(self, model_id: str, name: str, licence: str,
                 window_size: int = 60, process_res: int = 640) -> None:
        self.model_id = model_id
        self.name = name
        self.licence = licence
        self.window_size = window_size
        self.process_res = process_res
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise DepthModelUnavailable(
                f"backend {self.name!r} needs the `depth-anything-3` package "
                f"and torch, which are not installed ({exc}). Install them, or "
                "pass a different --backend"
            ) from None

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = load_hub_weights(
            self.name, self.model_id,
            lambda: DepthAnything3.from_pretrained(self.model_id)).to(self._device).eval()
        logger.info("[Tower][WorldBuilder][dense] depth backend %s (%s) on %s, window %d",
                    self.name, self.licence, self._device, self.window_size)
        self._torch = torch

    def predict_window(self, images, R=None, t=None, K=None):
        import cv2
        import numpy as _np

        self._load()
        ext = ixt = None
        if R is not None and t is not None and K is not None:
            n = len(images)
            ext = _np.repeat(_np.eye(4, dtype=_np.float64)[None], n, axis=0)
            for i in range(n):
                ext[i, :3, :3] = R[i]
                ext[i, :3, 3] = t[i]
            ixt = _np.repeat(_np.asarray(K, dtype=_np.float64)[None], n, axis=0)
        with self._torch.no_grad():
            pred = self._model.inference(
                image=[_np.ascontiguousarray(im) for im in images],
                extrinsics=ext, intrinsics=ixt,
                align_to_input_ext_scale=True,
                process_res=self.process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )
        depth = _np.asarray(pred.depth, dtype=_np.float32)
        out = []
        for i, im in enumerate(images):
            h, w = im.shape[:2]
            d = depth[i]
            if d.shape != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            # Hand back INVERSE depth, so the per-frame fit downstream is the
            # same fit it performs for every other backend.
            with _np.errstate(divide="ignore", invalid="ignore"):
                out.append(_np.where(d > 1e-6, 1.0 / d, _np.nan).astype(_np.float32))
        return out


# Only permissively licensed checkpoints are registered. Depth Anything V2
# Base and Large are CC-BY-NC-4.0 and are deliberately absent: this is a
# product, and a non-commercial weight cannot ship in one.
_BACKENDS: dict[str, Callable[[], DepthBackend]] = {
    "depth-anything-v2-small": lambda: TransformersDepthBackend(
        "depth-anything/Depth-Anything-V2-Small-hf",
        "depth-anything-v2-small",
        "Apache-2.0",
    ),
    "moge2-vitl": lambda: MoGeBackend(
        "Ruicheng/moge-2-vitl", "moge2-vitl", "MIT",
    ),
    "moge2-vits": lambda: MoGeBackend(
        "Ruicheng/moge-2-vits-normal", "moge2-vits", "MIT",
    ),
    "da3-base": lambda: DepthAnything3Backend(
        "depth-anything/DA3-BASE", "da3-base", "Apache-2.0",
    ),
    "da3-large": lambda: DepthAnything3Backend(
        "depth-anything/DA3-LARGE-1.1", "da3-large", "Apache-2.0",
    ),
    "da3-small": lambda: DepthAnything3Backend(
        "depth-anything/DA3-SMALL", "da3-small", "Apache-2.0",
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
        raise DepthModelUnavailable(
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


def depth_from_prediction(pred, a: float, b: float, kind: str = "disparity"):
    """Turn a model's affine-invariant output into depth, given the frame's fit.

    One function, used by the alignment, the scoring and the fusion, so the
    three can never disagree about what a stored map means.
    """
    if kind == "depth":
        # pred ~= a * z + b   ->   z = (pred - b) / a
        with np.errstate(divide="ignore", invalid="ignore"):
            return (pred - b) / a
    # pred ~= a / z + b   ->   z = a / (pred - b)
    den = pred - b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 1e-6, a / den, np.nan)


def fit_target(z_true: np.ndarray, kind: str = "disparity"):
    """The regressor the model's output is linear in."""
    return z_true if kind == "depth" else 1.0 / z_true


def _relative_residual(pred: np.ndarray, z_true: np.ndarray, a: float, b: float,
                       kind: str = "disparity"):
    """Relative depth error of an (a, b) fit, or None if it is infeasible."""
    if a <= 0:
        return None
    z = depth_from_prediction(pred, a, b, kind)
    ok = np.isfinite(z) & (z > 1e-6)
    if ok.sum() < 5:
        return None
    rel = np.abs(z[ok] - z_true[ok]) / z_true[ok]
    rel = rel[np.isfinite(rel)]
    return rel if len(rel) else None


def align_frame(disp_at_points: np.ndarray, z_sfm: np.ndarray, kind: str = "disparity"):
    """Fit one frame, and score it on sparse points the fit never saw.

    The held-out split is the whole point. An in-sample residual measures how
    well two free parameters can chase the data; it says nothing about whether
    the depth between the sparse points can be trusted, which is exactly what
    the dense stage is about to rely on.
    """
    x = fit_target(z_sfm, kind)
    idx = np.arange(len(x))
    fit_m, ho_m = idx % 2 == 0, idx % 2 == 1
    ho_med = None
    if fit_m.sum() >= 10 and ho_m.sum() >= 10:
        ha, hb = robust_affine(disp_at_points[fit_m], x[fit_m])
        rel = _relative_residual(disp_at_points[ho_m], z_sfm[ho_m], ha, hb, kind)
        if rel is not None:
            ho_med = float(np.median(rel))
    a, b = robust_affine(disp_at_points, x)
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
    if g.min() < 0 or g.max() >= (1 << 21):
        # Outside the packing's range two different cells share a key and the
        # reduction silently MERGES them -- points from opposite ends of a
        # scene averaged into one. Demonstrated on a two-cluster cloud three
        # million units apart: 329 distinct cells, 231 returned. The predictor
        # in `dense_render` guarded this and the function that actually does
        # the work did not, which is the wrong way round.
        uniq, inverse = np.unique(g, axis=0, return_inverse=True)
        key = inverse.astype(np.int64)
    else:
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
