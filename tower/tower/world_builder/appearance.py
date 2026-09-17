"""The appearance stage's parts: provenance, masks, gains, selection, encoding.

Contract: `docs/contracts/WORLD-BUILDER-APPEARANCE.md` (`wb-appearance-keyframes/1`).

The saved world's appearance is the wearer's own REDACTED keyframes projected
back onto the proxy surface, blended by view on the phone. This module holds
every piece of that which is a function of arrays; `appearance_pipeline.py`
orchestrates them over a store, publishes the artifact, and owns the lock.

THE PRIVACY BOUNDARY IS ONE FUNCTION. `keyframe_source` is the only code in
this stage that opens a keyframe image, and it opens exactly one directory:
`sessions/<sid>/images`. It never opens the solve's `images/` (the raw frames,
undistorted), a capture, `sources.json`, or the depth stage's `undist/` (which
holds TELEA-inpainted pixels inside the fill -- invented content). This is
enforced by a static test over this module and the pipeline, not only by this
paragraph: an earlier dense stage asserted the same boundary in a docstring and
then read around it.

Measured on the canonical capture by the fix-it privacy lane
(`Glasses-scratch/wb-final-recon/fixit/privacy/PRIVACY.md`): the solve's images
are the raw frames (177 of 181 filled frames non-black inside the fill), the
undist frames are inpainted inside it (mean 69.8 against 5.2), and stored fill
masks miss solid boxes that touch dark scene in 7 of 395 frames. Each rule
below answers one of those measurements.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

import numpy as np

APPEARANCE_FORMAT = "wb-appearance-keyframes/1"
APPEARANCE_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# the redaction label allowlist
# ---------------------------------------------------------------------------

_YUNET = "faces-detected-and-filled/yunet-2023mar@0.30"

# EXACT strings, versioned here, each with the measurement that admitted it.
# A detector or gate that is not in this table is NOT trusted, whatever its
# name suggests: `dense_pipeline` used `label != "none"`, which trusted any
# string at all, including the one gate measured to be weak (L2).
TRUSTED_REDACTION_LABELS = {
    _YUNET: "ungated; fills a superset of every later gate",
    f"{_YUNET}+plausibility1": "0 of 3,232 held-out close faces lost",
    f"{_YUNET}+plausibility3": "1 of 3,232 held-out close faces lost",
}
# Known, and re-redacted rather than trusted: weak at the frame edge.
RERUN_REDACTION_LABELS = {
    f"{_YUNET}+plausibility2": "32 of 3,232 held-out close faces lost, 4 of 594 at the edge",
}

SOURCE_SESSION_KEYFRAMES = "session-keyframes"
ORIGIN_STORED = "session-keyframe"
ORIGIN_REDACTED_HERE = "session-keyframe-redacted-here"
MASK_STORED = "stored-fill"
MASK_RERUN = "rerun-difference+guess"

REFUSED_IMAGE_MISSING = "refused-keyframe-image-missing"
REFUSED_UNDECODABLE = "refused-image-undecodable"
REFUSED_NO_FILL_MASK = "refused-no-fill-mask"
REFUSED_REDACTION_FAILED = "refused-redaction-failed"
REFUSED_CAMERA_MISMATCH = "refused-camera-mismatch"

# ---------------------------------------------------------------------------
# the unobserved mask
# ---------------------------------------------------------------------------

NEARBLACK_MAX = 6
NEARBLACK_OPEN = 16
UNOBSERVED_DILATE_PX = 2
UNOBSERVED_RULE = f"fill2|nearblack{NEARBLACK_MAX}open{NEARBLACK_OPEN}|dilate{UNOBSERVED_DILATE_PX}"

# Alpha is 0 over the transparent core dilated by this much: one ASTC 6x6
# block plus one bilinear tap, so no block that holds a zeroed texel also holds
# an opaque one, and a bilinear fetch of an opaque texel never reaches a zeroed
# one. Measured: alpha survives ASTC 6x6, WebP and ETC2 exactly at this ring.
ALPHA_RING_PX = 7

ENC_ASTC = "astc-6x6-rgba"
ENC_WEBP = "webp-rgba"
ENCODING_CODES = {ENC_ASTC: 1, ENC_WEBP: 2}
CHUNK_MAGIC = b"WBAPCK01"
CHUNK_SCHEMA = 1
CHUNK_SLOTS = 16
ASTC_BLOCK = 6
ASTC_QUALITY = 60.0
WEBP_QUALITY = 90

TIER_PHONE = "phone"
TIER_TOWER = "tower"


class AppearanceUnavailable(RuntimeError):
    """The appearance cannot be built for this session, and why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class AppearanceParams:
    quality: str = "final"
    # occluders (contract §5.3)
    occluder_ratio_max: float = 0.6
    occluder_ratio_min: float = 0.3
    occluder_mad_multiple: float = 4.0
    occluder_open_px: int = 5
    occluder_min_area_frac: float = 0.002
    occluder_dilate_px: int = 4
    # transients the proxy cannot see (contract §5.3, second test)
    transient_grid_px: int = 8
    transient_box_px: int = 15
    transient_vis_tol: float = 0.03
    transient_min_views: int = 6
    transient_consensus: float = 0.7
    transient_abs: float = 0.12
    transient_rel: float = 0.5
    transient_dark: float = 0.03
    transient_saturated: float = 0.92
    transient_frame_max: float = 0.2
    transient_shift_px: int = 6
    transient_open_cells: int = 1
    # the transient DETECTOR (transients.py, contract §5.3a): `union`
    # (Grounding DINO + SAM 2.1 with OneFormer), `oneformer`, or `off`
    transient_detector: str = "union"
    # exposure (§5.4)
    exposure_grid_px: int = 16
    exposure_vis_tol: float = 0.03
    exposure_border_px: int = 8
    exposure_box_px: int = 7
    exposure_dark: float = 0.02
    exposure_saturated: float = 0.97
    exposure_huber: float = 0.15
    exposure_iterations: int = 30
    exposure_min_views: int = 3
    # A cap, seeded, on the observations the IRLS runs over. The full set on the
    # canonical world is ~25M and peaked at 5.2 GB of a shared 12 GB card.
    exposure_max_observations: int = 6_000_000
    # The photometric model (§5.4): `gain` (per keyframe, per channel) or
    # `gain+slope+vignette` (also a per-keyframe log-linear tilt across the
    # image -- auto-exposure and lens falloff that is not radially symmetric --
    # and one radial falloff shared by every keyframe of the lens). The ridges
    # hold a keyframe that saw too little of the room to a flat field.
    exposure_model: str = "gain+slope+vignette"
    exposure_slope_ridge: float = 0.005
    exposure_vignette_ridge: float = 0.0
    exposure_cg_outer: int = 4
    exposure_cg_iterations: int = 40
    # selection (§5.5)
    selection_samples: int = 60_000
    selection_vis_tol: float = 0.03
    selection_second_weight: float = 0.5
    selection_min_gain_frac: float = 1e-4
    phone_budget: int = 128
    quality_floor: float = 0.15
    seed: int = 0
    # encoding (§5.6)
    chunk_slots: int = CHUNK_SLOTS
    astc_quality: float = ASTC_QUALITY
    webp_quality: int = WEBP_QUALITY
    # Alpha rises from 0 to 255 over this many pixels beyond the alpha ring
    # (§5.6): a mask edge fades INTO the evidence instead of cutting it, and
    # never reaches outward past the ring. 0 = the hard edge of the first build.
    alpha_feather_px: int = 8

    @classmethod
    def live(cls, **overrides) -> "AppearanceParams":
        """The walk-time preset: the same rules, less sampling. Being late is
        worse than being coarse, but a relaxed privacy or occluder rule is not
        coarse, it is wrong -- so neither moves."""
        base = dict(quality="live", selection_samples=30_000, exposure_grid_px=24,
                    exposure_iterations=20, transient_grid_px=12,
                    # OneFormer only during a walk: the union costs about
                    # twice as much per keyframe (contract §10).
                    transient_detector="oneformer")
        base.update(overrides)
        return cls(**base)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


@dataclass
class LabelPolicy:
    """What the session record says about its stored keyframes, and what this
    build will therefore do before a pixel is read."""

    session_redaction: str | None
    trusted: bool
    redactor: object | None = None
    redactor_label: str | None = None
    # `store.keyframe_image_set(...)`, read once with the label: which keyframes
    # this build reads. None (a policy built by hand) reads it at the frame.
    image_set: object | None = None

    @property
    def effective(self) -> str:
        if self.trusted:
            return str(self.session_redaction)
        return f"{self.session_redaction if self.session_redaction else 'none'}&{self.redactor_label}"


def read_session_redaction(store, world_id: str, session_id: str) -> str | None:
    """The label of the keyframes a build reads, or None when the record is
    absent or unreadable -- which is treated exactly like `none`.

    Through the store's one accessor: `session.json`'s `redaction` for the
    capture's own keyframes, the re-redacted set's label after a
    `world_reredact.py --apply` switch (contract section 6.5)."""
    return keyframe_set_identity(store, world_id, session_id)[0]


def keyframe_set_identity(store, world_id: str, session_id: str) -> tuple[str | None, str | None]:
    """(label, set token) of the keyframe set a build reads now. The token is
    None for the capture's own keyframes."""
    try:
        image_set = store.keyframe_image_set(world_id, session_id)
    except Exception:  # noqa: BLE001 -- unreadable is untrusted, never an error
        return None, None
    label = image_set.redaction
    return (label if isinstance(label, str) and label else None), image_set.cache_token


def label_is_trusted(label: str | None) -> bool:
    return isinstance(label, str) and label in TRUSTED_REDACTION_LABELS


def resolve_label_policy(store, world_id: str, session_id: str,
                         redactor_factory=None) -> LabelPolicy:
    """Read the label ONCE and decide. Refuses the whole build when the label
    needs a re-redaction and no redactor can run: a layer silently missing
    most of the room is worse than a clear refusal."""
    image_set = store.keyframe_image_set(world_id, session_id)
    label = image_set.redaction if isinstance(image_set.redaction, str) and image_set.redaction else None
    if label_is_trusted(label):
        return LabelPolicy(session_redaction=label, trusted=True, image_set=image_set)
    if redactor_factory is None:
        from tower.world_builder.redaction import FaceRedactor  # noqa: PLC0415

        redactor_factory = FaceRedactor
    redactor = redactor_factory()
    if not getattr(redactor, "available", False):
        raise AppearanceUnavailable(
            f"the session's redaction label {label!r} is not trusted, so every "
            "keyframe must be redacted again before it can become appearance, and "
            f"no face redactor is available ({getattr(redactor, 'unavailable_reason', None)})")
    return LabelPolicy(session_redaction=label, trusted=False, redactor=redactor,
                       redactor_label=str(getattr(redactor, "label", None)),
                       image_set=image_set)


@dataclass
class FrameSource:
    """One keyframe as appearance may use it, or the reason it may not."""

    ki: int
    keyframe_id: str
    refused: str | None = None
    image_bytes: bytes | None = None
    source_sha1: str | None = None
    image_sha1: str | None = None
    origin: str | None = None
    redaction_effective: str | None = None
    rgb: np.ndarray | None = None          # (H, W, 3) uint8, undistorted solve camera
    unobserved: np.ndarray | None = None   # (H, W) bool
    mask_origin: str | None = None


class Undistorter:
    """The solve's own undistortion (`global_solve._undistort_maps`), with its
    maps cached per source size, refusing any ROI that is not the solve camera."""

    def __init__(self, intrinsics, camera: dict):
        self.intrinsics = intrinsics
        self.size = (int(camera["width"]), int(camera["height"]))
        self._maps: dict = {}

    def maps(self, width: int, height: int):
        key = (width, height)
        if key not in self._maps:
            from tower.world_builder.global_solve import _undistort_maps  # noqa: PLC0415

            try:
                m1, m2, roi, _cam = _undistort_maps(self.intrinsics, width, height)
            except Exception:  # noqa: BLE001 -- recorded per frame as a mismatch
                self._maps[key] = None
            else:
                self._maps[key] = (m1, m2, roi) if (roi[2], roi[3]) == self.size else None
        return self._maps[key]

    def remap(self, image, maps, nearest: bool = False):
        import cv2  # noqa: PLC0415

        m1, m2, (x0, y0, w, h) = maps
        out = cv2.remap(image, m1, m2, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
        return out[y0:y0 + h, x0:x0 + w]


def nearblack_blocks(rgb: np.ndarray) -> np.ndarray:
    """Pixels <= NEARBLACK_MAX in every channel, opened with a 16x16 box: what a
    solid redaction box leaves, including one that touches dark scene."""
    import cv2  # noqa: PLC0415

    dark = (rgb.max(axis=2) <= NEARBLACK_MAX).astype(np.uint8)
    return cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            np.ones((NEARBLACK_OPEN, NEARBLACK_OPEN), np.uint8)).astype(bool)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2  # noqa: PLC0415

    if px <= 0 or not mask.any():
        return mask.astype(bool)
    k = 2 * int(px) + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def keyframe_source(store, world_id: str, session_id: str, keyframe_id: str, ki: int, *,
                    policy: LabelPolicy, align_record: dict | None, depth_dir,
                    undistorter: Undistorter, hash_only: bool = False) -> FrameSource:
    """THE provenance function: the only place this stage reads keyframe pixels.

    Returns the bytes the pixels came from, the effective redaction label, the
    unobserved mask in the solve camera, and where that mask came from -- or a
    refusal. Contract §6.
    """
    import cv2  # noqa: PLC0415

    from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
        FILL_RULE,
        _fill_mask_for,
    )

    out = FrameSource(ki=int(ki), keyframe_id=keyframe_id)
    seq = keyframe_id.rsplit(":", 1)[-1]
    image_set = policy.image_set or store.keyframe_image_set(world_id, session_id)
    path = image_set.directory / f"{seq}.jpg"
    try:
        stored = path.read_bytes()
    except OSError:
        out.refused = REFUSED_IMAGE_MISSING
        return out
    out.source_sha1 = hashlib.sha1(stored).hexdigest()
    if hash_only:
        return out

    if policy.trusted:
        data, origin = stored, ORIGIN_STORED
    else:
        result = policy.redactor.redact(stored)
        from tower.world_builder.redaction import REDACTION_NONE  # noqa: PLC0415

        if result.label == REDACTION_NONE:
            # `redact` never raises: it hands back the ORIGINAL bytes labelled
            # `none` when the detector throws on this image.
            out.refused = REFUSED_REDACTION_FAILED
            return out
        data, origin = result.image_bytes, ORIGIN_REDACTED_HERE
    out.image_sha1 = hashlib.sha1(data).hexdigest()

    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        out.refused = REFUSED_UNDECODABLE
        return out
    maps = undistorter.maps(bgr.shape[1], bgr.shape[0])
    if maps is None:
        out.refused = REFUSED_CAMERA_MISMATCH
        return out
    rgb = cv2.cvtColor(undistorter.remap(bgr, maps), cv2.COLOR_BGR2RGB)
    W, H = undistorter.size

    stored_fill = None
    rec_ok = (isinstance(align_record, dict)
              and align_record.get("fill_rule") == FILL_RULE
              and align_record.get("image_sha1") == out.image_sha1)
    if rec_ok and depth_dir is not None:
        try:
            f = np.load(depth_dir / f"{int(ki):05d}_fill.npy")
            if f.shape == (H, W):
                stored_fill = f.astype(bool)
        except (OSError, ValueError):
            stored_fill = None

    if policy.trusted:
        if stored_fill is None:
            # A MISSING MASK IS NOT AN EMPTY ONE (privacy L4).
            out.refused = REFUSED_NO_FILL_MASK
            return out
        fill = stored_fill
        mask_origin = MASK_STORED
    else:
        exact = _fill_mask_for(data, None, stored=stored)
        if exact is None:
            out.refused = REFUSED_NO_FILL_MASK
            return out
        exact = dilate(exact, 3)
        fill = undistorter.remap(exact.astype(np.uint8) * 255, maps, nearest=True) > 0
        mask_origin = MASK_RERUN
        if stored_fill is not None:
            fill = fill | stored_fill
            mask_origin += "+" + MASK_STORED
    unobserved = dilate(fill | nearblack_blocks(rgb), UNOBSERVED_DILATE_PX)

    out.image_bytes = data
    out.origin = origin
    out.redaction_effective = policy.effective
    out.rgb = rgb
    out.unobserved = unobserved
    out.mask_origin = mask_origin + "+nearblack"
    return out


def per_frame_sha1_digest(pairs) -> str:
    """SHA-1 over the sorted (keyframe id, stored-JPEG SHA-1) pairs."""
    h = hashlib.sha1()
    for kid, sha in sorted(pairs):
        h.update(f"{kid}\t{sha}\n".encode())
    return h.hexdigest()


def opaque_keyframe_id(session_id: str, keyframe_id: str) -> str:
    return hashlib.sha256(f"{session_id}:{keyframe_id}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# proxy depth
# ---------------------------------------------------------------------------


class ProxyCaster:
    """Nearest-hit depth of the proxy mesh from a pinhole camera, by ray casting
    (Embree through Open3D, on the CPU: ~18 ms for a 359x639 view of a
    229k-face mesh on the Tower). Depth is camera z, not ray length."""

    def __init__(self, V, F):
        try:
            import open3d as o3d  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment
            raise AppearanceUnavailable(f"open3d is not installed ({exc})") from None
        self._o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        if len(F):
            self.scene.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)),
                                     o3d.core.Tensor(np.asarray(F, np.uint32)))
        self.empty = not len(F)
        self._dirs: dict = {}

    def depth(self, R, t, K, width: int, height: int) -> np.ndarray:
        if self.empty:
            return np.full((height, width), np.inf, np.float32)
        key = (width, height, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        d = self._dirs.get(key)
        if d is None:
            ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
            d = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1],
                          np.ones_like(xs)], -1).reshape(-1, 3).astype(np.float32)
            self._dirs = {key: d}
        R = np.asarray(R, np.float32)
        c = (-R.T @ np.asarray(t, np.float32)).astype(np.float32)
        rays = np.empty((len(d), 6), np.float32)
        rays[:, :3] = c
        rays[:, 3:] = d @ R  # R^T d, per row; z-component 1 in the camera, so t_hit = z
        hit = self.scene.cast_rays(self._o3d.core.Tensor(rays))["t_hit"].numpy()
        return hit.reshape(height, width).astype(np.float32)


# ---------------------------------------------------------------------------
# occluders
# ---------------------------------------------------------------------------


def occluder_mask(zs: np.ndarray, zp: np.ndarray, unobserved: np.ndarray,
                  params: AppearanceParams):
    """Where the keyframe's own depth is much nearer than the proxy: the
    wearer's hand, their phone, or anything else the proxy does not contain.
    Contract §5.3. Returns (mask, record); record is None when the frame had
    too little overlap with the proxy to measure a threshold."""
    import cv2  # noqa: PLC0415

    zs = np.asarray(zs, np.float32)
    ok = (np.isfinite(zs) & (zs > 0) & np.isfinite(zp) & (zp > 0) & ~unobserved)
    if int(ok.sum()) < 64:
        return np.zeros(zp.shape, bool), None
    r = np.full(zp.shape, np.nan, np.float32)
    r[ok] = zs[ok] / zp[ok]
    rv = r[ok]
    s = float(np.median(rv))
    if not (s > 0):
        return np.zeros(zp.shape, bool), None
    mad = float(np.median(np.abs(rv - s)) / s)
    frac = min(params.occluder_ratio_max,
               max(params.occluder_ratio_min, 1.0 - params.occluder_mad_multiple * mad))
    tau = s * frac
    raw = ok & (r < tau)
    mask = raw
    if params.occluder_open_px > 1 and raw.any():
        k = np.ones((params.occluder_open_px, params.occluder_open_px), np.uint8)
        mask = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    if mask.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        min_area = params.occluder_min_area_frac * mask.size
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        mask = keep[labels]
    mask = dilate(mask, params.occluder_dilate_px) & np.isfinite(zp)
    return mask, {"median_ratio": round(s, 4), "mad": round(mad, 4),
                  "threshold": round(tau, 4)}


def transient_votes(rgbs, opaque, zps, gains, Rs, ts, K, params: AppearanceParams, device=None,
                    should_stop=None, slopes=None, vignette=None):
    """Where a keyframe shows something most OTHER keyframes did not see there.

    The depth test above removes what is nearer than the proxy. It cannot
    remove the wearer's hand and phone lying ON the desk: measured on the
    canonical capture, the proxy point under the hand in ki 330 is 1.90 from
    the camera and is confirmed, within 5%, by 146 other keyframes' own depth.
    Geometry cannot tell a hand from the desk it rests on. Colour can: those
    146 keyframes saw desk there.

    For a grid of each keyframe's proxy points, the colour (7x7 box, divided by
    the keyframe's exposure model: its gain, and with `slopes` / `vignette` the
    spatial field of §5.4) is sampled in every keyframe where the point
    is the nearest proxy surface, on an opaque texel, and away from the border.
    The reference is the per-channel median over the OTHER keyframes. A point
    is transient in this keyframe when at least `transient_min_views` others
    saw it, at least `transient_consensus` of them agree with the median, and
    this keyframe does not: its largest channel difference exceeds both
    `transient_abs` and `transient_rel` times the median's luma.

    A keyframe disagreeing on more than `transient_frame_max` of its tested
    points is misregistered rather than full of transients; its grid is left
    empty and its record says so.

    Returns one (coarse bool grid, record or None) per keyframe, or None if
    stopped.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as TF  # noqa: PLC0415

    dev = _torch_device(device)
    N = len(zps)
    if N < 2:
        return [(None, None)] * N
    H, W = np.asarray(zps[0]).shape
    half = torch.float16 if dev.type == "cuda" else torch.float32
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)
    R = torch.as_tensor(np.asarray(Rs), dtype=torch.float32, device=dev)
    t = torch.as_tensor(np.asarray(ts), dtype=torch.float32, device=dev)
    g = torch.as_tensor(np.asarray(gains, np.float32), device=dev)
    # A wide box: a hand is tens of pixels across, and a one-pixel
    # misregistration along a shelf edge must not read as a colour change.
    b = params.transient_box_px
    rgb = torch.empty((N, 3, H, W), dtype=half, device=dev)
    raw_ok = torch.empty((N, 1, H, W), dtype=half, device=dev)
    for i in range(N):
        x = torch.as_tensor(np.ascontiguousarray(rgbs[i]), device=dev).permute(2, 0, 1)[None].float() / 255
        blur = TF.avg_pool2d(x, b, stride=1, padding=b // 2, count_include_pad=False)[0]
        # Clipped or near-black samples say nothing about what was there.
        raw_ok[i, 0] = ((blur.amax(0) < params.transient_saturated)
                        & (blur.amax(0) > params.transient_dark)).to(half)
        div = g[i][:, None, None]
        if slopes is not None or vignette is not None:
            field = exposure_field(None if slopes is None else slopes[i], vignette, W, H)
            div = div * torch.as_tensor(field, device=dev)[None]
        rgb[i] = (blur / div.clamp(min=1e-3)).to(half)
    zp = torch.stack([torch.as_tensor(np.asarray(z, np.float32)) for z in zps]).to(dev, half)[:, None]
    op = torch.stack([torch.as_tensor(np.asarray(o, bool)) for o in opaque]).to(dev, half)[:, None]
    op = op * raw_ok
    del raw_ok
    step = params.transient_grid_px
    ys, xs = torch.meshgrid(torch.arange(step // 2, H, step, device=dev),
                            torch.arange(step // 2, W, step, device=dev), indexing="ij")
    gh, gw = ys.shape
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    bd = params.exposure_border_px
    group = 48
    out = []
    for a in range(N):
        if should_stop is not None and should_stop():
            return None
        z = zp[a, 0, ys, xs].float()
        ok = torch.isfinite(z) & (z > 0) & (op[a, 0, ys, xs] > 0.5)
        grid_out = torch.zeros(gh * gw, dtype=torch.bool, device=dev)
        if not bool(ok.any()):
            out.append((grid_out.reshape(gh, gw).cpu().numpy(), None))
            continue
        idx = torch.nonzero(ok, as_tuple=True)[0]
        own = rgb[a, :, ys[idx], xs[idx]].float().T                     # (P, 3)
        x, y, z0 = xs[idx].float() + 0.5, ys[idx].float() + 0.5, z[idx]
        xc = torch.stack([(x - Kt[0, 2]) / Kt[0, 0] * z0, (y - Kt[1, 2]) / Kt[1, 1] * z0, z0], 1)
        X = (xc - t[a]) @ R[a]
        cols = []
        for c0 in range(0, N, group):
            cs = torch.arange(c0, min(N, c0 + group), device=dev)
            cs = cs[cs != a]
            if not len(cs):
                continue
            pc = torch.einsum("sij,pj->spi", R[cs], X) + t[cs][:, None]
            zb = pc[..., 2]
            u = Kt[0, 0] * pc[..., 0] / zb.clamp(min=1e-6) + Kt[0, 2]
            v = Kt[1, 1] * pc[..., 1] / zb.clamp(min=1e-6) + Kt[1, 2]
            ins = (zb > 1e-3) & (u > bd) & (u < W - bd) & (v > bd) & (v < H - bd)
            gr = torch.stack([u / W * 2 - 1, v / H * 2 - 1], -1).unsqueeze(2).to(half)
            zpb = TF.grid_sample(zp[cs], gr, mode="nearest", align_corners=False)[:, 0, :, 0].float()
            opb = TF.grid_sample(op[cs], gr, mode="nearest", align_corners=False)[:, 0, :, 0]
            col = TF.grid_sample(rgb[cs], gr, mode="bilinear",
                                 align_corners=False)[..., 0].permute(0, 2, 1).float()
            vis = (ins & torch.isfinite(zpb) & ((zb - zpb).abs() < params.transient_vis_tol * zb)
                   & (opb > 0.5))
            cols.append(torch.where(vis[..., None], col, torch.full_like(col, float("nan"))))
        C = torch.cat(cols, 0)                                           # (views, P, 3)
        n = torch.isfinite(C[..., 0]).sum(0)
        med = torch.nanmedian(C, dim=0).values                           # (P, 3)
        luma = med.mean(1).clamp(min=0.02)
        tol = torch.maximum(torch.full_like(luma, params.transient_abs), params.transient_rel * luma)
        dev_views = (C - med[None]).abs().amax(2)                        # (views, P)
        agree = ((dev_views <= tol[None]) & torch.isfinite(dev_views)).sum(0)
        # One scale per keyframe first: a keyframe whose exposure gain is off
        # differs from every reference everywhere, and that is not a transient.
        have = (n >= params.transient_min_views) & torch.isfinite(med[:, 0])
        scale = torch.tensor(1.0, device=dev)
        if bool(have.any()):
            ratio = own[have].mean(1).clamp(min=1e-3) / med[have].mean(1).clamp(min=1e-3)
            scale = ratio.median().clamp(min=0.25, max=4.0)
        # THE KEYFRAME'S OWN COLOUR IS TAKEN AT THE BEST NEARBY OFFSET. A
        # keyframe misregistered by a pixel or two disagrees with the rest of
        # the walk in strips along every high-contrast edge (measured on the
        # canonical capture: frames 223, 227 and 255 lost their shelf boards in
        # stripes). Along an edge some offset within `transient_shift_px`
        # matches the reference; inside a hand or a phone, none does.
        sh = int(params.transient_shift_px)
        mine = None
        for dy in ((-sh, 0, sh) if sh > 0 else (0,)):
            for dx in ((-sh, 0, sh) if sh > 0 else (0,)):
                yy = (ys[idx] + dy).clamp(0, H - 1)
                xx = (xs[idx] + dx).clamp(0, W - 1)
                d = (rgb[a, :, yy, xx].float().T / scale - med).abs().amax(1)
                d = torch.where(op[a, 0, yy, xx] > 0.5, d, torch.full_like(d, float("inf")))
                mine = d if mine is None else torch.minimum(mine, d)
        hit = ((n >= params.transient_min_views)
               & (agree.float() >= params.transient_consensus * n.float().clamp(min=1))
               & (mine > tol) & torch.isfinite(mine))
        tested = int(have.sum())
        frac = float(hit.sum()) / tested if tested else 0.0
        # A keyframe that disagrees with the rest of the walk nearly everywhere
        # is misregistered (a pose or proxy error), not full of transients:
        # masking most of it would be noise. It is left unmasked and reported,
        # and the caller weights it down instead.
        applied = frac <= params.transient_frame_max
        if applied:
            grid_out[idx] = hit
        out.append((grid_out.reshape(gh, gw).cpu().numpy(),
                    {"tested": tested, "disagreeing_fraction": round(frac, 4),
                     "applied": applied}))
    del zp, op, rgb
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def transient_mask(grid, shape, zp: np.ndarray, params: AppearanceParams):
    """A keyframe's transient grid, lifted to full resolution: optionally
    opened with a `transient_open_cells` square (off by default: the hits inside
    a hand are sparse, and an opening removed most of the hands it was meant to
    keep), a cell with no hit neighbour dropped, each surviving hit covering its
    grid cell, components under the minimum area dropped, and the result
    dilated. Edges of a misregistered keyframe are kept out by the shifted
    comparison in `transient_votes`, not here."""
    import cv2  # noqa: PLC0415

    H, W = shape
    if grid is None or not grid.any():
        return np.zeros((H, W), bool)
    step = params.transient_grid_px
    g = grid.astype(np.uint8)
    k = max(1, int(params.transient_open_cells))
    if k > 1:
        g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((k, k), np.uint8),
                             borderType=cv2.BORDER_REPLICATE)
    # A lone cell is one sample's vote; keep cells with a hit neighbour.
    nb = cv2.filter2D(g, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    g = (g.astype(bool) & (nb >= 2)).astype(np.uint8)
    full = np.kron(g, np.ones((step, step), np.uint8))
    mask = np.zeros((H, W), bool)
    hh, ww = min(H, full.shape[0]), min(W, full.shape[1])
    mask[:hh, :ww] = full[:hh, :ww].astype(bool)
    if mask.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= params.occluder_min_area_frac * mask.size
        mask = keep[labels]
    return dilate(mask, params.occluder_dilate_px) & np.isfinite(zp)


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------


def sharpness(rgb: np.ndarray, transparent: np.ndarray) -> float:
    """Variance of the Laplacian over opaque pixels away from the mask edge."""
    import cv2  # noqa: PLC0415

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(grey, cv2.CV_32F)
    ok = ~dilate(transparent, 2)
    ok[:2, :] = ok[-2:, :] = False
    ok[:, :2] = ok[:, -2:] = False
    return float(lap[ok].var()) if int(ok.sum()) > 64 else 0.0


# ---------------------------------------------------------------------------
# exposure
# ---------------------------------------------------------------------------


def _torch_device(device=None):
    import torch  # noqa: PLC0415

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


EXPOSURE_MODEL_GAIN = "gain"
EXPOSURE_MODEL_SPATIAL = "gain+slope+vignette"
EXPOSURE_MODELS = (EXPOSURE_MODEL_GAIN, EXPOSURE_MODEL_SPATIAL)
# The image coordinates the spatial terms are written in (contract §5.4).
EXPOSURE_COORDINATES = ("xn=(u-W/2)/(W/2), yn=(v-H/2)/(H/2), "
                        "r2=((u-W/2)^2+(v-H/2)^2)/((W/2)^2+(H/2)^2); u,v pixel centres")


def exposure_coordinates(u, v, width: int, height: int):
    """(xn, yn, r2) of pixel positions `u`, `v` (numpy or torch, any shape)."""
    hw, hh = width / 2.0, height / 2.0
    du, dv = u - hw, v - hh
    return du / hw, dv / hh, (du * du + dv * dv) / (hw * hw + hh * hh)


def exposure_field(slope, vignette, width: int, height: int) -> np.ndarray:
    """The spatial factor of §5.4 over a whole image, (H, W) float32:
    `exp(slope_x * xn + slope_y * yn + k1 * r2 + k2 * r2^2)`. The recorded value
    at a pixel is `gain * field * radiance`."""
    vv, uu = np.mgrid[0:height, 0:width].astype(np.float32)
    xn, yn, r2 = exposure_coordinates(uu + 0.5, vv + 0.5, width, height)
    sx, sy = (float(slope[0]), float(slope[1])) if slope is not None else (0.0, 0.0)
    k1, k2 = (float(vignette[0]), float(vignette[1])) if vignette is not None else (0.0, 0.0)
    return np.exp(sx * xn + sy * yn + k1 * r2 + k2 * r2 * r2).astype(np.float32)


def _solve_spatial_exposure(L, P, S, XN, YN, R2, la, lg, npts, N, params, huber, dev):
    """Huber-IRLS over the joint weighted least squares of §5.4 -- albedos,
    per-keyframe (gain, tilt) and the shared falloff together -- each
    reweighting solved by Jacobi-preconditioned CGLS, warm-started from the
    gain model. Ridges: `exposure_slope_ridge` x a keyframe's weight on its tilt,
    `exposure_vignette_ridge` x the total weight on the falloff. Gauge: mean log
    gain 0 over keyframes with observations."""
    import torch  # noqa: PLC0415

    R4 = R2 * R2
    sl = torch.zeros((N, 2), device=dev)
    vg = torch.zeros(2, device=dev)
    outer = max(1, int(params.exposure_cg_outer))
    inner = max(1, int(params.exposure_cg_iterations))
    for _ in range(outer):
        pred = la[P] + lg[S] + (sl[S, 0] * XN + sl[S, 1] * YN + vg[0] * R2 + vg[1] * R4)[:, None]
        w = huber(L - pred)
        sw = w.sqrt()
        wsum = w.sum(1)
        fw = torch.zeros(N, device=dev).index_add_(0, S, wsum)
        lam_s = params.exposure_slope_ridge * fw                      # (N,)
        lam_v = params.exposure_vignette_ridge * wsum.sum()
        # column norms (diag of AtA) -> Jacobi scaling
        d_la = torch.zeros((npts, 3), device=dev).index_add_(0, P, w)
        d_lg = torch.zeros((N, 3), device=dev).index_add_(0, S, w)
        d_sl = torch.stack([torch.zeros(N, device=dev).index_add_(0, S, wsum * XN * XN),
                            torch.zeros(N, device=dev).index_add_(0, S, wsum * YN * YN)], 1) + lam_s[:, None]
        d_vg = torch.stack([(wsum * R2 * R2).sum(), (wsum * R4 * R4).sum()]) + lam_v
        D = [1 / d.clamp(min=1e-9).sqrt() for d in (d_la, d_lg, d_sl, d_vg)]
        sq_s, sq_v = lam_s.sqrt(), lam_v.sqrt()

        def A(x):                       # scaled unknowns -> weighted residual rows
            xa, xg, xs, xv = (x[0] * D[0], x[1] * D[1], x[2] * D[2], x[3] * D[3])
            obs = sw * (xa[P] + xg[S] + (xs[S, 0] * XN + xs[S, 1] * YN + xv[0] * R2 + xv[1] * R4)[:, None])
            return [obs, sq_s[:, None] * xs, sq_v * xv]

        def At(y):
            u = sw * y[0]
            us = u.sum(1)
            ga = torch.zeros((npts, 3), device=dev).index_add_(0, P, u)
            gg = torch.zeros((N, 3), device=dev).index_add_(0, S, u)
            gs = torch.stack([torch.zeros(N, device=dev).index_add_(0, S, us * XN),
                              torch.zeros(N, device=dev).index_add_(0, S, us * YN)], 1) + sq_s[:, None] * y[1]
            gv = torch.stack([(us * R2).sum(), (us * R4).sum()]) + sq_v * y[2]
            return [ga * D[0], gg * D[1], gs * D[2], gv * D[3]]

        def dot(a, b):
            return sum((x * y).sum() for x, y in zip(a, b))

        # residual of the current solution; CGLS on the correction
        r = [sw * (L - pred), -sq_s[:, None] * sl, -sq_v * vg]
        x = [torch.zeros_like(la), torch.zeros_like(lg), torch.zeros_like(sl), torch.zeros_like(vg)]
        sv = At(r)
        p = [t.clone() for t in sv]
        gamma = dot(sv, sv)
        for _it in range(inner):
            q = A(p)
            qq = dot(q, q)
            if float(qq) <= 1e-20:
                break
            alpha = gamma / qq
            x = [xi + alpha * pi for xi, pi in zip(x, p)]
            r = [ri - alpha * qi for ri, qi in zip(r, q)]
            sv = At(r)
            gnew = dot(sv, sv)
            if float(gnew) <= 1e-12 * float(gamma) + 1e-30:
                gamma = gnew
                break
            p = [si + (gnew / gamma) * pi for si, pi in zip(sv, p)]
            gamma = gnew
        la = la + x[0] * D[0]
        lg = lg + x[1] * D[1]
        sl = sl + x[2] * D[2]
        vg = vg + x[3] * D[3]
        seen = d_lg[:, 0] > 0
        if bool(seen.any()):
            shift = lg[seen].mean(0, keepdim=True)
            lg[seen] = lg[seen] - shift
            la = la + shift
    return la, lg, sl, vg


def solve_gains(rgbs, opaque, zps, Rs, ts, K, params: AppearanceParams, device=None,
                should_stop=None):
    """The per-keyframe photometric model (contract §5.4).

    `rgbs` (N,H,W,3) uint8, `opaque` (N,H,W) bool, `zps` (N,H,W) float proxy
    depth. Returns (gains (N,3) float32, observations per frame (N,), record).

    With `params.exposure_model == "gain+slope+vignette"` the recorded log
    colour of a surface point p seen by keyframe s at image position (xn, yn,
    r2) is modelled as

        log I = log g[s,c] + a[s]*xn + b[s]*yn + k1*r2 + k2*r2^2 + log A[p,c]

    -- a per-channel gain, a per-keyframe log-linear tilt (auto-exposure and
    off-axis falloff), and one radial falloff for the lens -- solved by
    alternating Huber-IRLS: point albedos as weighted means, then each
    keyframe's (g, a, b) as a ridge-regularised 5x5 weighted least squares,
    then (k1, k2). The record then also carries `slopes` (N,2), which the
    pipeline moves onto the keyframes, and `vignette`. With `gain` the model
    and the solve are the first build's, exactly.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as TF  # noqa: PLC0415

    model = params.exposure_model
    if model not in EXPOSURE_MODELS:
        raise ValueError(f"unknown exposure model {model!r}")
    spatial = model == EXPOSURE_MODEL_SPATIAL
    dev = _torch_device(device)
    N = len(rgbs)
    if N == 0:
        return np.ones((0, 3), np.float32), np.zeros(0, int), {"observations": 0, "model": model}
    H, W = rgbs[0].shape[:2]
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)
    R = torch.as_tensor(np.asarray(Rs), dtype=torch.float32, device=dev)
    t = torch.as_tensor(np.asarray(ts), dtype=torch.float32, device=dev)
    half = torch.float16 if dev.type == "cuda" else torch.float32
    b = params.exposure_box_px
    rgb = torch.empty((N, 3, H, W), dtype=half, device=dev)
    for i in range(N):
        x = torch.as_tensor(np.ascontiguousarray(rgbs[i]), device=dev).permute(2, 0, 1)[None].float() / 255
        rgb[i] = TF.avg_pool2d(x, b, stride=1, padding=b // 2, count_include_pad=False)[0].to(half)
    op = torch.as_tensor(np.asarray(opaque), device=dev)[:, None].to(half)
    zp = torch.as_tensor(np.asarray(zps), device=dev)[:, None].to(half)

    step = params.exposure_grid_px
    ys, xs = torch.meshgrid(torch.arange(step // 2, H, step, device=dev),
                            torch.arange(step // 2, W, step, device=dev), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    obs_p, obs_s, obs_c, obs_uv = [], [], [], []
    npts = 0
    group = 32
    bd = params.exposure_border_px
    for a in range(N):
        if should_stop is not None and should_stop():
            return None, None, None
        z = zp[a, 0, ys, xs].float()
        ok = torch.isfinite(z) & (op[a, 0, ys, xs] > 0.5)
        if not bool(ok.any()):
            continue
        x, y, z = xs[ok].float() + 0.5, ys[ok].float() + 0.5, z[ok]
        xc = torch.stack([(x - Kt[0, 2]) / Kt[0, 0] * z, (y - Kt[1, 2]) / Kt[1, 1] * z, z], 1)
        X = (xc - t[a]) @ R[a]
        n = X.shape[0]
        fp, fs, fc, fuv = [], [], [], []
        for c0 in range(0, N, group):
            cs = torch.arange(c0, min(N, c0 + group), device=dev)
            pc = torch.einsum("sij,pj->spi", R[cs], X) + t[cs][:, None]
            zz = pc[..., 2]
            u = Kt[0, 0] * pc[..., 0] / zz.clamp(min=1e-6) + Kt[0, 2]
            v = Kt[1, 1] * pc[..., 1] / zz.clamp(min=1e-6) + Kt[1, 2]
            ins = (zz > 1e-3) & (u > bd) & (u < W - bd) & (v > bd) & (v < H - bd)
            grid = torch.stack([u / W * 2 - 1, v / H * 2 - 1], -1).unsqueeze(2).to(half)
            zq = TF.grid_sample(zp[cs], grid, mode="nearest", align_corners=False)[:, 0, :, 0].float()
            oq = TF.grid_sample(op[cs], grid, mode="nearest", align_corners=False)[:, 0, :, 0]
            col = TF.grid_sample(rgb[cs], grid, mode="bilinear",
                                 align_corners=False)[..., 0].permute(0, 2, 1).float()
            good = (ins & ((zz - zq).abs() < params.exposure_vis_tol * zz) & (oq > 0.5)
                    & (col.amin(2) > params.exposure_dark) & (col.amax(2) < params.exposure_saturated))
            si, pi = torch.nonzero(good, as_tuple=True)
            fp.append((pi + npts).to(torch.int32))
            fs.append(cs[si].to(torch.int32))
            fc.append(col[si, pi].to(torch.float16))
            if spatial:
                fuv.append(torch.stack([u[si, pi], v[si, pi]], 1).to(torch.float16))
        # To the host once per source keyframe: 25M observations held on the
        # card while the stream grows peaked at 3.3 GB of a shared 12 GB GPU,
        # and a transfer per group cost more than the projection itself.
        obs_p.append(torch.cat(fp).cpu())
        obs_s.append(torch.cat(fs).cpu())
        obs_c.append(torch.cat(fc).cpu())
        if spatial:
            obs_uv.append(torch.cat(fuv).cpu())
        npts += n
    empty = {"observations": 0, "points": 0, "model": model}
    if not obs_p:
        return np.ones((N, 3), np.float32), np.zeros(N, int), empty
    P = torch.cat(obs_p)
    S = torch.cat(obs_s)
    C = torch.cat(obs_c)
    UV = torch.cat(obs_uv) if spatial else None
    if P.numel() == 0:
        return np.ones((N, 3), np.float32), np.zeros(N, int), empty
    cnt = torch.bincount(P.to(torch.int64), minlength=npts)
    keep = cnt[P] >= params.exposure_min_views
    P, S, C = P[keep], S[keep], C[keep]
    if spatial:
        UV = UV[keep]
    if P.numel() == 0:
        return np.ones((N, 3), np.float32), np.zeros(N, int), empty
    del rgb, op, zp, obs_p, obs_s, obs_c, obs_uv
    if P.numel() > params.exposure_max_observations:
        gen = torch.Generator(device="cpu").manual_seed(params.seed)
        pick = torch.randperm(P.numel(), generator=gen)[:params.exposure_max_observations]
        pick = pick.sort().values
        P, S, C = P[pick], S[pick], C[pick]
        if spatial:
            UV = UV[pick]
    P = P.to(dev, torch.int64)
    S = S.to(dev, torch.int64)
    C = C.to(dev, torch.float32)
    L = torch.log(C.clamp(min=1e-4))
    delta = params.exposure_huber

    def huber(res):
        return torch.where(res.abs() <= delta, torch.ones_like(res), delta / res.abs())

    lg = torch.zeros((N, 3), device=dev)
    sl = torch.zeros((N, 2), device=dev)
    vg = torch.zeros(2, device=dev)
    if spatial:
        UV = UV.to(dev, torch.float32)
        XN, YN, R2 = exposure_coordinates(UV[:, 0], UV[:, 1], W, H)
        del UV
    else:
        XN = YN = R2 = None

    def spatial_term():
        return sl[S, 0] * XN + sl[S, 1] * YN + vg[0] * R2 + vg[1] * R2 * R2

    la = None
    # The gain model's alternation. For the spatial model it is only the warm
    # start: alternating albedos against a lens falloff converges far too
    # slowly (measured on a synthetic falloff of -0.35: -0.12 after 30 rounds,
    # -0.33 after 300), so the joint problem is then solved outright below.
    for it in range(params.exposure_iterations if not spatial else 10):
        r0 = L - lg[S]
        w = torch.ones_like(L) if la is None else huber(r0 - la[P])
        acc = torch.zeros((npts, 3), device=dev).index_add_(0, P, w * r0)
        ws = torch.zeros((npts, 3), device=dev).index_add_(0, P, w)
        la = acc / ws.clamp(min=1e-9)
        r1 = L - la[P]
        w = huber(r1 - lg[S])
        g = torch.zeros((N, 3), device=dev).index_add_(0, S, w * r1)
        gw = torch.zeros((N, 3), device=dev).index_add_(0, S, w)
        lg = torch.where(gw > 0, g / gw.clamp(min=1e-9), torch.zeros_like(g))
        seen = gw[:, 0] > 0
        if bool(seen.any()):
            lg[seen] = lg[seen] - lg[seen].mean(0, keepdim=True)
    if spatial:
        la, lg, sl, vg = _solve_spatial_exposure(L, P, S, XN, YN, R2, la, lg, npts, N, params, huber, dev)

    la0 = torch.zeros((npts, 3), device=dev).index_add_(0, P, L)
    c0 = torch.zeros((npts, 3), device=dev).index_add_(0, P, torch.ones_like(L))
    before = (L - la0[P] / c0[P].clamp(min=1)).abs()
    after = (L - lg[S] - la[P] - (spatial_term()[:, None] if spatial else 0.0)).abs()
    per_frame = torch.bincount(S, minlength=N).cpu().numpy()
    gains = torch.exp(lg).cpu().numpy().astype(np.float32)
    gains[per_frame == 0] = 1.0
    slopes = sl.cpu().numpy().astype(np.float32)
    slopes[per_frame == 0] = 0.0
    sub = slice(None, None, max(1, before.numel() // 2_000_000))
    record = {
        "model": model,
        "observations": int(P.numel()),
        "points": int(torch.unique(P).numel()),
        "abs_log_residual_before": round(float(before.reshape(-1)[sub].median()), 4),
        "abs_log_residual_after": round(float(after.reshape(-1)[sub].median()), 4),
        "gain_range": [round(float(gains[per_frame > 0].mean(1).min()), 4),
                       round(float(gains[per_frame > 0].mean(1).max()), 4)]
        if (per_frame > 0).any() else None,
    }
    if spatial:
        centre, edge = R2 < 0.15, R2 > 0.45
        am = after.mean(1)
        record.update({
            "coordinates": EXPOSURE_COORDINATES,
            "vignette": [round(float(v), 5) for v in vg.cpu().numpy()],
            "slope_abs_median": round(float(np.median(np.abs(slopes[per_frame > 0])))
                                      if (per_frame > 0).any() else 0.0, 5),
            "abs_log_residual_after_centre": round(float(am[centre].median()), 4)
            if bool(centre.any()) else None,
            "abs_log_residual_after_edge": round(float(am[edge].median()), 4)
            if bool(edge.any()) else None,
            "slopes": slopes,
        })
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return gains, per_frame.astype(int), record


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def sample_proxy_points(V, F, n: int, seed: int = 0):
    """`n` area-weighted points on the mesh, with their face normals."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    if not len(F) or n <= 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = np.linalg.norm(cr, axis=1)
    if not (area.sum() > 0):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    rng = np.random.default_rng(seed)
    face = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    p = (1 - s)[:, None] * a[face] + (s * (1 - r2))[:, None] * b[face] + (s * r2)[:, None] * c[face]
    nrm = cr[face] / np.maximum(area[face], 1e-12)[:, None]
    return p.astype(np.float32), nrm.astype(np.float32)


def score_points(points, normals, zps, opaque, Rs, ts, K, quality, z_ref,
                 params: AppearanceParams, device=None):
    """(N, M) float32 selection scores (contract §5.5)."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as TF  # noqa: PLC0415

    dev = _torch_device(device)
    N, M = len(zps), len(points)
    if N == 0 or M == 0:
        return np.zeros((N, M), np.float32)
    H, W = np.asarray(zps[0]).shape
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)
    X = torch.as_tensor(points, dtype=torch.float32, device=dev)
    Nn = torch.as_tensor(normals, dtype=torch.float32, device=dev)
    out = np.zeros((N, M), np.float32)
    for i in range(N):
        R = torch.as_tensor(np.asarray(Rs[i]), dtype=torch.float32, device=dev)
        t = torch.as_tensor(np.asarray(ts[i]), dtype=torch.float32, device=dev)
        pc = X @ R.T + t
        z = pc[:, 2]
        u = Kt[0, 0] * pc[:, 0] / z.clamp(min=1e-6) + Kt[0, 2]
        v = Kt[1, 1] * pc[:, 1] / z.clamp(min=1e-6) + Kt[1, 2]
        ins = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        grid = torch.stack([u / W * 2 - 1, v / H * 2 - 1], -1)[None, :, None]
        zi = torch.as_tensor(np.asarray(zps[i], np.float32), device=dev)[None, None]
        oi = torch.as_tensor(np.asarray(opaque[i]), device=dev)[None, None].float()
        zq = TF.grid_sample(zi, grid, mode="nearest", align_corners=False)[0, 0, :, 0]
        oq = TF.grid_sample(oi, grid, mode="nearest", align_corners=False)[0, 0, :, 0]
        vis = ins & torch.isfinite(zq) & ((z - zq).abs() < params.selection_vis_tol * z) & (oq > 0.5)
        C = -R.T @ t
        ray = X - C
        ray = ray / ray.norm(dim=1, keepdim=True).clamp(min=1e-9)
        cos = (ray * Nn).sum(1).abs()
        s = cos * torch.clamp(z_ref / z.clamp(min=1e-6), max=1.0) * float(quality[i])
        out[i] = torch.where(vis, s, torch.zeros_like(s)).cpu().numpy()
    return out


def greedy_select(scores: np.ndarray, params: AppearanceParams, device=None):
    """Greedy maximisation of `sum best + w * sum second-best` up to the phone
    budget. Returns (order, marginal gains, objective). On the torch device:
    the numpy form took 19 s for 374 keyframes x 60k points."""
    import torch  # noqa: PLC0415

    dev = _torch_device(device)
    N, M = scores.shape
    S = torch.as_tensor(scores, dtype=torch.float32, device=dev)
    b1 = torch.zeros(M, device=dev)
    b2 = torch.zeros(M, device=dev)
    chosen: list[int] = []
    gains: list[float] = []
    total = 0.0
    remaining = torch.ones(N, dtype=torch.bool, device=dev)
    w2 = params.selection_second_weight
    while bool(remaining.any()) and len(chosen) < params.phone_budget:
        g = (torch.clamp(S - b1, min=0).sum(1)
             + w2 * torch.clamp(torch.minimum(b1, S) - b2, min=0).sum(1))
        g[~remaining] = -1.0
        best = int(torch.argmax(g))
        gb = float(g[best])
        if gb <= 0 or (total > 0 and gb < params.selection_min_gain_frac * total):
            break
        chosen.append(best)
        gains.append(gb)
        total += gb
        s = S[best]
        b2 = torch.maximum(b2, torch.minimum(b1, s))
        b1 = torch.maximum(b1, s)
        remaining[best] = False
    return chosen, gains, total


def coverage(scores: np.ndarray, rows) -> dict:
    rows = list(rows)
    if not rows or scores.shape[1] == 0:
        return {"seen1": 0.0, "seen2": 0.0}
    seen = (scores[rows] > 0).sum(0)
    return {"seen1": round(float((seen >= 1).mean()), 4),
            "seen2": round(float((seen >= 2).mean()), 4)}


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


def rgba_for(rgb: np.ndarray, transparent_core: np.ndarray, feather_px: int = 0) -> np.ndarray:
    """RGB zeroed inside the transparent core; alpha 0 over the core dilated by
    ALPHA_RING_PX (contract §5.6); beyond the ring, alpha rises smoothly to 255
    over `feather_px` pixels (Euclidean distance from the ring), so a masked
    patch's edge fades inward. The ring is untouched: no texel that is alpha 0
    without the feather gains alpha, and no texel gains alpha that the ring
    rule did not already give. The image border is not feathered here (the
    renderer feathers borders by its own rule)."""
    import cv2  # noqa: PLC0415

    rgba = np.empty(rgb.shape[:2] + (4,), np.uint8)
    rgba[..., :3] = rgb
    rgba[transparent_core, :3] = 0
    ring = dilate(transparent_core, ALPHA_RING_PX)
    if feather_px <= 0 or not ring.any():
        rgba[..., 3] = np.where(ring, 0, 255)
        return rgba
    # distance (px) of every texel outside the ring to the nearest ring texel
    d = cv2.distanceTransform((~ring).astype(np.uint8), cv2.DIST_L2, 5)
    x = np.clip((d - 1.0) / float(feather_px), 0.0, 1.0)
    alpha = np.rint(255.0 * x * x * (3.0 - 2.0 * x))
    alpha[ring] = 0
    rgba[..., 3] = alpha.astype(np.uint8)
    return rgba


_ASTC_CONTEXTS: dict = {}


def astc_available() -> tuple[bool, str | None]:
    try:
        import astc_encoder  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001
        return False, f"astc-encoder-py is not installed ({type(exc).__name__}: {exc})"
    return True, None


def encoder_versions(params: AppearanceParams) -> dict:
    import cv2  # noqa: PLC0415

    ok, _ = astc_available()
    astc_version = None
    if ok:
        import astc_encoder  # noqa: PLC0415

        astc_version = getattr(astc_encoder, "__version__", "unknown")
    return {
        ENC_ASTC: {"name": ENC_ASTC, "texel_format": "COMPRESSED_RGBA_ASTC_6x6_KHR",
                   "block": [ASTC_BLOCK, ASTC_BLOCK], "encoder": "astc-encoder-py",
                   "version": astc_version, "quality": params.astc_quality,
                   "profile": "ldr", "available": ok},
        ENC_WEBP: {"name": ENC_WEBP, "texel_format": "image/webp", "block": None,
                   "encoder": "opencv", "version": cv2.__version__,
                   "quality": params.webp_quality, "alpha": "lossless", "available": True},
    }


def encode_astc(rgba: np.ndarray, quality: float = ASTC_QUALITY, threads: int = 1) -> bytes:
    from astc_encoder import (  # noqa: PLC0415
        ASTCConfig,
        ASTCContext,
        ASTCImage,
        ASTCProfile,
        ASTCSwizzle,
        ASTCType,
    )

    key = (quality, threads)
    ctx = _ASTC_CONTEXTS.get(key)
    if ctx is None:
        ctx = ASTCContext(ASTCConfig(ASTCProfile.LDR, ASTC_BLOCK, ASTC_BLOCK, 1, float(quality)),
                          threads=threads)
        _ASTC_CONTEXTS[key] = ctx
    h, w = rgba.shape[:2]
    blocks = ctx.compress(ASTCImage(ASTCType.U8, w, h, 1, np.ascontiguousarray(rgba).tobytes()),
                          ASTCSwizzle.from_str("RGBA"))
    expected = astc_bytes(w, h)
    if len(blocks) != expected:
        raise AppearanceUnavailable(f"ASTC encoder returned {len(blocks)} bytes, expected {expected}")
    return bytes(blocks)


def decode_astc(blocks: bytes, width: int, height: int) -> np.ndarray:
    from astc_encoder import (  # noqa: PLC0415
        ASTCConfig,
        ASTCContext,
        ASTCImage,
        ASTCProfile,
        ASTCSwizzle,
        ASTCType,
    )

    ctx = ASTCContext(ASTCConfig(ASTCProfile.LDR, ASTC_BLOCK, ASTC_BLOCK, 1, ASTC_QUALITY))
    img = ASTCImage(ASTCType.U8, width, height, 1)
    ctx.decompress(blocks, img, ASTCSwizzle.from_str("RGBA"))
    return np.frombuffer(img.data, np.uint8).reshape(height, width, 4)


def astc_bytes(width: int, height: int) -> int:
    return -(-width // ASTC_BLOCK) * -(-height // ASTC_BLOCK) * 16


def encode_webp(rgba: np.ndarray, quality: int = WEBP_QUALITY) -> bytes:
    import cv2  # noqa: PLC0415

    # Every alpha-0 texel's RGB is zeroed before a LOSSY WebP encode. libwebp
    # (without `exact`, which OpenCV cannot set) rewrites the colour of
    # transparent blocks from their surroundings, and with a feathered alpha
    # (§5.6) it wrote scene colour (up to 159 of 255) into a zeroed redaction
    # fill of the synthetic world. Zero in, zero out; the page gives alpha 0 no
    # weight whatever its colour.
    rgba = rgba.copy()
    rgba[rgba[..., 3] == 0, :3] = 0
    ok, enc = cv2.imencode(".webp", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
                           [cv2.IMWRITE_WEBP_QUALITY, int(quality)])
    if not ok:
        raise AppearanceUnavailable("WebP encoding failed")
    return enc.tobytes()


def decode_webp(data: bytes) -> np.ndarray:
    import cv2  # noqa: PLC0415

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("not a colour WebP")
    if img.shape[2] == 3:
        # libwebp drops an alpha plane that is 255 everywhere: fully opaque.
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


_HEADER = struct.Struct("<8sIIIIII")


def pack_chunk(encoding: str, width: int, height: int, blobs: list[bytes]) -> bytes:
    """Contract §4.1."""
    n = len(blobs)
    head = _HEADER.pack(CHUNK_MAGIC, CHUNK_SCHEMA, ENCODING_CODES[encoding], n, width, height, 0)
    table, offset = [], 0
    for blob in blobs:
        table.append(struct.pack("<II", offset, len(blob)))
        offset += len(blob)
    return head + b"".join(table) + b"".join(blobs)


def read_chunk(buf: bytes) -> dict:
    """Parse a chunk, refusing anything whose lengths do not add up."""
    if len(buf) < _HEADER.size:
        raise ValueError("chunk shorter than its header")
    magic, schema, code, n, width, height, _ = _HEADER.unpack_from(buf, 0)
    if magic != CHUNK_MAGIC or schema != CHUNK_SCHEMA:
        raise ValueError("not an appearance chunk")
    names = {v: k for k, v in ENCODING_CODES.items()}
    if code not in names:
        raise ValueError(f"unknown chunk encoding {code}")
    base = _HEADER.size + 8 * n
    if len(buf) < base:
        raise ValueError("chunk shorter than its slot table")
    slots = []
    end = 0
    for i in range(n):
        off, ln = struct.unpack_from("<II", buf, _HEADER.size + 8 * i)
        slots.append(bytes(buf[base + off: base + off + ln]))
        end = max(end, off + ln)
        if base + off + ln > len(buf):
            raise ValueError("slot runs past the end of the chunk")
    if base + end != len(buf):
        raise ValueError("chunk length disagrees with its slot table")
    return {"encoding": names[code], "width": width, "height": height, "slots": slots}


@dataclass
class PreparedFrame:
    """A usable keyframe after every per-frame stage."""

    source: FrameSource
    R: np.ndarray
    t: np.ndarray
    zp: np.ndarray
    occluder: np.ndarray
    occluder_record: dict | None
    sharpness: float = 0.0
    quality: float = 1.0
    extra: dict = field(default_factory=dict)
    # The transient detector's mask (transients.py), or None when this frame
    # has none. Kept apart from `source.unobserved`, which is the privacy
    # mask: this one is a quality mask and may be absent; that one may not.
    detector: np.ndarray | None = None

    @property
    def transparent_core(self) -> np.ndarray:
        core = self.source.unobserved | self.occluder
        if self.detector is not None:
            core = core | self.detector
        return core
