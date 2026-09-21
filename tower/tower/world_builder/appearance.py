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

THE ONE EXCEPTION IS NAMED, OFF BY DEFAULT, AND LABELLED. `AppearanceParams.
imagery_source` is `redacted` unless a caller asks for `raw-local-research`:
the owner-sanctioned research bypass (2026-09-21) that builds from the
ORIGINAL local capture frames, so the campaign can measure what reconstruction
quality that imagery supports. Everything that bypass needs -- where the
original frames are, what the artifact is then called, and why it is not
privacy-safe -- lives in `raw_imagery.py`, not here. This module reaches it
through exactly one branch, in the provenance function, and a build that does
not ask for it behaves as it did before. An artifact built that way says so in
its manifest, its params digest, its served header and the page's caption, and
a reader that wants redacted appearance refuses it.

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

from tower.world_builder.raw_imagery import (
    IMAGERY_RAW,
    IMAGERY_REDACTED,
    IMAGERY_SOURCES,
    RAW_LABEL,
    RAW_UNOBSERVED_RULE,
    is_raw,
)

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
# The research bypass's origin. Deliberately not a `session-keyframe` spelling:
# a reader grepping for what these pixels are cannot mistake it for one.
ORIGIN_RAW_LOCAL = "raw-local-capture-frame"
MASK_STORED = "stored-fill"
MASK_RERUN = "rerun-difference+guess"
# No privacy mask at all, because there is no redaction to mask.
MASK_RAW_NONE = "none-raw-local-research"

REFUSED_IMAGE_MISSING = "refused-keyframe-image-missing"
REFUSED_UNDECODABLE = "refused-image-undecodable"
REFUSED_NO_FILL_MASK = "refused-no-fill-mask"
REFUSED_NO_RAW_FRAME = "refused-raw-frame-missing"
REFUSED_REDACTION_FAILED = "refused-redaction-failed"
REFUSED_CAMERA_MISMATCH = "refused-camera-mismatch"

# ---------------------------------------------------------------------------
# the unobserved mask
# ---------------------------------------------------------------------------

NEARBLACK_MAX = 6
NEARBLACK_OPEN = 16
UNOBSERVED_DILATE_PX = 2
UNOBSERVED_RULE = f"fill2|nearblack{NEARBLACK_MAX}open{NEARBLACK_OPEN}|dilate{UNOBSERVED_DILATE_PX}"

# ---------------------------------------------------------------------------
# the cross-frame redaction consensus
# ---------------------------------------------------------------------------

# A face is redacted per FRAME. The detector runs on each keyframe alone, so a
# face found in keyframe A and MISSED in keyframe B is filled in A and
# published by B -- and the page, blending by view, draws B's pixels at the
# very pose where A hid them. Measured on the canonical world by the fix-it
# cross-frame lane (`Glasses-scratch/wb-final-recon/fixit/privleak/PRIVLEAK.md`):
# 99.4% of the surface some keyframe's fill covered is published by another
# keyframe, and the one real face in that capture -- a printed portrait,
# detected and filled in 17 keyframes -- is published, recognisably, by 86
# others. This mask closes that in 3D: a surface point one keyframe hid is
# unobserved in EVERY keyframe.
CONSENSUS_OFF = "off"
CONSENSUS_PLAUSIBLE = "plausible"
CONSENSUS_UNION = "union"
CONSENSUS_MODES = (CONSENSUS_OFF, CONSENSUS_PLAUSIBLE, CONSENSUS_UNION)

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
    # WHICH IMAGERY THIS BUILD IS MADE OF (contract §6.6).
    #   `redacted`             the wearer's redacted keyframes. The product,
    #                          and the default everywhere: nothing reads the
    #                          environment to decide this for you.
    #   `raw-local-research`   the ORIGINAL local capture frames, no redaction
    #                          fill, no re-encode, no cross-frame consensus.
    #                          The owner-sanctioned research bypass; NOT
    #                          privacy-safe, labelled as such end to end, and
    #                          refused by a reader that wants the product.
    imagery_source: str = IMAGERY_REDACTED
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
    # the cross-frame redaction consensus (§5.3b)
    #   `union`      every fill region is unobserved for every keyframe. The
    #                guarantee, and on the canonical world it costs 93% of the
    #                published texels, because that capture's redaction is
    #                mostly wall-sized false positives.
    #   `plausible`  only a fill region that could be a face propagates: one
    #                that is at most `consensus_area_max` of its frame. 20 of
    #                20 eye-labelled printed-face regions pass; the 56-84%
    #                false positives do not. 14% of the published texels.
    #   `off`        per-frame redaction only -- what a build did before this
    #                mask existed. Never a default.
    redaction_consensus: str = CONSENSUS_PLAUSIBLE
    consensus_area_max: float = 0.10
    # A fill box straddles depth steps; only the surface at the region's own
    # depth is the surface the hidden thing was on.
    consensus_depth_tol: float = 0.20
    # The redactor dilates a face box by 1.6 about its centre (`HEAD_DILATION`),
    # so the detection is inside 62.5% of the fill. This erodes each region by
    # this fraction of its shorter side before projecting it, and the 3D
    # dilation below gives the tolerance back.
    consensus_erode: float = 0.20
    # A fill region this much inside the transient detector's hand/arm/phone
    # mask is the wearer's own hand, not a face, and does not propagate.
    consensus_hand_overlap: float = 0.5
    # The voxel side, in pixels at the median proxy depth. With
    # `consensus_dilate_cells` it is the whole tolerance of the rule: about
    # 6 px at the median depth here, which is where the cost stops falling
    # (PRIVLEAK.md: 17.3% of the published texels at 6 px a voxel, 14.6% at 3,
    # 13.6% at 2).
    consensus_voxel_px: float = 3.0
    consensus_dilate_cells: int = 1
    consensus_dilate_px: int = 2
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

    def __post_init__(self):
        if self.redaction_consensus not in CONSENSUS_MODES:
            raise ValueError(f"unknown redaction consensus mode "
                             f"{self.redaction_consensus!r}; one of {CONSENSUS_MODES}")
        if self.imagery_source not in IMAGERY_SOURCES:
            raise ValueError(f"unknown imagery source {self.imagery_source!r}; "
                             f"one of {IMAGERY_SOURCES}")
        if is_raw(self.imagery_source):
            # THE CONSENSUS IS A REDACTION RULE AND THERE IS NO REDACTION.
            # Left as the caller passed it, the params digest and the manifest
            # would both claim a cross-frame privacy guarantee this build did
            # not apply and could not apply. Forced here, once, so every
            # record downstream reads `off` because it IS off.
            object.__setattr__(self, "redaction_consensus", CONSENSUS_OFF)

    @classmethod
    def live(cls, **overrides) -> "AppearanceParams":
        """The walk-time preset: the same rules, less sampling. Being late is
        worse than being coarse, but a relaxed privacy or occluder rule is not
        coarse, it is wrong -- so neither the unobserved mask nor the
        cross-frame redaction consensus moves."""
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
    # The research bypass (`raw_imagery.py`). `redacted` is the product; the
    # other value means every field above describes a keyframe set this build
    # is NOT reading, and is kept only because it is still true of the stored
    # keyframes and a later re-enable will need it.
    imagery_source: str = IMAGERY_REDACTED
    # `raw_imagery.RawKeyframeImages` when, and only when, the bypass is on.
    raw_images: object | None = None

    @property
    def raw(self) -> bool:
        return is_raw(self.imagery_source)

    @property
    def effective(self) -> str:
        if self.raw:
            # NOT a redaction label, and not on any allowlist. Whatever reads
            # this -- the manifest, the served `X-World-Redaction` header, the
            # page's caption -- gets one string that cannot be misread as a
            # redaction having happened.
            return RAW_LABEL
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
    """THE trust decision for stored keyframe pixels, for every stage that
    reads them: this stage, the depth stage (`dense_pipeline.run_depth_stage`
    and `keyframe_image_bytes`) and, through their records, everything built on
    the depth stage. An exact allowlist; unknown never means trusted."""
    return isinstance(label, str) and label in TRUSTED_REDACTION_LABELS


def pixel_trust_token(label: str | None, redactor_label: str | None = None,
                      imagery_source: str = IMAGERY_REDACTED,
                      raw_token: str | None = None) -> str:
    """What a stage that read keyframe pixels did about the label, as one
    string for its cache keys and records: `trusted:<label>` when the stored
    bytes were used, `rerun:<label or none>&<redactor label>` when they were
    redacted again first. Two stages with equal tokens read equal pixels from
    equal stored bytes; a label change at Stop (`none` -> the real label)
    changes the token, so nothing cached under one is reused under the other
    (review 1, M3).

    The research bypass gets a token of its own shape, naming the exact set of
    original frames. It shares no prefix with either redacted spelling, so a
    depth stage, a transient mask or an appearance build made under it can
    never be reused for a redacted build, or the other way round."""
    if is_raw(imagery_source):
        return f"{IMAGERY_RAW}:{raw_token}"
    if label_is_trusted(label):
        return f"trusted:{label}"
    return f"rerun:{label if isinstance(label, str) and label else 'none'}&{redactor_label}"


def resolve_label_policy(store, world_id: str, session_id: str,
                         redactor_factory=None,
                         imagery_source: str = IMAGERY_REDACTED) -> LabelPolicy:
    """Read the label ONCE and decide. Refuses the whole build when the label
    needs a re-redaction and no redactor can run: a layer silently missing
    most of the room is worse than a clear refusal.

    Under the research bypass no redactor is needed or wanted: the build reads
    the original frames, so it never asks whether the stored ones may be
    trusted. It still records what the stored label says, because that
    statement stays true of the keyframe set on disk and the artifact's
    provenance reports both."""
    image_set = store.keyframe_image_set(world_id, session_id)
    label = image_set.redaction if isinstance(image_set.redaction, str) and image_set.redaction else None
    if is_raw(imagery_source):
        from tower.world_builder.raw_imagery import resolve_raw_keyframes  # noqa: PLC0415

        try:
            raw_images = resolve_raw_keyframes(store, world_id, session_id)
        except Exception as exc:  # noqa: BLE001 -- a refusal, with the reason
            raise AppearanceUnavailable(str(exc)) from None
        return LabelPolicy(session_redaction=label, trusted=False, image_set=image_set,
                           imagery_source=IMAGERY_RAW, raw_images=raw_images)
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


def _raw_keyframe_source(out: "FrameSource", policy: LabelPolicy, keyframe_id: str,
                         undistorter: Undistorter, hash_only: bool) -> "FrameSource":
    """The research bypass's half of the provenance function (contract §6.6).

    Reached only from `keyframe_source`, only when the caller asked for
    `raw-local-research`. It reads the ORIGINAL local frame this keyframe came
    from, undistorts it with the very same maps the redacted path uses -- so
    the two builds differ in their pixels and in nothing else -- and applies
    NO privacy mask, because there is no redaction here to mask and a
    near-black rule would blank the room's genuinely dark corners.

    The transient detector's hand and phone mask is NOT touched by this: it is
    a quality mask, it is applied later by `PreparedFrame.transparent_core`,
    and it stays on.
    """
    import cv2  # noqa: PLC0415

    data = policy.raw_images.read(keyframe_id) if policy.raw_images is not None else None
    if data is None:
        out.refused = REFUSED_NO_RAW_FRAME
        return out
    out.source_sha1 = hashlib.sha1(data).hexdigest()
    if hash_only:
        return out
    out.image_sha1 = out.source_sha1  # nothing re-encodes these bytes
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        out.refused = REFUSED_UNDECODABLE
        return out
    maps = undistorter.maps(bgr.shape[1], bgr.shape[0])
    if maps is None:
        out.refused = REFUSED_CAMERA_MISMATCH
        return out
    rgb = cv2.cvtColor(undistorter.remap(bgr, maps), cv2.COLOR_BGR2RGB)
    out.image_bytes = data
    out.origin = ORIGIN_RAW_LOCAL
    out.redaction_effective = policy.effective
    out.rgb = rgb
    out.unobserved = np.zeros(rgb.shape[:2], bool)
    out.mask_origin = MASK_RAW_NONE
    return out


def keyframe_source(store, world_id: str, session_id: str, keyframe_id: str, ki: int, *,
                    policy: LabelPolicy, align_record: dict | None, depth_dir,
                    undistorter: Undistorter, hash_only: bool = False) -> FrameSource:
    """THE provenance function: the only place this stage reads keyframe pixels.

    Returns the bytes the pixels came from, the effective redaction label, the
    unobserved mask in the solve camera, and where that mask came from -- or a
    refusal. Contract §6.

    One branch, and one only, leaves the redacted path: the research bypass
    (§6.6, `raw_imagery.py`), which a build reaches by asking for it in its
    params. `policy.raw` is false for every build that did not.
    """
    import cv2  # noqa: PLC0415

    from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
        FILL_RULE,
        _fill_mask_for,
    )

    out = FrameSource(ki=int(ki), keyframe_id=keyframe_id)
    if policy.raw:
        return _raw_keyframe_source(out, policy, keyframe_id, undistorter, hash_only)
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
# the cross-frame redaction consensus (contract §5.3b)
# ---------------------------------------------------------------------------


def consensus_rule_id(params: "AppearanceParams") -> str:
    """The rule a consensus mask was made under, as one string: recorded in the
    manifest and part of the cache key, so a change rebuilds."""
    if params.redaction_consensus == CONSENSUS_OFF:
        return "off"
    gate = ("any" if params.redaction_consensus == CONSENSUS_UNION
            else f"area<={params.consensus_area_max:g}")
    hand = ("" if params.redaction_consensus == CONSENSUS_UNION
            else f"|nothand>{params.consensus_hand_overlap:g}")
    return (f"{params.redaction_consensus}|{gate}{hand}|erode{params.consensus_erode:g}"
            f"|depth{params.consensus_depth_tol:g}|vox{params.consensus_voxel_px:g}px"
            f"|cells{params.consensus_dilate_cells}|dil{params.consensus_dilate_px}px|v1")


def _dilate_keys(keys: np.ndarray, strides, cells: int) -> np.ndarray:
    """Every voxel within `cells` of one of `keys`, as sorted keys.

    Sparse on purpose: the tolerance is the whole rule, and a dense grid fine
    enough to state it in a few pixels does not fit (the canonical world at
    3 px a voxel is 500M cells). The grid is padded by more than `cells`, so
    an offset never crosses into another row.
    """
    if cells <= 0 or not len(keys):
        return np.unique(keys)
    r = range(-cells, cells + 1)
    offsets = np.array([i * strides[0] + j * strides[1] + k * strides[2]
                        for i in r for j in r for k in r], np.int64)
    out = keys
    for c0 in range(0, len(offsets), 9):
        out = np.unique(np.concatenate(
            [out] + [keys + o for o in offsets[c0:c0 + 9]]))
    return out


def consensus_regions(unobserved: np.ndarray, zp: np.ndarray, params: "AppearanceParams",
                      detector: np.ndarray | None = None):
    """Which parts of a keyframe's privacy mask propagate into 3D, as boolean
    masks in that keyframe -- one per connected region of the mask.

    `union` takes every region. `plausible` takes only a region that could be a
    face:

    - at most `consensus_area_max` of the frame. Measured on the canonical
      world: all 20 eye-labelled printed-face regions are at most 9.95% of
      their frame, and the regions that make the union unaffordable are
      56-84%. What this does NOT cover is a face close enough to fill more
      than a tenth of the frame -- PRIVLEAK.md §7;
    - not the wearer's own hand, arm or phone, which the transient detector
      recognises (`transients.py`, 97.7% hand/arm pixel recall). A redaction
      box over the wearer's own hand is a false positive by construction, and
      on the canonical world those boxes are most of what the area gate still
      lets through. When no detector mask exists the region propagates: a
      missing detector never excuses a privacy mask.

    Each region is eroded by `consensus_erode` of its shorter side (the
    redactor's head dilation is a deliberate over-reach) and trimmed to its own
    depth, because a fill box straddling a depth step would otherwise be
    projected onto surfaces the hidden thing never stood on.
    """
    import cv2  # noqa: PLC0415

    out = []
    if params.redaction_consensus == CONSENSUS_OFF or not unobserved.any():
        return out
    n, labels, stats, _ = cv2.connectedComponentsWithStats(unobserved.astype(np.uint8), 8)
    finite = np.isfinite(zp) & (zp > 0)
    plausible = params.redaction_consensus == CONSENSUS_PLAUSIBLE
    for c in range(1, n):
        area = int(stats[c, cv2.CC_STAT_AREA])
        frac = area / unobserved.size
        if plausible and frac > params.consensus_area_max:
            out.append((None, {"area_fraction": round(frac, 4), "propagated": False,
                               "reason": "larger than a face"}))
            continue
        m = labels == c
        if plausible and detector is not None and detector.shape == m.shape:
            inside = float((m & detector).sum()) / max(1, area)
            if inside > params.consensus_hand_overlap:
                out.append((None, {"area_fraction": round(frac, 4), "propagated": False,
                                   "reason": "the wearer's own hand",
                                   "inside_detector": round(inside, 3)}))
                continue
        if params.consensus_erode > 0:
            side = min(int(stats[c, cv2.CC_STAT_WIDTH]), int(stats[c, cv2.CC_STAT_HEIGHT]))
            r = int(round(params.consensus_erode * side / 2))
            if r > 0:
                k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
                eroded = cv2.erode(m.astype(np.uint8), k).astype(bool)
                # A region too thin to erode keeps the whole of itself: failing
                # open here would drop a small face's protection silently.
                m = eroded if eroded.any() else m
        m &= finite
        if not m.any():
            out.append((None, {"area_fraction": round(frac, 4), "propagated": False,
                               "reason": "no proxy surface behind it"}))
            continue
        if params.consensus_depth_tol > 0:
            zm = float(np.median(zp[m]))
            m &= (zp > zm / (1 + params.consensus_depth_tol)) & (zp < zm * (1 + params.consensus_depth_tol))
            if not m.any():
                out.append((None, {"area_fraction": round(frac, 4), "propagated": False,
                                   "reason": "no proxy surface behind it"}))
                continue
        out.append((m, {"area_fraction": round(frac, 4), "propagated": True,
                        "surface_pixels": int(m.sum())}))
    return out


class RedactionConsensus:
    """The surface points some keyframe's redaction fill covered.

    A voxel grid over the proxy's own extent, marked by back-projecting each
    keyframe's propagating fill regions along the proxy depth, dilated by
    `consensus_dilate_cells` for pose and depth error, and then read back in
    every keyframe. What it returns for a keyframe is a privacy mask exactly
    like the fill: alpha 0, no weight, no colour.

    The grid is the whole tolerance model, so it is stated in pixels: a voxel
    is `consensus_voxel_px` pixels wide at the median proxy depth, and the
    total tolerance is that plus `consensus_dilate_cells` of it. It is held as
    sorted voxel keys, not as an array, because the tolerance that matters is
    finer than a dense grid of this room would fit.
    """

    PAD = 8   # cells of empty space around the proxy, so a dilation offset
    #           never wraps into the next row of the key space

    def __init__(self, V, z_ref: float, K, params: "AppearanceParams"):
        V = np.asarray(V, np.float32)
        self.params = params
        self.voxel = max(float(params.consensus_voxel_px) * float(z_ref) / float(K[0, 0]),
                         1e-6)
        pad = (self.PAD + params.consensus_dilate_cells) * self.voxel
        self.lo = V.min(0) - pad
        self.dims = np.maximum(1, np.ceil((V.max(0) + pad - self.lo) / self.voxel)
                               ).astype(np.int64) + 1
        self.strides = np.array([self.dims[1] * self.dims[2], self.dims[2], 1], np.int64)
        self._parts: list = []
        self.keys = np.zeros(0, np.int64)
        self.marked = 0
        self.regions = 0
        self.frames = 0
        self._final = False

    def _index(self, points):
        q = np.floor((points - self.lo) / self.voxel).astype(np.int64)
        np.clip(q, 0, self.dims - 1, out=q)
        return q @ self.strides

    @staticmethod
    def _world(zp, mask, R, t, K):
        H, W = zp.shape
        ys, xs = np.nonzero(mask)
        z = zp[ys, xs].astype(np.float32)
        d = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1],
                      np.ones_like(z)], -1).astype(np.float32)
        R = np.asarray(R, np.float32)
        c = (-R.T @ np.asarray(t, np.float32)).astype(np.float32)
        return (d * z[:, None]) @ R + c[None]

    def add(self, unobserved, zp, R, t, K, detector=None) -> dict:
        """Mark what one keyframe hid. Returns its record."""
        if self._final:
            raise RuntimeError("the consensus was already finalised")
        rows = consensus_regions(unobserved, zp, self.params, detector)
        marked = 0
        for m, rec in rows:
            if m is None:
                continue
            P = self._world(zp, m, R, t, K)
            if not len(P):
                continue
            keys = np.unique(self._index(P))
            self._parts.append(keys)
            marked += len(P)
            rec["voxels"] = int(len(keys))
            self.regions += 1
        if rows:
            self.frames += 1
        return {"regions": [r for _m, r in rows], "surface_pixels": marked}

    def finalize(self) -> dict:
        keys = (np.unique(np.concatenate(self._parts)) if self._parts
                else np.zeros(0, np.int64))
        self._parts = []
        core = int(len(keys))
        self.keys = _dilate_keys(keys, self.strides, int(self.params.consensus_dilate_cells))
        self.marked = int(len(self.keys))
        self._final = True
        return {"voxel": round(self.voxel, 6), "grid": [int(d) for d in self.dims],
                "voxels_marked": core, "voxels_dilated": self.marked,
                "regions_propagated": self.regions, "frames_with_fill": self.frames}

    def query(self, zp, R, t, K) -> np.ndarray:
        """The pixels of one keyframe whose proxy surface point some keyframe
        redacted. Pixels with no proxy behind them are never masked: the page
        cannot draw a source pixel whose own depth is unknown."""
        H, W = zp.shape
        out = np.zeros((H, W), bool)
        if not self.marked:
            return out
        ok = np.isfinite(zp) & (zp > 0)
        if not ok.any():
            return out
        q = self._index(self._world(zp, ok, R, t, K))
        at = np.searchsorted(self.keys, q)
        np.clip(at, 0, len(self.keys) - 1, out=at)
        out[ok] = self.keys[at] == q
        return dilate(out, self.params.consensus_dilate_px)


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
    # What ANOTHER keyframe's redaction fill covers of the surface this frame
    # sees (§5.3b). A privacy mask like `source.unobserved`, kept apart from it
    # because it is a function of every frame, not of this one.
    consensus: np.ndarray | None = None

    @property
    def transparent_core(self) -> np.ndarray:
        core = self.source.unobserved | self.occluder
        if self.detector is not None:
            core = core | self.detector
        if self.consensus is not None:
            core = core | self.consensus
        return core
