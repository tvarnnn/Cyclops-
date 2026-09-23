"""E2 -- chronological continuity: bridge tracking-loss gaps with the frames the frontend refused.

The phase-1 finding (wb-coherence-run-2026-09-23, FORENSICS section 0/3/5) is
that the target world's islands are born at ``tracking_lost`` events where
the keyframe selector refused a run of blurred raw frames that were still
matchable, and that the global solver only ever sees keyframes. This module
is the experiment side of one fix: give the solver, and only the solver,
some of those refused frames.

Four pieces, each usable alone:

1. **Gap-frame selection** (`label_frames`, `loss_gaps`, `select_frames`,
   `GapFrameRetainer`). An exact replay of the live frontend labels every
   delivered frame (``coherence_eval.trace.replay_frontend``). A *loss gap*
   runs from the last accepted keyframe before a ``tracking_lost`` event to
   the next accepted keyframe; the frames it holds are the ones the rule may
   retain. `GapFrameRetainer` is the same rule written the way a frontend
   would run it -- one frame at a time, with a bounded buffer of the frames
   refused since the last accepted keyframe, committing them only once a
   loss fires -- and the tests pin it frame-for-frame to the batch form. It
   uses no knowledge beyond the gap itself and no annotation of any kind.

2. **Sources** (`SessionRedactor`, `stage_frames`). Frames are undistorted
   exactly like ``global_solve.prepare_images`` (the same
   ``_undistort_maps``, the same crop, JPEG q95). ``raw`` stages the capture
   frame; ``redacted`` first runs the face redactor configured the way the
   SESSION records it (``session.json`` ``redaction``), so the product form
   without raw capture can be measured: ``...@0.30+plausibility3`` is the
   current ``redaction.FaceRedactor``; plain ``...@0.30`` is the redactor as
   it was before the plausibility gate (commit f8eb842^: every raw YuNet
   detection at 2x upscale, dilated 1.6x, filled). The fill fraction is
   reported per frame because the older redactor also blacked out hands,
   phones and walls.

3. **Learned bridge** (`EloftrMatcher`, `precompute_eloftr`, `bridge_augment`,
   `make_augment_hook`). EfficientLoFTR (Apache-2.0, via transformers) is run
   on consecutive and skip-2 frame pairs of each gap chain; its keypoints are
   quantised per image, appended AFTER the SIFT keypoints, its matches added
   with offset indices, and each such pair is re-verified with COLMAP's own
   two-view geometry (``pycolmap.estimate_two_view_geometry``, default
   options as ``match_sequential`` uses them). A learned link is then
   written only if it passes the **consensus rule** (see `consensus`): its
   relative rotation must close a triangle with neighbouring chain links
   within a bound measured on the known-good control world, and its rotation
   must be physically possible for a head in the time between the frames.

4. **Cut accounting** (`cut_strengths`, `chain_bridging`, `model_cut_tracks`).
   Before/after numbers for every cut, from a database and from a model.

Nothing here writes into a world directory: every function takes explicit
output paths, and the database it edits is the working copy the driver
hands the augment hook.

Privacy: ``raw`` staged frames are unredacted first-person imagery, exactly
like the raw capture they come from. They stay local and feed nothing but
the solver; only keyframe poses and points are ever scored or published.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Names and constants.

POLICY_LOSS_GAPS_ALL = "loss_gaps_all"
POLICY_LOSS_GAPS_EVERY_K = "loss_gaps_every_k"
POLICY_ALL_REFUSED_EVERY_K = "all_refused_every_k"
POLICIES = (POLICY_LOSS_GAPS_ALL, POLICY_LOSS_GAPS_EVERY_K, POLICY_ALL_REFUSED_EVERY_K)
DEFAULT_EVERY_K = 2

KIND_KEYFRAME = "k"
KIND_GAP = "g"

# Frontend outcomes (keyframes.ACCEPT etc.), restated so this module can label
# a replay table without importing the selector.
OUTCOME_ACCEPT = "accept"
OUTCOME_LOST = "tracking_lost"
# Frames the frontend could not even decode are never retained.
UNUSABLE_REASONS = frozenset({"malformed_frame", "frame_size_changed"})

# global_solve.prepare_images writes JPEG q95; staged frames must be the same.
STAGE_JPEG_QUALITY = 95

# EfficientLoFTR as D1 probed it (research/D1/probe_rawbridge_eloftr.py):
# Apache-2.0 weights from the HF hub, portrait input close to isotropic for
# the ~359x639 undistorted frames, and the model card's default threshold.
ELOFTR_MODEL_ID = "zju-community/efficientloftr"
ELOFTR_SIZE = {"height": 608, "width": 352}
ELOFTR_THRESHOLD = 0.2

# Quantisation of learned keypoints, per image, in undistorted pixels. The
# same scene point seen by one image in two different pairs comes back at
# slightly different sub-pixel positions; snapping both to one cell lets
# COLMAP build a track of >= 3 views through that image (GLOMAP drops shorter
# tracks: GlobalMapperOptions.track_min_num_views_per_track = 3). A 2 px cell
# keeps the snap error (<= 1.41 px) well inside COLMAP's own 4 px two-view
# RANSAC bound (TwoViewGeometryOptions.ransac.max_error).
DEFAULT_QUANT_PX = 2.0

# The verified-pair rule every other coherence tool uses (trace.py): >= 15
# inliers (GLOMAP's min_num_matches) and a configuration that carries geometry.
MIN_VERIFIED_INLIERS = 15
UNVERIFIED_CONFIGS = frozenset({0, 1})  # UNDEFINED, DEGENERATE

# The physical bound on head rotation between two frames, the one the harness
# uses for "physically implausible steps" (METRICS.md: 300 deg/s * dt + 30 deg;
# a generic walking, head-turning human, not tuned on any world).
PHYS_ROT_DEG_PER_S = 300.0
PHYS_ROT_SLACK_DEG = 30.0

# Consensus: the triangle-closure bound on a learned link's rotation. It is
# NOT a free parameter. `measure_cycle_noise` measures it on the known-good
# control world (b2a75ab4) and the experiment passes the measured value in;
# this default is only what the functions use when called without one, and
# the report always records which value ran.
DEFAULT_CYCLE_BOUND_DEG = 5.0

COLMAP_PAIR_BASE = 2147483647


def staged_name(capture_seq: int, kind: str) -> str:
    """The file name a staged frame gets: lexical order == capture order.

    ``capture_seq`` is the frame's position in the session's capture chain
    (``trace.load_capture_frames`` ``frame_index``), which is monotonic across
    a reconnect; ``source_seq`` is not guaranteed to be.
    """
    if kind not in (KIND_KEYFRAME, KIND_GAP):
        raise ValueError(f"kind must be {KIND_KEYFRAME!r} or {KIND_GAP!r}, not {kind!r}")
    if int(capture_seq) < 0 or int(capture_seq) > 999_999:
        raise ValueError(f"capture_seq {capture_seq} does not fit six digits")
    return f"{int(capture_seq):06d}_{kind}.jpg"


def parse_staged_name(name: str) -> tuple[int, str] | None:
    """``000584_k.jpg`` -> (584, 'k'); anything else -> None."""
    stem = Path(str(name).replace("\\", "/")).name
    if not stem.endswith(".jpg") or len(stem) != len("000000_k.jpg") or stem[6] != "_":
        return None
    seq, kind = stem[:6], stem[7]
    if not seq.isdigit() or kind not in (KIND_KEYFRAME, KIND_GAP):
        return None
    return int(seq), kind


def pair_id(image_id1: int, image_id2: int) -> int:
    a, b = sorted((int(image_id1), int(image_id2)))
    return a * COLMAP_PAIR_BASE + b


# ---------------------------------------------------------------------------
# 1. Gap-frame selection.


def label_frames(capture_dirs) -> "pd.DataFrame":
    """Every delivered frame of a capture chain, labelled by the live frontend.

    ``trace.replay_frontend`` re-runs FrameTracker + KeyframeSelector exactly as
    ``engine.observe`` does (phase 1 verified it reproduces the journal's
    keyframes frame-for-frame). Columns: frame_index (= capture_seq),
    capture_id, source_seq, received_at, outcome, reason, sharpness, ..., path.
    """
    from tower.world_builder.coherence_eval.trace import load_capture_frames, replay_frontend

    frames = load_capture_frames([Path(c) for c in capture_dirs])
    labels = replay_frontend(frames)
    labels["path"] = frames["path"].to_numpy()
    return labels


@dataclass(frozen=True)
class LossGap:
    """One tracking-loss gap: the frames between two accepted keyframes that a
    ``tracking_lost`` separates. ``start_frame``/``end_frame`` are the accepted
    keyframes' capture_seq (``end_frame`` is None when the walk ended inside
    the gap); ``refused`` are the usable frames strictly between them."""

    gap_index: int
    start_frame: int | None
    end_frame: int | None
    lost_frames: tuple[int, ...]
    refused: tuple[int, ...]
    preloss_refused: int  # refused frames before the first loss: the frontend's buffer need

    def to_json(self) -> dict:
        return asdict(self)


def _is_refused(outcome: str, reason) -> bool:
    return outcome != OUTCOME_ACCEPT and reason not in UNUSABLE_REASONS


def loss_gaps(labels) -> list[LossGap]:
    """Tracking-loss gaps of a labelled frame table, in capture order.

    A gap is a maximal run of non-accepted frames between two accepted frames
    that contains at least one ``tracking_lost``. Several losses inside one
    run (the tracker is reset and loses again before anything is accepted)
    are one gap.
    """
    gaps: list[LossGap] = []
    last_accept = None
    run: list[tuple[int, str, object]] = []

    def close(end):
        lost = tuple(int(f) for f, o, _ in run if o == OUTCOME_LOST)
        if lost:
            refused = tuple(int(f) for f, o, r in run if _is_refused(o, r))
            first_lost = lost[0]
            pre = sum(1 for f in refused if f < first_lost)
            gaps.append(LossGap(len(gaps), last_accept, end, lost, refused, pre))

    for f, o, r in zip(labels["frame_index"], labels["outcome"], labels["reason"]):
        if o == OUTCOME_ACCEPT:
            close(int(f))
            run = []
            last_accept = int(f)
        else:
            run.append((int(f), o, r))
    close(None)
    return gaps


class GapFrameRetainer:
    """The retention rule as the frontend would run it, one frame at a time.

    ``observe(frame, outcome, reason)`` returns the frames committed for
    retention by that observation (possibly frames buffered earlier);
    ``finish()`` flushes nothing -- a gap still open at the end of the walk
    has already committed every frame after its loss, and the pre-loss buffer
    of a run that never lost is dropped, exactly as on an accept.

    State it needs, all local and bounded: the refused frames since the last
    accepted keyframe (at most ``max_buffer`` of them, oldest dropped first),
    whether a loss has fired since that keyframe, and a position counter.
    Nothing from the future, and no region or annotation, is consulted.

    Policies:
      - ``loss_gaps_all``: every refused frame of a loss gap;
      - ``loss_gaps_every_k``: refused frames at positions 0, k, 2k, ... of the
        gap (position counted from the first refused frame after the keyframe);
      - ``all_refused_every_k`` (reference upper bound, not a loss rule): every
        k-th refused frame of every run, loss or not.
    """

    def __init__(self, policy: str, k: int = DEFAULT_EVERY_K, max_buffer: int | None = None):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.policy = policy
        self.k = 1 if policy == POLICY_LOSS_GAPS_ALL else int(k)
        self.max_buffer = max_buffer
        self.buffer: list[int] = []
        self.in_loss = False
        self.position = 0
        self.peak_buffer = 0

    def _keep(self, position: int) -> bool:
        return position % self.k == 0

    def observe(self, frame, outcome: str, reason=None) -> list:
        if outcome == OUTCOME_ACCEPT:
            self.buffer = []
            self.in_loss = False
            self.position = 0
            return []
        if not _is_refused(outcome, reason):
            return []
        keep = self._keep(self.position)
        self.position += 1
        if self.policy == POLICY_ALL_REFUSED_EVERY_K:
            return [frame] if keep else []
        out: list = []
        if outcome == OUTCOME_LOST and not self.in_loss:
            self.in_loss = True
            out.extend(self.buffer)
            self.buffer = []
        if self.in_loss:
            if keep:
                out.append(frame)
            return out
        if keep:
            self.buffer.append(frame)
            if self.max_buffer is not None and len(self.buffer) > self.max_buffer:
                self.buffer.pop(0)
            self.peak_buffer = max(self.peak_buffer, len(self.buffer))
        return out

    def run(self, labels) -> list:
        out = []
        for f, o, r in zip(labels["frame_index"], labels["outcome"], labels["reason"]):
            out.extend(self.observe(int(f), o, r))
        return out


def select_frames(labels, policy: str, k: int = DEFAULT_EVERY_K) -> list[int]:
    """Batch form of `GapFrameRetainer` (unbounded buffer): capture_seq of every
    retained frame, in capture order."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    k = 1 if policy == POLICY_LOSS_GAPS_ALL else int(k)
    if policy == POLICY_ALL_REFUSED_EVERY_K:
        out = []
        position = 0
        for f, o, r in zip(labels["frame_index"], labels["outcome"], labels["reason"]):
            if o == OUTCOME_ACCEPT:
                position = 0
            elif _is_refused(o, r):
                if position % k == 0:
                    out.append(int(f))
                position += 1
        return out
    out = []
    for gap in loss_gaps(labels):
        out.extend(f for p, f in enumerate(gap.refused) if p % k == 0)
    return out


def selection_summary(labels, gaps: list[LossGap], selected: list[int], *, bytes_per_frame: float | None = None,
                      frame_sizes: dict | None = None) -> dict:
    """Counts, disk and frontend buffer need for one policy on one world."""
    sel = set(int(f) for f in selected)
    per_gap = [{"gap": g.gap_index, "start": g.start_frame, "end": g.end_frame, "refused": len(g.refused),
                "retained": sum(1 for f in g.refused if f in sel), "preloss_refused": g.preloss_refused,
                "losses": len(g.lost_frames)} for g in gaps]
    disk = None
    if frame_sizes:
        disk = int(sum(frame_sizes.get(int(f), 0) for f in sel))
    elif bytes_per_frame:
        disk = int(bytes_per_frame * len(sel))
    reasons = Counter(labels.loc[labels["frame_index"].isin(sel), "reason"])
    return {
        "delivered_frames": int(len(labels)),
        "accepted": int((labels["outcome"] == OUTCOME_ACCEPT).sum()),
        "tracking_lost_events": int((labels["outcome"] == OUTCOME_LOST).sum()),
        "loss_gaps": len(gaps),
        "retained": len(sel),
        "retained_by_reason": dict(reasons),
        "disk_bytes": disk,
        "max_preloss_buffer_frames": max((g.preloss_refused for g in gaps), default=0),
        "per_gap": per_gap,
    }


# ---------------------------------------------------------------------------
# 2. Sources: undistortion and the session's redactor.


def undistortion_for(session_intrinsics: dict, width: int, height: int):
    """(m1, m2, (x, y, w, h), PinholeCamera) exactly as prepare_images builds them."""
    from tower.world_builder.global_solve import _undistort_maps
    from tower.world_builder.records import camera_intrinsics_from_json_dict

    return _undistort_maps(camera_intrinsics_from_json_dict(session_intrinsics), width, height)


def undistort_image(image, maps):
    import cv2

    m1, m2, (x, y, rw, rh), _camera = maps
    return cv2.remap(image, m1, m2, cv2.INTER_LINEAR)[y:y + rh, x:x + rw]


class SessionRedactor:
    """The face redactor configured the way a session's ``redaction`` names it.

    - ``none``: identity;
    - ``faces-detected-and-filled/yunet-2023mar@0.30+plausibility3``: the
      current ``redaction.FaceRedactor``, unchanged;
    - ``faces-detected-and-filled/yunet-2023mar@0.30``: the redactor before
      the plausibility gate (f8eb842^): every YuNet detection at 2x upscale,
      no landmark test, no corroboration, dilated 1.6x about its centre, filled.

    `redact_bgr` returns (image, boxes, filled_fraction); the fraction is of
    the frame area, the union of the filled boxes.
    """

    LEGACY = "faces-detected-and-filled/yunet-2023mar@0.30"

    def __init__(self, label: str | None):
        from tower.world_builder import redaction

        self.label = label or redaction.REDACTION_NONE
        self._r = redaction
        self._current = None
        self._legacy = None
        if self.label == redaction.REDACTION_NONE:
            self.mode = "none"
        elif self.label.endswith("+" + redaction.PLAUSIBILITY_ID):
            self.mode = "current"
            self._current = redaction.FaceRedactor()
            if not self._current.available:
                raise RuntimeError(self._current.unavailable_reason)
            if self._current.label != self.label:
                raise RuntimeError(f"session redaction {self.label!r} is not this code's {self._current.label!r}")
        elif self.label == self.LEGACY:
            self.mode = "legacy"
            path = redaction.model_path()
            if path is None:
                raise RuntimeError("no YuNet model for the legacy redactor")
            self._legacy_path = path
        else:
            raise ValueError(f"unknown redaction configuration {self.label!r}")

    def _legacy_boxes(self, image) -> list:
        import cv2

        r = self._r
        h, w = image.shape[:2]
        scaled = cv2.resize(image, (w * r.UPSCALE, h * r.UPSCALE), interpolation=cv2.INTER_CUBIC)
        size = (scaled.shape[1], scaled.shape[0])
        if self._legacy is None:
            self._legacy = cv2.FaceDetectorYN.create(str(self._legacy_path), "", size, r.CONFIDENCE,
                                                     r.NMS_THRESHOLD, r.TOP_K)
            self._legacy_size = size
        elif size != self._legacy_size:
            self._legacy.setInputSize(size)
            self._legacy_size = size
        _, faces = self._legacy.detect(scaled)
        boxes = []
        for face in ([] if faces is None else faces):
            x, y, bw, bh = (float(v) / r.UPSCALE for v in face[:4])
            cx, cy = x + bw / 2.0, y + bh / 2.0
            bw *= r.HEAD_DILATION
            bh *= r.HEAD_DILATION
            boxes.append((cx - bw / 2.0, cy - bh / 2.0, bw, bh))
        return boxes

    def boxes(self, image) -> list:
        if self.mode == "none":
            return []
        if self.mode == "current":
            return list(self._current._detect(image))
        return self._legacy_boxes(image)

    def redact_bgr(self, image):
        """(redacted copy, boxes, filled fraction of the frame)."""
        out = image.copy()
        boxes = self.boxes(out)
        h, w = out.shape[:2]
        mask = np.zeros((h, w), bool)
        for x, y, bw, bh in boxes:
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
            if x1 > x0 and y1 > y0:
                out[y0:y1, x0:x1] = self._r.FILL_VALUE
                mask[y0:y1, x0:x1] = True
        return out, boxes, float(mask.mean())

    def redact_jpeg_roundtrip(self, image):
        """What the product persists: the redacted frame re-encoded at the
        redactor's JPEG quality (only when something was filled, as
        ``FaceRedactor._redact`` does), decoded again."""
        import cv2

        out, boxes, frac = self.redact_bgr(image)
        if boxes:
            ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), self._r.JPEG_QUALITY])
            if ok:
                out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return out, boxes, frac


@dataclass
class StagedFrame:
    """One solver-only frame, as P's ``VariantConfig.extra_images`` wants it."""

    name_key: str
    undistorted_path: str
    capture_seq: int
    source: str  # "raw" | "redacted"
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def stage_frames(frame_paths: dict, capture_seqs: Iterable[int], out_dir, *, session_intrinsics: dict,
                 width: int, height: int, source: str = "raw", redaction_label: str | None = None,
                 kind: str = KIND_GAP, overwrite: bool = False) -> tuple[list[StagedFrame], dict]:
    """Undistort (and for ``redacted``, first redact) frames into ``out_dir``.

    ``frame_paths``: capture_seq -> raw frame path. Returns the staged frames
    and a summary (count, bytes, seconds, redaction fill statistics).
    """
    import cv2

    if source not in ("raw", "redacted"):
        raise ValueError(f"source must be raw or redacted, not {source!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maps = undistortion_for(session_intrinsics, width, height)
    redactor = SessionRedactor(redaction_label) if source == "redacted" else None
    staged: list[StagedFrame] = []
    fills = []
    started = time.perf_counter()
    redact_s = 0.0
    total_bytes = 0
    skipped = []
    for seq in capture_seqs:
        seq = int(seq)
        target = out_dir / staged_name(seq, kind)
        meta = {"raw_path": str(frame_paths[seq])}
        if target.exists() and not overwrite and source == "raw":
            total_bytes += target.stat().st_size
            staged.append(StagedFrame(target.name, str(target), seq, source, meta))
            continue
        image = cv2.imread(str(frame_paths[seq]), cv2.IMREAD_COLOR)
        if image is None or image.shape[1] != width or image.shape[0] != height:
            skipped.append(seq)
            continue
        if redactor is not None:
            t0 = time.perf_counter()
            image, boxes, frac = redactor.redact_jpeg_roundtrip(image)
            redact_s += time.perf_counter() - t0
            meta.update(redaction=redactor.label, redaction_boxes=len(boxes), fill_fraction=frac)
            fills.append(frac)
        und = undistort_image(image, maps)
        cv2.imwrite(str(target), und, [cv2.IMWRITE_JPEG_QUALITY, STAGE_JPEG_QUALITY])
        total_bytes += target.stat().st_size
        staged.append(StagedFrame(target.name, str(target), seq, source, meta))
    fills_a = np.asarray(fills, float)
    summary = {
        "source": source, "kind": kind, "staged": len(staged), "skipped_unreadable": skipped,
        "bytes": int(total_bytes), "seconds": round(time.perf_counter() - started, 2),
        "redaction": redactor.label if redactor else None, "redact_seconds": round(redact_s, 2),
        "camera": maps[3].to_json_dict(),
    }
    if redactor is not None:
        summary.update(
            frames_with_fill=int((fills_a > 0).sum()),
            fill_fraction_mean=float(fills_a.mean()) if len(fills_a) else 0.0,
            fill_fraction_p90=float(np.percentile(fills_a, 90)) if len(fills_a) else 0.0,
            fill_fraction_max=float(fills_a.max()) if len(fills_a) else 0.0,
        )
    return staged, summary


# ---------------------------------------------------------------------------
# 3. The learned bridge.


class EloftrMatcher:
    """EfficientLoFTR through transformers (Apache-2.0 code and weights).

    Keypoints come back in the input image's pixel coordinates (OpenCV
    convention, pixel centres at integers). ``cache_dir`` is the HF cache
    (the run keeps it under research/D1/hf_cache); offline is fine.
    """

    def __init__(self, device: str = "cuda", cache_dir=None, size=None, threshold: float = ELOFTR_THRESHOLD):
        import torch
        from transformers import AutoImageProcessor, AutoModelForKeypointMatching

        kw = {"cache_dir": str(cache_dir)} if cache_dir else {}
        self.processor = AutoImageProcessor.from_pretrained(ELOFTR_MODEL_ID, **kw)
        self.model = AutoModelForKeypointMatching.from_pretrained(ELOFTR_MODEL_ID, **kw).to(device).eval()
        self.device = device
        self.size = dict(size or ELOFTR_SIZE)
        self.threshold = float(threshold)
        self._torch = torch

    def match(self, image_a_rgb, image_b_rgb):
        """(kp_a (N,2) float64, kp_b (N,2) float64, score (N,) float32)."""
        torch = self._torch
        inputs = self.processor([image_a_rgb, image_b_rgb], return_tensors="pt", size=self.size).to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        res = self.processor.post_process_keypoint_matching(
            out, [[image_a_rgb.shape[:2], image_b_rgb.shape[:2]]], threshold=self.threshold)[0]
        ka = res["keypoints0"].cpu().numpy().astype(np.float64)
        kb = res["keypoints1"].cpu().numpy().astype(np.float64)
        sc = res["matching_scores"].cpu().numpy().astype(np.float32)
        return ka, kb, sc


@dataclass(frozen=True)
class Chain:
    """The images of one gap in capture order: context keyframes, the keyframe
    before the gap, its retained gap frames, the keyframe after, context.
    ``names[first_gap_kf]`` and ``names[last_gap_kf]`` are the two keyframes
    whose link the gap broke."""

    gap_index: int
    names: tuple[str, ...]
    kf_before: int  # position in names
    kf_after: int   # position in names (== kf_before + retained + 1)
    capture_seqs: tuple[int, ...]
    received_at: tuple[float, ...]

    def pairs(self, max_skip: int = 2) -> list[tuple[int, int]]:
        n = len(self.names)
        return [(i, j) for i in range(n) for j in range(i + 1, min(n, i + max_skip + 1))]


def build_chains(gaps: list[LossGap], retained: Iterable[int], keyframe_seqs: list[int], received_at: dict,
                 *, context: int = 2) -> list[Chain]:
    """One chain per loss gap that has a keyframe on both sides.

    ``keyframe_seqs``: capture_seq of every accepted keyframe, in order.
    ``context`` keyframes on each side give the end links triangles to close.
    """
    kept = set(int(f) for f in retained)
    kpos = {s: i for i, s in enumerate(keyframe_seqs)}
    chains = []
    for g in gaps:
        if g.start_frame is None or g.end_frame is None or g.start_frame not in kpos or g.end_frame not in kpos:
            continue
        a, b = kpos[g.start_frame], kpos[g.end_frame]
        before = keyframe_seqs[max(0, a - context):a + 1]
        after = keyframe_seqs[b:b + context + 1]
        mids = [f for f in g.refused if f in kept]
        seqs = list(before) + mids + list(after)
        names = [staged_name(s, KIND_KEYFRAME) for s in before] + [staged_name(s, KIND_GAP) for s in mids] + \
                [staged_name(s, KIND_KEYFRAME) for s in after]
        chains.append(Chain(g.gap_index, tuple(names), len(before) - 1, len(before) + len(mids),
                            tuple(int(s) for s in seqs), tuple(float(received_at.get(s, np.nan)) for s in seqs)))
    return chains


def _read_rgb(path):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def precompute_eloftr(chains: list[Chain], images_dir, cache_path, *, matcher: EloftrMatcher | None = None,
                      max_skip: int = 2, device: str = "cuda", hf_cache=None, log=print) -> dict:
    """Run ELoFTR on every consecutive and skip-``max_skip`` pair of every chain
    and store the raw correspondences in one ``.npz`` (GPU; run it under the
    run's gpulock). The augment hook then needs no GPU. Returns a summary."""
    import torch

    images_dir = Path(images_dir)
    matcher = matcher or EloftrMatcher(device=device, cache_dir=hf_cache)
    done = {}
    pairs = []
    for c in chains:
        for i, j in c.pairs(max_skip):
            key = f"{c.names[i]}|{c.names[j]}"
            if key not in done:
                done[key] = None
                pairs.append((c.names[i], c.names[j]))
    images = {}

    def rgb(name):
        if name not in images:
            images[name] = _read_rgb(images_dir / name)
            if len(images) > 64:
                images.pop(next(iter(images)))
        return images[name]

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    arrays = {}
    for n, (a, b) in enumerate(pairs):
        ka, kb, sc = matcher.match(rgb(a), rgb(b))
        arrays[f"{a}|{b}"] = np.concatenate([ka, kb, sc[:, None].astype(np.float64)], axis=1).astype(np.float32)
        if log and (n + 1) % 200 == 0:
            log(f"eloftr {n + 1}/{len(pairs)} pairs, {time.perf_counter() - t0:.0f}s")
    dt = time.perf_counter() - t0
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{k.replace("|", "__"): v for k, v in arrays.items()})
    summary = {"pairs": len(pairs), "seconds": round(dt, 1), "ms_per_pair": round(1000 * dt / max(len(pairs), 1), 1),
               "peak_vram_mib": (round(torch.cuda.max_memory_allocated() / 2 ** 20)
                                 if device.startswith("cuda") else None),
               "model": ELOFTR_MODEL_ID, "size": ELOFTR_SIZE, "threshold": matcher.threshold}
    cache_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_eloftr_cache(cache_path) -> dict:
    """``"a|b"`` -> (kp_a, kp_b, score)."""
    out = {}
    with np.load(cache_path) as z:
        for k in z.files:
            v = z[k].astype(np.float64)
            out[k.replace("__", "|")] = (v[:, 0:2], v[:, 2:4], v[:, 4])
    return out


class KeypointQuantiser:
    """Per-image keypoint pool for learned matches: positions snapped to a
    ``cell`` px grid, one keypoint per occupied cell, indices assigned in order
    of first use and offset by the image's existing (SIFT) keypoint count."""

    def __init__(self, cell: float = DEFAULT_QUANT_PX):
        self.cell = float(cell)
        self.base: dict[int, int] = {}
        self.cells: dict[int, dict[tuple[int, int], int]] = defaultdict(dict)
        self.xy: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def set_base(self, image_id: int, n_existing: int) -> None:
        self.base[int(image_id)] = int(n_existing)

    def indices(self, image_id: int, xy: np.ndarray) -> np.ndarray:
        """COLMAP keypoint indices for (N,2) OpenCV-convention positions."""
        image_id = int(image_id)
        if image_id not in self.base:
            raise KeyError(f"set_base({image_id}, ...) first")
        cells = self.cells[image_id]
        out = np.empty(len(xy), np.int64)
        for n, (x, y) in enumerate(np.asarray(xy, float)):
            key = (int(math.floor(x / self.cell + 0.5)), int(math.floor(y / self.cell + 0.5)))
            idx = cells.get(key)
            if idx is None:
                idx = len(cells)
                cells[key] = idx
                self.xy[image_id].append((key[0] * self.cell, key[1] * self.cell))
            out[n] = self.base[image_id] + idx
        return out

    def new_keypoints(self, image_id: int) -> np.ndarray:
        """(M, 6) float32 COLMAP rows for the appended keypoints: x, y in COLMAP's
        convention (pixel centres at +0.5, as hloc imports them), identity shape."""
        pts = np.asarray(self.xy.get(int(image_id), []), np.float32).reshape(-1, 2) + 0.5
        rows = np.zeros((len(pts), 6), np.float32)
        rows[:, 0:2] = pts
        rows[:, 2] = 1.0
        rows[:, 5] = 1.0
        return rows


def _rot_deg(R) -> float:
    c = (np.trace(np.asarray(R, float)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _tvg_rotation(tvg) -> np.ndarray | None:
    try:
        return np.asarray(tvg.cam2_from_cam1.rotation.matrix(), float)
    except Exception:  # noqa: BLE001 -- an unset pose is "no rotation"
        return None


def _verified(tvg) -> bool:
    n = 0 if tvg is None else len(tvg.inlier_matches)
    cfg = 0 if tvg is None else int(tvg.config.value if hasattr(tvg.config, "value") else tvg.config)
    return n >= MIN_VERIFIED_INLIERS and cfg not in UNVERIFIED_CONFIGS


def two_view(camera, pts1, pts2, matches, *, relative_pose: bool, seed: int = 0):
    """COLMAP's two-view geometry, default options (as match_sequential verifies)."""
    import pycolmap

    opts = pycolmap.TwoViewGeometryOptions()
    opts.compute_relative_pose = bool(relative_pose)
    opts.ransac.random_seed = int(seed)
    return pycolmap.estimate_two_view_geometry(camera, np.asarray(pts1, np.float64), camera,
                                               np.asarray(pts2, np.float64),
                                               np.asarray(matches, np.uint32).reshape(-1, 2), opts)


def consensus(chain_len: int, rot: dict, dt: dict, candidates: set, *, cycle_bound_deg: float,
              max_skip: int = 2) -> dict:
    """Which learned links to keep.

    ``rot[(i, j)]`` (i < j, chain positions): rotation of j from i for every
    link that verified (SIFT or learned). ``candidates``: the learned links.
    ``dt[(i, j)]``: seconds between the two frames.

    A learned link (i, j) is kept iff
      1. **physics**: its rotation is at most 300 deg/s * dt + 30 deg (the
         harness's bound for a head; METRICS.md); and
      2. **agreement with the neighbouring chain**: it closes at least one
         triangle (i, m, j) with two other verified links of the chain (|m-i|
         and |m-j| <= max_skip), and the median closure error
         angle(R_ij^T R_mj R_im) over those triangles is <= cycle_bound_deg.

    Why a triangle and not the inlier count: a learned matcher can return
    many "inliers" for a pure rotation or a blur smear, and the two-view
    geometry it fits is then self-consistent but wrong; a rotation that
    disagrees with the rotations of its neighbours cannot be right. The bound
    is the control world's noise floor (`measure_cycle_noise`), not a value
    chosen on the target.
    """
    verdicts = {}

    def R(i, j):
        if (i, j) in rot:
            return rot[(i, j)]
        if (j, i) in rot:
            return rot[(j, i)].T
        return None

    for (i, j) in sorted(candidates):
        if (i, j) not in rot:
            verdicts[(i, j)] = {"keep": False, "why": "no rotation"}
            continue
        ang = _rot_deg(rot[(i, j)])
        limit = PHYS_ROT_DEG_PER_S * float(dt.get((i, j), 0.0)) + PHYS_ROT_SLACK_DEG
        errs = []
        for m in range(max(0, i - max_skip), min(chain_len, j + max_skip + 1)):
            if m in (i, j) or abs(m - i) > max_skip or abs(m - j) > max_skip:
                continue
            Rim, Rmj = R(i, m), R(m, j)
            if Rim is None or Rmj is None:
                continue
            errs.append(_rot_deg(rot[(i, j)].T @ Rmj @ Rim))
        med = float(np.median(errs)) if errs else float("nan")
        if ang > limit:
            keep, why = False, f"rotation {ang:.1f} deg > physical {limit:.1f}"
        elif not errs:
            keep, why = False, "no closing triangle"
        elif med > cycle_bound_deg:
            keep, why = False, f"cycle {med:.2f} deg > {cycle_bound_deg:.2f}"
        else:
            keep, why = True, "ok"
        verdicts[(i, j)] = {"keep": keep, "why": why, "rot_deg": ang, "phys_limit_deg": limit,
                            "triangles": len(errs), "cycle_median_deg": med,
                            "cycle_max_deg": float(np.max(errs)) if errs else float("nan")}
    return verdicts


def _in_mask(mask, xy) -> np.ndarray:
    if mask is None:
        return np.zeros(len(xy), bool)
    h, w = mask.shape[:2]
    x = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return mask[y, x].astype(bool)


def bridge_augment(database_path, name_to_image_id: dict, chains: list[Chain], eloftr: dict | None, *,
                   cycle_bound_deg: float, quant_px: float = DEFAULT_QUANT_PX, max_skip: int = 2,
                   mask_for: Callable[[str], np.ndarray | None] | None = None, seed: int = 0,
                   dry_run: bool = False, record_triangles: bool = False) -> dict:
    """Add consensus-checked learned links across every chain, in place.

    For each chain pair (skip <= ``max_skip``) whose images are both in the
    database: if SIFT already verified it, its rotation is taken from COLMAP's
    two-view geometry on the stored inliers; otherwise, if ``eloftr`` has
    correspondences for it, they are quantised (``KeypointQuantiser``), masked
    (``mask_for(name)``: True = transient, e.g. hands/phone; matches landing
    there are dropped), combined with the pair's own SIFT matches and
    verified by COLMAP two-view geometry. Learned links that verify then face
    `consensus`. Kept links are written: appended keypoints (+ zero
    descriptors so the counts stay equal), the combined matches, and their
    two-view geometry. Rejected links leave the database exactly as SIFT left it.

    With ``eloftr=None`` nothing is written; the report still classifies the
    SIFT bridging of every chain (the SIFT-only arm).
    """
    import pycolmap

    db = pycolmap.Database.open(str(database_path))
    report = {"chains": [], "cycle_bound_deg": cycle_bound_deg, "quant_px": quant_px, "max_skip": max_skip,
              "learned_links_verified": 0, "learned_links_kept": 0, "masked_matches_dropped": 0}
    try:
        cam = None
        quant = KeypointQuantiser(quant_px)
        kp_cache = {}

        def kps(iid):
            if iid not in kp_cache:
                kp_cache[iid] = db.read_keypoints(iid)
            return kp_cache[iid]

        writes = []  # (id_a, id_b, matches (M,2) in a->b order, tvg)
        touched = set()
        for c in chains:
            ids = [name_to_image_id.get(n) for n in c.names]
            rot, dtm, cand, link = {}, {}, set(), {}
            learned_tvg = {}
            for i, j in c.pairs(max_skip):
                a, b = ids[i], ids[j]
                if a is None or b is None:
                    continue
                if cam is None:
                    cam = db.read_camera(db.read_image(a).camera_id)
                dtm[(i, j)] = abs(c.received_at[j] - c.received_at[i]) if np.isfinite(
                    c.received_at[j] - c.received_at[i]) else 0.0
                sift = db.read_two_view_geometry(a, b) if db.exists_two_view_geometry(a, b) else None
                if sift is not None and _verified(sift):
                    m = np.asarray(sift.inlier_matches, np.int64)
                    # read_two_view_geometry returns matches in (a, b) order
                    g = two_view(cam, kps(a)[m[:, 0], :2], kps(b)[m[:, 1], :2],
                                 np.stack([np.arange(len(m)), np.arange(len(m))], 1), relative_pose=True, seed=seed)
                    R = _tvg_rotation(g)
                    if R is not None:
                        rot[(i, j)] = R
                    link[(i, j)] = {"by": "sift", "inliers": int(len(m)),
                                    "config": int(sift.config.value if hasattr(sift.config, "value") else sift.config)}
                    continue
                link[(i, j)] = {"by": None, "inliers": int(len(sift.inlier_matches)) if sift is not None else 0}
                key = f"{c.names[i]}|{c.names[j]}"
                if eloftr is None or key not in eloftr:
                    continue
                ka, kb, _sc = eloftr[key]
                drop = _in_mask(mask_for(c.names[i]) if mask_for else None, ka) | \
                    _in_mask(mask_for(c.names[j]) if mask_for else None, kb)
                report["masked_matches_dropped"] += int(drop.sum())
                ka, kb = ka[~drop], kb[~drop]
                if len(ka) < MIN_VERIFIED_INLIERS:
                    link[(i, j)]["learned_raw"] = int(len(ka))
                    continue
                for iid in (a, b):
                    if iid not in quant.base:
                        quant.set_base(iid, len(kps(iid)))
                ia = quant.indices(a, ka)
                ib = quant.indices(b, kb)
                lm = np.unique(np.stack([ia, ib], 1), axis=0)
                raw = db.read_matches(a, b) if db.exists_matches(a, b) else np.zeros((0, 2), np.uint32)
                raw = np.asarray(raw, np.int64).reshape(-1, 2)
                allm = np.concatenate([raw, lm], 0)
                pa = np.concatenate([kps(a)[:, :2], quant.new_keypoints(a)[:, :2]], 0) if len(quant.xy[a]) else kps(a)[:, :2]
                pb = np.concatenate([kps(b)[:, :2], quant.new_keypoints(b)[:, :2]], 0) if len(quant.xy[b]) else kps(b)[:, :2]
                g = two_view(cam, pa, pb, allm, relative_pose=True, seed=seed)
                link[(i, j)]["learned_raw"] = int(len(lm))
                link[(i, j)]["learned_inliers"] = int(len(g.inlier_matches))
                if _verified(g):
                    report["learned_links_verified"] += 1
                    R = _tvg_rotation(g)
                    if R is not None:
                        rot[(i, j)] = R
                        cand.add((i, j))
                        learned_tvg[(i, j)] = (a, b, allm, g)
            verdicts = consensus(len(c.names), rot, dtm, cand, cycle_bound_deg=cycle_bound_deg, max_skip=max_skip)
            for (i, j), v in verdicts.items():
                link[(i, j)].update(v)
                if v["keep"]:
                    link[(i, j)]["by"] = "eloftr"
                    writes.append(learned_tvg[(i, j)])
                    report["learned_links_kept"] += 1
            entry = {
                "gap_index": c.gap_index, "names": list(c.names), "kf_before": c.kf_before, "kf_after": c.kf_after,
                "links": {f"{i}-{j}": v for (i, j), v in sorted(link.items())},
                "bridging": chain_bridging(len(c.names), c.kf_before, c.kf_after, link),
            }
            if record_triangles:
                kinds = {e: ("learned" if e in cand else "sift") for e in rot}
                entry["triangles"] = all_triangles(len(c.names), rot, kinds, max_skip)
            report["chains"].append(entry)
        report["writes"] = len(writes)
        if dry_run or not writes:
            return report
        # Keypoints first (so every match index exists), then matches + geometry.
        for iid in {w[0] for w in writes} | {w[1] for w in writes}:
            new = quant.new_keypoints(iid)
            if len(new):
                base = kps(iid)
                cols = base.shape[1]
                rows = np.zeros((len(new), cols), np.float32)
                rows[:, :min(cols, 6)] = new[:, :min(cols, 6)]
                db.update_keypoints(iid, np.concatenate([base, rows], 0).astype(np.float32))
                touched.add(iid)
        for a, b, allm, g in writes:
            if db.exists_matches(a, b):
                db.delete_matches(a, b)
            db.write_matches(a, b, np.asarray(allm, np.uint32))
            if db.exists_two_view_geometry(a, b):
                db.delete_two_view_geometry(a, b)
            db.write_two_view_geometry(a, b, g)
        report["images_with_appended_keypoints"] = len(touched)
    finally:
        db.close()
    if touched:
        _pad_descriptors(database_path, touched)
    return report


def _pad_descriptors(database_path, image_ids) -> None:
    """Zero descriptor rows for appended keypoints, so rows(keypoints) ==
    rows(descriptors) for every image (nothing matches after augmentation;
    this only keeps the database self-consistent)."""
    import sqlite3

    con = sqlite3.connect(str(database_path))
    try:
        for iid in image_ids:
            nk = con.execute("select rows from keypoints where image_id=?", (int(iid),)).fetchone()
            row = con.execute("select rows, cols, data from descriptors where image_id=?", (int(iid),)).fetchone()
            if nk is None or row is None or row[0] >= nk[0]:
                continue
            rows, cols, data = row
            dtype = np.uint8 if len(data) == rows * cols else np.float32
            d = np.frombuffer(data, dtype).reshape(rows, cols)
            pad = np.zeros((nk[0] - rows, cols), dtype)
            con.execute("update descriptors set rows=?, data=? where image_id=?",
                        (int(nk[0]), np.concatenate([d, pad], 0).tobytes(), int(iid)))
        con.commit()
    finally:
        con.close()


def gap_link_consensus(database_path, is_gap: Callable[[str], bool], received_at: dict, *,
                       cycle_bound_deg: float, seed: int = 0, max_partners: int = 12,
                       dry_run: bool = False) -> dict:
    """Refuse every verified pair that touches a solver-only frame and cannot be
    corroborated -- SIFT or learned alike.

    Why this exists: on the known-good control, raw gap frames matched by SIFT
    alone (arm B1) displaced a keyframe by 39 % of the room's extent and bent a
    block by 10-15 deg, through links of 15-45 inliers between blurred frames.
    Geometric verification passes such links; nothing checks them against
    their neighbours. So each verified pair (x, y) with a gap frame in it is
    kept only if (a) its rotation is physically possible for a head in the
    time between the frames (`PHYS_ROT_*`) and (b) it closes at least one
    triangle (x, m, y) with two other verified pairs of the database, the
    median closure error over up to ``max_partners`` partners m (nearest in
    capture order) being <= ``cycle_bound_deg`` -- the same bound as the
    learned links, measured on the control. Refused pairs lose their two-view
    geometry (the mapper never sees them); their raw matches stay.
    Keyframe-to-keyframe pairs are never touched.
    """
    import pycolmap

    db = pycolmap.Database.open(str(database_path))
    report = {"cycle_bound_deg": cycle_bound_deg, "max_partners": max_partners, "pairs": []}
    try:
        names = {im.image_id: im.name for im in db.read_all_images()}
        cam_of = {im.image_id: im.camera_id for im in db.read_all_images()}
        cams = {}
        pair_ids, tvgs = db.read_two_view_geometries()
        verified = {}
        nbrs = defaultdict(set)
        for pid, g in zip(pair_ids, tvgs):
            b = int(pid) % COLMAP_PAIR_BASE
            a = (int(pid) - b) // COLMAP_PAIR_BASE
            if a in names and b in names and _verified(g):
                verified[(a, b)] = g
                nbrs[a].add(b)
                nbrs[b].add(a)
        seq = {i: (parse_staged_name(n) or (i, ""))[0] for i, n in names.items()}
        kp = {}
        rot: dict = {}

        def kps(i):
            if i not in kp:
                kp[i] = db.read_keypoints(i)[:, :2]
            return kp[i]

        def camera(i):
            c = cam_of[i]
            if c not in cams:
                cams[c] = db.read_camera(c)
            return cams[c]

        def R(x, y):
            """rotation of y from x, from the stored inliers (None if unavailable)."""
            a, b = (x, y) if x < y else (y, x)
            if (a, b) not in rot:
                g = verified.get((a, b))
                val = None
                if g is not None:
                    m = np.asarray(g.inlier_matches, np.int64)
                    gg = two_view(camera(a), kps(a)[m[:, 0]], kps(b)[m[:, 1]],
                                  np.stack([np.arange(len(m)), np.arange(len(m))], 1), relative_pose=True,
                                  seed=seed)
                    val = _tvg_rotation(gg)
                rot[(a, b)] = val
            v = rot[(a, b)]
            if v is None:
                return None
            return v if x < y else v.T

        targets = [(a, b) for (a, b) in verified if is_gap(names[a]) or is_gap(names[b])]
        drop = []
        why_count = Counter()
        for a, b in sorted(targets):
            Rab = R(a, b)
            t_a, t_b = received_at.get(names[a]), received_at.get(names[b])
            dt = abs(t_b - t_a) if (t_a is not None and t_b is not None) else float("inf")
            limit = PHYS_ROT_DEG_PER_S * dt + PHYS_ROT_SLACK_DEG
            common = sorted(nbrs[a] & nbrs[b], key=lambda m: min(abs(seq[m] - seq[a]), abs(seq[m] - seq[b])))
            errs = []
            if Rab is not None:
                for m in common[:max_partners]:
                    Ram, Rmb = R(a, m), R(m, b)
                    if Ram is None or Rmb is None:
                        continue
                    errs.append(_rot_deg(Rab.T @ Rmb @ Ram))
            ang = _rot_deg(Rab) if Rab is not None else float("nan")
            med = float(np.median(errs)) if errs else float("nan")
            if Rab is None:
                keep, why = False, "no rotation"
            elif ang > limit:
                keep, why = False, "physical"
            elif not errs:
                keep, why = False, "no closing triangle"
            elif med > cycle_bound_deg:
                keep, why = False, "cycle"
            else:
                keep, why = True, "ok"
            why_count[why] += 1
            report["pairs"].append({"a": names[a], "b": names[b], "inliers": int(len(verified[(a, b)].inlier_matches)),
                                    "rot_deg": ang, "triangles": len(errs), "cycle_median_deg": med, "keep": keep,
                                    "why": why})
            if not keep:
                drop.append((a, b))
        report.update(gap_pairs=len(targets), kept=len(targets) - len(drop), refused=len(drop),
                      refused_by_reason={k: v for k, v in why_count.items() if k != "ok"},
                      rotations_computed=len(rot))
        if not dry_run:
            for a, b in drop:
                db.delete_two_view_geometry(a, b)
    finally:
        db.close()
    return report


def restore_keyframe_graph(database_path, reference_db, *, is_keyframe=None, scratch_dir=None) -> dict:
    """Make the keyframe-to-keyframe pair graph of ``database_path`` IDENTICAL to
    the reference database's (the product recipe run on the keyframes alone).

    Why: the driver matches every staged image sequentially with the product's
    overlap of 20 IMAGES and runs loop detection every 10th IMAGE. With solver-
    only frames interleaved, that window holds gap frames instead of keyframes:
    on the control, 734 keyframe pairs within 20 keyframes and 684 loop pairs of
    the product disappeared, and blocks held by them drifted (Sim(3) RMS 2.5 %
    of extent, one block 39 %). Gap frames must ADD pairs to the product's
    graph, never take pairs out of it. So every keyframe-keyframe row (matches
    and two-view geometry, verified or not) is replaced by the reference's.

    Requires the keyframes' SIFT keypoints to be identical in both databases
    (same staged bytes, same extraction options; appended learned keypoints
    after them are allowed); refuses otherwise.
    """
    import shutil
    import sqlite3

    import pycolmap

    is_keyframe = is_keyframe or (lambda n: n.endswith(f"_{KIND_KEYFRAME}.jpg"))
    scratch_dir = Path(scratch_dir or Path(database_path).parent)
    ref_copy = scratch_dir / f"reference.p{__import__('os').getpid()}.db"
    shutil.copyfile(reference_db, ref_copy)       # never open the reference (a cache) read-write
    try:
        def kp_table(path):
            con = sqlite3.connect(str(path))
            try:
                names = dict(con.execute("select image_id, name from images"))
                kp = {names[i]: (r, bytes(d)) for i, r, d in con.execute("select image_id, rows, data from keypoints")
                      if is_keyframe(names[i])}
                return {n: i for i, n in names.items()}, kp
            finally:
                con.close()

        arm_ids, arm_kp = kp_table(database_path)
        ref_ids, ref_kp = kp_table(ref_copy)
        kf = sorted(n for n in ref_kp if n in arm_kp)
        # identical, or the reference's rows followed by appended (learned) ones
        bad = [n for n in kf if arm_kp[n][0] < ref_kp[n][0] or not arm_kp[n][1].startswith(ref_kp[n][1])]
        if bad or len(kf) != len(ref_kp):
            raise RuntimeError(f"keyframe keypoints differ from the reference ({len(bad)} differ, "
                               f"{len(ref_kp) - len(kf)} missing): cannot transplant its pair graph")
        kf_arm = {arm_ids[n] for n in kf}
        rev_ref = {i: n for n, i in ref_ids.items()}
        # drop the arm's own keyframe-keyframe rows
        con = sqlite3.connect(str(database_path))
        removed = {"matches": 0, "two_view_geometries": 0}
        try:
            for table in ("matches", "two_view_geometries"):
                rows = [pid for (pid,) in con.execute(f"select pair_id from {table}")]
                kill = []
                for pid in rows:
                    b = pid % COLMAP_PAIR_BASE
                    a = (pid - b) // COLMAP_PAIR_BASE
                    if a in kf_arm and b in kf_arm:
                        kill.append((pid,))
                con.executemany(f"delete from {table} where pair_id=?", kill)
                removed[table] = len(kill)
            con.commit()
        finally:
            con.close()
        ref = pycolmap.Database.open(str(ref_copy))
        db = pycolmap.Database.open(str(database_path))
        added = {"matches": 0, "two_view_geometries": 0, "verified": 0}
        try:
            con = sqlite3.connect(ref_copy.as_uri() + "?mode=ro", uri=True)
            m_pairs = [pid for (pid,) in con.execute("select pair_id from matches")]
            g_pairs = [pid for (pid,) in con.execute("select pair_id from two_view_geometries")]
            con.close()
            for pids, kind in ((m_pairs, "matches"), (g_pairs, "two_view_geometries")):
                for pid in pids:
                    b = pid % COLMAP_PAIR_BASE
                    a = (pid - b) // COLMAP_PAIR_BASE
                    na, nb = rev_ref.get(a), rev_ref.get(b)
                    if na is None or nb is None or not (is_keyframe(na) and is_keyframe(nb)):
                        continue
                    if kind == "matches":
                        db.write_matches(arm_ids[na], arm_ids[nb], np.asarray(ref.read_matches(a, b), np.uint32))
                    else:
                        g = ref.read_two_view_geometry(a, b)
                        db.write_two_view_geometry(arm_ids[na], arm_ids[nb], g)
                        added["verified"] += int(_verified(g))
                    added[kind] += 1
        finally:
            db.close()
            ref.close()
    finally:
        Path(ref_copy).unlink(missing_ok=True)
        for side in ("-wal", "-shm"):
            Path(str(ref_copy) + side).unlink(missing_ok=True)
    return {"keyframes": len(kf), "removed": removed, "added": added}


def received_at_by_name(frames_csv) -> dict:
    """staged name (both kinds) -> receipt time, from a labelled frame table."""
    import pandas as pd

    fr = pd.read_csv(frames_csv)
    out = {}
    for f, t in zip(fr["frame_index"], fr["received_at"]):
        out[staged_name(int(f), KIND_KEYFRAME)] = float(t)
        out[staged_name(int(f), KIND_GAP)] = float(t)
    return out


def chain_bridging(n: int, kf_before: int, kf_after: int, link: dict) -> dict:
    """Is the keyframe before the gap joined to the keyframe after it through
    verified links of the chain, by SIFT alone, or only with learned links?"""
    def connected(allowed):
        adj = defaultdict(set)
        for (i, j), v in link.items():
            if v.get("by") in allowed:
                adj[i].add(j)
                adj[j].add(i)
        seen, stack = {kf_before}, [kf_before]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return kf_after in seen, len(seen)

    sift_ok, _ = connected({"sift"})
    all_ok, _ = connected({"sift", "eloftr"})
    return {"bridged_by": "sift" if sift_ok else ("eloftr" if all_ok else None)}


def save_chains(chains: list[Chain], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(c) for c in chains], indent=1), encoding="utf-8")


def load_chains(path) -> list[Chain]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Chain(r["gap_index"], tuple(r["names"]), int(r["kf_before"]), int(r["kf_after"]),
                  tuple(int(s) for s in r["capture_seqs"]), tuple(float(t) for t in r["received_at"])) for r in rows]


def workspace_mask_reader(workspace) -> Callable[[str], np.ndarray | None]:
    """``name -> bool (H, W)`` True where the driver's COLMAP mask says ignore
    (``<workspace>/masks/<name>.png``, 0 = ignore); None when there is none."""
    import cv2

    root = Path(workspace) / "masks"
    cache = {}

    def mask_for(name: str):
        if name not in cache:
            p = root / f"{name}.png"
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.is_file() else None
            cache[name] = None if m is None else (m == 0)
        return cache[name]

    return mask_for


def augment_hook(database_path, name_to_image_id, images_dir, workspace, *, chains_path, eloftr_cache=None,
                 cycle_bound_deg: float = DEFAULT_CYCLE_BOUND_DEG, quant_px: float = DEFAULT_QUANT_PX,
                 max_skip: int = 2, use_workspace_masks: bool = True, report_name: str = "bridge_report.json",
                 report_copy=None, gap_consensus: bool = False, frames_csv=None, reference_db=None) -> dict:
    """``VariantConfig.augment`` entry point (``coherence_exp.bridge:augment_hook``).

    Reads the chains (`save_chains`) and the precomputed ELoFTR correspondences
    (`precompute_eloftr`; no GPU here). ``eloftr_cache=None`` writes nothing and
    only classifies each chain's SIFT bridging (the SIFT-only arm). Learned
    matches that land on the driver's transient masks are dropped when masks
    are on. The full per-link report goes to ``<workspace>/<report_name>``.
    """
    t0 = time.perf_counter()
    restored = restore_keyframe_graph(database_path, reference_db) if reference_db else None
    chains = load_chains(chains_path)
    eloftr = load_eloftr_cache(eloftr_cache) if eloftr_cache else None
    mask_for = workspace_mask_reader(workspace) if use_workspace_masks and (Path(workspace) / "masks").is_dir() \
        else None
    rep = bridge_augment(database_path, name_to_image_id, chains, eloftr, cycle_bound_deg=float(cycle_bound_deg),
                         quant_px=float(quant_px), max_skip=int(max_skip), mask_for=mask_for)
    if gap_consensus:
        rat = received_at_by_name(frames_csv) if frames_csv else {}
        rep["gap_consensus"] = gap_link_consensus(database_path, lambda n: n.endswith(f"_{KIND_GAP}.jpg"), rat,
                                                  cycle_bound_deg=float(cycle_bound_deg))
    rep["seconds"] = round(time.perf_counter() - t0, 2)
    rep["masks_used"] = mask_for is not None
    rep["keyframe_graph_restored"] = restored
    rep["summary"] = summarize_bridging(rep)
    if restored is not None:
        rep["summary"]["keyframe_graph_restored"] = restored
    if gap_consensus:
        gc = rep["gap_consensus"]
        rep["summary"]["gap_consensus"] = {k: gc[k] for k in ("gap_pairs", "kept", "refused", "refused_by_reason")}
    text = json.dumps(rep, indent=1, default=_json_default)
    Path(workspace, report_name).write_text(text, encoding="utf-8")
    if report_copy:
        Path(report_copy).parent.mkdir(parents=True, exist_ok=True)
        Path(report_copy).write_text(text, encoding="utf-8")
    return dict(rep["summary"], seconds=rep["seconds"], masks_used=rep["masks_used"],
                images_with_appended_keypoints=rep.get("images_with_appended_keypoints", 0))


def summarize_bridging(rep: dict) -> dict:
    by = Counter(str(c["bridging"]["bridged_by"]) for c in rep.get("chains", []))
    return {"chains": len(rep.get("chains", [])), "bridged_by_sift": by.get("sift", 0),
            "bridged_by_eloftr": by.get("eloftr", 0), "unbridged": by.get("None", 0),
            "learned_links_verified": rep.get("learned_links_verified", 0),
            "learned_links_kept": rep.get("learned_links_kept", 0),
            "masked_matches_dropped": rep.get("masked_matches_dropped", 0),
            "cycle_bound_deg": rep.get("cycle_bound_deg")}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def all_triangles(chain_len: int, rot: dict, kinds: dict, max_skip: int = 2) -> list[dict]:
    """Closure error of every triangle (i < m < j, j - i <= max_skip) whose three
    links all have a rotation; ``kinds[(i, j)]`` is 'sift' or 'learned'."""
    out = []
    for i in range(chain_len):
        for j in range(i + 2, min(chain_len, i + max_skip + 1)):
            for m in range(i + 1, j):
                if (i, j) in rot and (i, m) in rot and (m, j) in rot:
                    err = _rot_deg(rot[(i, j)].T @ rot[(m, j)] @ rot[(i, m)])
                    ks = sorted(kinds.get(e, "?") for e in ((i, j), (i, m), (m, j)))
                    out.append({"i": i, "m": m, "j": j, "err_deg": err, "kinds": "+".join(ks)})
    return out


def measure_cycle_noise(chains: list[Chain], name_to_image_id: dict, database_path, eloftr: dict | None, *,
                        max_skip: int = 2, quant_px: float = DEFAULT_QUANT_PX, seed: int = 0,
                        mask_for=None) -> dict:
    """Triangle-closure errors on a world's gap chains, without writing.

    Every triangle of verified links is recorded with the kinds of its three
    links. Run on the known-good control world, the all-SIFT triangles are the
    matcher-independent noise floor of a genuine link's rotation, and that is
    what sets the consensus bound (`consensus`); the learned-only triangles
    show what the learned matcher adds on top.
    """
    rep = bridge_augment(database_path, name_to_image_id, chains, eloftr, cycle_bound_deg=float("inf"),
                         quant_px=quant_px, max_skip=max_skip, mask_for=mask_for, seed=seed, dry_run=True,
                         record_triangles=True)
    by_kind = defaultdict(list)
    for ch in rep["chains"]:
        for t in ch.get("triangles", []):
            by_kind[t["kinds"]].append(t["err_deg"])
    stats = {}
    for k, v in sorted(by_kind.items()):
        a = np.asarray(v, float)
        stats[k] = {"n": int(len(a)), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                    "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}
    return {"by_kind": stats, "errors": {k: list(map(float, v)) for k, v in by_kind.items()}, "report": rep}


# ---------------------------------------------------------------------------
# 4. Cut accounting.


def cut_strengths(order: list[str], verified_pairs: Iterable[tuple[str, str]], keyframe_names: list[str], *,
                  window: int = 10) -> list[int]:
    """D1's sequential-cut measure, generalised to solver-only frames.

    ``order``: every image in capture order (keyframes and gap frames).
    For each keyframe boundary k|k+1, the strength is the minimum, over every
    split point between keyframe k and keyframe k+1 in ``order``, of the
    number of verified pairs that straddle the split with both images inside
    the keyframe window [k-window+1, k+window]. With no gap frames this is
    D1's ``probe_cuts.py`` count (pairs spanning k|k+1 at most ``window``
    keyframes apart). Zero is a cut.
    """
    pos = {n: i for i, n in enumerate(order)}
    kpos = [pos[n] for n in keyframe_names]
    # keyframe ordinal of every image position: the last keyframe at or before it
    kord = np.searchsorted(np.asarray(kpos), np.arange(len(order)), side="right") - 1
    P = []
    for a, b in verified_pairs:
        if a in pos and b in pos:
            i, j = sorted((pos[a], pos[b]))
            P.append((i, j))
    P = np.asarray(P, np.int64).reshape(-1, 2)
    out = []
    for k in range(len(kpos) - 1):
        lo_k, hi_k = max(0, k - window + 1), min(len(kpos) - 1, k + window)
        lo, hi = kpos[lo_k], kpos[hi_k]
        sel = P[(P[:, 0] >= lo) & (P[:, 1] <= hi)] if len(P) else P
        best = None
        for s in range(kpos[k], kpos[k + 1]):
            n = int(((sel[:, 0] <= s) & (sel[:, 1] > s)).sum()) if len(sel) else 0
            best = n if best is None else min(best, n)
        out.append(0 if best is None else best)
    return out


def verified_pairs_from_db(database_path) -> list[tuple[str, str, int, int]]:
    """(name_a, name_b, inliers, config) for every stored two-view geometry."""
    import sqlite3

    con = sqlite3.connect(Path(database_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        names = dict(con.execute("select image_id, name from images"))
        out = []
        for pid, rows, cfg in con.execute("select pair_id, rows, config from two_view_geometries"):
            b = pid % COLMAP_PAIR_BASE
            a = (pid - b) // COLMAP_PAIR_BASE
            if a in names and b in names:
                out.append((names[a], names[b], int(rows), int(cfg)))
        return out
    finally:
        con.close()


def model_cut_tracks(model_dir, order: list[str], keyframe_names: list[str], *, window: int = 10) -> dict:
    """From a COLMAP model: per keyframe boundary k|k+1, the minimum over split
    points between the two keyframes of the number of 3-D points observed on
    both sides (any image), and the number observed by keyframes on both sides
    within ``window`` keyframes (keyframe observations only -- what the
    harness's rigid groups can see)."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    names, oi, op = [], [], []
    index = {}
    for img in rec.images.values():
        if img.has_pose:
            index[img.image_id] = len(names)
            names.append(img.name)
    for pid, p in rec.points3D.items():
        for e in p.track.elements:
            if e.image_id in index:
                oi.append(index[e.image_id])
                op.append(int(pid))
    out = model_cut_tracks_from_arrays(names, np.asarray(oi), np.asarray(op), order, keyframe_names, window=window)
    return {"boundaries": out, "registered_images": len(names), "points": int(len(rec.points3D))}


def model_cut_tracks_from_arrays(names, obs_image, obs_point, order: list[str], keyframe_names: list[str], *,
                                 window: int = 10) -> list[dict]:
    """`model_cut_tracks` on observation arrays (the driver's ``model.npz``):
    ``names[obs_image[i]]`` observes point ``obs_point[i]``."""
    pos = {n: i for i, n in enumerate(order)}
    kidx = {n: i for i, n in enumerate(keyframe_names)}
    kpos = [pos[n] for n in keyframe_names]
    registered = set(names)
    img_pos = np.array([pos.get(n, -1) for n in names])
    img_k = np.array([kidx.get(n, -1) for n in names])
    oi = np.asarray(obs_image, np.int64)
    opt = np.asarray(obs_point, np.int64)
    kspans = []
    tracks = []  # sorted capture positions of each track's observations
    if len(oi):
        order_ = np.argsort(opt, kind="stable")
        oi_s, op_s = oi[order_], opt[order_]
        starts = np.flatnonzero(np.r_[True, op_s[1:] != op_s[:-1]])
        ends = np.r_[starts[1:], len(op_s)]
        for s, e in zip(starts, ends):
            ps = np.unique(img_pos[oi_s[s:e]])
            ps = ps[ps >= 0]
            if len(ps) >= 2:
                tracks.append(ps)
            ks = np.unique(img_k[oi_s[s:e]])
            ks = ks[ks >= 0]
            if len(ks) >= 2:
                kspans.append(ks)
    t_lo = np.array([t[0] for t in tracks]) if tracks else np.zeros(0, int)
    t_hi = np.array([t[-1] for t in tracks]) if tracks else np.zeros(0, int)
    out = []
    for k in range(len(kpos) - 1):
        a, b = kpos[k], kpos[k + 1]
        # LOCAL cross-section: only observations inside the keyframe window count,
        # so a loop-closure track spanning the whole walk does not "bridge" a cut
        lo, hi = kpos[max(0, k - window + 1)], kpos[min(len(kpos) - 1, k + window)]
        any_side = 0
        if b > a and len(tracks):
            cand = np.flatnonzero((t_lo <= hi) & (t_hi >= lo))
            diff = np.zeros(hi - lo + 2)
            for t in cand:
                ps = tracks[t]
                ps = ps[(ps >= lo) & (ps <= hi)]
                if len(ps) >= 2 and ps[0] < ps[-1]:
                    diff[ps[0] - lo] += 1
                    diff[ps[-1] - lo] -= 1
            cross = np.cumsum(diff)
            any_side = int(cross[a - lo:b - lo].min())
        kk = 0
        for ks in kspans:
            if ks[0] > k + window or ks[-1] < k - window + 1:
                continue
            left = np.any((ks >= k - window + 1) & (ks <= k))
            right = np.any((ks >= k + 1) & (ks <= k + window))
            kk += int(left and right)
        out.append({"k": k, "tracks_min_cross_section": any_side, "keyframe_tracks_window": kk,
                    "both_registered": keyframe_names[k] in registered and keyframe_names[k + 1] in registered})
    return out
