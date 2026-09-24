"""Re-redaction: recover false-positive fill on a world captured under an older gate.

Contract: `docs/contracts/WORLD-BUILDER-APPEARANCE.md` section 6.5.
CLI: `scripts/world_reredact.py`.

WHY THIS EXISTS
---------------
Worlds captured before `plausibility3` were redacted by an older rule of the
same YuNet@0.30 family. The ungated rule filled 12.61% of every pixel of the
canonical capture, almost all of it the wearer's hands, a PC case and bare
wall; `plausibility3`, run on the same raw frames, fills 5.99%, and the page's
opening pose (ki 148) goes from 52% black to 0%. Measured by the fix-it
re-redaction lane (`Glasses-scratch/wb-final-recon/fixit/reredact/REREDACT.md`).

WHAT IT DOES, AND WHAT IT NEVER DOES
------------------------------------
It is an EXPLICIT, LOGGED step, run by hand. It never runs during a walk, and
no build path calls it: after it has run, dense, surface and appearance read
the re-redacted keyframes through `WorldStore.keyframe_image_set` like any
other keyframes, and none of them re-redacts from raw.

- The capture's own `sessions/<sid>/images/` is never modified or deleted. The
  new set is written BESIDE it, `images.redacted-<gate>/`, with a
  `record.json` naming every frame's origin and hashes.
- The switch is `sessions/<sid>/redaction_set.json`, written atomically LAST.
  Switching back (`--revert`) rewrites that pointer; no image moves.
- Only from an older label of the same family (`REREDACTABLE_LABELS`). Never
  from `none` or an unknown label: those images may themselves be raw, and they
  take the privacy lane's re-redact-the-stored-bytes path at build time.
- Raw bytes are read in exactly one function, `reredact_frame`, and never
  leave it: what comes out is always a fresh encode by the redactor, or the
  stored keyframe's own bytes.
- A frame is re-redacted only when its raw frame provably is that keyframe's
  source (`verify_raw_matches_stored`), the redactor ran on it (a `none` result
  keeps the stored frame), the new fill lies INSIDE the stored fill (the
  safety invariant: re-redaction may only un-fill, never move a redaction),
  and the output is the raw frame with only near-black boxes added. Any frame
  that fails keeps its stored keyframe, and the record says why.
- Redactor unavailable, a redactor of another label, or any exception: nothing
  is written that anything reads, and nothing is switched.

THE SET'S LABEL
---------------
A switched session reads under the CURRENT label (`TARGET_LABEL`), so every
reader's allowlist applies to it unchanged. That is honest only if every
frame's fill contains what the current rule fills on that frame's raw image:

- a re-redacted frame is the current rule's output by construction;
- a frame kept because nothing was recoverable has the same fill;
- a frame kept because its raw frame is missing or unverified, or the redactor
  returned `none`, holds the STORED rule's fill, which contains the current
  rule's only when the stored rule is a superset of it. The ungated rule is:
  every gate only removes boxes from the same detection pass. `plausibility1`
  and `plausibility2` are not (each fills some large box the other does not);
- a frame kept because a MEASUREMENT says the current rule is not contained in
  the stored fill (`fill-outside-stored`) or the redactor's output is not the
  raw frame plus boxes (`output-not-raw-plus-fill`) is not covered by any
  argument, the superset one included: that measurement is the argument failing.

So an apply whose kept frames are not all covered is REFUSED, whole.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TOOL_VERSION = "wb-reredact/1"
POINTER_FORMAT = "wb-redaction-set/1"
RECORD_FORMAT = "wb-reredaction-record/1"
RECORD_FILENAME = "record.json"

_YUNET = "faces-detected-and-filled/yunet-2023mar@0.30"

# The label a re-redacted set is written under. EXACT, versioned here: when
# `redaction.py` moves to another gate this no longer matches
# `FaceRedactor().label` and the step refuses until someone adds the new gate
# deliberately, with its measurement.
TARGET_LABEL = f"{_YUNET}+plausibility3"

# Older rules of the same detector, threshold, upscale and dilation (none of
# which has changed in git), keyed to whether the rule's fill is a SUPERSET of
# the target's on the same raw frame -- which decides whether a frame kept for
# a reason other than "nothing to recover" still meets the target label.
REREDACTABLE_LABELS = {
    _YUNET: True,                       # ungated: every gate only removes boxes
    f"{_YUNET}+plausibility1": False,   # above 25%: facelike OR native; p3 also takes 1/2, 1/4
    f"{_YUNET}+plausibility2": False,   # p3 adds the edge exception, so p2 fills less
}

# -- raw verification (REREDACT.md section 1: 398/398 true pairs, 0/397
#    adjacent keyframes, 0/398 shuffled) -------------------------------------
VERIFY_DIFF = 40            # a difference this large must be explained by fill
VERIFY_BLACK = 12           # stored near-black
VERIFY_NEAR_PX = 2          # ... within this many pixels
VERIFY_UNEXPLAINED_MAX = 0.001
VERIFY_MAD_MAX = 3.0        # mean abs difference everywhere else
VERIFY_RULE = (f"diff>{VERIFY_DIFF} within {VERIFY_NEAR_PX}px of stored<={VERIFY_BLACK} "
               f"(<= {VERIFY_UNEXPLAINED_MAX:.1%} unexplained); mean abs diff elsewhere "
               f"<= {VERIFY_MAD_MAX}")

# -- fill masks: exact, by difference against the raw frame ------------------
FILL_BLACK = 12
FILL_DIFF = 25
# The invariant allows JPEG ringing at a box edge, and nothing else.
INVARIANT_TOLERANCE_PX = 3
# "Raw with only boxes added": no difference over this, further than
# OUTPUT_BOX_PX from a near-black pixel of the output (REREDACT.md section 4c).
OUTPUT_DIFF = 40
OUTPUT_BOX_PX = 4
# Fewer recovered pixels than this is JPEG noise along an unchanged box edge,
# not a recovery; the frame keeps its stored bytes.
RECOVERED_MIN_PX = 64

ORIGIN_REREDACTED = "raw-reredacted"
KEPT_NOTHING_RECOVERED = "stored:nothing-recovered"
KEPT_RAW_MISSING = "stored:raw-missing"
KEPT_RAW_UNVERIFIED = "stored:raw-unverified"
KEPT_REDACTION_NONE = "stored:redaction-none"
KEPT_FILL_OUTSIDE_STORED = "stored:fill-outside-stored"
KEPT_OUTPUT_NOT_RAW = "stored:output-not-raw-plus-fill"
# Kept frames whose stored fill equals the target's by measurement.
_KEPT_AT_TARGET = {KEPT_NOTHING_RECOVERED}
# Kept frames where a measurement CONTRADICTS the stored rule containing the
# target's fill: the current rule filled pixels the stored keyframe leaves raw,
# or produced an output that is not the raw frame plus boxes, so nothing can be
# said about its fill. Keeping the stored bytes of such a frame under the
# target's label would publish a frame the label is false for, whatever the
# stored rule (review 1, M1): the ungated superset argument is exactly what the
# measurement failed on. An apply with any such frame is refused whole.
_KEPT_CONTRADICTS_TARGET = {KEPT_FILL_OUTSIDE_STORED, KEPT_OUTPUT_NOT_RAW}


class ReredactionRefused(RuntimeError):
    """The step will not run, or will not switch, and why. Nothing was switched."""


# ---------------------------------------------------------------------------
# pixels
# ---------------------------------------------------------------------------


def _decode(data: bytes | None):
    import cv2  # noqa: PLC0415

    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2  # noqa: PLC0415

    if px <= 0 or not mask.any():
        return mask.astype(bool)
    k = 2 * int(px) + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def _absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)


def verify_raw_matches_stored(stored: np.ndarray | None, raw: np.ndarray | None):
    """Is `raw` the frame `stored` was redacted from? (ok, detail)

    Every large difference must be explained by fill (stored near-black there),
    and everything else must differ by JPEG error only. The hard negative is
    the adjacent keyframe, a few centimetres of motion away: 0 of 397 pass."""
    if stored is None or raw is None:
        return False, "undecodable"
    if raw.shape != stored.shape:
        return False, f"shape {raw.shape[1]}x{raw.shape[0]} != {stored.shape[1]}x{stored.shape[0]}"
    d = _absdiff(stored, raw)
    explained = _dilate(stored.max(axis=2) <= VERIFY_BLACK, VERIFY_NEAR_PX)
    unexplained = float(((d > VERIFY_DIFF) & ~explained).mean())
    if unexplained > VERIFY_UNEXPLAINED_MAX:
        return False, f"unexplained difference over {unexplained:.4f} of the frame"
    rest = ~explained
    mad = float(d[rest].mean()) if rest.any() else float("inf")
    if mad > VERIFY_MAD_MAX:
        return False, f"mean abs difference {mad:.2f} outside the fill"
    return True, f"mean abs difference {mad:.2f}"


def fill_mask(image: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Pixels of `image` that are redaction fill: near-black where the camera
    saw something else. Exact, by difference; fill over genuinely black scene
    is not counted, and needs not be -- nothing is hidden there."""
    return (image.max(axis=2) <= FILL_BLACK) & (_absdiff(image, raw) > FILL_DIFF)


# ---------------------------------------------------------------------------
# THE boundary: the only code that reads a raw frame
# ---------------------------------------------------------------------------


@dataclass
class FrameOutcome:
    keyframe_id: str
    seq: str
    origin: str
    detail: str | None
    image_bytes: bytes = field(repr=False, default=b"")
    stored_sha1: str | None = None
    sha1: str | None = None
    raw_sha1: str | None = None
    pixels: int = 0
    stored_fill_px: int | None = None
    new_fill_px: int | None = None
    recovered_px: int = 0
    outside_stored_px: int | None = None
    regions: int | None = None

    @property
    def reredacted(self) -> bool:
        return self.origin == ORIGIN_REREDACTED

    @property
    def effective_fill_px(self) -> int | None:
        return self.new_fill_px if self.reredacted else self.stored_fill_px

    def as_record(self) -> dict:
        return {
            "keyframe_id": self.keyframe_id, "seq": self.seq, "origin": self.origin,
            "detail": self.detail, "stored_sha1": self.stored_sha1, "sha1": self.sha1,
            "raw_sha1": self.raw_sha1, "pixels": self.pixels,
            "stored_fill_px": self.stored_fill_px, "new_fill_px": self.new_fill_px,
            "recovered_px": self.recovered_px, "outside_stored_px": self.outside_stored_px,
            "regions": self.regions,
        }


def reredact_frame(keyframe_id: str, stored_bytes: bytes, raw_path: Path | None,
                   redactor) -> FrameOutcome:
    """One keyframe: the re-redacted bytes, or the stored bytes and why.

    RAW BYTES ARE READ HERE AND NOWHERE ELSE, and they are dropped here. What
    is returned is either `stored_bytes` or a fresh encode, never the raw file.
    Raises only on a redactor that raises (which `FaceRedactor` never does); the
    caller turns that into "nothing switched"."""
    import cv2  # noqa: PLC0415

    from tower.world_builder.redaction import JPEG_QUALITY, REDACTION_NONE  # noqa: PLC0415

    seq = keyframe_id.rsplit(":", 1)[-1]
    out = FrameOutcome(keyframe_id=keyframe_id, seq=seq, origin=KEPT_RAW_MISSING,
                       detail=None, image_bytes=stored_bytes,
                       stored_sha1=hashlib.sha1(stored_bytes).hexdigest())
    out.sha1 = out.stored_sha1
    stored = _decode(stored_bytes)
    if stored is not None:
        out.pixels = int(stored.shape[0] * stored.shape[1])

    if raw_path is None:
        out.detail = "no sources.json entry"
        return out
    try:
        raw_bytes = Path(raw_path).read_bytes()
    except OSError as exc:
        out.detail = f"raw frame not readable ({type(exc).__name__})"
        return out
    out.raw_sha1 = hashlib.sha1(raw_bytes).hexdigest()
    raw = _decode(raw_bytes)
    ok, why = verify_raw_matches_stored(stored, raw)
    if not ok:
        out.origin, out.detail = KEPT_RAW_UNVERIFIED, why
        return out
    stored_fill = fill_mask(stored, raw)
    out.stored_fill_px = int(stored_fill.sum())

    result = redactor.redact(raw_bytes)
    if result.label == REDACTION_NONE:
        out.origin = KEPT_REDACTION_NONE
        out.detail = getattr(result, "unavailable_reason", None) or "redactor returned none"
        return out
    if result.label != TARGET_LABEL:
        raise ReredactionRefused(
            f"the redactor returned label {result.label!r} mid-run, not {TARGET_LABEL!r}")
    out.regions = int(getattr(result, "regions", 0) or 0)
    if result.image_bytes == raw_bytes:
        # Nothing filled: `redact` hands back its input. Never persist that; a
        # fresh encode of the decoded pixels is what the engine would have kept.
        ok_enc, enc = cv2.imencode(".jpg", raw, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok_enc:
            raise ReredactionRefused(f"could not encode keyframe {keyframe_id}")
        candidate = enc.tobytes()
    else:
        candidate = result.image_bytes
    new = _decode(candidate)
    if candidate == raw_bytes or new is None or new.shape != raw.shape:
        out.origin = KEPT_OUTPUT_NOT_RAW
        out.detail = "the redactor's output is not a fresh image of the frame's shape"
        return out

    new_fill = fill_mask(new, raw)
    out.new_fill_px = int(new_fill.sum())
    outside = new_fill & ~_dilate(stored_fill, INVARIANT_TOLERANCE_PX)
    out.outside_stored_px = int(outside.sum())
    if out.outside_stored_px:
        # THE SAFETY INVARIANT. The current rule, on a verified source, filled
        # somewhere the stored rule did not. Re-redaction may only un-fill.
        out.origin = KEPT_FILL_OUTSIDE_STORED
        out.detail = f"{out.outside_stored_px} px of new fill outside the stored fill"
        logger.warning("[Tower][WorldBuilder][reredact] %s: %s; keeping the stored keyframe",
                       keyframe_id, out.detail)
        return out
    box = _dilate(new.max(axis=2) <= FILL_BLACK, OUTPUT_BOX_PX)
    changed = int(((_absdiff(new, raw) > OUTPUT_DIFF) & ~box).sum())
    if changed:
        out.origin = KEPT_OUTPUT_NOT_RAW
        out.detail = f"{changed} px differ from the raw frame away from any fill"
        return out
    out.recovered_px = int((stored_fill & ~new_fill).sum())
    if out.recovered_px < RECOVERED_MIN_PX:
        out.origin = KEPT_NOTHING_RECOVERED
        out.detail = None
        return out
    out.origin, out.detail = ORIGIN_REREDACTED, None
    out.image_bytes = candidate
    out.sha1 = hashlib.sha1(candidate).hexdigest()
    return out


# ---------------------------------------------------------------------------
# a session
# ---------------------------------------------------------------------------


def set_name(label: str = TARGET_LABEL) -> str:
    """`images.redacted-<gate>`: short, because Windows paths are."""
    from tower.world_builder.store import REREDACTED_IMAGES_PREFIX  # noqa: PLC0415

    gate = label.rsplit("+", 1)[-1] if "+" in label else label.rsplit("@", 1)[-1]
    slug = "".join(c if c.isalnum() or c in "-." else "-" for c in gate).strip("-.")
    return REREDACTED_IMAGES_PREFIX + (slug or "unnamed")


def frames_digest(pairs) -> str:
    h = hashlib.sha1()
    for kid, sha in sorted(pairs):
        h.update(f"{kid}\t{sha}\n".encode())
    return h.hexdigest()


@dataclass
class Plan:
    world_id: str
    session_id: str
    stored_redaction: str | None
    set_label: str
    name: str
    redactor_label: str
    frames: list = field(default_factory=list)
    refusal: str | None = None      # why an --apply of this plan would not switch
    stored_digest: str | None = None
    seconds: float = 0.0

    @property
    def counts(self) -> dict:
        c: dict = {}
        for f in self.frames:
            c[f.origin] = c.get(f.origin, 0) + 1
        return dict(sorted(c.items()))

    @property
    def recovered_frames(self) -> list:
        return [f for f in self.frames if f.reredacted]

    def totals(self) -> dict:
        measured = [f for f in self.frames if f.stored_fill_px is not None]
        pixels = sum(f.pixels for f in measured)
        stored = sum(f.stored_fill_px for f in measured)
        after = sum(f.effective_fill_px or 0 for f in measured)
        return {
            "frames": len(self.frames),
            "frames_measured": len(measured),
            "frames_reredacted": len(self.recovered_frames),
            "pixels_measured": pixels,
            "stored_fill_px": stored,
            "set_fill_px": after,
            "recovered_px": sum(f.recovered_px for f in self.recovered_frames),
            "stored_fill_fraction": (stored / pixels) if pixels else None,
            "set_fill_fraction": (after / pixels) if pixels else None,
        }


def _check_label(stored: str | None) -> None:
    if stored == TARGET_LABEL:
        raise ReredactionRefused(
            f"the session is already redacted under the current rule ({TARGET_LABEL}); "
            "nothing to recover")
    if stored not in REREDACTABLE_LABELS:
        raise ReredactionRefused(
            f"the session's redaction label {stored!r} is not an older rule of the "
            f"{_YUNET} family. Its stored keyframes may themselves be unredacted, so they "
            "are never re-redacted from raw; builds re-redact the stored bytes instead "
            "(WORLD-BUILDER-APPEARANCE.md section 6.2)")


def _check_redactor(redactor) -> str:
    if redactor is None or not getattr(redactor, "available", False):
        raise ReredactionRefused(
            "no face redactor is available "
            f"({getattr(redactor, 'unavailable_reason', None)}); nothing was switched")
    label = getattr(redactor, "label", None)
    if label != TARGET_LABEL:
        raise ReredactionRefused(
            f"the redactor's label is {label!r}, but this step only writes sets under "
            f"{TARGET_LABEL!r}; add the new gate to reredaction.py deliberately")
    return label


def plan_session(store, world_id: str, session_id: str, *, redactor=None,
                 tower_root=None, progress=None) -> Plan:
    """Run the current redactor over every keyframe's verified raw frame, in
    memory. Writes nothing. Raises `ReredactionRefused` when the step cannot run
    at all; a plan that ran but must not be applied carries `refusal`."""
    from tower.world_builder.global_solve import (  # noqa: PLC0415
        read_sources,
        resolve_source_path,
        workspace_for,
    )

    t0 = time.time()
    world = store.read_world(world_id)
    if getattr(world, "images_purged", False):
        raise ReredactionRefused("this world's keyframe imagery was purged")
    session = store.read_session(world_id, session_id)
    stored = session.redaction if isinstance(session.redaction, str) and session.redaction else None
    _check_label(stored)
    if session.ended_at is None:
        raise ReredactionRefused(
            "the session has not ended; a live walk is still writing keyframes "
            "(new captures are redacted under the current rule already)")
    if redactor is None:
        from tower.world_builder.redaction import FaceRedactor  # noqa: PLC0415

        redactor = FaceRedactor()
    redactor_label = _check_redactor(redactor)

    keyframes = store.read_keyframes(world_id, session_id)
    if not keyframes:
        raise ReredactionRefused("the session has no keyframes")
    sources = read_sources(workspace_for(store, world_id, session_id))
    images = store.images_dir(world_id, session_id)
    plan = Plan(world_id=world_id, session_id=session_id, stored_redaction=stored,
                set_label=TARGET_LABEL, name=set_name(TARGET_LABEL),
                redactor_label=redactor_label)
    pairs = []
    for n, kf in enumerate(keyframes):
        if progress is not None and n % 50 == 0:
            progress(n, len(keyframes))
        path = store.session_dir(world_id, session_id) / kf.image_relpath
        if path.parent != images:
            raise ReredactionRefused(
                f"keyframe {kf.keyframe_id} is not under the session's images/ ({kf.image_relpath})")
        try:
            stored_bytes = path.read_bytes()
        except OSError as exc:
            raise ReredactionRefused(
                f"stored keyframe {kf.keyframe_id} is unreadable ({exc}); a set cannot "
                "be written with a frame missing") from None
        raw_path = resolve_source_path(sources.get(kf.keyframe_id), tower_root)
        outcome = reredact_frame(kf.keyframe_id, stored_bytes, raw_path, redactor)
        pairs.append((kf.keyframe_id, outcome.stored_sha1))
        plan.frames.append(outcome)
    plan.stored_digest = frames_digest(pairs)

    uncovered = [f for f in plan.frames
                 if not f.reredacted and f.origin not in _KEPT_AT_TARGET]
    contradicted = [f for f in uncovered if f.origin in _KEPT_CONTRADICTS_TARGET]
    if contradicted and plan.recovered_frames:
        # THE SET'S LABEL MUST BE TRUE FOR EVERY FRAME IN IT. These frames were
        # measured NOT to meet the target rule, so no label this step can write
        # is honest for a set that holds them; refused, whatever `stored` is.
        plan.refusal = (
            f"{len(contradicted)} frame(s) would keep stored keyframes that the current rule "
            f"was measured not to be contained in ({contradicted[0].keyframe_id} "
            f"{contradicted[0].origin}: {contradicted[0].detail}), so a set labelled "
            f"{TARGET_LABEL!r} could not honestly hold them")
    elif uncovered and not REREDACTABLE_LABELS[stored]:
        plan.refusal = (
            f"{len(uncovered)} frame(s) would keep keyframes redacted under {stored!r}, "
            f"which does not fill everything {TARGET_LABEL!r} fills, so the set could not "
            "honestly carry the current label (first: "
            f"{uncovered[0].keyframe_id} {uncovered[0].origin})")
    elif not plan.recovered_frames:
        plan.refusal = "no frame has recoverable fill; nothing to switch"
    plan.seconds = time.time() - t0
    return plan


def _hash_dir(directory: Path, names) -> dict:
    return {n: hashlib.sha1((directory / n).read_bytes()).hexdigest() for n in names}


def _stored_digest_now(store, world_id, session_id) -> str:
    pairs = []
    for kf in store.read_keyframes(world_id, session_id):
        data = (store.session_dir(world_id, session_id) / kf.image_relpath).read_bytes()
        pairs.append((kf.keyframe_id, hashlib.sha1(data).hexdigest()))
    return frames_digest(pairs)


def _record(plan: Plan, set_digest: str, redactor) -> dict:
    from tower.world_builder import redaction as R  # noqa: PLC0415

    model = None
    try:
        path = R.model_path()
        model = path.name if path is not None else None
    except Exception:  # noqa: BLE001 -- a record field, never a failure
        model = None
    return {
        "format": RECORD_FORMAT,
        "tool_version": TOOL_VERSION,
        "created_at": time.time(),
        "world_id": plan.world_id,
        "session_id": plan.session_id,
        "stored_redaction": plan.stored_redaction,
        "redaction": plan.set_label,
        "redactor_label": plan.redactor_label,
        "redactor_model": model,
        "jpeg_quality": R.JPEG_QUALITY,
        "verification_rule": VERIFY_RULE,
        "fill_rule": f"image<={FILL_BLACK} and |image-raw|>{FILL_DIFF}",
        "invariant": (f"new fill inside the stored fill dilated {INVARIANT_TOLERANCE_PX}px; "
                      f"no |new-raw|>{OUTPUT_DIFF} further than {OUTPUT_BOX_PX}px from new fill; "
                      f"at least {RECOVERED_MIN_PX}px recovered"),
        "stored_digest": plan.stored_digest,
        "set_digest": set_digest,
        "counts": plan.counts,
        "totals": plan.totals(),
        "frames": [f.as_record() for f in plan.frames],
    }


def apply_plan(store, plan: Plan, *, redactor=None) -> dict:
    """Write the set beside `images/`, verify it, and switch the pointer LAST.

    Refuses (nothing switched) on a plan refusal, an active switch, a world held
    by a live writer, a stored keyframe that changed since planning, or any
    failure before the pointer is written."""
    from tower.storage import write_json_atomic  # noqa: PLC0415
    from tower.world_builder.store import WorldLockedError  # noqa: PLC0415

    if plan.refusal:
        raise ReredactionRefused(plan.refusal)
    w, s = plan.world_id, plan.session_id
    try:
        store.acquire_writer_lock(w)
    except WorldLockedError as exc:
        raise ReredactionRefused(f"the world is held by a live writer ({exc})") from None
    try:
        pointer = store.read_redaction_set_pointer(w, s) or {}
        if pointer.get("active"):
            raise ReredactionRefused(
                f"the session already reads {pointer['active']!r}; --revert first")
        if _stored_digest_now(store, w, s) != plan.stored_digest:
            raise ReredactionRefused("a stored keyframe changed since the plan was made")
        session_dir = store.session_dir(w, s)
        final = session_dir / plan.name
        names = [f"{f.seq}.jpg" for f in plan.frames]
        set_digest = frames_digest((f.keyframe_id, f.sha1) for f in plan.frames)
        reused = False
        if final.exists():
            # A set from an earlier apply, switched away from. Re-point to it
            # only if it is exactly this plan's; never overwrite or delete it.
            import json  # noqa: PLC0415

            try:
                record = json.loads((final / RECORD_FILENAME).read_text(encoding="utf-8"))
                on_disk = _hash_dir(final, names)
            except (OSError, ValueError):
                record, on_disk = None, None
            expected = {f"{f.seq}.jpg": f.sha1 for f in plan.frames}
            if not (record and record.get("set_digest") == set_digest
                    and record.get("stored_digest") == plan.stored_digest
                    and on_disk == expected):
                raise ReredactionRefused(
                    f"{final} already exists and is not this plan's set; move it aside "
                    "(it is never overwritten or deleted)")
            reused = True
        else:
            staging = session_dir / f"{plan.name}.partial-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            staging.mkdir(parents=False)
            for f in plan.frames:
                with open(staging / f"{f.seq}.jpg", "wb") as handle:
                    handle.write(f.image_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            write_json_atomic(staging / RECORD_FILENAME, _record(plan, set_digest, redactor))
            os.replace(staging, final)
        written = _hash_dir(final, names)
        if written != {f"{f.seq}.jpg": f.sha1 for f in plan.frames}:
            raise ReredactionRefused(f"{final} does not hold the bytes the plan wrote")
        if _stored_digest_now(store, w, s) != plan.stored_digest:
            raise ReredactionRefused("a stored keyframe changed while the set was written")
        history = list(pointer.get("history") or [])
        history.append({"at": time.time(), "action": "apply", "set": plan.name,
                        "set_digest": set_digest, "reused": reused})
        new_pointer = {
            "format": POINTER_FORMAT,
            "schema_version": 1,
            "active": plan.name,
            "redaction": plan.set_label,
            "stored_redaction": plan.stored_redaction,
            "set_digest": set_digest,
            "record": f"{plan.name}/{RECORD_FILENAME}",
            "tool_version": TOOL_VERSION,
            "switched_at": time.time(),
            "history": history,
        }
        # LAST, and atomic: before this line every reader still reads images/.
        store.write_redaction_set_pointer(w, s, new_pointer)
        _discard_area_builds(store, w)
        logger.info("[Tower][WorldBuilder][reredact] %s/%s now reads %s (%s)",
                    w, s, plan.name, plan.set_label)
        return {"set": plan.name, "set_digest": set_digest, "reused": reused,
                "pointer": str(store.redaction_set_path(w, s))}
    finally:
        store.release_writer_lock(w)


def _discard_area_builds(store, world_id: str) -> None:
    """A switch of the keyframe set invalidates the imagery an unfinished area build
    holds outside the world (`area_build.purge_area_builds`, `<root>/.ab/<area>`);
    removed here, except a build whose process is still alive -- none can be, since a
    switch holds the world's writer lock, which every area build holds too. Never
    fatal: the switch has happened, and the next area build starts clean anyway."""
    try:
        from tower.world_builder.area_build import purge_area_builds  # noqa: PLC0415

        purge_area_builds(store, world_id, only_stale=True)
    except Exception:  # noqa: BLE001
        logger.warning("[Tower][WorldBuilder][reredact] could not discard the area builds of %s",
                       world_id, exc_info=True)


def revert_session(store, world_id: str, session_id: str) -> dict:
    """Switch back to the capture's own keyframes. A pointer change only: the
    re-redacted set stays on disk, and a later --apply of the same plan re-points
    to it."""
    from tower.world_builder.store import WorldLockedError  # noqa: PLC0415

    pointer = store.read_redaction_set_pointer(world_id, session_id)
    if not pointer or not pointer.get("active"):
        raise ReredactionRefused("the session already reads its stored keyframes")
    try:
        store.acquire_writer_lock(world_id)
    except WorldLockedError as exc:
        raise ReredactionRefused(f"the world is held by a live writer ({exc})") from None
    try:
        previous = pointer["active"]
        history = list(pointer.get("history") or [])
        history.append({"at": time.time(), "action": "revert", "set": previous})
        reverted = dict(pointer)
        reverted.update({"active": None, "switched_at": time.time(), "history": history})
        store.write_redaction_set_pointer(world_id, session_id, reverted)
        _discard_area_builds(store, world_id)
        logger.info("[Tower][WorldBuilder][reredact] %s/%s reads images/ again (was %s)",
                    world_id, session_id, previous)
        return {"reverted_from": previous}
    finally:
        store.release_writer_lock(world_id)
