"""The dense stage, orchestrated over a solved world.

Four stages, each checkpointed under `<world>/dense/<session>/`:

    depth   undistort every posed keyframe, predict inverse depth, fit it to
            the sparse points, and write the per-frame alignment record
    fuse    mask, cross-check against neighbouring cameras, average the
            agreeing positions, and voxel-reduce
    pack    write the LOD ladder and the manifest

A stage whose output is already present and matches the current input digest is
skipped, so an interrupted run resumes. `should_stop` is polled between frames
and between stages; a stop is recorded as `stopped`, never as a failure and
never as a half-written artifact.

Nothing here writes into `derived/`. The dense subtree is additive and optional:
`require_schema` refuses any version but 1 and there is no migration machinery,
so a reader that does not know about `dense/` must be able to ignore it, exactly
as it ignores `solve/`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from tower.world_builder.dense import (
    DENSE_FORMAT,
    STAGE_DEPTH,
    STAGE_FUSE,
    STAGE_PACK,
    STATE_FAILED,
    STATE_OK,
    STATE_RUNNING,
    STATE_STOPPED,
    STATE_UNAVAILABLE,
    DenseParams,
    DenseResult,
    DenseUnavailable,
    align_frame,
    camera_centre,
    depth_from_prediction,
    make_backend,
    project,
    unproject,
    validity_mask,
    POINT_STRIDE_BYTES,
    voxel_reduce,
    write_points_bin,
)
from tower.world_builder.raw_imagery import IMAGERY_REDACTED, RAW_NOTE, is_raw

logger = logging.getLogger(__name__)

DENSE_SCHEMA_VERSION = 1


def _pid_is_running(pid: int) -> bool:
    """The store's own liveness probe, not os.kill.

    The signal-based probe is a console-signal call on Windows and reported a
    freshly dead process as still alive in testing, which would strand a lock
    forever. On any failure this answers "running", because refusing to start
    is recoverable and stealing a live lock is not.
    """
    from tower.world_builder.store import _pid_is_running as _probe  # noqa: PLC0415

    try:
        return bool(_probe(pid))
    except Exception:  # noqa: BLE001
        return True



# Parameters that do not change a single point of the output. Comparing them
# when deciding whether an artifact is already what was asked for made
# `--keep-intermediates` -- the flag the operations doc calls the one for the
# development loop -- miss the short-circuit every time.
_NON_OUTPUT_PARAMS = frozenset({"keep_intermediates"})


def _output_params(params: dict) -> dict:
    return {k: v for k, v in (params or {}).items() if k not in _NON_OUTPUT_PARAMS}


def _params_match(stored: dict, wanted: dict) -> bool:
    """Do a stored artifact's parameters agree with the ones being asked for?

    OVER THE KEYS BOTH SIDES KNOW, and that is the whole point. Comparing whole
    dicts means every parameter ever added silently invalidates every artifact
    already on disk: `pack_percentile` was added and seven of the eight worlds
    in this corpus stopped being recognised as complete, so re-running densify
    on a finished, current world reported `unavailable` and overwrote its
    status with no result -- exactly the destruction the completed-artifact
    guard was written to prevent, arriving by a different door.

    A key the stored artifact does not carry is a key that did not exist when
    it was packed. It cannot have changed the output, because the code that
    produced the output never read it. A key whose value differs is a real
    difference and still refuses.
    """
    stored = _output_params(stored)
    wanted = _output_params(wanted)
    shared = stored.keys() & wanted.keys()
    if not shared:
        return False
    return all(stored[k] == wanted[k] for k in shared)


def _depth_cache_key(digest, params: DenseParams, image_set: str | None = None,
                     trust: str | None = None) -> str:
    """Everything the DEPTH stage reads.

    `component` belongs here: re-running with a different one used to reuse the
    other component's predictions and then write a manifest claiming the new
    one. A `digest` of None must not match every solve either.

    `image_set` is `store.keyframe_image_set(...).cache_token`: None for the
    capture's own `images/` -- so every key written before re-redaction existed
    is unchanged -- and the re-redacted set's name and digest after a switch,
    so a switch (or a switch back) refits rather than reusing the other set's
    depth and fill masks. The solve digest cannot see a switch: the solve reads
    the raw frames, not either set.
    """
    fields = [digest, params.backend, params.component, params.min_sparse_points,
              f"fill{FILL_RULE}"]
    # §6.6, and only when it is NOT the product, so every key ever written
    # for a redacted build is unchanged and still matches.
    if is_raw(params.imagery_source):
        fields.append(f"imagery:{params.imagery_source}")
    if image_set:
        fields.append(f"set:{image_set}")
    # `trust` is `appearance.pixel_trust_token`: whether the stage used the
    # stored bytes or redacted them again, and under which labels. A walk's
    # depth stage runs under `none` (re-redacted bytes, their SHA-1 in every
    # record); the final one after Stop runs under the real label (the stored
    # bytes). The solve digest cannot see that change, and reusing the walk's
    # records made the trusted final appearance refuse every frame whose bytes
    # the re-redaction had changed (review 1, M3).
    if trust:
        fields.append(f"trust:{trust}")
    # `DenseParams.known_fov`: a prediction conditioned on the solve camera's
    # FoV is another prediction. Appended only when on, so every key written
    # without it is unchanged.
    if getattr(params, "known_fov", False):
        fields.append("fov:known")
    return "|".join(str(x) for x in fields)


def recorded_trust(align: dict | None) -> str | None:
    """The `pixel_trust_token` an `align.json` was produced under, or None
    when it cannot be established. A record from before the token existed is
    read from what it did record: it used the stored bytes exactly when it
    said so, and that is still the trusted token only if its label is on
    today's allowlist (an old `+plausibility2` stage trusted a label that is
    no longer trusted, so it is not reused)."""
    from tower.world_builder.appearance import label_is_trusted, pixel_trust_token  # noqa: PLC0415

    if not isinstance(align, dict):
        return None
    token = align.get("redaction_trust")
    if isinstance(token, str) and token:
        return token
    label = align.get("redaction")
    if align.get("keyframes_were_redacted_at_capture") is True and label_is_trusted(label):
        return pixel_trust_token(label)
    return None


def depth_trust_now(store, world_id: str, session_id: str, redactor=None,
                    imagery_source: str = IMAGERY_REDACTED) -> str:
    """The token a depth stage run NOW would record: the keyframe set's label
    through the one allowlist, and the current redactor's label when that
    label is not trusted. `FaceRedactor()` is cheap (its model loads lazily).

    Under the bypass (§6.6) the label is irrelevant and no redactor is loaded:
    the token names the imagery source instead, and shares no spelling with
    either redacted one."""
    from tower.world_builder.appearance import label_is_trusted, pixel_trust_token  # noqa: PLC0415

    if is_raw(imagery_source):
        return pixel_trust_token(None, None, imagery_source=imagery_source,
                                 raw_token=imagery_source)
    label = store.keyframe_image_set(world_id, session_id).redaction
    if label_is_trusted(label):
        return pixel_trust_token(label)
    if redactor is None:
        from tower.world_builder.redaction import FaceRedactor  # noqa: PLC0415

        redactor = FaceRedactor()
    return pixel_trust_token(label, getattr(redactor, "label", None)
                             if getattr(redactor, "available", False) else None)


def depth_cache_matches(cached: dict | None, digest, params: DenseParams,
                        image_set: str | None, trust: str) -> bool:
    """Whether a cached `align.json` is the depth stage of this solve, these
    parameters, this keyframe set AND this trust decision. A key written before
    the trust token existed matches when its recorded trust equals `trust`."""
    if not isinstance(cached, dict) or recorded_trust(cached) != trust:
        return False
    # §6.6, checked on the record itself and not only through the key: a
    # record written before the bypass existed has no `imagery_source` and is
    # a redacted one, which is what it was.
    if cached.get("imagery_source", IMAGERY_REDACTED) != params.imagery_source:
        return False
    return cached.get("cache_key") in (_depth_cache_key(digest, params, image_set, trust),
                                       _depth_cache_key(digest, params, image_set))


def _fuse_cache_key(digest, params: DenseParams, image_set: str | None = None) -> str:
    """Everything the FUSE stage reads, the depth stage's keyframe set included."""
    fields = [
        digest, params.backend, params.component, params.gate_rel, params.tau,
        params.min_views, params.neighbours, params.stride, params.edge_rel,
        params.max_grazing_deg, params.erode_px, params.average_views,
        params.max_depth_pct, params.max_extrapolation,
    ]
    if image_set:
        fields.append(f"set:{image_set}")
    return "|".join(str(x) for x in fields)


def dense_dir(store, world_id: str, session_id: str) -> Path:
    """`<world>/dense/<session>` -- beside `solve/`, never inside `derived/`."""
    return store.world_dir(world_id) / "dense" / session_id


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def _status(root: Path, **fields) -> None:
    """Write the stage's state where a cold reader can find it.

    `pid` is recorded on every state, not only the running one, because the
    supervisor can kill this process after finalization is already marked
    complete -- the dense stage deliberately outlives the world lock. Without a
    pid, a run killed mid-stage leaves `state: "running"` on disk forever and
    nothing can tell that from a run that is genuinely still going.
    """
    # CARRY THE DIGEST FORWARD. `status.json` is one file and every later run
    # overwrites it, so a failed or interrupted re-run used to erase the
    # `input_digest` the completing run wrote -- and that value is what arms
    # the BEHIND caption for any artifact whose manifest predates the key.
    # Losing it silently disables a correctness signal, so a write that does
    # not know the digest inherits the one already on disk.
    payload = {
        "schema_version": DENSE_SCHEMA_VERSION,
        "pid": os.getpid(),
        "updated_at": time.time(),
        **fields,
    }
    if payload.get("input_digest") is None:
        try:
            prior = json.loads((root / "status.json").read_text())
        except (OSError, ValueError):
            prior = {}
        if prior.get("input_digest"):
            payload["input_digest"] = prior["input_digest"]
    _write_json(root / "status.json", payload)


def status_is_stale(status: dict) -> bool:
    """True when a status says `running` but its process is gone -- or its pid
    now belongs to a process that started after the status was written.

    A bare pid probe here left `live: true` on `/render/revision` for as long
    as a recycled pid lived after a densify was killed (review 2, I2). The
    locks and the surface status already asked the store's question; the dense
    status now does too, from `updated_at`, which `_status` writes on every
    state.
    """
    if not status or status.get("state") != STATE_RUNNING:
        return False
    pid = status.get("pid")
    if not isinstance(pid, int):
        return True
    from tower.world_builder.store import _holder_is_running  # noqa: PLC0415

    written = status.get("updated_at")
    if not isinstance(written, (int, float)):
        written = None
    try:
        return not _holder_is_running(pid, None, lock_written_at=written)
    except Exception:  # noqa: BLE001 -- "running" is the recoverable answer
        return False


def _stopped(should_stop) -> bool:
    return bool(should_stop and should_stop())


class _DenseLock:
    """One densify at a time per session.

    The dense stage runs after `world_build_session.py` has released the world
    writer lock -- following the `--register` precedent, and deliberately, since
    holding it for the minutes this takes would block a new capture on the same
    world. That is safe against the world's other writers but not against another
    densify of the same session, which is easy to start by accident: run
    `scripts/world_densify.py` while a build is finalising. Two runs interleaving
    their writes would leave a points file of the wrong length, which does not
    fail loudly.

    `dense/<session>/` IS touched by something else, and this lock alone never
    covered it (review V9, Q8): the surface's depth stage and the evidence gate's
    (`coherence_publish.run_gate_depth`) write the same `work/`, `align.json` and
    `predictions/` under the SURFACE lock (`surface_pipeline._SurfaceLock`), which
    a densify did not take -- so a hand-run densify beside a finisher's re-gate or
    an owner's re-finish of the session could overwrite the gate's `<ki>_pred.npy`
    with predictions made under other parameters before its metric scale read them,
    or prune `work/` from under it. `densify` now holds that lock too, for its whole
    run: the one lock every writer of `dense/<session>/` holds.
    """

    def __init__(self, root: Path) -> None:
        self.path = root / ".densify.lock"
        self.held = False

    def _stale(self) -> bool:
        try:
            pid = int(json.loads(self.path.read_text()).get("pid", -1))
        except (OSError, ValueError, AttributeError):
            return True
        # Deliberately NOT treating our own pid as stale: that would let a
        # second lock object in the same process steal the first one's, which
        # defeats the point. The probe is the store's own, because
        # the signal-based probe is a console-signal call on Windows and reports a
        # freshly dead process as still alive -- which would strand the lock
        # for good.
        # Same recycled-pid guard as the surface lock: a process that started
        # after this lock was written did not write it.
        from tower.world_builder.store import _holder_is_running  # noqa: PLC0415

        try:
            written = self.path.stat().st_mtime
        except OSError:
            return True
        return not _holder_is_running(pid, None, lock_written_at=written)

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not self._stale():
                    return False
                try:
                    self.path.unlink()
                except OSError:
                    return False
                continue
            with os.fdopen(fd, "w") as handle:
                json.dump({"pid": os.getpid(), "at": time.time()}, handle)
            self.held = True
            return True
        return False

    def release(self) -> None:
        if not self.held:
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        self.held = False


# ---------------------------------------------------------------------------


# The label `FaceRedactor` returns when nothing was applied -- including
# when it returned the ORIGINAL bytes because the detector threw.
from tower.world_builder.redaction import REDACTION_NONE  # noqa: E402

# The research bypass's two outcomes (WORLD-BUILDER-APPEARANCE.md §6.6).
# Deliberately not spelled like any `world-keyframe` origin: a record carrying
# one of these cannot be skimmed as a redacted build's.
ORIGIN_RAW_LOCAL = "raw-local-capture-frame"
ORIGIN_RAW_MISSING = "refused-raw-frame-missing"


class DenseInputsPruned(DenseUnavailable):
    """Asked to run from intermediates that a successful run removed.

    A DenseUnavailable rather than a bug: the artifact is complete, its
    intermediates were pruned on purpose, and the caller wants a stage that
    consumes them. The answer is --force, and saying so beats a traceback
    ending in `depth/00081.npy`.
    """


def _source_paths(workspace_root: Path) -> dict[str, str]:
    p = workspace_root / "sources.json"
    if not p.exists():
        return {}
    try:
        return dict(json.loads(p.read_text()).get("sources") or {})
    except (OSError, ValueError):
        return {}


def _undistorted_image(ki: int, work: Path):
    import cv2

    p = work / "undist" / f"{ki:05d}.jpg"
    return cv2.imread(str(p)) if p.exists() else None


# WHICH RULE PRODUCED A FRAME'S `_fill.npy`. Part of the depth cache key and of
# every per-frame record, so a mask made under an earlier rule -- and the
# prediction made alongside it -- is never reused under this one.
#
#   (absent)  the difference against whatever image was re-redacted. A live
#             build during a walk re-redacts the STORED keyframe, which already
#             carries the engine's fill, so the difference was empty wherever
#             the engine had filled a face -- and the final build after Stop
#             reused that empty mask by image hash (review 2, I6).
#   2         the fill is measured against what the CAMERA saw: the raw capture
#             frame when it is readable, otherwise the exact difference of this
#             re-redaction united with the shape-gated guess on the image
#             itself, which is what a build that never re-redacted uses.
FILL_RULE = 2


def _fill_mask_for(redacted: bytes, raw: bytes | None, stored: bytes | None = None):
    """Which pixels of `redacted` are redaction fill rather than camera pixels.

    Exactly, by difference against the raw frame, when that is readable. Else,
    when `stored` is given -- the keyframe on disk that `redacted` was produced
    from by a re-redaction -- the difference against it (this re-redaction's
    own fill, exactly) united with the shape-gated guess on `redacted` (the
    fill that was already in `stored`, which no difference against `stored`
    can see). Else None, and the caller guesses on the image, as it always has.
    Undilated either way; the caller dilates.
    """
    import cv2

    if raw is None and stored is None:
        return None
    b = cv2.imdecode(np.frombuffer(redacted, np.uint8), cv2.IMREAD_COLOR)
    if b is None:
        return None
    if raw is not None:
        a = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if a is not None and a.shape == b.shape:
            return redaction_fill_mask(b, a, dilate_px=0)
    if stored is None:
        return None
    guess = redaction_fill_mask(b, None, dilate_px=0)
    if stored is redacted or stored == redacted:
        # Nothing was filled by this re-redaction; all of the fill predates it.
        return guess
    s = cv2.imdecode(np.frombuffer(stored, np.uint8), cv2.IMREAD_COLOR)
    if s is None or s.shape != b.shape:
        return guess
    return guess | redaction_fill_mask(b, s, dilate_px=0)


def keyframe_image_bytes(store, world_id: str, session_id: str, keyframe_id: str,
                         source_path: str | None, redactor, *,
                         keyframes_are_redacted: bool,
                         image_set=None,
                         imagery_source: str = IMAGERY_REDACTED,
                         ) -> tuple[bytes | None, str, "np.ndarray | None"]:
    """The pixels the dense stage is allowed to read, and where they came from.

    THIS IS A PRIVACY BOUNDARY, not a convenience. `engine.py` redacts faces
    BEFORE persisting a keyframe image, deliberately, so that the bytes any
    later reconstruction reads are the redacted ones rather than raw frames
    sitting on disk behind a display filter. A dense stage that reached past
    that to the original capture would rebuild the room out of exactly the
    pixels the privacy transformation removed, at far higher density than the
    sparse cloud ever exposed.

    So: the world's own redacted keyframe image is used when it exists AND the
    session record says it was redacted. That second condition is not
    decoration. `FaceRedactor.redact` returns the original bytes unchanged when
    the redactor is unavailable or throws, labelled `none`, and
    `engine._persist_keyframe` persists whatever comes back -- so `images/` can
    legitimately hold raw frames, and `session.redaction` is the only record
    that says which. An earlier version of this function asserted the boundary
    in this docstring and then read that directory unconditionally.

    `keyframes_are_redacted` is that record, resolved by the caller from
    `store.read_session(...).redaction`. When it is false the stored keyframe is
    treated exactly like a raw frame: THE REDACTION IS APPLIED HERE before
    anything looks at it, and the frame is refused if no redactor is available.

    When the keyframe image is missing entirely -- worlds migrated between roots
    lost their `images/` directory -- the raw source frame is read and the same
    redaction applied. If the redactor is unavailable, the frame is refused
    rather than used raw.

    Returns (image_bytes, origin, fill_mask). The mask is computed inside this
    function, from the difference against the raw frame, and the raw bytes are
    dropped here -- so raw pixels never leave the boundary, not even as an
    argument to the caller.

    WHICH `images/`: `store.keyframe_image_set` decides -- the capture's own
    keyframes, or the re-redacted set `world_reredact.py --apply` switched the
    session to -- and `keyframes_are_redacted` must come from that SAME set's
    label. `image_set` is the set, resolved once per stage by the caller so a
    switch mid-stage cannot mix two sets; None resolves it here.

    `source_path` is a `sources.json` entry, resolved against the Tower root
    (`global_solve.resolve_source_path`), never against the process cwd.

    THE LABEL IS CHECKED HERE TOO, through the appearance stage's allowlist
    (`appearance.label_is_trusted`): stored bytes are returned as
    `world-keyframe` only when the caller says the set is redacted AND the
    set's own label is on the allowlist. `label != "none"` trusted any string,
    the weak `+plausibility2` gate included, and served one session under two
    trust decisions (review 1, M2; privacy lane L2).

    `imagery_source` is the ONE way past all of the above, and it is
    `redacted` unless a caller asked otherwise (WORLD-BUILDER-APPEARANCE.md
    §6.6). Under `raw-local-research` this returns the original local frame
    itself with an origin that says so and NO fill mask, because there is no
    fill: the caller must then neither inpaint nor mask, and the records it
    writes carry the same source, so no redacted build reuses them.
    """
    from tower.world_builder.appearance import label_is_trusted
    from tower.world_builder.global_solve import resolve_source_path

    seq = keyframe_id.rsplit(":", 1)[-1]
    raw_bytes = None
    resolved = resolve_source_path(source_path)
    if resolved is not None and resolved.exists():
        try:
            raw_bytes = resolved.read_bytes()
        except OSError:
            raw_bytes = None
    if is_raw(imagery_source):
        if raw_bytes is None:
            return None, ORIGIN_RAW_MISSING, None
        return raw_bytes, ORIGIN_RAW_LOCAL, None
    if image_set is None:
        image_set = store.keyframe_image_set(world_id, session_id)
    keyframes_are_redacted = bool(keyframes_are_redacted) and label_is_trusted(image_set.redaction)
    p = image_set.directory / f"{seq}.jpg"
    if p.exists():
        try:
            # The fill mask is computed HERE, from the difference, and the raw
            # bytes are dropped on the way out -- so raw pixels never leave this
            # function even as an argument. Guessing the fill from the redacted
            # image alone runs at 36.2% precision, which means two thirds of
            # what it deletes is real scene.
            data = p.read_bytes()
            if keyframes_are_redacted:
                return data, "world-keyframe", _fill_mask_for(data, raw_bytes)
            # The session record says these pixels were never redacted. Do it
            # now rather than republish faces at ~50x the density the sparse
            # cloud ever exposed.
            if redactor is None or not getattr(redactor, "available", False):
                return None, "refused-unredacted-keyframe", None
            result = redactor.redact(data)
            if result.label == REDACTION_NONE:
                # `redact` NEVER RAISES: it returns the ORIGINAL bytes, labelled
                # `none`, when the detector throws on this particular image.
                # `available` is a load-time property and cannot see that. An
                # earlier version of this check tested the redactor and then
                # used whatever came back, which is the same mistake one level
                # down from the one it was written to fix.
                return None, "refused-redaction-failed", None
            filled = result.image_bytes
            # Against the RAW frame, not against `data`: during a walk the
            # session record says `none` although the engine has usually
            # filled the faces already, and a difference against the stored
            # image cannot see a fill both images share (FILL_RULE).
            return (filled, "world-keyframe-redacted-here",
                    _fill_mask_for(filled, raw_bytes, stored=data))
        except OSError:
            pass
    if raw_bytes is None:
        return None, ("absent" if not source_path else "unreadable"), None
    if redactor is None or not getattr(redactor, "available", False):
        return None, "refused-no-redactor", None
    result = redactor.redact(raw_bytes)
    if result.label == REDACTION_NONE:
        # Same trap on the fallback path, and this is the path the migrated
        # worlds actually take -- 506 of the corpus's frames went through it.
        return None, "refused-redaction-failed", None
    data = result.image_bytes
    return data, "raw-source-rereducted", _fill_mask_for(data, raw_bytes)


def redaction_fill_mask(image, raw=None, fill_value: int = 0,
                        min_area_fraction: float = 0.0015, dilate_px: int = 3,
                        rect_fill_min: float = 0.85):
    """Which pixels are redaction fill rather than scene.

    A filled rectangle is not an observation. A depth network handed one will
    happily invent a surface across it, and multi-view consensus will not
    always catch it, because the SAME detector fires on the SAME object from
    several nearby frames -- so several cameras agree on geometry that is
    really a black box. This is not hypothetical: on this corpus the face
    detector fires on the wearer's hands and on carpet, filling over 10% of
    the frame in 22 of 77 frames and up to 58% in the worst one.

    Excluding those pixels is both the honest choice and the one that recovers
    quality: the region is unobserved, so it should be a hole.

    When the raw image is available the mask is exact (a difference). When only
    the redacted image survives, large near-`fill_value` connected regions are
    used, which is what a solid fill leaves behind.
    """
    import cv2

    h, w = image.shape[:2]
    if raw is not None and raw.shape[:2] == (h, w):
        mask = (np.abs(image.astype(np.int16) - raw.astype(np.int16)).sum(-1) > 12)
    else:
        # Redaction fills an axis-aligned RECTANGLE with a constant value. A
        # dark bed, a shadowed floor and an unlit wall are all near-black too,
        # and without the shape test this ran at 36.2% precision -- measured
        # against the exact difference over 120 frames, two thirds of what it
        # deleted was real scene, 3.6 million pixels of it. Requiring the
        # component to fill its own bounding box takes precision to 88.2%.
        flat = (image.max(-1) <= fill_value + 8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            flat.astype(np.uint8), connectivity=4
        )
        mask = np.zeros((h, w), bool)
        min_area = max(64, int(min_area_fraction * h * w))
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if area < min_area or bw * bh == 0:
                continue
            if area / float(bw * bh) < rect_fill_min:
                continue
            mask |= labels == i
    if dilate_px:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                          iterations=dilate_px).astype(bool)
    return mask


# ---------------------------------------------------------------------------
# THE PREDICTION CACHE (review V8 H2, part R3: a re-finish must reproduce its depth).
#
# The evidence gate's metric scale reads this stage's raw per-keyframe predictions
# (`<ki>_pred.npy`), and they did not survive a finish: the room's final surface prunes
# `work/` (`prune_intermediates`) once its appearance is built, and the gate's hand-off
# cuts `align.json` to the room, so a re-finish -- and every draw of a consensus --
# predicted every frame again on the GPU. A borderline scale decision (6839fb8f's g6 at
# x1.2526 against the x1.25 bound) could then flip on the recomputation.
#
# With `prediction_cache=True` (the gate's depth stage only: `coherence_publish.run_gate_depth`)
# the network's output is kept in `<dense>/predictions/<token>/<sha1>.npy`, OUTSIDE
# `work/`, keyed by the SHA-1 of the exact pixels the network is shown (after
# undistortion, the redaction fill and its inpainting) and by `token`, the network and
# its parameters (`prediction_token`). Everything else -- the image read, the fill, the
# fit to the solve -- is computed as before, so the alignment is always this solve's;
# only the network call is replaced. A fresh prediction is rounded through float16, the
# precision it is stored in, BEFORE it is fitted, so a frame fitted from a fresh
# prediction and one fitted from the cache are fitted identically. Off (every other
# caller): the stage is byte for byte what it was.
#
# THE CACHE IS AN OPTIMISATION, NEVER A REASON TO LOSE THE DEPTH (review V9, M-11 and LOW).
# A write that fails (MAX_PATH, a full disk, a sharing violation) is counted
# (`prediction_cache.write_failed`) and the stage goes on with the float16-rounded prediction
# it already has; it used to raise, and the gate published a scale-unavailable fail-safe --
# how P3-H2 run A lost its depth. A cache file that exists but is not a whole float16 array
# of the frame's shape (empty, truncated, foreign, wrong shape) is a MISS, logged, and the
# network's fresh prediction replaces it; an empty file used to raise `EOFError` for ever.
#
# GROWTH, MEASURED, AND ITS BOUND (review V9, M-12). The cache of 6839fb8f in
# `RUN/experiments/P3-H2/real/r0` holds 678 frames in 311,154,540 bytes (about 459 kB a
# frame: float16 at 359x639), about 20x the keyframe images. Nothing reclaimed it:
# `prune_intermediates` keeps it on purpose, only `purge_world` removed it, and every change
# of the pixels (a re-redaction switch, the raw-imagery bypass) or of the network's call
# (its token) added a whole new set. Now, after every depth stage that COMPLETES with the
# cache on, `prune_prediction_cache` keeps exactly one set: the current token's directory,
# and in it one prediction per keyframe id -- the latest pixels each keyframe was predicted
# from, recorded in the token's own index (`PREDICTION_INDEX`). Removed, and recorded in
# `prediction_cache.pruned`: every other token's set, a prediction no keyframe's latest
# pixels name any more, and a dead writer's staging file. The index is per keyframe, not
# per `align.json`, on purpose: the draws of a consensus pose different keyframes and each
# rewrites `align.json`, so pruning by the last draw's record would remove the predictions
# of the frames only the chosen draw posed -- the very frames a re-gate of it reads. The
# bound is therefore the walk's keyframes, once. A re-finish's set-aside copy leaves
# `predictions/` out (`world_refinish.py`).
PREDICTIONS_DIRNAME = "predictions"
# Schema 2: the token also names the weights revision, the moge and torch versions, the
# device and fp16 (review V9, LOW). Every schema-1 set is another token's set and is pruned.
PREDICTION_CACHE_SCHEMA = 2
PREDICTION_INDEX = "index.json"
# SHORT NAMES, BECAUSE OF MAX_PATH. A world root under a Tower's data directory is already
# ~135 characters deep at `dense/<session>/`, and the atomic write adds `.p<pid>.<nonce>.tmp`:
# a 40-hex name under a 16-hex token measured 263 characters on a scratch copy and failed
# (`FileNotFoundError`), which the gate took as "no depth". 12 + 20 hex keeps the whole
# relative name, staging suffix included, near 70 characters -- the transient-mask cache's
# convention (`solve_masks.component_path`, SHA-1[:12]). 80 bits of the pixel digest is far
# beyond any collision a walk's few thousand frames could meet.
PREDICTION_TOKEN_HEX = 12
PREDICTION_INPUT_HEX = 20


def prediction_token(backend, fov_x) -> tuple[str, dict]:
    """(token, what it stands for): the network and every parameter of the call that
    changes its output, beyond the pixels. `PREDICTION_TOKEN_HEX` hex.

    Beyond the network's name and call: the weights revision the hub resolves it to (when
    it can be read), the moge and torch versions, the device and whether the call runs in
    fp16 (`token_runtime`). MoGe's `infer` autocasts to float16 by default, per device
    type, so a prediction made on the CPU, or by another release, is another prediction
    (review V9, LOW)."""
    doc = {"schema": PREDICTION_CACHE_SCHEMA, "backend": getattr(backend, "name", None),
           "model_id": getattr(backend, "model_id", None), "kind": getattr(backend, "kind", None),
           "resolution_level": getattr(backend, "resolution_level", None),
           "fov_x": None if fov_x is None else round(float(fov_x), 6),
           **token_runtime(backend)}
    token = hashlib.sha1(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:PREDICTION_TOKEN_HEX]
    return token, doc


def token_runtime(backend) -> dict:
    """What the prediction was made WITH, beyond the network's name: `weights_revision`,
    `moge`, `torch`, `device`, `fp16`. A backend's own attribute wins (`revision`,
    `device`, `use_fp16`, or the loaded MoGe backend's `_revision` / `_device`); else what
    this process would load and run it on. None where it cannot be known. Never raises."""
    revision = getattr(backend, "revision", None) or getattr(backend, "_revision", None)
    if revision is None:
        revision = _hub_revision(getattr(backend, "model_id", None))
    device = getattr(backend, "device", None) or getattr(backend, "_device", None)
    if device is None:
        device = _default_device()
    fp16 = getattr(backend, "use_fp16", None)
    if fp16 is None and _is_moge(backend):
        fp16 = _moge_infer_fp16()
    return {"weights_revision": revision, "moge": _package_version("moge"),
            "torch": _package_version("torch"), "device": None if device is None else str(device),
            "fp16": None if fp16 is None else bool(fp16)}


def _is_moge(backend) -> bool:
    return (type(backend).__name__ == "MoGeBackend"
            or str(getattr(backend, "model_id", "") or "").startswith("Ruicheng/moge"))


def _hub_revision(model_id) -> str | None:
    """The commit the hub cache resolves `model_id`'s `main` to -- what `from_pretrained`
    without a revision loads -- or None."""
    if not model_id:
        return None
    try:
        from tower.world_builder.dense import hub_model_cache  # noqa: PLC0415

        cache = hub_model_cache(str(model_id))
        ref = cache / "refs" / "main" if cache is not None else None
        if ref is None or not ref.is_file():
            return None
        return ref.read_text(encoding="utf-8").strip() or None
    except Exception:  # noqa: BLE001 -- unknown is a field of the token, not an error
        return None


_RUNTIME_CACHE: dict = {}


def _package_version(dist: str) -> str | None:
    if dist not in _RUNTIME_CACHE:
        try:
            from importlib.metadata import version  # noqa: PLC0415

            _RUNTIME_CACHE[dist] = version(dist)
        except Exception:  # noqa: BLE001
            _RUNTIME_CACHE[dist] = None
    return _RUNTIME_CACHE[dist]


def _default_device() -> str | None:
    """The device a depth backend loaded now would run on: the backends' own rule."""
    if "device" not in _RUNTIME_CACHE:
        try:
            import torch  # noqa: PLC0415

            _RUNTIME_CACHE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            _RUNTIME_CACHE["device"] = None
    return _RUNTIME_CACHE["device"]


def _moge_infer_fp16() -> bool | None:
    """MoGe's `infer(use_fp16=...)` default, which `MoGeBackend.predict` relies on."""
    if "moge_fp16" not in _RUNTIME_CACHE:
        try:
            import inspect  # noqa: PLC0415

            from moge.model.v2 import MoGeModel  # noqa: PLC0415

            default = inspect.signature(MoGeModel.infer).parameters["use_fp16"].default
            _RUNTIME_CACHE["moge_fp16"] = None if default is inspect.Parameter.empty else bool(default)
        except Exception:  # noqa: BLE001
            _RUNTIME_CACHE["moge_fp16"] = None
    return _RUNTIME_CACHE["moge_fp16"]


def prediction_cache_dir(root: Path, backend, fov_x) -> Path:
    token, _doc = prediction_token(backend, fov_x)
    return Path(root) / PREDICTIONS_DIRNAME / token


def network_input_sha1(rgb: np.ndarray) -> str:
    """The SHA-1 of the pixels a depth network is shown (shape and dtype included),
    `PREDICTION_INPUT_HEX` hex."""
    rgb = np.ascontiguousarray(rgb)
    h = hashlib.sha1(f"{rgb.shape}|{rgb.dtype.str}|".encode())
    h.update(rgb.tobytes())
    return h.hexdigest()[:PREDICTION_INPUT_HEX]


def _cached_prediction(path: Path, shape=None) -> np.ndarray | None:
    """The kept prediction at `path` as float32, or None: a miss. Anything but a whole
    float16 array of `shape` (the network input's height and width) is a miss -- an empty
    file (`EOFError`), a truncated or foreign one, an archive, a wrong shape -- and is
    logged; the network runs and its prediction replaces the file (review V9, LOW)."""
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 -- EOFError, ValueError, a zip error: a miss
        _log_bad_prediction(path, f"{type(exc).__name__}: {exc}")
        return None
    if not isinstance(arr, np.ndarray):
        close = getattr(arr, "close", None)
        if close is not None:
            close()
        _log_bad_prediction(path, "it is not an array")
        return None
    if arr.dtype != np.float16 or arr.ndim != 2 or (shape is not None and arr.shape != tuple(shape)):
        _log_bad_prediction(path, f"a {arr.dtype} array of shape {arr.shape}, not float16 of "
                                  f"{None if shape is None else tuple(shape)}")
        return None
    return arr.astype(np.float32)


def _log_bad_prediction(path: Path, why: str) -> None:
    logger.warning("[Tower][WorldBuilder][dense] the kept prediction %s is unusable (%s); it is a "
                   "miss, predicted again and rewritten", Path(path).name, why)


def _cache_prediction(path: Path, disp16: np.ndarray) -> None:
    from tower.storage import write_bytes_atomic  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(path, lambda handle: np.save(handle, disp16))


def _read_prediction_index(token_dir: Path) -> dict | None:
    """{keyframe id: input digest} of a token's set, {} when it has none, None when it
    exists and cannot be read (nothing is pruned on the strength of an unreadable index)."""
    path = Path(token_dir) / PREDICTION_INDEX
    try:
        if not path.is_file():
            return {}
        doc = json.loads(path.read_text(encoding="utf-8"))
        keyframes = doc.get("keyframes") if isinstance(doc, dict) else None
        if not isinstance(keyframes, dict):
            return None
        return {str(k): str(v) for k, v in keyframes.items()}
    except (OSError, ValueError, TypeError):
        return None


def prune_prediction_cache(root: Path, token: str, current: dict) -> dict:
    """Keep ONE prediction set: `token`'s, and in it the latest prediction of every
    keyframe (see "GROWTH" above). `current` is {keyframe id: input digest} of the depth
    stage that just COMPLETED. Returns what was removed. Never raises.

    Removed: every other token's directory (its predictions, its index, its dead writers'
    staging files; a live writer's file is left, and so is its directory); in `token`'s,
    every prediction that no keyframe's latest input names, and every dead writer's
    staging file. The index is updated first and written atomically; when it cannot be
    written, or an existing one cannot be read, nothing in `token`'s set is removed."""
    from tower.storage import sweep_abandoned_staging, write_json_atomic  # noqa: PLC0415

    base = Path(root) / PREDICTIONS_DIRNAME
    out = {"sets": 0, "files": 0, "bytes": 0, "staging": 0}

    def remove(path: Path) -> bool:
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            return False
        out["files"] += 1
        out["bytes"] += int(size)
        return True

    try:
        others = [d for d in base.iterdir() if d.is_dir() and d.name != token] if base.is_dir() else []
    except OSError:
        others = []
    for other in others:
        try:
            out["staging"] += sweep_abandoned_staging(other)
            for f in list(other.iterdir()):
                if f.is_file() and f.suffix == ".npy":
                    remove(f)
            (other / PREDICTION_INDEX).unlink(missing_ok=True)
            other.rmdir()
            out["sets"] += 1
        except OSError:
            # A live writer's staging file, a file held open, or a file this cache never
            # writes (not ours to judge): the directory stays, and is tried next time.
            continue
    here = base / token
    if not here.is_dir():
        return out
    try:
        out["staging"] += sweep_abandoned_staging(here)
    except OSError:
        pass
    index = _read_prediction_index(here)
    if index is None:
        out["kept_because"] = "the set's index is unreadable"
        return out
    index.update({str(k): str(v) for k, v in current.items()})
    try:
        write_json_atomic(here / PREDICTION_INDEX, {"schema": PREDICTION_CACHE_SCHEMA, "token": token,
                                                   "keyframes": dict(sorted(index.items()))})
    except OSError:
        out["kept_because"] = "the set's index could not be written"
        return out
    named = set(index.values())
    try:
        files = [f for f in here.iterdir() if f.is_file() and f.suffix == ".npy"]
    except OSError:
        files = []
    for f in files:
        if f.stem not in named:
            remove(f)
    return out


def run_depth_stage(
    store, world_id: str, session_id: str, solution, intrinsics, params: DenseParams,
    root: Path, *, should_stop=None, progress: Callable[[str, int, int], None] | None = None,
    prior: dict | None = None,
    reuse_predictions: dict | None = None,
    prediction_cache: bool = False,
) -> dict:
    """Undistort, predict depth, and align every posed keyframe in the component.

    The undistortion is the SOLVE's own -- `global_solve._undistort_maps` -- so
    the depth map and the poses are expressed in exactly the same camera. Doing
    this any other way silently shifts every back-projected point.

    `prediction_cache`: keep and reuse the network's raw predictions (see
    `PREDICTIONS_DIRNAME`); off is the stage as it always was.
    """
    import cv2

    from tower.world_builder.global_solve import _undistort_maps

    work = root / "work"
    (work / "undist").mkdir(parents=True, exist_ok=True)
    (work / "depth").mkdir(parents=True, exist_ok=True)

    cam = solution.camera or {}
    K = np.array(
        [[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], float
    )
    W, H = int(cam["width"]), int(cam["height"])
    sources = _source_paths(store.world_dir(world_id) / "solve" / session_id)
    # The solve workspace also holds undistorted `images/`, but those are the
    # RAW frames COLMAP was fed by some workspaces, not the world's redacted
    # keyframes, so the dense stage deliberately does not read them.
    from tower.world_builder.redaction import FaceRedactor

    from tower.world_builder.redaction import REDACTION_NONE

    # What the SESSION says about the pixels already on disk, not what a
    # redactor loaded now would do to them. `engine.py` persists whatever
    # `redact` returns, including the original bytes when redaction was
    # unavailable at capture time, so `images/` is only trustworthy when this
    # says so. Anything unrecognised is treated as unredacted.
    #
    # Read through the store's one accessor, ONCE: after a re-redaction switch
    # the images, and the label that describes them, are the re-redacted set's.
    image_set = store.keyframe_image_set(world_id, session_id)
    session_redaction = image_set.redaction
    # The one allowlist (`appearance.label_is_trusted`), not `!= "none"`: the
    # depth stage's pixels become `undist/`, the surface's vertex colours, the
    # dense points and the appearance proxy, and must be trusted exactly when
    # the appearance stage trusts the same set (review 1, M2).
    from tower.world_builder.appearance import label_is_trusted, pixel_trust_token

    keyframes_are_redacted = label_is_trusted(session_redaction)

    # §6.6. Under the bypass no redactor is loaded, because none is used: the
    # frames this stage reads were never redacted and the records say so.
    raw_imagery = is_raw(params.imagery_source)
    redactor = None if raw_imagery else FaceRedactor()
    trust = pixel_trust_token(
        session_redaction,
        None if keyframes_are_redacted or redactor is None
        else (getattr(redactor, "label", None) if redactor.available else None),
        imagery_source=params.imagery_source,
        raw_token=params.imagery_source)
    if raw_imagery:
        logger.warning(
            "[Tower][WorldBuilder][dense] session %s: building depth from the "
            "ORIGINAL local capture frames (%s). No redaction fill is measured, "
            "nothing is inpainted and nothing is masked for privacy.",
            session_id, RAW_NOTE)
    if not keyframes_are_redacted and not raw_imagery:
        logger.warning(
            "[Tower][WorldBuilder][dense] session %s records redaction=%r; its "
            "stored keyframes are NOT trusted as redacted and will be redacted "
            "here before any pixel is read",
            session_id, session_redaction,
        )
    if redactor is not None and not redactor.available:
        logger.warning(
            "[Tower][WorldBuilder][dense] face redaction unavailable (%s); frames "
            "whose redacted keyframe image is missing will be REFUSED rather than "
            "read raw", redactor.unavailable_reason,
        )
    origins: dict[str, int] = {}

    backend = make_backend(params.backend)
    # `DenseParams.known_fov`: the solve camera's horizontal FoV, for a backend
    # that can be told it. None -- the call today's stage makes -- otherwise.
    fov_x = None
    if getattr(params, "known_fov", False) and getattr(backend, "accepts_fov", False):
        fov_x = float(np.degrees(2.0 * np.arctan(W / (2.0 * float(cam["fx"])))))
    fov_record = {"known_fov": round(fov_x, 6)} if fov_x is not None else {}
    cache_dir = cache_record = None
    # {keyframe id: network input digest} of every frame this stage predicted or read from
    # the cache: the set's index, for `prune_prediction_cache`.
    cache_inputs: dict = {}
    if prediction_cache:
        token, token_doc = prediction_token(backend, fov_x)
        cache_dir = Path(root) / PREDICTIONS_DIRNAME / token
        cache_record = {"token": token, "network": token_doc, "hits": 0, "predicted": 0,
                        "write_failed": 0}
    kids = solution.keyframe_ids
    targets = []
    for i, kid in enumerate(kids):
        pose = solution.poses.get(kid)
        if not pose or pose.get("rotation") is None or pose.get("translation") is None:
            continue
        if int(pose.get("component", 0)) != params.component:
            continue
        targets.append((i, kid, pose))

    maps = None
    map_shape = None
    # Per-FRAME resume. A stop halfway through a 429-frame world should cost
    # only the frames not yet reached. The earlier version discarded a stopped
    # stage wholesale and re-predicted every frame, with its own prediction
    # files sitting unread on disk beside it.
    done = {}
    if prior:
        for rec in prior.get('records') or []:
            ki_prev = rec.get('ki')
            if ki_prev is None:
                continue
            if not rec.get('ok'):
                done[int(ki_prev)] = rec
            elif (work / 'depth' / ('%05d.npy' % int(ki_prev))).exists():
                done[int(ki_prev)] = rec
    if done:
        logger.info('[Tower][WorldBuilder][dense] resuming depth: %d frames already done',
                    len(done))
    # PREDICTIONS ARE REUSABLE ACROSS SOLVES; FITS ARE NOT.
    #
    # `prior` above resumes whole RECORDS, and a record carries the frame's
    # affine fit to the solve's sparse points -- valid only for the solve it
    # was fitted against. `reuse_predictions` is the other half:
    # `{ki: (kid, image_sha1)}` of frames whose raw network output on disk is
    # known to be for that keyframe's exact image from this backend. The network's output depends only on the
    # keyframe image, so it is loaded instead of re-predicted and the fit is
    # computed afresh against THIS solve. During a walk each live surface is
    # built from a new solve over more keyframes; without this, every one of
    # them re-predicted every frame -- 200 s of GPU on a 400-keyframe walk --
    # and with only a digest check it silently kept the old fits instead.
    reuse_predictions = reuse_predictions or {}
    reused = 0
    records: list[dict] = []
    t0 = time.time()
    obs_kf = solution.observations[:, 0]
    obs_pt = solution.observations[:, 2]

    for n, (ki, kid, pose) in enumerate(targets):
        cached_rec = done.get(int(ki))
        if cached_rec is not None:
            records.append(cached_rec)
            origins['resumed'] = origins.get('resumed', 0) + 1
            continue
        if _stopped(should_stop):
            return {"stopped_after": n, "records": records, "seconds": time.time() - t0,
                    "camera": cam, "targets": len(targets), "image_origins": origins,
                    "kind": backend.kind, "fill_rule": FILL_RULE,
                    "keyframe_image_set": image_set.cache_token,
                    "imagery_source": params.imagery_source,
                    "redaction_trust": trust, **fov_record}
        if progress and n % 25 == 0:
            progress(STAGE_DEPTH, n, len(targets))

        pred_path = work / "depth" / f"{ki:05d}_pred.npy"
        fill_path = work / "depth" / f"{ki:05d}_fill.npy"
        undist_path = work / "undist" / f"{ki:05d}.jpg"

        data, origin, exact_fill = keyframe_image_bytes(
            store, world_id, session_id, kid, sources.get(kid), redactor,
            keyframes_are_redacted=keyframes_are_redacted, image_set=image_set,
            imagery_source=params.imagery_source,
        )
        image_sha1 = hashlib.sha1(data).hexdigest() if data is not None else None
        # A prediction is reused only for the SAME IMAGE, not merely the same
        # keyframe id: a keyframe re-redacted or replaced since would otherwise
        # keep the old pixels' depth, fill mask and colour.
        offered = reuse_predictions.get(int(ki))
        if (image_sha1 is not None and offered == (kid, image_sha1)
                and pred_path.exists() and fill_path.exists() and undist_path.exists()):
            disp = np.load(pred_path).astype(np.float32)
            fill_u = np.load(fill_path)
            fill_fraction = float(fill_u.mean())
            origins["reused-prediction"] = origins.get("reused-prediction", 0) + 1
            reused += 1
            record = _fit_record(
                ki, kid, pose, disp, fill_u, fill_fraction, "reused-prediction",
                solution, obs_kf, obs_pt, K, W, H, params, backend, work)
            record["image_sha1"] = image_sha1
            record["fill_rule"] = FILL_RULE
            records.append(record)
            continue
        origins[origin] = origins.get(origin, 0) + 1
        if data is None:
            records.append({"ki": int(ki), "ok": False, "why": f"image {origin}"})
            continue
        raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            records.append({"ki": int(ki), "ok": False, "why": "image undecodable"})
            continue
        if maps is None:
            m1, m2, roi, _pin = _undistort_maps(intrinsics, raw.shape[1], raw.shape[0])
            maps = (m1, m2, roi)
            map_shape = raw.shape[:2]
        elif raw.shape[:2] != map_shape:
            # DAT can change resolution mid-stream. The rectification maps are
            # built once, for one size; remapping a different one would
            # silently reconstruct in the wrong camera.
            records.append({"ki": int(ki), "ok": False,
                            "why": f"frame is {raw.shape[1]}x{raw.shape[0]}, "
                                   f"the session's maps are for "
                                   f"{map_shape[1]}x{map_shape[0]}"})
            continue
        m1, m2, (x0, y0, rw, rh) = maps
        if (rw, rh) != (W, H):
            raise DenseUnavailable(
                f"undistorted ROI is {rw}x{rh} but the solve camera is {W}x{H}; "
                "the dense stage refuses to reconstruct in a camera the poses "
                "were not solved in"
            )
        img = cv2.remap(raw, m1, m2, cv2.INTER_LINEAR)[y0:y0 + rh, x0:x0 + rw]
        cv2.imwrite(str(work / "undist" / f"{ki:05d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        # The exact mask when the raw frame was available, and only then the
        # shape-gated guess. Under the bypass (§6.6) there is nothing to mask:
        # an EMPTY mask is written rather than none at all, because a missing
        # `_fill.npy` means "unknown" to three later readers and this one is
        # known -- these pixels are the camera's, entire.
        if raw_imagery:
            fill_u = np.zeros((rh, rw), bool)
        else:
            fill = (cv2.dilate(exact_fill.astype(np.uint8), np.ones((3, 3), np.uint8),
                               iterations=3).astype(bool)
                    if exact_fill is not None
                    else redaction_fill_mask(raw, None))
            fill_u = cv2.remap(fill.astype(np.uint8) * 255, m1, m2,
                               cv2.INTER_NEAREST)[y0:y0 + rh, x0:x0 + rw] > 0
        np.save(work / "depth" / f"{ki:05d}_fill.npy", fill_u)
        fill_fraction = float(fill_u.mean())
        if params.inpaint_redaction_fill and fill_u.any():
            # Give the network a continuous image instead of a hole. What
            # comes back inside the hole is invention and is thrown away by
            # the mask above; what this buys is the REST of the frame.
            img = cv2.inpaint(img, fill_u.astype(np.uint8), 5, cv2.INPAINT_TELEA)
            cv2.imwrite(str(work / "undist" / f"{ki:05d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        disp = cached_path = None
        if cache_dir is not None:
            input_sha1 = network_input_sha1(rgb)
            cached_path = cache_dir / f"{input_sha1}.npy"
            cache_inputs[kid] = input_sha1
            disp = _cached_prediction(cached_path, rgb.shape[:2])
            if disp is not None:
                cache_record["hits"] += 1
        if disp is None:
            disp = (backend.predict(rgb, fov_x=fov_x) if fov_x is not None
                    else backend.predict(rgb))
            if cached_path is not None:
                # Fitted at the precision it is kept in, so a later build that reads
                # it from the cache fits it exactly as this one does.
                disp16 = np.asarray(disp).astype(np.float16)
                disp = disp16.astype(np.float32)
                cache_record["predicted"] += 1
                try:
                    _cache_prediction(cached_path, disp16)
                except OSError as exc:
                    # Never a reason to lose the depth (review V9, M-11): the stage goes
                    # on with the float16 prediction it has, and says it kept none.
                    cache_record["write_failed"] += 1
                    if cache_record["write_failed"] == 1:
                        logger.warning("[Tower][WorldBuilder][dense] could not keep a depth "
                                       "prediction (%s: %s); the depth stage goes on without "
                                       "keeping it", type(exc).__name__, exc)
        # Saved BEFORE the fit, under its own name, so a frame whose fit fails
        # against this solve -- too few sparse points yet -- still has its
        # prediction for the next solve to fit against.
        np.save(pred_path, disp.astype(np.float16))
        record = _fit_record(
            ki, kid, pose, disp, fill_u, fill_fraction, origin, solution,
            obs_kf, obs_pt, K, W, H, params, backend, work)
        record["image_sha1"] = image_sha1
        record["fill_rule"] = FILL_RULE
        records.append(record)

    if progress:
        progress(STAGE_DEPTH, len(targets), len(targets))
    if reused:
        logger.info("[Tower][WorldBuilder][dense] depth: %d of %d predictions "
                    "reused from an earlier solve and refitted", reused, len(targets))
    payload = {"records": records, "seconds": time.time() - t0, "camera": cam,
               "targets": len(targets), "backend": backend.name,
               "backend_licence": backend.licence, "kind": backend.kind,
               "stopped_after": None,
               "fill_rule": FILL_RULE,
               "keyframe_image_set": image_set.cache_token,
               "imagery_source": params.imagery_source,
               "redaction_trust": trust,
               "image_origins": origins,
               "redaction": session_redaction,
               "keyframes_were_redacted_at_capture": keyframes_are_redacted and not raw_imagery,
               "redactor_applied_here": (
                   getattr(redactor, "label", None)
                   if redactor is not None and redactor.available else None),
               **fov_record}
    if cache_record is not None:
        # The stage COMPLETED: keep one set, the latest prediction of each keyframe (M-12).
        try:
            cache_record["pruned"] = prune_prediction_cache(root, cache_record["token"], cache_inputs)
        except Exception as exc:  # noqa: BLE001 -- pruning a cache never fails its stage
            logger.warning("[Tower][WorldBuilder][dense] could not prune the kept predictions "
                           "(%s: %s)", type(exc).__name__, exc)
            cache_record["pruned"] = {"failed": type(exc).__name__}
        payload["prediction_cache"] = cache_record
    _write_json(root / "align.json", payload)
    return payload


def _fit_record(ki, kid, pose, disp, fill_u, fill_fraction, origin, solution,
                obs_kf, obs_pt, K, W, H, params, backend, work) -> dict:
    """Fit one frame's depth prediction to THIS solve's sparse points.

    Separated from prediction so that a prediction made during an earlier
    solve can be fitted again against a later one. The body is the depth
    stage's original fit, moved and unchanged in what it computes.
    """
    m = obs_kf == ki
    if int(m.sum()) < params.min_sparse_points:
        return {"ki": int(ki), "kid": kid, "ok": False,
                "why": f"only {int(m.sum())} sparse observations"}
    uv = solution.observation_xy[m].astype(np.float64)
    R = np.asarray(pose["rotation"], float).reshape(3, 3)
    t = np.asarray(pose["translation"], float)
    _, zc = project(R, t, K, solution.xyz[obs_pt[m]].astype(np.float64))
    g = ((zc > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1)
         & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1))
    if int(g.sum()) < params.min_sparse_points:
        return {"ki": int(ki), "kid": kid, "ok": False,
                "why": f"only {int(g.sum())} sparse points inside the frame"}
    ui = np.clip(np.rint(uv[g, 0]).astype(int), 0, W - 1)
    vi = np.clip(np.rint(uv[g, 1]).astype(int), 0, H - 1)
    # THE FIT MUST NOT BE ANCHORED ON PIXELS NOBODY OBSERVED.
    #
    # `fill_u` marks what the face redactor blacked out and what this stage
    # then handed to an inpainter. Masking those pixels out of the CLOUD
    # afterwards -- which `run_fuse_stage` does -- removes the invented
    # points but not the invented FIT they produced, and (a, b) is a global
    # per-frame scale and offset applied to every surviving pixel. A fit
    # derived from invention was being applied to the real scene.
    #
    # It was not rare. On the widest traverse, 25.3% of fit anchors landed
    # in inpainted pixels on average, 30 gate-passing frames had over half
    # their anchors there, and nine had essentially all of them -- one at
    # 1.000, whose stored keyframe is entirely black, and which scored a
    # 1.88% held-out residual against the world's 2.9% median.
    #
    # THE GATE COULD NOT SEE IT, AND PREFERRED IT. Both halves of the
    # held-out split come from the same anchors in the same invented
    # region, and a TELEA inpaint is a smooth interpolant that an affine
    # model fits very well -- so more invention scored better. Re-anchoring
    # those 30 frames on clean points alone moves the depth by a median
    # 12.8% and a maximum of 467%, and three of them flip to a negative `a`,
    # the value the fusion stage explicitly refuses.
    clean = ~fill_u[vi, ui].astype(bool)
    n_clean = int(clean.sum())
    if n_clean < params.min_sparse_points:
        return {"ki": int(ki), "kid": kid, "ok": False,
                "why": (f"only {n_clean} sparse anchors outside the "
                                "redaction fill"),
                "anchors_total": int(g.sum()),
                "redaction_fill_fraction": fill_fraction}
    ui, vi = ui[clean], vi[clean]
    zc_fit = zc[g][clean]
    a, b, ho = align_frame(disp[vi, ui].astype(np.float64), zc_fit, backend.kind)
    # float16: the depth values run 0.2-40 in world units and the pipeline's
    # own error is a few percent, so three significant digits is far more
    # than the evidence supports -- and it halves the largest thing this
    # stage writes.
    np.save(work / "depth" / f"{ki:05d}.npy", disp.astype(np.float16))
    return {"ki": int(ki), "kid": kid, "ok": True, "a": a, "b": b,
                    "n_points": int(g.sum()), "held_out_rel": ho,
                    # From the CLEAN anchors, for the same reason the fit
                    # is: a bound computed from invented pixels bounds
                    # nothing.
                    "z_sparse_min": float(np.min(zc_fit)),
                    "z_sparse_max": float(np.max(zc_fit)),
                    "anchors_used": int(len(zc_fit)),
                    "anchors_in_fill": int(g.sum()) - int(len(zc_fit)),
                    "redaction_fill_fraction": fill_fraction,
                    "image_origin": origin}


def reusable_predictions(align_path: Path, backend: str,
                         imagery_source: str = IMAGERY_REDACTED,
                         known_fov: bool = False) -> dict:
    """`{ki: kid}` for every frame an earlier depth stage predicted with this
    backend, read off its `align.json`. Empty when there is none or it cannot
    be read; the depth stage then predicts every frame, which is slower and
    never wrong.

    THE IMAGE HASH IS NOT ENOUGH ACROSS §6.6. A keyframe the redactor found
    nothing in has a stored image byte-identical to its original frame, so
    the hashes match -- but the redacted stage may still have INPAINTED it,
    because the fill mask is a shape-gated guess with false positives, and
    the prediction it cached was made on invented pixels. Measured on the
    canonical world: 215 of 395 frames offered a hash-matching prediction to
    a raw build. So a prediction crosses modes only when both stages read the
    same kind of imagery.
    """
    try:
        cached = json.loads(align_path.read_text())
    except (OSError, ValueError):
        return {}
    if cached.get("backend") != backend:
        return {}
    if cached.get("imagery_source", IMAGERY_REDACTED) != imagery_source:
        return {}
    # Nor across `DenseParams.known_fov`: a prediction told the camera's FoV and
    # one left to estimate it are different predictions of the same image.
    # Absent is off, which every stage before the parameter was.
    if (cached.get("known_fov") is not None) != bool(known_fov):
        return {}
    # (kid, image hash): a record written before hashes were recorded offers
    # nothing, and costs one fresh prediction rather than a wrong reuse. Nor
    # does one whose fill mask was made under an earlier FILL_RULE: its
    # `_fill.npy` may be the empty mask of review 2's I6, and the reuse would
    # carry it into every later build of the session.
    return {int(r["ki"]): (r["kid"], r["image_sha1"])
            for r in cached.get("records") or []
            if isinstance(r, dict) and r.get("kid") and r.get("ki") is not None
            and r.get("image_sha1") and r.get("fill_rule") == FILL_RULE}


def run_fuse_stage(
    solution, align: dict, params: DenseParams, root: Path,
    *, should_stop=None, progress=None,
) -> dict:
    """Mask, cross-check against neighbouring cameras, average, voxel-reduce."""
    import cv2

    work = root / "work"
    cam = align["camera"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], float)
    W, H = int(cam["width"]), int(cam["height"])

    recs = {r["ki"]: r for r in align["records"] if r.get("ok")}
    kept = sorted(
        ki for ki, r in recs.items()
        if r.get("held_out_rel") is not None
        and r["held_out_rel"] <= params.gate_rel
        and r["a"] > 0
    )
    dropped = len(recs) - len(kept)
    if len(kept) < 3:
        raise DenseUnavailable(
            f"only {len(kept)} of {len(recs)} frames pass the {params.gate_rel:.0%} "
            "alignment gate; there is not enough reliable depth to fuse"
        )

    poses = {}
    for ki in kept:
        p = solution.poses[recs[ki]["kid"]]
        poses[ki] = (np.asarray(p["rotation"], float).reshape(3, 3),
                     np.asarray(p["translation"], float))

    D, VALID = {}, {}
    kind = align.get("kind", "disparity")
    for ki in kept:
        r = recs[ki]
        depth_path = work / "depth" / f"{ki:05d}.npy"
        if not depth_path.exists():
            # Pruned, or hand-deleted. Either way the depth stage's output is
            # gone and fusing without it would silently drop the frame.
            raise DenseInputsPruned(
                "the per-frame depth maps for this session are not on disk "
                "(a successful run prunes them). Re-run with --force."
            )
        pred = np.load(depth_path).astype(np.float32)
        # One function decides what a stored map means, shared with the
        # alignment and the scoring, so the three cannot drift apart.
        z = depth_from_prediction(pred, r["a"], r["b"], kind).astype(np.float32)
        z = np.where(np.isfinite(z) & (z > 1e-6), z, np.nan).astype(np.float32)
        D[ki] = z
        ok = validity_mask(z, K, edge_rel=params.edge_rel,
                           max_grazing_deg=params.max_grazing_deg,
                           erode_px=params.erode_px)
        # Refuse depth the frame's own sparse points never bracketed, past a
        # stated margin. This is what makes "interpolates between points the
        # solve earned" true rather than merely nearly true.
        zlo, zhi = r.get("z_sparse_min"), r.get("z_sparse_max")
        if zlo and zhi and params.max_extrapolation > 0:
            ok &= (z >= zlo / params.max_extrapolation) & (z <= zhi * params.max_extrapolation)
        fillp = work / "depth" / f"{ki:05d}_fill.npy"
        if fillp.exists():
            # Redaction fill is unobserved, so it stays a hole.
            ok &= ~np.load(fillp)
        VALID[ki] = ok

    sample = np.concatenate([d[np.isfinite(d)][::37] for d in D.values()])
    zmax = float(np.percentile(sample, params.max_depth_pct))
    # Every length below is expressed against this, because the world's gauge
    # is arbitrary and differs by more than an order of magnitude between solves.
    median_depth = float(np.median(sample))
    voxels = params.voxels_for(median_depth)

    C = np.array([camera_centre(*poses[ki]) for ki in kept])
    d2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nbrs = np.argsort(d2, axis=1)[:, : params.neighbours]
    idx_of = {ki: i for i, ki in enumerate(kept)}

    us, vs = np.meshgrid(np.arange(0, W, params.stride), np.arange(0, H, params.stride))
    us, vs = us.ravel(), vs.ravel()
    uv0 = np.stack([us + 0.5, vs + 0.5], 1).astype(np.float64)

    acc_x, acc_c, acc_f = [], [], []
    raw_total = keep_total = 0
    t0 = time.time()
    for n, ki in enumerate(kept):
        if _stopped(should_stop):
            return {"stopped_after": n, "seconds": time.time() - t0}
        if progress and n % 25 == 0:
            progress(STAGE_FUSE, n, len(kept))
        R, t = poses[ki]
        zs = D[ki][vs, us]
        m = (np.isfinite(zs) & (zs > 1e-3) & (zs < zmax) & VALID[ki][vs, us])
        if not m.any():
            continue
        X = unproject(R, t, K, uv0[m], zs[m].astype(np.float64))
        raw_total += len(X)
        agree = np.zeros(len(X), np.int16)
        Xsum = X.copy() if params.average_views else None
        for j in nbrs[idx_of[ki]]:
            kj = kept[j]
            Rj, tj = poses[kj]
            uvj, zpred = project(Rj, tj, K, X)
            ok = ((zpred > 1e-3) & (uvj[:, 0] >= 0) & (uvj[:, 0] < W - 1)
                  & (uvj[:, 1] >= 0) & (uvj[:, 1] < H - 1))
            if not ok.any():
                continue
            uj = np.rint(uvj[ok, 0]).astype(np.int32).clip(0, W - 1)
            vj = np.rint(uvj[ok, 1]).astype(np.int32).clip(0, H - 1)
            # Nearest, never bilinear: interpolating across a depth
            # discontinuity invents a depth no pixel actually holds.
            zj = np.where(VALID[kj][vj, uj], D[kj][vj, uj], np.nan)
            rel = np.abs(zj - zpred[ok]) / np.maximum(zpred[ok], 1e-6)
            good = np.isfinite(rel) & (rel < params.tau)
            if not good.any():
                continue
            sel = np.where(ok)[0][good]
            agree[sel] += 1
            if params.average_views:
                uvg = np.stack([uj[good] + 0.5, vj[good] + 0.5], 1).astype(np.float64)
                Xsum[sel] += unproject(Rj, tj, K, uvg, zj[good].astype(np.float64))
        if params.average_views:
            # Each agreeing camera has its own opinion of where the surface is.
            # Per-frame affine fits carry a few percent of independent error, so
            # the reference camera alone leaves the fused surface that many
            # percent thick and a z-buffer then reads its near face. Averaging
            # the agreeing positions cancels that instead of hiding it.
            X = Xsum / (1.0 + agree[:, None].astype(np.float64))
        keep = agree >= params.min_views
        if not keep.any():
            continue
        img = _undistorted_image(ki, work)
        if img is None:
            continue
        col = np.ascontiguousarray(img[vs[m][keep], us[m][keep]][:, ::-1])
        acc_x.append(X[keep].astype(np.float32))
        acc_c.append(col.astype(np.uint8))
        acc_f.append(np.clip(agree[keep], 0, 255).astype(np.uint8))
        keep_total += int(keep.sum())

    if not acc_x:
        raise DenseUnavailable("no point survived multi-view consensus")
    X = np.concatenate(acc_x)
    C_ = np.concatenate(acc_c)
    F = np.concatenate(acc_f)
    del acc_x, acc_c, acc_f
    Xv, Cv, Fv = voxel_reduce(X, C_, F, voxels[0])
    np.savez_compressed(root / "fused.npz", xyz=Xv, rgb=Cv, confidence=Fv,
                        median_depth=np.float64(median_depth))
    if progress:
        progress(STAGE_FUSE, len(kept), len(kept))
    return {
        "frames_used": len(kept), "frames_dropped": dropped,
        "sampled": int(raw_total), "survived": int(keep_total),
        "points": int(len(Xv)), "far_clip": zmax, "median_depth": median_depth,
        "seconds": time.time() - t0, "stopped_after": None,
    }


def run_pack_stage(params: DenseParams, root: Path, scale: dict,
                   input_digest: str | None = None,
                   keyframe_image_set: str | None = None) -> dict:
    """The LOD ladder plus the manifest a viewer reads."""
    with np.load(root / "fused.npz") as z:
        X = z["xyz"].astype(np.float32)
        C = z["rgb"]
        F = z["confidence"]
        median_depth = float(z["median_depth"]) if "median_depth" in z else 1.0
    voxels = params.voxels_for(median_depth)
    m = F >= params.min_confidence
    X, C, F = X[m], C[m], F[m]
    # A declared parameter, recorded in the manifest below. See DenseParams.
    dropped_outside_box = 0
    if params.pack_percentile > 0:
        lo = np.percentile(X, params.pack_percentile, 0)
        hi = np.percentile(X, 100.0 - params.pack_percentile, 0)
        inb = np.all((X >= lo) & (X <= hi), 1)
        dropped_outside_box = int(len(X) - int(inb.sum()))
        X, C, F = X[inb], C[inb], F[inb]
    else:
        # Disabling the filter must not crash the stage. `lo` and `hi` were
        # bound only inside the branch and then read unconditionally below, so
        # `pack_percentile = 0` -- the natural way to turn off a filter this
        # code documents as removing REAL observations -- raised
        # UnboundLocalError. The parameter was declared so it could be changed.
        lo, hi = X.min(0), X.max(0)

    levels = []
    for i, v in enumerate(voxels):
        Xi, Ci, Fi = (X, C, F) if i == 0 else voxel_reduce(X, C, F, v)
        size = write_points_bin(root / f"points_l{i}.bin", Xi, Ci, Fi)
        levels.append({"level": i, "voxel": v, "points": int(len(Xi)), "bytes": int(size)})

    manifest = {
        "schema_version": DENSE_SCHEMA_VERSION,
        "format": DENSE_FORMAT,
        "record": "float32 x,y,z; uint8 r,g,b; uint8 confidence",
        "endian": "little",
        "stride_bytes": POINT_STRIDE_BYTES,
        "confidence_meaning":
            "number of independent cameras whose depth agreed with this point",
        "min_confidence": params.min_confidence,
        # How many real observations the percentile box removed. A filter that
        # drops points has to be visible in the artifact it produced.
        "dropped_outside_pack_box": dropped_outside_box,
        "canonical_level": params.canonical_level,
        "mobile_level": params.mobile_level,
        "median_scene_depth": median_depth,
        "bbox_min": [float(v) for v in lo],
        "bbox_max": [float(v) for v in hi],
        # Repeated, not re-derived. The dense stage makes no new scale claim.
        "scale": dict(scale),
        "levels": levels,
        "params": params.as_dict(),
        # The solve this cloud was built from. Without it nothing at serve
        # time can tell that the world has since been re-solved, and the
        # phone is handed a reconstruction of a superseded geometry with no
        # way to say so. `status.json` carries the same value for artifacts
        # written before this key existed.
        "input_digest": input_digest,
        # The re-redacted keyframe set it was built from; null for `images/`.
        "keyframe_image_set": keyframe_image_set,
    }
    _write_json(root / "manifest.json", manifest)
    return {"levels": levels, "points": levels[0]["points"]}



# `solution.json` carries every keyframe id and every pose, so it is hundreds
# of kilobytes and json.loads on it costs milliseconds. `GET /worlds` asks for
# one field of it once per dense session, and a gallery walks every session of
# every world -- measured at 1.6-1.9 ms per session against 0.14 ms before the
# currency check existed, which is about a third of a second on a 200-session
# library and grows with adoption.
#
# The file only changes when a world is re-solved, so the answer is cached on
# (path, mtime, size). A rewritten solve changes at least one of the three, and
# an entry whose file has changed is simply recomputed. Bounded, because a
# library has finitely many sessions and the key set only grows with them.
_SOLVE_DIGEST_CACHE: dict = {}
_SOLVE_DIGEST_CACHE_MAX = 4096


def _solve_input_digest(solve_path: Path) -> str | None:
    """The `input_digest` of a persisted solve, cached on the file's identity."""
    try:
        st = solve_path.stat()
    except OSError:
        return None
    key = str(solve_path)
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _SOLVE_DIGEST_CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        digest = json.loads(solve_path.read_text()).get("input_digest")
    except (OSError, ValueError):
        return None
    if len(_SOLVE_DIGEST_CACHE) >= _SOLVE_DIGEST_CACHE_MAX:
        _SOLVE_DIGEST_CACHE.clear()
    _SOLVE_DIGEST_CACHE[key] = (stamp, digest)
    return digest


def dense_currency(store, world_id: str, session_id: str,
                   manifest: dict | None = None, *,
                   include_derived: bool = True) -> dict:
    """Whether the dense artifact still describes the geometry on disk.

    Contract `WORLD-BUILDER-WORLDS.md` rule 2 -- "the page never claims more
    than the caption says" -- and its BEHIND obligation apply to whatever the
    render route serves, and the dense page is now what it serves. Two
    different things can be behind, and they are reported separately:

    `solve_current`   the dense cloud was fused against THIS solve. False after
                      a re-solve or a second session: the points are real
                      observations, but of a superseded pose graph.
    `derived_current` `render.derived_current`'s answer, unchanged -- the
                      derived tree versus the newest keyframes.

    Either may be None, which means unknowable rather than stale, and an
    unknowable one never produces a BEHIND claim. Cheap on purpose: the solve
    digest is read from `solution.json` alone, never by loading the arrays.
    """
    out = {"solve_current": None, "derived_current": None,
           "artifact_digest": None, "solve_digest": None}
    if manifest is None:
        manifest = read_dense_manifest(store, world_id, session_id)
    artifact = (manifest or {}).get("input_digest")
    if artifact is None:
        # Artifacts packed before the manifest carried it; status.json has
        # recorded it since the first version of this stage.
        try:
            status = json.loads(
                (dense_dir(store, world_id, session_id) / "status.json").read_text())
            artifact = status.get("input_digest")
        except (OSError, ValueError):
            artifact = None
    out["artifact_digest"] = artifact

    try:
        from tower.world_builder.global_solve import workspace_for  # noqa: PLC0415

        solve_path = workspace_for(store, world_id, session_id).solution_path
        current = _solve_input_digest(solve_path)
    except Exception:  # noqa: BLE001 -- unreadable is "unknown", not "stale"
        current = None
    out["solve_digest"] = current
    if artifact is not None and current is not None:
        out["solve_current"] = bool(artifact == current)

    # `derived_current` re-reads and re-digests the whole keyframe journal, so
    # a caller that only needs the solve answer -- the worlds listing, which
    # walks every session of every world -- can say so and skip it.
    if include_derived:
        try:
            from tower.world_builder.render import derived_current  # noqa: PLC0415

            out["derived_current"] = derived_current(store, world_id, session_id)
        except Exception:  # noqa: BLE001
            out["derived_current"] = None
    return out


def _result_from_manifest(root: Path, manifest: dict) -> "DenseResult":
    """Report a finished artifact without re-deriving it.

    `status.json` holds the result the completing run wrote, and pruning keeps
    it precisely so that a completed run stays explainable after its
    intermediates are gone. Re-deriving the counts from `align.json` instead
    would get `frames_used` wrong: `align.json` records which frames ALIGNED,
    and the frames the fusion dropped are a different, smaller set.
    """
    stored = {}
    try:
        stored = (json.loads((root / "status.json").read_text()).get("result") or {})
    except (OSError, ValueError):
        stored = {}
    if not stored:
        # `status.json` is a single file that every later run overwrites, so a
        # failed or interrupted re-run can erase the completing run's record.
        # `align.json` and `fuse.json` are per-stage and survive, and between
        # them they hold the same counts. `align.json` alone would not: it
        # records which frames ALIGNED, and the frames fusion kept are fewer.
        try:
            align = json.loads((root / "align.json").read_text())
            records = align.get("records") or []
            ok = [r for r in records if r.get("ok")]
            hos = [r["held_out_rel"] for r in ok if r.get("held_out_rel") is not None]
            stored = {
                "frames_total": int(align.get("targets") or len(records)),
                "frames_aligned": len(ok),
                "align_rel_median": float(np.median(hos)) if hos else None,
            }
            fuse = json.loads((root / "fuse.json").read_text())
            stored["frames_used"] = int(fuse.get("frames_used") or 0)
            stored["frames_dropped"] = int(fuse.get("frames_dropped") or 0)
        except (OSError, ValueError, KeyError):
            pass
    levels = manifest.get("levels") or []
    fields = {f.name for f in dataclasses.fields(DenseResult)}
    kwargs = {k: v for k, v in stored.items() if k in fields}
    kwargs.update(
        state=STATE_OK,
        levels=levels,
        points=int(levels[0]["points"]) if levels else kwargs.get("points", 0),
        seconds={},
        stopped_after=None,
        reused=True,
    )
    return DenseResult(**kwargs)


def prune_intermediates(root: Path) -> int:
    """Remove what a successful run no longer needs, and report the bytes.

    Only ever the stage's OWN intermediates, inside its own subtree, and only
    after `pack` has succeeded: the per-frame depth maps, the undistorted
    frames, and fused.npz, whose points points_l0.bin already holds. The
    manifest, the point levels, align.json and status.json all stay, so the
    artifact remains complete and the run remains explainable.
    """
    import shutil

    freed = 0
    for path in (root / "work", root / "fused.npz"):
        if not path.exists():
            continue
        try:
            if path.is_dir():
                freed += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                shutil.rmtree(path)
            else:
                freed += path.stat().st_size
                path.unlink()
        except OSError:
            logger.warning("[Tower][WorldBuilder][dense] could not prune %s", path)
    return freed


def densify(
    store, world_id: str, session_id: str, *,
    params: DenseParams | None = None,
    should_stop=None,
    progress: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> DenseResult:
    """Densify one solved session. Idempotent, resumable, stop-aware."""
    from tower.world_builder.global_solve import load_solution

    params = params or DenseParams()
    root = dense_dir(store, world_id, session_id)
    root.mkdir(parents=True, exist_ok=True)
    seconds: dict = {}

    lock = _DenseLock(root)
    if not lock.acquire():
        detail = "another densify of this session is already running"
        # DO NOT WRITE status.json HERE. It is shared with the run that holds
        # the lock, so the loser of the race would stamp "unavailable" over a
        # perfectly healthy densify's record -- and, before the digest was
        # carried forward, erase the value that arms the BEHIND caption. The
        # loser has nothing to report about the session; it has something to
        # report about ITSELF, and that is the return value.
        logger.info("[Tower][WorldBuilder][dense] %s/%s: %s",
                    world_id, session_id, detail)
        return DenseResult(state=STATE_UNAVAILABLE, detail=detail)
    # THE SESSION'S SURFACE LOCK TOO (review V9, Q8; `_DenseLock`): the surface's and the
    # evidence gate's depth stages write this directory under it. Refused like the densify
    # lock above -- nothing written, because the holder owns `dense/<session>/` right now.
    from tower.world_builder.surface_pipeline import _SurfaceLock, surface_dir  # noqa: PLC0415

    surface_lock = _SurfaceLock(surface_dir(store, world_id, session_id))
    if not surface_lock.acquire():
        lock.release()
        detail = ("a surface build or the evidence gate of this session holds its lock, and "
                  "writes dense/<session>/ under it")
        logger.info("[Tower][WorldBuilder][dense] %s/%s: %s", world_id, session_id, detail)
        return DenseResult(state=STATE_UNAVAILABLE, detail=detail)

    # From here to the `finally` at the end, every exit path is inside the
    # lock. The three "unavailable" returns below used to sit OUTSIDE it, so
    # densifying a session with no solve -- the most ordinary failure there is
    # -- left the lock file behind and bricked that session permanently.
    try:
        solution = load_solution(store, world_id, session_id)
        if solution is None:
            _status(root, state=STATE_UNAVAILABLE,
                    detail="no global solution for this session")
            return DenseResult(state=STATE_UNAVAILABLE,
                               detail="no global solution for this session")

        session = store.read_session(world_id, session_id)
        intrinsics = session.intrinsics
        if intrinsics is None or getattr(intrinsics, "fx", None) is None:
            _status(root, state=STATE_UNAVAILABLE, detail="session has no intrinsics")
            return DenseResult(state=STATE_UNAVAILABLE, detail="session has no intrinsics")

        world = store.read_world(world_id)
        scale = getattr(world, "scale", None) or {"state": "unknown", "meters_per_unit": None}
        if hasattr(scale, "to_json_dict"):
            scale = scale.to_json_dict()
        scale = dict(scale)
        scale["note"] = ("inherited unchanged from the sparse solve; the dense stage "
                         "makes no new scale claim")

        digest = solution.input_digest
        # Which keyframe set this build reads: a re-redaction switch changes it
        # without changing the solve, so every cache below keys on it too.
        set_token = store.keyframe_image_set(world_id, session_id).cache_token

        # A COMPLETED ARTIFACT IS COMPLETE, and this is decided before the
        # first status write as well as before any stage runs -- an early
        # return that had already stamped `state: running` over the
        # completing run's record would destroy the frame counts it is
        # about to report. `prune_intermediates` deletes the
        # per-frame depth maps and `fused.npz` after a successful pack -- that
        # is the whole point of it -- so on a re-run the depth stage found its
        # cache key intact and reused it, the fuse stage found no `fused.npz`
        # and re-ran, and then died on the first `depth/00081.npy` that pruning
        # had removed. Every successfully densified world was in that state,
        # because pruning is the default, so `world_densify.py --world X` twice
        # raised FileNotFoundError the second time and the documented "a re-run
        # resumes" was false in exactly the ordinary case.
        manifest_path = root / "manifest.json"
        if manifest_path.exists() and not force:
            try:
                existing = json.loads(manifest_path.read_text())
            except (OSError, ValueError):
                existing = None
            # The digest may be absent from the manifest -- artifacts packed
            # before that key existed -- and those are precisely the ones that
            # cannot short-circuit and therefore fall into the pruned-inputs
            # path. `status.json` has recorded it since the first version, so
            # ask there too, the way `dense_currency` already does.
            existing_digest = (existing or {}).get("input_digest")
            if existing_digest is None:
                try:
                    existing_digest = json.loads(
                        (root / "status.json").read_text()).get("input_digest")
                except (OSError, ValueError):
                    existing_digest = None
            same_params = _params_match(
                (existing or {}).get("params") or {}, params.as_dict())
            # A digest of None must not match another None. A solve
            # without an input_digest and a manifest without one would
            # otherwise compare equal and short-circuit a rebuild that
            # nothing has established is unnecessary -- the same
            # "None matches everything" bug the depth cache key had.
            if (existing and digest is not None
                    and existing_digest == digest and same_params
                    and existing.get("keyframe_image_set") == set_token):
                levels = existing.get("levels") or []
                if levels and all((root / f"points_l{i}.bin").exists()
                                  for i in range(len(levels))):
                    logger.info(
                        "[Tower][WorldBuilder][dense] %s/%s is already densified "
                        "from this solve with these parameters; nothing to do "
                        "(--force rebuilds it)", world_id, session_id,
                    )
                    done = _result_from_manifest(root, existing)
                    # Repair the record while we are here. A run that crashed
                    # against this complete artifact left `state: running` over
                    # the completing run's result, and a cold reader cannot
                    # tell that from a densify still in progress. The counts
                    # were recovered above from the per-stage files, so write
                    # them back rather than leave the lie in place.
                    _status(root, state=STATE_OK, input_digest=digest,
                            params=params.as_dict(), result=done.as_dict())
                    return done

        _status(root, state=STATE_RUNNING, stage=STAGE_DEPTH,
                input_digest=digest, params=params.as_dict())

        align_path = root / "align.json"
        align = None
        prior = None
        trust_now = depth_trust_now(store, world_id, session_id)
        if align_path.exists() and not force:
            try:
                cached = json.loads(align_path.read_text())
                # The depth stage is only reusable if it was produced from the
                # same solve AND by the same network. Reusing depth maps from a
                # different backend while the manifest records the new one
                # would make the artifact unreproducible from its own params --
                # the most expensive kind of wrong, because everything still
                # runs and the numbers still look reasonable.
                want = _depth_cache_key(digest, params, set_token, trust_now)
                if depth_cache_matches(cached, digest, params, set_token, trust_now):
                    if cached.get("stopped_after") is None:
                        align = cached
                        logger.info("[Tower][WorldBuilder][dense] reusing depth stage")
                    else:
                        # Same parameters, interrupted run: resume per frame
                        # rather than discard several hundred predictions.
                        prior = cached
                else:
                    logger.info(
                        "[Tower][WorldBuilder][dense] depth cache is for %r, now "
                        "asked for %r: refitting", cached.get("cache_key"), want,
                    )
            except (OSError, ValueError):
                align = None
        if align is None:
            t = time.time()
            align = run_depth_stage(store, world_id, session_id, solution, intrinsics,
                                    params, root, should_stop=should_stop,
                                    progress=progress, prior=prior,
                                    reuse_predictions=reusable_predictions(
                                        align_path, params.backend,
                                        params.imagery_source,
                                        known_fov=params.known_fov))
            if align.get("stopped_after") is None:
                # Only a COMPLETE stage names its solve. A stopped one written
                # under the digest was trusted as a finished cache.
                align["digest"] = digest
                align["input_digest"] = digest
                align["cache_key"] = _depth_cache_key(
                    digest, params, align.get("keyframe_image_set"),
                    align.get("redaction_trust"))
            _write_json(align_path, align)
            seconds[STAGE_DEPTH] = time.time() - t
            if align.get("stopped_after") is not None:
                _status(root, state=STATE_STOPPED, stage=STAGE_DEPTH)
                return DenseResult(state=STATE_STOPPED, stopped_after=STAGE_DEPTH,
                                   detail="stopped during depth", seconds=seconds)

        ok = [r for r in align["records"] if r.get("ok")]
        hos = [r["held_out_rel"] for r in ok if r.get("held_out_rel") is not None]

        _status(root, state=STATE_RUNNING, stage=STAGE_FUSE, input_digest=digest)
        t = time.time()
        fuse_path = root / "fuse.json"
        fuse = None
        if (root / "fused.npz").exists() and fuse_path.exists() and not force:
            try:
                cached_fuse = json.loads(fuse_path.read_text())
                if (cached_fuse.get("cache_key") == _fuse_cache_key(
                        digest, params, align.get("keyframe_image_set"))
                        and cached_fuse.get("stopped_after") is None):
                    fuse = cached_fuse
                    logger.info("[Tower][WorldBuilder][dense] reusing fusion")
            except (OSError, ValueError):
                fuse = None
        if fuse is None:
            fuse = run_fuse_stage(solution, align, params, root,
                                  should_stop=should_stop, progress=progress)
            fuse["cache_key"] = _fuse_cache_key(digest, params,
                                                align.get("keyframe_image_set"))
            if fuse.get("stopped_after") is None:
                _write_json(fuse_path, fuse)
        seconds[STAGE_FUSE] = time.time() - t
        if fuse.get("stopped_after") is not None:
            _status(root, state=STATE_STOPPED, stage=STAGE_FUSE)
            return DenseResult(state=STATE_STOPPED, stopped_after=STAGE_FUSE,
                               detail="stopped during fusion", seconds=seconds)

        _status(root, state=STATE_RUNNING, stage=STAGE_PACK, input_digest=digest)
        t = time.time()
        pack = run_pack_stage(params, root, scale, input_digest=digest,
                              keyframe_image_set=align.get("keyframe_image_set"))
        seconds[STAGE_PACK] = time.time() - t

        result = DenseResult(
            state=STATE_OK,
            frames_total=align.get("targets", 0),
            frames_aligned=len(ok),
            frames_used=fuse["frames_used"],
            frames_dropped=fuse["frames_dropped"],
            points=pack["points"],
            levels=pack["levels"],
            seconds=seconds,
            align_rel_median=float(np.median(hos)) if hos else None,
        )
        if not params.keep_intermediates:
            freed = prune_intermediates(root)
            if freed:
                logger.info("[Tower][WorldBuilder][dense] pruned %.0f MB of intermediates",
                            freed / 1e6)
        _status(root, state=STATE_OK, input_digest=digest, result=result.as_dict())
        logger.info(
            "[Tower][WorldBuilder][dense] world %s session %s: %s of %s frames used, "
            "%s points, align residual %.1f%%, %.1fs",
            world_id, session_id, result.frames_used, result.frames_total,
            result.points, (result.align_rel_median or 0) * 100, sum(seconds.values()),
        )
        return result

    except DenseUnavailable as exc:
        _status(root, state=STATE_UNAVAILABLE, detail=str(exc))
        logger.info("[Tower][WorldBuilder][dense] unavailable: %s", exc)
        return DenseResult(state=STATE_UNAVAILABLE, detail=str(exc), seconds=seconds)
    except Exception as exc:  # noqa: BLE001 -- recorded, never propagated into finalization
        _status(root, state=STATE_FAILED, detail=f"{type(exc).__name__}: {exc}")
        logger.exception("[Tower][WorldBuilder][dense] failed")
        return DenseResult(state=STATE_FAILED, detail=f"{type(exc).__name__}: {exc}",
                           seconds=seconds)
    finally:
        # The lock's directory, `surface/<session>/`, is left even when this made it (as the
        # gate's depth stage leaves it): removing it could race a surface build's own
        # `mkdir` and lock creation. Nothing reads an empty one as a surface.
        surface_lock.release()
        lock.release()


def read_dense_manifest(store, world_id: str, session_id: str) -> dict | None:
    """The dense manifest for a session, or None. Never raises."""
    p = dense_dir(store, world_id, session_id) / "manifest.json"
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return m if m.get("format") == DENSE_FORMAT else None
