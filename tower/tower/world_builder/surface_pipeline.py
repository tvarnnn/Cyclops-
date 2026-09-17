"""The surface stage, orchestrated over a solved world.

Three stages, checkpointed under `<world>/surface/<session>/`:

    fuse    integrate every gated, posed depth map into a sparse TSDF
    mesh    marching cubes over the observed part of the field, then
            component pruning and feature-preserving smoothing
    pack    write the level-of-detail ladder and the manifest

The depth maps themselves come from the dense stage's `depth` stage, in
`<world>/dense/<session>/`, and are shared rather than recomputed: a keyframe's
monocular depth and its fit to the sparse points do not depend on what is done
with them afterwards, and that stage is the expensive one. Running the surface
stage on a world that has never been densified therefore runs `depth` first and
leaves it cached for the point stage too.

Nothing here writes into `derived/`. `surface/` is additive and optional in
exactly the way `solve/` and `dense/` are: a reader that does not know about it
must be able to ignore it.

Artifacts are published atomically and are self-checking. A torn mesh buffer is
refused by `read_mesh_bytes` on its length and magic rather than misread -- the
`BadZipFile` incident that cost a 795-keyframe session is the reason that guard
is in the format and not in the reader's caller.
"""

from __future__ import annotations

import json
import math
import logging
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from tower.storage import write_bytes_atomic, write_json_atomic
from tower.world_builder.dense import DenseUnavailable, DepthModelUnavailable
from tower.world_builder.surface import (
    STAGE_DEPTH,
    STAGE_FUSE,
    STAGE_MESH,
    STAGE_PACK,
    SURFACE_FORMAT,
    SURFACE_FORMAT_ENCLOSED_FILL,
    SURFACE_SCHEMA_VERSION,
    SurfaceParams,
    SurfaceResult,
    SurfaceUnavailable,
    SurfaceVolume,
    decimate,
    depth_bound,
    depth_validity,
    drop_small_components,
    evidence_filter,
    hidden_low_weight,
    keep_faces,
    weld_mesh,
    extract_sealed,
    fill_enclosed,
    snap_planes,
    taubin_smooth,
    truncation_for,
    truncation_rel,
    vertex_normals,
    write_mesh_bytes,
)

logger = logging.getLogger(__name__)

STATE_OK = "ok"
STATE_RUNNING = "running"
STATE_FAILED = "failed"
STATE_STOPPED = "stopped"
STATE_UNAVAILABLE = "unavailable"


def surface_dir(store, world_id: str, session_id: str) -> Path:
    """`<world>/surface/<session>` -- beside `solve/` and `dense/`."""
    return store.world_dir(world_id) / "surface" / session_id


def _holder_alive(pid, written_at) -> bool:
    """Whether the process that WROTE a lock or status is still that process.

    A bare pid probe is not enough on Windows, where pids are recycled
    quickly: a live surface child killed at Stop leaves its lock behind, and
    if its pid is reused before the final build asks, the final build is
    refused as "already running" and the world keeps its coarse live mesh.
    The store's own guard answers the real question -- a process that started
    clearly after the file was written cannot have written it.
    """
    from tower.world_builder.store import _holder_is_running  # noqa: PLC0415

    if not isinstance(pid, int) or pid <= 0:
        return False
    return _holder_is_running(pid, None, lock_written_at=written_at)


def _pid_is_running(pid: int) -> bool:
    from tower.world_builder.dense_pipeline import _pid_is_running as probe

    return probe(pid)


def _status(root: Path, **fields) -> None:
    """Write the stage's state where a cold reader can find it.

    `pid` on every state, not only `running`: the surface stage deliberately
    outlives the world lock, so the supervisor can kill it after finalization
    already reads complete. Without a pid a killed run leaves `running` on disk
    forever and nothing can tell that from a run still going.
    """
    payload = {"schema_version": SURFACE_SCHEMA_VERSION, "pid": os.getpid(),
               "updated_at": time.time(), **fields}
    if payload.get("input_digest") is None:
        try:
            prior = json.loads((root / "status.json").read_text())
        except (OSError, ValueError):
            prior = {}
        if prior.get("input_digest"):
            payload["input_digest"] = prior["input_digest"]
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / "status.json", payload)


def status_is_stale(status: dict) -> bool:
    """True when a status says `running` but its process is gone."""
    if not status or status.get("state") != STATE_RUNNING:
        return False
    pid = status.get("pid")
    if not isinstance(pid, int):
        return True
    return not _holder_alive(pid, status.get("updated_at"))


class _SurfaceLock:
    """One surface build at a time per session.

    Same reasoning as the dense stage's lock: this runs after the world writer
    lock is released, so it is safe against other writers but not against
    another surface build of the same session, which is easy to start by
    accident. Two runs interleaving their writes would publish a mesh of the
    wrong length -- which the format's own length check would catch on read,
    but only after the world had already been served as broken.
    """

    def __init__(self, root: Path) -> None:
        self.path = root / ".surface.lock"
        self.held = False

    def _stale(self) -> bool:
        try:
            pid = int(json.loads(self.path.read_text()).get("pid", -1))
            written = self.path.stat().st_mtime
        except (OSError, ValueError, AttributeError, TypeError):
            return True
        return not _holder_alive(pid, written)

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not self._stale():
                    return False
                # Ask AGAIN immediately before removing it. Two contenders can
                # both judge the same dead lock stale; the one that loses the
                # race would otherwise unlink the winner's fresh lock and both
                # would hold it. The window left is one read wide, and the
                # check after creation below closes most of that too.
                if not self._stale():
                    return False
                try:
                    self.path.unlink()
                except OSError:
                    return False
                continue
            token = f"{os.getpid()}-{time.time_ns()}"
            with os.fdopen(fd, "w") as handle:
                json.dump({"pid": os.getpid(), "at": time.time(), "token": token}, handle)
            try:
                mine = json.loads(self.path.read_text()).get("token") == token
            except (OSError, ValueError, AttributeError):
                mine = False
            if not mine:
                return False
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


def _stopped(should_stop) -> bool:
    return bool(should_stop and should_stop())


def _params_digest(params: SurfaceParams, input_digest: str | None) -> str:
    return "|".join(str(x) for x in (input_digest, *params.digest_fields()))


# ---------------------------------------------------------------------------
# reading the depth stage's output
# ---------------------------------------------------------------------------


def ensure_depth_stage(store, world_id: str, session_id: str, solution,
                       intrinsics, *, gate_rel: float, backend: str | None,
                       should_stop=None, progress=None) -> tuple[dict, Path]:
    """Return the dense stage's `align.json` and its work directory, running
    the depth stage first if it is absent or was produced from another solve.

    Deliberately shares `dense/<session>/` rather than keeping a private copy.
    The depth maps are a property of the keyframes and the solve, not of what
    is built from them, and they are the expensive part -- 68 s of GPU on this
    corpus. A surface build and a point build of the same session should pay
    for them once.
    """
    from tower.world_builder.dense import DenseParams
    from tower.world_builder.dense_pipeline import dense_dir, run_depth_stage

    root = dense_dir(store, world_id, session_id)
    root.mkdir(parents=True, exist_ok=True)
    dparams = DenseParams(gate_rel=gate_rel,
                          **({"backend": backend} if backend else {}))

    align_path = root / "align.json"
    prior = None
    set_token = store.keyframe_image_set(world_id, session_id).cache_token
    from tower.world_builder.dense_pipeline import (
        _depth_cache_key,
        depth_trust_now,
        reusable_predictions,
    )

    trust = depth_trust_now(store, world_id, session_id)
    if align_path.exists():
        try:
            cached = json.loads(align_path.read_text())
        except (OSError, ValueError):
            cached = None
        if cached and _depth_cache_usable(cached, root, solution, dparams,
                                          image_set=set_token, trust=trust):
            return cached, root / "work"

    # Never `prior=cached`: `prior` resumes whole records, fits included, and
    # a cache that was not usable above is by definition for another solve.
    # Its PREDICTIONS are still good and are reused; the fits are redone.
    align = run_depth_stage(store, world_id, session_id, solution, intrinsics,
                            dparams, root, should_stop=should_stop,
                            progress=progress, prior=None,
                            reuse_predictions=reusable_predictions(
                                align_path, dparams.backend))
    if align.get("stopped_after") is None:
        # Name the solve, in both spellings the two readers of this file use,
        # so neither can mistake it for a cache of another solve.
        align["input_digest"] = solution.input_digest
        align["digest"] = solution.input_digest
        align["cache_key"] = _depth_cache_key(solution.input_digest, dparams,
                                              align.get("keyframe_image_set"),
                                              align.get("redaction_trust"))
    write_json_atomic(align_path, align)
    return align, root / "work"


def _depth_cache_usable(cached: dict, root: Path, solution, dparams, *,
                        image_set: str | None = None, trust: str | None = None) -> bool:
    """A cached depth stage is reusable only if it came from this solve, this
    network, and still has its per-frame maps on disk.

    That last clause is the one that bites: `prune_intermediates` deletes
    `work/` after a successful pack -- which is its whole purpose -- so a world
    that has already been densified has a valid `align.json` and no depth maps
    behind it. Trusting the cache key alone would send the fuse stage looking
    for files pruning removed.
    """
    # THE CACHE MUST NAME THIS SOLVE, AND "UNNAMED" IS NOT A MATCH.
    #
    # This read `cached.get("input_digest") not in (None, digest)`, and the
    # depth stage never wrote the key, so every cache was "unnamed" and every
    # cache matched. Measured on a real-time replay of the canonical walk:
    # the one live surface cached a depth stage fitted against the first 51
    # keyframes, and the FINAL surface after Stop reused it -- a 51-keyframe
    # mesh at the live voxel, shipped as the finished world under the final
    # solve's digest.
    if solution.input_digest is None:
        return False
    named = cached.get("input_digest") or cached.get("digest")
    if named != solution.input_digest:
        return False
    # AN INTERRUPTED STAGE IS NOT A COMPLETE ONE. A stop mid-depth leaves an
    # `align.json` holding only the frames reached; trusting it built a
    # surface from 3 of 8 frames under the full solve's digest, and every
    # later run kept it.
    if cached.get("stopped_after") is not None:
        return False
    if cached.get("backend") != dparams.backend:
        return False
    # Fill masks made under an earlier rule may be the empty masks a live build
    # used to save over engine-filled faces (review 2, I6). Refit; the
    # predictions of that stage are not reused either (`reusable_predictions`).
    from tower.world_builder.dense_pipeline import FILL_RULE  # noqa: PLC0415

    if cached.get("fill_rule") != FILL_RULE:
        return False
    # THE KEYFRAME SET IT READ. A re-redaction switch (or a switch back) changes
    # the pixels without changing the solve; `image_set` is the store's
    # `keyframe_image_set(...).cache_token` now, None for the capture's
    # `images/`, which is also what every align.json written before the switch
    # existed reads as.
    if cached.get("keyframe_image_set") != image_set:
        return False
    # THE TRUST DECISION IT READ THE PIXELS UNDER (`dense_pipeline.recorded_trust`).
    # A walk's stage re-redacted under `none` and recorded those bytes' SHA-1;
    # after Stop the label is trusted, the solve digest is often unchanged, and
    # reusing the walk's records made the final appearance refuse every frame
    # the re-redaction had touched (review 1, M3). `trust` is what a stage run
    # now would record; every production caller passes it.
    if trust is not None:
        from tower.world_builder.dense_pipeline import recorded_trust  # noqa: PLC0415

        if recorded_trust(cached) != trust:
            return False
    records = [r for r in cached.get("records", []) if r.get("ok")]
    if not records:
        return False
    depth_dir = root / "work" / "depth"
    undist_dir = root / "work" / "undist"
    if not depth_dir.is_dir() or not undist_dir.is_dir():
        return False
    # Spot-check rather than stat every frame: pruning removes the whole
    # directory tree, it does not remove one map in the middle.
    probe = records[:3] + records[-3:]
    return all((depth_dir / f"{r['ki']:05d}.npy").exists()
               and (undist_dir / f"{r['ki']:05d}.jpg").exists() for r in probe)


def _depth_from_prediction(pred, a, b, kind):
    from tower.world_builder.dense import depth_from_prediction

    return depth_from_prediction(pred, a, b, kind)


def _dilate_fill(fill, px: int):
    """The redaction fill grown by `px` pixels on every side.

    The depth stage marks the fill itself (dilated 1 px). The depth network,
    though, predicts the pixels NEXT to a synthetic fill wedge from the wedge,
    and ghost faces sat exactly on its boundary (keyframe 211, half filled,
    passed the gate at 0.049)."""
    import torch

    if px <= 0 or not bool(fill.any()):
        return fill
    k = 2 * int(px) + 1
    return torch.nn.functional.max_pool2d(
        fill.to(torch.float32)[None, None], k, 1, int(px)).squeeze(0).squeeze(0) > 0.5


class _Frames:
    """The gated, posed frames of one session, with their depth on disk."""

    def __init__(self, align: dict, work: Path, solution, params: SurfaceParams):
        self.work = work
        # `transients.TransientReport`, attached once the masks are ensured.
        self.transients = None
        self.masked = 0
        self.kind = align.get("kind", "disparity")
        cam = align.get("camera") or solution.camera or {}
        self.K = np.array([[cam["fx"], 0, cam["cx"]],
                           [0, cam["fy"], cam["cy"]],
                           [0, 0, 1.0]], float)
        poses = solution.poses or {}
        self.items = []
        self.kids = {}
        self.image_sha1 = {}
        # The consistency field (`depth_consistency.py`), when one was applied:
        # `prepared` then yields corrected depth rather than the plain affine.
        self.correction = None
        self.consistency = None
        held = []
        for r in align.get("records", []):
            if not r.get("ok"):
                continue
            ho = r.get("held_out_rel")
            if ho is not None:
                held.append(ho)
                if ho > params.gate_rel:
                    continue
            pose = poses.get(r["kid"])
            if pose is None:
                continue
            ki = int(r["ki"])
            if not (work / "depth" / f"{ki:05d}.npy").exists():
                continue
            self.kids[ki] = r["kid"]
            self.image_sha1[ki] = r.get("image_sha1")
            self.items.append((ki, float(r["a"]), float(r["b"]),
                               np.array(pose["rotation"], float).reshape(3, 3),
                               np.array(pose["translation"], float),
                               1.0 if ho is None else float(ho),
                               r.get("z_sparse_max")))
        self.median_held_out = float(np.median(held)) if held else None
        self.offered = len(align.get("records", []))

    def __len__(self):
        return len(self.items)

    def poses(self):
        """(keyframe id, R, t) of every gated frame."""
        for ki, _a, _b, R, t, _ho, _zmax in self.items:
            yield self.kids[ki], R, t

    def load(self, ki, *, image: bool = True):
        import cv2

        pred = np.load(self.work / "depth" / f"{ki:05d}.npy").astype(np.float32)
        img = None
        if image:
            img = cv2.imread(str(self.work / "undist" / f"{ki:05d}.jpg"),
                             cv2.IMREAD_COLOR)
            if img is None:
                return None, None, None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fill = None
        fp = self.work / "depth" / f"{ki:05d}_fill.npy"
        if fp.exists():
            f = np.load(fp)
            if f.shape == pred.shape:
                fill = f
        return pred, img, fill

    def prepared(self, params: SurfaceParams, median_depth: float, device):
        import torch

        for ki, a, b, R, t, ho, zmax in self.items:
            pred, img, fill = self.load(ki)
            if pred is None or img.shape[:2] != pred.shape:
                continue
            z = _depth_from_prediction(pred, a, b, self.kind)
            zt = torch.as_tensor(np.asarray(z, np.float32), device=device)
            if self.correction is not None:
                # Validity below is decided on the CORRECTED depth, which is
                # the depth that is fused.
                zt = self.correction.correct(ki, zt)
            ok, cosang = depth_validity(
                zt, self.K, params, median_depth,
                max_depth=depth_bound(params, median_depth, zmax))
            if fill is not None:
                ok &= ~_dilate_fill(torch.as_tensor(fill, device=device),
                                    params.fill_margin_px)
            # The wearer's hands, arms and held phone: zero weight. Neither a
            # measurement (a hand on the desk is within depth noise of the desk
            # and fuses into it as a bump with the hand's colour) nor a carve.
            det = self.transients.mask(ki) if self.transients is not None else None
            if det is not None and det.shape == tuple(ok.shape):
                ok &= ~torch.as_tensor(det, device=device)
                self.masked += 1
            if not bool(ok.any()):
                continue
            fw = float(np.clip((params.gate_rel - ho) / max(params.gate_rel, 1e-9),
                               params.frame_weight_floor, 1.0))
            yield (zt, ok,
                   torch.as_tensor(img.astype(np.float32), device=device),
                   torch.as_tensor(R, device=device).float(),
                   torch.as_tensor(t, device=device).float(),
                   cosang * fw)


# ---------------------------------------------------------------------------
# the public entry point
# ---------------------------------------------------------------------------


def surfacify(store, world_id: str, session_id: str, *,
              params: SurfaceParams | None = None,
              should_stop=None,
              progress: Callable[[str, int, int], None] | None = None,
              force: bool = False,
              backend: str | None = None,
              transient_backend_factory=None) -> SurfaceResult:
    """Build the surface of one solved session. Idempotent, stop-aware."""
    from tower.world_builder.global_solve import load_solution

    params = params or SurfaceParams()
    root = surface_dir(store, world_id, session_id)
    root.mkdir(parents=True, exist_ok=True)
    seconds: dict = {}

    lock = _SurfaceLock(root)
    if not lock.acquire():
        detail = "another surface build of this session is already running"
        # Deliberately no status write: `status.json` belongs to the run that
        # holds the lock, and stamping "unavailable" over a healthy build's
        # record would erase the digest that arms the staleness caption. The
        # loser has nothing to say about the session, only about itself.
        logger.info("[Tower][WorldBuilder][surface] %s/%s: %s",
                    world_id, session_id, detail)
        return SurfaceResult(state=STATE_UNAVAILABLE, detail=detail)

    try:
        # Sweep what earlier builds of this session left unnamed -- a pack
        # that was killed, levels superseded long enough ago -- whether or not
        # this build then publishes. Otherwise a killed final pack, which no
        # later build of the session follows, kept its orphans forever.
        _sweep_unnamed_levels(root)
        solution = load_solution(store, world_id, session_id)
        if solution is None:
            return _unavailable(root, "no global solution for this session")
        session = store.read_session(world_id, session_id)
        intrinsics = session.intrinsics
        if intrinsics is None or getattr(intrinsics, "fx", None) is None:
            return _unavailable(root, "session has no intrinsics")

        digest = solution.input_digest
        from tower.world_builder.dense import DenseParams  # noqa: PLC0415

        # The depth backend changes every triangle; a request for a different
        # network must not be answered "already built".
        pdigest = _params_digest(params, digest) + "|" + (backend or DenseParams().backend)
        # A re-redaction switch rebuilds the surface too (its colours and fill
        # exclusion come from the depth stage's frames). Appended only when a
        # re-redacted set is active, so every digest written before is unchanged.
        set_token = store.keyframe_image_set(world_id, session_id).cache_token
        if set_token:
            pdigest += "|set:" + set_token
        from tower.world_builder.transients import TransientParams  # noqa: PLC0415

        tparams = TransientParams(mode=params.transient_detector)
        pdigest += "|transients:" + tparams.rule_id()

        done = _already_built(root, digest, pdigest, force)
        if done is not None:
            logger.info("[Tower][WorldBuilder][surface] %s/%s is already built "
                        "from this solve with these parameters (--force rebuilds)",
                        world_id, session_id)
            _status(root, state=STATE_OK, input_digest=digest,
                    params_digest=pdigest, result=done.as_dict())
            return done

        _status(root, state=STATE_RUNNING, stage=STAGE_DEPTH,
                input_digest=digest, params_digest=pdigest)

        t = time.time()
        align, work = ensure_depth_stage(
            store, world_id, session_id, solution, intrinsics,
            gate_rel=params.gate_rel, backend=backend,
            should_stop=should_stop, progress=progress)
        seconds[STAGE_DEPTH] = round(time.time() - t, 2)
        if _stopped(should_stop):
            return _stop(root, STAGE_DEPTH, seconds)

        frames = _Frames(align, work, solution, params)
        if not len(frames):
            return _unavailable(
                root, "no keyframe passed the alignment gate, so there is "
                      "nothing to fuse")

        # The transient detector's masks, computed for keyframes that have none
        # under this rule yet (new keyframes, during a walk) and cached beside
        # the depth work for the appearance stage after this one.
        _status(root, state=STATE_RUNNING, stage=STAGE_TRANSIENTS,
                input_digest=digest, params_digest=pdigest)
        t = time.time()
        frames.transients = _ensure_transients(
            store, world_id, session_id, solution, intrinsics, align, work, frames,
            tparams, should_stop, progress, transient_backend_factory)
        seconds[STAGE_TRANSIENTS] = round(time.time() - t, 2)
        if frames.transients.state == "stopped" or _stopped(should_stop):
            return _stop(root, STAGE_TRANSIENTS, seconds)

        median_depth, scale_source = _scene_scale(frames, solution)
        if not (median_depth > 0):
            return _unavailable(root, "the solve has no usable scene depth")
        voxel = params.voxel_frac * median_depth
        trunc = truncation_for(params, voxel, median_depth, frames.median_held_out)

        if params.depth_consistency:
            t = time.time()
            _status(root, state=STATE_RUNNING, stage=STAGE_CONSISTENCY)
            consistency = _consistency(work.parent, frames, solution, params, median_depth,
                                       should_stop)
            seconds[STAGE_CONSISTENCY] = round(time.time() - t, 2)
            if consistency.state == "stopped" or _stopped(should_stop):
                return _stop(root, STAGE_CONSISTENCY, seconds)
            frames.correction = consistency.field
            frames.consistency = consistency.summary()

        result = _build(root, frames, params, median_depth, voxel, trunc,
                        seconds, should_stop, progress)
        if result.state != STATE_OK:
            return result

        scale = _scale_note(store, world_id)
        _mark_superseded(root, {lv["file"] for lv in result.levels})
        _write_manifest(root, result, params, digest, pdigest, median_depth, scale,
                        scale_source, transients=dict(frames.transients.record(),
                                                      frames_fused_with_mask=frames.masked),
                        keyframe_image_set=set_token)
        _prune_superseded_levels(root, {lv["file"] for lv in result.levels})
        _status(root, state=STATE_OK, input_digest=digest,
                params_digest=pdigest, result=result.as_dict())
        return result

    except SurfaceUnavailable as exc:
        _status(root, state=STATE_UNAVAILABLE, detail=exc.reason)
        return SurfaceResult(state=STATE_UNAVAILABLE, detail=exc.reason)
    except DepthModelUnavailable as exc:
        # The depth network cannot run on this machine: not installed, or its
        # weights neither cached nor downloadable. That is a configuration, not
        # a crash: say so once, without a traceback, and mark it permanent so
        # the live worker stops relaunching a child every solve.
        #
        # PERMANENT EVEN WHEN THE CAUSE IS "OFFLINE", deliberately. "Permanent"
        # disables only THIS walk's live surfaces (`BackgroundSurface._reap`);
        # the final build after Stop, `world_surface.py`, and the next walk all
        # try again, so a network that comes back is used at the next of those.
        # What it gives up is a live surface later in the same walk -- and a
        # retry that did succeed would start a 1.3 GB download in a below-
        # normal-priority child, over the link the glasses' frames arrive on,
        # which the Stop then kills.
        reason = str(exc)
        logger.warning("[Tower][WorldBuilder][surface] %s/%s cannot build a "
                       "surface on this machine: %s", world_id, session_id, reason)
        _status(root, state=STATE_UNAVAILABLE, detail=reason, permanent=True)
        return SurfaceResult(state=STATE_UNAVAILABLE, detail=reason, permanent=True)
    except DenseUnavailable as exc:
        # A refusal about THIS session's inputs (a camera the poses were not
        # solved in, say). Unavailable, but nothing about the machine: the
        # next session may build, so it is not permanent.
        reason = str(exc)
        logger.warning("[Tower][WorldBuilder][surface] %s/%s cannot build a "
                       "surface for this session: %s", world_id, session_id, reason)
        _status(root, state=STATE_UNAVAILABLE, detail=reason)
        return SurfaceResult(state=STATE_UNAVAILABLE, detail=reason)
    except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed silently
        logger.exception("[Tower][WorldBuilder][surface] %s/%s failed",
                         world_id, session_id)
        _status(root, state=STATE_FAILED, detail=str(exc))
        return SurfaceResult(state=STATE_FAILED, detail=str(exc))
    finally:
        lock.release()


STAGE_CONSISTENCY = "consistency"


def _consistency(dense_root, frames, solution, params, median_depth, should_stop):
    """The consistency field for these frames; never raises. A failure is a
    `failed` record and the plain affine, not a failed surface."""
    from tower.world_builder.depth_consistency import (  # noqa: PLC0415
        STATE_FAILED as C_FAILED,
        ConsistencyResult,
        ensure_consistency,
    )

    try:
        return ensure_consistency(dense_root, frames, solution, params, median_depth,
                                  should_stop=should_stop)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[Tower][WorldBuilder][surface] consistency field failed; "
                         "fusing the plain affine")
        return ConsistencyResult(C_FAILED, {"state": C_FAILED,
                                            "reason": f"{type(exc).__name__}: {exc}"})
STAGE_TRANSIENTS = "transients"


def _ensure_transients(store, world_id, session_id, solution, intrinsics, align, work,
                       frames, tparams, should_stop, progress, backend_factory):
    """`transients.ensure_transient_masks` over the gated frames. Never fails
    the surface: a machine that cannot run the detector fuses without masks and
    the manifest says `unavailable`."""
    from tower.world_builder import transients as T  # noqa: PLC0415

    records = {int(r["ki"]): r for r in align.get("records", [])
               if isinstance(r, dict) and r.get("ki") is not None}
    try:
        return T.ensure_transient_masks(
            store, world_id, session_id, [(ki, frames.kids[ki]) for ki, *_ in frames.items],
            intrinsics=intrinsics, camera=solution.camera, align_records=records,
            depth_dir=work / "depth", params=tparams, backend_factory=backend_factory,
            should_stop=should_stop, progress=progress)
    except Exception as exc:  # noqa: BLE001 -- a quality mask never fails the surface
        logger.exception("[Tower][WorldBuilder][surface] %s/%s: transient masks failed",
                         world_id, session_id)
        return T.TransientReport(state=T.STATE_FAILED, params=tparams,
                                 detail=f"{type(exc).__name__}: {exc}")


def _unavailable(root: Path, detail: str) -> SurfaceResult:
    _status(root, state=STATE_UNAVAILABLE, detail=detail)
    return SurfaceResult(state=STATE_UNAVAILABLE, detail=detail)


def _stop(root: Path, stage: str, seconds: dict) -> SurfaceResult:
    _status(root, state=STATE_STOPPED, stage=stage)
    return SurfaceResult(state=STATE_STOPPED, stopped_after=stage, seconds=seconds)


MIN_SCALE_OBSERVATIONS = 64


def _scene_scale(frames: "_Frames", solution) -> tuple[float, str]:
    """The one length every other length in the surface stage is a fraction of.

    WHY THE SPARSE OBSERVATIONS AND NOT THE DEPTH MAPS. The scale used to be
    the median of dense depth sampled from every 14th frame. That definition
    moved with two things that have nothing to do with the room:

      * which frames were sampled -- changing only the sampling offset moved
        the canonical world's scale between 0.88x and 1.07x of itself;
      * where the wearer happened to be looking so far -- a build over the
        first 100 / 150 / 200 keyframes, as a live build during the walk is,
        got 1.27x / 1.39x / 1.43x the final scale, because the walk looked at
        the room first and the desk close up later.

    Every live build therefore used a coarser voxel and, while the far clip
    was tied to this number, a farther clip than the final one, so the room
    visibly SHRANK when the final surface replaced the live one.

    The median camera-frame depth of every sparse observation of the gated
    frames is pooled over all of them (no sampling offset exists) and is
    weighted by where features were tracked, not by which pixels a close-up
    fills: over the same prefixes it is 1.09x / 1.14x / 1.16x the final
    (`Glasses-scratch/wb-final-recon/surface-r2/gauge2.json`). It is not
    perfectly stable -- content still moves it -- and nothing that decides
    WHERE surface may exist depends on it any more (the far bound is per
    frame, `SurfaceParams.anchor_depth_multiple`); it sets resolution.

    Falls back to the dense-depth median over EVERY frame when the solve
    carries too few observations (older solves, synthetic worlds).
    """
    z = _sparse_observation_depths(frames, solution)
    if z is not None and z.size >= MIN_SCALE_OBSERVATIONS:
        return float(np.median(z)), "sparse-observation-depth"
    return _median_scene_depth(frames), "dense-depth-median"


def _sparse_observation_depths(frames: "_Frames", solution):
    """Camera-frame depth of every sparse observation made by a gated frame."""
    obs = getattr(solution, "observations", None)
    xyz = getattr(solution, "xyz", None)
    kids = getattr(solution, "keyframe_ids", None)
    if obs is None or xyz is None or not kids or len(obs) == 0 or len(xyz) == 0:
        return None
    obs = np.asarray(obs).reshape(-1, 3)
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    index = {kid: i for i, kid in enumerate(kids)}
    order = np.argsort(obs[:, 0], kind="stable")
    kf_sorted = obs[order, 0]
    out = []
    for kid, R, t in frames.poses():
        i = index.get(kid)
        if i is None:
            continue
        lo, hi = np.searchsorted(kf_sorted, [i, i + 1])
        pts = obs[order[lo:hi], 2]
        pts = pts[(pts >= 0) & (pts < len(xyz))]
        if not len(pts):
            continue
        zc = (xyz[pts] @ R.T + t)[:, 2]
        out.append(zc[np.isfinite(zc) & (zc > 0)])
    return np.concatenate(out) if out else None


def _median_scene_depth(frames: "_Frames") -> float:
    """Median of the aligned depth the cameras actually measured.

    Every length in the surface stage is a fraction of this, because
    `global_solve` never normalises and the same room has solved to a ten-unit
    extent and a three-hundred-unit one.

    The definition matters, and two wrong ones were tried first. Measured from
    the MEAN camera centre to the sparse points it is 5.10; measured
    observation-wise to the sparse points it is 4.71; measured over the dense
    depth maps -- which is what the dense stage uses, and what the frames'
    relative alignment error is relative to -- it is 4.07. Using either of the
    first two silently grew every voxel and every truncation by a quarter and
    made the two stages disagree about what "0.006 of the scene" means.
    """
    if not len(frames):
        return 0.0
    # Every frame, a fixed pixel stride: sampling every n-th FRAME made the
    # answer depend on which frames the stride happened to land on.
    stride = max(1, (len(frames) * 97) // 4096)
    chunks = []
    for ki, a, b, _R, _t, _ho, _zmax in frames.items:
        pred, _img, _fill = frames.load(ki, image=False)
        if pred is None:
            continue
        z = _depth_from_prediction(pred, a, b, frames.kind)
        z = z[np.isfinite(z) & (z > 0)]
        if z.size:
            chunks.append(z[::stride])
    if not chunks:
        return 0.0
    return float(np.median(np.concatenate(chunks)))


def _scale_note(store, world_id: str) -> dict:
    world = store.read_world(world_id)
    scale = getattr(world, "scale", None) or {"state": "unknown",
                                              "meters_per_unit": None}
    if hasattr(scale, "to_json_dict"):
        scale = scale.to_json_dict()
    scale = dict(scale)
    scale["note"] = ("inherited unchanged from the sparse solve; the surface "
                     "stage makes no new scale claim")
    return scale


def _build(root, frames, params, median_depth, voxel, trunc, seconds,
           should_stop, progress) -> SurfaceResult:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trel = truncation_rel(params, frames.median_held_out)

    def band_floor(vx):
        # With a depth-proportional band the absolute part is only the floor.
        if trel > 0:
            return params.trunc_voxels * vx
        return truncation_for(params, vx, median_depth, frames.median_held_out)

    vol = SurfaceVolume(voxel, band_floor(voxel), device=device, trunc_rel=trel,
                        trunc_max=params.trunc_max_voxels * voxel)

    t = time.time()
    _status(root, state=STATE_RUNNING, stage=STAGE_FUSE)
    # Prepare each frame ONCE. Allocation and integration are two passes over
    # the same frames, and decoding a JPEG and a depth map twice cost more than
    # all the fusion arithmetic put together -- 296 s against 10 s. Held
    # compactly and in HOST memory: on the GPU they cost 1.84 MiB a frame, which
    # on a 20-30 minute walk is 5-8 GiB of VRAM the field needs.
    cached = []
    for z, ok, img, R, tt, w in frames.prepared(params, median_depth, device):
        cached.append((z.to(torch.float16).cpu(), ok.cpu(), img.to(torch.uint8).cpu(),
                       R, tt, w.to(torch.float16).cpu()))
        if _stopped(should_stop):
            return _stop(root, STAGE_FUSE, seconds)
    if not cached:
        return _unavailable(root, "no frame produced usable depth")

    # Allocate within the block budget. If the walk's surface would need more
    # blocks than `max_blocks`, coarsen the voxel -- block count scales with
    # surface AREA, so by the square root of the overshoot -- and ask again.
    # Allocation is about a second since key-space expansion, so a second pass
    # is cheap, and it happens before a byte of field exists.
    coarsened = 1.0
    attempts = 12
    for attempt in range(attempts):
        keys = [vol.blocks_for_depth(z.to(device).float(), ok.to(device), R, tt, frames.K)
                for z, ok, _img, R, tt, _w in cached]
        allk = torch.unique(torch.cat(keys))
        del keys
        # The keys just computed always belong to `vol`'s voxel: the loop only
        # coarsens when it will go round again, so it never reserves keys from
        # the previous voxel size on its last pass.
        if params.max_blocks <= 0 or allk.numel() <= params.max_blocks:
            break
        if attempt == attempts - 1:
            # The budget is what stands between a pathological walk and an
            # out-of-memory failure, so the last attempt does not quietly
            # reserve over it. Refused by name; the previous surface stands.
            reason = (f"the walk still needs {allk.numel()} blocks after "
                      f"coarsening the voxel x{coarsened:.2f} over {attempts} "
                      f"attempts, over the block budget of {params.max_blocks}; "
                      "not built")
            logger.warning("[Tower][WorldBuilder][surface] %s", reason)
            return _unavailable(root, reason)
        # At least 10% a round: once the band's shell radius is a whole number of
        # blocks, block count stops following voxel area smoothly and a pure
        # square-root step can stall just above the budget.
        factor = max(1.1, math.sqrt(allk.numel() / params.max_blocks) * 1.05)
        coarsened *= factor
        voxel *= factor
        trunc = truncation_for(params, voxel, median_depth, frames.median_held_out)
        logger.info("[Tower][WorldBuilder][surface] %d blocks exceeds the budget of "
                    "%d; voxel coarsened x%.2f to %.5f", allk.numel(),
                    params.max_blocks, coarsened, voxel)
        vol = SurfaceVolume(voxel, band_floor(voxel), device=device, trunc_rel=trel,
                        trunc_max=params.trunc_max_voxels * voxel)
    vol.reserve(allk)
    del allk
    # The band a sample at the scene scale actually got, cap included --
    # `truncation_for` is the uncapped request and overstated it whenever the
    # frames' error asked for more than `trunc_max_voxels`.
    trunc = float(vol.trunc_at(float(median_depth)))

    used = 0
    for i, (z, ok, img, R, tt, w) in enumerate(cached):
        vol.integrate(z.to(device).float(), ok.to(device), img.to(device).float(),
                      R, tt, frames.K, params=params,
                      weight_img=w.to(device).float())
        used += 1
        if progress is not None and i % 25 == 0:
            progress(STAGE_FUSE, i, len(frames))
        if _stopped(should_stop):
            return _stop(root, STAGE_FUSE, seconds)
    # The depth and validity stay (host memory, ~0.7 MiB a frame) for the
    # evidence filter after extraction; the images and weights do not.
    views = [(z, ok, R, tt) for z, ok, _img, R, tt, _w in cached]
    del cached
    seconds[STAGE_FUSE] = round(time.time() - t, 2)
    _status(root, state=STATE_RUNNING, stage=STAGE_MESH)

    t = time.time()
    fill_stats = None
    strong = None
    radius = params.fill_radius_voxels()
    if radius > 0:
        # Opt-in only; see `SurfaceParams.fill_gap_frac`. The flag per vertex
        # is used for the record below and then dropped: the format has no
        # slot for it, and the manifest's format identifier is what tells a
        # reader that some of these triangles were not measured.
        fill = fill_enclosed(vol, params.min_weight, radius,
                             params.fill_enclose_dirs, progress=progress)
        if params.fill_sealed_only:
            V, F, C, G, seal = extract_sealed(vol, params.min_weight, fill,
                                              progress=progress)
        else:
            V, F, C, G = vol.extract_mesh(params.min_weight, progress=progress,
                                          tag=fill.tag)
            seal = {"voxels_filled": fill.voxels,
                    "filled_vertices": int(np.asarray(G).sum())}
        del fill
        fill_stats = {"radius_voxels": radius,
                      "gap_frac": params.fill_gap_frac,
                      "enclose_dirs": params.fill_enclose_dirs,
                      "sealed_only": params.fill_sealed_only, **seal}
    elif params.low_weight_evidence:
        # Cubes observed at every corner but not to `min_weight` emit too, and
        # only the frame tests below may admit them (SurfaceParams.
        # low_weight_evidence). `strong` follows each face through the weld.
        V, F, C, strong = vol.extract_mesh(params.min_weight, progress=progress,
                                           weak_floor=0.0)
    else:
        V, F, C = vol.extract_mesh(params.min_weight, progress=progress)
    if strong is None:
        V, F, C, weld_stats = weld_mesh(V, F, C, quantum=voxel * 1e-3)
    else:
        V, F, C, weld_stats, source = weld_mesh(V, F, C, quantum=voxel * 1e-3,
                                                return_index=True)
        strong = strong[source]
    n_blocks, trunc_at = vol.n_blocks, vol.trunc_at
    # The field is done with; the filter below needs the memory more.
    vol.release_field()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    evidence_stats = None
    weak = None if strong is None else ~strong
    if len(F) and (params.min_support_frames > 0 or params.contradiction_ratio > 0):
        keep, evidence_stats = evidence_filter(V, F, views, frames.K, trunc_at, params,
                                               device, weak=weak)
        if weak is not None and params.low_weight_hidden_test and keep.any():
            # Low-weight sheets no supporting camera could see (SurfaceParams.
            # low_weight_hidden_test), tested against the surface just kept.
            hidden, hidden_stats = hidden_low_weight(V, F, keep, weak, views, frames.K,
                                                     trunc_at, device)
            keep &= ~hidden
            evidence_stats.update(hidden_stats)
            evidence_stats["weak_kept"] = (evidence_stats.get("weak_kept", 0)
                                           - hidden_stats["dropped_weak_hidden"])
            evidence_stats["faces_kept"] = int(keep.sum())
        V, F, C = keep_faces(V, F, C, keep)
    elif weak is not None and weak.any():
        # A low-weight face is admitted by the frame tests or not at all.
        V, F, C = keep_faces(V, F, C, ~weak)
    if not len(F):
        if evidence_stats and evidence_stats.get("faces_in"):
            s = evidence_stats
            return _unavailable(
                root, f"the evidence filter removed all {s['faces_in']} faces the "
                      f"field emitted: {s.get('dropped_support', 0)} were measured by "
                      f"fewer than {params.min_support_frames} distinct frames, "
                      f"{s.get('dropped_contradicted', 0)} were seen past by at least "
                      f"{params.contradiction_ratio:g}x as many frames as measured "
                      f"them, {s.get('dropped_back_facing', 0)} were seen only "
                      f"from behind, and {s.get('dropped_weak_seen_through', 0) + s.get('dropped_weak_parallax', 0) + s.get('dropped_weak_hidden', 0)} "
                      "low-weight faces were seen through, measured from too narrow "
                      "a baseline, or hidden by kept surface from every frame that "
                      "measured them")
        return _unavailable(
            root, "the fused field held no cell with enough evidence to emit "
                  "a surface")
    V, F, C, comp_stats = drop_small_components(V, F, C, params.min_component_frac)
    V, moved = taubin_smooth(V, F, params.smooth_iterations,
                             params.smooth_lambda, params.smooth_mu)
    seconds[STAGE_MESH] = round(time.time() - t, 2)
    snap_stats = None
    if params.plane_snap and len(F):
        if _stopped(should_stop):
            return _stop(root, STAGE_MESH, seconds)
        t = time.time()
        # After smoothing, so the snap is the last thing to move a vertex, and
        # against the depth the fusion used.
        V, snap_stats = snap_planes(V, F, views, frames.K, params, voxel, median_depth,
                                    device)
        seconds[STAGE_SNAP] = round(time.time() - t, 2)
    del views
    if _stopped(should_stop):
        return _stop(root, STAGE_MESH, seconds)

    t = time.time()
    _status(root, state=STATE_RUNNING, stage=STAGE_PACK)
    # A BUILD IS PUBLISHED AS A UNIT. Each build writes its levels under names
    # of its own, and the manifest -- written last, atomically, by the caller --
    # names them. Before this every build wrote `mesh_l<n>.bin`, so a stop or a
    # kill between levels, or between the last level and the manifest, left new
    # levels under an old manifest: the page served one surface while the
    # revision and the listing described another, and a later run reported
    # "already built" over it. Now a reader following any manifest reads only
    # the files that manifest was published with.
    build_id = f"{time.time_ns():x}{os.getpid():x}"
    levels = []
    source = (V, F, C)
    mobile_fit = None
    for level, target in enumerate(params.lod_face_targets):
        if _stopped(should_stop):
            # Pack is the longest stage and outlasts a hard stop's grace; it
            # must notice the stop between levels, not only at the end.
            _discard_unpublished(root, levels)
            return _stop(root, STAGE_PACK, seconds)
        # Each level from the previous one: see `SurfaceParams.lod_face_targets`.
        parent = source
        Vl, Fl, Cl = (V, F, C) if target <= 0 else decimate(
            *source, target, boundary_weight=params.lod_boundary_weight)
        N = vertex_normals(Vl, Fl)
        buf = write_mesh_bytes(Vl, Fl, Cl, N)
        if level == params.mobile_level and params.mobile_page_bytes > 0:
            Vl, Fl, Cl, buf, mobile_fit = _fit_mobile_page(
                parent, (Vl, Fl, Cl), buf, target, params.mobile_page_bytes,
                boundary_weight=params.lod_boundary_weight)
        source = (Vl, Fl, Cl)
        name = f"mesh_l{level}.{build_id}.bin"
        write_bytes_atomic(root / name, lambda handle, data=buf: handle.write(data))
        levels.append({"level": level, "vertices": int(len(Vl)),
                       "faces": int(len(Fl)), "bytes": len(buf), "file": name})
    if _stopped(should_stop):
        # A stop that arrived while the last level was being decimated. The
        # levels are on disk under this build's own names, but no manifest
        # names them, so nothing reads them and the previous surface stands --
        # and since nothing ever will, they are removed now.
        _discard_unpublished(root, levels)
        return _stop(root, STAGE_PACK, seconds)
    seconds[STAGE_PACK] = round(time.time() - t, 2)

    return SurfaceResult(
        state=STATE_OK, frames_used=used, frames_offered=frames.offered,
        vertices=int(len(V)), faces=int(len(F)), blocks=n_blocks,
        voxel=voxel, trunc=trunc, levels=levels, seconds=seconds,
        detail=json.dumps({"components": comp_stats,
                           "median_vertex_move_voxels": round(moved / voxel, 3),
                           "weld": weld_stats,
                           "voxel_coarsened_by": round(coarsened, 4),
                           "truncation_floor": band_floor(voxel),
                           "truncation_rel": trel,
                           "evidence_filter": evidence_stats,
                           "mobile_page_fit": mobile_fit,
                           "depth_consistency": frames.consistency,
                           "plane_snap": _snap_summary(snap_stats),
                           **({"enclosed_fill": fill_stats} if fill_stats else {})}),
    )


STAGE_SNAP = "snap"


def _snap_summary(stats):
    """The snap record for the manifest: counts and areas, and each plane's
    fit; not the rejected examples, which stay in the log."""
    if stats is None:
        return None
    keep = {k: v for k, v in stats.items() if k != "rejected_examples"}
    keep["planes"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}
                      for p in stats.get("planes", [])]
    return keep


def _fit_mobile_page(parent, mesh, buf, target, page_bytes, attempts: int = 4,
                     boundary_weight: float = 1.0):
    """The phone level, decimated from its parent until its PAGE fits.

    The largest level that fits, not a fixed face count: each attempt scales
    the face count by the byte overshoot (bytes per face barely move under
    decimation), with 2% of margin, from the same parent -- never from the
    previous attempt, which would compound decimation error. A mesh that still
    does not fit after `attempts` is kept and the record says so; the page
    chooser then serves the smallest level there is.
    """
    from tower.world_builder.surface_render import (  # noqa: PLC0415
        mesh_bytes_for_page,
        page_bytes_for_mesh,
    )

    budget = mesh_bytes_for_page(page_bytes)
    Vl, Fl, Cl = mesh
    tries = 0
    while len(buf) > budget and len(Fl) > 1 and tries < attempts:
        fit = max(1, int(len(Fl) * budget / len(buf) * 0.98))
        Vl, Fl, Cl = decimate(*parent, fit, boundary_weight=boundary_weight)
        buf = write_mesh_bytes(Vl, Fl, Cl, vertex_normals(Vl, Fl))
        tries += 1
    record = {"page_budget_bytes": int(page_bytes), "mesh_budget_bytes": int(budget),
              "face_target": int(target), "faces": int(len(Fl)),
              "mesh_bytes": len(buf), "page_bytes_estimate": page_bytes_for_mesh(len(buf)),
              "decimations": tries, "fits": len(buf) <= budget}
    if not record["fits"]:
        logger.warning("[Tower][WorldBuilder][surface] the phone level is %d bytes, over "
                       "the %d a %d-byte page allows", len(buf), budget, page_bytes)
    return Vl, Fl, Cl, buf, record


def _write_manifest(root, result, params, digest, pdigest, median_depth, scale,
                    scale_source=None, transients=None, keyframe_image_set=None):
    filled = params.fill_radius_voxels() > 0
    try:
        detail = json.loads(result.detail) if result.detail else None
    except ValueError:
        detail = None
    write_json_atomic(root / "manifest.json", {
        # The contract's table described `detail.*` keys that only ever
        # reached `status.json`, which the next status write replaces.
        "detail": detail,
        "schema_version": SURFACE_SCHEMA_VERSION,
        # WORLD-BUILDER-SURFACE.md section 3: filling unobserved space breaks
        # the format identifier rather than quietly relaxing its promise.
        "format": SURFACE_FORMAT_ENCLOSED_FILL if filled else SURFACE_FORMAT,
        "record": ("header, then uint16[3] quantised position, uint8[3] rgb, "
                   "int8[3] normal per vertex, then uint16/uint32 indices"),
        "built_at": time.time(),
        "input_digest": digest,
        "params_digest": pdigest,
        # The re-redacted keyframe set the colours came from, null for the
        # capture's own `images/` (APPEARANCE §6.5). A surface built from a set
        # the session no longer reads is not drawable
        # (`store.built_from_an_inactive_keyframe_set`).
        "keyframe_image_set": keyframe_image_set,
        "params": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in params.__dict__.items()},
        "median_scene_depth": median_depth,
        "scene_scale_source": scale_source,
        "voxel": result.voxel,
        "truncation": result.trunc,
        "frames_used": result.frames_used,
        "frames_offered": result.frames_offered,
        "vertices": result.vertices,
        "faces": result.faces,
        "levels": result.levels,
        "canonical_level": params.canonical_level,
        "mobile_level": params.mobile_level,
        "seconds": result.seconds,
        "scale": scale,
        # WORLD-BUILDER-SURFACE.md §2: whether the wearer's hands were masked
        # out of fusion. `state` other than `ok` means they were NOT.
        "transients": transients or {"state": "off", "detail": "not recorded"},
        "closure": (
            ("enclosed-fill: unobserved voxels bracketed by observed field in "
             f"at least {params.fill_enclose_dirs} of 26 directions within "
             f"{params.fill_gap_frac} of median scene depth were interpolated "
             "at min_weight; "
             + ("filled patches left with an open rim were removed"
                if params.fill_sealed_only else
                "filled patches left with an open rim were KEPT"))
            if filled else
            ("none: a cube emits surface only where all eight corners were "
             "observed, and a face is kept only where at least "
             "min_support_frames distinct frames measured it and fewer than "
             "contradiction_ratio times as many saw through it"
             + (("; where a corner fell short of min_weight, only if no frame "
                 "saw through the face and its supporting cameras spanned "
                 "low_weight_min_parallax"
                 + (", and not where the kept surface hid it from every frame "
                    "that measured it" if params.low_weight_hidden_test else ""))
                if params.low_weight_evidence else
                "; and all eight reached min_weight")
             + ", so unobserved space is absent rather than closed over; a hole "
               "may also be space the frames disagreed about")),
    })


def level_file(root: Path, entry: dict) -> Path | None:
    """The file a manifest's level entry names, or None if it names none safely.

    Manifests written before builds were published as a unit carry no `file`
    and used `mesh_l<n>.bin`. A `file` must be a bare name inside this
    directory: a manifest is data, and a path in it is not followed anywhere.
    """
    if not isinstance(entry, dict) or not isinstance(entry.get("level"), int):
        return None
    name = entry.get("file")
    if name is None:
        name = f"mesh_l{entry['level']}.bin"
    if (not isinstance(name, str) or "/" in name or "\\" in name
            or name.startswith(".") or not name.endswith(".bin")):
        return None
    return root / name


def level_file_whole(root: Path, entry: dict) -> Path | None:
    """`level_file`, and only if the file on disk is exactly the size the
    manifest recorded. Byte counts drive which level a phone is sent; a level
    trusted without this was served at 339,912 bytes under a 4,294-byte
    budget."""
    path = level_file(root, entry)
    if path is None or not isinstance(entry.get("bytes"), int):
        return None
    try:
        if path.stat().st_size != entry["bytes"]:
            return None
    except OSError:
        return None
    return path


PRUNE_GRACE_S = 120.0
STAGING_GRACE_S = 600.0


def _mark_superseded(root: Path, keep: set) -> None:
    """Stamp the files the manifest about to be replaced names with NOW.

    The prune grace below is measured on a file's mtime, and a level's mtime
    was its WRITE time: a previous build older than the grace lost its files
    the instant the next manifest landed, and a reader between reading that
    manifest and reading its level got `SurfaceUnavailable` (review 2, I1).
    Stamped BEFORE the new manifest is written, so a crash in between can
    only lengthen the grace, never skip it.
    """
    try:
        previous = json.loads((root / "manifest.json").read_text())
    except (OSError, ValueError):
        return
    if not isinstance(previous, dict):
        return
    now = time.time()
    for entry in previous.get("levels") or []:
        path = level_file(root, entry)
        if path is None or path.name in keep:
            continue
        try:
            os.utime(path, (now, now))
        except OSError:
            pass


def _prune_superseded_levels(root: Path, keep: set,
                             older_than_s: float = PRUNE_GRACE_S) -> None:
    """Remove level files no current manifest names, once they are old.

    Not immediately: a reader that read the previous manifest a moment ago is
    still entitled to that manifest's files. Two minutes is far longer than any
    page composition, and far shorter than the gap between builds that matters
    for disk. "Old" counts from SUPERSESSION (`_mark_superseded`), not from
    when the file was written. A level no manifest ever named -- a pack that
    was stopped or killed -- has only its write time, and no reader.

    Staging files of a killed level write (`write_bytes_atomic` does not clean
    up after `TerminateProcess`) are removed once they are older than any
    level write takes; nothing reads a staging file.
    """
    now = time.time()
    for path in root.glob("mesh_l*.bin"):
        if path.name in keep:
            continue
        try:
            if now - path.stat().st_mtime >= older_than_s:
                path.unlink()
        except OSError:
            pass
    for path in root.glob("mesh_l*.bin.p*.tmp"):
        try:
            if now - path.stat().st_mtime >= STAGING_GRACE_S:
                path.unlink()
        except OSError:
            pass


def _sweep_unnamed_levels(root: Path) -> None:
    """`_prune_superseded_levels` against the manifest on disk now, if it reads.

    Only when it reads: an unreadable manifest names nothing, and pruning
    against an empty keep set would delete the published surface.
    """
    try:
        current = json.loads((root / "manifest.json").read_text())
    except FileNotFoundError:
        current = {"levels": []}
    except (OSError, ValueError):
        return
    if not isinstance(current, dict):
        return
    keep = set()
    for entry in current.get("levels") or []:
        path = level_file(root, entry)
        if path is None:
            return
        keep.add(path.name)
    _prune_superseded_levels(root, keep)


def _discard_unpublished(root: Path, levels: list) -> None:
    """Remove the level files a stopped pack wrote. No manifest names them, so
    no reader can hold them."""
    for entry in levels:
        try:
            (root / entry["file"]).unlink()
        except (OSError, KeyError, TypeError):
            pass


# The `detail` of a result that built nothing because the artifact on disk is
# already this solve's, with these parameters.
ALREADY_BUILT = "already built from this solve with these parameters (--force rebuilds)"


def _already_built(root: Path, digest, pdigest, force: bool):
    if force:
        return None
    path = root / "manifest.json"
    if not path.exists():
        return None
    try:
        man = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if digest is None or man.get("input_digest") != digest:
        return None
    if man.get("params_digest") != pdigest:
        return None
    # A surface fused from the plain affine because the consistency solve
    # FAILED (not refused: a refusal is a decision about the data) is not what
    # these parameters build; the next build tries the solve again.
    detail = man.get("detail") if isinstance(man.get("detail"), dict) else {}
    if ((detail or {}).get("depth_consistency") or {}).get("state") == "failed":
        return None
    levels = man.get("levels") or []
    if not levels or not all(level_file_whole(root, lv) for lv in levels):
        return None
    return SurfaceResult(
        state=STATE_OK,
        detail=ALREADY_BUILT,
        frames_used=man.get("frames_used", 0),
        frames_offered=man.get("frames_offered", 0),
        vertices=man.get("vertices", 0), faces=man.get("faces", 0),
        voxel=man.get("voxel", 0.0), trunc=man.get("truncation", 0.0),
        levels=levels, seconds=man.get("seconds", {}),
    )


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def read_surface_manifest(store, world_id: str, session_id: str) -> dict | None:
    """The manifest, or None. Never raises on a malformed one: a surface that
    cannot be read is a surface the ladder falls past, not a 500."""
    path = surface_dir(store, world_id, session_id) / "manifest.json"
    try:
        man = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(man, dict) or man.get("format") != SURFACE_FORMAT:
        return None
    if man.get("schema_version") != SURFACE_SCHEMA_VERSION:
        return None
    return man


def read_surface_level(store, world_id: str, session_id: str, level: int,
                       manifest: dict | None = None) -> bytes:
    """The bytes of one level, as the manifest that describes it names them.

    Read through the manifest, and checked against its byte count, so a level
    from another build -- or a torn one -- is refused here and the ladder falls
    to the next rung rather than serving it.
    """
    root = surface_dir(store, world_id, session_id)
    man = manifest if manifest is not None else read_surface_manifest(
        store, world_id, session_id)
    entry = next((lv for lv in (man or {}).get("levels") or []
                  if isinstance(lv, dict) and lv.get("level") == level), None)
    path = level_file_whole(root, entry) if entry is not None else None
    if path is None:
        raise SurfaceUnavailable(
            f"surface level {level} is not readable for this session")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SurfaceUnavailable(
            f"surface level {level} is not readable for this session") from exc
    if len(data) != entry["bytes"]:
        raise SurfaceUnavailable(
            f"surface level {level} changed while it was being read")
    return data


def surface_currency(store, world_id: str, session_id: str,
                     manifest: dict | None) -> dict:
    """Whether the surface on disk was built from the solve now on disk.

    Reported rather than enforced. A surface behind the newest keyframes is
    still the best picture of the world there is, and hiding it during a walk
    is what left the gallery empty for a whole capture last time.
    """
    from tower.world_builder.global_solve import load_solution

    if manifest is None:
        return {"present": False, "current": False, "reason": "no surface artifact"}
    try:
        solution = load_solution(store, world_id, session_id)
    except Exception:  # noqa: BLE001
        solution = None
    live = getattr(solution, "input_digest", None)
    built = manifest.get("input_digest")
    if live is None or built is None:
        return {"present": True, "current": False,
                "reason": "the solve does not record an input digest"}
    return {"present": True, "current": live == built,
            "reason": None if live == built else "built from an earlier solve"}
