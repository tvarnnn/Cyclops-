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
import logging
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from tower.storage import write_bytes_atomic, write_json_atomic
from tower.world_builder.surface import (
    STAGE_DEPTH,
    STAGE_FUSE,
    STAGE_MESH,
    STAGE_PACK,
    SURFACE_FORMAT,
    SURFACE_SCHEMA_VERSION,
    SurfaceParams,
    SurfaceResult,
    SurfaceUnavailable,
    SurfaceVolume,
    decimate,
    depth_validity,
    drop_small_components,
    taubin_smooth,
    truncation_for,
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
    return not _pid_is_running(pid)


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
        except (OSError, ValueError, AttributeError):
            return True
        return not _pid_is_running(pid)

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
    if align_path.exists():
        try:
            cached = json.loads(align_path.read_text())
        except (OSError, ValueError):
            cached = None
        if cached and _depth_cache_usable(cached, root, solution, dparams):
            return cached, root / "work"
        prior = cached

    align = run_depth_stage(store, world_id, session_id, solution, intrinsics,
                            dparams, root, should_stop=should_stop,
                            progress=progress, prior=prior)
    write_json_atomic(align_path, align)
    return align, root / "work"


def _depth_cache_usable(cached: dict, root: Path, solution, dparams) -> bool:
    """A cached depth stage is reusable only if it came from this solve, this
    network, and still has its per-frame maps on disk.

    That last clause is the one that bites: `prune_intermediates` deletes
    `work/` after a successful pack -- which is its whole purpose -- so a world
    that has already been densified has a valid `align.json` and no depth maps
    behind it. Trusting the cache key alone would send the fuse stage looking
    for files pruning removed.
    """
    if cached.get("input_digest") not in (None, solution.input_digest):
        return False
    if solution.input_digest is None:
        return False
    if cached.get("backend") not in (None, dparams.backend):
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


class _Frames:
    """The gated, posed frames of one session, with their depth on disk."""

    def __init__(self, align: dict, work: Path, solution, params: SurfaceParams):
        self.work = work
        self.kind = align.get("kind", "disparity")
        cam = align.get("camera") or solution.camera or {}
        self.K = np.array([[cam["fx"], 0, cam["cx"]],
                           [0, cam["fy"], cam["cy"]],
                           [0, 0, 1.0]], float)
        poses = solution.poses or {}
        self.items = []
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
            self.items.append((ki, float(r["a"]), float(r["b"]),
                               np.array(pose["rotation"], float).reshape(3, 3),
                               np.array(pose["translation"], float),
                               1.0 if ho is None else float(ho)))
        self.median_held_out = float(np.median(held)) if held else None
        self.offered = len(align.get("records", []))

    def __len__(self):
        return len(self.items)

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

        for ki, a, b, R, t, ho in self.items:
            pred, img, fill = self.load(ki)
            if pred is None or img.shape[:2] != pred.shape:
                continue
            z = _depth_from_prediction(pred, a, b, self.kind)
            zt = torch.as_tensor(np.asarray(z, np.float32), device=device)
            ok, cosang = depth_validity(zt, self.K, params, median_depth)
            if fill is not None:
                ok &= ~torch.as_tensor(fill, device=device)
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
              backend: str | None = None) -> SurfaceResult:
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
        solution = load_solution(store, world_id, session_id)
        if solution is None:
            return _unavailable(root, "no global solution for this session")
        session = store.read_session(world_id, session_id)
        intrinsics = session.intrinsics
        if intrinsics is None or getattr(intrinsics, "fx", None) is None:
            return _unavailable(root, "session has no intrinsics")

        digest = solution.input_digest
        pdigest = _params_digest(params, digest)

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

        median_depth = _median_scene_depth(frames)
        if not (median_depth > 0):
            return _unavailable(root, "the solve has no usable scene depth")
        voxel = params.voxel_frac * median_depth
        trunc = truncation_for(params, voxel, median_depth, frames.median_held_out)

        result = _build(root, frames, params, median_depth, voxel, trunc,
                        seconds, should_stop, progress)
        if result.state != STATE_OK:
            return result

        scale = _scale_note(store, world_id)
        _write_manifest(root, result, params, digest, pdigest, median_depth, scale)
        _status(root, state=STATE_OK, input_digest=digest,
                params_digest=pdigest, result=result.as_dict())
        return result

    except SurfaceUnavailable as exc:
        _status(root, state=STATE_UNAVAILABLE, detail=exc.reason)
        return SurfaceResult(state=STATE_UNAVAILABLE, detail=exc.reason)
    except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed silently
        logger.exception("[Tower][WorldBuilder][surface] %s/%s failed",
                         world_id, session_id)
        _status(root, state=STATE_FAILED, detail=str(exc))
        return SurfaceResult(state=STATE_FAILED, detail=str(exc))
    finally:
        lock.release()


def _unavailable(root: Path, detail: str) -> SurfaceResult:
    _status(root, state=STATE_UNAVAILABLE, detail=detail)
    return SurfaceResult(state=STATE_UNAVAILABLE, detail=detail)


def _stop(root: Path, stage: str, seconds: dict) -> SurfaceResult:
    _status(root, state=STATE_STOPPED, stage=stage)
    return SurfaceResult(state=STATE_STOPPED, stopped_after=stage, seconds=seconds)


def _median_scene_depth(frames: "_Frames", sample_frames: int = 24) -> float:
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
    step = max(1, len(frames) // sample_frames)
    chunks = []
    for ki, a, b, _R, _t, _ho in frames.items[::step]:
        pred, _img, _fill = frames.load(ki, image=False)
        if pred is None:
            continue
        z = _depth_from_prediction(pred, a, b, frames.kind)
        z = z[np.isfinite(z)]
        if z.size:
            chunks.append(z[::37])
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
    vol = SurfaceVolume(voxel, trunc, device=device)

    t = time.time()
    # Prepare each frame ONCE and hold it on the device. Allocation and
    # integration are two passes over the same frames, and decoding a JPEG and
    # a depth map twice cost more than all the fusion arithmetic put together
    # -- 296 s against 10 s of actual integration. Held compactly (float16
    # depth, uint8 colour, bool mask) a 400-keyframe walk is well under a
    # gigabyte beside a 2 GiB field.
    cached, keys = [], []
    for z, ok, img, R, tt, w in frames.prepared(params, median_depth, device):
        keys.append(vol.blocks_for_depth(z, ok, R, tt, frames.K))
        cached.append((z.to(torch.float16), ok, img.to(torch.uint8),
                       R, tt, w.to(torch.float16)))
        if _stopped(should_stop):
            return _stop(root, STAGE_FUSE, seconds)
    if not keys:
        return _unavailable(root, "no frame produced usable depth")
    vol.reserve(torch.cat(keys))
    del keys

    used = 0
    for i, (z, ok, img, R, tt, w) in enumerate(cached):
        vol.integrate(z.float(), ok, img.float(), R, tt, frames.K,
                      params=params, weight_img=w.float())
        used += 1
        if progress is not None and i % 25 == 0:
            progress(STAGE_FUSE, i, len(frames))
        if _stopped(should_stop):
            return _stop(root, STAGE_FUSE, seconds)
    del cached
    seconds[STAGE_FUSE] = round(time.time() - t, 2)

    t = time.time()
    V, F, C = vol.extract_mesh(params.min_weight, progress=progress)
    if not len(F):
        return _unavailable(
            root, "the fused field held no cell with enough evidence to emit "
                  "a surface")
    V, F, C, comp_stats = drop_small_components(V, F, C, params.min_component_frac)
    V, moved = taubin_smooth(V, F, params.smooth_iterations,
                             params.smooth_lambda, params.smooth_mu)
    seconds[STAGE_MESH] = round(time.time() - t, 2)
    if _stopped(should_stop):
        return _stop(root, STAGE_MESH, seconds)

    t = time.time()
    levels = []
    for level, target in enumerate(params.lod_face_targets):
        Vl, Fl, Cl = (V, F, C) if target <= 0 else decimate(V, F, C, target)
        N = vertex_normals(Vl, Fl)
        buf = write_mesh_bytes(Vl, Fl, Cl, N)
        write_bytes_atomic(root / f"mesh_l{level}.bin",
                           lambda handle, data=buf: handle.write(data))
        levels.append({"level": level, "vertices": int(len(Vl)),
                       "faces": int(len(Fl)), "bytes": len(buf)})
    seconds[STAGE_PACK] = round(time.time() - t, 2)

    return SurfaceResult(
        state=STATE_OK, frames_used=used, frames_offered=frames.offered,
        vertices=int(len(V)), faces=int(len(F)), blocks=vol.n_blocks,
        voxel=voxel, trunc=trunc, levels=levels, seconds=seconds,
        detail=json.dumps({"components": comp_stats,
                           "median_vertex_move_voxels": round(moved / voxel, 3)}),
    )


def _write_manifest(root, result, params, digest, pdigest, median_depth, scale):
    write_json_atomic(root / "manifest.json", {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "format": SURFACE_FORMAT,
        "record": ("header, then uint16[3] quantised position, uint8[3] rgb, "
                   "int8[3] normal per vertex, then uint16/uint32 indices"),
        "built_at": time.time(),
        "input_digest": digest,
        "params_digest": pdigest,
        "params": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in params.__dict__.items()},
        "median_scene_depth": median_depth,
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
        "closure": ("none: a cell emits surface only where accumulated "
                    "evidence reached min_weight, so unobserved space is "
                    "absent rather than closed over"),
    })


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
    levels = man.get("levels") or []
    if not levels or not all((root / f"mesh_l{i}.bin").exists()
                             for i in range(len(levels))):
        return None
    return SurfaceResult(
        state=STATE_OK, frames_used=man.get("frames_used", 0),
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


def read_surface_level(store, world_id: str, session_id: str, level: int) -> bytes:
    path = surface_dir(store, world_id, session_id) / f"mesh_l{level}.bin"
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SurfaceUnavailable(
            f"surface level {level} is not readable for this session") from exc


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
