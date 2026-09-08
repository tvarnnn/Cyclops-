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

import json
import logging
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
    make_backend,
    project,
    unproject,
    validity_mask,
    POINT_STRIDE_BYTES,
    voxel_reduce,
    write_points_bin,
)

logger = logging.getLogger(__name__)

DENSE_SCHEMA_VERSION = 1


def dense_dir(store, world_id: str, session_id: str) -> Path:
    """`<world>/dense/<session>` -- beside `solve/`, never inside `derived/`."""
    return store.world_dir(world_id) / "dense" / session_id


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def _status(root: Path, **fields) -> None:
    _write_json(root / "status.json", {"schema_version": DENSE_SCHEMA_VERSION, **fields})


def _stopped(should_stop) -> bool:
    return bool(should_stop and should_stop())


# ---------------------------------------------------------------------------


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


def keyframe_image_bytes(store, world_id: str, session_id: str, keyframe_id: str,
                         source_path: str | None, redactor
                         ) -> tuple[bytes | None, str, bytes | None]:
    """The pixels the dense stage is allowed to read, and where they came from.

    THIS IS A PRIVACY BOUNDARY, not a convenience. `engine.py` redacts faces
    BEFORE persisting a keyframe image, deliberately, so that the bytes any
    later reconstruction reads are the redacted ones rather than raw frames
    sitting on disk behind a display filter. A dense stage that reached past
    that to the original capture would rebuild the room out of exactly the
    pixels the privacy transformation removed, at far higher density than the
    sparse cloud ever exposed.

    So: the world's own redacted keyframe image is used when it exists. When it
    does not -- worlds migrated between roots lost their `images/` directory --
    the raw frame is read and THE SAME REDACTION IS RE-APPLIED before anything
    looks at it. If the redactor is unavailable, the frame is refused rather
    than used raw.

    Returns (image_bytes, origin, raw_bytes_if_we_redacted_them). The third
    value exists so the caller can difference the two and learn exactly which
    pixels were filled -- see `redaction_fill_mask`.
    """
    seq = keyframe_id.rsplit(":", 1)[-1]
    p = store.images_dir(world_id, session_id) / f"{seq}.jpg"
    if p.exists():
        try:
            return p.read_bytes(), "world-keyframe", None
        except OSError:
            pass
    if not source_path or not Path(source_path).exists():
        return None, "absent", None
    try:
        raw = Path(source_path).read_bytes()
    except OSError:
        return None, "unreadable", None
    if redactor is None or not getattr(redactor, "available", False):
        return None, "refused-no-redactor", None
    return redactor.redact(raw).image_bytes, "raw-source-rereducted", raw


def redaction_fill_mask(image, raw=None, fill_value: int = 0,
                        min_area_fraction: float = 0.0015, dilate_px: int = 3):
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
        flat = (image.max(-1) <= fill_value + 8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            flat.astype(np.uint8), connectivity=4
        )
        mask = np.zeros((h, w), bool)
        min_area = max(64, int(min_area_fraction * h * w))
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask |= labels == i
    if dilate_px:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                          iterations=dilate_px).astype(bool)
    return mask


def run_depth_stage(
    store, world_id: str, session_id: str, solution, intrinsics, params: DenseParams,
    root: Path, *, should_stop=None, progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Undistort, predict depth, and align every posed keyframe in the component.

    The undistortion is the SOLVE's own -- `global_solve._undistort_maps` -- so
    the depth map and the poses are expressed in exactly the same camera. Doing
    this any other way silently shifts every back-projected point.
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

    redactor = FaceRedactor()
    if not redactor.available:
        logger.warning(
            "[Tower][WorldBuilder][dense] face redaction unavailable (%s); frames "
            "whose redacted keyframe image is missing will be REFUSED rather than "
            "read raw", redactor.unavailable_reason,
        )
    origins: dict[str, int] = {}

    backend = make_backend(params.backend)
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
    records: list[dict] = []
    t0 = time.time()
    obs_kf = solution.observations[:, 0]
    obs_pt = solution.observations[:, 2]

    for n, (ki, kid, pose) in enumerate(targets):
        if _stopped(should_stop):
            return {"stopped_after": n, "records": records, "seconds": time.time() - t0,
                    "camera": cam, "targets": len(targets), "image_origins": origins}
        if progress and n % 25 == 0:
            progress(STAGE_DEPTH, n, len(targets))

        data, origin, raw_bytes = keyframe_image_bytes(
            store, world_id, session_id, kid, sources.get(kid), redactor
        )
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

        original = None
        if raw_bytes is not None:
            original = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        fill = redaction_fill_mask(raw, original)
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

        disp = backend.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        m = obs_kf == ki
        if int(m.sum()) < params.min_sparse_points:
            records.append({"ki": int(ki), "ok": False,
                            "why": f"only {int(m.sum())} sparse observations"})
            continue
        uv = solution.observation_xy[m].astype(np.float64)
        R = np.asarray(pose["rotation"], float).reshape(3, 3)
        t = np.asarray(pose["translation"], float)
        _, zc = project(R, t, K, solution.xyz[obs_pt[m]].astype(np.float64))
        g = ((zc > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1)
             & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1))
        if int(g.sum()) < params.min_sparse_points:
            records.append({"ki": int(ki), "ok": False,
                            "why": f"only {int(g.sum())} sparse points inside the frame"})
            continue
        ui = np.clip(np.rint(uv[g, 0]).astype(int), 0, W - 1)
        vi = np.clip(np.rint(uv[g, 1]).astype(int), 0, H - 1)
        a, b, ho = align_frame(disp[vi, ui].astype(np.float64), zc[g])
        np.save(work / "depth" / f"{ki:05d}.npy", disp)
        records.append({"ki": int(ki), "kid": kid, "ok": True, "a": a, "b": b,
                        "n_points": int(g.sum()), "held_out_rel": ho,
                        "redaction_fill_fraction": fill_fraction,
                        "image_origin": origin})

    if progress:
        progress(STAGE_DEPTH, len(targets), len(targets))
    payload = {"records": records, "seconds": time.time() - t0, "camera": cam,
               "targets": len(targets), "backend": backend.name,
               "backend_licence": backend.licence, "stopped_after": None,
               "image_origins": origins,
               "redaction": getattr(redactor, "label", None) if redactor.available else None}
    _write_json(root / "align.json", payload)
    return payload


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
    for ki in kept:
        r = recs[ki]
        disp = np.load(work / "depth" / f"{ki:05d}.npy").astype(np.float32)
        den = disp - r["b"]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(den > 1e-6, r["a"] / den, np.nan).astype(np.float32)
        D[ki] = z
        ok = validity_mask(z, K, edge_rel=params.edge_rel,
                           max_grazing_deg=params.max_grazing_deg,
                           erode_px=params.erode_px)
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


def run_pack_stage(params: DenseParams, root: Path, scale: dict) -> dict:
    """The LOD ladder plus the manifest a viewer reads."""
    with np.load(root / "fused.npz") as z:
        X = z["xyz"].astype(np.float32)
        C = z["rgb"]
        F = z["confidence"]
        median_depth = float(z["median_depth"]) if "median_depth" in z else 1.0
    voxels = params.voxels_for(median_depth)
    m = F >= params.min_confidence
    X, C, F = X[m], C[m], F[m]
    lo = np.percentile(X, 0.2, 0)
    hi = np.percentile(X, 99.8, 0)
    inb = np.all((X >= lo) & (X <= hi), 1)
    X, C, F = X[inb], C[inb], F[inb]

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
        "canonical_level": params.canonical_level,
        "mobile_level": params.mobile_level,
        "median_scene_depth": median_depth,
        "bbox_min": [float(v) for v in lo],
        "bbox_max": [float(v) for v in hi],
        # Repeated, not re-derived. The dense stage makes no new scale claim.
        "scale": dict(scale),
        "levels": levels,
        "params": params.as_dict(),
    }
    _write_json(root / "manifest.json", manifest)
    return {"levels": levels, "points": levels[0]["points"]}


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

    solution = load_solution(store, world_id, session_id)
    if solution is None:
        _status(root, state=STATE_UNAVAILABLE, detail="no global solution for this session")
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
    _status(root, state=STATE_RUNNING, stage=STAGE_DEPTH, input_digest=digest,
            params=params.as_dict())

    try:
        align_path = root / "align.json"
        align = None
        if align_path.exists() and not force:
            try:
                cached = json.loads(align_path.read_text())
                if cached.get("digest") in (None, digest) and cached.get("stopped_after") is None:
                    align = cached
                    logger.info("[Tower][WorldBuilder][dense] reusing depth stage")
            except (OSError, ValueError):
                align = None
        if align is None:
            t = time.time()
            align = run_depth_stage(store, world_id, session_id, solution, intrinsics,
                                    params, root, should_stop=should_stop, progress=progress)
            align["digest"] = digest
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
        fuse = run_fuse_stage(solution, align, params, root,
                              should_stop=should_stop, progress=progress)
        seconds[STAGE_FUSE] = time.time() - t
        if fuse.get("stopped_after") is not None:
            _status(root, state=STATE_STOPPED, stage=STAGE_FUSE)
            return DenseResult(state=STATE_STOPPED, stopped_after=STAGE_FUSE,
                               detail="stopped during fusion", seconds=seconds)

        _status(root, state=STATE_RUNNING, stage=STAGE_PACK, input_digest=digest)
        t = time.time()
        pack = run_pack_stage(params, root, scale)
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
