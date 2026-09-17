"""The appearance stage, orchestrated over a solved, surfaced session.

Contract: `docs/contracts/WORLD-BUILDER-APPEARANCE.md`.

Reads the solve (poses, camera), the surface artifact (its phone level is the
proxy), the depth stage's records and per-frame maps under `dense/<sid>/work/`
(fill masks, raw predictions), and the session's REDACTED keyframes through the
one provenance function in `appearance.py`. Writes only under
`<world>/appearance/<session>/`.

Published the way the surface artifact is (`surface_pipeline.py`): one build
at a time per session under a lock; every file written atomically under its
content name; the manifest last; superseded files pruned after a grace; a
stop between stages leaves the previous artifact standing and nothing that
looks finished.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tower.storage import write_bytes_atomic, write_json_atomic
from tower.world_builder import appearance as A

logger = logging.getLogger(__name__)

STATE_OK = "ok"
STATE_RUNNING = "running"
STATE_FAILED = "failed"
STATE_STOPPED = "stopped"
STATE_UNAVAILABLE = "unavailable"

STAGE_PROVENANCE = "provenance"
STAGE_DETECTOR = "detector"
STAGE_PROXY = "proxy"
STAGE_OCCLUDERS = "occluders"
STAGE_EXPOSURE = "exposure"
STAGE_TRANSIENTS = "transients"
STAGE_SELECTION = "selection"
STAGE_ENCODE = "encode"

PRUNE_GRACE_S = 120.0
STAGING_GRACE_S = 600.0
ALREADY_BUILT = "already built from these inputs with these parameters (--force rebuilds)"
STALE_LABEL_DETAIL = "appearance is stale against the session's redaction record"

_DIGEST_CHARS = frozenset("0123456789abcdef")


def appearance_dir(store, world_id: str, session_id: str) -> Path:
    """`<world>/appearance/<session>` -- beside `surface/` and `dense/`."""
    return store.world_dir(world_id) / "appearance" / session_id


@dataclass
class AppearanceResult:
    state: str
    detail: str | None = None
    build_id: str | None = None
    keyframes: int = 0
    phone: int = 0
    excluded: int = 0
    chunks: int = 0
    bytes: int = 0
    seconds: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _status(root: Path, **fields) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / "status.json", {
        "schema_version": A.APPEARANCE_SCHEMA_VERSION, "pid": os.getpid(),
        "updated_at": time.time(), **fields})


def status_is_stale(status: dict) -> bool:
    from tower.world_builder.surface_pipeline import status_is_stale as surface_stale  # noqa: PLC0415

    return surface_stale(status)


def _lock(root: Path):
    from tower.world_builder.surface_pipeline import _SurfaceLock  # noqa: PLC0415

    lock = _SurfaceLock(root)
    lock.path = root / ".appearance.lock"
    return lock


def _stopped(should_stop) -> bool:
    return bool(should_stop and should_stop())


# 32 hex characters: the first 128 bits of SHA-256. Not 64, because a Windows
# path is capped at 260 characters and a content name, its atomic-write staging
# suffix, and a world root in a project tree came to 263 on the first canonical
# build -- `write_bytes_atomic` failed with "No such file or directory".
DIGEST_HEX = 32


def content_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:DIGEST_HEX]


def is_digest(value) -> bool:
    return isinstance(value, str) and len(value) == DIGEST_HEX and set(value) <= _DIGEST_CHARS


def _file_name(kind: str, digest: str) -> str:
    return f"{kind[0]}.{digest}.bin"


def named_files(manifest: dict | None) -> dict:
    """{file name: bytes} for every file a manifest names."""
    out = {}
    if not isinstance(manifest, dict):
        return out
    proxy = manifest.get("proxy") or {}
    if is_digest(proxy.get("digest")) and isinstance(proxy.get("bytes"), int):
        out[_file_name("proxy", proxy["digest"])] = proxy["bytes"]
    for chunk in manifest.get("chunks") or []:
        if isinstance(chunk, dict) and is_digest(chunk.get("digest")) \
                and isinstance(chunk.get("bytes"), int):
            out[_file_name("chunk", chunk["digest"])] = chunk["bytes"]
    return out


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------


def build_appearance(store, world_id: str, session_id: str, *,
                     params: A.AppearanceParams | None = None,
                     should_stop=None, progress=None, force: bool = False,
                     redactor_factory=None, device=None,
                     transient_backend_factory=None) -> AppearanceResult:
    """Build the appearance artifact of one solved, surfaced session."""
    params = params or A.AppearanceParams()
    root = appearance_dir(store, world_id, session_id)
    root.mkdir(parents=True, exist_ok=True)
    lock = _lock(root)
    if not lock.acquire():
        detail = "another appearance build of this session is already running"
        logger.info("[Tower][WorldBuilder][appearance] %s/%s: %s", world_id, session_id, detail)
        return AppearanceResult(state=STATE_UNAVAILABLE, detail=detail)
    try:
        _sweep_unnamed(root)
        return _build(store, world_id, session_id, root, params, should_stop, progress,
                      force, redactor_factory, device, transient_backend_factory)
    except A.AppearanceUnavailable as exc:
        logger.warning("[Tower][WorldBuilder][appearance] %s/%s not built: %s",
                       world_id, session_id, exc.reason)
        _status(root, state=STATE_UNAVAILABLE, detail=exc.reason)
        return AppearanceResult(state=STATE_UNAVAILABLE, detail=exc.reason)
    except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed silently
        logger.exception("[Tower][WorldBuilder][appearance] %s/%s failed", world_id, session_id)
        _status(root, state=STATE_FAILED, detail=str(exc))
        return AppearanceResult(state=STATE_FAILED, detail=str(exc))
    finally:
        lock.release()


def _build(store, world_id, session_id, root, params, should_stop, progress, force,
           redactor_factory, device, transient_backend_factory=None) -> AppearanceResult:
    from tower.world_builder import transients as T  # noqa: PLC0415
    from tower.world_builder.dense import depth_from_prediction  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import FILL_RULE, dense_dir  # noqa: PLC0415
    from tower.world_builder.global_solve import load_solution  # noqa: PLC0415
    from tower.world_builder.surface import read_mesh_bytes  # noqa: PLC0415
    from tower.world_builder.surface_pipeline import (  # noqa: PLC0415
        read_surface_level,
        read_surface_manifest,
    )

    seconds: dict = {}
    t0 = time.time()
    world = store.read_world(world_id)
    if getattr(world, "images_purged", False):
        raise A.AppearanceUnavailable("this world's keyframe imagery was purged")
    solution = load_solution(store, world_id, session_id)
    if solution is None or not solution.camera:
        raise A.AppearanceUnavailable("no global solution for this session")
    try:
        session = store.read_session(world_id, session_id)
    except Exception:  # noqa: BLE001 -- an unreadable record is a refusal, not a crash
        raise A.AppearanceUnavailable("the session record is unreadable") from None
    intrinsics = session.intrinsics
    if intrinsics is None or getattr(intrinsics, "fx", None) is None:
        raise A.AppearanceUnavailable("session has no intrinsics")
    surface = read_surface_manifest(store, world_id, session_id)
    if surface is None:
        raise A.AppearanceUnavailable("no surface artifact to use as the proxy")
    level = int(surface.get("mobile_level", 0))
    proxy_bytes = read_surface_level(store, world_id, session_id, level, manifest=surface)
    proxy_digest = content_digest(proxy_bytes)
    V, F, _C, _N = read_mesh_bytes(proxy_bytes)
    if not len(F):
        raise A.AppearanceUnavailable("the surface's phone level is empty")

    droot = dense_dir(store, world_id, session_id)
    try:
        align = json.loads((droot / "align.json").read_text())
    except (OSError, ValueError):
        raise A.AppearanceUnavailable("the depth stage has no readable align.json") from None
    depth_dir = droot / "work" / "depth"
    records = {int(r["ki"]): r for r in align.get("records") or []
               if isinstance(r, dict) and r.get("ki") is not None}
    kind = align.get("kind", "disparity")

    policy = A.resolve_label_policy(store, world_id, session_id, redactor_factory)
    cam = solution.camera
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    W, H = int(cam["width"]), int(cam["height"])
    undistorter = A.Undistorter(intrinsics, cam)

    # -- candidates and the cheap hash pass ---------------------------------
    excluded: list[dict] = []
    candidates = []
    for ki, kid in enumerate(solution.keyframe_ids):
        pose = (solution.poses or {}).get(kid)
        if not pose or pose.get("rotation") is None or pose.get("translation") is None:
            excluded.append({"ki": ki, "reason": "no-pose"})
            continue
        rec = records.get(ki)
        if not rec or not rec.get("ok"):
            excluded.append({"ki": ki, "reason": "no-depth-fit"})
            continue
        candidates.append((ki, kid, pose, rec))
    hashes = []
    for ki, kid, _pose, _rec in candidates:
        src = A.keyframe_source(store, world_id, session_id, kid, ki, policy=policy,
                                align_record=None, depth_dir=None,
                                undistorter=undistorter, hash_only=True)
        hashes.append((kid, src.source_sha1 or "missing"))
    frame_digest = A.per_frame_sha1_digest(hashes)

    # -- transient detector masks (contract §5.3a) ----------------------------
    # Before the digest: which masks exist is an input. Cached per keyframe
    # beside the depth work, so this computes only keyframes no earlier build
    # (usually the surface stage, just before) already masked. A machine that
    # cannot run the detector gets `unavailable`, recorded, never "masked".
    td = time.time()
    _status(root, state=STATE_RUNNING, stage=STAGE_DETECTOR)
    tparams = T.TransientParams(mode=params.transient_detector)
    treport = T.ensure_transient_masks(
        store, world_id, session_id, [(ki, kid) for ki, kid, _p, _r in candidates],
        intrinsics=intrinsics, camera=cam, align_records=records, depth_dir=depth_dir,
        params=tparams, policy=policy, backend_factory=transient_backend_factory,
        should_stop=should_stop, progress=progress)
    if treport.state == T.STATE_STOPPED:
        return _stop(root, STAGE_DETECTOR, seconds)
    seconds[STAGE_DETECTOR] = round(time.time() - td, 2)
    encoders = A.encoder_versions(params)
    digest_inputs = {
        "schema": A.APPEARANCE_SCHEMA_VERSION,
        "input_digest": solution.input_digest,
        "proxy": proxy_digest,
        "surface_built_at": surface.get("built_at"),
        "depth_cache_key": align.get("cache_key"),
        "session_redaction": policy.session_redaction,
        "keyframe_image_set": getattr(policy.image_set, "cache_token", None),
        "redactor_applied_here": policy.redactor_label,
        "fill_rule": FILL_RULE,
        "unobserved_rule": A.UNOBSERVED_RULE,
        "alpha_ring_px": A.ALPHA_RING_PX,
        "per_frame_sha1_digest": frame_digest,
        "transients": {"rule": tparams.rule_id(), "state": treport.state,
                       "frames_digest": treport.frames_digest()},
        "params": params.as_dict(),
        "encoders": {k: [v["encoder"], v["version"], v["quality"], v["available"]]
                     for k, v in encoders.items()},
    }
    pdigest = hashlib.sha256(json.dumps(digest_inputs, sort_keys=True,
                                        default=str).encode()).hexdigest()
    done = _already_built(root, pdigest, force)
    if done is not None:
        _status(root, state=STATE_OK, params_digest=pdigest, result=done.as_dict())
        return done

    tp = time.time()
    _status(root, state=STATE_RUNNING, stage=STAGE_PROVENANCE, params_digest=pdigest)
    frames: list[A.PreparedFrame] = []
    refused: dict = {}
    for n, (ki, kid, pose, rec) in enumerate(candidates):
        if _stopped(should_stop):
            return _stop(root, STAGE_PROVENANCE, seconds)
        if progress is not None and n % 50 == 0:
            progress(STAGE_PROVENANCE, n, len(candidates))
        src = A.keyframe_source(store, world_id, session_id, kid, ki, policy=policy,
                                align_record=rec, depth_dir=depth_dir,
                                undistorter=undistorter)
        if src.refused:
            refused[src.refused] = refused.get(src.refused, 0) + 1
            excluded.append({"ki": ki, "reason": src.refused})
            continue
        R = np.asarray(pose["rotation"], float).reshape(3, 3)
        t = np.asarray(pose["translation"], float)
        det = treport.mask(ki)
        if det is not None and det.shape != src.unobserved.shape:
            det = None
        frames.append(A.PreparedFrame(source=src, R=R, t=t, zp=None, occluder=None,
                                      occluder_record=None, detector=det))
    # From its own start: the detector stage now runs before it (the first
    # canonical build reported 18.3 s here, 15.4 of them the detector).
    seconds[STAGE_PROVENANCE] = round(time.time() - tp, 2)

    # -- proxy depth and occluders ------------------------------------------
    t1 = time.time()
    _status(root, state=STATE_RUNNING, stage=STAGE_OCCLUDERS, params_digest=pdigest)
    caster = A.ProxyCaster(V, F)
    kept = []
    for n, fr in enumerate(frames):
        if _stopped(should_stop):
            return _stop(root, STAGE_OCCLUDERS, seconds)
        ki = fr.source.ki
        zp = caster.depth(fr.R, fr.t, K, W, H)
        if not np.isfinite(zp).any():
            excluded.append({"ki": ki, "reason": "no-proxy-in-view"})
            continue
        pred = None
        for name in (f"{ki:05d}_pred.npy", f"{ki:05d}.npy"):
            try:
                pred = np.load(depth_dir / name).astype(np.float32)
                break
            except (OSError, ValueError):
                continue
        if pred is None or pred.shape != (H, W):
            excluded.append({"ki": ki, "reason": "no-depth-prediction"})
            continue
        rec = records[ki]
        zs = depth_from_prediction(pred, float(rec["a"]), float(rec["b"]), kind)
        occ, occ_rec = A.occluder_mask(zs, zp, fr.source.unobserved, params)
        fr.zp = zp.astype(np.float16)
        fr.occluder = occ
        fr.occluder_record = occ_rec
        fr.sharpness = A.sharpness(fr.source.rgb, fr.transparent_core)
        kept.append(fr)
        if progress is not None and n % 50 == 0:
            progress(STAGE_OCCLUDERS, n, len(frames))
    frames = kept
    seconds[STAGE_OCCLUDERS] = round(time.time() - t1, 2)
    if not frames:
        raise A.AppearanceUnavailable(
            "no keyframe survived provenance, pose, depth and proxy checks"
            + (f" (refused: {refused})" if refused else ""))

    # -- exposure -----------------------------------------------------------
    t2 = time.time()
    _status(root, state=STATE_RUNNING, stage=STAGE_EXPOSURE, params_digest=pdigest)
    opaque = [~A.dilate(fr.transparent_core, A.ALPHA_RING_PX) for fr in frames]
    gains, gain_obs, exposure = A.solve_gains(
        [fr.source.rgb for fr in frames], opaque, [fr.zp for fr in frames],
        [fr.R for fr in frames], [fr.t for fr in frames], K, params, device=device,
        should_stop=should_stop)
    if gains is None:
        return _stop(root, STAGE_EXPOSURE, seconds)
    # the per-keyframe tilt travels on the keyframes, not in the record
    slopes = exposure.pop("slopes", None)
    vignette = exposure.get("vignette")
    seconds[STAGE_EXPOSURE] = round(time.time() - t2, 2)

    # -- transients: what most other keyframes did not see there ------------
    t2b = time.time()
    _status(root, state=STATE_RUNNING, stage=STAGE_TRANSIENTS, params_digest=pdigest)
    votes = A.transient_votes([fr.source.rgb for fr in frames], opaque,
                              [fr.zp for fr in frames], gains,
                              [fr.R for fr in frames], [fr.t for fr in frames], K, params,
                              device=device, should_stop=should_stop,
                              slopes=slopes, vignette=vignette)
    if votes is None:
        return _stop(root, STAGE_TRANSIENTS, seconds)
    for i, (fr, (grid, vrec)) in enumerate(zip(frames, votes)):
        near = fr.occluder
        transient = A.transient_mask(grid, (H, W), fr.zp.astype(np.float32), params)
        fr.extra["near_fraction"] = round(float(near.mean()), 4)
        fr.extra["transient_fraction"] = round(float(transient.mean()), 4)
        fr.extra["transient"] = vrec
        fr.occluder = near | transient
        opaque[i] = ~A.dilate(fr.transparent_core, A.ALPHA_RING_PX)
    seconds[STAGE_TRANSIENTS] = round(time.time() - t2b, 2)

    # -- selection ----------------------------------------------------------
    t3 = time.time()
    _status(root, state=STATE_RUNNING, stage=STAGE_SELECTION, params_digest=pdigest)
    sharp = np.array([fr.sharpness for fr in frames], float)
    med = float(np.median(sharp[sharp > 0])) if (sharp > 0).any() else 1.0
    for fr, s, op in zip(frames, sharp, opaque):
        q = float(np.clip(s / med, params.quality_floor, 1.0))
        fr.quality = q * (1.0 - 0.5 * float((~op).mean()))
        vrec = fr.extra.get("transient")
        if vrec and not vrec.get("applied", True):
            # Disagrees with the rest of the walk nearly everywhere: a
            # misregistered keyframe. Kept, but chosen for the phone last.
            fr.quality *= max(params.quality_floor, 1.0 - vrec["disagreeing_fraction"])
    zsamp = np.concatenate([fr.zp[::8, ::8][np.isfinite(fr.zp[::8, ::8])].astype(np.float32)
                            for fr in frames])
    z_ref = float(np.median(zsamp)) if zsamp.size else 1.0
    pts, nrm = A.sample_proxy_points(V, F, params.selection_samples, params.seed)
    scores = A.score_points(pts, nrm, [fr.zp for fr in frames], opaque,
                            [fr.R for fr in frames], [fr.t for fr in frames], K,
                            [fr.quality for fr in frames], z_ref, params, device=device)
    order, marginal, objective = A.greedy_select(scores, params, device=device)
    rank = {idx: r for r, idx in enumerate(order)}
    selection = {
        "phone_budget": params.phone_budget, "phone": len(order),
        "tower": len(frames) - len(order), "objective": round(objective, 3),
        "z_ref": round(z_ref, 4), "samples": int(len(pts)),
        "coverage": {"phone": A.coverage(scores, order),
                     "all": A.coverage(scores, range(len(frames)))},
    }
    seconds[STAGE_SELECTION] = round(time.time() - t3, 2)
    if _stopped(should_stop):
        return _stop(root, STAGE_SELECTION, seconds)

    # -- encoding -----------------------------------------------------------
    t4 = time.time()
    _status(root, state=STATE_RUNNING, stage=STAGE_ENCODE, params_digest=pdigest)
    astc_ok, astc_why = A.astc_available()
    threads = max(1, min(8, (os.cpu_count() or 2) // 2))
    phone = [frames[i] for i in order]
    tower = sorted((fr for i, fr in enumerate(frames) if i not in rank),
                   key=lambda fr: fr.source.ki)
    entry_chunks: dict = {}
    chunks: list[dict] = []
    written: list[str] = []

    def emit(group, encoding, tier):
        blobs = []
        for fr in group:
            rgba = A.rgba_for(fr.source.rgb, fr.transparent_core, params.alpha_feather_px)
            blobs.append(A.encode_astc(rgba, params.astc_quality, threads)
                         if encoding == A.ENC_ASTC else A.encode_webp(rgba, params.webp_quality))
        buf = A.pack_chunk(encoding, W, H, blobs)
        digest = content_digest(buf)
        name = _file_name("chunk", digest)
        _write_once(root / name, buf)
        written.append(name)
        chunks.append({"digest": digest, "bytes": len(buf), "encoding": encoding,
                       "slots": len(group), "tier": tier})
        for slot, fr in enumerate(group):
            entry_chunks.setdefault(fr.source.ki, {})[encoding] = {"digest": digest, "slot": slot}

    step = max(1, params.chunk_slots)
    for c0 in range(0, len(phone), step):
        if _stopped(should_stop):
            return _stop(root, STAGE_ENCODE, seconds, discard=(root, written))
        group = phone[c0:c0 + step]
        if astc_ok:
            emit(group, A.ENC_ASTC, A.TIER_PHONE)
        emit(group, A.ENC_WEBP, A.TIER_PHONE)
    for c0 in range(0, len(tower), step):
        if _stopped(should_stop):
            return _stop(root, STAGE_ENCODE, seconds, discard=(root, written))
        emit(tower[c0:c0 + step], A.ENC_WEBP, A.TIER_TOWER)
    proxy_name = _file_name("proxy", proxy_digest)
    _write_once(root / proxy_name, proxy_bytes)
    written.append(proxy_name)
    seconds[STAGE_ENCODE] = round(time.time() - t4, 2)

    # -- publish ------------------------------------------------------------
    now_label, now_set = A.keyframe_set_identity(store, world_id, session_id)
    built_set = getattr(policy.image_set, "cache_token", None)
    if now_label != policy.session_redaction or now_set != built_set:
        _discard_unpublished(root, written)
        raise A.AppearanceUnavailable(
            f"the session's redaction label changed during the build "
            f"({policy.session_redaction!r} -> {now_label!r}, keyframe set "
            f"{built_set!r} -> {now_set!r}); not published")
    if _stopped(should_stop):
        return _stop(root, STAGE_ENCODE, seconds, discard=(root, written))

    keyframes = []
    for i, fr in enumerate(frames):
        src = fr.source
        core = fr.transparent_core
        keyframes.append({
            "id": A.opaque_keyframe_id(session_id, src.keyframe_id),
            "ki": src.ki,
            "tier": A.TIER_PHONE if i in rank else A.TIER_TOWER,
            "rank": rank.get(i),
            "width": W, "height": H,
            "intrinsics": {k: float(cam[k]) for k in ("fx", "fy", "cx", "cy")},
            "rotation": [float(x) for x in fr.R.reshape(-1)],
            "translation": [float(x) for x in fr.t.reshape(-1)],
            "gain": [round(float(g), 5) for g in gains[i]],
            "gain_observations": int(gain_obs[i]),
            "gain_slope": ([0.0, 0.0] if slopes is None
                           else [round(float(v), 5) for v in slopes[i]]),
            "quality": round(fr.quality, 4),
            "sharpness": round(fr.sharpness, 2),
            "unobserved_fraction": round(float(src.unobserved.mean()), 4),
            "occluder_fraction": round(float(fr.occluder.mean()), 4),
            "transparent_fraction": round(float((~opaque[i]).mean()), 4),
            "near_fraction": fr.extra.get("near_fraction"),
            "transient_fraction": fr.extra.get("transient_fraction"),
            "detector_fraction": (None if fr.detector is None
                                  else round(float(fr.detector.mean()), 4)),
            "transient_mask": (None if fr.detector is None
                               else {"mode": tparams.mode, "rule": tparams.rule_id()}),
            "occluder": fr.occluder_record,
            "transient": fr.extra.get("transient"),
            "source_sha1": src.source_sha1, "image_sha1": src.image_sha1,
            "origin": src.origin, "mask_origin": src.mask_origin,
            "chunks": entry_chunks.get(src.ki, {}),
        })
    occ_px = [int(fr.occluder.sum()) for fr in frames]
    near_frac = [fr.extra.get("near_fraction") or 0.0 for fr in frames]
    tr_frac = [fr.extra.get("transient_fraction") or 0.0 for fr in frames]
    det_frac = [float(fr.detector.mean()) if fr.detector is not None else 0.0 for fr in frames]
    inconsistent = [fr.source.ki for fr in frames
                    if fr.extra.get("transient") and not fr.extra["transient"].get("applied", True)]
    seconds["total"] = round(time.time() - t0, 2)
    build_id = f"{time.time_ns():x}{os.getpid():x}"
    manifest = {
        "format": A.APPEARANCE_FORMAT,
        "schema_version": A.APPEARANCE_SCHEMA_VERSION,
        "build_id": build_id,
        "built_at": time.time(),
        "quality": params.quality,
        "input_digest": solution.input_digest,
        "params_digest": pdigest,
        "params": params.as_dict(),
        "appearance_provenance": {
            "session_redaction": policy.session_redaction,
            # The re-redacted keyframe set these pixels came from (contract
            # section 6.5); null for the capture's own keyframes.
            "keyframe_image_set": getattr(policy.image_set, "cache_token", None),
            "redaction_effective": policy.effective,
            "redactor_applied_here": policy.redactor_label,
            "label_trusted": policy.trusted,
            "fill_rule": FILL_RULE,
            "unobserved_rule": A.UNOBSERVED_RULE,
            "alpha_ring_px": A.ALPHA_RING_PX,
            "source": A.SOURCE_SESSION_KEYFRAMES,
            "frames": {"used": len(frames), "refused": refused},
            "per_frame_sha1_digest": frame_digest,
            "privacy_tags": list(getattr(session, "privacy_tags", ()) or ()),
            "retains_raw_imagery": bool(getattr(session, "retains_raw_imagery", True)),
            "note": ("best-effort face redaction with measured false negatives; not "
                     "anonymised; screens, documents and bodies are not redacted"),
        },
        "camera": {"fx": float(cam["fx"]), "fy": float(cam["fy"]), "cx": float(cam["cx"]),
                   "cy": float(cam["cy"]), "width": W, "height": H},
        "proxy": {"digest": proxy_digest, "bytes": len(proxy_bytes),
                  "vertices": int(len(V)), "faces": int(len(F)),
                  "format": "wb-surface-mesh/1",
                  "source": {"surface_built_at": surface.get("built_at"),
                             "surface_input_digest": surface.get("input_digest"),
                             "surface_params_digest": surface.get("params_digest"),
                             "surface_quality": (surface.get("params") or {}).get("quality"),
                             "level": level}},
        "encodings": {k: v for k, v in encoders.items() if k != A.ENC_ASTC or astc_ok},
        "encoding_notes": ({} if astc_ok else {A.ENC_ASTC: astc_why}),
        "chunks": chunks,
        "keyframes": keyframes,
        "excluded": sorted(excluded, key=lambda e: e["ki"]),
        "selection": selection,
        "exposure": exposure,
        "occluders": {"frames_with_occluders": int(sum(1 for p in occ_px if p)),
                      "total_pixels": int(sum(occ_px)),
                      "mean_fraction": round(float(np.mean([fr.occluder.mean()
                                                            for fr in frames])), 5),
                      "near_mean_fraction": round(float(np.mean(near_frac)), 5),
                      "transient_mean_fraction": round(float(np.mean(tr_frac)), 5),
                      "frames_with_near": int(sum(1 for f in near_frac if f > 0)),
                      "frames_with_transient": int(sum(1 for f in tr_frac if f > 0)),
                      "detector_mean_fraction": round(float(np.mean(det_frac)), 5),
                      "frames_with_detector_mask": int(sum(1 for fr in frames
                                                           if fr.detector is not None)),
                      "frames_with_detector_pixels": int(sum(1 for f in det_frac if f > 0)),
                      "misregistered_frames_unmasked": inconsistent,
                      "rule": (f"near:src<clamp(1-{params.occluder_mad_multiple}mad,"
                               f"{params.occluder_ratio_min},{params.occluder_ratio_max})"
                               f"*median*proxy|open{params.occluder_open_px}|"
                               f"minarea{params.occluder_min_area_frac}|"
                               f"dilate{params.occluder_dilate_px};"
                               f"transient:grid{params.transient_grid_px}|box{params.transient_box_px}|"
                               f"views>={params.transient_min_views}|consensus{params.transient_consensus}|"
                               f"diff>max({params.transient_abs},{params.transient_rel}*luma)|"
                               f"shift{params.transient_shift_px}|framemax{params.transient_frame_max}")},
        # THE MISSING-MASK POLICY (contract §5.3a): `state` is `ok` only when
        # the detector ran or its cache answered; otherwise `unavailable`,
        # `failed` or `off`, with the reason, and no keyframe says it is masked.
        "transients": treport.record(),
        "seconds": seconds,
        "scale": _scale_note(world),
    }
    _mark_superseded(root, set(named_files(manifest)))
    write_json_atomic(root / "manifest.json", manifest)
    _prune(root, set(named_files(manifest)))
    total_bytes = sum(c["bytes"] for c in chunks) + len(proxy_bytes)
    result = AppearanceResult(state=STATE_OK, build_id=build_id, keyframes=len(frames),
                              phone=len(order), excluded=len(excluded), chunks=len(chunks),
                              bytes=total_bytes, seconds=seconds)
    _status(root, state=STATE_OK, params_digest=pdigest, result=result.as_dict())
    logger.info("[Tower][WorldBuilder][appearance] %s/%s built: %d keyframes (%d phone), "
                "%d chunks, %.1f MB, %s", world_id, session_id, len(frames), len(order),
                len(chunks), total_bytes / 1e6, seconds)
    return result


def _scale_note(world) -> dict:
    scale = getattr(world, "scale", None)
    scale = scale.to_json_dict() if hasattr(scale, "to_json_dict") else dict(
        scale or {"state": "unknown", "meters_per_unit": None})
    scale["note"] = "inherited unchanged from the solve; appearance makes no scale claim"
    return scale


def _write_once(path: Path, data: bytes) -> None:
    """Content-addressed: a file already there with this name and size is this
    content (and may be named by the live manifest, so it is not rewritten)."""
    try:
        if path.stat().st_size == len(data):
            os.utime(path, None)
            return
    except OSError:
        pass
    write_bytes_atomic(path, lambda handle, d=data: handle.write(d))


def _stop(root: Path, stage: str, seconds: dict, discard=None) -> AppearanceResult:
    if discard is not None:
        _discard_unpublished(*discard)
    _status(root, state=STATE_STOPPED, stage=stage)
    return AppearanceResult(state=STATE_STOPPED, detail=f"stopped during {stage}",
                            seconds=seconds)


def _current_manifest(root: Path):
    try:
        return json.loads((root / "manifest.json").read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None


def _discard_unpublished(root: Path, names) -> None:
    """Remove files this build wrote that the manifest on disk does not name.
    Content addressing means a new build can write a file the current manifest
    already names; that one stays."""
    current = _current_manifest(root)
    if current is None:
        return
    keep = set(named_files(current))
    for name in set(names) - keep:
        try:
            (root / name).unlink()
        except OSError:
            pass


def _already_built(root: Path, pdigest: str, force: bool):
    if force:
        return None
    man = read_manifest_at(root)
    if man is None or man.get("params_digest") != pdigest:
        return None
    for name, size in named_files(man).items():
        try:
            if (root / name).stat().st_size != size:
                return None
        except OSError:
            return None
    kfs = man.get("keyframes") or []
    return AppearanceResult(
        state=STATE_OK, detail=ALREADY_BUILT, build_id=man.get("build_id"),
        keyframes=len(kfs), phone=sum(1 for k in kfs if k.get("tier") == A.TIER_PHONE),
        excluded=len(man.get("excluded") or []), chunks=len(man.get("chunks") or []),
        bytes=sum(named_files(man).values()), seconds=man.get("seconds") or {})


def _mark_superseded(root: Path, keep: set) -> None:
    previous = _current_manifest(root)
    if not previous:
        return
    now = time.time()
    for name in set(named_files(previous)) - keep:
        try:
            os.utime(root / name, (now, now))
        except OSError:
            pass


def _prune(root: Path, keep: set, older_than_s: float = PRUNE_GRACE_S) -> None:
    now = time.time()
    for pattern in ("c.*.bin", "p.*.bin"):
        for path in root.glob(pattern):
            if path.name in keep:
                continue
            try:
                if now - path.stat().st_mtime >= older_than_s:
                    path.unlink()
            except OSError:
                pass
    for pattern in ("c.*.bin.p*.tmp", "p.*.bin.p*.tmp"):
        for path in root.glob(pattern):
            try:
                if now - path.stat().st_mtime >= STAGING_GRACE_S:
                    path.unlink()
            except OSError:
                pass


def _sweep_unnamed(root: Path) -> None:
    current = _current_manifest(root)
    if current is None:
        return
    _prune(root, set(named_files(current)))


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def read_manifest_at(root: Path) -> dict | None:
    from tower.world_builder.store import _read_json_past_a_replace  # noqa: PLC0415

    path = root / "manifest.json"
    if not path.exists():
        return None
    try:
        man = _read_json_past_a_replace(path)
    except (OSError, ValueError):
        return None
    if not isinstance(man, dict) or man.get("format") != A.APPEARANCE_FORMAT:
        return None
    if man.get("schema_version") != A.APPEARANCE_SCHEMA_VERSION:
        return None
    return man


def read_appearance_manifest(store, world_id: str, session_id: str) -> dict | None:
    """The manifest, or None. Never raises on a malformed one."""
    return read_manifest_at(appearance_dir(store, world_id, session_id))


def read_appearance_file(store, world_id: str, session_id: str, kind: str, digest: str,
                         manifest: dict) -> bytes | None:
    """The bytes of a file the manifest names, checked against its recorded size
    and its content digest; None for anything else."""
    if kind not in ("chunk", "proxy") or not is_digest(digest):
        return None
    name = _file_name(kind, digest)
    size = named_files(manifest).get(name)
    if size is None:
        return None
    try:
        data = (appearance_dir(store, world_id, session_id) / name).read_bytes()
    except OSError:
        return None
    if len(data) != size or content_digest(data) != digest:
        return None
    return data


def label_matches(store, world_id: str, session_id: str, manifest: dict) -> bool:
    prov = manifest.get("appearance_provenance") or {}
    label, image_set = A.keyframe_set_identity(store, world_id, session_id)
    return (prov.get("session_redaction") == label
            and prov.get("keyframe_image_set") == image_set)


def appearance_currency(store, world_id: str, session_id: str, manifest: dict | None) -> dict:
    """Contract §8: reported, not enforced (the label is enforced elsewhere)."""
    from tower.world_builder.global_solve import load_solution  # noqa: PLC0415
    from tower.world_builder.surface_pipeline import read_surface_manifest  # noqa: PLC0415

    if manifest is None:
        return {"present": False, "current": False, "reason": "no appearance artifact"}
    try:
        solution = load_solution(store, world_id, session_id)
    except Exception:  # noqa: BLE001
        solution = None
    live = getattr(solution, "input_digest", None)
    if live is None or live != manifest.get("input_digest"):
        return {"present": True, "current": False, "reason": "built from an earlier solve"}
    surface = read_surface_manifest(store, world_id, session_id)
    src = (manifest.get("proxy") or {}).get("source") or {}
    if surface is None or surface.get("built_at") != src.get("surface_built_at"):
        return {"present": True, "current": False, "reason": "built on an earlier surface"}
    return {"present": True, "current": True, "reason": None}
