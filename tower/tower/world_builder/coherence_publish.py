"""Depth before publish, the evidence gate on the final solve, and the components record.

WHAT. With `TOWER_WORLD_SOLVE_GATE` on (`config.world_solve_gate_setting`; OFF by default, and off means
today's solve byte for byte), the FINAL solve of a session is not published as the solver returned it.
Between the mapper and `write_solution` (`global_solve.solve`):

  1. DEPTH BEFORE PUBLISH. The product depth stage (`dense_pipeline.run_depth_stage`, MoGe-2 ViT-L, told
     the solve camera's FoV: `DenseParams.known_fov`) runs on the CANDIDATE solve, over every posed keyframe
     of every solver component. Its predictions are what the surface stage would have made anyway; the
     live walk's predictions are reused where they were made the same way.
  2. METRIC SCALE. `coherence_scale.metric_scale`: per camera log(z_sfm / z_metric) from the solve
     database's verified inlier matches, the candidate's poses and those predictions.
  3. THE GATE. `coherence_gate.apply_gate` with the solve database's verified links and their two-view
     rotations, the metric scale, and `masks_applied` -- true only when the solve's transient masks were
     `applied` (`transients.state`; `partial` and `unavailable` are the fail-safe, manager 011).
  4. RELABEL. Every posed keyframe's `component` becomes its gate label (0 = the room), each 3-D point
     goes with the majority label of its observers, and `components` is rebuilt per label. Coordinates
     are untouched: an unplaced piece keeps the solver's frame, which is simply no longer claimed to be the
     room's. `merge` and every stage after it then publish the pieces as separate components -- nothing is
     merged into the room that the gate did not attach.

After `write_solution`:

  5. `solve/<session>/components.json` (contract `WORLD-BUILDER-COMPONENTS.md` §2, the record shape
     `components.read_components_record` reads -- agent PA's reader), written ONCE per published solve.
  6. THE SURFACE REUSES THE DEPTH. `dense/<session>/align.json` is stamped with the published solve's
     `input_digest` and cache key and cut to the room's keyframes, so the final surface
     (`surfacify` with `SurfaceParams(depth_known_fov=True)`) finds it current and predicts nothing. The
     depth stage wrote it UNSTAMPED before publish, so a solve killed in between leaves a record no stage
     trusts.

FAIL-SAFES (contract §2.2): masks not `applied` -> nothing attached, reason `masks-unavailable`; depth
unavailable, stopped or no camera with a ratio -> nothing attached, reason `scale-unavailable`. Both still
write components (the room = the anchor block; every other piece unplaced). A failure of the gate ITSELF
(an exception) publishes the solve exactly as the solver returned it, with `gate.state: "failed"` and no
components record -- `components: null`, "not computed", which is what every older world says.

THRESHOLDS: none new. The gate's are `GateParams` (evidence cited there); the scale's are the harness's
(`ScaleParams`); the area floor (30 keyframes or 5 s), the 2.0 s span join and the 8-span cap are the
contract's proposals (§2.3, §2.4 rule 5; OPEN T2, T4).
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_scale as CS

logger = logging.getLogger(__name__)

COMPONENTS_FILENAME = "components.json"
# What a solution published WITHOUT the gate does to the record of the solution it replaces: moves it
# aside (never deletes it), so no reader pairs an old record with a new solve.
SUPERSEDED_FILENAME = "components.superseded.json"
CONTRACT = "world_builder.components/2026-09-23"
RECORD = "wb-components-record/1"

GATE_STATE_APPLIED = "applied"
GATE_STATE_FAILED = "failed"

DEPTH_OK = "ok"
DEPTH_UNAVAILABLE = "unavailable"
DEPTH_STOPPED = "stopped"

# Contract §2.3: an unplaced component is an AREA with at least this many keyframes or this much total
# capture span; below both it is counted only (`none`). Proposed from one case (OPEN T2).
AREA_MIN_KEYFRAMES = 30
AREA_MIN_SPAN_S = 5.0
# Contract §2.4 rule 5 (OPEN T4): consecutive member keyframes at most this far apart are one span; more
# than MAX_SPANS spans are merged across the shortest gaps.
SPAN_JOIN_S = 2.0
MAX_SPANS = 8


class DepthUnavailable(RuntimeError):
    """The depth stage could not give the gate a metric depth (with the reason)."""


def gate_setting_for(final: bool, gate: bool | None = None) -> bool:
    """Whether this solve runs the gate. An explicit argument wins; otherwise only a FINAL solve, and only
    with `TOWER_WORLD_SOLVE_GATE` on. The walk's background solves never do."""
    if gate is not None:
        return bool(gate)
    if not final:
        return False
    from tower.config import world_solve_gate_setting  # noqa: PLC0415

    return bool(world_solve_gate_setting())


@dataclasses.dataclass
class GateResult:
    solution: object                 # the relabelled Solution (the input, unchanged, when the gate failed)
    record: dict                     # `solution.gate`: what ran (contract §2.5)
    components: dict | None          # the components.json document; None = not computed
    depth: dict | None = None        # {"align", "work", "dparams"} for the surface hand-off, when depth ran


# ---------------------------------------------------------------------------
# the solve, as the gate and the scale see it


def _image_names(keyframes) -> dict[str, str]:
    from tower.world_builder.global_solve import keyframe_image_name  # noqa: PLC0415

    return {k.keyframe_id: keyframe_image_name(k) for k in keyframes}


def solve_model(solution, name_of: dict[str, str]) -> "CG.SolveModel":
    """`coherence_gate.SolveModel` of a `global_solve.Solution`: its posed keyframes, named by their solve
    database image names (which sort in capture order), their observations re-indexed to cameras."""
    kids = [kid for kid in solution.keyframe_ids if kid in (solution.poses or {})]
    cam_of_kf = np.full(len(solution.keyframe_ids), -1, np.int64)
    pos = {kid: i for i, kid in enumerate(solution.keyframe_ids)}
    for c, kid in enumerate(kids):
        cam_of_kf[pos[kid]] = c
    obs = np.asarray(solution.observations, np.int64).reshape(-1, 3)
    cam = cam_of_kf[obs[:, 0]] if len(obs) else np.zeros(0, np.int64)
    keep = cam >= 0
    return CG.SolveModel(
        names=[name_of[kid] for kid in kids],
        component=np.asarray([int(solution.poses[k].get("component", 0)) for k in kids], np.int64),
        R_cw=np.asarray([np.asarray(solution.poses[k]["rotation"], np.float64).reshape(3, 3) for k in kids],
                        np.float64).reshape(-1, 3, 3),
        n_obs=np.asarray([int(solution.poses[k].get("observations", 0)) for k in kids], np.int64),
        obs_image=cam[keep],
        obs_point=obs[keep, 2] if len(obs) else np.zeros(0, np.int64),
        n_points=int(len(solution.xyz)),
    )


# ---------------------------------------------------------------------------
# depth before publish


def _all_posed_in_component_zero(solution):
    """The candidate with every posed keyframe in component 0, for the depth stage: it predicts and fits
    ONE component, and the gate needs a metric level in every one. Poses and points are untouched; each
    frame is still fitted against its own observations."""
    return dataclasses.replace(
        solution, poses={kid: {**p, "component": 0} for kid, p in (solution.poses or {}).items()})


def _depth_params():
    """The depth stage's parameters for the gate, equal to the final surface's (`SurfaceParams`, with the
    FoV known) in every field `ensure_depth_stage` builds, so the stamped record is that stage's cache."""
    from tower.world_builder.dense import DenseParams  # noqa: PLC0415
    from tower.world_builder.surface import SurfaceParams  # noqa: PLC0415

    sp = SurfaceParams(depth_known_fov=True)
    return DenseParams(gate_rel=sp.gate_rel, imagery_source=sp.imagery_source, known_fov=True)


def run_gate_depth(store, world_id: str, session_id: str, solution, intrinsics, *,
                   should_stop=None) -> tuple[dict, Path, object]:
    """The product depth stage on the candidate solve: `(align, work_dir, dparams)`.

    Under the session's surface lock (`surface_pipeline._SurfaceLock`), the one every writer of
    `dense/<session>/` holds, so it never interleaves with a surface build of the session. The record it
    writes is UNSTAMPED (no `input_digest`, no `cache_key`): until `hand_depth_to_surface` stamps it after
    publish, no stage reads it as a cache. Raises `DepthUnavailable` with the reason."""
    from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
        dense_dir,
        reusable_predictions,
        run_depth_stage,
    )
    from tower.world_builder.surface_pipeline import _SurfaceLock, surface_dir  # noqa: PLC0415

    if intrinsics is None or getattr(intrinsics, "fx", None) is None:
        raise DepthUnavailable("session has no intrinsics")
    if not solution.camera:
        raise DepthUnavailable("the solve has no camera")
    dparams = _depth_params()
    root = dense_dir(store, world_id, session_id)
    root.mkdir(parents=True, exist_ok=True)
    lock = _SurfaceLock(surface_dir(store, world_id, session_id))
    if not lock.acquire():
        raise DepthUnavailable("another surface build of this session holds its lock")
    try:
        reuse = reusable_predictions(root / "align.json", dparams.backend, dparams.imagery_source,
                                     known_fov=True)
        align = run_depth_stage(store, world_id, session_id, _all_posed_in_component_zero(solution),
                                intrinsics, dparams, root, should_stop=should_stop, prior=None,
                                reuse_predictions=reuse)
    except DepthUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 -- model missing, CUDA OOM, a camera mismatch: all "no depth"
        raise DepthUnavailable(f"{type(exc).__name__}: {exc}") from exc
    finally:
        lock.release()
    if align.get("stopped_after") is not None:
        raise DepthUnavailable(f"{DEPTH_STOPPED}: the depth stage was stopped after "
                               f"{align.get('stopped_after')} frames")
    return align, root / "work", dparams


def measure_metric_scale(solution, name_of: dict[str, str], database_path, work) -> dict:
    """`coherence_scale.metric_scale` over the depth stage's predictions in `work` (keyframe index = the
    position in `solution.keyframe_ids`, the depth stage's own `ki`)."""
    cameras = CS.cameras_from_solution(solution, name_of.__getitem__)
    ki_of_name = {name_of[kid]: i for i, kid in enumerate(solution.keyframe_ids) if kid in name_of}
    sampler = CS.DepthStageSampler(work, ki_of_name)
    pairs = CS.read_inlier_pairs(database_path)
    out = CS.metric_scale(cameras, pairs, solution.camera, sampler)
    out["frames_without_prediction"] = len(sampler.missing)
    return out


def hand_depth_to_surface(store, world_id: str, session_id: str, solution, depth: dict) -> dict:
    """Stamp the gate's depth record as the published solve's depth stage, cut to the room.

    `ensure_depth_stage` (the surface) then finds `input_digest`, backend, fill rule, keyframe set,
    imagery, trust and FoV all current, and reuses it -- predicting nothing. Records of keyframes outside
    the room are dropped from the record (their predictions stay on disk): the surface builds component 0,
    and `_Frames` fuses every record it is given."""
    from tower.storage import write_json_atomic  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import _depth_cache_key, dense_dir  # noqa: PLC0415
    from tower.world_builder.surface_pipeline import _SurfaceLock, surface_dir  # noqa: PLC0415

    align, dparams = depth["align"], depth["dparams"]
    room = {kid for kid, p in (solution.poses or {}).items() if int(p.get("component", 0)) == 0}
    kids = solution.keyframe_ids

    def kid_of(rec):
        kid = rec.get("kid")
        if kid is None and rec.get("ki") is not None and 0 <= int(rec["ki"]) < len(kids):
            kid = kids[int(rec["ki"])]
        return kid

    records = [r for r in align.get("records") or [] if kid_of(r) in room]
    stamped = dict(align, records=records)
    stamped["input_digest"] = solution.input_digest
    stamped["digest"] = solution.input_digest
    stamped["cache_key"] = _depth_cache_key(solution.input_digest, dparams, align.get("keyframe_image_set"),
                                            align.get("redaction_trust"))
    stamped["gate_handoff"] = {"room_keyframes": len(room), "records_kept": len(records),
                               "records_outside_room": len(align.get("records") or []) - len(records)}
    lock = _SurfaceLock(surface_dir(store, world_id, session_id))
    if not lock.acquire():
        return {"stamped": False, "detail": "another surface build of this session holds its lock"}
    try:
        write_json_atomic(dense_dir(store, world_id, session_id) / "align.json", stamped)
    finally:
        lock.release()
    return {"stamped": True, **stamped["gate_handoff"]}


# ---------------------------------------------------------------------------
# relabel


def _point_labels(solution, kf_label: np.ndarray) -> np.ndarray:
    """Each point's label: the majority label of its observers (ties: the lowest label); a point no
    labelled keyframe observes goes with its first observer's label, else 0."""
    n_pts = int(len(solution.xyz))
    out = np.full(n_pts, -1, np.int64)
    obs = np.asarray(solution.observations, np.int64).reshape(-1, 3)
    if n_pts and len(obs):
        lab = kf_label[obs[:, 0]]
        ok = lab >= 0
        pt, lab = obs[ok, 2], lab[ok]
        if len(pt):
            L = int(lab.max()) + 1
            uniq, cnt = np.unique(pt * L + lab, return_counts=True)
            p, lb = uniq // L, uniq % L
            order = np.lexsort((lb, -cnt, p))
            first = np.r_[True, p[order][1:] != p[order][:-1]]
            out[p[order][first]] = lb[order][first]
    missing = out < 0
    if missing.any():
        fk = np.asarray(solution.first_keyframe, np.int64).reshape(-1)
        fb = np.zeros(n_pts, np.int64)
        if len(kf_label) and len(fk) == n_pts:
            ok = (fk >= 0) & (fk < len(kf_label))
            fb[ok] = np.maximum(kf_label[fk[ok]], 0)
        out[missing] = fb[missing]
    return out


def relabel_solution(solution, labels_by_kid: dict[str, int], gate_components: list[dict],
                     min_obs: int = 30):
    """The solution with the gate's labels as its components (module docstring, step 4)."""
    poses = {kid: dict(p) for kid, p in (solution.poses or {}).items()}
    for kid, lab in labels_by_kid.items():
        if kid in poses:
            poses[kid]["component"] = int(lab)
    kf_label = np.full(len(solution.keyframe_ids), -1, np.int64)
    for i, kid in enumerate(solution.keyframe_ids):
        if kid in labels_by_kid:
            kf_label[i] = int(labels_by_kid[kid])
    point_label = _point_labels(solution, kf_label)
    err = np.asarray(solution.error, np.float64)
    by_label = {int(c["label"]): c for c in gate_components}
    components = []
    for lab in sorted(set(int(v) for v in labels_by_kid.values())):
        members = [kid for kid, v in labels_by_kid.items() if int(v) == lab]
        sel = point_label == lab
        g = by_label.get(lab, {})
        components.append({
            "index": lab,
            "images": len(members),
            "images_supported": sum(1 for k in members if int(poses[k].get("observations", 0)) >= min_obs),
            "points": int(sel.sum()),
            "mean_error_px": float(err[sel].mean()) if sel.any() and len(err) == len(sel) else None,
            "gate_state": g.get("state"),
            "gate_reasons": list(g.get("reasons") or []),
            "solver_components": list(g.get("source_components") or []),
        })
    return dataclasses.replace(solution, poses=poses, component=point_label.astype(np.int32),
                               components=components)


# ---------------------------------------------------------------------------
# the components record (contract §2; the shape `components.parse_components` reads)


def component_id(session_id: str, keyframe_ids) -> str:
    """Contract §2.4 rule 3: the first 16 hex of sha256 over the session id and the sorted member
    keyframe ids. Stable while membership is; a new final solve may change it."""
    h = hashlib.sha256(session_id.encode("utf-8"))
    for kid in sorted(keyframe_ids):
        h.update(b"\n")
        h.update(kid.encode("utf-8"))
    return h.hexdigest()[:16]


def capture_spans(times_s, join_s: float = SPAN_JOIN_S, max_spans: int = MAX_SPANS
                  ) -> tuple[list[list[float]], float]:
    """(spans, total): member keyframes' times (seconds since the session's start) joined into spans
    where consecutive times are at most `join_s` apart, then merged across the shortest gaps until at most
    `max_spans` remain (§2.4 rule 5). `total` is the summed length of the JOINED spans before that merge
    -- the capture time the component actually covers, which is what the area floor reads."""
    t = sorted(float(v) for v in times_s if v is not None and np.isfinite(v))
    spans: list[list[float]] = []
    for v in t:
        if spans and v - spans[-1][1] <= join_s:
            spans[-1][1] = v
        else:
            spans.append([v, v])
    total = float(sum(b - a for a, b in spans))
    while len(spans) > max_spans:
        gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        i = int(np.argmin(gaps))
        spans[i:i + 2] = [[spans[i][0], spans[i + 1][1]]]
    return [[round(a, 1), round(b, 1)] for a, b in spans], total


CG_PLACED = "placed"


def shown_as(state: str, keyframes: int, total_span_s: float) -> str:
    """Contract §2.3: the placed component is the room; an unplaced one is an area at >= 30 keyframes or
    >= 5 s of capture, else counted only."""
    if state == CG_PLACED:
        return "room"
    if keyframes >= AREA_MIN_KEYFRAMES or total_span_s >= AREA_MIN_SPAN_S:
        return "area"
    return "none"


def components_document(session_id: str, solution, labels_by_kid: dict[str, int], gate: dict,
                        keyframes, started_at: float, *, min_obs: int = 30) -> dict | None:
    """The components record, or None when it cannot hold the contract's shape (no published keyframe in
    the room: then nothing was computed that the phone could show, which is `components: null`).

    Per gate label: its PUBLISHED keyframes (>= `min_obs` observations; contract §2.4 rule 4 -- a label
    with none is not reported), their capture spans on the Tower's receipt clock, `shown_as`, and the
    Tower-internal `keyframe_ids` an area build is made from."""
    received = {k.keyframe_id: float(k.received_at) for k in keyframes}
    order = {kid: i for i, kid in enumerate(solution.keyframe_ids)}
    entries = []
    for comp in gate["components"]:
        lab = int(comp["label"])
        kids = [kid for kid, v in labels_by_kid.items()
                if int(v) == lab and int(solution.poses[kid].get("observations", 0)) >= min_obs]
        if not kids:
            continue
        kids.sort(key=lambda k: order.get(k, 0))
        # Floored at 0: the first keyframe's receipt can precede the session's recorded `started_at` by a
        # fraction of a second (-0.1 s on the target walk), and "before the walk began" is not a time.
        spans, total = capture_spans([max(0.0, received[k] - float(started_at)) for k in kids if k in received])
        state = comp["state"]
        reasons = [] if state == CG_PLACED else list(comp.get("reasons") or [])
        entries.append({
            "id": component_id(session_id, kids),
            "state": state,
            "reason": reasons[0] if reasons else None,
            "reasons": reasons,
            "shown_as": shown_as(state, len(kids), total),
            "keyframes": len(kids),
            "capture_spans_s": spans,
            "keyframe_ids": kids,
        })
    if sum(1 for e in entries if e["state"] == CG_PLACED) != 1:
        return None
    rank = {"room": 0, "area": 1, "none": 2}
    entries.sort(key=lambda e: (rank[e["shown_as"]], -e["keyframes"],
                                e["capture_spans_s"][0][0] if e["capture_spans_s"] else float("inf"), e["id"]))
    return {
        "record": RECORD,
        "contract": CONTRACT,
        "session_id": session_id,
        # Which published solve this record describes: a reader can refuse a record whose solve is gone.
        "input_digest": solution.input_digest,
        "solved_at": solution.solved_at,
        "gate": {"id": gate["gate"], "params": gate["params"], "params_digest": gate["params_digest"],
                 "masks_applied": gate["masks_applied"], "metric_available": gate["metric_available"]},
        "components": entries,
    }


def components_path(workspace_root) -> Path:
    return Path(workspace_root) / COMPONENTS_FILENAME


def write_components(workspace_root, document: dict) -> Path:
    from tower.storage import write_json_atomic  # noqa: PLC0415

    path = components_path(workspace_root)
    write_json_atomic(path, document)
    return path


def retire_components(workspace_root) -> bool:
    """A solution published without a components record moves the previous one aside (never deletes
    it): an old record must not describe a new solve. True when one was moved."""
    path = components_path(workspace_root)
    if not path.exists():
        return False
    try:
        os.replace(path, Path(workspace_root) / SUPERSEDED_FILENAME)
        return True
    except OSError:
        logger.warning("[Tower][WorldBuilder] could not retire %s; its solve was replaced", path,
                       exc_info=True)
        return False


# ---------------------------------------------------------------------------
# the whole step


def gate_final_solution(store, world_id: str, session_id: str, solution, *, database_path, keyframes,
                        should_stop=None, params: "CG.GateParams | None" = None,
                        depth_runner: Callable | None = None, metric_fn: Callable | None = None) -> GateResult:
    """Steps 1-4 of the module docstring on the candidate; never raises. `depth_runner` and `metric_fn`
    replace `run_gate_depth` / `measure_metric_scale` (tests)."""
    params = params or CG.GateParams()
    started = time.perf_counter()
    try:
        return _gate(store, world_id, session_id, solution, database_path=database_path, keyframes=keyframes,
                     should_stop=should_stop, params=params, depth_runner=depth_runner or run_gate_depth,
                     metric_fn=metric_fn or measure_metric_scale, started=started)
    except Exception as exc:  # noqa: BLE001 -- a broken gate publishes today's solve and says so
        logger.exception("[Tower][WorldBuilder] the evidence gate failed on %s/%s; the final solve is "
                         "published as the solver returned it, with no components record",
                         world_id, session_id)
        return GateResult(solution=solution, components=None, record={
            "state": GATE_STATE_FAILED, "detail": f"{type(exc).__name__}: {exc}", "gate": CG.GATE_ID,
            "params": params.to_json(), "params_digest": params.digest(),
            "seconds": round(time.perf_counter() - started, 3)})


def _gate(store, world_id, session_id, solution, *, database_path, keyframes, should_stop, params,
          depth_runner, metric_fn, started) -> GateResult:
    transients = solution.transients or {}
    masks_state = transients.get("state")
    masks_applied = masks_state == "applied"
    name_of = _image_names(keyframes)
    session = store.read_session(world_id, session_id)

    # 1. depth before publish
    t = time.perf_counter()
    depth = None
    depth_record: dict = {"state": DEPTH_OK}
    try:
        align, work, dparams = depth_runner(store, world_id, session_id, solution, session.intrinsics,
                                            should_stop=should_stop)
        depth = {"align": align, "work": work, "dparams": dparams}
        depth_record.update({"backend": align.get("backend"), "known_fov": align.get("known_fov"),
                             "frames": align.get("targets"), "image_origins": align.get("image_origins")})
    except DepthUnavailable as exc:
        detail = str(exc)
        depth_record = {"state": DEPTH_STOPPED if detail.startswith(DEPTH_STOPPED) else DEPTH_UNAVAILABLE,
                        "detail": detail}
        logger.warning("[Tower][WorldBuilder] gate %s/%s: no metric depth (%s); the gate attaches nothing "
                       "(scale-unavailable)", world_id, session_id, detail)
    depth_record["seconds"] = round(time.perf_counter() - t, 3)

    # 2. metric scale
    t = time.perf_counter()
    metric_log: dict = {}
    scale_record: dict = {"scale": CS.SCALE_ID, "params": CS.ScaleParams().to_json(),
                          "params_digest": CS.ScaleParams().digest()}
    if depth is not None:
        scale = metric_fn(solution, name_of, database_path, depth["work"])
        metric_log = scale["metric_log"]
        scale_record.update({k: scale.get(k) for k in ("cameras_published", "cameras_measured", "pairs_used",
                                                        "inliers_total", "inliers_gated",
                                                        "frames_without_prediction")})
    scale_record["state"] = "measured" if metric_log else "unavailable"
    scale_record["seconds"] = round(time.perf_counter() - t, 3)

    # 3. the gate
    t = time.perf_counter()
    links = CG.read_verified_links(database_path, min_inliers=params.min_link_inliers)
    rotations = CG.read_link_rotations(database_path, solution.camera, min_inliers=params.min_link_inliers)
    model = solve_model(solution, name_of)
    result = CG.apply_gate(model, links, metric_log, link_rotations=rotations, masks_applied=masks_applied,
                           params=params)
    gate_seconds = round(time.perf_counter() - t, 3)

    # 4. relabel
    kid_of_name = {v: k for k, v in name_of.items()}
    labels_by_kid = {kid_of_name[n]: int(lab) for n, lab in result["labels"].items()}
    relabelled = relabel_solution(solution, labels_by_kid, result["components"], min_obs=params.min_obs)
    doc = components_document(session_id, relabelled, labels_by_kid, result, keyframes, session.started_at,
                              min_obs=params.min_obs)
    counts = {"placed": 0, "area": 0, "none": 0}
    for e in (doc or {}).get("components", []):
        counts["placed" if e["state"] == CG_PLACED else e["shown_as"]] += 1
    record = {
        "state": GATE_STATE_APPLIED,
        "gate": CG.GATE_ID,
        "params": result["params"],
        "params_digest": result["params_digest"],
        "masks_applied": bool(masks_applied),
        "transients_state": masks_state,
        "metric_available": bool(result["metric_available"]),
        "attach": bool(result["evidence"].get("attach")),
        "evidence": result["evidence"],
        "depth": depth_record,
        "metric_scale": scale_record,
        "labels": len(result["components"]),
        "components": counts if doc is not None else None,
        "components_file": COMPONENTS_FILENAME if doc is not None else None,
        "gate_seconds": gate_seconds,
        "seconds": round(time.perf_counter() - started, 3),
    }
    return GateResult(solution=relabelled, record=record, components=doc, depth=depth)


def after_publish(store, world_id: str, session_id: str, workspace_root, solution,
                  result: GateResult | None) -> dict:
    """Steps 5-6, after `write_solution` published `solution`. Never raises. Without a gate result (the
    gate off) a previous record is moved aside; with one, the record is written once and the depth is
    handed to the surface."""
    out: dict = {}
    try:
        if result is None or result.components is None:
            out["components_retired"] = retire_components(workspace_root)
            if result is None:
                return out
        else:
            write_components(workspace_root, result.components)
            out["components_written"] = True
        if result.depth is not None:
            out["depth_handoff"] = hand_depth_to_surface(store, world_id, session_id, solution, result.depth)
    except Exception as exc:  # noqa: BLE001 -- the solve is published; this is its paperwork
        logger.exception("[Tower][WorldBuilder] after publishing %s/%s: %s", world_id, session_id, exc)
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def gate_and_publish(store, world_id: str, session_id: str, workspace, solution, *, final: bool,
                     gate: bool | None = None, database_path, keyframes, write: Callable,
                     should_stop=None) -> tuple[object, dict | None]:
    """The final solve's publish step, in one call (`global_solve.solve` makes it in place of
    `write_solution`): the gate when `gate_setting_for(final, gate)`, then `write(workspace, solution)`,
    then the record and the depth hand-off. Returns (the published solution, its gate record or None).

    With the gate off this is `write(workspace, solution)` and nothing else, except that a components record
    left by an earlier gated solve is moved aside: it does not describe this solve."""
    result = None
    if gate_setting_for(final, gate):
        result = gate_final_solution(store, world_id, session_id, solution, database_path=database_path,
                                     keyframes=keyframes, should_stop=should_stop)
        solution = result.solution
        if hasattr(solution, "gate"):
            solution.gate = result.record
        solution.timing = dict(getattr(solution, "timing", None) or {}, gate_s=result.record.get("seconds"))
    write(workspace, solution)
    published = after_publish(store, world_id, session_id, workspace.root, solution, result)
    if result is None:
        return solution, None
    return solution, dict(result.record, publish=published)
