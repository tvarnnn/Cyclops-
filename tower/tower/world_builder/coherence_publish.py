"""Depth before publish, the evidence gate on the final solve, and the components record.

WHAT. With `TOWER_WORLD_SOLVE_GATE` on (`config.world_solve_gate_setting`; OFF by default, and off means
today's solve byte for byte), the FINAL solve of a session is not published as the solver returned it.
Between the mapper and `write_solution` (`global_solve.solve`):

  1. DEPTH BEFORE PUBLISH. The product depth stage (`dense_pipeline.run_depth_stage`, MoGe-2 ViT-L, told
     the solve camera's FoV: `DenseParams.known_fov`) runs on the CANDIDATE solve, over every posed keyframe
     of every solver component. Its predictions are what the surface stage would have made anyway, and
     they are KEPT (`dense_pipeline.PREDICTIONS_DIRNAME`, keyed by the exact pixels and the network): a
     re-finish, a re-gate or a consensus draw re-fits the kept prediction instead of predicting again.
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
write components (the room = the anchor block; every other piece unplaced), and neither is silent: the row's
finalization detail says what is missing and who can fix it (`publish_notice`, review V8 M2). Only a depth
stage that did not finish owes a re-gate in place (`retryable`); a shortfall with depth in hand, and the
masks fail-safe, do not (a re-gate would reproduce them). A failure of the gate ITSELF
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
# `gate.cause` of a gate record that owes a re-gate (`gate.retryable`).
CAUSE_DEPTH_UNAVAILABLE = "depth-unavailable"
CAUSE_GATE_FAILED = "gate-failed"

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
    # In memory only, for the consensus: the metric scale the gate used (`measure_metric_scale`'s output,
    # None without depth), the gate's own output (`coherence_gate.apply_gate`: rounds, groups), the
    # candidate it gated, and the per-draw detail `after_publish` persists (`CONSENSUS_FILENAME`).
    scale: dict | None = None
    gated: dict | None = None
    candidate: object = None
    consensus_detail: dict | None = None


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
    publish, no stage reads it as a cache. Raises `DepthUnavailable` with the reason.

    THE PREDICTIONS COME FROM THE PREDICTION CACHE (`dense_pipeline.PREDICTIONS_DIRNAME`; review V8 H2,
    R3), keyed by the exact pixels the network is shown: a re-finish, a re-gate and every draw of a
    consensus read the prediction the first gated build made, and only the fit to THIS solve is redone.
    The older reuse path (`reusable_predictions`, offered from a previous `align.json`) is not taken
    here: a prediction it offers may predate the cache, and then two builds of one walk could disagree
    on it."""
    from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
        dense_dir,
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
        align = run_depth_stage(store, world_id, session_id, _all_posed_in_component_zero(solution),
                                intrinsics, dparams, root, should_stop=should_stop, prior=None,
                                prediction_cache=True)
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
    # WHICH SOLVE (review V7, M3): `input_digest` names the keyframes only, and two solves
    # of the same keyframes are two solves.
    from tower.world_builder.global_solve import solve_identity  # noqa: PLC0415

    stamped["solve_identity"] = solve_identity(solution)
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
            # The solver's own component, kept beside the gate's label: a re-gate starts
            # again from the solver's partition (`solver_candidate`).
            poses[kid].setdefault("solver_component", int(poses[kid].get("component", 0)))
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
        # Which published solve this record describes: the reader refuses a record whose solve is not
        # the published one (review V7, L-d).
        "input_digest": solution.input_digest,
        "solved_at": solution.solved_at,
        "solve_identity": _identity(solution),
        "gate": {"id": gate["gate"], "params": gate["params"], "params_digest": gate["params_digest"],
                 "masks_applied": gate["masks_applied"], "metric_available": gate["metric_available"]},
        "components": entries,
    }


def _identity(solution) -> str:
    from tower.world_builder.global_solve import solve_identity  # noqa: PLC0415

    return solve_identity(solution)


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
                        depth_runner: Callable | None = None, metric_fn: Callable | None = None,
                        withhold=None) -> GateResult:
    """Steps 1-4 of the module docstring on the candidate; never raises. `depth_runner` and `metric_fn`
    replace `run_gate_depth` / `measure_metric_scale` (tests, and the consensus, which gates a draw again
    with the depth and scale it already has). `withhold`: `coherence_gate.apply_gate`'s consensus hook."""
    params = params or CG.GateParams()
    started = time.perf_counter()
    try:
        return _gate(store, world_id, session_id, solution, database_path=database_path, keyframes=keyframes,
                     should_stop=should_stop, params=params, depth_runner=depth_runner or run_gate_depth,
                     metric_fn=metric_fn or measure_metric_scale, started=started, withhold=withhold)
    except Exception as exc:  # noqa: BLE001 -- a broken gate publishes today's solve and says so
        logger.exception("[Tower][WorldBuilder] the evidence gate failed on %s/%s; the final solve is "
                         "published as the solver returned it, with no components record",
                         world_id, session_id)
        return GateResult(solution=solution, components=None, record={
            "state": GATE_STATE_FAILED, "detail": f"{type(exc).__name__}: {exc}", "gate": CG.GATE_ID,
            "params": params.to_json(), "params_digest": params.digest(),
            # Owes a re-gate (review V7, L-c): the finisher runs it in place when idle.
            "retryable": True, "cause": CAUSE_GATE_FAILED,
            "seconds": round(time.perf_counter() - started, 3)})


def _gate(store, world_id, session_id, solution, *, database_path, keyframes, should_stop, params,
          depth_runner, metric_fn, started, withhold=None) -> GateResult:
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
        cache = align.get("prediction_cache")
        if isinstance(cache, dict):
            # Where the predictions came from (R3): the cache, or the network this time.
            depth_record["predictions"] = {"token": cache.get("token"), "cached": cache.get("hits"),
                                           "predicted": cache.get("predicted")}
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
    scale = None
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
                           params=params, **({"withhold": withhold} if withhold else {}))
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
    # RETRYABLE (review V7, H1b): the scale fail-safe caused by the DEPTH STAGE -- the network
    # missing, CUDA out of memory, the surface lock held, a stop -- is not the world's
    # fault, and the finisher owes it a re-gate in place. The masks fail-safe is the
    # solve's: a re-gate in place cannot mask it (an owner re-finishes; `publish_notice`).
    #
    # A fail-safe WITH DEPTH IN HAND (fewer than `min_metric_fraction` of the supported
    # cameras with a level) is deliberately NOT retryable (review V8, M2a). The depth
    # stage ran to the end, and it is deterministic on the same keyframes: the frames it
    # leaves without a prediction are missing, undecodable or re-sized images, never a
    # transient; and the ratios come from this solve's own inlier pairs. A re-gate in place
    # would reproduce the shortfall and spend the finisher's attempts for nothing. It is
    # not silent either: the row says what is missing and that a new walk is what fixes it
    # (`NOTICE_SCALE_SHORT`).
    retryable = bool(masks_applied and depth is None)
    record = {
        "state": GATE_STATE_APPLIED,
        "retryable": retryable,
        "cause": CAUSE_DEPTH_UNAVAILABLE if retryable else None,
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
    return GateResult(solution=relabelled, record=record, components=doc, depth=depth, scale=scale,
                      gated=result, candidate=solution)


# ---------------------------------------------------------------------------
# the consensus: attachment decided by a majority of mapper seeds (review V8 H2; manager 019)
#
# WHY. On 6839fb8f the closet (138 keyframes) detached in ONE of five mapper seeds on one frozen database and
# the same depth (RUN P3-PF, var step 1: room 526 / 526 / 361 / 525 / 526), so which room a finish published
# depended on the seed. A decision that flips with the seed rests on marginal evidence. Majority of 3 mapper
# seeds, over all 10 triples, gave a room of 525-526 and no group of >= 30 keyframes flipped (P3-PF step 3).
#
# THE RULE (no threshold of its own). With `TOWER_WORLD_SOLVE_CONSENSUS` = N >= 2 on a gated, seeded final solve:
#   1. DRAWS: N candidates on the SAME frozen database, masks and depth predictions, mapper seeds s, s+1, ...
#      (DRAW_UNIT: PF's decomposition put the flips on the mapper seed alone). Draw 0 is the solve's own.
#   2. Each draw is gated as usual (`gate_final_solution`; its depth stage re-fits the kept predictions).
#   3. VOTES: per published keyframe, "attached to the room" in each draw; consensus = a strict majority.
#   4. PUBLISH the draw whose attach vector agrees with the consensus on the most keyframes (ties: the lowest
#      seed). Its GROUPS -- the gate's own candidate groups in its room, and each of its unplaced components --
#      are voted on by keyframe overlap (a group is attached in a draw when most of its keyframes are). A group
#      of its room that fewer than a strict majority of draws attached is WITHHELD: the draw is gated again
#      (the same depth and scale, CPU only) with that group barred from the room, and it is published as its
#      own piece, reason `seed-unstable`. The room's anchor group is never withheld. A group the majority
#      attached but the published draw did not stays unplaced: no geometry is invented.
#   5. RECORD `gate.consensus` (additive, Tower-internal §2.5), and each draw's per-round gate decisions in
#      `solve/<session>/consensus.json` (review V8 LOW: "gate per-round decisions not persisted").
# A consensus whose first draw took a fail-safe (nothing attached) has nothing to vote on: `not-needed`, or
# `deferred` when that fail-safe is owed a re-gate -- the re-gate in place then runs the consensus.

CONSENSUS_FILENAME = "consensus.json"
CONSENSUS_RECORD = "wb-gate-consensus/1"
DRAW_UNIT_MAPPER_SEED = "mapper-seed"
CONSENSUS_APPLIED = "applied"          # the draws ran and voted; `detached` may be empty
CONSENSUS_NOT_NEEDED = "not-needed"    # the first draw's gate attached nothing (a fail-safe)
CONSENSUS_DEFERRED = "deferred"        # that fail-safe is owed a re-gate, which runs the consensus
CONSENSUS_NOT_RUN = "not-run"          # it could not run (`why`)
DECISION_ANCHOR = "anchor"
DECISION_ATTACHED = "attached"
DECISION_SEED_UNSTABLE = CG.REASON_SEED_UNSTABLE
DECISION_UNPLACED = "unplaced"


@dataclasses.dataclass
class ConsensusPlan:
    """What `gate_by_consensus` is asked for. `map_draw(seed)` maps one further draw -- the same database,
    masks and depth -- and returns its candidate; `refusal` says why a consensus cannot run at all."""

    draws: int
    seed: int | None
    map_draw: Callable | None = None
    refusal: str | None = None
    unit: str = DRAW_UNIT_MAPPER_SEED

    def seeds(self) -> list:
        return [None if self.seed is None else int(self.seed) + k for k in range(int(self.draws))]


def _published_kids(solution, min_obs: int) -> set:
    return {kid for kid, p in (getattr(solution, "poses", None) or {}).items()
            if int(p.get("observations", 0)) >= min_obs}


def _room_kids(solution, min_obs: int) -> set:
    return {kid for kid, p in (getattr(solution, "poses", None) or {}).items()
            if int(p.get("component", 0)) == 0 and int(p.get("observations", 0)) >= min_obs}


def decide_consensus(results: list, *, kid_of_name: dict, min_obs: int = 30) -> dict:
    """The votes, the published draw and its groups' decisions, from the gated draws (`GateResult`s in draw
    order). Pure: no IO. Draws whose gate failed do not vote. Returns {"voting", "chosen", "agreement",
    "keyframes", "consensus_attached", "unanimous", "groups", "withhold"}; `withhold` names the groups (by
    their first camera) `coherence_gate.apply_gate` must bar from the room."""
    voting = [k for k, r in enumerate(results) if (r.record or {}).get("state") == GATE_STATE_APPLIED
              and r.gated is not None]
    n = len(voting)
    attached = {k: _room_kids(results[k].solution, min_obs) for k in voting}
    universe = set().union(*(_published_kids(results[k].solution, min_obs) for k in voting)) if voting else set()
    votes = {kid: sum(kid in attached[k] for k in voting) for kid in universe}
    consensus = {kid for kid, v in votes.items() if 2 * v > n}
    agreement = {k: sum((kid in attached[k]) == (kid in consensus) for kid in universe) for k in voting}
    chosen = max(voting, key=lambda k: (agreement[k], -k)) if voting else 0
    groups: list[dict] = []
    withhold: list[str] = []
    if voting:
        best = results[chosen]
        order = {kid: i for i, kid in enumerate(best.solution.keyframe_ids)}
        units = []
        for g in best.gated.get("groups") or []:
            if g.get("label") != 0:
                continue
            kids = sorted((kid_of_name[nm] for nm in g["members"] if nm in kid_of_name),
                          key=lambda kid: order.get(kid, 0))
            units.append({"kids": kids, "in_room": True, "anchor": bool(g.get("reference")),
                          "first_camera": g["first_camera"]})
        by_label: dict = {}
        for kid, p in best.solution.poses.items():
            lab = int(p.get("component", 0))
            if lab != 0 and int(p.get("observations", 0)) >= min_obs:
                by_label.setdefault(lab, []).append(kid)
        for lab in sorted(by_label):
            units.append({"kids": sorted(by_label[lab], key=lambda kid: order.get(kid, 0)), "in_room": False,
                          "anchor": False, "first_camera": None, "label": lab})
        for u in units:
            kids = u["kids"]
            if not kids:
                continue
            per_draw = [2 * len(set(kids) & attached[k]) > len(kids) for k in voting]
            yes = sum(per_draw)
            majority = 2 * yes > n
            if u["anchor"]:
                decision = DECISION_ANCHOR
            elif u["in_room"]:
                decision = DECISION_ATTACHED if majority else DECISION_SEED_UNSTABLE
            else:
                decision = DECISION_UNPLACED
            if decision == DECISION_SEED_UNSTABLE:
                withhold.append(u["first_camera"])
            groups.append({"first_keyframe": kids[0], "keyframes": len(kids), "in_room": u["in_room"],
                           "votes": per_draw, "attached_votes": yes, "draws": n,
                           "ambiguous": 0 < yes < n, "decision": decision,
                           **({"label": u["label"]} if "label" in u else {})})
    # THE FLIPS THE PUBLISHED DRAW'S GROUPS CANNOT SHOW (reporting only). A piece another draw left out of
    # its room may sit inside the published draw's anchor block (6839fb8f's closet: its own piece in mapper
    # seed 2, part of the room's block in seeds 0 and 1), where no group of the published draw names it.
    # Every other draw's unplaced piece the draws disagree on is listed with its votes, and whether the
    # published room holds it against the majority (`against_majority`: the anchor is never withheld).
    pieces: list[dict] = []
    if voting:
        best = results[chosen]
        order = {kid: i for i, kid in enumerate(best.solution.keyframe_ids)}
        seen = {frozenset(g_kids) for g_kids in ([u["kids"] for u in units] if voting else [])}
        for k in voting:
            if k == chosen:
                continue
            by_label: dict = {}
            for kid, p in results[k].solution.poses.items():
                lab = int(p.get("component", 0))
                if lab != 0 and int(p.get("observations", 0)) >= min_obs:
                    by_label.setdefault(lab, set()).add(kid)
            for lab in sorted(by_label):
                kids = by_label[lab]
                if frozenset(kids) in seen:
                    continue
                seen.add(frozenset(kids))
                per_draw = [2 * len(kids & attached[j]) > len(kids) for j in voting]
                yes = sum(per_draw)
                if not 0 < yes < n:
                    continue
                in_room = 2 * len(kids & attached[chosen]) > len(kids)
                pieces.append({"first_keyframe": min(kids, key=lambda kid: order.get(kid, 0)),
                               "keyframes": len(kids), "from_draw": k, "votes": per_draw,
                               "attached_votes": yes, "draws": n, "majority_attached": 2 * yes > n,
                               "in_published_room": in_room, "against_majority": in_room != (2 * yes > n)})
    return {"voting": voting, "chosen": chosen, "agreement": agreement, "keyframes": len(universe),
            "consensus_attached": len(consensus),
            "unanimous": sum(1 for v in votes.values() if v in (0, n)),
            "groups": groups, "withhold": withhold, "pieces": pieces}


def _room_anchor(result) -> str | None:
    return next((g["first_camera"] for g in ((result.gated or {}).get("groups") or [])
                 if g.get("label") == 0 and g.get("reference")), None)


def gate_by_consensus(store, world_id: str, session_id: str, solution, *, plan: ConsensusPlan, database_path,
                      keyframes, should_stop=None, params: "CG.GateParams | None" = None,
                      gate_runner: Callable | None = None) -> GateResult:
    """The consensus (see above) on `solution`, draw 0. Never raises: a draw that cannot be mapped or gated
    does not vote, and with nothing to vote on the first draw is published exactly as `gate_final_solution`
    gave it. The published result's record carries `consensus`; `consensus_detail` holds what
    `after_publish` persists. `gate_runner` replaces `gate_final_solution` (tests)."""
    from tower.world_builder.global_solve import solve_identity  # noqa: PLC0415

    params = params or CG.GateParams()
    run = gate_runner or gate_final_solution
    started = time.perf_counter()
    seeds = plan.seeds()
    base = {"record": CONSENSUS_RECORD, "requested": int(plan.draws), "unit": plan.unit, "seeds": seeds}

    def gate(candidate, **kw):
        return run(store, world_id, session_id, candidate, database_path=database_path, keyframes=keyframes,
                   should_stop=should_stop, params=params, **kw)

    first = gate(solution)

    def done(result, record, detail=None):
        result.record = dict(result.record, consensus=dict(
            base, **record, seconds=round(time.perf_counter() - started, 3)))
        result.consensus_detail = detail
        return result

    if plan.refusal:
        return done(first, {"state": CONSENSUS_NOT_RUN, "why": plan.refusal})
    if first.record.get("state") != GATE_STATE_APPLIED or not first.record.get("attach"):
        owed = bool(first.record.get("retryable"))
        return done(first, {
            "state": CONSENSUS_DEFERRED if owed else CONSENSUS_NOT_NEEDED,
            "why": ("the gate could not finish on the first draw; the re-gate in place runs the consensus"
                    if owed else "the gate attached nothing to the room (a fail-safe): there is nothing to "
                                 "vote on")})
    name_of = _image_names(keyframes)
    kid_of_name = {v: k for k, v in name_of.items()}
    results = [first]
    draws = [{"draw": 0, "seed": seeds[0], "map_s": None, "gate_s": first.record.get("seconds")}]
    for k in range(1, int(plan.draws)):
        if should_stop is not None and should_stop():
            draws.append({"draw": k, "seed": seeds[k], "skipped": "a stop was asked for"})
            break
        t = time.perf_counter()
        try:
            candidate = plan.map_draw(seeds[k])
        except Exception as exc:  # noqa: BLE001 -- a draw that cannot be mapped does not vote
            logger.exception("[Tower][WorldBuilder] consensus draw %d of %s/%s could not be mapped",
                             k, world_id, session_id)
            draws.append({"draw": k, "seed": seeds[k], "failed": f"{type(exc).__name__}: {exc}"})
            continue
        map_s = round(time.perf_counter() - t, 3)
        result = gate(candidate)
        results.append(result)
        draws.append({"draw": k, "seed": seeds[k], "map_s": map_s, "gate_s": result.record.get("seconds")})
    mapped = [d for d in draws if "map_s" in d]          # one per entry of `results`, in draw order
    decision = decide_consensus(results, kid_of_name=kid_of_name, min_obs=params.min_obs)
    for i, (info, result) in enumerate(zip(mapped, results)):
        info.update({"solver": getattr(result.candidate, "solver", None),
                     "gate_state": result.record.get("state"), "attach": result.record.get("attach"),
                     "solve_identity": solve_identity(result.solution),
                     "room_keyframes": len(_room_kids(result.solution, params.min_obs)),
                     "published_keyframes": len(_published_kids(result.solution, params.min_obs)),
                     "components": result.record.get("components"),
                     "predictions": (result.record.get("depth") or {}).get("predictions"),
                     "votes": i in decision["voting"], "agreement": decision["agreement"].get(i)})
    chosen = results[decision["chosen"]]
    record = {"state": CONSENSUS_APPLIED, "draws": draws,
              "votes": {"draws": len(decision["voting"]), "keyframes": decision["keyframes"],
                        "consensus_attached": decision["consensus_attached"],
                        "unanimous": decision["unanimous"]},
              "chosen": {"draw": mapped[decision["chosen"]]["draw"], "seed": mapped[decision["chosen"]]["seed"]},
              "groups": decision["groups"],
              "pieces": [dict(p, from_draw=mapped[p["from_draw"]]["draw"]) for p in decision["pieces"]],
              "detached": [g["first_keyframe"] for g in decision["groups"]
                           if g["decision"] == DECISION_SEED_UNSTABLE],
              "ambiguous": [g["first_keyframe"] for g in decision["groups"] if g["ambiguous"]]}
    published = chosen
    if decision["withhold"]:
        depth = chosen.depth or {}
        regated = gate(chosen.candidate,
                       depth_runner=lambda *a, **kw: (depth.get("align"), depth.get("work"), depth.get("dparams")),
                       metric_fn=lambda *a, **kw: chosen.scale, withhold=decision["withhold"])
        if (regated.record.get("state") == GATE_STATE_APPLIED and regated.components is not None
                and _room_anchor(regated) == _room_anchor(chosen)):
            published = regated
            published.record = dict(published.record, depth=chosen.record.get("depth"),
                                    metric_scale=chosen.record.get("metric_scale"))
        else:
            record["state"] = "not-applied"
            record["why"] = ("barring the seed-unstable groups from the room changed which piece is the room; "
                             "the chosen draw is published as it was gated")
            record["detached"] = []
    detail = {"record": CONSENSUS_RECORD, "session_id": session_id,
              "solve_identity": solve_identity(published.solution),
              "draws": [dict(info, rounds=(r.gated or {}).get("rounds")) for info, r in zip(mapped, results)]}
    if published is not chosen:
        detail["published_rounds"] = (published.gated or {}).get("rounds")
    return done(published, record, detail)


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
        if result.consensus_detail is not None:
            from tower.storage import write_json_atomic  # noqa: PLC0415

            write_json_atomic(Path(workspace_root) / CONSENSUS_FILENAME, result.consensus_detail)
            out["consensus_written"] = True
        if result.depth is not None:
            out["depth_handoff"] = hand_depth_to_surface(store, world_id, session_id, solution, result.depth)
    except Exception as exc:  # noqa: BLE001 -- the solve is published; this is its paperwork
        logger.exception("[Tower][WorldBuilder] after publishing %s/%s: %s", world_id, session_id, exc)
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def gate_and_publish(store, world_id: str, session_id: str, workspace, solution, *, final: bool,
                     gate: bool | None = None, database_path, keyframes, write: Callable,
                     should_stop=None, consensus: ConsensusPlan | None = None) -> tuple[object, dict | None]:
    """The final solve's publish step, in one call (`global_solve.solve` makes it in place of
    `write_solution`): the gate when `gate_setting_for(final, gate)`, then `write(workspace, solution)`,
    then the record and the depth hand-off. Returns (the published solution, its gate record or None).

    With the gate off this is `write(workspace, solution)` and nothing else, except that a components record
    left by an earlier gated solve is moved aside: it does not describe this solve.

    `consensus`: a plan of two or more draws (`TOWER_WORLD_SOLVE_CONSENSUS`) gates by `gate_by_consensus`;
    None, the default, is the single gate of today."""
    result = None
    if gate_setting_for(final, gate):
        if consensus is not None and int(consensus.draws) >= 2:
            result = gate_by_consensus(store, world_id, session_id, solution, plan=consensus,
                                       database_path=database_path, keyframes=keyframes,
                                       should_stop=should_stop)
        else:
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


# ---------------------------------------------------------------------------
# the re-gate, in place (review V7, H1b and L-c)


def regate_owed(store, world_id: str, session_id: str) -> str | None:
    """The cause a PUBLISHED gated solve owes a re-gate for, or None. Only a solve the gate
    ran on (a `gate` record in `solution.json`) can owe one: an ungated world never does."""
    from tower.storage import read_json_closed  # noqa: PLC0415

    path = store.world_dir(world_id) / "solve" / session_id / "solution.json"
    try:
        meta = read_json_closed(path) if path.exists() else None
    except (OSError, ValueError):
        return None
    gate = (meta or {}).get("gate")
    if not isinstance(gate, dict) or not gate.get("retryable"):
        return None
    return gate.get("cause") or ("gate-failed" if gate.get("state") == GATE_STATE_FAILED
                                 else CAUSE_DEPTH_UNAVAILABLE)


def solver_candidate(solution):
    """The published solution with the SOLVER's partition back (`solver_component`, kept by
    `relabel_solution`): what the gate saw. A solution the gate never relabelled (it
    failed) is its own candidate."""
    poses = {}
    labels = {}
    for kid, p in (solution.poses or {}).items():
        q = dict(p)
        if "solver_component" in q:
            q["component"] = int(q.pop("solver_component"))
        poses[kid] = q
        labels[kid] = int(q.get("component", 0))
    kf_label = np.asarray([labels.get(k, -1) for k in solution.keyframe_ids], np.int64)
    point_label = _point_labels(solution, kf_label)
    return dataclasses.replace(solution, poses=poses, component=point_label.astype(np.int32),
                               gate=None)


class RegateRefused(RuntimeError):
    """The re-gate cannot start (no solution, no database): nothing was attempted."""


REFUSAL_NO_GATED_SOLUTION = "no gated solution is published for this session"
REFUSAL_DATABASE_GONE = "the solve's database {name} is gone; an owner can re-finish this walk"


def _regate_inputs(store, world_id: str, session_id: str):
    """(the published solution, its workspace, the feature database it was solved from), or
    `RegateRefused` saying why a re-gate cannot start. Reads only. The ONE statement of the
    refusals: `regate_published` and `regate_refusal` both ask it."""
    from tower.world_builder.global_solve import load_solution, workspace_for  # noqa: PLC0415

    solution = load_solution(store, world_id, session_id)
    if solution is None or not isinstance(solution.gate, dict):
        raise RegateRefused(REFUSAL_NO_GATED_SOLUTION)
    workspace = workspace_for(store, world_id, session_id)
    name = (solution.solve or {}).get("database") or workspace.database_path.name
    database = workspace.root / name
    if not database.is_file():
        raise RegateRefused(REFUSAL_DATABASE_GONE.format(name=name))
    return solution, workspace, database


def regate_refusal(store, world_id: str, session_id: str) -> str | None:
    """Would a re-gate of this session be refused, and why: the reason `regate_published`
    would raise `RegateRefused` with, word for word, or None when it would start. READ-ONLY
    (review V8, FIN's OPEN 3): no lock, no write, no GPU -- for a finisher that must decide
    before it takes the world's lock. `regate_published` still re-asks under the lock."""
    try:
        _regate_inputs(store, world_id, session_id)
    except RegateRefused as exc:
        return str(exc)
    return None


def regate_published(store, world_id: str, session_id: str, *, should_stop=None,
                     gate_runner: Callable | None = None) -> dict:
    """Depth stage + metric scale + gate again, IN PLACE, on the published solve: nothing is
    moved aside, the solve itself is not redone. The caller holds the world's writer lock.
    Publishes the relabelled solution and its components record (or retires the record if
    the gate fails again). Raises `RegateRefused` before doing anything it cannot finish
    (`regate_refusal` is the same test, read-only)."""
    from tower.world_builder.global_solve import write_solution  # noqa: PLC0415

    solution, workspace, database = _regate_inputs(store, world_id, session_id)
    previous = {k: solution.gate.get(k) for k in ("state", "cause", "metric_available", "params_digest")}
    keyframes = store.read_keyframes(world_id, session_id)
    candidate = solver_candidate(solution)
    owed = (solution.gate or {}).get("consensus") or {}
    if owed.get("state") == CONSENSUS_DEFERRED and int(owed.get("requested") or 1) >= 2:
        # The consensus the solve asked for and could not run (its first draw was owed this
        # re-gate): its draws are mapped now, on the database the published solve mapped.
        from tower.world_builder.global_solve import frozen_draw_mapper  # noqa: PLC0415

        seeds = owed.get("seeds") or [None]
        plan = ConsensusPlan(draws=int(owed["requested"]), seed=seeds[0],
                             map_draw=None if seeds[0] is None else frozen_draw_mapper(
                                 store, world_id, session_id, database, candidate, keyframes=keyframes),
                             refusal=None if seeds[0] is not None else "the solve was not seeded")
        result = gate_by_consensus(store, world_id, session_id, candidate, plan=plan, database_path=database,
                                   keyframes=keyframes, should_stop=should_stop, gate_runner=gate_runner)
    else:
        result = (gate_runner or gate_final_solution)(store, world_id, session_id, candidate,
                                                      database_path=database, keyframes=keyframes,
                                                      should_stop=should_stop)
    published = result.solution
    record = dict(result.record, regate={"at": time.time(), "previous": previous})
    published.gate = record
    published.timing = dict(published.timing or {}, regate_s=result.record.get("seconds"))
    write_solution(workspace, published)
    out = after_publish(store, world_id, session_id, workspace.root, published, result)
    return {"gate": {k: record.get(k) for k in ("state", "retryable", "cause", "metric_available",
                                                  "attach", "components")},
            "publish": out,
            # The row's sentence after the re-gate (None when nothing is owed any more).
            "notice": publish_notice({"gate": record, "transients": published.transients})}


# ---------------------------------------------------------------------------
# what the row says (review V7, H2 and L-c)

#
# EVERY FAIL-SAFE SAYS SO (review V8, M2). A published gated solve that is not "masks
# applied and metric scale available" attaches nothing to the room, and the row says
# what is missing and WHO can fix it -- the idle Tower (a re-gate in place), an owner
# (a re-finish, or a new walk), or an operator first (a Tower that cannot run the
# masks) -- in contract §2.2's order: masks, then scale. One sentence per cause, in the
# style of §2.2's own example (*masks were not applied (GPU out of memory); an owner can
# re-finish this walk*). No metric figure (§2.4 rule 6).

NOTICE_MASKS_OOM = ("masks were not applied (GPU out of memory); an owner can re-finish this walk")
NOTICE_MASKS_OFF = ("masks were not applied (they are off on this Tower: TOWER_WORLD_SOLVE_MASKS); "
                    "an operator can turn them on, then an owner can re-finish this walk")
NOTICE_MASKS_UNAVAILABLE = ("masks were not applied ({why}); an operator can make the transient "
                            "detector run on this Tower, then an owner can re-finish this walk")
NOTICE_MASKS_FALLBACK = ("masks were applied by OneFormer alone, not by the union rule the "
                         "evidence gate needs; an operator can make Grounding DINO and SAM "
                         "available on this Tower, then an owner can re-finish this walk")
NOTICE_MASKS_PARTIAL = ("masks were not applied to {unmasked} of {images} images; "
                        "an owner can re-finish this walk")
NOTICE_REGATE = ("the evidence gate could not measure metric scale ({why}); "
                 "the Tower re-runs the gate when it is idle")
NOTICE_SCALE_SHORT = ("the evidence gate had too little metric scale to place pieces by it "
                      "({why}); the depth stage ran to the end, so re-running the gate would not "
                      "change this; an owner can re-capture this walk")
NOTICE_GATE_FAILED = "the evidence gate failed ({why}); the Tower re-runs it when it is idle"


def _masks_notice(transients: dict, gate: dict) -> str | None:
    """The masks fail-safe's sentence, or None. The GPU-out-of-memory one is said whatever
    the gate did (as before); every other only when the gate ran and did not have masks."""
    if transients.get("retryable") and transients.get("cause") == "gpu-oom":
        return NOTICE_MASKS_OOM
    if gate.get("state") != GATE_STATE_APPLIED or gate.get("masks_applied") is not False:
        return None
    state = transients.get("state")
    if not transients or transients.get("requested") is False:
        return NOTICE_MASKS_OFF
    if state == "applied":
        return None                 # an inconsistent record: nothing true to say
    if state == "partial":
        if transients.get("rule_fallback"):
            return NOTICE_MASKS_FALLBACK
        return NOTICE_MASKS_PARTIAL.format(unmasked=transients.get("images_unmasked", "some"),
                                           images=transients.get("images", "its"))
    return NOTICE_MASKS_UNAVAILABLE.format(
        why=transients.get("detail") or transients.get("cause") or "the detector did not run")


def _scale_short_with_depth(gate: dict) -> bool:
    """The gate ran with its depth stage in hand and still had no usable metric scale."""
    return (gate.get("state") == GATE_STATE_APPLIED and gate.get("metric_available") is False
            and not gate.get("retryable")
            and (gate.get("depth") or {}).get("state") == DEPTH_OK)


def publish_notice(summary: dict | None) -> str | None:
    """One sentence for the session's finalization `detail` (the row carries it), when the
    published solve owes something an owner, an operator or the idle Tower will do, or
    took a fail-safe nobody can undo but a new walk, else None."""
    if not isinstance(summary, dict):
        return None
    transients = summary.get("transients") or {}
    gate = summary.get("gate") or {}
    parts = []
    masks = _masks_notice(transients, gate)
    if masks:
        parts.append(masks)
    if gate.get("state") == GATE_STATE_FAILED:
        parts.append(NOTICE_GATE_FAILED.format(why=gate.get("detail") or "an error"))
    elif gate.get("retryable"):
        why = (gate.get("depth") or {}).get("detail") or gate.get("cause") or "no depth"
        parts.append(NOTICE_REGATE.format(why=why))
    elif _scale_short_with_depth(gate):
        why = ((gate.get("evidence") or {}).get("metric_scale")
               or "too few cameras had a metric level")
        parts.append(NOTICE_SCALE_SHORT.format(why=why))
    return "; ".join(parts) or None
