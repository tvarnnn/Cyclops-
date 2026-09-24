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
stage that did not finish owes a re-gate in place (`retryable`); a shortfall with depth in hand, a walk
with no camera intrinsics or a solve with no camera (review V11, LOW-15), and the masks fail-safe, do not
(a re-gate would reproduce them). A failure of the gate ITSELF
(an exception) publishes the solve exactly as the solver returned it, with `gate.state: "failed"` and no
components record -- `components: null`, "not computed", which is what every older world says.

THRESHOLDS: none new. The gate's are `GateParams` (evidence cited there); the scale's are the harness's
(`ScaleParams`); the area floor (30 keyframes or 5 s), the 2.0 s span join and the 8-span cap are the
contract's proposals (§2.3, §2.4 rule 5; OPEN T2, T4).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import os
import re
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
# The depth causes (`_depth_cause`) no re-gate in place can cure: it re-runs on the same walk, which still has
# no camera intrinsics, and the same solve, which still has no camera (review V11, LOW-15). The gate records
# them NOT `retryable`, so the finisher never re-runs them; the notice says who can fix them (`_OWNERS`).
DEPTH_CAUSES_NOT_RETRYABLE = ("depth-no-intrinsics", "depth-no-camera")

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
    # In memory only (review V11, LOW-1): on a result of `gate_by_consensus`, draw 0's own gate result as its
    # gate returned it -- before any consensus record, owed cause or publish touched it -- so that a second
    # pass over the same solve (`global_solve._publish_draw_0_first`) hands it back (`draw_0=`) instead of
    # gating draw 0 again.
    draw_0: "GateResult | None" = None


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
                   should_stop=None, hold: list | None = None) -> tuple[dict, Path, object]:
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
    on it.

    `hold` (review V10, L-17): a list the caller owns. On success the surface lock is NOT released
    here but appended to it, and the caller releases it once it has READ `work/` -- the gate, after
    its metric scale (`_gate`). Released here, a densify could replace `work/<ki>_pred.npy` between
    this stage and that read. On failure the lock is always released here."""
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
    handed_on = False
    try:
        try:
            align = run_depth_stage(store, world_id, session_id, _all_posed_in_component_zero(solution),
                                    intrinsics, dparams, root, should_stop=should_stop, prior=None,
                                    prediction_cache=True)
        except DepthUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 -- model missing, CUDA OOM, a camera mismatch: all "no depth"
            raise DepthUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if align.get("stopped_after") is not None:
            raise DepthUnavailable(f"{DEPTH_STOPPED}: the depth stage was stopped after "
                                   f"{align.get('stopped_after')} frames")
        if hold is not None:
            hold.append(lock)
            handed_on = True
    finally:
        if not handed_on:
            lock.release()
    return align, root / "work", dparams


# The product depth stage itself, for `_gate` to tell it from a caller's `depth_runner` (a test's fake, or the
# consensus's re-use of a draw's depth): only the product stage is handed the surface lock to keep (L-17).
_PRODUCT_DEPTH_RUNNER = run_gate_depth


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


def retire_consensus(workspace_root) -> bool:
    """A solution published without a consensus detail moves the previous `consensus.json` aside, to
    `CONSENSUS_SUPERSEDED_FILENAME` (never deletes it), exactly as `retire_components` does for the
    components record: an old record of per-draw decisions must not sit beside a new solve (review V11,
    LOW-4). True when one was moved."""
    path = Path(workspace_root) / CONSENSUS_FILENAME
    if not path.exists():
        return False
    try:
        os.replace(path, Path(workspace_root) / CONSENSUS_SUPERSEDED_FILENAME)
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
                        withhold=None, room=None, link_reader: Callable | None = None) -> GateResult:
    """Steps 1-4 of the module docstring on the candidate; never raises. `depth_runner` and `metric_fn`
    replace `run_gate_depth` / `measure_metric_scale` (tests, and the consensus, which gates a draw again
    with the depth and scale it already has). `withhold` and `room`: `coherence_gate.apply_gate`'s consensus
    hooks. `link_reader(database_path, camera, min_inliers) -> (links, rotations)` replaces `read_links`
    (the consensus reads the one frozen database once for all its draws)."""
    params = params or CG.GateParams()
    started = time.perf_counter()
    try:
        return _gate(store, world_id, session_id, solution, database_path=database_path, keyframes=keyframes,
                     should_stop=should_stop, params=params, depth_runner=depth_runner or run_gate_depth,
                     metric_fn=metric_fn or measure_metric_scale, started=started, withhold=withhold,
                     room=room, link_reader=link_reader or read_links)
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


def read_links(database_path, camera, min_inliers: int) -> tuple[dict, dict]:
    """The solve database's verified links and their two-view rotations (`coherence_gate`'s readers)."""
    return (CG.read_verified_links(database_path, min_inliers=min_inliers),
            CG.read_link_rotations(database_path, camera, min_inliers=min_inliers))


def _gate(store, world_id, session_id, solution, *, database_path, keyframes, should_stop, params,
          depth_runner, metric_fn, started, withhold=None, room=None, link_reader=read_links) -> GateResult:
    transients = solution.transients or {}
    masks_state = transients.get("state")
    masks_applied = masks_state == "applied"
    name_of = _image_names(keyframes)
    session = store.read_session(world_id, session_id)

    # 1. depth before publish, and 2. the metric scale that reads its predictions -- both under the
    # session's surface lock when the product depth stage runs (review V10, L-17: it used to be released
    # between them, so a densify could replace `work/` before the scale read it).
    held: list = []
    try:
        t = time.perf_counter()
        depth = None
        depth_record: dict = {"state": DEPTH_OK}
        try:
            if depth_runner is _PRODUCT_DEPTH_RUNNER:
                align, work, dparams = depth_runner(store, world_id, session_id, solution, session.intrinsics,
                                                    should_stop=should_stop, hold=held)
            else:
                align, work, dparams = depth_runner(store, world_id, session_id, solution, session.intrinsics,
                                                    should_stop=should_stop)
            depth = {"align": align, "work": work, "dparams": dparams}
            depth_record.update({"backend": align.get("backend"), "known_fov": align.get("known_fov"),
                                 "frames": align.get("targets"), "image_origins": align.get("image_origins")})
            cache = align.get("prediction_cache")
            if isinstance(cache, dict):
                # Where the predictions came from (R3): the cache, or the network this time -- and how many
                # could not be written to the cache (review V10, L-18; absent from a stage that predates it).
                depth_record["predictions"] = {"token": cache.get("token"), "cached": cache.get("hits"),
                                               "predicted": cache.get("predicted")}
                if "write_failed" in cache:
                    depth_record["predictions"]["write_failed"] = cache.get("write_failed")
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
            scale_record.update({k: scale.get(k) for k in ("cameras_published", "cameras_measured",
                                                            "pairs_used", "inliers_total", "inliers_gated",
                                                            "frames_without_prediction")})
        scale_record["state"] = "measured" if metric_log else "unavailable"
        scale_record["seconds"] = round(time.perf_counter() - t, 3)
    finally:
        for lock in held:
            lock.release()

    # 3. the gate
    t = time.perf_counter()
    links, rotations = link_reader(database_path, solution.camera, params.min_link_inliers)
    model = solve_model(solution, name_of)
    hooks = {"withhold": withhold, "room": room} if withhold else {}
    result = CG.apply_gate(model, links, metric_log, link_rotations=rotations, masks_applied=masks_applied,
                           params=params, **hooks)
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
    #
    # Nor is a depth stage that could not START for the walk's or the solve's own reason: no camera
    # intrinsics, no solve camera (review V11, LOW-15; `DEPTH_CAUSES_NOT_RETRYABLE`). The re-gate in place
    # runs on the same walk and the same solve, so it would fail the same way; its sentence already says
    # re-running the gate would not change this, and names the owner who can.
    retryable = bool(masks_applied and depth is None
                     and _depth_cause(depth_record.get("detail")) not in DEPTH_CAUSES_NOT_RETRYABLE)
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
#   2. Each draw is gated as usual (`gate_final_solution`; its depth stage re-fits the kept predictions; the
#      database's links and rotations are read once for every draw).
#   3. VOTES (review V9, M-1): only a draw whose gate ATTACHED votes (`attach: true`). A draw whose gate took a
#      fail-safe (depth unavailable or stopped, a metric fraction under half), failed, could not be mapped or was
#      skipped has no attachment to vote with -- counting it would vote every piece "detached". Per published
#      keyframe, "attached to the room" in each voting draw; consensus = a strict majority of the VOTING draws.
#      Fewer voting draws than requested: `partial`, with the count; fewer than 2: draw 0 is published as one
#      draw would be. A stop that cost a draw its vote, or a further draw whose gate could not finish for a
#      RETRYABLE cause (review V10, L-1): draw 0 is published, `deferred` (the re-gate in place runs the
#      consensus) -- never a vote of the draws that happened to finish.
#   4. PUBLISH the draw whose attach vector agrees with the consensus on the most keyframes (ties: the lowest
#      draw). Its GROUPS -- the gate's own candidate groups in its room, and each of its unplaced components --
#      are voted on by keyframe overlap (a group is attached in a draw when most of its keyframes are). A group
#      of its room that fewer than a strict majority of draws attached is WITHHELD whole, whatever its unanimous
#      keyframes (review V10, MED-4: the `kept` exception of P3.6 let one boundary keyframe keep a minority group
#      in the room; RV9-A P6's cost is accepted as conservative). The count of its keyframes every voting draw
#      attached is recorded (`unanimous_keyframes`), for audit only. The room's anchor group is never withheld. A
#      group the majority attached but the published draw did not stays unplaced: no geometry is invented.
#   5. THE WITHHOLD RE-GATE (review V9, H-1). The chosen draw is gated again (the same depth and scale, CPU only)
#      with the withheld groups SEALED (each a piece of its own; `coherence_gate.apply_gate(withhold=...)`) and the
#      room ALLOW-LISTED (only the chosen room's groups may join it: `room=...`). It is published only if, per
#      published keyframe, (a) no keyframe outside the chosen room is attached, (b) the room is exactly the chosen
#      room minus the withheld groups, (c) every withheld keyframe's only reason is `seed-unstable`, and (d) every
#      piece outside the chosen room is the chosen draw's, published with the reasons the chosen draw gave it
#      (`keep_outside_pieces`: re-decided against the smaller room, a piece the allow-list bars would read
#      `no-verified-link` whatever its links). Otherwise the consensus is `not-applied`, with why, and the chosen
#      draw is published as it was gated.
#   6. RECORD `gate.consensus` (additive, Tower-internal §2.5), and each draw's per-round gate decisions in
#      `solve/<session>/consensus.json` (review V8 LOW: "gate per-round decisions not persisted"). Every state
#      carries the vote keys (`groups`, `pieces`, `detached`, `ambiguous`, `held_against_majority`; empty when it
#      did not vote -- review V10, L-4). On `not-applied` a group the vote would have withheld reads `not-withheld`.
# A consensus whose first draw took a fail-safe (nothing attached) has nothing to vote on: `not-needed`, or
# `deferred` when that fail-safe is owed a re-gate -- the re-gate in place then runs the consensus.
#
# A STOP NEVER REACHES DRAW 0 OF A STOPPED CONSENSUS (review V10, MED-1b). With `stopped` (the solve's draw loop,
# and its early publish), draw 0 is gated as N = 1 gates it -- WITHOUT the stop, which would otherwise stop draw
# 0's own depth stage and publish its scale fail-safe. And a writer that already published this solve
# (`regate_published`; `gate_and_publish(keep_on_stop=True)`) writes NOTHING when a stop reached draw 0's depth
# stage: a published room is never replaced by an anchor-only fail-safe because a capture started.
#
# HELD AGAINST THE MAJORITY (review V9, M-2; manager 025, decision 2; reporting only, no rule).
# `gate.consensus.held_against_majority` counts the published keyframes in the PUBLISHED room that a strict
# majority of the VOTING draws did not attach -- anchor-absorbed keyframes included, which no group names and the
# consensus cannot withhold (the anchor is never withheld). It is the visible size of the residual risk: marginal
# pieces inside the anchor block are not re-verified by the vote.

CONSENSUS_FILENAME = "consensus.json"
# Where a published solve whose gate wrote no per-draw detail moves an older `consensus.json` (review V11, LOW-4):
# aside, never deleted, as `SUPERSEDED_FILENAME` is for the components record.
CONSENSUS_SUPERSEDED_FILENAME = "consensus.superseded.json"
CONSENSUS_RECORD = "wb-gate-consensus/1"
DRAW_UNIT_MAPPER_SEED = "mapper-seed"
CONSENSUS_APPLIED = "applied"          # every requested draw voted; `detached` may be empty
CONSENSUS_PARTIAL = "partial"          # fewer draws voted than were requested (`votes.draws`, `why`)
CONSENSUS_NOT_NEEDED = "not-needed"    # the first draw's gate attached nothing (a fail-safe)
# Owed to the re-gate in place, which runs the consensus: the first draw's fail-safe is
# retryable, or a stop was asked for before the further draws voted (draw 0 is published).
CONSENSUS_DEFERRED = "deferred"
CONSENSUS_NOT_RUN = "not-run"          # it could not run (`why`)
CONSENSUS_NOT_APPLIED = "not-applied"  # the withhold re-gate failed its checks; the chosen draw is published
# `gate.cause` of a published draw 0 whose consensus a stop deferred (`gate.retryable`).
CAUSE_CONSENSUS_DEFERRED = "consensus-deferred"
WHY_STOPPED = ("a stop was asked for before the further draws voted; draw 0 is published as one draw would "
               "be, and the re-gate in place runs the consensus")
# Review V10, L-1: a further draw's gate could not finish for a cause a re-run can cure (its depth stage: GPU
# memory, the surface lock, the network; or the gate itself failed), so it could not vote.
WHY_DRAW_UNFINISHED = ("a further draw's gate could not finish (a cause the re-gate in place can cure), so it "
                       "could not vote; draw 0 is published as one draw would be, and the re-gate in place runs "
                       "the consensus")
# Review V10, MED-1(a) (the solve's early publish, `stopped=True, why_deferred=WHY_PUBLISHED_FIRST`): draw 0 is
# published before any further draw is mapped, so a kill during the draws still leaves a finished world.
WHY_PUBLISHED_FIRST = ("draw 0 is published first, before the further draws are mapped; the consensus replaces "
                       "it once they have voted, and if they never do the re-gate in place runs it")
# Review V10, MED-1(b): what `gate_and_publish(keep_on_stop=True)` and `regate_published` say when a stop reached
# draw 0's own depth stage and they therefore wrote nothing.
WHY_KEPT_ON_STOP = ("a stop reached draw 0's depth stage, so nothing was published; the solve published before "
                    "stands")
DECISION_ANCHOR = "anchor"
DECISION_ATTACHED = "attached"
DECISION_SEED_UNSTABLE = CG.REASON_SEED_UNSTABLE
DECISION_UNPLACED = "unplaced"
# Review V10, L-4: the vote would have withheld this room group (`seed-unstable`), but the withhold re-gate failed
# its checks (consensus `not-applied`), so the chosen draw is published as it was gated -- the group in its room.
DECISION_NOT_WITHHELD = "not-withheld"


def _no_vote() -> dict:
    """The vote keys every consensus record carries, whatever its state (review V10, L-4): a state that did not
    vote has them empty, so no reader needs a per-state key list."""
    return {"groups": [], "pieces": [], "detached": [], "ambiguous": [], "held_against_majority": 0}


@dataclasses.dataclass
class ConsensusPlan:
    """What `gate_by_consensus` is asked for. `map_draw(seed)` maps one further draw -- the same database,
    masks and depth -- and returns its candidate; `refusal` says why a consensus cannot run at all."""

    draws: int
    seed: int | None
    map_draw: Callable | None = None
    refusal: str | None = None
    unit: str = DRAW_UNIT_MAPPER_SEED
    # The draws' seeds in draw order, when they are not `seed, seed + 1, ...` (review V10, L-5: a re-gate of a
    # published draw k maps its own seed as draw 0 and the others' after it). None: `seed + k`.
    seed_list: list | None = None

    def seeds(self) -> list:
        if self.seed_list is not None:
            return [int(s) for s in list(self.seed_list)[:int(self.draws)]]
        return [None if self.seed is None else int(self.seed) + k for k in range(int(self.draws))]


def _published_kids(solution, min_obs: int) -> set:
    return {kid for kid, p in (getattr(solution, "poses", None) or {}).items()
            if int(p.get("observations", 0)) >= min_obs}


def _room_kids(solution, min_obs: int) -> set:
    return {kid for kid, p in (getattr(solution, "poses", None) or {}).items()
            if int(p.get("component", 0)) == 0 and int(p.get("observations", 0)) >= min_obs}


def draw_votes(result) -> bool:
    """Whether a gated draw votes (review V9, M-1): its gate ran (`applied`, with its groups) and ATTACHED --
    not a fail-safe, whose room is the anchor block alone whatever the evidence says."""
    record = getattr(result, "record", None) or {}
    return (record.get("state") == GATE_STATE_APPLIED and getattr(result, "gated", None) is not None
            and record.get("attach") is True)


def decide_consensus(results: list, *, kid_of_name: dict, min_obs: int = 30) -> dict:
    """The votes, the published draw and its groups' decisions, from the gated draws (`GateResult`s in draw
    order). Pure: no IO. Only draws whose gate attached vote (`draw_votes`). Returns {"voting", "chosen",
    "agreement", "keyframes", "consensus_attached", "unanimous", "groups", "withhold", "pieces", "tally"};
    `withhold` names the groups (by their first camera) `coherence_gate.apply_gate` must seal, `tally` is
    {keyframe: attached votes} (in memory only)."""
    voting = [k for k, r in enumerate(results) if draw_votes(r)]
    n = len(voting)
    attached = {k: _room_kids(results[k].solution, min_obs) for k in voting}
    universe = set().union(*(_published_kids(results[k].solution, min_obs) for k in voting)) if voting else set()
    votes = {kid: sum(kid in attached[k] for k in voting) for kid in universe}
    consensus = {kid for kid, v in votes.items() if 2 * v > n}
    agreement = {k: sum((kid in attached[k]) == (kid in consensus) for kid in universe) for k in voting}
    chosen = max(voting, key=lambda k: (agreement[k], -k)) if voting else 0
    groups: list[dict] = []
    withhold: list[str] = []
    units: list[dict] = []
    if voting:
        best = results[chosen]
        order = {kid: i for i, kid in enumerate(best.solution.keyframe_ids)}
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
            extra: dict = {}
            if u["anchor"]:
                decision = DECISION_ANCHOR
            elif u["in_room"] and majority:
                decision = DECISION_ATTACHED
            elif u["in_room"]:
                # THE GROUP-LEVEL VOTE DECIDES (review V10, MED-4): a room group fewer than a strict majority
                # of the voting draws attached is withheld whole, whatever its unanimous keyframes -- the gate's
                # unit is the group, and one boundary keyframe every seed happens to hold must not keep a
                # minority attachment in the room (RV9-A P6's cost is accepted as conservative). How many of its
                # keyframes every voting draw attached is recorded, for audit only.
                decision = DECISION_SEED_UNSTABLE
                extra["unanimous_keyframes"] = sum(1 for kid in kids if votes.get(kid, 0) == n)
            else:
                decision = DECISION_UNPLACED
            if decision == DECISION_SEED_UNSTABLE:
                withhold.append(u["first_camera"])
            # `keyframe_ids` (Tower-internal, for audit and the acceptance scorer): which keyframes the
            # group is, so a flip's coverage by an ambiguous report is exact, not a size bound.
            groups.append({"first_keyframe": kids[0], "keyframes": len(kids), "keyframe_ids": list(kids),
                           "in_room": u["in_room"],
                           "votes": per_draw, "attached_votes": yes, "draws": n,
                           "ambiguous": 0 < yes < n, "decision": decision, **extra,
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
        seen = {frozenset(u["kids"]) for u in units}
        for k in voting:
            if k == chosen:
                continue
            by_label = {}
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
                               "keyframes": len(kids),
                               "keyframe_ids": sorted(kids, key=lambda kid: order.get(kid, 0)),
                               "from_draw": k, "votes": per_draw,
                               "attached_votes": yes, "draws": n, "majority_attached": 2 * yes > n,
                               "in_published_room": in_room, "against_majority": in_room != (2 * yes > n)})
    return {"voting": voting, "chosen": chosen, "agreement": agreement, "keyframes": len(universe),
            "consensus_attached": len(consensus),
            "unanimous": sum(1 for v in votes.values() if v in (0, n)),
            "groups": groups, "withhold": withhold, "pieces": pieces, "tally": votes}


def held_against_majority(room_kids, tally: dict, voting_draws: int) -> int:
    """How many keyframes of a published room a strict majority of the VOTING draws did not attach (review
    V9, M-2): `2 * (n - attached votes) > n`. Anchor-absorbed keyframes count like any other."""
    n = int(voting_draws)
    return sum(1 for kid in room_kids if 2 * (n - int(tally.get(kid, 0))) > n)


def _owe_consensus(result: GateResult) -> GateResult:
    """Draw 0's gate result, owed the consensus a stop deferred: `retryable` with the cause
    `consensus-deferred`, so the finisher's re-gate in place (contract §7 rule 5: only a record
    that says `retryable`) runs it. Everything else is the draw's own record, as N = 1 publishes."""
    result.record = dict(result.record, retryable=True, cause=CAUSE_CONSENSUS_DEFERRED)
    return result


def _untouched_copy(result: GateResult) -> GateResult:
    """A copy of a gate result that nothing later done to the original reaches (review V11, LOW-1). The
    consensus and the publish REPLACE a result's `record`, `consensus_detail`, and its solution's `gate` and
    `timing` -- none of them is edited in place -- so a copy of the result, its record and its solution is
    enough; the poses and the arrays are shared, and nothing writes to them."""
    return dataclasses.replace(result, record=dict(result.record), solution=copy.copy(result.solution),
                               consensus_detail=None, draw_0=None)


def _seconds(value) -> float:
    """A record's `seconds`, or 0.0 when it has none."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _room_anchor(result) -> str | None:
    return next((g["first_camera"] for g in ((result.gated or {}).get("groups") or [])
                 if g.get("label") == 0 and g.get("reference")), None)


def withhold_refusal(chosen: GateResult, regated: GateResult, withheld_kids: set, *, min_obs: int) -> str | None:
    """Why the withhold re-gate may NOT be published (review V9, H-1), or None when it may: per published
    keyframe, it attaches nothing outside the chosen room, its room is exactly the chosen room minus the
    withheld groups, and every withheld keyframe is published with `seed-unstable` as its only reason."""
    if regated.record.get("state") != GATE_STATE_APPLIED or regated.components is None:
        return "the re-gate did not publish a components record"
    before = _room_kids(chosen.solution, min_obs)
    after = _room_kids(regated.solution, min_obs)
    added = after - before
    if added:
        return f"the re-gate attached {len(added)} keyframes that the chosen draw's room did not hold"
    expected = before - set(withheld_kids)
    if after != expected:
        return (f"the re-gate's room is not the chosen room minus the withheld groups: {len(expected - after)} "
                "further keyframes left it")
    reasons = {kid: list(e.get("reasons") or []) for e in regated.components.get("components") or []
               for kid in e.get("keyframe_ids") or []}
    published = _published_kids(regated.solution, min_obs)
    wrong = [kid for kid in withheld_kids if kid in published
             and reasons.get(kid) != [CG.REASON_SEED_UNSTABLE]]
    if wrong:
        return (f"{len(wrong)} withheld keyframes were not published with seed-unstable as their only "
                "reason")
    return None


def keep_outside_pieces(chosen: GateResult, regated: GateResult, withheld_kids: set, *,
                        min_obs: int) -> str | None:
    """A piece outside the chosen room is published as the chosen draw published it: the same keyframes, and
    the REASONS the chosen draw's gate gave it (patched into `regated`'s components record and its solution's
    `components[*].gate_reasons`). The re-gate re-decides such a piece against the smaller room, and a group
    the allow-list bars would otherwise read `no-verified-link` whatever its links (RV9 probe, case 1). Returns
    why not (the re-gate changed such a piece's keyframes), or None when every one was kept."""
    if chosen.components is None or regated.components is None:
        return "a draw has no components record"
    src_of = {kid: e for e in chosen.components.get("components") or [] for kid in e.get("keyframe_ids") or []}
    kept: dict = {}
    for e in regated.components.get("components") or []:
        kids = set(e.get("keyframe_ids") or [])
        if kids & set(withheld_kids) and not kids <= set(withheld_kids):
            return "a withheld group was published in one piece with other keyframes"
        if e.get("state") == CG_PLACED or not kids or kids & set(withheld_kids):
            continue
        src = src_of.get(next(iter(sorted(kids))))
        if src is None or set(src.get("keyframe_ids") or []) != kids:
            return "a piece outside the chosen room changed its keyframes"
        kept[e["id"]] = (list(src.get("reasons") or []), src.get("reason"))
    for e in regated.components.get("components") or []:
        if e.get("id") in kept:
            e["reasons"], e["reason"] = kept[e["id"]]
    reasons_of = {kid: kept[e["id"]][0] for e in regated.components.get("components") or [] if e.get("id") in kept
                  for kid in e.get("keyframe_ids") or []}
    poses = regated.solution.poses or {}
    for comp in getattr(regated.solution, "components", None) or []:
        members = [kid for kid, p in poses.items()
                   if int(p.get("component", 0)) == int(comp.get("index", -1))
                   and int(p.get("observations", 0)) >= min_obs]
        if members and members[0] in reasons_of:
            comp["gate_reasons"] = list(reasons_of[members[0]])
    return None


def draw_0_stopped(result) -> bool:
    """Whether a stop reached the depth stage of the gate that produced `result` (draw 0's, for a consensus:
    only draw 0's record is published with its own depth). Such a result is the scale fail-safe, and a writer
    that already published this solve must not replace it with that (review V10, MED-1b)."""
    record = getattr(result, "record", None) or {}
    depth = record.get("depth")
    return (record.get("state") == GATE_STATE_APPLIED and isinstance(depth, dict)
            and depth.get("state") == DEPTH_STOPPED)


def gate_by_consensus(store, world_id: str, session_id: str, solution, *, plan: ConsensusPlan, database_path,
                      keyframes, should_stop=None, params: "CG.GateParams | None" = None,
                      gate_runner: Callable | None = None, stopped: bool = False,
                      why_deferred: str | None = None, draw_0: GateResult | None = None) -> GateResult:
    """The consensus (see above) on `solution`, draw 0. Never raises: a draw that cannot be mapped or gated
    does not vote, and with nothing to vote on the first draw is published exactly as `gate_final_solution`
    gave it. The published result's record carries `consensus`; `consensus_detail` holds what
    `after_publish` persists. `gate_runner` replaces `gate_final_solution` (tests).

    `stopped` (review V9, M-3; the solve's draw loop): a stop was asked for before the further draws -- none
    is mapped, and draw 0 is published as N = 1 publishes it, `deferred`. Draw 0 is then gated WITHOUT
    `should_stop` (review V10, MED-1b), exactly as N = 1 gates it: handed on, the stop reached draw 0's own
    depth stage and published its scale fail-safe, so this branch was unreachable. `why_deferred` replaces
    the record's `why` (`WHY_STOPPED`) -- the solve's early publish says `WHY_PUBLISHED_FIRST`.

    `should_stop` is asked between the further draws and reaches their gates: a stop there publishes draw 0,
    `deferred`, and the re-gate in place owes the consensus. So does a further draw whose gate could not
    finish for a retryable cause (review V10, L-1).

    `draw_0` (review V11, LOW-1): draw 0's gate result from an earlier pass over THIS solve -- the `draw_0`
    of the result that pass returned. Draw 0 is then not gated again: it is the draw the earlier pass gated
    and published, so a transient failure (the surface lock, GPU memory) can no longer turn it into an
    anchor-only fail-safe the second time, and what the record says it cost is what it cost -- its
    `gate_s`, its `predictions` and its share of `seconds` are that gate's. Every result this returns
    carries draw 0's own gate result, untouched, as its `draw_0`."""
    from tower.world_builder.global_solve import solve_identity  # noqa: PLC0415

    params = params or CG.GateParams()
    run = gate_runner or gate_final_solution
    started = time.perf_counter()
    seeds = plan.seeds()
    requested = int(plan.draws)
    base = {"record": CONSENSUS_RECORD, "requested": requested, "unit": plan.unit, "seeds": seeds}
    # THE ONE FROZEN DATABASE IS READ ONCE (review V9 LOW, per-draw waste): every draw has the same links and
    # two-view rotations; only its poses differ.
    link_cache: dict = {}

    def link_reader(database, camera, min_inliers):
        key = (str(database), json.dumps(camera, sort_keys=True, default=str), int(min_inliers))
        if key not in link_cache:
            link_cache[key] = read_links(database, camera, min_inliers)
        return link_cache[key]

    def gate(candidate, **kw):
        if gate_runner is None:
            kw.setdefault("link_reader", link_reader)
        kw.setdefault("should_stop", should_stop)
        return run(store, world_id, session_id, candidate, database_path=database_path, keyframes=keyframes,
                   params=params, **kw)

    if draw_0 is not None:
        # DRAW 0 IS GATED ONCE (review V11, LOW-1): the earlier pass's gate of it, as that gate returned it.
        first = _untouched_copy(draw_0)
        draw_0_s = _seconds(first.record.get("seconds"))
    else:
        # A stop already asked for does not reach draw 0 (MED-1b): it is gated as N = 1 gates it.
        first = gate(solution, should_stop=None) if stopped else gate(solution)
        draw_0_s = 0.0
    untouched = _untouched_copy(first)

    def done(result, record, detail=None):
        record = dict(record)
        for key, empty in _no_vote().items():
            record.setdefault(key, empty)
        # `seconds` counts draw 0's gate where it ran: here, or in the earlier pass that handed it on.
        result.record = dict(result.record, consensus=dict(
            base, **record, seconds=round(time.perf_counter() - started + draw_0_s, 3)))
        result.consensus_detail = detail
        result.draw_0 = untouched
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
    if stopped:
        # A STOP BEFORE THE FURTHER DRAWS (review V9, M-1 and M-3): draw 0 is published as N = 1
        # would publish it, and the consensus is owed to the re-gate in place.
        return done(_owe_consensus(first), {"state": CONSENSUS_DEFERRED, "why": why_deferred or WHY_STOPPED})
    name_of = _image_names(keyframes)
    kid_of_name = {v: k for k, v in name_of.items()}
    results = [first]
    draws = [{"draw": 0, "seed": seeds[0], "map_s": None, "gate_s": first.record.get("seconds")}]
    stop_cost_a_vote = False
    unfinished = False          # a further draw's gate could not finish, for a retryable cause (L-1)
    for k in range(1, requested):
        if should_stop is not None and should_stop():
            draws.append({"draw": k, "seed": seeds[k], "skipped": "a stop was asked for"})
            stop_cost_a_vote = True
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
        if not draw_votes(result) and (result.record.get("depth") or {}).get("state") == DEPTH_STOPPED:
            stop_cost_a_vote = True          # the stop reached this draw's depth stage: it cannot vote
        elif not draw_votes(result) and result.record.get("retryable"):
            # Its gate could not finish for a cause a re-run can cure -- GPU memory, the surface lock, the
            # network, or the gate itself failing (review V10, L-1). A vote of the draws that did finish
            # would publish a smaller electorate for good (with 2 voters a strict majority is unanimity).
            unfinished = True
    mapped = [d for d in draws if "map_s" in d]          # one per entry of `results`, in draw order
    decision = decide_consensus(results, kid_of_name=kid_of_name, min_obs=params.min_obs)
    voting = decision["voting"]
    for i, (info, result) in enumerate(zip(mapped, results)):
        info.update({"solver": getattr(result.candidate, "solver", None),
                     "gate_state": result.record.get("state"), "attach": result.record.get("attach"),
                     "solve_identity": solve_identity(result.solution),
                     "room_keyframes": len(_room_kids(result.solution, params.min_obs)),
                     "published_keyframes": len(_published_kids(result.solution, params.min_obs)),
                     "components": result.record.get("components"),
                     "predictions": (result.record.get("depth") or {}).get("predictions"),
                     "votes": i in voting, "agreement": decision["agreement"].get(i)})

    def detail_of(published):
        return {"record": CONSENSUS_RECORD, "session_id": session_id,
                "solve_identity": solve_identity(published.solution),
                "draws": [dict(info, rounds=(r.gated or {}).get("rounds")) for info, r in zip(mapped, results)]}

    if (stop_cost_a_vote or unfinished) and len(voting) < requested:
        # A stop cost a draw its vote (review V9, M-1), or a draw's gate could not finish for a retryable
        # cause (review V10, L-1): draw 0, as one draw publishes it; owed to the re-gate in place.
        published = _owe_consensus(first)
        why = WHY_STOPPED if stop_cost_a_vote else WHY_DRAW_UNFINISHED
        return done(published, {"state": CONSENSUS_DEFERRED, "why": why, "draws": draws,
                                "votes": {"draws": len(voting)}, "chosen": {"draw": 0, "seed": seeds[0]}},
                    detail_of(published))
    if len(voting) < 2:
        # Nothing to vote with but draw 0 (review V9, M-1): published unchanged, not owed.
        return done(first, {"state": CONSENSUS_PARTIAL, "draws": draws, "votes": {"draws": len(voting)},
                            "chosen": {"draw": 0, "seed": seeds[0]}, "groups": [], "pieces": [],
                            "detached": [], "ambiguous": [], "held_against_majority": 0,
                            "why": (f"{len(voting)} of {requested} draws voted (a draw votes only when its gate "
                                    "attached); with fewer than 2 there is no vote, so draw 0 is published as "
                                    "one draw would be")}, detail_of(first))
    chosen = results[decision["chosen"]]
    groups = decision["groups"]
    ambiguous = [g["first_keyframe"] for g in groups if g["ambiguous"]]
    for p in decision["pieces"]:
        if p["first_keyframe"] not in ambiguous:
            ambiguous.append(p["first_keyframe"])       # M-2: a disputed piece is ambiguous too
    record = {"state": CONSENSUS_APPLIED if len(voting) == requested else CONSENSUS_PARTIAL,
              "draws": draws,
              "votes": {"draws": len(voting), "keyframes": decision["keyframes"],
                        "consensus_attached": decision["consensus_attached"],
                        "unanimous": decision["unanimous"]},
              "chosen": {"draw": mapped[decision["chosen"]]["draw"], "seed": mapped[decision["chosen"]]["seed"]},
              "groups": groups,
              "pieces": [dict(p, from_draw=mapped[p["from_draw"]]["draw"]) for p in decision["pieces"]],
              "detached": [g["first_keyframe"] for g in groups if g["decision"] == DECISION_SEED_UNSTABLE],
              "ambiguous": ambiguous}
    if record["state"] == CONSENSUS_PARTIAL:
        record["why"] = (f"{len(voting)} of {requested} draws voted (a draw votes only when its gate attached); "
                         "the strict majority is of the draws that voted")
    published = chosen
    withhold = [c for c in decision["withhold"] if c != _room_anchor(chosen)]
    if withhold:
        depth = chosen.depth or {}
        room_groups = [g["first_camera"] for g in (chosen.gated or {}).get("groups") or [] if g.get("label") == 0]
        withheld_kids = {kid_of_name[nm] for g in (chosen.gated or {}).get("groups") or []
                         if g.get("label") == 0 and g["first_camera"] in withhold
                         for nm in g["members"] if nm in kid_of_name}
        regated = gate(chosen.candidate,
                       depth_runner=lambda *a, **kw: (depth.get("align"), depth.get("work"), depth.get("dparams")),
                       metric_fn=lambda *a, **kw: chosen.scale, withhold=withhold, room=room_groups)
        refusal = (withhold_refusal(chosen, regated, withheld_kids, min_obs=params.min_obs)
                   or keep_outside_pieces(chosen, regated, withheld_kids, min_obs=params.min_obs))
        if refusal is None:
            published = regated
            published.record = dict(published.record, depth=chosen.record.get("depth"),
                                    metric_scale=chosen.record.get("metric_scale"))
        else:
            record["state"] = CONSENSUS_NOT_APPLIED
            record["why"] = (f"withholding the seed-unstable groups failed its check ({refusal}); the chosen "
                             "draw is published as it was gated")
            record["detached"] = []
            # The groups the vote would have withheld are published in the room (review V10, L-4): their
            # decision says so, and their votes still say why they were in dispute.
            record["groups"] = [dict(g, decision=DECISION_NOT_WITHHELD) if g["decision"] == DECISION_SEED_UNSTABLE
                                else g for g in groups]
    record["held_against_majority"] = held_against_majority(
        _room_kids(published.solution, params.min_obs), decision["tally"], len(voting))
    detail = detail_of(published)
    if published is not chosen:
        detail["published_rounds"] = (published.gated or {}).get("rounds")
    return done(published, record, detail)


def after_publish(store, world_id: str, session_id: str, workspace_root, solution,
                  result: GateResult | None) -> dict:
    """Steps 5-6, after `write_solution` published `solution`. Never raises. Without a gate result (the
    gate off) a previous record is moved aside; with one, the record is written once and the depth is
    handed to the surface. A previous `consensus.json` is moved aside whenever this solve has no
    per-draw detail to write in its place (review V11, LOW-4)."""
    out: dict = {}
    try:
        if result is None or result.components is None:
            out["components_retired"] = retire_components(workspace_root)
        else:
            write_components(workspace_root, result.components)
            out["components_written"] = True
        if result is None or result.consensus_detail is None:
            # AN OLDER `consensus.json` DOES NOT DESCRIBE THIS SOLVE (review V11, LOW-4): a solve published
            # with no per-draw detail -- N = 1, the gate off, a consensus deferred, not needed or not run,
            # and the solve's early publish of draw 0 (review V10, MED-1a) -- moves it aside, so no reader
            # pairs the old per-draw decisions with this solve. Said only when there was one.
            if retire_consensus(workspace_root):
                out["consensus_retired"] = True
            if result is None:
                return out
        else:
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
                     should_stop=None, consensus: ConsensusPlan | None = None,
                     stopped: bool = False, why_deferred: str | None = None,
                     keep_on_stop: bool = False, draw_0: GateResult | None = None,
                     gate_results: list | None = None) -> tuple[object, dict | None]:
    """The final solve's publish step, in one call (`global_solve.solve` makes it in place of
    `write_solution`): the gate when `gate_setting_for(final, gate)`, then `write(workspace, solution)`,
    then the record and the depth hand-off. Returns (the published solution, its gate record or None).

    With the gate off this is `write(workspace, solution)` and nothing else, except that a components record
    left by an earlier gated solve is moved aside: it does not describe this solve.

    `consensus`: a plan of two or more draws (`TOWER_WORLD_SOLVE_CONSENSUS`) gates by `gate_by_consensus`;
    None, the default, is the single gate of today. `stopped` and `why_deferred`: `gate_by_consensus`'s -- a
    stop was asked for before the further draws (or the solve publishes draw 0 first, review V10 MED-1a):
    draw 0 is gated without the stop and published as N = 1 would, the consensus `deferred`. Ignored
    without a consensus plan.

    `keep_on_stop` (review V10, MED-1b; for a caller that ALREADY published this solve, such as the solve's
    second pass after its early publish): when a stop reached the depth stage of draw 0's gate, nothing is
    written -- not the solution, not the record, not the depth -- and `(None, record)` is returned, the
    record's `publish` saying so (`{"written": False, "why": WHY_KEPT_ON_STOP}`). False, the default, is
    today's: such a result is published (a stop never loses a finish that has nothing published yet).

    `draw_0` and `gate_results` (review V11, LOW-1; the solve's two passes, `_publish_draw_0_first`):
    `draw_0` is `gate_by_consensus`'s -- draw 0's gate result from the earlier pass, so draw 0 is gated once
    -- and ignored without a consensus plan. `gate_results`, a list the caller owns, is given the gate
    result (its `draw_0` is what the second pass hands back). Neither changes what is published."""
    result = None
    if gate_setting_for(final, gate):
        if consensus is not None and int(consensus.draws) >= 2:
            result = gate_by_consensus(store, world_id, session_id, solution, plan=consensus,
                                       database_path=database_path, keyframes=keyframes,
                                       should_stop=should_stop, stopped=stopped, why_deferred=why_deferred,
                                       draw_0=draw_0)
        else:
            result = gate_final_solution(store, world_id, session_id, solution, database_path=database_path,
                                         keyframes=keyframes, should_stop=should_stop)
        if gate_results is not None:
            gate_results.append(result)
        if keep_on_stop and draw_0_stopped(result):
            logger.info("[Tower][WorldBuilder] %s/%s: a stop reached draw 0's depth stage; nothing is "
                        "published, and the solve published before stands", world_id, session_id)
            return None, dict(result.record, publish={"written": False, "why": WHY_KEPT_ON_STOP})
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
REFUSAL_NO_DRAW_CAMERA = ("the solve has no camera to map its consensus draws with; an owner can "
                          "re-finish this walk")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _owed_consensus(solution) -> tuple[int, list | None]:
    """(draws requested, the draws' seeds in draw order) of the consensus the published solve asked
    for; (1, None) when it asked for none, or its record is not readable as one. The seeds are None
    when the record names none (an unseeded solve).

    CAPPED (review V10, L-5) by the setting's own rule (`config.WORLD_SOLVE_CONSENSUS_VALUES`: 1, 3, 5
    or 7): a record that asked for anything else -- an old record with N = 30 -- is re-gated as one
    draw, and logged, exactly as `TOWER_WORLD_SOLVE_CONSENSUS` would treat that value today.

    THE PUBLISHED SOLVE IS DRAW 0 (review V10, L-5). A consensus may publish its draw k (seed s + k),
    and then `solver_candidate` of the published solve IS draw k. Its own seed (`solve.seed`) goes
    first, and the others of the record's seeds follow in their order: draw k's seed is never mapped a
    second time, and seed s is not skipped."""
    from tower.config import WORLD_SOLVE_CONSENSUS_VALUES  # noqa: PLC0415

    gate = getattr(solution, "gate", None)
    owed = gate.get("consensus") if isinstance(gate, dict) else None
    owed = owed if isinstance(owed, dict) else {}
    raw = owed.get("requested")
    requested = int(raw) if _is_int(raw) else 1
    if requested not in WORLD_SOLVE_CONSENSUS_VALUES:
        logger.warning("[Tower][WorldBuilder] the published solve asked for a consensus of %r draws, which "
                       "is not one of %s; it is re-gated as one draw", raw,
                       ", ".join(str(v) for v in WORLD_SOLVE_CONSENSUS_VALUES))
        requested = 1
    seeds = owed.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(_is_int(s) for s in seeds):
        return requested, None
    own = (getattr(solution, "solve", None) or {}).get("seed")
    order = list(dict.fromkeys(([own] if _is_int(own) and own in seeds else []) + seeds))
    while len(order) < requested:
        order.append(max(order) + 1)
    return requested, order[:requested]


def _draw_camera_readable(solution, workspace) -> bool:
    """Whether `global_solve.frozen_draw_mapper` finds the camera it maps each draw with: the
    solution's own, else the workspace's `camera.json`. Reads only."""
    from tower.storage import read_json_closed  # noqa: PLC0415
    from tower.world_builder.global_solve import PinholeCamera  # noqa: PLC0415

    try:
        PinholeCamera.from_json_dict(solution.camera or read_json_closed(workspace.camera_path))
    except Exception:  # noqa: BLE001 -- missing, unreadable or malformed: the draws cannot be mapped
        return False
    return True


def _regate_inputs(store, world_id: str, session_id: str):
    """(the published solution, its workspace, the feature database it was solved from), or
    `RegateRefused` saying why a re-gate cannot start. Reads only. The ONE statement of the
    refusals: `regate_published` and `regate_refusal` both ask it.

    A consensus the re-gate will run (`requested` >= 2, seeded) needs the camera its draws are
    mapped with: without one it is refused here, read-only, rather than raising inside
    `regate_published` after the finisher has counted the attempt (review V9 LOW)."""
    from tower.world_builder.global_solve import load_solution, workspace_for  # noqa: PLC0415

    solution = load_solution(store, world_id, session_id)
    if solution is None or not isinstance(solution.gate, dict):
        raise RegateRefused(REFUSAL_NO_GATED_SOLUTION)
    workspace = workspace_for(store, world_id, session_id)
    name = (solution.solve or {}).get("database") or workspace.database_path.name
    database = workspace.root / name
    if not database.is_file():
        raise RegateRefused(REFUSAL_DATABASE_GONE.format(name=name))
    requested, seeds = _owed_consensus(solution)
    if requested >= 2 and seeds is not None and not _draw_camera_readable(solution, workspace):
        raise RegateRefused(REFUSAL_NO_DRAW_CAMERA)
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
    (`regate_refusal` is the same test, read-only).

    A STOP THAT REACHED DRAW 0'S DEPTH STAGE WRITES NOTHING (review V10, MED-1b). Its gate took
    the scale fail-safe -- the anchor block alone -- only because a capture started; published, it
    replaced a room the solve had already published (a `consensus-deferred` world's attached draw
    0). So nothing is written, and the result says `stopped: True` with the published record's
    own `notice` and `detail`: the row is unchanged, the re-gate is still owed, and the attempt is
    the stop's (the finisher gives back an attempt a stop ended, `forgive_attempt`)."""
    from tower.world_builder.global_solve import write_solution  # noqa: PLC0415

    solution, workspace, database = _regate_inputs(store, world_id, session_id)
    previous = {k: solution.gate.get(k) for k in ("state", "cause", "metric_available", "params_digest")}
    keyframes = store.read_keyframes(world_id, session_id)
    candidate = solver_candidate(solution)
    requested, seeds = _owed_consensus(solution)
    if requested >= 2:
        # A solve that asked for a consensus is re-gated BY consensus (review V9, M-1), whatever
        # its record says -- `deferred` by its first draw's fail-safe or by a stop, or anything
        # else: a single gate here would publish one draw where the solve asked for N. Its draws
        # are mapped now, on the database the published solve mapped; the published solve is
        # draw 0, with its own seed (L-5).
        from tower.world_builder.global_solve import frozen_draw_mapper  # noqa: PLC0415

        plan = ConsensusPlan(draws=requested, seed=None if seeds is None else seeds[0], seed_list=seeds,
                             map_draw=None if seeds is None else frozen_draw_mapper(
                                 store, world_id, session_id, database, candidate, keyframes=keyframes),
                             refusal=None if seeds is not None else "the solve was not seeded")
        result = gate_by_consensus(store, world_id, session_id, candidate, plan=plan, database_path=database,
                                   keyframes=keyframes, should_stop=should_stop, gate_runner=gate_runner)
    else:
        result = (gate_runner or gate_final_solution)(store, world_id, session_id, candidate,
                                                      database_path=database, keyframes=keyframes,
                                                      should_stop=should_stop)
    if draw_0_stopped(result):
        logger.info("[Tower][WorldBuilder] %s/%s: a stop reached the re-gate's draw-0 depth stage; nothing "
                    "is written, and the re-gate is still owed", world_id, session_id)
        kept = {"gate": solution.gate, "transients": solution.transients}
        return {"gate": {k: solution.gate.get(k) for k in ("state", "retryable", "cause", "metric_available",
                                                             "attach", "components")},
                "publish": {"written": False, "why": WHY_KEPT_ON_STOP},
                "stopped": True,
                "notice": publish_notice(kept),
                "detail": publish_detail(kept)}
    published = result.solution
    record = dict(result.record, regate={"at": time.time(), "previous": previous})
    published.gate = record
    published.timing = dict(published.timing or {}, regate_s=result.record.get("seconds"))
    write_solution(workspace, published)
    out = after_publish(store, world_id, session_id, workspace.root, published, result)
    return {"gate": {k: record.get(k) for k in ("state", "retryable", "cause", "metric_available",
                                                  "attach", "components")},
            "publish": out,
            # The row's sentence after the re-gate (None when nothing is owed any more): the
            # closed set for `finalization.notice`, and its diagnostic twin for `detail`.
            "notice": publish_notice({"gate": record, "transients": published.transients}),
            "detail": publish_detail({"gate": record, "transients": published.transients})}


# ---------------------------------------------------------------------------
# what the row says (review V7, H2 and L-c; review V9, M-4 and M-8; manager 025, decision 3)

#
# EVERY FAIL-SAFE SAYS SO (review V8, M2). A published gated solve that is not "masks
# applied and metric scale available" attaches nothing to the room, and the row says
# what is missing and WHO can fix it -- the idle Tower (a re-gate in place), an owner
# (a re-finish, or a new walk), or an operator first (a Tower that cannot run the
# masks) -- in contract §2.2's order: masks, then scale. No metric figure (§2.4 rule 6).
#
# A CLOSED SET (review V9, M-4). `finalization.notice` is shown on the phone word for word,
# and contract §3.1 says the notice never carries an error string: `<why>` is a short
# Tower-written phrase. So every sentence a notice can hold is one of `NOTICE_SENTENCES`,
# keyed by its cause, and a notice is some of them joined by "; " in §2.2's order. The raw
# text a cause was recorded with -- an exception, a path, a memory figure -- stays where it
# was recorded (`gate.detail`, `gate.depth.detail`, `transients.detail` in solution.json).
# `publish_detail`, the diagnostic twin for the finalization's `detail`, carries it only as
# `client_safe_detail` makes it: one line, no path, no traceback, short (review V10, MED-5 --
# `detail` DOES reach the phone and the unauthenticated socket, inside `lifecycle.finalization`
# and, for an interrupted session, inside `lifecycle.reason`; and the `GET /worlds` row, review
# V11, MED-B). The only numbers a sentence
# carries are IMAGE counts, which an owner can read ("12 of 400 images"); the camera counts
# behind a scale shortfall stay in the detail. Every sentence is one line, with no slash or
# backslash, no exception class name, no `key=value` pair, and far under the phone's
# 700-character guard (Mac tips de1b045 and 3bb4431: `WorldTowerText` swaps any Tower text
# that has one of those for a generic sentence; `NOTICE_SENTENCES` must never trip it).

_BY_OWNER = "an owner can re-finish this walk"
_BY_OPERATOR = ("an operator can make the transient detector run on this Tower, then an owner "
                "can re-finish this walk")
_BY_IDLE_TOWER = "the Tower re-runs it when it is idle"
_BY_IDLE_REGATE = "the Tower re-runs the gate when it is idle"
_MASKS_NOT_APPLIED = "masks were not applied ({})"
_GATE_FAILED = "the evidence gate failed ({})"
_NO_SCALE = "the evidence gate could not measure metric scale ({})"

# cause -> the short, fixed phrase that names it (`notice_cause_phrase`): what the sentence
# says in its parentheses, and what the finisher's given-up sentence says.
NOTICE_PHRASES: dict[str, str] = {
    # the masks (a `transients` record)
    "masks-gpu-oom": "GPU out of memory",
    "masks-off": "they are off on this Tower: TOWER_WORLD_SOLVE_MASKS",
    "masks-no-gpu": "no GPU could run the transient detector",
    "masks-not-installed": "the transient detector is not installed on this Tower",
    "masks-detector-failed": "the transient detector failed",
    "masks-step-failed": "the mask step failed",
    "masks-no-image": "no solver image could be masked",
    "masks-none": "no solver image could be masked",
    "masks-unavailable": "the transient detector could not run",
    "masks-fallback": "only OneFormer could run",
    "masks-partial": "some images were not masked",
    "masks-partial-uncounted": "some images were not masked",
    "masks-excluded": "some images could not be masked",
    "masks-excluded-uncounted": "some images could not be masked",
    # the gate (a `gate` record)
    "gate-failed": "an internal error",
    "gate-failed-database": "the solve's feature database could not be read",
    "gate-failed-memory": "out of memory",
    "depth-unavailable": "the depth stage did not finish",
    "depth-stopped": "the depth stage was stopped",
    "depth-gpu-oom": "GPU out of memory",
    "depth-model-missing": "the depth model is not installed on this Tower",
    "depth-surface-busy": "another surface build of this walk was running",
    "depth-no-intrinsics": "the walk has no camera intrinsics",
    "depth-no-camera": "the solve has no camera",
    "scale-short": "too few images had a metric depth",
    # Review V10, L-1: owed by a stop, by a further draw's retryable failure, or by the solve's early
    # publish of draw 0 (MED-1a) -- "did not finish" is true of each; "was stopped" was not.
    "consensus-deferred": "the consensus of mapper seeds did not finish",
}


def _clause(cause: str) -> str:
    phrase = NOTICE_PHRASES[cause]
    if cause.startswith("masks-"):
        return _MASKS_NOT_APPLIED.format(phrase)
    if cause.startswith("gate-failed"):
        return _GATE_FAILED.format(phrase)
    return _NO_SCALE.format(phrase)


# cause -> the clause that says what happened, without who fixes it: `regate_clause`, and the
# finisher's given-up sentence (`world_finish_pending.regate_given_up_notice`).
NOTICE_CLAUSES: dict[str, str] = {
    **{c: _clause(c) for c in NOTICE_PHRASES if c.startswith(("masks-", "gate-failed", "depth-"))},
    "masks-fallback": ("masks were applied by OneFormer alone, not by the union rule the evidence "
                       "gate needs"),
    "masks-partial": "masks were not applied to {unmasked} of {images} images",
    "masks-partial-uncounted": "masks were not applied to some of its images",
    "masks-excluded": "{excluded} of {images} images could not be masked and were left out of the solve",
    "masks-excluded-uncounted": "some images could not be masked and were left out of the solve",
    "scale-short": "the evidence gate had too little metric scale to place pieces by it",
    "consensus-deferred": "the evidence gate's consensus of mapper seeds did not finish",
}

_OWNERS: dict[str, str] = {
    **{c: _BY_OPERATOR for c in ("masks-no-gpu", "masks-not-installed", "masks-detector-failed",
                                 "masks-step-failed", "masks-no-image", "masks-unavailable")},
    **{c: _BY_OWNER for c in ("masks-gpu-oom", "masks-none", "masks-partial", "masks-partial-uncounted",
                              "masks-excluded", "masks-excluded-uncounted")},
    "masks-off": "an operator can turn them on, then an owner can re-finish this walk",
    "masks-fallback": ("an operator can make Grounding DINO and SAM available on this Tower, then an "
                       "owner can re-finish this walk"),
    **{c: _BY_IDLE_TOWER for c in ("gate-failed", "gate-failed-database", "gate-failed-memory",
                                   "consensus-deferred")},
    **{c: _BY_IDLE_REGATE for c in NOTICE_PHRASES if c.startswith("depth-")},
    # NO IDLE RE-RUN ALONE CAN FIX THESE (review V10, L-19b): the gate is re-run on the same walk and the
    # same solve, which still have no intrinsics or no camera, on a Tower that still has no depth model. So
    # the sentence names who can: an owner (a new walk; a re-finish, which solves again and so has a
    # camera), or an operator first -- once the model is installed, the idle re-run the record still owes
    # does fix it.
    "depth-no-intrinsics": "re-running the gate would not change this; an owner can re-capture this walk",
    "depth-no-camera": "re-running the gate would not change this; an owner can re-finish this walk",
    "depth-model-missing": ("an operator can install the depth model on this Tower, then the Tower re-runs "
                            "the gate when it is idle"),
    "scale-short": ("the depth stage ran to the end, so re-running the gate would not change this; "
                    "an owner can re-capture this walk"),
}

# THE CLOSED SET: cause -> the sentence `finalization.notice` may carry. `{unmasked}`,
# `{images}` and `{excluded}` are image counts, the only fields.
NOTICE_SENTENCES: dict[str, str] = {c: f"{NOTICE_CLAUSES[c]}; {_OWNERS[c]}" for c in NOTICE_PHRASES}

NOTICE_MASKS_OOM = NOTICE_SENTENCES["masks-gpu-oom"]
NOTICE_MASKS_OFF = NOTICE_SENTENCES["masks-off"]
NOTICE_MASKS_FALLBACK = NOTICE_SENTENCES["masks-fallback"]
NOTICE_MASKS_PARTIAL = NOTICE_SENTENCES["masks-partial"]
NOTICE_MASKS_NONE = NOTICE_SENTENCES["masks-none"]
NOTICE_MASKS_EXCLUDED = NOTICE_SENTENCES["masks-excluded"]
NOTICE_SCALE_SHORT_FIXED = NOTICE_SENTENCES["scale-short"]
NOTICE_CONSENSUS_DEFERRED = NOTICE_SENTENCES["consensus-deferred"]
# The DIAGNOSTIC templates (`publish_detail`, the finalization's `detail`): `{why}` is the raw
# text the cause was recorded with. Never in a notice.
NOTICE_MASKS_UNAVAILABLE = ("masks were not applied ({why}); an operator can make the transient "
                            "detector run on this Tower, then an owner can re-finish this walk")
NOTICE_REGATE = ("the evidence gate could not measure metric scale ({why}); "
                 "the Tower re-runs the gate when it is idle")
NOTICE_SCALE_SHORT = ("the evidence gate had too little metric scale to place pieces by it "
                      "({why}); the depth stage ran to the end, so re-running the gate would not "
                      "change this; an owner can re-capture this walk")
NOTICE_GATE_FAILED = "the evidence gate failed ({why}); the Tower re-runs it when it is idle"

_GATE_CAUSES = (CAUSE_DEPTH_UNAVAILABLE, CAUSE_GATE_FAILED, CAUSE_CONSENSUS_DEFERRED)


def _dict(value) -> dict:
    """A record's sub-record, or {} when it is absent or not a dict (review V10, L-19d: a malformed
    `gate.depth` made `publish_notice` raise)."""
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# raw text made client-safe (review V10, MED-5)
#
# `finalization.detail` reaches the phone and the unauthenticated `/ws` socket (inside
# `lifecycle.finalization`, and inside `lifecycle.reason` for an interrupted session), and the
# unauthenticated `GET /worlds` listing (the session row's `finalization`, with its `notice`:
# `client_safe_finalization`, review V11, MED-B), so the raw
# text it quotes is made client-safe the way `logging_config.client_safe_reason` makes an exception
# client-safe, but for TEXT, which is what a record holds: no path (and so no user name: the paths
# that carry one are home directories), no traceback, one line. The full text stays in the
# Tower's log and in solution.json's own records (`gate.detail`, `gate.depth.detail`,
# `transients.detail`), which are Tower-internal. Text with none of those comes back unchanged, so
# an old record whose detail is already clean renders byte for byte as before.

# A `why` in `publish_detail` is the exception class and a SHORT message. OPEN: a presentation
# bound, not a measured one -- torch's CUDA out-of-memory message runs to about 500 characters,
# and its first sentence, the part an operator acts on, to about 100.
WHY_MAX_CHARS = 200
PATH_PLACEHOLDER = "[path]"
_TRACEBACK = "Traceback (most recent call last)"
# An exception class: a CamelCase name ending as Python's own do, optionally dotted (`torch.OutOfMemoryError`).
_EXCEPTION_NAME = re.compile(r"\b(?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception|Interrupt|Exit|Warning)\b")
# A CamelCase name directly followed by ": " -- how `f"{type(exc).__name__}: {exc}"` begins, whatever the
# suffix (`DepthModelUnavailable: ...`).
_CLASS_PREFIX = re.compile(r"\b(?:[A-Za-z_]\w*\.)*[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+:\s")
_WIN_ERROR = re.compile(r"\[WinError -?\d+\]\s*")
_ERRNO = re.compile(r"\[Errno -?\d+\]\s*")
# Line breaks other than CR and LF (review V11, LOW-16): VT, FF, the file, group and record separators, NEL, and
# the Unicode line and paragraph separators. Each is a space in a client-safe text.
_OTHER_LINE_BREAKS = str.maketrans({c: " " for c in "\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"})
# A traceback frame (`File "C:\x.py", line 3, in f`), quoted path and all (review V11, LOW-16).
_FRAME = re.compile(r"""File\s+(["'])[^"'\r\n]*\1,\s*line\s+\d+(?:,\s*in\s+[^\s,;]+)?""")
_QUOTED_PATH = re.compile(r"""(['"])(?:[A-Za-z]:[\\/]|\\\\|~[\w.-]*[\\/]|/)[^'"\r\n]*\1""")
_UNQUOTED_PATHS = (
    # a drive path (C:\... or C:/...), not the scheme of a URL
    re.compile(r"""(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s'"]*"""),
    # a home path: ~/x, ~user/x
    re.compile(r"""(?<![\w.-])~[\w.-]*[\\/][^\s'"]*"""),
    # an absolute POSIX path of two or more parts (not a URL's: its slashes follow ':' or '/' or a host)
    re.compile(r"""(?<![\w:/.-])/(?:[\w.@+-]+/)+[\w.@+-]*"""),
    # a RELATIVE POSIX path (review V11, LOW-16): two or more separators (`data/worlds/w/points.json`), or one
    # before a file name with an extension (`images/00000042.jpg`) -- never a fraction (`3/4`), a model id
    # (`IDEA-Research/grounding-dino-base`), or a part of a URL or of an absolute path (what precedes it)
    re.compile(r"""(?<![\w.@+:/~\\-])(?:[\w.@+-]+/){2,}[\w.@+-]*"""),
    re.compile(r"""(?<![\w.@+:/~\\-])[\w.@+-]+/[\w@+-][\w.@+-]*\.[A-Za-z][A-Za-z0-9]{0,7}(?![\w/])"""),
    # anything else with a backslash in it: a UNC path, a relative Windows path
    re.compile(r"""[^\s'"]*\\[^\s'"]*"""),
)
# The words after a space that are still the same path: each holds a separator (`C:\Program Files\x`), or is a
# file name with an extension (`C:\Users\x\my file.txt`, review V11, LOW-16).
_PATH_TAIL = re.compile(r"""(?:\s+(?:[^\s'"]*[\\/][^\s'"]*"""
                        r"""|[^\s'"\\/]*\.[A-Za-z][A-Za-z0-9]{0,7}(?=$|[\s)\]},;:'"]|\.(?:$|\s))))*""")
# A path that ends at a user's own directory (`C:\Users\John`, `/home/John`) and the capitalised words after it,
# which are the rest of a spaced name (`C:\Users\John Smith`, review V11, LOW-16).
_USER_DIR_END = re.compile(r"""(?:^|[\\/])(?:users|home)[\\/][^\\/]+$""", re.IGNORECASE)
_NAME_WORDS = re.compile(r"""(?:\s+[A-Z][^\s'"\\/]*)+""")
# Sentence punctuation a path match takes with it (`...\x.py);`): given back, unless it closes a bracket the
# path itself opened (`C:\Program Files (x86)`).
_CLOSING = {")": "(", "]": "[", "}": "{"}
_TRAILING = ")]},;.:"


USER_PLACEHOLDER = "[user]"
# What `owner_facing_detail` says in their place: plain words, no brackets (review V11, LOW-17).
OWNER_PATH_WORDS = "a path"
OWNER_USER_WORDS = "a user name"


def _user_names() -> set[str]:
    """This machine's user names -- the account's and its home directory's -- for the one place a user name can
    outlive the path scrub: a message that names the user outside a path. Names under 3 characters are left
    alone (they would match ordinary words)."""
    import getpass  # noqa: PLC0415

    names = set()
    for read in (getpass.getuser, lambda: Path.home().name):
        try:
            name = str(read() or "").strip()
        except Exception:  # noqa: BLE001 -- no user name is readable: nothing to scrub
            continue
        if len(name) >= 3:
            names.add(name)
    return names


def _exception_lines(text: str) -> list[str]:
    """The non-empty lines of a traceback that are not its frames: a `File "...", line N` line, and the
    indented source and caret lines under it, are left out (review V11, LOW-16), and so is a line of
    punctuation alone (the header's own colon)."""
    out, under_frame = [], False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _FRAME.match(line):
            under_frame = True
            continue
        if under_frame and raw[:1].isspace():
            continue
        under_frame = False
        if any(ch.isalnum() for ch in line):
            out.append(line)
    return out


def _one_line(text: str) -> str:
    """One line of `text`: a traceback -- with its header, or frames without one -- becomes the text
    before it and its last line that is not a frame (the exception itself); other multi-line text its
    first line."""
    at = text.find(_TRACEBACK)
    framed = at < 0 and any(_FRAME.match(ln.strip()) for ln in text.splitlines())
    if at >= 0 or framed:
        head = " ".join(text[:at].split()).rstrip(" :;,-(") if at >= 0 else ""
        rest = text[at + len(_TRACEBACK):] if at >= 0 else text
        lines = [ln.strip() for ln in rest.splitlines() if ln.strip()]
        if len(lines) > 1:
            kept = _exception_lines(rest)
            last = kept[-1] if kept else ""
        else:
            hits = list(_EXCEPTION_NAME.finditer(rest))
            last = rest[hits[-1].start():].strip() if hits else ""
        return ": ".join(part for part in (head, last) if part) or "a traceback"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def _without_frames(text: str) -> str:
    """`text` without the traceback frames left in one line of it (review V11, LOW-16)."""
    if not _FRAME.search(text):
        return text
    out = re.sub(r"\s{2,}", " ", _FRAME.sub("", text)).strip(" ,;:")
    return out or "a traceback"


def _given_back(path: str) -> tuple[str, str]:
    """`path` without the sentence punctuation its match took with it, and that punctuation: a closing
    bracket the path did not open, and `;`, `,`, `.` and `:` at its end (review V11, LOW-16)."""
    end = len(path)
    while end > 1 and path[end - 1] in _TRAILING:
        opener = _CLOSING.get(path[end - 1])
        if opener is not None and path[:end].count(opener) >= path[:end].count(path[end - 1]):
            break
        end -= 1
    return path[:end], path[end:]


def _scrub_paths(text: str) -> str:
    """Every path in `text` as `PATH_PLACEHOLDER`: quoted ones with their quotes kept; unquoted ones with
    the words after a space that are still the path (`_PATH_TAIL`, `_NAME_WORDS`), and without the
    sentence punctuation after them (`_given_back`)."""
    text = _QUOTED_PATH.sub(r"\1" + PATH_PLACEHOLDER + r"\1", text)
    for pattern in _UNQUOTED_PATHS:
        out, at = [], 0
        for m in pattern.finditer(text):
            if m.start() < at:
                continue
            end = m.end()
            while True:
                grown = _PATH_TAIL.match(text, end).end()
                if _USER_DIR_END.search(text, m.start(), grown):
                    name = _NAME_WORDS.match(text, grown)
                    grown = name.end() if name else grown
                if grown == end:
                    break
                end = grown
            back = _given_back(text[m.start():end])[1]
            out.append(text[at:m.start()] + PATH_PLACEHOLDER + back)
            at = end
        out.append(text[at:])
        text = "".join(out)
    return text


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max(1, max_chars - 3)]
    if " " in cut[max_chars // 2:]:
        cut = cut[:cut.rindex(" ")]
    return cut.rstrip(" ,;:(") + "..."


def client_safe_detail(text, *, max_chars: int | None = None) -> str:
    """Raw diagnostic text made safe for a client (review V10, MED-5): ONE line (a traceback is
    reduced to the text before it and its last line, the exception; other multi-line text to its
    first line; every other line break is a space, and no traceback frame is left), NO PATH (drive,
    UNC, home, absolute and relative POSIX paths, quoted or not, become `PATH_PLACEHOLDER`, and with
    them the user names they carry), NO USER NAME left anywhere else (this machine's,
    `USER_PLACEHOLDER`), and, with `max_chars`, at most that long. The exception class and its message
    stay: this is the diagnostic text. "" for no text. Single-line text with none of those, at most
    `max_chars` long, is returned unchanged."""
    if text is None:
        return ""
    text = str(text).translate(_OTHER_LINE_BREAKS)
    if "\n" in text or "\r" in text or _TRACEBACK in text:
        text = _one_line(text)
    text = _without_frames(text)
    text = _scrub_paths(text)
    for name in _user_names():
        text = re.sub(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", USER_PLACEHOLDER, text, flags=re.IGNORECASE)
    if max_chars is not None:
        text = _shorten(text, int(max_chars))
    return text


def owner_facing_detail(text) -> str:
    """`client_safe_detail`, then without the exception class names -- for a sentence an OWNER reads
    (`lifecycle.reason`, the phone's `model_state_reason`; review V10, MED-5, and the lead's
    preference given the Mac's `WorldTowerText` guard of 3bb4431, which swaps any Tower text holding a
    class name for a generic sentence). "RuntimeError: CUDA out of memory" reads "CUDA out of
    memory". And WITHOUT SQUARE BRACKETS (review V11, LOW-17: that guard reads brackets as JSON): a
    path reads "a path" and a user name "a user name", `[Errno N]` goes the way `[WinError N]` does,
    and no "(: " is left where a class name was. Text with no path, traceback, line break, class name
    or bracket is returned unchanged."""
    safe = client_safe_detail(text)
    owner = _WIN_ERROR.sub("", safe)
    owner = _ERRNO.sub("", owner)
    owner = _CLASS_PREFIX.sub("", owner)
    owner = _EXCEPTION_NAME.sub("", owner)
    owner = re.sub(r"""(['"]?)""" + re.escape(PATH_PLACEHOLDER) + r"\1", OWNER_PATH_WORDS, owner)
    owner = re.sub(r"""(['"]?)""" + re.escape(USER_PLACEHOLDER) + r"\1", OWNER_USER_WORDS, owner)
    owner = owner.replace("[", "").replace("]", "")
    if owner == safe:
        return safe
    owner = re.sub(r"\(\s*\)", "", owner)          # "(KeyboardInterrupt)" leaves "()"
    owner = re.sub(r"\(\s*:\s*", "(", owner)       # "(OSError: x" leaves "(: x"
    owner = re.sub(r":\s+:", ":", owner)           # "failed: EOFError: x" leaves "failed: : x"
    owner = re.sub(r"\(\s+", "(", owner)
    owner = re.sub(r"\s+([);,])", r"\1", owner)
    owner = re.sub(r"\s{2,}", " ", owner)
    return owner.strip(" :;,")


# The phone's text guard (`WorldTowerText`, Mac tips de1b045 and 3bb4431) replaces any Tower text longer than
# this with a generic sentence. The listing row's `finalization.detail` and `.notice` are bounded to it (review
# V11, MED-B); every closed-set notice is far shorter (`test_world_builder_notice_closed_set`).
FINALIZATION_TEXT_MAX_CHARS = 700


def client_safe_finalization(finalization):
    """A session's `finalization` record as a client is sent it (review V11, MED-B: the `GET /worlds` row
    sent it raw): a COPY whose `detail` and `notice` are `client_safe_detail` of the record's, at most
    `FINALIZATION_TEXT_MAX_CHARS` long. The record on disk is not touched. A record whose texts have
    nothing to scrub (every clean detail, every closed-set notice) is returned as it is, the same object,
    so it is sent byte for byte as before; so is anything that is not a dict, and a text that is not a
    string."""
    if not isinstance(finalization, dict):
        return finalization
    safe = {}
    for key in ("detail", "notice"):
        value = finalization.get(key)
        if isinstance(value, str):
            scrubbed = client_safe_detail(value, max_chars=FINALIZATION_TEXT_MAX_CHARS)
            if scrubbed != value:
                safe[key] = scrubbed
    return dict(finalization, **safe) if safe else finalization


def _count(value) -> int | None:
    """An image count as the record holds it, or None when it is not one."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _is_oom(low: str) -> bool:
    return "out of memory" in low or "outofmemory" in low


def _masks_cause_unavailable(transients: dict) -> str:
    """Which fixed phrase names a masks record that is not `partial`: by its code first
    (`cause`, `outcome`), then by the Tower's own wording of the few details it writes."""
    low = str(transients.get("detail") or "").strip().lower()
    cause = transients.get("cause")
    if cause == "gpu-oom":
        return "masks-gpu-oom"
    if low == "no solver image could be masked":
        # Written before `none_masked` existed (review V9, M-8): absent keys, today's sentence.
        return "masks-no-image"
    if cause == "detector-failed":
        return "masks-detector-failed"
    if transients.get("outcome") == "failed":
        return "masks-step-failed"         # the mask step itself raised (`failed_record`)
    if _is_oom(low):
        return "masks-gpu-oom"
    if "no cuda device" in low or "cuda probe failed" in low:
        return "masks-no-gpu"
    if any(s in low for s in ("not installed", "not importable", "cannot load", "hugging face cache")):
        return "masks-not-installed"
    return "masks-unavailable"


def _masks_cause(transients: dict, gate: dict) -> str | None:
    """The masks fail-safe's cause (a key of `NOTICE_SENTENCES`), or None. The GPU-out-of-memory
    one is said whatever the gate did (as before); every other only when the gate ran without
    masks: applied with `masks_applied` False, or FAILED on a solve whose masks were not applied
    (review V9 LOW: a failed gate used to drop the masks cause)."""
    if transients.get("retryable") and transients.get("cause") == "gpu-oom":
        return "masks-gpu-oom"
    if gate.get("state") == GATE_STATE_APPLIED:
        if gate.get("masks_applied") is not False:
            return None
    elif gate.get("state") == GATE_STATE_FAILED:
        if not transients or transients.get("state") == "applied":
            return None
    else:
        return None
    state = transients.get("state")
    if not transients or transients.get("requested") is False:
        return "masks-off"
    if state == "applied":
        return None                 # an inconsistent record: nothing true to say
    if transients.get("none_masked"):
        # No image could be masked (review V9, M-8): the images' owner, not the detector's.
        return "masks-none"
    if state == "partial":
        if transients.get("rule_fallback"):
            return "masks-fallback"
        if _count(transients.get("images_unmasked")) is not None and _count(transients.get("images")):
            return "masks-partial"
        return "masks-partial-uncounted"
    return _masks_cause_unavailable(transients)


def _exclusion_cause(transients: dict, gate: dict) -> str | None:
    """Images left out of the solve because they could not be masked, once there are enough of
    them to say so (`transients.exclusion_notice_due`, review V9 M-8: at least max(3, 2 %) of the
    images; the masks stage decides it). Absent: today's behaviour, nothing said. Only for a
    solve the gate ran on, like every masks sentence but the OOM one."""
    if not transients.get("exclusion_notice_due") or transients.get("none_masked"):
        return None
    if gate.get("state") not in (GATE_STATE_APPLIED, GATE_STATE_FAILED):
        return None
    if _count(transients.get("images_excluded")) and _count(transients.get("images")):
        return "masks-excluded"
    return "masks-excluded-uncounted"


def _gate_failed_cause(detail) -> str:
    low = str(detail or "").lower()
    if any(s in low for s in ("databaseerror", "operationalerror", "not a database", "sqlite")):
        return "gate-failed-database"
    if "memoryerror" in low or _is_oom(low):
        return "gate-failed-memory"
    return "gate-failed"


def _depth_cause(detail) -> str:
    """Which fixed phrase names a depth stage that did not give the gate its depth
    (`run_gate_depth`'s `DepthUnavailable` text, kept in `gate.depth.detail`)."""
    text = str(detail or "").strip()
    low = text.lower()
    if not text:
        return "depth-unavailable"
    if low.startswith(DEPTH_STOPPED):
        return "depth-stopped"
    if "no intrinsics" in low:
        return "depth-no-intrinsics"
    if "has no camera" in low:
        return "depth-no-camera"
    if "holds its lock" in low:
        return "depth-surface-busy"
    if _is_oom(low):
        return "depth-gpu-oom"
    if any(s in low for s in ("depthmodelunavailable", "not installed", "hugging face cache",
                              "modulenotfounderror", "importerror")):
        return "depth-model-missing"
    return "depth-unavailable"


def _gate_cause(gate: dict) -> str | None:
    """The gate's own cause (a key of `NOTICE_SENTENCES`), or None when it owes nothing and took
    no fail-safe a sentence is for."""
    if gate.get("state") == GATE_STATE_FAILED:
        return _gate_failed_cause(gate.get("detail"))
    if gate.get("retryable"):
        if gate.get("cause") == CAUSE_CONSENSUS_DEFERRED:
            return "consensus-deferred"
        return _depth_cause(_dict(gate.get("depth")).get("detail"))
    no_rerun = _depth_cause_not_retryable(gate)
    if no_rerun:
        return no_rerun
    if _scale_short_with_depth(gate):
        return "scale-short"
    return None


def _depth_cause_not_retryable(gate: dict) -> str | None:
    """The cause of a scale fail-safe whose depth stage could not start for the walk's or the solve's own
    reason (`DEPTH_CAUSES_NOT_RETRYABLE`), which the gate records NOT retryable (review V11, LOW-15), or None.
    Its notice is the one the same record said while it was retryable: exactly the records that were."""
    if gate.get("state") != GATE_STATE_APPLIED or gate.get("retryable") or not gate.get("masks_applied"):
        return None
    depth = _dict(gate.get("depth"))
    if depth.get("state") != DEPTH_UNAVAILABLE:
        return None
    cause = _depth_cause(depth.get("detail"))
    return cause if cause in DEPTH_CAUSES_NOT_RETRYABLE else None


def _scale_short_with_depth(gate: dict) -> bool:
    """The gate ran with its depth stage in hand and still had no usable metric scale."""
    return (gate.get("state") == GATE_STATE_APPLIED and gate.get("metric_available") is False
            and not gate.get("retryable")
            and _dict(gate.get("depth")).get("state") == DEPTH_OK)


def notice_causes(summary: dict | None) -> list[str]:
    """The causes the notice names, in contract §2.2's order (masks, then scale): keys of
    `NOTICE_SENTENCES`. Empty when nothing is owed and no fail-safe was taken."""
    if not isinstance(summary, dict):
        return []
    transients = summary.get("transients") or {}
    gate = summary.get("gate") or {}
    if not isinstance(transients, dict) or not isinstance(gate, dict):
        return []
    causes = [c for c in (_masks_cause(transients, gate), _exclusion_cause(transients, gate),
                          _gate_cause(gate)) if c]
    return causes


def _sentence(cause: str, transients: dict) -> str:
    return NOTICE_SENTENCES[cause].format(
        unmasked=_count(transients.get("images_unmasked")),
        images=_count(transients.get("images")),
        excluded=_count(transients.get("images_excluded")))


def publish_notice(summary: dict | None) -> str | None:
    """The session's `finalization.notice` (the phone shows it word for word), when the published
    solve owes something an owner, an operator or the idle Tower will do, or took a fail-safe
    nobody can undo but a new walk, else None. Only sentences of the closed set
    `NOTICE_SENTENCES`: no exception text, path or figure but an image count (review V9, M-4)."""
    causes = notice_causes(summary)
    if not causes:
        return None
    transients = (summary or {}).get("transients") or {}
    return "; ".join(_sentence(c, transients) for c in causes)


def publish_detail(summary: dict | None) -> str | None:
    """The same sentences as `publish_notice`, with the text each cause was recorded with, made
    client-safe (`client_safe_detail`), in place of its fixed phrase: the DIAGNOSTIC twin, for the
    finalization's `detail`. That field DOES reach clients -- the `GET /worlds` row and `/ws`
    `lifecycle.finalization` send it, and `lifecycle.reason`, which the phone shows on an interrupted
    session, quotes it owner-facing (`owner_facing_detail`; review V10, MED-5; V11, MED-B). What
    `publish_notice` said before review V9, M-4. None exactly when `publish_notice` is None."""
    causes = notice_causes(summary)
    if not causes:
        return None
    transients = _dict((summary or {}).get("transients"))
    gate = _dict((summary or {}).get("gate"))

    def why(*values, default: str) -> str:
        # The raw text, CLIENT-SAFE (review V10, MED-5): one line, no path, no traceback, short. It
        # keeps the exception class and its message -- the diagnostics -- and nothing that names a
        # file, a directory or the user whose home it is.
        raw = next((v for v in values if v), None)
        return client_safe_detail(raw, max_chars=WHY_MAX_CHARS) or default if raw else default

    parts = []
    for c in causes:
        if c in ("masks-no-gpu", "masks-not-installed", "masks-detector-failed", "masks-step-failed",
                 "masks-no-image", "masks-unavailable"):
            parts.append(NOTICE_MASKS_UNAVAILABLE.format(
                why=why(transients.get("detail"), transients.get("cause"), default="the detector did not run")))
        elif c == "masks-none" and transients.get("detail"):
            parts.append(f"{_MASKS_NOT_APPLIED.format(why(transients.get('detail'), default='no image'))}; "
                         f"{_BY_OWNER}")
        elif c.startswith("gate-failed"):
            parts.append(NOTICE_GATE_FAILED.format(why=why(gate.get("detail"), default="an error")))
        elif c.startswith("depth-"):
            # Who fixes it is the cause's (review V10, L-19b): the idle re-gate, or an owner or operator.
            text = why(_dict(gate.get("depth")).get("detail"), gate.get("cause"), default="no depth")
            parts.append(f"{_NO_SCALE.format(text)}; {_OWNERS[c]}")
        elif c == "scale-short":
            parts.append(NOTICE_SCALE_SHORT.format(
                why=why(_dict(gate.get("evidence")).get("metric_scale"),
                        default="too few cameras had a metric level")))
        else:
            parts.append(_sentence(c, transients))
    return "; ".join(parts)


def masks_notice(transients: dict | None, gate: dict | None) -> str | None:
    """The masks sentences alone (the masks fail-safe, then the excluded images), exactly as
    `publish_notice` words them for the same records, or None: for the finisher's given-up
    sentence, which keeps them and replaces the gate's."""
    t = transients if isinstance(transients, dict) else {}
    g = gate if isinstance(gate, dict) else {}
    causes = [c for c in (_masks_cause(t, g), _exclusion_cause(t, g)) if c]
    return "; ".join(_sentence(c, t) for c in causes) or None


def _is_gate_record(record: dict) -> bool:
    return (record.get("state") == GATE_STATE_FAILED or record.get("cause") in _GATE_CAUSES
            or any(k in record for k in ("gate", "depth", "evidence", "masks_applied", "metric_available",
                                         "params_digest", "consensus")))


def notice_cause_phrase(record: dict | None) -> str:
    """The fixed phrase (`NOTICE_PHRASES`) that names the cause of a GATE record (`solution.gate`)
    or a TRANSIENTS record (`solution.transients`): never the record's raw text. For the
    finisher's given-up sentence (review V9, M-4). A gate record that failed, or owes a re-gate,
    always has one; a record with nothing to say gives "" (a caller asks only about a cause)."""
    r = record if isinstance(record, dict) else {}
    if not r:
        return ""
    if _is_gate_record(r):
        cause = _gate_cause(r)
    else:
        cause = (_masks_cause(r, {"state": GATE_STATE_APPLIED, "masks_applied": False})
                 or _exclusion_cause(r, {"state": GATE_STATE_APPLIED}))
    return NOTICE_PHRASES.get(cause, "") if cause else ""


def regate_clause(gate: dict | None) -> str:
    """What happened to a gate that owes a re-gate, as a fixed clause of `NOTICE_CLAUSES` --
    "the evidence gate failed (...)", "... could not measure metric scale (...)", or the consensus
    a stop deferred -- for the finisher's given-up sentence, which puts its own ending after it."""
    g = gate if isinstance(gate, dict) else {}
    cause = _gate_cause(g) or ("gate-failed" if g.get("state") == GATE_STATE_FAILED else "depth-unavailable")
    return NOTICE_CLAUSES[cause]
