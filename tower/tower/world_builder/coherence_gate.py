"""The coherence gate: attach a piece of the final solve to the room only on independent, consistent evidence.

WHY. The target walk (6e6d3fc3) published a bathroom and a bed block glued into the room by feature matches
on the phone in the wearer's hand, at a wrong tilt, scale and position. With the hand/arm/held-phone masks
the glue is gone, and what remains is a set of pieces the solver may still have joined on weak evidence.
This gate decides, per piece, whether the images fix its placement relative to the room; a piece they do
not fix is published as its own component -- an "area", never drawn inside the room
(`docs/contracts/WORLD-BUILDER-COMPONENTS.md`).

THE RULE (generic: no room, landmark or keyframe index enters it). Per solver component:

  (a) CANDIDATE GROUPS are the biconnected blocks of the solver's VERIFIED view graph over the component's
      supported cameras (>= MIN_IMAGE_OBSERVATIONS observations). A link is a verified two-view geometry with
      >= `min_link_inliers` inliers (COLMAP's floor) THAT THE SOLVE HONOURS: its own two-view rotation is
      within `max_link_disagreement_deg` of the solve's relative rotation. A link the solve contradicts is not
      evidence that the solve placed the pair right (6839fb8f kf 92-98, misplaced 88-113 deg, was held by four
      UNCALIBRATED 15-18-inlier links the solve contradicted by 34-96 deg).
  (b) METRIC SCALE: each group is split in capture order where its per-camera metric level
      (log(z_sfm / z_metric)) steps by more than `scale_step_factor` with >= `scale_min_cameras` cameras on
      each side; segments at one level re-join. One similarity cannot hold a x4 scale step.
  (c) ATTACHMENT, peeled from the largest group: a group joins the kept set only if its honoured links to it
      are REDUNDANT (two vertex-disjoint pairs, or two pairs through one image closing a triangle) and its
      metric level agrees with the kept set's.
  (d) MASKS AND METRIC SCALE ARE HARD DEPENDENCIES. If the solve ran without its transient masks
      (`masks_applied=False`), no group is attached at all: every piece outside each component's anchor block is left unplaced with
      the reason `masks-unavailable` (manager 011). On the run's unmasked arms this fail-safe left 16
      misplaced and 0 scale-misplaced keyframes attached, against 103 and 10 for the rule with its old
      seed-stability test. Metric scale likewise: with no camera's ratio available the gate attaches
      nothing (reason `scale-unavailable`) -- without its scale tests the rule left 947 misplaced keyframes
      attached over the run's 24 world-arms, against 133 with them.

There is no seed-stability test: the product runs ONE seeded single-thread solve.

THRESHOLDS (evidence: run wb-coherence-run-2026-09-23, experiments P2-GT and P2-LM/gatefix):
`min_link_inliers` 15 is COLMAP's verification floor. `scale_step_factor` 1.25 and `scale_min_cameras` 10
are the harness's physical plausibility bounds and each the leave-one-world-out choice in 6/7 folds.
`max_link_disagreement_deg` 16.8 is the KNOWN-GOOD CONTROL's p90 of link-vs-solve rotation disagreement over
all 5,186 of its links, CHOSEN DELIBERATELY over its p95 (25.2): both came from the control alone and were
declared before any result, and the costs are asymmetric -- a false attachment is confidently wrong
geometry, a false split is coverage shown honestly as an area -- so the stricter value is the default.
Scored with independent ground truth on the masked arms of 7 worlds: 30 -> 24 misplaced keyframes left
attached, 0 correctly placed keyframes split off.

numpy / scipy; `read_link_rotations` uses pycolmap. The gate itself does no IO.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

GATE_ID = "coherence-gate:evidence+honoured-links/1"

# The contract's reasons (WORLD-BUILDER-COMPONENTS.md section 2.2), in precedence order.
REASON_MASKS_UNAVAILABLE = "masks-unavailable"
REASON_SOLVED_SEPARATELY = "solved-separately"
REASON_NO_VERIFIED_LINK = "no-verified-link"
REASON_SINGLE_UNCONFIRMED_LINK = "single-unconfirmed-link"
REASON_SCALE_MISMATCH = "scale-mismatch"
REASON_LINK_CONTRADICTED = "link-contradicted"
REASON_SCALE_UNAVAILABLE = "scale-unavailable"
# Precedence (review V7, M2; contract v5 §2.2): the link reasons together, then the level.
REASONS = (REASON_MASKS_UNAVAILABLE, REASON_SCALE_UNAVAILABLE, REASON_SOLVED_SEPARATELY, REASON_NO_VERIFIED_LINK,
           REASON_SINGLE_UNCONFIRMED_LINK, REASON_LINK_CONTRADICTED, REASON_SCALE_MISMATCH)

# COLMAP TwoViewGeometry configurations that are not verified evidence.
NOT_VERIFIED_CONFIGS = (0, 1)  # UNDEFINED, DEGENERATE
_COLMAP_PAIR_BASE = 2147483647


@dataclass(frozen=True)
class GateParams:
    # global_solve.MIN_IMAGE_OBSERVATIONS: fewer observations is not a pose.
    min_obs: int = 30
    # A verified pair counts as a link at COLMAP's verification floor.
    min_link_inliers: int = 15
    # The control's p90 of link-vs-solve disagreement; see the module docstring for why p90 and not p95.
    max_link_disagreement_deg: float = 16.8
    # A metric scale step beyond this factor splits; a mismatch beyond it refuses attachment.
    scale_step_factor: float = 1.25
    # Cameras with a metric ratio needed on EACH side of a scale comparison.
    scale_min_cameras: int = 10
    # METRIC SCALE COUNTS AS AVAILABLE only when at least this fraction of the supported
    # cameras has a finite ratio (review V7, H1a). Below it the gate takes its fail-safe
    # (`scale-unavailable`) rather than splitting and attaching on the few levels it has.
    # OPEN: 0.5 is a majority rule, not a measured threshold.
    min_metric_fraction: float = 0.5

    def to_json(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        doc = {"gate": GATE_ID, **self.to_json()}
        return hashlib.sha1(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class SolveModel:
    """The final solve, reduced to what the gate needs. Cameras are indexed 0..n-1 and named by their solver
    image names, which must sort in capture order. `component` is the solver model index (0 = largest)."""

    names: list[str]
    component: np.ndarray            # (n,) int
    R_cw: np.ndarray                 # (n, 3, 3)
    n_obs: np.ndarray                # (n,) int
    obs_image: np.ndarray            # (M,) int
    obs_point: np.ndarray            # (M,) int
    n_points: int
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.names)

    def index(self) -> dict[str, int]:
        return {nm: i for i, nm in enumerate(self.names)}


# ---------------------------------------------------------------------------
# inputs (read-only IO)


def read_verified_links(database_path, min_inliers: int = 15, exclude_configs=NOT_VERIFIED_CONFIGS) -> dict:
    """{(name_a, name_b) sorted: inliers} for every verified two-view geometry of a COLMAP database.
    Opened read-only and immutable, so a published database is never touched."""
    import sqlite3
    from pathlib import Path

    uri = Path(database_path).resolve().as_uri() + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        names = dict(con.execute("select image_id, name from images").fetchall())
        out = {}
        for pid, rows, config in con.execute("select pair_id, rows, config from two_view_geometries"):
            if config in exclude_configs or rows < min_inliers:
                continue
            b = int(pid) % _COLMAP_PAIR_BASE
            a = (int(pid) - b) // _COLMAP_PAIR_BASE
            if a in names and b in names:
                out[tuple(sorted((names[a], names[b])))] = int(rows)
        return out
    finally:
        con.close()


def read_link_rotations(database_path, camera: dict, min_inliers: int = 15,
                        exclude_configs=NOT_VERIFIED_CONFIGS) -> dict:
    """{(name_a, name_b): R_b_from_a} for every verified pair: COLMAP's own relative pose for the pair's
    stored geometry and inliers (`pycolmap.estimate_two_view_geometry_pose`, which decomposes E or H by the
    pair's configuration). `camera` is the single PINHOLE camera {fx, fy, cx, cy, width, height}. Pairs
    whose pose cannot be recovered are absent -- and so count as not honoured."""
    import sqlite3
    from pathlib import Path

    import pycolmap

    cam = pycolmap.Camera(model="PINHOLE", width=int(camera["width"]), height=int(camera["height"]),
                          params=[camera["fx"], camera["fy"], camera["cx"], camera["cy"]])
    uri = Path(database_path).resolve().as_uri() + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        names = dict(con.execute("select image_id, name from images").fetchall())
        kp: dict = {}

        def keypoints(i):
            if i not in kp:
                r, c, d = con.execute("select rows, cols, data from keypoints where image_id=?", (i,)).fetchone()
                kp[i] = np.frombuffer(d, np.float32).reshape(r, c)[:, :2].astype(np.float64)
            return kp[i]

        out = {}
        for pid, rows, config, data, F, E, H in con.execute(
                "select pair_id, rows, config, data, F, E, H from two_view_geometries"):
            if config in exclude_configs or rows < min_inliers or data is None:
                continue
            b = int(pid) % _COLMAP_PAIR_BASE
            a = (int(pid) - b) // _COLMAP_PAIR_BASE
            if a not in names or b not in names:
                continue
            m = np.frombuffer(data, np.uint32).reshape(rows, 2)
            tvg = pycolmap.TwoViewGeometry()
            tvg.config = pycolmap.TwoViewGeometryConfiguration(int(config))
            for attr, blob in (("F", F), ("E", E), ("H", H)):
                if blob is not None:
                    setattr(tvg, attr, np.frombuffer(blob, np.float64).reshape(3, 3))
            tvg.inlier_matches = np.arange(rows, dtype=np.uint32).repeat(2).reshape(rows, 2)
            if pycolmap.estimate_two_view_geometry_pose(cam, keypoints(a)[m[:, 0]], cam, keypoints(b)[m[:, 1]], tvg):
                out[(names[a], names[b])] = np.asarray(tvg.cam2_from_cam1.rotation.matrix())
        return out
    finally:
        con.close()


# ---------------------------------------------------------------------------
# graph and scale helpers (ported unchanged from the experiment gate, coherence_exp/gate.py)


def _shared_counts(model: SolveModel):
    """Camera x camera shared-point counts (scipy.sparse CSR)."""
    from scipy.sparse import csr_matrix

    data = np.ones(len(model.obs_image), dtype=np.int32)
    M = csr_matrix((data, (model.obs_image.astype(np.int64), model.obs_point.astype(np.int64))),
                   shape=(model.n, max(int(model.n_points), 1)))
    M.data[:] = 1
    M.sum_duplicates()
    M.data[:] = 1
    return (M @ M.T).tocsr()


def biconnected_blocks(adj: list[set]) -> list[set]:
    """Biconnected blocks (Hopcroft-Tarjan, iterative). Blocks share at most one (articulation) vertex; an
    isolated vertex is a block of its own."""
    n = len(adj)
    disc = [-1] * n
    low = [0] * n
    t = 0
    blocks: list[set] = []
    edges: list = []
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = t
        t += 1
        if not adj[root]:
            blocks.append({root})
            continue
        stack = [(root, -1, iter(sorted(adj[root])))]
        while stack:
            v, parent, it = stack[-1]
            advanced = False
            for w in it:
                if disc[w] == -1:
                    edges.append((v, w))
                    disc[w] = low[w] = t
                    t += 1
                    stack.append((w, v, iter(sorted(adj[w]))))
                    advanced = True
                    break
                if w != parent and disc[w] < disc[v]:
                    edges.append((v, w))
                    low[v] = min(low[v], disc[w])
            if advanced:
                continue
            stack.pop()
            if stack:
                u = stack[-1][0]
                low[u] = min(low[u], low[v])
                if low[v] >= disc[u]:
                    blk = set()
                    while edges:
                        e = edges.pop()
                        blk.update(e)
                        if e == (u, v):
                            break
                    blocks.append(blk)
    return blocks


def _disjoint_blocks(n: int, blocks: list[set]) -> list[np.ndarray]:
    """Each vertex goes to its largest block (an articulation vertex joins the larger side)."""
    order = sorted(range(len(blocks)), key=lambda b: (-len(blocks[b]), min(blocks[b])))
    owner = [-1] * n
    for b in order:
        for v in blocks[b]:
            if owner[v] == -1:
                owner[v] = b
    groups: dict = {}
    for v, b in enumerate(owner):
        groups.setdefault(b, []).append(v)
    out = [np.asarray(sorted(g), dtype=np.int64) for g in groups.values()]
    out.sort(key=lambda g: (-len(g), int(g[0])))
    return out


def redundant_links(cross: list[tuple[int, int]], adj: list[set]) -> bool:
    """`cross` = links (group camera, kept camera). Redundant when two share no image, or share an image and
    their other ends are themselves linked (a closed triangle)."""
    for i in range(len(cross)):
        for j in range(i + 1, len(cross)):
            (a1, b1), (a2, b2) = cross[i], cross[j]
            if a1 != a2 and b1 != b2:
                return True
            if a1 == a2 and b2 in adj[b1]:
                return True
            if b1 == b2 and a2 in adj[a1]:
                return True
    return False


def _level(r: np.ndarray, idx) -> tuple[float | None, int]:
    v = r[np.asarray(idx, dtype=np.int64)]
    v = v[np.isfinite(v)]
    return (float(np.median(v)), int(len(v))) if len(v) else (None, 0)


def scale_levels_differ(r: np.ndarray, a, b, params: GateParams) -> bool | None:
    """Do camera sets a and b sit at different metric levels (|delta| > log(scale_step_factor), >=
    scale_min_cameras ratios on both sides)? None when it cannot be measured."""
    la, na = _level(r, a)
    lb, nb = _level(r, b)
    if la is None or lb is None or min(na, nb) < params.scale_min_cameras:
        return None
    return abs(lb - la) > math.log(params.scale_step_factor)


def scale_split(group: np.ndarray, r: np.ndarray, rank: np.ndarray, params: GateParams) -> list[np.ndarray]:
    """Split a group in capture order (`rank`) at metric scale steps (binary segmentation; a cut is a
    candidate where the side MEDIANS differ by more than log(scale_step_factor), placed where the MEANS differ
    most, and confirmed by `scale_levels_differ`); then segments at one level (A | B | A) re-join."""
    parts = _scale_split(np.asarray(group, dtype=np.int64), r, rank, params)
    if len(parts) <= 1:
        return parts
    root = list(range(len(parts)))
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if scale_levels_differ(r, parts[i], parts[j], params) is False:
                ri, rj = root[i], root[j]
                root = [ri if x == rj else x for x in root]
    merged: dict = {}
    for k, q in zip(root, parts):
        merged.setdefault(k, []).append(q)
    out = [np.sort(np.concatenate(v)) for v in merged.values()]
    out.sort(key=lambda g: (-len(g), int(g.min())))
    return out


def _scale_split(g: np.ndarray, r: np.ndarray, rank: np.ndarray, params: GateParams) -> list[np.ndarray]:
    g = g[np.argsort(rank[g], kind="stable")]
    gi = g[np.isfinite(r[g])]
    m = params.scale_min_cameras
    if len(gi) < 2 * m:
        return [g]
    y = r[gi]
    cands = []
    for j in range(m, len(gi) - m + 1):
        if abs(float(np.median(y[j:]) - np.median(y[:j]))) > math.log(params.scale_step_factor):
            cands.append((abs(float(y[j:].mean() - y[:j].mean())), j))
    for _, j in sorted(cands, key=lambda c: (-c[0], c[1])):
        cut = rank[gi[j]]
        left, right = g[rank[g] < cut], g[rank[g] >= cut]
        if scale_levels_differ(r, left, right, params):
            return _scale_split(left, r, rank, params) + _scale_split(right, r, rank, params)
    return [g]


def _rot_deg(R) -> float:
    c = (float(np.trace(R)) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


# ---------------------------------------------------------------------------
# the gate


def apply_gate(model: SolveModel, links: dict, metric_log: dict, *, link_rotations: dict,
               masks_applied: bool, params: GateParams | None = None) -> dict:
    """The rule (module docstring) on one solve.

    links: {(name_a, name_b): inliers} (`read_verified_links`); link_rotations: {(name_a, name_b): R_b_from_a}
    (`read_link_rotations`); metric_log: {name: log(z_sfm / z_metric)} (cameras without a finite ratio are
    simply absent); masks_applied: False when the solve's transient masks were not (all) applied.

    Returns {"labels": {name: label}, "components": [...], "rounds": [...], "evidence": {...}, "params": ...,
    "params_digest": ...}. Label 0 is the room (most supported cameras); every other label is unplaced, with
    `reason` / `reasons` from the contract's vocabulary."""
    params = params or GateParams()
    names = model.names
    idx = model.index()
    C = _shared_counts(model)
    supported = model.n_obs >= params.min_obs
    rank = np.argsort(np.argsort(np.asarray(names)))
    r = np.full(model.n, np.nan)
    for nm, v in (metric_log or {}).items():
        if nm in idx and v is not None and np.isfinite(v):
            r[idx[nm]] = float(v)
    honoured = set()
    for (a, b), R_ba in (link_rotations or {}).items():
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None:
            continue
        if _rot_deg(np.asarray(R_ba).T @ (model.R_cw[ib] @ model.R_cw[ia].T)) <= params.max_link_disagreement_deg:
            honoured.add(tuple(sorted((a, b))))
    edges_all, edges = [], []
    for (a, b), inl in (links or {}).items():
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None or ia == ib or inl < params.min_link_inliers:
            continue
        edges_all.append((ia, ib))
        if tuple(sorted((a, b))) in honoured:
            edges.append((ia, ib))
    n_supported = int(supported.sum())
    n_with_ratio = int((np.isfinite(r) & supported).sum())
    metric_fraction = float(n_with_ratio / n_supported) if n_supported else 0.0
    metric_available = bool(np.isfinite(r).any()) and metric_fraction >= params.min_metric_fraction
    if not metric_available:
        # Levels on fewer than `min_metric_fraction` of the cameras are not used at all:
        # half a scale test is not a scale test.
        r = np.full(model.n, np.nan)
    attach = masks_applied and metric_available
    evidence = {
        "links": f"{len(edges_all)} verified pairs >= {params.min_link_inliers} inliers",
        "links_honoured": f"{len(edges)} of them within {params.max_link_disagreement_deg} deg of the solve",
        "metric_scale": (f"{n_with_ratio} of {n_supported} supported cameras with a ratio ({metric_fraction:.0%}; "
                         f"available at >= {params.min_metric_fraction:.0%})"),
        "metric_fraction": round(metric_fraction, 4),
        "masks": "applied" if masks_applied else "NOT applied: no piece attached (fail-safe)",
        "attach": attach,
    }
    labels = np.full(model.n, -1, dtype=np.int64)
    rounds = []
    group_decisions: dict = {}  # group min camera -> decision, for the reasons
    next_label = 0
    comps = [int(v) for v, _ in sorted(zip(*np.unique(model.component, return_counts=True)),
                                       key=lambda vc: (-vc[1], vc[0]))]
    for comp in comps:
        members = np.flatnonzero(model.component == comp)
        sup = members[supported[members]]
        local = {int(v): k for k, v in enumerate(sup)}
        adj = [set() for _ in range(len(sup))]
        adj_all = [set() for _ in range(len(sup))]
        for a, b in edges:
            if a in local and b in local:
                adj[local[a]].add(local[b])
                adj[local[b]].add(local[a])
        for a, b in edges_all:
            if a in local and b in local:
                adj_all[local[a]].add(local[b])
                adj_all[local[b]].add(local[a])
        groups = [sup[g] for g in _disjoint_blocks(len(sup), biconnected_blocks(adj))] if len(sup) else []
        groups = [part for g in groups for part in scale_split(g, r, rank, params)]
        groups.sort(key=lambda g: (-len(g), int(g.min())))
        pending = list(groups)
        first_round = True
        while pending:
            reference = pending.pop(0)
            kept = [reference]
            kept_set = set(int(i) for i in reference)
            decisions = []
            info: dict = {}
            changed = True
            while changed and pending:
                changed = False
                best, best_n = None, -1
                for j, g in enumerate(pending):
                    gs = set(int(i) for i in g)
                    cross, cross_all = [], []
                    for ea, eb in edges:
                        if ea in gs and eb in kept_set:
                            cross.append((local[ea], local[eb]))
                        elif eb in gs and ea in kept_set:
                            cross.append((local[eb], local[ea]))
                    for ea, eb in edges_all:
                        if ea in gs and eb in kept_set:
                            cross_all.append((local[ea], local[eb]))
                        elif eb in gs and ea in kept_set:
                            cross_all.append((local[eb], local[ea]))
                    redundant = redundant_links(cross, adj)
                    coupled = len(cross) > 0 or bool(C[sorted(gs)][:, sorted(kept_set)].nnz)
                    kept_idx = np.fromiter(kept_set, dtype=np.int64)
                    differ = scale_levels_differ(r, kept_idx, g, params)
                    lg, _ = _level(r, g)
                    lk, _ = _level(r, kept_idx)
                    info[int(g.min())] = d = {
                        "cameras": int(len(g)), "first_camera": names[int(g.min())],
                        "cross_links": len(cross), "cross_links_all": len(cross_all),
                        "redundant": bool(redundant),
                        "redundant_without_honouring": bool(redundant_links(cross_all, adj_all)),
                        "coupled": bool(coupled), "scale_ok": differ is not True,
                        "scale_factor": (math.exp(lg - lk) if lg is not None and lk is not None else None),
                    }
                    if attach and redundant and coupled and d["scale_ok"] and len(cross) > best_n:
                        best, best_n = j, len(cross)
                if best is not None:
                    g = pending.pop(best)
                    kept.append(g)
                    kept_set |= set(int(i) for i in g)
                    decisions.append({"group": int(g.min()), "kept": True, "cross_links": best_n})
                    changed = True
            for g in pending:
                d = info.get(int(g.min()), {"cameras": int(len(g)), "first_camera": names[int(g.min())]})
                decisions.append({"group": int(g.min()), "kept": False, **d})
                if first_round:
                    group_decisions[int(g.min())] = d
            ids = np.concatenate(kept)
            labels[ids] = next_label
            rounds.append({"source_component": comp, "label": next_label,
                           "reference_group": {"cameras": int(len(reference)),
                                               "first_camera": names[int(reference.min())]},
                           "kept_groups": len(kept), "kept_cameras": int(len(ids)), "decisions": decisions})
            next_label += 1
            first_round = False
        rest = members[~supported[members]]
        labelled = members[labels[members] >= 0]
        if not len(labelled):
            labels[members] = next_label
            rounds.append({"source_component": comp, "label": next_label, "reference_group": None,
                           "kept_groups": 0, "kept_cameras": 0, "decisions": []})
            next_label += 1
            continue
        first = next(rd["label"] for rd in rounds if rd["source_component"] == comp)
        for i in rest:
            row = C[i, labelled].toarray().ravel()
            labels[i] = labels[labelled[int(np.argmax(row))]] if row.max() > 0 else first
    return _finish(model, labels, supported, rounds, group_decisions, masks_applied, metric_available, params,
                   evidence)


def _reasons(round_: dict, room_component: int, group_decisions: dict, masks_applied: bool,
             metric_available: bool) -> list[str]:
    """The contract's reasons for one unplaced label, in precedence order."""
    if not masks_applied:
        return [REASON_MASKS_UNAVAILABLE]
    if not metric_available:
        return [REASON_SCALE_UNAVAILABLE]
    if round_["source_component"] != room_component or round_["reference_group"] is None:
        return [REASON_SOLVED_SEPARATELY]
    first = round_["reference_group"]["first_camera"]
    d = next((v for v in group_decisions.values() if v.get("first_camera") == first), None)
    if d is None:
        return [REASON_NO_VERIFIED_LINK]
    out = []
    if not d.get("redundant", False):
        if d.get("cross_links_all", d.get("cross_links", 0)) == 0:
            # No verified link at all, honoured or not -- whatever points the two share
            # (review V7, M2: shared points alone are not a link).
            out.append(REASON_NO_VERIFIED_LINK)
        elif d.get("redundant_without_honouring", False):
            out.append(REASON_LINK_CONTRADICTED)
        else:
            out.append(REASON_SINGLE_UNCONFIRMED_LINK)
    if not d.get("scale_ok", True):
        out.append(REASON_SCALE_MISMATCH)
    return out or [REASON_NO_VERIFIED_LINK]


def _finish(model, labels, supported, rounds, group_decisions, masks_applied, metric_available, params,
            evidence) -> dict:
    counts = {int(lab): int(((labels == lab) & supported).sum()) for lab in np.unique(labels)}
    order = sorted(counts, key=lambda lab: (-counts[lab], -int((labels == lab).sum()), lab))
    remap = {old: new for new, old in enumerate(order)}
    final = np.array([remap[int(v)] for v in labels], dtype=np.int64)
    for rd in rounds:
        rd["final_label"] = remap.get(rd["label"])
    room_round = next(rd for rd in rounds if rd["final_label"] == 0)
    room_component = room_round["source_component"]
    comps = []
    for lab in range(len(order)):
        sel = final == lab
        rd = next((x for x in rounds if x["final_label"] == lab), None)
        reasons = [] if lab == 0 else _reasons(rd, room_component, group_decisions, masks_applied,
                                               metric_available)
        comps.append({"label": lab, "state": "placed" if lab == 0 else "unplaced",
                      "reason": reasons[0] if reasons else None, "reasons": reasons,
                      "cameras": int(sel.sum()), "supported": int((sel & supported).sum()),
                      "source_components": sorted({int(c) for c in model.component[sel]})})
    return {"labels": {model.names[i]: int(final[i]) for i in range(model.n)}, "components": comps,
            "rounds": _json_clean(rounds), "evidence": evidence, "gate": GATE_ID,
            "params": params.to_json(), "params_digest": params.digest(), "masks_applied": bool(masks_applied),
            "metric_available": bool(metric_available)}


def _json_clean(obj):
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
