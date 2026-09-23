"""The rigid gate: publish a placement only where the data fixes it.

WHY. GLOMAP reports a component as a connected VIEW GRAPH. Its bundle
adjustment, however, ties two cameras together only through 3-D points both
observe. On the target world (6e6d3fc3) the main component is two rigid
islands that share ZERO 3-D points, glued by 7 verified pairs whose inliers
lie on the phone in the wearer's hand; the second island's placement is then
whatever rotation averaging and global positioning left it at, and it moves
by 20-96 % of the scene extent between seeds (D1 2.4). A placement that is
neither observed (shared structure) nor reproducible (seed-stable) is not a
measurement, so it must not be published in the same frame.

THE RULE (generic: no room, landmark or keyframe index enters it)

  (a) Per component of the REFERENCE seed, cameras with at least
      `min_obs` 3-D observations are linked when they share >= `k_shared`
      3-D points; the connected parts are the rigid groups.
  (b) Groups are peeled from the largest. For every other seed, a Sim(3)
      (Umeyama) is fitted on the reference group's camera centres; each
      other group's placement error is the median displacement of its
      centres after that alignment, as a fraction of the component's extent
      (||p95 - p5|| of the reference seed's centres). A group's SPREAD is
      the maximum over seeds; a seed in which the group is not (mostly) in
      the reference group's component counts as infinite.
  (c) A group joins the kept set only if it shares >= `k_shared` DISTINCT
      3-D points with the groups already kept AND its spread is <=
      `max_spread`. What is not kept becomes a new component and the rule
      recurses on it with its own largest group as reference.

Cameras below `min_obs` are not measurements (global_solve's publish floor)
and follow the kept group they share most points with.

THRESHOLDS. `k_shared` and `max_spread` are chosen from the CONTROL world
b2a75ab4's noise floor and from physics, never from the target; the
derivation is in RUN/experiments/P2-E1/GATE-THRESHOLDS.md and summarised on
`GateParams`.

THE EVIDENCE RULE (`GateParams(rule="evidence")`; P2-GT, manager 006.2)

The shared-point rule above fixed K = 10 after the first target runs and
could not be re-derived leave-one-world-out. Scored against independent
ground truth on 7 worlds x 4 arms (image-only pair rotations, image-only
vertical-vanishing-point roll, harness tilt / eye height), it left the
most misplaced keyframes attached. The evidence rule replaces the support
count with the three things that make a placement a measurement:

  (a) CANDIDATE GROUPS are the biconnected blocks of the solver's VERIFIED
      view graph (`links`: image pairs with >= `min_link_inliers` verified
      inliers, COLMAP's floor) over a component's supported cameras.
      Inside a block no single image is a single point of failure; a block
      that hangs on the rest by one pair, or through one image, is its own
      group. (One floor-level pair attached 991e5a15's kf 13-25, 55 deg off
      in image-only roll; one pair attaches the target's bathroom.)
  (b) METRIC SCALE: with `metric_log` (per camera log(z_sfm / z_metric),
      e.g. the harness's TRI ratio), each group is split in capture order
      where its level steps by more than `scale_step_factor` with at least
      `scale_min_cameras` cameras on each side; segments that return to the
      same level re-join. One similarity cannot hold a x4 scale step.
  (c) ATTACHMENT, peeled from the largest group: a group joins the kept set
      only if its links to the kept set are REDUNDANT (two vertex-disjoint
      verified pairs, or two pairs through one image whose other ends are
      themselves verified -- a closed triangle), it is SEED-STABLE (spread
      <= `max_spread`, the measure above) and its metric level agrees with
      the kept set's (x `scale_step_factor`, both sides >= `scale_min_cameras`
      cameras with a ratio).

Without `links` the candidate groups fall back to rigid groups at the
physical floor of 3 shared points and redundancy is not evaluated. Without
`metric_log` the rule REFUSES to run unless `require_metric=False` (then the
scale tests are skipped): the scale tests do most of its work. The report
records which evidence ran. The driver does not pass either input yet.

THRESHOLDS (derivation and scores: RUN/experiments/P2-GT/out/ -- grid.json,
lowo_v4.txt, ablation.txt, lowo_ablation.txt, final_table.txt):
`scale_step_factor` 1.25 and `scale_min_cameras` 10 are the harness's
physical plausibility bounds (PLAUSIBILITY scale_max_factor, min_group_kf),
and each is also the leave-one-world-out choice in 6/7 folds.
`max_spread` 0.075 = the harness's 0.3 m position tolerance over the
control's metric extent (3.95 m); in 6/7 leave-one-world-out folds any
value in [0.04, 0.3] is optimal (the target fold leaves it unconstrained).

numpy / scipy only; no pycolmap. The gate functions do no file IO;
`read_verified_links` is the one IO helper (stdlib sqlite3, read-only).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

INF = float("inf")


RULES = ("shared_k", "evidence")
# Per-rule default of `max_spread` (see GateParams).
DEFAULT_MAX_SPREAD = {"shared_k": 0.02, "evidence": 0.075}


@dataclass(frozen=True)
class GateParams:
    # "shared_k": the shared-point rule (the default, unchanged); "evidence":
    # verified-graph blocks + metric scale + redundant, seed-stable attachment.
    rule: str = "shared_k"
    # Minimum shared 3-D points that couple two cameras (a) and two groups (c).
    # Physics floor: a Sim(3) between two rigid bodies has 7 DOF and a 3-D
    # point seen by both gives 3 constraints, so 3 non-collinear points is the
    # minimum that fixes it at all. Control ceiling: the known-good control
    # b2a75ab4 (A0, 5 seeds) stays ONE rigid group for every K <= 20 and first
    # sheds a camera at K = 30. 10 = half that ceiling, >= 3x the physical
    # floor (RUN/experiments/P2-E1/GATE-THRESHOLDS.md).
    k_shared: int = 10
    # Maximum seed spread (fraction of component extent) of a coupled group.
    # None = the rule's default (DEFAULT_MAX_SPREAD).
    # shared_k, 0.02: control noise floor: its genuinely coupled parts
    # (capture-order blocks, K=50 sub-groups) move <= 0.0014 of extent between
    # seeds (whole main component: median 0.0002). 0.02 is ~14x that floor.
    # evidence, 0.075: the harness's 0.3 m position tolerance (eye-height band)
    # over the control's metric extent (18.8 units / 4.76 units per metre =
    # 3.95 m). 0.02 split the target's closet, which six image-only pairs
    # place within 0.6 deg (seed spread 0.024-0.031); the false-glued groups
    # it must split move 0.59-0.86. Leave-one-world-out: any value in
    # [0.04, 0.3] is optimal in 6/7 folds (P2-GT out/lowo_v4.txt).
    max_spread: float | None = None
    # global_solve.MIN_IMAGE_OBSERVATIONS: fewer observations is not a pose.
    min_obs: int = 30
    # A group counts as present in a seed's reference component when at least
    # this fraction of its cameras is registered there.
    min_present_frac: float = 0.5
    # Fewer common reference cameras than this and a seed cannot be aligned.
    min_align_cameras: int = 5
    # evidence rule: a verified pair counts as a link at COLMAP's verification
    # floor (two_view_geometry min_num_inliers; bridge.MIN_VERIFIED_INLIERS).
    min_link_inliers: int = 15
    # evidence rule: drop UNCALIBRATED two-view geometries (COLMAP config 3) from the links. The cameras are
    # calibrated; COLMAP labels a pair UNCALIBRATED when the essential matrix explains clearly fewer of its
    # matches than a fundamental matrix, i.e. the matches do not fit the known camera. On the control their
    # rotation disagrees with the solve by p95 47 deg (CALIBRATED: 14). 6839fb8f kf 92-98 (misplaced 88-113
    # deg) was attached by four such links, 15-18 inliers each, every one contradicted by the solve by
    # 34-96 deg (RUN/experiments/P2-LM/gatefix/b6839_links.json).
    exclude_uncalibrated_links: bool = False
    # evidence rule: attach a group only when its metric level is MEASURED and matches (scale_levels_differ
    # False). Default False keeps the lenient reading: a group whose level cannot be measured (fewer than
    # scale_min_cameras TRI ratios on a side) passes the scale test. On 2f447162 that let a 14-keyframe
    # island at scale x0.02-0.04, 37-41 deg tilt and ~1 m height error attach on two links alone.
    require_group_scale: bool = False
    # evidence rule: a verified link is evidence only when the solve HONOURS it -- its own two-view rotation
    # (COLMAP's pose for the pair's stored geometry, `read_link_rotations`) is within this many degrees of the
    # reference model's relative rotation. None = off. A link the solve contradicts is not evidence that the
    # solve placed the pair right: 6839fb8f kf 92-98 (misplaced 88-113 deg) was attached and held by links the
    # solve contradicts by 34-96 deg. The bound is the CONTROL's: p95 of link-vs-solve disagreement over all
    # b2a75ab4 links, 25.2 deg (p90 16.8). GT's 24 cases, masked arms: 6839 7 -> 0 misplaced attached, control
    # and target unchanged (RUN/experiments/P2-LM/gatefix/gt_agree.txt).
    # THE PRODUCT VALUE IS 16.8, THE CONTROL'S p90, CHOSEN DELIBERATELY (manager 011): p90 and p95 were both
    # derived from the control alone and both declared before any result; the costs are asymmetric -- a false
    # attachment is confidently wrong geometry, a false split is coverage shown honestly as an area -- so the
    # stricter of the two is the default. Masked arms at 16.8: 30 -> 24 misplaced attached, 0 correct split.
    max_link_disagreement_deg: float | None = None
    # evidence rule: False = attach NO group to a component's anchor block: everything outside the anchor is
    # left unplaced. The fail-safe when the transient masks are unavailable (manager 011): without masks,
    # dropping the seed-stability test costs 103 -> 131 misplaced and 10 -> 93 scale-misplaced keyframes on
    # GT's unmasked arms, so no group may be attached on evidence that masks would have cleaned. Measured on
    # the unmasked arms with honoured links (16.8): misplaced attached 16, scale-misplaced 0, correct keyframes
    # split 134 (vs 16 / 82 / 117 attaching, and 103 / 10 / 45 for the rule with its stability test) --
    # RUN/experiments/P2-LM/gatefix/gt_noattach.txt.
    attach_groups: bool = True
    # evidence rule: a metric scale step / mismatch beyond this factor splits
    # (harness PLAUSIBILITY scale_max_factor: MoGe's per-image error is a few
    # %, region bias ~10 %; x1.25 is a reconstruction error, not noise).
    scale_step_factor: float = 1.25
    # evidence rule: cameras with a metric ratio needed on EACH side of a scale
    # comparison (harness PLAUSIBILITY min_group_kf).
    scale_min_cameras: int = 10
    # evidence rule: refuse to run without metric_log. The scale tests carry most of the rule (without them it
    # left 947 misplaced keyframes attached over 24 world-arms vs 133 with them, and the shared-point rule 579),
    # so a caller that cannot supply metric scale must opt out explicitly.
    require_metric: bool = True

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"gate rule {self.rule!r} not in {RULES}")
        if self.max_spread is None:
            object.__setattr__(self, "max_spread", DEFAULT_MAX_SPREAD[self.rule])

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict | None) -> "GateParams":
        d = dict(d or {})
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class SeedModel:
    """One mapping run, reduced to what the gate and the export need.

    Cameras are indexed 0..n-1; `component` is the model index in the run
    (0 = most registered images). Observations are (image, point) incidences.
    """

    seed: int
    names: list[str]
    component: np.ndarray            # (n,) int
    R_cw: np.ndarray                 # (n, 3, 3)
    t_cw: np.ndarray                 # (n, 3)
    n_obs: np.ndarray                # (n,) int
    xyz: np.ndarray                  # (P, 3)
    point_component: np.ndarray      # (P,) int
    obs_image: np.ndarray            # (M,) int
    obs_point: np.ndarray            # (M,) int
    obs_uv: np.ndarray | None = None  # (M, 2)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def centres(self) -> np.ndarray:
        return -np.einsum("nji,nj->ni", self.R_cw, self.t_cw)

    def index(self) -> dict[str, int]:
        return {nm: i for i, nm in enumerate(self.names)}


# ---------------------------------------------------------------------------
# helpers


def incidence(model: SeedModel):
    """Binary camera x point incidence (scipy.sparse CSR)."""
    from scipy.sparse import csr_matrix

    P = len(model.xyz)
    data = np.ones(len(model.obs_image), dtype=np.int32)
    M = csr_matrix((data, (model.obs_image.astype(np.int64), model.obs_point.astype(np.int64))),
                   shape=(model.n, P))
    M.data[:] = 1
    M.sum_duplicates()
    M.data[:] = 1
    return M


def shared_counts(M):
    """Camera x camera shared-point counts."""
    return (M @ M.T).tocsr()


def rigid_groups(model: SeedModel, members, k_shared: int, *, C=None) -> list[np.ndarray]:
    """Connected parts of `members` under 'share >= k_shared points'. Sorted
    by size (desc), ties by smallest camera index."""
    from scipy.sparse.csgraph import connected_components

    members = np.asarray(sorted(int(i) for i in members), dtype=np.int64)
    if not len(members):
        return []
    if C is None:
        C = shared_counts(incidence(model))
    sub = C[members][:, members].tocoo()
    keep = (sub.data >= k_shared) & (sub.row != sub.col)
    from scipy.sparse import csr_matrix

    A = csr_matrix((np.ones(int(keep.sum())), (sub.row[keep], sub.col[keep])), shape=(len(members),) * 2)
    _, labels = connected_components(A, directed=False)
    groups = [members[labels == lab] for lab in np.unique(labels)]
    groups.sort(key=lambda g: (-len(g), int(g[0])))
    return groups


def extent(centres: np.ndarray) -> float:
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    if len(c) < 2:
        return 0.0
    return float(np.linalg.norm(np.percentile(c, 95, axis=0) - np.percentile(c, 5, axis=0)))


def umeyama(src: np.ndarray, dst: np.ndarray):
    """dst ~ s R src + t (least squares)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs ** 2).sum() / len(src)
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def placement_error(ref: SeedModel, other: SeedModel, ref_group_names, group_names, ext: float,
                    params: GateParams) -> dict:
    """Place `group_names` in `other` by the Sim(3) that aligns `ref_group_names`
    of `other` onto `ref`; return the median displacement / ext (inf when the
    group is not in the reference group's component of `other`)."""
    oi = other.index()
    ri = ref.index()
    rg = [n for n in ref_group_names if n in oi]
    if len(rg) < params.min_align_cameras:
        return {"error": None, "why": "reference group not registered in this seed"}
    comps = other.component[[oi[n] for n in rg]]
    vals, counts = np.unique(comps, return_counts=True)
    cm = int(vals[np.argmax(counts)])
    rg = [n for n in rg if other.component[oi[n]] == cm]
    if len(rg) < max(params.min_align_cameras, params.min_present_frac * len(ref_group_names)):
        return {"error": None, "why": "reference group not reproduced in this seed"}
    oc = other.centres
    rc = ref.centres
    s, R, t = umeyama(oc[[oi[n] for n in rg]], rc[[ri[n] for n in rg]])
    ref_res = np.linalg.norm((s * (R @ oc[[oi[n] for n in rg]].T)).T + t - rc[[ri[n] for n in rg]], axis=1)
    g_in = [n for n in group_names if n in oi and other.component[oi[n]] == cm]
    if len(g_in) < params.min_present_frac * len(group_names) or not g_in:
        return {"error": INF, "why": "group not in the reference group's component in this seed",
                "present": len(g_in), "reference_residual": float(np.median(ref_res) / ext) if ext else None}
    e = np.linalg.norm((s * (R @ oc[[oi[n] for n in g_in]].T)).T + t - rc[[ri[n] for n in g_in]], axis=1)
    return {"error": float(np.median(e) / ext) if ext > 0 else INF, "present": len(g_in),
            "reference_residual": float(np.median(ref_res) / ext) if ext else None}


def group_spread(models: list[SeedModel], ref_group_names, group_names, ext: float,
                 params: GateParams) -> dict:
    ref = models[0]
    per_seed = {}
    for m in models[1:]:
        per_seed[str(m.seed)] = placement_error(ref, m, ref_group_names, group_names, ext, params)
    errs = [r["error"] for r in per_seed.values() if r.get("error") is not None]
    if len(models) < 2:
        return {"spread": None, "measured": False, "why": "one seed: spread cannot be measured", "per_seed": {}}
    if not errs:
        return {"spread": INF, "measured": False, "why": "no seed reproduced the reference group",
                "per_seed": per_seed}
    return {"spread": float(max(errs)), "measured": True, "per_seed": per_seed}


# ---------------------------------------------------------------------------
# the gate


def apply_rigid_gate(models: list[SeedModel], params: GateParams | None = None, *,
                     links=None, metric_log=None, link_rotations=None) -> dict:
    """Final component per camera of the reference seed (models[0]).

    Returns {"labels": {name: int}, "components": [...], "rounds": [...],
    "params": ...}. Component labels are renumbered by supported size
    (0 = most supported cameras). `params.rule` selects the rule; `links` and
    `metric_log` are used by the evidence rule only (see
    `apply_evidence_gate`)."""
    params = params or GateParams()
    if params.rule == "evidence":
        return apply_evidence_gate(models, params, links=links, metric_log=metric_log, link_rotations=link_rotations)
    ref = models[0]
    M = incidence(ref)
    C = shared_counts(M)
    centres = ref.centres
    supported = ref.n_obs >= params.min_obs
    labels = np.full(ref.n, -1, dtype=np.int64)
    rounds = []
    next_label = 0
    for comp in _components_by_size(ref):
        members = np.flatnonzero(ref.component == comp)
        sup = members[supported[members]]
        ext = extent(centres[sup]) if len(sup) >= 2 else extent(centres[members])
        groups = rigid_groups(ref, sup, params.k_shared, C=C)
        pending = list(groups)
        while pending:
            reference = pending.pop(0)
            kept = [reference]
            kept_pts = np.asarray(M[reference].sum(axis=0)).ravel() > 0
            ref_names = [ref.names[i] for i in reference]
            info = {}
            for g in pending:
                gn = [ref.names[i] for i in g]
                sp = group_spread(models, ref_names, gn, ext, params)
                info[int(g[0])] = {"cameras": int(len(g)), "first_camera": ref.names[int(g[0])], **sp}
            decisions = []
            changed = True
            while changed and pending:
                changed = False
                best, best_shared = None, -1
                for j, g in enumerate(pending):
                    g_pts = np.asarray(M[g].sum(axis=0)).ravel() > 0
                    shared = int((g_pts & kept_pts).sum())
                    info[int(g[0])]["shared_with_kept"] = shared
                    sp = info[int(g[0])]["spread"]
                    stable = sp is None or sp <= params.max_spread
                    if shared >= params.k_shared and stable and shared > best_shared:
                        best, best_shared = j, shared
                if best is not None:
                    g = pending.pop(best)
                    kept.append(g)
                    kept_pts |= np.asarray(M[g].sum(axis=0)).ravel() > 0
                    decisions.append({"group": int(g[0]), "kept": True, "shared_with_kept": best_shared})
                    changed = True
            for g in pending:
                d = info[int(g[0])]
                why = []
                if d.get("shared_with_kept", 0) < params.k_shared:
                    why.append(f"shares {d.get('shared_with_kept', 0)} < {params.k_shared} points with the kept groups")
                if d.get("spread") is not None and d["spread"] > params.max_spread:
                    why.append(f"seed spread {d['spread']:.3f} > {params.max_spread}")
                decisions.append({"group": int(g[0]), "kept": False, "why": "; ".join(why) or "not reached"})
            idx = np.concatenate(kept)
            labels[idx] = next_label
            rounds.append({
                "source_component": int(comp), "label": next_label,
                "reference_group": {"cameras": int(len(reference)), "first_camera": ref.names[int(reference[0])]},
                "kept_groups": len(kept), "kept_cameras": int(len(idx)),
                "extent": ext, "groups": _json_clean(info), "decisions": decisions,
            })
            next_label += 1
        # unsupported cameras follow the labelled camera they share most with
        rest = members[~supported[members]]
        labelled = members[labels[members] >= 0]
        if not len(labelled):
            # a component with no supported camera at all stays one (unpublished) component
            labels[members] = next_label
            next_label += 1
            continue
        first = _first_label_of(rounds, comp)
        for i in rest:
            row = C[i, labelled].toarray().ravel()
            labels[i] = labels[labelled[int(np.argmax(row))]] if row.max() > 0 else first
    return _finish(ref, labels, supported, rounds, params)


def _finish(ref: SeedModel, labels: np.ndarray, supported: np.ndarray, rounds: list, params: GateParams) -> dict:
    """Renumber final labels by supported size and build the report."""
    counts = {int(lab): int(((labels == lab) & supported).sum()) for lab in np.unique(labels)}
    order = sorted(counts, key=lambda lab: (-counts[lab], -int((labels == lab).sum()), lab))
    remap = {old: new for new, old in enumerate(order)}
    final = np.array([remap[int(v)] for v in labels], dtype=np.int64)
    for r in rounds:
        r["final_label"] = remap.get(r["label"], None)
    comps = []
    for lab in range(len(order)):
        sel = final == lab
        src = sorted({int(c) for c in ref.component[sel]})
        comps.append({"label": lab, "cameras": int(sel.sum()), "supported": int((sel & supported).sum()),
                      "source_components": src})
    splits = sum(1 for r in rounds) - len({r["source_component"] for r in rounds})
    return {"labels": {ref.names[i]: int(final[i]) for i in range(ref.n)},
            "components": comps, "rounds": rounds, "params": params.to_json(),
            "splits": int(splits)}


# ---------------------------------------------------------------------------
# the evidence rule


def read_verified_links(database_path, min_inliers: int = 15, exclude_configs=(0, 1)) -> dict:
    """{(name_a, name_b) sorted: inliers} for every verified two-view geometry of a COLMAP database
    (config not in `exclude_configs` -- default UNDEFINED / DEGENERATE -- and >= `min_inliers` inliers).
    Opened read-only and immutable, so a frozen database is never touched."""
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


def read_link_rotations(database_path, camera: dict, min_inliers: int = 15, exclude_configs=(0, 1)) -> dict:
    """{(name_a, name_b): R_b_from_a} for every verified pair: COLMAP's own relative pose for the pair's stored
    geometry and inliers (`pycolmap.estimate_two_view_geometry_pose`, which decomposes E or H by the pair's
    configuration); `camera` is the single PINHOLE camera {fx, fy, cx, cy, width, height}. Pairs whose pose
    cannot be recovered are absent. Read-only and immutable, like `read_verified_links`."""
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


_COLMAP_PAIR_BASE = 2147483647
# COLMAP TwoViewGeometry configurations that are not link evidence for the gate.
NOT_VERIFIED_CONFIGS = (0, 1)          # UNDEFINED, DEGENERATE
UNCALIBRATED_CONFIG = 3


def biconnected_blocks(adj: list[set]) -> list[set]:
    """Biconnected blocks (Hopcroft-Tarjan, iterative) of an undirected graph given as adjacency sets.
    Blocks share at most one vertex (an articulation vertex); an isolated vertex is a block of its own."""
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
    """`cross` = links (group camera, kept camera). Redundant when two of them share no image, or when two of
    them share an image and their other ends are themselves linked (a closed triangle). One pair, or a star
    whose ends are unrelated, is a single point of failure."""
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


def scale_levels_differ(R: np.ndarray, a, b, params: GateParams) -> bool | None:
    """Do camera sets a and b sit at different metric levels? Every estimator (row of R) with >=
    scale_min_cameras ratios on both sides must see |delta| > log(scale_step_factor), all with one sign.
    None when no estimator can measure."""
    signs = []
    for r in np.atleast_2d(R):
        la, na = _level(r, a)
        lb, nb = _level(r, b)
        if la is None or lb is None or min(na, nb) < params.scale_min_cameras:
            continue
        d = lb - la
        if abs(d) <= math.log(params.scale_step_factor):
            return False
        signs.append(np.sign(d))
    if not signs:
        return None
    return all(s == signs[0] for s in signs)


def scale_split(group: np.ndarray, R: np.ndarray, rank: np.ndarray, params: GateParams) -> list[np.ndarray]:
    """Split a group, in capture order (`rank`), at metric scale steps: binary segmentation on the first
    estimator, each cut confirmed by `scale_levels_differ`; then segments that sit at one level (A | B | A)
    re-join. As in the harness's step finder, a cut is a candidate when the MEDIANS of the two sides differ
    by more than log(scale_step_factor) (robust detection) and is placed where the MEANS differ most (the
    median difference is flat around a clean step; the mean difference peaks on it)."""
    R = np.atleast_2d(R)
    parts = _scale_split(np.asarray(group, dtype=np.int64), R, rank, params)
    if len(parts) <= 1:
        return parts
    root = list(range(len(parts)))
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if scale_levels_differ(R, parts[i], parts[j], params) is False:
                ri, rj = root[i], root[j]
                root = [ri if x == rj else x for x in root]
    merged: dict = {}
    for k, q in zip(root, parts):
        merged.setdefault(k, []).append(q)
    out = [np.sort(np.concatenate(v)) for v in merged.values()]
    out.sort(key=lambda g: (-len(g), int(g.min())))
    return out


def _scale_split(g: np.ndarray, R: np.ndarray, rank: np.ndarray, params: GateParams) -> list[np.ndarray]:
    g = g[np.argsort(rank[g], kind="stable")]
    gi = g[np.isfinite(R[0][g])]
    m = params.scale_min_cameras
    if len(gi) < 2 * m:
        return [g]
    y = R[0][gi]
    cands = []
    for j in range(m, len(gi) - m + 1):
        if abs(float(np.median(y[j:]) - np.median(y[:j]))) > math.log(params.scale_step_factor):
            cands.append((abs(float(y[j:].mean() - y[:j].mean())), j))
    for _, j in sorted(cands, key=lambda c: (-c[0], c[1])):
        cut = rank[gi[j]]
        left, right = g[rank[g] < cut], g[rank[g] >= cut]
        if scale_levels_differ(R, left, right, params):
            return _scale_split(left, R, rank, params) + _scale_split(right, R, rank, params)
    return [g]


def apply_evidence_gate(models: list[SeedModel], params: GateParams | None = None, *,
                        links=None, metric_log=None, link_rotations=None) -> dict:
    """The evidence rule (module docstring). Cameras of the reference seed (models[0]).

    links: {(name_a, name_b): inliers} verified pairs (`read_verified_links`), or None.
    metric_log: {name: log(z_sfm / z_metric)} or a list of such maps (every estimator must agree), or None.
    Staged names must sort in capture order (driver.staged_name)."""
    params = params or GateParams(rule="evidence")
    ref = models[0]
    names = ref.names
    idx = ref.index()
    M = incidence(ref)
    C = shared_counts(M)
    centres = ref.centres
    supported = ref.n_obs >= params.min_obs
    rank = np.argsort(np.argsort(np.asarray(names)))
    maps = [] if not metric_log else (list(metric_log) if isinstance(metric_log, (list, tuple)) else [metric_log])
    R = np.full((max(1, len(maps)), ref.n), np.nan)
    for e, mp in enumerate(maps):
        for nm, v in mp.items():
            if nm in idx and v is not None and np.isfinite(v):
                R[e, idx[nm]] = float(v)
    have_metric = bool(maps) and bool(np.isfinite(R).any())
    if params.require_metric and not have_metric:
        raise ValueError("the evidence gate needs metric_log (per-camera log(z_sfm / z_metric), e.g. the harness "
                         "TRI ratio) and should get links (read_verified_links); pass require_metric=False to run "
                         "without scale evidence")
    honoured = None
    if params.max_link_disagreement_deg is not None and links is not None:
        if link_rotations is None:
            raise ValueError("max_link_disagreement_deg needs link_rotations (read_link_rotations)")
        honoured = set()
        for (a, b), R_ba in dict(link_rotations).items():
            ia, ib = idx.get(a), idx.get(b)
            if ia is None or ib is None:
                continue
            R_solve = ref.R_cw[ib] @ ref.R_cw[ia].T
            c = (np.trace(np.asarray(R_ba).T @ R_solve) - 1.0) / 2.0
            if math.degrees(math.acos(min(1.0, max(-1.0, c)))) <= params.max_link_disagreement_deg:
                honoured.add(tuple(sorted((a, b))))
    edges = []
    n_contradicted = 0
    if links is not None:
        for (a, b), inl in dict(links).items():
            ia, ib = idx.get(a), idx.get(b)
            if ia is not None and ib is not None and ia != ib and inl >= params.min_link_inliers:
                if honoured is not None and tuple(sorted((a, b))) not in honoured:
                    n_contradicted += 1
                    continue
                edges.append((ia, ib))
    no_links = "unavailable: candidate groups = rigid groups at 3 shared points; redundancy not tested"
    evidence = {"links": (f"{len(edges)} verified pairs >= {params.min_link_inliers} inliers" if links is not None
                          else no_links),
                "links_not_honoured": (None if honoured is None else
                                       f"{n_contradicted} verified pairs set aside: the solve contradicts them by more "
                                       f"than {params.max_link_disagreement_deg} deg, or their rotation is unknown"),
                "metric_scale": (f"{int(np.isfinite(R).any(0).sum())} cameras, {len(maps)} estimator(s)"
                                 if have_metric else "unavailable: scale split / scale agreement not tested")}
    labels = np.full(ref.n, -1, dtype=np.int64)
    rounds = []
    next_label = 0
    for comp in _components_by_size(ref):
        members = np.flatnonzero(ref.component == comp)
        sup = members[supported[members]]
        ext = extent(centres[sup]) if len(sup) >= 2 else extent(centres[members])
        local = {int(v): k for k, v in enumerate(sup)}
        adj = [set() for _ in range(len(sup))]
        for a, b in edges:
            if a in local and b in local:
                adj[local[a]].add(local[b])
                adj[local[b]].add(local[a])
        if links is not None:
            groups = [sup[g] for g in _disjoint_blocks(len(sup), biconnected_blocks(adj))]
        else:
            groups = rigid_groups(ref, sup, 3, C=C) if len(sup) else []
        if have_metric:
            groups = [part for g in groups for part in scale_split(g, R, rank, params)]
        groups.sort(key=lambda g: (-len(g), int(g.min())))
        pending = list(groups)
        while pending:
            reference = pending.pop(0)
            kept = [reference]
            kept_set = set(int(i) for i in reference)
            ref_names = [names[i] for i in reference]
            info = {}
            for g in pending:
                sp = group_spread(models, ref_names, [names[i] for i in g], ext, params)
                info[int(g.min())] = {"cameras": int(len(g)), "first_camera": names[int(g.min())], **sp}
            decisions = []
            changed = True
            while changed and pending:
                changed = False
                best, best_n = None, -1
                for j, g in enumerate(pending):
                    d = info[int(g.min())]
                    gs = set(int(i) for i in g)
                    cross = []
                    for a, b in edges:
                        if a in gs and b in kept_set:
                            cross.append((local[a], local[b]))
                        elif b in gs and a in kept_set:
                            cross.append((local[b], local[a]))
                    d["cross_links"] = len(cross)
                    sp = d.get("spread")
                    stable = sp is None or sp <= params.max_spread
                    redundant = True if links is None else redundant_links(cross, adj)
                    coupled = len(cross) > 0 or bool(C[list(gs)][:, sorted(kept_set)].nnz)
                    scale_ok = True
                    if have_metric:
                        kept_idx = np.fromiter(kept_set, dtype=np.int64)
                        lg, _ = _level(R[0], g)
                        lk, _ = _level(R[0], kept_idx)
                        if lg is not None and lk is not None:
                            d["scale_factor"] = math.exp(lg - lk)
                        differ = scale_levels_differ(R, kept_idx, g, params)
                        scale_ok = differ is False if params.require_group_scale else differ is not True
                        d["scale_measured"] = differ is not None
                    d.update(stable=bool(stable), redundant=bool(redundant), scale_ok=bool(scale_ok))
                    if params.attach_groups and stable and redundant and coupled and scale_ok and len(cross) > best_n:
                        best, best_n = j, len(cross)
                if best is not None:
                    g = pending.pop(best)
                    kept.append(g)
                    kept_set |= set(int(i) for i in g)
                    decisions.append({"group": int(g.min()), "kept": True, "cross_links": best_n})
                    changed = True
            for g in pending:
                d = info[int(g.min())]
                why = []
                if not d.get("redundant", True):
                    why.append(f"{d.get('cross_links', 0)} verified link(s) to the kept groups, not redundant")
                if not d.get("stable", True):
                    why.append(f"seed spread {d['spread']:.3f} > {params.max_spread}")
                if not d.get("scale_ok", True):
                    if d.get("scale_measured", True):
                        why.append(f"metric scale x{d.get('scale_factor', float('nan')):.2f} vs the kept groups")
                    else:
                        why.append(f"metric scale not measurable (< {params.scale_min_cameras} cameras with a ratio)")
                decisions.append({"group": int(g.min()), "kept": False, "why": "; ".join(why) or "not coupled"})
            ids = np.concatenate(kept)
            labels[ids] = next_label
            rounds.append({
                "source_component": int(comp), "label": next_label,
                "reference_group": {"cameras": int(len(reference)), "first_camera": names[int(reference.min())]},
                "kept_groups": len(kept), "kept_cameras": int(len(ids)),
                "extent": ext, "groups": _json_clean(info), "decisions": decisions,
            })
            next_label += 1
        rest = members[~supported[members]]
        labelled = members[labels[members] >= 0]
        if not len(labelled):
            labels[members] = next_label
            next_label += 1
            continue
        first = _first_label_of(rounds, comp)
        for i in rest:
            row = C[i, labelled].toarray().ravel()
            labels[i] = labels[labelled[int(np.argmax(row))]] if row.max() > 0 else first
    out = _finish(ref, labels, supported, rounds, params)
    out["evidence"] = evidence
    return out


def _components_by_size(model: SeedModel) -> list[int]:
    vals, counts = np.unique(model.component, return_counts=True)
    return [int(v) for v, _ in sorted(zip(vals, counts), key=lambda vc: (-vc[1], vc[0]))]


def _first_label_of(rounds, comp) -> int:
    for r in rounds:
        if r["source_component"] == comp:
            return r["label"]
    return -1


def _json_clean(obj):
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return "inf" if obj > 0 else "-inf"
    if isinstance(obj, (np.floating,)):
        return _json_clean(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def seed_components(models: list[SeedModel], keyframe_names: set[str]) -> list[dict]:
    """Per seed: component sizes in keyframes and in all images."""
    out = []
    for m in models:
        rows = []
        for comp in _components_by_size(m):
            sel = m.component == comp
            kf = sum(1 for i in np.flatnonzero(sel) if m.names[i] in keyframe_names)
            rows.append({"component": comp, "images": int(sel.sum()), "keyframes": int(kf),
                         "keyframes_supported": int(sum(1 for i in np.flatnonzero(sel)
                                                        if m.names[i] in keyframe_names and m.n_obs[i] >= 30))})
        out.append({"seed": m.seed, "components": rows})
    return out


def _main_names(m: SeedModel, names=None, min_obs: int = 30) -> list:
    main = _components_by_size(m)[0]
    sel = np.flatnonzero((m.component == main) & (m.n_obs >= min_obs))
    out = [m.names[i] for i in sel]
    return [n for n in out if n in names] if names is not None else out


def pairwise_disagreement(a: SeedModel, b: SeedModel, names=None, min_obs: int = 30) -> float:
    """Median displacement (fraction of a's main extent) of the supported cameras both
    seeds put in their main components, after a Sim(3) of b onto a. 1.0 when they share
    fewer than 5 such cameras (they do not even agree on what the main component is)."""
    an, bn = set(_main_names(a, names, min_obs)), set(_main_names(b, names, min_obs))
    common = sorted(an & bn)
    if len(common) < 5:
        return 1.0
    ai, bi = a.index(), b.index()
    A = b.centres[[bi[n] for n in common]]
    B = a.centres[[ai[n] for n in common]]
    s, R, t = umeyama(A, B)
    e = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    ext = extent(a.centres[[ai[n] for n in an]]) or 1.0
    # cameras only one seed supports in its main component count as disagreement too
    miss = len(an ^ bn) / max(1, len(an | bn))
    return float(max(np.median(e) / ext, miss))


def choose_reference(models: list[SeedModel], how="medoid", names=None) -> dict:
    """The seed whose model is published and anchors the gate.

    "medoid": the seed with the smallest median disagreement with the other seeds (main
    component placement and membership); ties go to the earlier seed. An int: that seed."""
    seeds = [m.seed for m in models]
    if how != "medoid":
        return {"index": seeds.index(int(how)), "seed": int(how), "rule": "fixed"}
    if len(models) == 1:
        return {"index": 0, "seed": seeds[0], "rule": "medoid (one seed)"}
    D = np.zeros((len(models), len(models)))
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i != j:
                D[i, j] = pairwise_disagreement(a, b, names)
    score = [float(np.median(np.delete(D[i], i))) for i in range(len(models))]
    k = int(np.argmin(score))
    return {"index": k, "seed": seeds[k], "rule": "medoid", "scores": dict(zip(map(str, seeds), score)),
            "disagreement": D.round(4).tolist()}


def seed_spread(models: list[SeedModel], names=None, min_obs: int = 30) -> dict:
    """Whole-component seed spread: every other seed aligned (Sim(3)) on the
    reference seed's largest component, over the cameras both register there.
    Median / p90 displacement as a fraction of extent."""
    ref = models[0]
    if len(models) < 2 or not ref.n:
        return {"measured": False}
    ri = ref.index()
    # SUPPORTED cameras only: a zero-support camera can sit hundreds of units out
    # (global_solve, E3) and would drag a least-squares Sim(3) with it
    ref_names = _main_names(ref, names, min_obs)
    rc = ref.centres
    ext = extent(rc[[ri[n] for n in ref_names]])
    out = {}
    for m in models[1:]:
        oi = m.index()
        common = [n for n in ref_names if n in oi]
        if len(common) < 5:
            out[str(m.seed)] = {"common": len(common)}
            continue
        comps = m.component[[oi[n] for n in common]]
        vals, counts = np.unique(comps, return_counts=True)
        cm = int(vals[np.argmax(counts)])
        common = [n for n in common if m.component[oi[n]] == cm and m.n_obs[oi[n]] >= min_obs]
        if len(common) < 5:
            out[str(m.seed)] = {"common": len(common)}
            continue
        A = m.centres[[oi[n] for n in common]]
        B = rc[[ri[n] for n in common]]
        s, R, t = umeyama(A, B)
        e = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1) / (ext or 1.0)
        out[str(m.seed)] = {"common": len(common), "of": len(ref_names), "median": float(np.median(e)),
                            "p90": float(np.percentile(e, 90)), "max": float(e.max())}
    return {"measured": True, "extent": ext, "per_seed": out}
