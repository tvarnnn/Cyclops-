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

numpy / scipy only; no pycolmap, no file IO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

INF = float("inf")


@dataclass(frozen=True)
class GateParams:
    # Minimum shared 3-D points that couple two cameras (a) and two groups (c).
    # Physics floor: a Sim(3) between two rigid bodies has 7 DOF and a 3-D
    # point seen by both gives 3 constraints, so 3 non-collinear points is the
    # minimum that fixes it at all. Control ceiling: the known-good control
    # b2a75ab4 (A0, 5 seeds) stays ONE rigid group for every K <= 20 and first
    # sheds a camera at K = 30. 10 = half that ceiling, >= 3x the physical
    # floor (RUN/experiments/P2-E1/GATE-THRESHOLDS.md).
    k_shared: int = 10
    # Maximum seed spread (fraction of component extent) of a coupled group.
    # Control noise floor: its genuinely coupled parts (capture-order blocks,
    # K=50 sub-groups) move <= 0.0014 of extent between seeds (whole main
    # component: median 0.0002). 0.02 is ~14x that floor, and on a 5-8 m walk
    # ~0.10-0.16 m -- inside the harness's own 0.3 m eye-height tolerance.
    max_spread: float = 0.02
    # global_solve.MIN_IMAGE_OBSERVATIONS: fewer observations is not a pose.
    min_obs: int = 30
    # A group counts as present in a seed's reference component when at least
    # this fraction of its cameras is registered there.
    min_present_frac: float = 0.5
    # Fewer common reference cameras than this and a seed cannot be aligned.
    min_align_cameras: int = 5

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


def apply_rigid_gate(models: list[SeedModel], params: GateParams | None = None) -> dict:
    """Final component per camera of the reference seed (models[0]).

    Returns {"labels": {name: int}, "components": [...], "rounds": [...],
    "params": ...}. Component labels are renumbered by supported size
    (0 = most supported cameras)."""
    params = params or GateParams()
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
    # renumber by supported size
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
