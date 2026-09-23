"""Placement between image islands, tilt, and harness-side triangulated depth.

Three measurements that do not rely on anything a variant chooses to publish
beyond its poses:

ISLANDS. The image-only verified pair graph (`eval_pairs`) is split into
connected components -- "islands" -- by construction of the capture: on the
target world the desk, the closet and the 265-382 block share no verified
pair. Where one island sits relative to another is then UNOBSERVABLE by any
pair metric, and a variant that misplaces a whole island scores the same as
one that places it right. `island_report` makes that explicit: it lists the
islands, the verified pairs (by tier) between every pair of large islands,
the variant's errors on those pairs where they exist, and says
"UNOBSERVED" where they do not.

TILT. People hold their heads roughly level (small roll), so a camera's x axis
(right) is nearly horizontal whatever its pitch and yaw. The "roll-free up" of
a set of cameras is therefore the direction most perpendicular to all their x
axes: the smallest-eigenvalue eigenvector of sum(x x^T), with its sign taken
from the mean camera-up (-y). It is pitch-invariant; it is ill-conditioned
when all x axes are parallel (no yaw spread), which is measured
(`conditioning` = second / smallest eigenvalue) and turned into an angular
uncertainty (roll scatter / yaw spread). The reference up is the main
island's (the whole main component's if that is ill-conditioned). Per island:
TILT = |median over its cameras of asin(x . up_ref)| -- the roll the island's
cameras would need under the reference up. A correctly placed island reads
about the level-head noise (the control world's islands: <= 2.4 deg); an
island misrotated by t about an axis perpendicular to its (concentrated) camera
x axes reads about t. It is BLIND to a misrotation about the vertical (no
gravity reference); for an island with a WIDE yaw spread a tilt makes the rolls
sinusoidal in yaw, so it shows in the roll scatter (`roll_under_reference_mad_deg`)
rather than in the median. A full level fit (`island_tilt`) is kept as
informative only: it amplifies head roll that correlates with yaw.
No room-specific constant is involved.

TRIANGULATED DEPTH (poses-only variants). The cached pair inliers (normalised
canonical coordinates, image-only) are triangulated with the VARIANT's
relative pose; each triangulated point gives a camera-frame depth in the
variant's units in both keyframes, and log(z / z_mono) against the cached
MoGe depth at that pixel is a scale sample. Inliers whose Sampson error under
the variant's relative pose exceeds ``tri_sampson_gate_px`` (the variant
disagrees with the images there) or whose triangulation angle is below
``tri_min_angle_deg`` are not used. The variant cannot curate these points:
they are the same image correspondences for every variant.
"""

from __future__ import annotations

import math

import numpy as np

PLACEMENT_PARAMS = {
    "island_min_kf": 10,
    "tilt_min_cameras": 5,
    "tilt_min_conditioning": 5.0,
    "tilt_trim_fraction": 0.1,
    "tilt_prior_deg": 5.0,
    "tri_sampson_gate_px": 2.0,
    "tri_min_angle_deg": 2.0,
    "tri_max_angle_deg": 90.0,
}


# ---------------------------------------------------------------------------
# islands


def island_labels(n: int, arrays_list) -> np.ndarray:
    """Islands over the union of the given pair arrays (each with i, j and an
    optional boolean `strict`; only strict pairs link)."""
    from tower.world_builder.coherence_eval.eval_pairs import islands

    I, J = [], []
    for A in arrays_list:
        if A is None or not len(A["i"]):
            continue
        sel = np.asarray(A["strict"], bool) if "strict" in A else np.ones(len(A["i"]), bool)
        I.append(np.asarray(A["i"])[sel])
        J.append(np.asarray(A["j"])[sel])
    if not I:
        return np.arange(n)
    return islands(n, np.concatenate(I), np.concatenate(J))


def _ranges(idx) -> list[list[int]]:
    out: list[list[int]] = []
    for i in sorted(int(x) for x in idx):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def roll_free_up(Rwc: np.ndarray) -> dict | None:
    """Robust up direction of a set of cameras (T_world_camera rotations, OpenCV
    axes) under the level-head assumption; see the module docstring."""
    P = PLACEMENT_PARAMS
    Rwc = np.asarray(Rwc, dtype=np.float64).reshape(-1, 3, 3)
    if len(Rwc) < P["tilt_min_cameras"]:
        return None
    x = Rwc[:, :, 0]
    cam_up = -Rwc[:, :, 1]
    keep = np.ones(len(x), bool)
    u = None
    ev = None
    for _ in range(2):
        S = x[keep].T @ x[keep]
        ev, V = np.linalg.eigh(S)
        u = V[:, 0]
        if u @ cam_up[keep].mean(0) < 0:
            u = -u
        # trim the cameras least consistent with a level head (largest |x.u|)
        dev = np.abs(x @ u)
        k = int(math.floor(P["tilt_trim_fraction"] * len(x)))
        if k == 0:
            break
        keep = np.ones(len(x), bool)
        keep[np.argsort(-dev, kind="stable")[:k]] = False
    cond = float(ev[1] / max(ev[0], 1e-12))
    # small-angle uncertainty of u about its weakest direction: the roll
    # scatter over the RMS component of the x axes along that direction
    nk = int(keep.sum())
    rolls = np.arcsin(np.clip(x[keep] @ u, -1, 1))
    sig = 1.4826 * float(np.median(np.abs(rolls - np.median(rolls)))) + 1e-6
    s_mid = math.sqrt(max(float(ev[1]), 1e-12) / max(nk, 1))
    unc = float(np.degrees(min(sig / s_mid, math.pi / 2)))
    mean_up = cam_up.mean(0)
    mean_up /= np.linalg.norm(mean_up) + 1e-15
    roll = np.degrees(np.arcsin(np.clip(np.abs(x @ u), 0, 1)))
    return {"up": u, "conditioning": cond, "well_conditioned": cond >= P["tilt_min_conditioning"],
            "up_uncertainty_deg": unc,
            "mean_camera_up": mean_up, "median_roll_deg": float(np.median(roll)), "cameras": int(len(x))}


def _angle(a, b) -> float:
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), float(np.dot(a, b)))))


def _rodrigues(w) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = np.asarray(w, dtype=np.float64) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def island_tilt(Rwc: np.ndarray, u_ref: np.ndarray) -> dict | None:
    """Tilt of a set of cameras against a reference up: ONE estimator for every
    island (review V2, B1).

    The island's up is the unit u minimising mean((x_i . u)^2) + prior |w|^2,
    u = rot(w) u_ref with w perpendicular to u_ref and prior =
    sin^2(`tilt_prior_deg`): along directions in which the island's camera x
    axes spread (a yaw spread), the level-head data decide; along directions
    in which they do not, u stays at the reference -- so a concentrated-yaw
    island reports the tilt it can show (its roll), and a well-spread one its
    full tilt. The worst `tilt_trim_fraction` of cameras (largest |x.u|) are
    trimmed once. Returns the tilt angle and the observability of the two
    tilt directions (their data curvature over the prior)."""
    from scipy.optimize import minimize

    P = PLACEMENT_PARAMS
    Rwc = np.asarray(Rwc, dtype=np.float64).reshape(-1, 3, 3)
    if len(Rwc) < P["tilt_min_cameras"]:
        return None
    u_ref = np.asarray(u_ref, dtype=np.float64) / np.linalg.norm(u_ref)
    x = Rwc[:, :, 0]
    a = np.cross(u_ref, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 0.1:
        a = np.cross(u_ref, [0.0, 1.0, 0.0])
    e1 = a / np.linalg.norm(a)
    e2 = np.cross(u_ref, e1)
    prior = math.sin(math.radians(P["tilt_prior_deg"])) ** 2
    keep = np.ones(len(x), bool)
    res = None
    for _ in range(2):
        xs = x[keep]

        def f(q, xs=xs):
            u = _rodrigues(q[0] * e1 + q[1] * e2) @ u_ref
            return float(np.mean((xs @ u) ** 2) + prior * (q[0] ** 2 + q[1] ** 2))

        res = minimize(f, np.zeros(2), method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 4000})
        u = _rodrigues(res.x[0] * e1 + res.x[1] * e2) @ u_ref
        k = int(math.floor(P["tilt_trim_fraction"] * len(x)))
        if k == 0:
            break
        keep = np.ones(len(x), bool)
        keep[np.argsort(-np.abs(x @ u), kind="stable")[:k]] = False
    u = _rodrigues(res.x[0] * e1 + res.x[1] * e2) @ u_ref
    xs = x[keep]
    curv = [float(np.mean((xs @ e) ** 2)) for e in (e1, e2)]
    roll = np.degrees(np.arcsin(np.clip(x @ u, -1, 1)))
    return {"up": u, "tilt_deg": _angle(u, u_ref),
            "observability": [c / prior for c in curv],
            "residual_roll_mad_deg": float(1.4826 * np.median(np.abs(roll - np.median(roll)))),
            "cameras": int(len(x))}


def island_report(p, pairs, rot_angle_deg, vec_angle_deg, t_min_parallax_deg: float) -> dict:
    """Islands of the image-only pair graph and what the variant does between them.

    `p` is the harness's Prepared view of the variant; `pairs` the world's PairSet."""
    P = PLACEMENT_PARAMS
    n = p.n
    A0 = pairs.arrays
    X = pairs.xarrays if getattr(pairs, "xarrays", None) is not None else None
    Lf = pairs.larrays if getattr(pairs, "larrays", None) is not None else None
    lab = island_labels(n, [A0])
    lab_all = island_labels(n, [A0, X, Lf])
    sizes = np.bincount(lab)
    big = [int(c) for c in np.nonzero(sizes >= P["island_min_kf"])[0]]
    islands_out = []
    for c in big:
        idx = np.nonzero(lab == c)[0]
        pub = idx[p.published[idx]]
        comps = {}
        for i in pub:
            comps[str(p.comp[i])] = comps.get(str(p.comp[i]), 0) + 1
        islands_out.append({"island": c, "keyframes": int(len(idx)), "ranges": _ranges(idx)[:12],
                            "published": int(len(pub)), "in_main": int(p.in_main[idx].sum()),
                            "variant_components": dict(sorted(comps.items(), key=lambda kv: -kv[1])),
                            "split_by_variant": len(comps) > 1})
    # cross-island pairs by tier
    tiers = []
    if X is not None:
        tiers.append(("sift_strict", X, np.asarray(X["strict"], bool)))
        tiers.append(("sift_relaxed_only", X, ~np.asarray(X["strict"], bool)))
    if Lf is not None:
        tiers.append(("loftr_strict", Lf, np.ones(len(Lf["i"]), bool)))
    pair_rows = []
    for k, a in enumerate(big):
        for b in big[k + 1:]:
            row = {"islands": [a, b], "pairs": {}, "variant": None}
            strict_I, strict_J, strict_R, strict_t, strict_rel, strict_par = [], [], [], [], [], []
            for name, T, sel in tiers:
                li, lj = lab[np.asarray(T["i"])], lab[np.asarray(T["j"])]
                m = sel & (((li == a) & (lj == b)) | ((li == b) & (lj == a)))
                row["pairs"][name] = int(m.sum())
                if name.endswith("strict") and m.any():
                    strict_I.append(np.asarray(T["i"])[m])
                    strict_J.append(np.asarray(T["j"])[m])
                    strict_R.append(np.asarray(T["R"])[m])
                    strict_t.append(np.asarray(T["t"])[m])
                    strict_rel.append(np.asarray(T["t_reliable"])[m])
                    strict_par.append(np.asarray(T["parallax_deg"])[m])
            n_strict = sum(len(x) for x in strict_I)
            row["placement"] = "observed" if n_strict else "UNOBSERVED"
            if n_strict:
                I = np.concatenate(strict_I).astype(int)
                J = np.concatenate(strict_J).astype(int)
                R = np.concatenate(strict_R)
                t = np.concatenate(strict_t)
                rel = np.concatenate(strict_rel)
                par = np.concatenate(strict_par)
                same = p.published[I] & p.published[J] & (p.comp[I] == p.comp[J])
                re_, te_ = [], []
                for q in np.nonzero(same)[0]:
                    i, j = I[q], J[q]
                    R_ji = p.Rcw(j) @ p.Rcw(i).T
                    t_ji = p.Rcw(j) @ (p.C[i] - p.C[j])
                    re_.append(float(rot_angle_deg(R[q].T @ R_ji)))
                    nt = np.linalg.norm(t_ji)
                    if rel[q] and par[q] >= t_min_parallax_deg and nt > 0:
                        te_.append(float(vec_angle_deg(t_ji / nt, t[q])[0]))
                row["variant"] = {"strict_pairs": int(n_strict), "joined": int(same.sum()),
                                  "joined_fraction": float(same.mean()),
                                  "rot_err_median_deg": float(np.median(re_)) if re_ else None,
                                  "rot_err_max_deg": float(np.max(re_)) if re_ else None,
                                  "t_err_median_deg": float(np.median(te_)) if te_ else None}
            pair_rows.append(row)
    observed = sum(1 for r in pair_rows if r["placement"] == "observed")
    # tilt per island, relative to the main island, in the variant's main component
    tilt = {}
    ups = {}
    for c in big:
        idx = np.nonzero((lab == c) & p.in_main)[0]
        ups[c] = (roll_free_up(p.Rwc[idx]) if len(idx) else None, idx)
    ref, ref_src = (ups[big[0]][0], "main island") if big else (None, None)
    if ref is None or not ref["well_conditioned"]:
        allm = np.nonzero(p.in_main)[0]
        ref, ref_src = roll_free_up(p.Rwc[allm]) if len(allm) else None, "whole main component (main island ill-conditioned)"
    max_tilt = None
    for c in big:
        est, idx = ups[c]
        if ref is None or not len(idx) or len(idx) < P["tilt_min_cameras"]:
            tilt[str(c)] = {"available": False, "why": "fewer than "
                            f"{P['tilt_min_cameras']} of its keyframes in the variant's main component"}
            continue
        x = p.Rwc[idx][:, :, 0]
        roll = np.degrees(np.arcsin(np.clip(x @ ref["up"], -1, 1)))
        tl = island_tilt(p.Rwc[idx], ref["up"])
        row = {"cameras": int(len(idx)), "reference": ref_src,
               "tilt_deg": float(abs(np.median(roll))),
               "median_roll_under_reference_deg": float(np.median(roll)),
               "roll_under_reference_mad_deg": float(1.4826 * np.median(np.abs(roll - np.median(roll)))),
               "level_fit_tilt_deg_informative": tl["tilt_deg"]}
        if c != big[0]:
            max_tilt = row["tilt_deg"] if max_tilt is None else max(max_tilt, row["tilt_deg"])
        tilt[str(c)] = row
    n_big = len(big)
    return {
        "definition": "connected components of the image-only verified pair graph (base tier)",
        "islands": int(len(sizes)), "singletons": int((sizes == 1).sum()),
        "large_islands": n_big, "large_island_min_kf": P["island_min_kf"],
        "largest_island_share": float(sizes.max() / n) if n else None,
        "large_island_list": islands_out,
        "islands_with_cross_tiers": int(len(np.unique(lab_all))),
        "large_island_pairs": len(pair_rows),
        "large_island_pairs_observed": observed,
        "large_island_pairs_unobserved": len(pair_rows) - observed,
        "cross_island_tiers_available": [t[0] for t in tiers],
        "island_pairs": pair_rows,
        "islands_split_by_variant": int(sum(1 for r in islands_out if r["split_by_variant"])),
        "tilt_definition": (
            "tilt = |median over the island's cameras of asin(x_axis . up_ref)|: the roll its cameras would "
            "need under up_ref, the roll-free up of the main island (level-head assumption). Blind to a "
            "misrotation about the vertical; for an island with a WIDE yaw spread a tilt makes the rolls "
            "sinusoidal in yaw, so it shows in roll_under_reference_mad_deg rather than in the median. "
            "level_fit_tilt_deg_informative (eval_placement.island_tilt) fits the full tilt but amplifies "
            "yaw-correlated head roll (26 deg on a control-world island whose median roll is 2 deg)"),
        "tilt": tilt,
        "reference_tilt_deg": (tilt.get(str(big[0])) or {}).get("tilt_deg") if big else None,
        "max_tilt_vs_main_island_deg": max_tilt,
        "_labels": lab,
    }


# ---------------------------------------------------------------------------
# harness-side triangulation


def triangulate_midpoint(x1, x2, R, t):
    """Depths (z1, z2) of normalised correspondences under x2 ~ R x1 + t
    (least-squares midpoint along the two rays) and the angle between the rays."""
    f1 = np.c_[x1, np.ones(len(x1))]
    f2 = np.c_[x2, np.ones(len(x2))]
    a = f1 @ R.T
    b = f2
    aa = (a * a).sum(1)
    bb = (b * b).sum(1)
    ab = (a * b).sum(1)
    at = a @ t
    bt = b @ t
    det = aa * bb - ab * ab
    with np.errstate(divide="ignore", invalid="ignore"):
        l1 = (-at * bb + ab * bt) / det
        l2 = (aa * bt - ab * at) / det
        ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), ab))
    return l1, l2, ang


def sampson_px_each(R, t, x1, x2, f):
    t = t / (np.linalg.norm(t) + 1e-15)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    a = np.c_[x1, np.ones(len(x1))]
    b = np.c_[x2, np.ones(len(x2))]
    Ea = (E @ a.T).T
    Etb = (E.T @ b.T).T
    num = (b * Ea).sum(1) ** 2
    den = Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-30)) * f


def triangulated_depth_ratios(p, arrays, depth_sample, K, *, min_samples: int, valid_m) -> dict:
    """Per keyframe: median log(z_triangulated / z_mono) over the cached pair
    inliers, triangulated with the variant's relative poses.

    depth_sample(i, uv_canonical) -> metric depth (NaN where unknown)."""
    P = PLACEMENT_PARAMS
    n = p.n
    f = float(K[0, 0])
    I = np.asarray(arrays["i"]).astype(int)
    J = np.asarray(arrays["j"]).astype(int)
    offs = np.asarray(arrays["inlier_offsets"])
    XI = np.asarray(arrays["inlier_xy_i"], dtype=np.float64)
    XJ = np.asarray(arrays["inlier_xy_j"], dtype=np.float64)
    per_kf_xy: list[list[np.ndarray]] = [[] for _ in range(n)]
    per_kf_z: list[list[np.ndarray]] = [[] for _ in range(n)]
    used_pairs = 0
    gated = 0
    total = 0
    for k in range(len(I)):
        i, j = I[k], J[k]
        if not (p.published[i] and p.published[j] and p.comp[i] == p.comp[j]):
            continue
        a, b = int(offs[k]), int(offs[k + 1])
        if b - a < 5:
            continue
        R_ji = p.Rcw(j) @ p.Rcw(i).T
        t_ji = p.Rcw(j) @ (p.C[i] - p.C[j])
        if not np.isfinite(t_ji).all() or np.linalg.norm(t_ji) <= 0:
            continue
        x1, x2 = XI[a:b], XJ[a:b]
        total += len(x1)
        s = sampson_px_each(R_ji, t_ji, x1, x2, f)
        z1, z2, ang = triangulate_midpoint(x1, x2, R_ji, t_ji)
        ok = (s <= P["tri_sampson_gate_px"]) & (ang >= P["tri_min_angle_deg"]) & (ang <= P["tri_max_angle_deg"])
        ok &= np.isfinite(z1) & np.isfinite(z2) & (z1 > 0) & (z2 > 0)
        gated += int((~ok).sum())
        if not ok.any():
            continue
        used_pairs += 1
        per_kf_xy[i].append(x1[ok])
        per_kf_z[i].append(z1[ok])
        per_kf_xy[j].append(x2[ok])
        per_kf_z[j].append(z2[ok])
    lo, hi = valid_m
    r = np.full(n, np.nan)
    cnt = np.zeros(n, int)
    spread = np.full(n, np.nan)
    for i in range(n):
        if not per_kf_z[i]:
            continue
        xy = np.concatenate(per_kf_xy[i])
        z = np.concatenate(per_kf_z[i])
        uv = (K @ np.c_[xy, np.ones(len(xy))].T).T[:, :2]
        zm = depth_sample(i, uv)
        ok = np.isfinite(zm) & (zm > lo) & (zm < hi)
        if ok.sum() < min_samples:
            continue
        lr = np.log(z[ok] / zm[ok])
        r[i] = float(np.median(lr))
        cnt[i] = int(ok.sum())
        spread[i] = float(1.4826 * np.median(np.abs(lr - np.median(lr))))
    return {"r": r, "n": cnt, "spread": spread,
            "pairs_used": used_pairs, "inliers_total": total, "inliers_gated": gated}
