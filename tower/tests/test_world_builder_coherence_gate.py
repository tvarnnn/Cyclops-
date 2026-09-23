"""The coherence gate (world_builder/coherence_gate.py): attach a piece of the solve to the room only on
independent, consistent evidence, and say why a piece was not attached, in the contract's words.

The rule is ported unchanged from the experiment gate (coherence_exp/gate.py, rule="evidence", honoured
links, no stability test); the port reproduces that gate's partition on all 24 ground-truth cases of run
wb-coherence-run-2026-09-23 (experiments/P2-LM/gatefix/port_check.py). These tests pin its behaviour on
synthetic solves.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tower.world_builder import coherence_gate as CG

N_A, N_B = 30, 20


def _rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])


def solve_model(*, components=None, shared_between=60):
    """Island A (cameras 0..29) and island B (30..49), every camera seeing tracks of length 3 with its
    neighbours; `shared_between` points each seen by one A and one B camera. Rotations vary per camera so
    relative rotations are not all identity."""
    n = N_A + N_B
    names = [f"{i:06d}_k.jpg" for i in range(n)]
    R_cw = np.stack([_rz(3.0 * i) for i in range(n)])
    obs_i, obs_p = [], []
    p = 0
    for island in (range(0, N_A), range(N_A, n)):
        idx = list(island)
        for j in range(len(idx)):
            for _ in range(13):
                for c in idx[j:j + 3]:
                    obs_i.append(c)
                    obs_p.append(p)
                p += 1
    for j in range(shared_between):
        for c in (j % N_A, N_A + j % N_B):
            obs_i.append(c)
            obs_p.append(p)
        p += 1
    for c in range(n):  # private points: every camera clears the 30-observation floor, island ends included
        for _ in range(30):
            obs_i.append(c)
            obs_p.append(p)
            p += 1
    obs_i, obs_p = np.asarray(obs_i), np.asarray(obs_p)
    n_obs = np.bincount(obs_i, minlength=n)
    comp = np.zeros(n, dtype=np.int64) if components is None else np.asarray(components)
    return CG.SolveModel(names=names, component=comp, R_cw=R_cw, n_obs=n_obs, obs_image=obs_i, obs_point=obs_p,
                         n_points=p)


def ladder(cross=(), inliers=100):
    names = [f"{i:06d}_k.jpg" for i in range(N_A + N_B)]
    out = {}
    for lo, hi in ((0, N_A), (N_A, N_A + N_B)):
        for i in range(lo, hi):
            for j in (i + 1, i + 2):
                if j < hi:
                    out[(names[i], names[j])] = inliers
    for a, b, n in cross:
        out[tuple(sorted((names[a], names[b])))] = n
    return out


def rotations(model, links, off_deg=0.0, off=()):
    """Each link's own rotation, R_b_from_a, as the solve has it -- or `off_deg` away for pairs in `off`."""
    idx = model.index()
    out = {}
    for a, b in links:
        R = model.R_cw[idx[b]] @ model.R_cw[idx[a]].T
        out[(a, b)] = (_rz(off_deg) @ R) if (a, b) in off else R
    return out


def metric(level_a, level_b, seed=0):
    rng = np.random.default_rng(seed)
    v = np.r_[np.full(N_A, math.log(level_a)), np.full(N_B, math.log(level_b))] + rng.normal(0, 0.03, N_A + N_B)
    return {f"{i:06d}_k.jpg": float(x) for i, x in enumerate(v)}


# B hangs on A through the cut vertex 30, linked to 28 and 29: its own block, attachable by a closed triangle.
CUT_VERTEX = [(29, 30, 50), (28, 30, 50)]


def _gate(model, links, level_b=4.0, *, rots=None, masks=True, metric_log=None):
    return CG.apply_gate(model, links, metric(4.0, level_b) if metric_log is None else metric_log,
                         link_rotations=rotations(model, links) if rots is None else rots, masks_applied=masks)


def _unplaced(out):
    return [c for c in out["components"] if c["state"] == "unplaced"]


def test_a_piece_on_redundant_honoured_links_at_the_rooms_scale_is_placed():
    m = solve_model()
    out = _gate(m, ladder(cross=CUT_VERTEX))
    assert [c["state"] for c in out["components"]] == ["placed"]
    assert out["components"][0]["reason"] is None and out["components"][0]["reasons"] == []


def test_contradicted_links_are_not_evidence():
    m = solve_model()
    links = ladder(cross=CUT_VERTEX)
    names = m.names
    cross = {tuple(sorted((names[a], names[b]))) for a, b, _ in CUT_VERTEX}
    out = _gate(m, links, rots=rotations(m, links, off_deg=60.0, off=cross))
    (piece,) = _unplaced(out)
    assert piece["reason"] == CG.REASON_LINK_CONTRADICTED
    # 16.8 deg is honoured: a link 10 deg off still counts
    out = _gate(m, links, rots=rotations(m, links, off_deg=10.0, off=cross))
    assert _unplaced(out) == []


def test_a_single_link_is_one_point_of_failure():
    m = solve_model()
    out = _gate(m, ladder(cross=[(29, 30, 50)]))
    (piece,) = _unplaced(out)
    assert piece["reason"] == CG.REASON_SINGLE_UNCONFIRMED_LINK


def test_a_piece_at_another_metric_level_is_not_placed():
    m = solve_model()
    out = _gate(m, ladder(cross=CUT_VERTEX), level_b=4.0 * 1.6)
    (piece,) = _unplaced(out)
    assert CG.REASON_SCALE_MISMATCH in piece["reasons"]


def test_a_piece_the_solver_returned_separately_says_so():
    m = solve_model(components=[0] * N_A + [1] * N_B)
    out = _gate(m, ladder())
    (piece,) = _unplaced(out)
    assert piece["reasons"] == [CG.REASON_SOLVED_SEPARATELY]


def test_without_masks_nothing_is_attached():
    """Masks are a hard dependency (manager 011): the same attachable piece stays out, and says why."""
    m = solve_model()
    out = _gate(m, ladder(cross=CUT_VERTEX), masks=False)
    (piece,) = _unplaced(out)
    assert piece["reasons"] == [CG.REASON_MASKS_UNAVAILABLE]
    assert out["masks_applied"] is False and out["evidence"]["attach"] is False


def test_without_any_metric_scale_nothing_is_attached():
    m = solve_model()
    out = _gate(m, ladder(cross=CUT_VERTEX), metric_log={})
    (piece,) = _unplaced(out)
    assert piece["reasons"] == [CG.REASON_SCALE_UNAVAILABLE]
    assert out["metric_available"] is False


def test_every_reason_is_in_the_contract_vocabulary():
    assert set(CG.REASONS) == {"masks-unavailable", "scale-unavailable", "solved-separately", "no-verified-link",
                               "single-unconfirmed-link", "scale-mismatch", "link-contradicted"}


def test_the_params_digest_records_the_bound():
    base = CG.GateParams()
    assert base.max_link_disagreement_deg == pytest.approx(16.8)
    assert base.digest() != CG.GateParams(max_link_disagreement_deg=25.2).digest()
    out = _gate(solve_model(), ladder(cross=CUT_VERTEX))
    assert out["params"]["max_link_disagreement_deg"] == pytest.approx(16.8)
    assert out["params_digest"] == base.digest() and out["gate"] == CG.GATE_ID


def test_labels_cover_every_camera_and_the_room_is_label_zero():
    m = solve_model()
    out = _gate(m, ladder(cross=[(29, 30, 50)]))
    assert set(out["labels"]) == set(m.names)
    room = [n for n, lab in out["labels"].items() if lab == 0]
    assert len(room) == out["components"][0]["cameras"] and len(room) >= N_A - 1
