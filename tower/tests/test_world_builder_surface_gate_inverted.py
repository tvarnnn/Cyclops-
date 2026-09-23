"""The surface stage refuses depth fits that would invert depth.

A depth fit `pred ~= a * z + b` with a <= 0 turns near into far. Its held-out
score is None exactly then (`dense._relative_residual` refuses a <= 0), and
`surface_pipeline._Frames` used to read None as "unscored, admit at the weight
floor". On the 2026-09-23 target walk that fused 24 inverted frames: sliver
sheets, and a 2x voxel coarsening of the whole world.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP
from tower.world_builder.dense import align_frame

CAMERA = {"fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 24.0, "width": 64, "height": 48}


def _pose(x):
    return {"rotation": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], "translation": [x, 0.0, 0.0],
            "component": 0}


def _world(tmp_path, records):
    """A work dir holding a prediction for every record, and a solution with a
    pose for each: only the fit decides admission."""
    work = tmp_path / "work"
    (work / "depth").mkdir(parents=True)
    poses = {}
    for r in records:
        np.save(work / "depth" / f"{r['ki']:05d}.npy", np.full((48, 64), 1.5, np.float16))
        poses[r["kid"]] = _pose(0.01 * r["ki"])
    solution = SimpleNamespace(poses=poses, camera=CAMERA)
    align = {"kind": "depth", "camera": CAMERA, "records": records}
    return SP._Frames(align, work, solution, S.SurfaceParams())


def _rec(ki, a, b, ho, ok=True):
    return {"ki": ki, "kid": f"s:{ki:08d}", "ok": ok, "a": a, "b": b, "held_out_rel": ho,
            "z_sparse_max": 3.0}


def test_an_inverted_fit_is_refused_and_a_legitimate_one_admitted(tmp_path):
    frames = _world(tmp_path, [
        _rec(0, 0.78, 0.12, 0.025),      # a desk frame: physical, scored, passes
        _rec(1, -0.048, 1.43, None),     # bathroom ki 325 of the target: inverted
        _rec(2, 0.52, 0.40, 0.30),       # physical but poorly aligned: the held-out gate
    ])
    kept = [item[0] for item in frames.items]
    assert kept == [0]
    assert frames.refused_inverted == [1]
    assert SP._outlier_record(frames)["frames_refused_inverted_fit"] == 1


def test_an_unscorable_fit_is_refused_like_the_dense_stage_does(tmp_path):
    """With min_sparse_points 20 a held-out split always exists, so a None
    held-out score means the held-out half's own fit was not physical. On
    2f447162 the five such frames with a full-fit a > 0 had in-sample errors of
    0.07-18.6 and depth stretched up to 1200x (ki 121: a = 4.7e-4)."""
    frames = _world(tmp_path, [_rec(0, 0.78, 0.12, 0.025), _rec(1, 4.7e-4, 1.3, None),
                               _rec(2, 0.61, 0.20, 0.079)])   # just under the gate: kept
    assert [item[0] for item in frames.items] == [0, 2]
    assert frames.refused_inverted == [] and frames.refused_unscorable == [1]
    rec = SP._outlier_record(frames)
    assert rec["frames_refused_inverted_fit"] == 0 and rec["frames_refused_unscorable_fit"] == 1


def test_a_scored_inverted_fit_is_refused_too(tmp_path):
    """An inverted full fit whose held-out half happened to score. The
    known-good control's ki 196 had a = -0.038 with a held-out error of 0.033,
    under the gate, and the old code fused it inverted at weight 0.59; the
    slope, not the score, decides whether depth can be fused at all."""
    frames = _world(tmp_path, [_rec(0, 0.78, 0.12, 0.025), _rec(1, -0.038, 1.20, 0.033)])
    assert [item[0] for item in frames.items] == [0]
    assert frames.refused_inverted == [1] and frames.refused_unscorable == []


def test_the_min_sparse_points_floor_guarantees_a_held_out_split():
    """The premise above: at the depth stage's floor of anchors, both halves
    of `align_frame`'s split reach its own 10-anchor minimum."""
    from tower.world_builder.dense import DenseParams
    n = DenseParams().min_sparse_points
    idx = np.arange(n)
    assert (idx % 2 == 0).sum() >= 10 and (idx % 2 == 1).sum() >= 10


@pytest.mark.parametrize("a,b", [(0.0, 1.0), (-1e-9, 0.5), (math.nan, 0.1), (0.5, math.inf),
                                 (None, 0.1), ("x", 0.1)])
def test_no_non_physical_fit_passes(a, b):
    assert not SP.fit_is_physical(a, b)


def test_physical_fits_pass():
    assert SP.fit_is_physical(0.78, 0.12)
    assert SP.fit_is_physical(1e-6, -3.0)


def test_the_real_mechanism_negative_slope_means_no_held_out_score():
    """What the depth stage writes for an inverted frame: a <= 0 and
    held_out_rel None -- the pair the old gate admitted."""
    rng = np.random.default_rng(0)
    z = rng.uniform(1.0, 3.0, 200)
    pred = 2.0 - 0.3 * z + rng.normal(0, 0.01, 200)   # prediction FALLS with distance
    a, b, ho = align_frame(pred, z, "depth")
    assert a < 0 and ho is None
    assert not SP.fit_is_physical(a, b)


def test_the_gate_is_in_the_params_digest():
    """A non-forced build must not answer "already built" with a surface fused
    under the old gate. (Forced product builds rebuild anyway, and deploying
    the gate does not refresh existing worlds.)"""
    assert SP.FIT_GATE_ID == "fit-gate:a>0+scored"
    src = open(SP.__file__, encoding="utf-8").read()
    assert 'pdigest += "|" + FIT_GATE_ID' in src
