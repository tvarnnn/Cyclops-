"""The dense stage: what it recovers, and what it refuses to invent.

These tests need no GPU, no network and no depth model. They build a synthetic
scene with a known camera, a known surface and a known depth map, so that every
claim the dense stage makes about geometry can be checked against a right answer
that exists independently of it.

The refusals matter as much as the recoveries. A dense reconstruction that
quietly fills in a wall it never saw is worse for spatial memory than an honest
hole, so the tests that assert something is DROPPED are load-bearing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tower.world_builder.dense import (
    DENSE_FORMAT,
    DenseParams,
    align_frame,
    camera_centre,
    project,
    read_points_bin,
    robust_affine,
    unproject,
    validity_mask,
    voxel_reduce,
    write_points_bin,
)

K = np.array([[465.71872223, 0.0, 176.43702376],
              [0.0, 465.05395439, 322.13758708],
              [0.0, 0.0, 1.0]])
W, H = 359, 639


def _rot(yaw=0.0, pitch=0.0):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rx @ Ry


# ---------------------------------------------------------------------------
# geometry -- the convention the solve actually uses
# ---------------------------------------------------------------------------


def test_unproject_inverts_project_exactly():
    """The whole dense stage is one big unproject. If this drifts, everything does."""
    rng = np.random.default_rng(0)
    R, t = _rot(0.3, -0.2), np.array([0.4, -1.1, 2.5])
    uv = rng.uniform([0, 0], [W, H], size=(500, 2))
    depth = rng.uniform(0.5, 12.0, size=500)
    X = unproject(R, t, K, uv, depth)
    uv2, z2 = project(R, t, K, X)
    assert np.allclose(uv2, uv, atol=1e-9)
    assert np.allclose(z2, depth, atol=1e-9)


def test_camera_centre_is_the_point_that_projects_to_zero_depth():
    R, t = _rot(-0.7, 0.15), np.array([2.0, 0.3, -4.0])
    C = camera_centre(R, t)
    _, z = project(R, t, K, C[None, :])
    assert abs(float(z[0])) < 1e-9


def test_project_marks_points_behind_the_camera_rather_than_folding_them_in_front():
    """A point behind the camera must not come back as a plausible pixel."""
    R, t = np.eye(3), np.zeros(3)
    X = np.array([[0.0, 0.0, -3.0]])
    uv, z = project(R, t, K, X)
    assert z[0] < 0
    assert np.isnan(uv).all()


# ---------------------------------------------------------------------------
# alignment -- where scale comes from
# ---------------------------------------------------------------------------


def test_robust_affine_recovers_a_known_transform():
    rng = np.random.default_rng(1)
    z = rng.uniform(0.8, 9.0, size=400)
    a_true, b_true = 2700.0, -215.0
    disp = a_true / z + b_true
    a, b = robust_affine(disp, 1.0 / z)
    assert a == pytest.approx(a_true, rel=1e-6)
    assert b == pytest.approx(b_true, abs=1e-6)


def test_robust_affine_survives_gross_outliers():
    """A few sparse points land on a reflection or a moving hand. Plain least
    squares lets one of those ruin a whole frame; the Huber weight must not."""
    rng = np.random.default_rng(2)
    z = rng.uniform(0.8, 9.0, size=400)
    a_true, b_true = 2700.0, -215.0
    disp = a_true / z + b_true
    disp[:40] += rng.uniform(4000, 9000, size=40)          # 10% wild outliers
    a, b = robust_affine(disp, 1.0 / z)
    assert a == pytest.approx(a_true, rel=0.02)
    assert b == pytest.approx(b_true, abs=25.0)

    lsq = np.linalg.lstsq(np.stack([1.0 / z, np.ones_like(z)], 1), disp, rcond=None)[0]
    assert abs(lsq[0] - a_true) > abs(a - a_true)          # and it beats plain least squares


def test_align_frame_reports_a_near_zero_held_out_residual_on_consistent_depth():
    rng = np.random.default_rng(3)
    z = rng.uniform(1.0, 8.0, size=300)
    disp = 2400.0 / z - 180.0
    a, b, held_out = align_frame(disp, z)
    assert a > 0
    assert held_out is not None
    assert held_out < 1e-6


def test_align_frame_reports_a_large_held_out_residual_when_the_depth_disagrees():
    """The gate exists to catch exactly this: a frame whose predicted shape does
    not match the geometry the solve already established."""
    rng = np.random.default_rng(4)
    z = rng.uniform(1.0, 8.0, size=300)
    disp = 2400.0 / z - 180.0
    disp += rng.normal(0, 400.0, size=300)                 # shape genuinely wrong
    _, _, held_out = align_frame(disp, z)
    assert held_out is not None
    assert held_out > DenseParams().gate_rel


def test_align_frame_holds_out_points_it_never_fitted():
    """If the score were in-sample it would be optimistic on noise. Adding noise
    must move the held-out number, which is what proves the split is real."""
    rng = np.random.default_rng(5)
    z = rng.uniform(1.0, 8.0, size=400)
    clean = 2400.0 / z - 180.0
    _, _, ho_clean = align_frame(clean, z)
    _, _, ho_noisy = align_frame(clean + rng.normal(0, 60.0, size=400), z)
    assert ho_noisy > ho_clean


# ---------------------------------------------------------------------------
# the validity mask -- refusing pixels that are not evidence
# ---------------------------------------------------------------------------


def _plane_depth(distance=3.0):
    return np.full((H, W), distance, np.float32)


def test_validity_mask_keeps_a_flat_wall_facing_the_camera():
    ok = validity_mask(_plane_depth(), K, edge_rel=0.03, max_grazing_deg=80.0, erode_px=1)
    inner = ok[40:-40, 40:-40]
    assert inner.mean() > 0.99


def test_validity_mask_rejects_the_pixels_across_a_depth_discontinuity():
    """Flying pixels: a pixel straddling an occlusion boundary gets a depth that
    belongs to neither surface and back-projects into empty space.

    A central-difference gradient flags the two columns either side of the step;
    eroding by one widens that to four. Those four are the pixels whose depth is
    actually contaminated, so those four are what must go -- and the surface a
    few pixels away must survive, or the mask would be eating real evidence."""
    z = _plane_depth(3.0)
    z[:, W // 2:] = 6.0
    ok = validity_mask(z, K, edge_rel=0.03, max_grazing_deg=80.0, erode_px=1)
    assert not ok[:, W // 2 - 2: W // 2 + 2].any()
    assert ok[100:200, 20:120].mean() > 0.95        # away from the edge, kept


def _tilted_plane_depth(tilt_deg: float, centre_depth: float = 3.0):
    """Depth map of a plane whose normal is tilted `tilt_deg` from the optical axis.

    n = (sin t, 0, cos t) and n . P = d, with P = (xn z, yn z, z), so
    z = d / (sin t * xn + cos t). Choosing d = centre_depth * cos t puts the
    plane at a fixed distance straight ahead whatever the tilt, so the two
    parameterised cases differ only in incidence and not in range.
    """
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xn = (uu - K[0, 2]) / K[0, 0]
    t = np.deg2rad(tilt_deg)
    den = np.sin(t) * xn + np.cos(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (centre_depth * np.cos(t)) / den
    return np.where((den > 1e-3) & (z > 0.2) & (z < 60.0), z, np.nan).astype(np.float32)


@pytest.mark.parametrize("tilt_deg,expect_kept", [(10.0, True), (88.0, False)])
def test_validity_mask_keeps_a_face_on_surface_and_rejects_a_grazing_one(tilt_deg, expect_kept):
    """Depth error explodes at grazing incidence, so those pixels are not evidence.

    Measured in a small window around the principal point, where the ray is
    (0, 0, 1) and the incidence angle therefore IS the tilt. Across the whole
    frame it is not: this camera is 359x639 with a short focal length, so a
    single plane spans a wide range of incidence angles and a frame-wide
    average would be testing the field of view rather than the mask.

    Parameterised deliberately -- asserting only that grazing is rejected would
    also pass if the mask rejected everything, which would be a silent disaster
    for coverage. The face-on case is what rules that out.
    """
    z = _tilted_plane_depth(tilt_deg)
    ok = validity_mask(z, K, edge_rel=1e9, max_grazing_deg=80.0, erode_px=0)
    # Just right of the principal point: at 88 degrees the plane sweeps to the
    # horizon a few pixels LEFT of centre, so a symmetric window would be half
    # empty. Here xn stays within 0.01-0.08 and the incidence stays within a
    # couple of degrees of the tilt.
    cy, cx = int(K[1, 2]), int(K[0, 2])
    win = (slice(cy - 20, cy + 20), slice(cx + 5, cx + 35))
    finite = np.isfinite(z[win])
    assert finite.mean() > 0.9, "the synthetic plane must be visible in the window"
    kept = ok[win][finite].mean()
    if expect_kept:
        assert kept > 0.9
    else:
        assert kept < 0.05


def test_validity_mask_rejects_non_finite_depth():
    z = _plane_depth()
    z[10:20, 10:20] = np.nan
    ok = validity_mask(z, K, edge_rel=0.03, max_grazing_deg=80.0, erode_px=0)
    assert not ok[10:20, 10:20].any()


# ---------------------------------------------------------------------------
# voxel reduction and the wire format
# ---------------------------------------------------------------------------


def test_voxel_reduce_averages_position_and_colour_and_keeps_the_best_confidence():
    X = np.array([[0.001, 0.001, 0.001], [0.003, 0.003, 0.003], [5.0, 5.0, 5.0]])
    C = np.array([[0, 0, 0], [100, 100, 100], [7, 7, 7]], np.uint8)
    F = np.array([2, 9, 4], np.uint8)
    Xv, Cv, Fv = voxel_reduce(X, C, F, 0.05)
    assert len(Xv) == 2
    merged = np.argmin(np.linalg.norm(Xv, axis=1))
    assert Xv[merged] == pytest.approx([0.002, 0.002, 0.002], abs=1e-6)
    assert Cv[merged].tolist() == [50, 50, 50]
    assert Fv[merged] == 9                       # the best evidence, not the mean


def test_voxel_reduce_handles_negative_coordinates():
    """The voxel key packs signed coordinates; a room straddling the origin must
    not alias two distant points onto one key."""
    X = np.array([[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]])
    C = np.zeros((2, 3), np.uint8)
    F = np.ones(2, np.uint8)
    Xv, _, _ = voxel_reduce(X, C, F, 0.02)
    assert len(Xv) == 2


def test_points_bin_round_trips_through_the_wire_format(tmp_path):
    rng = np.random.default_rng(6)
    X = rng.uniform(-5, 5, size=(1000, 3)).astype(np.float32)
    C = rng.integers(0, 256, size=(1000, 3)).astype(np.uint8)
    F = rng.integers(2, 40, size=1000).astype(np.uint8)
    p = tmp_path / "points_l0.bin"
    size = write_points_bin(p, X, C, F)
    assert size == 16 * 1000                     # the stride the viewer relies on
    X2, C2, F2 = read_points_bin(p)
    assert np.array_equal(X, X2)
    assert np.array_equal(C, C2)
    assert np.array_equal(F, F2)


def test_points_bin_is_little_endian_regardless_of_host():
    """The viewer decodes with a DataView that assumes little-endian."""
    from tower.world_builder.dense import POINT_DTYPE

    assert POINT_DTYPE["x"].byteorder in ("<", "=")
    assert POINT_DTYPE.itemsize == 16


# ---------------------------------------------------------------------------
# parameters and honesty
# ---------------------------------------------------------------------------


def test_default_params_require_agreement_from_more_than_one_other_camera():
    """Two cameras can agree by coincidence along a shared ray. The default must
    not accept geometry on that basis."""
    assert DenseParams().min_views >= 3


def test_default_backend_is_permissively_licensed():
    """Depth Anything V2 Base and Large are CC-BY-NC. A non-commercial weight
    must not be reachable as a default in a product."""
    from tower.world_builder.dense import available_backends, make_backend

    backend = make_backend(DenseParams().backend)
    assert "NC" not in backend.licence
    assert backend.licence in {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"}
    for name in available_backends():
        assert "NonCommercial" not in make_backend(name).licence


def test_unknown_backend_is_refused_by_name():
    from tower.world_builder.dense import DenseUnavailable, make_backend

    with pytest.raises(DenseUnavailable):
        make_backend("no-such-model")


def test_canonical_level_is_not_the_finest_level():
    """0.23 MP imagery gives 1-3 cm of depth noise. The finest voxel stores that
    noise; the canonical level must sit at the resolution the evidence supports."""
    p = DenseParams()
    assert p.canonical_level > 0
    assert p.lod_voxels[p.canonical_level] > p.lod_voxels[0]
    assert p.mobile_level >= p.canonical_level
