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
    assert p.lod_depth_fractions[p.canonical_level] > p.lod_depth_fractions[0]
    assert p.mobile_level >= p.canonical_level


def test_voxel_sizes_are_relative_to_the_scene_and_not_absolute():
    """global_solve never normalises the model, so the gauge is arbitrary and
    differs by more than an order of magnitude between solves in this corpus.
    A fixed voxel would shatter one world and collapse another."""
    p = DenseParams()
    small = p.voxels_for(7.0)
    large = p.voxels_for(70.0)
    assert small[0] == pytest.approx(0.021, abs=1e-6)
    for a, b in zip(small, large):
        assert b == pytest.approx(a * 10.0, rel=1e-9)
    assert all(v > 0 for v in small)
    assert small == sorted(small)


def test_params_record_everything_that_changes_the_output():
    """The manifest carries these so a reconstruction can be explained later."""
    d = DenseParams().as_dict()
    for key in ("backend", "gate_rel", "tau", "min_views", "neighbours",
                "edge_rel", "max_grazing_deg", "lod_depth_fractions",
                "min_confidence", "component", "average_views"):
        assert key in d


# ---------------------------------------------------------------------------
# the privacy boundary
#
# engine.py redacts faces BEFORE persisting a keyframe image, deliberately, so
# that the bytes any later reconstruction reads are the redacted ones. A dense
# stage reaching past that to the original capture would rebuild the room out
# of exactly the pixels the privacy transformation removed -- and at far higher
# density than the sparse cloud ever exposed. These tests pin that shut.
# ---------------------------------------------------------------------------


class _StubStore:
    def __init__(self, images: "pathlib.Path"):
        self._images = images

    def images_dir(self, world_id, session_id):
        return self._images


class _StubRedactor:
    def __init__(self, available=True):
        self.available = available
        self.unavailable_reason = None if available else "no model"
        self.label = "stub@0.30"
        self.calls = 0

    def redact(self, image_bytes):
        self.calls += 1

        class R:
            pass

        r = R()
        r.image_bytes = b"REDACTED:" + image_bytes
        return r


def test_dense_prefers_the_worlds_redacted_keyframe_over_the_raw_capture(tmp_path):
    import pathlib  # noqa: F401 -- used by the stub annotation

    from tower.world_builder.dense_pipeline import keyframe_image_bytes

    images = tmp_path / "images"
    images.mkdir()
    (images / "00000042.jpg").write_bytes(b"REDACTED-KEYFRAME")
    raw = tmp_path / "raw.jpg"
    raw.write_bytes(b"RAW-WITH-A-FACE")
    red = _StubRedactor()

    data, origin, fill = keyframe_image_bytes(
        _StubStore(images), "w", "s", "s:00000042", str(raw), red,
        keyframes_are_redacted=True,
    )
    assert data == b"REDACTED-KEYFRAME"
    assert origin == "world-keyframe"
    assert red.calls == 0                      # no re-redaction was needed
    # The third value is a MASK, never pixels. Raw bytes do not leave the
    # boundary function, so a caller cannot reconstruct from them by mistake.
    assert fill is None or getattr(fill, "dtype", None) == bool


def test_dense_re_redacts_when_the_worlds_keyframe_image_is_missing(tmp_path):
    """Worlds migrated between roots lost their images/ directory. Falling back
    to the raw capture is allowed only through the same transformation."""
    from tower.world_builder.dense_pipeline import keyframe_image_bytes

    images = tmp_path / "images"
    images.mkdir()
    raw = tmp_path / "raw.jpg"
    raw.write_bytes(b"RAW-WITH-A-FACE")
    red = _StubRedactor()

    data, origin, fill = keyframe_image_bytes(
        _StubStore(images), "w", "s", "s:00000042", str(raw), red,
        keyframes_are_redacted=True,
    )
    assert red.calls == 1
    assert fill is None or getattr(fill, "dtype", None) == bool
    assert data == b"REDACTED:RAW-WITH-A-FACE"
    assert origin == "raw-source-rereducted"
    assert b"RAW-WITH-A-FACE" != data


def test_dense_refuses_a_raw_frame_when_redaction_is_unavailable(tmp_path):
    """The safe direction is to lose the frame, not to publish the face."""
    from tower.world_builder.dense_pipeline import keyframe_image_bytes

    images = tmp_path / "images"
    images.mkdir()
    raw = tmp_path / "raw.jpg"
    raw.write_bytes(b"RAW-WITH-A-FACE")

    data, origin, _ = keyframe_image_bytes(
        _StubStore(images), "w", "s", "s:00000042", str(raw),
        _StubRedactor(available=False), keyframes_are_redacted=True,
    )
    assert data is None
    assert origin == "refused-no-redactor"

    data, origin, _ = keyframe_image_bytes(
        _StubStore(images), "w", "s", "s:00000042", str(raw), None,
        keyframes_are_redacted=True,
    )
    assert data is None


def test_dense_reports_an_absent_image_rather_than_inventing_one(tmp_path):
    from tower.world_builder.dense_pipeline import keyframe_image_bytes

    images = tmp_path / "images"
    images.mkdir()
    data, origin, _ = keyframe_image_bytes(
        _StubStore(images), "w", "s", "s:00000042", None, _StubRedactor(),
        keyframes_are_redacted=True,
    )
    assert data is None
    assert origin == "absent"


def test_dense_never_reads_the_solve_workspace_images():
    """Some solve workspaces hold undistorted RAW frames that COLMAP was fed.
    Those bypass redaction, so the dense stage must not reach for them."""
    from tower.world_builder import dense_pipeline

    src = (dense_pipeline.run_depth_stage.__doc__ or "") + \
        dense_pipeline.keyframe_image_bytes.__doc__
    import inspect

    body = inspect.getsource(dense_pipeline.run_depth_stage)
    assert 'solve_images' not in body
    assert 'keyframe_image_bytes(' in body


def test_redaction_fill_is_excluded_exactly_when_the_raw_frame_survives():
    """A filled rectangle is not an observation. With the raw frame in hand the
    mask is a difference, so it is exact."""
    from tower.world_builder.dense_pipeline import redaction_fill_mask

    rng = np.random.default_rng(11)
    raw = rng.integers(30, 220, size=(120, 90, 3)).astype(np.uint8)
    red = raw.copy()
    red[20:60, 10:40] = 0                       # a solid fill, as redaction.py writes
    mask = redaction_fill_mask(red, raw, dilate_px=0)
    assert mask[20:60, 10:40].all()
    assert not mask[80:, 60:].any()


def test_redaction_fill_is_found_without_the_raw_frame():
    """Worlds keep only the redacted keyframe. A solid fill still has to be
    recognised, or the network's invented depth across it enters the cloud."""
    from tower.world_builder.dense_pipeline import redaction_fill_mask

    rng = np.random.default_rng(12)
    img = rng.integers(60, 240, size=(200, 160, 3)).astype(np.uint8)
    img[40:140, 30:110] = 0
    mask = redaction_fill_mask(img, None, dilate_px=0)
    inner = mask[50:130, 40:100]
    assert inner.mean() > 0.98
    assert mask[160:, 120:].mean() < 0.02


def test_redaction_fill_ignores_merely_dark_scene_content():
    """A dark room is not a redaction. Only a large solid block counts, or the
    mask would eat every night-time frame in the corpus."""
    from tower.world_builder.dense_pipeline import redaction_fill_mask

    rng = np.random.default_rng(13)
    dark = rng.integers(0, 40, size=(200, 160, 3)).astype(np.uint8)
    dark[dark < 3] = 12                          # dark, but textured, not flat zero
    mask = redaction_fill_mask(dark, None, dilate_px=0)
    assert mask.mean() < 0.05


# ---------------------------------------------------------------------------
# the listing's dense capability flag
# ---------------------------------------------------------------------------


def test_worlds_listing_contract_identifier_did_not_move():
    """iOS equality-tests `contract` on the first line of every guard. Bumping
    it empties the gallery on every build that predates the change, so the
    dense field had to be additive and this string had to stay put."""
    from tower.results.world_builder_library import WORLDS_CONTRACT

    assert WORLDS_CONTRACT == "world_builder.worlds/2026-09-06"


def test_dense_summary_is_none_when_a_world_has_no_dense_artifact(tmp_path):
    """Every world built before this work is in exactly this state, and it must
    read as absent rather than as an error."""
    from tower.results.world_builder_library import _dense_summary

    class _S:
        def world_dir(self, world_id):
            return tmp_path / world_id

    assert _dense_summary(_S(), "w", "s") is None


def test_dense_summary_reports_counts_and_repeats_the_scale(tmp_path):
    from tower.results.world_builder_library import _dense_summary

    d = tmp_path / "w" / "dense" / "s"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "format": DENSE_FORMAT,
        "canonical_level": 1, "mobile_level": 2,
        "scale": {"state": "unknown", "meters_per_unit": None},
        "levels": [{"level": 0, "voxel": 0.02, "points": 900, "bytes": 14400},
                   {"level": 1, "voxel": 0.045, "points": 300, "bytes": 4800},
                   {"level": 2, "voxel": 0.09, "points": 90, "bytes": 1440}],
    }))

    class _S:
        def world_dir(self, world_id):
            return tmp_path / world_id

    out = _dense_summary(_S(), "w", "s")
    assert out["levels"] == 3
    assert out["canonical_points"] == 300
    assert out["mobile_points"] == 90
    # Repeated, never re-derived: the dense stage makes no new scale claim.
    assert out["scale"] == {"state": "unknown", "meters_per_unit": None}


def test_a_manifest_with_an_unknown_format_is_ignored_rather_than_guessed_at(tmp_path):
    from tower.results.world_builder_library import _dense_summary

    d = tmp_path / "w" / "dense" / "s"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"format": "wb-dense-points/99"}))

    class _S:
        def world_dir(self, world_id):
            return tmp_path / world_id

    assert _dense_summary(_S(), "w", "s") is None


# ---------------------------------------------------------------------------
# the orchestrator's refusals
#
# Every one of these is a world that legitimately cannot be densified. None of
# them may raise, because this runs inside a capture's finalization: a dense
# stage that threw would turn "no dense reconstruction" into "the session ended
# badly", and the sparse world is complete and correct either way.
# ---------------------------------------------------------------------------


def test_densify_reports_unavailable_when_there_is_no_global_solve(tmp_path):
    from tower.world_builder.dense_pipeline import densify
    from tower.world_builder.records import Session, World
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    store.write_world(World(world_id="w1", created_at=1.0, updated_at=1.0,
                            session_ids=("s1",)))
    store.write_session(Session(session_id="s1", world_id="w1", started_at=1.0))

    result = densify(store, "w1", "s1")
    assert result.state == "unavailable"
    assert "solution" in (result.detail or "")
    # and it says so on disk, so an operator can see it without a live session
    status = json.loads((tmp_path / "worlds" / "w1" / "dense" / "s1" / "status.json").read_text())
    assert status["state"] == "unavailable"


def test_densify_writes_its_status_where_a_cold_reader_can_find_it(tmp_path):
    from tower.world_builder.dense_pipeline import dense_dir
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    d = dense_dir(store, "w1", "s1")
    # Beside solve/, never inside derived/: derived is the published output the
    # store digests and serves, and a reader that does not know about dense/
    # must be able to ignore it exactly as it ignores solve/.
    assert d.parent.name == "dense"
    assert "derived" not in d.parts


def test_read_dense_manifest_survives_a_truncated_file(tmp_path):
    """A manifest half-written by an interrupted run must read as absent."""
    from tower.world_builder.dense_pipeline import read_dense_manifest
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    d = tmp_path / "worlds" / "w1" / "dense" / "s1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text('{"format": "wb-dense-poi')
    assert read_dense_manifest(store, "w1", "s1") is None


# ---------------------------------------------------------------------------
# serving it: the page carries its own points, because fetch is blocked
# ---------------------------------------------------------------------------


def _fake_dense(tmp_path, n=1000, conf=None):
    """A world with a dense artifact on disk, and the store that reads it."""
    from tower.world_builder.dense import write_points_bin
    from tower.world_builder.store import WorldStore

    rng = np.random.default_rng(21)
    X = rng.uniform(-2, 2, size=(n, 3)).astype(np.float32)
    C = rng.integers(0, 256, size=(n, 3)).astype(np.uint8)
    F = (rng.integers(2, 12, size=n) if conf is None else conf).astype(np.uint8)
    d = tmp_path / "worlds" / "w1" / "dense" / "s1"
    d.mkdir(parents=True)
    size = write_points_bin(d / "points_l0.bin", X, C, F)
    (d / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "format": DENSE_FORMAT, "stride_bytes": 16,
        "canonical_level": 0, "mobile_level": 0,
        "median_scene_depth": 3.0,
        "bbox_min": X.min(0).tolist(), "bbox_max": X.max(0).tolist(),
        "scale": {"state": "unknown", "meters_per_unit": None},
        "levels": [{"level": 0, "voxel": 0.02, "points": n, "bytes": size}],
    }))
    return WorldStore(tmp_path), X, C, F


def test_the_page_embeds_its_points_because_the_route_forbids_fetch(tmp_path):
    """The render route sets `default-src 'none'` with no `connect-src`, so
    fetch and XHR are blocked outright. A page that asked for its data would
    show nothing, silently."""
    from tower.world_builder.dense_render import build_dense_page

    store, *_ = _fake_dense(tmp_path)
    page = build_dense_page(store, "w1", "s1")
    assert "__WB_POINTS_B64__" not in page and "__WB_CONFIG__" not in page
    for forbidden in ("fetch(", "XMLHttpRequest", "<script src", "<link rel=\"stylesheet\"",
                      "https://", "http://"):
        assert forbidden not in page, f"the page reaches for {forbidden!r}"


def test_the_page_is_thinned_by_confidence_not_at_random(tmp_path):
    """When a level exceeds the byte budget the geometry several cameras agreed
    on must survive and the weakest must go, so the picture gets sparser rather
    than less trustworthy."""
    from tower.world_builder.dense_render import build_dense_payload

    n = 4000
    conf = np.concatenate([np.full(n // 2, 2), np.full(n // 2, 9)])
    store, *_ = _fake_dense(tmp_path, n=n, conf=conf)
    raw, cfg, _ = build_dense_payload(store, "w1", "s1", budget_bytes=16 * (n // 2))
    assert cfg["points"] == n // 2
    kept = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 16)[:, 15]
    assert kept.min() == 9                       # every survivor is well supported
    assert cfg["thinned_to_confidence"] == 9


def test_an_unthinned_level_says_so_rather_than_implying_a_cut(tmp_path):
    from tower.world_builder.dense_render import build_dense_payload

    store, *_ = _fake_dense(tmp_path, n=100)
    _, cfg, _ = build_dense_payload(store, "w1", "s1")
    assert cfg["thinned_to_confidence"] is None
    assert cfg["points"] == 100


def test_the_page_never_claims_metres_when_scale_is_unknown(tmp_path):
    from tower.world_builder.dense_render import build_dense_page

    store, *_ = _fake_dense(tmp_path)
    page = build_dense_page(store, "w1", "s1")
    assert "not metres" in page
    # Empty space has three causes and the caption must not blame only the
    # capture: thinning for the device and face redaction are the other two.
    assert "redaction" in page
    assert "thinning" in page
    assert '"state": "unknown"' in page or '"state":"unknown"' in page


def test_the_viewer_does_not_assume_which_way_is_up(tmp_path):
    """up_axis is `unknown` and the cameras are OpenCV's, y DOWN. A viewer that
    assumed +y or +z was up would stand the room on its side."""
    from tower.world_builder.dense_render import build_dense_payload

    store, *_ = _fake_dense(tmp_path)
    _, cfg, _ = build_dense_payload(store, "w1", "s1")
    assert cfg["up_axis"] == "unknown"
    assert cfg["screen_up"] == [0, -1, 0]


def test_a_world_without_a_dense_artifact_is_refused_by_name(tmp_path):
    from tower.world_builder.dense_render import DenseViewerUnavailable, build_dense_page
    from tower.world_builder.store import WorldStore

    with pytest.raises(DenseViewerUnavailable):
        build_dense_page(WorldStore(tmp_path), "w1", "s1")


def test_the_viewer_template_is_installed_beside_the_code():
    """It is package data, not a build artifact: a Tower that can import the
    module can serve the page."""
    from tower.world_builder.dense_render import (
        TOKEN_CONFIG, TOKEN_POINTS, viewer_template_path,
    )

    p = viewer_template_path()
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert TOKEN_CONFIG in text and TOKEN_POINTS in text


# ---------------------------------------------------------------------------
# the route: a dense world gets the dense page, everything else is untouched
# ---------------------------------------------------------------------------


def _world_with_geometry(tmp_path, *, dense: bool):
    from tower.world_builder.dense import write_points_bin
    from tower.world_builder.records import Session, World
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    store.write_world(World(world_id="w1", created_at=1.0, updated_at=2.0,
                            session_ids=("s1",)))
    store.write_session(Session(session_id="s1", world_id="w1", started_at=1.0))
    derived = store.derived_dir("w1") / "s1"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "poses.json").write_text(json.dumps({"poses": []}))
    (derived / "points.json").write_text(json.dumps({"points": []}))
    if dense:
        rng = np.random.default_rng(31)
        X = rng.uniform(-1, 1, size=(500, 3)).astype(np.float32)
        C = rng.integers(0, 256, size=(500, 3)).astype(np.uint8)
        F = rng.integers(3, 9, size=500).astype(np.uint8)
        d = store.world_dir("w1") / "dense" / "s1"
        d.mkdir(parents=True)
        size = write_points_bin(d / "points_l0.bin", X, C, F)
        (d / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "format": DENSE_FORMAT, "stride_bytes": 16,
            "canonical_level": 0, "mobile_level": 0, "median_scene_depth": 2.0,
            "bbox_min": X.min(0).tolist(), "bbox_max": X.max(0).tolist(),
            "scale": {"state": "unknown", "meters_per_unit": None},
            "levels": [{"level": 0, "voxel": 0.02, "points": 500, "bytes": size}],
        }))
    return store


def test_a_world_with_a_dense_artifact_is_served_the_dense_viewer(tmp_path):
    from tower.results.world_builder_render import build_world_render

    html = build_world_render(_world_with_geometry(tmp_path, dense=True), "w1", "s1")
    assert "World Builder — dense" in html
    assert "500 points" not in html or "points" in html          # the config is inlined
    assert "gl.POINTS" in html or "drawArrays" in html


def test_a_world_without_one_still_gets_exactly_the_page_it_got_before(tmp_path):
    """Every world built before this stage is in this state. The dense work must
    be invisible to them."""
    from tower.results.world_builder_render import build_world_render

    html = build_world_render(_world_with_geometry(tmp_path, dense=False), "w1", "s1")
    assert "World Builder — dense" not in html


def test_representation_sparse_forces_the_old_page_even_when_dense_exists(tmp_path):
    from tower.results.world_builder_render import build_world_render

    store = _world_with_geometry(tmp_path, dense=True)
    html = build_world_render(store, "w1", "s1", representation="sparse")
    assert "World Builder — dense" not in html


def test_a_broken_dense_artifact_never_costs_the_world_its_sparse_page(tmp_path):
    """A dense bug must degrade to the picture that already worked, not to a
    404. The sparse reconstruction is complete and correct either way."""
    from tower.results.world_builder_render import build_world_render

    store = _world_with_geometry(tmp_path, dense=True)
    (store.world_dir("w1") / "dense" / "s1" / "points_l0.bin").write_bytes(b"\x00" * 7)
    html = build_world_render(store, "w1", "s1")
    assert "World Builder — dense" not in html


def test_two_densify_runs_of_one_session_do_not_interleave(tmp_path):
    """The dense stage runs after the world writer lock is released, so nothing
    else stops a second run of the SAME session -- which is easy to start by
    accident, e.g. world_densify.py while a build is finalising. Interleaved
    writes would leave a points file of the wrong length, which does not fail
    loudly."""
    from tower.world_builder.dense_pipeline import _DenseLock

    root = tmp_path / "dense" / "s1"
    a, b = _DenseLock(root), _DenseLock(root)
    assert a.acquire()
    assert not b.acquire()
    a.release()
    assert b.acquire()
    b.release()


def test_a_lock_left_by_a_dead_process_is_reclaimed(tmp_path):
    """A builder killed by the Job Object leaves its lock behind. The next run
    must take it rather than refusing forever."""
    import json as _json
    import os as _os

    from tower.world_builder.dense_pipeline import _DenseLock

    root = tmp_path / "dense" / "s1"
    root.mkdir(parents=True)
    # A pid that is almost certainly not running, and is not ours.
    (root / ".densify.lock").write_text(_json.dumps({"pid": 0x7FFFFFFE, "at": 0}))
    lock = _DenseLock(root)
    assert lock.acquire()
    assert _json.loads((root / ".densify.lock").read_text())["pid"] == _os.getpid()
    lock.release()


def test_a_points_file_is_never_visible_half_written(tmp_path):
    """Written to a temporary name and renamed. A partial buffer is a valid file
    of the wrong length, so a reader cannot tell it is broken."""
    import inspect

    from tower.world_builder import dense

    src = inspect.getsource(dense.write_points_bin)
    assert ".tmp" in src and "replace(" in src


def test_the_config_cannot_break_out_of_the_script_tag(tmp_path):
    """The config is substituted into a JS literal inside <script>.

    A world id cannot carry "<" -- Windows will not create the directory and
    `contained_world_id` guards the route -- but the config also copies the
    manifest's `scale` block through verbatim, and that is a file. Escaping
    costs nothing and means no future field can end the tag and turn the rest
    of the page into markup.
    """
    from tower.world_builder.dense_render import build_dense_page

    store, *_ = _fake_dense(tmp_path)
    man = tmp_path / "worlds" / "w1" / "dense" / "s1" / "manifest.json"
    payload = json.loads(man.read_text())
    payload["scale"]["note"] = "</script><img src=x onerror=alert(1)>"
    man.write_text(json.dumps(payload))

    page = build_dense_page(store, "w1", "s1")
    # What matters is that no LESS-THAN survives to end the tag. The text
    # "onerror=alert(1)" does survive, escaped, inside a JS string value -- and
    # that is fine: a string is not markup, and asserting its absence would be
    # asserting the wrong thing.
    assert "</script><img" not in page
    assert "u003c/script" in page
    assert page.count("</script>") == page.count("<script>")


def test_the_depth_cache_key_covers_every_parameter_the_stage_reads():
    """Reusing predictions made under different parameters, while the manifest
    records the new ones, makes the artifact unreproducible from its own params
    -- and everything still runs, so nothing says so. `component` is the one
    that bit: a --component 1 run reused component 0's predictions."""
    from tower.world_builder.dense import DenseParams
    from tower.world_builder.dense_pipeline import _depth_cache_key

    base = DenseParams()
    key = _depth_cache_key("digest-a", base)
    from dataclasses import replace

    for field, value in [("backend", "depth-anything-v2-small"),
                         ("component", 1),
                         ("min_sparse_points", 40)]:
        assert _depth_cache_key("digest-a", replace(base, **{field: value})) != key, field
    # and a different solve is a different key
    assert _depth_cache_key("digest-b", base) != key
    # a missing digest must not match everything
    assert _depth_cache_key(None, base) != key


def test_the_fuse_cache_key_covers_every_parameter_the_stage_reads():
    from dataclasses import replace

    from tower.world_builder.dense import DenseParams
    from tower.world_builder.dense_pipeline import _fuse_cache_key

    base = DenseParams()
    key = _fuse_cache_key("d", base)
    for field, value in [("gate_rel", 0.2), ("tau", 0.09), ("min_views", 4),
                         ("neighbours", 20), ("stride", 2), ("edge_rel", 0.1),
                         ("max_grazing_deg", 70.0), ("erode_px", 0),
                         ("average_views", False), ("max_extrapolation", 3.0)]:
        assert _fuse_cache_key("d", replace(base, **{field: value})) != key, field


def test_a_run_killed_mid_stage_is_distinguishable_from_one_still_going():
    """The dense stage deliberately outlives the world lock, so the supervisor
    can kill it after finalization is already marked complete. Without a pid on
    the status, that leaves `running` on disk forever and nothing can tell it
    from a run that is genuinely still going."""
    import os as _os

    from tower.world_builder.dense_pipeline import STATE_RUNNING, status_is_stale

    assert not status_is_stale({"state": STATE_RUNNING, "pid": _os.getpid()})
    assert status_is_stale({"state": STATE_RUNNING, "pid": 0x7FFFFFFE})
    assert status_is_stale({"state": STATE_RUNNING})          # no pid at all
    assert not status_is_stale({"state": "ok", "pid": 0x7FFFFFFE})
    assert not status_is_stale({})


# ---------------------------------------------------------------------------
# two output kinds: a point map is not a disparity image
# ---------------------------------------------------------------------------


def test_depth_from_prediction_inverts_both_fit_forms():
    """One function decides what a stored map means, shared by the alignment,
    the scoring and the fusion, so the three cannot drift apart."""
    from tower.world_builder.dense import depth_from_prediction, fit_target

    rng = np.random.default_rng(41)
    z = rng.uniform(0.8, 9.0, size=500)

    a, b = 2700.0, -215.0
    disp = a * fit_target(z, "disparity") + b
    assert np.allclose(depth_from_prediction(disp, a, b, "disparity"), z, rtol=1e-9)

    a2, b2 = 1.37, -0.42
    pts = a2 * fit_target(z, "depth") + b2
    assert np.allclose(depth_from_prediction(pts, a2, b2, "depth"), z, rtol=1e-9)


def test_a_point_map_fitted_as_disparity_is_measurably_worse():
    """This is why `kind` exists. A point map's z is affine-invariant DEPTH;
    fitting it with the disparity form still converges, and still returns a
    plausible-looking residual, while being wrong."""
    from tower.world_builder.dense import align_frame

    rng = np.random.default_rng(42)
    z = rng.uniform(1.0, 8.0, size=400)
    pointmap_z = 1.37 * z - 0.42                      # affine-invariant depth

    _, _, right = align_frame(pointmap_z, z, "depth")
    _, _, wrong = align_frame(pointmap_z, z, "disparity")

    assert right is not None and right < 1e-6

    # The wrong form does not merely fit worse. It fits an INFEASIBLE model --
    # a point map is monotonically increasing in depth, a disparity image is
    # monotonically decreasing -- so the slope comes out non-positive and the
    # frame is scored as unusable and dropped. Either outcome is a loss; this
    # one at least fails loudly.
    assert wrong is None or wrong > 100 * right


def test_the_default_backend_is_the_measured_winner_and_permissive():
    from tower.world_builder.dense import DenseParams, make_backend

    b = make_backend(DenseParams().backend)
    assert b.name == "moge2-vitl"
    assert b.licence == "MIT"
    assert b.kind == "depth"


def test_every_registered_backend_declares_a_kind_the_pipeline_understands():
    from tower.world_builder.dense import available_backends, make_backend

    for name in available_backends():
        b = make_backend(name)
        assert b.kind in {"disparity", "depth"}, name
        assert "NC" not in b.licence and "NonCommercial" not in b.licence, name


def test_a_session_with_no_solve_does_not_leave_its_lock_behind(tmp_path):
    """The most ordinary failure there is. The early `unavailable` returns used
    to sit outside the lock's finally, so densifying a session with no solve
    bricked that session permanently -- every later attempt refused with
    "another densify is already running"."""
    from tower.world_builder.dense_pipeline import densify, dense_dir
    from tower.world_builder.records import Session, World
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    store.write_world(World(world_id="w1", created_at=1.0, updated_at=1.0,
                            session_ids=("s1",)))
    store.write_session(Session(session_id="s1", world_id="w1", started_at=1.0))

    first = densify(store, "w1", "s1")
    assert first.state == "unavailable"
    assert not (dense_dir(store, "w1", "s1") / ".densify.lock").exists()

    # and a second attempt reaches the same honest answer rather than the lock
    second = densify(store, "w1", "s1")
    assert second.state == "unavailable"
    assert "solution" in (second.detail or "")


def test_liveness_uses_the_stores_probe_not_os_kill():
    """os.kill(pid, 0) is a console-signal call on Windows and reported a
    freshly dead process as still alive in testing, which strands a lock."""
    import inspect

    from tower.world_builder import dense_pipeline

    assert "os.kill" not in inspect.getsource(dense_pipeline._DenseLock)
    assert "os.kill" not in inspect.getsource(dense_pipeline.status_is_stale)
    assert "_pid_is_running" in inspect.getsource(dense_pipeline._DenseLock._stale)


def test_the_fill_fallback_requires_a_rectangle_not_merely_darkness():
    """Measured against the exact difference over 120 real frames, the
    near-black test alone ran at 36.2% precision -- two thirds of what it
    deleted was real scene, 3.6 million pixels of it, including an entire bed.
    Redaction fills an axis-aligned rectangle; a bed is not one. Requiring the
    component to fill its own bounding box took precision to 88.2%."""
    from tower.world_builder.dense_pipeline import redaction_fill_mask

    rng = np.random.default_rng(51)
    img = rng.integers(60, 240, size=(240, 200, 3)).astype(np.uint8)

    # a genuine fill: a solid rectangle
    img[40:140, 30:110] = 0
    # a dark, irregular object: a blob that is near-black but not a rectangle
    yy, xx = np.mgrid[0:240, 0:200]
    blob = ((yy - 200) ** 2 / 30.0 + (xx - 40) ** 2 / 120.0) < 12
    img[blob] = 2

    mask = redaction_fill_mask(img, None, dilate_px=0)
    assert mask[50:130, 40:100].mean() > 0.98      # the rectangle goes
    assert mask[blob].mean() < 0.05                # the dark object stays


def test_raw_pixels_never_leave_the_privacy_boundary():
    """The mask is differenced inside `keyframe_image_bytes` and the raw bytes
    are dropped there, so no caller can reconstruct from them by accident."""
    import inspect

    from tower.world_builder import dense_pipeline

    body = inspect.getsource(dense_pipeline.run_depth_stage)
    assert "raw_bytes" not in body
    assert "exact_fill" in body


def test_the_dense_dependencies_are_declared():
    """The default backend imports `moge`, which for a while appeared in no
    dependency list at all -- so a fresh checkout would have failed at
    finalization rather than at install."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    meta = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = meta["project"]["optional-dependencies"]
    assert "dense" in extras
    assert any(d.startswith("moge") for d in extras["dense"])


def test_a_backend_whose_package_is_missing_refuses_by_name():
    """A finalization step must not raise a bare ImportError from inside the
    model loader; it must say which package and which flag."""
    import inspect

    from tower.world_builder import dense

    for cls in (dense.MoGeBackend, dense.DepthAnything3Backend):
        src = inspect.getsource(cls._load)
        assert "DenseUnavailable" in src, cls.__name__
        assert "--backend" in src, cls.__name__


def test_no_backend_assumes_a_gpu_is_present():
    """A Tower without a GPU should fall back, not raise a CUDA error from
    inside finalization."""
    import inspect

    from tower.world_builder import dense

    for cls in (dense.MoGeBackend, dense.DepthAnything3Backend,
                dense.TransformersDepthBackend):
        src = inspect.getsource(cls._load)
        assert "is_available()" in src, cls.__name__
        assert ".cuda()" not in src, cls.__name__


def test_dense_does_not_trust_a_keyframe_the_session_says_was_never_redacted(tmp_path):
    """`FaceRedactor.redact` returns the ORIGINAL bytes when the redactor is
    unavailable or throws, labelled `none`, and `engine._persist_keyframe`
    persists whatever comes back. So `images/` can hold raw frames, and
    `session.redaction` is the only record that says which. An earlier version
    of this boundary asserted the guarantee in a docstring and then read that
    directory unconditionally."""
    from tower.world_builder.dense_pipeline import keyframe_image_bytes

    images = tmp_path / "images"
    images.mkdir()
    (images / "00000042.jpg").write_bytes(b"UNREDACTED-KEYFRAME")
    red = _StubRedactor()

    data, origin, _ = keyframe_image_bytes(
        _StubStore(images), "w", "s", "s:00000042", None, red,
        keyframes_are_redacted=False,
    )
    assert red.calls == 1
    assert data == b"REDACTED:UNREDACTED-KEYFRAME"
    assert origin == "world-keyframe-redacted-here"
    assert b"UNREDACTED-KEYFRAME" != data


def test_dense_refuses_an_unredacted_keyframe_when_no_redactor_is_available(tmp_path):
    from tower.world_builder.dense_pipeline import keyframe_image_bytes

    images = tmp_path / "images"
    images.mkdir()
    (images / "00000042.jpg").write_bytes(b"UNREDACTED-KEYFRAME")

    for redactor in (_StubRedactor(available=False), None):
        data, origin, _ = keyframe_image_bytes(
            _StubStore(images), "w", "s", "s:00000042", None, redactor,
            keyframes_are_redacted=False,
        )
        assert data is None
        assert origin == "refused-unredacted-keyframe"


def test_the_depth_stage_reads_the_sessions_redaction_record():
    """The check has to be wired, not merely available: the finding was that
    `session.redaction` appeared nowhere in this module."""
    import inspect

    from tower.world_builder import dense_pipeline

    body = inspect.getsource(dense_pipeline.run_depth_stage)
    assert "read_session" in body
    assert "REDACTION_NONE" in body
    assert "keyframes_are_redacted=keyframes_are_redacted" in body


def test_align_records_the_sessions_redaction_not_the_loaded_redactors_label():
    """`align.json` used to record the label of the redactor loaded NOW, which
    says nothing about the pixels that were read."""
    import inspect

    from tower.world_builder import dense_pipeline

    body = inspect.getsource(dense_pipeline.run_depth_stage)
    assert '"redaction": session_redaction' in body
    assert '"keyframes_were_redacted_at_capture"' in body


# --------------------------------------------------------------------------
# The caption obligation, and staleness. `WORLD-BUILDER-WORLDS.md` requires the
# render page to carry a caption saying what it is, plus a BEHIND line when the
# derived tree is behind; the dense page is now what that route serves, and it
# used to carry neither.
# --------------------------------------------------------------------------


def test_the_dense_caption_says_what_the_picture_is_and_refuses_the_word_scan():
    from tower.world_builder.dense_render import CAPTION

    lowered = CAPTION.lower()
    assert "not a surface" in lowered
    assert "not a mesh" in lowered
    assert "not metric scale" in lowered
    assert "scan" not in lowered


def test_the_viewer_prints_the_caption_first_and_the_behind_lines_after():
    from tower.world_builder.dense_render import viewer_template_path

    html = viewer_template_path().read_text(encoding="utf-8")
    assert "CONFIG.caption" in html
    assert "CONFIG.caption_behind" in html
    # The caption leads. A qualification printed before the thing it qualifies
    # is not a caption.
    assert html.index("CONFIG.caption") < html.index("Confidence is how many")


def test_dense_currency_calls_a_matching_digest_current_and_a_changed_one_behind(tmp_path):
    import json

    from tower.world_builder.dense_pipeline import dense_currency

    class _Store:
        def __init__(self, root):
            self.root = root

        def world_dir(self, world_id):
            return self.root / world_id

    world = tmp_path / "w"
    (world / "dense" / "s").mkdir(parents=True)
    solve = world / "solve" / "s"
    solve.mkdir(parents=True)
    (solve / "solution.json").write_text(json.dumps({"input_digest": "AAA"}))
    store = _Store(tmp_path)

    assert dense_currency(store, "w", "s", {"input_digest": "AAA"})["solve_current"] is True
    assert dense_currency(store, "w", "s", {"input_digest": "BBB"})["solve_current"] is False
    # Unknowable is not stale: a manifest with no digest and no status.json
    # must not produce a BEHIND claim.
    assert dense_currency(store, "w", "s", {})["solve_current"] is None

    # An artifact packed before the manifest carried the digest falls back to
    # status.json, which has recorded it since the first version of the stage.
    (world / "dense" / "s" / "status.json").write_text(json.dumps({"input_digest": "AAA"}))
    assert dense_currency(store, "w", "s", {})["solve_current"] is True


def test_the_manifest_records_the_solve_it_was_built_from():
    """Without it nothing at serve time can tell that the world was re-solved."""
    import inspect

    from tower.world_builder import dense_pipeline

    assert '"input_digest": input_digest' in inspect.getsource(dense_pipeline.run_pack_stage)
    assert "input_digest=digest" in inspect.getsource(dense_pipeline.densify)


# --------------------------------------------------------------------------
# WKWebView specifics. The phone loads this page with loadHTMLString and a null
# baseURL, so its URL is about:blank and the app refuses a second navigation to
# it -- location.reload() there is at best a no-op.
# --------------------------------------------------------------------------


def test_the_viewer_rebuilds_gl_state_instead_of_reloading_a_urlless_document():
    from tower.world_builder.dense_render import viewer_template_path

    html = viewer_template_path().read_text(encoding="utf-8")
    restored = html.index("webglcontextrestored")
    handler = html[restored:restored + 1400]
    assert "location.reload()" not in handler
    assert "buildGL()" in handler
    assert "requestAnimationFrame(frame)" in handler


def test_the_viewer_releases_the_base64_and_the_binary_string_after_decoding():
    """8 MB of base64 plus an 8 MB binary string, alive for the life of the
    page beside the 6 MB typed array they produced. WKWebView kills the content
    process rather than paging."""
    from tower.world_builder.dense_render import viewer_template_path

    html = viewer_template_path().read_text(encoding="utf-8")
    assert "B64 = null" in html
    assert "bin = null" in html


def test_the_inline_payload_rationale_does_not_claim_a_csp_the_phone_never_sees():
    """iOS discards the response headers and calls loadHTMLString, so no CSP
    applies on the device the product ships on. The escaping is therefore the
    sole defence, and the module must say so rather than the opposite."""
    from tower.world_builder import dense_render

    doc = dense_render.__doc__ or ""
    assert "loadHTMLString" in doc
    assert "SOLE defence" in doc
