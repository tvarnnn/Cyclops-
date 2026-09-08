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

    data, origin, raw_bytes = keyframe_image_bytes(
        _StubStore(images), "w", "s", "s:00000042", str(raw), red
    )
    assert data == b"REDACTED-KEYFRAME"
    assert origin == "world-keyframe"
    assert red.calls == 0                      # the raw frame was never opened
    assert raw_bytes is None                   # and is not handed on either


def test_dense_re_redacts_when_the_worlds_keyframe_image_is_missing(tmp_path):
    """Worlds migrated between roots lost their images/ directory. Falling back
    to the raw capture is allowed only through the same transformation."""
    from tower.world_builder.dense_pipeline import keyframe_image_bytes

    images = tmp_path / "images"
    images.mkdir()
    raw = tmp_path / "raw.jpg"
    raw.write_bytes(b"RAW-WITH-A-FACE")
    red = _StubRedactor()

    data, origin, raw_bytes = keyframe_image_bytes(
        _StubStore(images), "w", "s", "s:00000042", str(raw), red
    )
    assert red.calls == 1
    assert raw_bytes == b"RAW-WITH-A-FACE"     # returned only so the fill can be differenced
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
        _StubStore(images), "w", "s", "s:00000042", str(raw), _StubRedactor(available=False)
    )
    assert data is None
    assert origin == "refused-no-redactor"

    data, origin, _ = keyframe_image_bytes(
        _StubStore(images), "w", "s", "s:00000042", str(raw), None
    )
    assert data is None


def test_dense_reports_an_absent_image_rather_than_inventing_one(tmp_path):
    from tower.world_builder.dense_pipeline import keyframe_image_bytes

    images = tmp_path / "images"
    images.mkdir()
    data, origin, _ = keyframe_image_bytes(
        _StubStore(images), "w", "s", "s:00000042", None, _StubRedactor()
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
