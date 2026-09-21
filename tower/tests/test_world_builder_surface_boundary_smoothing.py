"""A rim is smoothed ALONG the rim, not across the surface.

`Glasses-scratch/wb-final-recon/fixit/rims/RIMS.md`: 48% of the canonical
capture's phone-level faces touch a boundary, and the one umbrella the
smoothing used for every vertex did two wrong things to every one of them.
A boundary vertex's neighbours all lie on one side of it, so the average
dragged the rim across the surface, away from where the evidence ended
(-0.29 voxels, median); and none of those neighbours is itself on the rim, so
the marching-cubes staircase ALONG the rim survived the pass untouched. That
staircase is the ragged black rim the visual review ranked first.

`SurfaceParams.smooth_boundary_curve` gives a boundary vertex with exactly two
boundary neighbours the same lambda/mu passes over its own polyline, holds a
junction still, and leaves every interior vertex on the surface umbrella.
"""

from __future__ import annotations

import numpy as np

from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP


def _grid(nx, ny, step=1.0):
    """A flat triangulated grid in z = 0, row-major vertex ids."""
    V = np.array([[i * step, j * step, 0.0] for j in range(ny) for i in range(nx)],
                 np.float32)
    F = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            F.append((a, a + 1, a + nx))
            F.append((a + 1, a + nx + 1, a + nx))
    return V, np.asarray(F, np.int64)


def _staircase_grid(nx=13, ny=9, tooth=1):
    """The same grid with a one-quad staircase bitten out of its top rim, so
    the rim runs ...up, along, down, along... exactly as marching cubes leaves
    it at the edge of the observed field."""
    V, F = _grid(nx, ny)
    keep = []
    for f in F:
        j = min(v // nx for v in f)
        i = min(v % nx for v in f)
        if j == ny - 2 and (i % 2 == 0) and tooth:
            continue
        keep.append(f)
    return V, np.asarray(keep, np.int64)


def _roughness(V, F):
    mid, left, right, _junction = S.boundary_curve(F, len(V))
    return np.linalg.norm(V[mid] - 0.5 * (V[left] + V[right]), axis=1)


class TestFindingTheBoundary:
    def test_one_triangle_is_all_boundary(self):
        V = np.zeros((3, 3), np.float32)
        E = S.boundary_edges(np.array([[0, 1, 2]], np.int64))
        assert len(E) == 3
        assert {tuple(e) for e in E} == {(0, 1), (1, 2), (0, 2)}

    def test_a_shared_edge_is_not_boundary(self):
        F = np.array([[0, 1, 2], [1, 3, 2]], np.int64)
        E = S.boundary_edges(F)
        assert len(E) == 4
        assert (1, 2) not in {tuple(e) for e in E}

    def test_an_interior_vertex_owns_no_boundary_edge(self):
        V, F = _grid(5, 5)
        mid, _l, _r, junction = S.boundary_curve(F, len(V))
        centre = 2 * 5 + 2
        assert centre not in set(mid.tolist())
        assert not junction[centre]
        # the ring of a 5x5 grid is 16 vertices, every one with two neighbours
        assert len(mid) == 16 and junction.sum() == 0

    def test_a_junction_is_named_and_not_smoothed_along(self):
        # two quads meeting at a single vertex: that vertex owns four
        # boundary edges, so it belongs to no one rim.
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                      [2, 1, 0], [1, 2, 0], [2, 2, 0]], np.float32)
        F = np.array([[0, 1, 2], [1, 3, 2], [3, 4, 5], [4, 6, 5]], np.int64)
        mid, _l, _r, junction = S.boundary_curve(F, len(V))
        assert junction[3]
        assert 3 not in set(mid.tolist())


class TestTheRimIsStraightened:
    def test_the_staircase_survives_the_surface_umbrella(self):
        # The umbrella takes the teeth down only as far as the interior pass
        # happens to pull them: 0.71 -> 0.29 of a grid step here, 0.56 -> 0.31
        # of a voxel on the canonical capture. It is not a rim operator and
        # it does not finish the job.
        V, F = _staircase_grid()
        before = np.median(_roughness(V, F))
        V1, _m = S.taubin_smooth(V, F, 6, 0.5, -0.53)
        after = np.median(_roughness(V1, F))
        assert after > 0.3 * before, (before, after)

    def test_smoothing_along_the_rim_halves_it(self):
        V, F = _staircase_grid()
        before = np.median(_roughness(V, F))
        V1, _m = S.taubin_smooth(V, F, 6, 0.5, -0.53)
        V2, _m = S.taubin_smooth(V, F, 6, 0.5, -0.53, boundary_curve_smoothing=True)
        assert np.median(_roughness(V2, F)) < 0.5 * np.median(_roughness(V1, F))
        assert np.median(_roughness(V2, F)) < 0.5 * before

    def test_the_rim_is_dragged_across_the_surface_less(self):
        # A flat square patch: every rim vertex's inward direction is known,
        # so "drift across the surface" is the drop in the patch's extent.
        V, F = _grid(11, 11)
        V1, _m = S.taubin_smooth(V, F, 6, 0.5, -0.53)
        V2, _m = S.taubin_smooth(V, F, 6, 0.5, -0.53, boundary_curve_smoothing=True)
        area = lambda P: float(np.ptp(P[:, 0]) * np.ptp(P[:, 1]))  # noqa: E731
        assert area(V1) < area(V), "the umbrella did not shrink the patch"
        assert area(V2) > area(V1), (area(V), area(V1), area(V2))


class TestWhatItMayNotTouch:
    def test_an_interior_vertex_away_from_the_rim_moves_identically(self):
        # Twelve passes reach twelve rings, so "away from the rim" means
        # twelve rings away. On the canonical capture 63% of interior
        # vertices are, and do not move at all.
        nx = 41
        V, F = _grid(nx, nx)
        V1, _m = S.taubin_smooth(V, F, 6, 0.5, -0.53)
        V2, _m = S.taubin_smooth(V, F, 6, 0.5, -0.53, boundary_curve_smoothing=True)
        deep = np.array([j * nx + i for j in range(15, 26) for i in range(15, 26)])
        assert np.allclose(V1[deep], V2[deep], atol=1e-9), np.abs(V1[deep] - V2[deep]).max()

    def test_a_junction_does_not_move(self):
        # The two quads are deliberately lopsided, so the umbrella WOULD move
        # the junction: this test would pass on a symmetric one either way.
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                      [2.6, 1, 0], [1, 2, 0], [2.6, 2, 0]], np.float32)
        F = np.array([[0, 1, 2], [1, 3, 2], [3, 4, 5], [4, 6, 5]], np.int64)
        V1, _m = S.taubin_smooth(V, F, 6, 0.5, -0.53)
        assert not np.allclose(V1[3], V[3], atol=1e-3), "the umbrella left it alone"
        V2, _m = S.taubin_smooth(V, F, 6, 0.5, -0.53, boundary_curve_smoothing=True)
        assert np.allclose(V2[3], V[3], atol=1e-7), V2[3]

    def test_no_face_is_collapsed(self):
        V, F = _staircase_grid()
        V2, _m = S.taubin_smooth(V, F, 6, 0.5, -0.53, boundary_curve_smoothing=True)
        a, b, c = V2[F[:, 0]], V2[F[:, 1]], V2[F[:, 2]]
        assert (0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-6).all()

    def test_off_is_the_old_behaviour_exactly(self):
        V, F = _staircase_grid()
        a, _ = S.taubin_smooth(V, F, 6, 0.5, -0.53)
        b, _ = S.taubin_smooth(V, F, 6, 0.5, -0.53, boundary_curve_smoothing=False)
        assert np.array_equal(a, b)

    def test_a_closed_mesh_has_no_rim_to_smooth(self):
        # a tetrahedron: no boundary edge, so the flag can change nothing
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
        assert len(S.boundary_edges(F)) == 0
        a, _ = S.taubin_smooth(V, F, 4, 0.5, -0.53)
        b, _ = S.taubin_smooth(V, F, 4, 0.5, -0.53, boundary_curve_smoothing=True)
        assert np.array_equal(a, b)


class TestTheParameter:
    def test_default_and_digest(self):
        p = S.SurfaceParams()
        assert p.smooth_boundary_curve is True
        assert p.digest_fields() != S.SurfaceParams(
            smooth_boundary_curve=False).digest_fields()

    def test_the_live_preset_inherits_it(self):
        assert S.SurfaceParams.live().smooth_boundary_curve is True


class TestThroughTheProduct:
    def _build(self, tmp_path, name, **kw):
        from tests.test_world_builder_surface_low_weight import _far_wall_views
        from tests.test_world_builder_surface_pipeline import (
            SESSION, WORLD, _params, _synthetic_world)

        store = _synthetic_world(tmp_path / name,
                                 views=_far_wall_views(spread=0.4, n=8), anchors=True)
        r = SP.surfacify(store, WORLD, SESSION,
                         params=_params(min_weight=2.0, min_support_frames=2,
                                        lod_face_targets=(0,), mobile_level=0, **kw))
        assert r.state == SP.STATE_OK, r.detail
        return SP.read_surface_manifest(store, WORLD, SESSION)

    def test_the_manifest_records_the_rim_and_which_smoothing_made_it(self, tmp_path):
        on = self._build(tmp_path, "on")["detail"]["rims"]
        off = self._build(tmp_path, "off",
                          smooth_boundary_curve=False)["detail"]["rims"]
        assert on["smoothing"] == "curve" and off["smoothing"] == "umbrella"
        assert on["boundary_edges"] == off["boundary_edges"], "no face may change"
        assert on["roughness_voxels"] < off["roughness_voxels"], (on, off)
        assert on["length_voxels"] < off["length_voxels"], (on, off)
