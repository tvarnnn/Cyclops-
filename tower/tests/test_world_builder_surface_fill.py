"""The opt-in enclosed-hole fill, tested by outcome.

The field is written by hand, so the answer is known by construction: a
plane seen from one side, with

  * a small dropout in the middle of it -- a gap every side of which some
    camera measured, which the fill is FOR; and
  * a frontier -- a half-space nobody looked at, which the fill must never
    advance into.

The questions are the ones a hostile reviewer would ask: is the hole closed,
is the edge of knowledge still where it was, did any measured voxel change,
does a filled voxel ever outweigh a measured one, and is the whole thing off
unless someone turns it on.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")

from tower.world_builder import surface as S  # noqa: E402

N = 40              # voxels per axis
PLANE = 20.5        # the surface sits at z = 20.5
TRUNC = 3.0
MINW = 2.0
HOLE = dict(x=(18, 22), y=(18, 22), z=(18, 23))   # half-open voxel ranges
FRONTIER_X = 30     # voxels with x >= this were never observed


def _field(*, hole: bool = True, frontier: bool = True) -> S.SurfaceVolume:
    vol = S.SurfaceVolume(1.0, TRUNC, device=torch.device("cpu"))
    nb = N // S.BLOCK
    r = torch.arange(nb, dtype=torch.int64)
    bi, bj, bk = torch.meshgrid(r, r, r, indexing="ij")
    vol.reserve(S.block_key(torch.stack([bi.reshape(-1), bj.reshape(-1),
                                         bk.reshape(-1)], 1)))
    X = vol.voxel_world(torch.arange(vol.n_blocks)) - 0.5   # integer voxel index
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    zc = z + 0.5
    vol.tsdf[:] = ((zc - PLANE) / TRUNC).clamp(-1, 1)
    observed = zc >= PLANE - TRUNC           # the band and the carved front
    if hole:
        observed &= ~((x >= HOLE["x"][0]) & (x < HOLE["x"][1])
                      & (y >= HOLE["y"][0]) & (y < HOLE["y"][1])
                      & (z >= HOLE["z"][0]) & (z < HOLE["z"][1]))
    if frontier:
        observed &= x < FRONTIER_X
    vol.w[:] = torch.where(observed, torch.full_like(zc, 5.0), torch.zeros_like(zc))
    vol.rgb[:] = 128.0
    # unobserved voxels carry nothing a camera said
    vol.tsdf[:] = torch.where(observed, vol.tsdf, torch.ones_like(zc))
    return vol


def _in_hole(V, F):
    """Faces whose centroid is well inside the dropout's footprint.

    The margin is half a voxel: marching cubes puts a sliver of the hole's
    side wall just inside its nominal edge, and that sliver is measured
    surface meeting unmeasured space, not a triangle across the gap."""
    c = V[F].mean(axis=1)
    return ((c[:, 0] > HOLE["x"][0] + 0.5) & (c[:, 0] < HOLE["x"][1] - 1.5)
            & (c[:, 1] > HOLE["y"][0] + 0.5) & (c[:, 1] < HOLE["y"][1] - 1.5))


# ---------------------------------------------------------------------------
# off unless someone turns it on
# ---------------------------------------------------------------------------


class TestOptIn:
    def test_the_fill_is_off_by_default(self):
        p = S.SurfaceParams()
        assert p.fill_gap_frac == 0.0
        assert p.fill_radius_voxels() == 0
        assert S.SurfaceParams.live().fill_radius_voxels() == 0

    def test_existing_artifacts_keep_their_digest(self):
        """Adding the option must not rebuild every surface already on disk:
        with the fill off, the digest carries no fill field at all. (Its
        length is not pinned -- other parameters that DO change the output,
        like the block budget, rightly join it.)"""
        p = S.SurfaceParams()
        off = p.digest_fields()
        assert "fill" not in off
        assert p.fill_gap_frac not in off[len(off) - 2:]
        on = S.SurfaceParams(fill_gap_frac=0.024).digest_fields()
        assert on[:len(off)] == off, "turning the fill on must only append"

    def test_turning_it_on_changes_the_digest(self):
        on = S.SurfaceParams(fill_gap_frac=0.024)
        assert on.digest_fields() != S.SurfaceParams().digest_fields()

    def test_the_radius_is_the_whole_gap_not_half_of_it(self):
        """A corner voxel must see across the full width of the gap."""
        assert S.SurfaceParams(fill_gap_frac=0.024).fill_radius_voxels() == 4
        assert S.SurfaceParams(fill_gap_frac=0.018).fill_radius_voxels() == 3

    def test_a_filled_artifact_cannot_claim_the_measured_format(self):
        assert S.SURFACE_FORMAT_ENCLOSED_FILL != S.SURFACE_FORMAT
        assert S.SURFACE_FORMAT_ENCLOSED_FILL.startswith(S.SURFACE_FORMAT)


# ---------------------------------------------------------------------------
# what the fill does to the field
# ---------------------------------------------------------------------------


class TestField:
    def test_the_dropout_is_really_open_without_the_fill(self):
        V, F, C = _field().extract_mesh(MINW, tile_blocks=2)
        assert len(F) > 0
        assert not _in_hole(V, F).any()

    def test_a_bracketed_hole_is_closed_on_the_plane(self):
        vol = _field()
        rec = S.fill_enclosed(vol, MINW, radius=4, need_dirs=22, tile_blocks=2)
        V, F, C, G = vol.extract_mesh(MINW, tile_blocks=2, tag=rec.tag)
        inside = _in_hole(V, F)
        assert inside.sum() > 0
        # continued, not invented: the closed patch lies on the plane
        assert np.abs(V[F[inside]][..., 2] - PLANE).max() < 0.05
        # and every vertex in it is flagged as filled
        assert G[F[inside]].any(axis=1).all()

    def test_the_frontier_does_not_move(self):
        base = _field(hole=False)
        Vb, Fb, _ = base.extract_mesh(MINW, tile_blocks=2)
        vol = _field(hole=False)
        rec = S.fill_enclosed(vol, MINW, radius=4, need_dirs=22, tile_blocks=2)
        assert rec.voxels == 0
        V, F, C, G = vol.extract_mesh(MINW, tile_blocks=2, tag=rec.tag)
        assert V[:, 0].max() <= Vb[:, 0].max() + 1e-6
        assert len(F) == len(Fb)

    def test_space_behind_the_surface_stays_unobserved(self):
        vol = _field()
        rec = S.fill_enclosed(vol, MINW, radius=4, need_dirs=22, tile_blocks=2)
        X = vol.voxel_world(torch.arange(vol.n_blocks)) - 0.5
        deep = X[..., 2] < PLANE - TRUNC - 3
        assert not bool((rec.tag & deep).any())

    def test_a_measured_voxel_is_never_modified(self):
        before = _field()
        vol = _field()
        S.fill_enclosed(vol, MINW, radius=4, need_dirs=22, tile_blocks=2)
        measured = before.w >= MINW
        assert torch.equal(vol.w[measured], before.w[measured])
        assert torch.equal(vol.tsdf[measured], before.tsdf[measured])

    def test_a_filled_voxel_never_outweighs_a_measured_one(self):
        vol = _field()
        rec = S.fill_enclosed(vol, MINW, radius=4, need_dirs=22, tile_blocks=2)
        assert rec.voxels > 0
        assert torch.all(vol.w[rec.tag] == MINW)

    def test_a_gap_wider_than_the_cap_stays_open(self):
        vol = _field()
        rec = S.fill_enclosed(vol, MINW, radius=1, need_dirs=22, tile_blocks=2)
        V, F, C, G = vol.extract_mesh(MINW, tile_blocks=2, tag=rec.tag)
        c = V[F].mean(axis=1)
        centre = ((np.abs(c[:, 0] - 20.0) < 0.6) & (np.abs(c[:, 1] - 20.0) < 0.6))
        assert not centre.any()

    def test_an_undersized_radius_leaves_slivers_the_raw_fill_would_ship(self):
        """Why sealed-only exists: radius 3 on a 4-voxel gap fills the middle,
        misses the corners, and marching cubes builds walls between them."""
        vol = _field()
        rec = S.fill_enclosed(vol, MINW, radius=3, need_dirs=22, tile_blocks=2)
        V, F, C, G = vol.extract_mesh(MINW, tile_blocks=2, tag=rec.tag)
        inside = _in_hole(V, F)
        assert (np.abs(V[F[inside]][..., 2] - PLANE).max(axis=1) > 0.5).any()

    def test_sealed_extraction_reverts_a_bad_fill_to_exactly_the_baseline(self):
        Vb, Fb, _ = _field().extract_mesh(MINW, tile_blocks=2)
        before = _field()
        vol = _field()
        rec = S.fill_enclosed(vol, MINW, radius=3, need_dirs=22, tile_blocks=2)
        V, F, C, G, st = S.extract_sealed(vol, MINW, rec, tile_blocks=2)
        assert st["voxels_reverted"] > 0 and st["mesh_side_fallback_faces"] == 0
        assert not G.any()
        # nothing the cameras measured was lost on the way back out
        assert torch.equal(vol.w, before.w) and torch.equal(vol.tsdf, before.tsdf)
        assert len(F) == len(Fb)
        assert np.allclose(np.sort(V, axis=0), np.sort(Vb, axis=0))

    def test_sealed_extraction_keeps_a_clean_closure(self):
        vol = _field()
        rec = S.fill_enclosed(vol, MINW, radius=4, need_dirs=22, tile_blocks=2)
        V, F, C, G, st = S.extract_sealed(vol, MINW, rec, tile_blocks=2)
        assert st["rounds"] == 0 and st["voxels_reverted"] == 0
        inside = _in_hole(V, F)
        assert inside.sum() > 0
        assert np.abs(V[F[inside]][..., 2] - PLANE).max() < 0.05

    def test_tiling_does_not_change_the_answer(self):
        """A voxel filled by one tile is never evidence for the next."""
        a, b = _field(), _field()
        ra = S.fill_enclosed(a, MINW, radius=4, need_dirs=22, tile_blocks=1)
        rb = S.fill_enclosed(b, MINW, radius=4, need_dirs=22, tile_blocks=5)
        assert torch.equal(ra.tag, rb.tag)
        assert torch.allclose(a.tsdf, b.tsdf, atol=1e-5)


# ---------------------------------------------------------------------------
# sealed patches only
# ---------------------------------------------------------------------------


def _grid(n=10, duplicate=False):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    V = np.stack([ii.ravel(), jj.ravel(), np.zeros(n * n)], 1).astype(np.float32)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [(a, b, c), (a, c, d)]
    F = np.array(F, np.int64)
    C = np.full((len(V), 3), 100, np.uint8)
    if duplicate:
        # what a tile halo does: a second copy of every vertex and triangle
        F = np.concatenate([F, F + len(V)])
        V = np.concatenate([V, V])
        C = np.concatenate([C, C])
    return V, F, C


class TestSealedOnly:
    @pytest.mark.parametrize("duplicate", [False, True])
    def test_an_interior_patch_is_kept(self, duplicate):
        V, F, C = _grid(duplicate=duplicate)
        G = (np.abs(V[:, 0] - 4.5) < 1) & (np.abs(V[:, 1] - 4.5) < 1)
        V2, F2, C2, G2, st = S.keep_sealed_fill(V, F, C, G, 1e-3)
        assert st["frontier_patches"] == 0 and st["sealed_patches"] >= 1
        assert len(F2) == len(F)

    @pytest.mark.parametrize("duplicate", [False, True])
    def test_a_patch_on_the_rim_is_removed(self, duplicate):
        V, F, C = _grid(duplicate=duplicate)
        G = (V[:, 0] <= 1) & (np.abs(V[:, 1] - 4.5) < 1)
        V2, F2, C2, G2, st = S.keep_sealed_fill(V, F, C, G, 1e-3)
        assert st["frontier_patches"] >= 1
        assert st["frontier_faces"] > 0
        assert not G2[F2].any()
        assert len(V2) == len(np.unique(F2))

    def test_nothing_flagged_is_a_no_op(self):
        V, F, C = _grid()
        G = np.zeros(len(V), bool)
        V2, F2, C2, G2, st = S.keep_sealed_fill(V, F, C, G, 1e-3)
        assert len(F2) == len(F) and st["patches"] == 0
