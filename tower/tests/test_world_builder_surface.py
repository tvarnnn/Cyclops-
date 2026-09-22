"""The surface stage, tested by outcome.

The scene is synthetic and its answer is known by construction: a box room
with a slab in it, seen by cameras placed around the inside. That makes it
possible to ask the questions that matter -- is there surface where a camera
looked, is there NO surface where none did, does a thing that moved get
carved away -- rather than asserting that some function returned something.

The refusals are tested as outcomes too. A stage that cannot run has to say
so and leave nothing behind; the whole point of the artifact discipline is
that a half-written reconstruction never becomes the world.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tower.world_builder import surface as S


# ---------------------------------------------------------------------------
# a synthetic room, and cameras that look at it
# ---------------------------------------------------------------------------

ROOM = 3.0          # half-extent of the room, in scene units
WALL_Z = ROOM       # the far wall


def _camera(K_f=300.0, w=160, h=120):
    return np.array([[K_f, 0, w / 2], [0, K_f, h / 2], [0, 0, 1.0]]), w, h


def _look_from(eye, target, up=(0.0, -1.0, 0.0)):
    """World-to-camera (R, t) for a camera at `eye`, OpenCV axes."""
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.stack([x, y, z], axis=1)        # columns
    R = R_wc.T                                # world-to-camera
    return R, -R @ eye


def _render_box_depth(R, t, K, w, h, *, slab=None):
    """Exact depth of a box room (and an optional slab) by ray casting.

    Analytic rather than rasterised, so the test's ground truth has no
    reconstruction in it.
    """
    C = -R.T @ t
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dirs_cam = np.stack([(uu - K[0, 2]) / K[0, 0],
                         (vv - K[1, 2]) / K[1, 1],
                         np.ones_like(uu)], -1)
    dirs = dirs_cam @ R                       # R^T applied on the right
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    best = np.full((h, w), np.inf)

    def slab_hit(lo, hi):
        nonlocal best
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (lo - C) / dirs
            t1 = (hi - C) / dirs
        tmin = np.minimum(t0, t1).max(axis=-1)
        tmax = np.maximum(t0, t1).min(axis=-1)
        hit = (tmax >= np.maximum(tmin, 1e-6))
        cand = np.where(hit, np.where(tmin > 1e-6, tmin, tmax), np.inf)
        best = np.minimum(best, np.where(np.isfinite(cand), cand, np.inf))

    # the room is the inside of a box: its far side is what a ray hits
    slab_hit(np.array([-ROOM, -ROOM, -ROOM]), np.array([ROOM, ROOM, ROOM]))
    if slab is not None:
        slab_hit(np.asarray(slab[0], float), np.asarray(slab[1], float))
    depth = best * dirs_cam[..., 2] / np.linalg.norm(dirs_cam, axis=-1)
    return np.where(np.isfinite(depth), depth, np.nan).astype(np.float32)


def _fuse(views, *, params=None, voxel=0.05, trunc=None, slab=None,
          device="cpu"):
    import torch

    torch_dev = torch.device(device)
    params = params or S.SurfaceParams()
    K, w, h = _camera()
    trunc = trunc if trunc is not None else 4 * voxel
    vol = S.SurfaceVolume(voxel, trunc, device=torch_dev)

    prepared = []
    for eye, target, view_slab in views:
        R, t = _look_from(eye, target)
        d = _render_box_depth(R, t, K, w, h, slab=view_slab)
        zt = torch.as_tensor(d, device=torch_dev)
        ok = torch.isfinite(zt) & (zt > 1e-3)
        rgb = torch.full((h, w, 3), 200.0, device=torch_dev)
        prepared.append((zt, ok, rgb,
                         torch.as_tensor(R, device=torch_dev).float(),
                         torch.as_tensor(t, device=torch_dev).float()))

    keys = [vol.blocks_for_depth(z, ok, R, t, K) for z, ok, _, R, t in prepared]
    vol.reserve(torch.cat(keys))
    for z, ok, rgb, R, t in prepared:
        vol.integrate(z, ok, rgb, R, t, K, params=params)
    return vol


def _ring(n=8, radius=1.6, slab=None):
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        eye = np.array([radius * math.cos(a), 0.0, radius * math.sin(a)])
        out.append((eye, eye * 3.0, slab))
    return out


# ---------------------------------------------------------------------------
# the format
# ---------------------------------------------------------------------------


class TestTheMeshFormatIsSelfChecking:
    """A torn artifact must be refused on read, not misread.

    This is the `BadZipFile` lesson expressed in a format: the reader lives
    in a different process from the writer, and the failure that ended a
    795-keyframe session was a file that looked plausible until it did not.
    """

    def _mesh(self, n_v=400, n_f=700):
        rng = np.random.default_rng(7)
        V = (rng.normal(size=(n_v, 3)) * 2).astype(np.float32)
        F = rng.integers(0, n_v, size=(n_f, 3)).astype(np.int64)
        C = rng.integers(0, 256, size=(n_v, 3)).astype(np.uint8)
        return V, F, C

    def test_a_mesh_survives_the_round_trip(self):
        V, F, C = self._mesh()
        N = S.vertex_normals(V, F)
        V2, F2, C2, N2 = S.read_mesh_bytes(S.write_mesh_bytes(V, F, C, N))
        assert np.array_equal(F, F2)
        assert np.array_equal(C, C2)
        span = float((V.max(0) - V.min(0)).max())
        # positions are quantised to 16 bits across the mesh's own box
        assert np.abs(V - V2).max() < span / 30000

    def test_quantisation_is_far_finer_than_the_voxel_it_describes(self):
        """The compression must not be visible in the geometry."""
        V, F, C = self._mesh()
        V2, _, _, _ = S.read_mesh_bytes(S.write_mesh_bytes(V, F, C))
        span = float((V.max(0) - V.min(0)).max())
        voxel = span / 200        # a generous voxel for a scene this size
        assert np.abs(V - V2).max() < voxel / 100

    @pytest.mark.parametrize("cut", ["header", "half", "one_byte"])
    def test_a_truncated_buffer_is_refused(self, cut):
        V, F, C = self._mesh()
        buf = S.write_mesh_bytes(V, F, C)
        sliced = {"header": buf[:10], "half": buf[:len(buf) // 2],
                  "one_byte": buf[:-1]}[cut]
        with pytest.raises(S.SurfaceUnavailable):
            S.read_mesh_bytes(sliced)

    def test_a_foreign_buffer_is_refused(self):
        with pytest.raises(S.SurfaceUnavailable):
            S.read_mesh_bytes(b"NOTAMESH" + b"\0" * 64)

    def test_a_future_schema_is_refused_rather_than_guessed(self):
        V, F, C = self._mesh()
        buf = bytearray(S.write_mesh_bytes(V, F, C))
        buf[20:24] = (S.SURFACE_SCHEMA_VERSION + 1).to_bytes(4, "little")
        with pytest.raises(S.SurfaceUnavailable):
            S.read_mesh_bytes(bytes(buf))

    def test_an_empty_mesh_is_a_legal_artifact(self):
        """An honest "nothing was reconstructed" must round-trip, because the
        alternative is a reader that cannot distinguish it from corruption."""
        buf = S.write_mesh_bytes(np.zeros((0, 3), np.float32),
                                 np.zeros((0, 3), np.int64), None)
        V, F, C, N = S.read_mesh_bytes(buf)
        assert len(V) == 0 and len(F) == 0

    def test_small_meshes_use_16_bit_indices(self):
        """Same face count, different vertex count: the index width is the
        only thing that may differ, and it must narrow when it safely can."""
        header, per_vertex = 48, 9        # uint16[3] position + uint8[3] colour
        faces = 120
        small = len(S.write_mesh_bytes(*self._mesh(n_v=100, n_f=faces)))
        big = len(S.write_mesh_bytes(*self._mesh(n_v=70000, n_f=faces)))
        assert small - header - 100 * per_vertex == faces * 3 * 2
        assert big - header - 70000 * per_vertex == faces * 3 * 4


# ---------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------


class TestSurfaceAppearsOnlyWhereACameraLooked:

    def test_a_watched_wall_becomes_surface(self):
        vol = _fuse([(np.array([0.0, 0.0, -1.5]), np.array([0.0, 0.0, 5.0]), None)])
        V, F, C = vol.extract_mesh(min_weight=0.5)
        assert len(F) > 0, "a camera looking straight at a wall produced no surface"
        # the wall it looked at is at z = +ROOM
        near_wall = np.abs(V[:, 2] - WALL_Z) < 0.25
        assert near_wall.mean() > 0.5

    def test_space_no_camera_measured_stays_absent(self):
        """The whole claim of the format. One camera looking one way must not
        produce a closed room."""
        vol = _fuse([(np.array([0.0, 0.0, -1.5]), np.array([0.0, 0.0, 5.0]), None)])
        V, F, C = vol.extract_mesh(min_weight=0.5)
        behind = V[:, 2] < -ROOM + 0.3
        assert behind.mean() < 0.02, (
            "surface appeared on the wall behind the camera, which nothing "
            "observed -- the extractor is closing over unobserved space")

    def test_evidence_threshold_gates_the_surface(self):
        vol = _fuse([(np.array([0.0, 0.0, -1.5]), np.array([0.0, 0.0, 5.0]), None)])
        _, few, _ = vol.extract_mesh(min_weight=0.5)
        _, many, _ = vol.extract_mesh(min_weight=50.0)
        assert len(many) < len(few)

    def test_more_viewpoints_produce_more_surface(self):
        """The live promise: the scene fills in as the wearer keeps looking."""
        counts = []
        for n in (1, 3, 8):
            vol = _fuse(_ring(n=n))
            _, F, _ = vol.extract_mesh(min_weight=0.5)
            counts.append(len(F))
        assert counts[0] < counts[1] < counts[2], counts


class TestFreeSpaceCarving:

    def test_a_thing_that_moved_away_is_carved_out(self):
        """A slab present in one frame and gone in the rest must not survive.

        This is the dog, and the wearer's hands. A point pipeline keeps them
        because every point was genuinely measured once; a field deletes them
        because later frames measured the emptiness they left behind.

        The cameras are placed explicitly rather than taken from `_ring`,
        because the test camera has a 15 degree half-angle and ring cameras
        looking outward never see through each other's subject -- the first
        version of this test created no ghost at all and asserted on it.
        Here four cameras on one side look straight through the slab's
        position at the far wall, which is exactly the evidence carving needs.
        """
        slab = (np.array([2.0, -0.45, -0.45]), np.array([2.4, 0.45, 0.45]))
        far = np.array([3.0, 0.0, 0.0])

        views = [(np.array([1.0, 0.0, 0.0]), far, slab)]          # sees it
        for dy, dz in ((0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (-0.2, -0.2)):
            views.append((np.array([-1.6, dy, dz]), far, None))   # sees through

        in_slab = lambda V: np.all(
            (V > slab[0] - 0.12) & (V < slab[1] + 0.12), axis=1)

        V_on, _, _ = _fuse(views, params=S.SurfaceParams(carve=True)
                           ).extract_mesh(min_weight=0.5)
        V_off, _, _ = _fuse(views, params=S.SurfaceParams(carve=False)
                            ).extract_mesh(min_weight=0.5)

        ghost_on, ghost_off = int(in_slab(V_on).sum()), int(in_slab(V_off).sum())
        assert ghost_off > 0, "the test did not manage to create a ghost at all"
        assert ghost_on < ghost_off * 0.5, (
            f"carving left {ghost_on} of {ghost_off} ghost vertices; a thing "
            f"seen once and then seen through should not survive")

    def test_carving_does_not_delete_a_surface_everything_agrees_on(self):
        """The other half of the claim. Carving that ate real geometry would
        pass the test above and be useless."""
        views = [(np.array([-1.6, dy, dz]), np.array([3.0, 0.0, 0.0]), None)
                 for dy, dz in ((0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (-0.2, -0.2))]
        on = len(_fuse(views, params=S.SurfaceParams(carve=True)
                       ).extract_mesh(min_weight=0.5)[1])
        off = len(_fuse(views, params=S.SurfaceParams(carve=False)
                        ).extract_mesh(min_weight=0.5)[1])
        assert on > off * 0.8, (
            f"carving cut the agreed surface from {off} to {on} faces")


class TestTruncationMustExceedTheDisagreement:
    """The finding that decided coverage, kept as a regression.

    Frames disagree with each other by a few percent of scene depth. If the
    truncation band is narrower than that disagreement, a frame that ran long
    writes free space exactly where a correct frame wrote surface, and the two
    erase each other. A narrower truncation is not a tighter reconstruction.
    """

    def _noisy(self, rel_error, trunc_voxels, voxel=0.05):
        import torch

        rng = np.random.default_rng(3)
        K, w, h = _camera()
        vol = S.SurfaceVolume(voxel, trunc_voxels * voxel, device=torch.device("cpu"))
        prepared = []
        for eye, target, _ in _ring(n=10):
            R, t = _look_from(eye, target)
            d = _render_box_depth(R, t, K, w, h)
            d = d * (1.0 + rng.normal(0, rel_error))      # per-frame scale error
            zt = torch.as_tensor(d, device=vol.dev)
            ok = torch.isfinite(zt) & (zt > 1e-3)
            prepared.append((zt, ok, torch.full((h, w, 3), 200.0, device=vol.dev),
                             torch.as_tensor(R, device=vol.dev).float(),
                             torch.as_tensor(t, device=vol.dev).float()))
        vol.reserve(torch.cat([vol.blocks_for_depth(z, ok, R, t, K)
                               for z, ok, _, R, t in prepared]))
        for z, ok, rgb, R, t in prepared:
            vol.integrate(z, ok, rgb, R, t, K, params=S.SurfaceParams())
        _, F, _ = vol.extract_mesh(min_weight=0.5)
        return len(F)

    def test_a_truncation_narrower_than_the_error_loses_surface(self):
        # median depth here is about 3 units, so 3% is ~0.09 -- wider than a
        # 1-voxel (0.05) band and narrower than an 8-voxel one
        narrow = self._noisy(0.03, trunc_voxels=1.0)
        wide = self._noisy(0.03, trunc_voxels=8.0)
        assert wide > narrow * 1.2, (
            f"narrow truncation kept {narrow} faces and wide kept {wide}; the "
            f"frames were supposed to carve each other out at the narrow band")

    def test_truncation_is_raised_to_the_measured_error(self):
        p = S.SurfaceParams(trunc_voxels=3.0, trunc_error_multiple=2.0)
        # 2.3% of a 4.07 scene is 0.094; two of those is 0.187, well over
        # three voxels of 0.024
        got = S.truncation_for(p, voxel=0.0244, median_depth=4.07,
                               median_held_out_rel=0.023)
        assert got == pytest.approx(2.0 * 0.023 * 4.07, rel=1e-6)

    def test_truncation_never_falls_below_the_voxel_floor(self):
        p = S.SurfaceParams(trunc_voxels=3.0, trunc_error_multiple=2.0)
        got = S.truncation_for(p, voxel=0.05, median_depth=4.0,
                               median_held_out_rel=0.0001)
        assert got == pytest.approx(0.15)

    def test_a_world_with_no_measured_error_still_gets_a_truncation(self):
        p = S.SurfaceParams()
        assert S.truncation_for(p, 0.02, 4.0, None) == pytest.approx(0.06)


class TestEveryLengthScalesWithTheScene:
    """The gauge is arbitrary: `global_solve` never calls `normalize()`, and
    the same room has solved to a ten-unit extent and a three-hundred-unit
    one. A parameter expressed in absolute units would shatter one and
    collapse the other."""

    def test_the_same_scene_at_two_gauges_reconstructs_the_same_way(self):
        import torch

        def faces_at(gauge):
            K, w, h = _camera()
            p = S.SurfaceParams()
            voxel = p.voxel_frac * 3.0 * gauge
            vol = S.SurfaceVolume(voxel, 4 * voxel, device=torch.device("cpu"))
            prepared = []
            for eye, target, _ in _ring(n=4):
                R, t = _look_from(eye, target)
                d = _render_box_depth(R, t, K, w, h) * gauge
                R2, t2 = R, t * gauge
                zt = torch.as_tensor(d, device=vol.dev)
                ok = torch.isfinite(zt) & (zt > 1e-3)
                prepared.append((zt, ok, torch.full((h, w, 3), 200.0, device=vol.dev),
                                 torch.as_tensor(R2, device=vol.dev).float(),
                                 torch.as_tensor(t2, device=vol.dev).float()))
            vol.reserve(torch.cat([vol.blocks_for_depth(z, ok, R, t, K)
                                   for z, ok, _, R, t in prepared]))
            for z, ok, rgb, R, t in prepared:
                vol.integrate(z, ok, rgb, R, t, K, params=p)
            _, F, _ = vol.extract_mesh(min_weight=0.5)
            return len(F)

        small, large = faces_at(1.0), faces_at(12.0)
        assert large == pytest.approx(small, rel=0.15), (
            f"the same room reconstructed to {small} faces at one gauge and "
            f"{large} at another; a length is absolute somewhere")


class TestBlockKeysRefuseToAlias:
    """`voxel_reduce` learned this the hard way: outside the keyable range two
    cells share a key and the reduction silently MERGES them, averaging points
    from opposite ends of a scene into one."""

    def test_a_coordinate_outside_the_range_raises(self):
        import torch

        bc = torch.tensor([[1 << 22, 0, 0]], dtype=torch.int64)
        with pytest.raises(S.SurfaceUnavailable):
            S.block_key(bc)

    def test_keys_round_trip_inside_the_range(self):
        import torch

        bc = torch.tensor([[-5, 12, 900], [0, 0, 0], [7, -3, 1]], dtype=torch.int64)
        assert torch.equal(S.block_coords(S.block_key(bc)), bc)


class TestBlockAllocationIsTheSameSetMadeCheaply:
    """`blocks_for_depth` expands every sampled pixel's block by a truncation
    shell. It used to do that in coordinate space and deduplicate with
    `unique(dim=0)` -- a lexicographic row sort over pixels x shell x 3 int64.
    At a shell radius of 4 that is a 43.7M-row table per keyframe; it was 530 s
    of an 873 s build, and on a tight room it died inside the sort with
    `cudaErrorIllegalAddress`.

    The replacement collapses pixels to blocks first and expands in KEY space,
    because a key is affine in its coordinate. That is only a speed-up if it
    is the same answer, so these tests hold it to the row form, key for key
    and in order, including the range guard at the shell's extremes.
    """

    @staticmethod
    def _row_form(vol, depth, valid, R, t, K):
        """The previous implementation, kept as the reference. The shell radius
        is now per pixel (the band can be depth-proportional) with
        `SHELL_MARGIN_BLOCKS` of margin; the row-form expansion is verbatim."""
        import torch

        vy, vx = torch.nonzero(valid, as_tuple=True)
        step = max(1, vy.numel() // 60000)
        vy, vx = vy[::step], vx[::step]
        z = depth[vy, vx]
        x = (vx.to(torch.float32) - float(K[0, 2])) / float(K[0, 0]) * z
        y = (vy.to(torch.float32) - float(K[1, 2])) / float(K[1, 1]) * z
        Xw = (torch.stack([x, y, z], 1) - t) @ R
        bc = torch.floor(Xw / (vol.voxel * S.BLOCK)).to(torch.int64)
        rads = torch.tensor([int(math.ceil(vol.trunc_at(float(zz)) / (vol.voxel * S.BLOCK)))
                             + S.SHELL_MARGIN_BLOCKS for zz in z])
        cands = []
        for r in sorted(set(rads.tolist())):
            offs = torch.arange(-r, r + 1)
            oz, oy, ox = torch.meshgrid(offs, offs, offs, indexing="ij")
            off = torch.stack([ox, oy, oz], -1).reshape(-1, 3)
            cands.append((bc[rads == r].unsqueeze(1) + off.unsqueeze(0)).reshape(-1, 3))
        return S.block_key(torch.unique(torch.cat(cands), dim=0))

    @pytest.mark.parametrize("voxel,trunc", [(0.05, 0.15), (0.03, 0.40),
                                             (0.06, 1.03), (0.12, 0.30)])
    def test_the_block_set_is_identical_to_the_row_form(self, voxel, trunc):
        import torch

        K, w, h = _camera()
        R, t = _look_from((0.4, -0.2, 0.3), (0.0, 0.0, ROOM))
        depth = torch.as_tensor(_render_box_depth(
            R, t, K, w, h, slab=((-1.0, -1.0, 1.0), (0.5, 0.2, 1.4))))
        valid = torch.isfinite(depth)
        R, t = torch.as_tensor(R).float(), torch.as_tensor(t).float()
        vol = S.SurfaceVolume(voxel, trunc, device=torch.device("cpu"))
        got = vol.blocks_for_depth(depth, valid, R, t, K)
        want = self._row_form(vol, depth, valid, R, t, K)
        assert got.numel() > 0
        assert torch.equal(got, want)

    def test_a_shell_that_leaves_the_keyable_range_still_raises(self):
        """The CENTRE block is keyable here; only the shell's outer layer is
        not. Checking only the centres would alias silently."""
        import torch

        K, w, h = _camera()
        voxel, trunc = 0.05, 0.15
        edge = ((1 << 20) - 1) * S.BLOCK * voxel      # the last keyable block, +z
        depth = torch.full((h, w), float(edge))
        valid = torch.zeros((h, w), dtype=torch.bool)
        valid[h // 2, w // 2] = True                  # the principal ray only
        vol = S.SurfaceVolume(voxel, trunc, device=torch.device("cpu"))
        S.block_key(torch.tensor([[0, 0, (1 << 20) - 2]], dtype=torch.int64))
        with pytest.raises(S.SurfaceUnavailable):
            vol.blocks_for_depth(depth, valid, torch.eye(3), torch.zeros(3), K)

    def test_the_returned_keys_do_not_pin_the_sort_buffer(self):
        """The offline build holds every frame's keys until `reserve`. On CUDA
        `torch.unique` returns a view of its full-length sort buffer, so a
        result that is not copied keeps (centres x shell) int64 alive per
        frame -- 823 MiB of peak allocation over a 346-frame build."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("the retained-buffer behaviour is a CUDA allocator property")
        dev = torch.device("cuda")
        K, w, h = _camera()
        R, t = _look_from((0.4, -0.2, 0.3), (0.0, 0.0, ROOM))
        depth = torch.as_tensor(_render_box_depth(R, t, K, w, h), device=dev)
        vol = S.SurfaceVolume(0.03, 0.40, device=dev)
        keys = vol.blocks_for_depth(depth, torch.isfinite(depth),
                                    torch.as_tensor(R, device=dev).float(),
                                    torch.as_tensor(t, device=dev).float(), K)
        assert keys.numel() > 0
        assert keys.untyped_storage().nbytes() == keys.numel() * keys.element_size()


class TestExtractionVisitsOnlyTilesThatHoldField:
    """Marching cubes must iterate the field, not its bounding box.

    The tile list was the full bounding grid: the product of three extents.
    The real 795-keyframe field walk had 1,925,280 tiles of which 383 held a
    block, and spent 520 s of mesh stage against 5 s of fusion. The
    replacement is only acceptable if it is the SAME tile set the old loop
    found non-empty -- including tiles that see a block only through their
    one-block halo, which is where a careless version silently changes the
    seams.
    """

    TILE = 12

    @staticmethod
    def _full_grid_non_empty(bc, lo, hi, tile):
        """The previous loop's behaviour, brute force: every tile of the
        bounding grid whose (tile + 1)^3 block window holds a block."""
        out = []
        for x in range(int(lo[0]), int(hi[0]) + 1, tile):
            for y in range(int(lo[1]), int(hi[1]) + 1, tile):
                for z in range(int(lo[2]), int(hi[2]) + 1, tile):
                    d = bc - np.array([x, y, z])
                    if np.any(np.all((d >= 0) & (d <= tile), axis=1)):
                        out.append((x, y, z))
        return out

    def test_it_is_exactly_the_non_empty_part_of_the_full_grid(self):
        rng = np.random.default_rng(11)
        bc = rng.integers(-45, 45, size=(300, 3))
        # blocks on tile boundaries, read by the previous tile's halo, on one,
        # two and three axes at once
        bc = np.vstack([bc, [[-45 + 12, 0, 0], [-45 + 24, -45 + 12, 0],
                             [-45 + 36, -45 + 24, -45 + 12]]])
        lo, hi = bc.min(0), bc.max(0)
        assert S.occupied_tiles(bc, lo, hi, self.TILE) == \
            self._full_grid_non_empty(bc, lo, hi, self.TILE)

    def test_a_restricted_box_still_sees_blocks_just_outside_it(self):
        """The live path passes `only_blocks`: the box comes from those, but
        the tiles read the WHOLE field, so a block just past the box's corner
        is still read through the halo of the box's last tile."""
        rng = np.random.default_rng(5)
        allb = rng.integers(0, 60, size=(400, 3))
        sel = allb[:40]
        lo, hi = sel.min(0), sel.max(0)
        assert S.occupied_tiles(allb, lo, hi, self.TILE) == \
            self._full_grid_non_empty(allb, lo, hi, self.TILE)

    def test_a_diagonal_walk_skips_almost_all_of_its_box(self):
        t = np.arange(0, 900)
        bc = np.stack([t, t, t], 1)                 # a thin diagonal line
        lo, hi = bc.min(0), bc.max(0)
        got = S.occupied_tiles(bc, lo, hi, self.TILE)
        grid = len(range(0, 900, self.TILE)) ** 3
        assert len(got) < grid / 100             # 519 of 421,875
        assert got == sorted(got)                   # the old loop's order

    def test_the_mesh_is_unchanged(self, monkeypatch):
        """End to end on the synthetic room: identical arrays either way."""
        vol = _fuse(_ring(n=6))
        new = vol.extract_mesh(min_weight=0.5)

        def full_grid(bc, lo, hi, tile):
            return [(x, y, z)
                    for x in range(int(lo[0]), int(hi[0]) + 1, tile)
                    for y in range(int(lo[1]), int(hi[1]) + 1, tile)
                    for z in range(int(lo[2]), int(hi[2]) + 1, tile)]

        monkeypatch.setattr(S, "occupied_tiles", full_grid)
        old = vol.extract_mesh(min_weight=0.5)
        assert len(new[1]) > 0
        for a, b in zip(new, old):
            assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# mesh cleanup
# ---------------------------------------------------------------------------


class TestMeshCleanup:

    def _two_components(self):
        """One big grid and one tiny detached triangle."""
        n = 12
        xs, ys = np.meshgrid(np.arange(n), np.arange(n))
        V = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], 1).astype(np.float32)
        F = []
        for i in range(n - 1):
            for j in range(n - 1):
                a = i * n + j
                F += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
        F = np.array(F, np.int64)
        V2 = np.vstack([V, np.array([[50, 50, 0], [51, 50, 0], [50, 51, 0]], np.float32)])
        F2 = np.vstack([F, np.array([[len(V), len(V) + 1, len(V) + 2]], np.int64)])
        C = np.full((len(V2), 3), 128, np.uint8)
        return V2, F2, C

    def test_a_tiny_island_is_dropped(self):
        V, F, C = self._two_components()
        V2, F2, C2, stats = drop = S.drop_small_components(V, F, C, 0.01)
        assert stats["components"] == 2
        assert stats["dropped"] == 1
        assert len(F2) == len(F) - 1
        assert V2.max() < 40, "the far island survived"

    def test_nothing_is_dropped_when_the_threshold_is_off(self):
        V, F, C = self._two_components()
        _, F2, _, stats = S.drop_small_components(V, F, C, 0.0)
        assert len(F2) == len(F)

    def test_smoothing_moves_vertices_without_collapsing_the_model(self):
        V, F, C = self._two_components()
        V2, moved = S.taubin_smooth(V, F, iterations=6, lam=0.5, mu=-0.53)
        assert moved >= 0
        before = np.ptp(V[:144], axis=0)
        after = np.ptp(V2[:144], axis=0)
        # Taubin's expanding pass is what stops a Laplacian smooth from
        # shrinking the room; allow a little loss, not a collapse
        assert (after[:2] > before[:2] * 0.9).all(), (before, after)

    def test_smoothing_is_a_no_op_at_zero_iterations(self):
        V, F, C = self._two_components()
        V2, moved = S.taubin_smooth(V, F, 0, 0.5, -0.53)
        assert moved == 0.0
        assert np.array_equal(V, V2)


class TestVertexNormals:

    def test_a_flat_sheet_has_one_normal(self):
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
        F = np.array([[0, 1, 2], [1, 3, 2]], np.int64)
        N = S.vertex_normals(V, F)
        assert np.allclose(np.abs(N[:, 2]), 1.0, atol=1e-5)

    def test_an_empty_mesh_has_no_normals_and_does_not_raise(self):
        N = S.vertex_normals(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64))
        assert N.shape == (0, 3)



class TestTheTileSeamsAreWelded:
    """Extraction meshes every halo cube twice, once per tile. Unwelded, the
    canonical world carried 587,178 duplicate triangles (18.3%), and pruning
    measured "the largest component" against one tile of wall."""

    def _wall_across_tiles(self):
        import torch

        # a flat wall long enough to cross several extraction tiles
        K, w, h = _camera(K_f=120.0, w=240, h=120)
        vol = S.SurfaceVolume(0.05, 0.2, device=torch.device("cpu"))
        prepared = []
        for x in (-2.0, -1.0, 0.0, 1.0, 2.0):
            R, t = _look_from((x, 0.0, 0.0), (x, 0.0, ROOM))
            d = _render_box_depth(R, t, K, w, h)
            zt = torch.as_tensor(d)
            ok = torch.isfinite(zt)
            prepared.append((zt, ok, torch.full((h, w, 3), 180.0),
                             torch.as_tensor(R).float(), torch.as_tensor(t).float()))
        vol.reserve(torch.cat([vol.blocks_for_depth(z, ok, R, t, K)
                               for z, ok, _, R, t in prepared]))
        for z, ok, rgb, R, t in prepared:
            vol.integrate(z, ok, rgb, R, t, K, params=S.SurfaceParams())
        return vol, vol.extract_mesh(min_weight=0.5, tile_blocks=2)

    def test_duplicates_are_removed_and_nothing_else(self):
        vol, (V, F, C) = self._wall_across_tiles()
        V2, F2, C2, stats = S.weld_mesh(V, F, C, quantum=vol.voxel * 1e-3)
        assert stats["faces_duplicate"] > 0, "the fixture never crossed a tile seam"
        assert len(F2) == len(F) - stats["faces_duplicate"] - stats["faces_degenerate"]
        assert len(np.unique(np.sort(F2, axis=1), axis=0)) == len(F2)
        assert len(C2) == len(V2)

    def test_the_wall_is_one_component_once_welded(self):
        vol, (V, F, C) = self._wall_across_tiles()
        V2, F2, C2, _ = S.weld_mesh(V, F, C, quantum=vol.voxel * 1e-3)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        def count(v, f):
            e0 = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
            e1 = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
            g = coo_matrix((np.ones(len(e0)), (e0, e1)), shape=(len(v), len(v)))
            n, labels = connected_components(g, directed=False)
            return len(np.unique(labels[f[:, 0]]))

        assert count(V, F) > count(V2, F2)
        assert count(V2, F2) <= 2

    def test_welding_twice_changes_nothing(self):
        vol, (V, F, C) = self._wall_across_tiles()
        once = S.weld_mesh(V, F, C, quantum=vol.voxel * 1e-3)
        twice = S.weld_mesh(once[0], once[1], once[2], quantum=vol.voxel * 1e-3)
        assert np.array_equal(once[1], twice[1])
        assert twice[3]["faces_duplicate"] == 0 and twice[3]["vertices_merged"] == 0

    def test_an_empty_mesh_welds_to_an_empty_mesh(self):
        V, F, C, stats = S.weld_mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64),
                                     None, quantum=0.001)
        assert len(F) == 0 and stats["faces_duplicate"] == 0
