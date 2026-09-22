"""Low-weight surface: what `min_weight` withheld, admitted only on frame tests.

`Glasses-scratch/wb-final-recon/fixit/holes/HOLES.md` classified every black
pixel of the canonical capture's phone proxy at the judged views. After the
consistency field made frames agree, the largest cause of black where two or
more frames HAD measured the surface was the field's weight gate: `min_weight`
is a sum of 1/z^2-falloff, incidence-weighted samples, so a far or oblique wall
many frames measured never reaches it. `SurfaceParams.low_weight_evidence`
lets such cubes emit when every corner was observed, and admits their faces
only when the frame tests say so -- the usual ones, plus: nothing saw through
the face, and the supporting cameras span a parallax.

Each test builds the situation from ray-cast depth, so the ground truth holds
no reconstruction.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_world_builder_surface import ROOM, _camera, _fuse
from tests.test_world_builder_surface_evidence import _views_tensors
from tests.test_world_builder_surface_pipeline import SESSION, WORLD, _params, _synthetic_world

from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP

# A near panel fills the left 58% of every frame, so each frame's median depth
# is the panel's (0.5) and the far wall beside it, 8x farther, earns ~1/64 of a
# sample per frame. The cameras are spread along y only, so the panel covers
# the same share of each.
NEAR = ((-0.9, -0.9, -0.52), (0.02, 0.9, -0.48))


def _far_wall_views(spread=0.35, n=6):
    return [(np.array([0.0, dy, -1.0]), np.array([0.0, dy, 5.0]), NEAR)
            for dy in np.linspace(-spread, spread, n)]


def _on_wall(V, F, pad=0.1):
    c = V[F].mean(axis=1)
    return np.abs(c[:, 2] - ROOM) < pad


class TestTheWeightGateWithholdsAFarMeasuredWall:
    """The premise, pinned: every frame measured the far wall, and the field's
    weight gate alone does not let it emit."""

    def test_min_weight_withholds_it_and_the_weak_floor_emits_it(self):
        params = S.SurfaceParams()
        vol = _fuse(_far_wall_views(), voxel=0.05, trunc=0.2, params=params)
        V, F, _C = vol.extract_mesh(params.min_weight)
        assert _on_wall(V, F).sum() == 0, "fixture: the wall already reaches min_weight"
        V2, F2, _C2, strong = vol.extract_mesh(params.min_weight, weak_floor=0.0)
        wall = _on_wall(V2, F2)
        assert wall.sum() > 50, "the observed far wall was not emitted as low-weight"
        assert not strong[wall].any(), "far-wall faces flagged as full-weight"

    def test_the_weak_floor_still_never_emits_from_an_unobserved_corner(self):
        # M1 again (test_world_builder_surface_evidence), with the gate lowered:
        # behind a measured wall the voxels have weight exactly 0.
        voxel = 0.05
        offs = ((0, 0), (0.2, 0), (0, 0.2), (-0.2, 0), (0, -0.2), (0.15, 0.15))
        views = [(np.array([dx, dy, -1.0]), np.array([dx, dy, 5.0]), None) for dx, dy in offs]
        vol = _fuse(views, voxel=voxel, trunc=7.58 * voxel,
                    params=S.SurfaceParams(depth_falloff=False))
        V, F, _C, strong = vol.extract_mesh(min_weight=2.0, weak_floor=0.0)
        assert len(F) > 0
        assert not (V[:, 2] > ROOM + 3 * voxel).any(), "a phantom sheet behind the wall"

    def test_without_the_floor_the_return_is_unchanged(self):
        vol = _fuse(_far_wall_views(), voxel=0.05, trunc=0.2)
        assert len(vol.extract_mesh(2.0)) == 3
        assert len(vol.extract_mesh(2.0, weak_floor=0.0)) == 4


class TestALowWeightFaceIsAdmittedOnlyByTheFrames:
    """`evidence_filter(weak=)` on hand-placed faces, so each rule is isolated."""

    TR = 0.06

    def _filter(self, V, F, views, weak, **kw):
        import torch

        tv, K = _views_tensors(views)
        params = S.SurfaceParams(**kw)
        return S.evidence_filter(np.asarray(V, np.float32), np.asarray(F), tv, K,
                                 lambda d: torch.full_like(d, self.TR), params,
                                 torch.device("cpu"), weak=weak)

    @staticmethod
    def _pair(z, s=0.05):
        # Two coincident triangles facing the cameras (-z): one will be flagged
        # low-weight, the other full weight, so the only difference is the flag.
        V = [(-s, -s, z), (-s, s, z), (s, -s, z), (-s, -s, z), (-s, s, z), (s, -s, z)]
        F = [(0, 1, 2), (3, 4, 5)]
        n = np.cross(np.subtract(V[1], V[0]), np.subtract(V[2], V[0]))
        assert n[2] < 0, "fixture: faces point at the cameras"
        return V, F

    @staticmethod
    def _cams(offs, slab=None, z=-1.0):
        return [(np.array([dx, dy, z]), np.array([dx, dy, 5.0]), slab) for dx, dy in offs]

    SPREAD = ((0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3))

    def test_a_measured_never_seen_through_wide_baseline_face_is_kept(self):
        V, F = self._pair(ROOM)
        keep, stats = self._filter(V, F, self._cams(self.SPREAD), weak=[True, False])
        assert list(keep) == [True, True], stats
        assert stats["weak_in"] == 1 and stats["weak_kept"] == 1

    def test_one_frame_seeing_through_removes_a_low_weight_face_only(self):
        # A panel at z=2 that four frames measured and one frame saw past: the
        # ratio test keeps it (1 < 2 x 4); a low-weight face needs zero.
        panel = ((-0.4, -0.4, 1.98), (0.4, 0.4, 2.02))
        V, F = self._pair(1.98)
        views = self._cams(self.SPREAD, slab=panel) + self._cams(((0.05, 0.05),))
        keep, stats = self._filter(V, F, views, weak=[True, False])
        assert list(keep) == [False, True], stats
        assert stats["dropped_weak_seen_through"] == 1

    def test_a_low_weight_face_measured_from_one_viewpoint_is_removed(self):
        V, F = self._pair(ROOM)
        huddle = ((0.004, 0.0), (-0.004, 0.0), (0.0, 0.004), (0.0, -0.004))
        keep, stats = self._filter(V, F, self._cams(huddle), weak=[True, False])
        assert list(keep) == [False, True], stats
        assert stats["dropped_weak_parallax"] == 1
        # and it is the parallax rule, not the frame count
        keep0, _ = self._filter(V, F, self._cams(huddle), weak=[True, False],
                                low_weight_min_parallax=0.0)
        assert list(keep0) == [True, True]

    def test_the_usual_tests_still_apply_to_low_weight_faces(self):
        V, F = self._pair(ROOM)
        keep, stats = self._filter(V, F, self._cams(((0.3, 0.0),)), weak=[True, False])
        assert list(keep) == [False, False], "one frame is never enough"

    def test_no_weak_mask_is_the_old_filter(self):
        V, F = self._pair(ROOM)
        views = self._cams(self.SPREAD)
        a, sa = self._filter(V, F, views, weak=None)
        b, sb = self._filter(V, F, views, weak=[False, False])
        assert list(a) == list(b) == [True, True]
        assert "weak_in" not in sa and sb["weak_kept"] == 0


class TestTheWeldCarriesFaceAttributes:
    def test_return_index_names_each_surviving_face(self):
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                      [0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        F = np.array([[0, 0, 1],      # degenerate
                      [0, 1, 2],      # a
                      [3, 5, 4],      # b
                      [6, 7, 8]], np.int64)   # duplicate of a (tile seam)
        V2, F2, _C, stats, src = S.weld_mesh(V, F, None, 1e-3, return_index=True)
        assert list(src) == [1, 2]
        assert stats["faces_duplicate"] == 1 and stats["faces_degenerate"] == 1
        # the surviving faces are exactly the inputs they name
        np.testing.assert_allclose(V2[F2], V[F[src]])


class TestThePhoneLevelKeepsItsRims:
    """`lod_boundary_weight`: a surface that stops where evidence stops is mostly
    rim, and at weight 1 decimation pulled rims inward."""

    @staticmethod
    def _lace(n=60, seed=0):
        # A rippled sheet cut to a wobbly outline with holes near its rim: the
        # lace a surface has where the evidence stops.
        rng = np.random.default_rng(seed)
        xs = np.linspace(0, 4, n)
        gx, gy = np.meshgrid(xs, xs)
        gz = 0.03 * np.sin(gx * 5) * np.cos(gy * 4) + 0.004 * rng.standard_normal(gx.shape)
        V = np.stack([gx, gy, gz], -1).reshape(-1, 3).astype(np.float32)
        idx = np.arange(n * n).reshape(n, n)
        cy, cx = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
        r = np.hypot(cx - n / 2, cy - n / 2) / (n / 2)
        ang = np.arctan2(cy - n / 2, cx - n / 2)
        keep = (r < 0.85 + 0.08 * np.sin(ang * 9)) & ~((r > 0.6) & (rng.random(r.shape) < 0.15))
        a, b, c, d = (idx[:-1, :-1][keep], idx[:-1, 1:][keep], idx[1:, :-1][keep],
                      idx[1:, 1:][keep])
        F = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)]).astype(np.int64)
        return V, F

    @staticmethod
    def _cover(V, F, res=240):
        """Which cells of an x-y raster the mesh covers (orthographic, from above)."""
        img = np.zeros((res, res), bool)
        P = V[:, :2] / 4.0 * (res - 1)
        for tri in F:
            p = P[tri]
            x0, y0 = np.floor(p.min(0)).astype(int)
            x1, y1 = np.ceil(p.max(0)).astype(int)
            xx, yy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            e0, e1 = p[1] - p[0], p[2] - p[0]
            den = e0[0] * e1[1] - e1[0] * e0[1]
            if abs(den) < 1e-12:
                continue
            wx, wy = xx - p[0, 0], yy - p[0, 1]
            u = (wx * e1[1] - e1[0] * wy) / den
            v = (e0[0] * wy - wx * e0[1]) / den
            m = (u >= 0) & (v >= 0) & (u + v <= 1)
            img[np.clip(yy[m].astype(int), 0, res - 1), np.clip(xx[m].astype(int), 0, res - 1)] = True
        return img

    def test_a_boundary_weight_opens_fewer_cracks(self):
        V, F = self._lace()
        C = np.full((len(V), 3), 128, np.uint8)
        c0 = self._cover(V, F)
        lost = {}
        for bw in (1.0, 100.0):
            V2, F2, _ = S.decimate(V, F, C, len(F) // 10, boundary_weight=bw)
            assert abs(len(F2) - len(F) // 10) <= 2
            lost[bw] = float((c0 & ~self._cover(V2, F2)).sum() / c0.sum())
        # measured: 3.6% of the outline's cells opened at weight 1, 1.8% at 100
        assert lost[100.0] < 0.75 * lost[1.0], lost

    def test_the_pack_passes_the_parameter(self):
        import inspect

        body = inspect.getsource(SP._build)
        assert body.count("boundary_weight=params.lod_boundary_weight") == 2


class TestTheParameters:
    def test_on_by_default_and_in_the_digest(self):
        p = S.SurfaceParams()
        assert p.low_weight_evidence is True
        assert p.low_weight_min_parallax == pytest.approx(0.05)
        assert p.lod_boundary_weight == pytest.approx(100.0)
        d = p.digest_fields()
        assert d != S.SurfaceParams(low_weight_evidence=False).digest_fields()
        assert d != S.SurfaceParams(low_weight_min_parallax=0.1).digest_fields()
        assert d != S.SurfaceParams(lod_boundary_weight=1.0).digest_fields()

    def test_the_live_preset_keeps_the_frame_tests(self):
        live = S.SurfaceParams.live()
        assert live.low_weight_evidence is True
        assert live.min_support_frames >= 2 and live.contradiction_ratio > 0


class TestThroughTheBuild:
    """Through `surfacify`, on a world whose far wall only low-weight evidence
    reaches."""

    def _views(self):
        return _far_wall_views(spread=0.4, n=8)

    def _wall_vertices(self, store):
        man = SP.read_surface_manifest(store, WORLD, SESSION)
        V, F, _C, _N = S.read_mesh_bytes(SP.read_surface_level(store, WORLD, SESSION, 0,
                                                               manifest=man))
        return int(_on_wall(V, F, pad=0.15).sum()), man

    def test_the_far_wall_appears_with_the_frame_tests_and_not_without(self, tmp_path):
        kw = dict(min_weight=2.0, min_support_frames=2, lod_face_targets=(0,),
                  mobile_level=0)
        off = _synthetic_world(tmp_path / "off", views=self._views(), anchors=True)
        r = SP.surfacify(off, WORLD, SESSION, params=_params(low_weight_evidence=False, **kw))
        n_off = self._wall_vertices(off)[0] if r.state == SP.STATE_OK else 0
        on = _synthetic_world(tmp_path / "on", views=self._views(), anchors=True)
        r = SP.surfacify(on, WORLD, SESSION, params=_params(**kw))
        assert r.state == SP.STATE_OK, r.detail
        n_on, man = self._wall_vertices(on)
        assert n_on > 50 and n_on > 3 * max(1, n_off), (n_on, n_off)
        ev = man["detail"]["evidence_filter"]
        assert ev["weak_in"] > 0 and ev["weak_kept"] > 0
        assert "low_weight_min_parallax" in man["closure"]
