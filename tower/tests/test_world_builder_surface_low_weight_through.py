"""A low-weight face many frames measured survives a FEW frames that saw past it.

`Glasses-scratch/wb-final-recon/fixit/geom2/GEOM2.md`: the black fan-shaped
tears in the canonical capture's ceiling were low-weight ceiling faces that 15
frames of one pass supported and 2 frames of another pass "saw through", because
the second pass's depth put the ceiling slightly higher. "No frame at all may
see through a low-weight face" removed them. `SurfaceParams.
low_weight_through_frac` (0.34) with `low_weight_through_min_support` (8)
tolerates such a minority; held out of fusion, the faces it admits were
contradicted about as often as the rest of the surface.

Each test builds the situation from ray-cast depth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.test_world_builder_surface_evidence import _views_tensors

from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP

TR = 0.06
PANEL = ((-0.4, -0.4, 1.98), (0.4, 0.4, 2.02))


def _pair(z, s=0.05):
    # Two coincident triangles facing the cameras: the first is flagged
    # low-weight, the second full weight, so the flag is the only difference.
    V = [(-s, -s, z), (-s, s, z), (s, -s, z), (-s, -s, z), (-s, s, z), (s, -s, z)]
    return V, [(0, 1, 2), (3, 4, 5)]


def _ring(n, r=0.3, slab=PANEL):
    return [(np.array([r * math.cos(a), r * math.sin(a), -1.0]),
             np.array([r * math.cos(a), r * math.sin(a), 5.0]), slab)
            for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def _past(n):
    # Frames that do not hold the panel: their depth at its pixels is the wall
    # behind, i.e. they saw through it.
    return [(np.array([0.02 * k, 0.01, -1.0]), np.array([0.02 * k, 0.01, 5.0]), None)
            for k in range(n)]


def _filter(views, **kw):
    import torch

    V, F = _pair(1.98)
    tv, K = _views_tensors(views)
    return S.evidence_filter(np.asarray(V, np.float32), np.asarray(F), tv, K,
                             lambda d: torch.full_like(d, TR), S.SurfaceParams(**kw),
                             torch.device("cpu"), weak=[True, False])


class TestAFewSeeThroughFramesAgainstMany:
    def test_twelve_supporters_and_two_past_it_is_kept(self):
        keep, stats = _filter(_ring(12) + _past(2))
        assert list(keep) == [True, True], stats
        assert stats["weak_kept"] == 1 and stats["dropped_weak_seen_through"] == 0

    def test_without_the_tolerance_the_same_face_is_removed(self):
        keep, stats = _filter(_ring(12) + _past(2), low_weight_through_frac=0.0)
        assert list(keep) == [False, True], stats
        assert stats["dropped_weak_seen_through"] == 1

    def test_too_few_supporters_are_never_enough(self):
        # 4 supporters, 1 past it: under the ratio, but below the support floor.
        keep, stats = _filter(_ring(4) + _past(1))
        assert list(keep) == [False, True], stats

    def test_the_support_floor_is_the_parameter(self):
        keep, _ = _filter(_ring(6) + _past(1))
        assert list(keep) == [False, True]
        keep, _ = _filter(_ring(6) + _past(1), low_weight_through_min_support=6)
        assert list(keep) == [True, True]

    def test_more_than_the_fraction_is_removed(self):
        # 9 supporters, 4 past it: 4 > 0.34 x 9.
        keep, stats = _filter(_ring(9) + _past(4))
        assert list(keep) == [False, True], stats
        keep, _ = _filter(_ring(9) + _past(3))
        assert list(keep) == [True, True]

    def test_a_full_weight_face_is_judged_by_the_ratio_alone(self):
        # 9 supporters and 17 past it: 17 < 2 x 9 keeps the full-weight face.
        keep, _ = _filter(_ring(9) + _past(17))
        assert list(keep) == [False, True]


class TestTheParameters:
    def test_defaults_and_digest(self):
        p = S.SurfaceParams()
        assert p.low_weight_through_frac == pytest.approx(0.34)
        assert p.low_weight_through_min_support == 8
        d = p.digest_fields()
        assert d != S.SurfaceParams(low_weight_through_frac=0.0).digest_fields()
        assert d != S.SurfaceParams(low_weight_through_min_support=5).digest_fields()

    def test_the_live_preset_inherits_it(self):
        live = S.SurfaceParams.live()
        assert live.low_weight_through_frac == pytest.approx(0.34)


class TestTheManifestSaysSo:
    def test_the_closure_names_the_tolerance(self, tmp_path):
        from tests.test_world_builder_surface_low_weight import _far_wall_views
        from tests.test_world_builder_surface_pipeline import SESSION, WORLD, _params, _synthetic_world

        kw = dict(min_weight=2.0, min_support_frames=2, lod_face_targets=(0,), mobile_level=0)
        store = _synthetic_world(tmp_path / "w", views=_far_wall_views(spread=0.4, n=8), anchors=True)
        r = SP.surfacify(store, WORLD, SESSION, params=_params(**kw))
        assert r.state == SP.STATE_OK, r.detail
        man = SP.read_surface_manifest(store, WORLD, SESSION)
        assert "low_weight_through_frac" in man["closure"]
        store2 = _synthetic_world(tmp_path / "w0", views=_far_wall_views(spread=0.4, n=8), anchors=True)
        r = SP.surfacify(store2, WORLD, SESSION, params=_params(low_weight_through_frac=0.0, **kw))
        assert r.state == SP.STATE_OK, r.detail
        assert "low_weight_through_frac" not in SP.read_surface_manifest(store2, WORLD, SESSION)["closure"]
