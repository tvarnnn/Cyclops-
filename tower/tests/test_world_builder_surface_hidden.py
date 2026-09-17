"""Low-weight sheets no supporting camera could see (`hidden_low_weight`).

`Glasses-scratch/wb-final-recon/fixit/final/WALKTHROUGH.md` found that most of
the surface no kept keyframe image covers was LOW-WEIGHT sheets behind walls:
some frame's depth passed through the wall and landed on them, and since the
wall blocks every other view, nothing could ever see through them either.
`SurfaceParams.low_weight_hidden_test` removes a low-weight face that the kept
surface hides from every frame that supported it.

Hand-placed faces and ray-cast depth, so each case is isolated and the ground
truth holds no reconstruction.
"""

from __future__ import annotations

import numpy as np

from tests.test_world_builder_surface import ROOM
from tests.test_world_builder_surface_evidence import _views_tensors
from tests import test_world_builder_surface_low_weight as LW
from tests.test_world_builder_surface_pipeline import SESSION, WORLD, _params, _synthetic_world

from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP

TR = 0.06
SPREAD = ((0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3))


def _cams(offs, slab=None, z=-1.0):
    return [(np.array([dx, dy, z]), np.array([dx, dy, 5.0]), slab) for dx, dy in offs]


def _quad(z, x0, x1, y0, y1):
    """Two triangles at depth z facing the cameras (-z)."""
    V = [(x0, y0, z), (x0, y1, z), (x1, y0, z), (x1, y1, z)]
    F = [(0, 1, 2), (2, 1, 3)]
    n = np.cross(np.subtract(V[1], V[0]), np.subtract(V[2], V[0]))
    assert n[2] < 0
    return V, F


def _mesh(*parts):
    V, F = [], []
    for pv, pf in parts:
        F += [tuple(i + len(V) for i in f) for f in pf]
        V += list(pv)
    return np.asarray(V, np.float32), np.asarray(F, np.int64)


def _run(V, F, keep, weak, views):
    import torch

    tv, K = _views_tensors(views)
    return S.hidden_low_weight(V, F, np.asarray(keep, bool), np.asarray(weak, bool), tv, K,
                               lambda d: torch.full_like(d, TR), torch.device("cpu"))


# The sheet: a small quad on the far wall (faces 0, 1). The partition: a wide quad
# at z = 1 between every camera and the sheet (faces 2, 3). The frames' depth is
# the room only -- they measured the far wall THROUGH the partition, which is the
# overshoot that gives such a sheet its support.
SHEET = _quad(ROOM, -0.05, 0.05, -0.05, 0.05)
PARTITION = _quad(1.0, -1.5, 1.5, -1.5, 1.5)


class TestASheetTheKeptSurfaceHidesIsRemoved:
    def test_hidden_from_every_supporter_it_is_removed(self):
        V, F = _mesh(SHEET, PARTITION)
        drop, stats = _run(V, F, [1, 1, 1, 1], [1, 1, 0, 0], _cams(SPREAD))
        assert list(drop) == [True, True, False, False], stats
        assert stats["weak_tested"] == 2 and stats["dropped_weak_hidden"] == 2
        assert stats["hidden_rays"] == 2 * len(SPREAD)

    def test_without_the_partition_it_stays(self):
        V, F = _mesh(SHEET, PARTITION)
        drop, _ = _run(V, F, [1, 1, 0, 0], [1, 1, 0, 0], _cams(SPREAD))
        assert not drop.any()

    def test_a_full_weight_face_is_never_tested(self):
        V, F = _mesh(SHEET, PARTITION)
        drop, stats = _run(V, F, [1, 1, 1, 1], [0, 0, 0, 0], _cams(SPREAD))
        assert not drop.any() and stats["weak_tested"] == 0

    def test_one_supporter_that_can_see_it_keeps_it(self):
        # A partition that covers only the cameras on the -x side: the camera
        # at +0.3 x sees the sheet past its edge.
        part = _quad(1.0, -1.5, 0.05, -1.5, 1.5)
        V, F = _mesh(SHEET, part)
        drop, _ = _run(V, F, [1, 1, 1, 1], [1, 1, 0, 0], _cams(SPREAD))
        assert not drop[:2].any()
        # and with only the covered cameras, it goes
        drop2, _ = _run(V, F, [1, 1, 1, 1], [1, 1, 0, 0], _cams(((-0.3, 0.0), (0.0, 0.3))))
        assert drop2[:2].all()

    def test_a_frame_that_does_not_support_it_does_not_count(self):
        # Frames whose depth holds the partition (slab) measured the partition,
        # not the sheet: they neither support nor hide it.
        slab = ((-1.5, -1.5, 0.98), (1.5, 1.5, 1.02))
        V, F = _mesh(SHEET, PARTITION)
        drop, stats = _run(V, F, [1, 1, 1, 1], [1, 1, 0, 0], _cams(SPREAD, slab=slab))
        assert not drop.any() and stats["hidden_rays"] == 0

    def test_its_own_neighbours_do_not_hide_it(self):
        # The sheet inside a kept wall half a band nearer: within the band.
        wall = _quad(ROOM - 0.5 * TR, -1.5, 1.5, -1.5, 1.5)
        V, F = _mesh(SHEET, wall)
        drop, _ = _run(V, F, [1, 1, 1, 1], [1, 1, 0, 0], _cams(SPREAD))
        assert not drop.any()


class TestTheParameter:
    def test_on_by_default_and_in_the_digest(self):
        p = S.SurfaceParams()
        assert p.low_weight_hidden_test is True
        assert p.digest_fields() != S.SurfaceParams(low_weight_hidden_test=False).digest_fields()

    def test_the_live_preset_keeps_it(self):
        assert S.SurfaceParams.live().low_weight_hidden_test is True


class TestThroughTheBuild:
    """Through `surfacify`: the rule runs and is recorded, and a far wall the
    frames can see is not removed by it."""

    def test_the_visible_far_wall_survives_and_the_manifest_records_the_test(self, tmp_path):
        kw = dict(min_weight=2.0, min_support_frames=2, lod_face_targets=(0,), mobile_level=0)
        views = LW._far_wall_views(spread=0.4, n=8)
        off = _synthetic_world(tmp_path / "off", views=views, anchors=True)
        r = SP.surfacify(off, WORLD, SESSION, params=_params(low_weight_hidden_test=False, **kw))
        assert r.state == SP.STATE_OK, r.detail
        n_off, man_off = LW.TestThroughTheBuild()._wall_vertices(off)
        on = _synthetic_world(tmp_path / "on", views=views, anchors=True)
        r = SP.surfacify(on, WORLD, SESSION, params=_params(**kw))
        assert r.state == SP.STATE_OK, r.detail
        n_on, man = LW.TestThroughTheBuild()._wall_vertices(on)
        assert n_on > 50 and n_on >= 0.9 * n_off, (n_on, n_off)
        ev = man["detail"]["evidence_filter"]
        assert ev["weak_tested"] > 0 and "dropped_weak_hidden" in ev
        assert "dropped_weak_hidden" not in man_off["detail"]["evidence_filter"]
        assert "hid it from every frame" in man["closure"]

    def test_what_the_test_returns_is_removed_from_the_build(self, tmp_path, monkeypatch):
        # Stand-in verdict: every kept low-weight face hidden. The far wall is
        # low-weight only, so it must leave the published surface.
        seen = {}

        def all_hidden(V, F, keep, weak, views, K, trunc_at, device=None):
            drop = np.asarray(keep, bool) & np.asarray(weak, bool)
            seen["offered"] = int(drop.sum())
            return drop, {"weak_tested": int(drop.sum()), "dropped_weak_hidden": int(drop.sum()),
                          "hidden_rays": 0}

        monkeypatch.setattr(SP, "hidden_low_weight", all_hidden)
        kw = dict(min_weight=2.0, min_support_frames=2, lod_face_targets=(0,), mobile_level=0)
        store = _synthetic_world(tmp_path / "w", views=LW._far_wall_views(spread=0.4, n=8), anchors=True)
        r = SP.surfacify(store, WORLD, SESSION, params=_params(**kw))
        assert seen["offered"] > 0
        if r.state == SP.STATE_OK:
            n, man = LW.TestThroughTheBuild()._wall_vertices(store)
            assert n == 0, n
            ev = man["detail"]["evidence_filter"]
            assert ev["dropped_weak_hidden"] == seen["offered"] and ev["weak_kept"] == 0
        else:
            assert "hidden by kept surface" in (r.detail or ""), r.detail

    def test_the_pipeline_tests_the_surface_the_filter_kept(self):
        import inspect

        body = inspect.getsource(SP._build)
        i = body.index("evidence_filter(")
        j = body.index("hidden_low_weight(")
        k = body.index("keep_faces(V, F, C, keep)")
        assert i < j < k
