"""Per-vertex geometry confidence: the channel, its parts, and its lifecycle.

`Glasses-scratch/wb-final-recon/fixit/visual-review2/VISUAL-REVIEW-2.md`
established, with the page's own debug mode, that the two ugliest things in the
product -- "the ceiling one drag up from the opening looks like water damage
and peeling paint" and "a shredded cream-and-black column beside the door" --
are REAL observed imagery stretched over WRONG geometry, not missing data. The
page cannot tell good proxy from bad, so it paints both at full strength.

This is the number that tells it apart. `WORLD-BUILDER-SURFACE.md` claim 8 and
section 5a; `WORLD-BUILDER-APPEARANCE.md` section 4.2a. Its validation against
held-out keyframes is in `fixit/confidence/CONFIDENCE.md`; what is tested here
is that each component moves the score in the direction it claims to, that the
format refuses what it should, that the digest rebuilds, and that the manifest
and the appearance proxy carry it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_world_builder_surface import ROOM
from tests.test_world_builder_surface_low_weight import _far_wall_views
from tests.test_world_builder_surface_pipeline import (
    SESSION,
    WORLD,
    _params,
    _synthetic_world,
)

from tower.world_builder import appearance_pipeline as AP
from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP

MINW = 2.0


def _ev(**kw):
    """One face's evidence components, from counters given by name."""
    base = dict(sup=np.array([12.0]), thru=np.array([0.0]), wmin=np.array([8.0]),
                parallax=np.array([0.5]), resid_rms=np.array([0.1]))
    base.update({k: np.asarray([v], float) for k, v in kw.items()})
    return S.face_evidence_components(min_weight=MINW, **base)


def _score(**kw):
    return float(S._weighted_geomean(_ev(**kw), S.CONF_EVIDENCE_WEIGHTS)[0])


def _grid(nx=6, ny=6, z=0.0, jitter=None, stretch=1.0):
    """A flat (nx+1)x(ny+1) triangulated patch, optionally stretched in x or
    bent by a per-vertex z jitter."""
    xs = np.linspace(0, 1, nx + 1) * stretch
    ys = np.linspace(0, 1, ny + 1)
    V = np.array([[x, y, z] for y in ys for x in xs], np.float64)
    if jitter is not None:
        V[:, 2] += jitter
    F = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            F += [[a, a + 1, a + nx + 1], [a + 1, a + nx + 2, a + nx + 1]]
    return V.astype(np.float32), np.asarray(F, np.int64)


def _interior(V, F, margin=1e-6):
    """Vertices more than `CONF_RIM_HOPS` hops from the patch's boundary."""
    return np.asarray(S.mesh_geometry_components(V, F)["rim"]) >= 1.0 - margin


# ---------------------------------------------------------------------------
# the evidence half: each counter moves the score the way it claims to
# ---------------------------------------------------------------------------


class TestAWellMeasuredFaceOutscoresABadlyMeasuredOne:

    def test_the_well_measured_face_is_near_one_and_the_bad_one_near_zero(self):
        good = _score()
        bad = _score(sup=2.0, thru=2.0, wmin=0.3, parallax=0.01, resid_rms=0.9)
        assert good > 0.75, good
        assert bad < 0.15, bad

    def test_more_supporting_frames_score_higher(self):
        assert _score(sup=2.0) < _score(sup=10.0) < _score(sup=40.0)

    def test_the_support_component_is_not_already_saturated_at_a_handful(self):
        # The first constant tried saturated at 8 frames, and the component
        # then predicted nothing on held-out frames (AUC 0.499) because the
        # median kept face of the canonical capture has 13 supporters.
        assert _ev(sup=13.0)["support"][0] < 1.0
        assert S.CONF_SUPPORT_FULL >= 20.0

    def test_frames_that_saw_through_the_face_score_lower(self):
        assert _score(thru=0.0) > _score(thru=3.0) > _score(thru=11.0)

    def test_a_low_weight_face_scores_lower_than_a_full_weight_one(self):
        # `wmin` under `min_weight` is exactly "this came through the
        # low-weight exception"; there is no separate flag.
        assert _ev(wmin=0.5 * MINW)["weight"][0] < _ev(wmin=4 * MINW)["weight"][0]
        assert _score(wmin=0.5 * MINW) < _score(wmin=4 * MINW)

    def test_the_weight_component_is_not_already_saturated_at_min_weight(self):
        assert _ev(wmin=MINW)["weight"][0] < 1.0
        assert _ev(wmin=S.CONF_WEIGHT_FULL_MULTIPLE * MINW)["weight"][0] == pytest.approx(1.0)

    def test_frames_that_disagree_about_the_depth_score_lower(self):
        # The cross-frame disagreement local to the face, in units of the band.
        assert _score(resid_rms=0.02) > _score(resid_rms=0.3) > _score(resid_rms=0.58)

    def test_no_single_component_can_answer_zero_on_its_own(self):
        # Without the floor a geometric mean answers 0 for any face with (say)
        # a see-through frame for every supporter, however well everything else
        # was measured.
        assert _score(thru=99.0) > 0.0
        assert _score(resid_rms=9.0) > 0.0


class TestWhatWasMeasuredAndNotBelieved:
    """Three components the held-out test refused. They are still computed, so
    a later lane can re-measure them, and they change no score."""

    def test_parallax_and_correction_are_computed(self):
        c = S.face_evidence_components(
            sup=np.array([12.0]), thru=np.array([0.0]), wmin=np.array([8.0]),
            parallax=np.array([0.5]), resid_rms=np.array([0.1]),
            correction=np.array([0.02]), min_weight=MINW)
        assert "parallax" in c and "correction" in c

    def test_neither_is_weighted(self):
        assert "parallax" not in S.CONF_EVIDENCE_WEIGHTS
        assert "correction" not in S.CONF_EVIDENCE_WEIGHTS

    def test_a_face_with_no_parallax_at_all_scores_the_same(self):
        assert _score(parallax=0.0) == pytest.approx(_score(parallax=3.0))

    def test_a_heavily_corrected_face_scores_the_same(self):
        def sc(corr):
            c = S.face_evidence_components(
                sup=np.array([12.0]), thru=np.array([0.0]), wmin=np.array([8.0]),
                parallax=np.array([0.5]), resid_rms=np.array([0.1]),
                correction=np.array([corr]), min_weight=MINW)
            return float(S._weighted_geomean(c, S.CONF_EVIDENCE_WEIGHTS)[0])

        assert sc(0.0) == pytest.approx(sc(0.5))

    def test_the_geometric_mean_reads_the_weights_not_the_components(self):
        # The mechanism that makes "computed but not believed" possible.
        c = {"a": np.array([1.0]), "b": np.array([0.05])}
        assert S._weighted_geomean(c, {"a": 1.0})[0] == pytest.approx(1.0)
        assert S._weighted_geomean(c, {"a": 1.0, "b": 1.0})[0] < 0.3

    def test_weighting_nothing_is_no_opinion_and_not_a_verdict_of_zero(self):
        # A weights dict that names nothing present must not answer 0: that
        # would condemn a whole surface for a configuration mistake, and the
        # page's rule would fade all of it. It also must not raise.
        c = {"a": np.array([1.0, 0.5])}
        assert list(S._weighted_geomean(c, {})) == [1.0, 1.0]
        assert list(S._weighted_geomean(c, {"a": 0.0})) == [1.0, 1.0]
        assert list(S._weighted_geomean({}, {"a": 1.0})) == []


# ---------------------------------------------------------------------------
# the geometry half: this level's own triangles
# ---------------------------------------------------------------------------


class TestTheTrianglesGradeThemselves:

    def test_a_stretched_triangle_scores_below_an_equilateral_one(self):
        square = S.mesh_geometry_components(*_grid())["shape"]
        sliver = S.mesh_geometry_components(*_grid(stretch=40.0))["shape"]
        assert sliver.max() < square.min()
        assert sliver.max() < 0.5

    def test_a_vertex_on_the_boundary_scores_zero_and_the_interior_scores_one(self):
        V, F = _grid(nx=10, ny=10)
        rim = S.mesh_geometry_components(V, F)["rim"]
        on_edge = (V[:, 0] <= 1e-9) | (V[:, 0] >= 1 - 1e-9) | \
                  (V[:, 1] <= 1e-9) | (V[:, 1] >= 1 - 1e-9)
        assert rim[on_edge].max() == 0.0
        assert rim[~on_edge].max() == pytest.approx(1.0)
        assert rim.min() == 0.0 and rim.max() == pytest.approx(1.0)

    def test_the_rim_penalty_reaches_inward_by_the_documented_hops(self):
        V, F = _grid(nx=12, ny=12)
        rim = S.mesh_geometry_components(V, F)["rim"]
        assert sorted(set(np.round(rim, 6))) == [
            pytest.approx(h / S.CONF_RIM_HOPS) for h in range(S.CONF_RIM_HOPS + 1)]

    def test_a_crumpled_neighbourhood_scores_below_a_flat_one(self):
        flat = _grid(nx=10, ny=10)
        rng = np.random.default_rng(0)
        bent = _grid(nx=10, ny=10, jitter=rng.normal(0, 0.25, 121))
        fi, bi = _interior(*flat), _interior(*bent)
        fn = S.mesh_geometry_components(*flat)["normal"]
        bn = S.mesh_geometry_components(*bent)["normal"]
        assert fn[fi].min() > 0.99
        assert bn[bi].mean() < 0.9
        assert bn[bi].mean() < fn[fi].mean()

    def test_the_same_evidence_scores_lower_on_worse_triangles(self):
        ev = np.full(121, 0.9)
        flat = _grid(nx=10, ny=10)
        rng = np.random.default_rng(1)
        bent = _grid(nx=10, ny=10, jitter=rng.normal(0, 0.25, 121))
        cf = S.vertex_confidence(*flat, ev)
        cb = S.vertex_confidence(*bent, ev)
        fi = _interior(*flat)
        assert cb[fi].mean() < cf[fi].mean() - 5

    def test_an_empty_mesh_is_a_legal_input(self):
        V = np.zeros((0, 3), np.float32)
        F = np.zeros((0, 3), np.int64)
        assert len(S.vertex_confidence(V, F, np.zeros(0))) == 0
        assert len(S.face_to_vertex(0, V, F, np.zeros(0))) == 0


class TestTheWholeByte:

    def test_a_well_measured_interior_vertex_beats_a_badly_measured_rim_one(self):
        V, F = _grid(nx=10, ny=10)
        good = S.vertex_confidence(V, F, np.full(121, 0.95))
        bad = S.vertex_confidence(V, F, np.full(121, 0.05))
        i = _interior(V, F)
        assert good[i].min() > bad.max()
        assert good[i].min() > 150 and bad.max() < 100

    def test_it_is_a_byte_and_it_spans_the_range(self):
        V, F = _grid(nx=10, ny=10)
        c = S.vertex_confidence(V, F, np.linspace(0.0, 1.0, 121))
        assert c.dtype == np.uint8
        assert 0 <= c.min() and c.max() <= 255

    def test_a_decimated_level_inherits_the_evidence_by_nearest_vertex(self):
        V, F = _grid(nx=10, ny=10)
        ev = (V[:, 0] > 0.5).astype(np.float64)
        dst = np.array([[0.05, 0.5, 0.0], [0.95, 0.5, 0.0]], np.float32)
        assert list(S.transfer_vertex_values(V, ev, dst)) == [0.0, 1.0]

    def test_transfer_survives_an_empty_side(self):
        V, F = _grid(nx=2, ny=2)
        assert len(S.transfer_vertex_values(V, np.zeros(len(V)),
                                            np.zeros((0, 3), np.float32))) == 0


# ---------------------------------------------------------------------------
# the file format
# ---------------------------------------------------------------------------


class TestTheConfidenceFormat:

    def test_it_round_trips(self):
        conf = np.array([0, 1, 128, 254, 255], np.uint8)
        buf = S.write_confidence_bytes(conf)
        assert buf[:8] == S.CONF_MAGIC
        assert np.array_equal(S.read_confidence_bytes(buf), conf)

    def test_it_is_one_byte_a_vertex_and_a_twenty_byte_header(self):
        for n in (0, 1, 1000):
            assert len(S.write_confidence_bytes(np.zeros(n, np.uint8))) == 20 + n

    def test_an_empty_channel_round_trips(self):
        assert len(S.read_confidence_bytes(S.write_confidence_bytes(np.zeros(0, np.uint8)))) == 0

    def test_a_wrong_magic_is_refused(self):
        buf = bytearray(S.write_confidence_bytes(np.zeros(4, np.uint8)))
        buf[:8] = b"NOTACONF"
        with pytest.raises(S.SurfaceUnavailable, match="magic"):
            S.read_confidence_bytes(bytes(buf))

    def test_an_unknown_version_is_refused(self):
        import struct
        buf = bytearray(S.write_confidence_bytes(np.zeros(4, np.uint8)))
        struct.pack_into("<I", buf, 12, S.CONFIDENCE_VERSION + 1)
        with pytest.raises(S.SurfaceUnavailable, match="schema"):
            S.read_confidence_bytes(bytes(buf))

    def test_a_torn_file_is_refused(self):
        buf = S.write_confidence_bytes(np.zeros(40, np.uint8))
        with pytest.raises(S.SurfaceUnavailable, match="header implies"):
            S.read_confidence_bytes(buf[:-3])
        with pytest.raises(S.SurfaceUnavailable, match="shorter than its header"):
            S.read_confidence_bytes(buf[:10])

    def test_a_channel_for_another_level_is_refused(self):
        buf = S.write_confidence_bytes(np.zeros(40, np.uint8))
        assert len(S.read_confidence_bytes(buf, 40)) == 40
        with pytest.raises(S.SurfaceUnavailable, match="41"):
            S.read_confidence_bytes(buf, 41)

    def test_the_mesh_format_is_untouched(self):
        # The reason the channel is a sidecar: every reader of
        # `wb-surface-mesh/1` refuses a buffer that is not exactly the length
        # its header implies, so it could not have gone inside.
        V, F = _grid(nx=3, ny=3)
        buf = S.write_mesh_bytes(V, F, np.zeros((len(V), 3), np.uint8))
        with pytest.raises(S.SurfaceUnavailable, match="header implies"):
            S.read_mesh_bytes(buf + S.write_confidence_bytes(
                np.zeros(len(V), np.uint8)))


class TestTheParamsAndTheDigest:

    def test_it_is_on_by_default_and_in_the_digest(self):
        p = S.SurfaceParams()
        assert p.confidence is True
        assert p.digest_fields() != S.SurfaceParams(confidence=False).digest_fields()

    def test_the_digest_carries_the_version_so_a_new_definition_rebuilds(self):
        assert any(isinstance(f, tuple) and f and f[0] == "confidence"
                   and S.CONFIDENCE_VERSION in f
                   for f in S.SurfaceParams().digest_fields())

    def test_the_live_preset_publishes_it_too(self):
        assert S.SurfaceParams.live().confidence is True


# ---------------------------------------------------------------------------
# through the build
# ---------------------------------------------------------------------------


class TestThroughTheBuild:

    @staticmethod
    def _build(tmp_path, name, **kw):
        store = _synthetic_world(tmp_path / name, views=_far_wall_views(spread=0.4, n=8),
                                 anchors=True)
        r = SP.surfacify(store, WORLD, SESSION,
                         params=_params(min_weight=2.0, min_support_frames=2,
                                        lod_face_targets=(0, 400), mobile_level=1, **kw))
        assert r.state == SP.STATE_OK, r.detail
        return store, SP.read_surface_manifest(store, WORLD, SESSION)

    def test_every_level_publishes_a_channel_the_manifest_names(self, tmp_path):
        store, man = self._build(tmp_path, "on")
        assert man["confidence"]["format"] == S.CONFIDENCE_FORMAT
        assert man["confidence"]["version"] == S.CONFIDENCE_VERSION
        assert man["confidence"]["levels"] == len(man["levels"])
        root = SP.surface_dir(store, WORLD, SESSION)
        for lv in man["levels"]:
            rec = lv["confidence"]
            assert rec["vertices"] == lv["vertices"]
            path = SP.confidence_file(root, lv)
            assert path is not None and path.stat().st_size == rec["bytes"]
            conf = SP.read_surface_confidence(store, WORLD, SESSION, lv["level"],
                                              manifest=man)
            assert len(conf) == lv["vertices"]
            assert conf.dtype == np.uint8

    def test_turning_it_off_publishes_the_same_mesh_and_no_channel(self, tmp_path):
        on_store, on = self._build(tmp_path, "on2")
        off_store, off = self._build(tmp_path, "off2", confidence=False)
        assert off["confidence"] is None
        assert all("confidence" not in lv for lv in off["levels"])
        # the levels themselves are byte-identical
        for lv in on["levels"]:
            a = SP.read_surface_level(on_store, WORLD, SESSION, lv["level"], manifest=on)
            b = SP.read_surface_level(off_store, WORLD, SESSION, lv["level"], manifest=off)
            assert a == b
        with pytest.raises(S.SurfaceUnavailable, match="no confidence channel"):
            SP.read_surface_confidence(off_store, WORLD, SESSION, 0, manifest=off)
        assert not list(SP.surface_dir(off_store, WORLD, SESSION).glob("conf_l*.bin"))

    def test_the_same_wall_scores_lower_when_only_the_exception_admitted_it(self, tmp_path):
        # The SAME far wall, from the same eight cameras, twice: once behind a
        # near panel that takes over each frame's median depth, so the wall
        # never reaches `min_weight` and only the low-weight exception admits
        # it, and once with nothing in front of it. The geometry is the same
        # wall in both; only the evidence differs, which is the point.
        def wall_confidence(name, views):
            store = _synthetic_world(tmp_path / name, views=views, anchors=True)
            r = SP.surfacify(store, WORLD, SESSION,
                             params=_params(min_weight=2.0, min_support_frames=2,
                                            lod_face_targets=(0,), mobile_level=0))
            assert r.state == SP.STATE_OK, r.detail
            man = SP.read_surface_manifest(store, WORLD, SESSION)
            V, _F, _C, _N = S.read_mesh_bytes(
                SP.read_surface_level(store, WORLD, SESSION, 0, manifest=man))
            conf = np.asarray(SP.read_surface_confidence(store, WORLD, SESSION, 0,
                                                         manifest=man), float)
            on_wall = np.abs(V[:, 2] - ROOM) < 0.15
            assert on_wall.sum() > 100, on_wall.sum()
            return conf[on_wall].mean(), man

        weak_score, weak_man = wall_confidence("weak", _far_wall_views(spread=0.4, n=8))
        clear = [(eye, target, None) for eye, target, _slab
                 in _far_wall_views(spread=0.4, n=8)]
        full_score, _ = wall_confidence("full", clear)
        assert weak_man["detail"]["evidence_filter"]["weak_kept"] > 100
        assert weak_score < full_score - 5, (weak_score, full_score)

    def test_a_rebuild_with_the_flag_flipped_is_not_already_built(self, tmp_path):
        store = _synthetic_world(tmp_path / "again", views=_far_wall_views(n=8),
                                 anchors=True)
        kw = dict(min_weight=2.0, min_support_frames=2, lod_face_targets=(0,),
                  mobile_level=0)
        assert SP.surfacify(store, WORLD, SESSION,
                            params=_params(**kw)).state == SP.STATE_OK
        again = SP.surfacify(store, WORLD, SESSION, params=_params(**kw))
        assert again.detail == SP.ALREADY_BUILT
        flipped = SP.surfacify(store, WORLD, SESSION,
                               params=_params(confidence=False, **kw))
        assert flipped.state == SP.STATE_OK and flipped.detail != SP.ALREADY_BUILT

    def test_a_superseded_channel_is_pruned_with_its_level(self, tmp_path):
        store, man = self._build(tmp_path, "prune")
        root = SP.surface_dir(store, WORLD, SESSION)
        old = {SP.confidence_file(root, lv).name for lv in man["levels"]}
        assert old
        r = SP.surfacify(store, WORLD, SESSION,
                         params=_params(min_weight=2.0, min_support_frames=2,
                                        lod_face_targets=(0, 400), mobile_level=1,
                                        smooth_iterations=3),
                         force=True)
        assert r.state == SP.STATE_OK, r.detail
        SP._prune_superseded_levels(
            root, {SP.level_file(root, lv).name
                   for lv in SP.read_surface_manifest(store, WORLD, SESSION)["levels"]}
            | {SP.confidence_file(root, lv).name
               for lv in SP.read_surface_manifest(store, WORLD, SESSION)["levels"]},
            older_than_s=0.0)
        left = {p.name for p in root.glob("conf_l*.bin")}
        assert not (old & left), (old, left)

    def test_publishing_does_not_prune_its_own_confidence_sidecar(self, tmp_path):
        # The first real build of the canonical capture lost `conf_l0` this
        # way. Level 0's sidecar is written when level 0 is packed, which on a
        # 3.2M-face mesh is FIVE MINUTES before the manifest; the prune grace
        # is two. The publish then globbed `conf_l*.bin`, found level 0's not
        # in its keep set -- which held only `entry["file"]` -- and deleted the
        # file the manifest it had just written names.
        #
        # A test build takes a second, so the ages have to be made here; what
        # is under test is `published_names`, the keep set the publish path
        # itself now builds.
        import os
        import time

        store, man = self._build(tmp_path, "selfprune")
        root = SP.surface_dir(store, WORLD, SESSION)
        old = time.time() - 10 * SP.PRUNE_GRACE_S
        for path in root.glob("*.bin"):
            os.utime(path, (old, old))
        SP._prune_superseded_levels(root, SP.published_names(man["levels"]))
        for lv in man["levels"]:
            assert SP.level_file(root, lv).exists(), lv["level"]
            assert SP.confidence_file(root, lv).exists(), lv["level"]
            assert SP.read_surface_confidence(store, WORLD, SESSION, lv["level"],
                                              manifest=man) is not None

    def test_after_a_publish_every_file_in_the_directory_is_named(self, tmp_path):
        # The invariant the bug above broke, from the other side: nothing the
        # manifest names is missing, and nothing is left over.
        store, man = self._build(tmp_path, "named")
        root = SP.surface_dir(store, WORLD, SESSION)
        assert {p.name for p in root.glob("*.bin")} == SP.published_names(man["levels"])

    def test_the_keep_set_names_the_sidecar_beside_the_level(self):
        assert SP.published_names([
            {"level": 0, "file": "mesh_l0.b.bin",
             "confidence": {"file": "conf_l0.b.bin"}},
            {"level": 1, "file": "mesh_l1.b.bin"},
        ]) == {"mesh_l0.b.bin", "conf_l0.b.bin", "mesh_l1.b.bin"}
        assert SP.published_names(None) == set()
        assert SP.published_names([None, 7, {"confidence": {"file": 3}}]) == set()

    def test_a_manifest_path_is_never_followed_out_of_the_directory(self, tmp_path):
        root = tmp_path
        for name in ("../escape.bin", "a/b.bin", ".hidden.bin", "x.txt", 7, None):
            assert SP.confidence_file(root, {"level": 0,
                                             "confidence": {"file": name}}) is None
        assert SP.confidence_file(root, {"level": 0}) is None


# ---------------------------------------------------------------------------
# the appearance proxy
# ---------------------------------------------------------------------------


class TestTheProxyCarriesIt:

    @staticmethod
    def _mesh(n=40):
        V, F = _grid(nx=5, ny=5)
        return S.write_mesh_bytes(V, F, np.full((len(V), 3), 200, np.uint8)), len(V)

    def test_the_confidence_lands_in_red_and_green_and_blue_stay_zero(self):
        mesh, n = self._mesh()
        conf = np.arange(n, dtype=np.uint8)
        proxy = AP.proxy_with_confidence(mesh, conf)
        assert len(proxy) == len(mesh)
        _V, _F, C, _N = S.read_mesh_bytes(proxy)
        assert np.array_equal(C[:, 0], conf)
        assert not C[:, 1].any() and not C[:, 2].any()

    def test_positions_and_indices_are_untouched(self):
        mesh, n = self._mesh()
        proxy = AP.proxy_with_confidence(mesh, np.full(n, 7, np.uint8))
        sV, sF, _sC, sN = S.read_mesh_bytes(mesh)
        pV, pF, _pC, pN = S.read_mesh_bytes(proxy)
        assert np.array_equal(sV, pV) and np.array_equal(sF, pF)
        assert (sN is None) == (pN is None)

    def test_no_channel_zeroes_every_colour_byte_as_before(self):
        mesh, _n = self._mesh()
        proxy = AP.proxy_with_confidence(mesh, None)
        _V, _F, C, _N = S.read_mesh_bytes(proxy)
        assert not C.any()
        assert len(proxy) == len(mesh)

    def test_a_channel_of_the_wrong_length_is_refused(self):
        mesh, n = self._mesh()
        with pytest.raises(AP.A.AppearanceUnavailable, match="vertices"):
            AP.proxy_with_confidence(mesh, np.zeros(n + 1, np.uint8))

    def test_something_that_is_not_a_surface_mesh_is_refused(self):
        with pytest.raises(AP.A.AppearanceUnavailable, match="not a surface mesh"):
            AP.proxy_with_confidence(b"NOTAMESH" + bytes(80), None)

    def test_the_channel_it_declares_is_the_one_it_writes(self):
        assert AP.PROXY_CONFIDENCE_CHANNEL == "r"
        mesh, n = self._mesh()
        _V, _F, C, _N = S.read_mesh_bytes(
            AP.proxy_with_confidence(mesh, np.full(n, 99, np.uint8)))
        assert set(np.unique(C[:, 0])) == {99}


class TestThroughTheAppearanceBuild:
    """The phone's copy: the channel rides the proxy's already-zero colour
    bytes, at no cost in bytes and with no reader changed
    (`WORLD-BUILDER-APPEARANCE.md` section 4.2a)."""

    @staticmethod
    def _built(root, confidence=None):
        from tests.test_world_builder_appearance import World, _never_redact

        w = World(root)
        w.write_surface(confidence=confidence)
        r = w.build(redactor_factory=_never_redact)
        assert r.state == AP.STATE_OK, r.detail
        return w, w.manifest()

    def _proxy(self, w, man):
        return AP.read_appearance_file(w.store, "w", "s", "proxy",
                                       man["proxy"]["digest"], man)

    def test_the_proxy_carries_the_surfaces_own_bytes(self, tmp_path):
        from tests.test_world_builder_appearance import SESSION, WORLD

        w = self._built(tmp_path / "with", confidence=None)[0]
        n = len(S.read_mesh_bytes(
            SP.read_surface_level(w.store, WORLD, SESSION, 0))[0])
        conf = (np.arange(n) % 256).astype(np.uint8)

        w2, man = self._built(tmp_path / "conf", confidence=conf)
        proxy = AP.read_appearance_file(w2.store, WORLD, SESSION, "proxy",
                                        man["proxy"]["digest"], man)
        _V, _F, C, _N = S.read_mesh_bytes(proxy)
        assert np.array_equal(C[:, 0], conf)
        assert not C[:, 1].any() and not C[:, 2].any()
        assert man["proxy"]["confidence"]["present"] is True
        assert man["proxy"]["confidence"]["channel"] == "r"
        assert man["proxy"]["confidence"]["source_level"] == 0
        # no extra bytes on the wire
        assert man["proxy"]["bytes"] == len(
            SP.read_surface_level(w2.store, WORLD, SESSION, 0))

    def test_a_surface_without_a_channel_publishes_zeros_and_says_so(self, tmp_path):
        from tests.test_world_builder_appearance import SESSION, WORLD

        w, man = self._built(tmp_path / "none", confidence=None)
        proxy = AP.read_appearance_file(w.store, WORLD, SESSION, "proxy",
                                        man["proxy"]["digest"], man)
        _V, _F, C, _N = S.read_mesh_bytes(proxy)
        assert not C.any()
        assert man["proxy"]["confidence"]["present"] is False

    def test_the_channel_changes_the_proxy_digest(self, tmp_path):
        from tests.test_world_builder_appearance import SESSION, WORLD

        w = self._built(tmp_path / "a")[0]
        n = len(S.read_mesh_bytes(
            SP.read_surface_level(w.store, WORLD, SESSION, 0))[0])
        _wa, a = self._built(tmp_path / "b", confidence=np.zeros(n, np.uint8))
        _wb, b = self._built(tmp_path / "c", confidence=np.full(n, 200, np.uint8))
        assert a["proxy"]["digest"] != b["proxy"]["digest"]
        assert a["proxy"]["bytes"] == b["proxy"]["bytes"]
