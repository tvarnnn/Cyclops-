"""A solve with a handful of wildly-placed poses must still build its room.

WHY THIS FILE EXISTS

`global_solve` is not deterministic. The same 385-keyframe bedroom walk
(capture `a808611c…`) was replayed twice through the live builder path and
solved twice, to ~15,800 points and 379 images both times, one component both
times, 0.89 px mean reprojection error both times. The robust cores agreed:
p5-p95 extents of [7.7, 12.9, 10.8] and [4.6, 7.5, 3.7] units, the same
bedroom. One of the two ALSO placed a few hundred points and nine camera
centres about 10^6 units away.

The surface stage refused that solve outright::

    state: "unavailable", frames_used: 0, vertices: 0, faces: 0,
    detail: "the scene's block coordinates fall outside the keyable range;
             the SfM gauge is arbitrary and this world's is too large for the
             chosen voxel fraction"

and because the surface was not `ok` the appearance stage never ran, so the
wearer of a perfectly good bedroom walk got sparse points and nothing else --
on a coin flip. `block_key`'s raise is correct (outside the keyable range two
cells share a key and the field silently merges opposite ends of a scene); the
bug is that a stage whose own docstring says the gauge is arbitrary had no
answer to an arbitrary gauge but refusal.

The tests below are about the answer: drop the poses that are radius outliers
among the session's own poses, fuse inside a robust envelope of the observed
scene, coarsen the voxel if the grid still cannot be keyed, and record all
three. Every criterion here is RELATIVE -- a multiple of the solve's own
median -- because an absolute threshold would be the same mistake in a new
place.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tests.test_world_builder_surface import _camera, _look_from
from tests.test_world_builder_surface_pipeline import (
    SESSION,
    WORLD,
    _manifest,
    _params,
    _synthetic_world,
)

from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP


# ---------------------------------------------------------------------------
# the fixture: a clean eight-camera room, plus frames the solver misplaced
# ---------------------------------------------------------------------------


def _ring_views(n: int, radius: float = 1.2):
    """`n` cameras on a ring at the centre of the box room, looking outward."""
    views = []
    for i in range(n):
        a = 2 * np.pi * i / n
        eye = np.array([radius * np.cos(a), 0.0, radius * np.sin(a)])
        views.append((eye, eye * 3.0, None))
    return views


RING = 24
"""Frames in the healthy body of the wild-pose fixture.

Twenty-four, not eight, and the reason is the bug review 2 found. The gate
now measures detachment against the 95th-percentile pose radius, so ONE
outlier among eight poses is an eighth of the sample and lands inside p95 --
a fixture built that way would be a fixture of a solve that came apart, not
of run A, which was 4 poses in 271 (1.48%). One wild pose in twenty-five is
4%, the same order as the real thing.
"""


def _wild_world(tmp_path, *, ring: int = RING, n_wild: int = 1, **kw):
    """The box room with a realistic handful of solver noise grafted on."""
    store = _synthetic_world(tmp_path, views=_ring_views(ring))
    _graft_wild_frames(store, n=n_wild, **kw)
    return store


def _graft_wild_frames(store, *, n: int = 3, radius: float = 5.0e4,
                       depth: float = 1.2e6) -> list[int]:
    """Add `n` frames whose pose and depth are solver noise, as run A's were.

    Faithful to the measured failure rather than merely large: the camera
    CENTRE is a radius outlier (run A's worst was 53,026 units from a
    thirteen-unit room), and the frame's own far bound -- `z_sparse_max`,
    which is what `depth_bound` clips against -- is large enough to admit the
    depth, exactly as run A's 1.6e6 anchor was. A frame with an absurd pose
    but a sane far bound would be clipped by machinery that already exists
    and would test nothing.
    """
    import cv2

    from tower.world_builder.dense_pipeline import FILL_RULE
    from tower.world_builder.global_solve import workspace_for

    K, w, h = _camera()
    workspace = workspace_for(store, WORLD, SESSION)
    meta = json.loads(workspace.solution_path.read_text())
    dense = store.world_dir(WORLD) / "dense" / SESSION
    align = json.loads((dense / "align.json").read_text())
    images = store.images_dir(WORLD, SESSION)
    images.mkdir(parents=True, exist_ok=True)

    base_ki = max(r["ki"] for r in align["records"]) + 1
    grafted = []
    for j in range(n):
        ki = base_ki + j
        kid = f"{SESSION}:{900 + j:08d}"
        a = 2 * np.pi * j / max(n, 1)
        eye = np.array([np.cos(a), 0.3, np.sin(a)]) * radius
        R, t = _look_from(eye, np.zeros(3))
        meta["poses"][kid] = {"component": 0, "rotation": R.ravel().tolist(),
                              "translation": t.tolist(), "observations": 50}
        np.save(dense / "work" / "depth" / f"{ki:05d}.npy",
                np.full((h, w), depth, np.float32))
        np.save(dense / "work" / "depth" / f"{ki:05d}_fill.npy",
                np.zeros((h, w), bool))
        img = np.full((h, w, 3), 90, np.uint8)
        cv2.imwrite(str(dense / "work" / "undist" / f"{ki:05d}.jpg"), img)
        _ok, enc = cv2.imencode(".jpg", img)
        (images / f"{kid.rsplit(':', 1)[-1]}.jpg").write_bytes(enc.tobytes())
        align["records"].append({
            "ki": ki, "kid": kid, "ok": True, "a": 1.0, "b": 0.0,
            "held_out_rel": 0.01, "fill_rule": FILL_RULE,
            "image_sha1": hashlib.sha1(enc.tobytes()).hexdigest(),
            "z_sparse_min": depth * 0.5, "z_sparse_max": depth * 1.2,
        })
        grafted.append(ki)
    workspace.solution_path.write_text(json.dumps(meta))
    (dense / "align.json").write_text(json.dumps(align))
    return grafted


def _outliers(man) -> dict:
    assert man is not None
    rec = man.get("outliers")
    assert isinstance(rec, dict), man.keys()
    return rec


def _record(store, result) -> dict:
    """What the guards did, from whichever place this build wrote it.

    A published build puts it in the manifest; a refused or stopped one has
    only `status.json` and the returned result. All three must carry it --
    that is the whole of review 2's M3 -- so the helper accepts any."""
    rec = result.outliers
    assert isinstance(rec, dict) and rec, f"no outlier record on a {result.state}"
    status = json.loads((SP.surface_dir(store, WORLD, SESSION)
                         / "status.json").read_text())
    on_disk = status.get("outliers") or (status.get("result") or {}).get("outliers")
    assert on_disk == rec, "the record on disk differs from the one returned"
    return rec


# ---------------------------------------------------------------------------
# the defect
# ---------------------------------------------------------------------------


class TestASolveWithWildPosesStillBuildsItsRoom:

    def test_the_room_is_reconstructed_rather_than_refused(self, tmp_path):
        """The headline. Before the robust gate this returned `unavailable`
        with `faces: 0`, and the appearance stage never ran."""
        store = _wild_world(tmp_path)

        result = SP.surfacify(store, WORLD, SESSION, params=_params())

        assert result.state == SP.STATE_OK, result.detail
        assert result.faces > 0 and result.vertices > 0
        assert result.frames_used >= RING - 2

    def test_the_gated_poses_are_counted_and_named_in_the_manifest(self, tmp_path):
        """A surface built from fewer frames than the solve offered must say
        so. A silent drop is how a stage starts lying about its coverage."""
        store = _wild_world(tmp_path, n_wild=1)

        SP.surfacify(store, WORLD, SESSION, params=_params())

        rec = _outliers(_manifest(store))
        assert rec["poses_gated"] == 1
        assert rec["poses_offered"] == RING + 1
        assert rec["pose_radius_multiple"] == _params().outlier_radius_multiple
        # The criterion is a multiple of the solve's own median radius and of
        # its p95, and the record has to let a reader recompute both.
        assert rec["pose_radius_median"] > 0
        assert rec["pose_radius_p95"] > 0
        assert rec["pose_radius_gated_max"] > rec["pose_gate_threshold"]
        assert not rec["pose_gate_stood_down"]

    def test_the_bound_catches_what_the_gate_declines_to_judge(self, tmp_path):
        """DEFENCE IN DEPTH, and it is not theoretical. Three wild frames
        among eight is 27% of the session -- far too large a share for the
        gate to call a correction, so it stands down (see
        `SurfaceParams.max_gated_pose_fraction`) and does not drop them. The
        room is still reconstructed, because the fusion bound then finds that
        every pixel those frames offer lies outside the observed scene."""
        store = _synthetic_world(tmp_path, views=_ring_views(8))
        _graft_wild_frames(store, n=3)

        result = SP.surfacify(store, WORLD, SESSION, params=_params())

        assert result.state == SP.STATE_OK, result.detail
        assert result.faces > 0
        rec = _record(store, result)
        assert rec["poses_gated"] == 0
        assert rec["frames_emptied_by_bound"] == 3

    def test_the_fusion_volume_is_bounded_by_the_observed_scene(self, tmp_path):
        """Not by the full bounding box. Run A's full point extent was
        1,219,893 units around a thirteen-unit bedroom."""
        store = _wild_world(tmp_path)

        SP.surfacify(store, WORLD, SESSION, params=_params())

        rec = _outliers(_manifest(store))
        lo = np.asarray(rec["fusion_bound_lo"], float)
        hi = np.asarray(rec["fusion_bound_hi"], float)
        # the box room is six units across; the bound must hold it and
        # nothing like the 5e4 the grafted cameras sit at
        assert np.all(hi - lo >= 6.0)
        assert np.all(hi - lo < 1.0e3)


class TestACleanSolveIsUntouched:
    """The regression that matters. Nine of the eleven solved sessions in the
    store have no outlier at all, and their surfaces must not move."""

    def test_nothing_is_gated_and_nothing_is_bounded_away(self, tmp_path):
        store = _synthetic_world(tmp_path)
        result = SP.surfacify(store, WORLD, SESSION, params=_params())
        assert result.state == SP.STATE_OK, result.detail

        rec = _outliers(_manifest(store))
        assert rec["poses_gated"] == 0
        assert rec["frames_emptied_by_bound"] == 0
        # The clip tripwire has to be asserted too. Review 2 found it written
        # and checked nowhere, which is the same as not having it: on a
        # healthy world the bound must touch nothing, and a single clipped
        # PIXEL is the first sign that it has started eating the room.
        assert rec["frames_clipped_by_bound"] == 0
        assert rec["pixels_clipped_by_bound"] == 0

    def test_the_mesh_is_the_same_as_with_the_gate_switched_off(self, tmp_path):
        """Byte-for-byte on a clean world, because neither rule fires."""
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        guarded = SP.read_surface_level(store, WORLD, SESSION, 0)

        SP.surfacify(store, WORLD, SESSION,
                     params=_params(outlier_radius_multiple=0.0,
                                    fusion_bound_multiple=0.0))
        plain = SP.read_surface_level(store, WORLD, SESSION, 0)

        assert guarded == plain


# ---------------------------------------------------------------------------
# the statistics themselves
# ---------------------------------------------------------------------------


def _ring(n=12, radius=1.5):
    a = 2 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(a), np.zeros(n), radius * np.sin(a)], 1)


class TestTheRobustStatisticsAreGaugeFree:
    """`global_solve` never normalises: the same room has solved to a ten-unit
    extent and a three-hundred-unit one. A threshold in units would be the
    bug wearing a different hat."""

    def test_a_thousandfold_change_of_gauge_gates_the_same_poses(self):
        # forty in the body, one outlier: 2.4%, the order run A's was. One in
        # thirteen would sit inside its own p95 and be judged part of the body.
        centres = np.concatenate([_ring(40), [[900.0, 0.0, 0.0]]])
        near = S.robust_pose_outliers(centres)
        far = S.robust_pose_outliers(centres * 1000.0)
        assert np.array_equal(near.outlier, far.outlier)
        assert near.outlier.sum() == 1 and near.outlier[-1]

    def test_a_clean_ring_gates_nothing(self):
        assert S.robust_pose_outliers(_ring()).outlier.sum() == 0

    def test_a_walk_is_not_mistaken_for_an_outlier(self):
        """A straight corridor spreads the centres along one axis; every one
        of them is a legitimate pose and none may be dropped.

        THIS TEST WAS NOT ENOUGH, and review 2 was right about why: it walks
        at a CONSTANT pace, so the median radius grows with the corridor and
        the multiple-of-the-median never bites. Real captures dwell. See
        `TestADwellingCaptureKeepsEveryPose` below, which is the test that
        would have caught it.
        """
        walk = np.stack([np.linspace(0, 40, 60), np.zeros(60), np.zeros(60)], 1)
        assert S.robust_pose_outliers(walk).outlier.sum() == 0

    def test_too_few_poses_to_be_robust_gates_nothing(self):
        few = np.concatenate([_ring(4), [[9e5, 0.0, 0.0]]])
        report = S.robust_pose_outliers(few)
        assert report.outlier.sum() == 0
        assert "too few" in report.detail

    def test_the_rule_can_be_switched_off(self):
        centres = np.concatenate([_ring(40), [[9e5, 0.0, 0.0]]])
        assert S.robust_pose_outliers(centres, multiple=0.0).outlier.sum() == 0

    def test_the_detachment_test_can_be_switched_off(self):
        """And with it off the old behaviour returns, which is what makes the
        fraction cap worth keeping."""
        walk = _dwell_then_walk(0.80, 40)
        assert S.robust_pose_outliers(walk, detach=0.0, max_fraction=0.0).gated > 0

    def test_the_bound_holds_every_kept_camera_and_its_reach(self):
        """The envelope is not the point cloud's: a frame may measure farther
        than any sparse point it was fitted to, and clipping that would be
        deleting measured surface rather than solver noise."""
        centres = _ring(12, 2.0)
        points = np.concatenate([_ring(24, 3.0), [[0.0, 1.0e6, 0.0]]])
        bound = S.robust_fusion_bound(points, centres, reach=5.0)
        assert bound is not None
        assert np.all(bound.lo <= centres.min(0) - 5.0)
        assert np.all(bound.hi >= centres.max(0) + 5.0)
        # and the 10^6 point is nowhere near inside it
        assert bound.hi[1] < 1.0e5

    def test_the_bound_can_be_switched_off(self):
        assert S.robust_fusion_bound(_ring(24, 3.0), _ring(), reach=1.0,
                                     bound_multiple=0.0) is None


# ---------------------------------------------------------------------------
# a capture that dwells keeps every pose
# ---------------------------------------------------------------------------


def _blob(centre, spread, n, seed):
    r = np.random.default_rng(seed)
    return np.asarray(centre, float) + r.normal(0, spread, (n, 3))


def _paced_walk(length, n, seed, spread=0.3):
    r = np.random.default_rng(seed)
    x = np.linspace(0, length, n)
    return np.stack([x, r.normal(0, spread, n), r.normal(0, spread, n)], 1)


def _uniform_corridor(length, n=360, seed=0):
    return _paced_walk(length, n, seed, spread=length * 0.002)


def _l_shaped_apartment(n=360, seed=1):
    """Three rooms in an L, with the wearer dwelling 60/25/15."""
    return np.concatenate([_blob([0, 0, 0], 0.8, int(n * 0.60), seed),
                           _blob([8, 0, 0], 0.8, int(n * 0.25), seed + 1),
                           _blob([8, 0, 7], 0.8, int(n * 0.15), seed + 2)])


def _dwell_then_walk(dwell_frac, length, n=360, seed=2):
    k = int(round(n * dwell_frac))
    return np.concatenate([_blob([0, 0, 0], 0.5, k, seed),
                           _paced_walk(length, n - k, seed + 1)])


def _two_rooms(frac_b, distance, n=360, seed=4):
    k = int(round(n * frac_b))
    return np.concatenate([_blob([0, 0, 0], 1.0, n - k, seed),
                           _blob([distance, 0, 0], 1.0, k, seed + 1)])


DWELLING_CAPTURES = [
    ("uniform corridor, 10 units", _uniform_corridor(10)),
    ("uniform corridor, 50 units", _uniform_corridor(50)),
    ("uniform corridor, 200 units", _uniform_corridor(200)),
    ("uniform corridor, 1000 units", _uniform_corridor(1000)),
    ("L-shaped apartment, dwell 60/25/15", _l_shaped_apartment()),
    ("80% dwelling, then a 40-unit walk", _dwell_then_walk(0.80, 40)),
    ("70% in the start room, transit to 40", _dwell_then_walk(0.70, 40, seed=3)),
    ("two rooms, 20% of frames in B at 30", _two_rooms(0.20, 30)),
    ("two rooms, 30% of frames in B at 30", _two_rooms(0.30, 30)),
    ("two halves 55/45 at 100", _two_rooms(0.45, 100)),
]


class TestADwellingCaptureKeepsEveryPose:
    """THE BUG REVIEW 2 FOUND, and the reason the gate is not a multiple of
    the median alone.

    A multiple of the median radius is a statement about SCALE. It says
    nothing about whether a pose is attached to the rest of the walk, and a
    capture that dwells has a tiny median radius with a perfectly legitimate
    tail: stand in one spot for most of a walk, then cross the room, and the
    median collapses onto the spot while every pose of the crossing sits ten
    medians out. Measured on the old rule at n=360: 15.6% of an "80% dwell
    then walk" gated, 20.6% of a one-way transit, 20.0% and 30.0% of a
    two-room capture (the WHOLE of room B both times), 45.0% of a 55/45
    split. The physical test asks the wearer to walk, stand back and revisit,
    so this was not a corner case; it was the shape of the test.

    Worse, the gate runs in `_Frames.__init__`, so a gated pose is dropped
    from depth, consistency, transients, fusion AND the appearance -- it
    deletes the wearer's own photographs, which the product bar forbids.

    The fix is to require DETACHMENT as well as scale (see
    `SurfaceParams.pose_detach_multiple`). `max radius / p95 radius` separates
    the two cases with nothing in between: 1.00-1.77 across every capture
    below and every healthy solve in the store, against 13-11,392 on the
    solves that actually carry solver noise.
    """

    @pytest.mark.parametrize("name,centres", DWELLING_CAPTURES,
                             ids=[n for n, _ in DWELLING_CAPTURES])
    def test_no_pose_is_dropped(self, name, centres):
        report = S.robust_pose_outliers(centres)
        assert report.gated == 0, (
            f"{name}: dropped {report.gated} of {len(centres)} real poses "
            f"-- {report.detail}")

    def test_a_detached_pose_is_still_gated_beside_a_dwelling_body(self):
        """Detachment must not become an excuse to fuse anything. The body
        here is the pathological one -- 80% dwelling, then a walk -- and the
        solver noise beside it is still refused."""
        body = _dwell_then_walk(0.80, 40)
        noise = np.array([[1.0e6, 0.0, 0.0], [0.0, -5.0e5, 0.0]])
        report = S.robust_pose_outliers(np.concatenate([body, noise]))
        assert report.gated == 2
        assert report.outlier[-2] and report.outlier[-1]

    def test_the_detachment_is_recorded_so_a_reader_can_recompute_it(self, tmp_path):
        store = _synthetic_world(tmp_path)
        _graft_wild_frames(store)
        SP.surfacify(store, WORLD, SESSION, params=_params())

        rec = _outliers(_manifest(store))
        assert rec["pose_detach_multiple"] == _params().pose_detach_multiple
        assert rec["pose_radius_p95"] > 0
        assert rec["pose_gate_threshold"] >= rec["pose_radius_p95"]


class TestTheGateMayOnlyMakeASmallCorrection:
    """Review 2, M5: there was no limit on how much the gate could delete. A
    45% gate built and published half a world with a `logger.warning` and no
    change of state.

    The detachment test is the fix, and it turns out to bound the share by
    construction -- see `test_the_p95_test_caps_the_share_by_itself`. The
    explicit cap stays behind it because the bound holds only while the
    detachment test is on, and because a rule that silently relies on an
    algebraic accident of its own statistic is one refactor from not holding
    at all. Above the cap the gate stands down and lets the key-range and
    block-budget refusals speak, rather than publishing a confident
    half-world."""

    def test_the_p95_test_caps_the_share_by_itself(self):
        """Gating more than 5% would mean the pose AT the 95th percentile is
        gated -- that is, p95 > 2.5 x p95. It cannot happen while the
        detachment test is on, for any distribution.

        Asserted over every shape in this file plus deliberately hostile
        ones, because it is the property the cap is a second line for."""
        hostile = [
            np.concatenate([_blob([0, 0, 0], 1.0, 300, 20),
                            _blob([1e6, 0, 0], 1e4, 75, 21)]),
            np.concatenate([_blob([0, 0, 0], 1.0, 200, 22),
                            _blob([1e3, 0, 0], 1e2, 100, 23),
                            _blob([1e6, 0, 0], 1e4, 60, 24)]),
            np.exp(np.linspace(0, 16, 400))[:, None] * np.array([[1.0, 0, 0]]),
        ]
        for centres in [c for _n, c in DWELLING_CAPTURES] + hostile:
            report = S.robust_pose_outliers(centres, max_fraction=0.0)
            assert report.gated <= 0.05 * len(centres) + 1

    def test_a_small_correction_is_still_made(self):
        body = _blob([0, 0, 0], 1.0, 300, 10)
        strays = _blob([1.0e6, 0, 0], 1.0e4, 6, 11)
        report = S.robust_pose_outliers(np.concatenate([body, strays]))
        assert report.gated == 6 and not report.stood_down

    def test_a_gate_that_would_delete_a_large_share_stands_down(self):
        """With the detachment test off -- the configuration the cap exists
        for -- an 80%-dwell capture is exactly the 15.6% deletion review 2
        measured, and the cap refuses to make it."""
        walk = _dwell_then_walk(0.80, 40)
        loose = S.robust_pose_outliers(walk, detach=0.0, max_fraction=0.0)
        assert loose.gated > 0.05 * len(walk)          # the old behaviour

        report = S.robust_pose_outliers(walk, detach=0.0)
        assert report.gated == 0
        assert report.stood_down
        assert "%" in report.detail and "came apart" in report.detail

    def test_the_cap_can_be_switched_off(self):
        walk = _dwell_then_walk(0.80, 40)
        report = S.robust_pose_outliers(walk, detach=0.0, max_fraction=0.0)
        assert report.gated > 0 and not report.stood_down

    def test_the_stand_down_reaches_the_manifest(self, tmp_path):
        """A build that declined to correct anything has to say so. Standing
        down silently would look exactly like a clean solve."""
        store = _synthetic_world(tmp_path, views=_ring_views(8))
        _graft_wild_frames(store, n=3)          # 3 of 11 poses is 27%

        result = SP.surfacify(store, WORLD, SESSION,
                              params=_params(pose_detach_multiple=0.0))

        rec = _record(store, result)
        assert rec["poses_gated"] == 0
        assert rec["pose_gate_stood_down"] is True
        assert "came apart" in rec["pose_detail"]


# ---------------------------------------------------------------------------
# refusal is the last resort, not the first response
# ---------------------------------------------------------------------------


class TestAnUnkeyableGridIsCoarsenedBeforeItIsRefused:

    def test_block_key_still_refuses_and_says_by_how_much(self):
        """The guard stays. It is a correctness guard -- two cells sharing a
        key merge opposite ends of a scene -- and the fix is upstream of it.
        What is new is that the refusal carries the overshoot, so the caller
        can coarsen by exactly the factor needed instead of guessing."""
        import torch

        bc = torch.tensor([[0, 0, 1 << 21]], dtype=torch.int64)
        with pytest.raises(S.SurfaceUnavailable) as caught:
            S.block_key(bc)
        assert "keyable range" in caught.value.reason
        assert caught.value.key_overshoot > 1.0

    def test_a_grid_that_cannot_be_keyed_coarsens_and_builds(self, tmp_path, monkeypatch):
        """Allocation refuses once; the stage coarsens the voxel and asks
        again, rather than handing the wearer nothing."""
        store = _synthetic_world(tmp_path)
        real = S.SurfaceVolume.blocks_for_depth
        state = {"raised": False}

        def once(self, *args, **kw):
            if not state["raised"]:
                state["raised"] = True
                raise S.key_range_refusal(3.0)
            return real(self, *args, **kw)

        monkeypatch.setattr(S.SurfaceVolume, "blocks_for_depth", once)
        result = SP.surfacify(store, WORLD, SESSION, params=_params())

        assert state["raised"]
        assert result.state == SP.STATE_OK, result.detail
        assert result.faces > 0
        rec = _outliers(_manifest(store))
        assert rec["voxel_coarsened_for_key_range"] >= 3.0

    def test_a_refusal_that_survives_coarsening_is_still_honest(self, tmp_path,
                                                               monkeypatch):
        """Coarsening is a last resort, not a licence to alias. A grid that
        cannot be keyed at any voxel is still refused, by name."""
        store = _synthetic_world(tmp_path)

        def always(self, *args, **kw):
            raise S.key_range_refusal(10.0)

        monkeypatch.setattr(S.SurfaceVolume, "blocks_for_depth", always)
        result = SP.surfacify(store, WORLD, SESSION, params=_params())

        assert result.state == SP.STATE_UNAVAILABLE
        assert "keyable range" in (result.detail or "")
        assert "coarsen" in (result.detail or "")

    def test_the_coarsening_is_capped_rather_than_unbounded(self, tmp_path,
                                                            monkeypatch):
        """REVIEW 2, M4. Twelve attempts at `overshoot x 1.05` with no cap
        and no floor on the voxel will happily shrink a bedroom into a
        handful of blocks and publish it as `ok`, with a positive face count
        and an appearance built on top. A surface that is confidently wrong
        is worse than the refusal it replaced -- this project has already had
        a READY rejected because the mesh looked bad.

        The cap is a multiple of the requested voxel, so it is scene-relative
        like everything else here."""
        store = _synthetic_world(tmp_path)

        def always(self, *args, **kw):
            raise S.key_range_refusal(3.0)

        monkeypatch.setattr(S.SurfaceVolume, "blocks_for_depth", always)
        result = SP.surfacify(store, WORLD, SESSION,
                              params=_params(max_key_coarsening=4.0))

        assert result.state == SP.STATE_UNAVAILABLE
        detail = result.detail or ""
        assert "keyable range" in detail
        # named, with the cap it hit, so the refusal can be acted on
        assert "4" in detail and "coarsen" in detail

    def test_two_components_half_and_half_are_refused_not_collapsed(self, tmp_path):
        """REVIEW 2's measured route into M4, end to end and with nothing
        monkeypatched.

        A solve that came apart into two equal halves defeats both robust
        rules on purpose: the element-wise median centre lands BETWEEN the
        clusters, so every pose has about the same radius and none is an
        outlier (nothing physical privileges an even split -- it is an
        artefact of `np.median` averaging the two middle values), and the
        fusion envelope spans both halves, so no pixel is clipped either.

        What is left is the key range, and before the cap it coarsened until
        the voxel was 12,000 units across -- five thousand times the room --
        and then died of a float overflow deep inside `trunc_at`, reported as
        `failed` with no record. Now it is refused by name, with the factor
        it would have needed and the record of what the guards saw."""
        store = _synthetic_world(tmp_path, views=_ring_views(8))
        _graft_wild_frames(store, n=8, radius=5.0e5, depth=1.2e6)

        result = SP.surfacify(store, WORLD, SESSION, params=_params())

        assert result.state == SP.STATE_UNAVAILABLE, result.detail
        assert "keyable range" in result.detail
        assert "max_key_coarsening" in result.detail
        rec = _record(store, result)
        # both robust rules abstained, exactly as review 2 measured
        assert rec["poses_gated"] == 0
        assert rec["pixels_clipped_by_bound"] == 0

    def test_a_coarsening_inside_the_cap_still_builds(self, tmp_path, monkeypatch):
        store = _synthetic_world(tmp_path)
        real = S.SurfaceVolume.blocks_for_depth
        state = {"raised": False}

        def once(self, *args, **kw):
            if not state["raised"]:
                state["raised"] = True
                raise S.key_range_refusal(2.0)
            return real(self, *args, **kw)

        monkeypatch.setattr(S.SurfaceVolume, "blocks_for_depth", once)
        result = SP.surfacify(store, WORLD, SESSION,
                              params=_params(max_key_coarsening=4.0))

        assert state["raised"]
        assert result.state == SP.STATE_OK, result.detail
        rec = _outliers(_manifest(store))
        assert 2.0 <= rec["voxel_coarsened_for_key_range"] <= 4.0

    def test_too_few_poses_to_gate_falls_back_to_the_refusal(self, tmp_path):
        """REVIEW 2, the `MIN_ROBUST_POSES` discontinuity: seven poses and the
        10^6 one is FUSED, eight and it is gated. What matters is where the
        stand-down lands -- it must reach the honest refusal, not an unbounded
        coarsening that publishes a collapsed room as `ok`.

        The fusion bound is switched off here so that the pose gate is the
        only rule that could have acted, which is the state the discontinuity
        describes. With the bound on, it catches this case instead: see
        `test_a_solve_the_gate_cannot_judge_is_still_reconstructed`."""
        store = _synthetic_world(tmp_path, views=_ring_views(4))
        _graft_wild_frames(store, n=1)
        assert S.MIN_ROBUST_POSES > 5          # the gate must stand down here

        result = SP.surfacify(store, WORLD, SESSION,
                              params=_params(fusion_bound_multiple=0.0))

        assert result.state == SP.STATE_UNAVAILABLE, result.detail
        assert "keyable range" in (result.detail or "")
        rec = _record(store, result)
        assert rec["poses_gated"] == 0
        assert rec["pose_gate_stood_down"] is True
        assert "too few" in rec["pose_detail"]
        assert rec["voxel_coarsened_for_key_range"] <= _params().max_key_coarsening

    def test_a_solve_the_gate_cannot_judge_is_still_reconstructed(self, tmp_path):
        """The same five-pose session with the bound on. The room is built
        from the four real frames and the record says which rule did it."""
        store = _synthetic_world(tmp_path, views=_ring_views(4))
        _graft_wild_frames(store, n=1)

        result = SP.surfacify(store, WORLD, SESSION, params=_params())

        assert result.state == SP.STATE_OK, result.detail
        assert result.faces > 0
        rec = _record(store, result)
        assert rec["poses_gated"] == 0 and rec["pose_gate_stood_down"] is True
        assert rec["frames_emptied_by_bound"] == 1


# ---------------------------------------------------------------------------
# the tripwires have to fire on the failure they exist to detect
# ---------------------------------------------------------------------------


class TestARefusalStillSaysWhatTheGuardsDid:
    """REVIEW 2, M3. `_outlier_record` ran only on the `ok` return, so a build
    the guards themselves refused wrote `"outliers": null` and the wearer's
    one message was indistinguishable from a broken depth backend."""

    def test_a_bound_that_empties_every_frame_says_so_by_name(self, tmp_path):
        store = _synthetic_world(tmp_path)
        # a bound far too tight for the room: every frame loses every pixel
        result = SP.surfacify(store, WORLD, SESSION,
                              params=_params(fusion_bound_multiple=1e-4))

        assert result.state == SP.STATE_UNAVAILABLE
        assert "fusion bound" in (result.detail or ""), result.detail
        # NOT the generic "no frame produced usable depth", which is what a
        # missing depth network says
        assert "no frame produced usable depth" not in (result.detail or "")

    def test_the_record_survives_the_refusal(self, tmp_path):
        store = _synthetic_world(tmp_path)
        result = SP.surfacify(store, WORLD, SESSION,
                              params=_params(fusion_bound_multiple=1e-4))

        rec = _record(store, result)
        assert rec["frames_emptied_by_bound"] > 0
        assert rec["pixels_clipped_by_bound"] > 0
        assert rec["fusion_bound_lo"] is not None

    def test_clipped_pixels_are_counted_not_just_clipped_frames(self, tmp_path):
        """A frame counter cannot tell one stray pixel from a frame cut in
        half, and "one frame was clipped" is exactly the reading a person
        would shrug at."""
        store = _synthetic_world(tmp_path)
        _graft_wild_frames(store, n=2)
        # the pose gate off, so the wild frames reach the bound and are the
        # only thing it removes
        result = SP.surfacify(store, WORLD, SESSION,
                              params=_params(outlier_radius_multiple=0.0,
                                             max_key_coarsening=1e9))

        rec = _record(store, result)
        assert rec["frames_emptied_by_bound"] == 2
        assert rec["pixels_clipped_by_bound"] >= 2 * 1000
        assert rec["pixels_offered_to_bound"] > rec["pixels_clipped_by_bound"]
        assert result.state in (SP.STATE_OK, SP.STATE_UNAVAILABLE)
