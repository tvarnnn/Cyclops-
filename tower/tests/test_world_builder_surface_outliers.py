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


# ---------------------------------------------------------------------------
# the defect
# ---------------------------------------------------------------------------


class TestASolveWithWildPosesStillBuildsItsRoom:

    def test_the_room_is_reconstructed_rather_than_refused(self, tmp_path):
        """The headline. Before the robust gate this returned `unavailable`
        with `faces: 0`, and the appearance stage never ran."""
        store = _synthetic_world(tmp_path)
        _graft_wild_frames(store)

        result = SP.surfacify(store, WORLD, SESSION, params=_params())

        assert result.state == SP.STATE_OK, result.detail
        assert result.faces > 0 and result.vertices > 0
        assert result.frames_used >= 8

    def test_the_gated_poses_are_counted_and_named_in_the_manifest(self, tmp_path):
        """A surface built from fewer frames than the solve offered must say
        so. A silent drop is how a stage starts lying about its coverage."""
        store = _synthetic_world(tmp_path)
        grafted = _graft_wild_frames(store, n=3)

        SP.surfacify(store, WORLD, SESSION, params=_params())

        rec = _outliers(_manifest(store))
        assert rec["poses_gated"] == len(grafted)
        assert rec["poses_offered"] == 8 + len(grafted)
        assert rec["pose_radius_multiple"] == _params().outlier_radius_multiple
        # The criterion is a multiple of the solve's own median radius, and
        # the record has to let a reader recompute it.
        assert rec["pose_radius_median"] > 0
        assert rec["pose_radius_gated_max"] > rec["pose_radius_median"]

    def test_the_fusion_volume_is_bounded_by_the_observed_scene(self, tmp_path):
        """Not by the full bounding box. Run A's full point extent was
        1,219,893 units around a thirteen-unit bedroom."""
        store = _synthetic_world(tmp_path)
        _graft_wild_frames(store)

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
        centres = np.concatenate([_ring(), [[900.0, 0.0, 0.0]]])
        near = S.robust_pose_outliers(centres)
        far = S.robust_pose_outliers(centres * 1000.0)
        assert np.array_equal(near.outlier, far.outlier)
        assert near.outlier.sum() == 1 and near.outlier[-1]

    def test_a_clean_ring_gates_nothing(self):
        assert S.robust_pose_outliers(_ring()).outlier.sum() == 0

    def test_a_walk_is_not_mistaken_for_an_outlier(self):
        """A straight corridor spreads the centres along one axis; every one
        of them is a legitimate pose and none may be dropped."""
        walk = np.stack([np.linspace(0, 40, 60), np.zeros(60), np.zeros(60)], 1)
        assert S.robust_pose_outliers(walk).outlier.sum() == 0

    def test_too_few_poses_to_be_robust_gates_nothing(self):
        few = np.concatenate([_ring(4), [[9e5, 0.0, 0.0]]])
        report = S.robust_pose_outliers(few)
        assert report.outlier.sum() == 0
        assert "too few" in report.detail

    def test_the_rule_can_be_switched_off(self):
        centres = np.concatenate([_ring(), [[9e5, 0.0, 0.0]]])
        assert S.robust_pose_outliers(centres, multiple=0.0).outlier.sum() == 0

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
