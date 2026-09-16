"""The surface stage as the product runs it: artifact, ladder, lifecycle.

`test_world_builder_surface.py` tests the field and the format. This file
tests everything around them, end to end, on a synthetic world whose every
input is written by hand -- a solved session, a depth stage already on disk,
a box room ray-cast exactly -- so `surfacify()` runs for real without a
depth network and the questions are about behaviour:

  * does a build publish a complete, readable artifact, and only that
  * is a completed build reused, and rebuilt exactly when something changed
  * does a stop leave nothing that looks finished
  * does the final build replace the live one
  * does a broken or absent surface cost the wearer their world (it must not)
  * does the live worker leave anything running when the builder exits
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from tests.test_world_builder_surface import _camera, _look_from, _render_box_depth

from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP

WORLD, SESSION = "w1", "s1"


# ---------------------------------------------------------------------------
# a synthetic solved world with its depth stage already on disk
# ---------------------------------------------------------------------------


def _synthetic_world(tmp_path, *, n_frames=8, digest="digest-1", with_dense=True,
                     views=None, anchors=False):
    """`views`, if given, is a list of (eye, target, slab) replacing the ring.
    `anchors` writes sparse points and observations the way a solve does --
    ray-cast surface points each frame observed -- and each fit record's
    `z_sparse_min` / `z_sparse_max`."""
    import cv2

    from tower.world_builder.dense import DenseParams
    from tower.world_builder.dense_pipeline import FILL_RULE
    from tower.world_builder.global_solve import Solution, workspace_for, write_solution
    from tower.world_builder.records import CameraIntrinsics, Session, World
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    store.write_world(World(world_id=WORLD, created_at=1.0, updated_at=2.0,
                            session_ids=(SESSION,)))
    K, w, h = _camera()
    store.write_session(Session(
        session_id=SESSION, world_id=WORLD, started_at=1.0,
        redaction="faces-detected-and-filled/yunet-2023mar@0.30+plausibility1",
        intrinsics=CameraIntrinsics(
            source="self_calibrated", model="pinhole", fx=float(K[0, 0]),
            fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
            calibrated_width=w, calibrated_height=h)))
    derived = store.derived_dir(WORLD) / SESSION
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "poses.json").write_text(json.dumps({"poses": []}))
    # Real sparse points, because that is what every world built before the
    # surface stage has. An empty tree is "nothing to draw" since the
    # 869d715 hardening, which is a different world from the one modelled.
    (derived / "points.json").write_text(json.dumps({"points": [{"segment_index": 0, "xyz": [0.1 * i, 0.2, 1.0], "rgb": [120, 120, 120]} for i in range(8)]}))

    if views is None:
        views = []
        for i in range(n_frames):
            a = 2 * np.pi * i / n_frames
            eye = np.array([1.2 * np.cos(a), 0.0, 1.2 * np.sin(a)])
            views.append((eye, eye * 3.0, None))
    n_frames = len(views)
    kids, poses, depths = [], {}, []
    anchor_xyz, anchor_obs, zrange = [], [], []
    for i, (eye, target, slab) in enumerate(views):
        R, t = _look_from(eye, target)
        kid = f"{SESSION}:{i:08d}"
        kids.append(kid)
        poses[kid] = {"component": 0, "rotation": R.ravel().tolist(),
                      "translation": t.tolist(), "observations": 50}
        d = _render_box_depth(R, t, K, w, h, slab=slab)
        depths.append(d)
        if anchors:
            vv, uu = np.mgrid[4:h:16, 4:w:16]
            z = d[vv, uu]
            ok = np.isfinite(z)
            z, uu, vv = z[ok], uu[ok], vv[ok]
            Xc = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0] * z,
                           (vv + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
            base = sum(len(x) for x in anchor_xyz)
            anchor_xyz.append((Xc - t) @ R)
            anchor_obs.append(np.stack([np.full(len(z), i), np.arange(len(z)),
                                        base + np.arange(len(z))], 1))
            zrange.append((float(z.min()), float(z.max())))

    camera = {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]),
              "cy": float(K[1, 2]), "width": w, "height": h}
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-2.9, 2.9, size=(64, 3)).astype(np.float32)
    observations = np.zeros((0, 3), np.int32)
    if anchors:
        xyz = np.concatenate(anchor_xyz).astype(np.float32)
        observations = np.concatenate(anchor_obs).astype(np.int32)
    write_solution(workspace_for(store, WORLD, SESSION), Solution(
        solver="glomap", solved_at=time.time(), input_digest=digest,
        keyframe_ids=kids, poses=poses,
        components=[{"index": 0, "images": n_frames, "points": len(xyz)}],
        xyz=xyz, rgb=np.full((len(xyz), 3), 128, np.uint8),
        component=np.zeros(len(xyz), np.int32),
        first_keyframe=np.zeros(len(xyz), np.int32),
        track_length=np.full(len(xyz), 3, np.int32),
        error=np.full(len(xyz), 0.5, np.float32),
        observations=observations, camera=camera))

    if with_dense:
        dense = store.world_dir(WORLD) / "dense" / SESSION
        (dense / "work" / "depth").mkdir(parents=True)
        (dense / "work" / "undist").mkdir(parents=True)
        records = []
        for i, (kid, d) in enumerate(zip(kids, depths)):
            # kind "depth" with a = 1, b = 0 stores metric-gauge depth as-is
            np.save(dense / "work" / "depth" / f"{i:05d}.npy", d.astype(np.float16))
            np.save(dense / "work" / "depth" / f"{i:05d}_pred.npy", d.astype(np.float16))
            np.save(dense / "work" / "depth" / f"{i:05d}_fill.npy",
                    np.zeros(d.shape, bool))
            img = np.full((h, w, 3), 150 + 10 * (i % 3), np.uint8)
            cv2.imwrite(str(dense / "work" / "undist" / f"{i:05d}.jpg"), img)
            # The world's own keyframe image, and the hash a prediction is
            # bound to: a prediction is reused only for the image it came from.
            images = store.images_dir(WORLD, SESSION)
            images.mkdir(parents=True, exist_ok=True)
            ok_jpg, enc = cv2.imencode(".jpg", img)
            (images / f"{kid.rsplit(':', 1)[-1]}.jpg").write_bytes(enc.tobytes())
            records.append({"ki": i, "kid": kid, "ok": True, "a": 1.0, "b": 0.0,
                            "held_out_rel": 0.01,
                            "image_sha1": hashlib.sha1(enc.tobytes()).hexdigest(),
                            "fill_rule": FILL_RULE,
                            **({"z_sparse_min": zrange[i][0], "z_sparse_max": zrange[i][1]}
                               if anchors else {})})
        (dense / "align.json").write_text(json.dumps({
            "kind": "depth", "camera": camera, "backend": DenseParams().backend,
            "input_digest": digest, "fill_rule": FILL_RULE, "records": records}))
    return store


def _params(**kw):
    """Coarse enough to be quick on a CPU-only runner, fine enough to mesh.

    `min_support_frames=1` because this fixture's eight cameras look OUTWARD
    from the centre with a 30-degree field, so every wall patch is measured by
    exactly one frame by construction. The two-frame rule is tested on
    fixtures built to have overlap, in `test_world_builder_surface_evidence.py`:
    `TestANearGhostIsNotEmitted`, `TestTheContradictionTestIsARatio`,
    `TestASurfaceIsKeptOnlyFromItsFront`, and through `surfacify` at the
    default count in `TestAFarMeasuredWallIsKept`."""
    base = dict(voxel_frac=0.02, lod_face_targets=(0, 1500), canonical_level=0,
                mobile_level=1, smooth_iterations=2, min_weight=0.5,
                min_component_frac=0.0, min_support_frames=1)
    base.update(kw)
    return S.SurfaceParams(**base)


def _manifest(store):
    return SP.read_surface_manifest(store, WORLD, SESSION)


# ---------------------------------------------------------------------------
# a build publishes a whole artifact
# ---------------------------------------------------------------------------


class TestABuildPublishesAWholeArtifact:

    def test_the_artifact_is_complete_readable_and_says_what_it_is(self, tmp_path):
        store = _synthetic_world(tmp_path)
        result = SP.surfacify(store, WORLD, SESSION, params=_params())
        assert result.state == SP.STATE_OK, result.detail

        man = _manifest(store)
        assert man is not None
        assert man["format"] == S.SURFACE_FORMAT
        assert man["input_digest"] == "digest-1"
        assert man["params"]["quality"] == "final"
        assert "unobserved space is absent" in man["closure"]
        assert man["scale"]["state"] == "unknown"
        assert len(man["levels"]) == 2
        for lv in man["levels"]:
            V, F, C, N = S.read_mesh_bytes(
                SP.read_surface_level(store, WORLD, SESSION, lv["level"]))
            assert len(F) == lv["faces"] and len(F) > 0
            assert N is not None

    def test_nothing_is_left_behind_but_the_artifact(self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        root = SP.surface_dir(store, WORLD, SESSION)
        names = {p.name for p in root.iterdir()}
        assert ".surface.lock" not in names
        assert not [n for n in names if ".tmp" in n or n.endswith(".staging")], names
        status = json.loads((root / "status.json").read_text())
        assert status["state"] == SP.STATE_OK

    def test_the_median_scene_depth_is_the_depth_the_cameras_measured(self, tmp_path):
        """Not the distance from the mean camera centre, which on the real
        world was 25% larger and silently grew every voxel by a quarter."""
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        man = _manifest(store)
        # cameras sit 1.2 from the centre of a 6-unit box looking outwards,
        # so every measured depth lies between 1.8 and about 3.3
        assert 1.8 <= man["median_scene_depth"] <= 3.4


class TestACompletedBuildIsReusedAndRebuiltExactlyWhenItShouldBe:

    def test_a_second_run_with_nothing_changed_is_a_no_op(self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        first = _manifest(store)["built_at"]
        again = SP.surfacify(store, WORLD, SESSION, params=_params())
        assert again.state == SP.STATE_OK
        assert _manifest(store)["built_at"] == first

    def test_changed_parameters_rebuild(self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        first = _manifest(store)["built_at"]
        SP.surfacify(store, WORLD, SESSION, params=_params(min_weight=0.8))
        assert _manifest(store)["built_at"] != first

    def test_a_new_solve_makes_the_surface_behind_then_a_rebuild_catches_up(self, tmp_path):
        store = _synthetic_world(tmp_path, digest="digest-1")
        SP.surfacify(store, WORLD, SESSION, params=_params())
        _synthetic_world(tmp_path, digest="digest-2", with_dense=False)

        cur = SP.surface_currency(store, WORLD, SESSION, _manifest(store))
        assert cur["present"] and not cur["current"]

        # The depth stage for the new solve. Without it the cache names
        # digest-1 and is -- correctly, since the cache-identity fix --
        # refused; this test used to pass only because it was not.
        dense = store.world_dir(WORLD) / "dense" / SESSION
        align = json.loads((dense / "align.json").read_text())
        align["input_digest"] = "digest-2"
        (dense / "align.json").write_text(json.dumps(align))

        SP.surfacify(store, WORLD, SESSION, params=_params())
        man = _manifest(store)
        assert man["input_digest"] == "digest-2"
        assert SP.surface_currency(store, WORLD, SESSION, man)["current"]

    def test_a_missing_level_file_is_not_mistaken_for_a_finished_build(self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        root = SP.surface_dir(store, WORLD, SESSION)
        first = _manifest(store)["built_at"]
        level1 = SP.level_file(root, _manifest(store)["levels"][1])
        level1.unlink()
        SP.surfacify(store, WORLD, SESSION, params=_params())
        assert SP.level_file_whole(root, _manifest(store)["levels"][1]) is not None
        assert _manifest(store)["built_at"] != first

    def test_pruned_depth_maps_are_not_trusted_as_a_cache(self, tmp_path):
        """`prune_intermediates` deletes `work/` after a dense pack. A valid
        `align.json` with nothing behind it must send the stage back to the
        depth network, not to files that no longer exist."""
        import shutil

        from tower.world_builder.dense import DenseParams
        from tower.world_builder.global_solve import load_solution

        store = _synthetic_world(tmp_path)
        dense = store.world_dir(WORLD) / "dense" / SESSION
        align = json.loads((dense / "align.json").read_text())
        solution = load_solution(store, WORLD, SESSION)
        assert SP._depth_cache_usable(align, dense, solution, DenseParams())
        shutil.rmtree(dense / "work")
        assert not SP._depth_cache_usable(align, dense, solution, DenseParams())


class TestAStopLeavesNothingThatLooksFinished:

    def test_a_stop_during_fusion_publishes_no_surface(self, tmp_path):
        store = _synthetic_world(tmp_path)
        calls = {"n": 0}

        def stop_after_depth():
            calls["n"] += 1
            return calls["n"] > 1        # let the depth check pass, stop in fuse

        result = SP.surfacify(store, WORLD, SESSION, params=_params(),
                              should_stop=stop_after_depth)
        assert result.state == SP.STATE_STOPPED
        root = SP.surface_dir(store, WORLD, SESSION)
        assert _manifest(store) is None
        assert not list(root.glob("mesh_l*.bin"))
        assert not (root / ".surface.lock").exists()
        assert json.loads((root / "status.json").read_text())["state"] == SP.STATE_STOPPED

    def test_a_stop_does_not_destroy_a_previous_good_surface(self, tmp_path):
        """A re-run that is interrupted must leave the world's existing
        surface exactly as it was -- the wearer's world does not vanish
        because a refinement was cancelled."""
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        before = _manifest(store)
        level = SP.read_surface_level(store, WORLD, SESSION, 0)

        result = SP.surfacify(store, WORLD, SESSION, params=_params(min_weight=0.9),
                              force=True, should_stop=lambda: True)
        assert result.state == SP.STATE_STOPPED
        assert _manifest(store)["built_at"] == before["built_at"]
        assert SP.read_surface_level(store, WORLD, SESSION, 0) == level


class TestLiveThenFinal:

    def test_the_final_build_replaces_the_live_one(self, tmp_path):
        store = _synthetic_world(tmp_path)
        live = S.SurfaceParams.live(lod_face_targets=(0, 1500), min_component_frac=0.0,
                                    min_support_frames=1,
                                    voxel_frac=0.04, min_weight=0.5)
        assert SP.surfacify(store, WORLD, SESSION, params=live).state == SP.STATE_OK
        man_live = _manifest(store)
        assert man_live["params"]["quality"] == "live"

        final = _params()
        assert SP.surfacify(store, WORLD, SESSION, params=final, force=True).state == SP.STATE_OK
        man_final = _manifest(store)
        assert man_final["params"]["quality"] == "final"
        assert man_final["voxel"] < man_live["voxel"]
        assert man_final["faces"] > man_live["faces"]

    def test_live_and_final_parameters_never_share_a_cache_key(self):
        live, final = S.SurfaceParams.live(), S.SurfaceParams()
        assert SP._params_digest(live, "d") != SP._params_digest(final, "d")

    def test_live_relaxes_resolution_not_the_evidence_rule_beyond_a_floor(self):
        live, final = S.SurfaceParams.live(), S.SurfaceParams()
        assert live.voxel_frac > final.voxel_frac
        assert live.carve is True
        assert live.min_weight >= 1.0
        assert live.gate_rel == final.gate_rel


class TestRefusals:

    def test_no_solve_is_unavailable_and_releases_the_lock(self, tmp_path):
        from tower.world_builder.records import Session, World
        from tower.world_builder.store import WorldStore

        store = WorldStore(tmp_path)
        store.write_world(World(world_id=WORLD, created_at=1.0, updated_at=1.0,
                                session_ids=(SESSION,)))
        store.write_session(Session(session_id=SESSION, world_id=WORLD, started_at=1.0))
        result = SP.surfacify(store, WORLD, SESSION, params=_params())
        assert result.state == SP.STATE_UNAVAILABLE
        assert "global solution" in result.detail
        assert not (SP.surface_dir(store, WORLD, SESSION) / ".surface.lock").exists()

    def test_every_frame_failing_the_gate_is_unavailable_not_an_empty_world(self, tmp_path):
        store = _synthetic_world(tmp_path)
        dense = store.world_dir(WORLD) / "dense" / SESSION
        align = json.loads((dense / "align.json").read_text())
        for r in align["records"]:
            r["held_out_rel"] = 0.5
        (dense / "align.json").write_text(json.dumps(align))
        result = SP.surfacify(store, WORLD, SESSION, params=_params())
        assert result.state == SP.STATE_UNAVAILABLE
        assert _manifest(store) is None


class TestTheLock:

    def test_two_builds_of_one_session_do_not_interleave(self, tmp_path):
        root = tmp_path / "surface" / "s1"
        a, b = SP._SurfaceLock(root), SP._SurfaceLock(root)
        assert a.acquire()
        assert not b.acquire()
        a.release()
        assert b.acquire()
        b.release()

    def test_the_loser_of_the_race_does_not_overwrite_the_winners_status(self, tmp_path):
        store = _synthetic_world(tmp_path)
        root = SP.surface_dir(store, WORLD, SESSION)
        root.mkdir(parents=True, exist_ok=True)
        holder = SP._SurfaceLock(root)
        assert holder.acquire()
        (root / "status.json").write_text(json.dumps({"state": "running", "pid": os.getpid()}))
        result = SP.surfacify(store, WORLD, SESSION, params=_params())
        assert result.state == SP.STATE_UNAVAILABLE
        assert json.loads((root / "status.json").read_text())["state"] == "running"
        holder.release()

    def test_a_lock_left_by_a_dead_process_is_reclaimed(self, tmp_path):
        """The Job Object kills a builder on a 30 s grace and its lock stays
        behind. The next build must take it, not refuse forever."""
        root = tmp_path / "surface" / "s1"
        root.mkdir(parents=True)
        (root / ".surface.lock").write_text(json.dumps({"pid": 0x7FFFFFFE, "at": 0}))
        lock = SP._SurfaceLock(root)
        assert lock.acquire()
        assert json.loads((root / ".surface.lock").read_text())["pid"] == os.getpid()
        lock.release()

    def test_a_running_status_with_a_dead_pid_is_reported_stale(self):
        assert SP.status_is_stale({"state": "running", "pid": 0x7FFFFFFE})
        assert not SP.status_is_stale({"state": "running", "pid": os.getpid()})
        assert not SP.status_is_stale({"state": "ok", "pid": 0x7FFFFFFE})


# ---------------------------------------------------------------------------
# the ladder: the best rung the session has, and never less than it had
# ---------------------------------------------------------------------------


class TestTheLadder:

    def _built(self, tmp_path):
        store = _synthetic_world(tmp_path)
        assert SP.surfacify(store, WORLD, SESSION, params=_params()).state == SP.STATE_OK
        return store

    def test_auto_serves_the_surface(self, tmp_path):
        from tower.results.world_builder_render import build_world_render

        html = build_world_render(self._built(tmp_path), WORLD, SESSION)
        assert "webgl2" in html and "Reconstructed surface" in html

    def test_sparse_is_still_reachable_on_request(self, tmp_path):
        from tower.results.world_builder_render import build_world_render

        html = build_world_render(self._built(tmp_path), WORLD, SESSION,
                                  representation="sparse")
        assert "Reconstructed surface" not in html

    def test_a_world_with_no_surface_gets_the_page_it_got_before(self, tmp_path):
        from tower.results.world_builder_render import build_world_render

        store = _synthetic_world(tmp_path)
        before = build_world_render(store, WORLD, SESSION, representation="sparse")
        auto = build_world_render(store, WORLD, SESSION)
        assert "Reconstructed surface" not in auto
        assert auto == before

    def test_a_torn_level_falls_through_rather_than_serving_a_blank_canvas(self, tmp_path):
        from tower.results.world_builder_render import build_world_render

        store = self._built(tmp_path)
        root = SP.surface_dir(store, WORLD, SESSION)
        for level in root.glob("mesh_l*.bin"):
            level.write_bytes(level.read_bytes()[:-5])
        html = build_world_render(store, WORLD, SESSION)
        assert "Reconstructed surface" not in html

    def test_a_manifest_of_another_format_is_ignored(self, tmp_path):
        from tower.results.world_builder_render import build_world_render

        store = self._built(tmp_path)
        path = SP.surface_dir(store, WORLD, SESSION) / "manifest.json"
        man = json.loads(path.read_text())
        man["format"] = "wb-surface-mesh/999"
        path.write_text(json.dumps(man))
        assert SP.read_surface_manifest(store, WORLD, SESSION) is None
        assert "Reconstructed surface" not in build_world_render(store, WORLD, SESSION)

    def test_a_truncated_manifest_is_absent_not_an_error(self, tmp_path):
        store = self._built(tmp_path)
        path = SP.surface_dir(store, WORLD, SESSION) / "manifest.json"
        path.write_text(path.read_text()[:40])
        assert SP.read_surface_manifest(store, WORLD, SESSION) is None

    def test_pinning_surface_on_a_world_without_one_is_404_not_a_substitute(self, tmp_path):
        from tower.results.world_builder_render import (
            WorldRenderUnavailable,
            build_world_render,
        )

        store = _synthetic_world(tmp_path)
        with pytest.raises(WorldRenderUnavailable):
            build_world_render(store, WORLD, SESSION, representation="surface")

    def test_a_surface_module_that_will_not_import_costs_nothing(self, tmp_path, monkeypatch):
        """The failure the `_ViewerModuleMissing` binding exists for: with the
        exception name bound to None, `except None` raised TypeError out of the
        fallback, and the route 500ed instead of serving the old page."""
        import builtins

        from tower.results.world_builder_render import build_world_render

        store = self._built(tmp_path)
        real_import = builtins.__import__

        def broken(name, *a, **kw):
            if name == "tower.world_builder.surface_render":
                raise ImportError("simulated broken module")
            return real_import(name, *a, **kw)

        monkeypatch.setitem(sys.modules, "tower.world_builder.surface_render", None)
        monkeypatch.setattr(builtins, "__import__", broken)
        html = build_world_render(store, WORLD, SESSION)
        assert "Reconstructed surface" not in html


class TestThePage:

    def test_a_small_budget_selects_a_coarser_rung_and_stays_whole(self, tmp_path):
        from tower.world_builder.surface_render import build_surface_payload

        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        levels = _manifest(store)["levels"]
        smallest = min(levels, key=lambda lv: lv["bytes"])
        raw, config = build_surface_payload(store, WORLD, SESSION, budget_bytes=1)
        assert config["level"] == smallest["level"]
        S.read_mesh_bytes(raw)

    def test_max_points_is_honoured_rather_than_ignored(self, tmp_path):
        from tower.world_builder.surface_render import build_surface_page

        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        big = build_surface_page(store, WORLD, SESSION)
        small = build_surface_page(store, WORLD, SESSION, max_points=1)
        assert len(small) <= len(big)

    def test_the_page_loads_nothing_from_anywhere(self, tmp_path):
        """The WKWebView page has no origin and the route's CSP forbids every
        external resource; one stray URL is a blank screen on the phone."""
        import re

        from tower.world_builder.surface_render import build_surface_page

        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        html = build_surface_page(store, WORLD, SESSION)
        assert not re.search(r"""(src|href)\s*=\s*["'](https?:)?//""", html)
        assert "fetch(" not in html and "XMLHttpRequest" not in html
        assert "__WB_SURFACE" not in html

    def test_the_page_never_claims_metres_when_scale_is_unknown(self, tmp_path):
        from tower.world_builder.surface_render import build_surface_page

        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        html = build_surface_page(store, WORLD, SESSION)
        assert "Scale is unknown" in html
        assert " metres" not in html and " meters" not in html

    def test_the_config_cannot_break_out_of_the_script_tag(self):
        from tower.world_builder.surface_render import js_object_literal

        lit = js_object_literal({"x": "</script><script>alert(1)</script> "})
        assert "</script>" not in lit and " " not in lit

    def test_the_template_is_installed_beside_the_code(self):
        from tower.world_builder.surface_render import (
            TOKEN_CONFIG,
            TOKEN_MESH,
            viewer_template_path,
        )

        text = viewer_template_path().read_text(encoding="utf-8")
        assert TOKEN_CONFIG in text and TOKEN_MESH in text


# ---------------------------------------------------------------------------
# the live worker: owned, bounded, and gone when the builder is
# ---------------------------------------------------------------------------


class TestTheLiveWorker:

    def _surfacer(self, tmp_path, spawned):
        from scripts.world_build_session import BackgroundSurface

        def spawn(argv, **kw):
            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            spawned.append((argv, proc))
            return proc

        return BackgroundSurface(root=tmp_path, world_id=WORLD, session_id=SESSION,
                                 spawn=spawn)

    def test_it_runs_one_job_at_a_time_and_asks_for_the_live_preset(self, tmp_path):
        spawned = []
        s = self._surfacer(tmp_path, spawned)
        try:
            assert s.maybe_launch(None, solver_running=False)
            assert not s.maybe_launch(None, solver_running=False)
            argv = spawned[0][0]
            assert "--live" in argv and "--force" in argv
            assert argv[argv.index("--world") + 1] == WORLD
        finally:
            s.close()

    def test_close_leaves_no_process_behind(self, tmp_path):
        import psutil

        spawned = []
        s = self._surfacer(tmp_path, spawned)
        s.maybe_launch(None, solver_running=False)
        pid = spawned[0][1].pid
        assert psutil.pid_exists(pid)
        s.close()
        deadline = time.time() + 10
        while time.time() < deadline and psutil.pid_exists(pid):
            try:
                if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.1)
        assert not s.running
        assert (not psutil.pid_exists(pid)
                or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE)

    def test_close_with_nothing_running_is_harmless(self, tmp_path):
        s = self._surfacer(tmp_path, [])
        s.close()
        s.close()


class TestASurfaceIsGeometry:
    """A session whose only drawable geometry is its reconstruction.

    The render route and the Saved Worlds listing both ask
    `session_has_drawable_geometry` BEFORE the ladder is consulted, and it
    used to look at the sparse tree alone. A session with a complete surface
    and an empty sparse tree was refused as "no geometry yet" and listed with
    `has_geometry: false` -- a reconstructed world shown as empty.
    """

    def _surface_only(self, tmp_path):
        store = _synthetic_world(tmp_path)
        assert SP.surfacify(store, WORLD, SESSION, params=_params()).state == SP.STATE_OK
        derived = store.derived_dir(WORLD) / SESSION
        (derived / "points.json").write_text(json.dumps({"points": []}))
        return store

    def test_it_counts_as_drawable(self, tmp_path):
        from tower.world_builder.store import session_has_drawable_geometry

        assert session_has_drawable_geometry(self._surface_only(tmp_path), WORLD, SESSION)

    def test_it_is_served_rather_than_refused(self, tmp_path):
        from tower.results.world_builder_render import build_world_render

        html = build_world_render(self._surface_only(tmp_path), WORLD, SESSION)
        assert "Reconstructed surface" in html

    def test_a_torn_surface_with_nothing_else_is_honestly_nothing(self, tmp_path):
        from tower.world_builder.store import session_has_drawable_geometry

        store = self._surface_only(tmp_path)
        for level in SP.surface_dir(store, WORLD, SESSION).glob("mesh_l*.bin"):
            level.write_bytes(level.read_bytes()[:-3])
        assert not session_has_drawable_geometry(store, WORLD, SESSION)

    def test_a_surface_with_no_triangles_is_not_geometry(self, tmp_path):
        from tower.world_builder.store import session_has_drawable_geometry

        store = self._surface_only(tmp_path)
        path = SP.surface_dir(store, WORLD, SESSION) / "manifest.json"
        man = json.loads(path.read_text())
        man["faces"] = 0
        path.write_text(json.dumps(man))
        assert not session_has_drawable_geometry(store, WORLD, SESSION)


# ---------------------------------------------------------------------------
# every rung declares itself
#
# The phone's native caption follows the rung the page declares. The app used
# to caption every page "Not a surface"; now it reads
# <meta name="wb-representation"> from the first 4096 characters of the page
# and says what is shown, so every rung must declare itself, in the head.
# ---------------------------------------------------------------------------


def _declared(html):
    import re

    m = re.search(r'name="wb-representation" content="([a-z]+)"', html[:4096])
    return m.group(1) if m else None


def test_the_surface_page_declares_surface(tmp_path):
    from tower.world_builder.surface_render import build_surface_page

    store = _synthetic_world(tmp_path)
    SP.surfacify(store, WORLD, SESSION, params=_params())
    assert _declared(build_surface_page(store, WORLD, SESSION)) == "surface"


def test_the_sparse_page_declares_sparse(tmp_path):
    from tower.results.world_builder_render import build_world_render

    store = _synthetic_world(tmp_path)
    assert _declared(build_world_render(store, WORLD, SESSION,
                                        representation="sparse")) == "sparse"


def test_the_dense_template_declares_dense():
    from tower.world_builder.dense_render import viewer_template_path

    assert _declared(viewer_template_path().read_text(encoding="utf-8")) == "dense"


# ---------------------------------------------------------------------------
# the revision: how an open picture learns a better one was built
# ---------------------------------------------------------------------------


def _meta(html, name):
    import re

    m = re.search(rf'<meta name="{name}" content="([^"]*)">', html[:4096])
    return m.group(1) if m else None


def test_every_served_page_carries_the_revision_the_endpoint_reports(tmp_path):
    from tower.results.world_builder_render import (
        build_render_revision,
        build_world_render,
    )

    store = _synthetic_world(tmp_path)
    sparse = build_world_render(store, WORLD, SESSION)
    rev = build_render_revision(store, WORLD, SESSION)
    assert rev["representation"] == "sparse"
    assert _meta(sparse, "wb-revision") == rev["revision"]

    SP.surfacify(store, WORLD, SESSION, params=_params())
    surface = build_world_render(store, WORLD, SESSION)
    rev2 = build_render_revision(store, WORLD, SESSION)
    assert rev2["representation"] == "surface"
    assert _meta(surface, "wb-revision") == rev2["revision"]
    assert rev2["revision"] != rev["revision"], "stepping up the ladder must change it"


def test_a_rebuilt_surface_changes_the_revision(tmp_path):
    from tower.results.world_builder_render import build_render_revision

    store = _synthetic_world(tmp_path)
    SP.surfacify(store, WORLD, SESSION, params=_params())
    first = build_render_revision(store, WORLD, SESSION)["revision"]
    time.sleep(0.01)
    SP.surfacify(store, WORLD, SESSION, params=_params(min_weight=0.8))
    assert build_render_revision(store, WORLD, SESSION)["revision"] != first


def test_the_sparse_revision_does_not_churn_with_every_build(tmp_path):
    """The derived tree is rewritten every few keyframes during a walk. A
    picture that reloaded on each of those would be unusable to look at."""
    from tower.results.world_builder_render import build_render_revision

    store = _synthetic_world(tmp_path)
    first = build_render_revision(store, WORLD, SESSION)["revision"]
    derived = store.derived_dir(WORLD) / SESSION
    (derived / "points.json").write_text(json.dumps({"points": [
        {"segment_index": 0, "xyz": [0.3 * i, 0.1, 2.0], "rgb": [1, 2, 3]}
        for i in range(20)]}))
    assert build_render_revision(store, WORLD, SESSION)["revision"] == first


def test_the_revision_route_answers_and_404s_like_the_page(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tower.routes.geometry import router

    store = _synthetic_world(tmp_path)
    SP.surfacify(store, WORLD, SESSION, params=_params())
    app = FastAPI()
    app.include_router(router)
    app.state.world_root = tmp_path
    client = TestClient(app)

    ok = client.get(f"/worlds/{WORLD}/render/revision", params={"session_id": SESSION})
    assert ok.status_code == 200
    assert ok.headers["cache-control"] == "no-store"
    body = ok.json()
    assert body["representation"] == "surface"
    assert body["revision"].startswith(f"{SESSION}/surface:")
    assert len(ok.content) < 512

    missing = client.get("/worlds/nope/render/revision")
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# the depth cache belongs to one solve; its predictions belong to the keyframes
# ---------------------------------------------------------------------------


class _CountingBackend:
    """A depth network that records whether it was asked anything."""

    name = "counting-stub"
    licence = "test"
    windowed = False
    window_size = 0
    kind = "depth"
    calls = 0

    def predict(self, rgb):
        type(self).calls += 1
        return np.full(rgb.shape[:2], 2.0, np.float32)


class TestTheDepthCacheBelongsToOneSolve:
    """Measured on a real-time replay of the canonical walk: the one live
    surface cached a depth stage fitted against the first 51 keyframes, and
    the FINAL surface after Stop reused it, because the cache never named its
    solve and "unnamed" counted as a match. The finished world was a
    51-keyframe mesh at the live voxel, under the final solve's digest."""

    def _stub(self, tmp_path, cached_digest):
        from tower.world_builder.dense import DenseParams, register_backend

        _CountingBackend.calls = 0
        register_backend(_CountingBackend.name, _CountingBackend)
        store = _synthetic_world(tmp_path, digest="final-solve")
        dense = store.world_dir(WORLD) / "dense" / SESSION
        align = json.loads((dense / "align.json").read_text())
        align["backend"] = _CountingBackend.name
        if cached_digest is None:
            align.pop("input_digest", None)
        else:
            align["input_digest"] = cached_digest
        (dense / "align.json").write_text(json.dumps(align))
        return store, dense, DenseParams(backend=_CountingBackend.name)

    def test_an_unnamed_cache_is_not_a_match(self, tmp_path):
        from tower.world_builder.global_solve import load_solution

        store, dense, dparams = self._stub(tmp_path, None)
        align = json.loads((dense / "align.json").read_text())
        assert not SP._depth_cache_usable(
            align, dense, load_solution(store, WORLD, SESSION), dparams)

    def test_a_cache_for_another_solve_is_not_a_match(self, tmp_path):
        from tower.world_builder.global_solve import load_solution

        store, dense, dparams = self._stub(tmp_path, "live-solve-51")
        align = json.loads((dense / "align.json").read_text())
        assert not SP._depth_cache_usable(
            align, dense, load_solution(store, WORLD, SESSION), dparams)

    def test_a_cache_for_this_solve_is_used_as_is(self, tmp_path):
        from tower.world_builder.global_solve import load_solution

        store, dense, dparams = self._stub(tmp_path, "final-solve")
        align = json.loads((dense / "align.json").read_text())
        assert SP._depth_cache_usable(
            align, dense, load_solution(store, WORLD, SESSION), dparams)

    def test_another_solves_predictions_are_refitted_not_repredicted(self, tmp_path):
        """The fits are redone against this solve; the network is not rerun,
        because its output depends only on the keyframe image."""
        from tower.world_builder.global_solve import load_solution

        store, dense, dparams = self._stub(tmp_path, "live-solve-51")
        solution = load_solution(store, WORLD, SESSION)
        session = store.read_session(WORLD, SESSION)
        align, work = SP.ensure_depth_stage(
            store, WORLD, SESSION, solution, session.intrinsics,
            gate_rel=0.08, backend=_CountingBackend.name)

        assert _CountingBackend.calls == 0, "every prediction was on disk"
        assert align["input_digest"] == "final-solve"
        assert align["cache_key"].startswith("final-solve|")
        assert {r.get("kid") for r in align["records"]} == set(solution.keyframe_ids)
        # This synthetic solve has no sparse observations, so every refit
        # honestly refuses rather than keeping the old fit's a = 1, b = 0.
        assert not any(r.get("ok") for r in align["records"])
        on_disk = json.loads((dense / "align.json").read_text())
        assert on_disk["input_digest"] == "final-solve"

    def test_a_prediction_recorded_for_another_keyframe_is_not_offered(self, tmp_path):
        from tower.world_builder.dense_pipeline import reusable_predictions

        store, dense, dparams = self._stub(tmp_path, "live-solve-51")
        align = json.loads((dense / "align.json").read_text())
        align["records"][0]["kid"] = "someone-else:00000000"
        (dense / "align.json").write_text(json.dumps(align))
        mapping = reusable_predictions(dense / "align.json", _CountingBackend.name)
        assert mapping[0][0] == "someone-else:00000000"
        assert mapping[0][0] != f"{SESSION}:{0:08d}"

    def test_another_backends_predictions_are_never_offered(self, tmp_path):
        from tower.world_builder.dense_pipeline import reusable_predictions

        store, dense, dparams = self._stub(tmp_path, "live-solve-51")
        assert reusable_predictions(dense / "align.json", "some-other-network") == {}


class TestTheLiveSurfaceIsNotStarved:
    """On a real-time replay the live surface built ONCE in a 140 s walk: each
    time a solve landed the next solve launched first, and the surface waited
    for "no solve running", which never came."""

    def _surfacer(self, tmp_path, spawned):
        from scripts.world_build_session import BackgroundSurface

        def spawn(argv, **kw):
            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            spawned.append((argv, kw, proc))
            return proc

        return BackgroundSurface(root=tmp_path, world_id=WORLD, session_id=SESSION,
                                 spawn=spawn)

    def test_a_landed_solve_launches_a_build(self, tmp_path):
        spawned = []
        s = self._surfacer(tmp_path, spawned)
        try:
            assert s.solve_landed(None)
            assert len(spawned) == 1
        finally:
            s.close()

    def test_a_solve_that_lands_mid_build_is_built_next(self, tmp_path):
        spawned = []
        s = self._surfacer(tmp_path, spawned)
        try:
            assert s.solve_landed(None)
            assert not s.solve_landed(None), "one build at a time"
            assert not s.poll(None), "still building"
            spawned[0][2].kill()
            spawned[0][2].wait(timeout=10)
            assert s.poll(None), "the solve that landed mid-build was dropped"
            assert len(spawned) == 2
            assert not s.poll(None), "and it is built only once"
        finally:
            s.close()

    def test_nothing_is_launched_when_no_solve_is_owed(self, tmp_path):
        spawned = []
        s = self._surfacer(tmp_path, spawned)
        assert not s.poll(None)
        assert spawned == []

    @pytest.mark.skipif(os.name != "nt", reason="Windows priority classes")
    def test_it_yields_the_cpu_to_the_solve_and_to_ingestion(self, tmp_path):
        spawned = []
        s = self._surfacer(tmp_path, spawned)
        try:
            s.solve_landed(None)
            flags = spawned[0][1].get("creationflags", 0)
            assert flags & subprocess.BELOW_NORMAL_PRIORITY_CLASS
        finally:
            s.close()


def _settings(**kw):
    from tower.config import Settings

    return Settings(host="0.0.0.0", port=8000, dev_mode=True,
                    cv_experiment="baseline", cv_device="cpu",
                    world_root="C:/w", world_autobuild=True, **kw)


def test_the_product_builds_a_surface_by_default():
    from tower.config import Settings
    from tower.main import _world_build_spec

    spec = _world_build_spec(_settings())
    assert "--surface" in spec.argv and "--solve" in spec.argv


def test_the_surface_can_be_turned_off_and_needs_the_solve():
    from tower.config import Settings
    from tower.main import _world_build_spec

    off = _world_build_spec(_settings(world_surface=False))
    no_solve = _world_build_spec(_settings(world_solve=False))
    assert "--surface" not in off.argv
    assert "--surface" not in no_solve.argv


class TestTheFieldHasABudget:
    """A 20-30 minute walk projected to exhaust a 12 GB card past ~900 frames
    of new space. A walk that would exceed the budget gets a coarser voxel,
    recorded, instead of an out-of-memory failure."""

    def test_an_over_budget_walk_is_coarsened_and_says_so(self, tmp_path):
        store = _synthetic_world(tmp_path)
        free = SP.surfacify(store, WORLD, SESSION, params=_params())
        free_voxel = _manifest(store)["voxel"]
        free_blocks = free.blocks

        tight = SP.surfacify(store, WORLD, SESSION, force=True,
                             params=_params(max_blocks=max(1, free_blocks // 4)))
        assert tight.state == SP.STATE_OK
        man = _manifest(store)
        detail = json.loads(tight.detail)
        assert tight.blocks <= max(1, free_blocks // 4)
        assert man["voxel"] > free_voxel * 1.5
        assert detail["voxel_coarsened_by"] > 1.5
        assert man["faces"] > 0

    def test_a_walk_within_budget_is_untouched(self, tmp_path):
        store = _synthetic_world(tmp_path)
        result = SP.surfacify(store, WORLD, SESSION, params=_params())
        assert json.loads(result.detail)["voxel_coarsened_by"] == 1.0

    def test_every_level_after_the_first_is_decimated_from_its_predecessor(self, tmp_path):
        import inspect

        body = inspect.getsource(SP._build)
        assert "decimate(*source, target)" in body

    def test_the_status_names_the_stage_that_is_running(self, tmp_path):
        store = _synthetic_world(tmp_path)
        seen = []
        root = SP.surface_dir(store, WORLD, SESSION)

        def progress(stage, done, total):
            try:
                seen.append(json.loads((root / "status.json").read_text()).get("stage"))
            except (OSError, ValueError):
                pass

        SP.surfacify(store, WORLD, SESSION, params=_params(), progress=progress)
        assert "fuse" in seen



# ---------------------------------------------------------------------------
# the systems review, pinned: each class names the finding it closes
# ---------------------------------------------------------------------------


class TestABuildIsPublishedAsAUnit:
    """M1. Every build wrote `mesh_l<n>.bin` and the manifest last, so a stop or
    a kill between those writes left new levels under an old manifest: the page
    served one surface while the revision and the listing described another."""

    def test_levels_carry_their_builds_own_names(self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        levels = _manifest(store)["levels"]
        names = [lv["file"] for lv in levels]
        assert len(set(names)) == len(names)
        suffixes = {n.split(".", 1)[1] for n in names}
        assert len(suffixes) == 1, "one build, one suffix"

    def test_a_second_build_does_not_touch_the_first_builds_files(self, tmp_path):
        store = _synthetic_world(tmp_path)
        root = SP.surface_dir(store, WORLD, SESSION)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        first = _manifest(store)
        first_bytes = {lv["file"]: (root / lv["file"]).read_bytes() for lv in first["levels"]}
        SP.surfacify(store, WORLD, SESSION, params=_params(min_weight=0.8), force=True)
        for name, data in first_bytes.items():
            assert (root / name).read_bytes() == data, "a recent build's files were rewritten"

    def test_a_stop_during_pack_publishes_nothing_new(self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        before = _manifest(store)
        root = SP.surface_dir(store, WORLD, SESSION)

        def stop_in_pack():
            try:
                return json.loads((root / "status.json").read_text()).get("stage") == "pack"
            except (OSError, ValueError):
                return False

        result = SP.surfacify(store, WORLD, SESSION, params=_params(min_weight=0.8),
                              force=True, should_stop=stop_in_pack)
        assert result.state == SP.STATE_STOPPED
        assert _manifest(store) == before
        for lv in before["levels"]:
            assert SP.level_file_whole(root, lv) is not None

    def test_a_level_whose_size_does_not_match_its_manifest_is_refused(self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        root = SP.surface_dir(store, WORLD, SESSION)
        entry = _manifest(store)["levels"][0]
        path = SP.level_file(root, entry)
        path.write_bytes(path.read_bytes() + b"\0" * 16)
        with pytest.raises(S.SurfaceUnavailable):
            SP.read_surface_level(store, WORLD, SESSION, 0)
        assert SP.level_file_whole(root, entry) is None

    def test_a_manifest_cannot_name_a_file_outside_its_directory(self, tmp_path):
        root = tmp_path
        for bad in ("../x.bin", "..\\x.bin", "/etc/x.bin", ".hidden.bin", "x.json"):
            assert SP.level_file(root, {"level": 0, "file": bad, "bytes": 1}) is None

    def test_superseded_levels_are_pruned_only_once_old(self, tmp_path):
        import os as _os

        root = tmp_path
        (root / "mesh_l0.old.bin").write_bytes(b"x")
        (root / "mesh_l0.new.bin").write_bytes(b"y")
        SP._prune_superseded_levels(root, {"mesh_l0.new.bin"})
        assert (root / "mesh_l0.old.bin").exists(), "a reader may still hold that manifest"
        past = time.time() - 3600
        _os.utime(root / "mesh_l0.old.bin", (past, past))
        SP._prune_superseded_levels(root, {"mesh_l0.new.bin"})
        assert not (root / "mesh_l0.old.bin").exists()
        assert (root / "mesh_l0.new.bin").exists()


class TestAnIncompleteDepthStageIsNotACache:
    """M2. A stop mid-depth left an align.json holding only the frames reached,
    and it was trusted: a surface from 3 of 8 frames under the full digest."""

    def test_a_stopped_stage_is_not_usable(self, tmp_path):
        from tower.world_builder.dense import DenseParams
        from tower.world_builder.global_solve import load_solution

        store = _synthetic_world(tmp_path)
        dense = store.world_dir(WORLD) / "dense" / SESSION
        align = json.loads((dense / "align.json").read_text())
        align["stopped_after"] = 3
        assert not SP._depth_cache_usable(
            align, dense, load_solution(store, WORLD, SESSION), DenseParams())

    def test_a_cache_that_names_no_backend_is_not_usable(self, tmp_path):
        from tower.world_builder.dense import DenseParams
        from tower.world_builder.global_solve import load_solution

        store = _synthetic_world(tmp_path)
        dense = store.world_dir(WORLD) / "dense" / SESSION
        align = json.loads((dense / "align.json").read_text())
        align.pop("backend")
        assert not SP._depth_cache_usable(
            align, dense, load_solution(store, WORLD, SESSION), DenseParams())

    def test_densify_does_not_name_a_stopped_stage(self):
        import inspect

        from tower.world_builder import dense_pipeline

        body = inspect.getsource(dense_pipeline.densify)
        i = body.index('align["input_digest"] = digest')
        assert 'if align.get("stopped_after") is None:' in body[max(0, i - 400):i]

    def test_a_different_backend_is_not_already_built(self, tmp_path):
        import inspect

        body = inspect.getsource(SP.surfacify)
        assert "backend or DenseParams().backend" in body


class TestThePageIsStampedWithTheRevisionReadBeforeIt:
    """M3. Read after composition, a build landing in between stamped the OLD
    page with the NEW revision; the phone believed it had the final surface."""

    def test_a_build_landing_mid_composition_stamps_the_older_revision(self, tmp_path, monkeypatch):
        from tower.results import world_builder_render as R
        from tower.world_builder import surface_render

        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        before = R.build_render_revision(store, WORLD, SESSION)["revision"]
        real = surface_render.build_surface_page

        def page_then_rebuild(*a, **kw):
            html = real(*a, **kw)
            time.sleep(0.01)
            SP.surfacify(store, WORLD, SESSION, params=_params(min_weight=0.8), force=True)
            return html

        monkeypatch.setattr(surface_render, "build_surface_page", page_then_rebuild)
        html = R.build_world_render(store, WORLD, SESSION)
        after = R.build_render_revision(store, WORLD, SESSION)["revision"]
        assert after != before
        assert _meta(html, "wb-revision") == before, (
            "the page must carry the revision it was composed under, so the "
            "phone sees the newer one as newer")


class TestLocksSurvivePidReuse:
    """M4. A killed live child leaves its lock; if its pid is reused before the
    final build asks, a bare pid probe calls the lock live and the final build
    is refused."""

    def test_a_lock_older_than_the_process_holding_its_pid_is_stale(self, tmp_path):
        root = tmp_path / "surface" / "s1"
        root.mkdir(parents=True)
        lock_path = root / ".surface.lock"
        # our own pid is certainly running -- but it did not write a lock a
        # day before this process started
        lock_path.write_text(json.dumps({"pid": os.getpid(), "at": 0}))
        day_ago = time.time() - 86400
        os.utime(lock_path, (day_ago, day_ago))
        lock = SP._SurfaceLock(root)
        assert lock.acquire()
        lock.release()

    def test_a_running_status_older_than_its_pid_is_stale(self):
        assert SP.status_is_stale({"state": "running", "pid": os.getpid(),
                                   "updated_at": time.time() - 86400})
        assert not SP.status_is_stale({"state": "running", "pid": os.getpid(),
                                       "updated_at": time.time()})

    def test_a_fresh_lock_by_a_live_process_is_respected(self, tmp_path):
        root = tmp_path / "surface" / "s1"
        a, b = SP._SurfaceLock(root), SP._SurfaceLock(root)
        assert a.acquire()
        assert not b.acquire()
        a.release()


class TestTheFormatRefusesImpossibleIndices:

    def _buf(self, n_i_override=None, bad_index=False):
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        F = np.array([[0, 1, 2]], np.int64)
        buf = bytearray(S.write_mesh_bytes(V, F, None))
        if bad_index:
            buf[-2:] = (7).to_bytes(2, "little")
        if n_i_override is not None:
            buf[12:16] = n_i_override.to_bytes(4, "little")
        return bytes(buf)

    def test_an_index_count_that_is_not_triangles_is_refused(self):
        with pytest.raises(S.SurfaceUnavailable):
            S.read_mesh_bytes(self._buf(n_i_override=2)[:-2])

    def test_an_index_past_the_vertices_is_refused(self):
        with pytest.raises(S.SurfaceUnavailable):
            S.read_mesh_bytes(self._buf(bad_index=True))


class TestTheDiagnosticViewIsTheSparsePage:
    """iOS review: "open the solver's view" sends view=diagnostics and got the
    surface, and the app's text describing a diagnostics rendering was false."""

    def test_diagnostics_serves_sparse_even_with_a_surface(self, tmp_path):
        from tower.results.world_builder_render import (
            build_render_revision,
            build_world_render,
        )

        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        html = build_world_render(store, WORLD, SESSION, view="diagnostics")
        assert _declared(html) == "sparse"
        rev = build_render_revision(store, WORLD, SESSION, view="diagnostics")
        assert rev["representation"] == "sparse"
        assert _meta(html, "wb-revision") == rev["revision"]

    def test_an_explicit_representation_still_wins(self, tmp_path):
        from tower.results.world_builder_render import build_world_render

        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        html = build_world_render(store, WORLD, SESSION, view="diagnostics",
                                  representation="surface")
        assert _declared(html) == "surface"


class TestTheLiveWorkerOnlyBuildsWhatChanged:

    def _surfacer(self, tmp_path, spawned, code=None):
        from scripts.world_build_session import BackgroundSurface

        def spawn(argv, **kw):
            script = ("import time; time.sleep(120)" if code is None
                      else f"import sys; sys.exit({code})")
            proc = subprocess.Popen([sys.executable, "-c", script],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            spawned.append(proc)
            return proc

        return BackgroundSurface(root=tmp_path, world_id=WORLD, session_id=SESSION,
                                 spawn=spawn)

    def _solution(self, tmp_path, text):
        path = tmp_path / "worlds" / WORLD / "solve" / SESSION / "solution.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_a_solve_that_published_nothing_new_builds_nothing(self, tmp_path):
        spawned = []
        s = self._surfacer(tmp_path, spawned)
        try:
            self._solution(tmp_path, "{}")
            assert s.solve_landed(None)
            spawned[0].kill()
            spawned[0].wait(timeout=10)
            assert not s.solve_landed(None), "same solution file, nothing to rebuild"
            time.sleep(0.02)
            self._solution(tmp_path, '{"new": 1}')
            assert s.solve_landed(None)
        finally:
            s.close()

    def test_a_machine_without_the_depth_network_stops_trying(self, tmp_path):
        from scripts.world_build_session import SURFACE_EXIT_CANNOT_RUN_HERE

        spawned = []
        s = self._surfacer(tmp_path, spawned, code=SURFACE_EXIT_CANNOT_RUN_HERE)
        self._solution(tmp_path, "{}")
        assert s.solve_landed(None)
        spawned[0].wait(timeout=10)
        time.sleep(0.02)
        self._solution(tmp_path, '{"new": 1}')
        assert not s.solve_landed(None)
        assert len(spawned) == 1


def test_a_missing_depth_network_is_unavailable_and_permanent(tmp_path, monkeypatch):
    from tower.world_builder import dense_pipeline
    from tower.world_builder.dense import DenseUnavailable

    store = _synthetic_world(tmp_path)
    dense = store.world_dir(WORLD) / "dense" / SESSION
    align = json.loads((dense / "align.json").read_text())
    align["input_digest"] = "some-other-solve"
    (dense / "align.json").write_text(json.dumps(align))

    def no_network(*a, **kw):
        raise DenseUnavailable("depth backend 'moge2-vitl' is not installed")

    monkeypatch.setattr(dense_pipeline, "run_depth_stage", no_network)
    result = SP.surfacify(store, WORLD, SESSION, params=_params())
    assert result.state == SP.STATE_UNAVAILABLE
    assert result.permanent


def test_the_final_surface_runs_before_the_dense_stage_prunes_its_depth():
    import inspect

    import scripts.world_build_session as B

    src = inspect.getsource(B.main)
    assert src.index("surfacify(") < src.index("densify(")



def test_a_prediction_is_not_reused_once_its_keyframe_image_changes(tmp_path, monkeypatch):
    """Systems review: reuse was keyed on the keyframe id alone, so a keyframe
    re-redacted or replaced since kept its old pixels' depth and colour."""
    from tower.world_builder.dense import register_backend
    from tower.world_builder.global_solve import load_solution

    _CountingBackend.calls = 0
    register_backend(_CountingBackend.name, _CountingBackend)
    store = _synthetic_world(tmp_path, digest="final-solve")
    dense = store.world_dir(WORLD) / "dense" / SESSION
    align = json.loads((dense / "align.json").read_text())
    align["backend"] = _CountingBackend.name
    align["input_digest"] = "an-earlier-solve"
    (dense / "align.json").write_text(json.dumps(align))

    import cv2

    from tower.world_builder import global_solve as GS

    def identity_maps(_intrinsics, width, height):
        # The synthetic camera is already a pinhole; the real rectification
        # would trim a pixel of ROI, which is a different refusal's business.
        xs, ys = np.meshgrid(np.arange(width, dtype=np.float32),
                             np.arange(height, dtype=np.float32))
        return xs, ys, (0, 0, width, height), None

    monkeypatch.setattr(GS, "_undistort_maps", identity_maps)
    changed = store.images_dir(WORLD, SESSION) / f"{0:08d}.jpg"
    ok, enc = cv2.imencode(".jpg", np.full((120, 160, 3), 40, np.uint8))
    changed.write_bytes(enc.tobytes())

    solution = load_solution(store, WORLD, SESSION)
    SP.ensure_depth_stage(store, WORLD, SESSION, solution,
                          store.read_session(WORLD, SESSION).intrinsics,
                          gate_rel=0.08, backend=_CountingBackend.name)
    assert _CountingBackend.calls == 1, (
        "exactly the one keyframe whose image changed is predicted again")


# ---------------------------------------------------------------------------
# the fill mask does not depend on whether the engine or the stage redacted
# ---------------------------------------------------------------------------

_APPLIED = "faces-detected-and-filled/yunet-2023mar@0.30+plausibility1"


class _NoFaceRedactor:
    """Available, and finds nothing: `redact` hands back the input bytes under
    the applied label, exactly as `FaceRedactor._redact` does with no boxes."""

    available = True
    unavailable_reason = None
    label = _APPLIED

    def redact(self, data):
        from tower.world_builder.redaction import RedactionResult

        return RedactionResult(image_bytes=data, label=_APPLIED, regions=0)


class TestTheFillMaskSurvivesALiveBuild:
    """Review 2, I6. During a walk the session record says `none`, so a live
    build re-redacted every stored keyframe and took the fill mask as the
    difference against the STORED image -- empty wherever the engine had
    already filled a face, because both images carry the same black
    rectangle. After Stop the image hash still matched, and the final surface
    reused that empty mask: network depth invented across the fill rectangle
    was fused into the saved world. Measured on canonical keyframes, 48 of 100
    frames lost their fill this way."""

    def _world(self, tmp_path, monkeypatch, label, *, with_raw):
        import cv2
        import dataclasses

        from tower.world_builder import global_solve as GS
        from tower.world_builder import redaction as RD
        from tower.world_builder.dense import register_backend

        register_backend(_CountingBackend.name, _CountingBackend)
        monkeypatch.setattr(RD, "FaceRedactor", _NoFaceRedactor)

        def identity_maps(_intrinsics, width, height):
            xs, ys = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
            return xs, ys, (0, 0, width, height), None

        monkeypatch.setattr(GS, "_undistort_maps", identity_maps)
        store = _synthetic_world(tmp_path)
        kf = store.images_dir(WORLD, SESSION) / f"{0:08d}.jpg"
        img = cv2.imdecode(np.frombuffer(kf.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if with_raw:
            # The capture frame the engine redacted: textured where the face was.
            raw = img.copy()
            raw[30:90, 40:120] = np.random.default_rng(1).integers(
                60, 250, size=raw[30:90, 40:120].shape, dtype=np.uint8)
            raw_path = tmp_path / "raw-00000000.jpg"
            raw_path.write_bytes(cv2.imencode(".png", raw)[1].tobytes())
            ws = store.world_dir(WORLD) / "solve" / SESSION
            (ws / "sources.json").write_text(json.dumps(
                {"sources": {f"{SESSION}:{0:08d}": str(raw_path)}}))
        img[30:90, 40:120] = 0                          # the engine's fill
        kf.write_bytes(cv2.imencode(".png", img)[1].tobytes())
        s = store.read_session(WORLD, SESSION)
        store.write_session(dataclasses.replace(s, redaction=label))
        align_path = store.world_dir(WORLD) / "dense" / SESSION / "align.json"
        a = json.loads(align_path.read_text())
        a["backend"] = _CountingBackend.name
        align_path.write_text(json.dumps(a))
        return store, align_path

    def _build(self, store, align_path, solve):
        from tower.world_builder.global_solve import load_solution

        a = json.loads(align_path.read_text())
        a["input_digest"] = a["digest"] = solve        # another solve: refit
        align_path.write_text(json.dumps(a))
        _CountingBackend.calls = 0
        SP.ensure_depth_stage(store, WORLD, SESSION, load_solution(store, WORLD, SESSION),
                              store.read_session(WORLD, SESSION).intrinsics,
                              gate_rel=0.08, backend=_CountingBackend.name)
        fill = np.load(align_path.parent / "work" / "depth" / "00000_fill.npy")
        return float(fill.mean()), _CountingBackend.calls

    @pytest.mark.parametrize("with_raw", [False, True], ids=["no-raw-frame", "raw-frame"])
    def test_the_final_build_after_a_live_build_masks_the_engines_fill(
            self, tmp_path, monkeypatch, with_raw):
        import dataclasses

        store, align = self._world(tmp_path / "fresh", monkeypatch, _APPLIED,
                                   with_raw=with_raw)
        fresh, _ = self._build(store, align, "earlier")
        assert fresh > 0.2, "the rectangle is 25% of the frame"

        store, align = self._world(tmp_path / "walk", monkeypatch, "none",
                                   with_raw=with_raw)
        live, _ = self._build(store, align, "earlier")
        s = store.read_session(WORLD, SESSION)
        store.write_session(dataclasses.replace(s, redaction=_APPLIED))
        final, predicted = self._build(store, align, "the-final-solve")

        assert predicted == 0, "the live build's prediction is still reused"
        assert live == pytest.approx(fresh, abs=0.01), "the live build saw the fill"
        assert final == pytest.approx(fresh, abs=0.01), (
            "the saved world masks the same fill a fresh final build would")

    def test_a_prediction_made_under_an_earlier_fill_rule_is_not_offered(self, tmp_path):
        from tower.world_builder.dense_pipeline import reusable_predictions

        store = _synthetic_world(tmp_path)
        align_path = store.world_dir(WORLD) / "dense" / SESSION / "align.json"
        a = json.loads(align_path.read_text())
        backend = a["backend"]
        assert len(reusable_predictions(align_path, backend)) == len(a["records"])
        for r in a["records"]:
            r.pop("fill_rule")
        align_path.write_text(json.dumps(a))
        assert reusable_predictions(align_path, backend) == {}

    def test_a_depth_stage_made_under_an_earlier_fill_rule_is_not_a_cache(self, tmp_path):
        from tower.world_builder.dense import DenseParams
        from tower.world_builder.dense_pipeline import FILL_RULE, _depth_cache_key
        from tower.world_builder.global_solve import load_solution

        store = _synthetic_world(tmp_path)
        dense = store.world_dir(WORLD) / "dense" / SESSION
        a = json.loads((dense / "align.json").read_text())
        solution = load_solution(store, WORLD, SESSION)
        assert SP._depth_cache_usable(a, dense, solution, DenseParams())
        a.pop("fill_rule")
        assert not SP._depth_cache_usable(a, dense, solution, DenseParams())
        # densify's own cache test is the key
        assert f"fill{FILL_RULE}" in _depth_cache_key("d", DenseParams())


# ---------------------------------------------------------------------------
# review 2: the prune grace, orphans, the budget's last attempt, the manifest
# ---------------------------------------------------------------------------


class TestAReaderOfThePreviousManifestKeepsItsFiles:
    """Review 2, I1. The grace was measured from each level's WRITE time, so a
    previous build older than two minutes lost its levels the instant the next
    manifest landed, and a reader between the manifest and the level got
    `SurfaceUnavailable` -- a downgraded page, and on the phone an offer to
    replace a surface with points."""

    def test_a_reader_holding_the_previous_manifest_can_read_its_level(self, tmp_path):
        store = _synthetic_world(tmp_path)
        assert SP.surfacify(store, WORLD, SESSION, params=_params()).state == SP.STATE_OK
        root = SP.surface_dir(store, WORLD, SESSION)
        held = _manifest(store)
        past = time.time() - 600          # an ordinary gap before the final build
        for lv in held["levels"]:
            os.utime(root / lv["file"], (past, past))

        second = SP.surfacify(store, WORLD, SESSION, force=True,
                              params=_params(min_weight=0.8))
        assert second.state == SP.STATE_OK
        assert _manifest(store)["built_at"] != held["built_at"]
        assert SP.read_surface_level(store, WORLD, SESSION, 0, manifest=held)

    def test_superseded_levels_go_once_the_grace_after_supersession_has_passed(self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        root = SP.surface_dir(store, WORLD, SESSION)
        first = _manifest(store)
        SP.surfacify(store, WORLD, SESSION, force=True, params=_params(min_weight=0.8))
        old = [root / lv["file"] for lv in first["levels"]]
        assert all(p.exists() for p in old)
        past = time.time() - SP.PRUNE_GRACE_S - 5    # superseded long enough ago
        for p in old:
            os.utime(p, (past, past))
        SP.surfacify(store, WORLD, SESSION, force=True, params=_params(min_weight=0.7))
        assert not any(p.exists() for p in old)


class TestNothingUnnamedIsLeftBehind:
    """Review 2, I7. A stop between levels returned without removing the
    levels it had written, and the prune ran only after a later SUCCESSFUL
    publish -- which a final build that was stopped never gets. Staging files
    of a killed write were never pruned at all."""

    def test_a_stop_during_pack_removes_the_levels_it_wrote(self, tmp_path):
        import unittest.mock as mock

        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        before = _manifest(store)
        root = SP.surface_dir(store, WORLD, SESSION)
        wrote = []
        real = SP.write_bytes_atomic

        def spy(path, write):
            real(path, write)
            wrote.append(path)

        def stop_after_the_first_level():
            try:
                in_pack = json.loads((root / "status.json").read_text()).get("stage") == "pack"
            except (OSError, ValueError):
                return False
            return in_pack and bool(wrote)

        with mock.patch.object(SP, "write_bytes_atomic", spy):
            result = SP.surfacify(store, WORLD, SESSION, force=True,
                                  params=_params(min_weight=0.8),
                                  should_stop=stop_after_the_first_level)
        assert result.state == SP.STATE_STOPPED
        assert wrote, "fixture: the stop came before any level was written"
        assert not any(p.exists() for p in wrote)
        named = {lv["file"] for lv in before["levels"]}
        assert {p.name for p in root.glob("mesh_l*.bin")} == named

    def test_a_killed_packs_files_are_swept_by_the_next_run_even_one_that_builds_nothing(
            self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        root = SP.surface_dir(store, WORLD, SESSION)
        named = {lv["file"] for lv in _manifest(store)["levels"]}
        orphan = root / "mesh_l0.deadbeef.bin"
        staging = root / "mesh_l1.deadbeef.bin.p1234.abcdef01.tmp"
        fresh_staging = root / "mesh_l1.cafe.bin.p1234.abcdef02.tmp"
        for p in (orphan, staging, fresh_staging):
            p.write_bytes(b"torn")
        old = time.time() - SP.STAGING_GRACE_S - 5
        for p in (orphan, staging):
            os.utime(p, (old, old))

        again = SP.surfacify(store, WORLD, SESSION, params=_params())   # already built
        assert again.state == SP.STATE_OK
        assert not orphan.exists() and not staging.exists()
        assert fresh_staging.exists(), "a staging file young enough to be in flight stays"
        assert all((root / n).exists() for n in named)

    def test_an_unreadable_manifest_sweeps_nothing(self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        root = SP.surface_dir(store, WORLD, SESSION)
        files = list(root.glob("mesh_l*.bin"))
        old = time.time() - 3600
        for p in files:
            os.utime(p, (old, old))
        (root / "manifest.json").write_text("{torn", encoding="utf-8")
        SP._sweep_unnamed_levels(root)
        assert all(p.exists() for p in files)


class TestTheBudgetIsABudget:
    """Review 2, S7. On its last attempt the coarsening loop reserved whatever
    the pass computed, over budget, silently -- the out-of-memory failure the
    budget exists to prevent."""

    def test_a_walk_the_budget_cannot_hold_is_refused_by_name(self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        before = _manifest(store)
        result = SP.surfacify(store, WORLD, SESSION, force=True,
                              params=_params(max_blocks=1))
        assert result.state == SP.STATE_UNAVAILABLE
        assert "block budget of 1" in result.detail
        assert _manifest(store) == before, "the previous surface stands"


class TestTheManifestSaysWhatWasUsed:

    def test_truncation_is_the_band_used_cap_included(self, tmp_path):
        """Review 2, S6.2: `truncation_for` is uncapped, and the manifest
        recorded it even when `trunc_max_voxels` cut every sample's band."""
        store = _synthetic_world(tmp_path)
        params = _params(trunc_voxels=1.0, trunc_max_voxels=2.0, trunc_error_multiple=40.0)
        assert SP.surfacify(store, WORLD, SESSION, params=params).state == SP.STATE_OK
        man = _manifest(store)
        uncapped = S.truncation_for(params, man["voxel"], man["median_scene_depth"], 0.01)
        assert uncapped > 2.0 * man["voxel"] * 1.5, "fixture: the cap binds"
        assert man["truncation"] == pytest.approx(2.0 * man["voxel"])

    def test_a_build_the_evidence_filter_emptied_says_so(self, tmp_path):
        """Review 2, S6.3. Every face removed by the filter was reported as
        "the fused field held no cell with enough evidence"."""
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        before = _manifest(store)
        result = SP.surfacify(store, WORLD, SESSION, force=True,
                              params=_params(min_support_frames=2))
        assert result.state == SP.STATE_UNAVAILABLE
        assert "evidence filter removed all" in result.detail, result.detail
        assert "fewer than 2 distinct frames" in result.detail
        assert _manifest(store)["built_at"] == before["built_at"]


class TestTheCaptionPromisesOnlyWhatTheArtifactDid:
    """Review 2, S6.5. The page says "at least two camera views" for every
    surface, including one built before the per-face filter existed."""

    def test_a_filtered_build_says_so_in_its_manifest_and_its_page(self, tmp_path):
        from tower.world_builder.surface_render import build_surface_payload

        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        man = _manifest(store)
        assert man["detail"]["evidence_filter"]["faces_in"] > 0
        _raw, config = build_surface_payload(store, WORLD, SESSION)
        assert config["evidence_filter"] is True

    def test_a_surface_built_before_the_filter_does_not_claim_two_views(self, tmp_path):
        from tower.world_builder.surface_render import build_surface_payload

        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        path = SP.surface_dir(store, WORLD, SESSION) / "manifest.json"
        man = json.loads(path.read_text())
        man.pop("detail")
        for key in ("min_support_frames", "contradiction_ratio", "drop_back_facing"):
            man["params"].pop(key)
        path.write_text(json.dumps(man))
        _raw, config = build_surface_payload(store, WORLD, SESSION)
        assert config["evidence_filter"] is False

    def test_the_template_keys_the_two_view_sentence_on_it(self):
        from tower.world_builder.surface_render import viewer_template_path

        html = viewer_template_path().read_text(encoding="utf-8")
        at = html.index("at least two camera views")
        assert "CONFIG.evidence_filter" in html[at - 300:at]


def test_an_unreadable_surface_manifest_is_never_a_surface_none_revision(tmp_path, monkeypatch):
    """Review 2, iOS m5(c). The revision read the manifest with a bare
    `read_text`; on Windows a read racing the replace of a landing build
    raised, and the route answered `surface:None` -- which the phone took for
    a new build of the same rung, and when the page's own stamp hit the same
    race, swapped to an identical mesh and back."""
    from tower.results.world_builder_render import build_render_revision, build_world_render
    from pathlib import Path

    from tower.world_builder import store as store_module

    store = _synthetic_world(tmp_path)
    SP.surfacify(store, WORLD, SESSION, params=_params())
    good = build_render_revision(store, WORLD, SESSION)
    assert good["representation"] == "surface"

    # what a read that lost every retry to a replace returns
    real = store_module._read_json_past_a_replace
    monkeypatch.setattr(
        store_module, "_read_json_past_a_replace",
        lambda path, **kw: (None if Path(path).parent.parent.name == "surface"
                         else real(path, **kw)))
    rev = build_render_revision(store, WORLD, SESSION)
    assert "None" not in rev["revision"]
    assert rev["representation"] != "surface", "answered as absent: the next rung"
    page = build_world_render(store, WORLD, SESSION)
    assert _meta(page, "wb-representation") == "surface"
    assert _meta(page, "wb-revision") is None, "no stamp rather than a false one"


class TestThePhoneBudgetBoundsThePage:
    """Live replay D: the budget was applied to the mesh bytes, the page inlines
    the mesh base64-encoded, and a 5.88 MB level shipped as a 7.89 MB page
    against a 6 MiB budget."""

    def test_a_level_is_chosen_by_the_page_it_makes(self):
        from tower.world_builder.surface_render import (
            MOBILE_BYTE_BUDGET,
            choose_level,
            page_bytes_for_mesh,
        )

        mib = 1024 * 1024
        manifest = {"levels": [{"level": 0, "bytes": 44 * mib},
                               {"level": 1, "bytes": int(5.6 * mib)},
                               {"level": 2, "bytes": int(4.2 * mib)}]}
        assert manifest["levels"][1]["bytes"] < MOBILE_BYTE_BUDGET, "fixture: the mesh fits"
        assert page_bytes_for_mesh(manifest["levels"][1]["bytes"]) > MOBILE_BYTE_BUDGET
        assert choose_level(manifest, MOBILE_BYTE_BUDGET)["level"] == 2

    def test_the_pack_stage_makes_the_phone_level_fit_its_page(self, tmp_path):
        from tower.world_builder.surface_render import (
            build_surface_page,
            mesh_bytes_for_page,
            page_overhead_bytes,
        )

        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params(mobile_page_bytes=0))
        free = _manifest(store)["levels"][1]
        assert free["faces"] > 400, "fixture: a phone level worth shrinking"
        budget = page_overhead_bytes() + (4 * free["bytes"] // 3) // 2

        result = SP.surfacify(store, WORLD, SESSION, force=True,
                              params=_params(mobile_page_bytes=budget))
        assert result.state == SP.STATE_OK
        man = _manifest(store)
        phone = man["levels"][man["mobile_level"]]
        fit = man["detail"]["mobile_page_fit"]
        assert fit["fits"] and fit["decimations"] >= 1
        assert phone["bytes"] <= mesh_bytes_for_page(budget)
        assert phone["faces"] < free["faces"]
        assert phone["faces"] > 0.3 * free["faces"], "the largest that fits, not a token mesh"
        page = build_surface_page(store, WORLD, SESSION, budget_bytes=budget)
        assert len(page.encode("utf-8")) <= budget

    def test_the_phone_page_budget_is_part_of_what_was_built(self):
        a = S.SurfaceParams().digest_fields()
        b = S.SurfaceParams(mobile_page_bytes=1).digest_fields()
        assert a != b
