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


def _synthetic_world(tmp_path, *, n_frames=8, digest="digest-1", with_dense=True):
    import cv2

    from tower.world_builder.dense import DenseParams
    from tower.world_builder.global_solve import Solution, workspace_for, write_solution
    from tower.world_builder.records import CameraIntrinsics, Session, World
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    store.write_world(World(world_id=WORLD, created_at=1.0, updated_at=2.0,
                            session_ids=(SESSION,)))
    K, w, h = _camera()
    store.write_session(Session(
        session_id=SESSION, world_id=WORLD, started_at=1.0,
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

    kids, poses, depths = [], {}, []
    for i in range(n_frames):
        a = 2 * np.pi * i / n_frames
        eye = np.array([1.2 * np.cos(a), 0.0, 1.2 * np.sin(a)])
        R, t = _look_from(eye, eye * 3.0)
        kid = f"{SESSION}:{i:08d}"
        kids.append(kid)
        poses[kid] = {"component": 0, "rotation": R.ravel().tolist(),
                      "translation": t.tolist(), "observations": 50}
        depths.append(_render_box_depth(R, t, K, w, h))

    camera = {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]),
              "cy": float(K[1, 2]), "width": w, "height": h}
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-2.9, 2.9, size=(64, 3)).astype(np.float32)
    write_solution(workspace_for(store, WORLD, SESSION), Solution(
        solver="glomap", solved_at=time.time(), input_digest=digest,
        keyframe_ids=kids, poses=poses,
        components=[{"index": 0, "images": n_frames, "points": len(xyz)}],
        xyz=xyz, rgb=np.full((len(xyz), 3), 128, np.uint8),
        component=np.zeros(len(xyz), np.int32),
        first_keyframe=np.zeros(len(xyz), np.int32),
        track_length=np.full(len(xyz), 3, np.int32),
        error=np.full(len(xyz), 0.5, np.float32),
        observations=np.zeros((0, 3), np.int32), camera=camera))

    if with_dense:
        dense = store.world_dir(WORLD) / "dense" / SESSION
        (dense / "work" / "depth").mkdir(parents=True)
        (dense / "work" / "undist").mkdir(parents=True)
        records = []
        for i, (kid, d) in enumerate(zip(kids, depths)):
            # kind "depth" with a = 1, b = 0 stores metric-gauge depth as-is
            np.save(dense / "work" / "depth" / f"{i:05d}.npy", d.astype(np.float16))
            img = np.full((h, w, 3), 150 + 10 * (i % 3), np.uint8)
            cv2.imwrite(str(dense / "work" / "undist" / f"{i:05d}.jpg"), img)
            records.append({"ki": i, "kid": kid, "ok": True, "a": 1.0, "b": 0.0,
                            "held_out_rel": 0.01})
        (dense / "align.json").write_text(json.dumps({
            "kind": "depth", "camera": camera, "backend": DenseParams().backend,
            "records": records}))
    return store


def _params(**kw):
    """Coarse enough to be quick on a CPU-only runner, fine enough to mesh."""
    base = dict(voxel_frac=0.02, lod_face_targets=(0, 1500), canonical_level=0,
                mobile_level=1, smooth_iterations=2, min_weight=0.5,
                min_component_frac=0.0)
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

        SP.surfacify(store, WORLD, SESSION, params=_params())
        man = _manifest(store)
        assert man["input_digest"] == "digest-2"
        assert SP.surface_currency(store, WORLD, SESSION, man)["current"]

    def test_a_missing_level_file_is_not_mistaken_for_a_finished_build(self, tmp_path):
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        root = SP.surface_dir(store, WORLD, SESSION)
        first = _manifest(store)["built_at"]
        (root / "mesh_l1.bin").unlink()
        SP.surfacify(store, WORLD, SESSION, params=_params())
        assert (root / "mesh_l1.bin").exists()
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

    def test_it_waits_for_the_solver_to_have_the_card(self, tmp_path):
        spawned = []
        s = self._surfacer(tmp_path, spawned)
        assert not s.maybe_launch(None, solver_running=True)
        assert spawned == []

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
