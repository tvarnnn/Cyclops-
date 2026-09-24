"""The gate's depth predictions are made once and kept (review V8 H2, part R3).

The evidence gate's metric scale reads the depth stage's raw predictions, and a re-finish
used to predict every frame again: the room's final surface prunes `work/`, and the gate's
hand-off cuts `align.json` to the room. With `run_depth_stage(prediction_cache=True)` --
which only the gate's depth stage asks for (`coherence_publish.run_gate_depth`) -- the
network's output is kept under `<dense>/predictions/<token>/`, keyed by the exact pixels it
was shown and by the network and its parameters. Pinned here:

  * a second build of the same frames calls the network for NONE of them, even after the
    prune, and its metric-scale inputs (`<ki>_pred.npy`) are byte-identical;
  * the alignment to the solve is recomputed (a different solve, a different fit), and a
    frame fitted from the cache is fitted exactly as the build that predicted it;
  * other pixels or another network parameter is a miss; a torn cache file is a miss;
  * off (every other caller), the stage writes exactly what it did.
"""

from __future__ import annotations

import dataclasses as dc
import json

import cv2
import numpy as np
import pytest

import tower.world_builder.dense_pipeline as DP
from tests.test_world_builder_surface_pipeline import SESSION, WORLD, _synthetic_world
from tower.world_builder import coherence_publish as CP
from tower.world_builder import global_solve as GS


class _Network:
    """A depth network that depends only on its input, and counts its calls."""

    name = "moge2-vitl"
    model_id = "Ruicheng/moge-2-vitl"
    resolution_level = 9
    licence = "test"
    windowed = False
    window_size = 0
    kind = "depth"
    accepts_fov = True

    def __init__(self):
        self.calls = 0

    def predict(self, rgb, fov_x=None):
        self.calls += 1
        base = 1.0 + float(rgb.mean()) / 97.0 + (0.0 if fov_x is None else fov_x / 1000.0)
        yy, xx = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
        return (base + 0.0137 * xx + 0.0071 * yy).astype(np.float32)   # not float16-exact


@pytest.fixture
def world(tmp_path, monkeypatch):
    store = _synthetic_world(tmp_path, digest="final-solve", anchors=True)
    dense = store.world_dir(WORLD) / "dense" / SESSION
    (dense / "align.json").unlink()                 # no earlier depth stage
    # the anchors' pixel positions, as a solve records them, so every frame can be fitted
    sol = GS.load_solution(store, WORLD, SESSION)
    cam = sol.camera
    xy = []
    for ki, _feature, pi in sol.observations:
        pose = sol.poses[sol.keyframe_ids[int(ki)]]
        R = np.asarray(pose["rotation"], float).reshape(3, 3)
        Xc = R @ sol.xyz[int(pi)].astype(float) + np.asarray(pose["translation"], float)
        xy.append([cam["fx"] * Xc[0] / Xc[2] + cam["cx"], cam["fy"] * Xc[1] / Xc[2] + cam["cy"]])
    GS.write_solution(GS.workspace_for(store, WORLD, SESSION),
                      dc.replace(sol, observation_xy=np.asarray(xy, np.float32).reshape(-1, 2)))
    # distinct keyframe pixels, so every frame is its own network input
    rng = np.random.default_rng(3)
    images = store.images_dir(WORLD, SESSION)
    for p in sorted(images.glob("*.jpg")):
        h, w = cv2.imread(str(p)).shape[:2]
        ok, enc = cv2.imencode(".jpg", rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        p.write_bytes(enc.tobytes())
    net = _Network()
    monkeypatch.setattr(DP, "make_backend", lambda name: net)

    def identity_maps(_intrinsics, width, height):
        xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        return xs, ys, (0, 0, width, height), None

    monkeypatch.setattr(GS, "_undistort_maps", identity_maps)
    return store, net, dense


def _gate_depth(store, solution=None):
    sol = solution or GS.load_solution(store, WORLD, SESSION)
    return CP.run_gate_depth(store, WORLD, SESSION, sol, store.read_session(WORLD, SESSION).intrinsics)


def _preds(work):
    return {p.name: p.read_bytes() for p in sorted((work / "depth").glob("*_pred.npy"))}


def _fits(align):
    return {r["ki"]: (r.get("a"), r.get("b")) for r in align["records"] if r.get("ok")}


def test_a_second_gated_build_predicts_nothing_even_after_the_prune(world):
    store, net, dense = world
    first, work, _ = _gate_depth(store)
    assert net.calls == 8
    assert first["prediction_cache"]["predicted"] == 8 and first["prediction_cache"]["hits"] == 0
    kept = sorted((dense / DP.PREDICTIONS_DIRNAME).rglob("*.npy"))
    assert len(kept) == 8
    preds = _preds(work)
    assert len(preds) == 8
    # the room's final surface prunes the work; the kept predictions are outside it
    DP.prune_intermediates(dense)
    assert not (dense / "work").exists() and len(sorted((dense / DP.PREDICTIONS_DIRNAME).rglob("*.npy"))) == 8
    second, work2, _ = _gate_depth(store)
    assert net.calls == 8, "the network was not called again"
    assert second["prediction_cache"]["hits"] == 8 and second["prediction_cache"]["predicted"] == 0
    assert _preds(work2) == preds, "the metric scale reads byte-identical predictions"
    assert _fits(second) == _fits(first) and len(_fits(first)) == 8, \
        "a frame fitted from the cache is fitted exactly as the build that predicted it"


def test_the_fit_is_to_this_solve(world):
    """The prediction is kept; the alignment is not: another solve of the same frames
    (its poses scaled x2) gets its own fit, the one a fresh prediction would give it."""
    store, net, dense = world
    original = _fits(_gate_depth(store)[0])
    sol = GS.load_solution(store, WORLD, SESSION)
    scaled = dc.replace(sol, xyz=sol.xyz * 2.0, poses={
        k: dict(p, translation=[2.0 * v for v in p["translation"]]) for k, p in sol.poses.items()})
    from_cache = _fits(_gate_depth(store, scaled)[0])
    assert net.calls == 8
    assert from_cache != original, "a fit is the solve's own"
    # a fresh stage for the scaled solve, in a world with no cache
    for p in (dense / DP.PREDICTIONS_DIRNAME).rglob("*.npy"):
        p.rename(p.with_suffix(".set-aside"))
    fresh = _fits(_gate_depth(store, scaled)[0])
    assert net.calls == 16
    assert from_cache == fresh and len(fresh) == 8


def test_other_pixels_or_another_network_parameter_is_a_miss(world, monkeypatch):
    store, net, dense = world
    _gate_depth(store)
    # one keyframe's pixels changed: that frame alone is predicted again
    images = store.images_dir(WORLD, SESSION)
    first = sorted(images.glob("*.jpg"))[0]
    img = cv2.imread(str(first))
    img[:4, :4] = 0
    first.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    calls = net.calls
    second, _, _ = _gate_depth(store)
    assert net.calls == calls + 1
    assert second["prediction_cache"] == dict(second["prediction_cache"], hits=7, predicted=1)
    # another field of view (the solve's camera) is another token
    sol = GS.load_solution(store, WORLD, SESSION)
    wider = dc.replace(sol, camera=dict(sol.camera, fx=sol.camera["fx"] * 0.8,
                                        fy=sol.camera["fy"] * 0.8))
    again, _, _ = _gate_depth(store, wider)
    assert again["prediction_cache"]["predicted"] == 8
    # One set is kept (review V9, M-12): the other token's was pruned once this stage completed.
    assert [d.name for d in (dense / DP.PREDICTIONS_DIRNAME).iterdir()] == [again["prediction_cache"]["token"]]
    assert again["prediction_cache"]["pruned"]["sets"] == 1


def test_a_torn_cache_file_is_a_miss(world):
    store, net, dense = world
    _gate_depth(store)
    victim = sorted((dense / DP.PREDICTIONS_DIRNAME).rglob("*.npy"))[0]
    victim.write_bytes(b"torn")
    again, _, _ = _gate_depth(store)
    assert net.calls == 9 and again["prediction_cache"]["predicted"] == 1
    assert np.load(victim).dtype == np.float16, "and it is written again, whole"


def test_the_token_names_the_network_and_its_call(monkeypatch):
    runtime = {"weights_revision": "39c4d5e957afe587e04eec59dc2bcc3be5ecd968", "moge": "3.0.0",
               "torch": "2.13.0+cu132", "device": "cuda", "fp16": True}
    monkeypatch.setattr(DP, "token_runtime", lambda backend: dict(runtime))
    net = _Network()
    t1, doc = DP.prediction_token(net, 42.0)
    # Schema 2 (review V9, LOW): what the prediction was made WITH, beyond the network's name.
    assert doc == {"schema": 2, "backend": "moge2-vitl", "model_id": "Ruicheng/moge-2-vitl",
                   "kind": "depth", "resolution_level": 9, "fov_x": 42.0, **runtime}
    assert DP.prediction_token(net, 42.0000001)[0] == t1
    assert DP.prediction_token(net, 42.1)[0] != t1 and DP.prediction_token(net, None)[0] != t1
    net.resolution_level = 8
    assert DP.prediction_token(net, 42.0)[0] != t1
    a = np.zeros((4, 5, 3), np.uint8)
    assert DP.network_input_sha1(a) != DP.network_input_sha1(a.reshape(5, 4, 3))
    assert DP.network_input_sha1(a) == DP.network_input_sha1(a.copy())


def test_a_kept_prediction_stays_clear_of_max_path(world):
    """A 263-character cache name failed on a scratch copy of a real world (FileNotFoundError, which the gate
    took as no depth). Beneath `dense/<session>/` the name -- with the atomic write's staging suffix -- stays
    short enough for a world root under any Tower's data directory."""
    from tower.storage import staging_path

    store, net, dense = world
    _gate_depth(store)
    (kept,) = sorted((dense / DP.PREDICTIONS_DIRNAME).rglob("*.npy"))[:1]
    relative = staging_path(kept).relative_to(dense)
    assert len(str(relative)) <= 72, relative
    # a Tower's world root (data\\world_builder) is ~62 characters deep; a world and session add 72 more
    assert 62 + 72 + len(str(relative)) < 240


def test_off_the_depth_stage_is_todays(world):
    """Every caller but the gate: no cache is read or written, no key is added, and the
    fit uses the network's own float32 output, as it always did."""
    store, net, dense = world
    sol = GS.load_solution(store, WORLD, SESSION)
    align = DP.run_depth_stage(store, WORLD, SESSION, sol, store.read_session(WORLD, SESSION).intrinsics,
                               CP._depth_params(), dense)
    assert net.calls == 8
    assert "prediction_cache" not in align
    assert "prediction_cache" not in json.loads((dense / "align.json").read_text())
    assert not (dense / DP.PREDICTIONS_DIRNAME).exists()
    # What the metric scale reads is the same either way: the float16 prediction.
    off_preds = _preds(dense / "work")
    CP.run_gate_depth(store, WORLD, SESSION, sol, store.read_session(WORLD, SESSION).intrinsics)
    assert _preds(dense / "work") == off_preds and len(off_preds) == 8


def test_the_gate_records_where_its_predictions_came_from(world, monkeypatch):
    store, net, dense = world
    monkeypatch.setattr(CP, "measure_metric_scale", lambda *a, **k: {"metric_log": {}})
    monkeypatch.setattr(CP.CG, "read_verified_links", lambda db, min_inliers=15: {})
    monkeypatch.setattr(CP.CG, "read_link_rotations", lambda db, cam, min_inliers=15: {})
    from tower.world_builder.records import Keyframe

    sol = dc.replace(GS.load_solution(store, WORLD, SESSION), transients={"state": "applied"})
    kfs = [Keyframe(keyframe_id=kid, session_id=SESSION, source_seq=i, received_at=float(i),
                    image_relpath=f"images/{kid.rsplit(':', 1)[-1]}.jpg", width=1, height=1,
                    byte_count=1) for i, kid in enumerate(sol.keyframe_ids)]
    first = CP.gate_final_solution(store, WORLD, SESSION, sol, database_path="db", keyframes=kfs)
    second = CP.gate_final_solution(store, WORLD, SESSION, sol, database_path="db", keyframes=kfs)
    p1, p2 = first.record["depth"]["predictions"], second.record["depth"]["predictions"]
    assert (p1["cached"], p1["predicted"]) == (0, 8) and (p2["cached"], p2["predicted"]) == (8, 0)
    assert p1["token"] == p2["token"]
