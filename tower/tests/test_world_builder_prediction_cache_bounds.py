"""Review V9: the gate's depth-prediction cache never costs the depth, and it is bounded.

Module: `tower/world_builder/dense_pipeline.py` (`run_depth_stage(prediction_cache=True)`,
which only the evidence gate's depth stage asks for). RV9-C's probes, RUN
`baseline/review/V9/predcache/test_rv9c_probes.py`, are the red cases here:

  * M-11: a cache write that fails (a full disk, MAX_PATH, a sharing violation) raised out
    of the depth stage, and the gate published a scale-unavailable fail-safe (P6);
  * LOW: an empty cache file raised `EOFError` for ever (P1, P1b); a wrong-shape array was
    used (P2) or failed the stage for ever (P2b); the token did not see the device, fp16,
    the weights revision or the library versions (P3);
  * M-12: nothing ever removed a set: another token's, a superseded frame's, a dead
    writer's staging file;
  * Q8: a densify, holding only its own lock, could write `dense/<session>/` while the
    gate's depth stage (holding the session's surface lock) was writing it.
"""

from __future__ import annotations

import dataclasses as dc
import errno
import json
import logging
import os

import cv2
import numpy as np
import pytest

import tower.storage as ST
import tower.world_builder.dense_pipeline as DP
from tests.test_world_builder_depth_prediction_cache import (  # noqa: F401 -- fixture
    _fits,
    _gate_depth,
    _preds,
    world,
)
from tests.test_world_builder_surface_pipeline import SESSION, WORLD
from tower.world_builder import coherence_publish as CP
from tower.world_builder import global_solve as GS
from tower.world_builder.dense import STAGE_DEPTH, MoGeBackend

RUNTIME = {"weights_revision": "39c4d5e957afe587e04eec59dc2bcc3be5ecd968", "moge": "3.0.0",
           "torch": "2.13.0+cu132", "device": "cuda", "fp16": True}


def _kept(dense):
    return sorted((dense / DP.PREDICTIONS_DIRNAME).rglob("*.npy"))


# ---------------------------------------------------------------------------
# M-11: a failed cache write never throws the depth stage away
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("error", [
    OSError(errno.ENOSPC, "No space left on device"),
    PermissionError(13, "The process cannot access the file because it is being used by another process"),
    FileNotFoundError(errno.ENOENT, "No such file or directory"),        # MAX_PATH's face on Windows
])
def test_a_cache_write_that_fails_keeps_the_depth(world, monkeypatch, error):
    """RV9-C P6. At 6d4b567 the first failed write raised; `run_gate_depth` turned it into
    `DepthUnavailable` and the network's good predictions were discarded."""
    store, net, dense = world
    reference, _, _ = _gate_depth(store)                    # a working cache, for comparison
    ref_preds = _preds(dense / "work")
    for p in _kept(dense):
        p.unlink()

    def fails(path, write):
        raise error

    monkeypatch.setattr(ST, "write_bytes_atomic", fails)
    align, work, _ = _gate_depth(store)
    cache = align["prediction_cache"]
    assert cache["predicted"] == 8 and cache["hits"] == 0 and cache["write_failed"] == 8
    assert _kept(dense) == [], "nothing was kept"
    assert _preds(work) == ref_preds, "the stage went on with the float16-rounded prediction"
    assert _fits(align) == _fits(reference) and len(_fits(align)) == 8


def test_through_the_gate_a_failed_write_is_not_a_scale_fail_safe(world, monkeypatch):
    store, net, dense = world
    monkeypatch.setattr(CP, "measure_metric_scale", lambda *a, **k: {"metric_log": {}})
    monkeypatch.setattr(CP.CG, "read_verified_links", lambda db, min_inliers=15: {})
    monkeypatch.setattr(CP.CG, "read_link_rotations", lambda db, cam, min_inliers=15: {})
    monkeypatch.setattr(ST, "write_bytes_atomic",
                        lambda path, write: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")))
    from tower.world_builder.records import Keyframe

    sol = dc.replace(GS.load_solution(store, WORLD, SESSION), transients={"state": "applied"})
    kfs = [Keyframe(keyframe_id=kid, session_id=SESSION, source_seq=i, received_at=float(i),
                    image_relpath=f"images/{kid.rsplit(':', 1)[-1]}.jpg", width=1, height=1,
                    byte_count=1) for i, kid in enumerate(sol.keyframe_ids)]
    r = CP.gate_final_solution(store, WORLD, SESSION, sol, database_path="db", keyframes=kfs)
    assert r.record["depth"]["state"] == CP.DEPTH_OK, r.record["depth"]
    assert r.record["depth"]["predictions"]["predicted"] == 8


# ---------------------------------------------------------------------------
# LOW: a bad cache file is a logged miss, and the next write replaces it
# ---------------------------------------------------------------------------


def _damage(victim, how):
    good = np.load(victim)
    h, w = good.shape
    if how == "empty":
        victim.write_bytes(b"")
    elif how == "truncated":
        data = victim.read_bytes()
        victim.write_bytes(data[: len(data) // 2])
    elif how == "larger":
        np.save(victim, np.full((h + 7, w + 9), 3.0, np.float16))
    elif how == "smaller":
        np.save(victim, np.full((2, 2), 3.0, np.float16))
    elif how == "float32":
        np.save(victim, good.astype(np.float32))
    elif how == "archive":
        with open(victim, "wb") as handle:
            np.savez(handle, a=good)
    return good


@pytest.mark.parametrize("how", ["empty", "truncated", "larger", "smaller", "float32", "archive"])
def test_a_bad_cache_file_is_a_miss_and_is_rewritten(world, caplog, how):
    """RV9-C P1 (empty: `EOFError` for ever), P2 (larger: silently used), P2b (smaller: no
    depth for ever); a float32 or an archive under the name is foreign."""
    store, net, dense = world
    first, _, _ = _gate_depth(store)
    victim = _kept(dense)[0]
    good = _damage(victim, how)
    calls = net.calls
    with caplog.at_level(logging.WARNING, logger=DP.logger.name):
        again, _, _ = _gate_depth(store)
    assert net.calls == calls + 1, "that frame alone was predicted again"
    assert again["prediction_cache"]["hits"] == 7 and again["prediction_cache"]["predicted"] == 1
    rewritten = np.load(victim)
    assert rewritten.dtype == np.float16 and np.array_equal(rewritten, good), "rewritten whole"
    assert _fits(again) == _fits(first)
    if how != "truncated":          # a torn .npy was a quiet miss before; every other is logged now
        assert any("unusable" in r.getMessage() for r in caplog.records)


def test_an_empty_cache_file_through_the_gate_is_not_a_retryable_fail_safe(world, monkeypatch):
    """RV9-C P1b: at 6d4b567 the gate recorded depth unavailable, retryable, every time."""
    store, net, dense = world
    monkeypatch.setattr(CP, "measure_metric_scale", lambda *a, **k: {"metric_log": {}})
    monkeypatch.setattr(CP.CG, "read_verified_links", lambda db, min_inliers=15: {})
    monkeypatch.setattr(CP.CG, "read_link_rotations", lambda db, cam, min_inliers=15: {})
    from tower.world_builder.records import Keyframe

    sol = dc.replace(GS.load_solution(store, WORLD, SESSION), transients={"state": "applied"})
    kfs = [Keyframe(keyframe_id=kid, session_id=SESSION, source_seq=i, received_at=float(i),
                    image_relpath=f"images/{kid.rsplit(':', 1)[-1]}.jpg", width=1, height=1,
                    byte_count=1) for i, kid in enumerate(sol.keyframe_ids)]
    CP.gate_final_solution(store, WORLD, SESSION, sol, database_path="db", keyframes=kfs)
    _kept(dense)[0].write_bytes(b"")
    r = CP.gate_final_solution(store, WORLD, SESSION, sol, database_path="db", keyframes=kfs)
    assert r.record["depth"]["state"] == CP.DEPTH_OK
    assert r.record["retryable"] is False


# ---------------------------------------------------------------------------
# LOW: the token names what the prediction was made with
# ---------------------------------------------------------------------------


def _moge():
    return MoGeBackend("Ruicheng/moge-2-vitl", "moge2-vitl", "MIT")


def test_the_token_sees_the_device_and_the_weights_revision():
    """RV9-C P3: at 6d4b567 these two made one token."""
    a, b = _moge(), _moge()
    a._device, b._device = "cuda", "cpu"
    a._revision = b._revision = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
    assert DP.prediction_token(a, 42.155897)[0] != DP.prediction_token(b, 42.155897)[0]
    b._device = "cuda"
    assert DP.prediction_token(a, 42.155897)[0] == DP.prediction_token(b, 42.155897)[0]
    b._revision = "some-later-revision"
    assert DP.prediction_token(a, 42.155897)[0] != DP.prediction_token(b, 42.155897)[0]


def test_the_token_sees_fp16_and_the_library_versions(monkeypatch):
    a, b = _moge(), _moge()
    for x in (a, b):
        x._device, x._revision = "cuda", "r"
    base = DP.prediction_token(a, 42.0)[0]
    b.use_fp16 = False
    assert DP.prediction_token(b, 42.0)[0] != base
    versions = {"moge": "3.0.0", "torch": "2.13.0+cu132"}
    monkeypatch.setattr(DP, "_package_version", lambda d: versions.get(d), raising=False)
    t1 = DP.prediction_token(a, 42.0)[0]
    versions["torch"] = "2.14.0+cu132"
    assert DP.prediction_token(a, 42.0)[0] != t1
    versions["torch"], versions["moge"] = "2.13.0+cu132", "3.1.0"
    assert DP.prediction_token(a, 42.0)[0] != t1


def test_the_token_records_this_machines_runtime():
    """What the doc says for the real backend, unloaded, on this machine: the values the
    process would load and run it with."""
    from importlib.metadata import version

    import torch

    _token, doc = DP.prediction_token(_moge(), 42.0)
    assert doc["schema"] == 2
    assert doc["torch"] == version("torch") and doc["moge"] == version("moge")
    assert doc["device"] == ("cuda" if torch.cuda.is_available() else "cpu")
    assert doc["fp16"] is True, "MoGe's infer() autocasts to float16 unless told otherwise"
    from tower.world_builder.dense import hub_model_cache

    ref = hub_model_cache("Ruicheng/moge-2-vitl") / "refs" / "main"
    assert doc["weights_revision"] == (ref.read_text().strip() if ref.is_file() else None)


# ---------------------------------------------------------------------------
# M-12: one set, the latest prediction of each keyframe
# ---------------------------------------------------------------------------


def test_another_tokens_set_is_pruned_and_recorded(world):
    """At 6d4b567 both sets stayed (`test_other_pixels_or_another_network_parameter_is_a_miss`
    counted two directories)."""
    store, net, dense = world
    _gate_depth(store)
    (dense / DP.PREDICTIONS_DIRNAME / ("cf00cb91be99" + "8bc4")).mkdir()    # a pre-a7f2dad name, empty
    first_bytes = sum(p.stat().st_size for p in _kept(dense))
    sol = GS.load_solution(store, WORLD, SESSION)
    wider = dc.replace(sol, camera=dict(sol.camera, fx=sol.camera["fx"] * 0.8, fy=sol.camera["fy"] * 0.8))
    again, _, _ = _gate_depth(store, wider)
    token = again["prediction_cache"]["token"]
    assert [d.name for d in (dense / DP.PREDICTIONS_DIRNAME).iterdir()] == [token]
    assert again["prediction_cache"]["pruned"] == {"sets": 2, "files": 8, "bytes": first_bytes,
                                                   "staging": 0}
    assert len(_kept(dense)) == 8


def test_a_keyframe_whose_pixels_changed_keeps_only_its_latest_prediction(world):
    store, net, dense = world
    _gate_depth(store)
    first = sorted(store.images_dir(WORLD, SESSION).glob("*.jpg"))[0]
    img = cv2.imread(str(first))
    img[:4, :4] = 0
    first.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    again, _, _ = _gate_depth(store)
    assert again["prediction_cache"]["predicted"] == 1
    assert len(_kept(dense)) == 8, "the superseded prediction is gone"
    assert again["prediction_cache"]["pruned"]["files"] == 1
    index = json.loads((dense / DP.PREDICTIONS_DIRNAME / again["prediction_cache"]["token"]
                        / DP.PREDICTION_INDEX).read_text())
    assert sorted(index["keyframes"]) == sorted(GS.load_solution(store, WORLD, SESSION).keyframe_ids)
    assert {p.stem for p in _kept(dense)} == set(index["keyframes"].values())


def test_a_draw_posing_fewer_keyframes_keeps_the_others_predictions(world):
    """The consensus: each draw's depth stage rewrites `align.json` with ITS keyframes. The
    chosen draw's frames must survive a later draw's prune, or a re-gate of it predicts them
    again -- the non-reproducibility the cache exists to remove."""
    store, net, dense = world
    _gate_depth(store)
    sol = GS.load_solution(store, WORLD, SESSION)
    fewer = dc.replace(sol, poses={k: p for i, (k, p) in enumerate(sol.poses.items()) if i >= 2})
    draw, _, _ = _gate_depth(store, fewer)
    assert draw["targets"] == 6 and draw["prediction_cache"]["pruned"]["files"] == 0
    assert len(_kept(dense)) == 8
    again, _, _ = _gate_depth(store)
    assert net.calls == 8 and again["prediction_cache"]["hits"] == 8


def test_a_dead_writers_staging_file_is_swept_a_live_one_kept(world):
    store, net, dense = world
    first, _, _ = _gate_depth(store)
    here = dense / DP.PREDICTIONS_DIRNAME / first["prediction_cache"]["token"]
    dead = here / f"{'0' * 20}.npy.p{2 ** 22 + 7}.deadbeef.tmp"
    live = here / f"{'1' * 20}.npy.p{os.getpid()}.cafebabe.tmp"
    dead.write_bytes(b"x" * 100)
    live.write_bytes(b"y")
    again, _, _ = _gate_depth(store)
    assert not dead.exists() and live.exists()
    assert again["prediction_cache"]["pruned"]["staging"] == 1
    assert len(_kept(dense)) == 8


def test_a_stopped_stage_prunes_nothing(world):
    store, net, dense = world
    _gate_depth(store)
    stray = dense / DP.PREDICTIONS_DIRNAME / "000000000000"
    stray.mkdir()
    (stray / f"{'2' * 20}.npy").write_bytes(b"z")
    sol = GS.load_solution(store, WORLD, SESSION)
    with pytest.raises(CP.DepthUnavailable):
        CP.run_gate_depth(store, WORLD, SESSION, sol, store.read_session(WORLD, SESSION).intrinsics,
                          should_stop=lambda: True)
    assert (stray / f"{'2' * 20}.npy").exists()


def test_an_unreadable_index_prunes_nothing_in_its_set(world):
    store, net, dense = world
    first, _, _ = _gate_depth(store)
    here = dense / DP.PREDICTIONS_DIRNAME / first["prediction_cache"]["token"]
    (here / DP.PREDICTION_INDEX).write_text("{torn")
    extra = here / f"{'3' * 20}.npy"
    np.save(extra, np.zeros((2, 2), np.float16))
    again, _, _ = _gate_depth(store)
    assert extra.exists() and again["prediction_cache"]["pruned"]["kept_because"]


def test_off_nothing_is_pruned_or_indexed(world):
    """Every caller but the gate: the stage is today's, `predictions/` untouched."""
    store, net, dense = world
    _gate_depth(store)
    before = sorted(p.name for p in (dense / DP.PREDICTIONS_DIRNAME).rglob("*"))
    other = dense / DP.PREDICTIONS_DIRNAME / "000000000000"
    other.mkdir()
    sol = GS.load_solution(store, WORLD, SESSION)
    align = DP.run_depth_stage(store, WORLD, SESSION, sol, store.read_session(WORLD, SESSION).intrinsics,
                               CP._depth_params(), dense)
    assert "prediction_cache" not in align and other.is_dir()
    assert sorted(p.name for p in (dense / DP.PREDICTIONS_DIRNAME).rglob("*")) == sorted(
        before + [other.name])


# ---------------------------------------------------------------------------
# Q8: a densify and the gate's depth stage never write dense/<session>/ at once
# ---------------------------------------------------------------------------


def test_a_densify_waits_its_turn_while_the_sessions_surface_lock_is_held(world):
    """The gate's depth stage and a surface build hold `_SurfaceLock` while they write
    `dense/<session>/`. At 6d4b567 a densify took only `_DenseLock` and ran straight through:
    it wrote `work/` and `align.json` and pruned `work/` under the holder."""
    from tower.world_builder.surface_pipeline import _SurfaceLock, surface_dir

    store, net, dense = world

    def tree():
        return {str(p.relative_to(dense)): p.read_bytes() for p in sorted(dense.rglob("*")) if p.is_file()}

    before = tree()
    holder = _SurfaceLock(surface_dir(store, WORLD, SESSION))
    assert holder.acquire()
    try:
        result = DP.densify(store, WORLD, SESSION)
    finally:
        holder.release()
    assert net.calls == 0, "the densify predicted while the gate's depth stage held the session"
    assert tree() == before, "the loser writes nothing under dense/<session>/, and keeps no lock"
    assert result.state == "unavailable" and "lock" in (result.detail or ""), result


def test_the_gates_depth_stage_is_refused_while_a_densify_runs(world, monkeypatch):
    """The other direction: a gate depth stage started while a densify is writing is refused
    ("another surface build of this session holds its lock") -- the gate's retryable
    depth-unavailable -- rather than interleaving its `<ki>_pred.npy` with the densify's."""
    store, net, dense = world
    seen = []

    def progress(stage, n, total):
        if stage == STAGE_DEPTH and not seen:
            try:
                _gate_depth(store)
                seen.append("ran")
            except CP.DepthUnavailable as exc:
                seen.append(str(exc))

    DP.densify(store, WORLD, SESSION, progress=progress)
    assert seen == ["another surface build of this session holds its lock"]
