"""Review V10, L-14 and L-15: the gate's depth-prediction cache repairs itself, and a depth
prediction with no finite value is never a fitted frame.

Module: `tower/world_builder/dense_pipeline.py` (`prune_prediction_cache`, `_cached_prediction`,
`_fit_record`). RV10-D's probes, RUN `baseline/review/V10/dep/test_rv10d_probes.py` (D1, D7)
and `test_rv10d_nan.py`, are the red cases here. At 4a95a0d:

  * L-14: a torn `predictions/<token>/index.json` was never rewritten, so the in-token prune
    stopped for good -- three re-redactions of one frame kept 9, 10, 11 predictions (D7);
  * L-15: an all-NaN (or all-infinite) kept prediction was a HIT, never predicted again (D1),
    and `_fit_record` then recorded the frame `ok` with a = b = NaN.
"""

from __future__ import annotations

import dataclasses as dc
import json
import math
import pathlib

import cv2
import numpy as np
import pytest

import tower.world_builder.dense_pipeline as DP
from tests.test_world_builder_depth_prediction_cache import (  # noqa: F401 -- fixture
    _fits,
    _gate_depth,
    world,
)
from tests.test_world_builder_surface_pipeline import SESSION, WORLD
from tower.world_builder import coherence_publish as CP
from tower.world_builder import global_solve as GS


def _kept(dense):
    return sorted((dense / DP.PREDICTIONS_DIRNAME).rglob("*.npy"))


def _token_dir(dense, align):
    return dense / DP.PREDICTIONS_DIRNAME / align["prediction_cache"]["token"]


def _redact_again(store, k):
    """A re-redaction of the first keyframe: new pixels, so a new prediction of it."""
    first = sorted(store.images_dir(WORLD, SESSION).glob("*.jpg"))[0]
    img = cv2.imread(str(first))
    img[:4, :4] = 10 * (k + 1)
    first.write_bytes(cv2.imencode(".png", img)[1].tobytes())


# ---------------------------------------------------------------------------
# L-14: a torn index is rewritten, and the prune resumes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("torn", [b"{torn", b"[]", b'{"keyframes": 3}', b"\xff\xfe\x00junk"])
def test_a_torn_index_is_rewritten_and_the_prune_resumes(world, torn):
    """RV10-D D7. At 4a95a0d: kept 9, 10, 11 and the index still torn after all three."""
    store, net, dense = world
    first, _, _ = _gate_depth(store)
    here = _token_dir(dense, first)
    (here / DP.PREDICTION_INDEX).write_bytes(torn)
    counts, pruned = [], []
    for k in range(3):
        _redact_again(store, k)
        align, _, _ = _gate_depth(store)
        counts.append(len(_kept(dense)))
        pruned.append(align["prediction_cache"]["pruned"])
    index = json.loads((here / DP.PREDICTION_INDEX).read_text(encoding="utf-8"))
    kids = sorted(GS.load_solution(store, WORLD, SESSION).keyframe_ids)
    assert sorted(index["keyframes"]) == kids and index["token"] == here.name
    assert {p.stem for p in _kept(dense)} == set(index["keyframes"].values())
    # The run that met the torn index rewrote it and removed nothing on its strength;
    # every run after it prunes again.
    assert pruned[0].get("index_rewritten") is True and pruned[0]["files"] == 0
    assert counts == [9, 8, 8]
    assert pruned[1]["files"] == 2 and pruned[2]["files"] == 1
    assert "kept_because" not in pruned[1] and "kept_because" not in pruned[2]


def test_the_rewritten_index_keeps_every_frame_of_the_run_that_rewrote_it(world):
    """A consensus draw posing fewer frames meets the torn index: its frames are named, and a
    later draw's prune keeps both draws' frames."""
    store, net, dense = world
    first, _, _ = _gate_depth(store)
    here = _token_dir(dense, first)
    (here / DP.PREDICTION_INDEX).write_text("{torn")
    sol = GS.load_solution(store, WORLD, SESSION)
    fewer = dc.replace(sol, poses={k: p for i, (k, p) in enumerate(sol.poses.items()) if i >= 2})
    draw, _, _ = _gate_depth(store, fewer)
    assert draw["prediction_cache"]["pruned"]["index_rewritten"] is True
    index = json.loads((here / DP.PREDICTION_INDEX).read_text())
    assert len(index["keyframes"]) == 6
    again, _, _ = _gate_depth(store)
    assert again["prediction_cache"]["hits"] == 8 and net.calls == 8
    assert len(_kept(dense)) == 8


def test_an_index_that_cannot_be_opened_is_left_alone(world, monkeypatch):
    """A held index (a sharing violation on read) is not torn: it is neither rewritten nor pruned
    by, and the next stage reads it again."""
    store, net, dense = world
    first, _, _ = _gate_depth(store)
    here = _token_dir(dense, first)
    before = (here / DP.PREDICTION_INDEX).read_bytes()
    extra = here / f"{'3' * 20}.npy"
    np.save(extra, np.zeros((2, 2), np.float16))
    real_text, real_bytes = pathlib.Path.read_text, pathlib.Path.read_bytes

    def held(real):
        def read(self, *a, **k):
            if self.name == DP.PREDICTION_INDEX:
                raise PermissionError(13, "being used by another process")
            return real(self, *a, **k)
        return read

    monkeypatch.setattr(pathlib.Path, "read_text", held(real_text))
    monkeypatch.setattr(pathlib.Path, "read_bytes", held(real_bytes))
    again, _, _ = _gate_depth(store)
    assert again["prediction_cache"]["pruned"]["kept_because"]
    assert "index_rewritten" not in again["prediction_cache"]["pruned"]
    with open(here / DP.PREDICTION_INDEX, "rb") as handle:
        assert handle.read() == before
    assert extra.exists()


# ---------------------------------------------------------------------------
# L-15: a prediction with no finite value is a miss, and never a fitted frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fill", ["nan", "inf", "-inf", "nan+inf"])
def test_a_kept_prediction_with_no_finite_value_is_a_miss(world, fill):
    """RV10-D D1 (nan, inf). At 4a95a0d: network calls 0, hits 8, the file never rewritten, and
    the frame fitted `ok` with a = b = NaN."""
    store, net, dense = world
    first, _, _ = _gate_depth(store)
    victim = _kept(dense)[0]
    good = np.load(victim)
    bad = np.full(good.shape, np.nan if fill.startswith("nan") else float(fill), np.float16)
    if fill == "nan+inf":
        bad[::2] = np.inf
    np.save(victim, bad)
    calls = net.calls
    again, _, _ = _gate_depth(store)
    assert net.calls == calls + 1, "that frame alone was predicted again"
    assert again["prediction_cache"]["hits"] == 7 and again["prediction_cache"]["predicted"] == 1
    assert np.array_equal(np.load(victim), good), "and its prediction rewritten"
    assert _fits(again) == _fits(first) and len(_fits(again)) == 8


def test_a_kept_prediction_with_some_nan_is_still_a_hit(world):
    """MoGe marks the pixels it has no depth for NaN (`MoGeBackend.predict`): a prediction with
    SOME finite value is the network's own output, and is kept."""
    store, net, dense = world
    _gate_depth(store)
    victim = _kept(dense)[0]
    part = np.load(victim)
    part[: part.shape[0] // 3] = np.nan
    np.save(victim, part)
    calls = net.calls
    again, _, _ = _gate_depth(store)
    assert net.calls == calls and again["prediction_cache"]["hits"] == 8
    assert np.array_equal(np.load(victim), part, equal_nan=True)


class _BlindOn:
    """Wraps the fixture's network: frame `blind` (by call order) comes back all-NaN."""

    def __init__(self, net, blind=0, value=np.nan):
        self.net, self.blind, self.value, self.n = net, blind, value, 0
        for k in ("name", "model_id", "resolution_level", "licence", "windowed", "window_size",
                  "kind", "accepts_fov"):
            setattr(self, k, getattr(net, k))

    def predict(self, rgb, fov_x=None):
        out = self.net.predict(rgb, fov_x=fov_x)
        self.n += 1
        return np.full_like(out, self.value) if self.n - 1 == self.blind else out


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_a_prediction_with_no_finite_value_is_never_fitted_ok(world, monkeypatch, value):
    """RV10 L-15 and `test_rv10d_nan.py`, on the default (uncached) path: at 4a95a0d the frame
    was recorded `ok` with a non-finite fit."""
    store, net, dense = world
    blind = _BlindOn(net, value=value)
    monkeypatch.setattr(DP, "make_backend", lambda name: blind)
    depth = dense / "work" / "depth"
    before = {p.name: p.read_bytes() for p in depth.glob("*.npy")} if depth.is_dir() else {}
    sol = GS.load_solution(store, WORLD, SESSION)
    align = DP.run_depth_stage(store, WORLD, SESSION, sol, store.read_session(WORLD, SESSION).intrinsics,
                               CP._depth_params(), dense)
    records = align["records"]
    assert all(math.isfinite(r["a"]) and math.isfinite(r["b"]) for r in records if r.get("ok")), \
        "no frame is `ok` with a non-finite fit"
    bad = [r for r in records if not r.get("ok")]
    assert len(bad) == 1 and "finite" in bad[0]["why"]
    assert len([r for r in records if r.get("ok")]) == 7
    fitted = depth / f"{bad[0]['ki']:05d}.npy"
    assert (fitted.read_bytes() if fitted.exists() else None) == before.get(fitted.name), \
        "no fitted depth map is written for it"


def test_a_fresh_prediction_with_no_finite_value_is_not_kept(world, monkeypatch):
    """With the cache on it would only be a miss next time: it is not written."""
    store, net, dense = world
    blind = _BlindOn(net)
    monkeypatch.setattr(DP, "make_backend", lambda name: blind)
    align, _, _ = _gate_depth(store)
    cache = align["prediction_cache"]
    assert cache["predicted"] == 8 and cache["write_failed"] == 0
    assert len(_kept(dense)) == 7
    assert len([r for r in align["records"] if r.get("ok")]) == 7
