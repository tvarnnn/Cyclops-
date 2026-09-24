"""The ONE read-only answer to "would a re-gate of this session be refused, and why"
(`coherence_publish.regate_refusal`; review V8, FIN's OPEN 3).

`world_finish_pending._regate_refusal` restated `regate_published`'s two refusals word for
word because `coherence_publish.py` was another lane's file. The lane that owns it now
exposes the answer itself; these tests hold all three -- the new function, the finisher's
copy, and what `regate_published` actually raises -- to one answer on every shape, so the
lead can switch the finisher over without a behaviour change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_finish_pending as wfp  # noqa: E402
from tests.test_world_builder_finish_pending_gate import (  # noqa: E402
    DEPTH_LOST,
    ROOM_OK,
    _gated,
    _solution,
)
from tower.world_builder import coherence_publish as CP  # noqa: E402
from tower.world_builder.global_solve import workspace_for, write_solution  # noqa: E402


@pytest.fixture(autouse=True)
def _no_native_warm(monkeypatch):
    monkeypatch.setattr(wfp, "prewarm_world_builder", lambda *a, **k: ())


def _shape(tmp_path, shape):
    """A published session in one of the shapes a re-gate can meet."""
    if shape == "accepted":
        return _gated(tmp_path, stages=ROOM_OK)
    if shape == "database-gone":
        return _gated(tmp_path, stages=ROOM_OK, database=False)
    if shape == "solution-will-not-load":
        return _gated(tmp_path, stages=ROOM_OK, loadable=False)
    if shape == "ungated-solution":
        return _gated(tmp_path, stages=ROOM_OK, gate=None)
    if shape == "no-solution":
        store = _gated(tmp_path, stages=ROOM_OK)
        ws = workspace_for(store, "w1", "s1")
        ws.solution_path.unlink()
        ws.arrays_path.unlink()
        return store
    if shape in ("consensus-without-camera", "consensus-with-camera"):
        # A solve that asked for a consensus: the re-gate maps its draws, with the solve's camera
        # (or the workspace's camera.json). This fixture's solution has no camera (review V9 LOW).
        gate = dict(DEPTH_LOST, consensus={"state": "deferred", "requested": 3, "seeds": [7, 8, 9]})
        store = _gated(tmp_path, stages=ROOM_OK, gate=gate)
        if shape == "consensus-with-camera":
            workspace_for(store, "w1", "s1").camera_path.write_text(
                '{"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240}',
                encoding="utf-8")
        return store
    if shape in ("named-database-present", "named-database-gone"):
        # The solve mapped a per-solve masked database: that is the one a re-gate needs.
        store = _gated(tmp_path, stages=ROOM_OK)
        ws = workspace_for(store, "w1", "s1")
        sol = _solution("s1", DEPTH_LOST)
        sol.solve = {"database": "database.masked.p1.0123abcd.db"}
        write_solution(ws, sol)
        if shape == "named-database-present":
            (ws.root / "database.masked.p1.0123abcd.db").write_bytes(b"masked features")
        return store
    raise AssertionError(shape)


SHAPES = ("accepted", "database-gone", "solution-will-not-load", "ungated-solution", "no-solution",
          "named-database-present", "named-database-gone", "consensus-without-camera", "consensus-with-camera")
# THE ANSWERS, stated independently of the code under test (review V9 LOW: the finisher's-copy test
# compared the function with itself once the finisher called it).
EXPECTED = {
    "accepted": None,
    "database-gone": "the solve's database database.db is gone; an owner can re-finish this walk",
    "solution-will-not-load": "no gated solution is published for this session",
    "ungated-solution": "no gated solution is published for this session",
    "no-solution": "no gated solution is published for this session",
    "named-database-present": None,
    "named-database-gone": ("the solve's database database.masked.p1.0123abcd.db is gone; an owner can "
                            "re-finish this walk"),
    "consensus-without-camera": ("the solve has no camera to map its consensus draws with; an owner can "
                                 "re-finish this walk"),
    "consensus-with-camera": None,
}


def _what_regate_published_does(store) -> str | None:
    """The refusal `regate_published` raises, or None when it reached the gate."""

    class Reached(Exception):
        pass

    def gate_runner(*a, **k):
        raise Reached()

    try:
        CP.regate_published(store, "w1", "s1", gate_runner=gate_runner)
    except CP.RegateRefused as exc:
        return str(exc)
    except Reached:
        return None
    raise AssertionError("regate_published neither refused nor reached the gate")


@pytest.mark.parametrize("shape", SHAPES)
def test_the_read_only_refusal_is_regate_publisheds_own_answer(tmp_path, shape):
    store = _shape(tmp_path, shape)
    assert CP.regate_refusal(store, "w1", "s1") == _what_regate_published_does(store) == EXPECTED[shape]


@pytest.mark.parametrize("shape", SHAPES)
def test_the_finishers_copy_gives_the_same_answer(tmp_path, shape):
    """The finisher's read-only answer, against the independent statement of it."""
    store = _shape(tmp_path, shape)
    assert wfp._regate_refusal(store, "w1", "s1") == EXPECTED[shape]


@pytest.mark.parametrize("shape", SHAPES)
def test_the_refusal_is_read_only(tmp_path, shape):
    store = _shape(tmp_path, shape)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    CP.regate_refusal(store, "w1", "s1")
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert store.lock_holder("w1") is None


def test_the_reasons_are_the_published_sentences(tmp_path):
    assert CP.regate_refusal(_shape(tmp_path / "a", "no-solution"), "w1", "s1") == \
        CP.REFUSAL_NO_GATED_SOLUTION
    assert CP.regate_refusal(_shape(tmp_path / "b", "database-gone"), "w1", "s1") == \
        CP.REFUSAL_DATABASE_GONE.format(name="database.db")
    assert CP.regate_refusal(_shape(tmp_path / "c", "named-database-gone"), "w1", "s1") == \
        CP.REFUSAL_DATABASE_GONE.format(name="database.masked.p1.0123abcd.db")
    assert CP.regate_refusal(_shape(tmp_path / "d", "accepted"), "w1", "s1") is None
