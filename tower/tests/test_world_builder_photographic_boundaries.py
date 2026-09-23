"""The photographic lifecycle at its boundaries: what three reviewers broke.

Written on 2026-09-23 from the independent reviews of the finisher
remediation. Each test below was a PROBE first -- a reviewer's test that
asserted the defect and passed -- turned round to assert the fix:

* the real stage sequence read "Saved" for one record write between the
  surface's `ok` and the appearance's `running`, and a builder that died there
  left a grey mesh that said "Saved" for ever and that nothing would finish;
* a `surface_pipeline` that failed to import turned every HISTORICAL world
  `unobservable` -- "Finishing" on the phone -- silently;
* the recovery finisher's world lock relabelled a finished sibling session of
  the same world as being improved;
* a momentary appearance refusal (its lock busy, the redaction label moving
  under the build) was recorded as a FAILED photographic build: terminal,
  never retried, "Saved" over a grey mesh;
* a Tower with the appearance switched off rebuilt a surface over and over
  under an appearance entry left `stopped` by an earlier Tower.

The stage functions are faked; the RECORD writes, the readers and both
judges are real.
"""

import importlib.util
import logging
import os
import pathlib
import subprocess
import sys
import time

import pytest

from tests.test_world_builder_photographic_lifecycle import (
    _COMPLETE_SOLVED,
    _finalized,
    _payload,
    _stage,
    _state_of,
    _write_status,
)
from tower.results.world_builder_library import build_world_listing
from tower.world_builder.engine import WorldBuilderEngine
from tower.world_builder.events import WorldEvent
from tower.world_builder.records import (
    STAGE_APPEARANCE,
    STAGE_STATE_OK,
    STAGE_STATE_RUNNING,
    STAGE_STATE_STOPPED,
    STAGE_STATE_UNAVAILABLE,
    STAGE_SURFACE,
    Session,
)

TOWER_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _finisher():
    path = TOWER_ROOT / "scripts" / "world_finish_pending.py"
    spec = importlib.util.spec_from_file_location("wfp_boundaries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(store, session_id):
    for world in build_world_listing(store)["worlds"]:
        for row in world["sessions"]:
            if row["session_id"] == session_id:
                return row
    raise AssertionError(f"no row for {session_id}")


def _words(store, world_id, session_id):
    payload = _payload(store, world_id, session_id)
    row = _row(store, session_id)
    return {
        "model_state": payload["model_state"],
        "row": row["state"],
        "photographic": (row.get("photographic") or {}).get("state"),
    }


def _says_saved(words) -> bool:
    return words["model_state"] == "finalized" or words["row"] == "complete"


class _Result:
    def __init__(self, state, *, detail=None, retryable=False):
        self.state = state
        self.detail = detail
        self.retryable = retryable

    def as_dict(self):
        return {"state": self.state, "detail": self.detail}


def _run_stages(store, world_id, session_id, monkeypatch, *, appearance=True,
                appearance_result=None, snapshot=None):
    """The REAL `final_surface_stages`, with the two expensive stages faked."""
    import tower.world_builder.appearance_pipeline as appearance_pipeline
    import tower.world_builder.surface_pipeline as surface_pipeline
    from scripts.world_build_session import final_surface_stages

    engine = WorldBuilderEngine(store)

    def record(stage, *, state, detail=None, attempted=True):
        engine.mark_stage(world_id, session_id, stage, state=state, detail=detail,
                          attempted=attempted)
        if snapshot is not None:
            snapshot(f"after {stage}={state}")

    def surfacify(*_a, **_k):
        _write_status(store, world_id, session_id, STAGE_SURFACE, state="running",
                      pid=os.getpid(), updated_at=time.time())
        (store.world_dir(world_id) / STAGE_SURFACE / session_id / "manifest.json"
         ).write_text("{}")
        _write_status(store, world_id, session_id, STAGE_SURFACE, state="ok",
                      updated_at=time.time())
        return _Result("ok")

    def build_appearance(*_a, **_k):
        if snapshot is not None:
            snapshot("inside the appearance, before its first status")
        if appearance_result is not None:
            return appearance_result
        _write_status(store, world_id, session_id, STAGE_APPEARANCE, state="ok",
                      updated_at=time.time())
        return _Result("ok")

    monkeypatch.setattr(surface_pipeline, "surfacify", surfacify)
    monkeypatch.setattr(appearance_pipeline, "build_appearance", build_appearance)
    return final_surface_stages(
        store, world_id, session_id, solved=True, appearance=appearance,
        prune_depth_work=False, should_stop=lambda: False, record=record,
    )


class TestTheSurfaceToAppearanceBoundary:
    def test_the_real_sequence_never_reads_saved_before_the_appearance(
        self, derived_world, monkeypatch
    ):
        store, w, s = derived_world
        # The builder's own mark before it releases the lock.
        _finalized(store, w, s, stages={STAGE_SURFACE: _stage(STAGE_STATE_RUNNING)})
        snaps = []
        _run_stages(store, w, s, monkeypatch,
                    snapshot=lambda name: snaps.append((name, _words(store, w, s))))

        saved_early = [name for name, words in snaps[:-1] if _says_saved(words)]
        assert saved_early == [], snaps
        # And the last write is the one that settles it.
        assert _words(store, w, s)["photographic"] == "complete"
        assert _says_saved(_words(store, w, s))

    def test_a_builder_killed_just_after_the_surface_leaves_owed_work(
        self, derived_world, monkeypatch
    ):
        """What the record says if the process dies the instant the surface is `ok`."""
        store, w, s = derived_world
        _finalized(store, w, s, stages={STAGE_SURFACE: _stage(STAGE_STATE_RUNNING)})

        class _Killed(BaseException):
            pass

        def die_after_the_surface(name):
            if name == f"after {STAGE_SURFACE}={STAGE_STATE_OK}":
                raise _Killed

        with pytest.raises(_Killed):
            _run_stages(store, w, s, monkeypatch, snapshot=die_after_the_surface)

        words = _words(store, w, s)
        assert not _says_saved(words), words
        assert words["photographic"] == "owed"
        verdict = _finisher().assess(store, w, s)
        assert verdict.owed and verdict.stage == STAGE_APPEARANCE, verdict


class TestAMomentaryRefusalIsNotAFailure:
    @pytest.mark.parametrize("detail", [
        "another appearance build of this session is already running",
        "the session's redaction label changed during the build",
    ])
    def test_a_retryable_refusal_is_owed(self, derived_world, monkeypatch, detail):
        store, w, s = derived_world
        _finalized(store, w, s, stages={STAGE_SURFACE: _stage(STAGE_STATE_RUNNING)})
        _run_stages(store, w, s, monkeypatch, appearance_result=_Result(
            STAGE_STATE_UNAVAILABLE, detail=detail, retryable=True))

        entry = store.read_session(w, s).stages[STAGE_APPEARANCE]
        assert entry["state"] == STAGE_STATE_STOPPED
        assert _state_of(store, w, s)["state"] == "owed"
        assert _finisher().assess(store, w, s).owed

    def test_a_refusal_about_the_session_is_still_a_failure(
        self, derived_world, monkeypatch
    ):
        """Not every `unavailable` is momentary: open3d missing will say it again."""
        store, w, s = derived_world
        _finalized(store, w, s, stages={STAGE_SURFACE: _stage(STAGE_STATE_RUNNING)})
        _run_stages(store, w, s, monkeypatch, appearance_result=_Result(
            STAGE_STATE_UNAVAILABLE, detail="open3d is not installed"))
        assert _state_of(store, w, s)["state"] == "failed"
        assert not _finisher().assess(store, w, s).owed


class TestAppearanceSwitchedOff:
    def test_a_stale_stopped_appearance_is_overwritten(self, derived_world, monkeypatch):
        store, w, s = derived_world
        _finalized(store, w, s, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_STOPPED),
            STAGE_APPEARANCE: _stage(STAGE_STATE_STOPPED),
        })
        _run_stages(store, w, s, monkeypatch, appearance=False)

        entry = store.read_session(w, s).stages[STAGE_APPEARANCE]
        assert entry["state"] == STAGE_STATE_UNAVAILABLE
        assert entry["attempted"] is False
        # Not owed: the finisher would otherwise rebuild the surface until
        # its bound retired a perfectly good one.
        assert not _finisher().assess(store, w, s).owed
        assert _state_of(store, w, s)["state"] == "unattempted"


def _break_surface_pipeline(monkeypatch):
    monkeypatch.delitem(sys.modules, "tower.world_builder.surface_pipeline", raising=False)
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "tower.world_builder.surface_pipeline":
            raise NameError("surface.py is broken at import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)


class TestABrokenImportDoesNotRelabelHistory:
    @pytest.mark.parametrize("finalization", [dict(_COMPLETE_SOLVED), None])
    def test_historical_worlds_keep_their_words(
        self, derived_world, monkeypatch, finalization
    ):
        store, w, s = derived_world
        store.append_event(w, s, WorldEvent(event_id=1, kind="session_stopped",
                                            at=2.0, payload={}))
        session = store.read_session(w, s)
        store.write_session(Session(
            session_id=s, world_id=w, started_at=session.started_at,
            ended_at=session.ended_at, end_reason="stop",
            finalization=finalization, stages=None,
        ))
        before = _words(store, w, s)
        _break_surface_pipeline(monkeypatch)
        assert _words(store, w, s) == before

    def test_a_session_it_cannot_judge_is_unobservable_and_says_why(
        self, derived_world, monkeypatch, caplog
    ):
        store, w, s = derived_world
        _finalized(store, w, s, stages={STAGE_SURFACE: _stage(STAGE_STATE_RUNNING)})
        _write_status(store, w, s, STAGE_SURFACE, state="running", pid=os.getpid(),
                      updated_at=time.time())
        _break_surface_pipeline(monkeypatch)
        caplog.set_level(logging.WARNING)
        assert _state_of(store, w, s)["state"] == "unobservable"
        warned = [r for r in caplog.records if "unobservable" in r.getMessage()]
        assert warned, [r.getMessage() for r in caplog.records]
        assert warned[0].exc_info is not None, "the WHY is the traceback"


class TestTheLockSpeaksOnlyForItsSession:
    def _hold_the_lock(self, store, world_id):
        code = (
            "import sys, time; sys.path.insert(0, %r)\n"
            "from tower.world_builder.store import WorldStore\n"
            "WorldStore(%r).acquire_writer_lock(%r)\n"
            "print('locked', flush=True); time.sleep(60)\n"
            % (str(TOWER_ROOT), str(store.root), world_id)
        )
        child = subprocess.Popen([sys.executable, "-c", code],
                                 stdout=subprocess.PIPE, text=True)
        assert child.stdout.readline().strip() == "locked"
        return child

    def test_a_finished_sibling_is_not_relabelled(self, derived_world):
        store, w, s = derived_world
        _finalized(store, w, s, stages=None)  # historical: finished long ago
        before = _words(store, w, s)
        child = self._hold_the_lock(store, w)  # the finisher, on ANOTHER session
        try:
            during = _words(store, w, s)
        finally:
            child.kill()
            child.wait()
        assert during == before

    def test_the_session_being_written_still_reads_as_written(self, derived_world):
        """Open, or finalization pending: the lock IS about this session."""
        store, w, s = derived_world
        session = store.read_session(w, s)
        store.write_session(Session(
            session_id=s, world_id=w, started_at=session.started_at,
            ended_at=None, end_reason=None, finalization=None, stages=None,
        ))
        child = self._hold_the_lock(store, w)
        try:
            words = _words(store, w, s)
        finally:
            child.kill()
            child.wait()
        assert words["row"] == "receiving"
        assert words["model_state"] != "finalized"


class TestThePipelineSaysWhichRefusalsAreMomentary:
    """The flag the stage record is decided on, from the pipeline itself."""

    def test_a_busy_lock_is_retryable(self, derived_world):
        from tower.world_builder import appearance_pipeline as AP

        store, w, s = derived_world
        root = AP.appearance_dir(store, w, s)
        root.mkdir(parents=True, exist_ok=True)
        held = AP._lock(root)
        assert held.acquire()
        try:
            result = AP.build_appearance(store, w, s)
        finally:
            held.release()
        assert result.state == STAGE_STATE_UNAVAILABLE
        assert result.retryable is True

    @pytest.mark.parametrize("retryable", [True, False])
    def test_the_refusal_carries_its_flag_through(self, derived_world, monkeypatch,
                                                  retryable):
        from tower.world_builder import appearance as A
        from tower.world_builder import appearance_pipeline as AP

        store, w, s = derived_world

        def refuse(*_a, **_k):
            raise A.AppearanceUnavailable("because", retryable=retryable)

        monkeypatch.setattr(AP, "_build", refuse)
        result = AP.build_appearance(store, w, s)
        assert result.state == STAGE_STATE_UNAVAILABLE
        assert result.retryable is retryable
