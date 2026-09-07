"""An interim rebuild that cannot write must not end the mapping session.

On 2026-09-06 a live builder died mid-walk with a `PermissionError` from
`write_derived` (a reader held points.json across the replace). The
session record was never finalised, the LOCK stayed behind with a dead
pid, and the derived tree was left torn. Interim rebuilds are best-effort
views for the wearer; the next one rewrites everything. So a failed one
is logged and retried at the next rebuild, and the session goes on to
its stop, its final solve and its final build.
"""

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "world_build_session.py"
    spec = importlib.util.spec_from_file_location("world_build_session_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_interim_rebuild_that_cannot_write_is_retried_not_fatal(tmp_path, monkeypatch, caplog, capsys):
    script = _load_script()
    real_build = script.WorldBuilderEngine.build
    calls = {"n": 0}

    def flaky_build(self, world_id, session_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "Access is denied", "points.json")
        return real_build(self, world_id, session_id)

    monkeypatch.setattr(script.WorldBuilderEngine, "build", flaky_build)
    with caplog.at_level("WARNING", logger="tower.world_build_session"):
        code = script.main([
            "--synthetic", "--synthetic-frames", "10", "--rebuild-every", "2",
            "--root", str(tmp_path), "--format", "json",
        ])
    assert code == 0
    assert calls["n"] >= 2, "the session went on to rebuild again and to its final build"
    assert any("rebuild" in r.getMessage() and "PermissionError" in r.getMessage()
               for r in caplog.records), "the failure was said out loud"
    report = json.loads(capsys.readouterr().out)
    assert report["frames_observed"] == 10
    world_dir = next(p for p in (tmp_path / "worlds").iterdir() if p.is_dir())
    session = json.loads(next(world_dir.glob("sessions/*/session.json")).read_text(encoding="utf-8"))
    assert session["ended_at"] is not None and session["end_reason"] == "stop"
    assert session["keyframes_accepted"] == report["keyframes_accepted"] >= 2
    assert not (world_dir / "LOCK").exists()
    assert (world_dir / "derived" / "manifest.json").exists()
