"""The builder's background solve: cadence, one child at a time, and the
finalisation hand-off. No pycolmap, no images: the child is a stand-in
command, so what is tested is the supervision, not the solving."""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.world_build_session import BackgroundSolver  # noqa: E402
from tower.world_builder import global_solve  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402


class _Store:
    """Just enough of WorldStore for write_sources/workspace_for."""

    def __init__(self, root: Path):
        self._root = root

    def world_dir(self, world_id: str) -> Path:
        return self._root / "worlds" / world_id


@pytest.fixture
def solver(tmp_path, monkeypatch):
    # Replace the real child with a short sleep so the cadence is observable.
    stub = tmp_path / "stub_solve.py"
    stub.write_text("import sys, time\ntime.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 0.2)\n")
    original = BackgroundSolver.maybe_launch

    def launch_stub(self, store, accepted, sources):
        import subprocess

        self._reap()
        if self.running or accepted - self._launched_at_keyframes < self.every or accepted < 2:
            return False
        global_solve.write_sources(store, self.world_id, self.session_id, sources)
        workspace = global_solve.workspace_for(store, self.world_id, self.session_id)
        workspace.root.mkdir(parents=True, exist_ok=True)
        self._log = open(workspace.root / "solve.log", "ab")
        self._child = subprocess.Popen(
            [sys.executable, str(stub), "0.4"], stdout=self._log, stderr=subprocess.STDOUT
        )
        self._launched_at_keyframes = accepted
        self._launches += 1
        return True

    monkeypatch.setattr(BackgroundSolver, "maybe_launch", launch_stub)
    return BackgroundSolver(
        root=tmp_path, world_id="w", session_id="s", every=10, capture_dirs=[],
    )


def test_launches_only_every_n_keyframes_and_one_at_a_time(solver, tmp_path):
    store = _Store(tmp_path)
    assert solver.maybe_launch(store, accepted=1, sources={}) is False   # too few
    assert solver.maybe_launch(store, accepted=10, sources={"k": "p"}) is True
    assert solver.running
    assert solver.maybe_launch(store, accepted=25, sources={}) is False  # one at a time
    assert solver.launches == 1
    assert solver.wait(10.0) is True
    assert solver.finished() is True
    assert solver.finished() is False                                    # reported once
    assert solver.maybe_launch(store, accepted=15, sources={}) is False  # only 5 since last launch
    assert solver.maybe_launch(store, accepted=20, sources={}) is True
    solver.wait(10.0)
    assert solver.launches == 2


def test_sources_are_written_before_the_child_starts(solver, tmp_path):
    store = _Store(tmp_path)
    solver.maybe_launch(store, accepted=10, sources={"s:00000001": "C:/frames/1.jpg"})
    workspace = global_solve.workspace_for(store, "w", "s")
    recorded = json.loads((workspace.root / "sources.json").read_text())["sources"]
    assert recorded == {"s:00000001": "C:/frames/1.jpg"}
    solver.wait(10.0)


def test_wait_reports_an_abandoned_child(tmp_path, monkeypatch):
    stub = tmp_path / "slow.py"
    stub.write_text("import time\ntime.sleep(3)\n")
    solver = BackgroundSolver(root=tmp_path, world_id="w", session_id="s", every=1, capture_dirs=[])
    import subprocess

    solver._child = subprocess.Popen([sys.executable, str(stub)])
    started = time.perf_counter()
    assert solver.wait(0.2) is False
    assert time.perf_counter() - started < 2.5
    solver._child.kill()
    solver._child.wait()


def test_a_missing_solver_is_reported_not_raised(tmp_path):
    """`solve()` without pycolmap returns a reason; it never takes the build down."""
    store = WorldStore(tmp_path)
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "pycolmap":
            raise ImportError("absent")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = refuse
    try:
        summary = global_solve.solve(store, "w", "s")
    finally:
        builtins.__import__ = real_import
    assert summary["solved"] is False
    assert "pycolmap" in summary["reason"]
