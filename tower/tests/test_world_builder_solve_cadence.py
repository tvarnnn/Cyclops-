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


def test_wait_terminates_a_child_that_outlives_its_bound(tmp_path, monkeypatch):
    """Until 2026-09-06 `wait()` ABANDONED a slow child and the final solve
    then reused its workspace underneath it. It now terminates the child:
    the builder owns every solve process it starts, and leaves none behind
    (the 09-06 walk left one writing a solution nobody merged)."""
    stub = tmp_path / "slow.py"
    stub.write_text("import time\ntime.sleep(30)\n")
    solver = BackgroundSolver(root=tmp_path, world_id="w", session_id="s", every=1, capture_dirs=[])
    import subprocess

    child = subprocess.Popen([sys.executable, str(stub)])
    solver._child = child
    started = time.perf_counter()
    assert solver.wait(0.2) is False
    assert time.perf_counter() - started < 10.0
    assert child.poll() is not None, "the child must be gone, not abandoned"
    assert solver.running is False


def test_close_leaves_no_child_behind(tmp_path):
    stub = tmp_path / "slow.py"
    stub.write_text("import time\ntime.sleep(30)\n")
    solver = BackgroundSolver(root=tmp_path, world_id="w", session_id="s", every=1, capture_dirs=[])
    import subprocess

    child = subprocess.Popen([sys.executable, str(stub)])
    solver._child = child
    solver.close()
    assert child.poll() is not None
    assert solver.child_pid is None


def test_wait_ends_early_on_a_stop_request(tmp_path):
    stub = tmp_path / "slow.py"
    stub.write_text("import time\ntime.sleep(30)\n")
    solver = BackgroundSolver(root=tmp_path, world_id="w", session_id="s", every=1, capture_dirs=[])
    import subprocess

    child = subprocess.Popen([sys.executable, str(stub)])
    solver._child = child
    started = time.perf_counter()
    assert solver.wait(60.0, should_stop=lambda: True) is False
    assert time.perf_counter() - started < 10.0
    assert child.poll() is not None


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


def test_solve_children_are_spawned_as_one_owned_process(tmp_path):
    """The solve child must be ONE process, the pid the builder holds.

    On Windows `sys.executable` inside the venv is a launcher that spawns
    the real interpreter underneath it with silent breakaway, so a child
    started that way is two processes and the builder's `terminate()`
    reaches only the outer one. The 2026-09-06 walk's orphaned solver was
    that grandchild. The recipe in `tower.process_ownership` gives one
    process, and the builder must use both halves of it: the executable
    AND the environment that makes it venv-aware.
    """
    from scripts.world_build_session import python_executable
    from tower.process_ownership import interpreter_environment, interpreter_executable

    spawned = []

    class _Child:
        pid = 4321

        def poll(self):
            return None

    def spawn(argv, **kwargs):
        spawned.append((list(argv), kwargs))
        return _Child()

    solver = BackgroundSolver(
        root=tmp_path, world_id="w", session_id="s", every=1, capture_dirs=[],
        spawn=spawn,
    )
    solver.maybe_launch(_Store(tmp_path), accepted=5, sources={})

    assert python_executable() == interpreter_executable()
    argv, kwargs = spawned[0]
    assert argv[0] == interpreter_executable()
    assert kwargs.get("env") == interpreter_environment()
    solver.close()


class TestLoopDetectionIsALiveSettingNotAFinalisationOne:
    """Every solve looks for revisits, not just the last one.

    Sequential matching reaches `SEQUENTIAL_OVERLAP = 20` keyframes either
    side and no further, so a solve without loop detection can only chain
    forwards. It cannot discover that the wearer walked back into a room it
    already mapped, and the world therefore comes APART as the walk goes on
    rather than together.

    Measured on the 2026-09-09 capture at the field run's own solve
    horizons -- sequential only, then the same keyframes re-solved in one
    workspace with loop detection:

        horizon   components (seq -> loop)   largest component's share
           156          6 -> 3                        0.75
           311         11 -> 6                        0.77
           526         14 -> 5                        0.90
           646         16 -> 6                        0.91
           795          -    5                        0.95

    The cost is 1.2-1.9x, paid in matching, which is incremental: pairs
    already tested stay in the database.
    """

    def _argv(self, tmp_path, *, final: bool) -> list[str]:
        solver = BackgroundSolver(
            root=tmp_path, world_id="w", session_id="s", every=10, capture_dirs=[],
        )
        return solver._argv(final=final)

    def test_a_background_solve_asks_for_loop_detection(self, tmp_path):
        assert "--loop-detection" in self._argv(tmp_path, final=False)

    def test_a_background_solve_is_still_not_a_final_one(self, tmp_path):
        """`--final` remains the finalisation solve's own distinction.

        It is the one solve that sees every keyframe, including the tail no
        live solve reached -- on the field walk, 149 of 795.
        """
        assert "--final" not in self._argv(tmp_path, final=False)

    def test_the_final_solve_asks_for_both(self, tmp_path):
        argv = self._argv(tmp_path, final=True)
        assert "--loop-detection" in argv
        assert "--final" in argv

    def test_the_flag_is_passed_once(self, tmp_path):
        """The final solve used to add both flags together; adding
        `--loop-detection` unconditionally must not double it."""
        assert self._argv(tmp_path, final=True).count("--loop-detection") == 1


class TestTheRebuildIntervalGrowsWithTheWorld:
    """`--rebuild-every 4` is a fixed count; the rebuild is not a fixed cost.

    THE FIRST VERSION OF THIS MODELLED THE WRONG OPERATION. It costed
    `write_derived` -- 0.34 s at 795 keyframes -- and concluded the crossover
    was about 2,700 keyframes, roughly fourteen minutes. The loop does not
    call `write_derived`; it calls `engine.build()`, which is that write plus
    the merge, the placement pass and the manifest.

    Measured over the 204 rebuilds of a replay of the real 2026-09-09
    capture, against that walk's own arrival rate of 3.23 keyframes/second:

        keyframes     mean build     share at interval 4
          1- 200        0.141 s            11.4%
        201- 400        0.386 s            31.1%
        401- 600        0.585 s            47.2%
        601- 800        1.006 s            81.2%
        801-1000        0.895 s            72.2%

    81% where the model said 27%, and it is reached at about four minutes of
    walking rather than fourteen. The old knee also arrived a doubling late:
    `(accepted // 750).bit_length() - 1` is zero below 1500, so nothing
    widened until long after the builder had stopped keeping up.
    """

    def test_a_short_walk_is_unchanged(self):
        from scripts.world_build_session import rebuild_interval

        for accepted in (2, 100, 400, 599):
            assert rebuild_interval(4, accepted) == 4, accepted

    def test_the_walk_that_started_this_campaign_now_widens(self):
        """795 keyframes used to be explicitly exempted -- "nothing about the
        2026-09-09 walk moves" -- on a model that undercounted the rebuild by
        three times. At 795 the builder is measured at 81% of wall clock. It
        is the case that most needs the widening, not the one to protect from
        it."""
        from scripts.world_build_session import rebuild_interval

        assert rebuild_interval(4, 795) == 8

    def test_the_interval_doubles_as_the_world_doubles(self):
        from scripts.world_build_session import rebuild_interval

        assert rebuild_interval(4, 600) == 4
        assert rebuild_interval(4, 601) == 8
        assert rebuild_interval(4, 1200) == 16
        assert rebuild_interval(4, 2400) == 32
        assert rebuild_interval(4, 4800) == 64

    def test_the_rebuild_stays_a_bounded_share_of_the_walk(self):
        """The property, not the numbers: rebuilding must not outgrow walking.

        Costed on `engine.build()`, which is what the loop pays. The measured
        rows are the replay's own; the larger two extrapolate its slope
        (~0.0013 s per keyframe of world), and are the sizes a 20-30 minute
        walk reaches. With a fixed four the share runs 81% at 795 and past
        100% before 1,200 -- a builder that can never catch up.
        """
        from scripts.world_build_session import rebuild_interval

        measured_build_seconds = {700: 1.006, 900: 0.895, 1500: 1.95,
                                  3000: 3.90, 6000: 7.80}
        keyframes_per_second = 3.23
        for accepted, build in measured_build_seconds.items():
            interval = rebuild_interval(4, accepted)
            share = build / (interval / keyframes_per_second)
            assert share < 0.6, (
                f"at {accepted} keyframes the rebuild is {share:.0%} of the "
                "wall clock; the builder cannot keep up"
            )

    def test_a_fixed_interval_would_fail_that_property(self):
        """So the property above is not vacuously true of any schedule."""
        keyframes_per_second = 3.23
        share = 1.006 / (4 / keyframes_per_second)
        assert share > 0.6, (
            "the fixed interval this replaced would have passed; the "
            "property is measuring nothing"
        )

    def test_it_stops_doubling_rather_than_running_away(self):
        """A cap, so a pathological session cannot stop rebuilding entirely."""
        from scripts.world_build_session import rebuild_interval

        assert rebuild_interval(4, 10**6) == 4 << 4

    def test_a_zero_base_still_means_never(self):
        """`--rebuild-every 0` is "build once at the end" and must stay so.
        The caller guards on `args.rebuild_every` being truthy, but this
        must not turn a 0 into a positive interval if that ever changes."""
        from scripts.world_build_session import rebuild_interval

        assert rebuild_interval(0, 6000) == 0
