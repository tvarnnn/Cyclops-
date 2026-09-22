"""The World Builder workers must load their DLLs before they park a reader.

THE 95-MINUTE FINISH THAT WAS NEVER GOING TO FINISH.

On 2026-09-22 the Tower's recovery finisher, `scripts/world_finish_pending.py`,
took the writer lock of world `2f44716237544569b5f2faf782d9f877` and held it
for over ninety-five minutes against a job measured at roughly eight. It was
not slow. It burned **zero** CPU across repeated twenty-second samples, the
GPU sat at 1%, it wrote no log line and touched no artifact after the lock
and the first `status.json`, and the phone read "Finishing" for the whole
time because the lock arm has no staleness ceiling.

py-spy `--native` on the live process named it exactly:

    MainThread (idle)
        LdrLoadDll / LoadLibraryExW
        libscipy_openblas-197ee2fc9b4d071f7e048078cac74115.dll
        <module> (scipy/linalg/blas.py:247)
        ... moge/model/v2.py
        _load (tower/world_builder/dense.py:509)
        run_depth_stage -> ensure_depth_stage -> surfacify
        final_surface_stages -> finish -> main

    world-builder-stop-watch (idle)
        NtReadFile / ReadFile
        wait_for_close (scripts/world_build_session.py:529)

That is the SAME deadlock Object Memory hit on 2026-09-06 and fixed in its
own file: a DLL that spawns threads in `DllMain` cannot finish loading under
the Windows loader lock while another thread is parked in a blocking pipe
read. The fix had been written where the first bug was found, so World
Builder inherited the bug rather than the fix. `tower/native_prewarm.py` is
that fix extracted to one place; these tests are what keep both callers on
it.

The ordering tests below need no GPU and no weights: the defect is the ORDER
of two calls, and a fake for each records when it happened. The real
reproduction is gated at the bottom, because forming (and refusing to form)
the deadlock needs a real OpenBLAS load in a spawned process with the
watcher armed.
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

from tower.native_prewarm import (
    OBJECT_MEMORY_MODULES,
    WORLD_BUILDER_MODULES,
    prewarm,
)

TOWER_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_script(name):
    """A `scripts/` entry point, imported in-process under a private name.

    These are scripts, not package modules, so they load by path. Both are
    torch-free at module scope -- every heavy dependency is loaded lazily
    inside a stage -- which is what lets the ordering be asserted here
    without paying for any of it.
    """
    path = TOWER_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheWarmIsSharedAndHonest:
    """`tower/native_prewarm.py` itself: it warms, it reports, it never raises."""

    def test_it_imports_what_it_is_given(self):
        failed = prewarm(("json", "base64"), subsystem="Test")
        assert failed == ()
        assert "base64" in sys.modules

    def test_a_missing_library_is_a_warning_not_an_exception(self, capsys):
        """Warming is an optimisation of ORDER, never a dependency.

        A host that cannot import torch cannot run the stage that needed it
        either, and that stage must be allowed to say so in its own words --
        with the world id and the session in the message -- rather than
        through a crash at startup that names none of them.
        """
        failed = prewarm(
            ("tower_no_such_module_at_all",), subsystem="WorldBuilder"
        )
        assert failed == ("tower_no_such_module_at_all",)
        err = capsys.readouterr().err
        assert "could not pre-warm" in err
        assert "tower_no_such_module_at_all" in err
        # It has to point at the mechanism, because the symptom it prevents
        # -- a process at 0% CPU forever -- names nothing by itself.
        assert "native_prewarm" in err

    def test_the_world_builder_set_carries_the_dll_that_deadlocked(self):
        """`scipy.linalg` is the proven offender; it may never be dropped."""
        assert "scipy.linalg" in WORLD_BUILDER_MODULES
        assert "scipy.linalg" in OBJECT_MEMORY_MODULES
        # torch and cv2 ship their own thread-spawning runtimes and are
        # loaded by the same stages moments later.
        assert "torch" in WORLD_BUILDER_MODULES
        assert "cv2" in WORLD_BUILDER_MODULES


class TestTheFinisherWarmsBeforeItArms:
    """`scripts/world_finish_pending.py`: the process that actually hung."""

    def test_prewarm_runs_before_the_stdin_watcher_is_armed(
        self, tmp_path, monkeypatch
    ):
        module = _load_script("world_finish_pending.py")
        order = []

        monkeypatch.setattr(
            module, "prewarm_world_builder", lambda: order.append("prewarm")
        )

        def fake_install(self, *, watch_stdin=False):
            # Recorded, not performed: this test is about the order of the
            # two calls, and it runs on pytest's own main thread, where
            # arming real signal handlers and a real blocking reader would
            # be both pointless and hostile.
            order.append(("install", watch_stdin))

        monkeypatch.setattr(module.StopRequest, "install", fake_install)

        root = tmp_path / "world_builder"
        (root / "worlds").mkdir(parents=True)
        code = module.main(
            ["--root", str(root), "--stop-on-stdin-close", "--format", "json"]
        )

        assert code == 0
        assert order == ["prewarm", ("install", True)], order

    def test_the_warm_happens_even_without_the_stdin_flag(
        self, tmp_path, monkeypatch
    ):
        """A hand-run finisher gets the same ordering.

        `--stop-on-stdin-close` is the Tower's flag; a person running this
        by hand after a failed walk does not pass it. The warm is not
        conditional on it, because `install()` also arms signal handlers and
        because a rule with an exception is a rule that drifts.
        """
        module = _load_script("world_finish_pending.py")
        order = []
        monkeypatch.setattr(
            module, "prewarm_world_builder", lambda: order.append("prewarm")
        )
        monkeypatch.setattr(
            module.StopRequest,
            "install",
            lambda self, *, watch_stdin=False: order.append(
                ("install", watch_stdin)
            ),
        )
        root = tmp_path / "world_builder"
        (root / "worlds").mkdir(parents=True)

        module.main(["--root", str(root), "--format", "json"])

        assert order == ["prewarm", ("install", False)], order


class TestTheBuilderWarmsBeforeItArms:
    """`scripts/world_build_session.py` runs the same stages after Stop.

    It is the other process that calls `final_surface_stages`, so it loads
    the same DLLs with the same watcher armed. It had the same latent hang;
    it was only the finisher that was unlucky enough to hit it in front of
    somebody.
    """

    def test_prewarm_runs_before_the_stdin_watcher_is_armed(self, monkeypatch):
        module = _load_script("world_build_session.py")
        order = []

        monkeypatch.setattr(
            module, "prewarm_world_builder", lambda: order.append("prewarm")
        )
        monkeypatch.setattr(
            module.StopRequest,
            "install",
            lambda self, *, watch_stdin=False: order.append(
                ("install", watch_stdin)
            ),
        )

        # No frames, no capture: argparse refuses the combination straight
        # after the two calls under test, which is exactly far enough.
        with pytest.raises(SystemExit):
            module.main(["--stop-on-stdin-close"])

        assert order == ["prewarm", ("install", True)], order


class TestObjectMemoryStillWarmsThroughTheSharedModule:
    """The first subsystem keeps its fix while the code moves house."""

    def test_its_prewarm_delegates_and_still_runs_first(
        self, tmp_path, monkeypatch
    ):
        module = _load_script("object_memory_session.py")
        order = []
        monkeypatch.setattr(
            module, "prewarm_object_memory", lambda: order.append("prewarm")
        )
        monkeypatch.setattr(
            module._StopRequest,
            "install",
            lambda self, *, watch_stdin=False: order.append(
                ("install", watch_stdin)
            ),
        )

        # Real jpegs: `loose_frames` validates eagerly and exits if the
        # directory is empty, which would end `main` before the assertion
        # below could mean anything.
        frames = tmp_path / "frames"
        frames.mkdir()
        image = np.full((16, 16, 3), 100, np.uint8)
        for index in range(2):
            (frames / f"{index:03d}.jpg").write_bytes(
                cv2.imencode(".jpg", image)[1].tobytes()
            )

        module.main(
            [
                "--frames",
                str(frames),
                "--root",
                str(tmp_path / "om"),
                "--detector",
                "none",
                "--verifier",
                "none",
                "--stop-on-stdin-close",
                "--format",
                "json",
            ]
        )

        assert order[:2] == ["prewarm", ("install", True)], order


# The load-bearing case, opt-in. It is the only test that actually forms --
# and refuses to form -- the deadlock, because that needs a real OpenBLAS
# load in a SPAWNED process with a real pipe held open behind it. Everything
# above proves the ordering; this proves the ordering is the thing that
# matters.
_NATIVE_TESTS = os.environ.get("TOWER_RUN_MODEL_TESTS") == "1"


@pytest.mark.skipif(
    not _NATIVE_TESTS,
    reason="spawns a real interpreter that loads scipy/torch; "
    "set TOWER_RUN_MODEL_TESTS=1 to run",
)
class TestARealSpawnedWorkerGetsPastItsNativeLoad:
    """The physical shape: `stdin=PIPE` held open, the watcher armed, DLLs loaded.

    Before the warm, a process in this shape hung at 0% CPU forever on this
    host. The assertion is simply that it finishes at all inside a window
    far larger than the work: a timeout here IS the deadlock.
    """

    def test_the_finisher_completes_a_survey_with_its_watcher_armed(
        self, tmp_path
    ):
        root = tmp_path / "world_builder"
        (root / "worlds").mkdir(parents=True)

        process = subprocess.Popen(
            [
                sys.executable,
                "scripts/world_finish_pending.py",
                "--root",
                str(root),
                "--format",
                "json",
                "--stop-on-stdin-close",
            ],
            cwd=str(TOWER_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
        try:
            # The pipe stays OPEN for the whole run -- that is the half of
            # the deadlock the parent contributes, and closing it early
            # would test nothing. 180 s against a warm measured in single
            # digits: this is a liveness assertion, not a latency one.
            stdout, stderr = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            out, err = process.communicate()
            pytest.fail(
                "the finisher never returned from its native load with the "
                "stdin watcher armed -- the loader-lock deadlock is back. "
                "Dump it with `py-spy dump --pid <pid> --native` and look "
                "for LdrLoadDll on MainThread beside a ReadFile on "
                "world-builder-stop-watch.\nstderr tail:\n"
                + "\n".join(err.splitlines()[-20:])
            )
        finally:
            if process.poll() is None:  # pragma: no cover -- belt and braces
                process.kill()
                process.communicate()

        assert process.returncode == 0, (
            f"finisher exited {process.returncode}; stderr tail:\n"
            + "\n".join(stderr.splitlines()[-20:])
        )
        report = json.loads(stdout)
        # An empty root owes nothing; what is being asserted is that the
        # process got all the way to its report with the watcher armed.
        assert report["sessions_seen"] == 0
        assert report["finished"] == []

    def test_the_native_stack_is_actually_resident_after_the_warm(self):
        """The warm loads DLLs, not just `sys.modules` entries."""
        probe = (
            "import sys;"
            "from tower.native_prewarm import prewarm_world_builder;"
            "failed = prewarm_world_builder();"
            "import json;"
            "print(json.dumps({'failed': list(failed),"
            " 'scipy': 'scipy.linalg' in sys.modules,"
            " 'torch': 'torch' in sys.modules}))"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(TOWER_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        result = json.loads(out.stdout.strip().splitlines()[-1])
        assert result["failed"] == [], result
        assert result["scipy"] and result["torch"], result
