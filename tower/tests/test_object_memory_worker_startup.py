"""A spawned Object Memory worker must reach its frame loop, and leave cleanly.

TWO PRODUCTION FAILURES, ONE ON EACH SIDE OF THE FRAME LOOP.

The physical run on 2026-09-06 recorded a healthy capture -- 244 frames,
~21 s, zero drops -- and the Object Memory worker attached to it observed
`frames_observed: 0`. It was not slow and it was not misattached; it was
deadlocked inside its own model load and never reached the frame loop.

Dissected with py-spy on this host: the main thread is loading OpenBLAS
(pulled in by `transformers` -> `scipy.linalg` while the OWLv2 verifier
loads) under the Windows loader lock, and the `object-memory-stop-watch`
daemon thread is blocked in a synchronous `ReadFile` on the stdin pipe the
supervisor holds. A thread parked in a blocking pipe read while the loader
brings up a DLL that creates threads is a hard hang: 0% CPU, forever.

It was invisible to the whole suite because every OTHER subprocess test
pins `--verifier none` to avoid downloading weights -- so nothing loaded
transformers/scipy in a spawned worker while the stdin watcher was armed.
Production defaults to `owlv2`. `scripts/object_memory_session.
_prewarm_native_libraries` warms that native stack single-threaded before
the watcher is armed, which is the fix these tests guard.

The second failure is on the way OUT. When a walk ends because the capture
closed rather than because anyone pressed Stop, the parent still holds the
pipe, so the stop-watcher is still blocked in `ReadFile`. Letting the
interpreter finalize with that thread blocked races CUDA/torch teardown and
access-violates (`0xC0000005`) -- after the flush and the report, so no
data is lost, but the worker exits with a crash code and the supervisor's
reaper logs a perfect walk as `EXITED 3221225477`. `_run_and_exit` leaves
the process with `os._exit` once the report is flushed, which is what the
gated test below asserts by exit code.
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


def _load_worker_module():
    """The worker script, imported in-process for the fast contract tests.

    It is a script rather than a package module, so it is loaded by path
    under a private name. The import is torch-free and costs ~0.1 s: every
    heavy dependency is loaded lazily inside `load()`, never at module
    scope, which is exactly what lets the two contracts below be asserted
    without a GPU.
    """
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / (
        "object_memory_session.py"
    )
    spec = importlib.util.spec_from_file_location(
        "object_memory_session_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loose_frames_dir(tmp_path, count=2):
    directory = tmp_path / "frames"
    directory.mkdir()
    image = np.full((16, 16, 3), 100, np.uint8)
    for index in range(count):
        (directory / f"{index:03d}.jpg").write_bytes(
            cv2.imencode(".jpg", image)[1].tobytes()
        )
    return directory


class TestTheNativeStackIsWarmedBeforeTheWatcher:
    """The ordering that keeps the loader-lock deadlock from forming.

    Asserted on the real `main`, with no models: the point is the ORDER of
    two calls, and a fake for each records when it happened. If a future
    edit moves the watcher ahead of the pre-warm, this fails without a GPU
    and without the 1.2 GB of weights the real deadlock needs to form.
    """

    def test_prewarm_runs_before_the_stdin_watcher_is_armed(
        self, tmp_path, monkeypatch
    ):
        module = _load_worker_module()
        order = []

        monkeypatch.setattr(
            module, "_prewarm_native_libraries", lambda: order.append("prewarm")
        )

        def fake_install(self, *, watch_stdin=False):
            # Record, and do NOT arm real signals or a real reader thread:
            # this test is about ordering, not about the channels, and it
            # runs on pytest's own main thread.
            order.append(("install", watch_stdin))

        monkeypatch.setattr(module._StopRequest, "install", fake_install)

        frames = _loose_frames_dir(tmp_path)
        code = module.main(
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

        assert code == 0
        assert order == ["prewarm", ("install", True)], order


class TestTheProcessLeavesHard:
    """`_run_and_exit` uses `os._exit`, and only on the success path.

    The exit is what stops a clean walk from looking like a crash. It is
    asserted at the seam rather than by spawning a GPU process, because the
    access violation it prevents only forms when a real CUDA context is
    being torn down with a thread blocked in the kernel -- which is the
    gated test's job.
    """

    def test_it_calls_os_exit_with_mains_return_code(self, monkeypatch):
        module = _load_worker_module()
        monkeypatch.setattr(module, "main", lambda: 0)

        exited = []

        def fake_exit(code):
            exited.append(code)
            raise SystemExit(code)  # stop control flow as the real _exit would

        monkeypatch.setattr(module.os, "_exit", fake_exit)

        with pytest.raises(SystemExit):
            module._run_and_exit()

        assert exited == [0]

    def test_a_non_int_return_becomes_a_zero_exit(self, monkeypatch):
        module = _load_worker_module()
        monkeypatch.setattr(module, "main", lambda: None)
        exited = []
        monkeypatch.setattr(
            module.os,
            "_exit",
            lambda code: (exited.append(code), (_ for _ in ()).throw(SystemExit))[1],
        )
        with pytest.raises(SystemExit):
            module._run_and_exit()
        assert exited == [0]


# The load-bearing case, opt-in: it needs the real ssdlite weights and the
# ~1.2 GB OWLv2 weights, and it is the only test that actually forms (and
# refuses to form) the loader-lock deadlock, because that needs a real
# transformers/scipy load in a spawned worker with the stdin watcher armed.
_MODEL_TESTS = os.environ.get("TOWER_RUN_MODEL_TESTS") == "1"


@pytest.mark.skipif(
    not _MODEL_TESTS,
    reason="spawns the real owlv2 worker (~1.2 GB weights); "
    "set TOWER_RUN_MODEL_TESTS=1 to run",
)
class TestTheRealOwlv2WorkerReachesItsFrameLoop:
    """The real reproduction: owlv2 + a held stdin pipe, the physical shape.

    Before `_prewarm_native_libraries` this deadlocked at 0% CPU and this
    test would time out. Before `_run_and_exit` it finished the walk and
    then exited `0xC0000005`. Both are asserted here against a real spawned
    process, which is the only place either failure is real.
    """

    def _open_capture_with_frames(self, tmp_path, count=6):
        directory = tmp_path / "captures" / "cap-owlv2"
        (directory / "frames").mkdir(parents=True)
        image = np.full((360, 640, 3), 120, np.uint8)
        lines = []
        for seq in range(1, count + 1):
            relpath = f"frames/{seq:08d}.jpg"
            (directory / relpath).write_bytes(
                cv2.imencode(".jpg", image)[1].tobytes()
            )
            lines.append(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_seq": seq,
                        "received_at": 1000.0 + seq,
                        "relpath": relpath,
                    }
                )
            )
        # No `capture.json`: a missing manifest reads as "still open", so
        # the follower polls after the frames -- the exact state a wearer
        # is in mid-walk, and the state the stop watcher must be armed
        # through.
        (directory / "frames.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return directory

    def test_it_observes_frames_and_exits_cleanly(self, tmp_path):
        capture = self._open_capture_with_frames(tmp_path)
        root = tmp_path / "om"
        process = subprocess.Popen(
            [
                sys.executable,
                "scripts/object_memory_session.py",
                "--follow-capture",
                str(capture),
                "--root",
                str(root),
                "--attach-mode",
                "from-start",
                "--verifier",
                "owlv2",
                "--verifier-device",
                "auto",
                "--device",
                "auto",
                "--max-idle-polls",
                "40",
                "--format",
                "json",
                "--stop-on-stdin-close",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
        try:
            # Generous: the load is ~8 s cold, and the assertion is "it got
            # past the load at all", not a latency measurement. Without the
            # pre-warm the worker never leaves the load and this waits the
            # whole window.
            deadline = time.time() + 90
            observed = None
            while time.time() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(2)
                try:
                    records = [
                        json.loads(line)
                        for line in (root / "observations.jsonl").read_text().splitlines()
                        if line.strip()
                    ]
                except (OSError, ValueError):
                    records = []
                # The frames are blank, so detections are not guaranteed;
                # what proves the deadlock is gone is that the process is
                # alive and past its load. Give it a moment, then close the
                # pipe and read the report, which carries frames_observed.
            process.stdin.close()
            stdout, stderr = process.communicate(timeout=60)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

        assert process.returncode == 0, (
            f"worker exited {process.returncode} "
            f"({hex(process.returncode & 0xffffffff)}); stderr tail:\n"
            + "\n".join(stderr.splitlines()[-15:])
        )
        report = json.loads(stdout)
        assert report["frames_observed"] > 0, (
            "the worker never reached its frame loop -- the load-lock "
            "deadlock has returned. stderr tail:\n"
            + "\n".join(stderr.splitlines()[-15:])
        )
        assert report["verifier"] == "owlv2"
