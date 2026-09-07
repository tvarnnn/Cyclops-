#!/usr/bin/env python
"""Boot a REAL Tower with every cartridge configured, and walk all of them.

`scripts/cv_lab_smoke.py` proved one cartridge against a Tower a person
had already started. This proves the UNIFICATION: that four cartridges
merged from four branches coexist in one process, declare themselves
together, hold their own lifecycles apart, and shut down without leaving
anything behind.

It starts its own uvicorn, so there is nothing to remember to run first
and no chance of testing yesterday's process. It configures every
cartridge -- world root, observation root, document root, scene on -- so
the declaration under test is the FULL one rather than the accidental
subset a developer happens to have in their .env.

    .venv\\Scripts\\python.exe scripts/unified_cartridge_smoke.py
    .venv\\Scripts\\python.exe scripts/unified_cartridge_smoke.py --with-models

Without `--with-models` it exercises every surface that does not require
loading a detector or an OCR reader: declaration, routing, Object Memory's
session lifecycle, the CV Lab's full start/pause/resume/stop, and the
result channel. That is the part worth running on every change, and it
takes seconds.

With `--with-models` it additionally starts and stops the Scene and
Document sessions, which load torch and easyocr. Slower, and the only way
to see Scene's discard-on-stop and Document's keep-on-stop actually
happen.

Exit code 0 means every check passed. A FAILURE here is a Tower problem;
this script never talks to a phone.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

TOWER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOWER_ROOT))

WIDTH, HEIGHT = 640, 360


class Checks:
    """Every assertion, its outcome, and a non-zero exit if any failed."""

    def __init__(self) -> None:
        self.failures = 0
        self.passes = 0

    def that(self, condition, description: str) -> bool:
        ok = bool(condition)
        if ok:
            self.passes += 1
            print(f"  ok    {description}")
        else:
            self.failures += 1
            print(f"  FAIL  {description}")
        return ok

    def report(self) -> int:
        total = self.passes + self.failures
        if self.failures:
            print(f"\n{self.failures} of {total} checks FAILED")
            return 1
        print(f"\nall {total} checks passed")
        return 0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _jpeg(seed: int) -> str:
    """A textured frame. Flat grey gives an experiment nothing to find,
    which makes a working Tower look broken."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=60)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _get(base: str, path: str, timeout: float = 20.0):
    """(status, decoded body). A non-200 is data here, not an exception --
    a 404 from an unconfigured route is one of the things under test."""
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, raw


def _post(base: str, path: str, timeout: float = 60.0):
    request = urllib.request.Request(base + path, method="POST", data=b"")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, raw


# -- the server ---------------------------------------------------------


def start_tower(port: int, roots: dict, log_path: Path) -> subprocess.Popen:
    """A real uvicorn, with every cartridge switched on.

    Started from TOWER_ROOT with PYTHONPATH set to it, because the venv
    in this repository installs `glasses-tower` editable from a DIFFERENT
    checkout -- so a child started anywhere else imports another branch's
    Tower and the smoke silently tests the wrong code.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TOWER_ROOT)
    env["TOWER_WORLD_ROOT"] = str(roots["world"])
    env["TOWER_OBSERVATION_ROOT"] = str(roots["observation"])
    env["TOWER_CAPTURE_ROOT"] = str(roots["capture"])
    env["TOWER_DOCUMENT_ROOT"] = str(roots["document"])
    # SEPARATE from the root, and deliberately so: a root with capture off
    # is a Tower that serves a library recorded elsewhere and records
    # nothing itself. `/documents` answers either way; `/documents-session`
    # answers 404 without this, naming the variable. Both halves are
    # switched on here because this smoke is about the FULL configuration.
    env["TOWER_DOCUMENT_CAPTURE"] = "true"
    env["TOWER_SCENE_UNDERSTANDING"] = "true"
    # PROCESS-GLOBAL, and set deliberately: uncapped, torch takes about
    # four cores for identical throughput on this workload, and a smoke
    # that pins the box is a smoke nobody runs.
    env["TOWER_SCENE_TORCH_THREADS"] = "2"
    # Neither session should follow the stream here. This script starts
    # and stops sessions explicitly, and an autostart would make it
    # impossible to tell a session it began from one the stream did.
    env["TOWER_SCENE_AUTOSTART"] = "false"
    env["TOWER_DOCUMENT_AUTOSTART"] = "false"
    # ON, which is the SHIPPED DEFAULT, so a real world-build worker
    # attaches when a capture opens and the shutdown check at the end has
    # something to be about.
    #
    # This was `false` in the first version of this script, and a reviewer
    # showed what that cost: with no autobuild and the object-memory
    # session stopped before the only capture opened, ZERO capture workers
    # ever existed -- so "no child process outlives the Tower", the one
    # check that would catch a leaked producer, passed because there was
    # nothing to leak. A green check that cannot fail is worse than no
    # check, and this script is the gate the Mac handoff tells people to
    # trust.
    env["TOWER_WORLD_AUTOBUILD"] = "true"

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tower.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(TOWER_ROOT),
        env=env,
        # To a FILE, never a pipe. The world-build worker is a CHILD of
        # this Tower and inherits its stdout, and it logs per keyframe.
        # With `subprocess.PIPE` and nobody reading, the OS buffer fills
        # at around 64 KB and the WORKER blocks on write -- so the Tower
        # blocks detaching it, and the whole thing wedges. That is not a
        # Tower defect; it is this script holding the other end of a pipe
        # it never reads. It only became reachable when autobuild was
        # switched on, which is the point of switching it on.
        stdout=log_path.open("w", encoding="utf-8", errors="replace"),
        stderr=subprocess.STDOUT,
        # Its own process group, so CTRL_BREAK_EVENT can be delivered to
        # it alone at shutdown. Without this the only way to stop it on
        # Windows is TerminateProcess, which runs no lifespan and orphans
        # every capture worker. See `check_worker_lifecycle_and_shutdown`.
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        ),
    )


def wait_for_health(
    base: str, process: subprocess.Popen, log_path: Path, timeout_s: float = 120.0
):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(
                f"Tower exited before serving (code {process.returncode}); "
                f"its log is at {log_path}"
            )
        try:
            status, body = _get(base, "/health", timeout=2.0)
            if status == 200:
                return body
        except Exception:
            time.sleep(0.25)
    raise SystemExit(f"Tower did not answer /health within {timeout_s}s")


# -- the checks ---------------------------------------------------------

EXPECTED_OFFERS = {
    "world_builder": "status",
    "experimental_cv": "status",
    "scene_understanding": "live",
    "document_memory": "status",
}


def check_declaration(base: str, checks: Checks) -> dict:
    print("\n[1] the capability declaration")
    status, body = _get(base, "/cartridges")
    checks.that(status == 200, "GET /cartridges answers 200")

    offered = {c["cartridge"]: c for c in body.get("cartridges", [])}
    for name, result_type in EXPECTED_OFFERS.items():
        entry = offered.get(name)
        if not checks.that(entry is not None, f"{name} is offered"):
            continue
        checks.that(
            entry["result_type"] == result_type,
            f"{name} offers result_type {result_type!r}",
        )
        checks.that(
            entry["available"] is True,
            f"{name} is available with its root/flag configured",
        )
    checks.that(
        set(offered) == set(EXPECTED_OFFERS),
        "exactly the four expected cartridges are offered",
    )
    checks.that(
        body.get("not_offered") == [],
        "not_offered is empty -- every contract in this build is offered",
    )
    http_contracts = {c["cartridge"] for c in body.get("http_contracts", [])}
    checks.that(
        "document_memory" in http_contracts,
        "document_memory.library is declared under http_contracts",
    )
    # The one deliberate gap. Asserted so that closing it becomes a
    # decision somebody makes rather than something that drifts in.
    checks.that(
        "object_memory" not in offered,
        "object_memory is NOT declared (deliberate; awaits the iOS lane)",
    )
    return body


def check_routes(base: str, checks: Checks) -> None:
    print("\n[2] every cartridge's routes are registered")
    for path in (
        "/health",
        "/cartridges",
        "/cv-lab",
        "/scene",
        "/documents",
        "/documents-session",
        "/cartridges/object_memory/session",
        "/object-memory/observations",
    ):
        status, body = _get(base, path)
        # 200 or 503 both prove the route EXISTS and answered. A 404 is
        # ambiguous on this Tower -- several routes answer 404 to mean
        # "not configured" rather than "no such route" -- so a 404 passes
        # only when it explains itself by naming the variable that would
        # fix it. A missing router cannot do that, which is the failure
        # this check is for.
        if status == 404:
            reason = json.dumps(body)
            checks.that(
                "TOWER_" in reason,
                f"{path} 404s as a CONFIGURATION answer that names its variable",
            )
        else:
            checks.that(True, f"{path} is registered (answered {status})")


def check_object_memory_session(base: str, checks: Checks) -> None:
    print("\n[3] Object Memory: the shared session lifecycle")
    status, body = _get(base, "/cartridges/object_memory/session")
    checks.that(status == 200, "GET session answers 200")
    checks.that(body.get("supported") is True, "this Tower supports the session")
    checks.that(body.get("state") == "stopped", "a fresh Tower starts stopped")
    checks.that(
        body.get("state_means") == "intent-not-liveness",
        "state is declared as intent, not liveness",
    )

    status, body = _post(base, "/cartridges/object_memory/session/start")
    checks.that(status == 200, "POST start answers 200")
    checks.that(body.get("state") == "active", "start moves to active")
    checks.that(body.get("changed") is True, "the first start reports changed")

    status, body = _post(base, "/cartridges/object_memory/session/start")
    checks.that(status == 200, "a second start is honoured, not refused")
    checks.that(
        body.get("changed") is False,
        "the second start reports changed:false -- idempotent, not an error",
    )

    status, body = _post(base, "/cartridges/object_memory/session/pause")
    checks.that(body.get("state") == "paused", "pause moves to paused")

    status, body = _post(base, "/cartridges/object_memory/session/resume")
    checks.that(body.get("state") == "active", "resume returns to active")

    status, body = _post(base, "/cartridges/object_memory/session/stop")
    checks.that(body.get("state") == "stopped", "stop returns to stopped")

    status, body = _post(base, "/cartridges/object_memory/session/stop")
    checks.that(
        status == 200 and body.get("state") == "stopped",
        "stop is never refused, even from stopped",
    )

    status, body = _post(base, "/cartridges/object_memory/session/resume")
    checks.that(
        status == 409,
        "resume from stopped is refused 409 -- it claims to continue something",
    )

    status, body = _get(base, "/cartridges/translator/session")
    checks.that(status == 404, "an unknown cartridge's session answers 404")


def check_no_cross_stop(base: str, checks: Checks) -> None:
    print("\n[4] coexistence: no cartridge stops another")
    _post(base, "/cartridges/object_memory/session/start")
    before = _get(base, "/cv-lab")[1]
    run_before = (before.get("status") or {}).get("lifecycle", {}).get("run_id")

    # Scene's control surface is exercised whether or not a model loads:
    # a stop on a never-started session must still not disturb anything.
    _post(base, "/scene/stop")
    _post(base, "/documents-session/stop")

    after_session = _get(base, "/cartridges/object_memory/session")[1]
    checks.that(
        after_session.get("state") == "active",
        "stopping Scene and Document leaves Object Memory active",
    )
    after = _get(base, "/cv-lab")[1]
    run_after = (after.get("status") or {}).get("lifecycle", {}).get("run_id")
    checks.that(
        run_before == run_after,
        "stopping Scene and Document does not disturb the CV Lab run",
    )
    _post(base, "/cartridges/object_memory/session/stop")


def check_scene_and_document_models(base: str, checks: Checks) -> None:
    print("\n[5] Scene and Document sessions (models load here)")
    status, body = _post(base, "/scene/start", timeout=180.0)
    checks.that(status == 200, "POST /scene/start answers 200")

    deadline = time.monotonic() + 180.0
    state = None
    while time.monotonic() < deadline:
        body = _get(base, "/scene")[1]
        state = body.get("lifecycle", {}).get("state")
        if state in ("running", "failed"):
            break
        time.sleep(0.5)
    checks.that(state == "running", f"the scene session reaches running (saw {state!r})")

    body = _get(base, "/scene")[1]
    checks.that(
        body.get("count_is_lower_bound") is True,
        "counts are published as a LOWER BOUND",
    )
    checks.that(body.get("tracks") is None, "tracks is null -- no entity ever ships")
    checks.that(
        body.get("relations") is None, "relations is null -- structurally absent"
    )

    _post(base, "/scene/stop")
    body = _get(base, "/scene")[1]
    checks.that(
        body.get("scene_available") is False,
        "Stop DISCARDS the scene: scene_available goes false",
    )
    checks.that(body.get("counts") is None, "Stop discards counts")
    checks.that(
        bool(body.get("scene_unavailable_reason")),
        "the discard says why, rather than going silent",
    )

    print("\n[6] Document: Stop KEEPS what was recorded")
    status, _ = _post(base, "/documents-session/start", timeout=180.0)
    checks.that(status == 200, "POST /documents-session/start answers 200")
    deadline = time.monotonic() + 180.0
    state = None
    while time.monotonic() < deadline:
        body = _get(base, "/documents-session")[1]
        state = (body.get("session") or {}).get("state")
        if state in ("running", "failed"):
            break
        time.sleep(0.5)
    checks.that(
        state == "running", f"the document session reaches running (saw {state!r})"
    )

    before = _get(base, "/documents")[1]
    _post(base, "/documents-session/stop")
    after = _get(base, "/documents")[1]
    checks.that(
        after.get("document_count") == before.get("document_count"),
        "Stop keeps the library: document_count is unchanged",
    )
    checks.that(
        after.get("answer") in ("matched", "not_found", "no_observation"),
        "the library answers with the closed vocabulary, never an exception",
    )


# -- the socket ---------------------------------------------------------


async def _drain(ws, expect, limit: int = 60, timeout: float = 30.0) -> dict:
    for _ in range(limit):
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        message = json.loads(raw)
        if message.get("type") == expect:
            return message
    raise AssertionError(f"never saw a {expect!r} in {limit} messages")


async def check_socket(base_ws: str, http_declaration: dict, checks: Checks) -> None:
    import websockets

    print("\n[7] the socket: declaration parity, CV Lab control, subscriptions")
    async with websockets.connect(base_ws, max_size=8 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "cartridges"}))
        over_socket = await _drain(ws, "cartridges")
        checks.that(
            json.dumps(over_socket, sort_keys=True)
            == json.dumps(http_declaration, sort_keys=True),
            "socket {'type':'cartridges'} is byte-identical to GET /cartridges",
        )

        await ws.send(json.dumps({"type": "stream_start"}))

        await ws.send(
            json.dumps(
                {
                    "type": "cv_lab_start",
                    "experiment_id": "edge_detection",
                    "request_id": "smoke-1",
                }
            )
        )
        reply = await _drain(ws, "cv_lab_status")
        checks.that(
            reply.get("request_id") == "smoke-1", "the command reply echoes request_id"
        )
        run_id = reply["status"]["lifecycle"]["run_id"]
        checks.that(bool(run_id), "a start mints a run_id")

        deadline = time.monotonic() + 180.0
        seen_result = None
        while time.monotonic() < deadline and seen_result is None:
            await ws.send(
                json.dumps(
                    {
                        "type": "frame",
                        "seq": 1,
                        "width": WIDTH,
                        "height": HEIGHT,
                        "format": "jpeg",
                        "data": _jpeg(1),
                    }
                )
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            message = json.loads(raw)
            if message.get("type") == "frame_result":
                seen_result = message
            elif message.get("type") == "frame_error":
                # `cv_lab_starting` is the documented arming window, not a
                # failure. Anything else is.
                if message.get("reason") != "cv_lab_starting":
                    checks.that(False, f"unexpected frame_error {message!r}")
                    break
                time.sleep(0.2)

        if checks.that(seen_result is not None, "a frame produces a frame_result"):
            provenance = seen_result.get("cv_lab") or {}
            checks.that(
                provenance.get("run_id") == run_id,
                "the result carries the run_id of the run that produced it",
            )
            checks.that(
                provenance.get("experiment_id") == "edge_detection",
                "the result names the experiment that produced it",
            )
            checks.that(
                provenance.get("provenance") in ("measured", "inferred"),
                "every result declares measured or inferred provenance",
            )

        await ws.send(json.dumps({"type": "cv_lab_stop"}))
        await _drain(ws, "cv_lab_status")

        for cartridge, result_type in EXPECTED_OFFERS.items():
            await ws.send(
                json.dumps(
                    {
                        "type": "result_subscribe",
                        "cartridge": cartridge,
                        "result_type": result_type,
                    }
                )
            )
            reply = await _drain(ws, "result_subscribed")
            checks.that(
                reply.get("cartridge") == cartridge,
                f"a subscription to {cartridge}/{result_type} is accepted",
            )

        await ws.send(
            json.dumps(
                {
                    "type": "result_subscribe",
                    "cartridge": "object_memory",
                    "result_type": "status",
                }
            )
        )
        reply = await _drain(ws, "result_error")
        checks.that(
            reply.get("reason") == "unknown_cartridge",
            "object_memory is refused on the socket -- the deliberate gap holds",
        )

        await ws.send(json.dumps({"type": "stream_stop"}))


# -- teardown -----------------------------------------------------------


async def check_worker_lifecycle_and_shutdown(process, port: int, checks: Checks):
    """Open a capture, prove a worker attached, then shut down and look.

    THE POINT OF THIS FUNCTION IS THAT IT CAN FAIL.

    The first version of this script asserted "no child process outlives
    the Tower" after a run in which no capture worker had ever existed --
    autobuild was off and the object-memory session was stopped before the
    only capture opened. The assertion passed because there was nothing to
    leak. A reviewer caught it, and it is the exact failure mode this
    script exists to catch in others.

    So: hold a stream open, wait for a real `world_build_session.py` child
    to appear, and only then bring the Tower down -- with the connection
    still open, which is the state a wearer walking out of range leaves it
    in.

    On Windows the shutdown signal matters as much as the check.
    `Popen.terminate()` is `TerminateProcess`, which delivers nothing a
    process can handle: uvicorn's lifespan never runs, so
    `CaptureWorkerSupervisor.shutdown()` never runs, and every child is
    orphaned. That is not a defect in the Tower -- the teardown is correct
    and was measured killing a real child -- it is a defect in how the
    smoke asked it to stop. CTRL_BREAK_EVENT is the signal uvicorn
    installs a handler for, and it needs the process to have been started
    in its own process group.
    """
    import websockets

    try:
        import psutil
    except ImportError:
        psutil = None

    print("\n[8] a capture worker attaches, and does not outlive the Tower")

    # Wait for the Tower to be answering again before opening a socket.
    # The previous section's `stream_stop` detaches the world-build worker
    # that section attached, and `detach` gives a follower a 10 s grace
    # before terminating it -- which is exactly the "shutdown can cost ~24 s"
    # behaviour the integration report records. A default 10 s handshake
    # lands inside that window and times out on a Tower that is perfectly
    # healthy, so this waits rather than accusing it.
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            if _get(base, "/health", timeout=2.0)[0] == 200:
                break
        except Exception:
            pass
        await asyncio.sleep(0.5)

    async with websockets.connect(
        f"ws://127.0.0.1:{port}/ws",
        max_size=8 * 1024 * 1024,
        open_timeout=60,
    ) as ws:
        # THE GATE FIRST, and its absence proved before it is opened.
        #
        # Until 2026-09-06 a builder attached to EVERY capture, so this
        # section opened a stream and waited. It now attaches only while the
        # World Builder cartridge session is active -- the session the phone
        # opens when that workspace appears -- which is what stopped a CV Lab
        # camera session from starting background global solves nobody asked
        # for. A stream on its own must therefore attach NOTHING, and that is
        # worth asserting rather than assuming: it is the whole of the gate.
        await ws.send(json.dumps({"type": "stream_start"}))

        if psutil is not None:
            await asyncio.sleep(2.0)
            ungated = _world_build_children(process, psutil)
            checks.that(
                not ungated,
                "a stream with no World Builder session attaches NO builder "
                f"(found {[w.pid for w in ungated]})",
            )

        status, _ = _post(base, "/cartridges/world_builder/session/start")
        checks.that(
            status == 200, "the World Builder cartridge session starts over HTTP"
        )

        workers = []
        if psutil is not None:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                workers = _world_build_children(process, psutil)
                if workers:
                    break
                await asyncio.sleep(0.25)
            checks.that(
                bool(workers),
                "a world-build worker attached once the cartridge was active "
                f"(found {[w.pid for w in workers]})",
            )
        else:
            print("  skip  psutil is not installed; worker attachment not checked")

        # Down it goes, with the capture still open and the socket still
        # connected -- a wearer walking out of range, not a tidy stop.
        exited = _signal_shutdown(process)
        # Says only what it proves. Exiting is not evidence the lifespan
        # ran -- `TerminateProcess` also makes a process exit, and runs
        # nothing. The check below is the one that can tell the
        # difference, because a lifespan that did not run leaves the
        # workers behind.
        checks.that(exited, "the Tower exits when asked, rather than needing a kill")

        if psutil is not None and workers:
            await asyncio.sleep(1.0)
            alive = [w for w in workers if w.is_running()]
            checks.that(
                not alive,
                "no capture worker outlives the Tower "
                f"(still running: {[w.pid for w in alive]})",
            )


def _world_build_children(process, psutil):
    """Every live world-build child of this Tower, by command line."""
    try:
        return [
            child
            for child in psutil.Process(process.pid).children(recursive=True)
            if "world_build_session" in " ".join(child.cmdline())
        ]
    except Exception:
        return []


def _signal_shutdown(process) -> bool:
    """Ask the Tower to stop in a way its lifespan will actually see."""
    if sys.platform == "win32":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception:
            process.terminate()
    else:
        process.terminate()
    try:
        process.wait(timeout=90)
        return True
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        return False


def reap(process) -> None:
    """Last resort, in the `finally`. Never asserts."""
    if process.poll() is None:
        try:
            process.kill()
            process.wait(timeout=30)
        except Exception:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-models",
        action="store_true",
        help="also start and stop the Scene and Document sessions (slow: loads torch and easyocr)",
    )
    parser.add_argument("--keep-roots", action="store_true", help="do not delete the temp roots")
    args = parser.parse_args(argv)

    import tempfile

    print(f"tower package : {TOWER_ROOT / 'tower'}")
    workdir = Path(tempfile.mkdtemp(prefix="unified-smoke-"))
    roots = {
        name: workdir / name for name in ("world", "observation", "capture", "document")
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    checks = Checks()
    log_path = workdir / "tower.log"
    process = start_tower(port, roots, log_path)
    try:
        health = wait_for_health(base, process, log_path)
        print(f"Tower up on {base} (health: {json.dumps(health)[:120]}...)")

        declaration = check_declaration(base, checks)
        check_routes(base, checks)
        check_object_memory_session(base, checks)
        check_no_cross_stop(base, checks)
        if args.with_models:
            check_scene_and_document_models(base, checks)
        else:
            print("\n[5,6] Scene and Document sessions SKIPPED (--with-models to run)")
        asyncio.run(check_socket(f"ws://127.0.0.1:{port}/ws", declaration, checks))
        asyncio.run(check_worker_lifecycle_and_shutdown(process, port, checks))
    finally:
        reap(process)

    code = checks.report()
    if code != 0 or args.keep_roots:
        print(f"\nthe Tower's log: {log_path}")
    elif not args.keep_roots:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
