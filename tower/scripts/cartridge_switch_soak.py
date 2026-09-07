#!/usr/bin/env python
"""World Builder -> CV Lab -> World Builder, on ONE Tower, many times.

The requirement this measures (2026-09-06, after the first physical test of
the combined stack): the Tower is a long-lived host. Switching cartridges
from the phone must never need a Tower restart, an environment variable,
a killed worker or an app restart -- and repeated switching must not leak
processes, threads or memory, must not leave a builder following a camera
session another cartridge started, and must not leave a world locked by a
process that is gone.

What one cycle does, over the same `/ws` the phone uses plus the two HTTP
session routes the phone calls:

    POST /cartridges/world_builder/session/start    (the workspace appeared)
    stream_start ; N frames                          (the wearer walks)
      -> a builder must be following THIS capture
    stream_stop                                      (the wearer's Stop)
      -> the builder finalizes and exits on its own
    POST /cartridges/world_builder/session/stop     (the workspace left)
    cv_lab_start baseline ; stream_start ; N frames  (the Lab)
      -> NO builder may attach to this capture
    cv_lab_stop ; stream_stop

After every step the Tower process is sampled from the outside (threads,
RSS, handles, descendants by command line) and the world root is checked
for locks naming dead pids and sessions left open. The verdict compares
the last third of cycles with the first, as `cv_lab_switch_soak.py` does,
and additionally FAILS on any of: a builder attached during a CV Lab
capture, a `world_build_session`/`world_solve` process alive after the
last cycle settled, a LOCK with a dead pid, a session with `ended_at`
null, or a Tower that does not stop cleanly.

It starts its own uvicorn from this checkout (`--attach-port` uses one
you started, and then it cannot see children and says so). Frames are
synthetic; with no calibration in the world root the builder runs the
unposed backend, which is the cheapest walk that still exercises every
process boundary here. `--intrinsics-from DIR` copies a real calibration
in so the solver children run for real.

    .venv\\Scripts\\python.exe scripts/cartridge_switch_soak.py --cycles 5 --frames 24
"""

import argparse
import asyncio
import base64
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.process_ownership import (  # noqa: E402
    interpreter_command,
    interpreter_environment,
)

TOWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("data/soak/cartridge_switch")
WIDTH, HEIGHT = 360, 640
WORKER_MARKERS = ("world_build_session", "world_solve")
WB_SESSION = "/cartridges/world_builder/session"


# -- frames --------------------------------------------------------------


def build_frames(count: int = 8) -> list[str]:
    frames = []
    for seed in range(count):
        rng = np.random.default_rng(seed + 11)
        array = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=60)
        frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return frames


# -- HTTP ----------------------------------------------------------------


def _get(base: str, path: str, timeout: float = 20.0):
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


# -- the Tower -----------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_tower(port: int, root: Path, log_path: Path) -> subprocess.Popen:
    env = interpreter_environment()
    env["PYTHONPATH"] = str(TOWER_ROOT)
    env["TOWER_CAPTURE_ROOT"] = str(root / "capture")
    env["TOWER_WORLD_ROOT"] = str(root / "world")
    env["TOWER_WORLD_AUTOBUILD"] = "true"
    env["TOWER_WORLD_REBUILD_EVERY"] = "4"
    env["TOWER_SCENE_AUTOSTART"] = "false"
    env["TOWER_DOCUMENT_AUTOSTART"] = "false"
    env.pop("TOWER_OBSERVATION_ROOT", None)
    return subprocess.Popen(
        [
            *interpreter_command("-m", "uvicorn", "tower.main:app"),
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "info",
        ],
        cwd=str(TOWER_ROOT),
        env=env,
        # A FILE, never a pipe: workers inherit this stdout and log per
        # rebuild; a pipe nobody drains would wedge them.
        stdout=log_path.open("w", encoding="utf-8", errors="replace"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        ),
    )


def wait_for_health(base: str, process, log_path: Path, timeout_s: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise SystemExit(f"Tower exited before serving; log at {log_path}")
        try:
            status, body = _get(base, "/health", timeout=2.0)
            if status == 200:
                return body
        except Exception:
            time.sleep(0.25)
    raise SystemExit(f"Tower did not answer /health within {timeout_s}s")


def stop_tower(process) -> bool:
    if process.poll() is not None:
        return True
    if sys.platform == "win32":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception:
            process.terminate()
    else:
        process.terminate()
    try:
        process.wait(timeout=120)
        return True
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        return False


# -- sampling ------------------------------------------------------------


def _mb(value) -> float:
    return round(value / 1048576, 1)


def sample(tower_pid: int | None, world_root: Path) -> dict:
    """The Tower from the outside, plus every builder/solver on the box.

    Descendants come from the Tower pid when we own it. The process-table
    scan is there for `--attach-port`, and for the one thing descendants
    cannot show: a worker that has been reparented because the process
    between it and the Tower died.
    """
    out = {"threads": None, "rss_mb": None, "handles": None, "descendants": [], "strays": []}
    if tower_pid is not None:
        try:
            proc = psutil.Process(tower_pid)
            out["threads"] = proc.num_threads()
            out["rss_mb"] = _mb(proc.memory_info().rss)
            if hasattr(proc, "num_handles"):
                out["handles"] = proc.num_handles()
            for child in proc.children(recursive=True):
                out["descendants"].append(_describe(child))
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
    known = {d["pid"] for d in out["descendants"]}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(marker in cmdline for marker in WORKER_MARKERS) and str(world_root) in cmdline:
            if proc.info["pid"] not in known:
                out["strays"].append({"pid": proc.info["pid"], "cmdline": cmdline[-100:]})
    out["locks"] = lock_report(world_root)
    return out


def _describe(child: psutil.Process) -> dict:
    try:
        cmdline = " ".join(child.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        cmdline = "?"
    kind = next((m for m in WORKER_MARKERS if m in cmdline), "other")
    return {"pid": child.pid, "kind": kind, "cmdline": cmdline[-100:]}


def lock_report(world_root: Path) -> dict:
    """Locks naming dead pids, and sessions left open, under the world root."""
    from tower.world_builder.store import WorldStore

    report = {"dead_locks": [], "live_locks": [], "open_sessions": [], "interrupted": [], "complete": 0}
    worlds = world_root / "worlds"
    if not worlds.is_dir():
        return report
    store = WorldStore(world_root)
    for world_id in store.list_world_ids():
        holder = store.lock_holder(world_id)
        if holder is not None:
            (report["live_locks"] if holder["alive"] else report["dead_locks"]).append(world_id)
        for session_id in store.list_session_ids(world_id):
            try:
                session = store.read_session(world_id, session_id)
            except Exception:  # noqa: BLE001
                continue
            if session.ended_at is None:
                report["open_sessions"].append(f"{world_id}/{session_id}")
            elif session.end_reason != "stop" or (
                session.finalization or {}
            ).get("state") != "complete":
                report["interrupted"].append(
                    f"{world_id}/{session_id}: {session.end_reason}/{session.finalization}"
                )
            else:
                report["complete"] += 1
    return report


def builders_in(sample_row: dict) -> list[dict]:
    return [d for d in sample_row["descendants"] if d["kind"] == "world_build_session"]


# -- the walk ------------------------------------------------------------


async def _drain(ws, expect, limit: int = 200, timeout: float = 60.0) -> dict:
    for _ in range(limit):
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        message = json.loads(raw)
        if message.get("type") in expect:
            return message
    raise AssertionError(f"never saw one of {expect!r} in {limit} messages")


async def _feed(ws, frames: list[str], count: int, seq: int) -> tuple[int, int]:
    refused = 0
    for index in range(count):
        seq += 1
        await ws.send(json.dumps({
            "type": "frame", "seq": seq * 30, "width": WIDTH, "height": HEIGHT,
            "format": "jpeg", "data": frames[index % len(frames)],
        }))
        reply = await _drain(ws, ("frame_result", "frame_error"), timeout=180.0)
        if reply["type"] == "frame_error":
            refused += 1
    return seq, refused


def _wait_until(predicate, timeout: float, poll: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


async def soak(args, base: str, tower_pid: int | None, world_root: Path, findings: list[str]) -> dict:
    import websockets

    frames = build_frames()
    rows: list[dict] = []
    seq = 0

    def record(cycle: int, step: str, **extra) -> dict:
        row = {"cycle": cycle, "step": step, "t": round(time.time(), 3), **extra}
        row.update(sample(tower_pid, world_root))
        rows.append(row)
        return row

    record(-1, "baseline")
    async with websockets.connect(f"ws://{args.host}:{args.port}/ws", max_size=8 * 1024 * 1024) as ws:
        for cycle in range(args.cycles):
            # -- World Builder ------------------------------------------------
            t0 = time.perf_counter()
            status, body = _post(base, f"{WB_SESSION}/start")
            wb_start_ms = round((time.perf_counter() - t0) * 1000, 1)
            if status != 200 or body.get("state") != "active":
                findings.append(f"cycle {cycle}: world_builder start answered {status} {body}")
            await ws.send(json.dumps({"type": "stream_start"}))
            seq, refused = await _feed(ws, frames, args.frames, seq)
            row = record(cycle, "wb_walk", wb_start_ms=wb_start_ms, frames_refused=refused)
            attached = _wait_until(lambda: builders_in(sample(tower_pid, world_root)), 15.0)
            if tower_pid is not None and not attached:
                findings.append(f"cycle {cycle}: no builder attached during the World Builder walk")
            health = _get(base, "/health")[1]
            row["health_workers"] = health.get("capture_workers", {}).get("workers")
            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "stream_stop"}))
            await ws.send(json.dumps({"type": "ping"}))
            await _drain(ws, ("pong",))
            finalized = _wait_until(
                lambda: not (_get(base, "/health")[1].get("capture_workers", {}).get("workers")),
                args.finalize_timeout,
            )
            finalize_s = round(time.perf_counter() - t0, 2)
            if not finalized:
                findings.append(
                    f"cycle {cycle}: the builder was still registered {args.finalize_timeout}s after Stop"
                )
            t0 = time.perf_counter()
            status, body = _post(base, f"{WB_SESSION}/stop")
            wb_stop_ms = round((time.perf_counter() - t0) * 1000, 1)
            record(cycle, "wb_left", finalize_s=finalize_s, finalized=finalized, wb_stop_ms=wb_stop_ms)

            # -- CV Lab ----------------------------------------------------------
            t0 = time.perf_counter()
            await ws.send(json.dumps({
                "type": "cv_lab_start", "experiment_id": args.experiment,
                "request_id": f"soak-{cycle}",
            }))
            reply = await _drain(ws, ("cv_lab_status", "cv_lab_error"))
            arm_ms = round((time.perf_counter() - t0) * 1000, 1)
            if reply["type"] != "cv_lab_status":
                findings.append(f"cycle {cycle}: cv_lab_start refused: {reply.get('reason')}")
            await ws.send(json.dumps({"type": "stream_start"}))
            seq, refused = await _feed(ws, frames, args.frames, seq)
            row = record(cycle, "cv_lab_walk", arm_ms=arm_ms, frames_refused=refused)
            leaked = builders_in(row)
            if leaked:
                findings.append(
                    f"cycle {cycle}: a builder followed the CV Lab capture: {leaked}"
                )
            health = _get(base, "/health")[1]
            row["health_workers"] = health.get("capture_workers", {}).get("workers")
            if row["health_workers"]:
                findings.append(
                    f"cycle {cycle}: /health lists a worker during the CV Lab capture: "
                    f"{row['health_workers']}"
                )
            await ws.send(json.dumps({"type": "cv_lab_stop"}))
            await _drain(ws, ("cv_lab_status", "cv_lab_error"))
            await ws.send(json.dumps({"type": "stream_stop"}))
            await ws.send(json.dumps({"type": "ping"}))
            await _drain(ws, ("pong",))
            record(cycle, "cv_lab_left")

    settled = _wait_until(
        lambda: not (_get(base, "/health")[1].get("capture_workers", {}).get("workers")),
        args.finalize_timeout,
    )
    final = record(args.cycles, "settled", settled=settled)
    for d in final["descendants"]:
        if d["kind"] != "other":
            findings.append(f"after the last cycle a {d['kind']} process is still alive: {d}")
    for s in final["strays"]:
        findings.append(f"after the last cycle a stray worker is alive: {s}")
    locks = final["locks"]
    if locks["dead_locks"]:
        findings.append(f"locks naming dead pids: {locks['dead_locks']}")
    if locks["open_sessions"]:
        findings.append(f"sessions left open: {locks['open_sessions']}")
    if locks["interrupted"]:
        findings.append(f"sessions not complete/stop: {locks['interrupted']}")
    return {"rows": rows, "final": final}


# -- verdict -------------------------------------------------------------


def judge(rows: list[dict], cycles: int) -> dict:
    """Last third of cycles against the first, per resource, on the
    `cv_lab_left` samples (the same point in every cycle)."""
    settled = [r for r in rows if r["step"] == "cv_lab_left"]
    if len(settled) < 3:
        return {"verdict": "insufficient", "note": "need at least 3 cycles"}
    third = max(1, len(settled) // 3)
    first, last = settled[:third], settled[-third:]

    def mean(rows_, key):
        values = [r[key] for r in rows_ if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    tolerances = {"threads": 4, "rss_mb": 64, "handles": 50}
    result = {}
    growing = False
    for key, tolerance in tolerances.items():
        a, b = mean(first, key), mean(last, key)
        if a is None or b is None:
            result[key] = {"first": a, "last": b, "growing": None}
            continue
        grew = (b - a) > tolerance
        growing = growing or grew
        result[key] = {"first": round(a, 1), "last": round(b, 1), "delta": round(b - a, 1), "growing": grew}
    result["verdict"] = "GROWING" if growing else "flat"
    return result


def render(report: dict) -> str:
    lines = [
        f"{'cyc':>3} {'step':<12} {'thr':>4} {'rss_mb':>7} {'hnd':>5} {'kids':>4} {'bld':>3} {'fin_s':>6} {'arm_ms':>7}",
    ]
    for r in report["rows"]:
        lines.append(
            f"{r['cycle']:>3} {r['step']:<12} {str(r['threads']):>4} {str(r['rss_mb']):>7} "
            f"{str(r['handles']):>5} {len(r['descendants']):>4} {len(builders_in(r)):>3} "
            f"{str(r.get('finalize_s', '')):>6} {str(r.get('arm_ms', '')):>7}"
        )
    lines.append("")
    lines.append(f"verdict: {json.dumps(report['judge'])}")
    lines.append(f"locks: {json.dumps(report['final']['locks'])}")
    lines.append("findings: " + ("none" if not report["findings"] else ""))
    for finding in report["findings"]:
        lines.append(f"  - {finding}")
    return "\n".join(lines)


# -- CLI -----------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=artifact_root_arg, default=str(DEFAULT_ROOT),
                        help="where this soak's Tower keeps its capture and world roots")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--frames", type=int, default=24, help="frames per walk")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--finalize-timeout", type=float, default=120.0,
                        help="how long a builder may take after Stop before it is a finding")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="default: a free port")
    parser.add_argument("--attach-port", type=int, default=None,
                        help="use a Tower you started (its children are then invisible)")
    parser.add_argument("--intrinsics-from", type=Path, default=None,
                        help="copy a calibration directory into the world root so solves run for real")
    parser.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--keep", action="store_true", help="keep the soak roots afterwards")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if root.exists() and not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    (root / "capture").mkdir(parents=True, exist_ok=True)
    (root / "world").mkdir(parents=True, exist_ok=True)
    if args.intrinsics_from is not None:
        shutil.copytree(args.intrinsics_from, root / "world" / "intrinsics", dirs_exist_ok=True)

    findings: list[str] = []
    process = None
    tower_pid = None
    if args.attach_port is not None:
        args.port = args.attach_port
        findings.append("attached to an existing Tower: descendants are invisible; only the process-table scan and the locks are checked")
    else:
        args.port = args.port or _free_port()
        log_path = root / "tower.log"
        process = start_tower(args.port, root, log_path)
        tower_pid = process.pid
    base = f"http://{args.host}:{args.port}"
    started = time.perf_counter()
    try:
        health = wait_for_health(base, process, root / "tower.log")
        configured = (health.get("capture_workers") or {}).get("configured", [])
        if "world-build" not in configured and not any("world" in c for c in configured):
            findings.append(f"this Tower has no builder configured: {configured}")
        report = asyncio.run(soak(args, base, tower_pid, root / "world", findings))
    finally:
        clean = True
        if process is not None:
            t0 = time.perf_counter()
            clean = stop_tower(process)
            shutdown_s = round(time.perf_counter() - t0, 2)
            time.sleep(1.0)
            after = sample(None, root / "world")
            if after["strays"]:
                findings.append(f"workers outlived the Tower: {after['strays']}")
            if not clean:
                findings.append("the Tower did not stop within 120 s of being asked")
    report["judge"] = judge(report["rows"], args.cycles)
    report["findings"] = findings
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    if process is not None:
        report["shutdown_seconds"] = shutdown_s
    print(render(report))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report: {args.report}")
    ok = not findings and report["judge"].get("verdict") in ("flat", "insufficient")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
