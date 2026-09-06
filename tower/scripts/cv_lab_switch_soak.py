#!/usr/bin/env python
"""Switch experiments over and over, and watch what the process keeps.

The CV Lab holds ONE experiment. Every `cv_lab_start` drops the one it
has and arms another, and a person using the Lab does that all afternoon:
edges, then depth, then flow, then depth again. Each arm builds a
detector, may import torch, may spin up an OpenMP team, may open a CUDA
context, and may hand its load to a background thread. Each release is
supposed to give all of that back. This script checks that it does, by
doing the afternoon in a few minutes and sampling the process after every
switch.

What the numbers mean
---------------------

One row per transition (one `start` of one experiment inside one cycle):

    arm_ms      wall time from `start()` until the Lab left `starting`
    ok/ref/err  frames processed / refused (the Lab said no) / raised
    p50_ms      median `processing_ms` the experiment reported per frame
    thr         OS threads in the process (psutil), AFTER the frames
    rss_mb      resident set, MB
    hnd         Windows handle count (n/a elsewhere)
    chld        child processes, recursive
    cuda a/r    torch.cuda memory_allocated / memory_reserved, MB, when
                torch is already imported and CUDA is up; n/a otherwise

The verdict compares the MEAN of each resource over the last third of the
cycles against the mean over the first third, and flags `GROWING` when
the rise exceeds a tolerance that is meant to sit above allocator noise
and below a real leak: threads > 4, rss_mb > 64, handles > 50, children
> 0 (a follower process that outlives its arm is never noise), and
cuda_reserved_mb > 64.

Why a warm-up walk, and why the first third rather than sample zero: the
first pass over the order is EXPECTED to jump. `baseline` runs before
OpenCV has built its parallel worker pool (the first `edge_detection`
adds ~17 threads on this machine and they stay for the life of the
process); the first `depth` arm imports torch, creates the CUDA context,
and starts an OpenMP team. None of that is a leak -- it is the cost of
the first use, paid once -- and a verdict anchored on sample zero would
call every Tower with a GPU a leaker. So the order is walked once as a
warm-up (rows marked `w`, shown in the table, kept in the report, NOT
judged) and only the cycles after it are compared. Anchoring on a third
of those cycles rather than on one sample then averages over the
per-experiment differences within a cycle, so a walk that happens to end
on a heavy experiment is not read as growth. That is also why `--cycles`
should be at least 3: with fewer, the first and last "thirds" are the
same cycle and the verdict says nothing. The script warns.

Two modes
---------

`--in-process` (default) builds a real `CVLab` with the real registry in
this process, so the samples describe THIS process and the CUDA numbers
are readable. It says nothing about uvicorn or about the wire.

`--live --host H --port P [--tower-pid PID]` does the same walk against a
Tower you started yourself, over `/ws`, using the message shapes
`cv_lab_smoke.py` uses. The camera stream is started once and never
stopped between switches, because a stream that survives a switch is the
property under test. Threads and RSS come from the status document's
`process` block when the Tower publishes one, and `run.arm_ms` likewise;
either shows `n/a` on a Tower that predates them. With `--tower-pid` the
process is also sampled from the OUTSIDE with psutil, which is the only
way to see a child process (a World Builder follower, say) that the Tower
itself does not report.

    .venv\\Scripts\\python.exe scripts/cv_lab_switch_soak.py
    .venv\\Scripts\\python.exe scripts/cv_lab_switch_soak.py --cycles 6 --json soak.json
    .venv\\Scripts\\python.exe scripts/cv_lab_switch_soak.py --order baseline,depth --cycles 4
    .venv\\Scripts\\python.exe scripts/cv_lab_switch_soak.py --live --port 8000 --tower-pid 12345

Exit code 0 when nothing is GROWING and every transition reached
`running`; 1 otherwise; 2 for a usage error. Nothing is written to disk
unless `--json PATH` is given.
"""

import argparse
import asyncio
import base64
import io
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.cv_lab import CVLab  # noqa: E402
from tower.experiments import EXPERIMENTS, ExperimentSettings  # noqa: E402
from tower.modules.base import FrameProcessingError  # noqa: E402

# The resolution the glasses deliver today, as in the other Lab scripts.
WIDTH, HEIGHT = 640, 360

DEFAULT_ORDER = (
    "baseline,edge_detection,feature_detection,frame_quality,"
    "optical_flow,depth,object_detection,redaction_impact"
)

# Distinct frames, cycled. `optical_flow` tracks features from one frame
# to the next and reports nothing useful when every frame is identical;
# eight seeds is enough variety and small enough to build once.
FRAME_SEEDS = 8

# Growth above which a resource is called GROWING. Each is meant to sit
# above the noise the allocator, the OS and a busy machine produce
# between two samples, and below what a real per-arm leak reaches in
# three cycles of eight experiments.
TOLERANCES = {
    "threads": 4,
    "rss_mb": 64.0,
    "handles": 50,
    "children": 0,
    "cuda_reserved_mb": 64.0,
}

# Sample fields the verdict judges, in table order.
JUDGED = tuple(TOLERANCES)

# The cycle number of the warm-up walk. Negative so that it sorts before
# every measured cycle and cannot be confused with one.
WARMUP_CYCLE = -1


# -- frames -------------------------------------------------------------


def build_frames(count: int = FRAME_SEEDS) -> list[bytes]:
    """Textured 640x360 JPEGs, built once.

    Generated here rather than imported from `tests/cv_lab_fixtures.py`:
    that module imports `tower.main` and the FastAPI `TestClient` at
    module level, which would pull the whole app into the process being
    measured and put its threads into the baseline sample. Flat grey is
    avoided for the reason the smoke script gives -- it gives every
    experiment nothing to find, and a working Lab then looks broken.
    """
    frames = []
    for seed in range(count):
        rng = np.random.default_rng(seed + 1)
        array = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=60)
        frames.append(buffer.getvalue())
    return frames


# -- sampling -----------------------------------------------------------


def _mb(value) -> float | None:
    return None if value is None else round(value / 1048576, 1)


def sample_process(process: psutil.Process | None) -> dict:
    """What the OS says this process is holding, right now.

    torch is read out of `sys.modules` rather than imported, on purpose:
    importing it here would put its threads and its CUDA context into a
    sample that is supposed to show what the LAB brought in. Before the
    first model-backed arm, the CUDA columns are honestly `n/a`.
    """
    sample = {
        "threads": None,
        "rss_mb": None,
        "handles": None,
        "children": None,
        "children_detail": [],
        "cuda_alloc_mb": None,
        "cuda_reserved_mb": None,
        "py_threads": threading.active_count(),
        "tower_threads": 0,
        "tower_thread_names": [],
    }
    if process is not None:
        try:
            sample["threads"] = process.num_threads()
            sample["rss_mb"] = _mb(process.memory_info().rss)
            if hasattr(process, "num_handles"):
                sample["handles"] = process.num_handles()
            children = process.children(recursive=True)
            sample["children"] = len(children)
            sample["children_detail"] = [_describe_child(child) for child in children]
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            sample["error"] = f"{type(exc).__name__}: {exc}"

    tower_names = sorted(
        thread.name for thread in threading.enumerate() if thread.name.startswith("tower-")
    )
    sample["tower_threads"] = len(tower_names)
    sample["tower_thread_names"] = tower_names

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                sample["cuda_alloc_mb"] = _mb(torch.cuda.memory_allocated())
                sample["cuda_reserved_mb"] = _mb(torch.cuda.memory_reserved())
        except Exception as exc:  # noqa: BLE001 -- a broken CUDA is a finding, not a crash
            sample["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return sample


def _describe_child(child: psutil.Process) -> dict:
    try:
        cmdline = " ".join(child.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        cmdline = "?"
    return {"pid": child.pid, "cmdline": cmdline[:80]}


# -- the verdict --------------------------------------------------------


def judge(samples: list[dict]) -> dict:
    """Compare the last third of cycles with the first, per resource.

    Pure: takes the per-transition samples (each carrying `cycle` and the
    fields in `TOLERANCES`) and returns the verdict document. A field
    that is `None` in every sample of either third -- CUDA on a CPU
    Tower, handles off Windows -- is reported `n/a` and never GROWING.

    The thirds are thirds of CYCLES, not of samples, so each side holds
    whole walks of the order and the two means cover the same mix of
    experiments. With fewer than three cycles both thirds collapse to one
    cycle each, and with one cycle they are the same cycle; the caller
    warns about that, this function just reports which cycles it used.
    """
    cycles = sorted({sample["cycle"] for sample in samples})
    third = max(1, len(cycles) // 3)
    first, last = cycles[:third], cycles[-third:]

    def mean_over(chosen: list[int], field: str) -> float | None:
        values = [
            sample.get(field)
            for sample in samples
            if sample["cycle"] in chosen and sample.get(field) is not None
        ]
        return statistics.fmean(values) if values else None

    verdict = {
        "cycles_compared": {"first": first, "last": last, "total": len(cycles)},
        "growing": [],
    }
    for field, tolerance in TOLERANCES.items():
        first_mean = mean_over(first, field)
        last_mean = mean_over(last, field)
        entry = {
            "first_mean": None if first_mean is None else round(first_mean, 2),
            "last_mean": None if last_mean is None else round(last_mean, 2),
            "delta": None,
            "tolerance": tolerance,
            "verdict": "n/a",
        }
        if first_mean is not None and last_mean is not None:
            delta = last_mean - first_mean
            entry["delta"] = round(delta, 2)
            entry["verdict"] = "GROWING" if delta > tolerance else "flat"
            if delta > tolerance:
                verdict["growing"].append(field)
        verdict[field] = entry
    return verdict


# -- output -------------------------------------------------------------


def _cell(value, width: int, decimals: int = 0) -> str:
    if value is None:
        return "n/a".rjust(width)
    if isinstance(value, float):
        return f"{value:{width}.{decimals}f}"
    return f"{value:{width}}"


def render_table(transitions: list[dict]) -> str:
    header = (
        f"{'cyc':>3} {'experiment':<18} {'state':<9} {'arm_ms':>8} "
        f"{'ok/ref/err':>10} {'p50_ms':>8} {'thr':>4} {'rss_mb':>8} "
        f"{'hnd':>5} {'chld':>4} {'cuda a/r MB':>13}"
    )
    lines = [header, "-" * len(header)]
    for row in transitions:
        state = row["state"] if row["accepted"] else "REFUSED"
        frames = f"{row['frames_ok']}/{row['frames_refused']}/{row['frames_error']}"
        cuda = (
            "n/a"
            if row["cuda_reserved_mb"] is None
            else f"{row['cuda_alloc_mb']:.0f}/{row['cuda_reserved_mb']:.0f}"
        )
        cycle = "w" if row["cycle"] == WARMUP_CYCLE else row["cycle"]
        lines.append(
            f"{cycle:>3} {row['experiment']:<18} {state:<9} "
            f"{_cell(row['arm_ms'], 8, 1)} {frames:>10} "
            f"{_cell(row['p50_ms'], 8, 2)} {_cell(row['threads'], 4)} "
            f"{_cell(row['rss_mb'], 8, 1)} {_cell(row['handles'], 5)} "
            f"{_cell(row['children'], 4)} {cuda:>13}"
        )
    return "\n".join(lines)


def render_verdict(verdict: dict) -> str:
    compared = verdict["cycles_compared"]
    lines = [
        f"verdict: mean over cycles {compared['last']} minus mean over "
        f"cycles {compared['first']}",
    ]
    for field in JUDGED:
        entry = verdict[field]
        if entry["verdict"] == "n/a":
            lines.append(f"  {field:<17} n/a")
            continue
        lines.append(
            f"  {field:<17} {entry['first_mean']:>9.2f} -> {entry['last_mean']:>9.2f}"
            f"  delta {entry['delta']:>+9.2f}  (tolerance {entry['tolerance']})"
            f"  {entry['verdict']}"
        )
    return "\n".join(lines)


def _p50(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


# -- in-process walk ----------------------------------------------------


async def soak_in_process(args, frames: list[bytes]) -> dict:
    process = psutil.Process()
    lab = CVLab("baseline", ExperimentSettings(device=args.device))
    await lab.load_initial()

    rows = []
    cycles = []
    # Cycle -1 is the warm-up walk: sampled and reported, never judged.
    for cycle in range(WARMUP_CYCLE, args.cycles):
        for experiment_id in args.order:
            row = _new_row(cycle, experiment_id)
            t0 = time.perf_counter()
            outcome = lab.start(experiment_id)
            if not outcome.accepted:
                row["refusal"] = outcome.reason
                row["state"] = outcome.status["lifecycle"]["state"]
            else:
                row["accepted"] = True
                await lab.wait_until_armed()
                row["arm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                status = lab.status()
                row["state"] = status["lifecycle"]["state"]
                row["state_reason"] = status["lifecycle"]["reason"]
                run = status.get("run") or {}
                row["reported_arm_ms"] = run.get("arm_ms")
                if row["state"] == "running":
                    _feed_in_process(lab, frames, args.frames, row)
            row.update(sample_process(process))
            rows.append(row)

        # Release latency, once per cycle. A stop that takes seconds is a
        # release that is waiting on something -- a load thread, a CUDA
        # sync -- and that is worth a number even when nothing leaks.
        t0 = time.perf_counter()
        stopped = lab.stop()
        cycles.append(
            {
                "cycle": cycle,
                "stop_ms": round((time.perf_counter() - t0) * 1000, 1),
                "stop_accepted": stopped.accepted,
                "stop_refusal": stopped.reason,
            }
        )

    t0 = time.perf_counter()
    await lab.shutdown("soak finished")
    shutdown_ms = round((time.perf_counter() - t0) * 1000, 1)
    final = sample_process(process)
    return _split_report(rows, cycles, shutdown_ms, final)


def _new_row(cycle: int, experiment_id: str) -> dict:
    return {
        "cycle": cycle,
        "experiment": experiment_id,
        "accepted": False,
        "refusal": None,
        "state": None,
        "state_reason": None,
        "arm_ms": None,
        "reported_arm_ms": None,
        "frames_ok": 0,
        "frames_refused": 0,
        "frames_error": 0,
        "p50_ms": None,
        "error_samples": [],
    }


def _split_report(rows: list[dict], cycles: list[dict], shutdown_ms, final: dict) -> dict:
    """The warm-up walk goes beside the measured transitions, not among them."""
    return {
        "warmup": [row for row in rows if row["cycle"] == WARMUP_CYCLE],
        "transitions": [row for row in rows if row["cycle"] != WARMUP_CYCLE],
        "warmup_stop_ms": next(
            (entry["stop_ms"] for entry in cycles if entry["cycle"] == WARMUP_CYCLE), None
        ),
        "cycles": [entry for entry in cycles if entry["cycle"] != WARMUP_CYCLE],
        "shutdown_ms": shutdown_ms,
        "final_sample": final,
    }


def _feed_in_process(lab: CVLab, frames: list[bytes], count: int, row: dict) -> None:
    processing = []
    for index in range(count):
        try:
            result = lab.process(frames[index % len(frames)])
        except FrameProcessingError as exc:
            row["frames_refused"] += 1
            if len(row["error_samples"]) < 3:
                row["error_samples"].append(f"refused: {exc.reason}")
        except Exception as exc:  # noqa: BLE001 -- counted, not fatal; the soak goes on
            row["frames_error"] += 1
            if len(row["error_samples"]) < 3:
                row["error_samples"].append(f"{type(exc).__name__}: {exc}"[:120])
        else:
            row["frames_ok"] += 1
            processing.append(result.processing_ms)
    row["p50_ms"] = _p50(processing)


# -- live walk ----------------------------------------------------------


async def _drain(ws, expect, limit: int = 200, timeout: float = 30.0) -> dict:
    """The next message of an acceptable type, skipping the rest.

    The socket also pushes result-channel envelopes and whatever else the
    Tower learns to send; anything not in `expect` is skipped. Where a
    `cv_lab_status` is expected a `cv_lab_error` may arrive instead and
    carries the status anyway -- the smoke script's lesson, kept.
    """
    wanted = (expect,) if isinstance(expect, str) else tuple(expect)
    for _ in range(limit):
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if message.get("type") in wanted:
            return message
    raise AssertionError(f"never saw any of {wanted} in {limit} messages")


async def _await_running(ws, timeout_s: float) -> dict:
    """Poll status until the Lab leaves `starting`, or the timeout passes."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        await ws.send(json.dumps({"type": "cv_lab_status"}))
        status = (await _drain(ws, ("cv_lab_status", "cv_lab_error")))["status"]
        if status["lifecycle"]["state"] != "starting" or loop.time() > deadline:
            return status
        await asyncio.sleep(0.25)


def _fetch_process_over_http(host: str, port: int) -> dict:
    """`GET /cv-lab`'s `process` block, or an `error` key."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/cv-lab", timeout=5) as resp:
            return json.load(resp).get("process") or {}
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        return {"error": str(exc)}


def _sample_tower(pid: int | None) -> dict:
    """The Tower from the outside. Only `--tower-pid` can see its children."""
    if pid is None:
        return {}
    try:
        return sample_process(psutil.Process(pid))
    except psutil.NoSuchProcess:
        return {"error": f"no process with pid {pid}"}


async def soak_live(args, frames: list[bytes]) -> dict:
    import websockets

    encoded = [base64.b64encode(frame).decode("ascii") for frame in frames]
    ws_url = f"ws://{args.host}:{args.port}/ws"
    rows = []
    cycles = []
    request = 0

    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
        # Started once, and NOT stopped between switches. A `stream_stop`
        # ends the capture lineage on the Tower, and in the operator's
        # configuration a new stream may spawn a new follower process; a
        # switch is supposed to leave the stream alone.
        await ws.send(json.dumps({"type": "stream_start"}))
        seq = 0
        for cycle in range(WARMUP_CYCLE, args.cycles):
            for experiment_id in args.order:
                request += 1
                request_id = f"soak-{cycle}-{request}"
                row = _new_row(cycle, experiment_id)
                t0 = time.perf_counter()
                await ws.send(
                    json.dumps(
                        {
                            "type": "cv_lab_start",
                            "experiment_id": experiment_id,
                            "request_id": request_id,
                        }
                    )
                )
                reply = await _drain(ws, ("cv_lab_status", "cv_lab_error"))
                if reply["type"] != "cv_lab_status":
                    row["refusal"] = reply.get("reason")
                    row["state"] = reply.get("status", {}).get("lifecycle", {}).get("state")
                    status = reply.get("status", {})
                else:
                    row["accepted"] = True
                    status = await _await_running(ws, args.arm_timeout)
                    row["arm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    row["state"] = status["lifecycle"]["state"]
                    row["state_reason"] = status["lifecycle"].get("reason")
                    if row["state"] == "running":
                        seq = await _feed_live(ws, encoded, args.frames, seq, row)
                        await ws.send(json.dumps({"type": "cv_lab_status"}))
                        status = (
                            await _drain(ws, ("cv_lab_status", "cv_lab_error"))
                        )["status"]
                # The Tower's own reading of itself lives BESIDE the status
                # document on `GET /cv-lab`, not inside it (the document is
                # byte-equal on three surfaces; a live RSS figure would not
                # be). An older Tower without it leaves these n/a.
                reported = _fetch_process_over_http(args.host, args.port)
                run = status.get("run") or {}
                row["reported_arm_ms"] = run.get("arm_ms")
                row["reported_process"] = {
                    "pid": reported.get("pid"),
                    "threads": reported.get("threads"),
                    "rss_mb": reported.get("rss_mb"),
                }
                # The outside view wins where it exists, because it is the
                # one that can see children. Otherwise the Tower's own
                # report; otherwise n/a.
                outside = _sample_tower(args.tower_pid)
                row.update(
                    {
                        "threads": outside.get("threads", reported.get("threads")),
                        "rss_mb": outside.get("rss_mb", reported.get("rss_mb")),
                        "handles": outside.get("handles"),
                        "children": outside.get("children"),
                        "children_detail": outside.get("children_detail", []),
                        "cuda_alloc_mb": None,
                        "cuda_reserved_mb": None,
                        "tower_sample_error": outside.get("error"),
                    }
                )
                rows.append(row)

            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "cv_lab_stop"}))
            reply = await _drain(ws, ("cv_lab_status", "cv_lab_error"))
            cycles.append(
                {
                    "cycle": cycle,
                    "stop_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "stop_accepted": reply["type"] == "cv_lab_status",
                    "stop_refusal": reply.get("reason"),
                }
            )

        await ws.send(json.dumps({"type": "cv_lab_stop"}))
        await _drain(ws, ("cv_lab_status", "cv_lab_error"))
        await ws.send(json.dumps({"type": "stream_stop"}))

    return _split_report(rows, cycles, None, _sample_tower(args.tower_pid))


async def _feed_live(ws, encoded: list[str], count: int, seq: int, row: dict) -> int:
    processing = []
    for index in range(count):
        seq += 1
        await ws.send(
            json.dumps(
                {
                    "type": "frame",
                    "seq": seq * 30,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "format": "jpeg",
                    "data": encoded[index % len(encoded)],
                }
            )
        )
        reply = await _drain(ws, ("frame_result", "frame_error"), timeout=180.0)
        if reply["type"] == "frame_error":
            row["frames_refused"] += 1
            if len(row["error_samples"]) < 3:
                row["error_samples"].append(f"refused: {reply.get('reason')}")
        else:
            row["frames_ok"] += 1
            processing.append(reply["processing_ms"])
    row["p50_ms"] = _p50(processing)
    return seq


# -- CLI ----------------------------------------------------------------


def _parse_order(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Switch CV Lab experiments repeatedly and report whether "
            "threads, memory, handles, children or CUDA memory accumulate."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--in-process",
        action="store_true",
        help="build a CVLab in this process (the default)",
    )
    mode.add_argument(
        "--live", action="store_true", help="walk a running Tower over /ws"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tower-pid",
        type=int,
        default=None,
        help="with --live: also sample the Tower process from outside via psutil",
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--frames", type=int, default=15, help="frames per arm")
    parser.add_argument(
        "--order",
        default=DEFAULT_ORDER,
        help="comma-separated experiment ids, walked in this order each cycle",
    )
    parser.add_argument("--device", default="auto", help="ExperimentSettings.device")
    parser.add_argument(
        "--arm-timeout",
        type=float,
        default=130.0,
        help="with --live: seconds to wait for an arm before recording it as stuck",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        metavar="PATH",
        help="write the per-transition list and verdict here; default writes nothing",
    )
    args = parser.parse_args(argv)

    args.order = _parse_order(args.order)
    unknown = [name for name in args.order if name not in EXPERIMENTS]
    if unknown or not args.order:
        print(
            f"unknown experiment(s) in --order: {unknown or '(empty)'}; "
            f"the registry has: {', '.join(sorted(EXPERIMENTS))}",
            file=sys.stderr,
        )
        return 2
    if args.cycles < 1 or args.frames < 0:
        print("--cycles must be >= 1 and --frames >= 0", file=sys.stderr)
        return 2
    if args.cycles < 3:
        print(
            f"warning: --cycles {args.cycles} gives the verdict nothing to "
            "compare (the first and last thirds overlap); use 3 or more",
            file=sys.stderr,
        )

    frames = build_frames()
    try:
        if args.live:
            report = asyncio.run(soak_live(args, frames))
        else:
            report = asyncio.run(soak_in_process(args, frames))
    except ImportError as exc:
        print(f"--live needs the websockets package: {exc}", file=sys.stderr)
        return 2
    except (OSError, asyncio.TimeoutError) as exc:
        print(
            f"could not complete the walk against {args.host}:{args.port} -- {exc}",
            file=sys.stderr,
        )
        return 2

    transitions = report["transitions"]
    verdict = judge(transitions)
    failed = [
        f"cycle {row['cycle']} {row['experiment']}: "
        + (f"refused ({row['refusal']})" if not row["accepted"] else
           f"{row['state']} ({row['state_reason']})")
        for row in transitions
        if not row["accepted"] or row["state"] != "running"
    ]

    mode_name = "live" if args.live else "in-process"
    print(
        f"CV Lab switch soak -- {mode_name}, {args.cycles} cycles x "
        f"{len(args.order)} experiments, {args.frames} frames per arm, "
        f"device {args.device}"
    )
    print()
    print(render_table(report["warmup"] + transitions))
    print("  (w = warm-up walk: shown, not judged)")
    print()
    stop_line = "  ".join(
        f"c{entry['cycle']} {entry['stop_ms']:.1f} ms"
        + ("" if entry["stop_accepted"] else f" (refused: {entry['stop_refusal']})")
        for entry in report["cycles"]
    )
    print(f"stop latency per cycle: {stop_line}")
    if report["shutdown_ms"] is not None:
        final = report["final_sample"]
        print(
            f"shutdown {report['shutdown_ms']:.1f} ms; after shutdown: "
            f"threads {final.get('threads')}, rss_mb {final.get('rss_mb')}, "
            f"tower-* threads {final.get('tower_thread_names')}"
        )
    print()
    print(render_verdict(verdict))
    if failed:
        print()
        print(f"{len(failed)} transition(s) did not reach running:")
        for line in failed:
            print(f"  - {line}")
    children = [row for row in transitions if row.get("children_detail")]
    if children:
        print()
        print("child processes seen:")
        seen = set()
        for row in children:
            for child in row["children_detail"]:
                key = (child["pid"], child["cmdline"])
                if key in seen:
                    continue
                seen.add(key)
                print(f"  pid {child['pid']}  {child['cmdline']}")

    if args.json_path:
        document = {
            "mode": mode_name,
            "order": args.order,
            "cycles_requested": args.cycles,
            "frames_per_arm": args.frames,
            "device": args.device,
            "warmup": report["warmup"],
            "warmup_stop_ms": report["warmup_stop_ms"],
            "transitions": transitions,
            "cycles": report["cycles"],
            "shutdown_ms": report["shutdown_ms"],
            "final_sample": report["final_sample"],
            "verdict": verdict,
            "failed_transitions": failed,
        }
        Path(args.json_path).write_text(json.dumps(document, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_path}")

    return 1 if (verdict["growing"] or failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
