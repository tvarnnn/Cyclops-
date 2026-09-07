#!/usr/bin/env python
r"""Start and stop Scene Understanding over and over, and watch what stays.

The lifecycle the product needs is one long-lived Tower and many short
scene sessions: open the Scene screen, close it, open World Builder,
come back. Every one of those is a model load and a model release, a
worker taking a job and parking, and -- on CUDA -- a context that must
give its memory back. This script does an afternoon of that in minutes,
against the REAL app (`create_app()`), the real `/ws`, the real result
channel and the real models, and samples the process after every cycle.

    .venv\Scripts\python.exe scripts/scene_soak.py --cycles 10
    .venv\Scripts\python.exe scripts/scene_soak.py --cycles 6 --switch
    .venv\Scripts\python.exe scripts/scene_soak.py --cycles 6 --out <dir>

One row per cycle:

    start_ms    from result_subscribe (with a stream open) to `running`
    frames      frames offered / observed / skipped in the cycle
    scene       counts on the last scene of the cycle (person, laptop)
    stop_ms     from result_unsubscribe to `stopped`
    thr         OS threads in the process (psutil), after the stop
    rss_mb      resident set, MB
    vram a/r    torch.cuda memory_allocated / memory_reserved, MB
    hnd         Windows handle count

`--switch` interleaves a CV Lab experiment between scene cycles, over the
same socket, which is the "Scene -> CV Lab -> Scene" path a phone takes.

The verdict compares the mean of each resource over the last third of the
cycles against the first third and flags GROWING above a tolerance meant
to sit above allocator noise and below a real leak: threads > 4, rss_mb >
64, handles > 50, vram_reserved_mb > 64. The first cycle is a warm-up
(model download and first CUDA context) and is shown but not judged.

Frames come from a captures root, read only, and are fed as fast as the
socket accepts them; this is a lifecycle test, not a throughput one.
"""

import argparse
import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402

CONTRACT = "scene_understanding.live/2026-08-27"


def _sample() -> dict:
    out = {}
    try:
        import psutil

        process = psutil.Process()
        out["threads"] = process.num_threads()
        out["rss_mb"] = round(process.memory_info().rss / 1e6, 1)
        if hasattr(process, "num_handles"):
            out["handles"] = process.num_handles()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            out["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
            out["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 1)
    except Exception:
        pass
    # Python-visible threads by name prefix, so a growth in `threads` can
    # be attributed: `asyncio_` is the default executor filling toward
    # its bound, `tower-Scene-session` is the reused worker, and anything
    # else is news. Native OpenMP threads are invisible here.
    import threading
    from collections import Counter

    names = Counter()
    for thread in threading.enumerate():
        name = thread.name
        for prefix in ("asyncio_", "tower-Scene", "tower-results", "tower-stream", "ThreadPoolExecutor", "AnyIO"):
            if name.startswith(prefix):
                name = prefix
                break
        names[name] += 1
    out["python_threads"] = threading.active_count()
    out["python_threads_by_name"] = dict(names)
    return out


def _frames(captures: Path, count: int) -> list[str]:
    encoded = []
    for capture_dir in sorted(p for p in captures.iterdir() if p.is_dir()):
        for path in sorted((capture_dir / "frames").glob("*.jpg"))[:count]:
            encoded.append(base64.b64encode(path.read_bytes()).decode("ascii"))
            if len(encoded) >= count:
                return encoded
    if not encoded:
        raise SystemExit(f"no frames under {captures}")
    return encoded


def _await(client, want: str, timeout_s: float) -> tuple[dict, float]:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout_s:
        payload = client.get("/scene").json()
        if payload["lifecycle"]["state"] == want:
            return payload, (time.perf_counter() - started) * 1000
        if payload["lifecycle"]["state"] == "failed":
            raise RuntimeError(payload["lifecycle"]["failure_reason"])
        time.sleep(0.02)
    raise TimeoutError(f"never reached {want}")


def _drain(ws, until_type: str, limit: int = 200) -> dict:
    for _ in range(limit):
        message = ws.receive_json()
        if message["type"] == until_type:
            return message
    raise RuntimeError(f"no {until_type} in {limit} messages")


def run(args) -> dict:
    os.environ.setdefault("TOWER_SCENE_UNDERSTANDING", "auto")
    os.environ.pop("TOWER_CAPTURE_ROOT", None)
    from fastapi.testclient import TestClient

    from tower.main import create_app

    frames = _frames(args.captures, args.frames)
    rows = []
    app = create_app()
    with TestClient(app) as client:
        declared = next(
            e for e in client.get("/cartridges").json()["cartridges"] if e["cartridge"] == "scene_understanding"
        )
        if not declared["available"]:
            raise SystemExit(f"scene unavailable: {declared['unavailable_reason']}")
        client.app.state.result_hub._poll_seconds = 0.5
        baseline = _sample()
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "stream_start"})
            for cycle in range(args.cycles):
                row = {"cycle": cycle, "warmup": cycle == 0}
                ws.send_json({"type": "result_subscribe", "cartridge": "scene_understanding", "result_type": "live", "contract": CONTRACT})
                ack = _drain(ws, "result_subscribed")
                _drain(ws, "cartridge_result")
                _, row["start_ms"] = _await(client, "running", args.load_timeout)
                for seq, data in enumerate(frames):
                    ws.send_json({"type": "frame", "seq": seq, "width": 360, "height": 640, "format": "jpeg", "data": data})
                    _drain(ws, "frame_result")
                # Let the worker catch up on the last slot.
                time.sleep(0.3)
                scene = client.get("/scene").json()
                row["frames"] = {k: scene[k] for k in ("frames_offered", "frames_observed", "frames_skipped")}
                row["scene"] = None if not scene["scene_available"] else {
                    "person": scene["counts"]["person"],
                    "laptop": scene["counts"]["laptop"],
                    "partial": scene["people"]["partial_bottom_edge"],
                    "facing": scene["people"]["facing_wearer"],
                }
                row["latency_ms"] = None
                stop_started = time.perf_counter()
                ws.send_json({"type": "result_unsubscribe", "subscription_id": ack["subscription_id"]})
                _drain(ws, "result_unsubscribed")
                after, _ = _await(client, "stopped", 15.0)
                row["stop_ms"] = round((time.perf_counter() - stop_started) * 1000, 1)
                row["scene_after_stop_available"] = after["scene_available"]
                row["watchers_after"] = after["lifecycle"]["demand"]["watchers"]
                if args.switch:
                    ws.send_json({"type": "cv_lab_start", "experiment": "edge_detection"})
                    for seq, data in enumerate(frames[:10]):
                        ws.send_json({"type": "frame", "seq": seq, "width": 360, "height": 640, "format": "jpeg", "data": data})
                        _drain(ws, "frame_result")
                    ws.send_json({"type": "cv_lab_stop"})
                    time.sleep(0.2)
                row.update(_sample())
                rows.append(row)
                print(_row(row), flush=True)
            ws.send_json({"type": "stream_stop"})
        health = client.get("/health").status_code
        final = _sample()
    return {"baseline": baseline, "rows": rows, "final": final, "health_after": health, "verdict": judge(rows)}


def _row(row: dict) -> str:
    scene = row.get("scene") or {}
    return (
        f"c{row['cycle']:02d}{'w' if row['warmup'] else ' '} start {row['start_ms']:7.0f} ms  "
        f"frames {row['frames']['frames_observed']}/{row['frames']['frames_offered']} skip {row['frames']['frames_skipped']}  "
        f"person {scene.get('person')} laptop {scene.get('laptop')} partial {scene.get('partial')} facing {scene.get('facing')}  "
        f"stop {row['stop_ms']:6.0f} ms  thr {row.get('threads')}  rss {row.get('rss_mb')}  "
        f"vram {row.get('vram_allocated_mb')}/{row.get('vram_reserved_mb')}  hnd {row.get('handles')}"
    )


def judge(rows: list[dict]) -> dict:
    judged = [r for r in rows if not r["warmup"]]
    if len(judged) < 3:
        return {"verdict": "INCONCLUSIVE", "reason": "fewer than 3 judged cycles"}
    third = max(1, len(judged) // 3)
    first, last = judged[:third], judged[-third:]
    tolerance = {"threads": 4, "rss_mb": 64, "handles": 50, "vram_reserved_mb": 64}
    growth = {}
    flagged = []
    for field, limit in tolerance.items():
        a = [r[field] for r in first if field in r]
        b = [r[field] for r in last if field in r]
        if not a or not b:
            continue
        delta = statistics.fmean(b) - statistics.fmean(a)
        growth[field] = round(delta, 1)
        if delta > limit:
            flagged.append(field)
    leaked_scene = [r["cycle"] for r in rows if r["scene_after_stop_available"]]
    watchers = [r["cycle"] for r in rows if r["watchers_after"] != 0]
    verdict = "GROWING" if flagged else "STABLE"
    if leaked_scene or watchers:
        verdict = "DEFECT"
    return {
        "verdict": verdict,
        "growth_last_third_minus_first_third": growth,
        "flagged": flagged,
        "scene_survived_a_stop_in_cycles": leaked_scene,
        "watchers_left_over_in_cycles": watchers,
        "max_stop_ms": max(r["stop_ms"] for r in rows),
        "median_start_ms_after_warmup": round(statistics.median(r["start_ms"] for r in judged), 1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", type=Path, required=True, help="captures root, read only")
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--frames", type=int, default=40, help="frames per cycle")
    parser.add_argument("--switch", action="store_true", help="run a CV Lab experiment between cycles")
    parser.add_argument("--load-timeout", type=float, default=180.0)
    parser.add_argument("--out", type=artifact_root_arg, default=None, help="directory for the JSON report (guarded)")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    report = run(args)
    if args.out is not None:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "scene_soak_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print("baseline", report["baseline"])
        print("final   ", report["final"], "health", report["health_after"])
        print("verdict ", json.dumps(report["verdict"]))
    return 0 if report["verdict"]["verdict"] in ("STABLE", "INCONCLUSIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
