"""What a live cartridge costs a Tower that is also serving frames.

Two cartridges now attach a worker thread to the frame path, and the
question a deployment actually has is not "how fast is the detector" --
`scripts/scene_benchmark.py` answers that -- but **"can this keep up with
the glasses, and does it grow?"** Those are different questions and this
script answers the second.

WHAT IT MEASURES, AND WHY EACH ONE

- **Sustained throughput against a real delivery rate.** Frames are fed
  at the corpus's own measured interval (83.5 ms, 12.0 fps) rather than
  as fast as possible. Feeding faster measures the harness; feeding at
  the real rate measures whether the worker is ever behind.
- **`frames_skipped`.** The single number that says whether a Tower is
  overloaded. There is one slot and the newest frame wins, so a busy
  worker drops frames -- and for Scene Understanding that also stretches
  what the tracker's `max_misses` means, because it is a frame count
  derived from a 1.0 s absence at 12 fps.
- **Per-frame service time**, measured on the worker, not end to end:
  the interesting figure is how long the cartridge occupies its thread,
  because that is what determines whether it keeps up.
- **Event-loop occupancy.** How long `offer_frame` itself takes. It runs
  on the event loop, inline, per frame, and every microsecond there is
  paid by every connection.
- **Resident memory before, during and after.** A live session that grows
  over a long walk is the failure that does not show up in a short test,
  and both cartridges hold state that could: a tracker's track list, a
  dwell's retained frames.
- **CPU seconds**, so "40% of a core" is a measurement rather than an
  estimate.

WHAT IT DOES NOT MEASURE

Accuracy. Nothing here compares a count or a transcript against ground
truth; `scripts/scene_benchmark.py` and
`scripts/document_memory_benchmark.py` own that, and the detector's
recall against an oracle is measured in
`docs/superpowers/research/2026-08-26-detector-oracle-and-the-size-floor.md`.

NOTHING IS WRITTEN unless `--document-root` is given, and Scene
Understanding cannot write at all.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.capture_corpus_benchmark import iter_capture_frames  # noqa: E402

# The corpus's own measured inter-frame gap: 83.5 ms, 12.0 fps, from the
# receipt timestamps in `frames.jsonl` across the 14 captures with more
# than 50 frames. Feeding faster than this measures the harness.
DELIVERED_INTERVAL_S = 0.0835


def _rss_mb() -> float | None:
    """Resident memory, or None if psutil is not installed.

    Optional rather than a dependency: a benchmark that will not run
    without an extra package is a benchmark nobody runs. The growth
    figure is the valuable half and it degrades to None honestly.
    """
    try:
        import psutil
    except Exception:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _cpu_seconds() -> float:
    times = os.times()
    return times.user + times.system


class _Probe:
    """Times `offer_frame` from the caller's side. The event-loop cost."""

    def __init__(self):
        self.samples: list[float] = []

    def record(self, seconds: float) -> None:
        self.samples.append(seconds * 1000.0)

    def summary(self) -> dict:
        if not self.samples:
            return {"count": 0}
        ordered = sorted(self.samples)
        return {
            "count": len(ordered),
            "median_ms": round(statistics.median(ordered), 4),
            "p95_ms": round(ordered[int(len(ordered) * 0.95) - 1], 4),
            "max_ms": round(ordered[-1], 4),
        }


def _drive(session, frames, *, paced: bool, settle_s: float) -> dict:
    """Feed frames into a session and report what it cost.

    `paced` feeds at the delivered interval. Unpaced feeds as fast as the
    loop can, which is the OVERLOAD case -- worth measuring precisely
    because it is what a replay harness does, and because the skip
    behaviour under overload is a property somebody will rely on.
    """
    probe = _Probe()
    rss_before = _rss_mb()
    cpu_before = _cpu_seconds()
    wall_before = time.perf_counter()
    next_at = time.perf_counter()

    for source_seq, payload in frames:
        if paced:
            next_at += DELIVERED_INTERVAL_S
            delay = next_at - time.perf_counter()
            if delay > 0:
                threading.Event().wait(delay)
        started = time.perf_counter()
        session.offer_frame(payload, source_seq=source_seq)
        probe.record(time.perf_counter() - started)

    # Let the worker finish what is in flight, so the counters below
    # describe the whole run rather than a race with the loop.
    deadline = time.perf_counter() + settle_s
    while time.perf_counter() < deadline:
        status = session.status()
        if status["frames_observed"] + status["frames_skipped"] + status[
            "frames_dropped_not_running"
        ] >= status["frames_offered"]:
            break
        threading.Event().wait(0.01)

    wall = time.perf_counter() - wall_before
    cpu = _cpu_seconds() - cpu_before
    rss_after = _rss_mb()
    status = session.status()

    observed = status["frames_observed"]
    return {
        "paced": bool(paced),
        "frames_offered": status["frames_offered"],
        "frames_observed": observed,
        "frames_skipped": status["frames_skipped"],
        "frames_dropped_not_running": status["frames_dropped_not_running"],
        "skip_fraction": (
            round(status["frames_skipped"] / status["frames_offered"], 4)
            if status["frames_offered"]
            else None
        ),
        "wall_seconds": round(wall, 3),
        "observed_fps": round(observed / wall, 2) if wall else None,
        "cpu_seconds": round(cpu, 3),
        # The number a deployment cares about: what fraction of one core
        # this cartridge occupied while the stream ran.
        "cpu_core_fraction": round(cpu / wall, 3) if wall else None,
        "worker_service_ms_mean": (
            round((cpu / observed) * 1000.0, 3) if observed else None
        ),
        "offer_frame_cost": probe.summary(),
        "rss_mb_before": None if rss_before is None else round(rss_before, 1),
        "rss_mb_after": None if rss_after is None else round(rss_after, 1),
        "rss_mb_growth": (
            None
            if rss_before is None or rss_after is None
            else round(rss_after - rss_before, 2)
        ),
        "status": status,
    }


def _scene_session(device: str, orientation: bool, detector: str = "auto"):
    from tower.scene.detect import detector_for
    from tower.scene.engine import SceneEngine
    from tower.scene.live import SceneLive

    def make_engine():
        pose = None
        if orientation:
            from tower.scene.orientation import FaceVisibilityEstimator, model_path

            pose = FaceVisibilityEstimator(model_path())
        return SceneEngine(detector_for(device, detector), facing_estimator=pose)

    return SceneLive(make_engine)


def _document_session(root: str, recogniser: str):
    from tower.document_memory.live import DocumentLive

    factory = None
    if recogniser == "none":
        from tower.document_memory.ocr import FixedTextRecogniser

        def factory():
            return FixedTextRecogniser(pages=[])

    return DocumentLive(root, recogniser_factory=factory)


def _await_running(session, timeout_s: float) -> float:
    """Wait for the model load, and report what it cost.

    Reported rather than excluded: a first-run weight download is real
    time an operator waits, and a benchmark that silently discarded it
    would understate what starting a session costs.
    """
    started = time.perf_counter()
    deadline = started + timeout_s
    while time.perf_counter() < deadline:
        state = session.state
        if state == "running":
            return time.perf_counter() - started
        if state == "failed":
            raise SystemExit(f"the session failed: {session.status()['failure_reason']}")
        threading.Event().wait(0.02)
    raise SystemExit(f"the session never started within {timeout_s}s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--captures",
        default="data/captures",
        help="corpus root; real frames, read only",
    )
    parser.add_argument("--frames", type=int, default=200, help="frames per capture")
    parser.add_argument(
        "--cartridge",
        choices=("scene", "document", "both"),
        default="both",
    )
    parser.add_argument("--device", default="cpu", help="scene detector device")
    parser.add_argument(
        "--detector",
        default="auto",
        help="scene detector: auto (by device), ssdlite320, rtdetr_v2_r18, dfine_s, dfine_m",
    )
    parser.add_argument(
        "--orientation",
        action="store_true",
        help="enable coarse facing (face visibility on person boxes, ~9 ms a face)",
    )
    parser.add_argument(
        "--document-root",
        default=None,
        help=(
            "where a document session may write. WRITES REAL RECORDS. "
            "Omit to skip the document run entirely"
        ),
    )
    parser.add_argument(
        "--document-recogniser",
        choices=("easyocr", "none"),
        default="none",
        help=(
            "'none' substitutes a recogniser that reads nothing, which "
            "measures the CHEAP path -- the one that runs on every frame"
        ),
    )
    parser.add_argument(
        "--unpaced",
        action="store_true",
        help="feed as fast as possible instead of at 12 fps (the overload case)",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help=(
            "cap torch's intra-op thread pool. 0 leaves its default, which "
            "is one thread per core -- measured at 4.1 cores' worth of CPU "
            "for one scene session. PROCESS-GLOBAL: it affects every torch "
            "consumer in this process, which is why it is a knob and not a "
            "default"
        ),
    )
    parser.add_argument("--load-timeout", type=float, default=180.0)
    parser.add_argument("--settle", type=float, default=30.0)
    args = parser.parse_args(argv)

    if args.torch_threads > 0:
        import torch

        torch.set_num_threads(args.torch_threads)

    frames = [
        (seq, payload)
        for _capture, seq, payload in iter_capture_frames(
            args.captures, per_capture_limit=args.frames
        )
    ]
    if not frames:
        print(
            json.dumps(
                {
                    "error": "no frames",
                    "captures": args.captures,
                    "note": (
                        "this benchmark reads real frames and refuses to "
                        "invent them: a synthetic frame contains no COCO "
                        "object and no page, so every figure it produced "
                        "would describe the empty path"
                    ),
                },
                indent=2,
            )
        )
        return 1

    report = {
        "frames": len(frames),
        "captures_root": args.captures,
        "delivered_interval_s": DELIVERED_INTERVAL_S,
        "paced": not args.unpaced,
        "torch_threads": args.torch_threads or "default",
        "runs": {},
    }

    if args.cartridge in ("scene", "both"):
        session = _scene_session(args.device, args.orientation, args.detector)
        session.start()
        load_seconds = _await_running(session, args.load_timeout)
        try:
            run = _drive(session, frames, paced=not args.unpaced, settle_s=args.settle)
        finally:
            session.stop()
        run["load_seconds"] = round(load_seconds, 3)
        run["device"] = args.device
        run["orientation"] = bool(args.orientation)
        # Proof, not assertion: a stopped session must hold no scene.
        run["scene_after_stop"] = session.latest()[0] is None
        report["runs"]["scene"] = run

    if args.cartridge in ("document", "both"):
        if args.document_root is None:
            report["runs"]["document"] = {
                "skipped": "no --document-root; this run would write records"
            }
        else:
            session = _document_session(args.document_root, args.document_recogniser)
            session.start()
            load_seconds = _await_running(session, args.load_timeout)
            try:
                run = _drive(
                    session, frames, paced=not args.unpaced, settle_s=args.settle
                )
            finally:
                session.stop()
            run["load_seconds"] = round(load_seconds, 3)
            run["recogniser"] = args.document_recogniser
            report["runs"]["document"] = run

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
