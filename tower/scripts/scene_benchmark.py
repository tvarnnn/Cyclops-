#!/usr/bin/env python
r"""What Scene Understanding costs, and how a count behaves under dropout.

SYNTHETIC, NOT PHYSICAL. The timings are real measurements of this code on
this machine. The imagery is rendered and contains **no COCO object**, so
this measures the pipeline's cost, never its accuracy on a real room --
there is no imagery of people anywhere on this host.

Two things are measured:

  1. Per-frame cost, with and without the orientation stage. The gap
     between them is the whole reason orientation is off by default.
  2. Count stability under simulated detector dropout -- the property the
     brief singles out, measured rather than asserted.

    .venv\Scripts\python.exe scripts/scene_benchmark.py
    .venv\Scripts\python.exe scripts/scene_benchmark.py --format json
    .venv\Scripts\python.exe scripts/scene_benchmark.py --no-models
"""

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.scene.detect import FixedDetector, TorchvisionDetector  # noqa: E402
from tower.scene.engine import SceneEngine  # noqa: E402
from tower.scene.records import BoundingBox, Detection  # noqa: E402
from tower.scene.tracking import (  # noqa: E402
    DELIVERED_FRAME_INTERVAL_S,
    Tracker,
)

RESOLUTIONS = [(640, 360), (896, 504), (1280, 720)]


def _timed(fn, repeat: int) -> dict:
    fn()  # untimed warm-up
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": round(statistics.mean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "max_ms": round(max(samples), 3),
    }


def _room(width, height):
    from tests import synthetic_scene as ss

    scene = ss.furnished_room()
    pose = ss.strafe(1, step=0.05)
    image = ss.render_sequence(scene, pose, ss.camera_matrix(width, height), width, height)[0]
    return image


def bench_detection(repeat: int) -> list[dict]:
    detector = TorchvisionDetector()
    with contextlib.redirect_stdout(sys.stderr):
        detector.load()
    rows = []
    try:
        for width, height in RESOLUTIONS:
            frame = _room(width, height)
            rows.append(
                {
                    "resolution": f"{width}x{height}",
                    "detect": _timed(lambda: detector.detect(frame), repeat),
                    "detections": len(detector.detect(frame)),
                }
            )
    finally:
        detector.release()
    return rows


def bench_orientation(repeat: int) -> dict:
    """The 23x that decides the default."""
    from tower.scene.orientation import FaceVisibilityEstimator, model_path
    from tower.scene.records import BoundingBox

    estimator = FaceVisibilityEstimator(model_path())
    start = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        estimator.load()
    load_seconds = time.perf_counter() - start
    frame = _room(640, 360)
    try:
        timing = _timed(lambda: estimator.estimate(frame, [BoundingBox(40, 40, 200, 320)]), max(repeat // 6, 1))
    finally:
        estimator.release()
    return {"load_seconds": round(load_seconds, 2), "estimate": timing}


def bench_tracker_overhead(repeat: int) -> dict:
    """The cheap half, so its share of the budget is visible."""
    detections = [
        Detection("person", 0.9, BoundingBox(60, 80, 160, 300)),
        Detection("person", 0.9, BoundingBox(280, 90, 380, 305)),
        Detection("chair", 0.8, BoundingBox(470, 200, 580, 340)),
    ]
    frame = np.zeros((360, 640, 3), np.uint8)
    engine = SceneEngine(
        FixedDetector([detections] * 10_000), clock=lambda: 0.0
    )
    engine.load()
    counter = {"at": 0.0}

    def step():
        counter["at"] += DELIVERED_FRAME_INTERVAL_S
        engine.observe(frame, received_at=counter["at"])

    return _timed(step, repeat)


def bench_count_stability(
    dropout_rates=(0.0, 0.1, 0.2, 0.4, 0.6),
) -> list[dict]:
    """Does the count hold when the detector misses frames?

    The brief's requirement, measured rather than asserted. A count that
    flickers with the detector makes the wearer distrust every answer.

    The tracker is built with **no policy argument on purpose**: this
    bench used to construct `TrackerPolicy(min_hits=3, max_misses=5)`
    by hand, so it went on reporting the old constants' behaviour after
    they changed, and a benchmark that cannot see a regression cannot
    guard against one. Frames are stepped at the real delivered
    interval for the same reason -- it used to step 0.3 s, the ~3.3 fps
    that was never the rate.

    The 60% row is new and is where the difference lives: at 40% the
    old constants already read 0.939, but at 60% they collapse to 0.252
    while a miss budget of one real second holds 0.783.
    """
    rows = []
    both = [
        Detection("person", 0.9, BoundingBox(60, 80, 160, 300)),
        Detection("person", 0.9, BoundingBox(280, 90, 380, 305)),
    ]
    for rate in dropout_rates:
        rng = np.random.default_rng(7)
        tracker = Tracker()
        counts = []
        at = 0.0
        for index in range(120):
            visible = [
                detection for detection in both if rng.random() >= rate
            ]
            tracker.update(visible, at=at)
            at += DELIVERED_FRAME_INTERVAL_S
            if index >= 5:
                counts.append(tracker.count("person"))
        rows.append(
            {
                "dropout_rate": rate,
                "correct_count": 2,
                "modal_count": max(set(counts), key=counts.count),
                "distinct_counts": sorted(set(counts)),
                "fraction_correct": round(
                    sum(1 for value in counts if value == 2) / len(counts), 4
                ),
            }
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Scene Understanding's cost and count stability."
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Skip the model-backed benchmarks (needs the [ml] extra).",
    )
    args = parser.parse_args(argv)

    report = {
        "note": (
            "SYNTHETIC, NOT PHYSICAL. The rendered room contains no COCO "
            "object, so this measures cost, never accuracy on a real room."
        ),
        "repeat": args.repeat,
        "tracker_and_query_only": bench_tracker_overhead(args.repeat),
        "count_stability_under_dropout": bench_count_stability(),
    }
    if not args.no_models:
        report["detection"] = bench_detection(args.repeat)
        report["orientation"] = bench_orientation(args.repeat)

    if args.format == "json":
        print(json.dumps(report, indent=2))
        return 0

    print(report["note"])
    cheap = report["tracker_and_query_only"]
    print(f"\n=== The cheap half (tracking + relations + query) ===")
    print(f"  mean {cheap['mean_ms']:.3f} ms   max {cheap['max_ms']:.3f} ms")

    if "detection" in report:
        print("\n=== Detection (every frame) ===")
        print(f"{'resolution':14s}{'mean ms':>10s}{'max ms':>10s}  detections")
        for row in report["detection"]:
            print(
                f"{row['resolution']:14s}{row['detect']['mean_ms']:>10.2f}"
                f"{row['detect']['max_ms']:>10.2f}  {row['detections']}"
            )
        orientation = report["orientation"]
        print(
            f"\n=== Orientation (optional, OFF by default) ==="
            f"\n  load {orientation['load_seconds']}s, "
            f"estimate {orientation['estimate']['mean_ms']:.0f} ms"
        )
        detect_ms = report["detection"][0]["detect"]["mean_ms"]
        print(
            f"  {orientation['estimate']['mean_ms'] / max(detect_ms, 1e-6):.0f}x "
            "the detector -- which is why it runs at a cadence, not per frame"
        )

    print("\n=== Count stability under detector dropout ===")
    print(f"{'dropout':>10s}{'modal':>8s}{'seen':>16s}{'fraction correct':>18s}")
    for row in report["count_stability_under_dropout"]:
        print(
            f"{row['dropout_rate']:>10.0%}{row['modal_count']:>8d}"
            f"{str(row['distinct_counts']):>16s}{row['fraction_correct']:>18.3f}"
        )
    print(
        "\nThe correct count is 2 throughout. A count taken from raw "
        "detections would follow the dropout column exactly."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
