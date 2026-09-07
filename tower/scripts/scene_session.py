#!/usr/bin/env python
r"""What is around the wearer, from a frame stream.

Runs in a SEPARATE PROCESS from the Tower. Detection costs ~17 ms on CUDA
(RT-DETRv2-R18) or ~30 ms on CPU (SSDLite320); the optional orientation
stage ~9 ms per tracked person on CPU.

Frame sources:

  --frames DIR          a directory of .jpg files, in sorted order
  --follow-capture DIR  tail a capture the Tower is writing RIGHT NOW
  --synthetic           rendered frames. SYNTHETIC, NOT PHYSICAL, and
                        containing no COCO objects, so it exercises the
                        pipeline and detects nothing -- which is the
                        honest outcome, not a bug

Nothing is persisted. This cartridge answers "what is around me NOW"; a
durable record is Environmental Memory's job.

    .venv\Scripts\python.exe scripts/scene_session.py --synthetic
    .venv\Scripts\python.exe scripts/scene_session.py --follow-capture <dir> --facing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.capture import CaptureFollower  # noqa: E402
from tower.scene.detect import (  # noqa: E402
    SCORE_THRESHOLD,
    FixedDetector,
    TorchvisionDetector,
)
from tower.scene.engine import ORIENTATION_INTERVAL_S, SceneEngine  # noqa: E402
from tower.scene.orientation import FaceVisibilityEstimator, model_path  # noqa: E402
from tower.scene.query import SceneQuery  # noqa: E402
from tower.scene.tracking import TrackerPolicy  # noqa: E402


def synthetic_frames(count: int):
    """Rendered room frames. They contain no COCO object, deliberately.

    There is no imagery of people anywhere on this host, so a synthetic
    run exercises detection, tracking and the query layer and honestly
    detects nothing. Pretending otherwise would need fabricated people.
    """
    from tests import synthetic_scene as ss

    scene = ss.furnished_room()
    poses = ss.strafe(count, step=0.05)
    images = ss.render_sequence(scene, poses, ss.camera_matrix(640, 360), 640, 360)
    return [ss.encode_jpeg(image) for image in images]


def decode(payload: bytes):
    array = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Maintain a live scene state from a frame stream."
    )
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--follow-capture", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-frames", type=int, default=12)
    parser.add_argument(
        "--detector",
        choices=("ssdlite320", "none"),
        default="ssdlite320",
        help="'none' runs tracking and the query layer over no detections.",
    )
    parser.add_argument(
        "--facing",
        action="store_true",
        help=(
            "Estimate whether each tracked person appears to face the wearer, "
            "from a face detector on their box (~9 ms a face, CPU). "
            "Orientation evidence, never gaze."
            "orientation evidence, never gaze."
        ),
    )
    parser.add_argument(
        "--facing-interval",
        type=float,
        default=ORIENTATION_INTERVAL_S,
        help="Seconds between orientation estimates.",
    )
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    # Defaulted from the policy rather than repeated. This driver used
    # to say `default=5`, which is how a constant that was corrected in
    # one place goes on shipping from another.
    parser.add_argument(
        "--min-hits", type=int, default=TrackerPolicy.min_hits
    )
    parser.add_argument(
        "--max-misses", type=int, default=TrackerPolicy.max_misses
    )
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--max-idle-polls", type=int, default=None)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    chosen = [
        name
        for name, value in (
            ("--frames", args.frames),
            ("--follow-capture", args.follow_capture),
            ("--synthetic", args.synthetic),
        )
        if value
    ]
    if len(chosen) != 1:
        parser.error(
            "exactly one of --frames, --follow-capture or --synthetic is required"
        )

    if args.follow_capture:
        if not args.follow_capture.exists():
            raise SystemExit(f"no capture directory at {args.follow_capture}")
        follower = CaptureFollower(
            args.follow_capture, poll_seconds=args.poll_seconds
        )
        stream = (
            (frame.raw_bytes, frame.received_at)
            for frame in follower.follow(max_idle_polls=args.max_idle_polls)
        )
        source = "live-capture"
    elif args.synthetic:
        stream = ((payload, None) for payload in synthetic_frames(args.synthetic_frames))
        source = "synthetic"
    else:
        paths = sorted(args.frames.glob("*.jpg"))
        if not paths:
            raise SystemExit(f"no .jpg frames found under {args.frames}")
        stream = ((path.read_bytes(), None) for path in paths)
        source = "recorded-frames"

    detector = (
        TorchvisionDetector(score_threshold=args.score_threshold)
        if args.detector == "ssdlite320"
        else FixedDetector()
    )
    pose = FaceVisibilityEstimator(model_path()) if args.facing else None

    engine = SceneEngine(
        detector,
        TrackerPolicy(min_hits=args.min_hits, max_misses=args.max_misses),
        facing_estimator=pose,
        orientation_interval_s=args.facing_interval,
        score_threshold=args.score_threshold,
    )
    engine.load()

    started = time.perf_counter()
    state = None
    try:
        for payload, received_at in stream:
            frame = decode(payload)
            if frame is None:
                continue
            state = engine.observe(frame, received_at=received_at)
    finally:
        engine.release()
    elapsed = time.perf_counter() - started

    if state is None:
        raise SystemExit("no decodable frames in the stream")

    query = SceneQuery(state)
    report = {
        "frame_source": source,
        "detector": detector.name,
        "orientation_enabled": engine.orientation_enabled,
        "frames_observed": engine.frames_observed,
        "seconds": round(elapsed, 3),
        "ms_per_frame": round(elapsed * 1000 / max(engine.frames_observed, 1), 3),
        "scene": engine.describe(state),
        "answers": {
            "people": query.count("person").to_json_dict(),
            "chairs": query.count("chair").to_json_dict(),
            "facing_wearer": query.facing_wearer().to_json_dict(),
        },
        "persisted": False,
    }

    if args.format == "json":
        print(json.dumps(report, indent=2))
        return 0

    print(f"{'frame source':22s} {source}")
    print(f"{'detector':22s} {detector.name}")
    print(f"{'orientation':22s} {'on' if engine.orientation_enabled else 'off'}")
    print(f"{'frames observed':22s} {engine.frames_observed}")
    print(f"{'ms per frame':22s} {report['ms_per_frame']}")
    print(f"\ncounts (from CONFIRMED TRACKS, not detections):")
    if not state.counts:
        print("  nothing detected")
    for label, count in sorted(state.counts.items()):
        print(f"  {label:16s} {count}")

    facing = query.facing_wearer()
    print("\nhow many appear to be facing the wearer:")
    print(f"  {facing.value if facing.answered else 'NOT ANSWERED'}")
    print(f"  {facing.reason}")

    print("\nNOTHING PERSISTED. This is a live state, not a memory.")
    print("Positions and relations are CAMERA-RELATIVE: there is no live world pose.")
    if source == "synthetic":
        print("SYNTHETIC, NOT PHYSICAL: the rendered room contains no COCO object.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
