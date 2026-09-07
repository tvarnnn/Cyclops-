#!/usr/bin/env python
r"""Observe documents in a capture, live or recorded.

Runs in a SEPARATE PROCESS from the Tower, deliberately. OCR costs ~1.2 s
per page and the Tower's frame path must not pay for it -- the same
three-process shape World Builder uses, for the same reason.

Frame sources:

  --frames DIR          a directory of .jpg files, in sorted order
  --follow-capture DIR  tail a capture the Tower is writing RIGHT NOW, so
                        documents are observed while the wearer reads
  --synthetic           rendered pages. SYNTHETIC, NOT PHYSICAL

Nothing here claims anyone READ anything. It records that a page-like
region was present and steady in the camera view -- the camera cannot
establish attention (07-PLATFORM-CONSTRAINTS.md Limitation 8).

    .venv\Scripts\python.exe scripts/document_memory_session.py --synthetic
    .venv\Scripts\python.exe scripts/document_memory_session.py --follow-capture data/capture/captures/<id>
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.capture import CaptureFollower  # noqa: E402
from tower.document_memory.dwell import DwellPolicy  # noqa: E402
from tower.document_memory.engine import DocumentMemoryEngine  # noqa: E402
from tower.document_memory.ocr import (  # noqa: E402
    EasyOcrRecogniser,
    FixedTextRecogniser,
)
from tower.document_memory.store import DocumentStore  # noqa: E402

DEFAULT_ROOT = Path("data/document_memory")


def synthetic_frames(count: int) -> list:
    """Rendered pages, using the test harness -- the only renderer there is."""
    from tests import document_fixtures as fx

    return (
        fx.document_frames(fx.TRANSFORMER_PAPER, count)
        + [fx.encode(fx.no_page_frame())] * 5
        + fx.document_frames(fx.DEPTH_NOTES, count)
        + [fx.encode(fx.no_page_frame())] * 5
    )


def directory_frames(directory: Path) -> list:
    paths = sorted(directory.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no .jpg frames found under {directory}")
    return [path.read_bytes() for path in paths]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Observe documents in a frame stream (never claims reading)."
    )
    parser.add_argument(
        "--root", type=artifact_root_arg, default=str(DEFAULT_ROOT)
    )
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--follow-capture", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-pages", type=int, default=8)
    parser.add_argument(
        "--ocr",
        choices=("easyocr", "none"),
        default="easyocr",
        help=(
            "'none' runs detection and dwell but recognises no text, which "
            "is honest about having read nothing rather than pretending."
        ),
    )
    parser.add_argument(
        "--retention-days",
        type=float,
        default=30.0,
        help="Delete documents older than this. 0 means keep forever.",
    )
    parser.add_argument(
        "--keep-page-images",
        action="store_true",
        help=(
            "Persist the corrected page image. OFF by default: a document's "
            "point is to be readable, and this platform cannot redact."
        ),
    )
    parser.add_argument("--world-id", default=None)
    parser.add_argument("--world-session-id", default=None)
    parser.add_argument(
        "--assumed-fps",
        type=float,
        default=3.3,
        help=(
            "Frame rate to ASSUME for a source with no timestamps "
            "(--frames, --synthetic). Defaults to the rate the glasses "
            "actually deliver. Every document produced this way is stamped "
            "timing_source=assumed-interval so its duration is never "
            "mistaken for a measured one. Ignored for --follow-capture, "
            "which has real receipt times."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="where the OCR reader runs; auto downgrades to cpu, cuda refuses",
    )
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--max-idle-polls", type=int, default=None)
    parser.add_argument("--min-dwell-frames", type=int, default=None)
    parser.add_argument("--min-dwell-seconds", type=float, default=None)
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

    capture_id = None
    if args.follow_capture:
        if not args.follow_capture.exists():
            raise SystemExit(f"no capture directory at {args.follow_capture}")
        follower = CaptureFollower(
            args.follow_capture, poll_seconds=args.poll_seconds
        )
        frames = (
            (frame.raw_bytes, frame.source_seq, frame.received_at)
            for frame in follower.follow(max_idle_polls=args.max_idle_polls)
        )
        capture_id = args.follow_capture.name
        source = "live-capture"
        assumed_interval = None
    else:
        payloads = (
            synthetic_frames(args.synthetic_pages)
            if args.synthetic
            else directory_frames(args.frames)
        )
        frames = ((payload, index, None) for index, payload in enumerate(payloads))
        source = "synthetic" if args.synthetic else "recorded-frames"
        if args.assumed_fps <= 0:
            parser.error("--assumed-fps must be positive")
        assumed_interval = 1.0 / args.assumed_fps

    policy = DwellPolicy()
    if args.min_dwell_frames is not None:
        policy = type(policy)(**{**policy.__dict__, "min_frames": args.min_dwell_frames})
    if args.min_dwell_seconds is not None:
        policy = type(policy)(
            **{**policy.__dict__, "min_seconds": args.min_dwell_seconds}
        )

    recogniser = (
        EasyOcrRecogniser(device=args.device)
        if args.ocr == "easyocr"
        else FixedTextRecogniser()
    )
    retention = None if args.retention_days <= 0 else args.retention_days * 86400.0
    store = DocumentStore(args.root, retention_seconds=retention)
    engine = DocumentMemoryEngine(
        store,
        recogniser,
        policy=policy,
        capture_id=capture_id,
        # Supplied, never derived. This cartridge cannot see a world.
        world_id=args.world_id,
        world_session_id=args.world_session_id,
        keep_page_images=args.keep_page_images,
        assumed_frame_interval_s=assumed_interval,
    )

    started = time.perf_counter()
    recorded = []
    try:
        for payload, source_seq, received_at in frames:
            result = engine.observe(
                payload, received_at=received_at, source_seq=source_seq
            )
            if result.document_id:
                recorded.append(result.document_id)
        final = engine.flush()
        if final:
            recorded.append(final)
    finally:
        engine.release()
    elapsed = time.perf_counter() - started

    pruned = store.prune_expired()

    report = {
        "root": str(args.root),
        "frame_source": source,
        "capture_id": capture_id,
        "ocr": recogniser.name,
        "timing_source": (
            "capture-journal" if assumed_interval is None else "assumed-interval"
        ),
        "assumed_frame_interval_s": assumed_interval,
        "frames_observed": engine.frames_observed,
        "documents_observed": len(recorded),
        "document_ids": recorded,
        "pruned_expired": pruned["documents_removed"],
        "pruned_images_removed": pruned["images_removed"],
        # A retention pass that could not delete the pixels must say so.
        "prune_complete": pruned["complete"],
        "prune_images_retained": pruned["images_retained"],
        "stored_documents": store.count(),
        "bytes": store.bytes_used(),
        "keep_page_images": args.keep_page_images,
        "seconds": round(elapsed, 3),
        "ms_per_frame": round(
            elapsed * 1000 / max(engine.frames_observed, 1), 3
        ),
    }

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{key:22s} {value}")
        print(
            "\nOBSERVED, NOT READ: a page was in view and steady. The camera "
            "cannot establish that anyone looked at it."
        )
        if source == "synthetic":
            print("SYNTHETIC, NOT PHYSICAL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
