"""Replay real frames through the Document Memory engine and measure it.

What the session script cannot tell you: how many frames the steadiness
gate admitted, how often the text detector ran, how many dwells opened,
how long each stage took, and what the process cost in memory. This
reports all of that, over a capture directory or a directory of jpegs,
and optionally with a POSITIVE CONTROL -- a rendered page with known
text composited onto a run of real frames, so a real-footage replay can
show that the pipeline still fires when a page is actually there.

Nothing here writes into the source. The store goes under --root, which
is guarded like every other artifact root in this repository.

    python scripts/document_memory_replay.py --capture data/captures/<id> \
        --root C:/Users/<you>/Projects/Glasses-scratch/dm-replay/<id>
    python scripts/document_memory_replay.py --capture ... --inject-page \
        --inject-at 200 --inject-frames 40
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

TOWER = Path(__file__).resolve().parents[1]
if str(TOWER) not in sys.path:
    sys.path.insert(0, str(TOWER))

from tower.artifact_paths import artifact_root_arg  # noqa: E402

DEFAULT_ROOT = Path("data/document_memory_replay")


def _capture_frames(capture_dir: Path):
    """(bytes, received_at, source_seq) from a recorder capture, in order."""
    journal = capture_dir / "frames.jsonl"
    with journal.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            path = capture_dir / record["relpath"]
            if not path.exists():
                continue
            yield path.read_bytes(), float(record["received_at"]), int(record["source_seq"])


def _directory_frames(directory: Path, fps: float):
    at = 0.0
    for index, path in enumerate(sorted(directory.glob("*.jpg"))):
        at += 1.0 / fps
        yield path.read_bytes(), at, index


def _inject(frames, *, at_index: int, count: int, lines):
    """Composite a rendered page onto `count` consecutive frames."""
    import cv2
    import numpy as np

    from tests import document_fixtures as fx

    page = fx.render_page(lines)
    rng = np.random.default_rng(7)
    for index, (raw, received_at, seq) in enumerate(frames):
        if at_index <= index < at_index + count:
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            height, width = frame.shape[:2]
            base = fx._default_corners(width, height, 0.1)
            corners = base + rng.normal(0.0, 1.2, base.shape).astype(np.float32)
            frame, _ = _composite(frame, page, corners)
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            raw = buffer.tobytes()
        yield raw, received_at, seq


def _composite(frame, page, corners):
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    page_height, page_width = page.shape[:2]
    source = np.array(
        [[0, 0], [page_width - 1, 0], [page_width - 1, page_height - 1], [0, page_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    warped = cv2.warpPerspective(page, matrix, (width, height))
    mask = cv2.warpPerspective(
        np.full((page_height, page_width), 255, np.uint8), matrix, (width, height)
    )
    out = frame.copy()
    out[mask > 0] = warped[mask > 0]
    return out, corners


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        try:
            import resource  # type: ignore

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return None


def _gpu_mb():
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "allocated": round(torch.cuda.memory_allocated() / 1e6, 1),
            "reserved": round(torch.cuda.memory_reserved() / 1e6, 1),
            "peak_reserved": round(torch.cuda.max_memory_reserved() / 1e6, 1),
        }
    except Exception:
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=artifact_root_arg, default=str(DEFAULT_ROOT))
    parser.add_argument("--capture", type=Path, default=None, help="a recorder capture directory")
    parser.add_argument("--frames", type=Path, default=None, help="a directory of jpegs")
    parser.add_argument("--fps", type=float, default=12.0, help="assumed rate for --frames")
    parser.add_argument("--ocr", choices=("easyocr", "none"), default="easyocr")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many frames")
    parser.add_argument("--inject-page", action="store_true")
    parser.add_argument("--inject-at", type=int, default=100)
    parser.add_argument("--inject-frames", type=int, default=40)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    if (args.capture is None) == (args.frames is None):
        parser.error("give exactly one of --capture or --frames")

    from tower.document_memory.engine import DocumentMemoryEngine
    from tower.document_memory.ocr import EasyOcrRecogniser, FixedTextRecogniser
    from tower.document_memory.store import DocumentStore
    from tests import document_fixtures as fx

    recogniser = (
        EasyOcrRecogniser(device=args.device)
        if args.ocr == "easyocr"
        else FixedTextRecogniser(pages=[fx.page_regions(fx.TRANSFORMER_PAPER)])
    )
    load_started = time.perf_counter()
    if hasattr(recogniser, "load"):
        recogniser.load()
    load_seconds = time.perf_counter() - load_started

    store = DocumentStore(args.root)
    engine = DocumentMemoryEngine(store, recogniser)

    frames = (
        _capture_frames(args.capture)
        if args.capture is not None
        else _directory_frames(args.frames, args.fps)
    )
    if args.inject_page:
        frames = _inject(
            frames,
            at_index=args.inject_at,
            count=args.inject_frames,
            lines=fx.TRANSFORMER_PAPER,
        )

    per_frame_ms = []
    stable = 0
    detected = 0
    outcomes = {}
    documents = []
    rss_before = _rss_mb()
    started = time.perf_counter()
    count = 0
    for raw, received_at, seq in frames:
        if args.limit and count >= args.limit:
            break
        count += 1
        t0 = time.perf_counter()
        result = engine.observe(raw, received_at=received_at, source_seq=seq)
        per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
        verdict = engine.last_gate_verdict
        if verdict is not None and verdict.stable:
            stable += 1
        if result.page_detected:
            detected += 1
        outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
        if result.document_id:
            documents.append((result.document_id, result.outcome))
    flushed = engine.flush()
    if flushed:
        documents.append((flushed, "flushed"))
    elapsed = time.perf_counter() - started
    gpu = _gpu_mb()
    # Before release: the recogniser forgets its device when it lets
    # the model go.
    device = getattr(recogniser, "device", None) or "n/a"
    release_started = time.perf_counter()
    engine.release()
    release_seconds = time.perf_counter() - release_started
    gpu_after = _gpu_mb()

    stored = store.read_all()
    per_frame_ms.sort()

    def pct(p):
        if not per_frame_ms:
            return None
        return round(per_frame_ms[min(len(per_frame_ms) - 1, int(p * len(per_frame_ms)))], 2)

    report = {
        "source": str(args.capture or args.frames),
        "frames": count,
        "seconds": round(elapsed, 2),
        "ms_per_frame": {
            "mean": round(statistics.mean(per_frame_ms), 2) if per_frame_ms else None,
            "p50": pct(0.5),
            "p95": pct(0.95),
            "max": round(per_frame_ms[-1], 2) if per_frame_ms else None,
        },
        "gate": {
            "stable_frames": stable,
            "stable_fraction": round(stable / count, 4) if count else None,
            "detector_runs": engine.detections,
            "frames_with_region": detected,
            "dwells_qualified": len(documents),
            "pages_turned": engine.pages_turned,
            "pages_ocred": engine.pages_ocred,
        },
        "outcomes": outcomes,
        "documents_recorded": engine.documents_recorded,
        "documents_resighted": engine.documents_resighted,
        "dwells_unreadable": engine.dwells_unreadable,
        "stored": [
            {
                "document_id": d.document_id,
                "title": d.title,
                "pages": d.pages_observed,
                "readable_pages": sum(1 for p in d.pages if p.readable),
                "words": d.word_count,
                "sightings": d.sighting_count,
                "observed_seconds": round(d.observed_seconds, 2),
            }
            for d in stored
        ],
        "injected": (
            {"at": args.inject_at, "frames": args.inject_frames, "text": fx.page_text(fx.TRANSFORMER_PAPER)}
            if args.inject_page
            else None
        ),
        "ocr": {
            "engine": recogniser.name,
            "device": device,
            "load_seconds": round(load_seconds, 2),
            "release_seconds": round(release_seconds, 3),
        },
        "memory": {
            "rss_mb_before": rss_before,
            "rss_mb_after": _rss_mb(),
            "gpu_mb_during": gpu,
            "gpu_mb_after_release": gpu_after,
        },
    }
    if args.inject_page and stored:
        from tests.test_document_ocr_integration import word_recall

        report["injected"]["word_recall"] = round(
            max(word_recall(fx.page_text(fx.TRANSFORMER_PAPER), d.text) for d in stored), 3
        )

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(f"{report['frames']} frames in {report['seconds']} s, {report['ms_per_frame']}")
        print(f"gate: {report['gate']}")
        print(f"outcomes: {report['outcomes']}")
        print(f"stored: {len(stored)} document(s)")
        for entry in report["stored"]:
            print(f"  {entry}")
        if report["injected"]:
            print(f"injected: {report['injected']}")
        print(f"ocr: {report['ocr']}")
        print(f"memory: {report['memory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
