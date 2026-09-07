#!/usr/bin/env python
r"""Replay recorded glasses frames through the real Scene Understanding engine.

Deterministic, offline, and honest about what it can and cannot measure.

    .venv\Scripts\python.exe scripts/scene_replay.py --captures data/captures --per-capture 200
    .venv\Scripts\python.exe scripts/scene_replay.py --captures <dir> --device cuda --fixture labeled_fixture.json
    .venv\Scripts\python.exe scripts/scene_replay.py --frames <dir-of-jpgs> --format json

Two kinds of number come out, and the report keeps them apart:

  EXPLORATORY (any footage, no labels): detector / orientation / total
  latency per frame, effective throughput, counts per class over time,
  tracks created, count changes per minute, CPU seconds, RSS and VRAM.
  These describe COST and STABILITY, never accuracy.

  LABELLED (only with --fixture): a JSON list of {capture_id, frame_file,
  counts: {label: n}, bystander_count, wearer_body_visible} entries, and
  for every frame in it the engine's counts are compared with the
  labels -- exact-match rate and mean absolute error per class, plus how
  many person boxes were routed to `partial_bottom_edge` in frames where
  the label says the wearer's own body is visible and no bystander is.
  Labels from `Glasses-scratch/scene-understanding-v1/corpus-audit/` were
  made by an AI agent inspecting frames, not by a human annotator, and
  the report says so.

Frames are read in capture order and fed with their recorded receipt
timestamps, so the tracker's absence windows are evaluated at the
delivery rate the glasses actually produce, not as fast as the disk
reads. Nothing is written except the report the caller asks for.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.scene.detect import (  # noqa: E402
    CLASSES_OF_INTEREST,
    detector_for,
    resolve_choice,
)
from tower.scene.engine import SceneEngine  # noqa: E402
from tower.scene.orientation import FaceVisibilityEstimator, model_path  # noqa: E402


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def _vram_mb() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
            "vram_reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
            "vram_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        }
    except Exception:
        return {}


def _summary(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0}
    ordered = sorted(samples)
    return {
        "n": len(samples),
        "median_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "max_ms": round(ordered[-1], 2),
        "mean_ms": round(statistics.fmean(ordered), 2),
    }


def iter_frames(captures: Path | None, frames_dir: Path | None, per_capture: int | None):
    """Yield (capture_id, frame_name, jpeg_bytes, received_at) in order."""
    if frames_dir is not None:
        paths = sorted(frames_dir.glob("*.jpg"))
        for index, path in enumerate(paths):
            yield frames_dir.name, path.name, path.read_bytes(), index * 0.0835
        return
    for capture_dir in sorted(p for p in captures.iterdir() if p.is_dir()):
        journal = capture_dir / "frames.jsonl"
        frames = capture_dir / "frames"
        if not frames.is_dir():
            continue
        entries = []
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                entries.append((record["relpath"].split("/")[-1], float(record["received_at"])))
        else:
            entries = [(p.name, i * 0.0835) for i, p in enumerate(sorted(frames.glob("*.jpg")))]
        if per_capture is not None:
            entries = entries[:per_capture]
        for name, received_at in entries:
            path = frames / name
            if path.exists():
                yield capture_dir.name, name, path.read_bytes(), received_at


class _Timed:
    """Wraps a detector or estimator to time each call."""

    def __init__(self, inner):
        self._inner = inner
        self.samples: list[float] = []
        self.name = getattr(inner, "name", "unknown")
        self.score_threshold = getattr(inner, "score_threshold", None)

    def load(self):
        self._inner.load()

    def release(self):
        self._inner.release()

    def _sync(self):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def detect(self, frame):
        self._sync()
        started = time.perf_counter()
        result = self._inner.detect(frame)
        self._sync()
        self.samples.append((time.perf_counter() - started) * 1000)
        return result

    def estimate(self, frame, boxes):
        started = time.perf_counter()
        result = self._inner.estimate(frame, boxes)
        self.samples.append((time.perf_counter() - started) * 1000)
        return result


def run(args) -> dict:
    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    choice = resolve_choice(device, args.detector)
    detector = _Timed(detector_for(device, choice))
    facing = None
    if args.orientation:
        path = model_path()
        if path is None:
            print("no face model found; running without orientation", file=sys.stderr)
        else:
            facing = _Timed(FaceVisibilityEstimator(path))
    engine = SceneEngine(detector, facing_estimator=facing)

    rss_before = _rss_mb()
    cpu_before = time.process_time()
    load_started = time.perf_counter()
    engine.load()
    load_seconds = time.perf_counter() - load_started

    fixture = {}
    if args.fixture:
        for entry in json.loads(Path(args.fixture).read_text(encoding="utf-8")):
            fixture[(entry["capture_id"], entry["frame_file"])] = entry

    totals: list[float] = []
    decode: list[float] = []
    counts_over_time: dict[str, list[int]] = {label: [] for label in CLASSES_OF_INTEREST}
    count_changes = 0
    previous_counts: dict | None = None
    facing_over_time: list[int | None] = []
    partial_over_time: list[int] = []
    seen_ids: set = set()
    labelled_rows = []
    frames_read = 0
    # Footage time is summed PER CAPTURE: captures are days apart, and
    # the span from the first capture's first frame to the last capture's
    # last frame is calendar time, not footage.
    spans: dict[str, list[float]] = {}
    wall_started = time.perf_counter()

    for capture_id, name, raw, received_at in iter_frames(
        args.captures, args.frames, args.per_capture
    ):
        started = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        decode.append((time.perf_counter() - started) * 1000)
        if frame is None:
            continue
        frames_read += 1
        span = spans.setdefault(capture_id, [received_at, received_at])
        span[1] = received_at
        started = time.perf_counter()
        state = engine.observe(frame, received_at=received_at)
        totals.append((time.perf_counter() - started) * 1000)

        for label in CLASSES_OF_INTEREST:
            counts_over_time[label].append(state.count(label))
        if previous_counts is not None and state.counts != previous_counts:
            count_changes += 1
        previous_counts = dict(state.counts)
        for track in state.tracks:
            seen_ids.add(track.track_id)
        partial_over_time.append(len(state.partial_people))
        facing_over_time.append(
            len(state.facing_wearer()) if state.orientation_enabled else None
        )

        label = fixture.get((capture_id, name))
        if label is not None:
            labelled_rows.append(
                {
                    "capture_id": capture_id,
                    "frame": name,
                    "labels": label.get("counts", {}),
                    "bystander_count": label.get("bystander_count"),
                    "wearer_body_visible": label.get("wearer_body_visible"),
                    "engine": {k: state.count(k) for k in CLASSES_OF_INTEREST},
                    "partial_bottom_edge": len(state.partial_people),
                    "engine_detections_person_raw": sum(
                        1 for track in state.tracks if track.label == "person"
                    )
                    + len(state.partial_people),
                }
            )
        if args.max_frames and frames_read >= args.max_frames:
            break

    wall = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_before
    stop_started = time.perf_counter()
    engine.release()
    release_seconds = time.perf_counter() - stop_started

    minutes = sum(max(end - start, 0.0) for start, end in spans.values()) / 60.0
    report = {
        "kind": "exploratory-unless-labelled",
        "device": device,
        "detector": detector.name,
        "score_threshold": detector.score_threshold,
        "orientation": None if facing is None else facing.name,
        "frames": frames_read,
        "footage_minutes": round(minutes, 3),
        "load_seconds": round(load_seconds, 3),
        "release_seconds": round(release_seconds, 3),
        "wall_seconds": round(wall, 3),
        "effective_fps_unpaced": round(frames_read / wall, 2) if wall else None,
        "cpu_seconds": round(cpu_seconds, 2),
        "cpu_cores_mean": round(cpu_seconds / wall, 2) if wall else None,
        "rss_mb_before": rss_before,
        "rss_mb_after": _rss_mb(),
        **_vram_mb(),
        "latency_ms": {
            "decode": _summary(decode),
            "detector": _summary(detector.samples),
            "orientation_per_call": _summary(facing.samples) if facing else {"n": 0},
            "observe_total": _summary(totals),
        },
        "stability": {
            "count_changes_per_minute": round(count_changes / minutes, 2) if minutes else None,
            "tracks_created": len(seen_ids),
            "tracks_created_per_minute": round(len(seen_ids) / minutes, 2) if minutes else None,
            "mean_counts": {
                label: round(statistics.fmean(values), 3) if values else 0.0
                for label, values in counts_over_time.items()
            },
            "frames_with_person": sum(1 for v in counts_over_time["person"] if v > 0),
            "frames_with_partial_bottom_edge": sum(1 for v in partial_over_time if v > 0),
            "frames_with_facing": sum(1 for v in facing_over_time if v),
        },
    }
    if fixture:
        report["labelled"] = _score_labelled(labelled_rows)
    return report


def _score_labelled(rows: list[dict]) -> dict:
    if not rows:
        return {"frames": 0, "note": "no fixture frame was replayed"}
    per_class = {}
    for label in CLASSES_OF_INTEREST:
        pairs = [
            (row["labels"].get(label, row["labels"].get(label.replace(" ", "_"), 0)), row["engine"][label])
            for row in rows
        ]
        exact = sum(1 for truth, got in pairs if truth == got)
        under = sum(1 for truth, got in pairs if got < truth)
        over = sum(1 for truth, got in pairs if got > truth)
        per_class[label] = {
            "frames": len(pairs),
            "labelled_total": sum(truth for truth, _ in pairs),
            "engine_total": sum(got for _, got in pairs),
            "exact_rate": round(exact / len(pairs), 3),
            "undercount_rate": round(under / len(pairs), 3),
            "overcount_rate": round(over / len(pairs), 3),
            "mae": round(statistics.fmean(abs(truth - got) for truth, got in pairs), 3),
        }
    wearer_frames = [
        row for row in rows if row.get("wearer_body_visible") and not row.get("bystander_count")
    ]
    return {
        "frames": len(rows),
        "labels_from": (
            "an AI agent inspecting frames (corpus audit 2026-09-07), not a "
            "human annotator"
        ),
        "per_class": per_class,
        "wearer_body_frames": {
            "frames": len(wearer_frames),
            "false_person_in_count": sum(1 for row in wearer_frames if row["engine"]["person"] > 0),
            "routed_to_partial_bottom_edge": sum(
                1 for row in wearer_frames if row["partial_bottom_edge"] > 0
            ),
            "any_person_box_at_all": sum(
                1 for row in wearer_frames if row["engine_detections_person_raw"] > 0
            ),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--captures", type=Path, help="a captures root (read only)")
    source.add_argument("--frames", type=Path, help="a directory of .jpg frames (read only)")
    parser.add_argument("--per-capture", type=int, default=None, help="frames per capture")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--detector", default="auto", help="auto, ssdlite320, rtdetr_v2_r18, dfine_s, dfine_m")
    parser.add_argument("--no-orientation", dest="orientation", action="store_false")
    parser.add_argument("--fixture", type=Path, default=None, help="labelled fixture JSON")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--out",
        type=artifact_root_arg,
        default=None,
        help="directory to write report.json into (guarded: never a drive root)",
    )
    args = parser.parse_args(argv)

    report = run(args)
    if args.out is not None:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "scene_replay_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    if args.format == "json":
        print(json.dumps(report, indent=2))
        return 0
    print(f"device {report['device']}  detector {report['detector']} @ {report['score_threshold']}  orientation {report['orientation']}")
    print(f"frames {report['frames']}  footage {report['footage_minutes']} min  load {report['load_seconds']} s  release {report['release_seconds']} s")
    for stage, summary in report["latency_ms"].items():
        print(f"  {stage:20s} {summary}")
    print(f"unpaced fps {report['effective_fps_unpaced']}  cpu cores {report['cpu_cores_mean']}  rss {report['rss_mb_before']} -> {report['rss_mb_after']} MB  {', '.join(f'{k} {v}' for k, v in report.items() if k.startswith('vram'))}")
    print("stability", json.dumps(report["stability"]))
    if "labelled" in report:
        print("LABELLED", json.dumps(report["labelled"], indent=1))
    print("exploratory numbers are cost and stability, never accuracy; labelled numbers are against an AI-made fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
