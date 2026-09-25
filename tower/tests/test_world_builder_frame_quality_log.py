"""P5-SHARP: the per-frame quality log (`world_builder/frame_quality.py`,
`TOWER_WORLD_FRAME_QUALITY_LOG`, off by default).

THE GOLDEN: with the switch unset, blank, off, or garbage, a builder session's whole output --
every file under the world directory -- is what the product wrote BEFORE the log existed,
recorded from the ceb203a product tree (clean) by
RUN/experiments/P5-SHARP/golden_record.py, ids masked. With it on, those same files are still
byte-identical and exactly one file is added.

SYNTHETIC, NOT PHYSICAL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests import synthetic_scene as ss
from tower.world_builder import frame_quality as FQ
from tower.world_builder.engine import WorldBuilderEngine
from tower.world_builder.keyframes import KeyframePolicy, KeyframeSelector
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.redaction import RedactionResult
from tower.world_builder.schema import END_REASON_ERROR
from tower.world_builder.store import WorldStore

ENV = "TOWER_WORLD_FRAME_QUALITY_LOG"
GOLDEN = Path(__file__).parent / "golden" / "world_builder_frame_quality_ceb203a.json"
LOG_NAME = "sessions/ID/frames_quality.jsonl"

# -- KEEP IN STEP with RUN/experiments/P5-SHARP/golden_record.py: the recorded walk ------------

WIDTH, HEIGHT = 480, 360
_APPLIED = "faces-detected-and-filled/yunet-2023mar@0.30+plausibility1"


class NoFaceRedactor:
    available = True
    unavailable_reason = None
    label = _APPLIED

    def redact(self, data):
        return RedactionResult(image_bytes=data, label=_APPLIED, regions=0)


class Clock:
    def __init__(self):
        self.t = 5000.0

    def __call__(self):
        self.t += 0.125
        return self.t


def golden_frames():
    K = ss.camera_matrix(WIDTH, HEIGHT)
    images = ss.render_sequence(ss.furnished_room(), ss.strafe(14, step=0.09), K, WIDTH, HEIGHT)
    room = [ss.encode_jpeg(i) for i in images]
    blurred = [ss.encode_jpeg(ss.blur(i, 15)) for i in images[5:8]]
    rng = np.random.default_rng(0)
    noise = [ss.encode_jpeg(rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)) for _ in range(3)]
    small = ss.encode_jpeg(cv2.resize(images[0], (WIDTH // 2, HEIGHT // 2)))
    return (room[:5] + [room[4], room[4]] + blurred + room[5:9] + [b"not a jpeg", small]
            + noise + room[9:] + room[::-1][:6])


def _intrinsics():
    K = ss.camera_matrix(WIDTH, HEIGHT)
    return CameraIntrinsics(source="self_calibrated", model="pinhole", fx=float(K[0, 0]),
                            fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                            calibrated_width=WIDTH, calibrated_height=HEIGHT)


def golden_walk(root, frames, *, stop=True, **engine_kwargs):
    """The walk, exactly as recorded. Returns (engine, world_id, session_id, results)."""
    engine = WorldBuilderEngine(WorldStore(root), clock=Clock(), redactor_factory=NoFaceRedactor,
                                **engine_kwargs)
    world_id = engine.create_world("golden")
    session_id = engine.start_session(world_id, intrinsics=_intrinsics(), frame_source="synthetic",
                                      declared_size=(WIDTH, HEIGHT))
    results = []
    for index, payload in enumerate(frames):
        results.append(engine.observe(payload, received_at=1000.0 + 0.25 * index,
                                      source_seq=10 + index, wire_seq=index))
    if stop:
        engine.stop_session()
    return engine, world_id, session_id, results


_ID = re.compile(r"[0-9a-f]{32}")


def snapshot(root, world_id):
    """Every file under the world directory, ids masked (golden_record.snapshot)."""
    world_dir = WorldStore(root).world_dir(world_id)
    files = {}
    for path in sorted(world_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = _ID.sub("ID", path.relative_to(world_dir).as_posix())
        data = path.read_bytes()
        if path.suffix in (".json", ".jsonl"):
            masked = _ID.sub("ID", data.decode("utf-8"))
            entry = {"sha256": hashlib.sha256(masked.encode("utf-8")).hexdigest(),
                     "bytes": len(data), "text": masked}
        else:
            entry = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        files[rel] = entry
    return files


# -- helpers -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frames():
    return golden_frames()


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _quiet_builder_env(monkeypatch):
    """The recorded walk ran with every TOWER_WORLD_* builder switch unset."""
    for name in (ENV, "TOWER_WORLD_RELOCALIZER", "TOWER_WORLD_RELOCALIZER_WINDOW",
                 "TOWER_WORLD_RELOCALIZER_HISTORY", "TOWER_WORLD_RELOCALIZER_SUMMARY"):
        monkeypatch.delenv(name, raising=False)


def _round_floats(obj, places=6):
    if isinstance(obj, float):
        return round(obj, places)
    if isinstance(obj, list):
        return [_round_floats(v, places) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, places) for k, v in obj.items()}
    return obj


def _assert_matches_golden(files, golden):
    """Every recorded file, byte for byte (same OpenCV/NumPy build), or float for float to 6
    places on another build, where JPEG bytes and the last digits of LK may move."""
    expected = golden["files"]
    assert sorted(set(expected) - set(files)) == [], "a file the product wrote is missing"
    exact = (golden["cv2"], golden["numpy"]) == (cv2.__version__, np.__version__)
    for name, entry in expected.items():
        if exact:
            assert files[name] == entry, name
        elif "text" in entry:
            now = [_round_floats(json.loads(line)) for line in files[name]["text"].splitlines()]
            then = [_round_floats(json.loads(line)) for line in entry["text"].splitlines()]
            assert now == then, name


def _log_lines(root, world_id, session_id):
    path = FQ.frames_quality_path(WorldStore(root), world_id, session_id)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _decisions(results):
    return [[r.outcome, r.reason, r.keyframe_id is not None] for r in results]


# -- the switch --------------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    (None, False), ("", False), ("   ", False),
    ("1", True), ("true", True), ("yes", True), ("on", True), (" ON ", True), ("True", True),
    ("0", False), ("false", False), ("no", False), ("off", False), (" OFF ", False),
])
def test_the_switch_reads_like_every_other_flag(monkeypatch, caplog, value, expected):
    from tower.config import world_frame_quality_log_setting

    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    with caplog.at_level(logging.WARNING, logger="tower.config"):
        assert world_frame_quality_log_setting() is expected
    assert not [r for r in caplog.records if ENV in r.getMessage()], "a valid value was logged"


@pytest.mark.parametrize("value", ["maybe", "2", "enabled", "onn", "-1"])
def test_garbage_is_off_and_logged(monkeypatch, caplog, value):
    from tower.config import world_frame_quality_log_setting

    monkeypatch.setenv(ENV, value)
    with caplog.at_level(logging.WARNING, logger="tower.config"):
        assert world_frame_quality_log_setting() is False
    logged = [r for r in caplog.records if ENV in r.getMessage()]
    assert len(logged) == 1 and logged[0].levelno == logging.WARNING
    assert repr(value.strip()) in logged[0].getMessage()


def test_the_default_is_off():
    from tower.config import WORLD_FRAME_QUALITY_LOG_ENV, world_frame_quality_log_setting

    assert WORLD_FRAME_QUALITY_LOG_ENV == ENV
    assert world_frame_quality_log_setting() is False


# -- off: the golden ---------------------------------------------------------------------------


def test_the_golden_exercises_what_it_claims(golden):
    reasons = {reason for _, reason, _ in golden["decisions"]}
    assert {"session_seed", "parallax", "insufficient_motion", "blurred", "tracking_lost",
            "malformed_frame", "frame_size_changed"} <= reasons
    assert golden["recorded_from"].startswith("ceb203a") and golden["product_tree_dirty"] == ""
    assert not any("frames_quality" in name for name in golden["files"])
    assert "sessions/ID/events.jsonl" in golden["files"]


@pytest.mark.parametrize("value", [None, "", "off", "0", "false", "maybe"])
def test_off_every_output_is_byte_identical_to_ceb203a(tmp_path, monkeypatch, frames, golden, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    _, world_id, _, results = golden_walk(tmp_path, frames)
    files = snapshot(tmp_path, world_id)
    assert sorted(files) == sorted(golden["files"]), "off wrote a file today's builder does not"
    _assert_matches_golden(files, golden)
    assert _decisions(results) == golden["decisions"]


def test_off_by_the_engine_argument_even_with_the_switch_on(tmp_path, monkeypatch, frames, golden):
    monkeypatch.setenv(ENV, "on")
    _, world_id, _, _ = golden_walk(tmp_path, frames, frame_quality_log=False)
    assert sorted(snapshot(tmp_path, world_id)) == sorted(golden["files"])


# -- on ----------------------------------------------------------------------------------------


def test_on_adds_one_file_and_changes_no_other_byte(tmp_path, monkeypatch, frames, golden):
    monkeypatch.setenv(ENV, "on")
    _, world_id, session_id, results = golden_walk(tmp_path, frames)
    files = snapshot(tmp_path, world_id)
    assert sorted(set(files) - set(golden["files"])) == [LOG_NAME]
    _assert_matches_golden(files, golden)  # the journal included: nothing per-frame lands in it
    assert _decisions(results) == golden["decisions"]
    session = WorldStore(tmp_path).read_session(world_id, session_id)
    lines = _log_lines(tmp_path, world_id, session_id)
    assert len(lines) == session.frames_observed == len(frames) == 30
    assert all(line["v"] == FQ.FRAME_QUALITY_VERSION for line in lines)
    assert "\r" not in files[LOG_NAME]["text"]


def test_on_by_the_engine_argument_with_the_switch_unset(tmp_path, frames):
    _, world_id, session_id, _ = golden_walk(tmp_path, frames, frame_quality_log=True)
    assert len(_log_lines(tmp_path, world_id, session_id)) == len(frames)


def test_every_value_is_the_selectors_own(tmp_path, monkeypatch, frames):
    """Line for line against what `evaluate` was handed and what it returned, exactly."""
    seen = []
    original = KeyframeSelector.evaluate

    def spy(self, quality, motion):
        decision = original(self, quality, motion)
        seen.append((quality, motion, decision, self.frames_since_keyframe))
        return decision

    monkeypatch.setattr(KeyframeSelector, "evaluate", spy)
    monkeypatch.setenv(ENV, "on")
    _, world_id, session_id, results = golden_walk(tmp_path, frames)
    lines = _log_lines(tmp_path, world_id, session_id)

    assert [line["source_seq"] for line in lines] == [10 + i for i in range(len(frames))]
    assert [line["received_at"] for line in lines] == [1000.0 + 0.25 * i for i in range(len(frames))]
    for line, result in zip(lines, results):
        assert (line["outcome"], line["reason"], line["keyframe_id"]) == (
            result.outcome, result.reason, result.keyframe_id)

    scored = [line for line in lines if line["tracker"] is not None]
    unscored = [line for line in lines if line["tracker"] is None]
    assert len(scored) == len(seen) == len(frames) - 2
    assert [u["reason"] for u in unscored] == ["malformed_frame", "frame_size_changed"]
    for u in unscored:
        assert all(u[k] is None for k in ("sharpness", "feature_count", "tracked_count",
                                          "survival_ratio", "overlap_ratio", "median_parallax_px",
                                          "homography_residual_px", "frames_since_keyframe",
                                          "segment_index", "keyframe_id"))

    for line, (quality, motion, decision, since) in zip(scored, seen):
        assert line["sharpness"] == quality.sharpness  # exactly: repr round-trips a float
        assert (line["outcome"], line["reason"]) == (decision.outcome, decision.reason)
        assert line["frames_since_keyframe"] == since
        if motion is None:
            assert line["tracker"] == FQ.TRACKER_NO_REFERENCE
            assert line["survival_ratio"] is None and line["median_parallax_px"] is None
        else:
            assert line["tracker"] == FQ.TRACKER_REFERENCE
            assert line["feature_count"] == motion.seeded_count
            assert line["tracked_count"] == motion.tracked_count
            assert line["survival_ratio"] == motion.survival_ratio
            assert line["overlap_ratio"] == motion.overlap_ratio
            assert line["median_parallax_px"] == motion.median_displacement_px
            assert line["homography_residual_px"] == motion.homography_residual_px
    assert {line["tracker"] for line in scored} == {FQ.TRACKER_REFERENCE, FQ.TRACKER_NO_REFERENCE}

    # And against the persisted keyframe record: the same numbers, the same segment.
    keyframes = WorldStore(tmp_path).read_keyframes(world_id, session_id)
    by_id = {line["keyframe_id"]: line for line in lines if line["keyframe_id"]}
    assert sorted(by_id) == sorted(k.keyframe_id for k in keyframes)
    for kf in keyframes:
        line = by_id[kf.keyframe_id]
        assert line["source_seq"] == kf.source_seq and line["received_at"] == kf.received_at
        assert line["sharpness"] == kf.sharpness
        assert line["median_parallax_px"] == kf.median_parallax_px
        assert line["survival_ratio"] == kf.survival_ratio
        assert line["overlap_ratio"] == kf.overlap_ratio
        assert line["segment_index"] == kf.segment_index
        assert line["reason"] == kf.selection_reason and line["outcome"] == "accept"


def test_the_blur_verdict_is_reproducible_from_the_file_alone(tmp_path, monkeypatch, frames):
    """The ratio test's rolling median is rebuilt from the lines, in order: see frame_quality.py."""
    monkeypatch.setenv(ENV, "on")
    _, world_id, session_id, _ = golden_walk(tmp_path, frames)
    policy = KeyframePolicy()
    window, blurred = [], 0
    for line in _log_lines(tmp_path, world_id, session_id):
        if line["sharpness"] is None:
            continue
        window = (window + [line["sharpness"]])[-policy.sharpness_window:]
        verdict = line["sharpness"] < policy.min_sharpness
        if not verdict and len(window) >= 5:
            reference = sorted(window)[len(window) // 2]
            verdict = reference > 0 and line["sharpness"] / reference < policy.min_sharpness_ratio
        assert verdict == (line["reason"] == "blurred"), line["source_seq"]
        blurred += verdict
    assert blurred == 3


def test_flushed_and_closed_on_stop(tmp_path, monkeypatch, frames):
    monkeypatch.setenv(ENV, "on")
    engine, world_id, session_id, _ = golden_walk(tmp_path, frames, stop=False)
    log = engine._frame_log
    path = FQ.frames_quality_path(WorldStore(tmp_path), world_id, session_id)
    assert log is not None and log.path == path and not log.closed
    assert log.lines == len(frames)
    # Buffered: 30 lines are more than one buffer, so SOME is on disk and not all of it.
    assert 0 < len(path.read_bytes()) and path.read_bytes().count(b"\n") < len(frames)
    engine.stop_session()
    assert log.closed and not log.failed and engine._frame_log is None
    assert path.read_bytes().count(b"\n") == len(frames)
    # No handle is left: Windows refuses to rename a file a process holds open.
    path.rename(path.with_name("moved.jsonl"))


def test_closed_when_the_walk_ends_in_error(tmp_path, monkeypatch, frames):
    """world_build_session.py's unwind: stop_session(error, hold_lock=True), then release."""
    monkeypatch.setenv(ENV, "on")
    engine, world_id, session_id, _ = golden_walk(tmp_path, frames[:9], stop=False)
    log = engine._frame_log
    engine.stop_session(END_REASON_ERROR, hold_lock=True)
    assert log.closed and engine._frame_log is None
    engine.release_world(world_id)
    assert len(_log_lines(tmp_path, world_id, session_id)) == 9


def test_one_handle_per_session(tmp_path, monkeypatch, frames):
    monkeypatch.setenv(ENV, "on")
    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store, clock=Clock(), redactor_factory=NoFaceRedactor)
    world_id = engine.create_world("two")
    sessions, logs = [], []
    for n in (4, 6):
        sid = engine.start_session(world_id, intrinsics=_intrinsics(), frame_source="synthetic",
                                   declared_size=(WIDTH, HEIGHT))
        logs.append(engine._frame_log)
        for i, payload in enumerate(frames[:n]):
            engine.observe(payload, received_at=float(i), source_seq=i)
        engine.stop_session()
        sessions.append(sid)
    assert logs[0] is not logs[1] and all(log.closed for log in logs)
    assert [len(_log_lines(tmp_path, world_id, sid)) for sid in sessions] == [4, 6]


def test_a_session_left_open_is_closed_by_the_next_start(tmp_path, monkeypatch, frames):
    monkeypatch.setenv(ENV, "on")
    engine, world_id, _, _ = golden_walk(tmp_path, frames[:3], stop=False)
    stale = engine._frame_log
    engine._session = None  # abandoned without stop_session
    engine._store.release_writer_lock(world_id)
    engine.start_session(world_id, intrinsics=_intrinsics(), frame_source="synthetic")
    assert stale.closed and engine._frame_log is not stale
    engine.stop_session()


# -- an I/O error never stops the build --------------------------------------------------------


class _FailingHandle:
    """A text handle that takes `good` lines and then fails the way a full disk does."""

    def __init__(self, good, fail_on="write"):
        self.good, self.fail_on, self.writes, self.closed = good, fail_on, 0, False

    def write(self, text):
        if self.fail_on == "write" and self.writes >= self.good:
            raise OSError(28, "No space left on device")
        self.writes += 1
        return len(text)

    def close(self):
        self.closed = True
        if self.fail_on == "close":
            raise OSError(28, "No space left on device")


def _warnings(caplog):
    return [r for r in caplog.records
            if r.name == FQ.__name__ and "frame quality log" in r.getMessage()]


@pytest.mark.parametrize("fail_on", ["open", "write", "close"])
def test_an_io_error_is_logged_once_and_the_walk_is_todays(
        tmp_path, monkeypatch, caplog, frames, golden, fail_on):
    handles = []

    def failing_open(*args, **kwargs):
        if fail_on == "open":
            raise PermissionError(13, "Access is denied")
        handles.append(_FailingHandle(good=5, fail_on=fail_on))
        return handles[-1]

    monkeypatch.setattr(FQ, "open", failing_open, raising=False)
    monkeypatch.setenv(ENV, "on")
    with caplog.at_level(logging.WARNING, logger=FQ.__name__):
        engine, world_id, session_id, results = golden_walk(tmp_path, frames, stop=False)
        log = engine._frame_log
        summary = engine.stop_session()

    assert _decisions(results) == golden["decisions"]
    assert summary.frames_observed == len(frames)
    files = snapshot(tmp_path, world_id)
    assert sorted(files) == sorted(golden["files"])  # the fake handle wrote nothing real
    _assert_matches_golden(files, golden)
    assert log.failed and log.closed
    assert len(_warnings(caplog)) == 1, [r.getMessage() for r in _warnings(caplog)]
    if fail_on == "write":
        assert handles[0].writes == 5 and handles[0].closed and log.lines == 5
    if fail_on == "close":
        assert handles[0].writes == len(frames)


def test_a_log_that_raises_anything_costs_no_frame(tmp_path, monkeypatch, caplog, frames, golden):
    """Not only OSError: an unserialisable row, a closed file -- all of it is the log's problem."""
    monkeypatch.setenv(ENV, "on")
    monkeypatch.setattr(FQ, "frame_row", lambda **_: {"unserialisable": object()})
    with caplog.at_level(logging.WARNING, logger=FQ.__name__):
        _, world_id, session_id, results = golden_walk(tmp_path, frames)
    assert _decisions(results) == golden["decisions"]
    assert len(_warnings(caplog)) == 1
    files = snapshot(tmp_path, world_id)
    _assert_matches_golden(files, golden)
    assert files[LOG_NAME]["bytes"] == 0  # opened, never written, closed


def _raise(*_, **__):
    raise AttributeError("'MotionSummary' object has no attribute 'seeded_count'")


@pytest.mark.parametrize("where", ["frame_row", "the selector's counter"])
def test_building_a_line_that_raises_never_reaches_observe(
        tmp_path, monkeypatch, caplog, frames, golden, where):
    """Review V16 MED-1: an exception while BUILDING a line -- a future attribute rename in
    `frame_row`, or on the selector the engine reads -- is the log's failure, logged once;
    `observe()` never sees it and the walk's outputs are the log-off outputs exactly."""
    monkeypatch.setenv(ENV, "on")
    if where == "frame_row":
        monkeypatch.setattr(FQ, "frame_row", _raise)
    else:
        monkeypatch.setattr(KeyframeSelector, "frames_since_keyframe", property(_raise))
    with caplog.at_level(logging.WARNING, logger=FQ.__name__):
        engine, world_id, _, results = golden_walk(tmp_path, frames, stop=False)
        log = engine._frame_log
        summary = engine.stop_session()
    assert _decisions(results) == golden["decisions"]
    assert summary.frames_observed == len(frames) and summary.keyframes_accepted == 7
    files = snapshot(tmp_path, world_id)
    assert sorted(set(files) - set(golden["files"])) == [LOG_NAME]
    _assert_matches_golden(files, golden)
    assert files[LOG_NAME]["bytes"] == 0
    assert log.failed and log.closed and not log.active and log.lines == 0
    warnings = _warnings(caplog)
    assert len(warnings) == 1 and "AttributeError" in warnings[0].getMessage()


# -- readers, re-opens, and the values that are easy to get subtly wrong ------------------------


def _row(seq):
    return FQ.frame_row(source_seq=seq, received_at=1758790123.4567891 + seq,
                        outcome="skip", reason="insufficient_motion")


def test_the_reader_skips_a_torn_last_line(tmp_path):
    """Review V16 LOW-1: a builder killed mid-buffer leaves the last line cut at a byte."""
    path = tmp_path / FQ.FRAMES_QUALITY_FILENAME
    whole = "".join(json.dumps(_row(s), separators=(",", ":")) + "\n" for s in (1, 2, 3))
    torn = json.dumps(_row(4), separators=(",", ":"))[:57]
    path.write_bytes((whole + torn).encode("utf-8"))
    assert [r["source_seq"] for r in FQ.read_frames_quality(path)] == [1, 2, 3]
    # a complete last line that merely lacks its newline is a line, not a tear
    path.write_bytes(whole.encode("utf-8").rstrip(b"\n"))
    assert [r["source_seq"] for r in FQ.read_frames_quality(path)] == [1, 2, 3]
    # absent: every world before this, and every world written with the switch off
    assert FQ.read_frames_quality(tmp_path / "absent.jsonl") == []


def test_a_reopen_appends_and_ends_a_torn_line_first(tmp_path):
    """Review V16 LOW-2 ("a", never "w") and LOW-1 (a tear never fuses with the next record)."""
    path = tmp_path / FQ.FRAMES_QUALITY_FILENAME
    first = "".join(json.dumps(_row(s), separators=(",", ":")) + "\n" for s in (1, 2))
    torn = json.dumps(_row(3), separators=(",", ":"))[:40]
    path.write_bytes((first + torn).encode("utf-8"))
    log = FQ.FrameQualityLog(path)
    log.record(source_seq=9, received_at=2.5, outcome="skip", reason="insufficient_motion")
    log.close()
    data = path.read_bytes().decode("utf-8")
    assert data.startswith(first + torn + "\n"), "a re-open destroyed or fused what was there"
    assert [r["source_seq"] for r in FQ.read_frames_quality(path)] == [1, 2, 9]
    # and a clean file is appended to as it is, with no blank line
    log = FQ.FrameQualityLog(path)
    log.record(source_seq=10, received_at=3.5, outcome="skip", reason="insufficient_motion")
    log.close()
    assert [r["source_seq"] for r in FQ.read_frames_quality(path)] == [1, 2, 9, 10]
    assert "\n\n" not in path.read_bytes().decode("utf-8")


def test_segment_index_is_the_segment_each_frame_was_measured_in(tmp_path, monkeypatch, frames):
    """Review V16 LOW-2: the segment of EVERY scored line, not only the keyframes', rebuilt
    independently from the journal: a loss line is still in the old segment, and the next
    frame is in the new one; a solve chain break moves it on after its keyframe."""
    monkeypatch.setenv(ENV, "on")
    _, world_id, session_id, _ = golden_walk(tmp_path, frames)
    store = WorldStore(tmp_path)
    events = store.read_events(world_id, session_id)
    breakers = {
        events[i + 1]["payload"]["keyframe_id"] for i, e in enumerate(events[:-1])
        if e["kind"] == "solve_chain_broken"
    }
    losses = [e["payload"]["segment_index"] for e in events if e["kind"] == "tracking_lost"]
    expected, seen_later = 0, []
    for line in _log_lines(tmp_path, world_id, session_id):
        if line["tracker"] is None:
            assert line["segment_index"] is None
            continue
        assert line["segment_index"] == expected, line["source_seq"]
        if line["keyframe_id"] is None and expected > 0:
            seen_later.append(line["source_seq"])
        if line["outcome"] == "tracking_lost":
            expected += 1
            assert losses.pop(0) == expected  # the journal names the NEW segment
        elif line["keyframe_id"] in breakers:
            expected += 1
    assert losses == [] and expected == 2
    assert seen_later, "no non-keyframe line after a loss: the walk would not catch an off-by-one"


def test_received_at_and_source_seq_are_kept_exactly(tmp_path, monkeypatch, frames):
    """Review V16 LOW-2: non-round times, so rounding (to 3 places, or to float32) is caught."""
    monkeypatch.setenv(ENV, "on")
    engine = WorldBuilderEngine(WorldStore(tmp_path), clock=Clock(), redactor_factory=NoFaceRedactor)
    world_id = engine.create_world("times")
    session_id = engine.start_session(world_id, intrinsics=_intrinsics(), frame_source="synthetic")
    times = [1758790123.4567891 + i * 0.0833337 + (i % 3) * 1.1e-5 for i in range(12)]
    seqs = [7 + 3 * i for i in range(12)]
    for payload, t, seq in zip(frames[:12], times, seqs):
        engine.observe(payload, received_at=t, source_seq=seq)
    engine.stop_session()
    lines = _log_lines(tmp_path, world_id, session_id)
    assert [line["received_at"] for line in lines] == times
    assert [round(t, 3) for t in times] != times  # the check has teeth
    assert [line["source_seq"] for line in lines] == seqs
    keyframes = WorldStore(tmp_path).read_keyframes(world_id, session_id)
    by_seq = {line["source_seq"]: line for line in lines}
    assert keyframes and all(by_seq[k.source_seq]["received_at"] == k.received_at for k in keyframes)


# -- the saved world ---------------------------------------------------------------------------


def test_the_file_is_inert_for_build_and_purged_with_the_world(tmp_path, monkeypatch, frames):
    monkeypatch.setenv(ENV, "on")
    engine, world_id, session_id, _ = golden_walk(tmp_path, frames)
    store = WorldStore(tmp_path)
    path = FQ.frames_quality_path(store, world_id, session_id)
    before = path.read_bytes()
    result = engine.build(world_id, session_id)
    assert result.keyframes == 7
    assert path.read_bytes() == before  # a build neither reads it into anything nor rewrites it
    assert store.world_bytes(world_id)["journals"] >= len(before)
    report = store.purge_world(world_id)
    assert report.retained == () and not path.exists()


def test_frame_row_for_an_unscored_frame_is_all_null():
    row = FQ.frame_row(source_seq=3, received_at=1.5, outcome="reject", reason="malformed_frame")
    assert list(row) == ["v", "source_seq", "received_at", "sharpness", "tracker",
                         "feature_count", "tracked_count", "survival_ratio", "overlap_ratio",
                         "median_parallax_px", "homography_residual_px", "frames_since_keyframe",
                         "segment_index", "outcome", "reason", "keyframe_id"]
    assert row["source_seq"] == 3 and row["outcome"] == "reject"
    assert all(row[k] is None for k in row if k not in ("v", "source_seq", "received_at",
                                                         "outcome", "reason"))
