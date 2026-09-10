"""A WiFi hiccup mid-walk must produce ONE continuous world.

`handoff.md` 9.3 states the shape plainly, and states it as the EXPECTED
case rather than an edge case:

    stream_start -> frames -> socket dies -> new socket -> ping/pong ->
    stream_start AGAIN -> frames CONTINUING FROM THE PREVIOUS seq

with no `stream_stop` in between, because a dropped socket produces none.

Before this work the consequence was measured and severe: the follower saw
its capture close, ended the mapping session, and the rest of the walk sat
in a second directory that nothing read. The world ended at the hiccup.

Two sessions would not have been an acceptable fix either. `build()` is
per session, and the result channel reports the newest session's keyframe
count -- so on the phone, which replaces `WorldSnapshot` wholesale and has
no merge layer, the count would visibly reset to zero at the moment of the
hiccup.
"""

import base64
import json
import pathlib

import numpy as np
import pytest

from tower.capture import (
    CAPTURE_FILENAME,
    END_REASON_DISCONNECT,
    END_REASON_STOP,
    CaptureFollower,
    CaptureRecorder,
)


def _jpeg(value: int = 128) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", np.full((64, 64, 3), value, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def _frame_payload() -> str:
    return base64.b64encode(_jpeg()).decode("ascii")


def _frame(seq: int, data: str) -> dict:
    return {
        "type": "frame",
        "seq": seq,
        "width": 64,
        "height": 64,
        "format": "jpeg",
        "data": data,
    }


def _manifest_of(recorder, capture_id) -> dict:
    path = recorder.capture_dir(capture_id) / CAPTURE_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


# -- the recorder side --------------------------------------------------


def test_a_disconnect_leaves_a_capture_that_a_successor_can_continue(tmp_path):
    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(), source_seq=1)
    recorder.stop(END_REASON_DISCONNECT)

    assert recorder.resumable_capture() == first

    second = recorder.start(owner=object(), continues=recorder.resumable_capture())
    assert _manifest_of(recorder, second)["continues_capture"] == first


def test_a_polite_stop_is_not_resumable(tmp_path):
    """A stream_stop ends the walk deliberately. Only a drop leaves one open."""
    recorder = CaptureRecorder(tmp_path)
    recorder.start(owner=object())
    recorder.stop(END_REASON_STOP)
    assert recorder.resumable_capture() is None


def test_a_reconnect_after_the_grace_window_is_a_new_walk(tmp_path):
    """Past the point where iOS has stopped retrying, it is a different walk.

    handoff.md 6.4: iOS gives up after five attempts, roughly 45 s.
    """
    now = [1000.0]
    recorder = CaptureRecorder(tmp_path, clock=lambda: now[0])
    recorder.start(owner=object())
    recorder.stop(END_REASON_DISCONNECT)

    now[0] += 89.0
    assert recorder.resumable_capture() is not None
    now[0] += 2.0
    assert recorder.resumable_capture() is None


def test_a_new_recording_clears_the_resumable_capture(tmp_path):
    """A successor must not itself be offered as a predecessor twice."""
    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.stop(END_REASON_DISCONNECT)
    recorder.start(owner=object(), continues=first)
    assert recorder.resumable_capture() is None


# -- the follower side --------------------------------------------------


def test_the_follower_continues_into_a_successor(tmp_path):
    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(10), source_seq=1)
    recorder.write_frame(_jpeg(20), source_seq=2)
    recorder.stop(END_REASON_DISCONNECT)

    second = recorder.start(owner=object(), continues=first)
    recorder.write_frame(_jpeg(30), source_seq=3)
    recorder.write_frame(_jpeg(40), source_seq=4)
    recorder.stop(END_REASON_STOP)

    follower = CaptureFollower(
        recorder.capture_dir(first), poll_seconds=0.001, sleep=lambda _s: None
    )
    seqs = [frame.source_seq for frame in follower.follow(max_idle_polls=3)]

    assert seqs == [1, 2, 3, 4], "the walk was cut at the reconnect"
    assert follower.directory.name == second


def test_the_follower_does_not_wait_after_a_polite_stop(tmp_path):
    """Waiting on every ordinary session would add a fixed delay for nothing."""
    slept = []
    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(), source_seq=1)
    recorder.stop(END_REASON_STOP)

    follower = CaptureFollower(
        recorder.capture_dir(first),
        poll_seconds=0.001,
        sleep=lambda s: slept.append(s),
    )
    seqs = [frame.source_seq for frame in follower.follow(max_idle_polls=2)]

    assert seqs == [1]
    assert slept == [], "a cleanly stopped capture must end immediately"


def test_the_follower_gives_up_when_no_successor_arrives(tmp_path):
    """A phone that never comes back must not hang the follower forever."""
    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(), source_seq=1)
    recorder.stop(END_REASON_DISCONNECT)

    follower = CaptureFollower(
        recorder.capture_dir(first),
        poll_seconds=0.001,
        sleep=lambda _s: None,
        resume_grace_seconds=0.01,
    )
    seqs = [frame.source_seq for frame in follower.follow(max_idle_polls=3)]
    assert seqs == [1]


def test_the_follower_ignores_an_unrelated_capture(tmp_path):
    """Only a capture that NAMES this one continues it."""
    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(), source_seq=1)
    recorder.stop(END_REASON_DISCONNECT)

    # A different walk entirely, linking nothing.
    recorder.start(owner=object())
    recorder.write_frame(_jpeg(), source_seq=99)
    recorder.stop(END_REASON_STOP)

    follower = CaptureFollower(
        recorder.capture_dir(first),
        poll_seconds=0.001,
        sleep=lambda _s: None,
        resume_grace_seconds=0.01,
    )
    seqs = [frame.source_seq for frame in follower.follow(max_idle_polls=3)]
    assert seqs == [1], "an unrelated capture was mistaken for a continuation"


def test_following_reconnects_can_be_switched_off(tmp_path):
    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(), source_seq=1)
    recorder.stop(END_REASON_DISCONNECT)
    recorder.start(owner=object(), continues=first)
    recorder.write_frame(_jpeg(), source_seq=2)
    recorder.stop(END_REASON_STOP)

    follower = CaptureFollower(
        recorder.capture_dir(first),
        poll_seconds=0.001,
        sleep=lambda _s: None,
        follow_reconnects=False,
    )
    seqs = [frame.source_seq for frame in follower.follow(max_idle_polls=3)]
    assert seqs == [1]


# -- over the real wire -------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from tower.main import create_app

    monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(tmp_path / "capture"))
    return TestClient(create_app())


def test_a_reconnect_over_the_wire_produces_one_continuous_frame_stream(client):
    """The whole hiccup, driven through the real app.

    Note what iOS does NOT do here: it sends no stream_stop (the socket
    died), and seq CONTINUES rather than restarting -- handoff.md 9.3.
    """
    data = _frame_payload()
    recorder = client.app.state.frame_observers[0]

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for seq in (1, 3, 5):
            ws.send_json(_frame(seq, data))
            ws.receive_json()
        first = recorder.status.capture_id

    # The socket dropped. No stream_stop was sent.
    assert recorder.status.end_reason == END_REASON_DISCONNECT

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for seq in (7, 9, 11):
            ws.send_json(_frame(seq, data))
            ws.receive_json()
        second = recorder.status.capture_id
        ws.send_json({"type": "stream_stop"})

    assert second != first
    assert _manifest_of(recorder, second)["continues_capture"] == first

    follower = CaptureFollower(
        recorder.capture_dir(first), poll_seconds=0.001, sleep=lambda _s: None
    )
    seqs = [frame.wire_seq for frame in follower.follow(max_idle_polls=3)]
    assert seqs == [1, 3, 5, 7, 9, 11], (
        "the walk was not continuous across the reconnect"
    )


# -- the journal tail ---------------------------------------------------


def test_the_journal_tail_reads_only_what_is_new(tmp_path):
    """Poll cost must not grow with how long the walk has been going.

    Re-reading the whole journal every poll is O(n) per poll and O(n^2)
    over a capture -- measured at 36.9 ms for one poll against a
    20,000-line journal, in the same process that has to run observe() on
    every frame. Seeking to a remembered offset makes it 0.014 ms and flat.
    """
    from tower.capture import _JournalTail

    journal = tmp_path / "frames.jsonl"
    journal.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")

    tail = _JournalTail(journal)
    assert tail.read_new() == [{"a": 1}, {"a": 2}]
    assert tail.read_new() == [], "an unchanged journal must yield nothing"

    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"a": 3}\n')
    assert tail.read_new() == [{"a": 3}]


def test_a_partial_trailing_line_is_completed_not_dropped(tmp_path):
    """The recorder appends without fsync, so arriving mid-write is normal."""
    from tower.capture import _JournalTail

    journal = tmp_path / "frames.jsonl"
    journal.write_text('{"a": 1}\n{"a": 2', encoding="utf-8")

    tail = _JournalTail(journal)
    assert tail.read_new() == [{"a": 1}], "a torn line must not be yielded"

    with journal.open("a", encoding="utf-8") as handle:
        handle.write('}\n')
    assert tail.read_new() == [{"a": 2}], "the completed line was lost"


def test_a_replaced_journal_is_reread_from_the_start(tmp_path):
    """A shrunken append-only file means it was replaced, not truncated."""
    from tower.capture import _JournalTail

    journal = tmp_path / "frames.jsonl"
    journal.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
    tail = _JournalTail(journal)
    tail.read_new()

    journal.write_text('{"b": 1}\n', encoding="utf-8")
    assert tail.read_new() == [{"b": 1}]


def test_a_corrupt_complete_line_is_skipped(tmp_path):
    from tower.capture import _JournalTail

    journal = tmp_path / "frames.jsonl"
    journal.write_text('{"a": 1}\nnot json at all\n{"a": 2}\n', encoding="utf-8")
    assert _JournalTail(journal).read_new() == [{"a": 1}, {"a": 2}]


def test_an_absent_journal_yields_nothing(tmp_path):
    from tower.capture import _JournalTail

    assert _JournalTail(tmp_path / "missing.jsonl").read_new() == []


# -- a stop that arrives while a reconnect is in flight ------------------


def test_a_stop_during_the_reconnect_wait_is_noticed_immediately(tmp_path):
    """Ninety seconds is a long time to be deaf.

    `_await_successor` polled the whole grace window without ever asking
    `should_stop`, and `routes/ws.py` stops every cartridge session when
    the last connection goes -- which a slow reconnect is exactly what
    gets you. A reviewer measured this Tower's own 52 reconnect chains and
    found three at 31-35 s, against a socket the server notices as dead in
    20-40 s.
    """
    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(10), source_seq=1)
    recorder.stop(END_REASON_DISCONNECT)
    # No successor exists yet, so the follower would wait out the window.

    slept = []
    follower = CaptureFollower(
        recorder.capture_dir(first),
        poll_seconds=0.001,
        sleep=lambda s: slept.append(s),
        resume_grace_seconds=90.0,
    )
    seqs = list(follower.follow(should_stop=lambda: True))

    assert seqs == [], "a stop before the first poll must not pull in a frame"
    assert slept == [], (
        f"the follower slept {len(slept)} times after being told to stop"
    )
    assert follower.stopped_awaiting_successor() is False, (
        "nothing was awaited: the stop arrived before the journal was read"
    )


def test_a_stop_mid_reconnect_does_not_bind_to_a_capture_it_never_reads(tmp_path):
    """The truncation, not the delay -- and it is the worse half.

    The loop ran to the end of the grace window, found the successor,
    rebound onto it, and `continue`d straight into the `should_stop` check
    at the top of `follow`. It returned having read ZERO frames of the
    capture it had just bound to, and the rebind moved `end_reason()` onto
    a capture that was still open -- so the walk was silently truncated at
    the reconnect and nothing downstream could say so.
    """
    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(10), source_seq=1)
    recorder.stop(END_REASON_DISCONNECT)
    second = recorder.start(owner=object(), continues=first)
    recorder.write_frame(_jpeg(200), source_seq=2)

    # Stop on the SECOND ask: the first is the one at the top of `follow`,
    # before the journal is read, so the walk's own frame still lands.
    asks = {"n": 0}

    def should_stop():
        asks["n"] += 1
        return asks["n"] > 1

    follower = CaptureFollower(
        recorder.capture_dir(first), poll_seconds=0.001, sleep=lambda _s: None,
    )
    seqs = [frame.source_seq for frame in follower.follow(should_stop=should_stop)]

    assert seqs == [1], "the predecessor's frames were not read"
    assert follower.directory.name == first, (
        "the follower bound to a successor it never read a frame of; "
        f"end_reason() now describes {follower.directory.name}"
    )
    assert follower.stopped_awaiting_successor() is True, (
        "nothing recorded that a reconnect was in flight when the stop "
        "landed, so a caller cannot tell this walk from a finished one"
    )
    assert follower.end_reason() == END_REASON_DISCONNECT


def test_a_permanent_disconnect_is_still_an_ordinary_end(tmp_path):
    """The flag means "a reconnect was in flight", not "a stop arrived".

    The first version set it on ANY stop landing inside the 90 s grace
    window -- which is also the ordinary shape of a phone that disconnects
    for good: the socket dies, `routes/ws.py` stops the session 20-40 s
    later, comfortably inside the grace. So it quietly reversed the policy
    in `world_build_session.py` that a `disconnect` capture counts as
    finished, and a permanent disconnect started reporting `interrupted`
    -- this campaign's headline symptom, back through the door it was
    pushed out of. Measured by a reviewer running the CLI, not inferred.
    """
    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(10), source_seq=1)
    recorder.stop(END_REASON_DISCONNECT)
    # No successor, and none is coming.

    asks = {"n": 0}

    def should_stop():
        asks["n"] += 1
        return asks["n"] > 1

    follower = CaptureFollower(
        recorder.capture_dir(first), poll_seconds=0.001, sleep=lambda _s: None,
    )
    seqs = [frame.source_seq for frame in follower.follow(should_stop=should_stop)]

    assert seqs == [1]
    assert follower.stopped_awaiting_successor() is False, (
        "a phone that never came back was reported as an interrupted walk"
    )


def test_finding_a_successor_does_not_read_every_capture_on_the_disk(tmp_path):
    """The grace window has to be bounded by the grace window.

    `_find_successor` opened and parsed every `capture.json` under the
    captures root, in `iterdir` order, with no early exit -- and it runs
    once per poll for up to 360 polls. A reviewer measured **11 ms per scan
    against 104 real captures**, so a "ninety second" wait actually ran
    94 s, an overrun that grows with a directory that only ever grows.

    Newest first, and nothing older than the capture being followed: a
    successor is created after its predecessor ends, so an older directory
    cannot be one. The scan now stops at the first match instead of
    reading past it.
    """
    import tower.capture as capture_module

    recorder = CaptureRecorder(tmp_path)
    # Twenty finished, unrelated captures, all older than the walk.
    for _ in range(20):
        stale = recorder.start(owner=object())
        recorder.write_frame(_jpeg(1), source_seq=1)
        recorder.stop(END_REASON_STOP)

    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(10), source_seq=1)
    recorder.stop(END_REASON_DISCONNECT)
    second = recorder.start(owner=object(), continues=first)
    recorder.write_frame(_jpeg(200), source_seq=2)
    recorder.stop(END_REASON_STOP)

    opened = []
    real_read = capture_module.read_json_closed

    def counting_read(path):
        opened.append(pathlib.Path(path).parent.name)
        return real_read(path)

    capture_module.read_json_closed = counting_read
    try:
        follower = CaptureFollower(
            recorder.capture_dir(first), poll_seconds=0.001, sleep=lambda _s: None,
        )
        found = follower._find_successor()
    finally:
        capture_module.read_json_closed = real_read

    assert found is not None and found.name == second
    assert opened == [second], (
        f"the scan opened {len(opened)} manifests to find a successor that is "
        f"the newest directory there is: {opened}"
    )


def test_a_successor_created_before_its_predecessor_closed_is_still_found(tmp_path):
    """The prune must not be able to lose a successor permanently.

    `_find_successor` skips directories older than the capture it follows.
    A reviewer measured what that cuts: `write_json_atomic` on
    `capture.json` bumps the parent directory's mtime, so the predecessor's
    `stop()` moves it FORWARD -- and `CaptureRecorder.stop`'s own docstring
    records the ordering where the new connection arms a recording before
    the old connection's `finally` stops the previous one. The successor is
    then older than its predecessor, pruned by fractions of a second, and
    missed on **0 of 360 polls** -- both mtimes are frozen for the whole
    window, so a prune that misses once misses every time.

    The compound cost is the part that matters: the follower reports no
    reconnect, the walk finalises as an ordinary end, and the successor's
    frames are silently dropped from a world reported Saved.
    """
    import os

    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(10), source_seq=1)
    recorder.stop(END_REASON_DISCONNECT)
    second = recorder.start(owner=object(), continues=first)
    recorder.write_frame(_jpeg(200), source_seq=2)
    recorder.stop(END_REASON_STOP)

    # The inverted ordering, forced: the successor's directory predates the
    # predecessor's final manifest write.
    predecessor = recorder.capture_dir(first)
    successor = recorder.capture_dir(second)
    stamp = predecessor.stat().st_mtime
    os.utime(successor, (stamp - 0.5, stamp - 0.5))
    assert successor.stat().st_mtime < predecessor.stat().st_mtime

    follower = CaptureFollower(
        predecessor, poll_seconds=0.001, sleep=lambda _s: None,
    )
    assert follower._find_successor() is not None, (
        "a successor 0.5 s older than its own predecessor was pruned away"
    )

    seqs = [frame.source_seq for frame in follower.follow(max_idle_polls=3)]
    assert seqs == [1, 2], f"the walk was cut at the reconnect: {seqs}"
