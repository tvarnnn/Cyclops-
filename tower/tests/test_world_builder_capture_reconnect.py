"""After a reconnect, a keyframe's raw source must be the frame it came from.

`_follow_capture` builds each `ObservedFrame.source_path` from a directory,
and `relpath` is relative to the capture the frame CAME FROM. A reconnect
retargets the follower onto a successor -- `test_capture_continuity.py`
proves the frames themselves continue -- and this line was reading the
directory the generator was CALLED with, two captures stale by then.

Measured on the 2026-09-09 field walk, which reconnected once:
`solve/<session>/sources.json` named capture `6a1b544c` for all 643
keyframes, while 523 of them were actually in `dd885cca`. Every one of
those 523 resolved to a path that does not exist, so `_source_frame` fell
back to the session's face-redacted copies and COLMAP was fed those for
81% of the walk. The handoff had recorded this as "sources.json is already
523/643 stale". Nothing was stale; the ledger was wrong when written.

It could have been worse. The phone's source index ran 1..953 in the first
capture and 1309..6109 in the second, so no wrong path happened to exist.
Had the counter restarted at 1 -- which nothing guarantees -- the same bug
would have handed COLMAP a DIFFERENT REAL PHOTOGRAPH under the right name.
So this test asserts on the BYTES, not on the path: a wrong image that
exists must fail it just as loudly as a missing one.
"""

import numpy as np

from tower.capture import END_REASON_DISCONNECT, END_REASON_STOP, CaptureRecorder


def _jpeg(value: int) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", np.full((64, 64, 3), value, dtype=np.uint8))
    assert ok
    return bytes(buffer)


def _walk_across_a_reconnect(tmp_path, *, second_capture_restarts_numbering: bool):
    """Two captures, the second continuing the first, distinct pixels each."""
    recorder = CaptureRecorder(tmp_path)
    first = recorder.start(owner=object())
    recorder.write_frame(_jpeg(10), source_seq=1)
    recorder.write_frame(_jpeg(20), source_seq=2)
    recorder.stop(END_REASON_DISCONNECT)

    second = recorder.start(owner=object(), continues=first)
    # The phone's counter either continues or restarts. It continued on the
    # field walk; nothing makes that a guarantee, and the restarting case is
    # the one where a stale directory resolves to a real, wrong photograph.
    base = 1 if second_capture_restarts_numbering else 3
    recorder.write_frame(_jpeg(200), source_seq=base)
    recorder.write_frame(_jpeg(210), source_seq=base + 1)
    recorder.stop(END_REASON_STOP)
    return recorder, first


def _observed(recorder, first):
    from scripts.world_build_session import follow_capture

    return list(
        follow_capture(recorder.capture_dir(first), poll_seconds=0.001,
                       max_idle_polls=3)
    )


def test_a_frames_source_path_holds_that_frames_bytes_across_a_reconnect(tmp_path):
    recorder, first = _walk_across_a_reconnect(
        tmp_path, second_capture_restarts_numbering=False
    )
    frames = _observed(recorder, first)

    assert len(frames) == 4, "the walk was cut at the reconnect"
    for frame in frames:
        assert frame.source_path is not None
        assert frame.source_path.exists(), (
            f"source_path {frame.source_path} does not exist: the ledger points "
            "at the capture the follow STARTED in, not the one this frame is in"
        )
        assert frame.source_path.read_bytes() == frame.payload, (
            f"source_path {frame.source_path} exists and holds a DIFFERENT image"
        )


def test_a_restarted_source_index_does_not_resolve_to_a_wrong_photograph(tmp_path):
    """The variant that would have been silent corruption rather than a gap."""
    recorder, first = _walk_across_a_reconnect(
        tmp_path, second_capture_restarts_numbering=True
    )
    frames = _observed(recorder, first)

    assert len(frames) == 4
    after_the_reconnect = frames[2:]
    for frame in after_the_reconnect:
        assert frame.source_path.read_bytes() == frame.payload, (
            "a frame from the successor capture resolved to the same-numbered "
            "frame of the PREDECESSOR -- a real photograph, from another moment"
        )
