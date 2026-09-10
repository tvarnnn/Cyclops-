"""Frames that never reach `record_frame` must be visible in the snapshot.

`sampling_stride_avg` divides the observed capture-index span by the
number of frames actually *recorded*. Frames the Tower rejected
(`invalid_frame`, `frame_skipped`, `module_unavailable`) never increment
that denominator, but the capture indices they carried still sit inside
the span -- so intermittent Tower-side rejection inflates the apparent
sender stride. A sender forwarding every single frame, with every other
frame rejected, reports a stride of ~2.0: "the sender forwards 1-in-2",
which is a Tower-side loss misattributed to the sender.

`frames_rejected` exists so that misattribution is always detectable in
the same snapshot. Before it, the `invalid_frame` and `module_unavailable`
paths incremented no counter at all and left nothing to contradict the
inflated stride.
"""
from tower.metrics import SessionMetrics


def _record(metrics, seq, **overrides):
    kwargs = {
        "seq": seq,
        "byte_count": 100,
        "receive_to_result_ms": 1.0,
        "cv_processing_ms": 1.0,
    }
    kwargs.update(overrides)
    metrics.record_frame(**kwargs)


def test_frames_rejected_starts_at_zero():
    assert SessionMetrics().snapshot()["frames_rejected"] == 0


def test_record_frame_rejected_increments_the_counter():
    metrics = SessionMetrics()

    metrics.record_frame_rejected()
    metrics.record_frame_rejected()

    assert metrics.snapshot()["frames_rejected"] == 2


def test_rejected_frames_do_not_count_as_received():
    metrics = SessionMetrics()

    _record(metrics, seq=1)
    metrics.record_frame_rejected()

    snapshot = metrics.snapshot()
    assert snapshot["frames_received"] == 1
    assert snapshot["frames_rejected"] == 1


def test_intermittent_rejection_inflates_stride_but_is_visible():
    """The misattribution scenario, locked in as documented behavior.

    The sender forwards every capture frame (true stride 1.0). The Tower
    rejects every other one. `sampling_stride_avg` therefore reads ~2.0 --
    it can only measure the span between frames it recorded. That is not
    fixable from inside the metric, so the requirement is that
    `frames_rejected` makes the inflation visible rather than silent.
    """
    metrics = SessionMetrics()

    for seq in range(1, 100):
        if seq % 2:
            _record(metrics, seq=seq)
        else:
            metrics.record_frame_rejected()

    snapshot = metrics.snapshot()

    assert snapshot["sampling_stride_avg"] == 2.0  # apparent, not real
    assert snapshot["frames_received"] == 50
    assert snapshot["frames_rejected"] == 49  # the contradiction is visible


def test_a_clean_run_reports_no_rejections_so_the_stride_is_trustworthy():
    metrics = SessionMetrics()

    for seq in (1, 31, 61, 91):
        _record(metrics, seq=seq)

    snapshot = metrics.snapshot()

    assert snapshot["sampling_stride_avg"] == 30.0
    assert snapshot["frames_rejected"] == 0


def test_a_refused_frame_is_not_scored_as_transit_loss():
    """`tx_seq_gap_total` is the one number that says "lost in the air".

    A frame this Tower SAW and refused is not a lost message, but it never
    reaches `record_frame`, so it used to leave a hole that the next
    accepted frame turned into a gap. Measured by an adversarial review
    before the fix: 10 transmitted, 0 lost, and the number reported 3 --
    exactly `frames_rejected`.
    """
    import itertools

    from tower.metrics import SessionMetrics

    m = SessionMetrics(clock=itertools.count(0.0, 0.1).__next__)
    for tx in range(10):
        if tx in (3, 4, 7):
            m.record_frame_rejected(tx_seq=tx)
            continue
        m.record_frame(seq=tx, byte_count=100, receive_to_result_ms=1.0,
                       cv_processing_ms=1.0, tx_seq=tx, source_seq=tx)
    snapshot = m.snapshot()
    assert snapshot["frames_rejected"] == 3
    assert snapshot["tx_seq_gap_total"] == 0, (
        "a frame this Tower refused was counted as one it never received"
    )


def test_a_genuinely_lost_frame_is_still_scored():
    """The fix must not have made the instrument blind."""
    import itertools

    from tower.metrics import SessionMetrics

    m = SessionMetrics(clock=itertools.count(0.0, 0.1).__next__)
    for tx in (0, 1, 2, 5):          # 3 and 4 never arrived at all
        m.record_frame(seq=tx, byte_count=100, receive_to_result_ms=1.0,
                       cv_processing_ms=1.0, tx_seq=tx, source_seq=tx)
    assert m.snapshot()["tx_seq_gap_total"] == 2


def test_a_frame_refused_before_it_decoded_cannot_be_attributed():
    """And the residue is documented rather than guessed at.

    A message refused before it parsed has no `tx_seq` to read, so it
    leaves a hole nothing can fill. `frames_rejected` is reported beside
    the gap so a reader can see how much room for doubt there is.
    """
    import itertools

    from tower.metrics import SessionMetrics

    m = SessionMetrics(clock=itertools.count(0.0, 0.1).__next__)
    m.record_frame(seq=0, byte_count=100, receive_to_result_ms=1.0,
                   cv_processing_ms=1.0, tx_seq=0, source_seq=0)
    m.record_frame_rejected()                     # unparseable: no tx_seq
    m.record_frame(seq=2, byte_count=100, receive_to_result_ms=1.0,
                   cv_processing_ms=1.0, tx_seq=2, source_seq=2)
    snapshot = m.snapshot()
    assert snapshot["tx_seq_gap_total"] == 1      # honestly unattributable
    assert snapshot["frames_rejected"] == 1       # and the doubt is visible
