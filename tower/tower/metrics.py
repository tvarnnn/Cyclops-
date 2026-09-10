import time
from typing import Callable

import psutil


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class _Running:
    """Mean and maximum of a stream of values, in constant memory."""

    __slots__ = ("count", "total", "maximum")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        # 0.0 rather than -inf: every existing caller rounds this straight
        # into a report, and an empty window has always reported 0.0.
        self.maximum = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        if value > self.maximum:
            self.maximum = value

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


class SessionMetrics:
    """Tracks measurements for a single WebSocket connection's frame stream.

    A new instance must be created per connection, never reused across a
    reconnect: a new session is a new observation stream, not a
    continuation of the previous one (07-PLATFORM-CONSTRAINTS.md,
    Limitation 4).

    ``seq_gap_total`` is a RAW, causally-neutral count of discontinuities
    in the received ``seq`` field. It is deliberately NOT named/labeled as
    "frames lost in transit". Confirmed 2026-08-19: the current iOS sender
    assigns ``seq`` from the DAT/source capture-frame index (``frameCount``
    of incoming VideoFrames), but only forwards roughly 1-in-30 of them
    (a throttled capture -> transmit branch), so the Tower normally
    receives seq like 1, 30, 60, 90, ... by design. Under the CURRENT wire
    protocol (seq, width, height, format, data -- no separate
    transmission-attempt counter), a gap in ``seq`` cannot be attributed
    to any single cause: it may be intentional sender-side sampling,
    a sender-side drop, or genuine network/transit loss, and those look
    identical on the wire. Do not report this number as network loss.
    See 07-PLATFORM-CONSTRAINTS.md Limitation 9 for the future
    source_seq/tx_seq protocol split that would be required to actually
    distinguish these causes -- not implemented as of V0.7.

    ``source_fps_estimate`` and ``sampling_stride_avg`` exist so that the
    sender behavior described above no longer has to be inferred by hand.
    Both are derived from ``source_seq`` (the DAT/capture frame index,
    which falls back to ``seq`` for every sender that exists today):
    ``sampling_stride_avg`` is how many capture frames pass per frame the
    Tower actually receives (~30 for the current sender, 1.0 for a sender
    forwarding everything), and ``source_fps_estimate`` is the upstream
    capture rate that stride is sampling. Added 2026-08-21 after the first
    physical-glasses remote run required hand arithmetic over log lines to
    establish that its 0.8 received fps was a ~24 fps capture stream
    decimated 1-in-30, rather than transit loss. They are estimates
    labeled as such: they assume ``source_seq`` advances monotonically,
    once per captured frame, at a roughly steady rate, and they say
    nothing about *why* frames were not forwarded -- only how many were
    skipped. Both are computed from the first and last capture index
    observed, so a mid-window sender restart that resets the index
    understates them; a restart that resets it below the first index
    observed makes them unavailable rather than negative.

    Two further limits on ``sampling_stride_avg``, both deliberate:

    * Its denominator is frames actually RECORDED, so intermittent
      Tower-side rejection inflates it. A sender forwarding every capture
      frame, with every other frame answered ``invalid_frame``, reports a
      stride of ~2.0 -- a Tower-side loss misattributed to the sender.
      This is not fixable from inside the metric (rejected frames have no
      trustworthy capture index), so ``frames_rejected`` is reported
      alongside to make the condition visible. **Read the stride as
      trustworthy only when ``frames_rejected`` is 0.**
    * ``source_fps_estimate`` spans the first-to-last RECEIVED FRAME
      (reported as ``source_frame_span_s``), whereas ``effective_fps``
      spans the whole ``stream_start``-bounded window
      (``session_duration_s``). They are not two views of one interval: a
      burst of frames followed by a long silence gives a high
      ``source_fps_estimate`` and a low ``effective_fps`` with no
      contradiction between them.

    ``backpressure_drops`` is tracked separately from ``seq_gap_total`` on
    purpose (2026-08-19 V0.7 planning decision) -- the two have different
    causes (network/link loss or sender sampling vs. a Tower-side drop
    policy) and must never be combined into one number. As of V0.7, no
    code path increments ``backpressure_drops``: the receive -> process ->
    ack loop in tower/routes/ws.py is intentionally left unchanged this
    milestone, so this field will always read 0 until a future milestone
    adds a real drop mechanism.
    """

    SUMMARY_LOG_FRAME_INTERVAL = 150

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._process = psutil.Process()
        # Cumulative CPU time at session start. Deliberately NOT
        # cpu_percent(interval=None), which measures only since its own
        # previous call on this object: snapshot() is called periodically
        # as well as at finalize, so that would make the final summary's
        # CPU figure describe the sliver since the last periodic summary
        # rather than the session. Measured 2026-08-21: two back-to-back
        # snapshots of one busy session reported 95.9 then 0.0.
        self._cpu_times_at_start = self._process.cpu_times()
        self.start_time = clock()
        self.last_seq: int | None = None
        self.last_tx_seq: int | None = None
        self.frames_received = 0
        self.seq_gap_total = 0
        # Counted only for senders that actually provide tx_seq. Stays
        # None otherwise -- see snapshot() and Rule 3.
        self.tx_seq_gap_total: int | None = None
        self.backpressure_drops = 0
        self.frame_processing_errors = 0
        # Frames that arrived but never reached record_frame, so they are
        # absent from every other figure here -- see snapshot().
        self.frames_rejected = 0
        self.bytes_received = 0
        # First/last observed capture index and the wall-clock span between
        # them -- the two endpoints are all that is needed for the upstream
        # rate estimates, so this stays O(1) in memory regardless of
        # session length.
        self._first_source_seq: int | None = None
        self._last_source_seq: int | None = None
        self._first_frame_time: float | None = None
        self._last_frame_time: float | None = None
        # Running accumulators, not lists. Only the mean and the max are
        # ever reported, and both are O(1) to maintain -- while a list
        # grows one float per frame per measurement for as long as the
        # window is open. A measurement window ends at stream_stop or at a
        # disconnect, and `handoff.md` 9.3 says a stream_stop MAY NEVER
        # ARRIVE, so "as long as the window is open" means "as long as the
        # connection lives". Small per frame, unbounded in principle, and
        # free to fix.
        self._receive_to_result_ms = _Running()
        self._cv_processing_ms = _Running()
        self._stage_ms: dict[str, _Running] = {}

    def record_frame(
        self,
        seq: int,
        byte_count: int,
        receive_to_result_ms: float,
        cv_processing_ms: float,
        stage_ms: dict[str, float] | None = None,
        tx_seq: int | None = None,
        source_seq: int | None = None,
    ) -> None:
        # Legacy callers pass only `seq`, which *is* the capture index for
        # every sender that exists today (tower/frames.py applies the same
        # fallback on the wire side).
        capture_index = seq if source_seq is None else source_seq
        now = self._clock()
        if self._first_source_seq is None:
            self._first_source_seq = capture_index
            self._first_frame_time = now
        self._last_source_seq = capture_index
        self._last_frame_time = now

        if self.last_seq is not None and seq > self.last_seq + 1:
            self.seq_gap_total += seq - self.last_seq - 1
        self.last_seq = seq

        if tx_seq is not None:
            # First tx_seq-bearing frame establishes the counter at 0 gaps;
            # from then on a jump means messages the sender transmitted and
            # this Tower never SAW -- transit loss, unlike a seq gap, which
            # cannot be attributed to any single cause. A frame this Tower
            # saw and refused advances the counter too, from
            # `record_frame_rejected`, so a refusal is not scored as a loss.
            if self.tx_seq_gap_total is None:
                self.tx_seq_gap_total = 0
            elif self.last_tx_seq is not None and tx_seq > self.last_tx_seq + 1:
                self.tx_seq_gap_total += tx_seq - self.last_tx_seq - 1
            self.last_tx_seq = tx_seq
        self.frames_received += 1
        self.bytes_received += byte_count
        self._receive_to_result_ms.add(receive_to_result_ms)
        self._cv_processing_ms.add(cv_processing_ms)
        if stage_ms:
            for stage_name, ms in stage_ms.items():
                self._stage_ms.setdefault(stage_name, _Running()).add(ms)

    def record_frame_processing_error(self) -> None:
        self.frame_processing_errors += 1

    def record_frame_rejected(self, tx_seq: int | None = None) -> None:
        """Count a frame that arrived but was never recorded.

        Every frame answered with a `frame_error` -- invalid_frame,
        frame_skipped, or module_unavailable -- lands here. Distinct from
        ``frame_processing_errors``, which counts only the module-level
        subset: this one answers "how many arriving frames are missing
        from the numbers below", which is what makes an inflated
        ``sampling_stride_avg`` detectable.
        """
        self.frames_rejected += 1
        # A REFUSED FRAME STILL ARRIVED, so it must advance the transmit
        # counter or the next accepted frame looks like transit loss.
        #
        # `record_frame` is the only other place `last_tx_seq` moves, and a
        # refused frame never reaches it -- so `tx_seq_gap_total`, which
        # this file calls "genuine transit loss", counted every Tower-side
        # refusal as a lost message. Measured by an adversarial review: 10
        # transmitted, 0 lost, and the number reported 3, which was exactly
        # `frames_rejected`.
        #
        # `tx_seq` is None for a message refused BEFORE it decoded -- there
        # is no frame to read it from -- and that residue is not
        # attributable by any means this protocol has. `frames_rejected`
        # is reported beside the gap so the reader can see how much room
        # for doubt there is.
        if tx_seq is not None:
            if self.tx_seq_gap_total is None:
                self.tx_seq_gap_total = 0
            self.last_tx_seq = tx_seq

    def should_log_summary(self) -> bool:
        return (
            self.frames_received > 0
            and self.frames_received % self.SUMMARY_LOG_FRAME_INTERVAL == 0
        )

    def _session_cpu_percent(self, elapsed_s: float) -> float:
        """Average CPU use across the whole session, as percent of one core.

        Not clamped to 100: a multi-core process legitimately exceeds it,
        matching psutil's own convention.
        """
        now = self._process.cpu_times()
        cpu_seconds = (now.user - self._cpu_times_at_start.user) + (
            now.system - self._cpu_times_at_start.system
        )
        return round(cpu_seconds / elapsed_s * 100, 2)

    def _upstream_rate_fields(self) -> dict:
        """Estimate the upstream capture rate and the sender's sampling stride.

        All three fields stay None unless at least two frames have been
        received AND the capture index actually advanced between them. One
        frame cannot establish a rate, and a sender restart can make the
        index regress -- neither is a zero, so neither may be reported as
        one (Rule 3: unknown stays unknown).
        """
        unavailable = {
            "source_seq_span": None,
            "source_frame_span_s": None,
            "sampling_stride_avg": None,
            "source_fps_estimate": None,
        }
        if (
            self.frames_received < 2
            or self._first_source_seq is None
            or self._last_source_seq is None
        ):
            return unavailable

        span = self._last_source_seq - self._first_source_seq
        if span <= 0:
            return unavailable

        frame_span_s = (self._last_frame_time or 0.0) - (self._first_frame_time or 0.0)
        fields = {
            "source_seq_span": span,
            # Reported so source_fps_estimate is auditable rather than
            # trust-me: it is exactly source_seq_span / source_frame_span_s.
            # Note this is a DIFFERENT window from session_duration_s.
            "source_frame_span_s": round(frame_span_s, 3),
            "sampling_stride_avg": round(span / (self.frames_received - 1), 2),
            "source_fps_estimate": None,
        }
        if frame_span_s > 0:
            fields["source_fps_estimate"] = round(span / frame_span_s, 2)
        return fields

    def snapshot(self) -> dict:
        elapsed_s = max(self._clock() - self.start_time, 1e-9)
        return {
            "session_duration_s": round(elapsed_s, 3),
            "frames_received": self.frames_received,
            "effective_fps": round(self.frames_received / elapsed_s, 2),
            # Sit next to effective_fps deliberately: read together they
            # say "capture ran at X fps, the sender forwarded 1-in-Y, so
            # the Tower observed Z fps" without any hand arithmetic.
            **self._upstream_rate_fields(),
            "bytes_received": self.bytes_received,
            # BYTES per second, not bits -- the name predates the
            # observation and is kept for backward compatibility with
            # existing report templates. Documented in README.md;
            # misreading it as bits/s understates throughput 8x.
            "bandwidth_bps": round(self.bytes_received / elapsed_s, 2),
            "seq_gap_total": self.seq_gap_total,
            # None (not 0) when the sender doesn't send tx_seq: "we cannot
            # tell" must not be reported as "no loss occurred" (Rule 3).
            "tx_seq_gap_total": self.tx_seq_gap_total,
            "backpressure_drops": self.backpressure_drops,
            "frame_processing_errors": self.frame_processing_errors,
            # Non-zero means figures above understate what arrived, and in
            # particular that sampling_stride_avg may be overstated.
            "frames_rejected": self.frames_rejected,
            "receive_to_result_ms_avg": round(self._receive_to_result_ms.average, 3),
            "receive_to_result_ms_max": round(
                self._receive_to_result_ms.maximum, 3
            ),
            "cv_processing_ms_avg": round(self._cv_processing_ms.average, 3),
            # Session average, not an instantaneous sample -- see
            # _session_cpu_percent().
            "process_cpu_percent": self._session_cpu_percent(elapsed_s),
            "process_rss_bytes": self._process.memory_info().rss,
            "stage_ms_avg": {
                k: round(v.average, 3) for k, v in self._stage_ms.items()
            },
            "stage_ms_max": {
                k: round(v.maximum, 3) for k, v in self._stage_ms.items()
            },
        }
