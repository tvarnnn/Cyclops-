"""One CV Lab run: its identity, its counters, and its measurements.

A run is the unit of provenance. Everything the Lab publishes -- every
`frame_result`, every aggregate metric, every timing -- names the run that
produced it, and a run names exactly one experiment. That is the whole
mechanism by which a result from a previous experiment cannot be read as a
result from the current one: there is no shared accumulator to leak
through, because a new experiment is a new run object.

**Constant memory, by construction.** Every accumulator here is O(1) in
frames: a mean is a running total and a count, a maximum is a maximum. A
run has no frame list, no metric history and no sample buffer. This is not
frugality -- `handoff.md` 9.3 says a `stream_stop` MAY NEVER ARRIVE, so
"for the length of a run" means "for as long as the Tower is up", and
anything that grew per frame would grow without bound.

**Aggregation is the experiment's decision, not this module's.** A
per-frame `tracked_fraction` summed over 9,199 frames is 768, which is the
kind of number that gets printed and believed. `MetricKind` exists so the
producer says how its numbers combine, and this file asks rather than
guesses -- including for the metric it has never heard of, which is
reported as unclassified rather than folded in under a default.
"""

import logging
import time

from tower.cv_lab.contracts import (
    MAX_REPORTED_METRICS,
    MAX_TRACKED_STAGES,
    MAX_UNCLASSIFIED_REPORTED,
)
from tower.experiments import (
    PROVENANCE_MEASURED,
    MetricKind,
    UnclassifiedMetricError,
    classify_metric,
)

logger = logging.getLogger(__name__)


class Running:
    """Mean and maximum of a stream of values, in constant memory.

    The same shape as `tower.metrics._Running`, and deliberately a
    separate class rather than an import of a private name from a shared
    module: `SessionMetrics` measures a CONNECTION and this measures a
    RUN, the two windows do not coincide, and coupling them would make a
    change to one a change to the other.
    """

    __slots__ = ("count", "total", "maximum", "minimum", "last")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.minimum = 0.0
        self.last = 0.0

    def add(self, value: float) -> None:
        if self.count == 0:
            self.maximum = value
            self.minimum = value
        else:
            if value > self.maximum:
                self.maximum = value
            if value < self.minimum:
                self.minimum = value
        self.count += 1
        self.total += value
        self.last = value

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


class _MetricAccumulator:
    """One metric name, combined the way its experiment said to combine it.

    RATE          reports the mean. Summing a fraction is nonsense.
    COUNT         reports the sum, including for a 0/1 flag whose sum is
                  "how many frames it fired on".
    CONSTANT      reports the value observed, and says so if it ever
                  changed. Neither a sum nor a mean of an image width
                  means anything.
    UNAGGREGATED  reports NO value at all, only a frame count.
                  `dominant_direction_deg` is circular: the mean of 179
                  and -179 is 0, the one direction neither frame was
                  moving in. A null is the honest answer and the fourth
                  kind exists so that null gets published instead of a
                  confident zero.
    """

    __slots__ = ("kind", "running", "total", "first", "varied", "frames")

    def __init__(self, kind: MetricKind) -> None:
        self.kind = kind
        self.running = Running()
        self.total = 0.0
        self.first: float | None = None
        self.varied = False
        self.frames = 0

    def add(self, value: float) -> None:
        self.frames += 1
        if self.kind is MetricKind.RATE:
            self.running.add(value)
        elif self.kind is MetricKind.COUNT:
            self.total += value
        elif self.kind is MetricKind.CONSTANT:
            if self.first is None:
                self.first = value
            elif value != self.first:
                self.varied = True
        # UNAGGREGATED accumulates nothing on purpose.

    @property
    def value(self) -> float | None:
        if self.kind is MetricKind.RATE:
            return self.running.average if self.running.count else None
        if self.kind is MetricKind.COUNT:
            return self.total
        if self.kind is MetricKind.CONSTANT:
            # A constant that was not constant is not a constant. Reported
            # as no value with `varied` set, rather than as whichever one
            # happened to arrive first.
            return None if self.varied else self.first
        return None


class LabRun:
    """The measurements and the identity of one experiment run.

    Mutated only from the event loop (the frame path and the command
    handlers both run there) and read from a worker thread by the result
    channel's poller. Nothing in here takes a lock, and `CVLab` does not
    hold one while reading these counters either -- an earlier version of
    this comment claimed it did, and it does not.

    What makes that safe is not a lock. It is that the reader reads each
    counter ONCE into a local and derives `frames_offered` from those
    locals (see `CVLab._run_document`), so the three it publishes are one
    consistent triple whatever the writer does next. Reading the derived
    property and then the three attributes separately would be atomic
    only by accident of CPython's scheduling.
    """

    def __init__(
        self,
        *,
        run_id: str,
        experiment_id: str,
        descriptor: dict,
        origin: str,
        started_at: float,
    ) -> None:
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.descriptor = descriptor
        self.origin = origin
        self.started_at = started_at
        self.ended_at: float | None = None
        # How long the arm took, in milliseconds, from the moment the run
        # was minted to the moment its experiment was installed and could
        # answer a frame. `None` until then, and `None` forever for a run
        # whose arm failed. Measured on `perf_counter`, not on the
        # wall clock `started_at` uses: on Windows `time.time()` ticks
        # every ~15.6 ms, which is coarser than a cheap experiment's whole
        # arm.
        self.arm_ms: float | None = None
        self.arm_started_perf = time.perf_counter()
        # Runtime facts the experiment reports about itself once loaded --
        # device, weights, versions. Empty until `describe()` is asked,
        # and empty forever for an experiment that does not implement it.
        self.runtime: dict = {}

        # What became of every frame this run was offered. Three counters,
        # and `frames_offered` DERIVED from them rather than stored --
        # which is the only way the sum can be trusted.
        #
        # It used to be a fourth counter, incremented on arrival and the
        # others on the outcome. The status document is built on a worker
        # thread while the frame path runs on the loop, so a snapshot
        # taken between those two increments published four numbers that
        # did not add up: measured at 2,741 such snapshots over a run
        # with a 2 ms experiment. The invariant below is the whole
        # diagnostic -- offered 0 means the stream is not reaching the Lab
        # at all, offered>0 with processed 0 means it is and the Lab is
        # refusing -- and an invariant that is false 2,741 times is not
        # one. Locking the frame path to fix a diagnostic would have been
        # paying in the wrong currency; deriving it costs nothing and is
        # true by construction.
        #
        # A frame currently being processed is therefore not yet counted
        # anywhere, which is correct: its outcome is not known.
        self.frames_processed = 0
        self.frames_refused = 0
        self.frames_failed = 0

        # Dense, per run, starting at 1. Assigned to a frame at the moment
        # it is processed and carried on that frame's `frame_result`, so a
        # client can order results within a run without trusting the wire
        # `seq`, which is the phone's capture index and skips by design.
        self.result_seq = 0

        self.processing_ms = Running()
        self.stage_ms: dict[str, Running] = {}
        self.headline = Running()
        self.headline_label: str | None = None
        self._metrics: dict[str, _MetricAccumulator] = {}
        self._unclassified: list[str] = []
        self._stages_rejected = 0
        self.last_frame_at: float | None = None
        self.last_result_at: float | None = None

    @property
    def frames_offered(self) -> int:
        """Every frame whose outcome this run knows. Derived, see above."""
        return self.frames_processed + self.frames_refused + self.frames_failed

    @property
    def is_over(self) -> bool:
        return self.ended_at is not None

    # -- recording ------------------------------------------------------
    #
    # Every one of these is a no-op once the run has ended. A stopped run
    # publishes `ended_at`, which freezes `elapsed_s`, and the Tower tells
    # a client in as many words that "the last run's figures are final" --
    # so a run that kept counting refused frames against a frozen window
    # reported a throughput that climbed forever with nothing happening.
    # Measured: 8 frames over 9 s read 0.89 offered_fps; 400 more refused
    # frames on the same stopped run read 45.3, from the same 9 seconds.
    #
    # The question those counters were there to answer -- "I pressed Stop,
    # is the phone still streaming?" -- is answered by the Lab-scoped
    # `source` block, which never stops counting. It is a property of the
    # Tower, not of a run that is over.

    def record_refused(self, now: float) -> None:
        if self.is_over:
            return
        self.frames_refused += 1
        self.last_frame_at = now

    def record_failed(self, now: float) -> None:
        if self.is_over:
            return
        self.frames_failed += 1
        self.last_frame_at = now

    def record_result(self, result, now: float) -> int:
        """Fold one `ExperimentResult` in. Returns the frame's result_seq.

        Never raises. This runs on the frame path, and the frame path
        answering a client must not end because a measurement could not be
        filed.
        """
        if self.is_over:
            return self.result_seq
        self.frames_processed += 1
        self.result_seq += 1
        self.last_frame_at = now
        self.last_result_at = now
        try:
            self.processing_ms.add(float(result.processing_ms))
            for stage, ms in (result.stage_ms or {}).items():
                name = str(stage)[:64]
                running = self.stage_ms.get(name)
                if running is None:
                    if len(self.stage_ms) >= MAX_TRACKED_STAGES:
                        # An experiment naming a stage per frame is a bug,
                        # but it must not become an unbounded dict in a run
                        # that stays open for the life of the Tower.
                        self._stages_rejected += 1
                        continue
                    running = self.stage_ms[name] = Running()
                running.add(float(ms))
            self.headline_label = result.result_label
            self.headline.add(float(result.result_value))
            for name, value in (getattr(result, "metrics", None) or {}).items():
                self._record_metric(str(name), float(value))
        except Exception:
            logger.exception(
                "[Tower][CVLab] run %s could not record a result; the frame "
                "was still answered",
                self.run_id,
            )
        return self.result_seq

    def _record_metric(self, name: str, value: float) -> None:
        """File one metric, or refuse to guess at it.

        This is also where the accumulator is BOUNDED, and the bound is
        not a number in this file: a name is added only if
        `classify_metric` recognises it, which means only if the
        experiment declared it in its own `METRIC_KINDS`. That is a
        compile-time set -- twelve entries at its largest. An arbitrary
        cap on top of it would never fire.
        """
        accumulator = self._metrics.get(name)
        if accumulator is None:
            try:
                kind = classify_metric(self.experiment_id, name)
            except UnclassifiedMetricError:
                # Loud in the log, visible on the wire, and NOT folded in
                # under a guessed default -- guessing is what let fifteen
                # mis-summed rates go unnoticed before `MetricKind`
                # existed. Recorded once per name per run.
                if (
                    name not in self._unclassified
                    and len(self._unclassified) < MAX_UNCLASSIFIED_REPORTED
                ):
                    self._unclassified.append(name)
                    logger.warning(
                        "[Tower][CVLab] %s emitted %r, which it never "
                        "classified; excluded from this run's aggregate",
                        self.experiment_id,
                        name,
                    )
                return
            except LookupError:
                # The experiment is not in the registry at all. Only
                # reachable for an injected test experiment, and the same
                # answer serves: report nothing rather than guess.
                return
            accumulator = _MetricAccumulator(kind)
            self._metrics[name] = accumulator
        accumulator.add(value)

    # -- reporting ------------------------------------------------------

    def metric_rows(self, metadata) -> tuple[list[dict], int]:
        """Every aggregate metric, plus how many did not fit.

        The headline comes first and always: an experiment that cannot
        name its single most important number has not decided what it is
        measuring, and a list where it is buried alphabetically hides that
        decision.
        """
        rows: list[dict] = []
        seen: set[str] = set()

        if self.headline_label is not None and self.headline.count:
            seen.add(self.headline_label)
            rows.append(
                self._row(
                    label=self.headline_label,
                    value=self.headline.average,
                    kind=MetricKind.RATE,
                    frames=self.headline.count,
                    unit=metadata.headline_unit,
                    metadata=metadata,
                    headline=True,
                    observed=self.headline,
                )
            )

        for name in sorted(self._metrics):
            if name in seen:
                continue
            accumulator = self._metrics[name]
            rows.append(
                self._row(
                    label=name,
                    value=accumulator.value,
                    kind=accumulator.kind,
                    frames=accumulator.frames,
                    unit=metadata.metric_units.get(name),
                    metadata=metadata,
                    headline=False,
                    varied=accumulator.varied,
                    observed=(
                        accumulator.running
                        if accumulator.kind is MetricKind.RATE
                        else None
                    ),
                )
            )

        omitted = max(0, len(rows) - MAX_REPORTED_METRICS)
        return rows[:MAX_REPORTED_METRICS], omitted

    def _row(
        self,
        *,
        label: str,
        value: float | None,
        kind: MetricKind,
        frames: int,
        unit: str | None,
        metadata,
        headline: bool,
        varied: bool = False,
        observed=None,
    ) -> dict:
        # The three figures below are what makes a raw scalar readable
        # without inventing a threshold for it.
        #
        # `sharpness_laplacian_var: 483.068` says nothing to anybody. The
        # same number beside "this run has seen 79 to 1309" says the frame
        # is middling, and it says it from THIS camera in THIS room rather
        # than from a constant somebody picked once. The standard
        # reference for variance-of-Laplacian is explicit that its
        # threshold has to be tuned per dataset, and this Lab has one
        # physical run to tune against, which is none -- so the honest
        # move is to publish the distribution and let the reader place the
        # value in it.
        #
        # RATE metrics only. A COUNT's running total has no "range this
        # run", a CONSTANT has no range by definition, and an
        # UNAGGREGATED one has no meaningful anything -- reporting a
        # min/max for those would be three more numbers that mean nothing,
        # which is the problem rather than the fix.
        latest = observed.last if observed is not None and observed.count else None
        lowest = observed.minimum if observed is not None and observed.count else None
        highest = observed.maximum if observed is not None and observed.count else None
        return {
            # The Tower's word, displayed verbatim. iOS matches on no
            # metric name, ever.
            "label": label,
            # `null` is a real answer here and means "this metric has no
            # meaningful aggregate", not "zero". See MetricKind.
            "value": value,
            # `null` means the quantity genuinely has no unit and is
            # rendered bare. IOS-to-Tower.md 0.5.
            "unit": unit,
            # How this number was combined across frames, so a reader
            # knows whether they are looking at a mean, a total, or a
            # configured constant.
            "aggregation": kind.value,
            "frames": frames,
            # REQUIRED, never omitted. A CONSTANT is a configured value --
            # a threshold, an image dimension -- and is a fact about how
            # this Tower is set up rather than a model's opinion, so it is
            # measured even when the experiment as a whole infers.
            "provenance": (
                PROVENANCE_MEASURED
                if kind is MetricKind.CONSTANT
                else metadata.provenance
            ),
            # The Tower has no calibrated confidence for any of these.
            # `null` says so; a number here would be invented.
            "confidence": None,
            "headline": headline,
            # A CONSTANT that changed. Reported so that a null value is
            # not mistaken for "never observed".
            "varied": varied,
            # iOS renders a better/worse verdict only when BOTH a baseline
            # and a direction arrive. The Lab has neither and says why
            # rather than leaving a consumer to wonder.
            "baseline": None,
            "higher_is_better": None,
            # The most recent frame's value, and the range this RUN has
            # observed. All three are `null` for a metric that is not a
            # RATE. See the note at the top of this method for why they
            # exist and for what a consumer may and may not conclude from
            # them: they place a number inside THIS run and say nothing
            # whatever about whether it is good.
            "latest": latest,
            "observed_min": lowest,
            "observed_max": highest,
        }

    def metric_total(self, name: str) -> float | None:
        """The COUNT-kind total for one metric, or None if it has none.

        The one accessor the status document needs into the accumulator
        dict, so that reporting does not reach into a private mapping to
        answer "how many things did this run find".
        """
        accumulator = self._metrics.get(name)
        if accumulator is None or not accumulator.frames:
            return None
        if accumulator.kind is MetricKind.COUNT:
            return accumulator.total
        return accumulator.value

    @property
    def unclassified_metrics(self) -> list[str]:
        return list(self._unclassified)

    @property
    def stages_rejected(self) -> int:
        """Stage names dropped because the run already held the maximum."""
        return self._stages_rejected

    @property
    def tracked_metric_count(self) -> int:
        """How many distinct metric names this run has filed.

        Reported for the test that pins the bound, not for the wire. The
        number that matters on the wire is how many were SHOWN, and
        `metric_rows` returns that beside how many it dropped.
        """
        return len(self._metrics)
