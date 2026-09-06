"""The Experimental CV Lab's own types: a result, a protocol, a registry.

Two shapes matter here.

`ExperimentResult` carries a MANDATORY headline (`result_value` +
`result_label`) and an OPTIONAL bag of `metrics`. The headline is a
discipline, not a limitation: an experiment that cannot name its single
most important number has not decided what it is measuring. The bag is
what makes the Lab useful at all -- frame quality has six things to say,
optical flow four, detection a count and a distribution, and a single
scalar threw all of that away.

`Experiment` is a protocol with `load`/`run`/`release` rather than a bare
callable. Before this, a stateful experiment cost a whole `Module`
subclass -- there were two of them sharing one descriptor id, purely
because the depth experiment holds a model across frames. Two of the five
experiments added in V1 are also stateful, so the callable-only registry
would have meant four more near-identical Module classes.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, runtime_checkable

import cv2
import numpy as np

from tower.modules.base import FrameProcessingError


# ORB builds an image pyramid, and its internal resize asserts on a
# non-positive scale when either dimension is exactly 1. Measured across
# every (h, w) in 1..32: ORB fails if and only if a dimension is 1, and
# succeeds at 2xN and Nx2 including 2x1000. So the floor is 2, chosen
# from the measurement rather than from caution.
ORB_MIN_DIMENSION = 2


def _decode(raw_bytes: bytes, flag: int, min_dimension: int):
    """Decode a frame, or fail at FRAME scope -- never at module scope.

    Three distinct failure modes, all of which reach here and all of which
    would otherwise raise a bare `cv2.error`:

    1. `cv2.imdecode` returns **None** for a truncated file. Passing that
       to `cvtColor` raises.
    2. `cv2.imdecode` **raises** `!buf.empty()` for an empty buffer -- the
       `is None` check never sees it.
    3. A perfectly valid image with a **1-pixel dimension** decodes fine
       and then kills ORB.

    Why this matters more than it looks: `ModuleContainer.process` treats
    any exception that is not a `FrameProcessingError` as a MODULE
    failure, `mark_failed()` is terminal (FAILED can never return to
    UNLOADED), and the container is built once at process start with no
    swap path. So a single bad frame would end CV processing for every
    subsequent frame of every subsequent session, for the life of the
    server.

    All three are reachable from the wire. `tower/frames.py` validates a
    frame with `Image.open(...).size`, which parses the JPEG **header**
    only -- a truncated file passes it, and a 1x64 JPEG is not malformed
    at all, merely useless.
    """
    if not raw_bytes:
        raise FrameProcessingError("empty frame payload")
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    try:
        image = cv2.imdecode(array, flag)
    except cv2.error as exc:
        raise FrameProcessingError(f"undecodable frame: {exc}") from exc
    if image is None:
        raise FrameProcessingError("undecodable frame")
    height, width = image.shape[:2]
    if min(width, height) < min_dimension:
        raise FrameProcessingError(
            f"frame {width}x{height} is below the {min_dimension}px minimum "
            "this experiment can process"
        )
    return image


def decode_color(raw_bytes: bytes, *, min_dimension: int = 1):
    """Decode to BGR, or raise FrameProcessingError. See `_decode`."""
    return _decode(raw_bytes, cv2.IMREAD_COLOR, min_dimension)


def decode_gray(raw_bytes: bytes, *, min_dimension: int = 1):
    """Decode to grayscale, or raise FrameProcessingError. See `_decode`."""
    return _decode(raw_bytes, cv2.IMREAD_GRAYSCALE, min_dimension)


@dataclass(frozen=True)
class ExperimentSettings:
    """What an experiment may be told at load time.

    Deliberately tiny. Rule 4's prohibition on designing a generalised
    negotiation protocol before the real constraints are known applies
    here too: this grows a field when an experiment actually needs one.
    """

    device: str = "auto"
    # The intra-op thread budget torch gets while a model-backed experiment
    # holds a model. "auto" resolves per device in
    # `tower.experiments.depth.resolve_torch_threads` (2 on CUDA, 4 on
    # CPU, from a sweep on this host); a positive integer is applied as
    # given; 0 leaves torch's default alone. Cheap experiments ignore it.
    torch_threads: int | str = "auto"


@dataclass(frozen=True)
class ExperimentResult:
    result_value: float
    result_label: str
    processing_ms: float
    stage_ms: dict[str, float]
    mean_intensity: float | None = None
    # Additional measurements, name -> number. Empty for an experiment
    # whose headline says everything. Floats only, on purpose: this is a
    # measurement channel, not a general result channel. A structured
    # result type is V1.0 work and is blocked.
    metrics: dict[str, float] = field(default_factory=dict)


class MetricKind(Enum):
    """How a per-frame metric may be combined across many frames.

    Only the EXPERIMENT knows this. A corpus harness looking at the name
    `tracked_fraction` can guess, and a harness that guesses wrong sums a
    fraction to 768 over 9,199 frames and prints it as though it meant
    something. So the producer declares it, next to the line that
    produces it, and a consumer that meets an undeclared metric raises.

    RATE      a per-frame quantity -- a fraction, a score, a mean, a
              magnitude. The corpus summary is its MEAN. Summing is
              nonsense.
    COUNT     a per-frame tally, including a 0/1 flag whose sum is "how
              many frames". The corpus summary is its SUM. Averaging
              throws away the total.
    CONSTANT  the same number every frame by construction -- an image
              dimension, a configured threshold. BOTH summing and
              averaging are meaningless; it is reported as the value (or
              values) observed, with how many frames carried each.
    UNAGGREGATED
              a per-frame number with no meaningful corpus aggregate at
              all. `dominant_direction_deg` is circular: the mean of 179
              and -179 degrees is 0, which is the direction neither frame
              was moving. The fourth kind exists because the alternative
              is to call such a metric a RATE and publish that 0 -- which
              is the exact defect the other three kinds were added to
              kill. Reported as a frame count and nothing else.
    """

    RATE = "rate"
    COUNT = "count"
    CONSTANT = "constant"
    UNAGGREGATED = "unaggregated"


class UnclassifiedMetricError(LookupError):
    """An experiment emitted a metric it never classified.

    Deliberately an ERROR rather than a default. The predecessor of this
    module classified by allowlist and SUMMED anything it did not
    recognise, so eight dead names and fifteen mis-summed rates went
    unnoticed for as long as nobody happened to read the totals. A
    silent default is what made that invisible; the loud failure is the
    fix.
    """


# What a value on the wire is: a thing this Tower measured, or a thing a
# model inferred. EXPERIMENTAL-CV.md is explicit that experiment output
# "is model inference, not a measured sensor fact, unless the experiment
# specifically validates against a ground-truth reference", and iOS makes
# provenance a REQUIRED field on every metric rather than an optional one
# -- so the party that decodes a reply has to answer it. There is no
# third value and no default: an experiment declares one or it cannot be
# registered.
PROVENANCE_MEASURED = "measured"
PROVENANCE_INFERRED = "inferred"


# -- what an experiment can be SEEN to be doing ------------------------
#
# The kind of derived image an experiment can hand over for the live
# preview, declared in its metadata so the Lab can say "this one has a
# picture" before a frame has arrived. `None` -- the default, and the
# answer for six of the eight registered experiments -- means the
# experiment produces no image, which is a different statement from "it
# produced one and it failed".
#
# The value tells the renderer how to read the array and nothing else. A
# consumer must not switch on it to decide what a picture MEANS: that is
# what `result_label`, `headline_unit` and the copy on the phone are for,
# and a client that inferred "metres" from a depth preview would be
# making exactly the mistake `DepthEstimation`'s docstring exists to
# prevent.

# A 2-D uint8 array of {0, 255} -- `cv2.Canny`'s own output, at whatever
# resolution the frame decoded to. Rendered white-on-black, which is the
# way every Canny example has been printed since 1986.
PREVIEW_KIND_EDGE_MAP = "edge_map"

# A 2-D float32 array of RELATIVE INVERSE depth on an arbitrary scale:
# larger is nearer, the numbers are not metres, and two frames' values
# are not comparable to each other. Rendered through a normalisation
# that says so -- see `tower/cv_lab/preview.py`.
PREVIEW_KIND_RELATIVE_DEPTH = "relative_depth"

# A line drawing of the frame plus the luminance histogram the exposure
# figures were counted off. Deliberately carries no verdict: see
# `QualityPreview`.
PREVIEW_KIND_FRAME_QUALITY = "frame_quality"

# ORB keypoints over that line drawing, with the coverage grid the
# `spatial_coverage` metric is computed from. Answers "where is there
# trackable texture", which is what `keypoint_count: 995` was always
# trying to say and could not.
PREVIEW_KIND_KEYPOINTS = "keypoints"

# Boxes, class names and scores. The one visualisation that makes a wrong
# answer inspectable: a `person` box drawn around a coat stand is a bug
# report, and `count_person: 160` is a mystery.
PREVIEW_KIND_DETECTIONS = "detections"

# Sparse Lucas-Kanade tracks: where each seeded point went, coloured by
# direction, with the ones the forward-backward check threw away marked
# as thrown away.
PREVIEW_KIND_FLOW_TRACKS = "flow_tracks"

# The blurred rectangle and what the detector could still see through it.
# The one preview whose subject IS the privacy machinery.
PREVIEW_KIND_REDACTION = "redaction_regions"

PREVIEW_KINDS = (
    PREVIEW_KIND_EDGE_MAP,
    PREVIEW_KIND_RELATIVE_DEPTH,
    PREVIEW_KIND_FRAME_QUALITY,
    PREVIEW_KIND_KEYPOINTS,
    PREVIEW_KIND_DETECTIONS,
    PREVIEW_KIND_FLOW_TRACKS,
    PREVIEW_KIND_REDACTION,
)


# The longest side, in pixels, of the structure an experiment derives.
#
# The same number as `tower.cv_lab.contracts.PREVIEW_MAX_EDGE_PX`, and
# COPIED rather than imported. `cv_lab` imports this package, so importing
# back would be circular -- and more to the point the Lab is allowed to
# depend on the experiments while the experiments must not depend on the
# Lab. `test_the_preview_bound_is_the_same_on_both_sides` is what stops
# the two drifting, which is the same shape of guard `contracts.CARTRIDGE`
# already uses against `results.contracts`.
PREVIEW_STRUCTURE_MAX_EDGE_PX = 320

# Canny thresholds for the structure background. Lower than
# `edge_detection`'s 100/200 on purpose: that experiment is MEASURING edge
# density and wants a defensible operating point, while this is drawing a
# room for a person to recognise and wants the faint edge of a desk in it.
# The two numbers are unrelated and must never be shared.
_STRUCTURE_CANNY_LOW = 50
_STRUCTURE_CANNY_HIGH = 150


def scene_structure(gray, max_edge_px: int = PREVIEW_STRUCTURE_MAX_EDGE_PX):
    """A frame reduced to a small line drawing. Never raises.

    THE privacy decision of the whole preview surface, made here once
    rather than argued per experiment: **the CV Lab serves no photographic
    content, ever.** Not the frame, not a filtered frame, not a dimmed or
    posterised one. Every overlay -- keypoints, boxes, flow, the redaction
    rectangle -- is drawn over an edge map, which is the same class of
    derived image the Edge Detection experiment already serves and which
    nobody looking at it could mistake for a photograph.

    The alternative was the real frame with the display filter from
    `object_memory/imagery.py` applied and failing closed. It was
    considered and rejected, and the reasons are worth keeping:

    - it needs vendored YuNet weights, so a Tower without them shows
      nothing at all, and "the debug viewer is blank" is a bad failure
      mode for a debug viewer;
    - a display filter is a filter and not a redaction, and the moment a
      photograph is on the wire the argument becomes how good the filter
      is -- which that module itself measures as finding a real face in 4
      of 36 inspected fills;
    - an edge map is enough for the job. A person can tell a chair from a
      doorway from a monitor in one, and that is all a box or a keypoint
      needs to be placed against.

    It is NOT a claim that an edge map is anonymous. It keeps a jawline,
    a hairline and a silhouette, which is exactly why the preview contract
    still declares `raw_ephemeral` and still forbids persisting it.

    Returns a uint8 array of {0, 255} at preview scale. Callers hold this
    and never the frame it came from, which is what lets the module go on
    declaring `retains_raw_imagery=False`.

    **This is new work, and for five of the seven visual experiments it
    is the whole of what a preview costs on the frame path.**
    `edge_detection` and `depth` never call it -- they hand over an array
    they had already built for their own metric, for free. ORB, SSD and
    Lucas-Kanade produce no edge map at all, so four of the remaining
    five have nothing to reuse. `frame_quality` does have one, at
    100/200 and at full resolution, and re-derives anyway; the reasoning
    is at its call site and it is about what the picture looks like, not
    about the half-millisecond.
    """
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest > max_edge_px and longest > 0:
        scale = max_edge_px / float(longest)
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    # Blurred first, the way `edge_detection` does it, because Canny on an
    # unblurred downscale is mostly resampling noise.
    return cv2.Canny(
        cv2.GaussianBlur(gray, (3, 3), 0),
        _STRUCTURE_CANNY_LOW,
        _STRUCTURE_CANNY_HIGH,
    )


@dataclass(frozen=True)
class ScenePreview:
    """A line drawing of one frame, and the size everything else is in.

    Every overlay payload below carries one of these, and every overlay's
    coordinates are in ITS pixel space rather than the frame's. Rescaling
    happens once, at the moment the structure is derived, so no renderer
    has to be trusted with a second scale factor and no overlay can end up
    drawn in the wrong coordinate system -- which is the specific failure
    `IOS-to-Tower.md` 2.5 refuses annotation geometry over: "a wrong
    convention renders confidently in the wrong place".
    """

    structure: object

    @property
    def size(self) -> tuple:
        height, width = self.structure.shape[:2]
        return (int(width), int(height))


@dataclass(frozen=True)
class KeypointPreview:
    """Where the detector found texture, and how much of it there was."""

    scene: ScenePreview
    # Nx2 float32 in `scene` coordinates, ALREADY SUBSAMPLED. A thousand
    # markers on a 320 px panel is a grey rectangle. The experiment
    # decides which survive, because only it knows what `response` means
    # for its own detector.
    xy: object
    # How many were detected before subsampling, so the picture can say
    # "220 of 995" rather than quietly implying 220.
    detected: int
    # The occupancy of the same 8x8 grid `spatial_coverage` is computed
    # from, as a set of (column, row). Drawn as a grid so "995 keypoints
    # all in one corner" looks like what it is.
    coverage_cells: object
    coverage_grid: int
    coverage: float


@dataclass(frozen=True)
class DetectionPreview:
    """What the detector said, including the parts it was unsure about."""

    scene: ScenePreview
    # Nx4 float32 (x0, y0, x1, y1) in `scene` coordinates.
    boxes: object
    labels: tuple
    scores: object
    # The threshold the METRICS used. Detections below it are drawn
    # differently rather than dropped: somebody asking "why does it think
    # that is a person" needs to see the near-misses, and a viewer that
    # silently hid them would be answering a different question.
    threshold: float
    # How many the detector actually produced, before the picture kept the
    # highest-scoring `PREVIEW_MAX_BOXES` of them.
    #
    # Here because the caption without them is a lie in exactly the case
    # this view exists for. Physical testing produced 160 `person`
    # detections in one scene; a picture drawing 24 of them and captioning
    # itself "24 over 0.40" contradicts the `detections: 160` sitting
    # beside it, and does so most confidently when the number is most
    # surprising. `feature_detection` already reports "995 keypoints (54
    # drawn)" for the same reason.
    accepted_total: int
    raw_total: int


@dataclass(frozen=True)
class FlowPreview:
    """How the machine thinks the scene moved."""

    scene: ScenePreview
    # Nx2 each, in `scene` coordinates: where a point was, and where
    # Lucas-Kanade says it went.
    origins: object
    displacements: object
    # Nx2 of the seeds whose track failed the forward-backward check.
    # Drawn, faintly: "it tracked nothing" and "it tracked confident
    # nonsense and threw it away" are identical in the numbers and
    # completely different here.
    rejected: object
    tracked_count: int
    seeded_count: int
    median_flow_px: float


@dataclass(frozen=True)
class RedactionPreview:
    """What the blur covered, and what the detector could still see.

    `before` and `survivors` come from two INDEPENDENT ORB detections --
    one on the frame, one on the blurred copy -- so there is no
    correspondence between individual points and nothing here may claim
    one. `before` is drawn as a dim base layer meaning "there was texture
    here", and a base point with no survivor on it reads as lost without
    asserting an identity the experiment never established.
    """

    scene: ScenePreview
    # (x0, y0, x1, y1) in `scene` coordinates.
    region: tuple
    boundary_margin_px: float
    before: object
    # Survivors, split by where they are relative to the region. Inside is
    # the interesting one: texture the blur did not destroy.
    survived_inside: object
    survived_on_boundary: object
    survived_outside: object


@dataclass(frozen=True)
class QualityPreview:
    """The frame's structure, and the histogram the exposure figures came from.

    No verdict is drawn. Variance-of-Laplacian has no portable threshold
    -- the standard reference for the technique says so in as many words,
    and this Lab has one physical run to calibrate against, which is none
    -- so the picture shows the DISTRIBUTION and the run document says
    where this frame sits within the run's own observed range. Neither
    says "blurry".
    """

    scene: ScenePreview
    # The 256-bin luminance histogram `frame_quality` already computed.
    histogram: object
    overexposed_level: int
    underexposed_level: int
    overexposed_fraction: float
    underexposed_fraction: float


@dataclass(frozen=True)
class ExperimentMetadata:
    """What the Lab can say about an experiment before it runs a frame.

    Presentation, not internals. `METRIC_KINDS` lives beside the code that
    produces the numbers because only that code knows how they combine;
    this lives beside the registration because it is the WIRE surface --
    the name a person reads, the unit a figure is rendered with, and
    whether the figure was measured or inferred. Keeping it here is what
    lets one test assert that every registered experiment has it.

    `metric_units` is deliberately partial. `IOS-to-Tower.md` 0.5: iOS
    "renders a figure with the unit the Tower names and BARE when it names
    none -- a bare number being the honest rendering of an unlabelled
    quantity". An ORB response and a relative inverse depth have no unit,
    and inventing one for them would be worse than the bare number.

    `annotation_metric` names the metric that is an annotation COUNT, if
    the experiment produces one. Only `object_detection` does. A keypoint
    is not an annotation and a fraction is not one either; naming one
    would put a number in a field iOS renders as "things found in this
    frame".
    """

    name: str
    summary: str
    provenance: str
    stateful: bool
    requires_model: bool
    # The `result_label` this experiment will emit. Declared rather than
    # discovered so the Lab can say what an experiment measures BEFORE
    # anyone starts it, and pinned against the real thing by
    # `test_the_declared_headline_is_the_one_the_experiment_emits`.
    headline_label: str
    # What actually does the work. `opencv` or `torch` today. Declared,
    # not derived from `requires_model`: the two coincide for all eight
    # registered experiments and there is no reason a future one could
    # not be a model-free torch kernel or a model-backed OpenCV DNN.
    backend: str
    headline_unit: str | None = None
    metric_units: Mapping[str, str] = field(default_factory=dict)
    annotation_metric: str | None = None
    # The kind of derived image this experiment can hand to the live
    # preview, or `None` for one that produces no image. DECLARED here
    # rather than discovered by asking the loaded experiment, for the
    # same reason `requires_model` is: a phone deciding whether to draw
    # a viewer should not have to start a run to find out whether there
    # will be anything in it.
    #
    # Declaring a kind is not a promise that a picture exists. It is a
    # promise about what one would BE. Whether one exists right now is
    # `run.annotation.artifact`, which is null with a reason until a
    # frame has actually been through.
    preview_kind: str | None = None

    def __post_init__(self) -> None:
        if self.provenance not in (PROVENANCE_MEASURED, PROVENANCE_INFERRED):
            raise ValueError(
                f"provenance must be {PROVENANCE_MEASURED!r} or "
                f"{PROVENANCE_INFERRED!r}, got {self.provenance!r}"
            )
        if self.preview_kind is not None and self.preview_kind not in PREVIEW_KINDS:
            # Loud at import, like every other declaration in this
            # record. A typo here would otherwise reach a phone as a
            # picture the renderer cannot read, and the renderer's honest
            # answer to that is a refusal -- which is a much later and
            # much more confusing place to learn about a misspelling.
            raise ValueError(
                f"preview_kind must be one of {PREVIEW_KINDS!r} or None, "
                f"got {self.preview_kind!r}"
            )


@runtime_checkable
class Experiment(Protocol):
    """One CV experiment, stateful or not.

    `load()` may allocate. `release()` must free whatever it allocated and
    must be safe to call twice, and safe after a partial load -- it runs
    on the FAILED transition, which can be reached from anywhere.

    There is a fourth method, `describe() -> dict`, which is **optional**
    and deliberately not declared here. An experiment that holds a model
    implements it to report what it actually loaded -- the resolved
    device, the weights, the threshold -- because `TOWER_CV_DEVICE=auto`
    is a request and `resolve_device` decides the answer, so a run
    labelled "auto" has not said whether it used the GPU. An experiment
    that holds nothing has nothing to describe, and the Lab reports
    nothing rather than inventing a device it does not have. Adding it to
    the protocol would make six experiments implement a method returning
    an empty dict.

    There are two more, optional for the same reason and paired:

        set_preview_capture(self, enabled: bool) -> None
        take_preview(self) -> numpy.ndarray | None

    An experiment that declares a `preview_kind` implements both. The
    first is how the Lab says "somebody is watching": until it is called
    with `True` the experiment keeps NOTHING, which is what lets the
    module behind this package go on declaring `retains_raw_imagery=
    False` on every Tower that has previews off. The second hands over
    the array the experiment already computed -- by reference, not a
    copy, and clearing its own slot as it goes, so the array has exactly
    one owner at every instant and neither side has to defend against
    the other mutating it.

    `take_preview` returns `None` when there is nothing to hand over,
    which is the normal answer immediately after a take and the only
    answer while capture is off. Neither may raise: a preview is a
    convenience, and `ModuleContainer` treats an unexpected exception on
    the frame path as a TERMINAL module failure.
    """

    name: str

    def load(self, settings: ExperimentSettings) -> None: ...

    def run(self, raw_bytes: bytes) -> ExperimentResult: ...

    def release(self) -> None: ...


class ExperimentPreview:
    """One derived array, held only while somebody is watching it.

    The implementation behind the optional `set_preview_capture` /
    `take_preview` pair documented on `Experiment`. Two registered
    experiments hold one of these and there is exactly one copy of the
    logic, because the logic is the part that has to be right: the
    invariant is that an experiment retains NO imagery until the Lab
    turns capture on, and two hand-written versions of that invariant is
    one more than the number that can be verified at a glance.

    Bounded by shape, not by discipline. There is one slot. `offer`
    overwrites it; `take_preview` empties it. There is no list to append
    to and no size to check, so "the preview store grew" is not a state
    this can reach -- which matters more here than anywhere, because
    `handoff.md` 9.3 says a `stream_stop` may never arrive and a run
    therefore lasts as long as the Tower does.

    Never raises. `offer` is called from inside an experiment's `run()`,
    on the frame path, where `ModuleContainer` turns an unexpected
    exception into a TERMINAL module failure -- so a preview bug would
    end CV processing for the life of the process. There is nothing here
    that can raise, and that is deliberate rather than lucky.
    """

    __slots__ = ("_enabled", "_array")

    def __init__(self) -> None:
        # Off. A Tower that never turns previews on must be able to say
        # `retains_raw_imagery=False` and mean it, and the way to mean it
        # is to hold nothing rather than to hold something nobody asks
        # for.
        self._enabled = False
        self._array = None

    def set_preview_capture(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled:
            # Dropped on the way DOWN, not merely stopped being
            # refreshed. Otherwise pausing a run leaves the last frame
            # the wearer was looking at resident for as long as the
            # experiment stays loaded.
            self._array = None

    @property
    def wanted(self) -> bool:
        """Whether to bother deriving anything for this frame.

        Read INSIDE `run()`, before the derivation, by every experiment
        whose preview costs something to build -- a line drawing, a
        thinned keypoint set, a set of boxes pulled off a tensor. The Lab
        sets this per frame from its own throttle, so an experiment
        running at 60 frames a second derives a picture twenty times a
        second and the other forty frames cost one attribute read.

        `edge_detection` and `depth` do not consult it: their payload is
        an array they had already built, so `offer` is free and gating it
        would cost more than it saved.
        """
        return self._enabled

    def offer(self, array) -> None:
        """Hand over this frame's payload. Free when nobody is watching."""
        if self._enabled:
            self._array = array

    def take_preview(self):
        """The newest array, and this slot gives up its reference to it.

        Clearing is what makes ownership unambiguous: after a take,
        exactly one object holds the array, so neither side has to copy
        it defensively and neither can be surprised by the other.
        """
        array, self._array = self._array, None
        return array


class StatelessExperiment:
    """Adapts a plain `bytes -> ExperimentResult` function to the protocol.

    Keeps a stateless experiment costing exactly one function to write.
    """

    def __init__(self, name: str, run: Callable[[bytes], ExperimentResult]) -> None:
        self.name = name
        self._run = run

    def load(self, settings: ExperimentSettings) -> None:
        return None

    def run(self, raw_bytes: bytes) -> ExperimentResult:
        return self._run(raw_bytes)

    def release(self) -> None:
        return None


# Import order here is deliberate, not incidental: the types above must be
# defined before the submodules that import them. By the time this runs,
# this (partially-initialized) module already has them set as attributes,
# so the circular import resolves. Moving these imports above the
# definitions breaks every submodule at import time.
from tower.experiments import (  # noqa: E402
    baseline,
    depth,
    edge_detection,
    feature_detection,
    frame_quality,
    object_detection,
    optical_flow,
    redaction_impact,
)

# `depth.py` is under review in another lane and cannot be edited in this
# change, so its declaration sits here rather than beside the metrics it
# describes, where the other seven live. It belongs in `depth.py` and
# moves there the moment that file is writable again. Nothing depends on
# the location: `_REGISTRY` below cannot hold a factory without a
# declaration either way.
_DEPTH_METRIC_KINDS: dict[str, MetricKind] = {
    "mean_relative_depth": MetricKind.RATE,
    "min_relative_depth": MetricKind.RATE,
    "max_relative_depth": MetricKind.RATE,
    "std_relative_depth": MetricKind.RATE,
}


@dataclass(frozen=True)
class ExperimentRegistration:
    """A factory and, inseparably, what its numbers mean and are called.

    One record rather than three parallel dicts, so that registering an
    experiment without classifying its metrics -- or without saying what
    a person should call it and whether it measured or inferred -- is not
    a thing that can be done and then forgotten. It is a missing
    positional argument.

    `metadata` joined `metric_kinds` when the Lab stopped being selected
    by an environment variable. An experiment nobody can name is fine
    when the only way to pick one is to type its registry key into a
    shell; it is not fine when a phone lists them.
    """

    factory: Callable[[], Experiment]
    metric_kinds: Mapping[str, MetricKind]
    metadata: ExperimentMetadata


# name -> registration. A FACTORY, not an instance: constructing a
# detector at import time would load model weights in any process that so
# much as imports this module, including every unrelated test.
# Units, stated once each. Named rather than repeated so that two
# experiments reporting the same kind of quantity cannot drift into
# calling it two different things on the wire.
_FRACTION = "fraction"
_PIXELS = "px"
_LEVEL = "level"
_KEYPOINTS = "keypoints"
_SECONDS = "s"
_DEGREES = "deg"

_REGISTRY: dict[str, ExperimentRegistration] = {
    "baseline": ExperimentRegistration(
        lambda: StatelessExperiment("baseline", baseline.run),
        baseline.METRIC_KINDS,
        ExperimentMetadata(
            name="Baseline",
            headline_label="mean_intensity",
            backend="opencv",
            summary=(
                "Mean grayscale intensity of the frame. The cheapest thing "
                "that proves the whole glasses -> phone -> Tower -> CV path "
                "is alive."
            ),
            provenance=PROVENANCE_MEASURED,
            stateful=False,
            requires_model=False,
            # 8-bit intensity level, 0-255, not a fraction. It is
            # `gray.mean()` and nothing normalises it.
            headline_unit=_LEVEL,
            # NO PREVIEW, and it is the only registered experiment without
            # one. This is the control: the cheapest thing that proves the
            # whole path is alive, and the figure every other experiment's
            # cost is read against. A line drawing costs about 0.35 ms,
            # which is roughly what this entire experiment costs -- so
            # giving it a picture would double the number it exists to
            # produce and quietly destroy the baseline. `preview_kind`
            # stays `None` on purpose and a test says so.
        ),
    ),
    "edge_detection": ExperimentRegistration(
        # A class rather than `StatelessExperiment`, and NOT because it
        # became stateful: `stateful` below is still False, because that
        # field means "its answer depends on what came before it" and
        # this one's still does not. What it gained is a place to put the
        # `edges` array down. A `bytes -> ExperimentResult` function has
        # no `self`, and `ExperimentResult` is a measurement channel that
        # deliberately carries floats only.
        edge_detection.EdgeDetection,
        edge_detection.METRIC_KINDS,
        ExperimentMetadata(
            name="Edge detection",
            headline_label="edge_density",
            backend="opencv",
            summary=(
                "Fraction of pixels Canny calls an edge, after a Gaussian "
                "blur. Moves visibly between a blank wall and a cluttered "
                "desk."
            ),
            provenance=PROVENANCE_MEASURED,
            stateful=False,
            requires_model=False,
            headline_unit=_FRACTION,
            # The Canny output itself, which is the picture a person
            # actually wanted when they read `edge_density: 0.071` and
            # could not tell whether that was a desk or a wall.
            preview_kind=PREVIEW_KIND_EDGE_MAP,
        ),
    ),
    "frame_quality": ExperimentRegistration(
        frame_quality.FrameQuality,
        frame_quality.METRIC_KINDS,
        ExperimentMetadata(
            name="Frame quality",
            headline_label="sharpness_laplacian_var",
            backend="opencv",
            summary=(
                "Six usability signals from one decode -- sharpness, "
                "gradient energy, entropy, contrast, edge density and "
                "clipping -- so a threshold gets chosen from a "
                "distribution rather than from taste."
            ),
            provenance=PROVENANCE_MEASURED,
            stateful=False,
            requires_model=False,
            # The frame's structure and the histogram the exposure figures
            # were counted off, with the clipping levels marked. No
            # verdict: "a threshold gets chosen from a distribution rather
            # than from taste" is this experiment's own summary, and a
            # picture saying BLURRY would be exactly the taste it refuses.
            preview_kind=PREVIEW_KIND_FRAME_QUALITY,
            # Variance of a Laplacian over 8-bit levels. Squared levels is
            # what it is; a nicer word would name a quantity nobody could
            # reproduce.
            headline_unit="level^2",
            metric_units={
                "sharpness_laplacian_var": "level^2",
                "gradient_energy": _LEVEL,
                "entropy_bits": "bits",
                "contrast_std": _LEVEL,
                "edge_density": _FRACTION,
                "overexposed_fraction": _FRACTION,
                "underexposed_fraction": _FRACTION,
                "width": _PIXELS,
                "height": _PIXELS,
            },
        ),
    ),
    "feature_detection": ExperimentRegistration(
        feature_detection.FeatureDetection,
        feature_detection.METRIC_KINDS,
        ExperimentMetadata(
            name="Feature detection",
            headline_label="keypoint_count",
            backend="opencv",
            summary=(
                "How much trackable ORB texture a frame contains, and how "
                "evenly it is spread. A thousand keypoints in one corner is "
                "worse for geometry than three hundred across the view."
            ),
            provenance=PROVENANCE_MEASURED,
            stateful=False,
            requires_model=False,
            headline_unit=_KEYPOINTS,
            metric_units={
                "keypoint_count": _KEYPOINTS,
                "descriptor_count": "descriptors",
                "spatial_coverage": _FRACTION,
                # `mean_response` is deliberately absent: ORB's response
                # is a corner score on an arbitrary scale, and a bare
                # number is the honest rendering of it.
                "mean_keypoint_size": _PIXELS,
                "requested_features": _KEYPOINTS,
            },
            # The keypoints themselves, thinned, over a line drawing and
            # the coverage grid. "A thousand keypoints in one corner is
            # worse than three hundred across the view" is this
            # experiment's whole argument, and until now it was two
            # numbers a person had to hold in their head at once.
            preview_kind=PREVIEW_KIND_KEYPOINTS,
        ),
    ),
    "redaction_impact": ExperimentRegistration(
        redaction_impact.RedactionImpact,
        redaction_impact.METRIC_KINDS,
        ExperimentMetadata(
            name="Redaction impact",
            headline_label="region_keypoint_retention",
            backend="opencv",
            summary=(
                "What blurring a region costs the geometry that runs after "
                "it: how many keypoints survive inside the blur, and how "
                "many survivors sit on its boundary rather than on the "
                "scene."
            ),
            provenance=PROVENANCE_MEASURED,
            stateful=False,
            requires_model=False,
            headline_unit=_FRACTION,
            metric_units={
                "region_keypoint_retention": _FRACTION,
                "frame_keypoint_retention": _FRACTION,
                "keypoints_before": _KEYPOINTS,
                "keypoints_after": _KEYPOINTS,
                "keypoints_lost": _KEYPOINTS,
                "keypoints_in_region_before": _KEYPOINTS,
                "keypoints_in_region_after": _KEYPOINTS,
                "survivors_near_region": _KEYPOINTS,
                "survivors_on_boundary": _KEYPOINTS,
                "boundary_fraction": _FRACTION,
                "region_area_fraction": _FRACTION,
                "blur_kernel": _PIXELS,
            },
            # The blurred rectangle and what ORB could still see through
            # it, over a line drawing of the ALREADY-BLURRED frame. The
            # one preview whose subject is the privacy machinery, and the
            # only place `boundary_fraction: 0.31` becomes a picture of
            # keypoints sitting on the edge of the blur rather than on
            # anything in the room.
            preview_kind=PREVIEW_KIND_REDACTION,
        ),
    ),
    "optical_flow": ExperimentRegistration(
        optical_flow.OpticalFlowExperiment,
        optical_flow.METRIC_KINDS,
        ExperimentMetadata(
            name="Optical flow",
            headline_label="median_flow_px",
            backend="opencv",
            summary=(
                "How much the scene is moving and how coherently, from "
                "sparse Lucas-Kanade with a forward-backward check. The "
                "first frame of a session has no answer and says so."
            ),
            provenance=PROVENANCE_MEASURED,
            stateful=True,
            requires_model=False,
            headline_unit=_PIXELS,
            metric_units={
                "median_flow_px": _PIXELS,
                "mean_flow_px": _PIXELS,
                "max_flow_px": _PIXELS,
                "tracked_fraction": _FRACTION,
                "tracked_count": _KEYPOINTS,
                "seeded_count": _KEYPOINTS,
                "median_forward_backward_px": _PIXELS,
                "rejected_by_forward_backward": _KEYPOINTS,
                "direction_coherence": _FRACTION,
                "dominant_direction_deg": _DEGREES,
                # The three flags are 0/1 per frame and COUNT-kind, so
                # their corpus total is a number of frames.
                "has_reference": "frames",
                "resolution_changed": "frames",
                "reference_stale": "frames",
                "seconds_since_reference": _SECONDS,
            },
            # One arrow per tracked point, coloured by direction, with the
            # forward-backward rejects marked. `direction_coherence: 0.94`
            # and a picture of every arrow pointing the same way are the
            # same fact, and only one of them can be checked at a glance.
            preview_kind=PREVIEW_KIND_FLOW_TRACKS,
        ),
    ),
    "object_detection": ExperimentRegistration(
        object_detection.ObjectDetectionExperiment,
        object_detection.METRIC_KINDS,
        ExperimentMetadata(
            name="Object detection",
            headline_label="detections",
            backend="torch",
            summary=(
                "COCO classes from SSDLite/MobileNetV3, reported with the "
                "score they cleared. Evidence that something scored above a "
                "threshold -- not a statement that an object is there."
            ),
            # Model inference. object_detection.py's own header says so in
            # as many words, and Rule 16 / Core Principle 2 require that to
            # travel with the number rather than sit in a docstring.
            provenance=PROVENANCE_INFERRED,
            stateful=True,
            requires_model=True,
            headline_unit="detections",
            metric_units={
                "detections": "detections",
                "raw_detections": "detections",
                "score_threshold": "score",
                "mean_score": "score",
                "max_score": "score",
                **{
                    "count_" + name.replace(" ", "_"): "detections"
                    for name in object_detection.TRACKED_CLASSES
                },
            },
            annotation_metric="detections",
            # Boxes, class names and scores, including the ones below the
            # threshold. Physical testing reported 160 `person` detections
            # in a room with nobody in it; the numbers could not say
            # whether that was a broken class map or a head-mounted camera
            # looking at its wearer's own hands, and the box says it in
            # one glance. See `ObjectDetectionExperiment._preview_payload`.
            preview_kind=PREVIEW_KIND_DETECTIONS,
        ),
    ),
    "depth": ExperimentRegistration(
        depth.DepthEstimation,
        _DEPTH_METRIC_KINDS,
        ExperimentMetadata(
            name="Monocular depth",
            headline_label="mean_relative_depth",
            backend="torch",
            summary=(
                "Relative inverse depth from MiDaS-small. NOT metric "
                "distance: the model does not produce one, and no figure "
                "here may be read as metres."
            ),
            provenance=PROVENANCE_INFERRED,
            stateful=True,
            requires_model=True,
            # Deliberately unitless. IOS-to-Tower.md 0.5: "metric is not
            # metres". MiDaS-small emits relative inverse depth on an
            # arbitrary scale, so a bare number is the honest rendering
            # and any unit string here would be a claim about scale.
            headline_unit=None,
            # The same arbitrary scale, drawn. The preview is the one
            # place the unitlessness stops being a problem: nobody reads
            # a colour as a distance, and near-versus-far is exactly what
            # relative inverse depth is good for.
            preview_kind=PREVIEW_KIND_RELATIVE_DEPTH,
        ),
    ),
}

# The long-standing public shape, derived rather than duplicated: every
# caller that only wants "name -> factory" keeps working, and the two
# cannot drift apart because there is only one list.
EXPERIMENTS: dict[str, Callable[[], Experiment]] = {
    name: registration.factory for name, registration in _REGISTRY.items()
}


def experiment_metadata(experiment_name: str) -> ExperimentMetadata:
    """What to call this experiment and what its numbers are. Or KeyError.

    Separate from `metric_kinds` because the two answer different
    questions and a caller usually wants one of them, but they come from
    the same record and therefore cannot disagree about which experiments
    exist.
    """
    return _REGISTRY[experiment_name].metadata


def metric_kinds(experiment_name: str) -> Mapping[str, MetricKind]:
    """What every metric this experiment emits means. KeyError names a
    registry miss, which is a different bug from an unclassified metric
    and is therefore a different exception."""
    return _REGISTRY[experiment_name].metric_kinds


def classify_metric(experiment_name: str, metric_name: str) -> MetricKind:
    """How to combine one metric across frames, or raise.

    There is no default. See `UnclassifiedMetricError`.
    """
    kinds = metric_kinds(experiment_name)
    try:
        return kinds[metric_name]
    except KeyError:
        raise UnclassifiedMetricError(
            f"{experiment_name} emitted {metric_name!r}, which it never "
            f"classified. Add it to METRIC_KINDS in the experiment as one "
            f"of {', '.join(k.name for k in MetricKind)} -- the experiment "
            f"is the only thing that knows which. Classified: "
            f"{sorted(kinds)}"
        ) from None
