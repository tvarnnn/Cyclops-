"""Persisted record shapes for World Builder.

Follows tower/object_memory/records.py: frozen dataclasses, explicit
to_json_dict / *_from_json_dict, reserved-but-unused fields carried from
day one so a later need does not require rewriting already-persisted
records, and confidence stored as a LABEL rather than a score.

Confidence itself is imported from object_memory rather than redefined.
Two independently-evolving confidence vocabularies with the same four
labels is exactly the silent-divergence failure Rule 16 exists to prevent.
Rule 6 ("modules own their data") governs data, not a shared vocabulary --
importing a frozen enum is not promoting data to a shared service. Revisit
a tower/confidence.py only when a third consumer appears.
"""

import math
from dataclasses import dataclass, field, replace

from tower.confidence import Confidence
from tower.world_builder.schema import (
    DEGENERACY_NONE,
    INTRINSICS_SOURCE_SELF_CALIBRATED,
    INTRINSICS_SOURCE_UNKNOWN,
    POSE_CONVENTION,
    POSE_STATUS_UNAVAILABLE,
    SCALE_STATES_ALLOWING_METRES,
    SCALE_UNKNOWN,
    SCHEMA_VERSION,
    TIME_BASIS,
)


# Opaque ids: never derived from a display name (the user renames
# "Bedroom") and never a content hash (refinement changes content). Object
# Memory anchors will reference a world_id long after either has changed.
from tower.storage import new_id  # noqa: E402,F401


@dataclass(frozen=True)
class ScaleState:
    """How, if at all, this world's units relate to physical distance.

    Geometry is ALWAYS stored in world units and scale is metadata, never
    baked into coordinates. Applying a later metric estimate is then a
    one-field write instead of a rewrite of every point, pose, and anchor
    -- and it does not invalidate anchors written earlier.

    "relative" is the honest state for monocular reconstruction: the
    reconstruction is internally consistent, but its unit is whatever the
    first solved baseline happened to be. That is not a metre.
    """

    state: str = SCALE_UNKNOWN
    meters_per_unit: float | None = None
    method: str | None = None
    confidence: Confidence = Confidence.UNKNOWN
    # Superseded estimates are appended, never overwritten, so a consumer
    # that cached a distance can detect that it is stale.
    history: tuple[dict, ...] = ()

    @property
    def allows_metres(self) -> bool:
        return self.state in SCALE_STATES_ALLOWING_METRES

    def to_json_dict(self) -> dict:
        return {
            "state": self.state,
            "meters_per_unit": self.meters_per_unit,
            "method": self.method,
            "confidence": self.confidence.value,
            "history": [dict(entry) for entry in self.history],
        }


def scale_state_from_json_dict(data: dict) -> ScaleState:
    return ScaleState(
        state=data["state"],
        meters_per_unit=data["meters_per_unit"],
        method=data["method"],
        confidence=Confidence(data["confidence"]),
        history=tuple(dict(entry) for entry in data["history"]),
    )


def format_distance(value: float, scale: ScaleState) -> str:
    """Render a distance without ever claiming a unit we cannot support.

    The single choke point for turning a number into human-facing text.
    It exists as a function rather than a convention because "remember not
    to print metres" is not a mechanism, and a metre printed from an
    unscaled monocular reconstruction is precisely the
    inference-presented-as-measurement that Rule 16 forbids.
    """
    if scale.allows_metres and scale.meters_per_unit is not None:
        return f"{value * scale.meters_per_unit:.3f} m"
    return f"{value:.3f} world units"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics plus, crucially, where they came from.

    A world whose keyframes have no record of the camera configuration
    that produced them can never be re-projected, relocalised into, or
    re-run through a better pipeline -- it is permanently unrefinable.
    That is why this is authoritative rather than derived.

    Keyed by resolution deliberately: DAT's adaptive ladder changes
    resolution mid-stream and cannot be overridden, and whether intrinsics
    scale linearly across its three modes is genuinely unknown (recorded
    as scales_linearly_across_resolutions=None, not assumed True). A
    single unkeyed block would silently mis-attribute every frame captured
    after a ladder step.

    There is no constructor that produces a guessed value.
    """

    source: str = INTRINSICS_SOURCE_UNKNOWN
    model: str | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    dist_coeffs: tuple[float, ...] | None = None
    calibrated_width: int | None = None
    calibrated_height: int | None = None
    reprojection_rms_px: float | None = None
    view_count: int | None = None
    calibrated_at: float | None = None
    scales_linearly_across_resolutions: bool | None = None

    @classmethod
    def unknown(cls) -> "CameraIntrinsics":
        return cls()

    @property
    def is_known(self) -> bool:
        """Known AND physically possible.

        A presence check alone is not enough. fx=0, fx=-500 and fx=NaN all
        satisfy "is not None", all report is_known, and all route straight
        to the calibrated backend, which then builds a confident
        reconstruction from an impossible camera.

        Focal lengths must be finite and positive. The principal point
        need only be finite -- it can legitimately fall outside the image
        on a cropped or off-centre sensor.
        """
        import math

        if self.source == INTRINSICS_SOURCE_UNKNOWN:
            return False
        if None in (self.fx, self.fy, self.cx, self.cy):
            return False
        if not all(
            math.isfinite(value)
            for value in (self.fx, self.fy, self.cx, self.cy)
        ):
            return False
        return self.fx > 0.0 and self.fy > 0.0

    def camera_matrix(self):
        """3x3 K, or None when intrinsics are unknown.

        Returns None rather than raising so callers are pushed toward
        "do I have intrinsics?" rather than toward a try/except that
        would be easy to paper over with a default.
        """
        if not self.is_known:
            return None
        import numpy as np

        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Rescale to another resolution, ONLY when that is known to be valid.

        Refuses unless scales_linearly_across_resolutions is explicitly
        True. The linear-scaling assumption holds for a pure resize and
        fails for a crop, and which one DAT performs between its three
        modes has never been established -- so the default is to refuse
        rather than to silently produce intrinsics that are wrong by a
        crop factor.
        """
        if not self.is_known:
            return self
        if self.calibrated_width == width and self.calibrated_height == height:
            return self
        if self.scales_linearly_across_resolutions is not True:
            raise ValueError(
                "refusing to rescale intrinsics calibrated at "
                f"{self.calibrated_width}x{self.calibrated_height} to "
                f"{width}x{height}: scales_linearly_across_resolutions is "
                f"{self.scales_linearly_across_resolutions!r}, not True. "
                "Whether DAT resizes or crops between resolutions is not "
                "established; calibrate at the target resolution instead."
            )
        sx = width / self.calibrated_width
        sy = height / self.calibrated_height
        return replace(
            self,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            calibrated_width=width,
            calibrated_height=height,
        )

    def to_json_dict(self) -> dict:
        return {
            "source": self.source,
            "model": self.model,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "dist_coeffs": (
                list(self.dist_coeffs) if self.dist_coeffs is not None else None
            ),
            "calibrated_width": self.calibrated_width,
            "calibrated_height": self.calibrated_height,
            "reprojection_rms_px": self.reprojection_rms_px,
            "view_count": self.view_count,
            "calibrated_at": self.calibrated_at,
            "scales_linearly_across_resolutions": (
                self.scales_linearly_across_resolutions
            ),
        }


def camera_intrinsics_from_json_dict(data: dict) -> CameraIntrinsics:
    coeffs = data["dist_coeffs"]
    source = data["source"]
    if source == INTRINSICS_SOURCE_SELF_CALIBRATED and data["fx"] is None:
        raise ValueError(
            "intrinsics claim source=self_calibrated but carry no fx; "
            "a calibration that produced no focal length is not a calibration"
        )
    return CameraIntrinsics(
        source=source,
        model=data["model"],
        fx=data["fx"],
        fy=data["fy"],
        cx=data["cx"],
        cy=data["cy"],
        dist_coeffs=tuple(coeffs) if coeffs is not None else None,
        calibrated_width=data["calibrated_width"],
        calibrated_height=data["calibrated_height"],
        reprojection_rms_px=data["reprojection_rms_px"],
        view_count=data["view_count"],
        calibrated_at=data["calibrated_at"],
        scales_linearly_across_resolutions=data[
            "scales_linearly_across_resolutions"
        ],
    )


def _check_vector(values, length: int, name: str) -> None:
    """A transform component that is the wrong shape, or carries a NaN, is
    still applied by a renderer -- it just puts the geometry nowhere in
    particular. Checked here because this record is the last place that
    can refuse it before it reaches a screen."""
    if values is None or len(values) != length:
        raise ValueError(
            f"{name} must have {length} components, got "
            f"{None if values is None else len(values)}"
        )
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numbers, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite, got {value!r}; a non-finite "
                "transform is applied all the same and places the geometry "
                "nowhere in particular"
            )


# How far a stored quaternion may drift from unit length.
#
# A norm off by t scales the geometry it rotates by t, on top of `scale`,
# so this is a bound on silent scale error: 1e-3 is 0.1%, below anything
# the reconstruction can resolve.
#
# NOT tighter than that. A first attempt used 1e-6 and rejected real
# placements read back from disk: a quaternion serialised at five decimal
# places lands 1.9e-6 off unit, so the bar was tighter than the precision
# of the data it was judging. A validator that refuses its own valid
# output is worse than no validator, because it fails silently -- the
# rows are dropped and the world reads as unregistered.
QUATERNION_NORM_TOLERANCE = 1e-3

PLACEMENT_REGISTERED = "registered"
PLACEMENT_REFUSED = "refused"
PLACEMENT_STATES = (PLACEMENT_REGISTERED, PLACEMENT_REFUSED)


@dataclass(frozen=True)
class SegmentPlacement:
    """Where one segment sits in a shared world frame, or why it does not.

    Segment geometry is immutable once solved; WHERE a segment sits is
    not, and a later pass may place a segment that an earlier one could
    not. Keeping placement in its own record is what lets a placement
    change without the geometry changing -- and lets a refusal carry its
    reason instead of being indistinguishable from "nobody tried".

    `refused` is a first-class state for that reason. `registered: false`
    alone conflated "we tried and the two independent solves disagreed"
    with "no attempt has been made", and the refusal is the more
    interesting fact: on the real corpus most pairs are refused because
    the wearer stood still, which is a message about how to walk, not
    about the software.

    The invariants below are enforced rather than documented because each
    represents geometry that would be DRAWN. A refusal carrying a
    transform gets rendered; a registered placement missing its scale gets
    a default from somewhere, and that somewhere is wrong; a zero or
    non-finite scale collapses a segment to a dot at another's origin,
    which the registration research records as invisible in every
    aggregate metric.
    """

    segment_index: int
    state: str
    rotation_wxyz: tuple | None
    translation: tuple | None
    scale: float | None
    reference_segment: int | None
    refusal_reason: str | None
    evidence: dict
    # The build this placement was solved against.
    #
    # A placement is a statement about SPECIFIC points. Rewriting
    # poses.json and points.json does not touch placements.json, so
    # without this a Sim3 survives a full rebuild and is served against
    # geometry that no longer exists -- still flagged registered. The
    # cache guarantee does not rescue that: content_hash moves, the
    # client dutifully refetches, and is handed a transform solved
    # against points that are gone.
    #
    # None means "not bound", which cannot be verified and is therefore
    # not served. An unverifiable transform is the failure this exists to
    # prevent.
    input_digest: str | None = None
    # Which gauge this Sim3 is expressed in. A coordinate stamped with one
    # frame revision may not be silently reinterpreted under another.
    frame_revision: int = 1
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.state not in PLACEMENT_STATES:
            raise ValueError(
                f"unknown placement state {self.state!r}; the set is closed "
                f"because consumers switch on it: {PLACEMENT_STATES}"
            )
        has_transform = (
            self.rotation_wxyz is not None
            or self.translation is not None
            or self.scale is not None
        )
        if self.state == PLACEMENT_REFUSED and has_transform:
            raise ValueError(
                "a refused placement must not carry a transform: anything "
                "with a transform gets drawn"
            )
        if self.state == PLACEMENT_REGISTERED:
            if (
                self.rotation_wxyz is None
                or self.translation is None
                or self.scale is None
            ):
                raise ValueError(
                    "a registered placement needs a complete Sim3; half a "
                    "transform means a default supplies the rest, and the "
                    "default is wrong"
                )
            _check_vector(self.rotation_wxyz, 4, "rotation_wxyz")
            _check_vector(self.translation, 3, "translation")
            norm = math.sqrt(sum(v * v for v in self.rotation_wxyz))
            if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
                raise ValueError(
                    f"rotation_wxyz must be a unit quaternion, got norm "
                    f"{norm!r}; a non-unit quaternion scales the geometry it "
                    "rotates, silently and on top of `scale`"
                )
            if isinstance(self.scale, bool) or not isinstance(
                self.scale, (int, float)
            ):
                raise ValueError(
                    f"scale must be a real number, got {self.scale!r}"
                )
            if self.refusal_reason is not None:
                raise ValueError(
                    "a registered placement must not carry a refusal reason; "
                    "the wire drops it silently, so it would be a field that "
                    "looks meaningful and is not"
                )
            if not math.isfinite(self.scale) or self.scale <= 0:
                raise ValueError(
                    f"scale must be finite and positive, got {self.scale!r}; "
                    "a zero scale collapses the segment to a dot at the "
                    "reference's origin and is invisible in every aggregate"
                )
        if (
            isinstance(self.segment_index, bool)
            or not isinstance(self.segment_index, int)
            or self.segment_index < 0
        ):
            raise ValueError(
                f"segment_index must be a non-negative int, got "
                f"{self.segment_index!r}"
            )
        if (
            isinstance(self.frame_revision, bool)
            or not isinstance(self.frame_revision, int)
            or self.frame_revision < 1
        ):
            raise ValueError(
                f"frame_revision must be an int >= 1, got "
                f"{self.frame_revision!r}; the contract calls a revision "
                "mismatch a refuse-to-draw condition, so an impossible "
                "revision must not reach a client"
            )
        if self.state == PLACEMENT_REGISTERED:
            if (
                isinstance(self.reference_segment, bool)
                or not isinstance(self.reference_segment, int)
                or self.reference_segment < 0
            ):
                raise ValueError(
                    f"a registered placement needs a real reference_segment, "
                    f"got {self.reference_segment!r}; the composition rule is "
                    "defined only relative to one"
                )

    def to_json_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "segment_index": self.segment_index,
            "state": self.state,
            "rotation_wxyz": (
                list(self.rotation_wxyz) if self.rotation_wxyz else None
            ),
            "translation": (
                list(self.translation) if self.translation else None
            ),
            "scale": self.scale,
            "reference_segment": self.reference_segment,
            "refusal_reason": self.refusal_reason,
            "evidence": dict(self.evidence),
            "frame_revision": self.frame_revision,
            "input_digest": self.input_digest,
        }

    @classmethod
    def from_json_dict(cls, row: dict) -> "SegmentPlacement":
        rotation = row.get("rotation_wxyz")
        translation = row.get("translation")
        return cls(
            segment_index=int(row["segment_index"]),
            state=row["state"],
            rotation_wxyz=tuple(rotation) if rotation else None,
            translation=tuple(translation) if translation else None,
            scale=row.get("scale"),
            reference_segment=row.get("reference_segment"),
            refusal_reason=row.get("refusal_reason"),
            evidence=dict(row.get("evidence") or {}),
            frame_revision=int(row.get("frame_revision", 1)),
            input_digest=row.get("input_digest"),
            schema_version=int(row.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class World:
    """The long-lived artifact. Not a session, not a connection."""

    world_id: str
    created_at: float
    updated_at: float
    display_name: str | None = None
    schema_version: int = SCHEMA_VERSION
    # Monotonic gauge counter. V1 never advances it past 1 -- nothing
    # performs loop closure, bundle adjustment, or scale estimation. It is
    # persisted anyway because it is the un-retrofittable half of the
    # gauge design: a coordinate written without a revision stamp can
    # never be repaired, since you cannot determine after the fact which
    # gauge it meant.
    frame_revision: int = 1
    pose_convention: dict = field(default_factory=lambda: dict(POSE_CONVENTION))
    scale: ScaleState = field(default_factory=ScaleState)
    session_ids: tuple[str, ...] = ()
    # True once keyframe imagery has been purged for privacy. The world
    # survives as a record, but a rebuild must fail loudly rather than
    # quietly producing an empty reconstruction.
    images_purged: bool = False

    def to_json_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "world_id": self.world_id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "frame_revision": self.frame_revision,
            "pose_convention": dict(self.pose_convention),
            "scale": self.scale.to_json_dict(),
            "session_ids": list(self.session_ids),
            "images_purged": self.images_purged,
        }


def world_from_json_dict(data: dict) -> World:
    return World(
        schema_version=data["schema_version"],
        world_id=data["world_id"],
        display_name=data["display_name"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        frame_revision=data["frame_revision"],
        pose_convention=dict(data["pose_convention"]),
        scale=scale_state_from_json_dict(data["scale"]),
        session_ids=tuple(data["session_ids"]),
        images_purged=data["images_purged"],
    )


# How a session's finalization went. Recorded by the builder on the session
# record after `stop_session`, because until 2026-09-06 nothing on disk could
# tell "the builder finished", "the builder is still finishing" and "the
# builder was killed while finishing" apart -- the 09-06 physical walk ended
# with a lock naming a dead pid, a record with `ended_at: null`, and 463
# keyframes of usable geometry that the Tower could only call `failed`.
FINALIZATION_PENDING = "pending"
FINALIZATION_COMPLETE = "complete"
FINALIZATION_INTERRUPTED = "interrupted"
FINALIZATION_STATES = (
    FINALIZATION_PENDING,
    FINALIZATION_COMPLETE,
    FINALIZATION_INTERRUPTED,
)

# The stages that run AFTER finalization, and how each one went.
#
# `finalization` answers "did the builder finish?". It was answering yes
# several minutes before the world the wearer walked for existed: the builder
# marks finalization `complete` and releases the world lock in its `finally`,
# and only THEN runs the surface, the appearance and the dense stages, which
# take six to eleven minutes on a real walk. The release is deliberate -- the
# phone reads the world while they build -- but until 2026-09-22 their outcome
# lived only in a report dict on the child's stdout that nobody persists. A
# `surfacify()` that raised left a world that was sparse forever, beside a
# record proudly saying `complete`, with no way for an operator or a client to
# tell "never attempted" from "failed" from "not applicable".
#
# So the stages get their own record. It does NOT move a single call: the
# sequencing is right, the record was not.
STAGE_SURFACE = "surface"
STAGE_APPEARANCE = "appearance"
STAGE_DENSE = "dense"
POST_FINALIZATION_STAGES = (STAGE_SURFACE, STAGE_APPEARANCE, STAGE_DENSE)

# The five words a stage outcome can be. NOT A NEW VOCABULARY: these are the
# ones `surface_pipeline.py` already writes into every `status.json`, which
# `dense_pipeline` and `appearance_pipeline` already share, moved here so
# there is one definition rather than two sets of five strings that agree on
# the day they were written and drift afterwards -- the same reasoning this
# module's docstring gives for importing `Confidence` instead of redefining
# it. `surface_pipeline` re-exports them under its own names, so every
# existing importer is untouched.
#
# The definition lives HERE rather than there because this module is the
# light one: `records` is imported by the store, the engine and the result
# channel, and `surface_pipeline` pulls in the whole reconstruction stack.
#
# The practical payoff is that a `SurfaceResult.state`, an `AppearanceResult
# .state` and a `DenseResult.state` can be recorded verbatim, with no
# translation table to fall out of step with its sources.
STAGE_STATE_OK = "ok"
STAGE_STATE_RUNNING = "running"
STAGE_STATE_FAILED = "failed"
STAGE_STATE_STOPPED = "stopped"
STAGE_STATE_UNAVAILABLE = "unavailable"
STAGE_STATES = (
    STAGE_STATE_OK,
    STAGE_STATE_RUNNING,
    STAGE_STATE_FAILED,
    STAGE_STATE_STOPPED,
    STAGE_STATE_UNAVAILABLE,
)

# What became of the final global solve inside that finalization.
FINAL_SOLVE_PENDING = "pending"
FINAL_SOLVE_SOLVED = "solved"
FINAL_SOLVE_SKIPPED = "skipped"
FINAL_SOLVE_FAILED = "failed"
FINAL_SOLVE_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Session:
    """One mapping session: an explicitly started and stopped window."""

    session_id: str
    world_id: str
    started_at: float
    schema_version: int = SCHEMA_VERSION
    ended_at: float | None = None
    end_reason: str | None = None
    time_basis: str = TIME_BASIS
    frame_source: str = "unknown"
    capture_id: str | None = None
    declared_width: int | None = None
    declared_height: int | None = None
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics.unknown)
    backend_id: str | None = None
    backend_requires_intrinsics: bool | None = None
    backend_downgraded_from: str | None = None
    backend_downgrade_reason: str | None = None
    frames_observed: int = 0
    keyframes_accepted: int = 0
    rejected_by_reason: dict = field(default_factory=dict)
    # This module retains raw imagery, unlike every module shipped so far.
    # Declared on the record itself, not only on the module descriptor, so
    # the posture travels with the data.
    retains_raw_imagery: bool = True
    privacy_tags: tuple[str, ...] = ("raw-imagery", "first-person")
    # What, if anything, was removed from the imagery before it was
    # written. "none" is the honest V1 value: no redaction is implemented.
    #
    # This field exists now because it is the one genuinely
    # un-retrofittable part of any future redaction work. Without it, the
    # day redaction ships there is no way to tell whether an older
    # session's images were filtered, so every historical session must be
    # assumed unredacted forever. One string now avoids that permanently.
    redaction: str = "none"
    # The finalization record (see FINALIZATION_*), or None on a record that
    # predates it or whose session never stopped. Additive and optional:
    # the schema version does not move, and a reader that finds no key
    # reads None, never a state.
    #
    #   {"state": pending|complete|interrupted,
    #    "final_solve": pending|solved|skipped|failed|unavailable|null,
    #    "started_at": float, "updated_at": float, "detail": str|null}
    finalization: dict | None = None
    # What became of the stages that run AFTER finalization (see
    # POST_FINALIZATION_STAGES), or None on a record that predates this or a
    # builder that never reached them. Additive and optional in exactly the
    # way `finalization` is: the schema version does not move, and a reader
    # that finds no key reads None -- "never recorded", which is a different
    # fact from any state in the vocabulary.
    #
    #   {"surface"|"appearance"|"dense":
    #      {"attempted": bool,
    #       "state": ok|running|failed|stopped|unavailable,
    #       "started_at": float, "updated_at": float, "detail": str|null}}
    #
    # `running` left behind by a process that is gone is not a bug in the
    # record; it is the only honest thing to say about a builder the Job
    # Object killed thirty seconds into a six-minute surface.
    stages: dict | None = None

    def to_json_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "world_id": self.world_id,
            "capture_id": self.capture_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "end_reason": self.end_reason,
            "time_basis": self.time_basis,
            "frame_source": self.frame_source,
            "declared_width": self.declared_width,
            "declared_height": self.declared_height,
            "intrinsics": self.intrinsics.to_json_dict(),
            "backend_id": self.backend_id,
            "backend_requires_intrinsics": self.backend_requires_intrinsics,
            "backend_downgraded_from": self.backend_downgraded_from,
            "backend_downgrade_reason": self.backend_downgrade_reason,
            "frames_observed": self.frames_observed,
            "keyframes_accepted": self.keyframes_accepted,
            "rejected_by_reason": dict(self.rejected_by_reason),
            "retains_raw_imagery": self.retains_raw_imagery,
            "privacy_tags": list(self.privacy_tags),
            "redaction": self.redaction,
            "finalization": (
                dict(self.finalization) if self.finalization is not None else None
            ),
            # Copied one level DEEPER than `finalization` needs to be,
            # because this one is a dict of dicts: a shallow copy would hand
            # every caller a live reference to the per-stage blocks of a
            # frozen record.
            "stages": (
                {stage: dict(entry) for stage, entry in self.stages.items()}
                if self.stages is not None
                else None
            ),
        }


def session_from_json_dict(data: dict) -> Session:
    return Session(
        schema_version=data["schema_version"],
        session_id=data["session_id"],
        world_id=data["world_id"],
        capture_id=data["capture_id"],
        started_at=data["started_at"],
        ended_at=data["ended_at"],
        end_reason=data["end_reason"],
        time_basis=data["time_basis"],
        frame_source=data["frame_source"],
        declared_width=data["declared_width"],
        declared_height=data["declared_height"],
        intrinsics=camera_intrinsics_from_json_dict(data["intrinsics"]),
        backend_id=data["backend_id"],
        backend_requires_intrinsics=data["backend_requires_intrinsics"],
        backend_downgraded_from=data["backend_downgraded_from"],
        backend_downgrade_reason=data["backend_downgrade_reason"],
        frames_observed=data["frames_observed"],
        keyframes_accepted=data["keyframes_accepted"],
        rejected_by_reason=dict(data["rejected_by_reason"]),
        retains_raw_imagery=data["retains_raw_imagery"],
        privacy_tags=tuple(data["privacy_tags"]),
        redaction=data["redaction"],
        # `.get`, deliberately: every record written before 2026-09-06 lacks
        # the key, and those records are still valid sessions.
        finalization=(
            dict(data["finalization"])
            if isinstance(data.get("finalization"), dict)
            else None
        ),
        # `.get` and an isinstance check, for the same reason: every record
        # written before 2026-09-22 lacks the key, and those are still valid
        # sessions. Absent -- or any shape that is not an object -- reads
        # None, never a state.
        stages=(
            {
                stage: dict(entry)
                for stage, entry in data["stages"].items()
                if isinstance(entry, dict)
            }
            if isinstance(data.get("stages"), dict)
            else None
        ),
    )


@dataclass(frozen=True)
class Keyframe:
    """One selected observation. Deliberately pose-free.

    Poses are DERIVED -- they change every time the mapper changes, and
    the first thing that will happen to a V1 monocular mapper is that it
    changes. Keeping poses out of this record is what makes a mapper
    upgrade a reprocess rather than a data migration.

    The measured selection signals, by contrast, ARE authoritative: they
    were taken against candidate frames that were rejected and never
    persisted, so they can never be recomputed from the keyframe corpus.
    """

    keyframe_id: str
    session_id: str
    source_seq: int
    received_at: float
    image_relpath: str
    width: int
    height: int
    byte_count: int
    schema_version: int = SCHEMA_VERSION
    time_basis: str = TIME_BASIS
    wire_seq: int | None = None
    tx_seq: int | None = None
    # Segment, not "submap": a segment break means tracking was lost, so
    # poses either side are NOT in a common frame. Recording it is the
    # loop-closure foundation without the loop-closure machinery.
    segment_index: int = 0
    sharpness: float | None = None
    feature_count: int | None = None
    selection_reason: str = "unknown"
    median_parallax_px: float | None = None
    overlap_ratio: float | None = None
    survival_ratio: float | None = None
    tracked_count: int | None = None
    # The median distance, in PIXELS, between where a homography fitted to
    # the tracks predicts each point should land and where it actually
    # landed. Near zero means the motion carried no baseline.
    #
    # Deliberately NOT named r_h. KeyframeEdge.r_h is ORB-SLAM's
    # dimensionless H/(H+F) inlier ratio -- a different quantity, in
    # different units. Both once shared the name, which would have let a
    # successor compare them and get a meaningless answer with no error.
    # Renaming is free now and impossible once data exists.
    homography_residual_px: float | None = None
    quality: Confidence = Confidence.UNKNOWN
    frame_revision: int = 1
    # Reserved-but-unused, carried from day one so a later cross-module
    # need does not require rewriting persisted records.
    spatial_ref: None = None
    external_refs: tuple[()] = ()

    def to_json_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "keyframe_id": self.keyframe_id,
            "session_id": self.session_id,
            "source_seq": self.source_seq,
            "wire_seq": self.wire_seq,
            "tx_seq": self.tx_seq,
            "received_at": self.received_at,
            "time_basis": self.time_basis,
            "image_relpath": self.image_relpath,
            "width": self.width,
            "height": self.height,
            "byte_count": self.byte_count,
            "segment_index": self.segment_index,
            "sharpness": self.sharpness,
            "feature_count": self.feature_count,
            "selection_reason": self.selection_reason,
            "median_parallax_px": self.median_parallax_px,
            "overlap_ratio": self.overlap_ratio,
            "survival_ratio": self.survival_ratio,
            "tracked_count": self.tracked_count,
            "homography_residual_px": self.homography_residual_px,
            "quality": self.quality.value,
            "frame_revision": self.frame_revision,
            "spatial_ref": self.spatial_ref,
            "external_refs": list(self.external_refs),
        }


def keyframe_from_json_dict(data: dict) -> Keyframe:
    return Keyframe(
        schema_version=data["schema_version"],
        keyframe_id=data["keyframe_id"],
        session_id=data["session_id"],
        source_seq=data["source_seq"],
        wire_seq=data["wire_seq"],
        tx_seq=data["tx_seq"],
        received_at=data["received_at"],
        time_basis=data["time_basis"],
        image_relpath=data["image_relpath"],
        width=data["width"],
        height=data["height"],
        byte_count=data["byte_count"],
        segment_index=data["segment_index"],
        sharpness=data["sharpness"],
        feature_count=data["feature_count"],
        selection_reason=data["selection_reason"],
        median_parallax_px=data["median_parallax_px"],
        overlap_ratio=data["overlap_ratio"],
        survival_ratio=data["survival_ratio"],
        tracked_count=data["tracked_count"],
        homography_residual_px=data["homography_residual_px"],
        quality=Confidence(data["quality"]),
        frame_revision=data["frame_revision"],
        spatial_ref=None,
        external_refs=(),
    )


def make_keyframe_id(session_id: str, source_seq: int) -> str:
    """The globally stable frame key.

    (session_id, source_seq), never source_seq alone: source_seq is only
    monotonic within a session and resets when the glasses session
    restarts. Its deltas are also sampling, not elapsed time -- the sender
    decimates frames -- so nothing may derive a duration from them.
    """
    return f"{session_id}:{source_seq:08d}"


@dataclass(frozen=True)
class KeyframeEdge:
    """A measured pairwise relationship between two keyframes.

    Authoritative, not derived: these numbers were measured against
    candidate frames that were then discarded, so no later pass can
    reproduce them from the stored corpus.
    """

    from_keyframe_id: str
    to_keyframe_id: str
    schema_version: int = SCHEMA_VERSION
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float | None = None
    median_parallax_px: float | None = None
    median_parallax_deg: float | None = None
    # The actual degeneracy discriminator (see Keyframe.r_h for why it is
    # not r_h): the fraction of correspondences passing recoverPose's
    # cheirality check. Measured to be near-binary -- ~0.01 when
    # degenerate, ~1.0 when healthy -- at roughly 0.5 deg median parallax.
    cheirality_fraction: float | None = None
    r_h: float | None = None
    rotation_dominant: bool | None = None
    pose_status: str = POSE_STATUS_UNAVAILABLE
    degeneracy: str = DEGENERACY_NONE
    quality: Confidence = Confidence.UNKNOWN
    frame_revision: int = 1

    def to_json_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "from_keyframe_id": self.from_keyframe_id,
            "to_keyframe_id": self.to_keyframe_id,
            "matches": self.matches,
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "median_parallax_px": self.median_parallax_px,
            "median_parallax_deg": self.median_parallax_deg,
            "cheirality_fraction": self.cheirality_fraction,
            "r_h": self.r_h,
            "rotation_dominant": self.rotation_dominant,
            "pose_status": self.pose_status,
            "degeneracy": self.degeneracy,
            "quality": self.quality.value,
            "frame_revision": self.frame_revision,
        }


def keyframe_edge_from_json_dict(data: dict) -> KeyframeEdge:
    return KeyframeEdge(
        schema_version=data["schema_version"],
        from_keyframe_id=data["from_keyframe_id"],
        to_keyframe_id=data["to_keyframe_id"],
        matches=data["matches"],
        inliers=data["inliers"],
        inlier_ratio=data["inlier_ratio"],
        median_parallax_px=data["median_parallax_px"],
        median_parallax_deg=data["median_parallax_deg"],
        cheirality_fraction=data["cheirality_fraction"],
        r_h=data["r_h"],
        rotation_dominant=data["rotation_dominant"],
        pose_status=data["pose_status"],
        degeneracy=data["degeneracy"],
        quality=Confidence(data["quality"]),
        frame_revision=data["frame_revision"],
    )
