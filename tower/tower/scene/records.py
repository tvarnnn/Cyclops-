"""What Scene Understanding knows, and the words it is allowed to use.

Two constraints shape every name here.

**This is a live state, not a memory.** Nothing in this module is
persisted, so there is no schema version, no `time_basis` on a stored
record, and no retention policy -- because there is nothing stored to
version, stamp or expire. A cartridge answering "what is around me now"
has no reason to write to disk, and Environmental Memory is where a
durable record would belong.

**Nothing here is identity.** A `track_id` means "the same blob I
associated across adjacent frames", not "the same person as yesterday".
It is an integer, session-scoped, and meaningless once the process ends.

And nothing here is gaze. `07-PLATFORM-CONSTRAINTS.md` Limitation 8: the
camera cannot establish that anyone looked at anything. The field is
`appears_facing_wearer`, and it stays that even when it is inconvenient.
"""

from dataclasses import dataclass, field, replace

from tower.confidence import Confidence

# Coarse facing states. Since 2026-09-07 only two are ever PRODUCED --
# `toward_wearer`, when the front of a head is visible inside a person's
# box, and `unknown`, which means "not established" and covers facing
# away, side-on, too small and never measured alike. `away_from_wearer`
# and `profile` remain defined so a consumer that pinned the vocabulary
# keeps decoding, and so the reason they are not produced can be stated:
# nothing measured on this platform produces them with usable precision
# (`tower/scene/orientation.py`).
#
# There is deliberately no state meaning "looking at you". Head
# orientation is not eye direction: a person squarely facing the wearer
# may be reading something over their shoulder.
FACING_UNKNOWN = "unknown"
FACING_TOWARD = "toward_wearer"
FACING_AWAY = "away_from_wearer"
FACING_PROFILE = "profile"

# Where in the view a thing is, horizontally. Assigned with hysteresis
# by the engine (`state.assign_side`) so a thing near a boundary does
# not flicker between two words while nothing moved.
SIDE_LEFT = "left"
SIDE_CENTRE = "centre"
SIDE_RIGHT = "right"
SIDE_UNKNOWN = "unknown"

# Relationships this cartridge is willing to assert from 2-D boxes. The
# ones it REFUSES -- on, inside, near, in_front_of, behind, and
# nearer_than_same_class -- are absent on purpose and each is recorded in
# state.REFUSED_RELATIONSHIPS with what would settle it, because a
# relationship nobody can support is worse than a missing one.
REL_LEFT_OF = "left_of"
REL_RIGHT_OF = "right_of"
REL_HIGHER_IN_VIEW = "higher_in_view"

RELATIONSHIPS = (
    REL_LEFT_OF,
    REL_RIGHT_OF,
    REL_HIGHER_IN_VIEW,
)


@dataclass(frozen=True)
class BoundingBox:
    """Pixel coordinates, x0/y0 top-left, in the frame as received."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 0.0)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 0.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over union. The only association signal used.

        Deliberately NOT appearance similarity: matching by how something
        looks is the first step toward recognising it again, and this
        cartridge must never do that.
        """
        left = max(self.x0, other.x0)
        top = max(self.y0, other.y0)
        right = min(self.x1, other.x1)
        bottom = min(self.y1, other.y1)
        if right <= left or bottom <= top:
            return 0.0
        overlap = (right - left) * (bottom - top)
        union = self.area + other.area - overlap
        return float(overlap / union) if union > 0 else 0.0

    def to_json_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass(frozen=True)
class Detection:
    """One thing a detector claims to see, before any tracking.

    A detection is **evidence**, not a fact (Core Principle 1), and it is
    model inference rather than measurement (Core Principle 2). Counting
    these directly is the mistake the tracker exists to prevent.
    """

    label: str
    score: float
    box: BoundingBox

    def to_json_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "box": self.box.to_json_dict(),
        }


@dataclass(frozen=True)
class FacingEstimate:
    """Whether the front of a person's head is visible, and how stale that is.

    `age_seconds` is not optional decoration. Estimating this runs at a
    bounded cadence rather than per frame, so a consumer that cannot see
    the age would treat a stale answer as current.

    `evidence` says why the state is what it is -- a face was visible, no
    face was found, the face was too weak, the box was too small -- and
    stays on this side of the wire. It exists so a test and a log can
    tell "unknown because nobody looked" from "unknown because we looked
    and could not tell".
    """

    state: str = FACING_UNKNOWN
    confidence: Confidence = Confidence.UNKNOWN
    age_seconds: float | None = None
    evidence: str = "none"

    @property
    def appears_facing_wearer(self) -> bool:
        """Note the verb. This is orientation evidence, never gaze."""
        return self.state == FACING_TOWARD

    def to_json_dict(self) -> dict:
        return {
            "state": self.state,
            "confidence": self.confidence.value,
            "age_seconds": self.age_seconds,
            "evidence": self.evidence,
            "note": "orientation evidence, not gaze; the camera cannot see attention",
        }


@dataclass
class Track:
    """One thing followed across frames. Anonymous, and session-scoped.

    `track_id` is an integer that means "the same blob as last frame". It
    is not an identity, does not survive the process, and must never be
    persisted or matched against anything outside this session.
    """

    track_id: int
    label: str
    box: BoundingBox
    score: float
    first_seen_at: float
    last_seen_at: float
    hits: int = 1
    misses: int = 0
    # Consecutive hits, reset by a miss. `hits` is a lifetime total and
    # makes a poor confirmation test on its own: a reflection glimpsed
    # once every six frames accumulates three lifetime hits and becomes a
    # permanent phantom person, never having been seen twice in a row.
    streak: int = 1
    # Latched once a streak reaches the policy's minimum, and NOT
    # recomputed afterwards. Latching is what lets a confirmed track
    # survive a detector dropout without the count flickering -- which is
    # the whole point -- while still requiring real consecutive evidence
    # to earn confirmation in the first place.
    is_confirmed: bool = False
    facing: FacingEstimate = field(default_factory=FacingEstimate)
    # When THIS track's facing was last actually estimated -- per track,
    # not per orientation run. The distinction is the whole value of the
    # age: a run that fails to find this person must not refresh their
    # estimate's age, or a reading from ten seconds ago reports as one
    # second old and never expires.
    facing_estimated_at: float | None = None
    # The last few RAW facing states, oldest first, that the published
    # `facing` was voted from. A tuple, so `snapshot()`'s shallow copy
    # cannot share a list with the tracker.
    facing_history: tuple = ()
    # Which side of the view this track was last placed on, with
    # hysteresis. None until the engine has learned the frame size.
    side: str | None = None

    def snapshot(self) -> "Track":
        """A copy that will still be true in a second's time.

        `Track` is mutable and the tracker rewrites it in place on every
        frame -- `box`, `score`, `last_seen_at`, `hits`, `streak`,
        `misses`, `is_confirmed` -- so a `SceneState` that referenced the
        tracker's own objects was a live view wearing a timestamp. Held
        for one publish tick it serialised old scalars beside new track
        values: a state stamped `frames_observed: 3` reporting `hits: 8`.

        Shallow is deep enough, and that is a property worth stating
        rather than assuming: `BoundingBox` and `FacingEstimate` are both
        frozen and `facing_history` is a tuple, so the only mutable thing
        reachable from a track is the track. If any of those ever stops
        being immutable this must copy it too.

        Not `copy()`: the name says WHY. This is the moment the state is
        cut loose from the tracker, and it is the only place that
        happens.
        """
        return replace(self)

    @property
    def age_seconds(self) -> float:
        return max(self.last_seen_at - self.first_seen_at, 0.0)

    def confirmed(self, min_hits: int) -> bool:
        """Seen often enough, CONSECUTIVELY enough, to count.

        A single detection is a flicker. So is one every six frames --
        which a lifetime hit count would happily confirm and then keep
        confirmed forever. Confirmation needs a streak; staying confirmed
        needs only staying alive.
        """
        return self.is_confirmed or self.streak >= min_hits

    def counted(self, min_hits: int, max_count_misses: int) -> bool:
        """Confirmed AND seen recently enough to be counted as present.

        A track outlives its last detection for a while so the same
        thing coming back from behind a doorframe keeps its id; that is
        continuity. Being COUNTED is a stronger claim -- "this is in
        view now" -- and stops sooner. The two bounds are separate on
        purpose (`tracking.TrackerPolicy`).
        """
        return self.confirmed(min_hits) and self.misses <= max_count_misses

    def to_json_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "score": self.score,
            "box": self.box.to_json_dict(),
            "hits": self.hits,
            "streak": self.streak,
            "misses": self.misses,
            "confirmed": self.is_confirmed,
            "age_seconds": round(self.age_seconds, 3),
            "facing": self.facing.to_json_dict(),
            "side": self.side,
        }


@dataclass(frozen=True)
class Relation:
    """One asserted relationship between two tracks.

    `frame_of_reference` is on every relation because every relation this
    cartridge can assert is **camera-relative**. There is no live world
    pose to anchor to -- World Builder produces poses offline, after a
    session -- so "left of" means left in the wearer's current view, and
    will mean something different the moment they turn their head.
    """

    subject_track_id: int
    relationship: str
    object_track_id: int
    confidence: Confidence = Confidence.MEDIUM
    frame_of_reference: str = "camera"

    def to_json_dict(self) -> dict:
        return {
            "subject_track_id": self.subject_track_id,
            "relationship": self.relationship,
            "object_track_id": self.object_track_id,
            "confidence": self.confidence.value,
            "frame_of_reference": self.frame_of_reference,
        }
