"""Live-scene transport adapter for Scene Understanding.

Named after its cartridge, for the reason `world_builder.py` and
`object_memory.py` are: an adapter named after one cartridge cannot leak
that cartridge's assumptions into the next, because the next one gets its
own file.

THIS FILE PROJECTS. IT DOES NOT DRIVE.

It is handed a `SceneState` that a worker thread already produced and
turns it into JSON. It never calls `SceneEngine.observe`, never decodes a
frame and never starts anything -- `test_the_result_channel_never_writes`
forbids a call named `observe` or `build` anywhere under `tower/results/`,
and that prohibition is doing real work here rather than being an
inconvenience: a results module that could drive an engine would be a
second, unsynchronised frame path.

WHAT THIS PAYLOAD MAY NOT SAY, AND WHY THE REASONING IS NOT THE OBVIOUS ONE

The reflex is "minimise disclosure". That reflex is wrong here and the
correct analysis is narrower:

    Tower -> phone is inside the local-first boundary. THE PHONE SENT THE
    PIXELS. A count therefore discloses strictly LESS than the frame the
    phone already holds, and withholding it while shipping frames is
    theatre.

What is genuinely new is **joinability**. A stable `track_id` plus a
timestamp lets a recipient assemble the per-person dwell timeline this
cartridge refuses to keep -- persists-nothing laundered onto the consumer.
The cartridge's best property would be defeated not by what it says but
by what someone could accumulate from it.

So, for people: no `track_id`, no bounding box, no `normalised_x` or
`view_offset`, no per-person facing state, and no `visible_eyes` /
`visible_ears`. Facial-landmark evidence does not cross this boundary at
all. Facing is an aggregate count or an explicit `null` with a reason --
never zero, because zero is an answer and "never measured" is not.

`where` therefore excludes `person` and carries SIDE COUNTS rather than a
side. One side per label cannot describe a chair on the left and a chair
on the right, and picking one of them would be a wrong answer where a
refusal was available.

REFUSED RELATIONS ARE UNEXPRESSIBLE, NOT MERELY UNPOPULATED

`relations` is `None` and there is no schema slot anywhere below that
could hold `in_front_of`. A refusal that depends on remembering not to
fill a field is not a refusal. `refused_relations` names them so a client
can say WHY rather than showing an empty list that looks like "nothing is
near anything".

EVERY COUNT IS AN UNDERCOUNT AND MUST SAY SO

Measured against a `fasterrcnn_resnet50_fpn_v2` oracle over 14,128 real
frames (`docs/superpowers/research/2026-08-26-detector-oracle-and-the-
size-floor.md`), the shipped detector's recall is 0.306 for `person`,
0.497 for `cell phone`, 0.209 for `tv`, and it is effectively blind below
~2% of frame area (recall 0.000 under 1%). Because the oracle shares COCO
training data with the shipped model, 0.306 is an UPPER BOUND.

An undercount published without that disclosure looks exactly like a quiet
room, which is why `count_is_lower_bound` and `count_limitations` are
required fields rather than documentation.

EVERY BOOLEAN IS WRAPPED IN `bool()`

`bool` subclasses `int` in Python. A `registered: 1` already shipped once
in this repository and would fail every Swift `as? Bool` decode.
"""

from tower.results.contracts import SCENE_LIVE_CONTRACT, TIME_BASIS
from tower.scene.detect import CLASSES_OF_INTEREST
from tower.scene.live import (
    STATE_FAILED,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
)
from tower.scene.records import FACING_UNKNOWN
from tower.scene.state import (
    REFUSED_RELATIONSHIPS,
    SIDE_HYSTERESIS,
    SIDE_LEFT_BELOW,
    SIDE_RIGHT_ABOVE,
    SIZE_LARGE,
    SIZE_MEDIUM,
    SIZE_SMALL,
    SIZE_UNKNOWN,
    apparent_size,
    describe_position,
)

# The three sentences above, as values a decoder can switch on.
SCENE_CLAIM = "visible-now-not-a-record"
IDENTITY_SCOPE = "anonymous-and-unpublished"
ABSENCE_MEANING = "not-visible-to-this-cartridge"
PERSISTENCE = "none"
FRAME_OF_REFERENCE = "camera"
COUNT_BASIS = "confirmed-tracks"

# Fixed arity. The universe of labels that can ever appear in `counts` --
# which is what keeps the payload bounded without a truncation rule, and
# what lets a client tell "looked for and not seen" from "never looked
# for". Sorted so the payload is stable and its revision hash does not
# move because a set iterated differently.
REPORTED_CLASSES = tuple(sorted(CLASSES_OF_INTEREST))

# Since 2026-09-07 nothing is excluded from `where`: "is there a person
# on my left" is one of the questions this cartridge exists to answer,
# and a side count for people is a count, not a track. The earlier
# refusal reasoned that a per-person position sampled repeatedly is a
# movement trace; three buckets of counts, with no handle to join two
# payloads by, carry no more of a trace than the counts already do.
POSITION_EXCLUDED_LABELS = ()

# When these were measured, and on what. Carried on every payload,
# because a measurement asserted in the present tense reads as current
# state -- and this platform's corpus grows continuously, so it is not.
COUNT_MEASUREMENT = {
    "measured_at": "2026-09-07",
    "corpus_frames": 700,
    "corpus_captures": 0,
    "is_current": True,
    "note": (
        "accuracy was measured on 700 human-labelled COCO val2017 images "
        "resized to this camera's 640 px long side -- third-party stills, "
        "not this camera -- because the 45,594-frame corpus on this host "
        "contains no bystander. Latency was measured on 300 real corpus "
        "frames. Nothing here is validated on a person seen through these "
        "glasses"
    ),
}

COUNT_LIMITATIONS = (
    {
        "limitation": "size-floor",
        "detail": (
            "small things are missed. On the labelled images the CUDA "
            "detector (RT-DETRv2-R18, at its 0.5 floor) finds 0.40 of "
            "objects under 1% of frame area, 0.64 at 1-2%, 0.74 at 2-5%, "
            "0.81 at 5-20% and 0.88 above 20%; the CPU detector "
            "(SSDLite320, at 0.4) finds 0.00, 0.04, 0.36, 0.62 and 0.81. "
            "`detector` on this payload says which one counted"
        ),
    },
    {
        "limitation": "recall",
        "detail": (
            "per-class AP50 on the labelled images, CUDA detector: 0.86 "
            "person, 0.83 laptop, 0.82 tv, 0.86 mouse, 0.71 keyboard, "
            "0.71 bed, 0.67 couch, 0.68 bottle, 0.63 cup, 0.55 cell phone, "
            "0.54 chair, 0.50 dining table, 0.35 book. CPU detector: 0.60 "
            "person, 0.55 laptop, 0.56 tv, 0.20 chair, 0.09 book. Across "
            "the 13 classes the CUDA detector's recall at its floor is "
            "0.61 for a precision of 0.70; a count is a floor"
        ),
    },
    {
        "limitation": "people-count-accuracy",
        "detail": (
            "on 700 labelled stills the CUDA detector's per-image count "
            "of people is exact on 73% of images with a mean error of "
            "0.48 people (CPU detector at its 0.4 floor: 66%, 1.15). "
            "Those are stills of other people's photographs, the same "
            "700 the operating threshold was chosen on; no count has "
            "been checked against a person standing in front of these "
            "glasses"
        ),
    },
    {
        "limitation": "departure-lag",
        "detail": (
            "a confirmed track keeps being counted for up to 6 further "
            "OBSERVED frames after its last detection -- 0.5 s at the "
            "measured 12.0 fps delivery, and longer whenever "
            "frames_skipped is advancing, because the bound is a frame "
            "count and not a duration. A count can therefore include "
            "someone who left within the last half second. Confirmation "
            "needs 3 consecutive detections, so someone who just arrived "
            "is counted about a quarter of a second late"
        ),
    },
    {
        "limitation": "field-of-view",
        "detail": (
            "a count is about the camera's 45-degree forward cone at this "
            "instant, never about the room. Most of a room is behind the "
            "wearer"
        ),
    },
)

# Why `relations` is null, as a value rather than a silence.
RELATIONS_ABSENT_REASON = (
    "this cartridge asserts no relations on the wire. Three are "
    "computable from 2-D boxes -- left_of, right_of and higher_in_view -- "
    "and all three are withheld rather than refused: they are true, they "
    "are camera-relative, and they change the moment the wearer turns "
    "their head, so a client that cached one would be holding a claim "
    "about a view that no longer exists. Every relation worth having "
    "needs depth that survives motion, and that was measured and refused. "
    "See refused_relations"
)

# The relations this cartridge CAN compute and does not publish, kept
# apart from the ones it refuses. A client must be able to tell "we can
# and will not" from "we cannot, and here is the measurement".
WITHHELD_RELATIONS = ("left_of", "right_of", "higher_in_view")

# The lifecycle vocabulary, published so a client can pin it. A value
# outside this set is a value the phone decodes into nothing.
LIFECYCLE_STATES = (
    STATE_STOPPED,
    STATE_STARTING,
    STATE_RUNNING,
    STATE_PAUSED,
    STATE_FAILED,
)

# Fields whose value advances without anything having happened, or whose
# advance is not news about the SCENE. Excluded from the change revision
# so a client can tell new data from repeated data -- the same rule
# `world_builder.py: VOLATILE_PATHS` applies to `mapping_seconds`.
#
# `frames_observed` is in here and that deserves a sentence: a frame
# having been processed is not the same event as the scene having changed,
# and including it would make every single poll look like news and defeat
# coalescing entirely -- which is the one thing `IOS-to-Tower.md` 4.8 asks
# this cartridge for by name.
VOLATILE_PATHS = (
    "observed_at",
    "staleness_seconds",
    "frames_offered",
    "frames_observed",
    "frames_skipped",
    "frames_dropped_not_running",
    "decode_failures",
    "lifecycle.started_at",
    "lifecycle.ready_at",
    "lifecycle.loading_seconds",
    "people.oldest_estimate_seconds",
    # How many are streaming and watching is diagnostics about the
    # channel, not news about the scene. A subscriber joining must not
    # mint an envelope for every other subscriber.
    "lifecycle.demand.streams",
    "lifecycle.demand.watchers",
    "lifecycle.demand.operator_hold",
)

# The one thing the aggregates cannot hide, stated rather than left to
# be discovered. With exactly one person in view, `where.person`,
# `by_apparent_size` and `facing_wearer` describe that one person --
# which is the product question ("is there a person on my left, facing
# me") and is what the wearer's own camera preview already shows. What
# stays refused is what would make it a RECORD: no handle to join two
# payloads, nothing persisted, a session-scoped `session_id` that
# forbids comparison across sessions, and the client obligation
# (§12 of the contract) never to store a sequence of these.
SINGLE_PERSON_NOTE = (
    "with one person in view the side, size and facing counts describe "
    "that person for as long as they are in view. Nothing here is kept, "
    "nothing joins two payloads, and a client must not store a sequence "
    "of them; the phone's own camera preview shows the same and more"
)

# Why there is no entity list, as a value rather than an absence.
#
# `IOS-to-Tower.md` 4.1 asks for a session-scoped anonymous track handle
# -- explicitly NOT a durable identifier, which it separately forbids --
# so a client can tell the person on the left from the person on the
# right within one session and label rows positionally. Its types exist
# for it.
#
# This cartridge refuses, and the refusal was delivered as SILENCE until
# an adversarial review pointed out that this file's own rule forbids
# that: "a refusal that depends on remembering not to fill a field is not
# a refusal". A client finding nothing to decode cannot tell "refused"
# from "not implemented yet", and those call for opposite responses --
# build a different screen, or wait for the next Tower.
TRACKS_ABSENT_REASON = (
    "this cartridge publishes no per-entity list and no track handle, not "
    "even a session-scoped one. A handle plus a timestamp lets a "
    "recipient assemble the per-person dwell timeline this cartridge "
    "refuses to keep -- persists-nothing laundered onto the consumer. "
    "Counts and aggregate facing are the only representation offered, and "
    "there is no key below this one that could hold an entity"
)

REFUSED_ENTITY_FIELDS = (
    {"field": "track_id", "reason": "joinable across time within a session"},
    {"field": "box", "reason": "a repeated position is a movement trace"},
    {
        "field": "facing",
        "reason": "per-person orientation is per-person state",
    },
    {
        "field": "visible_eyes",
        "reason": "facial-landmark evidence does not cross this boundary",
    },
    {
        "field": "confidence",
        "reason": "requires a per-entity row, which does not exist here",
    },
)

# The one convention `IOS-to-Tower.md` declares rather than leaves open,
# answered. 4.3: "a bearing has to be signed somehow to be usable and a
# silent presumption is the dangerous version -- a Tower signing the
# other way would put every person on the wrong side of the wearer,
# rendering confidently and wrongly. Please state yours."
#
# This payload publishes no bearing. It publishes `where`, which is a
# coarse signed bearing under another name, and the same warning applies
# to it.
SIDE_CONVENTION = (
    "the wearer's own left and right, as the camera sees them. A thing is "
    f"'left' when its box centre falls below {SIDE_LEFT_BELOW} of frame "
    f"width in the frame as received, 'right' above {SIDE_RIGHT_ABOVE}, "
    "'centre' between the two -- about 14 degrees of this camera's 45 -- "
    f"with {SIDE_HYSTERESIS} of hysteresis so a thing on a line keeps its "
    "word, and 'unknown' when the frame size was never learned. The "
    "stream is assumed unmirrored and nothing on this wire verifies that. "
    "It is camera-relative and changes when the wearer turns their head"
)

# What the size words mean, and what they do not.
APPARENT_SIZE_NOTE = (
    "how much of the frame's height a person's box takes: 'large' is 0.6 "
    "or more, 'small' under 0.3, 'medium' between. For a standing adult "
    "larger usually means nearer -- about 2 m at 0.6 through this lens -- "
    "but a seated person, a child or a figure cut off by the frame edge "
    "breaks that, so these are sizes in the picture and never distances"
)

# What a partial figure at the bottom edge is, and why it is kept apart.
PARTIAL_BOTTOM_EDGE_NOTE = (
    "person boxes with no head region in view -- their top starts below "
    "the middle of the frame -- or that span the frame's width down to "
    "its bottom edge. From a camera worn at head height that is most "
    "often the wearer's own hands, arms, lap or legs; it can also be "
    "somebody's legs under a table. Either way the camera saw part of a "
    "figure, not a person in front of the wearer, so these are counted "
    "here and not in `count`"
)

# The states in which the scene below is being refreshed. `paused` is
# deliberately not one of them: a paused session's counts are the last
# thing it saw, not what is there now.
_CURRENT_STATES = (STATE_RUNNING,)


# Where the full reasoning lives. The refusals in `tower/scene/state.py`
# are research abstracts -- the `in_front_of` entry alone runs to 1,500
# characters of flip rates and sample sizes -- and every one of them is
# CONSTANT. Publishing them whole put 4 KB of unchanging prose into a
# payload the heartbeat re-sends every 2 s, per subscriber, on the socket
# that shares its send lock with the frame path.
#
# A client asking "why not?" wants a sentence. The measurement belongs in
# the repository, and this names where.
REFUSAL_EVIDENCE = (
    "tower/scene/state.py: REFUSED_RELATIONSHIPS, and "
    "docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md"
)


def _first_sentence(text: str) -> str:
    """The claim, without the workings.

    Splits on the first full stop followed by a space, which is enough
    for prose written as prose and degrades to the whole string when it
    is not -- a truncation that cut a sentence in half would be worse
    than a long one.
    """
    head, separator, _rest = text.partition(". ")
    return head + "." if separator else text


def _side_counts(state) -> dict:
    """Per-label side counts, for everything that is not a person.

    Every non-person reported class is present with four zeroed buckets,
    for the same reason `counts` is: a label that appeared and
    disappeared would be indistinguishable from a version skew, and a
    client could not tell "no chair on the left" from "this Tower does
    not report chairs". Fixed arity is also what bounds the block --
    12 labels x 4 integers, whatever is in the room.

    `describe_position` returns `side: "unknown"` when the frame size was
    never learned, and that lands in its own bucket rather than being
    folded into `centre` -- a scene whose frame size is unknown has not
    placed anything in the middle of the view, it has placed nothing.

    The buckets themselves are defined on the wire, in `SIDE_CONVENTION`.
    A left and a right with no stated convention is exactly the silent
    presumption `IOS-to-Tower.md` 4.3 spends three sentences warning
    about, arriving through the field it did not think to ask about.
    """
    where = {
        label: {"left": 0, "centre": 0, "right": 0, "unknown": 0}
        for label in REPORTED_CLASSES
        if label not in POSITION_EXCLUDED_LABELS
    }
    for track in state.tracks:
        if track.label in POSITION_EXCLUDED_LABELS:
            continue
        side = describe_position(track, state.frame_width, state.frame_height)["side"]
        bucket = where[track.label]
        bucket[side] = bucket.get(side, 0) + 1
    return where


def _people_block(state) -> dict:
    """People, as a count and an aggregate -- never as a list.

    `may_include_wearer` and `validated` are not hedging. The `person`
    boxes in this platform's only real corpus are almost certainly the
    wearer's own torso -- median box bottom edge 0.985, 58.4% touching
    the frame edge -- and the distribution is unimodal with a continuous
    tail rather than bimodal, leaving a 34.3% residual that no threshold
    separates and no confirmed bystander to validate against. No
    bystander footage exists on this host at all. An unqualified count
    would overclaim, and the overclaim would be invisible.

    `facing_wearer` is `null` -- with a reason -- whenever orientation
    never produced an estimate. It is never 0. Zero is an answer and
    "never measured" is not, and this is the one field on this payload
    where the difference is most likely to be mistaken.
    """
    people = state.of_class("person")
    unknown = sum(1 for track in people if track.facing.state == FACING_UNKNOWN)
    if state.orientation_enabled and not (people and unknown == len(people)):
        facing = len(state.facing_wearer())
        reason = None
        ages = [
            track.facing.age_seconds
            for track in people
            if track.facing.age_seconds is not None
        ]
        oldest = max(ages) if ages else None
    elif state.orientation_enabled:
        # Enabled, and no person currently has an established orientation.
        # `orientation_enabled` latches on ONE lifetime success and an
        # estimate ages out to unknown after 6 s -- so an estimator that
        # succeeded once and then failed for good would have taken the
        # branch above forever and published `facing_wearer: 0` while
        # measuring nothing.
        #
        # Zero is an answer. "Nothing is established" is not, and this is
        # the one field on this payload where the difference is most
        # likely to be mistaken for data. The reason says WHICH kind of
        # not-established it is -- a reviewer found the first version
        # asserting an expiry that had not happened for a person whose
        # face was simply not found.
        facing = None
        reason = _not_established_reason(people)
        oldest = max(
            (
                track.facing.age_seconds
                for track in people
                if track.facing.age_seconds is not None
            ),
            default=None,
        )
    else:
        facing = None
        reason = (
            "coarse orientation has never produced an estimate on this "
            "session -- either no pose estimator is configured, or the "
            "model has not once succeeded. Reporting 0 would be an "
            "observation gap presented as an observation of absence"
        )
        oldest = None
        unknown = None
    sizes = {SIZE_LARGE: 0, SIZE_MEDIUM: 0, SIZE_SMALL: 0, SIZE_UNKNOWN: 0}
    for track in people:
        sizes[apparent_size(track, state.frame_height)] += 1
    return {
        "count": len(people),
        # A category, never an identity. There is no list here, no handle,
        # and nothing a client could join across two of these payloads.
        "may_include_wearer": True,
        "validated": False,
        "partial_bottom_edge": len(state.partial_people),
        "partial_bottom_edge_note": PARTIAL_BOTTOM_EDGE_NOTE,
        "by_apparent_size": sizes,
        "apparent_size_note": APPARENT_SIZE_NOTE,
        "facing_wearer": facing,
        "facing_answered": bool(facing is not None),
        "facing_unavailable_reason": reason,
        "facing_unknown": unknown,
        # Two states are produced and the other two of the enum
        # `IOS-to-Tower.md` 4.2 asks for are not, and the reason is now
        # measurement rather than privacy: nothing evaluated on this
        # platform can say "away" or "side-on" with usable precision
        # (0.02-0.34 on 966 labelled persons), so `unknown` means "not
        # established" and covers both.
        "facing_states_reported": ["facing_wearer", "unknown"],
        "facing_states_withheld": ["away_from_wearer", "profile"],
        "facing_states_withheld_reason": (
            "'facing away' and 'side-on' are never claimed: no method "
            "measured on this platform produces them with usable "
            "precision, so a person who is not established as facing the "
            "wearer is reported as unknown, whichever way they face"
        ),
        "oldest_estimate_seconds": oldest,
        "orientation_method": state.orientation_method,
        "orientation_status": "experimental",
        "orientation_validation": (
            "a face detector on each tracked person's box; 'facing' needs "
            "a strong face in two of the last three estimates, about half "
            "a second. Measured at 0.83 precision and 0.64 recall on 966 "
            "human-labelled persons in COCO stills, where 47% of people "
            "face the camera. At the same recall and false-positive rate, "
            "a room where only 10-20% of people face the wearer at a given "
            "instant would give roughly 0.4-0.6 precision: about half of "
            "the 'facing' claims could be wrong there. Never measured on a "
            "person seen through these glasses"
        ),
        # The wording is part of the contract. `IOS-to-Tower.md` 4.2: this
        # is body/head orientation relative to the camera, it is NOT gaze,
        # and there is no eye tracking on the target glasses.
        "facing_note": (
            "whether the front of a person's head is visible to the "
            "camera. Render as 'appears to be facing your direction'; "
            "there is no eye tracking on this platform, so it cannot "
            "establish what anyone was looking at or whether they noticed "
            "the wearer"
        ),
    }


def _not_established_reason(people) -> str:
    """Why no person has an established orientation, from their evidence."""
    expired = sum(
        1
        for track in people
        if track.facing.age_seconds is not None and track.facing.age_seconds > 6.0
    )
    never = sum(1 for track in people if track.facing_estimated_at is None)
    looked = len(people) - expired - never
    parts = []
    if looked:
        parts.append(
            f"{looked} looked at and no face established (none found, too "
            "weak, or the box too small)"
        )
    if never:
        parts.append(f"{never} not yet estimated")
    if expired:
        parts.append(f"{expired} with an estimate older than the 6 s bound")
    return (
        "no person currently has an established orientation: "
        + ", ".join(parts)
        + ". Reporting 0 would be an observation gap presented as an "
        "observation of absence"
    )


def _lifecycle_block(status: dict) -> dict:
    state = status["state"]
    return {
        "state": state,
        "states": list(LIFECYCLE_STATES),
        # Session-scoped and meaningless afterwards, exactly like the
        # track ids it stands in for. Published so a client can tell that
        # two payloads came from two different tracking sessions and must
        # not be compared -- which is the ONLY joining this cartridge
        # wants a client to be able to do.
        "session_id": status["session_id"],
        "scene_is_current": bool(state in _CURRENT_STATES),
        "failure_reason": status["failure_reason"],
        "started_at": status["started_at"],
        "ready_at": status["ready_at"],
        "loading_seconds": status["loading_seconds"],
        "load_overdue": bool(status["load_overdue"]),
        "load_overdue_after_seconds": status["load_overdue_after_seconds"],
        # Whether this session starts and stops with the glasses' stream.
        # True is the default and is what makes the cartridge reachable
        # from a phone, which sends nothing when a cartridge is opened.
        "follows_stream": bool(status.get("follows_stream", False)),
        # Why the session is or is not running, as counts: how many
        # connections are streaming frames, how many subscriptions are
        # watching this live result, and whether an operator is holding
        # it open by hand. `runs_when` is the rule, pinned as a word.
        # Counts and never tokens -- a subscription id would be a handle.
        "demand": dict(status.get("demand") or {}),
    }


def _empty_scene_reason(status: dict) -> str:
    """Why there are no counts, in the client's own vocabulary.

    Four different silences, and a client that flattened them would show
    an empty room for all four. Only one of them means an empty room, and
    it is not in this function -- a running session that has observed a
    frame and found nothing returns counts, all zero.
    """
    state = status["state"]
    if state == STATE_STOPPED:
        return (
            "this session is stopped. Nothing is being observed, and the "
            "last scene was discarded rather than kept: a scene held past "
            "the end of a session is a claim about a room the wearer has "
            "left"
        )
    if state == STATE_STARTING:
        return (
            "the detector is still loading. This is not an empty room; it "
            "is a Tower that has not looked yet"
        )
    if state == STATE_FAILED:
        return status["failure_reason"] or "the session failed"
    return (
        "the session is running but has not finished observing a frame "
        "yet. No frame has been offered, or the first is still in flight"
    )


def live_payload(status: dict, state) -> dict:
    """The whole payload, from a session status and its latest scene.

    `state` may be None, and the four ways that happens are kept apart by
    `scene_unavailable_reason` rather than collapsed into an empty room.

    Every block is present in both cases with the SAME KEYS, because a
    strict decoder that saw `counts` appear and disappear would have to
    treat the field as optional and would lose the ability to distinguish
    "zero of these" from "this Tower did not say".
    """
    lifecycle = _lifecycle_block(status)
    payload = {
        "claim": SCENE_CLAIM,
        "identity": IDENTITY_SCOPE,
        "absence_means": ABSENCE_MEANING,
        "persistence": PERSISTENCE,
        "frame_of_reference": FRAME_OF_REFERENCE,
        "time_basis": TIME_BASIS,
        "lifecycle": lifecycle,
        "observed_at": status["observed_at"],
        # Named for what it is, because the name alone would mislead.
        # `IOS-to-Tower.md` 0.3 holds observedAt and receivedAt separately
        # and "will never substitute one for the other" -- and this is
        # receipt time. A decoder mapping by field name would make exactly
        # that substitution.
        "observed_at_note": (
            "tower-receipt time: when this Tower received the frame this "
            "scene came from, never when the glasses captured it, and not "
            "when the detector finished with it. There is no capture "
            "timestamp anywhere on this wire"
        ),
        "staleness_seconds": status["staleness_seconds"],
        "frames_offered": status["frames_offered"],
        "frames_observed": status["frames_observed"],
        "frames_skipped": status["frames_skipped"],
        "frames_dropped_not_running": status["frames_dropped_not_running"],
        "decode_failures": status["decode_failures"],
        "detector": status["detector"],
        "reported_classes": list(REPORTED_CLASSES),
        "count_basis": COUNT_BASIS,
        "count_is_lower_bound": True,
        "count_limitations": [dict(entry) for entry in COUNT_LIMITATIONS],
        "count_measurement": dict(COUNT_MEASUREMENT),
        # There is no confidence anywhere on this payload, and saying so
        # is not the same as omitting it. `IOS-to-Tower.md` 4.1 asks for
        # a confidence on every track; there are no tracks here, and a
        # confidence attached to a COUNT would be an average with no
        # defined meaning. `score_threshold` is the detector's floor and
        # is published instead.
        "confidence": None,
        "confidence_absent_reason": (
            "the detector emits a per-detection score, but the tracker "
            "never uses it for confirmation and this payload publishes no "
            "per-entity row for one to attach to. A confidence on a count "
            "would be an average of scores that did not decide anything. "
            "score_threshold is the floor those scores had to clear"
        ),
        # Null and unexpressible. There is no key below this one that
        # could hold a relation, which is what makes the refusal a
        # refusal rather than an omission somebody has to keep making.
        # Null and unexpressible, and the same treatment for the entity
        # list: a refusal a client can act on, rather than an absence it
        # has to interpret.
        "tracks": None,
        "tracks_absent_reason": TRACKS_ABSENT_REASON,
        "single_person_note": SINGLE_PERSON_NOTE,
        "refused_entity_fields": [dict(entry) for entry in REFUSED_ENTITY_FIELDS],
        "side_convention": SIDE_CONVENTION,
        "relations": None,
        "relations_absent_reason": RELATIONS_ABSENT_REASON,
        "withheld_relations": list(WITHHELD_RELATIONS),
        "refused_relations": [
            {
                "relation": name,
                "reason": _first_sentence(reason),
                "reason_source": REFUSAL_EVIDENCE,
            }
            for name, reason in sorted(REFUSED_RELATIONSHIPS.items())
        ],
        "where_excludes": list(POSITION_EXCLUDED_LABELS),
        "where_excludes_reason": (
            "nothing is excluded. `where.person` is side COUNTS for people "
            "-- left, centre, right -- with no handle that could join two "
            "payloads into a movement trace"
        ),
    }

    if state is None:
        payload["scene_available"] = False
        payload["scene_unavailable_reason"] = _empty_scene_reason(status)
        payload["score_threshold"] = None
        payload["counts"] = None
        payload["where"] = None
        payload["people"] = None
        return payload

    payload["scene_available"] = True
    payload["scene_unavailable_reason"] = None
    payload["score_threshold"] = state.score_threshold
    # Fixed arity: every reported class, present with 0 rather than
    # omitted. An absent key is a version skew; a zero is an answer -- and
    # `absence_means` at the top of the payload says what that zero is an
    # answer to.
    payload["counts"] = {
        label: int(state.counts.get(label, 0)) for label in REPORTED_CLASSES
    }
    payload["where"] = _side_counts(state)
    payload["people"] = _people_block(state)
    return payload
