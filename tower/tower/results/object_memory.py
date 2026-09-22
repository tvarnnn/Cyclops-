"""Observation transport adapter for Object Memory.

Named after its cartridge, for the same reason `world_builder.py` and
`world_builder_geometry.py` are: an adapter named after one cartridge
cannot leak that cartridge's assumptions into the next, because the next
one gets its own file. `tower/routes/observations.py` imports only this
module and never reaches `tower.object_memory` directly.

THIS FILE READS. IT DOES NOT DELETE.

`ObservationStore` also has `purge()` and `prune_expired()`. Neither is
reachable from here, and neither may become reachable. An unauthenticated
HTTP endpoint that erases a wearer's memory is not a feature, and
06-PRIVACY-DATA.md's real deletion stays where a human types it: the
`--purge-all` flag on `scripts/object_query.py`.

WHAT THIS PAYLOAD IS ALLOWED TO CLAIM

Every observation carries `spatial_ref: None`. The field is reserved,
never populated, and actively nulled on read -- nothing in this cartridge
knows where anything is in a room. So the payload answers "where" as a
FRAME REFERENCE: a capture id, a frame sequence number, a camera, and a
box in normalised frame coordinates. That is a pointer back into a
recording, not a place.

The names below are the enforcement. A doc that says "do not read this as
a location" is advice; a field called `where.kind == "frame-reference"`
with `spatial_ref` sitting next to it as an explicit null is a shape that
makes an overclaiming client awkward to write. Three claims are therefore
carried IN the payload rather than only in `docs/contracts/OBJECT-MEMORY.md`:

  claim          a category was VISIBLE ONCE -- not that it is there now
  identity       a CATEGORY, not an instance. Not "your" laptop.
  absence_means  no record is a statement about what this cartridge
                 captured, never about what is in the world

`spatial_ref` is carried as an explicit `null` rather than omitted, so a
consumer can see that the field exists and is empty. An absent key looks
like a version skew; a null is an answer.
"""

import time

from tower.object_memory import imagery
from tower.object_memory.imagery import (  # noqa: F401
    FILTER_UNAVAILABLE,
    IMAGERY_EXPIRED,
    NO_CAPTURE_ROOT,
    NO_FRAME_REFERENCE,
    NOT_FOUND,
    SOURCE_KEYFRAME,
    UNREADABLE,
)
from tower.object_memory.keyframes import KeyframeStore
from tower.object_memory.relevance import recordable_classes
from tower.object_memory.store import ObservationStore


def recorded_classes_for(verifier: str) -> tuple[str, ...]:
    """Which classes a Tower with this verifier will actually write.

    Lives in the ADAPTER, not in `main.py`, and that placement is the
    point. `main.py` is the wiring point and must not import a cartridge:
    it knows the world builder as an argv and it knows object memory the
    same way. This file is the one door -- it is already the only module
    outside `tower/object_memory/` that imports the cartridge's policy --
    so the answer travels through it as a tuple of strings.
    """
    return recordable_classes(verifier != "none")

# Opaque and dated, in the style of `world_builder.geometry/2026-08-25`.
# Compared for equality only: never parsed, never ordered, never used to
# infer that one contract is newer than another.
OBSERVATIONS_CONTRACT = "object_memory.observations/2026-08-26"

# The three sentences above, as values a decoder can switch on.
CATEGORY_CLAIM = "category-was-visible-once"
IDENTITY_SCOPE = "category-not-instance"
ABSENCE_MEANING = "not-observed-by-this-cartridge"

# `min(persisted, requested)` in one line, carried on every response so
# the rule travels with the data rather than living only in a document.
RETENTION_POLICY = (
    "min(persisted, requested): a reader may narrow this window and can "
    "never widen it"
)

SECONDS_PER_DAY = 86400.0


def retention_seconds_for(days: float | None) -> float | None:
    """A requested window in days, as the store's `retention_seconds`.

    None and "0 or less" both mean "no limit OF MY OWN", matching
    `scripts/object_query.py`. Neither means "keep forever": the store
    clamps to the window it was written under, so asking for nothing in
    particular gets the producer's promise, not the absence of one.
    """
    if days is None or days <= 0:
        return None
    return days * SECONDS_PER_DAY


def store_from_root(
    root, *, retention_days: float | None = None, clock=time.time
) -> ObservationStore:
    """Construct an `ObservationStore` for a configured root.

    Lives here, not in `tower/routes/observations.py`, so the route
    imports only this adapter -- the same rule that keeps
    `world_builder_geometry.store_from_root` where it is.
    """
    return ObservationStore(
        root, retention_seconds=retention_seconds_for(retention_days), clock=clock
    )


def _retention_view(store: ObservationStore, requested_days: float | None) -> dict:
    """What was asked for, what will be honoured, and whether they differ.

    `clamped` is True only when a caller asked for MORE than it received.
    A caller that asked for nothing has not been refused anything, and a
    caller that asked for less got exactly what it wanted.
    """
    effective = store.effective_retention_seconds()
    requested = retention_seconds_for(requested_days)
    if requested_days is None:
        clamped = False
    else:
        clamped = effective is not None and (
            requested is None or requested > effective
        )
    return {
        "requested_days": requested_days,
        # None is unbounded, and only ever reachable when the store itself
        # was written unbounded. Never 0: 0 days would mean "nothing is
        # visible", which is the opposite claim.
        "effective_days": None if effective is None else effective / SECONDS_PER_DAY,
        # A real bool. `bool` subclasses `int` in Python, and a `1` here
        # fails every Swift `as? Bool` decode.
        "clamped": bool(clamped),
        "policy": RETENTION_POLICY,
    }


def _where(observation) -> dict:
    """The only positional answer this cartridge can honestly give.

    The bounding box is nested HERE rather than beside the record on
    purpose. At the top level a box reads as a position; under
    `kind: "frame-reference"` it reads as what it is -- where in a
    picture, not where in a room.
    """
    box = observation.bounding_box
    return {
        "kind": "frame-reference",
        # Reserved, never populated. Present and null so a consumer sees
        # the field exists and is empty.
        "spatial_ref": None,
        "session_id": observation.session_id,
        "frame_seq": observation.frame_seq,
        # `source` is the camera the frame came from. Renamed on the wire
        # because "source" invites being read as a provenance system.
        "camera": observation.source,
        "bounding_box_normalized": list(box) if box is not None else None,
        # This pointer resolves into `data/captures/`, whose lifetime this
        # cartridge neither sets nor enforces. Purging every observation
        # here leaves the imagery exactly where it is.
        "imagery_retention": "capture-side",
    }


def _observation_view(observation) -> dict:
    return {
        # The handle the imagery routes take. Derived from the three
        # values that already identify a sighting, so the 64 records
        # written before this field existed have one too, permanently,
        # with no migration and no rewrite of a wearer's memory.
        "observation_id": observation.observation_id,
        "object_class": observation.object_class,
        "claim": CATEGORY_CLAIM,
        "identity": IDENTITY_SCOPE,
        # Three fields about strength, in the order a reader should trust
        # them. `confidence` is the interpretation and follows the best
        # look; the two raw numbers stay visible so the record is
        # auditable back to the sighting that created it.
        "confidence": observation.confidence.value,
        "detector_score": observation.detector_score,
        # None means "not tracked" -- records written before this field
        # existed. Never 0.0, which would be a claim of no evidence.
        "best_score": observation.best_score,
        "observed_at": observation.observed_at,
        # Qualified rather than implied: this slice can only know
        # tower-receipt time, never on-glasses capture time.
        "time_basis": observation.time_basis,
        "recorded_at": observation.recorded_at,
        # WHEN IT LEFT, and how much of it there was. `observed_at` says
        # when the category came into view and never moves; these two
        # accumulate while it stays there, and are what make a record an
        # observation rather than a detection. None on records written
        # before sightings existed -- never 0, which would claim a
        # sighting of no duration.
        "last_seen_at": observation.last_seen_at,
        "frame_count": observation.frame_count,
        # Which policy tier admitted this record, and what -- if
        # anything -- agreed with the detector about its label. Null
        # `verification` means nothing was asked, which is the ordinary
        # case for a class the detector can be trusted to name.
        "tier": observation.tier,
        "verification": observation.verification,
        "module_id": observation.module_id,
        "retention_tag": observation.retention_tag,
        "privacy_tags": list(observation.privacy_tags),
        "where": _where(observation),
    }


def _recorded_classes(recorded_classes) -> list[str]:
    """What this Tower will actually write, as the wire says it.

    Passed IN rather than imported from the cartridge's default, because
    the answer depends on configuration the adapter cannot see: a Tower
    with a semantic verifier records a wider set than one without, and
    naming a class here that nothing will ever write would turn "never
    looked for" -- the weaker silence, which a client words differently
    on purpose -- into "looked for and not seen".

    `None` falls back to the no-verifier set, which is the smaller and
    therefore the safer claim.
    """
    if recorded_classes is None:
        return list(recordable_classes(False))
    return list(recorded_classes)


def _envelope(
    store: ObservationStore, requested_days: float | None, recorded_classes=None
) -> dict:
    return {
        "contract": OBSERVATIONS_CONTRACT,
        "claim": CATEGORY_CLAIM,
        "identity": IDENTITY_SCOPE,
        "absence_means": ABSENCE_MEANING,
        # At the envelope too, not only per record: a client that reads
        # the header and stops must still learn there is no place here.
        "spatial_ref": None,
        # The universe of what could ever appear below. A class outside
        # this list has never been looked for, which is a different and
        # weaker kind of silence than "looked for and not seen".
        "recorded_classes": _recorded_classes(recorded_classes),
        # Where the pictures are, and what may be said about them. A
        # DESCRIPTOR, not an availability claim: whether a particular
        # record still has a picture is `.../imagery`'s answer, because
        # it depends on capture-side retention this cartridge neither
        # sets nor enforces.
        "imagery": IMAGERY_ROUTES,
        "retention": _retention_view(store, requested_days),
    }


def build_observations(
    store: ObservationStore,
    *,
    object_class: str | None = None,
    since: float | None = None,
    requested_retention_days: float | None = None,
    recorded_classes=None,
) -> dict:
    """Every observation the store will still serve, newest first.

    `all_observations` filters to the clamped window by default, and the
    `include_expired` opt-out is never passed from here. It exists for
    maintenance paths -- purge counting what it deletes, an operator
    auditing the file -- and is never the right answer for anything a
    wearer will be shown, which is everything on this wire.

    `since` IS THE ANSWER TO "WHAT DID THIS RECORDING REMEMBER", and it is
    why a failed recording can no longer wear a full store's history as its
    own. A physical run once recorded ZERO observations while the app kept
    showing 116 old ones, so "nothing happened" and "everything is fine"
    looked identical. A client that stamps a session's `started_at` and asks
    `since=<that>` gets exactly the records THIS recording created --
    `recorded_at` is the Tower-receipt moment a record was first written, on
    the same clock as `started_at` -- so an empty answer is an honest empty
    answer, and the historical view is a DIFFERENT request the wearer opts
    into. It filters by identity-in-time, not by a guess: a record is in or
    out by when the Tower wrote it, and nothing a reader passes moves that.
    """
    observations = store.all_observations()
    if object_class is not None:
        observations = [o for o in observations if o.object_class == object_class]
    if since is not None:
        # `recorded_at`, not `observed_at`: the boundary is when the Tower
        # WROTE the memory, which shares a clock with a session's
        # `started_at`. `observed_at` is capture-journal receipt time and
        # would answer a subtly different, wrong question near the edge.
        observations = [o for o in observations if o.recorded_at >= since]
    # Newest first: the question this cartridge exists to answer is "when
    # did I last see", so the most recent sighting should not be at the
    # bottom of a scroll.
    observations.sort(key=lambda o: o.observed_at, reverse=True)

    payload = _envelope(store, requested_retention_days, recorded_classes)
    payload["object_class"] = object_class
    # Echoed so the answer says what it was filtered to. `null` means "the
    # whole store", which is a different claim from "this session", and a
    # client drawing "new this recording" must be able to tell them apart
    # in the payload rather than only in its own memory of what it asked.
    payload["since"] = since
    payload["observation_count"] = len(observations)
    payload["observations"] = [_observation_view(o) for o in observations]
    return payload


def build_last_seen(
    store: ObservationStore,
    object_class: str,
    *,
    requested_retention_days: float | None = None,
    recorded_classes=None,
) -> dict:
    """When a category was last in view, or an honest silence.

    There is no 404 case here and that is a product decision, not an
    oversight. "No record of a laptop" answered as Not Found reads as
    "there is no laptop", which is a claim about the world this cartridge
    cannot make. The resource -- what Object Memory knows about a class --
    exists either way; sometimes it knows nothing, and it says so with
    `observed: false` alongside `absence_means`.

    `observed`, never `present`. The record says a category was visible
    once, not that it is still there.
    """
    observation = store.last_seen(object_class)

    payload = _envelope(store, requested_retention_days, recorded_classes)
    payload["object_class"] = object_class
    # A class this Tower never writes carries no information in its
    # absence at all. Widening that list is a decision about what the
    # system is allowed to remember AND about what it is able to read
    # correctly; reporting it is how a client tells the two silences
    # apart.
    payload["recordable"] = bool(object_class in payload["recorded_classes"])
    payload["observed"] = bool(observation is not None)
    payload["observation"] = (
        _observation_view(observation) if observation is not None else None
    )
    # Hoisted to the top level as well, because "where did I leave it" is
    # the question a client will actually bind to this response.
    payload["where"] = _where(observation) if observation is not None else None
    return payload


# --- resolving a record back to the picture it came from -----------------
#
# The pointer was always on the record. What was missing is the step that
# turns it into something a person can look at -- and the reason that is
# worth building is measured rather than assumed: MemPal's last-seen
# images showed the true location only 53% of the time and still moved
# retrieval accuracy from 0.81 to 0.95, with people searching 1.1 rooms
# instead of 1.9. A wrong-but-plausible picture plus a human beats a
# confident sentence.
#
# Everything here refuses rather than degrades. There is no path that
# serves an unfiltered first-person frame.
#
# The refusal REASONS are re-exported above rather than imported by the
# route. `tower/routes/observations.py` reaches only this adapter, and
# `test_the_object_memory_route_reaches_only_its_adapter` is what keeps
# that true -- a route that imported the cartridge "just for a constant"
# is how the boundary goes.

IMAGERY_CONTRACT = "object_memory.imagery/2026-08-27"

# What a served picture is, as a value a decoder can switch on. It is a
# FRAME FROM A RECORDING, filtered on the way out. It is not evidence of
# where anything is, and the box drawn on it is where in a picture, not
# where in a room -- the same claim `where` already makes.
IMAGERY_CLAIM = "frame-from-the-recording-this-record-was-derived-from"

# And WHEN the filter ran, which stopped being one answer on 2026-08-29.
#
# `tower/world_builder/redaction.py` runs before persistence and earns
# the name privacy transformation. A capture frame is the other case: the
# filter runs on READ, the raw frame stays exactly where it was, and
# saying so in the payload is what stops a client calling the result
# anonymised.
#
# An owned keyframe is the first case, not the second. `KeyframeStore.
# write` has no path by which the input crop reaches disk -- everything
# written derives from the filter's output -- so the file on disk IS the
# filtered image and there is no unfiltered original of it anywhere.
# Reporting that as "applied on read, the stored frame is unchanged"
# would be false in both halves of the sentence, and false in the
# direction that UNDERSTATES what protected the file. Neither direction
# is acceptable in a field whose whole job is to stop a client
# overclaiming, so the value follows the source.
FILTER_MEANS_ON_READ = "applied-on-read-the-stored-frame-is-unchanged"
FILTER_MEANS_BEFORE_WRITE = "applied-before-this-file-was-written"

# What `IMAGERY_ROUTES` advertises, before any particular picture has been
# asked for. The read-time answer, because it is the one that describes
# the capture tree this cartridge does not own, and because a route
# listing cannot know which source a future request will be served from.
FILTER_MEANS = FILTER_MEANS_ON_READ

# Templates rather than built URLs. The Tower does not know what host a
# phone reached it on, and a client that already holds a base URL should
# not be handed a second, possibly different one.
IMAGERY_ROUTES = {
    "contract": IMAGERY_CONTRACT,
    "claim": IMAGERY_CLAIM,
    "filter_means": FILTER_MEANS,
    "view": "/object-memory/observations/{observation_id}/imagery",
    "frame": "/object-memory/observations/{observation_id}/frame",
    "crop": "/object-memory/observations/{observation_id}/crop",
}


# Which store served a picture, and what that means for how long it will
# be there. Additive fields -- `IMAGERY_CONTRACT` does NOT move, because
# a shipped iOS build compares that identifier for equality and refuses a
# payload that does not carry it. A decoder that has never heard of a
# keyframe reads `imagery_retention` exactly as it always did.
RETENTION_CAPTURE_SIDE = "capture-side"
RETENTION_OBJECT_MEMORY = "object-memory"

# The prose, as values a decoder switches on rather than sentences it
# displays -- the same rule every other claim in this file follows.
CAPTURE_RETENTION_MEANS = (
    "this-picture-is-a-frame-in-data-captures-and-may-be-deleted-on-a-"
    "schedule-object-memory-does-not-set"
)
OBJECT_MEMORY_RETENTION_MEANS = (
    "this-picture-is-owned-by-object-memory-and-is-deleted-when-the-"
    "record-expires-or-is-purged"
)


def keyframe_store_from_root(root):
    """The owned-keyframe store for a configured observation root, or None.

    Built at the wiring point and held on `app.state`, like the face
    filter, rather than per request -- and built THROUGH THIS ADAPTER
    rather than imported in `main.py`, because the wiring point does not
    import a cartridge. `None` when this Tower serves no object memory at
    all, which is the same condition that makes the routes answer 404.
    """
    if root is None:
        return None
    return KeyframeStore(root)


def build_face_filter(path=None):
    """The filter every served picture passes through.

    Constructed at the wiring point and held on `app.state`, so the ONNX
    session is built once rather than per request -- and constructed
    through this adapter rather than imported there, because `main.py`
    does not import a cartridge.
    """
    return imagery.FaceFilter(path)


def find_observation(store: ObservationStore, observation_id: str):
    """The record with this handle, within retention, or None.

    A linear scan, and that is proportionate: the store is a JSONL file
    whose class docstring already expects it to stay small, and every
    read path here already parses all of it. An index would be the second
    thing to add after SQLite, not before it.

    Retention is NOT bypassed. `all_observations` filters to the clamped
    window by default, so a handle to an expired record resolves to
    nothing -- the same answer a listing gives about it, and the only one
    that keeps retention a promise rather than a default.
    """
    for observation in store.all_observations():
        if observation.observation_id == observation_id:
            return observation
    return None


def frame_is_available(capture_root, face_filter, observation) -> bool:
    """Whether `/frame` could serve this record's context view.

    An EXISTENCE check, not a render: it resolves the path and asks
    whether the filter could run, and decodes nothing. The view route is
    asked once per row on a screen, and paying a YuNet pass per row to
    answer a boolean would make a metadata route cost more than the
    picture it describes.
    """
    if observation is None or capture_root is None:
        return False
    if face_filter is None or not getattr(face_filter, "available", False):
        return False
    try:
        return imagery.frame_path(capture_root, observation) is not None
    except Exception:  # noqa: BLE001
        # Contained, and contained as "no". A metadata route must not
        # 500 because a manifest on disk is malformed, and reporting a
        # context view that cannot be fetched is the direction that puts
        # a broken image in front of a person.
        return False


def build_imagery_view(
    observation, image, *, observation_id: str, frame_available=None
) -> dict:
    """What a client needs to decide whether to show a picture, and what to say.

    Deliberately answerable WITHOUT fetching the bytes. A phone deciding
    between a thumbnail, a caption, and "the memory is kept and the
    picture is not" should not have to download an image to find out
    which -- and the third of those is a sentence a wearer needs to hear,
    not an empty box.

    `available` DESCRIBES WHAT CAN BE SERVED, NOT WHAT `/frame` HOLDS.

    That distinction did not exist until keyframes did, and getting it
    wrong made the whole feature invisible in the one case it was built
    for. The view route used to describe a `/frame` render. A record
    whose recording had been deleted but whose owned crop was sitting on
    disk therefore answered `available: false`, and the shipped iOS
    loader gates on exactly that -- `guard description.available else {
    .noPicture }` -- so it would never have asked `/crop` and the wearer
    would have been told the picture was gone while it was being held for
    them.

    So the view is built from a CROP render, which consults the owned
    keyframe and falls back to the capture frame. `frame_available` is
    additive and says separately whether the wider context view exists,
    for a client choosing between the two.
    """
    box = None
    if observation is not None:
        box = observation.best_bounding_box or observation.bounding_box
    # Read off the result rather than inferred from the route, because
    # the same `/crop` request is answered from the owned keyframe when
    # there is one and from the capture frame when there is not. None on
    # a refusal: nothing served the bytes, because there are none.
    served_by = image.source
    owned = served_by == SOURCE_KEYFRAME
    return {
        "contract": IMAGERY_CONTRACT,
        "observation_id": observation_id,
        "object_class": None if observation is None else observation.object_class,
        "claim": IMAGERY_CLAIM,
        "available": image.available,
        # None when a picture was served. A value, never a sentence: the
        # wording belongs to whoever is speaking to the wearer.
        "reason": image.reason,
        # The one thing this shape exists to be able to say. A record
        # whose imagery has aged out is still a memory; it has simply
        # stopped having a picture.
        "memory_retained": observation is not None,
        "filter": image.filter_label,
        # Follows the source. See the constants: a keyframe was filtered
        # before it was written and there is no unfiltered original of it,
        # which is a stronger statement than the read-time one and must
        # not be reported as the weaker.
        "filter_means": (
            FILTER_MEANS_BEFORE_WRITE if owned else FILTER_MEANS_ON_READ
        ),
        # How many regions the filter actually filled in what was served.
        # Zero means the detector found none, NOT that there were none --
        # it has measured blind spots, and a reader must be able to tell
        # "nothing detected" from "nothing there".
        "regions_filled": image.regions_filled,
        # How much of the record's own box the filter covered, 0.0 to
        # 1.0. Non-zero means part of the SUBJECT is behind a fill, and a
        # client should say so rather than show a black rectangle without
        # comment. It is not always a face: on frame 2708 of the
        # validated capture the filter fired twice on a desk with no
        # person in it, and one of the fills landed on the mouse the
        # record is about.
        "subject_obscured": image.subject_obscured,
        # Where in the picture, in fractions of the frame. The same box
        # `where.bounding_box_normalized` carries, with the same caveat:
        # it is where in a picture, never where in a room.
        "bounding_box_normalized": list(box) if box is not None else None,
        # WHICH STORE ACTUALLY SERVED THESE BYTES, and therefore whose
        # retention governs them. Not a constant any more, and it must
        # not be: a crop served from this cartridge's own keyframe
        # expires with the record and is deleted by `purge()`, while a
        # frame served out of `data/captures/` may vanish on a schedule
        # nothing here sets. Asserting one answer for both would be
        # telling a client the wrong thing half the time.
        #
        # `capture-side` remains the answer for every whole frame and for
        # every record with no keyframe, which is every record written
        # before keyframes existed -- so a decoder that only knows that
        # value keeps working.
        "imagery_retention": (
            RETENTION_OBJECT_MEMORY if owned else RETENTION_CAPTURE_SIDE
        ),
        # The same thing in words, so a payload read on its own says what
        # it means rather than requiring the contract document.
        "imagery_retention_means": (
            OBJECT_MEMORY_RETENTION_MEANS if owned else CAPTURE_RETENTION_MEANS
        ),
        # Which store, named. Additive, and the one a diagnostic wants:
        # "capture-frame" or "object-memory-keyframe".
        "imagery_source": served_by,
        # Whether the WIDER view exists, separately from `available`.
        #
        # A client picks between the object and its context, and after
        # keyframes those two no longer stand or fall together: an owned
        # crop outlives the recording, and the context frame does not.
        # `null` where it was not computed -- on a refusal detail from a
        # binary route, where the client already knows which one it
        # asked for.
        "frame_available": frame_available,
    }


def render_imagery(
    store: ObservationStore,
    observation_id: str,
    *,
    capture_root,
    face_filter,
    crop: bool,
    keyframes=None,
):
    """The observation and its picture, or the observation and the reason.

    Returns `(observation, Imagery)`. `observation` is None only when the
    handle matched nothing within retention, which is the one case a
    caller answers differently from every other.

    `keyframes` is optional and only ever consulted for a CROP. It is
    checked AFTER retention -- `find_observation` has already refused an
    expired handle -- so knowing an id is not a back door to an owned
    picture any more than it is to a capture frame. `prune_expired`
    deletes the keyframe as well, so the two agree even if a caller
    reaches this with a stale handle in the window between them.
    """
    observation = find_observation(store, observation_id)
    if observation is None:
        return None, imagery.Imagery(None, NOT_FOUND)
    return observation, imagery.render(
        capture_root, observation, face_filter, crop=crop, keyframes=keyframes
    )
