"""Object Memory's observations, over HTTP. Read-only.

Named after the question rather than the cartridge, exactly as
`geometry.py` is: a route file called `object_memory.py` would put a
cartridge's name into `main.py`'s import list, which is the boundary
`test_shared_code_does_not_import_a_cartridge` exists to hold. The
cartridge is reached only through `tower/results/object_memory.py`, the
adapter named after it.

Both handlers are declared `def` rather than `async def` on purpose.
FastAPI runs a sync endpoint in its threadpool, which keeps the JSONL
read off the event loop with no executor of our own -- the same reason
`tower/routes/geometry.py` gives for the same choice. An `async def`
here would block the frame path behind a disk read.

NOTHING HERE MUTATES.

`ObservationStore` can purge and can prune. This module reaches neither,
and `test_no_wire_module_can_reach_purge_or_prune` fails if it ever
does. Deletion stays with `scripts/object_query.py --purge-all`, where a
human types it.

THE IMAGERY ROUTES SERVE PIXELS, AND THAT IS NEW.

Until now this cartridge exposed a POINTER to a frame and never the
frame. Three rules hold that step:

  * every picture passes a face filter on the way out, and a Tower whose
    filter cannot run serves NOTHING rather than an unfiltered
    first-person frame;
  * the label says `display-filter/...`, because the stored frame is
    unchanged and calling this a privacy transformation would be false;
  * a record whose imagery has aged out answers "the memory is kept and
    the picture is not", which is the whole reason the shape exists.

`Cache-Control: no-store` on both binary routes. This is sensitive
first-person imagery on a LAN-local origin, and a proxy or a browser
holding a copy is a second store nobody chose.

RETENTION IS NARROWABLE, NEVER WIDENABLE.

`retention_days` is a request, not an authority. It travels into the
store, which clamps every read to min(persisted, requested). A caller
asking for 3650 days against a store written under the 30-day default
still gets 30 days, and the response says so in `retention.clamped`
rather than pretending the question was honoured. The clamp is the
store's, not this route's -- this file consumes the property and adds
nothing of its own that could weaken it.
"""

from fastapi import APIRouter, HTTPException, Query, Request, Response

from tower.results.object_memory import (
    frame_is_available,
    FILTER_UNAVAILABLE,
    IMAGERY_EXPIRED,
    NO_CAPTURE_ROOT,
    NO_FRAME_REFERENCE,
    NOT_FOUND,
    UNREADABLE,
    build_face_filter,
    build_imagery_view,
    build_last_seen,
    build_observations,
    render_imagery,
    store_from_root,
)

router = APIRouter()


def _recorded_classes(request: Request):
    """What this Tower will actually record, or None to mean the safe default.

    Read from `app.state` rather than imported, because it depends on
    whether a semantic verifier is configured -- and the route is not
    allowed to know what a verifier is. `main.py` puts the answer here
    from the same `Settings` object the producer's argv is built from, so
    the read surface and the producer cannot disagree about it any more
    than they can disagree about where the store lives.
    """
    return getattr(request.app.state, "object_memory_recorded_classes", None)


def _store(request: Request, retention_days: float | None):
    root = getattr(request.app.state, "object_memory_root", None)
    if root is None:
        # Unset means this Tower serves no object memory at all, which is
        # a different claim from "nothing has been observed". A 404 here
        # is about configuration; it is never the answer to a query about
        # a class, which is answered with `observed: false` instead.
        raise HTTPException(
            status_code=404, detail="no object memory root is configured"
        )
    return store_from_root(root, retention_days=retention_days)


# `ge=0` rather than an unbounded float: a negative window is meaningless
# and the store raises ValueError on it, which would surface as a 500. 0
# is allowed and means "no limit of my own" -- which still gets clamped
# to the persisted window, so it is not a back door.
_RETENTION_DAYS = Query(
    default=None,
    ge=0,
    description=(
        "Narrow the window this read may see. Clamped to "
        "min(persisted, requested); it can only ever serve less."
    ),
)


# `ge=0`: a negative moment is meaningless. It is a Tower-receipt epoch
# second (`time.time()`), the same clock a session's `started_at` is on, so
# a client asks "what did THIS recording remember" by passing the session's
# own start. Absent means the whole store, which is the historical view.
_SINCE = Query(
    default=None,
    ge=0,
    description=(
        "Only observations the Tower WROTE at or after this moment "
        "(recorded_at >= since). Pass a session's started_at to get just "
        "that recording's memories; omit for the whole store. Separates a "
        "recording's new memories from history so an empty recording cannot "
        "borrow history's."
    ),
)


@router.get("/object-memory/observations")
def observations(
    request: Request,
    object_class: str | None = Query(default=None),
    since: float | None = _SINCE,
    retention_days: float | None = _RETENTION_DAYS,
) -> dict:
    return build_observations(
        _store(request, retention_days),
        object_class=object_class,
        since=since,
        requested_retention_days=retention_days,
        recorded_classes=_recorded_classes(request),
    )


@router.get("/object-memory/last-seen/{object_class}")
def last_seen(
    object_class: str,
    request: Request,
    retention_days: float | None = _RETENTION_DAYS,
) -> dict:
    return build_last_seen(
        _store(request, retention_days),
        object_class,
        requested_retention_days=retention_days,
        recorded_classes=_recorded_classes(request),
    )


# How a refusal maps onto a status code.
#
# A picture that has aged out is 410 GONE, and that is the one carrying
# meaning: it says the resource existed and does not now, which is
# exactly what capture-side retention did. Everything else is either a
# handle that matched nothing (404) or this Tower being unable to serve
# imagery at all right now (503) -- a configuration answer, not a claim
# about the record.
#
# The reason VALUE is on the body in every case, and a client should
# switch on that rather than on the code.
_IMAGERY_STATUS = {
    NOT_FOUND: 404,
    NO_FRAME_REFERENCE: 404,
    IMAGERY_EXPIRED: 410,
    NO_CAPTURE_ROOT: 503,
    FILTER_UNAVAILABLE: 503,
    UNREADABLE: 503,
}

# Never cached. Sensitive first-person imagery on a LAN-local origin.
_NO_STORE = {"Cache-Control": "no-store"}

# The stand-in for an app that was built without a filter -- which is
# most of this repository's tests. The failure has to be a REFUSAL, never
# an AttributeError and never a raw frame.
#
# `path=""` means "no model", explicitly, since `imagery.FaceFilter` grew
# a blank check. It did not always: `Path("")` is `Path(".")` and
# `Path(".").exists()` is True, so this reported itself AVAILABLE and
# refused only because `cv2.FaceDetectorYN.create(".")` happened to
# raise. Every test that asserted a refusal here was passing for that
# reason rather than for this one.
_REFUSING_FILTER = build_face_filter(path="")


def _face_filter(request: Request):
    return getattr(request.app.state, "object_memory_face_filter", None) or (
        _REFUSING_FILTER
    )


def _capture_root(request: Request):
    return getattr(request.app.state, "capture_root", None)


def _keyframes(request: Request):
    """The crops this cartridge owns, or None on a Tower that has none.

    Read off `app.state` for the same reason the store root and the face
    filter are: this file is a route and may not construct a cartridge's
    objects. `main.py` builds it through the adapter from the same
    `observation_root` the store comes from, so the two cannot point at
    different directories.

    None is not a failure. It means "look in the capture tree", which is
    what every request did before keyframes existed.
    """
    return getattr(request.app.state, "object_memory_keyframes", None)


def _imagery(request: Request, observation_id: str, *, crop: bool):
    return render_imagery(
        _store(request, None),
        observation_id,
        capture_root=_capture_root(request),
        face_filter=_face_filter(request),
        keyframes=_keyframes(request),
        crop=crop,
    )


def _refuse(observation, image, observation_id: str):
    raise HTTPException(
        status_code=_IMAGERY_STATUS.get(image.reason, 503),
        detail=build_imagery_view(observation, image, observation_id=observation_id),
    )


@router.get("/object-memory/observations/{observation_id}/imagery")
def imagery_view(observation_id: str, request: Request) -> dict:
    """Whether there is a picture, and what may be said about it.

    Answerable without downloading anything, which is the point: a phone
    deciding between a thumbnail, a caption and "the memory is kept and
    the picture is not" should not have to fetch an image to find out
    which.

    200 even when there is no picture. The resource -- what this
    cartridge knows about the imagery behind a record -- exists either
    way, and a 404 here would read as "no such memory". A handle that
    matched nothing is the exception, and is a real 404.
    """
    # A CROP render, not a frame one, and the difference is the whole
    # reason keyframes are reachable at all. See `build_imagery_view`:
    # describing a `/frame` render here reported `available: false` for
    # every record whose recording had been deleted, including the ones
    # holding an owned crop -- and the shipped iOS loader gates on that
    # boolean and would never have asked for the crop.
    observation, image = _imagery(request, observation_id, crop=True)
    if observation is None:
        raise HTTPException(
            status_code=404,
            detail=build_imagery_view(None, image, observation_id=observation_id),
        )
    return build_imagery_view(
        observation,
        image,
        observation_id=observation_id,
        # An existence check, not a second render. Answers "is there also
        # a context view", which is a different question from "is there a
        # picture" now that the two have different lifetimes.
        frame_available=frame_is_available(
            _capture_root(request), _face_filter(request), observation
        ),
    )


@router.get("/object-memory/observations/{observation_id}/frame")
def frame(observation_id: str, request: Request):
    """The whole frame this record was derived from, filtered.

    The CONTEXT, not the object. Where the wearer was and what else was
    in view is most of what makes a small crop recognisable, and the
    published evidence on memory aids is that context is what people
    actually use.
    """
    observation, image = _imagery(request, observation_id, crop=False)
    if not image.available:
        _refuse(observation, image, observation_id)
    return Response(
        content=image.image_bytes, media_type="image/jpeg", headers=_NO_STORE
    )


@router.get("/object-memory/observations/{observation_id}/crop")
def crop(observation_id: str, request: Request):
    """The object itself, padded, filtered.

    Padded rather than tight: a 3%-of-frame box cropped exactly is
    unreadable, and the surroundings are most of what tells a person
    whether the label is right.

    THE ONE ROUTE WITH TWO SOURCES. It prefers the crop this cartridge
    OWNS -- filtered before it was written, kept under the observation
    root, and deleted when the record expires -- and falls back to
    cropping the capture frame when there is no owned one. So this route
    keeps answering after the whole capture tree is gone, which `/frame`
    cannot and should not: the full-frame context view belongs to the
    recording, and when the recording is deleted it is honestly 410.
    """
    observation, image = _imagery(request, observation_id, crop=True)
    if not image.available:
        _refuse(observation, image, observation_id)
    return Response(
        content=image.image_bytes, media_type="image/jpeg", headers=_NO_STORE
    )
