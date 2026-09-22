"""What this Tower has read, over HTTP. Read-only, plus a Start/Stop.

Named after the question rather than the cartridge, exactly as
`geometry.py` and `observations.py` are: a route file named after the
cartridge would put its name into `main.py`'s import list and into
`main.py`'s raw text, which two tests forbid.

WHY HTTP AND NOT THE RESULT SOCKET

`tower/routes/ws.py` gives the result sender and the frame path one
shared `asyncio.Lock`, so a large result send directly delays the next
`frame_result`. Document text is the largest thing this platform could
put on that socket and the most sensitive: `IOS-to-Tower.md` 3.2 already
requires that "the list carries a character count, not the text... Full
text is fetched when a person opens one", which is the manifest/segment
split World Builder's geometry uses, arrived at from the privacy side
rather than the size side.

So: the LIST is here and carries no text. ONE DOCUMENT is here and
carries its pages. The session's progress is on the result channel, where
a push is worth having.

Every handler is `def` rather than `async def` on purpose. FastAPI runs a
sync endpoint in its threadpool, which keeps a full journal parse off the
event loop with no executor of our own -- the same reason
`observations.py` gives for the same choice.

NOTHING HERE MUTATES A STORED DOCUMENT.

`DocumentStore` can `purge()` and `prune_expired()`. Neither is reachable
from this module and neither may become reachable: an unauthenticated
HTTP endpoint that erases what a wearer read is not a feature, and
06-PRIVACY-DATA.md's real deletion stays where a human types it --
`scripts/document_query.py --purge`.

The Start/Stop handlers below control a CAPTURE SESSION, which is a
different thing: they decide whether this Tower is currently looking for
documents. They cannot delete one, cannot change retention, and cannot
reach the store at all.
"""

from fastapi import APIRouter, HTTPException, Query, Request

from tower.results.document_memory import (
    documents_around,
    one_document,
    recent_documents,
    search_documents,
    session_view,
    store_from_root,
)

router = APIRouter()

# `ge=0` rather than an unbounded float: a negative window is meaningless
# and would surface as a 500. 0 means "no limit of my own".
_RETENTION_DAYS = Query(
    default=None,
    ge=0,
    description=(
        "Narrow what this read may see, in days. It can only ever serve "
        "less; note that DocumentStore records no writer-side window, so "
        "the response cannot report one"
    ),
)


def _root(request: Request):
    root = getattr(request.app.state, "document_root", None)
    if root is None:
        # A claim about CONFIGURATION. It is never the answer to "what did
        # I read this morning", which is answered with
        # `answer: "no_observation"` -- the same distinction
        # `observations.py` draws between a 404 and `observed: false`.
        raise HTTPException(
            status_code=404,
            detail=(
                "Document Memory is switched off on this Tower "
                "(TOWER_DOCUMENT_ENABLED is off), so no document root is "
                "configured (TOWER_DOCUMENT_ROOT is unset)"
            ),
        )
    return root


def _store(request: Request, retention_days):
    # The Tower's own window when the caller names none. Until 2026-09-07
    # a read with no `retention_days` saw FOREVER while the session wrote
    # under thirty days, so an expired-but-not-yet-pruned record was
    # served. A caller can still narrow; nothing can widen.
    configured = getattr(request.app.state, "document_retention_days", None)
    if retention_days is None:
        retention_days = configured
    elif configured is not None:
        retention_days = min(retention_days, configured)
    return store_from_root(_root(request), retention_days=retention_days)


@router.get("/documents")
def documents(
    request: Request,
    limit: int = Query(default=10, ge=1, le=200),
    retention_days: float | None = _RETENTION_DAYS,
) -> dict:
    """The most recent documents, newest first, WITHOUT their text.

    `limit` is bounded at 200 rather than unbounded, because this handler
    parses the whole journal and serialises what it returns, and an
    unbounded limit is a remote party choosing how much memory this
    process uses.
    """
    return recent_documents(
        _store(request, retention_days),
        limit=limit,
        requested_days=retention_days,
    )


@router.get("/documents/search")
def search(
    request: Request,
    text: str = Query(min_length=1),
    limit: int = Query(default=5, ge=1, le=50),
    retention_days: float | None = _RETENTION_DAYS,
) -> dict:
    """Literal term matching. NOT semantic, and the response says so.

    Declared before `/documents/{document_id}` deliberately: FastAPI
    matches routes in declaration order, and a path parameter registered
    first would swallow `/documents/search` as a document whose id is
    "search".
    """
    return search_documents(
        _store(request, retention_days),
        text=text,
        limit=limit,
        requested_days=retention_days,
    )


@router.get("/documents/around")
def around(
    request: Request,
    at: float = Query(description="Centre of the window, unix seconds"),
    window_seconds: float = Query(default=900.0, ge=1.0, le=86400.0),
    limit: int = Query(default=50, ge=1, le=200),
    retention_days: float | None = _RETENTION_DAYS,
) -> dict:
    """Documents observed within a window of an instant.

    A RANGE, not an instant. "This morning" and "around lunch" are
    approximate, and answering them exactly answers a different question.
    """
    return documents_around(
        _store(request, retention_days),
        when=at,
        window_seconds=window_seconds,
        limit=limit,
        requested_days=retention_days,
    )


@router.get("/documents/{document_id}")
def document(
    request: Request,
    document_id: str,
    retention_days: float | None = _RETENTION_DAYS,
) -> dict:
    """One document, with its pages and their text.

    The only route that carries text.

    404 here IS about the resource: an id that names nothing is a client
    asking about a document this Tower has never held, which is different
    from a query that matched nothing. The list routes never 404 for an
    empty result.
    """
    payload = one_document(
        _store(request, retention_days),
        document_id,
        requested_days=retention_days,
    )
    if payload is None:
        raise HTTPException(
            status_code=404, detail=f"no document {document_id!r} in this memory"
        )
    return payload


# -- the capture session ------------------------------------------------


def _session(request: Request):
    live = getattr(request.app.state, "live_cartridges", None)
    session = None if live is None else live.document
    if session is None:
        # The runtime's own reason when it has one (the OCR extra is not
        # installed, most likely), else the configuration reading. Both
        # name TOWER_DOCUMENT_CAPTURE: iOS keys its "no capture session"
        # outcome on that substring.
        reason = None if live is None else live.document_unavailable_reason
        raise HTTPException(
            status_code=404,
            detail=(
                reason
                or (
                    "this Tower runs no document capture session "
                    "(TOWER_DOCUMENT_CAPTURE is off, or TOWER_DOCUMENT_ROOT "
                    "is unset). Documents recorded elsewhere are still served"
                )
            ),
        )
    return session


@router.get("/documents-session")
def session_status(request: Request) -> dict:
    """What the capture session is doing. Identical to the socket's half.

    Named with a hyphen rather than nested under `/documents/` because
    `/documents/{document_id}` would otherwise match it, and a route that
    only works until somebody records a document called "session" is a
    bug waiting for a coincidence.
    """
    return session_view(_session(request))


@router.post("/documents-session/start")
def start(request: Request) -> dict:
    """Begin watching the stream for documents.

    Returns the status rather than an acknowledgement: the interesting
    answer is `state: "starting"` and a `loading_seconds` a caller can
    poll, because the OCR reader takes about five seconds to construct
    and a bare 204 would make the caller guess.
    """
    session = _session(request)
    session.start()
    return session_view(session)


@router.post("/documents-session/pause")
def pause(request: Request) -> dict:
    """Stop watching. Any dwell in progress is flushed, not discarded.

    A wearer still reading when a session pauses has read something, and
    throwing that away would lose a real observation to a UI action.
    """
    session = _session(request)
    session.pause()
    return session_view(session)


@router.post("/documents-session/resume")
def resume(request: Request) -> dict:
    session = _session(request)
    session.resume()
    return session_view(session)


@router.post("/documents-session/stop")
def stop(request: Request) -> dict:
    """End the session. What was already recorded stays recorded.

    Unlike Scene Understanding's Stop, this one does NOT discard
    anything, and the asymmetry is the difference between the two
    cartridges rather than an inconsistency. Scene Understanding's state
    is a claim about the present and expires the moment it stops looking.
    A document memory is a record of the past and is exactly as true
    after the session ends.
    """
    session = _session(request)
    session.stop()
    return session_view(session)
