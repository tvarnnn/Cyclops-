"""Library and session transport adapter for Document Memory.

Named after its cartridge, for the reason `world_builder.py`,
`world_builder_geometry.py` and `object_memory.py` are: an adapter named
after one cartridge cannot leak that cartridge's assumptions into the
next, because the next one gets its own file.

THIS FILE READS. IT DOES NOT DELETE, AND IT DOES NOT OCR.

`DocumentStore` can `purge()` and `prune_expired()`. Neither is reachable
from here and neither may become reachable -- an unauthenticated HTTP
endpoint that erases what a wearer read is not a feature. Deletion stays
with `scripts/document_query.py --purge`, where a human types it.
`DocumentMemoryEngine` can `observe()`. That is not reachable from here
either, and `test_the_result_channel_never_writes` fails on a call named
`observe` anywhere under `tower/results/`.

THE THREE ANSWERS, AND WHY TWO WOULD BE A LIE

`IOS-to-Tower.md` 3.5 is the sharpest requirement on this cartridge and
the easiest to get wrong:

    matched        documents were found
    notFound       the memory was searched and nothing matched
    noObservation  the memory holds nothing covering what was asked

Collapsing the third into the second lets a gap in what the glasses
happened to see read as a statement about the world. That distinction is
not decoration here -- it is this cartridge's ACTUAL STATE. On 9,199
frames of real first-person footage the page detector fired six times and
was wrong all six, and after `MIN_ROW_TRANSITIONS` was re-derived against
those same frames it fires zero times. A Tower asked "show me the parking
notice" today answers `no_observation`, and that is the truthful answer:
nothing has ever been recorded, and that is a fact about this platform's
camera, not about parking notices.

`RECORDING_LIMITATIONS` carries that on every response. A client that
renders an empty library as "no documents yet" would be inviting a person
to wait for something that is not coming.

THE LIST CARRIES A CHARACTER COUNT, NOT THE TEXT

`IOS-to-Tower.md` 3.2: "The list carries a character count, not the text,
so a list of documents is not also a bulk transfer of every document's
contents onto the phone. Full text is fetched when a person opens one."

That is the manifest/segment split `world_builder_geometry.py` already
uses, applied to the most sensitive data this platform handles.
`text_availability` is the typed form of iOS's `unknown` / `notReadable`
/ `extracted(characterCount:)`, and `not_readable` is a REAL ANSWER: "we
looked and found no readable text" is a different fact from "we never
looked", and the store records both.

NAMES THAT ARE PART OF THE CONTRACT

WHY THE CAVEATS ARE AT THE ENVELOPE AND NOT ON EVERY RECORD

`record_notes` carries, once, the five sentences that used to be repeated
on every document: the "in view, not read" qualification, the summary's
provenance, the clock, the capture-side imagery lifetime, and the
joinability of the frame reference. They are constants, and a
200-document listing was 488 KB with two thirds of it the same sentences
two hundred times.

They are hoisted, never dropped. Each is a caveat a client must be able
to render, and deleting a caveat to save bytes is the one saving this
contract may not make.

The duration field is `observed_seconds`, and it must never be renamed to
anything built on the word "viewing". `IOS-to-Tower.md` 3.3 asks for that
by name and gives the reason: "appearing in the camera does not establish
that the wearer looked at it, noticed it, or read it"; a field named for
viewing claims all three. iOS renders it as "In view 45s", never "viewed
for" or "read for". The literal banned spelling is not written out here,
because the vocabulary test in `tests/test_result_channel_isolation.py`
greps every string in this package for it -- docstrings included -- and a
rule that exempted the file explaining the rule is how the word gets back
in. That is also why this paragraph does not quote the test's own name.
"""

from tower.document_memory.records import TIME_BASIS
from tower.results.contracts import DOCUMENT_MEMORY_LIBRARY_CONTRACT
from tower.document_memory.retrieval import MIN_SCORE, DocumentMemory
from tower.document_memory.store import DocumentStore

# Opaque and dated, in the style of `object_memory.observations/2026-08-26`.
# Compared for equality only: never parsed, never ordered, never used to
# infer that one contract is newer than another.
#
# Defined in `tower/results/contracts.py` and re-exported here, so the
# capability declaration can name it without importing this adapter --
# `registry.py` must stay cartridge-blind.
LIBRARY_CONTRACT = DOCUMENT_MEMORY_LIBRARY_CONTRACT

# The three answers, as values a decoder can switch on.
ANSWER_MATCHED = "matched"
ANSWER_NOT_FOUND = "not_found"
ANSWER_NO_OBSERVATION = "no_observation"
ANSWERS = (ANSWER_MATCHED, ANSWER_NOT_FOUND, ANSWER_NO_OBSERVATION)

# The typed form of iOS's DocumentTextAvailability.
TEXT_UNKNOWN = "unknown"
TEXT_NOT_READABLE = "not_readable"
TEXT_EXTRACTED = "extracted"

# "OCRED", not "read", and the correction matters more than it looks.
#
# This file spends a paragraph explaining that the duration field must
# never be named for viewing, because "appearing in the camera does not
# establish that the wearer looked at it, noticed it, or read it" -- and
# then the top-level claim, the value a decoder is invited to switch on,
# said the document WAS READ. The intended reading was "read by OCR".
# Nothing in the payload said so, and the field five keys below is
# specifically about the wearer.
#
# Corrected before any consumer existed, which is why the contract
# identifier did not move: an identifier is bumped to protect decoders
# that exist, and none had ever decoded the earlier shape.
DOCUMENT_CLAIM = "a-page-was-in-view-and-was-ocred"
# Since 2026-09-07 a record IS joined across sightings, by one rule: a
# later dwell whose first readable page carries the same words AND looks
# the same (`document_memory/identity.py`) becomes a sighting of the
# record rather than a new one. Anything weaker stays a separate record.
# The earlier value, "no-document-identity-across-sightings", was true
# of a cartridge that made no such join; it is not true of this one,
# and the contract identifiers moved with it.
IDENTITY_SCOPE = "same-page-by-text-and-look-within-library"
ABSENCE_MEANING = "not-recorded-by-this-cartridge"

# What happened to the pixels, in this platform's terms and in iOS's.
#
# The earlier single constant was `"raw-ephemeral-not-served"`, and it
# was wrong twice. It is not one of the three states `IOS-to-Tower.md` 5
# defines, so a decoder falls through to unknown -- the safe direction,
# but a mapping that never lands. And it was a module constant, so a
# library that kept every page image and one that kept none emitted the
# same string, while a kept image is raw and PERSISTED, which is the
# opposite of ephemeral.
IMAGERY_NONE_RETAINED = "none-retained"
IMAGERY_RAW_PERSISTED = "raw-persisted"

# The strictest iOS state that applies, named in iOS's own vocabulary so
# the mapping is the Tower's decision rather than the phone's guess. It
# is `rawEphemeral` in both cases and not `redacted` in either, because
# this platform performs no redaction at all.
IMAGERY_IOS_STATE = "rawEphemeral"

IMAGERY_NOTE = (
    "this platform performs no redaction. No route serves an image. "
    "Where retains_raw_imagery is true, an unredacted page image exists "
    "on this Tower's disk under capture-side retention and is not "
    "reachable over any wire"
)

# What `imagery_retention` means, since it is this cartridge's own word
# and iOS has no channel through which it could verify it.
IMAGERY_RETENTION_NOTE = (
    "capture-side: the frames this record points at live in the capture "
    "store, whose lifetime this cartridge neither sets nor enforces. "
    "Purging every document here leaves that imagery exactly where it is"
)

# The closed set `privacy_tags` draws from, published so a client can pin
# it rather than discover it one record at a time.
PRIVACY_TAG_VOCABULARY = ("document-text", "first-person")

RETRIEVAL_KINDS = ("recent", "text", "observed_within")

# The sentences that are the same on every record, carried ONCE.
#
# Measured before this: a document summary was 2,351 bytes of which 1,603
# were these constants, so a 200-document listing was 488 KB with ~313 KB
# of it the same five sentences two hundred times, against 12 KB of
# actual titles. That is the argument the scene payload already made
# against itself when it cut `refused_relations` to a first sentence and
# a pointer; the document listing had never had that pass.
#
# They are hoisted rather than deleted. Every one of them is a caveat a
# client must be able to render -- "in view", not "read"; the summary is
# an excerpt; the capture pointer is durable and joinable -- and dropping
# them to save bytes would be dropping the disclosure, which is the one
# saving this contract may not make.
RECORD_NOTES = {
    "observed_seconds": (
        "how long the region was in view. This platform cannot establish "
        "that the wearer looked at it, noticed it, or read it"
    ),
    "summary_withheld": (
        "the stored summary is the document's first forty words verbatim "
        "-- an excerpt, not a paraphrase -- and is served with the "
        "document, never in a listing. What a listing does carry is a "
        "clipped title, and a search result additionally carries a "
        "bounded snippet around the matched term as evidence; both are "
        "capped, and the caps are published beside them. The rule is that "
        "a listing must not become a bulk transfer of what a wearer read, "
        "not that it carries no verbatim characters at all"
    ),
    "timing": (
        "tower-receipt time: when this Tower received the frames, never "
        "when the glasses captured them. There is no capture timestamp "
        "anywhere on this wire"
    ),
    "imagery_retention": (
        "capture-side: the frames this record points at live in the "
        "capture store, whose lifetime this cartridge neither sets nor "
        "enforces. Purging every document here leaves that imagery "
        "exactly where it is"
    ),
    "joinable": (
        "capture_id, page_source_seqs and observed_at together locate "
        "this reading in a recording. The link is durable across "
        "sessions, unlike anything Scene Understanding publishes"
    ),
}

# How much verbatim text a SEARCH result may carry per match.
#
# `retrieval._snippet` produces 160 characters around the matched term,
# and a review measured what that adds up to: with a 90-character title
# beside it, 252 verbatim characters per match, at a limit of 50 that is
# ~12.6 KB of document text -- delivered in the same object as a string
# promising the summary is withheld so "a listing cannot become a bulk
# transfer of what a wearer read".
#
# The snippet is not decoration and is not dropped: a match with no
# evidence is a number a client has to trust, and `IOS-to-Tower.md`
# models `DocumentQueryEvidence` for exactly this. It is bounded instead,
# to a phrase rather than a paragraph, and the payload now says what it
# serves rather than claiming it serves nothing.
SNIPPET_MAX_CHARS = 48

# And how much of a title. A title is a label and iOS asks for it in the
# list (3.1) knowing it is lifted from the document's own first line --
# but 90 characters times a 200-document listing is 18 KB of verbatim
# first lines for a caller that asked no question at all.
TITLE_MAX_CHARS_ON_WIRE = 60

# What this cartridge cannot do, as data rather than as a document
# somebody has to have read. Every response carries it.
RECORDING_LIMITATIONS = (
    {
        "limitation": "detection-rate",
        "detail": (
            "the contour-quad page detector this cartridge shipped with "
            "fired 6 times on 9,199 frames of real first-person footage "
            "and every one was a false positive (a venetian blind and a "
            "backlit keyboard). It was replaced on 2026-09-07 by a "
            "steadiness gate plus the OCR engine's own text detector, "
            "which needs no page outline. That detector found zero "
            "confident text boxes across 4,599 sampled real frames, so "
            "an empty library on this platform's existing footage is "
            "still the expected result, not a sign that nothing was read"
        ),
    },
    {
        "limitation": "no-validated-positive",
        "detail": (
            "no capture on this platform has contained a sheet of paper: "
            "a visual review of one sharp frame from each of 71 captures "
            "and a text-detector sweep of every tenth frame of 97 "
            "captures found none. The pipeline has been validated on "
            "rendered pages with known text, never on a physical page "
            "through the glasses, so the premise is untested rather than "
            "disproved"
        ),
    },
    {
        "limitation": "resolution",
        "detail": (
            "at the 360x640 the glasses deliver, EasyOCR returned zero "
            "dictionary words across 919 sampled real frames of screen "
            "text at median confidence 0.056. On rendered pages at that "
            "geometry word recall is 0.98 when the page fills the frame, "
            "0.74 at 60% of the frame height and 0.02 at 40%; at 504x896 "
            "it is 1.00 / 0.98 / 0.66. A page must be held close, or the "
            "stream must be raised a rung"
        ),
    },
    {
        "limitation": "resolution-remedy-is-not-a-fix",
        "detail": (
            "raising the stream to 720x1280 is measured to break World "
            "Builder's tracking (73% of frames rejected as blurred), so "
            "the resolution a session runs at is a cross-cartridge choice "
            "the phone makes, not this cartridge. 504x896 is the "
            "unmeasured middle rung. A false positive of the text "
            "detector -- a keyboard, a screen -- reaches OCR and is kept "
            "as a page that is not readable, which the record says"
        ),
    },
)

# When the figures above were measured, and on what.
#
# Asserted in the present tense they read as current state, and this
# platform's corpus grows continuously -- so a rate measured on a corpus
# says nothing about the frames recorded since.
RECORDING_MEASUREMENT = {
    "measured_at": "2026-09-06",
    "corpus_frames": 45594,
    "corpus_captures": 97,
    "is_current": False,
    "note": (
        "the corpus on this host grows with every session. These figures "
        "describe the frames they were measured on: the text-detector "
        "sweep covered every tenth frame, the recall ladder is synthetic"
    ),
}


def store_from_root(root, *, retention_days: float | None = None, clock=None):
    """Construct a `DocumentStore` for a configured root.

    Lives here, not in `tower/routes/documents.py`, so the route imports
    only this adapter -- the same rule that keeps
    `world_builder_geometry.store_from_root` and
    `object_memory.store_from_root` where they are.

    `retention_days` is accepted and NARROWS ONLY. It reaches the store
    as `retention_seconds`, and since 2026-08-27 the store genuinely
    filters reads to that window -- `DocumentStore.read_all` explains why
    that sentence used to be false and what it cost.

    The asymmetry with Object Memory is real and a caller must not be
    misled about it. `ObservationStore` persists the window it was
    written under and clamps to `min(persisted, requested)`.
    `DocumentStore` persists no such manifest, so it cannot clamp: a
    reader asking for a WIDER window than the writer used gets the wider
    window, and will simply see nothing the writer already pruned.
    `_retention_view` says exactly that rather than implying a guarantee
    that is not there.
    """
    seconds = None
    if retention_days is not None and retention_days > 0:
        seconds = retention_days * 86400.0
    return DocumentStore(root, retention_seconds=seconds, clock=clock)


def _retention_view(requested_days: float | None) -> dict:
    return {
        "requested_days": requested_days,
        # Honestly unknown, and said so. The store writes no retention
        # manifest, so this read cannot discover the window the WRITER
        # used. Reporting the requested value here would let a reader
        # believe it had learned the producer's promise.
        "writer_window_days": None,
        "writer_window_unavailable_reason": (
            "DocumentStore persists no retention manifest, so a reader "
            "cannot learn the window its writer used. A request here "
            "narrows this read and can never widen what was kept"
        ),
        "policy": "a reader may narrow this read; it cannot widen it",
    }


def _clip(text, limit: int):
    """Bound a verbatim excerpt, and say that it was bounded.

    An ellipsis rather than a silent cut: a client rendering a clipped
    title as a whole one is showing something the document does not say.
    """
    if text is None:
        return None
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _text_availability(document) -> dict:
    """Three states, and the character count that goes with one of them.

    `not_readable` is reachable and is a real answer: a page whose OCR
    returned nothing is still recorded, because "we looked and found no
    readable text" is a different fact from "we never looked".
    """
    characters = len(document.text)
    if not document.pages:
        return {"state": TEXT_UNKNOWN, "character_count": None}
    if characters == 0:
        return {"state": TEXT_NOT_READABLE, "character_count": 0}
    return {"state": TEXT_EXTRACTED, "character_count": characters}


def _provenance(document) -> dict:
    """Where this record came from, as a pointer into a recording.

    Nested under `kind: "frame-reference"` for the reason
    `object_memory._where` gives: at the top level a sequence number
    reads as a position, and under a frame reference it reads as what it
    is -- where in a recording, not where in a room.

    `page_source_seqs` is the sequence number of each frame that was
    actually OCR'd, at most two per document by construction. The other
    frames of the dwell contributed to `frames_considered` and left no
    per-frame trace, which is why both numbers are here.

    `capture_id` is UNVALIDATED and says so. It is the basename of the
    capture directory the session was following; nothing checks that such
    a capture still exists, and a client must not read it as a guarantee
    that the frames are still on disk.
    """
    return {
        "kind": "frame-reference",
        # Reserved, never populated. Document Memory does not know where
        # anything is in a room, and `test_document_memory_does_not_
        # import_another_cartridge` is what keeps it from inventing one.
        "spatial_ref": None,
        "capture_id": document.capture_id,
        "capture_id_validated": False,
        "page_source_seqs": [
            page.source_seq for page in document.pages if page.source_seq is not None
        ],
        "pages_without_source_seq": sum(
            1 for page in document.pages if page.source_seq is None
        ),
        "frames_considered": document.frames_considered,
        "frames_ocred": document.frames_ocred,
        "world_id": document.world_id,
        "world_session_id": document.world_session_id,
        "imagery_retention": "capture-side",
        # Said out loud rather than left for a reviewer to notice. This
        # block IS joinable: a capture id, frame sequence numbers and a
        # timestamp let a holder align this reading with a recording on
        # disk. That is what provenance is for and it is why
        # `IOS-to-Tower.md` 3.8 asks for it -- but it is a durable link
        # across sessions, which is exactly what Scene Understanding's
        # payload refuses to hand anyone, and the two cartridges differ
        # here on purpose. A document is a record; a scene is not.
        "joinable": True,
    }


def _timing(document) -> dict:
    """How the clock that stamped this record was obtained.

    Three states, never collapsed. `assumed-interval` means no frame
    carried a receipt time and a synthetic clock was advanced instead --
    a duration derived that way is a reconstruction, and a client that
    rendered it identically to a measured one would be overclaiming.
    """
    return {
        "time_basis": TIME_BASIS,
        "source": document.timing_source,
        "assumed_frame_interval_s": document.assumed_frame_interval_s,
    }


def _summary_view(document) -> dict:
    """One document, WITHOUT its text. The list shape.

    `title` may be null and a client must render that as a description of
    the RECORD -- "Untitled document" -- never as an invented name for
    the thing. It is one line, lifted from the document's own first text
    region, and iOS asks for it in the list knowing that.

    `summary` IS NOT HERE, and its absence is the point.

    `DocumentMemoryEngine._summarise` is the document's first forty words
    verbatim. That is an excerpt, not a summary, and forty words per
    document across a list is exactly the bulk transfer of contents that
    `IOS-to-Tower.md` 3.2 exists to prevent -- "a list of documents is
    not also a bulk transfer of every document's contents onto the
    phone". Serving it here would have honoured the letter of "the list
    carries a character count, not the text" while breaking it in fact,
    and the failure would have been invisible: the field is small, and
    nobody reviewing one response would notice that a hundred of them
    add up to the library.

    So the list says a summary EXISTS and where to get it, and
    `/documents/{document_id}` carries it beside the pages it came from.
    A caller that wanted it wanted the document.
    """
    return {
        "document_id": document.document_id,
        "claim": DOCUMENT_CLAIM,
        "identity": IDENTITY_SCOPE,
        "title": _clip(document.title, TITLE_MAX_CHARS_ON_WIRE),
        "title_is_derived": True,
        "title_max_chars": TITLE_MAX_CHARS_ON_WIRE,
        "summary_available": bool(document.summary),
        "confidence": document.confidence.value,
        "confidence_basis": "the weakest page read in this document",
        "observed_at": document.observed_at,
        "recorded_at": document.recorded_at,
        # NEVER `viewing_duration`. Appearing in the camera does not
        # establish that the wearer looked at it, noticed it, or read it.
        "observed_seconds": document.observed_seconds,
        # Later looks at this same record, merged by `identity`. The
        # FIRST observation's fields above are never rewritten by a
        # sighting; these three describe the record as a whole.
        "sighting_count": document.sighting_count,
        "last_observed_at": document.last_observed_at,
        "total_observed_seconds": document.total_observed_seconds,
        "pages_observed": document.pages_observed,
        "pages_readable": sum(1 for page in document.pages if page.readable),
        "text_availability": _text_availability(document),
        "end_reason": document.end_reason,
        "timing": _timing(document),
        "provenance": _provenance(document),
        "retains_raw_imagery": bool(document.retains_raw_imagery),
        "redaction": document.redaction,
        # Varies with the fact, rather than being a constant that said
        # the same thing whatever was on disk.
        "imagery_treatment": (
            IMAGERY_RAW_PERSISTED
            if document.retains_raw_imagery
            else IMAGERY_NONE_RETAINED
        ),
        "imagery_ios_state": IMAGERY_IOS_STATE,
        "imagery_served": False,
        "privacy_tags": list(document.privacy_tags),
        "schema_version": document.schema_version,
    }


def _page_view(page) -> dict:
    return {
        "page_index": page.page_index,
        "text": page.text,
        "text_source": page.text_source,
        "region_count": page.region_count,
        "mean_region_confidence": page.mean_region_confidence,
        "min_region_confidence": page.min_region_confidence,
        "confidence": page.confidence.value,
        "sharpness": page.sharpness,
        "squareness": page.squareness,
        "source_seq": page.source_seq,
        "observed_at": page.observed_at,
        # How many separate views of this page were merged into it. Two
        # readings of one page during one dwell is one page with an
        # observation count of two, not two pages.
        "observation_count": page.observation_count,
        # Whether the words above cleared the readability floor. A page
        # with regions and no readable text is one OCR looked at and
        # could not read -- which is a different fact from never looking.
        "readable": bool(page.readable),
        # How many text boxes the detector found where this page was
        # read from. Evidence that text was there even when it was not
        # readable.
        "box_count": page.box_count,
        # A BOOLEAN, not the path. The path told a reader that an
        # unredacted photograph of what the wearer was reading exists on
        # this Tower's disk, and where in the store -- disclosure with no
        # consumer, since no route resolves it and none may.
        #
        # False is the default and must stay the default.
        "image_kept": bool(page.image_relpath),
        "image_served": False,
    }


def _envelope(requested_days: float | None) -> dict:
    return {
        "contract": LIBRARY_CONTRACT,
        "claim": DOCUMENT_CLAIM,
        "identity": IDENTITY_SCOPE,
        "absence_means": ABSENCE_MEANING,
        "time_basis": TIME_BASIS,
        "spatial_ref": None,
        "answers": list(ANSWERS),
        "retrieval_kinds": list(RETRIEVAL_KINDS),
        "semantic_retrieval": False,
        "semantic_retrieval_unavailable_reason": (
            "this cartridge matches literal terms with BM25 and computes "
            "no embedding. Calling it semantic would be an overclaim"
        ),
        # iOS routes typed free text to `.semantic`, so its PRIMARY input
        # path has no Tower route. Saying only "no semantic retrieval"
        # leaves it to guess what to do instead.
        "semantic_retrieval_alternative": (
            "route free text to retrieval_kind 'text'. It will be matched "
            "literally, so a description of a document will usually miss "
            "where a quotation from one will hit"
        ),
        "pagination": {
            "supported": False,
            "bound": "limit",
            "reason": (
                "there is no cursor. Every listing route takes a `limit` "
                "and caps it, and the effective value is echoed in the "
                "`query` block. Truncation is detectable on `recent` by "
                "comparing document_count with documents_in_memory; on "
                "`text` and `observed_within` those two differ because of "
                "the query as well, so compare document_count with the "
                "limit instead"
            ),
        },
        "privacy_tag_vocabulary": list(PRIVACY_TAG_VOCABULARY),
        # The caveats every record carries, carried once. Each is keyed
        # by the field it qualifies; a client renders `documents[i]` with
        # `record_notes` beside it.
        "record_notes": dict(RECORD_NOTES),
        "recording_limitations": [dict(entry) for entry in RECORDING_LIMITATIONS],
        "recording_measurement": dict(RECORDING_MEASUREMENT),
        "imagery_treatment": IMAGERY_NONE_RETAINED,
        "imagery_ios_state": IMAGERY_IOS_STATE,
        "imagery_served": False,
        "imagery_note": IMAGERY_NOTE,
        "retention": _retention_view(requested_days),
    }


def _answer_for(total_documents: int, found: int) -> str:
    """Which of the three answers this result is.

    The order matters: an empty MEMORY is `no_observation` whatever was
    asked, because there is nothing that could have matched. Only a
    non-empty memory can produce `not_found`.
    """
    if total_documents == 0:
        return ANSWER_NO_OBSERVATION
    return ANSWER_MATCHED if found else ANSWER_NOT_FOUND


def _no_observation_note(total_documents: int) -> str | None:
    if total_documents:
        return None
    return (
        "this Tower has recorded no documents at all. That is a statement "
        "about what its camera captured, never about what exists -- and on "
        "this platform it is the expected result: see recording_limitations"
    )


def recent_documents(store, *, limit: int = 10, requested_days=None) -> dict:
    """The most recent documents, newest first, without their text."""
    memory = DocumentMemory(store)
    total = store.count()
    documents = memory.recent(limit=limit)

    payload = _envelope(requested_days)
    payload["query"] = {"kind": "recent", "limit": limit}
    payload["answer"] = _answer_for(total, len(documents))
    payload["no_observation_note"] = _no_observation_note(total)
    payload["documents_in_memory"] = total
    payload["document_count"] = len(documents)
    payload["documents"] = [_summary_view(document) for document in documents]
    return payload


def documents_around(
    store,
    *,
    when: float,
    window_seconds: float = 900.0,
    limit: int = 50,
    requested_days=None,
) -> dict:
    """Documents observed within a window of an instant.

    A RANGE, not an instant, and the distinction is iOS's: "this morning"
    and "around lunch" are approximate, and answering them exactly answers
    a different question.

    Bounded, like its two siblings and for the same reason. This shipped
    for one commit with no limit at all, while the envelope it returned
    claimed `limit` was the only bound: 1,000 documents in a 24-hour
    window measured 2.4 MB, on a sync handler that parses the whole
    journal. An unbounded limit is a remote party choosing how much
    memory this process uses.
    """
    memory = DocumentMemory(store)
    total = store.count()
    documents = memory.around(when, window_seconds=window_seconds)[:limit]

    payload = _envelope(requested_days)
    payload["query"] = {
        "kind": "observed_within",
        "centre": when,
        "window_seconds": window_seconds,
        "limit": limit,
    }
    payload["answer"] = _answer_for(total, len(documents))
    payload["no_observation_note"] = _no_observation_note(total)
    payload["documents_in_memory"] = total
    payload["document_count"] = len(documents)
    payload["documents"] = [_summary_view(document) for document in documents]
    return payload


def search_documents(
    store, *, text: str, limit: int = 5, min_score: float = MIN_SCORE,
    requested_days=None,
) -> dict:
    """Literal term matching over what was captured.

    Each match carries the snippet the score came from, so an answer is
    always traceable back to text that was actually read rather than to a
    number a client has to trust.
    """
    memory = DocumentMemory(store)
    total = store.count()
    result = memory.search_text(text, limit=limit, min_score=min_score)

    payload = _envelope(requested_days)
    payload["query"] = {"kind": "text", "text": text, "limit": limit}
    payload["answer"] = _answer_for(total, len(result.matches))
    payload["no_observation_note"] = _no_observation_note(total)
    payload["documents_in_memory"] = total
    payload["searched_documents"] = result.searched_documents
    payload["searched_pages"] = result.searched_pages
    payload["min_score"] = result.min_score
    payload["sufficient_evidence"] = bool(result.sufficient_evidence)
    payload["reason"] = result.reason
    payload["match_kind"] = "lexical"
    # Since 2026-09-07 a term of five or more characters also matches a
    # token within one edit of it or one it begins -- OCR reads
    # "Kubernetes" as "Kubemetes" often enough to matter. Each match says
    # whether it needed that, and an exact match outranks a near one.
    payload["match_tolerance"] = (
        "exact for terms under five characters; one edit or a prefix "
        "for longer terms, weighted below an exact match"
    )
    payload["document_count"] = len(result.matches)
    payload["snippet_max_chars"] = SNIPPET_MAX_CHARS
    payload["documents"] = [
        dict(
            _summary_view(match.document),
            score=round(match.score, 4),
            matched_terms=list(match.matched_terms),
            # Bounded evidence, not an excerpt. See SNIPPET_MAX_CHARS.
            snippet=_clip(match.snippet, SNIPPET_MAX_CHARS),
            # Which page the snippet and the score came from.
            page_index=match.page_index,
            fuzzy=bool(match.fuzzy),
        )
        for match in result.matches
    ]
    return payload


def one_document(store, document_id: str, *, requested_days=None) -> dict | None:
    """One document WITH its pages and their text, or None.

    This is the only route that carries text, and it is per-document by
    design: a list that carried text would be a bulk transfer of
    everything a wearer read onto whatever asked for a list.

    `coverage` rides along because the two questions a person opening a
    document asks -- "what does it say" and "how much of it did you get"
    -- are one round trip, and splitting them invites a client to render
    the first without the second.
    """
    memory = DocumentMemory(store)
    document = store.read_one(document_id)
    if document is None:
        return None

    payload = _envelope(requested_days)
    payload["query"] = {"kind": "document", "document_id": document_id}
    payload["answer"] = ANSWER_MATCHED
    payload["no_observation_note"] = None
    payload["document"] = dict(
        _summary_view(document),
        # The two things a list may not carry, carried here where a
        # person has asked for this document specifically.
        summary=document.summary or None,
        summary_is_model_output=True,
        summary_is_verbatim_excerpt=True,
        pages=[_page_view(page) for page in document.pages],
        word_count=document.word_count,
    )
    payload["coverage"] = memory.coverage(document_id)
    return payload


# -- the session status, which travels on the result channel ------------

# Fields that advance without anything having happened. Excluded from the
# change revision so a heartbeat is distinguishable from news, exactly as
# `world_builder.VOLATILE_PATHS` does for `mapping_seconds`.
STATUS_VOLATILE_PATHS = (
    "session.started_at",
    "session.ready_at",
    "session.loading_seconds",
    "session.frames_offered",
    "session.frames_observed",
    "session.frames_skipped",
    "session.frames_dropped_not_running",
    "session.decode_failures",
    "session.pages_detected",
    "library.bytes",
)

# The reason a Tower has no capture session, and the state that says so.
#
# `unavailable` is NOT one of `LIFECYCLE_STATES`, and that is deliberate
# rather than an omission: the lifecycle states describe a session, and
# this is the case where there is none. It is published in `states` all
# the same, so a decoder can pin the full vocabulary from any payload --
# an adversarial review found the earlier version emitting a `state`
# value the payload's own enum denied existed, in the one shape that did
# not carry the enum.
SESSION_UNAVAILABLE = "unavailable"

SESSION_ABSENT_REASON = (
    "no document capture session exists on this Tower "
    "(TOWER_DOCUMENT_CAPTURE is off, or TOWER_DOCUMENT_ROOT is unset). "
    "Documents recorded elsewhere are still served: this says nothing "
    "was recorded HERE"
)

# Every key a live session's status carries, present and null when there
# is no session.
#
# A block that changed shape forced a decoder to make 31 fields optional
# and to handle a `state` outside the published enum -- and `states` was
# itself one of the keys that disappeared, so a client could never learn
# the vocabulary from the payload that needed it. Scene Understanding's
# payload already keeps every key present with `null`; this now matches.
_SESSION_NULL_FIELDS = (
    "session_id",
    "failure_reason",
    "started_at",
    "ready_at",
    "loading_seconds",
    "load_overdue",
    "load_overdue_after_seconds",
    "engine",
    "frames_offered",
    "frames_observed",
    "frames_skipped",
    "frames_dropped_not_running",
    "recogniser",
    "ocr_device",
    "capture_id",
    "capture_id_validated",
    "in_dwell",
    "dwells_started",
    "pages_detected",
    "documents_recorded",
    "documents_resighted",
    "dwells_unreadable",
    "last_document_id",
    "last_document_at",
    "flushed_document_id",
    "keeps_page_images",
    "follows_stream",
    "idle_stop_seconds",
    "idle_stop_pending",
    "retention_days",
    "documents_pruned",
    "retention_incomplete",
    "library_count",
    "library_soft_limit",
    "library_over_soft_limit",
    "library_soft_limit_note",
)


def _session_absent(reason: str | None = None) -> dict:
    from tower.live_session import LIFECYCLE_STATES

    absent = {
        "state": SESSION_UNAVAILABLE,
        "states": list(LIFECYCLE_STATES) + [SESSION_UNAVAILABLE],
        # The runtime's own reason when it has one -- the OCR extra is
        # not installed, most likely -- else the configuration reading.
        "reason": reason or SESSION_ABSENT_REASON,
    }
    absent.update({name: None for name in _SESSION_NULL_FIELDS})
    return absent


def session_view(session) -> dict:
    """A session status, in the same envelope every other response has.

    The five session routes used to return `session.status()` raw: no
    contract, no claim, no recording limitations, no imagery statement --
    while this module's own header says those travel on EVERY response.
    They are responses. Scene Understanding's control routes already run
    everything through one function for the same reason.

    A client that polls the session and never calls a listing would
    otherwise never learn that an empty library is the expected result on
    this platform, which is the single most important thing this
    cartridge has to say.
    """
    return {
        "contract": LIBRARY_CONTRACT,
        "claim": DOCUMENT_CLAIM,
        "identity": IDENTITY_SCOPE,
        "absence_means": ABSENCE_MEANING,
        "time_basis": TIME_BASIS,
        "recording_limitations": [dict(entry) for entry in RECORDING_LIMITATIONS],
        "recording_measurement": dict(RECORDING_MEASUREMENT),
        "imagery_treatment": IMAGERY_NONE_RETAINED,
        "imagery_ios_state": IMAGERY_IOS_STATE,
        "imagery_served": False,
        "imagery_note": IMAGERY_NOTE,
        "session": _session_present(session),
    }


def _session_present(session) -> dict:
    status = dict(session.status())
    status["states"] = list(status["states"]) + [SESSION_UNAVAILABLE]
    # Present and null, so the two shapes have identical key sets.
    status["reason"] = None
    for name in _SESSION_NULL_FIELDS:
        status.setdefault(name, None)
    return status


class DocumentStatusProducer:
    """The session status, polled twice a second, without re-reading a file.

    Stat-gated for the reason `world_builder.py: _FileCache` is, and it
    is a requirement rather than an optimisation: the result hub polls
    every 0.5 s for as long as anyone is subscribed, and every figure in
    the `library` block below comes from parsing `documents.jsonl` end to
    end. Re-parsing an unchanged journal twice a second would make a
    subscription cost more than the capture it is watching, and the cost
    would grow with the library rather than staying flat.

    The gate is `(mtime_ns, size)`. Both, not either: a rewrite that
    preserved the size would slip past a size check, and a filesystem
    with coarse timestamps can produce two writes inside one tick.
    """

    def __init__(
        self, document_root, session, *, clock=None, unavailable_reason=None
    ) -> None:
        self._root = document_root
        self._session = session
        self._clock = clock
        self._unavailable_reason = unavailable_reason
        self._stamp = None
        self._summary = None

    def _library(self) -> dict:
        store = store_from_root(self._root, clock=self._clock)
        try:
            stat = store.path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            # No journal is not a failure. It is a Tower that has never
            # recorded a document -- which, on this platform, is the
            # expected state, and it is reported as an EMPTY library
            # rather than an unreadable one. Those are opposite claims.
            return {
                "available": True,
                "document_count_unfiltered": 0,
                "retention_applied": False,
                "unavailable_reason": None,
                "newest_observed_at": None,
                "revision": 0,
                "bytes": {"journal": 0, "images": 0, "total": 0},
                "location_disclosed": False,
            }
        except Exception:
            stamp = None

        if stamp is not None and stamp == self._stamp and self._summary is not None:
            return dict(self._summary)

        try:
            documents = store.read_all()
            summary = {
                "available": True,
                # UNFILTERED, and named so. `session.library_count` is the
                # same quantity through the session's retention window and
                # refreshed only by a prune, so the two diverge whenever
                # records exist past the window. Two numbers side by side
                # with one name would be the worst of both.
                "document_count_unfiltered": len(documents),
                "retention_applied": False,
                "unavailable_reason": None,
                "newest_observed_at": (
                    max(document.last_observed_at for document in documents)
                    if documents
                    else None
                ),
                # Changes whenever the journal changes -- an append, a
                # sighting merged onto an existing record, a prune. A
                # phone that holds a listing re-fetches it when this
                # moves and never otherwise; the count alone would miss a
                # sighting and a prune-plus-append in one tick. The
                # journal's mtime, which is not a clock a client should
                # read as one.
                "revision": int(stamp[0]) if stamp is not None else 0,
                "bytes": store.bytes_used(),
                # No path, ever. The channel already holds this rule for
                # World Builder and it is the channel's rule, not that
                # cartridge's.
                "location_disclosed": False,
            }
        except Exception:
            return {
                "available": False,
                "document_count_unfiltered": None,
                "retention_applied": False,
                "unavailable_reason": (
                    "this Tower's document journal could not be read"
                ),
                "newest_observed_at": None,
                "revision": None,
                "bytes": None,
                "location_disclosed": False,
            }

        self._stamp = stamp
        self._summary = summary
        return dict(summary)

    def payload(self) -> dict:
        """What this Tower is doing about documents, and what it holds.

        Two halves that must not be confused, and the field names keep
        them apart: `session` is a live capture that may not exist at
        all, and `library` is what is on disk regardless of whether
        anything is running. A Tower reprocessing captures offline has a
        library and no session; that is a normal configuration, not a
        degraded one.

        The documents themselves are NOT here. They are bulk, they are
        text, and they belong on HTTP for the same reason World Builder's
        geometry does -- `tower/routes/ws.py` gives the result sender and
        the frame path one shared lock.
        """
        return {
            "contract_note": (
                "session progress only. The documents themselves are on "
                "HTTP: /documents, /documents/{document_id}, "
                "/documents/search"
            ),
            "claim": DOCUMENT_CLAIM,
            "identity": IDENTITY_SCOPE,
            "absence_means": ABSENCE_MEANING,
            "time_basis": TIME_BASIS,
            "library": self._library(),
            "session": (
                _session_absent(self._unavailable_reason)
                if self._session is None
                else _session_present(self._session)
            ),
            "recording_limitations": [
                dict(entry) for entry in RECORDING_LIMITATIONS
            ],
            "recording_measurement": dict(RECORDING_MEASUREMENT),
            "imagery_treatment": IMAGERY_NONE_RETAINED,
            "imagery_ios_state": IMAGERY_IOS_STATE,
            "imagery_served": False,
            "imagery_note": IMAGERY_NOTE,
        }


__all__ = [
    "ANSWERS",
    "ANSWER_MATCHED",
    "ANSWER_NOT_FOUND",
    "ANSWER_NO_OBSERVATION",
    "LIBRARY_CONTRACT",
    "RECORDING_LIMITATIONS",
    "STATUS_VOLATILE_PATHS",
    "documents_around",
    "session_view",
    "one_document",
    "recent_documents",
    "search_documents",
    "DocumentStatusProducer",
    "store_from_root",
]
