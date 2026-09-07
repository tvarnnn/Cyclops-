"""What Document Memory persists, and the words it is allowed to use.

The naming here is a constraint, not a style choice.
`07-PLATFORM-CONSTRAINTS.md` Limitation 8: something appearing in the
glasses camera does not prove the user looked at it, noticed it, read it,
or understood it. There is no eye tracking on this hardware.

So this module records an **observation** -- a page-like region present in
the camera view, held steadily, long enough to be worth reading -- and
never a **reading**. `observed_at`, `observed_seconds`, `pages_observed`.
Nothing here is named `read` or `viewed`, and nothing may be.
"""

from dataclasses import dataclass, field

from tower.confidence import Confidence

# Bump when the meaning of any persisted field changes. An integer, not
# semver. A reader that meets a version it does not know refuses.
SCHEMA_VERSION = 1

# There is no capture timestamp anywhere on the wire, so every timestamp
# this module writes is tower-receipt time and says so (Rule 16).
TIME_BASIS = "tower-receipt"

# Why a dwell ended. Recorded so a short observation is explainable
# rather than merely short.
END_REASON_LOST = "region_lost"
END_REASON_STOPPED = "stream_stopped"
END_REASON_MAX_DURATION = "max_duration"

# Whether the corrected page image was kept. "none" is the honest value
# for imagery this platform cannot redact -- see 06-PRIVACY-DATA.md's
# Sensitive Visual Information section, and note that a crop is not
# inherently safe.
REDACTION_NONE = "none"

# What text extraction produced this text. There is deliberately no value
# meaning "assumed" or "inferred": text that was not recognised is absent,
# not guessed.
TEXT_SOURCE_OCR = "ocr"

# Where a dwell's DURATION came from. This matters because a directory of
# loose jpegs carries no timestamps at all, so replaying one has to assume
# a frame interval -- and an assumed duration must never be readable as a
# measured one.
#
# "capture-journal": real receipt times, from the recorder's journal.
# "assumed-interval": frames had no timestamps; a fixed interval was
#   supplied by the operator and `assumed_frame_interval_s` says which.
# "mixed": SOME frames carried real receipt times and some did not.
#   Reported rather than collapsed to either -- labelling a mixed document
#   as fully assumed understates what is known, and labelling it as fully
#   measured overstates it.
TIMING_CAPTURE_JOURNAL = "capture-journal"
TIMING_ASSUMED_INTERVAL = "assumed-interval"
TIMING_MIXED = "mixed"


@dataclass(frozen=True)
class PageObservation:
    """One OCR'd view of one page-like region.

    `text` is what OCR actually returned. `region_count` and
    `mean_region_confidence` describe how much of it OCR was sure about.
    A page whose OCR returned nothing is still recorded -- "we looked and
    found no readable text" is a different fact from "we never looked",
    and collapsing them is exactly what Core Principle 3 forbids.
    """

    page_index: int
    text: str
    text_source: str = TEXT_SOURCE_OCR
    region_count: int = 0
    mean_region_confidence: float | None = None
    min_region_confidence: float | None = None
    confidence: Confidence = Confidence.UNKNOWN
    sharpness: float | None = None
    squareness: float | None = None
    source_seq: int | None = None
    observed_at: float | None = None
    observation_count: int = 1
    image_relpath: str | None = None
    # A 64-bit perceptual hash of the text region, as 16 hex characters,
    # or None. NOT imagery: sixteen characters cannot be rendered back
    # into a page. It is what lets a later sighting of the same page be
    # recognised without keeping the pixels that would prove it.
    visual_hash: str | None = None
    # How many text boxes the detector found in the region this page
    # was read from. Evidence for the gate, kept so an unreadable page
    # can say "text was there" rather than nothing.
    box_count: int = 0
    # Whether this page's text passed the readability floor. A page can
    # have regions and no readable text: OCR looked, and what it found
    # was noise. That is a different fact from never looking.
    readable: bool = False

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_json_dict(self) -> dict:
        return {
            "page_index": self.page_index,
            "text": self.text,
            "text_source": self.text_source,
            "region_count": self.region_count,
            "mean_region_confidence": self.mean_region_confidence,
            "min_region_confidence": self.min_region_confidence,
            "confidence": self.confidence.value,
            "sharpness": self.sharpness,
            "squareness": self.squareness,
            "source_seq": self.source_seq,
            "observed_at": self.observed_at,
            "observation_count": self.observation_count,
            "image_relpath": self.image_relpath,
            "visual_hash": self.visual_hash,
            "box_count": self.box_count,
            "readable": self.readable,
        }


def page_observation_from_json_dict(data: dict) -> PageObservation:
    text = data["text"]
    return PageObservation(
        page_index=data["page_index"],
        text=text,
        text_source=data.get("text_source", TEXT_SOURCE_OCR),
        region_count=data.get("region_count", 0),
        mean_region_confidence=data.get("mean_region_confidence"),
        min_region_confidence=data.get("min_region_confidence"),
        confidence=Confidence(data.get("confidence", Confidence.UNKNOWN.value)),
        sharpness=data.get("sharpness"),
        squareness=data.get("squareness"),
        source_seq=data.get("source_seq"),
        observed_at=data.get("observed_at"),
        observation_count=data.get("observation_count", 1),
        image_relpath=data.get("image_relpath"),
        visual_hash=data.get("visual_hash"),
        box_count=data.get("box_count", 0),
        # A record written before this field existed is readable exactly
        # when it carries text: that is what "readable" meant then.
        readable=data.get("readable", bool(text.strip())),
    )


@dataclass(frozen=True)
class Sighting:
    """One later observation of a document already on record.

    A wearer who looks back at a page they read ten minutes ago has not
    read a new document. The record gains a sighting -- when, for how
    long, from which capture -- and nothing about the FIRST observation
    is rewritten: its provenance stays exactly what it was.
    """

    observed_at: float
    observed_seconds: float
    capture_id: str | None = None
    source_seq: int | None = None
    end_reason: str = END_REASON_LOST
    frames_considered: int = 0

    def to_json_dict(self) -> dict:
        return {
            "observed_at": self.observed_at,
            "observed_seconds": self.observed_seconds,
            "capture_id": self.capture_id,
            "source_seq": self.source_seq,
            "end_reason": self.end_reason,
            "frames_considered": self.frames_considered,
        }


def sighting_from_json_dict(data: dict) -> Sighting:
    return Sighting(
        observed_at=data["observed_at"],
        observed_seconds=data.get("observed_seconds", 0.0),
        capture_id=data.get("capture_id"),
        source_seq=data.get("source_seq"),
        end_reason=data.get("end_reason", END_REASON_LOST),
        frames_considered=data.get("frames_considered", 0),
    )


@dataclass(frozen=True)
class DocumentObservation:
    """One sustained observation of one document.

    NOT a claim that the wearer read it, understood it, or even noticed
    it. The system observed a page-like region in view, held steadily,
    for `observed_seconds`. That is a proxy for reading and a poor
    synonym for it.

    The spatial fields are **supplied by a caller or absent**. Nothing in
    this module derives them and this module must not import World
    Builder -- a test enforces that. `None` means unknown, which is not
    the same as "nowhere".
    """

    document_id: str
    observed_at: float
    recorded_at: float
    observed_seconds: float
    pages: tuple[PageObservation, ...]
    schema_version: int = SCHEMA_VERSION
    time_basis: str = TIME_BASIS
    title: str | None = None
    summary: str = ""
    frames_considered: int = 0
    frames_ocred: int = 0
    end_reason: str = END_REASON_LOST
    confidence: Confidence = Confidence.UNKNOWN
    capture_id: str | None = None
    # How `observed_seconds` was arrived at. An assumed duration must
    # never be mistaken for a measured one.
    timing_source: str = TIMING_CAPTURE_JOURNAL
    assumed_frame_interval_s: float | None = None
    # Supplied, never derived. See the class docstring.
    world_id: str | None = None
    world_session_id: str | None = None
    frame_revision: int | None = None
    retains_raw_imagery: bool = False
    redaction: str = REDACTION_NONE
    privacy_tags: tuple[str, ...] = ("document-text", "first-person")
    # Later observations of this same document, oldest first. Empty for
    # a document seen once. See `Sighting`.
    sightings: tuple[Sighting, ...] = ()

    @property
    def pages_observed(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        """Every page's text, in page order. The retrieval corpus."""
        return "\n".join(page.text for page in self.pages if page.text)

    @property
    def word_count(self) -> int:
        return sum(page.word_count for page in self.pages)

    @property
    def sighting_count(self) -> int:
        """How many times this document was observed, first one included."""
        return 1 + len(self.sightings)

    @property
    def last_observed_at(self) -> float:
        if not self.sightings:
            return self.observed_at
        return max(self.observed_at, max(s.observed_at for s in self.sightings))

    @property
    def total_observed_seconds(self) -> float:
        return self.observed_seconds + sum(s.observed_seconds for s in self.sightings)

    def to_json_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "observed_at": self.observed_at,
            "recorded_at": self.recorded_at,
            "observed_seconds": self.observed_seconds,
            "time_basis": self.time_basis,
            "title": self.title,
            "summary": self.summary,
            "frames_considered": self.frames_considered,
            "frames_ocred": self.frames_ocred,
            "end_reason": self.end_reason,
            "confidence": self.confidence.value,
            "capture_id": self.capture_id,
            "timing_source": self.timing_source,
            "assumed_frame_interval_s": self.assumed_frame_interval_s,
            "world_id": self.world_id,
            "world_session_id": self.world_session_id,
            "frame_revision": self.frame_revision,
            "retains_raw_imagery": self.retains_raw_imagery,
            "redaction": self.redaction,
            "privacy_tags": list(self.privacy_tags),
            "pages": [page.to_json_dict() for page in self.pages],
            "sightings": [sighting.to_json_dict() for sighting in self.sightings],
        }


def document_observation_from_json_dict(data: dict) -> DocumentObservation:
    return DocumentObservation(
        document_id=data["document_id"],
        observed_at=data["observed_at"],
        recorded_at=data["recorded_at"],
        observed_seconds=data["observed_seconds"],
        pages=tuple(
            page_observation_from_json_dict(page) for page in data.get("pages", [])
        ),
        schema_version=data.get("schema_version", SCHEMA_VERSION),
        time_basis=data.get("time_basis", TIME_BASIS),
        title=data.get("title"),
        summary=data.get("summary", ""),
        frames_considered=data.get("frames_considered", 0),
        frames_ocred=data.get("frames_ocred", 0),
        end_reason=data.get("end_reason", END_REASON_LOST),
        confidence=Confidence(data.get("confidence", Confidence.UNKNOWN.value)),
        capture_id=data.get("capture_id"),
        timing_source=data.get("timing_source", TIMING_CAPTURE_JOURNAL),
        assumed_frame_interval_s=data.get("assumed_frame_interval_s"),
        world_id=data.get("world_id"),
        world_session_id=data.get("world_session_id"),
        frame_revision=data.get("frame_revision"),
        retains_raw_imagery=data.get("retains_raw_imagery", False),
        redaction=data.get("redaction", REDACTION_NONE),
        privacy_tags=tuple(data.get("privacy_tags", ())),
        sightings=tuple(
            sighting_from_json_dict(entry) for entry in data.get("sightings", [])
        ),
    )
