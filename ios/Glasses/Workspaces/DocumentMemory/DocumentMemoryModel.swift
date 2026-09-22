//
//  DocumentMemoryModel.swift
//  Glasses
//

import Foundation

/// What the Document Memory workspace shows, and the rules it shows it under.
///
/// The wire types live in `DocumentMemoryLibrary.swift`; this file is the
/// presentation vocabulary those decode into. Contracts:
/// `document_memory.library/2026-09-07` over HTTP and
/// `document_memory.status/2026-09-07` on the socket.
///
/// ## Three refusals, each load-bearing
///
/// **No OCR on iOS.** Not a single character of text in this file is produced
/// on the phone. `docs/02-DEVELOPMENT-RULES.md` Rule 5 keeps heavy CV on the
/// Tower, and Rule 12 plus `docs/06-PRIVACY-DATA.md` make document contents
/// among the most sensitive data the platform touches. `DocumentTextAvailability`
/// reports whether the *Tower* extracted text; iOS never extracts, and cannot.
///
/// **No "viewing" anything.** `docs/07-PLATFORM-CONSTRAINTS.md` Limitation 8:
/// appearing in the camera does not prove the wearer looked at it, and the
/// mitigation is classified REQUIRES FUTURE HARDWARE/API. The Tower agrees in
/// its own `claim`: `a-page-was-in-view-and-was-ocred`, **not** "was read". So
/// a document was *observed*, for an `ObservedDuration`, and every label says
/// so.
///
/// **No image, from either side.** `imagery_served` is `false` on every
/// response and it is a **boolean, not a path** — the path told a reader where
/// in the store to find an unredacted photograph of what the wearer was
/// reading, which is disclosure with no consumer, since no route resolves it
/// and none may. `image_kept` is false unless page images were explicitly
/// enabled, which is off by default and must stay off.

// MARK: - Extracted text

/// Whether the Tower has text for a document, without this app ever producing
/// any.
///
/// The `.extracted` case carries a character count and **not the text**. The
/// count is what a list row needs in order to say the document is readable; the
/// text itself is fetched when a person opens one, so a list of documents is
/// not also a bulk transfer of every document's contents into the phone's
/// memory.
enum DocumentTextAvailability: Equatable, Sendable {
    /// The record has no pages. Distinct from `.notReadable`: silence is not a
    /// verdict.
    case unknown
    /// **A real answer**: we looked and found no readable text. `character_count`
    /// is `0` on the wire, which is the difference between this and `.unknown`.
    ///
    /// This is the case that is expected on this platform, not the exception —
    /// at 360×640 EasyOCR returned zero dictionary words across 919 sampled
    /// real frames at median confidence 0.056.
    case notReadable
    /// The Tower has text. `characterCount` is `nil` when it did not say how
    /// much.
    case extracted(characterCount: Int?)

    /// The wire's three words, mapped.
    init(state: String?, characterCount: Int?) {
        switch state {
        case "not_readable": self = .notReadable
        case "extracted": self = .extracted(characterCount: characterCount)
        // `unknown`, and anything this build has not heard of. Folding an
        // unrecognised state into `.notReadable` would turn a decode gap into
        // a verdict about the page.
        default: self = .unknown
        }
    }

    var hasText: Bool {
        if case .extracted = self { return true }
        return false
    }

    var displayName: String {
        switch self {
        case .unknown: return "Not reported"
        case .notReadable: return "No readable text"
        case .extracted(let count):
            guard let count else { return "Text available" }
            return "\(count) characters"
        }
    }

    /// The sentence `.notReadable` is owed.
    ///
    /// Without it a reader takes "No readable text" for a failure. It is not:
    /// the Tower looked at the page and found nothing it could read, which is
    /// an answer about the page and this camera, and on this platform it is the
    /// expected one.
    var explanation: String? {
        switch self {
        case .notReadable:
            return """
                The Tower read this page and found nothing legible. That is an \
                answer, not an error — at the resolution these glasses send, \
                text recognition returns almost nothing.
                """
        case .unknown:
            return "This record has no pages, so there was nothing to read."
        case .extracted:
            return nil
        }
    }
}

// MARK: - Source context

/// Where a document came from, when the Tower can say.
///
/// `worldID` is present because `docs/modules/ENVIRONMENTAL-MEMORY.md` allows
/// environmental observations to *optionally* reference spatial locations if
/// World Build later exposes a shared spatial service — and is explicit that
/// this "must be an explicit architecture evolution, not an assumed
/// dependency". So it is optional here, nothing requires it, and no view
/// degrades without it. Document Memory does not depend on World Builder.
struct DocumentSourceContext: Equatable, Sendable {
    /// The Tower's session identifier, opaque.
    var sessionID: String?
    /// The world this observation belongs to, if a world existed and the Tower
    /// chose to link them.
    var worldID: String?

    init(sessionID: String? = nil, worldID: String? = nil) {
        self.sessionID = sessionID
        self.worldID = worldID
    }

    var isEmpty: Bool { sessionID == nil && worldID == nil }
}

// MARK: - Document

/// One document the Tower recorded.
///
/// Every descriptive field is optional. A Tower that reports an identifier and
/// a timestamp and nothing else must not force the UI to invent a title, and a
/// row for such a document is a short row rather than a row full of dashes.
struct RememberedDocument: Equatable, Identifiable, Sendable {
    /// The Tower's identifier, opaque. Since 2026-09-07 it IS durable across
    /// sightings of the same page: a later look whose words and look both
    /// agree with a page of this record becomes a sighting of it
    /// (`sightingCount`, `lastObservedAt`) rather than a new record, which is
    /// what `identity: "same-page-by-text-and-look-within-library"` means.
    /// Anything weaker is still a separate record with its own id.
    let id: String
    /// A title the Tower **derived from the document's own first text region**,
    /// clipped to `titleMaxChars`. `nil` renders as "Untitled document", which
    /// is a description of the record rather than an invented name for the
    /// thing.
    var title: String?
    /// The document's first forty words, **verbatim**.
    ///
    /// Served only by `GET /documents/{id}`, never in a listing: forty words
    /// per document across a list is exactly the bulk transfer the split
    /// between the socket and HTTP exists to prevent. An excerpt, not a
    /// paraphrase — see `summaryIsVerbatimExcerpt`, which is what stops it
    /// being captioned as model output.
    var summary: String?
    /// Whether the Tower has the document's text, and how much.
    var text: DocumentTextAvailability
    /// When it was observed, and when this app heard about it — kept apart.
    ///
    /// **Both are Tower-receipt time.** There is no capture clock anywhere on
    /// this wire, and `record_notes.timing` says so in the Tower's own words.
    var time: ObservationTime
    /// How long the region was in the camera's field of view.
    ///
    /// **Not attention.** It is not a claim that the wearer looked at it,
    /// noticed it, or read it — the camera cannot establish any of those, and
    /// `record_notes.observed_seconds` carries that qualification as data.
    var observedDuration: ObservedDuration?
    /// How the Tower arrived at its title and text. `.inferred` for anything a
    /// model produced, which in practice is the title and the OCR.
    var provenance: ObservationProvenance
    /// A keyframe of the document, and whether it may be shown.
    ///
    /// **`.absent` on every record this Tower serves.** `imagery_served` is
    /// `false` and no route resolves an image, so there is nothing to fetch;
    /// the field stays because the display rule is worth keeping in one place
    /// for every cartridge that shows an image.
    var thumbnail: VisualArtifactState
    var source: DocumentSourceContext

    // MARK: The wire's own fields, kept as the wire has them

    /// `true` on every record. The title came from the document, not from a
    /// person and not from a model asked to name it.
    var titleIsDerived: Bool
    /// 60 on this Tower. Read rather than assumed, for the same reason
    /// `snippetMaxChars` is: both are configuration-dependent.
    var titleMaxChars: Int?
    /// Whether a summary exists. A listing says this and withholds the summary
    /// itself; `summaryWithheldReason` is the envelope's `record_notes` entry
    /// explaining why.
    var summaryAvailable: Bool
    /// The Tower's own word for how well this document was read — `low`,
    /// `high`, `unknown`. **A word, not a number**, derived from the mean
    /// region confidence, because one hard word should not condemn a page.
    var confidenceWord: String?
    /// `"the weakest page read in this document"`.
    var confidenceBasis: String?
    /// The Tower's own count of pages in this record.
    var pagesObserved: Int?
    /// Why the dwell ended. `"flushed"` on a record a stop closed.
    var endReason: String?
    /// How the durations above were arrived at. A duration derived from an
    /// assumed frame interval is a reconstruction and must not be rendered
    /// identically to a measured one.
    var timing: DocumentTiming?
    /// The pointer back into a recording. See `DocumentProvenance` — this block
    /// is **joinable**, on purpose, and the workspace says so.
    var frameReference: DocumentProvenance?
    /// Whether an unredacted page image exists on the Tower's disk under
    /// capture-side retention. Not reachable over any wire.
    var retainsRawImagery: Bool
    /// An enum of one, `"none"`, and that is the honest value for imagery this
    /// platform cannot redact.
    var redaction: String?
    /// `none-retained` or `raw-persisted`, per record. It varies with the fact.
    var imageryTreatment: String?
    /// Drawn from the closed set `["document-text", "first-person"]`.
    var privacyTags: [String]
    var schemaVersion: Int?
    /// Words across all pages, on the single-document route.
    var wordCount: Int?
    /// `true`. The summary is an excerpt of the document's own words.
    var summaryIsVerbatimExcerpt: Bool
    /// `true` as well, and the two together are why the summary is captioned as
    /// an excerpt rather than as a paraphrase: a model produced it, and what it
    /// produced is the first forty words unchanged.
    var summaryIsModelOutput: Bool
    /// Present only on a search result.
    var match: DocumentMatchEvidence?
    /// How many times this record was observed, the first time included. One
    /// for a record seen once. The fields above describe the FIRST
    /// observation and are never rewritten by a later sighting.
    var sightingCount: Int = 1
    /// The most recent of those observations. Tower-receipt time.
    var lastObservedAt: Date? = nil
    /// Pages whose text cleared the Tower's readability floor. A record can
    /// hold pages OCR looked at and could not read.
    var pagesReadable: Int? = nil

    init(
        id: String,
        title: String? = nil,
        summary: String? = nil,
        text: DocumentTextAvailability = .unknown,
        time: ObservationTime = ObservationTime(),
        observedDuration: ObservedDuration? = nil,
        provenance: ObservationProvenance = .unknown,
        thumbnail: VisualArtifactState = .absent,
        source: DocumentSourceContext = DocumentSourceContext(),
        titleIsDerived: Bool = false,
        titleMaxChars: Int? = nil,
        summaryAvailable: Bool = false,
        confidenceWord: String? = nil,
        confidenceBasis: String? = nil,
        pagesObserved: Int? = nil,
        endReason: String? = nil,
        timing: DocumentTiming? = nil,
        frameReference: DocumentProvenance? = nil,
        retainsRawImagery: Bool = false,
        redaction: String? = nil,
        imageryTreatment: String? = nil,
        privacyTags: [String] = [],
        schemaVersion: Int? = nil,
        wordCount: Int? = nil,
        summaryIsVerbatimExcerpt: Bool = false,
        summaryIsModelOutput: Bool = false,
        match: DocumentMatchEvidence? = nil
    ) {
        self.id = id
        self.title = title
        self.summary = summary
        self.text = text
        self.time = time
        self.observedDuration = observedDuration
        self.provenance = provenance
        self.thumbnail = thumbnail
        self.source = source
        self.titleIsDerived = titleIsDerived
        self.titleMaxChars = titleMaxChars
        self.summaryAvailable = summaryAvailable
        self.confidenceWord = confidenceWord
        self.confidenceBasis = confidenceBasis
        self.pagesObserved = pagesObserved
        self.endReason = endReason
        self.timing = timing
        self.frameReference = frameReference
        self.retainsRawImagery = retainsRawImagery
        self.redaction = redaction
        self.imageryTreatment = imageryTreatment
        self.privacyTags = privacyTags
        self.schemaVersion = schemaVersion
        self.wordCount = wordCount
        self.summaryIsVerbatimExcerpt = summaryIsVerbatimExcerpt
        self.summaryIsModelOutput = summaryIsModelOutput
        self.match = match
    }

    /// What to put on a row. Never an invented name — "Untitled document"
    /// describes the record, which is all this app knows.
    var displayTitle: String { title ?? "Untitled document" }

    /// Whether the thumbnail may be drawn. Delegates entirely to the artifact,
    /// so the rule lives in one place for every cartridge that shows an image.
    var isThumbnailDisplayable: Bool { thumbnail.isDisplayable }

    /// The caption a summary is owed.
    ///
    /// **Not "estimated by a model".** The summary is the document's first
    /// forty words unchanged, and captioning verbatim text as a paraphrase
    /// invites a reader to discount words the document actually contained.
    /// `ObservationProvenance.caveat` is right for the title and the OCR and
    /// wrong for this, which is why the summary has its own sentence.
    var summaryCaption: String? {
        guard summary != nil else { return nil }
        guard summaryIsVerbatimExcerpt else {
            return provenance.caveat
        }
        return "The document's own first words, copied exactly. Not a summary of it."
    }
}

// MARK: - Queries

/// A request for documents.
///
/// ## Why the input source is separate from the query
///
/// The query says *what* is being asked; `DocumentQueryOrigin` says *where the
/// words came from*. Keeping them apart is what lets a future Siri intent, a
/// custom wake word, or any other input layer submit the same query type
/// without this cartridge growing a dependency on speech, and without a text
/// field being the only way in.
///
/// **No voice input is implemented, required, or assumed.** The origin exists
/// so that adding one later is a new case and a new call site, not a redesign.
nonisolated enum DocumentQuery: Equatable, Sendable {
    /// The most recently observed documents.
    case recent(limit: Int)
    /// Documents whose text contains these terms, matched literally.
    case text(String)
    /// Documents observed in a period. Deliberately an interval rather than an
    /// instant: "this morning" and "around lunch" are ranges, and collapsing
    /// them to a point is how an approximate question gets an exact and wrong
    /// answer. Sent as a centre and a half-width, which is the shape the Tower
    /// takes.
    case observedWithin(DateInterval)
    /// A meaning-based query.
    ///
    /// ## This is an expression of intent, and the Tower cannot honour it
    ///
    /// `semantic_retrieval` is **`false`**: the Tower matches literal terms
    /// with BM25 and computes no embedding, and its own contract says calling
    /// that semantic would be an overclaim. Its `semantic_retrieval_alternative`
    /// says what to do instead — *"route free text to retrieval_kind 'text'. It
    /// will be matched literally, so a description of a document will usually
    /// miss where a quotation from one will hit."*
    ///
    /// The case survives because it records what the **person** meant, which is
    /// not the same as what the Tower did. A typed sentence is usually a
    /// description, the client sends it to the literal route, and the workspace
    /// says so on the answer. Deleting the case would lose the distinction and
    /// leave a person unable to tell a lexical miss from an empty memory.
    case semantic(String)

    /// What the user asked, for echoing back above the results.
    var displayText: String {
        switch self {
        case .recent: return "Recent documents"
        case .text(let text): return "Text: \(text)"
        case .observedWithin: return "By time observed"
        case .semantic(let text): return text
        }
    }

    /// The caveat an answer to this query owes the person who asked it.
    ///
    /// Only `.semantic` has one, and it is the Tower's point restated for a
    /// reader: the words were matched literally, so a description missing is
    /// not evidence the document was never seen.
    var matchingCaveat: String? {
        switch self {
        case .semantic:
            return """
                Matched literally, word for word. This memory does not search by \
                meaning, so a description of a document usually misses where a \
                phrase copied from it would hit.
                """
        case .recent, .text, .observedWithin:
            return nil
        }
    }
}

/// Where a query's words came from.
///
/// Recorded rather than ignored because a spoken query and a typed one may
/// deserve different confidence treatment later — a transcription is itself an
/// inference — and because a result set should be able to say how it was asked
/// for. Only `.appText` is reachable today.
enum DocumentQueryOrigin: String, Equatable, Sendable, CaseIterable {
    /// Typed in this app.
    case appText
    /// Submitted by another input layer on the device — a Siri intent, a
    /// shortcut, a wake word. **No such layer exists**; the case exists so that
    /// building one does not require changing this type.
    case externalIntent
}

/// How strongly the Tower's answer is supported.
///
/// `docs/modules/ENVIRONMENTAL-MEMORY.md`:
///
/// > If retrieval evidence is weak: return "not found" or uncertainty; never
/// > create a memory event retroactively to satisfy a query.
///
/// and `docs/07-PLATFORM-CONSTRAINTS.md` Core Principle 3: absence of
/// observation is not observation of absence. `.noObservation` and `.notFound`
/// are therefore different answers and are worded differently on screen — one
/// means the memory has nothing that could have matched, the other means it
/// looked and matched nothing. The Tower publishes exactly this distinction as
/// its `answer` vocabulary, and `DocumentLibraryAnswer` maps onto this type.
enum DocumentQueryEvidence: Equatable, Sendable {
    /// Matches, with the Tower's own confidence if it reported one. It does not
    /// report one for an *answer*: a BM25 score is a retrieval score and a
    /// per-document `confidence` is a word about how well a page was read.
    case matched(confidence: Double?)
    /// The memory was searched and nothing matched.
    case notFound
    /// The memory holds nothing covering what was asked, so the question cannot
    /// be answered either way. **Never rendered as "no".**
    case noObservation

    var explanation: String {
        switch self {
        case .matched(let confidence):
            guard let confidence else { return "" }
            return "Match confidence \(ObservationProvenance.percent(confidence))."
        case .notFound:
            return "Nothing in the Tower's document memory matched."
        case .noObservation:
            return """
                The Tower has no observations covering that. That is not the \
                same as the document not existing — the glasses may simply \
                never have seen it.
                """
        }
    }
}

/// One answer to one query.
struct DocumentQueryResult: Equatable, Sendable {
    let query: DocumentQuery
    let origin: DocumentQueryOrigin
    var documents: [RememberedDocument]
    var evidence: DocumentQueryEvidence
    /// The envelope the answer arrived in, when it came from a Tower.
    ///
    /// Carried rather than reduced to the three fields a row needs, because
    /// every obligation this cartridge has lives on the envelope:
    /// `recording_limitations`, `record_notes`, `no_observation_note`,
    /// `snippet_max_chars`, the retention policy, and the imagery statement. A
    /// result that dropped it would be a list of documents with every caveat
    /// removed.
    var response: DocumentLibraryResponse?

    /// Throws when the Tower's answer is internally inconsistent.
    ///
    /// ## Why this throws rather than coercing
    ///
    /// A result that claims `matched` while carrying no documents is not a
    /// coherent answer — a truncated payload, a partial decode, a contract this
    /// build half-understands. An earlier version quietly rewrote that to
    /// `.notFound`, whose user-facing sentence is *"Nothing in the Tower's
    /// document memory matched."*
    ///
    /// That is a **definite negative statement about the user's own memory,
    /// manufactured from a decode failure**, with the Tower's own confidence
    /// discarded on the way. Core Principle 3 forbids exactly that direction of
    /// travel: a gap in what we could read is not evidence about the world. The
    /// safe direction from a broken payload is a failure, not a stronger claim
    /// than the Tower made — and `CartridgeFailure.Kind.undecodableResponse`
    /// exists for it.
    init(
        query: DocumentQuery,
        origin: DocumentQueryOrigin,
        documents: [RememberedDocument] = [],
        evidence: DocumentQueryEvidence,
        response: DocumentLibraryResponse? = nil
    ) throws {
        if documents.isEmpty, case .matched = evidence {
            throw CartridgeFailure(
                kind: .undecodableResponse,
                message: "The Tower reported a match but sent no documents, so the answer could not be read."
            )
        }
        self.query = query
        self.origin = origin
        self.documents = documents
        self.evidence = evidence
        self.response = response
    }

    /// The limitations the Tower attached to this answer.
    ///
    /// Empty only when the answer did not come from a Tower. Every real
    /// response carries them, and an empty library rendered without them is the
    /// failure §8.3 exists to prevent.
    var recordingLimitations: [DocumentRecordingLimitation] {
        response?.recordingLimitations ?? []
    }
}

// MARK: - State

/// What the Document Memory workspace should be showing.
enum DocumentMemoryState: Equatable, Sendable {
    /// This Tower serves no document memory, in its own words where it gave
    /// any — which for the two configuration answers names the variable that
    /// would change it.
    case unsupported(reason: String)
    /// A memory exists and nothing has been asked of it.
    case idle
    /// A query is in flight.
    case searching(DocumentQuery)
    /// A query returned. Carries the whole result, including the "nothing
    /// matched" and "never observed" answers, which are results and not
    /// failures.
    case results(DocumentQueryResult)
    case failed(CartridgeFailure)

    var result: DocumentQueryResult? {
        if case .results(let result) = self { return result }
        return nil
    }

    var phase: CartridgePhase {
        switch self {
        case .unsupported: return .unsupported
        case .idle: return .idle
        case .searching: return .waiting
        // A returned answer is settled even when it is empty: the Tower is not
        // still working on it, and a spinner over "never observed" would
        // suggest otherwise.
        case .results: return .settled
        case .failed: return .failed
        }
    }
}

// MARK: - The session, as the workspace sees it

/// What a session verb actually did.
///
/// ## Why this is not a `Bool`
///
/// **200 is not success.** This cartridge silently no-ops a verb it cannot
/// honour: `POST /documents-session/resume` on a stopped session answers 200
/// with `state: "stopped"` and **no refusal field at all**. A caller that read
/// the status code would tell a person recording had resumed when nothing
/// resumed.
///
/// So the answer is the state that came back, compared against the states the
/// verb would have produced. `.notHonoured` is not an error — the Tower did
/// what it could and reported where the session is — and it is worded as a fact
/// rather than as a failure.
enum DocumentSessionOutcome: Equatable, Sendable {
    /// The session reached a state the verb asks for.
    case honoured(DocumentSessionStatus)
    /// 200, and the session did not move. Carries what came back.
    case notHonoured(DocumentSessionStatus, explanation: String)
    /// The request itself did not produce an answer.
    case failed(CartridgeFailure)

    var status: DocumentSessionStatus? {
        switch self {
        case .honoured(let status), .notHonoured(let status, _): return status
        case .failed: return nil
        }
    }

    /// Reads the returned state, which is the only thing that says whether the
    /// verb took effect.
    static func of(
        _ action: DocumentMemoryContract.SessionAction, reported: DocumentSessionStatus
    ) -> DocumentSessionOutcome {
        guard action.honouredStates.contains(reported.state) else {
            return .notHonoured(
                reported,
                explanation: action.silentNoOpExplanation(state: reported.state)
            )
        }
        return .honoured(reported)
    }
}
