//
//  DocumentMemoryLibrary.swift
//  Glasses
//

import Foundation

/// The Tower's document library and capture session, as they actually exist on
/// the wire.
///
/// Contracts: `document_memory.library/2026-09-07` over HTTP and
/// `document_memory.status/2026-09-07` on the socket. See
/// `tower/docs/contracts/CARTRIDGE-RESULTS.md` §15 and
/// `docs/contracts/TOWER-UNIFIED-CARTRIDGES.md` §8.
///
/// ## The one thing to read before anything else here
///
/// **An empty library is the expected result today.** On 9,199 frames of real
/// first-person footage the page detector fired six times and every one was a
/// false positive — a venetian blind and a backlit laptop keyboard. After the
/// gate was re-derived it fires zero times, and no capture on this platform has
/// ever contained a sheet of paper, so the detector has never been shown a
/// positive it was built for. Separately, at the 360×640 the glasses deliver,
/// EasyOCR returned zero dictionary words across 919 sampled real frames dense
/// with screen text, at median confidence 0.056.
///
/// The premise is **untested, not disproved**. Either way, a client that
/// renders an empty library as "no documents yet" is inviting a person to wait
/// for something that is not coming. Every response carries
/// `recording_limitations` saying so, and this app renders them.

// MARK: - The closed vocabulary

/// The `answer` on every library response. Exactly one, always present.
///
/// Collapsing `noObservation` into `notFound` lets a gap in what the glasses
/// happened to see read as a statement about the world. On this platform that
/// gap is the normal case, which is why `no_observation_note` is carried beside
/// it and why this enum has three cases rather than two.
enum DocumentLibraryAnswer: Equatable, Sendable {
    /// Documents were found. Render the list.
    case matched
    /// The memory was searched and nothing matched.
    case notFound
    /// The memory holds nothing that could have matched. **Never rendered as
    /// "no".**
    case noObservation
    /// A word outside `answers`. Carried rather than guessed at.
    case unrecognised(String)

    init(_ word: String) {
        switch word {
        case "matched": self = .matched
        case "not_found": self = .notFound
        case "no_observation": self = .noObservation
        default: self = .unrecognised(word)
        }
    }

    var wireValue: String {
        switch self {
        case .matched: return "matched"
        case .notFound: return "not_found"
        case .noObservation: return "no_observation"
        case .unrecognised(let word): return word
        }
    }

    /// The evidence vocabulary the workspace already speaks.
    ///
    /// `.matched(confidence: nil)` and not a number: the library publishes no
    /// numeric confidence for an *answer*. A search result carries a BM25
    /// `score` per document, which is a retrieval score and not a confidence,
    /// and a per-document `confidence` which is a word describing the worst
    /// page read. Rendering either as "match confidence 84%" would be inventing
    /// a figure the Tower never computed.
    var evidence: DocumentQueryEvidence {
        switch self {
        case .matched: return .matched(confidence: nil)
        case .notFound: return .notFound
        // An answer this build does not know is not a match. `noObservation` is
        // the weakest of the three and is therefore the safe direction — it
        // claims nothing about what was searched.
        case .noObservation, .unrecognised: return .noObservation
        }
    }
}

// MARK: - Envelope pieces

/// One measured limit on what this cartridge can record, as the Tower states
/// it.
///
/// Slugs on this cartridge: `detection-rate`, `no-validated-positive`,
/// `resolution`, `resolution-remedy-is-not-a-fix`.
struct DocumentRecordingLimitation: Equatable, Identifiable, Sendable {
    let slug: String
    let detail: String

    var id: String { slug }
}

/// When those limits were measured, and on what.
struct DocumentRecordingMeasurement: Equatable, Sendable {
    let measuredAt: String?
    let corpusFrames: Int?
    let corpusCaptures: Int?
    /// `false` on this Tower: the corpus has grown since and these have not
    /// been re-derived.
    let isCurrent: Bool
    let note: String?
}

/// `retention`, whole.
///
/// `writerWindowDays` is **always `nil`**, and honestly so: unlike Object
/// Memory's store, `DocumentStore` persists no retention manifest, so a reader
/// cannot learn the window its writer used. A `retention_days` query parameter
/// **narrows this read and can never widen what was kept** — passing a large
/// number does not recover an expired record and never will.
struct DocumentRetentionView: Equatable, Sendable {
    let requestedDays: Double?
    let writerWindowDays: Double?
    let writerWindowUnavailableReason: String?
    let policy: String?
}

/// `provenance` — a pointer into a recording, not a place.
///
/// ## The deliberate contrast, said out loud
///
/// `joinable` is `true` here, with a note, and that is not an oversight to be
/// tidied away. A capture id, the sequence numbers of the frames actually read,
/// and a timestamp together locate this reading in a recording on disk, and the
/// link is **durable across sessions** — which is precisely what Scene
/// Understanding refuses to hand anyone. The two cartridges differ here on
/// purpose: **a document is a record; a scene is not.** A screen that showed
/// this block without saying so would leave the most consequential fact about
/// it to be noticed.
struct DocumentProvenance: Equatable, Sendable {
    /// Always `"frame-reference"`.
    let kind: String?
    let captureID: String?
    /// **Always `false`.** Nothing checks that the capture still exists, so a
    /// capture id here may point at a recording that has been deleted.
    let captureIDValidated: Bool
    /// The sequence number of each frame actually read — at most two per
    /// document, because OCR costs about 1.19 s a page against 0.771 ms for
    /// detection.
    let pageSourceSeqs: [Int]
    let pagesWithoutSourceSeq: Int?
    /// Much larger than `framesOcred`, by design.
    let framesConsidered: Int?
    let framesOcred: Int?
    let worldID: String?
    let worldSessionID: String?
    /// Always `"capture-side"`: the frames this record points at live in the
    /// capture store, whose lifetime this cartridge neither sets nor enforces.
    /// Purging every document leaves that imagery exactly where it is.
    let imageryRetention: String?
    let imageryRetentionNote: String?
    /// Always `true`. See the type's note.
    let joinable: Bool
    let joinableNote: String?

    /// This app's own sentence about what the block means, for a screen rather
    /// than for a log. The Tower's `joinableNote` is shown beside it where it
    /// exists; this is what makes the contrast with Scene Understanding
    /// explicit rather than implicit.
    static let joinabilityHeadline = """
        This record points back into a recording on disk. A capture id, the \
        frame numbers that were read, and a time locate this reading inside \
        that recording, and the link keeps working after the session ends. \
        That is deliberate here and deliberately absent from the scene \
        cartridge: a document is a record, a scene is not.
        """
}

/// `timing` — how a duration was arrived at.
///
/// A duration derived from an assumed frame interval is a **reconstruction**
/// and must not be rendered identically to a measured one, which is why
/// `source` is carried rather than dropped.
struct DocumentTiming: Equatable, Sendable {
    let timeBasis: String?
    /// `capture-journal`, `assumed-interval`, or `mixed`.
    let source: String?
    let assumedFrameIntervalSeconds: Double?
    let note: String?

    var isReconstructed: Bool { source == "assumed-interval" || source == "mixed" }
}

/// One page inside one document. Only `GET /documents/{id}` carries these.
struct DocumentPage: Equatable, Identifiable, Sendable {
    let pageIndex: Int
    /// What OCR read. **An empty string is `not_readable`, not "no page".**
    let text: String
    /// An enum of one today (`"ocr"`), stated so a second source is a visible
    /// change rather than a silent one.
    let textSource: String?
    /// Text regions the recogniser returned. `0` with empty text is the
    /// readable-nothing case, which is a real answer.
    let regionCount: Int
    let meanRegionConfidence: Double?
    let minRegionConfidence: Double?
    /// A word, derived from the mean — one hard word should not condemn a page.
    let confidence: String?
    let sharpness: Double?
    let squareness: Double?
    /// The frame this page was read from. `nil` on a record written before
    /// provenance existed.
    let sourceSeq: Int?
    /// Tower-receipt time of that frame.
    let observedAt: Double?
    /// How many separate views of this page were merged into it. Two readings
    /// of one page during one dwell is one page with a count of two, not two
    /// pages.
    let observationCount: Int
    /// Whether a page image exists on the Tower's disk.
    ///
    /// **`false` unless page images were explicitly enabled**, which is off by
    /// default and must stay off: this platform performs no redaction, so a
    /// stored page image is an unredacted photograph of what the wearer was
    /// reading.
    let imageKept: Bool
    /// Always `false`. A **boolean**, not a path — the path told a reader where
    /// in the store to find that photograph, which is disclosure with no
    /// consumer, since no route resolves it and none may.
    let imageServed: Bool

    var id: Int { pageIndex }
}

/// `coverage` — how much of the document was **captured**, never how much
/// exists.
struct DocumentCoverage: Equatable, Sendable {
    let pagesObserved: Int?
    /// **Always `nil`.** The camera cannot know how many pages a document has.
    let pagesTotal: Int?
    let pagesTotalNote: String?
    let wordsCaptured: Int?
    /// Page indices whose confidence is `low` or `unknown`.
    let lowConfidencePages: [Int]
}

/// The four search-only fields on a matched document.
struct DocumentMatchEvidence: Equatable, Sendable {
    /// BM25, rounded to four places. **A retrieval score, not a confidence.**
    let score: Double?
    let matchedTerms: [String]
    /// A bounded window around the first matched term.
    ///
    /// Evidence, not an excerpt: a match with no evidence is a number a client
    /// has to trust. The cap is published beside it on the envelope as
    /// `snippet_max_chars` and is **read rather than assumed** — see
    /// `DocumentLibraryResponse.snippetMaxChars`.
    let snippet: String?
}

/// The `query` block, echoed back so a response is self-describing.
struct DocumentQueryEcho: Equatable, Sendable {
    /// One of `retrieval_kinds`, or `"document"` on the single-document route.
    let kind: String?
    /// The effective cap after the route's own bound. Truncation is detectable
    /// by comparing it with `documentCount`.
    let limit: Int?
    let text: String?
    let centre: Double?
    let windowSeconds: Double?
    let documentID: String?
}

// MARK: - The library response

/// One complete answer from any of the four `/documents*` routes.
///
/// ## `record_notes` is hoisted, and none of it may be dropped
///
/// Five caveats used to be repeated on every document — the "in view, not read"
/// qualification, the summary's provenance, the clock, the capture-side imagery
/// lifetime, and the joinability of the frame reference. A 200-document listing
/// was 488 KB with two thirds of it the same sentences two hundred times;
/// hoisting them to the envelope cut it to 249 KB **with nothing dropped**.
///
/// They are keyed by the field they qualify, and a client must render a
/// document with them beside it. Deleting a caveat to save bytes is the one
/// saving this contract may not make, and the hoist is what made keeping them
/// affordable.
struct DocumentLibraryResponse: Equatable, Sendable {
    let contract: String
    let claim: String
    let identity: String
    let absenceMeans: String
    let timeBasis: String
    /// Always `nil`. This cartridge does not know where anything is.
    let spatialRef: String?

    let answers: [String]
    let retrievalKinds: [String]
    /// **`false`.** BM25 over literal terms. Calling it semantic would be an
    /// overclaim.
    let semanticRetrieval: Bool
    let semanticRetrievalUnavailableReason: String?
    /// What to do instead: route free text to `text`. It will be matched
    /// literally, so a *description* of a document usually misses where a
    /// *quotation* from one hits — which is the sentence a person typing a
    /// description needs to see.
    let semanticRetrievalAlternative: String?

    /// `supported: false` — there is no cursor. `limit` is the only bound.
    let paginationSupported: Bool
    let paginationBound: String?
    let paginationReason: String?

    let privacyTagVocabulary: [String]
    /// Keyed by the field each caveat qualifies: `observed_seconds`,
    /// `summary_withheld`, `timing`, `imagery_retention`, `joinable`.
    let recordNotes: [String: String]
    let recordingLimitations: [DocumentRecordingLimitation]
    let recordingMeasurement: DocumentRecordingMeasurement

    /// `none-retained` or `raw-persisted`. It varies with the fact.
    let imageryTreatment: String?
    /// `rawEphemeral` — named in this app's own vocabulary so the mapping is
    /// the Tower's decision and not the phone's guess. **Never `redacted`:**
    /// this platform performs no redaction.
    let imageryIOSState: String?
    /// **Always `false`, and a boolean rather than a path.** No route serves an
    /// image.
    let imageryServed: Bool
    let imageryNote: String?

    let retention: DocumentRetentionView
    let query: DocumentQueryEcho
    let answer: DocumentLibraryAnswer
    /// The sentence that goes with `noObservation`. Rendered, not paraphrased.
    let noObservationNote: String?

    /// Every parseable record on disk, through this read's retention window.
    let documentsInMemory: Int
    /// How many are in `documents`.
    let documentCount: Int
    let documents: [RememberedDocument]

    // Search-only.
    let searchedDocuments: Int?
    let minScore: Double?
    /// `false` with `noObservation` is an empty memory; `false` with `notFound`
    /// is a query whose terms nothing contained. Two different things.
    let sufficientEvidence: Bool?
    let matchKind: String?
    let reason: String?
    /// **48 on this Tower, and read rather than hard-coded** (§12.10). The
    /// contract prose said 160 until 2026-08-27 while the code said 48; the
    /// instruction that resolved it was "read the field, not the prose".
    let snippetMaxChars: Int?

    // Single-document only.
    let document: RememberedDocument?
    let pages: [DocumentPage]
    let coverage: DocumentCoverage?

    /// Whether the listing was cut short.
    ///
    /// There is no cursor, so this is the only way to tell. On `recent`,
    /// compare `documentCount` with `documentsInMemory`; on `text` and
    /// `observed_within` those two differ because of the query as well, so
    /// compare `documentCount` with the effective `limit` instead. The Tower
    /// publishes that rule in `pagination.reason` rather than leaving a client
    /// to work it out.
    var isPossiblyTruncated: Bool {
        switch query.kind {
        case "recent": return documentCount < documentsInMemory
        case "text", "observed_within":
            guard let limit = query.limit else { return false }
            return documentCount >= limit
        default: return false
        }
    }
}

// MARK: - The session

/// `GET /documents-session` and the `session` block on the subscription.
///
/// **The block keeps its shape in every state.** When no session exists every
/// field is present and `null`, `state` is `"unavailable"`, and `states`
/// carries the full vocabulary including that value. A block that changed shape
/// forced a decoder to make thirty fields optional and to handle a `state` its
/// own enum denied existed — in the one shape that did not carry the enum.
struct DocumentSessionStatus: Equatable, Sendable {
    let state: String
    let states: [String]
    let sessionID: Int?
    let failureReason: String?
    let startedAt: Double?
    let readyAt: Double?
    let loadingSeconds: Double?
    /// Not a failure. See `SceneLifecycle.loadOverdue` for why; the OCR reader
    /// takes about five seconds to construct, so a start is not synchronous.
    let loadOverdue: Bool
    let loadOverdueAfterSeconds: Double?
    /// The text recogniser in use. Agrees with `recogniser` by construction —
    /// one is the generic lifecycle field every live session carries and the
    /// other is this cartridge's name for the same thing.
    let engine: String?
    let recogniser: String?

    let framesOffered: Int?
    let framesObserved: Int?
    let framesSkipped: Int?
    let framesDroppedNotRunning: Int?

    let captureID: String?
    /// Always `false`. Nothing checks the capture still exists.
    let captureIDValidated: Bool
    /// A dwell in progress. **Stopping flushes it rather than dropping it** — a
    /// wearer still reading when a session stops has read something.
    let inDwell: Bool
    let dwellsStarted: Int?
    /// Expected to stay at zero on this platform. See the file's header.
    let pagesDetected: Int?
    let documentsRecorded: Int?
    let lastDocumentID: String?
    let lastDocumentAt: Double?
    /// The id of the dwell that was flushed by the last stop, when there was
    /// one. This is the field that makes "stop keeps documents" observable
    /// rather than merely asserted.
    let flushedDocumentID: String?
    /// **`false`, and it must stay false.** No redaction exists on this
    /// platform, so a kept page image is an unredacted photograph of what the
    /// wearer was reading.
    let keepsPageImages: Bool
    /// **Defaults to `false` for this cartridge**, the opposite of Scene
    /// Understanding's. The asymmetry is the difference between the two: this
    /// one writes, and a session that persists what a wearer read gets an
    /// explicit start.
    let followsStream: Bool

    let retentionDays: Double?
    let documentsPruned: Int?
    /// Reported rather than logged: a deletion that quietly failed looks
    /// exactly like one that was kept.
    let retentionIncomplete: Bool
    /// The library count **through the session's retention window**, refreshed
    /// only by a prune. A different quantity from
    /// `DocumentLibrarySummary.documentCountUnfiltered`, and named apart for
    /// that reason.
    let libraryCount: Int?
    /// **Reported, never enforced.** This session evicts by age only: deleting
    /// a wearer's memories because a count grew is a policy decision, not a
    /// cleanup, and this cartridge declined to make it.
    let librarySoftLimit: Int?
    let libraryOverSoftLimit: Bool
    let librarySoftLimitNote: String?
    /// Why there is no session, when `state == "unavailable"`.
    let reason: String?
    /// Where the text recogniser actually loaded (`"cuda"` / `"cpu"`), once
    /// running. `TOWER_DOCUMENT_DEVICE` asks; this reports.
    var ocrDevice: String? = nil
    /// Dwells this session merged onto a record that already existed, as
    /// sightings. `documentsRecorded` counts only NEW records.
    var documentsResighted: Int? = nil
    /// Dwells whose every page OCR looked at and could not read. Persisted
    /// with `readable: false`; a session where this climbs while
    /// `documentsRecorded` does not is pointed at something that is not a page.
    var dwellsUnreadable: Int? = nil
    /// True while the Tower's idle timer runs: the stream closed and the
    /// session will stop itself unless one reopens.
    var idleStopPending: Bool = false

    var isUnavailable: Bool { state == "unavailable" }
    /// The one state in which frames are being read.
    var isRunning: Bool { state == "running" }
}

/// The `library` block: what is on disk, **regardless of whether anything is
/// running**.
///
/// A Tower with a library and no session is a normal configuration — it serves
/// documents recorded elsewhere and records nothing itself — not a degraded
/// one.
struct DocumentLibrarySummary: Equatable, Sendable {
    let available: Bool
    /// Every parseable record on disk. `retentionApplied` is always `false`,
    /// which is why the field is named "unfiltered".
    let documentCountUnfiltered: Int?
    let retentionApplied: Bool
    let unavailableReason: String?
    let newestObservedAt: Double?
    let journalBytes: Int?
    let imageBytes: Int?
    let totalBytes: Int?
    /// Always `false`. The Tower does not disclose where on disk the library
    /// lives.
    let locationDisclosed: Bool
    /// The live-update trigger (`document_memory.status/2026-09-07`). Opaque;
    /// changes whenever the Tower's journal changes — an append, a sighting
    /// merged onto a record, a prune — and never otherwise. A listing is
    /// re-fetched when this moves, and never on a timer. `nil` from a Tower
    /// that does not send it.
    var revision: Int? = nil
}

/// The whole `document_memory.status` subscription payload.
struct DocumentMemoryStatus: Equatable, Sendable {
    /// Carried **in** the payload so a client that reads only this channel
    /// still learns the documents are elsewhere.
    let contractNote: String?
    let claim: String
    let identity: String
    let absenceMeans: String
    let timeBasis: String
    let library: DocumentLibrarySummary
    let session: DocumentSessionStatus
    let recordingLimitations: [DocumentRecordingLimitation]
    let recordingMeasurement: DocumentRecordingMeasurement
    let imageryTreatment: String?
    let imageryIOSState: String?
    let imageryServed: Bool
    let imageryNote: String?
}

// MARK: - Transport failures

/// Why a request to the Tower's document memory did not produce an answer.
///
/// ## The two 404s are configuration answers, and neither is about a document
///
/// **`/documents*` answer 404 when `TOWER_DOCUMENT_ROOT` is unset.**
/// **`/documents-session*` answer 404 when `TOWER_DOCUMENT_CAPTURE` is off even
/// with a root set** — a root with capture off is a Tower serving a library
/// recorded elsewhere and recording nothing itself, which is a normal
/// configuration.
///
/// Both name the variable that would change them. Neither is ever the answer to
/// a query *about a document*: that is answered 200 with
/// `answer: "no_observation"`, precisely so a 404 can keep meaning only this.
///
/// The one 404 that **is** about a resource is `GET /documents/{id}` for an id
/// this memory has never held, and it does not name a variable. So the rule
/// this enum encodes is the contract's own: **a 404 with no `TOWER_` in it, on
/// any route but that one, is a genuine routing bug** and is reported as such
/// rather than swallowed as "not configured".
enum DocumentMemoryFetchError: Error, Equatable {
    /// **404**, `TOWER_DOCUMENT_ROOT` unset. This Tower serves no documents at
    /// all. Carries the Tower's own sentence, which names the variable.
    case noDocumentRootConfigured(String)
    /// **404**, `TOWER_DOCUMENT_CAPTURE` off. This Tower serves a library and
    /// records nothing. The documents routes may still answer.
    case noCaptureSessionConfigured(String)
    /// **404** on `GET /documents/{id}` for an id this memory has never held.
    /// A fact about the resource, not about configuration.
    case noSuchDocument(id: String, detail: String)
    /// **404** naming no `TOWER_` variable, on a route where that cannot be a
    /// resource answer. Surfaced as a defect rather than as "not configured",
    /// because telling a person their Tower is unconfigured when the app is
    /// asking for a route that does not exist sends them somewhere useless.
    case routingBug(String)
    /// The Tower's document memory speaks a different agreement. Opaque and
    /// compared for equality: not newer, not older, just different.
    case unsupportedContract(identifier: String)
    /// The answer arrived and could not be read as this contract.
    case undecodable
    /// The request did not complete. The Tower is unreachable, or the network
    /// is.
    case transport(String)

    /// What to show a person, and what kind of failure it is.
    ///
    /// The mapping matters: only `transport` is a connection problem, and
    /// rendering a configuration answer as a disconnection would send someone
    /// to check their network over a variable that is unset.
    var failure: CartridgeFailure {
        switch self {
        case .noDocumentRootConfigured(let detail),
             .noCaptureSessionConfigured(let detail):
            return CartridgeFailure(kind: .notSupported, message: detail)
        case .noSuchDocument(_, let detail):
            return CartridgeFailure(kind: .towerReportedFailure, message: detail)
        case .routingBug(let detail):
            return CartridgeFailure(
                kind: .undecodableResponse,
                message: """
                    The Tower answered 404 for a document route without naming a \
                    configuration variable, which means the route itself was not \
                    found: \(detail)
                    """
            )
        case .unsupportedContract(let identifier):
            return CartridgeFailure(
                kind: .notSupported,
                message: """
                    The Tower offers a document memory contract this version of the \
                    app does not understand (\(identifier)). Nothing is shown rather \
                    than something guessed.
                    """
            )
        case .undecodable:
            return CartridgeFailure(
                kind: .undecodableResponse,
                message: """
                    The Tower answered, and the answer could not be read as the \
                    document contract this build implements.
                    """
            )
        case .transport(let message):
            return CartridgeFailure(kind: .transport, message: message)
        }
    }
}

// MARK: - HTTP

/// The six routes, and nothing else.
///
/// ## Read-only for the library, write-only for the session
///
/// There is no delete and no purge on the wire, and this client has no method
/// that could grow into one. An unauthenticated LAN endpoint that erases a
/// wearer's reading history is not a feature, and neither is a phone button
/// that calls one.
///
/// What the session verbs *do* is bounded the other way: they start and stop a
/// **recorder**, and a stop keeps everything it recorded.
///
/// ## Bounded, and uncached
///
/// Every request carries an explicit timeout (Rule 15: bounded operations) and
/// `reloadIgnoringLocalCacheData`. A memory query answered out of a URL cache
/// would show a record as current that the retention window may since have
/// closed over.
///
/// Mirrors `ObjectMemoryHTTPClient` and `WorldGeometryClient` deliberately —
/// same shape, same `JSONSerialization` decoding, same "a 404 is its own case"
/// handling, same injectable `URLSession`. A second, differently-opinionated
/// networking layer in one app is two places for a timeout policy to be wrong.
nonisolated struct DocumentMemoryHTTPClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession = .shared
    /// Long enough for a Tailscale round trip to a Tower parsing a JSONL file —
    /// every query re-parses the journal, there is no index — and short enough
    /// that a dead Tower becomes a visible state rather than a spinner nobody
    /// can end.
    var timeout: TimeInterval = 10

    // MARK: The library

    /// The most recent documents, newest first, **without their text**.
    func recent(limit: Int? = nil, retentionDays: Double? = nil) async throws
        -> DocumentLibraryResponse {
        var query: [URLQueryItem] = []
        if let limit { query.append(URLQueryItem(name: "limit", value: String(limit))) }
        appendRetention(retentionDays, to: &query)
        let json = try await get(
            baseURL.appendingPathComponent(DocumentMemoryContract.documentsRoute),
            query: query,
            route: .library
        )
        return try decode(json)
    }

    /// Literal term matching over the concatenated page text, with bounded
    /// snippets.
    ///
    /// **Not semantic.** The Tower computes no embedding, and a description of
    /// a document will usually miss where a quotation from one will hit. That
    /// sentence is on the envelope as `semantic_retrieval_alternative` and the
    /// workspace renders it, because a person typing a description is exactly
    /// who needs to read it.
    func search(text: String, limit: Int? = nil, retentionDays: Double? = nil) async throws
        -> DocumentLibraryResponse {
        var query = [URLQueryItem(name: "text", value: text)]
        if let limit { query.append(URLQueryItem(name: "limit", value: String(limit))) }
        appendRetention(retentionDays, to: &query)
        let json = try await get(
            baseURL.appendingPathComponent(DocumentMemoryContract.searchRoute),
            query: query,
            route: .library
        )
        return try decode(json)
    }

    /// Documents observed within a window of an instant.
    ///
    /// A **range**, not an instant: "this morning" and "around lunch" are
    /// approximate, and answering them exactly answers a different question.
    func around(
        at instant: Double, windowSeconds: Double? = nil,
        limit: Int? = nil, retentionDays: Double? = nil
    ) async throws -> DocumentLibraryResponse {
        var query = [URLQueryItem(name: "at", value: String(instant))]
        if let windowSeconds {
            query.append(URLQueryItem(name: "window_seconds", value: String(windowSeconds)))
        }
        if let limit { query.append(URLQueryItem(name: "limit", value: String(limit))) }
        appendRetention(retentionDays, to: &query)
        let json = try await get(
            baseURL.appendingPathComponent(DocumentMemoryContract.aroundRoute),
            query: query,
            route: .library
        )
        return try decode(json)
    }

    /// One document, with its pages and their text. **The only route that
    /// carries text**, and the only one whose 404 is about a resource.
    func document(id: String, retentionDays: Double? = nil) async throws
        -> DocumentLibraryResponse {
        var query: [URLQueryItem] = []
        appendRetention(retentionDays, to: &query)
        let url = baseURL
            .appendingPathComponent(DocumentMemoryContract.documentsRoute)
            .appendingPathComponent(id)
        let json = try await get(url, query: query, route: .oneDocument(id: id))
        return try decode(json)
    }

    // MARK: The session

    func sessionStatus() async throws -> DocumentSessionStatus {
        let json = try await get(
            baseURL.appendingPathComponent(DocumentMemoryContract.sessionRoute),
            query: [],
            route: .session
        )
        try requireLibraryContract(json)
        guard
            let raw = json["session"] as? [String: Any],
            let session = DocumentMemoryDecoder.session(from: raw)
        else { throw DocumentMemoryFetchError.undecodable }
        return session
    }

    /// Sends one verb, and returns **the state the Tower reported afterwards**.
    ///
    /// The return value is the whole point. `POST /documents-session/resume` on
    /// a stopped session answers **200 with `state: "stopped"` and no refusal
    /// field**, and the same is true of a pause on a stopped session and a stop
    /// on a stopped one. This cartridge silently no-ops a verb it cannot
    /// honour, so a caller that treated 200 as "it worked" would tell a person
    /// they had started recording when nothing started.
    ///
    /// `DocumentMemoryContract.SessionAction.honouredStates` is what turns the
    /// returned state into an answer.
    func sendSession(
        _ action: DocumentMemoryContract.SessionAction
    ) async throws -> DocumentSessionStatus {
        let url = baseURL
            .appendingPathComponent(DocumentMemoryContract.sessionRoute)
            .appendingPathComponent(action.rawValue)
        let json = try await post(url, route: .session)
        try requireLibraryContract(json)
        guard
            let raw = json["session"] as? [String: Any],
            let session = DocumentMemoryDecoder.session(from: raw)
        else { throw DocumentMemoryFetchError.undecodable }
        return session
    }

    // MARK: Plumbing

    /// Which route a 404 came from, which is what decides what a 404 *means*.
    ///
    /// Internal rather than private so `meaning(ofNotFound:route:)` can be
    /// tested directly against the three real bodies this Tower produces,
    /// without standing up a URL loading system to deliver them.
    enum Route: Equatable, Sendable {
        case library
        case oneDocument(id: String)
        case session
    }

    /// `0` is a meaningful value — "no limit of my own", still clamped to what
    /// the writer kept — so it is only omitted when the caller asked for
    /// nothing. A negative value is refused here rather than sent for the route
    /// to answer 422.
    private func appendRetention(_ days: Double?, to query: inout [URLQueryItem]) {
        guard let days, days >= 0 else { return }
        query.append(URLQueryItem(name: "retention_days", value: String(days)))
    }

    private func decode(_ json: [String: Any]) throws -> DocumentLibraryResponse {
        try requireLibraryContract(json)
        guard let response = DocumentMemoryDecoder.library(from: json) else {
            throw DocumentMemoryFetchError.undecodable
        }
        return response
    }

    /// Equality, and nothing else.
    ///
    /// Against `libraryIdentifier` and never against `statusIdentifier`: the
    /// two govern different transports and a change to one is not a change to
    /// the other. `/documents-session` carries the **library** identifier as
    /// well, which is what the wire says and what this checks.
    private func requireLibraryContract(_ json: [String: Any]) throws {
        guard let identifier = json["contract"] as? String else {
            throw DocumentMemoryFetchError.undecodable
        }
        guard identifier == DocumentMemoryContract.libraryIdentifier else {
            throw DocumentMemoryFetchError.unsupportedContract(identifier: identifier)
        }
    }

    private func get(_ url: URL, query: [URLQueryItem], route: Route) async throws
        -> [String: Any] {
        var request = try self.request(url, query: query)
        request.httpMethod = "GET"
        return try await send(request, route: route)
    }

    private func post(_ url: URL, route: Route) async throws -> [String: Any] {
        var request = try self.request(url, query: [])
        request.httpMethod = "POST"
        return try await send(request, route: route)
    }

    private func request(_ url: URL, query: [URLQueryItem]) throws -> URLRequest {
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            throw DocumentMemoryFetchError.transport("The Tower address could not be read as a URL.")
        }
        // `nil` rather than `[]`: an empty `queryItems` array puts a bare `?` on
        // the URL, and `nil` is what "no query" means.
        components.queryItems = query.isEmpty ? nil : query
        guard let requestURL = components.url else {
            throw DocumentMemoryFetchError.transport("The request URL could not be built.")
        }
        return URLRequest(
            url: requestURL,
            cachePolicy: .reloadIgnoringLocalCacheData,
            timeoutInterval: timeout
        )
    }

    private func send(_ request: URLRequest, route: Route) async throws -> [String: Any] {
        do {
            let (data, response) = try await session.data(for: request)
            if let http = response as? HTTPURLResponse, http.statusCode == 404 {
                throw Self.meaning(ofNotFound: data, route: route)
            }
            if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                throw DocumentMemoryFetchError.transport("The Tower answered \(http.statusCode).")
            }
            // Parsed in its own `do`, because `jsonObject(with:)` *throws* on a
            // malformed body rather than returning something the cast rejects.
            // Left to the `catch` below that throw would be relabelled
            // `.transport` — a claim about the network that is false when the
            // Tower answered.
            let parsed: Any
            do {
                parsed = try JSONSerialization.jsonObject(with: data)
            } catch {
                throw DocumentMemoryFetchError.undecodable
            }
            guard let json = parsed as? [String: Any] else {
                throw DocumentMemoryFetchError.undecodable
            }
            return json
        } catch let error as DocumentMemoryFetchError {
            throw error
        } catch {
            throw DocumentMemoryFetchError.transport(error.localizedDescription)
        }
    }

    /// What a 404 from this Tower actually means.
    ///
    /// The order of the checks is the wire's: the session route's detail names
    /// **both** variables — *"TOWER_DOCUMENT_CAPTURE is off, or
    /// TOWER_DOCUMENT_ROOT is unset"* — so capture is tested first, or every
    /// session 404 would be reported as a missing root and send an operator to
    /// set a variable that is already set.
    static func meaning(ofNotFound body: Data, route: Route) -> DocumentMemoryFetchError {
        let detail = self.detail(from: body)
        if detail.contains("TOWER_DOCUMENT_CAPTURE") {
            return .noCaptureSessionConfigured(detail)
        }
        if detail.contains("TOWER_DOCUMENT_ROOT") {
            return .noDocumentRootConfigured(detail)
        }
        if case .oneDocument(let id) = route {
            // The one 404 on this cartridge that is genuinely about a resource:
            // an id that names nothing is a client asking about a document this
            // Tower has never held, which is different from a query that
            // matched nothing — and the list routes never 404 for an empty
            // result.
            return .noSuchDocument(id: id, detail: detail)
        }
        return .routingBug(detail)
    }

    /// FastAPI's `{"detail": "…"}`, or the raw body when it is not that.
    private static func detail(from body: Data) -> String {
        if let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
           let detail = json["detail"] as? String {
            return detail
        }
        return String(data: body, encoding: .utf8) ?? "The Tower answered 404 with no detail."
    }
}
