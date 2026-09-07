//
//  DocumentMemoryContract.swift
//  Glasses
//

import Foundation

/// The two Document Memory agreements this build was written against.
///
/// ## Why two, and why they must never be collapsed into one
///
/// `document_memory.status/2026-09-07` governs the **subscription**: a small
/// payload, pushed on the result socket, describing what the capture session is
/// doing and what is on disk. `document_memory.library/2026-09-07` governs
/// **HTTP**: the documents themselves, pulled on demand.
///
/// They are separate because they are carried by different transports with
/// different failure modes, and the Tower says so out loud —
///
/// > document text is bulk and is the most sensitive data this platform holds.
/// > The result socket shares its send lock with the frame path, and a listing
/// > is pulled on demand rather than pushed.
///
/// **A change to one is not a change to the other.** A build that treated them
/// as one identifier would have to refuse a working library because a status
/// field moved, or accept a changed library because the status contract still
/// matched. Two constants, compared separately, at the two places they arrive.
///
/// Both are opaque and compared for equality only. Dated rather than numbered
/// so nobody computes which is greater: a mismatch means "we are not talking
/// about the same agreement", which is neither newer nor older.
nonisolated enum DocumentMemoryContract {
    /// The **Tower's** name for the cartridge. Not this app's catalog id
    /// (`"document-memory"`).
    static let towerCartridge = "document_memory"

    // MARK: The subscription

    /// `status`, on the socket. Note that this cartridge's result type *is*
    /// `status` — Scene Understanding is the only one where it is not.
    static let resultType = "status"
    /// Moved from 2026-08-27 on 2026-09-07, together with the library
    /// identifier, because `identity` changed meaning (see `identityScope`).
    static let statusIdentifier = "document_memory.status/2026-09-07"

    // MARK: The library

    /// The HTTP half. Declared by the Tower under `http_contracts` with an
    /// `entry_route`, which is the shape World Builder's geometry and Object
    /// Memory's observations have and are not declared under.
    static let libraryIdentifier = "document_memory.library/2026-09-07"
    static let entryRoute = "/documents"

    // MARK: The constant self-description
    //
    // Carried on every library response and every session response, and
    // asserted at decode time rather than merely read. A Tower that changed one
    // of these while keeping the identifier would be making a different promise
    // under the same name.

    /// **Not "was read".** The camera cannot establish that a wearer read
    /// anything; it can establish that a page was in view and that OCR ran over
    /// it. The contract document itself carried an older spelling that said
    /// "was read" until 2026-08-27, five keys above the note saying the
    /// opposite, and a Tower test now pins the wire value against the prose.
    static let claim = "a-page-was-in-view-and-was-ocred"
    /// Since 2026-09-07 a later look at a page already on record becomes a
    /// **sighting** of that record (`sighting_count`, `last_observed_at`,
    /// `total_observed_seconds`) when its words AND its look agree with a
    /// page of the record; anything weaker stays a separate record. The
    /// earlier value, "no-document-identity-across-sightings", described a
    /// Tower that made no such join, and both identifiers moved with it.
    static let identityScope = "same-page-by-text-and-look-within-library"
    static let absenceMeans = "not-recorded-by-this-cartridge"
    /// Every timestamp. There is no capture clock anywhere on this wire.
    static let timeBasis = "tower-receipt"

    // MARK: Routes

    /// Both the recent listing and, with an id appended, the single-document
    /// route. One constant for one path, because they are one path — the id is
    /// appended with `appendingPathComponent` so an id containing a slash or a
    /// space is percent-encoded rather than splitting the path.
    static let documentsRoute = "documents"
    static let searchRoute = "documents/search"
    static let aroundRoute = "documents/around"
    static let sessionRoute = "documents-session"

    /// The four verbs on `POST /documents-session/{action}`.
    ///
    /// **200 is not success on any of them.** Scene and Document silently no-op
    /// a verb they cannot honour: a resume on a stopped session answers 200
    /// with `state: "stopped"` and no refusal field. The returned `state` is
    /// the only thing that says whether anything moved, which is why
    /// `DocumentSessionAction` exists as a type rather than as four strings.
    enum SessionAction: String, CaseIterable, Sendable {
        case start, pause, resume, stop

        /// The state the session should be in if the verb was honoured.
        ///
        /// Compared against what came back. `stop` and `start` are the two that
        /// may legitimately take a moment — the OCR reader takes about five
        /// seconds to construct, so a start answers `starting` and reaches
        /// `running` later — so `honouredStates` is a set rather than one word.
        var honouredStates: Set<String> {
            switch self {
            case .start, .resume: return ["starting", "running"]
            case .pause: return ["paused"]
            case .stop: return ["stopped"]
            }
        }

        /// What to tell a person when the Tower answered 200 and did not move.
        ///
        /// Phrased as a fact about the session's state rather than as an error,
        /// because it is not one: the Tower honoured the request as far as it
        /// could and is reporting where the session actually is.
        func silentNoOpExplanation(state: String) -> String {
            """
            The Tower accepted the request and the session is still "\(state)". \
            This cartridge answers every verb with a success code even when it \
            cannot act on one, so what the session is doing is read from the \
            state it reported rather than from the reply itself.
            """
        }
    }
}
