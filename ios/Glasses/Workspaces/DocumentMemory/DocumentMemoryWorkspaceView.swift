//
//  DocumentMemoryWorkspaceView.swift
//  Glasses
//

import Foundation
import SwiftUI

/// The Document Memory workspace: pages the glasses passed and OCR read, and a
/// way to ask about them.
///
/// ## The one thing this screen must not do
///
/// Render an empty library as "no documents yet".
///
/// > An empty library is the expected result today. A client that renders an
/// > empty library as "no documents yet" is inviting a person to wait for
/// > something that is not coming.
///
/// On 9,199 frames of real first-person footage the page detector fired six
/// times and every one was a false positive — a venetian blind and a backlit
/// laptop keyboard. After the gate was re-derived it fires zero times, and no
/// capture on this platform has ever contained a sheet of paper. Separately, at
/// the 360×640 the glasses deliver, OCR returned zero dictionary words across
/// 919 sampled real frames dense with screen text.
///
/// So an empty answer here is **"never observed"**, with the Tower's own
/// `recording_limitations` attached — and "never observed" is said explicitly
/// not to mean the document does not exist.
///
/// ## The three answers are three sentences
///
/// `matched` renders the list, `not_found` says the memory was searched and
/// nothing matched, `no_observation` says the memory holds nothing that could
/// have matched. Collapsing the third into the second lets a gap in what the
/// glasses happened to see read as a statement about the world.
///
/// ## No camera controls
///
/// The recorder's Start/Stop is here, and the camera's is not: this cartridge
/// writes, its session does **not** follow the stream, and a session that
/// persists what a wearer read gets an explicit start. Starting the recorder
/// does not start the camera, and this screen cannot.
struct DocumentMemoryWorkspaceView: View {
    /// A value, not the connection — see `TowerReachabilityReader`.
    let isTowerReachable: Bool

    @StateObject private var memory: DocumentMemoryViewModel

    /// The client is injected and owned by `ProjectManager`; see
    /// `CartridgeClients`. The camera is the same optional owner Object
    /// Memory's coordinator receives: present in DEBUG, `nil` in Release.
    init(
        isTowerReachable: Bool,
        client: any DocumentMemoryClient,
        camera: (any ObjectMemoryCaptureOwner)? = nil,
        cameraClaim: CartridgeCameraClaim? = nil
    ) {
        self.isTowerReachable = isTowerReachable
        _memory = StateObject(
            wrappedValue: DocumentMemoryViewModel(
                client: client, camera: camera, cameraClaim: cameraClaim
            )
        )
    }

    private var availability: CartridgeAvailability {
        memory.availability(isTowerReachable: isTowerReachable)
    }

    var body: some View {
        VStack(spacing: 16) {
            header
            searchField

            if let forcedPhase = availability.forcedPhase {
                CartridgeStatePanel(
                    title: "Documents",
                    phase: forcedPhase,
                    explanation: memory.unavailableExplanation(isTowerReachable: isTowerReachable),
                    futureDescription: Self.futureDescription
                )
            } else {
                resultsPanel
                if let session = memory.session {
                    DocumentSessionPanel(
                        session: session,
                        outcome: memory.lastSessionOutcome,
                        cameraNote: memory.cameraNote,
                        send: { memory.send($0) }
                    )
                }
            }

            if let failure = memory.lastRequestFailure {
                FailureBanner(text: failure.message)
            }
        }
        // One shot, not a poll: the subscription pushes the session twice a
        // second while it is open, and this covers the case where it is not.
        // The recent listing is asked for once, on first appearance; after
        // that the library's `revision` on the status push re-asks whatever
        // is showing whenever the Tower's journal changes.
        .task { memory.appear() }
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Document Memory")
                .font(.title2.weight(.semibold))
            // "In view and read by OCR", never "read by you". The Tower's own
            // `claim` says the same: `a-page-was-in-view-and-was-ocred`.
            Text("Pages that were in front of the camera long enough for the Tower to run text recognition over them. Being in view is not the same as your having read it, and the glasses cannot tell the difference.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    private static let futureDescription = """
        When this Tower serves a document library, this workspace lists what it \
        recorded, when, and how long each page was in view, with whether text \
        was readable. Searching happens on the Tower and matches words \
        literally.
        """

    // MARK: Query input

    @ViewBuilder
    private var searchField: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
                TextField("Words from a page you passed", text: $memory.queryText)
                    .textFieldStyle(.plain)
                    .submitLabel(.search)
                    .onSubmit { memory.submitTypedQuery() }
                    .disabled(!availability.isAvailable)
                if availability.isAvailable {
                    Button("Recent") { memory.showRecent() }
                        .font(.subheadline.weight(.medium))
                        .buttonStyle(.borderless)
                }
            }
            .padding(12)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 12))

            // Two reasons the field can be disabled, and they must not share a
            // sentence. "Needs a Tower that keeps a document memory" is true of
            // a Tower that never will; it is the wrong thing to tell someone
            // whose Tower simply is not connected.
            if availability == .towerUnreachable {
                Text("Searching needs the Tower. Nothing typed here is sent anywhere.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if !availability.isAvailable {
                Text("Searching needs a Tower that keeps a document library. Nothing typed here is sent anywhere.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                // Said before a person types, not after they get nothing back.
                Text("Matched word for word, on the Tower. A phrase copied from the page will hit where a description of it usually misses.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .contain)
    }

    // MARK: Results

    @ViewBuilder
    private var resultsPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("Documents")

            VStack(alignment: .leading, spacing: 12) {
                switch memory.state {
                case .unsupported(let reason):
                    Text(reason)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                case .idle:
                    Text("Nothing has been asked yet. Search above, or list what was recorded most recently.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                case .searching(let query):
                    HStack(spacing: 10) {
                        ProgressView()
                        Text("Searching for \(query.displayText)…")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }

                case .results(let result):
                    DocumentResultsView(result: result)

                case .failed(let failure):
                    Label("Search failed", systemImage: "exclamationmark.triangle.fill")
                        .font(.headline)
                    Text(failure.message)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        }
    }
}

// MARK: - Results

/// Renders one query's answer, including the two answers that are not lists.
struct DocumentResultsView: View {
    let result: DocumentQueryResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            switch result.evidence {
            case .matched:
                ForEach(result.documents) { document in
                    DocumentRow(
                        document: document,
                        notes: result.response?.recordNotes ?? [:],
                        snippetMaxChars: result.response?.snippetMaxChars
                    )
                }
                if let caveat = result.query.matchingCaveat {
                    Text(caveat)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if result.response?.isPossiblyTruncated == true,
                   let reason = result.response?.paginationReason {
                    // There is no cursor, so a truncated listing can only be
                    // detected by arithmetic. The Tower publishes that
                    // arithmetic rather than leaving a client to invent it.
                    Text("This list may be cut short. \(reason)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }

            case .notFound, .noObservation:
                // Two different sentences for two different facts. Collapsing
                // them into "No results" would let a gap in what the glasses
                // happened to see read as a statement about the world.
                Label(
                    result.evidence == .notFound ? "Nothing matched" : "Never observed",
                    systemImage: result.evidence == .notFound ? "magnifyingglass" : "eye.slash"
                )
                .font(.headline)

                // The Tower's own sentence wins where it has one, because it
                // is more specific: it says *this Tower has recorded no
                // documents at all*, which is a stronger and more useful
                // statement than the generic one.
                Text(result.response?.noObservationNote ?? result.evidence.explanation)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                if result.evidence == .noObservation {
                    // Said explicitly, every time, because this is the one
                    // sentence the closed vocabulary exists for.
                    Text("Never observed is not the same as the document not existing. The glasses may simply never have seen it.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if let response = result.response,
                   response.sufficientEvidence == false,
                   result.evidence == .notFound {
                    // `sufficient_evidence: false` with `not_found` is a query
                    // whose terms nothing contained; with `no_observation` it
                    // is an empty memory. Two different things, and the Tower
                    // distinguishes them so a client can.
                    Text("The memory was searched and none of those words appeared in it.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            // Attached to every answer, not just the empty ones. A list of two
            // documents from a detector that has never fired on a real page is
            // as much in need of this as a list of none.
            DocumentLimitationsView(
                limitations: result.recordingLimitations,
                measurement: result.response?.recordingMeasurement,
                isEmptyAnswer: result.documents.isEmpty
            )

            if let response = result.response {
                DocumentEnvelopeFootnotes(response: response)
            }
        }
    }
}

// MARK: - The limitation that must not be hidden

/// `recording_limitations`, rendered where a person sees them.
///
/// This appeared **zero times** in this app before this screen was rewritten,
/// while every response carried it. It is the single most important thing this
/// cartridge has to say, and it is most important precisely when there is
/// nothing else on screen.
struct DocumentLimitationsView: View {
    let limitations: [DocumentRecordingLimitation]
    var measurement: DocumentRecordingMeasurement?
    /// Whether the answer this accompanies had no documents in it. Changes the
    /// heading, not the content: an empty answer is the expected one, and the
    /// heading should say so rather than let a reader supply "yet".
    var isEmptyAnswer: Bool

    var body: some View {
        if !limitations.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Label(heading, systemImage: "exclamationmark.circle")
                    .font(.subheadline.weight(.medium))

                ForEach(limitations) { limitation in
                    Text(limitation.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if let measurement, let measuredAt = measurement.measuredAt {
                    Text(footnote(measuredAt, measurement))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(12)
            .background(Color(.tertiarySystemGroupedBackground), in: .rect(cornerRadius: 12))
            .accessibilityElement(children: .combine)
        }
    }

    private var heading: String { Self.heading(isEmptyAnswer: isEmptyAnswer) }

    /// Never "no documents yet". "Yet" is a promise the measurements do not
    /// support: the detector fires zero times on real footage and has never
    /// been shown a positive it was built for.
    ///
    /// A static function rather than a computed property on the view, so a test
    /// can assert the words directly instead of reaching into a `View` through
    /// a mirror.
    static func heading(isEmptyAnswer: Bool) -> String {
        isEmptyAnswer
            ? "An empty memory is what this platform produces today"
            : "What this memory can and cannot record"
    }

    private func footnote(_ measuredAt: String, _ measurement: DocumentRecordingMeasurement) -> String {
        var text = "Measured \(measuredAt)"
        if let frames = measurement.corpusFrames { text += " on \(frames) frames" }
        if let captures = measurement.corpusCaptures { text += " across \(captures) recordings" }
        text += measurement.isCurrent
            ? "."
            : ". Not re-derived since; these describe the frames they were measured on."
        return text
    }
}

// MARK: - Envelope footnotes

/// The obligations that live on the envelope rather than on a row.
struct DocumentEnvelopeFootnotes: View {
    let response: DocumentLibraryResponse

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            // Never a path, and never an image. Stated so a reader knows the
            // absence of a thumbnail is a decision rather than a load failure.
            if !response.imageryServed, let note = response.imageryNote {
                Text(note)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // The window this read used, and the honest admission that the
            // window the writer used cannot be learned from here.
            if response.retention.writerWindowDays == nil,
               let reason = response.retention.writerWindowUnavailableReason {
                Text(reason)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let requested = response.retention.requestedDays {
                Text("This read was narrowed to the last \(Int(requested)) days. Narrowing a read cannot widen what was kept.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // Documents on disk versus documents in this answer. Two numbers
            // that mean different things, shown together so neither is mistaken
            // for the other.
            if response.documentsInMemory > 0 {
                Text("\(response.documentCount) shown of \(response.documentsInMemory) in this memory.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: - One document

/// One document, with the caveats the envelope hoisted off it.
///
/// `record_notes` was moved to the envelope because repeating 2,351 bytes of
/// identical prose on every record made a 200-document listing 488 KB. **The
/// caveats were hoisted, never deleted** — so a row renders them from the
/// envelope rather than from itself, and none of them may be dropped.
struct DocumentRow: View {
    let document: RememberedDocument
    /// The envelope's hoisted `record_notes`, keyed by the field each qualifies.
    var notes: [String: String] = [:]
    /// `snippet_max_chars`, off the envelope. **Read, never hard-coded** — it
    /// is 48 today and it is configuration-dependent.
    var snippetMaxChars: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // "Untitled document" describes the record. Never an invented name:
            // a name for a document this app has not read is a claim about its
            // contents.
            Text(document.displayTitle)
                .font(.body.weight(.medium))
                .fixedSize(horizontal: false, vertical: true)

            if document.title != nil, document.titleIsDerived {
                Text(titleProvenance)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let match = document.match { matchEvidence(match) }

            if let summary = document.summary {
                Text(summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let caption = document.summaryCaption {
                    Text(caption)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else if document.summaryAvailable, let withheld = notes["summary_withheld"] {
                // Not silence. A listing withholds the summary on purpose, and
                // the reason is worth one line rather than an unexplained gap.
                Text(withheld)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 10) {
                Text(observedLabel)
                if let duration = document.observedDuration {
                    Text(duration.label)
                }
                Text(document.text.displayName)
                if document.sightingCount > 1 {
                    // A later look at the same page joined this record rather
                    // than becoming a second one.
                    Text("Seen \(document.sightingCount) times")
                }
                if let pages = document.pagesObserved, pages > 1 {
                    if let readable = document.pagesReadable {
                        Text("\(readable) of \(pages) pages readable")
                    } else {
                        Text("\(pages) pages")
                    }
                }
            }
            .font(.caption2)
            .foregroundStyle(.tertiary)

            // `not_readable` is a real answer, and a row that showed only the
            // words "No readable text" would read as a failure.
            if let explanation = document.text.explanation {
                Text(explanation)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // The envelope's caveat where there is one, and this app's own
            // constant where there is not. Both say the same thing: time in
            // view is not attention.
            if document.observedDuration != nil {
                Text(notes["observed_seconds"] ?? ObservedDuration.attentionCaveat)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let timing = document.timing, timing.isReconstructed {
                // A duration derived from an assumed frame interval is a
                // reconstruction and must not look like a measurement.
                Text("That duration was reconstructed from an assumed frame rate rather than measured.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let frameReference = document.frameReference {
                DocumentProvenanceView(provenance: frameReference, timingNote: notes["timing"])
            }

            if let withheld = document.thumbnail.withheldReason {
                Text(withheld)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 4)
        .accessibilityElement(children: .combine)
    }

    private var titleProvenance: String {
        guard let max = document.titleMaxChars else {
            return "Taken from the page's own first line of text."
        }
        return "Taken from the page's own first line of text, cut at \(max) characters."
    }

    /// The evidence a match is owed.
    ///
    /// A match with no evidence is a number a client has to trust. The snippet
    /// is bounded and the bound is published beside it — read off the envelope
    /// here rather than assumed, because it is configuration-dependent and the
    /// contract prose disagreed with the code about its value for months.
    @ViewBuilder
    private func matchEvidence(_ match: DocumentMatchEvidence) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            if let snippet = match.snippet, !snippet.isEmpty {
                Text("“\(snippet)”")
                    .font(.caption)
                    .italic()
                    .fixedSize(horizontal: false, vertical: true)
                if let max = snippetMaxChars {
                    Text("Up to \(max) characters around the matched word — evidence, not an excerpt.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            if !match.matchedTerms.isEmpty {
                Text("Matched: " + match.matchedTerms.joined(separator: ", "))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            // Deliberately **not** rendered as a percentage or as a
            // confidence. It is a BM25 retrieval score, it has no upper bound
            // and no calibration, and showing it as "84% confident" would be
            // inventing a figure the Tower never computed.
        }
    }

    /// Observation time, or an explicit statement that it is unknown.
    ///
    /// Never falls back to `receivedAt`. When the Tower saw something and when
    /// the phone heard about it are different facts, and substituting one for
    /// the other is the conflation Core Principle 5 forbids — even though both
    /// are, on this wire, the Tower's own clock.
    private var observedLabel: String {
        guard let observed = document.time.displayableObservationTime else {
            return "Time unknown"
        }
        return observed.formatted(date: .abbreviated, time: .shortened)
    }
}

// MARK: - Provenance

/// The frame reference, and the fact about it that must not be left to be
/// noticed.
///
/// This block **is** joinable — a capture id, the frame numbers actually read,
/// and a time locate this reading inside a recording on disk, durably across
/// sessions. Scene Understanding refuses exactly this, and the difference is
/// the point: a document is a record, a scene is not. Saying it out loud here
/// is what makes the contrast visible to somebody who only ever opens one of
/// the two screens.
struct DocumentProvenanceView: View {
    let provenance: DocumentProvenance
    var timingNote: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            if provenance.joinable {
                Text(provenance.joinableNote ?? DocumentProvenance.joinabilityHeadline)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let captureID = provenance.captureID {
                HStack(spacing: 6) {
                    Text("Recording \(captureID)")
                    if !provenance.pageSourceSeqs.isEmpty {
                        Text("frames " + provenance.pageSourceSeqs.map(String.init).joined(separator: ", "))
                    }
                }
                .font(.caption2)
                .foregroundStyle(.tertiary)

                if !provenance.captureIDValidated {
                    // Always false. Nothing checks the capture still exists, so
                    // the pointer above may resolve to nothing.
                    Text("Nothing has checked that this recording still exists.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if let note = provenance.imageryRetentionNote {
                // Purging every document leaves the frames exactly where they
                // are. Worth one line, because a person deleting their reading
                // history would otherwise assume the pictures went too.
                Text(note)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let timingNote {
                Text(timingNote)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: - The recorder

/// The capture session, and the four verbs.
///
/// ## Why the buttons read the state back
///
/// **200 is not success on this cartridge.** A resume on a stopped session
/// answers 200 with `state: "stopped"` and no refusal field at all. So the
/// panel reports what the session *is*, from the payload that came back, and
/// says plainly when a verb was accepted and nothing moved.
///
/// ## Why Stop is not framed as losing anything
///
/// Because it does not. Stop **keeps** every document, and a dwell in progress
/// is flushed rather than dropped — a wearer still reading when a session stops
/// has read something. That is the opposite of Scene Understanding's stop, and
/// the wording here says so rather than borrowing the caution that screen needs.
struct DocumentSessionPanel: View {
    let session: DocumentSessionStatus
    var outcome: DocumentSessionOutcome?
    /// The camera's relation to this recorder, in one line, or `nil` when
    /// this screen started the camera itself.
    var cameraNote: String? = nil
    let send: (DocumentMemoryContract.SessionAction) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("Recording")

            VStack(alignment: .leading, spacing: 10) {
                if session.isUnavailable {
                    // A root with capture off is a Tower serving a library
                    // recorded elsewhere and recording nothing itself. A normal
                    // configuration, not a degraded one.
                    Text(session.reason ?? "This Tower runs no document capture session. Documents recorded elsewhere are still served.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    status
                    controls
                }

                if case .notHonoured(_, let explanation) = outcome {
                    // Not an error banner. The Tower did what it could and
                    // reported where the session is; the person needs to know
                    // the button did not do what its label implies.
                    Text(explanation)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if case .failed(let failure) = outcome {
                    FailureBanner(text: failure.message)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        }
    }

    @ViewBuilder
    private var status: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(headline)
                    .font(.subheadline.weight(.medium))
                Spacer(minLength: 8)
                if let pages = session.pagesDetected {
                    Text("\(pages) pages detected")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }

            HStack(spacing: 10) {
                if let recorded = session.documentsRecorded {
                    Text("\(recorded) recorded")
                }
                if let resighted = session.documentsResighted, resighted > 0 {
                    Text("\(resighted) seen again")
                }
                if let unreadable = session.dwellsUnreadable, unreadable > 0 {
                    Text("\(unreadable) not readable")
                }
                if let device = session.ocrDevice {
                    Text("OCR on \(device)")
                }
            }
            .font(.caption2)
            .foregroundStyle(.tertiary)
            .monospacedDigit()

            if let cameraNote {
                Text(cameraNote)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if session.idleStopPending {
                Text("No camera is streaming to the Tower. The recorder will stop itself unless one starts.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if session.loadOverdue {
                // Not a failure. The recogniser takes about five seconds to
                // construct and a first-run download is slower still.
                Text("The text recogniser is still loading. That is slow, not broken.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let failure = session.failureReason {
                Text(failure)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if session.inDwell {
                Text("A page is in view now. Stopping will keep it — the page in progress is written out, not discarded.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let flushed = session.flushedDocumentID {
                Text("The last stop wrote out the page that was still in view (\(flushed)).")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if session.retentionIncomplete {
                // Reported rather than logged: a deletion that quietly failed
                // looks exactly like one that was kept.
                Text("Some records past the retention window could not be deleted, so they are still on the Tower.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !session.followsStream {
                Text("Recording is started here, not by the camera. Connecting the glasses does not begin keeping what comes into view.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// The state, in words, from `state` — with the qualification that `state`
    /// is what somebody asked for.
    ///
    /// ## None of these may say "read"
    ///
    /// This headline said **"Recording what is read"** and two strings above it
    /// said "keeping what you read" and "the reading in progress". All three
    /// were wrong in the same way, and the proof is 700 lines away in this
    /// app's own decoder: `DocumentMemoryDecoder` **refuses a Tower** whose
    /// `claim` says a document was read, because the contract's claim is
    /// `a-page-was-in-view-and-was-ocred` and the older spelling was the
    /// stronger claim.
    ///
    /// So this build declined the claim on the wire and then made it in its own
    /// voice. The Tower is explicit that it cannot be made — `record_notes`
    /// carries *"This platform cannot establish that the wearer looked at it,
    /// noticed it, or read it"* — and this very screen's header already says
    /// *"Being in view is not the same as your having read it"*, which the old
    /// headline directly contradicted.
    ///
    /// A camera that sees a page establishes that a page was in view. It
    /// establishes nothing whatever about a person's attention.
    /// ## The two negative cases may not be stated absolutely
    ///
    /// These read *"Paused. Nothing is being recorded."* and *"Not recording."*
    /// — absolute claims about whether a wearer is being recorded, derived from
    /// `state`, which is **what somebody asked for**. The doc line above said so
    /// and none of the rendered strings carried the qualification.
    ///
    /// The contract is explicit that a Pause whose producer ignores `SIGTERM`
    /// answers 200 with `state: "paused"` while still recording, and Object
    /// Memory renders liveness from `following` for exactly that reason.
    ///
    /// **Document Memory's session block carries no `following`** — it is not
    /// the generic `cartridge_session.control` surface and declares no
    /// `state_means`, checked on the live wire. So the corroborating field is
    /// genuinely unavailable here, and the honest move is not to fake it but to
    /// stop overstating: "the recorder was asked to stop" is true of an intent
    /// field; "nothing is being recorded" is a claim about the world.
    ///
    /// The asymmetry is deliberate. `running` may be stated plainly, because
    /// the failure direction there is harmless — a session that says it is
    /// recording and is not costs a person nothing. The reverse tells someone
    /// they are not being recorded when they may be, and that is the one
    /// direction this screen must never be confidently wrong in.
    private var headline: String {
        switch session.state {
        case "running": return "Recording pages that come into view"
        case "starting": return "Starting the text recogniser"
        case "paused": return "Pause was requested. The Tower does not report whether the recorder has stopped."
        // NOT "Stop was requested" unconditionally. `stopped` is also where a
        // session lands when the engine could not load — the bench Tower right
        // now reports `state: "stopped"` with `failure_reason: "…No module
        // named 'easyocr'"`, and nobody requested anything. The old copy ("Not
        // recording.") was merely incomplete; attributing it to a human action
        // is a positive false claim, and `failure_reason` is rendered as a
        // caption directly underneath, so the screen would contradict itself
        // in two adjacent lines with the false one in bold.
        case "stopped":
            return session.failureReason == nil
                ? "Stop was requested. Everything already recorded is kept."
                : "Not recording. Everything already recorded is kept."
        case "failed": return "The recorder failed."
        default: return session.state
        }
    }

    private var controls: some View {
        HStack(spacing: 12) {
            ForEach(DocumentMemoryContract.SessionAction.allCases, id: \.self) { action in
                Button(Self.label(for: action)) { send(action) }
                    .font(.subheadline.weight(.medium))
                    .buttonStyle(.borderless)
            }
        }
        .accessibilityElement(children: .contain)
    }

    /// "Stop" and not "Stop and discard". Stop keeps everything.
    static func label(for action: DocumentMemoryContract.SessionAction) -> String {
        switch action {
        case .start: return "Start"
        case .pause: return "Pause"
        case .resume: return "Resume"
        case .stop: return "Stop"
        }
    }
}
