//
//  DocumentMemoryTests.swift
//  GlassesTests
//

import XCTest

@testable import Glasses

// MARK: - Fixtures

/// Payloads curled off a **running Tower** on 2026-08-27 with
/// `TOWER_DOCUMENT_ROOT` set and an empty library — which is the expected
/// state of this cartridge on this platform, not a degraded one.
///
/// `recent`, `search` and `session` are verbatim. The single record and the
/// single-document response are **assembled** to the shape
/// `tower/results/document_memory.py` produces (`_summary_view`, `_page_view`,
/// `_provenance`, `_timing`), because no capture on this platform has ever
/// contained a sheet of paper and the detector has never produced a positive.
/// That is the fact this cartridge exists under and it is stated here so a
/// reader does not take the fixture for an observation.
enum DocumentFixtures {
    /// `GET /documents`, verbatim.
    static let recent: [String: Any] = [
        "contract": "document_memory.library/2026-09-07",
        "claim": "a-page-was-in-view-and-was-ocred",
        "identity": "same-page-by-text-and-look-within-library",
        "absence_means": "not-recorded-by-this-cartridge",
        "time_basis": "tower-receipt",
        "spatial_ref": NSNull(),
        "answers": [
            "matched",
            "not_found",
            "no_observation",
        ],
        "retrieval_kinds": [
            "recent",
            "text",
            "observed_within",
        ],
        "semantic_retrieval": false,
        "semantic_retrieval_unavailable_reason": "this cartridge matches literal terms with BM25 and computes no embedding. Calling it semantic would be an overclaim",
        "semantic_retrieval_alternative": "route free text to retrieval_kind 'text'. It will be matched literally, so a description of a document will usually miss where a quotation from one will hit",
        "pagination": [
            "supported": false,
            "bound": "limit",
            "reason": "there is no cursor. Every listing route takes a `limit` and caps it, and the effective value is echoed in the `query` block. Truncation is detectable on `recent` by comparing document_count with documents_in_memory; on `text` and `observed_within` those two differ because of the query as well, so compare document_count with the limit instead",
        ],
        "privacy_tag_vocabulary": [
            "document-text",
            "first-person",
        ],
        "record_notes": [
            "observed_seconds": "how long the region was in view. This platform cannot establish that the wearer looked at it, noticed it, or read it",
            "summary_withheld": "the stored summary is the document's first forty words verbatim -- an excerpt, not a paraphrase -- and is served with the document, never in a listing. What a listing does carry is a clipped title, and a search result additionally carries a bounded snippet around the matched term as evidence; both are capped, and the caps are published beside them. The rule is that a listing must not become a bulk transfer of what a wearer read, not that it carries no verbatim characters at all",
            "timing": "tower-receipt time: when this Tower received the frames, never when the glasses captured them. There is no capture timestamp anywhere on this wire",
            "imagery_retention": "capture-side: the frames this record points at live in the capture store, whose lifetime this cartridge neither sets nor enforces. Purging every document here leaves that imagery exactly where it is",
            "joinable": "capture_id, page_source_seqs and observed_at together locate this reading in a recording. The link is durable across sessions, unlike anything Scene Understanding publishes",
        ],
        "recording_limitations": [
            [
                "limitation": "detection-rate",
                "detail": "on 9,199 frames of real first-person footage the page detector fired 6 times and every one was a false positive (a venetian blind and a backlit keyboard). After MIN_ROW_TRANSITIONS was re-derived against those same frames it fires 0 times. An empty library on this platform is the expected result, not a sign that nothing was read",
            ],
            [
                "limitation": "no-validated-positive",
                "detail": "no SAMPLED capture on this platform has contained a sheet of paper -- a visual review of 51 frames at quartile positions across 18 captures, on a corpus that has since grown. The detector has never been shown a positive it was built for, so the premise is untested rather than disproved",
            ],
            [
                "limitation": "resolution",
                "detail": "at the 360x640 the glasses deliver, EasyOCR returned zero dictionary words across 919 sampled real frames -- every tenth frame of a corpus that is dense with screen text and also full of walls and carpet -- at median confidence 0.056. Word recall on rendered pages at this geometry is 0.629-0.952 with the page at its own aspect, which is the fairest estimate of the delivered case",
            ],
            [
                "limitation": "resolution-remedy-is-not-a-fix",
                "detail": "a high-resolution still is the measured remedy for RECOGNITION, and recognition is not what is failing. Detection is the binding constraint and a still does not touch it: the glyph gate is derived for 360x640 only and must be re-derived at any other geometry, where the usable window between real negatives and readable pages may close entirely. Nobody has run that derivation",
            ],
        ],
        "recording_measurement": [
            "measured_at": "2026-09-06",
            "corpus_frames": 45594,
            "corpus_captures": 97,
            "is_current": false,
            "note": "the corpus on this host has grown since. These figures describe the frames they were measured on and have not been re-derived",
        ],
        "imagery_treatment": "none-retained",
        "imagery_ios_state": "rawEphemeral",
        "imagery_served": false,
        "imagery_note": "this platform performs no redaction. No route serves an image. Where retains_raw_imagery is true, an unredacted page image exists on this Tower's disk under capture-side retention and is not reachable over any wire",
        "retention": [
            "requested_days": NSNull(),
            "writer_window_days": NSNull(),
            "writer_window_unavailable_reason": "DocumentStore persists no retention manifest, so a reader cannot learn the window its writer used. A request here narrows this read and can never widen what was kept",
            "policy": "a reader may narrow this read; it cannot widen it",
        ],
        "query": [
            "kind": "recent",
            "limit": 10,
        ],
        "answer": "no_observation",
        "no_observation_note": "this Tower has recorded no documents at all. That is a statement about what its camera captured, never about what exists -- and on this platform it is the expected result: see recording_limitations",
        "documents_in_memory": 0,
        "document_count": 0,
        "documents": [String](),
    ]

    /// `GET /documents/search?text=invoice`, verbatim.
    static let search: [String: Any] = [
        "contract": "document_memory.library/2026-09-07",
        "claim": "a-page-was-in-view-and-was-ocred",
        "identity": "same-page-by-text-and-look-within-library",
        "absence_means": "not-recorded-by-this-cartridge",
        "time_basis": "tower-receipt",
        "spatial_ref": NSNull(),
        "answers": [
            "matched",
            "not_found",
            "no_observation",
        ],
        "retrieval_kinds": [
            "recent",
            "text",
            "observed_within",
        ],
        "semantic_retrieval": false,
        "semantic_retrieval_unavailable_reason": "this cartridge matches literal terms with BM25 and computes no embedding. Calling it semantic would be an overclaim",
        "semantic_retrieval_alternative": "route free text to retrieval_kind 'text'. It will be matched literally, so a description of a document will usually miss where a quotation from one will hit",
        "pagination": [
            "supported": false,
            "bound": "limit",
            "reason": "there is no cursor. Every listing route takes a `limit` and caps it, and the effective value is echoed in the `query` block. Truncation is detectable on `recent` by comparing document_count with documents_in_memory; on `text` and `observed_within` those two differ because of the query as well, so compare document_count with the limit instead",
        ],
        "privacy_tag_vocabulary": [
            "document-text",
            "first-person",
        ],
        "record_notes": [
            "observed_seconds": "how long the region was in view. This platform cannot establish that the wearer looked at it, noticed it, or read it",
            "summary_withheld": "the stored summary is the document's first forty words verbatim -- an excerpt, not a paraphrase -- and is served with the document, never in a listing. What a listing does carry is a clipped title, and a search result additionally carries a bounded snippet around the matched term as evidence; both are capped, and the caps are published beside them. The rule is that a listing must not become a bulk transfer of what a wearer read, not that it carries no verbatim characters at all",
            "timing": "tower-receipt time: when this Tower received the frames, never when the glasses captured them. There is no capture timestamp anywhere on this wire",
            "imagery_retention": "capture-side: the frames this record points at live in the capture store, whose lifetime this cartridge neither sets nor enforces. Purging every document here leaves that imagery exactly where it is",
            "joinable": "capture_id, page_source_seqs and observed_at together locate this reading in a recording. The link is durable across sessions, unlike anything Scene Understanding publishes",
        ],
        "recording_limitations": [
            [
                "limitation": "detection-rate",
                "detail": "on 9,199 frames of real first-person footage the page detector fired 6 times and every one was a false positive (a venetian blind and a backlit keyboard). After MIN_ROW_TRANSITIONS was re-derived against those same frames it fires 0 times. An empty library on this platform is the expected result, not a sign that nothing was read",
            ],
            [
                "limitation": "no-validated-positive",
                "detail": "no SAMPLED capture on this platform has contained a sheet of paper -- a visual review of 51 frames at quartile positions across 18 captures, on a corpus that has since grown. The detector has never been shown a positive it was built for, so the premise is untested rather than disproved",
            ],
            [
                "limitation": "resolution",
                "detail": "at the 360x640 the glasses deliver, EasyOCR returned zero dictionary words across 919 sampled real frames -- every tenth frame of a corpus that is dense with screen text and also full of walls and carpet -- at median confidence 0.056. Word recall on rendered pages at this geometry is 0.629-0.952 with the page at its own aspect, which is the fairest estimate of the delivered case",
            ],
            [
                "limitation": "resolution-remedy-is-not-a-fix",
                "detail": "a high-resolution still is the measured remedy for RECOGNITION, and recognition is not what is failing. Detection is the binding constraint and a still does not touch it: the glyph gate is derived for 360x640 only and must be re-derived at any other geometry, where the usable window between real negatives and readable pages may close entirely. Nobody has run that derivation",
            ],
        ],
        "recording_measurement": [
            "measured_at": "2026-09-06",
            "corpus_frames": 45594,
            "corpus_captures": 97,
            "is_current": false,
            "note": "the corpus on this host has grown since. These figures describe the frames they were measured on and have not been re-derived",
        ],
        "imagery_treatment": "none-retained",
        "imagery_ios_state": "rawEphemeral",
        "imagery_served": false,
        "imagery_note": "this platform performs no redaction. No route serves an image. Where retains_raw_imagery is true, an unredacted page image exists on this Tower's disk under capture-side retention and is not reachable over any wire",
        "retention": [
            "requested_days": NSNull(),
            "writer_window_days": NSNull(),
            "writer_window_unavailable_reason": "DocumentStore persists no retention manifest, so a reader cannot learn the window its writer used. A request here narrows this read and can never widen what was kept",
            "policy": "a reader may narrow this read; it cannot widen it",
        ],
        "query": [
            "kind": "text",
            "text": "invoice",
            "limit": 5,
        ],
        "answer": "no_observation",
        "no_observation_note": "this Tower has recorded no documents at all. That is a statement about what its camera captured, never about what exists -- and on this platform it is the expected result: see recording_limitations",
        "documents_in_memory": 0,
        "searched_documents": 0,
        "min_score": 0.1,
        "sufficient_evidence": false,
        "reason": "no documents have been observed",
        "match_kind": "lexical",
        "document_count": 0,
        "snippet_max_chars": 48,
        "documents": [String](),
    ]

    /// `GET /documents-session`, verbatim.
    static let session: [String: Any] = [
        "contract": "document_memory.library/2026-09-07",
        "claim": "a-page-was-in-view-and-was-ocred",
        "identity": "same-page-by-text-and-look-within-library",
        "absence_means": "not-recorded-by-this-cartridge",
        "time_basis": "tower-receipt",
        "recording_limitations": [
            [
                "limitation": "detection-rate",
                "detail": "on 9,199 frames of real first-person footage the page detector fired 6 times and every one was a false positive (a venetian blind and a backlit keyboard). After MIN_ROW_TRANSITIONS was re-derived against those same frames it fires 0 times. An empty library on this platform is the expected result, not a sign that nothing was read",
            ],
            [
                "limitation": "no-validated-positive",
                "detail": "no SAMPLED capture on this platform has contained a sheet of paper -- a visual review of 51 frames at quartile positions across 18 captures, on a corpus that has since grown. The detector has never been shown a positive it was built for, so the premise is untested rather than disproved",
            ],
            [
                "limitation": "resolution",
                "detail": "at the 360x640 the glasses deliver, EasyOCR returned zero dictionary words across 919 sampled real frames -- every tenth frame of a corpus that is dense with screen text and also full of walls and carpet -- at median confidence 0.056. Word recall on rendered pages at this geometry is 0.629-0.952 with the page at its own aspect, which is the fairest estimate of the delivered case",
            ],
            [
                "limitation": "resolution-remedy-is-not-a-fix",
                "detail": "a high-resolution still is the measured remedy for RECOGNITION, and recognition is not what is failing. Detection is the binding constraint and a still does not touch it: the glyph gate is derived for 360x640 only and must be re-derived at any other geometry, where the usable window between real negatives and readable pages may close entirely. Nobody has run that derivation",
            ],
        ],
        "recording_measurement": [
            "measured_at": "2026-09-06",
            "corpus_frames": 45594,
            "corpus_captures": 97,
            "is_current": false,
            "note": "the corpus on this host has grown since. These figures describe the frames they were measured on and have not been re-derived",
        ],
        "imagery_treatment": "none-retained",
        "imagery_ios_state": "rawEphemeral",
        "imagery_served": false,
        "imagery_note": "this platform performs no redaction. No route serves an image. Where retains_raw_imagery is true, an unredacted page image exists on this Tower's disk under capture-side retention and is not reachable over any wire",
        "session": [
            "state": "stopped",
            "states": [
                "stopped",
                "starting",
                "running",
                "paused",
                "failed",
                "unavailable",
            ],
            "session_id": 0,
            "failure_reason": NSNull(),
            "started_at": NSNull(),
            "ready_at": NSNull(),
            "loading_seconds": NSNull(),
            "load_overdue": false,
            "load_overdue_after_seconds": 120.0,
            "engine": NSNull(),
            "frames_offered": 13,
            "frames_observed": 0,
            "frames_skipped": 0,
            "frames_dropped_not_running": 13,
            "recogniser": NSNull(),
            "capture_id": NSNull(),
            "capture_id_validated": false,
            "in_dwell": false,
            "dwells_started": 0,
            "pages_detected": 0,
            "documents_recorded": 0,
            "last_document_id": NSNull(),
            "last_document_at": NSNull(),
            "flushed_document_id": NSNull(),
            "keeps_page_images": false,
            "follows_stream": false,
            "retention_days": 30.0,
            "documents_pruned": 0,
            "retention_incomplete": false,
            "library_count": NSNull(),
            "library_soft_limit": 10000,
            "library_over_soft_limit": false,
            "library_soft_limit_note": "a soft limit is reported, never enforced. This session evicts by AGE only: deleting a wearer's memories because a count grew is a policy decision, not a cleanup",
            "reason": NSNull(),
        ],
    ]

    /// The `document_memory.status` payload as it arrives on the socket, from a
    /// live `result_subscribe`. Note what it does **not** contain: the
    /// documents. `contract_note` carries that fact inside the payload so a
    /// client reading only this channel still learns they are on HTTP.
    static let socketStatus: [String: Any] = [
        "contract_note": "session progress only. The documents themselves are on HTTP: /documents, /documents/{document_id}, /documents/search",
        "claim": "a-page-was-in-view-and-was-ocred",
        "identity": "same-page-by-text-and-look-within-library",
        "absence_means": "not-recorded-by-this-cartridge",
        "time_basis": "tower-receipt",
        "library": [
            "available": true,
            "document_count_unfiltered": 0,
            "retention_applied": false,
            "unavailable_reason": NSNull(),
            "newest_observed_at": NSNull(),
            "bytes": [
                "journal": 0,
                "images": 0,
                "total": 0,
            ],
            "location_disclosed": false,
        ],
        "session": [
            "state": "stopped",
            "states": [
                "stopped",
                "starting",
                "running",
                "paused",
                "failed",
                "unavailable",
            ],
            "session_id": 0,
            "failure_reason": NSNull(),
            "started_at": NSNull(),
            "ready_at": NSNull(),
            "loading_seconds": NSNull(),
            "load_overdue": false,
            "load_overdue_after_seconds": 120.0,
            "engine": NSNull(),
            "frames_offered": 13,
            "frames_observed": 0,
            "frames_skipped": 0,
            "frames_dropped_not_running": 13,
            "recogniser": NSNull(),
            "capture_id": NSNull(),
            "capture_id_validated": false,
            "in_dwell": false,
            "dwells_started": 0,
            "pages_detected": 0,
            "documents_recorded": 0,
            "last_document_id": NSNull(),
            "last_document_at": NSNull(),
            "flushed_document_id": NSNull(),
            "keeps_page_images": false,
            "follows_stream": false,
            "retention_days": 30.0,
            "documents_pruned": 0,
            "retention_incomplete": false,
            "library_count": NSNull(),
            "library_soft_limit": 10000,
            "library_over_soft_limit": false,
            "library_soft_limit_note": "a soft limit is reported, never enforced. This session evicts by AGE only: deleting a wearer's memories because a count grew is a policy decision, not a cleanup",
            "reason": NSNull(),
        ],
        "recording_limitations": [
            [
                "limitation": "detection-rate",
                "detail": "on 9,199 frames of real first-person footage the page detector fired 6 times and every one was a false positive (a venetian blind and a backlit keyboard). After MIN_ROW_TRANSITIONS was re-derived against those same frames it fires 0 times. An empty library on this platform is the expected result, not a sign that nothing was read",
            ],
            [
                "limitation": "no-validated-positive",
                "detail": "no SAMPLED capture on this platform has contained a sheet of paper -- a visual review of 51 frames at quartile positions across 18 captures, on a corpus that has since grown. The detector has never been shown a positive it was built for, so the premise is untested rather than disproved",
            ],
            [
                "limitation": "resolution",
                "detail": "at the 360x640 the glasses deliver, EasyOCR returned zero dictionary words across 919 sampled real frames -- every tenth frame of a corpus that is dense with screen text and also full of walls and carpet -- at median confidence 0.056. Word recall on rendered pages at this geometry is 0.629-0.952 with the page at its own aspect, which is the fairest estimate of the delivered case",
            ],
            [
                "limitation": "resolution-remedy-is-not-a-fix",
                "detail": "a high-resolution still is the measured remedy for RECOGNITION, and recognition is not what is failing. Detection is the binding constraint and a still does not touch it: the glyph gate is derived for 360x640 only and must be re-derived at any other geometry, where the usable window between real negatives and readable pages may close entirely. Nobody has run that derivation",
            ],
        ],
        "recording_measurement": [
            "measured_at": "2026-09-06",
            "corpus_frames": 45594,
            "corpus_captures": 97,
            "is_current": false,
            "note": "the corpus on this host has grown since. These figures describe the frames they were measured on and have not been re-derived",
        ],
        "imagery_treatment": "none-retained",
        "imagery_ios_state": "rawEphemeral",
        "imagery_served": false,
        "imagery_note": "this platform performs no redaction. No route serves an image. Where retains_raw_imagery is true, an unredacted page image exists on this Tower's disk under capture-side retention and is not reachable over any wire",
    ]

    // MARK: One record, and one document

    /// A record in the shape `_summary_view` produces, with a **null title**
    /// and `not_readable` text — the two cases most likely to be rendered
    /// wrongly.
    static let record: [String: Any] = [
        "document_id": "doc-1",
        "claim": "a-page-was-in-view-and-was-ocred",
        "identity": "same-page-by-text-and-look-within-library",
        "title": NSNull(),
        "title_is_derived": true,
        "title_max_chars": 60,
        "summary_available": true,
        "confidence": "low",
        "confidence_basis": "the weakest page read in this document",
        "observed_at": 1787830000.0,
        "recorded_at": 1787830050.0,
        "observed_seconds": 12.5,
        "pages_observed": 1,
        "text_availability": ["state": "not_readable", "character_count": 0],
        "end_reason": "flushed",
        "timing": [
            "time_basis": "tower-receipt",
            "source": "assumed-interval",
            "assumed_frame_interval_s": 0.0835,
        ],
        "provenance": [
            "kind": "frame-reference",
            "spatial_ref": NSNull(),
            "capture_id": "cap-77",
            "capture_id_validated": false,
            "page_source_seqs": [412, 419],
            "pages_without_source_seq": 0,
            "frames_considered": 1500,
            "frames_ocred": 2,
            "world_id": NSNull(),
            "world_session_id": NSNull(),
            "imagery_retention": "capture-side",
            "joinable": true,
        ],
        "retains_raw_imagery": false,
        "redaction": "none",
        "imagery_treatment": "none-retained",
        "imagery_ios_state": "rawEphemeral",
        "imagery_served": false,
        "privacy_tags": ["document-text", "first-person"],
        "schema_version": 1,
    ]

    /// The same envelope, answering `matched` with that record in it.
    static var matched: [String: Any] {
        var payload = recent
        payload["answer"] = "matched"
        payload["no_observation_note"] = NSNull()
        payload["documents_in_memory"] = 1
        payload["document_count"] = 1
        payload["documents"] = [record]
        return payload
    }

    /// `not_found`: the memory was searched and nothing matched. Distinct from
    /// `no_observation`, and the distinction is the whole point of the closed
    /// vocabulary.
    static var notFound: [String: Any] {
        var payload = search
        payload["answer"] = "not_found"
        payload["no_observation_note"] = NSNull()
        payload["documents_in_memory"] = 7
        payload["searched_documents"] = 7
        payload["sufficient_evidence"] = false
        return payload
    }

    /// A search result: the record plus the four search-only fields.
    static var searchMatch: [String: Any] {
        var payload = search
        payload["answer"] = "matched"
        payload["no_observation_note"] = NSNull()
        payload["documents_in_memory"] = 3
        payload["searched_documents"] = 3
        payload["sufficient_evidence"] = true
        payload["document_count"] = 1
        var one = record
        one["score"] = 4.1732
        one["matched_terms"] = ["parking"]
        // 48 characters — the cap the envelope publishes, applied by the Tower.
        one["snippet"] = "…restrictions on parking are in force from Mon"
        payload["documents"] = [one]
        return payload
    }

    /// `GET /documents/{id}` — the only route that carries text, and the only
    /// place the summary and the pages appear.
    static var oneDocument: [String: Any] {
        var payload = recent
        payload["answer"] = "matched"
        payload["no_observation_note"] = NSNull()
        payload["query"] = ["kind": "document", "document_id": "doc-1"]
        var document = record
        document["summary"] = "Notice of parking restrictions in force from Monday"
        document["summary_is_verbatim_excerpt"] = true
        document["summary_is_model_output"] = true
        document["word_count"] = 8
        payload["document"] = document
        payload["pages"] = [[
            "page_index": 0,
            // Empty text with zero regions is the readable-nothing case, which
            // is a real answer and not a missing page.
            "text": "",
            "text_source": "ocr",
            "region_count": 0,
            "mean_region_confidence": NSNull(),
            "min_region_confidence": NSNull(),
            "confidence": "unknown",
            "sharpness": 41.2,
            "squareness": 0.83,
            "source_seq": 412,
            "observed_at": 1787830000.0,
            "observation_count": 2,
            "image_kept": false,
            "image_served": false,
        ]]
        payload["coverage"] = [
            "pages_observed": 1,
            "pages_total": NSNull(),
            "pages_total_note": "the camera cannot know how many pages a document has",
            "words_captured": 0,
            "low_confidence_pages": [0],
        ]
        return payload
    }

    /// The `session` block out of `GET /documents-session`.
    static var sessionBlock: [String: Any] { session["session"] as! [String: Any] }

    static func session(state: String) -> [String: Any] {
        var block = sessionBlock
        block["state"] = state
        return block
    }
}

// MARK: - The closed vocabulary

/// Three answers, three sentences, and the third is the one that matters here.
@MainActor
final class DocumentAnswerTests: XCTestCase {

    /// An empty library answers `no_observation`, **not** `not_found`.
    ///
    /// Collapsing the third into the second lets a gap in what the glasses
    /// happened to see read as a statement about the world. On this platform
    /// that gap is the normal case.
    func testAnEmptyLibraryAnswersNeverObservedRatherThanNothingMatched() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.recent))
        XCTAssertEqual(response.answer, .noObservation)
        XCTAssertEqual(response.answer.evidence, .noObservation)
        XCTAssertEqual(response.documentsInMemory, 0)
        let note = try XCTUnwrap(response.noObservationNote)
        // The Tower's own sentence, and it is stronger than anything this app
        // could write: it says what the *camera captured*, never what exists.
        XCTAssertTrue(note.contains("never about what exists"), "got: \(note)")
    }

    func testTheThreeAnswersAreThreeDifferentThings() throws {
        let never = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.recent))
        let nothing = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.notFound))
        let found = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.matched))
        XCTAssertEqual(never.answer, .noObservation)
        XCTAssertEqual(nothing.answer, .notFound)
        XCTAssertEqual(found.answer, .matched)
        XCTAssertNotEqual(never.answer.evidence, nothing.answer.evidence)
        XCTAssertNotEqual(
            DocumentQueryEvidence.notFound.explanation,
            DocumentQueryEvidence.noObservation.explanation
        )
        XCTAssertTrue(
            DocumentQueryEvidence.noObservation.explanation.contains("not the same"),
            "a never-observed answer failed to disclaim absence"
        )
    }

    /// An `answer` this build has not heard of resolves to the **weakest** of
    /// the three, which claims nothing about what was searched.
    func testAnUnknownAnswerIsNotTreatedAsAMatch() throws {
        var odd = DocumentFixtures.recent
        odd["answer"] = "partially_matched"
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: odd))
        XCTAssertEqual(response.answer, .unrecognised("partially_matched"))
        XCTAssertEqual(response.answer.evidence, .noObservation)
    }

    /// A `matched` answer carrying no documents is a broken payload, and the
    /// safe direction from one is a failure — never `.notFound`, whose sentence
    /// is a definite negative about the wearer's own memory.
    func testAMatchWithNoDocumentsIsUndecodableRatherThanNotFound() {
        XCTAssertThrowsError(
            try DocumentQueryResult(
                query: .text("parking"), origin: .appText, documents: [], evidence: .matched(confidence: 0.9)
            )
        ) { error in
            XCTAssertEqual((error as? CartridgeFailure)?.kind, .undecodableResponse)
        }
    }

    /// A BM25 score is a retrieval score, not a confidence. Rendering one as a
    /// percentage would invent a figure the Tower never computed.
    func testAnAnswerCarriesNoInventedConfidence() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.searchMatch))
        XCTAssertEqual(response.answer.evidence, .matched(confidence: nil))
        XCTAssertEqual(response.answer.evidence.explanation, "")
        XCTAssertEqual(response.documents.first?.match?.score, 4.1732)
    }
}

// MARK: - The limitation that must not be hidden

/// §8.3. It appeared **zero times** in this app before this work, while every
/// response carried it.
@MainActor
final class DocumentLimitationTests: XCTestCase {

    func testEveryResponseCarriesTheRecordingLimitations() throws {
        for payload in [
            DocumentFixtures.recent, DocumentFixtures.search, DocumentFixtures.matched,
            DocumentFixtures.notFound, DocumentFixtures.oneDocument,
        ] {
            let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: payload))
            let slugs = Set(response.recordingLimitations.map(\.slug))
            XCTAssertEqual(slugs.count, 4, "got \(slugs)")
            for expected in [
                "detection-rate", "no-validated-positive", "resolution",
                "resolution-remedy-is-not-a-fix",
            ] {
                XCTAssertTrue(slugs.contains(expected), "missing \(expected)")
            }
        }
    }

    /// The premise is **untested, not disproved**, and the wire says both
    /// halves: six false positives in 9,199 frames, zero after the gate was
    /// re-derived, and no capture that has ever contained a sheet of paper.
    func testTheLimitationsSayTheDetectorHasNeverSeenAPositive() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.recent))
        let byslug = Dictionary(
            uniqueKeysWithValues: response.recordingLimitations.map { ($0.slug, $0.detail) }
        )
        XCTAssertTrue(byslug["detection-rate"]?.contains("9,199") == true)
        XCTAssertTrue(byslug["detection-rate"]?.contains("false positive") == true)
        XCTAssertTrue(byslug["no-validated-positive"]?.contains("untested") == true)
        XCTAssertTrue(byslug["resolution"]?.contains("zero dictionary words") == true)
    }

    /// `is_current: false`, and never defaulted to `true`.
    func testTheMeasurementIsNotPresentedAsCurrent() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.recent))
        XCTAssertFalse(response.recordingMeasurement.isCurrent)
        XCTAssertEqual(response.recordingMeasurement.corpusFrames, 45594)
        var bare = DocumentFixtures.recent
        bare["recording_measurement"] = NSNull()
        let plain = try XCTUnwrap(DocumentMemoryDecoder.library(from: bare))
        XCTAssertFalse(plain.recordingMeasurement.isCurrent)
    }

    /// The heading over an empty answer must not contain the word this
    /// cartridge cannot support.
    func testAnEmptyLibraryIsNotDescribedAsNotYet() {
        // The view's own heading, not a copy of it, so a copy edit that
        // reintroduces the promise fails here.
        let heading = DocumentLimitationsView.heading(isEmptyAnswer: true).lowercased()
        // "Yet" is what turns a measured dead end into a promise — it invites a
        // person to wait for something that is not coming.
        XCTAssertFalse(heading.contains("yet"), "got: \(heading)")
        XCTAssertFalse(heading.contains("no documents"), "got: \(heading)")
        XCTAssertTrue(heading.contains("expected") || heading.contains("today"), "got: \(heading)")
        XCTAssertNotEqual(heading, DocumentLimitationsView.heading(isEmptyAnswer: false).lowercased())
    }

    /// And the empty answer itself is rendered as "never observed", with the
    /// explicit disclaimer the closed vocabulary exists for.
    func testAnEmptyLibraryIsRenderedAsNeverObserved() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.recent))
        XCTAssertEqual(response.answer, .noObservation)
        XCTAssertTrue(response.documents.isEmpty)
        XCTAssertFalse(response.recordingLimitations.isEmpty)
    }
}

// MARK: - Fields read rather than assumed

@MainActor
final class DocumentFieldTests: XCTestCase {

    /// **48, off the envelope.** The contract prose said 160 for months while
    /// the code said 48, and the instruction that resolved it was to read the
    /// field rather than the prose.
    func testTheSnippetCapIsReadFromThePayload() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.search))
        XCTAssertEqual(response.snippetMaxChars, 48)
        // And a Tower that publishes a different cap is believed.
        var other = DocumentFixtures.search
        other["snippet_max_chars"] = 96
        XCTAssertEqual(
            try XCTUnwrap(DocumentMemoryDecoder.library(from: other)).snippetMaxChars, 96
        )
    }

    /// A null title renders as a description of the **record**, never as an
    /// invented name for the thing.
    func testANullTitleIsDescribedRatherThanNamed() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.matched))
        let document = try XCTUnwrap(response.documents.first)
        XCTAssertNil(document.title)
        XCTAssertEqual(document.displayTitle, "Untitled document")
        XCTAssertEqual(document.titleMaxChars, 60)
        XCTAssertTrue(document.titleIsDerived)
    }

    /// `not_readable` is **a real answer**: we looked and found no readable
    /// text. Silence (`unknown`) is not a verdict, and the two must not share a
    /// sentence.
    func testNotReadableIsAnAnswerAndUnknownIsNot() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.matched))
        let document = try XCTUnwrap(response.documents.first)
        XCTAssertEqual(document.text, .notReadable)
        XCTAssertNotEqual(
            DocumentTextAvailability.unknown.displayName,
            DocumentTextAvailability.notReadable.displayName
        )
        XCTAssertEqual(
            document.text.explanation?.contains("not an error"), true,
            "an answer was rendered as a failure"
        )
        XCTAssertFalse(DocumentTextAvailability.unknown.hasText)
        XCTAssertFalse(DocumentTextAvailability.notReadable.hasText)
        XCTAssertTrue(DocumentTextAvailability.extracted(characterCount: nil).hasText)

        // A state this build has not heard of is not a verdict about the page.
        XCTAssertEqual(
            DocumentTextAvailability(state: "partially_readable", characterCount: 3), .unknown
        )
    }

    /// `observed_seconds` is **not** a claim the wearer looked at it, noticed
    /// it, or read it — and the hoisted caveat says so as data.
    func testTimeInViewIsNotAClaimAboutAttention() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.matched))
        let document = try XCTUnwrap(response.documents.first)
        XCTAssertEqual(document.observedDuration?.label, "In view 13s")
        let note = try XCTUnwrap(response.recordNotes["observed_seconds"])
        XCTAssertTrue(note.contains("cannot establish"), "got: \(note)")
        XCTAssertTrue(ObservedDuration.attentionCaveat.lowercased().contains("cannot tell"))
    }

    /// `record_notes` is hoisted to the envelope, keyed by field name, and
    /// **none of the five may be dropped.** The saving was the repetition, not
    /// the caveats.
    func testTheHoistedCaveatsAreAllPresent() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.recent))
        XCTAssertEqual(
            Set(response.recordNotes.keys),
            ["observed_seconds", "summary_withheld", "timing", "imagery_retention", "joinable"]
        )
        XCTAssertTrue(response.recordNotes.values.allSatisfy { !$0.isEmpty })
    }

    /// The summary is the first forty words **verbatim**, so it is captioned as
    /// the document's own words rather than as model output. Captioning
    /// verbatim text as a paraphrase invites a reader to discount words the
    /// document actually contained.
    func testTheSummaryIsCaptionedAsAnExcerptRatherThanAsAParaphrase() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.oneDocument))
        let document = try XCTUnwrap(response.document)
        XCTAssertTrue(document.summaryIsVerbatimExcerpt)
        let caption = try XCTUnwrap(document.summaryCaption)
        XCTAssertTrue(caption.contains("copied exactly"), "got: \(caption)")
        XCTAssertFalse(caption.lowercased().contains("estimated by a model"))

        // And it is served only here. A listing says it exists and withholds it.
        let listing = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.matched))
        XCTAssertTrue(listing.documents[0].summaryAvailable)
        XCTAssertNil(listing.documents[0].summary)
        XCTAssertNotNil(listing.recordNotes["summary_withheld"])
    }

    /// `writer_window_days` is **always null**, honestly: the store persists no
    /// retention manifest, so a reader cannot learn the window its writer used.
    /// A request narrows a read and can never widen it.
    func testTheWritersRetentionWindowCannotBeLearnedFromARead() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.recent))
        XCTAssertNil(response.retention.writerWindowDays)
        let reason = try XCTUnwrap(response.retention.writerWindowUnavailableReason)
        XCTAssertTrue(reason.contains("never widen"), "got: \(reason)")
        XCTAssertEqual(response.retention.policy?.contains("cannot widen"), true)
    }

    /// `imagery_served` is a **boolean, not a path** — the path told a reader
    /// where in the store to find an unredacted photograph of what the wearer
    /// was reading, which is disclosure with no consumer.
    func testNoImageIsServedAndNoPathIsPublished() throws {
        for payload in [DocumentFixtures.recent, DocumentFixtures.oneDocument] {
            let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: payload))
            XCTAssertFalse(response.imageryServed)
            XCTAssertEqual(response.imageryIOSState, "rawEphemeral")
            // Never `redacted`: this platform performs no redaction.
            XCTAssertNotEqual(response.imageryIOSState, "redacted")
        }
        let one = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.oneDocument))
        XCTAssertFalse(one.pages[0].imageKept)
        XCTAssertFalse(one.pages[0].imageServed)
        XCTAssertFalse(try XCTUnwrap(one.document).isThumbnailDisplayable)
    }

    /// A page with empty text and zero regions is the readable-nothing case.
    /// It is not a missing page, and `pages_total` is null because the camera
    /// cannot know how many pages a document has.
    func testAPageThatReadNothingIsStillAPage() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.oneDocument))
        XCTAssertEqual(response.pages.count, 1)
        XCTAssertTrue(response.pages[0].text.isEmpty)
        XCTAssertEqual(response.pages[0].regionCount, 0)
        XCTAssertEqual(response.pages[0].observationCount, 2)
        XCTAssertNil(response.coverage?.pagesTotal)
        XCTAssertNotNil(response.coverage?.pagesTotalNote)
    }

    /// A duration from an assumed frame interval is a **reconstruction** and
    /// must not be rendered identically to a measured one.
    func testAReconstructedDurationSaysSo() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.matched))
        XCTAssertEqual(response.documents[0].timing?.isReconstructed, true)
        var measured = DocumentFixtures.matched
        var one = DocumentFixtures.record
        one["timing"] = ["time_basis": "tower-receipt", "source": "capture-journal"]
        measured["documents"] = [one]
        let honest = try XCTUnwrap(DocumentMemoryDecoder.library(from: measured))
        XCTAssertEqual(honest.documents[0].timing?.isReconstructed, false)
    }

    /// A search result carries evidence, because a match with no evidence is a
    /// number a client has to trust.
    func testAMatchCarriesTheTextItMatchedOn() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.searchMatch))
        let match = try XCTUnwrap(response.documents.first?.match)
        XCTAssertEqual(match.matchedTerms, ["parking"])
        let snippet = try XCTUnwrap(match.snippet)
        XCTAssertFalse(snippet.isEmpty)
        XCTAssertLessThanOrEqual(snippet.count, try XCTUnwrap(response.snippetMaxChars))
        XCTAssertEqual(response.matchKind, "lexical")
    }

    /// There is no cursor, so truncation is detectable only by arithmetic —
    /// and the rule differs by query kind, which is why the Tower publishes it.
    func testTruncationIsDetectedTheWayTheTowerSaysToDetectIt() throws {
        var listing = DocumentFixtures.matched
        listing["documents_in_memory"] = 40
        listing["document_count"] = 10
        listing["query"] = ["kind": "recent", "limit": 10]
        XCTAssertTrue(try XCTUnwrap(DocumentMemoryDecoder.library(from: listing)).isPossiblyTruncated)

        var full = DocumentFixtures.matched
        full["documents_in_memory"] = 1
        full["document_count"] = 1
        full["query"] = ["kind": "recent", "limit": 10]
        XCTAssertFalse(try XCTUnwrap(DocumentMemoryDecoder.library(from: full)).isPossiblyTruncated)
    }
}

// MARK: - Provenance, and the deliberate contrast

/// A document is a record; a scene is not.
@MainActor
final class DocumentProvenanceTests: XCTestCase {

    /// `joinable: true`, said out loud. A capture id, the frame numbers
    /// actually read, and a time locate this reading inside a recording on
    /// disk, durably across sessions — which is precisely what Scene
    /// Understanding refuses to hand anyone.
    func testTheFrameReferenceIsJoinableAndSaysSo() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.matched))
        let provenance = try XCTUnwrap(response.documents.first?.frameReference)
        XCTAssertTrue(provenance.joinable)
        XCTAssertEqual(provenance.kind, "frame-reference")
        XCTAssertEqual(provenance.captureID, "cap-77")
        XCTAssertEqual(provenance.pageSourceSeqs, [412, 419])
        // The envelope's hoisted note is attached back onto the record, so a
        // row can be rendered with its caveat rather than beside it.
        XCTAssertEqual(provenance.joinableNote?.contains("durable across sessions"), true)
        XCTAssertTrue(
            DocumentProvenance.joinabilityHeadline.contains("a scene is not"),
            "the contrast with Scene Understanding was dropped"
        )
    }

    /// **Always false.** Nothing checks that the capture still exists, so the
    /// pointer may resolve to nothing — and a decode gap must not upgrade an
    /// unvalidated pointer into a validated one.
    func testTheCaptureIsNeverClaimedToStillExist() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.matched))
        XCTAssertEqual(response.documents.first?.frameReference?.captureIDValidated, false)

        var missing = DocumentFixtures.record
        var block = missing["provenance"] as! [String: Any]
        block["capture_id_validated"] = nil
        missing["provenance"] = block
        var payload = DocumentFixtures.matched
        payload["documents"] = [missing]
        let decoded = try XCTUnwrap(DocumentMemoryDecoder.library(from: payload))
        XCTAssertEqual(decoded.documents[0].frameReference?.captureIDValidated, false)
    }

    /// Purging every document leaves the frames exactly where they are, and a
    /// person deleting their reading history would otherwise assume otherwise.
    func testTheImageryLifetimeIsNotThisCartridgesToSet() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.matched))
        XCTAssertEqual(response.documents.first?.frameReference?.imageryRetention, "capture-side")
        let note = try XCTUnwrap(response.recordNotes["imagery_retention"])
        XCTAssertTrue(note.contains("leaves that imagery exactly where it is"), "got: \(note)")
    }
}

// MARK: - Lifecycle

/// Stop **keeps** documents, and 200 is not success.
@MainActor
final class DocumentSessionTests: XCTestCase {

    func testTheSessionBlockKeepsItsShapeAndItsDefaults() throws {
        let session = try XCTUnwrap(DocumentMemoryDecoder.session(from: DocumentFixtures.sessionBlock))
        XCTAssertEqual(session.state, "stopped")
        XCTAssertTrue(session.states.contains("unavailable"))
        // The opposite of Scene Understanding's default, and the asymmetry is
        // the difference between the two cartridges: this one writes.
        XCTAssertFalse(session.followsStream)
        // Off by default, and it must stay off: no redaction exists here.
        XCTAssertFalse(session.keepsPageImages)
        XCTAssertFalse(session.captureIDValidated)
        XCTAssertEqual(session.retentionDays, 30.0)
        XCTAssertEqual(session.librarySoftLimit, 10000)
        XCTAssertFalse(session.libraryOverSoftLimit)
        XCTAssertEqual(session.librarySoftLimitNote?.contains("never enforced"), true)
    }

    /// The absent shape: every field present and null, `state: "unavailable"`,
    /// and a reason. A block that changed shape forced a decoder to make thirty
    /// fields optional and to handle a state its own enum denied existed.
    func testTheAbsentSessionIsTheSameShape() throws {
        var absent = DocumentFixtures.sessionBlock
        absent["state"] = "unavailable"
        absent["reason"] =
            "this Tower runs no document capture session (TOWER_DOCUMENT_CAPTURE is off, or TOWER_DOCUMENT_ROOT is unset). Documents recorded elsewhere are still served"
        let session = try XCTUnwrap(DocumentMemoryDecoder.session(from: absent))
        XCTAssertTrue(session.isUnavailable)
        XCTAssertEqual(session.reason?.contains("TOWER_DOCUMENT_CAPTURE"), true)
    }

    /// **200 is not success.** A resume on a stopped session answers 200 with
    /// `state: "stopped"` and no refusal field at all, so the returned state is
    /// the only thing that says whether anything moved.
    func testAVerbIsJudgedByTheStateThatComesBackNotByTheStatusCode() throws {
        let stopped = try XCTUnwrap(DocumentMemoryDecoder.session(from: DocumentFixtures.sessionBlock))
        for action in [
            DocumentMemoryContract.SessionAction.start, .pause, .resume,
        ] {
            let outcome = DocumentSessionOutcome.of(action, reported: stopped)
            guard case .notHonoured(_, let explanation) = outcome else {
                return XCTFail("\(action.rawValue) on a stopped session was read as success")
            }
            XCTAssertTrue(explanation.contains("stopped"), "the state was not named")
            XCTAssertTrue(
                explanation.contains("success code"),
                "the silent no-op was not explained"
            )
        }
        // Stop from stopped is genuinely honoured — the state asked for is the
        // state it is in.
        guard case .honoured = DocumentSessionOutcome.of(.stop, reported: stopped) else {
            return XCTFail("stop on a stopped session should be honoured")
        }
    }

    /// A start is not synchronous: the OCR reader takes about five seconds to
    /// construct, so `starting` counts as honoured and the caller polls.
    func testAStartIsHonouredWhileItIsStillLoading() throws {
        let starting = try XCTUnwrap(
            DocumentMemoryDecoder.session(from: DocumentFixtures.session(state: "starting"))
        )
        guard case .honoured = DocumentSessionOutcome.of(.start, reported: starting) else {
            return XCTFail("a loading recogniser was read as a refused start")
        }
        let running = try XCTUnwrap(
            DocumentMemoryDecoder.session(from: DocumentFixtures.session(state: "running"))
        )
        guard case .honoured = DocumentSessionOutcome.of(.resume, reported: running) else {
            return XCTFail("a running session was read as a refused resume")
        }
    }

    /// **Stop keeps documents, and flushes a dwell in progress.** The opposite
    /// of Scene Understanding's stop, and the field that makes it observable is
    /// `flushed_document_id`.
    func testAStopFlushesTheReadingInProgressRatherThanDroppingIt() throws {
        var block = DocumentFixtures.session(state: "stopped")
        block["flushed_document_id"] = "doc-9"
        block["documents_recorded"] = 3
        let session = try XCTUnwrap(DocumentMemoryDecoder.session(from: block))
        XCTAssertEqual(session.flushedDocumentID, "doc-9")
        XCTAssertEqual(session.documentsRecorded, 3)
        // Nothing about a stop clears what was recorded, which is why there is
        // no code here to assert against: the absence is the behaviour. What is
        // asserted is that the count survives the state.
        XCTAssertFalse(session.isRunning)
    }

    /// The subscription carries the session and the library, and **not the
    /// documents.** They are bulk text, the result sender shares a send lock
    /// with the frame path, and a listing must not become a bulk transfer of
    /// everything a wearer read onto whatever subscribed.
    func testTheSubscriptionCarriesProgressAndNotTheDocuments() throws {
        let status = try XCTUnwrap(
            DocumentMemoryDecoder.status(from: DocumentFixtures.socketStatus)
        )
        // Carried IN the payload, so a client that reads only this channel
        // still learns the documents are elsewhere.
        XCTAssertEqual(status.contractNote?.contains("/documents"), true)
        XCTAssertEqual(status.claim, "a-page-was-in-view-and-was-ocred")
        XCTAssertFalse(status.imageryServed)
        XCTAssertEqual(status.recordingLimitations.count, 4)
        XCTAssertEqual(status.session.state, "stopped")
        XCTAssertFalse(status.session.followsStream)

        // There is no key on this payload that could hold a document.
        XCTAssertNil(DocumentFixtures.socketStatus["documents"])
        XCTAssertNil(DocumentFixtures.socketStatus["document"])
        XCTAssertNil(DocumentFixtures.socketStatus["pages"])
    }

    /// `library.document_count_unfiltered` and `session.library_count` are
    /// **different quantities** and are named apart for that reason: the first
    /// counts every parseable record on disk, the second is the same count
    /// through the session's retention window and is refreshed only by a prune.
    func testTheTwoLibraryCountsAreNotTheSameNumber() throws {
        let status = try XCTUnwrap(
            DocumentMemoryDecoder.status(from: DocumentFixtures.socketStatus)
        )
        XCTAssertEqual(status.library.documentCountUnfiltered, 0)
        XCTAssertFalse(status.library.retentionApplied)
        // Null until a prune has run, which is not the same as zero.
        XCTAssertNil(status.session.libraryCount)
        // The Tower does not disclose where on disk the library lives.
        XCTAssertFalse(status.library.locationDisclosed)
    }

    /// A Tower whose constants have drifted is refused on this channel too, not
    /// just on HTTP. The two contracts are separate, and so are the two checks.
    func testTheStatusPayloadIsAlsoAssertedAgainstTheConstants() {
        var drifted = DocumentFixtures.socketStatus
        drifted["identity"] = "no-document-identity-across-sightings"
        XCTAssertNil(DocumentMemoryDecoder.status(from: drifted))
    }

    /// The 2026-09-07 additions decode, and their absence decodes to the
    /// defaults a 2026-08-27-shaped payload implies.
    func testTheLiveUpdateFieldsDecodeWithDefaults() throws {
        let bare = try XCTUnwrap(
            DocumentMemoryDecoder.status(from: DocumentFixtures.socketStatus)
        )
        XCTAssertNil(bare.library.revision)
        XCTAssertNil(bare.session.ocrDevice)
        XCTAssertNil(bare.session.documentsResighted)
        XCTAssertFalse(bare.session.idleStopPending)

        var live = DocumentFixtures.socketStatus
        var library = live["library"] as! [String: Any]
        library["revision"] = 1_757_200_000_000_000_000
        live["library"] = library
        var session = live["session"] as! [String: Any]
        session["ocr_device"] = "cuda"
        session["documents_resighted"] = 2
        session["dwells_unreadable"] = 1
        session["idle_stop_pending"] = true
        live["session"] = session

        let decoded = try XCTUnwrap(DocumentMemoryDecoder.status(from: live))
        XCTAssertEqual(decoded.library.revision, 1_757_200_000_000_000_000)
        XCTAssertEqual(decoded.session.ocrDevice, "cuda")
        XCTAssertEqual(decoded.session.documentsResighted, 2)
        XCTAssertEqual(decoded.session.dwellsUnreadable, 1)
        XCTAssertTrue(decoded.session.idleStopPending)
    }

    /// A record seen once decodes as such; a record with sightings carries
    /// them, and the first observation's `observed_at` is untouched.
    func testSightingsDecodeOnADocument() throws {
        let response = try XCTUnwrap(
DocumentMemoryDecoder.library(from: DocumentFixtures.recent)
        )
        let first = try XCTUnwrap(response.documents.first)
        XCTAssertEqual(first.sightingCount, 1)
        XCTAssertEqual(first.lastObservedAt, first.time.observedAt)

        var payload = DocumentFixtures.recent
        var documents = payload["documents"] as! [[String: Any]]
        documents[0]["sighting_count"] = 3
        documents[0]["last_observed_at"] = 1_700_000_900.0
        documents[0]["pages_readable"] = 1
        payload["documents"] = documents
        let seenAgain = try XCTUnwrap(
            DocumentMemoryDecoder.library(from: payload)
        )
        let record = try XCTUnwrap(seenAgain.documents.first)
        XCTAssertEqual(record.sightingCount, 3)
        XCTAssertEqual(record.lastObservedAt, Date(timeIntervalSince1970: 1_700_000_900))
        XCTAssertEqual(record.pagesReadable, 1)
        XCTAssertEqual(record.time.observedAt, first.time.observedAt)
    }

    /// A deletion that quietly failed looks exactly like one that was kept, so
    /// it is reported rather than logged.
    func testAnIncompleteRetentionSweepIsReported() throws {
        var block = DocumentFixtures.session(state: "running")
        block["retention_incomplete"] = true
        block["documents_pruned"] = 2
        let session = try XCTUnwrap(DocumentMemoryDecoder.session(from: block))
        XCTAssertTrue(session.retentionIncomplete)
    }
}

// MARK: - Configuration answers

/// What a 404 means on this Tower, which is usually **not** "no such route".
@MainActor
final class DocumentNotFoundTests: XCTestCase {

    private func body(_ detail: String) -> Data {
        try! JSONSerialization.data(withJSONObject: ["detail": detail])
    }

    /// `/documents*` 404 when `TOWER_DOCUMENT_ROOT` is unset. A statement about
    /// configuration, and **never** the answer to a query about a document.
    func testAMissingRootIsAConfigurationAnswerNamingTheVariable() {
        let error = DocumentMemoryHTTPClient.meaning(
            ofNotFound: body("no document root is configured on this Tower (TOWER_DOCUMENT_ROOT is unset)"),
            route: .library
        )
        guard case .noDocumentRootConfigured(let detail) = error else {
            return XCTFail("got \(error)")
        }
        XCTAssertTrue(detail.contains("TOWER_DOCUMENT_ROOT"))
        XCTAssertEqual(error.failure.kind, .notSupported)
        // Never `.transport`: sending someone to check their network over an
        // unset variable wastes their time.
        XCTAssertNotEqual(error.failure.kind, .transport)
    }

    /// `/documents-session*` 404 when capture is off **even with a root set** —
    /// a Tower serving a library recorded elsewhere and recording nothing
    /// itself, which is a normal configuration.
    ///
    /// The detail names both variables, so capture must be tested first or
    /// every session 404 reports a missing root and sends an operator to set a
    /// variable that is already set.
    func testCaptureIsCheckedBeforeRootBecauseTheDetailNamesBoth() {
        let error = DocumentMemoryHTTPClient.meaning(
            ofNotFound: body(
                "this Tower runs no document capture session (TOWER_DOCUMENT_CAPTURE is off, or TOWER_DOCUMENT_ROOT is unset). Documents recorded elsewhere are still served"
            ),
            route: .session
        )
        guard case .noCaptureSessionConfigured = error else { return XCTFail("got \(error)") }
    }

    /// The one 404 on this cartridge that **is** about a resource.
    func testAnUnknownDocumentIdIsAResourceAnswer() {
        let error = DocumentMemoryHTTPClient.meaning(
            ofNotFound: body("no document 'nosuch' in this memory"),
            route: .oneDocument(id: "nosuch")
        )
        guard case .noSuchDocument(let id, _) = error else { return XCTFail("got \(error)") }
        XCTAssertEqual(id, "nosuch")
    }

    /// **A 404 with no `TOWER_` in it, anywhere but that one route, is a
    /// genuine routing bug** and is surfaced as one rather than swallowed as
    /// "not configured".
    func testAStray404IsReportedAsADefectRatherThanAsConfiguration() {
        let error = DocumentMemoryHTTPClient.meaning(ofNotFound: body("Not Found"), route: .library)
        guard case .routingBug = error else { return XCTFail("got \(error)") }
        XCTAssertEqual(error.failure.kind, .undecodableResponse)
        XCTAssertTrue(error.failure.message.contains("route itself was not found"))
    }
}

// MARK: - Contract identity

@MainActor
final class DocumentContractTests: XCTestCase {

    /// **Two identifiers, deliberately.** They govern different transports with
    /// different failure modes, and a change to one is not a change to the
    /// other.
    func testTheTwoIdentifiersAreSeparateAndNeverInterchanged() {
        XCTAssertEqual(
            DocumentMemoryContract.statusIdentifier, "document_memory.status/2026-09-07"
        )
        XCTAssertEqual(
            DocumentMemoryContract.libraryIdentifier, "document_memory.library/2026-09-07"
        )
        XCTAssertNotEqual(
            DocumentMemoryContract.statusIdentifier, DocumentMemoryContract.libraryIdentifier
        )
        XCTAssertEqual(DocumentMemoryContract.entryRoute, "/documents")
        // Unlike Scene Understanding, this cartridge's result type IS `status`.
        XCTAssertEqual(DocumentMemoryContract.resultType, "status")
    }

    /// The library identifier is what every HTTP response carries — including
    /// `/documents-session`, which is what the wire actually says.
    func testEveryHTTPResponseCarriesTheLibraryIdentifier() throws {
        for payload in [DocumentFixtures.recent, DocumentFixtures.search, DocumentFixtures.session] {
            XCTAssertEqual(
                payload["contract"] as? String, DocumentMemoryContract.libraryIdentifier
            )
        }
    }

    /// The claim is `a-page-was-in-view-and-was-ocred`, **not** "was read". The
    /// contract document carried the older spelling until 2026-08-27, five keys
    /// above the note saying the camera cannot establish it.
    func testTheRetiredWasReadClaimIsRefusedRatherThanRendered() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.recent))
        XCTAssertEqual(response.claim, "a-page-was-in-view-and-was-ocred")
        var old = DocumentFixtures.recent
        old["claim"] = "a-document-was-read"
        XCTAssertNil(DocumentMemoryDecoder.library(from: old))
    }

    /// `semantic_retrieval` is false and the alternative names the literal
    /// route. Calling it semantic would be an overclaim.
    func testRetrievalIsLexicalAndTheAlternativeIsPublished() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.recent))
        XCTAssertFalse(response.semanticRetrieval)
        XCTAssertNotNil(response.semanticRetrievalUnavailableReason)
        XCTAssertEqual(response.semanticRetrievalAlternative?.contains("'text'"), true)
        XCTAssertEqual(response.retrievalKinds, ["recent", "text", "observed_within"])
        XCTAssertFalse(response.paginationSupported)
        XCTAssertEqual(response.paginationBound, "limit")
    }

    /// A typed sentence is a **description**, and the answer says the matching
    /// was literal so a miss is not read as an empty memory.
    func testATypedQueryCarriesTheLexicalCaveat() {
        XCTAssertNotNil(DocumentQuery.semantic("the parking notice").matchingCaveat)
        XCTAssertEqual(
            DocumentQuery.semantic("x").matchingCaveat?.contains("does not search by meaning"), true
        )
        // A literal query needs no such caveat: the person quoted the page.
        XCTAssertNil(DocumentQuery.text("parking").matchingCaveat)
        XCTAssertNil(DocumentQuery.recent(limit: 5).matchingCaveat)
    }
}

// MARK: - The client

@MainActor
final class DocumentMemoryClientTests: XCTestCase {

    /// The stub says only what is observable from here, and neither of the two
    /// false claims that used to be on it.
    ///
    /// The retired sentence was: *"The Tower keeps no document memory, so there
    /// is nothing to search. This app sends camera frames to the Tower while a
    /// session is running; it reads no text from them and receives none back."*
    /// The Tower has a document store, a journal, six routes and a typed
    /// contract, and a search receives titles, snippets and page text.
    func testTheStubMakesNoClaimAboutWhatAnyTowerStores() {
        let reason = UnavailableDocumentMemoryClient.reason.lowercased()
        XCTAssertFalse(reason.contains("keeps no document memory"), "the retired sentence is still shipping")
        XCTAssertFalse(reason.contains("receives none back"))
        XCTAssertTrue(reason.contains("declared"), "got: \(reason)")
        XCTAssertEqual(UnavailableDocumentMemoryClient().state.phase, .unsupported)
    }

    /// A refusal, not a silent empty answer: an empty result reads as "you have
    /// no documents", which is a false statement about the user's own memory.
    func testSearchingIsRefusedRatherThanReturningEmpty() {
        let client = UnavailableDocumentMemoryClient()
        XCTAssertThrowsError(try client.search(.recent(limit: 10), origin: .appText)) { error in
            guard let failure = error as? CartridgeFailure else {
                return XCTFail("expected a CartridgeFailure, got \(error)")
            }
            // Not `.towerReportedFailure`: the Tower reported nothing, and
            // attributing a local refusal to the other machine is a fabricated
            // claim about it.
            XCTAssertEqual(failure.kind, .notSupported)
            XCTAssertFalse(failure.message.isEmpty)
        }
    }

    func testEveryDocumentStateMapsToTheRightPhase() throws {
        let result = try DocumentQueryResult(
            query: .recent(limit: 5), origin: .appText, evidence: .notFound
        )
        let expected: [(DocumentMemoryState, CartridgePhase)] = [
            (.unsupported(reason: "x"), .unsupported),
            (.idle, .idle),
            (.searching(.recent(limit: 5)), .waiting),
            (.results(result), .settled),
            (.failed(CartridgeFailure(kind: .transport, message: "x")), .failed),
        ]
        for (state, phase) in expected {
            XCTAssertEqual(state.phase, phase, "\(state) mapped to the wrong phase")
        }
    }

    /// The envelope travels with the answer, because every obligation this
    /// cartridge has lives on it. A result that dropped it would be a list of
    /// documents with every caveat removed.
    func testAResultKeepsTheEnvelopeItArrivedIn() throws {
        let response = try XCTUnwrap(DocumentMemoryDecoder.library(from: DocumentFixtures.recent))
        let result = try DocumentQueryResult(
            query: .recent(limit: 10), origin: .appText,
            documents: response.documents, evidence: response.answer.evidence,
            response: response
        )
        XCTAssertEqual(result.recordingLimitations.count, 4)
        XCTAssertNotNil(result.response?.noObservationNote)
    }
}
