//
//  DocumentMemoryDecoder.swift
//  Glasses
//

import Foundation

/// Turns the Tower's document payloads into the types the workspace renders.
///
/// ## What it refuses to do
///
/// Absent stays absent, and **nothing is defaulted to zero**. `null ≠ 0` on
/// this wire in more than one load-bearing place: `retention.writer_window_days`
/// is null because the store keeps no manifest for a reader to learn one from,
/// and a document's `observed_seconds` being absent is not a document that was
/// in view for no time.
///
/// The one place a default *is* applied is where the Tower publishes a constant
/// this build asserts anyway — `imagery_served: false`, `capture_id_validated:
/// false`, `joinable: true`. Those default to the value the contract fixes, so
/// a decode gap cannot turn "no image is served" into "an image might be".
enum DocumentMemoryDecoder {

    // MARK: The library

    /// A whole `/documents*` response, or `nil` when it cannot be read as this
    /// contract.
    ///
    /// `nil` becomes `DocumentMemoryFetchError.undecodable`, which the
    /// workspace renders as a failure. **It never becomes an empty library**:
    /// an unreadable answer and a memory with nothing in it are different
    /// facts, and turning the first into the second manufactures a statement
    /// about the wearer's own memory out of a decode gap.
    static func library(from json: [String: Any]) -> DocumentLibraryResponse? {
        guard
            let contract = json["contract"] as? String,
            let claim = json["claim"] as? String,
            let identity = json["identity"] as? String,
            let absenceMeans = json["absence_means"] as? String,
            let timeBasis = json["time_basis"] as? String,
            let answer = json["answer"] as? String
        else { return nil }

        // The constants, checked rather than trusted. `claim` in particular:
        // the contract document itself carried an older spelling that said a
        // document "was read" until 2026-08-27, five keys above the note saying
        // the camera cannot establish that. A Tower that started sending the
        // old spelling again would be making the stronger claim, and this build
        // refuses it rather than rendering it.
        guard
            claim == DocumentMemoryContract.claim,
            identity == DocumentMemoryContract.identityScope,
            absenceMeans == DocumentMemoryContract.absenceMeans,
            timeBasis == DocumentMemoryContract.timeBasis
        else { return nil }

        let pagination = json["pagination"] as? [String: Any] ?? [:]
        let notes = recordNotes(from: json["record_notes"])
        let receivedAt = Date()

        // `documents` on a listing, `document` on the single-document route.
        // Both are decoded through the same function, so a field rendered on a
        // row cannot be read differently from the same field on a detail
        // screen.
        var documents: [RememberedDocument] = []
        for raw in json["documents"] as? [[String: Any]] ?? [] {
            if let one = document(from: raw, receivedAt: receivedAt, notes: notes) {
                documents.append(one)
            }
        }

        var single: RememberedDocument?
        if let raw = json["document"] as? [String: Any] {
            single = document(from: raw, receivedAt: receivedAt, notes: notes)
        }

        // Pages live INSIDE the document, not beside it. The Tower builds the
        // single-document payload as
        // `payload["document"] = dict(_summary_view(...), pages=[...])`
        // (`tower/tower/results/document_memory.py`), and its own wire test
        // asserts `payload["document"]["pages"][0]["text"]`.
        //
        // This read was at the top level, where the Tower has never put it, so
        // `pages` came back empty against a real Tower and the page text --
        // the only thing the single-document route exists to carry -- never
        // arrived. Two things hid it: nothing renders pages yet, and the
        // fixture was written to match this decoder rather than the Tower, so
        // the test agreed with the bug rather than catching it.
        var pages: [DocumentPage] = []
        let documentBlock = json["document"] as? [String: Any]
        let rawPages = documentBlock?["pages"] as? [[String: Any]] ?? []
        for raw in rawPages {
            if let page = self.page(from: raw) { pages.append(page) }
        }

        return DocumentLibraryResponse(
            contract: contract,
            claim: claim,
            identity: identity,
            absenceMeans: absenceMeans,
            timeBasis: timeBasis,
            spatialRef: json["spatial_ref"] as? String,
            answers: json["answers"] as? [String] ?? [],
            retrievalKinds: json["retrieval_kinds"] as? [String] ?? [],
            // Defaults to `false`, which is the claim the Tower makes. A decode
            // gap must not turn "this is literal matching" into "this searches
            // by meaning" — that is the overclaim the field exists to prevent.
            semanticRetrieval: json["semantic_retrieval"] as? Bool ?? false,
            semanticRetrievalUnavailableReason:
                json["semantic_retrieval_unavailable_reason"] as? String,
            semanticRetrievalAlternative: json["semantic_retrieval_alternative"] as? String,
            paginationSupported: pagination["supported"] as? Bool ?? false,
            paginationBound: pagination["bound"] as? String,
            paginationReason: pagination["reason"] as? String,
            privacyTagVocabulary: json["privacy_tag_vocabulary"] as? [String] ?? [],
            recordNotes: notes,
            recordingLimitations: limitations(from: json["recording_limitations"]),
            recordingMeasurement: measurement(from: json["recording_measurement"] as? [String: Any]),
            imageryTreatment: json["imagery_treatment"] as? String,
            imageryIOSState: json["imagery_ios_state"] as? String,
            // `false` when absent, always. This is a boolean and not a path,
            // and the safe direction is "no image is served".
            imageryServed: json["imagery_served"] as? Bool ?? false,
            imageryNote: json["imagery_note"] as? String,
            retention: retention(from: json["retention"] as? [String: Any]),
            query: queryEcho(from: json["query"] as? [String: Any]),
            answer: DocumentLibraryAnswer(answer),
            noObservationNote: json["no_observation_note"] as? String,
            documentsInMemory: json["documents_in_memory"] as? Int ?? 0,
            documentCount: json["document_count"] as? Int ?? documents.count,
            documents: documents,
            searchedDocuments: json["searched_documents"] as? Int,
            minScore: json["min_score"] as? Double,
            sufficientEvidence: json["sufficient_evidence"] as? Bool,
            matchKind: json["match_kind"] as? String,
            reason: json["reason"] as? String,
            // **Read, never assumed.** 48 on this Tower, and configuration
            // dependent — the contract prose said 160 for months while the code
            // said 48, and the instruction that resolved it was to read the
            // field.
            snippetMaxChars: json["snippet_max_chars"] as? Int,
            document: single,
            pages: pages,
            coverage: coverage(from: json["coverage"] as? [String: Any])
        )
    }

    /// `record_notes`, hoisted to the envelope and keyed by the field each
    /// caveat qualifies.
    ///
    /// Five entries: `observed_seconds`, `summary_withheld`, `timing`,
    /// `imagery_retention`, `joinable`. **None of them may be dropped.** They
    /// used to be repeated on every record — 2,351 bytes per document — and
    /// hoisting them cut a 200-document listing from 488 KB to 249 KB with
    /// nothing lost. The saving was the repetition, not the caveats.
    static func recordNotes(from raw: Any?) -> [String: String] {
        guard let json = raw as? [String: Any] else { return [:] }
        var notes: [String: String] = [:]
        for (key, value) in json {
            if let text = value as? String { notes[key] = text }
        }
        return notes
    }

    static func limitations(from raw: Any?) -> [DocumentRecordingLimitation] {
        guard let entries = raw as? [[String: Any]] else { return [] }
        var result: [DocumentRecordingLimitation] = []
        for entry in entries {
            guard
                let slug = entry["limitation"] as? String,
                let detail = entry["detail"] as? String
            else { continue }
            result.append(DocumentRecordingLimitation(slug: slug, detail: detail))
        }
        return result
    }

    static func measurement(from json: [String: Any]?) -> DocumentRecordingMeasurement {
        let json = json ?? [:]
        return DocumentRecordingMeasurement(
            measuredAt: json["measured_at"] as? String,
            corpusFrames: json["corpus_frames"] as? Int,
            corpusCaptures: json["corpus_captures"] as? Int,
            // `false` when absent: the figures describe the frames they were
            // measured on and have not been re-derived. Defaulting to `true`
            // would assert a currency nobody has established.
            isCurrent: json["is_current"] as? Bool ?? false,
            note: json["note"] as? String
        )
    }

    /// `retention`. `writer_window_days` is read with no default and stays
    /// `nil`, which is the honest answer: `DocumentStore` persists no retention
    /// manifest, so a reader cannot learn the window its writer used.
    static func retention(from json: [String: Any]?) -> DocumentRetentionView {
        let json = json ?? [:]
        return DocumentRetentionView(
            requestedDays: json["requested_days"] as? Double,
            writerWindowDays: json["writer_window_days"] as? Double,
            writerWindowUnavailableReason: json["writer_window_unavailable_reason"] as? String,
            policy: json["policy"] as? String
        )
    }

    static func queryEcho(from json: [String: Any]?) -> DocumentQueryEcho {
        let json = json ?? [:]
        return DocumentQueryEcho(
            kind: json["kind"] as? String,
            limit: json["limit"] as? Int,
            text: json["text"] as? String,
            centre: json["centre"] as? Double,
            windowSeconds: json["window_seconds"] as? Double,
            documentID: json["document_id"] as? String
        )
    }

    static func coverage(from json: [String: Any]?) -> DocumentCoverage? {
        guard let json else { return nil }
        return DocumentCoverage(
            pagesObserved: json["pages_observed"] as? Int,
            // Always null on the wire, and read anyway rather than hard-coded
            // nil: the camera cannot know how many pages a document has, and
            // `pages_total_note` says exactly that.
            pagesTotal: json["pages_total"] as? Int,
            pagesTotalNote: json["pages_total_note"] as? String,
            wordsCaptured: json["words_captured"] as? Int,
            lowConfidencePages: json["low_confidence_pages"] as? [Int] ?? []
        )
    }

    // MARK: One document

    /// One record, from a listing or from the single-document route.
    ///
    /// `notes` is the envelope's hoisted `record_notes`; the caveats that used
    /// to live on the record are attached back here so a row can render a
    /// document *with* them rather than beside them.
    static func document(
        from json: [String: Any], receivedAt: Date, notes: [String: String]
    ) -> RememberedDocument? {
        guard let id = json["document_id"] as? String else { return nil }

        let availability = json["text_availability"] as? [String: Any] ?? [:]
        let observedAt = json["observed_at"] as? Double

        var document = RememberedDocument(
            id: id,
            // Already clipped to `title_max_chars` by the Tower. Not clipped
            // again here: two places applying the same bound is two places for
            // it to be applied differently.
            title: json["title"] as? String,
            summary: json["summary"] as? String,
            text: DocumentTextAvailability(
                state: availability["state"] as? String,
                characterCount: availability["character_count"] as? Int
            ),
            // Tower-receipt time in both fields, which is why the caveat below
            // is attached rather than assumed to be understood: this app's
            // `ObservationTime.observedAt` is documented as when the glasses
            // observed something, and on this wire that clock does not exist.
            // `record_notes.timing` is the Tower's own sentence saying so and
            // the workspace renders it.
            time: ObservationTime(
                observedAt: observedAt.map(Date.init(timeIntervalSince1970:)),
                receivedAt: receivedAt
            ),
            observedDuration: (json["observed_seconds"] as? Double)
                .map(ObservedDuration.init(seconds:)),
            // The title is lifted from the document's own first text region by
            // a recogniser, and the text is OCR output. Both are model output,
            // which is `.inferred` — and with no numeric confidence, because
            // the Tower publishes a *word* rather than a number and inventing a
            // percentage from it would be worse than showing the word.
            provenance: .inferred(confidence: nil),
            // No route serves an image, so there is nothing to fetch and
            // nothing to decide about.
            thumbnail: .absent,
            source: DocumentSourceContext(
                sessionID: (json["provenance"] as? [String: Any])?["world_session_id"] as? String,
                worldID: (json["provenance"] as? [String: Any])?["world_id"] as? String
            ),
            titleIsDerived: json["title_is_derived"] as? Bool ?? true,
            titleMaxChars: json["title_max_chars"] as? Int,
            summaryAvailable: json["summary_available"] as? Bool ?? false,
            confidenceWord: json["confidence"] as? String,
            confidenceBasis: json["confidence_basis"] as? String,
            pagesObserved: json["pages_observed"] as? Int,
            endReason: json["end_reason"] as? String,
            timing: timing(from: json["timing"] as? [String: Any], note: notes["timing"]),
            frameReference: provenance(
                from: json["provenance"] as? [String: Any],
                joinableNote: notes["joinable"],
                retentionNote: notes["imagery_retention"]
            ),
            retainsRawImagery: json["retains_raw_imagery"] as? Bool ?? false,
            redaction: json["redaction"] as? String,
            imageryTreatment: json["imagery_treatment"] as? String,
            privacyTags: json["privacy_tags"] as? [String] ?? [],
            schemaVersion: json["schema_version"] as? Int,
            wordCount: json["word_count"] as? Int,
            summaryIsVerbatimExcerpt: json["summary_is_verbatim_excerpt"] as? Bool ?? false,
            summaryIsModelOutput: json["summary_is_model_output"] as? Bool ?? false,
            match: matchEvidence(from: json)
        )
        // Sightings (`document_memory.library/2026-09-07`). Defaults describe
        // a record seen once, which is what an absent field means.
        document.sightingCount = json["sighting_count"] as? Int ?? 1
        document.lastObservedAt = ((json["last_observed_at"] as? Double) ?? observedAt)
            .map(Date.init(timeIntervalSince1970:))
        document.pagesReadable = json["pages_readable"] as? Int
        return document
    }

    /// The four search-only fields, or `nil` on a listing that has none.
    static func matchEvidence(from json: [String: Any]) -> DocumentMatchEvidence? {
        let score = json["score"] as? Double
        let terms = json["matched_terms"] as? [String]
        let snippet = json["snippet"] as? String
        guard score != nil || terms != nil || snippet != nil else { return nil }
        return DocumentMatchEvidence(
            score: score, matchedTerms: terms ?? [], snippet: snippet
        )
    }

    static func timing(from json: [String: Any]?, note: String?) -> DocumentTiming? {
        guard let json else { return nil }
        return DocumentTiming(
            timeBasis: json["time_basis"] as? String,
            source: json["source"] as? String,
            assumedFrameIntervalSeconds: json["assumed_frame_interval_s"] as? Double,
            // The envelope's hoisted note wins where the record carries none,
            // which is the normal case now that the caveats live upstairs.
            note: json["note"] as? String ?? note
        )
    }

    static func provenance(
        from json: [String: Any]?, joinableNote: String?, retentionNote: String?
    ) -> DocumentProvenance? {
        guard let json else { return nil }
        return DocumentProvenance(
            kind: json["kind"] as? String,
            captureID: json["capture_id"] as? String,
            // `false` when absent, always. Nothing on this Tower checks that
            // the capture still exists, and a decode gap must not upgrade an
            // unvalidated pointer into a validated one.
            captureIDValidated: json["capture_id_validated"] as? Bool ?? false,
            pageSourceSeqs: json["page_source_seqs"] as? [Int] ?? [],
            pagesWithoutSourceSeq: json["pages_without_source_seq"] as? Int,
            framesConsidered: json["frames_considered"] as? Int,
            framesOcred: json["frames_ocred"] as? Int,
            worldID: json["world_id"] as? String,
            worldSessionID: json["world_session_id"] as? String,
            imageryRetention: json["imagery_retention"] as? String,
            imageryRetentionNote: json["imagery_retention_note"] as? String ?? retentionNote,
            // `true` when absent. This block *is* joinable and the honest
            // default is the one that says so — a decode gap must not quietly
            // downgrade a durable pointer into a recording to "not joinable".
            joinable: json["joinable"] as? Bool ?? true,
            joinableNote: json["joinable_note"] as? String ?? joinableNote
        )
    }

    static func page(from json: [String: Any]) -> DocumentPage? {
        guard let index = json["page_index"] as? Int else { return nil }
        return DocumentPage(
            pageIndex: index,
            // An empty string is `not_readable`, not "no page". Defaulted to
            // empty rather than refused for that reason.
            text: json["text"] as? String ?? "",
            textSource: json["text_source"] as? String,
            regionCount: json["region_count"] as? Int ?? 0,
            meanRegionConfidence: json["mean_region_confidence"] as? Double,
            minRegionConfidence: json["min_region_confidence"] as? Double,
            confidence: json["confidence"] as? String,
            sharpness: json["sharpness"] as? Double,
            squareness: json["squareness"] as? Double,
            sourceSeq: json["source_seq"] as? Int,
            observedAt: json["observed_at"] as? Double,
            observationCount: json["observation_count"] as? Int ?? 1,
            imageKept: json["image_kept"] as? Bool ?? false,
            // Always `false`. A boolean and not the path.
            imageServed: json["image_served"] as? Bool ?? false
        )
    }

    // MARK: The session and the subscription

    /// The whole `document_memory.status` payload from the socket.
    ///
    /// Note what is **not** here: the documents. They are bulk text, the result
    /// sender shares a lock with the frame path, and a listing must not become
    /// a bulk transfer of everything a wearer read onto whatever subscribed.
    /// `contract_note` carries that fact inside the payload so a client reading
    /// only this channel still learns the documents are on HTTP.
    static func status(from payload: [String: Any]) -> DocumentMemoryStatus? {
        guard
            let claim = payload["claim"] as? String,
            let identity = payload["identity"] as? String,
            let absenceMeans = payload["absence_means"] as? String,
            let timeBasis = payload["time_basis"] as? String,
            let rawSession = payload["session"] as? [String: Any],
            let session = self.session(from: rawSession)
        else { return nil }

        guard
            claim == DocumentMemoryContract.claim,
            identity == DocumentMemoryContract.identityScope,
            absenceMeans == DocumentMemoryContract.absenceMeans,
            timeBasis == DocumentMemoryContract.timeBasis
        else { return nil }

        return DocumentMemoryStatus(
            contractNote: payload["contract_note"] as? String,
            claim: claim,
            identity: identity,
            absenceMeans: absenceMeans,
            timeBasis: timeBasis,
            library: librarySummary(from: payload["library"] as? [String: Any]),
            session: session,
            recordingLimitations: limitations(from: payload["recording_limitations"]),
            recordingMeasurement: measurement(from: payload["recording_measurement"] as? [String: Any]),
            imageryTreatment: payload["imagery_treatment"] as? String,
            imageryIOSState: payload["imagery_ios_state"] as? String,
            imageryServed: payload["imagery_served"] as? Bool ?? false,
            imageryNote: payload["imagery_note"] as? String
        )
    }

    static func librarySummary(from json: [String: Any]?) -> DocumentLibrarySummary {
        let json = json ?? [:]
        let bytes = json["bytes"] as? [String: Any] ?? [:]
        return DocumentLibrarySummary(
            available: json["available"] as? Bool ?? false,
            documentCountUnfiltered: json["document_count_unfiltered"] as? Int,
            retentionApplied: json["retention_applied"] as? Bool ?? false,
            unavailableReason: json["unavailable_reason"] as? String,
            newestObservedAt: json["newest_observed_at"] as? Double,
            journalBytes: bytes["journal"] as? Int,
            imageBytes: bytes["images"] as? Int,
            totalBytes: bytes["total"] as? Int,
            locationDisclosed: json["location_disclosed"] as? Bool ?? false,
            revision: json["revision"] as? Int
        )
    }

    /// The session block, which keeps its shape in every state.
    ///
    /// `state` is the only required key: when no session exists it is
    /// `"unavailable"` and every other field is present and null, so a decoder
    /// that demanded more would refuse the one shape that says why there is
    /// nothing.
    static func session(from json: [String: Any]) -> DocumentSessionStatus? {
        guard let state = json["state"] as? String else { return nil }
        return DocumentSessionStatus(
            state: state,
            states: json["states"] as? [String] ?? [],
            sessionID: json["session_id"] as? Int,
            failureReason: json["failure_reason"] as? String,
            startedAt: json["started_at"] as? Double,
            readyAt: json["ready_at"] as? Double,
            loadingSeconds: json["loading_seconds"] as? Double,
            loadOverdue: json["load_overdue"] as? Bool ?? false,
            loadOverdueAfterSeconds: json["load_overdue_after_seconds"] as? Double,
            engine: json["engine"] as? String,
            recogniser: json["recogniser"] as? String,
            framesOffered: json["frames_offered"] as? Int,
            framesObserved: json["frames_observed"] as? Int,
            framesSkipped: json["frames_skipped"] as? Int,
            framesDroppedNotRunning: json["frames_dropped_not_running"] as? Int,
            captureID: json["capture_id"] as? String,
            captureIDValidated: json["capture_id_validated"] as? Bool ?? false,
            inDwell: json["in_dwell"] as? Bool ?? false,
            dwellsStarted: json["dwells_started"] as? Int,
            pagesDetected: json["pages_detected"] as? Int,
            documentsRecorded: json["documents_recorded"] as? Int,
            lastDocumentID: json["last_document_id"] as? String,
            lastDocumentAt: json["last_document_at"] as? Double,
            flushedDocumentID: json["flushed_document_id"] as? String,
            // `false` when absent. Page images must stay off, and the safe
            // direction for a decode gap is the one that does not claim an
            // unredacted photograph is being kept.
            keepsPageImages: json["keeps_page_images"] as? Bool ?? false,
            // `false` when absent, which is this cartridge's own default and
            // the opposite of Scene Understanding's: a session that persists
            // what a wearer read gets an explicit start.
            followsStream: json["follows_stream"] as? Bool ?? false,
            retentionDays: json["retention_days"] as? Double,
            documentsPruned: json["documents_pruned"] as? Int,
            retentionIncomplete: json["retention_incomplete"] as? Bool ?? false,
            libraryCount: json["library_count"] as? Int,
            librarySoftLimit: json["library_soft_limit"] as? Int,
            libraryOverSoftLimit: json["library_over_soft_limit"] as? Bool ?? false,
            librarySoftLimitNote: json["library_soft_limit_note"] as? String,
            reason: json["reason"] as? String,
            // Present and null on a 2026-08-27 payload's shape; absent
            // entirely from an older Tower. Both decode to the defaults.
            ocrDevice: json["ocr_device"] as? String,
            documentsResighted: json["documents_resighted"] as? Int,
            dwellsUnreadable: json["dwells_unreadable"] as? Int,
            idleStopPending: json["idle_stop_pending"] as? Bool ?? false
        )
    }
}
