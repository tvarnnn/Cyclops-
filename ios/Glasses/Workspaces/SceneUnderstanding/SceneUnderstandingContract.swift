//
//  SceneUnderstandingContract.swift
//  Glasses
//

import Foundation

// MARK: - The contract this build implements

/// The one Scene Understanding agreement this build was written against.
///
/// Opaque, and compared for equality only. Dated rather than numbered precisely
/// so that nobody is tempted to compute which is greater: a mismatch means "we
/// are not talking about the same agreement", which is neither newer nor older,
/// and `CartridgeAvailability.unsupportedContract` is the honest rendering of
/// it.
///
/// ## The result type is `live`, and it is the only cartridge where it differs
///
/// World Builder, Document Memory and the Experimental CV Lab all publish
/// `status`. This one publishes `live`, because the payload **is** the answer
/// rather than progress toward one. A `result_subscribe` naming the wrong
/// result type is refused by the Tower with `unknown_result_type`, so this
/// constant is load-bearing rather than descriptive.
nonisolated enum SceneUnderstandingContract {
    /// The **Tower's** name for the cartridge. Not this app's catalog id
    /// (`"scene-understanding"`); the two strings differ and the mapping lives
    /// in `TowerCapabilities`.
    static let towerCartridge = "scene_understanding"
    /// `live`, not `status`. See the type's note.
    static let resultType = "live"
    static let identifier = "scene_understanding.live/2026-08-27"

    // MARK: The constant self-description
    //
    // Six strings the Tower publishes on every payload in every state. They are
    // asserted at decode time rather than merely read: a Tower that changed one
    // of them while keeping the identifier would be making a different promise
    // under the same name, and absorbing that silently is how a client ends up
    // rendering a record as if it were a live view.

    /// This is the present, not history.
    static let claim = "visible-now-not-a-record"
    /// Nothing here identifies anyone, and no handle is published.
    static let identityScope = "anonymous-and-unpublished"
    /// A zero is about this camera's forward cone, never about the room.
    static let absenceMeans = "not-visible-to-this-cartridge"
    /// Nothing is written; there is nothing to purge.
    static let persistence = "none"
    /// Everything positional is camera-relative and changes when the wearer
    /// turns their head.
    static let frameOfReference = "camera"
    /// Every timestamp on this wire. There is no capture clock.
    static let timeBasis = "tower-receipt"

    // MARK: Routes
    //
    // Listed for completeness and **not called by this app**. `GET /scene`
    // carries the same payload the subscription does, and the four POSTs are an
    // operator surface: `IOS-to-Tower.md` §6.2 is explicit that opening a
    // cartridge on the phone sends nothing, and a Tower-side test asserts the
    // wire stays silent. The session follows the stream instead —
    // `stream_start` opens it, `stream_stop` or a disconnect ends it — which is
    // the normal case for a wearable.
    //
    // Two further reasons not to grow a Start button here, both measured:
    //
    // 1. `POST /scene/resume` on a stopped scene answers **200 with
    //    `state: "stopped"` and no refusal field**. A client keyed on the
    //    status code would report success for a verb that did nothing. The
    //    cheapest way to get that right is not to send it.
    // 2. A phone's `stream_stop` can already end a session an operator started
    //    by hand. Adding a second way for the phone to move the session makes
    //    that worse, not better.
    static let statusRoute = "/scene"
    static let controlRoutes = ["/scene/start", "/scene/pause", "/scene/resume", "/scene/stop"]

    /// Why this app sends nothing, in a sentence a person could be shown.
    static let phoneSendsNothingNote = """
        This app never starts or stops the Tower's scene reading. The session \
        follows the camera stream, and the controls belong to whoever runs the \
        Tower.
        """
}

// MARK: - Decoding

/// Turns a `scene_understanding.live` payload into `SceneReading`.
///
/// ## Why so few optionals
///
/// > Every key is present in every state. A key that appeared and disappeared
/// > would force a decoder to treat it as optional and lose the ability to tell
/// > "zero of these" from "this Tower did not say".
///
/// So a missing key here is not "the Tower had nothing to say about that" — it
/// is a payload that is not this contract, and the honest response is to refuse
/// the whole thing rather than to decode a partial scene. `nil` from
/// `reading(from:)` is rendered as `CartridgeFailure.Kind.undecodableResponse`
/// by the caller, never as an empty room.
///
/// The optionals that remain are the fields the wire genuinely carries as
/// `null`: `observed_at`, `staleness_seconds`, `detector`, `score_threshold`,
/// `failure_reason`, `people.facing_wearer`, and the scene block itself.
enum SceneUnderstandingDecoder {

    /// The whole payload, or `nil` when it cannot be read as this contract.
    static func reading(from payload: [String: Any]) -> SceneReading? {
        guard
            let claim = payload["claim"] as? String,
            let identity = payload["identity"] as? String,
            let absenceMeans = payload["absence_means"] as? String,
            let persistence = payload["persistence"] as? String,
            let frameOfReference = payload["frame_of_reference"] as? String,
            let timeBasis = payload["time_basis"] as? String,
            let rawLifecycle = payload["lifecycle"] as? [String: Any],
            let lifecycle = self.lifecycle(from: rawLifecycle),
            let reportedClasses = payload["reported_classes"] as? [String],
            let countBasis = payload["count_basis"] as? String,
            let countIsLowerBound = payload["count_is_lower_bound"] as? Bool,
            let sceneAvailable = payload["scene_available"] as? Bool
        else { return nil }

        // The six constants, checked rather than trusted. A Tower that changed
        // one of these while keeping the identifier is making a different
        // promise under the same name; refusing is what turns that into a
        // visible failure instead of a subtly wrong screen.
        guard
            claim == SceneUnderstandingContract.claim,
            identity == SceneUnderstandingContract.identityScope,
            absenceMeans == SceneUnderstandingContract.absenceMeans,
            persistence == SceneUnderstandingContract.persistence,
            frameOfReference == SceneUnderstandingContract.frameOfReference,
            timeBasis == SceneUnderstandingContract.timeBasis
        else { return nil }

        // `count_is_lower_bound` is `true` on every payload the Tower can
        // produce. It is read rather than assumed anyway — the obligation is to
        // render what the Tower said, and a hard-coded `true` would keep
        // rendering the disclosure if the Tower ever stopped making the claim,
        // which is the mirror-image mistake.

        let observation: SceneObservation?
        let unavailableReason: SceneUnavailableReason?
        let unavailableText = payload["scene_unavailable_reason"] as? String

        if sceneAvailable {
            // `counts`, `where` and `people` are non-null together with
            // `scene_available: true`. A payload where they disagree is not a
            // half-scene to be rendered carefully; it is a payload this build
            // does not understand.
            guard
                let rawCounts = payload["counts"] as? [String: Any],
                let rawWhere = payload["where"] as? [String: Any],
                let rawPeople = payload["people"] as? [String: Any],
                let people = self.people(from: rawPeople)
            else { return nil }

            observation = SceneObservation(
                // Ordered by `reported_classes` rather than by the dictionary,
                // which has no order. The Tower's order is fixed at build time,
                // so a reader's eye stays in the same place between readings.
                counts: reportedClasses.map { label in
                    SceneClassCount(label: label, count: rawCounts[label] as? Int ?? 0)
                },
                positions: positions(from: rawWhere, ordering: reportedClasses),
                people: people,
                scoreThreshold: payload["score_threshold"] as? Double
            )
            unavailableReason = nil
        } else {
            observation = nil
            unavailableReason = self.unavailableReason(
                lifecycle: lifecycle, text: unavailableText
            )
        }

        return SceneReading(
            claim: claim,
            identity: identity,
            absenceMeans: absenceMeans,
            persistence: persistence,
            frameOfReference: frameOfReference,
            timeBasis: timeBasis,
            lifecycle: lifecycle,
            observedAtTowerReceipt: payload["observed_at"] as? Double,
            observedAtNote: payload["observed_at_note"] as? String,
            stalenessSeconds: payload["staleness_seconds"] as? Double,
            framesOffered: payload["frames_offered"] as? Int ?? 0,
            framesObserved: payload["frames_observed"] as? Int ?? 0,
            framesSkipped: payload["frames_skipped"] as? Int ?? 0,
            framesDroppedNotRunning: payload["frames_dropped_not_running"] as? Int ?? 0,
            decodeFailures: payload["decode_failures"] as? Int ?? 0,
            detector: payload["detector"] as? String,
            reportedClasses: reportedClasses,
            countBasis: countBasis,
            countIsLowerBound: countIsLowerBound,
            countLimitations: limitations(from: payload["count_limitations"]),
            countMeasurement: measurement(from: payload["count_measurement"] as? [String: Any]),
            sideConvention: payload["side_convention"] as? String,
            refusals: refusals(from: payload),
            observation: observation,
            unavailableReason: unavailableReason,
            unavailableReasonText: unavailableText,
            singlePersonNote: payload["single_person_note"] as? String
        )
    }

    // MARK: Lifecycle

    static func lifecycle(from json: [String: Any]) -> SceneLifecycle? {
        guard
            let state = json["state"] as? String,
            let sessionID = json["session_id"] as? Int,
            let sceneIsCurrent = json["scene_is_current"] as? Bool
        else { return nil }

        return SceneLifecycle(
            state: SceneLifecycleState(state),
            states: json["states"] as? [String] ?? [],
            sessionID: sessionID,
            sceneIsCurrent: sceneIsCurrent,
            failureReason: json["failure_reason"] as? String,
            startedAt: json["started_at"] as? Double,
            readyAt: json["ready_at"] as? Double,
            loadingSeconds: json["loading_seconds"] as? Double,
            // Defaulted to `false` rather than refused: an overdue flag that
            // failed to decode must not become an overdue load on screen, and
            // "not overdue" is the weaker of the two claims.
            loadOverdue: json["load_overdue"] as? Bool ?? false,
            loadOverdueAfterSeconds: json["load_overdue_after_seconds"] as? Double ?? 120,
            followsStream: json["follows_stream"] as? Bool ?? false
        )
    }

    /// The four silences, from the state that produced them.
    ///
    /// Derived from `lifecycle.state` rather than by matching the Tower's prose.
    /// The prose is shown to the person; matching on it would make a copy edit
    /// on the Tower a behaviour change on the phone.
    static func unavailableReason(
        lifecycle: SceneLifecycle, text: String?
    ) -> SceneUnavailableReason {
        switch lifecycle.state {
        case .stopped: return .stopped
        case .starting: return .stillLoading
        case .failed:
            return .failed(
                lifecycle.failureReason ?? text ?? "The Tower did not say why."
            )
        case .running: return .runningButNoFrameYet
        // A paused session with no scene: it was paused before it ever observed
        // one. Not a fifth silence — the same "no frame yet", reached from a
        // state that is no longer asking for frames.
        case .paused: return .runningButNoFrameYet
        case .unrecognised(let word): return .unrecognised(text ?? word)
        }
    }

    // MARK: The scene

    /// `where` → one row per non-person label, in the class list's order.
    ///
    /// Labels absent from `where` are skipped rather than zero-filled: the
    /// Tower publishes every non-person class with four zeroed buckets, so a
    /// label that is missing here is a payload that disagrees with its own
    /// `reported_classes`, and inventing zeros for it would hide that.
    static func positions(
        from json: [String: Any], ordering: [String]
    ) -> [SceneLabelPositions] {
        ordering.compactMap { label in
            guard let raw = json[label] as? [String: Any] else { return nil }
            return SceneLabelPositions(
                label: label,
                sides: SceneSideCounts(
                    left: raw["left"] as? Int ?? 0,
                    centre: raw["centre"] as? Int ?? 0,
                    right: raw["right"] as? Int ?? 0,
                    unknown: raw["unknown"] as? Int ?? 0
                )
            )
        }
    }

    /// `people` → the aggregate.
    ///
    /// `facing_wearer` is read with `as? Int` and stays `nil` when the wire says
    /// `null`. **There is no `?? 0` on that line and there must never be one:**
    /// zero is an answer and "never measured" is not, and this is the field on
    /// this payload where the difference is most likely to be lost.
    static func people(from json: [String: Any]) -> ScenePeople? {
        guard let count = json["count"] as? Int else { return nil }
        return ScenePeople(
            count: count,
            // Defaults chosen so a failure to decode understates the count's
            // reliability rather than overstating it: "may include the wearer"
            // and "not validated" are the cautious readings.
            mayIncludeWearer: json["may_include_wearer"] as? Bool ?? true,
            validated: json["validated"] as? Bool ?? false,
            facingWearer: json["facing_wearer"] as? Int,
            facingAnswered: json["facing_answered"] as? Bool ?? false,
            facingUnavailableReason: json["facing_unavailable_reason"] as? String,
            facingUnknown: json["facing_unknown"] as? Int,
            oldestEstimateSeconds: json["oldest_estimate_seconds"] as? Double,
            facingStatesReported: json["facing_states_reported"] as? [String] ?? [],
            facingStatesWithheld: json["facing_states_withheld"] as? [String] ?? [],
            facingStatesWithheldReason: json["facing_states_withheld_reason"] as? String,
            facingNote: json["facing_note"] as? String,
            // 2026-09-07 additions, every one optional: an older Tower omits
            // them and the defaults add nothing.
            partialBottomEdge: json["partial_bottom_edge"] as? Int ?? 0,
            byApparentSize: json["by_apparent_size"] as? [String: Int] ?? [:],
            apparentSizeNote: json["apparent_size_note"] as? String,
            orientationStatus: json["orientation_status"] as? String
        )
    }

    // MARK: Disclosure

    static func limitations(from raw: Any?) -> [SceneCountLimitation] {
        guard let entries = raw as? [[String: Any]] else { return [] }
        var result: [SceneCountLimitation] = []
        for entry in entries {
            guard
                let slug = entry["limitation"] as? String,
                let detail = entry["detail"] as? String
            else { continue }
            result.append(SceneCountLimitation(slug: slug, detail: detail))
        }
        return result
    }

    static func measurement(from json: [String: Any]?) -> SceneCountMeasurement {
        let json = json ?? [:]
        return SceneCountMeasurement(
            measuredAt: json["measured_at"] as? String,
            corpusFrames: json["corpus_frames"] as? Int,
            corpusCaptures: json["corpus_captures"] as? Int,
            // `false` when absent, which is the claim the Tower makes: these
            // figures describe the frames they were measured on and have not
            // been re-derived. Defaulting to `true` would assert currency
            // nobody has established.
            isCurrent: json["is_current"] as? Bool ?? false,
            note: json["note"] as? String
        )
    }

    static func refusals(from payload: [String: Any]) -> SceneRefusals {
        var fields: [SceneRefusedField] = []
        for entry in payload["refused_entity_fields"] as? [[String: Any]] ?? [] {
            guard
                let field = entry["field"] as? String,
                let reason = entry["reason"] as? String
            else { continue }
            fields.append(SceneRefusedField(field: field, reason: reason))
        }

        var relations: [SceneRefusedRelation] = []
        for entry in payload["refused_relations"] as? [[String: Any]] ?? [] {
            guard
                let relation = entry["relation"] as? String,
                let reason = entry["reason"] as? String
            else { continue }
            relations.append(
                SceneRefusedRelation(
                    relation: relation,
                    reason: reason,
                    reasonSource: entry["reason_source"] as? String
                )
            )
        }

        return SceneRefusals(
            tracksAbsentReason: payload["tracks_absent_reason"] as? String,
            refusedEntityFields: fields,
            confidenceAbsentReason: payload["confidence_absent_reason"] as? String,
            relationsAbsentReason: payload["relations_absent_reason"] as? String,
            withheldRelations: payload["withheld_relations"] as? [String] ?? [],
            refusedRelations: relations,
            whereExcludes: payload["where_excludes"] as? [String] ?? [],
            whereExcludesReason: payload["where_excludes_reason"] as? String
        )
    }
}
