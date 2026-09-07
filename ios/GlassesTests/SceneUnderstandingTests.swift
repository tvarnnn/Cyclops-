//
//  SceneUnderstandingTests.swift
//  GlassesTests
//

import Combine
import XCTest

@testable import Glasses

// MARK: - Fixtures

/// Payloads curled off a **running Tower** on 2026-08-27, with both cartridges
/// enabled, and transcribed as inline dictionaries — the convention this suite
/// already uses, rather than JSON resources.
///
/// ## Which of these are the wire's own bytes, and which are assembled
///
/// `stopped` is verbatim: `GET /scene` on a Tower whose Scene session had been
/// started and then stopped. Every constant, every limitation slug, every
/// refusal reason and the whole `side_convention` sentence are the Tower's.
///
/// `running` is **assembled** from that same payload plus the scene block
/// `tower/results/scene_understanding.py` produces — `counts` over
/// `REPORTED_CLASSES`, `_side_counts`, `_people_block`. It has to be: `torch`
/// is not installed on this host, so `POST /scene/start` reaches `failed` with
/// *"the engine could not be loaded: ModuleNotFoundError: No module named
/// 'torch'"* and no live payload with counts in it exists here. The shape is
/// taken from the Tower's own builder rather than invented, and the field is
/// flagged here so nobody later mistakes it for something that was observed.
///
/// The `failed` fixture's `failure_reason` **is** real: it is the sentence this
/// host produced.
enum SceneFixtures {
    /// `GET /scene`, verbatim, on a stopped session.
    static let stopped: [String: Any] = [
        "claim": "visible-now-not-a-record",
        "identity": "anonymous-and-unpublished",
        "absence_means": "not-visible-to-this-cartridge",
        "persistence": "none",
        "frame_of_reference": "camera",
        "time_basis": "tower-receipt",
        "lifecycle": [
            "state": "stopped",
            "states": [
                "stopped",
                "starting",
                "running",
                "paused",
                "failed",
            ],
            "session_id": 4,
            "scene_is_current": false,
            "failure_reason": "the engine could not be loaded: ModuleNotFoundError: No module named 'torch'",
            "started_at": 1787833054.9303071,
            "ready_at": NSNull(),
            "loading_seconds": 325.5497679710388,
            "load_overdue": false,
            "load_overdue_after_seconds": 120.0,
            "follows_stream": true,
        ],
        "observed_at": NSNull(),
        "observed_at_note": "tower-receipt time: when this Tower received the frame this scene came from, never when the glasses captured it, and not when the detector finished with it. There is no capture timestamp anywhere on this wire",
        "staleness_seconds": NSNull(),
        "frames_offered": 0,
        "frames_observed": 0,
        "frames_skipped": 0,
        "frames_dropped_not_running": 0,
        "decode_failures": 0,
        "detector": NSNull(),
        "reported_classes": [
            "bed",
            "book",
            "bottle",
            "cell phone",
            "chair",
            "couch",
            "cup",
            "dining table",
            "keyboard",
            "laptop",
            "mouse",
            "person",
            "tv",
        ],
        "count_basis": "confirmed-tracks",
        "count_is_lower_bound": true,
        "count_limitations": [
            [
                "limitation": "size-floor",
                "detail": "the detector is effectively blind below ~2% of frame area: recall 0.000 under 1% and 0.009 at 1-2%, measured over 14,128 real frames against a fasterrcnn_resnet50_fpn_v2 oracle",
            ],
            [
                "limitation": "recall",
                "detail": "class recall against the same oracle: 0.306 person, 0.730 laptop, 0.497 cell phone, 0.388 chair, 0.209 tv, 0.108 couch. The oracle shares COCO training data with the shipped model, so every one is an upper bound, not an estimate. The two worst are furniture -- couch 0.108 and chair 0.161 by the stricter of the two measures -- and both are reported by this payload, so 0.209 is not the floor",
            ],
            [
                "limitation": "noise-classes",
                "detail": "chair appeared in 4 of 340 sampled corpus frames and dining table once in 9,199. Their counts and positions are published as detector output, not as evidence that a chair was there. The wire-path design excluded them for exactly this reason; they are published with the disclosure instead, because a class silently absent from `reported_classes` would be indistinguishable from one that was looked for and not seen",
            ],
            [
                "limitation": "departure-lag",
                "detail": "a confirmed track keeps being counted for up to 12 further OBSERVED frames after its last detection -- 1.0 s at the measured 12.0 fps delivery, and longer whenever frames_skipped is advancing, because the bound is a frame count and not a duration. A count can therefore include someone who has already left. Confirmation is latched on purpose: it is what stops the count flickering when the detector drops a frame",
            ],
            [
                "limitation": "field-of-view",
                "detail": "a count is about the camera's forward cone at this instant, never about the room. Most of a room is behind the wearer",
            ],
        ],
        "count_measurement": [
            "measured_at": "2026-08-26",
            "corpus_frames": 14128,
            "corpus_captures": 28,
            "is_current": false,
            "note": "the corpus on this host has grown since. These figures describe the frames they were measured on and have not been re-derived",
        ],
        "confidence": NSNull(),
        "confidence_absent_reason": "the detector emits a per-detection score, but the tracker never uses it for confirmation and this payload publishes no per-entity row for one to attach to. A confidence on a count would be an average of scores that did not decide anything. score_threshold is the floor those scores had to clear",
        "tracks": NSNull(),
        "tracks_absent_reason": "this cartridge publishes no per-entity list and no track handle, not even a session-scoped one. A handle plus a timestamp lets a recipient assemble the per-person dwell timeline this cartridge refuses to keep -- persists-nothing laundered onto the consumer. Counts and aggregate facing are the only representation offered, and there is no key below this one that could hold an entity",
        "refused_entity_fields": [
            [
                "field": "track_id",
                "reason": "joinable across time within a session",
            ],
            [
                "field": "box",
                "reason": "a repeated position is a movement trace",
            ],
            [
                "field": "facing",
                "reason": "per-person orientation is per-person state",
            ],
            [
                "field": "visible_eyes",
                "reason": "facial-landmark evidence does not cross this boundary",
            ],
            [
                "field": "confidence",
                "reason": "requires a per-entity row, which does not exist here",
            ],
        ],
        "side_convention": "the wearer's own left and right, as the camera sees them. A track is 'left' when its box centre falls below 0.45 of frame width in the frame as received, 'right' above 0.55, 'centre' between the two, and 'unknown' when the frame size was never learned. The stream is assumed unmirrored and nothing on this wire verifies that. It is camera-relative and changes when the wearer turns their head",
        "relations": NSNull(),
        "relations_absent_reason": "this cartridge asserts no relations on the wire. Three are computable from 2-D boxes -- left_of, right_of and higher_in_view -- and all three are withheld rather than refused: they are true, they are camera-relative, and they change the moment the wearer turns their head, so a client that cached one would be holding a claim about a view that no longer exists. Every relation worth having needs depth that survives motion, and that was measured and refused. See refused_relations",
        "withheld_relations": [
            "left_of",
            "right_of",
            "higher_in_view",
        ],
        "refused_relations": [
            [
                "relation": "behind",
                "reason": "the inverse of in_front_of, and blocked by the same measurement: an ordering that holds in a still scene and degrades to 11.5% reversals under the little motion this corpus contains.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
            [
                "relation": "in_front_of",
                "reason": "needs depth that survives motion, and MiDaS does not.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
            [
                "relation": "inside",
                "reason": "same as `on` -- 2-D containment cannot distinguish it.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
            [
                "relation": "near",
                "reason": "image proximity is not world proximity.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
            [
                "relation": "nearer_than_same_class",
                "reason": "SHIPPED, THEN WITHDRAWN.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
            [
                "relation": "on",
                "reason": "needs support-surface reasoning and depth.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
        ],
        "where_excludes": [
            "person",
        ],
        "where_excludes_reason": "a per-person position, sampled repeatedly, is a movement trace. This cartridge keeps none and will not hand a client the parts to assemble one",
        "scene_available": false,
        "scene_unavailable_reason": "this session is stopped. Nothing is being observed, and the last scene was discarded rather than kept: a scene held past the end of a session is a claim about a room the wearer has left",
        "score_threshold": NSNull(),
        "counts": NSNull(),
        "where": NSNull(),
        "people": NSNull(),
        "contract": "scene_understanding.live/2026-08-27",
    ]

    /// The same payload with the scene block the Tower's own builder produces
    /// (`counts` over `REPORTED_CLASSES`, `_side_counts`, `_people_block`).
    /// Assembled rather than observed — see the type note.
    static let running: [String: Any] = [
        "claim": "visible-now-not-a-record",
        "identity": "anonymous-and-unpublished",
        "absence_means": "not-visible-to-this-cartridge",
        "persistence": "none",
        "frame_of_reference": "camera",
        "time_basis": "tower-receipt",
        "lifecycle": [
            "state": "running",
            "states": [
                "stopped",
                "starting",
                "running",
                "paused",
                "failed",
            ],
            "session_id": 7,
            "scene_is_current": true,
            "failure_reason": "the engine could not be loaded: ModuleNotFoundError: No module named 'torch'",
            "started_at": 1787833000.0,
            "ready_at": 1787833005.0,
            "loading_seconds": NSNull(),
            "load_overdue": false,
            "load_overdue_after_seconds": 120.0,
            "follows_stream": true,
        ],
        "observed_at": 1787833100.5,
        "observed_at_note": "tower-receipt time: when this Tower received the frame this scene came from, never when the glasses captured it, and not when the detector finished with it. There is no capture timestamp anywhere on this wire",
        "staleness_seconds": 0.4,
        "frames_offered": 120,
        "frames_observed": 118,
        "frames_skipped": 2,
        "frames_dropped_not_running": 0,
        "decode_failures": 0,
        "detector": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "reported_classes": [
            "bed",
            "book",
            "bottle",
            "cell phone",
            "chair",
            "couch",
            "cup",
            "dining table",
            "keyboard",
            "laptop",
            "mouse",
            "person",
            "tv",
        ],
        "count_basis": "confirmed-tracks",
        "count_is_lower_bound": true,
        "count_limitations": [
            [
                "limitation": "size-floor",
                "detail": "the detector is effectively blind below ~2% of frame area: recall 0.000 under 1% and 0.009 at 1-2%, measured over 14,128 real frames against a fasterrcnn_resnet50_fpn_v2 oracle",
            ],
            [
                "limitation": "recall",
                "detail": "class recall against the same oracle: 0.306 person, 0.730 laptop, 0.497 cell phone, 0.388 chair, 0.209 tv, 0.108 couch. The oracle shares COCO training data with the shipped model, so every one is an upper bound, not an estimate. The two worst are furniture -- couch 0.108 and chair 0.161 by the stricter of the two measures -- and both are reported by this payload, so 0.209 is not the floor",
            ],
            [
                "limitation": "noise-classes",
                "detail": "chair appeared in 4 of 340 sampled corpus frames and dining table once in 9,199. Their counts and positions are published as detector output, not as evidence that a chair was there. The wire-path design excluded them for exactly this reason; they are published with the disclosure instead, because a class silently absent from `reported_classes` would be indistinguishable from one that was looked for and not seen",
            ],
            [
                "limitation": "departure-lag",
                "detail": "a confirmed track keeps being counted for up to 12 further OBSERVED frames after its last detection -- 1.0 s at the measured 12.0 fps delivery, and longer whenever frames_skipped is advancing, because the bound is a frame count and not a duration. A count can therefore include someone who has already left. Confirmation is latched on purpose: it is what stops the count flickering when the detector drops a frame",
            ],
            [
                "limitation": "field-of-view",
                "detail": "a count is about the camera's forward cone at this instant, never about the room. Most of a room is behind the wearer",
            ],
        ],
        "count_measurement": [
            "measured_at": "2026-08-26",
            "corpus_frames": 14128,
            "corpus_captures": 28,
            "is_current": false,
            "note": "the corpus on this host has grown since. These figures describe the frames they were measured on and have not been re-derived",
        ],
        "confidence": NSNull(),
        "confidence_absent_reason": "the detector emits a per-detection score, but the tracker never uses it for confirmation and this payload publishes no per-entity row for one to attach to. A confidence on a count would be an average of scores that did not decide anything. score_threshold is the floor those scores had to clear",
        "tracks": NSNull(),
        "tracks_absent_reason": "this cartridge publishes no per-entity list and no track handle, not even a session-scoped one. A handle plus a timestamp lets a recipient assemble the per-person dwell timeline this cartridge refuses to keep -- persists-nothing laundered onto the consumer. Counts and aggregate facing are the only representation offered, and there is no key below this one that could hold an entity",
        "refused_entity_fields": [
            [
                "field": "track_id",
                "reason": "joinable across time within a session",
            ],
            [
                "field": "box",
                "reason": "a repeated position is a movement trace",
            ],
            [
                "field": "facing",
                "reason": "per-person orientation is per-person state",
            ],
            [
                "field": "visible_eyes",
                "reason": "facial-landmark evidence does not cross this boundary",
            ],
            [
                "field": "confidence",
                "reason": "requires a per-entity row, which does not exist here",
            ],
        ],
        "side_convention": "the wearer's own left and right, as the camera sees them. A track is 'left' when its box centre falls below 0.45 of frame width in the frame as received, 'right' above 0.55, 'centre' between the two, and 'unknown' when the frame size was never learned. The stream is assumed unmirrored and nothing on this wire verifies that. It is camera-relative and changes when the wearer turns their head",
        "relations": NSNull(),
        "relations_absent_reason": "this cartridge asserts no relations on the wire. Three are computable from 2-D boxes -- left_of, right_of and higher_in_view -- and all three are withheld rather than refused: they are true, they are camera-relative, and they change the moment the wearer turns their head, so a client that cached one would be holding a claim about a view that no longer exists. Every relation worth having needs depth that survives motion, and that was measured and refused. See refused_relations",
        "withheld_relations": [
            "left_of",
            "right_of",
            "higher_in_view",
        ],
        "refused_relations": [
            [
                "relation": "behind",
                "reason": "the inverse of in_front_of, and blocked by the same measurement: an ordering that holds in a still scene and degrades to 11.5% reversals under the little motion this corpus contains.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
            [
                "relation": "in_front_of",
                "reason": "needs depth that survives motion, and MiDaS does not.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
            [
                "relation": "inside",
                "reason": "same as `on` -- 2-D containment cannot distinguish it.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
            [
                "relation": "near",
                "reason": "image proximity is not world proximity.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
            [
                "relation": "nearer_than_same_class",
                "reason": "SHIPPED, THEN WITHDRAWN.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
            [
                "relation": "on",
                "reason": "needs support-surface reasoning and depth.",
                "reason_source": "tower/scene/state.py: REFUSED_RELATIONSHIPS, and docs/superpowers/research/2026-08-26-depth-ordering-on-real-frames.md",
            ],
        ],
        "where_excludes": [
            "person",
        ],
        "where_excludes_reason": "a per-person position, sampled repeatedly, is a movement trace. This cartridge keeps none and will not hand a client the parts to assemble one",
        "scene_available": true,
        "scene_unavailable_reason": NSNull(),
        "score_threshold": 0.5,
        "counts": [
            "bed": 0,
            "book": 0,
            "bottle": 0,
            "cell phone": 0,
            "chair": 1,
            "couch": 0,
            "cup": 0,
            "dining table": 0,
            "keyboard": 0,
            "laptop": 1,
            "mouse": 0,
            "person": 2,
            "tv": 0,
        ],
        "where": [
            "bed": [
                "left": 0,
                "centre": 0,
                "right": 0,
                "unknown": 0,
            ],
            "book": [
                "left": 0,
                "centre": 0,
                "right": 0,
                "unknown": 0,
            ],
            "bottle": [
                "left": 0,
                "centre": 0,
                "right": 0,
                "unknown": 0,
            ],
            "cell phone": [
                "left": 0,
                "centre": 0,
                "right": 0,
                "unknown": 0,
            ],
            "chair": [
                "left": 1,
                "centre": 0,
                "right": 0,
                "unknown": 0,
            ],
            "couch": [
                "left": 0,
                "centre": 0,
                "right": 0,
                "unknown": 0,
            ],
            "cup": [
                "left": 0,
                "centre": 0,
                "right": 0,
                "unknown": 0,
            ],
            "dining table": [
                "left": 0,
                "centre": 0,
                "right": 0,
                "unknown": 0,
            ],
            "keyboard": [
                "left": 0,
                "centre": 0,
                "right": 0,
                "unknown": 0,
            ],
            "laptop": [
                "left": 0,
                "centre": 1,
                "right": 0,
                "unknown": 0,
            ],
            "mouse": [
                "left": 0,
                "centre": 0,
                "right": 0,
                "unknown": 0,
            ],
            "tv": [
                "left": 0,
                "centre": 0,
                "right": 0,
                "unknown": 0,
            ],
        ],
        "people": [
            "count": 2,
            "may_include_wearer": true,
            "validated": false,
            "facing_wearer": NSNull(),
            "facing_answered": false,
            "facing_unavailable_reason": "coarse orientation has never produced an estimate on this session -- either no pose estimator is configured, or the model has not once succeeded. Reporting 0 would be an observation gap presented as an observation of absence",
            "facing_unknown": NSNull(),
            "oldest_estimate_seconds": NSNull(),
            "facing_states_reported": [
                "facing_wearer",
                "unknown",
            ],
            "facing_states_withheld": [
                "away_from_wearer",
                "profile",
            ],
            "facing_states_withheld_reason": "a per-person facing state is per-person state. Publishing the full enum as counts would narrow to one person's orientation the moment only one person is in view",
            "facing_note": "coarse head and body orientation relative to the camera. Render as 'facing your direction'; there is no eye tracking on this platform, so it cannot establish what anyone was looking at or whether they noticed the wearer",
        ],
        "contract": "scene_understanding.live/2026-08-27",
    ]

    // MARK: The other four lifecycle states

    /// Running, and the first frame has not come back yet. The Tower's own
    /// sentence for it, and the fourth of the four unavailable reasons.
    static var runningWithNoFrameYet: [String: Any] {
        var payload = running
        payload["scene_available"] = false
        payload["scene_unavailable_reason"] =
            "the session is running but has not finished observing a frame yet. No frame has been offered, or the first is still in flight"
        payload["counts"] = NSNull()
        payload["where"] = NSNull()
        payload["people"] = NSNull()
        payload["score_threshold"] = NSNull()
        payload["observed_at"] = NSNull()
        payload["staleness_seconds"] = NSNull()
        return payload
    }

    /// The scene the Tower kept when the session was **paused**: the same
    /// counts, `scene_is_current` false, and an age.
    static var paused: [String: Any] {
        var payload = running
        payload["lifecycle"] = lifecycle(of: running, state: "paused", sceneIsCurrent: false)
        payload["staleness_seconds"] = 42.0
        return payload
    }

    /// Running, a frame observed, and **none of the thirteen confirmed in it.**
    /// The fifth case, and the only one of the five that is about a room.
    static var lookedAndSawNothing: [String: Any] {
        var payload = running
        let classes = running["reported_classes"] as! [String]
        payload["counts"] = Dictionary(uniqueKeysWithValues: classes.map { ($0, 0) })
        payload["where"] = Dictionary(
            uniqueKeysWithValues: classes.filter { $0 != "person" }.map {
                ($0, ["left": 0, "centre": 0, "right": 0, "unknown": 0])
            }
        )
        var people = running["people"] as! [String: Any]
        people["count"] = 0
        payload["people"] = people
        return payload
    }

    static var starting: [String: Any] {
        var payload = stopped
        payload["lifecycle"] = lifecycle(of: stopped, state: "starting", sceneIsCurrent: false)
        payload["scene_unavailable_reason"] =
            "the detector is still loading. This is not an empty room; it is a Tower that has not looked yet"
        return payload
    }

    /// The reason is the one this host actually produced.
    static var failed: [String: Any] {
        var payload = stopped
        var block = lifecycle(of: stopped, state: "failed", sceneIsCurrent: false)
        block["failure_reason"] =
            "the engine could not be loaded: ModuleNotFoundError: No module named 'torch'"
        payload["lifecycle"] = block
        payload["scene_unavailable_reason"] =
            "the engine could not be loaded: ModuleNotFoundError: No module named 'torch'"
        return payload
    }

    static func lifecycle(
        of payload: [String: Any], state: String, sceneIsCurrent: Bool
    ) -> [String: Any] {
        var block = payload["lifecycle"] as! [String: Any]
        block["state"] = state
        block["scene_is_current"] = sceneIsCurrent
        return block
    }
}

// MARK: - Decoding

/// The payload, read the way the contract says it must be read.
@MainActor
final class SceneUnderstandingDecodingTests: XCTestCase {

    func testTheConstantSelfDescriptionIsAssertedRatherThanTrusted() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.stopped))
        XCTAssertEqual(reading.claim, "visible-now-not-a-record")
        XCTAssertEqual(reading.identity, "anonymous-and-unpublished")
        XCTAssertEqual(reading.absenceMeans, "not-visible-to-this-cartridge")
        XCTAssertEqual(reading.persistence, "none")
        XCTAssertEqual(reading.frameOfReference, "camera")
        XCTAssertEqual(reading.timeBasis, "tower-receipt")

        // A Tower that changed one of these while keeping the identifier would
        // be making a different promise under the same name. Absorbing that
        // silently is how a client comes to render a record as a live view.
        var mangled = SceneFixtures.stopped
        mangled["claim"] = "a-record"
        XCTAssertNil(SceneUnderstandingDecoder.reading(from: mangled))
        mangled = SceneFixtures.stopped
        mangled["persistence"] = "session"
        XCTAssertNil(SceneUnderstandingDecoder.reading(from: mangled))
    }

    /// The result type is `live`, not `status`, and this is the only cartridge
    /// where it differs. A `result_subscribe` naming the wrong one is refused
    /// with `unknown_result_type`, so the constant is load-bearing.
    func testTheResultTypeIsLiveAndNotStatus() {
        XCTAssertEqual(SceneUnderstandingContract.resultType, "live")
        XCTAssertNotEqual(SceneUnderstandingContract.resultType, "status")
        XCTAssertEqual(
            SceneUnderstandingContract.identifier, "scene_understanding.live/2026-08-27"
        )
    }

    /// `counts` has one entry per reported class, **present at 0 rather than
    /// omitted.** A class silently absent would be indistinguishable from one
    /// that was looked for and not seen.
    func testEveryReportedClassHasACountEvenWhenItIsZero() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.running))
        let observation = try XCTUnwrap(reading.observation)
        XCTAssertEqual(reading.reportedClasses.count, 13)
        XCTAssertEqual(observation.counts.count, 13)
        // Ordered by `reported_classes`, which is fixed at build time, so a
        // reader's eye stays in the same place between two readings.
        XCTAssertEqual(observation.counts.map(\.label), reading.reportedClasses)
        XCTAssertEqual(observation.counts.first { $0.label == "bed" }?.count, 0)
        XCTAssertEqual(observation.present.map(\.label).sorted(), ["chair", "laptop", "person"])
    }

    /// `where` carries side **counts** per label, for non-person labels only.
    ///
    /// One side cannot describe a chair on the left and a chair on the right,
    /// and `unknown` is its own bucket rather than being folded into `centre`:
    /// a scene whose frame size was never learned has not placed anything in
    /// the middle of the view, it has placed nothing.
    func testWhereExcludesPeopleAndCountsSidesRatherThanPickingOne() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.running))
        let observation = try XCTUnwrap(reading.observation)
        XCTAssertEqual(reading.refusals.whereExcludes, ["person"])
        XCTAssertNil(observation.positions.first { $0.label == "person" })
        XCTAssertEqual(observation.positions.count, 12)
        XCTAssertEqual(observation.positions.first { $0.label == "chair" }?.sides.left, 1)
        XCTAssertNotNil(reading.refusals.whereExcludesReason)
        // The convention is declared on the payload precisely because a left
        // and a right with no stated convention is a silent presumption, and a
        // Tower signing the other way would put everything on the wrong side.
        XCTAssertEqual(reading.sideConvention?.contains("0.45"), true)
    }

    /// **`facing_wearer` is null, never 0, when unmeasured** — the field on
    /// this payload where the difference is most likely to be lost.
    func testFacingWearerIsNullRatherThanZeroWhenNothingMeasuredIt() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.running))
        let people = try XCTUnwrap(reading.observation?.people)
        XCTAssertNil(people.facingWearer, "an observation gap became an observation of absence")
        XCTAssertFalse(people.facingAnswered)
        XCTAssertNotNil(people.facingUnavailableReason)

        // And 0 stays 0 when it is a measurement.
        var measured = SceneFixtures.running
        var block = measured["people"] as! [String: Any]
        block["facing_wearer"] = 0
        block["facing_answered"] = true
        block["facing_unavailable_reason"] = NSNull()
        measured["people"] = block
        let answered = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: measured))
        XCTAssertEqual(answered.observation?.people.facingWearer, 0)
        XCTAssertEqual(answered.observation?.people.facingAnswered, true)
    }

    /// People are a count and an aggregate. The three qualifications on that
    /// count travel with it.
    func testPeopleAreACountAndAnAggregateNeverAList() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.running))
        let people = try XCTUnwrap(reading.observation?.people)
        XCTAssertEqual(people.count, 2)
        // Every `person` box in this platform's only real corpus is the
        // wearer's own torso. A count rendered without this reads as a count
        // of other people.
        XCTAssertTrue(people.mayIncludeWearer)
        XCTAssertFalse(people.validated)
        XCTAssertEqual(people.facingStatesReported, ["facing_wearer", "unknown"])
        XCTAssertEqual(people.facingStatesWithheld, ["away_from_wearer", "profile"])
    }

    /// `scene_available: true` with a null scene block is not half a scene to
    /// be rendered carefully. It is a payload this build does not understand.
    func testAPayloadThatDisagreesWithItselfIsRefused() {
        for key in ["counts", "where", "people"] {
            var half = SceneFixtures.running
            half[key] = NSNull()
            XCTAssertNil(
                SceneUnderstandingDecoder.reading(from: half),
                "a null \(key) beside scene_available:true was decoded anyway"
            )
        }
    }

    /// Times on this wire are the **Tower's receipt clock**, and this app's
    /// `ObservationTime.observedAt` is documented as when the glasses observed
    /// something. Mapping one onto the other by field name is exactly the
    /// substitution Core Principle 5 forbids, so the reading keeps its own
    /// field and carries the Tower's sentence about what it is.
    func testTheTimestampIsNotPresentedAsACaptureClock() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.running))
        XCTAssertNotNil(reading.observedAtTowerReceipt)
        XCTAssertEqual(reading.observedAtNote?.contains("never when the glasses captured it"), true)
        XCTAssertEqual(reading.timeBasis, "tower-receipt")
    }
}

// MARK: - The five silences

/// `scene_available: false` in **four** distinct situations, plus a fifth that
/// is `scene_available: true` with every count at zero.
///
/// A client that flattened them would show an empty room for all five, when
/// only the last is about a room. These five assertions are the whole reason
/// `SceneSilence` exists as a type.
@MainActor
final class SceneSilenceTests: XCTestCase {

    private func silence(_ payload: [String: Any]) throws -> SceneSilence? {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: payload))
        return SceneUnderstandingState.forReading(reading).silence
    }

    /// A cleanly stopped session: the bench fixture with its failure cleared.
    ///
    /// `SceneFixtures.stopped` is verbatim bytes from a Tower whose engine
    /// could not load, so it carries a `failure_reason` **through** the stop --
    /// the Tower clears that only when a new session begins. That makes it the
    /// wrong fixture for "somebody pressed stop", which is what `.stopped` is
    /// for, and the right fixture for the case below it.
    private func cleanlyStopped() -> [String: Any] {
        var payload = SceneFixtures.stopped
        var lifecycle = payload["lifecycle"] as! [String: Any]
        lifecycle["failure_reason"] = NSNull()
        payload["lifecycle"] = lifecycle
        return payload
    }

    func testTheFiveSilencesAreToldApart() throws {
        XCTAssertEqual(try silence(cleanlyStopped()), .stopped)
        // The same state, from the real bench, is NOT the same sentence. A
        // stopped session carrying a failure did not stop because anybody
        // asked, and "the last scene was discarded" is a comforting story about
        // a dead engine.
        XCTAssertEqual(
            try silence(SceneFixtures.stopped),
            .towerFailed("the engine could not be loaded: ModuleNotFoundError: No module named 'torch'"),
            "a stopped-because-it-failed session read as an ordinary stop"
        )
        XCTAssertEqual(try silence(SceneFixtures.starting), .stillLoading)
        XCTAssertEqual(
            try silence(SceneFixtures.failed),
            .towerFailed("the engine could not be loaded: ModuleNotFoundError: No module named 'torch'")
        )
        XCTAssertEqual(try silence(SceneFixtures.runningWithNoFrameYet), .runningButNoFrameYet)
        XCTAssertEqual(try silence(SceneFixtures.lookedAndSawNothing), .lookedAndSawNothing)
        // And a populated scene is not a silence at all.
        XCTAssertNil(try silence(SceneFixtures.running))
    }

    /// Five headlines and five explanations. Any two that matched would undo
    /// the distinction the type exists to draw.
    func testEachSilenceSaysSomethingDifferent() {
        let all: [SceneSilence] = [
            .stopped, .stillLoading, .towerFailed("x"), .runningButNoFrameYet, .lookedAndSawNothing,
        ]
        XCTAssertEqual(Set(all.map(\.headline)).count, all.count)
        XCTAssertEqual(Set(all.map(\.explanation)).count, all.count)
    }

    /// The one that is about a room says the count is a floor, because "none
    /// confirmed" and "none there" are different at 0.306 recall.
    func testTheEmptyRoomCaseStillDisclosesTheLowerBound() {
        let explanation = SceneSilence.lookedAndSawNothing.explanation.lowercased()
        XCTAssertTrue(explanation.contains("floor"), "got: \(explanation)")
        XCTAssertTrue(explanation.contains("not about the room"), "got: \(explanation)")
    }

    /// A reason this build has never heard of reaches the screen as the
    /// Tower's own prose rather than being assigned one of the four headlines,
    /// which would state the wrong cause with full confidence.
    func testAnUnrecognisedStateIsNotGivenOneOfTheFourHeadlines() throws {
        var odd = SceneFixtures.stopped
        odd["lifecycle"] = SceneFixtures.lifecycle(
            of: SceneFixtures.stopped, state: "quiescing", sceneIsCurrent: false
        )
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: odd))
        XCTAssertEqual(reading.lifecycle.state, .unrecognised("quiescing"))
        XCTAssertNil(SceneUnderstandingState.forReading(reading).silence)
        // It still lands somewhere safe: nothing is shown, and nothing is kept.
        XCTAssertNil(SceneUnderstandingState.forReading(reading).observation)
    }
}

// MARK: - Stop discards, pause keeps

/// The rule this cartridge's whole state machine exists to enforce.
///
/// > A scene held past the end of a session is a claim about a room the wearer
/// > has left. No staleness number makes that safe, because a client that
/// > renders counts above staleness shows the room first.
@MainActor
final class SceneLifecycleTests: XCTestCase {

    func testPauseKeepsTheSceneAndMarksItNotCurrent() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.paused))
        let state = SceneUnderstandingState.forReading(reading)
        guard case .lastKnown = state else {
            return XCTFail("a paused session must reach .lastKnown, got \(state)")
        }
        XCTAssertFalse(state.isCurrent)
        XCTAssertFalse(reading.lifecycle.sceneIsCurrent)
        XCTAssertNotNil(reading.stalenessSeconds, "a last-known scene without its age is a lie")
        XCTAssertEqual(state.phase, .settled, "a paused scene is not being refined")
    }

    func testStopDiscardsTheSceneRatherThanKeepingIt() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.stopped))
        let state = SceneUnderstandingState.forReading(reading)
        guard case .idle = state else {
            return XCTFail("a stopped session must reach .idle, got \(state)")
        }
        XCTAssertNil(state.observation)
        XCTAssertFalse(state.isCurrent)
    }

    /// The violation this guards is entering `.lastKnown` on a **stop**. Even
    /// a payload that arrived with a scene in it — which this Tower does not
    /// send, and which a future one must not be trusted not to — is discarded.
    func testAStoppedPayloadCarryingASceneIsStillDiscarded() throws {
        var forced = SceneFixtures.running
        forced["lifecycle"] = SceneFixtures.lifecycle(
            of: SceneFixtures.running, state: "stopped", sceneIsCurrent: false
        )
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: forced))
        XCTAssertNotNil(reading.observation, "the fixture must actually carry a scene")
        let state = SceneUnderstandingState.forReading(reading)
        guard case .idle = state else {
            return XCTFail("a stop must discard, got \(state)")
        }
        XCTAssertNil(state.observation, "a scene survived a stop")
    }

    /// The gate is a property of the lifecycle rather than a condition somebody
    /// has to remember to write at each call site.
    func testOnlyPausedMayHoldALastKnownScene() {
        XCTAssertTrue(SceneLifecycleState.paused.mayHoldLastKnownScene)
        for state: SceneLifecycleState in [.stopped, .starting, .running, .failed, .unrecognised("x")] {
            XCTAssertFalse(
                state.mayHoldLastKnownScene,
                "\(state.wireValue) was allowed to hold a scene past its session"
            )
        }
    }

    /// `load_overdue` is not a failure: nothing can interrupt a blocking model
    /// load, and a first-run weight download is slow and still correct.
    func testAnOverdueLoadIsNotAFailure() throws {
        var overdue = SceneFixtures.starting
        var block = overdue["lifecycle"] as! [String: Any]
        block["load_overdue"] = true
        block["loading_seconds"] = 180.0
        overdue["lifecycle"] = block
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: overdue))
        XCTAssertTrue(reading.lifecycle.loadOverdue)
        XCTAssertEqual(reading.lifecycle.loadOverdueAfterSeconds, 120.0)
        guard case .awaitingFirstScene = SceneUnderstandingState.forReading(reading) else {
            return XCTFail("an overdue load was rendered as a failure")
        }
        XCTAssertFalse(SceneLifecycle.loadOverdueNote.lowercased().contains("failed"))
    }

    /// A failure is the Tower's, and it is reported with the Tower's own words.
    func testAFailedSessionCarriesTheTowersOwnReason() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.failed))
        guard case .failed(let failure) = SceneUnderstandingState.forReading(reading) else {
            return XCTFail("a failed session must reach .failed")
        }
        XCTAssertEqual(failure.kind, .towerReportedFailure)
        XCTAssertTrue(failure.message.contains("torch"))
    }
}

// MARK: - Disclosure

/// `count_is_lower_bound` is an obligation, not a flag.
///
/// > An undercount published without disclosure looks exactly like a quiet
/// > room.
@MainActor
final class SceneDisclosureTests: XCTestCase {

    func testTheLowerBoundIsTrueOnEveryPayloadIncludingEmptyOnes() throws {
        for payload in [
            SceneFixtures.stopped, SceneFixtures.starting, SceneFixtures.failed,
            SceneFixtures.running, SceneFixtures.paused, SceneFixtures.lookedAndSawNothing,
        ] {
            let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: payload))
            XCTAssertTrue(reading.countIsLowerBound)
            XCTAssertFalse(
                reading.countLimitations.isEmpty,
                "a payload arrived with no limitations attached"
            )
        }
    }

    /// The slugs the contract names, carried as data so a client can key off a
    /// class of limit rather than matching prose.
    func testTheMeasuredLimitationsArriveWithTheirSlugs() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.stopped))
        let slugs = Set(reading.countLimitations.map(\.slug))
        for expected in ["size-floor", "recall", "field-of-view", "departure-lag"] {
            XCTAssertTrue(slugs.contains(expected), "missing \(expected) from \(slugs)")
        }
        // Every one is a sentence a person can be shown, which is what makes
        // rendering them possible at all.
        XCTAssertTrue(reading.countLimitations.allSatisfy { !$0.detail.isEmpty })
    }

    /// `is_current: false`, and it is not defaulted to `true` anywhere.
    ///
    /// A rate asserted in the present tense would read as current state, and
    /// this platform's corpus grows continuously.
    func testTheMeasurementIsNotPresentedAsCurrent() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.stopped))
        XCTAssertFalse(reading.countMeasurement.isCurrent)
        XCTAssertEqual(reading.countMeasurement.corpusFrames, 14128)

        // And a payload that omits the block entirely still does not claim
        // currency.
        var bare = SceneFixtures.stopped
        bare["count_measurement"] = NSNull()
        let plain = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: bare))
        XCTAssertFalse(plain.countMeasurement.isCurrent)
    }

    /// The caveat this app writes must not be weaker than the measurement.
    func testTheCountCaveatSaysItIsAFloorAndWhy() {
        let caveat = SceneReading.countCaveat.lowercased()
        XCTAssertTrue(caveat.contains("floor"), "got: \(caveat)")
        XCTAssertTrue(caveat.contains("misses"), "got: \(caveat)")
        XCTAssertTrue(caveat.contains("behind the wearer"), "got: \(caveat)")
        // Never "present". Say "observed".
        XCTAssertFalse(caveat.contains("present in the room"))
    }
}

// MARK: - Privacy

/// The refusals, and the fact that there is nowhere to put what was refused.
@MainActor
final class ScenePrivacyTests: XCTestCase {

    /// `tracks`, `relations` and `confidence` are null with a reason and a
    /// refusal list. The refusal is delivered as a **value**, because "refused"
    /// and "not implemented yet" are different instructions and a client
    /// finding nothing to decode cannot tell them apart.
    func testTheRefusalsArriveAsValuesRatherThanAsSilence() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.running))
        XCTAssertNotNil(reading.refusals.tracksAbsentReason)
        XCTAssertNotNil(reading.refusals.relationsAbsentReason)
        XCTAssertNotNil(reading.refusals.confidenceAbsentReason)

        let refused = Set(reading.refusals.refusedEntityFields.map(\.field))
        for field in ["track_id", "box", "facing", "visible_eyes", "confidence"] {
            XCTAssertTrue(refused.contains(field), "\(field) was not named as refused")
        }
        XCTAssertEqual(
            Set(reading.refusals.withheldRelations), ["left_of", "right_of", "higher_in_view"]
        )
        XCTAssertFalse(reading.refusals.refusedRelations.isEmpty)
    }

    /// The point is joinability, not minimising disclosure — the phone sent the
    /// pixels, so a count tells a recipient strictly less than the frame this
    /// app already holds. What is genuinely new is that a stable handle plus a
    /// timestamp would let someone assemble the per-person timeline this
    /// cartridge keeps none of.
    func testTheAnonymityNoteExplainsJoinabilityRatherThanClaimingSecrecy() {
        let note = SceneRefusals.joinabilityNote.lowercased()
        XCTAssertTrue(note.contains("timeline"), "got: \(note)")
        XCTAssertTrue(note.contains("the phone sent the frames"), "got: \(note)")
        // It must not claim the app is protecting something the phone already
        // has. That would be theatre, and the contract says so.
        XCTAssertFalse(note.contains("we do not share"))
    }

    /// Orientation is body-facing, never gaze. There is no eye tracking on the
    /// target glasses, so there is no gaze to report at any confidence.
    func testOrientationIsNeverDescribedAsLooking() {
        for facing in SceneFacing.allCases {
            let label = facing.displayName.lowercased()
            for forbidden in ["looking", "watching", "eye contact", "staring", "gaze"] {
                XCTAssertFalse(label.contains(forbidden), "\(facing) says '\(label)'")
            }
        }
        XCTAssertEqual(SceneFacing.towardCamera.displayName, "Facing your direction")
        XCTAssertTrue(SceneFacing.gazeCaveat.lowercased().contains("no eye tracking"))
    }

    /// Two of the four orientation buckets are published and two are withheld,
    /// with a reason. A screen that showed the other two as `0` would report a
    /// measurement that was never taken.
    func testOnlyTwoOrientationBucketsAreReported() {
        XCTAssertTrue(SceneFacing.towardCamera.isReportedByTower)
        XCTAssertTrue(SceneFacing.unknown.isReportedByTower)
        XCTAssertFalse(SceneFacing.awayFromCamera.isReportedByTower)
        XCTAssertFalse(SceneFacing.acrossView.isReportedByTower)
        XCTAssertEqual(SceneFacing.towardCamera.wireName, "facing_wearer")
        XCTAssertEqual(SceneFacing.awayFromCamera.wireName, "away_from_wearer")
    }

    /// `count − facing_wearer − facing_unknown` is a remainder, not a fifth
    /// category, because the two states that would name it have no bucket.
    func testTheOrientationRemainderIsNotCalledFacingAway() throws {
        var measured = SceneFixtures.running
        var block = measured["people"] as! [String: Any]
        block["count"] = 5
        block["facing_wearer"] = 1
        block["facing_answered"] = true
        block["facing_unknown"] = 2
        block["facing_unavailable_reason"] = NSNull()
        measured["people"] = block
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: measured))
        XCTAssertEqual(reading.observation?.people.undifferentiatedRemainder, 2)
    }

    /// This cartridge serves no image and has no artifact fetch contract.
    /// There is nothing in the reading that could hold one.
    func testNothingInAReadingCouldCarryAnImageOrAnEntity() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.running))
        // Structural, and asserted through the only surface that exposes the
        // scene: a count, a side histogram, and a people aggregate. There is no
        // per-entity collection to be empty, which is why this test can only
        // check the shape rather than a count of rows.
        let observation = try XCTUnwrap(reading.observation)
        XCTAssertEqual(observation.counts.count, reading.reportedClasses.count)
        XCTAssertEqual(observation.positions.count, reading.reportedClasses.count - 1)
        XCTAssertEqual(observation.people.count, 2)
    }
}

// MARK: - The client

/// What the client keeps, and what it refuses to keep.
@MainActor
final class SceneUnderstandingClientTests: XCTestCase {

    /// The stub says only what is observable from here, and none of the three
    /// claims that used to be on it.
    ///
    /// The retired sentence was: *"The Tower does not analyse scenes yet. Its
    /// only reply to a frame is a single brightness measurement, so nothing
    /// about anyone the glasses pass ever reaches this app."* Every clause is
    /// now false, and it was a privacy assurance about bystanders on the one
    /// screen whose subject is bystanders.
    func testTheStubMakesNoClaimAboutWhatAnyTowerDoes() {
        let reason = UnavailableSceneUnderstandingClient.reason.lowercased()
        XCTAssertFalse(reason.contains("brightness"), "the retired sentence is still shipping")
        XCTAssertFalse(reason.contains("does not analyse scenes"))
        XCTAssertFalse(reason.contains("ever reaches this app"))
        // What it may say: this Tower declared nothing.
        XCTAssertTrue(reason.contains("declared"), "got: \(reason)")
        XCTAssertEqual(
            UnavailableSceneUnderstandingClient().state.phase, .unsupported
        )
    }

    /// The view model projects the client's state and adds nothing.
    /// Renamed, because "whatever the connection says" is no longer the
    /// behaviour and was never the right one: `.unsupported` means "this Tower
    /// will never do this" and `.disconnected` means "ask again when
    /// connected". Scene Understanding has a client and a Tower name, so on a
    /// cold launch the second is true and the first is a claim about a machine
    /// nobody has spoken to.
    func testTheViewModelBlamesTheConnectionWhenThereIsNone() {
        let scene = SceneUnderstandingViewModel(client: UnavailableSceneUnderstandingClient())
        XCTAssertEqual(scene.phase(isTowerReachable: true), .unsupported)
        XCTAssertEqual(scene.phase(isTowerReachable: false), .disconnected)
    }

    /// Every state that has no scene exposes no scene, which is the invariant
    /// `CartridgePhase.mayCarryData` exists to make checkable.
    func testNoPhaseWithoutDataCarriesData() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.running))
        let empty = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.stopped))
        let cases: [SceneUnderstandingState] = [
            .unsupported(reason: "x"),
            .idle(nil),
            .idle(empty),
            .awaitingFirstScene(empty),
            .observing(reading),
            .lastKnown(reading),
            .failed(CartridgeFailure(kind: .transport, message: "x")),
        ]
        for state in cases where !state.phase.mayCarryData {
            XCTAssertNil(state.observation, "\(state.phase) carried a scene")
        }
    }

    func testEverySceneStateMapsToTheRightPhase() throws {
        let reading = try XCTUnwrap(SceneUnderstandingDecoder.reading(from: SceneFixtures.running))
        let expected: [(SceneUnderstandingState, CartridgePhase)] = [
            (.unsupported(reason: "x"), .unsupported),
            (.idle(nil), .idle),
            (.awaitingFirstScene(reading), .waiting),
            (.observing(reading), .live),
            (.lastKnown(reading), .settled),
            (.failed(CartridgeFailure(kind: .transport, message: "x")), .failed),
        ]
        for (state, phase) in expected {
            XCTAssertEqual(state.phase, phase)
        }
    }

    /// **The phone sends nothing to open this cartridge.** The four control
    /// routes exist and belong to an operator; this app knows their names and
    /// calls none of them.
    ///
    /// Asserted as a property of the client's surface rather than of the wire,
    /// because there is no method here that could send one: `SceneUnderstandingClient`
    /// has no verb at all, which is the strongest form this can take.
    func testThisAppHasNoWayToStartOrStopASceneSession() {
        XCTAssertEqual(SceneUnderstandingContract.controlRoutes.count, 4)
        XCTAssertTrue(
            SceneUnderstandingContract.phoneSendsNothingNote.lowercased().contains("never starts")
        )
        // The protocol's whole surface: a state and a stream of states.
        let client: any SceneUnderstandingClient = UnavailableSceneUnderstandingClient()
        XCTAssertEqual(client.cartridgeID, "scene-understanding")
    }
}

// MARK: - 2026-09-07 additions

// @MainActor, like every other test class in this file. The app target
// sets SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor, so the decoder, the
// view and the client this class drives are all main-actor isolated at
// their declaration, and a test class with no annotation cannot call
// them synchronously.
@MainActor
final class SceneUnderstandingAdditionsTests: XCTestCase {
    private func people(_ extra: [String: Any]) -> ScenePeople {
        var json: [String: Any] = ["count": 2, "facing_wearer": 1, "facing_answered": true]
        json.merge(extra) { _, new in new }
        return SceneUnderstandingDecoder.people(from: json)!
    }

    func testAnOlderTowerDecodesWithNothingInvented() {
        let decoded = people([:])
        XCTAssertEqual(decoded.partialBottomEdge, 0)
        XCTAssertEqual(decoded.byApparentSize, [:])
        XCTAssertNil(decoded.orientationStatus)
        XCTAssertNil(decoded.apparentSizeNote)
    }

    func testThePartialBucketAndSizesDecode() {
        let decoded = people([
            "partial_bottom_edge": 1,
            "by_apparent_size": ["large": 1, "medium": 0, "small": 1, "unknown": 0],
            "apparent_size_note": "sizes, never distances",
            "orientation_status": "experimental",
        ])
        XCTAssertEqual(decoded.partialBottomEdge, 1)
        XCTAssertEqual(decoded.byApparentSize["large"], 1)
        XCTAssertEqual(decoded.byApparentSize["small"], 1)
        XCTAssertEqual(decoded.orientationStatus, "experimental")
        XCTAssertEqual(decoded.apparentSizeNote, "sizes, never distances")
    }

    func testPeopleAreNeverPersons() {
        XCTAssertEqual(SceneReadingView.everydayName(for: "person", count: 1), "person")
        XCTAssertEqual(SceneReadingView.everydayName(for: "person", count: 2), "people")
    }

    func testSizesTextOmitsEmptyBucketsAndSaysInView() {
        XCTAssertNil(SceneReadingView.sizesText([:]))
        XCTAssertNil(SceneReadingView.sizesText(["large": 0, "unknown": 2]))
        XCTAssertEqual(
            SceneReadingView.sizesText(["large": 1, "medium": 0, "small": 2]),
            "Size in view: 1 large, 2 small"
        )
    }

    func testTheStubClientIgnoresVisibility() {
        let client = UnavailableSceneUnderstandingClient()
        client.workspaceVisibilityChanged(isVisible: true)
        client.workspaceVisibilityChanged(isVisible: false)
        if case .unsupported = client.state {} else { XCTFail("the stub's state must not move") }
    }

    /// Visibility must reach the CONFORMER through an existential.
    ///
    /// This is the one thing about `workspaceVisibilityChanged` that a
    /// compiler cannot tell you. If the method is declared only in the
    /// protocol extension and not in the protocol's requirement list, a
    /// call through `any SceneUnderstandingClient` -- which is how
    /// `SceneUnderstandingViewModel` holds its client -- is dispatched
    /// STATICALLY to the extension's no-op default. It compiles, every
    /// other test passes, and the screen silently stops opening and
    /// closing the live subscription. Since the Tower runs a scene session
    /// only while somebody streams AND somebody watches, the detector then
    /// runs for as long as the socket lives.
    ///
    /// The lane shipped it in that state. This test fails if it returns:
    /// it calls through the existential and asserts the override ran.
    func testVisibilityReachesTheConformerThroughAnExistential() {
        final class Recorder: SceneUnderstandingClient {
            var seen: [Bool] = []
            let cartridgeID = "scene-understanding"
            let state: SceneUnderstandingState = .unsupported(reason: "a test double")
            func workspaceVisibilityChanged(isVisible: Bool) { seen.append(isVisible) }
        }

        let recorder = Recorder()
        let erased: any SceneUnderstandingClient = recorder
        erased.workspaceVisibilityChanged(isVisible: true)
        erased.workspaceVisibilityChanged(isVisible: false)

        XCTAssertEqual(
            recorder.seen,
            [true, false],
            "workspaceVisibilityChanged must be a protocol REQUIREMENT. Declared "
                + "only in the extension it dispatches statically through an "
                + "existential, the no-op default runs, and the Scene screen "
                + "never opens or closes its subscription."
        )
    }
}
