//
//  WorldBuilderIntegrationTests.swift
//  GlassesTests
//
//  The Tower-backed World Builder client, against a real local WebSocket
//  server speaking the payloads `tower/tower/results/world_builder.py`
//  actually produces. Every fixture below was written from that file and from
//  `docs/contracts/CARTRIDGE-RESULTS.md` §10, not from a summary of either.
//

import Combine
import XCTest

@testable import Glasses

// MARK: - Payload decoding

/// The decode half, with no socket in it. These are the assertions that would
/// notice a field being renamed, defaulted to zero, or quietly upgraded.
@MainActor
final class WorldBuilderPayloadTests: XCTestCase {

    /// The complete payload the Tower builds for a live session, trimmed to the
    /// two halves iOS decodes. The Tower-native evidence blocks are omitted
    /// deliberately: this build reads neither, and a test that fed them in
    /// would imply otherwise.
    private func payload(
        modelState: String = "receiving",
        reason: Any = NSNull(),
        snapshot: [String: Any]? = nil
    ) -> [String: Any] {
        [
            "model_state": modelState,
            "model_state_reason": reason,
            // `as Any?` is not decoration: `??` must unify `[String: Any]`
            // and `NSNull` into one `T`, and the only such type is `Any`.
            // Spelling it out saves the solver a backtrack it may not make.
            "world_snapshot": (snapshot as Any?) ?? NSNull(),
        ]
    }

    private func fullSnapshot(
        overrides: [String: Any] = [:]
    ) -> [String: Any] {
        var snapshot: [String: Any] = [
            "name": "Probe Room",
            "world_id": "be5076514e0d4727ab06f2ad1df5a5bf",
            "keyframe_count": 4,
            "revision": "b00bfe85819804da",
            "tracking": "good",
            "scale": "relative",
            "mapping_seconds": 0.0789,
            "calibration": "calibrated",
            "geometry": [
                "representation": "sparse point cloud",
                "element_count": 1360,
                "is_incremental": false,
            ],
            "trajectory": [
                "pose_count": 4,
                "path_length": 2.853251890377782,
                "path_length_unit": "world units",
                "scale": "relative",
            ],
            "persistence": ["state": "saved", "revision": "67ccaee79f212c8d"],
        ]
        for (key, value) in overrides { snapshot[key] = value }
        return snapshot
    }

    // MARK: The six states

    func testEveryModelStateTheTowerSendsMapsToAWorldModelState() {
        XCTAssertEqual(
            WorldBuilderResultDecoder.modelState(from: payload(modelState: "idle")),
            .idle
        )

        let receiving = WorldBuilderResultDecoder.modelState(
            from: payload(modelState: "receiving", snapshot: fullSnapshot())
        )
        XCTAssertTrue(receiving?.isReceivingUpdates ?? false)
        XCTAssertEqual(receiving?.snapshot?.keyframeCount, 4)

        guard
            case .finalizing(let finalizing, _)? = WorldBuilderResultDecoder.modelState(
                from: payload(modelState: "finalizing", snapshot: fullSnapshot())
            )
        else { return XCTFail("finalizing did not decode") }
        XCTAssertEqual(finalizing.keyframeCount, 4)

        guard
            case .finalized(let finalized)? = WorldBuilderResultDecoder.modelState(
                from: payload(modelState: "finalized", snapshot: fullSnapshot())
            )
        else { return XCTFail("finalized did not decode") }
        XCTAssertEqual(finalized.worldID, "be5076514e0d4727ab06f2ad1df5a5bf")
    }

    /// `unsupported` is the Tower saying it cannot serve World Builder at all —
    /// "do not invite the user to wait". It must carry the Tower's own words.
    func testUnsupportedCarriesTheTowersOwnExplanation() {
        guard
            case .unsupported(let reason)? = WorldBuilderResultDecoder.modelState(
                from: payload(modelState: "unsupported", reason: "no world root is configured")
            )
        else { return XCTFail("unsupported did not decode") }
        XCTAssertEqual(reason, "no world root is configured")

        // And when it explains nothing, the app still must not render a blank.
        guard
            case .unsupported(let fallback)? = WorldBuilderResultDecoder.modelState(
                from: payload(modelState: "unsupported")
            )
        else { return XCTFail("unsupported without a reason did not decode") }
        XCTAssertFalse(fallback.isEmpty)
    }

    /// A failure the Tower reported is `.towerReportedFailure` and nothing
    /// else. Attributing it to transport, or to a local refusal, would be a
    /// fabricated claim about which machine failed.
    func testAFailedSessionIsAttributedToTheTower() {
        guard
            case .failed(let failure)? = WorldBuilderResultDecoder.modelState(
                from: payload(modelState: "failed", reason: "the builder died")
            )
        else { return XCTFail("failed did not decode") }
        XCTAssertEqual(failure.kind, .towerReportedFailure)
        XCTAssertEqual(failure.message, "the builder died")
    }

    /// `awaiting_first_update` is never sent — it is a fact about the phone's
    /// own situation — and a state word this build does not know is a contract
    /// disagreement rather than an empty world.
    func testAnUnknownModelStateIsARefusalNotAnEmptyWorld() {
        XCTAssertNil(WorldBuilderResultDecoder.modelState(from: payload(modelState: "mapping")))
        XCTAssertNil(
            WorldBuilderResultDecoder.modelState(from: payload(modelState: "awaiting_first_update"))
        )
        XCTAssertNil(WorldBuilderResultDecoder.modelState(from: [:]))
    }

    /// A live session with no world yet is the one honest use of
    /// `.awaitingFirstUpdate`: frames are going out and the Tower has said
    /// nothing about a world.
    func testReceivingWithoutASnapshotIsAwaitingRatherThanAnEmptyWorld() {
        XCTAssertEqual(
            WorldBuilderResultDecoder.modelState(from: payload(modelState: "receiving")),
            .awaitingFirstUpdate
        )
    }

    // MARK: Field-level truthfulness

    func testEveryWorldSnapshotFieldSurvivesTheWire() {
        let snapshot = WorldBuilderResultDecoder.snapshot(from: fullSnapshot())

        XCTAssertEqual(snapshot.name, "Probe Room")
        XCTAssertEqual(snapshot.worldID, "be5076514e0d4727ab06f2ad1df5a5bf")
        XCTAssertEqual(snapshot.keyframeCount, 4)
        XCTAssertEqual(snapshot.revision, "b00bfe85819804da")
        XCTAssertEqual(snapshot.tracking, .good)
        XCTAssertEqual(snapshot.scale, .relative)
        XCTAssertEqual(snapshot.mappingSeconds, 0.0789)
        XCTAssertEqual(snapshot.calibration, .calibrated)
        XCTAssertEqual(snapshot.geometry.representation, "sparse point cloud")
        XCTAssertEqual(snapshot.geometry.elementCount, 1360)
        XCTAssertEqual(snapshot.geometry.isIncremental, false)
        XCTAssertEqual(snapshot.trajectory.poseCount, 4)
        XCTAssertEqual(snapshot.trajectory.pathLength, 2.853251890377782)
        XCTAssertEqual(snapshot.trajectory.pathLengthUnit, "world units")
        XCTAssertEqual(snapshot.trajectory.scale, .relative)
        XCTAssertEqual(snapshot.persistence, .saved(revision: "67ccaee79f212c8d"))
    }

    /// Null means absent, never zero. `frames_observed` being genuinely
    /// unknowable during a live session is why the contract says so, and a
    /// keyframe count defaulted to `0` would report a world that had accepted
    /// nothing when the truth is that nobody counted.
    func testAbsentFieldsStayAbsentRatherThanBecomingZero() {
        let sparse: [String: Any] = [
            "name": NSNull(),
            "world_id": "w1",
            "keyframe_count": NSNull(),
            "revision": NSNull(),
            "tracking": "unavailable",
            "scale": "unknown",
            "mapping_seconds": NSNull(),
            "calibration": "unknown",
            "geometry": [
                "representation": NSNull(),
                "element_count": NSNull(),
                "is_incremental": false,
            ],
            "trajectory": [
                "pose_count": NSNull(),
                "path_length": NSNull(),
                "path_length_unit": NSNull(),
                "scale": "unknown",
            ],
            "persistence": ["state": "saved", "revision": NSNull()],
        ]
        let snapshot = WorldBuilderResultDecoder.snapshot(from: sparse)

        XCTAssertNil(snapshot.name)
        XCTAssertNil(snapshot.keyframeCount)
        XCTAssertNil(snapshot.mappingSeconds)
        XCTAssertNil(snapshot.geometry.elementCount)
        XCTAssertNil(snapshot.geometry.representation)
        XCTAssertNil(snapshot.trajectory.poseCount)
        XCTAssertNil(snapshot.trajectory.pathLength)
        XCTAssertFalse(snapshot.geometry.hasReport)
        XCTAssertFalse(snapshot.trajectory.hasReport)
        // A world id is still a report, so the snapshot is not empty.
        XCTAssertFalse(snapshot.isEmpty)
        XCTAssertEqual(snapshot.persistence, .saved(revision: nil))
    }

    // MARK: Scale

    /// **The load-bearing refusal.** `relative` is internally consistent with an
    /// arbitrary unit; it is not metric, and no figure carrying it may be shown
    /// as one. Both `inferredMetric` and `measuredMetric` are unreachable on
    /// this hardware and neither will arrive.
    func testRelativeScaleIsNeverRenderableAsMetres() {
        let snapshot = WorldBuilderResultDecoder.snapshot(from: fullSnapshot())
        XCTAssertEqual(snapshot.scale, .relative)
        XCTAssertFalse(snapshot.permitsMetricDisplay, "a relative world claimed metric display")
        XCTAssertFalse(
            snapshot.trajectory.distanceDisplayable,
            "a relative path length claimed to be a distance"
        )
        XCTAssertFalse(snapshot.scale.isEstimate, "relative is not an estimate of metres")
    }

    /// It is still a labelled figure, and the Tower names its unit precisely so
    /// it can be shown. "2.9 world units" is not a distance claim; a bare "2.9"
    /// would be, because a reader supplies metres.
    func testARelativePathLengthIsShownWithTheTowersOwnUnit() {
        let snapshot = WorldBuilderResultDecoder.snapshot(from: fullSnapshot())
        XCTAssertTrue(snapshot.trajectory.labelledFigureDisplayable)
        XCTAssertEqual(
            ReportedFigure.format(
                snapshot.trajectory.pathLength ?? 0,
                unit: snapshot.trajectory.pathLengthUnit
            ),
            "2.9 world units"
        )
    }

    /// The Tower refuses a path length outright when it could not be labelled —
    /// a refused pose, more than one segment, or an unknown scale. iOS must
    /// refuse to render the resulting `null` rather than printing a bare zero.
    func testAWithheldPathLengthIsNotRenderedAtAll() {
        let withheld = fullSnapshot(overrides: [
            "trajectory": [
                "pose_count": 12,
                "path_length": NSNull(),
                "path_length_unit": NSNull(),
                "scale": "unknown",
            ]
        ])
        let snapshot = WorldBuilderResultDecoder.snapshot(from: withheld)
        XCTAssertEqual(snapshot.trajectory.poseCount, 12, "the pose count is still a report")
        XCTAssertFalse(snapshot.trajectory.labelledFigureDisplayable)
        XCTAssertFalse(snapshot.trajectory.distanceDisplayable)
    }

    /// An unlabelled figure is refused even when a scale is present: the unit
    /// is the gate, because without one the reader supplies metres.
    func testAFigureWithNoUnitIsRefused() {
        let unlabelled = WorldTrajectoryReport(pathLength: 2.8, pathLengthUnit: nil, scale: .relative)
        XCTAssertFalse(unlabelled.labelledFigureDisplayable)
        XCTAssertFalse(WorldTrajectoryReport(pathLength: 2.8, pathLengthUnit: "", scale: .relative)
            .labelledFigureDisplayable)
    }

    /// `unknown` is a strictly weaker claim than `relative` — the
    /// reconstruction has no unit at all — and must not be mapped to it.
    func testUnknownScaleIsNotUpgradedToRelative() {
        XCTAssertEqual(WorldBuilderResultDecoder.scale(nil), .unknown)
        XCTAssertEqual(WorldBuilderResultDecoder.scale("unknown"), .unknown)
        XCTAssertEqual(WorldBuilderResultDecoder.scale("nonsense"), .unknown)
        XCTAssertEqual(WorldBuilderResultDecoder.scale("relative"), .relative)
    }

    /// The two the Tower cannot reach today are still mapped rather than
    /// discarded — silently downgrading an arriving metric claim to `.relative`
    /// would understate it, and `inferredMetric` must stay labelled as an
    /// estimate wherever it is shown.
    func testAnInferredMetricFigureWouldStayLabelledAsAnEstimate() {
        XCTAssertEqual(WorldBuilderResultDecoder.scale("inferredMetric"), .inferredMetric)
        XCTAssertEqual(WorldBuilderResultDecoder.scale("measuredMetric"), .measuredMetric)
        XCTAssertTrue(WorldScaleSemantics.inferredMetric.isEstimate)
        XCTAssertEqual(WorldScaleSemantics.inferredMetric.displayName, "Estimated")
    }

    // MARK: The vocabularies the Tower never sends

    /// `limited` needs a threshold nobody has defined and is not emitted;
    /// `calibrating` has no in-session procedure to be in the middle of. Both
    /// are mapped anyway, so that one arriving later is not silently folded
    /// into a weaker state.
    func testTheStatesTheTowerNeverSendsStillMapCorrectly() {
        XCTAssertEqual(WorldBuilderResultDecoder.tracking("good"), .good)
        XCTAssertEqual(WorldBuilderResultDecoder.tracking("lost"), .lost)
        XCTAssertEqual(WorldBuilderResultDecoder.tracking("unavailable"), .unavailable)
        XCTAssertEqual(WorldBuilderResultDecoder.tracking(nil), .unavailable)
        XCTAssertEqual(WorldBuilderResultDecoder.tracking("limited"), .limited)

        XCTAssertEqual(WorldBuilderResultDecoder.calibration("calibrated"), .calibrated)
        XCTAssertEqual(WorldBuilderResultDecoder.calibration("uncalibrated"), .uncalibrated)
        XCTAssertEqual(WorldBuilderResultDecoder.calibration("unknown"), .unknown)
        XCTAssertEqual(WorldBuilderResultDecoder.calibration(nil), .unknown)
        XCTAssertEqual(WorldBuilderResultDecoder.calibration("calibrating"), .calibrating)
    }

    /// `saved` is the only state the Tower reaches, and silence is not a
    /// promise that the world is discarded.
    func testPersistenceSilenceIsNotSessionOnly() {
        XCTAssertEqual(WorldBuilderResultDecoder.persistence(state: nil, revision: nil), .unknown)
        XCTAssertEqual(
            WorldBuilderResultDecoder.persistence(state: "saved", revision: "r1"),
            .saved(revision: "r1")
        )
        XCTAssertNotEqual(
            WorldBuilderResultDecoder.persistence(state: nil, revision: nil),
            .session
        )
    }

    // MARK: Against the real Tower's bytes

    /// **A payload captured verbatim from a running Tower**, not composed here.
    ///
    /// Every other fixture in this file was written from
    /// `tower/tower/results/world_builder.py` and could therefore be wrong in
    /// the same way the reading of that file was wrong. This one was taken off
    /// the wire on 2026-08-24 from `tower.main:app` at
    /// `world_builder.status/2026-08-23`, after a build that produced 8
    /// keyframes, 2336 points and 7 solved poses.
    ///
    /// **Left at `/2026-08-23`, because that is when it was captured.** The
    /// identifier has since moved to `/2026-08-25`, and this fixture is
    /// deliberately not relabelled: rewriting the provenance of a captured
    /// payload would be a claim about a wire nobody listened to. It still
    /// decodes field for field, which is the point — the bump changed the
    /// *meaning* of `trajectory.pose_count` (see
    /// `WorldBuilderResultContract.identifier`) and not the encoding, so the
    /// bytes are unchanged and only the reading of one number is not. Under
    /// the old rule this world's 8 was `keyframes - poses_refused`; under the
    /// new one it is `poses_positioned` — 7 solved plus 1 segment anchor. The
    /// same 8 for a different reason, and the two agree only because this
    /// world has exactly one segment and that segment solved something.
    ///
    /// The world it describes is **synthetic** — rendered, not photographed —
    /// which is a fact about the reconstruction and not about the wire. What
    /// this pins is the encoding: the key names, the nesting, the types, and
    /// which fields carry `null`.
    ///
    /// It was taken at `world_builder.status/2026-08-23`, and it is kept at that
    /// vintage on purpose: `world_snapshot`'s own keys did not change when the
    /// identifier moved to `.../2026-08-25` — `poses_anchor` and `segments` were
    /// added to the payload's separate `trajectory` block, not to this one — so
    /// these bytes remain a valid encoding of a snapshot, and the fact that they
    /// still decode is itself the assertion.
    private static let capturedFromTower = """
        {
          "name": "Wire Check (synthetic)",
          "world_id": "d53fa2781dc94339b1c8640ceb50d6a3",
          "keyframe_count": 8,
          "revision": "05576f31fec4fb79",
          "tracking": "good",
          "scale": "relative",
          "mapping_seconds": 0.2812764644622803,
          "calibration": "calibrated",
          "geometry": {
            "representation": "sparse point cloud",
            "element_count": 2336,
            "is_incremental": false
          },
          "trajectory": {
            "pose_count": 8,
            "path_length": 6.626530639332543,
            "path_length_unit": "world units",
            "scale": "relative"
          },
          "persistence": {
            "state": "saved",
            "revision": "a8ae1778a260e976"
          }
        }
        """

    func testARealTowerSnapshotDecodesFieldForField() throws {
        let data = Data(Self.capturedFromTower.utf8)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let snapshot = WorldBuilderResultDecoder.snapshot(from: json)

        XCTAssertEqual(snapshot.name, "Wire Check (synthetic)")
        XCTAssertEqual(snapshot.worldID, "d53fa2781dc94339b1c8640ceb50d6a3")
        XCTAssertEqual(snapshot.keyframeCount, 8)
        XCTAssertEqual(snapshot.revision, "05576f31fec4fb79")
        XCTAssertEqual(snapshot.tracking, .good)
        XCTAssertEqual(snapshot.scale, .relative)
        XCTAssertEqual(snapshot.calibration, .calibrated)
        XCTAssertEqual(snapshot.geometry.elementCount, 2336)
        XCTAssertEqual(snapshot.geometry.representation, "sparse point cloud")
        XCTAssertEqual(snapshot.trajectory.poseCount, 8)
        XCTAssertEqual(snapshot.persistence, .saved(revision: "a8ae1778a260e976"))
        XCTAssertFalse(snapshot.isEmpty)

        // And what a person would actually be shown, which is the assertion
        // that would catch a decode that "worked" into an unrenderable state.
        XCTAssertFalse(snapshot.permitsMetricDisplay, "a synthetic relative world claimed metres")
        XCTAssertTrue(snapshot.trajectory.labelledFigureDisplayable)
        XCTAssertEqual(
            ReportedFigure.format(
                snapshot.trajectory.pathLength ?? 0,
                unit: snapshot.trajectory.pathLengthUnit
            ),
            "6.6 world units"
        )
    }

    /// The Tower with no worlds yet, also captured verbatim. `world_snapshot`
    /// is `null` and `model_state_reason` explains why — and `.idle` must not
    /// be confused for a world with every field missing.
    func testARealIdleTowerDecodesToIdleWithNoWorld() throws {
        let raw = """
            {"model_state": "idle",
             "model_state_reason": "no worlds exist under this Tower's world root",
             "world_snapshot": null}
            """
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
        )
        let state = WorldBuilderResultDecoder.modelState(from: json)
        XCTAssertEqual(state, .idle)
        XCTAssertNil(state?.snapshot)
        XCTAssertFalse(state?.hasWorld ?? true)
        XCTAssertFalse(state?.phase.mayCarryData ?? true)
    }

    // MARK: Revisions

    /// A revision is an opaque change identity, and a snapshot supersedes every
    /// earlier one. Two revisions differing means the state differs; nothing
    /// orders them.
    func testANewRevisionSupersedesTheSnapshotBeforeIt() {
        let first = WorldBuilderResultDecoder.snapshot(from: fullSnapshot())
        let second = WorldBuilderResultDecoder.snapshot(
            from: fullSnapshot(overrides: ["revision": "ffff0000", "keyframe_count": 9])
        )
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(second.revision, "ffff0000")
        XCTAssertEqual(second.keyframeCount, 9)
    }
}

// MARK: - The client, against a socket

/// The Tower-backed client end to end: discovery, subscription, lifecycle, and
/// what a reconnect does to all three.
@MainActor
final class TowerWorldBuilderClientTests: XCTestCase {

    /// `nonisolated` because it is read as a default argument below, and a
    /// default argument is evaluated in the caller's context rather than in
    /// this class's. A constant string has no isolation to give up.
    private nonisolated static let contract = "world_builder.status/2026-09-06"

    private func url(port: UInt16) -> URL { URL(string: "ws://127.0.0.1:\(port)/")! }

    /// A server that answers the handshake, answers `{"type":"cartridges"}`
    /// with a declaration, and acknowledges a subscribe — the three exchanges
    /// the real Tower performs before the first snapshot.
    private func serve(
        _ server: MockTowerServer,
        available: Bool = true,
        unavailableReason: String? = nil,
        contract: String = TowerWorldBuilderClientTests.contract,
        recorder: MessageRecorder? = nil
    ) {
        let reason = unavailableReason.map { "\"\($0)\"" } ?? "null"
        // Numbered per connection, as the Tower's `next_subscription_id`
        // does: a client that re-subscribes gets `sub-2`, and a heartbeat
        // for `sub-1` after that is one it has left. A constant `sub-1`
        // here once hid exactly that race.
        var subscribeCount = 0
        server.onText = { text in
            recorder?.record(text)
            guard
                let data = text.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let type = json["type"] as? String
            else { return }
            switch type {
            case "ping":
                server.send(text: #"{"type":"pong"}"#)
            case "cartridges":
                // Asked once per connection, so a reconnect starts the
                // numbering again, as the Tower's per-connection channel does.
                subscribeCount = 0
                server.send(text: """
                    {"type":"cartridges",
                     "envelope_contract":"cartridge_results.envelope/2026-08-23",
                     "cartridges":[{"cartridge":"world_builder","result_type":"status",
                        "contract":"\(contract)","available":\(available),
                        "unavailable_reason":\(reason),"snapshot_only":true}],
                     "not_offered":[]}
                    """)
            case "result_subscribe":
                subscribeCount += 1
                server.send(text: """
                    {"type":"result_subscribed",
                     "envelope_contract":"cartridge_results.envelope/2026-08-23",
                     "subscription_id":"sub-\(subscribeCount)","cartridge":"world_builder",
                     "result_type":"status","contract":"\(contract)",
                     "snapshot_only":true,"world_id":null,"session_id":null,
                     "cursor_status":"absent"}
                    """)
            default:
                break
            }
        }
    }

    private func snapshotMessage(
        seq: Int,
        modelState: String,
        keyframes: Int,
        revision: String,
        revisionChanged: Bool = true,
        tracking: String = "good",
        subscription: String = "sub-1"
    ) -> String {
        """
        {"type":"cartridge_result",
         "envelope_contract":"cartridge_results.envelope/2026-08-23",
         "subscription_id":"\(subscription)","cartridge":"world_builder","result_type":"status",
         "contract":"\(Self.contract)","seq":\(seq),"revision":"\(revision)",
         "revision_changed":\(revisionChanged),"coalesced":0,"cursor_status":null,
         "snapshot":true,"tower_sent_at":1787463092.9,"time_basis":"tower-receipt",
         "payload":{"model_state":"\(modelState)","model_state_reason":null,
           "world_snapshot":{"name":"Probe Room","world_id":"w1",
             "keyframe_count":\(keyframes),"revision":"\(revision)",
             "tracking":"\(tracking)","scale":"relative","mapping_seconds":12.5,
             "calibration":"calibrated",
             "geometry":{"representation":"sparse point cloud","element_count":1360,
                         "is_incremental":false},
             "trajectory":{"pose_count":\(keyframes),"path_length":2.85,
                           "path_length_unit":"world units","scale":"relative"},
             "persistence":{"state":"saved","revision":"p1"}}}}
        """
    }

    /// See `TowerClientTests.expect` — an autoclosure cannot carry an `await`,
    /// so the assertion is wrapped rather than hoisted at every call site.
    private func expect(
        _ message: @autoclosure () -> String = "the condition was never met",
        timeout: TimeInterval = 3,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ condition: @MainActor () -> Bool
    ) async {
        let met = await waitUntil(timeout: timeout, condition)
        XCTAssertTrue(met, message(), file: file, line: line)
    }

    private func waitUntil(
        timeout: TimeInterval = 3,
        _ condition: @MainActor () -> Bool
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 25_000_000)
        }
        return condition()
    }

    private func decode(_ text: String) -> [String: Any]? {
        guard let data = text.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    // MARK: The lifecycle

    /// The whole sequence, driven by the Tower's own `model_state` values:
    /// idle → awaitingFirstUpdate → receiving → finalizing → finalized.
    ///
    /// `awaitingFirstUpdate` is the one the Tower never sends. It is reached by
    /// the phone having subscribed and not yet been answered, which is the only
    /// machine that can know it.
    func testTheWorldBuilderLifecycleRunsEndToEnd() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        var seen: [CartridgePhase] = []
        let cancellable = client.stateUpdates.sink { seen.append($0.phase) }
        defer { cancellable.cancel() }

        XCTAssertEqual(client.state, .idle, "the client claimed something before connecting")

        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }
        XCTAssertEqual(client.state.phase, .waiting, "a spinner is only honest while waiting")

        server.send(text: snapshotMessage(seq: 1, modelState: "receiving", keyframes: 4, revision: "r1"))
        await expect { client.state.isReceivingUpdates }
        XCTAssertEqual(client.state.snapshot?.keyframeCount, 4)

        server.send(text: snapshotMessage(seq: 2, modelState: "receiving", keyframes: 17, revision: "r2"))
        await expect { client.state.snapshot?.keyframeCount == 17 }

        server.send(text: snapshotMessage(seq: 3, modelState: "finalizing", keyframes: 17, revision: "r3"))
        await expect {
            if case .finalizing = client.state { return true }
            return false
        }
        XCTAssertFalse(
            client.state.isReceivingUpdates,
            "finalizing claimed to be live; no new observations are arriving"
        )

        server.send(text: snapshotMessage(seq: 4, modelState: "finalized", keyframes: 17, revision: "r4"))
        await expect {
            if case .finalized = client.state { return true }
            return false
        }
        XCTAssertEqual(client.state.phase, .settled)
        XCTAssertTrue(client.state.hasWorld)

        XCTAssertEqual(
            seen,
            [.waiting, .live, .live, .live, .settled],
            "the lifecycle did not pass through the phases in order: \(seen)"
        )

        tower.disconnect()
    }

    /// The wait for `result_subscribed` is bounded, and ends in a state a
    /// person can read.
    ///
    /// `subscribeIfPossible` shows a spinner *before* sending, and
    /// `sendResultMessage` deliberately swallows a send failure so the result
    /// channel can never take down the frame path. Both are right alone;
    /// together they left a subscribe that never landed, on a socket that
    /// stayed up, waiting forever with nothing to end it.
    ///
    /// This server answers the ping and the cartridge declaration — so the
    /// socket is genuinely healthy and `.online` — and then simply never
    /// acknowledges the subscription.
    func testAnUnacknowledgedSubscriptionTimesOutRatherThanSpinningForever() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        defer { server.stop() }

        server.onText = { text in
            guard
                let data = text.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let type = json["type"] as? String
            else { return }
            switch type {
            case "ping":
                server.send(text: #"{"type":"pong"}"#)
            case "cartridges":
                server.send(text: """
                    {"type":"cartridges",
                     "envelope_contract":"cartridge_results.envelope/2026-08-23",
                     "cartridges":[{"cartridge":"world_builder","result_type":"status",
                        "contract":"\(Self.contract)","available":true,
                        "unavailable_reason":null,"snapshot_only":true}],
                     "not_offered":[]}
                    """)
            // `result_subscribe` is deliberately unanswered.
            default:
                break
            }
        }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower, subscribeAckTimeout: .milliseconds(300))
        tower.connect(to: url(port: port))

        await expect { client.state == .awaitingFirstUpdate }
        XCTAssertEqual(client.state.phase, .waiting, "a spinner is honest while the wait is live")

        await expect("the subscription wait never ended") {
            if case .failed = client.state { return true }
            return false
        }

        guard case .failed(let failure) = client.state else {
            return XCTFail("expected a bounded failure, got \(client.state)")
        }
        XCTAssertEqual(
            failure.kind, .timedOut,
            "a Tower that is reachable but silent is a timeout, not a transport failure"
        )
        XCTAssertEqual(tower.status, .online, "the socket was healthy throughout")
        XCTAssertFalse(client.state.hasWorld)

        tower.disconnect()
    }

    /// The bound must not fire into a subscription that succeeded.
    ///
    /// The timeout sleeps off the main actor, so the ack can land while it is
    /// still sleeping. Without the disarm — and the attempt counter behind it —
    /// a healthy subscription would be torn down a few hundred milliseconds
    /// after it started working.
    func testAnAcknowledgedSubscriptionIsNotTornDownByItsOwnTimeout() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower, subscribeAckTimeout: .milliseconds(200))
        tower.connect(to: url(port: port))

        await expect { client.state == .awaitingFirstUpdate }
        server.send(text: snapshotMessage(seq: 1, modelState: "receiving", keyframes: 4, revision: "r1"))
        await expect { client.state.snapshot?.keyframeCount == 4 }

        // Well past the bound, which must have been disarmed by the ack.
        try? await Task.sleep(nanoseconds: 600_000_000)

        XCTAssertEqual(
            client.state.snapshot?.keyframeCount, 4,
            "a live subscription was torn down by a timeout that should have been disarmed"
        )
        if case .failed = client.state { XCTFail("the bound fired into a working subscription") }

        tower.disconnect()
    }

    /// A result marked as a partial update is refused, and — this is the point
    /// — refused *loudly*.
    ///
    /// `snapshot` is `true` on every envelope the Tower sends today;
    /// `ResultEnvelope` defaults it and nothing overrides it. The field exists
    /// so that a future delta mode cannot be mistaken for this one.
    ///
    /// Without the guard the failure is silent rather than absent. A delta
    /// saying `model_state: "receiving"` with no `world_snapshot` decodes
    /// *cleanly*: `modelState(from:)` returns `.awaitingFirstUpdate`, which
    /// collapses a populated world back to a spinner. This test drives exactly
    /// that shape, after a real world is on screen, so a regression shows up as
    /// the wrong state rather than as a missing error.
    func testAPartialResultIsRefusedRatherThanQuietlyBlankingTheWorld() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        // A real world first, so the refusal has something to protect.
        server.send(text: snapshotMessage(seq: 1, modelState: "receiving", keyframes: 17, revision: "r1"))
        await expect { client.state.snapshot?.keyframeCount == 17 }

        // The dangerous shape: `snapshot: false`, and a payload that would
        // otherwise decode without complaint.
        server.send(text: """
            {"type":"cartridge_result",
             "envelope_contract":"cartridge_results.envelope/2026-08-23",
             "subscription_id":"sub-1","cartridge":"world_builder","result_type":"status",
             "contract":"\(Self.contract)","seq":2,"revision":"r2",
             "revision_changed":true,"coalesced":0,"cursor_status":null,
             "snapshot":false,"tower_sent_at":1787463092.9,"time_basis":"tower-receipt",
             "payload":{"model_state":"receiving","model_state_reason":null}}
            """)

        await expect("a partial result was not refused") {
            if case .failed = client.state { return true }
            return false
        }

        guard case .failed(let failure) = client.state else {
            return XCTFail("expected a refusal, got \(client.state)")
        }
        XCTAssertEqual(
            failure.kind, .notSupported,
            "a delta this build cannot merge is an unsupported result, not an unreadable one"
        )
        XCTAssertNotEqual(
            client.state, .awaitingFirstUpdate,
            "the partial update silently collapsed the world into a spinner"
        )
        XCTAssertFalse(
            client.state.hasWorld,
            "a world must not survive as a fact assembled from a piece that was refused"
        )

        tower.disconnect()
    }

    /// The ~2 s heartbeat re-sends an unchanged snapshot to refresh the fields
    /// excluded from the revision hash. A republish for one of those would put
    /// a list diff on the main actor for nothing.
    func testAnUnchangedHeartbeatDoesNotRepublish() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        var publishes = 0
        let cancellable = client.stateUpdates.sink { _ in publishes += 1 }
        defer { cancellable.cancel() }

        server.send(text: snapshotMessage(seq: 1, modelState: "receiving", keyframes: 4, revision: "r1"))
        await expect { publishes == 1 }

        // Three heartbeats: same revision, same everything.
        for seq in 2...4 {
            server.send(text: snapshotMessage(
                seq: seq, modelState: "receiving", keyframes: 4,
                revision: "r1", revisionChanged: false
            ))
        }
        try? await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertEqual(publishes, 1, "an unchanged snapshot invalidated the view tree")

        // A real change still gets through.
        server.send(text: snapshotMessage(seq: 5, modelState: "receiving", keyframes: 5, revision: "r2"))
        await expect { publishes == 2 }

        tower.disconnect()
    }

    // MARK: Availability

    /// The client resolves availability against the **live** declaration. A
    /// Tower that has said nothing is `.noContract`; one that has declared and
    /// is reachable is `.available`.
    func testAvailabilityFollowsTheLiveDeclaration() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        // `.towerUnreachable`, not `.noContract`: nothing has been declared
        // because nobody has connected yet, and World Builder is a cartridge
        // this build can receive a declaration for. Blaming the Tower here
        // would be a claim about a machine no socket has reached.
        XCTAssertEqual(client.availability(isTowerReachable: false), .towerUnreachable)

        tower.connect(to: url(port: port))
        await expect { tower.cartridgeDeclaration != nil }
        XCTAssertTrue(client.availability(isTowerReachable: true).isAvailable)
        XCTAssertNil(
            client.availability(isTowerReachable: true).forcedPhase,
            "an available cartridge must let its own state through"
        )

        tower.disconnect()
    }

    /// Offered and unserveable is not the same as never offered. The Tower's
    /// own prose is the only honest explanation, and the app shows it verbatim
    /// rather than composing one.
    func testAnUnavailableOfferShowsTheTowersReasonAndDoesNotSubscribe() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(
            server,
            available: false,
            unavailableReason: "no world root is configured on this Tower",
            recorder: recorder
        )
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))

        await expect {
            if case .unsupported = client.state { return true }
            return false
        }
        guard case .unsupported(let reason) = client.state else {
            return XCTFail("expected .unsupported, got \(client.state)")
        }
        XCTAssertEqual(reason, "no world root is configured on this Tower")

        try? await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertFalse(
            recorder.all.compactMap(decode).contains { $0["type"] as? String == "result_subscribe" },
            "the client subscribed to a cartridge the Tower said it could not serve"
        )

        tower.disconnect()
    }

    /// A contract this build does not implement must not be decoded on a guess.
    /// `.unsupportedContract` tells a person to update the app; subscribing
    /// anyway would produce a payload nothing here was written against.
    func testAnUnknownContractIsNeitherSubscribedToNorDecoded() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, contract: "world_builder.status/2027-06-01", recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { tower.cartridgeDeclaration != nil }

        try? await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertFalse(
            recorder.all.compactMap(decode).contains { $0["type"] as? String == "result_subscribe" },
            "the client subscribed against a contract it does not implement"
        )
        // `.needsUpdate`: the Tower declared World Builder, so it can do this
        // and the app is what is behind. Rendering that as "Nothing yet" would
        // tell the wearer the cartridge does not exist.
        XCTAssertEqual(
            client.availability(isTowerReachable: true).forcedPhase,
            .needsUpdate
        )

        tower.disconnect()
    }

    // MARK: Reconnect

    /// The reconnect property the whole design rests on: **there is no delta
    /// stream, so a reconnect cannot lose data.** Subscribe again and the first
    /// message is a complete snapshot.
    ///
    /// One client object throughout, one subscription per socket, and the world
    /// that was on screen is not forgotten on the way down — availability
    /// reports the connection, so the client does not have to fabricate one.
    func testAReconnectResubscribesExactlyOnceAndDoesNotDuplicateTheClient() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics(), autoReconnect: true)
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        server.send(text: snapshotMessage(seq: 1, modelState: "receiving", keyframes: 30, revision: "r1"))
        await expect { client.state.snapshot?.keyframeCount == 30 }

        let subscribesBefore = recorder.all.compactMap(decode)
            .filter { $0["type"] as? String == "result_subscribe" }.count
        XCTAssertEqual(subscribesBefore, 1)

        // The socket dies without a close frame, exactly like a WiFi drop.
        server.dropConnection()
        await expect { tower.status != .online }

        // Nothing about the world is forgotten on the way down; the connection
        // is what changed, and availability is where that is reported.
        XCTAssertEqual(client.state.snapshot?.keyframeCount, 30)
        XCTAssertEqual(
            client.availability(isTowerReachable: false).forcedPhase,
            .disconnected,
            "a dropped socket must read as disconnected, not as unsupported"
        )

        await expect(timeout: 8) { tower.status == .online }
        await expect(timeout: 5) {
            recorder.all.compactMap(self.decode)
                .filter { $0["type"] as? String == "result_subscribe" }.count == 2
        }
        XCTAssertEqual(
            recorder.all.compactMap(decode).filter { $0["type"] as? String == "result_subscribe" }.count,
            2,
            "the reconnect opened more than one subscription"
        )

        // The keyframe count continues from the Tower's figure, which is the
        // one that survived — the capture lineage is the Tower's to keep.
        server.send(text: snapshotMessage(seq: 1, modelState: "receiving", keyframes: 41, revision: "r9"))
        await expect { client.state.snapshot?.keyframeCount == 41 }

        tower.disconnect()
    }

    /// A declaration republished while a subscribe is still in flight must not
    /// open a second subscription. Both paths into `subscribeIfPossible` are
    /// guarded by the same flags, and this is that guarantee as a test.
    func testARepublishedDeclarationDoesNotOpenASecondSubscription() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        // Three more declarations on the same socket, as a Tower restarting its
        // own declaration cache would produce.
        for _ in 0..<3 { tower.requestCartridgeDeclaration() }
        try? await Task.sleep(nanoseconds: 400_000_000)

        XCTAssertEqual(
            recorder.all.compactMap(decode).filter { $0["type"] as? String == "result_subscribe" }.count,
            1,
            "a republished declaration opened another subscription"
        )

        tower.disconnect()
    }

    // MARK: Errors

    /// `consumer_too_slow` and `channel_failed` close a subscription and are
    /// recoverable by subscribing again — but only a bounded number of times,
    /// so a Tower that closes every subscription becomes visible rather than
    /// staying in motion.
    func testAClosedSubscriptionIsRetriedAndThenReportedRatherThanLooping() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        for _ in 0..<6 {
            server.send(text: """
                {"type":"result_error","reason":"channel_failed","subscription_id":"sub-1",
                 "cartridge":"world_builder","result_type":"status","message":"the reader died"}
                """)
            try? await Task.sleep(nanoseconds: 120_000_000)
        }

        guard case .failed(let failure) = client.state else {
            return XCTFail("expected .failed after an unrecoverable channel, got \(client.state)")
        }
        XCTAssertEqual(failure.kind, .transport)

        let subscribes = recorder.all.compactMap(decode)
            .filter { $0["type"] as? String == "result_subscribe" }.count
        XCTAssertEqual(subscribes, 4, "the resubscribe budget was not bounded: \(subscribes) attempts")
        XCTAssertEqual(tower.status, .online, "a result-channel failure took the connection down")

        tower.disconnect()
    }

    /// An error naming another cartridge is not this client's to claim.
    /// Attributing it here would be a fabricated report about the Tower.
    func testAnErrorForAnotherCartridgeIsNotClaimed() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        server.send(text: snapshotMessage(seq: 1, modelState: "receiving", keyframes: 4, revision: "r1"))
        await expect { client.state.isReceivingUpdates }

        server.send(text: """
            {"type":"result_error","envelope_contract":"cartridge_results.envelope/2026-08-23",
             "reason":"unknown_cartridge","message":"no such cartridge",
             "cartridge":"document_memory","result_type":"status","offered":["world_builder"]}
            """)
        try? await Task.sleep(nanoseconds: 250_000_000)

        XCTAssertTrue(
            client.state.isReceivingUpdates,
            "another cartridge's failure was rendered as this one's"
        )

        tower.disconnect()
    }

    /// A payload this build cannot read is `.undecodableResponse` — a
    /// disagreement discovered on arrival — and not an empty world.
    func testAnUnreadablePayloadIsAFailureNotAnEmptyWorld() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        server.send(text: """
            {"type":"cartridge_result","envelope_contract":"cartridge_results.envelope/2026-08-23",
             "subscription_id":"sub-1","cartridge":"world_builder","result_type":"status",
             "contract":"\(Self.contract)","seq":1,"revision":"r1","revision_changed":true,
             "coalesced":0,"cursor_status":null,"snapshot":true,
             "tower_sent_at":1.0,"time_basis":"tower-receipt",
             "payload":{"model_state":"reticulating_splines","world_snapshot":null}}
            """)

        await expect {
            if case .failed = client.state { return true }
            return false
        }
        guard case .failed(let failure) = client.state else { return XCTFail("no failure") }
        XCTAssertEqual(failure.kind, .undecodableResponse)
        XCTAssertFalse(client.state.hasWorld)

        tower.disconnect()
    }

    // MARK: Ownership

    /// The client is owned above the workspace, so a cartridge switch — which
    /// constructs and releases view models — cannot cost it its subscription or
    /// its accumulated world.
    func testAWorkspaceSwitchDoesNotDisturbTheClientOrItsSubscription() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }
        server.send(text: snapshotMessage(seq: 1, modelState: "receiving", keyframes: 12, revision: "r1"))
        await expect { client.state.snapshot?.keyframeCount == 12 }

        let wireBefore = recorder.all.count
        for _ in 0..<5 {
            let viewModel = WorldBuilderViewModel(client: client)
            XCTAssertEqual(
                viewModel.state.snapshot?.keyframeCount,
                12,
                "a fresh view model did not see the world the client already held"
            )
        }
        try? await Task.sleep(nanoseconds: 200_000_000)

        XCTAssertEqual(recorder.all.count, wireBefore, "a view model put something on the wire")
        XCTAssertEqual(client.state.snapshot?.keyframeCount, 12)

        // And the client is still live: the next snapshot still lands.
        server.send(text: snapshotMessage(seq: 2, modelState: "receiving", keyframes: 13, revision: "r2"))
        await expect { client.state.snapshot?.keyframeCount == 13 }

        tower.disconnect()
    }

    /// The app graph builds exactly one Tower-backed client, and it is the one
    /// the workspace is handed — the failure this would catch is a second
    /// client constructed at the point of use, holding its own subscription
    /// and dying on every cartridge switch.
    ///
    /// `GlassesConnection` is given a scripted `WearablesInterface` rather than
    /// the real one. Constructing the true production graph here would subscribe
    /// to live DAT streams from a unit test, which is both a camera-owning
    /// object nobody asked for and a source of interference for every other
    /// test in the run. What is under test is the *wiring*, and the wiring is
    /// unaffected by which `WearablesInterface` is underneath.
    func testTheAppGraphOwnsOneTowerBackedWorldBuilderClient() {
        let project = ProjectManager(
            glassesConnection: GlassesConnection(wearables: ScriptedWearables(permissionResults: []))
        )
        XCTAssertTrue(
            project.cartridgeClients.worldBuilder is TowerWorldBuilderClient,
            "the app graph is not wiring the Tower-backed client"
        )
        XCTAssertTrue(
            project.cartridgeClients.worldBuilder === project.cartridgeClients.worldBuilder,
            "the container is handing out a new client per access"
        )
        XCTAssertEqual(project.cartridgeClients.worldBuilder.cartridgeID, "world-build")

        // And the client is wired to the graph's one connection, not to a
        // second `TowerClient` it made for itself: asking the connection what
        // it has declared is the only way this client answers at all.
        XCTAssertEqual(
            project.cartridgeClients.worldBuilder.availability(isTowerReachable: false),
            .towerUnreachable,
            "the client answered from something other than the graph's connection"
        )
    }

    // MARK: Stored worlds

    private func subscribes(_ recorder: MessageRecorder) -> [[String: Any]] {
        recorder.all.compactMap(decode).filter { $0["type"] as? String == "result_subscribe" }
    }

    /// Following the live world sends **no** pin. The keys are absent, not
    /// `null`: a `"world_id": null` would be a claim about a world rather than
    /// the absence of one, and the Tower's own default is what resolves the
    /// live world.
    func testFollowingTheLiveWorldSendsNeitherWorldNorSessionID() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        let sent = subscribes(recorder)
        XCTAssertEqual(sent.count, 1)
        XCTAssertFalse(sent[0].keys.contains("world_id"), "an unpinned subscribe named a world")
        XCTAssertFalse(sent[0].keys.contains("session_id"), "an unpinned subscribe named a session")
        XCTAssertEqual(client.inspection, .live)

        tower.disconnect()
    }

    /// Opening a stored world closes the live subscription and opens a new one
    /// carrying `world_id` and `session_id`; returning to live opens a third
    /// with neither. The live world's snapshot does not survive the switch —
    /// the pinned subscribe is answered with a complete snapshot of the other
    /// world, and until it lands the honest state is "waiting".
    func testInspectingAStoredWorldResubscribesWithThePinAndBackWithout() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        var modes: [WorldInspectionMode] = []
        let cancellable = client.inspectionUpdates.sink { modes.append($0) }
        defer { cancellable.cancel() }

        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }
        server.send(text: snapshotMessage(seq: 1, modelState: "receiving", keyframes: 30, revision: "r1"))
        await expect { client.state.snapshot?.keyframeCount == 30 }

        client.inspect(worldID: "w-stored", sessionID: "s-stored")
        XCTAssertEqual(client.inspection, .inspecting(worldID: "w-stored"))
        XCTAssertEqual(client.state, .awaitingFirstUpdate, "the live world's snapshot survived the switch")

        await expect { self.subscribes(recorder).count == 2 }
        let pinned = subscribes(recorder)[1]
        XCTAssertEqual(pinned["world_id"] as? String, "w-stored")
        XCTAssertEqual(pinned["session_id"] as? String, "s-stored")
        XCTAssertEqual(pinned["cartridge"] as? String, "world_builder")
        XCTAssertEqual(pinned["contract"] as? String, Self.contract)
        XCTAssertTrue(
            recorder.all.compactMap(decode).contains {
                $0["type"] as? String == "result_unsubscribe"
                    && $0["subscription_id"] as? String == "sub-1"
            },
            "the live subscription was left open under the pinned one"
        )

        // The pinned world's snapshot lands like any other, under the new id.
        server.send(text: snapshotMessage(seq: 1, modelState: "finalized", keyframes: 143, revision: "r-stored", subscription: "sub-2"))
        await expect { client.state.snapshot?.keyframeCount == 143 }

        client.followLive()
        XCTAssertEqual(client.inspection, .live)
        await expect { self.subscribes(recorder).count == 3 }
        let live = subscribes(recorder)[2]
        XCTAssertFalse(live.keys.contains("world_id"))
        XCTAssertFalse(live.keys.contains("session_id"))

        XCTAssertEqual(modes, [.inspecting(worldID: "w-stored"), .live])

        tower.disconnect()
    }

    /// An envelope for a subscription this client has already left is not
    /// applied. The Tower's sender may have queued a heartbeat for the old
    /// subscription before the unsubscribe reached it; that envelope names
    /// the world the reader just left, and applying it would put that
    /// world's snapshot — and its geometry address — back on screen under a
    /// header naming the new one.
    func testAnEnvelopeFromTheSubscriptionJustLeftIsIgnored() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }
        server.send(text: snapshotMessage(seq: 1, modelState: "receiving", keyframes: 30, revision: "r1"))
        await expect { client.state.snapshot?.keyframeCount == 30 }

        client.inspect(worldID: "w-stored", sessionID: "s-stored")
        await expect { self.subscribes(recorder).count == 2 }

        // A late heartbeat for `sub-1`: the live world, which the reader has
        // left. It must not become the stored world's state.
        server.send(text: snapshotMessage(seq: 2, modelState: "receiving", keyframes: 31, revision: "r2"))
        // Then the stored world's own snapshot under `sub-2`, which must.
        server.send(text: snapshotMessage(seq: 1, modelState: "finalized", keyframes: 143, revision: "r-stored", subscription: "sub-2"))
        await expect { client.state.snapshot?.keyframeCount == 143 }
        XCTAssertEqual(client.state.snapshot?.keyframeCount, 143)
        XCTAssertEqual(client.inspection, .inspecting(worldID: "w-stored"))

        // And after Back to live, a straggler for `sub-2` is dropped the same way.
        client.followLive()
        await expect { self.subscribes(recorder).count == 3 }
        server.send(text: snapshotMessage(seq: 2, modelState: "finalized", keyframes: 144, revision: "r-stored2", subscription: "sub-2"))
        server.send(text: snapshotMessage(seq: 3, modelState: "receiving", keyframes: 32, revision: "r3", subscription: "sub-3"))
        await expect { client.state.snapshot?.keyframeCount == 32 }
        XCTAssertEqual(client.state.snapshot?.keyframeCount, 32)

        tower.disconnect()
    }

    /// A pin with no session lets the Tower choose the session, and sends
    /// only the world.
    func testInspectingAWorldWithoutASessionSendsOnlyTheWorldID() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        client.inspect(worldID: "w-stored", sessionID: nil)
        await expect { self.subscribes(recorder).count == 2 }
        let pinned = subscribes(recorder)[1]
        XCTAssertEqual(pinned["world_id"] as? String, "w-stored")
        XCTAssertFalse(pinned.keys.contains("session_id"), "a nil session was sent as a key")

        tower.disconnect()
    }

    /// While pinned, the capture bracket is not consulted: the reader asked
    /// for that world by name, so the gate's answer is `.none` and the stored
    /// world is drawn even though this phone has a capture open. Without the
    /// pin the same message — no `session` block — would present as waiting.
    func testAPinnedWorldIsDrawnEvenWhileThisPhoneHasACaptureOpen() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        tower.sendStreamStart()
        await expect { tower.isStreamingToTower }

        // Unpinned, a snapshot with no session block is `.awaiting` and is
        // presented as waiting — the existing gate.
        server.send(text: snapshotMessage(seq: 1, modelState: "receiving", keyframes: 30, revision: "r1"))
        try? await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertEqual(client.state, .awaitingFirstUpdate)
        XCTAssertEqual(client.sessionBinding, .awaiting(captureID: nil))

        client.inspect(worldID: "w-stored", sessionID: "s-stored")
        server.send(text: snapshotMessage(seq: 1, modelState: "finalized", keyframes: 143, revision: "r-stored", subscription: "sub-2"))
        await expect { client.state.snapshot?.keyframeCount == 143 }
        XCTAssertEqual(client.sessionBinding, WorldSessionBinding.none)

        tower.sendStreamStop()
        tower.disconnect()
    }

    /// The pin is kept across a reconnect: a reader looking at a stored world
    /// who loses WiFi is still looking at that world when it comes back.
    func testThePinSurvivesAReconnect() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics(), autoReconnect: true)
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        client.inspect(worldID: "w-stored", sessionID: "s-stored")
        await expect { self.subscribes(recorder).count == 2 }

        server.dropConnection()
        await expect { tower.status != .online }
        await expect(timeout: 8) { tower.status == .online }
        await expect(timeout: 5) { self.subscribes(recorder).count == 3 }

        let resubscribed = subscribes(recorder)[2]
        XCTAssertEqual(resubscribed["world_id"] as? String, "w-stored")
        XCTAssertEqual(resubscribed["session_id"] as? String, "s-stored")
        XCTAssertEqual(client.inspection, .inspecting(worldID: "w-stored"))

        tower.disconnect()
    }
}

// MARK: - The 2026-08-25 contract: anchors, segments, and whose world this is

/// What changed on the wire between `world_builder.status/2026-08-23` and
/// `world_builder.status/2026-08-25`, and the gate that decides whether a
/// snapshot describes the capture this phone has open.
///
/// Every fixture here was written from
/// `tower/docs/contracts/CARTRIDGE-RESULTS.md` §10 and
/// `tower/tower/results/world_builder.py` on
/// `integration/world-builder-lifecycle-v1`, not from a summary of either.
@MainActor
final class WorldBuilderContract20260825Tests: XCTestCase {

    // MARK: The identifier

    /// The identifier moved because **a field changed meaning**, not because a
    /// field was added — and then again because `model_state` gained a word.
    /// Pinning either old one now would ask the Tower for a contract it no
    /// longer serves.
    func testThisBuildImplementsTheContractTheTowerNowOffers() {
        XCTAssertEqual(WorldBuilderResultContract.identifier, "world_builder.status/2026-09-06")
        XCTAssertFalse(
            TowerCapabilities.supported.contains("world_builder.status/2026-08-25"),
            "the 2026-08-25 contract is still claimed; it has no word for an interrupted world"
        )
        // Membership, not set equality. This test is about *World Builder's*
        // identifier being the one implemented; it stopped being the only
        // member on 2026-08-27 when four Tower lanes were unified and three
        // more cartridges gained clients. Exclusivity — that the set is exactly
        // the five this build decodes — is pinned once, centrally, in
        // `ProductShellTests.testTheImplementedContractsAreExactlyTheFiveThisBuildDecodes`.
        // Re-asserting it here made a World Builder test fail for a Scene
        // Understanding change, which points a reader at the wrong lane.
        XCTAssertTrue(TowerCapabilities.supported.contains("world_builder.status/2026-09-06"))
        XCTAssertFalse(
            TowerCapabilities.supported.contains("world_builder.status/2026-08-23"),
            """
            the superseded contract is still claimed. Its `pose_count` was \
            `keyframes - poses_refused`, which promoted every segment anchor \
            to a camera position.
            """
        )
    }

    // MARK: Positioned poses versus anchors

    /// The trajectory block of the 2026-08-24 walk, as the new producer would
    /// report it: a `unposed` build with no intrinsics, which solved nothing.
    ///
    /// `pose_count: 0` beside `poses_anchor: 36` is the whole correction. The
    /// old contract reported 36 here and a phone displayed "Camera poses: 36"
    /// for a reconstruction that positioned no camera at all.
    private static let uncalibratedWalkTrajectory: [String: Any] = [
        "available": true,
        "current": true,
        "built_from_keyframes": 155,
        "keyframes_now": 155,
        "stale_reason": NSNull(),
        "pose_count": 0,
        "poses_solved": 0,
        "poses_refused": 119,
        "poses_anchor": 36,
        "keyframes": 155,
        "segments": 36,
        "path_length": ["available": false, "reason": "the session has more than one segment"],
        "revision": "4a1f",
        "provenance": "inferred",
        "confidence": NSNull(),
        "unavailable_reason": NSNull(),
    ]

    private func snapshotJSON(poseCount: Any = 0) -> [String: Any] {
        [
            "name": NSNull(),
            "world_id": "w-uncalibrated",
            "keyframe_count": 155,
            "revision": "r1",
            "tracking": "good",
            "scale": "relative",
            "mapping_seconds": 411.2,
            "calibration": "uncalibrated",
            "geometry": [
                "representation": "sparse point cloud",
                "element_count": 0,
                "is_incremental": false,
            ],
            "trajectory": [
                "pose_count": poseCount,
                "path_length": NSNull(),
                "path_length_unit": NSNull(),
                "scale": "unknown",
            ],
            "persistence": ["state": "saved", "revision": "p1"],
        ]
    }

    func testAnUncalibratedWalkReadsAsSegmentOriginsAndNoTrajectory() {
        let snapshot = WorldBuilderResultDecoder.snapshot(
            from: snapshotJSON(),
            trajectoryEvidence: Self.uncalibratedWalkTrajectory
        )

        XCTAssertEqual(snapshot.trajectory.poseCount, 0, "a positioned-pose count of zero was lost")
        XCTAssertEqual(snapshot.trajectory.posesAnchor, 36)
        XCTAssertEqual(snapshot.trajectory.posesSolved, 0)
        XCTAssertEqual(snapshot.trajectory.posesRefused, 119)
        XCTAssertEqual(snapshot.trajectory.segments, 36)
        XCTAssertTrue(
            snapshot.trajectory.isAnchorsOnly,
            "36 anchors with no solved pose did not read as origins-without-a-trajectory"
        )
        XCTAssertFalse(
            snapshot.trajectory.hasPositionedPoses,
            "a build that positioned no camera claimed a camera path"
        )
    }

    /// `null` and `0` are different claims all the way to the screen: one is
    /// "the Tower did not say", the other is "the Tower counted none".
    func testAnAbsentTrajectoryIsNotAZeroedOne() {
        let absent: [String: Any] = [
            "available": false,
            "current": false,
            "built_from_keyframes": NSNull(),
            "keyframes_now": NSNull(),
            "stale_reason": NSNull(),
            "pose_count": NSNull(),
            "poses_solved": NSNull(),
            "poses_refused": NSNull(),
            "poses_anchor": NSNull(),
            "keyframes": NSNull(),
            "segments": NSNull(),
            "path_length": NSNull(),
            "revision": NSNull(),
            "provenance": NSNull(),
            "confidence": NSNull(),
            "unavailable_reason": "no build has run for this session, so no poses exist",
        ]
        var json = snapshotJSON(poseCount: NSNull())
        json["trajectory"] = [
            "pose_count": NSNull(), "path_length": NSNull(),
            "path_length_unit": NSNull(), "scale": "unknown",
        ]

        let snapshot = WorldBuilderResultDecoder.snapshot(from: json, trajectoryEvidence: absent)
        XCTAssertNil(snapshot.trajectory.poseCount)
        XCTAssertNil(snapshot.trajectory.posesAnchor)
        XCTAssertNil(snapshot.trajectory.segments)
        XCTAssertFalse(snapshot.trajectory.isAnchorsOnly, "silence was read as zero anchors")
    }

    /// The evidence blocks are optional on the wire only in the sense that a
    /// payload without a world has none. A snapshot decoded without them must
    /// still carry everything the snapshot itself said.
    func testASnapshotDecodedWithoutTheEvidenceBlockKeepsItsOwnFigures() {
        let snapshot = WorldBuilderResultDecoder.snapshot(from: snapshotJSON(poseCount: 4))
        XCTAssertEqual(snapshot.trajectory.poseCount, 4)
        XCTAssertNil(snapshot.trajectory.posesAnchor)
        XCTAssertNil(snapshot.trajectory.segments)
    }

    // MARK: The session block

    private func sessionJSON(
        captureID: Any = "6bf1c84c92f94fb68db62d5ba24c3ad2",
        endedAt: Any = NSNull(),
        endReason: Any = NSNull(),
        frameSource: String = "live-capture"
    ) -> [String: Any] {
        [
            "session_id": "s1",
            "started_at": 1787463000.0,
            "ended_at": endedAt,
            "end_reason": endReason,
            "frame_source": frameSource,
            "capture_id": captureID,
            "retains_raw_imagery": true,
        ]
    }

    func testTheSessionBlockNamesTheCaptureTheWorldWasBuiltFrom() {
        let report = WorldBuilderResultDecoder.session(from: ["session": sessionJSON()])
        XCTAssertEqual(report?.sessionID, "s1")
        XCTAssertEqual(report?.captureID, "6bf1c84c92f94fb68db62d5ba24c3ad2")
        XCTAssertEqual(report?.frameSource, "live-capture")
        XCTAssertFalse(report?.hasEnded ?? true)
        XCTAssertTrue(report?.isLiveCapture ?? false)

        // A world with no sessions sends `session: null`, which is absent and
        // not an empty session.
        XCTAssertNil(WorldBuilderResultDecoder.session(from: ["session": NSNull()]))
        XCTAssertNil(WorldBuilderResultDecoder.session(from: [:]))
    }

    func testAStoppedSessionCarriesItsEndAndItsReason() {
        let report = WorldBuilderResultDecoder.session(
            from: ["session": sessionJSON(endedAt: 1787463122.0, endReason: "disconnect")]
        )
        XCTAssertEqual(report?.endedAt, 1787463122.0)
        XCTAssertEqual(report?.endReason, "disconnect")
        XCTAssertTrue(report?.hasEnded ?? false)
    }

    // MARK: The gate

    private var ours: WorldSessionReport {
        WorldBuilderResultDecoder.session(from: ["session": sessionJSON()])!
    }

    private var finishedEarlier: WorldSessionReport {
        WorldBuilderResultDecoder.session(
            from: ["session": sessionJSON(
                captureID: "2e6cff0d1a3b4c5d6e7f8091a2b3c4d5",
                endedAt: 1787462800.0,
                endReason: "disconnect"
            )]
        )!
    }

    /// **The load-bearing negative, and the 2026-08-24 bug.**
    ///
    /// On that walk the phone showed camera LIVE and "Capture has ended." at
    /// the same time, with frozen figures: the result channel had answered with
    /// the most recently updated world, which was a finished one from earlier.
    /// A snapshot describing a capture that is not the one this phone has open
    /// is not this session's result, whatever the Tower calls its state.
    func testAWorldFromAnEarlierCaptureIsNotShownAsThisSessionsResult() {
        let snapshot = WorldSnapshot(worldID: "w-old", keyframeCount: 143)
        let binding = WorldSessionGate.binding(
            isCaptureBracketOpen: true,
            session: finishedEarlier,
            modelState: .finalized(snapshot)
        )
        XCTAssertEqual(binding, .foreign(captureID: "2e6cff0d1a3b4c5d6e7f8091a2b3c4d5"))

        let presented = WorldSessionGate.presented(.finalized(snapshot), binding: binding)
        XCTAssertEqual(presented, .awaitingFirstUpdate)
        XCTAssertFalse(presented.hasWorld, "a foreign world was rendered as this session's")
    }

    /// A live capture, still open, with a capture directory behind it, while
    /// this phone's bracket is open: that is this session's world.
    func testTheLiveCaptureThisPhoneOpenedBindsAndIsRendered() {
        let snapshot = WorldSnapshot(worldID: "w-live", keyframeCount: 44)
        let binding = WorldSessionGate.binding(
            isCaptureBracketOpen: true,
            session: ours,
            modelState: .receiving(snapshot)
        )
        XCTAssertEqual(binding, .bound(captureID: "6bf1c84c92f94fb68db62d5ba24c3ad2"))
        XCTAssertEqual(
            WorldSessionGate.presented(.receiving(snapshot), binding: binding),
            .receiving(snapshot)
        )
    }

    /// Bracket open, and the Tower has resolved no session at all — the state
    /// between `stream_start` and the follower creating its world. "Frames are
    /// going out and nothing has reported a world" is `.awaitingFirstUpdate`,
    /// which is the one state only the phone can know.
    func testABracketWithNoSessionBehindItIsWaitingAndNotIdle() {
        let binding = WorldSessionGate.binding(
            isCaptureBracketOpen: true, session: nil, modelState: .idle
        )
        XCTAssertEqual(binding, .awaiting(captureID: nil))
        XCTAssertEqual(WorldSessionGate.presented(.idle, binding: binding), .awaitingFirstUpdate)
    }

    /// A recorded or synthetic session is never the capture this phone opened,
    /// however live the Tower says it is.
    func testASessionFedFromDiskIsNeverThisPhonesCapture() {
        for source in ["recorded-capture", "synthetic", "unknown"] {
            let session = WorldBuilderResultDecoder.session(
                from: ["session": sessionJSON(captureID: NSNull(), frameSource: source)]
            )!
            let binding = WorldSessionGate.binding(
                isCaptureBracketOpen: true,
                session: session,
                modelState: .receiving(WorldSnapshot())
            )
            XCTAssertEqual(binding, .foreign(captureID: nil), "\(source) was bound to this phone")
        }
    }

    /// With no bracket open the phone has nothing to compare against, so the
    /// Tower's own state is the whole answer — which is what makes Stop, and a
    /// Release build with no capture control at all, behave exactly as before.
    func testWithNoBracketOpenTheTowersOwnStateIsWhatIsShown() {
        let snapshot = WorldSnapshot(worldID: "w1", keyframeCount: 44)
        for state in [
            WorldModelState.idle,
            .receiving(snapshot),
            .finalizing(snapshot),
            .finalizing(snapshot, buildInProgress: true),
            .finalized(snapshot),
            .interrupted(snapshot, reason: "the process building this world exited"),
        ] {
            let binding = WorldSessionGate.binding(
                isCaptureBracketOpen: false, session: finishedEarlier, modelState: state
            )
            XCTAssertEqual(binding, WorldSessionBinding.none)
            XCTAssertEqual(WorldSessionGate.presented(state, binding: binding), state)
        }
    }

    /// The gate decides *whose world this is*. It must never decide whether the
    /// Tower is broken: `.unsupported` and `.failed` are reports about the other
    /// machine, and swallowing either into "waiting" would hide a real fault
    /// behind a spinner.
    func testTheGateNeverSwallowsAWordAboutTheTowerItself() {
        let unsupported = WorldModelState.unsupported(reason: "no world root is configured")
        let failed = WorldModelState.failed(
            CartridgeFailure(kind: .towerReportedFailure, message: "the builder died")
        )
        for binding in [
            WorldSessionBinding.foreign(captureID: "other"),
            .awaiting(captureID: nil),
        ] {
            XCTAssertEqual(WorldSessionGate.presented(unsupported, binding: binding), unsupported)
            XCTAssertEqual(WorldSessionGate.presented(failed, binding: binding), failed)
        }
    }
}

// MARK: - The gate over a real socket

/// The session gate as the physical iPhone exercises it: a bracket opened by
/// `stream_start`, snapshots arriving on the result channel, and Stop.
@MainActor
final class TowerWorldBuilderSessionBindingTests: XCTestCase {

    /// Written out rather than read from `WorldBuilderResultContract`, so this
    /// suite pins the string the Tower actually offers instead of agreeing with
    /// whatever the app happens to hold.
    private static let contract = "world_builder.status/2026-09-06"
    private static let ourCapture = "6bf1c84c92f94fb68db62d5ba24c3ad2"
    private static let earlierCapture = "2e6cff0d1a3b4c5d6e7f8091a2b3c4d5"

    private func url(port: UInt16) -> URL { URL(string: "ws://127.0.0.1:\(port)/")! }

    private func serve(_ server: MockTowerServer, contract: String = "world_builder.status/2026-09-06") {
        server.onText = { text in
            guard
                let data = text.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let type = json["type"] as? String
            else { return }
            switch type {
            case "ping":
                server.send(text: #"{"type":"pong"}"#)
            case "cartridges":
                server.send(text: """
                    {"type":"cartridges",
                     "envelope_contract":"cartridge_results.envelope/2026-08-23",
                     "cartridges":[{"cartridge":"world_builder","result_type":"status",
                        "contract":"\(contract)","available":true,
                        "unavailable_reason":null,"snapshot_only":true}],
                     "not_offered":[]}
                    """)
            case "result_subscribe":
                server.send(text: """
                    {"type":"result_subscribed",
                     "envelope_contract":"cartridge_results.envelope/2026-08-23",
                     "subscription_id":"sub-1","cartridge":"world_builder",
                     "result_type":"status","contract":"\(contract)",
                     "snapshot_only":true,"world_id":null,"session_id":null,
                     "cursor_status":"absent"}
                    """)
            default:
                break
            }
        }
    }

    /// A whole payload, evidence blocks included — which is what the Tower
    /// actually sends and what the gate reads.
    private func snapshotMessage(
        seq: Int,
        modelState: String,
        worldID: String,
        keyframes: Int,
        revision: String,
        captureID: String?,
        endedAt: Double?,
        frameSource: String = "live-capture",
        geometryElements: Int?,
        poseCount: Int?,
        posesAnchor: Int?,
        segments: Int?
    ) -> String {
        func number(_ value: Int?) -> String { value.map(String.init) ?? "null" }
        let capture = captureID.map { "\"\($0)\"" } ?? "null"
        let ended = endedAt.map { String($0) } ?? "null"
        return """
            {"type":"cartridge_result",
             "envelope_contract":"cartridge_results.envelope/2026-08-23",
             "subscription_id":"sub-1","cartridge":"world_builder","result_type":"status",
             "contract":"\(Self.contract)","seq":\(seq),"revision":"\(revision)",
             "revision_changed":true,"coalesced":0,"cursor_status":null,
             "snapshot":true,"tower_sent_at":1787463092.9,"time_basis":"tower-receipt",
             "payload":{
               "session":{"session_id":"s-\(worldID)","started_at":1787463000.0,
                          "ended_at":\(ended),"end_reason":null,
                          "frame_source":"\(frameSource)","capture_id":\(capture),
                          "retains_raw_imagery":true},
               "trajectory":{"available":true,"current":true,
                             "built_from_keyframes":\(keyframes),"keyframes_now":\(keyframes),
                             "stale_reason":null,"pose_count":\(number(poseCount)),
                             "poses_solved":0,"poses_refused":0,
                             "poses_anchor":\(number(posesAnchor)),
                             "keyframes":\(keyframes),"segments":\(number(segments)),
                             "path_length":{"available":false,"reason":"more than one segment"},
                             "revision":"t-\(revision)","provenance":"inferred",
                             "confidence":null,"unavailable_reason":null},
               "model_state":"\(modelState)","model_state_reason":null,
               "world_snapshot":{"name":null,"world_id":"\(worldID)",
                 "keyframe_count":\(keyframes),"revision":"\(revision)",
                 "tracking":"good","scale":"relative","mapping_seconds":12.5,
                 "calibration":"uncalibrated",
                 "geometry":{"representation":"sparse point cloud",
                             "element_count":\(number(geometryElements)),
                             "is_incremental":false},
                 "trajectory":{"pose_count":\(number(poseCount)),"path_length":null,
                               "path_length_unit":null,"scale":"unknown"},
                 "persistence":{"state":"saved","revision":"p1"}}}}
            """
    }

    private func expect(
        _ message: @autoclosure () -> String = "the condition was never met",
        timeout: TimeInterval = 3,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ condition: @MainActor () -> Bool
    ) async {
        let deadline = Date().addingTimeInterval(timeout)
        var met = false
        while Date() < deadline {
            if condition() { met = true; break }
            try? await Task.sleep(nanoseconds: 25_000_000)
        }
        XCTAssertTrue(met || condition(), message(), file: file, line: line)
    }

    /// **The physical bug, end to end.** The bracket is open, the Tower answers
    /// with a finished world from an earlier capture, and the phone must show
    /// "waiting" — never a frozen world beside a live camera.
    func testAFinishedWorldFromAnotherCaptureIsNotRenderedWhileThisPhoneStreams() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        tower.sendStreamStart()
        await expect { tower.isStreamingToTower }

        server.send(text: snapshotMessage(
            seq: 1, modelState: "finalized", worldID: "w-earlier", keyframes: 143,
            revision: "r1", captureID: Self.earlierCapture, endedAt: 1787462800.0,
            geometryElements: 0, poseCount: 0, posesAnchor: 36, segments: 36
        ))
        await expect { client.sessionBinding == .foreign(captureID: Self.earlierCapture) }

        XCTAssertEqual(
            client.state, .awaitingFirstUpdate,
            "a finished world from another capture was rendered as this session's"
        )
        XCTAssertFalse(client.state.hasWorld)

        tower.sendStreamStop()
        tower.disconnect()
    }

    /// Goal of the whole lifecycle change: while the wearer walks, the world
    /// grows on screen. Keyframes climb and geometry appears mid-walk, because
    /// the Tower now rebuilds every four keyframes instead of once at the end.
    func testTheWorldGrowsOnScreenWhileTheBracketIsOpen() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        tower.sendStreamStart()
        await expect { tower.isStreamingToTower }

        // Four live rebuilds of the capture this phone opened.
        for (index, keyframes) in [28, 32, 36, 40].enumerated() {
            server.send(text: snapshotMessage(
                seq: index + 1, modelState: "receiving", worldID: "w-live",
                keyframes: keyframes, revision: "r\(keyframes)",
                captureID: Self.ourCapture, endedAt: nil,
                geometryElements: 0, poseCount: 0, posesAnchor: 3, segments: 3
            ))
            await expect { client.state.snapshot?.keyframeCount == keyframes }
            XCTAssertTrue(client.state.isReceivingUpdates, "a live rebuild stopped reading as live")
        }

        XCTAssertEqual(client.sessionBinding, .bound(captureID: Self.ourCapture))
        XCTAssertEqual(client.state.snapshot?.trajectory.posesAnchor, 3)
        XCTAssertEqual(client.state.snapshot?.trajectory.segments, 3)
        XCTAssertEqual(
            client.state.snapshot?.trajectory.poseCount, 0,
            "an uncalibrated walk reported camera positions it did not have"
        )

        tower.sendStreamStop()
        tower.disconnect()
    }

    /// Stop closes the bracket, and the Tower's own words are then the whole
    /// answer: `stopped_unbuilt` is `.finalizing`, `ready` is `.finalized`.
    func testStopThenFinalizingThenFinalized() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        tower.sendStreamStart()
        await expect { tower.isStreamingToTower }
        server.send(text: snapshotMessage(
            seq: 1, modelState: "receiving", worldID: "w-live", keyframes: 44,
            revision: "r44", captureID: Self.ourCapture, endedAt: nil,
            geometryElements: 0, poseCount: 0, posesAnchor: 3, segments: 3
        ))
        await expect { client.state.isReceivingUpdates }

        tower.sendStreamStop()
        await expect { client.sessionBinding == WorldSessionBinding.none }

        server.send(text: snapshotMessage(
            seq: 2, modelState: "finalizing", worldID: "w-live", keyframes: 44,
            revision: "r45", captureID: Self.ourCapture, endedAt: 1787463122.0,
            geometryElements: 0, poseCount: 0, posesAnchor: 3, segments: 3
        ))
        await expect {
            if case .finalizing = client.state { return true }
            return false
        }
        XCTAssertFalse(client.state.isReceivingUpdates)

        server.send(text: snapshotMessage(
            seq: 3, modelState: "finalized", worldID: "w-live", keyframes: 44,
            revision: "r46", captureID: Self.ourCapture, endedAt: 1787463122.0,
            geometryElements: 812, poseCount: 41, posesAnchor: 3, segments: 3
        ))
        await expect {
            if case .finalized = client.state { return true }
            return false
        }
        XCTAssertEqual(client.state.phase, .settled)
        XCTAssertEqual(client.state.snapshot?.geometry.elementCount, 812)
        XCTAssertEqual(client.state.snapshot?.trajectory.poseCount, 41)

        tower.disconnect()
    }

    /// The binding has two inputs and only one of them arrives on the wire.
    /// Pressing Start re-judges the snapshot already on screen: a finished
    /// world that was a legitimate thing to show a moment ago stops being one
    /// the instant this phone opens a capture of its own.
    func testPressingStartRejudgesTheWorldAlreadyOnScreen() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        // No bracket: the Tower's own state is the whole answer, and a finished
        // world is a perfectly good thing to be looking at.
        server.send(text: snapshotMessage(
            seq: 1, modelState: "finalized", worldID: "w-earlier", keyframes: 143,
            revision: "r1", captureID: Self.earlierCapture, endedAt: 1787462800.0,
            geometryElements: 2336, poseCount: 137, posesAnchor: 6, segments: 6
        ))
        await expect {
            if case .finalized = client.state { return true }
            return false
        }
        XCTAssertEqual(client.sessionBinding, WorldSessionBinding.none)

        // Start. Nothing new arrives from the Tower, and the same snapshot must
        // stop being rendered as this session's.
        tower.sendStreamStart()
        await expect { client.state == .awaitingFirstUpdate }
        XCTAssertEqual(client.sessionBinding, .foreign(captureID: Self.earlierCapture))

        // Stop, and it is a saved world again rather than a wrong one.
        tower.sendStreamStop()
        await expect {
            if case .finalized = client.state { return true }
            return false
        }
        XCTAssertEqual(client.state.snapshot?.keyframeCount, 143)
        XCTAssertEqual(client.sessionBinding, WorldSessionBinding.none)

        tower.disconnect()
    }

    /// A Tower still offering the superseded contract is not decoded on a
    /// guess. `.unsupportedContract` tells a person to update the app — which
    /// is exactly the message the physical iPhone showed for the *new* contract
    /// before this change, with the two identifiers the other way round.
    func testTheSupersededContractIsNeitherSubscribedToNorDecoded() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, contract: "world_builder.status/2026-08-23")
        let inner = server.onText
        server.onText = { text in
            recorder.record(text)
            inner?(text)
        }
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { tower.cartridgeDeclaration != nil }
        try? await Task.sleep(nanoseconds: 250_000_000)

        XCTAssertFalse(
            recorder.all.contains { $0.contains("result_subscribe") },
            "the client subscribed against the superseded contract"
        )
        XCTAssertEqual(
            client.availability(isTowerReachable: true).forcedPhase,
            .needsUpdate,
            "the superseded contract was treated as one this build implements"
        )

        tower.disconnect()
    }
}

// MARK: - The interactive picture (`GET /worlds/{id}/render`)

/// The viewer's fetch, its address, its sentences and its navigation policy,
/// with no `WKWebView` in any of them. The page itself is the Tower's; what
/// this app owns is asking for it correctly, saying truthfully why it did not
/// come, and refusing to go anywhere else.
@MainActor
final class WorldRenderViewerTests: XCTestCase {

    private static let host = URL(string: "http://stub.invalid")!

    private func client() -> WorldRenderClient {
        WorldRenderClient(baseURL: Self.host, session: StubbedGeometryProtocol.makeSession())
    }

    // MARK: Address

    func testTheAddressIsTheContractRouteWithTheSessionAsAQuery() {
        let pinned = WorldRenderClient.url(
            for: WorldRenderTarget(worldID: "w1", sessionID: "s1"), baseURL: Self.host
        )
        XCTAssertEqual(pinned?.absoluteString, "http://stub.invalid/worlds/w1/render?session_id=s1")

        let unpinned = WorldRenderClient.url(
            for: WorldRenderTarget(worldID: "w1", sessionID: nil), baseURL: Self.host
        )
        XCTAssertEqual(unpinned?.absoluteString, "http://stub.invalid/worlds/w1/render")
    }

    /// A wire-supplied id stays one path component. A `/` inside it must not
    /// become a second segment that addresses a different route.
    func testAnIdIsOnePathComponentHoweverItIsSpelled() {
        let url = WorldRenderClient.url(
            for: WorldRenderTarget(worldID: "a/b c", sessionID: "s&1"), baseURL: Self.host
        )
        XCTAssertEqual(url?.path, "/worlds/a/b c/render")
        XCTAssertEqual(url?.absoluteString, "http://stub.invalid/worlds/a%2Fb%20c/render?session_id=s%261")
    }

    func testAnEmptyIdIsRefusedRatherThanAddressed() {
        XCTAssertNil(WorldRenderClient.url(
            for: WorldRenderTarget(worldID: "", sessionID: nil), baseURL: Self.host
        ))
        XCTAssertNil(WorldRenderClient.url(
            for: WorldRenderTarget(worldID: "w1", sessionID: ""), baseURL: Self.host
        ))
    }

    // MARK: Fetch

    func testAPageComesBackAsItsText() async throws {
        StubbedGeometryProtocol.reset(routes: [
            "/worlds/w1/render": (200, "<!doctype html><html><body>hi</body></html>"),
        ])
        let html = try await client().page(for: WorldRenderTarget(worldID: "w1", sessionID: "s1"))
        XCTAssertTrue(html.hasPrefix("<!doctype html>"))
        XCTAssertEqual(StubbedGeometryProtocol.requestCount(for: "/worlds/w1/render"), 1)
    }

    /// The Tower's 404 says WHICH thing is missing, and the phone keeps it.
    func testAbsenceCarriesTheTowersOwnDetail() async {
        StubbedGeometryProtocol.reset(routes: [
            "/worlds/w1/render": (404, #"{"detail": "world 'w1' has no session with geometry yet"}"#),
        ])
        do {
            _ = try await client().page(for: WorldRenderTarget(worldID: "w1", sessionID: nil))
            XCTFail("a 404 must throw")
        } catch let error as WorldRenderFetchError {
            XCTAssertEqual(error, .absent(detail: "world 'w1' has no session with geometry yet"))
            XCTAssertTrue(error.message.contains("no session with geometry yet"))
            XCTAssertFalse(error.message.contains("yet:"), "the detail says whether it is a 'yet'; the sentence must not")
            XCTAssertTrue(error.isRetryable, "a world still being built answers 404 until its first solve lands")
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    /// FastAPI's "Not Found" is a Tower without the route, not a world
    /// without geometry, and trying again will not grow it a route.
    func testAnUnmatchedRouteIsNamedAsSuchAndNotRetried() async {
        StubbedGeometryProtocol.reset(routes: ["/worlds/w1/render": (404, #"{"detail": "Not Found"}"#)])
        do {
            _ = try await client().page(for: WorldRenderTarget(worldID: "w1", sessionID: nil))
            XCTFail("a 404 must throw")
        } catch let error as WorldRenderFetchError {
            XCTAssertEqual(error, .absent(detail: "Not Found"))
            XCTAssertTrue(error.message.contains("does not serve the picture route"))
            XCTAssertFalse(error.message.contains("no picture for this world"))
            XCTAssertFalse(error.isRetryable)
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testA404WithNoReadableBodyIsStillAbsent() async {
        StubbedGeometryProtocol.reset(routes: ["/worlds/w1/render": (404, "<html>nope</html>")])
        do {
            _ = try await client().page(for: WorldRenderTarget(worldID: "w1", sessionID: nil))
            XCTFail("a 404 must throw")
        } catch let error as WorldRenderFetchError {
            XCTAssertEqual(error, .absent(detail: nil))
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testAnotherStatusIsATowerError() async {
        StubbedGeometryProtocol.reset(routes: ["/worlds/w1/render": (500, "")])
        do {
            _ = try await client().page(for: WorldRenderTarget(worldID: "w1", sessionID: nil))
            XCTFail("a 500 must throw")
        } catch let error as WorldRenderFetchError {
            XCTAssertEqual(error, .towerError(status: 500))
            XCTAssertTrue(error.message.contains("500"))
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testAnUnreachableTowerIsATransportFailureNotAMissingWorld() async {
        StubbedGeometryProtocol.reset(routes: [:])
        do {
            _ = try await client().page(for: WorldRenderTarget(worldID: "w1", sessionID: nil))
            XCTFail("a dropped connection must throw")
        } catch let error as WorldRenderFetchError {
            guard case .transport = error else { return XCTFail("wrong error: \(error)") }
            XCTAssertFalse(error.message.contains("no picture for this world"),
                           "a link that dropped must not be blamed on the world")
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    // MARK: The sheet's model

    func testTheModelReportsLoadingThenReadyOrFailed() async {
        StubbedGeometryProtocol.reset(routes: ["/worlds/w1/render": (200, "<!doctype html>")])
        let model = WorldRenderViewerModel(
            target: WorldRenderTarget(worldID: "w1", sessionID: nil), client: client()
        )
        XCTAssertEqual(model.state, .loading)
        await model.load()
        XCTAssertEqual(model.state, .ready(html: "<!doctype html>"))

        StubbedGeometryProtocol.set(route: "/worlds/w1/render", to: (404, #"{"detail": "no world 'w1'"}"#))
        await model.load()
        guard case .failed(let message, let retryable) = model.state else {
            return XCTFail("expected failed, got \(model.state)")
        }
        XCTAssertTrue(message.contains("no world 'w1'"))
        XCTAssertTrue(retryable)
    }

    // MARK: Navigation policy

    /// The page is loaded as a string with no base URL, so its one legitimate
    /// navigation is the initial `about:blank`. Everything else is refused.
    func testOnlyTheInitialBlankNavigationIsAllowed() {
        XCTAssertTrue(WorldRenderNavigationPolicy.allows(URL(string: "about:blank"), isInitialLoad: true))
        XCTAssertFalse(WorldRenderNavigationPolicy.allows(URL(string: "about:blank"), isInitialLoad: false),
                       "a link click to about:blank is still a link click")
        XCTAssertFalse(WorldRenderNavigationPolicy.allows(URL(string: "http://stub.invalid/"), isInitialLoad: true))
        XCTAssertFalse(WorldRenderNavigationPolicy.allows(URL(string: "https://example.com/"), isInitialLoad: true))
        XCTAssertFalse(WorldRenderNavigationPolicy.allows(URL(string: "file:///etc/passwd"), isInitialLoad: true))
        XCTAssertFalse(WorldRenderNavigationPolicy.allows(URL(string: "about:srcdoc"), isInitialLoad: true))
        XCTAssertFalse(WorldRenderNavigationPolicy.allows(nil, isInitialLoad: true))
    }

    // MARK: Where the picture button gets its target

    /// Opening a stored world names it for the viewer at once — the person
    /// chose it, and if the Tower has nothing built the viewer says so in
    /// the Tower's words. Returning to live forgets it until the Tower names
    /// the live world's geometry.
    func testOpeningAStoredWorldNamesItAndReturningToLiveForgetsIt() {
        let viewModel = WorldBuilderViewModel(client: UnavailableWorldBuilderClient())
        XCTAssertNil(viewModel.renderTarget, "nothing has been named yet")

        viewModel.open(worldID: "w-old", sessionID: nil)
        XCTAssertEqual(viewModel.renderTarget, WorldRenderTarget(worldID: "w-old", sessionID: nil))

        viewModel.open(worldID: "w-old", sessionID: "s-2")
        XCTAssertEqual(viewModel.renderTarget, WorldRenderTarget(worldID: "w-old", sessionID: "s-2"))

        viewModel.returnToLive()
        XCTAssertNil(viewModel.renderTarget)
    }

    /// The live world earns a picture only when the Tower names geometry for
    /// it — and keeps it even when the manifest that follows cannot be
    /// fetched, because the render route is a different question.
    func testGeometryCoordinatesNameTheLiveWorldEvenWhenTheManifestFails() async {
        StubbedGeometryProtocol.reset(routes: [:])  // every fetch fails
        let viewModel = WorldBuilderViewModel(
            client: UnavailableWorldBuilderClient(),
            geometry: WorldGeometryClient(
                baseURL: Self.host, session: StubbedGeometryProtocol.makeSession()
            )
        )
        await viewModel.geometryDidChange(worldID: "w-live", sessionID: "s-live", revision: "g1")
        XCTAssertEqual(viewModel.renderTarget, WorldRenderTarget(worldID: "w-live", sessionID: "s-live"))

        // A heartbeat under the same revision changes nothing and publishes
        // nothing new.
        var publishes = 0
        let cancellable = viewModel.$renderTarget.dropFirst().sink { _ in publishes += 1 }
        await viewModel.geometryDidChange(worldID: "w-live", sessionID: "s-live", revision: "g1")
        XCTAssertEqual(publishes, 0)
        cancellable.cancel()

        // A half-known address names nothing, exactly as it fetches nothing.
        viewModel.returnToLive()
        await viewModel.geometryDidChange(worldID: "w-live", sessionID: nil, revision: "g2")
        XCTAssertNil(viewModel.renderTarget)
    }
}


// MARK: - The 2026-09-06 contract: interrupted, selection, finalization

/// The decode half of `world_builder.status/2026-09-06`, with no socket in
/// it. The new word, the new block that says why a world is on the wire, and
/// the builder's finalization record.
@MainActor
final class WorldBuilderContract20260906PayloadTests: XCTestCase {

    private func snapshotJSON(worldID: String = "fcbca9e90b244785bdb671530b33c6a5") -> [String: Any] {
        [
            "name": NSNull(),
            "world_id": worldID,
            "keyframe_count": 463,
            "revision": "r-463",
            "tracking": "good",
            "scale": "relative",
            "mapping_seconds": 176.0,
            "calibration": "uncalibrated",
            "geometry": ["representation": "sparse point cloud", "element_count": 17_674, "is_incremental": false],
            "trajectory": ["pose_count": 467, "path_length": NSNull(), "path_length_unit": NSNull(), "scale": "unknown"],
            "persistence": ["state": "saved", "revision": "p1"],
        ]
    }

    /// The payload the Tower builds for the 2026-09-06 walk's world: a dead
    /// lock, 463 keyframes of geometry, and the word for it.
    private func interruptedPayload(
        selection: [String: Any]? = ["mode": "pinned", "world_id": "fcbca9e90b244785bdb671530b33c6a5",
                                     "session_id": "158ef0efb5e5416b87d6faa8f5c28e55",
                                     "reason": "the client named this world"],
        finalization: Any = NSNull(),
        snapshot: [String: Any]? = nil
    ) -> [String: Any] {
        var payload: [String: Any] = [
            "model_state": "interrupted",
            "model_state_reason": "the process building this world exited without stopping its session; its keyframes are persisted but the session was never closed",
            "world_snapshot": (snapshot ?? snapshotJSON()) as Any,
            "world": ["world_id": "fcbca9e90b244785bdb671530b33c6a5", "display_name": NSNull(),
                      "schema_version": 1, "created_at": 1788894856.2, "updated_at": 1788895032.8],
            "session": ["session_id": "158ef0efb5e5416b87d6faa8f5c28e55", "started_at": 1788894857.46,
                        "ended_at": NSNull(), "end_reason": NSNull(), "frame_source": "live-capture",
                        "capture_id": "7febdae8", "retains_raw_imagery": true],
            "lifecycle": ["state": "interrupted",
                          "evidence": "the writer lock is held by pid 19604, which is no longer running",
                          "reason": "the process building this world exited without stopping its session; its keyframes are persisted but the session was never closed",
                          "build_in_progress": false,
                          "build_in_progress_unavailable_reason": NSNull(),
                          "finalization": finalization],
            "geometry": ["available": true, "current": false, "revision": "g-463"],
        ]
        if let selection { payload["selection"] = selection }
        return payload
    }

    // MARK: interrupted

    /// `interrupted` keeps its snapshot and carries the Tower's reason. It is
    /// neither `.failed` (no world) nor `.finalized` (a finished one).
    func testAnInterruptedWorldKeepsItsSnapshotAndTheTowersReason() throws {
        let state = try XCTUnwrap(WorldBuilderResultDecoder.modelState(from: interruptedPayload()))
        guard case .interrupted(let snapshot, let reason) = state else {
            return XCTFail("interrupted decoded as \(state)")
        }
        XCTAssertEqual(snapshot.keyframeCount, 463)
        XCTAssertEqual(snapshot.worldID, "fcbca9e90b244785bdb671530b33c6a5")
        XCTAssertEqual(snapshot.geometry.elementCount, 17_674)
        XCTAssertTrue(reason.hasPrefix("the process building this world exited"))

        XCTAssertTrue(state.hasWorld)
        XCTAssertNotNil(state.snapshot)
        XCTAssertFalse(state.isReceivingUpdates, "an interrupted world claimed live updates")
        XCTAssertEqual(state.phase, .settled)
        XCTAssertTrue(state.phase.mayCarryData)
    }

    /// The word arrived with no world behind it. Nothing to draw, so it is a
    /// failure the Tower reported — never an empty world.
    func testAnInterruptedPayloadWithoutASnapshotIsAFailureNotAnEmptyWorld() throws {
        var payload = interruptedPayload()
        payload["world_snapshot"] = NSNull()
        let state = try XCTUnwrap(WorldBuilderResultDecoder.modelState(from: payload))
        guard case .failed(let failure) = state else { return XCTFail("decoded as \(state)") }
        XCTAssertEqual(failure.kind, .towerReportedFailure)
        XCTAssertTrue(failure.message.hasPrefix("the process building this world exited"))
    }

    /// Without a reason the headline still has a sentence, and it says the
    /// Tower did not explain rather than inventing an explanation.
    func testAnInterruptedWorldWithoutAReasonStillHasASentence() throws {
        var payload = interruptedPayload()
        payload["model_state_reason"] = NSNull()
        let state = try XCTUnwrap(WorldBuilderResultDecoder.modelState(from: payload))
        guard case .interrupted(_, let reason) = state else { return XCTFail("decoded as \(state)") }
        XCTAssertEqual(reason, WorldBuilderResultDecoder.unexplainedInterruption)
    }

    /// The geometry address travels with the interrupted word exactly as it
    /// does with `finalized`: the reconstruction exists whatever happened to
    /// the process.
    func testAnInterruptedWorldStillAddressesItsGeometry() {
        let coordinates = WorldBuilderResultDecoder.geometryCoordinates(from: interruptedPayload())
        XCTAssertEqual(
            coordinates,
            WorldGeometryCoordinates(
                worldID: "fcbca9e90b244785bdb671530b33c6a5",
                sessionID: "158ef0efb5e5416b87d6faa8f5c28e55",
                revision: "g-463"
            )
        )
    }

    /// `failed` is still `failed`: a Tower that says so with no geometry is
    /// not upgraded to interrupted on the phone's guess.
    func testAFailedWorldWithoutGeometryStaysFailed() throws {
        var payload = interruptedPayload()
        payload["model_state"] = "failed"
        payload["model_state_reason"] = "the builder died"
        let state = try XCTUnwrap(WorldBuilderResultDecoder.modelState(from: payload))
        guard case .failed(let failure) = state else { return XCTFail("decoded as \(state)") }
        XCTAssertEqual(failure.message, "the builder died")
    }

    // MARK: finalizing

    /// `lifecycle.build_in_progress` rides on `.finalizing`: `true` on the
    /// evidence of a live lock, `null` on an older record, and the two are
    /// worded differently on the canvas.
    func testFinalizingCarriesWhetherABuildIsInProgress() throws {
        var payload = interruptedPayload()
        payload["model_state"] = "finalizing"
        payload["lifecycle"] = ["state": "finalizing", "build_in_progress": true, "finalization": NSNull()]
        var state = try XCTUnwrap(WorldBuilderResultDecoder.modelState(from: payload))
        guard case .finalizing(_, let building) = state else { return XCTFail("decoded as \(state)") }
        XCTAssertEqual(building, true)

        payload["lifecycle"] = ["state": "stopped_unbuilt", "build_in_progress": NSNull(), "finalization": NSNull()]
        state = try XCTUnwrap(WorldBuilderResultDecoder.modelState(from: payload))
        guard case .finalizing(_, let unknown) = state else { return XCTFail("decoded as \(state)") }
        XCTAssertNil(unknown, "an older record's null became a claim")

        payload["lifecycle"] = nil
        state = try XCTUnwrap(WorldBuilderResultDecoder.modelState(from: payload))
        guard case .finalizing(_, let absent) = state else { return XCTFail("decoded as \(state)") }
        XCTAssertNil(absent)

        // The default keeps every existing construction site compiling and
        // means the same thing as the Tower's null.
        XCTAssertEqual(
            WorldModelState.finalizing(WorldSnapshot()),
            .finalizing(WorldSnapshot(), buildInProgress: nil)
        )
    }

    // MARK: selection

    func testTheSelectionBlockIsDecodedFieldForField() {
        let selection = WorldBuilderResultDecoder.selection(from: interruptedPayload(
            selection: ["mode": "latest", "world_id": "w-newest", "session_id": "s-newest",
                        "reason": "nothing is live; this is the most recently updated world"]
        ))
        XCTAssertEqual(selection.mode, .latest)
        XCTAssertEqual(selection.worldID, "w-newest")
        XCTAssertEqual(selection.sessionID, "s-newest")
        XCTAssertEqual(selection.reason, "nothing is live; this is the most recently updated world")
        XCTAssertTrue(selection.isHistoryOfferedAsLive)
        XCTAssertFalse(selection.isCurrentWorld)
        XCTAssertTrue(selection.mode.isRecognised)

        for word in ["live", "finalizing"] {
            let current = WorldBuilderResultDecoder.selection(from: interruptedPayload(
                selection: ["mode": word, "world_id": "w", "session_id": NSNull(), "reason": "r"]
            ))
            XCTAssertTrue(current.isCurrentWorld, word)
            XCTAssertFalse(current.isHistoryOfferedAsLive, word)
            XCTAssertNil(current.sessionID)
        }

        let none = WorldBuilderResultDecoder.selection(from: [
            "model_state": "idle",
            "selection": ["mode": "none", "world_id": NSNull(), "session_id": NSNull(), "reason": "no worlds exist"],
        ])
        XCTAssertEqual(none.mode, WorldSelectionMode.none)
        XCTAssertNil(none.worldID)
        XCTAssertFalse(none.isCurrentWorld)
        XCTAssertFalse(none.isHistoryOfferedAsLive)
    }

    /// No block is `.unknown` — an older Tower — and `.unknown` is not any of
    /// the five words. A block with no `mode` is the same.
    func testAnAbsentSelectionIsUnknownNotNone() {
        let absent = WorldBuilderResultDecoder.selection(from: interruptedPayload(selection: nil))
        XCTAssertEqual(absent, .unknown)
        XCTAssertEqual(absent.mode, .unknown)
        XCTAssertFalse(absent.mode.isRecognised)
        XCTAssertNotEqual(absent.mode, WorldSelectionMode.none)
        XCTAssertFalse(absent.isCurrentWorld)
        XCTAssertFalse(absent.isHistoryOfferedAsLive)

        let modeless = WorldBuilderResultDecoder.selection(from: interruptedPayload(
            selection: ["world_id": "w", "reason": "r"]
        ))
        XCTAssertEqual(modeless, .unknown)
    }

    /// A sixth word arrives as itself. It is neither current nor history and
    /// the client treats it as it treats an older Tower.
    func testAnUnknownSelectionWordSurvivesAsItself() {
        let odd = WorldBuilderResultDecoder.selection(from: interruptedPayload(
            selection: ["mode": "replaying", "world_id": "w", "session_id": "s", "reason": "r"]
        ))
        XCTAssertEqual(odd.mode.rawValue, "replaying")
        XCTAssertFalse(odd.mode.isRecognised)
        XCTAssertNotEqual(odd.mode, .unknown)
        XCTAssertFalse(odd.isCurrentWorld)
        XCTAssertFalse(odd.isHistoryOfferedAsLive)
    }

    // MARK: finalization

    func testTheFinalizationRecordIsDecodedAndNullIsNil() throws {
        XCTAssertNil(WorldBuilderResultDecoder.finalization(from: interruptedPayload()))
        XCTAssertNil(WorldBuilderResultDecoder.finalization(from: ["model_state": "idle"]))

        let report = try XCTUnwrap(WorldBuilderResultDecoder.finalization(from: interruptedPayload(
            finalization: ["state": "interrupted", "final_solve": "skipped",
                           "started_at": 1788895053.0, "updated_at": 1788895060.5,
                           "detail": "stdin closed while observing"]
        )))
        XCTAssertEqual(report.state, .interrupted)
        XCTAssertEqual(report.finalSolve, "skipped")
        XCTAssertEqual(report.startedAt, 1788895053.0)
        XCTAssertEqual(report.updatedAt, 1788895060.5)
        XCTAssertEqual(report.detail, "stdin closed while observing")

        let pending = try XCTUnwrap(WorldBuilderResultDecoder.finalization(from: interruptedPayload(
            finalization: ["state": "pending", "final_solve": NSNull(), "started_at": 1.0,
                           "updated_at": 1.0, "detail": NSNull()]
        )))
        XCTAssertEqual(pending.state, .pending)
        XCTAssertNil(pending.finalSolve)
        XCTAssertNil(pending.detail)

        // A block with no state is not a report.
        XCTAssertNil(WorldBuilderResultDecoder.finalization(from: interruptedPayload(
            finalization: ["final_solve": "solved"]
        )))
    }

    // MARK: the recent-world reference

    func testTheRecentReferenceIsBuiltFromALatestSelection() throws {
        var payload = interruptedPayload(
            selection: ["mode": "latest", "world_id": "fcbca9e90b244785bdb671530b33c6a5",
                        "session_id": "158ef0efb5e5416b87d6faa8f5c28e55", "reason": "nothing is live"]
        )
        let reference = try XCTUnwrap(WorldBuilderResultDecoder.recentReference(from: payload))
        XCTAssertEqual(reference.worldID, "fcbca9e90b244785bdb671530b33c6a5")
        XCTAssertEqual(reference.sessionID, "158ef0efb5e5416b87d6faa8f5c28e55")
        XCTAssertNil(reference.name)
        XCTAssertEqual(reference.modelState, "interrupted")
        XCTAssertEqual(reference.stateLabel, "interrupted")
        XCTAssertEqual(reference.updatedAt, 1788895032.8)
        XCTAssertEqual(reference.title, WorldListingPresentation.datedTitle(updatedAt: 1788895032.8))

        var snapshot = snapshotJSON()
        snapshot["name"] = "Kitchen walk"
        payload["world_snapshot"] = snapshot
        payload["model_state"] = "finalized"
        let named = try XCTUnwrap(WorldBuilderResultDecoder.recentReference(from: payload))
        XCTAssertEqual(named.title, "Kitchen walk")
        XCTAssertEqual(named.stateLabel, "finished")

        // No world, nothing to offer.
        XCTAssertNil(WorldBuilderResultDecoder.recentReference(from: [
            "model_state": "idle", "world_snapshot": NSNull(),
            "selection": ["mode": "none", "world_id": NSNull(), "session_id": NSNull(), "reason": "none"],
        ]))
    }

    // MARK: the identifier

    func testTheContractIdentifierIsTheNewOneAndTheOldOneIsNoLongerClaimed() {
        XCTAssertEqual(WorldBuilderResultContract.identifier, "world_builder.status/2026-09-06")
        XCTAssertTrue(TowerCapabilities.supported.contains("world_builder.status/2026-09-06"))
        XCTAssertFalse(TowerCapabilities.supported.contains("world_builder.status/2026-08-25"))
        XCTAssertFalse(TowerCapabilities.supported.contains("world_builder.status/2026-08-23"))
    }

    /// The gate treats `.interrupted` as a world state: foreign while a
    /// bracket is open, passed through with none.
    func testTheGateTreatsAnInterruptedWorldAsAWorldState() {
        let state = WorldModelState.interrupted(WorldSnapshot(worldID: "w"), reason: "r")
        XCTAssertEqual(WorldSessionGate.presented(state, binding: WorldSessionBinding.none), state)
        XCTAssertEqual(WorldSessionGate.presented(state, binding: .foreign(captureID: "c")), .awaitingFirstUpdate)
        XCTAssertEqual(WorldSessionGate.presented(state, binding: .awaiting(captureID: nil)), .awaitingFirstUpdate)
        let binding = WorldSessionGate.binding(
            isCaptureBracketOpen: true,
            session: WorldSessionReport(sessionID: "s", captureID: "c", endedAt: nil, frameSource: "live-capture"),
            modelState: state
        )
        XCTAssertEqual(binding, .foreign(captureID: "c"), "an interrupted world was bound to a live capture")
    }
}

// MARK: - Live versus History over a real socket

/// The ownership rule and the C3 subscription race, against the mock Tower
/// speaking `world_builder.status/2026-09-06` with its `selection` block.
@MainActor
final class TowerWorldBuilderLiveHistoryTests: XCTestCase {

    private nonisolated static let contract = "world_builder.status/2026-09-06"
    private static let host = URL(string: "http://stub.invalid")!

    private func url(port: UInt16) -> URL { URL(string: "ws://127.0.0.1:\(port)/")! }

    /// The three exchanges before the first snapshot, with per-connection
    /// subscription numbering as the Tower does it. Every subscribe is acked,
    /// in order, which is what lets the C3 tests below open two.
    private func serve(_ server: MockTowerServer, recorder: MessageRecorder? = nil) {
        var subscribeCount = 0
        server.onText = { text in
            recorder?.record(text)
            guard
                let data = text.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let type = json["type"] as? String
            else { return }
            switch type {
            case "ping":
                server.send(text: #"{"type":"pong"}"#)
            case "cartridges":
                subscribeCount = 0
                server.send(text: """
                    {"type":"cartridges",
                     "envelope_contract":"cartridge_results.envelope/2026-08-23",
                     "cartridges":[{"cartridge":"world_builder","result_type":"status",
                        "contract":"\(Self.contract)","available":true,
                        "unavailable_reason":null,"snapshot_only":true}],
                     "not_offered":[]}
                    """)
            case "result_subscribe":
                subscribeCount += 1
                let worldID = json["world_id"] as? String
                let world = worldID.map { "\"\($0)\"" } ?? "null"
                server.send(text: """
                    {"type":"result_subscribed",
                     "envelope_contract":"cartridge_results.envelope/2026-08-23",
                     "subscription_id":"sub-\(subscribeCount)","cartridge":"world_builder",
                     "result_type":"status","contract":"\(Self.contract)",
                     "snapshot_only":true,"world_id":\(world),"session_id":null,
                     "cursor_status":"absent"}
                    """)
            default:
                break
            }
        }
    }

    /// A whole payload: the two halves iOS decodes plus the blocks the
    /// ownership rule, the gate and the geometry address read.
    private func message(
        seq: Int,
        subscription: String = "sub-1",
        modelState: String,
        worldID: String,
        name: String? = nil,
        keyframes: Int = 40,
        revision: String,
        geometryRevision: String? = nil,
        selection: String? = "live",
        captureID: String? = "cap-1",
        endedAt: Double? = nil,
        buildInProgress: Bool? = nil,
        reason: String? = nil,
        updatedAt: Double = 1788895032.0
    ) -> String {
        let sessionID = "s-\(worldID)"
        let capture = captureID.map { "\"\($0)\"" } ?? "null"
        let ended = endedAt.map { String($0) } ?? "null"
        let geometry = geometryRevision.map { "\"\($0)\"" } ?? "null"
        let building = buildInProgress.map { String($0) } ?? "null"
        let why = reason.map { "\"\($0)\"" } ?? "null"
        let displayName = name.map { "\"\($0)\"" } ?? "null"
        let selectionBlock = selection.map {
            """
            "selection":{"mode":"\($0)","world_id":"\(worldID)","session_id":"\(sessionID)",
                         "reason":"fixture"},
            """
        } ?? ""
        return """
            {"type":"cartridge_result",
             "envelope_contract":"cartridge_results.envelope/2026-08-23",
             "subscription_id":"\(subscription)","cartridge":"world_builder","result_type":"status",
             "contract":"\(Self.contract)","seq":\(seq),"revision":"\(revision)",
             "revision_changed":true,"coalesced":0,"cursor_status":null,
             "snapshot":true,"tower_sent_at":1788895032.9,"time_basis":"tower-receipt",
             "payload":{
               \(selectionBlock)
               "world":{"world_id":"\(worldID)","display_name":\(displayName),
                        "schema_version":1,"created_at":1788894856.0,"updated_at":\(updatedAt)},
               "session":{"session_id":"\(sessionID)","started_at":1788894857.0,
                          "ended_at":\(ended),"end_reason":null,
                          "frame_source":"live-capture","capture_id":\(capture),
                          "retains_raw_imagery":true},
               "lifecycle":{"state":"\(modelState)","evidence":"fixture","reason":\(why),
                            "build_in_progress":\(building),
                            "build_in_progress_unavailable_reason":null,"finalization":null},
               "geometry":{"available":\(geometryRevision != nil),"current":true,
                           "revision":\(geometry)},
               "model_state":"\(modelState)","model_state_reason":\(why),
               "world_snapshot":{"name":\(displayName),"world_id":"\(worldID)",
                 "keyframe_count":\(keyframes),"revision":"\(revision)",
                 "tracking":"good","scale":"relative","mapping_seconds":12.5,
                 "calibration":"uncalibrated",
                 "geometry":{"representation":"sparse point cloud","element_count":812,
                             "is_incremental":false},
                 "trajectory":{"pose_count":\(keyframes),"path_length":null,
                               "path_length_unit":null,"scale":"unknown"},
                 "persistence":{"state":"saved","revision":"p1"}}}}
            """
    }

    private func expect(
        _ message: @autoclosure () -> String = "the condition was never met",
        timeout: TimeInterval = 3,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ condition: @MainActor () -> Bool
    ) async {
        let deadline = Date().addingTimeInterval(timeout)
        var met = false
        while Date() < deadline {
            if condition() { met = true; break }
            try? await Task.sleep(nanoseconds: 25_000_000)
        }
        XCTAssertTrue(met || condition(), message(), file: file, line: line)
    }

    private func decode(_ text: String) -> [String: Any]? {
        guard let data = text.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private func sent(_ recorder: MessageRecorder, type: String) -> [[String: Any]] {
        recorder.all.compactMap(decode).filter { $0["type"] as? String == type }
    }

    /// A settled pause, for the assertions that something did **not** happen.
    private func settle() async {
        try? await Task.sleep(nanoseconds: 250_000_000)
    }

    // MARK: The ownership rule

    /// **The stale live canvas, end to end.** Nothing is live; the Tower
    /// answers the unpinned subscribe with the newest world on disk and says
    /// so. Following live, that is `.idle` with a "last saved world" on offer
    /// — not a finished world under the Live heading — and its geometry is
    /// never addressed.
    func testALatestSelectionInLiveModeIsIdleWithARecentWorldAndNoGeometry() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        var coordinates: [WorldGeometryCoordinates] = []
        let cancellable = client.geometryUpdates.sink { coordinates.append($0) }
        defer { cancellable.cancel() }
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        server.send(text: message(
            seq: 1, modelState: "finalized", worldID: "w-prev", keyframes: 143,
            revision: "r1", geometryRevision: "g1", selection: "latest",
            endedAt: 1788895000.0
        ))
        await expect { client.recentWorld != nil }

        XCTAssertEqual(client.state, .idle, "history was drawn as the live world")
        XCTAssertFalse(client.state.hasWorld)
        XCTAssertEqual(client.sessionBinding, WorldSessionBinding.none)
        XCTAssertEqual(client.inspection, .live)
        XCTAssertEqual(client.selection?.mode, .latest)
        XCTAssertEqual(
            client.recentWorld,
            WorldRecentReference(
                worldID: "w-prev", sessionID: "s-w-prev", name: nil,
                modelState: "finalized", updatedAt: 1788895032.0
            )
        )
        await settle()
        XCTAssertTrue(coordinates.isEmpty, "a stored world's geometry was addressed in Live")

        tower.disconnect()
    }

    /// A `live` selection is the current world: it goes through the gate,
    /// reads as receiving, offers no "last saved world", and its geometry is
    /// addressed.
    func testALiveSelectionIsTheLiveWorldAndItsGeometryIsAddressed() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        var coordinates: [WorldGeometryCoordinates] = []
        let cancellable = client.geometryUpdates.sink { coordinates.append($0) }
        defer { cancellable.cancel() }
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        server.send(text: message(
            seq: 1, modelState: "receiving", worldID: "w-live", keyframes: 28,
            revision: "r28", geometryRevision: "g28", selection: "live"
        ))
        await expect { client.state.isReceivingUpdates }
        XCTAssertEqual(client.state.snapshot?.worldID, "w-live")
        XCTAssertNil(client.recentWorld)
        XCTAssertEqual(client.selection?.mode, .live)
        await expect { !coordinates.isEmpty }
        XCTAssertEqual(
            coordinates.last,
            WorldGeometryCoordinates(worldID: "w-live", sessionID: "s-w-live", revision: "g28")
        )

        tower.disconnect()
    }

    /// A `finalizing` selection is the current world too — the two minutes
    /// after Stop — and with `build_in_progress: true` the state says a build
    /// is running, on the Tower's evidence.
    func testAFinalizingSelectionIsTheCurrentWorldAndSaysABuildIsRunning() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        server.send(text: message(
            seq: 1, modelState: "finalizing", worldID: "w-live", keyframes: 44,
            revision: "r45", geometryRevision: "g44", selection: "finalizing",
            endedAt: 1788895122.0, buildInProgress: true,
            reason: "the builder is finishing this world"
        ))
        await expect { client.state.hasWorld }
        guard case .finalizing(let snapshot, let building) = client.state else {
            return XCTFail("finalizing presented as \(client.state)")
        }
        XCTAssertEqual(snapshot.worldID, "w-live")
        XCTAssertEqual(building, true)
        XCTAssertNil(client.recentWorld)

        tower.disconnect()
    }

    /// No `selection` block — an older Tower — and the gate is the only
    /// judge, exactly as before: with no bracket open the stored world passes
    /// through, and nothing is offered as "recent" because nothing said it
    /// was history.
    func testAnOlderTowerWithoutASelectionBlockKeepsTheGateAsTheOnlyJudge() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        server.send(text: message(
            seq: 1, modelState: "finalized", worldID: "w-prev", keyframes: 143,
            revision: "r1", geometryRevision: "g1", selection: nil,
            endedAt: 1788895000.0
        ))
        await expect { client.state.hasWorld }
        guard case .finalized = client.state else { return XCTFail("presented as \(client.state)") }
        XCTAssertEqual(client.selection, .unknown)
        XCTAssertNil(client.recentWorld)

        tower.disconnect()
    }

    /// Pinned is history by construction. The selection is not consulted —
    /// even a nonsensical `latest` on a pinned subscription changes nothing.
    func testAPinnedReportIsHistoryWhateverItsSelectionSays() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        var coordinates: [WorldGeometryCoordinates] = []
        let cancellable = client.geometryUpdates.sink { coordinates.append($0) }
        defer { cancellable.cancel() }
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        client.inspect(worldID: "w-stored", sessionID: "s-w-stored")
        await expect { self.sent(recorder, type: "result_subscribe").count == 2 }

        for selection in ["pinned", "latest"] {
            server.send(text: message(
                seq: 1, subscription: "sub-2", modelState: "finalized", worldID: "w-stored",
                keyframes: 143, revision: "r-\(selection)", geometryRevision: "g-\(selection)",
                selection: selection, endedAt: 1788895000.0
            ))
            await expect { client.state.snapshot?.revision == "r-\(selection)" }
            guard case .finalized = client.state else { return XCTFail("presented as \(client.state)") }
            XCTAssertNil(client.recentWorld, "a pinned world was offered as recent")
            XCTAssertEqual(client.inspection, .inspecting(worldID: "w-stored"))
        }
        await expect { coordinates.count == 2 }

        tower.disconnect()
    }

    /// An interrupted world, pinned, is `.interrupted` — not failed, not
    /// finished — and its geometry is addressed so the gallery and the
    /// picture can show what exists.
    func testAnInterruptedWorldOnTheWireIsInterruptedNotFailedAndAddressesItsGeometry() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        var coordinates: [WorldGeometryCoordinates] = []
        let cancellable = client.geometryUpdates.sink { coordinates.append($0) }
        defer { cancellable.cancel() }
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        client.inspect(worldID: "w-dead", sessionID: "s-w-dead")
        await expect { self.sent(recorder, type: "result_subscribe").count == 2 }
        server.send(text: message(
            seq: 1, subscription: "sub-2", modelState: "interrupted", worldID: "w-dead",
            keyframes: 463, revision: "r463", geometryRevision: "g463", selection: "pinned",
            reason: "the process building this world exited without stopping its session"
        ))
        await expect { client.state.hasWorld }
        guard case .interrupted(let snapshot, let reason) = client.state else {
            return XCTFail("presented as \(client.state)")
        }
        XCTAssertEqual(snapshot.keyframeCount, 463)
        XCTAssertTrue(reason.hasPrefix("the process building this world exited"))
        XCTAssertEqual(client.state.phase, .settled)
        await expect { !coordinates.isEmpty }
        XCTAssertEqual(coordinates.last?.worldID, "w-dead")

        tower.disconnect()
    }

    /// Back to Live when nothing is live re-creates the `latest` answer, and
    /// it must be `.idle` with the world on offer — never the stored world
    /// re-hydrated under the Live heading.
    func testBackToLiveWhenIdleDoesNotRehydrateAStoredWorldAsLive() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        client.inspect(worldID: "w-stored", sessionID: "s-w-stored")
        await expect { self.sent(recorder, type: "result_subscribe").count == 2 }
        server.send(text: message(
            seq: 1, subscription: "sub-2", modelState: "finalized", worldID: "w-stored",
            keyframes: 143, revision: "r1", geometryRevision: "g1", selection: "pinned",
            endedAt: 1788895000.0
        ))
        await expect { client.state.hasWorld }

        client.followLive()
        XCTAssertEqual(client.state, .awaitingFirstUpdate)
        XCTAssertNil(client.recentWorld, "the pinned report's offer survived the switch")
        await expect { self.sent(recorder, type: "result_subscribe").count == 3 }
        server.send(text: message(
            seq: 1, subscription: "sub-3", modelState: "finalized", worldID: "w-stored",
            keyframes: 143, revision: "r1", geometryRevision: "g1", selection: "latest",
            endedAt: 1788895000.0
        ))
        await expect { client.recentWorld?.worldID == "w-stored" }
        XCTAssertEqual(client.state, .idle)
        XCTAssertEqual(client.inspection, .live)

        tower.disconnect()
    }

    /// With this phone's bracket open and nothing live yet — the seconds
    /// between Start and the builder attaching — a `latest` answer is
    /// "waiting, nothing is building from these frames yet", not "foreign".
    func testALatestSelectionWhileTheBracketIsOpenIsWaitingNotForeign() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        var coordinates: [WorldGeometryCoordinates] = []
        let cancellable = client.geometryUpdates.sink { coordinates.append($0) }
        defer { cancellable.cancel() }
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }
        tower.sendStreamStart()
        await expect { tower.isStreamingToTower }

        server.send(text: message(
            seq: 1, modelState: "finalized", worldID: "w-prev", keyframes: 143,
            revision: "r1", geometryRevision: "g1", selection: "latest",
            endedAt: 1788895000.0
        ))
        await expect { client.recentWorld != nil }
        XCTAssertEqual(client.state, .awaitingFirstUpdate)
        XCTAssertEqual(client.sessionBinding, .awaiting(captureID: nil))

        // Then the builder attaches and the same subscription drifts to the
        // new world: bound, receiving, and the offer withdrawn.
        server.send(text: message(
            seq: 2, modelState: "receiving", worldID: "w-new", keyframes: 4,
            revision: "r4", selection: "live"
        ))
        await expect { client.state.isReceivingUpdates }
        XCTAssertEqual(client.sessionBinding, .bound(captureID: "cap-1"))
        XCTAssertNil(client.recentWorld)
        await settle()
        XCTAssertTrue(coordinates.isEmpty, "w-prev's geometry was addressed while waiting")

        tower.sendStreamStop()
        tower.disconnect()
    }

    /// C12. A snapshot the gate refuses names nothing for the picture and
    /// fetches nothing: no coordinates leave the client, so the view model
    /// makes no manifest request and holds no target.
    func testAForeignSnapshotNeitherNamesThePictureNorFetchesItsGeometry() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }
        StubbedGeometryProtocol.reset(routes: [:])

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        let viewModel = WorldBuilderViewModel(
            client: client,
            geometry: WorldGeometryClient(baseURL: Self.host, session: StubbedGeometryProtocol.makeSession())
        )
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }
        tower.sendStreamStart()
        await expect { tower.isStreamingToTower }

        // An older Tower's answer (no selection): a finished world from an
        // earlier capture, with geometry to fetch.
        server.send(text: message(
            seq: 1, modelState: "finalized", worldID: "w-earlier", keyframes: 143,
            revision: "r1", geometryRevision: "g1", selection: nil,
            captureID: "cap-earlier", endedAt: 1788895000.0
        ))
        await expect { client.sessionBinding == .foreign(captureID: "cap-earlier") }
        XCTAssertEqual(client.state, .awaitingFirstUpdate)
        await settle()
        XCTAssertNil(viewModel.renderTarget, "a foreign snapshot named the picture target")
        XCTAssertEqual(
            StubbedGeometryProtocol.requestCount(for: "/worlds/w-earlier/geometry/manifest"), 0,
            "a foreign snapshot's geometry was fetched"
        )

        tower.sendStreamStop()
        tower.disconnect()
    }

    /// The 2026-09-06 drift, on an older Tower: the unpinned subscription
    /// carries a finished world with geometry, then a brand-new world with
    /// none, on the same id. The old world's fragments and picture target
    /// must not survive under the new world's heading.
    func testALiveSubscriptionThatDriftsToANewWorldDropsTheOldWorldsGeometry() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        serve(server)
        defer { server.stop() }
        StubbedGeometryProtocol.reset(routes: [
            "/worlds/w-prev/geometry/manifest": (200, Self.manifest(worldID: "w-prev")),
            "/worlds/w-prev/geometry/segment/0": (200, Self.segment),
        ])

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        let viewModel = WorldBuilderViewModel(
            client: client,
            geometry: WorldGeometryClient(baseURL: Self.host, session: StubbedGeometryProtocol.makeSession())
        )
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }

        server.send(text: message(
            seq: 1, modelState: "finalized", worldID: "w-prev", keyframes: 143,
            revision: "r1", geometryRevision: "g1", selection: nil,
            endedAt: 1788895000.0
        ))
        await expect { viewModel.fragmentsModel.segments.count == 1 }
        XCTAssertEqual(viewModel.renderTarget, WorldRenderTarget(worldID: "w-prev", sessionID: "s-w-prev"))

        server.send(text: message(
            seq: 2, modelState: "receiving", worldID: "w-new", keyframes: 2,
            revision: "r2", geometryRevision: nil, selection: nil
        ))
        await expect { viewModel.state.isReceivingUpdates }
        await expect { viewModel.fragmentsModel.segments.isEmpty }
        XCTAssertTrue(viewModel.geometryChunks.isEmpty)
        XCTAssertNil(viewModel.renderTarget, "the previous world's picture target survived the drift")

        tower.disconnect()
    }

    // MARK: C3 — a pin changed before its ack

    /// `inspect` then `followLive` inside one round trip sends two subscribes
    /// and the Tower opens both. The first ack belongs to an attempt already
    /// superseded: it is unsubscribed at once and its envelopes are dropped,
    /// and the state follows the second subscription only.
    func testAPinChangedBeforeItsAckIsUnsubscribedAndItsEnvelopesAreDropped() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }
        // Let the live subscription's own ack land first, so the race under
        // test is exactly the one named: two subscribes, both unanswered.
        await settle()

        // Both calls on one main-actor turn: no ack can land between them.
        client.inspect(worldID: "w-stored", sessionID: "s-w-stored")
        client.followLive()
        XCTAssertEqual(client.inspection, .live)

        await expect { self.sent(recorder, type: "result_subscribe").count == 3 }
        await expect(timeout: 3) {
            self.sent(recorder, type: "result_unsubscribe")
                .contains { $0["subscription_id"] as? String == "sub-2" }
        }
        let unsubscribed = sent(recorder, type: "result_unsubscribe").compactMap { $0["subscription_id"] as? String }
        XCTAssertEqual(unsubscribed.filter { $0 == "sub-2" }.count, 1, "\(unsubscribed)")
        XCTAssertFalse(unsubscribed.contains("sub-3"), "the current subscription was closed")

        // A heartbeat for the superseded pin is dropped; the live one lands.
        server.send(text: message(
            seq: 1, subscription: "sub-2", modelState: "finalized", worldID: "w-stored",
            keyframes: 143, revision: "r-stored", selection: "pinned", endedAt: 1788895000.0
        ))
        server.send(text: message(
            seq: 1, subscription: "sub-3", modelState: "receiving", worldID: "w-live",
            keyframes: 31, revision: "r31", selection: "live"
        ))
        await expect { client.state.snapshot?.keyframeCount == 31 }
        XCTAssertEqual(client.state.snapshot?.worldID, "w-live")

        // And it stays: another straggler for sub-2 changes nothing.
        server.send(text: message(
            seq: 2, subscription: "sub-2", modelState: "finalized", worldID: "w-stored",
            keyframes: 144, revision: "r-stored2", selection: "pinned", endedAt: 1788895000.0
        ))
        await settle()
        XCTAssertEqual(client.state.snapshot?.keyframeCount, 31)

        tower.disconnect()
    }

    /// Two rapid inspects: exactly one subscription stays open — the second —
    /// and the first is closed by name.
    func testTwoRapidInspectsLeaveExactlyOneLiveSubscription() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = MessageRecorder()
        serve(server, recorder: recorder)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        let client = TowerWorldBuilderClient(tower: tower)
        tower.connect(to: url(port: port))
        await expect { client.state == .awaitingFirstUpdate }
        await settle()

        client.inspect(worldID: "w-1", sessionID: nil)
        client.inspect(worldID: "w-2", sessionID: nil)
        XCTAssertEqual(client.inspection, .inspecting(worldID: "w-2"))

        await expect { self.sent(recorder, type: "result_subscribe").count == 3 }
        await expect {
            Set(self.sent(recorder, type: "result_unsubscribe").compactMap { $0["subscription_id"] as? String })
                == ["sub-1", "sub-2"]
        }
        server.send(text: message(
            seq: 1, subscription: "sub-2", modelState: "finalized", worldID: "w-1",
            keyframes: 10, revision: "r-1", selection: "pinned", endedAt: 1788895000.0
        ))
        server.send(text: message(
            seq: 1, subscription: "sub-3", modelState: "finalized", worldID: "w-2",
            keyframes: 20, revision: "r-2", selection: "pinned", endedAt: 1788895000.0
        ))
        await expect { client.state.snapshot?.worldID == "w-2" }
        await settle()
        XCTAssertEqual(client.state.snapshot?.worldID, "w-2")
        XCTAssertEqual(
            sent(recorder, type: "result_unsubscribe").count, 2,
            "the current subscription was closed, or one was closed twice"
        )

        tower.disconnect()
    }

    // MARK: Fixtures

    private static func manifest(worldID: String) -> String {
        """
        {"contract": "world_builder.geometry/2026-08-25",
         "world_id": "\(worldID)", "session_id": "s-\(worldID)", "geometry_revision": "g1",
         "pose_convention": {
           "pose_type": "T_world_camera", "quaternion_order": "wxyz",
           "handedness": "right",
           "camera_axes": "opencv_x_right_y_down_z_forward",
           "translation_units": "world",
           "world_axes_origin": "first_keyframe_camera",
           "up_axis": "unknown", "pose_dtype": "float64",
           "point_dtype": "float32"},
         "segment_count": 1,
         "segments": [
           {"segment_index": 0, "content_hash": "h0", "frame_id": "segment:0",
            "registered": false, "transform_to_world": null,
            "resolution_state": "resolved", "dominant_degeneracy": null,
            "keyframe_count": 2, "solved_count": 1, "point_count": 1,
            "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}}]}
        """
    }

    fileprivate static let segment = """
        {"contract": "world_builder.geometry/2026-08-25",
         "segment_index": 0, "content_hash": "h0", "frame_id": "segment:0",
         "registered": false, "transform_to_world": null,
         "poses": [{"keyframe_id": "s1:0", "status": "anchor", "degeneracy": "",
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0]}],
         "points": [[0.5, 0.5, 0.5]],
         "points_sent": 1, "points_total": 1, "point_sampling": "none"}
        """
}

// MARK: - The view model's gallery is keyed on world identity

/// A `WorldBuilderClient` a test drives by hand: four subjects, no socket.
/// The seam the audit asked for, so view-model behaviour under client-driven
/// transitions is testable without a Tower.
@MainActor
final class ScriptedWorldBuilderClient: WorldBuilderClient {
    let cartridgeID = "world-build"
    private(set) var state: WorldModelState = .idle
    private(set) var sessionBinding: WorldSessionBinding = .none
    private(set) var inspection: WorldInspectionMode = .live
    private(set) var recentWorld: WorldRecentReference?

    private let stateSubject = PassthroughSubject<WorldModelState, Never>()
    private let bindingSubject = PassthroughSubject<WorldSessionBinding, Never>()
    private let inspectionSubject = PassthroughSubject<WorldInspectionMode, Never>()
    private let recentSubject = PassthroughSubject<WorldRecentReference?, Never>()
    private let geometrySubject = PassthroughSubject<WorldGeometryCoordinates, Never>()

    var stateUpdates: AnyPublisher<WorldModelState, Never> { stateSubject.eraseToAnyPublisher() }
    var bindingUpdates: AnyPublisher<WorldSessionBinding, Never> { bindingSubject.eraseToAnyPublisher() }
    var inspectionUpdates: AnyPublisher<WorldInspectionMode, Never> { inspectionSubject.eraseToAnyPublisher() }
    var recentWorldUpdates: AnyPublisher<WorldRecentReference?, Never> { recentSubject.eraseToAnyPublisher() }
    var geometryUpdates: AnyPublisher<WorldGeometryCoordinates, Never> { geometrySubject.eraseToAnyPublisher() }

    /// Recorded so a test can assert the view model asked for the pin it was
    /// told to.
    private(set) var pins: [(worldID: String, sessionID: String?)] = []

    init(recentWorld: WorldRecentReference? = nil) {
        self.recentWorld = recentWorld
    }

    func send(_ state: WorldModelState) {
        self.state = state
        stateSubject.send(state)
    }

    func send(recent: WorldRecentReference?) {
        recentWorld = recent
        recentSubject.send(recent)
    }

    func send(_ coordinates: WorldGeometryCoordinates) {
        geometrySubject.send(coordinates)
    }

    func inspect(worldID: String, sessionID: String?) {
        pins.append((worldID, sessionID))
        inspection = .inspecting(worldID: worldID)
        inspectionSubject.send(inspection)
    }

    func followLive() {
        inspection = .live
        inspectionSubject.send(inspection)
    }
}

@MainActor
final class WorldBuilderViewModelOwnershipTests: XCTestCase {

    private static let host = URL(string: "http://stub.invalid")!

    private func manifest(worldID: String, sessionID: String) -> String {
        """
        {"contract": "world_builder.geometry/2026-08-25",
         "world_id": "\(worldID)", "session_id": "\(sessionID)", "geometry_revision": "g1",
         "pose_convention": {
           "pose_type": "T_world_camera", "quaternion_order": "wxyz",
           "handedness": "right",
           "camera_axes": "opencv_x_right_y_down_z_forward",
           "translation_units": "world",
           "world_axes_origin": "first_keyframe_camera",
           "up_axis": "unknown", "pose_dtype": "float64",
           "point_dtype": "float32"},
         "segment_count": 1,
         "segments": [
           {"segment_index": 0, "content_hash": "h0", "frame_id": "segment:0",
            "registered": false, "transform_to_world": null,
            "resolution_state": "resolved", "dominant_degeneracy": null,
            "keyframe_count": 2, "solved_count": 1, "point_count": 1,
            "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}}]}
        """
    }

    /// A view model whose stub answers world `w-a` / session `s-a` and
    /// nothing else.
    private func makeViewModel(client: any WorldBuilderClient) -> WorldBuilderViewModel {
        StubbedGeometryProtocol.reset(routes: [
            "/worlds/w-a/geometry/manifest": (200, manifest(worldID: "w-a", sessionID: "s-a")),
            "/worlds/w-a/geometry/segment/0": (200, TowerWorldBuilderLiveHistoryTests.segment),
        ])
        return WorldBuilderViewModel(
            client: client,
            geometry: WorldGeometryClient(baseURL: Self.host, session: StubbedGeometryProtocol.makeSession())
        )
    }

    private func waitUntil(
        timeout: TimeInterval = 3, _ condition: @MainActor () -> Bool
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return condition()
    }

    /// Fill the gallery with `w-a`'s one segment.
    private func populate(_ viewModel: WorldBuilderViewModel) async {
        await viewModel.geometryDidChange(worldID: "w-a", sessionID: "s-a", revision: "g1")
        XCTAssertEqual(viewModel.fragmentsModel.segments.count, 1, "the fixture did not populate the gallery")
        XCTAssertEqual(viewModel.renderTarget, WorldRenderTarget(worldID: "w-a", sessionID: "s-a"))
    }

    /// Coordinates for a different world clear what is drawn before anything
    /// of the new world is fetched — even when the new world's manifest then
    /// fails, which is when the old fragments used to survive.
    func testGeometryFromADifferentWorldReplacesTheGalleryRatherThanJoiningIt() async {
        let viewModel = makeViewModel(client: UnavailableWorldBuilderClient())
        await populate(viewModel)

        await viewModel.geometryDidChange(worldID: "w-b", sessionID: "s-b", revision: "g1")
        XCTAssertTrue(viewModel.fragmentsModel.segments.isEmpty, "w-a's gallery survived under w-b")
        XCTAssertTrue(viewModel.geometryChunks.isEmpty)
        XCTAssertEqual(viewModel.renderTarget, WorldRenderTarget(worldID: "w-b", sessionID: "s-b"))
    }

    /// Another session of the same world is another gallery.
    func testGeometryFromAnotherSessionOfTheSameWorldAlsoClearsTheGallery() async {
        let viewModel = makeViewModel(client: UnavailableWorldBuilderClient())
        await populate(viewModel)
        await viewModel.geometryDidChange(worldID: "w-a", sessionID: "s-a2", revision: "g1")
        XCTAssertTrue(viewModel.fragmentsModel.segments.isEmpty)
        XCTAssertEqual(viewModel.renderTarget, WorldRenderTarget(worldID: "w-a", sessionID: "s-a2"))
    }

    /// The same world under a heartbeat keeps everything.
    func testTheSameWorldUnderTheSameRevisionKeepsTheGallery() async {
        let viewModel = makeViewModel(client: UnavailableWorldBuilderClient())
        await populate(viewModel)
        await viewModel.geometryDidChange(worldID: "w-a", sessionID: "s-a", revision: "g1")
        XCTAssertEqual(viewModel.fragmentsModel.segments.count, 1)
        XCTAssertEqual(StubbedGeometryProtocol.requestCount(for: "/worlds/w-a/geometry/manifest"), 1)
    }

    /// An inspection change clears the gallery. The picture target set by
    /// `open` for the world now being inspected survives the change the
    /// client publishes a turn later; any other target does not.
    func testAnInspectionChangeClearsTheGalleryButKeepsAMatchingPictureTarget() async {
        let client = ScriptedWorldBuilderClient()
        let viewModel = makeViewModel(client: client)
        await populate(viewModel)

        viewModel.open(worldID: "w-b", sessionID: nil)
        XCTAssertEqual(client.pins.last?.worldID, "w-b")
        let published = await waitUntil { viewModel.inspection == .inspecting(worldID: "w-b") }
        XCTAssertTrue(published, "the inspection change never reached the view model")
        XCTAssertTrue(viewModel.fragmentsModel.segments.isEmpty)
        XCTAssertEqual(
            viewModel.renderTarget, WorldRenderTarget(worldID: "w-b", sessionID: nil),
            "the pinned world's picture target was lost to its own inspection change"
        )

        // Driven directly: an inspection naming some other world drops it.
        viewModel.inspectionDidChange(to: .inspecting(worldID: "w-c"))
        XCTAssertNil(viewModel.renderTarget)

        // And Live forgets it until coordinates re-earn one.
        await populate(viewModel)
        viewModel.inspectionDidChange(to: .live)
        XCTAssertTrue(viewModel.fragmentsModel.segments.isEmpty)
        XCTAssertNil(viewModel.renderTarget)
    }

    func testReturningToLiveClearsTheGalleryAndThePictureTarget() async {
        let client = ScriptedWorldBuilderClient()
        let viewModel = makeViewModel(client: client)
        await populate(viewModel)
        viewModel.returnToLive()
        XCTAssertTrue(viewModel.fragmentsModel.segments.isEmpty)
        XCTAssertTrue(viewModel.geometryChunks.isEmpty)
        XCTAssertNil(viewModel.renderTarget)
        XCTAssertEqual(client.inspection, .live)
    }

    /// `.idle`, `.failed`, `.unsupported` have no world; the gallery and the
    /// picture target go with the world. `.awaitingFirstUpdate` keeps both —
    /// it is what a reconnect passes through on the way back to the same
    /// world.
    func testAStateWithoutASnapshotForgetsTheGalleryAndThePictureTarget() async {
        let client = ScriptedWorldBuilderClient()
        let viewModel = makeViewModel(client: client)

        await populate(viewModel)
        viewModel.stateDidChange(to: .awaitingFirstUpdate)
        XCTAssertEqual(viewModel.fragmentsModel.segments.count, 1, "waiting for the same world dropped its gallery")
        XCTAssertNotNil(viewModel.renderTarget)

        for state in [
            WorldModelState.idle,
            .failed(CartridgeFailure(kind: .transport, message: "x")),
            .unsupported(reason: "x"),
        ] {
            await populate(viewModel)
            viewModel.stateDidChange(to: state)
            XCTAssertTrue(viewModel.fragmentsModel.segments.isEmpty, "\(state) kept the gallery")
            XCTAssertTrue(viewModel.geometryChunks.isEmpty)
            XCTAssertNil(viewModel.renderTarget, "\(state) kept the picture target")
            XCTAssertEqual(viewModel.state, state)
        }
    }

    /// A world state naming a different world than the gallery's owner
    /// forgets the gallery before any of the new world's geometry arrives —
    /// the drift case, when the new world has nothing to fetch yet.
    func testAWorldStateNamingAnotherWorldForgetsThePreviousGallery() async {
        let viewModel = makeViewModel(client: ScriptedWorldBuilderClient())
        await populate(viewModel)

        viewModel.stateDidChange(to: .receiving(WorldSnapshot(worldID: "w-a", keyframeCount: 5)))
        XCTAssertEqual(viewModel.fragmentsModel.segments.count, 1, "the same world's state dropped its gallery")

        viewModel.stateDidChange(to: .receiving(WorldSnapshot(worldID: "w-new", keyframeCount: 2)))
        XCTAssertTrue(viewModel.fragmentsModel.segments.isEmpty)
        XCTAssertNil(viewModel.renderTarget)

        // A pinned world's target, set before any state of its own, survives
        // that state: nothing was drawn, so nothing is forgotten.
        viewModel.open(worldID: "w-pinned", sessionID: nil)
        viewModel.stateDidChange(to: .finalized(WorldSnapshot(worldID: "w-pinned", keyframeCount: 9)))
        XCTAssertEqual(viewModel.renderTarget, WorldRenderTarget(worldID: "w-pinned", sessionID: nil))
    }

    /// The offer is seeded from the client and follows its updates; the
    /// Open action pins exactly what was offered.
    func testTheRecentWorldIsSeededAndRepublishedFromTheClient() async {
        let offered = WorldRecentReference(worldID: "w-prev", sessionID: "s-prev", modelState: "finalized")
        let client = ScriptedWorldBuilderClient(recentWorld: offered)
        let viewModel = WorldBuilderViewModel(client: client)
        XCTAssertEqual(viewModel.recentWorld, offered)

        client.send(recent: nil)
        let cleared = await waitUntil { viewModel.recentWorld == nil }
        XCTAssertTrue(cleared)

        client.send(recent: offered)
        let restored = await waitUntil { viewModel.recentWorld == offered }
        XCTAssertTrue(restored)

        viewModel.open(worldID: offered.worldID, sessionID: offered.sessionID)
        XCTAssertEqual(client.pins.last?.worldID, "w-prev")
        XCTAssertEqual(client.pins.last?.sessionID, "s-prev")
    }
}

// MARK: - The World Builder cartridge session

/// Answers the session surface without a network. Keyed by method and path,
/// records every request, and holds only Foundation types so it can be
/// driven from the URL loading system's own threads.
final class WorldBuilderSessionStubProtocol: URLProtocol {
    private static let lock = NSLock()
    /// `"POST /cartridges/world_builder/session/start"` → (status, body).
    /// A key with no entry fails as a transport error.
    private static var routes: [String: (Int, [String: Any])] = [:]
    private static var requests: [(method: String, path: String, timeout: TimeInterval)] = []

    static func reset(routes: [String: (Int, [String: Any])]) {
        lock.lock()
        defer { lock.unlock() }
        self.routes = routes
        requests = []
    }

    static func recorded() -> [(method: String, path: String, timeout: TimeInterval)] {
        lock.lock()
        defer { lock.unlock() }
        return requests
    }

    static func makeSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [WorldBuilderSessionStubProtocol.self]
        configuration.urlCache = nil
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let method = request.httpMethod ?? "GET"
        let path = request.url?.path ?? ""
        WorldBuilderSessionStubProtocol.lock.lock()
        WorldBuilderSessionStubProtocol.requests.append((method, path, request.timeoutInterval))
        let route = WorldBuilderSessionStubProtocol.routes["\(method) \(path)"]
        WorldBuilderSessionStubProtocol.lock.unlock()

        guard let route, let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
            return
        }
        let response = HTTPURLResponse(url: url, statusCode: route.0, httpVersion: "HTTP/1.1", headerFields: nil)!
        let data = (try? JSONSerialization.data(withJSONObject: route.1)) ?? Data()
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@MainActor
final class WorldBuilderSessionControllerTests: XCTestCase {

    private static let host = URL(string: "http://stub.invalid")!
    private static let startKey = "POST /cartridges/world_builder/session/start"
    private static let stopKey = "POST /cartridges/world_builder/session/stop"

    /// A `cartridge_session.control/2026-08-27` snapshot for `world_builder`,
    /// as the generic router answers it.
    private func session(state: String, accepted: Bool = true, changed: Bool = true) -> [String: Any] {
        [
            "contract": "cartridge_session.control/2026-08-27",
            "cartridge": "world_builder",
            "worker": "world-build-session",
            "supported": true,
            "state": state,
            "state_means": "intent-not-liveness",
            "states": ["stopped", "active", "paused"],
            "actions": ["start", "pause", "resume", "stop"],
            "session_id": (state == "stopped" ? NSNull() : "sess-1") as Any,
            "started_at": 1788895000.0,
            "changed_at": 1788895000.0,
            "following": [String](),
            "following_this_session": [String](),
            "captures": [String](),
            "accepted": accepted,
            "changed": changed,
            "attached_capture_id": NSNull(),
            "stop_policy": "request",
        ]
    }

    private func makeController() -> WorldBuilderSessionController {
        WorldBuilderSessionController(
            control: CartridgeSessionHTTPClient(
                baseURL: Self.host,
                session: WorldBuilderSessionStubProtocol.makeSession(),
                cartridge: "world_builder"
            )
        )
    }

    private func waitUntil(
        timeout: TimeInterval = 3, _ condition: @MainActor () -> Bool
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return condition()
    }

    private func requests(_ method: String = "POST") -> [String] {
        WorldBuilderSessionStubProtocol.recorded().filter { $0.method == method }.map(\.path)
    }

    override func setUp() {
        super.setUp()
        WorldBuilderSessionStubProtocol.reset(routes: [
            Self.startKey: (200, session(state: "active")),
            Self.stopKey: (200, session(state: "stopped")),
        ])
    }

    func testAppearingWithAReachableTowerSendsStartAndReportsActive() async {
        let controller = makeController()
        XCTAssertEqual(controller.status, .notAsked)
        XCTAssertEqual(controller.footnote, "World Builder has not been asked for on the Tower yet.")

        controller.workspaceDidAppear(isTowerReachable: true)
        XCTAssertEqual(controller.status, .starting)
        let active = await waitUntil { controller.status == .active }
        XCTAssertTrue(active, "status is \(controller.status)")
        XCTAssertEqual(requests(), ["/cartridges/world_builder/session/start"])
        XCTAssertEqual(controller.footnote, "World Builder is active on the Tower.")
    }

    func testDisappearingSendsStop() async {
        let controller = makeController()
        controller.workspaceDidAppear(isTowerReachable: true)
        _ = await waitUntil { controller.status == .active }

        controller.workspaceDidDisappear()
        let stopped = await waitUntil { self.requests().contains("/cartridges/world_builder/session/stop") }
        XCTAssertTrue(stopped)
        let reset = await waitUntil { controller.status == .notAsked }
        XCTAssertTrue(reset, "status is \(controller.status)")
        XCTAssertEqual(requests(), [
            "/cartridges/world_builder/session/start",
            "/cartridges/world_builder/session/stop",
        ])
    }

    /// Nothing is sent to a Tower that is not there; the socket coming back
    /// is the moment to ask, and a Tower that went away and came back —
    /// probably restarted, and restarted means `stopped` — is asked again.
    func testTheTowerComingBackOnlineRestartsWhileOnScreen() async {
        let controller = makeController()
        controller.workspaceDidAppear(isTowerReachable: false)
        XCTAssertEqual(controller.status, .waitingForTower)
        try? await Task.sleep(nanoseconds: 150_000_000)
        XCTAssertTrue(requests().isEmpty, "a start was sent to an unreachable Tower")

        controller.towerReachabilityChanged(isReachable: true)
        let active = await waitUntil { controller.status == .active }
        XCTAssertTrue(active)
        XCTAssertEqual(requests().count, 1)

        controller.towerReachabilityChanged(isReachable: false)
        XCTAssertEqual(controller.status, .active, "an honoured start was forgotten on a drop")
        controller.towerReachabilityChanged(isReachable: true)
        let restarted = await waitUntil { self.requests().count == 2 }
        XCTAssertTrue(restarted)
        XCTAssertEqual(requests(), [
            "/cartridges/world_builder/session/start",
            "/cartridges/world_builder/session/start",
        ])
    }

    /// Off screen, a reconnect asks for nothing.
    func testAReconnectOffScreenSendsNothing() async {
        let controller = makeController()
        controller.towerReachabilityChanged(isReachable: true)
        try? await Task.sleep(nanoseconds: 150_000_000)
        XCTAssertTrue(requests().isEmpty)
        XCTAssertEqual(controller.status, .notAsked)
    }

    /// A 409 is the Tower's own sentence, in the footnote.
    func testARefusalIsSurfacedInTheFootnote() async {
        var refusal = session(state: "stopped", accepted: false, changed: false)
        refusal["reason"] = "unsupported"
        refusal["message"] = "no World Builder producer is configured on this Tower"
        WorldBuilderSessionStubProtocol.reset(routes: [
            Self.startKey: (409, ["detail": refusal]),
            Self.stopKey: (200, session(state: "stopped")),
        ])
        let controller = makeController()
        controller.workspaceDidAppear(isTowerReachable: true)
        let refused = await waitUntil {
            if case .refused = controller.status { return true }
            return false
        }
        XCTAssertTrue(refused, "status is \(controller.status)")
        XCTAssertEqual(
            controller.footnote,
            "The Tower refused to activate World Builder: no World Builder producer is configured on this Tower"
        )
    }

    /// A 404 is a configuration answer — no session control for this
    /// cartridge — and is worded as that, not as a network failure.
    func testA404IsNoSessionControlNotATransportFailure() async {
        WorldBuilderSessionStubProtocol.reset(routes: [
            Self.startKey: (404, ["detail": "no such cartridge session"]),
        ])
        let controller = makeController()
        controller.workspaceDidAppear(isTowerReachable: true)
        let answered = await waitUntil { controller.status == .noSessionControl }
        XCTAssertTrue(answered, "status is \(controller.status)")
        XCTAssertTrue(controller.footnote.contains("no World Builder session control"))
    }

    /// No route at all: the request fails, and the footnote says it could
    /// not be asked for, with the detail.
    func testATransportFailureIsSurfacedAndLeavesTheTowerUnclaimed() async {
        WorldBuilderSessionStubProtocol.reset(routes: [:])
        let controller = makeController()
        controller.workspaceDidAppear(isTowerReachable: true)
        let failed = await waitUntil {
            if case .failed = controller.status { return true }
            return false
        }
        XCTAssertTrue(failed, "status is \(controller.status)")
        XCTAssertTrue(controller.footnote.hasPrefix("World Builder could not be asked for on the Tower:"))
    }

    /// Rule 15: every request carries the ten-second bound, on the request
    /// that actually ran.
    func testTheRequestIsBounded() async {
        let controller = makeController()
        controller.workspaceDidAppear(isTowerReachable: true)
        _ = await waitUntil { controller.status == .active }
        XCTAssertEqual(WorldBuilderSessionStubProtocol.recorded().first?.timeout, 10)
    }

    /// A start that answers after the workspace has gone must not report
    /// `.active` for a screen that is not there; the stop still goes out.
    func testAStartThatLandsAfterDisappearingDoesNotReportActive() async {
        let controller = makeController()
        controller.workspaceDidAppear(isTowerReachable: true)
        controller.workspaceDidDisappear()
        let stopped = await waitUntil { self.requests().contains("/cartridges/world_builder/session/stop") }
        XCTAssertTrue(stopped)
        try? await Task.sleep(nanoseconds: 150_000_000)
        XCTAssertEqual(controller.status, .notAsked, "status is \(controller.status)")
    }

    /// A view model built beside this controller still sends nothing on its
    /// own: session control is the view's, not the view model's.
    func testTheViewModelItselfAsksForNothing() async {
        _ = WorldBuilderViewModel(client: UnavailableWorldBuilderClient())
        try? await Task.sleep(nanoseconds: 150_000_000)
        XCTAssertTrue(requests().isEmpty)
    }
}
