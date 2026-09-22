//
//  WorldPresentationTests.swift
//  GlassesTests
//

import Combine
import XCTest
@testable import Glasses

// MARK: - The failure these tests exist for

/// The 2026-09-06 walk, as a fixture.
///
/// The phone showed this world's figures — 463 keyframes, 467 camera poses,
/// 17,674 points — and, underneath them, "Nothing mapped yet" and "The glasses
/// have not mapped anything here yet."
///
/// Both sentences came from one condition: `WorldFragmentsModel.fragments`
/// being empty. That array is assigned in exactly one place — the end of a
/// successful manifest-plus-chunks fetch — so it is empty in five unrelated
/// situations, four of which are the phone's own ignorance:
///
/// 1. No geometry address was ever published (`geometry.revision` was `null`).
/// 2. The session gate suppressed the address for a world that had one.
/// 3. The manifest fetch failed.
/// 4. One manifest row failed to decode, dropping the whole manifest.
/// 5. The manifest arrived under a pose convention this build refuses.
///
/// Every test in `WorldGeometryAccountTests` walks one of those and asserts
/// that the account produced is **not** the one that says nothing was mapped.
@MainActor
private enum FieldWalk {

    /// The world as the Tower's own status payload described it.
    static var snapshot: WorldSnapshot {
        WorldSnapshot(
            name: nil,
            worldID: "fcbca9e90b244785bdb671530b33c6a5",
            keyframeCount: 463,
            revision: "r-463",
            tracking: .good,
            scale: .relative,
            calibration: .uncalibrated,
            geometry: WorldGeometryReport(
                representation: "sparse point cloud", elementCount: 17_674, isIncremental: false
            ),
            trajectory: WorldTrajectoryReport(poseCount: 467, segments: 34, scale: .unknown)
        )
    }

    static var evidence: WorldEvidence { WorldEvidence(snapshot: snapshot)! }

    /// A world the Tower genuinely reported nothing for: a session that was
    /// opened, produced no keyframes worth solving, and was closed.
    static var barrenSnapshot: WorldSnapshot {
        WorldSnapshot(worldID: "w-barren", keyframeCount: 3, trajectory: WorldTrajectoryReport(poseCount: 0))
    }

    static var barrenEvidence: WorldEvidence { WorldEvidence(snapshot: barrenSnapshot)! }
}

// MARK: - The sentence

/// The whole point of the pass: **a world the Tower says holds geometry can
/// never be described as one the glasses did not map**, whatever the phone's
/// own fetch did.
@MainActor
final class WorldGeometryAccountTests: XCTestCase {

    /// The exact words that were wrong in the field, so a future edit that
    /// reintroduces them anywhere reachable from geometry-bearing evidence
    /// fails here rather than on a walk.
    private let forbidden = "Nothing mapped yet"

    private func account(
        _ status: WorldGeometryStatus, _ evidence: WorldEvidence?
    ) -> WorldGeometryAccount {
        WorldGeometryAccount.account(for: status, evidence: evidence)
    }

    // MARK: The five situations, one at a time

    /// Situations 1 and 2: no address was ever published, or the gate
    /// suppressed one. Indistinguishable from the phone, so one account covers
    /// both — and it says the Tower reports geometry, because it does.
    func testAWorldWithGeometryAndNoAddressIsNotCalledUnmapped() throws {
        let result = account(.notAddressed, FieldWalk.evidence)
        XCTAssertNotEqual(result.headline, forbidden)
        XCTAssertTrue(result.isPhoneSideUnknown, "this is the phone lacking an address")
        // The Tower's whole summary, not a substring of it, and not a
        // three-way `||` that passes on any one of them — that form would keep
        // passing after two of the three figures stopped reaching the sentence.
        let summary = try XCTUnwrap(FieldWalk.evidence.summary)
        XCTAssertTrue(result.detail?.contains(summary) == true,
                      "the Tower's own figures must appear: \(result.detail ?? "nil")")
    }

    /// Situation 3: the manifest request did not complete.
    func testAFailedFetchIsNamedAsAFetchFailureAndNotAsAnEmptyWorld() {
        let result = account(
            .failed(WorldGeometryFailure(kind: .unreachable, detail: "The request timed out.")),
            FieldWalk.evidence
        )
        XCTAssertNotEqual(result.headline, forbidden)
        XCTAssertTrue(result.detail?.contains("timed out") == true, "the transport's own words")
        XCTAssertTrue(result.detail?.contains("not a world that is empty") == true)
    }

    /// Situation 4: one unreadable row drops the whole manifest, by design.
    /// That is a decode refusal, not a statement about the room.
    func testAnUndecodableManifestSaysSoRatherThanReportingAnEmptyWorld() {
        let result = account(
            .failed(WorldGeometryFailure(kind: .undecodable, detail: nil)), FieldWalk.evidence
        )
        XCTAssertNotEqual(result.headline, forbidden)
        XCTAssertTrue(result.detail?.contains("One unreadable row") == true)
    }

    /// Situation 5: a pose convention this build does not implement. Refused
    /// rather than drawn — and now the refusal is visible.
    func testARefusedPoseConventionIsVisibleRatherThanSilent() {
        let result = account(
            .failed(WorldGeometryFailure(kind: .poseConvention, detail: nil)), FieldWalk.evidence
        )
        XCTAssertNotEqual(result.headline, forbidden)
        XCTAssertTrue(result.detail?.contains("pose convention") == true)
    }

    // MARK: The one situation that may say it

    /// The Tower answering 404 on the manifest route, for a world whose own
    /// snapshot claims nothing either. This is the only path to those words.
    func testOnlyTheTowerSayingSoProducesTheNothingMappedAccount() {
        let result = account(.towerReportsNone(detail: nil), FieldWalk.barrenEvidence)
        XCTAssertEqual(result.headline, forbidden)
        XCTAssertFalse(result.isPhoneSideUnknown)
    }

    /// A loaded manifest with nothing drawable in it, for a world that claims
    /// nothing. Also the Tower's answer, by a different route.
    func testAnEmptyManifestForAnEmptyWorldSaysNothingWasMapped() {
        let result = account(
            .loaded(WorldFragmentsModel(segments: []), unfetched: 0), FieldWalk.barrenEvidence
        )
        XCTAssertEqual(result, .nothingMapped)
    }

    /// **The load-bearing negative.** The Tower's 404 against the Tower's own
    /// snapshot claiming 17,674 points is the two halves of one machine
    /// disagreeing, and the phone reports the disagreement rather than picking
    /// the half that is easier to draw.
    func testATowerThatContradictsItselfIsReportedAsSuchAndNotAsAnEmptyRoom() {
        let result = account(
            .towerReportsNone(detail: "no geometry for this session"), FieldWalk.evidence
        )
        XCTAssertNotEqual(result.headline, forbidden)
        XCTAssertTrue(result.headline.contains("disagrees"))
        XCTAssertTrue(result.detail?.contains("no geometry for this session") == true,
                      "the Tower's own words survive")
    }

    /// The same rule for a manifest that arrived and named nothing drawable
    /// while the snapshot claimed geometry.
    func testAnEmptyManifestForAWorldWithGeometryIsNotCalledUnmapped() {
        let result = account(
            .loaded(WorldFragmentsModel(segments: []), unfetched: 0), FieldWalk.evidence
        )
        XCTAssertNotEqual(result.headline, forbidden)
    }

    // MARK: The states around the ladder

    func testNoWorldIsNotTheSameAsAnEmptyWorld() {
        let result = account(.noWorld, nil)
        XCTAssertNotEqual(result.headline, forbidden)
        XCTAssertEqual(result.headline, "No world")
    }

    func testAFetchInFlightSaysSo() {
        XCTAssertTrue(account(.loading, FieldWalk.evidence).isPhoneSideUnknown)
        XCTAssertNotEqual(account(.loading, FieldWalk.evidence).headline, forbidden)
    }

    /// A world with fragments drawn keeps the gallery's own count, and says
    /// how many segments did not arrive.
    func testAPartiallyFetchedWorldSaysWhichPartIsMissing() {
        let drawn = WorldFragmentsModel(segments: [Self.drawableSegment(index: 0, points: 500)])
        let result = account(.loaded(drawn, unfetched: 5), FieldWalk.evidence)
        // The literal the gallery draws, not `drawn.headline` — asserting that
        // the account equals the expression the account is implemented as
        // cannot fail, and this line was exactly that until a review said so.
        XCTAssertEqual(result.headline, "1 fragment, not yet connected")
        XCTAssertTrue(result.detail?.contains("5 segments") == true)
    }

    /// Whatever else changes, the five phone-side situations must never be
    /// reported as the Tower's answer. Asserted as a set so a sixth case added
    /// later has to be classified deliberately.
    func testEveryPhoneSideIgnoranceIsMarkedAsSuch() {
        let phoneSide: [WorldGeometryStatus] = [
            .notAddressed,
            .loading,
            .failed(WorldGeometryFailure(kind: .unreachable, detail: nil)),
            .failed(WorldGeometryFailure(kind: .undecodable, detail: nil)),
            .failed(WorldGeometryFailure(kind: .poseConvention, detail: nil)),
        ]
        for status in phoneSide {
            XCTAssertTrue(
                account(status, FieldWalk.evidence).isPhoneSideUnknown,
                "\(status) is the phone not knowing, and must be marked so"
            )
            XCTAssertNotEqual(account(status, FieldWalk.evidence).headline, forbidden)
        }
        // And the Tower's own answers are not marked as the phone's ignorance.
        XCTAssertFalse(account(.towerReportsNone(detail: nil), FieldWalk.barrenEvidence).isPhoneSideUnknown)
        XCTAssertFalse(
            account(.loaded(WorldFragmentsModel(segments: []), unfetched: 0), FieldWalk.barrenEvidence)
                .isPhoneSideUnknown
        )
    }

    /// **The gap a future edit could walk through.** Keeping the headline
    /// honest is not enough on its own: an edit that changed only the headline
    /// and left `nothingMapped`'s detail underneath it would pass every
    /// assertion above while still telling a wearer their world is empty.
    func testNoAccountOverGeometryEverReusesTheNothingMappedWording() {
        let everyStatusButNoWorld: [WorldGeometryStatus] = [
            .notAddressed,
            .loading,
            .failed(WorldGeometryFailure(kind: .unreachable, detail: "timed out")),
            .failed(WorldGeometryFailure(kind: .undecodable, detail: nil)),
            .failed(WorldGeometryFailure(kind: .poseConvention, detail: nil)),
            .towerReportsNone(detail: "no geometry for this session"),
            .loaded(WorldFragmentsModel(segments: []), unfetched: 0),
        ]
        for status in everyStatusButNoWorld {
            let result = account(status, FieldWalk.evidence)
            XCTAssertNotEqual(result.headline, WorldGeometryAccount.nothingMapped.headline, "\(status)")
            XCTAssertNotEqual(result.detail, WorldGeometryAccount.nothingMapped.detail, "\(status)")
            XCTAssertNotEqual(result, .nothingMapped, "\(status)")
        }
    }

    /// The app's strongest negative claim must not be what a caller that
    /// supplied nothing ends up saying.
    func testTheDefaultAccountAssertsNothing() {
        XCTAssertNotEqual(WorldGeometryAccount.undescribed, .nothingMapped)
        XCTAssertNil(WorldGeometryAccount.undescribed.detail)
        XCTAssertTrue(WorldGeometryAccount.undescribed.isPhoneSideUnknown)
        XCTAssertEqual(WorldPresentation.empty.account, .undescribed,
                       "a presentation nobody filled in says nothing about the world")
    }

    /// A drawable row: resolved, with bounds, with points.
    static func drawableSegment(index: Int, points: Int) -> WorldSegmentSummary {
        WorldSegmentSummary(
            segmentIndex: index, contentHash: "h\(index)",
            frameID: "segment:\(index)",
            registered: false,
            placement: WorldPlacementFields(
                registered: false, state: .unplaced, refusalReason: nil,
                transform: nil, placementHash: nil
            ),
            resolutionState: .resolved, dominantDegeneracy: nil,
            keyframeCount: 10, solvedCount: 8, pointCount: points,
            bounds: WorldBounds(json: ["min": [-1.0, 0.0, -1.0], "max": [1.0, 2.0, 1.0]]),
            coverage: nil
        )
    }
}

// MARK: - What the Tower said exists

@MainActor
final class WorldEvidenceTests: XCTestCase {

    func testThereIsNoEvidenceWithoutASnapshot() {
        XCTAssertNil(WorldEvidence(snapshot: nil))
    }

    /// Either figure on its own is geometry. Requiring both would let the field
    /// case back through: a build that recovers points across segments whose
    /// cameras were never placed is still a build with geometry.
    func testEitherPointsOrPosesCountAsGeometry() {
        XCTAssertTrue(WorldEvidence(elements: 17_674).hasGeometry)
        XCTAssertTrue(WorldEvidence(poses: 467).hasGeometry)
        XCTAssertFalse(WorldEvidence(keyframes: 463).hasGeometry,
                       "keyframes accepted are frames, not reconstructed geometry")
        XCTAssertFalse(WorldEvidence(elements: 0, poses: 0).hasGeometry)
        XCTAssertFalse(WorldEvidence().hasGeometry)
    }

    /// The representation is the Tower's word and is quoted, never parsed.
    ///
    /// The count is **ungrouped** on purpose and this pins it: `17674`, not
    /// `17,674`. Every other Tower figure this app draws reaches the screen as
    /// the Tower's own digits, and a thousands separator inserted here — in a
    /// sentence that also carries the Tower's representation word verbatim —
    /// would be the one formatted number among them. The doc comments in this
    /// lane write "17,674" as English prose about the walk; the wire and the
    /// screen carry "17674". If that is ever to change, it changes here first.
    func testTheSummaryQuotesTheTowersOwnVocabulary() {
        let summary = FieldWalk.evidence.summary
        XCTAssertEqual(summary, "17674 sparse point cloud · 467 camera poses · 463 keyframes")
    }

    func testAnEvidenceFreeSnapshotHasNoSummary() {
        XCTAssertNil(WorldEvidence(snapshot: WorldSnapshot())?.summary)
    }
}

// MARK: - The stage vocabulary

@MainActor
final class WorldStageTests: XCTestCase {

    private func stage(
        _ state: WorldModelState,
        evidence: WorldEvidence?,
        finalization: WorldFinalizationReport? = nil
    ) -> WorldStage? {
        WorldStage.stage(for: state, evidence: evidence, finalization: finalization)
    }

    /// The field walk's evidence unless a test says otherwise. This is an
    /// overload rather than a default argument on purpose: `FieldWalk` is
    /// main-actor isolated (the app target's default isolation), and in Swift
    /// 5 language mode a default-argument expression is evaluated outside the
    /// actor, so `= FieldWalk.evidence` did not compile. The Windows lane that
    /// wrote it could not build the test target to find out.
    private func stage(
        _ state: WorldModelState,
        finalization: WorldFinalizationReport? = nil
    ) -> WorldStage? {
        stage(state, evidence: FieldWalk.evidence, finalization: finalization)
    }

    /// The states with no world in them stay out of the vocabulary. "Mapping"
    /// over an idle Tower would claim a build that nothing on the phone can
    /// start or see.
    func testStatesWithNoWorldHaveNoStage() {
        XCTAssertNil(stage(.idle, evidence: nil))
        XCTAssertNil(stage(.awaitingFirstUpdate, evidence: nil))
        XCTAssertNil(stage(.unsupported(reason: "none"), evidence: nil))
    }

    func testReceivingIsMappingUntilThereIsGeometryAndBuildingAfter() {
        XCTAssertEqual(stage(.receiving(FieldWalk.barrenSnapshot), evidence: FieldWalk.barrenEvidence), .mapping)
        XCTAssertEqual(stage(.receiving(FieldWalk.snapshot)), .building)
    }

    /// "Improving" is claimed only where a live process holds the lock and
    /// there is something to improve. Everywhere else the weaker word.
    func testImprovingIsClaimedOnlyOnEvidenceOfALiveBuildOverExistingGeometry() {
        XCTAssertEqual(
            stage(.finalizing(FieldWalk.snapshot, buildInProgress: true)), .improving
        )
        XCTAssertEqual(
            stage(.finalizing(FieldWalk.snapshot, buildInProgress: nil)), .finalizing,
            "a Tower that cannot say whether a build runs earns the weaker word"
        )
        XCTAssertEqual(
            stage(
                .finalizing(FieldWalk.barrenSnapshot, buildInProgress: true),
                evidence: FieldWalk.barrenEvidence
            ),
            .finalizing,
            "nothing to improve yet"
        )
    }

    /// The field world: interrupted, with real geometry. Never `.needsRetry`,
    /// which draws nothing, and never `.saved`, which would present a
    /// half-built walk as a complete one.
    func testAnInterruptedWorldWithGeometryIsInterruptedAndNotFailed() {
        XCTAssertEqual(stage(.interrupted(FieldWalk.snapshot, reason: "the builder exited")), .interrupted)
    }

    func testAnInterruptedWorldWithNothingInItNeedsARetry() {
        XCTAssertEqual(
            stage(
                .interrupted(FieldWalk.barrenSnapshot, reason: "the builder exited"),
                evidence: FieldWalk.barrenEvidence
            ),
            .needsRetry
        )
    }

    /// The final solve, read for the first time. A world stored as finished
    /// whose final pass was skipped or failed is `Partial`, not `Saved`.
    func testTheFinalSolvesWordDecidesSavedFromPartial() {
        func report(_ word: String?) -> WorldFinalizationReport {
            WorldFinalizationReport(state: .complete, finalSolve: word)
        }
        XCTAssertEqual(stage(.finalized(FieldWalk.snapshot), finalization: report("solved")), .saved)
        XCTAssertEqual(stage(.finalized(FieldWalk.snapshot), finalization: report("failed")), .partial)
        XCTAssertEqual(stage(.finalized(FieldWalk.snapshot), finalization: report("skipped")), .partial)
        XCTAssertEqual(stage(.finalized(FieldWalk.snapshot), finalization: report("unavailable")), .partial)
        XCTAssertEqual(
            stage(.finalized(FieldWalk.snapshot), finalization: report("pending")), .finalizing,
            "a final pass that has not run yet means the world is still going to change"
        )
        XCTAssertEqual(
            stage(.finalized(FieldWalk.snapshot), finalization: nil), .saved,
            "a record that kept no account of the final solve has not said it failed"
        )
    }

    /// A word this build has not heard of survives as itself and is not folded
    /// into the nearest known one.
    func testAnUnknownFinalSolveWordIsNotTreatedAsAFailure() {
        let odd = WorldFinalizationReport(state: .complete, finalSolve: "requeued")
        XCTAssertEqual(stage(.finalized(FieldWalk.snapshot), finalization: odd), .saved)
        XCTAssertEqual(WorldFinalSolve(word: "requeued"), .other("requeued"))
        XCTAssertTrue(WorldFinalSolve(word: "requeued").sentence?.contains("requeued") == true)
    }

    func testOnlyAWordTheTowerActuallySentProducesASentence() {
        XCTAssertNil(WorldFinalSolve.solved.sentence,
                     "a world that finished does not need a line saying it finished")
        // **Silence says nothing.** This returned a sentence beginning "This
        // world has no record of a final pass", composed out of a `nil` that
        // means any of: no pinned report has arrived yet, this Tower predates
        // the field, the world never reached finalization, or the record kept
        // no account of it. The phone cannot tell those apart — and the
        // sentence also contradicted the card above it, which called the same
        // world Saved.
        XCTAssertNil(WorldFinalSolve.notReported.sentence)
        XCTAssertNotNil(WorldFinalSolve.pending.sentence)
        XCTAssertNotNil(WorldFinalSolve.failed.sentence)
        XCTAssertNotNil(WorldFinalSolve.skipped.sentence)
        XCTAssertNotNil(WorldFinalSolve.unavailable.sentence)
    }

    /// Only the words that actually say the world did not finish. `.notReported`
    /// is absence and must not be read as a negative answer.
    func testAbsenceOfARecordIsNotAFailureRecord() {
        XCTAssertFalse(WorldFinalSolve.notReported.deniesAFinishedWorld)
        XCTAssertFalse(WorldFinalSolve.pending.deniesAFinishedWorld)
        XCTAssertFalse(WorldFinalSolve.solved.deniesAFinishedWorld)
        XCTAssertTrue(WorldFinalSolve.failed.deniesAFinishedWorld)
        XCTAssertTrue(WorldFinalSolve.skipped.deniesAFinishedWorld)
        XCTAssertTrue(WorldFinalSolve.unavailable.deniesAFinishedWorld)
    }

    func testOnlyTheUnsettledStagesSayTheWorldMayStillChange() {
        for stage in [WorldStage.mapping, .building, .improving, .finalizing] {
            XCTAssertTrue(stage.isStillChanging, "\(stage.label)")
        }
        for stage in [WorldStage.saved, .partial, .interrupted, .needsRetry] {
            XCTAssertFalse(stage.isStillChanging, "\(stage.label)")
        }
    }
}

// MARK: - Recoverability, with no rebuild route to offer

/// `tower/tower/routes/geometry.py` serves four routes and every one is a read.
/// Nothing on the phone can ask the Tower to finish or rebuild a world, so an
/// interrupted world gets a truthful sentence and never a button.
@MainActor
final class WorldRecoverabilityTests: XCTestCase {

    func testAnInterruptedWorldThatKeptItsGeometryReadsAsRecoverable() {
        let recovery = WorldRecoverability.of(
            stage: .interrupted, evidence: FieldWalk.evidence, reason: "the builder exited"
        )
        XCTAssertTrue(recovery.geometrySurvived)
        XCTAssertTrue(recovery.sentence?.contains("stored and can be opened") == true)
        XCTAssertTrue(recovery.sentence?.contains("Tower's to do") == true,
                      "the app must not imply it can ask for a rebuild")
    }

    func testAWorldWhereNothingSurvivedSaysThatInstead() {
        let recovery = WorldRecoverability.of(
            stage: .needsRetry, evidence: FieldWalk.barrenEvidence, reason: "the builder exited"
        )
        XCTAssertFalse(recovery.geometrySurvived)
        XCTAssertTrue(recovery.sentence?.contains("cannot") == true)
        XCTAssertTrue(recovery.sentence?.contains("the builder exited") == true,
                      "the Tower's own reason survives")
    }

    func testAWorldStillBeingBuiltHasLostNothingAndSaysNothing() {
        XCTAssertNil(
            WorldRecoverability.of(stage: .building, evidence: FieldWalk.evidence, reason: nil).sentence
        )
    }
}

// MARK: - The 3D ladder

@MainActor
final class WorldReconstructionLadderTests: XCTestCase {

    private let target = WorldRenderTarget(worldID: "w1", sessionID: "s1")

    func testAFinishedWorldIsTheFinalReconstruction() {
        let ladder = WorldReconstruction.ladder(
            target: target, stage: .saved, finalSolve: .solved, evidence: FieldWalk.evidence
        )
        XCTAssertEqual(ladder, .final(target))
        XCTAssertEqual(ladder.target, target)
    }

    /// Every stage that is not settled-and-finished offers the same world and
    /// says why it is not the last word. The reconstruction is still shown —
    /// "best available partial 3D world" outranks a degraded state.
    func testEveryUnfinishedStageStillOffersTheWorldWithANote() {
        for stage in [WorldStage.mapping, .building, .improving, .finalizing, .partial, .interrupted, .needsRetry] {
            let ladder = WorldReconstruction.ladder(
                target: target, stage: stage, finalSolve: .notReported, evidence: FieldWalk.evidence
            )
            XCTAssertEqual(ladder.target, target, "\(stage.label) must still offer its world")
            // `continue`, not `return`. This was `return XCTFail(...)`, so the
            // first stage that regressed hid the other six — the opposite of
            // what a loop over every case is for.
            guard case .partial(_, let note) = ladder else {
                XCTFail("\(stage.label) should be partial, got \(ladder)")
                continue
            }
            XCTAssertFalse(note.isEmpty, "\(stage.label) must say why it is not final")
        }
    }

    /// No address is a sentence, not a disabled control. And it is a
    /// **different** state from the render route answering 404, which happens
    /// later and carries the Tower's own detail.
    func testNoAddressIsExplainedInTermsOfWhatTheTowerReported() {
        let claimsGeometry = WorldReconstruction.ladder(
            target: nil, stage: .interrupted, finalSolve: .notReported, evidence: FieldWalk.evidence
        )
        XCTAssertNil(claimsGeometry.target)
        guard case .unavailable(let reason) = claimsGeometry else {
            return XCTFail("expected unavailable, got \(claimsGeometry)")
        }
        XCTAssertTrue(reason.contains("has not named where"),
                      "a world the Tower says holds geometry but gave no address for")

        let stillBuilding = WorldReconstruction.ladder(
            target: nil, stage: .mapping, finalSolve: .notReported, evidence: FieldWalk.barrenEvidence
        )
        guard case .unavailable(let building) = stillBuilding else {
            return XCTFail("expected unavailable, got \(stillBuilding)")
        }
        XCTAssertTrue(building.contains("Nothing has been built to look at yet"))

        let settled = WorldReconstruction.ladder(
            target: nil, stage: .saved, finalSolve: .solved, evidence: FieldWalk.barrenEvidence
        )
        guard case .unavailable(let none) = settled else {
            return XCTFail("expected unavailable, got \(settled)")
        }
        XCTAssertTrue(none.contains("built no 3D world"))
    }

    /// A world pinned before its first status report has a real address and an
    /// unknown standing. Offered, with that said.
    func testAPinnedWorldWithNoStageYetIsStillOffered() {
        let ladder = WorldReconstruction.ladder(
            target: target, stage: nil, finalSolve: .notReported, evidence: nil
        )
        XCTAssertEqual(ladder.target, target)
    }
}

// MARK: - The render route's address

@MainActor
final class WorldRenderViewAddressTests: XCTestCase {

    private static let host = URL(string: "http://stub.invalid")!

    /// The product view sends no `view`: a Tower that has never heard of it is
    /// never sent it. It does declare `viewer=appearance-1` (§4), which such a
    /// Tower ignores.
    func testTheProductViewSendsNoViewParameterAtAll() {
        let url = WorldRenderClient.url(
            for: WorldRenderTarget(worldID: "w1", sessionID: "s1"), baseURL: Self.host
        )
        XCTAssertEqual(url?.absoluteString,
                       "http://stub.invalid/worlds/w1/render?session_id=s1&viewer=appearance-1")

        let unpinned = WorldRenderClient.url(
            for: WorldRenderTarget(worldID: "w1", sessionID: nil), baseURL: Self.host
        )
        XCTAssertEqual(unpinned?.absoluteString, "http://stub.invalid/worlds/w1/render?viewer=appearance-1")
    }

    func testTheDiagnosticsViewAsksForItByName() {
        let url = WorldRenderClient.url(
            for: WorldRenderTarget(worldID: "w1", sessionID: "s1", view: .diagnostics),
            baseURL: Self.host
        )
        XCTAssertEqual(
            url?.absoluteString,
            "http://stub.invalid/worlds/w1/render?session_id=s1&view=diagnostics&viewer=appearance-1"
        )
    }

    /// `.sheet(item:)` and `navigationDestination(item:)` key off `id`, so the
    /// two renderings of one world must not share one.
    func testTheTwoRenderingsOfOneWorldAreDifferentItems() {
        let product = WorldRenderTarget(worldID: "w1", sessionID: "s1")
        XCTAssertNotEqual(product.id, product.showing(.diagnostics).id)
        XCTAssertNotEqual(product, product.showing(.diagnostics))
        XCTAssertEqual(product.showing(.product), product)
    }
}

// MARK: - The view model's geometry ladder, against a stubbed Tower

/// The five situations again, this time driven through
/// `WorldBuilderViewModel.geometryDidChange` with a stubbed `URLProtocol`
/// underneath, so what is asserted is the state the screen would actually be
/// in rather than a hand-built value.
@MainActor
final class WorldBuilderViewModelGeometryStatusTests: XCTestCase {

    private static let host = URL(string: "http://stub.invalid")!
    private static let manifestPath = "/worlds/w1/geometry/manifest"
    private static let segmentPath = "/worlds/w1/geometry/segment/0"

    /// A manifest whose one segment is `unresolved`: real, decodable, and with
    /// nothing the gallery can draw. The shape that made the spinner flicker.
    private static func undrawableManifest(revision: String) -> String {
        """
        {"contract": "world_builder.geometry/2026-08-25",
         "world_id": "w1", "session_id": "s1", "geometry_revision": "\(revision)",
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
            "resolution_state": "unresolved", "dominant_degeneracy": null,
            "keyframe_count": 2, "solved_count": 0, "point_count": 0,
            "bounds": null}]}
        """
    }

    /// One resolved segment with bounds: a row the gallery draws, so a missing
    /// chunk is a visibly blank tile.
    private static let drawableManifest = """
        {"contract": "world_builder.geometry/2026-08-25",
         "world_id": "w1", "session_id": "s1", "geometry_revision": "g1",
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

    private static let drawableSegmentBody = """
        {"contract": "world_builder.geometry/2026-08-25",
         "segment_index": 0, "content_hash": "h0", "frame_id": "segment:0",
         "registered": false, "transform_to_world": null,
         "poses": [{"keyframe_id": "s1:0", "status": "anchor", "degeneracy": "",
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0]}],
         "points": [[0.5, 0.5, 0.5]],
         "points_sent": 1, "points_total": 1, "point_sampling": "none"}
        """

    private func viewModel() -> WorldBuilderViewModel {
        WorldBuilderViewModel(
            client: UnavailableWorldBuilderClient(),
            geometry: WorldGeometryClient(
                baseURL: Self.host, session: StubbedGeometryProtocol.makeSession()
            )
        )
    }

    /// Nothing has happened yet. Not "an empty world".
    func testAFreshViewModelHasNoWorldRatherThanAnEmptyOne() {
        XCTAssertEqual(viewModel().geometryStatus, .noWorld)
    }

    /// The Tower's 404 on the manifest route is the Tower's own claim and is
    /// filed as such — not as a fetch failure, and not as an empty gallery.
    func testATowersFourOhFourIsRecordedAsTheTowersOwnAnswer() async {
        StubbedGeometryProtocol.reset(routes: [
            Self.manifestPath: (404, #"{"detail": "no geometry for this session"}"#),
        ])
        let model = viewModel()
        await model.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        guard case .towerReportsNone = model.geometryStatus else {
            return XCTFail("expected towerReportsNone, got \(model.geometryStatus)")
        }
        XCTAssertTrue(model.fragmentsModel.segments.isEmpty)
    }

    /// A request that never completed is the phone's failure, and says so.
    func testAnUnreachableTowerIsAFetchFailureAndNotAnEmptyWorld() async {
        StubbedGeometryProtocol.reset(routes: [:])  // every request fails at the transport
        let model = viewModel()
        await model.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        guard case .failed(let failure) = model.geometryStatus else {
            return XCTFail("expected failed, got \(model.geometryStatus)")
        }
        XCTAssertEqual(failure.kind, .unreachable)
        XCTAssertNotEqual(
            WorldGeometryAccount.account(for: model.geometryStatus, evidence: FieldWalk.evidence)
                .headline,
            "Nothing mapped yet"
        )
    }

    /// A body that is not the manifest contract. One unreadable row does this
    /// too, by design — the decoder drops the whole manifest.
    func testAnUnreadableManifestIsRecordedAsUndecodable() async {
        StubbedGeometryProtocol.reset(routes: [Self.manifestPath: (200, #"{"contract": "who?"}"#)])
        let model = viewModel()
        await model.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        guard case .failed(let failure) = model.geometryStatus else {
            return XCTFail("expected failed, got \(model.geometryStatus)")
        }
        XCTAssertEqual(failure.kind, .undecodable)
    }

    /// The render target is named whatever the manifest did, because the render
    /// route is a different question — the behaviour
    /// `testGeometryCoordinatesNameTheLiveWorldEvenWhenTheManifestFails`
    /// already pins, restated here against the ladder that now reads it.
    func testAFailedManifestStillLeavesA3DWorldToOpen() async {
        StubbedGeometryProtocol.reset(routes: [:])
        let model = viewModel()
        await model.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        XCTAssertEqual(model.renderTarget, WorldRenderTarget(worldID: "w1", sessionID: "s1"))
        XCTAssertNotNil(
            model.reconstruction.target,
            "the sparse manifest failing must not take the 3D world away"
        )
    }

    /// Opening a stored world clears whatever was drawn and does **not** claim
    /// the Tower reported nothing for the new one.
    func testOpeningAWorldSaysItsGeometryIsUnaddressedRatherThanAbsent() {
        let model = viewModel()
        model.open(worldID: "w-old", sessionID: "s-1")
        XCTAssertEqual(model.geometryStatus, .notAddressed)
        XCTAssertEqual(model.renderTarget, WorldRenderTarget(worldID: "w-old", sessionID: "s-1"))
    }

    /// Returning to live forgets the world entirely — a different thing again.
    func testReturningToLiveForgetsThatThereWasAWorldAtAll() {
        let model = viewModel()
        model.open(worldID: "w-old", sessionID: "s-1")
        model.returnToLive()
        XCTAssertEqual(model.geometryStatus, .noWorld)
        XCTAssertNil(model.renderTarget)
    }

    /// A world state arriving with no address for its geometry moves off
    /// `.noWorld`: there is a world now, and no address for its geometry. That
    /// is situations 1 and 2, and it must not read as "nothing was mapped".
    func testAWorldStateWithNoAddressIsUnaddressedAndNotEmpty() {
        let model = viewModel()
        model.stateDidChange(to: .interrupted(FieldWalk.snapshot, reason: "the builder exited"))
        XCTAssertEqual(model.geometryStatus, .notAddressed)
        XCTAssertNotEqual(model.geometryAccount.headline, "Nothing mapped yet")
        XCTAssertEqual(model.stage, .interrupted)
        XCTAssertTrue(model.recoverability?.geometrySurvived == true)
    }

    // MARK: The spinner, and the flicker it used to cause

    /// A world the phone has already fetched never goes back to a spinner.
    ///
    /// ## The loop this pins shut
    ///
    /// The guard here was `if !geometryStatus.hasDrawableGeometry`. That is
    /// false for a `.loaded` manifest whose segments are all unresolved or
    /// resolved without bounds — the normal first minute of any walk, and the
    /// permanent state of an anchors-only build — so every one of the ~30
    /// revisions a minute a live walk produces dropped the Diagnostics gallery
    /// to "Fetching the geometry…" and back. The comment above it claimed the
    /// case was handled; it only covered worlds that already had drawn tiles.
    ///
    /// The manifest here has exactly one segment and it is `unresolved`, so
    /// `fragments` is empty and `hasDrawableGeometry` is false — the shape that
    /// flickered. The count of `.loading` publishes after the first fetch must
    /// be zero.
    func testAWorldAlreadyFetchedNeverReturnsToASpinner() async {
        StubbedGeometryProtocol.reset(routes: [
            Self.manifestPath: (200, Self.undrawableManifest(revision: "g1")),
        ])
        let model = viewModel()
        await model.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        guard case .loaded(let loaded, _) = model.geometryStatus else {
            return XCTFail("expected loaded, got \(model.geometryStatus)")
        }
        XCTAssertTrue(loaded.fragments.isEmpty, "the fixture must be the undrawable shape")
        XCTAssertFalse(model.geometryStatus.hasDrawableGeometry)
        XCTAssertTrue(model.geometryStatus.hasAnswered)

        var spinners = 0
        let cancellable = model.$geometryStatus.sink { if $0 == .loading { spinners += 1 } }
        StubbedGeometryProtocol.set(
            route: Self.manifestPath, to: (200, Self.undrawableManifest(revision: "g2"))
        )
        await model.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g2")
        await model.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g3")
        cancellable.cancel()

        XCTAssertEqual(spinners, 0, "a refetch of a world already answered must be silent")
    }

    /// The first fetch does say it is fetching. The negative control for the
    /// test above, without which "never spins" would pass on a spinner that
    /// never appears at all.
    func testTheFirstFetchDoesShowASpinner() async {
        StubbedGeometryProtocol.reset(routes: [
            Self.manifestPath: (200, Self.undrawableManifest(revision: "g1")),
        ])
        let model = viewModel()
        var spinners = 0
        let cancellable = model.$geometryStatus.sink { if $0 == .loading { spinners += 1 } }
        await model.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        cancellable.cancel()
        XCTAssertEqual(spinners, 1)
    }

    // MARK: How many segments are missing

    /// Counted over the manifest's own rows, not as a difference of counts.
    ///
    /// `unfetched` was `manifest.segments.count - chunks.count`, and `chunks` is
    /// keyed by `cacheKey`. A chunk that raced a rebuild is filed under its own
    /// key rather than the key that asked for it, so it swells the dictionary
    /// while drawing nothing and the subtraction reads zero over a blank tile.
    func testTheMissingCountIsWhatTheGalleryCannotDraw() async {
        StubbedGeometryProtocol.reset(routes: [
            Self.manifestPath: (200, Self.drawableManifest),
            Self.segmentPath: (404, ""),
        ])
        let model = viewModel()
        await model.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        guard case .loaded(_, let unfetched) = model.geometryStatus else {
            return XCTFail("expected loaded, got \(model.geometryStatus)")
        }
        XCTAssertEqual(unfetched, 1, "the manifest named a segment the gallery cannot draw")

        StubbedGeometryProtocol.set(route: Self.segmentPath, to: (200, Self.drawableSegmentBody))
        await model.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        guard case .loaded(_, let afterRetry) = model.geometryStatus else {
            return XCTFail("expected loaded, got \(model.geometryStatus)")
        }
        XCTAssertEqual(afterRetry, 0, "the retry filled the gap and the count must follow")
    }

    // MARK: The final-solve sentence, and the report that used to arrive silently

    /// A finalization report that arrives **without** a state change reaches
    /// the screen.
    ///
    /// ## The staleness this pins shut
    ///
    /// `TowerWorldBuilderClient.state` drops repeats at the source
    /// (`didSet { guard state != oldValue }`), and `presentation` is a computed
    /// property that nothing else invalidates. `finalization` was read live off
    /// the client during a view body — so a Tower that moved `final_solve` from
    /// `pending` to `solved` while the snapshot stood still published nothing,
    /// redrew nothing, and left "The final pass has not run yet." on screen
    /// indefinitely. Not for one report: for the whole length of a final solve,
    /// which is exactly the window that sentence describes.
    ///
    /// The scripted client sends the report on its own, with no state beside
    /// it, which is the shape that used to arrive as nothing at all.
    func testAFinalizationReportWithNoStateChangeStillReachesTheScreen() async {
        let client = ScriptedWorldBuilderClient()
        let model = WorldBuilderViewModel(client: client)

        // A finished world whose final pass has not run. Sent once, so the
        // state is settled and will not change again.
        client.send(.finalized(FieldWalk.snapshot))
        client.send(finalization: WorldFinalizationReport(state: .pending, finalSolve: "pending"))
        var settled = await waitUntil { model.finalSolve == .pending }
        XCTAssertTrue(settled, "the report never reached the view model")
        XCTAssertEqual(model.stage, .finalizing, "a final pass that has not run is not Saved")
        XCTAssertNotNil(model.presentation.finalSolve.sentence)

        // The solve lands. **No state is sent** — the snapshot has not moved.
        var invalidations = 0
        let cancellable = model.objectWillChange.sink { _ in invalidations += 1 }
        client.send(finalization: WorldFinalizationReport(state: .complete, finalSolve: "solved"))
        settled = await waitUntil { model.finalSolve == .solved }
        cancellable.cancel()

        XCTAssertTrue(settled, "a report that changed only the finalization changed nothing")
        XCTAssertGreaterThan(invalidations, 0, "the view was never told to redraw")
        XCTAssertEqual(model.stage, .saved)
        XCTAssertNil(model.presentation.finalSolve.sentence,
                     "a world that finished needs no line saying it did")
    }

    /// A heartbeat carrying the same finalization publishes nothing. The
    /// negative control: without it, "it reaches the screen" would pass on a
    /// client that republished on every two-second tick.
    func testAnUnchangedFinalizationIsNotRepublished() async {
        let client = ScriptedWorldBuilderClient()
        let model = WorldBuilderViewModel(client: client)
        client.send(finalization: WorldFinalizationReport(state: .complete, finalSolve: "solved"))
        _ = await waitUntil { model.finalSolve == .solved }

        var republishes = 0
        let cancellable = model.$finalization.dropFirst().sink { _ in republishes += 1 }
        client.send(finalization: WorldFinalizationReport(state: .complete, finalSolve: "solved"))
        try? await Task.sleep(nanoseconds: 60_000_000)
        cancellable.cancel()
        XCTAssertEqual(republishes, 0)
    }

    private func waitUntil(
        timeout: TimeInterval = 2,
        _ condition: @MainActor () -> Bool
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return condition()
    }

    /// The end-to-end statement of the whole pass, on the field walk's own
    /// figures: a snapshot that carries geometry never produces the sentence
    /// that was drawn over it.
    func testTheFieldWalksWorldNeverSaysNothingWasMapped() async {
        let statuses: [WorldGeometryStatus] = [
            .notAddressed,
            .loading,
            .failed(WorldGeometryFailure(kind: .unreachable, detail: "timed out")),
            .failed(WorldGeometryFailure(kind: .undecodable, detail: nil)),
            .failed(WorldGeometryFailure(kind: .poseConvention, detail: nil)),
            .towerReportsNone(detail: "no geometry for this session"),
            .loaded(WorldFragmentsModel(segments: []), unfetched: 0),
        ]
        for status in statuses {
            let account = WorldGeometryAccount.account(for: status, evidence: FieldWalk.evidence)
            XCTAssertNotEqual(
                account.headline, "Nothing mapped yet",
                "\(status) over 463 keyframes and 17,674 points"
            )
        }
    }
}

// MARK: - Render page: a policy refusal is not a failure

/// `WorldRenderWebView`'s navigation policy refuses every navigation but the
/// first, and WebKit reports that refusal to the delegate as an error. The
/// coordinator must swallow exactly those and report everything else.
///
/// The WebKit half of the predicate is a hand-spelled domain and code: the
/// public `WKError` enum has no case for a policy-interrupted frame load, and
/// the campaign that wrote this line named a `WKError.Code.frameLoadInterrupted`
/// that does not exist, which is why the branch did not compile until the Mac
/// gate. This pins the spelling so a future "tidy-up" cannot re-break it.
final class WorldRenderNavigationErrorTests: XCTestCase {
    private typealias Coordinator = WorldRenderWebView.Coordinator

    func testAPolicyRefusalIsSwallowed() {
        let cancelled = NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)
        XCTAssertTrue(Coordinator.isNavigationCancellation(cancelled))

        let interrupted = NSError(
            domain: Coordinator.webKitLegacyErrorDomain,
            code: Coordinator.webKitFrameLoadInterruptedByPolicyChange
        )
        XCTAssertTrue(Coordinator.isNavigationCancellation(interrupted))

        // The literal values WebKit actually uses, so the constants cannot
        // drift to something that merely agrees with itself.
        XCTAssertEqual(Coordinator.webKitLegacyErrorDomain, "WebKitErrorDomain")
        XCTAssertEqual(Coordinator.webKitFrameLoadInterruptedByPolicyChange, 102)
    }

    func testAGenuineLoadFailureIsReported() {
        let offline = NSError(domain: NSURLErrorDomain, code: NSURLErrorNotConnectedToInternet)
        XCTAssertFalse(Coordinator.isNavigationCancellation(offline))

        let timedOut = NSError(domain: NSURLErrorDomain, code: NSURLErrorTimedOut)
        XCTAssertFalse(Coordinator.isNavigationCancellation(timedOut))

        // Same code in the wrong domain, and the wrong code in the right
        // domain, are both failures: the predicate is a pair, not either half.
        XCTAssertFalse(
            Coordinator.isNavigationCancellation(
                NSError(domain: NSURLErrorDomain, code: Coordinator.webKitFrameLoadInterruptedByPolicyChange)))
        XCTAssertFalse(
            Coordinator.isNavigationCancellation(
                NSError(domain: Coordinator.webKitLegacyErrorDomain, code: NSURLErrorCancelled)))
    }
}


// MARK: - Mac gate 2026-09-14: the words the row and the canvas share

/// `WorldListingPresentation` states an invariant in its own docstring — a
/// row that says "Interrupted" opens onto a canvas headlined "Interrupted" —
/// and two rows broke it: "Complete · no final pass" over a canvas reading
/// "Partial", and "Interrupted" over a canvas reading "Needs retry". These
/// pin the row to the canvas's own rules (`WorldStage.stage`).
@MainActor
final class WorldListingBadgeAgreementTests: XCTestCase {

    private func session(
        state: String, hasGeometry: Bool, finalSolve: String? = nil, ended: Bool = true
    ) -> WorldListingSession {
        var json: [String: Any] = [
            "session_id": "s1", "started_at": 1788894857.0,
            "frame_source": "live-capture", "has_geometry": hasGeometry, "state": state,
        ]
        if ended { json["ended_at"] = 1788895000.0 }
        if let finalSolve {
            json["finalization"] = ["state": "complete", "final_solve": finalSolve]
        }
        return WorldListingSession(json: json)!
    }

    func testACompleteRowWhoseFinalPassDidNotRunSaysPartialLikeTheCanvas() {
        for word in ["skipped", "failed", "unavailable"] {
            let row = session(state: "complete", hasGeometry: true, finalSolve: word)
            XCTAssertEqual(WorldListingPresentation.stateBadge(for: row), "Partial", word)
            // The same record through the canvas's rule.
            let stage = WorldStage.stage(
                for: .finalized(FieldWalk.snapshot), evidence: FieldWalk.evidence,
                finalization: WorldFinalizationReport(state: .complete, finalSolve: word)
            )
            XCTAssertEqual(stage, .partial, word)
        }
    }

    func testACompleteRowWithASolvedOrSilentRecordStaysComplete() {
        XCTAssertEqual(
            WorldListingPresentation.stateBadge(for: session(state: "complete", hasGeometry: true, finalSolve: "solved")),
            "Complete")
        XCTAssertEqual(
            WorldListingPresentation.stateBadge(for: session(state: "complete", hasGeometry: true)),
            "Complete")
        XCTAssertEqual(
            WorldListingPresentation.stateBadge(for: session(state: "complete", hasGeometry: true, finalSolve: "pending")),
            "Complete")
    }

    func testAnInterruptedRowWithoutGeometrySaysNeedsRetryLikeTheCanvas() {
        XCTAssertEqual(
            WorldListingPresentation.stateBadge(for: session(state: "interrupted", hasGeometry: false)),
            "Needs retry")
        XCTAssertEqual(
            WorldListingPresentation.stateBadge(for: session(state: "interrupted", hasGeometry: true)),
            "Interrupted")
        XCTAssertEqual(
            WorldStage.stage(for: .interrupted(FieldWalk.barrenSnapshot, reason: "killed"),
                             evidence: FieldWalk.barrenEvidence, finalization: nil),
            .needsRetry)
        XCTAssertEqual(
            WorldStage.stage(for: .interrupted(FieldWalk.snapshot, reason: "killed"),
                             evidence: FieldWalk.evidence, finalization: nil),
            .interrupted)
    }

    /// The caption under a row that cannot be opened names WHICH of the
    /// three "nothing to open" cases it is, because the Tower tells them
    /// apart on purpose.
    func testTheNoGeometryCaptionSaysWhichKindOfNothing() {
        XCTAssertNil(WorldListingPresentation.noGeometryCaption(for: session(state: "complete", hasGeometry: true)))
        XCTAssertEqual(
            WorldListingPresentation.noGeometryCaption(for: session(state: "receiving", hasGeometry: false, ended: false)),
            "No geometry yet. The Tower may still build it for this walk.")
        XCTAssertEqual(
            WorldListingPresentation.noGeometryCaption(for: session(state: "interrupted", hasGeometry: false)),
            "This walk's geometry is no longer on the Tower, so there is nothing to open.")
        XCTAssertEqual(
            WorldListingPresentation.noGeometryCaption(for: session(state: "unbuilt", hasGeometry: false)),
            "No geometry was built for this walk, so there is nothing to open.")
    }
}

/// Two stage rules the Mac gate tightened.
@MainActor
final class WorldStageMacGateTests: XCTestCase {

    /// A `solved` report stands in for the figures only when there are no
    /// figures. Over counted zeros it does not make a world "Saved".
    func testASolvedReportOverZeroFiguresIsNotSaved() {
        let zeros = WorldSnapshot(
            worldID: "w-zero", keyframeCount: 3,
            geometry: WorldGeometryReport(representation: "sparse point cloud", elementCount: 0, isIncremental: false),
            trajectory: WorldTrajectoryReport(poseCount: 0)
        )
        let solved = WorldFinalizationReport(state: .complete, finalSolve: "solved")
        XCTAssertEqual(
            WorldStage.stage(for: .finalized(zeros), evidence: WorldEvidence(snapshot: zeros), finalization: solved),
            .needsRetry)
        // No figures at all — an older record — and the solve's word stands.
        XCTAssertEqual(
            WorldStage.stage(for: .finalized(WorldSnapshot()), evidence: nil, finalization: solved),
            .saved)
        // And real figures are Saved with or without the word.
        XCTAssertEqual(
            WorldStage.stage(for: .finalized(FieldWalk.snapshot), evidence: FieldWalk.evidence, finalization: nil),
            .saved)
    }

    /// The `.finalizing` note is reached with a live builder holding the
    /// lock over a walk with no drawable geometry yet, under a spinner that
    /// says the Tower is finishing it. It must not say nothing is running.
    func testTheFinalizingNoteDoesNotContradictTheSpinner() {
        let target = WorldRenderTarget(worldID: "w1", sessionID: "s1")
        let ladder = WorldReconstruction.ladder(
            target: target, stage: .finalizing, finalSolve: .notReported, evidence: FieldWalk.barrenEvidence
        )
        guard case .partial(_, let note) = ladder else { return XCTFail("\(ladder)") }
        XCTAssertFalse(note.lowercased().contains("see a build running"), note)
        XCTAssertFalse(note.lowercased().contains("built again"), note)
        XCTAssertTrue(note.contains("final pass has not landed"), note)
        // A live builder without drawable geometry is `.finalizing`, not
        // `.improving` — the state that reaches this note.
        XCTAssertEqual(
            WorldStage.stage(for: .finalizing(FieldWalk.barrenSnapshot, buildInProgress: true),
                             evidence: FieldWalk.barrenEvidence, finalization: nil),
            .finalizing)
    }

    /// The record's own sentence ("The final pass was skipped, …") is drawn
    /// by the canvas on its own line under the card. The card's note must
    /// not repeat it — the Mac gate's screenshot showed it twice in a row.
    func testThePartialNoteDoesNotRepeatTheRecordsOwnSentence() {
        let target = WorldRenderTarget(worldID: "w1", sessionID: "s1")
        for solve in [WorldFinalSolve.skipped, .failed, .unavailable] {
            let ladder = WorldReconstruction.ladder(
                target: target, stage: .partial, finalSolve: solve, evidence: FieldWalk.evidence
            )
            guard case .partial(_, let note) = ladder else { XCTFail("\(ladder)"); continue }
            XCTAssertNotEqual(note, solve.sentence, "\(solve)")
            XCTAssertFalse(note.isEmpty)
        }
    }
}


/// The caption under the viewer follows the rung the Tower served.
///
/// It used to say "Not a surface" unconditionally. Once the Tower serves a
/// surface that sentence denies what is on screen, which is as wrong as a
/// caption that overclaims.
@MainActor
final class WorldRenderRepresentationTests: XCTestCase {
    private func page(_ rung: String) -> String {
        "<!doctype html><html><head><meta charset=\"utf-8\">"
            + "<meta name=\"wb-representation\" content=\"\(rung)\">"
            + "<title>x</title></head><body></body></html>"
    }

    func testEachRungIsReadFromThePage() {
        XCTAssertEqual(WorldRenderRepresentation.declared(in: page("surface")), .surface)
        XCTAssertEqual(WorldRenderRepresentation.declared(in: page("dense")), .dense)
        XCTAssertEqual(WorldRenderRepresentation.declared(in: page("sparse")), .sparse)
    }

    func testAPageFromAnOlderTowerDeclaresNothing() {
        XCTAssertNil(WorldRenderRepresentation.declared(in: "<html><head></head></html>"))
    }

    func testAnUnknownRungIsNotGuessed() {
        XCTAssertNil(WorldRenderRepresentation.declared(in: page("splat")))
    }

    func testTheTagIsOnlyLookedForInTheHead() {
        let late = String(repeating: " ", count: 5000) + page("surface")
        XCTAssertNil(WorldRenderRepresentation.declared(in: late))
    }

    func testTheSurfaceCaptionDoesNotDenyTheSurface() {
        let text = WorldRenderRepresentation.caption(for: .surface)
        XCTAssertFalse(text.localizedCaseInsensitiveContains("not a surface"))
    }

    /// Review 2, S1. The page and the contract retracted "gaps are places
    /// nothing looked"; this caption sits directly above the page and kept
    /// saying it. A hole is also where the views disagreed.
    func testTheSurfaceCaptionDoesNotPresentAGapAsProofNobodyLooked() {
        let text = WorldRenderRepresentation.caption(for: .surface)
        XCTAssertFalse(text.localizedCaseInsensitiveContains("nothing looked"))
        XCTAssertFalse(text.localizedCaseInsensitiveContains("gaps are places"))
        XCTAssertTrue(text.localizedCaseInsensitiveContains("not proof"))
        XCTAssertTrue(text.hasPrefix("Surfaces the Tower reconstructed"),
                      "TowerSmokeUITests finds the surface rung by this prefix")
    }

    func testThePointCaptionsStillRefuseToCallPointsASurface() {
        for rung in [WorldRenderRepresentation.dense, .sparse] {
            XCTAssertTrue(WorldRenderRepresentation.caption(for: rung).contains("Not a surface"))
        }
    }

    func testNoCaptionClaimsAScale() {
        // `.appearance` was missing from this list, which is how the one rung
        // whose pixels look like a photograph — and so the one a wearer is
        // likeliest to measure off — was the one rung not held to the rule
        // (review 2).
        let rungs: [WorldRenderRepresentation?] = [.appearance, .surface, .dense, .sparse, nil]
        for rung in rungs {
            XCTAssertTrue(
                WorldRenderRepresentation.caption(for: rung).localizedCaseInsensitiveContains("not to scale"),
                "caption for \(String(describing: rung)) must not imply a size"
            )
        }
    }

    func testTheStateReadsTheRungOffItsPage() {
        XCTAssertEqual(WorldRenderViewerState.ready(html: page("surface")).representation, .surface)
        XCTAssertEqual(WorldRenderViewerState.rendering(html: page("dense")).representation, .dense)
        XCTAssertNil(WorldRenderViewerState.fetching.representation)
    }
}

// MARK: - What a world opened from Saved Worlds says while it is still built

/// 2026-09-22 Mac validation of 83534e2. A "Finishing" row opened onto the
/// sparse points with no word that the photographic world was still being
/// made, while the workspace's own route into the same viewer said "it is
/// worth waiting for Saved"; and the Storage row said "Saved" under
/// "Improving" for the whole of the build.
@MainActor
final class WorldPickerOpenedNoteTests: XCTestCase {

    private let target = WorldRenderTarget(worldID: "w1", sessionID: "s1")

    private func session(state: String?, finalSolve: String? = "solved") throws -> WorldListingSession {
        var json: [String: Any] = [
            "session_id": "s1", "started_at": 1790059063.0, "ended_at": 1790059162.0,
            "end_reason": "stop", "frame_source": "live-capture", "has_geometry": true,
            "finalization": ["state": "complete", "final_solve": finalSolve as Any],
        ]
        if let state { json["state"] = state }
        return try XCTUnwrap(WorldListingSession(json: json))
    }

    private var improvingNote: String? {
        guard case .partial(_, let note) = WorldReconstruction.ladder(
            target: target, stage: .improving, finalSolve: .solved, evidence: nil
        ) else { return nil }
        return note
    }

    func testAFinishingRowSaysTheWorldIsStillBeingFinishedBeforeThePinReports() throws {
        let note = WorldPickerView.note(
            forOpened: try session(state: "finalizing"), target: target,
            pinnedStage: nil, pinnedReconstruction: nil)
        XCTAssertEqual(note, improvingNote)
        XCTAssertTrue(note?.contains("worth waiting for Saved") == true)
    }

    func testTheNoteFollowsThePinOnceItReports() throws {
        let finishing = try session(state: "finalizing")
        let stillImproving = WorldPickerView.note(
            forOpened: finishing, target: target, pinnedStage: .improving,
            pinnedReconstruction: WorldReconstruction.ladder(
                target: target, stage: .improving, finalSolve: .solved, evidence: nil))
        XCTAssertEqual(stillImproving, improvingNote)

        // The build landed while the screen was open: the sentence leaves with
        // it, rather than when the list is next refreshed.
        let landed = WorldPickerView.note(
            forOpened: finishing, target: target, pinnedStage: .saved,
            pinnedReconstruction: .final(target))
        XCTAssertNil(landed, "a finished world must not keep saying it is still being finished")
    }

    func testAReceivingRowSaysItIsStillBeingBuilt() throws {
        let note = WorldPickerView.note(
            forOpened: try session(state: "receiving"), target: target,
            pinnedStage: nil, pinnedReconstruction: nil)
        XCTAssertEqual(note, "This world is still being built, so it will change.")
    }

    /// Unchanged: a settled row says only what its final-pass record denies.
    func testASettledRowSaysWhatItAlwaysSaid() throws {
        XCTAssertNil(WorldPickerView.note(
            forOpened: try session(state: "complete"), target: target,
            pinnedStage: .improving, pinnedReconstruction: nil))
        let skipped = try session(state: "complete", finalSolve: "skipped")
        XCTAssertEqual(
            WorldPickerView.note(forOpened: skipped, target: target, pinnedStage: nil, pinnedReconstruction: nil),
            WorldFinalSolve(word: "skipped").sentence)
        XCTAssertNil(WorldPickerView.note(
            forOpened: try session(state: nil), target: target, pinnedStage: nil, pinnedReconstruction: nil))
    }

    func testTheImprovingNoteDoesNotPromiseAFewMinutes() {
        XCTAssertFalse(improvingNote?.contains("few minutes") ?? true,
                       "the photographic stages take 8-13 minutes measured, twenty planned")
    }

    func testTheStorageRowNeverSaysTheStageWordForAFinishedWorld() {
        let storage = WorldPersistenceState.saved(revision: "r1").displayName
        XCTAssertNotEqual(storage, WorldStage.saved.label)
        XCTAssertFalse(storage.localizedCaseInsensitiveContains("saved"), storage)
    }
}
