//
//  WorldPhotographicTests.swift
//  GlassesTests
//
//  The Tower's `photographic` block (`WORLD-BUILDER-IOS.md` §3a,
//  `WORLD-BUILDER-WORLDS.md` §2a), read on the status channel and on the
//  saved-worlds rows, and what the phone says from it. The three defects the
//  Mac gate B0 (2026-09-23, at e4b2372) found in the app that ignored it:
//
//  - F1: a failed photographic build read plain "Saved" / "Complete";
//  - F2: an `owed` world read "Finalizing" under two false sentences -- "The
//    Tower does not report whether a build is running" (it had: `false`) and
//    "The final pass has not landed" (it had);
//  - F3: the Saved Worlds note assumed a `finalizing` row is always a live
//    process, and promised "worth waiting for Saved" over owed work.
//
//  The payload fixtures are the Tower's own shapes at 17e6d3d, captured from a
//  local Tower on the B0 fixtures (`mac-validation/B0/wire/`).
//

import XCTest

@testable import Glasses

/// A world with real geometry, and one with none.
@MainActor
private enum PhotoWalk {
    static var snapshot: WorldSnapshot {
        WorldSnapshot(
            worldID: "w2",
            keyframeCount: 24,
            revision: "r-24",
            tracking: .good,
            scale: .unknown,
            calibration: .uncalibrated,
            geometry: WorldGeometryReport(
                representation: "sparse point cloud", elementCount: 1_200, isIncremental: false
            ),
            trajectory: WorldTrajectoryReport(poseCount: 24, segments: 1, scale: .unknown)
        )
    }

    static var evidence: WorldEvidence { WorldEvidence(snapshot: snapshot)! }

    static var barrenSnapshot: WorldSnapshot {
        WorldSnapshot(worldID: "w-barren", keyframeCount: 3, trajectory: WorldTrajectoryReport(poseCount: 0))
    }

    static var barrenEvidence: WorldEvidence { WorldEvidence(snapshot: barrenSnapshot)! }

    static func report(_ word: String, detail: String? = nil) -> WorldPhotographicReport {
        WorldPhotographicReport(state: WorldPhotographicState(rawValue: word), stage: "appearance", detail: detail)
    }

    static let solved = WorldFinalizationReport(state: .complete, finalSolve: "solved")

    /// The improving note as it has always read.
    static let liveImprovingNote = "This world is still being finished. After a long walk, "
        + "finishing can take twenty minutes or more, and the finished "
        + "world is very different from this one — it is worth waiting for Saved."
}

// MARK: - Decoding

@MainActor
final class WorldPhotographicDecodingTests: XCTestCase {

    func testEveryWordTheTowerSendsDecodesAsItself() throws {
        let words = ["complete", "running", "owed", "failed", "unattempted", "never_recorded", "unobservable"]
        for word in words {
            let report = try XCTUnwrap(WorldPhotographicReport(json: ["state": word, "stage": NSNull(), "detail": "d"]))
            XCTAssertEqual(report.state.rawValue, word)
            XCTAssertNil(report.stage)
            XCTAssertEqual(report.detail, "d")
        }
        XCTAssertEqual(WorldPhotographicReport(json: ["state": "never_recorded"])?.state, .neverRecorded)
    }

    func testTheStandingOfEachWord() {
        XCTAssertEqual(PhotoWalk.report("running").standing, .building)
        XCTAssertEqual(PhotoWalk.report("owed").standing, .owed)
        XCTAssertEqual(PhotoWalk.report("unobservable").standing, .unobservable)
        XCTAssertEqual(PhotoWalk.report("failed", detail: "why").standing, .failed(detail: "why"))
        for settled in ["complete", "unattempted", "never_recorded"] {
            XCTAssertEqual(PhotoWalk.report(settled).standing, .settled, settled)
            XCTAssertFalse(PhotoWalk.report(settled).standing.isUnfinished, settled)
        }
        for unfinished in ["running", "owed", "unobservable"] {
            XCTAssertTrue(PhotoWalk.report(unfinished).standing.isUnfinished, unfinished)
        }
    }

    /// A newer Tower's word survives as itself and says nothing the phone
    /// cannot back: the Tower has already moved `lifecycle.state` for the
    /// words that mean "not finished".
    func testAnUnknownWordSurvivesAndAddsNothing() throws {
        let report = try XCTUnwrap(WorldPhotographicReport(json: ["state": "rebuilding-later"]))
        XCTAssertEqual(report.state.rawValue, "rebuilding-later")
        XCTAssertEqual(report.standing, .settled)
    }

    /// Absent is never a word -- in particular never `complete`.
    func testNullAbsentAndWordlessBlocksAreNoReport() {
        XCTAssertNil(WorldPhotographicReport(json: nil))
        XCTAssertNil(WorldPhotographicReport(json: NSNull()))
        XCTAssertNil(WorldPhotographicReport(json: ["stage": "surface"]))
        XCTAssertNil(WorldPhotographicReport(json: ["state": ""]))
        XCTAssertNil(WorldPhotographicReport(json: ["state": 3]))
        XCTAssertNil(WorldPhotographicReport(json: "owed"))
    }

    func testEmptyStageAndDetailAreAbsent() throws {
        let report = try XCTUnwrap(WorldPhotographicReport(json: ["state": "failed", "stage": "", "detail": ""]))
        XCTAssertNil(report.stage)
        XCTAssertNil(report.detail)
        XCTAssertNil(WorldPhotographicCopy.failedReason(report.detail))
    }

    /// The status channel: `lifecycle.photographic`, verbatim from the Tower at
    /// 17e6d3d on the B0 owed fixture.
    func testTheStatusChannelBlockIsRead() throws {
        let payload: [String: Any] = [
            "model_state": "finalizing",
            "lifecycle": [
                "state": "finalizing",
                "build_in_progress": false,
                "photographic": [
                    "state": "owed", "stage": "appearance",
                    "detail": "the appearance stage is unfinished and no process is working on it",
                ],
            ],
        ]
        let report = try XCTUnwrap(WorldBuilderResultDecoder.photographic(from: payload))
        XCTAssertEqual(report.state, .owed)
        XCTAssertEqual(report.stage, "appearance")
        XCTAssertEqual(report.detail, "the appearance stage is unfinished and no process is working on it")
    }

    /// `null` is what a lifecycle computed from the record alone carries, and
    /// every Tower before 2026-09-22 sends no key at all.
    func testAStatusPayloadWithoutTheBlockHasNoReport() {
        XCTAssertNil(WorldBuilderResultDecoder.photographic(from: ["model_state": "idle"]))
        XCTAssertNil(WorldBuilderResultDecoder.photographic(from: [
            "lifecycle": ["state": "ready", "photographic": NSNull()],
        ]))
        XCTAssertNil(WorldBuilderResultDecoder.photographic(from: ["lifecycle": ["state": "ready"]]))
    }

    /// The listing row carries the same block through the same decoder.
    func testTheListingRowBlockIsRead() throws {
        var json: [String: Any] = [
            "session_id": "s1", "started_at": 1790048283.0, "ended_at": 1790048583.0,
            "frame_source": "synthetic-fixture", "has_geometry": true, "state": "complete",
            "photographic": ["state": "failed", "stage": "appearance", "detail": "fixture: the appearance encode raised"],
        ]
        let row = try XCTUnwrap(WorldListingSession(json: json))
        XCTAssertEqual(row.photographic?.state, .failed)
        XCTAssertEqual(row.photographic?.detail, "fixture: the appearance encode raised")

        json["photographic"] = nil
        XCTAssertNil(try XCTUnwrap(WorldListingSession(json: json)).photographic,
                     "an older Tower's row has no word, and no word is invented")
    }
}

// MARK: - The stage word (B0 F2)

@MainActor
final class WorldPhotographicStageTests: XCTestCase {

    private func stage(
        _ state: WorldModelState,
        evidence: WorldEvidence?,
        photographic: WorldPhotographicReport?
    ) -> WorldStage? {
        WorldStage.stage(for: state, evidence: evidence, finalization: PhotoWalk.solved, photographic: photographic)
    }

    /// The drawable walk's evidence. An overload rather than a default argument,
    /// for the reason `WorldStageTests` gives: a default-argument expression is
    /// evaluated outside the main actor in Swift 5 mode.
    private func stage(_ state: WorldModelState, photographic: WorldPhotographicReport?) -> WorldStage? {
        stage(state, evidence: PhotoWalk.evidence, photographic: photographic)
    }

    /// `owed`: `finalizing` with `build_in_progress: false`, over a world with
    /// geometry. Not finished being made, and said so.
    func testAnOwedWorldIsImprovingThoughNothingIsRunning() {
        XCTAssertEqual(
            stage(.finalizing(PhotoWalk.snapshot, buildInProgress: false), photographic: PhotoWalk.report("owed")),
            .improving)
    }

    func testAnUnobservableWorldIsImprovingWithAnUnknownBoolean() {
        XCTAssertEqual(
            stage(.finalizing(PhotoWalk.snapshot, buildInProgress: nil), photographic: PhotoWalk.report("unobservable")),
            .improving)
    }

    func testARunningWordIsImprovingEvenBetweenTwoLiveBooleans() {
        XCTAssertEqual(
            stage(.finalizing(PhotoWalk.snapshot, buildInProgress: false), photographic: PhotoWalk.report("running")),
            .improving)
        XCTAssertEqual(
            stage(.finalizing(PhotoWalk.snapshot, buildInProgress: true), photographic: PhotoWalk.report("running")),
            .improving)
    }

    /// Unchanged where the Tower said nothing more: `false` and `nil` with no
    /// photographic word stay "Finalizing", as before.
    func testWithoutAPhotographicWordTheBooleanStillDecides() {
        XCTAssertEqual(stage(.finalizing(PhotoWalk.snapshot, buildInProgress: false), photographic: nil), .finalizing)
        XCTAssertEqual(stage(.finalizing(PhotoWalk.snapshot, buildInProgress: nil), photographic: nil), .finalizing)
        XCTAssertEqual(stage(.finalizing(PhotoWalk.snapshot, buildInProgress: true), photographic: nil), .improving)
        for settled in ["complete", "failed", "unattempted", "never_recorded", "a-new-word"] {
            XCTAssertEqual(
                stage(.finalizing(PhotoWalk.snapshot, buildInProgress: false), photographic: PhotoWalk.report(settled)),
                .finalizing, settled)
        }
    }

    /// Nothing to improve yet is still the weaker word, whatever is owed.
    func testNoGeometryIsNeverImproving() {
        XCTAssertEqual(
            stage(.finalizing(PhotoWalk.barrenSnapshot, buildInProgress: false),
                  evidence: PhotoWalk.barrenEvidence, photographic: PhotoWalk.report("owed")),
            .finalizing)
    }

    /// The failed word never changes the stage: the world IS saved (§3a). What
    /// changes is the headline, below.
    func testAFailedPhotographicBuildLeavesASavedWorldSaved() {
        XCTAssertEqual(stage(.finalized(PhotoWalk.snapshot), photographic: PhotoWalk.report("failed")), .saved)
        XCTAssertEqual(stage(.finalized(PhotoWalk.snapshot), photographic: PhotoWalk.report("complete")), .saved)
        XCTAssertEqual(stage(.finalized(PhotoWalk.snapshot), photographic: nil), .saved)
    }
}

// MARK: - The canvas sentence (B0 F2)

@MainActor
final class WorldPhotographicCanvasSentenceTests: XCTestCase {

    private let falseClaim = "does not report whether a build is running"

    /// The sentence the B0 gate found false: drawn for `false`, a value the
    /// Tower had just reported. Now only for a Tower that sent neither a
    /// boolean nor a photographic word.
    func testTheDoesNotReportSentenceOnlyWhenTheTowerReportedNothing() {
        let legacy = WorldPresentation.finalizingDetail(buildInProgress: nil, photographic: nil)
        XCTAssertTrue(legacy.contains(falseClaim), legacy)
        for photographic in [nil, PhotoWalk.report("owed"), PhotoWalk.report("unobservable"),
                             PhotoWalk.report("running"), PhotoWalk.report("complete")] {
            let sentence = WorldPresentation.finalizingDetail(buildInProgress: false, photographic: photographic)
            XCTAssertFalse(sentence.contains(falseClaim), "\(String(describing: photographic)): \(sentence)")
        }
    }

    func testFalseWithNoWordSaysTheTowerReportedNothingIsBuilding() {
        let sentence = WorldPresentation.finalizingDetail(buildInProgress: false, photographic: nil)
        XCTAssertTrue(sentence.contains("reports that nothing is building"), sentence)
    }

    func testOwedHasItsOwnTrueSentence() {
        let sentence = WorldPresentation.finalizingDetail(buildInProgress: false, photographic: PhotoWalk.report("owed"))
        XCTAssertEqual(sentence, WorldPhotographicCopy.owedSentence)
        XCTAssertTrue(sentence.contains("nothing is building it right now"), sentence)
        // Not a promise the phone cannot see: the Tower's own reason says a
        // Tower with the stages switched off will not pick it up.
        XCTAssertTrue(sentence.contains("if its photographic stages are switched on"), sentence)
        XCTAssertFalse(sentence.contains("figures are not final"),
                       "an owed world's figures are the final solve's; only the photographic room is outstanding")
    }

    func testUnobservableSaysTheTowerCouldNotTell() {
        let sentence = WorldPresentation.finalizingDetail(buildInProgress: nil, photographic: PhotoWalk.report("unobservable"))
        XCTAssertEqual(sentence, WorldPhotographicCopy.unobservableSentence)
    }

    func testALiveBuildKeepsItsSentenceWhateverTheWord() {
        for photographic in [nil, PhotoWalk.report("running"), PhotoWalk.report("owed")] {
            let sentence = WorldPresentation.finalizingDetail(buildInProgress: true, photographic: photographic)
            XCTAssertTrue(sentence.hasPrefix("The Tower is still finishing this world"), sentence)
        }
    }

    func testRunningWithoutALiveBooleanSaysItIsBuilding() {
        XCTAssertEqual(
            WorldPresentation.finalizingDetail(buildInProgress: false, photographic: PhotoWalk.report("running")),
            WorldPhotographicCopy.buildingSentence)
    }
}

// MARK: - The 3D viewer's note (B0 F2)

@MainActor
final class WorldPhotographicLadderTests: XCTestCase {

    private let target = WorldRenderTarget(worldID: "w2", sessionID: "s1")

    private func note(_ stage: WorldStage, _ photographic: WorldPhotographicReport?) -> String? {
        guard case .partial(_, let note) = WorldReconstruction.ladder(
            target: target, stage: stage, finalSolve: .solved, evidence: PhotoWalk.evidence,
            photographic: photographic
        ) else { return nil }
        return note
    }

    /// The "worth waiting for Saved" promise is true of a live build and of
    /// nothing else.
    func testTheWaitingPromiseIsOnlyMadeOverALiveBuild() {
        XCTAssertEqual(note(.improving, nil), PhotoWalk.liveImprovingNote)
        XCTAssertEqual(note(.improving, PhotoWalk.report("running")), PhotoWalk.liveImprovingNote)
        for word in ["owed", "unobservable"] {
            let text = note(.improving, PhotoWalk.report(word)) ?? ""
            XCTAssertFalse(text.contains("worth waiting for Saved"), "\(word): \(text)")
            XCTAssertTrue(text.contains(WorldPhotographicCopy.finishedLooksDifferent), "\(word): \(text)")
        }
        XCTAssertTrue(note(.improving, PhotoWalk.report("owed"))?.hasPrefix(WorldPhotographicCopy.owedSentence) == true)
        XCTAssertTrue(note(.improving, PhotoWalk.report("unobservable"))?.hasPrefix(
            WorldPhotographicCopy.unobservableSentence) == true)
    }

    /// "The final pass has not landed" was false over owed work (B0 F2): the
    /// final pass had landed. It stays only where the Tower said nothing more.
    func testTheFinalPassSentenceIsNeverDrawnOverOutstandingPhotographicWork() {
        XCTAssertTrue(note(.finalizing, nil)?.contains("final pass has not landed") == true)
        for word in ["owed", "unobservable", "running"] {
            let text = note(.finalizing, PhotoWalk.report(word)) ?? ""
            XCTAssertFalse(text.contains("final pass has not landed"), "\(word): \(text)")
        }
    }
}

// MARK: - The headline over a failed build (B0 F1)

@MainActor
final class WorldPhotographicHeadlineTests: XCTestCase {

    private let target = WorldRenderTarget(worldID: "w3", sessionID: "s1")

    private func presentation(stage: WorldStage, _ photographic: WorldPhotographicReport?) -> WorldPresentation {
        WorldPresentation(
            stage: stage,
            reconstruction: WorldReconstruction.ladder(
                target: target, stage: stage, finalSolve: .solved, evidence: PhotoWalk.evidence,
                photographic: photographic),
            photographic: photographic
        )
    }

    func testASavedWorldWhosePhotographicBuildFailedDoesNotSayPlainSaved() {
        let failed = presentation(stage: .saved, PhotoWalk.report("failed", detail: "the encode raised"))
        XCTAssertEqual(failed.headline, "Saved — the photographic version could not be built")
        XCTAssertTrue(failed.isSavedWithoutItsPhotographicVersion)
        XCTAssertEqual(failed.viewerNote, "Saved — the photographic version could not be built.")
        XCTAssertEqual(WorldPhotographicCopy.failedReason("the encode raised"), "The Tower's reason: the encode raised")
    }

    /// Everything else keeps the word it had. `never_recorded` in particular
    /// must render exactly as before (§3a).
    func testEveryOtherWordKeepsTheStageLabel() {
        for word in ["complete", "unattempted", "never_recorded", "a-new-word"] {
            let saved = presentation(stage: .saved, PhotoWalk.report(word))
            XCTAssertEqual(saved.headline, "Saved", word)
            XCTAssertNil(saved.viewerNote, word)
        }
        XCTAssertEqual(presentation(stage: .saved, nil).headline, "Saved")
        XCTAssertNil(presentation(stage: .saved, nil).viewerNote)
    }

    /// A failed photographic word over a stage that already says something
    /// truer than "Saved" leaves that stage alone.
    func testOnlyASavedWorldGetsTheFailedHeadline() {
        for stage in [WorldStage.partial, .interrupted, .needsRetry, .improving, .finalizing] {
            let other = presentation(stage: stage, PhotoWalk.report("failed"))
            XCTAssertEqual(other.headline, stage.label, "\(stage)")
            XCTAssertFalse(other.isSavedWithoutItsPhotographicVersion)
        }
        XCTAssertNil(WorldPresentation(photographic: PhotoWalk.report("failed")).headline,
                     "no world, no headline")
    }
}

// MARK: - The Saved Worlds row and note (B0 F1, F3)

@MainActor
final class WorldPhotographicListingTests: XCTestCase {

    private let target = WorldRenderTarget(worldID: "w2", sessionID: "s1")

    private func row(state: String, photographic: String?, finalSolve: String = "solved") throws -> WorldListingSession {
        var json: [String: Any] = [
            "session_id": "s1", "started_at": 1790048283.0, "ended_at": 1790048583.0,
            "end_reason": "stop", "frame_source": "synthetic-fixture", "has_geometry": true,
            "state": state, "finalization": ["state": "complete", "final_solve": finalSolve],
        ]
        if let photographic {
            json["photographic"] = ["state": photographic, "stage": "appearance", "detail": "fixture"]
        }
        return try XCTUnwrap(WorldListingSession(json: json))
    }

    func testACompleteRowWhosePhotographicBuildFailedIsNotPlainComplete() throws {
        let failed = try row(state: "complete", photographic: "failed")
        XCTAssertEqual(WorldListingPresentation.stateBadge(for: failed), "Photographic failed")
        XCTAssertEqual(WorldListingPresentation.photographicCaption(for: failed),
                       "Saved — the photographic version could not be built")
    }

    func testEveryOtherCompleteRowIsUnchanged() throws {
        for word in ["complete", "unattempted", "never_recorded", nil] as [String?] {
            let settled = try row(state: "complete", photographic: word)
            XCTAssertEqual(WorldListingPresentation.stateBadge(for: settled), "Complete", "\(String(describing: word))")
            XCTAssertNil(WorldListingPresentation.photographicCaption(for: settled))
        }
    }

    /// "Partial" already says the truer thing, and a failed photographic word
    /// cannot honestly arrive over a skipped final pass anyway.
    func testPartialOutranksAFailedPhotographicWord() throws {
        let partial = try row(state: "complete", photographic: "failed", finalSolve: "skipped")
        XCTAssertEqual(WorldListingPresentation.stateBadge(for: partial), "Partial")
        XCTAssertNil(WorldListingPresentation.photographicCaption(for: partial))
    }

    // MARK: The opened note (B0 F3)

    /// The premise that was false: `finalizing` in the listing is not always a
    /// live process. An owed row must not promise "worth waiting for Saved".
    func testAnOwedFinishingRowDoesNotPromiseTheWait() throws {
        let note = WorldPickerView.note(
            forOpened: try row(state: "finalizing", photographic: "owed"), target: target,
            pinnedStage: nil, pinnedReconstruction: nil)
        XCTAssertNotNil(note)
        XCTAssertFalse(note?.contains("worth waiting for Saved") ?? true, note ?? "")
        XCTAssertTrue(note?.hasPrefix(WorldPhotographicCopy.owedSentence) == true, note ?? "")
    }

    func testAnUnobservableFinishingRowSaysTheTowerCouldNotTell() throws {
        let note = WorldPickerView.note(
            forOpened: try row(state: "finalizing", photographic: "unobservable"), target: target,
            pinnedStage: nil, pinnedReconstruction: nil)
        XCTAssertTrue(note?.hasPrefix(WorldPhotographicCopy.unobservableSentence) == true, note ?? "")
    }

    /// A live build, or an older Tower whose `finalizing` really was a held
    /// lock, keeps the note it had.
    func testARunningOrWordlessFinishingRowKeepsTheImprovingNote() throws {
        for word in ["running", nil] as [String?] {
            let note = WorldPickerView.note(
                forOpened: try row(state: "finalizing", photographic: word), target: target,
                pinnedStage: nil, pinnedReconstruction: nil)
            XCTAssertEqual(note, PhotoWalk.liveImprovingNote, "\(String(describing: word))")
        }
    }

    func testACompleteRowWhosePhotographicBuildFailedSaysSoWhenOpened() throws {
        let note = WorldPickerView.note(
            forOpened: try row(state: "complete", photographic: "failed"), target: target,
            pinnedStage: nil, pinnedReconstruction: nil)
        XCTAssertEqual(note, "Saved — the photographic version could not be built.")
        XCTAssertNil(WorldPickerView.note(
            forOpened: try row(state: "complete", photographic: "complete"), target: target,
            pinnedStage: nil, pinnedReconstruction: nil))
    }

    /// Once the pin reports, its word is newer than the row's.
    func testThePinsWordOutranksTheRowOnceItReports() throws {
        let owedRow = try row(state: "finalizing", photographic: "owed")
        // The owed build then failed while the screen was open.
        let landedFailed = WorldPickerView.note(
            forOpened: owedRow, target: target, pinnedStage: .saved,
            pinnedReconstruction: .final(target), pinnedPhotographic: PhotoWalk.report("failed"))
        XCTAssertEqual(landedFailed, "Saved — the photographic version could not be built.")
        // Or it completed.
        let landedComplete = WorldPickerView.note(
            forOpened: owedRow, target: target, pinnedStage: .saved,
            pinnedReconstruction: .final(target), pinnedPhotographic: PhotoWalk.report("complete"))
        XCTAssertNil(landedComplete)
    }
}

// MARK: - The view model (the word reaches the screen)

@MainActor
final class WorldPhotographicViewModelTests: XCTestCase {

    private func waitUntil(timeout: TimeInterval = 2, _ condition: @MainActor () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return condition()
    }

    /// `owed` → `failed` with the snapshot standing still: `state` is deduped
    /// at the source, so without its own publisher the headline would keep
    /// saying "Improving" until something else moved.
    func testAPhotographicWordWithNoStateChangeStillReachesTheScreen() async {
        let client = ScriptedWorldBuilderClient()
        let model = WorldBuilderViewModel(client: client)
        client.send(.finalizing(PhotoWalk.snapshot, buildInProgress: false))
        client.send(finalization: PhotoWalk.solved)
        client.send(photographic: PhotoWalk.report("owed"))
        var settled = await waitUntil { model.stage == .improving }
        XCTAssertTrue(settled, "an owed world never read Improving")
        XCTAssertEqual(model.presentation.headline, "Improving")

        // The Tower then settles the word on a finished world.
        client.send(.finalized(PhotoWalk.snapshot))
        client.send(photographic: PhotoWalk.report("failed", detail: "the encode raised"))
        settled = await waitUntil { model.presentation.isSavedWithoutItsPhotographicVersion }
        XCTAssertTrue(settled)
        XCTAssertEqual(model.presentation.headline, WorldPhotographicCopy.failedHeadline)
        XCTAssertEqual(model.presentation.viewerNote, WorldPhotographicCopy.failedHeadline + ".")

        // And a heartbeat carrying the same word publishes nothing.
        var republishes = 0
        let cancellable = model.$photographic.dropFirst().sink { _ in republishes += 1 }
        client.send(photographic: PhotoWalk.report("failed", detail: "the encode raised"))
        try? await Task.sleep(nanoseconds: 60_000_000)
        cancellable.cancel()
        XCTAssertEqual(republishes, 0)
    }

    /// A client with no Tower behind it has no word, never a made-up one.
    func testTheDefaultClientHasNoPhotographicWord() {
        let model = WorldBuilderViewModel(client: UnavailableWorldBuilderClient())
        XCTAssertNil(model.photographic)
        XCTAssertNil(model.presentation.photographic)
    }
}

// MARK: - Against the Tower's own output at 17e6d3d

/// The B0 fixtures -- `w1` (an appearance world with no stage record,
/// `never_recorded`), `w2` (surface ok, appearance `running` with no process:
/// `owed`), `w3` (appearance `failed`) and the sparse historical fixture --
/// served by the Tower at 17e6d3d on 127.0.0.1:8010, and captured verbatim:
/// `GET /worlds` and each world's pinned `world_builder.status` payload
/// (`mac-validation/P3.3/wire/`). Nothing below is hand-written.
@MainActor
final class WorldPhotographicTowerFixtureTests: XCTestCase {

    private static let listingJSON = #"""
{"contract":"world_builder.worlds/2026-09-10","world_count":4,"worlds":[{"created_at":1790048283.193669,"display_name":"Owed fixture (Mac B0)","live":false,"session_count":1,"sessions":[{"abandoned":false,"appearance":null,"capture_id":"appearance-fixture","dense":null,"end_reason":"stop","ended_at":1790048583.193669,"finalization":{"detail":null,"final_solve":"solved","started_at":1790048583.193669,"state":"complete","updated_at":1790048683.193669},"frame_source":"synthetic-fixture","has_geometry":true,"keyframes_accepted":24,"keyframes_journaled":0,"photographic":{"detail":"the appearance stage is unfinished and no process is working on it","stage":"appearance","state":"owed"},"session_id":"s1","started_at":1790048283.193669,"state":"finalizing"}],"updated_at":1790160840.5748532,"world_id":"w2"},{"created_at":1790048283.193669,"display_name":"Failed fixture (Mac B0)","live":false,"session_count":1,"sessions":[{"abandoned":false,"appearance":null,"capture_id":"appearance-fixture","dense":null,"end_reason":"stop","ended_at":1790048583.193669,"finalization":{"detail":null,"final_solve":"solved","started_at":1790048583.193669,"state":"complete","updated_at":1790048683.193669},"frame_source":"synthetic-fixture","has_geometry":true,"keyframes_accepted":24,"keyframes_journaled":0,"photographic":{"detail":"fixture: the appearance encode raised","stage":"appearance","state":"failed"},"session_id":"s1","started_at":1790048283.193669,"state":"complete"}],"updated_at":1790048883.193669,"world_id":"w3"},{"created_at":1790048283.193669,"display_name":"Appearance fixture (Mac)","live":false,"session_count":1,"sessions":[{"abandoned":false,"appearance":{"bytes":239242,"format":"wb-appearance-keyframes/1","imagery":"first-person keyframe imagery of a private space; best-effort face redaction with measured false negatives; not anonymised; screens, documents and bodies are not redacted","imagery_source":"redacted","keyframe_image_set":null,"keyframes":24,"keyframes_phone":24,"label_trusted":true,"privacy_safe":true,"privacy_tags":["raw-imagery","first-person"],"quality":"final","redaction":"faces-detected-and-filled/yunet-2023mar@0.30+plausibility3","redaction_effective":"faces-detected-and-filled/yunet-2023mar@0.30+plausibility3","retains_raw_imagery":true,"retention":"kept with the world under appearance/<session>/ until the session is rebuilt or the world is purged; derived from the session keyframes, so it is deleted and rebuilt, never edited","state":"served"},"capture_id":"appearance-fixture","dense":null,"end_reason":"stop","ended_at":1790048583.193669,"finalization":{"detail":null,"final_solve":"solved","started_at":1790048583.193669,"state":"complete","updated_at":1790048683.193669},"frame_source":"synthetic-fixture","has_geometry":true,"keyframes_accepted":24,"keyframes_journaled":0,"photographic":{"detail":"no Tower ever ran a photographic stage for this session, which is not the same as one that tried and failed","stage":null,"state":"never_recorded"},"session_id":"s1","started_at":1790048283.193669,"state":"complete"}],"updated_at":1790048883.193669,"world_id":"w1"},{"created_at":1788719719.9311092,"display_name":"Synthetic room (Mac fixture)","live":false,"session_count":2,"sessions":[{"abandoned":false,"appearance":null,"capture_id":null,"dense":null,"end_reason":"stop","ended_at":1788720319.9311092,"finalization":{"at":1788720319.9311092,"detail":null,"final_solve":"skipped","state":"complete"},"frame_source":"synthetic","has_geometry":true,"keyframes_accepted":30,"keyframes_journaled":30,"photographic":{"detail":"this session has no finished global solve to build a photographic room from","stage":null,"state":"unattempted"},"session_id":"1111aaaa2222bbbb3333cccc4444dddd","started_at":1788719719.9311092,"state":"complete"},{"abandoned":false,"appearance":null,"capture_id":null,"dense":null,"end_reason":"stop","ended_at":1788721419.9311092,"finalization":null,"frame_source":"synthetic","has_geometry":false,"keyframes_accepted":3,"keyframes_journaled":0,"photographic":{"detail":"this session has no finished global solve to build a photographic room from","stage":null,"state":"unattempted"},"session_id":"5555eeee6666ffff7777000088881111","started_at":1788721319.9311092,"state":"unbuilt"}],"updated_at":1788723259.9311092,"world_id":"a1b2c3d4e5f60718293a4b5c6d7e8f90"}]}
"""#

    private static let w1Payload = #"""
{"artifacts":{"images_purged_declared":false,"images_purged_meaning":"a declaration that rebuilds are refused for this world, not a verified deletion of the imagery","images_purged_verified":null,"keyframe_images":{"count":24,"fetchable":false,"present":true,"reason":"no artifact transfer contract exists, and these remain first-person frames whose redaction is best-effort with measured false negatives; a consumer must withhold imagery it cannot verify","redaction":"faces-detected-and-filled/yunet-2023mar@0.30+plausibility3","reredacted_set":null}},"calibration":{"calibrated_height":120,"calibrated_width":160,"calibrating_ever_reported":false,"reprojection_rms_px":null,"scope":"session","source":"self_calibrated","state":"calibrated","view_count":null},"geometry":{"available":true,"backend_id":"synthetic","built_at":1790048633.193669,"built_from_keyframes":24,"confidence":null,"current":true,"element_count":16,"element_name":"point","is_incremental":false,"keyframes_now":24,"provenance":"inferred","representation":"sparse point cloud","revision":"f0f3c85ee0c76a3f","stale_reason":null,"time_basis":"tower-receipt","unavailable_reason":null},"lifecycle":{"build_in_progress":false,"build_in_progress_unavailable_reason":null,"evidence":"session_stopped was written, the finalization record is complete, the lock was released and the derived tree is there","finalization":{"detail":null,"final_solve":"solved","started_at":1790048583.193669,"state":"complete","updated_at":1790048683.193669},"photographic":{"detail":"no Tower ever ran a photographic stage for this session, which is not the same as one that tried and failed","stage":null,"state":"never_recorded"},"reason":null,"state":"ready"},"model_state":"finalized","model_state_reason":null,"persistence":{"images_purged":false,"location_disclosed":false,"revision":"ec0b10ba35f9dff5","state":"saved"},"progress":{"frames_observed":24,"frames_observed_unavailable_reason":null,"frames_rejected_malformed":0,"frames_rejected_wrong_size":0,"journal_corrupt_lines":0,"keyframes_accepted":24,"keyframes_accepted_provenance":"measured","keyframes_accepted_source":"session record","mapping_clock":"tower","mapping_seconds":300.0,"mapping_seconds_unavailable_reason":null,"rejected_by_reason":{},"time_basis":"tower-receipt"},"scale":{"allows_metres":false,"confidence":"unknown","meters_per_unit":null,"method":null,"semantics":null,"state":"unknown","unavailable_reason":null,"unit":null},"selection":{"mode":"pinned","reason":"the client named this world","session_id":"s1","world_id":"w1"},"session":{"capture_id":"appearance-fixture","end_reason":"stop","ended_at":1790048583.193669,"frame_source":"synthetic-fixture","retains_raw_imagery":true,"session_id":"s1","started_at":1790048283.193669},"time_basis":"tower-receipt","tracking":{"evidence":"no keyframe has been accepted and no loss recorded","limited_ever_reported":false,"state":"unknown"},"trajectory":{"available":true,"built_from_keyframes":24,"chain_breaks":0,"confidence":null,"current":true,"keyframes":24,"keyframes_now":24,"path_length":{"available":false,"reason":"this world has no scale state, so a distance figure could not be labelled and would not be renderable"},"pose_count":24,"poses_anchor":0,"poses_refused":0,"poses_solved":24,"provenance":"inferred","revision":"3a7c5dea7099b746","segments":1,"stale_reason":null,"tracking_restarts":0,"unavailable_reason":null},"world":{"created_at":1790048283.193669,"display_name":"Appearance fixture (Mac)","schema_version":1,"updated_at":1790048883.193669,"world_id":"w1"},"world_snapshot":{"calibration":"calibrated","geometry":{"element_count":16,"is_incremental":false,"representation":"sparse point cloud"},"keyframe_count":24,"mapping_seconds":300.0,"name":"Appearance fixture (Mac)","persistence":{"revision":"ec0b10ba35f9dff5","state":"saved"},"revision":"538a57285972e659","scale":"unknown","tracking":"unavailable","trajectory":{"path_length":null,"path_length_unit":null,"pose_count":24,"scale":"unknown"},"world_id":"w1"}}
"""#

    private static let w2Payload = #"""
{"artifacts":{"images_purged_declared":false,"images_purged_meaning":"a declaration that rebuilds are refused for this world, not a verified deletion of the imagery","images_purged_verified":null,"keyframe_images":{"count":24,"fetchable":false,"present":true,"reason":"no artifact transfer contract exists, and these remain first-person frames whose redaction is best-effort with measured false negatives; a consumer must withhold imagery it cannot verify","redaction":"faces-detected-and-filled/yunet-2023mar@0.30+plausibility3","reredacted_set":null}},"calibration":{"calibrated_height":120,"calibrated_width":160,"calibrating_ever_reported":false,"reprojection_rms_px":null,"scope":"session","source":"self_calibrated","state":"calibrated","view_count":null},"geometry":{"available":true,"backend_id":"synthetic","built_at":1790048633.193669,"built_from_keyframes":24,"confidence":null,"current":true,"element_count":16,"element_name":"point","is_incremental":false,"keyframes_now":24,"provenance":"inferred","representation":"sparse point cloud","revision":"f0f3c85ee0c76a3f","stale_reason":null,"time_basis":"tower-receipt","unavailable_reason":null},"lifecycle":{"build_in_progress":false,"build_in_progress_unavailable_reason":null,"evidence":"session_stopped was written, the finalization record is complete, the lock was released and the derived tree is there, and the appearance stage is unfinished and no process is working on it","finalization":{"detail":null,"final_solve":"solved","started_at":1790048583.193669,"state":"complete","updated_at":1790048683.193669},"photographic":{"detail":"the appearance stage is unfinished and no process is working on it","stage":"appearance","state":"owed"},"reason":"this world does not have the photographic representation it is owed yet: the appearance stage is unfinished and no process is working on it. A Tower with the photographic stages enabled finishes owed work the next time it is idle; one that has them switched off will not, and this world then stays as it is until somebody runs scripts/world_finish_pending.py by hand","state":"finalizing"},"model_state":"finalizing","model_state_reason":"this world does not have the photographic representation it is owed yet: the appearance stage is unfinished and no process is working on it. A Tower with the photographic stages enabled finishes owed work the next time it is idle; one that has them switched off will not, and this world then stays as it is until somebody runs scripts/world_finish_pending.py by hand","persistence":{"images_purged":false,"location_disclosed":false,"revision":"761c2421198d7f92","state":"saved"},"progress":{"frames_observed":24,"frames_observed_unavailable_reason":null,"frames_rejected_malformed":0,"frames_rejected_wrong_size":0,"journal_corrupt_lines":0,"keyframes_accepted":24,"keyframes_accepted_provenance":"measured","keyframes_accepted_source":"session record","mapping_clock":"tower","mapping_seconds":300.0,"mapping_seconds_unavailable_reason":null,"rejected_by_reason":{},"time_basis":"tower-receipt"},"scale":{"allows_metres":false,"confidence":"unknown","meters_per_unit":null,"method":null,"semantics":null,"state":"unknown","unavailable_reason":null,"unit":null},"selection":{"mode":"pinned","reason":"the client named this world","session_id":"s1","world_id":"w2"},"session":{"capture_id":"appearance-fixture","end_reason":"stop","ended_at":1790048583.193669,"frame_source":"synthetic-fixture","retains_raw_imagery":true,"session_id":"s1","started_at":1790048283.193669},"time_basis":"tower-receipt","tracking":{"evidence":"no keyframe has been accepted and no loss recorded","limited_ever_reported":false,"state":"unknown"},"trajectory":{"available":true,"built_from_keyframes":24,"chain_breaks":0,"confidence":null,"current":true,"keyframes":24,"keyframes_now":24,"path_length":{"available":false,"reason":"this world has no scale state, so a distance figure could not be labelled and would not be renderable"},"pose_count":24,"poses_anchor":0,"poses_refused":0,"poses_solved":24,"provenance":"inferred","revision":"3a7c5dea7099b746","segments":1,"stale_reason":null,"tracking_restarts":0,"unavailable_reason":null},"world":{"created_at":1790048283.193669,"display_name":"Owed fixture (Mac B0)","schema_version":1,"updated_at":1790160840.5748532,"world_id":"w2"},"world_snapshot":{"calibration":"calibrated","geometry":{"element_count":16,"is_incremental":false,"representation":"sparse point cloud"},"keyframe_count":24,"mapping_seconds":300.0,"name":"Owed fixture (Mac B0)","persistence":{"revision":"761c2421198d7f92","state":"saved"},"revision":"4e9b7ee86b40749b","scale":"unknown","tracking":"unavailable","trajectory":{"path_length":null,"path_length_unit":null,"pose_count":24,"scale":"unknown"},"world_id":"w2"}}
"""#

    private static let w3Payload = #"""
{"artifacts":{"images_purged_declared":false,"images_purged_meaning":"a declaration that rebuilds are refused for this world, not a verified deletion of the imagery","images_purged_verified":null,"keyframe_images":{"count":24,"fetchable":false,"present":true,"reason":"no artifact transfer contract exists, and these remain first-person frames whose redaction is best-effort with measured false negatives; a consumer must withhold imagery it cannot verify","redaction":"faces-detected-and-filled/yunet-2023mar@0.30+plausibility3","reredacted_set":null}},"calibration":{"calibrated_height":120,"calibrated_width":160,"calibrating_ever_reported":false,"reprojection_rms_px":null,"scope":"session","source":"self_calibrated","state":"calibrated","view_count":null},"geometry":{"available":true,"backend_id":"synthetic","built_at":1790048633.193669,"built_from_keyframes":24,"confidence":null,"current":true,"element_count":16,"element_name":"point","is_incremental":false,"keyframes_now":24,"provenance":"inferred","representation":"sparse point cloud","revision":"f0f3c85ee0c76a3f","stale_reason":null,"time_basis":"tower-receipt","unavailable_reason":null},"lifecycle":{"build_in_progress":false,"build_in_progress_unavailable_reason":null,"evidence":"session_stopped was written, the finalization record is complete, the lock was released and the derived tree is there","finalization":{"detail":null,"final_solve":"solved","started_at":1790048583.193669,"state":"complete","updated_at":1790048683.193669},"photographic":{"detail":"fixture: the appearance encode raised","stage":"appearance","state":"failed"},"reason":null,"state":"ready"},"model_state":"finalized","model_state_reason":null,"persistence":{"images_purged":false,"location_disclosed":false,"revision":"8d07a20997ff9a09","state":"saved"},"progress":{"frames_observed":24,"frames_observed_unavailable_reason":null,"frames_rejected_malformed":0,"frames_rejected_wrong_size":0,"journal_corrupt_lines":0,"keyframes_accepted":24,"keyframes_accepted_provenance":"measured","keyframes_accepted_source":"session record","mapping_clock":"tower","mapping_seconds":300.0,"mapping_seconds_unavailable_reason":null,"rejected_by_reason":{},"time_basis":"tower-receipt"},"scale":{"allows_metres":false,"confidence":"unknown","meters_per_unit":null,"method":null,"semantics":null,"state":"unknown","unavailable_reason":null,"unit":null},"selection":{"mode":"pinned","reason":"the client named this world","session_id":"s1","world_id":"w3"},"session":{"capture_id":"appearance-fixture","end_reason":"stop","ended_at":1790048583.193669,"frame_source":"synthetic-fixture","retains_raw_imagery":true,"session_id":"s1","started_at":1790048283.193669},"time_basis":"tower-receipt","tracking":{"evidence":"no keyframe has been accepted and no loss recorded","limited_ever_reported":false,"state":"unknown"},"trajectory":{"available":true,"built_from_keyframes":24,"chain_breaks":0,"confidence":null,"current":true,"keyframes":24,"keyframes_now":24,"path_length":{"available":false,"reason":"this world has no scale state, so a distance figure could not be labelled and would not be renderable"},"pose_count":24,"poses_anchor":0,"poses_refused":0,"poses_solved":24,"provenance":"inferred","revision":"3a7c5dea7099b746","segments":1,"stale_reason":null,"tracking_restarts":0,"unavailable_reason":null},"world":{"created_at":1790048283.193669,"display_name":"Failed fixture (Mac B0)","schema_version":1,"updated_at":1790048883.193669,"world_id":"w3"},"world_snapshot":{"calibration":"calibrated","geometry":{"element_count":16,"is_incremental":false,"representation":"sparse point cloud"},"keyframe_count":24,"mapping_seconds":300.0,"name":"Failed fixture (Mac B0)","persistence":{"revision":"8d07a20997ff9a09","state":"saved"},"revision":"1b09c37128cff3eb","scale":"unknown","tracking":"unavailable","trajectory":{"path_length":null,"path_length_unit":null,"pose_count":24,"scale":"unknown"},"world_id":"w3"}}
"""#

    private func json(_ text: String) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }

    private func presentation(_ text: String) throws -> (WorldModelState, WorldPresentation, Bool?) {
        let payload = try json(text)
        let state = try XCTUnwrap(WorldBuilderResultDecoder.modelState(from: payload))
        let evidence = WorldEvidence(snapshot: state.snapshot)
        let finalization = WorldBuilderResultDecoder.finalization(from: payload)
        let photographic = WorldBuilderResultDecoder.photographic(from: payload)
        let stage = WorldStage.stage(for: state, evidence: evidence, finalization: finalization,
                                     photographic: photographic)
        let target = WorldRenderTarget(worldID: state.snapshot?.worldID ?? "", sessionID: "s1")
        let finalSolve = WorldFinalSolve(word: finalization?.finalSolve)
        let presentation = WorldPresentation(
            stage: stage, evidence: evidence, finalSolve: finalSolve,
            reconstruction: WorldReconstruction.ladder(
                target: target, stage: stage, finalSolve: finalSolve,
                evidence: evidence, photographic: photographic),
            photographic: photographic)
        var building: Bool?
        if case .finalizing(_, let flag) = state { building = flag }
        return (state, presentation, building)
    }

    /// `w2`, owed: "Improving", no spinner, the owed sentence on the canvas and
    /// in the viewer's note -- never "does not report whether a build is
    /// running" and never "the final pass has not landed".
    func testTheOwedWorldAsTheTowerSendsIt() throws {
        let (state, presentation, building) = try presentation(Self.w2Payload)
        guard case .finalizing = state else { return XCTFail("\(state)") }
        XCTAssertEqual(building, false, "the Tower says, honestly, that nothing is building it")
        XCTAssertEqual(presentation.photographic?.state, .owed)
        XCTAssertEqual(presentation.stage, .improving)
        XCTAssertEqual(presentation.headline, "Improving")
        let canvas = WorldPresentation.finalizingDetail(
            buildInProgress: building, photographic: presentation.photographic)
        XCTAssertEqual(canvas, WorldPhotographicCopy.owedSentence)
        XCTAssertFalse(canvas.contains("does not report whether a build is running"))
        let viewer = try XCTUnwrap(presentation.viewerNote)
        XCTAssertTrue(viewer.hasPrefix(WorldPhotographicCopy.owedSentence), viewer)
        XCTAssertFalse(viewer.contains("final pass has not landed"), viewer)
        XCTAssertFalse(viewer.contains("worth waiting for Saved"), viewer)
    }

    /// `w3`, failed: saved, and says the photographic version could not be
    /// built, with the Tower's reason.
    func testTheFailedWorldAsTheTowerSendsIt() throws {
        let (state, presentation, _) = try presentation(Self.w3Payload)
        guard case .finalized = state else { return XCTFail("\(state)") }
        XCTAssertEqual(presentation.stage, .saved)
        XCTAssertEqual(presentation.headline, "Saved — the photographic version could not be built")
        XCTAssertEqual(presentation.viewerNote, "Saved — the photographic version could not be built.")
        guard case .failed(let detail) = presentation.photographic?.standing else {
            return XCTFail("\(String(describing: presentation.photographic))")
        }
        XCTAssertEqual(WorldPhotographicCopy.failedReason(detail),
                       "The Tower's reason: fixture: the appearance encode raised")
    }

    /// `w1`, never_recorded: exactly as before -- "Saved", no note (§3a).
    func testANeverRecordedWorldRendersExactlyAsBefore() throws {
        let (state, presentation, _) = try presentation(Self.w1Payload)
        guard case .finalized = state else { return XCTFail("\(state)") }
        XCTAssertEqual(presentation.photographic?.state, .neverRecorded)
        XCTAssertEqual(presentation.headline, "Saved")
        XCTAssertNil(presentation.viewerNote)
        XCTAssertFalse(presentation.isSavedWithoutItsPhotographicVersion)
    }

    /// The Saved Worlds rows for the same four worlds: badge, caption, and the
    /// note under the viewer when a row is opened before its pin reports.
    func testTheSavedWorldsRowsAsTheTowerListsThem() throws {
        let listing = try XCTUnwrap(WorldListingDecoder.listing(from: try json(Self.listingJSON)),
                                    "the listing contract did not decode")
        func row(_ world: String) throws -> WorldListingSession {
            try XCTUnwrap(listing.worlds.first { $0.worldID == world }?.sessions.first)
        }
        let target = WorldRenderTarget(worldID: "w", sessionID: "s1")
        func note(_ session: WorldListingSession) -> String? {
            WorldPickerView.note(forOpened: session, target: target, pinnedStage: nil, pinnedReconstruction: nil)
        }

        let owed = try row("w2")
        XCTAssertEqual(WorldListingPresentation.stateBadge(for: owed), "Finishing")
        XCTAssertNil(WorldListingPresentation.photographicCaption(for: owed))
        XCTAssertTrue(note(owed)?.hasPrefix(WorldPhotographicCopy.owedSentence) == true, note(owed) ?? "")

        let failed = try row("w3")
        XCTAssertEqual(WorldListingPresentation.stateBadge(for: failed), "Photographic failed")
        XCTAssertEqual(WorldListingPresentation.photographicCaption(for: failed),
                       "Saved — the photographic version could not be built")
        XCTAssertEqual(note(failed), "Saved — the photographic version could not be built.")

        let old = try row("w1")
        XCTAssertEqual(WorldListingPresentation.stateBadge(for: old), "Complete")
        XCTAssertNil(WorldListingPresentation.photographicCaption(for: old))
        XCTAssertNil(note(old))

        // The sparse historical fixture: `unattempted` on both rows, nothing
        // new said about either. Its first walk's final pass was skipped, so
        // it read "Partial" before this change and still does.
        let historical = try XCTUnwrap(listing.worlds.first { $0.worldID.hasPrefix("a1b2") })
        XCTAssertEqual(historical.sessions.map { WorldListingPresentation.stateBadge(for: $0) },
                       ["Partial", "No geometry"])
        XCTAssertEqual(historical.sessions.compactMap(\.photographic?.state), [.unattempted, .unattempted])
        XCTAssertTrue(historical.sessions.allSatisfy { WorldListingPresentation.photographicCaption(for: $0) == nil })
    }
}
