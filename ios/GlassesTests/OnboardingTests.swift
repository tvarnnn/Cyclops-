//
//  OnboardingTests.swift
//  GlassesTests
//
//  U0.3: "How Glasses works", the first-run cards. The flag (seen once,
//  marked seen on finish or skip), the gate (never over a capture that is
//  running or starting), every way in and out of the cards, and the words
//  (plain, the four walking habits, symbols that exist).
//

import Foundation
import UIKit
import XCTest

@testable import Glasses

// MARK: - The flag

final class OnboardingStoreTests: XCTestCase {

    private var suiteName = ""
    private var defaults: UserDefaults!

    override func setUp() {
        super.setUp()
        suiteName = "OnboardingStoreTests.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suiteName)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suiteName)
        defaults = nil
        super.tearDown()
    }

    func testANewPhoneHasNotSeenTheCards() {
        XCTAssertFalse(OnboardingStore(defaults: defaults).isSeen)
    }

    func testSeenPersistsForTheNextLaunchAndResetForgetsIt() {
        OnboardingStore(defaults: defaults).markSeen()
        XCTAssertTrue(OnboardingStore(defaults: defaults).isSeen, "a new store on the same defaults reads it")
        XCTAssertTrue(defaults.bool(forKey: "onboardingSeen"))

        OnboardingStore(defaults: defaults).reset()
        XCTAssertFalse(OnboardingStore(defaults: defaults).isSeen)
        XCTAssertNil(defaults.object(forKey: "onboardingSeen"))
    }

    /// The names UI tests and people with `defaults` depend on.
    func testTheKeyAndLaunchArgumentsAreStable() {
        XCTAssertEqual(OnboardingStore.key, "onboardingSeen")
        XCTAssertEqual(OnboardingStore.uiTestSkipArgument, "-UITestSkipOnboarding")
        XCTAssertEqual(OnboardingStore.uiTestResetArgument, "-UITestResetOnboarding")
    }
}

// MARK: - The gate

final class OnboardingGateTests: XCTestCase {

    private let allClaims: [CaptureClaim] = [.unclaimed, .running, .devicePaused, .ending]

    /// The phases with no payload, which is every one the Start path passes
    /// through, plus the ends a person sees afterwards.
    private let phases: [ObjectMemoryRecordingPhase] = [
        .idle, .starting, .waitingToBeFollowed, .notObserved, .remembering, .pausing, .paused,
        .resuming, .stopping, .stopped, .stillFollowing(after: .start), .unsupported,
    ]

    func testACaptureIsBusyWheneverTheCameraIsClaimed() {
        for claim in allClaims {
            for phase in phases {
                let busy = OnboardingGate.isCaptureBusy(claim: claim, recordingPhase: phase)
                if claim != .unclaimed {
                    XCTAssertTrue(busy, "\(claim) with \(phase)")
                }
            }
        }
    }

    /// The glasses pausing themselves is still a capture: the session is held.
    func testADevicePauseIsBusy() {
        XCTAssertTrue(OnboardingGate.isCaptureBusy(claim: .devicePaused, recordingPhase: .idle))
    }

    /// Object Memory asks the Tower before it asks the camera, so it is
    /// starting a capture before the claim moves.
    func testObjectMemoryStartingIsBusyBeforeTheCameraIsClaimed() {
        XCTAssertTrue(OnboardingGate.isCaptureBusy(claim: .unclaimed, recordingPhase: .starting))
        for phase in phases where phase != .starting {
            XCTAssertFalse(OnboardingGate.isCaptureBusy(claim: .unclaimed, recordingPhase: phase), "\(phase)")
        }
    }

    func testTheCardsOpenAtLaunchOnlyWhenUnseenUnskippedAndIdle() {
        for isSeen in [false, true] {
            for skip in [false, true] {
                for busy in [false, true] {
                    let arguments = skip ? ["Glasses", OnboardingStore.uiTestSkipArgument] : ["Glasses"]
                    XCTAssertEqual(
                        OnboardingGate.presentsAtLaunch(isSeen: isSeen, launchArguments: arguments, isCaptureBusy: busy),
                        !isSeen && !skip && !busy,
                        "seen \(isSeen) skip \(skip) busy \(busy)"
                    )
                }
            }
        }
    }

    func testSettingsCanReopenThemOnlyWhenIdle() {
        XCTAssertTrue(OnboardingGate.canReopen(isCaptureBusy: false))
        XCTAssertFalse(OnboardingGate.canReopen(isCaptureBusy: true))
    }
}

// MARK: - Every way in and out

final class OnboardingPresentationTests: XCTestCase {

    private var suiteName = ""
    private var store: OnboardingStore!

    override func setUp() {
        super.setUp()
        suiteName = "OnboardingPresentationTests.\(UUID().uuidString)"
        store = OnboardingStore(defaults: UserDefaults(suiteName: suiteName)!)
    }

    override func tearDown() {
        store.defaults.removePersistentDomain(forName: suiteName)
        store = nil
        super.tearDown()
    }

    private func launched(busy: Bool = false, arguments: [String] = ["Glasses"]) -> OnboardingPresentation {
        var presentation = OnboardingPresentation()
        presentation.launched(isSeen: store.isSeen, launchArguments: arguments, isCaptureBusy: busy)
        return presentation
    }

    func testAFirstLaunchShowsTheFirstRunCards() {
        XCTAssertEqual(launched().showing, .firstRun)
    }

    func testFinishingMarksThemSeenAndTheNextLaunchShowsNothing() {
        var presentation = launched()
        presentation.ended(.finished, store: store)
        XCTAssertNil(presentation.showing)
        XCTAssertTrue(store.isSeen)
        XCTAssertNil(launched().showing)
    }

    func testSkippingMarksThemSeenToo() {
        var presentation = launched()
        presentation.ended(.skipped, store: store)
        XCTAssertNil(presentation.showing)
        XCTAssertTrue(store.isSeen)
        XCTAssertNil(launched().showing)
    }

    func testTheSkipArgumentOpensNothingAndLeavesTheFlag() {
        XCTAssertNil(launched(arguments: ["Glasses", "-UITestSkipOnboarding"]).showing)
        XCTAssertFalse(store.isSeen)
    }

    func testNotAtLaunchDuringACapture() {
        XCTAssertNil(launched(busy: true).showing)
        XCTAssertFalse(store.isSeen, "not shown is not seen")
    }

    /// Closed, not marked: an interrupted first run returns at the next
    /// launch, not in the middle of the walk.
    func testACaptureStartingClosesThemWithoutMarkingThemSeen() {
        var presentation = launched()
        presentation.captureChanged(isBusy: true)
        XCTAssertNil(presentation.showing)
        XCTAssertFalse(store.isSeen)
        XCTAssertEqual(launched().showing, .firstRun)
    }

    func testACaptureEndingOpensNothing() {
        var presentation = OnboardingPresentation()
        presentation.captureChanged(isBusy: false)
        XCTAssertNil(presentation.showing)
    }

    func testSettingsOpensThemOnlyOnceItHasClosed() {
        store.markSeen()
        var presentation = launched()
        XCTAssertNil(presentation.showing)

        presentation.requestedFromSettings(isCaptureBusy: false)
        XCTAssertNil(presentation.showing, "not over the closing sheet")
        XCTAssertEqual(presentation.pending, .revisit)

        presentation.sheetDismissed(isCaptureBusy: false)
        XCTAssertEqual(presentation.showing, .revisit)
        XCTAssertNil(presentation.pending)

        presentation.ended(.finished, store: store)
        XCTAssertNil(presentation.showing)
    }

    func testSettingsOpensNothingDuringACapture() {
        var presentation = OnboardingPresentation()
        presentation.requestedFromSettings(isCaptureBusy: true)
        XCTAssertNil(presentation.pending)
        presentation.sheetDismissed(isCaptureBusy: true)
        XCTAssertNil(presentation.showing)
    }

    /// A capture that starts while Settings is closing wins.
    func testACaptureStartingBeforeSettingsHasClosedCancelsTheRequest() {
        var viaWatch = OnboardingPresentation()
        viaWatch.requestedFromSettings(isCaptureBusy: false)
        viaWatch.captureChanged(isBusy: true)
        viaWatch.sheetDismissed(isCaptureBusy: false)
        XCTAssertNil(viaWatch.showing)

        var viaDismissal = OnboardingPresentation()
        viaDismissal.requestedFromSettings(isCaptureBusy: false)
        viaDismissal.sheetDismissed(isCaptureBusy: true)
        XCTAssertNil(viaDismissal.showing)
        XCTAssertNil(viaDismissal.pending)
    }

    /// Settings closed with Done: nothing was asked for.
    func testAnOrdinarySheetDismissalOpensNothing() {
        var presentation = OnboardingPresentation()
        presentation.sheetDismissed(isCaptureBusy: false)
        XCTAssertNil(presentation.showing)
    }

    func testClosingWhatIsNotShowingMarksNothing() {
        var presentation = OnboardingPresentation()
        presentation.ended(.finished, store: store)
        XCTAssertFalse(store.isSeen)

        var dismissed = launched()
        dismissed.coverDismissed()
        XCTAssertNil(dismissed.showing)
        XCTAssertFalse(store.isSeen, "SwiftUI taking the cover down is not the person ending it")
    }

    /// Every sequence of up to five events, from a first launch: the cards are
    /// never on screen while a capture is busy, and they are marked seen only
    /// by a finish or a skip of cards that were showing.
    func testTheCardsAreNeverShownDuringACaptureWhateverHappens() {
        enum Event: CaseIterable {
            case launch, settings, sheetClosed, captureStarts, captureEnds, finish, skip
        }
        let local = store!
        func explore(_ prefix: [Event], depth: Int) {
            if depth == 0 { return }
            for event in Event.allCases {
                let events = prefix + [event]
                local.reset()
                var presentation = OnboardingPresentation()
                var busy = false
                var markedByPerson = false
                for step in events {
                    let wasShowing = presentation.showing != nil
                    switch step {
                    case .launch:
                        presentation.launched(isSeen: local.isSeen, launchArguments: [], isCaptureBusy: busy)
                    case .settings:
                        presentation.requestedFromSettings(isCaptureBusy: busy)
                    case .sheetClosed:
                        presentation.sheetDismissed(isCaptureBusy: busy)
                    case .captureStarts:
                        busy = true
                        presentation.captureChanged(isBusy: true)
                    case .captureEnds:
                        busy = false
                        presentation.captureChanged(isBusy: false)
                    case .finish:
                        presentation.ended(.finished, store: local)
                        if wasShowing { markedByPerson = true }
                    case .skip:
                        presentation.ended(.skipped, store: local)
                        if wasShowing { markedByPerson = true }
                    }
                    if busy {
                        XCTAssertNil(presentation.showing, "shown during a capture after \(events)")
                    }
                    XCTAssertEqual(local.isSeen, markedByPerson, "seen flag after \(events)")
                }
                explore(events, depth: depth - 1)
            }
        }
        explore([], depth: 5)
    }
}

// MARK: - The words

final class OnboardingTextTests: XCTestCase {

    private var cards: [OnboardingCard] { OnboardingText.cards }

    private var everySentence: [String] {
        cards.flatMap { card in
            [card.title, card.lead] + card.points.flatMap { [$0.title, $0.text] }
        } + [
            OnboardingText.continueButton, OnboardingText.openConnections, OnboardingText.openConnectionsHint,
            OnboardingText.openSettings, OnboardingText.openSettingsHint, OnboardingText.closeHint,
            OnboardingText.settingsRow, OnboardingText.settingsRowUnavailable,
            OnboardingText.finishButton(.firstRun), OnboardingText.finishButton(.revisit),
            OnboardingText.closeButton(.firstRun), OnboardingText.closeButton(.revisit),
        ]
    }

    func testFourCardsInOrder() {
        XCTAssertEqual(cards.map(\.id), OnboardingCard.ID.allCases)
        XCTAssertEqual(Set(cards.map(\.id.slug)).count, cards.count)
        for card in cards {
            XCTAssertFalse(card.points.isEmpty, "\(card.id)")
            XCTAssertLessThanOrEqual(card.points.count, 4, "\(card.id): short cards")
        }
    }

    private func text(of id: OnboardingCard.ID) -> String {
        let card = cards.first { $0.id == id }!
        return ([card.title, card.lead] + card.points.flatMap { [$0.title, $0.text] })
            .joined(separator: " ").lowercased()
    }

    func testTheFirstCardSaysWhatTheAppDoes() {
        let first = text(of: .whatItDoes)
        for phrase in ["walk", "glasses", "tower", "3d world", "phone"] {
            XCTAssertTrue(first.contains(phrase), phrase)
        }
    }

    func testTheSecondCardNamesBothThingsNeededAndWhereToSetThemUp() {
        let second = text(of: .whatYouNeed)
        for phrase in ["meta ai", "pair", "tower", "computer", "settings", "connections", "register"] {
            XCTAssertTrue(second.contains(phrase), phrase)
        }
        // Non-breaking, so the largest text sizes never end a line at "Wi-".
        XCTAssertTrue(second.contains("wi\u{2011}fi"))
        XCTAssertFalse(second.contains("wi-fi"))
    }

    /// The four habits the walks' evidence supports (UX-PLAN principles 1–2).
    func testTheThirdCardTeachesTheFourHabits() {
        let card = cards.first { $0.id == .howToWalk }!
        let titles = card.points.map { $0.title.lowercased() }
        XCTAssertEqual(titles.count, 4)
        XCTAssertTrue(titles[0].contains("phone") && titles[0].contains("pocket"), titles[0])
        XCTAssertTrue(titles[1].contains("turn") && titles[1].contains("slowly"), titles[1])
        XCTAssertTrue(titles[2].contains("doorway") && titles[2].contains("slowly"), titles[2])
        XCTAssertTrue(card.points[2].text.lowercased().contains("look through"), card.points[2].text)
        XCTAssertTrue(titles[3].contains("small rooms") && titles[3].contains("side to side"), titles[3])
        XCTAssertTrue(card.points[0].text.lowercased().contains("confuses"), "says why")
    }

    func testTheFourthCardExplainsRoomsAndAreas() {
        let fourth = text(of: .whatYouSee)
        for phrase in ["room", "area", "joined", "saved worlds"] {
            XCTAssertTrue(fourth.contains(phrase), phrase)
        }
    }

    /// Plain words for a first-time reader: none of the pipeline's vocabulary.
    func testNoJargon() {
        let forbidden = ["keyframe", "keyframes", "producer", "cartridge", "cartridges", "slam", "sfm",
                         "pose", "poses", "point cloud", "mesh", "sparse", "dense", "reconstruction",
                         "session", "sessions", "dat", "websocket", "contract", "fps", "imu", "gyro"]
        for sentence in everySentence {
            let words = Set(sentence.lowercased()
                .components(separatedBy: CharacterSet.letters.inverted)
                .filter { !$0.isEmpty })
            for term in forbidden {
                let hit = term.contains(" ") ? sentence.lowercased().contains(term) : words.contains(term)
                XCTAssertFalse(hit, "“\(sentence)” says \(term)")
            }
        }
    }

    /// A misspelt SF Symbol draws nothing and says nothing, so check each one.
    func testEverySymbolExists() {
        let symbols = cards.flatMap { [$0.symbol] + $0.points.map(\.symbol) }
        for symbol in symbols {
            XCTAssertNotNil(UIImage(systemName: symbol), symbol)
        }
    }

    func testTheClosingWordsFollowTheMode() {
        XCTAssertEqual(OnboardingText.closeButton(.firstRun), "Skip")
        XCTAssertEqual(OnboardingText.finishButton(.firstRun), "Get started")
        XCTAssertEqual(OnboardingText.closeButton(.revisit), "Close")
        XCTAssertEqual(OnboardingText.finishButton(.revisit), "Done")
        XCTAssertEqual(OnboardingText.pageValue(1, of: 4), "2 of 4")
    }

    /// The view decides nothing and reaches nothing: no capture, no Tower, no
    /// flag. Those are `OnboardingPresentation` and the shell's.
    func testTheCardsViewTouchesNoRuntime() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Glasses/Views/OnboardingView.swift")
        guard FileManager.default.fileExists(atPath: url.path) else {
            #if targetEnvironment(simulator)
            return XCTFail("sources not readable at \(url.path)")
            #else
            throw XCTSkip("sources are not on a device")
            #endif
        }
        let code = try String(contentsOf: url, encoding: .utf8)
            .components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
            .joined(separator: "\n")
        for forbidden in ["GlassesConnection", "TowerClient", "ProjectManager", "startCameraSession",
                          "UserDefaults", "OnboardingStore", "markSeen"] {
            XCTAssertFalse(code.contains(forbidden), "OnboardingView.swift contains \(forbidden)")
        }
    }
}
