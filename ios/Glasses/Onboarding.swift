//
//  Onboarding.swift
//  Glasses
//
//  "How Glasses works": the first-run cards, and when they may be shown.
//  The view is `Views/OnboardingView.swift`; this file holds everything a
//  test can check without rendering it.
//

import Foundation

// MARK: - The flag

/// Whether this phone has seen the cards, in `UserDefaults`.
///
/// Written only when the cards end: finishing them and skipping them both mark
/// them seen, so nobody is shown them twice unasked. Reopening them from
/// Settings does not depend on it.
nonisolated struct OnboardingStore {
    static let key = "onboardingSeen"

    /// A UI test launches with this so the cards never cover the screen it is
    /// testing. It only stops them opening at launch; the flag is left as it is.
    static let uiTestSkipArgument = "-UITestSkipOnboarding"

    /// DEBUG only: a UI test launches with this to start from a first launch.
    /// Honoured in `GlassesApp.init`, before the root view appears.
    static let uiTestResetArgument = "-UITestResetOnboarding"

    let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    var isSeen: Bool {
        defaults.bool(forKey: Self.key)
    }

    func markSeen() {
        defaults.set(true, forKey: Self.key)
    }

    func reset() {
        defaults.removeObject(forKey: Self.key)
    }
}

// MARK: - The rules

/// When the cards may be on screen.
///
/// The one rule that outranks everything: **never over a capture.** The cards
/// cover the whole screen, and a capture that is running, starting, paused by
/// the glasses or stopping has controls and a viewfinder a person must be able
/// to see.
nonisolated enum OnboardingGate {
    /// Whether a capture is running or on its way, from the two places one can
    /// be: the glasses' camera claim, and Object Memory's Start, which asks the
    /// Tower before it asks the camera and so is "starting" before the claim
    /// moves.
    ///
    /// Exhaustive over `CaptureClaim` on purpose, for the reason given at
    /// `GlassesConnection.isCaptureSessionClaimed`: a `default` is how
    /// `.devicePaused` gets treated as idle.
    static func isCaptureBusy(claim: CaptureClaim, recordingPhase: ObjectMemoryRecordingPhase) -> Bool {
        switch claim {
        case .running, .devicePaused, .ending:
            return true
        case .unclaimed:
            break
        }
        if case .starting = recordingPhase { return true }
        return false
    }

    /// The first-run cards, when the app opens.
    static func presentsAtLaunch(isSeen: Bool, launchArguments: [String], isCaptureBusy: Bool) -> Bool {
        !isSeen
            && !launchArguments.contains(OnboardingStore.uiTestSkipArgument)
            && !isCaptureBusy
    }

    /// "How Glasses works" in Settings.
    static func canReopen(isCaptureBusy: Bool) -> Bool {
        !isCaptureBusy
    }
}

/// Which set of cards: the first run, or reopened from Settings. They differ
/// only in the words that close them.
nonisolated enum OnboardingMode: String, Identifiable, Sendable {
    case firstRun
    case revisit

    var id: String { rawValue }
}

/// How the cards were closed by the person reading them.
nonisolated enum OnboardingEnd: Equatable, Sendable {
    /// "Get started" (or "Done") on the last card.
    case finished
    /// "Skip" (or "Close") on any card.
    case skipped
}

/// Which cards are on screen, and why. A value, so that every way in and out
/// is tested without a simulator (`OnboardingTests`).
///
/// Three ways in, each gated:
/// - **launch**, when they have not been seen;
/// - **Settings**, which closes first; the cards follow once it has gone,
///   because a full-screen cover cannot open over a sheet that is closing;
/// - nothing else.
///
/// Two ways out: the person ends them (marked seen, either way), or a capture
/// starts (closed, **not** marked seen, so a first run that was interrupted
/// comes back at the next launch rather than in the middle of a walk).
nonisolated struct OnboardingPresentation: Equatable, Sendable {
    /// What is on screen, or `nil`.
    private(set) var showing: OnboardingMode?
    /// Asked for from Settings, waiting for Settings to close.
    private(set) var pending: OnboardingMode?

    init() {}

    mutating func launched(isSeen: Bool, launchArguments: [String], isCaptureBusy: Bool) {
        guard showing == nil,
              OnboardingGate.presentsAtLaunch(isSeen: isSeen, launchArguments: launchArguments,
                                              isCaptureBusy: isCaptureBusy)
        else { return }
        showing = .firstRun
    }

    mutating func requestedFromSettings(isCaptureBusy: Bool) {
        guard showing == nil, OnboardingGate.canReopen(isCaptureBusy: isCaptureBusy) else { return }
        pending = .revisit
    }

    /// Settings has closed. The gate is asked again: a capture could have
    /// started in between.
    mutating func sheetDismissed(isCaptureBusy: Bool) {
        defer { pending = nil }
        guard let pending, OnboardingGate.canReopen(isCaptureBusy: isCaptureBusy) else { return }
        showing = pending
    }

    mutating func captureChanged(isBusy: Bool) {
        guard isBusy else { return }
        showing = nil
        pending = nil
    }

    /// The person closed them. Marked seen however they ended.
    mutating func ended(_ end: OnboardingEnd, store: OnboardingStore) {
        guard showing != nil else { return }
        showing = nil
        store.markSeen()
    }

    /// SwiftUI took the cover down by itself. Not an ending, so not marked.
    mutating func coverDismissed() {
        showing = nil
    }
}

// MARK: - Words

/// One point on a card: a symbol, a short title and a sentence or two.
nonisolated struct OnboardingPoint: Equatable, Sendable {
    let symbol: String
    let title: String
    let text: String
}

/// One card.
nonisolated struct OnboardingCard: Identifiable, Equatable, Sendable {
    nonisolated enum ID: Int, CaseIterable, Sendable {
        case whatItDoes
        case whatYouNeed
        case howToWalk
        case whatYouSee

        /// For UI tests and screenshots.
        var slug: String {
            switch self {
            case .whatItDoes: return "what-it-does"
            case .whatYouNeed: return "what-you-need"
            case .howToWalk: return "how-to-walk"
            case .whatYouSee: return "what-you-see"
            }
        }
    }

    let id: ID
    let symbol: String
    let title: String
    let lead: String
    let points: [OnboardingPoint]
}

/// Every sentence the cards show, in one place, so a test can pin the plain
/// language without rendering a view.
///
/// Written for someone who has never seen the app. The words that are the
/// app's own — the Tower, Connections, Settings, Register, World Builder,
/// Saved worlds, areas — are the ones already on its screens.
nonisolated enum OnboardingText {
    static let cards: [OnboardingCard] = [
        OnboardingCard(
            id: .whatItDoes,
            symbol: "eyeglasses",
            title: "Welcome to Glasses",
            lead: "Walk through your home wearing the camera glasses. Your computer, called the Tower, builds a 3D world from what they saw, and you can go back to it on your phone.",
            points: [
                OnboardingPoint(
                    symbol: "figure.walk",
                    title: "You walk",
                    text: "Wear the glasses and walk slowly from room to room."
                ),
                OnboardingPoint(
                    symbol: "desktopcomputer",
                    title: "The Tower builds",
                    text: "Your computer turns what the glasses saw into a 3D world."
                ),
                OnboardingPoint(
                    symbol: "iphone",
                    title: "You look around",
                    text: "Open the world on your phone whenever you like, and turn it to see every side."
                ),
            ]
        ),
        OnboardingCard(
            id: .whatYouNeed,
            symbol: "checklist",
            title: "What you need",
            lead: "Two things, each set up once.",
            points: [
                OnboardingPoint(
                    symbol: "eyeglasses",
                    title: "The glasses, paired in the Meta AI app",
                    text: "Pair them in Meta AI first. Then tap Register in Connections, so Glasses can use them."
                ),
                OnboardingPoint(
                    symbol: "desktopcomputer",
                    title: "The Tower, running on your computer",
                    // A non-breaking hyphen in Wi-Fi: at the largest text
                    // sizes a line would otherwise end at "Wi-".
                    text: "Start it on your computer, then enter its address in Settings. Your phone must be able to reach the computer, for example on the same Wi\u{2011}Fi."
                ),
            ]
        ),
        OnboardingCard(
            id: .howToWalk,
            symbol: "figure.walk",
            title: "How to walk",
            lead: "For a good world, move gently. Fast movement blurs what the glasses see, and blur breaks a world apart.",
            points: [
                OnboardingPoint(
                    symbol: "iphone.slash",
                    title: "Keep your phone in your pocket",
                    text: "A phone in view confuses the world. Put it away before you start walking."
                ),
                OnboardingPoint(
                    symbol: "tortoise",
                    title: "Turn your head slowly",
                    text: "Quick turns blur the picture. Turn slower than feels natural."
                ),
                OnboardingPoint(
                    symbol: "door.left.hand.open",
                    title: "Cross doorways slowly",
                    text: "Look through the doorway as you go, so the two rooms can be joined."
                ),
                OnboardingPoint(
                    symbol: "arrow.left.and.right",
                    title: "In small rooms, step side to side",
                    text: "Turning on the spot shows too little. Step a little sideways as you look around."
                ),
            ]
        ),
        OnboardingCard(
            id: .whatYouSee,
            symbol: "cube.transparent",
            title: "What you’ll see",
            lead: "After your walk, the Tower takes a little while to finish the world. Then you’ll find it in World Builder, under Saved worlds.",
            points: [
                OnboardingPoint(
                    symbol: "house",
                    title: "Rooms",
                    text: "The parts of your walk that join up become a room you can turn and look around."
                ),
                OnboardingPoint(
                    symbol: "square.dashed",
                    title: "Areas",
                    text: "A part that couldn’t be joined to the room yet is shown on its own, as Area 1, Area 2 and so on. Glasses never guesses where it goes."
                ),
            ]
        ),
    ]

    static let continueButton = "Continue"
    static let openConnections = "Open Connections"
    static let openConnectionsHint = "Shows whether the glasses and the Tower are connected, and lets you register the glasses."
    static let openSettings = "Open Settings"
    static let openSettingsHint = "Where you enter the Tower’s address."
    static let pageLabel = "Card"

    static func pageValue(_ index: Int, of count: Int) -> String {
        "\(index + 1) of \(count)"
    }

    static func finishButton(_ mode: OnboardingMode) -> String {
        switch mode {
        case .firstRun: return "Get started"
        case .revisit: return "Done"
        }
    }

    static func closeButton(_ mode: OnboardingMode) -> String {
        switch mode {
        case .firstRun: return "Skip"
        case .revisit: return "Close"
        }
    }

    static let closeHint = "Closes the introduction. You can open it again from Settings."

    /// Settings' row, and why it can be off.
    static let settingsRow = "How Glasses works"
    static let settingsRowUnavailable = "Available when no capture is running."
}
