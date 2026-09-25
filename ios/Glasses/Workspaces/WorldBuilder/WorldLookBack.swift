//
//  WorldLookBack.swift
//  Glasses
//

import Combine
import SwiftUI
import UIKit

// MARK: - What the live relocalizer reports

/// `tracking.recovery.state` (`WORLD-BUILDER-COMPONENTS.md` §6.2). A
/// `RawRepresentable` struct, for the reason `WorldSelectionMode` gives.
nonisolated struct WorldRecoveryState: RawRepresentable, Equatable, Sendable, Hashable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    /// No episode yet this session.
    static let none = WorldRecoveryState(rawValue: "none")
    /// Tracking was lost; the builder is matching frames against what was
    /// seen before it, silently so far.
    static let searching = WorldRecoveryState(rawValue: "searching")
    /// Still searching, and the builder has asked the wearer to look back.
    static let prompting = WorldRecoveryState(rawValue: "prompting")
    /// A verified revisit edge was written. NOT "placed" (§6.3).
    static let recovered = WorldRecoveryState(rawValue: "recovered")
    /// The episode ended without one.
    static let timedOut = WorldRecoveryState(rawValue: "timed_out")
}

/// `tracking.recovery.prompt`: the latest look-back prompt the builder issued
/// this session (§6.2).
nonisolated struct WorldLookBackPrompt: Equatable, Sendable {
    /// 1-based, strictly increasing within the session, never reused: the
    /// phone's speak-once key (§6.5).
    let id: Int
    let episode: Int
    /// `"look-back"`, the only kind. An unknown kind is never spoken.
    let kind: String
    /// Tower clock.
    let issuedAt: Double
    /// Tower clock: after it the prompt is stale and is never spoken.
    let speakUntil: Double

    static let lookBackKind = "look-back"

    init(id: Int, episode: Int, kind: String = Self.lookBackKind, issuedAt: Double, speakUntil: Double) {
        self.id = id
        self.episode = episode
        self.kind = kind
        self.issuedAt = issuedAt
        self.speakUntil = speakUntil
    }

    init?(json: Any?) {
        guard
            let json = json as? [String: Any],
            let id = json["id"] as? Int,
            let episode = json["episode"] as? Int,
            let kind = json["kind"] as? String,
            let issuedAt = (json["issued_at"] as? NSNumber)?.doubleValue,
            let speakUntil = (json["speak_until"] as? NSNumber)?.doubleValue
        else { return nil }
        self.init(id: id, episode: episode, kind: kind, issuedAt: issuedAt, speakUntil: speakUntil)
    }
}

/// `tracking.recovery` (§6.2): the relocalizer's current or last episode, and
/// the latest prompt. Only what the phone uses is decoded; `counts`, `limiter`
/// and `acceptance` are the Tower's audit and stay on the wire.
///
/// **Read from the payload's `tracking` block, in its own type, never folded
/// into `world_snapshot`** (C1 M5, `WORLD-BUILDER-IOS.md` §2.4): the snapshot
/// is the world's identity and redraws the canvas when it changes, and a
/// prompt is neither.
nonisolated struct WorldRecoveryReport: Equatable, Sendable {
    let state: WorldRecoveryState
    let episode: Int
    let promptsEnabled: Bool
    let prompt: WorldLookBackPrompt?

    init(state: WorldRecoveryState, episode: Int, promptsEnabled: Bool = true, prompt: WorldLookBackPrompt? = nil) {
        self.state = state
        self.episode = episode
        self.promptsEnabled = promptsEnabled
        self.prompt = prompt
    }

    /// `nil` for `null` -- "not recorded", never "no losses" (§6.2): every
    /// session today, any Tower without the relocalizer, every older world --
    /// and for a block with no `state`.
    init?(json: Any?) {
        guard
            let json = json as? [String: Any],
            let state = json["state"] as? String, !state.isEmpty
        else { return nil }
        self.state = WorldRecoveryState(rawValue: state)
        self.episode = json["episode"] as? Int ?? 0
        // Absent is not "enabled": a block that does not say whether prompts
        // are on is not a block the phone speaks for.
        self.promptsEnabled = json["prompts_enabled"] as? Bool ?? false
        self.prompt = WorldLookBackPrompt(json: json["prompt"])
    }

    /// The world screen's line (§8, accepted by C1), or `nil` for no episode
    /// yet and for a word this build does not know.
    var displayLine: String? {
        switch state {
        case .searching, .prompting: return "Finding where you are…"
        case .recovered: return "Linked back to what you saw before"
        case .timedOut: return "Could not link back; this part may be shown as a separate area."
        default: return nil
        }
    }
}


// MARK: - Showing each prompt exactly once (§6.5)

/// The phone's rule, pure: whether `recovery.prompt` is a new prompt to show
/// now.
///
/// All four of §6.5's conditions, and C1 E1's: only while following the live
/// session (unpinned) and bound to it. A pinned subscription is never bound
/// (`TowerWorldBuilderClient.isCaptureBracketOpen`) and receives another
/// world's payload, so opening Saved Worlds mid-walk silences prompts until
/// *Back to live*; a Release build is never bound at all (no capture path,
/// C1 M14), so it never shows one.
enum WorldLookBackRule {
    static func promptToSpeak(
        recovery: WorldRecoveryReport?,
        binding: WorldSessionBinding,
        followingLive: Bool,
        towerSentAt: Double?,
        lastSpoken: Int?
    ) -> WorldLookBackPrompt? {
        // 1. Live, and the session this phone is streaming to.
        guard followingLive, case .bound = binding else { return nil }
        // 2. Prompting, in this episode, a kind the phone knows.
        guard
            let recovery, recovery.promptsEnabled, recovery.state == .prompting,
            let prompt = recovery.prompt,
            prompt.episode == recovery.episode,
            prompt.kind == WorldLookBackPrompt.lookBackKind
        else { return nil }
        // 3. Never shown before in this session; `lastSpoken` only rises.
        if let lastSpoken, prompt.id <= lastSpoken { return nil }
        // 4. Not stale, on the Tower's own clock. No `tower_sent_at` is no
        //    way to know, and a late "look back" is wrong advice.
        guard let towerSentAt, towerSentAt <= prompt.speakUntil else { return nil }
        return prompt
    }
}

/// What the World Builder live screen shows for the look-back prompt.
///
/// **Shown, never spoken** (walk 1 at `4b4b444`, 2026-09-24): speech to the
/// glasses over A2DP ended the DAT camera session ~3 s later -- *"Session
/// ended by device"* -- and the walk's stream with it. Nothing in this path
/// may touch the audio session, speech, or the Bluetooth audio route;
/// `WorldLookBackAudioFreeTests` fails the build's tests if it does.
nonisolated enum WorldLookBackBanner: Equatable, Sendable {
    /// Prompt `promptID` is live: the wearer should look back.
    case lookBack(promptID: Int)
    /// The episode that prompted was recovered; shown briefly, then gone.
    case backOnTrack

    var text: String {
        switch self {
        case .lookBack: return WorldLookBackPrompter.sentence
        case .backOnTrack: return WorldLookBackPrompter.recoveredSentence
        }
    }
}

/// The one non-visual cue that goes with a new banner, and the one fact about
/// the app the ledger needs: whether anyone can see or feel a prompt right
/// now. A protocol so the ledger is tested without a device. **No audio of
/// any kind**: a haptic uses no audio session and never reaches the glasses.
@MainActor
protocol WorldLookBackCue: AnyObject {
    /// Whether the app is foreground-active. A prompt is presented -- and
    /// only then counted as shown -- while this is true; one that arrives
    /// otherwise is held for its window (`WorldLookBackPrompter`).
    var isForegroundActive: Bool { get }
    /// A new look-back banner, for prompt `promptID`, has just appeared.
    func alert(promptID: Int)
}

/// One `UINotificationFeedbackGenerator` `.warning`. Called only while the
/// app is foreground-active, the only time a feedback generator does
/// anything.
@MainActor
final class WorldHapticLookBackCue: WorldLookBackCue {
    var isForegroundActive: Bool {
        UIApplication.shared.applicationState == .active
    }

    func alert(promptID: Int) {
        UINotificationFeedbackGenerator().notificationOccurred(.warning)
        WorldLookBackLog.write("prompt \(promptID) shown with a haptic")
    }
}

/// The ledger and the rule, driving the banner. Held by
/// `TowerWorldBuilderClient`, which hands it every report.
///
/// ## A prompt the wearer cannot see yet
///
/// A prompt that arrives while the app is not foreground-active (the phone
/// locked in a pocket) is **held, not consumed**: nothing is shown, no haptic
/// fires, and `lastSpoken` is not advanced. It is presented when the app
/// becomes active -- or on the next report that repeats it while active --
/// provided its window has not ended; otherwise it is dropped. A held prompt
/// is also dropped when the episode stops prompting or the phone stops
/// following the live walk. No notification, sound or other audio is ever
/// used to reach a locked phone.
///
/// ## How long the banner stays
///
/// What is left of the window on the Tower's clock at delivery,
/// `speak_until - tower_sent_at` (§6.5 rule 4 compares the same two), counted
/// from when this phone received the report, clamped at 0 -- never the full
/// `speak_window_s` again for a late delivery. It comes down sooner if the
/// episode stops prompting.
@MainActor
final class WorldLookBackPrompter {
    /// The banner's words. The phone owns them (§8, C1 E14).
    nonisolated static let sentence = "Tracking lost — slowly look back the way you came."
    /// Shown briefly when the prompting episode is recovered.
    nonisolated static let recoveredSentence = "Back on track"
    static let backOnTrackSeconds: TimeInterval = 3

    /// Defence in depth only (§6.4): the Tower's limiter is the design, and
    /// the phone never delays a prompt. It refuses a third inside 60 s of its
    /// own clock.
    static let localCap = 2
    static let localWindow: TimeInterval = 60

    private let cue: WorldLookBackCue
    /// `lastSpoken[(world, session)]`: the last prompt id **presented** in
    /// the foreground. In memory only (C1 M4). A relaunch closes the capture
    /// bracket, so the binding already refuses a repeat, and rule 4 bounds
    /// what is left to `speak_window_s`.
    private(set) var lastSpoken: [String: Int] = [:]
    private var shownAt: [Date] = []
    var clock: () -> Date = { Date() }

    /// A prompt that passed the rule while the app was not foreground-active.
    struct Held: Equatable {
        let key: String
        let promptID: Int
        /// On `clock`: when its window ends.
        let deadline: Date
    }
    private(set) var held: Held?

    /// What the live screen shows now, or `nil`. `onChange` hears every
    /// change.
    private(set) var banner: WorldLookBackBanner? {
        didSet {
            guard banner != oldValue else { return }
            onChange?(banner)
        }
    }
    var onChange: ((WorldLookBackBanner?) -> Void)?
    private var bannerDeadline: Date?
    private var expiry: Task<Void, Never>?
    private var becameActive: AnyCancellable?

    init(cue: WorldLookBackCue, notificationCenter: NotificationCenter = .default) {
        self.cue = cue
        becameActive = notificationCenter
            .publisher(for: UIApplication.didBecomeActiveNotification)
            .sink { [weak self] _ in
                Task { @MainActor [weak self] in self?.appDidBecomeActive() }
            }
    }

    /// Consider one report. Presents each prompt id at most once per session.
    func consider(
        recovery: WorldRecoveryReport?,
        binding: WorldSessionBinding,
        followingLive: Bool,
        towerSentAt: Double?,
        worldID: String?,
        sessionID: String?
    ) {
        guard followingLive, case .bound = binding, let worldID, let sessionID else {
            dismiss()
            return
        }
        // The banner follows the episode: gone once it stops prompting, and
        // "Back on track" for a moment if it was recovered. A held prompt of
        // an episode that stopped prompting is moot.
        if recovery?.state != .prompting {
            held = nil
            if case .lookBack = banner {
                if recovery?.state == .recovered {
                    show(.backOnTrack, for: Self.backOnTrackSeconds)
                } else {
                    dismiss()
                }
            }
        }
        let key = "\(worldID)/\(sessionID)"
        guard
            let prompt = WorldLookBackRule.promptToSpeak(
                recovery: recovery, binding: binding, followingLive: followingLive,
                towerSentAt: towerSentAt, lastSpoken: lastSpoken[key]),
            let towerSentAt
        else { return }
        let deadline = clock().addingTimeInterval(max(0, prompt.speakUntil - towerSentAt))
        guard cue.isForegroundActive else {
            if held?.promptID != prompt.id {
                WorldLookBackLog.write("prompt \(prompt.id) held: the app is not foreground-active")
            }
            // A repeat of the same id keeps the deadline of its first
            // delivery; the window only ever shrinks.
            if held?.key != key || held?.promptID != prompt.id {
                held = Held(key: key, promptID: prompt.id, deadline: deadline)
            }
            return
        }
        present(promptID: prompt.id, key: key, until: deadline)
    }

    /// The app became foreground-active: present the held prompt if its
    /// window is still open. Wired to `didBecomeActiveNotification`; public
    /// so a test drives it.
    func appDidBecomeActive() {
        guard let held else { return }
        self.held = nil
        guard clock() < held.deadline, (lastSpoken[held.key] ?? 0) < held.promptID else {
            WorldLookBackLog.write("prompt \(held.promptID) not shown: its window ended before the app became active")
            return
        }
        present(promptID: held.promptID, key: held.key, until: held.deadline)
    }

    /// Take the banner down now, and drop any held prompt: pinned, unbound,
    /// or the walk ended.
    func dismiss() {
        held = nil
        expiry?.cancel()
        expiry = nil
        bannerDeadline = nil
        banner = nil
    }

    /// Take the banner down if its time is up on `clock`. Called by the
    /// banner's own timer; public so a test drives it with a fake clock.
    func expireIfDue() {
        guard let bannerDeadline, clock() >= bannerDeadline else { return }
        dismiss()
    }

    /// Show prompt `promptID` until `deadline`, with the haptic. Only ever
    /// called while the app is foreground-active.
    private func present(promptID: Int, key: String, until deadline: Date) {
        held = nil
        // Recorded BEFORE it is shown (§6.5): a heartbeat carrying the same
        // id never shows it again.
        lastSpoken[key] = promptID
        let now = clock()
        shownAt = shownAt.filter { now.timeIntervalSince($0) < Self.localWindow }
        guard shownAt.count < Self.localCap else {
            WorldLookBackLog.write("prompt \(promptID) not shown: a third inside 60 s on this phone's clock")
            return
        }
        shownAt.append(now)
        show(.lookBack(promptID: promptID), for: max(0, deadline.timeIntervalSince(now)))
        cue.alert(promptID: promptID)
    }

    private func show(_ next: WorldLookBackBanner, for seconds: TimeInterval) {
        expiry?.cancel()
        banner = next
        bannerDeadline = clock().addingTimeInterval(seconds)
        expiry = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds + 0.05))
            guard !Task.isCancelled else { return }
            self?.expireIfDue()
        }
    }
}

/// The banner itself: at the top of the World Builder screen, large and
/// coloured, because the wearer glances at it mid-walk.
struct WorldLookBackBannerView: View {
    let banner: WorldLookBackBanner

    var body: some View {
        let isLookBack: Bool = { if case .lookBack = banner { return true } else { return false } }()
        Label(banner.text, systemImage: isLookBack ? "arrow.uturn.backward.circle.fill" : "checkmark.circle.fill")
            .font(.title3.weight(.semibold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(isLookBack ? Color.orange : Color.green, in: .rect(cornerRadius: 14))
            .accessibilityIdentifier(isLookBack ? "world-lookback-banner" : "world-back-on-track-banner")
            .transition(.move(edge: .top).combined(with: .opacity))
    }
}

/// One line per prompt decision, DEBUG only.
nonisolated enum WorldLookBackLog {
    static func write(_ line: String) {
        #if DEBUG
        print("[Glasses][WorldBuilder][LookBack] \(line)")
        #endif
    }
}
