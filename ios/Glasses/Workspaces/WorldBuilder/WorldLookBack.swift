//
//  WorldLookBack.swift
//  Glasses
//

import AVFoundation
import Foundation

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

// MARK: - Speaking each prompt exactly once (§6.5)

/// The phone's rule, pure: whether to speak `recovery.prompt` now.
///
/// All four of §6.5's conditions, and C1 E1's: only while following the live
/// session (unpinned) and bound to it. A pinned subscription is never bound
/// (`TowerWorldBuilderClient.isCaptureBracketOpen`) and receives another
/// world's payload, so opening Saved Worlds mid-walk silences prompts until
/// *Back to live*; a Release build is never bound at all (no capture path,
/// C1 M14), so it never speaks.
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
        // 3. Never spoken before in this session; `lastSpoken` only rises.
        if let lastSpoken, prompt.id <= lastSpoken { return nil }
        // 4. Not stale, on the Tower's own clock. No `tower_sent_at` is no
        //    way to know, and a late "look back" is wrong advice.
        guard let towerSentAt, towerSentAt <= prompt.speakUntil else { return nil }
        return prompt
    }
}

/// Something that can say the prompt aloud. A protocol so the rule and the
/// ledger are tested without audio.
@MainActor
protocol WorldLookBackVoice: AnyObject {
    /// Get ready while the walk is live, so the first prompt does not pay for
    /// loading a voice. Idempotent.
    func prepare()
    /// Say `text`, for prompt `promptID`. Returns whether speech was started;
    /// a prompt not spoken (no Bluetooth route, the session refused) is logged
    /// by the voice and never retried.
    @discardableResult
    func speak(_ text: String, promptID: Int) -> Bool
}

/// The ledger and the rule, wired to a voice. Held by
/// `TowerWorldBuilderClient`, which hands it every report.
@MainActor
final class WorldLookBackPrompter {
    /// The words. The phone owns them (§8, C1 E14): short, because every extra
    /// word is more Bluetooth audio during the look-back.
    static let sentence = "Look back the way you came."

    /// Defence in depth only (§6.4): the Tower's limiter is the design, and
    /// the phone never delays a prompt. It refuses a third inside 60 s of its
    /// own clock.
    static let localCap = 2
    static let localWindow: TimeInterval = 60

    private let voice: WorldLookBackVoice
    /// `lastSpoken[(world, session)]`: in memory only (C1 M4). A relaunch
    /// closes the capture bracket, so the binding already refuses a repeat,
    /// and rule 4 bounds what is left to `speak_window_s`.
    private(set) var lastSpoken: [String: Int] = [:]
    private var spokenAt: [Date] = []
    var clock: () -> Date = { Date() }

    init(voice: WorldLookBackVoice) {
        self.voice = voice
    }

    /// Consider one report. Speaks at most once per prompt id, per session.
    func consider(
        recovery: WorldRecoveryReport?,
        binding: WorldSessionBinding,
        followingLive: Bool,
        towerSentAt: Double?,
        worldID: String?,
        sessionID: String?
    ) {
        guard followingLive, case .bound = binding, let worldID, let sessionID else { return }
        voice.prepare()
        let key = "\(worldID)/\(sessionID)"
        guard let prompt = WorldLookBackRule.promptToSpeak(
            recovery: recovery, binding: binding, followingLive: followingLive,
            towerSentAt: towerSentAt, lastSpoken: lastSpoken[key])
        else { return }
        // Recorded BEFORE speech starts (§6.5): a heartbeat carrying the same
        // id, or a second report arriving while the voice is still talking,
        // never speaks it again.
        lastSpoken[key] = prompt.id
        let now = clock()
        spokenAt = spokenAt.filter { now.timeIntervalSince($0) < Self.localWindow }
        guard spokenAt.count < Self.localCap else {
            WorldLookBackLog.write("prompt \(prompt.id) not spoken: a third inside 60 s on this phone's clock")
            return
        }
        spokenAt.append(now)
        voice.speak(Self.sentence, promptID: prompt.id)
    }
}

// MARK: - The voice: AVSpeechSynthesizer over A2DP (C1 E6)

/// Speaks through the glasses over **A2DP only, never HFP**
/// (`WORLD-BUILDER-COMPONENTS.md` §6.5 "How the phone speaks"):
///
/// - category `.playback`, mode `.voicePrompt`, option `.duckOthers`,
///   activated around each utterance and deactivated with
///   `.notifyOthersOnDeactivation` when it ends. HFP would switch the glasses
///   off A2DP and, the DAT docs say, must be configured before the camera
///   stream starts -- so it is never touched;
/// - only when the current output route is Bluetooth (A2DP or LE): a phone
///   speaker in a pocket is useless to the wearer and audible to others. A
///   prompt not spoken for want of a route is logged, never retried;
/// - the app declares the `audio` background mode, so speech can start with
///   the phone locked in a pocket.
///
/// Three things only a device settles, and the physical test measures them:
/// audible on the glasses with the phone locked; the camera's frame rate and
/// resolution while speaking (A2DP shares the Bluetooth Classic link with the
/// camera stream); and the latency from `issued_at` to audible. The DEBUG log
/// lines below give the phone's half of the last one.
@MainActor
final class WorldSpeechLookBackVoice: NSObject, WorldLookBackVoice, AVSpeechSynthesizerDelegate {
    /// Created on first `prepare()`, and held strongly: a synthesizer that is
    /// released mid-utterance stops talking.
    private var synthesizer: AVSpeechSynthesizer?
    private var voice: AVSpeechSynthesisVoice?
    private var startedSpeakingAt: [ObjectIdentifier: (promptID: Int, asked: Date)] = [:]

    func prepare() {
        guard synthesizer == nil else { return }
        let synthesizer = AVSpeechSynthesizer()
        synthesizer.delegate = self
        // The app's own session, so the category below governs it.
        synthesizer.usesApplicationAudioSession = true
        self.synthesizer = synthesizer
        voice = AVSpeechSynthesisVoice(language: AVSpeechSynthesisVoice.currentLanguageCode())
    }

    /// Whether `route` has a Bluetooth output the glasses can play on.
    nonisolated static func isBluetoothOutput(_ route: AVAudioSessionRouteDescription) -> Bool {
        route.outputs.contains { $0.portType == .bluetoothA2DP || $0.portType == .bluetoothLE }
    }

    @discardableResult
    func speak(_ text: String, promptID: Int) -> Bool {
        prepare()
        guard let synthesizer else { return false }
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, mode: .voicePrompt, options: [.duckOthers])
        } catch {
            WorldLookBackLog.write("prompt \(promptID) not spoken: the audio category was refused (\(error.localizedDescription))")
            return false
        }
        guard Self.isBluetoothOutput(session.currentRoute) else {
            let ports = session.currentRoute.outputs.map(\.portType.rawValue).joined(separator: ",")
            WorldLookBackLog.write("prompt \(promptID) not spoken: no Bluetooth output route (\(ports))")
            return false
        }
        do {
            try session.setActive(true)
        } catch {
            WorldLookBackLog.write("prompt \(promptID) not spoken: the audio session did not activate (\(error.localizedDescription))")
            return false
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        startedSpeakingAt[ObjectIdentifier(utterance)] = (promptID, Date())
        synthesizer.speak(utterance)
        let ports = session.currentRoute.outputs.map(\.portType.rawValue).joined(separator: ",")
        WorldLookBackLog.write("prompt \(promptID) asked to speak on \(ports)")
        return true
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in
            guard let entry = self.startedSpeakingAt[id] else { return }
            let ms = Int(Date().timeIntervalSince(entry.asked) * 1000)
            WorldLookBackLog.write("prompt \(entry.promptID) speech started \(ms) ms after it was asked")
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in self.finish(id, how: "finished") }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in self.finish(id, how: "cancelled") }
    }

    private func finish(_ id: ObjectIdentifier, how: String) {
        if let entry = startedSpeakingAt.removeValue(forKey: id) {
            WorldLookBackLog.write("prompt \(entry.promptID) speech \(how)")
        }
        guard startedSpeakingAt.isEmpty else { return }
        // Give the route back so ducked audio returns.
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}

/// One line per prompt decision, DEBUG only: the phone's half of the latency
/// the physical test measures.
nonisolated enum WorldLookBackLog {
    static func write(_ line: String) {
        #if DEBUG
        print("[Glasses][WorldBuilder][LookBack] \(line)")
        #endif
    }
}
