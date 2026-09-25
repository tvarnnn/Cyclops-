//
//  WorldLookBackTests.swift
//  GlassesTests
//
//  `tracking.recovery` and the look-back prompt (`WORLD-BUILDER-COMPONENTS.md`
//  §6, contract v2 @ bb80b2c; C1 E1, E2, E7, E14, M4, M5, M14), as amended
//  after walk 1 at `4b4b444`: the prompt is SHOWN on the phone with a haptic
//  and never spoken, because speech to the glasses ended the DAT camera
//  session ("Session ended by device").
//

import UIKit
import XCTest

@testable import Glasses

/// Records the haptic, instead of firing it, and plays the app's state.
@MainActor
final class RecordingLookBackCue: WorldLookBackCue {
    private(set) var alerts: [Int] = []
    var isForegroundActive = true

    func alert(promptID: Int) { alerts.append(promptID) }
}

@MainActor
final class WorldRecoveryDecodingTests: XCTestCase {

    private let block: [String: Any] = [
        "state": "prompting", "episode": 2, "lost_at": 1790100030.5, "resolved_at": NSNull(),
        "recovered_by": NSNull(), "prompts_enabled": true,
        "prompt": ["id": 3, "episode": 2, "kind": "look-back", "issued_at": 1790100033.5, "speak_until": 1790100037.5],
        "counts": ["episodes": 2, "recovered": 1, "recovered_after_prompt": 0, "timed_out": 0, "prompts": 3,
                   "withheld_by_limiter": 0, "withheld_disabled": 0],
        "limiter": ["max_prompts": 2, "window_s": 60, "mechanism": "cooldown", "cooldown_s": 20,
                    "prompt_after_s": 3, "timeout_s": 15, "speak_window_s": 4],
        "acceptance": ["matcher": "sift", "reference_keyframes": 12, "scan_hz": 2],
    ]

    func testTheBlockDecodes() throws {
        let report = try XCTUnwrap(WorldRecoveryReport(json: block))
        XCTAssertEqual(report.state, .prompting)
        XCTAssertEqual(report.episode, 2)
        XCTAssertTrue(report.promptsEnabled)
        XCTAssertEqual(report.prompt, WorldLookBackPrompt(id: 3, episode: 2, issuedAt: 1790100033.5, speakUntil: 1790100037.5))
    }

    /// `null` is "not recorded", never "no losses" (§6.2).
    func testNullAndAbsentAreNotRecorded() {
        XCTAssertNil(WorldRecoveryReport(json: nil))
        XCTAssertNil(WorldRecoveryReport(json: NSNull()))
        XCTAssertNil(WorldRecoveryReport(json: ["episode": 1]))
        XCTAssertNil(WorldBuilderResultDecoder.recovery(from: ["tracking": ["state": "good"]]))
        XCTAssertNil(WorldBuilderResultDecoder.recovery(from: ["tracking": ["state": "good", "recovery": NSNull()]]))
        XCTAssertEqual(WorldBuilderResultDecoder.recovery(from: ["tracking": ["recovery": block]])?.state, .prompting)
    }

    /// A block that does not say prompts are on is not one the phone speaks for.
    func testAbsentPromptsEnabledIsNotEnabled() throws {
        var partial = block
        partial["prompts_enabled"] = nil
        XCTAssertFalse(try XCTUnwrap(WorldRecoveryReport(json: partial)).promptsEnabled)
    }

    /// §8's lines, and nothing for `none` or a word this build does not know.
    func testTheWorldScreenLine() {
        func line(_ state: String) -> String? { WorldRecoveryReport(state: WorldRecoveryState(rawValue: state), episode: 1).displayLine }
        XCTAssertEqual(line("searching"), "Finding where you are…")
        XCTAssertEqual(line("prompting"), "Finding where you are…")
        XCTAssertEqual(line("recovered"), "Linked back to what you saw before")
        XCTAssertEqual(line("timed_out"), "Could not link back; this part may be shown as a separate area.")
        XCTAssertNil(line("none"))
        XCTAssertNil(line("relocalizing-v2"))
        XCTAssertFalse(line("recovered")!.contains("placed"), "recovered is not placed (§6.3)")
    }

    /// The envelope's `tower_sent_at` is decoded now (C1 E2).
    func testTheEnvelopeCarriesTowerSentAt() throws {
        let json: [String: Any] = ["cartridge": "world_builder", "result_type": "status",
                                   "tower_sent_at": 1790100034.25, "payload": ["model_state": "idle"]]
        XCTAssertEqual(try XCTUnwrap(CartridgeResultEnvelope(json: json)).towerSentAt, 1790100034.25)
        var bare = json
        bare["tower_sent_at"] = nil
        XCTAssertNil(try XCTUnwrap(CartridgeResultEnvelope(json: bare)).towerSentAt)
    }
}

@MainActor
final class WorldLookBackRuleTests: XCTestCase {

    private let bound = WorldSessionBinding.bound(captureID: "cap-1")
    private let prompt = WorldLookBackPrompt(id: 3, episode: 2, issuedAt: 100, speakUntil: 104)

    private func recovery(
        state: WorldRecoveryState = .prompting, episode: Int = 2, enabled: Bool = true,
        prompt: WorldLookBackPrompt? = nil
    ) -> WorldRecoveryReport {
        WorldRecoveryReport(state: state, episode: episode, promptsEnabled: enabled, prompt: prompt ?? self.prompt)
    }

    private func speaks(
        _ recovery: WorldRecoveryReport? = nil, binding: WorldSessionBinding? = nil, followingLive: Bool = true,
        towerSentAt: Double? = 101, lastSpoken: Int? = nil
    ) -> Bool {
        WorldLookBackRule.promptToSpeak(
            recovery: recovery ?? self.recovery(), binding: binding ?? bound, followingLive: followingLive,
            towerSentAt: towerSentAt, lastSpoken: lastSpoken) != nil
    }

    func testAllFourConditionsSpeak() {
        XCTAssertTrue(speaks())
        XCTAssertTrue(speaks(towerSentAt: 104), "speak_until is inclusive")
    }

    /// 1 (and C1 E1): live, bound, following -- never pinned, never unbound.
    func testOnlyALiveBoundUnpinnedWalkSpeaks() {
        XCTAssertFalse(speaks(followingLive: false), "pinned")
        for binding in [WorldSessionBinding.none, .awaiting(captureID: nil), .foreign(captureID: "other")] {
            XCTAssertFalse(speaks(binding: binding), "\(binding)")
        }
    }

    /// 2: prompting, this episode, a known kind, prompts on.
    func testOnlyAPromptOfThisEpisodeAndAKnownKindSpeaks() {
        for state in [WorldRecoveryState.none, .searching, .recovered, .timedOut, WorldRecoveryState(rawValue: "x")] {
            XCTAssertFalse(speaks(recovery(state: state)), state.rawValue)
        }
        XCTAssertFalse(speaks(recovery(episode: 3)), "a prompt from an earlier episode")
        XCTAssertFalse(speaks(recovery(prompt: WorldLookBackPrompt(id: 3, episode: 2, kind: "turn-left",
                                                                   issuedAt: 100, speakUntil: 104))),
                       "an unknown kind is never spoken")
        XCTAssertFalse(speaks(recovery(enabled: false)), "prompts switched off")
        XCTAssertFalse(speaks(WorldRecoveryReport(state: .prompting, episode: 2, promptsEnabled: true, prompt: nil)))
        XCTAssertNil(WorldLookBackRule.promptToSpeak(recovery: nil, binding: bound, followingLive: true,
                                                     towerSentAt: 101, lastSpoken: nil))
    }

    /// 3: an id at or below the last spoken is never spoken.
    func testLastSpokenOnlyRises() {
        XCTAssertFalse(speaks(lastSpoken: 3))
        XCTAssertFalse(speaks(lastSpoken: 7))
        XCTAssertTrue(speaks(lastSpoken: 2))
    }

    /// 4: stale on the Tower's own clock, or no clock at all.
    func testAStalePromptIsNeverSpoken() {
        XCTAssertFalse(speaks(towerSentAt: 104.01))
        XCTAssertFalse(speaks(towerSentAt: nil))
    }
}

@MainActor
final class WorldLookBackPrompterTests: XCTestCase {

    private let bound = WorldSessionBinding.bound(captureID: "cap-1")

    private func report(id: Int, episode: Int? = nil, state: WorldRecoveryState = .prompting) -> WorldRecoveryReport {
        WorldRecoveryReport(state: state, episode: episode ?? id, promptsEnabled: true,
                            prompt: WorldLookBackPrompt(id: id, episode: episode ?? id, issuedAt: 100, speakUntil: 104))
    }

    /// `towerSentAt` 101 against `speak_until` 104: three seconds are left.
    private func consider(_ prompter: WorldLookBackPrompter, _ report: WorldRecoveryReport?,
                          session: String = "s1", binding: WorldSessionBinding? = nil, followingLive: Bool = true,
                          towerSentAt: Double = 101) {
        prompter.consider(recovery: report, binding: binding ?? bound, followingLive: followingLive,
                          towerSentAt: towerSentAt, worldID: "w1", sessionID: session)
    }

    /// A prompter on a fake clock, with a private notification center so no
    /// other test's `didBecomeActive` reaches it.
    private func prompter(_ cue: RecordingLookBackCue, now: @escaping () -> Date) -> WorldLookBackPrompter {
        let prompter = WorldLookBackPrompter(cue: cue, notificationCenter: NotificationCenter())
        prompter.clock = now
        return prompter
    }

    func testEachPromptIsShownOncePerSessionWithOneHaptic() {
        let cue = RecordingLookBackCue()
        let prompter = WorldLookBackPrompter(cue: cue)
        consider(prompter, report(id: 1))
        consider(prompter, report(id: 1))
        consider(prompter, report(id: 1))
        XCTAssertEqual(cue.alerts, [1])
        XCTAssertEqual(prompter.banner, .lookBack(promptID: 1))
        XCTAssertEqual(prompter.banner?.text, "Tracking lost — slowly look back the way you came.")
        XCTAssertEqual(prompter.lastSpoken["w1/s1"], 1)

        // Another session is another key.
        consider(prompter, report(id: 1), session: "s2")
        XCTAssertEqual(cue.alerts, [1, 1])
    }

    /// Defence in depth (§6.4): a third inside 60 s of the phone's own clock
    /// is refused; the Tower's limiter is the design.
    func testTheLocalCapRefusesAThirdInsideAMinute() {
        let cue = RecordingLookBackCue()
        let prompter = WorldLookBackPrompter(cue: cue)
        var now = Date(timeIntervalSince1970: 1_000)
        prompter.clock = { now }
        consider(prompter, report(id: 1))
        now += 10
        consider(prompter, report(id: 2))
        now += 10
        consider(prompter, report(id: 3))
        XCTAssertEqual(cue.alerts, [1, 2])
        now += 45
        consider(prompter, report(id: 4))
        XCTAssertEqual(cue.alerts, [1, 2, 4], "the window moved on")
    }

    /// Nothing is shown for a pinned or unbound phone, and a banner that was
    /// up comes down the moment the phone stops following the live walk.
    func testAPinnedOrUnboundPhoneShowsNothing() {
        let cue = RecordingLookBackCue()
        let prompter = WorldLookBackPrompter(cue: cue)
        consider(prompter, report(id: 1), followingLive: false)
        consider(prompter, report(id: 1), binding: WorldSessionBinding.none)
        XCTAssertNil(prompter.banner)
        XCTAssertTrue(cue.alerts.isEmpty)

        consider(prompter, report(id: 1))
        XCTAssertEqual(prompter.banner, .lookBack(promptID: 1))
        consider(prompter, report(id: 1), followingLive: false)
        XCTAssertNil(prompter.banner, "a pinned phone kept the live walk's banner")
        XCTAssertEqual(cue.alerts, [1])
    }

    /// The banner lasts what is left of the window at delivery,
    /// `speak_until - tower_sent_at` (3 s here), counted from receipt -- not
    /// the full `speak_window_s` (4 s).
    func testTheBannerExpiresWhenTheWindowEnds() {
        var now = Date(timeIntervalSince1970: 1_000)
        let prompter = prompter(RecordingLookBackCue()) { now }
        consider(prompter, report(id: 1))
        now += 2.9
        prompter.expireIfDue()
        XCTAssertEqual(prompter.banner, .lookBack(promptID: 1))
        now += 0.2
        prompter.expireIfDue()
        XCTAssertNil(prompter.banner, "the banner outlived speak_until")
    }

    /// A late delivery (sent 3.5 s into a 4 s window) shows for the 0.5 s
    /// left, and one sent exactly at `speak_until` for nothing more (clamped
    /// at 0), though it still counts as shown.
    func testADelayedPromptShowsOnlyForWhatIsLeftOfItsWindow() {
        let cue = RecordingLookBackCue()
        var now = Date(timeIntervalSince1970: 1_000)
        let prompter = prompter(cue) { now }
        consider(prompter, report(id: 1), towerSentAt: 103.5)
        XCTAssertEqual(prompter.banner, .lookBack(promptID: 1))
        XCTAssertEqual(cue.alerts, [1])
        now += 0.4
        prompter.expireIfDue()
        XCTAssertEqual(prompter.banner, .lookBack(promptID: 1))
        now += 0.2
        prompter.expireIfDue()
        XCTAssertNil(prompter.banner, "a late prompt was shown for its full window")

        consider(prompter, report(id: 2), towerSentAt: 104)
        XCTAssertEqual(cue.alerts, [1, 2])
        prompter.expireIfDue()
        XCTAssertNil(prompter.banner)
    }

    /// Locked in a pocket: the prompt is held, not consumed -- no banner, no
    /// haptic, `lastSpoken` untouched, however many heartbeats repeat it --
    /// and is presented when the app becomes active inside its window, for
    /// what is left of it.
    func testAPromptThatArrivesInTheBackgroundIsShownOnBecomingActiveInItsWindow() {
        let cue = RecordingLookBackCue()
        cue.isForegroundActive = false
        var now = Date(timeIntervalSince1970: 1_000)
        let prompter = prompter(cue) { now }
        consider(prompter, report(id: 1))
        now += 0.5
        consider(prompter, report(id: 1), towerSentAt: 101.5)
        XCTAssertNil(prompter.banner)
        XCTAssertTrue(cue.alerts.isEmpty)
        XCTAssertNil(prompter.lastSpoken["w1/s1"], "a prompt nobody saw was consumed")
        XCTAssertEqual(prompter.held?.promptID, 1)

        now += 1.5
        cue.isForegroundActive = true
        prompter.appDidBecomeActive()
        XCTAssertEqual(prompter.banner, .lookBack(promptID: 1))
        XCTAssertEqual(cue.alerts, [1])
        XCTAssertEqual(prompter.lastSpoken["w1/s1"], 1)
        XCTAssertNil(prompter.held)

        // Its deadline is the first delivery's: 3 s after receipt, so 1 s
        // is left.
        now += 0.9
        prompter.expireIfDue()
        XCTAssertEqual(prompter.banner, .lookBack(promptID: 1))
        now += 0.2
        prompter.expireIfDue()
        XCTAssertNil(prompter.banner)

        // Already shown: neither a heartbeat nor another activation repeats it.
        consider(prompter, report(id: 1), towerSentAt: 103.9)
        prompter.appDidBecomeActive()
        XCTAssertEqual(cue.alerts, [1])
    }

    /// Becoming active after the window ended shows nothing late.
    func testABackgroundPromptWhoseWindowEndedIsNeverShown() {
        let cue = RecordingLookBackCue()
        cue.isForegroundActive = false
        var now = Date(timeIntervalSince1970: 1_000)
        let prompter = prompter(cue) { now }
        consider(prompter, report(id: 1))
        now += 3.1
        cue.isForegroundActive = true
        prompter.appDidBecomeActive()
        XCTAssertNil(prompter.banner)
        XCTAssertTrue(cue.alerts.isEmpty)
        XCTAssertNil(prompter.held)
    }

    /// While active, the next report that repeats a held prompt presents it
    /// at once, and the activation that follows does not present it again.
    func testAHeldPromptIsShownByTheNextReportOnceActive() {
        let cue = RecordingLookBackCue()
        cue.isForegroundActive = false
        var now = Date(timeIntervalSince1970: 1_000)
        let prompter = prompter(cue) { now }
        consider(prompter, report(id: 1))
        now += 1
        cue.isForegroundActive = true
        consider(prompter, report(id: 1), towerSentAt: 102)
        prompter.appDidBecomeActive()
        XCTAssertEqual(prompter.banner, .lookBack(promptID: 1))
        XCTAssertEqual(cue.alerts, [1])
    }

    /// A held prompt is dropped when its episode stops prompting, and when the
    /// phone stops following the live walk.
    func testAHeldPromptIsDroppedWhenTheEpisodeEndsOrThePhoneStopsFollowing() {
        let cue = RecordingLookBackCue()
        cue.isForegroundActive = false
        let now = Date(timeIntervalSince1970: 1_000)
        let prompter = prompter(cue) { now }

        consider(prompter, report(id: 1))
        consider(prompter, report(id: 1, state: .recovered))
        XCTAssertNil(prompter.held)
        cue.isForegroundActive = true
        prompter.appDidBecomeActive()
        XCTAssertNil(prompter.banner, "a recovered episode's prompt was shown")

        cue.isForegroundActive = false
        consider(prompter, report(id: 2))
        consider(prompter, report(id: 2), followingLive: false)
        XCTAssertNil(prompter.held)
        cue.isForegroundActive = true
        prompter.appDidBecomeActive()
        XCTAssertNil(prompter.banner, "a pinned phone showed the live walk's prompt")
        XCTAssertTrue(cue.alerts.isEmpty)
    }

    /// The real wiring: `didBecomeActiveNotification` on the prompter's
    /// center presents the held prompt.
    func testTheDidBecomeActiveNotificationPresentsTheHeldPrompt() async {
        let cue = RecordingLookBackCue()
        cue.isForegroundActive = false
        let center = NotificationCenter()
        let prompter = WorldLookBackPrompter(cue: cue, notificationCenter: center)
        consider(prompter, report(id: 1))
        cue.isForegroundActive = true
        center.post(name: UIApplication.didBecomeActiveNotification, object: nil)
        for _ in 0..<50 where cue.alerts.isEmpty { try? await Task.sleep(for: .milliseconds(20)) }
        XCTAssertEqual(cue.alerts, [1])
        XCTAssertEqual(prompter.banner, .lookBack(promptID: 1))
    }

    /// Leaving `prompting` takes the banner down at once; `recovered` says
    /// "Back on track" for a moment first.
    func testTheBannerFollowsTheEpisode() {
        let prompter = WorldLookBackPrompter(cue: RecordingLookBackCue())
        var now = Date(timeIntervalSince1970: 1_000)
        prompter.clock = { now }

        consider(prompter, report(id: 1))
        consider(prompter, report(id: 1, state: .searching))
        XCTAssertNil(prompter.banner, "still up after the episode stopped prompting")

        consider(prompter, report(id: 2))
        XCTAssertEqual(prompter.banner, .lookBack(promptID: 2))
        consider(prompter, report(id: 2, state: .recovered))
        XCTAssertEqual(prompter.banner, .backOnTrack)
        XCTAssertEqual(prompter.banner?.text, "Back on track")
        now += WorldLookBackPrompter.backOnTrackSeconds + 0.1
        prompter.expireIfDue()
        XCTAssertNil(prompter.banner)

        // A recovery with no banner up shows nothing.
        consider(prompter, report(id: 2, state: .recovered))
        XCTAssertNil(prompter.banner)
    }

    /// Every change reaches the screen through `onChange`.
    func testEveryBannerChangeIsPublished() {
        let prompter = WorldLookBackPrompter(cue: RecordingLookBackCue())
        var seen: [WorldLookBackBanner?] = []
        prompter.onChange = { seen.append($0) }
        consider(prompter, report(id: 1))
        consider(prompter, report(id: 1))
        consider(prompter, report(id: 1, state: .recovered))
        prompter.dismiss()
        XCTAssertEqual(seen, [.lookBack(promptID: 1), .backOnTrack, nil])
    }
}

/// **The invariant from walk 1:** the look-back path, and the app as a whole,
/// never touches the audio session, speech, or the Bluetooth audio route.
/// Speech to the glasses over A2DP ended the DAT camera session ~3 s later.
///
/// Two layers, both failing closed on the Simulator (where this suite runs):
/// the app's Swift sources, read from beside this file, and -- whatever the
/// sources say, and whatever language added it -- the app's own built
/// binaries, which must not link an audio framework or reference an audio
/// class or function.
final class WorldLookBackAudioFreeTests: XCTestCase {

    /// Anything that configures, activates, or plays through the audio
    /// system, or posts a notification (which can carry a sound).
    static let audioAPIs = [
        "import AVFoundation", "import AVFAudio", "import AudioToolbox", "import MediaPlayer",
        "import CoreAudio", "import AudioUnit", "import UserNotifications",
        "AVAudioSession", "AVAudioApplication", "AVSpeechSynthesizer", "AVSpeechUtterance", "AVAudioPlayer",
        "AVAudioEngine", "AVPlayer", "AudioServicesPlay", "SystemSoundID", "AudioComponent",
        "UNNotificationSound", "UNUserNotificationCenter", "overrideOutputAudioPort", "setPreferredInput",
    ]

    /// Byte markers in a linked Mach-O: an audio framework in a load
    /// command, or an audio class or function among its symbols. The spoken
    /// build (`4b4b444`) carries `/AVFAudio.framework/`,
    /// `_OBJC_CLASS_$_AVAudio` and `_OBJC_CLASS_$_AVSpeech`; `f33bdfe` none.
    static let binaryMarkers = [
        "/AVFAudio.framework/", "/AudioToolbox.framework/", "/MediaPlayer.framework/",
        "/CoreAudio.framework/", "/UserNotifications.framework/",
        "_OBJC_CLASS_$_AVAudio", "_OBJC_CLASS_$_AVSpeech", "_OBJC_CLASS_$_AVPlayer",
        "_OBJC_CLASS_$_AVQueuePlayer", "_OBJC_CLASS_$_UNNotificationSound",
        "_OBJC_CLASS_$_UNUserNotificationCenter", "_AudioServicesPlay",
        "_AudioServicesCreateSystemSoundID", "_AudioComponentInstanceNew", "_AudioOutputUnitStart",
    ]

    static func markers(in binary: Data) -> [String] {
        binaryMarkers.filter { binary.range(of: Data($0.utf8)) != nil }
    }

    /// Unreadable sources fail on the Simulator, where they are always on
    /// the host; only a device run skips, and the binary test covers it.
    private func requireReadable(_ url: URL) throws {
        guard !FileManager.default.fileExists(atPath: url.path) else { return }
        #if targetEnvironment(simulator)
        throw NSError(domain: "WorldLookBackAudioFreeTests", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "the app's sources are not readable from this Simulator run (\(url.path))",
        ])
        #else
        throw XCTSkip("sources are not on a device; testTheBuiltAppLinksNoAudio covers it")
        #endif
    }

    private var appSources: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Glasses")
    }

    private func swiftFiles(under root: URL) throws -> [URL] {
        try requireReadable(root)
        let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)
        return (walker?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }
    }

    private func offences(in file: URL) throws -> [String] {
        let text = try String(contentsOf: file, encoding: .utf8)
        return Self.audioAPIs.filter { text.contains($0) }
    }

    func testThePromptPathTouchesNoAudioAPI() throws {
        let file = appSources.appendingPathComponent("Workspaces/WorldBuilder/WorldLookBack.swift")
        try requireReadable(file)
        XCTAssertEqual(try offences(in: file), [], "the look-back path references an audio API")
    }

    /// Wider than the prompt: nothing else in the app plays audio either, so
    /// nothing can move the glasses' Bluetooth route mid-stream.
    func testNoAppSourceTouchesTheAudioSystem() throws {
        var found: [String: [String]] = [:]
        for file in try swiftFiles(under: appSources) {
            let hits = try offences(in: file)
            if !hits.isEmpty { found[file.lastPathComponent] = hits }
        }
        XCTAssertEqual(found, [:], "an app source references an audio API")
    }

    /// The binaries this test's host app was built into: the executable, and
    /// in a DEBUG build the `Glasses.debug.dylib` beside it that holds the
    /// app's code. Neither may carry an audio marker.
    func testTheBuiltAppLinksNoAudio() throws {
        let executable = try XCTUnwrap(Bundle.main.executableURL, "the host app has no executable")
        XCTAssertEqual(executable.lastPathComponent, "Glasses", "not the Glasses app's bundle")
        var binaries = [executable]
        let debugDylib = Bundle.main.bundleURL.appendingPathComponent("Glasses.debug.dylib")
        if FileManager.default.fileExists(atPath: debugDylib.path) { binaries.append(debugDylib) }
        var found: [String: [String]] = [:]
        for binary in binaries {
            let hits = Self.markers(in: try Data(contentsOf: binary))
            if !hits.isEmpty { found[binary.lastPathComponent] = hits }
        }
        XCTAssertEqual(found, [:], "the built app links or references audio")
    }

    /// The binary check is not vacuous: the spoken build's own markers trip it.
    func testTheBinaryCheckCatchesTheSpokenBuild() {
        let spoken = Data("x/System/Library/Frameworks/AVFAudio.framework/AVFAudio\0_OBJC_CLASS_$_AVAudioSession\0_OBJC_CLASS_$_AVSpeechSynthesizer\0".utf8)
        XCTAssertEqual(Self.markers(in: spoken),
                       ["/AVFAudio.framework/", "_OBJC_CLASS_$_AVAudio", "_OBJC_CLASS_$_AVSpeech"])
        XCTAssertEqual(Self.markers(in: Data("CoreMedia libswiftAVFoundation".utf8)), [])
    }

    /// The `audio` background mode existed only for the spoken prompt.
    func testTheAppDeclaresNoAudioBackgroundMode() throws {
        let modes = Bundle.main.object(forInfoDictionaryKey: "UIBackgroundModes") as? [String] ?? []
        XCTAssertFalse(modes.contains("audio"), "UIBackgroundModes still declares audio: \(modes)")
        XCTAssertTrue(modes.contains("bluetooth-central"), "read the host app's Info.plist, not another bundle's")
    }

    /// The prompter's only non-visual seam is a haptic cue.
    @MainActor
    func testTheDefaultCueIsTheHaptic() {
        let cue: WorldLookBackCue = WorldHapticLookBackCue()
        XCTAssertTrue(cue is WorldHapticLookBackCue)
    }
}
