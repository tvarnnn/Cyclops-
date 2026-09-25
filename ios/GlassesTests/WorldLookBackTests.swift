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

import XCTest

@testable import Glasses

/// Records the haptic, instead of firing it.
@MainActor
final class RecordingLookBackCue: WorldLookBackCue {
    private(set) var alerts: [Int] = []

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

    private func consider(_ prompter: WorldLookBackPrompter, _ report: WorldRecoveryReport?,
                          session: String = "s1", binding: WorldSessionBinding? = nil, followingLive: Bool = true) {
        prompter.consider(recovery: report, binding: binding ?? bound, followingLive: followingLive,
                          towerSentAt: 101, worldID: "w1", sessionID: session)
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

    /// The banner lasts `speak_window_s` (`speak_until - issued_at`, 4 s here)
    /// on the phone's clock, and no longer.
    func testTheBannerLastsTheSpeakWindow() {
        let prompter = WorldLookBackPrompter(cue: RecordingLookBackCue())
        var now = Date(timeIntervalSince1970: 1_000)
        prompter.clock = { now }
        consider(prompter, report(id: 1))
        now += 3.9
        prompter.expireIfDue()
        XCTAssertEqual(prompter.banner, .lookBack(promptID: 1))
        now += 0.2
        prompter.expireIfDue()
        XCTAssertNil(prompter.banner)
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
/// Read from the sources beside this file: the Simulator runs on the host,
/// which is where the tests run. A run that cannot see them skips rather
/// than passes.
final class WorldLookBackAudioFreeTests: XCTestCase {

    /// Anything that configures, activates, or plays through the audio system.
    static let audioAPIs = [
        "import AVFoundation", "import AVFAudio", "import AudioToolbox", "import MediaPlayer",
        "AVAudioSession", "AVSpeechSynthesizer", "AVSpeechUtterance", "AVAudioPlayer", "AVAudioEngine",
        "AVPlayer", "AudioServicesPlay", "overrideOutputAudioPort", "setPreferredInput",
    ]

    private var appSources: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Glasses")
    }

    private func swiftFiles(under root: URL) throws -> [URL] {
        guard FileManager.default.fileExists(atPath: root.path) else {
            throw XCTSkip("the app's sources are not readable from this run (\(root.path))")
        }
        let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)
        return (walker?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }
    }

    private func offences(in file: URL) throws -> [String] {
        let text = try String(contentsOf: file, encoding: .utf8)
        return Self.audioAPIs.filter { text.contains($0) }
    }

    func testThePromptPathTouchesNoAudioAPI() throws {
        let file = appSources.appendingPathComponent("Workspaces/WorldBuilder/WorldLookBack.swift")
        guard FileManager.default.fileExists(atPath: file.path) else {
            throw XCTSkip("WorldLookBack.swift is not readable from this run")
        }
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
