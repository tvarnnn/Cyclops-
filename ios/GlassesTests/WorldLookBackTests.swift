//
//  WorldLookBackTests.swift
//  GlassesTests
//
//  `tracking.recovery` and the spoken look-back prompt
//  (`WORLD-BUILDER-COMPONENTS.md` §6, contract v2 @ bb80b2c; C1 E1, E2, E6,
//  E7, E14, M4, M5, M14). The Tower's relocalizer does not exist yet (P3.2);
//  the blocks below are written from §6.2's table.
//

import AVFoundation
import XCTest

@testable import Glasses

/// Records what would be said, instead of saying it.
@MainActor
final class RecordingLookBackVoice: WorldLookBackVoice {
    private(set) var spoken: [(text: String, promptID: Int)] = []
    private(set) var prepared = 0
    var answers = true

    func prepare() { prepared += 1 }

    @discardableResult
    func speak(_ text: String, promptID: Int) -> Bool {
        spoken.append((text, promptID))
        return answers
    }
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

    private func report(id: Int, episode: Int? = nil) -> WorldRecoveryReport {
        WorldRecoveryReport(state: .prompting, episode: episode ?? id, promptsEnabled: true,
                            prompt: WorldLookBackPrompt(id: id, episode: episode ?? id, issuedAt: 100, speakUntil: 104))
    }

    private func consider(_ prompter: WorldLookBackPrompter, _ report: WorldRecoveryReport,
                          session: String = "s1", binding: WorldSessionBinding? = nil, followingLive: Bool = true) {
        prompter.consider(recovery: report, binding: binding ?? bound, followingLive: followingLive,
                          towerSentAt: 101, worldID: "w1", sessionID: session)
    }

    func testEachPromptIsSpokenOncePerSession() {
        let voice = RecordingLookBackVoice()
        let prompter = WorldLookBackPrompter(voice: voice)
        consider(prompter, report(id: 1))
        consider(prompter, report(id: 1))
        consider(prompter, report(id: 1))
        XCTAssertEqual(voice.spoken.map(\.promptID), [1])
        XCTAssertEqual(voice.spoken.first?.text, "Look back the way you came.")
        XCTAssertEqual(prompter.lastSpoken["w1/s1"], 1)

        // Another session is another key.
        consider(prompter, report(id: 1), session: "s2")
        XCTAssertEqual(voice.spoken.map(\.promptID), [1, 1])
    }

    /// `lastSpoken` is set BEFORE speech starts: a voice that could not speak
    /// (no Bluetooth route) is not asked again for the same id.
    func testAPromptTheVoiceCouldNotSayIsNotRetried() {
        let voice = RecordingLookBackVoice()
        voice.answers = false
        let prompter = WorldLookBackPrompter(voice: voice)
        consider(prompter, report(id: 1))
        consider(prompter, report(id: 1))
        XCTAssertEqual(voice.spoken.count, 1)
    }

    /// Defence in depth (§6.4): a third inside 60 s of the phone's own clock
    /// is refused; the Tower's limiter is the design.
    func testTheLocalCapRefusesAThirdInsideAMinute() {
        let voice = RecordingLookBackVoice()
        let prompter = WorldLookBackPrompter(voice: voice)
        var now = Date(timeIntervalSince1970: 1_000)
        prompter.clock = { now }
        consider(prompter, report(id: 1))
        now += 10
        consider(prompter, report(id: 2))
        now += 10
        consider(prompter, report(id: 3))
        XCTAssertEqual(voice.spoken.map(\.promptID), [1, 2])
        now += 45
        consider(prompter, report(id: 4))
        XCTAssertEqual(voice.spoken.map(\.promptID), [1, 2, 4], "the window moved on")
    }

    /// Nothing is prepared, let alone spoken, for a pinned or unbound phone.
    func testAPinnedOrUnboundPhoneDoesNotEvenPrepare() {
        let voice = RecordingLookBackVoice()
        let prompter = WorldLookBackPrompter(voice: voice)
        consider(prompter, report(id: 1), followingLive: false)
        consider(prompter, report(id: 1), binding: WorldSessionBinding.none)
        XCTAssertEqual(voice.prepared, 0)
        XCTAssertTrue(voice.spoken.isEmpty)
        consider(prompter, report(id: 1))
        XCTAssertEqual(voice.prepared, 1)
    }

    /// The real voice's route rule: A2DP or LE only, never the phone speaker
    /// and never HFP (C1 E6).
    func testTheVoiceSpeaksOnlyOnABluetoothPlaybackRoute() {
        XCTAssertFalse(WorldSpeechLookBackVoice.isBluetoothOutput(AVAudioSession.sharedInstance().currentRoute),
                       "the Simulator's route is not Bluetooth")
        XCTAssertEqual(WorldLookBackPrompter.sentence, "Look back the way you came.")
    }
}
