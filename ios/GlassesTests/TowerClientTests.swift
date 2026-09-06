//
//  TowerClientTests.swift
//  GlassesTests
//
//  Exercises the WebSocket observability/liveness behavior added to
//  TowerClient: the continuous receive loop, delegate close observation,
//  send-path truthfulness, and connect/disconnect lifecycle cleanliness.
//  Runs against a real local WebSocket server (MockTowerServer) rather than
//  the actual Tower.
//

import Combine
import Network
import os
import UIKit
import XCTest

@testable import Glasses

@MainActor
final class TowerClientTests: XCTestCase {

    private func url(port: UInt16) -> URL {
        URL(string: "ws://127.0.0.1:\(port)/")!
    }

    /// Responds to the client's initial ping with a pong, exactly like the
    /// real Tower's handshake.
    private func respondToPing(_ server: MockTowerServer) {
        server.onText = { text in
            guard
                let data = text.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
                json["type"] == "ping"
            else { return }
            server.send(text: #"{"type":"pong"}"#)
        }
    }

    /// Same as `respondToPing`, but also records every text message the
    /// server receives (pings included) for later inspection.
    private func attachRecorder(_ server: MockTowerServer) -> MessageRecorder {
        let recorder = MessageRecorder()
        server.onText = { text in
            recorder.record(text)
            guard
                let data = text.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
                json["type"] == "ping"
            else { return }
            server.send(text: #"{"type":"pong"}"#)
        }
        return recorder
    }

    /// Decodes a recorded text message's top-level JSON object.
    private func decode(_ text: String) -> [String: Any]? {
        guard let data = text.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    /// `XCTAssertTrue(await waitUntil { … })` does not compile — an autoclosure
    /// cannot carry an `await` — so the assertion is wrapped rather than
    /// hoisted at every call site. `#filePath`/`#line` default arguments keep
    /// the failure pointing at the test rather than at this helper.
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

    /// Waits for the connection's own capability request to land.
    ///
    /// `TowerClient` sends `{"type":"cartridges"}` once per connection, right
    /// after the pong. Tests that assert "nothing else reached the wire" have
    /// to take their baseline *after* it, or they are racing the handshake
    /// rather than measuring what they claim to measure.
    private func waitForDiscovery(_ recorder: MessageRecorder) async -> Bool {
        await waitUntil {
            recorder.all.compactMap(self.decode).contains { $0["type"] as? String == "cartridges" }
        }
    }

    private func makeTestImage() -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 2, height: 2))
        return renderer.image { context in
            UIColor.red.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
        }
    }

    // MARK: - 1. Initial ping/pong still works

    func testInitialPingPongStillWorks() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))

        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline, "expected .online, got \(client.status)")

        client.disconnect()
    }

    // MARK: - 2. Receive loop handles multiple frame_result messages

    func testReceiveLoopHandlesMultipleFrameResultMessages() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        for seq in 1...3 {
            server.send(text: #"{"type":"frame_result","seq":\#(seq),"mean_intensity":0.42,"processing_ms":8.5}"#)
        }

        let receivedAll = await waitUntil { client.frameResultCount >= 3 }
        XCTAssertTrue(receivedAll, "expected 3 frame_result messages processed, got \(client.frameResultCount)")
        XCTAssertEqual(client.status, .online, "processing frame_result must not affect connection status")

        client.disconnect()
    }

    // MARK: - 3. Unknown inbound message does not kill the connection

    func testUnknownInboundMessageDoesNotKillConnection() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        server.send(text: #"{"type":"something_unexpected","foo":"bar"}"#)
        // Prove the connection survived the unknown message by sending a
        // real frame_result right after and confirming it's still processed.
        server.send(text: #"{"type":"frame_result","seq":1,"mean_intensity":0.1,"processing_ms":1.0}"#)

        let stillHealthy = await waitUntil { client.frameResultCount >= 1 }
        XCTAssertTrue(stillHealthy)
        XCTAssertEqual(client.status, .online)

        client.disconnect()
    }

    // MARK: - 4. Receive failure changes status away from .online

    func testReceiveFailureChangesStatusAwayFromOnline() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        server.dropConnection()
        server.stop()

        let leftOnline = await waitUntil { client.status != .online }
        XCTAssertTrue(leftOnline, "status is stuck at .online after the connection was dropped")
        guard case .failed = client.status else {
            XCTFail("expected .failed after a receive failure, got \(client.status)")
            return
        }
    }

    // MARK: - 5. Delegate close changes status appropriately

    func testDelegateCloseChangesStatusAppropriately() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.simulateDelegateCloseForTesting(code: .normalClosure)

        let leftOnline = await waitUntil { client.status != .online }
        XCTAssertTrue(leftOnline, "status did not change after a delegate close callback")
        guard case .failed = client.status else {
            XCTFail("expected .failed after delegate close, got \(client.status)")
            return
        }
    }

    // MARK: - 6. disconnect() cancels the receive loop cleanly

    func testDisconnectCancelsReceiveLoopCleanly() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.disconnect()
        XCTAssertEqual(client.status, .offline)

        // A message sent after disconnect() must never be observed — proves
        // the receive loop actually stopped rather than lingering.
        server.send(text: #"{"type":"frame_result","seq":99,"mean_intensity":0,"processing_ms":0}"#)
        try? await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertEqual(client.frameResultCount, 0)
        XCTAssertEqual(client.status, .offline)
    }

    // MARK: - 7. Repeated connect/disconnect does not leave duplicate listeners

    func testRepeatedConnectDisconnectDoesNotLeaveDuplicateListeners() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient()

        client.connect(to: url(port: port))
        var becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)
        client.disconnect()
        XCTAssertEqual(client.status, .offline)

        client.connect(to: url(port: port))
        becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        server.send(text: #"{"type":"frame_result","seq":1,"mean_intensity":0,"processing_ms":0}"#)
        let received = await waitUntil { client.frameResultCount == 1 }
        XCTAssertTrue(
            received,
            "expected exactly 1 frame_result, got \(client.frameResultCount) — a leaked prior receive loop would double-count or misbehave"
        )

        client.disconnect()
    }

    // MARK: - 8. Send failure does not leave stale .online state

    func testSendFailureDoesNotLeaveStaleOnlineState() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        // The stream bracket has to be opened *before* the drop, or sendFrame
        // returns at the isStreamingToTower guard and never reaches
        // `task.send` — in which case this test would pass on the receive
        // loop alone and prove nothing about the send path it is named for.
        client.sendStreamStart()
        XCTAssertTrue(client.isStreamingToTower, "stream_start must succeed while online")

        server.dropConnection()
        server.stop()

        // Either the concurrent receive loop or this send's completion
        // handler may be the one to observe the drop first; the guarantee
        // under test is the outcome (status leaves .online), not which path
        // wins that race.
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 1)

        let leftOnline = await waitUntil { client.status != .online }
        XCTAssertTrue(leftOnline, "status is stuck at .online after a confirmed send/receive failure")
    }

    // MARK: - 9. stream_start is sent exactly once, with the exact payload

    func testStreamStartSendsExactPayloadOnce() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.sendStreamStart()
        client.sendStreamStart() // redundant call — must not send a second time
        try? await Task.sleep(nanoseconds: 200_000_000)

        let streamStarts = recorder.all.compactMap(decode).filter { $0["type"] as? String == "stream_start" }
        XCTAssertEqual(streamStarts.count, 1, "expected exactly one stream_start, got \(streamStarts.count)")
        XCTAssertEqual(streamStarts.first?.keys.sorted(), ["type"], "stream_start payload must contain exactly {\"type\":\"stream_start\"}")

        client.disconnect()
    }

    // MARK: - 10. stream_stop is sent exactly once, with the exact payload

    func testStreamStopSendsExactPayloadOnce() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.sendStreamStart()
        client.sendStreamStop()
        client.sendStreamStop() // redundant call — must not send a second time
        try? await Task.sleep(nanoseconds: 200_000_000)

        let streamStops = recorder.all.compactMap(decode).filter { $0["type"] as? String == "stream_stop" }
        XCTAssertEqual(streamStops.count, 1, "expected exactly one stream_stop, got \(streamStops.count)")
        XCTAssertEqual(streamStops.first?.keys.sorted(), ["type"], "stream_stop payload must contain exactly {\"type\":\"stream_stop\"}")

        client.disconnect()
    }

    // MARK: - 11. Frames are suppressed after stream_stop until the next stream_start

    func testFramesSuppressedAfterStreamStopUntilNextStreamStart() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        func frameCount() -> Int {
            recorder.all.compactMap(decode).filter { $0["type"] as? String == "frame" }.count
        }

        // Before any stream_start: frames must not be forwarded.
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 1)
        try? await Task.sleep(nanoseconds: 150_000_000)
        XCTAssertEqual(frameCount(), 0, "frame sent before any stream_start")

        client.sendStreamStart()
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 2)
        let sawFirstFrame = await waitUntil { frameCount() == 1 }
        XCTAssertTrue(sawFirstFrame, "frame not sent while streaming was active")

        client.sendStreamStop()
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 3)
        try? await Task.sleep(nanoseconds: 150_000_000)
        XCTAssertEqual(frameCount(), 1, "a frame was sent after stream_stop, before the next stream_start")

        client.sendStreamStart()
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 4)
        let sawSecondFrame = await waitUntil { frameCount() == 2 }
        XCTAssertTrue(sawSecondFrame, "frame not sent again after a fresh stream_start")

        client.disconnect()
    }

    // MARK: - 12. Multiple start/stop cycles on one Tower connection work

    func testMultipleStartStopCyclesOnOneConnection() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        for _ in 0..<3 {
            client.sendStreamStart()
            client.sendStreamStop()
        }
        try? await Task.sleep(nanoseconds: 200_000_000)

        let types = recorder.all.compactMap(decode).compactMap { $0["type"] as? String }
        let starts = types.filter { $0 == "stream_start" }.count
        let stops = types.filter { $0 == "stream_stop" }.count
        XCTAssertEqual(starts, 3, "expected 3 stream_start messages across 3 cycles, got \(starts)")
        XCTAssertEqual(stops, 3, "expected 3 stream_stop messages across 3 cycles, got \(stops)")
        // Same connection throughout — no reconnect was ever triggered.
        XCTAssertEqual(client.status, .online)

        client.disconnect()
    }

    // MARK: - 13. The bounded send window drops the newest frame when full

    /// The window exists so a slow uplink sheds frames instead of queueing
    /// them. Driven synchronously: `sendFrame` hands the message to
    /// URLSession and the completion hops back through
    /// `Task { @MainActor }`, so consecutive calls with no suspension point
    /// between them cannot have had a completion processed yet. That makes the
    /// drop deterministic rather than timing-dependent.
    func testSendWindowDropsFramesWhileEarlierSendsAreStillInFlight() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let metrics = SenderMetrics()
        let client = TowerClient(metrics: metrics, maxFramesInFlight: 2)
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        metrics.begin()
        client.sendStreamStart()

        // Five frames, no await between them.
        let image = makeTestImage()
        for sequence in 1...5 {
            client.sendFrame(image, width: 2, height: 2, sequence: sequence)
        }

        XCTAssertEqual(
            metrics.currentSnapshot.sendAttempts,
            2,
            "only the window's worth of frames may be handed to the socket"
        )
        XCTAssertEqual(
            metrics.currentSnapshot.sendWindowDrops,
            3,
            "the three frames past the window must be dropped, not queued"
        )
        // Dropped frames must not have been encoded — the drop check comes
        // first precisely so a doomed frame costs nothing.
        XCTAssertEqual(metrics.currentSnapshot.framesEncoded, 2)

        let arrived = await waitUntil {
            recorder.all.compactMap(self.decode).filter { $0["type"] as? String == "frame" }.count == 2
        }
        XCTAssertTrue(arrived, "exactly the two admitted frames must reach the server")

        client.disconnect()
    }

    /// Once earlier sends complete the window must reopen, or the pipeline
    /// would send `maxFramesInFlight` frames and then stop forever.
    func testSendWindowReopensAfterSendsComplete() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let metrics = SenderMetrics()
        let client = TowerClient(metrics: metrics, maxFramesInFlight: 1)
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        metrics.begin()
        client.sendStreamStart()

        let image = makeTestImage()
        func frameCount() -> Int {
            recorder.all.compactMap(decode).filter { $0["type"] as? String == "frame" }.count
        }

        for sequence in 1...4 {
            client.sendFrame(image, width: 2, height: 2, sequence: sequence)
            let delivered = await waitUntil { metrics.currentSnapshot.sendSuccesses == sequence }
            XCTAssertTrue(delivered, "send \(sequence) never completed, so the window never reopened")
        }

        XCTAssertEqual(metrics.currentSnapshot.sendWindowDrops, 0, "no drop should occur when sends are awaited")
        let allArrived = await waitUntil { frameCount() == 4 }
        XCTAssertTrue(allArrived)

        client.disconnect()
    }

    /// A dropped connection leaves completion handlers pending. If teardown
    /// did not clear the window, those slots would never be returned and the
    /// next connection could never send anything.
    func testSendWindowIsClearedByReconnectSoSendingResumes() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let metrics = SenderMetrics()
        let client = TowerClient(metrics: metrics, maxFramesInFlight: 1)
        client.connect(to: url(port: port))
        var becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        metrics.begin()
        client.sendStreamStart()
        // Fill the window, then tear the connection down before the send can
        // complete, so a slot is outstanding at teardown time.
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 1)
        // Without this the test would pass even if sendStreamStart() had
        // silently failed and the frame never occupied a window slot at all —
        // proving nothing about the leak it is named for.
        XCTAssertEqual(
            metrics.currentSnapshot.sendAttempts,
            1,
            "frame 1 must actually have consumed the only window slot"
        )
        client.disconnect()

        client.connect(to: url(port: port))
        becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.sendStreamStart()
        let before = metrics.currentSnapshot.sendWindowDrops
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 2)
        XCTAssertEqual(
            metrics.currentSnapshot.sendWindowDrops,
            before,
            "the window was still full after reconnect — a leaked in-flight slot"
        )

        let arrived = await waitUntil {
            recorder.all.compactMap(self.decode)
                .contains { $0["type"] as? String == "frame" && $0["seq"] as? Int == 2 }
        }
        XCTAssertTrue(arrived, "the frame after reconnect never reached the server")

        client.disconnect()
    }

    // MARK: - 14. Stream bracket truthfulness

    /// `isStreamingToTower` claims a `stream_start` is outstanding. A start
    /// attempted while offline reaches no socket, so claiming it would let a
    /// whole session's frames be forwarded outside any bracket.
    func testStreamStartWhileOfflineDoesNotClaimToBeStreaming() {
        let client = TowerClient()
        XCTAssertEqual(client.status, .offline)

        client.sendStreamStart()

        XCTAssertFalse(
            client.isStreamingToTower,
            "a stream_start that reached no socket must not mark the stream open"
        )
    }

    /// No `stream_start` survives a socket, so the flag must not either —
    /// otherwise frames flow onto a connection the Tower never saw a start on.
    func testDisconnectClearsTheStreamBracket() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.sendStreamStart()
        XCTAssertTrue(client.isStreamingToTower)

        client.disconnect()
        XCTAssertFalse(client.isStreamingToTower, "the stream bracket outlived the socket it was opened on")
    }

    /// Clearing the bracket on teardown is only safe because it can be
    /// reopened: `ProjectManager` re-sends `stream_start` when the Tower comes
    /// back online during a live camera stream. If a fresh `stream_start`
    /// could not restore frame flow after a reconnect, a single mid-session
    /// network blip would silently discard every remaining frame.
    func testStreamBracketCanBeReopenedAfterAReconnect() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        var becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.sendStreamStart()
        client.disconnect()
        XCTAssertFalse(client.isStreamingToTower)

        // What ProjectManager now does when status returns to .online.
        client.connect(to: url(port: port))
        becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)
        client.sendStreamStart()
        XCTAssertTrue(client.isStreamingToTower, "the bracket must be reopenable on the new socket")

        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 7)
        let arrived = await waitUntil {
            recorder.all.compactMap(self.decode)
                .contains { $0["type"] as? String == "frame" && $0["seq"] as? Int == 7 }
        }
        XCTAssertTrue(arrived, "frames must flow again after the bracket is reopened")

        client.disconnect()
    }

    /// The counter sits beside the per-session `frameCount` on the dashboard,
    /// so it has to reset on the same boundary rather than accumulating for
    /// the app's lifetime.
    func testFrameResultCountResetsPerStreamBracket() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.sendStreamStart()
        for seq in 1...2 {
            server.send(text: #"{"type":"frame_result","seq":\#(seq),"mean_intensity":0.1,"processing_ms":1.0}"#)
        }
        let counted = await waitUntil { client.frameResultCount == 2 }
        XCTAssertTrue(counted, "expected 2 replies, got \(client.frameResultCount)")

        client.sendStreamStop()
        client.sendStreamStart()
        XCTAssertEqual(client.frameResultCount, 0, "a new stream bracket must start from zero replies")

        client.disconnect()
    }

    // MARK: - 15. Send window sizing and slot instrumentation

    /// The window's capacity is derived from a latency budget rather than
    /// picked, so the shipped number and the arithmetic that justifies it must
    /// not be able to drift apart.
    func testDefaultWindowCapacityIsTheLatencyBudgetedOne() {
        let client = TowerClient(metrics: SenderMetrics())
        XCTAssertEqual(client.maxFramesInFlight, TowerClient.defaultMaxFramesInFlight)
        XCTAssertEqual(
            TowerClient.defaultMaxFramesInFlight,
            SendWindow.capacity(
                forTargetFPS: FrameRateGate.towerTargetFPS,
                latencyBudget: TowerClient.outboundLatencyBudget
            )
        )
        XCTAssertGreaterThan(
            client.maxFramesInFlight,
            2,
            "the baseline capacity of 2 is what held the physical run to 3.4 fps"
        )
    }

    /// Every completed send must contribute a timing sample, or the
    /// `capacity / slotLifetime` diagnosis has nothing to divide.
    func testACompletedSendRecordsItsSlotTimings() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let metrics = SenderMetrics()
        let client = TowerClient(metrics: metrics, maxFramesInFlight: 2)
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        metrics.begin()
        client.sendStreamStart()
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 1)

        let completed = await waitUntil { metrics.currentSnapshot.sendSuccesses == 1 }
        XCTAssertTrue(completed)

        let snapshot = metrics.currentSnapshot
        XCTAssertEqual(snapshot.slotSamples, 1, "a completed send must report a slot lifetime")
        let latency = try XCTUnwrap(snapshot.sendLatencyMsAverage)
        let lifetime = try XCTUnwrap(snapshot.slotLifetimeMsAverage)
        XCTAssertGreaterThanOrEqual(
            lifetime,
            latency,
            "the slot is held until the main actor returns it, so it cannot outlive its own transport time"
        )
        XCTAssertNotNil(snapshot.windowLimitedFPS(capacity: client.maxFramesInFlight))

        client.disconnect()
    }

    /// A send abandoned by teardown has no slot left to time. It must still be
    /// accounted as a terminal outcome, but it must not contribute a sample —
    /// a teardown-length "slot lifetime" would poison the average that the
    /// window is sized against.
    func testAnAbandonedSendContributesNoTimingSample() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let metrics = SenderMetrics()
        let client = TowerClient(metrics: metrics, maxFramesInFlight: 1)
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        metrics.begin()
        client.sendStreamStart()
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 1)
        XCTAssertEqual(metrics.currentSnapshot.sendAttempts, 1)
        client.disconnect()

        let settled = await waitUntil { metrics.currentSnapshot.sendAbandoned == 1 }
        XCTAssertTrue(settled, "the outstanding send never reached a terminal outcome")
        XCTAssertEqual(metrics.currentSnapshot.slotSamples, 0)
        XCTAssertNil(metrics.currentSnapshot.slotLifetimeMsAverage)
    }

    /// Stall detection must not fire on ordinary backpressure. A full window
    /// that is still draining is the mechanism working, and answering it with
    /// a reconnect would turn a shed frame into a dropped connection.
    func testOrdinaryWindowDropsDoNotTripStallDetection() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let metrics = SenderMetrics()
        let client = TowerClient(metrics: metrics, maxFramesInFlight: 2)
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        metrics.begin()
        client.sendStreamStart()
        let image = makeTestImage()
        for sequence in 1...8 {
            client.sendFrame(image, width: 2, height: 2, sequence: sequence)
        }

        XCTAssertGreaterThan(metrics.currentSnapshot.sendWindowDrops, 0, "the window must actually have filled")
        XCTAssertEqual(metrics.currentSnapshot.stallRecoveries, 0)
        XCTAssertEqual(client.status, .online, "a full window is not a broken connection")

        client.disconnect()
    }

    // MARK: - 16. Automatic reconnect

    /// A mid-session drop on a remote Tailscale path is the expected case, and
    /// until now nothing in the app re-established the connection: the Tower
    /// pill went red and stayed red until someone tapped Connect. The
    /// stream-bracket reopening in `ProjectManager` was written for a
    /// reconnect that could not happen.
    func testADroppedConnectionIsReestablishedAutomatically() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics(), autoReconnect: true)
        client.connect(to: url(port: port))
        var becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.simulateDelegateCloseForTesting(code: .abnormalClosure)

        // Waiting for the drop to actually land first is what makes the second
        // wait meaningful. The delegate callback hops to the main actor, so
        // polling for `.online` straight away would observe the *pre-drop*
        // state and pass without a reconnect ever happening.
        let leftOnline = await waitUntil { client.status != .online }
        XCTAssertTrue(leftOnline, "the simulated close never took effect")

        becameOnline = await waitUntil(timeout: 6) { client.status == .online }
        XCTAssertTrue(becameOnline, "the connection was never re-established")

        client.disconnect()
    }

    /// Reconnect must never override the user. A deliberate disconnect that
    /// silently came back would make the control a lie.
    func testDisconnectIsNotUndoneByAPendingReconnect() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics(), autoReconnect: true)
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        // The reconnect has to be genuinely scheduled before it can be
        // genuinely countermanded: `disconnect()` called in the same actor turn
        // as the simulated close would tear the socket down first, and the
        // close would then be discarded as stale without ever scheduling
        // anything. Waiting for `.failed` is what makes this test about
        // cancellation rather than about ordering.
        client.simulateDelegateCloseForTesting(code: .abnormalClosure)
        let failed = await waitUntil { client.status != .online }
        XCTAssertTrue(failed, "a reconnect must have been scheduled for this test to mean anything")

        client.disconnect()

        // Comfortably past the first backoff step.
        try? await Task.sleep(nanoseconds: 1_500_000_000)
        XCTAssertEqual(client.status, .offline, "a cancelled reconnect must not resurrect the connection")
    }

    /// Reconnect is opt-in precisely so that the failure-path tests above keep
    /// asserting about a settled status. This pins that default.
    func testReconnectIsOffUnlessAskedFor() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.simulateDelegateCloseForTesting(code: .abnormalClosure)
        let leftOnline = await waitUntil { client.status != .online }
        XCTAssertTrue(leftOnline)

        try? await Task.sleep(nanoseconds: 1_500_000_000)
        guard case .failed = client.status else {
            XCTFail("expected the connection to stay failed, got \(client.status)")
            return
        }
    }

    // MARK: - 17. Stall detection end to end

    /// Occupies the whole send window and then holds the main actor without
    /// yielding, so the outstanding send's completion cannot hop back to
    /// release its slot.
    ///
    /// A busy-wait rather than `Task.sleep`, deliberately: sleeping yields the
    /// actor, the queued completion runs, the slot comes back and there is no
    /// stall left to detect. Blocking is the only way to hold a window open
    /// against a loopback server that answers instantly, and it makes the test
    /// deterministic rather than timing-dependent.
    private func fillWindowAndHoldMainActor(
        _ client: TowerClient,
        seconds: TimeInterval
    ) {
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 1)
        let holdUntil = MonotonicClock.now + seconds
        while MonotonicClock.now < holdUntil {
            // Intentionally empty: the point is to not suspend.
        }
    }

    /// The path that exists because `URLSessionWebSocketTask` cannot cancel or
    /// time out one outstanding send: a window that stops draining has to be
    /// answered at connection granularity or the pipeline simply reports drops
    /// for as long as the peer stays wedged — 52 seconds, in the physical
    /// baseline.
    func testAWedgedSendWindowReplacesTheConnection() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let metrics = SenderMetrics()
        let client = TowerClient(
            metrics: metrics,
            maxFramesInFlight: 1,
            stallTimeout: 0.05,
            autoReconnect: true
        )
        client.connect(to: url(port: port))
        var becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        metrics.begin()
        client.sendStreamStart()

        // 0.2 s is past `stallTimeout` but well inside `mainActorGapAllowance`,
        // so the next frame is offered by a demonstrably responsive actor and
        // the verdict is reached rather than deferred.
        fillWindowAndHoldMainActor(client, seconds: 0.2)
        XCTAssertEqual(metrics.currentSnapshot.sendAttempts, 1, "the window must actually be occupied")

        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 2)

        XCTAssertEqual(metrics.currentSnapshot.stallRecoveries, 1, "the wedged window was not detected")
        XCTAssertNotEqual(client.status, .online, "a wedged socket must be torn down, not kept")
        // The frame that triggered the teardown still has to reach a terminal
        // outcome, or `framesUnaccounted` would drift by one per stall.
        XCTAssertGreaterThan(metrics.currentSnapshot.sendWindowDrops, 0)

        becameOnline = await waitUntil(timeout: 6) { client.status == .online }
        XCTAssertTrue(becameOnline, "the replacement connection never came up")

        client.disconnect()
    }

    /// The false-positive guard, and the reason `mainActorGapAllowance` exists.
    ///
    /// A slot is held until the completion handler's hop back to the main actor
    /// runs, so if *this actor* stalls, every slot looks old through no fault of
    /// the socket. Tearing down a healthy connection because the main thread
    /// hitched would invert the very diagnosis the slot-timing instrumentation
    /// was added to make possible.
    func testAStalledMainActorIsNotMistakenForAWedgedSocket() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let metrics = SenderMetrics()
        let client = TowerClient(
            metrics: metrics,
            maxFramesInFlight: 1,
            stallTimeout: 0.05
        )
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        metrics.begin()
        client.sendStreamStart()

        // Same setup as above, but the actor is held past
        // `mainActorGapAllowance` — which is exactly what a main-actor stall
        // looks like from here.
        fillWindowAndHoldMainActor(client, seconds: TowerClient.mainActorGapAllowance + 0.1)
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 2)

        XCTAssertEqual(
            metrics.currentSnapshot.stallRecoveries,
            0,
            "a main-actor stall must not be blamed on the socket"
        )
        XCTAssertEqual(client.status, .online, "the connection was healthy and must be kept")

        client.disconnect()
    }

    // MARK: - 18. Automatic connection

    /// The invariant the whole auto-connect design rests on: bringing the Tower
    /// up at launch must not put a single frame on the wire.
    ///
    /// It holds structurally rather than by convention — `sendFrame` requires a
    /// `stream_start` bracket as well as an online socket, and only a live
    /// camera stream opens one — but that is exactly the kind of guarantee that
    /// quietly stops being true, so it is pinned here.
    func testAutomaticConnectOpensTheSocketButSendsNothing() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let metrics = SenderMetrics()
        let client = TowerClient(metrics: metrics)
        client.connectIfIdle(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        metrics.begin()
        XCTAssertFalse(client.isStreamingToTower, "an automatic connect must not open a stream bracket")

        // A frame offered without a bracket must be refused, not sent.
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 1)
        XCTAssertEqual(metrics.currentSnapshot.sendAttempts, 0)
        XCTAssertEqual(metrics.currentSnapshot.sessionGateDrops, 1)

        let frames = recorder.all.compactMap(decode).filter { $0["type"] as? String == "frame" }
        XCTAssertTrue(frames.isEmpty, "no frame may reach the Tower from an automatic connect")

        client.disconnect()
    }

    /// `connectIfIdle` is called from a SwiftUI `.task`, which can run more than
    /// once. A second call must not punch a hole in a live connection —
    /// `connect()` in that situation tears the socket down, which would close
    /// the stream bracket mid-session and drop frames.
    func testAutomaticConnectLeavesAHealthyConnectionAlone() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.sendStreamStart()
        XCTAssertTrue(client.isStreamingToTower)

        client.connectIfIdle(to: url(port: port))

        // Teardown would have cleared the bracket, so this is the precise
        // observable difference between "did nothing" and "reconnected".
        XCTAssertTrue(client.isStreamingToTower, "an automatic connect tore down a live connection")
        XCTAssertEqual(client.status, .online)

        client.disconnect()
    }

    /// A Tower that has failed must stay visibly failed. Automation retrying on
    /// its own is how an app ends up looking like it is still trying after it
    /// has given up — and it would bypass the bounded backoff that exists to
    /// stop exactly that.
    func testAutomaticConnectDoesNotRetryAFailedConnection() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.simulateDelegateCloseForTesting(code: .abnormalClosure)
        let failed = await waitUntil { client.status != .online }
        XCTAssertTrue(failed)

        client.connectIfIdle(to: url(port: port))
        try? await Task.sleep(nanoseconds: 400_000_000)

        guard case .failed = client.status else {
            XCTFail("automatic connect resurrected a failed connection: \(client.status)")
            return
        }
    }

    /// After a deliberate disconnect the client is `.offline`, and automation
    /// may bring it back — that is the state auto-connect exists for. Pinned so
    /// the `.offline` guard is not mistaken for "never reconnect".
    func testAutomaticConnectDoesConnectWhenOffline() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        XCTAssertEqual(client.status, .offline)

        client.connectIfIdle(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline, "automatic connect must work from the idle state")

        client.disconnect()
    }

    // MARK: - 19. The Tower's own reply, surfaced

    /// `mean_intensity` was decoded for a log line and discarded. It is the only
    /// thing the Tower says about a frame's *content*, and a workspace that
    /// shows it is describing what the Tower really does instead of a
    /// capability it does not have.
    func testTheLatestFrameResultIsSurfacedNotJustCounted() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        XCTAssertNil(client.latestFrameResult, "nothing has been reported yet")

        server.send(text: #"{"type":"frame_result","seq":7,"mean_intensity":0.42,"processing_ms":8.5}"#)
        let arrived = await waitUntil { client.latestFrameResult != nil }
        XCTAssertTrue(arrived)

        XCTAssertEqual(client.latestFrameResult?.sequence, 7)
        XCTAssertEqual(client.latestFrameResult?.meanIntensity ?? -1, 0.42, accuracy: 0.0001)
        XCTAssertEqual(client.latestFrameResult?.processingMs ?? -1, 8.5, accuracy: 0.0001)

        client.disconnect()
    }

    /// Captured for *every* reply, not only the one-in-twelve that gets logged.
    /// The decode used to sit inside the log gate; a surfaced value that
    /// updates at a twelfth of the reply rate would be stale on screen.
    func testEveryReplyUpdatesTheSurfacedResultNotJustLoggedOnes() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        // Two consecutive replies: the second is not on the logging stride, so
        // under the old decode-inside-the-log-gate arrangement it would be lost.
        server.send(text: #"{"type":"frame_result","seq":1,"mean_intensity":0.10,"processing_ms":1.0}"#)
        let first = await waitUntil { client.latestFrameResult?.sequence == 1 }
        XCTAssertTrue(first)

        server.send(text: #"{"type":"frame_result","seq":2,"mean_intensity":0.90,"processing_ms":1.0}"#)
        let second = await waitUntil { client.latestFrameResult?.sequence == 2 }
        XCTAssertTrue(second, "a reply between log lines was not surfaced")
        XCTAssertEqual(client.latestFrameResult?.meanIntensity ?? -1, 0.90, accuracy: 0.0001)

        client.disconnect()
    }

    /// Scoped to the stream bracket, like `frameResultCount` beside it: a
    /// reading from the previous bracket shown against a fresh one is a stale
    /// claim about the current session.
    func testTheSurfacedResultResetsWithTheStreamBracket() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.sendStreamStart()
        server.send(text: #"{"type":"frame_result","seq":1,"mean_intensity":0.5,"processing_ms":1.0}"#)
        let arrived = await waitUntil { client.latestFrameResult != nil }
        XCTAssertTrue(arrived)

        client.sendStreamStop()
        client.sendStreamStart()
        XCTAssertNil(client.latestFrameResult, "a new bracket must not inherit the previous one's reply")

        client.disconnect()
    }

    /// The declaration survives a reconnect and does NOT survive a retarget.
    ///
    /// Keeping it across a drop is deliberate — what a Tower can do belongs to
    /// its build, not to one socket, and clearing it would turn every dropped
    /// connection into "this will never work" when the truth is "cannot reach
    /// it". That reasoning stops at the endpoint: a *different* Tower's
    /// capabilities are not this one's. Left standing, the stale declaration
    /// races the new connection's own `cartridges` reply and can drive the
    /// first `result_subscribe` with the previous Tower's contract.
    func testTheDeclarationDoesNotFollowTheClientToADifferentTower() async throws {
        let first = try MockTowerServer()
        let firstPort = try await first.start()
        first.onText = { text in
            guard text.contains("\"ping\"") || text.contains("cartridges") else { return }
            if text.contains("\"ping\"") {
                first.send(text: #"{"type":"pong"}"#)
            } else {
                first.send(text: """
                    {"type":"cartridges",
                     "envelope_contract":"cartridge_results.envelope/2026-08-23",
                     "cartridges":[{"cartridge":"world_builder","result_type":"status",
                        "contract":"world_builder.status/2026-08-25","available":true,
                        "unavailable_reason":null,"snapshot_only":true}],
                     "not_offered":[]}
                    """)
            }
        }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: firstPort))
        let declared = await waitUntil { client.cartridgeDeclaration != nil }
        XCTAssertTrue(declared, "the first Tower never declared anything")

        // A drop, not a retarget: the declaration must survive this.
        first.stop()
        _ = await waitUntil { client.status != .online }
        XCTAssertNotNil(
            client.cartridgeDeclaration,
            "a dropped connection cleared what the Tower can do, which is a different claim"
        )

        // A different endpoint. The previous Tower's answer must not follow.
        let second = try MockTowerServer()
        let secondPort = try await second.start()
        defer { second.stop() }
        XCTAssertNotEqual(firstPort, secondPort, "the two mock servers shared a port")

        client.connect(to: url(port: secondPort))
        XCTAssertNil(
            client.cartridgeDeclaration,
            "the previous Tower's declaration followed the client to a different Tower"
        )

        client.disconnect()
    }

    /// A connect that never completes must terminate, not hang forever.
    ///
    /// `validateConnection` bounded the pong at 6 s and left the `send` beside
    /// it — which cannot return until TCP connect and the HTTP Upgrade have
    /// finished, so it *is* the connect leg — with no deadline at all. A peer
    /// that accepts the TCP connection and never upgrades parked the client in
    /// `.connecting` permanently: `fail(_:task:)` was never reached, so the
    /// reconnect schedule never advanced and the attempt budget was never
    /// spent. Nothing tried again, and nothing said so.
    ///
    /// Found while decomposing a real 9-second reconnect on a physical walk,
    /// ~8.5 s of which was spent in `.connecting`.
    ///
    /// The listener here is deliberately **plain TCP** — no
    /// `NWProtocolWebSocket` in its stack — so it completes the handshake's
    /// first half and then simply never speaks.
    func testAConnectThatNeverUpgradesFailsInsteadOfHangingForever() async throws {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        let listener = try NWListener(using: parameters)
        let queue = DispatchQueue(label: "SilentTCPListener")

        // Accepted and then ignored: started so the TCP handshake completes,
        // and never written to, so the upgrade cannot.
        let accepted = OSAllocatedUnfairLock(initialState: [NWConnection]())
        listener.newConnectionHandler = { connection in
            connection.start(queue: queue)
            accepted.withLock { $0.append(connection) }
        }

        let port: UInt16 = try await withCheckedThrowingContinuation { continuation in
            // Locked rather than a captured `var`: `stateUpdateHandler` runs on
            // the listener's queue, and a continuation resumed twice traps.
            let resumed = OSAllocatedUnfairLock(initialState: false)
            listener.stateUpdateHandler = { state in
                let claim = resumed.withLock { already -> Bool in
                    if already { return false }
                    already = true
                    return true
                }
                switch state {
                case .ready:
                    guard let port = listener.port else { return }
                    if claim { continuation.resume(returning: port.rawValue) }
                case .failed(let error):
                    if claim { continuation.resume(throwing: error) }
                default:
                    break
                }
            }
            listener.start(queue: queue)
        }
        defer { listener.cancel() }

        let client = TowerClient(metrics: SenderMetrics(), handshakeLegTimeout: 1)
        client.connect(to: url(port: port))

        let reachedConnecting = await waitUntil { client.status == .connecting }
        XCTAssertTrue(reachedConnecting, "the client never began connecting")

        let terminated = await waitUntil(timeout: 8) {
            if case .failed = client.status { return true }
            return false
        }
        XCTAssertTrue(
            terminated,
            "the client is still connecting to a peer that will never answer — the connect leg is unbounded"
        )

        client.disconnect()
    }

    /// The Tower's per-frame vocabulary is six keys, not three.
    ///
    /// `tower/tower/routes/ws.py:148-155` builds every `frame_result` with
    /// `seq`, `processing_ms`, **`result_value`, `result_label` and
    /// `stage_ms`** — all five unconditional — then adds `mean_intensity` and
    /// `metrics` only when the experiment produced them. `ExperimentResult`
    /// (`tower/tower/experiments/__init__.py:106-117`) is where those fields
    /// come from.
    ///
    /// This app decoded three of them and dropped the rest, and the doc
    /// comment on `TowerFrameResult` asserted that three *was* the whole
    /// vocabulary. `result_label` is the experiment's own headline answer, so
    /// what was being discarded was the only thing the Tower says about what
    /// it *concluded*, as opposed to how long it took.
    func testTheWholeFrameResultVocabularyIsDecodedNotJustTheTimings() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        // Shaped exactly as `ws.py` builds it, including the optional pair.
        server.send(text: #"{"type":"frame_result","seq":11,"processing_ms":8.5,"result_value":0.73,"result_label":"baseline","stage_ms":{"decode":2.0,"infer":6.5},"mean_intensity":0.42,"metrics":{"edges":1024.0}}"#)

        let arrived = await waitUntil { client.latestFrameResult?.sequence == 11 }
        XCTAssertTrue(arrived)

        let result = try XCTUnwrap(client.latestFrameResult)
        XCTAssertEqual(result.processingMs ?? -1, 8.5, accuracy: 0.0001)
        XCTAssertEqual(result.meanIntensity ?? -1, 0.42, accuracy: 0.0001)

        XCTAssertEqual(result.resultValue ?? -1, 0.73, accuracy: 0.0001,
                       "the experiment's headline number was dropped")
        XCTAssertEqual(result.resultLabel, "baseline",
                       "the experiment's own name for its answer was dropped")
        XCTAssertEqual(result.stageMs, ["decode": 2.0, "infer": 6.5],
                       "the per-stage timings were dropped")
        XCTAssertEqual(result.metrics, ["edges": 1024.0],
                       "the additive measurement channel was dropped")

        client.disconnect()
    }

    /// The two optional keys are genuinely optional, and their absence must
    /// not be read as zero.
    ///
    /// `ws.py:156-165` omits `mean_intensity` when the experiment reported
    /// none and omits `metrics` entirely when empty — "an experiment whose
    /// headline says everything does not pay for an empty object on every
    /// frame". Absent is not `0.0`.
    func testTheOptionalHalfOfAFrameResultIsAbsentRatherThanZero() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        server.send(text: #"{"type":"frame_result","seq":3,"processing_ms":1.0,"result_value":2.0,"result_label":"edges","stage_ms":{}}"#)
        let arrived = await waitUntil { client.latestFrameResult?.sequence == 3 }
        XCTAssertTrue(arrived)

        let result = try XCTUnwrap(client.latestFrameResult)
        XCTAssertNil(result.meanIntensity, "an omitted intensity is unknown, not dark")
        XCTAssertTrue(result.metrics.isEmpty)
        XCTAssertTrue(result.stageMs.isEmpty)
        XCTAssertEqual(result.resultLabel, "edges")

        client.disconnect()
    }

    /// A reply belongs to the socket that carried it.
    ///
    /// `teardownConnection` already clears everything else scoped to one
    /// connection — the send window, `lastSendFrameAt`, `isStreamingToTower` —
    /// and says of the last of those that leaving it set across a teardown
    /// "would be a lie". `latestFrameResult` was cleared only by
    /// `sendStreamStart`/`sendStreamStop`, so a socket that dropped mid-capture
    /// left the dead connection's reading on screen under the caption "latest
    /// Tower reply" (`HomeWorkspaceView.swift`) — for the whole outage, and
    /// permanently once the reconnect budget is spent.
    func testAFrameResultDoesNotOutliveTheSocketThatCarriedIt() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        // Mid-capture: a bracket is open, so nothing else would clear this.
        client.sendStreamStart()
        server.send(text: #"{"type":"frame_result","seq":5,"processing_ms":1.0,"result_value":1.0,"result_label":"baseline","stage_ms":{},"mean_intensity":0.5}"#)
        let arrived = await waitUntil { client.latestFrameResult != nil }
        XCTAssertTrue(arrived)

        // The socket drops under it. `autoReconnect` is off by default, so this
        // settles rather than racing a retry.
        server.stop()
        let dropped = await waitUntil { client.status != .online }
        XCTAssertTrue(dropped, "the client did not notice the socket close")

        XCTAssertNil(
            client.latestFrameResult,
            "a reading from a dead socket was still being offered as the latest Tower reply"
        )

        client.disconnect()
    }

    // MARK: - Cartridge integration: runtime ownership

    /// Building and discarding every cartridge view model must not disturb the
    /// runtime.
    ///
    /// ## What this proves, and — precisely — what it does not
    ///
    /// A workspace's `@StateObject` view model is destroyed when the cartridge
    /// changes. If one of them held the `TowerClient` or the `GlassesConnection`,
    /// destroying it could close the socket or stop the camera — and a stream
    /// bracket that closes mid-session silently discards every frame after it.
    ///
    /// So this constructs and releases all four view models repeatedly while a
    /// real socket is open with a live stream bracket, then asserts the
    /// connection is untouched and nothing new reached the wire. It would fail
    /// if any view model retained a runtime object, opened one, or sent
    /// anything.
    ///
    /// **It does not switch cartridges.** No workspace is installed, no view is
    /// mounted, and `ContentView` is not involved — this is a unit test of what
    /// the view models own, which is the half of a cartridge switch that can be
    /// tested without a view hierarchy. The other half (SwiftUI actually tearing
    /// a workspace down, and the camera surviving it) needs DAT and a device,
    /// and `ScriptedWearables` covers the DAT half separately (see `ConnectionLifetimeTests`). The structural argument stands
    /// in for it — no view model is handed a `GlassesConnection`, and
    /// `ContentView` passes one to World Builder only — and it is a
    /// Simulator/device check.
    func testDiscardingCartridgeViewModelsDoesNotDisturbALiveStream() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let metrics = SenderMetrics()
        let client = TowerClient(metrics: metrics)
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline)

        client.sendStreamStart()
        XCTAssertTrue(client.isStreamingToTower, "the test needs a live bracket to be meaningful")

        // Wait for the bracket to actually reach the server before snapshotting
        // the wire. Counting before it lands would make the later "nothing new
        // was sent" assertion race against this send rather than against the
        // view models.
        let bracketLanded = await waitUntil {
            recorder.all.compactMap(self.decode).contains { $0["type"] as? String == "stream_start" }
        }
        XCTAssertTrue(bracketLanded)
        let discovered = await waitForDiscovery(recorder)
        XCTAssertTrue(discovered, "the connection's own discovery never landed")

        let baseline = recorder.all.count

        // Three rounds of constructing and releasing every cartridge view
        // model — the object-lifetime half of what a cartridge switch does to
        // them.
        for _ in 0..<3 {
            _ = WorldBuilderViewModel(client: UnavailableWorldBuilderClient())
            _ = ExperimentalCVViewModel(client: UnavailableExperimentalCVClient())
            _ = DocumentMemoryViewModel(client: UnavailableDocumentMemoryClient())
            _ = SceneUnderstandingViewModel(client: UnavailableSceneUnderstandingClient())
        }

        XCTAssertEqual(client.status, .online, "a workspace teardown closed the Tower connection")
        XCTAssertTrue(
            client.isStreamingToTower,
            "a workspace teardown closed the stream bracket; every later frame would be dropped"
        )
        XCTAssertEqual(
            recorder.all.count,
            baseline,
            "a cartridge view model put something on the wire"
        )

        // The bracket still works after all that, which is the property a
        // silently-closed socket would break without changing `status`.
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 1)
        let frameArrived = await waitUntil {
            recorder.all.compactMap(self.decode).contains { $0["type"] as? String == "frame" }
        }
        XCTAssertTrue(frameArrived, "the sender stopped working after cartridge switching")

        client.sendStreamStop()
        client.disconnect()
    }

    /// A cartridge view model must not be able to reach the Tower at all.
    ///
    /// Constructing every one of them against a live, idle connection and then
    /// asserting the wire is silent is the observable form of "these hold no
    /// runtime references": a view model that opened its own socket, sent a
    /// module-selection message, or asked the Tower anything would show up here.
    ///
    /// `docs/08-IOS-CARTRIDGE-SHELL.md` forbids inventing a module-selection
    /// message. This is that prohibition as a test rather than as a promise.
    func testCartridgeViewModelsSendNothingToTheTower() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline, "expected .online, got \(client.status)")
        let discovered = await waitForDiscovery(recorder)
        XCTAssertTrue(discovered, "the connection's own discovery never landed")

        let baseline = recorder.all.count

        let world = WorldBuilderViewModel(client: UnavailableWorldBuilderClient())
        let lab = ExperimentalCVViewModel(client: UnavailableExperimentalCVClient())
        let memory = DocumentMemoryViewModel(client: UnavailableDocumentMemoryClient())
        let scene = SceneUnderstandingViewModel(client: UnavailableSceneUnderstandingClient())

        // Every request surface any of them exposes, exercised. Each must be
        // refused locally rather than reaching the socket.
        lab.run(CVExperiment(id: "anything", name: "Anything"))
        memory.queryText = "anything"
        memory.submitTypedQuery()
        memory.submit(.recent(limit: 3), origin: .externalIntent)

        // Give anything asynchronous a chance to have escaped to the wire.
        try? await Task.sleep(nanoseconds: 200_000_000)

        XCTAssertEqual(
            recorder.all.count,
            baseline,
            "a cartridge view model sent a message the Tower protocol does not contain"
        )
        XCTAssertFalse(client.isStreamingToTower, "a view model opened a stream bracket")
        XCTAssertEqual(client.status, .online)

        // And each of them still reports the truth rather than a state invented
        // from a request that went nowhere.
        XCTAssertEqual(world.phase(isTowerReachable: true), .unsupported)
        XCTAssertEqual(lab.phase(isTowerReachable: true), .unsupported)
        XCTAssertEqual(memory.phase(isTowerReachable: true), .unsupported)
        XCTAssertEqual(scene.phase(isTowerReachable: true), .unsupported)
        XCTAssertNotNil(lab.lastRequestFailure, "the refusal was swallowed instead of surfaced")
        XCTAssertNotNil(memory.lastRequestFailure)

        client.disconnect()
    }

    /// A connected Tower does not make any cartridge available.
    ///
    /// The counterpart to `testNoCartridgeIsAvailableWhetherOrNotTheTowerIsReachable`
    /// in `ProductShellTests`, asserted against a genuinely open socket rather
    /// than a boolean — because "reachable" being passed as a parameter is
    /// exactly the sort of thing that can be wired to the wrong value.
    func testAnOnlineTowerStillDeclaresNoCartridgeContracts() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        let becameOnline = await waitUntil { client.status == .online }
        XCTAssertTrue(becameOnline, "expected .online, got \(client.status)")

        let isReachable = client.status == .online
        XCTAssertTrue(isReachable)

        for cartridge in Cartridge.catalog {
            XCTAssertEqual(
                TowerCapabilities.availability(for: cartridge.id, isTowerReachable: isReachable),
                .noContract,
                "\(cartridge.name) became available merely because the socket opened"
            )
        }

        client.disconnect()
    }

    // MARK: - 20. The result channel

    private static let worldBuilderContract = "world_builder.status/2026-08-25"
    private static let experimentalCVContract = "experimental_cv.status/2026-08-27"

    /// The declaration a **live** Tower sends, byte for byte in shape.
    ///
    /// ## This fixture was wrong, and it was wrong in the direction that hides
    ///
    /// It declared one cartridge and put `document_memory` under `not_offered`
    /// with the reason *"no contract offered"* — which was true of the Tower it
    /// was written against and is false of the Tower that runs today. Captured
    /// from `GET /cartridges` on a running Tower, the declaration is **four
    /// cartridges, `not_offered: []`, and one `http_contracts` entry**. Every
    /// cartridge in that build with a wire contract now offers it; `not_offered`
    /// being empty is a *claim*, not an absence.
    ///
    /// A fixture that keeps a cartridge in `not_offered` is a test asserting
    /// that the app behaves correctly against a Tower nobody runs, which is the
    /// most comfortable kind of green there is.
    private var declarationJSON: String {
        """
        {"type":"cartridges",
         "envelope_contract":"cartridge_results.envelope/2026-08-23",
         "cartridges":[{"cartridge":"world_builder","result_type":"status",
                        "contract":"\(Self.worldBuilderContract)",
                        "available":true,"unavailable_reason":null,
                        "snapshot_only":true},
                       {"cartridge":"scene_understanding","result_type":"live",
                        "contract":"scene_understanding.live/2026-08-27",
                        "available":true,"unavailable_reason":null,
                        "snapshot_only":true},
                       {"cartridge":"document_memory","result_type":"status",
                        "contract":"document_memory.status/2026-08-27",
                        "available":true,"unavailable_reason":null,
                        "snapshot_only":true},
                       {"cartridge":"experimental_cv","result_type":"status",
                        "contract":"\(Self.experimentalCVContract)",
                        "available":true,"unavailable_reason":null,
                        "snapshot_only":true}],
         "not_offered":[],
         "http_contracts":[{"cartridge":"document_memory",
                            "contract":"document_memory.library/2026-08-27",
                            "entry_route":"/documents","available":true,
                            "unavailable_reason":null,
                            "why_not_a_subscription":"document text is bulk and is the most sensitive data this platform holds"}]}
        """
    }

    /// A `cartridge_result` carrying a live world, shaped exactly as the Tower's
    /// `_attach_ios_projection` builds it.
    private func resultJSON(
        seq: Int,
        revision: String,
        revisionChanged: Bool = true,
        modelState: String = "receiving",
        keyframes: Int = 4
    ) -> String {
        """
        {"type":"cartridge_result",
         "envelope_contract":"cartridge_results.envelope/2026-08-23",
         "subscription_id":"sub-1","cartridge":"world_builder","result_type":"status",
         "contract":"\(Self.worldBuilderContract)",
         "seq":\(seq),"revision":"\(revision)","revision_changed":\(revisionChanged),
         "coalesced":0,"cursor_status":null,"snapshot":true,
         "tower_sent_at":1787463092.958,"time_basis":"tower-receipt",
         "payload":{"model_state":"\(modelState)","model_state_reason":null,
           "world_snapshot":{"name":"Probe Room","world_id":"be5076",
             "keyframe_count":\(keyframes),"revision":"\(revision)",
             "tracking":"good","scale":"relative","mapping_seconds":0.0789,
             "calibration":"calibrated",
             "geometry":{"representation":"sparse point cloud","element_count":1360,
                         "is_incremental":false},
             "trajectory":{"pose_count":\(keyframes),"path_length":2.853,
                           "path_length_unit":"world units","scale":"relative"},
             "persistence":{"state":"saved","revision":"67ccaee7"}}}}
        """
    }

    /// Collects result-channel events off the client's publisher.
    private func collectEvents(_ client: TowerClient) -> EventRecorder {
        let recorder = EventRecorder()
        recorder.cancellable = client.cartridgeResults.sink { recorder.record($0) }
        return recorder
    }

    /// Discovery must follow the pong, never precede it.
    ///
    /// The Tower never speaks first, so nothing can arrive early — but a client
    /// that asked for capabilities before validating would read its own reply
    /// into the handshake, which is the one ordering the contract calls out.
    func testCapabilityDiscoveryIsSentOnceAndOnlyAfterThePong() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }
        let discovered = await waitForDiscovery(recorder)
        XCTAssertTrue(discovered)

        let types = recorder.all.compactMap(decode).compactMap { $0["type"] as? String }
        XCTAssertEqual(types.first, "ping", "something reached the Tower before the handshake")
        XCTAssertEqual(
            types.filter { $0 == "cartridges" }.count,
            1,
            "capability discovery was sent more than once for one connection"
        )

        client.disconnect()
    }

    /// The declaration is cached, and it is cached as what the Tower said —
    /// not reduced to a boolean on the way in.
    func testTheCartridgeDeclarationIsDecodedAndCached() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        server.send(text: declarationJSON)
        await expect { client.cartridgeDeclaration != nil }

        let offer = client.cartridgeDeclaration?.offer(forTowerCartridge: "world_builder")
        XCTAssertEqual(offer?.contract, Self.worldBuilderContract)
        XCTAssertEqual(offer?.resultType, "status")
        XCTAssertEqual(offer?.available, true)
        XCTAssertEqual(offer?.snapshotOnly, true)
        XCTAssertNil(offer?.unavailableReason)
        XCTAssertEqual(
            client.cartridgeDeclaration?.envelopeContract,
            "cartridge_results.envelope/2026-08-23"
        )
        // The four cartridges a live Tower declares, and the one HTTP contract
        // beside them.
        //
        // This assertion used to read `XCTAssertNil(offer(forTowerCartridge:
        // "document_memory"))`, on the grounds that `not_offered` is for
        // operators and presence there must never be read as an offer. The
        // reasoning is still right and the assertion had stopped being: a live
        // Tower offers Document Memory outright, so the nil was asserting that
        // this client drops an offer it is now sent. The reasoning is preserved
        // where it belongs — `TowerCartridgeDeclaration` still does not decode
        // `not_offered` at all — and what is checked here is what arrives.
        XCTAssertEqual(client.cartridgeDeclaration?.offers.count, 4)
        XCTAssertNotNil(client.cartridgeDeclaration?.offer(forTowerCartridge: "document_memory"))
        XCTAssertNotNil(client.cartridgeDeclaration?.offer(forTowerCartridge: "scene_understanding"))
        let cvOffer = client.cartridgeDeclaration?.offer(forTowerCartridge: "experimental_cv")
        XCTAssertEqual(cvOffer?.contract, Self.experimentalCVContract)
        XCTAssertEqual(cvOffer?.resultType, "status")
        XCTAssertEqual(cvOffer?.available, true)
        // The fixture also carries the live declaration's single
        // `http_contracts` entry. It is deliberately **not** asserted on here:
        // Document Memory's library route is that cartridge's contract, and
        // what this test owns is that a subscription offer decodes correctly
        // beside it. The `offers.count` above is the assertion that the extra
        // top-level key disturbed nothing.
        //
        // A cartridge the Tower says nothing about is still absent, which is
        // the half of the old assertion that was about this client rather than
        // about that Tower.
        XCTAssertNil(client.cartridgeDeclaration?.offer(forTowerCartridge: "object_memory"))

        client.disconnect()
    }

    /// What the Tower can do is a property of the Tower's build, not of this
    /// socket. Clearing the cache on a drop would turn every blip into "this
    /// will never work" when the truthful reading is "not reachable".
    func testTheDeclarationSurvivesADisconnect() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }
        server.send(text: declarationJSON)
        await expect { client.cartridgeDeclaration != nil }

        client.disconnect()
        XCTAssertNotNil(client.cartridgeDeclaration)
        XCTAssertEqual(
            TowerCapabilities.availability(
                for: "world-build",
                declaredBy: client.cartridgeDeclaration,
                isTowerReachable: false
            ),
            .towerUnreachable
        )
    }

    /// The envelope decodes, and the ordering between the ack and the snapshot
    /// that follows it is preserved — which is why they share one stream.
    func testTheResultChannelDeliversAckThenSnapshotInOrder() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        let events = collectEvents(client)
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        server.send(text: """
            {"type":"result_subscribed","envelope_contract":"cartridge_results.envelope/2026-08-23",
             "subscription_id":"sub-1","cartridge":"world_builder","result_type":"status",
             "contract":"\(Self.worldBuilderContract)","snapshot_only":true,
             "world_id":null,"session_id":null,"cursor_status":"absent"}
            """)
        server.send(text: resultJSON(seq: 1, revision: "e252f739c1cdedab"))

        await expect { events.all.count >= 2 }

        guard case .subscribed(let ack) = events.all[0] else {
            return XCTFail("expected the ack first, got \(events.all[0])")
        }
        XCTAssertEqual(ack.subscriptionID, "sub-1")
        XCTAssertEqual(ack.cursorStatus, "absent")

        guard case .result(let envelope) = events.all[1] else {
            return XCTFail("expected the snapshot second, got \(events.all[1])")
        }
        XCTAssertEqual(envelope.sequence, 1)
        XCTAssertEqual(envelope.revision, "e252f739c1cdedab")
        XCTAssertTrue(envelope.revisionChanged)
        XCTAssertEqual(envelope.coalesced, 0)
        XCTAssertTrue(envelope.isSnapshot)
        XCTAssertEqual(envelope.payload["model_state"] as? String, "receiving")

        client.disconnect()
    }

    /// Every `result_error` is non-fatal on this socket, including the two the
    /// Tower sends unsolicited — which carry no `envelope_contract` at all, and
    /// would fail a decoder that required one.
    func testEveryResultErrorLeavesTheFramePathWorking() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        let events = collectEvents(client)
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        // Solicited, with reason-specific extras.
        server.send(text: """
            {"type":"result_error","envelope_contract":"cartridge_results.envelope/2026-08-23",
             "reason":"unknown_cartridge","message":"no such cartridge",
             "cartridge":"nope","result_type":"status","offered":["world_builder"]}
            """)
        // Unsolicited, and deliberately without `envelope_contract`.
        server.send(text: """
            {"type":"result_error","reason":"consumer_too_slow","subscription_id":"sub-1",
             "cartridge":"world_builder","result_type":"status","message":"closed"}
            """)
        // Two keys and nothing else.
        server.send(text: #"{"type":"result_unsubscribed","subscription_id":"sub-1"}"#)
        // The Tower saying it does not implement something we sent.
        server.send(text: """
            {"type":"protocol_error","reason":"unknown_message_type",
             "message_type":42,"message":"this Tower does not implement that message type"}
            """)

        await expect { events.all.count >= 3 }

        let errors = events.all.compactMap { event -> CartridgeResultError? in
            if case .failed(let error) = event { return error }
            return nil
        }
        XCTAssertEqual(errors.map(\.reason), ["unknown_cartridge", "consumer_too_slow"])
        XCTAssertFalse(errors[0].closesSubscription)
        XCTAssertTrue(errors[1].closesSubscription)
        XCTAssertEqual(errors[1].subscriptionID, "sub-1")

        // The whole point: the connection and the frame path are untouched.
        server.send(text: #"{"type":"frame_result","seq":7,"mean_intensity":0.3,"processing_ms":3.2}"#)
        await expect { client.frameResultCount >= 1 }
        XCTAssertEqual(client.status, .online)

        client.disconnect()
    }

    /// `frame_result` must be field-for-field what it always was while results
    /// share the socket. One TCP stream, two message families, demultiplexed by
    /// `type` and nothing else.
    func testFrameResultIsUnaffectedByResultTrafficOnTheSameSocket() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        server.send(text: declarationJSON)
        server.send(text: #"{"type":"frame_result","seq":1,"mean_intensity":0.42,"processing_ms":8.5}"#)
        server.send(text: resultJSON(seq: 1, revision: "aaaa"))
        server.send(text: #"{"type":"frame_result","seq":2,"mean_intensity":0.51,"processing_ms":3.1}"#)

        await expect { client.frameResultCount >= 2 }
        XCTAssertEqual(client.latestFrameResult?.sequence, 2)
        XCTAssertEqual(client.latestFrameResult?.meanIntensity, 0.51)
        XCTAssertEqual(client.latestFrameResult?.processingMs, 3.1)
        XCTAssertEqual(client.status, .online)

        client.disconnect()
    }

    /// A malformed result message is dropped, not escalated. It must not be
    /// published as an event and must not disturb anything else.
    func testAnUndecodableResultMessageIsDroppedRatherThanPublished() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        let events = collectEvents(client)
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        // No `payload`, so there is no result to publish.
        server.send(text: #"{"type":"cartridge_result","cartridge":"world_builder","result_type":"status"}"#)
        // No `reason`, so there is no error to publish.
        server.send(text: #"{"type":"result_error","message":"?"}"#)
        server.send(text: #"{"type":"frame_result","seq":1,"mean_intensity":0.1,"processing_ms":1.0}"#)

        await expect { client.frameResultCount >= 1 }
        XCTAssertTrue(events.all.isEmpty, "an undecodable message was published as an event")
        XCTAssertEqual(client.status, .online)

        client.disconnect()
    }

    /// A subscribe request is exactly the four fields the Tower validates, and
    /// the contract is included so a disagreement is refused rather than
    /// misinterpreted.
    func testSubscribeSendsTheContractSoAMismatchIsRefused() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        client.subscribeToResults(
            cartridge: "world_builder",
            resultType: "status",
            contract: Self.worldBuilderContract
        )
        let landed = await waitUntil {
            recorder.all.compactMap(self.decode).contains { $0["type"] as? String == "result_subscribe" }
        }
        XCTAssertTrue(landed)

        let request = recorder.all.compactMap(decode)
            .first { $0["type"] as? String == "result_subscribe" }
        XCTAssertEqual(request?["cartridge"] as? String, "world_builder")
        XCTAssertEqual(request?["result_type"] as? String, "status")
        XCTAssertEqual(request?["contract"] as? String, Self.worldBuilderContract)

        client.disconnect()
    }

    /// A result-channel send that fails must not fail the connection.
    ///
    /// The contract promises the result channel cannot affect the frame path,
    /// and tearing down the socket the camera is streaming over because a
    /// capability query did not land would break exactly that promise. The
    /// lifecycle markers deliberately do the opposite, because losing one
    /// corrupts a frame bracket.
    func testAFailedResultSendIsNotEscalatedToTheConnection() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        // The server goes away without a close frame, so the next send fails
        // at the socket rather than being refused by the `status` guard.
        server.dropConnection()
        server.stop()
        client.subscribeToResults(
            cartridge: "world_builder",
            resultType: "status",
            contract: Self.worldBuilderContract
        )

        // The receive loop will still notice the dead link on its own terms —
        // what must not happen is this client reaching `.failed` *because of
        // the subscribe*. Give the send's completion time to run and assert the
        // failure, if any, did not come from here.
        try? await Task.sleep(nanoseconds: 150_000_000)
        if case .failed(let message) = client.status {
            XCTAssertFalse(
                message.contains("result_subscribe"),
                "a result-channel send failure was escalated to the connection"
            )
        }

        client.disconnect()
    }

    // MARK: - 21. The Tower's dataset-recorder state, read from /health

    // The Tower has reported `capture.armed` / `capture.recording` /
    // `frames_written` on `GET /health` since 2026-08-22 and this app read
    // none of it, so "is the Tower writing my frames to disk right now?" was
    // unanswerable from the phone. It matters more than completeness: while
    // the recorder is armed the Tower fsyncs every received frame **unredacted**
    // (`tower/tower/capture.py` declares `retains_raw_imagery: true`,
    // `redaction: "none"`).
    //
    // These are decode tests against real payload shapes rather than tests
    // against a running Tower, matching how the project pins contract
    // fixtures elsewhere. The thing being defended is not "does HTTP work" —
    // it is that a field the Tower did not send never becomes a confident
    // `false` or `0` on a screen a person reads to answer a privacy question.

    /// The exact body a live Tower returned this session, whitespace aside.
    private static let liveHealthPayload = """
        {"status":"ok","service":"glasses-tower","version":"0.1.0",
         "module_state":"active","module_id":"experimental-cv",
         "capture":{"armed":true,"recording":false,
                    "capture_id":"ab10cb203cf048e58cf2f79a120a54a4",
                    "frames_written":1343,"bytes_written":27303448},
         "capture_workers":{"enabled":true,"workers":[]}}
        """

    private func healthData(_ json: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: json)
    }

    func testTheLiveHealthPayloadDecodesFieldByField() throws {
        let health = try TowerHealthDecoder.health(from: Data(Self.liveHealthPayload.utf8))

        XCTAssertEqual(health.status, "ok")
        XCTAssertEqual(health.service, "glasses-tower")
        XCTAssertEqual(health.version, "0.1.0")
        XCTAssertEqual(health.moduleState, "active")
        XCTAssertEqual(health.moduleID, "experimental-cv")

        guard case .present(let capture) = health.capture else {
            return XCTFail("the live payload carries a capture object: \(health.capture)")
        }
        XCTAssertEqual(capture.armed, true)
        XCTAssertEqual(capture.recording, false)
        XCTAssertEqual(capture.captureID, .present("ab10cb203cf048e58cf2f79a120a54a4"))
        XCTAssertEqual(capture.framesWritten, 1343)
        XCTAssertEqual(capture.bytesWritten, 27303448)
        XCTAssertNil(capture.error)

        guard case .present(let workers) = health.captureWorkers else {
            return XCTFail("the live payload carries a worker object: \(health.captureWorkers)")
        }
        XCTAssertEqual(workers.enabled, true)
        // A real, sent, empty array — so a real `0`, which is exactly the
        // value the rest of these tests refuse to invent.
        XCTAssertEqual(workers.workerCount, 0)
        XCTAssertNil(workers.error)
    }

    /// An older Tower, or one whose health route grows a different shape,
    /// simply does not mention capture. That must not read as "not recording".
    func testCaptureKeyAbsentEntirelyIsUnreportedRatherThanFalseOrZero() throws {
        let health = try TowerHealthDecoder.health(
            from: healthData([
                "status": "ok", "service": "glasses-tower", "version": "0.1.0",
                "module_state": "active", "module_id": "experimental-cv",
            ])
        )

        XCTAssertEqual(health.capture, .unreported)
        XCTAssertEqual(health.captureWorkers, .unreported)
        XCTAssertNil(health.capture.value?.armed)
        XCTAssertNil(health.capture.value?.recording)
        XCTAssertNil(health.capture.value?.framesWritten)
    }

    /// `capture: null` is the Tower saying something specific — no recorder is
    /// registered at all — and it is not the same statement as saying nothing.
    /// `tower/tower/routes/health.py` documents the difference; collapsing the
    /// two would make "we are definitely not recording" indistinguishable from
    /// "this build never told us".
    func testAnExplicitNullCaptureIsAnAnswerAndNotSilence() throws {
        let health = try TowerHealthDecoder.health(
            from: healthData([
                "status": "ok", "capture": NSNull(), "capture_workers": NSNull(),
            ])
        )

        XCTAssertEqual(health.capture, .absent)
        XCTAssertEqual(health.captureWorkers, .absent)
        XCTAssertNotEqual(health.capture, .unreported)
    }

    /// The Tower's own error path sends `{"armed": true, "error": "unavailable"}`
    /// — armed, and everything else unknown. Nothing there may become a zero.
    func testFramesWrittenAbsentIsNilAndNotZero() throws {
        let health = try TowerHealthDecoder.health(
            from: healthData([
                "status": "ok",
                "capture": ["armed": true, "error": "unavailable"],
            ])
        )

        guard case .present(let capture) = health.capture else {
            return XCTFail("an object arrived, so it is present: \(health.capture)")
        }
        XCTAssertEqual(capture.armed, true)
        XCTAssertEqual(capture.error, "unavailable")
        XCTAssertNil(capture.recording, "an unsent `recording` must not read as not-recording")
        XCTAssertNil(capture.framesWritten, "an unsent count must not read as zero frames written")
        XCTAssertNil(capture.bytesWritten)
        // Not `.absent`: the Tower's error branch omits the key entirely, which
        // is silence about the id rather than a statement that no recording is
        // open.
        XCTAssertEqual(capture.captureID, .unreported)
    }

    /// `capture_id: null` is a positive statement, not a withheld field.
    ///
    /// `tower/tower/routes/health.py` always emits the key whenever it emits a
    /// `capture` object at all, and sends `null` for it in exactly one
    /// situation: a recorder is registered and has not opened a recording yet.
    /// The Tower's `.error` branch is the only one that omits the key, and that
    /// omission means something else entirely — the Tower could not read its
    /// own recorder.
    ///
    /// Reading both as one Swift `nil` reproduces the `unreported`-vs-`absent`
    /// conflation `TowerReported` exists to prevent, one level down at the
    /// field, and puts "The Tower did not say" on screen for an answer the
    /// Tower gave clearly.
    func testANullCaptureIDIsNoRecordingYetRatherThanASilentTower() throws {
        let notOpened = try TowerHealthDecoder.health(
            from: healthData([
                "capture": [
                    "armed": true, "recording": false, "capture_id": NSNull(),
                    "frames_written": 0, "bytes_written": 0,
                ]
            ])
        )
        let unsaid = try TowerHealthDecoder.health(
            from: healthData(["capture": ["armed": true, "error": "unavailable"]])
        )
        let opened = try TowerHealthDecoder.health(
            from: healthData([
                "capture": ["armed": true, "recording": true, "capture_id": "ab10cb20"]
            ])
        )

        XCTAssertNotEqual(
            notOpened.capture.value?.captureID,
            unsaid.capture.value?.captureID,
            "a recorder that has opened no recording reads the same as a Tower that withheld the field"
        )
        XCTAssertNotEqual(opened.capture.value?.captureID, notOpened.capture.value?.captureID)

        XCTAssertEqual(opened.capture.value?.captureID, .present("ab10cb20"))
        XCTAssertEqual(notOpened.capture.value?.captureID, .absent)
        XCTAssertEqual(unsaid.capture.value?.captureID, .unreported)

        // The two counts beside it are deliberately *not* changed to match.
        // The Tower always sends them on success and sends `0` when there is no
        // recording, so for them a single `nil` carries a single meaning.
        XCTAssertEqual(notOpened.capture.value?.framesWritten, 0)
        XCTAssertEqual(notOpened.capture.value?.bytesWritten, 0)
        XCTAssertNil(unsaid.capture.value?.framesWritten)
    }

    /// A `capture_id` that is neither a string nor a null. The Tower spoke and
    /// this app could not read it, which is a third thing again.
    func testACaptureIDOfTheWrongShapeIsUnreadable() throws {
        let wrongShape = try TowerHealthDecoder.health(
            from: healthData(["capture": ["armed": true, "capture_id": 7]])
        )
        let unsaid = try TowerHealthDecoder.health(
            from: healthData(["capture": ["armed": true]])
        )
        XCTAssertNotEqual(
            wrongShape.capture.value?.captureID,
            unsaid.capture.value?.captureID,
            "an unreadable answer reads the same as no answer"
        )
        XCTAssertEqual(wrongShape.capture.value?.captureID, .unreadable)
    }

    /// The one distinction the whole feature exists for.
    func testArmedAndIdleIsDistinguishableFromArmedAndRecording() throws {
        let idle = try TowerHealthDecoder.health(
            from: healthData(["capture": ["armed": true, "recording": false, "frames_written": 0]])
        )
        let recording = try TowerHealthDecoder.health(
            from: healthData(["capture": ["armed": true, "recording": true, "frames_written": 12]])
        )

        XCTAssertEqual(idle.capture.value?.armed, true)
        XCTAssertEqual(idle.capture.value?.recording, false)
        XCTAssertEqual(recording.capture.value?.recording, true)
        XCTAssertNotEqual(idle.capture, recording.capture)
        // A sent zero is a fact, and stays one.
        XCTAssertEqual(idle.capture.value?.framesWritten, 0)
    }

    /// The fourth case, and the only one of `TowerReported`'s four with no test
    /// until now — which mattered because it is also the only one that renders
    /// its own distinct sentence ("Answered in a shape this app cannot read").
    ///
    /// A `capture` that is neither an object nor a null is the Tower speaking
    /// and this app failing to understand. That is not silence (`.unreported`)
    /// and it is not "there is no recorder" (`.absent`), and a screen answering
    /// a privacy question must not merge it into either.
    func testACaptureOfTheWrongShapeIsUnreadableRatherThanSilenceOrAbsence() throws {
        let health = try TowerHealthDecoder.health(
            from: Data(#"{"capture": "unavailable", "capture_workers": 7}"#.utf8)
        )

        XCTAssertEqual(health.capture, .unreadable)
        XCTAssertEqual(health.captureWorkers, .unreadable)
        XCTAssertNotEqual(health.capture, .unreported, "an unreadable answer became silence")
        XCTAssertNotEqual(health.capture, .absent, "an unreadable answer became a registered nothing")
        XCTAssertNil(health.capture.value, "an unreadable answer yielded a value to draw")
    }

    func testGarbageIsAFailureRatherThanASilentlyEmptySuccess() {
        XCTAssertThrowsError(try TowerHealthDecoder.health(from: Data("not json".utf8))) { error in
            XCTAssertEqual(error as? TowerHealthFetchError, .undecodable)
        }
        XCTAssertThrowsError(try TowerHealthDecoder.health(from: Data())) { error in
            XCTAssertEqual(error as? TowerHealthFetchError, .undecodable)
        }
    }

    /// Well-formed JSON that is not an object at all. It parses, so the
    /// failure has to be the decoder's own decision rather than the parser's.
    func testATopLevelArrayIsAFailureRatherThanAnEmptyHealth() {
        XCTAssertThrowsError(try TowerHealthDecoder.health(from: Data("[1,2,3]".utf8))) { error in
            XCTAssertEqual(error as? TowerHealthFetchError, .undecodable)
        }
    }

    // MARK: Transport, and the four states a screen has to be able to draw

    private func stubbedHealthClient(status: Int = 200, body: Data) -> TowerHealthHTTPClient {
        stubbedHealthClient { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil
            )!
            return (response, body)
        }
    }

    /// The same client, over a stub that never produces a response at all —
    /// the shape a Tower that is not there has. Separate from the one above
    /// because a thrown `URLError` and a returned `HTTPURLResponse` are the two
    /// different things `TowerHealthFetchError` exists to keep apart, and one
    /// helper that did both would take an argument nobody could read.
    private func stubbedHealthClient(
        failingWith error: URLError
    ) -> TowerHealthHTTPClient {
        stubbedHealthClient { _ in throw error }
    }

    private func stubbedHealthClient(
        _ handler: @escaping (URLRequest) throws -> (HTTPURLResponse, Data)
    ) -> TowerHealthHTTPClient {
        ObjectMemoryStubProtocol.requestedURLs = []
        ObjectMemoryStubProtocol.requestedRequests = []
        ObjectMemoryStubProtocol.handler = handler
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ObjectMemoryStubProtocol.self]
        return TowerHealthHTTPClient(
            baseURL: URL(string: "http://tower.test")!,
            session: URLSession(configuration: configuration)
        )
    }

    /// Rule 15. A read with no ceiling is a spinner nobody can end.
    ///
    /// Asserted against **the request that actually ran**, not against the
    /// client's own `timeout` property. Reading the property back only proves
    /// the literal it was initialised with is still there; the deadline is only
    /// real if it reaches the `URLRequest`, and dropping `timeoutInterval:` from
    /// `health()` would leave the call inheriting `URLSession`'s 60-second
    /// default with the property still reading 10.
    func testTheHealthReadIsBounded() async {
        defer { ObjectMemoryStubProtocol.handler = nil }
        let client = TowerClient()
        client.healthClient = stubbedHealthClient(body: Data(Self.liveHealthPayload.utf8))

        client.refreshHealth()
        await expect("the fetch never settled: \(client.healthState)") {
            if case .fetched = client.healthState { return true }
            return false
        }

        guard let request = ObjectMemoryStubProtocol.requestedRequests.first else {
            return XCTFail("no request reached the stub")
        }
        XCTAssertEqual(
            request.timeoutInterval, 10,
            "the health read ran without the deadline the client claims to set"
        )
        // Same reasoning, same request: a health answer served out of a URL
        // cache would report a recorder as idle after it had started writing.
        XCTAssertEqual(request.cachePolicy, .reloadIgnoringLocalCacheData)
        // And the value the client is configured with, so the two cannot drift
        // apart silently. Matches `ObjectMemoryHTTPClient.timeout` deliberately.
        XCTAssertEqual(TowerHealthHTTPClient().timeout, 10)
    }

    func testNothingIsFetchedUntilSomebodyAsks() {
        XCTAssertEqual(TowerClient().healthState, .notFetched)
    }

    func testAnAnsweredFetchSettlesIntoTheFetchedState() async {
        defer { ObjectMemoryStubProtocol.handler = nil }
        let client = TowerClient()
        client.healthClient = stubbedHealthClient(body: Data(Self.liveHealthPayload.utf8))

        client.refreshHealth()
        await expect("the fetch never settled: \(client.healthState)") {
            if case .fetched = client.healthState { return true }
            return false
        }

        guard case .fetched(let health, _) = client.healthState else { return }
        XCTAssertEqual(health.capture.value?.recording, false)
        XCTAssertEqual(ObjectMemoryStubProtocol.requestedURLs.first?.path, "/health")
    }

    /// Failure is a state. It must be reachable, and it must not look like
    /// "nobody has asked yet".
    ///
    /// The body is the **live payload**, which decodes perfectly. That is the
    /// whole point: an earlier version of this test sent `"nope"`, which is not
    /// JSON, so the decoder threw `.undecodable` on its own and the test passed
    /// with the HTTP status guard deleted. A Tower that refuses the question is
    /// not a Tower that answered it, and only the status guard in
    /// `TowerHealthHTTPClient.health()` can tell the difference here — without
    /// it this settles into `.fetched` and the screen reports a recorder state
    /// read out of a 503.
    func testARefusedFetchBecomesAVisibleFailureAndNotSilence() async {
        defer { ObjectMemoryStubProtocol.handler = nil }
        let client = TowerClient()
        client.healthClient = stubbedHealthClient(
            status: 503, body: Data(Self.liveHealthPayload.utf8)
        )

        client.refreshHealth()
        await expect("a refusal never surfaced: \(client.healthState)") {
            if case .failed = client.healthState { return true }
            return false
        }
        XCTAssertNotEqual(client.healthState, .notFetched)

        guard case .failed(let error, _) = client.healthState else { return }
        // `.transport`, not `.undecodable`: the answer was readable and the
        // Tower declined to give one. Naming the wrong case here would send a
        // reader looking for a decoding bug that does not exist.
        guard case .transport(let description) = error else {
            return XCTFail("a refusal was reported as a disagreement about the answer: \(error)")
        }
        XCTAssertTrue(
            description.contains("503"),
            "the refusal did not say what the Tower actually answered: \(description)"
        )
    }

    /// The other half of the split, and the one with no test at all until now:
    /// a request that never completes.
    ///
    /// `.transport` is the only case that is evidence about the network, and
    /// nothing anywhere produced it — every failing path in this file went
    /// through the decoder. A Tower that is switched off is the ordinary
    /// failure this screen exists to name, so it gets a test that reaches it
    /// the way it is actually reached: no response, an error from the URL
    /// loading system.
    func testATowerThatNeverAnswersIsATransportFailureAndNotADecodeProblem() async {
        defer { ObjectMemoryStubProtocol.handler = nil }
        let client = TowerClient()
        client.healthClient = stubbedHealthClient(failingWith: URLError(.cannotConnectToHost))

        client.refreshHealth()
        await expect("an unreachable Tower never surfaced: \(client.healthState)") {
            if case .failed = client.healthState { return true }
            return false
        }

        guard case .failed(let error, _) = client.healthState else { return }
        guard case .transport(let description) = error else {
            return XCTFail("an unreachable Tower was reported as an unreadable answer: \(error)")
        }
        XCTAssertFalse(
            description.isEmpty,
            "the failure state carried no reason, which is the silence it exists to prevent"
        )
    }

    /// A body that arrived and could not be read is a disagreement about the
    /// answer, not a failure to get one — the same split
    /// `ObjectMemoryHTTPClient` keeps, and for the same reason.
    func testAnUnreadableAnswerIsUndecodableRatherThanTransport() async {
        defer { ObjectMemoryStubProtocol.handler = nil }
        let client = TowerClient()
        client.healthClient = stubbedHealthClient(body: Data("<html>hi</html>".utf8))

        client.refreshHealth()
        await expect("the unreadable answer never settled: \(client.healthState)") {
            if case .failed = client.healthState { return true }
            return false
        }
        guard case .failed(let error, _) = client.healthState else { return }
        XCTAssertEqual(error, .undecodable)
    }

    // MARK: - 22. The CV Lab's control plane, and the staleness gate
    //
    // Every fixture below is a message captured from a **live Tower** at
    // 127.0.0.1:8765 over a raw WebSocket — the replies to `cv_lab_status`,
    // `cv_lab_start`, `cv_lab_pause`, `cv_lab_resume` and `cv_lab_stop`, the
    // refusals for an unknown experiment and a stale run, and the
    // `frame_result` / `frame_error` a real frame produced. Nothing here is a
    // guess at a shape.

    /// A `cv_lab_status` reply, trimmed to the keys this client reads.
    ///
    /// The catalog and the run block are decoded by the cartridge, not here —
    /// this file passes the document through undecoded — so the fixture carries
    /// enough of them to prove the pass-through and no more.
    private func cvLabStatusJSON(
        runID: String?,
        state: String = "running",
        acceptedCommand: String? = nil,
        requestID: String? = nil
    ) -> String {
        let run = runID.map { "\"\($0)\"" } ?? "null"
        let accepted = acceptedCommand.map { ",\"accepted_command\":\"\($0)\"" } ?? ""
        let request = requestID.map { ",\"request_id\":\"\($0)\"" } ?? ""
        return """
            {"type":"cv_lab_status",
             "control_contract":"experimental_cv.control/2026-08-27",
             "contract":"experimental_cv.status/2026-08-27"\(request)\(accepted),
             "status":{"contract":"experimental_cv.status/2026-08-27",
               "tower_instance_id":"f863dcc35bce",
               "lifecycle":{"state":"\(state)","reason":null,
                            "since":1787833032.19,"run_id":\(run)},
               "available":[],"selected":"edge_detection",
               "source":{"clients_connected":1,"receiving_frames":true,
                         "last_frame_at":1787833032.32,"frames_offered_total":2,
                         "frames_rejected_before_lab":0,"idle_after_s":5.0}}}
            """
    }

    /// A `frame_result` with the `cv_lab` block a Tower running a Lab attaches.
    private func frameResultJSON(seq: Int, runID: String, resultSeq: Int) -> String {
        """
        {"type":"frame_result","seq":\(seq),"processing_ms":0.33,
         "result_value":0.107,"result_label":"edge_density",
         "stage_ms":{"decode":0.17,"blur":0.04,"canny":0.11,"summarize":0.006},
         "cv_lab":{"contract":"experimental_cv.frame_result/2026-08-27",
           "tower_instance_id":"f863dcc35bce","run_id":"\(runID)",
           "result_seq":\(resultSeq),"experiment_id":"edge_detection",
           "experiment_name":"Edge detection","provenance":"measured",
           "backend":"opencv","device":null,"device_requested":"auto",
           "result_label":"edge_density","processing_ms":0.33,
           "tower_received_at":1787833032.32,"time_basis":"tower-receipt"}}
        """
    }

    private func collectCVLabEvents(_ client: TowerClient) -> CVLabEventRecorder {
        let recorder = CVLabEventRecorder()
        recorder.cancellable = client.cvLabEvents.sink { recorder.record($0) }
        return recorder
    }

    /// The five commands, on the socket, shaped as the Tower parses them.
    ///
    /// **There is no HTTP surface for any of these**, so this is the only place
    /// they can be got wrong. `experiment_id` on a start and `run_id` on the
    /// other three are the fields the Tower refuses on — `malformed_request`
    /// for a missing or wrong-typed one — so their presence is the assertion.
    func testEveryCVLabCommandGoesOutOnTheSocketInTheTowersShape() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }
        _ = await waitForDiscovery(recorder)

        client.sendCVLabStatusRequest(requestID: "cv-1")
        client.sendCVLabStart(experimentID: "edge_detection", requestID: "cv-2")
        client.sendCVLabPause(runID: "f863dcc35bce-2", requestID: "cv-3")
        client.sendCVLabResume(runID: "f863dcc35bce-2", requestID: "cv-4")
        client.sendCVLabStop(runID: "f863dcc35bce-2", requestID: "cv-5")

        await expect {
            recorder.all.compactMap(self.decode)
                .filter { ($0["type"] as? String)?.hasPrefix("cv_lab_") == true }
                .count == 5
        }

        let sent = recorder.all.compactMap(decode)
            .filter { ($0["type"] as? String)?.hasPrefix("cv_lab_") == true }
        XCTAssertEqual(
            sent.compactMap { $0["type"] as? String },
            ["cv_lab_status", "cv_lab_start", "cv_lab_pause", "cv_lab_resume", "cv_lab_stop"]
        )
        XCTAssertEqual(sent[1]["experiment_id"] as? String, "edge_detection")
        // Sent on all three of the commands that take one, because a stale id
        // is refused rather than applied to whichever run is current — which is
        // the difference between a refusal and somebody else's run being
        // stopped.
        for index in 2...4 {
            XCTAssertEqual(sent[index]["run_id"] as? String, "f863dcc35bce-2")
        }
        for (index, message) in sent.enumerated() {
            XCTAssertEqual(message["request_id"] as? String, "cv-\(index + 1)")
        }
        // Nothing that is not a command went out on the CV Lab's account, and
        // in particular no `result_subscribe` carrying a mutation: commands
        // never travel on the result channel.
        XCTAssertFalse(
            recorder.all.contains { $0.contains("result_subscribe") && $0.contains("cv_lab") }
        )

        client.disconnect()
    }

    /// A `request_id` longer than the Tower's bound is dropped **locally**.
    ///
    /// The Tower drops an oversized one without refusing the command — verified
    /// against the live Tower, which answered a 65-character id with a reply
    /// carrying no `request_id` at all. That is the worst shape of failure for
    /// a client matching replies to buttons: the command applies and the answer
    /// cannot be attributed. So this client declines to send one it knows will
    /// be dropped, rather than discovering the loss on the reply.
    ///
    /// Dropped, never truncated: a truncated id is a *different* id, and one
    /// that could collide with another button's.
    func testAnOversizedRequestIDIsDroppedBeforeItReachesTheWire() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }
        _ = await waitForDiscovery(recorder)

        let atTheBound = String(repeating: "x", count: 64)
        let overTheBound = String(repeating: "x", count: 65)
        client.sendCVLabStatusRequest(requestID: atTheBound)
        client.sendCVLabStatusRequest(requestID: overTheBound)

        await expect {
            recorder.all.compactMap(self.decode)
                .filter { $0["type"] as? String == "cv_lab_status" }
                .count == 2
        }
        let sent = recorder.all.compactMap(decode)
            .filter { $0["type"] as? String == "cv_lab_status" }
        XCTAssertEqual(sent[0]["request_id"] as? String, atTheBound, "the bound is inclusive")
        XCTAssertNil(sent[1]["request_id"], "an oversized request_id reached the wire")
        // And the command still went, because dropping the id must not drop the
        // request — that is what the Tower does, and doing otherwise here would
        // make the two sides disagree about whether anything was asked.
        XCTAssertEqual(sent.count, 2)

        client.disconnect()
    }

    /// `accepted_command` is the field that tells an answer from a push.
    ///
    /// Both arrive as `cv_lab_status`, and a client that read every arriving
    /// document as an answer would report "started" on every two-second
    /// heartbeat. Verified against the live Tower: a plain read comes back with
    /// no `accepted_command`, and each command echoes its own name.
    func testAPushedStatusIsDistinguishableFromAnAnswerToACommand() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        let events = collectCVLabEvents(client)
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        server.send(text: cvLabStatusJSON(runID: "f863dcc35bce-2"))
        server.send(
            text: cvLabStatusJSON(
                runID: "f863dcc35bce-2", acceptedCommand: "cv_lab_start", requestID: "cv-2"
            )
        )
        await expect { events.all.count >= 2 }

        guard case .status(let pushed) = events.all[0] else {
            return XCTFail("expected a status, got \(events.all[0])")
        }
        XCTAssertNil(pushed.acceptedCommand, "a pushed document claimed to answer a command")
        XCTAssertNil(pushed.requestID)

        guard case .status(let answer) = events.all[1] else {
            return XCTFail("expected a status, got \(events.all[1])")
        }
        XCTAssertEqual(answer.acceptedCommand, "cv_lab_start")
        XCTAssertEqual(answer.requestID, "cv-2")
        // The document is passed through undecoded, for the reason
        // `CartridgeResultEnvelope.payload` is: this client owns the transport
        // and the cartridge owns the contract.
        XCTAssertEqual(answer.status["selected"] as? String, "edge_detection")
        XCTAssertEqual(answer.runID, "f863dcc35bce-2")

        client.disconnect()
    }

    /// The eight refusals reach a consumer with their reason-specific extras.
    ///
    /// Every one of them means the request did not take effect, so what matters
    /// is that the refusal arrives *whole*: `available` on an unknown
    /// experiment and `current_run_id` on a stale run are what let a client say
    /// something useful instead of "something went wrong". Both fixtures are
    /// live Tower replies.
    func testARefusalArrivesWithItsExtrasAndItsUnchangedDocument() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        let events = collectCVLabEvents(client)
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        server.send(text: """
            {"type":"cv_lab_error","control_contract":"experimental_cv.control/2026-08-27",
             "reason":"unknown_experiment","message":"this Tower has no experiment 'nope'",
             "command":"cv_lab_start","request_id":"cv-2",
             "status":{"lifecycle":{"state":"running","run_id":"f863dcc35bce-1"}},
             "available":["baseline","depth","edge_detection"]}
            """)
        server.send(text: """
            {"type":"cv_lab_error","control_contract":"experimental_cv.control/2026-08-27",
             "reason":"stale_run",
             "message":"run 'bogus-99' is not the current run; the Lab is now on 'f863dcc35bce-1'",
             "command":"cv_lab_pause","request_id":"cv-5",
             "status":{"lifecycle":{"state":"running","run_id":"f863dcc35bce-1"}},
             "current_run_id":"f863dcc35bce-1"}
            """)
        await expect { events.all.count >= 2 }

        guard case .refused(let unknown) = events.all[0] else {
            return XCTFail("expected a refusal, got \(events.all[0])")
        }
        XCTAssertEqual(unknown.reason, "unknown_experiment")
        XCTAssertEqual(unknown.command, "cv_lab_start")
        XCTAssertEqual(unknown.available, ["baseline", "depth", "edge_detection"])
        XCTAssertNotNil(unknown.status, "a refusal arrived without the document it must carry")

        guard case .refused(let stale) = events.all[1] else {
            return XCTFail("expected a refusal, got \(events.all[1])")
        }
        XCTAssertEqual(stale.reason, "stale_run")
        XCTAssertEqual(stale.currentRunID, "f863dcc35bce-1")

        // A refusal is not a connection failure. The socket the camera streams
        // over must survive every one of them.
        XCTAssertEqual(client.status, .online)

        client.disconnect()
    }

    /// `frame_error` reaches a consumer instead of being logged and discarded.
    ///
    /// It had **no case at all**: every one fell through to `default:` as an
    /// unknown message type. A paused Lab, an arming model and an undecodable
    /// frame were therefore indistinguishable from silence.
    ///
    /// And it is not a failure. Nothing about the connection, the send window
    /// or the reply counters may move — a Lab paused for five minutes has not
    /// failed hundreds of times.
    func testAFrameErrorIsDeliveredAndCostsTheFramePathNothing() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let metrics = SenderMetrics()
        let client = TowerClient(metrics: metrics)
        let events = collectCVLabEvents(client)
        client.connect(to: url(port: port))
        await expect { client.status == .online }
        let repliesBefore = metrics.snapshot.frameResults

        server.send(text: """
            {"type":"frame_error","seq":90,"reason":"cv_lab_paused",
             "message":"the CV Lab is paused; send cv_lab_resume to continue"}
            """)
        await expect { client.latestFrameRefusal != nil }

        XCTAssertEqual(client.latestFrameRefusal?.reason, "cv_lab_paused")
        XCTAssertEqual(client.latestFrameRefusal?.sequence, 90)
        guard case .frameRefused(let refusal) = events.all.last else {
            return XCTFail("the refusal never reached a consumer: \(events.all)")
        }
        XCTAssertEqual(refusal.reason, "cv_lab_paused")
        XCTAssertEqual(client.status, .online, "a refused frame took down the connection")
        XCTAssertEqual(
            metrics.snapshot.frameResults, repliesBefore,
            "a refusal was counted as a reply"
        )

        client.disconnect()
    }

    /// A frame that failed validation before its `seq` was readable reports a
    /// null sequence rather than an invented one.
    func testAFrameErrorWithNoSequenceIsDecodedRatherThanDropped() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        server.send(text: """
            {"type":"frame_error","seq":null,"reason":"invalid_frame",
             "message":"malformed frame message, missing fields: ['seq']"}
            """)
        await expect { client.latestFrameRefusal != nil }
        XCTAssertNil(client.latestFrameRefusal?.sequence)
        XCTAssertEqual(client.latestFrameRefusal?.reason, "invalid_frame")

        client.disconnect()
    }

    /// >>> The staleness rule. <<<
    ///
    /// A `frame_result` whose `cv_lab.run_id` is not the run being watched is
    /// discarded, and the run being watched is learned from the **status
    /// document** — never from a result, which would let a stale reply nominate
    /// itself as the run to watch and then match.
    func testAResultFromAnotherRunIsDiscardedRatherThanShown() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }
        client.sendStreamStart()

        server.send(text: cvLabStatusJSON(runID: "f863dcc35bce-2"))
        await expect { client.watchedCVLabRunID == "f863dcc35bce-2" }

        server.send(text: frameResultJSON(seq: 60, runID: "f863dcc35bce-2", resultSeq: 2))
        await expect { client.latestFrameResult != nil }
        XCTAssertEqual(client.latestFrameResult?.cvLab?.resultSeq, 2)

        // A result from the run that was replaced. Structurally impossible on
        // the Tower's side within one connection — the old experiment is
        // released before the new run id is published — which is exactly why
        // the client-side rule exists: it covers the case the Tower cannot,
        // a reconnect to a restarted Tower whose run numbering starts again.
        server.send(text: frameResultJSON(seq: 90, runID: "f863dcc35bce-1", resultSeq: 99))
        await expect { client.staleFrameResultCount == 1 }
        XCTAssertEqual(
            client.latestFrameResult?.cvLab?.resultSeq, 2,
            "a result from another run replaced the one being watched"
        )

        client.disconnect()
    }

    /// A reply with no `cv_lab` block is passed through, gate or no gate.
    ///
    /// A Tower running no Lab attaches no provenance and has therefore made no
    /// claim about which run a result belongs to. Discarding those would drop
    /// every result from such a Tower — the gate is for contradicting a claim,
    /// not for requiring one.
    func testAResultWithNoProvenanceBlockIsNotGatedOut() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }
        client.sendStreamStart()

        server.send(text: cvLabStatusJSON(runID: "f863dcc35bce-2"))
        await expect { client.watchedCVLabRunID == "f863dcc35bce-2" }

        server.send(
            text: #"{"type":"frame_result","seq":1,"mean_intensity":0.5,"processing_ms":1.0}"#
        )
        await expect { client.latestFrameResult != nil }
        XCTAssertNil(client.latestFrameResult?.cvLab)
        XCTAssertEqual(client.staleFrameResultCount, 0)

        client.disconnect()
    }

    /// A run that changes takes its reading with it.
    ///
    /// A run is the unit of provenance, and the Tower makes the same move on
    /// its side: starting a different experiment mints a new run and takes the
    /// previous one's figures out of the document. Leaving the last reading on
    /// screen across a switch would put the old experiment's number under the
    /// new experiment's name.
    func testAdoptingANewRunDropsThePreviousRunsReading() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }
        client.sendStreamStart()

        server.send(text: cvLabStatusJSON(runID: "f863dcc35bce-2"))
        await expect { client.watchedCVLabRunID == "f863dcc35bce-2" }
        server.send(text: frameResultJSON(seq: 30, runID: "f863dcc35bce-2", resultSeq: 1))
        await expect { client.latestFrameResult != nil }

        server.send(
            text: cvLabStatusJSON(
                runID: "f863dcc35bce-3", state: "starting", acceptedCommand: "cv_lab_start"
            )
        )
        await expect { client.watchedCVLabRunID == "f863dcc35bce-3" }
        XCTAssertNil(
            client.latestFrameResult,
            "the previous run's reading survived a switch to a new experiment"
        )

        client.disconnect()
    }

    /// The run change this phone did **not** cause.
    ///
    /// ## Why the test above is not enough
    ///
    /// `testAdoptingANewRunDropsThePreviousRunsReading` drives the change with a
    /// `cv_lab_status` message — and the Tower sends that **only in reply to a
    /// command on the same connection** (`tower/routes/cv_lab_ws.py`). There is
    /// no unsolicited push. So that test covers exactly the case where this
    /// phone started the new run itself.
    ///
    /// The Lab has **one slot shared by every connection, and last start wins**.
    /// When a second operator — or the bench's own `cv_lab_smoke.py` — starts a
    /// different experiment, the only thing that reaches this app is a
    /// `cartridge_result` on the subscription. That path did not feed the gate,
    /// and the consequences were entirely silent:
    ///
    ///  - every following `frame_result` carried the new run and was discarded,
    ///    so the counter the product screen reads froze while frames were being
    ///    answered normally;
    ///  - the held reading was never cleared, so the card kept showing the
    ///    **previous experiment's figures** under the new experiment's name.
    ///
    /// Reproduced against a live Tower during review. This is the regression.
    func testARunStartedByAnotherClientIsAdoptedFromTheSubscription() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let tower = TowerClient(metrics: SenderMetrics())
        // Constructed and retained: the client is what listens to the result
        // channel and feeds the gate. Without it this test would pass against
        // the bug, because nothing would be subscribed.
        let lab = TowerExperimentalCVClient(tower: tower)
        withExtendedLifetime(lab) {}
        tower.connect(to: url(port: port))
        await expect { tower.status == .online }
        tower.sendStreamStart()

        // This phone's own run, learned the ordinary way.
        server.send(text: cvLabStatusJSON(runID: "f863dcc35bce-6"))
        await expect { tower.watchedCVLabRunID == "f863dcc35bce-6" }
        server.send(text: frameResultJSON(seq: 30, runID: "f863dcc35bce-6", resultSeq: 1))
        await expect { tower.latestFrameResult != nil }

        // Somebody else starts a different experiment. The Tower pushes the new
        // document on the SUBSCRIPTION, and sends this phone no `cv_lab_status`.
        server.send(text: cvLabStatusOnSubscriptionJSON(runID: "f863dcc35bce-7"))
        await expect { tower.watchedCVLabRunID == "f863dcc35bce-7" }

        XCTAssertNil(
            tower.latestFrameResult,
            """
            the previous experiment's reading survived a run change this phone \
            did not cause -- it would sit on screen under the new experiment's name
            """
        )

        // And the new run's results are accepted rather than discarded.
        server.send(text: frameResultJSON(seq: 60, runID: "f863dcc35bce-7", resultSeq: 1))
        await expect { tower.latestFrameResult != nil }

        tower.disconnect()
    }

    /// The CV Lab's status document as it arrives on the **result channel**,
    /// which is the only way a run change caused by another client is seen.
    private func cvLabStatusOnSubscriptionJSON(runID: String) -> String {
        """
        {"type":"cartridge_result",
         "envelope_contract":"cartridge_results.envelope/2026-08-23",
         "subscription_id":"sub-cv","cartridge":"experimental_cv","result_type":"status",
         "contract":"experimental_cv.status/2026-08-27",
         "seq":2,"revision":"r2","revision_changed":true,
         "coalesced":0,"cursor_status":null,"snapshot":true,
         "tower_sent_at":1787463092.958,"time_basis":"tower-receipt",
         "payload":{"contract":"experimental_cv.status/2026-08-27",
           "tower_instance_id":"f863dcc35bce",
           "lifecycle":{"state":"running","reason":null,
                        "since":1787833032.19,"run_id":"\(runID)"},
           "available":[],"selected":"optical_flow",
           "source":{"clients_connected":2,"receiving_frames":true,
                     "last_frame_at":1787833032.32,"frames_offered_total":9,
                     "frames_rejected_before_lab":0,"idle_after_s":5.0}}}
        """
    }

    /// The three clear points, asserted as three.
    ///
    /// The held run id has the same lifetime as the held reading —
    /// `sendStreamStart()`, `sendStreamStop()` and teardown — because a run id
    /// that outlived a bracket would gate the next bracket's replies against a
    /// run that ended and discard every one of them.
    func testTheRunWatchIsClearedAtTheSameThreePointsAsTheReading() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        // 1. stream_stop.
        client.sendStreamStart()
        server.send(text: cvLabStatusJSON(runID: "f863dcc35bce-2"))
        await expect { client.watchedCVLabRunID != nil }
        client.sendStreamStop()
        XCTAssertNil(client.watchedCVLabRunID, "a stream_stop left the run watch standing")

        // 2. stream_start. A fresh bracket must not inherit the previous one's
        // run any more than it inherits the previous one's reply.
        server.send(text: cvLabStatusJSON(runID: "f863dcc35bce-2"))
        await expect { client.watchedCVLabRunID != nil }
        client.sendStreamStart()
        XCTAssertNil(client.watchedCVLabRunID, "a stream_start inherited the previous bracket's run")

        // 3. teardown. The only one of the three that runs in a Release build,
        // and the one that covers a reconnect to a restarted Tower.
        server.send(text: cvLabStatusJSON(runID: "f863dcc35bce-2"))
        await expect { client.watchedCVLabRunID != nil }
        client.disconnect()
        XCTAssertNil(client.watchedCVLabRunID, "a teardown left the run watch standing")
    }

    /// The frame path keeps working through every CV Lab message.
    ///
    /// The same guarantee the result channel makes, extended to the control
    /// plane: a status, a refusal and a frame refusal in a row leave `status`,
    /// the stream bracket and the send window exactly where they were.
    func testTheControlPlaneCannotDisturbTheFramePath() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient(metrics: SenderMetrics())
        client.connect(to: url(port: port))
        await expect { client.status == .online }
        client.sendStreamStart()
        XCTAssertTrue(client.isStreamingToTower)

        server.send(text: cvLabStatusJSON(runID: "f863dcc35bce-2"))
        server.send(text: """
            {"type":"cv_lab_error","control_contract":"experimental_cv.control/2026-08-27",
             "reason":"internal_error","message":"the CV Lab could not answer that request",
             "command":"cv_lab_start","status":{"lifecycle":{"state":"running","run_id":"f863dcc35bce-2"}}}
            """)
        server.send(text: """
            {"type":"frame_error","seq":90,"reason":"cv_lab_starting",
             "message":"the CV Lab is arming an experiment; frames are refused until it is ready"}
            """)
        await expect { client.latestFrameRefusal?.reason == "cv_lab_starting" }

        XCTAssertEqual(client.status, .online)
        XCTAssertTrue(client.isStreamingToTower, "the control plane closed the stream bracket")
        XCTAssertEqual(client.maxFramesInFlight, TowerClient.defaultMaxFramesInFlight)

        client.disconnect()
    }

    // MARK: - The frame gate

    /// The hold is cleared by the camera STOPPING, not by a `stream_stop`
    /// actually leaving. A socket drop closes the bracket on its own
    /// (`teardownConnection`); a camera stopped during that gap used to keep
    /// the hold into the next session, where Home shows no control for it.
    func testStopClearsTheFrameGateEvenWhenTheBracketIsAlreadyGone() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        await expect { client.status == .online }
        // No `sendStreamStart()`: the bracket is not open, as after a drop.
        client.pauseFrameSending()
        XCTAssertTrue(client.isFrameSendingPaused)

        client.sendStreamStop()

        XCTAssertFalse(client.isFrameSendingPaused, "the hold outlived the camera session")
        XCTAssertFalse(client.isStreamingToTower)
        try? await Task.sleep(nanoseconds: 100_000_000)
        let stops = recorder.all.compactMap(decode).filter { $0["type"] as? String == "stream_stop" }
        XCTAssertEqual(stops.count, 0, "a stop with no open bracket must still send nothing")
    }

    /// Paused: the frame is selected, counted, and never leaves the phone.
    /// Resumed: the next one does. The bracket is untouched throughout — no
    /// `stream_stop` goes out, because the whole point of the gate is that the
    /// Tower keeps the capture lineage and the armed experiment.
    func testPausedFrameSendingHoldsFramesOnThePhone() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        let recorder = attachRecorder(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        await expect { client.status == .online }
        client.sendStreamStart()

        func count(_ type: String) -> Int {
            recorder.all.compactMap(decode).filter { $0["type"] as? String == type }.count
        }

        XCTAssertFalse(client.isFrameSendingPaused, "the gate must start open")
        client.pauseFrameSending()
        XCTAssertTrue(client.isFrameSendingPaused)
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 1)
        try? await Task.sleep(nanoseconds: 150_000_000)
        XCTAssertEqual(count("frame"), 0, "a frame left the phone while sending was paused")
        XCTAssertEqual(count("stream_stop"), 0, "the gate must hold frames, not end the bracket")
        XCTAssertTrue(client.isStreamingToTower, "the gate closed the stream bracket")

        client.resumeFrameSending()
        XCTAssertFalse(client.isFrameSendingPaused)
        client.sendFrame(makeTestImage(), width: 2, height: 2, sequence: 2)
        await expect("no frame flowed after resume") { count("frame") == 1 }

        client.disconnect()
    }

    /// `disconnect()` is the user ending the connection; a hold set during it
    /// must not survive into the next connect.
    func testDisconnectResetsTheFrameGate() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        client.pauseFrameSending()
        XCTAssertTrue(client.isFrameSendingPaused)
        client.disconnect()
        XCTAssertFalse(client.isFrameSendingPaused, "a disconnect left the frame gate closed")
    }

    /// A hold belongs to the camera session it was set in. `stream_stop` ends
    /// that session's bracket, and the next start must stream.
    func testStreamStopResetsTheFrameGate() async throws {
        let server = try MockTowerServer()
        let port = try await server.start()
        respondToPing(server)
        defer { server.stop() }

        let client = TowerClient()
        client.connect(to: url(port: port))
        await expect { client.status == .online }

        client.sendStreamStart()
        client.pauseFrameSending()
        XCTAssertTrue(client.isFrameSendingPaused)
        client.sendStreamStop()
        XCTAssertFalse(client.isFrameSendingPaused, "a stream_stop left the frame gate closed")

        client.disconnect()
    }
}

/// Collects CV Lab control-plane events in arrival order.
///
/// A class for the reason `EventRecorder` is one: a `sink` appends from an
/// escaping closure, and `@MainActor` because that is where `TowerClient`
/// publishes.
@MainActor
final class CVLabEventRecorder {
    private(set) var all: [CVLabEvent] = []
    var cancellable: AnyCancellable?

    func record(_ event: CVLabEvent) { all.append(event) }
}

/// Thread-safe capture of every text message a MockTowerServer receives, so
/// tests can assert on the exact sequence/content sent over the wire.
///
/// Not `private`: `WorldBuilderIntegrationTests` asserts on the wire for the
/// same reason this file does — "the client sent exactly one subscribe" is only
/// checkable against what actually left the socket — and a second copy of this
/// would be two things that could disagree about what was recorded.
final class MessageRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var messages: [String] = []

    func record(_ text: String) {
        lock.lock()
        messages.append(text)
        lock.unlock()
    }

    var all: [String] {
        lock.lock()
        defer { lock.unlock() }
        return messages
    }
}


/// Collects result-channel events in arrival order.
///
/// A class rather than an array so a `sink` can append to it from the escaping
/// closure, and `@MainActor` because that is where `TowerClient` publishes.
@MainActor
final class EventRecorder {
    private(set) var all: [CartridgeResultEvent] = []
    var cancellable: AnyCancellable?

    func record(_ event: CartridgeResultEvent) { all.append(event) }
}
