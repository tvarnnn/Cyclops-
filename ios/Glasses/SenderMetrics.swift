//
//  SenderMetrics.swift
//  Glasses
//

import Combine
import Foundation

/// One streaming session's sender-side counters, as a plain value.
///
/// Every field is a raw count or an accumulated duration. Rates are derived
/// (see the extension below) rather than stored, so a snapshot can never
/// disagree with itself.
///
/// The Tower reports what it *received*. Before this existed there was no way
/// to tell a Tower-side loss from an iOS-side decision, which is exactly how a
/// deliberate 1-in-30 sampling stride went unnoticed as a 0.8 fps "bottleneck".
/// Each counter below marks one place a frame can stop travelling.
struct SenderMetricsSnapshot: Equatable, Sendable {

    // MARK: Capture

    /// DAT camera callbacks observed. Sequence numbers are assigned one per
    /// callback, so this is also the number of sequence IDs issued.
    var framesCaptured = 0
    /// Highest sequence number assigned. Compare against `framesCaptured` —
    /// see `sequenceInvariantHolds`.
    var latestSequence = 0

    // MARK: Sampling

    /// Frames the rate gate admitted for transmission.
    var framesSelected = 0
    /// Frames the rate gate deliberately skipped. Not a defect: this is the
    /// intended difference between the capture rate and the target send rate.
    var framesSkipped = 0
    /// Selected frames DAT could not hand us pixels or dimensions for.
    var decodeFailures = 0

    // MARK: Encoding

    var framesEncoded = 0
    var encodeFailures = 0
    /// Wall time spent inside JPEG encode + base64 + JSON serialisation.
    var encodeSecondsTotal: Double = 0
    var encodeSecondsMax: Double = 0

    // MARK: Transmission

    /// Frames handed to the WebSocket for sending.
    var sendAttempts = 0
    /// Frames dropped because the bounded in-flight send window was full.
    /// A non-zero value here is the pipeline correctly shedding load rather
    /// than queueing it.
    var sendWindowDrops = 0
    /// Frames dropped because the Tower was not online, no `stream_start`
    /// was in effect, or frame sending was paused on the phone
    /// (`TowerClient.isFrameSendingPaused`).
    var sessionGateDrops = 0
    /// Bytes handed to the WebSocket (JSON envelope, excluding WebSocket and
    /// TCP framing).
    var wireBytes = 0

    // MARK: Completion

    var sendSuccesses = 0
    var sendFailures = 0
    /// Sends whose socket was torn down before they completed. Neither a
    /// success nor a failure of this connection, but still a terminal outcome
    /// for the frame.
    var sendAbandoned = 0
    /// `frame_result` messages received back from the Tower.
    var frameResults = 0

    // MARK: Slot lifetime

    /// Completions that carried a valid send-window slot, and so contributed a
    /// timing sample below. Abandoned sends do not: their slot was already
    /// reclaimed by teardown, so they have no lifetime to report.
    var slotSamples = 0

    /// Time from a frame taking a send-window slot to
    /// `URLSessionWebSocketTask.send` invoking its completion handler —
    /// measured *inside* that handler, before hopping to the main actor.
    ///
    /// This is the transport's own write time, and it is the single most
    /// important number this pipeline was not measuring. The send window's
    /// capacity divided by the slot lifetime *is* the achievable send rate
    /// (see `SendWindow`), so a rate below target is either this being large
    /// or the window being small, and nothing else.
    var sendLatencySecondsTotal: Double = 0
    var sendLatencySecondsMax: Double = 0

    /// Time from a frame taking a send-window slot to that slot actually being
    /// returned on the main actor — i.e. `sendLatency` plus the completion
    /// handler's hop back onto the main actor.
    ///
    /// The slot is held for this whole span, so this, not `sendLatency`, is
    /// the true rate denominator. Recorded separately because the difference
    /// between the two attributes a shortfall to the *network* or to *main-actor
    /// congestion*, which are opposite diagnoses with opposite fixes and were
    /// previously indistinguishable.
    var slotLifetimeSecondsTotal: Double = 0
    var slotLifetimeSecondsMax: Double = 0

    /// Times a full, non-draining send window caused the connection to be torn
    /// down. Each one is a stall that would previously have blocked the
    /// pipeline for as long as the peer took to resume — 52 seconds, in the
    /// physical baseline.
    ///
    /// Counts the teardown, not the recovery: whether a new connection follows
    /// depends on `TowerClient.autoReconnect`, which the app enables and the
    /// default initialiser does not. The teardown is the part this counter can
    /// truthfully speak for.
    var stallRecoveries = 0

    // MARK: Timing

    /// Seconds since `SenderMetrics.begin()`, measured monotonically.
    var duration: TimeInterval = 0
}

extension SenderMetricsSnapshot {

    /// Rates are `nil` until there is enough elapsed time to make one
    /// meaningful. Dividing a handful of frames by a few milliseconds
    /// fabricates precision the pipeline never measured.
    private func rate(_ count: Int) -> Double? {
        guard duration >= 0.5 else { return nil }
        return Double(count) / duration
    }

    var captureFPS: Double? { rate(framesCaptured) }
    var selectedFPS: Double? { rate(framesSelected) }
    var sendAttemptFPS: Double? { rate(sendAttempts) }
    /// Frames confirmed written per second.
    ///
    /// "Written" means into the kernel socket buffer, which is exactly what
    /// `URLSessionWebSocketTask.send`'s completion documents and no more: it is
    /// not an acknowledgement that the Tower received anything. `frameResults`
    /// is the only end-to-end figure. The two normally track closely, and a
    /// gap between them is the Tower falling behind rather than the uplink.
    var successfulSendFPS: Double? { rate(sendSuccesses) }
    var towerResultFPS: Double? { rate(frameResults) }
    var wireBytesPerSecond: Double? { rate(wireBytes) }

    var encodeMsAverage: Double? {
        guard framesEncoded > 0 else { return nil }
        return encodeSecondsTotal / Double(framesEncoded) * 1000
    }

    var encodeMsMax: Double? {
        guard framesEncoded > 0 else { return nil }
        return encodeSecondsMax * 1000
    }

    var sendLatencyMsAverage: Double? {
        guard slotSamples > 0 else { return nil }
        return sendLatencySecondsTotal / Double(slotSamples) * 1000
    }

    var sendLatencyMsMax: Double? {
        guard slotSamples > 0 else { return nil }
        return sendLatencySecondsMax * 1000
    }

    var slotLifetimeMsAverage: Double? {
        guard slotSamples > 0 else { return nil }
        return slotLifetimeSecondsTotal / Double(slotSamples) * 1000
    }

    var slotLifetimeMsMax: Double? {
        guard slotSamples > 0 else { return nil }
        return slotLifetimeSecondsMax * 1000
    }

    /// Mean main-actor hop between the transport finishing a write and the
    /// send window getting its slot back.
    ///
    /// Derived from the two means rather than sampled directly, which is valid
    /// because both are means over the same `slotSamples` completions. There is
    /// deliberately no "max hop": the two maxima can come from different
    /// frames, so subtracting them would fabricate a measurement. Compare
    /// `sendLatencyMsMax` and `slotLifetimeMsMax` directly instead.
    var completionHopMsAverage: Double? {
        guard let lifetime = slotLifetimeMsAverage, let latency = sendLatencyMsAverage else { return nil }
        return max(0, lifetime - latency)
    }

    /// The send rate the window can sustain at the measured slot lifetime,
    /// independent of what the gate selected: `capacity / slotLifetime`.
    ///
    /// The diagnosis in one number. If this sits at or above
    /// `FrameRateGate.towerTargetFPS` the window is not the constraint; if it
    /// sits below, `successfulSendFPS` cannot exceed it however many frames the
    /// gate admits, and the shortfall is arithmetic rather than mysterious.
    ///
    /// - Parameter capacity: the live `SendWindow.capacity`, which the
    ///   snapshot deliberately does not store — it is configuration, not a
    ///   measurement, and a stale copy of it here could disagree with the
    ///   window actually in use.
    func windowLimitedFPS(capacity: Int) -> Double? {
        guard let lifetime = slotLifetimeMsAverage, lifetime > 0 else { return nil }
        return Double(capacity) / (lifetime / 1000)
    }

    /// Fraction of captured frames that reached the socket. The 0.8 fps
    /// baseline run would have reported ≈ 0.033 here.
    var deliveredFraction: Double? {
        guard framesCaptured > 0 else { return nil }
        return Double(sendSuccesses) / Double(framesCaptured)
    }

    /// Sequence numbers are DAT callback ordinals, one per callback, so the
    /// highest one issued must equal the number of callbacks seen. A mismatch
    /// means callbacks were processed out of order, or a counter was reset
    /// mid-session — either of which would make the Tower's `seq_gap` figures
    /// unreadable.
    var sequenceInvariantHolds: Bool {
        latestSequence == framesCaptured
    }

    /// Selected frames that have not yet reached any terminal outcome — the
    /// pipeline's backlog depth.
    ///
    /// Deliberately a count and not a "queue is bounded" boolean. A selected
    /// frame reaches `TowerClient` one main-queue turn later (`ProjectManager`
    /// bridges it through `receive(on: DispatchQueue.main)`), so at the target
    /// rate there is essentially always one frame legitimately mid-hop, and a
    /// boolean would read "unbounded queue!" several times a second during
    /// perfectly healthy operation. The number cannot lie the same way: a
    /// small steady value is the hop plus the send window, and a value that
    /// climbs is the queue growth docs/03-ROADMAP.md V0.7 forbids.
    ///
    /// Expect roughly 0...(`maxFramesInFlight` + 1).
    var framesUnaccounted: Int {
        let terminal = decodeFailures
            + encodeFailures
            + sendWindowDrops
            + sessionGateDrops
            + sendSuccesses
            + sendFailures
            + sendAbandoned
        // Sends already handed to the socket are accounted for, just not
        // finished, so they must not read as backlog.
        let inFlight = sendAttempts - sendSuccesses - sendFailures - sendAbandoned
        return framesSelected - terminal - inFlight
    }
}

/// Collects `SenderMetricsSnapshot` counters across the whole sender path and
/// republishes them for the developer surface.
///
/// Shared by `GlassesConnection` (capture and sampling) and `TowerClient`
/// (encode and transmission). Neither of those types learns about the other by
/// holding this — it is a sink, not a channel — so the boundary in
/// docs/02-DEVELOPMENT-RULES.md Rule 1 is preserved, with `ProjectManager`
/// still the only owner of both.
///
/// Recording is a couple of integer increments plus one `Double` comparison.
/// *Publishing* invalidates the SwiftUI view tree, so it is rate limited to
/// `publishInterval`: instrumentation must not cost more than the thing it
/// measures.
@MainActor
final class SenderMetrics: ObservableObject {

    /// How often `snapshot` is refreshed while a session is running. 2 Hz is
    /// well above human reading speed and far below the 24 Hz capture rate.
    static let publishInterval: TimeInterval = 0.5

    @Published private(set) var snapshot = SenderMetricsSnapshot()

    /// Hot counters. Deliberately not `@Published` — see the note above about
    /// publishing cost.
    private var live = SenderMetricsSnapshot()
    /// Set by the first capture, not by `begin()`. `startCameraSession()` runs
    /// several seconds before DAT delivers a frame — creating the session,
    /// starting it, adding the camera, starting the stream — and counting that
    /// dead time would understate every rate by ~10% and send someone hunting
    /// a shortfall that isn't there. It also means a session that fails to
    /// start reports no duration rather than a clock that runs forever.
    private var startedAt: TimeInterval?
    /// Frozen at `finish()` so a session's final rates stop moving. Without
    /// it, any late arrival — a `frame_result` for a frame still in flight at
    /// Stop — republishes with a larger duration and unchanged counts, so the
    /// numbers visibly decay after the session ends.
    private var endedAt: TimeInterval?
    private var lastPublishedAt: TimeInterval = 0

    /// Starts a fresh session, discarding the previous one's counters.
    func begin(at now: TimeInterval = MonotonicClock.now) {
        live = SenderMetricsSnapshot()
        startedAt = nil
        endedAt = nil
        publish(at: now)
    }

    /// Publishes the counters for a session that has ended, so the numbers
    /// stay readable after Stop instead of being wiped or left mid-interval.
    ///
    /// Counts are not frozen — sends still outstanding when Stop was tapped
    /// land afterwards and are still counted, because they really were sent.
    /// The elapsed time is. Only `begin()` clears anything.
    func finish(at now: TimeInterval = MonotonicClock.now) {
        if endedAt == nil { endedAt = now }
        publish(at: now)
    }

    // MARK: Capture

    func recordCapture(sequence: Int) {
        if startedAt == nil { startedAt = MonotonicClock.now }
        live.framesCaptured += 1
        live.latestSequence = max(live.latestSequence, sequence)
        publishIfDue()
    }

    // MARK: Sampling

    func recordSelection() {
        live.framesSelected += 1
        publishIfDue()
    }

    func recordSkip() {
        live.framesSkipped += 1
        publishIfDue()
    }

    func recordDecodeFailure() {
        live.decodeFailures += 1
        publishIfDue()
    }

    // MARK: Encoding

    func recordEncode(seconds: TimeInterval) {
        live.framesEncoded += 1
        live.encodeSecondsTotal += seconds
        live.encodeSecondsMax = max(live.encodeSecondsMax, seconds)
        publishIfDue()
    }

    func recordEncodeFailure() {
        live.encodeFailures += 1
        publishIfDue()
    }

    // MARK: Transmission

    func recordSendAttempt(wireBytes: Int) {
        live.sendAttempts += 1
        live.wireBytes += wireBytes
        publishIfDue()
    }

    func recordSendWindowDrop() {
        live.sendWindowDrops += 1
        publishIfDue()
    }

    func recordSessionGateDrop() {
        live.sessionGateDrops += 1
        publishIfDue()
    }

    func recordSendSuccess() {
        live.sendSuccesses += 1
        publishIfDue()
    }

    func recordSendFailure() {
        live.sendFailures += 1
        publishIfDue()
    }

    func recordSendAbandoned() {
        live.sendAbandoned += 1
        publishIfDue()
    }

    /// Records one completed send's slot timings.
    ///
    /// Called alongside `recordSendSuccess()`/`recordSendFailure()` rather than
    /// folded into them: a failed send still occupied its slot for a real
    /// duration, and excluding it would bias the average towards the healthy
    /// case — precisely the case that needs no diagnosis.
    ///
    /// - Parameters:
    ///   - sendLatency: slot taken → transport completion handler entered.
    ///   - slotLifetime: slot taken → slot returned on the main actor. Never
    ///     less than `sendLatency`; clamped rather than trusted, since the two
    ///     are sampled from different execution contexts.
    func recordSlotTiming(sendLatency: TimeInterval, slotLifetime: TimeInterval) {
        let latency = max(0, sendLatency)
        let lifetime = max(latency, slotLifetime)
        live.slotSamples += 1
        live.sendLatencySecondsTotal += latency
        live.sendLatencySecondsMax = max(live.sendLatencySecondsMax, latency)
        live.slotLifetimeSecondsTotal += lifetime
        live.slotLifetimeSecondsMax = max(live.slotLifetimeSecondsMax, lifetime)
        publishIfDue()
    }

    func recordStallRecovery() {
        live.stallRecoveries += 1
        // Deliberately not rate limited: a stall recovery is rare, it is the
        // event this instrumentation exists to catch, and waiting up to half a
        // second to surface it would hide the moment the pipeline resumed.
        publish(at: MonotonicClock.now)
    }

    func recordFrameResult() {
        live.frameResults += 1
        publishIfDue()
    }

    // MARK: Publishing

    /// Test seam: the counters as they stand, without waiting for the
    /// publish interval. Production code reads `snapshot`.
    var currentSnapshot: SenderMetricsSnapshot {
        makeSnapshot(at: MonotonicClock.now)
    }

    private func publishIfDue() {
        let now = MonotonicClock.now
        guard now - lastPublishedAt >= Self.publishInterval else { return }
        publish(at: now)
    }

    private func publish(at now: TimeInterval) {
        lastPublishedAt = now
        snapshot = makeSnapshot(at: now)
    }

    private func makeSnapshot(at now: TimeInterval) -> SenderMetricsSnapshot {
        var value = live
        guard let startedAt else { return value }
        // Clamped: a session stopped before its first frame arrived, whose
        // first frame then lands late, would otherwise report negative time.
        value.duration = max(0, (endedAt ?? now) - startedAt)
        return value
    }
}
