//
//  DeveloperToolsView.swift
//  Glasses
//

// Entire file is DEBUG-only. Everything it touches — the Mock Device Kit
// controls, the raw session state, and the Tower counters — is already gated
// that way in the model, so gating the whole file keeps every control on one
// side of the conditional boundary.
#if DEBUG

import SwiftUI

/// Developer surface for the plumbing the product screen deliberately hides.
///
/// The Mock Device Kit controls live here now that the physical glasses work.
/// They are unchanged in behaviour and call exactly the same methods in the
/// same order as before; they are simply no longer part of the primary
/// experience. Mock support remains fully available for development and for
/// working without hardware.
struct DeveloperToolsView: View {
    @ObservedObject var glasses: GlassesConnection
    @ObservedObject var tower: TowerClient
    @ObservedObject var stream: StreamManager
    @ObservedObject var senderMetrics: SenderMetrics
    @ObservedObject var health: DeviceHealth

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                senderSection
                deviceHealthSection
                mockDeviceSection
                captureResolutionSection
                rawStateSection
                towerSection
                placeholderSection
            }
            .navigationTitle("Developer")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            // The root view's alert cannot present while this sheet is up, and
            // the mock controls are the main producers of `errorMessage`
            // ("Pair a mock device first", pairing failures). Without a local
            // alert those errors would be invisible until dismissal — on the
            // one surface built for diagnostics.
            .alert(
                "Something went wrong",
                isPresented: Binding(
                    get: { glasses.errorMessage != nil },
                    set: { isPresented in
                        if !isPresented { glasses.errorMessage = nil }
                    }
                )
            ) {
                Button("OK") { glasses.errorMessage = nil }
            } message: {
                Text(glasses.errorMessage ?? "")
            }
        }
    }

    // MARK: Sender pipeline

    /// One row per place a frame can stop travelling, so a rate below target
    /// can be attributed instead of guessed at. Reads a snapshot republished
    /// at 2 Hz rather than the live counters, so opening this sheet does not
    /// re-diff the `List` at the capture rate.
    private var senderSection: some View {
        let s = senderMetrics.snapshot
        return Section {
            LabeledContent("Session", value: Self.seconds(s.duration))

            LabeledContent("Captured", value: "\(s.framesCaptured)  \(Self.fps(s.captureFPS))")
            LabeledContent("Selected", value: "\(s.framesSelected)  \(Self.fps(s.selectedFPS))")
            LabeledContent("Skipped by gate", value: "\(s.framesSkipped)")
            LabeledContent("Send attempts", value: "\(s.sendAttempts)  \(Self.fps(s.sendAttemptFPS))")
            LabeledContent("Sent OK", value: "\(s.sendSuccesses)  \(Self.fps(s.successfulSendFPS))")
            LabeledContent("Tower replies", value: "\(s.frameResults)  \(Self.fps(s.towerResultFPS))")

            LabeledContent("Send-window drops", value: "\(s.sendWindowDrops)")
            LabeledContent("Session-gate drops", value: "\(s.sessionGateDrops)")
            LabeledContent("Decode failures", value: "\(s.decodeFailures)")
            LabeledContent("Encode failures", value: "\(s.encodeFailures)")
            LabeledContent("Send failures", value: "\(s.sendFailures)")
            LabeledContent("Abandoned by teardown", value: "\(s.sendAbandoned)")

            LabeledContent("Encode ms", value: "\(Self.ms(s.encodeMsAverage)) avg / \(Self.ms(s.encodeMsMax)) max")
            LabeledContent("Uplink", value: Self.bytesPerSecond(s.wireBytesPerSecond))
            LabeledContent("Delivered", value: Self.percent(s.deliveredFraction))

            // The rate arithmetic, spelled out. "Sent OK" above cannot exceed
            // "Window limit", so whenever it falls short of the target these
            // rows say whether the cause is the network (send ms), the main
            // actor (hop ms), or the window being too small for the measured
            // latency.
            LabeledContent("Send ms", value: "\(Self.ms(s.sendLatencyMsAverage)) avg / \(Self.ms(s.sendLatencyMsMax)) max")
            LabeledContent("Slot ms", value: "\(Self.ms(s.slotLifetimeMsAverage)) avg / \(Self.ms(s.slotLifetimeMsMax)) max")
            LabeledContent("Main-actor hop ms", value: Self.ms(s.completionHopMsAverage))
            LabeledContent(
                "Window limit",
                value: "\(Self.fps(s.windowLimitedFPS(capacity: tower.maxFramesInFlight)))  (\(tower.maxFramesInFlight) slots)"
            )
            LabeledContent("Stall recoveries", value: "\(s.stallRecoveries)")

            // Health checks rather than measurements. "Backlog" is a count,
            // not a verdict: a small steady number is the main-queue hop plus
            // the send window; a climbing one is queue growth. The healthy
            // ceiling scales with the window, so it is stated in the footer
            // rather than left for the reader to infer.
            LabeledContent("Backlog", value: "\(s.framesUnaccounted)")
            LabeledContent("Sequence 1:1", value: s.sequenceInvariantHolds ? "yes" : "NO")
        } header: {
            Text("Sender Pipeline")
        } footer: {
            Text("Counts and rates for one camera session, from DAT callback to Tower reply. Target send rate is \(String(format: "%.0f", FrameRateGate.towerTargetFPS)) fps. \"Window limit\" is slots ÷ slot ms — the ceiling on \"Sent OK\" regardless of how many frames the gate selects. \"Backlog\" is healthy up to about \(tower.maxFramesInFlight + 1); what matters is that it does not climb. Debug build with per-frame logging — treat rates as a floor.")
        }
    }

    private static func fps(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.1f/s", value)
    }

    private static func ms(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.2f", value)
    }

    private static func seconds(_ value: TimeInterval) -> String {
        String(format: "%.1fs", value)
    }

    private static func percent(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.1f%%", value * 100)
    }

    private static func bytesPerSecond(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.0f KB/s", value / 1024)
    }

    // MARK: Device health

    /// Thermal, power and battery telemetry for both ends of the link.
    ///
    /// Present because a five-minute physical run left the glasses warm to the
    /// touch and the sender rate decayed over the same minutes, and there was
    /// no way to tell whether those facts were related. Every row is a value
    /// the OS or the SDK actually reports; nothing here is estimated.
    private var deviceHealthSection: some View {
        Section {
            LabeledContent("Glasses thermal", value: glassesThermalText)
            LabeledContent("iPhone thermal", value: phoneThermalText)
            LabeledContent("iPhone battery", value: batteryText)
            LabeledContent("Low Power Mode", value: health.isLowPowerModeEnabled ? "On" : "Off")
        } header: {
            Text("Device Health")
        } footer: {
            Text("Glasses thermal is DAT's ThermalLevel. The pinned 1.0.0 SDK also reports glasses battery and charging state, which this screen does not read yet, and no numeric temperature at all — so those are absent rather than estimated. \"iPhone thermal\" at Serious or above means the system is throttling, which shows up in the sender rows above as encode and hop times growing.")
        }
    }

    /// Spells out the consequence rather than leaving the reader to know which
    /// `ThermalState` cases mean the system has started shedding performance.
    private var phoneThermalText: String {
        let name = health.thermalState.displayName
        return health.thermalState.isThrottling ? "\(name) — throttling" : name
    }

    /// `nil` — no active device, or the stream has not yielded yet — is shown
    /// as unavailable rather than as a benign-looking level.
    private var glassesThermalText: String {
        guard let level = glasses.glassesThermalLevel else { return "—" }
        return "\(level)"
    }

    private var batteryText: String {
        guard let level = health.batteryLevel else { return health.batteryState.displayName }
        return "\(Int((level * 100).rounded()))%  \(health.batteryState.displayName)"
    }

    // MARK: Mock Device Kit

    private var mockDeviceSection: some View {
        Section {
            Button(glasses.mockDeviceKitEnabled ? "Disable Mock Device Kit" : "Enable Mock Device Kit") {
                glasses.toggleMockDeviceKit()
            }
            Button("Pair Mock Glasses") {
                glasses.pairMockGlasses()
            }
            Button("Configure Mock Camera Feed") {
                glasses.configureMockCameraFeed()
            }
            LabeledContent("Mock Device Paired", value: glasses.isMockDevicePaired ? "Yes" : "No")
        } header: {
            Text("Mock Device Kit")
        } footer: {
            Text("Simulates Ray-Ban Meta glasses using this iPhone's camera. Enable, pair, then configure the feed — in that order — before starting a session.")
        }
    }

    // MARK: Capture resolution

    /// Chooses the rung the next capture session requests.
    ///
    /// This exists so a Document Memory experiment can be run on a device
    /// without an edit-and-rebuild. The app hardcoded `.low`, and Document
    /// Memory's premise cannot be tested at 360x640 — its word recall is
    /// 0.429-0.810 there against 0.957-1.000 at 1280x720. Raising the rung
    /// globally is not the answer either: 73.3% of 720p frames fall below
    /// World Builder's absolute `min_sharpness` and are rejected as blurred.
    /// That conflict is a cross-cartridge decision, so this stays a developer
    /// control rather than becoming a product setting. See
    /// `docs/agent-handoffs/TOWER-LANE-HANDOFF-FROM-MAC.md` 2.3.
    ///
    /// The picker is disabled on `isCaptureSessionClaimed` rather than on
    /// `isCaptureEngaged`, and the difference is the whole point.
    /// `StreamConfiguration` is consumed once by `addCamera(config:)` and DAT
    /// offers no way to renegotiate a live stream, so the control must be
    /// locked whenever a session is *held* — not merely whenever capture is
    /// actively running. `isCaptureEngaged` is false during `.paused` and
    /// `.stopping`, and a change accepted in either window is silently dropped
    /// by `beginCameraStream`'s `guard camera == nil` while this panel goes on
    /// displaying the rung that was never requested. That is exactly the state
    /// the UI asserts and the system does not hold.
    private var captureResolutionSection: some View {
        Section {
            Picker("Next session", selection: $glasses.captureResolution) {
                ForEach(CaptureResolutionPreference.allCases) { rung in
                    Text(rung.shortLabel).tag(rung)
                }
            }
            .pickerStyle(.segmented)
            .disabled(glasses.isCaptureSessionClaimed)

            LabeledContent("DAT declares", value: glasses.captureResolution.declaredSizeDescription)
        } header: {
            Text("Capture Resolution")
        } footer: {
            Text(captureResolutionFooter)
        }
    }

    /// Says which of the two situations the reader is in, rather than one
    /// sentence covering both. A footer that explains the disabled case while
    /// the control is enabled reads as a control that is broken.
    private var captureResolutionFooter: String {
        if glasses.isCaptureSessionClaimed {
            return "A capture session is still held — including while it is paused or stopping. DAT fixes the resolution when the stream is created and cannot renegotiate it, so changing this now could not take effect. Stop capture and wait for it to finish."
        }
        return "Applies when capture next starts, and resets to Low when the app relaunches. Low is the rung every existing measurement was taken at. Raising it harms World Builder tracking and helps Document Memory's OCR — but the axis that matters most is privacy: frames reach the Tower at this resolution and are never downscaled, and while the Tower's dataset recorder is armed every one is written to disk unredacted. A higher rung means more identifiable bystanders in that recording. It is a developer control, not a product setting."
    }

    // MARK: Raw state

    private var rawStateSection: some View {
        Section {
            LabeledContent("Registration", value: "\(glasses.registrationState)")
            LabeledContent("Devices", value: "\(glasses.devices.count)")
            LabeledContent("Camera Permission", value: glasses.cameraPermissionStatus.map { "\($0)" } ?? "nil")
            LabeledContent("Active Device", value: glasses.hasActiveDevice ? "Yes" : "No")
            LabeledContent("Device Session", value: "\(glasses.deviceSessionState)")
            LabeledContent("Camera Stream", value: "\(glasses.cameraStreamState)")
            LabeledContent("Frames Received", value: "\(glasses.frameCount)")

            Button("Check Camera Permission") {
                glasses.checkCameraPermission()
            }
        } header: {
            Text("Raw State")
        } footer: {
            Text("Unformatted enum values, as reported by DAT.")
        }
    }

    // MARK: Tower

    private var towerSection: some View {
        Section {
            LabeledContent("Endpoint") {
                Text(TowerConfiguration.webSocketURL.absoluteString)
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.trailing)
            }
            LabeledContent("Status", value: "\(tower.status)")
            LabeledContent("Streaming To Tower", value: tower.isStreamingToTower ? "Yes" : "No")
            // Labelled as per-bracket on purpose. It counts replies since the
            // last `stream_start`, and a reconnect reopens the bracket — so
            // after one it will read lower than the per-session "Tower replies"
            // on the Sender Pipeline section above. That difference is
            // information here, not a discrepancy, but only if the label says
            // which is which.
            LabeledContent("Frame Results (this bracket)", value: "\(tower.frameResultCount)")

            towerHealthRows

            Button(isCheckingHealth ? "Checking…" : "Check Tower State") {
                tower.refreshHealth()
            }
            .disabled(isCheckingHealth)
        } header: {
            Text("Tower")
        } footer: {
            Text(Self.towerFooter)
        }
    }

    private var isCheckingHealth: Bool {
        if case .fetching = tower.healthState { return true }
        return false
    }

    /// A constant, hoisted off the view so it is not rebuilt on every render of
    /// a sheet that redraws at the Tower's reply rate while it is open.
    private static let towerFooter =
        "The endpoint is compiled in and read-only. A configurable endpoint is a separate task. Everything below \"Frame Results\" is the Tower's own answer to GET /health at the moment the button was pressed — nothing polls, so it does not update on its own. \"\(Self.notSaid)\" means the field was missing from that answer, which is not a no and is not a zero. The dataset recorder is armed by the Tower's own TOWER_CAPTURE_ROOT setting when the Tower starts, and there is no way to arm or stop it from this app — which is why there is no control here, only a reading. While it is recording, every frame this phone sends is written to the Tower's disk unredacted."

    // MARK: What the Tower says about itself

    /// The wording for a field the Tower's health report did not contain.
    ///
    /// Not "—", and never "No" or "0". The question these rows exist to answer
    /// is "is the Tower writing my frames to disk right now?", and a missing
    /// field rendered as a reassuring `false` would answer it wrongly in the
    /// one direction that matters.
    private static let notSaid = "The Tower did not say"

    /// The Tower's own state, in the four states this app can be in about it.
    ///
    /// "Nobody has asked", "we are asking", "here is what it said" and "we
    /// asked and could not find out" are four different things, and a screen
    /// that draws the first and the last the same way has turned a failure
    /// into silence. Each gets its own rows.
    @ViewBuilder private var towerHealthRows: some View {
        switch tower.healthState {
        case .notFetched:
            LabeledContent("Tower State", value: "Not checked")

        case .fetching:
            LabeledContent("Tower State", value: "Checking…")

        case .failed(let error, let at):
            LabeledContent("Tower State", value: "Could not be read")
            LabeledContent("Why") {
                Text(Self.explain(error))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.trailing)
            }
            LabeledContent("Asked At", value: Self.clockTime(at))

        case .fetched(let health, let at):
            LabeledContent("Service", value: Self.serviceText(health))
            LabeledContent("Module", value: Self.moduleText(health))
            captureRows(health.capture)
            workerRows(health.captureWorkers)
            LabeledContent("Asked At", value: Self.clockTime(at))
        }
    }

    /// The dataset recorder, which is the reason any of this is on screen.
    ///
    /// The three non-`present` cases are kept apart deliberately: "no recorder
    /// is registered" is the Tower stating a fact about itself and is the only
    /// one of the three that means the frames are definitely not being kept.
    @ViewBuilder
    private func captureRows(_ capture: TowerReported<TowerCaptureState>) -> some View {
        switch capture {
        case .unreported:
            LabeledContent("Dataset Recorder", value: Self.notSaid)
        case .absent:
            LabeledContent("Dataset Recorder", value: "None registered")
        case .unreadable:
            LabeledContent("Dataset Recorder", value: "Answered in a shape this app cannot read")
        case .present(let state):
            LabeledContent(
                "Dataset Recorder",
                value: Self.said(state.armed, yes: "Armed", no: "Not armed")
            )
            LabeledContent(
                "Recording Now",
                value: Self.said(state.recording, yes: "Yes — writing to disk", no: "No")
            )
            captureIDRow(state.captureID)
            LabeledContent(
                "Frames Written", value: state.framesWritten.map(String.init) ?? Self.notSaid
            )
            LabeledContent("Bytes Written", value: Self.bytesText(state.bytesWritten))
            if let error = state.error {
                // Sits beside an "Armed" that is still true: the Tower knows a
                // recorder is registered and could not read the rest.
                LabeledContent("Recorder Fault", value: error)
            }
        }
    }

    /// The recording the counts belong to — and the three different answers a
    /// single Optional used to render as two.
    ///
    /// A `capture_id` of `null` is the Tower stating that its recorder has not
    /// opened a recording yet, and it always sends the key when it sends a
    /// `capture` object at all. Drawing that as "\(Self.notSaid)" told the
    /// reader the Tower withheld the id when the Tower had answered plainly —
    /// which is the field-level version of the mistake `TowerReported` exists
    /// to prevent, on the one screen built to answer a privacy question.
    @ViewBuilder
    private func captureIDRow(_ captureID: TowerReported<String>) -> some View {
        switch captureID {
        case .unreported:
            LabeledContent("Recording ID", value: Self.notSaid)
        case .absent:
            LabeledContent("Recording ID", value: "No recording opened yet")
        case .unreadable:
            LabeledContent("Recording ID", value: "Answered in a shape this app cannot read")
        case .present(let id):
            LabeledContent("Recording ID", value: id)
        }
    }

    /// Whether anything on the Tower is following the capture. `Enabled` with
    /// no workers is correct between walks and wrong during one, which is why
    /// the count is shown rather than folded into a yes/no.
    @ViewBuilder
    private func workerRows(_ workers: TowerReported<TowerCaptureWorkers>) -> some View {
        switch workers {
        case .unreported:
            LabeledContent("Capture Followers", value: Self.notSaid)
        case .absent:
            LabeledContent("Capture Followers", value: "None registered")
        case .unreadable:
            LabeledContent("Capture Followers", value: "Answered in a shape this app cannot read")
        case .present(let state):
            LabeledContent(
                "Capture Followers",
                value: Self.said(state.enabled, yes: "Enabled", no: "Disabled")
            )
            LabeledContent(
                "Followers Running", value: state.workerCount.map(String.init) ?? Self.notSaid
            )
            if let error = state.error {
                LabeledContent("Follower Fault", value: error)
            }
        }
    }

    /// `nil` is the Tower's silence and gets said as such — the whole point of
    /// keeping the field optional all the way from the wire to this row.
    private static func said(_ flag: Bool?, yes: String, no: String) -> String {
        guard let flag else { return notSaid }
        return flag ? yes : no
    }

    private static func serviceText(_ health: TowerHealth) -> String {
        guard let service = health.service else { return notSaid }
        guard let version = health.version else { return service }
        return "\(service) \(version)"
    }

    private static func moduleText(_ health: TowerHealth) -> String {
        guard let id = health.moduleID else { return health.moduleState ?? notSaid }
        guard let state = health.moduleState else { return id }
        return "\(id) — \(state)"
    }

    private static func bytesText(_ bytes: Int?) -> String {
        guard let bytes else { return notSaid }
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }

    /// The date, not just the clock time.
    ///
    /// Nothing ever clears `healthState`: it is written only by
    /// `refreshHealth()` and is never reset on disconnect, reconnect, teardown
    /// or backgrounding, and `TowerClient` lives for the app's lifetime. So a
    /// reading taken yesterday afternoon survives intact, and with the date
    /// omitted it rendered identically to one taken five minutes ago — on the
    /// one screen built to answer "is the Tower writing my frames to disk
    /// **right now**?".
    ///
    /// A relative "3 minutes ago" would read better and would need a timer to
    /// stay true; a stamp that is simply complete needs nothing and cannot go
    /// stale, because it never claimed to be current in the first place.
    private static func clockTime(_ date: Date) -> String {
        date.formatted(date: .abbreviated, time: .standard)
    }

    private static func explain(_ error: TowerHealthFetchError) -> String {
        switch error {
        case .undecodable:
            return "The Tower answered, and the answer could not be read as a health report. Nothing is shown rather than something guessed."
        case .transport(let description):
            return "The Tower did not answer: \(description)"
        }
    }

    // MARK: Placeholders

    private var placeholderSection: some View {
        Section {
            LabeledContent("State", value: "\(stream.state)")
            LabeledContent("Metrics", value: stream.metrics.map { "\($0)" } ?? "nil")
        } header: {
            Text("StreamManager")
        } footer: {
            Text("Placeholder type. Never assigned, so these never change. The real streaming state is Camera Stream above. Kept here rather than on the dashboard, where it contradicted the live frame counter.")
        }
    }
}

#endif
