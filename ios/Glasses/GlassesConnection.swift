//
//  GlassesConnection.swift
//  Glasses
//
//  Created by Tristan Varner on 8/18/26.
//

import Combine
import Foundation
import MWDATCore

#if DEBUG
import CoreMedia
import MWDATCamera
import MWDATMockDevice
import MWDATMotion
import UIKit
#endif

/// Wraps the Meta Wearables DAT `WearablesInterface` and exposes registration
/// and device-availability state to the rest of the app. This is the only
/// type in the app that talks to DAT directly — the boundary described in
/// docs/05-DAT-INTEGRATION.md.
///
/// Scope is intentionally limited to registration/connection state. Camera
/// streaming is not implemented here.
#if DEBUG
/// One decoded frame, ready for Tower transmission. Carries dimensions
/// derived directly from the `CMSampleBuffer` format description (the same
/// source used for the confirmed on-screen frame-dimension logging), not
/// re-derived from `UIImage.size`, to avoid any scale-factor ambiguity.
struct CapturedFrame {
    let image: UIImage
    let sequence: Int
    let width: Int
    let height: Int
}
#endif

#if DEBUG
/// Which rung of DAT's resolution enum the *next* capture session requests.
///
/// ## Why this exists, and why it is DEBUG-only
///
/// `StreamingResolution` is a **fixed three-case enum chosen once**, in the
/// `StreamConfiguration` handed to `session.addCamera(config:)`. It is not an
/// adaptive ladder: nothing in DAT renegotiates it mid-stream, and the P3
/// clean walk recorded **108 frames, every one 360x640, with zero variation**
/// (`docs/evidence/2026-08-26-p3-clean-walk-console.txt`). The app had this
/// value hardcoded to `.low`, so changing it required an edit and a rebuild.
///
/// That hardcoding blocks a measurement the Tower lane needs. Document
/// Memory's word recall is **0.429-0.810 at 360x640 and 0.957-1.000 at
/// 1280x720**, so its central premise cannot be tested at all without a way to
/// raise the rung on a device. Meanwhile 720p is *actively harmful* to World
/// Builder tracking — `min_sharpness = 25.0` is absolute and **73.3%** of 720p
/// frames are rejected as blurred.
///
/// Those two facts do not reconcile at a single rung, and choosing between
/// them is a cross-cartridge product decision that is not iOS's to make alone
/// (see `docs/agent-handoffs/TOWER-LANE-HANDOFF-FROM-MAC.md` 2.3). So this is
/// deliberately **a developer control, not a product setting**: it makes the
/// experiment runnable without pre-empting the decision.
///
/// ## The privacy consequence, which is the one that actually matters
///
/// This was reviewed and the first draft of this comment was found to omit it.
/// The rung is not merely an image-quality dial. `TowerClient.sendFrame` sends
/// the frame at its full captured size with **no downscale anywhere**, and when
/// the Tower's dataset recorder is armed it fsyncs those bytes to disk
/// verbatim — its own manifest declares `retains_raw_imagery: true`,
/// `redaction: "none"`, and the privacy tag `raw-imagery`. `.high` is **four
/// times the pixels** of the default.
///
/// So raising the rung raises the fidelity of unredacted, first-person imagery
/// of bystanders in a recording that persists. Face redaction does exist on the
/// Tower now, but it is applied only to World Builder's keyframe corpus, in
/// `_persist_keyframe`; the recorder's copy is written upstream of it and is
/// never redacted, so both copies coexist on disk.
///
/// That is why the footer on this control says so in plain words rather than
/// presenting the choice as OCR-versus-tracking, which is how it was first
/// written and which is a complete-sounding account that omits the axis a
/// person would most want to know about.
///
/// `.low` remains the default, so World Builder's physically-proven path is
/// unchanged unless someone deliberately changes it.
enum CaptureResolutionPreference: String, CaseIterable, Identifiable, Sendable {
    case low
    case medium
    case high

    /// The default, and the rung every existing measurement was taken at.
    static let `default`: CaptureResolutionPreference = .low

    var id: String { rawValue }

    var streamingResolution: StreamingResolution {
        switch self {
        case .low: return .low
        case .medium: return .medium
        case .high: return .high
        }
    }

    /// Short name for a picker segment.
    var shortLabel: String {
        switch self {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        }
    }

    /// The frame size **DAT itself declares** for this rung, rendered for
    /// display.
    ///
    /// Read from `StreamingResolution.videoFrameSize` rather than written out
    /// as a literal here. A hardcoded "360x640" would be this app asserting a
    /// number it did not measure, and it would silently go stale the first
    /// time the SDK changed a rung. What arrives on the wire is separately and
    /// independently taken from the decoded buffer's format description in
    /// `pixelDimensions(for:)` — so if the two ever disagree, the log will say
    /// so rather than this label quietly winning.
    var declaredSizeDescription: String {
        let size = streamingResolution.videoFrameSize
        return "\(size.width)x\(size.height)"
    }
}
#endif

/// What claim this app currently holds on the glasses camera.
///
/// ## Why this exists beside `isCaptureEngaged` and `isCaptureSessionClaimed`
///
/// Those two are `Bool`s, and a `Bool` cannot tell the two "not running" cases
/// apart — which is the defect this type was added to fix.
/// `isCaptureEngaged` excludes `.paused`, so during a **device-initiated
/// pause** both Home and World Builder flip their primary control back to
/// "Start capture", and a tap on it hits `startCameraSession()`'s
/// `guard deviceSession == nil` and does nothing observable at all: a silent
/// no-op offered as the primary action.
///
/// `.devicePaused` is not a state **this app** can leave, and the difference
/// between that and "the SDK has no such call" is one this comment used to
/// blur. Two separate facts, kept separate on purpose:
///
/// 1. **What is documented.** `docs/05-DAT-INTEGRATION.md` §104-107 and
///    `07-PLATFORM-CONSTRAINTS.md` §146 record the pause as *device-initiated*
///    — a temple or cap-touch press, or a thermal pause — and as **keeping the
///    connection alive, stopping delivery, and resuming to `.started` on its
///    own**. That is the device's behaviour, from the vendor's own docs.
///
/// 2. **What this app calls.** `GlassesConnection` calls exactly four things on
///    DAT for capture — `session.start()`, `session.stop()`, `stream.start()`
///    and `camera.stop()` — and none of them resumes a paused device.
///
/// Together those support exactly one claim: **this app has no resume to
/// offer**, which is what every screen here is written from. They do not
/// support "DAT exposes none". The previous wording — "there is no override to
/// write" — argued that from the list of calls this app happens to make, and a
/// list of one's own calls proves nothing about somebody else's SDK. If a
/// resume is ever found in DAT, what changes is this comment and the four calls
/// above; the documented device behaviour and the sentences on screen today
/// stay as they are. Until then a screen that finds `.devicePaused` says what
/// it is and waits.
///
/// Deliberately not an `Optional<Bool>` or a pair of flags: a caller that has
/// to spell `.devicePaused` cannot fall into the `default` clause that
/// swallowed this case the first time.
nonisolated enum CaptureClaim: Equatable, Sendable {
    /// No session and no stream. A Start here is a real start.
    case unclaimed
    /// A session is up, or coming up, and delivery is expected.
    case running
    /// A session is held and **the device stopped delivering**. Not "off", not
    /// "startable", and not something this app can end or override.
    case devicePaused
    /// A session is being torn down. `startCameraSession()` refuses in this
    /// window, so a Start offered here would be a refusal in waiting.
    case ending
}

/// Why a `startCameraSession()` did not proceed.
///
/// ## Why this is a value and not just a `print`
///
/// Every refusal below already logged to the console, and three of them also
/// wrote `errorMessage`. But the first one — a session already existing —
/// logged and **returned silently**, so a caller had no way to tell "already
/// running" from "the device is paused" from "it started", and a composed
/// caller such as `ObjectMemoryRecordingCoordinator` could not word a truthful
/// sentence about what had just happened. A machine-readable answer is the
/// difference between a screen that explains and one that shrugs.
///
/// The console logging is unchanged and still happens. This is carried
/// alongside it, in `GlassesConnection.lastCaptureStartRefusal`.
///
/// ## Two of these are also written by a caller, not only by this file
///
/// `startCameraSession()` is not the only producer of these values.
/// `ObjectMemoryRecordingCoordinator` reads `captureClaim` *before* it calls,
/// and for two claims it can answer the question without calling at all —
/// `.devicePaused` and `.ending`. That is deliberate: calling into a refusal
/// and then translating the refusal back is a round trip that can only lose
/// information, and for `.ending` it loses the one thing a person needs (see
/// `captureIsShuttingDown`). So a case here means "why a capture did not
/// start", **not** "what `startCameraSession()` returned".
nonisolated enum CaptureStartRefusal: Equatable, Sendable {
    /// A device session already exists. Home or World Builder started one.
    case alreadyRunning
    /// A session exists and the **device** paused it. See `CaptureClaim`:
    /// there is no way to override this and none is offered.
    case deviceHasPausedCapture
    /// A capture is being torn down, and a new one cannot be created until the
    /// teardown finishes.
    ///
    /// **`startCameraSession()` never returns this**, and that is the reason
    /// the case exists. Its `guard deviceSession == nil` reports
    /// `.alreadyRunning` in this window, which is true of the `DeviceSession`
    /// object and false of the situation a person is in: nobody owns a capture
    /// that is dying, and "a capture was already under way, so this screen left
    /// it alone" sends a wearer to look for a stream that is on its way out.
    ///
    /// A caller that reads `CaptureClaim.ending` before it calls can tell the
    /// two apart, which is why this is synthesised there rather than returned
    /// from here. `ObjectMemoryRecordingCoordinator` used to fold `.ending` in
    /// with `.running` and record "somebody else owns it" — the wearer got no
    /// camera, no refusal, and a Start that reported nothing at all.
    case captureIsShuttingDown
    /// `AutoDeviceSelector` has not yielded an eligible device yet.
    case noActiveDevice
    /// The session started and the stream could not, because camera permission
    /// is not granted.
    case cameraPermissionNotGranted
    /// DAT refused, with its own description carried verbatim.
    case datRefused(String)
}

#if DEBUG
/// A DAT error that DAT documents as **nonblocking**, seen and deliberately not
/// shown as an alert.
///
/// DAT 1.0.0 added `DeviceSessionError.dwaOutOfStuRange`: "the app and glasses
/// are on versions outside the recommended compatibility range. Not blocking.
/// The session can continue ... Prompt the user to update occasionally rather
/// than block the session." It arrives through the same session error listener
/// as the terminal errors, and that listener wrote every error to
/// `errorMessage`, which the root view presents as the modal "Something went
/// wrong" -- mid-capture, for a session that is working. So this one case is
/// logged and counted here, for Developer Tools, and nothing else changes.
nonisolated struct DATNonblockingWarning: Equatable, Sendable {
    let description: String
    var count: Int
    var lastSeen: Date
    /// Where it arrived: the session error listener, or `session.start()`.
    var source: String
}
#endif

@MainActor
final class GlassesConnection: ObservableObject {
    @Published private(set) var registrationState: RegistrationState
    @Published private(set) var devices: [DeviceIdentifier]
    @Published private(set) var cameraPermissionStatus: PermissionStatus?
    @Published var errorMessage: String?

    #if DEBUG
    /// Mirrors `MockDeviceKit.shared.isEnabled` — read back from the SDK
    /// after every call, not toggled optimistically.
    @Published private(set) var mockDeviceKitEnabled = false
    /// Mirrors `MockDeviceKit.shared.pairedDevices.isEmpty == false`.
    @Published private(set) var isMockDevicePaired = false
    /// The device returned by `pairGlasses(model:)`, retained so we can
    /// power it on and don it, matching the official sample's flow.
    private var pairedMockDevice: MockGlasses?

    // MARK: Camera session (proof-of-path milestone; DEBUG-only for now)

    @Published private(set) var deviceSessionState: DeviceSessionState = .idle
    // Explicitly qualified: this app already declares its own `StreamState`
    // (Glasses/StreamManager.swift, unrelated to DAT) which would otherwise
    // shadow MWDATCamera's type of the same name.
    @Published private(set) var cameraStreamState: MWDATCamera.StreamState = .stopped

    /// Why the last `startCameraSession()` did not proceed, or `nil` if it did.
    ///
    /// Cleared at the top of every `startCameraSession()`, beside
    /// `errorMessage` and for the same reason: a refusal from a previous
    /// attempt must not be misread as the answer to whatever the person did
    /// next. `nil` therefore means "the most recent start reached
    /// `session.start()`" — which is not the same as "a stream is running",
    /// because everything after `session.start()` is asynchronous. Liveness is
    /// still `captureClaim`.
    @Published private(set) var lastCaptureStartRefusal: CaptureStartRefusal?

    /// The rung the **next** capture session will request.
    ///
    /// Deliberately not `private(set)`: the Developer Tools picker writes it.
    /// Equally deliberately, changing it does **not** disturb a running
    /// session. `StreamConfiguration` is consumed once, by
    /// `session.addCamera(config:)`, and DAT exposes no way to renegotiate a
    /// live stream — so a control that appeared to change the rung mid-capture
    /// would be claiming something that did not happen (Rule 3, Truthful State
    /// Only). The picker says "next session" because that is the truth.
    @Published var captureResolution: CaptureResolutionPreference = .default

    @Published private(set) var frameCount: Int = 0
    /// The most recent frame decoded for Tower transmission, sampled down to
    /// `FrameRateGate.towerTargetFPS` by `frameRateGate` — not every frame,
    /// per the backpressure policy in docs/03-ROADMAP.md V0.7.
    /// `ProjectManager` observes this and forwards it to `TowerClient`;
    /// `GlassesConnection` never talks to `TowerClient` directly, preserving
    /// the boundary in docs/02-DEVELOPMENT-RULES.md Rule 1.
    @Published private(set) var latestCapturedFrame: CapturedFrame?
    /// True once `AutoDeviceSelector.activeDeviceStream()` has yielded a
    /// non-nil device. `startCameraSession()` must not call `createSession`
    /// before this is true, per Meta's documented `AutoDeviceSelector`
    /// guidance — calling earlier throws `DeviceSessionError.noEligibleDevice`.
    @Published private(set) var hasActiveDevice = false

    /// The glasses' own thermal pressure, straight from the active device's
    /// `Device.addDeviceStateListener(_:)`.
    ///
    /// Through DAT 0.9.0 this was the *only* proactive health signal DAT
    /// exposed for the device. 1.0.0 adds battery level, charging, don and
    /// hinge state to `DeviceState`; this app does not read them yet, so they
    /// stay absent rather than shown. No numeric temperature exists in either
    /// release. Everything else health-shaped is *reactive*:
    /// `DeviceSessionError` carries `thermalCritical`, `thermalEmergency`,
    /// `peakPowerShutdown` and `batteryCritical`, and 1.0.0's `StreamError`
    /// carries `thermalHot`, `peakPowerLimit` and `batteryLow`, but those
    /// arrive at the moment the stream is already dying. See
    /// docs/05-DAT-INTEGRATION.md.
    ///
    /// `nil` until the listener delivers, which is also what it stays if no
    /// device is active. Nothing here estimates or interpolates a temperature
    /// — Rule 3: unknown values remain unavailable.
    @Published private(set) var glassesThermalLevel: ThermalLevel?

    /// The device-state listener for whichever device is active. Replaced, not
    /// added to, when the active device changes — see `observeDeviceState`.
    private let deviceStateTokenBag = ListenerTokenBag()
    /// The device the listener above belongs to. A delivery already queued for
    /// the main actor when the device changed is dropped by comparing against
    /// this, so the old glasses' reading cannot land on the new ones.
    private var deviceStateIdentifier: DeviceIdentifier?

    // MARK: Motion probe (DAT 1.0.0 spike; DEBUG-only, default OFF)

    /// Whether the **next** capture session also starts DAT's experimental
    /// Motion capability, at `motionProbeConfiguration`'s rate.
    ///
    /// Follows `captureResolution`'s pattern exactly: a developer control, not
    /// a product setting; written by Developer Tools; not persisted, so it is
    /// OFF again after every relaunch; and read once, when the camera starts,
    /// so changing it never disturbs a running session. It is a spike to prove
    /// the samples arrive. Nothing here sends them anywhere, shows them to a
    /// wearer, or acts on them.
    @Published var motionProbeEnabled = false

    /// 30 Hz: DAT's "responsive head-motion input" rate. The default is 10 Hz.
    static let motionProbeConfiguration = MotionConfiguration(samplingRate: .hz30)

    /// The samples, for the Developer Tools readout. Its own object so that
    /// the IMU rate never reaches this one's observers.
    let motionProbe = MotionProbe()

    private var motion: Motion?
    private var motionSamplesTask: Task<Void, Never>?
    private let motionTokenBag = ListenerTokenBag()

    /// Whether a Motion capability is attached to the current session.
    var isMotionAttached: Bool { motion != nil }

    /// DAT's nonblocking compatibility warning, if it has been seen. Shown in
    /// Developer Tools instead of as an alert; see `DATNonblockingWarning`.
    @Published private(set) var datNonblockingWarning: DATNonblockingWarning?

    /// Fires once when the camera stream is confirmed live (`StreamState
    /// .streaming`) — the earliest point it's true that a session "has
    /// successfully started and is about to begin forwarding frames".
    /// `ProjectManager` observes this and calls `TowerClient.sendStreamStart()`;
    /// `GlassesConnection` never talks to `TowerClient` directly, preserving
    /// the boundary in docs/02-DEVELOPMENT-RULES.md Rule 1.
    let cameraStreamDidStart = PassthroughSubject<Void, Never>()
    /// Fires once when `stopCameraSession()` is invoked for an active camera
    /// stream — the point the current streaming session is ending.
    let cameraStreamDidStop = PassthroughSubject<Void, Never>()

    private var deviceSession: DeviceSession?
    private var camera: MWDATCamera.Camera?
    /// Created once, in `init()`, and retained for the object's lifetime —
    /// it must observe `devicesStream()` continuously, not just at the
    /// moment a session is requested. Matches the official sample's
    /// `CameraViewModel`, which does the same.
    private var deviceSelector: AutoDeviceSelector?
    private var activeDeviceTask: Task<Void, Never>?
    private let sessionTokenBag = ListenerTokenBag()
    private let streamTokenBag = ListenerTokenBag()

    /// Decides which captured frames are forwarded for transmission. Reset at
    /// the start of every session so cadence never carries across sessions.
    private var frameRateGate = FrameRateGate(targetFPS: FrameRateGate.towerTargetFPS)

    /// Console-logging cadence, independent of transmission: until this task
    /// the same `% 30` condition governed both, which is what held Tower
    /// delivery at 24/30 ≈ 0.8 fps. Logging stays at ~1 Hz per category
    /// because `print` on the main actor is not free at 24 Hz; the
    /// authoritative per-stage counts live in `metrics`.
    ///
    /// Time-based rather than every Nth sequence number, because a sequence
    /// stride aliases against the rate gate. At a 24 fps source the gate
    /// admits every other frame, so sequences 1, 3, 5, … are selected and
    /// 2, 4, 6, … are skipped — and an even stride like the old 30 therefore
    /// lands *only* on skipped frames. The selected-frame line, the one
    /// carrying the frame dimensions that proved the physical path works,
    /// would never print again after frame #1.
    private static let logInterval: TimeInterval = 1.0
    private var lastSelectedLogAt: TimeInterval = -.infinity
    private var lastSkippedLogAt: TimeInterval = -.infinity
    #endif

    /// Link-state listeners for every known device, rebuilt whenever the
    /// device list changes. Always compiled: `cameraPermissionStatus` is
    /// displayed in Release too, and a status stuck on "Not checked yet"
    /// there is the same lie it is in Debug.
    private let linkStateTokenBag = ListenerTokenBag()
    /// Prevents a second read being started while one is in flight, so the
    /// several availability triggers collapse to a single query.
    private var isRefreshingCameraPermission = false

    private let wearables: WearablesInterface
    /// Sender-side instrumentation. Shared with `TowerClient` via
    /// `ProjectManager`, which owns both.
    private let metrics: SenderMetrics
    private var registrationTask: Task<Void, Never>?
    private var deviceStreamTask: Task<Void, Never>?

    /// - Parameter metrics: Shared sender instrumentation. Defaults to a
    ///   private instance so callers that don't care (tests, previews) need not
    ///   supply one. Built in the body rather than as a default argument
    ///   because default arguments are evaluated outside this type's actor.
    init(
        wearables: WearablesInterface = Wearables.shared,
        metrics: SenderMetrics? = nil
    ) {
        self.wearables = wearables
        self.metrics = metrics ?? SenderMetrics()
        self.registrationState = wearables.registrationState
        self.devices = wearables.devices

        #if DEBUG
        // The object-lifetime invariant, made directly observable. Exactly one
        // of these lines must appear per app launch: this type owns the camera
        // and the device session, and its `deinit` stops both, so a second
        // construction means a SwiftUI change has started tearing the runtime
        // graph down mid-session. The smoke test used to infer this from the
        // first registration-state line, which is a weaker signal because it
        // depends on DAT emitting.
        print("[Glasses][Init] GlassesConnection created")
        #endif

        #if DEBUG
        mockDeviceKitEnabled = MockDeviceKit.shared.isEnabled
        isMockDevicePaired = !MockDeviceKit.shared.pairedDevices.isEmpty

        // Created early, per Meta's documented AutoDeviceSelector guidance:
        // "Create it before the user taps the [session] action ... otherwise
        // createSession can throw DeviceSessionError.noEligibleDevice."
        let selector = AutoDeviceSelector(wearables: wearables)
        deviceSelector = selector
        // `guard let self` **inside** the loop, not outside it. Outside, the
        // guard promotes the weak capture to a strong reference held for the
        // whole life of an unbounded `for await` — so `self` owns the task and
        // the task owns `self`, the retain cycle is closed, and the
        // `isolated deinit` below (which stops the camera and the device
        // session) can never run. Inside, the strong reference lasts one
        // iteration.
        activeDeviceTask = Task { [weak self] in
            for await activeDeviceId in selector.activeDeviceStream() {
                guard let self else { return }
                self.hasActiveDevice = activeDeviceId != nil
                print("[Glasses][Camera] activeDeviceStream changed: \(String(describing: activeDeviceId)) (hasActiveDevice=\(self.hasActiveDevice))")
                // A third chance at the read, not a guarantee of one:
                // `AutoDeviceSelector` falls back to the first *known* device
                // when none is connected, so a non-nil active device does not
                // by itself mean DAT can answer. `observeLinkStates(for:)`
                // is the trigger that does.
                if self.hasActiveDevice {
                    self.refreshCameraPermissionForAvailableDevice()
                }
                // Device state is per-device, so the observation follows
                // whichever device is active rather than being started once.
                self.observeDeviceState(for: activeDeviceId)
            }
        }
        #endif

        registrationTask = Task { [weak self] in
            for await state in wearables.registrationStateStream() {
                guard let self else { return }
                self.registrationState = state
                #if DEBUG
                print("[Glasses][Registration] state changed: \(state)")
                #endif
            }
        }

        deviceStreamTask = Task { [weak self] in
            for await devices in wearables.devicesStream() {
                guard let self else { return }
                self.devices = devices
                #if DEBUG
                print("[Glasses][Devices] devicesStream changed: count=\(devices.count) ids=\(devices)")
                #endif
                // Follows the new list's link states, and takes the earliest
                // chance at a permission read. Both are no-ops once the
                // status is known — see
                // `refreshCameraPermissionForAvailableDevice()`.
                self.observeLinkStates(for: devices)
                if !devices.isEmpty {
                    self.refreshCameraPermissionForAvailableDevice()
                }
            }
        }
    }

    isolated deinit {
        registrationTask?.cancel()
        deviceStreamTask?.cancel()
        #if DEBUG
        activeDeviceTask?.cancel()
        deviceStateTokenBag.clear()
        motionSamplesTask?.cancel()
        motion?.stop()
        camera?.stop()
        deviceSession?.stop()
        #endif
    }

    func connect() {
        guard registrationState != .registering else { return }
        Task {
            do {
                try await wearables.startRegistration()
            } catch let error as RegistrationError {
                errorMessage = error.description
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func disconnect() {
        Task {
            do {
                try await wearables.startUnregistration()
            } catch let error as UnregistrationError {
                errorMessage = error.description
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    /// Reads the current camera permission without prompting.
    ///
    /// - Parameter reportErrors: whether a failure should surface as
    ///   `errorMessage`, which the root view presents as a modal alert. `true`
    ///   for a user-initiated check, because the user is looking at the result.
    ///   `false` for the automatic read at launch: an alert nobody's action
    ///   caused is worse than a permission that stays unknown, and a failed
    ///   read simply leaves the status nil, which is displayed truthfully as
    ///   "Not checked yet".
    func checkCameraPermission(reportErrors: Bool = true) {
        Task { await readCameraPermission(reportErrors: reportErrors) }
    }

    /// The single permission read, shared by the user-initiated check and the
    /// automatic refresh so both report identically.
    private func readCameraPermission(reportErrors: Bool) async {
        do {
            cameraPermissionStatus = try await wearables.checkPermissionStatus(.camera)
            #if DEBUG
            print("[Glasses][CameraPermission] checkPermissionStatus -> \(String(describing: cameraPermissionStatus))")
            #endif
        } catch {
            #if DEBUG
            print("[Glasses][CameraPermission] check failed: \(error.localizedDescription)")
            #endif
            if reportErrors { errorMessage = error.localizedDescription }
        }
    }

    /// Re-reads camera permission at the moment DAT first becomes able to
    /// answer the question.
    ///
    /// `checkPermissionStatus(.camera)` throws `PermissionError.noDevice`
    /// ("No wearable devices have been discovered or registered") until DAT
    /// has a device to answer *for*. The automatic read in
    /// `ProjectManager.startAutomaticConnections()` runs from a SwiftUI
    /// `.task` at first view appearance, which on real hardware is reliably
    /// *before* discovery completes — the throw was observed on every cold
    /// launch on a physical iPhone. Nothing re-read it afterwards, so
    /// `cameraPermissionStatus` stayed `nil` for the whole process and
    /// `beginCameraStream` refused every session for a permission the user
    /// had already granted.
    ///
    /// Driven by device availability rather than by a one-shot edge, because
    /// no single edge is reliably the right moment:
    ///
    /// - Discovery is not enough. DAT answers a permission query from the
    ///   *connected* devices — glasses that are known but powered off or in
    ///   their case throw `.noDeviceWithConnection` instead. So the read is
    ///   also driven by link state reaching `.connected`
    ///   (`observeLinkStates(for:)`), which is Meta's documented availability
    ///   signal and the only one that guarantees an answer.
    /// - An empty→non-empty edge on `devices` can never fire at all. `init`
    ///   seeds `devices` from `wearables.devices` synchronously, so on a warm
    ///   relaunch the list is already populated and the first stream yield is
    ///   not a transition.
    ///
    /// The `nil` guard is what bounds the work instead, and bounds it
    /// absolutely: the moment a read succeeds every trigger below becomes a
    /// no-op, so glasses that connect and drop repeatedly cannot turn this
    /// into a loop. Nothing polls. A status that is already known is left
    /// alone — `ConnectionSheet`'s Re-check covers a permission changed
    /// outside the app, which is the case that needs a person anyway.
    ///
    /// This is a pure query. It presents nothing, never calls
    /// `requestPermission`, and starts no camera — a device becoming
    /// available must not activate capture, which stays behind
    /// `startCameraSession()` and one explicit tap.
    private func refreshCameraPermissionForAvailableDevice() {
        guard cameraPermissionStatus == nil, !isRefreshingCameraPermission else { return }
        isRefreshingCameraPermission = true
        #if DEBUG
        print("[Glasses][CameraPermission] device available; re-reading permission")
        #endif
        Task {
            await readCameraPermission(reportErrors: false)
            isRefreshingCameraPermission = false
        }
    }

    /// Watches every known device's link state, so the permission read above
    /// gets its chance the moment glasses actually connect.
    ///
    /// Rebuilt whenever the device list changes, and seeded from each
    /// device's current `linkState`: glasses that are already connected when
    /// the list arrives may never emit a change, and waiting for one would
    /// reintroduce exactly the "waits for an event that already happened"
    /// defect this fixes.
    private func observeLinkStates(for devices: [DeviceIdentifier]) {
        linkStateTokenBag.clear()
        for identifier in devices {
            guard let device = wearables.deviceForIdentifier(identifier) else { continue }
            handleLinkState(device.linkState)
            device.addLinkStateListener { [weak self] state in
                Task { @MainActor [weak self] in
                    self?.handleLinkState(state)
                }
            }.store(in: linkStateTokenBag)
        }
    }

    private func handleLinkState(_ state: LinkState) {
        guard state == .connected else { return }
        refreshCameraPermissionForAvailableDevice()
    }

    func requestCameraPermission() {
        Task {
            do {
                cameraPermissionStatus = try await wearables.requestPermission(.camera)
                #if DEBUG
                print("[Glasses][CameraPermission] requestPermission -> \(String(describing: cameraPermissionStatus))")
                #endif
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func nameForDevice(_ id: DeviceIdentifier) -> String {
        wearables.deviceForIdentifier(id)?.nameOrId() ?? "Unknown device"
    }

    #if DEBUG
    func toggleMockDeviceKit() {
        print("[Glasses][MockDeviceKit] toggle tapped (currently enabled=\(mockDeviceKitEnabled))")
        if mockDeviceKitEnabled {
            MockDeviceKit.shared.disable()
            print("[Glasses][MockDeviceKit] disable() called")
            pairedMockDevice = nil
            isMockDevicePaired = false
        } else {
            MockDeviceKit.shared.enable()
            print("[Glasses][MockDeviceKit] enable() called")
        }
        mockDeviceKitEnabled = MockDeviceKit.shared.isEnabled
        print("[Glasses][MockDeviceKit] authoritative isEnabled after call: \(mockDeviceKitEnabled)")
    }

    /// Pairs one mock Ray-Ban Meta device, then powers it on and dons it —
    /// the same two follow-up steps the official CameraAccess sample's
    /// MockDeviceCardViewModel performs before a mock device is usable.
    /// Does not configure a camera feed — streaming is not implemented here.
    func pairMockGlasses() {
        print("[Glasses][MockDeviceKit] pairGlasses(model: .rayBanMeta) called")
        do {
            let device = try MockDeviceKit.shared.pairGlasses(model: .rayBanMeta)
            print("[Glasses][MockDeviceKit] pairGlasses succeeded, deviceIdentifier=\(device.deviceIdentifier)")

            pairedMockDevice = device
            isMockDevicePaired = !MockDeviceKit.shared.pairedDevices.isEmpty
            print("[Glasses][MockDeviceKit] pairedDevices count: \(MockDeviceKit.shared.pairedDevices.count)")

            device.powerOn()
            print("[Glasses][MockDeviceKit] powerOn() called/completed")

            device.don()
            print("[Glasses][MockDeviceKit] don() called/completed")
        } catch {
            print("[Glasses][MockDeviceKit] pairGlasses failed: \(error.localizedDescription)")
            errorMessage = error.localizedDescription
        }
    }

    // MARK: Camera session (proof-of-path milestone)

    /// Points the paired mock device's simulated camera at the iPhone's own
    /// back camera, so DAT delivers real live frames through the same
    /// Camera/Stream API a physical glasses camera would use.
    func configureMockCameraFeed() {
        print("[Glasses][Camera][Diag] --- Configure Mock Camera Feed tapped ---")
        print("[Glasses][Camera][Diag] MockDeviceKit.shared.isEnabled = \(MockDeviceKit.shared.isEnabled)")
        let paired = MockDeviceKit.shared.pairedDevices
        print("[Glasses][Camera][Diag] MockDeviceKit.shared.pairedDevices.count = \(paired.count)")
        for device in paired {
            print("[Glasses][Camera][Diag]   pairedDevice.deviceIdentifier = \(device.deviceIdentifier)")
        }
        print("[Glasses][Camera][Diag]   power/don state: not exposed as a readable property on MockGlasses/MockDevice in any confirmed source; only write-only powerOn()/don() actions exist")
        print("[Glasses][Camera][Diag] pairedMockDevice (our retained ref) = \(String(describing: pairedMockDevice?.deviceIdentifier))")
        print("[Glasses][Camera][Diag] wearables.devices.count = \(wearables.devices.count) ids=\(wearables.devices)")
        print("[Glasses][Camera][Diag] self.devices (last devicesStream value) = \(devices.count) ids=\(devices)")
        print("[Glasses][Camera][Diag] registrationState = \(registrationState)")
        print("[Glasses][Camera][Diag] cameraPermissionStatus = \(String(describing: cameraPermissionStatus))")
        print("[Glasses][Camera][Diag] current errorMessage at tap time (may be stale from a prior action) = \(String(describing: errorMessage))")

        guard let device = pairedMockDevice else {
            print("[Glasses][Camera] configureMockCameraFeed called with no paired mock device")
            errorMessage = "Pair a mock device first"
            return
        }
        device.services.camera.setCameraFeed(cameraFacing: .back)
        print("[Glasses][Camera] mock camera feed configured: back camera")
    }

    /// Whether a capture session is currently claimed, from the tap that asks
    /// for one until it is fully torn down.
    ///
    /// The workspaces used to decide Start-vs-Stop from `cameraStreamState`
    /// alone. That leaves a window — `session.start()` has been called but
    /// the camera stream has not yet reached `.starting` — in which a session
    /// exists and the button still reads "Start capture". A tap in that
    /// window is refused by `startCameraSession()`'s `deviceSession == nil`
    /// guard and does nothing observable, which is the silent no-op the
    /// primary action must never be. Including the session state closes it:
    /// the control flips to Stop the moment a session is claimed.
    var isCaptureEngaged: Bool {
        switch deviceSessionState {
        case .starting, .started: return true
        default: break
        }
        switch cameraStreamState {
        case .streaming, .starting, .waitingForDevice: return true
        default: return false
        }
    }

    /// Whether a capture session is **claimed at all**, including the states
    /// `isCaptureEngaged` deliberately excludes.
    ///
    /// ## Why this is not `isCaptureEngaged`
    ///
    /// They answer different questions and they disagree in exactly the places
    /// that matter here. `isCaptureEngaged` answers *"should the primary button
    /// read Stop?"* — a question about whether capture is actively running. It
    /// treats `.paused` and `.stopping` as not-engaged, which is right for a
    /// button.
    ///
    /// This answers *"is a `DeviceSession` still held?"* — and during `.paused`
    /// and `.stopping` the answer is **yes**. Those two states are where a
    /// control gated on `isCaptureEngaged` would unlock while the session and
    /// camera are both still alive:
    ///
    /// - **`.paused`** is not hypothetical. `07-PLATFORM-CONSTRAINTS.md` §146
    ///   records it as device-initiated — a cap-touch or a thermal pause keeps
    ///   the connection alive, stops delivery, and resumes to `.started` on its
    ///   own. On that resume `beginCameraStream` returns immediately at its
    ///   `guard camera == nil`, so a `StreamConfiguration` chosen while paused
    ///   is **never read**, and no log line records that it was dropped.
    /// - **`.stopping`** is set synchronously by `stopCameraSession()` while
    ///   `deviceSession` stays non-nil until the `.stopped` callback runs
    ///   `cleanupCameraSession()`. In that window `startCameraSession()` still
    ///   refuses at its `guard deviceSession == nil`, so "next start" is a
    ///   start the app is currently declining to perform.
    ///
    /// Both switches are exhaustive rather than `default`-terminated. Both DAT
    /// enums are `@frozen`, so this cannot silently acquire an unhandled case —
    /// and writing every case out is what makes a future reader confront
    /// `.paused` instead of inheriting a `default` that quietly swallowed it,
    /// which is how this defect was introduced in the first place.
    var isCaptureSessionClaimed: Bool {
        Self.isCaptureSessionClaimed(
            session: deviceSessionState,
            stream: cameraStreamState
        )
    }

    /// The decision itself, lifted off the instance so it is testable.
    ///
    /// `deviceSessionState` and `cameraStreamState` are `private(set)` and are
    /// driven by DAT callbacks, so a test cannot put a real `GlassesConnection`
    /// into `.paused` — and `.paused` is precisely the case this exists to get
    /// right. A pure function over the two states can be exercised across every
    /// combination, which is the same reason `CartridgeAvailability.resolve` is
    /// a static function rather than a method.
    /// `nonisolated` because it touches no actor state — it is a function of
    /// its two arguments and nothing else. Without it the enclosing
    /// `@MainActor` propagates and a synchronous test cannot call it, which
    /// would have pushed this decision back out of reach of the suite.
    nonisolated static func isCaptureSessionClaimed(
        session: DeviceSessionState,
        stream: MWDATCamera.StreamState
    ) -> Bool {
        switch session {
        case .starting, .started, .paused, .stopping: return true
        case .idle, .stopped: break
        }
        switch stream {
        case .starting, .streaming, .waitingForDevice, .paused, .stopping: return true
        case .stopped: return false
        }
    }

    /// The camera claim, in the four-way form a caller can word a sentence
    /// from. See `CaptureClaim` for why the two `Bool`s above are not enough.
    var captureClaim: CaptureClaim {
        Self.captureClaim(session: deviceSessionState, stream: cameraStreamState)
    }

    /// Every later value of `captureClaim`, for a caller that must react to a
    /// capture ending for a reason it did not ask for.
    ///
    /// Built from the two `@Published` properties rather than from
    /// `objectWillChange`, which fires *before* the change and would therefore
    /// publish the previous claim under the new one's name. `removeDuplicates`
    /// because the two states move independently and most of their transitions
    /// leave the claim exactly where it was — a subscriber redrawing on
    /// `.starting` → `.started` is redrawing for nothing.
    var captureClaimUpdates: AnyPublisher<CaptureClaim, Never> {
        $deviceSessionState
            .combineLatest($cameraStreamState)
            .map { pair in Self.captureClaim(session: pair.0, stream: pair.1) }
            .removeDuplicates()
            .eraseToAnyPublisher()
    }

    /// The decision, lifted off the instance so it is testable — for the reason
    /// `isCaptureSessionClaimed` gives: `deviceSessionState` and
    /// `cameraStreamState` are `private(set)` and driven by DAT callbacks, so
    /// no test can put a real `GlassesConnection` into `.paused`, and `.paused`
    /// is precisely the case this exists to get right.
    ///
    /// The device's own pause is checked **first and in both enums**, because
    /// it is the reading that must never be lost: a session that is `.started`
    /// while its stream is `.paused` is a device that has stopped delivering,
    /// and reporting that as `.running` would put a live-capture indicator over
    /// a camera sending nothing.
    ///
    /// Both switches are exhaustive rather than `default`-terminated, for the
    /// reason written out at `isCaptureSessionClaimed`: a `default` is how
    /// `.paused` got swallowed in the first place.
    nonisolated static func captureClaim(
        session: DeviceSessionState,
        stream: MWDATCamera.StreamState
    ) -> CaptureClaim {
        switch session {
        case .paused:
            return .devicePaused
        case .starting, .started:
            if case .paused = stream { return .devicePaused }
            return .running
        case .stopping:
            return .ending
        case .idle, .stopped:
            break
        }
        switch stream {
        case .paused: return .devicePaused
        case .starting, .streaming, .waitingForDevice: return .running
        case .stopping: return .ending
        case .stopped: return .unclaimed
        }
    }

    /// Creates and starts a `DeviceSession` via `AutoDeviceSelector`. Once the
    /// session reaches `.started`, the state observer starts the camera
    /// stream automatically — this is the only entry point the UI needs.
    ///
    /// Requires `hasActiveDevice` to already be true (i.e. `deviceSelector`'s
    /// `activeDeviceStream()` has yielded a device). Calling `createSession`
    /// before that is confirmed to throw `DeviceSessionError.noEligibleDevice`
    /// per Meta's documented AutoDeviceSelector guidance.
    ///
    /// ## It records why it refused
    ///
    /// Every `return` below now also sets `lastCaptureStartRefusal`. Nothing
    /// else about the behaviour of the Home and World Builder call sites
    /// changes — the `print`s are the same, `errorMessage` is written in
    /// exactly the same places, and the method still returns `Void`, so both
    /// compile untouched and neither has to read the new value. What changes is that the **first** guard is no longer silent
    /// to code: a session already existing and a session the *device* has
    /// paused used to be one indistinguishable no-op, and the second of those
    /// is the case a person most needs explained.
    func startCameraSession() {
        // Scope out any stale error from a prior attempt so it can't be
        // misattributed to whatever the user does next. The refusal goes with
        // it, for the same reason.
        errorMessage = nil
        lastCaptureStartRefusal = nil
        print("[Glasses][Camera] startCameraSession called (hasActiveDevice=\(hasActiveDevice))")

        guard deviceSession == nil else {
            print("[Glasses][Camera] startCameraSession called while a session already exists")
            // A device-initiated pause holds a session open while delivering
            // nothing, and there is no way to override it — see `CaptureClaim`.
            // Told apart here rather than left as one refusal, because "a
            // capture is already running" and "the glasses paused themselves"
            // send a person to two completely different places.
            if deviceSessionState == .paused {
                print("[Glasses][Camera] the existing session is device-paused; delivery resumes on its own")
                lastCaptureStartRefusal = .deviceHasPausedCapture
            } else {
                lastCaptureStartRefusal = .alreadyRunning
            }
            return
        }
        guard hasActiveDevice, let deviceSelector else {
            print("[Glasses][Camera] startCameraSession refused: no active eligible device yet")
            lastCaptureStartRefusal = .noActiveDevice
            errorMessage = "No eligible glasses device yet. Make sure Mock Device Kit is enabled, paired, powered on, and donned, then wait a moment for the device to become active."
            return
        }

        frameCount = 0
        // Otherwise the viewfinder and the "Latest frame #N" tile keep showing
        // the previous session's image and sequence, while `frameCount` has
        // already restarted at 0 — two contradictory claims on screen at once.
        latestCapturedFrame = nil
        frameRateGate.reset()
        // So the first frame of every session logs, even one started within a
        // second of the last one ending.
        lastSelectedLogAt = -.infinity
        lastSkippedLogAt = -.infinity
        metrics.begin()
        print("[Glasses][Camera] creating session via AutoDeviceSelector (active device confirmed)")

        do {
            let session = try wearables.createSession(deviceSelector: deviceSelector)
            deviceSession = session
            observeSession(session)
            deviceSessionState = .starting
            try session.start()
            print("[Glasses][Camera] session.start() called — createSession succeeded with an eligible device present")
        } catch {
            // DAT documents `dwaOutOfStuRange` as the one version error that
            // lets a session proceed. Raised by `start()`, the session exists
            // and is starting, so it is kept and the warning goes to the log
            // and Developer Tools, not to a modal alert. Raised by
            // `createSession`, there is no session to keep, so it falls through
            // to the failure path below like every other error.
            if deviceSession != nil, Self.isNonblockingWarning(error) {
                noteNonblockingWarning("\(error)", source: "session.start()")
                return
            }
            // Untyped: `createSession` and `start()` throw `DeviceSessionError`
            // and nothing else, so `catch let error as DeviceSessionError` was
            // a test the compiler proves is always true. The clause was already
            // exhaustive — this function does not rethrow — so dropping the
            // pattern changes nothing but the warning.
            print("[Glasses][Camera] session creation/start failed: \(error.localizedDescription)")
            lastCaptureStartRefusal = .datRefused(error.localizedDescription)
            errorMessage = error.localizedDescription
            deviceSession = nil
            deviceSessionState = .idle
        }
    }

    /// Stops the camera/stream (if any) and the device session, following the
    /// official 0.9.0 teardown: `camera.stop()` for the stream, then
    /// `deviceSession.stop()` for the full session.
    func stopCameraSession() {
        print("[Glasses][Camera] stopCameraSession called")
        if let camera {
            cameraStreamState = .stopping
            camera.stop()
            // Kept, not cleared. The frame is the last thing the glasses really
            // saw, and blanking the preview on Stop would discard a true
            // observation. What must not happen is it continuing to *read* as
            // live — `ViewfinderCard` dims it and labels it with its sequence
            // number once `isStreaming` goes false, which is the honest
            // presentation of "this is the last frame, the camera is off".
            cameraStreamDidStop.send(())
            // Freezes the session's final counters so they stay readable
            // after Stop rather than being wiped or left mid-interval.
            metrics.finish()
        }
        if let deviceSession {
            deviceSessionState = .stopping
            deviceSession.stop()
        }
    }

    private func observeSession(_ session: DeviceSession) {
        session.statePublisher.listen { [weak self] state in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.deviceSessionState = state
                print("[Glasses][Camera] DeviceSessionState changed: \(state)")
                if state == .started {
                    self.beginCameraStream(on: session)
                } else if state == .stopped {
                    self.cleanupCameraSession()
                }
            }
        }.store(in: sessionTokenBag)

        session.errorPublisher.listen { [weak self] error in
            Task { @MainActor [weak self] in
                self?.handleSessionError(error)
            }
        }.store(in: sessionTokenBag)
    }

    /// Whether DAT documents this session error as nonblocking: the session
    /// continues, and the right response is a log line, not an alert. Today
    /// that is exactly `dwaOutOfStuRange`; every other case is unchanged.
    nonisolated static func isNonblockingWarning(_ error: DeviceSessionError) -> Bool {
        if case .dwaOutOfStuRange = error { return true }
        return false
    }

    /// The same, for a catch clause whose error type is not spelled out.
    nonisolated static func isNonblockingWarning(_ error: any Error) -> Bool {
        (error as? DeviceSessionError).map { isNonblockingWarning($0) } ?? false
    }

    /// The one route from the session error listener to the screen.
    ///
    /// Internal rather than private so a test can drive it: DAT's own session
    /// cannot be made to publish a chosen error. Everything but a nonblocking
    /// warning is written to `errorMessage` exactly as before.
    func handleSessionError(_ error: DeviceSessionError) {
        if Self.isNonblockingWarning(error) {
            noteNonblockingWarning(error.description, source: "session error listener")
            return
        }
        print("[Glasses][Camera] session error: \(error.localizedDescription)")
        errorMessage = error.localizedDescription
    }

    /// Logs a nonblocking DAT warning and counts it for Developer Tools. Never
    /// touches `errorMessage`, and never ends the session.
    private func noteNonblockingWarning(_ description: String, source: String) {
        let now = Date()
        var warning = datNonblockingWarning
            ?? DATNonblockingWarning(description: description, count: 0, lastSeen: now, source: source)
        warning.count += 1
        warning.lastSeen = now
        warning.source = source
        datNonblockingWarning = warning
        print("[Glasses][Camera] DAT nonblocking warning from \(source), logged and not shown as an alert (seen \(warning.count)x): \(description)")
    }

    /// Adds the camera capability and starts its stream. Requires camera
    /// permission already granted via the existing check/request controls —
    /// this milestone does not trigger the permission redirect itself.
    private func beginCameraStream(on session: DeviceSession) {
        guard camera == nil else { return }
        guard cameraPermissionStatus == .granted else {
            print("[Glasses][Camera] camera permission not granted (\(String(describing: cameraPermissionStatus))); not starting stream")
            abandonSessionAfterFailedStart(
                refusal: .cameraPermissionNotGranted,
                reason: "Camera permission is not granted. Open Connections, allow camera access for the glasses, then start capture again."
            )
            return
        }

        // Read from `captureResolution` rather than the former hardcoded
        // `.low`. The default is still `.low`, so this is a no-op unless a
        // developer deliberately changed it; see `CaptureResolutionPreference`
        // for why the control exists and why it is not a product setting.
        let requestedResolution = captureResolution
        let config = StreamConfiguration(
            videoCodec: VideoCodec.raw,
            resolution: requestedResolution.streamingResolution,
            frameRate: 24
        )
        // Logged because the rung is otherwise invisible in the console, and
        // every frame-dimension line below should be read against it. If the
        // decoded dimensions disagree with the rung DAT declared, that is a
        // finding and this pair of lines is what makes it visible.
        print("[Glasses][Camera] requesting resolution \(requestedResolution.rawValue) (DAT declares \(requestedResolution.declaredSizeDescription))")

        do {
            guard let newCamera = try session.addCamera(config: config) else {
                print("[Glasses][Camera] addCamera returned nil")
                abandonSessionAfterFailedStart(
                    refusal: .datRefused("Could not create camera"),
                    reason: "Could not create camera"
                )
                return
            }
            camera = newCamera
            print("[Glasses][Camera] addCamera succeeded")
            setupStreamListeners(for: newCamera.stream)
            cameraStreamState = .starting
            newCamera.stream.start()
            print("[Glasses][Camera] stream.start() called")
            // After the camera, and only once it is up: the camera path above is
            // the product, and nothing about Motion may stand in its way.
            if motionProbeEnabled {
                startMotionProbe(on: session)
            }
        } catch {
            print("[Glasses][Camera] addCamera failed: \(error.localizedDescription)")
            abandonSessionAfterFailedStart(
                refusal: .datRefused(error.localizedDescription),
                reason: error.localizedDescription
            )
        }
    }

    /// Ends a `DeviceSession` that started but could not begin streaming, so
    /// the next Start is a real attempt rather than a silent refusal.
    ///
    /// Every failure inside `beginCameraStream` happens *after*
    /// `session.start()` has already succeeded, which leaves `deviceSession`
    /// non-nil with no camera attached to it. `startCameraSession()`'s
    /// `guard deviceSession == nil` then rejects every later attempt, while
    /// the workspaces — which derive Start/Stop from `cameraStreamState`,
    /// still `.stopped` because the stream never started — keep offering
    /// Start and never offer Stop. Nothing can clear the session, so capture
    /// is dead for the rest of the process.
    ///
    /// This was observed on a physical iPhone: one refused start, then
    /// twelve consecutive taps that each logged "session already exists" and
    /// changed nothing on screen. It is the defect recorded as latent in
    /// `docs/agent-handoffs/product-shell-v2-handoff.md` §19; a stale camera
    /// permission is simply the first thing that reached it.
    ///
    /// Teardown is synchronous rather than waiting for the session's
    /// `.stopped` callback, because that callback is exactly what may not
    /// arrive for a session that never got going — and waiting for it is
    /// what left the invariant broken. `session.stop()` is still called, on
    /// a captured reference, so DAT releases the session too.
    ///
    /// `refusal` is carried beside `reason` rather than derived from it: the
    /// reason is a sentence written for a person and the refusal is a value
    /// written for a caller, and parsing one out of the other is how a screen
    /// ends up branching on English. Every failure that reaches here happens
    /// **after** `startCameraSession()` returned, so this is the second of the
    /// two places `lastCaptureStartRefusal` is set, and the only one that can
    /// report a permission that was not granted.
    private func abandonSessionAfterFailedStart(refusal: CaptureStartRefusal, reason: String) {
        print("[Glasses][Camera] start failed; abandoning the device session")
        lastCaptureStartRefusal = refusal
        errorMessage = reason
        let session = deviceSession
        // Clears deviceSession, camera, both token bags, and resets the
        // stream state and metrics. `camera` is nil here, so this does not
        // emit a `cameraStreamDidStop` for a stream that never started.
        cleanupCameraSession()
        deviceSessionState = .idle
        session?.stop()
    }

    private func setupStreamListeners(for stream: MWDATCamera.Stream) {
        stream.statePublisher.listen { [weak self] state in
            Task { @MainActor [weak self] in
                self?.cameraStreamState = state
                print("[Glasses][Camera] StreamState changed: \(state)")
                if case .streaming = state {
                    self?.cameraStreamDidStart.send(())
                }
            }
        }.store(in: streamTokenBag)

        stream.videoFramePublisher.listen { [weak self] frame in
            // Sampled HERE, on DAT's callback thread, before the main-actor
            // hop below. The `now` inside the Task carries main-actor queueing
            // latency as well as transport latency, so it cannot be used to ask
            // what clock the buffer's timestamp is on. Pure observation: the
            // probe reads a timestamp and prints, and nothing downstream sees
            // it. See FramePTSProbe.
            FramePTSProbe.shared.record(
                sampleBuffer: frame.sampleBuffer, hostNow: MonotonicClock.now
            )

            Task { @MainActor [weak self] in
                guard let self else { return }
                // One clock read serves both the gate and the log budgets.
                let now = MonotonicClock.now

                // Sequence numbers are DAT callback ordinals: every delivered
                // frame gets one, whether or not it is transmitted. That is
                // what lets the Tower compute the source rate from `seq` gaps,
                // and it is deliberately unchanged.
                self.frameCount += 1
                let sequence = self.frameCount
                self.metrics.recordCapture(sequence: sequence)

                guard self.frameRateGate.shouldSelect(at: now) else {
                    self.metrics.recordSkip()
                    if now - self.lastSkippedLogAt >= Self.logInterval {
                        self.lastSkippedLogAt = now
                        print("[Glasses][Camera] frame #\(sequence) skipped by the \(FrameRateGate.towerTargetFPS) fps gate")
                    }
                    return
                }
                self.metrics.recordSelection()

                // Selected frames get their own budget, so this line cannot be
                // crowded out by the skipped-frame line above.
                let shouldLog = now - self.lastSelectedLogAt >= Self.logInterval
                if shouldLog { self.lastSelectedLogAt = now }

                // Both conversions happen only for selected frames, so the
                // cost scales with the send rate rather than the capture rate.
                guard let dimensions = Self.pixelDimensions(for: frame) else {
                    self.metrics.recordDecodeFailure()
                    // Failures are counted unconditionally but logged on the
                    // same cadence as everything else: at the target rate an
                    // unguarded print here would flood the console 12 times a
                    // second. `metrics.decodeFailures` is the real record.
                    if shouldLog {
                        print("[Glasses][Camera] frame received #\(sequence) dimensions=unknown")
                    }
                    return
                }
                guard let image = frame.makeUIImage() else {
                    self.metrics.recordDecodeFailure()
                    if shouldLog {
                        print("[Glasses][Camera] frame #\(sequence) makeUIImage() returned nil")
                    }
                    return
                }
                if shouldLog {
                    print("[Glasses][Camera] frame received #\(sequence) dimensions=\(dimensions.width)x\(dimensions.height)")
                }

                self.latestCapturedFrame = CapturedFrame(
                    image: image,
                    sequence: sequence,
                    width: dimensions.width,
                    height: dimensions.height
                )
            }
        }.store(in: streamTokenBag)

        stream.errorPublisher.listen { [weak self] error in
            Task { @MainActor [weak self] in
                print("[Glasses][Camera] stream error: \(error.localizedDescription)")
                self?.errorMessage = error.localizedDescription
            }
        }.store(in: streamTokenBag)
    }

    /// Attaches DAT 1.0.0's experimental Motion capability to a started
    /// session and feeds its samples to `motionProbe`.
    ///
    /// Every failure here is logged and shown in the probe's own readout, and
    /// deliberately **not** written to `errorMessage`: that one presents a
    /// modal alert, and a developer probe must never interrupt a capture that
    /// is otherwise working. Nor does a failure end the session.
    private func startMotionProbe(on session: DeviceSession) {
        guard motion == nil else { return }
        motionProbe.reset()
        do {
            guard let newMotion = try session.addMotion(configuration: Self.motionProbeConfiguration) else {
                print("[Glasses][Motion] addMotion returned nil (session not started)")
                motionProbe.noteFailure("addMotion returned nil")
                return
            }
            motion = newMotion
            let probe = motionProbe
            newMotion.statePublisher.listen { state in
                Task { @MainActor in
                    probe.noteState(state.description)
                    print("[Glasses][Motion] MotionState changed: \(state)")
                }
            }.store(in: motionTokenBag)
            newMotion.errorPublisher.listen { error in
                Task { @MainActor in
                    probe.noteFailure(error.description)
                    print("[Glasses][Motion] motion error: \(error.description)")
                }
            }.store(in: motionTokenBag)
            // Detached so the receipt time is read as the sample leaves DAT's
            // stream, before the hop to the main actor adds its own queueing
            // — the same reason `FramePTSProbe` samples on DAT's thread. The
            // stream finishes when the session tears down; the task is also
            // cancelled in `cleanupCameraSession()`.
            let samples = newMotion.samples
            motionSamplesTask = Task.detached {
                for await sample in samples {
                    let received = MotionProbeSample(sample, receivedAt: MonotonicClock.now)
                    await probe.record(received)
                }
            }
            newMotion.start()
            print("[Glasses][Motion] motion.start() called (\(Self.motionProbeConfiguration.samplingRate))")
        } catch {
            print("[Glasses][Motion] addMotion failed: \(error.description)")
            motionProbe.noteFailure(error.description)
        }
    }

    /// Detaches the Motion probe. The session tears its capabilities down
    /// with it, so this only releases this app's references and listeners.
    private func stopMotionProbe() {
        motionSamplesTask?.cancel()
        motionSamplesTask = nil
        motionTokenBag.clear()
        guard let motion else { return }
        motion.stop()
        self.motion = nil
        motionProbe.noteState("off")
        print("[Glasses][Motion] motion probe stopped")
    }

    private func cleanupCameraSession() {
        print("[Glasses][Camera] session cleanup")
        FramePTSProbe.shared.reportFinal(reason: "session cleanup")
        let hadCamera = camera != nil
        stopMotionProbe()
        sessionTokenBag.clear()
        streamTokenBag.clear()
        deviceSession = nil
        camera = nil

        // A DAT-initiated stop — glasses folded or doffed, Bluetooth lost, a
        // stream error — arrives here without ever passing through
        // `stopCameraSession()`, which until now was the only place
        // `cameraStreamDidStop` fired. That left the Tower's stream bracket
        // open, which then suppressed the *next* session's `stream_start`
        // while `frameCount` restarted at 0 — the Tower would see sequence
        // numbers rewind inside what it still considered one session, making
        // its own seq-gap and FPS figures unreadable. `sendStreamStop()` is
        // idempotent, so the ordinary Stop path reaching here second is a
        // no-op.
        if hadCamera {
            cameraStreamDidStop.send(())
        }
        // A session that ends without stopCameraSession() (a device drop, an
        // error) must still leave truthful final counters, and must not leave
        // the gate mid-cadence for the next session.
        metrics.finish()
        frameRateGate.reset()
        // deviceSelector is intentionally NOT cleared here — it's created
        // once in init() and must keep observing activeDeviceStream() for
        // the object's lifetime, independent of individual session cycles.
        cameraStreamState = .stopped
    }

    /// Follows `Device.addDeviceStateListener(_:)` for whichever device is
    /// currently active, replacing any previous observation.
    ///
    /// DAT 1.0.0 removed `Wearables.deviceStateStream(for:)`; device state now
    /// lives on `Device`, whose listener delivers the current state at once and
    /// then every change. Restarted rather than kept because a listener is
    /// bound to one device: leaving the old one registered would keep
    /// publishing a thermal level for glasses that are no longer the ones in
    /// use, which is worse than publishing nothing.
    private func observeDeviceState(for identifier: DeviceIdentifier?) {
        deviceStateTokenBag.clear()
        deviceStateIdentifier = identifier
        // Cleared before the new observation starts, so the previous device's
        // last reading cannot linger on screen as if it described the new one.
        // It stays nil until the new listener actually delivers — and stays nil
        // if DAT has no `Device` for the identifier, as the stream did.
        glassesThermalLevel = nil

        guard let identifier, let device = wearables.deviceForIdentifier(identifier) else { return }
        // `[weak self]` throughout: the bag that owns the listener is owned by
        // `self`, so a strong capture would be a cycle.
        device.addDeviceStateListener { [weak self] state in
            Task { @MainActor [weak self] in
                guard let self, self.deviceStateIdentifier == identifier else { return }
                self.glassesThermalLevel = state.thermalLevel
                print("[Glasses][Health] glasses thermalLevel: \(state.thermalLevel)")
            }
        }.store(in: deviceStateTokenBag)
    }

    private static func pixelDimensions(for frame: VideoFrame) -> (width: Int, height: Int)? {
        guard let formatDescription = CMSampleBufferGetFormatDescription(frame.sampleBuffer) else {
            return nil
        }
        let dimensions = CMVideoFormatDescriptionGetDimensions(formatDescription)
        return (Int(dimensions.width), Int(dimensions.height))
    }
    #endif
}
