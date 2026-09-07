//
//  ProjectManager.swift
//  Glasses
//
//  Created by Tristan Varner on 8/18/26.
//

import Combine
import Foundation

#if DEBUG
// For `MWDATCamera.StreamState`, read when deciding whether a returning Tower
// connection needs its stream bracket reopened.
import MWDATCamera
#endif

/// Root app-level state container. Owns the connection, stream, and tower
/// managers so the dashboard has a single source of truth to observe.
@MainActor
final class ProjectManager: ObservableObject {
    let glassesConnection: GlassesConnection
    let streamManager: StreamManager
    let towerClient: TowerClient

    /// Sender-side instrumentation for the whole capture → transmit path.
    /// `ProjectManager` owns it and hands the same instance to both halves,
    /// which is what makes end-to-end counts (captured vs. selected vs. sent
    /// vs. replied) comparable within one session.
    let senderMetrics: SenderMetrics

    /// Object Memory's composed recording lifecycle: the Tower's producer
    /// and the glasses camera, driven by one Start button.
    ///
    /// **Owned here for the reason `CartridgeClients` exists**, and it is the
    /// first thing in this app for which that reason is not a precaution. The
    /// coordinator holds two pieces of state a cartridge switch must not
    /// destroy:
    ///
    /// - **whether this app started the capture that is running.** A workspace
    ///   `@StateObject` is destroyed the moment a person opens Home to look at
    ///   something. Rebuilt on their return it would believe it had started
    ///   nothing, and its Stop would then leave the glasses recording — the
    ///   exact failure of the one control a wearer has over being remembered.
    /// - **an action in flight.** Start is a POST, then a camera call, then a
    ///   bounded convergence; navigating away in the middle of that would
    ///   cancel it and leave the two halves disagreeing.
    ///
    /// The Product Shell V2 handoff §11 states the rule and even names this
    /// case: *"Anything that must outlive the workspace (accumulated geometry,
    /// an object-memory buffer) belongs on `ProjectManager`, not in a view's
    /// `@StateObject`."*
    let objectMemoryRecording: ObjectMemoryRecordingCoordinator

    /// The same camera owner, for Document Memory's Start and Stop. Owned here
    /// rather than resolved in the view so both cartridges start the one
    /// capture Home and World Builder can see and stop. `nil` in Release,
    /// like the coordinator's.
    let documentMemoryCamera: (any ObjectMemoryCaptureOwner)?

    /// The four cartridge clients.
    ///
    /// Owned here rather than in the workspace views because a workspace's
    /// `@StateObject` is destroyed on every cartridge switch, and a
    /// Tower-backed client will hold a subscription and whatever it has
    /// accumulated. See `CartridgeClients` for the full argument, and the
    /// Product Shell V2 handoff §11 for the rule it follows.
    ///
    /// **This adds no runtime ownership.** The clients that exist today are
    /// stateless constants that connect to nothing; the container is here so
    /// the correct wiring is already the only wiring available on the day a
    /// real client needs it.
    let cartridgeClients: CartridgeClients

    /// iPhone-side thermal, power and battery telemetry, so a rate that decays
    /// over a long run can be attributed to the device throttling rather than
    /// only to the network. The glasses' own thermal level comes from
    /// `glassesConnection.glassesThermalLevel`, which keeps DAT behind its
    /// boundary (docs/02-DEVELOPMENT-RULES.md Rule 1).
    let deviceHealth: DeviceHealth

    /// Guards `startAutomaticConnections()` against running twice.
    ///
    /// The call site is a SwiftUI `.task`, which re-runs whenever its view's
    /// identity changes — and the whole point of automation is that nobody is
    /// watching it. Idempotence lives here rather than at the call site so it
    /// cannot be lost by a later change to the view hierarchy.
    private var hasStartedAutomaticConnections = false

    /// Retains the frame/lifecycle bridges below.
    ///
    /// This used to *also* fan every child's `objectWillChange` into this
    /// object's own publisher, so that a view observing `ProjectManager` would
    /// re-render on any child change. That fan-in has been removed, and its
    /// absence is deliberate.
    ///
    /// It re-broadcast `GlassesConnection.frameCount` at the 24 Hz capture
    /// rate, which meant the root view's `body` — and therefore the whole
    /// shell — was re-evaluated 24 times a second during a session. That cost
    /// lands on the main actor, which is the actor the send window's completion
    /// handlers hop back to in order to release their slots; the sender's
    /// achievable rate is `capacity / slotLifetime`, and slot lifetime includes
    /// that hop. A presentation convenience was therefore paying out of the
    /// sender's throughput budget.
    ///
    /// Nothing observes `ProjectManager` any more. Every view takes the
    /// specific children it needs and observes those, so each re-renders on its
    /// own data and no faster.
    private var cancellables: Set<AnyCancellable> = []

    /// Injected children keep whatever `SenderMetrics` instance they were
    /// built with, so a test can wire its own; the default graph shares one.
    init(
        glassesConnection: GlassesConnection? = nil,
        streamManager: StreamManager? = nil,
        towerClient: TowerClient? = nil,
        senderMetrics: SenderMetrics? = nil,
        cartridgeClients: CartridgeClients? = nil
    ) {
        let metrics = senderMetrics ?? SenderMetrics()
        self.senderMetrics = metrics
        self.glassesConnection = glassesConnection ?? GlassesConnection(metrics: metrics)
        self.streamManager = streamManager ?? StreamManager()
        // `autoReconnect` is opted into here rather than defaulted on inside
        // `TowerClient`, so that unit tests asserting "a dropped connection
        // settles at .failed" keep asserting about a settled value. In the app
        // the opposite is wanted: a mid-session drop on a remote Tailscale
        // path is the expected case, and until now it ended Tower delivery for
        // good — nothing in the app called `connect()` again except the
        // developer's own button. The stream-bracket reopening wired up below
        // was already written for a reconnect that could not happen.
        let tower = towerClient ?? TowerClient(metrics: metrics, autoReconnect: true)
        self.towerClient = tower

        // Built after the connection and from it, which is the ordering the
        // whole container was created for: the Tower-backed World Builder
        // client holds a subscription to the result channel and whatever world
        // the Tower has reported, and both must outlive every cartridge switch.
        // It owns no socket — it sends three message types over the one
        // `TowerClient` already has — so this adds a subscriber, not a
        // transport.
        //
        // Four Tower-backed clients now, not one. The three that joined it in
        // the 2026-08-27 unification each subscribe to the same result channel
        // over the same `TowerClient`, so this still adds subscribers rather
        // than transports — but the outlive-the-switch argument above now has
        // real weight behind it rather than being a precaution: a CV Lab run in
        // flight, a paused scene's last-known reading and a document session
        // are all state a cartridge switch must not destroy.
        //
        // Object Memory is deliberately absent from this list because its
        // client is constructed by `CartridgeClients` itself — it reaches the
        // Tower over HTTP and needs no socket handed to it.
        self.cartridgeClients = cartridgeClients
            ?? CartridgeClients(
                worldBuilder: TowerWorldBuilderClient(tower: tower),
                experimentalCV: TowerExperimentalCVClient(tower: tower),
                documentMemory: TowerDocumentMemoryClient(tower: tower),
                sceneUnderstanding: TowerSceneUnderstandingClient(tower: tower)
            )

        // After the clients, and from the same connection every other screen
        // uses. There is no second `GlassesConnection` to hand it: a capture
        // this coordinator starts is the capture Home and World Builder can
        // see, and the one they can stop.
        #if DEBUG
        let cameraOwner: (any ObjectMemoryCaptureOwner)? = self.glassesConnection
        #else
        // `GlassesConnection`'s whole capture surface is DEBUG-only, exactly as
        // Home's and World Builder's controls are, so in Release there is no
        // camera to hand over. The Tower half still works and the copy says the
        // camera half cannot be reached.
        let cameraOwner: (any ObjectMemoryCaptureOwner)? = nil
        #endif
        self.objectMemoryRecording = ObjectMemoryRecordingCoordinator(
            camera: cameraOwner,
            client: self.cartridgeClients.objectMemory
        )
        self.documentMemoryCamera = cameraOwner

        let health = DeviceHealth()
        self.deviceHealth = health

        #if DEBUG
        // Bridges captured camera frames to the Tower. GlassesConnection and
        // TowerClient never reference each other directly — ProjectManager,
        // which already owns both, is the integration point. Preserves the
        // boundary in docs/02-DEVELOPMENT-RULES.md Rule 1.
        self.glassesConnection.$latestCapturedFrame
            .compactMap { $0 }
            .receive(on: DispatchQueue.main)
            .sink { [weak self] frame in
                self?.towerClient.sendFrame(
                    frame.image,
                    width: frame.width,
                    height: frame.height,
                    sequence: frame.sequence
                )
            }
            .store(in: &cancellables)

        // Bridges the V0.7 stream lifecycle markers the same way.
        self.glassesConnection.cameraStreamDidStart
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.towerClient.sendStreamStart()
                // Battery notifications fire at most once a minute, so without
                // an explicit read the health figures shown against a session
                // can predate it.
                self?.deviceHealth.refresh()
            }
            .store(in: &cancellables)

        self.glassesConnection.cameraStreamDidStop
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.towerClient.sendStreamStop()
            }
            .store(in: &cancellables)

        // Re-opens the stream bracket after the Tower connection is replaced.
        //
        // `cameraStreamDidStart` fires once per *camera* session, but the
        // Tower socket has a shorter life: a drop tears it down and
        // `stream_start` does not survive it. Without this, a single
        // mid-session blip — the expected case on a remote Tailscale path —
        // would leave the Tower pill green while every remaining frame was
        // silently discarded for want of a bracket, recoverable only by
        // stopping and restarting the camera. `sendStreamStart()` is a no-op
        // when a bracket is already open, so an ordinary connect during an
        // idle camera is unaffected.
        self.towerClient.$status
            .filter { $0 == .online }
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                guard let self, self.glassesConnection.cameraStreamState == .streaming else { return }
                self.towerClient.sendStreamStart()
            }
            .store(in: &cancellables)
        #endif
    }

    // MARK: Automatic connection

    /// Brings up the infrastructure the app can establish on its own, once per
    /// process.
    ///
    /// **This does not start the camera and cannot.** Capture begins in exactly
    /// one place, `GlassesConnection.startCameraSession()`, reached from one
    /// button; and even an open socket transmits nothing, because `sendFrame`
    /// additionally requires a `stream_start` that only a live camera stream
    /// emits. Auto-connect ≠ auto-stream is a property of the pipeline's shape
    /// here, not a promise this method keeps.
    ///
    /// Two calls, chosen because they are the only two that are honest to make
    /// without the user asking:
    ///
    /// - `checkCameraPermission()` is a pure query. It presents nothing and
    ///   changes no authorization. It also fixes a real defect: nothing
    ///   populated `cameraPermissionStatus` automatically, so it began every
    ///   launch as "Not checked yet" and the first session of each launch was
    ///   refused for a permission the user had already granted. Reading it is
    ///   *more* truthful than not reading it.
    ///
    /// - `connectIfIdle()` opens the Tower socket. Deliberately not
    ///   `connect()`: that means "the user asked to retry", refills the bounded
    ///   reconnect budget and will replace a live connection. Code running on
    ///   its own initiative gets neither privilege, so a Tower that has given
    ///   up stays visibly failed until a person intervenes.
    ///
    /// Deliberately absent: `connect()` (Meta AI registration hands off to
    /// another app and re-registers an already-registered user),
    /// `requestCameraPermission()` (a context-free prompt at launch is how
    /// permissions get denied), and anything touching the camera. Registration
    /// and device state need no call at all — `GlassesConnection` already
    /// follows those streams from `init`.
    func startAutomaticConnections() {
        guard !hasStartedAutomaticConnections else { return }
        hasStartedAutomaticConnections = true

        // `reportErrors: false` because this runs with nobody watching. A DAT
        // failure here would otherwise write `errorMessage`, which the root
        // view presents as a modal "Something went wrong" — an unexplained
        // alert on first launch, attributable to nothing the user did. The
        // failure still leaves `cameraPermissionStatus` unset, which the
        // Connections sheet reports honestly as "Not checked yet".
        glassesConnection.checkCameraPermission(reportErrors: false)
        towerClient.connectIfIdle()
    }
}
