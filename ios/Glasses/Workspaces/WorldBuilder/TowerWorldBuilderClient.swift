//
//  TowerWorldBuilderClient.swift
//  Glasses
//

import Combine
import Foundation

// MARK: - The contract this build implements

/// The one World Builder agreement this build was written against.
///
/// Opaque, and compared for equality only. It is dated rather than numbered
/// precisely so that nobody is tempted to compute which is greater: a mismatch
/// means "we are not talking about the same agreement", which is neither newer
/// nor older, and `CartridgeAvailability.unsupportedContract` is the honest
/// rendering of it.
nonisolated enum WorldBuilderResultContract {
    /// The **Tower's** name for the cartridge. Not this app's catalog id
    /// (`"world-build"`); the two strings are different and the mapping lives
    /// in `TowerCapabilities`.
    static let towerCartridge = "world_builder"
    static let resultType = "status"

    /// Adopted deliberately, and **not** because a field was added.
    ///
    /// `world_builder.status/2026-08-25` supersedes `.../2026-08-23` because
    /// `trajectory.pose_count` changed *meaning*: it was
    /// `keyframes - poses_refused`, which counts a segment anchor — identity
    /// rotation at the origin, by construction — as a camera position. On the
    /// 2026-08-24 physical walk that reported 36 camera poses from a build
    /// whose manifest read `poses_solved: 0, points: 0, segments: 36`.
    ///
    /// The Tower refused the old identifier rather than quietly serving a
    /// figure that means something different, which is what put *"The Tower
    /// offers a World Builder contract this version of the app does not
    /// understand"* on the phone. `WorldTrajectoryReport` is where the new
    /// meaning is honoured; adopting the string without that would be the
    /// silent widening the refusal existed to prevent.
    static let identifier = "world_builder.status/2026-08-25"
}

// MARK: - Where the geometry lives

/// The address of the geometry the status payload is describing, plus the
/// identity that says whether it has moved.
///
/// Three strings and no optionals, because a partial address is not a weaker
/// address — it is no address at all. The Tower's manifest endpoint requires
/// `session_id`, so a world id without one cannot be fetched, and an absent
/// `geometry.revision` means the Tower has built nothing to point at rather
/// than "revision zero".
///
/// **Deliberately not folded into `WorldSnapshot`.** That type's doc comment
/// promises it maps field for field onto the payload's `world_snapshot` block,
/// and neither of these two values lives there: `session_id` is in the
/// payload's `session` block and the geometry identity is in its top-level
/// `geometry` block. Widening `WorldSnapshot` would break that promise and
/// would put transport addressing inside a presentation type.
struct WorldGeometryCoordinates: Equatable, Sendable {
    let worldID: String
    let sessionID: String
    /// `geometry.revision`, **not** the snapshot's.
    ///
    /// The snapshot revision changes whenever any reported field changes — a
    /// keyframe count, a tracking state — and most of those changes leave the
    /// built geometry exactly where it was. Keying the fetch on this one means
    /// a megabyte of points is pulled when the points moved, and not when the
    /// keyframe counter did.
    let revision: String
}

// MARK: - Payload decoding

/// Turns the Tower's `world_builder.status` payload into the types the
/// workspace already had.
///
/// ## Why the mapping is this thin
///
/// Because the Tower does it. `model_state` names a `WorldModelState` case and
/// `world_snapshot` is shaped onto `WorldSnapshot` field for field — deliberately,
/// so that the translation table lives on the machine where changing it is a
/// restart rather than an App Store release.
///
/// ## The blocks that are read, and why only those
///
/// Everything else in the payload is Tower-native evidence for those two
/// values, and a reader looking for where `lifecycle`, `progress`, `world`,
/// `tracking`, `calibration`, `persistence` or `artifacts` are consumed will
/// correctly find that they are not. Three exceptions, each for something the
/// projection cannot carry:
///
/// - **`trajectory`**, for `poses_anchor` and `segments`. The projection keeps
///   `pose_count` and drops both, and those are exactly what separates a walk
///   that positioned 36 cameras from one that produced 36 segment origins and
///   positioned none.
/// - **`session`**, for `capture_id`, `ended_at` and `frame_source`. Not a
///   figure and never drawn as one — it answers *whose world this is*, which
///   `world_snapshot` cannot, and which `WorldSessionGate` needs.
/// - **`session.session_id` and `geometry.revision`**, read by
///   `geometryCoordinates(from:)` for **addressing rather than display**. The
///   geometry itself is fetched over HTTP and those are what address it.
///   Neither is rendered.
///
/// ## What it refuses to do
///
/// Absent stays absent. Every optional here is `nil` when the key is missing or
/// null, and nothing is defaulted to zero — the contract is explicit that
/// `null ≠ 0`, and `frames_observed` being genuinely unknowable during a live
/// session is the reason it says so.
enum WorldBuilderResultDecoder {

    /// The state the payload describes, or `nil` when it could not be read as
    /// this contract at all — which the caller renders as a decode failure
    /// rather than as an empty world.
    static func modelState(from payload: [String: Any]) -> WorldModelState? {
        guard let word = payload["model_state"] as? String else { return nil }
        let reason = payload["model_state_reason"] as? String
        // `if let` rather than `Optional.map(snapshot(from:))`, for the reason
        // `TowerCartridgeDeclaration.init` gives: a function reference passed
        // to `map` is called from a nonisolated context under this target's
        // default `MainActor` isolation.
        var snapshot: WorldSnapshot?
        if let raw = payload["world_snapshot"] as? [String: Any] {
            // The payload's own `trajectory` block, which carries the two
            // figures `world_snapshot.trajectory` does not: `poses_anchor` and
            // `segments`. See `snapshot(from:trajectoryEvidence:)`.
            snapshot = self.snapshot(
                from: raw,
                trajectoryEvidence: payload["trajectory"] as? [String: Any]
            )
        }

        switch word {
        case "unsupported":
            return .unsupported(reason: reason ?? Self.unexplainedUnsupported)

        case "idle":
            return .idle

        case "receiving":
            // A live session with nothing yet said about a world is exactly
            // what `.awaitingFirstUpdate` means. Substituting an empty
            // snapshot would draw a world panel with every row missing, which
            // reads as a broken world rather than as an early one.
            guard let snapshot else { return .awaitingFirstUpdate }
            return .receiving(snapshot)

        case "finalizing":
            // The Tower's own caveat, carried as-is: this state means "the
            // stored figures are not the final figures", **not** "a process is
            // working right now". Tower cannot observe the latter — the writer
            // lock is released before the build starts — and `WorldModelState`
            // documents `.finalizing` as the case where the world on screen is
            // not yet the world that will be stored, which is the same claim.
            return .finalizing(snapshot ?? WorldSnapshot())

        case "finalized":
            return .finalized(snapshot ?? WorldSnapshot())

        case "failed":
            return .failed(
                CartridgeFailure(
                    kind: .towerReportedFailure,
                    message: reason ?? "The Tower reported that its World Builder session failed."
                )
            )

        default:
            // A `model_state` this build does not know is a contract
            // disagreement discovered on arrival, which is precisely
            // `.undecodableResponse` rather than `.notSupported`.
            return nil
        }
    }

    static let unexplainedUnsupported = """
        This Tower cannot serve World Builder, and did not say why.
        """

    /// Where the geometry this payload describes can be fetched, or `nil` when
    /// the payload does not carry all three parts of the address.
    ///
    /// All-or-nothing on purpose. `session_id` is a **required** query
    /// parameter on the Tower's manifest route, not an optional one, so a world
    /// id on its own addresses nothing; and `geometry.revision` is `null`
    /// exactly when no build has produced output for this session, which is
    /// absence and not zero. Returning a half-filled address would turn an
    /// honest "there is nothing built yet" into a request that 404s every two
    /// seconds.
    ///
    /// `world_id` is read from `world_snapshot` rather than from the top-level
    /// `world` block: the two carry the same id, and the snapshot is the half
    /// of the payload this build has already agreed to decode.
    static func geometryCoordinates(from payload: [String: Any]) -> WorldGeometryCoordinates? {
        let snapshot = payload["world_snapshot"] as? [String: Any] ?? [:]
        let session = payload["session"] as? [String: Any] ?? [:]
        let geometry = payload["geometry"] as? [String: Any] ?? [:]
        guard
            let worldID = snapshot["world_id"] as? String,
            let sessionID = session["session_id"] as? String,
            let revision = geometry["revision"] as? String
        else { return nil }
        return WorldGeometryCoordinates(
            worldID: worldID, sessionID: sessionID, revision: revision
        )
    }

    /// `world_snapshot` → `WorldSnapshot`.
    ///
    /// The three nested blocks are always present when `world_snapshot` is, so
    /// their absence here is treated as an empty report rather than as an
    /// error: a snapshot that lost its geometry block still carries a truthful
    /// keyframe count, and refusing the whole thing would show less than the
    /// Tower said.
    static func snapshot(
        from json: [String: Any],
        trajectoryEvidence: [String: Any]? = nil
    ) -> WorldSnapshot {
        let geometry = json["geometry"] as? [String: Any] ?? [:]
        let trajectory = json["trajectory"] as? [String: Any] ?? [:]
        let persistence = json["persistence"] as? [String: Any] ?? [:]
        // The Tower projects `world_snapshot.trajectory` from its own
        // `trajectory` block but carries only four of its keys across.
        // `poses_anchor` and `segments` stay behind, and they are precisely
        // what distinguishes "36 segment origins" from "36 camera poses" —
        // the distinction `world_builder.status/2026-08-25` was cut for. So
        // this one evidence block is read, and only for figures the projection
        // does not carry. `?? [:]` rather than a refusal: a payload without it
        // still has a truthful snapshot, and the two extra figures are simply
        // absent.
        let evidence = trajectoryEvidence ?? [:]

        return WorldSnapshot(
            name: json["name"] as? String,
            worldID: json["world_id"] as? String,
            keyframeCount: json["keyframe_count"] as? Int,
            revision: json["revision"] as? String,
            tracking: tracking(json["tracking"] as? String),
            scale: scale(json["scale"] as? String),
            mappingSeconds: json["mapping_seconds"] as? Double,
            calibration: calibration(json["calibration"] as? String),
            geometry: WorldGeometryReport(
                representation: geometry["representation"] as? String,
                elementCount: geometry["element_count"] as? Int,
                isIncremental: geometry["is_incremental"] as? Bool
            ),
            trajectory: WorldTrajectoryReport(
                poseCount: trajectory["pose_count"] as? Int,
                // Never folded into the count above. An anchor is definitional
                // — identity rotation, zero translation — and adding the two
                // is the arithmetic the contract moved to stop.
                posesAnchor: evidence["poses_anchor"] as? Int,
                posesSolved: evidence["poses_solved"] as? Int,
                posesRefused: evidence["poses_refused"] as? Int,
                segments: evidence["segments"] as? Int,
                pathLength: trajectory["path_length"] as? Double,
                pathLengthUnit: trajectory["path_length_unit"] as? String,
                // Carried separately from the snapshot's own scale: a spatial
                // figure travels with its own provenance, and the two can
                // legitimately differ.
                scale: scale(trajectory["scale"] as? String)
            ),
            persistence: self.persistence(
                state: persistence["state"] as? String,
                revision: persistence["revision"] as? String
            )
        )
    }

    /// The payload's `session` block → `WorldSessionReport`, or `nil`.
    ///
    /// `nil` for `"session": null`, which the Tower sends for a world that has
    /// no sessions and for a payload with no world at all. Absent, not empty:
    /// a `WorldSessionReport` with every field `nil` would claim a session
    /// exists whose capture is unknown, and the gate would then have to tell
    /// that apart from a real one.
    static func session(from payload: [String: Any]) -> WorldSessionReport? {
        guard let json = payload["session"] as? [String: Any] else { return nil }
        return WorldSessionReport(
            sessionID: json["session_id"] as? String,
            captureID: json["capture_id"] as? String,
            endedAt: json["ended_at"] as? Double,
            endReason: json["end_reason"] as? String,
            frameSource: json["frame_source"] as? String
        )
    }

    /// `limited` is in the Tower's vocabulary nowhere — it would need a
    /// threshold nobody has defined — so it is not produced today. It is still
    /// mapped, because the iOS case exists and silently folding a future
    /// `limited` into `.unavailable` would understate a real report.
    static func tracking(_ word: String?) -> WorldTrackingQuality {
        switch word {
        case "good": return .good
        case "limited": return .limited
        case "lost": return .lost
        default: return .unavailable
        }
    }

    /// The Tower sends **iOS's** scale vocabulary, not its own — `relative`,
    /// `inferredMetric`, `measuredMetric`, `unknown`. The last two have no code
    /// path that produces them on monocular hardware and will not arrive; they
    /// are mapped anyway rather than discarded, because a value that did arrive
    /// and was silently downgraded to `.relative` would understate a metric
    /// claim, and `.unknown` is a strictly weaker claim than `.relative` — it
    /// means the reconstruction has no unit at all.
    static func scale(_ word: String?) -> WorldScaleSemantics {
        switch word {
        case "relative": return .relative
        case "inferredMetric": return .inferredMetric
        case "measuredMetric": return .measuredMetric
        default: return .unknown
        }
    }

    /// `calibrating` is never sent — calibration is an offline procedure run
    /// before a session, so there is no in-session state to be in the middle
    /// of — and there is deliberately no percentage anywhere in the contract.
    static func calibration(_ word: String?) -> WorldCalibrationState {
        switch word {
        case "calibrated": return .calibrated
        case "uncalibrated": return .uncalibrated
        case "calibrating": return .calibrating
        default: return .unknown
        }
    }

    /// `saved` is the only state the Tower reaches — World Builder persists by
    /// construction, which makes `.session` unreachable rather than merely
    /// unused.
    static func persistence(state: String?, revision: String?) -> WorldPersistenceState {
        switch state {
        case "saved": return .saved(revision: revision)
        case "session": return .session
        case "reloading": return .reloading
        default: return .unknown
        }
    }
}

// MARK: - The client

/// The Tower-backed World Builder client.
///
/// ## Where it sits
///
/// ```
/// TowerClient            owns the socket, decodes the envelope, knows no cartridge
///     ↓ cartridgeResults
/// TowerWorldBuilderClient   owns the subscription and the World Builder contract
///     ↓ stateUpdates, geometryUpdates
/// WorldBuilderViewModel     republishes into SwiftUI, and fetches geometry
///                           over HTTP from the address it was handed
///     ↓
/// WorldCanvasView           renders facts
/// ```
///
/// It is constructed by `ProjectManager` and held in `CartridgeClients`, so it
/// **outlives every workspace switch** — which is the whole reason that
/// container exists. A partly-built world survives closing the cartridge, and
/// nothing here is torn down by SwiftUI.
///
/// ## What it does not own
///
/// No socket. It holds a reference to the one `TowerClient` the app already
/// has and sends three message types over it; it never opens a connection,
/// never reconnects, and never touches the frame path. `GlassesConnection` is
/// not reachable from here at all.
///
/// **And no geometry.** It publishes the *address* of the geometry the Tower
/// reports and fetches none of it. The points travel over HTTP, from the view
/// model, because the Tower gives its result sender and its frame path one
/// shared lock and a megabyte of points down this socket would starve the
/// frames.
///
/// ## Reconnect
///
/// There is no delta stream, so there is no gap, so a reconnect cannot lose
/// data: subscribe again and the first message is a complete snapshot.
/// `subscriptionID` restarts at `sub-1` on every socket, so it is cleared
/// whenever the connection leaves `.online` and re-earned on the way back.
///
/// The last known state is **not** cleared on a drop. Availability already
/// resolves to `.towerUnreachable` while the socket is down, and
/// `CartridgeAvailability.forcedPhase` makes that outrank any domain state — so
/// the screen says "disconnected" without this client having to fabricate a
/// world that stopped existing. When the socket returns, the next snapshot
/// replaces whatever was held.
@MainActor
final class TowerWorldBuilderClient: WorldBuilderClient {

    let cartridgeID = "world-build"

    /// What the workspace draws. **Already gated**: a snapshot the phone could
    /// not establish as this session's never reaches here as a result. See
    /// `WorldSessionGate`.
    private(set) var state: WorldModelState = .idle {
        didSet {
            guard state != oldValue else { return }
            log(state)
            stateSubject.send(state)
        }
    }

    /// What the phone established about the last snapshot it was sent.
    ///
    /// Published separately from `state` because the two change independently:
    /// closing the capture bracket changes the binding while the Tower's words
    /// are unchanged, and a foreign snapshot arriving changes the binding while
    /// `state` stays `.awaitingFirstUpdate`. The view needs both — one decides
    /// what is drawn, the other decides what is *said* about why.
    private(set) var sessionBinding: WorldSessionBinding = .none {
        didSet {
            guard sessionBinding != oldValue else { return }
            logBinding()
            bindingSubject.send(sessionBinding)
        }
    }

    var stateUpdates: AnyPublisher<WorldModelState, Never> {
        stateSubject.eraseToAnyPublisher()
    }

    var bindingUpdates: AnyPublisher<WorldSessionBinding, Never> {
        bindingSubject.eraseToAnyPublisher()
    }

    /// Live, or pinned to a stored world. Changes only through `inspect` and
    /// `followLive`, and is published on a real change like the two above.
    private(set) var inspection: WorldInspectionMode = .live {
        didSet {
            guard inspection != oldValue else { return }
            inspectionSubject.send(inspection)
        }
    }

    var inspectionUpdates: AnyPublisher<WorldInspectionMode, Never> {
        inspectionSubject.eraseToAnyPublisher()
    }

    /// The pin the next `result_subscribe` carries, or `nil` to follow the
    /// live world. Kept across reconnects on purpose: a reader looking at a
    /// stored world who loses WiFi is still looking at that world when it
    /// comes back.
    private var pinned: (worldID: String, sessionID: String?)?

    private let stateSubject = PassthroughSubject<WorldModelState, Never>()
    private let bindingSubject = PassthroughSubject<WorldSessionBinding, Never>()
    private let inspectionSubject = PassthroughSubject<WorldInspectionMode, Never>()
    /// The geometry address carried by every snapshot that has one — the
    /// heartbeat's included.
    ///
    /// Unfiltered on purpose, unlike `state`, whose `didSet` drops repeats.
    /// Deciding whether geometry has moved requires knowing what is already
    /// held, and what is already held is the view model's cache, not this
    /// object's. Filtering here as well would give two objects a private and
    /// separately-wrong opinion about the same revision. What is sent here is a
    /// fact — "the Tower says its geometry is at this address, under this
    /// identity" — and the reader decides whether that is news.
    var geometryUpdates: AnyPublisher<WorldGeometryCoordinates, Never> {
        geometrySubject.eraseToAnyPublisher()
    }

    private let geometrySubject = PassthroughSubject<WorldGeometryCoordinates, Never>()
    private let tower: TowerClient
    private var cancellables: Set<AnyCancellable> = []

    /// The last thing the Tower said, before the gate.
    ///
    /// Kept because the binding has two inputs and only one of them arrives on
    /// the result channel: the phone's bracket opens and closes on its own
    /// clock, and when it does the same payload has to be re-judged. Storing
    /// the decoded pair rather than the raw dictionary keeps the decode on the
    /// arrival path, where a failure is still attributable to a message.
    private var lastReport: (state: WorldModelState, session: WorldSessionReport?)?

    /// The open subscription on the **current** socket, or `nil`. Cleared on
    /// every disconnect because the Tower's ids are per connection.
    private var subscriptionID: String?

    /// Subscription ids this client has closed on the current socket, so a
    /// heartbeat the Tower had already queued for one of them is recognised
    /// and dropped. Ids are per connection and restart at `sub-1` on a new
    /// one, so the set is cleared with the connection. See `handle(.result)`.
    private var retiredSubscriptionIDs: Set<String> = []
    /// A `result_subscribe` has been sent and not yet answered. Without this a
    /// declaration republished while the ack is in flight would open a second
    /// subscription for the same cartridge.
    private var isSubscribing = false
    /// Bounds the wait for a `result_subscribed`. See `armSubscribeTimeout`.
    private var subscribeTimeout: Task<Void, Never>?
    /// Which subscribe attempt a timeout belongs to. Without it, a timeout
    /// armed for one attempt can fire into a later one that is legitimately
    /// still waiting.
    private var subscribeAttempt = 0
    /// Long enough that a congested Tailscale link mid-walk is not called
    /// dead — the whole handshake ahead of this one completed in well under a
    /// second on the physical run — and short enough that a person is not left
    /// in front of a spinner with no end. Matches the bound
    /// `ObjectMemoryHTTPClient` puts on its own requests.
    private static let defaultSubscribeAckTimeout: Duration = .seconds(10)
    /// Injectable so a test can bound the wait in milliseconds rather than
    /// spending ten seconds proving a ten-second bound exists.
    private let subscribeAckTimeout: Duration

    /// Resubscribes spent on the current connection.
    ///
    /// `consumer_too_slow` and `channel_failed` close a subscription and are
    /// recoverable by subscribing again — but a Tower that closes every
    /// subscription immediately would otherwise have this client resubscribing
    /// in a loop for as long as the socket stayed up. Bounded, and refilled by
    /// a new connection, for the same reason `TowerClient`'s reconnect schedule
    /// is bounded: a failure that will not resolve must become visible rather
    /// than stay in motion.
    private var resubscribesUsed = 0
    private static let resubscribeBudget = 3

    init(tower: TowerClient, subscribeAckTimeout: Duration? = nil) {
        self.tower = tower
        self.subscribeAckTimeout = subscribeAckTimeout ?? Self.defaultSubscribeAckTimeout

        // `.receive(on:)` on both, and it is load-bearing rather than
        // stylistic. A `@Published` publisher fires from `willSet`, so a sink
        // that reads the property it was notified about sees the value *before*
        // the change — a connection that had just come online still read
        // `.offline`, and a declaration that had just arrived still read `nil`,
        // so nothing ever subscribed. Deferring to the next main-queue turn is
        // what `WorldBuilderViewModel` and `ProjectManager`'s bridges already
        // do, for the same reason.
        tower.$status
            .removeDuplicates()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] status in self?.connectionChanged(to: status) }
            .store(in: &cancellables)

        tower.$cartridgeDeclaration
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.subscribeIfPossible() }
            .store(in: &cancellables)

        tower.cartridgeResults
            .sink { [weak self] event in self?.handle(event) }
            .store(in: &cancellables)

        // The phone's own half of the binding. `isStreamingToTower` is true
        // between a sent `stream_start` and its `stream_stop`, which is exactly
        // "this phone has a capture open" — and it is permanently false in a
        // Release build, which has no capture control and therefore never has
        // a session of its own to bind a world to.
        //
        // `.receive(on:)` for the reason the two sinks above give: a
        // `@Published` publisher fires from `willSet`, so a sink that reads the
        // property would see the value before the change.
        tower.$isStreamingToTower
            .removeDuplicates()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.rejudgeLastReport() }
            .store(in: &cancellables)
    }

    // MARK: Availability

    /// Resolved against the Tower's **live** declaration rather than the static
    /// table in `TowerCapabilities`.
    ///
    /// That table is still the right answer for the three cartridges the Tower
    /// offers no contract for. World Builder is the first one it does, and the
    /// declaration arrives over the socket — so reading a compile-time constant
    /// here would report `.noContract` against a Tower that had just said
    /// otherwise.
    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        TowerCapabilities.availability(
            for: cartridgeID,
            declaredBy: tower.cartridgeDeclaration,
            isTowerReachable: isTowerReachable
        )
    }

    // MARK: Connection lifecycle

    private func connectionChanged(to status: TowerStatus) {
        guard status == .online else {
            // The subscription belonged to a socket that is gone. Nothing is
            // sent to close it — the Tower treats a closed socket as
            // sufficient cleanup — and nothing about the world is forgotten,
            // because availability already reports the connection truthfully.
            subscriptionID = nil
            retiredSubscriptionIDs = []
            isSubscribing = false
            // The socket that the subscribe was sent on is gone, so the bound
            // has nothing left to bound. Leaving it armed would report a
            // timeout against a connection the reconnect path already owns.
            disarmSubscribeTimeout()
            return
        }
        resubscribesUsed = 0
        subscribeIfPossible()
    }

    /// Idempotent by construction: every path back into it is guarded by the
    /// same two flags, so a status change and a republished declaration racing
    /// each other cannot open two subscriptions.
    private func subscribeIfPossible() {
        guard tower.status == .online, subscriptionID == nil, !isSubscribing else { return }
        guard let declaration = tower.cartridgeDeclaration else { return }
        guard let offer = declaration.offer(forTowerCartridge: WorldBuilderResultContract.towerCartridge)
        else {
            // The Tower said nothing about World Builder. Availability already
            // renders that as `.noContract`; there is nothing to subscribe to
            // and nothing for the domain state to add.
            return
        }
        guard TowerCapabilities.supported.contains(offer.contract) else {
            // A contract this build does not implement. `.unsupportedContract`
            // availability already outranks any state, so the state is left
            // where it was rather than being given a second, weaker wording of
            // the same fact.
            return
        }
        guard offer.available else {
            // Offered and unserveable — no world root configured, typically.
            // The Tower's own prose is the only honest explanation available,
            // so it is shown verbatim.
            state = .unsupported(
                reason: offer.unavailableReason ?? WorldBuilderResultDecoder.unexplainedUnsupported
            )
            return
        }

        isSubscribing = true
        // A new subscription is answered with a complete snapshot, so whatever
        // was held describes a socket that is gone. Cleared together with the
        // state it produced, so a bracket opening in the window before that
        // snapshot arrives cannot re-publish it over the wait.
        lastReport = nil
        sessionBinding = bindingWithNoReport
        state = .awaitingFirstUpdate
        tower.subscribeToResults(
            cartridge: offer.cartridge,
            resultType: offer.resultType,
            contract: offer.contract,
            worldID: pinned?.worldID,
            sessionID: pinned?.sessionID
        )
        armSubscribeTimeout()
    }

    // MARK: Stored worlds

    /// Pin the subscription to a stored world.
    ///
    /// Closes the open subscription, forgets the report it produced, and opens
    /// a new one carrying the pin. The Tower answers a pinned subscribe with a
    /// complete snapshot of that world, so nothing is merged and nothing from
    /// the live world survives the switch. An id the Tower does not know comes
    /// back as its own `unsupported`/error wording, which the existing paths
    /// already render verbatim.
    func inspect(worldID: String, sessionID: String?) {
        pinned = (worldID, sessionID)
        inspection = .inspecting(worldID: worldID)
        restartSubscription()
    }

    /// Back to the live world, by the same route.
    func followLive() {
        pinned = nil
        inspection = .live
        restartSubscription()
    }

    /// Tear down the current subscription and open one under the current pin.
    ///
    /// The unsubscribe is best-effort and its `result_unsubscribed` is ignored
    /// by construction: `subscriptionID` is cleared here, so the ack for the
    /// old id no longer matches anything, and the new subscribe's own ack is
    /// what sets it again. The Tower treats a closed socket as sufficient
    /// cleanup anyway; this just spares it a subscription nobody is reading.
    private func restartSubscription() {
        if let id = subscriptionID {
            tower.unsubscribeFromResults(subscriptionID: id)
            retiredSubscriptionIDs.insert(id)
            subscriptionID = nil
        }
        isSubscribing = false
        disarmSubscribeTimeout()
        lastReport = nil
        subscribeIfPossible()
    }

    /// Bound the wait for the Tower's `result_subscribed`.
    ///
    /// ## Why the wait was unbounded, and what that looked like
    ///
    /// `subscribeIfPossible` sets `.awaitingFirstUpdate` — a spinner — *before*
    /// sending, and `TowerClient.sendResultMessage` deliberately does not
    /// escalate a failed send: it returns silently if the socket is not online
    /// and swallows an async send error, because a result-channel message
    /// failing must not take down the frame path. Both of those are right on
    /// their own. Together they mean a `result_subscribe` that never reaches
    /// the wire, on a socket that then stays up, leaves the workspace waiting
    /// for an answer that is not coming. Nothing cleared `isSubscribing` except
    /// leaving `.online` or an inbound error, so the spinner had no end.
    ///
    /// `CartridgeFailure.Kind.timedOut` exists for exactly this — Rule 15,
    /// bounded operations — and was never constructed anywhere in the app.
    ///
    /// Deliberately a *failure*, not a silent retry: the socket is up and the
    /// Tower is not answering a message it acknowledges within milliseconds
    /// when healthy. That is worth telling someone about rather than papering
    /// over, and the reconnect path already owns the case where the socket
    /// itself is the problem.
    private func armSubscribeTimeout() {
        subscribeAttempt += 1
        let attempt = subscribeAttempt
        let timeout = subscribeAckTimeout
        subscribeTimeout?.cancel()
        subscribeTimeout = Task { [weak self] in
            try? await Task.sleep(for: timeout)
            guard !Task.isCancelled else { return }
            self?.subscribeDidTimeOut(attempt: attempt)
        }
    }

    /// Cancel a pending bound. Called wherever the wait legitimately ends.
    private func disarmSubscribeTimeout() {
        subscribeTimeout?.cancel()
        subscribeTimeout = nil
    }

    private func subscribeDidTimeOut(attempt: Int) {
        // Three guards, and each one closes a real race: a newer attempt has
        // superseded this timeout; the ack arrived while it was sleeping; or
        // the connection went away and the reconnect path already owns the
        // state.
        guard attempt == subscribeAttempt, isSubscribing, subscriptionID == nil else { return }
        guard tower.status == .online else { return }

        isSubscribing = false
        lastReport = nil
        sessionBinding = bindingWithNoReport
        state = .failed(
            CartridgeFailure(
                kind: .timedOut,
                message: """
                    The Tower did not acknowledge the World Builder subscription \
                    within \(subscribeAckTimeout.components.seconds) seconds. \
                    The connection is still open, so this is the Tower not \
                    answering rather than the network being gone.
                    """
            )
        )
    }

    // MARK: Result channel

    private func handle(_ event: CartridgeResultEvent) {
        switch event {
        case .declaration:
            // Handled through `$cartridgeDeclaration` instead, so the cached
            // value and the trigger cannot disagree.
            break

        case .subscribed(let ack):
            guard ack.cartridge == WorldBuilderResultContract.towerCartridge else { return }
            subscriptionID = ack.subscriptionID
            isSubscribing = false
            disarmSubscribeTimeout()

        case .unsubscribed(let id):
            guard id == subscriptionID else { return }
            subscriptionID = nil

        case .result(let envelope):
            guard envelope.cartridge == WorldBuilderResultContract.towerCartridge else { return }
            // Matched on the subscription as well as the cartridge. A pin
            // change (`inspect` / `followLive`) closes one subscription and
            // opens another, and the Tower's sender may already have queued
            // a heartbeat for the old one; that envelope names a world the
            // reader has just left, and applying it would name that world
            // for the picture button — for a live world with no geometry
            // yet, permanently, since nothing else would arrive to correct
            // it. Matched against the ids this client has *left* rather
            // than for equality with the current one, so an envelope that
            // races its own `result_subscribed` is still applied.
            if let id = envelope.subscriptionID, retiredSubscriptionIDs.contains(id) { return }
            apply(envelope)

        case .failed(let error):
            guard isOurs(error) else { return }
            apply(error)
        }
    }

    /// Whether an error belongs to this cartridge.
    ///
    /// Matched on either name or subscription id because the Tower's extras are
    /// reason-dependent: `unknown_subscription` names only the subscription,
    /// and the two unsolicited errors name both. An error carrying neither is
    /// not claimed — attributing another cartridge's failure to this one would
    /// be a fabricated report about the Tower.
    private func isOurs(_ error: CartridgeResultError) -> Bool {
        if let cartridge = error.cartridge {
            return cartridge == WorldBuilderResultContract.towerCartridge
        }
        if let id = error.subscriptionID { return id == subscriptionID }
        return false
    }

    private func apply(_ envelope: CartridgeResultEnvelope) {
        // The envelope says whether it is a complete state or a delta, and
        // this build knows how to merge exactly nothing.
        //
        // `snapshot` is `true` on every envelope today — `ResultEnvelope`
        // defaults it and no Tower code overrides it — and it exists for this
        // moment: "a consumer that reads this field and finds `false` one day
        // will know to look for delta-merge rules; one that never sees the
        // field at all would quietly assume whichever it was written against."
        //
        // Read this before deleting it as speculative. Without the guard, a
        // delta does not fail loudly, it fails *silently and wrongly*: a
        // partial payload saying `model_state: "receiving"` with no
        // `world_snapshot` decodes cleanly and returns `.awaitingFirstUpdate`,
        // which collapses a fully populated world panel to "waiting for the
        // first update" with no error, no log, and nothing on screen to
        // suggest anything was missed. Refusing costs one comparison; not
        // refusing costs a wrong answer that looks like a right one.
        guard envelope.isSnapshot else {
            lastReport = nil
            sessionBinding = bindingWithNoReport
            state = .failed(
                CartridgeFailure(
                    kind: .notSupported,
                    message: """
                        The Tower sent a World Builder result marked as a partial update \
                        rather than a complete one. This build can only read complete \
                        results, so it is showing nothing rather than a world assembled \
                        from a piece it does not know how to merge.
                        """
                )
            )
            return
        }

        guard let next = WorldBuilderResultDecoder.modelState(from: envelope.payload) else {
            // A payload that could not be read is not a world of unknown
            // ownership — there is nothing to judge — so the gate is bypassed
            // and the last report is cleared rather than left to be re-judged
            // against a bracket change later.
            lastReport = nil
            sessionBinding = bindingWithNoReport
            state = .failed(
                CartridgeFailure(
                    kind: .undecodableResponse,
                    message: """
                        The Tower sent a World Builder result this build could not read. \
                        It declared contract \(envelope.contract ?? "none"), which this \
                        app implements, so the two disagree about what that contract means.
                        """
                )
            )
            return
        }
        lastReport = (next, WorldBuilderResultDecoder.session(from: envelope.payload))
        publishLastReport()

        // Sent whether or not the state changed, and whether or not the
        // geometry did. See `geometryUpdates` for why this one is not filtered
        // here. Absent when the Tower has built nothing yet, which is the
        // common case for most of a session's first seconds.
        //
        // Here rather than in `publishLastReport()` because this is a fact the
        // Tower just stated, and `publishLastReport()` also runs on a
        // *bracket* change, where the Tower has said nothing new.
        if let coordinates = WorldBuilderResultDecoder.geometryCoordinates(from: envelope.payload) {
            geometrySubject.send(coordinates)
        }
    }

    /// Re-runs the gate over the last thing the Tower said, because the phone's
    /// half of the binding changed.
    ///
    /// A no-op before the first snapshot: with nothing to judge there is
    /// nothing to say, and `state` is already whatever `subscribeIfPossible`
    /// left it as.
    private func rejudgeLastReport() {
        guard lastReport != nil else {
            sessionBinding = bindingWithNoReport
            return
        }
        publishLastReport()
    }

    /// The phone's half of the binding, as the gate should see it.
    ///
    /// While pinned to a stored world the bracket is **not** consulted: the
    /// reader asked for that world by name, so the question "is this the
    /// capture I have open?" has no bearing on whether to draw it, and the
    /// gate's answer is `.none` — the state passes through and the label says
    /// "Saved world". Following the live world, this is the capture bracket
    /// exactly as before.
    private var isCaptureBracketOpen: Bool {
        pinned == nil && tower.isStreamingToTower
    }

    /// The binding when there is nothing to judge yet.
    ///
    /// Not a hardcoded `.none`: a bracket can be open with no snapshot behind
    /// it — the seconds after Start, and the whole of a resubscribe — and
    /// `.awaiting` is the truthful word for that. Routed through the gate so
    /// there is still exactly one place that decides.
    private var bindingWithNoReport: WorldSessionBinding {
        WorldSessionGate.binding(
            isCaptureBracketOpen: isCaptureBracketOpen,
            session: nil,
            modelState: .awaitingFirstUpdate
        )
    }

    private func publishLastReport() {
        guard let report = lastReport else { return }
        let binding = WorldSessionGate.binding(
            isCaptureBracketOpen: isCaptureBracketOpen,
            session: report.session,
            modelState: report.state
        )
        // Before the state, so a subscriber woken by `stateUpdates` that reads
        // `sessionBinding` sees the binding that produced it.
        sessionBinding = binding
        // Assigned unconditionally; the `didSet` publishes only on a real
        // change. That is what keeps the ~2 s heartbeat — which re-sends an
        // unchanged snapshot to refresh the fields excluded from the revision
        // hash — from invalidating the view tree for nothing.
        state = WorldSessionGate.presented(report.state, binding: binding)
    }

    /// One line per **change**, which at the channel's ~2 Hz ceiling and with
    /// the heartbeat already filtered out by the `didSet` guard is roughly one
    /// line per keyframe. It exists for the physical session: on a real walk
    /// this is the only place the mapped state is observable, and "the Tower
    /// sent a result" (which `TowerClient` logs) is a different claim from
    /// "this is what the phone made of it".
    private func log(_ state: WorldModelState) {
        #if DEBUG
        let detail: String
        switch state {
        case .unsupported(let reason):
            detail = "unsupported — \(reason)"
        case .idle:
            detail = "idle"
        case .awaitingFirstUpdate:
            detail = "awaitingFirstUpdate"
        case .receiving(let snapshot), .finalizing(let snapshot), .finalized(let snapshot):
            let name: String
            switch state {
            case .receiving: name = "receiving"
            case .finalizing: name = "finalizing"
            default: name = "finalized"
            }
            detail = "\(name) keyframes=\(snapshot.keyframeCount.map(String.init) ?? "-")"
                + " tracking=\(snapshot.tracking.displayName)"
                + " scale=\(snapshot.scale.displayName)"
                + " calibration=\(snapshot.calibration.displayName)"
                + " geometry=\(snapshot.geometry.elementCount.map(String.init) ?? "-")"
                // Both, always, and never summed: on an uncalibrated walk the
                // pair reads `poses=0 anchors=36`, which is the whole of what
                // the 2026-08-25 contract corrected.
                + " poses=\(snapshot.trajectory.poseCount.map(String.init) ?? "-")"
                + " anchors=\(snapshot.trajectory.posesAnchor.map(String.init) ?? "-")"
                + " segments=\(snapshot.trajectory.segments.map(String.init) ?? "-")"
                + " revision=\(snapshot.revision ?? "-")"
        case .failed(let failure):
            detail = "failed(\(failure.kind.rawValue)) — \(failure.message)"
        }
        print("[Glasses][WorldBuilder] \(detail) binding=\(bindingDescription)")
        #endif
    }

    /// One line per binding change, beside the one per state change.
    ///
    /// Separate because the two move independently, and a binding flip with no
    /// state change is the interesting case on a physical walk: `.awaiting` →
    /// `.foreign` means the Tower answered with somebody else's world, and the
    /// screen says "waiting" either way.
    private func logBinding() {
        #if DEBUG
        print("[Glasses][WorldBuilder] binding=\(bindingDescription)")
        #endif
    }

    /// The binding, for the log line. Names the capture when the Tower named
    /// one, because on a physical walk "foreign" is only actionable beside
    /// *which* capture the Tower answered with.
    private var bindingDescription: String {
        switch sessionBinding {
        case .none: return "none"
        case .awaiting: return "awaiting"
        case .bound(let id): return "bound(\(id))"
        case .foreign(let id): return "FOREIGN(\(id ?? "no capture"))"
        }
    }

    private func apply(_ error: CartridgeResultError) {
        if error.closesSubscription {
            subscriptionID = nil
            isSubscribing = false
            // The Tower answered, so the wait is over however it ended. The
            // resubscribe below arms a fresh bound of its own.
            disarmSubscribeTimeout()
            guard resubscribesUsed < Self.resubscribeBudget else {
                state = .failed(
                    CartridgeFailure(
                        kind: .transport,
                        message: """
                            The Tower closed this world's result subscription \
                            \(Self.resubscribeBudget) times on one connection. \
                            Reconnecting is what resolves it.
                            """
                    )
                )
                return
            }
            resubscribesUsed += 1
            subscribeIfPossible()
            return
        }

        switch error.reason {
        case "cartridge_unavailable":
            // Offered, nothing to serve. The Tower's prose is the explanation.
            state = .unsupported(reason: error.message)
        case "contract_mismatch", "unknown_cartridge", "unknown_result_type":
            state = .failed(CartridgeFailure(kind: .notSupported, message: error.message))
        case "snapshot_failed":
            state = .failed(CartridgeFailure(kind: .towerReportedFailure, message: error.message))
        default:
            state = .failed(CartridgeFailure(kind: .transport, message: error.message))
        }
        isSubscribing = false
        // Any inbound error ends the wait: the Tower answered, even if what it
        // said was that this cannot work.
        disarmSubscribeTimeout()
    }
}
