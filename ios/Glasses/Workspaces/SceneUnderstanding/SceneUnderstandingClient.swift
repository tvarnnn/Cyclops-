//
//  SceneUnderstandingClient.swift
//  Glasses
//

import Combine
import Foundation

/// Supplies `SceneUnderstandingState` to the Scene Understanding workspace.
///
/// The fourth interaction shape: a state that is **replaced wholesale** and
/// never accumulates. World Builder pushes one accumulating artefact,
/// Experimental CV Lab runs a bounded job, Document Memory answers point
/// queries and keeps what it recorded, and this one describes a moment and
/// keeps nothing. All four sit behind `CartridgeClient` for the one question
/// they share and behind their own protocol for everything else — which is what
/// keeps the shared layer at a handful of small files instead of a framework.
@MainActor
protocol SceneUnderstandingClient: CartridgeClient {
    var state: SceneUnderstandingState { get }

    /// Every reading after the one `state` held when the view model was built.
    ///
    /// **This is the cartridge whose real client emits fastest**, so whatever
    /// conforms to it should coalesce before it publishes. The Tower already
    /// does most of that work: the payload is published at the standard 0.5 s
    /// poll with a 2 s heartbeat, **not at frame rate**, and `observed_at`,
    /// `staleness_seconds` and every `frames_*` counter are excluded from
    /// `revision` so an unchanged scene coalesces on the wire. What arrives
    /// here is therefore already at a rate a `@Published` property can carry.
    var stateUpdates: AnyPublisher<SceneUnderstandingState, Never> { get }
}

extension SceneUnderstandingClient {
    var stateUpdates: AnyPublisher<SceneUnderstandingState, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }

    /// The Scene screen appeared or disappeared. A default no-op, so a client
    /// with nothing to subscribe to (the stub) needs no code for it.
    func workspaceVisibilityChanged(isVisible: Bool) {}
}

// MARK: - The Tower-backed client

/// The real Scene Understanding client.
///
/// ## Where it sits
///
/// ```
/// TowerClient                   owns the socket, decodes the envelope, knows no cartridge
///     ↓ cartridgeResults
/// TowerSceneUnderstandingClient owns the subscription and this contract
///     ↓ stateUpdates
/// SceneUnderstandingViewModel   republishes into SwiftUI
///     ↓
/// SceneUnderstandingWorkspaceView renders counts, and the disclosure they are owed
/// ```
///
/// ## What it sends: nothing but a subscription
///
/// No `stream_start`, no `POST /scene/start`, no control message of any kind.
/// `IOS-to-Tower.md` §6.2 is explicit that opening a cartridge on the phone
/// sends nothing and a Tower-side test asserts the wire stays silent. The
/// session follows the camera stream, which the app already opens and closes
/// for its own reasons — see `SceneUnderstandingContract.phoneSendsNothingNote`
/// for the two measured reasons a Start button here would be worse than
/// useless.
///
/// ## What it refuses to keep: everything, the moment a session ends
///
/// This is the one place this app differs from `TowerWorldBuilderClient` on
/// purpose. That client deliberately keeps its last world across a socket drop,
/// because a partly-built world is still true when the connection returns and
/// availability already renders the disconnection.
///
/// **A scene is not.** A scene held past the end of a session is a claim about
/// a room the wearer has left, and no staleness number makes it safe, because a
/// client that draws counts above a staleness line shows the room first. So
/// this client discards on all three of:
///
/// - `lifecycle.state == "stopped"` — the Tower has already discarded its own
///   copy (`scene_available: false`, `counts`/`where`/`people` null), and
///   `SceneUnderstandingState.forReading` maps it to `.idle`;
/// - the socket leaving `.online` — a disconnect *ends the session* here, since
///   the session follows the stream. World Builder's "keep it, availability
///   explains" reasoning does not transfer;
/// - `lifecycle.session_id` changing — two payloads from two tracking sessions
///   must not be compared, and holding one across the boundary is the mildest
///   form of comparing them.
///
/// Pause is the deliberately different case and is the one thing that *is*
/// kept: `.lastKnown`, with its age, visually apart from observing.
@MainActor
final class TowerSceneUnderstandingClient: SceneUnderstandingClient {

    let cartridgeID = "scene-understanding"

    private(set) var state: SceneUnderstandingState = .idle(nil) {
        didSet {
            guard state != oldValue else { return }
            stateSubject.send(state)
        }
    }

    var stateUpdates: AnyPublisher<SceneUnderstandingState, Never> {
        stateSubject.eraseToAnyPublisher()
    }

    private let stateSubject = PassthroughSubject<SceneUnderstandingState, Never>()
    private let tower: TowerClient
    private var cancellables: Set<AnyCancellable> = []

    /// The open subscription on the **current** socket, or `nil`. The Tower's
    /// ids restart at `sub-1` on every connection, so nothing survives a drop.
    private var subscriptionID: String?
    /// A `result_subscribe` is in flight. Without this, a declaration
    /// republished while the ack is outstanding opens a second subscription for
    /// the same cartridge.
    private var isSubscribing = false
    /// Resubscribes spent on this connection, and the cap.
    ///
    /// `channel_failed` and `consumer_too_slow` are the two errors the Tower
    /// sends **unsolicited**, after which the subscription is gone and a new
    /// `result_subscribe` is the only way to resume — the Tower says exactly
    /// that in the message text. Clearing the flags without resubscribing
    /// leaves the socket up, the cartridge silent, and nothing to retrigger it:
    /// `subscribeIfPossible()` is only reached from the declaration sink and
    /// from going `.online`, and neither fires on a mid-connection error.
    ///
    /// `consumer_too_slow` is not exotic — it fires when this phone does not
    /// accept a result inside the send timeout, which is a backgrounded or
    /// thermally-throttled phone, i.e. the normal state of a device on a walk.
    ///
    /// Bounded rather than unlimited, and matched to `TowerWorldBuilderClient`
    /// deliberately: a Tower that closes a subscription three times on one
    /// connection is not going to be fixed by a fourth attempt, and an
    /// unbounded retry against the Tower's 8-subscription cap is how a client
    /// starves its own siblings.
    private var resubscribesUsed = 0
    private static let resubscribeBudget = 3
    /// The tracking session the held reading belongs to. See the type note.
    private var heldSessionID: Int?
    /// Whether the Scene Understanding screen is on screen right now.
    ///
    /// **This is what starts and stops the detector on the Tower.** Since
    /// 2026-09-07 a Tower runs a scene session only while somebody is
    /// STREAMING and somebody is WATCHING, and a `result_subscribe` for the
    /// live scene is what "watching" means. Subscribing at connection time —
    /// what this client did before — kept a people detector running behind
    /// World Builder's and the CV Lab's camera for as long as the app was
    /// connected. So the subscription is opened when this screen appears and
    /// closed when it disappears; the socket, the declaration and the
    /// resubscribe budget all still gate it, but none of them can open it
    /// while this is `false`.
    private var workspaceVisible = false

    init(tower: TowerClient) {
        self.tower = tower

        // `.receive(on:)` on both, and it is load-bearing rather than
        // stylistic: a `@Published` publisher fires from `willSet`, so a sink
        // that reads the property it was notified about sees the value *before*
        // the change. `TowerWorldBuilderClient` documents the same trap.
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
    }

    // MARK: Availability

    /// Resolved against the Tower's **live** declaration.
    ///
    /// It goes through `CartridgeAvailability.resolve` — the shared precedence —
    /// with this cartridge's own contract rather than through
    /// `TowerCapabilities.supported`, because that table is owned by the
    /// integration layer and this client must not reach across a boundary to
    /// register itself in it. When the integration layer adds
    /// `"scene-understanding": SceneUnderstandingContract.towerCartridge` and
    /// the identifier to `supported`, the two answers agree by construction —
    /// both read the same declaration and the same constant.
    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        CartridgeAvailability.resolve(
            declared: declaredContract,
            supported: [SceneUnderstandingContract.identifier],
            isTowerReachable: isTowerReachable,
            // This build implements the contract, so a silent socket means
            // "nobody has asked yet", not "no such contract" — the exact
            // case `knownToThisBuild` exists for, and the one this client
            // used to get wrong on a cold launch.
            knownToThisBuild: true
        )
    }

    private var declaredContract: CartridgeContract? {
        guard
            let offer = tower.cartridgeDeclaration?
                .offer(forTowerCartridge: SceneUnderstandingContract.towerCartridge)
        else { return nil }
        return CartridgeContract(cartridgeID: cartridgeID, identifier: offer.contract)
    }

    // MARK: Connection lifecycle

    private func connectionChanged(to status: TowerStatus) {
        guard status == .online else {
            subscriptionID = nil
            isSubscribing = false
            // A fresh connection gets a fresh budget: the cap exists to stop a
            // hopeless retry loop within one socket, not to remember a bad
            // socket forever.
            resubscribesUsed = 0
            // The discard that World Builder does not do. A dropped socket ends
            // the session that produced this scene — `stream_stop` or a
            // disconnect is how a wearable's session normally ends — so the
            // counts on screen stop describing anything. `.idle(nil)` rather
            // than `.idle(reading)`: keeping the reading would keep its
            // `scene_unavailable_reason`, which would say "stopped" about a
            // session the Tower may still consider running.
            heldSessionID = nil
            state = .idle(nil)
            return
        }
        subscribeIfPossible()
    }

    /// Idempotent by construction: every path into it is guarded by the same
    /// two flags, so a status change and a republished declaration racing each
    /// other cannot open two subscriptions.
    private func subscribeIfPossible() {
        guard workspaceVisible else { return }
        guard tower.status == .online, subscriptionID == nil, !isSubscribing else { return }
        guard let declaration = tower.cartridgeDeclaration else { return }
        guard
            let offer = declaration.offer(
                forTowerCartridge: SceneUnderstandingContract.towerCartridge
            )
        else {
            // The Tower said nothing about Scene Understanding. Availability
            // renders that as `.noContract`; there is nothing to subscribe to
            // and nothing for the domain state to add.
            return
        }
        guard offer.contract == SceneUnderstandingContract.identifier else {
            // A contract this build does not implement. `.unsupportedContract`
            // availability already outranks any state, so the state is left
            // where it is rather than given a second, weaker wording of the
            // same fact.
            return
        }
        guard offer.resultType == SceneUnderstandingContract.resultType else {
            // The one cartridge whose result type is not `status`. If the
            // declaration ever disagrees with this build about that, sending
            // the subscribe anyway earns an `unknown_result_type` refusal and
            // an empty screen with no explanation; refusing here at least
            // leaves the availability wording intact.
            return
        }
        guard offer.available else {
            // Offered and unserveable — `TOWER_SCENE_UNDERSTANDING` unset,
            // typically. The Tower's own prose names the variable that would
            // fix it, so it is shown verbatim rather than paraphrased.
            state = .unsupported(
                reason: offer.unavailableReason ?? Self.unexplainedUnavailable
            )
            return
        }

        isSubscribing = true
        // A new subscription is answered with a complete snapshot, so whatever
        // was held describes a socket that is gone.
        heldSessionID = nil
        state = .idle(nil)
        tower.subscribeToResults(
            cartridge: offer.cartridge,
            resultType: offer.resultType,
            contract: offer.contract
        )
    }

    static let unexplainedUnavailable = """
        This Tower cannot serve Scene Understanding, and did not say why.
        """

    // MARK: Visibility

    func workspaceVisibilityChanged(isVisible: Bool) {
        guard isVisible != workspaceVisible else { return }
        workspaceVisible = isVisible
        if isVisible {
            subscribeIfPossible()
            return
        }
        // Leaving the screen ends the subscription, which — with no other
        // watcher — ends the Tower's session and releases its detector. The
        // unsubscribe is best-effort and its `result_unsubscribed` is ignored
        // by construction: `subscriptionID` is cleared here, so the ack for
        // the old id matches nothing. The Tower treats a closed socket as
        // sufficient cleanup anyway; this spares it a session nobody is
        // reading. The reading is discarded for the same reason a stop
        // discards it: nothing is refreshing it any more.
        if let id = subscriptionID {
            tower.unsubscribeFromResults(subscriptionID: id)
        }
        subscriptionID = nil
        isSubscribing = false
        heldSessionID = nil
        state = .idle(nil)
    }

    // MARK: Result channel

    private func handle(_ event: CartridgeResultEvent) {
        switch event {
        case .declaration:
            // Handled through `$cartridgeDeclaration` instead, so the cached
            // value and the trigger cannot disagree.
            break

        case .subscribed(let ack):
            guard ack.cartridge == SceneUnderstandingContract.towerCartridge else { return }
            subscriptionID = ack.subscriptionID
            isSubscribing = false

        case .unsubscribed(let id):
            guard id == subscriptionID else { return }
            subscriptionID = nil
            // The subscription is gone, so nothing is refreshing this scene.
            // Holding it would leave a live-looking count that stopped being
            // fed, which is the same failure as caching across a stop.
            heldSessionID = nil
            state = .idle(nil)

        case .result(let envelope):
            guard envelope.cartridge == SceneUnderstandingContract.towerCartridge else { return }
            apply(envelope)

        case .failed(let error):
            guard isOurs(error) else { return }
            apply(error)
        }
    }

    /// Whether an error belongs to this cartridge.
    ///
    /// Matched on either name or subscription id because the Tower's extras are
    /// reason-dependent: `unknown_subscription` names only the subscription and
    /// the two unsolicited errors name both. An error carrying neither is not
    /// claimed — attributing another cartridge's failure to this one would be a
    /// fabricated report about the Tower.
    private func isOurs(_ error: CartridgeResultError) -> Bool {
        if let cartridge = error.cartridge {
            return cartridge == SceneUnderstandingContract.towerCartridge
        }
        if let id = error.subscriptionID { return id == subscriptionID }
        return false
    }

    private func apply(_ envelope: CartridgeResultEnvelope) {
        guard envelope.isSnapshot else {
            // This build knows how to merge exactly nothing. A delta would not
            // fail loudly — a partial payload decodes cleanly into a scene with
            // fewer things in it, which is a *quieter room* rendered with no
            // error and nothing on screen to suggest anything was missed.
            heldSessionID = nil
            state = .failed(
                CartridgeFailure(
                    kind: .notSupported,
                    message: """
                        The Tower sent a Scene Understanding result marked as a partial \
                        update rather than a complete one. This build can only read \
                        complete readings, so it is showing nothing rather than a scene \
                        assembled from a piece it does not know how to merge.
                        """
                )
            )
            return
        }

        guard let reading = SceneUnderstandingDecoder.reading(from: envelope.payload) else {
            heldSessionID = nil
            state = .failed(
                CartridgeFailure(
                    kind: .undecodableResponse,
                    message: """
                        The Tower sent a Scene Understanding result this build could not \
                        read. It declared contract \(envelope.contract ?? "none"), which \
                        this app implements, so the two disagree about what that contract \
                        means. Nothing is shown rather than a room assembled from a \
                        payload that was not understood.
                        """
                )
            )
            return
        }

        // Two payloads with different `session_id` came from different tracking
        // sessions and must not be compared. Nothing here carries anything
        // across that boundary — the payload is a complete snapshot and
        // `forReading` builds the state from it alone — so what this tracks is
        // the boundary itself, for the disconnect and unsubscribe paths above
        // to clear against and for anything that later wants to know a session
        // changed rather than merely that the counts did.
        heldSessionID = reading.lifecycle.sessionID
        state = SceneUnderstandingState.forReading(reading)
    }

    private func apply(_ error: CartridgeResultError) {
        if error.closesSubscription {
            subscriptionID = nil
            isSubscribing = false
            // The Tower closed this subscription and its own message says to
            // subscribe again to resume. Nothing else will: `subscribeIfPossible()`
            // is reached only from the declaration sink and from going `.online`,
            // and the socket is still up, so without this the cartridge goes
            // permanently silent on a live connection.
            guard resubscribesUsed < Self.resubscribeBudget else {
                state = .failed(
                    CartridgeFailure(
                        kind: .transport,
                        message: """
                            The Tower closed this cartridge's result subscription \
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
        // Any error of ours ends an in-flight subscribe attempt.
        //
        // `closesSubscription` answers a different question -- "is an
        // ESTABLISHED subscription now gone?" -- and is true for only two
        // reasons. Using it alone to clear `isSubscribing` left the flag
        // stuck `true` for every other refusal, and `subscribeIfPossible()`
        // guards on `!isSubscribing`, so **the cartridge never retried for
        // the life of the connection**.
        //
        // That is reachable on transients, not just on permanent faults:
        // the Tower sends `snapshot_failed` INSTEAD of `result_subscribed`
        // when the first snapshot raises, which is exactly the case that
        // must recover. `contract_mismatch`, `unknown_cartridge`,
        // `cartridge_unavailable` and `too_many_subscriptions` all behaved
        // the same way.
        //
        // Clearing it unconditionally is correct because the Tower answers
        // a `result_subscribe` with either an ack or an error: once an
        // error of ours arrives, nothing is in flight any more.
        isSubscribing = false
        heldSessionID = nil
        state = .failed(
            CartridgeFailure(
                kind: error.reason == "cartridge_unavailable" ? .notSupported : .towerReportedFailure,
                message: error.message
            )
        )
    }
}

// MARK: - The stub, for a Tower that declares nothing

/// The client for a Tower that has declared no Scene Understanding contract.
///
/// It produces **no sample scene**. A demo reading with two people in it would
/// be indistinguishable on screen from a real observation of whoever is
/// actually in the room, and it would be this app asserting that the glasses
/// had detected people. That is the single most consequential fake datum in
/// this codebase, and it is not shipped, not behind a flag, and not in
/// `#if DEBUG`.
///
/// ## The sentence that used to be here, and why it was deleted
///
/// > "The Tower does not analyse scenes yet. Its only reply to a frame is a
/// > single brightness measurement, so nothing about anyone the glasses pass
/// > ever reaches this app."
///
/// Every clause of that is now false. The Tower has a detector, a tracker and a
/// typed live contract; its reply to a frame is no longer one brightness
/// figure; and a count of the people in front of the wearer does reach this app
/// when the cartridge is enabled. Leaving it on screen would have been a
/// privacy assurance about bystanders, on the one screen whose subject is
/// bystanders, that stopped being true.
///
/// What replaced it says only what is observable from here: this Tower has
/// declared nothing. It makes no claim about what any Tower stores, because
/// this app has no channel through which it could know that.
@MainActor
final class UnavailableSceneUnderstandingClient: SceneUnderstandingClient {
    static let reason = """
        This Tower has not declared a Scene Understanding contract, so there is \
        nothing to subscribe to and no scene to show. That is a statement about \
        what this Tower offers, not about what is in front of you.
        """

    let cartridgeID = "scene-understanding"

    let state: SceneUnderstandingState =
        .unsupported(reason: UnavailableSceneUnderstandingClient.reason)

    init() {}
}

// MARK: - View model

/// Publishes Scene Understanding state into SwiftUI.
///
/// Holds no runtime references, for the reason given on `WorldBuilderViewModel`.
/// That matters more here than anywhere else: a view model that held the camera
/// and was destroyed on a cartridge switch would end a live session, and this
/// is the cartridge whose subject matter most invites someone to wire it
/// straight to the frame stream.
@MainActor
final class SceneUnderstandingViewModel: ObservableObject {
    @Published private(set) var state: SceneUnderstandingState

    private let client: any SceneUnderstandingClient
    private var cancellables: Set<AnyCancellable> = []

    /// No default argument — see `WorldBuilderViewModel.init(client:)`.
    init(client: any SceneUnderstandingClient) {
        self.client = client
        self.state = client.state

        client.stateUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in self?.state = state }
            .store(in: &cancellables)
    }

    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        client.availability(isTowerReachable: isTowerReachable)
    }

    /// The screen appeared or disappeared. Forwarded to the client, which is
    /// what opens and closes the live subscription — and with it the Tower's
    /// session.
    func workspaceVisibilityChanged(isVisible: Bool) {
        client.workspaceVisibilityChanged(isVisible: isVisible)
    }

    func phase(isTowerReachable: Bool) -> CartridgePhase {
        availability(isTowerReachable: isTowerReachable).forcedPhase ?? state.phase
    }

    func unavailableExplanation(isTowerReachable: Bool) -> String {
        availability(isTowerReachable: isTowerReachable)
            .explanation(cartridgeName: "Scene Understanding", clientReason: clientReason)
    }

    private var clientReason: String? {
        switch state {
        case .unsupported(let reason): return reason
        case .failed(let failure): return failure.message
        case .idle, .awaitingFirstScene, .observing, .lastKnown: return nil
        }
    }
}
