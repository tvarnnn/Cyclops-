//
//  ExperimentalCVClient.swift
//  Glasses
//

import Combine
import Foundation

/// Supplies `ExperimentalCVState` to the Experimental CV Lab workspace.
///
/// ## A different interaction shape from World Builder's, on purpose
///
/// World Builder's client publishes a state that is always current, because a
/// world is being built continuously. This one has *commands*: an experiment is
/// chosen and started, paused, resumed, stopped. That difference is why there
/// is no shared `CartridgeDataSource<T>` protocol above both — the shared layer
/// stops at the question all four answer identically (may this be used, and why
/// not), and each cartridge describes its own interaction below that line.
///
/// Every command is `throws` and returns nothing. Each reports its outcome
/// through `state`, and that is not a stylistic choice: **the Tower answers a
/// start `accepted` before the experiment has loaded**, so a command that
/// returned an outcome would have to either block for up to two minutes or
/// return a success that has not happened yet. The outcome arrives as state, on
/// the next status document. `throws` covers only the narrower failure of not
/// being able to make the request at all.
@MainActor
protocol ExperimentalCVClient: CartridgeClient {
    var state: ExperimentalCVState { get }

    /// Every state after the one `state` held when the view model was built.
    /// See `WorldBuilderClient.stateUpdates` for why this is a concrete
    /// `AnyPublisher` rather than an `ObservableObject` conformance.
    var stateUpdates: AnyPublisher<ExperimentalCVState, Never> { get }

    /// The whole status document, when one has been read.
    ///
    /// Kept beside `state` rather than folded into it because several of its
    /// fields are **not** properties of the run: `source` is Tower-wide,
    /// `available` is the catalog whatever is running, and `selected` outlives
    /// a stop. A state enum that carried them would have to carry them in every
    /// case.
    var status: CVLabStatus? { get }

    /// Every status document after the one `status` held at construction.
    var statusUpdates: AnyPublisher<CVLabStatus?, Never> { get }

    /// The most recent refusal, or `nil` if the last command was accepted.
    ///
    /// Separate from `state` so a refused request does not erase whatever the
    /// workspace was already showing — which is exactly what the contract asks
    /// for, since a refusal changes nothing and carries the unchanged document.
    var lastRefusal: CVLabControlRefusal? { get }

    /// The command this client has sent and the Tower has not yet answered,
    /// or `nil`. See `CVLabPendingCommand` for what sets and clears it.
    ///
    /// Not folded into `state`, for the same reason `lastRefusal` is not: a
    /// question in flight changes nothing on the Tower, and the state is the
    /// Tower's.
    var pendingCommand: CVLabPendingCommand? { get }

    /// Every change to `pendingCommand` after the value held at construction.
    var pendingCommandUpdates: AnyPublisher<CVLabPendingCommand?, Never> { get }

    /// Failures discovered **after** a command was sent — today, exactly one:
    /// the reply bound passing with no answer.
    ///
    /// A publisher rather than a thrown error because the command's `throws`
    /// covers only "could not be sent", and this is discovered ten seconds
    /// later, on nobody's call stack. The view model files it where a refused
    /// send goes, so a person reads both in one place.
    var commandFailures: AnyPublisher<CartridgeFailure, Never> { get }

    /// Whether this build can issue commands at all.
    ///
    /// `false` where the Tower has not offered the cartridge, where the socket
    /// is down, or where this build has no camera to feed the experiment it
    /// would start. The workspace asks before drawing a control, because a
    /// button that cannot work is worse than no button.
    var canSendCommands: Bool { get }

    /// Asks the Tower to select and arm an experiment.
    ///
    /// One call, because the wire has one message: `cv_lab_start` selects
    /// **and** arms, replacing whatever ran. There is no separate select, and
    /// this app must not invent a two-step interaction over a one-step
    /// protocol.
    func run(_ experiment: CVExperiment) throws

    /// Stops processing and keeps the experiment loaded.
    func pause() throws
    /// Resumes a paused run.
    func resume() throws
    /// Ends the run and freezes its figures.
    func stop() throws
}

extension ExperimentalCVClient {
    var stateUpdates: AnyPublisher<ExperimentalCVState, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }

    /// Defaults for a client that has read no document and can send nothing.
    ///
    /// They exist so that a client which only reports a state — the unavailable
    /// stand-in below, and the fixtures in the test target — does not have to
    /// restate four refusals that all say the same thing. A client that can
    /// genuinely do any of it overrides all of them together.
    var status: CVLabStatus? { nil }

    var statusUpdates: AnyPublisher<CVLabStatus?, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }

    var lastRefusal: CVLabControlRefusal? { nil }

    /// A client that sends nothing has nothing pending, ever.
    var pendingCommand: CVLabPendingCommand? { nil }

    var pendingCommandUpdates: AnyPublisher<CVLabPendingCommand?, Never> {
        Just<CVLabPendingCommand?>(nil).eraseToAnyPublisher()
    }

    var commandFailures: AnyPublisher<CartridgeFailure, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }

    var canSendCommands: Bool { false }

    func pause() throws { throw Self.noCommandChannel }
    func resume() throws { throw Self.noCommandChannel }
    func stop() throws { throw Self.noCommandChannel }

    /// `.notSupported`, not `.towerReportedFailure`: the Tower reported
    /// nothing. There may not even be a socket open, and attributing a local
    /// refusal to the other machine is a fabricated claim about it.
    private static var noCommandChannel: CartridgeFailure {
        CartridgeFailure(
            kind: .notSupported,
            message: """
                This app has no command channel to the Experimental CV Lab on \
                this Tower, so nothing can be started, paused or stopped from \
                here.
                """
        )
    }
}

/// The stand-in for a Tower that does not offer the Experimental CV Lab.
///
/// ## The sentence that used to be here, and why it had to go
///
/// This type shipped a refusal that read, in both of its two compiled forms:
///
/// > what does not exist yet is a way to list the experiments, request one, or
/// > read a result with provenance attached
///
/// **All three of those exist.** `cv_lab_status` lists the experiments with
/// their ids, names, summaries and per-experiment availability; `cv_lab_start`
/// requests one; and every `frame_result` carries a `cv_lab` block whose
/// `provenance` field is *required and never omitted*. The sentence was written
/// against a Tower that listed `experimental_cv` under `not_offered`, and it
/// went on being shipped after that stopped being true — which is precisely the
/// failure mode Rule 3 exists to prevent, told to a wearer in the app's own
/// voice.
///
/// So the string is gone, and so is the claim behind it. What is left is the
/// one thing this type can still honestly say: **this Tower has not offered the
/// cartridge to this connection.** That is a statement about a declaration, not
/// about the platform, and it is true exactly when it is used — this client is
/// the fallback `CartridgeClients` constructs when nothing Tower-backed was
/// injected.
///
/// It still declares **no experiments**, and that has not changed: the Tower is
/// the registry, and a picker populated from a hardcoded list is a claim that
/// those specific experiments exist.
@MainActor
final class UnavailableExperimentalCVClient: ExperimentalCVClient {
    /// Compiled once, not twice.
    ///
    /// The two versions this used to have differed in a clause about whether
    /// *this build* sends frames — which mattered when the sentence was about
    /// reading per-frame results. It is not any more: not having been offered
    /// the cartridge is a fact about the Tower's declaration, and a Release
    /// build and a Debug build are told exactly the same thing about it. The
    /// build-specific half of the truth belongs where it is build-specific,
    /// which is the workspace's own copy about frames.
    static let reason = """
        This Tower has not offered the Experimental CV Lab on this connection, \
        so there is no catalog of experiments to list and nothing to start. A \
        Tower that offers it declares `experimental_cv` in its capabilities and \
        answers `cv_lab_status` with everything it can run.
        """

    let cartridgeID = ExperimentalCVContract.catalogID

    let state: ExperimentalCVState = .unsupported(reason: UnavailableExperimentalCVClient.reason)

    init() {}

    /// Always throws. A silent no-op would leave a button that appears to work,
    /// which is the failure mode this whole cartridge layer exists to prevent —
    /// and `docs/04-MODULE-SYSTEM.md` requires an unsupported request to
    /// "produce a clear degraded/failed state rather than silently pretending"
    /// it was applied.
    func run(_ experiment: CVExperiment) throws {
        throw CartridgeFailure(kind: .notSupported, message: Self.reason)
    }
}

// MARK: - The Tower-backed client

/// The Experimental CV Lab client that speaks to a live Tower.
///
/// ## Where it sits
///
/// ```
/// TowerClient                   owns the socket; decodes envelopes, gates on run_id
///     ↓ cvLabEvents, cartridgeResults
/// TowerExperimentalCVClient     owns the subscription, the commands, and the contract
///     ↓ stateUpdates, statusUpdates
/// ExperimentalCVViewModel       republishes into SwiftUI
///     ↓
/// ExperimentalCVWorkspaceView   renders facts
/// ```
///
/// Constructed by `ProjectManager` and held in `CartridgeClients`, so it
/// outlives every workspace switch. It owns no socket, opens no connection, and
/// never touches the frame path.
///
/// ## Why it both subscribes and sends
///
/// Because **a client that sends commands and does not also read status will
/// never learn that a start failed.** There is no `start_failed` message and
/// there cannot be one: an arm is asynchronous — the whole reason a start
/// returns immediately — so by the time a load fails, the command has already
/// been answered `accepted`, and a second reply to a reply is not a thing this
/// wire has. `lifecycle.state` becomes `failed` with a reason, and it arrives
/// on the subscription or on the next `cv_lab_status`. The subscription is
/// therefore not an optimisation; it is how half the outcomes are observed at
/// all.
///
/// The two channels are also *different transports for different things*, and
/// the Tower is deliberate about which carries which: the **document** is
/// published on the result channel, and the **commands** are not. `tower/results/`
/// is a read-only reporting surface, so a mutation must not travel on it — the
/// CV Lab's commands are plain socket messages instead, and there is no HTTP
/// surface for any of them.
///
/// ## State comes from the document, never from the command
///
/// Nothing here moves the state machine on the way *out* of `run`, `pause`,
/// `resume` or `stop`. Every reply, accepted or refused, carries the whole
/// status document, and that document is the only thing that writes `state`.
/// A refusal means *the request did not take effect*, with no partial
/// application, so a client that had already applied it optimistically would
/// have to undo something the Tower never did.
@MainActor
final class TowerExperimentalCVClient: ExperimentalCVClient {

    let cartridgeID = ExperimentalCVContract.catalogID

    private(set) var state: ExperimentalCVState = .idle(available: []) {
        didSet {
            guard state != oldValue else { return }
            stateSubject.send(state)
        }
    }

    private(set) var status: CVLabStatus? {
        didSet { statusSubject.send(status) }
    }

    private(set) var lastRefusal: CVLabControlRefusal?

    /// The command sent and not yet answered. `settlePending(...)` and
    /// `clearPending()` below are its only writers, so the reply bound and the
    /// value can never disagree about whether something is in flight.
    private(set) var pendingCommand: CVLabPendingCommand? {
        didSet {
            guard pendingCommand != oldValue else { return }
            pendingSubject.send(pendingCommand)
        }
    }

    var stateUpdates: AnyPublisher<ExperimentalCVState, Never> {
        stateSubject.eraseToAnyPublisher()
    }

    var statusUpdates: AnyPublisher<CVLabStatus?, Never> {
        statusSubject.eraseToAnyPublisher()
    }

    var pendingCommandUpdates: AnyPublisher<CVLabPendingCommand?, Never> {
        pendingSubject.eraseToAnyPublisher()
    }

    var commandFailures: AnyPublisher<CartridgeFailure, Never> {
        commandFailureSubject.eraseToAnyPublisher()
    }

    /// Whether a command may be sent right now.
    ///
    /// Three conditions, and the third is the one that is easy to miss.
    ///
    /// 1. The socket is up. A command sent to a closed socket is logged and
    ///    dropped by `TowerClient` — deliberately, so a control message cannot
    ///    cost the frame path — which means an unreachable Tower would swallow
    ///    a start silently.
    /// 2. The Tower offered the cartridge under a contract this build
    ///    implements. Anything else and there is no agreement to send under.
    /// 3. **This build has a camera.** The frame path is `#if DEBUG`, so a
    ///    Release build sends no frame and can never receive a `frame_result` —
    ///    an experiment armed from here would sit `running` and measure
    ///    nothing, forever, and the screen would be offering to start something
    ///    it cannot feed. The read-only half stays fully available in Release,
    ///    which is exactly what the contract versions the control vocabulary
    ///    separately for.
    ///
    /// Note what condition 3 is **not**: it is not this *client* acquiring a
    /// session control. Nothing in this file calls `startCameraSession()`. The
    /// workspace now carries a `CVCameraCard`, and that card reaches the camera
    /// the way Home and World Builder do — one button, the one
    /// `GlassesConnection`, no `.onAppear` — so the invariant that the app
    /// never starts the camera on its own is untouched. This is the narrower
    /// statement that a build with no camera must not offer to arm an
    /// experiment for it.
    var canSendCommands: Bool {
        #if DEBUG
        return tower.status == .online && isOfferedUnderASupportedContract
        #else
        return false
        #endif
    }

    private let stateSubject = PassthroughSubject<ExperimentalCVState, Never>()
    private let statusSubject = PassthroughSubject<CVLabStatus?, Never>()
    private let pendingSubject = PassthroughSubject<CVLabPendingCommand?, Never>()
    private let commandFailureSubject = PassthroughSubject<CartridgeFailure, Never>()
    private let tower: TowerClient
    private var cancellables: Set<AnyCancellable> = []

    /// The sleep that ends the current pending command if nothing else does.
    /// Cancelled by every clear, so a reply that arrives in time leaves no
    /// timer behind to fire against the *next* command.
    private var replyBoundTask: Task<Void, Never>?
    /// `Self.replyBoundSeconds`, except in a test that cannot wait ten seconds.
    private let replyBound: TimeInterval

    /// The open subscription on the **current** socket, or `nil`. The Tower's
    /// ids restart at `sub-1` on every connection, so nothing keeps one across
    /// a drop.
    private var subscriptionID: String?
    /// A `result_subscribe` has been sent and not yet answered. Without this a
    /// declaration republished while the ack is in flight would open a second
    /// subscription for the same cartridge.
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

    /// Monotonic, per connection, and used only to build a `request_id`.
    ///
    /// Bounded by construction: `"cv-1"` reaches the Tower's 64-character limit
    /// somewhere past `10^60` commands. `TowerClient` bounds it again on the
    /// way out, because a length that is only enforced at the call site is
    /// enforced by whoever remembers.
    private var commandCounter = 0

    /// - Parameter replyBoundSeconds: how long a sent command may go unanswered
    ///   before it is given up on. Overridable for tests only; every production
    ///   construction takes the default.
    init(
        tower: TowerClient,
        replyBoundSeconds: TimeInterval = TowerExperimentalCVClient.replyBoundSeconds
    ) {
        self.tower = tower
        self.replyBound = replyBoundSeconds

        // `.receive(on:)` on the two `@Published` sources, and it is
        // load-bearing rather than stylistic: a `@Published` publisher fires
        // from `willSet`, so a sink that reads the property it was notified
        // about sees the value *before* the change. `TowerWorldBuilderClient`
        // documents the same trap at length.
        tower.$status
            .removeDuplicates()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] status in self?.connectionChanged(to: status) }
            .store(in: &cancellables)

        tower.$cartridgeDeclaration
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.subscribeIfPossible() }
            .store(in: &cancellables)

        tower.cvLabEvents
            .sink { [weak self] event in self?.handle(event) }
            .store(in: &cancellables)

        tower.cartridgeResults
            .sink { [weak self] event in self?.handle(event) }
            .store(in: &cancellables)

        // The phone's own half of "is this live". `isStreamingToTower` is true
        // between a sent `stream_start` and its `stream_stop`, and it is
        // permanently false in a Release build — so a bracket opening or
        // closing changes what this screen may claim, without a single byte
        // arriving from the Tower.
        tower.$isStreamingToTower
            .removeDuplicates()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.republishState() }
            .store(in: &cancellables)
    }

    // MARK: Availability

    /// Resolved against the Tower's **live** declaration.
    ///
    /// Reading the static table in `TowerCapabilities` here would report
    /// `.noContract` against a Tower that had just declared
    /// `experimental_cv.status/2026-08-27` over the socket.
    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        TowerCapabilities.availability(
            for: cartridgeID,
            declaredBy: tower.cartridgeDeclaration,
            isTowerReachable: isTowerReachable
        )
    }

    /// Whether the Tower's declaration names this cartridge under a contract
    /// this build implements **and** says it can serve it.
    private var isOfferedUnderASupportedContract: Bool {
        guard let offer else { return false }
        return TowerCapabilities.supported.contains(offer.contract) && offer.available
    }

    private var offer: TowerCartridgeOffer? {
        tower.cartridgeDeclaration?.offer(
            forTowerCartridge: ExperimentalCVContract.towerCartridge
        )
    }

    // MARK: Commands

    func run(_ experiment: CVExperiment) throws {
        // Refused here rather than sent and refused by the Tower, because the
        // Tower's own refusal is `experiment_unavailable` and this app already
        // holds the answer: the catalog entry says so, per experiment, and it
        // said so at the moment the row was drawn.
        guard experiment.isStartable else {
            throw CartridgeFailure(
                kind: .notSupported,
                message: experiment.unavailableReason
                    ?? "This Tower cannot run \(experiment.name)."
            )
        }
        try guardCommandChannel()
        let requestID = nextRequestID()
        tower.sendCVLabStart(experimentID: experiment.id, requestID: requestID)
        // Recorded after the send rather than before it, and the order is not
        // cosmetic: `guardCommandChannel()` can still throw between the two,
        // and a pending command for a message that never went out would sit on
        // screen until the reply bound cleared it. `TowerClient` reports
        // nothing back — its send path is fire-and-forget by design, so the
        // control plane cannot cost the frame path — which is what the reply
        // bound is for: a message the socket dropped is indistinguishable here
        // from one the Tower is slow to answer, and both end the same way.
        settlePending(.start, experimentID: experiment.id, requestID: requestID)
    }

    func pause() throws {
        try guardCommandChannel()
        let requestID = nextRequestID()
        tower.sendCVLabPause(runID: currentRunID, requestID: requestID)
        settlePending(.pause, experimentID: nil, requestID: requestID)
    }

    func resume() throws {
        try guardCommandChannel()
        let requestID = nextRequestID()
        tower.sendCVLabResume(runID: currentRunID, requestID: requestID)
        settlePending(.resume, experimentID: nil, requestID: requestID)
    }

    func stop() throws {
        try guardCommandChannel()
        let requestID = nextRequestID()
        tower.sendCVLabStop(runID: currentRunID, requestID: requestID)
        settlePending(.stop, experimentID: nil, requestID: requestID)
    }

    /// Asks the Lab to describe itself, without changing anything.
    ///
    /// Not gated on `canSendCommands`: this is the read-only half, and a
    /// Release build with no camera is entitled to it. Only the socket has to
    /// be up.
    func refreshStatus() {
        guard tower.status == .online else { return }
        tower.sendCVLabStatusRequest(requestID: nextRequestID())
    }

    /// The run a button was drawn against, sent with every command that takes
    /// one.
    ///
    /// > A stale `run_id` is refused `stale_run` rather than applied to
    /// > whichever run is current.
    ///
    /// Which is the entire reason to send it. Two operators share one Lab slot
    /// and last start wins, so a Stop drawn against a run that has since been
    /// replaced would otherwise end *the replacement*, on somebody else's
    /// behalf. Sending the id turns that into a refusal naming the current run.
    private var currentRunID: String? { status?.lifecycle.runID }

    private func guardCommandChannel() throws {
        guard canSendCommands else {
            throw CartridgeFailure(kind: .notSupported, message: commandChannelRefusal)
        }
    }

    /// Says which of the three conditions failed, because they call for
    /// completely different responses from a person.
    private var commandChannelRefusal: String {
        #if DEBUG
        if tower.status != .online {
            return "The Tower is not connected, so the Experimental CV Lab cannot be asked to do anything."
        }
        guard let offer else { return UnavailableExperimentalCVClient.reason }
        guard TowerCapabilities.supported.contains(offer.contract) else {
            return """
                The Tower offers the Experimental CV Lab under an agreement this \
                version of the app does not understand (\(offer.contract)). \
                Updating the app is what resolves this.
                """
        }
        return offer.unavailableReason
            ?? "The Tower says it cannot serve the Experimental CV Lab right now."
        #else
        return """
            This build has no camera, so it sends the Tower no frames and an \
            experiment started from here would measure nothing. It can read what \
            the Lab is doing, and cannot ask it to do anything.
            """
        #endif
    }

    /// A short, unique token so a reply can be matched to the button that was
    /// pressed — which is the entire purpose of the field and why the Tower
    /// bounds it at 64 characters.
    private func nextRequestID() -> String {
        commandCounter += 1
        return "cv-\(commandCounter)"
    }

    // MARK: The command in flight

    /// How long a sent command may go unanswered before this client stops
    /// waiting for it.
    ///
    /// The Tower answers every command **immediately** — a start is answered
    /// `accepted` before the experiment loads, which is the whole reason the
    /// outcome arrives as state — so a reply that has not come in ten seconds
    /// is not a slow reply. It is a lost one: a socket that closed between the
    /// guard and the send, a message the Tower never received, or a reply that
    /// did not survive the trip. Ten seconds, not the 120 s arm timeout,
    /// because this bounds the *reply* and the arm is reported by the document.
    ///
    /// `nonisolated` so it can be a default argument: default arguments are
    /// evaluated in the caller's context, and an immutable `Sendable` value
    /// needs no isolation to read.
    nonisolated static let replyBoundSeconds: TimeInterval = 10

    /// Records a command as in flight and starts its reply bound.
    ///
    /// A second command while one is pending **replaces** it rather than being
    /// refused here. The workspace disables its controls while something is
    /// pending, so this is reached only by code, and a client that refused
    /// would be inventing a `lab_busy` the Tower did not send. The superseded
    /// command's reply, if it comes, no longer matches and clears nothing.
    private func settlePending(_ command: CVLabCommand, experimentID: String?, requestID: String) {
        replyBoundTask?.cancel()
        pendingCommand = CVLabPendingCommand(
            command: command,
            experimentID: experimentID,
            requestID: requestID,
            sentAt: Date()
        )
        let bound = replyBound
        replyBoundTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(bound * 1_000_000_000))
            guard !Task.isCancelled else { return }
            self?.replyBoundElapsed(for: requestID)
        }
    }

    /// Clears the pending command if, and only if, `replyRequestID` names it.
    ///
    /// Called with every `cv_lab_status` and `cv_lab_error`, most of which
    /// name nothing: the document is pushed on the result channel and read on
    /// connect, and neither carries a request id. `CVLabPendingCommand`
    /// explains why those must not clear it.
    private func clearPending(ifAnsweredBy replyRequestID: String?) {
        guard let pending = pendingCommand, pending.isAnswered(by: replyRequestID) else { return }
        clearPending()
    }

    private func clearPending() {
        replyBoundTask?.cancel()
        replyBoundTask = nil
        pendingCommand = nil
    }

    /// The reply bound passed with no answer. Cleared, and reported as the
    /// request failure it is — **not** as a state change. Nothing is known
    /// about whether the command took effect; the next document says.
    private func replyBoundElapsed(for requestID: String) {
        guard let pending = pendingCommand, pending.requestID == requestID else { return }
        clearPending()
        let seconds = String(format: "%g", replyBound)
        commandFailureSubject.send(
            CartridgeFailure(
                kind: .timedOut,
                message: """
                    The Tower did not answer the \(pending.verb) request within \
                    \(seconds) s. Whether it took effect is not known from here; \
                    the Lab's state above is what the Tower last said.
                    """
            )
        )
    }

    // MARK: Connection lifecycle

    private func connectionChanged(to status: TowerStatus) {
        guard status == .online else {
            // The subscription belonged to a socket that is gone. Nothing is
            // sent to close it — the Tower treats a closed socket as sufficient
            // cleanup.
            subscriptionID = nil
            isSubscribing = false
            // A fresh connection gets a fresh budget: the cap exists to stop a
            // hopeless retry loop within one socket, not to remember a bad
            // socket forever.
            resubscribesUsed = 0
            // The **document** is dropped, unlike World Builder's last report.
            // A CV Lab status carries `source.receiving_frames`,
            // `clients_connected` and a run's elapsed time, all of which
            // describe a Tower this app can no longer see; holding them would
            // let a disconnected screen say frames were arriving. Availability
            // renders the disconnection on its own, and the first thing a
            // reconnection brings is a complete snapshot.
            self.status = nil
            state = .idle(available: [])
            lastRefusal = nil
            // A reply to a command sent on the old socket cannot arrive on the
            // new one — the Tower answers on the connection that asked — so
            // waiting for it would only ever end at the reply bound, ten
            // seconds after the answer became impossible.
            clearPending()
            return
        }
        subscribeIfPossible()
        // Asked immediately rather than waiting for the subscription's first
        // snapshot. They are the same document, and asking costs one small
        // message — but a subscription that is refused or never acknowledged
        // would otherwise leave the screen with nothing at all, and the
        // read-only half is the half that works everywhere.
        refreshStatus()
    }

    /// Idempotent by construction: every path into it is guarded by the same
    /// two flags, so a status change and a republished declaration racing each
    /// other cannot open two subscriptions.
    private func subscribeIfPossible() {
        guard tower.status == .online, subscriptionID == nil, !isSubscribing else { return }
        guard let offer else {
            // The Tower said nothing about the CV Lab. Availability already
            // renders that as `.noContract`; there is nothing to subscribe to.
            return
        }
        guard TowerCapabilities.supported.contains(offer.contract) else {
            // A contract this build does not implement. `.unsupportedContract`
            // availability already outranks any state, so the state is left
            // where it was rather than given a second, weaker wording of the
            // same fact.
            return
        }
        guard offer.available else {
            // Offered and unserveable. The Tower's own prose is the only honest
            // explanation available, so it is shown verbatim.
            state = .unsupported(
                reason: offer.unavailableReason
                    ?? "The Tower says it cannot serve the Experimental CV Lab right now."
            )
            return
        }

        isSubscribing = true
        tower.subscribeToResults(
            cartridge: offer.cartridge,
            resultType: offer.resultType,
            contract: offer.contract
        )
    }

    // MARK: Inbound

    private func handle(_ event: CVLabEvent) {
        switch event {
        case .status(let reply):
            clearPending(ifAnsweredBy: reply.requestID)
            // `acceptedCommand` is read for one purpose: an accepted command
            // clears the refusal that a previous one left on screen. It is
            // deliberately **not** used to decide what state to move to — the
            // document decides that, whether it was pushed, read, or sent in
            // answer to a button.
            if reply.acceptedCommand != nil { lastRefusal = nil }
            apply(CVLabStatus(json: reply.status))

        case .refused(let refusal):
            clearPending(ifAnsweredBy: refusal.requestID)
            lastRefusal = refusal
            // The document is applied even on a refusal, and it is the
            // *unchanged* one: the request did not take effect, so this is not
            // a rollback, it is the current truth arriving with the news that
            // nothing happened.
            if let document = refusal.decodedStatus {
                apply(document)
            }
            // The one refusal that is terminal. `lab_unavailable` means this
            // Tower runs no CV Lab or its module failed, and the honest
            // rendering is "this Tower cannot do this" rather than an error
            // inviting a retry. Every other reason, including `internal_error`,
            // leaves the state where the document put it — telling a person to
            // give up on a working Tower is worse than telling them to try
            // again.
            if refusal.disposition == .terminal {
                state = .unsupported(reason: refusal.message)
            }
            // `lastRefusal` is read through the view model rather than
            // published by it, so something has to invalidate the view. A
            // refusal that carried a document has already done so by applying
            // it; one that did not — the only shape this client has ever seen
            // is a malformed reply — would otherwise be recorded and never
            // drawn.
            republishState()

        case .frameRefused:
            // Deliberately ignored **here**, and rendered by the panel that
            // renders the frame path's results, which already observes
            // `TowerClient.latestFrameRefusal`.
            //
            // It is not used to drive the lifecycle. A `frame_error` says what
            // the Lab was doing when one frame arrived; the status document
            // says what it is doing now, and at the sender's ~0.8 frames per
            // second the two can disagree by more than a second. One source of
            // truth for the state, and it is the one the Tower calls the whole
            // truth.
            break
        }
    }

    private func handle(_ event: CartridgeResultEvent) {
        switch event {
        case .declaration:
            // Handled through `$cartridgeDeclaration` instead, so the cached
            // value and the trigger cannot disagree.
            break

        case .subscribed(let ack):
            guard ack.cartridge == ExperimentalCVContract.towerCartridge else { return }
            subscriptionID = ack.subscriptionID
            isSubscribing = false

        case .unsubscribed(let id):
            guard id == subscriptionID else { return }
            subscriptionID = nil

        case .result(let envelope):
            guard envelope.cartridge == ExperimentalCVContract.towerCartridge else { return }
            // The envelope says whether it is a complete state or a delta, and
            // this build knows how to merge exactly nothing. `snapshot_only` is
            // `true` for this cartridge in the declaration and every envelope
            // says so again; a partial document would decode cleanly into a
            // state with no run and quietly empty the screen.
            guard envelope.isSnapshot else {
                state = .failed(
                    CartridgeFailure(
                        kind: .notSupported,
                        message: """
                            The Tower sent a CV Lab document marked as a partial update \
                            rather than a complete one. This build can only read complete \
                            documents, so it is showing nothing rather than a state \
                            assembled from a piece it does not know how to merge.
                            """
                    )
                )
                return
            }
            let pushed = CVLabStatus(json: envelope.payload)
            // Feed the frame gate from the subscription, not only from replies
            // to our own commands.
            //
            // The Lab has one slot shared by every connection and last start
            // wins, and the Tower sends `cv_lab_status` ONLY to the client that
            // sent a command. So this document is the sole way this app learns
            // about a run somebody else started — and until it fed the gate,
            // that case discarded every subsequent `frame_result` while leaving
            // the previous experiment's figures on screen under the new
            // experiment's name.
            //
            // Still a status document, never a `frame_result`: the rule that a
            // result must not nominate its own run is untouched.
            // `lifecycle.run_id` specifically, which is the same field the
            // direct `cv_lab_status` path reads. Two feeds into one gate must
            // not disagree about where the run id lives.
            // Only when the document decoded. A payload this build could not
            // read says nothing about which run is current, and clearing the
            // watch on it would discard the results of a run that is fine.
            if let pushed { tower.watchCVLabRun(pushed.lifecycle.runID) }
            apply(pushed)

        case .failed(let error):
            guard isOurs(error) else { return }
            // A result-channel failure is about the **subscription**, not about
            // the Lab: the commands and `cv_lab_status` still work, and the
            // screen keeps whatever document it last read rather than being
            // emptied by a transport problem one layer below it. That reasoning
            // is why `state` is deliberately untouched here.
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
            // But the in-flight attempt is over either way, and this cleared
            // nothing at all before — so any refused subscribe left
            // `isSubscribing == true` and `subscribeIfPossible()` returned at
            // its first guard for the rest of the connection. The Lab kept
            // showing the document it read at connect time, forever, with no
            // retry and nothing on screen to say so.
            //
            // `snapshot_failed` is the reachable transient: the Tower sends it
            // instead of `result_subscribed` when the first snapshot raises.
            isSubscribing = false
        }
    }

    /// Whether a result-channel error belongs to this cartridge.
    ///
    /// Matched on either name or subscription id because the Tower's extras are
    /// reason-dependent. An error carrying neither is not claimed —
    /// attributing another cartridge's failure to this one would be a
    /// fabricated report about the Tower.
    private func isOurs(_ error: CartridgeResultError) -> Bool {
        if let cartridge = error.cartridge {
            return cartridge == ExperimentalCVContract.towerCartridge
        }
        if let id = error.subscriptionID { return id == subscriptionID }
        return false
    }

    // MARK: Projection

    /// Applies one status document.
    ///
    /// The single writer of `state` for every arriving document, whichever of
    /// the three surfaces it came in on. They are the same document from one
    /// builder on the Tower, and they are *not* byte-identical across time —
    /// `elapsed_s`, the throughput figures and `receiving_frames` advance with
    /// the clock — so nothing here compares two of them to decide whether
    /// something changed.
    private func apply(_ document: CVLabStatus?) {
        guard let document else {
            state = .failed(
                CartridgeFailure(
                    kind: .undecodableResponse,
                    message: """
                        The Tower's Experimental CV Lab reply could not be read as \
                        \(ExperimentalCVContract.status). Nothing is shown rather than \
                        something guessed.
                        """
                )
            )
            return
        }
        // Checked, not assumed. A document arriving under a different
        // identifier is a different agreement, and decoding it on the
        // resemblance of its keys is how a field that changed meaning gets
        // rendered as though it had not.
        if let contract = document.contract, contract != ExperimentalCVContract.status {
            state = .failed(
                CartridgeFailure(
                    kind: .undecodableResponse,
                    message: """
                        The Tower sent an Experimental CV Lab document under \
                        \(contract), and this build implements \
                        \(ExperimentalCVContract.status). Updating the app is what \
                        resolves this.
                        """
                )
            )
            return
        }
        status = document
        state = Self.project(document)
    }

    /// Re-derives the state from the document already held.
    ///
    /// Called when something changed on **this** side rather than the Tower's —
    /// the stream bracket opening or closing. Nothing in `state` depends on it
    /// today; the liveness question is answered by
    /// `ExperimentalCVState.isLive(isStreaming:isReceivingFrames:)` at the point
    /// of display, with both halves passed in. This exists so the view is
    /// re-evaluated when the phone's half changes, because otherwise a screen
    /// showing "live" would keep showing it after the camera stopped until the
    /// Tower's next two-second heartbeat.
    private func republishState() {
        statusSubject.send(status)
    }

    /// The seven Tower lifecycle states, projected onto this app's six cases.
    ///
    /// The one place they disagree is deliberate and is the Tower's own
    /// decision: it says `stopped` and this says `.completed`, because *"a
    /// bench run does not complete; it is stopped by a person"*. The Tower says
    /// what happened; iOS renders it with the case its state machine has.
    static func project(_ document: CVLabStatus) -> ExperimentalCVState {
        let runOrNil = document.run
        switch document.lifecycle.state {
        case "unavailable":
            return .unsupported(
                reason: document.lifecycle.reason
                    ?? "This Tower cannot run experiments. It did not say why."
            )

        case "idle":
            return .idle(available: document.available)

        case "starting":
            // The experiment being armed, named from the run when the document
            // carries one and from the catalog otherwise. A `.starting` with no
            // name is not constructible, so an unnameable start falls back to
            // `.idle` rather than to a spinner over a blank.
            if let experiment = runOrNil?.experiment ?? document.selectedExperiment {
                return .starting(experiment)
            }
            return .idle(available: document.available)

        case "running":
            guard let run = runOrNil else { return .idle(available: document.available) }
            return .running(run)

        case "paused":
            guard let run = runOrNil else { return .idle(available: document.available) }
            return .paused(run)

        case "stopped":
            guard let run = runOrNil else { return .idle(available: document.available) }
            return .completed(run)

        case "failed":
            return .failed(
                CartridgeFailure(
                    kind: .towerReportedFailure,
                    message: document.lifecycle.reason
                        ?? """
                            The Tower's last attempt to start an experiment failed and it \
                            did not say why.
                            """
                )
            )

        default:
            // An eighth state. Not guessed at, and not collapsed into `.idle`,
            // which would invite a person to press Start against a Lab that may
            // be doing anything at all.
            return .failed(
                CartridgeFailure(
                    kind: .undecodableResponse,
                    message: """
                        The Tower reports the Experimental CV Lab in a state this build \
                        does not know: "\(document.lifecycle.state)". Nothing is shown \
                        rather than something guessed.
                        """
                )
            )
        }
    }
}

// MARK: - View model

/// Publishes Experimental CV Lab state into SwiftUI.
///
/// Holds no runtime references — no `GlassesConnection`, no `TowerClient`, no
/// socket — for the same reason `WorldBuilderViewModel` does not: a workspace
/// `@StateObject` is destroyed when the cartridge changes, and destroying
/// something that owns the camera would end the session. Connectivity arrives
/// as a value; the client is injected and owned above this object.
@MainActor
final class ExperimentalCVViewModel: ObservableObject {
    @Published private(set) var state: ExperimentalCVState
    /// The whole document, for the fields no state case can carry — the
    /// catalog, the Tower-wide `source` block, which experiment is selected.
    @Published private(set) var status: CVLabStatus?

    /// The most recent failed attempt to command the Lab, kept separate from
    /// `state` so a rejected request does not erase whatever the workspace was
    /// already showing.
    @Published private(set) var lastRequestFailure: CartridgeFailure?

    /// The command sent and not yet answered, republished so the row that
    /// asked can say so and the others can wait. See `CVLabPendingCommand`.
    @Published private(set) var pendingCommand: CVLabPendingCommand?

    private let client: any ExperimentalCVClient
    private var cancellables: Set<AnyCancellable> = []

    /// No default argument — see `WorldBuilderViewModel.init(client:)`.
    init(client: any ExperimentalCVClient) {
        self.client = client
        self.state = client.state
        self.status = client.status
        self.pendingCommand = client.pendingCommand

        client.stateUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in self?.state = state }
            .store(in: &cancellables)

        client.statusUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] status in self?.status = status }
            .store(in: &cancellables)

        client.pendingCommandUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] pending in self?.pendingCommand = pending }
            .store(in: &cancellables)

        // Filed where a refused send goes, deliberately: "could not be sent"
        // and "was sent and never answered" are both answers to "what happened
        // when I pressed that", and a person should find both in one place.
        client.commandFailures
            .receive(on: DispatchQueue.main)
            .sink { [weak self] failure in self?.lastRequestFailure = failure }
            .store(in: &cancellables)
    }

    /// Whether any command is awaiting the Tower's answer.
    var isCommandPending: Bool { pendingCommand != nil }

    /// Whether `experiment` is the one an unanswered start named.
    func isAwaitingStart(of experiment: CVExperiment) -> Bool {
        guard let pending = pendingCommand, pending.command == .start else { return false }
        return pending.experimentID == experiment.id
    }

    /// Experiments the Tower declared, whatever the Lab is currently doing.
    ///
    /// Read off the **document** rather than off `.idle`'s payload, which is
    /// the change that makes a picker usable: the catalog does not stop
    /// existing because something is running, and `cv_lab_start` replaces
    /// whatever ran, so choosing a different experiment mid-run is a supported
    /// action rather than an error. Empty when no document has been read.
    var availableExperiments: [CVExperiment] {
        if let status { return status.available }
        // The fallback that keeps every non-Tower client working: a client that
        // only reports a state carries its catalog in `.idle`.
        if case .idle(let available) = state { return available }
        return []
    }

    /// The Tower-wide view of whether anything is feeding the Lab.
    var source: CVLabStatus.Source? { status?.source }

    /// The refusal the Tower sent for the last command, if it refused one.
    var lastRefusal: CVLabControlRefusal? { client.lastRefusal }

    /// Whether this build may draw controls that command the Lab.
    var canSendCommands: Bool { client.canSendCommands }

    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        client.availability(isTowerReachable: isTowerReachable)
    }

    func phase(isTowerReachable: Bool) -> CartridgePhase {
        availability(isTowerReachable: isTowerReachable).forcedPhase ?? state.phase
    }

    func unavailableExplanation(isTowerReachable: Bool) -> String {
        availability(isTowerReachable: isTowerReachable)
            .explanation(cartridgeName: "Experimental CV Lab", clientReason: clientReason)
    }

    private var clientReason: String? {
        switch state {
        case .unsupported(let reason): return reason
        case .failed(let failure): return failure.message
        case .idle, .starting, .running, .paused, .completed: return nil
        }
    }

    /// Requests a run, and records the refusal when there is one.
    ///
    /// Deliberately does not rethrow. The caller is a SwiftUI button action,
    /// which cannot handle an error usefully; recording it as published state
    /// is what puts the refusal on screen instead of in a log.
    func run(_ experiment: CVExperiment) {
        perform { try client.run(experiment) }
    }

    func pause() { perform { try client.pause() } }
    func resume() { perform { try client.resume() } }
    func stop() { perform { try client.stop() } }

    /// One place where a local refusal becomes published state.
    ///
    /// **A cleared `lastRequestFailure` means the request was sent, not that it
    /// succeeded.** The outcome arrives as state, on the next document, which
    /// is the shape every command on this wire has.
    private func perform(_ command: () throws -> Void) {
        do {
            try command()
            lastRequestFailure = nil
        } catch {
            lastRequestFailure = CartridgeFailure.wrapping(error)
        }
    }
}
