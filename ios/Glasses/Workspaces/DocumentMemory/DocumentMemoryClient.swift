//
//  DocumentMemoryClient.swift
//  Glasses
//

import Combine
import Foundation

/// Supplies `DocumentMemoryState` to the Document Memory workspace.
///
/// ## A third interaction shape
///
/// World Builder pushes a continuously current state. Experimental CV Lab takes
/// a command and reports progress. Scene Understanding publishes a moment that
/// is replaced wholesale. This one answers **point queries against something
/// that persists**: nothing on screen is current, everything is in response to
/// a question, and the answers outlive the session that produced them. Four
/// cartridges, four shapes — which is the concrete evidence that a generic
/// `CartridgeDataSource<Request, Response>` above all of them would be an
/// abstraction over things that are not alike.
///
/// `search(_:origin:)` carries the origin so a future Siri intent or wake-word
/// layer submits through this same method rather than needing its own path.
@MainActor
protocol DocumentMemoryClient: CartridgeClient {
    var state: DocumentMemoryState { get }

    /// Every state after the one `state` held when the view model was built.
    var stateUpdates: AnyPublisher<DocumentMemoryState, Never> { get }

    /// Asks the Tower's document memory a question.
    ///
    /// Throws when the request cannot be made at all. A query that runs and
    /// matches nothing is **not** a throw — that is a `DocumentQueryResult`
    /// with `.notFound` or `.noObservation` evidence, and conflating an empty
    /// answer with a failure is how "we have no record" turns into "something
    /// went wrong".
    func search(_ query: DocumentQuery, origin: DocumentQueryOrigin) throws

    // MARK: The capture session
    //
    // Defaulted below, so a test double that only answers queries does not have
    // to implement a recorder it has no opinion about.

    /// What the Tower's recorder is doing, or `nil` when nothing has said.
    var session: DocumentSessionStatus? { get }
    var sessionUpdates: AnyPublisher<DocumentSessionStatus?, Never> { get }
    /// What is on the Tower's disk, from the same status push, or `nil` when
    /// nothing has said. Its `revision` is the live-update trigger.
    var library: DocumentLibrarySummary? { get }
    var libraryUpdates: AnyPublisher<DocumentLibrarySummary?, Never> { get }
    /// The outcome of the last verb sent. See `DocumentSessionOutcome` for why
    /// this is not a `Bool`.
    var lastSessionOutcome: DocumentSessionOutcome? { get }
    /// Sends one of `start`, `pause`, `resume`, `stop`.
    func send(_ action: DocumentMemoryContract.SessionAction)
    /// Asks for the session over HTTP.
    ///
    /// Needed because the two transports are independent: a Tower reachable
    /// over HTTP with no socket open still has a recorder, and a workspace that
    /// could only learn about it from a subscription would show nothing.
    func refreshSession()
}

extension DocumentMemoryClient {
    var stateUpdates: AnyPublisher<DocumentMemoryState, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }
    var session: DocumentSessionStatus? { nil }
    var sessionUpdates: AnyPublisher<DocumentSessionStatus?, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }
    var library: DocumentLibrarySummary? { nil }
    var libraryUpdates: AnyPublisher<DocumentLibrarySummary?, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }
    var lastSessionOutcome: DocumentSessionOutcome? { nil }
    func send(_ action: DocumentMemoryContract.SessionAction) {}
    func refreshSession() {}
}

// MARK: - The Tower-backed client

/// The real Document Memory client.
///
/// ## Two transports, two contracts, one client
///
/// ```
/// TowerClient                socket: document_memory.status/2026-09-07
///     ↓ cartridgeResults        → what the recorder is doing, and what is on disk
/// TowerDocumentMemoryClient
///     ↓ DocumentMemoryHTTPClient   HTTP: document_memory.library/2026-09-07
///                                  → the documents themselves
/// ```
///
/// The split is the Tower's and it is deliberate: document text is the largest
/// and most sensitive thing this platform holds, the result sender shares a
/// send lock with the frame path, and a listing is pulled on demand rather than
/// pushed at a phone that asked no question.
///
/// **The two identifiers are compared separately, at the two places they
/// arrive.** A change to the status contract is not a change to the library
/// contract, and a build that checked one string would refuse a working library
/// because a session field moved.
///
/// ## Stop keeps documents
///
/// Nothing in this client clears a result when a session ends, and that is the
/// point. A record of what was read is exactly as true after the session stops
/// — and a dwell in progress is **flushed, not dropped**, so a stop can add a
/// document rather than lose one. This is the deliberate opposite of
/// `TowerSceneUnderstandingClient`, which discards on stop, on disconnect and
/// on a session-id change. A document is a record; a scene is not.
@MainActor
final class TowerDocumentMemoryClient: DocumentMemoryClient {

    let cartridgeID = "document-memory"

    private(set) var state: DocumentMemoryState = .idle {
        didSet {
            guard state != oldValue else { return }
            stateSubject.send(state)
        }
    }

    private(set) var session: DocumentSessionStatus? {
        didSet {
            guard session != oldValue else { return }
            sessionSubject.send(session)
        }
    }

    private(set) var lastSessionOutcome: DocumentSessionOutcome?

    /// The `library` half of the status push. Kept since 2026-09-07: until
    /// then every push was decoded in full and everything but `session` was
    /// dropped, so a page recorded mid-session reached the screen only when
    /// a person tapped Recent. `revision` on it is what the view model
    /// re-fetches on.
    private(set) var library: DocumentLibrarySummary? {
        didSet {
            guard library != oldValue else { return }
            librarySubject.send(library)
        }
    }

    var stateUpdates: AnyPublisher<DocumentMemoryState, Never> {
        stateSubject.eraseToAnyPublisher()
    }
    var sessionUpdates: AnyPublisher<DocumentSessionStatus?, Never> {
        sessionSubject.eraseToAnyPublisher()
    }
    var libraryUpdates: AnyPublisher<DocumentLibrarySummary?, Never> {
        librarySubject.eraseToAnyPublisher()
    }

    private let stateSubject = PassthroughSubject<DocumentMemoryState, Never>()
    private let sessionSubject = PassthroughSubject<DocumentSessionStatus?, Never>()
    private let librarySubject = PassthroughSubject<DocumentLibrarySummary?, Never>()

    private let tower: TowerClient
    private let http: DocumentMemoryHTTPClient
    private var cancellables: Set<AnyCancellable> = []

    private var subscriptionID: String?
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
    /// The query in flight, so a late answer to a superseded question cannot
    /// overwrite a newer one.
    private var inFlight: Int = 0

    init(tower: TowerClient, http: DocumentMemoryHTTPClient = DocumentMemoryHTTPClient()) {
        self.tower = tower
        self.http = http

        // `.receive(on:)` on both: a `@Published` publisher fires from
        // `willSet`, so a sink that reads the property it was notified about
        // sees the value *before* the change. `TowerWorldBuilderClient`
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

        tower.cartridgeResults
            .sink { [weak self] event in self?.handle(event) }
            .store(in: &cancellables)
    }

    // MARK: Availability

    /// Resolved against the Tower's live declaration, from the **HTTP**
    /// contract.
    ///
    /// The library is what this workspace is for, so the library's offer is
    /// what decides whether the workspace can do anything. The subscription
    /// offer is checked separately, where the subscription is opened — the two
    /// are keyed on the same variable on this Tower and could still disagree on
    /// another, and folding them together would let a missing recorder read as
    /// a missing library.
    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        CartridgeAvailability.resolve(
            declared: declaredLibraryContract,
            supported: [DocumentMemoryContract.libraryIdentifier],
            isTowerReachable: isTowerReachable,
            // This build has a Document Memory client, so a silent socket means
            // "nobody has asked yet", not "this Tower will never do it".
            knownToThisBuild: true
        )
    }

    private var declaredLibraryContract: CartridgeContract? {
        guard
            let offer = tower.cartridgeDeclaration?
                .httpContract(forTowerCartridge: DocumentMemoryContract.towerCartridge)
        else { return nil }
        // `available` is honoured, and it was not before.
        //
        // `TowerHTTPContractOffer` carries six fields and only `contract` was
        // ever read here, so with `TOWER_DOCUMENT_ROOT` unset this resolved
        // `.available`, enabled the search field, and let the query fail with a
        // 404 -- while the Tower's own sentence naming the variable that would
        // fix it was decoded and never shown. Every other cartridge checks this
        // flag at subscribe time; the library is fetched rather than subscribed,
        // so it had no equivalent gate.
        //
        // The reason travels to the screen through `clientReason` rather than
        // being lost here, so a person is told what the Tower said and not
        // merely that nothing is available.
        guard offer.available else { return nil }
        return CartridgeContract(cartridgeID: cartridgeID, identifier: offer.contract)
    }

    /// The Tower's own words for why the library cannot be served, when it has
    /// declared the contract and marked it unavailable.
    ///
    /// Surfaced so the reason survives `declaredLibraryContract` returning nil
    /// -- it names the `TOWER_` variable that would fix it, which nothing local
    /// can.
    var libraryUnavailableReason: String? {
        guard
            let offer = tower.cartridgeDeclaration?
                .httpContract(forTowerCartridge: DocumentMemoryContract.towerCartridge),
            !offer.available
        else { return nil }
        return offer.unavailableReason
    }

    // MARK: Queries

    /// Runs a query, and publishes whatever came back — including the two
    /// answers that are not lists.
    ///
    /// Throws only when the request cannot be made at all. `.notFound` and
    /// `.noObservation` are **results**, and the whole design of this cartridge
    /// turns on their not being failures.
    func search(_ query: DocumentQuery, origin: DocumentQueryOrigin) throws {
        let availability = availability(isTowerReachable: tower.status == .online)
        guard availability.isAvailable else {
            // A local refusal, and attributed locally: the Tower reported
            // nothing here and there may not even be a socket open.
            throw CartridgeFailure(
                kind: .notSupported,
                message: availability.explanation(cartridgeName: "Document Memory")
                    ?? "This Tower serves no document memory."
            )
        }

        inFlight += 1
        let token = inFlight
        state = .searching(query)

        Task { [http] in
            do {
                let response = try await Self.run(query, using: http)
                await MainActor.run { self.apply(response, query: query, origin: origin, token: token) }
            } catch let error as DocumentMemoryFetchError {
                await MainActor.run { self.apply(error, token: token) }
            } catch {
                await MainActor.run {
                    self.apply(DocumentMemoryFetchError.transport(error.localizedDescription), token: token)
                }
            }
        }
    }

    /// Which route answers which query.
    ///
    /// `.semantic` goes to the **literal** route, which is what the Tower's own
    /// `semantic_retrieval_alternative` says to do: it computes no embedding,
    /// and routing free text nowhere would leave the app's primary input path
    /// with no Tower route at all. The caveat that the match was lexical
    /// travels with the answer — see `DocumentQuery.matchingCaveat` — because a
    /// description missing is not evidence the document was never seen.
    nonisolated static func run(
        _ query: DocumentQuery, using http: DocumentMemoryHTTPClient
    ) async throws -> DocumentLibraryResponse {
        switch query {
        case .recent(let limit):
            return try await http.recent(limit: limit)
        case .text(let text), .semantic(let text):
            return try await http.search(text: text)
        case .observedWithin(let interval):
            // A centre and a half-width, which is the shape the route takes.
            // The interval stays an interval up to this line, so an approximate
            // question is never collapsed to an instant on the way.
            let centre = interval.start.timeIntervalSince1970 + interval.duration / 2
            return try await http.around(at: centre, windowSeconds: interval.duration / 2)
        }
    }

    private func apply(
        _ response: DocumentLibraryResponse,
        query: DocumentQuery,
        origin: DocumentQueryOrigin,
        token: Int
    ) {
        guard token == inFlight else { return }
        do {
            state = .results(
                try DocumentQueryResult(
                    query: query,
                    origin: origin,
                    documents: response.documents,
                    evidence: response.answer.evidence,
                    response: response
                )
            )
        } catch {
            // A `matched` answer with no documents. The safe direction from a
            // broken payload is a failure, never a stronger claim than the
            // Tower made — and never `.notFound`, whose sentence is a definite
            // negative about the wearer's own memory.
            state = .failed(CartridgeFailure.wrapping(error))
        }
    }

    private func apply(_ error: DocumentMemoryFetchError, token: Int) {
        guard token == inFlight else { return }
        switch error {
        case .noDocumentRootConfigured(let detail):
            // A configuration answer, and never an answer about a document.
            // `.unsupported` rather than `.failed` so the workspace says "this
            // Tower keeps no documents" with the variable named, rather than
            // "the search failed".
            state = .unsupported(reason: detail)
        default:
            state = .failed(error.failure)
        }
    }

    // MARK: The capture session

    /// Sends a verb and reads the state that came back.
    ///
    /// **The reply's status code is not consulted for success**, because on
    /// this cartridge it cannot mean it: a resume on a stopped session answers
    /// 200 with `state: "stopped"` and no refusal field. What moved, if
    /// anything, is in `DocumentSessionOutcome`.
    func send(_ action: DocumentMemoryContract.SessionAction) {
        Task { [http] in
            do {
                let reported = try await http.sendSession(action)
                await MainActor.run {
                    self.session = reported
                    self.lastSessionOutcome = .of(action, reported: reported)
                }
            } catch let error as DocumentMemoryFetchError {
                await MainActor.run { self.lastSessionOutcome = .failed(error.failure) }
            } catch {
                await MainActor.run {
                    self.lastSessionOutcome = .failed(CartridgeFailure.wrapping(error))
                }
            }
        }
    }

    /// Fetches the session once, over HTTP.
    ///
    /// Useful before a subscription exists — a Tower reachable over HTTP but
    /// with no socket open still has a recorder, and a workspace that could
    /// only learn about it from the socket would show nothing.
    func refreshSession() {
        Task { [http] in
            let reported = try? await http.sessionStatus()
            await MainActor.run { if let reported { self.session = reported } }
        }
    }

    // MARK: The subscription

    private func connectionChanged(to status: TowerStatus) {
        guard status == .online else {
            subscriptionID = nil
            isSubscribing = false
            // A fresh connection gets a fresh budget: the cap exists to stop a
            // hopeless retry loop within one socket, not to remember a bad
            // socket forever.
            resubscribesUsed = 0
            // The session status describes a socket that is gone, so it stops
            // being current. **The query result is deliberately not cleared:**
            // a document the Tower recorded is exactly as true with the socket
            // down, and clearing it would be this app discarding a record the
            // Tower kept.
            session = nil
            return
        }
        subscribeIfPossible()
    }

    private func subscribeIfPossible() {
        guard tower.status == .online, subscriptionID == nil, !isSubscribing else { return }
        guard
            let offer = tower.cartridgeDeclaration?
                .offer(forTowerCartridge: DocumentMemoryContract.towerCartridge)
        else { return }
        // The **status** identifier, compared here and nowhere else. The
        // library identifier is compared on every HTTP response.
        guard
            offer.contract == DocumentMemoryContract.statusIdentifier,
            offer.resultType == DocumentMemoryContract.resultType,
            offer.available
        else { return }

        isSubscribing = true
        tower.subscribeToResults(
            cartridge: offer.cartridge,
            resultType: offer.resultType,
            contract: offer.contract
        )
    }

    private func handle(_ event: CartridgeResultEvent) {
        switch event {
        case .declaration:
            break

        case .subscribed(let ack):
            guard ack.cartridge == DocumentMemoryContract.towerCartridge else { return }
            subscriptionID = ack.subscriptionID
            isSubscribing = false

        case .unsubscribed(let id):
            guard id == subscriptionID else { return }
            subscriptionID = nil

        case .result(let envelope):
            guard envelope.cartridge == DocumentMemoryContract.towerCartridge else { return }
            guard envelope.isSnapshot else { return }
            guard let status = DocumentMemoryDecoder.status(from: envelope.payload) else { return }
            session = status.session
            library = status.library

        case .failed(let error):
            guard error.cartridge == DocumentMemoryContract.towerCartridge
                || (error.subscriptionID != nil && error.subscriptionID == subscriptionID)
            else { return }
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
            // The recorder's status stops being current, and nothing about the
            // library changes: an error on the status channel says nothing
            // about what is on disk.
            session = nil
        }
    }
}

// MARK: - The stub, for a Tower that declares nothing

/// The client for a Tower that has declared no document library.
///
/// It holds **no sample documents**. A handful of plausible rows would make the
/// workspace look finished and would be indistinguishable, on screen, from real
/// memories of the wearer's own documents — the single most misleading fixture
/// this app could ship. Test fixtures live in the test target, where they can
/// never reach a device.
///
/// ## The sentence that used to be here, and why it was deleted
///
/// > "The Tower keeps no document memory, so there is nothing to search. This
/// > app sends camera frames to the Tower while a session is running; it reads
/// > no text from them and receives none back."
///
/// The first clause is a claim about what the Tower **stores**, and it is now
/// false: the Tower has a document store, a journal, six routes and a typed
/// contract. The last clause is false too — a search receives titles, snippets
/// and, on one route, page text. Leaving it up would have told a person their
/// Tower keeps no record of what they read while it was keeping one.
///
/// What replaced it says only what is observable from here: **this** Tower has
/// declared nothing. It makes no claim about what any Tower stores, because
/// this app has no channel through which it could know that.
@MainActor
final class UnavailableDocumentMemoryClient: DocumentMemoryClient {
    static let reason = """
        This Tower has not declared a document library, so there is nothing to \
        search. That is a statement about what this Tower offers, not about \
        what you have read.
        """

    let cartridgeID = "document-memory"

    let state: DocumentMemoryState = .unsupported(reason: UnavailableDocumentMemoryClient.reason)

    init() {}

    /// Always throws, for the same reason the Experimental CV client does: a
    /// silent no-op leaves a search field that appears to work and returns
    /// nothing, which a person reads as "I have no documents" rather than as
    /// "this Tower has none to offer".
    func search(_ query: DocumentQuery, origin: DocumentQueryOrigin) throws {
        throw CartridgeFailure(kind: .notSupported, message: Self.reason)
    }
}

// MARK: - View model

/// Publishes Document Memory state into SwiftUI.
///
/// Holds no runtime references, for the reason given on `WorldBuilderViewModel`.
@MainActor
final class DocumentMemoryViewModel: ObservableObject {
    @Published private(set) var state: DocumentMemoryState
    @Published private(set) var session: DocumentSessionStatus?
    @Published private(set) var lastSessionOutcome: DocumentSessionOutcome?

    /// What the person has typed. Owned here rather than in the view so a query
    /// can also arrive from somewhere that has no text field — the origin
    /// parameter on `submit` is the seam a Siri intent would use.
    @Published var queryText: String = ""

    @Published private(set) var lastRequestFailure: CartridgeFailure?

    /// A one-line statement of the camera's relation to this recorder, for
    /// the session panel. `nil` when this screen started the camera itself.
    @Published private(set) var cameraNote: String?

    private let client: any DocumentMemoryClient
    /// The glasses camera, when this build can reach it (DEBUG only, exactly
    /// as Object Memory's coordinator has it). `nil` in Release, where the
    /// Tower half still works and the panel says the camera half cannot be
    /// reached from here.
    private let camera: (any ObjectMemoryCaptureOwner)?
    /// Whether this app started the running capture. Injected rather than
    /// stored, because this view model is a `@StateObject` and is destroyed
    /// on every cartridge switch -- see `CartridgeCameraClaim` for what that
    /// cost when the fact lived here.
    private let cameraClaim: CartridgeCameraClaim
    private var startedTheCamera: Bool {
        get { cameraClaim.startedByThisApp }
        set { cameraClaim.startedByThisApp = newValue }
    }
    private var cancellables: Set<AnyCancellable> = []

    /// The question the screen is currently showing the answer to, re-asked
    /// when the Tower's library changes. `nil` until something is asked.
    private var standingQuery: DocumentQuery?
    private var lastLibraryRevision: Int?

    /// No default argument for `client` — see `WorldBuilderViewModel.init(client:)`.
    init(
        client: any DocumentMemoryClient,
        camera: (any ObjectMemoryCaptureOwner)? = nil,
        // Optional with a nil default, and built in the body rather than in
        // the signature: a default-argument expression is evaluated in a
        // nonisolated context, and `CartridgeCameraClaim` is @MainActor, so
        // `= CartridgeCameraClaim()` here is a main-actor call from outside
        // the actor. The same trap this project has hit before with
        // `= WorldGeometryClient()`.
        cameraClaim: CartridgeCameraClaim? = nil
    ) {
        self.client = client
        self.camera = camera
        // A view model given no claim gets its own, which is right for the
        // tests and previews that construct one directly: nothing else
        // shares their camera.
        self.cameraClaim = cameraClaim ?? CartridgeCameraClaim()
        self.state = client.state
        self.session = client.session
        self.lastSessionOutcome = client.lastSessionOutcome
        self.lastLibraryRevision = client.library?.revision

        client.stateUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in self?.state = state }
            .store(in: &cancellables)

        client.sessionUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] session in
                self?.session = session
                // Pulled from the client rather than published separately: the
                // outcome and the status change together, and two publishers
                // would let a view render a new state beside an old verdict.
                self?.lastSessionOutcome = self?.client.lastSessionOutcome
                self?.updateCameraNote()
            }
            .store(in: &cancellables)

        // The live library. `revision` moves whenever the Tower's journal
        // changes -- a page recorded, a sighting merged, a prune -- and the
        // standing question is re-asked exactly then. Not a poll: nothing
        // here fires on a timer, and an unchanged revision re-fetches nothing.
        client.libraryUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] library in self?.libraryChanged(library) }
            .store(in: &cancellables)

        camera?.captureClaimUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] claim in self?.cameraClaimChanged(claim) }
            .store(in: &cancellables)
        updateCameraNote()
    }

    /// The camera's claim changed underneath this screen.
    ///
    /// Ownership is dropped when the capture ends, whoever ended it — the
    /// same rule, for the same reason, as
    /// `ObjectMemoryRecordingCoordinator.cameraClaimChanged`. This type is
    /// documented as "Modelled on `ObjectMemoryRecordingCoordinator`" and
    /// copied the branch structure of `send(_:)` without this half of it.
    ///
    /// The memory of having started the camera now lives on a
    /// `CartridgeCameraClaim` that `ProjectManager` owns, so that a cartridge
    /// switch cannot lose it and strand a running capture. That is right, and
    /// it is exactly what makes dropping it here necessary rather than
    /// optional: a fact that outlives the screen also outlives the capture it
    /// describes. Without this, a Stop pressed on Home followed by a Start
    /// pressed on Home leaves this screen believing it owns a capture it
    /// never started — so `cameraNote` goes quiet where it should say the
    /// camera belongs to another screen, and Stop reaches for
    /// `stopCameraSession()` on somebody else's session.
    ///
    /// This screen's own Stop is unaffected: it reads the flag and clears it
    /// in the same turn, before the `.unclaimed` that follows arrives here.
    private func cameraClaimChanged(_ claim: CaptureClaim) {
        if claim == .unclaimed { startedTheCamera = false }
        updateCameraNote()
    }

    private func libraryChanged(_ library: DocumentLibrarySummary?) {
        guard let revision = library?.revision else { return }
        guard revision != lastLibraryRevision else { return }
        let first = lastLibraryRevision == nil
        lastLibraryRevision = revision
        // The first revision seen is the baseline, not news.
        guard !first else { return }
        if case .searching = state { return }
        submit(standingQuery ?? .recent(limit: 20), origin: .appText)
    }

    /// Called when the workspace appears: the session, one shot, and the
    /// recent listing if nothing has been asked yet.
    func appear() {
        refreshSession()
        if case .idle = state { showRecent() }
    }

    private func updateCameraNote() {
        guard let camera else {
            cameraNote = "The camera cannot be reached from this build. Start the camera from Home; the recorder here reads whatever the glasses stream."
            return
        }
        switch camera.captureClaim {
        case .unclaimed:
            cameraNote = "No camera is streaming. Start will start it."
        case .running:
            cameraNote = startedTheCamera ? nil : "The camera is streaming from another screen; the recorder reads those frames."
        case .devicePaused:
            cameraNote = "The glasses paused delivery themselves; frames resume when they do."
        case .ending:
            cameraNote = "The camera is stopping."
        }
    }

    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        client.availability(isTowerReachable: isTowerReachable)
    }

    func phase(isTowerReachable: Bool) -> CartridgePhase {
        availability(isTowerReachable: isTowerReachable).forcedPhase ?? state.phase
    }

    func unavailableExplanation(isTowerReachable: Bool) -> String {
        availability(isTowerReachable: isTowerReachable)
            .explanation(cartridgeName: "Document Memory", clientReason: clientReason)
    }

    private var clientReason: String? {
        // The declaration's own reason outranks any local state: it names the
        // `TOWER_` variable that would fix it, which no state here can.
        if let reason = (client as? TowerDocumentMemoryClient)?.libraryUnavailableReason {
            return reason
        }
        switch state {
        case .unsupported(let reason): return reason
        case .failed(let failure): return failure.message
        case .idle, .searching, .results: return nil
        }
    }

    /// Submits whatever is in `queryText`.
    ///
    /// Typed text becomes a `.semantic` query rather than a `.text` one: a
    /// person asking "the parking notice from this morning" is **describing** a
    /// document, not quoting it, and recording that as a literal query would
    /// lose the distinction the answer needs.
    ///
    /// What the Tower does with it is a separate question, and the answer is
    /// "matches it literally" — `semantic_retrieval` is `false` and its own
    /// `semantic_retrieval_alternative` says to route free text to the literal
    /// route. So the client sends it there and the workspace shows
    /// `DocumentQuery.matchingCaveat` on the answer. This states what the user
    /// meant; the client states how it was satisfied.
    func submitTypedQuery() {
        let trimmed = queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        submit(.semantic(trimmed), origin: .appText)
    }

    /// The entry point any input layer uses.
    func submit(_ query: DocumentQuery, origin: DocumentQueryOrigin) {
        standingQuery = query
        do {
            try client.search(query, origin: origin)
            lastRequestFailure = nil
        } catch {
            lastRequestFailure = CartridgeFailure.wrapping(error)
        }
    }

    /// The most recent documents, which is the question a workspace opens on.
    func showRecent(limit: Int = 20) {
        submit(.recent(limit: limit), origin: .appText)
    }

    /// The Tower verb, and -- for Start and Stop -- the camera.
    ///
    /// Modelled on `ObjectMemoryRecordingCoordinator`: the Tower first, then
    /// the camera only if nothing already holds it, and on Stop only a camera
    /// this screen started. A capture Home or World Builder started is
    /// somebody else's; the recorder simply reads its frames. Pause and
    /// Resume are Tower verbs only -- there is no way to pause a DAT stream.
    func send(_ action: DocumentMemoryContract.SessionAction) {
        client.send(action)
        guard let camera else { return }
        switch action {
        case .start:
            if case .unclaimed = camera.captureClaim {
                camera.startCameraSession()
                if camera.lastCaptureStartRefusal == nil {
                    startedTheCamera = true
                }
            }
        case .stop:
            if startedTheCamera {
                camera.stopCameraSession()
                startedTheCamera = false
            }
        case .pause, .resume:
            break
        }
        updateCameraNote()
    }

    /// Called when the workspace appears.
    ///
    /// Deliberately **not** a poll. The subscription pushes the session twice a
    /// second while it is open; this is the one-shot that covers the case where
    /// it is not, and repeating it on a timer would re-parse the Tower's
    /// journal on a schedule nobody asked for.
    func refreshSession() {
        client.refreshSession()
    }
}
