//
//  WorldBuilderClient.swift
//  Glasses
//

import Combine
import Foundation

/// Supplies `WorldModelState` to the World Builder workspace.
///
/// The seam. One conformer exists today and it reports only that the capability
/// is absent; the Tower-backed conformer is a later pass, once the real contract
/// is known.
///
/// ## What the shared half buys, and where it stops
///
/// `CartridgeClient` contributes the two questions every cartridge answers the
/// same way — which cartridge is this, and may it be used — so that
/// `TowerCapabilities` decides availability once for all four rather than each
/// client deciding for itself. Everything below that line is World Builder's
/// own: a *continuously changing* `state` is the right shape here and the wrong
/// shape for Document Memory, which answers point queries and would be
/// misdescribed by a property that is always current.
///
/// That is the whole reason there is no generic `fetch<Request, Response>` in
/// the shared layer. Four cartridges, four genuinely different interaction
/// shapes, one shared question.
///
/// ## `stateUpdates` exists because `state` alone is a dead end
///
/// A `{ get }` property can be *read*; it cannot *announce*. A Tower-backed
/// client whose world changes would have no way to tell the view model, and the
/// view model's `@Published state` would hold whatever `init` happened to see.
/// The publisher is the missing half, and it is a concrete `AnyPublisher` rather
/// than an `ObservableObject` conformance so `any WorldBuilderClient` stays a
/// usable existential — `ObservableObject` has an associated type and would force
/// every holder to become generic.
///
/// The default implementation never emits, which is the correct and complete
/// behaviour for a client whose state is a constant.
///
/// ## Renamed from `WorldModelSource`
///
/// Product Shell V2 called this `WorldModelSource` and its implementation
/// `UnavailableWorldModelSource`. Nothing about the contract changed — the
/// names now match the three cartridge clients added alongside it, so a reader
/// finds the same word (`…Client`, `Unavailable…Client`, `…ViewModel`) at the
/// same layer in all four.
@MainActor
protocol WorldBuilderClient: CartridgeClient {
    var state: WorldModelState { get }

    /// Every state after the one `state` held when the view model was built.
    var stateUpdates: AnyPublisher<WorldModelState, Never> { get }

    /// What this client has established about whether the world it is
    /// reporting belongs to the capture the phone currently has open.
    ///
    /// Paired with `state` rather than folded into it because the two answer
    /// different questions and change on different clocks. `state` is *what to
    /// draw*; this is *why*, and a client with no transport under it honestly
    /// has nothing to say about either — hence the `.none` default.
    var sessionBinding: WorldSessionBinding { get }

    /// Every binding after the one `sessionBinding` held when the view model
    /// was built.
    var bindingUpdates: AnyPublisher<WorldSessionBinding, Never> { get }

    /// Where the world's geometry can be fetched from, each time the Tower
    /// reports an address for it.
    ///
    /// A second publisher rather than a field inside `WorldModelState`, because
    /// the two answer different questions. The state says what to *draw*; this
    /// says where to *fetch*, and the fetch does not happen over the socket the
    /// state arrived on. Folding a session id and a transport revision into
    /// `WorldSnapshot` would put addressing inside a presentation type and
    /// would break that type's standing promise to map field for field onto the
    /// payload's `world_snapshot` block — which carries neither value.
    ///
    /// Emits nothing at all for a client with no Tower behind it, which is the
    /// correct and complete behaviour rather than an omission.
    var geometryUpdates: AnyPublisher<WorldGeometryCoordinates, Never> { get }

    /// Whether this client is following the live world or pinned to a stored
    /// one. `.live` for every client that cannot pin, which is every client
    /// but the Tower-backed one.
    var inspection: WorldInspectionMode { get }

    /// Every mode after the one `inspection` held when the view model was
    /// built.
    var inspectionUpdates: AnyPublisher<WorldInspectionMode, Never> { get }

    /// Pin the world subscription to a stored world, and optionally to one of
    /// its sessions. A no-op for a client with no transport.
    func inspect(worldID: String, sessionID: String?)

    /// Return to following the live world. A no-op for a client with no
    /// transport.
    func followLive()
}

extension WorldBuilderClient {
    /// Never emits. `Empty(completeImmediately: false)` rather than
    /// `completeImmediately: true` so a subscriber sees an open stream that
    /// happens to be silent, not a finished one — the difference matters the
    /// day a real client replaces this and a completed publisher would have
    /// already torn the subscription down.
    var stateUpdates: AnyPublisher<WorldModelState, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }

    /// A client that is not watching a capture cannot be looking at the wrong
    /// one. `.none` is the correct constant for every client with no transport,
    /// and for a Release build, which has no capture control at all.
    var sessionBinding: WorldSessionBinding { .none }

    var bindingUpdates: AnyPublisher<WorldSessionBinding, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }

    /// Never emits, for the same reason and in the same shape. A client with no
    /// Tower behind it has no geometry to address, and an open-but-silent
    /// stream says exactly that.
    var geometryUpdates: AnyPublisher<WorldGeometryCoordinates, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }

    /// A client that cannot open a stored world is always live, and says so
    /// once.
    var inspection: WorldInspectionMode { .live }

    var inspectionUpdates: AnyPublisher<WorldInspectionMode, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }

    /// Nothing to pin; nothing happens. Not an error, because the picker is
    /// simply not offered a world by such a client.
    func inspect(worldID: String, sessionID: String?) {}

    func followLive() {}
}

/// A World Builder client with no Tower behind it.
///
/// **No longer the only one.** `TowerWorldBuilderClient` is what the app graph
/// builds, and this is what remains when there is deliberately no connection to
/// build it from — the `CartridgeClients` default, and the client a test
/// substitutes when it wants a workspace with no transport underneath it.
///
/// Kept rather than deleted because "this build has no Tower-backed client for
/// this cartridge" is still a state the other three cartridges are in, and
/// because the shape of a client that reports one constant is the thing the
/// protocol's default `stateUpdates` was written for.
@MainActor
final class UnavailableWorldBuilderClient: WorldBuilderClient {
    /// Written for a person, not a log. The workspace shows this verbatim, so
    /// it has to explain the situation without implying either that something
    /// is broken or that a world is coming imminently.
    ///
    /// Note what it does **not** say: anything about what the Tower can or
    /// cannot do. This client has no channel through which it could know —
    /// that is precisely what makes it this client — and describing the other
    /// machine from here would be a fabricated report about it (Rule 3).
    static let reason = """
        This screen is not connected to a world builder. Nothing is being \
        asked of the Tower and nothing it may have built is being read.
        """

    let cartridgeID = "world-build"

    let state: WorldModelState = .unsupported(reason: UnavailableWorldBuilderClient.reason)

    init() {}
}

/// Publishes World Builder state into SwiftUI.
///
/// Separate from the client protocol because the protocol describes *supplying*
/// state, while this describes *publishing* it. When a Tower-backed client
/// exists it replaces the injected client, and this type starts republishing
/// real updates without any view changing.
///
/// ## Runtime ownership
///
/// **Holds no runtime references.** No `GlassesConnection`, no `TowerClient`,
/// no DAT object, no socket, and no `deinit` — so being destroyed when the
/// cartridge is deselected loses nothing real and tears nothing down. That is
/// what makes it safe as a workspace-owned `@StateObject`.
///
/// ## It does make HTTP requests, and that is not a contradiction
///
/// Geometry is fetched here, over HTTP, from the address the client hands it.
/// That is deliberate on both counts. **Over HTTP** because the Tower gives its
/// result sender and its frame path one shared lock, and a megabyte of points
/// down the WebSocket would starve `frame_result`. **From here** rather than
/// from the client because the fetch has no lifecycle: a request in flight when
/// this object is destroyed resolves into a `[weak self]` that is gone, and
/// nothing is left open. `URLSession.shared` is the app's, not this object's;
/// `WorldGeometryStore` is an actor holding a dictionary. Neither is a
/// connection, neither is torn down, and the wire this type still may not touch
/// — the socket — it still does not.
///
/// `TowerClientTests.testCartridgeViewModelsSendNothingToTheTower` remains the
/// enforcement of that, and remains true: it holds the socket to account, and
/// the client it is given publishes no geometry address to fetch from.
///
/// It holds a *subscription to its client*, which is a different thing: the
/// client is owned by `ProjectManager` and outlives the workspace, so a client
/// that has accumulated a partly-built world still has it when the cartridge is
/// reopened. The subscription is cancelled by `Set<AnyCancellable>`'s own
/// deallocation, which is why there is still no `deinit`.
///
/// Connectivity reaches it as a parameter (`isTowerReachable`), never as an
/// object it could act on. `CartridgeIntegrationTests` and
/// `TowerClientTests.testCartridgeViewModelsSendNothingToTheTower` check that
/// rather than leaving it to inspection.
@MainActor
final class WorldBuilderViewModel: ObservableObject {
    /// Seeded from the client and republished from `stateUpdates`. Nothing
    /// republishes it yet — the only client reports a constant — but the path
    /// exists, which is what makes "wiring a Tower-backed client is an
    /// injection, not a change of shape" a true statement rather than an
    /// aspiration.
    @Published private(set) var state: WorldModelState

    /// Live vs. stored-world inspection. Seeded from the client and
    /// republished from `inspectionUpdates`, for the reason `state` is: the
    /// client owns the pin, because the pin is a fact about its subscription.
    @Published private(set) var inspection: WorldInspectionMode

    /// The stored worlds the Tower listed, newest first, or empty until
    /// `loadWorlds()` has answered. Empty is also what a Tower with no world
    /// root reports; `worldListFailure` says which.
    @Published private(set) var worlds: [WorldListingEntry] = []

    /// Why the last `loadWorlds()` produced nothing, in a sentence, or `nil`
    /// after a listing that succeeded. `WorldListFetchError.notFound` is the
    /// Tower's own answer — no world root configured — and is worded as that.
    @Published private(set) var worldListFailure: String?

    /// The world whose interactive picture can be opened, or `nil` when none
    /// has been named yet.
    ///
    /// Set from two places and cleared from one. Opening a stored world sets
    /// it to the pin — the person chose that world, and if the Tower has built
    /// nothing for it the viewer shows the Tower's own sentence saying so.
    /// Geometry coordinates arriving on the status channel set it to the world
    /// and session the Tower named, which is the only way the *live* world
    /// earns one: a live world without geometry has no picture to open, and a
    /// button that could not lead anywhere is not offered. `returnToLive()`
    /// clears it, and the next report re-earns it.
    @Published private(set) var renderTarget: WorldRenderTarget?

    /// Whether the world on screen belongs to the capture the phone has open.
    ///
    /// Republished rather than derived, for the reason `state` is: the client
    /// owns the judgment, and a view model that recomputed it would be a second
    /// answer able to disagree with the one the state was gated on.
    @Published private(set) var sessionBinding: WorldSessionBinding
    /// The segments the Tower's manifest currently names, in the shape the
    /// gallery draws them.
    ///
    /// Empty until a manifest arrives, and emptied again when one arrives under
    /// a pose convention this build does not implement — an empty gallery says
    /// "nothing mapped", which is a true thing to say about geometry that
    /// cannot be read, whereas drawing it under the wrong convention would look
    /// like a room and mean nothing.
    @Published private(set) var fragmentsModel = WorldFragmentsModel(segments: [])

    /// The points and poses for those segments, keyed by
    /// `(content_hash, placement_hash)` — `WorldSegmentSummary.cacheKey`.
    ///
    /// Keyed by hash and not by index so that a segment re-solved under the
    /// same index cannot be drawn from the previous solve's points. Keyed by
    /// **both** hashes so that a segment whose points did not move but whose
    /// placement did cannot be drawn from the placement it used to have — the
    /// failure that looks like nothing at all, because the fragment simply
    /// sits in the wrong place forever. A segment whose chunk failed to fetch
    /// is simply absent here, and `FragmentCanvas` draws an empty tile for it
    /// rather than guessing.
    @Published private(set) var geometryChunks: [String: WorldSegmentChunk] = [:]

    private let client: any WorldBuilderClient
    private var cancellables: Set<AnyCancellable> = []

    /// Geometry transport. A `struct` and an `actor`, neither of which holds a
    /// connection: the client carries a `URL` and a `URLSession` — the app's
    /// shared one unless a test substitutes a stubbed one — and the store is a
    /// dictionary. So the claim in this type's doc comment, that it holds no
    /// runtime references and tears nothing down, still stands.
    private let geometry: WorldGeometryClient
    private let geometryStore = WorldGeometryStore()
    /// The saved-worlds list, over HTTP. A struct holding a `URL` and the
    /// shared session, like `geometry`, and defaulted for the same reason.
    private let library: WorldListClient

    /// The `geometry.revision` whose manifest is currently on screen, or `nil`
    /// when there is none.
    ///
    /// Cleared again after *any* fetch that failed — the manifest, or any one
    /// segment — so that the next report retries rather than being locked out.
    /// This marker is the only thing that unlocks a refetch, and a finalized
    /// world's revision never moves again, so leaving it set after a failure
    /// would make one refused request permanent.
    private var lastGeometryRevision: String?

    /// No default argument on `client`, deliberately.
    ///
    /// A default would make "swap the unavailable client for a Tower-backed one
    /// right here in the workspace view" the path of least resistance — and
    /// that client would hold a socket subscription and accumulated world
    /// state inside an object destroyed on every cartridge switch. The Product
    /// Shell V2 handoff §11 names that exact failure. Requiring injection means
    /// the correct wiring is the only wiring available.
    ///
    /// `geometry` **does** default, and the asymmetry is the point. The danger
    /// the paragraph above describes is a client that accumulates state and
    /// holds a subscription; `WorldGeometryClient` is a struct holding a `URL`
    /// and `URLSession.shared`, accumulates nothing, and owns no connection to
    /// lose. Its two properties were given defaults when it was written for
    /// exactly this reason — so a test can point it at a stubbed
    /// `URLSessionConfiguration` and watch what this type does when a fetch
    /// fails, which is behaviour no amount of reading proves.
    init(
        client: any WorldBuilderClient,
        geometry: WorldGeometryClient = WorldGeometryClient(),
        library: WorldListClient = WorldListClient()
    ) {
        self.client = client
        self.state = client.state
        self.sessionBinding = client.sessionBinding
        self.inspection = client.inspection
        self.geometry = geometry
        self.library = library

        client.stateUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in self?.state = state }
            .store(in: &cancellables)

        client.bindingUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] binding in self?.sessionBinding = binding }
            .store(in: &cancellables)
        client.inspectionUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] mode in self?.inspection = mode }
            .store(in: &cancellables)
        client.geometryUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] coordinates in self?.fetchGeometry(at: coordinates) }
            .store(in: &cancellables)
    }

    /// The synchronous half of the fetch: start it, and return.
    ///
    /// The `sink` closure must not block — it runs on the main queue, on the
    /// same turn the Tower's snapshot arrived — so the work goes into a `Task`
    /// and this returns immediately. `[weak self]` because the task outlives
    /// the sink call: a cartridge switch during a fetch leaves the request to
    /// resolve into a `self` that is gone, which drops it and opens nothing.
    private func fetchGeometry(at coordinates: WorldGeometryCoordinates) {
        Task { [weak self] in
            await self?.geometryDidChange(
                worldID: coordinates.worldID,
                sessionID: coordinates.sessionID,
                revision: coordinates.revision
            )
        }
    }

    /// Fetch the geometry the Tower has just named, unless it is the geometry
    /// already on screen.
    ///
    /// **Keyed on the revision, never on arrival.** The status channel
    /// heartbeats an unchanged snapshot about every two seconds, so a fetch
    /// triggered by a message rather than by a *changed* identity would pull a
    /// megabyte of points twice a second for a world that is standing still.
    ///
    /// Optional parameters, though `WorldGeometryCoordinates` carries none, so
    /// that the guard reads as one statement and so a future caller reaching
    /// this from a partially-known address is refused here rather than
    /// composing a URL out of what it happened to have.
    func geometryDidChange(worldID: String?, sessionID: String?, revision: String?) async {
        guard let worldID, let sessionID, let revision else { return }

        // The Tower has named a world with geometry; that is what the viewer
        // can draw. Recorded before the revision guard and before the fetch,
        // deliberately: a heartbeat under an unchanged revision still names
        // the world truthfully, and a manifest that fails — a 404 that raced
        // a rebuild — must not withhold a picture the render route would
        // serve. Guarded on equality so the two-second heartbeat does not
        // republish an unchanged value.
        let named = WorldRenderTarget(worldID: worldID, sessionID: sessionID)
        if renderTarget != named { renderTarget = named }

        guard revision != lastGeometryRevision else { return }
        lastGeometryRevision = revision

        guard let manifest = try? await geometry.manifest(
            worldID: worldID, sessionID: sessionID
        ) else {
            // Cleared rather than kept. A manifest that failed once — a 404
            // from a world root that was not configured yet, a request that
            // raced a rebuild — would otherwise be locked out until the world
            // happened to change again, and a *finalized* world never changes
            // again. The retry costs one small request on the next report.
            //
            // Guarded, because a newer call may already have claimed the
            // marker: clearing it then would make that newer fetch's own
            // result look superseded and be refetched from scratch.
            if revision == lastGeometryRevision { lastGeometryRevision = nil }
            logGeometry("manifest FAILED world=\(worldID) revision=\(revision) — will retry on the next report")
            return
        }

        // A newer report may have started its own fetch while this one was in
        // flight — likely on a live walk, where the revision moves about as
        // often as a fetch takes. The last writer must be the newest report and
        // not the slowest request, so a superseded fetch publishes nothing and
        // simply ends here.
        guard revision == lastGeometryRevision else { return }

        // A convention this build does not implement renders plausibly and
        // wrongly, so it renders not at all.
        guard manifest.poseConvention.matchesThisBuild else {
            fragmentsModel = WorldFragmentsModel(segments: [])
            geometryChunks = [:]
            logGeometry("manifest REFUSED — pose convention is not the one this build implements")
            return
        }

        // Everything the manifest still names is kept and everything else is
        // dropped, so a long walk does not accumulate superseded segments
        // forever. The cache and the published copy are rebuilt from the same
        // list in the same pass: a key held by one and not the other would
        // either refetch what is already in hand or draw what is already gone.
        //
        // `cacheKey`, not `contentHash`: a registration pass moves every
        // placement hash without moving a single content hash, so retaining by
        // content alone would keep 51 chunks that no key in the new manifest
        // can ever name again.
        await geometryStore.retainOnly(Set(manifest.segments.map(\.cacheKey)))

        logGeometry(
            "manifest world=\(worldID) revision=\(revision) segments=\(manifest.segments.count) "
                + "withPoints=\(manifest.segments.filter { $0.pointCount > 0 }.count) "
                + "current=\(manifest.current)"
        )

        var chunks: [String: WorldSegmentChunk] = [:]
        var fetched = 0
        var cacheHits = 0
        // A segment that could not be fetched is drawn as a blank tile — which
        // is honest — but the world must not be *left* that way. See the clear
        // at the end of this function.
        var anySegmentFailed = false
        for summary in manifest.segments {
            // The cache hit that decides whether a placement change is ever
            // seen. Keyed on `contentHash` this line is the whole bug: the
            // content hash of a segment that gained a placement is unchanged
            // BY DESIGN, so this would hit, the refetch below would be
            // skipped, and the unplaced chunk would be drawn for the life of
            // the world. `cacheKey` carries the placement half.
            if let cached = await geometryStore.chunk(forKey: summary.cacheKey) {
                chunks[summary.cacheKey] = cached
                cacheHits += 1
                continue
            }
            guard let chunk = try? await geometry.segment(
                worldID: worldID, sessionID: sessionID, index: summary.segmentIndex
            ) else {
                anySegmentFailed = true
                continue
            }
            await geometryStore.insert(chunk)
            fetched += 1
            // Filed under the chunk's OWN key, not the summary's. A rebuild
            // between the two requests returns different geometry under the
            // same segment index, and filing it under the key we asked for
            // would draw the new points inside the old segment's bounds. Filed
            // under its own, it simply does not match and is not drawn — which
            // is the honest outcome, and the next manifest resolves it.
            //
            // The key is composite, so this now also covers a REGISTRATION
            // landing between the manifest and the chunk: the points are the
            // same, the placement is not, and a chunk placed differently from
            // the row that asked for it is exactly as unusable as one built
            // from different points.
            chunks[chunk.cacheKey] = chunk
        }

        // Checked again, for the same reason: the segment fetches above are the
        // slow part, and a newer manifest may have landed during them.
        guard revision == lastGeometryRevision else { return }
        geometryChunks = chunks
        fragmentsModel = WorldFragmentsModel(
            segments: manifest.segments, isCurrent: manifest.current
        )

        // Published first, and *then* the marker is cleared: whatever did
        // arrive is on screen, and only the retry is rearmed. A partially
        // fetched world showing the segments it has beats showing none.
        //
        // Without this, one refused segment request would blank that fragment
        // for good. The revision is the only thing that unlocks a refetch, and
        // by the reasoning three guards above, a *finalized* world's revision
        // never moves again — so "until the world changes" would mean "never".
        // The manifest path clears the marker for the same reason and under the
        // same staleness guard, so a newer update already in flight is not
        // stomped.
        if anySegmentFailed, revision == lastGeometryRevision {
            lastGeometryRevision = nil
        }

        logGeometry(
            "segments drawn=\(chunks.count) fetched=\(fetched) cached=\(cacheHits) "
                + "points=\(chunks.values.reduce(0) { $0 + $1.points.count })"
                + (anySegmentFailed ? " SOME FAILED — will retry on the next report" : "")
        )
    }

    // MARK: Saved worlds

    /// Ask the Tower which worlds it holds. Called by the picker as it opens.
    ///
    /// `try?`-shaped like the geometry fetches, with the failure kept as a
    /// sentence rather than swallowed: the picker has to say why it is empty,
    /// and "no world root" and "the request failed" are different answers.
    func loadWorlds() async {
        do {
            worlds = try await library.worlds().worlds
            worldListFailure = nil
        } catch let error as WorldListFetchError {
            worlds = []
            switch error {
            case .notFound:
                worldListFailure = "The Tower answered that no world root is configured, so it has no saved worlds to list."
            case .undecodable:
                worldListFailure = "The Tower's world list could not be read as the contract this build implements."
            case .transport(let detail):
                worldListFailure = "The world list could not be fetched: \(detail)"
            }
            logGeometry("worlds FAILED — \(worldListFailure ?? "")")
        } catch {
            worlds = []
            worldListFailure = "The world list could not be fetched: \(error.localizedDescription)"
            logGeometry("worlds FAILED — \(error.localizedDescription)")
        }
    }

    /// Open a stored world. The client re-subscribes with the pin, and the
    /// pinned status payload then carries the geometry address exactly as the
    /// live one does — so the fetch path below needs no change.
    ///
    /// The gallery is cleared **here**, not when the new manifest lands: until
    /// it does, the fragments on screen would be the previous world's, under a
    /// heading naming this one.
    func open(worldID: String, sessionID: String?) {
        client.inspect(worldID: worldID, sessionID: sessionID)
        clearGeometry()
        renderTarget = WorldRenderTarget(worldID: worldID, sessionID: sessionID)
    }

    /// Back to the live world, by the same route.
    func returnToLive() {
        client.followLive()
        clearGeometry()
        renderTarget = nil
    }

    /// Forget what is drawn and rearm the fetch. A fetch already in flight for
    /// the previous world finds the marker moved and publishes nothing —
    /// the same staleness guard `geometryDidChange` already relies on.
    private func clearGeometry() {
        lastGeometryRevision = nil
        fragmentsModel = WorldFragmentsModel(segments: [])
        geometryChunks = [:]
    }

    /// The geometry pull, in the console.
    ///
    /// ## Why this exists, and what its absence cost
    ///
    /// This path had **no logging at all**, and it is the one that answers the
    /// program's central physical question — *do fragments appear while the
    /// wearer walks*. On the first real walk the phone produced 482 console
    /// lines across seven subsystems and **not one of them said whether the
    /// geometry manifest was ever fetched.** The status channel logs what the
    /// Tower said; nothing logged what the phone then went and got. So P3 came
    /// back unanswerable — not failed, unanswerable — and a walk is expensive
    /// to repeat.
    ///
    /// Deliberately one line per *manifest*, not per segment: on a live walk
    /// the revision moves every couple of seconds and a line per segment would
    /// be ~50 prints a tick, which is the noise level that made the camera path
    /// decimate its own logging.
    private func logGeometry(_ message: String) {
        #if DEBUG
        print("[Glasses][Geometry] \(message)")
        #endif
    }

    /// Why the cartridge is or is not usable, given the current connection.
    ///
    /// A function of the caller's connectivity rather than a stored property,
    /// so this object never holds a `TowerClient` and never answers from a
    /// stale copy of the connection state.
    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        client.availability(isTowerReachable: isTowerReachable)
    }

    /// The phase to draw, once availability has had its say.
    ///
    /// Availability outranks the client's own state: a Tower that cannot serve
    /// this cartridge makes every domain state moot, and letting `.idle` show
    /// through would invite a user to start something that cannot run.
    func phase(isTowerReachable: Bool) -> CartridgePhase {
        availability(isTowerReachable: isTowerReachable).forcedPhase ?? state.phase
    }

    /// The full explanation for the unavailable panel: the shared sentence about
    /// the Tower, plus whatever this cartridge's own state adds.
    func unavailableExplanation(isTowerReachable: Bool) -> String {
        availability(isTowerReachable: isTowerReachable)
            .explanation(cartridgeName: "World Builder", clientReason: clientReason)
    }

    /// The client's own words, when its state carries any.
    private var clientReason: String? {
        switch state {
        case .unsupported(let reason): return reason
        case .failed(let failure): return failure.message
        case .idle, .awaitingFirstUpdate, .receiving, .finalizing, .finalized: return nil
        }
    }
}
