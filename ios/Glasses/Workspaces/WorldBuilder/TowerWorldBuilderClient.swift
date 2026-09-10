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
    ///
    /// `world_builder.status/2026-09-06` supersedes `.../2026-08-25` because
    /// `model_state` gained a word. `interrupted` is what a builder that died
    /// mid-walk now reports, with the snapshot and the geometry beside it;
    /// under the old identifier this build refused any word it did not know
    /// as `.undecodableResponse`, and an older app must be told to update
    /// rather than shown that. The same payload also gained `selection` —
    /// why this world is on the wire — and `lifecycle.finalization`, both
    /// additive; the new word is what earned the bump.
    ///
    /// `world_builder.status/2026-09-10` supersedes `.../2026-09-06` for a
    /// subtler reason: no word was added, and a state this app already
    /// decodes started arriving where a different one used to. A
    /// `lifecycle.state: "stopped_unbuilt"` with `geometry.available: false`
    /// now projects to `interrupted` instead of `finalizing`, because there
    /// is nothing to wait for — the old projection put a **permanent
    /// "Finalizing"** on this screen over a walk that had produced no
    /// geometry, and four separate reviews found it by four separate routes.
    ///
    /// A build that adopts the string without reading `geometry.available`
    /// beside `lifecycle.state` gets the right answer anyway, because the
    /// stage ladder switches on `model_state`. The bump is here because the
    /// MEANING moved, and this file's own comment above says what serving
    /// an old identifier over new meaning is: the silent widening the
    /// refusal exists to prevent.
    ///
    /// **A Tower on this branch will refuse a phone built before this
    /// line.** That is deliberate and it is loud, which is the whole point
    /// of a dated identifier — but it makes rebuilding the app a
    /// precondition for the retest rather than a caution.
    static let identifier = "world_builder.status/2026-09-10"
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
            // Two Tower facts share this word, and `lifecycle.build_in_progress`
            // tells them apart. Since 2026-09-06 the live builder keeps the
            // writer lock through finalization, so `true` means a live
            // process is finishing this world — evidence, not inference. On a
            // record from before that, the Tower still cannot see whether a
            // build is running, sends `null`, and the state means only "the
            // stored figures are not the final figures". Carried as-is either
            // way; the canvas words the two differently.
            let lifecycle = payload["lifecycle"] as? [String: Any] ?? [:]
            return .finalizing(
                snapshot ?? WorldSnapshot(),
                buildInProgress: lifecycle["build_in_progress"] as? Bool
            )

        case "finalized":
            return .finalized(snapshot ?? WorldSnapshot())

        case "interrupted":
            // The session ended abnormally and the Tower can still describe
            // the world: `world_snapshot`, `geometry`, `session` and
            // `trajectory` all travel with this word. The reason is the
            // Tower's prose ("the process building this world exited without
            // stopping its session; …") and is shown verbatim.
            //
            // Without a snapshot there is nothing to draw, and a report about
            // a session that ended badly with no world to show is, for the
            // screen, a failure the Tower reported. That branch is not
            // produced by today's Tower and exists so an empty payload cannot
            // become an empty world.
            let interruption = reason ?? Self.unexplainedInterruption
            guard let snapshot else {
                return .failed(CartridgeFailure(kind: .towerReportedFailure, message: interruption))
            }
            return .interrupted(snapshot, reason: interruption)

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

    static let unexplainedInterruption = """
        The Tower reported that this world's session was interrupted, and did not say how.
        """

    /// The `model_state` word itself, or `nil` when the payload has none.
    ///
    /// Read beside `modelState(from:)` rather than recovered from the decoded
    /// case, because the decoded case is the phone's reading and this is the
    /// Tower's word — and `WorldRecentReference` labels a world with the
    /// Tower's word deliberately.
    static func modelStateWord(from payload: [String: Any]) -> String? {
        payload["model_state"] as? String
    }

    /// The `selection` block → `WorldSelection`, or `.unknown` when the block
    /// is absent or carries no `mode`.
    ///
    /// `.unknown`, not `.none`: an older Tower that sends no block has not
    /// said nothing is live, it has said nothing about how it chose — and the
    /// client keeps its pre-2026-09-06 behaviour under `.unknown` for exactly
    /// that reason. A `mode` this build does not know survives as its own
    /// word; see `WorldSelectionMode`.
    static func selection(from payload: [String: Any]) -> WorldSelection {
        guard
            let json = payload["selection"] as? [String: Any],
            let mode = json["mode"] as? String
        else { return .unknown }
        return WorldSelection(
            mode: WorldSelectionMode(rawValue: mode),
            worldID: json["world_id"] as? String,
            sessionID: json["session_id"] as? String,
            reason: json["reason"] as? String
        )
    }

    /// `lifecycle.finalization` → `WorldFinalizationReport`, or `nil` for
    /// `null`, for an absent key, and for a payload with no `lifecycle` block
    /// at all. A record written before the builder kept this report has none,
    /// and that is a fact rather than an error.
    static func finalization(from payload: [String: Any]) -> WorldFinalizationReport? {
        let lifecycle = payload["lifecycle"] as? [String: Any] ?? [:]
        return WorldFinalizationReport(json: lifecycle["finalization"])
    }

    /// `world.updated_at`, when the payload carries the Tower-native `world`
    /// block. Read for the "last saved world" line only — it is a clock
    /// reading, not a figure, and it is never drawn as a duration.
    static func worldUpdatedAt(from payload: [String: Any]) -> Double? {
        let world = payload["world"] as? [String: Any] ?? [:]
        return world["updated_at"] as? Double
    }

    /// The stored world a `latest` selection offered, as a reference the
    /// person can follow. `nil` when the payload names no world — which is
    /// the `none` selection, and has nothing to offer.
    static func recentReference(from payload: [String: Any]) -> WorldRecentReference? {
        let snapshot = payload["world_snapshot"] as? [String: Any] ?? [:]
        let session = payload["session"] as? [String: Any] ?? [:]
        let selection = self.selection(from: payload)
        guard
            let worldID = (snapshot["world_id"] as? String) ?? selection.worldID,
            let word = modelStateWord(from: payload)
        else { return nil }
        return WorldRecentReference(
            worldID: worldID,
            sessionID: (session["session_id"] as? String) ?? selection.sessionID,
            name: snapshot["name"] as? String,
            modelState: word,
            updatedAt: worldUpdatedAt(from: payload)
        )
    }

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
                // Counted by the Tower, never derived here. `segments - 1`
                // was the old sentence's number and it was wrong by 57 on
                // the 2026-09-09 walk; see `WorldTrajectoryReport.segments`.
                // Absent on a Tower that predates the field, and the view
                // stays silent rather than falling back to arithmetic.
                trackingRestarts: evidence["tracking_restarts"] as? Int,
                chainBreaks: evidence["chain_breaks"] as? Int,
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

    /// The stored world the Tower offered in place of a live one, while
    /// following live, or `nil`.
    ///
    /// Set when an unpinned report carries a `latest` selection — nothing is
    /// live and the Tower answered with the newest world on disk. That report
    /// is presented as `.idle` rather than as the stored world's own state,
    /// and this is what remains of it: a name, a state word and an Open
    /// action, so the person can choose to look at history rather than have
    /// it drawn under the Live heading. Cleared by any report that is not
    /// that, and whenever `lastReport` is.
    private(set) var recentWorld: WorldRecentReference? {
        didSet {
            guard recentWorld != oldValue else { return }
            recentWorldSubject.send(recentWorld)
        }
    }

    var recentWorldUpdates: AnyPublisher<WorldRecentReference?, Never> {
        recentWorldSubject.eraseToAnyPublisher()
    }

    /// Why the Tower served the world in the last report, or `nil` before the
    /// first one. Read-only, for the log line and for tests; the decision it
    /// drives is in `publishLastReport()`.
    var selection: WorldSelection? { lastReport?.selection }

    /// The builder's account of finalization from the last report, or `nil`.
    ///
    /// Stored and published rather than computed off `lastReport`, and the
    /// difference is not cosmetic. `state` above drops repeats at the source,
    /// so a report that moves `final_solve` from `pending` to `solved` while
    /// the snapshot stands still used to announce nothing at all — leaving
    /// "The final pass has not run yet." on screen for the whole length of a
    /// final solve, which is the exact window that sentence is about. Kept
    /// current by `lastReport`'s `didSet`, so every assignment to the report,
    /// including the one that clears it, is covered by one line.
    private(set) var finalization: WorldFinalizationReport? {
        didSet {
            guard finalization != oldValue else { return }
            finalizationSubject.send(finalization)
        }
    }

    var finalizationUpdates: AnyPublisher<WorldFinalizationReport?, Never> {
        finalizationSubject.eraseToAnyPublisher()
    }

    /// The pin the next `result_subscribe` carries, or `nil` to follow the
    /// live world. Kept across reconnects on purpose: a reader looking at a
    /// stored world who loses WiFi is still looking at that world when it
    /// comes back.
    private var pinned: (worldID: String, sessionID: String?)?

    private let stateSubject = PassthroughSubject<WorldModelState, Never>()
    private let bindingSubject = PassthroughSubject<WorldSessionBinding, Never>()
    private let inspectionSubject = PassthroughSubject<WorldInspectionMode, Never>()
    private let recentWorldSubject = PassthroughSubject<WorldRecentReference?, Never>()
    private let finalizationSubject = PassthroughSubject<WorldFinalizationReport?, Never>()
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
    ///
    /// **Gated on presentation, though.** An address is emitted only when the
    /// state this client just presented carries a snapshot. A report the gate
    /// refused (foreign, awaiting) or the ownership rule set aside (`latest`
    /// while live) names a world this screen is not showing, and an address
    /// for it would still fetch its fragments and name it for the picture
    /// button — which is how a previous walk's gallery came to sit under the
    /// next walk's "Building" heading.
    var geometryUpdates: AnyPublisher<WorldGeometryCoordinates, Never> {
        geometrySubject.eraseToAnyPublisher()
    }

    private let geometrySubject = PassthroughSubject<WorldGeometryCoordinates, Never>()
    private let tower: TowerClient
    private var cancellables: Set<AnyCancellable> = []

    /// One decoded status payload, before the gate.
    ///
    /// `state` and `session` are what the gate reads; `selection` is what the
    /// ownership rule reads; the identity fields and `finalization` are kept
    /// for the log line, the "last saved world" reference and tests.
    struct StatusReport: Equatable, Sendable {
        var state: WorldModelState
        var session: WorldSessionReport?
        var selection: WorldSelection
        var finalization: WorldFinalizationReport?
        /// `world_snapshot.world_id` / `session.session_id`, as decoded — the
        /// same two strings `WorldGeometryCoordinates` is addressed by.
        var worldID: String?
        var sessionID: String?
        /// The stored world this report offered instead of a live one, when
        /// its selection was `latest`; `nil` otherwise.
        var recentWorld: WorldRecentReference?
    }

    /// The last thing the Tower said, before the gate.
    ///
    /// Kept because the binding has two inputs and only one of them arrives on
    /// the result channel: the phone's bracket opens and closes on its own
    /// clock, and when it does the same payload has to be re-judged. Storing
    /// the decoded report rather than the raw dictionary keeps the decode on
    /// the arrival path, where a failure is still attributable to a message.
    private var lastReport: StatusReport? {
        didSet {
            // Nothing offered stands once the report that offered it is gone.
            if lastReport == nil { recentWorld = nil }
            // Every assignment, including the clearing one above. Its own
            // `didSet` drops repeats, so a two-second heartbeat carrying an
            // unchanged finalization publishes nothing.
            finalization = lastReport?.finalization
        }
    }

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
    /// Subscribes sent on the current socket that the Tower has not yet
    /// answered — with a `result_subscribed`, or with an error in its place.
    ///
    /// `isSubscribing` cannot count. A pin change (`inspect`, `followLive`)
    /// legitimately sends a second subscribe while the first is still
    /// unanswered, because the new pin has to go out now; and the Tower opens
    /// **both**. Its acks arrive in the order the subscribes were sent, on one
    /// ordered channel, so an ack that lands while a newer subscribe is still
    /// pending belongs to an attempt this client has already superseded. That
    /// subscription is closed and retired on the spot rather than becoming
    /// `subscriptionID` for the instant before the next ack overwrites it —
    /// which used to leave it open and delivering the world just left, every
    /// heartbeat, for the life of the socket.
    private var pendingSubscribeAcks = 0
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
            pendingSubscribeAcks = 0
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
        pendingSubscribeAcks += 1
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
            if pendingSubscribeAcks > 1 {
                // An ack for a subscribe this client has already superseded
                // with a newer one — a pin changed before the Tower answered.
                // The Tower opened it and will heartbeat the world just left
                // on it until told otherwise; tell it now, and drop whatever
                // it had already queued. See `pendingSubscribeAcks`.
                pendingSubscribeAcks -= 1
                tower.unsubscribeFromResults(subscriptionID: ack.subscriptionID)
                retiredSubscriptionIDs.insert(ack.subscriptionID)
                return
            }
            pendingSubscribeAcks = 0
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
        let payload = envelope.payload
        let selection = WorldBuilderResultDecoder.selection(from: payload)
        let session = WorldBuilderResultDecoder.session(from: payload)
        lastReport = StatusReport(
            state: next,
            session: session,
            selection: selection,
            finalization: WorldBuilderResultDecoder.finalization(from: payload),
            worldID: (payload["world_snapshot"] as? [String: Any])?["world_id"] as? String,
            sessionID: session?.sessionID,
            recentWorld: selection.isHistoryOfferedAsLive
                ? WorldBuilderResultDecoder.recentReference(from: payload)
                : nil
        )
        publishLastReport()

        // Sent whether or not the state changed, and whether or not the
        // geometry did. See `geometryUpdates` for why this one is not filtered
        // here. Absent when the Tower has built nothing yet, which is the
        // common case for most of a session's first seconds.
        //
        // Here rather than in `publishLastReport()` because this is a fact the
        // Tower just stated, and `publishLastReport()` also runs on a
        // *bracket* change, where the Tower has said nothing new.
        //
        // And only when what was just presented carries a snapshot. A report
        // the gate turned into "waiting", or the ownership rule turned into
        // `.idle`, names a world this screen is not showing; emitting its
        // address anyway would fetch that world's fragments into the cache
        // and name it for the picture button, to be drawn the moment the
        // state next became a world state — for a *different* world.
        guard state.snapshot != nil else { return }
        if let coordinates = WorldBuilderResultDecoder.geometryCoordinates(from: payload) {
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

        // ## Live versus History
        //
        // Following live, a report whose selection is `latest` is **not** the
        // live world: the Tower has said nothing is live and it answered with
        // the most recently updated world on disk. That is history, and
        // history is reached by pinning (`inspect`), never by arriving on the
        // unpinned subscription. So the report is presented as the empty
        // state — `.idle`, or "waiting" if this phone has a capture open and
        // the Tower has simply not attached a builder yet — and what it
        // offered is kept as `recentWorld`, for an Open action to pin.
        //
        // `live` and `finalizing` selections are the current world and go
        // through the gate as before. `.unknown` — an older Tower, no block —
        // keeps the gate as the only judge, because a Tower that has not said
        // how it chose has not said the world is stored, either. A pinned
        // report is history by construction and the selection is not read.
        if pinned == nil, report.selection.isHistoryOfferedAsLive {
            recentWorld = report.recentWorld
            let binding = bindingWithNoReport
            sessionBinding = binding
            state = WorldSessionGate.presented(.idle, binding: binding)
            return
        }
        recentWorld = nil

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
        case .receiving(let snapshot),
             .finalizing(let snapshot, _),
             .finalized(let snapshot),
             .interrupted(let snapshot, _):
            let name: String
            switch state {
            case .receiving:
                name = "receiving"
            case .finalizing(_, let building):
                name = "finalizing(building=\(building.map { String($0) } ?? "unknown"))"
            case .interrupted(_, let reason):
                name = "interrupted(\(reason))"
            default:
                name = "finalized"
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
        print("[Glasses][WorldBuilder] \(detail) binding=\(bindingDescription) selection=\(selectionDescription)")
        #endif
    }

    /// The Tower's selection, for the log line: the one word that says
    /// whether a world arrived because it is live or because it was newest.
    private var selectionDescription: String {
        guard let selection = lastReport?.selection else { return "-" }
        return "\(selection.mode.rawValue)(\(selection.worldID ?? "no world"))"
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

        // A reply to a subscribe names the cartridge and no subscription id
        // (none was opened). It answers one pending subscribe the way an ack
        // would have, and the count has to say so, or the next real ack is
        // taken for a superseded one. `unknown_subscription`, which answers an
        // unsubscribe, names only the id and does not reach here.
        if error.cartridge != nil, error.subscriptionID == nil, pendingSubscribeAcks > 0 {
            pendingSubscribeAcks -= 1
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
