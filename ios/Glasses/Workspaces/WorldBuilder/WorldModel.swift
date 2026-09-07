//
//  WorldModel.swift
//  Glasses
//

import Foundation

/// The boundary between the World Builder workspace and whatever the Tower
/// eventually builds.
///
/// **Nothing in this file is a Tower protocol.** The Tower's World Builder is
/// being designed independently and its wire contract does not exist yet, so
/// inventing one here would guarantee a rewrite and — worse — would let the UI
/// display shapes the Tower never agreed to produce. What this file does
/// instead is state the *presentation's* requirements: the smallest set of
/// facts the workspace needs in order to draw something honest.
///
/// Read the types below as a question addressed to the Tower — "can you tell us
/// these things, and in what units?" — not as an answer. When the real contract
/// lands, the reconciliation is one new `WorldBuilderClient` implementation plus
/// whatever mapping it needs; no view changes.
///
/// `WorldScaleSemantics` used to live here and now lives in
/// `Cartridges/Integration/Observation.swift`, unchanged — Scene Understanding
/// needs the same rule, and one copy of it is the only safe number of copies.
///
/// The full list of what the Tower must be able to report is in
/// `docs/agent-handoffs/IOS-TO-TOWER.md`, which supersedes and extends the
/// shorter list in `docs/agent-handoffs/product-shell-v2-handoff.md` section 10.

// MARK: - Tracking

/// How well the Tower believes it is following the camera through space.
///
/// Deliberately coarse. A percentage would imply a calibrated confidence model
/// that neither side has defined.
enum WorldTrackingQuality: Equatable, Sendable {
    case good
    case limited
    case lost
    /// The Tower does not report tracking, or is not tracking at all.
    case unavailable

    var displayName: String {
        switch self {
        case .good: return "Good"
        case .limited: return "Limited"
        case .lost: return "Lost"
        case .unavailable: return "—"
        }
    }
}

// MARK: - Calibration

/// Whether the Tower has established the reference it needs before any spatial
/// figure means anything.
///
/// Coarse for the same reason tracking is: the calibration *procedure* is the
/// Tower's to design — it may be an initialisation motion, a known-object
/// reference, a plane fit, or nothing at all — and iOS must not presume which.
/// What the UI needs is only whether a figure is yet trustworthy enough to
/// show, and that has four honest answers.
///
/// Note the deliberate absence of a progress percentage. "Calibrating, 62%"
/// implies a measurable denominator, and no such quantity has been defined.
enum WorldCalibrationState: Equatable, Sendable, CaseIterable {
    /// The Tower has not reported calibration at all. Not the same as
    /// uncalibrated — one is silence, the other is a report.
    case unknown
    /// Reported, and not established.
    case uncalibrated
    /// Reported as in progress.
    case calibrating
    /// Reported as established. **Says nothing about accuracy** — a calibrated
    /// monocular pipeline still produces inferred depth, and
    /// `WorldScaleSemantics` remains the authority on how a figure may be
    /// presented.
    case calibrated

    var displayName: String {
        switch self {
        case .unknown: return "Not reported"
        case .uncalibrated: return "Not calibrated"
        case .calibrating: return "Calibrating…"
        case .calibrated: return "Calibrated"
        }
    }

    /// Whether spatial figures derived under this calibration may be shown at
    /// all. False for everything but `.calibrated`: an uncalibrated figure is
    /// not a rough figure, it is an unanchored one.
    var permitsSpatialFigures: Bool { self == .calibrated }
}

// MARK: - Geometry

/// What the Tower says its world representation *is*, without this app deciding
/// what that may be.
///
/// ## The single most important refusal in this file
///
/// The Tower has not chosen a representation. It could be a point cloud, sparse
/// landmarks, a pose graph, a mesh, trajectory-only, or something none of us
/// has named. Encoding any one of those here — even as an enum case — would
/// prejudge a decision that is not the phone's to make, and would leave a
/// rendering path that quietly expects the wrong shape.
///
/// So `representation` is an **opaque label**. This app stores it, shows it
/// verbatim, and never branches on it. `elementCount` is likewise a number the
/// Tower keeps, of a unit the Tower names; iOS neither knows nor claims to know
/// what one element is, which is why the count is always displayed next to the
/// label rather than alone.
///
/// The consequence is that a workspace can honestly say "the Tower reports a
/// representation of this kind, and this build cannot draw it" — a true and
/// useful statement — instead of either drawing nothing or drawing a guess.
struct WorldGeometryReport: Equatable, Sendable {
    /// The Tower's own name for what it built. Displayed verbatim, never
    /// parsed, never matched against a known set.
    var representation: String?
    /// How many units of that representation exist, in whatever unit the
    /// representation implies.
    var elementCount: Int?
    /// Whether the Tower sends changes or whole snapshots. `nil` when it has
    /// not said — which matters, because a UI that assumes incremental updates
    /// will show a partial world as a complete one.
    var isIncremental: Bool?

    init(representation: String? = nil, elementCount: Int? = nil, isIncremental: Bool? = nil) {
        self.representation = representation
        self.elementCount = elementCount
        self.isIncremental = isIncremental
    }

    /// True when the Tower reported geometry of some kind. A report with only a
    /// count and no name is still a report — it just cannot be labelled.
    var hasReport: Bool {
        representation != nil || elementCount != nil
    }
}

// MARK: - Trajectory

/// The path the camera took, summarised.
///
/// ## Why there are no poses in here
///
/// A pose array requires a pose schema — position, rotation convention, handedness,
/// coordinate frame, units — and every one of those is a Tower decision that
/// does not exist yet. Getting any of them wrong produces a path that renders
/// plausibly and is wrong, which is worse than not rendering one.
///
/// What can be held without that schema is the Tower's own summary of the path:
/// how many poses it kept, and how far it thinks the camera travelled. The
/// distance carries `WorldScaleSemantics` because it is a spatial figure and
/// `docs/modules/WORLD-BUILD.md` admits no unlabelled ones — on monocular RGB
/// it will normally be `.relative`, in which case it is a shape statistic and
/// not a number of metres, and `distanceDisplayable` refuses to show it as one.
///
/// ## A camera position and a segment origin are not the same figure
///
/// `world_builder.status/2026-08-25` exists because they were being added
/// together. Under the superseded contract `pose_count` was
/// `keyframes - poses_refused`, and the Tower's build counts a segment ANCHOR
/// as neither solved nor refused — so every anchor was promoted to a camera
/// position. On the 2026-08-24 walk that produced *"Camera poses: 36"* from a
/// build whose manifest read `poses_solved: 0, points: 0`: nothing had been
/// reconstructed, and 36 was the segment count.
///
/// An anchor is definitional, not measured — identity rotation, zero
/// translation — so the two are now separate figures and this type keeps them
/// separate. `posesAnchor` is reported *beside* `poseCount`, never folded into
/// it, which is what lets an uncalibrated walk read as "36 segment origins, no
/// trajectory" instead of as a path.
struct WorldTrajectoryReport: Equatable, Sendable {
    /// Poses carrying a position that is **evidence**: every solved pose, plus
    /// the anchor of each segment that solved something.
    ///
    /// `0` is a real answer and is not `nil`. A build that solved nothing
    /// reports zero here however many anchors it produced, because a segment
    /// that resolved nothing contributes the origin of an empty coordinate
    /// frame rather than a camera position.
    var poseCount: Int?
    /// How many of the build's poses were segment anchors.
    ///
    /// Never added to `poseCount`. Present so the panel can say what the walk
    /// actually produced when no camera was positioned at all.
    var posesAnchor: Int?
    /// The underlying figures the Tower counted, carried for the same reason
    /// the count above is: an anchor-only build is a different thing from a
    /// build that refused everything, and only these tell them apart.
    var posesSolved: Int?
    var posesRefused: Int?
    /// Tracking segments. A break means tracking was lost, and poses either
    /// side of one share no coordinate frame — which is why the Tower refuses
    /// a path length across more than one.
    var segments: Int?
    /// Path length, in the unit the Tower names below.
    var pathLength: Double?
    /// The Tower's unit string for `pathLength`, if it gave one.
    ///
    /// **Never assumed to be metres.** `WorldScaleSemantics.inferredMetric`
    /// says a figure is *metric in kind*; it does not say what unit it counts
    /// in, and the Tower has named none. Rendering "14.2 m" from a bare
    /// `Double` would invent the unit — the same mistake `CVMetric.unit`
    /// deliberately refuses to make two files away.
    var pathLengthUnit: String?
    /// How `pathLength` was arrived at. `.unknown` by default, because an
    /// unlabelled distance is the thing WORLD-BUILD.md forbids.
    var scale: WorldScaleSemantics

    init(
        poseCount: Int? = nil,
        posesAnchor: Int? = nil,
        posesSolved: Int? = nil,
        posesRefused: Int? = nil,
        segments: Int? = nil,
        pathLength: Double? = nil,
        pathLengthUnit: String? = nil,
        scale: WorldScaleSemantics = .unknown
    ) {
        self.poseCount = poseCount
        self.posesAnchor = posesAnchor
        self.posesSolved = posesSolved
        self.posesRefused = posesRefused
        self.segments = segments
        self.pathLength = pathLength
        self.pathLengthUnit = pathLengthUnit
        self.scale = scale
    }

    /// Whether the path length may be shown as a physical distance.
    ///
    /// Only when the scale claims to be metric at all. A `.relative` path
    /// length is a number in arbitrary units, and printing "14.2 m" beside it
    /// would be a fabrication of exactly the kind WORLD-BUILD.md names.
    var distanceDisplayable: Bool {
        guard pathLength != nil else { return false }
        switch scale {
        case .inferredMetric, .measuredMetric: return true
        case .relative, .unknown: return false
        }
    }

    /// Whether the path length may be shown **as the labelled figure it is**,
    /// which is a weaker question than `distanceDisplayable` and has a
    /// different answer.
    ///
    /// The Tower reconstructs monocular RGB, so `.relative` is the best scale
    /// it can reach and `distanceDisplayable` is correctly false for every
    /// figure it will ever send. Refusing on that basis alone would mean this
    /// panel never shows a path length at all — and "2.9 world units", with the
    /// Tower's own unit attached and its own scale named beside it, is not a
    /// distance claim. It is the honest rendering of a shape statistic.
    ///
    /// The gate is the **unit**, not the scale. A bare number is what
    /// `ReportedFigure` exists to prevent: without a unit a reader supplies
    /// their own, and the one they supply is metres. `.unknown` scale is
    /// excluded separately — the Tower sends no distance figure at all in that
    /// state, because it could not be labelled.
    var labelledFigureDisplayable: Bool {
        guard pathLength != nil, let pathLengthUnit, !pathLengthUnit.isEmpty else { return false }
        return scale != .unknown
    }

    /// Whether the Tower positioned a camera anywhere at all.
    ///
    /// `false` for `poseCount == 0`, which is the uncalibrated case, and also
    /// `false` when the count is absent — "no trajectory to draw" is the same
    /// answer either way, and the two are still distinguished everywhere the
    /// difference is a claim rather than a drawing decision.
    var hasPositionedPoses: Bool { (poseCount ?? 0) > 0 }

    /// The uncalibrated outcome, named: the build produced segment origins and
    /// positioned no camera.
    ///
    /// This is the figure the panel shows instead of a pose count, because
    /// "36 segment origins" is a precise description of what happened and
    /// "0 camera poses" alone is not.
    var isAnchorsOnly: Bool { poseCount == 0 && (posesAnchor ?? 0) > 0 }

    var hasReport: Bool {
        poseCount != nil
            || pathLength != nil
            || posesAnchor != nil
            || segments != nil
    }
}

// MARK: - Persistence

/// Whether the world survives the session, and whether this app is looking at a
/// live one or a stored one.
///
/// The Tower owns persistence entirely (`docs/04-MODULE-SYSTEM.md` — each module
/// owns its storage namespace); iOS stores no world data and must not imply it
/// could. These cases describe only what the Tower reported about its own
/// storage.
enum WorldPersistenceState: Equatable, Sendable {
    /// The Tower did not say. The honest default, and distinct from `.session`:
    /// silence is not a promise that the world is discarded.
    case unknown
    /// Exists for this session only and will not be reloadable.
    case session
    /// The Tower saved it. `revision` is whatever marker the Tower uses to
    /// distinguish one saved version from another, so the UI can tell a reload
    /// apart from a repeat. Opaque, compared for equality only — see
    /// `WorldSnapshot.revision`.
    case saved(revision: String?)
    /// A saved world is being loaded back.
    case reloading

    var displayName: String {
        switch self {
        case .unknown: return "Not reported"
        case .session: return "This session only"
        case .saved: return "Saved"
        case .reloading: return "Loading…"
        }
    }
}

/// Whether the workspace is watching a world being built or examining one the
/// Tower already finished.
///
/// Two modes rather than a boolean because the difference changes what every
/// control on screen means: in `.inspecting` there is no capture to start, and
/// a counter that moves would be a bug rather than progress.
enum WorldInspectionMode: Equatable, Sendable {
    /// Following the world the current session is contributing to.
    case live
    /// Looking at a stored world. Carries the identifier so the UI can say
    /// *which*, and so it cannot silently drift onto a different one.
    case inspecting(worldID: String?)

    var isInspecting: Bool {
        if case .inspecting = self { return true }
        return false
    }
}

// MARK: - Snapshot

/// One observation of the world the Tower is building.
///
/// Every field is optional, and that is the point: a Tower that reports only a
/// keyframe count must not force the UI to invent a world name, a tracking
/// verdict, or a scale. Absent means absent, and the views render it as such
/// (docs/02-DEVELOPMENT-RULES.md Rule 3).
struct WorldSnapshot: Equatable, Sendable {
    /// Human-readable name, if the Tower names worlds at all.
    var name: String?
    /// Stable identifier for the world being built, if it has one.
    var worldID: String?
    /// Keyframes accepted into the reconstruction.
    var keyframeCount: Int?
    /// A marker that changes when the world does, so the UI can tell "new data"
    /// from "same data" without diffing geometry.
    ///
    /// Deliberately a `String` and not an `Int`. Nothing in this app orders or
    /// compares revisions — inequality is the whole requirement — so an integer
    /// would presume a monotonic counter the Tower has not agreed to keep. This
    /// is the same reasoning `CartridgeContract.identifier` uses, applied to the
    /// same class of problem.
    var revision: String?
    var tracking: WorldTrackingQuality
    var scale: WorldScaleSemantics
    /// Seconds of mapping the Tower has accumulated, if it keeps that clock.
    /// Deliberately not derived from an iOS timer: the iPhone's idea of elapsed
    /// time is not the Tower's idea of mapping time, and showing one as the
    /// other would be a fabrication.
    var mappingSeconds: TimeInterval?
    /// Whether the Tower has established a spatial reference.
    var calibration: WorldCalibrationState
    /// What the Tower built, in the Tower's own words.
    var geometry: WorldGeometryReport
    /// The camera path, summarised by the Tower.
    var trajectory: WorldTrajectoryReport
    /// What the Tower did with the world when it was done.
    var persistence: WorldPersistenceState

    init(
        name: String? = nil,
        worldID: String? = nil,
        keyframeCount: Int? = nil,
        revision: String? = nil,
        tracking: WorldTrackingQuality = .unavailable,
        scale: WorldScaleSemantics = .unknown,
        mappingSeconds: TimeInterval? = nil,
        calibration: WorldCalibrationState = .unknown,
        geometry: WorldGeometryReport = WorldGeometryReport(),
        trajectory: WorldTrajectoryReport = WorldTrajectoryReport(),
        persistence: WorldPersistenceState = .unknown
    ) {
        self.name = name
        self.worldID = worldID
        self.keyframeCount = keyframeCount
        self.revision = revision
        self.tracking = tracking
        self.scale = scale
        self.mappingSeconds = mappingSeconds
        self.calibration = calibration
        self.geometry = geometry
        self.trajectory = trajectory
        self.persistence = persistence
    }

    /// True when the Tower has told us nothing beyond "a world exists".
    var isEmpty: Bool {
        name == nil
            && worldID == nil
            && keyframeCount == nil
            && revision == nil
            && mappingSeconds == nil
            && tracking == .unavailable
            && scale == .unknown
            && calibration == .unknown
            && !geometry.hasReport
            && !trajectory.hasReport
            && persistence == .unknown
    }

    /// Whether any spatial figure in this snapshot may be presented as a
    /// distance.
    ///
    /// Both gates, in one place so no view can satisfy one and forget the
    /// other: the pipeline has to have a reference (`calibration`) *and* the
    /// figure has to claim metric provenance (`scale`). On the current
    /// monocular hardware the honest outcome is usually `false`, and a `false`
    /// here means the layout omits the row rather than dashing it.
    var permitsMetricDisplay: Bool {
        calibration.permitsSpatialFigures && (scale == .inferredMetric || scale == .measuredMetric)
    }
}

// MARK: - State

/// What the world-visualization container should be showing.
///
/// The case that matters today is `.unsupported`. The Tower cannot build a
/// world at all yet — `docs/03-ROADMAP.md` puts the module container at V0.8
/// and the first module at V0.9, and neither exists — so that is the only state
/// the app can currently reach, and the only one it may claim.
///
/// The remaining cases are the shape the container must be able to take, so the
/// view is written once against the full lifecycle instead of being rebuilt
/// when the contract lands. They are unreachable today by construction: the
/// only `WorldBuilderClient` that exists returns `.unsupported` and nothing else.
enum WorldModelState: Equatable, Sendable {
    /// No Tower-side world builder exists. Carries the reason so the view never
    /// has to compose an explanation of its own.
    case unsupported(reason: String)
    /// A world builder exists, but this session has not begun feeding it.
    case idle
    /// Capture is running; the Tower has not yet returned anything about a
    /// world. Distinct from `.receiving` so "we are sending frames" is never
    /// drawn as "a world is appearing".
    case awaitingFirstUpdate
    /// The Tower is actively reporting world updates.
    case receiving(WorldSnapshot)
    /// Capture has ended and the Tower is still working — closing loops,
    /// optimising, writing. Separate from `.finalized` because the world on
    /// screen is not yet the world that will be stored, and separate from
    /// `.receiving` because no new observations are arriving. Both distinctions
    /// change what the UI may claim, which is what earns the case.
    ///
    /// `buildInProgress` is the Tower's `lifecycle.build_in_progress`, carried
    /// rather than assumed. Since `world_builder.status/2026-09-06` the live
    /// builder keeps the writer lock through finalization, so `true` here is
    /// evidence — a live process holds the lock and the session has stopped.
    /// `nil` is the older record: the Tower cannot see whether a build is
    /// running, and says so. `false` is not sent for this state today and
    /// is drawn like `nil`, because "not building" is a claim this app
    /// cannot check either.
    case finalizing(WorldSnapshot, buildInProgress: Bool? = nil)
    /// Capture ended and the world is final and inspectable.
    case finalized(WorldSnapshot)
    /// The session ended abnormally — the builder died mid-walk, was killed
    /// while finalizing, or recorded an error — and the Tower can say so.
    ///
    /// **Not `.failed`, and not `.finalized`.** The 2026-09-06 walk left 463
    /// keyframes of derived geometry and a working render behind a builder
    /// that exited without closing its session; drawn as "failed" the phone
    /// showed none of it, and drawn as "finished" it would have presented a
    /// half-built walk as a complete one. The snapshot travels with the
    /// state so the rows, the gallery and the picture can be shown, and the
    /// Tower's own reason travels beside it so the headline can say what
    /// happened without this app composing a sentence about the other
    /// machine.
    ///
    /// Nothing more will arrive for this world — that is what makes it a
    /// settled state rather than a live one.
    case interrupted(WorldSnapshot, reason: String)
    /// Carries `CartridgeFailure` rather than a bare string, matching the other
    /// three cartridges. A `reason: String` cannot distinguish a dropped socket
    /// from a payload this build could not decode from the Tower reporting its
    /// own module failed — which is the entire point of `CartridgeFailure.Kind`,
    /// and the three call for different responses.
    case failed(CartridgeFailure)

    /// The snapshot to draw metrics from, when there is one.
    var snapshot: WorldSnapshot? {
        switch self {
        case .receiving(let snapshot),
             .finalizing(let snapshot, _),
             .finalized(let snapshot),
             .interrupted(let snapshot, _):
            return snapshot
        case .unsupported, .idle, .awaitingFirstUpdate, .failed:
            return nil
        }
    }

    /// Whether the Tower is currently contributing to a world. Drives the
    /// "live" affordance, and must be false in every state where the app would
    /// otherwise be implying activity it cannot verify.
    ///
    /// False in `.finalizing`: the Tower is working, but it is not observing,
    /// and a badge that says "live" while the camera is off is a lie about the
    /// sensor rather than about the compute. False in `.interrupted` for the
    /// stronger reason that nothing is working at all.
    var isReceivingUpdates: Bool {
        if case .receiving = self { return true }
        return false
    }

    /// Whether a world exists to look at, mapped or finished.
    var hasWorld: Bool {
        switch self {
        case .receiving, .finalizing, .finalized, .interrupted: return true
        case .unsupported, .idle, .awaitingFirstUpdate, .failed: return false
        }
    }

    /// The shared coarse phase, for the shell's generic panel and for the
    /// cross-cartridge invariant tests. The domain cases above stay the
    /// authority for anything World Builder actually decides.
    var phase: CartridgePhase {
        switch self {
        case .unsupported: return .unsupported
        case .idle: return .idle
        case .awaitingFirstUpdate: return .waiting
        // Both are the Tower doing work that may still change the answer.
        case .receiving, .finalizing: return .live
        // Both are worlds nothing will add to. An interrupted world is not
        // `.failed`: that phase draws no data, and the whole point of the
        // case is that there is data to draw.
        case .finalized, .interrupted: return .settled
        case .failed: return .failed
        }
    }
}
