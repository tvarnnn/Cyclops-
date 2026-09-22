//
//  WorldGeometry.swift
//  Glasses
//

import Foundation

/// The geometry agreement this build implements. Separate from the status
/// contract so either may move without the other, and opaque: compared for
/// equality only.
nonisolated enum WorldGeometryContract {
    static let identifier = "world_builder.geometry/2026-08-25"
}

/// The nine keys that decide what a pose means.
///
/// Every one of them renders plausibly and wrongly if guessed — inverting
/// `T_world_camera` still produces a map that looks like a map, and that was a
/// real shipped bug. So the convention travels on the wire, and this build
/// compares the five keys that change how a pose is interpreted
/// (`poseType`, `quaternionOrder`, `handedness`, `cameraAxes`,
/// `translationUnits`) before drawing anything. `upAxis` is deliberately
/// excluded — see `matchesThisBuild` below — because the 2D top-down view
/// does not depend on which way is up.
nonisolated struct WorldPoseConvention: Equatable, Sendable {
    let poseType: String
    let quaternionOrder: String
    let handedness: String
    let cameraAxes: String
    let translationUnits: String
    let worldAxesOrigin: String
    /// `"unknown"` today. It becomes a real axis when a floor plane exists,
    /// which is the signal a 3D renderer needs to stop guessing which way is up.
    let upAxis: String
    let poseDtype: String
    let pointDtype: String

    /// What this build knows how to draw. `upAxis` is deliberately absent:
    /// the 2D top-down view does not depend on it, so an unknown up-axis is
    /// not a mismatch.
    var matchesThisBuild: Bool {
        poseType == "T_world_camera"
            && quaternionOrder == "wxyz"
            && handedness == "right"
            && cameraAxes == "opencv_x_right_y_down_z_forward"
            && translationUnits == "world"
    }
}

// In an extension so the memberwise initialiser survives, as above.
nonisolated extension WorldPoseConvention {
    init?(json: [String: Any]) {
        guard
            let poseType = json["pose_type"] as? String,
            let quaternionOrder = json["quaternion_order"] as? String,
            let handedness = json["handedness"] as? String,
            let cameraAxes = json["camera_axes"] as? String,
            let translationUnits = json["translation_units"] as? String,
            let worldAxesOrigin = json["world_axes_origin"] as? String,
            let upAxis = json["up_axis"] as? String,
            let poseDtype = json["pose_dtype"] as? String,
            let pointDtype = json["point_dtype"] as? String
        else { return nil }
        self.poseType = poseType
        self.quaternionOrder = quaternionOrder
        self.handedness = handedness
        self.cameraAxes = cameraAxes
        self.translationUnits = translationUnits
        self.worldAxesOrigin = worldAxesOrigin
        self.upAxis = upAxis
        self.poseDtype = poseDtype
        self.pointDtype = pointDtype
    }
}

/// Whether a segment produced geometry, or produced nothing while looking.
nonisolated enum WorldSegmentResolution: String, Equatable, Sendable {
    case resolved
    case unresolved
}

nonisolated struct WorldBounds: Equatable, Sendable {
    let min: [Double]
    let max: [Double]

    init?(json: Any?) {
        guard
            let dict = json as? [String: Any],
            let min = dict["min"] as? [Double], min.count == 3,
            let max = dict["max"] as? [Double], max.count == 3
        else { return nil }
        self.min = min
        self.max = max
    }
}

nonisolated extension WorldBounds {
    /// A box built on this side, over points already transformed into one
    /// reference frame. `init?(json:)` in the struct body suppressed the
    /// synthesized memberwise initialiser, so this restores exactly that and
    /// nothing more — no validation, because the caller (`WorldClusterBuilder`)
    /// only ever passes triples it computed itself.
    init(min: [Double], max: [Double]) {
        self.min = min
        self.max = max
    }
}

/// Where a segment sits, as a three-valued fact rather than a bool.
///
/// `registered: false` alone conflates two different situations — "we tried
/// and the two independent solves disagreed" and "nobody has looked yet" — and
/// on the real corpus the refusal is the more interesting one: it is usually
/// *the wearer stood still*, which is a message about how to walk rather than
/// about the software. So the Tower sends this beside the bool, and this build
/// keeps them apart.
nonisolated enum WorldRegistrationState: String, Equatable, Sendable {
    case unplaced
    case registered
    case refused
}

/// The Sim3 that maps a segment's own frame into the frame of its
/// `reference_segment`: `p_ref = scale * (R(q) · p_segment) + t`.
///
/// **It is not a world-absolute pose.** There is no global world frame, and
/// there is no `reference_segment` that means "the world". Two segments are in
/// one space when they name the *same* reference under the *same* frame
/// revision, and in no other case.
///
/// ## What `nil` means where this type is optional
///
/// `transform_to_world: null` means the segment **is not registered into any
/// shared frame**. It does **not** mean identity. Treating it as identity
/// places the segment at the reference origin, which draws a plausible-looking
/// map of a room that was never observed — the exact failure
/// `WORLD-BUILDER-GEOMETRY.md` §6 names. That is why this is a real optional
/// all the way to the renderer and never gets a default.
///
/// ## The wrong-basin caveat, carried forward deliberately
///
/// On roughly **2% of scenes the two-view solve lands in a second,
/// self-consistent basin about 90° from the truth, reports `solved`, and looks
/// healthy** — measured 1 in 50 seeds, 865–925 matches, 527–870 inliers,
/// deterministic across runs. A transform decoded here can therefore be
/// well-formed, internally consistent, and wrong, and nothing on this side can
/// tell.
///
/// **The gate that would catch it was measured and REJECTED, and must not be
/// reintroduced here or anywhere else without new evidence.** The
/// discriminator — cheirality inliers over epipolar inliers — separates
/// cleanly on synthetic scenes, but real footage has a long low tail the
/// synthetic scenes do not, and a gate at the synthetic separation point
/// **refuses 17.1% of currently-solved edges**: a measured 17% loss traded
/// against an unmeasured 2.5% gain, on a corpus with no ground truth to settle
/// it. Whether it happens on real Ray-Ban footage is unknown, and the corpus
/// has no ground truth to detect it with — which is precisely what makes it
/// dangerous. PT-1 footage is what would unlock it. Adding a client-side
/// plausibility filter here would be the same rejected trade, made blind.
nonisolated struct WorldTransform: Equatable, Sendable {
    /// A quaternion, **w first** — not a matrix, so no row/column-major
    /// question arises. Validated as unit length on decode; see
    /// `quaternionNormTolerance`.
    let rotationWXYZ: [Double]
    /// In the **reference segment's** units. `translation_units` is `"world"`,
    /// **never metres**.
    let translation: [Double]
    /// Uniform. Applied after the rotation.
    let scale: Double
    /// Whose frame `p_ref` is in. There is no value of this that means "the
    /// world".
    let referenceSegment: Int
    /// Which gauge this Sim3 is expressed in. A coordinate stamped with one
    /// revision may not be reinterpreted under another, and a mismatch is a
    /// refuse-to-draw condition rather than something to guess past.
    let frameRevision: Int

    /// How far a quaternion may drift from unit length before it is refused.
    ///
    /// A norm off by *t* scales the geometry it rotates by *t*, silently and on
    /// top of `scale` — so this is a bound on silent scale error, and 1e-3 is
    /// 0.1%, below anything the reconstruction can resolve.
    ///
    /// **Deliberately not tighter**, and this is the Tower's own number for the
    /// Tower's own reason (`records.py`, `QUATERNION_NORM_TOLERANCE`): a first
    /// attempt at 1e-6 rejected real placements read back from disk, because a
    /// quaternion serialised at five decimal places lands ~1.9e-6 off unit. A
    /// validator tighter than the precision of the data it judges refuses its
    /// own valid input, and it does it silently — the rows drop and the world
    /// reads as unregistered.
    static let quaternionNormTolerance = 1e-3

    /// `p_ref = scale * (R(q) · p_segment) + t`.
    ///
    /// Returns `nil` for a point that is not a triple rather than indexing past
    /// the end of a short row — the wire is where a short row would come from,
    /// and `FragmentCanvas` already guards points the same way.
    func apply(to point: [Double]) -> [Double]? {
        guard point.count == 3 else { return nil }
        let w = rotationWXYZ[0]
        let x = rotationWXYZ[1]
        let y = rotationWXYZ[2]
        let z = rotationWXYZ[3]
        let (px, py, pz) = (point[0], point[1], point[2])
        // v' = v + 2w(u x v) + 2u x (u x v), with u the vector part. The
        // quaternion form rather than a built matrix: building a 3x3 is where
        // a transpose creeps in, and a transposed rotation still draws a map.
        let tx = 2 * (y * pz - z * py)
        let ty = 2 * (z * px - x * pz)
        let tz = 2 * (x * py - y * px)
        let rx = px + w * tx + (y * tz - z * ty)
        let ry = py + w * ty + (z * tx - x * tz)
        let rz = pz + w * tz + (x * ty - y * tx)
        return [
            scale * rx + translation[0],
            scale * ry + translation[1],
            scale * rz + translation[2],
        ]
    }

    /// Whether two placed segments are in one space.
    ///
    /// The **only** place this question is answered, so a second caller cannot
    /// answer it a second way. Same reference and same gauge, or nothing:
    /// segments with different reference segments must never be composited,
    /// and a coordinate stamped with one frame revision may not be
    /// reinterpreted under another.
    func sharesFrame(with other: WorldTransform) -> Bool {
        referenceSegment == other.referenceSegment
            && frameRevision == other.frameRevision
    }
}

// In an extension so the memberwise initialiser survives, as above.
nonisolated extension WorldTransform {
    /// Decodes and **validates**. Every refusal below is geometry that would
    /// otherwise be drawn.
    init?(json: Any?) {
        guard
            let dict = json as? [String: Any],
            let rotation = dict["rotation_wxyz"] as? [Double], rotation.count == 4,
            let translation = dict["translation"] as? [Double], translation.count == 3,
            let scale = dict["scale"] as? Double,
            let referenceSegment = dict["reference_segment"] as? Int,
            let frameRevision = dict["frame_revision"] as? Int
        else { return nil }

        // A non-finite transform is applied all the same and places the
        // geometry nowhere in particular.
        guard
            rotation.allSatisfy(\.isFinite), translation.allSatisfy(\.isFinite),
            scale.isFinite
        else { return nil }

        // A zero or negative scale collapses the segment to a dot at the
        // reference's origin, which the registration research records as
        // invisible in every aggregate metric.
        guard scale > 0 else { return nil }

        // The composition rule is defined only relative to a real reference,
        // and the contract calls a revision mismatch a refuse-to-draw
        // condition — so an impossible revision must not reach the renderer
        // as something to compare.
        guard referenceSegment >= 0, frameRevision >= 1 else { return nil }

        // The unit check. A non-unit quaternion scales the geometry it
        // rotates, silently and on top of `scale`, so a transform that fails
        // it is refused rather than normalised: normalising would invent a
        // rotation the Tower never sent and hide a disagreement the two sides
        // need to see.
        let norm = rotation.reduce(0) { $0 + $1 * $1 }.squareRoot()
        guard abs(norm - 1) <= Self.quaternionNormTolerance else { return nil }

        self.rotationWXYZ = rotation
        self.translation = translation
        self.scale = scale
        self.referenceSegment = referenceSegment
        self.frameRevision = frameRevision
    }
}

/// The manifest's top-level `scale` block: how, if at all, this world's units
/// relate to physical distance.
///
/// ## `unknown` is never `relative`, and `nil` is never 1.0
///
/// Two states are reachable on this hardware and only two. `relative` means
/// internally consistent with an arbitrary unit — **not metric**. `unknown`
/// means **no unit at all**, which is a strictly *weaker* claim, and mapping it
/// up to `relative` would assert an internal consistency nobody established. A
/// world with more than one segment stays `unknown` and will stay `unknown`:
/// segments do not share a coordinate frame, and calibration does not change
/// that — intrinsics unlock *poses*, not size.
///
/// `metersPerUnit` is an honest optional for the same reason `bounds` is.
/// `null` means no metric scale was ever established. It does not mean 1.0, and
/// there is deliberately no `?? 1.0` anywhere in this build.
///
/// `estimated` and `measured` have no code path that produces them on monocular
/// hardware and will not arrive. They are mapped rather than discarded so that
/// one arriving later is not silently downgraded into a claim weaker than the
/// one it made.
nonisolated struct WorldGeometryScale: Equatable, Sendable {
    let state: WorldScaleSemantics
    /// `nil` unless the state is metric — which is unreachable here. **Never
    /// read as 1.0.**
    let metersPerUnit: Double?

    /// The weakest claim available, and what an absent block decodes to.
    static let unknown = WorldGeometryScale(state: .unknown, metersPerUnit: nil)

    /// Whether a figure in these units may be converted to metres at all.
    ///
    /// Mirrors the Tower's own `ScaleState.allows_metres`, which admits
    /// `measured` and nothing else — so this is `false` on every payload this
    /// hardware can produce, and the conversion it guards is not written.
    var convertibleToMetres: Bool {
        state == .measuredMetric && metersPerUnit != nil
    }
}

// In an extension so the memberwise initialiser survives, as above.
nonisolated extension WorldGeometryScale {
    /// Never fails, and the fallback is the *weakest* claim rather than a
    /// convenient one: a manifest with no readable scale block has told us
    /// nothing about units, which is exactly what `unknown` says. Refusing the
    /// whole manifest instead would discard real geometry over metadata the
    /// viewer is forbidden from turning into a distance anyway.
    init(json: Any?) {
        guard let dict = json as? [String: Any] else {
            self = .unknown
            return
        }
        self.state = Self.semantics(dict["state"] as? String)
        self.metersPerUnit = dict["meters_per_unit"] as? Double
    }

    /// The Tower's world-builder vocabulary, which is **not** the status
    /// channel's.
    ///
    /// `tower/world_builder/schema.py` names these four —
    /// `unknown`/`relative`/`estimated`/`measured` — while the status payload
    /// sends iOS's own words (`relative`/`inferredMetric`/…). Two vocabularies
    /// for one concept is a real seam, and mapping it in one named function is
    /// what keeps the two from being confused at a call site.
    ///
    /// Anything unrecognised lands on `.unknown`, never `.relative`: a word
    /// this build does not know is a claim it cannot verify, and the weakest
    /// reading is the only safe one.
    static func semantics(_ word: String?) -> WorldScaleSemantics {
        switch word {
        case "relative": return .relative
        case "estimated": return .inferredMetric
        case "measured": return .measuredMetric
        default: return .unknown
        }
    }
}

/// The per-segment cache key, built in exactly one place.
///
/// ## Why `content_hash` alone is a trap
///
/// `content_hash` covers **poses and points only**, deliberately, so that a
/// segment that gains a placement keeps its content hash and every cached chunk
/// stays valid across a registration pass. That is safe *only* because
/// `placement_hash` moves instead. Keyed on the content half alone, the day a
/// segment gains a placement the client keeps its cached chunk forever and
/// draws an **unplaced** version of a segment the world now knows how to place.
/// Nothing throws, nothing logs, no tile goes blank — the fragment simply sits
/// in the wrong place, permanently.
///
/// ## Why the placement half is optional
///
/// The Tower added `placement_hash` **without bumping**
/// `world_builder.geometry/2026-08-25`, and the verbatim real-Tower fixtures in
/// `WorldGeometryTests` — 51 segments, captured before the field existed — do
/// not carry it. So it is decoded as optional and is *not* in either decoder's
/// required guard list; requiring it would refuse every payload from a Tower
/// that is otherwise speaking this exact contract.
///
/// `"-"` rather than the empty string for the absent case so the key of a
/// segment with no placement hash is visibly different from a segment whose
/// placement hash is somehow empty, and so a key is never `"h0:"`, which reads
/// like a truncation.
nonisolated enum WorldGeometryCacheKey {
    static func make(contentHash: String, placementHash: String?) -> String {
        "\(contentHash):\(placementHash ?? "-")"
    }
}

/// The five wire fields that say **where** a segment sits, decoded and
/// cross-checked in one place.
///
/// They ride on both the manifest row and the chunk with identical meanings, so
/// a rule enforced in one decoder and not the other would be a rule this client
/// does not actually have. `placement_hash` covers exactly this set —
/// `state`, `rotation_wxyz`, `translation`, `scale`, `reference_segment`,
/// `frame_revision` **and `refusal_reason`** — which is why they are decoded
/// together.
///
/// `refusal_reason` is inside that hash because it was once left out: on the
/// real corpus **26 of 29 segments are refused and share one
/// `placement_hash`**, so a re-registration that changed every refusal reason
/// moved no hash a client is told to key on, and a conforming client showed
/// stale reason text forever.
nonisolated struct WorldPlacementFields: Equatable, Sendable {
    let registered: Bool
    /// `nil` when the Tower did not send `registration_state` at all.
    ///
    /// **Not defaulted to `.unplaced`.** The whole reason this field exists is
    /// that `registered: false` cannot tell a refusal from an untried segment,
    /// so inventing `.unplaced` from the bool would fabricate the very
    /// distinction the field was added to carry.
    let state: WorldRegistrationState?
    /// Why this segment is not placed. Usually "the wearer stood still", which
    /// is a message to the wearer and not a fault.
    let refusalReason: String?
    /// `nil` means **not registered into any shared frame** — never identity.
    let transform: WorldTransform?
    /// `nil` only from a Tower predating the field. See `WorldGeometryCacheKey`.
    let placementHash: String?
}

nonisolated extension WorldPlacementFields {
    /// Returns `nil` when the payload disagrees with itself, and the caller
    /// drops the whole row rather than drawing half a claim.
    init?(json: [String: Any]) {
        guard let registered = json["registered"] as? Bool else { return nil }
        let rawTransform = json["transform_to_world"]
        let transform = WorldTransform(json: rawTransform)

        // A registered segment whose Sim3 is missing, malformed, non-unit or
        // non-positively scaled is a contract disagreement, not a segment to
        // place approximately. Half a transform means a default supplies the
        // rest, and the default is wrong.
        if registered, transform == nil { return nil }

        // And the converse, which the Tower enforces on its own side in the
        // same words: a refused or unplaced row must not carry a transform,
        // because anything with a transform gets drawn.
        if !registered, rawTransform is [String: Any] { return nil }

        self.registered = registered
        self.state = (json["registration_state"] as? String)
            .flatMap(WorldRegistrationState.init(rawValue:))
        self.refusalReason = json["registration_refusal_reason"] as? String
        self.transform = transform
        self.placementHash = json["placement_hash"] as? String
    }
}

nonisolated struct WorldSegmentSummary: Equatable, Sendable {
    let segmentIndex: Int
    /// Over the segment's poses and points, and **nothing about where it
    /// sits**. Half the cache key; see `cacheKey`.
    let contentHash: String
    let frameID: String
    /// False for every segment on the corpus this was built against. Two
    /// *unregistered* segments may never be drawn in one space — they share no
    /// coordinate frame and their scales disagree by up to ~87x on a real
    /// walk — and two *registered* ones may only be drawn together when
    /// `mayBeCompositedWith` says so.
    let registered: Bool
    /// Everything the Tower said about where this segment sits.
    let placement: WorldPlacementFields
    let resolutionState: WorldSegmentResolution
    /// Why this segment's refused poses were refused. Genuinely actionable:
    /// `low_parallax` means move sideways rather than turning on the spot.
    let dominantDegeneracy: String?
    let keyframeCount: Int
    let solvedCount: Int
    let pointCount: Int
    /// `nil` when the segment resolved to nothing. Never a zero-size box —
    /// absent and empty are different claims.
    let bounds: WorldBounds?
    /// How much the Tower's global solve trusts this segment's geometry:
    /// `"confident"`, `"partial"` or `"unresolved"` — `WORLD-BUILDER-GEOMETRY.md`
    /// §8. Carried as the Tower's own word rather than an enum so a value this
    /// build does not know is still shown verbatim rather than dropped.
    ///
    /// `nil` when the Tower sent nothing or `null`, which means **nothing has
    /// judged it** (no solve ran). It is not `"unresolved"` and it is not a
    /// weaker `"partial"`: absent is absent. Additive on the wire, so it is
    /// deliberately not in the decoder's required guard.
    let coverage: String?

    /// `nil` unless this segment is registered. **Never identity.**
    var transformToWorld: WorldTransform? { placement.transform }

    /// The cache key for this segment's chunk: content **and** placement.
    ///
    /// Keyed on the content half alone, a segment that gains a placement keeps
    /// its content hash, the client never refetches, and it draws an unplaced
    /// version of a segment the world now knows how to place — permanently,
    /// and without anything looking broken. See `WorldGeometryCacheKey`.
    var cacheKey: String {
        WorldGeometryCacheKey.make(
            contentHash: contentHash, placementHash: placement.placementHash
        )
    }

    /// Whether this segment and another may be drawn in **one** space.
    ///
    /// The single choke point for the composition rule, so that no caller gets
    /// to answer it a second way:
    ///
    /// - `registered: false` forbids it outright. Their scales disagree by up
    ///   to ~87x on a real walk, and a renderer that ignores that fabricates
    ///   geometry.
    /// - `transform_to_world: null` forbids it. Not registered is not identity.
    /// - **Different `reference_segment`s forbid it.** Same reference is fine;
    ///   different, never. There is no global world frame for them to fall back
    ///   into.
    /// - A different `frame_revision` forbids it: a coordinate stamped with one
    ///   gauge may not be reinterpreted under another.
    ///
    /// A segment is trivially compositable with itself only if it is placed at
    /// all, which is the honest answer for a one-segment "cluster".
    func mayBeCompositedWith(_ other: WorldSegmentSummary) -> Bool {
        guard registered, other.registered else { return false }
        guard let mine = placement.transform,
              let theirs = other.placement.transform
        else { return false }
        return mine.sharesFrame(with: theirs)
    }
}

// The JSON initialiser lives in an extension ON PURPOSE. Declaring an `init`
// inside the struct body suppresses Swift's memberwise initialiser, and the
// tests build these values directly rather than from JSON.
nonisolated extension WorldSegmentSummary {
    init?(json: [String: Any]) {
        // `placement_hash`, `registration_state` and
        // `registration_refusal_reason` are deliberately NOT in this guard.
        // The Tower added them without bumping
        // `world_builder.geometry/2026-08-25`, and the verbatim real-Tower
        // fixtures this decoder is pinned against predate all three — so
        // requiring them would refuse 51 segments of a payload from a Tower
        // that is speaking this exact contract. They are decoded as optional
        // inside `WorldPlacementFields` instead, and the cache key names the
        // absent case rather than pretending it is a value.
        guard
            let segmentIndex = json["segment_index"] as? Int,
            let contentHash = json["content_hash"] as? String,
            let frameID = json["frame_id"] as? String,
            let placement = WorldPlacementFields(json: json),
            let stateWord = json["resolution_state"] as? String,
            let state = WorldSegmentResolution(rawValue: stateWord),
            let keyframeCount = json["keyframe_count"] as? Int,
            let solvedCount = json["solved_count"] as? Int,
            let pointCount = json["point_count"] as? Int
        else { return nil }
        self.segmentIndex = segmentIndex
        self.contentHash = contentHash
        self.frameID = frameID
        self.registered = placement.registered
        self.placement = placement
        self.resolutionState = state
        self.dominantDegeneracy = json["dominant_degeneracy"] as? String
        self.keyframeCount = keyframeCount
        self.solvedCount = solvedCount
        self.pointCount = pointCount
        self.bounds = WorldBounds(json: json["bounds"])
        // `as? String` reads `null` and a missing key the same way, as nil.
        // Neither is a coverage claim.
        self.coverage = json["coverage"] as? String
    }
}

nonisolated struct WorldGeometryManifest: Equatable, Sendable {
    let worldID: String
    let sessionID: String
    /// Whether this geometry reflects every keyframe the Tower has accepted.
    ///
    /// `false` is the NORMAL state during a walk, not an error: the Tower
    /// rebuilds as it goes, and the next keyframe puts the finished build
    /// behind. The Tower used to hide behind-but-real geometry behind a 404,
    /// which meant the gallery stayed empty for the whole capture. It now
    /// serves it with this flag, and the flag is the only thing standing
    /// between "a partial world" and "the finished world".
    let current: Bool
    let geometryRevision: String
    let poseConvention: WorldPoseConvention
    /// How, if at all, this world's units relate to physical distance.
    ///
    /// Read and carried rather than ignored, because the alternative to
    /// carrying it is a viewer that has a number and no idea what it counts —
    /// and the unit a reader supplies for themselves is metres.
    let scale: WorldGeometryScale
    let segments: [WorldSegmentSummary]

    var resolvedSegments: [WorldSegmentSummary] {
        segments.filter { $0.resolutionState == .resolved }
    }

    /// Segments that hold keyframes and recovered nothing. On the real walk
    /// this was 32 of 51. They are counted, never placed: we know
    /// reconstruction failed, not where it failed.
    var unresolvedSegments: [WorldSegmentSummary] {
        segments.filter { $0.resolutionState == .unresolved }
    }
}

nonisolated struct WorldPose: Equatable, Sendable {
    let keyframeID: String
    let status: String
    let degeneracy: String
    let rotation: [Double]?
    /// `nil` means the pose was refused. The renderer draws a break here, not
    /// a line through the gap, and never substitutes zero.
    let translation: [Double]?
}

// In an extension so the memberwise initialiser survives, as above.
nonisolated extension WorldPose {
    init?(json: [String: Any]) {
        guard
            let keyframeID = json["keyframe_id"] as? String,
            let status = json["status"] as? String
        else { return nil }
        self.keyframeID = keyframeID
        self.status = status
        self.degeneracy = json["degeneracy"] as? String ?? ""
        self.rotation = json["rotation"] as? [Double]
        self.translation = json["translation"] as? [Double]
    }
}

nonisolated struct WorldSegmentChunk: Equatable, Sendable {
    let segmentIndex: Int
    /// Over this segment's whole poses and points — identical to the
    /// manifest's, and identical whether or not the cloud was sampled: the
    /// hash identifies the **segment**, not the transfer.
    let contentHash: String
    let registered: Bool
    /// Everything the Tower said about where this segment sits. Repeated on
    /// the chunk on purpose, exactly as `current` is: a client holding a
    /// cached chunk and not re-reading the manifest would otherwise have
    /// nothing to place it by.
    let placement: WorldPlacementFields
    let poses: [WorldPose]
    /// In the **segment's own frame**. `transformToWorld` is what maps them
    /// into the reference segment's frame, and there is no other route: a
    /// point drawn without it is drawn where the segment thinks it is, which
    /// is only ever true of that segment alone.
    let points: [[Double]]
    let pointsSent: Int
    let pointsTotal: Int
    let pointSampling: String

    /// True when the cloud on screen is not the whole cloud. The UI must say
    /// so rather than let a coarse world read as a complete one.
    var isSampled: Bool { pointsSent < pointsTotal }

    /// `nil` unless this segment is registered. **Never identity.**
    var transformToWorld: WorldTransform? { placement.transform }

    /// Content **and** placement. See `WorldGeometryCacheKey` for why keying on
    /// the content half alone is a failure that looks like nothing at all.
    var cacheKey: String {
        WorldGeometryCacheKey.make(
            contentHash: contentHash, placementHash: placement.placementHash
        )
    }

    /// This segment's points in its **reference segment's** frame:
    /// `p_ref = scale * (R(q) · p_segment) + t`.
    ///
    /// `nil` — not the segment-local points — when there is no transform,
    /// because "not registered" is not "registered at the origin". A caller
    /// that wants the untransformed cloud already has `points`, and one that
    /// wanted a shared frame must be told it does not have one rather than
    /// handed coordinates that silently mean something else.
    ///
    /// These may only be drawn beside another segment's when
    /// `WorldSegmentSummary.mayBeCompositedWith` says the two share a frame.
    var pointsInReferenceFrame: [[Double]]? {
        guard let transform = placement.transform else { return nil }
        return points.compactMap(transform.apply(to:))
    }
}

nonisolated enum WorldGeometryDecoder {

    static func manifest(from json: [String: Any]) -> WorldGeometryManifest? {
        guard
            json["contract"] as? String == WorldGeometryContract.identifier,
            let worldID = json["world_id"] as? String,
            let sessionID = json["session_id"] as? String,
            let revision = json["geometry_revision"] as? String,
            let conventionJSON = json["pose_convention"] as? [String: Any],
            let convention = WorldPoseConvention(json: conventionJSON),
            let rawSegments = json["segments"] as? [[String: Any]]
        else { return nil }

        // A row that will not decode drops the whole manifest rather than
        // silently shrinking the world.
        var segments: [WorldSegmentSummary] = []
        for raw in rawSegments {
            guard let segment = WorldSegmentSummary(json: raw) else { return nil }
            segments.append(segment)
        }

        return WorldGeometryManifest(
            worldID: worldID, sessionID: sessionID,
            // Absent means TRUE, and that is not an optimistic default. A
            // Tower old enough not to send this field is one that answered
            // 404 for anything behind the journal -- so on that Tower,
            // geometry that arrived at all was current by construction.
            current: json["current"] as? Bool ?? true,
            geometryRevision: revision,
            poseConvention: convention,
            // Absent decodes to `unknown`, which is the WEAKEST claim
            // available and not a convenient one. `unknown` must never become
            // `relative`; see `WorldGeometryScale`.
            scale: WorldGeometryScale(json: json["scale"]),
            segments: segments
        )
    }

    static func chunk(from json: [String: Any]) -> WorldSegmentChunk? {
        // `placement_hash` is absent from this guard for the same reason it is
        // absent from the manifest row's: the Tower added it without bumping
        // the contract, and the real-Tower chunk fixture predates it. See
        // `WorldGeometryCacheKey`.
        guard
            json["contract"] as? String == WorldGeometryContract.identifier,
            let segmentIndex = json["segment_index"] as? Int,
            let contentHash = json["content_hash"] as? String,
            let placement = WorldPlacementFields(json: json),
            let rawPoses = json["poses"] as? [[String: Any]],
            let points = json["points"] as? [[Double]],
            let sent = json["points_sent"] as? Int,
            let total = json["points_total"] as? Int,
            let sampling = json["point_sampling"] as? String
        else { return nil }

        var poses: [WorldPose] = []
        for raw in rawPoses {
            guard let pose = WorldPose(json: raw) else { return nil }
            poses.append(pose)
        }

        return WorldSegmentChunk(
            segmentIndex: segmentIndex, contentHash: contentHash,
            registered: placement.registered, placement: placement,
            poses: poses, points: points,
            pointsSent: sent, pointsTotal: total, pointSampling: sampling
        )
    }
}
