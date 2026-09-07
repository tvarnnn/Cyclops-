//
//  SceneUnderstandingModel.swift
//  Glasses
//

import Foundation

/// The Scene Understanding cartridge, as the Tower actually serves it.
///
/// Contract: `scene_understanding.live/2026-08-27`, `docs/contracts/TOWER-UNIFIED-CARTRIDGES.md`
/// §7 and `tower/docs/contracts/CARTRIDGE-RESULTS.md` §14. Every type below was
/// written against bytes curled off a running Tower, not against a summary of
/// them.
///
/// ## The result type is `live`, and that is not a naming detail
///
/// World Builder's `status` describes a *build* — how far it has got. This
/// payload **is** the answer: what is in front of the camera right now. There
/// is no progress to report and nothing accumulates, which is why the state
/// vocabulary below has no `finalizing` and no `finalized`.
///
/// ## What this file used to contain, and why it does not any more
///
/// It used to model `SceneEntity`, `SceneRelationship`, `SceneTrackID`,
/// `ScenePosition` and `SceneFrameOfReference`: per-entity rows, each with a
/// track handle, a bearing and a distance. The Tower refuses every one of them,
/// and refuses them **structurally** —
///
/// > There is no key anywhere in this payload that could hold an entity or a
/// > relation.
///
/// `tracks`, `relations` and `confidence` are `null` on every payload, each
/// with an `*_absent_reason` and a refusal list naming `track_id`, `box`,
/// `facing`, `visible_eyes` and `confidence`. iOS asked for a session-scoped
/// anonymous track handle (`IOS-to-Tower.md` §4.1) and a signed bearing (§4.3);
/// V1 serves neither, and the handoff is explicit that if that is the wrong
/// trade it becomes a contract change with a new identifier, **not a field
/// quietly populated later**.
///
/// Types that can never be filled are not harmless. They are an invitation:
/// the next person to read them concludes the data is coming and builds the
/// screen that waits for it. Worse, `SceneTrackID` plus a timestamp is exactly
/// the joinable pair the Tower refuses to publish — a stable handle lets a
/// recipient assemble the per-person dwell timeline this cartridge keeps none
/// of. Modelling the shape is the first half of laundering persists-nothing
/// onto the consumer. So they are gone, and what replaced them is the
/// aggregate the wire actually carries: a count, a per-label side histogram,
/// and a people block that is a count and nothing else.
///
/// ## What remains, and what each surviving rule is for
///
/// **No identity.** There is no name, no descriptor, no handle, and no field
/// that could carry one. No face recognition exists on this platform: no
/// detector is present, and the keypoints the pose model produces locate eyes
/// and ears as anonymous landmarks that yield no descriptor and support no
/// matching.
///
/// **No gaze.** `SceneFacing` describes body and head orientation and the
/// labels say "facing your direction" — never "looking at you". There is no eye
/// tracking on the target glasses, so there is no gaze to report at any
/// confidence.
///
/// **No absence claims.** A count of zero means zero *confirmed tracks in the
/// camera's forward cone*, which is a statement about the camera and not about
/// the room — and it is an undercount besides. `absence_means:
/// "not-visible-to-this-cartridge"` is the Tower saying so in the payload.
///
/// **No distances and no bearings.** Not because they would be unlabelled, but
/// because they are not on the wire at all. `where` is a coarse signed bearing
/// under another name, and it is published as side *counts* per label rather
/// than one side per label, because one side cannot describe a chair on the
/// left and a chair on the right.

// MARK: - Lifecycle

/// The session's own state, in the Tower's closed vocabulary.
///
/// `unrecognised` exists so a state this build has never heard of reaches the
/// screen as itself rather than being folded into `stopped` — the two call for
/// opposite things to be said to a person, and a decoder that silently picked
/// one would be inventing the answer.
enum SceneLifecycleState: Equatable, Sendable {
    case stopped
    case starting
    case running
    case paused
    case failed
    /// A word outside `lifecycle.states`. Carried verbatim.
    case unrecognised(String)

    init(_ word: String) {
        switch word {
        case "stopped": self = .stopped
        case "starting": self = .starting
        case "running": self = .running
        case "paused": self = .paused
        case "failed": self = .failed
        default: self = .unrecognised(word)
        }
    }

    var wireValue: String {
        switch self {
        case .stopped: return "stopped"
        case .starting: return "starting"
        case .running: return "running"
        case .paused: return "paused"
        case .failed: return "failed"
        case .unrecognised(let word): return word
        }
    }

    /// The **only** state in which a scene may be shown as "last known".
    ///
    /// This property is the enforcement point for the rule in §4 of the
    /// contract, and it is deliberately a property of the lifecycle rather than
    /// a condition somebody has to remember to write at the call site. See
    /// `SceneUnderstandingState.lastKnown`.
    var mayHoldLastKnownScene: Bool { self == .paused }
}

/// `lifecycle`, whole.
///
/// **Every field is non-optional unless the wire itself carries `null`.** The
/// contract is explicit that every key is present in every state, and modelling
/// a present-but-null field as an absent one would lose the difference between
/// "zero of these" and "this Tower did not say".
struct SceneLifecycle: Equatable, Sendable {
    let state: SceneLifecycleState
    /// The closed vocabulary the Tower published, carried so a client can show
    /// that an `unrecognised` state really was outside it.
    let states: [String]
    /// Increments once per Start, and is meaningless across a restart.
    ///
    /// > Two payloads with different `session_id` came from different tracking
    /// > sessions and must not be compared.
    ///
    /// `SceneUnderstandingState` cannot express that rule on its own — it holds
    /// one reading at a time — so the client that holds readings across time is
    /// where it is enforced, by discarding what it held when this changes.
    let sessionID: Int
    /// `true` only while running. Not derived here: the Tower computes it, and
    /// two answers to one question is how they come to disagree.
    let sceneIsCurrent: Bool
    /// Non-null only in `failed`.
    let failureReason: String?
    /// Tower-receipt time, in unix seconds. Never a capture clock.
    let startedAt: Double?
    /// When the model finished loading. Tower-receipt.
    let readyAt: Double?
    /// Non-null only while still loading.
    let loadingSeconds: Double?
    /// The load has taken longer than `loadOverdueAfterSeconds`.
    ///
    /// **Not a failure.** Nothing can interrupt a blocking model load, and a
    /// first-run weight download is slow and still correct. A client that drew
    /// this as an error would tell someone to give up on a load that was going
    /// to succeed.
    let loadOverdue: Bool
    /// `120.0` on this Tower. Read rather than assumed.
    let loadOverdueAfterSeconds: Double
    /// Whether `stream_start` starts this session and `stream_stop` ends it.
    ///
    /// `true` by default for Scene Understanding, which is what makes the
    /// cartridge reachable from a phone at all: **the phone sends nothing to
    /// open a cartridge.** Document Memory's defaults to `false`, and the
    /// asymmetry is the difference between the two — that one writes.
    let followsStream: Bool

    /// The sentence a person is owed when a load has gone long. Kept as a
    /// constant so no screen can render `loadOverdue` as a failure by
    /// accident.
    static let loadOverdueNote = """
        The detector is still loading and has taken longer than expected. That \
        is not a failure — a model load cannot be interrupted, and a first-run \
        download of the weights is slow and still correct.
        """
}

// MARK: - Counts

/// One reported class and how many confirmed tracks of it are in view.
///
/// A row rather than a dictionary entry because the order matters on screen and
/// `reported_classes` fixes it: the class list is the *universe* of labels, and
/// rendering it in the Tower's order keeps a reader's eye in the same place
/// between two readings.
struct SceneClassCount: Equatable, Identifiable, Sendable {
    /// A COCO class name, carrying COCO's meanings. `mouse` is the pointing
    /// device and `tv` is any large display — worth saying out loud, because a
    /// reader who has not been told will supply the everyday meaning.
    let label: String
    /// Confirmed tracks, never raw detections. Present at `0` rather than
    /// omitted: a class silently absent would be indistinguishable from one
    /// that was looked for and not seen.
    let count: Int

    var id: String { label }
}

/// `where`, for one label: how many of them are on each side.
///
/// Side **counts**, not a side. One side cannot describe a chair on the left
/// and a chair on the right, and picking one would be a wrong answer where a
/// refusal was available.
///
/// `unknown` is its own bucket rather than being folded into `centre`: a scene
/// whose frame size was never learned has not placed anything in the middle of
/// the view, it has placed nothing.
struct SceneSideCounts: Equatable, Sendable {
    let left: Int
    let centre: Int
    let right: Int
    let unknown: Int

    /// Overflow-safe. These four are `as? Int ?? 0` straight off the wire and
    /// this runs on the WebSocket push path for every positions row, so a Tower
    /// sending four large integers would trap on the `+` rather than render a
    /// strange total. `&+` is wrong here (it would wrap to a negative count);
    /// saturating at `Int.max` keeps the number meaningless-but-safe, and
    /// `isEmpty` — the only thing that reads it for a decision — stays correct.
    var total: Int {
        var sum = 0
        for part in [left, centre, right, unknown] {
            let (next, overflowed) = sum.addingReportingOverflow(part)
            if overflowed { return .max }
            sum = next
        }
        return sum
    }
    var isEmpty: Bool { total == 0 }
}

/// One label's positions.
struct SceneLabelPositions: Equatable, Identifiable, Sendable {
    let label: String
    let sides: SceneSideCounts

    var id: String { label }
}

// MARK: - People

/// People, as a count and an aggregate. **Never a list.**
///
/// There is no per-person row anywhere on this wire and no key that could hold
/// one, so there is no per-person row here. What the Tower publishes is a
/// count, three qualifications on that count, and an aggregate orientation
/// figure that is `null` — never `0` — when nothing measured it.
struct ScenePeople: Equatable, Sendable {
    /// Confirmed person tracks in the camera's forward cone.
    let count: Int
    /// `true` on this platform, always.
    ///
    /// Every `person` box in the only real corpus this host has is the wearer's
    /// own torso: median box bottom edge 0.985, 58.4% touching the frame edge.
    /// A count rendered without this reads as a count of *other people*.
    let mayIncludeWearer: Bool
    /// `false` on this platform, always. No bystander footage exists here, so
    /// nothing in this block has been checked against ground truth.
    let validated: Bool
    /// How many are oriented toward the wearer.
    ///
    /// **`nil` is not zero, and this is the field on this payload where the
    /// difference is most likely to be mistaken.** `nil` means orientation
    /// never produced an estimate, or every estimate has expired; `0` means it
    /// measured and found none. A screen that renders both as "0 facing you"
    /// has turned an observation gap into an observation of absence.
    let facingWearer: Int?
    /// Whether `facingWearer` is a measurement. Redundant with `facingWearer !=
    /// nil` by construction, and carried anyway: the Tower states it, and a
    /// client asserting the redundancy is how a future divergence goes unseen.
    let facingAnswered: Bool
    /// Why `facingWearer` is `nil`. Prose, shown verbatim.
    let facingUnavailableReason: String?
    /// People whose orientation is unknown. `nil` when orientation is off
    /// entirely — there is no "unknown" to count when nothing was attempted.
    let facingUnknown: Int?
    /// Age of the stalest orientation estimate, in seconds.
    let oldestEstimateSeconds: Double?
    /// The two buckets that exist: `facing_wearer` and `unknown`.
    let facingStatesReported: [String]
    /// `away_from_wearer` and `profile`, which do not.
    ///
    /// Carried so a screen can say the remainder is undifferentiated rather
    /// than let a reader do arithmetic that does not close.
    let facingStatesWithheld: [String]
    let facingStatesWithheldReason: String?
    /// The Tower's own wording for what facing means. Data, not decoration.
    let facingNote: String?

    /// Person boxes cut off by the bottom edge with no head region in view,
    /// kept OUT of `count`. From a camera worn at head height that is most
    /// often the wearer's own hands, arms, lap or legs. Since 2026-09-07;
    /// defaults to 0 for an older Tower, which is the reading that adds
    /// nothing rather than the one that invents something.
    var partialBottomEdge: Int = 0
    /// How many counted people are `large`, `medium`, `small` or `unknown`
    /// in the frame — sizes in the picture, never distances. Empty for an
    /// older Tower.
    var byApparentSize: [String: Int] = [:]
    /// The Tower's own wording for what a size means. Shown, not paraphrased.
    var apparentSizeNote: String?
    /// `"experimental"` while the facing stage has been validated only on
    /// still photographs. `nil` for an older Tower.
    var orientationStatus: String?

    /// `count − facingWearer − facingUnknown`, when both are measurements.
    ///
    /// Named a remainder rather than a category, because that is what it is:
    /// the two withheld states have no bucket, so this number is not "people
    /// facing away".
    var undifferentiatedRemainder: Int? {
        guard let facingWearer, let facingUnknown else { return nil }
        // Subtraction before the clamp traps on overflow, and all three are
        // unvalidated wire integers on the push path. Reporting form first,
        // then the clamp that was always intended.
        let (afterWearer, o1) = count.subtractingReportingOverflow(facingWearer)
        guard !o1 else { return nil }
        let (remainder, o2) = afterWearer.subtractingReportingOverflow(facingUnknown)
        guard !o2 else { return nil }
        return max(0, remainder)
    }
}

// MARK: - Orientation vocabulary

/// Which way a person is oriented, relative to the wearer.
///
/// ## The wording is the feature
///
/// `docs/07-PLATFORM-CONSTRAINTS.md` Limitation 8:
///
/// > Something appearing in the glasses camera does not prove the user looked
/// > directly at it, noticed it, read it, understood it, or interacted with it.
///
/// The same holds in the other direction: a person's body facing the camera
/// does not establish that they are looking at the wearer, and the glasses have
/// no sensor that could establish it. So `.towardCamera` reads **"Facing your
/// direction"** and never "Looking at you", "Watching you", or a softened
/// version of either. A test asserts none of those words can appear here,
/// because this is exactly the copy that gets "improved" later by someone who
/// does not know why it is worded this way.
///
/// ## Two of these are reported and two are refused
///
/// The Tower publishes buckets for `facing_wearer` and `unknown` only.
/// `away_from_wearer` and `profile` are **withheld**, with a reason: a
/// per-person facing state narrows to one person's orientation the moment only
/// one person is in view.
///
/// They are kept in this enum rather than deleted precisely because
/// `facing_states_withheld` is data on the wire — the app has to be able to
/// name what is withheld in order to say it is withheld. `isReportedByTower`
/// is what stops a screen from rendering a bucket that does not exist.
enum SceneFacing: Equatable, Sendable, CaseIterable {
    /// The Tower reported no orientation for these.
    case unknown
    /// Oriented toward the camera.
    case towardCamera
    /// Oriented away from the camera. **Withheld** — no bucket is published.
    case awayFromCamera
    /// Oriented across the camera's view. **Withheld** — no bucket is
    /// published.
    case acrossView

    var displayName: String {
        switch self {
        case .unknown: return "Orientation unknown"
        case .towardCamera: return "Facing your direction"
        case .awayFromCamera: return "Facing away"
        case .acrossView: return "Facing across"
        }
    }

    /// The Tower's own name for this bucket, as it appears in
    /// `people.facing_states_reported` / `facing_states_withheld`.
    var wireName: String {
        switch self {
        case .unknown: return "unknown"
        case .towardCamera: return "facing_wearer"
        case .awayFromCamera: return "away_from_wearer"
        case .acrossView: return "profile"
        }
    }

    /// Whether this cartridge publishes a count for this state at all.
    ///
    /// Two do and two do not, and a screen that showed the other two as `0`
    /// would be reporting a measurement that was never taken.
    var isReportedByTower: Bool {
        switch self {
        case .unknown, .towardCamera: return true
        case .awayFromCamera, .acrossView: return false
        }
    }

    /// The caveat any surface showing orientation owes the reader.
    ///
    /// A single shared constant rather than per-case prose, so it cannot be
    /// dropped from one case and kept in another.
    static let gazeCaveat =
        "Body orientation only. The glasses have no eye tracking and cannot tell where anyone is looking."
}

// MARK: - Disclosure

/// One measured limit on the counts, as the Tower states it.
///
/// The slug is for a client that wants to key off a class of limit —
/// `size-floor`, `recall`, `field-of-view`, `noise-classes`, `departure-lag` —
/// and `detail` is the sentence a person can be shown. Both are carried; the
/// slug alone is not renderable and the detail alone is not matchable.
struct SceneCountLimitation: Equatable, Identifiable, Sendable {
    let slug: String
    let detail: String

    var id: String { slug }
}

/// When the limitations were measured, and on what.
///
/// `isCurrent` is `false` on this Tower, deliberately: the corpus grows
/// continuously, and a rate asserted in the present tense would read as current
/// state.
struct SceneCountMeasurement: Equatable, Sendable {
    let measuredAt: String?
    let corpusFrames: Int?
    let corpusCaptures: Int?
    let isCurrent: Bool
    let note: String?
}

/// The refusals, carried as values rather than as silence.
///
/// A refusal delivered as an absence cannot be told apart from an unimplemented
/// field, and those are different instructions: "refused" means build a
/// different screen, "not implemented yet" means wait for the next Tower.
struct SceneRefusals: Equatable, Sendable {
    /// Why there is no per-entity list.
    let tracksAbsentReason: String?
    /// `{field, reason}` for `track_id`, `box`, `facing`, `visible_eyes`,
    /// `confidence`.
    let refusedEntityFields: [SceneRefusedField]
    /// Why there is no confidence.
    let confidenceAbsentReason: String?
    /// Why there are no relations.
    let relationsAbsentReason: String?
    /// `left_of`, `right_of`, `higher_in_view` — computable, true, and withheld
    /// because they stop being true the moment the wearer turns their head.
    let withheldRelations: [String]
    /// The stronger list: relations that were measured and refused.
    let refusedRelations: [SceneRefusedRelation]
    /// Labels that never appear in `where`. `["person"]`.
    let whereExcludes: [String]
    let whereExcludesReason: String?

    /// The sentence this app adds on its own behalf, because it is the half of
    /// the reasoning a reader cannot get from the Tower's prose.
    ///
    /// The point is **not** minimising disclosure — the phone sent the pixels,
    /// so a count discloses strictly less than the frame this app already
    /// holds, and withholding it while shipping frames would be theatre. What
    /// is genuinely new is **joinability**.
    static let joinabilityNote = """
        This cartridge publishes no handle for anyone in view — no track id, no \
        box, no per-person row — and there is no field on the wire that could \
        carry one. That is not about keeping the count small: the phone sent \
        the frames, so a count tells you less than the picture already on this \
        device. It is about what a handle plus a timestamp would let someone \
        assemble later, which is the per-person timeline this cartridge keeps \
        none of.
        """
}

struct SceneRefusedField: Equatable, Identifiable, Sendable {
    let field: String
    let reason: String

    var id: String { field }
}

struct SceneRefusedRelation: Equatable, Identifiable, Sendable {
    let relation: String
    let reason: String
    /// Where the measurement behind the refusal lives, in the Tower's
    /// repository. Operator-facing.
    let reasonSource: String?

    var id: String { relation }
}

// MARK: - The scene itself

/// The half of the payload that exists only when `scene_available` is `true`.
///
/// Kept as its own type so that "there is no scene" is the absence of this
/// value rather than a scattering of nils. `counts`, `where` and `people` are
/// null together and non-null together on the wire, and the decoder refuses a
/// payload where they disagree — three independently-nil fields is three ways
/// to render half a scene.
struct SceneObservation: Equatable, Sendable {
    /// One entry per reported class, in the Tower's order, present at `0`.
    let counts: [SceneClassCount]
    /// Side counts per **non-person** label.
    let positions: [SceneLabelPositions]
    let people: ScenePeople
    /// The detector's own score floor. Published in place of a confidence,
    /// which this payload does not have and could not attach to anything.
    let scoreThreshold: Double?

    /// Every count is zero and the Tower was looking.
    ///
    /// **This is the fifth case**, and it is not one of the four silences: it
    /// means "I looked at a frame and saw none of the thirteen things I can
    /// see". It still carries the lower-bound disclosure, because 0.306 recall
    /// on `person` makes "I saw nothing" and "there is nobody there" different
    /// statements.
    var sawNothing: Bool { counts.allSatisfy { $0.count == 0 } }

    /// Classes with at least one confirmed track, in the Tower's order.
    var present: [SceneClassCount] { counts.filter { $0.count > 0 } }
}

/// Why there is no scene. **Four distinct situations**, and a client that
/// flattened them would show an empty room for all four — when only one of them
/// is even about a room, and that one is not in this enum.
enum SceneUnavailableReason: Equatable, Sendable {
    /// Stopped. The last scene was **discarded**, not kept.
    case stopped
    /// The detector is still loading. Not an empty room; a Tower that has not
    /// looked yet.
    case stillLoading
    /// The session failed. Carries the Tower's own reason.
    case failed(String)
    /// Running, and no frame has been observed yet.
    case runningButNoFrameYet
    /// A state this build did not recognise, with whatever the Tower said.
    case unrecognised(String)

    /// The Tower's prose is preferred wherever it exists; this is the headline
    /// above it, and the four differ because they call for four different
    /// things to be said to a person.
    var headline: String {
        switch self {
        case .stopped: return "Not observing"
        case .stillLoading: return "Still loading"
        case .failed: return "Scene reading failed"
        case .runningButNoFrameYet: return "No frame yet"
        case .unrecognised: return "Unavailable"
        }
    }
}

/// One complete Scene Understanding payload, decoded.
///
/// Named a *reading* rather than a snapshot: a snapshot is of something that
/// persists, and `persistence: "none"` is the first thing this cartridge says
/// about itself.
struct SceneReading: Equatable, Sendable {
    // The constant self-description. Carried rather than assumed, and asserted
    // against the constants in `SceneUnderstandingContract` at decode time —
    // a Tower that changed one of these while keeping the identifier would be
    // a contract violation this app should notice rather than absorb.
    let claim: String
    let identity: String
    let absenceMeans: String
    let persistence: String
    let frameOfReference: String
    let timeBasis: String

    let lifecycle: SceneLifecycle

    /// When the Tower **received the frame** this reading came from, in unix
    /// seconds.
    ///
    /// Not when the glasses captured it, and not when the detector finished
    /// with it. There is no capture clock anywhere on this wire, which is why
    /// this is not folded into `ObservationTime.observedAt` — that field is
    /// documented as "when the glasses observed it", and mapping by field name
    /// would make exactly the substitution Core Principle 5 forbids.
    let observedAtTowerReceipt: Double?
    /// The Tower's own sentence saying the above. Rendered, not paraphrased.
    let observedAtNote: String?
    /// Now minus `observedAtTowerReceipt`, as the Tower computed it.
    let stalenessSeconds: Double?

    let framesOffered: Int
    let framesObserved: Int
    /// Frames displaced from the single slot because the worker was busy.
    ///
    /// On the wire deliberately: a silently dropped frame is indistinguishable
    /// from a quiet room. A sustained non-zero value also stretches what the
    /// tracker's `max_misses` means, because that bound is a frame count and
    /// not a duration — which is why `departure-lag` appears in
    /// `count_limitations` when this is advancing.
    let framesSkipped: Int
    let framesDroppedNotRunning: Int
    let decodeFailures: Int

    /// The model that produced the counts, when there is one.
    let detector: String?
    /// The universe of labels that can appear in `counts` — 13 COCO names,
    /// fixed at build time. A label outside it has never been looked for, which
    /// is a weaker silence than "looked for and not seen".
    let reportedClasses: [String]
    /// `"confirmed-tracks"`. Counts come from the tracker, never from raw
    /// detections.
    let countBasis: String

    /// **`true` on every payload**, and an obligation rather than a flag.
    ///
    /// > An undercount published without disclosure looks exactly like a quiet
    /// > room.
    let countIsLowerBound: Bool
    let countLimitations: [SceneCountLimitation]
    let countMeasurement: SceneCountMeasurement

    /// The wearer's own left and right as the camera sees them, with the
    /// thresholds and the unverified assumption stated. `where` is a coarse
    /// signed bearing under another name, so it declares its sign convention
    /// rather than leaving it to be presumed.
    let sideConvention: String?

    let refusals: SceneRefusals

    /// The scene, or the reason there is none. Exactly one of the two.
    let observation: SceneObservation?
    let unavailableReason: SceneUnavailableReason?
    /// The Tower's own prose for `unavailableReason`, shown verbatim where it
    /// exists — it is more specific than any headline this app can write.
    let unavailableReasonText: String?

    /// The disclosure that must appear wherever a count does.
    ///
    /// Two sentences, and both are load-bearing. The first is Core Principle 3:
    /// the camera not seeing someone is not evidence that nobody is there. The
    /// second is the measurement: recall against an oracle over 14,128 real
    /// frames is 0.306 for `person`, and because the oracle shares COCO
    /// training data with the shipped model, **0.306 is an upper bound**.
    static let countCaveat = """
        Every count here is a floor, not a total. It counts only what this \
        detector confirmed in the camera's forward cone, and it misses most of \
        what is there — measured recall is about 0.31 for people and near zero \
        for anything smaller than 2% of the frame. Most of a room is behind the \
        wearer.
        """

    /// The one-line version, for a place a paragraph will not fit.
    static let countCaveatShort = "A floor, not a total. Counts miss most of what is there."
}

// MARK: - State

/// What the Scene Understanding workspace should be showing.
enum SceneUnderstandingState: Equatable, Sendable {
    /// This Tower cannot serve Scene Understanding, in the Tower's own words
    /// where it gave any.
    case unsupported(reason: String)
    /// Nothing is being observed. Reached from `stopped`, and from a `paused`
    /// session that has no scene to show.
    ///
    /// **A stop lands here and the reading is dropped.** See `lastKnown`.
    case idle(SceneReading?)
    /// Observing, and no scene yet: still loading, or running and waiting for
    /// its first frame. Two different sentences, both honest as "waiting".
    case awaitingFirstScene(SceneReading)
    /// A live scene. The only state in which the counts describe now.
    case observing(SceneReading)
    /// The scene as it was when the session was **paused**.
    ///
    /// ## The one rule this case exists to carry, and the one way it goes wrong
    ///
    /// Pause keeps the scene, marks `lifecycle.scene_is_current` false, and
    /// lets its age accumulate. That is genuinely useful and must be visually
    /// distinct from observing — which is what this case is for.
    ///
    /// **Stop is the opposite and must never reach here.** A scene held past
    /// the end of a session is a claim about a room the wearer has left, and no
    /// staleness number makes that safe: a client that renders counts above a
    /// staleness line shows the room first. The Tower already discards on stop
    /// — `scene_available` goes false and `counts`/`where`/`people` go null —
    /// and this app must not reinstate what the Tower threw away by holding its
    /// own copy.
    ///
    /// The gate is `SceneLifecycleState.mayHoldLastKnownScene`, checked in
    /// `SceneUnderstandingState.forReading(_:)`, which is the only constructor
    /// any decode path uses.
    case lastKnown(SceneReading)
    case failed(CartridgeFailure)

    /// The reading behind whatever is on screen, if there is one.
    var reading: SceneReading? {
        switch self {
        case .idle(let reading): return reading
        case .awaitingFirstScene(let reading), .observing(let reading), .lastKnown(let reading):
            return reading
        case .unsupported, .failed: return nil
        }
    }

    /// The scene, when one is being shown. `nil` in every state that has none —
    /// including `.idle`, which may still carry the reading that says *why*.
    var observation: SceneObservation? {
        switch self {
        case .observing(let reading), .lastKnown(let reading): return reading.observation
        case .idle, .awaitingFirstScene, .unsupported, .failed: return nil
        }
    }

    /// Whether what is on screen describes now. `false` for `.lastKnown`, which
    /// is the whole reason that case is separate.
    var isCurrent: Bool {
        if case .observing = self { return true }
        return false
    }

    /// The five silences, told apart.
    ///
    /// Four of them are `scene_available: false` with different reasons; the
    /// fifth is `scene_available: true` with every count at zero, which is the
    /// only one that is about a room. A client that rendered all five the same
    /// way would show "nothing here" to a person whose Tower had not started,
    /// had not finished loading, had crashed, had not yet seen a frame, or had
    /// genuinely looked and seen nothing.
    var silence: SceneSilence? {
        switch self {
        case .observing(let reading), .lastKnown(let reading):
            guard let observation = reading.observation else { return nil }
            return observation.sawNothing ? .lookedAndSawNothing : nil
        // Written twice rather than combined, because the two cases carry
        // different types: `.idle` may hold nothing at all (a stop, a
        // disconnect) and `.awaitingFirstScene` always holds the reading that
        // says what it is waiting for.
        case .idle(let reading):
            // A `stopped` session carrying a `failure_reason` did not stop
            // because anybody asked, and saying "the last scene was discarded"
            // tells an operator a comforting story about a dead engine. The
            // bench Tower reports exactly this: `state: "stopped"` with *"the
            // engine could not be loaded: ModuleNotFoundError: No module named
            // 'torch'"*.
            //
            // Corrected in the **sentence**, not in the state. Routing it to
            // `.failed` was tried and is wrong: the Tower keeps
            // `failure_reason` across a stop, so a session that failed, was
            // restarted successfully and then stopped normally would report a
            // failure it had recovered from — and it would put a
            // `.failed` state where `.idle` belongs, weakening the
            // stop-discards invariant to fix a copy problem.
            // `stopped` ONLY. An `unrecognised` state must still yield `nil` so
            // it reaches the screen as the Tower's own prose rather than being
            // assigned one of the known headlines — that is the whole reason
            // `unrecognised` exists, and a failure reason riding along does not
            // license this build to name a state it does not know.
            if reading?.lifecycle.state == .stopped,
               let why = reading?.lifecycle.failureReason, !why.isEmpty {
                return .towerFailed(why)
            }
            return Self.silence(for: reading?.unavailableReason)
        case .awaitingFirstScene(let reading):
            return Self.silence(for: reading.unavailableReason)
        // A Tower-reported failure IS one of the five silences, and reaching it
        // needs its own line because a failed lifecycle is routed to `.failed`
        // rather than to `.idle` — so the `unavailableReason` path above never
        // sees it, and this returned `nil` for one of the five cases the type
        // exists to tell apart.
        //
        // `failure.message` is the Tower's own `lifecycle.failure_reason`
        // (see `forReading`), not prose invented here, which is what makes it
        // safe to put in `.towerFailed`. The fallback there is generic on
        // purpose; a silence carrying it says "failed" without inventing a
        // cause.
        case .failed(let failure):
            return .towerFailed(failure.message)
        // `.unsupported` is deliberately NOT a silence. A silence is this
        // cartridge having nothing to say right now; an unsupported Tower is
        // one that will never say anything, which is a different sentence and a
        // different instruction to a person.
        case .unsupported: return nil
        }
    }

    var phase: CartridgePhase {
        switch self {
        case .unsupported: return .unsupported
        case .idle: return .idle
        case .awaitingFirstScene: return .waiting
        case .observing: return .live
        // Settled rather than live: a paused scene is not being refined, and a
        // spinner over it would claim work that is not happening.
        case .lastKnown: return .settled
        case .failed: return .failed
        }
    }

    /// The four unavailable reasons, as silences.
    ///
    /// `.unrecognised` deliberately yields `nil`: a reason this build has never
    /// heard of must reach the screen as the Tower's own prose rather than
    /// being assigned one of the four headlines, which would state the wrong
    /// cause with full confidence.
    private static func silence(for reason: SceneUnavailableReason?) -> SceneSilence? {
        switch reason {
        case .stopped: return .stopped
        case .stillLoading: return .stillLoading
        case .failed(let why): return .towerFailed(why)
        case .runningButNoFrameYet: return .runningButNoFrameYet
        case .unrecognised, .none: return nil
        }
    }

    /// The **only** way a decode path turns a reading into a state.
    ///
    /// Centralised precisely so the stop-discards rule is written once. A
    /// `switch` at each call site is a rule that holds until somebody adds a
    /// call site.
    static func forReading(_ reading: SceneReading) -> SceneUnderstandingState {
        switch reading.lifecycle.state {
        case .failed:
            return .failed(
                CartridgeFailure(
                    kind: .towerReportedFailure,
                    message: reading.lifecycle.failureReason
                        ?? reading.unavailableReasonText
                        ?? "The Tower reported that its Scene Understanding session failed."
                )
            )

        case .starting:
            // Loading is genuinely in flight, which is the one phase in which a
            // spinner is honest. `load_overdue` does not change that: an
            // overdue load is still a load.
            return .awaitingFirstScene(reading)

        case .running:
            guard let _ = reading.observation else {
                // Running with no frame observed yet. Distinct from `stopped`
                // and distinct from an empty room, and the reading carries the
                // reason that says which.
                return .awaitingFirstScene(reading)
            }
            return .observing(reading)

        case .paused:
            // The one state that may hold a scene that is not current. Guarded
            // by the lifecycle rather than by this switch alone, so that the
            // rule survives someone rearranging the cases.
            guard reading.lifecycle.state.mayHoldLastKnownScene,
                  reading.observation != nil
            else { return .idle(reading) }
            return .lastKnown(reading)

        case .stopped, .unrecognised:
            // **Stop discards.** The reading is kept only for the sentence it
            // carries about why there is nothing; its `observation` is `nil` on
            // the wire in this state, and nothing here reinstates one.
            return .idle(reading)
        }
    }
}

/// The five ways this cartridge can have nothing to show, which are five
/// different things to say to a person.
enum SceneSilence: Equatable, Sendable {
    /// Nobody has started a session, or one was stopped. The last scene was
    /// discarded on purpose.
    case stopped
    /// The detector is still loading. Not an empty room.
    case stillLoading
    /// The session failed, with the Tower's reason.
    case towerFailed(String)
    /// Running, and the first frame has not come back yet.
    case runningButNoFrameYet
    /// **The only one that is about a room.** A frame was observed and none of
    /// the thirteen reported classes was confirmed in it.
    case lookedAndSawNothing

    /// What to put above the explanation. Five headlines, because collapsing
    /// any two of them tells a person the wrong thing to do next.
    var headline: String {
        switch self {
        case .stopped: return "Not observing"
        case .stillLoading: return "Still loading"
        case .towerFailed: return "Scene reading failed"
        case .runningButNoFrameYet: return "No frame yet"
        case .lookedAndSawNothing: return "Nothing in view"
        }
    }

    /// The sentence beneath it. The four unavailable cases defer to the Tower's
    /// own prose where the caller has it; this is the fallback and the wording
    /// for the fifth, which the Tower does not write a sentence for because it
    /// is not an error.
    var explanation: String {
        switch self {
        case .stopped:
            return """
                No session is running, so nothing is being observed. The last \
                scene was discarded rather than kept — a scene held past the \
                end of a session is a claim about a room you have already left.
                """
        case .stillLoading:
            return """
                The detector is loading. This is not an empty room; it is a \
                Tower that has not looked yet.
                """
        case .towerFailed(let why):
            return why
        case .runningButNoFrameYet:
            return """
                The session is running and has not finished observing a frame \
                yet. No frame has been offered, or the first is still in flight.
                """
        case .lookedAndSawNothing:
            return """
                A frame was read and none of the things this cartridge can \
                recognise was confirmed in it. That is about the camera's \
                forward cone at that instant, not about the room — and the \
                count is a floor, so "none confirmed" is not "none there".
                """
        }
    }
}
