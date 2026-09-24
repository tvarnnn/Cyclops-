//
//  WorldPhotographic.swift
//  Glasses
//

import Foundation

// MARK: - Whether the room the wearer walked actually exists

/// One word from `photographic.state`, the Tower's settled answer to *does this
/// world still owe a photographic room* (`WORLD-BUILDER-IOS.md` §3a,
/// `WORLD-BUILDER-WORLDS.md` §2a).
///
/// A `RawRepresentable` struct for the reason `WorldSelectionMode` gives: the
/// Tower owns the vocabulary, and a word this build has not heard of must
/// survive as itself rather than be folded into the nearest known one.
nonisolated struct WorldPhotographicState: RawRepresentable, Equatable, Sendable, Hashable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    /// The appearance stage finished; the photographic room exists.
    static let complete = WorldPhotographicState(rawValue: "complete")
    /// A stage is running under a live process.
    static let running = WorldPhotographicState(rawValue: "running")
    /// Unfinished, and nothing is working on it. The Tower finishes owed work
    /// when it is next idle -- if its photographic stages are switched on.
    static let owed = WorldPhotographicState(rawValue: "owed")
    /// A stage ran and failed. Terminal: nothing more is coming.
    static let failed = WorldPhotographicState(rawValue: "failed")
    /// The stages declined: no global solve, or the Tower's appearance setting
    /// is off. Nothing is wrong and nothing is coming.
    static let unattempted = WorldPhotographicState(rawValue: "unattempted")
    /// A world from before the photographic stages existed.
    static let neverRecorded = WorldPhotographicState(rawValue: "never_recorded")
    /// The Tower's probe could not answer.
    static let unobservable = WorldPhotographicState(rawValue: "unobservable")
}

/// The `photographic` block: `{state, stage, detail}`.
///
/// ## Where it is read, and why from one type
///
/// The Tower computes one block with one helper and sends it in two places:
/// `lifecycle.photographic` on the status channel
/// (`WorldBuilderResultDecoder.photographic(from:)`) and `photographic` on
/// every `GET /worlds` session row (`WorldListingSession.photographic`). The
/// components contract (`WORLD-BUILDER-COMPONENTS.md` §2.1) adds a third: one
/// block per component, "the seven-word vocabulary unchanged". All of them
/// decode through `init?(json:)` below, so the phone reads one fact one way.
///
/// ## What is deliberately NOT here yet
///
/// Contract v2 adds `scope: "room" | "area"` (additive; absent means `room`,
/// COMPONENTS §3.4). When it is implemented it is decoded here, and
/// `standing` is the one place that decides what an area's word means for the
/// room -- no reader below switches on `state` directly, so none of them has
/// to change when it lands.
nonisolated struct WorldPhotographicReport: Equatable, Sendable {
    var state: WorldPhotographicState
    /// `"surface"`, `"appearance"`, or `nil`. Verbatim; the stage the word is
    /// about. Not branched on.
    var stage: String?
    /// The Tower's prose. "Safe to show, names no path" (WORLDS §2a).
    var detail: String?

    init(state: WorldPhotographicState, stage: String? = nil, detail: String? = nil) {
        self.state = state
        self.stage = stage
        self.detail = detail
    }

    /// `nil` for `null`, an absent key, and a block with no `state` word.
    ///
    /// A status payload built from the record alone carries `null` here
    /// (CARTRIDGE-RESULTS `lifecycle.photographic`), and every Tower before
    /// 2026-09-22 carries nothing at all. Neither is a word, and inventing one
    /// -- `complete` in particular -- would be the exact claim the block exists
    /// to stop the phone making on no evidence.
    init?(json: Any?) {
        guard
            let json = json as? [String: Any],
            let word = json["state"] as? String, !word.isEmpty
        else { return nil }
        self.state = WorldPhotographicState(rawValue: word)
        self.stage = (json["stage"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        self.detail = (json["detail"] as? String).flatMap { $0.isEmpty ? nil : $0 }
    }

    /// What the phone makes of the word. Every reader goes through this.
    var standing: WorldPhotographicStanding {
        switch state {
        case .running: return .building
        case .owed: return .owed
        case .unobservable: return .unobservable
        case .failed: return .failed(detail: detail)
        default:
            // `complete`, `unattempted`, `never_recorded`, and any word this
            // build has not heard of. The Tower has already moved
            // `lifecycle.state` for the words that mean "not finished" (§3a),
            // so an unknown word adds nothing the phone can honestly say, and
            // saying nothing keeps an older world exactly as it was.
            return .settled
        }
    }
}

/// What a `WorldPhotographicReport` means for what the phone says.
///
/// Separate from the word because two words can mean the same thing on screen
/// (`complete`, `unattempted` and `never_recorded` all mean "nothing to add"),
/// and because contract v2's `scope` will make one word mean two things (an
/// area still building is not the room still building).
nonisolated enum WorldPhotographicStanding: Equatable, Sendable {
    /// A stage is building the photographic room now (`running`).
    case building
    /// Unfinished, and nothing is building it (`owed`).
    case owed
    /// The Tower could not tell whether it is being built (`unobservable`).
    case unobservable
    /// A stage ran and failed; the world is saved without its photographic
    /// room, and the Tower's reason if it gave one.
    case failed(detail: String?)
    /// Nothing to add: finished, never attempted, from before the stages
    /// existed, or a word this build does not know.
    case settled

    /// The three words the Tower reports as "not finished being made"
    /// (`photographic.is_unsettled`). A world in one of them is `finalizing`
    /// on the wire, whatever `build_in_progress` says this millisecond.
    var isUnfinished: Bool {
        switch self {
        case .building, .owed, .unobservable: return true
        case .failed, .settled: return false
        }
    }

    var isFailed: Bool {
        if case .failed = self { return true }
        return false
    }
}

// MARK: - The words

/// Every sentence the phone says about the photographic room, in one place, so
/// the canvas, the 3D viewer's note, the picker row and the picker's note
/// cannot word one fact four ways.
nonisolated enum WorldPhotographicCopy {

    /// The headline for a saved world whose photographic build failed
    /// (`WORLD-BUILDER-IOS.md` §3a: "the honest middle"). Replaces the stage
    /// word "Saved", which alone is the T3 defect.
    static let failedHeadline = "Saved — the photographic version could not be built"

    /// Under that headline: what the wearer can still do, and that waiting will
    /// not help. `failed` is terminal -- the finisher retries interrupted
    /// stages, never failed ones.
    static let failedExplanation = "This world is saved and opens as the best reconstruction "
        + "the Tower made. Its photographic version will not arrive."

    /// The Tower's own reason, when it gave one. Its prose, verbatim.
    static func failedReason(_ detail: String?) -> String? {
        guard let detail, !detail.isEmpty else { return nil }
        return "The Tower's reason: \(detail)"
    }

    /// The Saved Worlds row badge for a complete session whose photographic
    /// build failed. Short, because it sits beside the row; the row's caption
    /// and the opened note carry the full sentence.
    static let failedBadge = "Photographic failed"

    /// `owed`: true, and without the spinner's promise. `build_in_progress`
    /// is `false` here, honestly -- nothing is building -- and whether the
    /// Tower will pick it up depends on a setting the phone cannot see
    /// (`TOWER_WORLD_FINISH_PENDING`, and the photographic stages being on).
    static let owedSentence = "This world's photographic version is not finished, and nothing is "
        + "building it right now. The Tower finishes it the next time it is idle, if its "
        + "photographic stages are switched on."

    /// `unobservable`: the Tower declined to call the world finished because it
    /// could not tell. Saying so, and nothing more.
    static let unobservableSentence = "The Tower could not tell whether this world's photographic "
        + "version is still being built, so it is not reporting the world as finished."

    /// `running`, reported without a live `build_in_progress` beside it -- a
    /// poll that landed between a stage's record and its status file.
    static let buildingSentence = "The Tower is building this world's photographic version."

    /// Appended to a viewer note about unfinished photographic work: the
    /// reason opening now is worth a caveat at all.
    static let finishedLooksDifferent = "The finished world is very different from this one."
}
