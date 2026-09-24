//
//  WorldSelection.swift
//  Glasses
//

import Foundation

// MARK: - Why this world is on the wire

/// The Tower's word for how it chose the world a status payload describes.
///
/// A `RawRepresentable` struct rather than an `enum`, for the reason
/// `CartridgeSessionState` is one: a word this build has never heard of must
/// arrive **as itself** rather than fail the decode or, worse, default to one
/// of the five it knows. The client branches on the words it understands and
/// treats any other exactly as it treats a Tower that sent no word at all.
nonisolated struct WorldSelectionMode: RawRepresentable, Equatable, Sendable, Hashable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    /// The client named this world. History, by construction.
    static let pinned = WorldSelectionMode(rawValue: "pinned")
    /// A live builder holds this world's writer lock and its session is open.
    static let live = WorldSelectionMode(rawValue: "live")
    /// A live builder holds the lock and the session has stopped: the two
    /// minutes after Stop, when the final solve and the final build run.
    static let finalizing = WorldSelectionMode(rawValue: "finalizing")
    /// **Nothing is live.** The Tower answered an unpinned subscribe with the
    /// most recently updated world on disk. This is the case that put a
    /// stored world on the Live canvas on 2026-09-06, and the client owns
    /// what to do with it.
    static let latest = WorldSelectionMode(rawValue: "latest")
    /// No world could be chosen — none exist, or none could be read.
    static let none = WorldSelectionMode(rawValue: "none")

    /// The payload carried no `selection` block: a Tower older than
    /// `world_builder.status/2026-09-06`. Spelled so that no Tower word can
    /// collide with it.
    static let unknown = WorldSelectionMode(rawValue: "(absent)")

    /// Whether this is a word the client acts on. `.none` is included: it is
    /// understood, and it means the payload has no world.
    var isRecognised: Bool {
        [Self.pinned, .live, .finalizing, .latest, .none].contains(self)
    }
}

/// The `selection` block of a `world_builder.status` payload, decoded.
///
/// Answers the question the audit found nobody could answer from the wire:
/// **did the Tower serve this world because it is live, or because it was
/// the newest thing on disk?** Before this block the two were
/// indistinguishable, and the phone's only defence was `WorldSessionGate`,
/// which by design passes any snapshot through when no capture bracket is
/// open — so with the cartridge idle the newest stored world hydrated into
/// the Live canvas as a finished one.
nonisolated struct WorldSelection: Equatable, Sendable {
    var mode: WorldSelectionMode
    /// The world and session the Tower resolved, in its own words. Carried
    /// beside the snapshot's ids rather than instead of them: the snapshot
    /// is what is drawn, this is why.
    var worldID: String?
    var sessionID: String?
    /// The Tower's prose for the choice. Shown nowhere today; kept because a
    /// log line that says *why* a world was served is what turns the next
    /// physical-session surprise into a one-line diagnosis.
    var reason: String?

    init(
        mode: WorldSelectionMode,
        worldID: String? = nil,
        sessionID: String? = nil,
        reason: String? = nil
    ) {
        self.mode = mode
        self.worldID = worldID
        self.sessionID = sessionID
        self.reason = reason
    }

    /// What an older Tower's payload decodes to. The client keeps today's
    /// behaviour under it — the gate alone — because a Tower that has not
    /// said how it chose has not said the world is stored, either.
    static let unknown = WorldSelection(mode: .unknown)

    /// Whether the Tower said, in a word this build understands, that this
    /// world is being served because it is **current** — being built now, or
    /// being finished now.
    var isCurrentWorld: Bool { mode == .live || mode == .finalizing }

    /// Whether the Tower said nothing is live and this is history offered in
    /// its place.
    var isHistoryOfferedAsLive: Bool { mode == .latest }
}

// MARK: - How finalization went

/// The builder's own account of the work after Stop, from
/// `lifecycle.finalization` (status) or a session row's `finalization`
/// (listing). `nil` on a record written before the builder kept one.
///
/// Both words are `RawRepresentable` structs, for the reason
/// `WorldSelectionMode` is: the Tower owns the vocabulary and a new word must
/// survive to the screen rather than be folded into one of the known ones.
nonisolated struct WorldFinalizationState: RawRepresentable, Equatable, Sendable, Hashable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    static let pending = WorldFinalizationState(rawValue: "pending")
    static let complete = WorldFinalizationState(rawValue: "complete")
    static let interrupted = WorldFinalizationState(rawValue: "interrupted")
}

nonisolated struct WorldFinalizationReport: Equatable, Sendable {
    var state: WorldFinalizationState
    /// `pending`, `solved`, `skipped`, `failed`, `unavailable`, or `nil`.
    /// Verbatim; this app does not branch on it.
    var finalSolve: String?
    /// Tower clock, seconds. Never defaulted to zero — a missing time is a
    /// missing time.
    var startedAt: Double?
    var updatedAt: Double?
    /// The builder's prose — an error text when the state is `interrupted`.
    /// **Not for display** (`WORLD-BUILDER-COMPONENTS.md` §3.1, v6): it can
    /// carry an error string. `notice` is the sentence meant for the owner.
    var detail: String?
    /// `finalization.notice` (v6, `WORLD-BUILDER-COMPONENTS.md` §3.1, §8):
    /// what the evidence gate could not do for this walk, and who can fix it
    /// -- an owner, an operator, the idle Tower, or a new walk. Shown
    /// **verbatim** below the room caption, as a plain note, never matched
    /// on: the wording is the Tower's and may change between Towers. `nil`
    /// when absent -- every older session, and every gated session with
    /// nothing owed -- and for anything that is not a non-empty string.
    var notice: String?

    init(
        state: WorldFinalizationState,
        finalSolve: String? = nil,
        startedAt: Double? = nil,
        updatedAt: Double? = nil,
        detail: String? = nil,
        notice: String? = nil
    ) {
        self.state = state
        self.finalSolve = finalSolve
        self.startedAt = startedAt
        self.updatedAt = updatedAt
        self.detail = detail
        self.notice = notice
    }

    /// `nil` for `null`, for an absent key, and for a block with no `state`
    /// — a record that says nothing about finalization has no report, and a
    /// report with a made-up state would be a claim.
    init?(json: Any?) {
        guard let json = json as? [String: Any], let word = json["state"] as? String else {
            return nil
        }
        self.state = WorldFinalizationState(rawValue: word)
        self.finalSolve = json["final_solve"] as? String
        self.startedAt = json["started_at"] as? Double
        self.updatedAt = json["updated_at"] as? Double
        self.detail = json["detail"] as? String
        self.notice = Self.notice(json["notice"])
    }

    /// The notice as sent, or `nil`: absent, `null`, not a string, or nothing
    /// but whitespace. The text itself is kept exactly as the Tower wrote it.
    nonisolated static func notice(_ value: Any?) -> String? {
        guard let text = value as? String,
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        return text
    }
}

// MARK: - The last saved world, offered rather than drawn

/// A stored world the Tower offered in place of a live one, kept as a
/// reference the person can follow rather than as a state the canvas draws.
///
/// Built from a `latest` selection while following live. Nothing here is a
/// figure: it names the world, says what state it would have shown in, and
/// gives the Open action enough to pin it. The rows and the gallery arrive
/// only after that pin, under the "Saved world" heading, which is where
/// history belongs.
nonisolated struct WorldRecentReference: Equatable, Sendable {
    let worldID: String
    let sessionID: String?
    /// The Tower's display name, or `nil`. Never derived.
    let name: String?
    /// The `model_state` word the payload carried — `finalized`,
    /// `interrupted`, `finalizing`, … — kept as the Tower's word so the label
    /// below cannot claim more than the payload did.
    let modelState: String
    /// `world.updated_at`, Tower clock, when the payload carried it.
    let updatedAt: Double?

    init(
        worldID: String,
        sessionID: String? = nil,
        name: String? = nil,
        modelState: String,
        updatedAt: Double? = nil
    ) {
        self.worldID = worldID
        self.sessionID = sessionID
        self.name = name
        self.modelState = modelState
        self.updatedAt = updatedAt
    }

    /// What to call it on one line. The name when the Tower gave one; a dated
    /// title from `updatedAt` otherwise, so two unnamed worlds are told apart
    /// by when they were last touched rather than by a 32-character id; the
    /// id itself only when there is neither.
    var title: String {
        if let name, !name.isEmpty { return name }
        if let updatedAt { return WorldListingPresentation.datedTitle(updatedAt: updatedAt) }
        return worldID
    }

    /// The Tower's word, in the picker's vocabulary. Unknown words pass
    /// through verbatim rather than being mapped to the nearest known one.
    var stateLabel: String {
        switch modelState {
        case "receiving": return "building"
        case "finalizing": return "unfinished"
        case "finalized": return "finished"
        case "interrupted": return "interrupted"
        case "failed": return "failed"
        case "idle": return "empty"
        default: return modelState
        }
    }
}
