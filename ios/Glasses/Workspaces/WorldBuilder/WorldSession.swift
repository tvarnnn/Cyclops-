//
//  WorldSession.swift
//  Glasses
//

import Foundation

// MARK: - What the Tower says about the session behind a world

/// The `session` block of a `world_builder.status` payload, decoded.
///
/// ## Why this block is read at all
///
/// `docs/contracts/WORLD-BUILDER-IOS.md` §2.4 records that iOS consumes
/// `model_state` and `world_snapshot` and none of the Tower-native evidence
/// beside them. That is still the rule for the *figures*. This block is the one
/// exception, and it earns it by answering a question no figure can:
/// **whose world is this?**
///
/// A `WorldSnapshot` describes a world directory on the Tower's disk. It says
/// nothing about which capture produced it, so a snapshot of a world finished
/// an hour ago is indistinguishable — field for field — from a snapshot of the
/// one the wearer is walking through right now. On 2026-08-24 the phone showed
/// camera LIVE beside "Capture has ended." and a frozen figure count, and both
/// halves were telling the truth about different machines.
///
/// Nothing here is a figure and nothing here is drawn as one. It is identity
/// and lifetime, which is exactly what the gate needs.
struct WorldSessionReport: Equatable, Sendable {
    /// Opaque, and this app compares it for equality only.
    var sessionID: String?
    /// The capture directory this session is reading, if any. `nil` for a
    /// synthetic session, which has no capture behind it at all.
    var captureID: String?
    /// The Tower's clock, or `nil` while the session is live. **`nil` is the
    /// live signal**, and the reason this block is worth decoding.
    var endedAt: TimeInterval?
    /// `"stop"`, `"disconnect"` or `"bounded_limit"`, or `nil`. Carried
    /// verbatim; this app does not branch on it.
    var endReason: String?
    /// `"live-capture"`, `"recorded-capture"`, `"synthetic"` or `"unknown"`.
    var frameSource: String?

    init(
        sessionID: String? = nil,
        captureID: String? = nil,
        endedAt: TimeInterval? = nil,
        endReason: String? = nil,
        frameSource: String? = nil
    ) {
        self.sessionID = sessionID
        self.captureID = captureID
        self.endedAt = endedAt
        self.endReason = endReason
        self.frameSource = frameSource
    }

    /// Whether the Tower has closed this session. Absent means live — the
    /// contract is explicit that `ended_at` is null while a session runs.
    var hasEnded: Bool { endedAt != nil }

    /// Whether this session is being fed by a capture as it is written.
    ///
    /// The string is the Tower's, set by `world_build_session.py` when it is
    /// launched with `--follow-capture` — which is the only way a session comes
    /// to exist because a phone opened a stream. Matched exactly rather than
    /// prefixed or contained: `"recorded-capture"` is a directory of frames
    /// somebody replayed, and reading it as live is the whole class of mistake
    /// this type exists to stop.
    var isLiveCapture: Bool { frameSource == Self.liveCapture }

    static let liveCapture = "live-capture"
}

// MARK: - Whose world the phone is looking at

/// What the phone has established about the relationship between the snapshot
/// on screen and the capture it currently has open.
///
/// Four cases rather than a `Bool`, because "we have not established it" and
/// "we have established that it is not ours" call for different words on
/// screen, and neither is "no capture is open".
enum WorldSessionBinding: Equatable, Sendable {
    /// No capture bracket is open on this phone, so there is nothing to bind
    /// to and the Tower's own state is the whole answer. This is also the
    /// permanent value in a Release build, which has no capture control.
    case none
    /// A bracket is open and the Tower has resolved no session for it yet —
    /// the window between `stream_start` and the builder creating its world.
    case awaiting(captureID: String?)
    /// The snapshot describes the capture this phone has open.
    case bound(captureID: String)
    /// The snapshot describes some other capture. **This is the 2026-08-24
    /// bug's signature**, and the case that must never be drawn as a result.
    case foreign(captureID: String?)

    /// The capture the Tower's snapshot named, when it named one.
    var captureID: String? {
        switch self {
        case .none: return nil
        case .awaiting(let id), .foreign(let id): return id
        case .bound(let id): return id
        }
    }

    /// Whether the phone has a capture open at all.
    var isBracketOpen: Bool { self != .none }

    /// Whether the snapshot on the wire has been established as belonging to
    /// some capture other than this phone's.
    var isForeign: Bool {
        if case .foreign = self { return true }
        return false
    }
}

// MARK: - The gate

/// The one rule that keeps another capture's world off this session's screen.
///
/// > **No `WorldModelState` carrying a snapshot may be rendered as this
/// > session's result unless iOS can establish that the snapshot belongs to the
/// > capture iOS currently has open.**
///
/// ## What the phone can and cannot establish today
///
/// The Tower mints a capture id in its web process at the moment the phone
/// sends `stream_start`, and it does not tell the phone what that id is:
/// `stream_start` gets no reply, and the `stream_started` message that would
/// carry one is deliberately unimplemented on the Tower
/// (`tower/docs/agent-handoffs/IOS-WORLD-BUILDER-INTEGRATION.md` §5). So the
/// phone cannot compare capture ids, and nothing here pretends it can.
///
/// What it can establish, from the payload alone, is **liveness**. The Tower
/// attaches a builder to every capture at the moment the id is minted, one per
/// capture lineage; the result channel prefers a world whose writer lock is
/// held by a running process; and a session fed by that builder records
/// `frame_source: "live-capture"`, its capture directory's name, and a null
/// `ended_at` for exactly as long as it runs. A snapshot with all three, while
/// this phone has a bracket open, is this session's. A snapshot with any of
/// them missing describes a capture that is over, replayed, or synthetic —
/// none of which this phone opened.
///
/// That is strictly weaker than an id comparison and strictly stronger than
/// what shipped before, which was nothing. When `stream_started` lands, the
/// equality check drops into `binding(isCaptureBracketOpen:session:modelState:)`
/// and nothing else moves.
enum WorldSessionGate {

    /// What the phone has established, given its own bracket and the payload.
    static func binding(
        isCaptureBracketOpen: Bool,
        session: WorldSessionReport?,
        modelState: WorldModelState
    ) -> WorldSessionBinding {
        // No bracket, nothing to bind. Deliberately not `.foreign`: with no
        // capture of its own the phone has no grounds to call another world
        // foreign, and after Stop the world the Tower is finishing is the one
        // the wearer just walked.
        guard isCaptureBracketOpen else { return .none }

        // A bracket with no session behind it is the honest `.awaiting`: the
        // Tower has resolved no session at all, which is what it reports in the
        // seconds between `stream_start` and the builder creating its world.
        guard let session else { return .awaiting(captureID: nil) }

        guard
            case .receiving = modelState,
            !session.hasEnded,
            session.isLiveCapture,
            // `.bound` names a capture, so a live session with no capture
            // directory behind it cannot be one — a phone's stream always has
            // one, because the id *is* the directory.
            let captureID = session.captureID
        else {
            return .foreign(captureID: session.captureID)
        }
        return .bound(captureID: captureID)
    }

    /// The state to render, once the binding has had its say.
    ///
    /// A foreign snapshot renders as **waiting**, never as a result. That one
    /// substitution is the correctness fix; everything else passes through.
    ///
    /// `.unsupported` and `.failed` pass through under every binding. They are
    /// reports about the *Tower* rather than about a world — "this Tower cannot
    /// serve World Builder", "the builder died" — and the phone has no way to
    /// establish whose builder died. Swallowing either into a spinner would
    /// hide a real fault behind an animation, which is the failure mode this
    /// whole gate exists to remove rather than relocate.
    static func presented(
        _ state: WorldModelState,
        binding: WorldSessionBinding
    ) -> WorldModelState {
        switch state {
        case .unsupported, .failed:
            return state
        // `.interrupted` is a world state like `.finalized` and is gated like
        // one: an interrupted walk from another capture is not this session's
        // result either.
        case .idle, .awaitingFirstUpdate, .receiving, .finalizing, .finalized, .interrupted:
            switch binding {
            case .none, .bound:
                return state
            case .awaiting, .foreign:
                return .awaitingFirstUpdate
            }
        }
    }
}
