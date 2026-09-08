//
//  CartridgeAvailability.swift
//  Glasses
//

import Foundation

// MARK: - Contract identity

/// The Tower's declaration that it can serve a particular cartridge, and which
/// revision of that agreement it is offering.
///
/// ## What this deliberately does not decide
///
/// The Tower's contracts do not exist yet. Every field below is therefore
/// **opaque to this app**: the identifier is a token the Tower chooses and iOS
/// only ever compares for equality, never parses, orders, or ranges over.
///
/// That opacity is the point. A `version: Int` with a `>=` comparison would
/// have baked in three assumptions the Tower has not agreed to — that contracts
/// are versioned by integer, that versions are totally ordered, and that a
/// newer Tower contract is backward compatible with an older client. All three
/// are plausible and none is ours to decide. Equality against a set of
/// identifiers this build was written against assumes only that the Tower can
/// name the thing it is offering, which any contract scheme must do.
///
/// **There is no wire format here.** No message name, no route, no key. This
/// says only "if and when the Tower tells us which contract it speaks, this is
/// the shape of the answer we will hold onto".
struct CartridgeContract: Equatable, Hashable, Sendable {
    /// Which cartridge the contract is for, using this app's catalog id.
    let cartridgeID: String
    /// An opaque token naming the agreement. Compared for equality only.
    let identifier: String

    init(cartridgeID: String, identifier: String) {
        self.cartridgeID = cartridgeID
        self.identifier = identifier
    }
}

// MARK: - Availability

/// Why a cartridge's Tower backing is, or is not, usable right now.
///
/// Five cartridges need this and they need exactly the same four answers, which
/// is what justifies it being shared rather than restated per cartridge. Four of
/// them reach it through `resolve(declared:supported:isTowerReachable:)` below,
/// so the ordering of the checks — which failure wins when several apply — is
/// decided once instead of five times. Object Memory is the fifth and projects
/// its own learned state onto these cases directly, because its contract is not
/// declared in advance and `.unprobed` is a state none of the four can express;
/// it follows the same precedence deliberately, and says so where it does.
enum CartridgeAvailability: Equatable, Sendable {
    /// The Tower has not declared a contract for this cartridge at all.
    ///
    /// This was once the answer for every cartridge. It no longer is: a live
    /// Tower declares `world_builder.status/2026-08-25` over `GET /cartridges`,
    /// which reaches `declared` as a `TowerCartridgeDeclaration` offer, so World
    /// Builder resolves past this case.
    ///
    /// **The roadmap framing this comment used to carry is now wrong and has
    /// been removed rather than softened.** It said this was "a statement about
    /// the Tower's roadmap rather than about this connection", because the
    /// module container was V0.8 and the first module V0.9, and because the
    /// Tower listed Experimental CV Lab, Document Memory and Scene
    /// Understanding under `not_offered`. As of 2026-08-27 the Tower declares
    /// four contracts and `not_offered` is `[]`.
    ///
    /// So this is now a statement about **this connection and this build**, and
    /// nothing else: either the Tower declared no contract for the cartridge,
    /// or this app has no name for it. Both are answerable facts about two
    /// machines, not a position on a roadmap — which is the better meaning,
    /// because a person reading "not built yet" can at least be told truthfully
    /// which of the two it is.
    ///
    /// Kept distinct from `unsupportedContract` for the reason the Tower's
    /// contract §2 gives: a cartridge absent from every list means *"this Tower
    /// has never heard of it"*, while one present with an unreadable contract
    /// means *"update the app"*. Those are opposite instructions to a person.
    case noContract
    /// The Tower declared a contract this build does not implement. Kept
    /// separate from `noContract` because they call for opposite responses:
    /// one waits for the Tower, the other waits for an app update, and telling
    /// a user to reconnect when they need a new build wastes their time.
    case unsupportedContract(declared: CartridgeContract)
    /// The contract is supported, but the Tower is not reachable, so nothing
    /// can be asked of it. Not an error — a connection state.
    case towerUnreachable
    /// The Tower speaks a contract this build implements and is reachable.
    ///
    /// This case used to be documented as unreachable by construction, on the
    /// grounds that `TowerCapabilities` declared no contracts and this build
    /// supported none. Both halves of that are now false:
    /// `TowerCapabilities.supported` contains
    /// `WorldBuilderResultContract.identifier`, and a live Tower offers exactly
    /// that identifier for `world_builder`. **World Builder genuinely resolves
    /// here against a connected Tower.**
    ///
    /// It is still unreachable for every other cartridge, for the reason
    /// `noContract` gives — which is a fact about what those cartridges' Towers
    /// offer, not a property of this type.
    case available(CartridgeContract)

    /// Whether a request may be made at all.
    var isAvailable: Bool {
        if case .available = self { return true }
        return false
    }

    /// One sentence for a person, written so a view never has to compose an
    /// explanation of its own — that is how "not yet built" turns into
    /// "something went wrong" on screen.
    ///
    /// `nil` when available: there is nothing to explain, and a caller that
    /// forgets to handle the available case gets no string rather than a
    /// reassuring one.
    func explanation(cartridgeName: String) -> String? {
        switch self {
        case .noContract:
            // Deliberately says only what is observable from here. It used to
            // assert that the Tower "has no module runtime at all", which was
            // true of every cartridge until one of them gained a contract and
            // is now a claim this app cannot make on the Tower's behalf — the
            // Tower declares each cartridge separately, and silence about one
            // says nothing about the others.
            return """
                This Tower has not declared a \(cartridgeName) contract, so \
                there is nothing to ask it for. That is a statement about what \
                the Tower offers, not about this connection.
                """
        case .unsupportedContract(let contract):
            return """
                The Tower offers a \(cartridgeName) contract this version of \
                the app does not understand (\(contract.identifier)). Nothing \
                is shown rather than something guessed. Updating the app is \
                what resolves this.
                """
        case .towerUnreachable:
            return "The Tower is not connected, so \(cartridgeName) has nothing to report."
        case .available:
            return nil
        }
    }

    /// The phase a cartridge is in *because of* its availability, when
    /// availability alone settles the question.
    ///
    /// `nil` means availability does not settle it — the cartridge is usable
    /// and its own state decides. Keeping this here rather than in each
    /// cartridge is what stops one of them from rendering an unreachable Tower
    /// as `idle`, which would invite a user to press a button that cannot work.
    var forcedPhase: CartridgePhase? {
        switch self {
        case .noContract: return .unsupported
        // Not `.unsupported`. The Tower is already serving this cartridge; the
        // app is the half that is behind, and that is the one case on this
        // screen a person can actually fix.
        case .unsupportedContract: return .needsUpdate
        // Not `.unsupported`. A Tower that is merely unreachable may well be
        // able to do this, and drawing it with the same headline and glyph as
        // "will never do this" is the half of the distinction a person actually
        // reads — the explanation string was carrying it alone.
        case .towerUnreachable: return .disconnected
        case .available: return nil
        }
    }

    /// The unavailable explanation, joined with whatever the cartridge's own
    /// client had to add.
    ///
    /// Extracted because every workspace was restating the same join —
    /// which of the two sentences comes first, and what happens when either is
    /// absent. The per-cartridge part (which of its own states yields a reason)
    /// stays in the cartridge, where it belongs; the ordering is a shared
    /// invariant and belongs here.
    ///
    /// Joined rather than concatenated: a stray blank paragraph in a panel
    /// reads as a missing string, which is how a reader starts to distrust the
    /// rest of it.
    func explanation(cartridgeName: String, clientReason: String?) -> String {
        let shared = explanation(cartridgeName: cartridgeName)
        let client = (clientReason?.isEmpty ?? true) ? nil : clientReason
        switch (shared, client) {
        case (nil, nil): return ""
        case (let shared?, nil): return shared
        case (nil, let client?): return client
        case (let shared?, let client?): return shared + "\n\n" + client
        }
    }

    // MARK: Resolution

    /// Decides availability from the three facts that determine it.
    ///
    /// Pure, so the decision is testable without a socket, and written once so
    /// the precedence is uniform: **what the Tower can do outranks whether we
    /// can currently reach it.** A cartridge the Tower will never serve should
    /// say so even while connected, and an unsupported contract is not fixed by
    /// reconnecting — reporting `towerUnreachable` first would send a user
    /// round a loop that cannot terminate.
    ///
    /// - Parameters:
    ///   - declared: what the Tower says it offers for this cartridge, or `nil`
    ///     if it has said nothing. Non-`nil` today for exactly one cartridge:
    ///     World Builder, whose offer arrives over the wire in the Tower's
    ///     `GET /cartridges` declaration and reaches here via
    ///     `TowerCapabilities.declaredContract(for:in:)`. `nil` for every other
    ///     cartridge, because the local `TowerCapabilities.declared` table is
    ///     empty and the Tower declares nothing for them.
    ///   - supported: the contract identifiers this build implements — the
    ///     five in `TowerCapabilities.supported`, one each for World Builder,
    ///     the CV Lab and Scene Understanding, and two for Document Memory.
    ///     Four of the five are subscription contracts;
    ///     `document_memory.library` is HTTP, so calling the set
    ///     "the subscription contracts" would contradict the comment on
    ///     `supported` itself, which excludes `world_builder.geometry` for
    ///     being HTTP-only. Not empty, and every one matches what a live
    ///     Tower offers, which is why `.available` is a reachable outcome
    ///     rather than a theoretical one.
    ///
    ///     Deliberately not spelled out as literals here. This comment used to
    ///     say "one element today" and quote `world_builder.status/2026-08-25`,
    ///     and both halves went stale — the set grew to five, and that
    ///     identifier was superseded by `/2026-09-06`. The quoted string was
    ///     doing real damage rather than merely being wrong:
    ///     `ios/scripts/contract-drift-check.py` decides what this build
    ///     implements by sweeping *quoted* identifiers out of Swift source, so
    ///     a dead identifier in a doc comment was reported as implemented —
    ///     the one thing that check exists to catch. `TowerCapabilities.supported`
    ///     is the list; naming it is enough.
    ///   - isTowerReachable: whether the connection is currently up.
    /// - Parameter knownToThisBuild: whether this build has a client and a
    ///   Tower-name mapping for the cartridge — i.e. whether a declaration for
    ///   it could ever arrive.
    ///
    ///   This is the discriminator between the two honest answers when nothing
    ///   has been declared. For Visual Q&A there is **no Tower code anywhere**
    ///   and no mapping, so "the Tower is unreachable" would be its own false
    ///   story — `.noContract` is right whether or not a socket is up. For
    ///   Scene Understanding, which has both, a silent socket means only that
    ///   nobody has asked yet.
    ///
    ///   Defaults to `false` so that a caller which has not thought about it
    ///   gets the old, more conservative answer rather than a claim about a
    ///   connection.
    static func resolve(
        declared: CartridgeContract?,
        supported: Set<String>,
        isTowerReachable: Bool,
        knownToThisBuild: Bool = false
    ) -> CartridgeAvailability {
        guard let declared else {
            // Only a cartridge this build could actually receive a declaration
            // for may report `.towerUnreachable` here. See `knownToThisBuild`.
            guard knownToThisBuild else { return .noContract }
            // ## Reachability is checked FIRST when nothing was declared
            //
            // This used to `return .noContract` unconditionally, and on a cold
            // launch with no Tower that was a **false statement rendered to a
            // person** — the explanation for `.noContract` reads *"That is a
            // statement about what the Tower offers, not about this
            // connection"*, which is exactly wrong in the case that produced
            // it.
            //
            // Four of the five cartridges learn `declared` **only** from the
            // socket declaration. Before any successful connection there is no
            // declaration to have, so `.towerUnreachable` was unreachable on a
            // first run and every one of them said the Tower had never heard of
            // it. Object Memory was the only one that got this right, because
            // it probes over HTTP instead of waiting on the declaration.
            //
            // The contract names this failure and prescribes the cure: `GET
            // /cartridges` exists over HTTP as well as the socket precisely
            // because *"a client that can only learn the contract set by
            // opening the socket cannot tell 'unreachable' from 'not built'"*.
            // This app does not issue that request yet, so it cannot always
            // tell them apart — but when the socket is **down** it knows which
            // one it is looking at, and saying so is free.
            //
            // The reasoning already existed one layer up: `TowerClient` keeps a
            // declaration across a teardown, arguing that clearing it *"would
            // turn every dropped connection into `.noContract` — 'this will
            // never work' — when the truthful reading is `.towerUnreachable`,
            // and those two call for opposite responses."* That fixed the
            // drop-after-success case. This fixes never-connected.
            return isTowerReachable ? .noContract : .towerUnreachable
        }
        guard supported.contains(declared.identifier) else {
            return .unsupportedContract(declared: declared)
        }
        return isTowerReachable ? .available(declared) : .towerUnreachable
    }
}

// MARK: - Failure

/// A cartridge-level failure, stated so a view can show it without inventing
/// wording and a test can assert what it is without matching prose.
///
/// Shared because every cartridge fails the same ways, and because Rule 3
/// (Truthful State Only, docs/02-DEVELOPMENT-RULES.md) makes "the backend
/// failed" a state the UI must be able to reach — a cartridge that can only
/// render success and emptiness will render a failure as emptiness.
struct CartridgeFailure: Equatable, Sendable, Error {
    enum Kind: String, Equatable, Sendable, CaseIterable {
        /// This app refused locally, because the Tower has no contract for the
        /// thing being asked for.
        ///
        /// Deliberately **not** `towerReportedFailure`. The Tower reported
        /// nothing — there may not even be a socket open — and attributing a
        /// local refusal to the other machine is a fabricated claim about it
        /// (Rule 3), which is exactly what a later log or telemetry consumer
        /// would read back as fact.
        case notSupported
        /// The Tower reported that the module itself failed
        /// (docs/04-MODULE-SYSTEM.md — Failure).
        case towerReportedFailure
        /// The connection dropped or the request never completed.
        case transport
        /// The Tower's answer could not be understood as the contract this
        /// build implements. Distinct from `unsupportedContract` availability:
        /// that is a disagreement known in advance, this is one discovered on
        /// arrival.
        case undecodableResponse
        /// A bounded operation ran out of time (Rule 15, Bounded Operations).
        case timedOut
    }

    let kind: Kind
    /// What to show a person. Required, and non-empty by construction below —
    /// an error with no explanation is how "something went wrong" ends up on
    /// screen attached to nothing.
    let message: String

    init(kind: Kind, message: String) {
        self.kind = kind
        // An empty message would render as a blank failure panel. Substituting
        // the kind is worse than prose but strictly better than nothing, and it
        // keeps the type's guarantee absolute rather than by convention.
        self.message = message.isEmpty ? "The \(kind.rawValue) step failed." : message
    }

    /// Wraps an arbitrary error, passing a `CartridgeFailure` through unchanged.
    ///
    /// The two clients that accept a request were each restating this ladder;
    /// deciding once means a future third cannot decide differently. An error
    /// that is not already a cartridge failure is `.transport` — it came from
    /// somewhere below this layer, and that is the only honest attribution
    /// available without knowing where.
    static func wrapping(_ error: Error) -> CartridgeFailure {
        if let failure = error as? CartridgeFailure { return failure }
        return CartridgeFailure(kind: .transport, message: error.localizedDescription)
    }
}
