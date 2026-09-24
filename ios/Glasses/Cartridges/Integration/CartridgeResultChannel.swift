//
//  CartridgeResultChannel.swift
//  Glasses
//

import Foundation

/// The Tower's structured result channel, as it actually exists on the wire.
///
/// Contract: `docs/contracts/CARTRIDGE-RESULTS.md`, envelope
/// `cartridge_results.envelope/2026-08-23`. Every type here is a decoding of a
/// message the Tower genuinely sends today — verified against
/// `tower/tower/results/` and `tower/tower/routes/results_ws.py`, not against a
/// summary of them.
///
/// ## Why this is cartridge-blind
///
/// The envelope is generic and the `payload` is not. Nothing in this file knows
/// what a world is; it carries `payload` through as the dictionary the Tower
/// sent, and the cartridge that owns the contract decodes it. That is the same
/// split the Tower enforces on its own side with
/// `test_the_result_channel_core_is_cartridge_blind`, and it is what lets a
/// second cartridge arrive without touching the transport.
///
/// ## Why `[String: Any]` rather than `Codable`
///
/// `TowerClient` already decodes every inbound message with
/// `JSONSerialization`, and the Tower's error messages are deliberately
/// heterogeneous — `result_error` carries different extra keys per `reason`,
/// `result_unsubscribed` carries two keys and no `envelope_contract`, and
/// `frame_result` omits `mean_intensity` rather than nulling it. A strict
/// `Decodable` over that shape is a decoder that fails on messages the Tower
/// considers well-formed. Reading the keys we need and ignoring the rest is the
/// tolerant half of the robustness rule, and the contract's "any field may be
/// null, and null means absent" makes it the correct half.

// MARK: - Capability declaration

/// One cartridge the Tower declares it can report on.
///
/// `available == false` is **not** the same as absent: the Tower is saying it
/// knows this cartridge and cannot serve it right now, and
/// `unavailableReason` is prose for a person explaining which. A cartridge the
/// Tower says nothing about simply does not appear.
struct TowerCartridgeOffer: Equatable, Sendable {
    /// The Tower's name for the cartridge (`"world_builder"`), which is not the
    /// same string as this app's catalog id (`"world-build"`). The mapping
    /// lives in `TowerCapabilities`, in one place.
    let cartridge: String
    let resultType: String
    /// Opaque. Compared for equality only — never parsed, never ordered.
    let contract: String
    let available: Bool
    /// Prose, when `available` is false. Shown verbatim.
    let unavailableReason: String?
    /// Whether every result is a complete snapshot. `true` today, and present
    /// so a future delta mode cannot be mistaken for this one.
    let snapshotOnly: Bool

    init?(json: [String: Any]) {
        guard
            let cartridge = json["cartridge"] as? String,
            let resultType = json["result_type"] as? String,
            let contract = json["contract"] as? String
        else { return nil }
        self.cartridge = cartridge
        self.resultType = resultType
        self.contract = contract
        self.available = json["available"] as? Bool ?? false
        self.unavailableReason = json["unavailable_reason"] as? String
        self.snapshotOnly = json["snapshot_only"] as? Bool ?? true
    }
}

/// A capability the Tower serves over **HTTP** rather than by subscription.
///
/// ## Why this is a separate type from `TowerCartridgeOffer`
///
/// They answer different questions and cannot be merged without losing one of
/// them. A subscription offer says *"you may `result_subscribe` to this and
/// results will be pushed at you"*; this says *"there is a route, here is where
/// it starts, and here is the Tower's own reason it is not a subscription"*.
/// The giveaway is `resultType`, which a fetched contract does not have and
/// cannot be given a sensible default for — a `TowerCartridgeOffer` with an
/// invented `resultType` would be subscribable-looking data about something
/// that cannot be subscribed to.
///
/// `whyNotASubscription` is carried rather than dropped because it is the
/// Tower explaining a design decision in its own words — for Document Memory,
/// that document text is bulk and the result sender shares its send lock with
/// the frame path. That is worth showing an operator verbatim rather than
/// paraphrasing here, where it would rot.
///
/// **Not every HTTP contract appears here.** `world_builder.geometry/2026-08-25`
/// and `object_memory.observations/2026-08-26` are the same shape and are
/// declared nowhere — declaring them would mean moving their identifiers out of
/// adapter modules into the Tower's `contracts.py`, and its `registry.py` must
/// stay cartridge-blind so it cannot import an adapter. Those two lanes own the
/// move. Until they make it, this app hard-codes those two identifiers, and
/// **their absence from this list is not evidence they are unavailable.**
struct TowerHTTPContractOffer: Equatable, Sendable {
    /// The Tower's name for the cartridge, in the Tower's vocabulary.
    let cartridge: String
    /// Opaque. Compared for equality only — never parsed, never ordered.
    let contract: String
    /// Where the route set begins (`"/documents"`). A prefix, not a full URL:
    /// the host comes from the connection, not from the declaration.
    let entryRoute: String
    let available: Bool
    /// Prose, when `available` is false. Shown verbatim.
    let unavailableReason: String?
    /// The Tower's own reasoning for the transport choice. Operator-facing.
    let whyNotASubscription: String?

    init?(json: [String: Any]) {
        guard
            let cartridge = json["cartridge"] as? String,
            let contract = json["contract"] as? String,
            let entryRoute = json["entry_route"] as? String
        else { return nil }
        self.cartridge = cartridge
        self.contract = contract
        self.entryRoute = entryRoute
        self.available = json["available"] as? Bool ?? false
        self.unavailableReason = json["unavailable_reason"] as? String
        self.whyNotASubscription = json["why_not_a_subscription"] as? String
    }
}

/// The Tower's answer to `{"type":"cartridges"}`.
///
/// `not_offered` is deliberately **not** decoded. The contract states it exists
/// for operators and that presence there must never be read as an offer — and
/// both "listed as not offered" and "absent entirely" mean the same thing to
/// this app: the Tower has declared no contract, so `CartridgeAvailability`
/// resolves `.noContract`. Decoding it would create a second way to ask a
/// question that already has one answer.
///
/// **That reasoning survived the unification, and is now trivially true:**
/// `not_offered` is `[]` on the current Tower, and empty is a *claim* — every
/// cartridge in that build with a wire contract now offers it. A cartridge
/// belongs in `not_offered` only while it can say nothing at all; one that can
/// say "I have observed nothing, and here is precisely why" belongs in
/// `cartridges`, available or not.
struct TowerCartridgeDeclaration: Equatable, Sendable {
    let envelopeContract: String?
    let offers: [TowerCartridgeOffer]
    /// Capabilities served over HTTP. Separate from `offers` because the two
    /// are reached by different code paths and confusing them is how a client
    /// tries to `result_subscribe` to a document library.
    let httpContracts: [TowerHTTPContractOffer]

    init(
        envelopeContract: String?,
        offers: [TowerCartridgeOffer],
        httpContracts: [TowerHTTPContractOffer] = []
    ) {
        self.envelopeContract = envelopeContract
        self.offers = offers
        self.httpContracts = httpContracts
    }

    init(json: [String: Any]) {
        self.envelopeContract = json["envelope_contract"] as? String
        let raw = json["cartridges"] as? [[String: Any]] ?? []
        // An explicit loop rather than `compactMap(TowerCartridgeOffer.init)`:
        // the target defaults to `MainActor` isolation, and a function
        // reference handed to `compactMap` is called from a nonisolated
        // context. Decoding an offer is pure, but staying inside the isolated
        // caller is simpler than declaring that.
        var decoded: [TowerCartridgeOffer] = []
        for entry in raw {
            if let offer = TowerCartridgeOffer(json: entry) { decoded.append(offer) }
        }
        self.offers = decoded

        let rawHTTP = json["http_contracts"] as? [[String: Any]] ?? []
        var decodedHTTP: [TowerHTTPContractOffer] = []
        for entry in rawHTTP {
            if let offer = TowerHTTPContractOffer(json: entry) { decodedHTTP.append(offer) }
        }
        self.httpContracts = decodedHTTP
    }

    func offer(forTowerCartridge name: String) -> TowerCartridgeOffer? {
        offers.first { $0.cartridge == name }
    }

    /// The HTTP contract the Tower declared for this cartridge, if any.
    ///
    /// Deliberately a **separate lookup** from `offer(forTowerCartridge:)`
    /// rather than a fallback inside it. A caller wanting a subscription and a
    /// caller wanting a route are asking different questions, and a single
    /// method answering both would let a subscription path silently succeed on
    /// a fetch-only capability.
    func httpContract(forTowerCartridge name: String) -> TowerHTTPContractOffer? {
        httpContracts.first { $0.cartridge == name }
    }
}

// MARK: - Subscription

/// The Tower's acknowledgement of a `result_subscribe`.
///
/// `subscriptionID` is unique **per connection** and restarts at `sub-1` on
/// every new socket, which is why nothing stores it across a reconnect.
struct CartridgeSubscriptionAck: Equatable, Sendable {
    let subscriptionID: String
    let cartridge: String
    let resultType: String
    let contract: String?
    /// `absent`, `matched`, `stale` or `unrecognised`. Advisory only — the
    /// first result is a complete snapshot regardless.
    let cursorStatus: String?
    /// The pin the Tower echoes from the `result_subscribe` it is answering
    /// (`world_id` / `session_id`, `nil` for an unpinned subscribe). This is
    /// what says WHICH subscribe an ack answers when more than one is out:
    /// a pinned one and the unpinned one it replaced are not interchangeable,
    /// and a count of outstanding acks cannot tell them apart.
    let worldID: String?
    let sessionID: String?

    init?(json: [String: Any]) {
        guard
            let subscriptionID = json["subscription_id"] as? String,
            let cartridge = json["cartridge"] as? String,
            let resultType = json["result_type"] as? String
        else { return nil }
        self.subscriptionID = subscriptionID
        self.cartridge = cartridge
        self.resultType = resultType
        self.contract = json["contract"] as? String
        self.cursorStatus = json["cursor_status"] as? String
        self.worldID = json["world_id"] as? String
        self.sessionID = json["session_id"] as? String
    }
}

// MARK: - Result envelope

/// One `cartridge_result`: a complete snapshot, never a delta.
///
/// The `payload` is passed through undecoded. Everything above it is the
/// generic envelope every cartridge shares.
struct CartridgeResultEnvelope {
    let subscriptionID: String?
    let cartridge: String
    let resultType: String
    let contract: String?
    /// Dense per subscription, starting at 1. A gap is corruption, never a
    /// drop — the channel has no queue to drop from.
    let sequence: Int?
    /// Opaque change identity. Equality only.
    let revision: String?
    /// Whether `revision` differs from the one last sent on this subscription.
    /// `false` on the ~2 s heartbeat that refreshes the fields excluded from
    /// the revision hash.
    let revisionChanged: Bool
    /// How many snapshots were superseded in our slot since the last delivery.
    /// `> 0` means we read slowly; it does **not** mean information was lost,
    /// because every snapshot is complete.
    let coalesced: Int
    /// Always `true` today.
    let isSnapshot: Bool
    /// `tower_sent_at`: when the Tower sent this result, on the Tower's clock
    /// (`CARTRIDGE-RESULTS.md` §4). Compared only with other Tower times --
    /// the look-back prompt's `speak_until` (`WORLD-BUILDER-COMPONENTS.md`
    /// §6.5 rule 4) -- never with the phone's clock. `nil` when absent.
    var towerSentAt: Double? = nil
    let payload: [String: Any]

    init?(json: [String: Any]) {
        guard
            let cartridge = json["cartridge"] as? String,
            let resultType = json["result_type"] as? String,
            let payload = json["payload"] as? [String: Any]
        else { return nil }
        self.subscriptionID = json["subscription_id"] as? String
        self.cartridge = cartridge
        self.resultType = resultType
        self.contract = json["contract"] as? String
        self.sequence = json["seq"] as? Int
        self.revision = json["revision"] as? String
        self.revisionChanged = json["revision_changed"] as? Bool ?? true
        self.coalesced = json["coalesced"] as? Int ?? 0
        self.isSnapshot = json["snapshot"] as? Bool ?? true
        self.towerSentAt = (json["tower_sent_at"] as? NSNumber)?.doubleValue
        self.payload = payload
    }
}

// MARK: - Errors

/// A `result_error`. **None of these close the socket** — the frame path keeps
/// working through every one.
///
/// Every field but `reason` and `message` is optional because the Tower's
/// extras genuinely vary by reason: `unknown_subscription` names only the
/// subscription, `unknown_cartridge` names the cartridge and what is offered,
/// and the two unsolicited errors carry no `envelope_contract` at all.
struct CartridgeResultError: Equatable, Sendable {
    /// One of the ten codes in the contract. Compared as a string rather than
    /// modelled as an enum: an unrecognised code must still be shown to a
    /// person rather than collapsed into "unknown".
    let reason: String
    let message: String
    let subscriptionID: String?
    let cartridge: String?
    let resultType: String?

    init?(json: [String: Any]) {
        guard let reason = json["reason"] as? String else { return nil }
        self.reason = reason
        self.message = json["message"] as? String ?? reason
        self.subscriptionID = json["subscription_id"] as? String
        self.cartridge = json["cartridge"] as? String
        self.resultType = json["result_type"] as? String
    }

    /// The two the Tower sends unsolicited, after which the subscription is
    /// gone and a new `result_subscribe` is the only way to resume.
    var closesSubscription: Bool {
        reason == "consumer_too_slow" || reason == "channel_failed"
    }
}

// MARK: - Events

/// Everything the result channel can deliver, as one stream.
///
/// A single enum rather than five publishers so that ordering between them is
/// preserved: a `result_subscribed` and the `cartridge_result` that
/// immediately follows it must not be observed out of order, and separate
/// subjects would make that a scheduling accident.
enum CartridgeResultEvent {
    case declaration(TowerCartridgeDeclaration)
    case subscribed(CartridgeSubscriptionAck)
    case unsubscribed(subscriptionID: String)
    case result(CartridgeResultEnvelope)
    case failed(CartridgeResultError)
}
