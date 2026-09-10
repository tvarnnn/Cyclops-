//
//  WorldLibrary.swift
//  Glasses
//

import Foundation

/// The saved-worlds agreement this build implements: `GET /worlds`.
///
/// A third World Builder contract beside status and geometry, on the geometry
/// transport, versioned on its own. Opaque and compared for equality only, as
/// the other two are.
nonisolated enum WorldListingContract {
    static let identifier = "world_builder.worlds/2026-09-10"
}

/// One session of a stored world, as the Tower lists it.
///
/// The same facts `WorldSessionReport` decodes off the status channel, plus
/// `has_geometry`, which is what a picker needs to say whether opening a
/// session will show anything. Kept as its own type rather than reusing that
/// one because this row has required fields and that one has none — the list
/// route promises them, the status block does not.
nonisolated struct WorldListingSession: Equatable, Sendable {
    let sessionID: String
    /// The Tower's clock, seconds.
    let startedAt: Double
    /// `nil` while the record is open. **Not** the live signal on its own any
    /// more: a builder killed mid-walk leaves `ended_at: null` behind forever,
    /// and 29 such records on the real root read as "still open". `abandoned`
    /// and `state` say which kind of open this is.
    let endedAt: Double?
    let endReason: String?
    let frameSource: String
    let captureID: String?
    let hasGeometry: Bool

    // Additive since 2026-09-06, all optional: an older Tower omits them and
    // the row is drawn from what it did send.

    /// The record's own count — written at start (zero) and rewritten at
    /// stop, so a session that never stopped keeps the zero.
    let keyframesAccepted: Int?
    /// The journal's count: one line per keyframe, whatever the record says.
    /// 467 on the walk whose record said 0.
    let keyframesJournaled: Int?
    /// `ended_at` is null **and** nobody is writing. The Tower's judgment,
    /// carried rather than recomputed.
    let abandoned: Bool?
    /// One word from the status channel's vocabulary: `receiving`,
    /// `finalizing`, `complete`, `interrupted`, `unbuilt`.
    let state: WorldListingSessionState?
    /// The builder's account of finalization, or `nil` on an older record.
    let finalization: WorldFinalizationReport?

    /// The record is open and the Tower did **not** call it abandoned. On a
    /// Tower that does not send `abandoned`, the record being open is all
    /// that is known and is what this reports.
    var isStillOpen: Bool { endedAt == nil && abandoned != true }

    /// The keyframe count worth showing: the record's, unless it is the
    /// start-of-session zero (or absent) and the journal knows better.
    var keyframeCount: Int? {
        if let accepted = keyframesAccepted, accepted > 0 { return accepted }
        if let journaled = keyframesJournaled { return journaled }
        return keyframesAccepted
    }
}

/// A session's state word, as the Tower lists it. A `RawRepresentable` struct
/// so a sixth word survives as itself; see `WorldSelectionMode`.
nonisolated struct WorldListingSessionState: RawRepresentable, Equatable, Sendable, Hashable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    static let receiving = WorldListingSessionState(rawValue: "receiving")
    static let finalizing = WorldListingSessionState(rawValue: "finalizing")
    static let complete = WorldListingSessionState(rawValue: "complete")
    static let interrupted = WorldListingSessionState(rawValue: "interrupted")
    static let unbuilt = WorldListingSessionState(rawValue: "unbuilt")
}

// In an extension so the memberwise initialiser survives.
nonisolated extension WorldListingSession {
    init?(json: [String: Any]) {
        guard
            let sessionID = json["session_id"] as? String,
            let startedAt = json["started_at"] as? Double,
            let frameSource = json["frame_source"] as? String,
            let hasGeometry = json["has_geometry"] as? Bool
        else { return nil }
        self.sessionID = sessionID
        self.startedAt = startedAt
        self.endedAt = json["ended_at"] as? Double
        self.endReason = json["end_reason"] as? String
        self.frameSource = frameSource
        self.captureID = json["capture_id"] as? String
        self.hasGeometry = hasGeometry
        self.keyframesAccepted = json["keyframes_accepted"] as? Int
        self.keyframesJournaled = json["keyframes_journaled"] as? Int
        self.abandoned = json["abandoned"] as? Bool
        self.state = (json["state"] as? String).map(WorldListingSessionState.init(rawValue:))
        self.finalization = WorldFinalizationReport(json: json["finalization"])
    }
}

/// One stored world.
nonisolated struct WorldListingEntry: Equatable, Sendable {
    let worldID: String
    /// `nil` when the Tower has no name for it. Not the empty string, and not
    /// the id: the picker falls back to the id itself, visibly.
    let displayName: String?
    let createdAt: Double
    let updatedAt: Double
    /// Whether a builder currently holds this world's writer lock.
    let live: Bool
    let sessions: [WorldListingSession]

    /// What the picker calls it: the Tower's name, or a dated title from
    /// `updatedAt` when it has none. Never the bare id — 156 of the 162
    /// worlds on the real root are unnamed, and a column of 32-character
    /// hex strings distinguishes nothing. The id stays available as
    /// `worldID` for a monospaced caption.
    var title: String { WorldListingPresentation.title(for: self) }

    /// A world with no sessions has nothing to open: the Tower answers a pin
    /// on it with "the world exists and has no sessions" and the picture
    /// 404s. 96 of 162 on the real root are such shells.
    var hasSessions: Bool { !sessions.isEmpty }
}

// In an extension so the memberwise initialiser survives.
nonisolated extension WorldListingEntry {
    /// `nil` when the row disagrees with the contract, including when any one
    /// of its sessions does — the caller drops the whole listing rather than
    /// offering a world whose sessions it half-read.
    init?(json: [String: Any]) {
        guard
            let worldID = json["world_id"] as? String,
            let createdAt = json["created_at"] as? Double,
            let updatedAt = json["updated_at"] as? Double,
            let live = json["live"] as? Bool,
            let rawSessions = json["sessions"] as? [[String: Any]]
        else { return nil }
        var sessions: [WorldListingSession] = []
        for raw in rawSessions {
            guard let session = WorldListingSession(json: raw) else { return nil }
            sessions.append(session)
        }
        self.worldID = worldID
        self.displayName = json["display_name"] as? String
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.live = live
        self.sessions = sessions
    }
}

/// Every world the Tower's world root holds, newest first — the Tower's
/// order, kept.
nonisolated struct WorldListing: Equatable, Sendable {
    let worlds: [WorldListingEntry]
}

nonisolated enum WorldListingDecoder {
    /// `nil` unless the payload names this contract exactly and every row
    /// decodes. A row that will not decode drops the whole listing rather
    /// than silently shortening it.
    static func listing(from json: [String: Any]) -> WorldListing? {
        guard
            json["contract"] as? String == WorldListingContract.identifier,
            let rawWorlds = json["worlds"] as? [[String: Any]]
        else { return nil }
        var worlds: [WorldListingEntry] = []
        for raw in rawWorlds {
            guard let world = WorldListingEntry(json: raw) else { return nil }
            worlds.append(world)
        }
        return WorldListing(worlds: worlds)
    }
}

nonisolated enum WorldListFetchError: Error, Equatable {
    /// The Tower answered 404, which on this route means no world root is
    /// configured.
    case notFound
    case undecodable
    case transport(String)
}

/// Fetches the saved-worlds list over HTTP. Modelled on `WorldGeometryClient`
/// and, like it, deliberately not on the WebSocket.
nonisolated struct WorldListClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession = .shared
    /// Bounded (Rule 15), and the same bound Object Memory puts on its own
    /// requests. A list is a small body; ten seconds is a stalled link, not a
    /// slow one.
    var timeout: TimeInterval = 10

    func worlds() async throws -> WorldListing {
        let url = baseURL.appendingPathComponent("worlds")
        let json = try await get(url)
        guard let listing = WorldListingDecoder.listing(from: json) else {
            throw WorldListFetchError.undecodable
        }
        return listing
    }

    private func get(_ url: URL) async throws -> [String: Any] {
        // `.reloadIgnoringLocalCacheData` for the reason `WorldGeometryClient`
        // gives: the Tower sends no validators, and a list answered out of a
        // URL cache would omit the world that was just walked.
        let request = URLRequest(
            url: url,
            cachePolicy: .reloadIgnoringLocalCacheData,
            timeoutInterval: timeout
        )
        do {
            let (data, response) = try await session.data(for: request)
            if let http = response as? HTTPURLResponse, http.statusCode == 404 {
                throw WorldListFetchError.notFound
            }
            guard
                let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { throw WorldListFetchError.undecodable }
            return json
        } catch let error as WorldListFetchError {
            throw error
        } catch {
            throw WorldListFetchError.transport(error.localizedDescription)
        }
    }
}
