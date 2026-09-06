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
    static let identifier = "world_builder.worlds/2026-09-06"
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
    /// `nil` while the session is still open. **`nil` is the live signal**, as
    /// on the status channel; it is never read as zero.
    let endedAt: Double?
    let endReason: String?
    let frameSource: String
    let captureID: String?
    let hasGeometry: Bool

    /// The Tower has not closed this session.
    var isStillOpen: Bool { endedAt == nil }
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

    /// What the picker calls it.
    var title: String { displayName ?? worldID }
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
