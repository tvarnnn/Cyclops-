//
//  WorldGeometryClient.swift
//  Glasses
//

import Foundation

/// Cached segment geometry, keyed by `(content_hash, placement_hash)`.
///
/// Keyed by hash rather than by segment index on purpose: a re-solved segment
/// keeps its index and changes its contents, so an index key would serve stale
/// geometry under a fresh revision. A hash key cannot.
///
/// ## Why the key is a pair, and what keying on content alone cost
///
/// Because the Tower freezes a segment when tracking is lost, a closed
/// segment's **geometry** never changes again — so its points and poses cross
/// the wire once and are kept for the life of the world, and only the open
/// segment churns.
///
/// **Its geometry. Not its identity, and not where it sits.** Registration is a
/// layer *above* frozen segments: geometry immutable, placements mutable. A
/// later pass places a segment an earlier one could not, and loop closure moves
/// placements rather than points — so `content_hash`, which covers poses and
/// points only, deliberately does **not** move when a segment gains a
/// placement. That is what keeps every cached chunk valid across a registration
/// pass, and it is safe only because `placement_hash` moves instead.
///
/// Keyed on `content_hash` alone this cache therefore never refetches a
/// re-placed segment. It keeps the chunk it has and the client draws an
/// **unplaced** version of a segment the world now knows how to place. Nothing
/// throws, nothing logs, no tile goes blank: the fragment simply sits in the
/// wrong place, permanently. `geometry_revision` rolls up both hashes so the
/// change signal does fire — the entire risk was in this dictionary.
///
/// The key is built by `WorldGeometryCacheKey` and nowhere else, so the two
/// halves cannot drift apart at a call site.
actor WorldGeometryStore {
    private var chunks: [String: WorldSegmentChunk] = [:]

    /// Filed under the chunk's **own** key, never under the key it was asked
    /// for. See the call site in `WorldBuilderViewModel`.
    func insert(_ chunk: WorldSegmentChunk) {
        chunks[chunk.cacheKey] = chunk
    }

    /// Takes a composite key, not a content hash — and is named so that a
    /// caller reaching for `summary.contentHash` out of habit does not
    /// typecheck into a silent cache hit on a stale placement.
    func chunk(forKey key: String) -> WorldSegmentChunk? {
        chunks[key]
    }

    func keysMissing(from wanted: [String]) -> [String] {
        wanted.filter { chunks[$0] == nil }
    }

    /// Drop everything not named by the current manifest. Called after a
    /// manifest arrives so a long walk does not accumulate superseded
    /// segments forever.
    ///
    /// Composite keys, so a segment whose placement moved has its old chunk
    /// dropped here as well as missed by the lookup above — one registration
    /// pass over 51 segments would otherwise leave 51 orphans behind.
    func retainOnly(_ wanted: Set<String>) {
        chunks = chunks.filter { wanted.contains($0.key) }
    }
}

nonisolated enum WorldGeometryFetchError: Error, Equatable {
    case notFound
    case undecodable
    case transport(String)
}

/// Fetches geometry over HTTP. Deliberately not on the WebSocket.
nonisolated struct WorldGeometryClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession = .shared
    /// Bounded (Rule 15). This was the one HTTP client in the app with no
    /// bound of its own — the list, session, health and Object Memory clients
    /// all carry 10 s and the render client 30 s — so a request against a
    /// Tower that accepted the connection and then hung sat on `URLRequest`'s
    /// 60 s default. The segment loop in `WorldBuilderViewModel` is
    /// sequential, so that was a minute *per segment*.
    ///
    /// Thirty seconds, the render client's bound, rather than the ten the
    /// small bodies get: a segment is up to a megabyte of points, and the
    /// 2026-09-06 walk was fetched over a Tailscale path, where ten seconds
    /// is a slow link rather than a stalled one.
    var timeout: TimeInterval = 30

    func manifest(worldID: String, sessionID: String) async throws -> WorldGeometryManifest {
        let url = baseURL
            .appendingPathComponent("worlds/\(worldID)/geometry/manifest")
        let json = try await get(url, query: [URLQueryItem(name: "session_id", value: sessionID)])
        guard let manifest = WorldGeometryDecoder.manifest(from: json) else {
            throw WorldGeometryFetchError.undecodable
        }
        return manifest
    }

    func segment(
        worldID: String, sessionID: String, index: Int, maxPoints: Int? = nil
    ) async throws -> WorldSegmentChunk {
        let url = baseURL
            .appendingPathComponent("worlds/\(worldID)/geometry/segment/\(index)")
        var query = [URLQueryItem(name: "session_id", value: sessionID)]
        if let maxPoints {
            query.append(URLQueryItem(name: "max_points", value: String(maxPoints)))
        }
        let json = try await get(url, query: query)
        guard let chunk = WorldGeometryDecoder.chunk(from: json) else {
            throw WorldGeometryFetchError.undecodable
        }
        return chunk
    }

    private func get(_ url: URL, query: [URLQueryItem]) async throws -> [String: Any] {
        // Not force-unwrapped. Both of these are built from a wire-supplied
        // `world_id`/`session_id`, so "this cannot fail" is a claim about the
        // Tower's output, not about this code.
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            throw WorldGeometryFetchError.transport("the geometry address could not be parsed")
        }
        components.queryItems = query
        guard let requestURL = components.url else {
            throw WorldGeometryFetchError.transport("the geometry address could not be built")
        }
        // ## Why this does not use `session.data(from:)`
        //
        // That builds a request with `.useProtocolCachePolicy` against
        // `URLSession.shared`'s on-disk `URLCache`, and this was the **only**
        // HTTP client in the app that did not opt out — TowerClient, all three
        // Object Memory clients and the Document library all set a reload
        // policy explicitly.
        //
        // It matters more here than anywhere else, because of what this route's
        // URL is: `/worlds/{id}/geometry/segment/{index}`. **A segment gaining
        // a placement does not change that URL.** The whole point of keying the
        // chunk store on `(content_hash, placement_hash)` is that a re-placed
        // segment must be refetched — and a URL-keyed cache sitting in front of
        // that store can answer the refetch with the unplaced body, defeating
        // the tuple key one layer below where it is written.
        //
        // The Tower sends no validators at all on this route — no
        // `Cache-Control`, no `ETag`, no `Last-Modified` (checked against a
        // running Tower) — so freshness would fall to CFNetwork's heuristic.
        // Correctness here is not something to leave to a heuristic in the one
        // subsystem whose entire thesis is that the caching must be exact.
        //
        // The test harness already sets `.ephemeral` with `urlCache = nil` and
        // this same policy, which means the layer being disabled in test was
        // left on in production — the configuration under which the deliberate
        // placement regression test passes was not the shipped one.
        let request = URLRequest(
            url: requestURL,
            cachePolicy: .reloadIgnoringLocalCacheData,
            timeoutInterval: timeout
        )
        do {
            let (data, response) = try await session.data(for: request)
            if let http = response as? HTTPURLResponse, http.statusCode == 404 {
                throw WorldGeometryFetchError.notFound
            }
            guard
                let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { throw WorldGeometryFetchError.undecodable }
            return json
        } catch let error as WorldGeometryFetchError {
            throw error
        } catch {
            throw WorldGeometryFetchError.transport(error.localizedDescription)
        }
    }
}
