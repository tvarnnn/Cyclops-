//
//  WorldAssetTransport.swift
//  Glasses
//

import Foundation
import WebKit

// MARK: - The private scheme the viewer page is loaded from

/// The one URL scheme the world viewer's `WKWebView` loads from and fetches
/// through, `WORLD-BUILDER-IOS.md` §10 and `WORLD-BUILDER-WORLDS.md` §4.
///
/// ## Why a scheme at all
///
/// Every page the Tower served before the appearance rung carried its data
/// inside it and was loaded with `loadHTMLString(_:baseURL: nil)`: no origin,
/// nothing to fetch, one navigation. The appearance page cannot work that way.
/// Its imagery is ~13 MB of compressed texture blocks, and a live walk lands a
/// new appearance build on every solve, which a page carrying its data could
/// only follow by being replaced (and resetting the wearer's camera). So the
/// page fetches -- and every fetch it makes comes HERE, to a native handler
/// that serves the page string the app already fetched and proxies a short
/// whitelist of this world's Tower routes, and nothing else.
///
/// The Tower's page states the same boundary from its side: its
/// Content-Security-Policy is `connect-src glasses-world:`, so the web content
/// cannot reach the network, the Tower or any other origin directly.
///
/// URLs mirror the Tower's own paths, under a fixed host:
/// `glasses-world://tower/worlds/<world>/render` is the page.
nonisolated enum WorldAssetScheme {
    /// Lower case, one word: WebKit lowercases scheme names, and a scheme WebKit
    /// handles itself (http, https, file, about, data, blob) cannot be
    /// registered at all.
    static let name = "glasses-world"
    static let host = "tower"

    /// The page's address for `worldID`, or `nil` when the id will not go into
    /// a URL. Encoded as ONE path component, like `WorldRenderClient.url`.
    static func pageURL(worldID: String) -> URL? {
        guard !worldID.isEmpty, let world = encoded(worldID) else { return nil }
        var components = URLComponents()
        components.scheme = name
        components.host = host
        components.percentEncodedPath = "/worlds/\(world)/render"
        return components.url
    }

    /// What this app declares it can draw, sent as `viewer=` on the page and
    /// revision requests (`WORLD-BUILDER-WORLDS.md` §4). The Tower's `auto`
    /// ladder offers the appearance page only to a client that declares it,
    /// because a build older than this scheme handler cannot fetch its imagery
    /// and was served a page that drew nothing.
    ///
    /// Sent on EVERY request that decides a rung -- the page (`WorldRenderClient
    /// .url`), the native revision poll (`WorldRenderClient.revisionURL`) and the
    /// page's own revision poll proxied by the scheme handler
    /// (`WorldAssetRequest.towerURL`) -- so all three agree on the rung. One of
    /// them answering without it would report "surface" about an appearance page
    /// on screen, and the follower would swap the page down.
    static let viewerCapability = "appearance-1"
    static let viewerQueryName = "viewer"

    /// RFC 3986 unreserved characters only; everything else, `/` included, is
    /// percent-encoded. The same set `WorldRenderClient` uses.
    static func encoded(_ id: String) -> String? {
        id.addingPercentEncoding(withAllowedCharacters: unreserved)
    }

    private static let unreserved = CharacterSet(
        charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    )
}

// MARK: - What the page may ask for

/// A request the scheme handler will answer, or nothing. Pure, so the whitelist
/// is tested without a `WKWebView` (the pattern `WorldRenderNavigationPolicy`
/// set).
///
/// **Everything not listed here is refused and never reaches the Tower**: any
/// other world, any other session, any other route, any method but GET, a
/// query on a route that takes none, a digest that is not 32 lower-hex
/// characters, a user, port or fragment, a `..` segment.
nonisolated enum WorldAssetRequest: Equatable, Sendable {
    /// `/worlds/<w>/render`: the page string the app already fetched, from memory.
    case page
    /// `/worlds/<w>/appearance/<s>/manifest`
    case appearanceManifest
    /// `/worlds/<w>/appearance/<s>/chunk/<digest>`
    case appearanceChunk(digest: String)
    /// `/worlds/<w>/appearance/<s>/proxy/<digest>`
    case appearanceProxy(digest: String)
    /// `/worlds/<w>/render/revision?session_id=<s>`: how the page follows new
    /// appearance builds without being reloaded. Proxied WITH the app's
    /// `viewer` declaration added (`towerURL`), which the page does not send.
    case renderRevision

    /// `worldID` is the world the viewer was opened for. `sessionID` is the
    /// session the page on screen draws (see `WorldAssetSchemeHandler.session`);
    /// with none known, only the page itself is served.
    static func parse(
        _ url: URL?, method: String?, worldID: String, sessionID: String?
    ) -> WorldAssetRequest? {
        guard
            let url,
            method == nil || method == "GET",
            url.scheme?.lowercased() == WorldAssetScheme.name,
            url.host == WorldAssetScheme.host,
            url.user == nil, url.password == nil, url.port == nil, url.fragment == nil,
            let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
            let world = WorldAssetScheme.encoded(worldID), !worldID.isEmpty
        else { return nil }
        let path = components.percentEncodedPath
        guard path.hasPrefix("/") else { return nil }
        let segments = path.dropFirst().split(separator: "/", omittingEmptySubsequences: false)
            .map(String.init)
        guard !segments.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }) else {
            return nil
        }
        let query = components.percentEncodedQuery
        guard segments.count >= 3, segments[0] == "worlds", segments[1] == world else { return nil }

        if segments.count == 3, segments[2] == "render", query == nil {
            return .page
        }
        guard let sessionID, !sessionID.isEmpty, let session = WorldAssetScheme.encoded(sessionID)
        else { return nil }
        if segments.count == 4, segments[2] == "render", segments[3] == "revision",
           query == "session_id=\(session)" {
            return .renderRevision
        }
        guard query == nil, segments.count >= 5, segments[2] == "appearance", segments[3] == session
        else { return nil }
        if segments.count == 5, segments[4] == "manifest" {
            return .appearanceManifest
        }
        if segments.count == 6, isDigest(segments[5]) {
            switch segments[4] {
            case "chunk": return .appearanceChunk(digest: segments[5])
            case "proxy": return .appearanceProxy(digest: segments[5])
            default: return nil
            }
        }
        return nil
    }

    /// 32 lower-hex characters: `WORLD-BUILDER-APPEARANCE.md` §4.
    static func isDigest(_ value: String) -> Bool {
        value.utf8.count == 32 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    /// The Tower address this request is proxied to, or `nil` for the page,
    /// which is never fetched here.
    func towerURL(baseURL: URL, worldID: String, sessionID: String?) -> URL? {
        guard
            var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false),
            let world = WorldAssetScheme.encoded(worldID)
        else { return nil }
        let base = components.percentEncodedPath.hasSuffix("/")
            ? String(components.percentEncodedPath.dropLast())
            : components.percentEncodedPath
        let session = sessionID.flatMap(WorldAssetScheme.encoded)
        switch self {
        case .page:
            return nil
        case .renderRevision:
            guard let session else { return nil }
            components.percentEncodedPath = "\(base)/worlds/\(world)/render/revision"
            // The page asks without a declaration; the app is the one that can
            // draw the appearance page, so the handler adds it. Without it the
            // Tower would answer the rung an old app gets (surface).
            components.percentEncodedQuery = "session_id=\(session)&"
                + "\(WorldAssetScheme.viewerQueryName)=\(WorldAssetScheme.viewerCapability)"
        case .appearanceManifest:
            guard let session else { return nil }
            components.percentEncodedPath = "\(base)/worlds/\(world)/appearance/\(session)/manifest"
            components.percentEncodedQuery = nil
        case .appearanceChunk(let digest):
            guard let session else { return nil }
            components.percentEncodedPath = "\(base)/worlds/\(world)/appearance/\(session)/chunk/\(digest)"
            components.percentEncodedQuery = nil
        case .appearanceProxy(let digest):
            guard let session else { return nil }
            components.percentEncodedPath = "\(base)/worlds/\(world)/appearance/\(session)/proxy/\(digest)"
            components.percentEncodedQuery = nil
        }
        return components.url
    }

    /// Immutable content named by a digest, which may be reused from memory.
    var digest: String? {
        switch self {
        case .appearanceChunk(let digest), .appearanceProxy(let digest): return digest
        case .page, .appearanceManifest, .renderRevision: return nil
        }
    }
}

// MARK: - Fetching from the Tower

/// What the Tower answered, carried back to the web view.
nonisolated struct WorldAssetResponse: Sendable, Equatable {
    let status: Int
    let mimeType: String
    let data: Data
}

/// Proxies whitelisted requests to the Tower.
///
/// **Compression is URLSession's.** The Tower gzips the appearance bodies when
/// the request accepts it (`WORLD-BUILDER-APPEARANCE.md` §9); `URLSession` sends
/// `Accept-Encoding: gzip, deflate, br` by itself and hands back DECODED bytes.
/// So `data` below is already the plain chunk, and this type must never set
/// `Accept-Encoding` itself (a hand-set value is passed through verbatim and the
/// body is no longer guaranteed to be decoded). The handler then describes the
/// decoded bytes to WebKit (`WorldAssetSchemeHandler.responseHeaders`): no
/// `Content-Encoding`, `Content-Length` of what it actually sends.
///
/// **An ephemeral session with no URL cache**, and every request asks for
/// `.reloadIgnoringLocalAndRemoteCacheData`, for the reason
/// `ObjectMemoryImageryHTTPClient` gives: these are first-person images of a
/// private room (`WORLD-BUILDER-APPEARANCE.md` §3), and honouring the Tower's
/// `no-store` must not depend on which session a caller happened to pass.
/// Nothing is written to disk by this type, ever.
nonisolated struct WorldAssetClient: Sendable {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession
    /// A bundle is ~1.6 MB; bounded all the same (Rule 15).
    var timeout: TimeInterval = 30

    init(baseURL: URL = TowerConfiguration.httpBaseURL, session: URLSession? = nil,
         timeout: TimeInterval = 30) {
        self.baseURL = baseURL
        self.session = session ?? Self.sharedUncachedSession
        self.timeout = timeout
    }

    /// One process-wide session of that configuration, so a viewer opened a
    /// hundred times does not leave a hundred sessions behind. Ephemeral: its
    /// connection pool and nothing else outlives a request.
    static let sharedUncachedSession = uncachedSession()

    /// Ephemeral, `urlCache = nil`, local and remote caches ignored.
    static func uncachedSession() -> URLSession {
        URLSession(configuration: uncachedConfiguration())
    }

    static func uncachedConfiguration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        return configuration
    }

    /// The request for `asset`, or `nil` for the page (never fetched here) or an
    /// address that will not form.
    func request(for asset: WorldAssetRequest, worldID: String, sessionID: String?) -> URLRequest? {
        guard let url = asset.towerURL(baseURL: baseURL, worldID: worldID, sessionID: sessionID)
        else { return nil }
        var request = URLRequest(
            url: url, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData, timeoutInterval: timeout
        )
        request.httpMethod = "GET"
        request.httpShouldHandleCookies = false
        return request
    }

    func fetch(_ asset: WorldAssetRequest, worldID: String, sessionID: String?) async throws
        -> WorldAssetResponse
    {
        guard let request = request(for: asset, worldID: worldID, sessionID: sessionID) else {
            throw WorldRenderFetchError.badAddress
        }
        let (data, response) = try await session.data(for: request)
        let http = response as? HTTPURLResponse
        return WorldAssetResponse(
            status: http?.statusCode ?? 502,
            mimeType: http?.mimeType ?? "application/octet-stream",
            data: data
        )
    }
}

// MARK: - What may be answered from memory

/// The viewer's in-memory copy of content-addressed bundles and the proxy, and
/// the rule for when a copy may be answered WITHOUT asking the Tower.
///
/// Pure and a value, so the rule is tested without a web view or a network.
///
/// ## Why a copy needs a fresh authorisation (review 1, M5)
///
/// The Tower re-checks the session's redaction label on every appearance
/// request (`WORLD-BUILDER-APPEARANCE.md` §9). A memory copy answered on a hit
/// skipped that check entirely: after a relabel or a purge, a restored WebGL
/// context or a reloaded page redrew withdrawn imagery from this copy until the
/// next revision poll happened to drop it, up to two minutes later. So a hit is
/// served only while the Tower has answered this session's MANIFEST with 200
/// within `authorizationWindow` -- the page fetches the manifest before it asks
/// for any bundle, at boot, on a new build and on a restored context, so in
/// practice every page load is authorised by its own manifest request. A hit
/// outside the window is `revalidate`: the handler asks for the manifest first
/// (a few hundred kilobytes, `no-store`) and answers from memory only if that
/// comes back 200. Any manifest or revision answer that does not serve the
/// appearance drops the copy and the authorisation with it.
nonisolated struct WorldAssetMemory: Sendable {
    static let limit = 64 << 20
    /// Long enough for a page to fetch its manifest and then its bundles;
    /// short enough that a later load must ask the Tower again.
    static let authorizationWindow: TimeInterval = 20

    private(set) var entries: [String: WorldAssetResponse] = [:]
    private(set) var bytes = 0
    /// When the Tower last served this session's manifest.
    private(set) var authorizedAt: Date?

    enum Decision: Equatable, Sendable {
        /// Answer from memory, now.
        case serve(WorldAssetResponse)
        /// A copy exists but its authorisation is stale: fetch the manifest
        /// first, then decide again.
        case revalidate
        /// Proxy the request to the Tower.
        case fetch
    }

    func isAuthorized(at now: Date) -> Bool {
        guard let authorizedAt else { return false }
        let age = now.timeIntervalSince(authorizedAt)
        return age >= 0 && age < Self.authorizationWindow
    }

    func decision(for asset: WorldAssetRequest, now: Date) -> Decision {
        guard let digest = asset.digest, let hit = entries[digest] else { return .fetch }
        return isAuthorized(at: now) ? .serve(hit) : .revalidate
    }

    /// Learn from what the Tower answered.
    mutating func record(_ asset: WorldAssetRequest, _ response: WorldAssetResponse, now: Date) {
        switch asset {
        case .appearanceManifest:
            if response.status == 200 {
                authorizedAt = now
            } else {
                drop()
            }
        case .renderRevision:
            if response.status != 200 || !WorldAssetSchemeHandler.revisionServesAppearance(response.data) {
                drop()
            }
        case .appearanceChunk(let digest), .appearanceProxy(let digest):
            guard response.status == 200, entries[digest] == nil,
                  bytes + response.data.count <= Self.limit
            else { return }
            entries[digest] = response
            bytes += response.data.count
        case .page:
            break
        }
    }

    mutating func drop() {
        entries = [:]
        bytes = 0
        authorizedAt = nil
    }
}

// MARK: - The handler

/// Answers the viewer page's `glasses-world:` requests.
///
/// MainActor, by the project's default isolation, which is where WebKit calls
/// `start` and `stop`. A `WKURLSchemeTask` must never be answered after WebKit
/// stopped it -- that raises an Objective-C exception -- so every asynchronous
/// completion checks that its task is still live, and a stopped task's fetch is
/// cancelled.
///
/// **Memory only.** Content-addressed bundles and the proxy are kept in memory
/// for the life of this viewer, capped at `WorldAssetMemory.limit`, so a
/// content-process kill does not download 14 MB again. A copy is answered only
/// under a fresh authorisation from the Tower (`WorldAssetMemory`), and dropped
/// whenever the Tower stops serving this session's appearance, because the
/// Tower re-checks the redaction label on every request and a copy must not
/// outlive that check. Nothing is written to disk.
final class WorldAssetSchemeHandler: NSObject, WKURLSchemeHandler {
    let worldID: String
    private let client: WorldAssetClient

    /// The page string `WorldRenderClient` fetched, served for `.page`.
    var pageHTML: String?
    /// The session the page on screen draws. Only its routes are proxied.
    var sessionID: String?

    static let cacheLimit = WorldAssetMemory.limit
    private(set) var memory = WorldAssetMemory()
    var cacheBytes: Int { memory.bytes }
    /// Injectable so a test can move past the authorisation window.
    var clock: () -> Date = { Date() }

    private var liveTasks: Set<ObjectIdentifier> = []
    private var inflight: [ObjectIdentifier: Task<Void, Never>] = [:]

    init(worldID: String, client: WorldAssetClient = WorldAssetClient()) {
        self.worldID = worldID
        self.client = client
    }

    /// The session a page may reach: the one the viewer was opened for, or --
    /// when the Tower chose -- the one the page's own revision stamp names.
    static func session(for target: WorldRenderTarget, page html: String?) -> String? {
        if let pinned = target.sessionID { return pinned }
        return WorldRenderRepresentation.session(
            of: html.flatMap { WorldRenderRepresentation.meta(named: "wb-revision", in: $0) })
    }

    func dropCache() {
        memory.drop()
    }

    /// What the page gets for `asset` (anything but `.page`): from memory under
    /// a fresh authorisation, after revalidating the manifest when the
    /// authorisation is stale, or from the Tower. Every Tower answer is
    /// recorded, so a manifest or revision that withdraws the appearance drops
    /// the copy before anything else is answered from it.
    func answer(_ asset: WorldAssetRequest, sessionID: String?) async throws -> WorldAssetResponse {
        switch memory.decision(for: asset, now: clock()) {
        case .serve(let hit):
            return hit
        case .revalidate:
            let manifest = try await client.fetch(.appearanceManifest, worldID: worldID, sessionID: sessionID)
            memory.record(.appearanceManifest, manifest, now: clock())
            if case .serve(let hit) = memory.decision(for: asset, now: clock()) {
                return hit
            }
        case .fetch:
            break
        }
        let response = try await client.fetch(asset, worldID: worldID, sessionID: sessionID)
        memory.record(asset, response, now: clock())
        return response
    }

    func webView(_ webView: WKWebView, start urlSchemeTask: any WKURLSchemeTask) {
        let request = urlSchemeTask.request
        guard let asset = WorldAssetRequest.parse(
            request.url, method: request.httpMethod, worldID: worldID, sessionID: sessionID
        ) else {
            respond(urlSchemeTask, status: 404, mimeType: "text/plain", data: Data("not served".utf8))
            return
        }
        if asset == .page {
            respond(urlSchemeTask, status: 200, mimeType: "text/html; charset=utf-8",
                    data: Data((pageHTML ?? "").utf8))
            return
        }
        if case .serve(let hit) = memory.decision(for: asset, now: clock()) {
            respond(urlSchemeTask, status: hit.status, mimeType: hit.mimeType, data: hit.data)
            return
        }
        let id = ObjectIdentifier(urlSchemeTask as AnyObject)
        liveTasks.insert(id)
        let sessionID = sessionID
        inflight[id] = Task { [weak self] in
            guard let self else { return }
            let result: Result<WorldAssetResponse, Error>
            do {
                result = .success(try await self.answer(asset, sessionID: sessionID))
            } catch {
                result = .failure(error)
            }
            // Back on the main actor: the task inherited it.
            guard self.liveTasks.remove(id) != nil else { return }
            self.inflight[id] = nil
            switch result {
            case .success(let response):
                self.respond(urlSchemeTask, status: response.status, mimeType: response.mimeType,
                             data: response.data)
            case .failure(let error):
                urlSchemeTask.didFailWithError(error)
            }
        }
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: any WKURLSchemeTask) {
        let id = ObjectIdentifier(urlSchemeTask as AnyObject)
        liveTasks.remove(id)
        inflight.removeValue(forKey: id)?.cancel()
    }

    /// Whether a revision body still names a served appearance
    /// (`appearance.revision` non-null).
    nonisolated static func revisionServesAppearance(_ data: Data) -> Bool {
        guard
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let appearance = json["appearance"] as? [String: Any],
            let revision = appearance["revision"] as? String
        else { return false }
        return !revision.isEmpty
    }

    /// The headers WebKit is given for a body of `byteCount` bytes. Only these:
    /// in particular never the Tower's `Content-Encoding`, because `URLSession`
    /// already decoded the body -- forwarding `gzip` would make WebKit decode
    /// plain bytes -- and `Content-Length` is the decoded length, not the wire's.
    nonisolated static func responseHeaders(mimeType: String, byteCount: Int) -> [String: String] {
        [
            "Content-Type": mimeType,
            "Content-Length": String(byteCount),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        ]
    }

    private func respond(_ task: any WKURLSchemeTask, status: Int, mimeType: String, data: Data) {
        guard let url = task.request.url else { return }
        let headers = Self.responseHeaders(mimeType: mimeType, byteCount: data.count)
        guard let response = HTTPURLResponse(
            url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers
        ) else { return }
        task.didReceive(response)
        // In pieces: one multi-megabyte IPC message to the content process is
        // avoidable, and a bundle is ~1.6 MB.
        let piece = 1 << 20
        var offset = 0
        while offset < data.count {
            task.didReceive(data.subdata(in: offset..<min(data.count, offset + piece)))
            offset += piece
        }
        task.didFinish()
    }
}
