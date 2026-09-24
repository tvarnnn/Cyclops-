//
//  WorldRenderViewer.swift
//  Glasses
//

import Combine
import Foundation
import SwiftUI
import WebKit

// MARK: - What is being looked at

/// Which of the Tower's two renderings of the same world is wanted.
///
/// The Tower's render page is the **product** view of a world: the largest
/// connected component composed as one world and coloured by real photometric
/// RGB. A diagnostics variant of the same page — segment-index colouring, the
/// solver's own view — is requested with `?view=diagnostics`.
///
/// **Two ways this can silently do nothing, both known and both accepted.**
///
/// 1. This app loads the page with `loadHTMLString(_:baseURL: nil)` — no
///    origin, one navigation, everything else refused — so `location.search`
///    is empty and an in-page link goes nowhere. That is deliberate and is
///    not being loosened: an origin on that page would undo the reason
///    `WorldRenderClient` fetches it in the first place.
/// 2. So the Tower has to honour `view` **server-side**, when it composes
///    the page. It does: `tower/routes/geometry.py::world_render` declares
///    `view`, and `render.py` substitutes the opening mode into the script
///    rather than reading it from a URL the phone cannot supply. The page
///    was reading `location.search` until an iOS review pointed out that no
///    client of it can ever set one.
///
/// A Tower too old to declare `view` answers with the ordinary world page
/// and no error, because FastAPI ignores query parameters a route does not
/// declare. That is the right failure: the wearer asked to see something and
/// sees their world.
///
/// The sparse-geometry diagnostics that *this* app owns — the fragment gallery,
/// the placement counts, the refusal prose, the segment origins — do not depend
/// on any of that. They are behind the world screen's own Diagnostics
/// disclosure and are reachable whatever the Tower does here.
nonisolated enum WorldRenderView: String, Equatable, Hashable, Sendable {
    /// The default. Sends no `view` parameter at all, so the URL is byte-for-byte
    /// the one this app has always asked for.
    case product
    case diagnostics
}

/// A world, and optionally one of its sessions, whose viewer page can be
/// asked for: `GET /worlds/{world_id}/render`, `WORLD-BUILDER-WORLDS.md` §4.
///
/// `sessionID` is `nil` when the Tower is to choose — the newest session with
/// geometry — which is what a picker tap on the world's own row means. It is
/// never the empty string.
nonisolated struct WorldRenderTarget: Equatable, Hashable, Sendable, Identifiable {
    let worldID: String
    let sessionID: String?
    /// A `var` with a default rather than a `let` with one: a `let` with an
    /// initial value is left out of the memberwise initialiser entirely, which
    /// would make `WorldRenderTarget(worldID:sessionID:view:)` unavailable to
    /// the one call site that needs it.
    var view: WorldRenderView = .product
    /// One of the session's areas (`WORLD-BUILDER-COMPONENTS.md` §5), or `nil`
    /// for the room. An area target always names its session: the area routes
    /// carry it in the path.
    var areaID: String? = nil

    /// Includes the view and the area, so a navigation destination keyed on
    /// the target treats the room and each area as different screens -- which
    /// is what makes opening an area REPLACE the room viewer rather than stack
    /// on it (C1 E5).
    var id: String {
        "\(worldID)/\(sessionID ?? "")/\(view.rawValue)" + (areaID.map { "/area:\($0)" } ?? "")
    }

    var isArea: Bool { areaID != nil }

    /// The same world and session, in the other rendering.
    func showing(_ view: WorldRenderView) -> WorldRenderTarget {
        WorldRenderTarget(worldID: worldID, sessionID: sessionID, view: view, areaID: areaID)
    }

    /// One of this session's areas, by the id the listing or revision gave.
    func area(_ areaID: String) -> WorldRenderTarget {
        WorldRenderTarget(worldID: worldID, sessionID: sessionID, areaID: areaID)
    }

    /// The room of the same world and session: *Back to the room*.
    var room: WorldRenderTarget {
        WorldRenderTarget(worldID: worldID, sessionID: sessionID)
    }
}

/// What an area viewer needs to say about the area it shows, taken from the
/// list it was opened from and kept for the life of the screen (C1 M7): its
/// number, how many areas the walk has, and when it was captured.
nonisolated struct WorldAreaOpening: Equatable, Hashable, Sendable {
    let number: Int
    let total: Int
    let spans: [WorldCaptureSpan]

    /// The area viewer's header (§5.4), whole: the navigation title, which is
    /// also what accessibility reads.
    var header: String { WorldComponentsPresentation.areaHeader(number: number, of: total) }
    /// The header's two halves, drawn on two lines: on a phone the one-line
    /// title truncated to "Area 1 of 1 — not placed i…", losing the half that
    /// matters (the Mac's UI check of 46928fa).
    var numberLine: String { "Area \(number) of \(total)" }
    static let notPlacedLine = "not placed in the room"
}

// MARK: - The fetch

nonisolated enum WorldRenderFetchError: Error, Equatable {
    /// The Tower answered 404: no root, no such world or session, or no
    /// geometry built yet. Carries the Tower's own `detail`, which names
    /// which, when it could be read.
    case absent(detail: String?)
    /// A `world_id` or `session_id` the URL machinery refused. Both are
    /// wire-supplied, so this is a claim about the Tower's output, not this
    /// code, and is reported rather than force-unwrapped.
    case badAddress
    /// Any other non-2xx.
    case towerError(status: Int)
    /// A body that is not text.
    case undecodable
    case transport(String)
}

/// What `GET /worlds/{id}/render/revision` said, `WORLD-BUILDER-WORLDS.md` §4a.
nonisolated struct WorldRenderRevision: Equatable, Sendable {
    /// Opaque. Compared for equality only (§4a rule 1).
    let revision: String
    /// The rung that revision would serve, or `nil` for a value this app does
    /// not know. Read from the payload rather than from the revision, which
    /// is opaque: this is what decides whether a new picture replaces the one
    /// on screen by itself or is only offered.
    let representation: WorldRenderRepresentation?
    /// Whether the Tower is building this session right now. `nil` when the
    /// Tower did not say. `false` means "slow down", not "stop" (§4a rule 6).
    let live: Bool?
    /// The session's appearance build (`appearance.revision`), or `nil` when
    /// none is served. Opaque, and deliberately NOT part of `revision`: the
    /// appearance page follows these builds itself, overwriting its texture
    /// layers in place, so the follower never reloads the page for one -- that
    /// would reset the wearer's camera on every solve.
    var appearance: String? = nil
    /// `appearance.state` (`WORLD-BUILDER-APPEARANCE.md` §9): `served`,
    /// `rebuilding` (the label changed at Stop and the final build is coming;
    /// the page keeps what it drew), `withdrawn` or `absent`. `nil` from a
    /// Tower that does not say.
    var appearanceState: String? = nil
    /// The resolved session's `components` (`WORLD-BUILDER-COMPONENTS.md`
    /// §3.2), or `nil` when not computed. Deliberately NOT part of
    /// `revision`: an area finishing never swaps the room page. The room
    /// screen uses it to keep its areas row current.
    var components: WorldComponents? = nil
}

/// Fetches the viewer page over HTTP, as a string.
///
/// ## Why `URLSession` fetches the page and the web view only renders it
///
/// A `WKWebView` pointed straight at the route would work until the day it
/// did not, and on that day it would say nothing useful: its navigation
/// delegate sees a 404's status but never its body, so the Tower's careful
/// "no geometry yet" versus "no such world" would collapse into a blank
/// page. Fetching here means the failure arrives as a typed error with the
/// Tower's own words in it, the request carries the app's timeout and cache
/// policy like every other HTTP client in this app, and the web view is handed
/// that string through the app's own `glasses-world:` scheme
/// (`WorldAssetSchemeHandler`), which serves it and a short whitelist of this
/// world's routes and nothing else — so refusing every other navigation is
/// still a one-line policy rather than an allow-list.
///
/// Modelled on `WorldListClient`: a struct holding a `URL` and a session,
/// defaulted so a test can substitute a stubbed one.
nonisolated struct WorldRenderClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    /// Ephemeral and cache-less (`WorldAssetClient.uncachedSession()`), not
    /// `.shared`: the appearance page's configuration names the imagery it
    /// fetches, and `URLSession.shared` writes responses to a disk cache unless
    /// told otherwise. Privacy review §3.6.
    var session: URLSession = WorldAssetClient.sharedUncachedSession
    /// Longer than the list's ten seconds: a page for a full walk is a few
    /// megabytes of point coordinates, and a slow link is not a stalled one
    /// at that size. Still bounded (Rule 15).
    var timeout: TimeInterval = 30

    func page(for target: WorldRenderTarget) async throws -> String {
        guard let url = Self.url(for: target, baseURL: baseURL) else {
            throw WorldRenderFetchError.badAddress
        }
        // Local AND remote caches ignored, for the reason `WorldGeometryClient`
        // gives and for privacy §3.6; the Tower also answers `Cache-Control:
        // no-store`, and a world under construction changes with every build.
        let request = Self.request(url, timeout: timeout)
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw WorldRenderFetchError.transport(error.localizedDescription)
        }
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            if http.statusCode == 404 {
                throw WorldRenderFetchError.absent(detail: Self.detail(in: data))
            }
            throw WorldRenderFetchError.towerError(status: http.statusCode)
        }
        guard let html = String(data: data, encoding: .utf8) else {
            throw WorldRenderFetchError.undecodable
        }
        return html
    }

    /// Which picture the Tower would serve now, or `nil` when there is no
    /// revision to follow.
    ///
    /// A few hundred bytes, so an open picture can learn that a better one was
    /// built -- during a walk the surface is rebuilt each time a global solve
    /// lands -- without re-downloading megabytes to find out.
    ///
    /// `nil` means ONLY a Tower without the route: a 404 whose detail is
    /// FastAPI's own `Not Found`. Every other 404 is one of §4's sentences --
    /// "no geometry yet" among them, which a world mid-build can answer while
    /// its derived tree is replaced -- and is thrown, so the follower asks
    /// again rather than giving up for the life of the screen
    /// (`WORLD-BUILDER-WORLDS.md` §4a rule 5).
    func revision(for target: WorldRenderTarget) async throws -> WorldRenderRevision? {
        guard let url = Self.revisionURL(for: target, baseURL: baseURL) else {
            throw WorldRenderFetchError.badAddress
        }
        let request = Self.request(url, timeout: 10)
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw WorldRenderFetchError.transport(error.localizedDescription)
        }
        if let http = response as? HTTPURLResponse {
            if http.statusCode == 404 {
                let detail = Self.detail(in: data)
                if detail == WorldRenderFetchError.unmatchedRouteDetail { return nil }
                throw WorldRenderFetchError.absent(detail: detail)
            }
            if !(200..<300).contains(http.statusCode) {
                throw WorldRenderFetchError.towerError(status: http.statusCode)
            }
        }
        return try Self.decodeRevision(data)
    }

    /// Every request this client makes. Split out so the cache policy is
    /// tested without a stubbed session.
    static func request(_ url: URL, timeout: TimeInterval) -> URLRequest {
        URLRequest(url: url, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData,
                   timeoutInterval: timeout)
    }

    /// The revision route's body. Split out so the decoding is tested without
    /// a stubbed session.
    static func decodeRevision(_ data: Data) throws -> WorldRenderRevision {
        guard
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let revision = json["revision"] as? String, !revision.isEmpty
        else { throw WorldRenderFetchError.undecodable }
        let representation = (json["representation"] as? String)
            .flatMap(WorldRenderRepresentation.init(rawValue:))
        let appearanceObject = json["appearance"] as? [String: Any]
        let appearance = appearanceObject?["revision"] as? String
        let appearanceState = appearanceObject?["state"] as? String
        return WorldRenderRevision(
            revision: revision,
            representation: representation,
            live: json["live"] as? Bool,
            appearance: (appearance?.isEmpty ?? true) ? nil : appearance,
            appearanceState: (appearanceState?.isEmpty ?? true) ? nil : appearanceState,
            components: WorldComponents(json: json["components"])
        )
    }

    /// The revision route's address: the page's own path plus `/revision`,
    /// with the session and the `viewer` declaration, and without `view`,
    /// which does not change what is built. The declaration must match the
    /// page's: a revision answered without it names the surface rung while an
    /// appearance page is on screen.
    static func revisionURL(for target: WorldRenderTarget, baseURL: URL) -> URL? {
        if target.isArea {
            // `/worlds/<w>/areas/<s>/<a>/render/revision`, with no query at all
            // (§5.2): the same address the page's own poll is proxied to, so
            // the two cannot disagree.
            guard let page = url(for: target, baseURL: baseURL),
                  var components = URLComponents(url: page, resolvingAgainstBaseURL: false)
            else { return nil }
            components.percentEncodedPath += "/revision"
            components.queryItems = nil
            return components.url
        }
        guard
            let page = url(for: target, baseURL: baseURL),
            var components = URLComponents(url: page, resolvingAgainstBaseURL: false)
        else { return nil }
        components.percentEncodedPath += "/revision"
        let kept = (components.queryItems ?? []).filter {
            $0.name == "session_id" || $0.name == WorldAssetScheme.viewerQueryName
        }
        components.queryItems = kept.isEmpty ? nil : kept
        return components.url
    }

    /// The route's address, or `nil` when an id will not go into a URL.
    ///
    /// The id is percent-encoded as **one** path component by hand.
    /// `appendingPathComponent` does not encode `/` — a test showed
    /// `"a/b"` becoming two segments — so a wire-supplied id could otherwise
    /// address a route this client never meant to. Everything outside the
    /// unreserved set is encoded, `/` included.
    static func url(for target: WorldRenderTarget, baseURL: URL) -> URL? {
        guard !target.worldID.isEmpty, target.sessionID != "" else { return nil }
        if let areaID = target.areaID {
            // `/worlds/<w>/areas/<s>/<a>/render` (`WORLD-BUILDER-COMPONENTS.md`
            // §5.1). The session is in the path, never a query. No `viewer`:
            // the area route ignores it and offers the appearance rung to
            // every client that can name it (C1 M12). No `view`: an area has
            // no diagnostics rendering.
            guard
                let sessionID = target.sessionID,
                let session = sessionID.addingPercentEncoding(withAllowedCharacters: Self.unreserved),
                WorldComponent.isAreaID(areaID),
                var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false),
                let world = target.worldID.addingPercentEncoding(withAllowedCharacters: Self.unreserved)
            else { return nil }
            let base = components.percentEncodedPath.hasSuffix("/")
                ? String(components.percentEncodedPath.dropLast())
                : components.percentEncodedPath
            components.percentEncodedPath = "\(base)/worlds/\(world)/areas/\(session)/\(areaID)/render"
            components.queryItems = nil
            return components.url
        }
        guard
            var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false),
            let world = target.worldID.addingPercentEncoding(withAllowedCharacters: Self.unreserved)
        else { return nil }
        let base = components.percentEncodedPath.hasSuffix("/")
            ? String(components.percentEncodedPath.dropLast())
            : components.percentEncodedPath
        components.percentEncodedPath = "\(base)/worlds/\(world)/render"
        var query: [URLQueryItem] = []
        if let sessionID = target.sessionID {
            query.append(URLQueryItem(name: "session_id", value: sessionID))
        }
        // Only for the diagnostics rendering. The product view sends no `view`
        // parameter, so a Tower that has never heard of the parameter is never
        // sent it.
        if target.view != .product {
            query.append(URLQueryItem(name: "view", value: target.view.rawValue))
        }
        // What this app can draw (`WORLD-BUILDER-WORLDS.md` §4): the Tower's
        // `auto` offers the appearance page only to a client that says so,
        // because an older build cannot fetch its imagery. A Tower older than
        // the parameter ignores it (FastAPI drops unknown query parameters).
        query.append(URLQueryItem(name: WorldAssetScheme.viewerQueryName,
                                  value: WorldAssetScheme.viewerCapability))
        components.queryItems = query.isEmpty ? nil : query
        return components.url
    }

    /// RFC 3986 unreserved characters: the only ones an id may keep.
    private static let unreserved = CharacterSet(
        charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    )

    /// FastAPI's `{"detail": "..."}`, when that is what came back.
    static func detail(in data: Data) -> String? {
        guard
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let detail = json["detail"] as? String, !detail.isEmpty
        else { return nil }
        return detail
    }
}

// MARK: - The viewer's state

/// What the viewer sheet knows. The failure carries its sentence so the view
/// composes nothing.
///
/// ## Why fetching and rendering are two states
///
/// They were one, called `.loading`, and it ended at the moment the HTTP
/// response arrived. Everything after that was unobserved: the page was handed
/// to a `WKWebView` and the app never learned whether it drew. There was no
/// `didFinish`, no `didFail`, and no timeout covering the render — the client's
/// 30 s bound is on the **fetch** — so a page that failed to execute left the
/// wearer looking at a black rectangle that is indistinguishable from a crash,
/// with no spinner, no message and no end. On a multi-megabyte point cloud on a
/// phone under memory pressure that is not a hypothetical.
///
/// So the page's own life is a state now. `.rendering` means the string is in
/// the web view and `didFinish` has not arrived; it is bounded by
/// `WorldRenderViewerModel.renderTimeout` and resolves to `.ready` or to a
/// `.failed` that says which of the two halves gave up.
enum WorldRenderViewerState: Equatable {
    /// Asking the Tower for the page. Bounded by `WorldRenderClient.timeout`.
    case fetching
    /// The page is in the web view and has not finished loading. Bounded by
    /// `WorldRenderViewerModel.renderTimeout`.
    case rendering(html: String)
    /// `didFinish` arrived: the page is on screen.
    case ready(html: String)
    case failed(message: String, retryable: Bool)

    /// The page to hand the web view, in the two states that have one.
    ///
    /// Read by the view so that **one** `WorldRenderWebView` spans both:
    /// branching on the state inside a `ViewBuilder` would put the two in
    /// different `_ConditionalContent` arms, and SwiftUI would tear down the
    /// `WKWebView` and build a new one at the exact moment the first finished
    /// rendering — throwing away the render this state exists to observe.
    var html: String? {
        switch self {
        case .rendering(let html), .ready(let html): return html
        case .fetching, .failed: return nil
        }
    }

    /// Whether the page is in the web view and has not reported finishing.
    var isRendering: Bool {
        if case .rendering = self { return true }
        return false
    }

    /// Whether the page is still being asked for. The one state in which
    /// "Reload" would do nothing but restart a fetch already in flight.
    var isFetching: Bool {
        if case .fetching = self { return true }
        return false
    }

    /// The failure's sentence, or `nil`. A property rather than an `if case`
    /// binding in the view for the reason this codebase gives everywhere else:
    /// a result-builder block with a binding in it caused trouble in Product
    /// Shell V2.
    var failureMessage: String? {
        if case .failed(let message, _) = self { return message }
        return nil
    }

    var failureIsRetryable: Bool {
        if case .failed(_, let retryable) = self { return retryable }
        return false
    }

    /// Which reconstruction the Tower's page is, once there is a page.
    var representation: WorldRenderRepresentation? {
        html.flatMap(WorldRenderRepresentation.declared(in:))
    }

    /// The revision the Tower stamped into the page on screen, if any.
    var revision: String? {
        html.flatMap { WorldRenderRepresentation.meta(named: "wb-revision", in: $0) }
    }
}

/// Which rung of the Tower's reconstruction ladder a page is.
///
/// The Tower serves the best reconstruction a session has -- a surface, then
/// dense points, then the sparse points -- and each page names its rung in a
/// `<meta name="wb-representation">` tag. It is read from the page itself,
/// rather than from a response header, so `WorldRenderClient.page(for:)` keeps
/// its signature and every stub of it keeps working.
///
/// It exists for ONE sentence: the caption under the viewer. That caption used
/// to say "Not a surface" unconditionally, which was true of every page the
/// Tower could serve until it could serve a surface. A caption that denies what
/// is on screen is the same failure as one that overclaims.
nonisolated enum WorldRenderRepresentation: String, Equatable, Sendable {
    /// The wearer's own redacted keyframes blended over the surface
    /// (`WORLD-BUILDER-APPEARANCE.md`). The top rung.
    case appearance
    case surface
    case dense
    case sparse

    /// Whether a page of rung `latest` is a better picture than one of rung
    /// `shown`: sparse (or unknown) < dense < surface < appearance. Only an
    /// improvement
    /// replaces the picture on screen by itself; a rebuild of the same rung
    /// is offered instead, because a swap reloads the page and resets the
    /// reader's camera mid-look.
    static func isUpgrade(
        from shown: WorldRenderRepresentation?, to latest: WorldRenderRepresentation?
    ) -> Bool {
        rank(latest) > rank(shown)
    }

    private static func rank(_ representation: WorldRenderRepresentation?) -> Int {
        switch representation {
        case .appearance: return 3
        case .surface: return 2
        case .dense: return 1
        case .sparse, nil: return 0
        }
    }

    /// The rung a page declares, or `nil` for a page that declares none -- every
    /// page from a Tower that predates the declaration.
    static func declared(in html: String) -> WorldRenderRepresentation? {
        meta(named: "wb-representation", in: html).flatMap(WorldRenderRepresentation.init(rawValue:))
    }

    /// The `content` of a `<meta name=...>` the Tower wrote into the head.
    ///
    /// A plain scan, not a parser: the tags are written by the Tower, in the
    /// head, in one fixed spelling, and the page can be megabytes long, so
    /// only its first few kilobytes are looked at.
    static func meta(named name: String, in html: String) -> String? {
        let head = html.prefix(4096)
        guard let marker = head.range(of: "name=\"\(name)\" content=\"") else {
            return nil
        }
        let rest = head[marker.upperBound...]
        guard let end = rest.firstIndex(of: "\"") else { return nil }
        let value = String(rest[..<end])
        return value.isEmpty ? nil : value
    }

    /// `html` with the Tower's `wb-revision` stamp taken out of its head.
    ///
    /// Two pages that differ only in that stamp are the same picture. A page
    /// composed while the Tower could not read its manifest carries no stamp
    /// (`WORLD-BUILDER-WORLDS.md` §4a rule 7), and the same page fetched a
    /// moment later carries one -- which compared unequal, so the follower
    /// swapped an identical mesh in, with the "Drawing" overlay and a camera
    /// reset (review 3, iOS MINOR-3).
    static func withoutRevisionStamp(_ html: String) -> String {
        guard let value = meta(named: "wb-revision", in: html),
              let range = html.prefix(4096).range(of: "<meta name=\"wb-revision\" content=\"\(value)\">")
        else { return html }
        var unstamped = html
        unstamped.removeSubrange(range)
        return unstamped
    }

    /// The session a revision names. The Tower writes every revision as
    /// `<session>/<rung revision>` (§4a); `nil` for one with no session in it.
    static func session(of revision: String?) -> String? {
        guard let revision, let slash = revision.firstIndex(of: "/") else { return nil }
        return String(revision[..<slash])
    }

    /// The caption for a page of this rung, or for a page not yet known.
    ///
    /// Every variant keeps "not to scale" in it, because the one claim no rung
    /// can make yet is a real-world size: scale is unknown on every world.
    static func caption(for representation: WorldRenderRepresentation?) -> String {
        switch representation {
        case .appearance:
            // Agrees with the page's own caption and WORLD-BUILDER-WORLDS.md §4
            // ("what it draws"), which is where the page's drawing rules live.
            //
            // ## Why this sentence changed (review 2, M-5)
            //
            // It used to end "Dark gaps were not seen or were masked as
            // unreliable; nothing is filled in." Both halves stopped being true
            // at `21d6f1a`, in opposite directions:
            //
            // - **Nothing is filled in** is no longer true. A pixel with no
            //   surface whose two sides, within a few device pixels, lie on one
            //   plane is placed on that plane and shaded (`FS_FILL`). The
            //   PIXELS are still real camera pixels -- APPEARANCE §2 claim 1 and
            //   §3 hold -- but the GEOMETRY under them is invented there, and a
            //   wearer who is told "nothing is filled in" would read a closed
            //   crack as measured. It is bounded and the caption says so: only
            //   cracks a few pixels wide, only across one plane.
            // - **Dark gaps** is no longer the shape of the error either. A
            //   void is painted as an unlit grey fog lifted from the mean of
            //   what is drawn around it, not as black. "Grey haze" is what the
            //   wearer actually sees, and (since the fog's mean is
            //   coverage-weighted) it is always darker than the room beside it,
            //   so "haze" and "darker" agree.
            //
            // Short on purpose: this sits under a picture on a phone, and the
            // page's own About carries the long form.
            return "The camera's own images, faces redacted, placed on the reconstructed room. Grey haze is where no kept image looked; only cracks a few pixels wide are filled, from the images beside them. Not to scale."
        case .surface:
            // Agrees with the page's own caption and WORLD-BUILDER-SURFACE.md
            // section 2 claim 2: a gap is NOT proof that nobody looked. On the
            // canonical capture most measured depth with no surface near it
            // was outvoted or lacked a second agreeing view.
            return "Surfaces the Tower reconstructed from the walk, only where the cameras measured them. A gap is not proof that nothing is there. Not to scale."
        case .dense:
            return "Points the Tower measured densely from the walk. Not a surface, and not to scale."
        case .sparse:
            return "Points the Tower measured from the walk, with the camera path through them. Not a surface, and not to scale."
        case nil:
            return "What the Tower reconstructed from the walk. Not to scale."
        }
    }
}

/// Something the page itself did, reported by the web view's delegate.
///
/// A value rather than three closures, so the model has one entry point and a
/// test can drive the whole render lifecycle without a `WKWebView`.
nonisolated enum WorldRenderPageEvent: Equatable {
    /// `didFinish`: the page loaded and its script ran.
    case rendered
    /// `didFail` / `didFailProvisionalNavigation` with an error that is not a
    /// navigation this app deliberately cancelled.
    case failed(String)
    /// The content process died and the page is being put back. The page on
    /// screen is gone until it finishes, so this returns the screen to
    /// `.rendering` and re-arms its bound — without it, a reload that never
    /// completes is the same unobserved black rectangle one layer down.
    case reloadingAfterTermination
    /// The page asked to reload ITSELF (the appearance page's last-ditch
    /// recovery from a WebGL context WebKit never gave back) and the navigation
    /// policy allowed it. Handled exactly like a termination reload: the page
    /// on screen is gone until the new one finishes, so the screen goes back to
    /// a bounded `.rendering` with the "Drawing the world…" overlay over it.
    ///
    /// Without this the screen stayed `.ready` through the one recovery that
    /// exists to avoid an unexplained black rectangle, and `followRevisions`
    /// kept polling into a document that was mid-reload (review 2, m-9).
    case reloadingItself
    /// WebKit's content process was killed this many times and the web view
    /// has stopped putting the page back. See `Coordinator.reloadBudget`.
    case gaveUpAfterTerminations(Int)
}

/// What one revision poll means for an appearance page whose imagery was
/// withdrawn while it was on screen.
///
/// Pure, because the interesting cases are a relabel that comes back while the
/// Tower is briefly unreachable, and a page that has already fixed itself — and
/// neither is reproducible with a real `WKWebView`.
///
/// ## Two recoveries, and why the page's wins
///
/// When withdrawn imagery is served again there are two mechanisms that can put
/// the picture back, and before review 2 they BOTH ran:
///
/// - **The page's.** `appearance_viewer.html`'s own follower sees a non-null
///   `appearance.revision` against its nulled `S.revision`, refetches the
///   manifest and redraws **keeping the wearer's camera**. Nothing is
///   re-downloaded that the handler still holds.
/// - **The app's.** `refresh(to:evenIfUnchanged:)` replaces the whole document:
///   the camera resets to the opening pose, the "Drawing the world…" overlay
///   covers the screen, and because the withdrawn poll dropped the memory copy
///   the page fetches every chunk again — about 13 MB over Tailscale.
///
/// Both pollers run at 10 s, so they collided routinely and the slower, more
/// expensive one won (review 2, M-2). The app now waits one poll: if the page
/// fetched a served manifest through the scheme handler in the meantime
/// (`WorldAssetSchemeHandler.servedAppearanceToPageAt`), the page is alive and
/// has already recovered, and the app does nothing at all. Only a page that did
/// not take the imagery back — its script died, or it never started one — is
/// replaced. Exactly one mechanism acts.
nonisolated enum WorldAppearanceFollow: Equatable, Sendable {
    /// Nothing to do this poll; `withdrawn` is what the flag becomes.
    case keepWatching(withdrawn: Bool)
    /// Served again, and the page has already redrawn itself. Forget the
    /// withdrawal; do not touch the document.
    case leaveItToThePage
    /// Served again, but the page has not fetched a manifest yet. Give it one
    /// poll interval before replacing the document under it.
    case waitOnePollForThePage
    /// Served again and the page did not take it. Replace the document.
    case replaceThePage
}

nonisolated extension WorldRenderFetchError {
    /// What the sheet says. Each case names the thing that is missing or
    /// broken, and never blames the world for a link that dropped.
    var message: String {
        switch self {
        case .absent(let detail?) where detail == Self.unmatchedRouteDetail:
            // FastAPI's default for a path it does not serve, which is never
            // one of the contract's five sentences. That is a Tower older
            // than the route, not a world with nothing built.
            return "This Tower is too old to serve a picture of a saved world. Its worlds are still listed and their diagnostics still open."
        case .absent(let detail?) where WorldAreaRouteAnswer.isAreaAnswer(detail):
            // One of the area routes' four stable sentences (§5.1).
            return "The Tower has no picture of this area: \(detail)."
        case .absent(let detail?):
            // No "yet": the detail says which — a world still being built
            // answers "no geometry yet", and a world that does not exist
            // answers "no world", and only the Tower knows which.
            return "The Tower has no picture for this world: \(detail)."
        case .absent(nil):
            return "The Tower has no picture for this world — no geometry has been built for it, or the world is gone."
        case .badAddress:
            return "This world's identifier could not be made into a Tower address."
        case .towerError(let status):
            return "The Tower could not produce the picture (HTTP \(status))."
        case .undecodable:
            return "The Tower's page could not be read as text."
        case .transport(let detail):
            return "The picture could not be fetched: \(detail)"
        }
    }

    /// Whether trying again could change the answer. A world still being
    /// built answers 404 until its first solve lands, so `absent` is
    /// retryable; a bad id is not, and neither is a Tower without the route.
    var isRetryable: Bool {
        switch self {
        case .badAddress, .undecodable: return false
        case .absent(let detail?) where detail == Self.unmatchedRouteDetail: return false
        // An area that is gone, could not be built, or never existed will not
        // appear on a retry; one not built YET might (§5.1).
        case .absent(let detail?) where WorldAreaRouteAnswer.isTerminal(detail): return false
        case .absent, .towerError, .transport: return true
        }
    }

    /// What FastAPI answers for a path no route serves.
    static let unmatchedRouteDetail = "Not Found"
}

/// Owns one fetch of one page. Created by the sheet and destroyed with it;
/// it holds a string, not a connection, so losing it loses nothing.
@MainActor
final class WorldRenderViewerModel: ObservableObject {
    @Published private(set) var state: WorldRenderViewerState = .fetching

    /// The world, the session and which rendering was asked for. A `let`: this
    /// screen shows one page.
    ///
    /// It was briefly a `@Published var` with a `show(_:)` beside it, so a
    /// segmented control in Details could switch renderings. That control is
    /// gone — see `details` for why — and with it the only reason this could
    /// change. A reader who wants the other rendering either uses the page's
    /// own button, which costs nothing, or opens it from the world screen's
    /// Diagnostics section, which is a new screen with its own model.
    let target: WorldRenderTarget
    private let client: WorldRenderClient

    /// How long the page gets to draw itself after the fetch succeeds.
    ///
    /// Generous: the Tower decimates to a mobile budget, but the script still
    /// builds typed arrays over tens of thousands of points on a phone. Bounded
    /// all the same (Rule 15) — an unbounded wait here is exactly the black
    /// rectangle this state was added to end. Injectable so a test can prove
    /// the bound exists in milliseconds rather than spending twenty seconds.
    var renderTimeout: Duration = .seconds(20)

    /// Bumped on every `load()`.
    ///
    /// Handed to `WorldRenderWebView` so a **retry of the same page** actually
    /// reloads. The coordinator skips a `loadHTMLString` when the string has
    /// not changed — which is right for the re-renders SwiftUI does constantly,
    /// and wrong for a deliberate "Try again" after a render failure, where the
    /// string is identical and reloading it is the entire point.
    @Published private(set) var renderAttempt = 0

    /// Which content-process-kill budget the page on screen is under.
    ///
    /// Separate from `renderAttempt` because the two questions came apart
    /// (review 2, m-1). "Load this string again even though it is identical" is
    /// `renderAttempt`; "and give it a fresh kill budget" is this. They agreed
    /// while the only forced reloads were the reader's own "Try again" and a
    /// revert of a page that had already drawn — both of which do deserve a
    /// fresh budget. The withdrawal-recovery reload does not: it fires once per
    /// relabel cycle, and handing the page a new budget each time made
    /// `Coordinator.reloadBudget` unbounded for exactly the page most likely to
    /// need it.
    @Published private(set) var renderBudgetToken = 0

    /// Bounds `.rendering`. Cancelled when the page reports either way.
    ///
    /// Unstructured (`Task { }`), so it does not die with the SwiftUI `.task`
    /// that started the fetch; `[weak self]` so a dismissed sheet is not kept
    /// alive for the length of the timeout.
    private var renderWatchdog: Task<Void, Never>?

    /// How often an open picture asks whether a better one has been built,
    /// while the Tower says it is building.
    ///
    /// Ten seconds against a live surface that is rebuilt when a global solve
    /// lands -- tens of seconds apart on a real walk -- so a new picture is on
    /// screen within one interval of existing. Injectable so a test proves the
    /// behaviour in milliseconds.
    var revisionPollInterval: Duration = .seconds(10)

    /// The longest the follower waits between asks when nothing is building.
    ///
    /// A saved, finished world never changes, and asking every ten seconds for
    /// as long as it is open keeps the radio awake for nothing. So while the
    /// Tower answers `live: false` and the revision stays put, the interval
    /// doubles up to this. It is a ceiling and not a stop, because `false` is
    /// not a promise: the Tower starts the final surface a few seconds AFTER it
    /// releases the world lock (`WORLD-BUILDER-WORLDS.md` §4a rule 6), and a
    /// follower that stopped there would miss the picture the walk was for.
    var revisionPollCeiling: Duration = .seconds(120)

    /// The revision stamped into the page on screen (or, for a page that
    /// carries none, the one that led to it). Starts as the page's own stamp, so
    /// a build that lands between fetching the page and the first poll is still
    /// noticed.
    private var shownRevision: String?

    /// The last revision the follower ACTED on: fetched, offered, or found to
    /// change nothing. Separate from `shownRevision` because the two can differ
    /// for good -- the Tower's revision route can say "surface" about a page it
    /// then serves as a lower rung (§4a rule 4) -- and comparing only with the
    /// page's stamp would download that page again on every poll.
    private var handledRevision: String?

    /// Revisions whose page could not be drawn on this phone. Never offered or
    /// swapped in again by the follower -- except once, for a revision refused
    /// while its build was live that the Tower then reports finished
    /// (`refusedWhileLive`) -- and forgotten by the reader's own "Try again"
    /// (`newerPictureRefused`, then `load()`).
    private var refusedRevisions: Set<String> = []

    /// The subset of `refusedRevisions` refused while the Tower said the build
    /// was live, each owed one try when the same revision is reported finished.
    /// See `followRevisions`.
    private var refusedWhileLive: Set<String> = []

    /// How many refreshes to each rung failed to draw, and the rungs that have
    /// failed twice. A rung too large for this phone used to be refused per
    /// REVISION, so every live build of it was downloaded, killed and reverted
    /// again (review 2, iOS m4). Twice, not once: one failure can be memory
    /// another app was holding.
    ///
    /// Only a failure to go UP a rung counts -- a page of the rung already on
    /// screen drew on this phone, so its rebuild failing is pressure, not size
    /// -- and a refresh of the rung that does draw forgets its failures. Both
    /// were missing, so two transient failures in a walk refused the rung for
    /// the life of the screen (review 3, R2). Cleared by `load()`.
    private var rungDrawFailures: [WorldRenderRepresentation: Int] = [:]
    private var refusedRungs: Set<WorldRenderRepresentation> = []

    /// Refused rungs a FINISHED build (`live == false`) has already been let
    /// through once. The finished world after Stop is the picture the walk was
    /// for, and the capture that was competing for memory has stopped; it gets
    /// one more try per rung, and only one, so a rung that really is too large
    /// is not downloaded on every poll of a saved world.
    private var finishedBuildRetried: Set<WorldRenderRepresentation> = []

    /// A refresh could not be drawn and the previous page was put back. Published
    /// so the scene can offer "Try again" (`load()`), which forgets every
    /// refusal. A reverted screen is `.ready`, where the failure view and its
    /// button are not shown, so without this the refusals could only be lifted
    /// by closing the screen (review 3, R2).
    @Published private(set) var newerPictureRefused = false

    /// The page that was on screen when an automatic refresh replaced it, and
    /// the revision it carried, kept until the replacement reports drawn. A
    /// refresh that cannot be drawn -- watchdog, `didFail`, or the content
    /// process running out of memory, which is exactly what a larger rung does
    /// -- returns to it instead of to a failure. `wasLive` is whether the
    /// Tower said the build being swapped IN was live, so a refusal can be
    /// told apart from one of a finished build (`refusedWhileLive`).
    private var fallback: (html: String, revision: String?, wasLive: Bool)?

    /// A newer build of the SAME rung, waiting for the reader to ask for it.
    ///
    /// Published so the scene can show a button. A same-rung rebuild is not
    /// swapped in by itself: a swap reloads the page, which covers it with the
    /// "Drawing" overlay and resets the camera and the walk/orbit mode, and a
    /// surface is rebuilt on every global solve during a walk.
    @Published private(set) var newerPictureAvailable = false
    private var pendingRevision: String?

    /// The appearance page on screen was told its imagery is withdrawn (a
    /// relabel, a purge: `appearance.state` neither `served` nor `rebuilding`)
    /// and dropped its textures. When a served appearance comes back the page is
    /// replaced by itself, whatever its page revision says (review 1, B1): the
    /// page recovers on its own, but one from a Tower that predates epochs (or
    /// one whose script died) would otherwise show "no longer served" for good,
    /// because an unchanged page revision is never fetched again. Not set by
    /// `rebuilding`, the ordinary Stop, where the page keeps its textures and
    /// picks the final build up in place without a reload.
    private var appearanceWithdrawnWhileShown = false
    /// When that flag was last set, so a manifest the PAGE fetched afterwards
    /// can be told from one it fetched before (`WorldAppearanceFollow`).
    private var appearanceWithdrawnAt: Date?
    /// Whether the page has already been given its one poll's grace to recover
    /// on its own since the appearance came back.
    private var appearanceWaitedForPage = false

    /// Serves the page and proxies this world's whitelisted routes.
    ///
    /// **The viewer owns it, not the web view.** The in-memory copy of the
    /// imagery is the VIEWER's (`PRIVACY.md` §3.6: "in-memory reuse within one
    /// open viewer is fine; drop it when the viewer closes"), and a "Try again"
    /// destroys the `WKWebView` and builds another one. Owning it here also
    /// gives the follower the one fact only the handler knows: whether the page
    /// itself has fetched a served manifest (`WorldAppearanceFollow`).
    let assets: WorldAssetSchemeHandler

    /// The room's `components` (`WORLD-BUILDER-COMPONENTS.md` §3.2): seeded
    /// from the listing row the viewer was opened from, then kept current from
    /// every revision the follower reads. `nil` is not computed -- every older
    /// world -- and draws exactly today's screen. Never set on an area viewer.
    @Published private(set) var components: WorldComponents?

    /// The Tower said this area will not be served (§5.2): kept picture,
    /// follower stopped, and the scene offers the way back to the room.
    @Published private(set) var areaNoLongerServed = false

    /// `assets` is injectable for the same reason `client` is: the follower's
    /// one piece of evidence about the page's own recovery comes through the
    /// handler, and a test needs to be able to put it there from a stubbed
    /// Tower rather than a real one.
    init(
        target: WorldRenderTarget,
        client: WorldRenderClient = WorldRenderClient(),
        assets: WorldAssetSchemeHandler? = nil,
        components: WorldComponents? = nil
    ) {
        self.target = target
        self.components = target.isArea ? nil : components
        self.client = client
        self.assets = assets
            ?? WorldAssetSchemeHandler(worldID: target.worldID, scope: WorldAssetScope.of(target))
    }

    /// The viewer closed. Drops the in-memory imagery and its authorisation,
    /// which `PRIVACY.md` §3.6 and `WORLD-BUILDER-IOS.md` §10 promise and which
    /// nothing did: `dropCache()` had no caller at all, and up to 64 MB of
    /// first-person room imagery was left to ARC releasing an object graph that
    /// `WKWebView` teardown is exactly the place to linger in (review 2, M-4).
    func viewerClosed() {
        assets.tearDown()
    }

    /// Keep the areas row current from a revision the Tower sent (§3.2),
    /// republishing only when something a reader can see moved.
    ///
    /// A `nil` from the Tower replaces the list the viewer had: `null` is the
    /// Tower's current answer ("not computed"), and a re-solve that dropped
    /// its components must not keep offering areas whose routes now 404.
    func adoptComponents(_ latest: WorldComponents?) {
        guard !target.isArea else { return }
        if latest?.displayKey != components?.displayKey {
            components = latest
        }
    }

    /// One revision read, for the areas row alone, as the screen opens: the
    /// follower's first poll is an interval away, and a room opened from the
    /// live screen has no listing row to seed it. Errors are ignored -- the
    /// follower asks again -- and the page is never swapped from here.
    func refreshComponents() async {
        guard !target.isArea, target.view == .product else { return }
        guard let latest = try? await client.revision(for: target) else { return }
        guard !Task.isCancelled else { return }
        adoptComponents(latest.components)
    }

    /// Keep the picture current for as long as the screen is open.
    ///
    /// Runs in the screen's `.task`, so it ends when the screen goes. Asks only
    /// while a page is actually on screen (`.ready`): a page still drawing, or a
    /// failure the reader is reading, is left alone. Returns for good when the
    /// Tower has no revision route, and never starts for the diagnostics
    /// rendering: that page is always the sparse one, and §4a would compare it
    /// with the product ladder's revision and refetch it on every rebuild.
    ///
    /// A better RUNG replaces the page by itself; a rebuild of the same rung
    /// sets `newerPictureAvailable` and waits for `showNewerPicture()`.
    ///
    /// **An appearance build is never acted on here.** It arrives as
    /// `WorldRenderRevision.appearance`, which is deliberately not part of
    /// `revision`: the appearance page polls the same route through the scheme
    /// handler and overwrites its texture layers in place, keeping the camera.
    /// Only the PAGE revision -- a rung change, a session change, a page
    /// program change -- reloads, exactly as before.
    func followRevisions() async {
        guard target.view == .product else { return }
        var interval = revisionPollInterval
        while !Task.isCancelled {
            try? await Task.sleep(for: interval)
            guard !Task.isCancelled else { return }
            guard case .ready = state else { continue }
            if shownRevision == nil { shownRevision = state.revision }
            let latest: WorldRenderRevision?
            do {
                latest = try await client.revision(for: target)
            } catch {
                // An area the Tower says will not be served -- re-finished
                // away, failed, or never computed (§5.2, C1 E4) -- is terminal
                // for this id: keep the picture, stop following, and say so,
                // so the reader can go back to the room, whose areas row is
                // current. Everything else is transient.
                if target.isArea, case .absent(let detail) = error as? WorldRenderFetchError,
                   WorldAreaRouteAnswer.isTerminal(detail) {
                    areaNoLongerServed = true
                    return
                }
                // A dropped request, or one of §4's own 404 sentences, is not a
                // reason to change anything on screen; the next interval asks
                // again.
                continue
            }
            guard !Task.isCancelled else { return }
            guard let latest else { return }
            adoptComponents(latest.components)
            // A revision refused while its build was LIVE gets one more try
            // when the Tower reports that same revision FINISHED -- the
            // per-revision form of `finishedBuildRetried`, and for the same
            // reason: the finished world is the picture the walk was for, and
            // the capture that was competing for memory has stopped.
            //
            // It matters for the appearance, whose page revision
            // (`<session>/appearance:1@<epoch>`) survives the ordinary Stop.
            // Without it, one walk-time appearance page that could not be drawn
            // refused the FINISHED world's page too: the Stop gap's surface
            // then drew, took the "Try again" button away with it, and the
            // photographic world that followed under the same revision was
            // never tried, never offered and never mentioned -- the bare
            // surface stayed on screen over it. Found at the 2026-09-22 Mac
            // validation of 83534e2.
            //
            // A revision refused when it was already finished is not retried,
            // and this one retry is spent once taken, so a page too large for
            // this phone is still fetched a bounded number of times.
            if latest.live == false, refusedWhileLive.remove(latest.revision) != nil {
                refusedRevisions.remove(latest.revision)
                if handledRevision == latest.revision { handledRevision = nil }
            }
            if state.representation == .appearance {
                let pageRecovered = appearanceWithdrawnAt.map { withdrawnAt in
                    (assets.servedAppearanceToPageAt ?? .distantPast) > withdrawnAt
                } ?? false
                switch Self.appearanceFollow(
                    withdrawn: appearanceWithdrawnWhileShown,
                    waitedForThePage: appearanceWaitedForPage,
                    pageRecovered: pageRecovered,
                    latest: latest
                ) {
                case .keepWatching(let withdrawn):
                    if withdrawn, !appearanceWithdrawnWhileShown { appearanceWithdrawnAt = Date() }
                    if !withdrawn { forgetTheWithdrawal() }
                    appearanceWithdrawnWhileShown = withdrawn
                case .leaveItToThePage:
                    forgetTheWithdrawal()
                case .waitOnePollForThePage:
                    appearanceWaitedForPage = true
                    interval = revisionPollInterval
                    continue
                case .replaceThePage:
                    let sameWalk = target.sessionID != nil
                        || WorldRenderRepresentation.session(of: latest.revision)
                            == WorldRenderRepresentation.session(of: shownRevision)
                    if sameWalk {
                        interval = revisionPollInterval
                        // **The flag survives a failed fetch** (review 2, M-1).
                        // It used to be cleared before this call, and `refresh`
                        // returns silently on any fetch failure — so one
                        // unreachable moment, from the same Tower that had just
                        // finished the rebuild, disarmed the only recovery this
                        // screen had and left the wearer on "no longer served"
                        // until they closed it. On a pre-epoch world (review 2,
                        // A-1) the page revision never changes, so that flag is
                        // the ONLY path back.
                        if await refresh(to: latest.revision, evenIfUnchanged: true,
                                         live: latest.live == true) {
                            forgetTheWithdrawal()
                        }
                        continue
                    }
                    // Another walk's world. The flag is deliberately NOT
                    // cleared: this walk's imagery is still withdrawn on the
                    // page, and the rules below offer the other walk's build
                    // rather than swapping the reader's world out.
                }
            }
            if latest.revision == shownRevision, newerPictureAvailable {
                // The build that was offered is no longer what the Tower would
                // serve; the button must not outlive its reason.
                pendingRevision = nil
                newerPictureAvailable = false
            }
            let isNew = latest.revision != shownRevision
                && latest.revision != handledRevision
                && !refusedRevisions.contains(latest.revision)
            interval = Self.nextPollInterval(
                after: interval, base: revisionPollInterval, ceiling: revisionPollCeiling,
                live: latest.live, changed: isNew
            )
            guard isNew else { continue }
            if let rung = latest.representation, refusedRungs.contains(rung) {
                if latest.live == false, !finishedBuildRetried.contains(rung) {
                    // The finished build gets one more try (see
                    // `finishedBuildRetried`), and goes through every rule below.
                    finishedBuildRetried.insert(rung)
                } else {
                    // This phone could not draw that rung twice; neither
                    // swapped nor offered. "Try again" still can.
                    handledRevision = latest.revision
                    continue
                }
            }
            if WorldRenderRepresentation.isUpgrade(from: latest.representation, to: state.representation) {
                // A worse rung than the one on screen -- a surface briefly
                // undrawable on the Tower, a pruned artifact -- is not "a newer
                // reconstruction" and is not offered as one (review 2, iOS m6).
                handledRevision = latest.revision
                continue
            }
            // With no session named the Tower answers its newest session with
            // geometry, which can be another walk; the session in the revision
            // is what tells the two apart.
            let sameWalk = target.sessionID != nil
                || WorldRenderRepresentation.session(of: latest.revision)
                    == WorldRenderRepresentation.session(of: shownRevision)
            if Self.swapsBySelf(shown: state.representation, latest: latest, sameWalk: sameWalk) {
                await refresh(to: latest.revision, live: latest.live == true)
            } else {
                handledRevision = latest.revision
                pendingRevision = latest.revision
                newerPictureAvailable = true
            }
        }
    }

    /// Whether a new revision replaces the page without asking.
    ///
    /// A better rung always does. So does a build of the same rung that the
    /// Tower says is FINISHED (`live == false`): that is the final surface
    /// built after Stop, a world does not rebuild again after it, and a wearer
    /// who stopped walking should see the finished world rather than a button.
    /// A same-rung build while the walk is live is offered, because live builds
    /// land on every solve and a swap resets the camera mid-look. Nothing
    /// replaces a page with a worse rung by itself.
    ///
    /// Nothing swaps by itself unless `sameWalk`: the screen names its session,
    /// or the revision names the session already on screen. With no session in
    /// the target the Tower picks the newest session with geometry, so a new
    /// revision -- a better rung as much as a finished build -- can be ANOTHER
    /// walk's world, and swapping it in without asking would replace the world
    /// the reader opened (review 2, iOS m5; review 3, iOS MINOR-4). It is
    /// offered instead. No default: a caller that forgets to say was the
    /// regression a review could not see (review 3, T-1).
    nonisolated static func swapsBySelf(
        shown: WorldRenderRepresentation?, latest: WorldRenderRevision, sameWalk: Bool
    ) -> Bool {
        guard sameWalk else { return false }
        if WorldRenderRepresentation.isUpgrade(from: shown, to: latest.representation) {
            return true
        }
        let downgrade = WorldRenderRepresentation.isUpgrade(
            from: latest.representation, to: shown
        )
        return !downgrade && latest.live == false
    }

    /// What one poll means for an appearance page on screen. Pure, so the whole
    /// rule is tested without a page, a web view or a network.
    ///
    /// - a served appearance, nothing withdrawn: `keepWatching(false)`;
    /// - a served appearance of this rung after a withdrawal: the page's own
    ///   recovery first (`leaveItToThePage` if it already fetched a manifest,
    ///   else `waitOnePollForThePage` once, then `replaceThePage`);
    /// - a served appearance the app is NOT being given (the revision names a
    ///   lower rung): `keepWatching(false)` — there is nothing to reload into;
    /// - `rebuilding` (the ordinary Stop): the flag is unchanged, because that
    ///   is not a withdrawal and the page kept its textures;
    /// - anything else with no served appearance: `keepWatching(true)`.
    nonisolated static func appearanceFollow(
        withdrawn: Bool, waitedForThePage: Bool, pageRecovered: Bool, latest: WorldRenderRevision
    ) -> WorldAppearanceFollow {
        guard latest.appearance != nil else {
            if latest.appearanceState == "rebuilding" { return .keepWatching(withdrawn: withdrawn) }
            return .keepWatching(withdrawn: true)
        }
        guard withdrawn, latest.representation == .appearance else {
            return .keepWatching(withdrawn: false)
        }
        if pageRecovered { return .leaveItToThePage }
        return waitedForThePage ? .replaceThePage : .waitOnePollForThePage
    }

    /// An appearance page whose imagery this app cannot even ask for: it
    /// declares the appearance rung, but nothing names the session it draws, so
    /// `WorldAssetRequest.parse` refuses every route but the page itself. Pure
    /// (review 2, m-16).
    /// Why an area viewer will not draw `html`, or `nil` when it may: the
    /// page's `wb-area` must be this area, and its `wb-revision` must name this
    /// session's area (`<s>/area:<a>/…`, `WORLD-BUILDER-COMPONENTS.md` §5.1).
    /// Always `nil` for the room.
    nonisolated static func areaPageMismatch(html: String, target: WorldRenderTarget) -> String? {
        guard let areaID = target.areaID, let sessionID = target.sessionID else { return nil }
        let declaredArea = WorldRenderRepresentation.meta(named: "wb-area", in: html)
        let revision = WorldRenderRepresentation.meta(named: "wb-revision", in: html)
        guard declaredArea == areaID, revision?.hasPrefix("\(sessionID)/area:\(areaID)/") == true else {
            return "The Tower's page is not a page of this area, so it was not drawn."
        }
        return nil
    }

    nonisolated static func pageCannotReachItsImagery(
        html: String, target: WorldRenderTarget
    ) -> Bool {
        WorldRenderRepresentation.declared(in: html) == .appearance
            && WorldAssetSchemeHandler.session(for: target, page: html) == nil
    }

    /// The withdrawal is over, however it ended.
    private func forgetTheWithdrawal() {
        appearanceWithdrawnWhileShown = false
        appearanceWaitedForPage = false
        appearanceWithdrawnAt = nil
    }

    /// How long to wait before the next ask.
    ///
    /// Back to `base` whenever the revision changed or the Tower is building;
    /// otherwise double, up to `ceiling`. A Tower that does not say (`nil`) is
    /// treated as not building: slower, never silent.
    nonisolated static func nextPollInterval(
        after current: Duration, base: Duration, ceiling: Duration, live: Bool?, changed: Bool
    ) -> Duration {
        if changed || live == true { return base }
        return min(max(current * 2, base), ceiling)
    }

    /// Swap in the newer same-rung picture the reader asked for.
    func showNewerPicture() async {
        guard let revision = pendingRevision, case .ready = state else { return }
        pendingRevision = nil
        newerPictureAvailable = false
        // A tap is not the follower's live swap; its failure is not owed a retry.
        await refresh(to: revision, live: false)
    }

    /// Fetch the newer page and put it on screen, or leave the old one.
    ///
    /// Never `.fetching` and never `.failed`: the reader is looking at a world,
    /// and a refresh that fails must not take it away -- not the fetch, and not
    /// the draw either (see `fallback`). `renderAttempt` is NOT bumped, so the
    /// web view's content-process kill budget is not reset by an automatic
    /// refresh. The page's own stamped revision is recorded, not the polled one,
    /// so a build that lands between the poll and the fetch costs no second
    /// download.
    /// Returns whether a page was actually put on screen to be drawn — which is
    /// what tells a withdrawal-recovery that it worked (`WorldAppearanceFollow`).
    @discardableResult
    private func refresh(
        to revision: String, evenIfUnchanged: Bool = false, live: Bool = false
    ) async -> Bool {
        let html: String
        do {
            html = try await client.page(for: target)
        } catch {
            return false
        }
        guard !Task.isCancelled, case .ready(let current) = state else { return false }
        handledRevision = revision
        let stamped = WorldRenderViewerState.ready(html: html).revision ?? revision
        // `evenIfUnchanged`: the page on screen dropped its textures, so the
        // same page must be loaded again (a new attempt reloads an identical
        // string; see `renderAttempt`).
        if evenIfUnchanged, html == current {
            renderWatchdog?.cancel()
            // **A fallback, like every other path into `.rendering`** (review 2,
            // m-1). This branch used to clear it, so a reload of the identical
            // page that then hit the watchdog or a `didFail` had nothing to
            // revert to and took a world off the screen that had been drawing a
            // moment earlier. Reverting to the same string is not a no-op: it
            // reloads it with a fresh kill budget and offers "Try again".
            fallback = (html: current, revision: shownRevision, wasLive: live)
            shownRevision = stamped
            // The identity changes so an identical string is loaded again; the
            // BUDGET token does not, because nobody asked for a fresh
            // content-process-kill budget (review 2, m-1).
            renderAttempt += 1
            state = .rendering(html: html)
            startRenderWatchdog()
            return true
        }
        guard evenIfUnchanged || (html != current
              && WorldRenderRepresentation.withoutRevisionStamp(html)
                != WorldRenderRepresentation.withoutRevisionStamp(current))
        else {
            // The same picture, possibly now carrying the stamp it lacked.
            shownRevision = stamped
            return false
        }
        // The page is what decides, not the poll that led to it. A tapped offer
        // whose build has since become undrawable on the Tower, or a fetch that
        // raced a manifest replace, is served a lower rung, and swapping that in
        // replaced a surface with points (review 3, iOS MINOR-2 and e2e m4).
        guard !WorldRenderRepresentation.isUpgrade(
            from: WorldRenderRepresentation.declared(in: html), to: state.representation
        ) else { return false }
        guard !refusedRevisions.contains(stamped) else { return false }
        // Whatever was offered is superseded by the page now on its way.
        pendingRevision = nil
        newerPictureAvailable = false
        renderWatchdog?.cancel()
        fallback = (html: current, revision: shownRevision, wasLive: live)
        shownRevision = stamped
        state = .rendering(html: html)
        startRenderWatchdog()
        return true
    }

    /// Put the last drawn page back after an automatic refresh failed to draw.
    ///
    /// Returns `false` when there is nothing to fall back to -- no refresh was
    /// in flight, or the fallback page itself is what failed -- and the caller
    /// reports the failure as before. Bounded: the refused revision is never
    /// swapped in again, and `fallback` is consumed here.
    private func revertRefresh() -> Bool {
        guard let previous = fallback else { return false }
        fallback = nil
        if let refused = shownRevision { refusedRevisions.insert(refused) }
        if let handled = handledRevision { refusedRevisions.insert(handled) }
        if previous.wasLive {
            if let refused = shownRevision { refusedWhileLive.insert(refused) }
            if let handled = handledRevision { refusedWhileLive.insert(handled) }
        }
        if let rung = state.representation,
           rung != WorldRenderRepresentation.declared(in: previous.html) {
            let failures = rungDrawFailures[rung, default: 0] + 1
            rungDrawFailures[rung] = failures
            if failures >= 2 { refusedRungs.insert(rung) }
        }
        newerPictureRefused = true
        shownRevision = previous.revision
        // A fresh kill budget for a page that already drew once on this phone.
        renderAttempt += 1
        renderBudgetToken += 1
        state = .rendering(html: previous.html)
        startRenderWatchdog()
        return true
    }

    // No `deinit` cancelling the watchdog, deliberately. It captures `[weak
    // self]`, so a dismissed sheet is released immediately and the task that
    // outlives it holds nothing but a sleeping timer — and a `deinit` on a
    // global-actor-isolated class reaching for an isolated stored property is
    // the kind of thing that is legal today and an error the next time the
    // language mode moves. The `[weak self]` is the guarantee; the `deinit`
    // would only have been an optimisation of a timer.

    func load() async {
        renderWatchdog?.cancel()
        renderWatchdog = nil
        renderAttempt += 1
        renderBudgetToken += 1
        forgetTheWithdrawal()
        fallback = nil
        pendingRevision = nil
        newerPictureAvailable = false
        handledRevision = nil
        rungDrawFailures = [:]
        refusedRungs = []
        finishedBuildRetried = []
        // The refused REVISIONS too, as `WORLD-BUILDER-IOS.md` §10 and the
        // property's own comment always said. They were the one refusal this
        // kept, and for the appearance rung that is not a detail: its page
        // revision (`<session>/appearance:1@<epoch>`) does not change between
        // the builds of one epoch, and the ordinary Stop keeps the epoch. So
        // one walk-time appearance page that failed to draw refused the
        // FINISHED world's page as well, for the life of the screen -- and a
        // "Try again" tapped while the Tower served the surface in the Stop
        // gap cleared the button without clearing the refusal, leaving the
        // bare surface on screen over a photographic world with nothing
        // saying so. Found at the 2026-09-22 Mac validation of 83534e2.
        refusedRevisions = []
        refusedWhileLive = []
        newerPictureRefused = false
        state = .fetching
        do {
            var html = try await client.page(for: target)
            guard !Task.isCancelled else { return }
            if Self.pageCannotReachItsImagery(html: html, target: target) {
                // One refetch, then say so (review 2, m-16). The Tower composes
                // an appearance page with no `wb-revision` stamp when
                // `_appearance_revision` raced away between choosing the rung
                // and stamping it (WORLDS §4a rule 7). With no pinned session
                // that stamp is the ONLY thing that tells the scheme handler
                // which session's routes to proxy, so every manifest, chunk and
                // proxy request is refused locally, nothing reaches the Tower,
                // the Tower log shows nothing, and the field report reads "the
                // world opened empty". The race is narrow and the refetch wins
                // it; the sentence is for when it does not.
                html = try await client.page(for: target)
                guard !Task.isCancelled else { return }
                if Self.pageCannotReachItsImagery(html: html, target: target) {
                    state = .failed(
                        message: "The Tower's page for this world arrived without the session its "
                            + "images belong to, so none of them could be fetched.",
                        retryable: true
                    )
                    return
                }
            }
            // An area viewer draws only the area it was opened for (C1 M10):
            // the page must say so in its own head. A page naming another
            // area, another session, or no area at all is a Tower answering a
            // different question, and drawing it would put the wrong walk's
            // imagery under this area's caption.
            if let refusal = Self.areaPageMismatch(html: html, target: target) {
                state = .failed(message: refusal, retryable: true)
                return
            }
            // Not `.ready`. The page has arrived; nothing has drawn it yet.
            state = .rendering(html: html)
            shownRevision = state.revision
            startRenderWatchdog()
        } catch let error as WorldRenderFetchError {
            // A dismissed sheet cancels the task mid-fetch; that is not a
            // failure to report, and nobody is looking.
            guard !Task.isCancelled else { return }
            state = .failed(message: error.message, retryable: error.isRetryable)
        } catch {
            guard !Task.isCancelled else { return }
            state = .failed(message: error.localizedDescription, retryable: true)
        }
    }

    /// What the page did. The only route from the web view into this model.
    func pageEvent(_ event: WorldRenderPageEvent) {
        renderWatchdog?.cancel()
        renderWatchdog = nil
        switch event {
        case .rendered:
            // Only from `.rendering`. A `didFinish` that arrives after a
            // timeout already reported failure must not quietly un-fail the
            // screen underneath a reader who has started reading the message.
            guard case .rendering(let html) = state else { return }
            state = .ready(html: html)
            if fallback != nil, let rung = state.representation {
                // A refresh to this rung DREW. Whatever failed it before has
                // passed; a later failure starts counting again.
                rungDrawFailures[rung] = nil
                refusedRungs.remove(rung)
                newerPictureRefused = false
            }
            fallback = nil
        case .reloadingAfterTermination, .reloadingItself:
            // Back to a bounded wait, with the overlay over it. Truthful: the
            // page really is being drawn again. The two cases differ only in
            // who asked — WebKit's content process died, or the page gave up on
            // a WebGL context it was never given back — and from the reader's
            // side they are the same wait.
            guard let html = state.html else { return }
            state = .rendering(html: html)
            startRenderWatchdog()
        case .failed(let detail):
            if revertRefresh() { return }
            state = .failed(
                message: "The Tower's page arrived but could not be drawn: \(detail)",
                retryable: true
            )
        case .gaveUpAfterTerminations(let count):
            if revertRefresh() { return }
            // Retryable, and the sentence says why it stopped on its own.
            // Memory pressure is transient — closing another app can genuinely
            // change the answer — so this is not a control that cannot work.
            // What it must not do is keep reloading a page that keeps killing
            // the process, which is what it did before there was a budget.
            state = .failed(
                message: "This world was too large to draw on this phone: the page ran out of "
                    + "memory \(count) times and was not reloaded again.",
                retryable: true
            )
        }
    }

    /// Fail truthfully if the page never reports finishing -- or, when the page
    /// that did not finish was an automatic refresh, put the previous one back.
    ///
    /// The message says the fetch succeeded, because it did — blaming the Tower
    /// for a page it delivered would send the reader to the wrong machine.
    private func startRenderWatchdog() {
        let timeout = renderTimeout
        renderWatchdog = Task { [weak self] in
            try? await Task.sleep(for: timeout)
            guard !Task.isCancelled, let self, self.state.isRendering else { return }
            if self.revertRefresh() { return }
            self.state = .failed(
                message: "The Tower's page arrived but did not finish drawing. This world may be "
                    + "too large to render on this phone.",
                retryable: true
            )
        }
    }
}

// MARK: - The web view

/// Decides what the web view may navigate to: the page it was handed, and
/// nothing else. Pure, so it is tested without a `WKWebView`.
///
/// The page is loaded from the app's private scheme
/// (`WorldAssetScheme.pageURL`), whose handler serves the string
/// `WorldRenderClient` fetched, so its only navigation is the initial one, to
/// exactly that URL. Anything else — a link the page does not contain today, a
/// `window.location` a future page might set, `about:blank`, another world's
/// page, the same page with a query — is refused, and so is a *second*
/// navigation to the page URL, which would reload it and reset the camera.
///
/// This governs navigations. Subresource loads — a `fetch`, an `<img src>`,
/// a `<script src>`, a WebSocket — are not navigations and are not seen
/// here. They are governed twice: the page's CSP allows `connect-src
/// glasses-world:` and nothing else, and the scheme handler serves only the
/// whitelist in `WorldAssetRequest`.
///
/// **One exception: the page reloading itself** (`isReload`, a main-frame
/// `.reload` of exactly the page URL). The appearance page asks for that when
/// WebKit never gives back a lost WebGL context -- it often does not -- rather
/// than sit on "Restoring…" for good (review 1, m8). The reload is served the
/// same string from memory by the scheme handler, costs the wearer's camera,
/// and goes nowhere else.
nonisolated enum WorldRenderNavigationPolicy {
    static func allows(_ url: URL?, isInitialLoad: Bool, pageURL: URL?, isReload: Bool = false) -> Bool {
        guard isInitialLoad || isReload, let url, let pageURL else { return false }
        return url.absoluteString == pageURL.absoluteString
    }
}

/// A `WKWebView` that shows one page and goes nowhere.
///
/// The page is the string the model fetched, served by a
/// `WorldAssetSchemeHandler` at `glasses-world://tower/worlds/<world>/render`,
/// in a web view with a NON-PERSISTENT website data store: nothing the page
/// does -- its fetched imagery, storage, caches -- outlives the viewer.
struct WorldRenderWebView: UIViewRepresentable {
    /// Which world's routes the page may reach through the scheme.
    let target: WorldRenderTarget
    let html: String
    /// Serves the page and proxies this world's routes. Owned by the model, so
    /// the imagery it holds outlives this web view and dies with the viewer;
    /// see `WorldRenderViewerModel.assets`.
    let assets: WorldAssetSchemeHandler
    /// Which `load()` this page belongs to. A retry of the **same** page after
    /// a render failure carries a new number, which is what makes the reload
    /// happen at all; see `WorldRenderViewerModel.renderAttempt`.
    var attempt: Int = 0
    /// Which content-process-kill budget this page is under. A new value is a
    /// fresh budget; see `WorldRenderViewerModel.renderBudgetToken`.
    var budgetToken: Int = 0
    /// What the page did. No cycle: the coordinator holds this closure, the
    /// closure holds the model, and the model holds neither.
    ///
    /// `@MainActor` on the closure type, matching `decidePolicyFor`'s own
    /// handler in this file: every `WKNavigationDelegate` callback arrives on
    /// the main thread, and the model it calls is `@MainActor`.
    var onEvent: (@MainActor (WorldRenderPageEvent) -> Void)? = nil

    func makeCoordinator() -> Coordinator { Coordinator(target: target, assets: assets) }

    /// The configuration every viewer web view is built with. Split out so
    /// the data store and the scheme registration are tested without a view.
    static func makeConfiguration(assets: WorldAssetSchemeHandler) -> WKWebViewConfiguration {
        let configuration = WKWebViewConfiguration()
        // Privacy §3.6: the default store is persistent; this one is memory only.
        configuration.websiteDataStore = .nonPersistent()
        // Must be registered before the web view exists.
        configuration.setURLSchemeHandler(assets, forURLScheme: WorldAssetScheme.name)
        // The page needs its script and nothing else: no media, no data
        // detectors turning a coordinate into a phone number.
        configuration.dataDetectorTypes = []
        configuration.allowsInlineMediaPlayback = false
        return configuration
    }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = Self.makeConfiguration(assets: context.coordinator.assets)
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.allowsBackForwardNavigationGestures = false
        webView.allowsLinkPreview = false
        webView.isOpaque = false
        webView.backgroundColor = .black
        // The page draws on a canvas that owns every touch (`touch-action:
        // none`) and sets `overflow: hidden`; the scroll view underneath it
        // would otherwise bounce on the first drag and steal the orbit.
        webView.scrollView.isScrollEnabled = false
        webView.scrollView.bounces = false
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        // Assigned before the load, so an event from a page that fails
        // immediately still has somewhere to go.
        context.coordinator.onEvent = onEvent
        // A previous web view for this viewer may have been dismantled; the
        // handler is the model's and outlives both.
        assets.attach()
        context.coordinator.load(html, attempt: attempt, budget: budgetToken, into: webView)
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // Re-assigned on every update: this struct is rebuilt on every parent
        // render and the closure it carries captures the current model, while
        // the coordinator persists across all of them.
        context.coordinator.onEvent = onEvent
        context.coordinator.load(html, attempt: attempt, budget: budgetToken, into: webView)
    }

    /// The web view is going away. There was **no teardown hook at all** before
    /// review 2 (M-3): no `stopLoading()`, no delegate cleared, no scheme task
    /// cancelled, nothing dropped. The last of those is the dangerous one --
    /// a `WKURLSchemeTask` answered after its web view is gone raises
    /// `NSInternalInconsistencyException`, which Swift cannot catch -- and it
    /// composes with the page's own worst case: the page has no fetch timeout,
    /// so a task the handler silently dropped left the page waiting forever on
    /// "Restoring…" (page review, P-1). Both ends are fixed; this is the end
    /// the app owns.
    ///
    /// `static`, as `UIViewRepresentable` requires. The imagery is NOT dropped
    /// here: the viewer may be about to build another web view (a "Try again",
    /// a rung swap), and the copy belongs to the viewer. `viewerClosed()` drops
    /// it.
    static func dismantleUIView(_ webView: WKWebView, coordinator: Coordinator) {
        webView.stopLoading()
        webView.navigationDelegate = nil
        coordinator.onEvent = nil
        coordinator.assets.detach()
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        let target: WorldRenderTarget
        /// Serves the page and proxies this world's whitelisted routes. The
        /// MODEL owns it (`WorldRenderViewerModel.assets`) and it outlives any
        /// one web view; this is a reference, and the configuration retains it
        /// too for as long as the web view exists.
        let assets: WorldAssetSchemeHandler
        /// The one URL this web view may navigate to.
        let pageURL: URL?

        init(target: WorldRenderTarget, assets: WorldAssetSchemeHandler) {
            self.target = target
            self.assets = assets
            // The room's page, or an area's (`WORLD-BUILDER-COMPONENTS.md`
            // §5.5): the navigation policy admits exactly this one URL.
            self.pageURL = WorldAssetScheme.pageURL(worldID: target.worldID, scope: assets.scope)
            super.init()
        }

        /// The string on screen, so `updateUIView` — called on every parent
        /// re-render — reloads only when the page actually changed.
        private var loaded: String?
        /// Which attempt that string belongs to. Without it, "Try again" after
        /// a render failure is a no-op: the html is identical, `loaded != html`
        /// is false, and nothing reloads.
        private var loadedAttempt: Int?
        /// Which budget token the string on screen carries; a change is the
        /// only thing that resets the kill budget.
        private var loadedBudgetToken: Int?
        /// Whether the one allowed navigation has been decided. Each
        /// `loadHTMLString` is one; the flag is reset when one is issued.
        private var hasDecidedInitialLoad = false
        /// What the page did, back to the model.
        var onEvent: (@MainActor (WorldRenderPageEvent) -> Void)?

        /// How many content-process kills this coordinator will put the page
        /// back after, before it stops and says so.
        ///
        /// `webViewWebContentProcessDidTerminate` used to reload
        /// unconditionally and forever. A page that runs the content process
        /// out of memory on load runs it out of memory on the reload too, so
        /// the honest reading of that loop is: an app that reloads a
        /// multi-megabyte point cloud into a dying process indefinitely, with
        /// nothing on screen and no way for the reader to learn why. Two
        /// retries is enough for a transient kill — another app spiking, a
        /// backgrounded return — and short of a loop.
        ///
        /// Counted over a **sliding window** (`terminationWindow`), **not**
        /// reset by a successful `didFinish`: a page that renders and then
        /// OOMs on the first gesture is the same failure arriving later, and
        /// resetting on `didFinish` would make the budget unbounded again for
        /// exactly that case. A page that kills the process on load kills it
        /// three times well inside a minute and is still stopped.
        ///
        /// It was counted for the life of the coordinator, and that was wrong
        /// once this screen became one a wearer keeps open for a whole walk:
        /// iOS routinely kills a backgrounded app's WebContent process, so
        /// three pocket-locks spread over twenty minutes read "too large to
        /// draw on this phone" over a world that drew fine each time.
        ///
        /// A deliberate retry resets it, because that is the reader asking for
        /// a fresh budget with full knowledge; see `load(_:attempt:into:)`.
        private static let reloadBudget = 2
        nonisolated static let terminationWindow: TimeInterval = 60
        private var terminationTimes: [Date] = []

        /// The terminations that still count at `now`: those less than
        /// `window` seconds old. Pure, so the window is tested without a
        /// `WKWebView`.
        nonisolated static func recentTerminations(
            _ times: [Date], now: Date, window: TimeInterval
        ) -> [Date] {
            times.filter { now.timeIntervalSince($0) < window }
        }

        func load(_ html: String, attempt: Int, budget: Int, into webView: WKWebView) {
            guard loaded != html || loadedAttempt != attempt else { return }
            loaded = html
            if loadedBudgetToken != budget {
                // The reader asking again, knowingly, or a page that already
                // drew once on this phone being put back. Fresh budget. A
                // forced reload of an identical string is NOT that, and carries
                // the same token (review 2, m-1).
                terminationTimes = []
                selfReloads = []
            }
            loadedAttempt = attempt
            loadedBudgetToken = budget
            reload(into: webView)
        }

        private func reload(into webView: WKWebView) {
            guard let html = loaded else { return }
            guard let pageURL else {
                // Asynchronously: `reload` is reached synchronously from
                // `makeUIView`, so reporting from here would set `@Published`
                // state inside SwiftUI's own view-construction pass ("Modifying
                // state during view update…"). Unreachable in practice -- an id
                // that will not percent-encode already fails in
                // `WorldRenderClient.url` -- but the shape was wrong (review 2,
                // m-6).
                let report = onEvent
                Task { @MainActor in
                    report?(.failed("this world's identifier could not be made into a page address"))
                }
                return
            }
            hasDecidedInitialLoad = false
            // The handler serves THIS string for the page URL, and the session
            // the page draws bounds which appearance routes it may reach. The
            // page's own fetches follow new appearance builds without a reload.
            let session = WorldAssetSchemeHandler.session(for: target, page: html)
            if session != assets.sessionID {
                // A different walk's page. `WorldAssetMemory.entries` is keyed
                // by digest alone and `authorizedAt` was earned by the OTHER
                // session's manifest, so carrying either over would let one
                // session's authorisation answer for another's bytes. Nothing
                // wrong has ever been shown -- a hit needs two sessions to have
                // byte-identical content-addressed chunks -- but the invariant
                // was enforced by luck (review 2, m-4).
                assets.dropCache()
            }
            assets.pageHTML = html
            assets.sessionID = session
            webView.load(URLRequest(url: pageURL))
        }

        /// How many times the PAGE may put itself back, within the same window
        /// as the kill budget.
        ///
        /// The appearance page reloads itself when WebKit never gives back a
        /// lost WebGL context (review 1, m8). Two things about that navigation
        /// are not in WebKit's gift to promise: that a script-initiated reload
        /// arrives as `.reload` rather than `.other` (it maps from
        /// `FrameLoadType::Reload`, which *should*, and the SDK does not say
        /// so), and that it arrives at all. Cancelling it because the type was
        /// the other one leaves the wearer on "Restoring…" for good, which is
        /// the exact failure the reload exists to prevent (review 2, m-8).
        ///
        /// So both types are accepted for a main-frame navigation to **exactly**
        /// the page URL after the initial load -- and counted, because "accept
        /// `.other` to the page URL" without a bound is an invitation to a
        /// reload loop that nothing else would stop. Three inside a minute is
        /// well past any real recovery.
        private static let selfReloadBudget = 3
        private var selfReloads: [Date] = []

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void
        ) {
            let isMainFrame = navigationAction.targetFrame?.isMainFrame == true
            let isInitialLoad = !hasDecidedInitialLoad
                && navigationAction.navigationType == .other
                && isMainFrame
            let now = Date()
            let couldBeSelfReload = hasDecidedInitialLoad && isMainFrame
                && (navigationAction.navigationType == .reload
                    || navigationAction.navigationType == .other)
            let recent = Self.recentTerminations(selfReloads, now: now, window: Self.terminationWindow)
            let isReload = couldBeSelfReload && recent.count < Self.selfReloadBudget
            let allowed = WorldRenderNavigationPolicy.allows(
                navigationAction.request.url, isInitialLoad: isInitialLoad, pageURL: pageURL,
                isReload: isReload
            )
            if allowed {
                hasDecidedInitialLoad = true
                if isReload {
                    selfReloads = recent + [now]
                    // The page on screen is gone until this finishes, so the
                    // screen goes back to a bounded wait with the overlay over
                    // it rather than claiming to be ready over a black
                    // rectangle (review 2, m-9).
                    onEvent?(.reloadingItself)
                }
            }
            decisionHandler(allowed ? .allow : .cancel)
        }

        /// The page loaded and its script ran. The one signal that says the
        /// black rectangle became a world.
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            onEvent?(.rendered)
        }

        /// The load failed after it had started.
        func webView(
            _ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error
        ) {
            report(error)
        }

        /// The load failed before it had started. Both are reported the same
        /// way: from the reader's side there is no difference between a page
        /// that never began and one that stopped.
        func webView(
            _ webView: WKWebView,
            didFailProvisionalNavigation navigation: WKNavigation!,
            withError error: Error
        ) {
            report(error)
        }

        /// A failure worth telling the reader about, or nothing.
        ///
        /// **Cancellations are not failures here.** `decidePolicyFor` refuses
        /// every navigation but the first, by design, and a refusal surfaces as
        /// a provisional-navigation failure with `NSURLErrorCancelled` or
        /// WebKit's "frame load interrupted by policy change". Reporting those
        /// would turn the navigation policy doing its job into "the page could
        /// not be drawn" — on a page that is on screen and working.
        ///
        /// The WebKit one is spelled out by hand because the public `WKError`
        /// enum has no case for it: WebKit reports a policy refusal under its
        /// legacy `WebKitErrorDomain` with code 102
        /// (`WebKitErrorFrameLoadInterruptedByPolicyChange`), and neither the
        /// domain string nor the code is exported to Swift on iOS. The first
        /// version of this line named a `WKError.Code.frameLoadInterrupted`
        /// that does not exist, and the branch did not compile until the Mac
        /// gate caught it.
        nonisolated static let webKitLegacyErrorDomain = "WebKitErrorDomain"
        nonisolated static let webKitFrameLoadInterruptedByPolicyChange = 102

        /// Whether a navigation error is the policy refusing a navigation
        /// (not a failure) rather than the page failing to load (a failure).
        nonisolated static func isNavigationCancellation(_ error: Error) -> Bool {
            let ns = error as NSError
            return (ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled)
                || (ns.domain == webKitLegacyErrorDomain
                    && ns.code == webKitFrameLoadInterruptedByPolicyChange)
        }

        private func report(_ error: Error) {
            guard !Self.isNavigationCancellation(error) else { return }
            onEvent?(.failed((error as NSError).localizedDescription))
        }

        /// WebKit's content process was killed — under memory pressure, a
        /// multi-megabyte page redrawing 80k points on every gesture is a
        /// candidate — and the view is now blank. The string is still here,
        /// so put it back rather than leaving the reader a black rectangle
        /// with nothing to do but Close and reopen.
        ///
        /// **Up to `reloadBudget` times within `terminationWindow`.** Past that
        /// the page is not put back and the reader is told, because a page
        /// that kills the process on load kills it again on the reload, and an
        /// unbounded loop of that is a blank screen plus a warm phone.
        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            let now = Date()
            terminationTimes = Self.recentTerminations(
                terminationTimes, now: now, window: Self.terminationWindow
            )
            terminationTimes.append(now)
            let terminations = terminationTimes.count
            guard terminations <= Self.reloadBudget else {
                onEvent?(.gaveUpAfterTerminations(terminations))
                return
            }
            // Announced before the reload, so the screen is back in a bounded
            // `.rendering` before the page has a chance to die again silently.
            onEvent?(.reloadingAfterTermination)
            reload(into: webView)
        }
    }
}

// MARK: - The 3D world on screen

/// The sheet form: a navigation stack of its own, and a Close that dismisses
/// it. Presented from the World Builder workspace, which has no stack to push
/// onto.
///
/// Split from `WorldRenderScene` because the saved-worlds picker **pushes** the
/// same screen onto its own `NavigationStack`, and a `NavigationStack` nested
/// inside another one breaks the back gesture, the title and the toolbar. The
/// scene owns the model and the content; this owns the presentation.
struct WorldRenderViewerView: View {
    @Environment(\.dismiss) private var dismiss

    private let target: WorldRenderTarget
    private let title: String?
    private let note: String?
    /// The walk's `finalization.notice`, for the room (v6).
    private let notice: String?
    private let client: WorldRenderClient

    init(
        target: WorldRenderTarget,
        title: String? = nil,
        note: String? = nil,
        notice: String? = nil,
        client: WorldRenderClient = WorldRenderClient()
    ) {
        self.target = target
        self.title = title
        self.note = note
        self.notice = notice
        self.client = client
    }

    /// What the sheet shows now: the room it was opened for, or one of its
    /// areas. Opening an area REPLACES the room's scene -- its own model, web
    /// view and WebGL context go with it -- and *Back to the room* replaces it
    /// again (`WORLD-BUILDER-COMPONENTS.md` §5.5, C1 E5). Never both at once.
    @State private var shown: WorldRenderTarget?
    @State private var shownArea: WorldAreaOpening?

    var body: some View {
        let current = shown ?? target
        NavigationStack {
            WorldRenderScene(
                target: current,
                title: current.isArea ? nil : title,
                note: current.isArea ? nil : note,
                client: client,
                notice: current.isArea ? nil : notice,
                area: current.isArea ? shownArea : nil,
                openArea: { area, opening in
                    shownArea = opening
                    shown = area
                },
                backToRoom: current.isArea ? {
                    shownArea = nil
                    shown = target.room
                } : nil
            )
            // A new identity per target: SwiftUI builds a new scene (and a new
            // `@StateObject` model) rather than handing the room's model the
            // area's target.
            .id(current.id)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
    }
}

/// The 3D world itself, without a stack around it. **The primary way a saved
/// world is seen.**
///
/// It fetches the Tower's composed page as it appears, shows the Tower's own
/// sentence when there is none, and hands the page to a web view that can go
/// nowhere else.
///
/// ## Why the ids left the caption
///
/// The caption used to read `World fcbca9e90b244785bdb671530b33c6a5 · session
/// 8f2c…` in a monospaced font, over the sentence "Sparse structure-from-motion
/// output: feature points and camera poses." Both are true and neither belongs
/// on the first screen a person sees after opening a saved world: one is a
/// database key and the other is the name of an algorithm. The ids moved into
/// Details, at the bottom, and nothing was deleted.
///
/// Details holds **only** the ids. A segmented control for the two renderings
/// briefly lived there too and was removed: it refetched a multi-megabyte page
/// to change one token, and the Tower's page already carries an instant toggle
/// of its own. See `details`.
///
/// The caption that replaced them still refuses the one claim this screen must
/// not make — that a point cloud is a scan of the room — and does it in words
/// that are true of both of the Tower's renderings.
struct WorldRenderScene: View {
    @StateObject private var model: WorldRenderViewerModel

    /// What to call this world on screen: the Tower's display name, or a dated
    /// title from the picker. `nil` falls back to "Saved world" — never to the
    /// id, which is what the id being in Details is for.
    private let title: String?
    /// The ladder's note for a world that is not final, or `nil`. Composed by
    /// `WorldReconstruction`, not here.
    private let note: String?

    /// Whether the Details disclosure is open. View state, and deliberately
    /// not remembered across presentations: Details is where a person goes on
    /// purpose, and a sheet that opens with it expanded is back to being a
    /// debugger.
    @State private var isShowingDetails = false

    /// `finalization.notice` for the walk this room belongs to (v6, §8): what
    /// the Tower could not do and who can fix it, shown verbatim below the
    /// room caption as a plain note. `nil` shows nothing -- every older world,
    /// and every walk with nothing owed -- and an area viewer never shows it.
    private let notice: String?
    /// For an area viewer: its number, the walk's area count and its capture
    /// spans, from the list it was opened from (C1 M7). `nil` for the room.
    private let area: WorldAreaOpening?
    /// Opens one of this room's areas in place of the room (C1 E5). `nil`
    /// where the host cannot, and then the areas row offers no control.
    private let openArea: ((WorldRenderTarget, WorldAreaOpening) -> Void)?
    /// *Back to the room*, for an area viewer.
    private let backToRoom: (() -> Void)?

    init(
        target: WorldRenderTarget,
        title: String? = nil,
        note: String? = nil,
        client: WorldRenderClient = WorldRenderClient(),
        components: WorldComponents? = nil,
        notice: String? = nil,
        area: WorldAreaOpening? = nil,
        openArea: ((WorldRenderTarget, WorldAreaOpening) -> Void)? = nil,
        backToRoom: (() -> Void)? = nil
    ) {
        _model = StateObject(wrappedValue: WorldRenderViewerModel(
            target: target, client: client, components: components))
        self.title = title
        self.note = note
        // Guarded (G1-F3): machine output never reaches the wearer verbatim.
        self.notice = target.isArea ? nil : WorldNoticeGuard.displayText(notice)
        self.area = area
        self.openArea = openArea
        self.backToRoom = backToRoom
    }

    /// The area's header, or the host's title, or "Saved world".
    private var screenTitle: String {
        if model.target.isArea { return area?.header ?? "Area — not placed in the room" }
        return title ?? "Saved world"
    }

    var body: some View {
        VStack(spacing: 0) {
            caption
            content
            if !model.target.isArea { areasRow }
            details
        }
        .navigationTitle(screenTitle)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if model.target.isArea, let area {
                ToolbarItem(placement: .principal) {
                    VStack(spacing: 0) {
                        Text(area.numberLine).font(.headline)
                        Text(WorldAreaOpening.notPlacedLine).font(.caption).foregroundStyle(.secondary)
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel(area.header)
                }
            }
        }
        // One fetch, for the life of the screen. `model.target` is a `let`, so
        // there is nothing for a `.task(id:)` to key on; a second rendering of
        // the same world is a second screen, and the page's own button switches
        // in place without fetching at all.
        .task {
            await model.load()
            await model.refreshComponents()
            await model.followRevisions()
        }
        // `PRIVACY.md` §3.6: in-memory reuse within one open viewer is fine;
        // drop it when the viewer closes. Nothing did (review 2, M-4).
        .onDisappear { model.viewerClosed() }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                // ## Why there is always a way to ask again
                //
                // Every other control on this screen is conditional on the
                // model's state, and the state that needed one most had none:
                // the page reports `didFinish`, so the screen is `.ready`, the
                // failure view and its "Try again" are not shown -- and the
                // page underneath is showing its own sentence, because it
                // placed no imagery, or its fetches were all refused, or it
                // boots into a world whose appearance is withdrawn. The wearer
                // was left with a message and a Close (review 2, M-0 and m-7).
                //
                // This is not a second "Try again": it is the same `load()`,
                // reachable whatever the screen is showing. It is deliberately
                // plain and in the toolbar rather than under the caption, so it
                // is never mistaken for a claim about the picture.
                Button("Reload") { Task { await model.load() } }
                    .disabled(model.state.isFetching)
                    .accessibilityIdentifier("world-render-reload")
            }
        }
    }

    private var caption: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let note {
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // Follows the page. "Not a surface" was load-bearing while every
            // page was points; the Tower can now serve a surface, and a caption
            // that denies what is on screen is as wrong as one that overclaims.
            // Before the page arrives it claims neither.
            Text(nativeCaption)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            // The walk's notice (v6, §8): verbatim, below the room caption, a
            // plain note and not an error -- no icon, no tint, no control.
            if let notice {
                Text(notice)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("world-render-notice")
            }
            if model.target.isArea {
                areaControls
            }
            // A rebuild of the rung on screen is offered, never forced: the
            // swap reloads the page and resets the camera the reader is using.
            // A better rung replaces the picture without asking.
            if model.newerPictureAvailable {
                Button("A newer reconstruction is ready. Show it") {
                    Task { await model.showNewerPicture() }
                }
                .font(.caption)
                .accessibilityIdentifier("world-render-newer-picture")
            }
            // After a refresh could not be drawn the old picture is back and
            // the screen is `.ready`, so the failure view's "Try again" is not
            // there. This is that control: `load()` forgets every refusal.
            if model.newerPictureRefused {
                Button("A newer reconstruction could not be drawn on this phone. Try again") {
                    Task { await model.load() }
                }
                .font(.caption)
                .accessibilityIdentifier("world-render-retry-refused")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    /// The caption line under the title. For an area: that it is an area, when
    /// it was captured, that it is not placed, and not to scale
    /// (`WORLD-BUILDER-COMPONENTS.md` §5.4). For the room: the rung's caption,
    /// plus ` · N more areas shown separately` when the walk has areas (§4).
    private var nativeCaption: String {
        if model.target.isArea {
            return WorldComponentsPresentation.areaCaption(
                representation: model.state.representation, spans: area?.spans ?? [])
        }
        return WorldRenderRepresentation.caption(for: model.state.representation)
            + (WorldComponentsPresentation.roomCaptionSuffix(model.components) ?? "")
    }

    /// *Back to the room*, and -- when the Tower has said this area will not be
    /// served (§5.2) -- why the picture stopped following.
    @ViewBuilder
    private var areaControls: some View {
        if model.areaNoLongerServed {
            Text("The Tower no longer serves this area -- the walk may have been finished again. "
                 + "The room's list of areas is current.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        if let backToRoom {
            Button("Back to the room") { backToRoom() }
                .font(.caption)
                .accessibilityIdentifier("world-render-back-to-room")
        }
    }

    /// Below the room: the walk's areas, and the stretches it can only count
    /// (§8). Nothing at all for `components: null` -- every older world -- and
    /// nothing for a walk the gate kept whole, so both look exactly as today.
    @ViewBuilder
    private var areasRow: some View {
        if let components = model.components, !components.isWhole {
            VStack(alignment: .leading, spacing: 6) {
                if let heading = WorldComponentsPresentation.areasHeading(components) {
                    Text(heading)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                let areas = components.areas
                ForEach(Array(areas.enumerated()), id: \.element.id) { index, area in
                    areaEntry(number: index + 1, of: areas.count, area: area)
                }
                if let footer = WorldComponentsPresentation.footer(components) {
                    Text(footer)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .accessibilityIdentifier("world-render-areas")
        }
    }

    @ViewBuilder
    private func areaEntry(number: Int, of total: Int, area: WorldComponent) -> some View {
        let line = WorldComponentsPresentation.areaLine(number: number, area: area)
        let availability = WorldComponentsPresentation.availability(of: area)
        if availability == .opens, let openArea {
            Button {
                openArea(model.target.area(area.id),
                         WorldAreaOpening(number: number, total: total, spans: area.captureSpans))
            } label: {
                HStack {
                    Text(line).font(.footnote)
                    Spacer()
                    Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary)
                }
            }
            .buttonStyle(.plain)
        } else {
            HStack {
                Text(line).font(.footnote).foregroundStyle(.secondary)
                Spacer()
                if let word = WorldComponentsPresentation.availabilityWord(availability) {
                    Text(word).font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }

    /// The page, the two waits, and the failure.
    ///
    /// ## Why this is not a `switch`
    ///
    /// `.rendering` and `.ready` hold the same page, and the web view must be
    /// the **same view** across both: separate arms of a `switch` inside a
    /// `ViewBuilder` are separate `_ConditionalContent` branches, so SwiftUI
    /// would destroy the `WKWebView` and build a new one at the instant the
    /// first one finished loading — discarding the render, and starting a
    /// second one that would finish and be discarded in turn. One `if let` over
    /// `state.html` keeps one web view for the life of the page.
    @ViewBuilder
    private var content: some View {
        if let html = model.state.html {
            WorldRenderWebView(target: model.target, html: html, assets: model.assets,
                               attempt: model.renderAttempt, budgetToken: model.renderBudgetToken,
                               onEvent: model.pageEvent)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .overlay { renderingOverlay }
        } else if model.state.failureMessage != nil {
            failureView
        } else {
            waiting("Fetching this world from the Tower…")
        }
    }

    /// Over the web view while the page has not reported finishing.
    ///
    /// Opaque on purpose: underneath it is a black rectangle, and a
    /// half-transparent spinner over black reads as a stuck page rather than as
    /// a page arriving. It disappears on `didFinish`, and if that never comes
    /// the watchdog replaces the whole thing with a sentence.
    @ViewBuilder
    private var renderingOverlay: some View {
        if model.state.isRendering {
            waiting("Drawing the world…")
                .background(Color(.systemBackground))
        }
    }

    /// The two waits, worded for what is actually happening.
    ///
    /// The fetching sentence read "Building the 3D world…", which described
    /// neither half: the Tower composed the page before this screen existed,
    /// and what this app is doing is downloading it. Then drawing it. Those are
    /// different waits, they fail differently, and they now say so.
    private func waiting(_ sentence: String) -> some View {
        VStack(spacing: 12) {
            ProgressView()
            Text(sentence)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var failureView: some View {
        VStack(spacing: 12) {
            Image(systemName: "cube.transparent")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text(model.state.failureMessage ?? "")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
            if model.state.failureIsRetryable {
                Button("Try again") {
                    Task { await model.load() }
                }
                .buttonStyle(.bordered)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// The identifiers and the other rendering, both one tap away and neither
    /// on the way to anything.
    ///
    /// `.ignoresSafeArea` came off the web view above when this arrived: a
    /// disclosure underneath a view that draws into the home-indicator area
    /// would be drawn over by it.
    private var details: some View {
        DisclosureGroup("Details", isExpanded: $isShowingDetails) {
            VStack(alignment: .leading, spacing: 8) {
                Text((model.target.sessionID.map { "World \(model.target.worldID)\nSession \($0)" }
                      ?? "World \(model.target.worldID)\nNewest session with geometry")
                     + (model.target.areaID.map { "\nArea \($0)" } ?? ""))
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)

                // ## There is no rendering switch here, deliberately
                //
                // A segmented control stood here, bound to `model.show(_:)`.
                // It worked and it was the wrong control, for two independent
                // reasons that a review found before a wearer did.
                //
                // **It refetched the world to change one token.** Switching the
                // view changes `WorldRenderTarget.id`, which restarts
                // `.task(id:)`, which pulls the whole page again — several
                // megabytes of point coordinates over Tailscale — to flip a
                // single `let mode` in a script that is already on screen.
                //
                // **The page already has the control, and it is better.** The
                // Tower's page carries `<button id="mode">` and a `d` key
                // binding (`toggleMode()` in
                // `tower/world_builder/render.py`), both of which switch in
                // place with no request at all, reset the camera, and relabel
                // themselves. It is on screen, above this disclosure, in front
                // of the reader. Duplicating it worse, one level down, in a
                // collapsed section, is not an affordance — it is a second
                // way to do the same thing that costs a download.
                //
                // `WorldRenderView` and the `?view=diagnostics` parameter stay
                // and are still used: they choose which rendering a page OPENS
                // in, which is a fetch that was going to happen anyway. The
                // world screen's Diagnostics section uses that, so a reader who
                // wants the solver's view gets it without first loading the
                // one they did not ask for.
                //
                // Only the SPARSE page has that button. A surface or dense
                // page has none, and a sentence pointing "above" at a button
                // that is not there is the false copy an iOS review found. A
                // page with no declared rung is from a Tower older than the
                // ladder, whose only page was the sparse one.
                if model.state.representation == .sparse || model.state.representation == nil {
                    Text("The page's own Diagnostics button, above, switches to the solver's view of this session without fetching anything. The solver's geometry is also drawn in full under Diagnostics on the world screen.")
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("The solver's own view of this session is under Diagnostics on the world screen.")
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 4)
        }
        .font(.footnote)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }
}
