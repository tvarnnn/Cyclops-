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

    /// Includes the view, so `.sheet(item:)` re-presents when the reader asks
    /// for the other rendering of the same world rather than treating it as
    /// the same item.
    var id: String { "\(worldID)/\(sessionID ?? "")/\(view.rawValue)" }

    /// The same world and session, in the other rendering.
    func showing(_ view: WorldRenderView) -> WorldRenderTarget {
        WorldRenderTarget(worldID: worldID, sessionID: sessionID, view: view)
    }
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
/// a string with no origin and nothing to navigate to — which is also what
/// makes refusing every other navigation a one-line policy rather than an
/// allow-list.
///
/// Modelled on `WorldListClient`: a struct holding a `URL` and the shared
/// session, defaulted so a test can substitute a stubbed one.
nonisolated struct WorldRenderClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession = .shared
    /// Longer than the list's ten seconds: a page for a full walk is a few
    /// megabytes of point coordinates, and a slow link is not a stalled one
    /// at that size. Still bounded (Rule 15).
    var timeout: TimeInterval = 30

    func page(for target: WorldRenderTarget) async throws -> String {
        guard let url = Self.url(for: target, baseURL: baseURL) else {
            throw WorldRenderFetchError.badAddress
        }
        // `.reloadIgnoringLocalCacheData` for the reason `WorldGeometryClient`
        // gives; the Tower also answers `Cache-Control: no-store`, and a world
        // under construction changes with every build.
        let request = URLRequest(
            url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: timeout
        )
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

    /// The route's address, or `nil` when an id will not go into a URL.
    ///
    /// The id is percent-encoded as **one** path component by hand.
    /// `appendingPathComponent` does not encode `/` — a test showed
    /// `"a/b"` becoming two segments — so a wire-supplied id could otherwise
    /// address a route this client never meant to. Everything outside the
    /// unreserved set is encoded, `/` included.
    static func url(for target: WorldRenderTarget, baseURL: URL) -> URL? {
        guard !target.worldID.isEmpty, target.sessionID != "" else { return nil }
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
        // parameter, so its URL is unchanged from every build before this one
        // and a Tower that has never heard of the parameter is never sent it.
        if target.view != .product {
            query.append(URLQueryItem(name: "view", value: target.view.rawValue))
        }
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

/// What the viewer sheet knows. One of three, and the failure carries its
/// sentence so the view composes nothing.
enum WorldRenderViewerState: Equatable {
    case loading
    case ready(html: String)
    case failed(message: String, retryable: Bool)
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
            return "This Tower does not serve the picture route; it predates WORLD-BUILDER-WORLDS.md §4."
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
    @Published private(set) var state: WorldRenderViewerState = .loading

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

    init(target: WorldRenderTarget, client: WorldRenderClient = WorldRenderClient()) {
        self.target = target
        self.client = client
    }

    func load() async {
        state = .loading
        do {
            state = .ready(html: try await client.page(for: target))
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
}

// MARK: - The web view

/// Decides what the web view may navigate to: the page it was handed, and
/// nothing else. Pure, so it is tested without a `WKWebView`.
///
/// The page is loaded with `loadHTMLString(_:baseURL:nil)`, so its only
/// navigation is the initial one, to `about:blank`. Anything else — a link
/// the page does not contain today, a `window.location` a future page might
/// set, a target the Tower's HTML could carry — is refused, and so is a
/// *second* navigation to `about:blank`, which would blank the page. The
/// page is self-contained by contract (§4), so refusing costs it nothing.
///
/// This governs navigations. Subresource loads — a `fetch`, an `<img src>`,
/// a `<script src>`, a WebSocket — are not navigations and are not seen
/// here; the contract's "no external script, stylesheet, image or fetch"
/// is what keeps them absent, and the page is the Tower's own.
nonisolated enum WorldRenderNavigationPolicy {
    static func allows(_ url: URL?, isInitialLoad: Bool) -> Bool {
        guard isInitialLoad, let url else { return false }
        return url.scheme == "about" && (url.absoluteString == "about:blank")
    }
}

/// A `WKWebView` that shows one string and goes nowhere.
struct WorldRenderWebView: UIViewRepresentable {
    let html: String

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        // The page needs its script and nothing else: no media, no data
        // detectors turning a coordinate into a phone number.
        configuration.dataDetectorTypes = []
        configuration.allowsInlineMediaPlayback = false
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
        context.coordinator.load(html, into: webView)
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        context.coordinator.load(html, into: webView)
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        /// The string on screen, so `updateUIView` — called on every parent
        /// re-render — reloads only when the page actually changed.
        private var loaded: String?
        /// Whether the one allowed navigation has been decided. Each
        /// `loadHTMLString` is one; the flag is reset when one is issued.
        private var hasDecidedInitialLoad = false

        func load(_ html: String, into webView: WKWebView) {
            guard loaded != html else { return }
            loaded = html
            reload(into: webView)
        }

        private func reload(into webView: WKWebView) {
            guard let html = loaded else { return }
            hasDecidedInitialLoad = false
            // No base URL: the page has no origin, can name no relative
            // resource, and its one navigation is to `about:blank`.
            webView.loadHTMLString(html, baseURL: nil)
        }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void
        ) {
            let isInitialLoad = !hasDecidedInitialLoad
                && navigationAction.navigationType == .other
                && navigationAction.targetFrame?.isMainFrame == true
            let allowed = WorldRenderNavigationPolicy.allows(
                navigationAction.request.url, isInitialLoad: isInitialLoad
            )
            if allowed { hasDecidedInitialLoad = true }
            decisionHandler(allowed ? .allow : .cancel)
        }

        /// WebKit's content process was killed — under memory pressure, a
        /// multi-megabyte page redrawing 80k points on every gesture is a
        /// candidate — and the view is now blank. The string is still here,
        /// so put it back rather than leaving the reader a black rectangle
        /// with nothing to do but Close and reopen.
        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
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
    private let client: WorldRenderClient

    init(
        target: WorldRenderTarget,
        title: String? = nil,
        note: String? = nil,
        client: WorldRenderClient = WorldRenderClient()
    ) {
        self.target = target
        self.title = title
        self.note = note
        self.client = client
    }

    var body: some View {
        NavigationStack {
            WorldRenderScene(target: target, title: title, note: note, client: client)
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
/// database key and the other is the name of an algorithm. They moved into
/// Details, at the bottom, along with the switch to the Tower's diagnostics
/// rendering. Nothing was deleted.
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

    init(
        target: WorldRenderTarget,
        title: String? = nil,
        note: String? = nil,
        client: WorldRenderClient = WorldRenderClient()
    ) {
        _model = StateObject(wrappedValue: WorldRenderViewerModel(target: target, client: client))
        self.title = title
        self.note = note
    }

    var body: some View {
        VStack(spacing: 0) {
            caption
            content
            details
        }
        .navigationTitle(title ?? "Saved world")
        .navigationBarTitleDisplayMode(.inline)
        // One fetch, for the life of the screen. `model.target` is a `let`, so
        // there is nothing for a `.task(id:)` to key on; a second rendering of
        // the same world is a second screen, and the page's own button switches
        // in place without fetching at all.
        .task { await model.load() }
    }

    private var caption: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let note {
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // True of the product rendering and of the diagnostics one, and
            // true whether the Tower colours by photometric RGB or by segment
            // index. "Not a surface" is the load-bearing half.
            Text("Points the Tower measured from the walk, with the camera path through them. Not a surface, and not to scale.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private var content: some View {
        switch model.state {
        case .loading:
            VStack(spacing: 12) {
                ProgressView()
                Text("Building the 3D world…")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .ready(let html):
            // Explicitly flexible. `.ignoresSafeArea(edges: .bottom)` came off
            // when Details arrived underneath — a view drawing into the
            // home-indicator area would be drawn over the disclosure — and a
            // `UIViewRepresentable` has no intrinsic size to fall back on.
            WorldRenderWebView(html: html)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .failed(let message, let retryable):
            VStack(spacing: 12) {
                Image(systemName: "cube.transparent")
                    .font(.largeTitle)
                    .foregroundStyle(.secondary)
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                if retryable {
                    Button("Try again") {
                        Task { await model.load() }
                    }
                    .buttonStyle(.bordered)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
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
                Text(model.target.sessionID.map { "World \(model.target.worldID)\nSession \($0)" }
                     ?? "World \(model.target.worldID)\nNewest session with geometry")
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
                Text("The page's own Diagnostics button, above, switches to the solver's view of this session without fetching anything. The solver's geometry is also drawn in full under Diagnostics on the world screen.")
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 4)
        }
        .font(.footnote)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }
}
