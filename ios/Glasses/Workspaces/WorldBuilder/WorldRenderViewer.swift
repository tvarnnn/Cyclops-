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
    /// WebKit's content process was killed this many times and the web view
    /// has stopped putting the page back. See `Coordinator.reloadBudget`.
    case gaveUpAfterTerminations(Int)
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

    /// Bounds `.rendering`. Cancelled when the page reports either way.
    ///
    /// Unstructured (`Task { }`), so it does not die with the SwiftUI `.task`
    /// that started the fetch; `[weak self]` so a dismissed sheet is not kept
    /// alive for the length of the timeout.
    private var renderWatchdog: Task<Void, Never>?

    init(target: WorldRenderTarget, client: WorldRenderClient = WorldRenderClient()) {
        self.target = target
        self.client = client
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
        state = .fetching
        do {
            let html = try await client.page(for: target)
            guard !Task.isCancelled else { return }
            // Not `.ready`. The page has arrived; nothing has drawn it yet.
            state = .rendering(html: html)
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
        case .reloadingAfterTermination:
            // Back to a bounded wait, with the overlay over it. Truthful: the
            // page really is being drawn again.
            guard let html = state.html else { return }
            state = .rendering(html: html)
            startRenderWatchdog()
        case .failed(let detail):
            state = .failed(
                message: "The Tower's page arrived but could not be drawn: \(detail)",
                retryable: true
            )
        case .gaveUpAfterTerminations(let count):
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

    /// Fail truthfully if the page never reports finishing.
    ///
    /// The message says the fetch succeeded, because it did — blaming the Tower
    /// for a page it delivered would send the reader to the wrong machine.
    private func startRenderWatchdog() {
        let timeout = renderTimeout
        renderWatchdog = Task { [weak self] in
            try? await Task.sleep(for: timeout)
            guard !Task.isCancelled, let self, self.state.isRendering else { return }
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
    /// Which `load()` this page belongs to. A retry of the **same** page after
    /// a render failure carries a new number, which is what makes the reload
    /// happen at all; see `WorldRenderViewerModel.renderAttempt`.
    var attempt: Int = 0
    /// What the page did. No cycle: the coordinator holds this closure, the
    /// closure holds the model, and the model holds neither.
    ///
    /// `@MainActor` on the closure type, matching `decidePolicyFor`'s own
    /// handler in this file: every `WKNavigationDelegate` callback arrives on
    /// the main thread, and the model it calls is `@MainActor`.
    var onEvent: (@MainActor (WorldRenderPageEvent) -> Void)? = nil

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
        // Assigned before the load, so an event from a page that fails
        // immediately still has somewhere to go.
        context.coordinator.onEvent = onEvent
        context.coordinator.load(html, attempt: attempt, into: webView)
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // Re-assigned on every update: this struct is rebuilt on every parent
        // render and the closure it carries captures the current model, while
        // the coordinator persists across all of them.
        context.coordinator.onEvent = onEvent
        context.coordinator.load(html, attempt: attempt, into: webView)
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        /// The string on screen, so `updateUIView` — called on every parent
        /// re-render — reloads only when the page actually changed.
        private var loaded: String?
        /// Which attempt that string belongs to. Without it, "Try again" after
        /// a render failure is a no-op: the html is identical, `loaded != html`
        /// is false, and nothing reloads.
        private var loadedAttempt: Int?
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
        /// Counted for the life of this coordinator, **not** reset by a
        /// successful `didFinish`: a page that renders and then OOMs on the
        /// first gesture is the same failure arriving later, and resetting
        /// would make the budget unbounded again for exactly that case. A
        /// deliberate retry does reset it, because that is the reader asking
        /// for a fresh budget with full knowledge; see `load(_:attempt:into:)`.
        private static let reloadBudget = 2
        private var terminations = 0

        func load(_ html: String, attempt: Int, into webView: WKWebView) {
            guard loaded != html || loadedAttempt != attempt else { return }
            loaded = html
            if loadedAttempt != attempt {
                // A new attempt is the reader asking again, knowingly. Fresh
                // budget.
                terminations = 0
            }
            loadedAttempt = attempt
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
        /// **Up to `reloadBudget` times.** Past that the page is not put back
        /// and the reader is told, because a page that kills the process on
        /// load kills it again on the reload, and an unbounded loop of that is
        /// a blank screen plus a warm phone.
        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            terminations += 1
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
            WorldRenderWebView(html: html, attempt: model.renderAttempt, onEvent: model.pageEvent)
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
