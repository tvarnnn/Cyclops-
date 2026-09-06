//
//  WorldRenderViewer.swift
//  Glasses
//

import Combine
import Foundation
import SwiftUI
import WebKit

// MARK: - What is being looked at

/// A world, and optionally one of its sessions, whose viewer page can be
/// asked for: `GET /worlds/{world_id}/render`, `WORLD-BUILDER-WORLDS.md` §4.
///
/// `sessionID` is `nil` when the Tower is to choose — the newest session with
/// geometry — which is what a picker tap on the world's own row means. It is
/// never the empty string.
nonisolated struct WorldRenderTarget: Equatable, Sendable, Identifiable {
    let worldID: String
    let sessionID: String?

    var id: String { "\(worldID)/\(sessionID ?? "")" }
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
        if let sessionID = target.sessionID {
            components.queryItems = [URLQueryItem(name: "session_id", value: sessionID)]
        }
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
        case .absent(let detail?):
            return "The Tower has no picture for this world yet: \(detail)."
        case .absent(nil):
            return "The Tower has no picture for this world yet — no geometry has been built for it, or the world is gone."
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
    /// retryable; a bad id is not.
    var isRetryable: Bool {
        switch self {
        case .badAddress, .undecodable: return false
        case .absent, .towerError, .transport: return true
        }
    }
}

/// Owns one fetch of one page. Created by the sheet and destroyed with it;
/// it holds a string, not a connection, so losing it loses nothing.
@MainActor
final class WorldRenderViewerModel: ObservableObject {
    @Published private(set) var state: WorldRenderViewerState = .loading

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
            state = .failed(message: error.message, retryable: error.isRetryable)
        } catch {
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
/// set, a target the Tower's HTML could carry — is refused. The page is
/// self-contained by contract (§4), so refusing costs it nothing.
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

        func load(_ html: String, into webView: WKWebView) {
            guard loaded != html else { return }
            loaded = html
            // No base URL: the page has no origin, can name no relative
            // resource, and its one navigation is to `about:blank`.
            webView.loadHTMLString(html, baseURL: nil)
        }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void
        ) {
            let isInitialLoad = navigationAction.navigationType == .other
                && navigationAction.targetFrame?.isMainFrame == true
            decisionHandler(
                WorldRenderNavigationPolicy.allows(
                    navigationAction.request.url, isInitialLoad: isInitialLoad
                ) ? .allow : .cancel
            )
        }
    }
}

// MARK: - The sheet

/// The interactive picture of a saved world, inside the app.
///
/// Presented as a sheet from the World Builder header. It fetches the page as
/// it appears, shows the Tower's own sentence when there is none, and hands
/// the page to a web view that can go nowhere else. The caption at the top
/// says what the picture is, in the same words the page itself uses, because
/// a point cloud read as a scan of the room is the one claim this screen
/// must not make.
struct WorldRenderViewerView: View {
    @StateObject private var model: WorldRenderViewerModel
    @Environment(\.dismiss) private var dismiss

    init(target: WorldRenderTarget, client: WorldRenderClient = WorldRenderClient()) {
        _model = StateObject(wrappedValue: WorldRenderViewerModel(target: target, client: client))
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                caption
                content
            }
            .navigationTitle("Reconstruction")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
            .task { await model.load() }
        }
    }

    private var caption: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(model.target.sessionID.map { "World \(model.target.worldID) · session \($0)" }
                 ?? "World \(model.target.worldID) · newest session with geometry")
                .font(.caption.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
            Text("Sparse structure-from-motion output: feature points and camera poses. Not a surface, not metric scale.")
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
                Text("Asking the Tower for the picture…")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .ready(let html):
            WorldRenderWebView(html: html)
                .ignoresSafeArea(edges: .bottom)
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
}
