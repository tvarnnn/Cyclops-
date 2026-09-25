//
//  TowerConnectionTester.swift
//  Glasses
//

import Foundation

/// What Settings' "Test connection" found at an address. Exactly one of three.
nonisolated enum TowerProbeOutcome: Equatable, Sendable {
    /// Nothing answered: no route, nothing listening, no such name, the time
    /// ran out, or this build's transport security refused the address.
    case notReachable(TowerProbeFailure)
    /// Something answered, and what it said is not a Glasses Tower's
    /// capability declaration.
    case notATower(TowerProbeMismatch)
    /// A Tower's declaration came back. `contract` is its
    /// `envelope_contract`, when it named one.
    case tower(contract: String?)
}

nonisolated enum TowerProbeFailure: Equatable, Sendable {
    case timedOut
    case refused
    case nameNotFound
    case noNetwork
    case blockedByTransportSecurity
    case other
}

nonisolated enum TowerProbeMismatch: Equatable, Sendable {
    /// An HTTP status outside 2xx — including a redirect, which is never
    /// followed.
    case status(Int)
    /// A 2xx whose body is not `{"type": "cartridges", "cartridges": [...]}`.
    case notADeclaration
    /// Bytes came back that are not an HTTP response at all.
    case notHTTP
}

/// One read-only request to find out whether an address is a Tower.
///
/// ## Read-only, and structurally so
///
/// It sends `GET /cartridges` and nothing else, and it is the only thing it can
/// send: there is no method here that takes a path or a verb. `/cartridges` is
/// the Tower's capability declaration — a pure function of its configuration
/// (`tower/tower/routes/cartridges.py`, `registry.declare`), which changes
/// nothing on the Tower and starts nothing. It does not go through
/// `TowerClient`, opens no socket, and cannot reach `session/start`.
///
/// `GET /cartridges` rather than `/health` because the declaration is also
/// the proof of identity: `{"type": "cartridges", …}` is a Glasses Tower's
/// answer and nobody else's, and it names the result-envelope contract.
///
/// ## Bounded
///
/// Five seconds, in total, not per packet: the session's
/// `timeoutIntervalForResource` caps the whole exchange, so a server that
/// accepts the connection and then dribbles cannot hold the button spinning.
/// Redirects are refused rather than followed, so the one request cannot turn
/// into a request to somewhere else. No cache, no cookies.
nonisolated struct TowerConnectionTester {
    static let timeout: TimeInterval = 5
    static let path = "cartridges"

    /// `nil` builds a fresh ephemeral session per probe. Tests inject one
    /// whose protocol classes stand in for the network.
    var session: URLSession?

    init(session: URLSession? = nil) {
        self.session = session
    }

    /// The request a probe of `authority` sends. Separate so a test can pin
    /// the method, path and timeout without a network.
    static func request(for authority: String) -> URLRequest? {
        guard let base = URL(string: "http://\(authority)") else { return nil }
        var request = URLRequest(
            url: base.appendingPathComponent(path),
            cachePolicy: .reloadIgnoringLocalCacheData,
            timeoutInterval: timeout
        )
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        return request
    }

    static func sessionConfiguration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.waitsForConnectivity = false
        configuration.urlCache = nil
        configuration.httpShouldSetCookies = false
        return configuration
    }

    func probe(authority: String) async -> TowerProbeOutcome {
        guard let request = Self.request(for: authority) else { return .notReachable(.other) }
        let session = self.session ?? URLSession(configuration: Self.sessionConfiguration())
        defer { if self.session == nil { session.finishTasksAndInvalidate() } }
        do {
            let (data, response) = try await session.data(for: request, delegate: RedirectRefuser())
            return Self.classify(data: data, response: response)
        } catch {
            return Self.classify(error: error)
        }
    }

    static func classify(data: Data, response: URLResponse) -> TowerProbeOutcome {
        guard let http = response as? HTTPURLResponse else { return .notATower(.notHTTP) }
        guard (200...299).contains(http.statusCode) else { return .notATower(.status(http.statusCode)) }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              json["type"] as? String == "cartridges",
              json["cartridges"] is [Any]
        else { return .notATower(.notADeclaration) }
        return .tower(contract: json["envelope_contract"] as? String)
    }

    static func classify(error: Error) -> TowerProbeOutcome {
        guard let urlError = error as? URLError else { return .notReachable(.other) }
        switch urlError.code {
        case .timedOut:
            return .notReachable(.timedOut)
        case .cannotConnectToHost:
            return .notReachable(.refused)
        case .cannotFindHost, .dnsLookupFailed:
            return .notReachable(.nameNotFound)
        case .notConnectedToInternet, .internationalRoamingOff, .dataNotAllowed:
            return .notReachable(.noNetwork)
        case .appTransportSecurityRequiresSecureConnection:
            return .notReachable(.blockedByTransportSecurity)
        // Something answered, in bytes that are not a usable HTTP response:
        // an SSH banner, a TLS-only server, a raw TCP service.
        case .badServerResponse, .cannotParseResponse, .cannotDecodeRawData,
             .cannotDecodeContentData, .dataLengthExceedsMaximum:
            return .notATower(.notHTTP)
        default:
            return .notReachable(.other)
        }
    }
}

/// Answers every redirect with "do not follow", so the 3xx itself is what the
/// probe classifies.
///
/// The completion-handler form, not the `async` one: Swift 6.3.3 crashes in
/// SILGen emitting the Objective-C thunk for the `async` variant here.
private nonisolated final class RedirectRefuser: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}
