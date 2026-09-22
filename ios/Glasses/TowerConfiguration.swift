//
//  TowerConfiguration.swift
//  Glasses
//
//  Created by Tristan Varner on 8/18/26.
//

import Foundation

/// Single source of truth for the development Tower endpoint. The Tower is
/// currently remote, reached over Tailscale; this hardcoded address is
/// expected to change across sessions/networks until a discovery or
/// configuration mechanism is designed.
nonisolated enum TowerConfiguration {
    /// `host:port` of the development Tower.
    static let defaultAuthority = "100.110.156.55:8000"

    /// The environment variable a DEBUG build reads for a different Tower.
    ///
    /// Exists so the Simulator can be pointed at a Tower running on the Mac
    /// beside it — the only way the World Builder and CV Lab screens can be
    /// exercised against a Tower on the branch under test without a Windows
    /// box in the loop:
    ///
    ///     xcrun simctl launch --setenv GLASSES_TOWER_AUTHORITY=127.0.0.1:8000 \
    ///         booted com.tristanvarner.Glasses
    ///
    /// Read once, in DEBUG only, and only as a `host[:port]`: a value that
    /// does not form a URL is ignored rather than trusted, so a typo lands on
    /// the development Tower and not on a crash. Release builds never look.
    static let overrideVariable = "GLASSES_TOWER_AUTHORITY"

    /// What every Tower URL below is built from.
    static let authority: String = {
        #if DEBUG
        if let candidate = ProcessInfo.processInfo.environment[overrideVariable],
           let accepted = acceptedAuthority(candidate) {
            return accepted
        }
        #endif
        return defaultAuthority
    }()

    /// `host[:port]` with no scheme, path, credentials or query, or `nil`.
    /// Checked by building the URL that would be built from it and reading
    /// back only a host and a port.
    static func acceptedAuthority(_ candidate: String) -> String? {
        let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        // `%` is refused outright: `URLComponents.host` comes back decoded, so
        // `%2F:1` would pass every other check as `/:1` and then fail to build
        // a URL below — or worse, build one the checks meant to forbid.
        guard !trimmed.isEmpty, !trimmed.contains("/"), !trimmed.contains("@"),
              !trimmed.contains("?"), !trimmed.contains("#"), !trimmed.contains("%"),
              let components = URLComponents(string: "http://\(trimmed)"),
              let host = components.host, !host.isEmpty,
              components.path.isEmpty, components.user == nil
        else { return nil }
        // An IPv6 literal needs its brackets; Foundation may or may not have
        // kept them on the way through `host`.
        let literal = host.contains(":") && !host.hasPrefix("[") ? "[\(host)]" : host
        let authority = components.port.map { "\(literal):\($0)" } ?? literal
        // The two `static let`s below force-unwrap what they build from this,
        // so the claim "a bad value is ignored, never a crash" is checked
        // here, against exactly the URLs they will build.
        guard URL(string: "http://\(authority)") != nil,
              URL(string: "ws://\(authority)/ws") != nil
        else { return nil }
        return authority
    }

    static let webSocketURL = URL(string: "ws://\(authority)/ws")!

    /// The same Tower, over HTTP. Geometry is fetched here rather than over the
    /// socket because the Tower gives its result sender and its frame path one
    /// shared lock, and a megabyte of points there would starve the frames.
    static let httpBaseURL = URL(string: "http://\(authority)")!
}
