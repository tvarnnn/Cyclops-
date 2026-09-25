//
//  TowerConfiguration.swift
//  Glasses
//
//  Created by Tristan Varner on 8/18/26.
//

import Foundation

/// Single source of truth for the Tower endpoint.
///
/// ## Where the address comes from, in order
///
/// 1. **`GLASSES_TOWER_AUTHORITY`**, in DEBUG builds only — the developer's
///    override, which is how the Simulator and `TowerSmokeUITests` reach a
///    Tower on the Mac (or the read-only proxy in front of a live one).
/// 2. **The address saved in Settings**, if it passes this build's
///    `TowerAddressPolicy`.
/// 3. **`defaultAuthority`**, the development Tower over Tailscale.
///
/// With nothing saved and no override this is exactly the compiled-in address
/// it always was.
///
/// ## Fixed for the life of the process, on purpose
///
/// Every value here is a `static let`, resolved once at launch, and Settings
/// does not change it: a saved address takes effect on the **next** launch, and
/// Settings says so. About twenty clients default their `baseURL` to
/// `httpBaseURL` when they are built, some of them owned for the whole process
/// (`ProjectManager.cartridgeClients`), and the socket can be mid-capture.
/// Changing the address under them would leave frames going to one Tower
/// while worlds are fetched from another. One address per process is the
/// guarantee that cannot happen, and it costs one relaunch.
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
    /// does not form a URL is ignored rather than trusted, so a typo falls
    /// through to the saved address or the development Tower and not to a
    /// crash. Release builds never look. It wins over a saved address, so a
    /// test that sets it steers the app whatever the Simulator has saved.
    static let overrideVariable = "GLASSES_TOWER_AUTHORITY"

    /// Where the address in use came from.
    enum Source: Equatable, Sendable {
        /// `GLASSES_TOWER_AUTHORITY`, DEBUG only.
        case developerOverride
        /// Settings.
        case saved
        /// `defaultAuthority`.
        case builtIn
    }

    struct Resolution: Equatable, Sendable {
        let authority: String
        let source: Source
    }

    /// The precedence rule, as a pure function so every combination can be
    /// tested without a launch. A value that is present but unacceptable is
    /// skipped, never trusted and never fatal.
    static func resolve(
        overrideValue: String?,
        savedValue: String?,
        policy: TowerAddressPolicy
    ) -> Resolution {
        if let overrideValue, let accepted = acceptedAuthority(overrideValue) {
            return Resolution(authority: accepted, source: .developerOverride)
        }
        if let savedValue, case .accepted(let accepted) = policy.check(savedValue) {
            return Resolution(authority: accepted, source: .saved)
        }
        return Resolution(authority: defaultAuthority, source: .builtIn)
    }

    /// `GLASSES_TOWER_AUTHORITY` as this process was launched with it, or
    /// `nil` — always `nil` in a Release build.
    static var launchOverrideValue: String? {
        #if DEBUG
        return ProcessInfo.processInfo.environment[overrideVariable]
        #else
        return nil
        #endif
    }

    /// What this process uses, decided once.
    static let resolution: Resolution = resolve(
        overrideValue: launchOverrideValue,
        savedValue: TowerAddressStore().saved,
        policy: .current
    )

    /// What every Tower URL below is built from.
    static let authority: String = resolution.authority

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
        // A port outside 1...65535 parses, and builds a URL, and can never be
        // connected to.
        if let port = components.port, !(1...65535).contains(port) { return nil }
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
