//
//  TowerAddress.swift
//  Glasses
//

import Foundation
import Network

// MARK: - What an address may be

/// Why a typed Tower address was or was not accepted.
///
/// A reason, not a sentence: the words are `TowerSettingsText`'s, so a test can
/// pin each case without rendering anything.
nonisolated enum TowerAddressCheck: Equatable, Sendable {
    /// `host[:port]`, trimmed, exactly as the URLs will be built from it.
    case accepted(String)
    case empty
    /// `http://…` or `ws://…`: the scheme is the app's, never the person's.
    case includesScheme
    /// Not a `host[:port]` that builds a URL (a path, credentials, a bad port…).
    case malformed
    /// Well-formed, but outside what this build's App Transport Security lets
    /// through. Release builds only.
    case notPrivate
}

/// Which addresses this build may use for a Tower.
///
/// ## Why the rule depends on the build
///
/// The Tower speaks plain `http://` and `ws://`. What a build can reach over
/// plain text is decided by App Transport Security in `Info.plist`, not by
/// this code, and the two configurations differ (see the ATS comment there):
///
/// - **Debug** allows arbitrary loads, so any well-formed host works and this
///   checks syntax only.
/// - **Release** allows only the private ranges below, `.local`, and `.ts.net`.
///   An address outside them would be refused by ATS at the first request,
///   with an error nobody could act on. Refusing it here instead says why,
///   before anything is saved.
///
/// `privateRanges` and `privateNameSuffixes` must list exactly the Release
/// exceptions in `Info.plist`; `TowerSettingsTests` reads the file and checks.
nonisolated enum TowerAddressPolicy: Equatable, Sendable {
    /// Syntax only. Debug builds.
    case anyHost
    /// Syntax, plus a private, link-local, Tailscale, `.local` or `.ts.net`
    /// host. Release builds.
    case privateNetworks

    /// The policy of the running build.
    static var current: TowerAddressPolicy {
        #if DEBUG
        return .anyHost
        #else
        return .privateNetworks
        #endif
    }

    /// RFC 1918, IPv4 link-local, Tailscale's CGNAT range, and IPv6
    /// unique-local (Tailscale's `fd7a:115c:a1e0::/48` is inside it).
    /// Loopback is deliberately absent: a Tower is never the phone itself.
    static let privateRanges = [
        "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
        "100.64.0.0/10", "fc00::/7",
    ]

    /// Name suffixes a Release build may use. `.local` is covered by
    /// `NSAllowsLocalNetworking`; `ts.net` (Tailscale MagicDNS) by an
    /// exception domain that includes its subdomains.
    static let privateNameSuffixes = ["local", "ts.net"]

    private static let parsedRanges: [TowerAddressRange] = privateRanges.compactMap(TowerAddressRange.init)

    func check(_ candidate: String) -> TowerAddressCheck {
        let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return .empty }
        if trimmed.contains("://") { return .includesScheme }
        guard let accepted = TowerConfiguration.acceptedAuthority(trimmed) else { return .malformed }
        switch self {
        case .anyHost:
            return .accepted(accepted)
        case .privateNetworks:
            return Self.isPrivate(authority: accepted) ? .accepted(accepted) : .notPrivate
        }
    }

    /// Whether an already-accepted `host[:port]` names a host a Release build
    /// may reach.
    static func isPrivate(authority: String) -> Bool {
        guard let host = URLComponents(string: "http://\(authority)")?.host else {
            return false
        }
        let bare = host.hasPrefix("[") && host.hasSuffix("]") ? String(host.dropFirst().dropLast()) : host
        if let bytes = TowerAddressRange.bytes(ofLiteral: bare) {
            return parsedRanges.contains { $0.contains(bytes) }
        }
        let name = bare.lowercased()
        return privateNameSuffixes.contains { suffix in
            name.hasSuffix(".\(suffix)") && name.count > suffix.count + 1
        }
    }
}

/// One CIDR range, as ATS's `NSExceptionDomains` spells it.
nonisolated struct TowerAddressRange: Equatable, Sendable {
    let cidr: String
    private let network: [UInt8]
    private let prefixLength: Int

    init?(_ cidr: String) {
        let parts = cidr.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, let prefix = Int(parts[1]),
              let network = Self.bytes(ofLiteral: String(parts[0])),
              (0...(network.count * 8)).contains(prefix)
        else { return nil }
        self.cidr = cidr
        self.network = network
        self.prefixLength = prefix
    }

    /// The address's bytes when `literal` is a dotted-quad IPv4 or an IPv6
    /// literal, else `nil` (a name). Strict dotted quad: Network's parser is
    /// not relied on to refuse `10.1` or `010.0.0.1`, which some C parsers read
    /// as other addresses entirely.
    static func bytes(ofLiteral literal: String) -> [UInt8]? {
        if literal.contains(":") {
            return IPv6Address(literal).map { [UInt8]($0.rawValue) }
        }
        let octets = literal.split(separator: ".", omittingEmptySubsequences: false)
        guard octets.count == 4 else { return nil }
        var bytes: [UInt8] = []
        for octet in octets {
            guard !octet.isEmpty, octet.count <= 3, octet.allSatisfy(\.isASCII), octet.allSatisfy(\.isNumber),
                  octet.count == 1 || octet.first != "0",
                  let value = UInt8(octet)
            else { return nil }
            bytes.append(value)
        }
        return bytes
    }

    func contains(_ bytes: [UInt8]) -> Bool {
        guard bytes.count == network.count else { return false }
        var remaining = prefixLength
        for index in bytes.indices where remaining > 0 {
            let bits = min(8, remaining)
            let mask = UInt8(truncatingIfNeeded: 0xFF << (8 - bits))
            if bytes[index] & mask != network[index] & mask { return false }
            remaining -= bits
        }
        return true
    }
}

// MARK: - Where a saved address lives

/// The one saved Tower address, in `UserDefaults`.
///
/// Read once per launch, by `TowerConfiguration.resolution`, and written only
/// by Settings. Nothing else reads it: every Tower URL in the app comes from
/// `TowerConfiguration`, so a saved value reaches the whole app together at
/// the next launch or not at all.
nonisolated struct TowerAddressStore {
    static let key = "GlassesTowerAuthority"

    /// DEBUG only: a UI test launches with this to start from "nothing saved",
    /// so a run that failed half-way cannot leave the next one pointed at a
    /// mock. Honoured in `GlassesApp.init`, before anything reads the address.
    static let uiTestResetArgument = "-UITestResetTowerAddress"

    let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    var saved: String? {
        defaults.string(forKey: Self.key)
    }

    func save(_ authority: String) {
        defaults.set(authority, forKey: Self.key)
    }

    func clear() {
        defaults.removeObject(forKey: Self.key)
    }
}
