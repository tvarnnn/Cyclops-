//
//  TowerConnectionRow.swift
//  Glasses
//

import SwiftUI

/// The four words `TowerConnectionRow` uses for `TowerStatus`, and the one
/// action each offers.
///
/// Its own mapping rather than `StateDisplay.tower(_:)`, and the difference is
/// deliberate. The shell's pill says "Offline", and "Disconnected" for a
/// failure, because it is read at a glance beside two other pills;
/// `ProductShellTests.testTowerStatusLabels` pins those words and this does
/// not touch them. This row sits beside a button that acts on the state, so it
/// names the state the way the button reads: Disconnected → Connect,
/// Connecting → Cancel, Connected → Disconnect, Error → Connect.
///
/// A pure mapping with no view in it, so a test can pin the four words and the
/// three titles without rendering anything.
enum CVTowerConnectionLabel {
    static func text(for status: TowerStatus) -> String {
        switch status {
        case .offline: return "Disconnected"
        case .connecting: return "Connecting"
        case .online: return "Connected"
        case .failed: return "Error"
        }
    }

    /// Mirrors `ConnectionSheet`: Disconnect stays available while
    /// `.connecting`, because `connect()` is a no-op in that state and the
    /// handshake has no cancel of its own, so without it a hung connect would
    /// strand a person until URLSession's own timeout expired.
    static func actionTitle(for status: TowerStatus) -> String {
        switch status {
        case .online: return "Disconnect"
        case .connecting: return "Cancel"
        case .offline, .failed: return "Connect"
        }
    }
}

/// The Tower connection, from inside a workspace.
///
/// ## Why a leaf
///
/// Observes `TowerClient` and nothing else. `TowerClient` republishes once per
/// reply, at the ~12 Hz target rate while a session streams; this row is
/// invalidated at that rate and the workspace that mounts it is not, because
/// the workspace holds `tower` as a plain reference. Same trick as
/// `CVFrameReadingPanel`, same reason.
///
/// ## What it offers
///
/// The state in one of four words with the shell's colour mapping, the
/// failure detail when there is one, the endpoint in use — an address that
/// changes between networks (Settings sets it), so "cannot reach the Tower" is
/// a mystery without it and a fixable problem with it — and the one action `ConnectionSheet` would offer for the same state.
/// Nothing here that the sheet does not do; it is here so a person running an
/// experiment does not have to leave the Lab to do it.
struct TowerConnectionRow: View {
    @ObservedObject var tower: TowerClient

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            StatusPill(
                title: "Tower",
                value: CVTowerConnectionLabel.text(for: tower.status),
                level: level
            )
            .frame(maxWidth: 132)

            VStack(alignment: .leading, spacing: 6) {
                Text(endpoint)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let detail = StateDisplay.towerFailureDetail(tower.status) {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Button(CVTowerConnectionLabel.actionTitle(for: tower.status)) {
                    switch tower.status {
                    case .online, .connecting: tower.disconnect()
                    case .offline, .failed: tower.connect()
                    }
                }
                .buttonStyle(.bordered)
                .font(.subheadline.weight(.medium))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 4)
        }
        .accessibilityElement(children: .contain)
    }

    /// The shell's mapping, so the same state is never two colours on one
    /// screen. `ShellStatusBar.towerLevel` is private to the shell; four lines
    /// here are cheaper than a shared type for one switch.
    private var level: StatusLevel {
        switch tower.status {
        case .online: return .ok
        case .connecting: return .working
        case .failed: return .problem
        case .offline: return .idle
        }
    }

    /// `host:port` rather than the whole URL: the scheme and path are the same
    /// on every network, and the address is the part that changes.
    private var endpoint: String {
        let url = TowerConfiguration.webSocketURL
        guard let host = url.host(percentEncoded: false) else { return url.absoluteString }
        if let port = url.port { return "\(host):\(port)" }
        return host
    }
}
