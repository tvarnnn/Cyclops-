//
//  IMURecordingBadge.swift
//  Glasses
//

// DEBUG-only, like the recorder it reports on.
#if DEBUG

import SwiftUI

/// The persistent sign that this phone is writing an IMU log.
///
/// Motion is raw sensor data (docs/06-PRIVACY-DATA.md), so while the recorder
/// holds a file this badge sits in the navigation bar above every workspace,
/// where no cartridge can hide it and no scroll moves it. It observes only the
/// recorder's readout, not `GlassesConnection`, so it redraws with the
/// recorder rather than with every camera frame. Nothing shows when no
/// recorder is open.
struct IMURecordingBadge: View {
    @ObservedObject var readout: IMURecorderReadout

    var body: some View {
        switch readout.indicator {
        case .off:
            EmptyView()
        case .writing:
            badge(symbol: "record.circle.fill", title: "IMU", tint: .red, value: "recording")
        case .stopped(let reason):
            badge(symbol: "exclamationmark.triangle.fill", title: "IMU", tint: .orange, value: "stopped, \(reason)")
        }
    }

    private func badge(symbol: String, title: String, tint: Color, value: String) -> some View {
        Label(title, systemImage: symbol)
            .labelStyle(.titleAndIcon)
            .font(.caption.weight(.semibold))
            .foregroundStyle(tint)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(tint.opacity(0.15), in: .capsule)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("IMU log")
            .accessibilityValue(value)
    }
}

#endif
