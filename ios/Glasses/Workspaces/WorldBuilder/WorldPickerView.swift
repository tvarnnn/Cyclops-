//
//  WorldPickerView.swift
//  Glasses
//

import SwiftUI

/// The Tower's saved worlds, one section per world, one row per session.
///
/// Presented as a sheet from the World Builder header. It asks the view model
/// to load the list as it appears and draws exactly what came back: the
/// Tower's display name or, when it has none, the world id itself; the
/// session count; a "live" badge when a builder holds the world's writer lock;
/// and per session the Tower's `frame_source` word and whether the session is
/// still open. Tapping a session opens it; tapping the world's own row opens
/// the world and lets the Tower pick the session.
///
/// Not `#if DEBUG`: opening a stored world reads, and a Release build with no
/// camera is entitled to read.
struct WorldPickerView: View {
    @ObservedObject var world: WorldBuilderViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if let failure = world.worldListFailure {
                    Text(failure)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } else if world.worlds.isEmpty {
                    // Empty before the first answer and after an empty one.
                    // "None yet" is true of both without claiming which.
                    Text("No saved worlds have been listed yet.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                ForEach(world.worlds, id: \.worldID) { entry in
                    Section {
                        Button {
                            open(worldID: entry.worldID, sessionID: nil)
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(entry.title)
                                        .font(.headline)
                                    Text(sessionCount(entry.sessions.count))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if entry.live {
                                    Text("live")
                                        .font(.caption2.weight(.semibold))
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 2)
                                        .background(Color.accentColor.opacity(0.15), in: Capsule())
                                }
                            }
                        }
                        .buttonStyle(.plain)

                        ForEach(entry.sessions, id: \.sessionID) { session in
                            Button {
                                open(worldID: entry.worldID, sessionID: session.sessionID)
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(session.sessionID)
                                            .font(.caption.monospaced())
                                            .lineLimit(1)
                                            .truncationMode(.middle)
                                        Text(session.frameSource)
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    if session.isStillOpen {
                                        Text("still open")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                    if !session.hasGeometry {
                                        Text("no geometry")
                                            .font(.caption2)
                                            .foregroundStyle(.tertiary)
                                    }
                                }
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            .navigationTitle("Saved worlds")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
            .task { await world.loadWorlds() }
        }
    }

    private func open(worldID: String, sessionID: String?) {
        world.open(worldID: worldID, sessionID: sessionID)
        dismiss()
    }

    private func sessionCount(_ count: Int) -> String {
        count == 1 ? "1 session" : "\(count) sessions"
    }
}
