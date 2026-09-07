//
//  WorldPickerView.swift
//  Glasses
//

import SwiftUI

/// The Tower's saved worlds, one section per world, one row per session.
///
/// Presented as a sheet from the World Builder header. It asks the view model
/// to load the list as it appears and draws exactly what came back, shaped by
/// `WorldListingPresentation`: the Tower's display name or, when it has none,
/// a dated title with the id as a caption; the session count; a "live" badge
/// when a builder holds the world's writer lock; and per session the Tower's
/// `frame_source` word, its state word as a badge, and its keyframe count.
/// Tapping a session opens it; tapping the world's own row opens the world
/// and lets the Tower pick the session.
///
/// Worlds with no sessions are not offered as rows. There is nothing to open
/// in one — the Tower answers "the world exists and has no sessions" and the
/// picture 404s — and on the real root they were 96 of 162. They are counted
/// behind one disclosure at the bottom, so the number is still visible and a
/// person who wants one can still reach it.
///
/// Not `#if DEBUG`: opening a stored world reads, and a Release build with no
/// camera is entitled to read.
struct WorldPickerView: View {
    @ObservedObject var world: WorldBuilderViewModel
    @Environment(\.dismiss) private var dismiss

    private var grouped: WorldListingPresentation.Grouped {
        WorldListingPresentation.grouped(world.worlds)
    }

    var body: some View {
        NavigationStack {
            List {
                statusRows

                ForEach(grouped.primary, id: \.worldID) { entry in
                    Section {
                        worldRow(entry)
                        ForEach(entry.sessions, id: \.sessionID) { session in
                            sessionRow(entry: entry, session: session)
                        }
                    }
                }

                if let heading = grouped.emptyHeading {
                    Section {
                        DisclosureGroup(heading) {
                            ForEach(grouped.empty, id: \.worldID) { entry in
                                worldRow(entry)
                            }
                        }
                        .font(.footnote)
                        .foregroundStyle(.secondary)
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

    // MARK: Status

    /// Loading, failure with a Retry, or the empty sentence — in that order,
    /// because a request in flight outranks the answer it is about to
    /// replace.
    @ViewBuilder
    private var statusRows: some View {
        if world.isLoadingWorlds {
            // Honest: a request really is out.
            HStack(spacing: 10) {
                ProgressView()
                Text("Asking the Tower for its saved worlds…")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        } else if let failure = world.worldListFailure {
            VStack(alignment: .leading, spacing: 8) {
                Text(failure)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button("Retry") {
                    Task { await world.loadWorlds() }
                }
                .font(.footnote)
                .buttonStyle(.bordered)
            }
        } else if world.worlds.isEmpty {
            // Empty after an empty answer. "None yet" is true of a Tower with
            // no worlds and of one whose root resolved to nowhere, without
            // claiming which; the log line says which.
            Text("No saved worlds have been listed yet.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: Rows

    private func worldRow(_ entry: WorldListingEntry) -> some View {
        Button {
            open(worldID: entry.worldID, sessionID: nil)
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.title)
                        .font(.headline)
                    // The id stays reachable — it is what the Tower's logs
                    // and the canvas header name — but as a caption under a
                    // title, never as the title.
                    Text(entry.worldID)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(WorldListingPresentation.sessionCount(entry.sessions.count))
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
    }

    private func sessionRow(entry: WorldListingEntry, session: WorldListingSession) -> some View {
        Button {
            open(worldID: entry.worldID, sessionID: session.sessionID)
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(session.sessionID)
                        .font(.caption.monospaced())
                        .lineLimit(1)
                        .truncationMode(.middle)
                    HStack(spacing: 6) {
                        Text(session.frameSource)
                        if let keyframes = WorldListingPresentation.keyframeCaption(for: session) {
                            Text("·")
                            Text(keyframes)
                        }
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }
                Spacer()
                if let badge = WorldListingPresentation.stateBadge(for: session) {
                    Text(badge)
                        .font(.caption2)
                        .foregroundStyle(badgeStyle(for: session))
                }
            }
        }
        .buttonStyle(.plain)
    }

    /// Muted for a session with nothing to show, ordinary otherwise. The
    /// words carry the meaning; the colour only stops "No geometry" from
    /// reading as an alarm.
    private func badgeStyle(for session: WorldListingSession) -> HierarchicalShapeStyle {
        session.hasGeometry ? .secondary : .tertiary
    }

    private func open(worldID: String, sessionID: String?) {
        world.open(worldID: worldID, sessionID: sessionID)
        dismiss()
    }
}
