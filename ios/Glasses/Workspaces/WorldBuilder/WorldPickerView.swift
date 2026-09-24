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
/// Tapping a session pushes its 3D world; tapping the world's own row pushes
/// the world and lets the Tower pick the session.
///
/// **A row is a control only when there is something behind it.** The render
/// route answers 404 for a world with no sessions, for a world none of whose
/// sessions has geometry, and for a session that has none — and `absent` is
/// classified retryable, so every one of those taps would land on the Tower's
/// absence prose beside a Try again that nothing on this screen can satisfy.
/// Those three are drawn as text that says which case it is. See
/// `emptyWorldRow`, `worldRow` and `sessionRow`.
///
/// Worlds with no sessions are not offered as **controls**. There is nothing to
/// open in one — the Tower answers "the world exists and has no sessions" and
/// the render route 404s — and on the real root they were 96 of 162. They are
/// listed behind one disclosure at the bottom, as text with their ids, so the
/// number and the identifiers stay visible and no tap leads into a 404 with a
/// Try again that cannot succeed. See `emptyWorldRow`.
///
/// Not `#if DEBUG`: opening a stored world reads, and a Release build with no
/// camera is entitled to read.
struct WorldPickerView: View {
    @ObservedObject var world: WorldBuilderViewModel
    @Environment(\.dismiss) private var dismiss

    /// The world whose 3D reconstruction is pushed, or `nil`.
    ///
    /// ## Why a tap here goes straight to the 3D world
    ///
    /// It used to pin the world and dismiss, leaving the reader on the
    /// workspace looking at a gallery of top-down sparse point clouds captioned
    /// with the solver's registration vocabulary. The 3D reconstruction — the
    /// thing they opened a saved world to see — was behind a small bordered
    /// "Picture" button in the header, two taps and a scroll away.
    ///
    /// So the tap pushes it. The pin still happens, so the workspace behind is
    /// showing the same world when Close is used, and the picker stays up
    /// underneath so a person comparing two walks does not have to reopen it.
    @State private var opened: WorldRenderTarget?

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
                                emptyWorldRow(entry)
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
            // Pushed, not presented: a sheet over a sheet stacks two dimming
            // layers and two Close buttons, and the back gesture is what a
            // person reaches for after looking at one of several walks.
            .navigationDestination(item: $opened) { target in
                WorldRenderScene(target: target, title: openedTitle, note: openedNote)
            }
            .task { await world.loadWorlds() }
            // A pull, so a "Finishing" row can become "Complete" without
            // closing and reopening the sheet — which the caption under a
            // still-open row promises "a refresh of this list will show".
            .refreshable { await world.loadWorlds() }
        }
    }

    /// What to call the world that is pushed. Resolved from the listing rather
    /// than from the live snapshot, because the pin's first status report has
    /// usually not arrived by the time this is drawn.
    private var openedTitle: String? {
        openedEntry.map { WorldListingPresentation.title(for: $0) }
    }

    /// What the pushed screen says about the world under its caption, or `nil`.
    /// See `note(forOpened:target:pinnedStage:pinnedReconstruction:)`.
    ///
    /// The pin's own report is consulted only once it describes THIS world:
    /// `open()` sets `renderTarget` at the tap, but the status channel's answer
    /// reaches the view model a main-queue hop later, so for a moment the
    /// presentation is still the world the workspace was showing before.
    private var openedNote: String? {
        guard let opened, let session = openedSession else { return nil }
        let pinned = world.renderTarget == opened
            && world.state.snapshot?.worldID == opened.worldID
        return Self.note(
            forOpened: session, target: opened,
            pinnedStage: pinned ? world.presentation.stage : nil,
            pinnedReconstruction: pinned ? world.presentation.reconstruction : nil,
            pinnedPhotographic: pinned ? world.presentation.photographic : nil
        )
    }

    /// The note for a session pushed from this list.
    ///
    /// **A settled row** says only what it always said: from the **listing's**
    /// finalization record, a sentence about a final pass the Tower said did
    /// not happen, and otherwise nothing — silence stays silent, per
    /// `WorldFinalSolve.notReported`.
    ///
    /// **A row the Tower lists as still changing** (`finalizing`, `receiving`)
    /// used to get the same treatment, which for a `finalizing` row is nothing
    /// at all. That is the photographic build: Stop, open Saved Worlds, tap the
    /// "Finishing" row, and the screen showed the sparse points with no word
    /// that the picture the walk was for is still being made — while the
    /// workspace's own route into the same viewer said "it is worth waiting for
    /// Saved". The same misreading as the 2026-09-22 incident, on the other
    /// screen (found by the Windows lane's review round 2, left for a Mac).
    ///
    /// So such a row gets the ladder's note: the pin's own live one once its
    /// report has arrived — so the sentence leaves when the build lands, rather
    /// than when this list is next refreshed — and until then the note for
    /// what the row says.
    ///
    /// ## `finalizing` is NOT always a live process (corrected 2026-09-23)
    ///
    /// This said it was: "a held lock, or a photographic stage running". Since
    /// the Tower's `photographic` block (WORLDS §2a) a row is also `finalizing`
    /// when its photographic room is `owed` -- nothing is building it, and it
    /// waits for an idle Tower with its photographic stages on -- or
    /// `unobservable`. The Improving note's "it is worth waiting for Saved" is
    /// a promise neither can keep (Mac gate B0, F3). So the row's own
    /// `photographic` word picks the note, through the same ladder the canvas
    /// uses; a row with no word (an older Tower, whose `finalizing` really was
    /// a live lock) keeps the Improving note.
    ///
    /// **A settled row whose photographic build failed** says so: the one
    /// case `WORLD-BUILDER-IOS.md` §3a says is worth new copy.
    static func note(
        forOpened session: WorldListingSession,
        target: WorldRenderTarget,
        pinnedStage: WorldStage?,
        pinnedReconstruction: WorldReconstruction?,
        pinnedPhotographic: WorldPhotographicReport? = nil
    ) -> String? {
        let solve = WorldFinalSolve(word: session.finalization?.finalSolve)
        // The pin's report is newer than the row once it has arrived; until
        // then the row is all there is.
        let photographic = pinnedStage != nil ? pinnedPhotographic : session.photographic
        let failed = photographic?.standing.isFailed == true && !solve.deniesAFinishedWorld
        let settled = solve.deniesAFinishedWorld
            ? solve.sentence
            : (failed ? WorldPhotographicCopy.failedHeadline + "." : nil)
        guard session.state == .finalizing || session.state == .receiving else { return settled }
        if pinnedStage != nil, let pinnedReconstruction {
            if case .partial(_, let note) = pinnedReconstruction { return note }
            return settled
        }
        let listed: WorldStage = session.state == .receiving ? .mapping : .improving
        if case .partial(_, let note) = WorldReconstruction.ladder(
            target: target, stage: listed, finalSolve: .notReported, evidence: nil,
            photographic: session.photographic
        ) {
            return note
        }
        return settled
    }

    private var openedEntry: WorldListingEntry? {
        guard let opened else { return nil }
        return world.worlds.first { $0.worldID == opened.worldID }
    }

    /// The listing row for the session that was pushed, when one was named. A
    /// tap on a world's own row names none — the Tower chooses — so this is
    /// `nil` there and the screen says nothing rather than guessing which
    /// session the Tower will pick.
    private var openedSession: WorldListingSession? {
        guard let sessionID = opened?.sessionID else { return nil }
        return openedEntry?.sessions.first { $0.sessionID == sessionID }
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

    /// A world with at least one session that has geometry: tapping it lets the
    /// Tower choose the session and pushes the 3D world.
    ///
    /// A world whose sessions **all** report `has_geometry: false` is drawn the
    /// way a session-less world is, and for the same reason: the render route
    /// picks the newest session with geometry, finds none, and 404s. Tapping it
    /// could only ever land on the Tower's absence prose under a Try again that
    /// this screen cannot make succeed.
    ///
    /// **The session is named, not left to the Tower.** With `sessionID: nil`
    /// the two halves of what opens were resolved by two different rules:
    /// the render route draws the newest session WITH geometry, while the
    /// status pin behind it resolves the world's latest session regardless
    /// — so on a world whose newest walk found nothing, the 3D view drew
    /// one walk and the panel and Diagnostics under it described another.
    /// Naming the newest session with geometry here — the same choice the
    /// render route makes, from the same listing — pins both to one walk.
    @ViewBuilder
    private func worldRow(_ entry: WorldListingEntry) -> some View {
        if let drawable = entry.sessions.last(where: \.hasGeometry) {
            Button {
                open(worldID: entry.worldID, sessionID: drawable.sessionID)
            } label: {
                HStack {
                    worldLabel(entry)
                    Spacer()
                    liveBadge(entry)
                }
            }
            .buttonStyle(.plain)
        } else {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    worldLabel(entry)
                    Text("None of these sessions has geometry, so there is nothing to open.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                liveBadge(entry)
            }
        }
    }

    @ViewBuilder
    private func liveBadge(_ entry: WorldListingEntry) -> some View {
        if entry.live {
            Text("live")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.accentColor.opacity(0.15), in: Capsule())
        }
    }

    /// A world the Tower lists with **no sessions**, drawn as text and not as a
    /// control.
    ///
    /// ## Why this is not a button
    ///
    /// It was one, briefly, and that was a regression a review caught before a
    /// person did. A world with no sessions has nothing to render: the Tower
    /// answers 404 on the render route, and `WorldRenderFetchError.absent` is
    /// classified retryable — correctly, for a world still being built — so the
    /// reader would be pushed into the Tower's 404 prose under a "Try again"
    /// button that can never succeed, because nothing reachable from this
    /// screen can give a world a session. On the real root that is 96 of 162
    /// rows.
    ///
    /// A control for an operation that cannot work is exactly what this pass
    /// set out to remove. So the row says what the world is and offers nothing.
    /// It keeps its id, because the id is the only reason a person would be
    /// looking inside this disclosure at all.
    private func emptyWorldRow(_ entry: WorldListingEntry) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            worldLabel(entry)
            Text("No sessions, so there is nothing to open.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    /// The title, the id and the session count. Shared so an openable world and
    /// a shell are described in the same words.
    private func worldLabel(_ entry: WorldListingEntry) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(entry.title)
                .font(.headline)
            // The id, restored as a caption.
            //
            // It was removed here on the grounds that 162 rows of hex is a wall
            // — which is true — and the removal made a world id from a Tower log
            // unfindable, because every row then read "Walk · <date>" and
            // nothing else told them apart. That is the worse failure: the id
            // is the one string a person arrives at this screen already
            // holding. It goes back where it was, as a caption under a human
            // title, which is the arrangement that was never the complaint.
            Text(entry.worldID)
                .font(.caption2.monospaced())
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .truncationMode(.middle)
            Text(WorldListingPresentation.sessionCount(entry.sessions.count))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// One session. A **control** only when the Tower says it has geometry.
    ///
    /// ## The same defect, one level down
    ///
    /// `emptyWorldRow` exists because a world with no sessions pushes into a
    /// guaranteed 404 under a Try again that cannot succeed. A session the
    /// Tower reports as `has_geometry: false` is the identical situation:
    /// `GET /worlds/{id}/render?session_id=` answers 404, `absent` is
    /// classified retryable, and nothing this screen can do will build geometry
    /// for a session that has none. `hasGeometry` was consulted here for
    /// exactly one thing — which shade of grey to paint the badge.
    ///
    /// So the row stops being a button and says what it is instead. The Tower's
    /// own state word stays on it, and for a session that is still open the
    /// sentence says the Tower may yet build one, because that is the one case
    /// where the answer can change — and it changes by the Tower building,
    /// which a refresh of this list will show, not by a Try again on a 404.
    @ViewBuilder
    private func sessionRow(entry: WorldListingEntry, session: WorldListingSession) -> some View {
        if session.hasGeometry {
            Button {
                open(worldID: entry.worldID, sessionID: session.sessionID)
            } label: {
                sessionLabel(session)
            }
            .buttonStyle(.plain)
        } else {
            VStack(alignment: .leading, spacing: 2) {
                sessionLabel(session)
                Text(WorldListingPresentation.noGeometryCaption(for: session) ?? "")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func sessionLabel(_ session: WorldListingSession) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                // When the walk happened, as the row's primary label.
                //
                // It used to be `session.sessionID` — a raw 32-character
                // hex string, in a monospaced font, as the **primary**
                // label of every session row. Nothing about a session is
                // less useful to a person choosing which walk to look at,
                // and nothing on the screen was louder. The id is shown for
                // the session actually opened, in the 3D viewer's Details
                // and in the canvas's Diagnostics.
                Text(WorldListingPresentation.sessionTitle(for: session))
                    .font(.subheadline)
                HStack(spacing: 6) {
                    Text(session.frameSource)
                    if let keyframes = WorldListingPresentation.keyframeCaption(for: session) {
                        Text("·")
                        Text(keyframes)
                    }
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
                // The final pass, when the Tower's own record says it did
                // not happen. `WorldListingSession.finalization` has been
                // decoded off this route since 2026-09-06 and read by
                // nothing; it is the one source of that word that does not
                // require pinning the session first, and it is what makes
                // "this walk is not everything it could be" visible while
                // the reader is still choosing which walk to open.
                if let solve = WorldListingPresentation.finalSolveCaption(for: session) {
                    Text(solve)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                // A complete walk whose photographic build failed: the badge
                // is short, so the whole sentence is here, where the reader
                // chooses which walk to open.
                if let photographic = WorldListingPresentation.photographicCaption(for: session) {
                    Text(photographic)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                // The session id, restored as a caption rather than as the
                // row's primary label.
                //
                // Removing it outright made two sessions of one world
                // started in the same minute indistinguishable, and left a
                // session id from a Tower log with nothing to match
                // against. The complaint it answered was that this string
                // was the LOUDEST thing on the row; as a tertiary caption
                // under a dated title, it is not.
                Text(session.sessionID)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            if let badge = WorldListingPresentation.stateBadge(for: session) {
                Text(badge)
                    .font(.caption2)
                    .foregroundStyle(badgeStyle(for: session))
            }
        }
    }

    /// Muted for a session with nothing to show, ordinary otherwise. The
    /// words carry the meaning; the colour only stops "No geometry" from
    /// reading as an alarm.
    private func badgeStyle(for session: WorldListingSession) -> HierarchicalShapeStyle {
        session.hasGeometry ? .secondary : .tertiary
    }

    /// Pin the world for the workspace behind, and push its 3D reconstruction.
    ///
    /// Both, in that order. The pin is what makes the workspace describe this
    /// world when the reader comes back to it, and what starts the geometry
    /// fetch that fills Diagnostics; the push is what they came for.
    ///
    /// The picker is **not** dismissed. It was, and the effect was that opening
    /// a world dropped the reader onto a workspace showing a sparse fragment
    /// gallery, with the 3D world still two taps away.
    private func open(worldID: String, sessionID: String?) {
        world.open(worldID: worldID, sessionID: sessionID)
        opened = WorldRenderTarget(worldID: worldID, sessionID: sessionID)
    }
}
