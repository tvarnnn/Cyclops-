//
//  WorldCanvasView.swift
//  Glasses
//

import Foundation
import SwiftUI

/// The world-visualization container: the place a spatial model will eventually
/// be drawn, and today the place the app explains that there is not one.
///
/// It renders `WorldModelState` and nothing else, so it cannot show geometry
/// the Tower did not send. Every state is now reachable:
/// `TowerWorldBuilderClient` maps the Tower's `model_state` onto them.
///
/// ## The 3D world is the primary content, and it is not drawn here
///
/// A saved world's main content is the Tower's own composed 3D page, opened by
/// `reconstructionCard` into `WorldRenderViewerView`'s `WKWebView`. That is the
/// only surface in this target that draws 3D, and it is deliberately the only
/// one: the Tower composes the world, colours it and decimates it to a mobile
/// budget, and a second renderer on the phone would be a second answer able to
/// disagree with the first.
///
/// The sparse per-segment gallery below is a **diagnostic**, behind the
/// Diagnostics disclosure. It used to be the primary content of a saved world,
/// which made opening one an act of debugging.
///
/// Deliberately absent, and each for a reason:
///
/// - **No 3D framework in this target.** SceneKit, RealityKit and Metal are
///   still absent. The reason is no longer "nothing to render" and no longer
///   "`up_axis` is unknown" — the Tower renders the world itself, in a page
///   this app already shows. Adding one here would duplicate that.
/// - **No single world map drawn by this app.** Each fragment in the
///   diagnostics gallery gets its own frame, its own scale and its own box
///   until the Tower registers them. Per-segment scale disagrees by up to ~87x
///   on a real walk; one shared scale would draw something that looks like a
///   room and means nothing. Compositing the largest component into one world
///   is the Tower's job, and the render page is where it is done.
/// - **No animated placeholder.** A drifting point cloud would look like
///   progress and be a fabrication.
/// - **No spinner in the unsupported state.** A spinner claims work is
///   underway; nothing is underway.
///
/// ## Availability outranks state
///
/// The `availability` parameter is consulted *before* the domain state. If the
/// Tower cannot serve this cartridge — no contract, a contract this build does
/// not implement, or an unreachable Tower — then no `WorldModelState` is worth
/// drawing, and the shared `CartridgeStatePanel` says which of those three it
/// is. That ordering lives in one place (`CartridgeAvailability.forcedPhase`)
/// so all four cartridges obey it identically.
struct WorldCanvasView: View {
    let state: WorldModelState
    let availability: CartridgeAvailability
    /// Composed by `WorldBuilderViewModel`, not here — so the shared layer owns
    /// the ordering of the two sentences and all four cartridges join them the
    /// same way.
    ///
    /// **This is not a performance fix, and it should not be read as one.** This
    /// view sits under `WorldBuilderWorkspaceView`, which observes
    /// `GlassesConnection` because it draws the viewfinder, so that body runs at
    /// the 24 Hz capture rate during a session — and it evaluates this argument
    /// on every one of those. The composition moved up a level; it did not
    /// disappear. Two small string allocations per body, against frame encoding
    /// happening in the same window, is not obviously worth caching, but it is
    /// worth *measuring* on the Mac rather than assuming either way.
    ///
    /// What genuinely was removed is the three other workspaces' invalidation at
    /// the Tower's reply rate — see `TowerReachabilityReader`. That one was a
    /// dead dependency, so removing it cost nothing.
    let explanation: String
    var inspection: WorldInspectionMode = .live
    /// What the client established about whose world this is.
    ///
    /// It changes no figure and hides nothing — the gate in
    /// `TowerWorldBuilderClient` has already decided what `state` may be. All
    /// this does is let the waiting state say *why* it is waiting, which is
    /// the sentence nobody could see on 2026-08-24.
    var sessionBinding: WorldSessionBinding = .none

    /// The segments the Tower's manifest names, and their points and poses.
    ///
    /// Defaulted to empty so that the states with nothing to draw — and the
    /// previews, and any caller that only wants the summary rows — construct
    /// this view exactly as they did before. An empty model no longer implies
    /// anything about the world: what an empty gallery *means* is decided by
    /// `presentation.account`, from the fetch's own state.
    var fragments = WorldFragmentsModel(segments: [])
    var geometryChunks: [String: WorldSegmentChunk] = [:]

    /// The derived half of the screen: the stage word, the 3D ladder, the
    /// account for a gallery with nothing in it, the final-solve sentence and
    /// what survived. Built by `WorldBuilderViewModel.presentation` and by
    /// nothing else, so this view composes no judgment of its own.
    ///
    /// Defaulted to `.empty` so previews and the states with no world in them
    /// construct this view exactly as they did before.
    var presentation: WorldPresentation = .empty

    /// Opens the 3D world. `nil` in a context that cannot present a sheet —
    /// previews, and any future embedding — in which case the primary control
    /// is not drawn at all rather than drawn and inert.
    var openReconstruction: ((WorldRenderTarget) -> Void)? = nil

    /// Whether the Diagnostics disclosure is open.
    ///
    /// View state, and deliberately not remembered: every time this screen
    /// opens it opens as the product surface. The sparse solver output behind
    /// it is one tap away and nothing in it was removed — see
    /// `WorldFragmentsView`'s own doc comment — but a screen that reopens
    /// expanded is a screen that has quietly gone back to being a debugger.
    @State private var isShowingDiagnostics = false

    /// The stored world the Tower offered in place of a live one, when
    /// following live and nothing is live. Drawn in `.idle` as one line with
    /// an Open action and nowhere else — its rows and gallery arrive only
    /// after `openRecent` pins it, under the "Saved world" heading.
    var recentWorld: WorldRecentReference? = nil
    var openRecent: ((WorldRecentReference) -> Void)? = nil

    var body: some View {
        if let forcedPhase = availability.forcedPhase {
            CartridgeStatePanel(
                title: "What the Tower builds",
                phase: forcedPhase,
                explanation: explanation,
                futureDescription: Self.futureDescription
            )
        } else {
            worldPanel
        }
    }

    /// Shown only where the panel has nothing else to say. It describes the
    /// fields this panel draws **when the Tower reports them** — which is a
    /// claim about this build, not a promise about the Tower's roadmap.
    private static let futureDescription = """
        When the Tower reports a world, this panel shows the keyframes it kept, \
        how well it is tracking, whether its scale is relative or unknown, and \
        what it built. Fields it does not report are not drawn.
        """

    // MARK: The panel for a Tower that can actually answer

    private var worldPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(inspection.isInspecting ? "Saved world" : "What the Tower builds")

            VStack(alignment: .leading, spacing: 12) {
                content
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        }
    }

    @ViewBuilder
    private var content: some View {
        switch state {
        case .unsupported(let reason):
            unsupported(reason)

        case .idle:
            headline("No world yet", systemImage: "cube.transparent")
            // Not "start a capture session to begin building a world". The
            // Tower reaches this state by having no world to report, and
            // capture alone does not create one — reconstruction runs in a
            // separate Tower process reading the capture from disk, which
            // this app can neither start nor see. Promising a world in
            // exchange for a tap would be a claim about the other machine.
            detailText(idleDetail)
            if let recent = recentWorld, !inspection.isInspecting {
                recentWorldLine(recent)
            }

        case .awaitingFirstUpdate:
            // The one honest use of a progress indicator: frames really are
            // going out and the Tower really has not answered yet.
            HStack(spacing: 10) {
                ProgressView()
                Text(waitingHeadline)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            if let detail = waitingDetail {
                detailText(detail)
            }

        case .receiving(let snapshot):
            stageHeadline(fallback: "Building", systemImage: "cube")
            worldName(snapshot)
            reconstructionCard
            WorldSummaryView(snapshot: snapshot, isLive: true)
            diagnostics

        case .finalizing(let snapshot, let buildInProgress):
            // ## A spinner only on evidence
            //
            // This once drew a `ProgressView` beside "Finishing the world…"
            // under a comment reading *"Progress is honest here — the Tower
            // is genuinely working"*, and the Tower of the day said it could
            // not know that: the writer lock was released before `build()`
            // ran, so *"a build in progress is indistinguishable on disk from
            // one that never started and from one that crashed"*. If the
            // builder crashed, the spinner span forever. It was removed.
            //
            // Since 2026-09-06 the live builder keeps the lock through
            // finalization and the Tower sends `build_in_progress: true` on
            // the evidence of a live process holding it. That is the one case
            // where an indicator is a report rather than an assertion, and it
            // is drawn for that case alone. `null` — an older record — keeps
            // the staleness sentence, which is what that state means.
            stageHeadline(fallback: "Finalizing", systemImage: "cube")
            worldName(snapshot)
            if buildInProgress == true {
                HStack(spacing: 10) {
                    ProgressView()
                    Text("The Tower is finishing this world.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            reconstructionCard
            WorldSummaryView(snapshot: snapshot, isLive: false)
            if buildInProgress == true {
                detailText("A live process holds this world's writer lock and its session has stopped: the final solve and the final build are running, and these figures are not final.")
            } else {
                detailText("Capture has ended and these figures are not final. The Tower does not report whether a build is running, so this app cannot say whether one is.")
            }
            diagnostics

        case .finalized(let snapshot):
            stageHeadline(fallback: "Saved", systemImage: "cube.fill")
            worldName(snapshot)
            reconstructionCard
            finalSolveNote
            WorldSummaryView(snapshot: snapshot, isLive: false)
            diagnostics

        case .interrupted(let snapshot, let reason):
            // Its own state, never "failed" and never a finished world. The
            // Tower's reason is the detail, verbatim; the rows, the gallery
            // and — through `renderTarget`, which the coordinates for this
            // snapshot still earn — the picture button all stay, because the
            // geometry exists whatever happened to the process.
            stageHeadline(fallback: "Interrupted", systemImage: "exclamationmark.triangle")
            worldName(snapshot)
            // What survived and what can be done about it, before the Tower's
            // own reason. There is no rebuild route on the Tower — see
            // `WorldRecoverability` — so this is a sentence and never a button.
            if let recoverability = presentation.recoverability,
               let sentence = recoverability.sentence {
                detailText(sentence)
            }
            reconstructionCard
            finalSolveNote
            // The Tower's own words for what happened, kept verbatim. Under the
            // recovery sentence rather than over it: "your world is still here"
            // is what a reader needs first, and the process's error text is
            // what they need second.
            detailText(reason)
            WorldSummaryView(snapshot: snapshot, isLive: false)
            diagnostics

        case .failed(let failure):
            // Two different claims share this state. A `.transport` or
            // `.timedOut` failure is about the *channel* — the Tower did not
            // answer this screen's subscription — and says nothing about
            // whether a world is being built; headlining it "World building
            // failed" told a wearer whose walk was fine that it was not.
            // Everything else is the Tower's own report of a failure.
            switch failure.kind {
            case .transport, .timedOut:
                headline("World Builder is not reporting", systemImage: "antenna.radiowaves.left.and.right.slash")
            case .notSupported, .towerReportedFailure, .undecodableResponse:
                headline("World building failed", systemImage: "exclamationmark.triangle.fill")
            }
            detailText(failure.message)
        }
    }

    /// The `.idle` sentence. With a stored world on offer the second half is
    /// dropped: "what it builds is its own to start" is true but the line
    /// under it is about to say what it *did* build.
    private var idleDetail: String {
        if recentWorld != nil, !inspection.isInspecting {
            return "Nothing is being built right now."
        }
        return "The Tower has not reported a world. Frames from a capture session reach it; what it builds from them is its own to start."
    }

    /// One line, one action. `Last saved world: <title> · <state>` names what
    /// the Tower offered without drawing any of it; Open pins it, and the
    /// canvas then says "Saved world" over everything it shows.
    private func recentWorldLine(_ recent: WorldRecentReference) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("Last saved world: \(recent.title) · \(recent.stateLabel)")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .truncationMode(.middle)
            Spacer(minLength: 8)
            if let openRecent {
                Button("Open") { openRecent(recent) }
                    .font(.footnote)
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Open the last saved world")
            }
        }
    }

    /// What the wait is, in the two cases the phone can tell apart.
    private var waitingHeadline: String {
        switch sessionBinding {
        case .none, .bound:
            return "Waiting for the Tower's first world update…"
        case .awaiting, .foreign:
            return "Frames are reaching the Tower."
        }
    }

    /// The sentence that was missing on 2026-08-24, when the phone showed a
    /// frozen world beside a live camera and nothing said which capture the
    /// figures belonged to.
    ///
    /// Both strings are claims the phone can actually support. `.foreign` means
    /// a snapshot arrived and described a capture that is not this one, so the
    /// stronger sentence is earned. `.awaiting` means a capture is open and the
    /// Tower has resolved no session for it at all — which is true in the
    /// seconds after Start, and stays true if nothing ever attaches a builder.
    private var waitingDetail: String? {
        switch sessionBinding {
        case .none, .bound:
            return nil
        case .awaiting:
            return "Nothing is building a world from them yet."
        case .foreign:
            // "Could not be matched", not "is a different capture". Both the
            // 2026-08-24 case (a world finished an hour ago) and the narrow one
            // where this phone's own capture ended before DAT reported the stop
            // arrive here, and only the weaker sentence is true of both.
            return """
                The world it is reporting could not be matched to this \
                session's capture, so nothing here describes it yet.
                """
        }
    }

    // MARK: The primary thing on screen

    /// The best available 3D reconstruction, and a control that opens it.
    ///
    /// ## Why this is above the rows and the gallery
    ///
    /// The one surface in this app that can show 3D is the `WKWebView` in
    /// `WorldRenderViewer`, which renders the Tower's own composed page. It was
    /// reachable only through a small bordered "Picture" button in the
    /// workspace header, while the *primary* content of a saved world was the
    /// sparse fragment gallery — per-segment point clouds in unregistered
    /// frames, drawn top-down, captioned with the solver's registration
    /// vocabulary. That is a reconstruction debugger, and it was the default
    /// experience of opening a saved world.
    ///
    /// So the ladder — final 3D world, then partial 3D world, then a truthful
    /// degraded state — is drawn here, first, in every state that has a world.
    /// The gallery moved behind Diagnostics and lost nothing.
    ///
    /// The three cases are genuinely different and are drawn differently.
    /// `.unavailable` is *the Tower named no address*, which is not the same as
    /// the render route answering 404 — that happens later, inside the viewer,
    /// and arrives with the Tower's own detail attached.
    @ViewBuilder
    private var reconstructionCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let target = presentation.reconstruction.target, let open = openReconstruction {
                Button {
                    open(target)
                } label: {
                    Label(presentation.reconstruction.actionTitle, systemImage: "cube.transparent")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)
                .accessibilityLabel("Open the 3D world")
                if let note = reconstructionNote {
                    Text(note)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else if let reason = reconstructionUnavailableReason {
                // No control at all, and a sentence saying why. A disabled
                // button here would be a control with no explanation beside it,
                // which is what the header's "Picture" button was.
                Text(reason)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// `World <id>` and, when one is known, `Session <id>`, or `nil` when the
    /// Tower has named neither. Built here rather than in the builder above for
    /// the reason `reconstructionNote` gives.
    private var identifiers: String? {
        var lines: [String] = []
        if let worldID = state.snapshot?.worldID { lines.append("World \(worldID)") }
        if let sessionID = presentation.reconstruction.target?.sessionID {
            lines.append("Session \(sessionID)")
        }
        return lines.isEmpty ? nil : lines.joined(separator: "\n")
    }

    /// The `.partial` note, or `nil`. Read through a property rather than bound
    /// inside the builder above — the pattern this codebase settled on after a
    /// result-builder block with a binding in it caused trouble in Product
    /// Shell V2.
    private var reconstructionNote: String? {
        if case .partial(_, let note) = presentation.reconstruction { return note }
        return nil
    }

    private var reconstructionUnavailableReason: String? {
        if case .unavailable(let reason) = presentation.reconstruction { return reason }
        return nil
    }

    /// The final solve's sentence, when there is one worth making. `.solved`
    /// returns none: a world that finished properly does not need a line saying
    /// it finished properly.
    ///
    /// This field — `lifecycle.finalization.final_solve` — has been decoded
    /// since 2026-09-06 and read by no view until now. It is `null` on the
    /// field world, meaning the final solve never ran, and the phone presented
    /// that world as a finished one.
    @ViewBuilder
    private var finalSolveNote: some View {
        if let sentence = presentation.finalSolve.sentence {
            Text(sentence)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Diagnostics

    /// The sparse solver output, one tap away and never in the way.
    ///
    /// Everything that used to be on the normal surface is in here and nothing
    /// was deleted: the fragment gallery, the per-segment point counts, the
    /// Tower's registration words and refusal prose, the coverage tokens, the
    /// segment origins and the segment count. Debugging observability is the
    /// point of the exercise — it just stopped being the first thing a person
    /// sees when they open a saved world.
    ///
    /// Shown in the four states that carry a snapshot and in no others.
    /// `.idle` and `.failed` have no world to have geometry for, and offering a
    /// diagnostics container under them would offer a container where there is
    /// not even a world.
    private var diagnostics: some View {
        DisclosureGroup("Diagnostics", isExpanded: $isShowingDiagnostics) {
            VStack(alignment: .leading, spacing: 12) {
                Text("The solver's own view: geometry per tracking segment, in the frames the solver placed them in. Not the world above.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)

                // The Tower's own page, opened straight into its diagnostics
                // rendering (`?view=diagnostics`, served since 2026-09-09 by
                // `routes/geometry.py::world_render`).
                //
                // The entry point lives **here** rather than as a switch inside
                // the viewer. A switch there would refetch several megabytes to
                // change one token in a script already on screen, and the
                // Tower's page carries its own instant toggle anyway. Choosing
                // the rendering at the moment a page is opened costs a fetch
                // that was going to happen regardless, and it puts the solver's
                // view where the rest of the solver's output already is.
                if let target = presentation.reconstruction.target, let open = openReconstruction {
                    Button {
                        open(target.showing(.diagnostics))
                    } label: {
                        Label("Open the solver's 3D view", systemImage: "scope")
                            .font(.footnote)
                    }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Open the solver's 3D view")
                }

                if let snapshot = state.snapshot {
                    WorldSummaryView(snapshot: snapshot, isLive: false, scope: .solver)
                }

                // The raw identifiers, kept reachable. They used to be on the
                // ordinary surface — the workspace header's "Looking at saved
                // world <32 hex characters>" and the picker's session rows,
                // where a session id was the row's *primary* label. They are
                // what the Tower's own logs are keyed by, so a person
                // diagnosing a walk needs them; they are also the least
                // meaningful thing on a screen about a room.
                if let identifiers {
                    Text(identifiers)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.tertiary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }

                WorldFragmentsView(
                    model: fragments, chunks: geometryChunks, account: presentation.account
                )
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 8)
        }
        .font(.subheadline)
        .accessibilityIdentifier("world-diagnostics")
    }

    /// The state that is actually reachable today.
    @ViewBuilder
    private func unsupported(_ reason: String) -> some View {
        headline("Nothing yet", systemImage: "cube.transparent")
        detailText(reason)
        // One sentence instead of a grid of empty metric tiles. It teaches the
        // product concept and creates no dead UI to explain away later.
        Text(Self.futureDescription)
            .font(.footnote)
            .foregroundStyle(.tertiary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func headline(_ text: String, systemImage: String) -> some View {
        Label(text, systemImage: systemImage)
            .font(.headline)
    }

    /// The stage word, or the caller's fallback when there is no stage.
    ///
    /// The four world states used to headline themselves with the world's
    /// *name* — `snapshot.name ?? "World"` — which meant the largest words on
    /// the screen said nothing about what the world was doing, and on a Tower
    /// that names nothing (156 of the real root's 162 worlds) said "World".
    /// The stage says what is happening in one word from a vocabulary a person
    /// already has: Mapping, Building, Improving, Finalizing, Saved, Partial,
    /// Interrupted, Needs retry. The name moved to the line under it, where a
    /// name belongs.
    ///
    /// The fallback is per-state and is what this build drew before there was a
    /// stage, so a `WorldPresentation` that was never built — a preview, a
    /// caller that passes only a state — is unchanged.
    private func stageHeadline(fallback: String, systemImage: String) -> some View {
        headline(
            presentation.stage?.label ?? fallback,
            systemImage: presentation.stage?.systemImage ?? systemImage
        )
    }

    /// The Tower's own name for this world, under the stage word, when it has
    /// one. Never the id — that is in Diagnostics.
    @ViewBuilder
    private func worldName(_ snapshot: WorldSnapshot) -> some View {
        if let name = snapshot.name, !name.isEmpty {
            Text(name)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
    }

    /// Named `detailText` rather than `body` on purpose: a method sharing the
    /// base name of the `View` protocol's own `body` requirement is legal but
    /// invites confusion at exactly the place a reader is looking for the real
    /// one.
    private func detailText(_ text: String) -> some View {
        Text(text)
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// Renders only the fields the Tower actually reported.
///
/// Every value is optional and an absent one is simply not drawn, rather than
/// drawn as "—". That is the difference between a panel that is early and a
/// panel that looks broken.
///
/// Reached from `.receiving`, `.finalizing`, `.finalized` and `.interrupted`,
/// all of which `TowerWorldBuilderClient` produces from real Tower snapshots.
struct WorldSummaryView: View {
    /// Which half of the Tower's figures this instance draws.
    ///
    /// Two rows moved out of the ordinary panel and into Diagnostics, and only
    /// two: **Segment origins** and **Segments**. Both are statements about how
    /// the solver cut the walk up, not about the room — "36 segment origins" is
    /// the count of coordinate frames it opened — and a person looking at a
    /// saved world has no use for either. Everything else stayed where it was,
    /// including the anchors-only sentence and the tracking-restart sentence,
    /// because those explain something a reader can otherwise see and not
    /// understand: why there is no path length.
    enum Scope: Equatable {
        /// Everything but the two solver rows.
        case product
        /// Only the two solver rows, plus the chain-break count that has never
        /// been drawn anywhere.
        case solver
    }

    let snapshot: WorldSnapshot
    let isLive: Bool
    var scope: Scope = .product

    @ViewBuilder
    var body: some View {
        if scope == .solver {
            solverRows
        } else {
            productRows
        }
    }

    /// The solver's own cut of the walk. Diagnostics only.
    private var solverRows: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Beside the pose count, never folded into it.
            // `world_builder.status/2026-08-25` exists because the two were
            // being added: an anchor is a segment's origin — identity rotation,
            // zero translation — and 36 of them are not 36 camera positions.
            if let anchors = snapshot.trajectory.posesAnchor {
                row("Segment origins", "\(anchors)")
            }
            if let segments = snapshot.trajectory.segments {
                row("Segments", "\(segments)")
            }
            // Never drawn anywhere before. The Tower counts these separately
            // from tracking losses precisely so the two are not confused; see
            // `WorldTrajectoryReport.segments`.
            if let breaks = snapshot.trajectory.chainBreaks {
                row("Solve-chain breaks", "\(breaks)")
            }
            if let restarts = snapshot.trajectory.trackingRestarts {
                row("Tracking restarts", "\(restarts)")
            }
        }
    }

    private var productRows: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let keyframes = snapshot.keyframeCount {
                row("Keyframes", "\(keyframes)")
            }
            if snapshot.tracking != .unavailable {
                row("Tracking", snapshot.tracking.displayName)
            }
            if snapshot.calibration != .unknown {
                row("Calibration", snapshot.calibration.displayName)
            }
            if snapshot.scale != .unknown {
                row("Scale", snapshot.scale.displayName)
                // The monocular-depth rule from docs/modules/WORLD-BUILD.md,
                // enforced at the point of display: an inferred figure is never
                // shown without saying it is an estimate.
                if snapshot.scale.isEstimate {
                    caption(snapshot.scale.explanation)
                }
            }

            geometryRows
            trajectoryRows

            if snapshot.persistence != .unknown {
                row("Storage", snapshot.persistence.displayName)
            }

            if !isLive {
                Text("Capture has ended.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// The Tower's representation, quoted rather than interpreted.
    ///
    /// The count is never shown on its own. "18,432" beside the word
    /// "Geometry" invites a reader to supply their own unit — points, faces,
    /// landmarks — and the Tower has not said which. Shown next to the label it
    /// chose, the number means whatever that label means, which is the only
    /// true reading available.
    @ViewBuilder
    private var geometryRows: some View {
        if snapshot.geometry.hasReport {
            row("Geometry", geometryValue)
        }
    }

    /// Composed outside the `@ViewBuilder` above rather than with a local `let`
    /// inside it — the pattern this codebase settled on after a result-builder
    /// block with a binding in it caused trouble in Product Shell V2.
    private var geometryValue: String {
        let name = snapshot.geometry.representation ?? "Unnamed representation"
        guard let count = snapshot.geometry.elementCount else { return name }
        return "\(count) · \(name)"
    }

    /// Path length is shown only where the scale permits it to mean a distance.
    @ViewBuilder
    private var trajectoryRows: some View {
        if let poses = snapshot.trajectory.poseCount {
            row("Camera poses", "\(poses)")
        }
        // "Segment origins" and "Segments" used to sit here, beside the pose
        // count. They are the solver's own cut of the walk rather than
        // anything about the room, and they moved to `solverRows` — behind
        // Diagnostics — where they are still shown, still beside the pose count
        // in the reader's memory, and no longer in the way of a person who
        // opened a saved world to look at it.
        if snapshot.trajectory.isAnchorsOnly {
            // The uncalibrated outcome, said plainly. No intrinsics exist for
            // this camera yet, so the backend that solves poses withholds every
            // one — and "0 camera poses, 36 segment origins" is a true and
            // uninterpretable pair without this line.
            caption("""
                No camera position was reconstructed. Each origin marks where \
                tracking restarted, not where the camera was.
                """)
        } else if let restarts = snapshot.trajectory.trackingRestarts, restarts > 0 {
            // Why there is no path length beside these figures. The Tower
            // refuses one across a segment break, because poses either side of
            // it share no coordinate frame.
            //
            // The number is the Tower's COUNT of tracking losses. This line
            // rendered `segments - 1` until 2026-09-09, which on that walk read
            // "Tracking restarted 121 times" against 64 actual losses — the
            // other 57 segments were opened by the solver failing to place a
            // keyframe while tracking was fine. See
            // `WorldTrajectoryReport.segments`.
            caption("""
                Tracking restarted \(restarts) time\(restarts == 1 ? "" : "s"). \
                Distances either side of a restart are not comparable, so the \
                Tower reports no path length.
                """)
        } else if let segments = snapshot.trajectory.segments, segments > 1 {
            // A Tower that does not count restarts. Say only what the segment
            // count actually supports — that the pieces do not share one frame
            // — and not how many times tracking was lost, which this number
            // cannot tell us.
            caption("""
                This walk is in \(segments) pieces that do not share one \
                coordinate frame, so the Tower reports no path length.
                """)
        }
        if snapshot.trajectory.labelledFigureDisplayable, let length = snapshot.trajectory.pathLength {
            // The Tower's unit, always. `labelledFigureDisplayable` refuses
            // when there is none, so `ReportedFigure` cannot be reached here
            // with a bare number.
            row("Path length", ReportedFigure.format(length, unit: snapshot.trajectory.pathLengthUnit))
            // Said for every scale that is not a plain measurement, not only
            // for estimates: `.relative` needs the sentence most of all,
            // because "2.9 world units" is the one figure on this panel a
            // reader might otherwise take for a distance.
            if snapshot.trajectory.scale != .measuredMetric {
                caption(snapshot.trajectory.scale.explanation)
            }
        }
    }

    private func caption(_ text: String) -> some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(.tertiary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            Text(value)
                .font(.subheadline.weight(.medium))
                .monospacedDigit()
        }
        .accessibilityElement(children: .combine)
    }
}

#Preview("Unsupported") {
    WorldCanvasView(
        state: .unsupported(reason: UnavailableWorldBuilderClient.reason),
        availability: .noContract,
        explanation: UnavailableWorldBuilderClient.reason
    )
    .padding()
}
