//
//  WorldBuilderWorkspaceView.swift
//  Glasses
//

import MWDATCore
import SwiftUI

// `MWDATCamera` is imported for `StreamState`'s cases specifically. The target
// builds with `SWIFT_UPCOMING_FEATURE_MEMBER_IMPORT_VISIBILITY`, under which a
// type's members are only visible where the defining module is imported — so
// `== .streaming` does not resolve through `GlassesConnection` alone. Gated to
// match the property it reads.
#if DEBUG
import MWDATCamera
#endif

/// The World Builder workspace: what the glasses see, and what the Tower builds
/// from it.
///
/// ## What this screen may and may not claim
///
/// Those two halves are still in different states, and the whole design of this
/// view follows from saying so plainly.
///
/// **The capture half is real.** The iPhone can start the glasses camera, and
/// frames genuinely reach the Tower — that is the V0.7 pipeline, measured and
/// working.
///
/// **The world half is now reported rather than absent.** The Tower declares a
/// World Builder contract over the socket and reports what it has built, and
/// `TowerWorldBuilderClient` decodes it. What this screen shows is therefore
/// whatever the Tower says, and nothing else.
///
/// **Starting capture is still not the same as starting a build.** The Tower's
/// web process writes frames to a capture and answers `frame_result`; the
/// reconstruction runs in a *separate* process reading that capture from disk.
/// Whether one is running is not visible from the phone, and this app must not
/// imply it started one.
///
/// Three consequences, each of which is a deliberate refusal:
///
/// - **No "Start Mapping" button.** A verb-labelled primary button is the
///   strongest readiness claim a UI can make, and tapping it does not start a
///   build. The control says what it does — it starts a capture session — and
///   the world panel reports what the Tower says came of it.
/// - **No placeholder metrics.** A field the Tower did not report is not drawn
///   at all, rather than drawn as "—": six redacted values read as *broken*
///   rather than as *absent*.
/// - **No fabricated geometry.** No mesh, no spinner outside the two states
///   where work genuinely is underway, and no single world map. The Tower now
///   does send points and poses — over HTTP, per segment, never down the
///   socket that carries the frames — and this screen draws exactly those,
///   each segment in its own frame because the Tower has not registered them
///   into a shared one. What it will not do is composite them into a room
///   nobody measured.
///
/// The live preview presents `GlassesConnection.latestCapturedFrame` through
/// the same `ViewfinderCard` the Home workspace uses. It does not open a second
/// camera session; there is exactly one stream and one owner.
struct WorldBuilderWorkspaceView: View {
    @ObservedObject var glasses: GlassesConnection
    @ObservedObject var tower: TowerClient

    /// The world-model boundary. Two implementations exist:
    /// `TowerWorldBuilderClient`, which is what the app graph builds, and
    /// `UnavailableWorldBuilderClient`, which reports exactly one thing — that
    /// this screen is not connected to a world builder at all.
    ///
    /// A `@StateObject` so it is constructed once per workspace installation
    /// rather than on every render. It holds no runtime resources — no camera,
    /// no socket, no DAT reference — so losing it when the cartridge is
    /// deselected loses nothing real. Anything that must outlive the workspace
    /// belongs on `ProjectManager`, not here.
    @StateObject private var world: WorldBuilderViewModel

    /// Whether the saved-worlds sheet is up. View state, so it lives here.
    @State private var isShowingWorlds = false

    /// The world whose interactive picture is up, or `nil`. Captured from
    /// `world.renderTarget` at the tap rather than read live, so a report
    /// that renames the live world mid-look does not swap the sheet's content
    /// under the reader.
    @State private var viewerTarget: WorldRenderTarget?

    /// Tells the Tower this workspace is on screen, so a builder may attach
    /// to a capture. View-owned on purpose — see the type's own doc comment
    /// for why it is not on the view model — and a `@StateObject` so the
    /// `stop` on disappear comes from the same object that sent `start`.
    @StateObject private var session: WorldBuilderSessionController

    /// The client is injected rather than constructed here, and owned by
    /// `ProjectManager`. See `CartridgeClients` for why: this `@StateObject` is
    /// destroyed on every cartridge switch, and a Tower-backed client holding a
    /// subscription and a partly-built world must not be.
    init(
        glasses: GlassesConnection,
        tower: TowerClient,
        client: any WorldBuilderClient,
        session: WorldBuilderSessionController? = nil
    ) {
        self.glasses = glasses
        self.tower = tower
        _world = StateObject(wrappedValue: WorldBuilderViewModel(client: client))
        _session = StateObject(wrappedValue: session ?? WorldBuilderSessionController())
    }

    /// Connectivity reaches the view model as a value, never as an object.
    ///
    /// This view genuinely needs `tower`: the capture control warns when the
    /// Tower is offline, because a `stream_start` sent while it is down is
    /// dropped and every frame after it is then suppressed for the whole
    /// session. So the observation is not a dead dependency here, and reading
    /// the status costs nothing extra — passing the *fact* rather than the
    /// client is what keeps the view model free of a reference it could act on.
    ///
    /// The three cartridge workspaces that have no capture control receive this
    /// `Bool` from `TowerReachabilityReader` instead, and do not observe the
    /// connection at all.
    private var isTowerReachable: Bool { tower.status == .online }

    var body: some View {
        VStack(spacing: 16) {
            header

            #if DEBUG
            glassesPanel
            #endif

            WorldCanvasView(
                state: world.state,
                availability: world.availability(isTowerReachable: isTowerReachable),
                explanation: world.unavailableExplanation(isTowerReachable: isTowerReachable),
                inspection: world.inspection,
                sessionBinding: world.sessionBinding,
                fragments: world.fragmentsModel,
                geometryChunks: world.geometryChunks,
                recentWorld: world.recentWorld,
                openRecent: { recent in
                    world.open(worldID: recent.worldID, sessionID: recent.sessionID)
                }
            )

            #if DEBUG
            captureControl
            #else
            HelperText("Capture is not available in this build.")
            #endif

            // Under the capture control in both configurations: the Tower's
            // gate applies to a Release phone's neighbour as much as to a
            // DEBUG phone's own capture. One line, worded as what was asked
            // for — `active` is intent, and whether a builder attached is the
            // canvas's report, not this line's.
            HelperText(session.footnote)
        }
        .sheet(isPresented: $isShowingWorlds) {
            WorldPickerView(world: world)
        }
        .sheet(item: $viewerTarget) { target in
            WorldRenderViewerView(target: target)
        }
        // The World Builder cartridge session: `start` on appearance and
        // whenever the socket comes back while on screen, `stop` on
        // disappearance. Nothing else on the phone starts or stops a builder.
        .onAppear { session.workspaceDidAppear(isTowerReachable: isTowerReachable) }
        .onDisappear { session.workspaceDidDisappear() }
        .onChange(of: isTowerReachable) { _, isReachable in
            session.towerReachabilityChanged(isReachable: isReachable)
        }
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("World Builder")
                    .font(.title2.weight(.semibold))
                Spacer()
                // Both read only, so neither is `#if DEBUG`: a Release build
                // with no camera can still look at what the Tower has stored.
                //
                // The picture button is disabled, not hidden, until the Tower
                // has named a world with geometry (`renderTarget`): a control
                // that appears from nowhere is one nobody looks for, and a
                // disabled one says "not yet" truthfully.
                Button {
                    viewerTarget = world.renderTarget
                } label: {
                    Label("Picture", systemImage: "cube.transparent")
                        .font(.subheadline)
                }
                .buttonStyle(.bordered)
                .disabled(world.renderTarget == nil)
                .accessibilityLabel("Interactive picture of the world")
                Button {
                    isShowingWorlds = true
                } label: {
                    Label("Saved worlds", systemImage: "archivebox")
                        .font(.subheadline)
                }
                .buttonStyle(.bordered)
            }
            Text("What the glasses see, and what the Tower reports it has built from that. Figures come from the Tower; absent ones are not drawn.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if case .inspecting(let worldID) = world.inspection {
                HStack(spacing: 8) {
                    Text("Looking at saved world \(worldID ?? "(unnamed)").")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button("Back to live") {
                        world.returnToLive()
                    }
                    .font(.footnote)
                    .buttonStyle(.bordered)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - DEBUG-only capture surface

// The camera path is DEBUG-only in the model, so the capture half of this
// workspace is gated to match. In Release the workspace still exists and still
// tells the truth about the world half — it simply has no capture controls,
// exactly as no other screen in the app does.
#if DEBUG
private extension WorldBuilderWorkspaceView {

    var isStreaming: Bool { glasses.cameraStreamState == .streaming }

    /// Includes the device-session states, not just the stream's. See
    /// `GlassesConnection.isCaptureEngaged` — deriving this from the stream
    /// alone left a window in which a session existed but the control still
    /// read "Start", and a tap in it did nothing observable.
    var isRunning: Bool { glasses.isCaptureEngaged }

    /// What the wearer currently sees.
    @ViewBuilder
    var glassesPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("What the glasses see")
            ViewfinderCard(
                frame: glasses.latestCapturedFrame,
                isStreaming: isStreaming,
                placeholderReason: placeholder
            )
        }
    }

    var placeholder: String {
        if !glasses.hasActiveDevice { return "Waiting for the glasses to become active." }
        if glasses.cameraPermissionStatus != .granted {
            return "Camera access is needed before a session can stream."
        }
        return "Start a capture session to see what the glasses see."
    }

    /// Labelled for what it actually does. See the type's doc comment for why
    /// this is not "Start Mapping".
    @ViewBuilder
    var captureControl: some View {
        VStack(spacing: 8) {
            if isRunning {
                Button {
                    glasses.stopCameraSession()
                } label: {
                    Label("Stop capture", systemImage: "stop.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.bordered)
                .disabled(glasses.cameraStreamState == .stopping)
            } else {
                Button {
                    glasses.startCameraSession()
                } label: {
                    Label("Start capture", systemImage: "play.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!glasses.hasActiveDevice)
            }

            // Neither string claims a build. The Tower reconstructs in a
            // separate process reading the capture from disk, and nothing on
            // the phone can see whether one is running — so the panel above,
            // which reports only what the Tower said, is where that question
            // is answered.
            HelperText(
                isRunning
                    ? "Frames are streaming to the Tower. What it builds from them is reported above."
                    : "Streams frames to the Tower. What it builds from them is reported above."
            )

            if !glasses.hasActiveDevice && !isRunning {
                HelperText("Waiting for the glasses to become active.")
            } else if glasses.cameraPermissionStatus == .denied && !isRunning {
                // Advice, not a `.disabled` condition — see the equivalent
                // branch in `HomeWorkspaceView.sessionControl`.
                HelperText("Camera access is not granted. Allow it under Connections, then start capture.")
            } else if tower.status != .online {
                // The Tower must be online *before* capture starts: a
                // `stream_start` sent while it is offline is dropped, and every
                // frame after it is then suppressed for the whole session.
                // Advice rather than a new `.disabled` condition, so the
                // control's semantics stay what they were.
                HelperText("The Tower is not connected. Frames from this session would not reach it.")
            }
        }
    }
}
#endif
