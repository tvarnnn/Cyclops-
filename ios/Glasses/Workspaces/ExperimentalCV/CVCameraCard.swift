//
//  CVCameraCard.swift
//  Glasses
//

import MWDATCore
import SwiftUI

// `MWDATCamera` is imported for `StreamState`'s cases specifically. The target
// builds with `SWIFT_UPCOMING_FEATURE_MEMBER_IMPORT_VISIBILITY`, under which a
// type's members are only visible where the defining module is imported — so
// `== .stopping` does not resolve through `GlassesConnection` alone. Gated to
// match the property it reads.
#if DEBUG
import MWDATCamera
#endif

/// The camera, from inside the Experimental CV Lab.
///
/// ## Why this exists
///
/// An experiment is fed by the glasses camera, and until this card a person
/// running one started the camera on Home, walked to the Lab to choose an
/// experiment, and walked back to Home to stop it. Three screens for one
/// question. The card puts the four controls a bench run needs — Start, Pause,
/// Resume, Stop — beside the experiment they feed.
///
/// ## What each control actually does
///
/// **Start and Stop are the app's one camera path.** They call
/// `GlassesConnection.startCameraSession()` and `stopCameraSession()`, the
/// same two methods Home and World Builder call, on the same single
/// `GlassesConnection`. There is no second pipeline and this card owns nothing.
///
/// **Pause and Resume are not camera operations.** DAT offers no
/// app-initiated camera pause — `StreamState.paused` is something the glasses
/// do on their own — and a `stream_stop` would end the capture lineage on the
/// Tower. So they toggle `TowerClient.isFrameSendingPaused`, a gate on the one
/// hop this app fully owns: the camera keeps running, the socket and the
/// stream bracket stay up, and frames are held on the phone. The Tower sees
/// silence, and its `source.receiving_frames` turns false after five seconds,
/// which is what the Lab's own liveness reads. The status line says "held on
/// phone" for exactly this reason: the glasses are not paused, and the card
/// must not say they are.
///
/// ## Debug and Release
///
/// The capture surface is `#if DEBUG` in the model (`ProjectManager`), so the
/// controls are gated to match, exactly as Home's are. A Release build gets
/// the same sentence Home and World Builder show in its place.
///
/// ## Runtime ownership
///
/// Observes both objects and constructs neither. This is the one leaf in the
/// Lab that observes `GlassesConnection`, which publishes at the 24 Hz capture
/// rate; the workspace above holds it as a plain reference so that rate stops
/// here.
struct CVCameraCard: View {
    @ObservedObject var glasses: GlassesConnection
    @ObservedObject var tower: TowerClient

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("Camera")

            VStack(spacing: 12) {
                #if DEBUG
                statusLine
                controls
                advice
                if let refusal = glasses.lastCaptureStartRefusal {
                    FailureBanner(text: Self.refusalLine(refusal))
                }
                #else
                // Mirrors Home and World Builder. The whole capture surface is
                // DEBUG-only, so a Release build must not imply a control it
                // does not have.
                HelperText("Capture is not available in this build.")
                #endif
            }
            .frame(maxWidth: .infinity)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        }
    }
}

// MARK: - DEBUG-only capture surface

// The camera path — `startCameraSession`, `cameraStreamState`,
// `isCaptureEngaged`, `lastCaptureStartRefusal` — is DEBUG-only in the model,
// so every control that touches it is gated to match.
#if DEBUG
private extension CVCameraCard {

    /// Includes the device-session states, not just the stream's — see
    /// `GlassesConnection.isCaptureEngaged` for the window that closes.
    var isRunning: Bool { glasses.isCaptureEngaged }

    var isStopping: Bool { glasses.cameraStreamState == .stopping }
    var isStreaming: Bool { glasses.cameraStreamState == .streaming }

    /// Frames are being held on the phone. A fact about `TowerClient`, not
    /// about the glasses — see the type's doc comment.
    var isHeld: Bool { tower.isFrameSendingPaused }

    /// One line, one truthful word about where the frames are.
    var statusLine: some View {
        HStack(spacing: 8) {
            Image(systemName: statusLevel.symbol)
                .foregroundStyle(statusLevel.tint)
                .symbolRenderingMode(.hierarchical)
                .accessibilityHidden(true)
            Text(statusValue)
                .font(.subheadline.weight(.medium))
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Camera")
        .accessibilityValue(statusValue)
    }

    /// The hold outranks the stream state while the camera is up, because it
    /// is the one condition under which "Streaming" would be the most
    /// convincing untrue thing on the card: the glasses are streaming, and
    /// nothing is leaving the phone.
    var statusValue: String {
        if isHeld && isRunning { return "Paused (held on phone)" }
        switch glasses.cameraStreamState {
        // A device session can exist before the stream has reached
        // `.starting` — the window `isCaptureEngaged` closes — and "Stopped"
        // beside a Stop button would be two claims at once.
        case .stopped: return isRunning ? "Starting" : "Stopped"
        case .waitingForDevice, .starting: return "Starting"
        case .streaming: return "Streaming"
        // The glasses' own pause — a temple press, or heat. Told apart from
        // the phone's hold above because they are undone by different parties.
        case .paused: return "Paused by the glasses"
        case .stopping: return "Stopping"
        @unknown default: return "Unknown"
        }
    }

    var statusLevel: StatusLevel {
        if isHeld && isRunning { return .working }
        switch glasses.cameraStreamState {
        case .stopped: return isRunning ? .working : .idle
        case .waitingForDevice, .starting, .paused, .stopping: return .working
        case .streaming: return .ok
        @unknown default: return .idle
        }
    }

    /// Start when nothing is engaged; Stop beside Pause or Resume otherwise.
    ///
    /// Stop is `.bordered` and Start is `.borderedProminent`, matching Home's
    /// session control. Resume is prominent because while frames are held it
    /// is the action a person came back to take; Pause is not, because
    /// stopping the frames is never the primary thing to do with a running
    /// experiment.
    @ViewBuilder
    var controls: some View {
        if isRunning {
            HStack(spacing: 12) {
                Button {
                    glasses.stopCameraSession()
                } label: {
                    Label("Stop camera", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.bordered)
                .disabled(isStopping)

                if isHeld {
                    Button {
                        tower.resumeFrameSending()
                    } label: {
                        Label("Resume frames", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 6)
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    Button {
                        tower.pauseFrameSending()
                    } label: {
                        Label("Pause frames", systemImage: "pause.fill")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 6)
                    }
                    .buttonStyle(.bordered)
                    // Only while frames exist to hold. A hold set during
                    // "Starting" outlives a start that then fails: nothing
                    // sends `stream_stop` for a camera that never opened, so
                    // the gate would stay closed into the next session, where
                    // Home has no control that shows it.
                    .disabled(!isStreaming)
                }
            }
            .font(.headline)
        } else {
            Button {
                glasses.startCameraSession()
            } label: {
                Label("Start camera", systemImage: "play.fill")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!glasses.hasActiveDevice)
        }
    }

    /// One sentence, in dependency order, the way Home names a single blocker.
    @ViewBuilder
    var advice: some View {
        if !glasses.hasActiveDevice && !isRunning {
            HelperText("Waiting for the glasses to become active.")
        } else if glasses.cameraPermissionStatus == .denied && !isRunning {
            // Advice, not a `.disabled` condition — see the equivalent branch
            // in `HomeWorkspaceView.sessionControl`.
            HelperText("Camera access is not granted. Allow it under Connections, then start the camera.")
        } else if tower.status != .online {
            // The Tower must be online *before* the camera starts: a
            // `stream_start` sent while it is offline is dropped, and every
            // frame after it is suppressed for the whole session. With the
            // camera already running the sentence is different: the Tower row
            // above is the way back, and `ProjectManager` re-sends
            // `stream_start` when it returns.
            HelperText(
                isRunning
                    ? "The Tower is not connected; frames are dropped until it is. Reconnect from the Tower row above."
                    : "Connect to the Tower before starting the camera; frames sent while offline are dropped."
            )
        } else if isHeld && isRunning {
            HelperText("Frames are held on the phone. The camera and the connection stay up; the Tower stops seeing frames after about five seconds.")
        } else if isRunning {
            HelperText("Frames are streaming to the Tower. Pause holds them on the phone without stopping the camera.")
        } else {
            HelperText("Starts the glasses camera and streams frames to the Tower for the experiment below.")
        }
    }

    /// Why the camera did not start, in this card's own words.
    ///
    /// `GlassesConnection` records the reason, and Object Memory writes its own
    /// sentences about it — every one of which is about a Tower session that
    /// stays open. None of that applies here, where no Tower session was asked
    /// for, so this card has its own six, each saying what a person can do
    /// next. Exhaustive on purpose: a seventh reason must be written about,
    /// not dropped.
    static func refusalLine(_ refusal: CaptureStartRefusal) -> String {
        switch refusal {
        case .alreadyRunning:
            return """
                A capture is already running, started from another screen. This \
                card controls that same camera, so nothing else was started.
                """
        case .deviceHasPausedCapture:
            return """
                The glasses have paused the capture themselves — a press on the \
                temple, or heat — and delivery comes back on its own. This app \
                cannot override that.
                """
        case .captureIsShuttingDown:
            return """
                The camera is shutting down and a new capture cannot open until \
                it has finished. Start again in a moment.
                """
        case .noActiveDevice:
            return "No glasses are active yet, so no capture could be started."
        case .cameraPermissionNotGranted:
            return """
                Camera access is not granted, so no capture could be started. \
                Allow it under Connections, then start again.
                """
        case .datRefused(let reason):
            return "The glasses refused to start a capture: \(reason)"
        }
    }
}
#endif
