//
//  ConnectionSheet.swift
//  Glasses
//

import MWDATCore
import SwiftUI

/// Every manual connection control, off the primary surface but always one tap
/// away.
///
/// The old dashboard presented these three rows as a permanent setup checklist.
/// With registration state streamed from launch, camera permission read
/// automatically, and the Tower dialled automatically, a checklist is the wrong
/// resting state — it turns a solved problem into a chore. So the rows moved
/// here, reached by tapping the shell status bar.
///
/// Moving them must not make them unreachable, and that is the point of this
/// sheet rather than deleting them outright. The Tower's reconnect schedule is
/// bounded and ends at a visible `.failed`; `startAutomaticConnections()` runs
/// once per process and will not restart it. Manual retry is therefore the only
/// recovery, and it has to exist somewhere a person can find. Every row keeps
/// its action live in every state, exactly as before.
struct ConnectionSheet: View {
    @ObservedObject var glasses: GlassesConnection
    @ObservedObject var tower: TowerClient

    @Environment(\.dismiss) private var dismiss
    @State private var isConfirmingUnregister = false

    var body: some View {
        NavigationStack {
            List {
                Section {
                    SetupRow(
                        title: "Meta AI",
                        detail: StateDisplay.registration(glasses.registrationState),
                        isComplete: glasses.registrationState == .registered,
                        actionTitle: glasses.registrationState == .registered ? "Re-register" : "Register",
                        action: { glasses.connect() }
                    )
                    // Still reachable outside DEBUG, and still necessary. The
                    // launch-time automatic read covers the ordinary case, but
                    // a permission changed in Settings while the app was
                    // running is only observable by asking again.
                    SetupRow(
                        title: "Camera access",
                        detail: StateDisplay.cameraPermission(glasses.cameraPermissionStatus),
                        isComplete: glasses.cameraPermissionStatus == .granted,
                        actionTitle: cameraActionTitle,
                        action: {
                            if glasses.cameraPermissionStatus == .denied {
                                glasses.requestCameraPermission()
                            } else {
                                glasses.checkCameraPermission()
                            }
                        }
                    )
                    // Disconnect stays available while `.connecting`, not just
                    // while `.online`: `connect()` is a no-op in that state, and
                    // the validation ping has no timeout of its own, so without
                    // an explicit teardown a hung connect would strand the user
                    // until URLSession's own timeout expired.
                    SetupRow(
                        title: "Tower",
                        detail: StateDisplay.tower(tower.status),
                        isComplete: tower.status == .online,
                        actionTitle: towerActionTitle,
                        action: {
                            switch tower.status {
                            case .online, .connecting: tower.disconnect()
                            case .offline, .failed: tower.connect()
                            }
                        }
                    )
                    // Where "can't reach the Tower" is fixed when the address
                    // is wrong for this network: Connect can only retry it.
                    NavigationLink {
                        SettingsView()
                    } label: {
                        LabeledContent("Tower address") {
                            Text(TowerConfiguration.authority)
                                .font(.footnote.monospaced())
                        }
                    }
                    .accessibilityHint("Opens Settings, to change or test the Tower's address")
                    .accessibilityIdentifier("connections-tower-address")
                } header: {
                    Text("Connections")
                } footer: {
                    Text(towerFooter)
                }

                Section {
                    Button("Unregister from Meta AI", role: .destructive) {
                        isConfirmingUnregister = true
                    }
                } footer: {
                    Text("Unregistering means completing the full pairing flow again before the glasses can be used.")
                }
            }
            .navigationTitle("Connections")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .confirmationDialog(
                "Unregister from Meta AI?",
                isPresented: $isConfirmingUnregister,
                titleVisibility: .visible
            ) {
                Button("Unregister", role: .destructive) { glasses.disconnect() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("You will have to complete the full pairing flow again before the glasses can be used.")
            }
        }
    }

    /// Names the endpoint. The address changes between networks, so "can't
    /// reach the Tower" is an unactionable mystery without it and a fixable
    /// problem with it — fixed in Settings, one row up.
    private var towerFooter: String {
        let endpoint = TowerConfiguration.webSocketURL.absoluteString
        if case .failed(let message) = tower.status {
            return "Could not reach \(endpoint)\n\(message)"
        }
        return "Tower endpoint: \(endpoint)"
    }

    private var cameraActionTitle: String {
        switch glasses.cameraPermissionStatus {
        case .none: return "Check"
        case .denied: return "Allow"
        default: return "Re-check"
        }
    }

    private var towerActionTitle: String {
        switch tower.status {
        case .online: return "Disconnect"
        case .connecting: return "Cancel"
        case .offline, .failed: return "Connect"
        }
    }
}
