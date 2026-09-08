//
//  CartridgeDrawerView.swift
//  Glasses
//

import SwiftUI

/// The cartridge tray: the shell's workspace picker.
///
/// ## What selecting a cartridge does, and does not do
///
/// Choosing a cartridge here changes **which workspace this app shows**. It
/// sends nothing to the Tower, selects no module there, and changes no Tower
/// state — there is no module-selection message in the protocol, and inventing
/// one is forbidden (docs/08-IOS-CARTRIDGE-SHELL.md). The Tower's vocabulary is
/// still exactly `ping`, `pong`, `frame`, `frame_result`, `stream_start`,
/// `stream_stop`.
///
/// The two were once fully independent, which is how a row came to be tappable
/// while its badge read "Future". They are no longer: a cartridge with nothing
/// to open cannot be `.readyToTest`, and one that ships a workspace is never
/// `.notBuilt`. `CartridgeCatalogTests` pins that in both directions.
/// The badge describes the *module's* position on the Tower roadmap; being
/// tappable describes whether *this app* ships a screen for it. Only cartridges
/// with a workspace are tappable — the rest stay exactly as they were, with no
/// `Button`, no `NavigationLink` and no tap target, because for them there
/// genuinely is nothing to open.
///
/// When the Tower can run modules, the active cartridge must be driven by what
/// the Tower reports is running, not by what was tapped here.
struct CartridgeDrawerView: View {
    @Binding var selectedCartridgeID: String

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button {
                        selectedCartridgeID = ""
                        dismiss()
                    } label: {
                        HomeRow(isSelected: selectedCartridgeID.isEmpty)
                    }
                    .buttonStyle(.plain)
                }

                Section {
                    // Every catalog entry, in catalog order, and — critically —
                    // the openability decision is read off the row rather than
                    // re-derived here. `Cartridge.selectable` is defined as the
                    // openable rows of this same list, so the drawer and the
                    // rest of the app cannot come to different conclusions
                    // about what may be opened.
                    ForEach(Cartridge.drawerRows) { row in
                        switch row {
                        case .openable(let cartridge, _):
                            Button {
                                selectedCartridgeID = cartridge.id
                                dismiss()
                            } label: {
                                CartridgeRow(
                                    row: row,
                                    isSelected: selectedCartridgeID == cartridge.id
                                )
                            }
                            .buttonStyle(.plain)
                        case .informational:
                            CartridgeRow(row: row, isSelected: false)
                        }
                    }
                } header: {
                    Text("Modules")
                } footer: {
                    // The middle clause of this footer used to read: "It does
                    // not start anything on the Tower — the Tower chooses what
                    // it runs at startup and this app cannot ask it for
                    // anything else."
                    //
                    // That was true when written and is now false in the worst
                    // direction. This build sends `cv_lab_start`,
                    // `POST /documents-session/start` and
                    // `POST /cartridges/object_memory/session/start`, and three
                    // workspaces draw a Start control.
                    //
                    // It is not merely stale — it is the sentence a person
                    // reads *before* deciding whether opening a screen can make
                    // the Tower begin recording. Telling someone the app cannot
                    // ask the Tower for anything, on a build that can ask it to
                    // start keeping what its camera sees, is the one direction
                    // this claim must never be wrong in.
                    //
                    // What survives is the true and useful half: opening a
                    // cartridge is not itself a *recording*. The recording
                    // verbs are deliberate and live inside the workspace.
                    //
                    // Corrected again on 2026-09-08, in the same direction and
                    // for the same reason. "Opening one does not start
                    // anything on the Tower by itself" had become false for
                    // two of the five, and the hedge that followed it — "but
                    // some workspaces can, and they say so where the control
                    // is" — pointed at the Start controls, which is not where
                    // this happens:
                    //
                    //   * World Builder POSTs its cartridge session `start`
                    //     from `.onAppear` (`WorldBuilderSessionController.
                    //     workspaceDidAppear`). An active session is precisely
                    //     what lets a builder attach to a capture, so opening
                    //     the screen while a camera is streaming starts a
                    //     world-build worker. No control was touched.
                    //   * Scene Understanding subscribes to the live scene
                    //     from `.onAppear`, and on this Tower a subscription
                    //     *is* the watcher half of "somebody streams and
                    //     somebody watches" — so opening the screen over an
                    //     open stream loads and runs a people detector.
                    //
                    // **The two are not alike on retention, and an earlier
                    // draft of this very sentence got that wrong.** It said
                    // "neither keeps anything", generalising from the fact
                    // that a `CartridgeSession` is not itself persisted. That
                    // is a different fact. World Builder's session is the gate
                    // on the world-build worker (`tower/tower/main.py`), and
                    // that worker exists to write: it is launched with
                    // `--root <world_root>`, opens a `WorldStore` on it, and
                    // writes sources, placements and a solve log — the very
                    // worlds "Saved worlds" later lists. So opening this
                    // screen over a live capture can put a reconstruction on
                    // the Tower's disk.
                    //
                    // Scene Understanding really does keep nothing, and its
                    // own screen says so. Saying it about both would be an
                    // active reassurance that is false, in the one direction
                    // this claim must never be wrong in — worse than the
                    // stale-but-conservative sentence it replaced.
                    Text("Opening a cartridge changes this app's workspace. Two of them also set the Tower working as they open: World Builder activates its session, and Scene Understanding starts watching the scene — so if a camera is already streaming, opening those is enough. They differ in what is left behind: Scene Understanding keeps nothing, while World Builder's session is what lets the Tower build and save a world from a capture that is already running. Recording is otherwise a deliberate act, and those controls live inside the workspace. Badges describe this app: \u{201C}Ready to test\u{201D} means there is something here to try, not that the Tower you are connected to is serving it right now. Each workspace says what that Tower can actually do.")
                        .padding(.top, 4)
                }
            }
            .navigationTitle("Cartridges")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

private struct HomeRow: View {
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Home")
                    .font(.body.weight(.medium))
                Text("Infrastructure status and a plain capture session.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            if isSelected {
                Image(systemName: "checkmark")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.tint)
                    .accessibilityHidden(true)
            }
        }
        .padding(.vertical, 4)
        .contentShape(.rect)
        .accessibilityElement(children: .combine)
        // Only `.isSelected`. The row is already wrapped in a `Button`, which
        // supplies the button trait; adding it here makes VoiceOver say it twice.
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

private struct CartridgeRow: View {
    /// The row, not the cartridge: openability arrives already decided rather
    /// than being worked out again here. The hint below is the second half of
    /// that decision and comes from the same place.
    let row: CartridgeDrawerRow
    let isSelected: Bool

    private var cartridge: Cartridge { row.cartridge }
    private var isOpenable: Bool { row.isOpenable }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(cartridge.name)
                    .font(.body.weight(.medium))
                Text(cartridge.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 8)

            VStack(alignment: .trailing, spacing: 6) {
                // Tinted for exactly one status, so the drawer answers "which
                // of these can I try?" in a glance rather than after reading
                // eight identical capsules. Colour is not the only carrier —
                // the badge still spells the status out — because a tint alone
                // would be invisible to a reader who cannot distinguish it.
                Text(cartridge.status.badge)
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        cartridge.status.isProminent
                            ? AnyShapeStyle(Color.accentColor.opacity(0.18))
                            : AnyShapeStyle(Color(.tertiarySystemFill)),
                        in: .capsule
                    )
                    .foregroundStyle(cartridge.status.isProminent ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))

                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.body.weight(.semibold))
                        .foregroundStyle(.tint)
                        .accessibilityHidden(true)
                }
            }
        }
        .padding(.vertical, 4)
        .contentShape(.rect)
        .accessibilityElement(children: .combine)
        // Two different truths, so two different hints. An unopenable row is
        // informational exactly as it always was. The strings live on the row
        // beside the decision that picks between them, so a row cannot be
        // untappable while announcing that it opens something.
        .accessibilityHint(row.accessibilityHint)
        // An openable row is already inside a `Button`, so only selection needs
        // stating. A row without a workspace gets no traits at all, which is
        // what makes it read as informational rather than actionable.
        .accessibilityAddTraits(isOpenable && isSelected ? .isSelected : [])
    }
}

#Preview {
    CartridgeDrawerView(selectedCartridgeID: .constant(""))
}
