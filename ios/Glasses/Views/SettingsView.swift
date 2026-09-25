//
//  SettingsView.swift
//  Glasses
//

import Combine
import SwiftUI

// MARK: - Words

/// Every sentence Settings shows, in one place, so a test can pin them without
/// rendering a view.
enum TowerSettingsText {
    static let title = "Settings"
    static let sectionHeader = "Tower"
    static let inUseLabel = "In use now"
    static let fieldLabel = "Tower address"
    static let fieldPrompt = "host:port"
    static let testButton = "Test connection"
    static let testing = "Testing…"
    static let saveButton = "Save"
    static let useBuiltInButton = "Use the built-in address"
    static let unencryptedTitle = "Not encrypted"
    static let unencryptedBody = "The phone talks to the Tower over plain http:// and ws://, so frames and results cross the network unencrypted. Use an address on your own network or on Tailscale."
    static let restartTitle = "Restart Glasses to use it"

    static func source(_ source: TowerConfiguration.Source) -> String {
        switch source {
        case .builtIn: return "Built-in address"
        case .saved: return "Saved in Settings"
        case .developerOverride: return "Developer override (GLASSES_TOWER_AUTHORITY)"
        }
    }

    /// Why the typed address cannot be used, or `nil` when it can.
    static func problem(_ check: TowerAddressCheck) -> String? {
        switch check {
        case .accepted:
            return nil
        case .empty:
            return "Enter the Tower's address, like 192.168.1.20:8000."
        case .includesScheme:
            return "Leave out http:// or ws://. Enter just host:port, like 192.168.1.20:8000."
        case .malformed:
            return "That isn't an address. Use host:port, like 192.168.1.20:8000, with no path."
        case .notPrivate:
            return "This build connects only to your own network or Tailscale: 10.x, 172.16–31.x, 192.168.x, 169.254.x or 100.64–127.x addresses, or names ending in .local or .ts.net."
        }
    }

    /// The one-line answer. Exactly one of three, by design.
    static func outcomeTitle(_ outcome: TowerProbeOutcome) -> String {
        switch outcome {
        case .notReachable: return "Not reachable"
        case .notATower: return "Reached something, but it isn't a Glasses Tower"
        case .tower: return "Connected to a Tower"
        }
    }

    static func outcomeDetail(_ outcome: TowerProbeOutcome, authority: String) -> String {
        switch outcome {
        case .notReachable(.timedOut):
            return "No answer from \(authority) within \(Int(TowerConnectionTester.timeout)) seconds."
        case .notReachable(.refused):
            return "Nothing at \(authority) accepted the connection. Check the port, and that the Tower is running."
        case .notReachable(.nameNotFound):
            return "The name in \(authority) could not be found on this network."
        case .notReachable(.noNetwork):
            return "This phone is not connected to a network."
        case .notReachable(.blockedByTransportSecurity):
            return "This build does not allow an unencrypted connection to \(authority)."
        case .notReachable(.other):
            return "The request to \(authority) failed before anything answered."
        case .notATower(.status(let code)):
            return "\(authority) answered HTTP \(code) when asked for its cartridges."
        case .notATower(.notADeclaration):
            return "\(authority) answered, but not with a Tower's list of cartridges."
        case .notATower(.notHTTP):
            return "\(authority) answered with something that is not a web response."
        case .tower(let contract?):
            return "\(authority) · contract \(contract)"
        case .tower(nil):
            return "\(authority) · it did not name a contract"
        }
    }

    static func restartBody(inUse: String, next: String) -> String {
        "Saved. Glasses keeps using \(inUse) until you close it from the app switcher and open it again. Then it uses \(next)."
    }

    static func overrideBody(overrideAuthority: String, withoutOverride: String) -> String {
        "This run uses \(overrideAuthority) from GLASSES_TOWER_AUTHORITY, a developer setting that wins over a saved address. Launched without it, Glasses uses \(withoutOverride)."
    }

    static func footer(policy: TowerAddressPolicy) -> String {
        let test = "Test connection sends one request, GET /cartridges, which changes nothing on the Tower, and gives up after \(Int(TowerConnectionTester.timeout)) seconds. A saved address is used from the next launch."
        switch policy {
        case .anyHost:
            return test
        case .privateNetworks:
            return test + " This build accepts private, link-local and Tailscale addresses, and names ending in .local or .ts.net."
        }
    }
}

// MARK: - Model

/// Where "Test connection" is.
enum TowerProbeState: Equatable {
    case idle
    case testing(String)
    case finished(String, TowerProbeOutcome)
}

/// Settings' Tower section: a draft address, the saved one, and one probe.
///
/// ## What it never does
///
/// It does not touch `TowerClient`, and it does not change the address the
/// running app uses. `inUse` is `TowerConfiguration.resolution`, fixed for the
/// process (see the note there on why); saving writes `TowerAddressStore`, and
/// the screen then says a restart is needed. The probe is
/// `TowerConnectionTester`, one `GET /cartridges` — never the socket, never a
/// session.
@MainActor
final class TowerSettingsModel: ObservableObject {
    @Published var draft: String {
        didSet {
            // A result is about the address that was tested. Once the field
            // says something else, showing it would answer the wrong question.
            if draft != oldValue { cancelProbe() }
        }
    }
    @Published private(set) var probe: TowerProbeState = .idle
    @Published private(set) var saved: String?

    let inUse: TowerConfiguration.Resolution
    let overrideValue: String?
    let policy: TowerAddressPolicy
    private let store: TowerAddressStore
    private let tester: TowerConnectionTester
    private var probeTask: Task<Void, Never>?

    init(
        inUse: TowerConfiguration.Resolution,
        overrideValue: String?,
        policy: TowerAddressPolicy,
        store: TowerAddressStore,
        tester: TowerConnectionTester
    ) {
        self.inUse = inUse
        self.overrideValue = overrideValue
        self.policy = policy
        self.store = store
        self.tester = tester
        let saved = store.saved
        self.saved = saved
        self.draft = saved ?? inUse.authority
    }

    /// The running app's own state. Reading `TowerConfiguration.resolution`
    /// here also fixes it, if nothing has yet, before anything can be saved.
    convenience init() {
        self.init(
            inUse: TowerConfiguration.resolution,
            overrideValue: TowerConfiguration.launchOverrideValue,
            policy: .current,
            store: TowerAddressStore(),
            tester: TowerConnectionTester()
        )
    }

    var check: TowerAddressCheck { policy.check(draft) }

    var acceptedDraft: String? {
        if case .accepted(let authority) = check { return authority }
        return nil
    }

    var isTesting: Bool {
        if case .testing = probe { return true }
        return false
    }

    var canTest: Bool { acceptedDraft != nil && !isTesting }

    /// Only when saving would change what a launch uses.
    var canSave: Bool {
        guard let accepted = acceptedDraft else { return false }
        return accepted != (saved ?? TowerConfiguration.defaultAuthority)
    }

    var canUseBuiltIn: Bool { saved != nil }

    /// What the next launch will use, launched the way this one was.
    var nextLaunch: TowerConfiguration.Resolution {
        TowerConfiguration.resolve(overrideValue: overrideValue, savedValue: saved, policy: policy)
    }

    /// What a launch without the developer override would use.
    var withoutOverride: TowerConfiguration.Resolution {
        TowerConfiguration.resolve(overrideValue: nil, savedValue: saved, policy: policy)
    }

    var needsRestart: Bool { nextLaunch.authority != inUse.authority }

    func save() {
        guard let accepted = acceptedDraft else { return }
        store.save(accepted)
        saved = accepted
        if draft != accepted { draft = accepted }
    }

    func useBuiltIn() {
        store.clear()
        saved = nil
        draft = TowerConfiguration.defaultAuthority
    }

    func test() {
        guard let authority = acceptedDraft, !isTesting else { return }
        probe = .testing(authority)
        let tester = self.tester
        probeTask = Task { [weak self] in
            let outcome = await tester.probe(authority: authority)
            guard let self, !Task.isCancelled, self.probe == .testing(authority) else { return }
            self.probe = .finished(authority, outcome)
        }
    }

    func cancelProbe() {
        probeTask?.cancel()
        probeTask = nil
        if probe != .idle { probe = .idle }
    }
}

// MARK: - Views

/// "How Glasses works" in Settings: whether it can open now, and what opens it.
///
/// Handed in by whoever presents Settings, because the cards are shown by the
/// shell (`ContentView`) and never over a capture. Where Settings is reached
/// from somewhere that cannot show them -- Connections' pushed Settings, or
/// Settings opened from the cards themselves -- there is no entry and no row.
struct HowGlassesWorksEntry {
    /// False while a capture is running or starting.
    let isAvailable: Bool
    let open: () -> Void
}

/// Settings, presented as a sheet from the shell.
struct SettingsSheet: View {
    var howItWorks: HowGlassesWorksEntry?

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            SettingsView(howItWorks: howItWorks)
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
        }
    }
}

/// The Settings screen. Pushed from Connections, or inside `SettingsSheet`.
///
/// Nothing here navigates anywhere else. In particular it never opens a
/// workspace: World Builder's sends `session/start` as soon as it appears.
struct SettingsView: View {
    /// Built fresh on each appearance, from what is saved now.
    @StateObject private var model = TowerSettingsModel()
    @FocusState private var isEditingAddress: Bool
    private let howItWorks: HowGlassesWorksEntry?

    init(howItWorks: HowGlassesWorksEntry? = nil) {
        self.howItWorks = howItWorks
    }

    var body: some View {
        List {
            Section {
                inUseRow
                addressRow
                testButton
                if case .finished(let authority, let outcome) = model.probe {
                    TowerProbeResultRow(authority: authority, outcome: outcome)
                }
                Button(TowerSettingsText.saveButton) { model.save() }
                    .disabled(!model.canSave)
                    .accessibilityIdentifier("tower-save")
                if model.canUseBuiltIn {
                    Button(TowerSettingsText.useBuiltInButton) { model.useBuiltIn() }
                        .accessibilityHint("Forgets the saved address. The next launch uses \(TowerConfiguration.defaultAuthority).")
                        .accessibilityIdentifier("tower-use-built-in")
                }
                if model.needsRestart {
                    NoteRow(
                        symbol: "arrow.clockwise.circle.fill",
                        tint: .accentColor,
                        title: TowerSettingsText.restartTitle,
                        text: TowerSettingsText.restartBody(
                            inUse: model.inUse.authority,
                            next: model.nextLaunch.authority
                        )
                    )
                    .accessibilityIdentifier("tower-restart-notice")
                }
                NoteRow(
                    symbol: "lock.open.fill",
                    tint: .orange,
                    title: TowerSettingsText.unencryptedTitle,
                    text: TowerSettingsText.unencryptedBody
                )
                .accessibilityIdentifier("tower-unencrypted-note")
            } header: {
                Text(TowerSettingsText.sectionHeader)
            } footer: {
                Text(TowerSettingsText.footer(policy: model.policy))
            }

            if model.inUse.source == .developerOverride {
                Section {
                    NoteRow(
                        symbol: "hammer.fill",
                        tint: .secondary,
                        title: TowerSettingsText.source(.developerOverride),
                        text: TowerSettingsText.overrideBody(
                            overrideAuthority: model.inUse.authority,
                            withoutOverride: model.withoutOverride.authority
                        )
                    )
                    .accessibilityIdentifier("tower-override-notice")
                }
            }

            // Last, so the Tower section stays the first thing on the screen.
            if let howItWorks {
                Section {
                    Button(action: howItWorks.open) {
                        Label(OnboardingText.settingsRow, systemImage: "questionmark.circle")
                    }
                    .disabled(!howItWorks.isAvailable)
                    .accessibilityIdentifier("settings-how-it-works")
                } footer: {
                    if !howItWorks.isAvailable {
                        Text(OnboardingText.settingsRowUnavailable)
                    }
                }
            }
        }
        .navigationTitle(TowerSettingsText.title)
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear { model.cancelProbe() }
    }

    private var inUseRow: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(TowerSettingsText.inUseLabel)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text(model.inUse.authority)
                .font(.body.monospaced())
            Text(TowerSettingsText.source(model.inUse.source))
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("tower-in-use")
    }

    private var addressRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(TowerSettingsText.fieldLabel)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            // Vertical, so a long address wraps at the accessibility sizes
            // instead of scrolling out of sight. Return still means "done":
            // a vertical field would insert a newline, so it is taken out and
            // the keyboard put away.
            TextField(
                TowerSettingsText.fieldLabel,
                text: $model.draft,
                prompt: Text(TowerSettingsText.fieldPrompt),
                axis: .vertical
            )
            .lineLimit(1...4)
            .focused($isEditingAddress)
            .onChange(of: model.draft) { _, draft in
                guard draft.contains("\n") else { return }
                model.draft = draft.replacingOccurrences(of: "\n", with: "")
                isEditingAddress = false
            }
            .font(.body.monospaced())
            .keyboardType(.URL)
            .textContentType(.URL)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .submitLabel(.done)
            .accessibilityLabel(TowerSettingsText.fieldLabel)
            .accessibilityIdentifier("tower-address-field")
            if let problem = TowerSettingsText.problem(model.check) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Image(systemName: "exclamationmark.circle.fill")
                        .foregroundStyle(.red)
                        .accessibilityHidden(true)
                    Text(problem)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .font(.footnote)
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("tower-address-problem")
            }
        }
        // The separator spans the row, not just the problem's text.
        .alignmentGuide(.listRowSeparatorLeading) { _ in 0 }
    }

    private var testButton: some View {
        Button {
            model.test()
        } label: {
            HStack(spacing: 8) {
                if model.isTesting {
                    ProgressView()
                        .accessibilityHidden(true)
                    Text(TowerSettingsText.testing)
                } else {
                    Text(TowerSettingsText.testButton)
                }
            }
        }
        .disabled(!model.canTest)
        .accessibilityIdentifier("tower-test")
    }
}

/// The answer to "Test connection", with the address it is about.
private struct TowerProbeResultRow: View {
    let authority: String
    let outcome: TowerProbeOutcome

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: symbol)
                .foregroundStyle(tint)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(TowerSettingsText.outcomeTitle(outcome))
                    .font(.body.weight(.semibold))
                Text(TowerSettingsText.outcomeDetail(outcome, authority: authority))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("tower-test-result")
    }

    private var symbol: String {
        switch outcome {
        case .tower: return "checkmark.circle.fill"
        case .notATower: return "questionmark.circle.fill"
        case .notReachable: return "xmark.circle.fill"
        }
    }

    private var tint: Color {
        switch outcome {
        case .tower: return .green
        case .notATower: return .orange
        case .notReachable: return .red
        }
    }
}

/// A titled note with an icon: the restart, encryption and override notes.
private struct NoteRow: View {
    let symbol: String
    let tint: Color
    let title: String
    let text: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: symbol)
                .foregroundStyle(tint)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(text)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
    }
}
