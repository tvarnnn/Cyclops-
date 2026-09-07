//
//  WorldBuilderSessionController.swift
//  Glasses
//

import Foundation
import os

/// Tells the Tower that the World Builder workspace is on screen.
///
/// ## What it is for
///
/// Since 2026-09-06 a builder attaches to a camera capture **only while the
/// `world_builder` cartridge session is `active`** on the Tower. Before that,
/// every capture spawned a builder whatever cartridge the phone was in, which
/// is how Object Memory sessions grew World Builder worlds nobody asked for.
/// The generic session surface (`POST /cartridges/{id}/session/start|stop`,
/// `cartridge_session.control/2026-08-27`) already existed and is
/// cartridge-blind; this type sends `start` when the workspace appears and
/// `stop` when it disappears, and nothing else on the phone starts or stops a
/// builder.
///
/// ## What it deliberately is not
///
/// **Not part of `WorldBuilderViewModel`.** That object is constructed in
/// tests beside a live socket with the assertion that a cartridge view model
/// sends the Tower nothing it was not asked for
/// (`TowerClientTests.testCartridgeViewModelsSendNothingToTheTower`). A view
/// model that POSTed on construction would honour the letter of that test —
/// HTTP is not the socket — and break its point. This controller is owned by
/// the *view*, fires on the view's appearance, and is handed its transport, so
/// a test that wants no traffic constructs a view model and gets none.
///
/// **Not a claim that a builder is running.** `state: "active"` is intent —
/// `state_means: "intent-not-liveness"`, in the payload's own words — and the
/// footnote is worded as what was asked for. Whether a builder attached is
/// what the canvas above reports, from the result channel.
///
/// **Not a Release-only or DEBUG-only path.** The workspace exists in both
/// configurations and the Tower's gate applies to both, so a Release build
/// that never sent `start` would leave the Tower refusing to build for a
/// DEBUG phone standing next to it.
@MainActor
final class WorldBuilderSessionController: ObservableObject {

    /// The Tower's name for the cartridge on the session surface — the same
    /// string the result channel uses, not this app's catalog id.
    static let cartridge = WorldBuilderResultContract.towerCartridge

    /// Where the request stands, as far as this phone can say.
    enum Status: Equatable, Sendable {
        /// The workspace has not appeared, or has disappeared and the stop
        /// was sent.
        case notAsked
        /// On screen with no Tower to ask. `start` goes out when the socket
        /// comes back.
        case waitingForTower
        /// A `start` is in flight.
        case starting
        /// The Tower honoured `start`. **Intent, not liveness.**
        case active
        /// The Tower answered 409 with its own sentence.
        case refused(String)
        /// The Tower answered 404: no controllable session for this
        /// cartridge on this Tower.
        case noSessionControl
        /// The request did not complete, or the answer could not be read.
        case failed(String)
        /// A `stop` is in flight.
        case stopping
    }

    @Published private(set) var status: Status = .notAsked

    /// One line for under the capture control. Every sentence is a claim the
    /// phone can support from what it sent and what came back.
    var footnote: String {
        switch status {
        case .notAsked:
            return "World Builder has not been asked for on the Tower yet."
        case .waitingForTower:
            return "The Tower is not connected. World Builder will be asked for when it is."
        case .starting:
            return "Asking the Tower to activate World Builder…"
        case .active:
            return "World Builder is active on the Tower."
        case .refused(let message):
            return "The Tower refused to activate World Builder: \(message)"
        case .noSessionControl:
            return "This Tower offers no World Builder session control, so a builder cannot be asked for from here."
        case .failed(let detail):
            return "World Builder could not be asked for on the Tower: \(detail)"
        case .stopping:
            return "Telling the Tower World Builder is no longer wanted…"
        }
    }

    private let control: CartridgeSessionHTTPClient
    /// Whether the workspace is currently on screen. A `start` that finishes
    /// after `workspaceDidDisappear` must not report `.active` for a screen
    /// that is gone.
    private var isOnScreen = false
    /// Which request is current. A response from an older generation is
    /// dropped rather than applied over a newer one: appear, disappear,
    /// appear within one round trip is a real sequence on a cartridge switch.
    private var generation = 0
    private var inFlight: Task<Void, Never>?

    /// `control` is injectable so a test can point it at a stubbed
    /// `URLSession`. The default is the generic client keyed to this
    /// cartridge, with its own ten-second bound (Rule 15).
    init(control: CartridgeSessionHTTPClient? = nil) {
        self.control = control ?? CartridgeSessionHTTPClient(cartridge: Self.cartridge)
    }

    // MARK: The three moments

    /// The workspace appeared. `start` if the Tower is there; otherwise wait
    /// for `towerReachabilityChanged(isReachable: true)`.
    ///
    /// Not sent to an unreachable Tower on purpose. The request would fail
    /// after its bound and the footnote would say so, truthfully but
    /// uselessly; "will be asked for when it is" is the better sentence, and
    /// the socket coming back is the moment to act.
    func workspaceDidAppear(isTowerReachable: Bool) {
        isOnScreen = true
        guard isTowerReachable else {
            status = .waitingForTower
            return
        }
        start()
    }

    /// The Tower's socket came or went while the workspace is on screen.
    /// Coming back re-sends `start`: the Tower may have restarted, and a
    /// restarted Tower comes back `stopped` (nothing is persisted, by the
    /// session contract's own rule).
    func towerReachabilityChanged(isReachable: Bool) {
        guard isOnScreen else { return }
        if isReachable {
            start()
        } else if status != .active {
            // A start that was waiting or failing has lost its Tower. An
            // `.active` answer already given stands as what was asked for.
            status = .waitingForTower
        }
    }

    /// The workspace disappeared. `stop` is sent whatever state the start
    /// reached — it is never refused, and a stop for a session that never
    /// started is 200 with `changed: false`.
    func workspaceDidDisappear() {
        isOnScreen = false
        stop()
    }

    // MARK: Requests

    private func start() {
        begin(.start, pending: .starting) { [weak self] outcome in
            guard let self, self.isOnScreen else { return }
            switch outcome {
            case .honoured:
                self.status = .active
            case .refused(let refusal):
                self.status = .refused(refusal.message)
            }
        }
    }

    private func stop() {
        begin(.stop, pending: .stopping) { [weak self] _ in
            // Off screen either way; the footnote is not visible and the
            // next appearance starts from nothing.
            guard let self, !self.isOnScreen else { return }
            self.status = .notAsked
        }
    }

    /// One request at a time. A newer one supersedes an older one's *effect*
    /// on `status`, not its delivery — a `stop` that follows a `start` still
    /// goes out, because the Tower needs both.
    private func begin(
        _ action: CartridgeSessionAction,
        pending: Status,
        apply: @escaping @MainActor (CartridgeSessionOutcome) -> Void
    ) {
        generation += 1
        let mine = generation
        status = pending
        let control = self.control
        inFlight = Task { [weak self] in
            let result: Result<CartridgeSessionOutcome, Error>
            do {
                result = .success(try await control.apply(action))
            } catch {
                result = .failure(error)
            }
            guard let self, mine == self.generation else { return }
            switch result {
            case .success(let outcome):
                Self.logger.info(
                    "\(action.rawValue, privacy: .public) honoured=\(outcome.snapshot.accepted ?? true, privacy: .public) state=\(outcome.snapshot.state.rawValue, privacy: .public)"
                )
                apply(outcome)
            case .failure(let error):
                let detail = Self.describe(error)
                Self.logger.error(
                    "\(action.rawValue, privacy: .public) FAILED — \(detail, privacy: .public)"
                )
                guard action == .start, self.isOnScreen else {
                    if !self.isOnScreen { self.status = .notAsked }
                    return
                }
                if case CartridgeSessionFetchError.noSuchCartridgeSession = error {
                    self.status = .noSessionControl
                } else {
                    self.status = .failed(detail)
                }
            }
        }
    }

    /// The failure in a sentence a footnote can carry.
    private static func describe(_ error: Error) -> String {
        guard let fetch = error as? CartridgeSessionFetchError else {
            return error.localizedDescription
        }
        switch fetch {
        case .noSuchCartridgeSession:
            return "this Tower has no session control for World Builder"
        case .unsupportedContract(let identifier):
            return "the Tower's session surface speaks \(identifier), which this build does not"
        case .undecodable:
            return "the Tower's answer could not be read"
        case .transport(let detail):
            return detail
        }
    }

    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "Glasses",
        category: "WorldBuilderSession"
    )
}
