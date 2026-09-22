//
//  ObjectMemoryRecording.swift
//  Glasses
//

import Combine
import Foundation

/// One tap that starts both halves of remembering: the Tower's producer, and
/// the glasses camera.
///
/// ## Why this type exists
///
/// Before it, Object Memory's Start button started the *producer* and nothing
/// else. A person who pressed it on a phone with no capture running got an
/// `active` session with `attached_capture_id: null`, no frames, no records,
/// and no sentence anywhere on the screen explaining that they also had to go
/// to Home and press a second, differently-named button first. The cartridge
/// was not self-contained, and the screen did not say so.
///
/// ## What it owns, and what it very deliberately does not
///
/// It owns **sequencing and honesty**, and no camera code whatsoever. The
/// camera is `GlassesConnection` — still the only type in this app that talks
/// to DAT, still reached through `startCameraSession()` / `stopCameraSession()`
/// and nothing else. The Tower's producer is `CartridgeSessionHTTPClient`
/// behind `ObjectMemoryClient`, unchanged. This composes the two.
///
/// Both are reached through protocols rather than concrete types, and that is
/// not ceremony: `GlassesConnection`'s capture surface is `#if DEBUG`, needs a
/// paired device and a permission grant, and cannot be driven from a test at
/// all. A seam is the only way this sequencing is testable, and untested
/// sequencing is exactly where the race below would live.
///
/// ## The one rule inherited from `ObjectMemorySession.swift`
///
/// **`state` is intent. `following` is fact.** Every phase this type publishes
/// that claims something is being recorded is derived from
/// `snapshot.isFollowingACapture`, and never from `state`. The whole reason
/// `.waitingToBeFollowed` and `.notObserved` are separate cases from
/// `.remembering` is that "we asked" must never be drawn as "it is happening".

// MARK: - The camera seam

/// The half of `GlassesConnection` this cartridge is allowed to reach.
///
/// Five members, all of which already existed or were added for this: there is
/// no camera behaviour here that `GlassesConnection` does not already own, and
/// there is deliberately no `pause`.
///
/// **This app has no way to pause a stream.** `GlassesConnection` calls exactly
/// four things on DAT for capture — `session.start()`, `session.stop()`,
/// `stream.start()` and `camera.stop()` — and none of them pauses one, so a
/// `pauseCapture()` here would be a method with nothing to call. That is a fact
/// about this app's code, and it is the only fact needed: this protocol can
/// only expose what `GlassesConnection` already owns.
///
/// It is **not** a claim that DAT exposes no pause, which an earlier version of
/// this comment made and could not support — the evidence offered was the list
/// of calls this app happens to make. What is separately documented, in
/// `docs/05-DAT-INTEGRATION.md` §104-107, is the pause that does exist: a
/// *device-initiated* one, from a temple or cap-touch press or heat, which
/// stops delivery, keeps the connection, and resumes on its own. That is
/// `CaptureClaim.devicePaused`, it is something this app observes rather than
/// commands, and the copy beside Pause says so.
@MainActor
protocol ObjectMemoryCaptureOwner: AnyObject {
    /// What claim this app holds on the glasses camera right this moment.
    var captureClaim: CaptureClaim { get }

    /// Every later reading of `captureClaim`. Subscribed rather than polled, so
    /// a capture that ends underneath this screen — glasses folded, Bluetooth
    /// dropped, Stop pressed on Home — reaches the phase without a timer.
    var captureClaimUpdates: AnyPublisher<CaptureClaim, Never> { get }

    /// Why the last `startCameraSession()` did not proceed, or `nil` if it did.
    /// Read immediately after the call; see `CaptureStartRefusal`.
    var lastCaptureStartRefusal: CaptureStartRefusal? { get }

    func startCameraSession()
    func stopCameraSession()
}

/// Whether THIS APP started the capture that is running, kept somewhere that
/// outlives a screen.
///
/// ## Why this is not a property on a view model
///
/// `ObjectMemoryRecordingCoordinator` holds the same fact as
/// `startedTheCamera`, and it is a `let` on `ProjectManager` for a stated
/// reason: a workspace's `@StateObject` is destroyed the moment the wearer
/// opens a different cartridge, and rebuilt when they come back believing it
/// started nothing.
///
/// Document Memory shipped that fact inside its view model, and the
/// consequence is exactly what the rule predicts. A wearer starts Document
/// Memory, presses Start -- the camera is now streaming and this app started
/// it -- and switches cartridges. The view model is destroyed with no
/// `deinit` that stops anything, so the camera keeps streaming. Coming back
/// builds a new view model with the flag false, its panel then says "The
/// camera is streaming from another screen", which is untrue, and its Stop
/// sends only the Tower verb because the branch that would call
/// `stopCameraSession()` is guarded on the flag it just lost. The capture can
/// then only be ended from Home.
///
/// A reference type held by `ProjectManager` and handed to the view model
/// fixes the lifetime without moving any of the decision-making: who may
/// start and stop the camera is unchanged, and only the memory of having
/// done so now lives long enough to be true.
@MainActor
final class CartridgeCameraClaim {
    /// True while a capture this app started is believed to be running.
    var startedByThisApp = false

    init() {}
}

#if DEBUG
/// `GlassesConnection` already satisfies every requirement, so the conformance
/// is empty by construction — which is the point: the protocol was extracted
/// from the type rather than invented beside it, so there is no adapter layer
/// where a second camera path could grow.
///
/// An empty body is therefore the expected shape, not a guarantee. A rename on
/// `GlassesConnection`, or a member added here with a slightly different
/// signature, would legitimately need a one-line forwarding member and prove
/// nothing bad. What would be the warning sign is a body that *decides*
/// anything — branching on state, holding a flag, calling DAT — because that is
/// capture behaviour living outside the one type that owns it. If you find
/// yourself writing that here, the invariant "exactly one camera pipeline, in
/// `GlassesConnection`" is the thing being broken.
///
/// `#if DEBUG` because the members it conforms with are: the whole capture
/// surface of `GlassesConnection` is DEBUG-only, exactly as Home's and World
/// Builder's controls are. In a Release build the coordinator below is
/// constructed with no camera at all and says so.
extension GlassesConnection: ObjectMemoryCaptureOwner {}
#endif

// MARK: - What the screen may say about remembering

/// Where a run of remembering has actually got to.
///
/// **Sixteen cases rather than three, and every extra one is a refusal to
/// round.** `starting`, `waitingToBeFollowed`, `notObserved` and `remembering`
/// are four different truths that a single `isRecording: Bool` would collapse
/// into one claim, and three of the four would be false claims. The Tower's own
/// contract makes the distinction unavoidable: a Start answers 200 with
/// `state: "active"` and `attached_capture_id: null` — an honest, successful,
/// documented answer that means *nothing is being recorded yet*.
nonisolated enum ObjectMemoryRecordingPhase: Equatable, Sendable {
    /// Nothing has been asked for from this screen, and nothing has been read.
    case idle

    /// A Start is in flight: the Tower has been asked, or the camera is being
    /// asked. **Not a claim that either succeeded.**
    case starting

    /// The Tower accepted the Start, and no producer has attached to a
    /// recording yet. Legal, expected for the first seconds, and bounded — see
    /// `convergenceBudget`.
    case waitingToBeFollowed

    /// The wait ran out, or a later read found the same thing: the session is
    /// active and `following` is empty.
    ///
    /// **This is the case the whole type exists for.** It is not an error, it
    /// is not "recording", and it is not "stopped". It is *asked for, and not
    /// observed*, and it is what a Start against a Tower whose producer never
    /// attached looks like from here.
    case notObserved

    /// A producer is alive on a recording. The only phase that may be drawn as
    /// recording, and it is read from `following`.
    case remembering

    case pausing
    /// Pause honoured, and nothing is being followed.
    case paused
    case resuming
    case stopping
    /// Stop honoured, and nothing is being followed.
    case stopped

    /// The Tower reported the action as honoured **and** a producer alive on a
    /// recording. The reproduced `SIGTERM` failure; shown loudly, never
    /// reconciled.
    case stillFollowing(after: CartridgeSessionAction)

    /// The session could not be read. **The liveness claim is dropped**, not
    /// held: a screen that keeps saying "remembering" after it stopped being
    /// able to check is asserting something it no longer knows.
    case cannotTell(CartridgeFailure)

    /// A 409. An answer, not a failure — it says which control would have
    /// worked.
    case refused(CartridgeSessionRefusal)

    /// The Tower agreed to remember and the glasses camera did not start.
    /// Carries which of the five reasons it was.
    case cameraRefused(CaptureStartRefusal)

    /// This Tower has no producer to start, or no session surface at all.
    case unsupported

    case failed(CartridgeFailure)

    /// Whether a producer is confirmed alive on a recording.
    ///
    /// `true` for exactly two phases, and both of them were read from
    /// `following`. Every other phase — including `starting`,
    /// `waitingToBeFollowed` and `notObserved`, all three of which follow a
    /// successful Start — is `false`, because a successful Start is not a
    /// recording.
    var isFollowingACapture: Bool {
        switch self {
        case .remembering, .stillFollowing: return true
        case .idle, .starting, .waitingToBeFollowed, .notObserved, .pausing, .paused,
            .resuming, .stopping, .stopped, .cannotTell, .refused, .cameraRefused,
            .unsupported, .failed:
            return false
        }
    }

    /// Whether an action asked for from this screen has not resolved yet.
    var isInFlight: Bool {
        switch self {
        case .starting, .pausing, .resuming, .stopping: return true
        default: return false
        }
    }
}

/// Everything one sentence about remembering has to be written from.
///
/// The phase alone is not enough, and the gap is the honest half: a session can
/// be `paused` while the glasses camera keeps streaming frames to the Tower,
/// because Pause detaches a *producer* and DAT has no pause at all. A reading
/// that carried only the phase would let the screen say "paused" beside a live
/// camera and never mention it.
nonisolated struct ObjectMemoryRecordingReading: Equatable, Sendable {
    let phase: ObjectMemoryRecordingPhase
    /// What this app's claim on the glasses camera is, independently of the
    /// Tower.
    let camera: CaptureClaim
    /// Whether the capture that is running was started by *this* screen. Decides
    /// whether Stop may end it — and decides which sentence is true about what
    /// Stop just did.
    let cameraStartedHere: Bool
    /// Whether this build can start a capture at all. `false` in Release, where
    /// the whole capture surface is compiled out.
    let cameraIsReachable: Bool
}

// MARK: - The coordinator

/// Start, Pause, Resume and Stop for Object Memory, composed over the Tower's
/// producer and the glasses camera.
///
/// ## The start order, and why it is this way round
///
/// 1. **`POST start` first, before any capture exists.** The contract is
///    explicit that this is supported: "Start before the camera is normal. The
///    session goes `active` with `attached_capture_id: null` and the next
///    capture to open finds the gate open." Doing it first is what makes the
///    common case lossless.
///
///    The other order loses frames or loses everything. A producer that
///    attaches to a capture already in progress runs with
///    `--attach-mode from-now` and **does not read back the earlier frames**,
///    so starting the camera first throws away however long the round trip to
///    the Tower took. Worse, if the Start were to fail after the camera came
///    up, a capture would be open against a closed gate with a camera running
///    and nothing remembering.
///
/// 2. **Then the camera, and only if nothing already holds it.** Home and World
///    Builder can each have started it, and `startCameraSession()` refuses when
///    a session exists. Calling it anyway would be asking for a refusal and
///    then having to explain one.
///
/// 3. **Then converge on `following`, with a deadline.** A 200 from Start is
///    intent. Whether a producer actually attached is `following`, and that is
///    read back, repeatedly, until it is true or the budget runs out. If the
///    budget runs out with the session `active` and nothing followed, the
///    screen says exactly that — `notObserved` — rather than claiming a
///    recording that this app has never observed.
///
/// ## Ownership
///
/// Whether *this* screen started the capture is recorded, and Stop honours it.
/// A capture started on Home is left running by Object Memory's Stop, because
/// ending it would silently reach across two other screens and turn off a
/// stream somebody else asked for. The copy says which of the two happened.
@MainActor
final class ObjectMemoryRecordingCoordinator: ObservableObject {

    /// How often `following` is re-read while a Start is converging.
    ///
    /// The same three seconds as `ObjectMemoryViewModel.livenessRefreshInterval`.
    /// Written out rather than read from it because that one is isolated to the
    /// main actor and this default argument is not — so the two are tied
    /// together by `testTheConvergenceCadenceMatchesTheWatchLoop` instead,
    /// which fails the day either moves.
    nonisolated static let convergenceInterval: Duration = .seconds(3)

    /// How long a Start may go unobserved before the screen says so.
    ///
    /// **Bounded, deliberately, and not a timeout in the failure sense**
    /// (Rule 15). Nothing is cancelled when it expires and nothing is reported
    /// as broken: the session stays exactly as the Tower has it, and the only
    /// thing that changes is the sentence on screen, from "waiting" to "asked
    /// for, and not observed". Twelve seconds is four reads at the interval
    /// above — long enough for a producer to spawn and open its first record,
    /// short enough that a person is not left watching a spinner over a
    /// producer that never arrived.
    nonisolated static let convergenceBudget: Duration = .seconds(12)

    /// Where a run of remembering has got to. Never set from a button tap —
    /// only from an answer, or from the absence of one at a deadline.
    @Published private(set) var phase: ObjectMemoryRecordingPhase = .idle

    /// This app's claim on the glasses camera, kept live by a subscription so
    /// that a capture ending underneath this screen is visible here without a
    /// poll.
    @Published private(set) var cameraClaim: CaptureClaim

    /// Whether the capture currently running was started by this screen.
    @Published private(set) var startedTheCamera = false

    /// Whether a **mutating** step asked for here is in flight: the `POST` to
    /// the Tower, and the `startCameraSession()` / `stopCameraSession()` call
    /// beside it. Drives the controls' `disabled`.
    ///
    /// ## Why this is the mutating step and not the whole sequence
    ///
    /// It used to be the whole sequence, and a Start's sequence ends with
    /// `converge()`, which runs to `convergenceBudget` — **twelve seconds**.
    /// For those twelve seconds both `.disabled` gates in
    /// `ObjectMemoryWorkspaceView` were live, so every control on the screen,
    /// **including Stop**, was dead while the Tower session was open and the
    /// camera this screen had just started was streaming. The only way to stop
    /// being recorded was to leave the screen. That is precisely the failure
    /// this whole composed control was justified by removing, reintroduced by
    /// the guard meant to protect it.
    ///
    /// The double tap worth protecting against is two `POST`s and two camera
    /// calls racing over `startedTheCamera` — not two polls. So the gate covers
    /// exactly the mutating part and is dropped the moment `converge()` begins;
    /// a Stop that lands during convergence cancels the poll and runs. See
    /// `perform`.
    @Published private(set) var isActing = false

    /// Whether a sequence started here is running **at all**, convergence poll
    /// included.
    ///
    /// Not what the controls gate on — see `isActing` — and deliberately a
    /// second flag rather than a widening of it. It has one job in production:
    /// `sessionChanged` must not adopt a reading pushed by the workspace's
    /// watch loop while a sequence is reading the client directly and in order.
    /// During a Start's convergence that matters more than ever, because the
    /// watch loop polls the same session at the same cadence and `resting`
    /// would turn a legal, expected "accepted, nothing attached yet" into
    /// `notObserved` seconds before the deadline that word belongs to.
    ///
    /// Published because a test has to be able to wait for a sequence to
    /// finish, and `isActing` no longer answers that question.
    @Published private(set) var isSequenceRunning = false

    /// Called when a run of remembering ends, so the records list below can
    /// show what was just written.
    ///
    /// A closure rather than a call into `ObjectMemoryViewModel`, because the
    /// category the reader narrowed to lives on the view model and re-asking
    /// for everything would silently widen their question. The view supplies
    /// the closure and keeps that decision.
    var refreshRecords: (@MainActor () -> Void)?

    private let client: any ObjectMemoryClient
    private let camera: (any ObjectMemoryCaptureOwner)?
    private let convergenceInterval: Duration
    private let convergenceBudget: Duration
    private var cancellables: Set<AnyCancellable> = []
    /// The sequence currently running. Bounded by `convergenceBudget`, so the
    /// strong reference it holds to this object is bounded too.
    ///
    /// Cancelled — for real, and this is the only thing that cancels it — when
    /// a later verb supersedes it in `perform`. A Stop pressed while a Start is
    /// converging is the case that matters.
    private var work: Task<Void, Never>?

    /// Which run `work` is. Incremented by every `perform`.
    ///
    /// Cancellation in Swift is cooperative, so a superseded run does not stop
    /// where it was told to — it resumes on the main actor, unwinds, and
    /// reaches the completion block *after* the run that replaced it has
    /// already set its own flags. Without this it would clear `isActing` and
    /// `isSequenceRunning` for a sequence that is still mutating things, and
    /// the double-tap guard would be open during exactly the window it exists
    /// to close.
    private var run = 0

    /// - Parameters:
    ///   - camera: the app's single camera owner, or `nil` in a build or a
    ///     preview that has none. `nil` is a supported state, not a degraded
    ///     one: the Tower half still works and the copy says the camera half
    ///     cannot be reached.
    ///   - client: the Object Memory client, which owns the session transport
    ///     and does the read-back after every action.
    ///   - interval / budget: injectable so a test can converge in
    ///     milliseconds. Nothing else may change them.
    init(
        camera: (any ObjectMemoryCaptureOwner)?,
        client: any ObjectMemoryClient,
        interval: Duration = ObjectMemoryRecordingCoordinator.convergenceInterval,
        budget: Duration = ObjectMemoryRecordingCoordinator.convergenceBudget
    ) {
        self.camera = camera
        self.client = client
        self.convergenceInterval = interval
        self.convergenceBudget = budget
        self.cameraClaim = camera?.captureClaim ?? .unclaimed

        // Same shape as `ObjectMemoryViewModel`'s two subscriptions, and for
        // the same reason: the publisher is the only thing that reports a
        // capture ending for a reason nobody on this screen asked for.
        camera?.captureClaimUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] claim in self?.cameraClaimChanged(claim) }
            .store(in: &cancellables)

        // Delivered where it is sent, and deliberately without a
        // `.receive(on:)` hop. `sessionChanged`'s gate asks "is a sequence
        // running *now*", and a hop makes that question unanswerable: the
        // read-back a sequence's own `apply` publishes would arrive a turn
        // after the sequence had ended, with the gate already down, and
        // `resting` would replace the verdict that sequence just wrote — a
        // camera refusal, or a failure — with the reading that produced it.
        // `ObjectMemoryClient` is `@MainActor`, so there is no thread to hop
        // from and the hop was only ever costing this ordering.
        client.sessionUpdates
            .sink { [weak self] session in self?.sessionChanged(session) }
            .store(in: &cancellables)
    }

    // **There is deliberately no `deinit` cancelling `work`.** There used to
    // be, and it was unreachable code that read as a safety net. `perform`
    // builds its task around a closure that captures `self` strongly — it has
    // to, because the body is a method call on this object — so a running
    // sequence keeps the coordinator alive until it returns, and `deinit`
    // cannot run while `work` is unfinished. A cancel there could never observe
    // anything to cancel.
    //
    // The cancellation that is real happens in `perform`, where a later verb
    // supersedes an earlier sequence's convergence poll. That is what makes
    // `converge()`'s `Task.isCancelled` check and its `catch` around
    // `Task.sleep` live branches rather than decoration, and it is the only
    // thing that reaches them.
    //
    // The lifetime is bounded regardless: `converge()` runs to
    // `convergenceBudget` and every other step is one request, so the strong
    // reference a sequence holds is measured in seconds. And in this app the
    // question is moot in the other direction too — `ProjectManager` owns this
    // object for the life of the process, which is exactly why the view may not
    // leave a closure on it (see `refreshRecords`).

    // MARK: What the screen reads

    /// The whole truth, in one value. See `ObjectMemoryRecordingReading`.
    var reading: ObjectMemoryRecordingReading {
        ObjectMemoryRecordingReading(
            phase: phase,
            camera: cameraClaim,
            cameraStartedHere: startedTheCamera,
            cameraIsReachable: camera != nil
        )
    }

    /// Which verb the primary button sends.
    ///
    /// Start whenever nothing is under way, Stop whenever something is — including
    /// `notObserved`, where a session exists on the Tower even though nothing
    /// has been observed following it. Offering Start there would leave no way
    /// to clear a session that is open and doing nothing.
    ///
    /// **`cameraRefused` is in the Stop list for that same reason, and it used
    /// to be in the Start list against it.** Every branch that produces
    /// `cameraRefused` leaves the Tower session `active` on purpose — the
    /// session is a gate, not a recording, and tearing it down because the
    /// camera half was refused would throw away correct work (see
    /// `startSequence`). So `cameraRefused` *is* the shape the rule describes:
    /// a session open on the Tower with nothing happening under it. Offering
    /// Start there offered the one verb that changes nothing, and left the open
    /// session with no control that could close it.
    ///
    /// The rule, stated once so the two lists cannot drift from it: **Stop is
    /// offered wherever this app has reason to believe a session exists on the
    /// Tower.** `idle` and `stopped` are the two phases where it does not.
    /// `cannotTell`, `refused`, `unsupported` and `failed` are phases where
    /// this app could not establish one, and Start is the honest offer there
    /// because a Stop against a session that was never opened is a request with
    /// nothing behind it.
    var primaryAction: CartridgeSessionAction {
        switch phase {
        case .starting, .waitingToBeFollowed, .notObserved, .remembering, .stillFollowing,
            .pausing, .paused, .resuming, .stopping, .cameraRefused:
            return .stop
        case .idle, .stopped, .cannotTell, .refused, .unsupported, .failed:
            return .start
        }
    }

    // MARK: Asking

    /// Sends one verb through the composed lifecycle.
    ///
    /// The vocabulary is still read off the Tower's own `actions` list by the
    /// view; this is the only place that decides what each verb *does* to the
    /// camera, which is why the view no longer reaches `ObjectMemoryClient.apply`
    /// directly.
    func apply(_ action: CartridgeSessionAction) {
        switch action {
        case .start: start()
        case .pause: pause()
        case .resume: resume()
        case .stop: stop()
        }
    }

    /// Starts remembering: the Tower first, then the camera, then convergence.
    func start() {
        perform { await self.startSequence() }
    }

    /// Stops remembering: the Tower first, then the camera **if this screen
    /// started it**, then a refresh of the records.
    func stop() {
        perform { await self.stopSequence() }
    }

    /// Stops remembering without touching the camera.
    ///
    /// **Pause is a Tower verb only.** There is no way to pause a DAT stream,
    /// so if a capture is running it keeps running and keeps sending frames to
    /// the Tower — they are simply no longer read into this memory. The copy
    /// says that in as many words whenever it is true, because "paused" beside
    /// a live camera is otherwise read as "the camera stopped".
    func pause() {
        perform { await self.towerOnly(.pause, whileAsking: .pausing) }
    }

    /// Resumes remembering. From `stopped` this is a 409, which is an answer —
    /// it says Start is the control that works from there — and is reported as
    /// one rather than as a failure.
    func resume() {
        perform { await self.towerOnly(.resume, whileAsking: .resuming) }
    }

    // MARK: The sequences

    private func startSequence() async {
        phase = .starting

        // 1. The Tower, first and always. See the type's documentation for why
        //    this order is not interchangeable with the one below it.
        await client.apply(.start)
        // A second Start answers 200 with `changed: false`. That is the Tower
        // saying "you already have what you asked for" and it arrives here as
        // `.known`, indistinguishable from the first Start — which is correct,
        // and is why there is no branch on `changed` anywhere in this file.
        if let stopped = refusalOrFailure(client.session) {
            phase = stopped
            return
        }

        // 2. The camera, only if nothing already holds it.
        if let camera {
            // Asked of the camera directly rather than read off the published
            // mirror below. The mirror is a *display* value delivered through a
            // main-queue hop, and deciding whether to start a second capture
            // from a value that is one turn stale is exactly the kind of race
            // this sequencing exists to remove.
            switch camera.captureClaim {
            case .unclaimed:
                camera.startCameraSession()
                if let refusal = camera.lastCaptureStartRefusal {
                    // The Tower is left `active` deliberately. It is a gate,
                    // not a recording, and the next capture to open — from
                    // here, from Home, or from World Builder — finds it open.
                    // Stopping it here would throw away a correct half of the
                    // work because the other half was refused.
                    phase = .cameraRefused(refusal)
                    return
                }
                startedTheCamera = true
                // Kept in step immediately rather than waiting for the
                // publisher's hop, so the sentence drawn on the very next
                // render already describes the capture that was just started.
                cameraClaim = camera.captureClaim
            case .devicePaused:
                // A hardware pause. `startCameraSession()` would refuse this
                // too, and there is nothing this app may call to override it —
                // the glasses resume delivery on their own.
                phase = .cameraRefused(.deviceHasPausedCapture)
                return
            case .running:
                // Somebody else's capture — Home's or World Builder's — and
                // ownership is *not* claimed here, so Stop below leaves their
                // stream alone.
                //
                // Ownership is not *surrendered* here either. `.running` says
                // a capture exists; it does not say who opened it, and this
                // may well be the capture this screen started, because every
                // phase that maps to `primaryAction == .start` offers the
                // button back and a Tower that blips off Wi-Fi for one poll
                // is enough to produce one. Clearing the flag on that second
                // Start made the Stop that followed skip
                // `stopCameraSession()`: the Tower session ended, the wearer
                // was told remembering had stopped, and the glasses kept
                // recording. A capture that ends drops ownership through
                // `cameraClaimChanged`, which is where that belongs.
                break
            case .ending:
                // **Nobody owns a capture that is dying.** This used to share
                // the branch above and record "somebody else started it",
                // which was false in both halves: the previous owner has
                // already let go, and this screen got no camera and said
                // nothing about it. `startCameraSession()` refuses in this
                // window too — its `guard deviceSession == nil` sees a session
                // that is still being torn down — so calling it would produce
                // `.alreadyRunning`, a refusal whose sentence sends a wearer
                // to look for a stream that is on its way out.
                //
                // Reported instead as what it is. Not retried: a retry would
                // need a second deadline inside a sequence that already has
                // one, and the teardown it is waiting on is DAT's, with no
                // bound this app can state. The Tower's half stands, exactly as
                // it does for every other camera refusal, so the next Start —
                // a second or two later, when the claim has reached
                // `.unclaimed` through `captureClaimUpdates` — is a real one.
                phase = .cameraRefused(.captureIsShuttingDown)
                return
            }
        }

        // 3. Converge on `following`, bounded.
        //
        //    The gate comes off here, before the poll and not after it. From
        //    this line on the sequence only *reads*: there is nothing left for
        //    a second tap to race, and there is a camera streaming that a
        //    person must be able to stop. See `isActing`.
        isActing = false

        // Whether the camera half of *this* run is the one being watched
        // below. Captured now rather than read off `startedTheCamera` inside
        // the loop, because `cameraClaimChanged` clears that flag as soon as
        // the failed capture's teardown reaches the publisher — which is the
        // same event that sets the refusal, and would race it away.
        await converge(watchingACameraStartedHere: startedTheCamera)
    }

    private func stopSequence() async {
        phase = .stopping

        // The Tower first: `stop` is never refused from any state, and it is
        // what detaches the producer so it can finalise the record it is
        // holding. Killing the frames first would leave the producer to notice
        // the stream ending on its own.
        await client.apply(.stop)

        // Only a camera this screen started. A capture Home started is somebody
        // else's, and ending it from here would reach across two screens.
        if startedTheCamera, let camera {
            camera.stopCameraSession()
            startedTheCamera = false
            cameraClaim = camera.captureClaim
        }

        phase = resting(client.session)

        // The point of stopping is usually to look at what was just written.
        refreshRecords?()
    }

    /// Pause and Resume: the Tower, a read-back, and nothing else.
    private func towerOnly(
        _ action: CartridgeSessionAction, whileAsking asking: ObjectMemoryRecordingPhase
    ) async {
        phase = asking
        await client.apply(action)
        phase = resting(client.session)
    }

    /// Re-reads the session until a producer is observed following a capture,
    /// or until the budget runs out.
    ///
    /// The loop reads what `ObjectMemoryClient.apply` already read back before
    /// asking for anything more, so a Start that attached immediately settles
    /// without a single extra request. Every wait is a real `Task.sleep` with a
    /// deadline above it; there is no unbounded poll and no retry budget that
    /// can be refilled from inside.
    ///
    /// **Nothing here mutates anything**, which is why `isActing` is already
    /// false by the time this runs and why a Stop may arrive in the middle of
    /// it. A Stop cancels this task — `perform` does that — and the two exits
    /// below (`Task.isCancelled`, and the `catch` around the sleep) are the
    /// paths it takes out. Neither writes a phase: a cancelled wait produced no
    /// verdict, and the sequence that superseded it is writing its own.
    ///
    /// - Parameter watchingACameraStartedHere: whether this run started a
    ///   capture. See the camera re-check below.
    private func converge(watchingACameraStartedHere: Bool) async {
        phase = .waitingToBeFollowed
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: convergenceBudget)

        while !Task.isCancelled {
            if let settled = self.settled(client.session) {
                phase = settled
                return
            }
            // **The camera can still refuse after `startCameraSession()` has
            // returned, and one refusal only ever arrives that way.**
            //
            // `startCameraSession()` sets `lastCaptureStartRefusal`
            // synchronously for `.alreadyRunning`, `.deviceHasPausedCapture`,
            // `.noActiveDevice` and `.datRefused`. Camera permission is not in
            // that list: the session starts, DAT's state observer fires,
            // `beginCameraStream` finds the permission ungranted and
            // `abandonSessionAfterFailedStart` writes the refusal — all of it
            // after the call this sequence made had already returned `nil`.
            //
            // So the read at the call site claimed ownership, and twelve
            // seconds later this loop reported `notObserved` — "asked for, and
            // not observed" — while the true answer was known, specific,
            // actionable and had a written sentence that could never be
            // reached. Re-read here so it is.
            //
            // Only for a capture this run started. `lastCaptureStartRefusal`
            // is cleared at the top of every `startCameraSession()` and
            // survives until the next one, so on a run that started nothing it
            // may hold Home's answer to Home's question.
            if watchingACameraStartedHere, let camera,
                let refusal = camera.lastCaptureStartRefusal {
                // The capture this run believed it owned is gone with it.
                startedTheCamera = false
                cameraClaim = camera.captureClaim
                phase = .cameraRefused(refusal)
                return
            }
            guard clock.now < deadline else {
                // Asked for, and not observed. Said plainly rather than
                // dressed as an error: the session is exactly where the Tower
                // reports it, and the only thing that has happened is that this
                // app has still never seen a producer following anything.
                phase = .notObserved
                return
            }
            do {
                try await Task.sleep(for: convergenceInterval)
            } catch {
                // Cancellation, which is the only way out of the sleep. The
                // phase is left where it is rather than being given a verdict
                // nothing produced.
                return
            }
            await client.readSession()
        }
    }

    // MARK: Reading a session into a phase

    /// Runs one action at a time — with "at a time" meaning *one mutating step
    /// at a time*, not one sequence.
    ///
    /// ## The double-tap guard, narrowed to what it is actually protecting
    ///
    /// A second tap while a **mutating** step is in flight is dropped entirely
    /// — not queued, not sent — because two overlapping Start sequences would
    /// race over `startedTheCamera` and could leave this screen believing it
    /// owns a capture it did not start. That is a local fact rather than a
    /// guess about the Tower, which is why gating on it does not make this
    /// app's model authoritative over the Tower's. A tap *after* a sequence
    /// finishes is sent normally, and the Tower answers a repeated verb 200
    /// with `changed: false`, which is not an error and is never drawn as one.
    ///
    /// ## Supersession, and why it is not the same as a dropped tap
    ///
    /// The gate used to cover the whole sequence, and a Start's sequence ends
    /// with a twelve-second convergence poll. **Every control on the screen,
    /// Stop included, was disabled for those twelve seconds** — over a live
    /// Tower session and a camera this screen had just started. See `isActing`.
    ///
    /// So `converge()` runs with the gate down, and a verb that arrives during
    /// it **cancels the poll and takes over**. That is right for all four:
    /// Stop is the one that matters and must always be reachable; Pause and
    /// Resume are answers about the same session and belong to the newer
    /// intent; a second Start re-`POST`s a verb the Tower answers
    /// `changed: false` and converges again. Nothing is queued, ever — a
    /// person's most recent tap is the one this screen acts on.
    ///
    /// The cancelled run does not stop where it was told to; it unwinds
    /// through the main actor and reaches its completion block afterwards,
    /// which is what `run` is for.
    private func perform(_ body: @escaping @MainActor () async -> Void) {
        guard !isActing else {
            print("[Glasses][ObjectMemory] an action was already in flight; the second tap was dropped")
            return
        }
        // Only ever a sequence that has reached its convergence poll — a
        // mutating one would have been dropped by the guard above.
        work?.cancel()

        run &+= 1
        let thisRun = run
        isActing = true
        isSequenceRunning = true
        // `body` captures `self` strongly, so this task keeps the coordinator
        // alive until the sequence returns. Said out loud because it is the
        // reason there is no `deinit` cancel — see the note above `reading`.
        work = Task { [weak self] in
            await body()
            guard let self, self.run == thisRun else { return }
            self.isActing = false
            self.isSequenceRunning = false
        }
    }

    /// The phase for a reading that ends a wait, or `nil` while the only honest
    /// answer is "asked for, and nothing observed yet".
    ///
    /// `nil` is returned for exactly one shape — an `active` session with an
    /// empty `following` — and that shape is the reason this function is
    /// separate from `resting`. During a Start it means *keep waiting*; at rest
    /// it means *notObserved*. Same payload, two different true sentences, and
    /// the difference is whether a deadline has passed.
    private func settled(_ session: ObjectMemorySessionState) -> ObjectMemoryRecordingPhase? {
        switch session {
        case .unread, .working:
            return nil
        case .noSessionControl:
            return .unsupported
        case .failed(let failure):
            return .cannotTell(failure)
        case .refused(let refusal):
            return .refused(refusal)
        case .known(let snapshot):
            guard snapshot.supported else { return .unsupported }
            // Liveness, from `following`, and from nothing else.
            if snapshot.isFollowingACapture {
                if snapshot.state == .paused { return .stillFollowing(after: .pause) }
                if snapshot.state == .stopped { return .stillFollowing(after: .stop) }
                return .remembering
            }
            if snapshot.state == .paused { return .paused }
            if snapshot.state == .stopped { return .stopped }
            // `active` with nothing followed, and every state this build does
            // not recognise. Both are "the Tower reports no producer on a
            // recording", which is what `notObserved` says — and the session
            // panel below still prints the raw intent uninterpreted.
            return nil
        }
    }

    /// The phase for a reading with no wait outstanding.
    private func resting(_ session: ObjectMemorySessionState) -> ObjectMemoryRecordingPhase {
        switch session {
        case .unread:
            // Never read. Not `stopped`: silence is not an answer.
            return .idle
        case .working:
            // A request is in flight somewhere. Whatever was last true is still
            // the best thing known, and replacing it would flash a phase that
            // describes nothing.
            return phase
        case .known, .refused, .noSessionControl, .failed:
            return settled(session) ?? .notObserved
        }
    }

    /// The phases that must stop a Start sequence before it touches the camera.
    /// A refusal, a failure, or a Tower with no producer are all answers that
    /// make starting a capture pointless — and starting one anyway would leave
    /// a camera running for a memory that cannot be written into.
    private func refusalOrFailure(
        _ session: ObjectMemorySessionState
    ) -> ObjectMemoryRecordingPhase? {
        switch session {
        case .refused(let refusal): return .refused(refusal)
        case .failed(let failure): return .failed(failure)
        case .noSessionControl: return .unsupported
        case .known(let snapshot): return snapshot.supported ? nil : .unsupported
        case .unread, .working: return nil
        }
    }

    // MARK: Reacting to things nobody on this screen asked for

    /// A session reading that arrived from the workspace's own watch loop.
    ///
    /// This is what makes a producer that dies, a Tower that goes away, and a
    /// Pause honoured by somebody else all reach the phase. In particular a
    /// failed read becomes `cannotTell` and **clears the liveness claim**: a
    /// screen that keeps saying "remembering" after it lost the ability to
    /// check is asserting something it does not know.
    private func sessionChanged(_ session: ObjectMemorySessionState) {
        // While a sequence is running it is reading the client directly and in
        // order. A pushed reading here would race that sequence — the
        // `.working` and post-`POST` states both arrive on this publisher —
        // and could put "remembering" back on screen between a Stop's `POST`
        // and its read-back.
        //
        // `isSequenceRunning` and **not** `isActing`, which now covers only the
        // mutating step. The difference is a Start's convergence, and it is not
        // a corner case: the workspace's own watch loop polls this same session
        // at `livenessRefreshInterval`, which is the same three seconds the
        // convergence poll uses, so a reading lands here two or three times
        // during every Start. `resting` turns an `active` session with nothing
        // following into `notObserved` — the legal, expected, documented shape
        // of the first seconds after a Start — and would print "asked for, and
        // not observed" while the wait it describes was still running.
        // `converge()` is reading the same field on the same cadence and owns
        // the verdict.
        guard !isSequenceRunning else { return }
        phase = resting(session)
    }

    /// The camera's claim changed underneath this screen.
    ///
    /// Ownership is dropped when the capture ends, whoever ended it. Keeping
    /// `startedTheCamera` true across a capture that Home stopped would make a
    /// later Stop here call `stopCameraSession()` against whatever session
    /// exists by then, which might be one this screen never started.
    private func cameraClaimChanged(_ claim: CaptureClaim) {
        cameraClaim = claim
        if claim == .unclaimed { startedTheCamera = false }
    }
}
