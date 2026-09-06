//
//  ExperimentalCVWorkspaceView.swift
//  Glasses
//

import Foundation
import SwiftUI

/// The Experimental CV Lab workspace: a measurement surface for experiments the
/// Tower runs.
///
/// ## What this screen may and may not claim
///
/// It used to open by saying the Tower "runs exactly one experiment, and it
/// chose which one when it started", that there was "no message to list the
/// experiments, pick a different one, or start or stop the one that is
/// running", and that "the reply names the *number*, not the experiment, so
/// this screen cannot even say which one produced it".
///
/// **Every one of those sentences is now false.** `cv_lab_status` lists the
/// experiments; `cv_lab_start` selects and arms one, replacing whatever ran;
/// pause, resume and stop are requests with legible refusals; and every
/// `frame_result` carries a `cv_lab` block naming the experiment, the run and
/// the provenance. So the copy is gone along with the limitation it described.
/// What replaced it is not a promise — it is the Tower's own document, drawn
/// field for field.
///
/// Two things this screen still refuses to do, and both are the Tower's
/// position as much as this app's:
///
/// 1. **No better/worse verdict.** `baseline`, `higher_is_better` and
///    `confidence` are `null` on every metric, always, because the Lab holds no
///    reference run to compare against. The machinery that would have rendered
///    a verdict is deleted rather than left dormant — see `CVMetric`.
/// 2. **No photographic frame, ever.** There *is* a live view now — see
///    `CVLivePreviewPanel` — and it is the first thing on the screen. What it
///    draws is derived: an edge map, a colourised depth map, or an overlay
///    over a line drawing of the room. The Tower serves no photograph on this
///    path, every picture states its treatment, and this app draws none whose
///    treatment it could not read. An experiment still gets no privacy
///    exemption for being a debug surface; it got a picture that does not need
///    one.
///
/// ## Debug and Release are genuinely different products here
///
/// The frame path is `#if DEBUG` (`ProjectManager`), so **a Release build has
/// no camera, sends no frame, and receives no `frame_result`.** That predates
/// this work and is not fixed by it. What this screen does about it:
///
/// - the read-only half is drawn in full, because it is fully reachable: the
///   declaration, the subscription and the status document do not depend on
///   frames;
/// - no control that starts an experiment is drawn at all, because there is no
///   `startCameraSession` in this configuration to feed the experiment it would
///   arm — see `TowerExperimentalCVClient.canSendCommands`;
/// - `.running` is never drawn as **live**, because live requires this build to
///   be streaming and `source.receiving_frames` to be true, and the first half
///   is permanently false here.
///
/// ## The camera card, and what it does and does not change
///
/// This workspace used to carry no capture control at all, and said so here.
/// It now mounts a `CVCameraCard`, because a person running an experiment
/// otherwise had to leave for Home to start the camera that feeds it and come
/// back to read the result — three screens for one question. What has **not**
/// changed is the shape of the invariant: one button, the one
/// `GlassesConnection`, reached through `startCameraSession()` and
/// `stopCameraSession()` exactly as Home and World Builder reach it, and no
/// `.onAppear` anywhere. Starting an *experiment* still opens no camera — that
/// is a request to the Tower about what to compute — and the card's own Pause
/// is a frame gate on `TowerClient`, not a camera operation at all; see
/// `TowerClient.isFrameSendingPaused` for why.
struct ExperimentalCVWorkspaceView: View {
    /// A value, not the connection. Reachability changes almost never, and
    /// `TowerReachabilityReader` — which supplies this — exists so a workspace
    /// does not observe a 12 Hz object to learn it.
    let isTowerReachable: Bool

    /// Held as a plain reference and **not** `@ObservedObject`, which is the
    /// whole trick that lets this workspace show a live figure without paying
    /// for it everywhere.
    ///
    /// `TowerClient` republishes once per reply, at the ~12 Hz target rate while
    /// a session streams. Observing it here would re-evaluate this entire body
    /// at that rate — the experiment panel, the unavailable panel, all of it —
    /// on the main actor the frame sender needs. Instead the reference is passed
    /// down to `CVFrameReadingPanel`, a leaf that holds nothing else, and only
    /// that leaf is invalidated when a reply lands. SwiftUI compares this stored
    /// reference by identity, so a reply does not re-run this body at all.
    let tower: TowerClient

    /// The camera, held the same way `tower` is and for the same reason.
    /// `GlassesConnection` publishes `frameCount` and `latestCapturedFrame` at
    /// the 24 Hz capture rate; observing it here would re-evaluate the whole
    /// workspace at that rate. `CVCameraCard` is the leaf that observes it,
    /// and the only thing in this file that touches it.
    let glasses: GlassesConnection

    @StateObject private var lab: ExperimentalCVViewModel

    /// The client is injected and owned by `ProjectManager`; see
    /// `CartridgeClients`.
    init(
        isTowerReachable: Bool,
        glasses: GlassesConnection,
        tower: TowerClient,
        client: any ExperimentalCVClient
    ) {
        self.isTowerReachable = isTowerReachable
        self.glasses = glasses
        self.tower = tower
        _lab = StateObject(wrappedValue: ExperimentalCVViewModel(client: client))
    }

    var body: some View {
        VStack(spacing: 16) {
            header

            // The socket, before anything that depends on it. Every panel
            // below reads what the Tower said; this row is where a person
            // finds out whether it can say anything, and reconnects without
            // leaving for the shell's sheet. A leaf that observes `tower` on
            // its own, so the reply-rate invalidation stops there.
            TowerConnectionRow(tower: tower)

            // FIRST among the Lab's own panels, because it is the answer to
            // the question a person wearing the glasses is actually asking. This
            // screen used to open with eleven numbers at equal weight, every
            // one of them true and none of them saying whether the algorithm
            // could see the doorway. The numbers did not go away; they stopped
            // being the first thing.
            if let live = livePreview {
                CVLivePreviewPanel(
                    preview: live.preview,
                    runID: live.runID,
                    isRunning: live.isRunning,
                    experimentName: live.name
                )
            }

            // The latest per-frame reading, under the picture. This is the
            // answer the Tower is producing right now; everything below it is
            // about choosing what produces the next one.
            CVFrameReadingPanel(tower: tower, isTowerReachable: isTowerReachable)

            // Above the Lab and its catalog, because the camera is what feeds
            // them: an experiment armed with the camera off measures nothing,
            // and the panels below say so in their own words. Drawn under
            // every availability, including an unreachable Tower — the card
            // itself says what a start would do about that.
            CVCameraCard(glasses: glasses, tower: tower)

            if let forcedPhase = lab.availability(isTowerReachable: isTowerReachable).forcedPhase {
                CartridgeStatePanel(
                    title: "Experiments",
                    phase: forcedPhase,
                    explanation: lab.unavailableExplanation(isTowerReachable: isTowerReachable)
                )
            } else {
                labPanel
                // Hidden only for `.unsupported`, which is this Tower saying it
                // cannot run experiments at all — a catalog under that sentence
                // would be a list of things that cannot be started.
                //
                // Shown for `.failed`, deliberately: **choosing another
                // experiment is the recovery.** The Lab refuses a stop from
                // `failed`, so a "try again" control would be refused on step
                // one, and the list below is the thing that actually works.
                if !isUnsupported {
                    experimentPanel
                }
            }

            if let failure = lab.lastRequestFailure {
                FailureBanner(text: failure.message)
            }
            if let refusal = lab.lastRefusal {
                CVRefusalBanner(refusal: refusal)
            }
        }
    }

    // MARK: The live view

    /// The descriptor to draw, if there is one this build may draw.
    ///
    /// Only `running` and `paused`. A `completed` run keeps its FIGURES — the
    /// Tower freezes them and says so — but its picture is gone by design: the
    /// Tower empties the slot on the way out of `running`, because
    /// `raw_ephemeral` means live-view-only in both directions and a frozen
    /// last frame under a "stopped" label reads as live to anybody not reading
    /// the label.
    ///
    /// `paused` still draws the panel, with the loop stopped. That is
    /// deliberate: the panel says "the live view runs while the experiment
    /// does" in the space the picture was in, which is a better answer than
    /// the whole panel vanishing and the layout jumping.
    private var livePreview:
        (preview: CVLivePreview, runID: String?, isRunning: Bool, name: String)?
    {
        switch lab.state {
        case .running(let run):
            guard let preview = run.annotation.drawablePreview else { return nil }
            return (preview, run.runID, true, run.experiment.name)
        case .paused(let run):
            guard let preview = run.annotation.drawablePreview else { return nil }
            return (preview, run.runID, false, run.experiment.name)
        case .unsupported, .idle, .starting, .completed, .failed:
            return nil
        }
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Experimental CV Lab")
                .font(.title2.weight(.semibold))
            Text(Self.headerSubtitle)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    // MARK: The one sentence that is still only true in one configuration
    //
    // The four strings that used to be compiled twice are down to one, and the
    // reduction is the measure of what changed. Three of them existed to
    // describe a *protocol* limitation — no list, no request, no provenance —
    // that turned out to differ between builds only in whether a result could
    // arrive to be un-labelled. The protocol limitation is gone, so those three
    // are gone.
    //
    // This one remains because the difference it describes is real and
    // structural: the frame path is `#if DEBUG`, so a Release build genuinely
    // cannot feed an experiment. Gating the whole workspace out of Release would
    // still be the wrong repair — the read-only half is honest and useful there,
    // and it is exactly what the Tower versions the control vocabulary
    // separately to support.

    #if DEBUG
    private static let headerSubtitle = """
        Vision experiments the Tower can run on the glasses feed. It lists what \
        it has, arms the one you choose, and labels every figure with the run \
        and the experiment that produced it.
        """
    #else
    private static let headerSubtitle = """
        Vision experiments the Tower can run on the glasses feed. This build has \
        no camera, so it sends the Tower no frames and can read what the Lab is \
        doing without asking it to do anything.
        """
    #endif

    // MARK: The Lab itself

    /// Lifecycle, liveness, and the controls for the run that is loaded.
    @ViewBuilder
    private var labPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("The Lab")

            VStack(alignment: .leading, spacing: 12) {
                switch lab.state {
                case .unsupported(let reason):
                    Text(reason)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                case .idle:
                    if isStillAsking {
                        // A `cv_lab_status` is genuinely in flight, which is the
                        // one condition under which a spinner is truthful.
                        // Without this branch the moment between a socket coming
                        // up and the first document arriving reads as "the Lab is
                        // idle and declares nothing", which is a claim about the
                        // Tower made before it has said anything.
                        HStack(spacing: 10) {
                            ProgressView()
                            Text("Asking the Tower what it can run…")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        Text("Nothing is armed. Choose an experiment below to start one.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }

                case .starting(let experiment):
                    HStack(spacing: 10) {
                        // The one honest spinner on this screen: something is
                        // genuinely in flight.
                        ProgressView()
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Arming \(experiment.name)…")
                                .font(.subheadline)
                            // Bounded, and unreportable. Said out loud because a
                            // model download can take a minute and a half and a
                            // screen that says nothing about it reads as stuck.
                            Text(Self.armingCaveat)
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }

                case .running(let run):
                    runHeader(run, isPaused: false)
                    CVRunSummaryView(run: run, isFinal: false)
                    controls(for: run, isPaused: false)

                case .paused(let run):
                    runHeader(run, isPaused: true)
                    // Not "the run ended". A paused run keeps the experiment
                    // loaded — resuming a model-backed one costs nothing, while
                    // a stopped one pays the load again — and it is still
                    // counting: the metrics stand still while `framesRefused`
                    // climbs with every frame that arrives.
                    Text("Paused. The experiment is still loaded, so resuming costs nothing.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    CVRunSummaryView(run: run, isFinal: false)
                    controls(for: run, isPaused: true)

                case .completed(let run):
                    Label(run.experiment.name, systemImage: "checkmark.circle")
                        .font(.headline)
                    // The Tower's word is `stopped`, and the distinction it is
                    // making is worth keeping: a bench run does not complete, a
                    // person ends it.
                    Text("Stopped. These figures are final and stay until the next start.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    CVRunSummaryView(run: run, isFinal: true)

                case .failed(let failure):
                    Label("The last start failed", systemImage: "exclamationmark.triangle.fill")
                        .font(.headline)
                    Text(failure.message)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    // No "Stop, then Start" recovery, and not for tidiness:
                    // **the Lab refuses `cv_lab_stop` from `failed`**, so that
                    // path is refused on step one. Another start is the
                    // recovery, and choosing one below is how it is sent.
                    Text(Self.failedRecovery)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                sourceRows
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        }
    }

    /// Whether the Tower has said anything about the Lab yet.
    ///
    /// Three conditions rather than one, because a client that carries its own
    /// catalog — the unavailable stand-in, and the fixtures in the test target —
    /// has genuinely answered without ever holding a document. Only a state that
    /// is idle, a catalog that is empty and a document that is absent together
    /// mean "nothing has come back yet".
    private var isStillAsking: Bool {
        lab.status == nil && lab.availableExperiments.isEmpty
    }

    /// Whether this Tower says it cannot run experiments at all.
    private var isUnsupported: Bool {
        if case .unsupported = lab.state { return true }
        return false
    }

    /// The run's name, and whether this build may call it live.
    ///
    /// > `.running` may be shown as LIVE only when this build is itself
    /// > streaming **and** `source.receiving_frames` is true.
    ///
    /// Both halves, every time. `source` is **Tower-wide**, so it reads `true`
    /// for a build with no camera whenever a second phone is attached — and
    /// this phone's own bracket is the half that catches that. The other way
    /// round, a phone streaming to a Lab that has seen nothing for five seconds
    /// is not producing results either. Anything short of both is "armed",
    /// which is a true statement about the Tower and no statement at all about
    /// where the frames are.
    @ViewBuilder
    private func runHeader(_ run: CVExperimentRun, isPaused: Bool) -> some View {
        let receiving = lab.source?.receivingFrames ?? false
        // A hold is this phone not streaming, and this phone knows it before
        // the Tower does: `receiving_frames` stays true for `idle_after_s`
        // (5 s) after the last frame, and LIVE for those seconds above a card
        // that reads "held on phone" would be the screen disagreeing with
        // itself.
        let isLive = lab.state.isLive(
            isStreaming: tower.isStreamingToTower && !tower.isFrameSendingPaused,
            isReceivingFrames: receiving
        )
        VStack(alignment: .leading, spacing: 4) {
            Label(
                isLive
                    ? "\(run.experiment.name) · live"
                    : (isPaused ? "\(run.experiment.name) · paused" : "\(run.experiment.name) · armed"),
                systemImage: isLive ? "dot.radiowaves.left.and.right" : "pause.circle"
            )
            .font(.headline)

            if !isPaused && !isLive {
                Text(notLiveExplanation(isReceivingFrames: receiving))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // Reported so that "the Lab is running" never reads as "somebody
            // chose this". The Tower arms a startup default at boot, and a run
            // nobody asked for looks identical to one somebody did without
            // this line.
            if run.origin == "startup_default" {
                Text("The Tower armed this at startup. Nobody chose it from here.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// Which half of "live" is missing, said in the words that fit the build.
    private func notLiveExplanation(isReceivingFrames: Bool) -> String {
        #if DEBUG
        if tower.isFrameSendingPaused {
            // The camera card directly above says "Paused (held on phone)";
            // telling the user to check the glasses here would contradict it.
            return "Frames are held on the phone — see the Camera card above. Resume to feed it."
        }
        if !tower.isStreamingToTower {
            return isReceivingFrames
                ? """
                    Armed and processing, but the frames are not coming from this \
                    phone — this session is not streaming, and another client is \
                    feeding the Tower.
                    """
                : "Armed and waiting for a stream. Start the camera above to feed it."
        }
        return """
            Armed, and the Tower has not seen a frame recently enough to call \
            itself fed. Check that the glasses are streaming.
            """
        #else
        return """
            Armed on the Tower. This build has no camera, so none of the frames \
            it is measuring are from this phone.
            """
        #endif
    }

    /// Pause, resume and stop — drawn only where a command can be sent.
    ///
    /// Each carries the run id it was drawn against, one layer down in the
    /// client. That is what turns "this button was drawn against a run that has
    /// since been replaced" into a `stale_run` refusal naming the current run,
    /// instead of stopping somebody else's experiment.
    @ViewBuilder
    private func controls(for run: CVExperimentRun, isPaused: Bool) -> some View {
        if lab.canSendCommands {
            // Every control is disabled while any command is out, not only
            // the one pressed: pause and stop against one run are not
            // independent while the Tower has not said what the run is doing.
            // Bounded — the client clears it at the reply or at ten seconds,
            // whichever comes first. On each button rather than on the row,
            // so the spinner beside them is not drawn dimmed.
            HStack(spacing: 12) {
                if isPaused {
                    Button("Resume") { lab.resume() }
                        .disabled(lab.isCommandPending)
                } else {
                    Button("Pause") { lab.pause() }
                        .disabled(lab.isCommandPending)
                }
                Button("Stop") { lab.stop() }
                    .disabled(lab.isCommandPending)
                // The question that is out, while it is out. A pending start
                // is drawn on its catalog row instead, which is where the tap
                // happened.
                if let pending = lab.pendingCommand, pending.command != .start {
                    ProgressView()
                        .padding(.leading, 4)
                    Text("Asking the Tower…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.bordered)
            .font(.subheadline)
            // Stop keeps the figures; switching does not. Said here because it
            // is the only place the difference is actionable.
            Text("Stop keeps this run's figures. Starting another experiment replaces them.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// What the Tower says about whether anything is feeding it.
    ///
    /// Drawn under every state, because "nothing is arriving" is the answer to
    /// the question this screen is most often opened to ask, and it is a
    /// property of the Tower rather than of any run.
    @ViewBuilder
    private var sourceRows: some View {
        if let source = lab.source {
            Divider()
            CVFigureRow(
                caption: "Frames reaching the Tower",
                value: source.receivingFrames ? "yes" : "no"
            )
            // One Lab, shared by every connection. `clients_connected` is
            // reported so that "somebody else may be driving this" is at least
            // visible: last start wins, and there is no ownership model.
            if let clients = source.clientsConnected, clients > 1 {
                Text("\(clients) clients are connected to this Tower. The Lab has one slot, and the last start wins.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let diagnosis = lab.state.diagnosis(source: source) {
                Text(Self.explain(diagnosis))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// The Tower's own diagnosis table, in sentences.
    private static func explain(_ diagnosis: CVLabDiagnosis) -> String {
        switch diagnosis {
        case .nothingArriving:
            return "No frames have reached the Tower at all, so the stream is not running."
        case .arrivingButUndecodable(let count):
            return """
                \(count) frame\(count == 1 ? "" : "s") reached the Tower and could not be \
                decoded, so they never reached the Lab. That is a sender problem, not a \
                Lab one.
                """
        case .arrivingButRefused:
            return "Frames are arriving and the Lab is refusing them. Its state above says why."
        case .someFramesFailed(let count):
            return """
                The experiment raised on \(count) frame\(count == 1 ? "" : "s"). It is still \
                armed, and those frames produced nothing.
                """
        case .measuring:
            return "Frames are arriving and being measured."
        }
    }

    private static let armingCaveat = """
        Loading a model can take up to two minutes, and the Tower gives up after \
        that. There is no progress to report — the download does not offer any.
        """

    private static let failedRecovery = """
        Another start is the recovery. Do not stop first: the Lab refuses a stop \
        from this state, so choosing an experiment below is what gets it going.
        """

    // MARK: The catalog

    /// Every experiment the Tower declared, in the order it declared them.
    ///
    /// Read off the status document rather than off `.idle`, so the list does
    /// not vanish while something is running — `cv_lab_start` replaces whatever
    /// ran, so switching mid-run is a supported action and not an error.
    ///
    /// **iOS holds no list of its own.** A hardcoded subset here would be the
    /// app asserting that those experiments exist, and this is the file where
    /// that would be easiest to do and hardest to notice.
    @ViewBuilder
    private var experimentPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("Experiments")

            VStack(alignment: .leading, spacing: 12) {
                let experiments = lab.availableExperiments
                if experiments.isEmpty {
                    // "Not asked yet" and "asked, and it declares nothing" are
                    // different answers about different things, and only the
                    // second is a statement about the Tower.
                    Text(
                        isStillAsking
                            ? "Waiting for the Tower's catalog."
                            : "The Tower is connected and declares no experiments."
                    )
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                } else {
                    ForEach(experiments) { experiment in
                        experimentRow(experiment)
                    }
                    if !lab.canSendCommands {
                        Text(Self.cannotStartHere)
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        }
    }

    /// One catalog entry.
    ///
    /// A button where a start would be accepted, and **plain text where it
    /// would not** — never a greyed-out button, which still invites the press
    /// that produces the refusal. The Tower's own reason is shown instead,
    /// verbatim, because only the Tower knows which module is missing.
    ///
    /// The one disabled state is different in kind: while a command is out,
    /// every row is disabled for at most ten seconds, and the row that asked
    /// says so. That is not "this cannot be started" but "the last question
    /// has not been answered", and on a one-slot Lab where last start wins, a
    /// second tap in that window is a race, not a retry.
    @ViewBuilder
    private func experimentRow(_ experiment: CVExperiment) -> some View {
        if lab.isAwaitingStart(of: experiment) {
            // Asked, not loading. The Tower's `starting` is the next phase and
            // is drawn from the document, in `labPanel`, when it arrives.
            VStack(alignment: .leading, spacing: 6) {
                experimentLabel(experiment)
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Asking the Tower…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        } else if lab.canSendCommands && experiment.isStartable {
            Button {
                lab.run(experiment)
            } label: {
                experimentLabel(experiment)
            }
            .buttonStyle(.plain)
            .disabled(lab.isCommandPending)
            .accessibilityHint("Starts this experiment on the Tower, replacing whatever is running")
        } else {
            experimentLabel(experiment)
        }
    }

    private func experimentLabel(_ experiment: CVExperiment) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Text(experiment.name)
                    .font(.body.weight(.medium))
                    .foregroundStyle(experiment.isStartable ? .primary : .secondary)
                // Declared before it runs, which is the useful moment: this
                // says whether the numbers it is about to produce will be
                // measurements or model output.
                if experiment.provenance.isInference {
                    CVTag(text: "Estimates")
                }
                if experiment.requiresModel {
                    // Worth saying at the moment of choosing rather than after
                    // pressing: a model-backed start is the one that takes a
                    // minute and a half.
                    CVTag(text: "Loads a model")
                }
            }
            if let summary = experiment.summary {
                Text(summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let reason = experiment.unavailableReason, !experiment.isStartable {
                Text(reason)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(.rect)
    }

    #if DEBUG
    private static let cannotStartHere = """
        These are what the Tower can run. Nothing can be started from here until \
        the Tower is connected and offering the Lab.
        """
    #else
    private static let cannotStartHere = """
        These are what the Tower can run. This build has no camera, so an \
        experiment armed from here would measure nothing and no control to start \
        one is offered.
        """
    #endif
}

/// A refusal the Tower sent, with the one thing this app adds to it: what to do
/// about it.
///
/// The Tower's `message` is shown verbatim — it is the only text that knows
/// which module is missing or which run is current — and the sentence beneath
/// it comes from `disposition`, which is the whole reason the eight reasons are
/// classified rather than concatenated.
private struct CVRefusalBanner: View {
    let refusal: CVLabControlRefusal

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(refusal.message)
                .font(.subheadline)
                .fixedSize(horizontal: false, vertical: true)
            if let advice {
                Text(advice)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color(.tertiarySystemFill), in: .rect(cornerRadius: 12))
    }

    private var advice: String? {
        switch refusal.disposition {
        case .terminal:
            // Rendered as unsupported by the state machine as well; this is the
            // sentence beside the Tower's own words.
            return "This Tower cannot run experiments, so there is nothing to try again."
        case .transient:
            return "The Tower could not answer this one. Nothing changed, and the same request may work now."
        case .requestRefused:
            // Deliberately says "nothing changed" rather than suggesting a
            // retry: every one of these five is refused identically the second
            // time.
            return "Nothing changed on the Tower."
        case .unrecognised:
            return nil
        }
    }
}

/// The running experiment's answer for the most recent frame the Tower replied
/// to.
///
/// ## Why this observes `TowerClient` when the workspace above it does not
///
/// `TowerReachabilityReader` exists because the quiet cartridge workspaces
/// needed one rarely-changing fact from an object that republishes at the
/// Tower's reply rate, and observing it for that was a dead dependency with a
/// real main-actor cost.
///
/// This panel is the opposite case. The value it draws *is* the thing that
/// changes at reply rate, and a stale rendering of it is the defect rather than
/// the saving. So it observes — and it is a leaf holding nothing else, so the
/// invalidation stops here instead of taking the rest of the workspace with it.
///
/// ## What it is now allowed to say
///
/// The reading it draws has already been **gated on `cv_lab.run_id`** by
/// `TowerClient`: a result belonging to a run this client is not watching never
/// reaches `latestFrameResult` at all. So this panel can name the experiment
/// and the run beside the number without that being a claim it is not in a
/// position to make.
///
/// ## Runtime ownership
///
/// Observes. Owns nothing, constructs nothing, sends nothing. It cannot start a
/// session and is not handed anything that could.
private struct CVFrameReadingPanel: View {
    @ObservedObject var tower: TowerClient
    let isTowerReachable: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("Latest result")

            VStack(alignment: .leading, spacing: 10) {
                if let reading, reading.hasAnything {
                    figures(reading)
                } else if reading != nil {
                    // A reply that carried nothing is a real answer and a
                    // different one from no reply: the Tower processed the frame
                    // and the experiment concluded nothing about it. Drawing
                    // this as "waiting" would leave someone waiting for a result
                    // that has already arrived and was empty.
                    Text("The Tower answered the most recent frame and reported no result for it.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                } else if let refusal = tower.latestFrameRefusal {
                    // The `frame_error` case that did not exist until now. Every
                    // one of these used to be logged as an unknown message type
                    // and discarded, so this panel said "no frame has been
                    // answered yet" while the Tower was answering every single
                    // one with a reason.
                    refusalRows(refusal)
                } else {
                    // No spinner and no zeroes. Nothing is in flight, so a
                    // progress indicator would be the most convincing untrue
                    // thing this screen could draw.
                    Text(nothingYet)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        }
    }

    /// Every figure the Tower actually sent, and no other.
    @ViewBuilder
    private func figures(_ reading: CVFrameReading) -> some View {
        // Which experiment produced this. **The sentence this screen could not
        // say before**: the reply named the number and not the experiment, so a
        // figure sat here under a caption with nothing behind it.
        if let name = reading.experimentName {
            Text(name)
                .font(.subheadline.weight(.medium))
        }
        // The headline pair, drawn under the Tower's own name for it or not at
        // all. `CVFrameReading` will not hand over one half without the other,
        // so there is no way to caption this number with a word this app chose.
        if let headline = reading.headline {
            CVFigureRow(caption: headline.label, value: headline.displayValue)
        }
        // Optional on the wire. Its absence is "the experiment said nothing
        // about intensity", not "the frame was dark", so the row disappears
        // rather than showing a zero.
        if let intensity = reading.meanIntensity {
            CVFigureRow(caption: "Mean intensity", value: CVMetric.format(intensity))
        }
        // "ms" is the Tower's own declaration — the wire field is named
        // `processing_ms` — not this app's assumption. Same reasoning as
        // `CVRunSummaryView`, and the same wording, because it is the same
        // quantity.
        if let processing = reading.processingMs {
            CVFigureRow(caption: "Tower processing", value: "\(CVMetric.format(processing)) ms")
        }
        // The Tower's additive measurements, each under the name the Tower gave
        // it. `stage_ms` is deliberately not drawn here: it is a breakdown of
        // the processing time already shown, and putting both on one card
        // invites arithmetic the Tower never promised would add up.
        ForEach(reading.measurements) { measurement in
            CVFigureRow(caption: measurement.label, value: measurement.displayValue)
        }

        provenanceRows(reading)
    }

    /// Where this reading sits in its run, and where the numbers came from.
    ///
    /// ## The line that used to read "From frame 30."
    ///
    /// It was drawn from the wire's `seq`, which is the **phone's capture
    /// index**, and the sender forwards one frame in thirty by design — so
    /// consecutive replies read 30, 60, 90 and the gaps were not losses. Worse,
    /// that number is chosen by the sender and cannot order results at all.
    ///
    /// `result_seq` is the Tower's own dense counter within the run, from 1,
    /// and it is what says whether a reading is newer than the last. The
    /// capture index is still drawn, because it is the only thread back to a
    /// specific captured frame — under a caption that says what it is.
    @ViewBuilder
    private func provenanceRows(_ reading: CVFrameReading) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            if let resultSeq = reading.resultSeq {
                Text("Result \(resultSeq) of this run.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            if let captureIndex = reading.captureIndex {
                // NOT "one frame in thirty". That stride was deleted --
                // `FrameRateGate` says so in its own words -- and replaced by a
                // rate gate at `towerTargetFPS` (12). At a 24 fps source that
                // is roughly one frame in two, so the old sentence told a
                // reader 28 captures were skipped between consecutive numbers
                // when the real figure is about one.
                //
                // Worth naming the failure mode, because it is not the usual
                // one: this is a claim about THIS APP, not about the Tower, and
                // the code that refutes it is in this same target. The contract
                // document repeats the stale figure too -- iOS is the authority
                // on its own sender, so the contract is what is wrong there.
                //
                // The number is deliberately not restated here. A literal would
                // be a third copy to go stale; what a reader needs is that the
                // index skips and why, not the ratio.
                Text("From capture \(captureIndex). The sender does not forward every frame, so this number skips.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // Rule 16, discharged once for the card. `frame_result` now carries
            // provenance on every reply, so this is the Tower's own answer
            // rather than the blanket "the Tower did not say" this panel used
            // to show — and `.measured` owes no caveat at all, which is why
            // most replies now draw nothing here.
            if let caveat = reading.provenance.caveat {
                Text(caveat)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// Why the Tower answered a frame without measuring it.
    @ViewBuilder
    private func refusalRows(_ refusal: TowerFrameRefusal) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            // This app's sentence where it has one — `cv_lab_starting` is
            // arming rather than an error and has to be said that way — and the
            // Tower's own words otherwise, because only the Tower knows which
            // field was malformed or which module is missing.
            Text(refusal.kind.summary ?? refusal.message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if refusal.kind.summary != nil {
                Text(refusal.message)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let sequence = refusal.sequence {
                Text("Capture \(sequence).")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    /// `nil` when the Tower has not answered a frame at all.
    ///
    /// In a Release build there is nothing to read: the frame path is
    /// `#if DEBUG` (`ProjectManager`), no frames are sent, and `TowerClient`
    /// does not declare `latestFrameResult` outside DEBUG. This therefore
    /// degrades to the same empty card a Debug build shows before a session
    /// starts, with `nothingYet` saying the Release-specific reason — never a
    /// spinner, never a zero.
    ///
    /// A closure rather than `map(CVFrameReading.init)`: the target builds with
    /// main-actor default isolation, and an unapplied initializer reference
    /// does not inherit the caller's isolation the way a closure literal does.
    private var reading: CVFrameReading? {
        #if DEBUG
        return tower.latestFrameResult.map { CVFrameReading($0) }
        #else
        return nil
        #endif
    }

    /// Why there is nothing, in the words that fit the actual reason.
    private var nothingYet: String {
        #if DEBUG
        if !isTowerReachable {
            return "The Tower is not connected, so it has not answered a frame."
        }
        return """
            No frame has been answered yet. The Tower replies to every frame a \
            session sends, and the answer for the most recent one appears here.
            """
        #else
        return """
            This build does not send frames to the Tower, so there is nothing \
            for the experiment to answer and nothing to show here.
            """
        #endif
    }
}

/// One caption-and-figure line.
///
/// Shared by the two result panels in this file so a live reading and a
/// completed run read identically. Not an abstraction over cartridges — it is
/// eight points of layout that were about to exist twice in one file.
struct CVFigureRow: View {
    let caption: String
    let value: String

    var body: some View {
        HStack {
            Text(caption)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            Text(value)
                .font(.subheadline.weight(.medium))
                .monospacedDigit()
        }
    }
}

/// A small capsule label. Two points of layout, three call sites.
/// Internal rather than file-private since `CVLivePreviewPanel` gained a LIVE
/// badge: two capsules that looked almost the same would be worse than one
/// that is shared.
struct CVTag: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Color(.tertiarySystemFill), in: .capsule)
    }
}

/// Renders one run's results.
///
/// ## What is deliberately absent
///
/// **A better/worse verdict.** There is no code path to one: `CVMetric` carries
/// no baseline, no stated direction and no `comparison`, because the Tower's
/// metrics carry `baseline: null` and `higher_is_better: null` on every metric,
/// always, and a comparison against nothing is the *"declaring an approach
/// 'better' without a measurement"* that `docs/modules/EXPERIMENTAL-CV.md`
/// rules out. This view used to render exactly that when both fields arrived;
/// the fields never will, and the renderer is gone rather than dormant.
///
/// **A zero standing in for a null.** Every timing and every rate on this
/// contract is nullable, and the Tower is emphatic that *"`null` is 'nothing has
/// been measured'; it is never a zero you can render as one"* — a rate over a
/// zero-length window is undefined, not slow. Each row below is present only
/// when there is a number for it, and `notMeasuredNote` says so once for the
/// group rather than drawing a grid of dashes.
struct CVRunSummaryView: View {
    let run: CVExperimentRun
    let isFinal: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // PRIMARY: the one number the experiment says matters, and where
            // it sits in what this run has actually seen.
            headline
            // SECONDARY: is it healthy. Three figures, not eleven.
            health

            if !isFinal {
                Text("Running. These figures may still change.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // DIAGNOSTICS: everything else, collapsed.
            //
            // Collapsed, not deleted, and the distinction is the whole design.
            // Every figure below was on the screen before this change and every
            // one of them earns its place while somebody is debugging: the
            // stage timings, the resolved device beside the requested one, the
            // frame counters whose invariant is the diagnostic. What they did
            // not earn is being the FIRST thing, at the same weight as the
            // headline, on a screen whose purpose is to show what the algorithm
            // sees.
            DisclosureGroup {
                diagnostics
                    .padding(.top, 8)
            } label: {
                // `SectionLabel` rather than a bare string, so the one
                // collapsed section on this screen looks like every other
                // section heading in the app rather than like a control.
                SectionLabel("Diagnostics")
            }
        }
    }

    /// The experiment's own most important number, drawn large.
    ///
    /// The Tower puts it first in the list and marks it; this reads the mark
    /// rather than re-deciding, because "an experiment that cannot name its
    /// single most important number has not decided what it is measuring" is
    /// the Tower's discipline and not this app's to second-guess.
    @ViewBuilder
    private var headline: some View {
        if let metric = run.metrics.first(where: { $0.isHeadline }) {
            VStack(alignment: .leading, spacing: 4) {
                Text(metric.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(metric.displayValue ?? "—")
                        .font(.title2.weight(.semibold))
                        .monospacedDigit()
                    if metric.provenance.isInference {
                        CVTag(text: "Estimate")
                    }
                }
                // Where this frame sits in the range this run has seen. NOT a
                // verdict: see `CVMetric.rangeNote`, and see the Tower's own
                // refusal to publish a calibrated threshold for any of these.
                if let note = metric.rangeNote {
                    Text(note)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let reason = metric.unavailableReason {
                    Text(reason)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    /// Three figures that say whether the run is healthy, and no more.
    ///
    /// Chosen because each answers a different question a person actually
    /// asks: how hard is the Tower working, how much is getting through, and
    /// is anything arriving at all.
    @ViewBuilder
    private var health: some View {
        HStack(alignment: .top, spacing: 16) {
            if let processing = run.timings.processingMs {
                healthFigure("Processing", "\(CVMetric.format(processing)) ms")
            }
            if let processed = run.throughput.processedFps {
                healthFigure("Result rate", "\(CVMetric.format(processed)) fps")
            }
            if let frames = run.framesProcessed {
                healthFigure("Frames", "\(frames)")
            }
            Spacer(minLength: 0)
        }
        if !run.hasMeasuredAnything {
            Text(Self.notMeasuredNote)
                .font(.caption)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func healthFigure(_ caption: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(caption)
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Text(value)
                .font(.subheadline.weight(.medium))
                .monospacedDigit()
        }
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private var diagnostics: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(run.metrics) { metric in
                metricRow(metric)
            }
            // Reported rather than silently truncated: the Tower bounds a run at
            // sixteen metric rows and says how many it dropped.
            if run.metricsOmitted > 0 {
                Text("\(run.metricsOmitted) more metrics did not fit the Tower's per-run limit.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            // Empty is the only correct value and a Tower test enforces it for
            // every registered experiment. Drawn because this is what the wire
            // says if one ever reaches production anyway, and a silent drop
            // would make that invisible from here.
            if !run.unclassifiedMetrics.isEmpty {
                Text(
                    "The experiment emitted metrics without saying how they combine across frames: "
                        + run.unclassifiedMetrics.joined(separator: ", ")
                        + "."
                )
                .font(.caption)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
            }

            frameCounters
            timingRows
            annotationRows
            previewRows
            runtimeRows

            if run.containsInference {
                // Stated once for the run rather than repeated per row, so it
                // reads as a property of the results instead of as noise.
                Text("Values marked as estimates are model output, not measurements.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// What the live view cost, beside what the experiment cost and never
    /// mixed into it.
    ///
    /// This is the block that answers "did the picture slow the pipeline
    /// down?" after a physical test, with numbers instead of impressions.
    /// `renderMs` is worker-thread time and is deliberately NOT added to
    /// `timings.processing_ms`, which has to stay comparable against every
    /// figure recorded before previews existed.
    @ViewBuilder
    private var previewRows: some View {
        if let preview = run.preview {
            if let captured = preview.captured {
                row("Previews captured", "\(captured)")
            }
            if let skipped = preview.skippedByThrottle {
                row("Previews skipped by the throttle", "\(skipped)")
            }
            if let encoded = preview.encoded {
                row("Previews encoded", "\(encoded)")
            }
            if let unchanged = preview.notModified, unchanged > 0 {
                row("Fetches answered unchanged", "\(unchanged)")
            }
            if let failures = preview.encodeFailures, failures > 0 {
                row("Previews that failed to render", "\(failures)")
            }
            if let render = preview.renderMs {
                row("Preview render, mean", "\(CVMetric.format(render)) ms")
            }
            if let worst = preview.renderMsMax {
                row("Preview render, worst", "\(CVMetric.format(worst)) ms")
            }
            if let bytes = preview.payloadBytes {
                row("Preview size, mean", "\(bytes / 1024) KB")
            }
            if let note = preview.deliveryNote {
                Text(note)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func metricRow(_ metric: CVMetric) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            if let value = metric.displayValue {
                row(metric.label, value)
            } else {
                // A metric the Tower reported with no value is still a row: it
                // said this metric exists and has no meaningful aggregate,
                // which is a different fact from never having mentioned it.
                Text(metric.label)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                if metric.isHeadline {
                    CVTag(text: "Headline")
                }
                if metric.provenance.isInference {
                    CVTag(text: "Estimate")
                }
                // How the number was arrived at across frames — a mean, a sum,
                // or a single observed value. Shown because summing a fraction
                // and averaging a count are different mistakes, and a reader
                // who cannot tell which this is cannot catch either.
                if let aggregation = metric.aggregation, let frames = metric.frames, frames > 0 {
                    Text("\(aggregation) over \(frames) frame\(frames == 1 ? "" : "s")")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            // Where the newest frame sits in what this run has seen. A
            // placement inside one run's own range, and never a verdict --
            // "Sharpness: Good" would be this app inventing the calibration
            // both sides refuse to invent. See `CVMetric.rangeNote`.
            if let note = metric.rangeNote {
                Text(note)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let reason = metric.unavailableReason {
                Text(reason)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // Not gated on `isInference`. `caveat` is already nil for
            // `.measured`, and gating additionally on inference hid the
            // `.unknown` caveat entirely — drawing a value whose provenance the
            // Tower never stated as though it were a measurement, which is Rule
            // 16 exactly backwards.
            if let caveat = metric.provenance.caveat {
                Text(caveat)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
    }

    /// The four counters, which the Tower keeps in an invariant:
    /// `offered == processed + refused + failed`, holding at every read.
    ///
    /// `framesRefused` is drawn even at zero, and that is the point of it: on a
    /// paused run it is the one figure still moving, and it is how you see
    /// whether the phone is still sending while the metrics stand still.
    @ViewBuilder
    private var frameCounters: some View {
        if let processed = run.framesProcessed {
            row("Frames measured", "\(processed)")
        }
        if let refused = run.framesRefused, refused > 0 {
            row("Frames refused", "\(refused)")
        }
        if let failed = run.framesFailed, failed > 0 {
            row("Frames the experiment failed on", "\(failed)")
        }
    }

    @ViewBuilder
    private var timingRows: some View {
        if let processing = run.timings.processingMs {
            // The Tower measuring its own work — the only timing here, because
            // it is the only one with an unambiguous meaning (see `CVTimings`).
            // And "ms" is not an invented unit, unlike the metres
            // `ReportedFigure` exists to prevent: the field is named
            // `processing_ms` on the wire, so milliseconds is the Tower's own
            // declaration rather than this app's assumption.
            row("Tower processing, mean", "\(CVMetric.format(processing)) ms")
        }
        if let worst = run.timings.processingMsMax {
            row("Tower processing, worst frame", "\(CVMetric.format(worst)) ms")
        }
        if let processed = run.throughput.processedFps {
            row("Measured", "\(CVMetric.format(processed)) fps")
        }
        if let capacity = run.throughput.capacityFps {
            // Read beside the rate above, this says whether the Lab or the link
            // is the limit. The sender forwards one frame in thirty, so the
            // measured rate is normally bounded by what arrives.
            row("The Lab could manage", "\(CVMetric.format(capacity)) fps")
        }
    }

    private static let notMeasuredNote = """
        No frame has been measured on this run, so the Tower reports no timings \
        and no rate. That is "nothing measured yet", not zero.
        """

    /// The annotation count, and the Tower's own reason when there is none.
    ///
    /// `0` is a real result meaning "found nothing" and must not merge with
    /// "did not say", which is why the reason is drawn rather than an absence
    /// being left to speak for itself.
    @ViewBuilder
    private var annotationRows: some View {
        if let count = run.annotation.count {
            row("Annotations", "\(count)")
        } else if let reason = run.annotation.countUnavailableReason {
            Text(reason)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
        // The Tower's own sentence for why there is no live view, when there
        // is none -- previews switched off, or an experiment with nothing to
        // draw. Drawn verbatim rather than paraphrased: it is the Tower
        // explaining a decision in its own words, and a paraphrase would rot.
        // When there IS one, this is nil and `CVLivePreviewPanel` has the
        // picture.
        if let reason = run.annotation.artifactUnavailableReason {
            Text(reason)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// What the experiment says it actually loaded.
    ///
    /// Sorted by key, because a dictionary has no order and a list that
    /// reshuffles itself on every update is unreadable. The keys are the
    /// experiment's own and nothing here switches on them — this exists because
    /// `device: auto` is a *request* and a CPU figure carrying a GPU label is a
    /// real failure it closes.
    @ViewBuilder
    private var runtimeRows: some View {
        if !run.runtime.isEmpty {
            ForEach(run.runtime.sorted(by: { $0.key < $1.key }), id: \.key) { entry in
                row(entry.key, entry.value)
            }
        }
    }

    private func row(_ caption: String, _ value: String) -> some View {
        CVFigureRow(caption: caption, value: value)
    }
}
