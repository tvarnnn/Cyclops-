//
//  TowerClient.swift
//  Glasses
//
//  Created by Tristan Varner on 8/18/26.
//

import Combine
import Foundation

#if DEBUG
import UIKit
#endif

/// One `frame_result` message, as the Tower actually sends it today.
///
/// Not a guess at a future protocol — every field here is one the Tower
/// builds in `tower/tower/routes/ws.py:148-165`, and every one is optional (or
/// empty-by-default) because the decoder must not fabricate a value the
/// message omitted.
///
/// ## The earlier version of this comment was wrong, and it cost a cartridge
///
/// It read: *"The Tower's whole per-frame vocabulary is `seq`,
/// `mean_intensity` and `processing_ms` — it runs one fixed handler and has no
/// module runtime."* Both halves were false. `ws.py` sends **five keys
/// unconditionally** — `seq`, `processing_ms`, `result_value`, `result_label`
/// and `stage_ms` — and adds `mean_intensity` and `metrics` when the
/// experiment produced them. The Tower does have a module runtime:
/// `tower/tower/main.py` builds a `ModuleContainer` around a live
/// `ExperimentalCVModule`, and a running Tower reports `module_state: active`
/// on `/health`.
///
/// So `result_value` and `result_label` — the experiment's own answer, the
/// only thing the Tower says about what it *concluded* rather than how long it
/// took — were arriving on every single frame and being dropped on the floor,
/// while the Experimental CV Lab workspace told the wearer the Tower "cannot
/// run experiments yet".
///
/// The Tower's registry used to say the same thing from its side:
/// `experimental_cv` was `not_offered` on the result channel because *"results
/// already reach the client on `frame_result`"*. **That has changed.** A Tower
/// speaking `experimental_cv.status/2026-08-27` offers the cartridge, and every
/// `frame_result` now carries a `cv_lab` block naming the run, the experiment
/// and the provenance that produced the numbers beside it — see `cvLab` below.
/// Everything that was on this message before is still there, unchanged, in the
/// same place.
struct TowerFrameResult: Equatable, Sendable {
    /// The frame this result answers, matching the `seq` the app sent.
    let sequence: Int?
    /// Mean pixel intensity, 0...1. Present only when the experiment reported
    /// one — `nil` means the experiment said nothing, never that the frame was
    /// dark.
    let meanIntensity: Double?
    /// How long the Tower spent on the frame.
    let processingMs: Double?
    /// The experiment's headline number. Its meaning is the experiment's, not
    /// this app's: it is paired with `resultLabel` and must never be rendered
    /// without it, because a bare number implies a unit nobody promised.
    let resultValue: Double?
    /// The experiment's own name for what it measured. The one piece of
    /// provenance on this channel.
    let resultLabel: String?
    /// Per-stage timings inside the Tower's own processing, name -> ms.
    /// Empty when the Tower sent an empty object; there is no distinction on
    /// the wire between empty and absent, and none is invented here.
    let stageMs: [String: Double]
    /// Additive measurements, name -> number, omitted entirely when empty.
    /// Deliberately numbers only: `ws.py` calls this "a MEASUREMENT channel,
    /// not the structured result channel", and the structured one is blocked
    /// on module-contract work this app must not pre-empt.
    let metrics: [String: Double]
    /// Who produced this number, and for which run.
    ///
    /// `nil` from a Tower that runs no CV Lab — the block is additive and
    /// omitted entirely rather than nulled. Every field of it is read in
    /// `TowerFrameResultProvenance`, and one of them, `runID`, is read by this
    /// client itself: see `watchedCVLabRunID`.
    let cvLab: TowerFrameResultProvenance?
}

/// The `cv_lab` block on a `frame_result` —
/// `experimental_cv.frame_result/2026-08-27`.
///
/// ## What this closes
///
/// Before it, the Tower's per-frame reply named the *number* and not the
/// experiment: `result_value` with `result_label` beside it, and no way to tell
/// which of eight experiments produced them, on which run, measured or
/// inferred. A result from a previous experiment could be read as a result from
/// the current one, and the Experimental CV Lab workspace said so on screen
/// because it was true.
///
/// ## The one rule a client must not get wrong
///
/// > **Discard any `frame_result` whose `cv_lab.run_id` is not the run you are
/// > watching.**
///
/// The Tower makes this structural rather than checked: a new experiment is a
/// new run, and the old experiment is released **before** the new run id is
/// published, so no result computed by one experiment can carry another's name.
/// The client-side rule exists for the case the Tower cannot cover — a
/// reconnect to a **restarted** Tower, which starts counting runs again from 1.
/// `runID` is `"<tower_instance_id>-<n>"` and `towerInstanceID` is part of it
/// for exactly that reason, which is why comparing `runID` alone is sufficient
/// and no separate instance check is needed.
///
/// The gate itself is in `handleInboundMessage`, not here: a value type cannot
/// know which run is being watched.
struct TowerFrameResultProvenance: Equatable, Sendable {
    /// Opaque, compared for equality only.
    let contract: String?
    /// Which Tower. Already inside `runID`; carried because a client that
    /// reports a run id to a person should be able to say which machine it
    /// belongs to.
    let towerInstanceID: String?
    /// `"<tower_instance_id>-<n>"`. The run this result belongs to.
    let runID: String?
    /// **Dense within the run, from 1.**
    ///
    /// This is the field that orders results. The message's own `seq` is the
    /// phone's capture index and **skips by design** — the sender forwards one
    /// frame in thirty — so it cannot order anything. That the two are
    /// different numbers with different meanings on the same message is the
    /// single easiest thing to get wrong here.
    let resultSeq: Int?
    let experimentID: String?
    /// The Tower's own name for the experiment, for display.
    let experimentName: String?
    /// `measured` or `inferred`, as a string. Interpreted by the cartridge that
    /// owns the contract — this file does not know what an experiment is.
    let provenance: String?
    /// `opencv` or `torch`.
    let backend: String?
    /// What the experiment actually ran on, or `nil` when it holds nothing.
    /// Distinct from `deviceRequested`, which is a *request*: `auto` is the
    /// question, and this is the answer.
    let device: String?
    let deviceRequested: String?
    /// The experiment's own name for the number, repeated here so the block is
    /// self-contained.
    let resultLabel: String?
    let processingMs: Double?
    /// When the **Tower** received the frame. `timeBasis` says so on every
    /// block, because there is no capture timestamp anywhere on this wire —
    /// `tower/frames.py` carries no time field.
    let towerReceivedAt: Double?
    let timeBasis: String?

    init?(json: [String: Any]) {
        // Nothing is required. A block that arrived with only a `run_id` still
        // gates correctly, and a block with everything but one field still
        // displays the rest — the alternative, refusing the whole block on a
        // missing optional, would silently turn a gated result into an ungated
        // one, which is the failure this type exists to prevent.
        self.contract = json["contract"] as? String
        self.towerInstanceID = json["tower_instance_id"] as? String
        self.runID = json["run_id"] as? String
        self.resultSeq = json["result_seq"] as? Int
        self.experimentID = json["experiment_id"] as? String
        self.experimentName = json["experiment_name"] as? String
        self.provenance = json["provenance"] as? String
        self.backend = json["backend"] as? String
        self.device = json["device"] as? String
        self.deviceRequested = json["device_requested"] as? String
        self.resultLabel = json["result_label"] as? String
        self.processingMs = json["processing_ms"] as? Double
        self.towerReceivedAt = json["tower_received_at"] as? Double
        self.timeBasis = json["time_basis"] as? String
    }
}

// MARK: - The frame path's refusals

/// One `frame_error`: the Tower answering a frame it did not process.
///
/// ## Why this had to be added
///
/// `handleInboundMessage` had **no case for it**. Every `frame_error` the Tower
/// has ever sent fell through to `default:`, was logged as an unknown message
/// type, and was discarded — so "the Lab is paused", "that frame was
/// undecodable" and "the socket is wedged" were indistinguishable from this
/// side, and the answer to *"I pressed Start and nothing happened"* was on the
/// wire and being thrown away.
///
/// ## Refusal, not failure
///
/// Six of the nine reasons are the CV Lab declining a frame on purpose. The
/// Tower is explicit that these are counted under `frames_rejected` and **not**
/// under `frame_processing_errors` — *"a Lab paused for five minutes has not
/// failed hundreds of times"* — and this client owes the same discipline:
/// nothing here touches `status`, the send window, the stream bracket or
/// `SenderMetrics`' error counters. It decodes, publishes, and returns.
///
/// The nine reasons are interpreted by the cartridge that owns the CV Lab
/// contract, not here. This file decodes three fields and knows what none of
/// them mean.
struct TowerFrameRefusal: Equatable, Sendable {
    /// The frame this answers — the phone's **capture index**, and `nil` when
    /// the frame failed validation before its `seq` was even readable. The
    /// Tower reports null rather than inventing one.
    let sequence: Int?
    /// The Tower's code: `cv_lab_idle`, `cv_lab_starting`, `cv_lab_paused`,
    /// `cv_lab_stopped`, `cv_lab_failed`, `cv_lab_unavailable`,
    /// `invalid_frame`, `frame_skipped` or `module_unavailable`.
    ///
    /// Kept as a **string**, like every other refusal vocabulary this client
    /// carries: a code this build does not recognise must still reach a person
    /// rather than being collapsed into "unknown".
    let reason: String
    /// Prose for a person. The Tower's message says what to send next.
    let message: String

    init?(json: [String: Any]) {
        guard let reason = json["reason"] as? String else { return nil }
        self.sequence = json["seq"] as? Int
        self.reason = reason
        self.message = json["message"] as? String ?? reason
    }
}

// MARK: - The CV Lab control plane

/// A `cv_lab_status` message: the Lab's whole state, in one snapshot.
///
/// ## `acceptedCommand` is the field that makes this two messages
///
/// The Tower uses one name for one document, whichever direction it is
/// travelling and whatever prompted it — a read, an accepted command, or a push
/// on the result channel. `acceptedCommand` is present **only** on a reply to a
/// command, and it is therefore the only way to tell an answer from a pushed
/// state. A client that treated every arriving document as an answer would
/// report "started" every two seconds for as long as the Lab was running.
///
/// Verified against a live Tower: a plain `cv_lab_status` read comes back with
/// no `accepted_command` at all, while `cv_lab_start`/`pause`/`resume`/`stop`
/// each echo their own name.
///
/// The `status` document itself is passed through **undecoded**, for the same
/// reason `CartridgeResultEnvelope.payload` is: this file owns the transport
/// and the cartridge owns the contract. Nothing here knows what an experiment
/// is. The one exception is `runID`, which the frame path needs — see
/// `watchedCVLabRunID`.
struct CVLabControlReply {
    /// The status document's contract identifier. Opaque.
    let contract: String?
    /// The **control vocabulary's** identifier, which versions separately from
    /// the document's precisely so a client can implement the read-only half
    /// and never send a command.
    let controlContract: String?
    /// Echoed from the request, when one was sent and was at most 64
    /// characters. **A longer one is dropped, not refused** — the command still
    /// applies and the reply simply carries none — which is why this client
    /// bounds its own ids rather than relying on noticing the absence.
    let requestID: String?
    /// The command this answers, or `nil` when the document was pushed or read.
    let acceptedCommand: String?
    /// The whole document, undecoded.
    let status: [String: Any]

    init?(json: [String: Any]) {
        guard let status = json["status"] as? [String: Any] else { return nil }
        self.contract = json["contract"] as? String
        self.controlContract = json["control_contract"] as? String
        self.requestID = json["request_id"] as? String
        self.acceptedCommand = json["accepted_command"] as? String
        self.status = status
    }

    /// `lifecycle.run_id`, the one field this file reads out of the document.
    ///
    /// Read here rather than in the cartridge because the **frame path** needs
    /// it, and the frame path is this file's. It is a single traversal of two
    /// keys; it does not make this client cartridge-aware in any other respect.
    var runID: String? {
        (status["lifecycle"] as? [String: Any])?["run_id"] as? String
    }

    /// `lifecycle.state`, read only so a log line says something useful.
    var lifecycleState: String? {
        (status["lifecycle"] as? [String: Any])?["state"] as? String
    }
}

/// A `cv_lab_error`: a refusal.
///
/// > **Every one of the eight reasons means the request did not take effect.
/// > There is no partial application.**
///
/// Which is why this carries the unchanged `status`: a refused client never has
/// to guess what state it is now in, and nothing in this app has to model a
/// half-applied command. Even the `lab_unavailable` refusal from a Tower with
/// no Lab at all carries a document — a hollow one, with the real contract
/// identifiers in it — so a decoder is entitled to require the field. It is
/// still optional here, because a refusal that arrived without one is worth
/// showing rather than dropping.
struct CVLabControlRefusal {
    /// `malformed_request`, `unknown_experiment`, `experiment_unavailable`,
    /// `lab_busy`, `invalid_state`, `stale_run`, `lab_unavailable` or
    /// `internal_error`. A string, so a ninth reaches a person as itself.
    let reason: String
    /// Prose for a person, from the Tower. The only text that knows *which*
    /// module was missing or *which* run is current.
    let message: String
    /// Which command was refused. Lets a refusal be attributed to a button even
    /// when no `request_id` was sent.
    let command: String?
    let requestID: String?
    let controlContract: String?
    /// `unknown_experiment` only: the ids this Tower does have.
    let available: [String]
    /// `experiment_unavailable` only.
    let experimentID: String?
    /// `stale_run` only: the run the Lab is actually on now. A stale `run_id`
    /// is refused rather than applied to whichever run is current, so a Stop
    /// drawn against a run that has already been replaced cannot end the one
    /// that replaced it.
    let currentRunID: String?
    /// The document, unchanged. Undecoded, for the reason `CVLabControlReply`
    /// gives.
    let status: [String: Any]?

    init?(json: [String: Any]) {
        guard let reason = json["reason"] as? String else { return nil }
        self.reason = reason
        self.message = json["message"] as? String ?? reason
        self.command = json["command"] as? String
        self.requestID = json["request_id"] as? String
        self.controlContract = json["control_contract"] as? String
        self.available = json["available"] as? [String] ?? []
        self.experimentID = json["experiment_id"] as? String
        self.currentRunID = json["current_run_id"] as? String
        self.status = json["status"] as? [String: Any]
    }
}

/// Everything the CV Lab's control plane delivers, as one stream.
///
/// One subject rather than three published properties, for the reason
/// `CartridgeResultEvent` gives: ordering between them is load-bearing. A
/// `cv_lab_error` and the pushed status that follows it must not be observed
/// out of order, or a client would file the refusal against the wrong state.
///
/// `frameRefused` rides this stream too. A `frame_error` is a frame-path
/// message and six of its nine reasons are the Lab's own — *"the Lab is
/// paused"* is the answer to *"why is there no result"*, which is a CV Lab
/// question wherever it arrives from.
enum CVLabEvent {
    /// A status document: an answer to a command when `acceptedCommand` is
    /// non-nil, a read or a push otherwise.
    case status(CVLabControlReply)
    /// A refused command. The request did not take effect.
    case refused(CVLabControlRefusal)
    /// A frame the Tower did not process.
    case frameRefused(TowerFrameRefusal)
}

/// Connection status to the Tower (the project's base-station/hub service).
enum TowerStatus: Equatable {
    case offline
    case connecting
    case online
    case failed(String)
}

// MARK: - The Tower's own state, as reported by GET /health

/// Something the Tower's health report may or may not have mentioned.
///
/// ## Why three cases where a `nil` would fit
///
/// `GET /health` makes genuinely different statements about a subsystem, and
/// the difference is the whole reason this type exists:
///
/// - The key is **missing**. This Tower's health route says nothing about the
///   subsystem at all — an older build, or one whose shape has moved on. We
///   were not told.
/// - The key is present and **null**. The Tower is telling us something
///   specific: nothing is registered. `tower/tower/routes/health.py` is
///   explicit that `capture: null` means no recorder exists, "which is
///   different from a registered recorder that is idle", and that collapsing
///   the two would make "we are definitely not recording" indistinguishable
///   from "we are armed and one `stream_start` away".
/// - The key is present and carries an **object**, which may still leave any
///   individual field unsaid.
///
/// Folding the first two into one `nil` would answer a privacy question with a
/// shrug dressed up as a fact. `TOWER-TO-IOS` reconciliation §1.8 is a worked
/// example of the same failure in the other direction: reading the obvious
/// field there would have reported a confident, wrong **0**.
nonisolated enum TowerReported<Value: Equatable & Sendable>: Equatable, Sendable {
    /// The report did not mention this at all.
    case unreported
    /// The report mentioned it and said there is nothing there.
    case absent
    /// The report carried a value, whose own fields may still be unsaid.
    case present(Value)
    /// The report carried something this app could not read as this shape.
    /// Not the same as silence: the Tower spoke and we failed to understand.
    case unreadable

    var value: Value? {
        if case .present(let value) = self { return value }
        return nil
    }
}

/// The Tower's dataset recorder, as `/health` reports it.
///
/// ## Why every field is optional
///
/// Because the Tower's own error path sends `{"armed": true, "error":
/// "unavailable"}` — armed, and nothing else known — and because an omitted
/// `recording` must never be read as "not recording" on the one screen a
/// person would consult to find out. This is the project's `nil ≠ 0` rule at
/// the point where it costs the most to get wrong: while the recorder is
/// armed, the Tower fsyncs every frame it receives to disk **unredacted**
/// (`tower/tower/capture.py` declares `retains_raw_imagery: true`,
/// `redaction: "none"`, and tags the manifest `raw-imagery`,
/// `first-person`).
///
/// ## There is deliberately no way to arm it from here
///
/// `TOWER_CAPTURE_ROOT` arms the recorder at Tower start-up and
/// `stream_start`/`stream_stop` bound each recording; the reconciliation
/// document lists arming from iOS as **BLOCKED**, with no route to call. A
/// control on this side would be a fabricated capability, so this type is a
/// reading and nothing else.
nonisolated struct TowerCaptureState: Equatable, Sendable {
    /// A recorder is registered and will write whenever a stream is running.
    let armed: Bool?
    /// Frames are being written **right now**.
    let recording: Bool?
    /// The recording the counts below belong to.
    ///
    /// `TowerReported` rather than `String?`, because **three** different
    /// things arrive in this one field and only two of them are a `nil`:
    ///
    /// - `.present(id)` — a recording is open and this is its id.
    /// - `.absent` — the key arrived carrying `null`. That is a *positive*
    ///   statement: `tower/tower/routes/health.py` emits `capture_id` on every
    ///   success, and sends `null` in exactly one situation — a recorder is
    ///   registered and has not opened a recording yet.
    /// - `.unreported` — the key was not there at all, which happens only on
    ///   the Tower's `.error` branch (`{"armed": true, "error": "unavailable"}`)
    ///   and means the Tower could not read its own recorder.
    ///
    /// Flattening the middle two into one Optional is the same
    /// `unreported`-vs-`absent` conflation this enum exists to prevent, one
    /// level down at the field: it renders "The Tower did not say" over an
    /// answer the Tower gave clearly.
    ///
    /// `framesWritten` and `bytesWritten` below stay plain Optionals on
    /// purpose. They are `0` — a real, sent zero — in that same no-recording
    /// case, and absent only on the `.error` branch, so for them one `nil`
    /// carries exactly one meaning.
    let captureID: TowerReported<String>
    /// Frames written to disk in the latest recording. `nil` is "the Tower did
    /// not say"; `0` is the Tower saying zero, and the two must not be drawn
    /// the same way.
    let framesWritten: Int?
    let bytesWritten: Int?
    /// The Tower could not read its own recorder's state. It still knows the
    /// recorder is registered, which is why `armed` can be true beside this.
    let error: String?
}

/// Whether anything on the Tower is turning captures into anything.
///
/// `enabled: false` means nothing is configured to follow a capture;
/// `enabled: true` with no workers is correct between walks and wrong during
/// one. Carried here because it answers "why isn't World Builder changing?"
/// from the phone, which was previously only answerable by noticing that no
/// world directory had appeared on a machine nobody was looking at.
nonisolated struct TowerCaptureWorkers: Equatable, Sendable {
    let enabled: Bool?
    /// How many workers are running. `nil` when the Tower sent no `workers`
    /// list at all — an empty list it *did* send is a real `0`.
    let workerCount: Int?
    let error: String?
}

/// One `GET /health` answer.
///
/// Every field is optional and nothing is defaulted, because this type's only
/// job is to say what the Tower said. A build that omits a field has not
/// claimed anything about it.
nonisolated struct TowerHealth: Equatable, Sendable {
    let status: String?
    let service: String?
    let version: String?
    /// The module runtime's state, e.g. `active`.
    let moduleState: String?
    /// Which module is loaded, e.g. `experimental-cv`.
    let moduleID: String?
    let capture: TowerReported<TowerCaptureState>
    let captureWorkers: TowerReported<TowerCaptureWorkers>
}

/// Why a health read did not produce an answer.
///
/// The same split `ObjectMemoryFetchError` keeps, minus the cases that have no
/// meaning here: `/health` has no contract field to disagree about and no 404
/// that means anything but "this is not a Tower". Two cases, and they are not
/// interchangeable — an answer that arrived and could not be read is a
/// disagreement about the answer, not a failure to get one, and only the
/// second is evidence about the network.
nonisolated enum TowerHealthFetchError: Error, Equatable {
    /// The answer arrived and could not be read as a health report.
    case undecodable
    /// The request did not complete, or the Tower refused it.
    case transport(String)
}

/// Turns a `/health` body into what the Tower said, and nothing more.
///
/// Split out from the HTTP client so the shapes that actually matter — a
/// missing `capture`, an explicit null, an error object, a count the Tower
/// omitted — are testable against real payloads without standing up a server.
nonisolated enum TowerHealthDecoder {

    /// - Throws: `TowerHealthFetchError.undecodable` when the bytes are not a
    ///   JSON object. A *well-formed* object that mentions nothing is not an
    ///   error: it decodes to a health report in which everything is unsaid,
    ///   which is the truth about it.
    static func health(from data: Data) throws -> TowerHealth {
        // Parsed in its own `do` because `jsonObject(with:)` throws on a
        // malformed body rather than returning something the cast rejects,
        // and that throw must not be relabelled as a transport failure — the
        // Tower answered.
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw TowerHealthFetchError.undecodable
        }
        guard let json = parsed as? [String: Any] else {
            throw TowerHealthFetchError.undecodable
        }
        return health(from: json)
    }

    static func health(from json: [String: Any]) -> TowerHealth {
        TowerHealth(
            status: json["status"] as? String,
            service: json["service"] as? String,
            version: json["version"] as? String,
            moduleState: json["module_state"] as? String,
            moduleID: json["module_id"] as? String,
            capture: reported(json, key: "capture", as: captureState),
            captureWorkers: reported(json, key: "capture_workers", as: captureWorkers)
        )
    }

    /// The three-way read that `TowerReported` exists for, plus the fourth
    /// case for a value that is neither an object nor a null.
    private static func reported<Value>(
        _ json: [String: Any], key: String, as decode: ([String: Any]) -> Value
    ) -> TowerReported<Value> {
        guard let raw = json[key] else { return .unreported }
        if raw is NSNull { return .absent }
        guard let object = raw as? [String: Any] else { return .unreadable }
        return .present(decode(object))
    }

    private static func captureState(from json: [String: Any]) -> TowerCaptureState {
        TowerCaptureState(
            armed: json["armed"] as? Bool,
            recording: json["recording"] as? Bool,
            // Null on the wire until the recorder opens its first recording,
            // and that null is an answer rather than a silence — see
            // `TowerCaptureState.captureID`.
            captureID: reportedString(json, key: "capture_id"),
            framesWritten: json["frames_written"] as? Int,
            bytesWritten: json["bytes_written"] as? Int,
            error: json["error"] as? String
        )
    }

    /// The same three-way read for a field whose value is a bare string rather
    /// than an object.
    ///
    /// Separate from `reported(_:key:as:)` because that one's `decode` closure
    /// takes a dictionary — a scalar has no fields to project — and because the
    /// `.unreadable` case here means something narrower: the key arrived
    /// carrying neither a string nor a null, which is the Tower speaking in a
    /// shape this app cannot read.
    private static func reportedString(
        _ json: [String: Any], key: String
    ) -> TowerReported<String> {
        guard let raw = json[key] else { return .unreported }
        if raw is NSNull { return .absent }
        guard let string = raw as? String else { return .unreadable }
        return .present(string)
    }

    private static func captureWorkers(from json: [String: Any]) -> TowerCaptureWorkers {
        TowerCaptureWorkers(
            enabled: json["enabled"] as? Bool,
            // `nil` when no list was sent; `0` when an empty one was. The
            // Tower distinguishes them and so does this.
            workerCount: (json["workers"] as? [Any])?.count,
            error: json["error"] as? String
        )
    }
}

/// The one `GET`, and nothing else.
///
/// ## Read-only, and structurally so
///
/// There is no method here that changes anything on the Tower, and there is
/// nothing on the Tower's side to call: the dataset recorder is armed by
/// `TOWER_CAPTURE_ROOT` at start-up. This type can report the recorder's state
/// and can never alter it.
///
/// ## Why this mirrors `ObjectMemoryHTTPClient` rather than inventing anything
///
/// Same shape, same `JSONSerialization` decoding, same injectable
/// `URLSession`, same timeout policy. A second, differently-opinionated
/// networking layer in one app is two places for that policy to be wrong.
///
/// ## Bounded, and uncached
///
/// Rule 15. The request carries an explicit timeout and
/// `reloadIgnoringLocalCacheData` — a health answer served out of a URL cache
/// would report a recorder as idle after it had started writing, which is the
/// single worst staleness this particular reading can carry.
nonisolated struct TowerHealthHTTPClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession = .shared
    /// Long enough for a Tailscale round trip, short enough that a dead Tower
    /// becomes a visible state rather than a spinner nobody can end. Matches
    /// `ObjectMemoryHTTPClient.timeout` deliberately.
    var timeout: TimeInterval = 10

    func health() async throws -> TowerHealth {
        let request = URLRequest(
            url: baseURL.appendingPathComponent("health"),
            cachePolicy: .reloadIgnoringLocalCacheData,
            timeoutInterval: timeout
        )

        do {
            let (data, response) = try await session.data(for: request)
            if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                throw TowerHealthFetchError.transport("The Tower answered \(http.statusCode).")
            }
            return try TowerHealthDecoder.health(from: data)
        } catch let error as TowerHealthFetchError {
            throw error
        } catch {
            throw TowerHealthFetchError.transport(error.localizedDescription)
        }
    }
}

/// What a screen showing the Tower's state should be showing.
///
/// Four cases because there are four different things to say, and three of
/// them are routinely drawn as the fourth: "nobody has asked", "we are
/// asking", "here is what it said" and "we asked and could not find out" are
/// not the same, and a screen that renders the first and last identically has
/// turned a failure into silence.
nonisolated enum TowerHealthState: Equatable, Sendable {
    /// Nothing has been asked. **Not** "the Tower is fine", and not "the
    /// recorder is off".
    case notFetched
    case fetching
    /// An answer, stamped with when it arrived — because this reading goes
    /// stale the moment a capture starts, and an unstamped one invites being
    /// read as current.
    case fetched(TowerHealth, at: Date)
    case failed(TowerHealthFetchError, at: Date)
}

/// WebSocket client for the Tower connection. Validates connectivity with an
/// initial ping/pong handshake, then keeps a continuous receive loop running
/// for as long as the connection is online so the client can observe
/// `frame_result` messages and — just as importantly — actually notice when
/// the Tower or the OS closes the socket out from under it. A
/// `URLSessionWebSocketDelegate` close callback is a second, independent
/// signal for the same event — though note that callback only fires for a real
/// close *frame*, so the receive loop is the one that catches a dropped link.
///
/// A dropped or wedged connection is re-established automatically when
/// `autoReconnect` is set, on a bounded backoff. This does not implement the
/// future frame-streaming protocol beyond what's described here.
@MainActor
final class TowerClient: NSObject, ObservableObject {
    @Published private(set) var status: TowerStatus = .offline

    #if DEBUG
    /// How many `frame_result` messages the receive loop has processed — the
    /// only end-to-end proof that the Tower received a frame and replied.
    /// `@Published` so the dashboard can show it live; it is otherwise
    /// unchanged, and nothing reads it to make a decision. It now invalidates
    /// the view tree at the Tower's reply rate (target ~12 Hz), which is the
    /// same order as `GlassesConnection.frameCount` has always done at 24 Hz.
    @Published private(set) var frameResultCount = 0

    /// The most recent `frame_result` the Tower returned.
    ///
    /// The Tower's reply already carries a `mean_intensity`, and until now it
    /// was formatted into a decimated log line and thrown away. It is the only
    /// thing the Tower currently says *about a frame's content*, which makes it
    /// the one piece of real evidence the app can show that the round trip is
    /// doing something rather than merely completing. Surfacing it is what lets
    /// a workspace describe what the Tower actually does today without
    /// inventing a capability it does not have.
    ///
    /// Republished at the reply rate, like `frameResultCount` beside it.
    ///
    /// **Gated on `cv_lab.run_id`.** A reply belonging to a run this client is
    /// not watching never reaches here — see `watchedCVLabRunID`.
    @Published private(set) var latestFrameResult: TowerFrameResult?

    #endif

    /// The most recent `frame_error` the Tower returned.
    ///
    /// **Deliberately not `#if DEBUG`, unlike `latestFrameResult` beside it.**
    /// That one is gated because it is a *figure a screen renders*, and a
    /// Release build has no camera, sends no frame, and would render a stale
    /// number from a bracket it cannot open. This is a refusal, and the honest
    /// Release rendering of a refusal is the same as the honest Release
    /// rendering of everything else on the frame path: nothing arrives, because
    /// nothing was sent. Publishing it unconditionally costs one optional and
    /// keeps the CV Lab client free of a second conditional-compilation branch
    /// in the middle of its state machine.
    ///
    /// Cleared alongside `latestFrameResult` at all three points, for the same
    /// reason: a refusal from a bracket that has closed is not the current
    /// answer to anything.
    @Published private(set) var latestFrameRefusal: TowerFrameRefusal?

    /// The CV Lab run this client is watching, or `nil`.
    ///
    /// ## The whole staleness rule, in one property
    ///
    /// > **Discard any `frame_result` whose `cv_lab.run_id` is not the run you
    /// > are watching.**
    ///
    /// Learned from the **status document**, never from a `frame_result`, and
    /// that direction is the point: the status document is what says which run
    /// is current, and adopting a run id from a result would make the result
    /// its own authority — the stale reply would nominate itself as the run to
    /// watch and then match.
    ///
    /// While this is `nil` nothing is discarded. That is not a hole: it is the
    /// state of a client that has never read a status, so there is no run it
    /// could be said to be watching and no claim to contradict. The moment a
    /// document arrives, gating begins.
    ///
    /// Because `run_id` is `"<tower_instance_id>-<n>"`, comparing it alone also
    /// covers a reconnect to a **restarted** Tower, whose run numbering starts
    /// again at 1. No separate instance-id check is needed and none is done.
    ///
    /// Cleared at exactly the three points `latestFrameResult` is cleared —
    /// `sendStreamStart()`, `sendStreamStop()` and `teardownConnection` — and
    /// for the same reason each time: a run id held across a boundary the
    /// results did not survive would gate the next bracket's replies against
    /// the previous one's run.
    private(set) var watchedCVLabRunID: String?

    /// How many `frame_result` replies were discarded as belonging to another
    /// run. Non-zero is worth knowing about; it is not an error.
    private(set) var staleFrameResultCount = 0

    /// True between a sent `stream_start` and the matching `stream_stop`.
    /// `sendFrame` will not forward anything while this is false, so a frame
    /// captured in the brief window after `stopCameraSession()` fires (but
    /// before DAT actually tears the stream down) can never reach the Tower.
    ///
    /// **Not `#if DEBUG`, while everything that writes it is.** The two
    /// functions that set it — `sendStreamStart()` and `sendStreamStop()` —
    /// live in the DEBUG-only frame path, so in a Release build this is
    /// permanently `false`, which is the truth about a build with no capture
    /// control on any screen.
    ///
    /// It is readable in both configurations because `TowerWorldBuilderClient`
    /// needs it: "this phone has a capture open" is a fact about the phone's
    /// own situation, and it is what `WorldSessionGate` compares the Tower's
    /// session against. A Release build asks the same question and correctly
    /// gets "no", rather than the question being unaskable there.
    @Published private(set) var isStreamingToTower = false

    /// Whether frames are being **held on the phone** rather than sent.
    ///
    /// ## Why a gate here, and not a camera pause or a `stream_stop`
    ///
    /// The CV Lab's camera card offers Pause and Resume, and neither of the
    /// two obvious implementations is available. DAT offers no app-initiated
    /// camera pause: `StreamState.paused` is something the **glasses** do, on
    /// a temple press or on heat, and `GlassesConnection` documents that it
    /// cannot be overridden in either direction. And sending `stream_stop`
    /// would end the capture lineage on the Tower — the bracket is what a
    /// capture is keyed to, and in the operator's configuration a new bracket
    /// spawns a new follower on the next start.
    ///
    /// So the gate sits on the one hop this app fully owns. While it is
    /// closed the glasses camera keeps running, the socket and the stream
    /// bracket stay up, the Tower keeps the experiment armed, and the only
    /// thing the Tower sees is silence: its `source.receiving_frames` turns
    /// false after `idle_after_s` (5 s), which is exactly what the Lab's LIVE
    /// indicator already reads. Nothing is sent to announce the hold, because
    /// there is no message for it and the Tower's own idle detection is the
    /// truthful report.
    ///
    /// Readable in both configurations, like `isStreamingToTower` beside it,
    /// so `disconnect()` can clear it without a build-conditional. It is
    /// consulted only on the frame path, which is `#if DEBUG`.
    ///
    /// Cleared by `disconnect()` and by `sendStreamStop()`, and deliberately
    /// **not** by `teardownConnection`: a socket that drops and reconnects on
    /// its own schedule is not a person changing their mind, and a hold they
    /// set should still be there when the link comes back.
    @Published private(set) var isFrameSendingPaused = false

    /// Closes the frame gate. Idempotent.
    ///
    /// Frames selected while it is closed are counted as session-gate drops:
    /// they reached a terminal outcome on this side, and
    /// `SenderMetrics.framesUnaccounted` — the one number that exists to prove
    /// frames are not quietly queueing — must not read them as backlog.
    func pauseFrameSending() {
        guard !isFrameSendingPaused else { return }
        isFrameSendingPaused = true
        log("frame sending paused — frames are held on the phone; the camera and the stream bracket stay up")
    }

    /// Reopens the frame gate. Idempotent.
    func resumeFrameSending() {
        guard isFrameSendingPaused else { return }
        isFrameSendingPaused = false
        log("frame sending resumed")
    }

    /// How much outbound latency a frame may carry before the window that
    /// admitted it is considered oversized.
    ///
    /// The send window's capacity is `targetFPS * this` (see
    /// `SendWindow.capacity(forTargetFPS:latencyBudget:)`), so this constant —
    /// not a frame count — is the reviewable decision. At the 12 fps target it
    /// yields a capacity of 4.
    ///
    /// 1/3 s is chosen against the physical baseline, where the measured send
    /// completion time was ~290 ms early in a run. A capacity of 2 against a
    /// 290 ms slot lifetime admits 6.9 fps, which is what that run delivered;
    /// covering that latency at the target rate needs 4 slots. It is
    /// deliberately not larger: at 12 fps every extra slot is another 83 ms a
    /// frame can be stale before it is even written.
    static let outboundLatencyBudget: TimeInterval = 1.0 / 3.0

    /// How long a *full* send window may go without a single completion before
    /// the socket is treated as wedged and replaced.
    ///
    /// `URLSessionWebSocketTask` cannot cancel or time out one outstanding
    /// `send`, so the only lever is the connection itself — which makes this a
    /// deliberately reluctant threshold rather than a latency target. 2 s is
    /// long enough that ordinary congestion, a cellular handover or a Tailscale
    /// path change is ridden out rather than answered with a reconnect, and
    /// short enough that the 52-second peer stall the physical baseline
    /// recorded is cut to about 4% of its cost.
    ///
    /// Anything above this is unrecoverable staleness regardless: a frame
    /// written 2 s after it was captured is not a real-time frame, so nothing
    /// is lost by abandoning it.
    static let sendStallTimeout: TimeInterval = 2.0

    /// The longest gap between consecutive `sendFrame` calls that still counts
    /// as "the main actor was running normally".
    ///
    /// This exists because a send window slot is held from `reserve` until the
    /// completion handler's hop *back onto the main actor* has run — so a slot
    /// that looks old may be a slow network **or** a busy main actor, and those
    /// are exactly the two diagnoses the rest of this file works to keep apart.
    /// Tearing down a perfectly healthy socket because the main thread hitched
    /// would be the worst possible reading of the evidence.
    ///
    /// A main-actor stall is directly observable here: `sendFrame` is called at
    /// the selection rate (~83 ms apart at 12 fps), so a gap far larger than
    /// that means this actor was not running. When that happens the stall
    /// verdict is skipped for one frame, by which time the completion hops
    /// queued during the hitch have run and released their slots.
    ///
    /// 1 s is ~12 missed frames — far beyond any normal scheduling jitter, and
    /// well under `sendStallTimeout`, so a genuine transport stall is still
    /// caught on the very next frame.
    static let mainActorGapAllowance: TimeInterval = 1.0

    /// The bounded set of sends outstanding on the current socket, and the
    /// pipeline's actual rate limiter. See `SendWindow` for why its capacity is
    /// derived from a latency budget rather than picked.
    private var sendWindow: SendWindow

    /// When `sendFrame` last ran, used only to tell a wedged socket from a
    /// wedged main actor. See `mainActorGapAllowance`. Cleared on teardown, so
    /// the first frame of a new connection never inherits the old one's pulse.
    private var lastSendFrameAt: TimeInterval?

    /// Capacity of the send window, exposed so the developer surface can show
    /// the `capacity / slotLifetime` arithmetic that explains the send rate,
    /// and so tests can assert the derived sizing.
    var maxFramesInFlight: Int { sendWindow.capacity }

    /// Per-frame logging cadence, in send calls. At the target rate this path
    /// runs ~12 times a second and `print` with string interpolation is not
    /// free, so routine success and routine drops are decimated. The
    /// authoritative per-stage counts live in `metrics`.
    private static let frameLogStride = 12
    private var frameLogCounter = 0
    /// Separate budget from `frameLogCounter` so the outbound and inbound
    /// lines cannot crowd each other out — each stays at ~1 Hz.
    private var resultLogCounter = 0

    /// Sender-side instrumentation. Shared with `GlassesConnection` via
    /// `ProjectManager`, which owns both.
    private let metrics: SenderMetrics

    // MARK: Result channel

    /// The Tower's most recent capability declaration, cached.
    ///
    /// Requested once per connection, immediately after the pong — the contract
    /// requires discovery to follow handshake validation, and asking earlier
    /// would read our own reply into the handshake.
    ///
    /// **Deliberately not cleared on teardown.** What the Tower can do is a
    /// property of the Tower's build, not of this socket. Clearing it would
    /// turn every dropped connection into `.noContract` — "this will never
    /// work" — when the truthful reading is `.towerUnreachable`, and those two
    /// call for opposite responses from a person.
    @Published private(set) var cartridgeDeclaration: TowerCartridgeDeclaration?

    /// Every result-channel message, in arrival order.
    ///
    /// A subject rather than four `@Published` properties because ordering
    /// between them is load-bearing: `result_subscribed` is followed
    /// immediately by the first `cartridge_result`, and a consumer that saw
    /// them out of order would file the snapshot against no subscription.
    private let resultEvents = PassthroughSubject<CartridgeResultEvent, Never>()

    /// The result channel, for whoever owns a cartridge's contract.
    ///
    /// `TowerClient` decodes the envelope and nothing else. It does not know
    /// what a world is, does not subscribe on anyone's behalf, and holds no
    /// cartridge state — the cartridge client owned by `ProjectManager` does
    /// all three. That split is what keeps this file cartridge-blind.
    var cartridgeResults: AnyPublisher<CartridgeResultEvent, Never> {
        resultEvents.eraseToAnyPublisher()
    }

    // MARK: CV Lab control plane

    private let cvLabControlEvents = PassthroughSubject<CVLabEvent, Never>()

    /// The CV Lab's control plane, for whoever owns its contract.
    ///
    /// The same split as `cartridgeResults`: this file decodes the message
    /// envelope and passes the status document through undecoded. It does not
    /// know what an experiment is, holds no lifecycle state, and sends no
    /// command on anyone's behalf.
    ///
    /// **The CV Lab's commands travel here rather than on the result channel,
    /// and that is the contract's own exception.** `tower/results/` is a
    /// read-only reporting surface — a Tower test forbids a call named
    /// `observe` or `build` anywhere inside it — so a mutation must not travel
    /// on it. The CV Lab's start, pause and stop are plain socket messages on
    /// `/ws` instead, and there is **no HTTP surface for any of them**: a
    /// command needs the connection it was issued on to still be there when the
    /// outcome arrives, and an arm may take two minutes.
    var cvLabEvents: AnyPublisher<CVLabEvent, Never> {
        cvLabControlEvents.eraseToAnyPublisher()
    }

    private var session: URLSession?
    private var webSocketTask: URLSessionWebSocketTask?
    private var validationTask: Task<Void, Never>?
    private var receiveTask: Task<Void, Never>?

    // MARK: Reconnect

    /// Whether a connection that drops or stalls is re-established
    /// automatically.
    ///
    /// Defaults to `false` — and the production graph in `ProjectManager`
    /// passes `true`. Off by default because reconnect makes `status` a
    /// *sequence* rather than a settled value, and every existing test that
    /// asserts "a dropped connection ends at `.failed`" is asserting about the
    /// settled value. Opting in explicitly keeps those assertions meaningful
    /// instead of racing a reconnect.
    private let autoReconnect: Bool

    /// Endpoint to return to. Set on every `openConnection(to:)` and cleared
    /// by `disconnect()`, so a user-initiated disconnect is never undone by a
    /// reconnect scheduled moments earlier.
    private var reconnectURL: URL?
    private var reconnectTask: Task<Void, Never>?
    /// Failed attempts since the last connection that *held*. Drives the
    /// backoff and the give-up point.
    private var reconnectAttempt = 0

    /// When the current connection reached `.online`, or `nil` if it has not.
    ///
    /// The budget is refilled from this on the way *down* rather than the way
    /// up, because reaching `.online` proves only that the socket opened and
    /// the Tower answered one ping. A Tower that accepts a connection and then
    /// immediately wedges would otherwise reset the counter on every lap and
    /// reconnect forever — a flap loop the bounded schedule exists to prevent.
    private var becameOnlineAt: TimeInterval?

    /// How long a connection must survive before it counts as healthy enough
    /// to earn a fresh reconnect budget. Comfortably longer than one full
    /// backoff schedule, so a flapping Tower always exhausts the schedule and
    /// stops, while a session that runs for minutes and then drops is treated
    /// as the isolated blip it is.
    private static let reconnectBudgetRefillAfter: TimeInterval = 30

    /// Backoff schedule, in seconds. Bounded on purpose: a Tower that is
    /// simply not running must end at a visible `.failed` rather than retrying
    /// forever behind a pill that never settles, and the app has a manual
    /// Connect control for the deliberate retry.
    ///
    /// The delays total 15.5 s, but each attempt also carries up to the 6 s
    /// pong timeout in `validateConnection`, so giving up against a dead
    /// endpoint takes up to ~45 s.
    private static let reconnectBackoff: [TimeInterval] = [0.5, 1, 2, 4, 8]

    /// The shipped send-window capacity, as the arithmetic that justifies it
    /// rather than as a literal.
    static var defaultMaxFramesInFlight: Int {
        SendWindow.capacity(
            forTargetFPS: FrameRateGate.towerTargetFPS,
            latencyBudget: outboundLatencyBudget
        )
    }

    override init() {
        self.metrics = SenderMetrics()
        self.sendWindow = SendWindow(
            capacity: Self.defaultMaxFramesInFlight,
            stallTimeout: Self.sendStallTimeout
        )
        self.autoReconnect = false
        self.handshakeLegTimeout = Self.defaultHandshakeLegTimeout
        super.init()
    }

    /// - Parameters:
    ///   - metrics: Shared sender instrumentation.
    ///   - maxFramesInFlight: Overridable so tests can drive the bounded send
    ///     window deterministically. `nil` uses the latency-budgeted capacity
    ///     described on `outboundLatencyBudget`.
    ///   - stallTimeout: Overridable so tests can trip stall detection without
    ///     waiting `sendStallTimeout` seconds. `nil` uses the shipped value.
    ///   - autoReconnect: See the property of the same name.
    ///
    /// Both overrides are `nil`-defaulted and resolved in the body rather than
    /// being computed default arguments: default arguments are evaluated
    /// outside this type's actor, and `defaultMaxFramesInFlight` reads
    /// main-actor-isolated configuration. `GlassesConnection.init` avoids the
    /// same trap for the same reason.
    init(
        metrics: SenderMetrics,
        maxFramesInFlight: Int? = nil,
        stallTimeout: TimeInterval? = nil,
        autoReconnect: Bool = false,
        handshakeLegTimeout: Int? = nil
    ) {
        self.metrics = metrics
        self.handshakeLegTimeout = handshakeLegTimeout ?? Self.defaultHandshakeLegTimeout
        self.sendWindow = SendWindow(
            capacity: maxFramesInFlight ?? Self.defaultMaxFramesInFlight,
            stallTimeout: stallTimeout ?? Self.sendStallTimeout
        )
        self.autoReconnect = autoReconnect
        super.init()
    }

    /// - Parameter url: The Tower endpoint. `nil` uses the real Tower
    ///   (`TowerConfiguration.webSocketURL`); overridable so tests can point
    ///   this at a local mock server instead. `nil`-defaulted and resolved in
    ///   the body for the reason `init` gives: a default argument is evaluated
    ///   outside this type's actor, and `webSocketURL` is main-actor isolated.
    ///
    /// A caller-initiated connect also refills the reconnect budget: an
    /// exhausted schedule is how the client says "I have stopped trying", and
    /// a deliberate tap on Connect is the user saying to try again.
    func connect(to url: URL? = nil) {
        let url = url ?? TowerConfiguration.webSocketURL
        // Refilled only when this call is actually going to open a socket. It
        // used to be reset unconditionally, before `openConnection`'s
        // `.connecting` guard — so a redundant tap during an in-flight connect
        // did nothing visible while silently resurrecting an exhausted
        // schedule. The budget is meant to say "I have stopped trying"; a
        // no-op must not undo that.
        if status != .connecting { reconnectAttempt = 0 }
        openConnection(to: url)
    }

    /// Connects only if nothing is connected or in flight, and **without**
    /// refilling the reconnect budget.
    ///
    /// The entry point for automation — app launch, specifically. It is
    /// deliberately not `connect()`: that call means "the user asked to try
    /// again", which is why it refills the budget and why it is allowed to
    /// replace a live connection. Neither is true of code running on its own
    /// initiative, and routing automation through the same door would dissolve
    /// the bound that stops a dead Tower from being retried forever.
    ///
    /// Guarding on `.offline` also makes this safe to call more than once: it
    /// will not disturb a healthy connection, cancel a pending reconnect, or
    /// restart a schedule that has already given up. A Tower that has failed
    /// stays failed and visible until the user acts.
    func connectIfIdle(to url: URL? = nil) {
        let url = url ?? TowerConfiguration.webSocketURL
        guard status == .offline else {
            log("automatic connect skipped — status is \(status)")
            return
        }
        openConnection(to: url)
    }

    /// The connect path itself, without refilling the reconnect budget — so a
    /// scheduled reconnect advances through the backoff rather than resetting
    /// it and retrying forever.
    private func openConnection(to url: URL) {
        // A different Tower is a different set of capabilities.
        //
        // `cartridgeDeclaration` deliberately survives a teardown, and that is
        // right: what a Tower can do is a property of its build, not of one
        // socket, so clearing it on every drop would turn a reconnect into
        // "this will never work". None of that reasoning applies when the
        // endpoint itself changes — the previous Tower's declaration says
        // nothing about this one. Left standing, it would race the new
        // connection's own `cartridges` reply and could drive the first
        // `result_subscribe` with the old Tower's contract, which the new one
        // answers with `contract_mismatch` and prose about the wrong thing.
        if let previous = reconnectURL, previous != url {
            log("endpoint changed \(previous) -> \(url); dropping the previous Tower's declaration")
            cartridgeDeclaration = nil
        }

        // Recorded before the in-flight guard below, so that a connect made
        // while one is already under way still retargets a later reconnect. Do
        // not move this after the guard: a pending reconnect would then quietly
        // return to the *previous* endpoint.
        reconnectURL = url

        guard status != .connecting else { return }

        // A caller-initiated connect supersedes any pending reconnect, so the
        // two cannot both open a socket. The reconnect path clears this itself
        // before calling in, so this is a no-op there rather than
        // self-cancellation.
        reconnectTask?.cancel()
        reconnectTask = nil

        if webSocketTask != nil {
            log("connect() called with a previous connection still active — tearing it down first")
        }
        teardownConnection(cancelWith: .normalClosure)

        log("connection attempt: \(url)")
        status = .connecting

        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        self.session = session
        let task = session.webSocketTask(with: url)
        webSocketTask = task
        task.resume()
        log("WebSocket opened (resume() called)")

        validationTask = Task { [weak self] in
            await self?.validateConnection(task: task)
        }
        armHandshakeWatchdog(for: task)
    }

    /// Guarantee that a connection attempt ends.
    ///
    /// `validateConnection` awaits a `send` and a `receive` on the socket, and
    /// neither reliably returns when a peer accepts the TCP connection and then
    /// never completes the WebSocket upgrade. Structured-concurrency timeouts
    /// cannot rescue that — a task group has to await its own child before it
    /// can throw, and that child is the stuck call — so the bound is enforced
    /// from outside, on the thing that really is cancellable: the socket.
    ///
    /// `fail(_:task:)` tears the connection down, which makes the pending calls
    /// error out, sets `.failed`, and lets `scheduleReconnect` advance. Without
    /// it a hung connect parked the client in `.connecting` permanently: the
    /// attempt budget was never spent, nothing tried again, and nothing said so.
    ///
    /// Found while decomposing a real 9-second reconnect on a physical walk,
    /// ~8.5 s of which was spent in `.connecting`. This does not make that walk
    /// faster — it makes the failure terminate.
    private func armHandshakeWatchdog(for task: URLSessionWebSocketTask) {
        handshakeWatchdog?.cancel()
        // Both legs plus room for the ordinary case to finish on its own, so
        // this fires only when the per-leg bounds have failed to.
        let budget = handshakeLegTimeout * 2
        handshakeWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(budget))
            guard !Task.isCancelled else { return }
            self?.handshakeDidTimeOut(task: task, after: budget)
        }
    }

    private func handshakeDidTimeOut(task: URLSessionWebSocketTask, after seconds: Int) {
        // Only if this attempt is still the current one and still unresolved. A
        // connection that came online, failed on its own, or was superseded has
        // already had its outcome.
        guard isCurrent(task), status == .connecting else { return }
        fail("Handshake did not complete within \(seconds)s", task: task)
    }

    func disconnect() {
        log("disconnect() called")
        // Cleared before teardown, so a failure observed on the way down
        // cannot schedule a reconnect the user just asked to stop. `fail()`
        // also refuses to act once `status` is `.offline`, but that is set
        // after teardown — this is what closes the window between the two.
        cancelReconnect()
        teardownConnection(cancelWith: .normalClosure)
        status = .offline
        // The user asked for this connection to end, so the hold they set
        // during it ends too — a fresh connect streams. See the property for
        // why an *automatic* teardown does not do this.
        isFrameSendingPaused = false
        log("disconnect cleanup complete")
    }

    private func cancelReconnect() {
        reconnectTask?.cancel()
        reconnectTask = nil
        reconnectURL = nil
        reconnectAttempt = 0
        becameOnlineAt = nil
    }

    /// Queues one delayed reconnect attempt, if automatic reconnect is enabled
    /// and the schedule has not been exhausted.
    ///
    /// Reconnect is not an optimisation here — it is what makes stall recovery
    /// possible at all. Tearing down a wedged socket without restoring one
    /// would trade a stalled pipeline for a dead one.
    private func scheduleReconnect() {
        guard autoReconnect, let url = reconnectURL else { return }
        guard reconnectTask == nil else { return }

        // Refill the budget only for a connection that actually held. Reading
        // it here, at the point of failure, is what makes "it worked for a
        // while" mean something — see `becameOnlineAt`.
        if let onlineAt = becameOnlineAt,
           MonotonicClock.now - onlineAt >= Self.reconnectBudgetRefillAfter {
            reconnectAttempt = 0
        }
        becameOnlineAt = nil

        guard reconnectAttempt < Self.reconnectBackoff.count else {
            log("reconnect given up after \(reconnectAttempt) attempts — use Connect to retry")
            return
        }

        let delay = Self.reconnectBackoff[reconnectAttempt]
        reconnectAttempt += 1
        let attempt = reconnectAttempt
        log("reconnect attempt \(attempt) scheduled in \(delay)s")

        reconnectTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled else { return }
            guard let self else { return }
            // Cleared before `connect(to:)` so that call's own
            // supersede-any-pending-reconnect step does not cancel the task it
            // is currently running on.
            self.reconnectTask = nil
            guard self.reconnectURL == url else {
                self.log("reconnect attempt \(attempt) abandoned — endpoint changed or disconnected")
                return
            }
            self.log("reconnect attempt \(attempt) starting")
            self.openConnection(to: url)
        }
    }

    #if DEBUG
    /// Sends one already-decoded camera frame to the Tower as a JSON text
    /// message over the existing open WebSocket connection — the same
    /// connection/`send` path validated by the ping/pong milestone. No new
    /// networking, no binary framing: this reuses `webSocketTask.send(.string(...))`
    /// exactly as `validateConnection` already does.
    ///
    /// Minimal proof-of-path only: one JPEG-encoded, base64-in-JSON frame at
    /// a time. Not batching, not compressing beyond a fixed JPEG quality, not
    /// adapting rate — see docs/03-ROADMAP.md V0.7 for where that belongs.
    func sendFrame(_ image: UIImage, width: Int, height: Int, sequence: Int) {
        frameLogCounter += 1
        let shouldLog = frameLogCounter % Self.frameLogStride == 1

        guard status == .online, let task = webSocketTask else {
            metrics.recordSessionGateDrop()
            if shouldLog {
                log("frame #\(sequence) not sent — Tower not online (status=\(status))")
            }
            return
        }
        guard isStreamingToTower else {
            metrics.recordSessionGateDrop()
            if shouldLog {
                log("frame #\(sequence) not sent — no stream_start sent yet (or stream_stop already sent)")
            }
            return
        }
        guard !isFrameSendingPaused else {
            // A person closed the gate. Not an error, and not a stall: the
            // socket is fine and nothing is queued. Counted as a session-gate
            // drop for the reason `pauseFrameSending()` gives, and checked
            // before the stall test below so a long hold cannot be misread as
            // a socket that stopped draining.
            metrics.recordSessionGateDrop()
            if shouldLog {
                log("frame #\(sequence) held — frame sending is paused")
            }
            return
        }
        let now = MonotonicClock.now

        // How long since the previous frame was offered. This is the main
        // actor's own pulse, and it is what stops a main-actor hitch from
        // being misread as a wedged socket — see `mainActorGapAllowance`.
        let sinceLastFrame = lastSendFrameAt.map { now - $0 }
        lastSendFrameAt = now

        // A full window that has not returned a single slot within
        // `stallTimeout` is a socket that is not draining. There is no way to
        // cancel the outstanding sends individually, so the connection is the
        // unit of recovery — and without this the pipeline simply reports
        // send-window drops for as long as the peer takes to resume, which in
        // the physical baseline was 52 seconds.
        //
        // Checked here, on the send path, rather than from a timer: this runs
        // at the selection rate whenever there are frames to send, which is
        // exactly when a stall costs something. With no frames arriving there
        // is no throughput to lose, and the next frame detects it immediately.
        //
        // The gap test is the false-positive guard. A slot's age includes the
        // completion handler's hop back to this actor, so if this actor has
        // itself been stalled, every slot looks old through no fault of the
        // socket. Requiring that the previous frame was offered recently means
        // the verdict is only ever reached while the main actor is demonstrably
        // running — and one frame later, the hops queued during the hitch have
        // drained. A first frame (`nil`) is treated as a gap, since there is no
        // pulse to judge yet.
        let mainActorWasResponsive = (sinceLastFrame ?? .infinity) <= Self.mainActorGapAllowance
        if mainActorWasResponsive, sendWindow.isStalled(at: now) {
            let age = sendWindow.oldestAge(at: now) ?? 0
            // The healthy distribution alongside the verdict. A stall is only
            // interpretable against what this link normally does: an oldest-slot
            // age of 2 s means one thing when the running max is 90 ms and quite
            // another when it is 1.8 s. These rows exist on the developer
            // surface and had never been recorded next to an actual stall.
            let snapshot = metrics.snapshot
            let slotMax: String = snapshot.slotLifetimeMsMax.map { String(format: "%.0f", $0) } ?? "-"
            let sendMax: String = snapshot.sendLatencyMsMax.map { String(format: "%.0f", $0) } ?? "-"
            let ageText: String = String(format: "%.1f", age)
            let outstanding: Int = sendWindow.inFlight
            let priorRecoveries: Int = snapshot.stallRecoveries
            let diagnosis: String = "slot ms max \(slotMax), send ms max \(sendMax), prior stall recoveries \(priorRecoveries)"
            log("send window stalled — \(outstanding) sends outstanding, oldest \(ageText)s (\(diagnosis)); replacing the connection")
            // This frame still has to reach a terminal outcome, or every stall
            // would leave one selected frame permanently unaccounted for and
            // `framesUnaccounted` would drift upwards — the one number that
            // exists to prove frames are not quietly queueing. A window drop is
            // the honest label: `isStalled` implies `isFull`, so the frame was
            // dropped for a full window, exactly like the ones before it.
            metrics.recordSendWindowDrop()
            // Recorded after the teardown it describes, so the counter can only
            // ever report recoveries that actually happened.
            fail("Send stalled for \(String(format: "%.1f", age))s", task: task)
            metrics.recordStallRecovery()
            return
        }

        // Checked before encoding, so a frame we are going to drop never costs
        // a JPEG encode.
        guard !sendWindow.isFull else {
            metrics.recordSendWindowDrop()
            if shouldLog {
                log("frame #\(sequence) dropped — \(sendWindow.inFlight) sends already in flight (window \(sendWindow.capacity))")
            }
            return
        }

        // Read again rather than reusing `now`, so the encode figure covers the
        // encode and nothing else — `now` was taken before the stall and
        // window checks above.
        let encodeStart = MonotonicClock.now
        guard let jpegData = image.jpegData(compressionQuality: 0.5) else {
            metrics.recordEncodeFailure()
            log("frame #\(sequence) failed to encode as JPEG")
            return
        }

        let payload: [String: Any] = [
            "type": "frame",
            "seq": sequence,
            "width": width,
            "height": height,
            "format": "jpeg",
            "data": jpegData.base64EncodedString(),
        ]

        guard
            let jsonData = try? JSONSerialization.data(withJSONObject: payload),
            let jsonText = String(data: jsonData, encoding: .utf8)
        else {
            metrics.recordEncodeFailure()
            log("frame #\(sequence) failed to serialize JSON payload")
            return
        }
        let reservedAt = MonotonicClock.now
        metrics.recordEncode(seconds: reservedAt - encodeStart)

        // Cannot fail: the window was checked not-full a few lines above and
        // nothing else can reserve in between — `sendFrame` is main-actor
        // isolated and contains no suspension point. Handled rather than
        // force-unwrapped so a future edit that breaks that property loses a
        // frame instead of trapping in the user's hands.
        guard let token = sendWindow.reserve(at: reservedAt) else {
            metrics.recordSendWindowDrop()
            log("frame #\(sequence) dropped — send window closed between check and reserve")
            return
        }
        metrics.recordSendAttempt(wireBytes: jsonData.count)
        if shouldLog {
            log("frame #\(sequence) sending \(jsonData.count) bytes (\(width)x\(height), jpeg \(jpegData.count) bytes)")
        }

        task.send(.string(jsonText)) { [weak self] error in
            // Sampled here, in the transport's own completion handler, before
            // the main-actor hop. That is the whole point: the difference
            // between this instant and the one measured after the hop
            // separates "the network is slow" from "the main actor is busy",
            // which are opposite diagnoses and were previously folded into a
            // single unmeasured number.
            let completedAt = MonotonicClock.now
            // `[weak self]` on the Task itself, not inherited from the send
            // completion's capture. A weak capture is a mutable binding, and
            // reading the enclosing closure's copy from concurrently-executing
            // code is an error under the Swift 6 language mode. Re-capturing
            // here is evaluated when the Task is created, which is allowed,
            // and keeps the lifetime semantics identical.
            Task { @MainActor [weak self] in
                guard let self else { return }
                // A completion for a socket this client no longer owns: its
                // reservation was already cleared by teardown, so `release`
                // returns nil and the slot count is left alone — otherwise
                // this would credit a slot on the *next* connection and
                // permanently widen its window. The outcome is also not this
                // connection's to report — but it still has to be *recorded*,
                // or the frame would look permanently in flight and the
                // accounting invariant would false-alarm after every
                // disconnect.
                let releasedAt = MonotonicClock.now
                guard
                    self.isCurrent(task),
                    let slotLifetime = self.sendWindow.release(token, at: releasedAt)
                else {
                    self.metrics.recordSendAbandoned()
                    return
                }
                self.metrics.recordSlotTiming(
                    sendLatency: completedAt - reservedAt,
                    slotLifetime: slotLifetime
                )

                if let error {
                    self.metrics.recordSendFailure()
                    self.log("frame #\(sequence) send failed: \(error.localizedDescription)")
                    self.fail("Send failed: \(error.localizedDescription)", task: task)
                } else {
                    self.metrics.recordSendSuccess()
                    if shouldLog {
                        self.log("frame #\(sequence) sent")
                    }
                }
            }
        }
    }

    /// Marks the stream as active and sends `{"type":"stream_start"}` once,
    /// over the existing Tower WebSocket, so the Tower knows to expect
    /// frames. Fire-and-forget: no response is awaited or expected. A no-op
    /// if already streaming, so a redundant call (e.g. DAT re-delivering the
    /// `.streaming` state) can't send it twice for the same session.
    func sendStreamStart() {
        guard !isStreamingToTower else {
            log("stream_start suppressed — already streaming")
            return
        }
        // The flag is set only if the marker actually reached a socket.
        // Setting it first meant a start attempted while the Tower was
        // offline left `isStreamingToTower == true` with the Tower never
        // having been told, so every frame of that session was forwarded
        // outside any stream bracket and the eventual `stream_stop` was
        // unmatched.
        guard sendLifecycleMarker(type: "stream_start") else { return }
        isStreamingToTower = true
        // Scoped to one stream bracket, which is a narrower thing than a
        // camera session now that a dropped connection reopens the bracket on
        // its own. The product screen therefore reads
        // `SenderMetrics.frameResults` instead; this counter stays per-bracket
        // and is shown only on the developer surface, where "replies on the
        // current bracket" is the useful reading. A lifetime-cumulative
        // counter would diverge by tens of thousands over a long run and
        // invite reading the pair as a delivery ratio.
        frameResultCount = 0
        // Scoped to the bracket for the same reason as the count above: a reply
        // from the previous bracket displayed against a fresh one is a stale
        // claim about the current session.
        latestFrameResult = nil
        clearCVLabRunWatch()
    }

    /// Marks the stream as inactive and sends `{"type":"stream_stop"}` once.
    /// From this point, `sendFrame` will not forward anything until the next
    /// `sendStreamStart()`. A no-op if not currently streaming.
    func sendStreamStop() {
        // A hold belongs to the camera session it was set in. The camera has
        // stopped; a later start must stream, not inherit a pause nobody
        // remembers setting and that no control on that later screen shows.
        // Cleared BEFORE the bracket guard, deliberately: a socket drop
        // already closed the bracket (`teardownConnection`), and a camera
        // stopped during that gap would otherwise keep the hold into the
        // next session, where Home has no control that shows it.
        isFrameSendingPaused = false
        guard isStreamingToTower else {
            log("stream_stop suppressed — not currently streaming")
            return
        }
        isStreamingToTower = false
        // Cleared with the bracket it belongs to. The tile that shows it is
        // captioned "latest Tower reply", and after a stop there is no current
        // reply - leaving the last one on screen would date it silently.
        latestFrameResult = nil
        clearCVLabRunWatch()
        _ = sendLifecycleMarker(type: "stream_stop")
    }

    #endif

    /// Forget which CV Lab run this client is watching, and the last refusal.
    ///
    /// ## Why this is called from exactly three places
    ///
    /// The same three that clear `latestFrameResult`: `sendStreamStart()`,
    /// `sendStreamStop()` and `teardownConnection`. The held run id and the
    /// held reply have identical lifetimes because they answer the same
    /// question — *what is the Tower currently telling us about frames* — and a
    /// run id that outlived a bracket would silently change what the gate
    /// means: replies from the next bracket would be measured against a run
    /// that ended, and every one of them discarded.
    ///
    /// It is **not** called when the run merely changes. That case is handled
    /// where the new run id arrives, by adopting it and dropping the reading
    /// that belonged to the old one — see `watchCVLabRun(_:)`.
    ///
    /// Outside `#if DEBUG` although two of its three callers are inside it, so
    /// that `teardownConnection`'s Release path can call it: the run watch is
    /// fed by the status document, which is a read-only surface a Release build
    /// reaches perfectly well.
    private func clearCVLabRunWatch() {
        watchedCVLabRunID = nil
        latestFrameRefusal = nil
    }

    /// Adopt the run the Tower's status document says is current.
    ///
    /// The **only** writer of `watchedCVLabRunID` other than the three clears,
    /// and it is fed exclusively from status documents — a `cv_lab_status`
    /// reply, a pushed one, or the unchanged document a `cv_lab_error` carries.
    /// Never from a `frame_result`: a result that nominated its own run as the
    /// one to watch would always match, which is the gate deleting itself.
    ///
    /// When the run **changes**, the held reading is dropped with it. A run is
    /// the unit of provenance, and the Tower makes the same move on its side —
    /// starting a different experiment mints a new run and takes the previous
    /// one's figures out of the document entirely, *"because keeping an old
    /// summary beside a new one is how a number from the wrong experiment ends
    /// up on a screen"*. Leaving the last reading on screen across a switch
    /// would put the old experiment's number under the new experiment's name.
    ///
    /// A `nil` run id — the Lab is idle, or unavailable — clears the watch
    /// rather than being ignored: there is no current run, so there is nothing
    /// to gate against, and holding the previous one would discard the results
    /// of whatever starts next.
    ///
    /// ## Why this is not `private`, and why that is the whole fix
    ///
    /// It was, and the two callers below are both `cv_lab_status` messages.
    /// **The Tower sends `cv_lab_status` only in reply to a message on that
    /// same connection** (`tower/routes/cv_lab_ws.py`) — there is no unsolicited
    /// push. So those two callers cover exactly one case: a run change *this
    /// phone caused*.
    ///
    /// The Lab has **one slot shared by every connection, and last start wins**.
    /// A run change caused by anyone else — a second operator, the bench's own
    /// `cv_lab_smoke.py` — reaches this app only as a `cartridge_result` on the
    /// subscription. With the gate private, nothing on that path could update
    /// it, and the consequences were silent and severe:
    ///
    ///  - every subsequent `frame_result` carried the new run and was
    ///    discarded, so `SenderMetrics.frameResults` — the counter the product
    ///    screen reads — **froze while frames were being answered normally**;
    ///  - `latestFrameResult` was never cleared, so the card kept showing the
    ///    **previous experiment's figures**, indefinitely, while the panel above
    ///    it correctly updated to the new experiment's name with a `live` badge.
    ///
    /// That is precisely the outcome this gate exists to prevent — a number
    /// from the wrong experiment on a screen — produced *by* the gate.
    ///
    /// So the cartridge client calls this when it applies a status document it
    /// received over the subscription. That does not weaken the rule the doc
    /// above states: the feed is still a **status document** and never a
    /// `frame_result`, which is the distinction that matters. It widens the set
    /// of status documents from "the ones we asked for" to "all of them".
    ///
    /// Idempotent by the first guard, so a client that also asked directly can
    /// call it twice with no effect.
    func watchCVLabRun(_ runID: String?) {
        guard runID != watchedCVLabRunID else { return }
        log("cv_lab run watch: \(watchedCVLabRunID ?? "none") -> \(runID ?? "none")")
        watchedCVLabRunID = runID
        #if DEBUG
        latestFrameResult = nil
        #endif
        latestFrameRefusal = nil
    }

    #if DEBUG

    /// Shared send path for the two stream lifecycle markers — same
    /// WebSocket, same fire-and-forget `send` used by `sendFrame`, no new
    /// connection, no reply awaited. Deliberately bypasses the frame send
    /// window: markers are two-byte payloads that define session boundaries,
    /// and delaying or dropping one corrupts every frame count on either side
    /// of it.
    ///
    /// - Returns: whether the marker was handed to a socket. Not whether the
    ///   Tower received it — that is still fire-and-forget.
    private func sendLifecycleMarker(type: String) -> Bool {
        guard status == .online, let task = webSocketTask else {
            log("\(type) not sent — Tower not online (status=\(status))")
            return false
        }
        guard
            let jsonData = try? JSONSerialization.data(withJSONObject: ["type": type]),
            let jsonText = String(data: jsonData, encoding: .utf8)
        else {
            log("\(type) failed to serialize JSON payload")
            return false
        }
        task.send(.string(jsonText)) { [weak self] error in
            // Re-captured weakly here for the same reason as in `sendFrame`.
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let error {
                    self.log("\(type) send failed: \(error.localizedDescription)")
                    self.fail("Send failed: \(error.localizedDescription)", task: task)
                } else {
                    self.log("\(type) sent")
                }
            }
        }
        return true
    }
    #endif

    // MARK: - Result channel: outbound

    /// Asks the Tower what it can report on.
    ///
    /// Sent once per connection, from `validateConnection` after the pong has
    /// been read — never before. Not `#if DEBUG`: the result channel is
    /// read-only and says nothing about frames, so a Release build that cannot
    /// stream is still entitled to a truthful answer about what the Tower can
    /// do.
    func requestCartridgeDeclaration() {
        sendResultMessage(["type": "cartridges"], label: "cartridges")
    }

    /// Opens a subscription. The reply is a `result_subscribed` followed
    /// immediately by a complete snapshot, whatever cursor was sent.
    ///
    /// `contract` is included so the Tower refuses outright rather than
    /// serving a payload this build was not written against — a
    /// `contract_mismatch` error is a better outcome than a silent
    /// misinterpretation.
    func subscribeToResults(cartridge: String, resultType: String, contract: String) {
        sendResultMessage(
            [
                "type": "result_subscribe",
                "cartridge": cartridge,
                "result_type": resultType,
                "contract": contract,
            ],
            label: "result_subscribe(\(cartridge))"
        )
    }

    /// Closes a subscription. Not required before disconnecting — the Tower
    /// treats a closed socket as sufficient cleanup — so this exists for the
    /// case where the connection outlives the reason to be subscribed.
    func unsubscribeFromResults(subscriptionID: String) {
        sendResultMessage(
            ["type": "result_unsubscribe", "subscription_id": subscriptionID],
            label: "result_unsubscribe(\(subscriptionID))"
        )
    }

    // MARK: - CV Lab control: outbound

    /// Asks the Lab for its whole state.
    ///
    /// **Not `#if DEBUG`, and this is the message that proves the rule.** The
    /// CV Lab's read-only half is reachable from a build with no camera: a
    /// Release build can enumerate the experiments and read the Lab's state
    /// truthfully, and the contract versions the control vocabulary separately
    /// from the status document precisely so that *"a client may implement the
    /// read-only half and never send a command"*.
    ///
    /// - Parameter requestID: echoed back on the reply, so an answer can be
    ///   matched to the button that was pressed. Bounded by
    ///   `boundedRequestID(_:)` before it goes out.
    func sendCVLabStatusRequest(requestID: String? = nil) {
        sendCVLabCommand(.status, requestID: requestID)
    }

    /// Selects **and arms** an experiment, replacing whatever was running.
    ///
    /// There is deliberately no separate select message: selection without
    /// arming is a state nobody needs on the wire.
    ///
    /// The reply is immediate and says `starting`, not `running` — an arm is
    /// asynchronous, and a start that later fails to load has **already been
    /// answered `accepted`** by then. The outcome arrives as state on a later
    /// status document, which is why a client that sends this must also read
    /// status.
    func sendCVLabStart(experimentID: String, requestID: String? = nil) {
        sendCVLabCommand(
            .start,
            extra: ["experiment_id": experimentID],
            requestID: requestID
        )
    }

    /// Stops processing and keeps the experiment loaded.
    ///
    /// - Parameter runID: the run the button was drawn against. **Send it
    ///   whenever there is one.** A command naming a run that is no longer
    ///   current is refused `stale_run` rather than applied to whichever run
    ///   is — which is the difference between a refusal and the wrong run being
    ///   paused by the wrong person.
    func sendCVLabPause(runID: String? = nil, requestID: String? = nil) {
        sendCVLabCommand(.pause, runID: runID, requestID: requestID)
    }

    /// Resumes a paused run. Costs nothing: the experiment never left memory.
    func sendCVLabResume(runID: String? = nil, requestID: String? = nil) {
        sendCVLabCommand(.resume, runID: runID, requestID: requestID)
    }

    /// Ends the run, releases the experiment, and **keeps the figures**.
    ///
    /// The only way to keep a run readable: starting a different experiment
    /// mints a new run and takes the previous one's figures out of the document
    /// entirely.
    func sendCVLabStop(runID: String? = nil, requestID: String? = nil) {
        sendCVLabCommand(.stop, runID: runID, requestID: requestID)
    }

    /// Shared shape for the five control messages.
    ///
    /// Sent down `sendUnescalatedMessage` rather than `sendLifecycleMarker`,
    /// and that choice is the important one: **a refused or undeliverable
    /// command must not tear down the frame socket.** A start is a request
    /// about what the Tower should compute; the frames are the session. Failing
    /// the connection because a control message did not land would let the
    /// control plane do the one thing the contract promises it cannot — affect
    /// the frame path.
    private func sendCVLabCommand(
        _ command: CVLabCommand,
        extra: [String: Any] = [:],
        runID: String? = nil,
        requestID: String? = nil
    ) {
        var object: [String: Any] = ["type": command.rawValue]
        for (key, value) in extra { object[key] = value }
        if let runID { object["run_id"] = runID }
        if let requestID = boundedRequestID(requestID) { object["request_id"] = requestID }
        sendUnescalatedMessage(object, label: command.rawValue)
    }

    /// Bounds a `request_id` to the 64 characters the Tower echoes.
    ///
    /// **A longer one is dropped, not refused**: the command applies and the
    /// reply simply comes back carrying no `request_id`. That is the worst
    /// possible failure for a client that matches replies to buttons — the
    /// command takes effect and the answer cannot be attributed to it — so this
    /// client refuses to send one it knows will be dropped rather than
    /// discovering the loss on the reply.
    ///
    /// Dropped locally rather than truncated: a truncated id is a *different*
    /// id, and one that could collide with another button's.
    private func boundedRequestID(_ requestID: String?) -> String? {
        guard let requestID, !requestID.isEmpty else { return nil }
        guard requestID.count <= ExperimentalCVContract.requestIDMaxLength else {
            log(
                "cv_lab request_id dropped locally: \(requestID.count) characters, "
                    + "the Tower echoes at most \(ExperimentalCVContract.requestIDMaxLength) "
                    + "and drops longer ones without refusing the command"
            )
            return nil
        }
        return requestID
    }

    /// Shared send path for the three result-channel messages.
    ///
    /// Kept as a named wrapper rather than folded into
    /// `sendUnescalatedMessage`: "these three are the result channel" is a fact
    /// worth keeping visible now that a fourth kind of message — the CV Lab's
    /// commands, which are **not** on the result channel — shares the same send
    /// discipline.
    private func sendResultMessage(_ object: [String: Any], label: String) {
        sendUnescalatedMessage(object, label: label)
    }

    /// Sends a message whose failure must not cost the frame path.
    ///
    /// **A send failure here is logged and not escalated**, which is the one
    /// way this differs from `sendLifecycleMarker`. A lifecycle marker defines
    /// a frame bracket and losing one corrupts the counts on both sides, so
    /// that path fails the connection; a subscribe is a request for a report
    /// and a CV Lab command is a request about what to compute, and tearing
    /// down the socket the camera is streaming over because either did not land
    /// would let a control surface do the one thing the contract promises it
    /// cannot — affect the frame path. If the socket really is gone, the
    /// receive loop notices it on its own terms.
    ///
    /// Note that this sits **outside** the `#if DEBUG` block that covers the
    /// frame path, so commands can be sent from a Release build. That is not an
    /// oversight: a Release build has no camera and will get no `frame_result`,
    /// but nothing about the control plane depends on frames, and a Tower
    /// driven from a phone that is not the one streaming is a real bench
    /// configuration.
    private func sendUnescalatedMessage(_ object: [String: Any], label: String) {
        guard status == .online, let task = webSocketTask else {
            log("\(label) not sent — Tower not online (status=\(status))")
            return
        }
        guard
            let jsonData = try? JSONSerialization.data(withJSONObject: object),
            let jsonText = String(data: jsonData, encoding: .utf8)
        else {
            log("\(label) failed to serialize JSON payload")
            return
        }
        task.send(.string(jsonText)) { [weak self] error in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let error {
                    self.log("\(label) send failed (not escalated): \(error.localizedDescription)")
                } else {
                    self.log("\(label) sent")
                }
            }
        }
    }

    /// The bound on each half of the opening handshake — the connect-and-ping
    /// leg, and the pong. See `validateConnection`.
    private static let defaultHandshakeLegTimeout = 6
    /// Injectable so a test can prove the bound exists without spending six
    /// seconds per leg doing it.
    private let handshakeLegTimeout: Int
    /// Bounds one whole connection attempt. See `armHandshakeWatchdog`.
    private var handshakeWatchdog: Task<Void, Never>?

    /// Sends one ping and validates the pong within a bounded timeout. On
    /// success, hands off to the continuous receive loop.
    private func validateConnection(task: URLSessionWebSocketTask) async {
        do {
            let pingPayload = try JSONSerialization.data(withJSONObject: ["type": "ping"])
            guard let pingText = String(data: pingPayload, encoding: .utf8) else {
                fail("Could not encode ping payload", task: task)
                return
            }

            // Leg timings, because a reconnect's cost was guessed once and the
            // guess was wrong. A 9-second outage on the walk was attributed to
            // "the reconnect"; decomposing the console showed ~8.5 s of it was
            // spent in `.connecting` — i.e. a brand-new socket could not
            // complete its upgrade either — which means the teardown decision
            // cost well under a second and something about the path or the peer
            // cost the rest. Three hypotheses fit that (a new flow paying SYN
            // retransmission, a Tower event loop blocked in reconstruction, a
            // genuinely dead link) and they imply different fixes, so the next
            // walk must not have to guess again.
            let handshakeBegan = MonotonicClock.now

            // NOT wrapped in `withTimeout`, deliberately — `armHandshakeWatchdog`
            // is what bounds this.
            //
            // This `send` cannot return until TCP connect and the HTTP Upgrade
            // have completed, so it *is* the connect leg, and it had no
            // deadline at all. Wrapping it in `withTimeout` was tried first and
            // **does not work**: a throwing task group must await its remaining
            // child before it can propagate the sleeper's error, and that child
            // is the call that is stuck. Measured rather than reasoned —
            // against a listener that accepts TCP and never upgrades, a
            // one-second `withTimeout` here left the client `.connecting` for
            // the full twelve seconds the test was willing to wait.
            //
            // The same caveat applies to the pong's `withTimeout` below. It is
            // kept because it does bound the ordinary case; it is simply not
            // the guarantee. Cancelling the socket is.
            try await task.send(.string(pingText))
            let connectMs = (MonotonicClock.now - handshakeBegan) * 1000
            log("ping sent (connect+upgrade \(Int(connectMs)) ms): \(pingText)")

            let message = try await withTimeout(seconds: handshakeLegTimeout) {
                try await task.receive()
            }
            log("message received: \(message)")

            guard
                case .string(let text) = message,
                let data = text.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
                json["type"] == "pong"
            else {
                fail("Unexpected/malformed response from Tower", task: task)
                return
            }

            let handshakeMs = (MonotonicClock.now - handshakeBegan) * 1000
            log(
                "pong validated (connect+upgrade \(Int(connectMs)) ms,"
                    + " pong \(Int(handshakeMs - connectMs)) ms,"
                    + " handshake total \(Int(handshakeMs)) ms)"
            )
            guard !Task.isCancelled, isCurrent(task) else { return }
            // Deliberately does *not* refill the reconnect budget yet — see
            // `becameOnlineAt`. Reaching `.online` is not evidence of a working
            // connection; staying there is.
            becameOnlineAt = MonotonicClock.now
            // The handshake is done; its bound has nothing left to bound.
            handshakeWatchdog?.cancel()
            handshakeWatchdog = nil
            status = .online
            startReceiveLoop(task: task)
            // After the pong, never before. The Tower never speaks first, so
            // nothing could have arrived early — but asking before validating
            // would mean reading our own reply into the handshake, which is
            // the failure the contract warns about. The receive loop is
            // already running, so the answer has somewhere to land.
            requestCartridgeDeclaration()
        } catch is CancellationError {
            // disconnect() was called mid-validation; state already handled there.
        } catch {
            fail("Connection failed: \(error.localizedDescription)", task: task)
        }
    }

    private func startReceiveLoop(task: URLSessionWebSocketTask) {
        receiveTask?.cancel()
        receiveTask = Task { [weak self] in
            await self?.receiveLoop(task: task)
        }
    }

    /// Runs for the lifetime of one connection, continuously draining
    /// inbound messages (chiefly `frame_result`). A `receive()` failure is
    /// the definitive signal that the connection is gone, so it's the one
    /// place (alongside the delegate close callback) responsible for moving
    /// `status` off `.online` truthfully instead of leaving it stale.
    private func receiveLoop(task: URLSessionWebSocketTask) async {
        log("receive loop started")
        while !Task.isCancelled {
            do {
                let message = try await task.receive()
                guard isCurrent(task) else {
                    log("receive loop stopped (superseded connection)")
                    return
                }
                handleInboundMessage(message)
            } catch {
                guard isCurrent(task) else {
                    log("receive loop stopped (superseded connection)")
                    return
                }
                log("receive failed: \(error.localizedDescription)")
                fail("Connection lost: \(error.localizedDescription)", task: task)
                return
            }
        }
        log("receive loop stopped (cancelled)")
    }

    private func handleInboundMessage(_ message: URLSessionWebSocketTask.Message) {
        guard case .string(let text) = message else {
            log("unknown message type: non-text frame received")
            return
        }
        guard
            let data = text.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = json["type"] as? String
        else {
            log("unknown message type: undecodable payload")
            return
        }

        switch type {
        case "frame_result":
            // Decimated on the same 1-in-`frameLogStride` cadence as the send
            // path. This arrives once per delivered frame, so at the target
            // rate an unguarded line here is ~12 prints a second — and the
            // string builds two `Optional.map` allocations before `print` even
            // takes its lock. `metrics.frameResults` is the real count.
            let seq = json["seq"] as? Int
            let meanIntensity = json["mean_intensity"] as? Double
            let processingMs = json["processing_ms"] as? Double
            let resultValue = json["result_value"] as? Double
            let resultLabel = json["result_label"] as? String
            // `?? [:]` rather than an optional: the Tower sends `stage_ms`
            // unconditionally but may send it empty, and `metrics` is omitted
            // when empty. Both mean "no stages/measurements to report", so
            // collapsing absent and empty here loses nothing.
            let stageMs = json["stage_ms"] as? [String: Double] ?? [:]
            // Not `metrics`: that name is this client's `SenderMetrics`, and
            // shadowing it here silently rebinds every use below.
            let extraMetrics = json["metrics"] as? [String: Double] ?? [:]
            // Additive, and omitted entirely by a Tower that runs no Lab. An
            // `if let` rather than `flatMap(TowerFrameResultProvenance.init)`
            // for the isolation reason `TowerCartridgeDeclaration.init` gives.
            var cvLab: TowerFrameResultProvenance?
            if let rawCVLab = json["cv_lab"] as? [String: Any] {
                cvLab = TowerFrameResultProvenance(json: rawCVLab)
            }

            // >>> The staleness gate. <<<
            //
            // A result belonging to a run this client is not watching is
            // discarded here and reaches nothing below — not the published
            // reading, not the reply counter, not the metrics. It is the whole
            // of the rule the contract states in one line, and the reason it
            // sits at the top of this case rather than at the point of display
            // is that there is more than one consumer downstream and a gate per
            // consumer is a gate somebody forgets.
            //
            // `watchedCVLabRunID == nil` means no status has ever been read, so
            // there is no run being watched and nothing to contradict — every
            // result is passed through. A reply with no `cv_lab` block at all
            // is passed through for the same reason: a Tower that attaches no
            // provenance has made no claim about which run this belongs to, and
            // discarding it would drop every result from a Tower running no Lab.
            if let watched = watchedCVLabRunID, let runID = cvLab?.runID, runID != watched {
                staleFrameResultCount += 1
                log(
                    "frame_result discarded: run \(runID) is not the watched run "
                        + "\(watched) (stale results so far: \(staleFrameResultCount))"
                )
                return
            }

            resultLogCounter += 1
            if resultLogCounter % Self.frameLogStride == 1 {
                log(
                    "frame_result received: seq=\(seq.map(String.init) ?? "?")"
                        + " mean_intensity=\(meanIntensity.map { String($0) } ?? "?")"
                        + " processing_ms=\(processingMs.map { String($0) } ?? "?")"
                )
            }
            #if DEBUG
            frameResultCount += 1
            // Decoding moved above the log gate so the value is kept for every
            // reply rather than only for the one-in-twelve that gets logged —
            // the counters were always exact and the surfaced value has to be
            // too. It is three optional casts on an existing dictionary, at the
            // reply rate; the publish that follows is the only real cost, and
            // it is the same order as `frameResultCount` next to it.
            latestFrameResult = TowerFrameResult(
                sequence: seq,
                meanIntensity: meanIntensity,
                processingMs: processingMs,
                resultValue: resultValue,
                resultLabel: resultLabel,
                stageMs: stageMs,
                metrics: extraMetrics,
                cvLab: cvLab
            )
            #endif
            // A result cleared the gate, so whatever refusal was last on screen
            // has been answered. Cleared here rather than left to age: "the Lab
            // is paused" beside a fresh reading is two claims that cannot both
            // be current.
            latestFrameRefusal = nil
            metrics.recordFrameResult()

        case "frame_error":
            // The case that did not exist. Every `frame_error` the Tower has
            // ever sent fell through to `default:` below and was logged as an
            // unknown message type — so the Lab's six refusal codes, which are
            // the answer to "I pressed Start and nothing happened", were
            // arriving and being discarded.
            //
            // **Refusals are not failures**, and nothing here treats them as
            // such: no `fail()`, no send-window change, no stream-bracket
            // change, and `metrics` is not told a frame failed. The Tower
            // counts these under `frames_rejected` and deliberately not under
            // `frame_processing_errors`, for the reason it gives in one line —
            // a Lab paused for five minutes has not failed hundreds of times —
            // and a client that counted them as errors would reproduce exactly
            // the number the Tower fixed.
            guard let refusal = TowerFrameRefusal(json: json) else {
                log("frame_error could not be decoded")
                return
            }
            // Not decimated. These arrive at the refusal rate, which is the
            // sender's ~0.8 frames per second and not the 12 Hz reply rate, and
            // each one is a state change worth seeing in full.
            log(
                "frame_error: seq=\(refusal.sequence.map(String.init) ?? "?")"
                    + " reason=\(refusal.reason) — \(refusal.message)"
            )
            latestFrameRefusal = refusal
            cvLabControlEvents.send(.frameRefused(refusal))

        // MARK: CV Lab control plane
        //
        // Two more cases that did not exist. Both are additive, neither can
        // affect the frame path, and both pass the status document through
        // undecoded — this file owns the transport, the cartridge owns the
        // contract.

        case "cv_lab_status":
            guard let reply = CVLabControlReply(json: json) else {
                log("cv_lab_status could not be decoded")
                return
            }
            log(
                "cv_lab_status: state=\(reply.lifecycleState ?? "?")"
                    + " run=\(reply.runID ?? "none")"
                    // The field that separates an answer from a push. Logged
                    // because reading a pushed heartbeat as the answer to a
                    // command is the mistake this field exists to prevent, and
                    // a log that does not distinguish them cannot show it.
                    + " accepted_command=\(reply.acceptedCommand ?? "—")"
                    + " request_id=\(reply.requestID ?? "—")"
            )
            watchCVLabRun(reply.runID)
            cvLabControlEvents.send(.status(reply))

        case "cv_lab_error":
            guard let refusal = CVLabControlRefusal(json: json) else {
                log("cv_lab_error could not be decoded")
                return
            }
            // Every one of the eight reasons means the request did not take
            // effect, so nothing is applied here — not even optimistically. The
            // refusal carries the unchanged document and the run watch is
            // updated from it, which is the same path an accepted command
            // takes: state comes from the document, never from the command.
            log(
                "cv_lab_error: \(refusal.reason) for \(refusal.command ?? "?")"
                    + " — \(refusal.message)"
            )
            if let status = refusal.status {
                watchCVLabRun((status["lifecycle"] as? [String: Any])?["run_id"] as? String)
            }
            cvLabControlEvents.send(.refused(refusal))

        // MARK: Result channel
        //
        // Every case below is additive and none of them can affect the frame
        // path: they decode, publish, and return. Nothing here touches
        // `status`, the send window, or the stream bracket — which is the iOS
        // half of the guarantee the contract makes on the Tower side.

        case "cartridges":
            let declaration = TowerCartridgeDeclaration(json: json)
            log(
                "cartridges declared: "
                    + declaration.offers
                    .map { "\($0.cartridge)/\($0.resultType) available=\($0.available)" }
                    .joined(separator: ", ")
            )
            cartridgeDeclaration = declaration
            resultEvents.send(.declaration(declaration))

        case "result_subscribed":
            guard let ack = CartridgeSubscriptionAck(json: json) else {
                log("result_subscribed could not be decoded")
                return
            }
            log("result_subscribed: \(ack.subscriptionID) \(ack.cartridge)/\(ack.resultType)")
            resultEvents.send(.subscribed(ack))

        case "result_unsubscribed":
            guard let id = json["subscription_id"] as? String else { return }
            log("result_unsubscribed: \(id)")
            resultEvents.send(.unsubscribed(subscriptionID: id))

        case "cartridge_result":
            guard let envelope = CartridgeResultEnvelope(json: json) else {
                log("cartridge_result could not be decoded")
                return
            }
            // Decimated like `frame_result`, and for a weaker reason: this
            // arrives at most twice a second. One line per change rather than
            // one per heartbeat is still the useful reading.
            if envelope.revisionChanged {
                log(
                    "cartridge_result: \(envelope.cartridge)/\(envelope.resultType)"
                        + " seq=\(envelope.sequence.map(String.init) ?? "?")"
                        + " revision=\(envelope.revision ?? "?")"
                        + " coalesced=\(envelope.coalesced)"
                )
            }
            resultEvents.send(.result(envelope))

        case "result_error":
            guard let error = CartridgeResultError(json: json) else {
                log("result_error could not be decoded")
                return
            }
            log("result_error: \(error.reason) — \(error.message)")
            resultEvents.send(.failed(error))

        case "protocol_error":
            // The Tower telling us it does not implement something we sent.
            // Additive on its side and non-fatal on ours: previously an
            // unrecognised message produced only a server-side log line, so
            // "not implemented" and "lost in flight" were indistinguishable
            // from here.
            let messageType = json["message_type"].map { String(describing: $0) } ?? "nil"
            log("protocol_error from Tower: \(json["reason"] as? String ?? "?") for \(messageType)")

        default:
            log("unknown message type: \(type)")
        }
    }

    /// True only if `task` is still the socket this client currently owns —
    /// used to ignore work (receive-loop errors, validation results) that
    /// belongs to a connection already superseded by a later connect()/
    /// disconnect(), so a stale callback can never clobber current state.
    private func isCurrent(_ task: URLSessionWebSocketTask) -> Bool {
        guard let current = webSocketTask else { return false }
        return current === task
    }

    /// Fails only if `task` is still current; otherwise the failure belongs
    /// to an already-superseded connection and is logged, not acted on.
    private func fail(_ message: String, task: URLSessionWebSocketTask) {
        guard isCurrent(task) else {
            log("ignoring stale failure (superseded connection): \(message)")
            return
        }
        fail(message)
    }

    private func fail(_ message: String) {
        guard status != .offline else { return }
        log("error: \(message)")
        teardownConnection(cancelWith: .abnormalClosure)
        status = .failed(message)
        // After the status is settled, so an observer that reacts to `.failed`
        // sees a consistent client. Documented URLSession behaviour is that
        // one send error fails *all* outstanding work on the task, so the
        // sibling completions arriving next are already stale by `isCurrent`
        // and cannot schedule a second reconnect.
        scheduleReconnect()
    }

    private func teardownConnection(cancelWith closeCode: URLSessionWebSocketTask.CloseCode) {
        handshakeWatchdog?.cancel()
        handshakeWatchdog = nil
        validationTask?.cancel()
        validationTask = nil
        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: closeCode, reason: nil)
        webSocketTask = nil
        // `URLSession` retains its delegate — this object — until it is
        // invalidated. Dropping the reference is not enough: without this, the
        // session, its delegate queue and this client all outlive every
        // teardown, once per connect. That was survivable when connecting was
        // something the user did by tapping; it is not now that `autoReconnect`
        // re-opens on a schedule whose budget refills after every 30s of
        // healthy connection, which is once per drop on exactly the flaky link
        // this client was built for.
        //
        // `invalidateAndCancel` rather than `finishTasksAndInvalidate`: the
        // only task was cancelled on the line above, so there is nothing left
        // to finish, and waiting would keep the session alive past the point
        // this method promises it is gone. Late delegate callbacks from the
        // invalidated session are already ignored — `webSocketTask` is nil by
        // then, so `handleDelegateClose`'s identity guard drops them.
        session?.invalidateAndCancel()
        session = nil

        // The send window belongs to one socket. Any completion handlers still
        // pending for the old task are ignored by their `isCurrent` guard, so
        // clearing here is the only thing that reopens the window for the next
        // connection — otherwise a dropped connection would permanently leak
        // window slots and eventually stop sending altogether. `SendWindow`
        // does not rewind its token counter, so those late completions cannot
        // release a slot belonging to the *next* connection either.
        sendWindow.reset()
        // Belongs to the socket that is going away: the next connection's
        // first frame must not be judged against the old one's pulse.
        lastSendFrameAt = nil

        #if DEBUG
        // `isStreamingToTower` means "a stream_start has been sent and not yet
        // matched by a stream_stop". No stream_start survives a socket, so
        // leaving this true across a teardown would be a lie, and would let
        // frames flow to a Tower that never received a start for the
        // connection they arrive on. Set directly rather than via
        // `sendStreamStop()`: there is no socket left to send on.
        isStreamingToTower = false
        // Scoped to the socket that carried it, for the same reason as the
        // line above. `HomeWorkspaceView` renders this under the caption
        // "latest Tower reply", and a reading from a connection that is gone
        // is not the latest anything. It was cleared only by the stream
        // bracket, so a socket that dropped mid-capture left the dead
        // connection's number on screen for the whole outage — and forever
        // once the reconnect budget is spent, because no `stream_stop` is
        // ever sent for a socket that is not there.
        latestFrameResult = nil
        #endif
        // The third of the three clears, and the only one that runs in a
        // Release build. A run id is a fact about the Tower on the other end of
        // *this* socket, and the socket is what just went away: the next one
        // may reach a different Tower, or the same one restarted, and either
        // way the run it names has to be learned again from a document rather
        // than assumed to have survived. Holding a run id across a teardown
        // would gate the next connection's replies against the last one's run
        // and discard every one of them until a status arrived.
        clearCVLabRunWatch()
    }

    // MARK: - What the Tower says about itself

    /// The reader for `GET /health`. A `var` so a test can point it at a
    /// stubbed `URLSession`; nothing in the app ever reassigns it.
    var healthClient = TowerHealthHTTPClient()

    /// What this app has been told about the Tower's own state, including
    /// whether its dataset recorder is armed and writing.
    ///
    /// Starts at `.notFetched`, which is a real state and not a stand-in for
    /// "nothing is happening" — see `TowerHealthState`.
    @Published private(set) var healthState: TowerHealthState = .notFetched

    /// The one read in flight, or `nil`. Held so a second tap on Refresh
    /// cannot start a second request that would race the first into the same
    /// property.
    private var healthTask: Task<Void, Never>?

    /// Asks the Tower how it is, once.
    ///
    /// ## Why this is a button and not a timer
    ///
    /// Nothing in the app makes a decision from this; it exists so a person
    /// can *ask*, and the answer they get is the answer from the moment they
    /// asked, stamped with that moment. A poll would spend battery and a
    /// Tailscale round trip every few seconds to keep a developer screen warm
    /// that is usually not open, and it would add a timer to manage across
    /// backgrounding — a resource with no reader.
    ///
    /// ## Why it never throws
    ///
    /// An unreachable Tower and an unreadable answer are both things the
    /// screen has to be able to say, so they are states rather than errors,
    /// exactly as `TowerObjectMemoryClient.ask` treats the same two.
    func refreshHealth() {
        // A second tap while one is in flight is dropped rather than queued:
        // they are the same button, and two answers racing into one property
        // is a worse outcome than one ignored tap. Same rule as
        // `TowerObjectMemoryClient.isAsking`.
        guard healthTask == nil else { return }
        healthState = .fetching

        // Copied out before the hop so the request itself never has to touch
        // this actor, and captured directly rather than through `self`.
        let client = healthClient
        healthTask = Task { [weak self] in
            let outcome: Result<TowerHealth, TowerHealthFetchError>
            do {
                outcome = .success(try await client.health())
            } catch let error as TowerHealthFetchError {
                outcome = .failure(error)
            } catch {
                // The only honest attribution available without knowing where
                // it came from.
                outcome = .failure(.transport(error.localizedDescription))
            }

            // Resolved *after* the await, so this client is only held for the
            // hop back and not for the lifetime of the request. The body is a
            // single bounded await rather than a loop, which is what makes
            // that safe here — a `guard let self` outside an unbounded `for
            // await` is the shape that promotes a weak capture to a strong one
            // for the task's whole life, and three of those were removed from
            // this file.
            guard let self else { return }
            self.healthTask = nil
            // A cancelled read has no answer to report — but it cannot simply
            // return, because `.fetching` was set *before* the await and the
            // previous answer is already gone. Leaving `.fetching` standing
            // would leave `DeveloperToolsView.isCheckingHealth` true forever,
            // with the button reading "Checking…" and disabled while nothing is
            // in flight: a control that has quietly stopped working, which is
            // exactly the failure this screen exists to make visible.
            //
            // `.notFetched` is what is true afterwards — the question was
            // abandoned, not answered, and nobody has asked since. Guarded on
            // `.fetching` so a state somebody else has since written is not
            // stamped over.
            if Task.isCancelled {
                if case .fetching = self.healthState { self.healthState = .notFetched }
                return
            }

            let at = Date()
            switch outcome {
            case .success(let health):
                self.healthState = .fetched(health, at: at)
            case .failure(let error):
                self.healthState = .failed(error, at: at)
            }
        }
    }

    private func log(_ message: String) {
        #if DEBUG
        print("[Glasses][Tower] \(message)")
        #endif
    }
}

extension TowerClient: URLSessionWebSocketDelegate {
    /// Independent, socket-level signal that the Tower (or the OS) closed
    /// the connection — a second detection path alongside the receive
    /// loop's error case, for whichever one notices first. Hops to the main
    /// actor before touching any state, and only acts if the closed task is
    /// still the one this client currently owns, so it can never race a
    /// receive-loop failure (or a newer connection) into a conflicting
    /// status update.
    nonisolated func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        let closedTaskID = ObjectIdentifier(webSocketTask)
        let codeValue = closeCode.rawValue
        let reasonText = reason.flatMap { String(data: $0, encoding: .utf8) }
        Task { @MainActor [weak self] in
            self?.handleDelegateClose(closedTaskID: closedTaskID, codeValue: codeValue, reason: reasonText)
        }
    }

    private func handleDelegateClose(closedTaskID: ObjectIdentifier, codeValue: Int, reason: String?) {
        guard let current = webSocketTask, ObjectIdentifier(current) == closedTaskID else {
            log("delegate close ignored (stale/superseded connection), code=\(codeValue)")
            return
        }
        log("delegate close: code=\(codeValue) reason=\(reason ?? "none")")
        fail("Tower closed the connection (code \(codeValue))")
    }

    #if DEBUG
    /// Test-only hook: invokes the real `didCloseWith` delegate callback for
    /// the current connection, exercising the exact production code path
    /// without requiring a real socket-level close frame from the network.
    func simulateDelegateCloseForTesting(code: URLSessionWebSocketTask.CloseCode) {
        guard let task = webSocketTask else { return }
        urlSession(session ?? URLSession(configuration: .default), webSocketTask: task, didCloseWith: code, reason: nil)
    }
    #endif
}

/// Races an async operation against a timeout, since
/// `URLSessionWebSocketTask.receive()` has no built-in timeout.
private func withTimeout<T: Sendable>(
    seconds: Int,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await operation() }
        group.addTask {
            try await Task.sleep(for: .seconds(seconds))
            throw TowerClientError.timedOut
        }
        guard let result = try await group.next() else {
            throw TowerClientError.timedOut
        }
        group.cancelAll()
        return result
    }
}

private enum TowerClientError: LocalizedError {
    case timedOut

    var errorDescription: String? {
        switch self {
        case .timedOut: return "Timed out waiting for pong"
        }
    }
}
