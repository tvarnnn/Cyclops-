//
//  ExperimentalCVModel.swift
//  Glasses
//

import Foundation

/// The Experimental CV Lab's domain types, decoded from the Tower's status
/// document.
///
/// **This file used to open by saying nothing in it was a Tower protocol.** It
/// is now the opposite, and the change is the whole point of this work: the
/// Tower declares `experimental_cv.status/2026-08-27`, enumerates its
/// experiments with stable ids, and attaches provenance to every figure. So
/// these types are read off that document rather than invented ahead of it, and
/// `ExperimentalCVContract` names the three agreements they answer to.
///
/// What has **not** changed is where the list of experiments comes from. This
/// file still hard-codes no experiment, no algorithm and no metric name.
/// `docs/modules/EXPERIMENTAL-CV.md` calls its candidate list "intentionally
/// broad"; picking any subset here would make the iOS app the place the
/// experiment list is decided, which is exactly backwards. The Tower names its
/// experiments, names its metrics, and names their units; iOS displays those
/// names and never matches on them.
///
/// What iOS owns is the discipline the module spec imposes on results:
///
/// > Experiment output (detections, depth estimates, classifications, tracked
/// > poses, etc.) is model inference, not a measured sensor fact, unless the
/// > experiment specifically validates against a ground-truth reference.
/// > Results/logs must distinguish the two.
///
/// That one is enforced in the type system — see `CVMetric.provenance`, which
/// is not optional and has no default.
///
/// The other rule that used to be enforced here — *"avoid declaring an approach
/// 'better' without a measurement"* — is now enforced by **deletion**. See
/// `CVMetric`.

// MARK: - Experiment identity

/// One experiment the Tower says it can run.
///
/// The Tower is the registry. iOS holds no list of its own: every field here is
/// read off one entry of `status.available`, which the Tower sorts by `id` and
/// this app displays in the order given.
///
/// `id`, `name` and `summary` are the three that are always drawn. The rest is
/// additive and a client that ignored all of it would still have a working
/// picker — but two of them are load-bearing rather than decorative:
/// `isAvailable`/`unavailableReason`, because starting an unavailable
/// experiment is refused in advance and a picker that offers it is offering a
/// refusal; and `provenance`, because it says in advance whether this
/// experiment's numbers will be measurements or model output.
struct CVExperiment: Equatable, Identifiable, Sendable {
    /// The Tower's identifier. Opaque: compared for equality, never parsed.
    let id: String
    /// The Tower's own name for it.
    let name: String
    /// What it does, if the Tower says. `nil` renders as no subtitle rather
    /// than as an invented one.
    var summary: String?
    /// Whether this experiment measures or infers. Declared **before** it runs,
    /// which is what lets the picker say so at the moment of choosing rather
    /// than only once numbers are on screen.
    var provenance: ObservationProvenance
    /// What this will measure, and in what. A `nil` unit means the quantity
    /// genuinely has none and is rendered bare — depth is the case, because
    /// MiDaS-small emits relative inverse depth on an arbitrary scale.
    var headlineLabel: String?
    var headlineUnit: String?
    /// Carries state across frames, so its first frame is not like its
    /// hundredth. Worth saying where someone is about to read a two-frame run.
    var isStateful: Bool
    /// Needs the optional `[ml]` extra. A start may take a hundred times longer
    /// than a cheap experiment's — see `ExperimentalCVContract.armTimeoutSeconds`.
    var requiresModel: Bool
    /// `opencv` or `torch`, in the Tower's words. Displayed, never switched on.
    var backend: String?
    /// Whether **this Tower** can run it, checked per experiment: `depth` needs
    /// `torch` and `timm`, `object_detection` needs `torch` and `torchvision`.
    ///
    /// What this cannot check is the network. `depth` fetches its weights
    /// through `torch.hub` on first use, so an offline Tower reports it
    /// available, accepts the start, and then goes `failed` with the reason.
    /// That is why a failed interactive start is recoverable.
    var isAvailable: Bool
    /// The Tower's own words for why not. Shown verbatim — only the Tower knows
    /// which module is missing.
    var unavailableReason: String?
    /// What kind of live picture this experiment can draw, or `nil` for one
    /// that draws none. Declared in the catalog so the picker can say "this
    /// one has a live view" **before** anybody commits to a two-minute model
    /// load to find out.
    ///
    /// `nil` is a real answer and not a gap: `baseline` will never have a
    /// picture, deliberately, because it is the control every other
    /// experiment's cost is read against and drawing one would roughly double
    /// what it costs.
    ///
    /// Displayed, never switched on. The Tower's vocabulary is open.
    var previewKind: String?

    init(
        id: String,
        name: String,
        summary: String? = nil,
        provenance: ObservationProvenance = .unknown,
        headlineLabel: String? = nil,
        headlineUnit: String? = nil,
        isStateful: Bool = false,
        requiresModel: Bool = false,
        backend: String? = nil,
        isAvailable: Bool = true,
        unavailableReason: String? = nil,
        previewKind: String? = nil
    ) {
        self.id = id
        self.name = name
        self.summary = summary
        self.provenance = provenance
        self.headlineLabel = headlineLabel
        self.headlineUnit = headlineUnit
        self.isStateful = isStateful
        self.requiresModel = requiresModel
        self.backend = backend
        self.isAvailable = isAvailable
        self.unavailableReason = unavailableReason
        self.previewKind = previewKind
    }

    /// One catalog entry, or `nil` when the Tower sent something that is not
    /// one. An entry with no id cannot be started and an entry with no name
    /// cannot be drawn, so neither is defaulted into existence.
    init?(json: [String: Any]) {
        guard let id = json["id"] as? String, let name = json["name"] as? String else {
            return nil
        }
        self.init(
            id: id,
            name: name,
            summary: json["summary"] as? String,
            provenance: CVWireProvenance.read(json["provenance"] as? String),
            headlineLabel: json["headline_label"] as? String,
            headlineUnit: json["headline_unit"] as? String,
            isStateful: json["stateful"] as? Bool ?? false,
            requiresModel: json["requires_model"] as? Bool ?? false,
            backend: json["backend"] as? String,
            // Defaulting to `false` on an unreadable entry, deliberately. The
            // failure of offering an experiment this Tower cannot run is a
            // person pressing a button and being refused; the failure of
            // withholding one it can is a person seeing a reason and picking
            // something else. The second is recoverable from the screen.
            isAvailable: json["available"] as? Bool ?? false,
            unavailableReason: json["unavailable_reason"] as? String,
            previewKind: json["preview_kind"] as? String
        )
    }

    /// Whether a start for this experiment would be refused before it began.
    var isStartable: Bool { isAvailable }

    /// Whether starting this would put a live picture on screen.
    ///
    /// A promise about what a preview would BE, not that one exists: whether
    /// one exists right now is `run.annotation.artifact`, which is `nil` with
    /// a reason until a frame has actually been through.
    var hasLiveView: Bool { previewKind != nil }
}

/// Reads `provenance` off any of the three places the Tower states it — a
/// catalog entry, a metric, or a `frame_result`'s `cv_lab` block.
///
/// One reader, because the Tower uses one vocabulary in all three and a second
/// copy of the mapping is how the copies come to disagree.
///
/// **`confidence` is always `null` on this wire and is not read here.** The
/// Tower has no calibrated confidence for any of these experiments and says so;
/// `.inferred(confidence: nil)` is therefore the only inference this cartridge
/// can construct, and `ObservationProvenance.caveat` already renders that as
/// *"Estimated by a model. The Tower did not report a confidence"* rather than
/// as a percentage it does not have.
nonisolated enum CVWireProvenance {
    /// - Parameter word: the wire's own string, already read off the message.
    ///   Typed as `String?` rather than `Any?` so a call site cannot hand this
    ///   a value of the wrong shape and get `.unknown` back as though the Tower
    ///   had been silent.
    static func read(_ word: String?) -> ObservationProvenance {
        switch word {
        case "measured": return .measured
        case "inferred": return .inferred(confidence: nil)
        // Rule 16: silence is not "measured". The Tower requires this field on
        // every metric and never omits it, so reaching here means a document
        // this build could not read — which is exactly what `.unknown` is for.
        default: return .unknown
        }
    }
}

// MARK: - Results

/// One number an experiment produced, aggregated across the run.
///
/// ## Why `provenance` is not optional
///
/// Because `docs/modules/EXPERIMENTAL-CV.md` requires results to distinguish
/// model inference from measured fact, and an optional field with a `nil`
/// default is a field that gets skipped at the call site. The Tower takes the
/// same position from its side — *"required, never omitted"* — so requiring it
/// here means the question "is this a measurement or a guess?" is answered by
/// whoever decodes the Tower's reply, which is the only party that can answer
/// it.
///
/// ## Why there is no `baseline`, and no better/worse verdict
///
/// There used to be. `baseline`, `higherIsBetter`, a `comparison` property and
/// a `Comparison` enum that rendered *"Better by 0.4"* / *"Worse by 0.4"* — all
/// of it written against a contract that did not exist yet, and all of it
/// **deleted** now that the contract does.
///
/// The Tower's position is unambiguous and is not a temporary gap:
///
/// > `baseline` and `higher_is_better` — **always `null`**, and this is not an
/// > omission. … the Lab holds no reference run to compare against. A
/// > comparison against nothing is the "declaring an approach 'better' without
/// > a measurement" that `EXPERIMENTAL-CV.md` rules out. Offline corpus
/// > comparison is `scripts/cv_lab_benchmark.py`, and it is not this channel.
///
/// Keeping the machinery and relying on the nulls to keep it quiet would leave
/// a verdict renderer one non-null field away from firing, in a cartridge whose
/// entire purpose is not to claim more than was measured. Deleting it means the
/// verdict has no code path at all. If a future contract carries a reference
/// run, this comes back **with** that contract.
///
/// `confidence` is deleted for the same reason and by the same rule: always
/// `null`, so `ObservationProvenance.inferred(confidence:)` is only ever
/// constructed with `nil` here.
struct CVMetric: Equatable, Identifiable, Sendable {
    /// The Tower's label. Displayed verbatim. **iOS matches on no metric name,
    /// ever.**
    let label: String
    /// The aggregate, or `nil`.
    ///
    /// **`nil` is a real answer meaning "this metric has no meaningful
    /// aggregate", and never zero.** Two ways to reach it, and
    /// `unavailableReason` says which: an `unaggregated` metric, whose value is
    /// null by construction because averaging it would be nonsense
    /// (`dominant_direction_deg` is circular — the mean of 179° and −179° is
    /// 0°, the one direction neither frame was moving in); or a `constant` that
    /// was not constant, which `varied` reports so that null is not read as
    /// "never observed".
    let value: Double?
    /// The Tower's unit string, if any. Never assumed — a metric with no unit
    /// is shown as a bare number, which is what an unlabelled quantity is.
    var unit: String?
    /// Whether the Tower measured this or a model produced it.
    let provenance: ObservationProvenance
    /// How this number was combined across frames, in the Tower's word:
    /// `rate` (mean), `count` (sum), `constant` (the value observed), or
    /// `unaggregated` (nothing — `value` is `nil`). Displayed, never switched
    /// on, because an experiment may declare a fifth.
    var aggregation: String?
    /// How many frames contributed. `nil` when the Tower did not say.
    var frames: Int?
    /// The experiment's single most important number. The Tower puts it first
    /// in the list and this app preserves that order rather than re-sorting.
    var isHeadline: Bool
    /// A `constant` that was not constant. Its `value` is `nil` and this is
    /// why — so a null is not read as "never observed".
    var varied: Bool
    /// The most recent frame's value, and the range **this run** has observed.
    /// All three are `nil` for anything that is not a `rate`.
    ///
    /// ## What these are for, and the one thing they are not
    ///
    /// `sharpness_laplacian_var: 483.068` says nothing to anybody. The same
    /// number beside *"this run has seen 79 to 1309"* says the frame is
    /// middling — from **this** camera in **this** room, rather than from a
    /// constant somebody picked once. That is the whole of what they buy.
    ///
    /// **They are not a verdict and must never be rendered as one.** The
    /// standard reference for variance-of-Laplacian is explicit that its
    /// threshold has to be tuned per dataset; the Tower has one physical run
    /// to tune against, which is none; and a row reading "Sharpness: Good"
    /// derived from these would be this app inventing the calibration both
    /// sides refuse to invent. A run four frames long has a range four frames
    /// wide, and the copy has to survive that case as well as a long one.
    var latest: Double?
    var observedMin: Double?
    var observedMax: Double?

    var id: String { label }

    init(
        label: String,
        value: Double?,
        unit: String? = nil,
        provenance: ObservationProvenance,
        aggregation: String? = nil,
        frames: Int? = nil,
        isHeadline: Bool = false,
        varied: Bool = false,
        latest: Double? = nil,
        observedMin: Double? = nil,
        observedMax: Double? = nil
    ) {
        self.label = label
        self.value = value
        self.unit = unit
        self.provenance = provenance
        self.aggregation = aggregation
        self.frames = frames
        self.isHeadline = isHeadline
        self.varied = varied
        self.latest = latest
        self.observedMin = observedMin
        self.observedMax = observedMax
    }

    init?(json: [String: Any]) {
        guard let label = json["label"] as? String else { return nil }
        self.init(
            label: label,
            // `as? Double` and nothing else: a JSON null decodes to `NSNull`,
            // which is not a `Double`, so a null value arrives here as `nil`
            // without a special case. That is the behaviour the contract asks
            // for — null is "no meaningful aggregate", not zero.
            value: json["value"] as? Double,
            unit: json["unit"] as? String,
            provenance: CVWireProvenance.read(json["provenance"] as? String),
            aggregation: json["aggregation"] as? String,
            frames: json["frames"] as? Int,
            isHeadline: json["headline"] as? Bool ?? false,
            varied: json["varied"] as? Bool ?? false,
            latest: json["latest"] as? Double,
            observedMin: json["observed_min"] as? Double,
            observedMax: json["observed_max"] as? Double
        )
    }

    /// Where the newest frame sits inside the range this run has seen, as a
    /// sentence. `nil` unless there is a range with width to it.
    ///
    /// Deliberately says *"in this run's range"* every time, in those words.
    /// Drop the qualifier and the sentence becomes a claim about the world:
    /// this is a comparison against four frames of the wearer's own living
    /// room, not against anything calibrated.
    var rangeNote: String? {
        guard let latest, let low = observedMin, let high = observedMax,
            high > low, let frames, frames > 1
        else { return nil }
        let position = (latest - low) / (high - low)
        let placement: String
        switch position {
        case ..<0.2: placement = "near the low end of"
        case ..<0.4: placement = "below the middle of"
        case ..<0.6: placement = "around the middle of"
        case ..<0.8: placement = "above the middle of"
        default: placement = "near the high end of"
        }
        return """
            Latest \(Self.format(latest)) — \(placement) this run's range so far \
            (\(Self.format(low)) to \(Self.format(high)) over \(frames) frames).
            """
    }

    /// Formatted for a row, or `nil` when there is no number to show.
    ///
    /// Unit omitted entirely when absent, rather than substituted.
    var displayValue: String? {
        guard let value else { return nil }
        let formatted = Self.format(value)
        guard let unit, !unit.isEmpty else { return formatted }
        return "\(formatted) \(unit)"
    }

    /// Why there is no number, when there is none. `nil` when there is one.
    ///
    /// A row that simply vanished would be worse than a row that says why: the
    /// Tower reported this metric on purpose, and "the experiment emitted
    /// nothing" and "the aggregate is meaningless" are different facts about
    /// the run.
    var unavailableReason: String? {
        guard value == nil else { return nil }
        if varied {
            return "Reported as constant, but it varied during the run, so no single value describes it."
        }
        if aggregation == "unaggregated" {
            return "No meaningful average across frames, so the Tower reports none."
        }
        return "The Tower reported no value for this metric."
    }

    /// Three significant-ish decimals, trimmed. Chosen so a probability and a
    /// millisecond timing both read sensibly without the Tower having to say
    /// which it sent.
    static func format(_ value: Double) -> String {
        if value == value.rounded() && abs(value) < 1e9 {
            return String(Int(value))
        }
        return String(format: "%.3f", value)
    }
}

/// What an experiment drew on top of a frame, and whether this app may show it.
///
/// The annotation *geometry* is deliberately absent — boxes, masks, keypoints
/// and flow fields each need a schema the Tower has not defined, and a wrong
/// coordinate convention renders confidently in the wrong place. What is here
/// is the count the Tower reports and the state of the rendered image it
/// produced, which is enough for the workspace to be honest about both.
///
/// `artifact` used to be `null` in this contract, always, because no
/// redaction-state vocabulary was shared between the two sides and no artifact
/// fetch contract existed on either. Both now do — landed together, in one
/// change, with `EXPERIMENTAL-CV-LAB.md` §5 written down — so this carries a
/// real descriptor whenever the running experiment has a picture and the Tower
/// is serving one.
///
/// **What did not change is the strictness.** The descriptor names a path, not
/// an image: no bytes ride the result channel, and the treatment is stated on
/// every one of them. An experiment still gets no privacy exemption for being a
/// debug surface, `artifactUnavailableReason` still carries the Tower's own
/// sentence when there is nothing, and the two are mutually exclusive — a
/// document with neither is a Tower this build cannot read.
struct CVAnnotationReport: Equatable, Sendable {
    /// How many things the experiment marked. `nil` when not reported; `0` is a
    /// real result meaning "found nothing", and the two must not merge.
    var count: Int?
    /// The Tower's words for why there is no count — most often "this
    /// experiment reports no annotation count", which is a different statement
    /// from "it found none".
    var countUnavailableReason: String?
    /// The state machine around fetching the live view. `.absent` when this
    /// experiment has no picture, `.notFetched(treatment)` when it does —
    /// `CVLivePreviewLoader` owns the fetch itself and the bytes never live
    /// here, because this type is copied into every status update and an image
    /// on it would be an image copied at the status channel's rate.
    var artifact: VisualArtifactState
    /// The live preview's descriptor, when there is one. `nil` when the Tower
    /// offered none, or offered one this build could not read.
    var preview: CVLivePreview?
    /// The Tower's own explanation for the absence above. Mutually exclusive
    /// with `preview`.
    var artifactUnavailableReason: String?

    init(
        count: Int? = nil,
        countUnavailableReason: String? = nil,
        artifact: VisualArtifactState = .absent,
        preview: CVLivePreview? = nil,
        artifactUnavailableReason: String? = nil
    ) {
        self.count = count
        self.countUnavailableReason = countUnavailableReason
        self.artifact = artifact
        self.preview = preview
        self.artifactUnavailableReason = artifactUnavailableReason
    }

    init(json: [String: Any]) {
        // Read once, so the descriptor and the state machine cannot disagree
        // about whether there is a picture.
        let preview = (json["artifact"] as? [String: Any]).flatMap(CVLivePreview.init)
        self.init(
            count: json["count"] as? Int,
            countUnavailableReason: json["count_unavailable_reason"] as? String,
            // `.notFetched`, never `.available`: this type carries no bytes and
            // never will. It is copied into every status update, and an image
            // on it would be an image copied at the status channel's rate for
            // as long as a run lasted. `CVLivePreviewLoader` holds exactly one,
            // and drops it the moment its view goes away.
            artifact: preview.map { .notFetched($0.redaction) } ?? .absent,
            preview: preview,
            artifactUnavailableReason: json["artifact_unavailable_reason"] as? String
        )
    }

    var hasReport: Bool {
        count != nil || artifact != .absent
    }

    /// The descriptor, if there is one this build may actually draw.
    ///
    /// Fails closed at the last step rather than at the first: a preview whose
    /// treatment is `.unknown` is still decoded and still reported, so the
    /// panel can say why it is not drawing rather than behaving as though the
    /// Tower had offered nothing.
    var drawablePreview: CVLivePreview? {
        guard let preview, preview.isDrawable else { return nil }
        return preview
    }
}

/// What the live view cost, kept apart from what the EXPERIMENT cost.
///
/// The whole reason this block exists is one question a physical test has to
/// be able to answer with evidence rather than opinion: **did adding a picture
/// slow the CV pipeline down?**
///
/// Two figures are deliberately not mixed. `CVTimings.processingMs` is what the
/// experiment costs and stays comparable against every number recorded before
/// previews existed; `renderMs` here is what a picture costs, on a worker
/// thread, at a different rate. Averaging them together would destroy the one
/// measurement that answers the question.
///
/// Nothing here is switched on. It is a diagnostics panel, and every field is
/// `nil` on a Tower that does not report it rather than defaulted to a zero
/// that would read as a measurement.
nonisolated struct CVPreviewDiagnostics: Equatable, Sendable {
    /// Frames the Lab told the preview about.
    var framesOffered: Int?
    /// …of which this many became the newest picture.
    var captured: Int?
    /// …and this many were never even asked for, because the throttle said the
    /// last capture was too recent. **This is the figure that says
    /// visualisation runs at its own rate**, independent of the experiment's.
    var skippedByThrottle: Int?
    /// Captures overwritten before anything drew them. The intended behaviour,
    /// counted rather than hidden: `captured - replacedUnread` is roughly what
    /// the phone actually saw.
    var replacedUnread: Int?
    /// Encodes actually performed — far fewer than `captured`, and the gap is
    /// the design: nothing is encoded until somebody asks for it.
    var encoded: Int?
    /// Renders that raised. The run is unaffected by construction.
    var encodeFailures: Int?
    var served: Int?
    /// Fetches answered `304`. A high figure means the phone is polling faster
    /// than the Tower produces, which costs a round trip and no encode.
    var notModified: Int?
    var refused: Int?
    var renderMs: Double?
    var renderMsMax: Double?
    var payloadBytes: Int?
    var payloadBytesMax: Int?

    init(json: [String: Any]) {
        framesOffered = json["frames_offered"] as? Int
        captured = json["captured"] as? Int
        skippedByThrottle = json["skipped_by_throttle"] as? Int
        replacedUnread = json["replaced_unread"] as? Int
        encoded = json["encoded"] as? Int
        encodeFailures = json["encode_failures"] as? Int
        served = json["served"] as? Int
        notModified = json["not_modified"] as? Int
        refused = json["refused"] as? Int
        renderMs = json["render_ms"] as? Double
        renderMsMax = json["render_ms_max"] as? Double
        payloadBytes = json["payload_bytes"] as? Int
        payloadBytesMax = json["payload_bytes_max"] as? Int
    }

    /// Roughly how many of the Tower's captures reached a screen, as a
    /// sentence. `nil` when nothing has been captured yet.
    ///
    /// "Roughly" is load-bearing and stays in the copy: the Tower counts a
    /// capture as read when it was rendered, and a render that was fetched but
    /// never drawn — a dropped packet, a view that went away mid-flight — is
    /// counted here as seen. The figure is an upper bound on what a person
    /// looked at, and the copy must not imply otherwise.
    var deliveryNote: String? {
        guard let captured, captured > 0 else { return nil }
        let dropped = replacedUnread ?? 0
        return """
            \(captured - dropped) of \(captured) captured previews were drawn; \
            \(dropped) were replaced by a newer one first.
            """
    }
}

/// Timings for one run, keeping the clocks the platform requires be kept apart.
///
/// `docs/07-PLATFORM-CONSTRAINTS.md` Limitation 9: Tower receipt time must not
/// be treated as camera capture time, and processing time is a third quantity
/// again. `processingMs` is the Tower measuring itself — a genuine measurement,
/// and the only one of the three iOS can trust unreservedly.
///
/// **There is no end-to-end latency field**, and its absence is deliberate on
/// both sides. It would have to be computed across the phone's clock and the
/// Tower's, and the DAT frame timestamp's semantics are an open question in
/// that same document. A number computed from two clocks of unknown
/// relationship is not a latency. The Tower agrees: *"iOS asked for none."*
struct CVTimings: Equatable, Sendable {
    /// The **mean** wall-clock milliseconds the Tower spent per frame, as the
    /// Tower measured it. `nil` until a frame has been processed.
    var processingMs: Double?
    /// The worst single frame of the run.
    var processingMsMax: Double?
    /// Per-stage breakdown, name → ms. An **open map**: its keys are the
    /// experiment's own stage names and a client must not switch on them.
    /// Bounded by the Tower at 16 names per run.
    var stageMs: [String: Double]
    /// When the underlying observation happened and when the report arrived.
    /// `observedAt` is when the Tower last produced a result for this run —
    /// Tower receipt, never shutter.
    var time: ObservationTime

    init(
        processingMs: Double? = nil,
        processingMsMax: Double? = nil,
        stageMs: [String: Double] = [:],
        time: ObservationTime = ObservationTime()
    ) {
        self.processingMs = processingMs
        self.processingMsMax = processingMsMax
        self.stageMs = stageMs
        self.time = time
    }

    init(json: [String: Any], receivedAt: Date?) {
        self.init(
            processingMs: json["processing_ms"] as? Double,
            processingMsMax: json["processing_ms_max"] as? Double,
            stageMs: json["stage_ms"] as? [String: Double] ?? [:],
            time: ObservationTime(
                observedAt: CVLabWireTime.date(json["observed_at"]),
                receivedAt: receivedAt
            )
        )
    }
}

/// How fast the run went, and how fast it could have gone.
///
/// **Every figure here is nullable and none of them may be rendered as zero.**
/// The two rates are `nil` specifically while `elapsed_s` is `0.0`, because a
/// rate over a zero-length window is undefined rather than zero — and that is
/// not a rare edge: the Tower measured it on 11 of 12 `cv_lab_start` replies.
/// `capacityFps` is `nil` until one frame has been processed, because it is
/// derived from measured per-frame cost.
///
/// Read together, `processedFps` and `capacityFps` say whether the Lab or the
/// link is the limit: the current sender forwards roughly one frame in thirty,
/// so the processed rate is normally bounded by what arrives.
struct CVThroughput: Equatable, Sendable {
    var processedFps: Double?
    var offeredFps: Double?
    /// `1000 / processing_ms` — how fast the Lab could go if frames never
    /// stopped arriving.
    var capacityFps: Double?

    init(processedFps: Double? = nil, offeredFps: Double? = nil, capacityFps: Double? = nil) {
        self.processedFps = processedFps
        self.offeredFps = offeredFps
        self.capacityFps = capacityFps
    }

    init(json: [String: Any]) {
        self.init(
            processedFps: json["processed_fps"] as? Double,
            offeredFps: json["offered_fps"] as? Double,
            capacityFps: json["capacity_fps"] as? Double
        )
    }
}

/// One execution of one experiment.
///
/// A run is **the unit of provenance**. Every figure below belongs to exactly
/// one `runID`, and starting a different experiment mints a new run and takes
/// the previous one's figures out of the document entirely — because keeping an
/// old summary beside a new one is how a number from the wrong experiment ends
/// up on a screen. Stop, rather than switch, to keep a run readable.
struct CVExperimentRun: Equatable, Sendable {
    /// `"<tower_instance_id>-<n>"`. Opaque, compared for equality only — and
    /// the one field `TowerClient` gates `frame_result` on.
    var runID: String?
    let experiment: CVExperiment
    /// `client_request` or `startup_default`. **`startup_default` means nobody
    /// asked for this run** — the Tower armed it at boot from
    /// `TOWER_CV_EXPERIMENT`. Reported so that "the Lab is running" never reads
    /// as "somebody chose this".
    var origin: String?
    var startedAt: Date?
    /// When a stop froze this run, or `nil` while it is still open.
    var endedAt: Date?
    var elapsedSeconds: Double?
    /// What the experiment says it actually loaded — device, model, backend.
    /// An **open map** whose keys are the experiment's own; do not switch on
    /// them. It exists because `TOWER_CV_DEVICE=auto` is a *request* and the
    /// Tower decides the answer: a run labelled "auto" has not said whether it
    /// used the GPU, and a CPU figure with a GPU label on it is a real failure
    /// this closes.
    var runtime: [String: String]
    var metrics: [CVMetric]
    /// How many aggregate metrics did not fit the Tower's 16-row bound.
    /// Reported rather than silently truncated.
    var metricsOmitted: Int
    /// Metrics an experiment emitted without declaring how they combine across
    /// frames. Empty is the only correct value and a Tower test enforces it;
    /// this is what the wire says if one ever reaches production anyway.
    var unclassifiedMetrics: [String]
    var annotation: CVAnnotationReport
    /// What the live view cost, as the Tower measured it. `nil` on a Tower
    /// that does not report one.
    var preview: CVPreviewDiagnostics?
    var timings: CVTimings
    var throughput: CVThroughput
    /// Frames the Lab measured.
    var framesProcessed: Int?
    /// Frames the Lab declined — paused, arming, stopped. **Not failures.** On
    /// a paused run this is the counter that keeps moving, and it is how you
    /// see whether the phone is still sending while paused.
    var framesRefused: Int?
    /// Frames the experiment raised on. It stays armed; those frames produced
    /// nothing.
    var framesFailed: Int?
    /// Derived by the Tower as processed + refused + failed, so the sum holds
    /// at every read rather than only between them. That invariant is what
    /// makes a dead start diagnosable — see `ExperimentalCVState.diagnosis`.
    var framesOffered: Int?

    init(
        runID: String? = nil,
        experiment: CVExperiment,
        origin: String? = nil,
        startedAt: Date? = nil,
        endedAt: Date? = nil,
        elapsedSeconds: Double? = nil,
        runtime: [String: String] = [:],
        metrics: [CVMetric] = [],
        metricsOmitted: Int = 0,
        unclassifiedMetrics: [String] = [],
        annotation: CVAnnotationReport = CVAnnotationReport(),
        preview: CVPreviewDiagnostics? = nil,
        timings: CVTimings = CVTimings(),
        throughput: CVThroughput = CVThroughput(),
        framesProcessed: Int? = nil,
        framesRefused: Int? = nil,
        framesFailed: Int? = nil,
        framesOffered: Int? = nil
    ) {
        self.runID = runID
        self.experiment = experiment
        self.origin = origin
        self.startedAt = startedAt
        self.endedAt = endedAt
        self.elapsedSeconds = elapsedSeconds
        self.runtime = runtime
        self.metrics = metrics
        self.metricsOmitted = metricsOmitted
        self.unclassifiedMetrics = unclassifiedMetrics
        self.annotation = annotation
        self.preview = preview
        self.timings = timings
        self.throughput = throughput
        self.framesProcessed = framesProcessed
        self.framesRefused = framesRefused
        self.framesFailed = framesFailed
        self.framesOffered = framesOffered
    }

    /// One `run` block, or `nil` when the Tower sent something that is not one.
    ///
    /// A run with no experiment cannot be described — every figure below is
    /// "what *this experiment* measured" — so the whole block is refused rather
    /// than shown against a placeholder name.
    init?(json: [String: Any]) {
        guard let rawExperiment = json["experiment"] as? [String: Any],
            let experiment = CVExperiment(json: rawExperiment)
        else { return nil }

        var metrics: [CVMetric] = []
        for entry in json["metrics"] as? [[String: Any]] ?? [] {
            if let metric = CVMetric(json: entry) { metrics.append(metric) }
        }

        // `receivedAt` is the phone's clock and is only ever used as "when this
        // app was told", never promoted to observation time. `ObservationTime`
        // has no path that would allow the promotion.
        let receivedAt = Date()

        self.init(
            runID: json["run_id"] as? String,
            experiment: experiment,
            origin: json["origin"] as? String,
            startedAt: CVLabWireTime.date(json["started_at"]),
            endedAt: CVLabWireTime.date(json["ended_at"]),
            elapsedSeconds: json["elapsed_s"] as? Double,
            // Stringified rather than typed: the Tower's own contract says the
            // keys and the values are the experiment's, and a `[String: Any]`
            // that reached a view would be formatted by whoever drew it.
            runtime: CVExperimentRun.readRuntime(json["runtime"]),
            metrics: metrics,
            metricsOmitted: json["metrics_omitted"] as? Int ?? 0,
            unclassifiedMetrics: json["unclassified_metrics"] as? [String] ?? [],
            annotation: CVAnnotationReport(json: json["annotation"] as? [String: Any] ?? [:]),
            preview: (json["preview"] as? [String: Any]).map(CVPreviewDiagnostics.init),
            timings: CVTimings(
                json: json["timings"] as? [String: Any] ?? [:],
                receivedAt: receivedAt
            ),
            throughput: CVThroughput(json: json["throughput"] as? [String: Any] ?? [:]),
            framesProcessed: json["frames_processed"] as? Int,
            framesRefused: json["frames_refused"] as? Int,
            framesFailed: json["frames_failed"] as? Int,
            framesOffered: json["frames_offered"] as? Int
        )
    }

    /// The Tower's `runtime` map, rendered to strings at the decode boundary.
    ///
    /// Numbers arrive as numbers and strings as strings — `device: "cuda:0"`
    /// beside a numeric setting — and the view must not be the place that
    /// decides how each prints.
    private static func readRuntime(_ value: Any?) -> [String: String] {
        guard let raw = value as? [String: Any] else { return [:] }
        var runtime: [String: String] = [:]
        for (key, value) in raw {
            if let text = value as? String {
                runtime[key] = text
            } else if let number = value as? Double {
                runtime[key] = CVMetric.format(number)
            } else if let flag = value as? Bool {
                runtime[key] = flag ? "yes" : "no"
            }
            // Anything else — a nested object, a null — is left out rather
            // than rendered as its Swift description.
        }
        return runtime
    }

    /// True when at least one metric came from a model rather than a
    /// measurement, which is what obliges the workspace to show the inference
    /// caveat once for the whole run rather than repeating it per row.
    var containsInference: Bool {
        metrics.contains { $0.provenance.isInference }
    }

    /// Whether this run has measured anything at all.
    ///
    /// The distinction the Tower's nulls exist for: a run that has processed no
    /// frame — *"which is every Release build, and every Tower nobody has
    /// streamed to yet"* — publishes empty metrics and null timings, and none
    /// of that may be drawn as zero.
    var hasMeasuredAnything: Bool { (framesProcessed ?? 0) > 0 }
}

/// What the frame counters say about why nothing is arriving.
///
/// The Tower's `frames_offered == processed + refused + failed` invariant is
/// what makes a dead start diagnosable, and this is the table it published for
/// reading it. Kept as a type rather than as four `if`s in a view so that the
/// reading is asserted once in a test instead of eyeballed on a screen.
enum CVLabDiagnosis: Equatable, Sendable {
    /// Nothing is reaching the Tower at all: the stream is not running.
    case nothingArriving
    /// Frames **are** arriving and the transport cannot decode them. A sender
    /// problem, not a Lab one — and it looks identical to the case above
    /// without `source.frames_rejected_before_lab`, which is why that field
    /// exists.
    case arrivingButUndecodable(count: Int)
    /// Frames are arriving and the Lab is refusing them. Check the lifecycle
    /// state: paused, arming, stopped.
    case arrivingButRefused
    /// The experiment raised on at least one frame. It stays armed; those
    /// frames produced nothing.
    case someFramesFailed(count: Int)
    /// Frames are arriving and being measured.
    case measuring
}

// MARK: - The frame channel

/// The running experiment's own answer for one frame, as the Tower's per-frame
/// reply reports it.
///
/// ## Why this is still a separate type from `ExperimentalCVState`
///
/// It used to be separate because the frame channel had **no contract behind it
/// and no provenance attached**, and folding it into the cartridge state would
/// have meant either inventing a state the Tower never offered or letting
/// `.unsupported` carry a payload.
///
/// Half of that reason is now gone: `frame_result` carries a `cv_lab` block
/// under `experimental_cv.frame_result/2026-08-27`, with provenance, the
/// experiment's id and name, and the run it belongs to. The other half stands,
/// and is why the split survives: **it is a different transport with a
/// different contract identifier and a different rate.** The status document
/// arrives ~1.3 times a second and describes a run; this arrives per frame and
/// describes one frame. The Tower versions the two separately for exactly that
/// reason, and a client that merged them would be unable to implement the
/// read-only half alone — which is what a Release build with no camera is.
///
/// ## The pair rule
///
/// `result_value` is a bare number whose meaning belongs to the experiment;
/// `result_label` is the experiment's own name for it. Neither is readable
/// alone here — they are held as one `Labelled` or not at all — so no view can
/// render the number under a caption of its own invention.
struct CVFrameReading: Equatable, Sendable {
    /// A number and the Tower's own name for it. There is no way to construct
    /// one without both, which is where the pair rule actually lives.
    struct Labelled: Equatable, Identifiable, Sendable {
        /// What tells one row from another within a single reading.
        ///
        /// **Not the label.** For a measurement it is the Tower's own
        /// dictionary key, kept exactly as sent: keys are unique by
        /// construction, whereas `label` is trimmed, so `{"edges": 1,
        /// " edges": 2}` collapses to one caption. When identity was the
        /// trimmed caption, `ForEach` saw two rows with one id and the sort key
        /// was equal for both — and `sorted(by:)` is not stable, so the pair
        /// could swap places on every reply, which is the twelve-times-a-second
        /// reshuffle the sort was added to prevent.
        let id: String
        /// The Tower's words, shown verbatim and never matched on.
        let label: String
        let value: Double

        /// The only initialiser, and so genuinely the only way to build one.
        ///
        /// Failable rather than throwing because there is exactly one outcome
        /// for every way of failing — the Tower did not send a usable pair —
        /// and nothing downstream can act differently on which.
        ///
        /// A blank label counts as no label. The wire cannot distinguish an
        /// absent string from an empty one, and a row captioned with whitespace
        /// is a bare number with extra steps.
        ///
        /// - Parameter id: The wire key this came from, when there is one.
        ///   Defaults to the trimmed label, which is right for the headline —
        ///   a reading has at most one.
        init?(id: String? = nil, label: String?, value: Double?) {
            guard let value, let label else { return nil }
            let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            self.id = id ?? trimmed
            self.label = trimmed
            self.value = value
        }

        /// Formatted by `CVMetric.format` — this app's single answer to "how do
        /// you print a number whose unit nobody named". Shared with the home
        /// screen's tile, so the two places this one figure is drawn cannot
        /// drift apart on how it reads.
        var displayValue: String { CVMetric.format(value) }
    }

    /// Which result this is **within its run**: dense, from 1.
    ///
    /// ## The field that replaced `sequence`, and why the swap matters
    ///
    /// This type used to carry the wire's `seq` under the name `sequence` and
    /// the workspace drew it as *"From frame 30."* That number is the **phone's
    /// capture index**, and the sender forwards one frame in thirty by design —
    /// so consecutive replies read 30, 60, 90, and the gaps are not losses.
    /// Worse, the index is chosen by the sender and says nothing about the
    /// order the Tower produced results in, so it **cannot order results**. The
    /// Tower says so in as many words.
    ///
    /// `result_seq` is the Tower's own dense counter for this run, and it is
    /// what says whether a reading is newer than the one already on screen.
    /// `nil` when the reply carried no `cv_lab` block at all — a Tower running
    /// no Lab.
    let resultSeq: Int?
    /// The phone's capture index, kept and **labelled as one**.
    ///
    /// Retained rather than dropped because it is still the only thread back to
    /// a specific captured frame, and during a stall the difference between a
    /// fresh answer and an old one is the whole story. It is never used to
    /// order anything.
    let captureIndex: Int?
    /// The run this reading belongs to. The gate that decides whether it is
    /// shown at all lives in `TowerClient`; this is carried so a view can say
    /// which run it is looking at.
    let runID: String?
    /// The experiment that produced it, in the Tower's own words. **This is
    /// what the old screen could not say**: the reply used to name the number
    /// and not the experiment, so the workspace could not report which one
    /// produced it.
    let experimentName: String?
    let experimentID: String?
    /// The experiment's headline answer, or `nil` when either half of the pair
    /// was missing.
    let headline: Labelled?
    /// Mean pixel intensity. `nil` means the experiment reported none, never
    /// that the frame was dark.
    let meanIntensity: Double?
    /// Wall-clock milliseconds the Tower spent on this frame, as the Tower
    /// measured it. "ms" is not an invented unit: the wire field is named
    /// `processing_ms`, so it is the Tower's own declaration.
    let processingMs: Double?
    /// When the **Tower** received the frame this answers. Never when the
    /// glasses captured it: there is no capture timestamp anywhere on this
    /// wire.
    let towerReceivedAt: Date?
    /// The Tower's additive measurements, each under the name the Tower gave
    /// it. Sorted by that name, because a dictionary has no stable order and a
    /// list of figures that reshuffles itself twelve times a second is
    /// unreadable — and the sort is *total*, breaking a tie between two
    /// captions on the untrimmed wire key, so that two keys differing only in
    /// whitespace cannot leave the order undetermined.
    let measurements: [Labelled]
    /// What the reply said about where its numbers came from.
    ///
    /// `.measured` or `.inferred` when the `cv_lab` block is present, which is
    /// every reply from a Tower running a Lab. `Self.provenance` — `.unknown` —
    /// when it is not.
    let provenance: ObservationProvenance

    /// The provenance of a reply that carried **no** `cv_lab` block.
    ///
    /// This used to be the provenance of every reply, because `frame_result`
    /// had no provenance field at all and Rule 16 does not permit silence to be
    /// read as "measured". A Tower speaking
    /// `experimental_cv.frame_result/2026-08-27` now states it on every frame,
    /// so the instance property above is the one to read.
    ///
    /// It stays for the Tower that does not: an older build, or one running no
    /// Lab. `.unknown` is still the honest answer there, and the caveat it
    /// carries is still owed wherever those figures are drawn.
    ///
    /// **`HomeWorkspaceView` reads this static.** That is not wrong — it states
    /// the strongest caveat, which is never a lie — but it understates what a
    /// current Tower says, and moving it to the instance property is noted in
    /// this cartridge's handoff.
    static let provenance: ObservationProvenance = .unknown

    /// Whether the Tower reported anything at all about the frame.
    ///
    /// A reply carrying only a sequence number is a real answer and a different
    /// one from no reply at all: the Tower processed the frame and the
    /// experiment concluded nothing about it. A workspace that showed those two
    /// identically would leave someone waiting for a result that has already
    /// arrived and was empty.
    var hasAnything: Bool {
        headline != nil || meanIntensity != nil || processingMs != nil || !measurements.isEmpty
    }

    /// Projects one Tower reply. Every field is optional on the wire and stays
    /// optional here; nothing is defaulted into existence.
    init(_ result: TowerFrameResult) {
        let cvLab = result.cvLab
        resultSeq = cvLab?.resultSeq
        captureIndex = result.sequence
        runID = cvLab?.runID
        experimentID = cvLab?.experimentID
        experimentName = cvLab?.experimentName
        headline = Labelled(label: result.resultLabel, value: result.resultValue)
        meanIntensity = result.meanIntensity
        processingMs = result.processingMs
        towerReceivedAt = CVLabWireTime.date(seconds: cvLab?.towerReceivedAt)
        provenance = cvLab.map { CVWireProvenance.read($0.provenance) } ?? Self.provenance
        measurements = result.metrics
            .compactMap { Labelled(id: $0.key, label: $0.key, value: $0.value) }
            // Ordered on the caption first, because that is what a reader sees,
            // and on the wire key second so the order is total. Dictionary keys
            // are unique, so no two entries can tie on both — which is what
            // makes this order independent of the dictionary's own, and of
            // `sorted(by:)` not being stable.
            .sorted { ($0.label, $0.id) < ($1.label, $1.id) }
    }
}

// MARK: - State

/// What the Experimental CV Lab workspace should be showing.
///
/// Seven Tower lifecycle states map onto six cases here, and the one place they
/// disagree is deliberate: the Tower says `stopped` and this says `.completed`.
/// *"A bench run does not complete; it is stopped by a person."* The Tower says
/// what happened and iOS renders it with the case its state machine has.
enum ExperimentalCVState: Equatable, Sendable {
    /// This Tower cannot run experiments at all. Carries the reason.
    ///
    /// Reached from `lifecycle.state == "unavailable"` and from the terminal
    /// `lab_unavailable` refusal — the two ways the Tower says the same thing.
    case unsupported(reason: String)
    /// Nothing armed; a start would be accepted. `available` is the Tower's
    /// catalog — possibly empty, which is a different and equally honest answer
    /// from "cannot run experiments at all".
    case idle(available: [CVExperiment])
    /// A start was accepted and the experiment is loading.
    ///
    /// **Bounded and unreportable.** The Tower gives an arm 120 s, and
    /// `torch.hub` offers no progress, so this is a spinner with a deadline and
    /// nothing else. A percentage here would be invented.
    case starting(CVExperiment)
    /// Processing frames.
    ///
    /// **Not the same as "results are arriving."** See `isLive`.
    case running(CVExperimentRun)
    /// Armed and deliberately not processing, with the experiment still loaded.
    ///
    /// A new case, and the difference from `.completed` is real and is two
    /// differences: a paused run keeps the experiment loaded, so resuming a
    /// `depth` run costs nothing while a stopped one pays the model load again;
    /// and a paused run is **not over**, so `framesRefused` keeps climbing with
    /// every frame that arrives while the metrics stand still.
    case paused(CVExperimentRun)
    /// The last run ended; its figures are final. The Tower calls this
    /// `stopped`.
    case completed(CVExperimentRun)
    /// The last **start** failed.
    ///
    /// ## There is no `start_failed` message, and this case is how one arrives
    ///
    /// An arm is asynchronous — that is the whole reason a start returns
    /// immediately — so by the time a load fails the command has already been
    /// answered `accepted`. **The outcome arrives as state**, on the next
    /// status document. A client that sends commands and does not also read
    /// status will never learn that a start failed, which is why
    /// `TowerExperimentalCVClient` subscribes as well as sends.
    ///
    /// A failed *interactive* start is recoverable: send another `cv_lab_start`.
    /// **Not a stop first** — the Lab refuses `cv_lab_stop` from `failed`, so
    /// "Stop then Start" is refused on step one.
    case failed(CartridgeFailure)

    /// The run to draw results from, when there is one.
    var run: CVExperimentRun? {
        switch self {
        case .running(let run), .paused(let run), .completed(let run): return run
        case .unsupported, .idle, .starting, .failed: return nil
        }
    }

    /// Whether an experiment is armed and processing on the Tower right now.
    ///
    /// A statement about the **Lab**, not about this phone's frames. See
    /// `isLive(isStreaming:)`, which is the one a screen may draw as live.
    var isRunning: Bool {
        if case .running = self { return true }
        return false
    }

    /// Whether this build may show the run as **live**.
    ///
    /// > `.running` may be shown as LIVE only when this build is itself
    /// > streaming **and** `source.receiving_frames` is true.
    ///
    /// Both halves are needed and neither is sufficient. `source` is
    /// **Tower-wide**: on a Tower with a second phone attached it reads `true`
    /// for a Release build that has no camera at all, and the client's own
    /// streaming state is the half that catches that. The other way round, a
    /// phone that is streaming to a Tower whose Lab has seen nothing for five
    /// seconds is not producing results either.
    ///
    /// - Parameters:
    ///   - isStreaming: whether **this build** has a stream bracket open —
    ///     `TowerClient.isStreamingToTower`, which is permanently `false` in
    ///     Release because the frame path is `#if DEBUG`.
    ///   - isReceivingFrames: `source.receiving_frames` from the Tower.
    func isLive(isStreaming: Bool, isReceivingFrames: Bool) -> Bool {
        isRunning && isStreaming && isReceivingFrames
    }

    var phase: CartridgePhase {
        switch self {
        case .unsupported: return .unsupported
        case .idle: return .idle
        case .starting: return .waiting
        case .running: return .live
        // `.settled` rather than `.live`: a paused run is not producing
        // anything, and a spinner or a live glyph over it would claim work that
        // is deliberately not happening. It is not `.idle` either — the figures
        // are real and `mayCarryData` has to permit them. The screen says
        // "Paused" in its own words; the phase only decides what is said when
        // there is nothing specific to show.
        case .paused: return .settled
        case .completed: return .settled
        case .failed: return .failed
        }
    }

    /// Why nothing is arriving, read off the run's counters and the Tower's
    /// `source` block.
    ///
    /// The Tower publishes the counters specifically so this question has an
    /// answer, and its own table is reproduced in `CVLabDiagnosis`. `nil` when
    /// there is no run to diagnose.
    /// ## Why `receivingFrames` is consulted before any counter
    ///
    /// Every counter on `run` is **cumulative for the life of the run**, and
    /// three of this enum's five cases are written in the present tense
    /// ("Frames **are** arriving…"). Reading a present-tense claim off a
    /// monotonic counter means the claim latches: one frame processed an hour
    /// ago made `.measuring` true forever, and the panel printed
    ///
    ///     Frames reaching the Tower    no
    ///     Frames are arriving and being measured.
    ///
    /// as adjacent rows, because `sourceRows` is drawn under every state. The
    /// contradiction was visible on screen and the app asserted the false half.
    ///
    /// `source.receivingFrames` is the Tower's own answer to "now": it is
    /// `lastFrameAt` within `idleAfterSeconds`. Consulting it first turns the
    /// present tense back into a statement about the present.
    ///
    /// `.someFramesFailed` stays above the gate deliberately — it is the one
    /// case that is *not* a claim about now. An experiment that raised on a
    /// frame stays worth reporting after the frames stop, and its wording is
    /// past tense already.
    func diagnosis(source: CVLabStatus.Source) -> CVLabDiagnosis? {
        guard let run else { return nil }
        if let failed = run.framesFailed, failed > 0 { return .someFramesFailed(count: failed) }
        guard source.receivingFrames else {
            // Nothing is arriving *now*, whatever this run measured earlier.
            // Undecodable frames are still worth naming when they are the
            // reason nothing reached the Lab, but only as the count they are.
            let rejected = source.framesRejectedBeforeLab ?? 0
            return rejected > 0 ? .arrivingButUndecodable(count: rejected) : .nothingArriving
        }
        let offered = run.framesOffered ?? 0
        guard offered > 0 else {
            let rejected = source.framesRejectedBeforeLab ?? 0
            return rejected > 0 ? .arrivingButUndecodable(count: rejected) : .nothingArriving
        }
        return (run.framesProcessed ?? 0) > 0 ? .measuring : .arrivingButRefused
    }
}

// MARK: - A command in flight

/// A command this app has sent and the Tower has not yet answered.
///
/// ## Why the phone tracks this at all
///
/// Every command on this wire is answered immediately and the outcome arrives
/// as state — that is the shape `ExperimentalCVClient` documents. Between the
/// tap and the reply, though, the state has not changed, because the document
/// is the only thing that changes it. So without this the row that was tapped
/// looked exactly like one that was not, for as long as the reply took, and a
/// person's reasonable response to a button that did nothing is to press it
/// again — which on a one-slot Lab is a second start racing the first.
///
/// This is deliberately **not** an optimistic state change. The lifecycle
/// stays where the document put it; this records only that a question was
/// asked, and which one. `.starting` remains the Tower's own, distinct phase,
/// drawn from the document, so a person sees two truthful phases: *asked*,
/// then *loading*.
///
/// ## What clears it
///
/// A `cv_lab_status` or `cv_lab_error` whose `request_id` matches, the socket
/// leaving `.online`, or `TowerExperimentalCVClient.replyBoundSeconds` passing.
/// **A reply with no `request_id` clears nothing.** The status document is also
/// pushed on the result channel and read on connect, and neither of those
/// carries a request id; letting them clear this would have the subscription's
/// heartbeat answer a question it was never asked.
nonisolated struct CVLabPendingCommand: Equatable, Sendable {
    let command: CVLabCommand
    /// The experiment a start named. `nil` for pause, resume and stop, which
    /// act on whatever run is current.
    let experimentID: String?
    /// The `request_id` that went out, and the only thing a reply is matched on.
    let requestID: String
    let sentAt: Date

    /// Whether a reply carrying `replyRequestID` answers this command.
    ///
    /// `nil` is never a match, by design and not by accident of `Optional`
    /// comparison: a reply that names no request answers no request.
    func isAnswered(by replyRequestID: String?) -> Bool {
        guard let replyRequestID else { return false }
        return replyRequestID == requestID
    }

    /// The command as a person would say it, for the one sentence this app
    /// writes about an unanswered one.
    var verb: String {
        switch command {
        case .status: return "status"
        case .start: return "start"
        case .pause: return "pause"
        case .resume: return "resume"
        case .stop: return "stop"
        }
    }
}
