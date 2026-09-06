//
//  CVLabContractTests.swift
//  GlassesTests
//
//  The Experimental CV Lab's decode, projection and display rules, asserted
//  against messages captured from a **live Tower** at 127.0.0.1:8765 rather
//  than against a summary of the contract. Every fixture below is a real reply:
//  the status document from an `optical_flow` run that measured three frames,
//  the refusals a real Tower sent for an unknown experiment and a stale run,
//  and the `frame_result` / `frame_error` a real frame produced.
//
//  What is deliberately NOT tested here is a better/worse verdict, because
//  there is no longer any code that could produce one. `CVMetric` carries no
//  `baseline`, no `higherIsBetter` and no `comparison`; the Tower sends those
//  fields as `null` on every metric, always, and a comparison against nothing
//  is the "declaring an approach 'better' without a measurement" that
//  `docs/modules/EXPERIMENTAL-CV.md` rules out. Deleting the machinery rather
//  than relying on the nulls is what makes its absence a property of the build
//  instead of a property of today's payload.
//

import UIKit
import XCTest

@testable import Glasses

@MainActor
final class CVLabContractTests: XCTestCase {

    // MARK: - Fixtures

    /// A live `optical_flow` status document, trimmed to three catalog entries
    /// and four metrics.
    ///
    /// The four metrics are chosen for what they prove rather than for
    /// coverage: a `rate` headline, a second `rate`, a `count`, and
    /// **`dominant_direction_deg`, whose value is `null` because it is
    /// `unaggregated`** — the mean of 179 degrees and -179 degrees is 0, the
    /// one direction neither frame was moving in. That null is the single most
    /// important byte in this file.
    private static let liveStatusJSON = """
        {
          "contract": "experimental_cv.status/2026-08-27",
          "control_contract": "experimental_cv.control/2026-08-27",
          "frame_result_contract": "experimental_cv.frame_result/2026-08-27",
          "tower_instance_id": "f863dcc35bce",
          "time_basis": "tower-receipt",
          "lifecycle": {
            "state": "running",
            "reason": null,
            "since": 1787834577.72117,
            "run_id": "f863dcc35bce-6"
          },
          "available": [
            {
              "id": "baseline",
              "name": "Baseline",
              "summary": "Mean grayscale intensity of the frame. The cheapest thing that proves the whole glasses -> phone -> Tower -> CV path is alive.",
              "provenance": "measured",
              "headline_label": "mean_intensity",
              "headline_unit": "level",
              "stateful": false,
              "requires_model": false,
              "backend": "opencv",
              "annotation_metric": null,
              "available": true,
              "unavailable_reason": null
            },
            {
              "id": "depth",
              "name": "Monocular depth",
              "summary": "Relative inverse depth from MiDaS-small. NOT metric distance: the model does not produce one, and no figure here may be read as metres.",
              "provenance": "inferred",
              "headline_label": "mean_relative_depth",
              "headline_unit": null,
              "stateful": true,
              "requires_model": true,
              "backend": "torch",
              "annotation_metric": null,
              "available": false,
              "unavailable_reason": "needs the optional [ml] extra (torch), which is not installed on this Tower"
            },
            {
              "id": "optical_flow",
              "name": "Optical flow",
              "summary": "How much the scene is moving and how coherently, from sparse Lucas-Kanade with a forward-backward check. The first frame of a session has no answer and says so.",
              "provenance": "measured",
              "headline_label": "median_flow_px",
              "headline_unit": "px",
              "stateful": true,
              "requires_model": false,
              "backend": "opencv",
              "annotation_metric": null,
              "available": true,
              "unavailable_reason": null
            }
          ],
          "selected": "optical_flow",
          "default_experiment": "baseline",
          "device_requested": "auto",
          "run": {
            "run_id": "f863dcc35bce-6",
            "experiment": {
              "id": "optical_flow",
              "name": "Optical flow",
              "summary": "How much the scene is moving and how coherently, from sparse Lucas-Kanade with a forward-backward check. The first frame of a session has no answer and says so.",
              "provenance": "measured",
              "headline_label": "median_flow_px",
              "headline_unit": "px",
              "stateful": true,
              "requires_model": false,
              "backend": "opencv",
              "annotation_metric": null,
              "available": true,
              "unavailable_reason": null
            },
            "origin": "client_request",
            "started_at": 1787834577.720414,
            "ended_at": null,
            "elapsed_s": 0.084,
            "runtime": {},
            "frames_offered": 3,
            "frames_processed": 3,
            "frames_refused": 0,
            "frames_failed": 0,
            "metrics": [
              {
                "label": "median_flow_px",
                "value": 3.7360259691874185,
                "unit": "px",
                "aggregation": "rate",
                "frames": 3,
                "provenance": "measured",
                "confidence": null,
                "headline": true,
                "varied": false,
                "baseline": null,
                "higher_is_better": null
              },
              {
                "label": "direction_coherence",
                "value": 0.4194576144218445,
                "unit": "fraction",
                "aggregation": "rate",
                "frames": 2,
                "provenance": "measured",
                "confidence": null,
                "headline": false,
                "varied": false,
                "baseline": null,
                "higher_is_better": null
              },
              {
                "label": "dominant_direction_deg",
                "value": null,
                "unit": "deg",
                "aggregation": "unaggregated",
                "frames": 2,
                "provenance": "measured",
                "confidence": null,
                "headline": false,
                "varied": false,
                "baseline": null,
                "higher_is_better": null
              },
              {
                "label": "has_reference",
                "value": 2.0,
                "unit": "frames",
                "aggregation": "count",
                "frames": 3,
                "provenance": "measured",
                "confidence": null,
                "headline": false,
                "varied": false,
                "baseline": null,
                "higher_is_better": null
              }
            ],
            "metrics_omitted": 0,
            "unclassified_metrics": [],
            "annotation": {
              "count": null,
              "count_unavailable_reason": "this experiment reports no annotation count",
              "artifact": {
                "contract": "experimental_cv.preview/2026-08-29",
                "kind": "live_preview",
                "visual_kind": "flow_tracks",
                "description": "One arrow per tracked point, coloured by direction, over a line drawing of the frame. Red dots are seeds the forward-backward check rejected.",
                "treatment": "raw_ephemeral",
                "face_filter": "none",
                "persistence": "none",
                "derived_from": "one frame, transiently, in memory",
                "path": "/cv-lab/preview",
                "media_type": "image/png",
                "run_id": "5a4a5f01ac52-2",
                "max_age_s": 2.0,
                "poll_interval_s": 0.1,
                "max_edge_px": 320
              },
              "artifact_unavailable_reason": null
            },
            "timings": {
              "processing_ms": 18.4667,
              "processing_ms_max": 43.6125,
              "stage_ms": {
                "decode": 2.7846,
                "seed": 4.0148,
                "summarize": 14.633,
                "track": 4.8753
              },
              "observed_at": 1787834577.800264,
              "time_basis": "tower-receipt"
            },
            "throughput": {
              "processed_fps": 35.754,
              "offered_fps": 35.754,
              "capacity_fps": 54.15
            }
          },
          "source": {
            "clients_connected": 1,
            "receiving_frames": true,
            "last_frame_at": 1787834577.800264,
            "frames_offered_total": 16,
            "frames_rejected_before_lab": 3,
            "idle_after_s": 5.0
          }
        }
        """

    private func decode(_ text: String) -> [String: Any]? {
        guard let data = text.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private func liveStatus() throws -> CVLabStatus {
        let json = try XCTUnwrap(decode(Self.liveStatusJSON))
        return try XCTUnwrap(CVLabStatus(json: json))
    }

    /// A status document with the lifecycle swapped, everything else identical.
    ///
    /// Built by editing the live one rather than by writing seven fixtures, so
    /// the projection is exercised against the shape the Tower actually sends
    /// in every case — including the fields a hand-written `failed` fixture
    /// would quietly leave out.
    private func status(state: String, reason: String? = nil, keepRun: Bool = true) throws
        -> CVLabStatus
    {
        var json = try XCTUnwrap(decode(Self.liveStatusJSON))
        var lifecycle = try XCTUnwrap(json["lifecycle"] as? [String: Any])
        lifecycle["state"] = state
        lifecycle["reason"] = reason as Any
        json["lifecycle"] = lifecycle
        if !keepRun { json["run"] = NSNull() }
        return try XCTUnwrap(CVLabStatus(json: json))
    }

    // MARK: - 1. The three identifiers

    /// Opaque, compared for equality only, and **three of them** because the
    /// Lab's three surfaces version independently — which is what lets a client
    /// implement the read-only half and never send a command.
    func testTheContractIdentifiersAreTheOnesTheTowerDeclares() {
        XCTAssertEqual(ExperimentalCVContract.status, "experimental_cv.status/2026-08-27")
        XCTAssertEqual(ExperimentalCVContract.control, "experimental_cv.control/2026-08-27")
        XCTAssertEqual(
            ExperimentalCVContract.frameResult, "experimental_cv.frame_result/2026-08-27"
        )
        XCTAssertEqual(ExperimentalCVContract.resultType, "status")
    }

    /// The name mapping trap, in one assertion.
    ///
    /// The Tower advertises `experimental_cv` with an underscore and this app's
    /// catalog says `experimental-cv` with a hyphen. Without an entry joining
    /// them in `TowerCapabilities.towerCartridgeNames`, availability resolves
    /// `.noContract` and this cartridge reports that the Tower has declared
    /// nothing against a Tower that has declared everything.
    func testTheTowersNameAndThisAppsCatalogIDAreDifferentStrings() {
        XCTAssertEqual(ExperimentalCVContract.towerCartridge, "experimental_cv")
        XCTAssertEqual(ExperimentalCVContract.catalogID, "experimental-cv")
        XCTAssertNotEqual(
            ExperimentalCVContract.towerCartridge, ExperimentalCVContract.catalogID,
            "if these ever match, the mapping this cartridge depends on has been deleted"
        )
        XCTAssertTrue(
            Cartridge.catalog.contains { $0.id == ExperimentalCVContract.catalogID },
            "the catalog id this client answers for is not in the catalog"
        )
    }

    /// 64 characters, and the reason the bound is enforced on this side.
    func testTheRequestIDBoundIsTheTowers() {
        XCTAssertEqual(ExperimentalCVContract.requestIDMaxLength, 64)
        // The arm timeout is 120 s because 119 MB of MiDaS weights does not fit
        // a 10 s bound on any ordinary link, and the workspace quotes it to a
        // person who is watching a spinner.
        XCTAssertEqual(ExperimentalCVContract.armTimeoutSeconds, 120)
    }

    // MARK: - 2. The status document

    func testTheLiveStatusDocumentDecodesFieldForField() throws {
        let status = try liveStatus()

        XCTAssertEqual(status.contract, ExperimentalCVContract.status)
        XCTAssertEqual(status.controlContract, ExperimentalCVContract.control)
        XCTAssertEqual(status.frameResultContract, ExperimentalCVContract.frameResult)
        XCTAssertEqual(status.towerInstanceID, "f863dcc35bce")
        // Every timestamp on this wire. There is no capture timestamp anywhere
        // on it, so a Tower time is when the Tower saw something.
        XCTAssertEqual(status.timeBasis, "tower-receipt")

        XCTAssertEqual(status.lifecycle.state, "running")
        XCTAssertEqual(status.lifecycle.runID, "f863dcc35bce-6")
        // `nil` is not "no reason" — it is "the state speaks for itself".
        XCTAssertNil(status.lifecycle.reason)
        XCTAssertNotNil(status.lifecycle.since)

        XCTAssertEqual(status.selected, "optical_flow")
        // Reported so that "the Lab is running" never reads as "somebody chose
        // this": the startup default is a different fact from the selection.
        XCTAssertEqual(status.defaultExperiment, "baseline")
        // A *request*, not an answer. What the run actually used is in
        // `run.runtime`, and a CPU figure with a GPU label on it is the real
        // failure that distinction closes.
        XCTAssertEqual(status.deviceRequested, "auto")
    }

    /// The catalog is the Tower's, in the Tower's order, with its own
    /// per-experiment availability.
    func testTheCatalogIsTheTowersAndAnUnavailableEntryCarriesItsReason() throws {
        let status = try liveStatus()
        XCTAssertEqual(status.available.map(\.id), ["baseline", "depth", "optical_flow"])

        let depth = try XCTUnwrap(status.available.first { $0.id == "depth" })
        XCTAssertFalse(depth.isStartable, "an experiment this Tower cannot run was offered")
        XCTAssertEqual(
            depth.unavailableReason,
            "needs the optional [ml] extra (torch), which is not installed on this Tower"
        )
        // Declared before it runs, which is the useful moment: this says the
        // numbers it would produce are model output.
        XCTAssertTrue(depth.provenance.isInference)
        XCTAssertTrue(depth.requiresModel)
        // A `nil` unit means the quantity genuinely has none and is rendered
        // bare. Depth is the case: MiDaS-small emits relative inverse depth on
        // an arbitrary scale, and rendering it with a unit would be metres by
        // another name.
        XCTAssertNil(depth.headlineUnit)

        let baseline = try XCTUnwrap(status.available.first { $0.id == "baseline" })
        XCTAssertTrue(baseline.isStartable)
        XCTAssertEqual(baseline.provenance, .measured)
        XCTAssertEqual(status.selectedExperiment?.id, "optical_flow")
    }

    /// Tower-wide, every figure of it — which is the fact that stops a build
    /// with no camera from reading `receiving_frames` as its own.
    func testTheSourceBlockIsDecodedIncludingTheFramesThatNeverReachedTheLab() throws {
        let source = try liveStatus().source
        XCTAssertTrue(source.receivingFrames)
        XCTAssertEqual(source.clientsConnected, 1)
        XCTAssertEqual(source.framesOfferedTotal, 16)
        // Without this figure, a phone sending garbage reads exactly like a
        // phone sending nothing — and those need opposite fixes.
        XCTAssertEqual(source.framesRejectedBeforeLab, 3)
        XCTAssertEqual(source.idleAfterSeconds, 5.0)
    }

    // MARK: - 3. Metrics, and the nulls that are not zeroes

    /// **`value: null` is "no meaningful aggregate", never zero.**
    ///
    /// `dominant_direction_deg` is circular, so the Tower reports it
    /// `unaggregated` with a null value rather than averaging 179 and -179 into
    /// 0 — the one direction neither frame was moving in. A client that
    /// defaulted the null to zero would draw exactly that number.
    func testAnUnaggregatedMetricHasNoValueAndSaysWhy() throws {
        let run = try XCTUnwrap(liveStatus().run)
        let direction = try XCTUnwrap(run.metrics.first { $0.label == "dominant_direction_deg" })

        XCTAssertNil(direction.value)
        XCTAssertNil(direction.displayValue, "a null aggregate was rendered as a number")
        XCTAssertEqual(direction.aggregation, "unaggregated")
        XCTAssertEqual(
            direction.unavailableReason,
            "No meaningful average across frames, so the Tower reports none."
        )
    }

    /// The headline is the experiment's most important number and the Tower
    /// puts it first. The order is preserved rather than re-sorted.
    func testTheHeadlineMetricIsFirstAndIsMarkedAsSuch() throws {
        let run = try XCTUnwrap(liveStatus().run)
        let first = try XCTUnwrap(run.metrics.first)
        XCTAssertEqual(first.label, "median_flow_px")
        XCTAssertTrue(first.isHeadline)
        XCTAssertEqual(first.unit, "px")
        XCTAssertEqual(first.aggregation, "rate")
        XCTAssertEqual(first.frames, 3)
        XCTAssertEqual(first.provenance, .measured)
        XCTAssertNil(first.unavailableReason, "a metric with a value claimed to have none")
    }

    /// A unit the Tower did not send is omitted, not substituted.
    func testAMissingUnitIsOmittedRatherThanInvented() {
        let bare = CVMetric(label: "count", value: 12, provenance: .measured)
        XCTAssertEqual(bare.displayValue, "12")
        let withUnit = CVMetric(label: "latency", value: 12, unit: "ms", provenance: .measured)
        XCTAssertEqual(withUnit.displayValue, "12 ms")
    }

    /// A `constant` that was not constant reports null with a *different*
    /// reason, so the null is not read as "never observed".
    func testAVariedConstantExplainsItsNullSeparately() {
        let varied = CVMetric(
            label: "frame_width", value: nil, provenance: .measured,
            aggregation: "constant", varied: true
        )
        XCTAssertNil(varied.displayValue)
        XCTAssertEqual(
            varied.unavailableReason,
            "Reported as constant, but it varied during the run, so no single value describes it."
        )
    }

    /// Provenance is required and never omitted, and silence is not "measured".
    ///
    /// Rule 16. A document this build could not read reaches `.unknown`, which
    /// carries a caveat, rather than defaulting to the reassuring answer.
    func testProvenanceIsReadFromTheWireAndSilenceIsNotMeasured() {
        XCTAssertEqual(CVWireProvenance.read("measured"), .measured)
        XCTAssertEqual(CVWireProvenance.read("inferred"), .inferred(confidence: nil))
        XCTAssertEqual(CVWireProvenance.read(nil), .unknown)
        XCTAssertEqual(CVWireProvenance.read("something else"), .unknown)
        XCTAssertNotNil(ObservationProvenance.unknown.caveat, "an unstated provenance said nothing")
        // `confidence` is always null on this wire — the Tower has no
        // calibrated confidence for any of these — so an inference here can
        // only ever be built without one, and the caveat says so in words
        // rather than as a percentage it does not have.
        XCTAssertNil(ObservationProvenance.inferred(confidence: nil).confidence)
    }

    // MARK: - 4. The run, and null-not-zero

    func testTheRunDecodesItsCountersTimingsAndThroughput() throws {
        let run = try XCTUnwrap(liveStatus().run)

        XCTAssertEqual(run.runID, "f863dcc35bce-6")
        XCTAssertEqual(run.origin, "client_request")
        XCTAssertNil(run.endedAt, "an open run reported an end")
        // Derived by the Tower as processed + refused + failed, holding at
        // every read — which is what makes a dead start diagnosable.
        XCTAssertEqual(run.framesOffered, 3)
        XCTAssertEqual(run.framesProcessed, 3)
        XCTAssertEqual(run.framesRefused, 0)
        XCTAssertEqual(run.framesFailed, 0)
        XCTAssertEqual(run.timings.processingMs, 18.4667)
        XCTAssertEqual(run.timings.processingMsMax, 43.6125)
        // An open map whose keys are the experiment's own. Read, never switched
        // on.
        XCTAssertEqual(Set(run.timings.stageMs.keys), ["decode", "seed", "summarize", "track"])
        XCTAssertEqual(run.throughput.processedFps, 35.754)
        XCTAssertEqual(run.throughput.capacityFps, 54.15)
        XCTAssertEqual(run.metricsOmitted, 0)
        // Empty is the only correct value, and a Tower test enforces it for
        // every registered experiment.
        XCTAssertTrue(run.unclassifiedMetrics.isEmpty)
        XCTAssertTrue(run.hasMeasuredAnything)
    }

    /// **On a run that has measured nothing, the Tower publishes null, not
    /// zero** — which is every Release build and every Tower nobody has
    /// streamed to yet. A rate over a zero-length window is undefined, not
    /// slow.
    func testARunThatMeasuredNothingReportsNullRatherThanZero() throws {
        var json = try XCTUnwrap(decode(Self.liveStatusJSON))
        var run = try XCTUnwrap(json["run"] as? [String: Any])
        run["frames_offered"] = 0
        run["frames_processed"] = 0
        run["metrics"] = []
        run["timings"] = [
            "processing_ms": NSNull(), "processing_ms_max": NSNull(),
            "stage_ms": [String: Any](), "observed_at": NSNull(),
            "time_basis": "tower-receipt",
        ]
        run["throughput"] = [
            "processed_fps": NSNull(), "offered_fps": NSNull(), "capacity_fps": NSNull(),
        ]
        json["run"] = run

        let decoded = try XCTUnwrap(CVLabStatus(json: json)?.run)
        XCTAssertNil(decoded.timings.processingMs)
        XCTAssertNil(decoded.timings.processingMsMax)
        XCTAssertNil(decoded.throughput.processedFps)
        XCTAssertNil(decoded.throughput.capacityFps)
        XCTAssertNil(decoded.timings.time.observedAt)
        XCTAssertFalse(decoded.hasMeasuredAnything)
        // And the observation time is never filled in from the phone's clock:
        // an iOS `Date()` at decode time is arrival time wearing observation
        // time's label.
        XCTAssertTrue(decoded.timings.time.isObservationTimeUnknown)
    }

    /// `0` annotations is "found nothing" and must not merge with "did not
    /// say" — and the artifact carries a descriptor whose treatment is stated,
    /// which is why the treatment is read rather than the presence assumed.
    func testTheAnnotationBlockKeepsSilenceAndZeroApart() throws {
        let annotation = try XCTUnwrap(liveStatus().run?.annotation)
        XCTAssertNil(annotation.count)
        XCTAssertEqual(
            annotation.countUnavailableReason, "this experiment reports no annotation count"
        )
        // A descriptor, and therefore no reason: the two are mutually
        // exclusive and a document with both would be a Tower disagreeing with
        // itself.
        XCTAssertEqual(annotation.artifact, .notFetched(.rawEphemeral))
        XCTAssertNil(annotation.artifactUnavailableReason)
        XCTAssertTrue(CVAnnotationReport(count: 0).hasReport)
        XCTAssertFalse(CVAnnotationReport().hasReport)
    }

    // MARK: - 4a. The live preview

    /// The descriptor is read, and the treatment governs whether it is drawn.
    func testThePreviewDescriptorIsReadOffTheArtifactBlock() throws {
        let preview = try XCTUnwrap(liveStatus().run?.annotation.preview)
        XCTAssertEqual(preview.contract, ExperimentalCVContract.preview)
        XCTAssertEqual(preview.visualKind, "flow_tracks")
        XCTAssertEqual(preview.redaction, .rawEphemeral)
        XCTAssertEqual(preview.faceFilter, "none")
        XCTAssertEqual(preview.path, "/cv-lab/preview")
        XCTAssertEqual(preview.runID, "5a4a5f01ac52-2")
        XCTAssertEqual(preview.maxAgeSeconds, 2.0)
        XCTAssertEqual(preview.pollIntervalSeconds, 0.1)
        XCTAssertTrue(preview.isDrawable)
        XCTAssertNil(preview.withheldReason)
        XCTAssertNotNil(try liveStatus().run?.annotation.drawablePreview)
    }

    /// **The privacy gate, from the strict side.** An unstated treatment is
    /// not a treatment, and the picture is not drawn.
    func testAPreviewWithNoStatedTreatmentIsNotDrawn() throws {
        let preview = try XCTUnwrap(
            CVLivePreview(json: [
                "contract": ExperimentalCVContract.preview,
                "visual_kind": "edge_map",
                "path": "/cv-lab/preview",
            ])
        )
        XCTAssertEqual(preview.redaction, .unknown)
        XCTAssertFalse(preview.isDrawable)
        XCTAssertNotNil(preview.withheldReason)
    }

    /// A treatment word this build has never heard of is `.unknown`, not a
    /// reason to guess. Same rule as `CVWireProvenance`.
    func testAnUnrecognisedTreatmentIsHandledAsStrictlyAsRaw() {
        XCTAssertEqual(CVWireRedaction.read("probably_safe"), .unknown)
        XCTAssertEqual(CVWireRedaction.read(nil), .unknown)
        XCTAssertEqual(CVWireRedaction.read("raw_ephemeral"), .rawEphemeral)
        XCTAssertEqual(CVWireRedaction.read("redacted"), .redacted)
        // `rawEphemeral` is live-displayable and NOT persisted-displayable.
        // Both halves matter: the first is what makes this feature possible
        // and the second is what keeps it from leaking onto a stored surface.
        XCTAssertTrue(RedactionState.rawEphemeral.isDisplayableLive)
        XCTAssertFalse(RedactionState.rawEphemeral.isDisplayableWhenPersisted)
        XCTAssertFalse(RedactionState.unknown.isDisplayableLive)
    }

    /// A preview offered under a contract this build does not implement is
    /// refused with a sentence, not drawn hopefully.
    func testAPreviewFromAFutureContractIsNotDrawn() throws {
        let preview = try XCTUnwrap(
            CVLivePreview(json: [
                "contract": "experimental_cv.preview/2099-01-01",
                "visual_kind": "hologram",
                "path": "/cv-lab/preview",
                "treatment": "raw_ephemeral",
            ])
        )
        XCTAssertFalse(preview.isDrawable)
        XCTAssertTrue(try XCTUnwrap(preview.withheldReason).contains("Update the app"))
    }

    /// A block with no path cannot be fetched, so it is not a descriptor.
    func testAnArtifactBlockWithNothingToFetchIsNotDecoded() {
        XCTAssertNil(
            CVLivePreview(json: [
                "contract": ExperimentalCVContract.preview, "visual_kind": "edge_map",
            ])
        )
        XCTAssertNil(CVLivePreview(json: [:]))
    }

    /// A run with no artifact at all reports the absence and its reason,
    /// exactly as it did before this contract existed.
    func testAnExperimentWithNoPictureStillSaysWhy() throws {
        let annotation = CVAnnotationReport(json: [
            "count": NSNull(),
            "artifact": NSNull(),
            "artifact_unavailable_reason":
                "this experiment produces no visual output, so there is no picture to serve",
        ])
        XCTAssertEqual(annotation.artifact, .absent)
        XCTAssertNil(annotation.preview)
        XCTAssertNil(annotation.drawablePreview)
        XCTAssertNotNil(annotation.artifactUnavailableReason)
    }

    /// The catalog says which experiments have one, before anything is armed.
    func testTheCatalogSaysWhichExperimentsHaveALiveView() throws {
        let available = try liveStatus().available
        XCTAssertFalse(available.isEmpty)
        // Not asserted per id: the Tower owns the registry and this app holds
        // no list. What is asserted is that the field is READ rather than
        // dropped, and that `nil` survives as `nil`.
        let decoded = try XCTUnwrap(
            CVExperiment(json: [
                "id": "edge_detection", "name": "Edge detection",
                "available": true, "preview_kind": "edge_map",
            ])
        )
        XCTAssertEqual(decoded.previewKind, "edge_map")
        XCTAssertTrue(decoded.hasLiveView)

        let blind = try XCTUnwrap(
            CVExperiment(json: ["id": "baseline", "name": "Baseline", "available": true])
        )
        XCTAssertNil(blind.previewKind)
        XCTAssertFalse(blind.hasLiveView)
    }

    /// Where a number sits in this run's range, and never a verdict.
    func testAMetricPlacesItselfInThisRunsRangeWithoutJudgingIt() throws {
        let metric = try XCTUnwrap(
            CVMetric(json: [
                "label": "sharpness_laplacian_var", "value": 402.0,
                "provenance": "measured", "aggregation": "rate", "frames": 767,
                "latest": 1240.0, "observed_min": 79.0, "observed_max": 1309.0,
            ])
        )
        let note = try XCTUnwrap(metric.rangeNote)
        XCTAssertTrue(note.contains("this run's range"))
        XCTAssertTrue(note.contains("near the high end of"))
        // The words this app must never produce from an uncalibrated metric.
        for verdict in ["Good", "Blurry", "Poor", "Sharp", "Bad"] {
            XCTAssertFalse(note.contains(verdict), "\(verdict) is a verdict")
        }
    }

    /// One frame is not a range, and a range of one value is not a placement.
    func testAMetricWithNoRangeYetSaysNothingRatherThanSayingMiddle() throws {
        let single = try XCTUnwrap(
            CVMetric(json: [
                "label": "x", "value": 1.0, "provenance": "measured",
                "aggregation": "rate", "frames": 1,
                "latest": 1.0, "observed_min": 1.0, "observed_max": 1.0,
            ])
        )
        XCTAssertNil(single.rangeNote)

        let counted = try XCTUnwrap(
            CVMetric(json: [
                "label": "y", "value": 5.0, "provenance": "measured",
                "aggregation": "count", "frames": 9,
            ])
        )
        XCTAssertNil(counted.latest)
        XCTAssertNil(counted.rangeNote)
    }

    /// The preview's own cost is read, and kept apart from the experiment's.
    func testThePreviewDiagnosticsAreReadAndKeptApartFromTheExperimentsTimings()
        throws
    {
        let run = try XCTUnwrap(liveStatus().run)
        // Absent on this fixture, which is itself the assertion worth making:
        // a Tower that reports no preview block leaves this `nil` rather than
        // producing zeros that would read as measurements.
        XCTAssertNil(run.preview)

        let stats = CVPreviewDiagnostics(json: [
            "captured": 120, "skipped_by_throttle": 640, "replaced_unread": 108,
            "encoded": 12, "encode_failures": 0, "served": 14, "not_modified": 2,
            "render_ms": 1.8, "render_ms_max": 6.4, "payload_bytes": 9216,
        ])
        XCTAssertEqual(stats.captured, 120)
        XCTAssertEqual(stats.skippedByThrottle, 640)
        XCTAssertEqual(stats.renderMs, 1.8)
        // The experiment's own timings are untouched by any of it.
        XCTAssertEqual(run.timings.processingMs, 18.4667)
        XCTAssertNotNil(stats.deliveryNote)
        XCTAssertNil(CVPreviewDiagnostics(json: [:]).deliveryNote)
    }

    // MARK: - 5. Projection onto this app's state machine

    /// Seven Tower states onto six cases, with the one deliberate rename.
    func testEveryLifecycleStateProjectsOntoTheRightCase() throws {
        XCTAssertEqual(TowerExperimentalCVClient.project(try status(state: "idle")).phase, .idle)
        XCTAssertEqual(
            TowerExperimentalCVClient.project(try status(state: "starting")).phase, .waiting
        )
        XCTAssertEqual(
            TowerExperimentalCVClient.project(try status(state: "running")).phase, .live
        )
        XCTAssertEqual(
            TowerExperimentalCVClient.project(try status(state: "paused")).phase, .settled
        )
        XCTAssertEqual(
            TowerExperimentalCVClient.project(try status(state: "stopped")).phase, .settled
        )
        XCTAssertEqual(
            TowerExperimentalCVClient.project(try status(state: "failed")).phase, .failed
        )
        XCTAssertEqual(
            TowerExperimentalCVClient.project(try status(state: "unavailable")).phase,
            .unsupported
        )

        // The Tower says `stopped` and this says `.completed`, deliberately: a
        // bench run does not complete, it is stopped by a person. The Tower
        // says what happened and iOS renders it with the case it has.
        if case .completed = TowerExperimentalCVClient.project(try status(state: "stopped")) {
        } else {
            XCTFail("`stopped` did not project onto `.completed`")
        }
    }

    /// `paused` and `stopped` are different states because the difference is
    /// real, and a client that collapsed them would offer a Resume that costs
    /// a model reload — or withhold one that costs nothing.
    func testPausedIsItsOwnCaseAndIsNotCompleted() throws {
        let paused = TowerExperimentalCVClient.project(try status(state: "paused"))
        let stopped = TowerExperimentalCVClient.project(try status(state: "stopped"))
        XCTAssertNotEqual(paused, stopped)
        guard case .paused(let run) = paused else { return XCTFail("expected `.paused`") }
        XCTAssertEqual(run.runID, "f863dcc35bce-6")
        // A paused run is not running, so nothing may draw it as live.
        XCTAssertFalse(paused.isRunning)
        XCTAssertFalse(paused.isLive(isStreaming: true, isReceivingFrames: true))
        // But it does carry data, which is the half `.idle` would lose.
        XCTAssertTrue(paused.phase.mayCarryData)
        XCTAssertNotNil(paused.run)
    }

    /// A failed start carries the Tower's own reason and stays recoverable.
    ///
    /// **There is no `start_failed` message**: an arm is asynchronous, so by
    /// the time a load fails the command has already been answered `accepted`.
    /// The outcome arrives here, as state, on a later document — which is why a
    /// client that only sends commands never learns that a start failed.
    func testAFailedStartArrivesAsStateWithTheTowersReason() throws {
        let failed = TowerExperimentalCVClient.project(
            try status(state: "failed", reason: "could not fetch MiDaS weights: no route to host")
        )
        guard case .failed(let failure) = failed else { return XCTFail("expected `.failed`") }
        XCTAssertEqual(failure.kind, .towerReportedFailure)
        XCTAssertEqual(failure.message, "could not fetch MiDaS weights: no route to host")
        XCTAssertNil(failed.run, "a failed start produced a run")
    }

    /// An eighth state is a failure to understand, not an invitation to press
    /// Start against a Lab that may be doing anything at all.
    func testAnUnknownLifecycleStateIsNotCollapsedIntoIdle() throws {
        let unknown = TowerExperimentalCVClient.project(try status(state: "quiescing"))
        guard case .failed(let failure) = unknown else {
            return XCTFail("an unknown state was rendered as something familiar")
        }
        XCTAssertEqual(failure.kind, .undecodableResponse)
        XCTAssertTrue(failure.message.contains("quiescing"))
    }

    /// A document with no lifecycle is not a document. Rendering "idle" for one
    /// this build could not read would be the fabricated answer the whole
    /// cartridge layer exists to prevent.
    func testADocumentWithNoLifecycleIsRefusedRatherThanDefaulted() {
        XCTAssertNil(CVLabStatus(json: ["contract": ExperimentalCVContract.status]))
    }

    // MARK: - 6. Live means both halves

    /// > `.running` may be shown as LIVE only when this build is itself
    /// > streaming **and** `source.receiving_frames` is true.
    ///
    /// `source` is Tower-wide, so it reads `true` for a Release build with no
    /// camera whenever a second phone is attached — and this phone's own
    /// bracket is the half that catches that.
    func testLiveRequiresBothHalvesAndNeitherAlone() throws {
        let running = TowerExperimentalCVClient.project(try status(state: "running"))
        XCTAssertTrue(running.isRunning)

        XCTAssertTrue(running.isLive(isStreaming: true, isReceivingFrames: true))
        // The Release case exactly: the Lab is running and somebody else's
        // phone is feeding it.
        XCTAssertFalse(
            running.isLive(isStreaming: false, isReceivingFrames: true),
            "a build that is not streaming claimed the Tower's frames as its own"
        )
        // And the other way: streaming to a Lab that has seen nothing for five
        // seconds is not producing results either.
        XCTAssertFalse(running.isLive(isStreaming: true, isReceivingFrames: false))
        XCTAssertFalse(running.isLive(isStreaming: false, isReceivingFrames: false))

        // A paused or stopped Lab is never live however the frames are flowing.
        let paused = TowerExperimentalCVClient.project(try status(state: "paused"))
        XCTAssertFalse(paused.isLive(isStreaming: true, isReceivingFrames: true))
    }

    /// The Tower's own diagnosis table, which is what "I pressed Start and
    /// nothing happened" is really asking.
    func testTheFrameCountersDiagnoseADeadStart() throws {
        let running = TowerExperimentalCVClient.project(try status(state: "running"))
        let source = try liveStatus().source
        XCTAssertEqual(running.diagnosis(source: source), .measuring)

        let nothing = CVLabStatus.Source(json: [
            "receiving_frames": false, "frames_offered_total": 0,
            "frames_rejected_before_lab": 0,
        ])
        var json = try XCTUnwrap(decode(Self.liveStatusJSON))
        var run = try XCTUnwrap(json["run"] as? [String: Any])
        run["frames_offered"] = 0
        run["frames_processed"] = 0
        json["run"] = run
        let idleRun = TowerExperimentalCVClient.project(try XCTUnwrap(CVLabStatus(json: json)))
        XCTAssertEqual(idleRun.diagnosis(source: nothing), .nothingArriving)

        // The two that look identical without `frames_rejected_before_lab`, and
        // need opposite fixes.
        let garbage = CVLabStatus.Source(json: [
            "receiving_frames": true, "frames_offered_total": 0,
            "frames_rejected_before_lab": 12,
        ])
        XCTAssertEqual(idleRun.diagnosis(source: garbage), .arrivingButUndecodable(count: 12))
    }

    // MARK: - 7. Refusals

    private func refusal(_ json: String) throws -> CVLabControlRefusal {
        try XCTUnwrap(CVLabControlRefusal(json: try XCTUnwrap(decode(json))))
    }

    /// **Terminal and transient are different instructions to a person**, and
    /// the Tower separated them on purpose: telling somebody to give up on a
    /// working Tower is worse than telling them to try again.
    func testTheEightRefusalsAreClassifiedRatherThanMerged() throws {
        let terminal = try refusal(
            #"{"reason":"lab_unavailable","message":"this Tower runs no CV Lab"}"#
        )
        XCTAssertEqual(terminal.disposition, .terminal)

        for reason in ["internal_error", "lab_busy"] {
            let transient = try refusal(#"{"reason":"\#(reason)","message":"x"}"#)
            XCTAssertEqual(
                transient.disposition, .transient,
                "\(reason) was not treated as retryable"
            )
        }

        for reason in [
            "malformed_request", "unknown_experiment", "experiment_unavailable",
            "invalid_state", "stale_run",
        ] {
            let refused = try refusal(#"{"reason":"\#(reason)","message":"x"}"#)
            XCTAssertEqual(
                refused.disposition, .requestRefused,
                "\(reason) was classified as something a retry or a giving-up would fix"
            )
        }

        // A ninth reason must reach a person as itself rather than being
        // collapsed into terminal or retryable, neither of which is safe to
        // assume.
        let ninth = try refusal(#"{"reason":"some_new_reason","message":"x"}"#)
        XCTAssertEqual(ninth.disposition, .unrecognised)
        XCTAssertEqual(ninth.reason, "some_new_reason")
    }

    /// The reason-specific extras, from live Tower replies.
    func testARefusalCarriesTheExtrasThatMakeItActionable() throws {
        let unknown = try refusal(
            """
            {"reason":"unknown_experiment","message":"this Tower has no experiment 'nope'",
             "command":"cv_lab_start","request_id":"cv-2",
             "available":["baseline","depth","edge_detection"],
             "status":{"lifecycle":{"state":"running","run_id":"f863dcc35bce-1"}}}
            """
        )
        XCTAssertEqual(unknown.available, ["baseline", "depth", "edge_detection"])
        XCTAssertEqual(unknown.requestID, "cv-2")
        // The unchanged document, so a refused client never has to guess what
        // state it is now in.
        XCTAssertEqual(unknown.decodedStatus?.lifecycle.runID, "f863dcc35bce-1")

        let stale = try refusal(
            """
            {"reason":"stale_run",
             "message":"run 'bogus-99' is not the current run; the Lab is now on 'f863dcc35bce-1'",
             "command":"cv_lab_pause","current_run_id":"f863dcc35bce-1"}
            """
        )
        XCTAssertEqual(stale.currentRunID, "f863dcc35bce-1")
        XCTAssertEqual(stale.command, "cv_lab_pause")

        let unavailable = try refusal(
            """
            {"reason":"experiment_unavailable",
             "message":"'depth' needs the optional [ml] extra (torch), which is not installed on this Tower",
             "command":"cv_lab_start","experiment_id":"depth"}
            """
        )
        XCTAssertEqual(unavailable.experimentID, "depth")
    }

    // MARK: - 8. Frame refusals

    private func frameRefusal(_ reason: String, message: String = "…") throws
        -> TowerFrameRefusal
    {
        try XCTUnwrap(TowerFrameRefusal(json: ["seq": 90, "reason": reason, "message": message]))
    }

    /// Nine reasons on one field, and six of them are the Lab saying "not now".
    ///
    /// **A refusal is not a processing error.** The Tower counts these under
    /// `frames_rejected` and deliberately not under `frame_processing_errors` —
    /// a Lab paused for five minutes has not failed hundreds of times — and
    /// this side owes the same reading.
    func testTheSixLabReasonsAreRefusalsAndTheThreeTransportOnesAreNot() throws {
        let refusals = [
            "cv_lab_idle", "cv_lab_starting", "cv_lab_paused", "cv_lab_stopped",
            "cv_lab_failed", "cv_lab_unavailable",
        ]
        for reason in refusals {
            XCTAssertTrue(
                try frameRefusal(reason).kind.isRefusal,
                "\(reason) was treated as a failure rather than as a refusal"
            )
        }
        for reason in ["invalid_frame", "frame_skipped", "module_unavailable"] {
            XCTAssertFalse(
                try frameRefusal(reason).kind.isRefusal,
                "\(reason) was treated as a deliberate refusal"
            )
        }
        // A tenth reason is neither claimed as a refusal nor rendered with a
        // sentence this app invented.
        let unknown = try frameRefusal("something_new")
        XCTAssertEqual(unknown.kind, .unrecognised)
        XCTAssertNil(unknown.kind.summary, "an unrecognised reason got an invented explanation")
    }

    /// **`cv_lab_starting` is arming, not an error**, and the sentence says so
    /// along with the only two facts there are about it: it is bounded at 120
    /// seconds, and there is no progress to report because `torch.hub` offers
    /// none.
    func testArmingIsDescribedAsArmingAndCarriesItsBound() throws {
        let arming = try frameRefusal(
            "cv_lab_starting",
            message: "the CV Lab is arming an experiment; frames are refused until it is ready"
        )
        XCTAssertEqual(arming.kind, .labArming)
        let summary = try XCTUnwrap(arming.kind.summary)
        XCTAssertTrue(summary.contains("120"), "the arm bound was not stated")
        XCTAssertTrue(
            summary.lowercased().contains("no progress"),
            "the absence of progress reporting was not stated"
        )
        // The Tower's own message is still available beside it — only the Tower
        // knows which experiment is loading.
        XCTAssertTrue(arming.message.contains("arming"))
    }

    /// The transport's own codes are rendered in the Tower's words, because
    /// only the Tower knows which field was malformed or which module is
    /// missing.
    func testTransportRefusalsAreLeftInTheTowersWords() throws {
        XCTAssertNil(try frameRefusal("invalid_frame").kind.summary)
        XCTAssertNil(try frameRefusal("module_unavailable").kind.summary)
    }

    // MARK: - 9. The frame reading

    private func frameResult(runID: String, resultSeq: Int, captureIndex: Int)
        -> TowerFrameResult
    {
        TowerFrameResult(
            sequence: captureIndex,
            meanIntensity: nil,
            processingMs: 0.33,
            resultValue: 0.107,
            resultLabel: "edge_density",
            stageMs: [:],
            metrics: [:],
            cvLab: TowerFrameResultProvenance(json: [
                "contract": ExperimentalCVContract.frameResult,
                "tower_instance_id": "f863dcc35bce",
                "run_id": runID,
                "result_seq": resultSeq,
                "experiment_id": "edge_detection",
                "experiment_name": "Edge detection",
                "provenance": "measured",
                "backend": "opencv",
                "device_requested": "auto",
                "result_label": "edge_density",
                "processing_ms": 0.33,
                "tower_received_at": 1787833032.32,
                "time_basis": "tower-receipt",
            ])
        )
    }

    /// **The wire `seq` is the phone's capture index and cannot order
    /// results.** The sender forwards one frame in thirty by design, so it
    /// skips; `result_seq` is the Tower's dense counter within the run, and it
    /// is the one that says whether a reading is newer.
    ///
    /// This reading used to be drawn as "From frame 30", from the capture
    /// index, under a caption that implied it was a position in a sequence.
    func testAReadingSeparatesTheRunsResultCounterFromTheCaptureIndex() {
        let reading = CVFrameReading(
            frameResult(runID: "f863dcc35bce-2", resultSeq: 2, captureIndex: 60)
        )
        XCTAssertEqual(reading.resultSeq, 2, "the run's dense counter was lost")
        XCTAssertEqual(reading.captureIndex, 60)
        XCTAssertNotEqual(
            reading.resultSeq, reading.captureIndex,
            "the fixture must actually exercise the two being different numbers"
        )
        XCTAssertEqual(reading.runID, "f863dcc35bce-2")
        // The sentence this screen could not say before: the reply named the
        // number and not the experiment.
        XCTAssertEqual(reading.experimentName, "Edge detection")
        XCTAssertEqual(reading.experimentID, "edge_detection")
        XCTAssertEqual(reading.headline?.label, "edge_density")
        XCTAssertNotNil(reading.towerReceivedAt)
    }

    /// Provenance now comes off the wire, and a `.measured` reply owes no
    /// caveat at all — which is the difference the `cv_lab` block bought.
    func testAReadingTakesItsProvenanceFromTheWireAndFallsBackWhenThereIsNone() {
        let attributed = CVFrameReading(
            frameResult(runID: "f863dcc35bce-2", resultSeq: 1, captureIndex: 30)
        )
        XCTAssertEqual(attributed.provenance, .measured)
        XCTAssertNil(attributed.provenance.caveat)

        // A Tower running no Lab attaches no block, and Rule 16 does not permit
        // silence to be read as "measured".
        let unattributed = CVFrameReading(
            TowerFrameResult(
                sequence: 30, meanIntensity: 0.5, processingMs: 1.0, resultValue: nil,
                resultLabel: nil, stageMs: [:], metrics: [:], cvLab: nil
            )
        )
        XCTAssertEqual(unattributed.provenance, .unknown)
        XCTAssertEqual(unattributed.provenance, CVFrameReading.provenance)
        XCTAssertNotNil(unattributed.provenance.caveat)
        XCTAssertNil(unattributed.resultSeq, "a reply with no provenance block invented one")
        XCTAssertNil(unattributed.runID)
    }

    /// The pair rule: a number and the Tower's own name for it, or neither.
    func testTheHeadlinePairIsAllOrNothing() {
        XCTAssertNil(CVFrameReading.Labelled(label: nil, value: 0.4))
        XCTAssertNil(CVFrameReading.Labelled(label: "edges", value: nil))
        XCTAssertNil(CVFrameReading.Labelled(label: "   ", value: 0.4))
        XCTAssertNotNil(CVFrameReading.Labelled(label: "edges", value: 0.4))
    }
}


// MARK: - The bytes, not the descriptor

/// Answers every request from a script, so the preview client's real decode
/// path can be exercised without a Tower.
///
/// `nonisolated(unsafe)` because `URLProtocol` calls `startLoading()` off the
/// main actor and this project defaults types to `@MainActor`. Test-only, and
/// each test sets the script before it makes its one request.
private final class CVPreviewStubProtocol: URLProtocol {
    struct Answer {
        var status = 200
        var headers: [String: String] = [:]
        var body = Data()
    }

    nonisolated(unsafe) static var answer = Answer()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let scripted = Self.answer
        let response = HTTPURLResponse(
            url: request.url ?? URL(string: "http://stub.invalid")!,
            statusCode: scripted.status,
            httpVersion: "HTTP/1.1",
            headerFields: scripted.headers
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: scripted.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// **The privacy gate that governs pixels, and the run identity that governs
/// which pixels.**
///
/// Everything in `CVLabContractTests` above asserts against the *descriptor* —
/// the status document's account of what a preview would be. Nothing there
/// touches the path that actually carries an image, which is where the
/// treatment header is read and where a frame is accepted or discarded. These
/// do.
@MainActor
final class CVLabPreviewBytesTests: XCTestCase {

    private func makeClient() -> CVLivePreviewHTTPClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CVPreviewStubProtocol.self]
        return CVLivePreviewHTTPClient(
            baseURL: URL(string: "http://tower.invalid")!,
            session: URLSession(configuration: configuration),
            timeout: 2.0
        )
    }

    /// A 2x2 PNG. Real bytes, so `UIImage(data:)` genuinely decodes.
    private func tinyPNG() -> Data {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 2, height: 2))
        return renderer.image { context in
            UIColor.white.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
        }.pngData()!
    }

    private func serve(headers: [String: String]) {
        CVPreviewStubProtocol.answer = .init(
            status: 200, headers: headers, body: tinyPNG()
        )
    }

    private var drawablePreview: CVLivePreview {
        CVLivePreview(json: [
            "contract": ExperimentalCVContract.preview,
            "visual_kind": "edge_map",
            "path": "/cv-lab/preview",
            "treatment": "raw_ephemeral",
            "poll_interval_s": 0.05,
        ])!
    }

    // MARK: The gate

    /// **Bytes that state no treatment are not drawn.** The descriptor saying
    /// `raw_ephemeral` is not permission: the treatment travels on the image
    /// as well, and an image whose treatment arrives separately is an image
    /// whose treatment can be lost.
    func testBytesWithNoStatedTreatmentAreNotDrawn() async {
        serve(headers: ["ETag": "e1"])
        let outcome = await makeClient().fetch(
            path: "/cv-lab/preview", runID: "r1", ifNoneMatch: nil
        )
        guard case .waiting = outcome else {
            return XCTFail("an unstated treatment must not be drawn: \(outcome)")
        }
    }

    /// A treatment word this build has never heard of is refused as strictly
    /// as raw, on the bytes exactly as in the descriptor.
    func testBytesWithAnUnknownTreatmentAreNotDrawn() async {
        serve(headers: ["X-CV-Preview-Treatment": "probably_safe", "ETag": "e1"])
        let outcome = await makeClient().fetch(
            path: "/cv-lab/preview", runID: "r1", ifNoneMatch: nil
        )
        guard case .waiting = outcome else {
            return XCTFail("an unknown treatment must fail closed: \(outcome)")
        }
    }

    /// The permitted case, so the gate is not merely refusing everything.
    func testRawEphemeralBytesAreDrawn() async {
        serve(headers: [
            "X-CV-Preview-Treatment": "raw_ephemeral",
            "X-CV-Preview-Run": "r1",
            "X-CV-Preview-Seq": "7",
            "ETag": "e1",
        ])
        let outcome = await makeClient().fetch(
            path: "/cv-lab/preview", runID: "r1", ifNoneMatch: nil
        )
        guard case .frame(let frame) = outcome else {
            return XCTFail("raw_ephemeral is the live view's whole purpose: \(outcome)")
        }
        XCTAssertEqual(frame.treatment, .rawEphemeral)
        XCTAssertEqual(frame.runID, "r1")
        XCTAssertEqual(frame.resultSeq, 7)
    }

    // MARK: Run identity

    /// **An unstamped run is "no id", not a run named "".**
    ///
    /// `routes/cv_lab_preview.py` sends `rendered.run_id or ""`, and `lab.py`
    /// builds a preview with `run_id=None` whenever it has no run object. So
    /// "the Tower did not say" reaches this app as `Optional("")` and never as
    /// `nil`. Comparing it raw discarded every frame of such a run — silently,
    /// and at full polling rate, because the discard path never set the
    /// `ETag`, so each rejected frame was a fresh full-body JPEG.
    func testAFrameTheTowerLeftUnstampedIsStillDrawn() async {
        serve(headers: [
            "X-CV-Preview-Treatment": "raw_ephemeral",
            "X-CV-Preview-Run": "",
            "ETag": "e1",
        ])
        let loader = CVLivePreviewLoader(client: makeClient())
        loader.start(drawablePreview, runID: "a-run-the-tower-did-not-stamp")
        await settle { if case .showing = loader.phase { return true } else { return false } }

        guard case .showing = loader.phase else {
            return XCTFail("an empty run header is not a mismatch: \(loader.phase)")
        }
        XCTAssertEqual(loader.framesShown, 1)
        loader.stop()
    }

    /// A frame stamped with a run this loader is not watching is discarded.
    /// The other half of the same guard, so the fix above did not open it.
    func testAFrameFromAnotherRunIsNotDrawn() async {
        serve(headers: [
            "X-CV-Preview-Treatment": "raw_ephemeral",
            "X-CV-Preview-Run": "edge-detection-run",
            "ETag": "e1",
        ])
        let loader = CVLivePreviewLoader(client: makeClient())
        loader.start(drawablePreview, runID: "depth-run")
        await settle { loader.unchangedResponses > 0 || loader.framesShown > 0 }

        XCTAssertEqual(
            loader.framesShown, 0,
            "Depth must never display a frame Edge produced"
        )
        if case .showing = loader.phase {
            XCTFail("a frame from another run reached the screen")
        }
        loader.stop()
    }

    // MARK: Release

    /// **Stop drops the picture, not merely the polling.** A frozen last frame
    /// under a stopped label is still a picture of the wearer's room on
    /// screen, and `raw_ephemeral` is live-view-only in both directions.
    func testStopDropsThePicture() async {
        serve(headers: [
            "X-CV-Preview-Treatment": "raw_ephemeral",
            "X-CV-Preview-Run": "r1",
            "ETag": "e1",
        ])
        let loader = CVLivePreviewLoader(client: makeClient())
        loader.start(drawablePreview, runID: "r1")
        await settle { if case .showing = loader.phase { return true } else { return false } }
        guard case .showing = loader.phase else {
            return XCTFail("nothing was showing, so the release is untested")
        }

        loader.stop()
        XCTAssertEqual(loader.phase, .idle, "the image goes with the loop")
        XCTAssertEqual(loader.framesShown, 0)
    }

    /// A preview whose treatment this build does not understand is withheld
    /// with a sentence, and no request is made at all.
    func testAnUndrawablePreviewIsWithheldWithoutFetching() async {
        let untreated = CVLivePreview(json: [
            "contract": ExperimentalCVContract.preview,
            "visual_kind": "edge_map",
            "path": "/cv-lab/preview",
        ])!
        let loader = CVLivePreviewLoader(client: makeClient())
        loader.start(untreated, runID: "r1")

        guard case .withheld = loader.phase else {
            return XCTFail("an unstated treatment must be withheld: \(loader.phase)")
        }
        XCTAssertEqual(loader.framesShown, 0)
        loader.stop()
    }

    /// Waits for a condition, or gives up. The loader polls on its own task.
    private func settle(
        timeout: TimeInterval = 3.0, until condition: () -> Bool
    ) async {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return }
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    // MARK: - The command in flight: the value

    /// Two records of the same send are the same record, and a different
    /// request id is a different send even for the same button.
    func testPendingCommandEqualityIsByEveryField() {
        let sentAt = Date(timeIntervalSince1970: 1_787_833_032)
        let a = CVLabPendingCommand(command: .start, experimentID: "depth", requestID: "cv-3", sentAt: sentAt)
        let b = CVLabPendingCommand(command: .start, experimentID: "depth", requestID: "cv-3", sentAt: sentAt)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(
            a, CVLabPendingCommand(command: .start, experimentID: "depth", requestID: "cv-4", sentAt: sentAt),
            "a different request id is a different send"
        )
        XCTAssertNotEqual(
            a, CVLabPendingCommand(command: .stop, experimentID: nil, requestID: "cv-3", sentAt: sentAt),
            "a different command is a different send"
        )
        XCTAssertNotEqual(
            a, CVLabPendingCommand(command: .start, experimentID: "depth", requestID: "cv-3", sentAt: sentAt.addingTimeInterval(1)),
            "a different send time is a different send"
        )
    }

    /// A reply that names no request answers no request. The status document
    /// is pushed on the subscription and read on connect without one, and
    /// neither of those may clear a question it was never asked.
    func testAReplyWithoutARequestIDAnswersNothing() {
        let pending = CVLabPendingCommand(command: .pause, experimentID: nil, requestID: "cv-7", sentAt: Date())
        XCTAssertTrue(pending.isAnswered(by: "cv-7"))
        XCTAssertFalse(pending.isAnswered(by: "cv-8"))
        XCTAssertFalse(pending.isAnswered(by: nil))
    }

    func testPendingCommandVerbsAreTheWordsAPersonWouldUse() {
        func verb(_ command: CVLabCommand) -> String {
            CVLabPendingCommand(command: command, experimentID: nil, requestID: "cv-1", sentAt: Date()).verb
        }
        XCTAssertEqual(verb(.start), "start")
        XCTAssertEqual(verb(.pause), "pause")
        XCTAssertEqual(verb(.resume), "resume")
        XCTAssertEqual(verb(.stop), "stop")
        XCTAssertEqual(verb(.status), "status")
    }

    // MARK: - The connection row's four words

    /// Four statuses, four distinct words. Pinned separately from
    /// `StateDisplay.tower(_:)`, which the shell's own test pins, because the
    /// row deliberately says "Disconnected" where the shell says "Offline".
    func testConnectionRowLabelsAreFourDistinctWords() {
        XCTAssertEqual(CVTowerConnectionLabel.text(for: .offline), "Disconnected")
        XCTAssertEqual(CVTowerConnectionLabel.text(for: .connecting), "Connecting")
        XCTAssertEqual(CVTowerConnectionLabel.text(for: .online), "Connected")
        XCTAssertEqual(CVTowerConnectionLabel.text(for: .failed("Could not connect")), "Error")
        let words: Set<String> = [
            CVTowerConnectionLabel.text(for: .offline),
            CVTowerConnectionLabel.text(for: .connecting),
            CVTowerConnectionLabel.text(for: .online),
            CVTowerConnectionLabel.text(for: .failed("x")),
        ]
        XCTAssertEqual(words.count, 4, "two statuses share a word")
    }

    /// The one action per state, the same three `ConnectionSheet` offers.
    func testConnectionRowActionMirrorsTheConnectionSheet() {
        XCTAssertEqual(CVTowerConnectionLabel.actionTitle(for: .offline), "Connect")
        XCTAssertEqual(CVTowerConnectionLabel.actionTitle(for: .failed("x")), "Connect")
        XCTAssertEqual(CVTowerConnectionLabel.actionTitle(for: .connecting), "Cancel")
        XCTAssertEqual(CVTowerConnectionLabel.actionTitle(for: .online), "Disconnect")
    }

    // MARK: - The command in flight: the clearing rules, against a socket

    /// A start records what it sent, with the id that actually went out.
    ///
    /// The wire is asserted, not only the property: a pending id that differed
    /// from the one on the wire could never be cleared by a reply, and the
    /// row would sit "asking" until the bound.
    func testAStartRecordsAPendingCommandWithTheRequestIDItSent() async throws {
        let bench = try await CVLabBench.start()
        defer { bench.stop() }
        await settle { bench.lab.canSendCommands }
        XCTAssertTrue(bench.lab.canSendCommands, "the declaration never made the Lab commandable")

        try bench.lab.run(CVExperiment(id: "edge_detection", name: "Edge detection"))

        let pending = try XCTUnwrap(bench.lab.pendingCommand)
        XCTAssertEqual(pending.command, .start)
        XCTAssertEqual(pending.experimentID, "edge_detection")

        await settle { bench.sent(ofType: "cv_lab_start").count == 1 }
        let wire = try XCTUnwrap(bench.sent(ofType: "cv_lab_start").first)
        XCTAssertEqual(wire["request_id"] as? String, pending.requestID)
        XCTAssertEqual(wire["experiment_id"] as? String, "edge_detection")
    }

    /// A `cv_lab_status` echoing the request id is the answer, and clears it.
    func testAReplyEchoingTheRequestIDClearsThePendingCommand() async throws {
        let bench = try await CVLabBench.start()
        defer { bench.stop() }
        await settle { bench.lab.canSendCommands }
        try bench.lab.run(CVExperiment(id: "edge_detection", name: "Edge detection"))
        let pending = try XCTUnwrap(bench.lab.pendingCommand)

        bench.server.send(
            text: bench.statusReply(runID: "f863dcc35bce-9", requestID: pending.requestID, acceptedCommand: "cv_lab_start")
        )

        await settle { bench.lab.pendingCommand == nil }
        XCTAssertNil(bench.lab.pendingCommand, "the Tower answered and the row is still asking")
        XCTAssertEqual(bench.lab.status?.lifecycle.runID, "f863dcc35bce-9", "the answer's document was not applied")
    }

    /// A `cv_lab_error` echoing the request id is also the answer.
    func testARefusalEchoingTheRequestIDClearsThePendingCommand() async throws {
        let bench = try await CVLabBench.start()
        defer { bench.stop() }
        await settle { bench.lab.canSendCommands }
        try bench.lab.run(CVExperiment(id: "edge_detection", name: "Edge detection"))
        let pending = try XCTUnwrap(bench.lab.pendingCommand)

        bench.server.send(text: bench.refusal(requestID: pending.requestID))

        await settle { bench.lab.pendingCommand == nil }
        XCTAssertNil(bench.lab.pendingCommand, "a refusal is an answer, and the row is still asking")
        XCTAssertEqual(bench.lab.lastRefusal?.reason, "lab_busy")
    }

    /// The two documents that arrive without asking — a push, and somebody
    /// else's reply — leave the question standing.
    func testDocumentsNamingAnotherOrNoRequestLeaveThePendingCommandAlone() async throws {
        let bench = try await CVLabBench.start()
        defer { bench.stop() }
        await settle { bench.lab.canSendCommands }
        try bench.lab.run(CVExperiment(id: "edge_detection", name: "Edge detection"))
        let pending = try XCTUnwrap(bench.lab.pendingCommand)

        // No request id: the shape of a read or a push.
        bench.server.send(text: bench.statusReply(runID: "f863dcc35bce-1", requestID: nil, acceptedCommand: nil))
        // Another client's request id, with an accepted command.
        bench.server.send(
            text: bench.statusReply(runID: "f863dcc35bce-2", requestID: "somebody-else", acceptedCommand: "cv_lab_start")
        )

        // Both documents were applied — the second is the one on record — and
        // the question is still out.
        await settle { bench.lab.status?.lifecycle.runID == "f863dcc35bce-2" }
        XCTAssertEqual(bench.lab.status?.lifecycle.runID, "f863dcc35bce-2")
        XCTAssertEqual(bench.lab.pendingCommand, pending, "a document that named no request, or another's, cleared it")
    }

    /// The reply cannot arrive on a socket that is gone.
    func testLeavingOnlineClearsThePendingCommand() async throws {
        let bench = try await CVLabBench.start()
        defer { bench.stop() }
        await settle { bench.lab.canSendCommands }
        try bench.lab.run(CVExperiment(id: "edge_detection", name: "Edge detection"))
        XCTAssertNotNil(bench.lab.pendingCommand)

        bench.tower.disconnect()

        await settle { bench.lab.pendingCommand == nil }
        XCTAssertNil(bench.lab.pendingCommand, "a pending command survived the socket it was sent on")
    }

    /// No answer within the bound: cleared, and reported where a refused send
    /// is reported — never as a state change.
    func testAnUnansweredCommandExpiresAsARequestFailure() async throws {
        let bench = try await CVLabBench.start(replyBoundSeconds: 0.3)
        defer { bench.stop() }
        let model = ExperimentalCVViewModel(client: bench.lab)
        await settle { bench.lab.canSendCommands }
        let stateBefore = bench.lab.state

        try bench.lab.run(CVExperiment(id: "edge_detection", name: "Edge detection"))
        XCTAssertNotNil(bench.lab.pendingCommand)
        XCTAssertNil(model.lastRequestFailure, "the send went out; nothing has failed yet")

        await settle { bench.lab.pendingCommand == nil }
        XCTAssertNil(bench.lab.pendingCommand, "the reply bound passed and the row is still asking")
        await settle { model.lastRequestFailure != nil }
        let failure = try XCTUnwrap(model.lastRequestFailure)
        XCTAssertEqual(failure.kind, .timedOut)
        XCTAssertTrue(
            failure.message.contains("did not answer the start request"),
            "the failure must name the command: \(failure.message)"
        )
        XCTAssertEqual(bench.lab.state, stateBefore, "a missing reply moved the state machine")
    }
}

/// A mock Tower, a `TowerClient` on it, and a `TowerExperimentalCVClient`
/// wired to that — the three things every clearing-rule test needs — already
/// online and already declaring the Lab, so the client's `canSendCommands`
/// can come true.
///
/// Lives here rather than borrowing `TowerClientTests`' helpers because those
/// are private to that case, and what they encode is small: answer the
/// client's handshake ping with a pong, and record what was sent.
@MainActor
private final class CVLabBench {
    let server: MockTowerServer
    let tower: TowerClient
    let lab: TowerExperimentalCVClient
    private let recorder = MessageRecorder()

    private init(server: MockTowerServer, tower: TowerClient, lab: TowerExperimentalCVClient) {
        self.server = server
        self.tower = tower
        self.lab = lab
    }

    /// Connects, waits for `.online`, and sends the declaration that offers
    /// the Lab under the contract this build implements.
    static func start(
        replyBoundSeconds: TimeInterval = TowerExperimentalCVClient.replyBoundSeconds
    ) async throws -> CVLabBench {
        let server = try MockTowerServer()
        let port = try await server.start()
        let tower = TowerClient(metrics: SenderMetrics())
        let lab = TowerExperimentalCVClient(tower: tower, replyBoundSeconds: replyBoundSeconds)
        let bench = CVLabBench(server: server, tower: tower, lab: lab)
        let recorder = bench.recorder
        server.onText = { text in
            recorder.record(text)
            guard
                let data = text.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                json["type"] as? String == "ping"
            else { return }
            server.send(text: #"{"type":"pong"}"#)
        }
        tower.connect(to: URL(string: "ws://127.0.0.1:\(port)/")!)
        let deadline = Date().addingTimeInterval(3)
        while tower.status != .online, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertEqual(tower.status, .online, "the client never came online against the mock Tower")
        server.send(text: bench.declaration)
        return bench
    }

    func stop() {
        tower.disconnect()
        server.stop()
    }

    /// Every message the client sent of one type, decoded.
    func sent(ofType type: String) -> [[String: Any]] {
        recorder.all.compactMap { text -> [String: Any]? in
            guard let data = text.data(using: .utf8) else { return nil }
            return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        }
        .filter { $0["type"] as? String == type }
    }

    /// The Tower offering the Lab, available, under this build's contract.
    private var declaration: String {
        """
        {"type":"cartridges",
         "envelope_contract":"cartridge_results.envelope/2026-08-23",
         "cartridges":[{"cartridge":"experimental_cv","result_type":"status",
                        "contract":"\(ExperimentalCVContract.status)",
                        "available":true,"unavailable_reason":null,
                        "snapshot_only":true}],
         "not_offered":[],
         "http_contracts":[]}
        """
    }

    /// A `cv_lab_status`, shaped as the Tower sends it, with or without the
    /// two fields only a reply to a command carries.
    func statusReply(runID: String, requestID: String?, acceptedCommand: String?) -> String {
        let request = requestID.map { ",\"request_id\":\"\($0)\"" } ?? ""
        let accepted = acceptedCommand.map { ",\"accepted_command\":\"\($0)\"" } ?? ""
        return """
            {"type":"cv_lab_status",
             "control_contract":"experimental_cv.control/2026-08-27",
             "contract":"\(ExperimentalCVContract.status)"\(request)\(accepted),
             "status":{"contract":"\(ExperimentalCVContract.status)",
               "tower_instance_id":"f863dcc35bce",
               "lifecycle":{"state":"running","reason":null,
                            "since":1787833032.19,"run_id":"\(runID)"},
               "available":[],"selected":"edge_detection",
               "source":{"clients_connected":1,"receiving_frames":true,
                         "last_frame_at":1787833032.32,"frames_offered_total":2,
                         "frames_rejected_before_lab":0,"idle_after_s":5.0}}}
            """
    }

    /// A transient refusal echoing the request, with the unchanged document.
    func refusal(requestID: String) -> String {
        """
        {"type":"cv_lab_error","control_contract":"experimental_cv.control/2026-08-27",
         "reason":"lab_busy","message":"a start is already in flight",
         "command":"cv_lab_start","request_id":"\(requestID)",
         "status":{"contract":"\(ExperimentalCVContract.status)",
           "tower_instance_id":"f863dcc35bce",
           "lifecycle":{"state":"starting","reason":null,
                        "since":1787833032.19,"run_id":"f863dcc35bce-4"},
           "available":[],"selected":"edge_detection",
           "source":{"clients_connected":1,"receiving_frames":false,
                     "last_frame_at":null,"frames_offered_total":0,
                     "frames_rejected_before_lab":0,"idle_after_s":5.0}}}
        """
    }
}
