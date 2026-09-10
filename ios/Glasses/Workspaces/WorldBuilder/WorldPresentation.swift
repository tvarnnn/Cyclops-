//
//  WorldPresentation.swift
//  Glasses
//

import Foundation

// MARK: - Why this file exists

/// The World Builder screen's state, separated from the views that draw it.
///
/// ## The failure this was written after
///
/// On the 2026-09-06 walk the phone showed a world with 463 keyframes, 467
/// camera poses and 17,674 points, and under it the two sentences
/// "Nothing mapped yet" and "The glasses have not mapped anything here yet."
/// Both were drawn from **one** condition — `WorldFragmentsModel.fragments`
/// being empty — and that array is assigned in exactly one place: the end of a
/// successful manifest-plus-chunks fetch (`WorldBuilderViewModel`
/// `geometryDidChange`). It is therefore empty, and those two sentences are
/// therefore drawn, in at least five unrelated situations:
///
/// 1. No fetch was ever attempted, because the status payload carried no
///    `geometry.revision` (`WorldBuilderResultDecoder.geometryCoordinates`
///    requires all three parts of the address).
/// 2. The publish was suppressed by the session gate, so no address was ever
///    emitted for a world that has one.
/// 3. The manifest fetch failed — 404, transport, anything.
/// 4. One manifest row failed to decode, which drops the **whole** manifest
///    (`WorldGeometryDecoder.manifest`, deliberately).
/// 5. The manifest arrived under a pose convention this build does not
///    implement, and was refused.
///
/// None of the five is "the glasses have not mapped anything". Four of them are
/// the phone's own ignorance, and the fifth is the Tower saying it built
/// nothing, which is the only one that sentence describes. There was no state,
/// no string and no Release-visible log distinguishing them — the geometry path
/// logged under `#if DEBUG print` and the field phone was a Release build.
///
/// So the fetch gets a state of its own, `WorldGeometryStatus`, the Tower's own
/// claims get one, `WorldEvidence`, and the sentence is a **pure function of
/// both** (`WorldGeometryAccount`). The rule that failure made necessary is
/// asserted in one place and tested there: a snapshot that carries geometry can
/// never produce the "nothing mapped" account, whatever the fetch did.
///
/// ## Everything here is a pure value
///
/// No view, no client, no `URLSession`, no clock. That is what lets
/// `WorldPresentationTests` drive the whole ladder — including the five
/// situations above — without a Tower and without rendering.
///
/// None of these types is marked `nonisolated`, so under this target's
/// `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor` they are main-actor isolated —
/// deliberately, and to match `WorldFragmentsModel`, `WorldSnapshot` and
/// `WorldModelState`, which they read. `WorldRenderTarget` stays `nonisolated`
/// because it addresses a URL and touches none of them.

// MARK: - The fetch, as a state rather than as an empty array

/// Where the phone is in going and getting the geometry the Tower named.
///
/// The ladder is honest about ignorance: `.notAddressed` is *the phone does not
/// have an address*, `.towerReportsNone` is *the Tower answered that it has
/// nothing*, and those are different answers to the reader even though both
/// leave the gallery empty.
///
/// `.loaded` carries the model **and** how many segments the manifest named
/// whose points could not be fetched, because a gallery drawn from 29 of 34
/// segments is a true picture of part of a world and must say which part is
/// missing rather than reading as the whole.
enum WorldGeometryStatus: Equatable {
    /// There is no world on screen to have geometry for: `.idle`, `.failed`,
    /// `.unsupported`, or nothing reported yet.
    case noWorld
    /// A world is on screen and no geometry address has arrived for it. The
    /// Tower sends `geometry.revision: null` until a build has produced output
    /// for the session, and the client suppresses the address for a report the
    /// session gate refused — this state covers both, and the account below
    /// says only what is true of both.
    case notAddressed
    /// An address arrived and a fetch is out.
    case loading
    /// A manifest arrived, decoded, and its segments are drawn. Zero drawable
    /// segments here is a real answer — the Tower has a manifest and nothing in
    /// it resolved — and is **not** the same as any state above it.
    case loaded(WorldFragmentsModel, unfetched: Int)
    /// The Tower answered 404 on the manifest route: it has no geometry for
    /// this session. Its own words, when it sent any.
    case towerReportsNone(detail: String?)
    /// The fetch happened and did not produce a manifest this build could use.
    case failed(WorldGeometryFailure)

    /// The fragments to draw, or an empty model. Every caller that used to read
    /// `viewModel.fragmentsModel` reads this, so the gallery is unchanged for
    /// the case where geometry exists.
    var fragments: WorldFragmentsModel {
        if case .loaded(let model, _) = self { return model }
        return WorldFragmentsModel(segments: [])
    }

    /// Whether the phone holds geometry it can draw. Distinct from
    /// `WorldEvidence.hasGeometry`, which is what the **Tower** says exists:
    /// the whole 2026-09-06 defect is that those two can disagree.
    var hasDrawableGeometry: Bool {
        if case .loaded(let model, _) = self { return !model.fragments.isEmpty }
        return false
    }

    /// Whether a fetch for this world has produced **any** answer yet — a
    /// manifest, the Tower's own 404, or a failure.
    ///
    /// Deliberately not `hasDrawableGeometry`, and the distinction is the whole
    /// point. A `.loaded` manifest whose segments are all unresolved, or all
    /// resolved without bounds, has no drawable tile — and that is the normal
    /// first minute of a walk and the permanent state of an anchors-only build.
    /// Deciding "show the spinner" from *tiles* therefore restarts the spinner
    /// on every one of the ~30 revisions a minute a live walk produces, on a
    /// world the phone has already successfully fetched. Deciding it from
    /// *having an answer* does not. `WorldBuilderViewModel.geometryDidChange`
    /// is the only caller.
    var hasAnswered: Bool {
        switch self {
        case .noWorld, .notAddressed, .loading: return false
        case .loaded, .towerReportsNone, .failed: return true
        }
    }
}

/// What went wrong going to get the geometry, in the shape the reader is owed.
///
/// A struct rather than a bare string so the diagnostics surface can branch on
/// `kind` while the normal surface shows `message` and nothing else.
struct WorldGeometryFailure: Equatable {
    enum Kind: Equatable {
        /// The request did not complete: no route, no Tower, a dropped link.
        case unreachable
        /// The manifest came back and this build could not read it as
        /// `world_builder.geometry`. Situation 4 above lives here: one bad row
        /// drops the whole manifest by design, and this is what that looks
        /// like from the outside.
        case undecodable
        /// The manifest is in a pose convention this build does not implement.
        /// Refused rather than drawn, because it would draw plausibly and
        /// wrongly.
        case poseConvention
    }

    var kind: Kind
    /// The transport's or the Tower's own words, when there were any. Never
    /// composed here.
    var detail: String?

    /// One sentence for a person. Names the thing that failed and never blames
    /// the world for a link that dropped.
    var message: String {
        switch kind {
        case .unreachable:
            if let detail, !detail.isEmpty {
                return "The Tower's geometry could not be fetched: \(detail)"
            }
            return "The Tower's geometry could not be fetched."
        case .undecodable:
            return "The Tower's geometry could not be read as the contract this build implements. "
                + "One unreadable row refuses the whole manifest, so this is not a claim that the "
                + "world is empty."
        case .poseConvention:
            return "The Tower's geometry is in a pose convention this build does not implement, "
                + "so it is not drawn. Drawing it would look like a room and mean nothing."
        }
    }
}

// MARK: - What the Tower itself claims exists

/// The Tower's own account of what it built, lifted off the snapshot.
///
/// This is the half that makes the 2026-09-06 sentence impossible to draw
/// again. The phone's fetch can fail in five ways; the **snapshot** said 463
/// keyframes, 467 poses and 17,674 points, and that statement is independent of
/// every one of them. Where the two disagree, the reader is told the Tower's
/// figures and told that the phone could not fetch the points behind them —
/// which is what was actually true on that walk.
struct WorldEvidence: Equatable {
    var keyframes: Int?
    /// `world_snapshot.geometry.element_count` — points, on today's Tower, in
    /// whatever unit `representation` names.
    var elements: Int?
    var representation: String?
    /// `trajectory.pose_count`: poses carrying a position that is evidence.
    var poses: Int?
    var segments: Int?

    init(
        keyframes: Int? = nil,
        elements: Int? = nil,
        representation: String? = nil,
        poses: Int? = nil,
        segments: Int? = nil
    ) {
        self.keyframes = keyframes
        self.elements = elements
        self.representation = representation
        self.poses = poses
        self.segments = segments
    }

    /// `nil` where there is no snapshot, so a caller cannot accidentally treat
    /// "no world" as "a world that reported nothing".
    init?(snapshot: WorldSnapshot?) {
        guard let snapshot else { return nil }
        self.init(
            keyframes: snapshot.keyframeCount,
            elements: snapshot.geometry.elementCount,
            representation: snapshot.geometry.representation,
            poses: snapshot.trajectory.poseCount,
            segments: snapshot.trajectory.segments
        )
    }

    /// Whether the **Tower** says this world holds geometry.
    ///
    /// Either figure on its own is enough. A build can position cameras and
    /// recover few points, or recover points across segments whose cameras were
    /// never placed; both are geometry, and requiring both would let the field
    /// case slip back through.
    var hasGeometry: Bool { (elements ?? 0) > 0 || (poses ?? 0) > 0 }

    /// The Tower's figures on one line, in its own vocabulary, or `nil` when it
    /// gave none. `representation` is the Tower's word and is quoted, never
    /// parsed — the rule `WorldSummaryView.geometryValue` already follows.
    var summary: String? {
        var parts: [String] = []
        if let elements {
            parts.append("\(elements) \(representation ?? "elements")")
        } else if let representation {
            parts.append(representation)
        }
        if let poses, poses > 0 { parts.append("\(poses) camera poses") }
        if let keyframes { parts.append("\(keyframes) keyframes") }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}

// MARK: - The sentence, decided in one place

/// What to say where the fragment gallery has nothing to draw.
///
/// **This type is the fix for the 2026-09-06 defect.** The gallery no longer
/// composes its own empty-state prose; it is handed one of these, and the one
/// account that says the glasses mapped nothing is reachable only from a
/// `.loaded` manifest with nothing in it *and* a Tower that reported no
/// geometry. `WorldPresentationTests` asserts exactly that, over the field
/// walk's own figures.
struct WorldGeometryAccount: Equatable {
    /// One short line. The gallery's headline when there is nothing to draw.
    var headline: String
    /// The explanation under it, when there is one worth making.
    var detail: String?
    /// Whether this account is the phone admitting ignorance rather than
    /// reporting an answer. Drives nothing but the diagnostics label today;
    /// present so a caller never has to string-match to find out.
    var isPhoneSideUnknown: Bool

    /// The one account that says nothing was mapped. Spelled once so no other
    /// branch can produce these words by accident.
    ///
    /// **Never a default.** It is produced by `account(for:evidence:)` and by
    /// nothing else. It was briefly the default argument of both
    /// `WorldPresentation.init` and `WorldFragmentsView.account`, which meant
    /// the app's strongest negative claim about a world was what any caller
    /// that forgot to supply an account would say — a preview, a new call
    /// site, a future refactor. The forbidden sentence must not be anyone's
    /// fallback; `undescribed` is.
    static let nothingMapped = WorldGeometryAccount(
        headline: "Nothing mapped yet",
        detail: "The Tower reports no geometry for this session.",
        isPhoneSideUnknown: false
    )

    /// The account for "nobody has said". The default everywhere one is needed.
    ///
    /// It asserts nothing about the world and nothing about the fetch, which is
    /// the correct thing for a value that exists because a caller supplied no
    /// information. Marked as the phone's own ignorance, because that is what
    /// it is.
    static let undescribed = WorldGeometryAccount(
        headline: "Geometry not described",
        detail: nil,
        isPhoneSideUnknown: true
    )

    /// The account for a (status, evidence) pair.
    ///
    /// `evidence` is `nil` when no snapshot exists — there is no world, so
    /// there is nothing for the Tower to have claimed.
    ///
    /// The ordering matters and is deliberate: **the Tower's claim is consulted
    /// before the phone's fetch result in every branch where the fetch failed
    /// or never happened.** That is the whole lesson of the field failure. A
    /// phone that could not fetch says so; it does not get to report on the
    /// world.
    static func account(
        for status: WorldGeometryStatus, evidence: WorldEvidence?
    ) -> WorldGeometryAccount {
        switch status {
        case .noWorld:
            return WorldGeometryAccount(
                headline: "No world",
                detail: "There is no world on screen for geometry to belong to.",
                isPhoneSideUnknown: false
            )

        case .notAddressed:
            // Situations 1 and 2. If the Tower's own snapshot says geometry
            // exists, this is the phone lacking an address for geometry that
            // is genuinely there — which is precisely the field case, and it
            // must not read as an empty room.
            if let evidence, evidence.hasGeometry {
                return WorldGeometryAccount(
                    headline: "Built, not yet fetched",
                    detail: "The Tower reports \(evidence.summary ?? "geometry") for this world "
                        + "but has not published an address to fetch the points from, so none are "
                        + "drawn here. The 3D reconstruction is served by a different route and "
                        + "may still be available.",
                    isPhoneSideUnknown: true
                )
            }
            return WorldGeometryAccount(
                headline: "Nothing built yet",
                detail: "The Tower has not published geometry for this session. "
                    + "Reconstruction runs in a separate Tower process, which this app "
                    + "can neither start nor see.",
                isPhoneSideUnknown: true
            )

        case .loading:
            return WorldGeometryAccount(
                headline: "Fetching the geometry…",
                detail: nil,
                isPhoneSideUnknown: true
            )

        case .failed(let failure):
            // Situations 3, 4 and 5. The reason is the Tower's or the
            // transport's; the claim about the world is the Tower's evidence,
            // if it made one.
            var detail = failure.message
            if let evidence, evidence.hasGeometry, let summary = evidence.summary {
                detail += " The Tower reports \(summary) for this world, so this is a fetch "
                    + "that failed and not a world that is empty."
            }
            return WorldGeometryAccount(
                headline: "Geometry could not be fetched",
                detail: detail,
                isPhoneSideUnknown: true
            )

        case .towerReportsNone(let towerDetail):
            // The Tower's own 404. It is entitled to say this — but if its own
            // snapshot said otherwise, the two disagree and the reader is told
            // that rather than being handed the more comfortable half.
            if let evidence, evidence.hasGeometry, let summary = evidence.summary {
                return WorldGeometryAccount(
                    headline: "The Tower disagrees with itself",
                    detail: "Its status report says \(summary) for this world; its geometry route "
                        + "answers that it has none"
                        + (towerDetail.map { " (\($0))" } ?? "")
                        + ". Nothing is drawn here, because which of the two is right is the "
                        + "Tower's question and not this app's.",
                    isPhoneSideUnknown: false
                )
            }
            // The Tower's own words when it sent any, and the standing sentence
            // otherwise. The **headline** is taken from `nothingMapped` rather
            // than retyped, so those four words genuinely exist in one place
            // and a search for them finds every way they can be drawn.
            return WorldGeometryAccount(
                headline: Self.nothingMapped.headline,
                detail: towerDetail.map { "The Tower answered: \($0)." }
                    ?? Self.nothingMapped.detail,
                isPhoneSideUnknown: false
            )

        case .loaded(let model, let unfetched):
            if !model.fragments.isEmpty || model.unresolvedCount > 0 {
                // There is something to draw, or something counted. The gallery
                // draws its own headline; this account is only consulted for
                // the empty case, and returning the gallery's own headline here
                // keeps the two from ever disagreeing.
                //
                // The `??` covers a manifest whose only rows are unresolved:
                // nothing is drawable, so `headline` is `nil` by design, and
                // "seen and not reconstructed" is what that manifest says.
                return WorldGeometryAccount(
                    headline: model.headline
                        ?? "\(model.unresolvedCount) area\(model.unresolvedCount == 1 ? "" : "s") "
                            + "seen, none reconstructed",
                    detail: unfetched > 0
                        ? "\(unfetched) segment\(unfetched == 1 ? "" : "s") the Tower named could "
                            + "not be fetched and are not drawn."
                        : nil,
                    isPhoneSideUnknown: false
                )
            }
            // An empty manifest. The Tower has a manifest for this session and
            // nothing in it resolved — but if its snapshot claimed geometry,
            // that is again a disagreement and not an empty room.
            if let evidence, evidence.hasGeometry, let summary = evidence.summary {
                return WorldGeometryAccount(
                    headline: "Nothing drawable in this session's geometry",
                    detail: "The Tower's manifest for this session names no segment that "
                        + "resolved, while its status report says \(summary). Nothing is drawn "
                        + "rather than a guess at which is right.",
                    isPhoneSideUnknown: false
                )
            }
            return .nothingMapped
        }
    }
}

// MARK: - The normal surface's vocabulary

/// What a world is doing, in words a person uses.
///
/// The brief for this vocabulary is that the ordinary screen must not read as a
/// reconstruction debugger. "2 registered, 120 refused", "coverage partial",
/// "N segments in the frame of segment 120" and the rest are all still
/// reachable — they moved behind Diagnostics, they were not deleted — and what
/// stands in their place on the normal surface is one of these words.
///
/// Every case is earned from something the Tower said. None is a guess about
/// the other machine, which is the rule the whole workspace is built on.
enum WorldStage: Equatable {
    /// Frames are reaching the Tower and it has reported no geometry yet.
    case mapping
    /// The Tower is receiving and has built geometry.
    case building
    /// The session has stopped and a live process is working on a world that
    /// already has geometry. `lifecycle.build_in_progress == true` is the
    /// evidence; see `WorldModelState.finalizing`.
    case improving
    /// The session has stopped and the world is being finished. Reached with
    /// no geometry yet, or on a Tower that cannot say whether a build runs.
    case finalizing
    /// Finished, with the final solve done or with no record either way.
    case saved
    /// Finished or interrupted, holding real geometry, with the final solve
    /// absent, skipped or failed. The world is usable and is not complete.
    case partial
    /// The session ended abnormally and geometry survived.
    case interrupted
    /// Nothing usable came of it.
    case needsRetry

    /// The word on screen. Sentence case, no jargon, no acronyms.
    var label: String {
        switch self {
        case .mapping: return "Mapping"
        case .building: return "Building"
        case .improving: return "Improving"
        case .finalizing: return "Finalizing"
        case .saved: return "Saved"
        case .partial: return "Partial"
        case .interrupted: return "Interrupted"
        case .needsRetry: return "Needs retry"
        }
    }

    var systemImage: String {
        switch self {
        case .mapping: return "dot.radiowaves.left.and.right"
        case .building, .improving: return "cube"
        case .finalizing: return "cube.transparent"
        case .saved: return "cube.fill"
        case .partial: return "cube.transparent.fill"
        case .interrupted: return "exclamationmark.triangle"
        case .needsRetry: return "exclamationmark.triangle.fill"
        }
    }

    /// Whether the Tower may still change this world. `false` means nothing
    /// more is coming, which is what makes "Partial" a settled description
    /// rather than a stage on the way somewhere.
    var isStillChanging: Bool {
        switch self {
        case .mapping, .building, .improving, .finalizing: return true
        case .saved, .partial, .interrupted, .needsRetry: return false
        }
    }

    /// The stage for a state, the Tower's evidence and its finalization record.
    ///
    /// `nil` for the states with no world in them — `.idle`,
    /// `.awaitingFirstUpdate`, `.unsupported`. Those have their own careful
    /// wording on the canvas and are deliberately not folded into this
    /// vocabulary: "Mapping" over an idle Tower would claim a build that
    /// nothing on the phone can start or see.
    static func stage(
        for state: WorldModelState,
        evidence: WorldEvidence?,
        finalization: WorldFinalizationReport?
    ) -> WorldStage? {
        switch state {
        case .unsupported, .idle, .awaitingFirstUpdate:
            return nil
        case .receiving:
            return evidence?.hasGeometry == true ? .building : .mapping
        case .finalizing(_, let buildInProgress):
            // "Improving" is claimed only where there is something to improve
            // and a live process holding the lock. Everywhere else the honest
            // word is the weaker one.
            if buildInProgress == true, evidence?.hasGeometry == true { return .improving }
            return .finalizing
        case .finalized:
            let solve = WorldFinalSolve(word: finalization?.finalSolve)
            if solve.deniesAFinishedWorld {
                return evidence?.hasGeometry == true ? .partial : .needsRetry
            }
            if solve == .pending { return .finalizing }
            // `.solved`, `.notReported` and any word this build has not heard
            // of. A record that says nothing about the final solve is an older
            // record, and the world it describes was stored as finished.
            return evidence?.hasGeometry == true || solve == .solved ? .saved : .needsRetry
        case .interrupted:
            return evidence?.hasGeometry == true ? .interrupted : .needsRetry
        case .failed:
            return .needsRetry
        }
    }
}

// MARK: - The final solve, which nothing read until now

/// `WorldFinalizationReport.finalSolve`, given meaning.
///
/// The field is decoded (`WorldSelection.swift`), carried on both the status
/// channel and the listing route, exposed on the client — and **read by no
/// view**. On the 2026-09-06 world it is `null`: the final solve never ran, and
/// nothing on screen said so. The world was presented as a finished one.
///
/// The words are the Tower's: `pending`, `solved`, `skipped`, `failed`,
/// `unavailable`. An unrecognised one survives as itself, for the reason
/// `WorldSelectionMode` gives.
/// `nonisolated`, unlike the rest of this file: it reads a `String?` and
/// nothing else, and `WorldListingPresentation` — which is itself
/// `nonisolated`, because it is a pure function of a decoded listing — needs to
/// ask it what a session's `final_solve` word means.
nonisolated enum WorldFinalSolve: Equatable, Sendable {
    /// No finalization record reached this screen.
    ///
    /// **Silence, and not one kind of silence.** A `nil` here means any of: no
    /// pinned status report has arrived yet, this Tower predates the field, the
    /// world never reached finalization, or the record genuinely kept no
    /// account of it. Nothing on the phone can tell those apart, so this case
    /// says **nothing at all** — see `sentence`.
    case notReported
    case pending
    case solved
    case skipped
    case failed
    case unavailable
    case other(String)

    init(word: String?) {
        switch word {
        case nil: self = .notReported
        case "pending": self = .pending
        case "solved": self = .solved
        case "skipped": self = .skipped
        case "failed": self = .failed
        case "unavailable": self = .unavailable
        case let other?: self = .other(other)
        }
    }

    /// Whether this word says the world was **not** finished properly.
    /// `.notReported` is excluded on purpose: a record that kept no account of
    /// the final solve has not said it failed.
    var deniesAFinishedWorld: Bool {
        switch self {
        case .skipped, .failed, .unavailable: return true
        case .notReported, .pending, .solved, .other: return false
        }
    }

    /// One line for the reader, or `nil` where there is nothing worth saying.
    ///
    /// `.solved` returns `nil`: a world that finished properly does not need a
    /// sentence explaining that it finished properly, and the stage word
    /// ("Saved") already carries it.
    var sentence: String? {
        switch self {
        case .notReported:
            // Nothing. This returned "This world has no record of a final pass,
            // so it may hold less than a finished one would." — a sentence the
            // phone composed out of a `nil` it could not interpret, which is
            // the exact defect class this whole pass exists to remove: the
            // phone reporting on the Tower from silence.
            //
            // It also contradicted the screen it was drawn on. `WorldStage`
            // maps a `.finalized` state with no record to `.saved`, so the card
            // said the world was final while the line under it implied it was
            // not. A word the Tower did not say cannot resolve that; only the
            // Tower saying one can.
            return nil
        case .pending:
            return "The final pass has not run yet."
        case .solved:
            return nil
        case .skipped:
            return "The final pass was skipped, so this world is what the walk produced "
                + "without it."
        case .failed:
            return "The final pass failed, so this world is what the walk produced without it."
        case .unavailable:
            return "The final pass could not run on this Tower, so this world is what the "
                + "walk produced without it."
        case .other(let word):
            // The Tower's word, quoted. A word this build has not heard of is
            // shown rather than folded into the nearest one it knows.
            return "The Tower reports its final pass as \"\(word)\"."
        }
    }
}

/// What can and cannot be done about a world that did not finish.
///
/// **There is no rebuild route.** `tower/tower/routes/geometry.py` serves
/// `GET /worlds`, `GET /worlds/{id}/geometry/manifest`,
/// `GET /worlds/{id}/geometry/segment/{i}` and `GET /worlds/{id}/render`, and
/// nothing else; every one is a read. Nothing on the phone can ask the Tower to
/// finish, re-solve or rebuild a world. So this type says the state truthfully
/// and offers no button, which is the rule the workspace already applies to
/// "Start Mapping".
struct WorldRecoverability: Equatable {
    /// Whether real geometry survived whatever happened.
    var geometrySurvived: Bool
    /// `nil` for the stages where there is nothing to recover from — a world
    /// still being built has not lost anything yet.
    var sentence: String?

    static func of(
        stage: WorldStage, evidence: WorldEvidence?, reason: String?
    ) -> WorldRecoverability {
        let survived = evidence?.hasGeometry == true
        switch stage {
        case .interrupted, .partial:
            if survived {
                return WorldRecoverability(
                    geometrySurvived: true,
                    sentence: "What was built before this stopped is stored and can be opened. "
                        + "Finishing it is the Tower's to do; this app has no way to ask for it."
                )
            }
            return WorldRecoverability(
                geometrySurvived: false,
                sentence: "Nothing usable survived this session"
                    + (reason.map { ": \($0)" } ?? ".")
            )
        case .needsRetry:
            return WorldRecoverability(
                geometrySurvived: false,
                sentence: "Nothing usable came of this session"
                    + (reason.map { ": \($0)" } ?? ".")
                    + " Walking the space again is what produces another one; this app cannot "
                    + "ask the Tower to rebuild this one."
            )
        case .mapping, .building, .improving, .finalizing, .saved:
            // Nothing has been lost, so there is nothing to say about
            // recovering it. `reason` belongs to the interrupted cases above.
            return WorldRecoverability(geometrySurvived: survived, sentence: nil)
        }
    }
}

// MARK: - Everything the canvas needs, in one value

/// The derived half of the World Builder screen, bundled.
///
/// One parameter rather than six on `WorldCanvasView`, and — more importantly —
/// one **value** rather than six computations spread across a view body. The
/// canvas draws this and `WorldModelState` and nothing else, so what is on
/// screen is a pure function of two values a test can construct, and the whole
/// state machine is assertable without SwiftUI.
///
/// `WorldBuilderViewModel.presentation` is the only place that builds one from
/// live data.
struct WorldPresentation: Equatable {
    /// The normal surface's word for what this world is doing. `nil` where
    /// there is no world.
    var stage: WorldStage?
    /// What the Tower says it built. `nil` where there is no snapshot.
    var evidence: WorldEvidence?
    var finalSolve: WorldFinalSolve
    /// The 3D ladder, resolved.
    var reconstruction: WorldReconstruction
    /// What the sparse gallery says when it has nothing to draw.
    var account: WorldGeometryAccount
    /// What survived, and what can be done about it. `nil` where there is no
    /// world.
    var recoverability: WorldRecoverability?

    init(
        stage: WorldStage? = nil,
        evidence: WorldEvidence? = nil,
        finalSolve: WorldFinalSolve = .notReported,
        reconstruction: WorldReconstruction = .unavailable(
            reason: "The Tower has named no world to look at."
        ),
        account: WorldGeometryAccount = .undescribed,
        recoverability: WorldRecoverability? = nil
    ) {
        self.stage = stage
        self.evidence = evidence
        self.finalSolve = finalSolve
        self.reconstruction = reconstruction
        self.account = account
        self.recoverability = recoverability
    }

    /// The value for a screen with no world on it. What previews and the
    /// states that carry no snapshot are drawn from.
    static let empty = WorldPresentation()
}

// MARK: - The 3D ladder

/// Which 3D reconstruction a saved world can actually show, best first.
///
/// The product rule this implements: **final 3D world > best available partial
/// 3D world > truthful degraded state.** The sparse fragment gallery is not on
/// this ladder at all — it is a diagnostic, and it lives behind Diagnostics.
///
/// The one surface in this app that can draw 3D is the `WKWebView` in
/// `WorldRenderViewer`, which shows the Tower's own composed page from
/// `GET /worlds/{id}/render`. There is no SceneKit, RealityKit or Metal in this
/// target, and there is no plan to add one for this: the Tower composes the
/// world, colours it and decimates it to a mobile budget, and duplicating that
/// on the phone would be a second renderer able to disagree with the first.
enum WorldReconstruction: Equatable {
    /// The Tower named geometry for a world nothing will add to, and its final
    /// pass ran. This is the world as it will always be.
    case final(WorldRenderTarget)
    /// The Tower named geometry for a world that is still changing, or whose
    /// final pass did not run. Real, and not the last word.
    case partial(WorldRenderTarget, note: String)
    /// No address for a 3D view. The reason is a sentence, because "the button
    /// is disabled" is not one.
    case unavailable(reason: String)

    /// The address to fetch, when there is one.
    var target: WorldRenderTarget? {
        switch self {
        case .final(let target): return target
        case .partial(let target, _): return target
        case .unavailable: return nil
        }
    }

    /// What the primary control says. A verb the app can actually deliver.
    var actionTitle: String { "Open the 3D world" }

    /// The ladder, resolved.
    ///
    /// `target` is `WorldBuilderViewModel.renderTarget`: the world and session
    /// the Tower named geometry for, or the world a person pinned. `nil` there
    /// is the honest "there is nothing to open", and it is a **different** state
    /// from the render route answering 404 — which happens after this, inside
    /// the viewer, and carries the Tower's own detail.
    static func ladder(
        target: WorldRenderTarget?,
        stage: WorldStage?,
        finalSolve: WorldFinalSolve,
        evidence: WorldEvidence?
    ) -> WorldReconstruction {
        guard let target else {
            // Said in terms of what the Tower reported, never in terms of a
            // nil pointer.
            if let evidence, evidence.hasGeometry {
                return .unavailable(
                    reason: "The Tower reports \(evidence.summary ?? "geometry") for this world "
                        + "but has not named where to fetch a picture of it."
                )
            }
            if let stage, stage.isStillChanging {
                return .unavailable(
                    reason: "Nothing has been built to look at yet. A 3D world appears once the "
                        + "Tower's first build produces one."
                )
            }
            return .unavailable(
                reason: "The Tower has built no 3D world for this session."
            )
        }
        guard let stage else {
            // A target with no stage: a world was pinned before its first
            // status arrived. Real address, unknown standing.
            return .partial(target, note: "The Tower has not yet said what state this world is in.")
        }
        switch stage {
        case .saved:
            // Nothing will add to this world, so it is the best that exists.
            // Whether its *record* mentions a final pass is a separate
            // statement, made by `WorldFinalSolve.sentence` beside this one —
            // folding it in here would make the ladder answer two questions
            // and be wrong about one of them.
            return .final(target)
        case .mapping, .building:
            return .partial(target, note: "This world is still being built, so it will change.")
        case .improving, .finalizing:
            return .partial(target, note: "This world is being finished, so it will change.")
        case .partial:
            return .partial(
                target,
                note: finalSolve.sentence ?? "This world did not finish, so it is what the walk "
                    + "produced without a final pass."
            )
        case .interrupted:
            return .partial(
                target, note: "This world was interrupted. What was built before it stopped is here."
            )
        case .needsRetry:
            // A target exists but nothing survived; the route will very likely
            // 404, and the viewer will say so in the Tower's own words. Offered
            // rather than hidden, because "very likely" is not knowledge.
            return .partial(target, note: "Little or nothing was reconstructed for this session.")
        }
    }
}
