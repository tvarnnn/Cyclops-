//
//  WorldFragmentsView.swift
//  Glasses
//

import SwiftUI

/// Layout decisions for the fragment gallery, kept out of the view so they can
/// be tested without rendering.
///
/// ## Why fragments and not a map
///
/// Every segment anchor the Tower produces sits at exactly the origin with
/// identity rotation, and per-segment scale disagrees by up to ~87x on a real
/// walk. Drawing them in one space would superimpose independent
/// reconstructions — geometry that looks like a room and means nothing.
/// `docs/modules/WORLD-BUILD.md` forbids exactly that.
///
/// So each fragment gets its own frame, its own scale, and its own box. When
/// the Tower registers segments, `registered` flips and a `transform_to_world`
/// arrives, and this model merges them without the view changing — but only
/// the ones that name the **same** `reference_segment` under the same
/// `frame_revision`. `registered: true` is a necessary condition and not a
/// sufficient one: a Sim3 maps a segment into its reference's frame, and there
/// is no global world frame, so two registered segments with different
/// references are as unmergeable as two unregistered ones. See
/// `hasSharedFrame` and `clusters`.
///
/// ## Since the global solve
///
/// `WORLD-BUILDER-GEOMETRY.md` §8: most of a walk now arrives registered into
/// one reference — 29 of 34 segments on 2026-09-06 — with `scale` exactly
/// `1.0`. Those are grouped by `clusters` and drawn together on one canvas per
/// group; whatever is in no group keeps its own tile, exactly as before. The
/// rule that decides membership did not change and is still asked in one place.
struct WorldFragmentsModel: Equatable {
    let segments: [WorldSegmentSummary]

    /// Whether the Tower's geometry reflects every keyframe it has accepted.
    ///
    /// A `var` with a default so the memberwise initialiser keeps working for
    /// the empty and cleared cases, which have no manifest to ask.
    var isCurrent: Bool = true

    /// The Tower's coverage words this model treats specially. Anything else
    /// it does not know is still shown verbatim on the tile, and never acted
    /// on. `nonisolated` so the static helpers below can read them.
    nonisolated static let partialCoverage = "partial"
    nonisolated static let unresolvedCoverage = "unresolved"

    /// Only resolved segments with real bounds can be drawn. A resolved
    /// segment with no bounds is incoherent and is refused rather than framed
    /// by guess.
    ///
    /// A segment the solve judged `unresolved` is refused here too, whatever
    /// its `resolution_state` says: the contract defines that word as
    /// "keyframes exist, no geometry", and the two fields agreeing is the
    /// expected case, but a renderer must not place a segment the Tower has
    /// said has nothing to place. It is counted with the other unresolved ones.
    ///
    /// The filter decides membership; `ranked` decides only the order they are
    /// read in. The two are kept separate on purpose — see `ranked` — so that
    /// a change to display order can never quietly change which fragments the
    /// grid shows, or what `unresolvedCount` says about the rest.
    var fragments: [WorldSegmentSummary] {
        Self.ranked(
            segments.filter {
                $0.resolutionState == .resolved && $0.bounds != nil
                    && $0.coverage != Self.unresolvedCoverage
            }
        )
    }

    /// Segments that hold keyframes and recovered nothing. Counted, never
    /// placed: we know reconstruction failed, not where.
    ///
    /// Either the Tower's `resolution_state` or its `coverage` may say so, and
    /// a segment is counted once whichever of them did.
    var unresolvedCount: Int {
        segments.filter {
            $0.resolutionState == .unresolved || $0.coverage == Self.unresolvedCoverage
        }.count
    }

    // MARK: Registration, counted

    /// By the Tower's `registration_state`, never inferred from the bool: a
    /// row that says `registered: false` and nothing more has not said which
    /// of "refused" and "nobody looked" applies, and it is counted under
    /// `placementUnknownCount` rather than under a guess.
    var registeredCount: Int { segments.filter { $0.placement.state == .registered }.count }
    var refusedCount: Int { segments.filter { $0.placement.state == .refused }.count }
    var unplacedCount: Int { segments.filter { $0.placement.state == .unplaced }.count }
    /// Rows from a Tower that predates `registration_state`.
    var placementUnknownCount: Int { segments.filter { $0.placement.state == nil }.count }

    /// One line of counts in the Tower's own vocabulary, or `nil` when the
    /// Tower said nothing about placement at all. Zero terms are omitted: a
    /// world with no refusals has nothing to say about refusals.
    var placementSummary: String? {
        var parts: [String] = []
        if registeredCount > 0 { parts.append("\(registeredCount) registered") }
        if refusedCount > 0 { parts.append("\(refusedCount) refused") }
        if unplacedCount > 0 { parts.append("\(unplacedCount) unplaced") }
        guard !parts.isEmpty else { return nil }
        return parts.joined(separator: ", ")
    }

    /// The Tower's registration word for a tile, and its refusal reason
    /// **verbatim** beside it when there is one. `nil` when the Tower did not
    /// send `registration_state`, because there is then nothing to say that is
    /// not invented.
    nonisolated static func placementCaption(for segment: WorldSegmentSummary) -> String? {
        guard let state = segment.placement.state else { return nil }
        if let reason = segment.placement.refusalReason, !reason.isEmpty {
            return "\(state.rawValue) — \(reason)"
        }
        return state.rawValue
    }

    /// The Tower's coverage word, when it sent one. Shown as the word it used.
    nonisolated static func coverageCaption(for segment: WorldSegmentSummary) -> String? {
        guard let coverage = segment.coverage else { return nil }
        return "coverage \(coverage)"
    }

    /// Whether a tile is drawn muted. Only `partial` is: `confident` is drawn
    /// normally, `unresolved` is never drawn, and a word this build does not
    /// know is drawn normally because muting it would be a judgment the Tower
    /// did not make.
    nonisolated static func isMuted(_ segment: WorldSegmentSummary) -> Bool {
        segment.coverage == partialCoverage
    }

    // MARK: Shared frames

    /// True only when every segment here is placed **into the same frame**.
    ///
    /// `registered` on every row is not enough on its own. A Sim3 maps a
    /// segment into the frame of its `reference_segment`, and there is no
    /// global world frame to fall back into — so two registered segments that
    /// name different references share no space, and drawing them together
    /// would composite independent reconstructions exactly as drawing two
    /// unregistered ones would. The same goes for two rows stamped with
    /// different `frame_revision`s: a coordinate expressed in one gauge may not
    /// be reinterpreted under another.
    ///
    /// `mayBeCompositedWith` is the one place that rule is written; this asks
    /// it of every row against the first rather than restating it.
    var hasSharedFrame: Bool {
        guard let first = segments.first else { return false }
        return segments.allSatisfy { first.mayBeCompositedWith($0) }
    }

    /// The drawable segments the Tower has placed into shared frames, grouped
    /// one group per frame. Empty on every manifest that predates the global
    /// solve, which is what keeps the gallery unchanged for those.
    var clusters: [WorldCluster] {
        WorldClusterBuilder.clusters(from: fragments)
    }

    /// The drawable segments in no cluster: their own tile, own frame, own
    /// scale, exactly as every fragment was drawn before registration existed.
    var unclusteredFragments: [WorldSegmentSummary] {
        var clustered = Set<Int>()
        for cluster in clusters { clustered.formUnion(cluster.memberIndexes) }
        return fragments.filter { !clustered.contains($0.segmentIndex) }
    }

    /// Said out loud when the fragments on screen are real but behind. Not a
    /// warning and not an error state: the world is still being built, and
    /// this is what "still building" looks like from here.
    var buildingNote: String? {
        if isCurrent { return nil }
        return "The Tower is still building this world, so these fragments "
            + "may be behind the newest frames."
    }

    /// What is connected and what is not, in one line.
    ///
    /// A "connected world" is a cluster — segments the Tower placed into one
    /// frame — and the count after it is how many segments that frame holds.
    /// The loose fragments are named separately and never added to it.
    var headline: String {
        let clusters = self.clusters
        let loose = unclusteredFragments.count
        if clusters.isEmpty && loose == 0 { return "Nothing mapped yet" }

        var parts: [String] = []
        if !clusters.isEmpty {
            let placed = clusters.reduce(0) { $0 + $1.members.count }
            parts.append(
                clusters.count == 1
                    ? "1 connected world of \(placed) segments"
                    : "\(clusters.count) connected groups of \(placed) segments"
            )
        }
        if loose > 0 {
            // The wording the gallery has always used when nothing is
            // connected, kept verbatim; a shorter clause when it follows a
            // connected world.
            if clusters.isEmpty {
                parts.append(
                    loose == 1
                        ? "1 fragment, not yet connected"
                        : "\(loose) fragments, not yet connected"
                )
            } else {
                parts.append(
                    loose == 1
                        ? "1 fragment not yet connected"
                        : "\(loose) fragments not yet connected"
                )
            }
        }
        return parts.joined(separator: ", ")
    }
}

extension WorldFragmentsModel {
    /// Puts the most-mapped fragments first, and does it totally.
    ///
    /// ## Why the grid needs an order at all
    ///
    /// Manifest order is capture order, which says when a segment was walked
    /// through and nothing about whether anything was recovered from it. That
    /// was survivable while a walk produced tens of segments. It stops being
    /// survivable as the Tower's segmentation gets finer — an unrestricted
    /// version of it takes a real walk to hundreds of segments — because then
    /// the parts of the room that actually reconstructed are scattered among
    /// the parts that were barely seen, and the reader has to scan the whole
    /// grid to find them.
    ///
    /// `point_count` is the Tower's own count of the points a segment
    /// recovered, so ordering by it puts the fragments with the most recovered
    /// geometry at the top. That is a statement about QUANTITY and nothing
    /// else: a fragment above another has more points in it, not a better or
    /// more trustworthy reconstruction. Nothing here, and nothing in the view,
    /// may say otherwise.
    ///
    /// ## Why the tie-break is not optional
    ///
    /// `sorted(by:)` is not documented as stable, so two segments with equal
    /// point counts could come back in either order — and in a `ForEach` that
    /// is cards swapping places for no reason a reader can see. `segmentIndex`
    /// is unique within a manifest (`docs/contracts/WORLD-BUILDER-GEOMETRY.md`
    /// §2.1 defines it as identity within the session), so breaking ties on it
    /// makes the order total: **one manifest has exactly one display order.**
    ///
    /// ## What this does NOT make stable, said plainly
    ///
    /// The order is a pure function of one manifest. It is **not** stable
    /// across manifests, and during a live walk the manifest is refetched every
    /// time the revision moves — 67 times in the two-minute P3 walk. The Tower
    /// re-solves segments in place, so `point_count` for a segment that already
    /// existed can change between polls, and the primary sort key changes with
    /// it. A card can therefore move up the grid mid-walk.
    ///
    /// That is a real cost and it is accepted deliberately, not overlooked.
    /// Capture order never moved an existing card, and the trade is that it
    /// scatters the segments that actually reconstructed among the ones that
    /// barely did — which is survivable at tens of segments and not survivable
    /// at the hundreds the Tower's finer segmentation produces. Ranking is
    /// worth more when the grid is read, which is after the walk, than the
    /// movement costs while it is being built.
    ///
    /// **If that movement proves distracting on a real walk, the fix is not to
    /// drop the ranking** — it is to rank only once the world stops changing,
    /// or to animate the reorder so it reads as motion rather than as a
    /// glitch. Neither is worth building before a wearer says it is a problem.
    ///
    /// `nonisolated` because `WorldClusterBuilder`, which is a nonisolated
    /// value-level helper, orders a cluster's members with the same rule — and
    /// one rule is the only safe number of rules. It reads its argument and
    /// nothing else.
    nonisolated static func ranked(_ fragments: [WorldSegmentSummary]) -> [WorldSegmentSummary] {
        fragments.sorted { lhs, rhs in
            if lhs.pointCount != rhs.pointCount {
                return lhs.pointCount > rhs.pointCount
            }
            return lhs.segmentIndex < rhs.segmentIndex
        }
    }

    /// Maps a segment-local `(x, z)` into that segment's OWN tile.
    ///
    /// Lifted off the view deliberately: this is the single place a shared
    /// scale could leak in and composite two fragments that share no
    /// coordinate frame, so it is the one piece of layout that must be
    /// directly testable rather than merely structurally correct.
    ///
    /// `ClusterCanvas` uses it too, with bounds computed over points that are
    /// **already in one frame** (`WorldClusterBuilder.referenceFrameBounds`).
    /// That is not a shared scale leaking in; it is the same projector applied
    /// to coordinates the composition rule has already admitted.
    static func projector(
        bounds: WorldBounds, size: CGSize
    ) -> (Double, Double) -> CGPoint {
        let spanX = Swift.max(bounds.max[0] - bounds.min[0], 1e-6)
        let spanZ = Swift.max(bounds.max[2] - bounds.min[2], 1e-6)
        let scale = Swift.min(size.width / spanX, size.height / spanZ) * 0.9
        let offsetX = (size.width - spanX * scale) / 2
        let offsetZ = (size.height - spanZ * scale) / 2
        return { x, z in
            CGPoint(
                x: offsetX + (x - bounds.min[0]) * scale,
                y: offsetZ + (z - bounds.min[2]) * scale
            )
        }
    }
}

/// One fragment, drawn top-down in its own frame.
///
/// Top-down `(x, z)` and not 3D because `up_axis` is `"unknown"` — a 3D view
/// would have to guess which way is up. SceneKit earns its weight once a floor
/// plane exists; until then this is both cheaper and more honest.
struct FragmentCanvas: View {
    let summary: WorldSegmentSummary
    let chunk: WorldSegmentChunk?

    var body: some View {
        Canvas { context, size in
            guard let chunk, let bounds = summary.bounds else { return }
            let project = WorldFragmentsModel.projector(bounds: bounds, size: size)

            for point in chunk.points where point.count == 3 {
                let p = project(point[0], point[2])
                context.fill(
                    Path(ellipseIn: CGRect(x: p.x - 1, y: p.y - 1, width: 2, height: 2)),
                    with: .color(.secondary)
                )
            }

            // The camera path, broken wherever a pose was refused. A line
            // through the gap would assert motion that was never measured.
            var path = Path()
            var pendingMove = true
            for pose in chunk.poses {
                guard let t = pose.translation, t.count == 3 else {
                    pendingMove = true
                    continue
                }
                let p = project(t[0], t[2])
                if pendingMove {
                    path.move(to: p)
                    pendingMove = false
                } else {
                    path.addLine(to: p)
                }
            }
            context.stroke(path, with: .color(.accentColor), lineWidth: 1.5)
        }
        .background(Color.secondary.opacity(0.08))
    }
}

/// One cluster, drawn top-down in its **reference segment's** frame, with every
/// member's points and camera path mapped there by its own `transform_to_world`.
///
/// Only members of one `WorldCluster` are ever drawn here, and
/// `WorldClusterBuilder` admitted each of them through `mayBeCompositedWith`.
/// Nothing in this view composites; it draws what was already established as
/// sharing a frame. The points come from `pointsInReferenceFrame`, which is
/// `nil` — not the local points — for a chunk with no transform, so a chunk
/// that somehow lost its placement between manifest and fetch draws nothing
/// rather than drawing at the origin.
///
/// Wrapped in `InteractiveWorldCanvas` so a 29-segment world can be examined:
/// pinch, drag and twist, double-tap to reset. The transform is applied by the
/// wrapper; this closure draws in plain canvas coordinates.
struct ClusterCanvas: View {
    let cluster: WorldCluster
    let chunks: [String: WorldSegmentChunk]

    var body: some View {
        InteractiveWorldCanvas { context, size in
            guard
                let bounds = WorldClusterBuilder.referenceFrameBounds(of: cluster, chunks: chunks)
            else { return }
            let project = WorldFragmentsModel.projector(bounds: bounds, size: size)

            for member in cluster.members {
                guard
                    let chunk = chunks[member.cacheKey],
                    let transform = chunk.placement.transform,
                    let points = chunk.pointsInReferenceFrame
                else { continue }
                // `partial` is drawn fainter, and only `partial`. See
                // `WorldFragmentsModel.isMuted`.
                let alpha: Double = WorldFragmentsModel.isMuted(member) ? 0.35 : 1

                for point in points where point.count == 3 {
                    let p = project(point[0], point[2])
                    context.fill(
                        Path(ellipseIn: CGRect(x: p.x - 1, y: p.y - 1, width: 2, height: 2)),
                        with: .color(.secondary.opacity(alpha))
                    )
                }

                // The camera path, in the reference frame, broken wherever a
                // pose was refused — for the reason `FragmentCanvas` gives.
                var path = Path()
                var pendingMove = true
                for pose in chunk.poses {
                    guard
                        let t = pose.translation, t.count == 3,
                        let placed = transform.apply(to: t)
                    else {
                        pendingMove = true
                        continue
                    }
                    let p = project(placed[0], placed[2])
                    if pendingMove {
                        path.move(to: p)
                        pendingMove = false
                    } else {
                        path.addLine(to: p)
                    }
                }
                context.stroke(path, with: .color(.accentColor.opacity(alpha)), lineWidth: 1.5)
            }
        }
        .background(Color.secondary.opacity(0.08))
    }
}

/// The gallery: the connected world(s), the known-but-unconnected fragments,
/// plus honest accounts of the states that have no geometry to draw.
struct WorldFragmentsView: View {
    let model: WorldFragmentsModel
    let chunks: [String: WorldSegmentChunk]

    private var columns: [GridItem] { [GridItem(.adaptive(minimum: 140), spacing: 12)] }

    // Read through properties rather than bound with a `let` inside the
    // builder below — the pattern this codebase settled on after a
    // result-builder block with a binding in it caused trouble in Product
    // Shell V2. Grouping walks every fragment, and is cheap at the tens of
    // segments a walk produces.
    private var clusters: [WorldCluster] { model.clusters }
    private var loose: [WorldSegmentSummary] { model.unclusteredFragments }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(model.headline)
                .font(.headline)

            if let note = model.buildingNote {
                Text(note)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            if let summary = model.placementSummary {
                Text(summary)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            ForEach(clusters, id: \.id) { cluster in
                clusterSection(cluster)
            }

            if clusters.isEmpty && loose.isEmpty {
                // UNKNOWN: nothing has been mapped. Not an empty canvas,
                // which would read as an empty room.
                Text("The glasses have not mapped anything here yet.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else if !loose.isEmpty {
                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(loose, id: \.segmentIndex) { segment in
                        tile(segment)
                    }
                }
            }

            if model.unresolvedCount > 0 {
                // OBSERVED BUT UNRESOLVED. Deliberately not drawn: we know
                // reconstruction failed, not where it failed, and a region
                // would invent a location.
                Text("\(model.unresolvedCount) areas were seen but could not be reconstructed.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// One connected world: its canvas, then what is in it and what is still
    /// on its way.
    private func clusterSection(_ cluster: WorldCluster) -> some View {
        let missing = cluster.members.filter { chunks[$0.cacheKey] == nil }.count
        return VStack(alignment: .leading, spacing: 4) {
            ClusterCanvas(cluster: cluster, chunks: chunks)
                .frame(height: 260)
                .clipShape(RoundedRectangle(cornerRadius: 8))

            Text(
                "\(cluster.members.count) segments in the frame of segment "
                    + "\(cluster.referenceSegment) · \(cluster.pointCount) points"
            )
            .font(.caption2)
            .foregroundStyle(.secondary)

            if missing > 0 {
                // The canvas stays empty until every member is in hand; see
                // `WorldClusterBuilder.referenceFrameBounds`.
                Text("Waiting for \(missing) of \(cluster.members.count) segments before drawing them together.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Text("Drag to pan, pinch to zoom, twist to turn. Double-tap to reset.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    /// One unconnected fragment and the Tower's words about where it stands.
    private func tile(_ segment: WorldSegmentSummary) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            FragmentCanvas(
                summary: segment,
                chunk: chunks[segment.cacheKey]
            )
            .frame(height: 120)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .opacity(WorldFragmentsModel.isMuted(segment) ? 0.45 : 1)

            Text("\(segment.pointCount) points")
                .font(.caption2)
                .foregroundStyle(.secondary)
            if let caption = WorldFragmentsModel.placementCaption(for: segment) {
                // The Tower's registration word and its refusal reason,
                // verbatim. Usually "the wearer stood still", which is advice
                // to the wearer rather than a fault.
                Text(caption)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let coverage = WorldFragmentsModel.coverageCaption(for: segment) {
                Text(coverage)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            if let chunk = chunks[segment.cacheKey], chunk.isSampled {
                Text("showing \(chunk.pointsSent) of \(chunk.pointsTotal)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
