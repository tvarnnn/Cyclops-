//
//  WorldClusters.swift
//  Glasses
//

import Foundation

/// Segments the Tower has placed into **one** shared frame, and only those.
///
/// ## What a cluster is, and what it is not
///
/// A `transform_to_world` maps a segment's own frame into the frame of its
/// `reference_segment`, and there is no global world frame above that. So the
/// only space two segments can share is "the same reference under the same
/// `frame_revision`", and a cluster is exactly that set — nothing bigger. Two
/// clusters with different references are as unmergeable as two unregistered
/// segments, and are never drawn on one canvas.
///
/// Since the global solve (`WORLD-BUILDER-GEOMETRY.md` §8) most of a walk lands
/// in one cluster: on 2026-09-06, 29 of 34 segments named the same reference
/// with `scale` exactly `1.0`. That is what makes drawing them together worth
/// building. Nothing here depends on it being one cluster, though — a walk
/// whose tracking broke into two solve components shows two.
///
/// Membership is decided by `WorldSegmentSummary.mayBeCompositedWith`, which is
/// the one place the composition rule is written. This type groups by asking
/// it, not by restating it.
nonisolated struct WorldCluster: Equatable, Sendable {
    /// Whose frame every member's points are in once transformed.
    let referenceSegment: Int
    /// Which gauge that frame is expressed in.
    let frameRevision: Int
    /// In display order (`WorldFragmentsModel.ranked`), so a cluster's own
    /// caption lists its members the same way the gallery does.
    let members: [WorldSegmentSummary]

    /// Stable within one manifest, which is all a `ForEach` needs.
    var id: String { "\(referenceSegment):\(frameRevision)" }

    var pointCount: Int { members.reduce(0) { $0 + $1.pointCount } }

    var memberIndexes: Set<Int> { Set(members.map(\.segmentIndex)) }
}

nonisolated enum WorldClusterBuilder {

    /// A group of one is not "connected" to anything, so it stays a tile and
    /// is not promoted to a canvas of its own.
    static let minimumMembers = 2

    /// Groups the segments that may be drawn in one space, and returns only the
    /// groups big enough to mean something.
    ///
    /// Grouping is done by asking `mayBeCompositedWith` of a candidate against
    /// each group's first member, rather than by keying on
    /// `(reference_segment, frame_revision)` and trusting the key: the two
    /// agree today, but the choke point is the choke point precisely so that
    /// nobody has to notice if they stop agreeing. A segment that shares a frame
    /// with nobody — unregistered, no transform, a lone reference — forms no
    /// group, and a group below `minimumMembers` is dropped.
    ///
    /// Ordered by member count descending, then by reference segment, so the
    /// order is a pure function of the manifest and a `ForEach` over it does
    /// not shuffle.
    static func clusters(from segments: [WorldSegmentSummary]) -> [WorldCluster] {
        var groups: [[WorldSegmentSummary]] = []
        for segment in segments {
            // Not placed at all: `mayBeCompositedWith` refuses a segment even
            // against itself when it has no frame to be in.
            guard segment.mayBeCompositedWith(segment) else { continue }
            if let index = groups.firstIndex(where: { $0[0].mayBeCompositedWith(segment) }) {
                groups[index].append(segment)
            } else {
                groups.append([segment])
            }
        }

        var clusters: [WorldCluster] = []
        for group in groups where group.count >= minimumMembers {
            // Every member passed `mayBeCompositedWith` against the first, so
            // the first member's transform names the frame they all share.
            guard let frame = group[0].placement.transform else { continue }
            clusters.append(
                WorldCluster(
                    referenceSegment: frame.referenceSegment,
                    frameRevision: frame.frameRevision,
                    members: WorldFragmentsModel.ranked(group)
                )
            )
        }
        return clusters.sorted { lhs, rhs in
            if lhs.members.count != rhs.members.count {
                return lhs.members.count > rhs.members.count
            }
            if lhs.referenceSegment != rhs.referenceSegment {
                return lhs.referenceSegment < rhs.referenceSegment
            }
            return lhs.frameRevision < rhs.frameRevision
        }
    }

    /// The box, in the reference segment's frame, around everything a cluster
    /// would draw: every member's transformed points **and** its transformed
    /// pose translations, so the camera path is inside the frame too.
    ///
    /// `nil` when any member's chunk is not in hand, or has no transform, or
    /// when nothing contributed a coordinate. A box over the members that
    /// happened to arrive would frame a partial world as the whole one and then
    /// jump when the rest landed; drawing nothing until it is all here is the
    /// honest state, and the view says so.
    static func referenceFrameBounds(
        of cluster: WorldCluster, chunks: [String: WorldSegmentChunk]
    ) -> WorldBounds? {
        var low = [Double.infinity, Double.infinity, Double.infinity]
        var high = [-Double.infinity, -Double.infinity, -Double.infinity]
        var sawAny = false

        for member in cluster.members {
            guard
                let chunk = chunks[member.cacheKey],
                let transform = chunk.placement.transform,
                let points = chunk.pointsInReferenceFrame
            else { return nil }
            var coordinates = points
            for pose in chunk.poses {
                guard let t = pose.translation, let placed = transform.apply(to: t) else { continue }
                coordinates.append(placed)
            }
            for coordinate in coordinates where coordinate.count == 3 {
                sawAny = true
                for axis in 0..<3 {
                    low[axis] = Swift.min(low[axis], coordinate[axis])
                    high[axis] = Swift.max(high[axis], coordinate[axis])
                }
            }
        }

        guard sawAny else { return nil }
        return WorldBounds(min: low, max: high)
    }
}
