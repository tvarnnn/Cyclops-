import Foundation
import XCTest
import CoreGraphics
import Combine
@testable import Glasses

// MARK: - Placement fixtures

/// A segment the Tower reported as not registered, with nothing more said —
/// the shape a row from a Tower predating `registration_state` and
/// `placement_hash` decodes to, and the right placement for any test whose
/// subject is not placement.
private func notRegistered(placementHash: String? = nil) -> WorldPlacementFields {
    WorldPlacementFields(
        registered: false, state: nil, refusalReason: nil,
        transform: nil, placementHash: placementHash
    )
}

/// A placed segment, in the frame of `reference`.
///
/// Identity rotation because the tests that care about the rotation build their
/// own; what varies here is `reference`, because "same reference or nothing" is
/// the composition rule these fixtures exist to exercise.
private func placed(
    into reference: Int, frameRevision: Int = 1, placementHash: String? = nil
) -> WorldPlacementFields {
    WorldPlacementFields(
        registered: true, state: .registered, refusalReason: nil,
        transform: WorldTransform(
            rotationWXYZ: [1, 0, 0, 0], translation: [0, 0, 0], scale: 1,
            referenceSegment: reference, frameRevision: frameRevision
        ),
        placementHash: placementHash
    )
}

final class WorldGeometryDecoderTests: XCTestCase {

    private func manifestJSON(upAxis: String = "unknown") -> [String: Any] {
        [
            "contract": "world_builder.geometry/2026-08-25",
            "world_id": "w0", "session_id": "s0",
            "geometry_revision": "abc123",
            "pose_convention": [
                "pose_type": "T_world_camera", "quaternion_order": "wxyz",
                "handedness": "right",
                "camera_axes": "opencv_x_right_y_down_z_forward",
                "translation_units": "world",
                "world_axes_origin": "first_keyframe_camera",
                "up_axis": upAxis, "pose_dtype": "float64",
                "point_dtype": "float32",
            ],
            "scale": ["state": "unknown", "meters_per_unit": NSNull()],
            "segment_count": 2,
            "segments": [
                ["segment_index": 0, "content_hash": "h0", "frame_id": "segment:0",
                 "registered": false, "transform_to_world": NSNull(),
                 "resolution_state": "resolved", "dominant_degeneracy": NSNull(),
                 "keyframe_count": 2, "solved_count": 1, "point_count": 2,
                 "bounds": ["min": [-1.0, 0.0, 3.0], "max": [1.0, 2.0, 5.0]]],
                ["segment_index": 1, "content_hash": "h1", "frame_id": "segment:1",
                 "registered": false, "transform_to_world": NSNull(),
                 "resolution_state": "unresolved",
                 "dominant_degeneracy": "low_parallax",
                 "keyframe_count": 2, "solved_count": 0, "point_count": 0,
                 "bounds": NSNull()],
            ],
        ]
    }

    func testAManifestDecodesFieldForField() {
        let manifest = WorldGeometryDecoder.manifest(from: manifestJSON())
        XCTAssertEqual(manifest?.segments.count, 2)
        XCTAssertEqual(manifest?.segments[0].contentHash, "h0")
        XCTAssertEqual(manifest?.segments[0].pointCount, 2)
    }

    /// `coverage` is additive (`WORLD-BUILDER-GEOMETRY.md` §8). Absent and
    /// `null` both mean "nothing has judged this segment" and decode to `nil`
    /// — never to `"unresolved"`, which is a judgment. A word the Tower sends
    /// is carried verbatim, including one this build does not know.
    func testCoverageIsNilWhenAbsentOrNullAndVerbatimWhenSent() throws {
        XCTAssertNil(WorldGeometryDecoder.manifest(from: manifestJSON())?.segments[0].coverage)

        var json = manifestJSON()
        var segments = try XCTUnwrap(json["segments"] as? [[String: Any]])
        segments[0]["coverage"] = NSNull()
        segments[1]["coverage"] = "unresolved"
        json["segments"] = segments
        let decoded = try XCTUnwrap(WorldGeometryDecoder.manifest(from: json))
        XCTAssertNil(decoded.segments[0].coverage)
        XCTAssertEqual(decoded.segments[1].coverage, "unresolved")

        segments[0]["coverage"] = "partial"
        segments[1]["coverage"] = "a-word-this-build-does-not-know"
        json["segments"] = segments
        let verbatim = try XCTUnwrap(WorldGeometryDecoder.manifest(from: json))
        XCTAssertEqual(verbatim.segments[0].coverage, "partial")
        XCTAssertEqual(verbatim.segments[1].coverage, "a-word-this-build-does-not-know")
    }

    func testAManifestThatIsBehindTheJournalDecodesAsNotCurrent() {
        // The Tower used to answer 404 for geometry that was real but behind,
        // which during a walk meant the whole capture. It now serves it with
        // current: false, and this flag is the only thing separating a
        // partial world from the finished one.
        var json = manifestJSON()
        json["current"] = false

        let manifest = WorldGeometryDecoder.manifest(from: json)
        XCTAssertNotNil(manifest, "behind is served, not refused")
        XCTAssertEqual(manifest?.current, false)
        XCTAssertEqual(manifest?.segments.count, 2, "the geometry is unchanged")
    }

    func testACurrentManifestSaysSo() {
        var json = manifestJSON()
        json["current"] = true
        XCTAssertEqual(WorldGeometryDecoder.manifest(from: json)?.current, true)
    }

    func testAManifestWithoutTheFieldIsTakenAsCurrent() {
        // An older Tower never sends it -- and that Tower hid anything behind
        // the journal behind a 404, so what it did serve was current by
        // construction.
        XCTAssertNil(manifestJSON()["current"])
        XCTAssertEqual(WorldGeometryDecoder.manifest(from: manifestJSON())?.current, true)
    }

    func testAnUnresolvedSegmentKeepsNilBoundsRatherThanAZeroBox() {
        let manifest = WorldGeometryDecoder.manifest(from: manifestJSON())
        let unresolved = manifest?.segments[1]
        XCTAssertNil(unresolved?.bounds)
        XCTAssertEqual(unresolved?.resolutionState, .unresolved)
        XCTAssertEqual(unresolved?.dominantDegeneracy, "low_parallax")
    }

    func testAPoseConventionMismatchIsRefused() {
        // Inverting T_world_camera still draws a plausible map. That was a
        // real shipped bug, so any mismatch refuses rather than renders.
        var json = manifestJSON()
        var convention = json["pose_convention"] as! [String: Any]
        convention["quaternion_order"] = "xyzw"
        json["pose_convention"] = convention

        let manifest = WorldGeometryDecoder.manifest(from: json)
        XCTAssertNotNil(manifest, "a mismatch decodes; it is the RENDER that refuses")
        XCTAssertFalse(manifest!.poseConvention.matchesThisBuild)
    }

    func testTheExpectedConventionIsAcceptedIncludingUnknownUpAxis() {
        let manifest = WorldGeometryDecoder.manifest(from: manifestJSON())
        XCTAssertTrue(manifest!.poseConvention.matchesThisBuild)
    }

    func testAWrongContractIdentifierIsRefused() {
        var json = manifestJSON()
        json["contract"] = "world_builder.geometry/2027-01-01"
        XCTAssertNil(WorldGeometryDecoder.manifest(from: json))
    }

    func testAMalformedSegmentRowDropsTheWholeManifest() {
        // Skipping the bad row would silently shrink the world and the viewer
        // would render a smaller map as if it were complete.
        var json = manifestJSON()
        var segments = json["segments"] as! [[String: Any]]
        segments[1].removeValue(forKey: "content_hash")
        json["segments"] = segments

        XCTAssertNil(WorldGeometryDecoder.manifest(from: json))
    }

    func testARefusedPoseDecodesAsNilTranslationNotZero() {
        let json: [String: Any] = [
            "contract": "world_builder.geometry/2026-08-25",
            "segment_index": 1, "content_hash": "h1", "frame_id": "segment:1",
            "registered": false, "transform_to_world": NSNull(),
            "poses": [
                ["keyframe_id": "s0:1", "status": "anchor", "degeneracy": "",
                 "rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]],
                ["keyframe_id": "s0:2", "status": "unavailable",
                 "degeneracy": "low_parallax",
                 "rotation": NSNull(), "translation": NSNull()],
            ],
            "points": [[Double]](), "points_sent": 0, "points_total": 0,
            "point_sampling": "none",
        ]

        let chunk = WorldGeometryDecoder.chunk(from: json)
        XCTAssertNil(chunk?.poses[1].translation)
        XCTAssertNotNil(chunk?.poses[0].translation)
    }

    func testTheManifestCarriesTheScaleBlock() {
        let manifest = WorldGeometryDecoder.manifest(from: manifestJSON())
        XCTAssertEqual(manifest?.scale.state, .unknown)
        XCTAssertNil(manifest?.scale.metersPerUnit)
        XCTAssertFalse(manifest!.scale.convertibleToMetres)
    }

    /// An absent scale block reads as `unknown` rather than refusing the
    /// manifest — the weakest claim, not a convenient one. Refusing would
    /// discard real geometry over metadata the viewer is forbidden from
    /// turning into a distance anyway.
    func testAManifestWithNoScaleBlockReadsAsUnknownAndNotAsRelative() {
        var json = manifestJSON()
        json.removeValue(forKey: "scale")
        let manifest = WorldGeometryDecoder.manifest(from: json)
        XCTAssertNotNil(manifest)
        XCTAssertEqual(manifest?.scale.state, .unknown)
    }

    /// The manifest row carries the placement, and `transform_to_world: null`
    /// arrives as nothing rather than as identity.
    func testAManifestRowCarriesItsPlacement() {
        var json = manifestJSON()
        var segments = json["segments"] as! [[String: Any]]
        segments[0]["registration_state"] = "refused"
        segments[0]["registration_refusal_reason"] = "span_over_depth"
        segments[0]["placement_hash"] = "p0"
        json["segments"] = segments

        let manifest = WorldGeometryDecoder.manifest(from: json)
        XCTAssertEqual(manifest?.segments[0].placement.state, .refused)
        XCTAssertEqual(
            manifest?.segments[0].placement.refusalReason, "span_over_depth"
        )
        XCTAssertNil(manifest?.segments[0].transformToWorld)
        XCTAssertEqual(manifest?.segments[0].cacheKey, "h0:p0")
        // The row that carries no placement hash names the absence instead.
        XCTAssertEqual(manifest?.segments[1].cacheKey, "h1:-")
    }

    /// A row that disagrees with itself about its own placement drops the
    /// whole manifest, for the same reason a malformed row does: skipping it
    /// would silently shrink the world.
    func testARowThatContradictsItsOwnPlacementDropsTheWholeManifest() {
        var json = manifestJSON()
        var segments = json["segments"] as! [[String: Any]]
        segments[0]["registered"] = true  // and transform_to_world is still null
        json["segments"] = segments
        XCTAssertNil(WorldGeometryDecoder.manifest(from: json))
    }

    /// A chunk's points are in the SEGMENT's frame. `transform_to_world` is
    /// the only route into the reference segment's frame, and there is no
    /// route at all when the segment is not placed — "not registered" is not
    /// "registered at the origin".
    func testAChunksPointsReachTheReferenceFrameOnlyThroughItsTransform() {
        let placed: [String: Any] = [
            "contract": "world_builder.geometry/2026-08-25",
            "segment_index": 0, "content_hash": "h0", "frame_id": "segment:0",
            "registered": true, "registration_state": "registered",
            "transform_to_world": [
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation": [10.0, 0.0, 0.0], "scale": 2.0,
                "reference_segment": 4, "frame_revision": 1,
            ],
            "placement_hash": "p1",
            "poses": [[String: Any]](), "points": [[1.0, 0.0, 0.0]],
            "points_sent": 1, "points_total": 1, "point_sampling": "none",
        ]
        let chunk = WorldGeometryDecoder.chunk(from: placed)
        XCTAssertEqual(chunk?.transformToWorld?.referenceSegment, 4)
        XCTAssertEqual(chunk?.pointsInReferenceFrame ?? [], [[12.0, 0.0, 0.0]])
        XCTAssertEqual(chunk?.cacheKey, "h0:p1")

        var unplaced = placed
        unplaced["registered"] = false
        unplaced["registration_state"] = "unplaced"
        unplaced["transform_to_world"] = NSNull()
        let loose = WorldGeometryDecoder.chunk(from: unplaced)
        XCTAssertNotNil(loose)
        XCTAssertNil(
            loose?.pointsInReferenceFrame,
            "an unplaced segment's points were offered as though they were in a "
                + "shared frame; they are in its own and nowhere else"
        )
        XCTAssertEqual(loose?.points, [[1.0, 0.0, 0.0]])
    }

    func testASampledChunkKnowsItIsPartial() {
        let json: [String: Any] = [
            "contract": "world_builder.geometry/2026-08-25",
            "segment_index": 0, "content_hash": "h0", "frame_id": "segment:0",
            "registered": false, "transform_to_world": NSNull(),
            "poses": [[String: Any]](), "points": [[1.0, 2.0, 3.0]],
            "points_sent": 1, "points_total": 3000, "point_sampling": "stride",
        ]

        let chunk = WorldGeometryDecoder.chunk(from: json)
        XCTAssertTrue(chunk!.isSampled)
        XCTAssertEqual(chunk?.pointsTotal, 3000)
    }
}

// MARK: - transform_to_world

/// The Sim3, and the four refusals around it.
final class WorldTransformDecoderTests: XCTestCase {

    private func transformJSON(
        rotation: [Double] = [1.0, 0.0, 0.0, 0.0],
        translation: [Double] = [1.0, 2.0, 3.0],
        scale: Double = 0.3533,
        reference: Int = 4,
        frameRevision: Int = 1
    ) -> [String: Any] {
        [
            "rotation_wxyz": rotation, "translation": translation,
            "scale": scale, "reference_segment": reference,
            "frame_revision": frameRevision,
        ]
    }

    func testASim3DecodesFieldForField() {
        let transform = WorldTransform(json: transformJSON())
        XCTAssertEqual(transform?.rotationWXYZ, [1, 0, 0, 0])
        XCTAssertEqual(transform?.translation, [1, 2, 3])
        XCTAssertEqual(transform?.scale, 0.3533)
        XCTAssertEqual(transform?.referenceSegment, 4)
        XCTAssertEqual(transform?.frameRevision, 1)
    }

    /// `p_ref = scale * (R(q) · p_segment) + t`, asserted on a rotation that
    /// would come out differently under any of the plausible wrong readings —
    /// a transposed matrix, an xyzw quaternion, or scale applied before the
    /// rotation instead of after.
    func testAPointIsMappedIntoTheReferenceFrame() throws {
        // +90 degrees about +Y, right-handed. (1,0,0) -> (0,0,-1).
        let s = (0.5 as Double).squareRoot()
        let transform = try XCTUnwrap(
            WorldTransform(
                json: transformJSON(
                    rotation: [s, 0, s, 0], translation: [10, 0, 0], scale: 2
                )
            )
        )

        let mapped = try XCTUnwrap(transform.apply(to: [1, 0, 0]))
        XCTAssertEqual(mapped[0], 10, accuracy: 1e-9)
        XCTAssertEqual(mapped[1], 0, accuracy: 1e-9)
        XCTAssertEqual(mapped[2], -2, accuracy: 1e-9)

        // The identity case, which must be exactly the input plus t.
        let identity = try XCTUnwrap(WorldTransform(json: transformJSON(scale: 1)))
        XCTAssertEqual(try XCTUnwrap(identity.apply(to: [5, 6, 7])), [6, 8, 10])
    }

    func testAShortPointIsRefusedRatherThanIndexedPastItsEnd() throws {
        let transform = try XCTUnwrap(WorldTransform(json: transformJSON()))
        XCTAssertNil(transform.apply(to: [1, 2]))
    }

    /// **A non-unit quaternion scales the geometry it rotates, silently and on
    /// top of `scale`.** The Tower validates at write time; this validates at
    /// read time, because a client that trusts the writer has no way to notice
    /// when the writer is a Tower it has never met.
    func testANonUnitQuaternionIsRefusedRatherThanNormalised() {
        XCTAssertNil(WorldTransform(json: transformJSON(rotation: [2, 0, 0, 0])))
        XCTAssertNil(WorldTransform(json: transformJSON(rotation: [0, 0, 0, 0])))
        XCTAssertNil(
            WorldTransform(json: transformJSON(rotation: [0.5, 0.5, 0.5, 0.6]))
        )
    }

    /// And not tighter than the Tower's own bar. A quaternion serialised at
    /// five decimal places lands ~1.9e-6 off unit, so a validator at 1e-6
    /// would refuse the Tower's own valid output — silently, since the rows
    /// would drop and the world would read as unregistered.
    func testAQuaternionOffUnitByLessThanTheToleranceIsAccepted() {
        XCTAssertEqual(WorldTransform.quaternionNormTolerance, 1e-3)
        XCTAssertNotNil(
            WorldTransform(json: transformJSON(rotation: [1.0000019, 0, 0, 0]))
        )
        XCTAssertNotNil(
            WorldTransform(json: transformJSON(rotation: [0.9999, 0, 0, 0]))
        )
    }

    /// A zero scale collapses the segment to a dot at the reference's origin,
    /// which the registration research records as invisible in every aggregate
    /// metric. A non-finite one is applied all the same and places the
    /// geometry nowhere in particular.
    func testAnUnusableScaleIsRefused() {
        XCTAssertNil(WorldTransform(json: transformJSON(scale: 0)))
        XCTAssertNil(WorldTransform(json: transformJSON(scale: -1)))
        XCTAssertNil(WorldTransform(json: transformJSON(scale: .infinity)))
        XCTAssertNil(WorldTransform(json: transformJSON(scale: .nan)))
    }

    func testAnIncompleteSim3IsRefusedRatherThanDefaulted() {
        // Half a transform means a default supplies the rest, and the default
        // is wrong.
        for key in ["rotation_wxyz", "translation", "scale",
                    "reference_segment", "frame_revision"] {
            var json = transformJSON()
            json.removeValue(forKey: key)
            XCTAssertNil(WorldTransform(json: json), "\(key) was defaulted")
        }
        XCTAssertNil(WorldTransform(json: transformJSON(rotation: [1, 0, 0])))
        XCTAssertNil(WorldTransform(json: transformJSON(translation: [1, 2])))
    }

    /// The contract calls a `frame_revision` mismatch a refuse-to-draw
    /// condition, so a revision that cannot be compared must not reach a
    /// renderer as something to compare against.
    func testAnImpossibleReferenceOrRevisionIsRefused() {
        XCTAssertNil(WorldTransform(json: transformJSON(reference: -1)))
        XCTAssertNil(WorldTransform(json: transformJSON(frameRevision: 0)))
    }

    /// `null` is the wire saying **not registered into any shared frame**.
    /// Decoding it to anything at all — identity most of all — would place the
    /// segment at the reference origin, and the result looks like a map.
    func testANullTransformDecodesToNothingAndNotToIdentity() {
        XCTAssertNil(WorldTransform(json: NSNull()))
        XCTAssertNil(WorldTransform(json: nil))
    }
}

// MARK: - The placement fields, and the rows they refuse

final class WorldPlacementDecoderTests: XCTestCase {

    private static let sim3: [String: Any] = [
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "translation": [1.0, 2.0, 3.0], "scale": 0.5,
        "reference_segment": 4, "frame_revision": 1,
    ]

    func testAPlacedRowCarriesItsStateAndItsSim3() {
        let placement = WorldPlacementFields(json: [
            "registered": true, "registration_state": "registered",
            "registration_refusal_reason": NSNull(),
            "transform_to_world": Self.sim3, "placement_hash": "p1",
        ])
        XCTAssertEqual(placement?.registered, true)
        XCTAssertEqual(placement?.state, .registered)
        XCTAssertNil(placement?.refusalReason)
        XCTAssertEqual(placement?.transform?.referenceSegment, 4)
        XCTAssertEqual(placement?.placementHash, "p1")
    }

    /// `refused` and `unplaced` are the distinction `registered: false` alone
    /// could not carry — "we tried and the solves disagreed" against "nobody
    /// looked" — and on the real corpus the refusal is usually *the wearer
    /// stood still*, which is a message to the wearer.
    func testARefusalKeepsItsReasonAndIsNotJustUnplaced() {
        let refused = WorldPlacementFields(json: [
            "registered": false, "registration_state": "refused",
            "registration_refusal_reason": "span_over_depth",
            "transform_to_world": NSNull(), "placement_hash": "p2",
        ])
        XCTAssertEqual(refused?.state, .refused)
        XCTAssertEqual(refused?.refusalReason, "span_over_depth")

        let unplaced = WorldPlacementFields(json: [
            "registered": false, "registration_state": "unplaced",
            "registration_refusal_reason": NSNull(),
            "transform_to_world": NSNull(), "placement_hash": "p3",
        ])
        XCTAssertEqual(unplaced?.state, .unplaced)
        XCTAssertNil(unplaced?.refusalReason)
        XCTAssertNotEqual(unplaced?.state, refused?.state)
    }

    /// **`registration_state` absent is not `unplaced`.** The whole reason the
    /// field exists is that the bool cannot tell a refusal from an untried
    /// segment, so synthesising `.unplaced` from the bool would fabricate the
    /// distinction the field was added to carry. A Tower that predates the
    /// field has simply not said.
    func testAnAbsentRegistrationStateStaysAbsent() {
        let placement = WorldPlacementFields(json: [
            "registered": false, "transform_to_world": NSNull(),
        ])
        XCTAssertEqual(placement?.registered, false)
        XCTAssertNil(placement?.state)
        XCTAssertNil(placement?.placementHash)
    }

    /// A word this build does not know is not quietly folded into one it does.
    func testAnUnknownRegistrationStateIsNotGuessed() {
        let placement = WorldPlacementFields(json: [
            "registered": false, "registration_state": "provisional",
            "transform_to_world": NSNull(),
        ])
        XCTAssertNil(placement?.state)
    }

    /// A row claiming to be placed and carrying no usable Sim3 has told us it
    /// is placed and not where. There is no safe reading of that, so the row
    /// is refused — which drops the whole payload rather than drawing half a
    /// claim.
    func testARegisteredRowWithNoUsableTransformIsRefused() {
        XCTAssertNil(WorldPlacementFields(json: [
            "registered": true, "registration_state": "registered",
            "transform_to_world": NSNull(),
        ]))
        var broken = Self.sim3
        broken["rotation_wxyz"] = [3.0, 0.0, 0.0, 0.0]
        XCTAssertNil(WorldPlacementFields(json: [
            "registered": true, "registration_state": "registered",
            "transform_to_world": broken,
        ]))
    }

    /// The converse, which the Tower enforces on its own side in the same
    /// words: anything with a transform gets drawn, so a row that is not
    /// registered must not carry one.
    func testAnUnregisteredRowCarryingATransformIsRefused() {
        XCTAssertNil(WorldPlacementFields(json: [
            "registered": false, "registration_state": "refused",
            "transform_to_world": Self.sim3,
        ]))
    }

    func testAMissingRegisteredFlagRefusesTheRow() {
        XCTAssertNil(WorldPlacementFields(json: ["transform_to_world": NSNull()]))
    }
}

// MARK: - The manifest's scale block

/// The truthfulness rule, at the one place the wire states it.
final class WorldGeometryScaleTests: XCTestCase {

    /// **The load-bearing refusal.** `unknown` means no unit at all, which is
    /// a strictly weaker claim than `relative`'s "internally consistent with
    /// an arbitrary unit". Mapping it up would assert a consistency nobody
    /// established.
    func testUnknownIsNeverMappedToRelative() {
        let scale = WorldGeometryScale(
            json: ["state": "unknown", "meters_per_unit": NSNull()]
        )
        XCTAssertEqual(scale.state, .unknown)
        XCTAssertNotEqual(scale.state, .relative)
    }

    /// `meters_per_unit: null` means no metric scale was ever established. It
    /// does not mean 1.0, and there is no `?? 1.0` in this build to make it so.
    func testANullMetersPerUnitIsAbsentAndNotOne() {
        let scale = WorldGeometryScale(
            json: ["state": "relative", "meters_per_unit": NSNull()]
        )
        XCTAssertEqual(scale.state, .relative)
        XCTAssertNil(scale.metersPerUnit)
        XCTAssertFalse(scale.convertibleToMetres)
    }

    /// The two the Tower cannot reach on this hardware are mapped rather than
    /// discarded, so one arriving later is not silently downgraded into a
    /// claim weaker than the one it made.
    func testTheUnreachableStatesAreStillMapped() {
        XCTAssertEqual(
            WorldGeometryScale(json: ["state": "estimated"]).state, .inferredMetric
        )
        XCTAssertEqual(
            WorldGeometryScale(json: ["state": "measured"]).state, .measuredMetric
        )
        // And only `measured` with a real factor ever admits metres, which is
        // the Tower's own rule.
        XCTAssertTrue(
            WorldGeometryScale(
                json: ["state": "measured", "meters_per_unit": 0.5]
            ).convertibleToMetres
        )
        XCTAssertFalse(
            WorldGeometryScale(
                json: ["state": "estimated", "meters_per_unit": 0.5]
            ).convertibleToMetres
        )
    }

    /// A word this build does not know is a claim it cannot verify, so it
    /// reads as the weakest state and never as `relative`.
    func testAnUnknownWordFallsToUnknownAndNotToRelative() {
        XCTAssertEqual(WorldGeometryScale(json: ["state": "metric"]).state, .unknown)
        XCTAssertEqual(WorldGeometryScale(json: [:]).state, .unknown)
        XCTAssertEqual(WorldGeometryScale(json: NSNull()).state, .unknown)
        XCTAssertEqual(WorldGeometryScale(json: nil).state, .unknown)
    }
}

final class WorldGeometryStoreTests: XCTestCase {

    private func chunk(
        index: Int, hash: String, placement: WorldPlacementFields = notRegistered()
    ) -> WorldSegmentChunk {
        WorldSegmentChunk(
            segmentIndex: index, contentHash: hash,
            registered: placement.registered, placement: placement,
            poses: [], points: [[0, 0, 0]], pointsSent: 1, pointsTotal: 1,
            pointSampling: "none"
        )
    }

    func testACachedSegmentIsNotRefetched() async {
        // The property the whole design rests on: a closed segment's GEOMETRY
        // is frozen, so its points and poses cross the wire exactly once —
        // for as long as nothing about where it sits has moved. The key is a
        // pair for precisely that reason; see the placement tests below.
        let store = WorldGeometryStore()
        let cached = chunk(index: 0, hash: "h0")
        await store.insert(cached)

        let needed = await store.keysMissing(from: [cached.cacheKey, "h1:-"])
        XCTAssertEqual(needed, ["h1:-"])
    }

    func testAChangedContentHashIsRefetched() async {
        let store = WorldGeometryStore()
        await store.insert(chunk(index: 0, hash: "h0"))

        let needed = await store.keysMissing(from: ["h0-moved:-"])
        XCTAssertEqual(needed, ["h0-moved:-"])
    }

    /// **The pin that used to say `content_hash` alone was the key, and was
    /// wrong.**
    ///
    /// `content_hash` covers poses and points only, BY DESIGN, so that a
    /// segment that gains a placement keeps it and every cached chunk stays
    /// valid across a registration pass. That is safe only because
    /// `placement_hash` moves instead — and only if the cache is keyed on
    /// both. Keyed on the content half alone this store hands back the
    /// unplaced chunk forever, and the client draws a segment the world knows
    /// how to place in the place it used to have no claim about. Nothing
    /// throws and no tile goes blank.
    func testTheCacheIsKeyedOnContentAndPlacementNotContentAlone() async {
        let store = WorldGeometryStore()
        // Same points. Same content hash. Only the placement moved.
        let unplaced = chunk(
            index: 0, hash: "same-points",
            placement: notRegistered(placementHash: "place-none")
        )
        let registered = chunk(
            index: 0, hash: "same-points",
            placement: placed(into: 4, placementHash: "place-moved")
        )
        XCTAssertEqual(unplaced.contentHash, registered.contentHash)
        XCTAssertNotEqual(
            unplaced.cacheKey, registered.cacheKey,
            "a placement change moved no cache key; the client will never refetch"
        )

        await store.insert(unplaced)
        let stillMissing = await store.keysMissing(from: [registered.cacheKey])
        XCTAssertEqual(stillMissing, [registered.cacheKey])

        await store.insert(registered)
        let placedBack = await store.chunk(forKey: registered.cacheKey)
        XCTAssertEqual(placedBack?.transformToWorld?.referenceSegment, 4)
        // And the old one is still addressable under its own key, so
        // `retainOnly` is what drops it rather than a silent overwrite.
        let unplacedBack = await store.chunk(forKey: unplaced.cacheKey)
        XCTAssertNil(unplacedBack?.transformToWorld)
    }

    /// A registration pass moves every placement hash and no content hash, so
    /// retaining by content alone would leave every superseded chunk behind.
    func testRePlacedChunksAreDroppedByTheManifestsOwnKeys() async {
        let store = WorldGeometryStore()
        let unplaced = chunk(
            index: 0, hash: "same-points",
            placement: notRegistered(placementHash: "place-none")
        )
        let registered = chunk(
            index: 0, hash: "same-points",
            placement: placed(into: 4, placementHash: "place-moved")
        )
        await store.insert(unplaced)
        await store.insert(registered)

        await store.retainOnly([registered.cacheKey])
        // Hoisted rather than written inside the assertion: `XCTAssert…` takes
        // an autoclosure, and an autoclosure cannot carry an `await`.
        let dropped = await store.chunk(forKey: unplaced.cacheKey)
        let kept = await store.chunk(forKey: registered.cacheKey)
        XCTAssertNil(dropped)
        XCTAssertNotNil(kept)
    }

    func testTheCacheIsKeyedByHashNotBySegmentIndex() async {
        // A re-solved segment keeps its index and changes its content. Keying
        // on the index would serve stale geometry under a fresh revision.
        let store = WorldGeometryStore()
        let old = chunk(index: 0, hash: "old")
        let new = chunk(index: 0, hash: "new")
        await store.insert(old)
        await store.insert(new)

        let fromOld = await store.chunk(forKey: old.cacheKey)
        let fromNew = await store.chunk(forKey: new.cacheKey)
        XCTAssertNotNil(fromOld)
        XCTAssertNotNil(fromNew)
        XCTAssertEqual(fromNew?.contentHash, "new")
    }

    /// The absent placement hash is NAMED in the key rather than dropped.
    ///
    /// The Tower added `placement_hash` without bumping
    /// `world_builder.geometry/2026-08-25`, so a conforming Tower may not send
    /// it and the verbatim fixtures in this file do not. `"h0"` and `"h0:"`
    /// would both read as truncations of a key; `"h0:-"` reads as a key with a
    /// stated absence in it.
    func testAnAbsentPlacementHashIsNamedInTheKey() {
        XCTAssertEqual(
            WorldGeometryCacheKey.make(contentHash: "h0", placementHash: nil),
            "h0:-"
        )
        XCTAssertEqual(
            WorldGeometryCacheKey.make(contentHash: "h0", placementHash: "p1"),
            "h0:p1"
        )
        XCTAssertNotEqual(
            WorldGeometryCacheKey.make(contentHash: "h0", placementHash: nil),
            WorldGeometryCacheKey.make(contentHash: "h0", placementHash: "p1")
        )
    }
}

final class WorldFragmentsModelTests: XCTestCase {

    private func summary(
        index: Int, points: Int, state: WorldSegmentResolution,
        bounds: WorldBounds? = nil,
        placement: WorldPlacementFields = notRegistered(),
        coverage: String? = nil
    ) -> WorldSegmentSummary {
        WorldSegmentSummary(
            segmentIndex: index, contentHash: "h\(index)",
            frameID: "segment:\(index)", registered: placement.registered,
            placement: placement,
            resolutionState: state, dominantDegeneracy: "low_parallax",
            keyframeCount: 10, solvedCount: points > 0 ? 5 : 0,
            pointCount: points, bounds: bounds, coverage: coverage
        )
    }

    /// A segment the Tower tried to place and could not, in its own words.
    private func refused(reason: String?) -> WorldPlacementFields {
        WorldPlacementFields(
            registered: false, state: .refused, refusalReason: reason,
            transform: nil, placementHash: nil
        )
    }

    /// A segment nobody has looked at yet.
    private func unplaced() -> WorldPlacementFields {
        WorldPlacementFields(
            registered: false, state: .unplaced, refusalReason: nil,
            transform: nil, placementHash: nil
        )
    }

    private let box = WorldBounds(json: ["min": [-1.0, 0.0, -1.0],
                                         "max": [1.0, 2.0, 1.0]])!

    func testUnregisteredSegmentsAreNeverCompositedIntoOneCanvas() {
        // The load-bearing negative. Segment anchors all sit at the origin and
        // per-segment scale disagrees by up to ~87x on a real walk, so one
        // shared canvas would superimpose independent reconstructions.
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box),
            summary(index: 1, points: 200, state: .resolved, bounds: box),
        ])

        XCTAssertEqual(model.fragments.count, 2)
        XCTAssertFalse(model.hasSharedFrame)
    }

    func testAnUnresolvedSegmentIsCountedButNeverGivenAFragment() {
        // We know reconstruction failed. We do not know where. Drawing it as a
        // region would invent a location.
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box),
            summary(index: 1, points: 0, state: .unresolved),
            summary(index: 2, points: 0, state: .unresolved),
        ])

        XCTAssertEqual(model.fragments.count, 1)
        XCTAssertEqual(model.unresolvedCount, 2)
    }

    func testTheHeadlineCountsFragmentsNotSegments() {
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box),
            summary(index: 1, points: 0, state: .unresolved),
        ])

        XCTAssertEqual(model.headline, "1 fragment, not yet connected")
    }

    /// An empty model no longer answers the question "was anything mapped?",
    /// because it never could.
    ///
    /// This test asserted `model.headline == "Nothing mapped yet"`. That string
    /// was drawn over the 2026-09-06 walk's world — 463 keyframes, 17,674
    /// points — because this model is empty whenever the *fetch* produced
    /// nothing, which includes every way the fetch can fail. The model now
    /// returns `nil` and `WorldGeometryAccount` answers instead, from the
    /// fetch's state and the Tower's own claims. See `WorldPresentationTests`.
    func testAnEmptyModelMakesNoClaimAboutWhetherAnythingWasMapped() {
        let model = WorldFragmentsModel(segments: [])
        XCTAssertTrue(model.fragments.isEmpty)
        XCTAssertNil(
            model.headline,
            "an empty gallery must not assert that nothing was mapped; it does not know"
        )
    }

    /// A model built with no manifest behind it does not claim its geometry is
    /// up to date.
    ///
    /// `isCurrent` defaulted to `true`, so every empty and cleared model — a
    /// gallery that had never fetched anything — asserted currency, and the one
    /// line that would have said otherwise was suppressed by the same default.
    func testAModelWithNoManifestBehindItAssertsNoCurrency() {
        let never = WorldFragmentsModel(segments: [])
        XCTAssertNil(never.isCurrent, "no manifest has said anything about currency")
        XCTAssertNil(never.buildingNote)
    }

    func testAWorldStillBeingBuiltSaysSoRatherThanPassingAsFinished() {
        let model = WorldFragmentsModel(
            segments: [summary(index: 0, points: 100, state: .resolved, bounds: box)],
            isCurrent: false
        )

        XCTAssertEqual(model.fragments.count, 1, "behind geometry is still drawn")
        XCTAssertNotNil(model.buildingNote)
        XCTAssertTrue(model.buildingNote!.contains("still building"))
    }

    func testACurrentWorldAddsNoCaveat() {
        let model = WorldFragmentsModel(
            segments: [summary(index: 0, points: 100, state: .resolved, bounds: box)],
            isCurrent: true
        )
        XCTAssertNil(model.buildingNote)
    }

    func testAResolvedSegmentWithoutBoundsIsNotDrawn() {
        // bounds nil with points > 0 is incoherent; refuse rather than guess
        // a frame for it.
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: nil),
        ])
        XCTAssertTrue(model.fragments.isEmpty)
    }

    func testSegmentsRegisteredIntoTheSameReferenceShareAFrame() {
        // Forward compatibility: when registration lands, the renderer does
        // not change -- the fragments merge. But only these do: same
        // reference, same gauge.
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 4)),
            summary(index: 1, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 4)),
        ])
        XCTAssertTrue(model.hasSharedFrame)
        // One cluster holding every drawable segment, and the headline says
        // how many it holds rather than only that there is one.
        XCTAssertEqual(model.clusters.count, 1)
        XCTAssertTrue(model.unclusteredFragments.isEmpty)
        XCTAssertEqual(model.headline, "1 connected world of 2 segments")
    }

    /// **The refusal `registered` alone cannot express.**
    ///
    /// A Sim3 maps a segment into the frame of its `reference_segment`, and
    /// there is no global world frame — so two segments registered against
    /// different references are in different spaces, and compositing them
    /// fabricates geometry exactly as compositing two unregistered ones would.
    /// Both rows say `registered: true`, which is why the bool is not the test.
    func testSegmentsRegisteredAgainstDifferentReferencesAreNeverComposited() {
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 4)),
            summary(index: 1, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 19)),
        ])

        XCTAssertTrue(model.segments.allSatisfy(\.registered))
        XCTAssertFalse(
            model.hasSharedFrame,
            "two reference frames were composited into one; the map would look "
                + "like a room and mean nothing"
        )
        XCTAssertEqual(model.headline, "2 fragments, not yet connected")
    }

    /// A coordinate stamped with one gauge may not be reinterpreted under
    /// another. The contract calls a `frame_revision` mismatch a refuse-to-draw
    /// condition rather than something to guess past.
    func testSegmentsStampedWithDifferentFrameRevisionsAreNeverComposited() {
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 4, frameRevision: 1)),
            summary(index: 1, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 4, frameRevision: 2)),
        ])
        XCTAssertFalse(model.hasSharedFrame)
    }

    /// **`transform_to_world: null` is not identity.**
    ///
    /// A row that claims `registered: true` and carries no transform has told
    /// us it is placed and not where. Treating the missing Sim3 as identity
    /// would put the segment at the reference origin, which looks entirely
    /// plausible. It is refused instead.
    func testARegisteredSegmentWithNoTransformIsNotTreatedAsIdentity() {
        let placedRow = summary(index: 0, points: 100, state: .resolved,
                                bounds: box, placement: placed(into: 4))
        let claimsPlacedButIsNot = summary(
            index: 1, points: 100, state: .resolved, bounds: box,
            placement: WorldPlacementFields(
                registered: true, state: .registered, refusalReason: nil,
                transform: nil, placementHash: nil
            )
        )

        XCTAssertFalse(placedRow.mayBeCompositedWith(claimsPlacedButIsNot))
        XCTAssertFalse(
            WorldFragmentsModel(segments: [placedRow, claimsPlacedButIsNot])
                .hasSharedFrame
        )
    }

    /// `registered: false` forbids placing two segments in one space outright.
    /// Their scales disagree by up to ~87x on a real walk.
    func testAnUnregisteredSegmentIsNeverCompositedWithARegisteredOne() {
        let placedRow = summary(index: 0, points: 100, state: .resolved,
                                bounds: box, placement: placed(into: 4))
        let unplacedRow = summary(index: 1, points: 100, state: .resolved,
                                  bounds: box)
        XCTAssertFalse(placedRow.mayBeCompositedWith(unplacedRow))
        XCTAssertFalse(unplacedRow.mayBeCompositedWith(placedRow))
        // And an unplaced segment is not even in one space with itself, which
        // is the honest answer: there is no frame to be in.
        XCTAssertFalse(unplacedRow.mayBeCompositedWith(unplacedRow))
    }

    func testEachFragmentIsScaledToItsOwnBoundsAndNeverToASharedOne() {
        // The load-bearing guarantee, asserted directly rather than by proxy.
        // Segments share no coordinate frame and their scales disagree by up to
        // ~87x on a real walk, so a shared scale would composite independent
        // reconstructions into one plausible-looking, meaningless map.
        let size = CGSize(width: 100, height: 100)
        let small = WorldBounds(json: ["min": [0.0, 0.0, 0.0],
                                       "max": [1.0, 0.0, 1.0]])!
        let large = WorldBounds(json: ["min": [0.0, 0.0, 0.0],
                                       "max": [100.0, 0.0, 100.0]])!

        let projectSmall = WorldFragmentsModel.projector(bounds: small, size: size)
        let projectLarge = WorldFragmentsModel.projector(bounds: large, size: size)

        // The same world coordinate lands in different places, because each
        // tile is framed to its own extent.
        XCTAssertNotEqual(projectSmall(1.0, 1.0), projectLarge(1.0, 1.0))

        // And each fragment's own far corner lands at the same relative spot.
        XCTAssertEqual(projectSmall(1.0, 1.0).x,
                       projectLarge(100.0, 100.0).x, accuracy: 0.001)
    }

    func testFragmentsAreOrderedByTheirPointCountDescending() {
        // The grid is about to get crowded -- an unrestricted segmentation
        // takes a real walk to ~470 segments -- and an unordered grid buries
        // the parts of the room that were actually mapped behind the parts
        // that were barely seen. point_count is the Tower's own measure of
        // how much geometry a segment recovered, so it is what orders them.
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 50, state: .resolved, bounds: box),
            summary(index: 1, points: 900, state: .resolved, bounds: box),
            summary(index: 2, points: 300, state: .resolved, bounds: box),
        ])

        XCTAssertEqual(model.fragments.map(\.pointCount), [900, 300, 50])
        XCTAssertEqual(model.fragments.map(\.segmentIndex), [1, 2, 0])
    }

    func testFragmentsTiedOnPointCountFallBackToTheirSegmentIndex() {
        // Swift's sort is not documented as stable, and a ForEach whose order
        // moves between refreshes shuffles cards under the reader's finger.
        // Equal point counts therefore need a second key that is total, and
        // segment_index is unique per manifest.
        let model = WorldFragmentsModel(segments: [
            summary(index: 4, points: 100, state: .resolved, bounds: box),
            summary(index: 0, points: 100, state: .resolved, bounds: box),
            summary(index: 2, points: 100, state: .resolved, bounds: box),
        ])

        XCTAssertEqual(model.fragments.map(\.segmentIndex), [0, 2, 4])

        // The order must be a function of the manifest's CONTENT, not of the
        // order the segments happened to arrive in. `model.fragments` compared
        // against itself would be a tautology -- `ranked` is pure, so that
        // holds for any implementation including an unstable sort and including
        // identity. Feeding the same segments in a different arrival order and
        // demanding the same output is the assertion that actually
        // discriminates.
        let shuffled = WorldFragmentsModel(segments: [
            summary(index: 2, points: 100, state: .resolved, bounds: box),
            summary(index: 4, points: 100, state: .resolved, bounds: box),
            summary(index: 0, points: 100, state: .resolved, bounds: box),
        ])
        XCTAssertEqual(shuffled.fragments.map(\.segmentIndex), [0, 2, 4])
        XCTAssertEqual(shuffled.fragments, model.fragments)
    }

    func testRankingReordersTheGridAndNeverChangesWhatIsInIt() {
        // Ranking is display order only. If it moved membership -- either the
        // resolved-with-bounds filter or the unresolved tally -- something is
        // wrong, and both numbers are read by other surfaces.
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 10, state: .resolved, bounds: box),
            summary(index: 1, points: 0, state: .unresolved),
            summary(index: 2, points: 900, state: .resolved, bounds: box),
            summary(index: 3, points: 700, state: .resolved, bounds: nil),
            summary(index: 4, points: 0, state: .unresolved),
        ])

        // The incoherent segment 3 (points, no bounds) stays out, and neither
        // unresolved segment is promoted into the grid by its ranking.
        XCTAssertEqual(model.fragments.map(\.segmentIndex), [2, 0])
        XCTAssertEqual(model.fragments.count, 2)
        XCTAssertEqual(model.unresolvedCount, 2)
        XCTAssertEqual(model.headline, "2 fragments, not yet connected")
    }

    /// A degenerate-case guard, and named as one.
    ///
    /// This passes whether or not `ranked` does anything at all -- zero and one
    /// element have exactly one possible order. It is kept because sorting an
    /// empty collection is a classic crash site, not because it pins ranking;
    /// the tests above are what would fail if `ranked` were reverted to
    /// identity. Naming it "ranking..." would claim coverage it does not have.
    func testAnEmptyOrSingleFragmentWorldSurvivesBeingOrdered() {
        let empty = WorldFragmentsModel(segments: [])
        XCTAssertTrue(empty.fragments.isEmpty)

        let one = WorldFragmentsModel(segments: [
            summary(index: 7, points: 42, state: .resolved, bounds: box),
        ])
        XCTAssertEqual(one.fragments.map(\.segmentIndex), [7])
    }

    // MARK: Connected worlds beside loose fragments

    /// The headline names both halves and never adds them: two segments in
    /// one frame are "1 connected world of 2 segments", and the third, which
    /// shares a frame with nobody, is counted on its own.
    func testAConnectedWorldAndLooseFragmentsAreCountedSeparately() {
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 4)),
            summary(index: 1, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 4)),
            summary(index: 2, points: 900, state: .resolved, bounds: box),
        ])

        XCTAssertEqual(model.clusters.count, 1)
        XCTAssertEqual(model.clusters.first?.memberIndexes, [0, 1])
        XCTAssertEqual(model.unclusteredFragments.map(\.segmentIndex), [2])
        XCTAssertEqual(
            model.headline,
            "1 connected world of 2 segments, 1 fragment not yet connected"
        )
    }

    /// Two references are two groups. They are counted together in the
    /// headline and drawn on two canvases — never on one.
    func testTwoConnectedGroupsAreNamedAsTwoAndNeverMerged() {
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 4)),
            summary(index: 1, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 4)),
            summary(index: 2, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 19)),
            summary(index: 3, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 19)),
        ])

        XCTAssertEqual(model.clusters.count, 2)
        XCTAssertTrue(model.unclusteredFragments.isEmpty)
        XCTAssertEqual(model.headline, "2 connected groups of 4 segments")
        for cluster in model.clusters {
            XCTAssertEqual(cluster.members.count, 2)
            XCTAssertTrue(
                cluster.members.allSatisfy { $0.transformToWorld?.referenceSegment == cluster.referenceSegment },
                "a cluster holds a segment from another reference frame"
            )
        }
    }

    /// A registered segment whose reference nobody else shares is a group of
    /// one, which is not "connected" to anything: it keeps its tile.
    func testALoneRegisteredSegmentStaysATileNotAWorld() {
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 4)),
            summary(index: 1, points: 50, state: .resolved, bounds: box),
        ])
        XCTAssertTrue(model.clusters.isEmpty)
        XCTAssertEqual(model.unclusteredFragments.map(\.segmentIndex), [0, 1])
        XCTAssertEqual(model.headline, "2 fragments, not yet connected")
    }

    // MARK: Coverage

    /// `partial` is drawn muted and only `partial` is. `confident` is drawn as
    /// it was, an unknown word is drawn as it was — muting it would be a
    /// judgment the Tower did not make — and `nil` is nothing at all.
    func testOnlyPartialCoverageIsMuted() {
        XCTAssertTrue(WorldFragmentsModel.isMuted(
            summary(index: 0, points: 10, state: .resolved, bounds: box, coverage: "partial")
        ))
        XCTAssertFalse(WorldFragmentsModel.isMuted(
            summary(index: 0, points: 10, state: .resolved, bounds: box, coverage: "confident")
        ))
        XCTAssertFalse(WorldFragmentsModel.isMuted(
            summary(index: 0, points: 10, state: .resolved, bounds: box, coverage: "speculative")
        ))
        XCTAssertFalse(WorldFragmentsModel.isMuted(
            summary(index: 0, points: 10, state: .resolved, bounds: box, coverage: nil)
        ))
    }

    /// A segment the solve called `unresolved` is counted with the other
    /// unresolved ones and is never given a tile, whatever `resolution_state`
    /// and `bounds` happen to say. Counted once, not twice, when both fields
    /// agree.
    func testUnresolvedCoverageIsCountedAndNeverPlaced() {
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box, coverage: "confident"),
            summary(index: 1, points: 3, state: .resolved, bounds: box, coverage: "unresolved"),
            summary(index: 2, points: 0, state: .unresolved, coverage: "unresolved"),
            summary(index: 3, points: 0, state: .unresolved),
        ])
        XCTAssertEqual(model.fragments.map(\.segmentIndex), [0])
        XCTAssertEqual(model.unresolvedCount, 3)
    }

    /// The tile caption is the Tower's registration word and, when it gave
    /// one, its refusal reason verbatim. No state, no caption: a Tower that
    /// predates `registration_state` has said nothing to repeat.
    func testTilesCaptionTheTowersWordsVerbatim() {
        XCTAssertEqual(
            WorldFragmentsModel.placementCaption(for: summary(
                index: 0, points: 10, state: .resolved, bounds: box,
                placement: refused(reason: "the wearer stood still")
            )),
            "refused — the wearer stood still"
        )
        XCTAssertEqual(
            WorldFragmentsModel.placementCaption(for: summary(
                index: 0, points: 10, state: .resolved, bounds: box,
                placement: refused(reason: nil)
            )),
            "refused"
        )
        XCTAssertEqual(
            WorldFragmentsModel.placementCaption(for: summary(
                index: 0, points: 10, state: .resolved, bounds: box,
                placement: placed(into: 4)
            )),
            "registered"
        )
        XCTAssertEqual(
            WorldFragmentsModel.placementCaption(for: summary(
                index: 0, points: 10, state: .resolved, bounds: box,
                placement: unplaced()
            )),
            "unplaced"
        )
        XCTAssertNil(
            WorldFragmentsModel.placementCaption(for: summary(
                index: 0, points: 10, state: .resolved, bounds: box
            ))
        )

        XCTAssertEqual(
            WorldFragmentsModel.coverageCaption(for: summary(
                index: 0, points: 10, state: .resolved, bounds: box, coverage: "partial"
            )),
            "coverage partial"
        )
        XCTAssertNil(
            WorldFragmentsModel.coverageCaption(for: summary(
                index: 0, points: 10, state: .resolved, bounds: box
            ))
        )
    }

    /// Counted by the Tower's `registration_state`, never inferred from the
    /// bool, and a row that carries no state is counted as unknown rather
    /// than as unplaced.
    func testPlacementCountsFollowTheTowersStateAndOmitZeroes() {
        let model = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box,
                    placement: placed(into: 4)),
            summary(index: 1, points: 10, state: .resolved, bounds: box,
                    placement: refused(reason: "the wearer stood still")),
            summary(index: 2, points: 0, state: .unresolved,
                    placement: refused(reason: "the wearer stood still")),
            summary(index: 3, points: 10, state: .resolved, bounds: box,
                    placement: unplaced()),
            summary(index: 4, points: 10, state: .resolved, bounds: box),
        ])
        XCTAssertEqual(model.registeredCount, 1)
        XCTAssertEqual(model.refusedCount, 2)
        XCTAssertEqual(model.unplacedCount, 1)
        XCTAssertEqual(model.placementUnknownCount, 1)
        XCTAssertEqual(model.placementSummary, "1 registered, 2 refused, 1 unplaced")

        let olderTower = WorldFragmentsModel(segments: [
            summary(index: 0, points: 100, state: .resolved, bounds: box),
        ])
        XCTAssertEqual(olderTower.placementUnknownCount, 1)
        XCTAssertNil(olderTower.placementSummary, "counts were invented for a Tower that sent none")
    }

}

// MARK: - Clusters

/// `WorldClusterBuilder` on its own: the grouping and the box around a group,
/// with no view in the loop.
final class WorldClusterBuilderTests: XCTestCase {

    private let box = WorldBounds(json: ["min": [-1.0, 0.0, -1.0],
                                         "max": [1.0, 2.0, 1.0]])!

    private func summary(
        index: Int, points: Int = 100, placement: WorldPlacementFields
    ) -> WorldSegmentSummary {
        WorldSegmentSummary(
            segmentIndex: index, contentHash: "h\(index)",
            frameID: "segment:\(index)", registered: placement.registered,
            placement: placement, resolutionState: .resolved,
            dominantDegeneracy: nil, keyframeCount: 4, solvedCount: 3,
            pointCount: points, bounds: box, coverage: nil
        )
    }

    /// A placement with a real Sim3, for the bounds tests. Identity rotation
    /// so the arithmetic under test is the scale and translation.
    private func placedWith(
        translation: [Double], scale: Double, into reference: Int = 4, hash: String
    ) -> WorldPlacementFields {
        WorldPlacementFields(
            registered: true, state: .registered, refusalReason: nil,
            transform: WorldTransform(
                rotationWXYZ: [1, 0, 0, 0], translation: translation, scale: scale,
                referenceSegment: reference, frameRevision: 1
            ),
            placementHash: hash
        )
    }

    private func chunk(
        for summary: WorldSegmentSummary, points: [[Double]], poses: [WorldPose] = []
    ) -> WorldSegmentChunk {
        WorldSegmentChunk(
            segmentIndex: summary.segmentIndex, contentHash: summary.contentHash,
            registered: summary.registered, placement: summary.placement,
            poses: poses, points: points,
            pointsSent: points.count, pointsTotal: points.count, pointSampling: "none"
        )
    }

    func testSegmentsSharingAReferenceFormOneCluster() {
        let clusters = WorldClusterBuilder.clusters(from: [
            summary(index: 0, placement: placed(into: 4)),
            summary(index: 1, placement: placed(into: 4)),
            summary(index: 2, placement: placed(into: 4)),
        ])
        XCTAssertEqual(clusters.count, 1)
        XCTAssertEqual(clusters.first?.referenceSegment, 4)
        XCTAssertEqual(clusters.first?.frameRevision, 1)
        XCTAssertEqual(clusters.first?.memberIndexes, [0, 1, 2])
    }

    /// **Different references never cluster.** A Sim3 maps into its own
    /// reference's frame and there is no global frame above that, so two
    /// registered segments naming different references share no space.
    func testDifferentReferencesNeverCluster() {
        XCTAssertTrue(
            WorldClusterBuilder.clusters(from: [
                summary(index: 0, placement: placed(into: 4)),
                summary(index: 1, placement: placed(into: 19)),
            ]).isEmpty
        )

        let two = WorldClusterBuilder.clusters(from: [
            summary(index: 0, placement: placed(into: 4)),
            summary(index: 1, placement: placed(into: 19)),
            summary(index: 2, placement: placed(into: 4)),
            summary(index: 3, placement: placed(into: 19)),
        ])
        XCTAssertEqual(two.count, 2)
        XCTAssertEqual(two.map(\.referenceSegment), [4, 19])
        XCTAssertEqual(two[0].memberIndexes, [0, 2])
        XCTAssertEqual(two[1].memberIndexes, [1, 3])
    }

    /// A coordinate stamped with one gauge may not be reinterpreted under
    /// another.
    func testDifferentFrameRevisionsNeverCluster() {
        XCTAssertTrue(
            WorldClusterBuilder.clusters(from: [
                summary(index: 0, placement: placed(into: 4, frameRevision: 1)),
                summary(index: 1, placement: placed(into: 4, frameRevision: 2)),
            ]).isEmpty
        )
    }

    /// `registered: false` forbids it outright, and an unregistered segment
    /// beside two registered ones is left out rather than pulled in.
    func testUnregisteredSegmentsNeverCluster() {
        XCTAssertTrue(
            WorldClusterBuilder.clusters(from: [
                summary(index: 0, placement: notRegistered()),
                summary(index: 1, placement: notRegistered()),
            ]).isEmpty
        )

        let mixed = WorldClusterBuilder.clusters(from: [
            summary(index: 0, placement: placed(into: 4)),
            summary(index: 1, placement: notRegistered()),
            summary(index: 2, placement: placed(into: 4)),
        ])
        XCTAssertEqual(mixed.count, 1)
        XCTAssertEqual(mixed.first?.memberIndexes, [0, 2])
    }

    func testAGroupOfOneIsNotACluster() {
        XCTAssertTrue(
            WorldClusterBuilder.clusters(from: [
                summary(index: 0, placement: placed(into: 4)),
            ]).isEmpty
        )
    }

    /// Members are in gallery order — most points first, index breaking ties
    /// — so a cluster's caption reads the way the grid does.
    func testAClusterOrdersItsMembersLikeTheGallery() {
        let clusters = WorldClusterBuilder.clusters(from: [
            summary(index: 0, points: 50, placement: placed(into: 4)),
            summary(index: 1, points: 900, placement: placed(into: 4)),
            summary(index: 2, points: 50, placement: placed(into: 4)),
        ])
        XCTAssertEqual(clusters.first?.members.map(\.segmentIndex), [1, 0, 2])
        XCTAssertEqual(clusters.first?.pointCount, 1000)
    }

    /// A box over the members that happened to arrive would frame a partial
    /// world as the whole one; until every chunk is in hand there is no box.
    func testBoundsAreNilWhenAnyMemberChunkIsMissing() throws {
        let a = summary(index: 0, placement: placedWith(translation: [0, 0, 0], scale: 1, hash: "pa"))
        let b = summary(index: 1, placement: placedWith(translation: [0, 0, 0], scale: 1, hash: "pb"))
        let cluster = try XCTUnwrap(WorldClusterBuilder.clusters(from: [a, b]).first)

        let onlyA = [a.cacheKey: chunk(for: a, points: [[1, 1, 1]])]
        XCTAssertNil(WorldClusterBuilder.referenceFrameBounds(of: cluster, chunks: onlyA))

        let both = [
            a.cacheKey: chunk(for: a, points: [[1, 1, 1]]),
            b.cacheKey: chunk(for: b, points: [[2, 2, 2]]),
        ]
        XCTAssertNotNil(WorldClusterBuilder.referenceFrameBounds(of: cluster, chunks: both))
    }

    /// The box is over the **transformed** points, and the corner it reports
    /// is exactly what `WorldTransform.apply` says that corner is. A box over
    /// the local points would place every member at its own origin — the
    /// identity-transform failure §6 names.
    func testBoundsAreTheTransformAppliedToEachMembersPoints() throws {
        let moved = summary(
            index: 0, placement: placedWith(translation: [10, 0, -5], scale: 2, hash: "pm")
        )
        let still = summary(
            index: 1, placement: placedWith(translation: [0, 0, 0], scale: 1, hash: "ps")
        )
        let cluster = try XCTUnwrap(WorldClusterBuilder.clusters(from: [moved, still]).first)
        let chunks = [
            moved.cacheKey: chunk(for: moved, points: [[1, 2, 3]]),
            still.cacheKey: chunk(for: still, points: [[-1, -1, -1]]),
        ]

        let bounds = try XCTUnwrap(
            WorldClusterBuilder.referenceFrameBounds(of: cluster, chunks: chunks)
        )
        // 2 · [1, 2, 3] + [10, 0, -5] = [12, 4, 1].
        XCTAssertEqual(bounds.max, [12, 4, 1])
        XCTAssertEqual(bounds.min, [-1, -1, -1])
        let transform = try XCTUnwrap(moved.transformToWorld)
        XCTAssertEqual(bounds.max, transform.apply(to: [1, 2, 3]))
    }

    /// The camera path is inside the box too, in the same frame, and a refused
    /// pose contributes nothing rather than a point at the origin.
    func testPoseTranslationsAreInsideTheBoundsAndRefusedPosesAreNot() throws {
        let a = summary(index: 0, placement: placedWith(translation: [0, 0, 0], scale: 1, hash: "pa"))
        let b = summary(index: 1, placement: placedWith(translation: [0, 0, 0], scale: 1, hash: "pb"))
        let cluster = try XCTUnwrap(WorldClusterBuilder.clusters(from: [a, b]).first)
        let chunks = [
            a.cacheKey: chunk(
                for: a, points: [[1, 1, 1]],
                poses: [
                    WorldPose(keyframeID: "k0", status: "solved", degeneracy: "",
                              rotation: [1, 0, 0, 0], translation: [100, 1, 1]),
                    WorldPose(keyframeID: "k1", status: "refused", degeneracy: "low_parallax",
                              rotation: nil, translation: nil),
                ]
            ),
            b.cacheKey: chunk(for: b, points: [[2, 2, 2]]),
        ]

        let bounds = try XCTUnwrap(
            WorldClusterBuilder.referenceFrameBounds(of: cluster, chunks: chunks)
        )
        XCTAssertEqual(bounds.max[0], 100)
        XCTAssertEqual(bounds.min, [1, 1, 1], "a refused pose was placed at the origin")
    }
}

// MARK: - The viewport

/// The pan/zoom/rotate arithmetic, with no gesture in the loop.
final class WorldViewportStateTests: XCTestCase {

    /// Centre `(100, 50)`.
    private let size = CGSize(width: 200, height: 100)

    private func assertEqual(
        _ actual: CGPoint, _ expected: CGPoint,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.x, expected.x, accuracy: 1e-9, file: file, line: line)
        XCTAssertEqual(actual.y, expected.y, accuracy: 1e-9, file: file, line: line)
    }

    func testTheIdentityStateMovesNothing() {
        assertEqual(
            WorldViewportState.identity.apply(CGPoint(x: 30, y: 70), in: size),
            CGPoint(x: 30, y: 70)
        )
    }

    /// A pinch zooms into the middle of the canvas, not into its top-left
    /// corner: the centre stays put and everything else moves away from it.
    func testScaleIsAboutTheCentre() {
        var state = WorldViewportState.identity
        state.scale = 2
        assertEqual(state.apply(CGPoint(x: 100, y: 50), in: size), CGPoint(x: 100, y: 50))
        assertEqual(state.apply(CGPoint(x: 110, y: 50), in: size), CGPoint(x: 120, y: 50))
        assertEqual(state.apply(CGPoint(x: 100, y: 40), in: size), CGPoint(x: 100, y: 30))
    }

    /// The pan is applied last, in canvas points, so it is the same slide at
    /// every zoom.
    func testOffsetSlidesAfterScaling() {
        var state = WorldViewportState.identity
        state.scale = 2
        state.offset = CGSize(width: 5, height: -5)
        assertEqual(state.apply(CGPoint(x: 110, y: 50), in: size), CGPoint(x: 125, y: 45))
    }

    /// A quarter turn about the centre takes a point one unit to the right of
    /// it to one unit below it, in CoreGraphics' own convention — which is the
    /// convention `GraphicsContext` draws in, so nothing here flips a sign.
    func testRotationIsAboutTheCentre() {
        var state = WorldViewportState.identity
        state.rotation = .pi / 2
        assertEqual(state.apply(CGPoint(x: 100, y: 50), in: size), CGPoint(x: 100, y: 50))
        assertEqual(state.apply(CGPoint(x: 101, y: 50), in: size), CGPoint(x: 100, y: 51))
    }

    func testScaleIsClampedToTheRange() {
        XCTAssertEqual(WorldViewportState.clampedScale(0.01), 0.25)
        XCTAssertEqual(WorldViewportState.clampedScale(100), 20)
        XCTAssertEqual(WorldViewportState.clampedScale(3), 3)

        var state = WorldViewportState.identity
        state.scale = 1000
        XCTAssertEqual(state.clamped().scale, 20)
        XCTAssertEqual(state.scale, 1000, "`clamped()` mutated its receiver")
    }
}

// MARK: - The saved-worlds listing

/// `GET /worlds`, decoded. Refusals are the point: a wrong contract or a row
/// that will not decode drops the whole listing rather than offering a world
/// that was half-read.
final class WorldListingDecoderTests: XCTestCase {

    /// Two worlds, newest first as the Tower orders them; the second has no
    /// display name and no sessions. Timestamps are written as `Double`
    /// literals because a Swift `Int` boxed in `Any` does not cast to
    /// `Double` — on the wire every number is an `NSNumber`, and the JSON-text
    /// test below covers that path.
    private func listingJSON() -> [String: Any] {
        [
            "contract": "world_builder.worlds/2026-09-10",
            "worlds": [
                [
                    "world_id": "w-new", "display_name": "Kitchen walk",
                    "created_at": 1787463000.0, "updated_at": 1787463900.5,
                    "live": true,
                    "sessions": [
                        [
                            "session_id": "s-open", "started_at": 1787463001.0,
                            "ended_at": NSNull(), "end_reason": NSNull(),
                            "frame_source": "live-capture", "capture_id": "c1",
                            "has_geometry": true,
                        ],
                        [
                            "session_id": "s-done", "started_at": 1787462000.0,
                            "ended_at": 1787462500.0, "end_reason": "stop",
                            "frame_source": "recorded-capture", "capture_id": NSNull(),
                            "has_geometry": false,
                        ],
                    ],
                ],
                [
                    "world_id": "w-old", "display_name": NSNull(),
                    "created_at": 1787400000.0, "updated_at": 1787400000.0,
                    "live": false, "sessions": [[String: Any]](),
                ],
            ],
        ]
    }

    func testAListingDecodesFieldForFieldInTheTowersOrder() throws {
        let listing = try XCTUnwrap(WorldListingDecoder.listing(from: listingJSON()))
        XCTAssertEqual(listing.worlds.map(\.worldID), ["w-new", "w-old"])

        let newest = listing.worlds[0]
        XCTAssertEqual(newest.displayName, "Kitchen walk")
        XCTAssertEqual(newest.title, "Kitchen walk")
        XCTAssertTrue(newest.live)
        XCTAssertEqual(newest.createdAt, 1787463000.0)
        XCTAssertEqual(newest.sessions.count, 2)

        let open = newest.sessions[0]
        XCTAssertEqual(open.sessionID, "s-open")
        XCTAssertNil(open.endedAt)
        XCTAssertTrue(open.isStillOpen)
        XCTAssertEqual(open.frameSource, "live-capture")
        XCTAssertEqual(open.captureID, "c1")
        XCTAssertTrue(open.hasGeometry)

        let done = newest.sessions[1]
        XCTAssertEqual(done.endedAt, 1787462500.0)
        XCTAssertFalse(done.isStillOpen)
        XCTAssertEqual(done.endReason, "stop")
        XCTAssertNil(done.captureID)
        XCTAssertFalse(done.hasGeometry)
    }

    /// `display_name: null` is `nil`, and the picker falls back to a dated
    /// title from `updated_at` — never to an empty row, and no longer to the
    /// id, which 156 of 162 worlds on the real root would have shared the
    /// shape of. The id stays on the entry for a caption.
    func testANullDisplayNameFallsBackToADatedTitleAndKeepsTheID() throws {
        let listing = try XCTUnwrap(WorldListingDecoder.listing(from: listingJSON()))
        let unnamed = listing.worlds[1]
        XCTAssertNil(unnamed.displayName)
        XCTAssertEqual(
            unnamed.title,
            WorldListingPresentation.datedTitle(updatedAt: 1787400000.0)
        )
        XCTAssertTrue(unnamed.title.hasPrefix("Walk · "), unnamed.title)
        XCTAssertEqual(unnamed.worldID, "w-old")
        XCTAssertFalse(unnamed.live)
        XCTAssertTrue(unnamed.sessions.isEmpty)
        XCTAssertFalse(unnamed.hasSessions)
    }

    /// The fields an older Tower does not send decode as absent, not as
    /// zero or `false`, and the row still reads from what it did send.
    func testAnOldListingWithoutTheAdditiveFieldsStillDecodes() throws {
        let listing = try XCTUnwrap(WorldListingDecoder.listing(from: listingJSON()))
        let open = listing.worlds[0].sessions[0]
        XCTAssertNil(open.state)
        XCTAssertNil(open.abandoned)
        XCTAssertNil(open.keyframesAccepted)
        XCTAssertNil(open.keyframesJournaled)
        XCTAssertNil(open.finalization)
        XCTAssertNil(open.keyframeCount)
        // With no `abandoned` to say otherwise, an open record is all that
        // is known, and "still open" is what an old Tower's row says.
        XCTAssertTrue(open.isStillOpen)
        XCTAssertEqual(WorldListingPresentation.stateBadge(for: open), "still open")
        XCTAssertNil(WorldListingPresentation.keyframeCaption(for: open))
    }

    /// The additive fields of `world_builder.worlds/2026-09-10`, as the Tower
    /// sends them: `state`, `keyframes_accepted`, `keyframes_journaled`,
    /// `abandoned`, `finalization`. Through `JSONSerialization`, so the
    /// integers arrive as the `NSNumber`s they will be on the phone.
    func testANewListingDecodesItsAdditiveFields() throws {
        let text = """
            {"contract":"world_builder.worlds/2026-09-10",
             "worlds":[{"world_id":"fcbca9e90b244785bdb671530b33c6a5","display_name":null,
                        "created_at":1788894856,"updated_at":1788895032,
                        "live":false,
                        "sessions":[{"session_id":"158ef0efb5e5416b87d6faa8f5c28e55",
                                     "started_at":1788894857,
                                     "ended_at":null,"end_reason":null,
                                     "frame_source":"live-capture",
                                     "capture_id":"7febdae8",
                                     "keyframes_accepted":0,
                                     "keyframes_journaled":467,
                                     "has_geometry":true,
                                     "abandoned":true,
                                     "state":"interrupted",
                                     "finalization":null},
                                    {"session_id":"s-done","started_at":1788895100,
                                     "ended_at":1788895400,"end_reason":"stop",
                                     "frame_source":"live-capture","capture_id":"c2",
                                     "keyframes_accepted":120,"keyframes_journaled":120,
                                     "has_geometry":true,"abandoned":false,
                                     "state":"complete",
                                     "finalization":{"state":"complete","final_solve":"solved",
                                                     "started_at":1788895400.5,
                                                     "updated_at":1788895460.0,
                                                     "detail":null}}]}]}
            """
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        )
        let listing = try XCTUnwrap(WorldListingDecoder.listing(from: json))
        let world = try XCTUnwrap(listing.worlds.first)
        XCTAssertEqual(world.sessions.count, 2)

        let interrupted = world.sessions[0]
        XCTAssertEqual(interrupted.state, .interrupted)
        XCTAssertEqual(interrupted.abandoned, true)
        XCTAssertEqual(interrupted.keyframesAccepted, 0)
        XCTAssertEqual(interrupted.keyframesJournaled, 467)
        XCTAssertNil(interrupted.finalization)
        XCTAssertNil(interrupted.endedAt)

        let complete = world.sessions[1]
        XCTAssertEqual(complete.state, .complete)
        XCTAssertEqual(complete.abandoned, false)
        let finalization = try XCTUnwrap(complete.finalization)
        XCTAssertEqual(finalization.state, .complete)
        XCTAssertEqual(finalization.finalSolve, "solved")
        XCTAssertEqual(finalization.startedAt, 1788895400.5)
        XCTAssertEqual(finalization.updatedAt, 1788895460.0)
        XCTAssertNil(finalization.detail)
    }

    /// The 2026-09-06 walk's own record: `ended_at: null` with a dead
    /// builder. It is **not** "still open" — the Tower said `abandoned` — and
    /// the keyframe figure is the journal's 467, not the record's
    /// start-of-session zero.
    func testAnAbandonedSessionIsNotStillOpen() throws {
        var json = listingJSON()
        var worlds = try XCTUnwrap(json["worlds"] as? [[String: Any]])
        var sessions = try XCTUnwrap(worlds[0]["sessions"] as? [[String: Any]])
        sessions[0]["abandoned"] = true
        sessions[0]["keyframes_accepted"] = 0
        sessions[0]["keyframes_journaled"] = 467
        worlds[0]["sessions"] = sessions
        json["worlds"] = worlds

        let listing = try XCTUnwrap(WorldListingDecoder.listing(from: json))
        let session = listing.worlds[0].sessions[0]
        XCTAssertNil(session.endedAt)
        XCTAssertFalse(session.isStillOpen, "a dead builder's record read as still open")
        XCTAssertEqual(session.keyframeCount, 467)
        XCTAssertEqual(WorldListingPresentation.keyframeCaption(for: session), "467 keyframes")
        // No `state` word yet, but `abandoned` is enough for the badge.
        XCTAssertEqual(WorldListingPresentation.stateBadge(for: session), "unfinished")
    }

    /// An unknown `state` word survives as itself rather than being mapped to
    /// the nearest known one — the Tower owns the vocabulary.
    func testAnUnknownSessionStateWordSurvivesAsItself() throws {
        var json = listingJSON()
        var worlds = try XCTUnwrap(json["worlds"] as? [[String: Any]])
        var sessions = try XCTUnwrap(worlds[0]["sessions"] as? [[String: Any]])
        sessions[1]["state"] = "consolidating"
        worlds[0]["sessions"] = sessions
        json["worlds"] = worlds
        let listing = try XCTUnwrap(WorldListingDecoder.listing(from: json))
        let session = listing.worlds[0].sessions[1]
        XCTAssertEqual(session.state, WorldListingSessionState(rawValue: "consolidating"))
        XCTAssertEqual(WorldListingPresentation.stateBadge(for: session), "consolidating")
    }

    func testAnotherContractIsRefusedWhole() {
        var json = listingJSON()
        json["contract"] = "world_builder.worlds/2027-01-01"
        XCTAssertNil(WorldListingDecoder.listing(from: json))
        json["contract"] = nil
        XCTAssertNil(WorldListingDecoder.listing(from: json))
    }

    /// One session missing a required field drops the listing, not the
    /// session: a world offered with the sessions this build could read would
    /// be a world it half-read.
    func testAMalformedSessionRowDropsTheWholeListing() throws {
        var json = listingJSON()
        var worlds = try XCTUnwrap(json["worlds"] as? [[String: Any]])
        var sessions = try XCTUnwrap(worlds[0]["sessions"] as? [[String: Any]])
        sessions[1]["frame_source"] = nil
        worlds[0]["sessions"] = sessions
        json["worlds"] = worlds
        XCTAssertNil(WorldListingDecoder.listing(from: json))
    }

    /// `has_geometry` and `live` are required and absent is not `false`.
    func testAMissingBoolIsARefusalNotFalse() throws {
        var json = listingJSON()
        var worlds = try XCTUnwrap(json["worlds"] as? [[String: Any]])
        worlds[1]["live"] = nil
        json["worlds"] = worlds
        XCTAssertNil(WorldListingDecoder.listing(from: json))
    }

    /// The bytes as the Tower would send them, through `JSONSerialization`,
    /// with an integer timestamp — which arrives as an `NSNumber` and must
    /// still read as a `Double`.
    func testAListingDecodesFromJSONTextWithIntegerTimestamps() throws {
        let text = """
            {"contract":"world_builder.worlds/2026-09-10",
             "worlds":[{"world_id":"w1","display_name":null,
                        "created_at":1787463000,"updated_at":1787463900,
                        "live":false,
                        "sessions":[{"session_id":"s1","started_at":1787463001,
                                     "ended_at":null,"end_reason":null,
                                     "frame_source":"live-capture","capture_id":null,
                                     "has_geometry":true}]}]}
            """
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        )
        let listing = try XCTUnwrap(WorldListingDecoder.listing(from: json))
        XCTAssertEqual(listing.worlds.first?.createdAt, 1787463000)
        XCTAssertEqual(listing.worlds.first?.sessions.first?.startedAt, 1787463001)
        XCTAssertNil(listing.worlds.first?.sessions.first?.endedAt)
    }

    func testTheListingContractIsItsOwn() {
        XCTAssertEqual(WorldListingContract.identifier, "world_builder.worlds/2026-09-10")
        XCTAssertNotEqual(WorldListingContract.identifier, WorldGeometryContract.identifier)
        XCTAssertNotEqual(WorldListingContract.identifier, WorldBuilderResultContract.identifier)
    }
}

// MARK: - The picker at the size of a real root

/// `WorldListingPresentation` against a synthetic root shaped like the real
/// one on 2026-09-06: 161 worlds, 96 with no sessions, 29 interrupted, 6
/// named, the rest a mix of complete, unbuilt, finishing and building. The
/// picker was written against two worlds and had never been asked what this
/// looks like; these are the assertions that would have said.
final class WorldListingPresentationTests: XCTestCase {

    /// Newest first, as the Tower orders them. Worlds are dealt into the
    /// roles by index so the mix is spread through the list rather than
    /// clumped at one end.
    private func syntheticRoot() throws -> WorldListing {
        var worlds: [[String: Any]] = []
        let base = 1788800000.0
        var interrupted = 0, named = 0, empty = 0
        for index in 0..<161 {
            let updated = base - Double(index) * 600
            let worldID = String(format: "w%03d", index)
            var world: [String: Any] = [
                "world_id": worldID,
                "display_name": NSNull(),
                "created_at": updated - 300,
                "updated_at": updated,
                "live": false,
            ]
            // 96 empty shells: every world whose index leaves 0 or 1 (mod 3)
            // after the first 17, which lands on exactly 96 of 161.
            let isEmpty = index >= 17 && index % 3 != 2
            if isEmpty {
                empty += 1
                world["sessions"] = [[String: Any]]()
                worlds.append(world)
                continue
            }
            if named < 6 {
                named += 1
                world["display_name"] = "Kitchen walk \(named)"
            }
            var session: [String: Any] = [
                "session_id": "s-\(worldID)",
                "started_at": updated - 200,
                "frame_source": "live-capture",
                "capture_id": "c-\(worldID)",
            ]
            if interrupted < 29 {
                interrupted += 1
                session["ended_at"] = NSNull()
                session["end_reason"] = NSNull()
                session["has_geometry"] = true
                session["abandoned"] = true
                session["state"] = "interrupted"
                session["keyframes_accepted"] = 0
                session["keyframes_journaled"] = 467
                session["finalization"] = NSNull()
            } else {
                let states = ["complete", "unbuilt", "finalizing", "receiving"]
                let state = states[index % states.count]
                if state == "receiving" {
                    session["ended_at"] = NSNull()
                    session["end_reason"] = NSNull()
                } else {
                    session["ended_at"] = updated - 10
                    session["end_reason"] = "stop"
                }
                session["has_geometry"] = state == "complete" || state == "finalizing"
                session["abandoned"] = false
                session["state"] = state
                session["keyframes_accepted"] = 120
                session["keyframes_journaled"] = 120
                if state == "complete" {
                    session["finalization"] = [
                        "state": "complete", "final_solve": "solved",
                        "started_at": updated - 9, "updated_at": updated, "detail": NSNull(),
                    ] as [String: Any]
                } else {
                    session["finalization"] = NSNull()
                }
                if state == "receiving" || state == "finalizing" { world["live"] = true }
            }
            world["sessions"] = [session]
            worlds.append(world)
        }
        XCTAssertEqual(empty, 96, "the fixture did not land on 96 empty worlds")
        XCTAssertEqual(interrupted, 29)
        XCTAssertEqual(named, 6)
        // Through the bytes, so integers and nulls arrive as they would from
        // the Tower.
        let data = try JSONSerialization.data(withJSONObject: [
            "contract": "world_builder.worlds/2026-09-10", "worlds": worlds,
        ])
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try XCTUnwrap(WorldListingDecoder.listing(from: json))
    }

    func testASyntheticRootOf161WorldsCollapsesTheEmptyOnesAndBadgesTheRest() throws {
        let listing = try syntheticRoot()
        XCTAssertEqual(listing.worlds.count, 161)

        let grouped = WorldListingPresentation.grouped(listing.worlds)
        XCTAssertEqual(grouped.primary.count, 65)
        XCTAssertEqual(grouped.empty.count, 96)
        XCTAssertEqual(grouped.emptyHeading, "96 worlds with no sessions")
        XCTAssertTrue(grouped.primary.allSatisfy(\.hasSessions))
        XCTAssertTrue(grouped.empty.allSatisfy { !$0.hasSessions })

        // The Tower's order is kept within each group.
        let primaryIDs = grouped.primary.map(\.worldID)
        XCTAssertEqual(primaryIDs, listing.worlds.filter(\.hasSessions).map(\.worldID))
        XCTAssertEqual(grouped.empty.map(\.worldID), listing.worlds.filter { !$0.hasSessions }.map(\.worldID))

        // Every row has a badge, and the vocabulary is the panel's.
        let badges = grouped.primary.flatMap(\.sessions).map(WorldListingPresentation.stateBadge(for:))
        XCTAssertEqual(badges.filter { $0 == "Interrupted" }.count, 29)
        XCTAssertFalse(badges.contains(nil), "a session row had no state badge")
        XCTAssertFalse(badges.contains("still open"), "a dead builder's record read as still open")
        let known: Set<String> = ["Building", "Finishing", "Complete", "Interrupted", "No geometry"]
        XCTAssertTrue(badges.compactMap { $0 }.allSatisfy(known.contains), "\(badges)")

        // Interrupted rows count the journal, not the record's zero.
        let interruptedRows = grouped.primary.flatMap(\.sessions).filter { $0.state == .interrupted }
        XCTAssertTrue(interruptedRows.allSatisfy {
            WorldListingPresentation.keyframeCaption(for: $0) == "467 keyframes"
        })
        XCTAssertTrue(interruptedRows.allSatisfy { !$0.isStillOpen })
    }

    func testNamedWorldsKeepTheirNamesAndUnnamedOnesGetDatedTitles() throws {
        let listing = try syntheticRoot()
        let titles = listing.worlds.map(\.title)
        let named = titles.filter { $0.hasPrefix("Kitchen walk ") }
        XCTAssertEqual(named.count, 6)
        let dated = titles.filter { $0.hasPrefix("Walk · ") }
        XCTAssertEqual(dated.count, 155)
        XCTAssertFalse(titles.contains { $0.hasPrefix("w0") || $0.hasPrefix("w1") }, "a bare id was used as a title")
        // Two unnamed worlds ten minutes apart are told apart by their titles.
        XCTAssertNotEqual(listing.worlds[6].title, listing.worlds[7].title)
    }

    /// A dated title is stable, names the day, and does not depend on the
    /// phone's locale — the same instant formats the same way twice.
    func testTheDatedTitleIsStableAndNamesTheDay() {
        let updatedAt = 1788895032.0
        let title = WorldListingPresentation.datedTitle(updatedAt: updatedAt)
        XCTAssertEqual(title, WorldListingPresentation.datedTitle(updatedAt: updatedAt))
        XCTAssertTrue(title.hasPrefix("Walk · "), title)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "d MMM yyyy, HH:mm"
        XCTAssertEqual(
            title,
            "Walk · " + formatter.string(from: Date(timeIntervalSince1970: updatedAt))
        )
    }

    func testTheEmptyHeadingIsAbsentWhenNothingIsEmpty() throws {
        let listing = try syntheticRoot()
        let onlyPrimary = WorldListingPresentation.grouped(listing.worlds.filter(\.hasSessions))
        XCTAssertNil(onlyPrimary.emptyHeading)
        let one = WorldListingPresentation.grouped(Array(listing.worlds.filter { !$0.hasSessions }.prefix(1)))
        XCTAssertEqual(one.emptyHeading, "1 world with no sessions")
    }

    /// The word for a `finalizing` row is "Finishing" and for `receiving`
    /// "Building" — the canvas's words — and an unbuilt session says what it
    /// lacks rather than that it failed.
    func testTheStateWordsAreThePanels() throws {
        let listing = try syntheticRoot()
        let sessions = listing.worlds.flatMap(\.sessions)
        func badge(_ state: WorldListingSessionState) -> String? {
            sessions.first { $0.state == state }.flatMap(WorldListingPresentation.stateBadge(for:))
        }
        XCTAssertEqual(badge(.receiving), "Building")
        XCTAssertEqual(badge(.finalizing), "Finishing")
        XCTAssertEqual(badge(.complete), "Complete")
        XCTAssertEqual(badge(.interrupted), "Interrupted")
        XCTAssertEqual(badge(.unbuilt), "No geometry")
    }
}

// MARK: - The contract this build adopted

/// The status contract moved, and this pins which one is in force.
///
/// It is a whole test class for one string because the move was **not** a
/// version bump for its own sake. Nothing in the encoding changed; one number
/// changed meaning, and a number that quietly means something else is the exact
/// failure a dated identifier exists to make loud.
final class WorldBuilderContractAdoptionTests: XCTestCase {

    /// Moved from `/2026-08-23` because `trajectory.pose_count` changed
    /// **meaning**, and from `/2026-08-25` because `model_state` gained the
    /// word `interrupted` — which this build refused as undecodable under the
    /// old identifier, and which is exactly the kind of change a dated
    /// identifier exists to make loud.
    func testTheStatusContractIsTheOneThisTowerServes() {
        XCTAssertEqual(
            WorldBuilderResultContract.identifier,
            "world_builder.status/2026-09-10"
        )
    }

    /// Same date, different agreement.
    ///
    /// The two were adopted together and it would be easy to start treating
    /// "2026-08-25" as one version of one thing. They are two contracts on two
    /// transports — status over the WebSocket, geometry over HTTP — and either
    /// can move without the other.
    func testTheGeometryContractIsSeparateFromTheStatusContract() {
        XCTAssertNotEqual(
            WorldGeometryContract.identifier,
            WorldBuilderResultContract.identifier
        )
    }

    /// The identifier this build subscribes with is the identifier it
    /// implements. A bump that reached one and not the other would leave the
    /// app refusing a Tower that was speaking its own contract.
    func testTheAdoptedContractIsTheOneThisBuildDeclaresItImplements() {
        // Membership, not set equality — see the note in
        // `WorldBuilderIntegrationTests.testThisBuildImplementsTheContractTheTowerNowOffers`.
        // What this test is for is the *link* between the identifier the client
        // subscribes with and the set the subscribe guard consults; a bump that
        // reached one and not the other stops the subscription. Four other
        // contracts joining that set does not weaken it.
        XCTAssertTrue(
            TowerCapabilities.supported.contains(WorldBuilderResultContract.identifier),
            "the subscribe guard reads `supported`; a bump that missed it stops the subscription"
        )
    }
}

// MARK: - Where the geometry lives

/// Reading the fetch address out of the status payload.
///
/// The address is the one part of the payload outside `world_snapshot` that
/// this build reads at all, so what it refuses matters as much as what it
/// accepts.
final class WorldGeometryCoordinatesTests: XCTestCase {

    /// `nil` for a block means the Tower sent `null` there, which is how the
    /// wire says "absent" — so each parameter defaults to a present block and
    /// is passed `nil` by the test that wants it gone.
    private func payload(
        worldSnapshot: [String: Any]? = ["world_id": "w1", "revision": "snapshot-rev"],
        session: [String: Any]? = ["session_id": "s1"],
        geometry: [String: Any]? = ["revision": "g1"]
    ) -> [String: Any] {
        var json: [String: Any] = [
            "model_state": "receiving",
            "world_snapshot": NSNull(),
            "session": NSNull(),
            "geometry": NSNull(),
        ]
        if let worldSnapshot { json["world_snapshot"] = worldSnapshot }
        if let session { json["session"] = session }
        if let geometry { json["geometry"] = geometry }
        return json
    }

    func testAllThreePartsOfTheAddressAreRead() {
        let coordinates = WorldBuilderResultDecoder.geometryCoordinates(from: payload())
        XCTAssertEqual(coordinates?.worldID, "w1")
        XCTAssertEqual(coordinates?.sessionID, "s1")
        XCTAssertEqual(coordinates?.revision, "g1")
    }

    /// `session_id` is a **required** query parameter on the Tower's manifest
    /// route. A world id on its own therefore addresses nothing, and half an
    /// address must not become a request.
    func testAMissingSessionRefusesTheWholeAddress() {
        XCTAssertNil(
            WorldBuilderResultDecoder.geometryCoordinates(from: payload(session: nil))
        )
    }

    /// `null` is absent, never zero. No build has produced output for this
    /// session, so there is nothing to point at — which is a different thing
    /// from geometry at "revision nothing".
    func testANullGeometryRevisionMeansNothingIsBuiltNotRevisionZero() {
        XCTAssertNil(
            WorldBuilderResultDecoder.geometryCoordinates(
                from: payload(geometry: ["available": false, "revision": NSNull()])
            )
        )
    }

    /// The load-bearing distinction. The snapshot revision changes whenever any
    /// reported field changes — a keyframe count, a tracking state — and most
    /// of those leave the built points exactly where they were. Keying the
    /// fetch on the snapshot's revision would pull a megabyte for a counter.
    func testTheRevisionComesFromTheGeometryBlockAndNotFromTheSnapshot() {
        let coordinates = WorldBuilderResultDecoder.geometryCoordinates(
            from: payload(geometry: ["revision": "geometry-rev"])
        )
        XCTAssertEqual(coordinates?.revision, "geometry-rev")
        XCTAssertNotEqual(coordinates?.revision, "snapshot-rev")
    }

    /// A payload from a Tower with no session and no build — the shape of most
    /// of a session's first seconds — addresses nothing rather than 404ing
    /// every two seconds.
    func testAnIdleTowerAddressesNothing() {
        XCTAssertNil(
            WorldBuilderResultDecoder.geometryCoordinates(
                from: payload(worldSnapshot: nil, session: nil, geometry: nil)
            )
        )
    }
}

@MainActor
final class WorldBuilderViewModelGeometryTests: XCTestCase {

    /// An incomplete address is refused at the view model too, not only at the
    /// decoder, so that a future caller reaching `geometryDidChange` from
    /// somewhere else cannot compose a URL out of what it happened to have.
    func testAnIncompleteAddressFetchesNothing() async {
        let viewModel = WorldBuilderViewModel(client: UnavailableWorldBuilderClient())
        await viewModel.geometryDidChange(worldID: "w1", sessionID: nil, revision: "g1")
        await viewModel.geometryDidChange(worldID: nil, sessionID: "s1", revision: "g1")
        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: nil)
        XCTAssertTrue(viewModel.fragmentsModel.segments.isEmpty)
        XCTAssertTrue(viewModel.geometryChunks.isEmpty)
    }

    /// A client with no Tower behind it publishes no geometry address, so a
    /// view model built on one issues no request of any kind. This is the HTTP
    /// half of what
    /// `TowerClientTests.testCartridgeViewModelsSendNothingToTheTower` asserts
    /// about the socket.
    func testAClientWithNoTowerBehindItNeverAddressesGeometry() async {
        let client = UnavailableWorldBuilderClient()
        var received: [WorldGeometryCoordinates] = []
        let cancellable = client.geometryUpdates.sink { received.append($0) }
        try? await Task.sleep(nanoseconds: 50_000_000)
        cancellable.cancel()
        XCTAssertTrue(received.isEmpty)
    }
}

// MARK: - A failed fetch must not wedge the world

/// A stubbed HTTP layer for the geometry routes.
///
/// `URLProtocol` rather than a local server because the thing under test is not
/// the network — it is what `WorldBuilderViewModel` does to its own retry
/// marker when a request fails. A protocol stub can refuse one specific path,
/// count how many times it was asked, and then start succeeding, which is
/// exactly the sequence the sticky-segment bug lives in.
///
/// `WorldGeometryClient` was written with `baseURL` and `session` as defaulted
/// properties; this is what that was for.
final class StubbedGeometryProtocol: URLProtocol {

    /// Request path → (status code, body). A path with no entry fails as a
    /// transport error, which is a third failure shape worth having.
    private static var routes: [String: (Int, String)] = [:]
    private static var paths: [String] = []
    private static let lock = NSLock()

    static func reset(routes: [String: (Int, String)]) {
        lock.lock()
        defer { lock.unlock() }
        self.routes = routes
        paths = []
    }

    static func set(route: String, to response: (Int, String)) {
        lock.lock()
        defer { lock.unlock() }
        routes[route] = response
    }

    /// How many times a path was requested. The assertion that matters is a
    /// *count*, not a boolean: "it was fetched" is true after the first failing
    /// attempt too, and would pass whether or not the retry ever happened.
    static func requestCount(for path: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return paths.filter { $0 == path }.count
    }

    /// A session wired to this stub. `.ephemeral` so nothing is cached between
    /// tests — a cached 404 would make the retry look like it never happened.
    static func makeSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubbedGeometryProtocol.self]
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let path = request.url?.path ?? ""
        StubbedGeometryProtocol.lock.lock()
        StubbedGeometryProtocol.paths.append(path)
        let route = StubbedGeometryProtocol.routes[path]
        StubbedGeometryProtocol.lock.unlock()

        guard let route, let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
            return
        }
        let response = HTTPURLResponse(
            url: url, statusCode: route.0, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(route.1.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@MainActor
final class WorldGeometryRetryTests: XCTestCase {

    private static let host = URL(string: "http://stub.invalid")!
    private static let manifestPath = "/worlds/w1/geometry/manifest"
    private static let segmentPath = "/worlds/w1/geometry/segment/0"

    /// One segment, resolved, with bounds — the shape that gets a drawn tile,
    /// so a missing chunk is a visibly blank fragment and not a non-event.
    private static let manifestBody = """
        {"contract": "world_builder.geometry/2026-08-25",
         "world_id": "w1", "session_id": "s1", "geometry_revision": "g1",
         "pose_convention": {
           "pose_type": "T_world_camera", "quaternion_order": "wxyz",
           "handedness": "right",
           "camera_axes": "opencv_x_right_y_down_z_forward",
           "translation_units": "world",
           "world_axes_origin": "first_keyframe_camera",
           "up_axis": "unknown", "pose_dtype": "float64",
           "point_dtype": "float32"},
         "segment_count": 1,
         "segments": [
           {"segment_index": 0, "content_hash": "h0", "frame_id": "segment:0",
            "registered": false, "transform_to_world": null,
            "resolution_state": "resolved", "dominant_degeneracy": null,
            "keyframe_count": 2, "solved_count": 1, "point_count": 1,
            "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}}]}
        """

    private static let segmentBody = """
        {"contract": "world_builder.geometry/2026-08-25",
         "segment_index": 0, "content_hash": "h0", "frame_id": "segment:0",
         "registered": false, "transform_to_world": null,
         "poses": [{"keyframe_id": "s1:0", "status": "anchor", "degeneracy": "",
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0]}],
         "points": [[0.5, 0.5, 0.5]],
         "points_sent": 1, "points_total": 1, "point_sampling": "none"}
        """

    private func makeViewModel() -> WorldBuilderViewModel {
        WorldBuilderViewModel(
            client: UnavailableWorldBuilderClient(),
            geometry: WorldGeometryClient(
                baseURL: Self.host, session: StubbedGeometryProtocol.makeSession()
            )
        )
    }

    /// A refused segment request must not blank a fragment permanently.
    ///
    /// The revision is the only thing that unlocks a refetch, and a
    /// **finalized** world's revision never moves again — so a segment left
    /// unfetched "until the world changes" is a segment left unfetched forever.
    /// One transient blip on a walk that has already ended would cost that
    /// fragment for good.
    func testARefusedSegmentIsRetriedUnderTheSameRevision() async {
        StubbedGeometryProtocol.reset(routes: [
            Self.manifestPath: (200, Self.manifestBody),
            Self.segmentPath: (404, ""),
        ])
        let viewModel = makeViewModel()

        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")

        // What did arrive is on screen: the manifest's segment is drawn, with
        // no chunk behind it. A blank tile is honest; refusing the whole world
        // because one request failed would show less than the Tower said.
        XCTAssertEqual(viewModel.fragmentsModel.segments.count, 1)
        XCTAssertTrue(viewModel.geometryChunks.isEmpty)
        XCTAssertEqual(StubbedGeometryProtocol.requestCount(for: Self.segmentPath), 1)

        // The next report carries the SAME revision — which is all a finalized
        // world will ever carry — and the fetch must be attempted again.
        StubbedGeometryProtocol.set(route: Self.segmentPath, to: (200, Self.segmentBody))
        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")

        XCTAssertEqual(
            StubbedGeometryProtocol.requestCount(for: Self.segmentPath),
            2,
            "a refused segment was never retried; on a finalized world that tile stays blank forever"
        )
        // Keyed on the composite key. This fixture predates `placement_hash`,
        // so the placement half is the stated absence rather than a value.
        XCTAssertEqual(
            viewModel.geometryChunks[
                WorldGeometryCacheKey.make(contentHash: "h0", placementHash: nil)
            ]?.points.count,
            1
        )
    }

    /// The negative control, and the reason the test above is not vacuous.
    ///
    /// If `geometryDidChange` simply refetched on every call, the retry
    /// assertion would pass without the marker doing anything at all. This pins
    /// the other half: a fetch that fully succeeded is **not** repeated under an
    /// unchanged revision, which is what stops the ~2 s heartbeat from pulling a
    /// megabyte twice a second.
    func testASucceededFetchIsNotRepeatedUnderTheSameRevision() async {
        StubbedGeometryProtocol.reset(routes: [
            Self.manifestPath: (200, Self.manifestBody),
            Self.segmentPath: (200, Self.segmentBody),
        ])
        let viewModel = makeViewModel()

        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")

        XCTAssertEqual(
            StubbedGeometryProtocol.requestCount(for: Self.manifestPath),
            1,
            "an unchanged revision refetched; the heartbeat would pull a megabyte every 2 s"
        )
        XCTAssertEqual(StubbedGeometryProtocol.requestCount(for: Self.segmentPath), 1)
        XCTAssertEqual(viewModel.geometryChunks.count, 1)
    }

    /// The manifest path's own escape hatch, which had no test before.
    func testARefusedManifestIsRetriedUnderTheSameRevision() async {
        StubbedGeometryProtocol.reset(routes: [Self.manifestPath: (404, "")])
        let viewModel = makeViewModel()

        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        XCTAssertTrue(viewModel.fragmentsModel.segments.isEmpty)

        StubbedGeometryProtocol.set(route: Self.manifestPath, to: (200, Self.manifestBody))
        StubbedGeometryProtocol.set(route: Self.segmentPath, to: (200, Self.segmentBody))
        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")

        XCTAssertEqual(StubbedGeometryProtocol.requestCount(for: Self.manifestPath), 2)
        XCTAssertEqual(viewModel.fragmentsModel.segments.count, 1)
        XCTAssertEqual(viewModel.geometryChunks.count, 1)
    }
}

// MARK: - The cache trap: a placement that moves under unmoved points

/// **The deliberate regression test**, and the one that is invisible if the
/// cache key is wrong.
///
/// `content_hash` covers poses and points *only*, by design, so a segment that
/// gains a placement keeps its content hash. Keyed on that alone the client
/// holds its cached chunk forever and draws an **unplaced** version of a
/// segment the world now knows how to place. Nothing throws, nothing logs,
/// no tile goes blank — the fragment simply sits in the wrong place, for good.
///
/// So the fixtures below change *only* the placement. Same segment index, same
/// `content_hash`, byte-identical points and poses. Only `placement_hash`,
/// `registered`, `registration_state` and `transform_to_world` move, and the
/// `geometry_revision` moves with them because the Tower rolls it up over both
/// hashes.
@MainActor
final class WorldGeometryPlacementCacheTests: XCTestCase {

    private static let host = URL(string: "http://stub.invalid")!
    private static let manifestPath = "/worlds/w1/geometry/manifest"
    private static let segmentPath = "/worlds/w1/geometry/segment/0"

    /// One resolved segment, unplaced. `placement_hash` is the hash the Tower
    /// computes over `{"state": "unplaced"}` — a real value, not a null.
    private static let unplacedManifest = """
        {"contract": "world_builder.geometry/2026-08-25",
         "world_id": "w1", "session_id": "s1", "geometry_revision": "g-unplaced",
         "pose_convention": {
           "pose_type": "T_world_camera", "quaternion_order": "wxyz",
           "handedness": "right",
           "camera_axes": "opencv_x_right_y_down_z_forward",
           "translation_units": "world",
           "world_axes_origin": "first_keyframe_camera",
           "up_axis": "unknown", "pose_dtype": "float64",
           "point_dtype": "float32"},
         "scale": {"state": "unknown", "meters_per_unit": null},
         "segment_count": 1,
         "segments": [
           {"segment_index": 0, "content_hash": "same-points", "frame_id": "segment:0",
            "registered": false, "registration_state": "unplaced",
            "registration_refusal_reason": null,
            "transform_to_world": null, "placement_hash": "place-none",
            "resolution_state": "resolved", "dominant_degeneracy": null,
            "keyframe_count": 2, "solved_count": 1, "point_count": 1,
            "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}}]}
        """

    /// The same world after a registration pass. **`content_hash` is
    /// unchanged** — that is the whole point — and everything about where the
    /// segment sits has moved.
    private static let placedManifest = """
        {"contract": "world_builder.geometry/2026-08-25",
         "world_id": "w1", "session_id": "s1", "geometry_revision": "g-placed",
         "pose_convention": {
           "pose_type": "T_world_camera", "quaternion_order": "wxyz",
           "handedness": "right",
           "camera_axes": "opencv_x_right_y_down_z_forward",
           "translation_units": "world",
           "world_axes_origin": "first_keyframe_camera",
           "up_axis": "unknown", "pose_dtype": "float64",
           "point_dtype": "float32"},
         "scale": {"state": "unknown", "meters_per_unit": null},
         "segment_count": 1,
         "segments": [
           {"segment_index": 0, "content_hash": "same-points", "frame_id": "segment:0",
            "registered": true, "registration_state": "registered",
            "registration_refusal_reason": null,
            "transform_to_world": {"rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                                   "translation": [10.0, 0.0, -4.0],
                                   "scale": 0.3533, "reference_segment": 4,
                                   "frame_revision": 1},
            "placement_hash": "place-moved",
            "resolution_state": "resolved", "dominant_degeneracy": null,
            "keyframe_count": 2, "solved_count": 1, "point_count": 1,
            "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}}]}
        """

    private static func segmentBody(placed: Bool) -> String {
        let placement = placed
            ? """
              "registered": true, "registration_state": "registered",
              "registration_refusal_reason": null,
              "transform_to_world": {"rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                                     "translation": [10.0, 0.0, -4.0],
                                     "scale": 0.3533, "reference_segment": 4,
                                     "frame_revision": 1},
              "placement_hash": "place-moved",
              """
            : """
              "registered": false, "registration_state": "unplaced",
              "registration_refusal_reason": null,
              "transform_to_world": null, "placement_hash": "place-none",
              """
        // The points and poses are IDENTICAL in both branches. If they were
        // not, `content_hash` would have moved and the refetch would happen
        // for a reason that has nothing to do with the placement.
        return """
            {"contract": "world_builder.geometry/2026-08-25",
             "segment_index": 0, "content_hash": "same-points",
             "frame_id": "segment:0",
             \(placement)
             "poses": [{"keyframe_id": "s1:0", "status": "anchor",
                        "degeneracy": "",
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                        "translation": [0.0, 0.0, 0.0]}],
             "points": [[0.5, 0.5, 0.5]],
             "points_sent": 1, "points_total": 1, "point_sampling": "none"}
            """
    }

    private func makeViewModel() -> WorldBuilderViewModel {
        WorldBuilderViewModel(
            client: UnavailableWorldBuilderClient(),
            geometry: WorldGeometryClient(
                baseURL: Self.host, session: StubbedGeometryProtocol.makeSession()
            )
        )
    }

    func testASegmentThatGainsAPlacementIsRefetchedAndRePlaced() async {
        StubbedGeometryProtocol.reset(routes: [
            Self.manifestPath: (200, Self.unplacedManifest),
            Self.segmentPath: (200, Self.segmentBody(placed: false)),
        ])
        let viewModel = makeViewModel()

        await viewModel.geometryDidChange(
            worldID: "w1", sessionID: "s1", revision: "g-unplaced"
        )
        XCTAssertEqual(StubbedGeometryProtocol.requestCount(for: Self.segmentPath), 1)
        XCTAssertEqual(viewModel.geometryChunks.count, 1)

        // The registration pass lands. Same points, new placement.
        StubbedGeometryProtocol.set(route: Self.manifestPath, to: (200, Self.placedManifest))
        StubbedGeometryProtocol.set(
            route: Self.segmentPath, to: (200, Self.segmentBody(placed: true))
        )
        await viewModel.geometryDidChange(
            worldID: "w1", sessionID: "s1", revision: "g-placed"
        )

        XCTAssertEqual(
            StubbedGeometryProtocol.requestCount(for: Self.segmentPath),
            2,
            "the cached chunk was reused across a placement change; the client "
                + "is drawing an unplaced version of a segment the world can place"
        )

        // And it was RE-PLACED, not merely refetched. The chunk on screen
        // carries the Sim3, under a key that names the new placement.
        XCTAssertEqual(viewModel.geometryChunks.count, 1)
        let key = WorldGeometryCacheKey.make(
            contentHash: "same-points", placementHash: "place-moved"
        )
        let drawn = viewModel.geometryChunks[key]
        XCTAssertNotNil(drawn, "the chunk is still filed under its old placement")
        XCTAssertEqual(drawn?.transformToWorld?.referenceSegment, 4)
        XCTAssertEqual(drawn?.transformToWorld?.scale, 0.3533)
        XCTAssertEqual(drawn?.transformToWorld?.translation, [10.0, 0.0, -4.0])
        XCTAssertEqual(viewModel.fragmentsModel.segments.first?.placement.state,
                       .registered)

        // The superseded chunk is gone rather than lingering under its old
        // key: one registration pass over a real 51-segment world would
        // otherwise strand 51 of them.
        XCTAssertNil(
            viewModel.geometryChunks[
                WorldGeometryCacheKey.make(
                    contentHash: "same-points", placementHash: "place-none"
                )
            ]
        )
    }

    /// The negative control, and the reason the test above is not vacuous.
    ///
    /// If the client simply refetched every segment on every manifest, the
    /// assertion above would pass with no cache key doing any work at all.
    /// This pins the other half: an unchanged placement over unchanged points
    /// is served from cache, which is what stops a two-minute walk's 67
    /// manifests from pulling a megabyte 67 times.
    func testAnUnchangedPlacementIsStillServedFromCache() async {
        StubbedGeometryProtocol.reset(routes: [
            Self.manifestPath: (200, Self.unplacedManifest),
            Self.segmentPath: (200, Self.segmentBody(placed: false)),
        ])
        let viewModel = makeViewModel()

        await viewModel.geometryDidChange(
            worldID: "w1", sessionID: "s1", revision: "g-unplaced"
        )
        // A second manifest under a moved revision, identical in every field.
        // The revision is what unlocks the manifest fetch; the cache key is
        // what decides whether the SEGMENT is fetched again.
        await viewModel.geometryDidChange(
            worldID: "w1", sessionID: "s1", revision: "g-unplaced-again"
        )

        XCTAssertEqual(StubbedGeometryProtocol.requestCount(for: Self.manifestPath), 2)
        XCTAssertEqual(
            StubbedGeometryProtocol.requestCount(for: Self.segmentPath),
            1,
            "an unchanged segment was refetched; the cache key is doing nothing"
        )
        XCTAssertEqual(viewModel.geometryChunks.count, 1)
    }
}

// MARK: - Pinned against a real Tower

/// The geometry contract, decoded from **bytes taken verbatim off a running
/// Tower** rather than from a reading of its source.
///
/// World `3dd986b1c2364d4b85de97152f2e39f4`, session
/// `dd5d13a2381e430db9b27c7da2cf2928` — the first build that ever crossed the
/// pose boundary, and the one the geometry transport was designed against. Its
/// shape is the whole reason the viewer draws fragments instead of a world:
/// 51 segments, **19 of them carrying points and 32 carrying none**, none
/// registered, every one in its own coordinate frame.
///
/// The composed fixtures above prove the decoder reads what this build expects.
/// These two prove the Tower sends it.
final class WorldGeometryRealTowerTests: XCTestCase {

    private func decode(_ text: String) throws -> [String: Any] {
        try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        )
    }

    func testARealManifestDecodesFieldForField() throws {
        let manifest = try XCTUnwrap(
            WorldGeometryDecoder.manifest(from: decode(Self.manifestFromTower))
        )

        XCTAssertEqual(manifest.worldID, "3dd986b1c2364d4b85de97152f2e39f4")
        XCTAssertEqual(manifest.sessionID, "dd5d13a2381e430db9b27c7da2cf2928")
        XCTAssertEqual(manifest.segments.count, 51)
        XCTAssertTrue(manifest.current)

        // The convention the Tower actually stamps matches the one this build
        // implements. A mismatch here renders plausibly and wrongly, which is
        // why `manifest(from:)` refuses rather than guesses.
        XCTAssertTrue(manifest.poseConvention.matchesThisBuild)

        // The 19 islands. This is the number the fragment viewer exists for:
        // a world of 51 segments that yielded geometry in 19 of them, sharing
        // no frame and no unit.
        let withGeometry = manifest.segments.filter { $0.pointCount > 0 }
        XCTAssertEqual(withGeometry.count, 19)
        XCTAssertEqual(manifest.segments.filter { $0.bounds != nil }.count, 19)

        // Not one is registered, so not one may be composited with another.
        XCTAssertTrue(manifest.segments.allSatisfy { !$0.registered })

        // A segment with no points keeps `nil` bounds rather than a zero box —
        // absent and empty stay different claims all the way from the wire.
        let empty = try XCTUnwrap(manifest.segments.first { $0.pointCount == 0 })
        XCTAssertNil(empty.bounds)

        // Every `transform_to_world` on this world is null, and null decodes
        // to nothing — never to identity, which would stack all 51 segments on
        // one origin and draw a room nobody walked.
        XCTAssertTrue(manifest.segments.allSatisfy { $0.transformToWorld == nil })

        // Scale: `unknown`, with no metres-per-unit. A world with more than
        // one segment stays `unknown` — segments do not share a coordinate
        // frame — and calibration will not change it.
        XCTAssertEqual(manifest.scale.state, .unknown)
        XCTAssertNil(manifest.scale.metersPerUnit)
        XCTAssertFalse(manifest.scale.convertibleToMetres)
    }

    /// **This capture predates `placement_hash`**, and it is what the
    /// compatibility decision was made for: the Tower added the field without
    /// bumping `world_builder.geometry/2026-08-25`, so a decoder that required
    /// it would refuse all 51 of these segments from a Tower speaking this
    /// exact contract.
    func testTheRealFixturesPredatePlacementHashAndStillDecode() throws {
        XCTAssertFalse(Self.manifestFromTower.contains("placement_hash"))
        XCTAssertFalse(Self.segmentFromTower.contains("placement_hash"))

        let manifest = try XCTUnwrap(
            WorldGeometryDecoder.manifest(from: decode(Self.manifestFromTower))
        )
        XCTAssertEqual(manifest.segments.count, 51)
        XCTAssertTrue(manifest.segments.allSatisfy { $0.placement.placementHash == nil })
        // Absent is NAMED in the key rather than dropped, so these rows still
        // key distinctly from one another and from any later placed version of
        // themselves.
        XCTAssertEqual(
            Set(manifest.segments.map(\.cacheKey)).count, 51,
            "two segments collided on one cache key; one of them would never be drawn"
        )
        XCTAssertEqual(manifest.segments[1].cacheKey, "5dec8e3d298549d3:-")

        // And `registration_state` stays absent rather than being invented
        // from `registered: false` — this Tower has not said which of "refused"
        // and "nobody looked" applies.
        XCTAssertTrue(manifest.segments.allSatisfy { $0.placement.state == nil })
    }

    /// **No imagery, ever.** `image_relpath` and every keyframe byte stay
    /// Tower-side, and the assertion is made against the bytes rather than
    /// against the Swift type, because the type could only ever prove that
    /// this decoder ignores a field — not that the field never arrives.
    func testNeitherRealPayloadCarriesAnyImagery() {
        for payload in [Self.manifestFromTower, Self.segmentFromTower] {
            XCTAssertFalse(payload.contains("image"))
            XCTAssertFalse(payload.contains("relpath"))
            XCTAssertFalse(payload.contains("keyframe_path"))
            XCTAssertFalse(payload.contains("thumbnail"))
        }
    }

    func testARealSegmentChunkDecodesFieldForField() throws {
        let chunk = try XCTUnwrap(
            WorldGeometryDecoder.chunk(from: decode(Self.segmentFromTower))
        )

        XCTAssertEqual(chunk.segmentIndex, 1)
        XCTAssertEqual(chunk.contentHash, "5dec8e3d298549d3")
        XCTAssertFalse(chunk.registered)
        XCTAssertEqual(chunk.points.count, 79)
        XCTAssertEqual(chunk.pointsSent, 79)
        XCTAssertEqual(chunk.pointsTotal, 79)
        XCTAssertEqual(chunk.pointSampling, "none")

        // Every point is a triple. A viewer that indexed [2] on a shorter row
        // would trap, and the wire is where that would come from.
        XCTAssertTrue(chunk.points.allSatisfy { $0.count == 3 })

        // The Tower's anchors are definitional, not measured, and they arrive
        // saying so: identity rotation at the origin under status `anchor`.
        let anchor = try XCTUnwrap(chunk.poses.first { $0.status == "anchor" })
        XCTAssertEqual(anchor.rotation, [1, 0, 0, 0])
        XCTAssertEqual(anchor.translation, [0, 0, 0])

        // `degeneracy: ""` is the Tower's sentinel for "none", not a missing
        // field, and it must not decode as a reason a pose was refused.
        XCTAssertTrue(chunk.poses.allSatisfy { $0.degeneracy.isEmpty })
    }

    /// `GET /worlds/{id}/geometry/manifest?session_id=…`, verbatim.
    private static let manifestFromTower = """
        {"contract":"world_builder.geometry/2026-08-25","world_id":"3dd986b1c2364d4b85de97152f2e39f4","session_id":"dd5d13a2381e430db9b27c7da2cf2928","current":true,"geometry_revision":"9de9491f943043c1","pose_convention":{"pose_type":"T_world_camera","quaternion_order":"wxyz","handedness":"right","camera_axes":"opencv_x_right_y_down_z_forward","translation_units":"world","world_axes_origin":"first_keyframe_camera","up_axis":"unknown","pose_dtype":"float64","point_dtype":"float32"},"scale":{"state":"unknown","meters_per_unit":null},"segment_count":51,"segments":[{"segment_index":0,"content_hash":"3da991270a0f8a38","frame_id":"segment:0","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"pure_rotation","keyframe_count":4,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":1,"content_hash":"5dec8e3d298549d3","frame_id":"segment:1","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":null,"keyframe_count":2,"solved_count":1,"point_count":79,"bounds":{"min":[-3.217444658279419,-6.030884265899658,13.622379302978516],"max":[1.191226840019226,8.488019943237305,17.423053741455078]}},{"segment_index":2,"content_hash":"cf1397fb1471625f","frame_id":"segment:2","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":null,"keyframe_count":1,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":3,"content_hash":"8a59ebfd32d47e63","frame_id":"segment:3","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"no_correspondence","keyframe_count":8,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":4,"content_hash":"36a82a59b303e475","frame_id":"segment:4","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":null,"keyframe_count":5,"solved_count":4,"point_count":456,"bounds":{"min":[-44.12472152709961,-10.98112678527832,0.5343655347824097],"max":[82.84456634521484,162.01988220214844,306.6292724609375]}},{"segment_index":5,"content_hash":"1c11ef12b6ffe36c","frame_id":"segment:5","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":null,"keyframe_count":11,"solved_count":10,"point_count":1872,"bounds":{"min":[-1680.0914306640625,-1777.6131591796875,0.23954088985919952],"max":[3.903865098953247,1064.3702392578125,3752.35107421875]}},{"segment_index":6,"content_hash":"522165e4516c9078","frame_id":"segment:6","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":null,"keyframe_count":10,"solved_count":9,"point_count":1115,"bounds":{"min":[-1624.46337890625,-1419.1756591796875,-0.020327942445874214],"max":[270.33721923828125,5059.27294921875,18630.4140625]}},{"segment_index":7,"content_hash":"2fb5fb21e41aafcb","frame_id":"segment:7","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"pure_rotation","keyframe_count":19,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":8,"content_hash":"a02481c9eae0b668","frame_id":"segment:8","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"no_correspondence","keyframe_count":11,"solved_count":7,"point_count":904,"bounds":{"min":[-8.5051908493042,-84.74126434326172,0.4020862579345703],"max":[1374.79638671875,1621.95947265625,2912.61376953125]}},{"segment_index":9,"content_hash":"8c71432b35a6cb5b","frame_id":"segment:9","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"pure_rotation","keyframe_count":19,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":10,"content_hash":"a7a5d0b4e4e5360b","frame_id":"segment:10","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":12,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":11,"content_hash":"09bd33715e4bc794","frame_id":"segment:11","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":null,"keyframe_count":1,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":12,"content_hash":"31f814b8f013faf7","frame_id":"segment:12","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"low_parallax","keyframe_count":17,"solved_count":8,"point_count":933,"bounds":{"min":[-2430.474365234375,-1245.3900146484375,1.2197234630584717],"max":[2.369088649749756,68.86315155029297,10964.783203125]}},{"segment_index":13,"content_hash":"9db3fa86d95a198d","frame_id":"segment:13","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"no_correspondence","keyframe_count":25,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":14,"content_hash":"087d613f02c51e01","frame_id":"segment:14","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"pure_rotation","keyframe_count":12,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":15,"content_hash":"541b65287f354325","frame_id":"segment:15","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":9,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":16,"content_hash":"462ddfdd294b52bc","frame_id":"segment:16","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":4,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":17,"content_hash":"987e3b28c579da41","frame_id":"segment:17","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"pure_rotation","keyframe_count":10,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":18,"content_hash":"ef132c61ace6bc0b","frame_id":"segment:18","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":14,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":19,"content_hash":"fba6af9fd4278634","frame_id":"segment:19","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"low_parallax","keyframe_count":32,"solved_count":22,"point_count":3033,"bounds":{"min":[-15257.7578125,-95652.84375,-0.5573309063911438],"max":[3684.925048828125,8.297386169433594,112474.1171875]}},{"segment_index":20,"content_hash":"a8c0158e0606307d","frame_id":"segment:20","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":31,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":21,"content_hash":"551c02f355cb311f","frame_id":"segment:21","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":null,"keyframe_count":2,"solved_count":1,"point_count":62,"bounds":{"min":[-0.006597032770514488,-9.226815223693848,10.386744499206543],"max":[10.921476364135742,5.419946193695068,34.93913269042969]}},{"segment_index":22,"content_hash":"bd775a8a6f866277","frame_id":"segment:22","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":null,"keyframe_count":1,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":23,"content_hash":"43f0763dfadedc56","frame_id":"segment:23","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"no_correspondence","keyframe_count":11,"solved_count":1,"point_count":20,"bounds":{"min":[-2.6136257648468018,2.751641273498535,20.178184509277344],"max":[7.826806545257568,28.70836067199707,49.54364776611328]}},{"segment_index":24,"content_hash":"29134ea547904239","frame_id":"segment:24","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"no_correspondence","keyframe_count":10,"solved_count":1,"point_count":112,"bounds":{"min":[-12.640734672546387,-2.6889231204986572,2.79256534576416],"max":[14.3943452835083,14.106026649475098,49.360164642333984]}},{"segment_index":25,"content_hash":"126b1418912c9709","frame_id":"segment:25","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"pure_rotation","keyframe_count":3,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":26,"content_hash":"8df9ed38b0232ea7","frame_id":"segment:26","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":8,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":27,"content_hash":"ba5f64b339915ec6","frame_id":"segment:27","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":6,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":28,"content_hash":"e52d8616a711fed1","frame_id":"segment:28","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":4,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":29,"content_hash":"6de22f11337e9629","frame_id":"segment:29","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":null,"keyframe_count":1,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":30,"content_hash":"659c506474c7db62","frame_id":"segment:30","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"no_correspondence","keyframe_count":10,"solved_count":7,"point_count":838,"bounds":{"min":[-1.030050277709961,-6.796903133392334,1.1115909814834595],"max":[132.16558837890625,1344.8076171875,1526.7178955078125]}},{"segment_index":31,"content_hash":"f6cff1f2c49e8440","frame_id":"segment:31","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"no_correspondence","keyframe_count":14,"solved_count":1,"point_count":52,"bounds":{"min":[-14.211556434631348,-23.96739959716797,14.956808090209961],"max":[11.325815200805664,18.558734893798828,49.69028091430664]}},{"segment_index":32,"content_hash":"b369ee9fc7a60142","frame_id":"segment:32","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"low_parallax","keyframe_count":36,"solved_count":13,"point_count":1411,"bounds":{"min":[-1413.814453125,-1949.8609619140625,0.4194486737251282],"max":[27.71381187438965,3156.80029296875,5902.07421875]}},{"segment_index":33,"content_hash":"3592abef51d7cbc0","frame_id":"segment:33","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":null,"keyframe_count":1,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":34,"content_hash":"fa21ca460f0a3017","frame_id":"segment:34","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":null,"keyframe_count":1,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":35,"content_hash":"2873579642782bc4","frame_id":"segment:35","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":null,"keyframe_count":1,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":36,"content_hash":"d48e5b4d1bf4683c","frame_id":"segment:36","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":null,"keyframe_count":1,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":37,"content_hash":"ee50cf1623ede61c","frame_id":"segment:37","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":null,"keyframe_count":2,"solved_count":1,"point_count":28,"bounds":{"min":[-1.1772572994232178,-1.4971747398376465,1.0569877624511719],"max":[-0.1985848993062973,1.5174877643585205,3.862852096557617]}},{"segment_index":38,"content_hash":"6d5786e25ba3ba53","frame_id":"segment:38","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":4,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":39,"content_hash":"8bb5aa288897bdf5","frame_id":"segment:39","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"no_correspondence","keyframe_count":3,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":40,"content_hash":"bad6f11462b19f6a","frame_id":"segment:40","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"no_correspondence","keyframe_count":2,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":41,"content_hash":"fb352980ff2ef9d9","frame_id":"segment:41","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"no_correspondence","keyframe_count":3,"solved_count":1,"point_count":20,"bounds":{"min":[-2.97108793258667,-3.8489911556243896,5.200233459472656],"max":[1.8662967681884766,-0.6281641721725464,13.3079252243042]}},{"segment_index":42,"content_hash":"fb014e5b568f5d51","frame_id":"segment:42","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":5,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":43,"content_hash":"ffcff7e4624f91ee","frame_id":"segment:43","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"no_correspondence","keyframe_count":5,"solved_count":1,"point_count":23,"bounds":{"min":[-0.057936739176511765,-0.3442666530609131,0.29405468702316284],"max":[0.027114015072584152,0.046292420476675034,0.7407951951026917]}},{"segment_index":44,"content_hash":"eef9df3661e53c93","frame_id":"segment:44","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":null,"keyframe_count":1,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":45,"content_hash":"bf552e4335a6f3f1","frame_id":"segment:45","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":8,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":46,"content_hash":"9f8181f4d4448290","frame_id":"segment:46","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":null,"keyframe_count":2,"solved_count":1,"point_count":381,"bounds":{"min":[-8.699913024902344,-15.35916805267334,20.32090950012207],"max":[10.770417213439941,5.1212639808654785,39.5633430480957]}},{"segment_index":47,"content_hash":"a7372b78d8fbddbb","frame_id":"segment:47","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"no_correspondence","keyframe_count":6,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":48,"content_hash":"f5ee8ab76df9ff12","frame_id":"segment:48","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"no_correspondence","keyframe_count":12,"solved_count":1,"point_count":129,"bounds":{"min":[-12.362543106079102,9.900866508483887,30.314125061035156],"max":[-1.979128360748291,24.497386932373047,49.93107986450195]}},{"segment_index":49,"content_hash":"4c1104155f026f7c","frame_id":"segment:49","registered":false,"transform_to_world":null,"resolution_state":"unresolved","dominant_degeneracy":"low_parallax","keyframe_count":19,"solved_count":0,"point_count":0,"bounds":null},{"segment_index":50,"content_hash":"75e07edf8b2302ec","frame_id":"segment:50","registered":false,"transform_to_world":null,"resolution_state":"resolved","dominant_degeneracy":"low_parallax","keyframe_count":18,"solved_count":4,"point_count":555,"bounds":{"min":[-64.2022933959961,-12.542752265930176,-0.5754527449607849],"max":[8.299182891845703,87.44731140136719,693.1806030273438]}}]}
        """

    /// `GET /worlds/{id}/geometry/segment/1?session_id=…&max_points=200`,
    /// verbatim. Sampling reports `none` because the segment holds 79 points
    /// and 200 were allowed — a budget that did not bind still says so.
    private static let segmentFromTower = """
        {"contract":"world_builder.geometry/2026-08-25","current":true,"segment_index":1,"content_hash":"5dec8e3d298549d3","frame_id":"segment:1","registered":false,"transform_to_world":null,"poses":[{"keyframe_id":"dd5d13a2381e430db9b27c7da2cf2928:00000227","status":"anchor","degeneracy":"","rotation":[1.0,0.0,0.0,0.0],"translation":[0.0,0.0,0.0]},{"keyframe_id":"dd5d13a2381e430db9b27c7da2cf2928:00000231","status":"solved","degeneracy":"","rotation":[0.9998091786737427,0.0003777739565228689,0.018821894384747233,0.005215344508603099],"translation":[0.6468719904899493,-0.48572241488399487,0.5879033624660022]}],"points":[[-2.8251864910125732,5.851417541503906,15.733039855957031],[-2.186521053314209,5.635050296783447,16.11659812927246],[-1.8723406791687012,4.938211441040039,14.373701095581055],[-1.1238046884536743,4.631518840789795,14.525111198425293],[-2.8285560607910156,4.136114597320557,13.772270202636719],[-2.0341405868530273,4.704507827758789,15.736750602722168],[-0.30785658955574036,5.988716125488281,14.859391212463379],[-0.8918599486351013,7.088468074798584,15.09296989440918],[-1.1977990865707397,7.724607467651367,15.978129386901855],[-2.2038235664367676,2.6321513652801514,14.866150856018066],[-2.417578935623169,3.8686137199401855,14.086015701293945],[-1.0965282917022705,7.609363079071045,15.096478462219238],[-0.6792806386947632,7.607414245605469,15.949424743652344],[-1.612669587135315,5.767972946166992,14.382736206054688],[-0.19934609532356262,5.844282150268555,14.771583557128906],[-1.575760841369629,6.185068607330322,15.624072074890137],[-1.9238148927688599,4.948044776916504,14.391816139221191],[-1.650587558746338,4.203355312347412,14.41063117980957],[-2.4525389671325684,3.9035892486572266,14.213698387145996],[-1.8688095808029175,6.067925453186035,14.986702919006348],[1.077223300933838,7.307618618011475,15.748136520385742],[-0.681450605392456,7.692735195159912,16.147676467895508],[-0.11376218497753143,7.142821311950684,15.280884742736816],[-0.5584840178489685,7.026514530181885,14.85836410522461],[-3.217444658279419,7.9374284744262695,14.789570808410645],[-1.895555019378662,4.887986660003662,14.239521026611328],[-1.5252230167388916,4.860350608825684,14.724884033203125],[-1.8334002494812012,5.045142650604248,15.393455505371094],[-1.5201964378356934,4.753427028656006,14.689470291137695],[-1.8639649152755737,4.360745906829834,14.802159309387207],[-1.8759959936141968,2.1857194900512695,14.06497859954834],[-2.7521438598632812,5.463398456573486,14.620830535888672],[-1.776002287864685,5.552976608276367,14.027909278869629],[-0.22061984241008759,5.946706295013428,15.015033721923828],[-1.4285938739776611,5.690371990203857,14.253219604492188],[-2.8464677333831787,6.021506309509277,13.909974098205566],[-0.08249344676733017,7.755697727203369,15.897381782531738],[-0.11279231309890747,7.099442481994629,15.242985725402832],[-0.5737655162811279,7.15782356262207,15.177337646484375],[-1.5470340251922607,-5.597318649291992,13.622379302978516],[0.07912556827068329,7.038555145263672,15.571170806884766],[-0.23409663140773773,5.770210266113281,14.620176315307617],[-2.5984976291656494,5.637725353240967,14.002154350280762],[-2.938666820526123,5.790436267852783,15.463438987731934],[-2.5228145122528076,5.095469951629639,13.786197662353516],[-1.8832415342330933,4.371485233306885,14.806140899658203],[0.8296099305152893,7.777370929718018,17.423053741455078],[0.8274226188659668,6.7445573806762695,15.283087730407715],[-1.9186128377914429,5.287703037261963,15.600086212158203],[-2.3309764862060547,-6.030884265899658,14.042960166931152],[-1.584289789199829,-5.597256660461426,13.699226379394531],[-1.9601857662200928,7.7627458572387695,15.391879081726074],[-1.4358505010604858,7.430853843688965,14.97073745727539],[-2.1320760250091553,4.935245513916016,16.512163162231445],[-2.307460069656372,5.512132167816162,15.523356437683105],[-1.9194636344909668,5.908429145812988,14.840858459472656],[-0.23655809462070465,6.21229362487793,15.582989692687988],[-0.932158350944519,6.59898567199707,16.164398193359375],[-1.2383867502212524,8.200161933898926,16.154293060302734],[-1.7708415985107422,8.232685089111328,15.95460033416748],[1.0707435607910156,7.812127113342285,16.662673950195312],[-2.3162841796875,2.7824103832244873,14.676214218139648],[-1.4269706010818481,7.442532539367676,14.938467979431152],[-1.7120250463485718,5.125080108642578,15.626901626586914],[-0.11175594478845596,7.88596248626709,16.01896858215332],[-0.5071048140525818,8.488019943237305,16.455476760864258],[-1.1431732177734375,7.7255635261535645,15.955605506896973],[-0.0829896628856659,7.001388072967529,15.05306339263916],[-0.6029805541038513,7.452394008636475,14.997210502624512],[-0.2849757671356201,6.4143147468566895,16.07878875732422],[-1.7868516445159912,5.963043689727783,14.772315979003906],[-1.9040725231170654,4.456711769104004,15.061767578125],[1.191226840019226,7.255112171173096,15.383050918579102],[0.7692299485206604,6.889119625091553,15.530254364013672],[-0.851668119430542,5.994761943817139,14.815544128417969],[-1.620657205581665,-5.754885673522949,14.050065994262695],[-1.7382028102874756,6.8811492919921875,14.401448249816895],[-0.2518391013145447,5.793506145477295,14.708212852478027],[-0.763960599899292,5.903599739074707,14.752820014953613]],"points_sent":79,"points_total":79,"point_sampling":"none"}
        """
}
