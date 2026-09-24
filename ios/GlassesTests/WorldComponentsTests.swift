//
//  WorldComponentsTests.swift
//  GlassesTests
//
//  The phone side of `WORLD-BUILDER-COMPONENTS.md` (contract v2,
//  `world-builder/coherence-v1` @ bb80b2c): `components[]` on the listing row
//  and the render revision, `photographic.scope`, the areas row, the area
//  viewer's own whitelist and routes, and the four area 404 identifiers.
//
//  The Tower does not implement any of this yet (P3.2). Every fixture below is
//  written from the contract text, section by section, and says which.
//

import XCTest

@testable import Glasses

/// Payloads written from the contract.
private enum V2 {
    static let roomID = "0f1e2d3c4b5a6978"
    static let bathroomID = "a1b2c3d4e5f60718"
    static let bedID = "1234567890abcdef"
    static let crumbID = "fedcba9876543210"

    /// §2.1 / §8's target walk: the room, two areas (the bed corner, 66
    /// photos in two spans; the bathroom, 55 photos, still being built), and
    /// one short stretch counted in the footer.
    static var components: [[String: Any]] {
        [
            ["id": roomID, "state": "placed", "reason": NSNull(), "reasons": [], "shown_as": "room",
             "keyframes": 128, "keyframes_phone": 96,
             "capture_spans_s": [[0.0, 29.4], [33.8, 86.1], [131.0, 170.2]],
             "has_geometry": true,
             "photographic": ["state": "complete", "stage": "appearance", "detail": "the appearance stage finished"]],
            ["id": bedID, "state": "unplaced", "reason": "solved-separately", "reasons": ["solved-separately"],
             "shown_as": "area", "keyframes": 66, "keyframes_phone": 48,
             "capture_spans_s": [[29.4, 33.2], [109.0, 131.6]],
             "has_geometry": true,
             "photographic": ["state": "complete", "stage": "appearance", "detail": "the appearance stage finished"]],
            ["id": bathroomID, "state": "unplaced", "reason": "single-unconfirmed-link",
             "reasons": ["single-unconfirmed-link", "scale-mismatch"],
             "shown_as": "area", "keyframes": 55, "keyframes_phone": NSNull(),
             "capture_spans_s": [[86.7, 97.1], [106.4, 109.6]],
             "has_geometry": false,
             "photographic": ["state": "running", "stage": "surface", "detail": "the surface stage is running under a live process"]],
            ["id": crumbID, "state": "unplaced", "reason": "no-verified-link", "reasons": ["no-verified-link"],
             "shown_as": "none", "keyframes": 21, "keyframes_phone": NSNull(),
             "capture_spans_s": [[97.2, 99.9]],
             "has_geometry": false, "photographic": NSNull()],
        ]
    }

    static func row(components: Any?, state: String = "complete",
                    photographic: [String: Any]? = ["state": "complete", "stage": "appearance", "detail": "d"])
        -> [String: Any]
    {
        var row: [String: Any] = [
            "session_id": "s1", "started_at": 1790100000.0, "ended_at": 1790100180.0,
            "end_reason": "stop", "frame_source": "live-capture", "has_geometry": true, "state": state,
            "finalization": ["state": "complete", "final_solve": "solved"],
        ]
        if let photographic { row["photographic"] = photographic }
        if let components { row["components"] = components }
        return row
    }
}

// MARK: - Decoding (§2, §3.1, §3.2)

@MainActor
final class WorldComponentsDecodingTests: XCTestCase {

    func testTheTargetWalkDecodesAsItsRoomTwoAreasAndAShortStretch() throws {
        let components = try XCTUnwrap(WorldComponents(json: V2.components))
        XCTAssertEqual(components.room.id, V2.roomID)
        XCTAssertEqual(components.room.state, .placed)
        XCTAssertEqual(components.areas.map(\.id), [V2.bedID, V2.bathroomID], "list order is area order")
        XCTAssertEqual(components.shortStretches.map(\.id), [V2.crumbID])
        XCTAssertEqual(components.shortStretchKeyframes, 21)
        XCTAssertEqual(components.areasStillFinishing, 1)
        XCTAssertFalse(components.isWhole)

        let bathroom = components.areas[1]
        XCTAssertEqual(bathroom.reason, "single-unconfirmed-link")
        XCTAssertEqual(bathroom.reasons, ["single-unconfirmed-link", "scale-mismatch"])
        XCTAssertEqual(bathroom.captureSpans, [WorldCaptureSpan(start: 86.7, end: 97.1),
                                               WorldCaptureSpan(start: 106.4, end: 109.6)])
        XCTAssertNil(bathroom.keyframesPhone)
        XCTAssertEqual(bathroom.photographic?.state, .running)
        XCTAssertNil(components.shortStretches[0].photographic)
    }

    /// §2.4 rule 1: `null`, no key, and an empty list are all "not computed".
    func testAbsentIsNeverZero() {
        XCTAssertNil(WorldComponents(json: nil))
        XCTAssertNil(WorldComponents(json: NSNull()))
        XCTAssertNil(WorldComponents(json: [Any]()))
        XCTAssertNil(WorldComponents(json: "components"))
    }

    /// §2.4 rule 8: a list with no understood `placed` room first is `null`.
    func testAListWithoutItsRoomFirstIsNotComputed() {
        var list = V2.components
        list.swapAt(0, 1)
        XCTAssertNil(WorldComponents(json: list), "the room is not first")

        list = V2.components
        list[0]["state"] = "anchored"
        XCTAssertNil(WorldComponents(json: list), "no placed entry this build understands")

        list = V2.components
        list[1]["state"] = "placed"
        XCTAssertNil(WorldComponents(json: list), "two rooms")
    }

    /// Rule 8, the other half: an unknown `shown_as` is treated as `none`, and
    /// an unknown reason is carried and changes nothing.
    func testUnknownWordsFromANewerTowerDegradeToCountedOnly() throws {
        var list = V2.components
        list[1]["shown_as"] = "hologram"
        list[2]["reason"] = "a-reason-from-2027"
        list[2]["reasons"] = ["a-reason-from-2027"]
        let components = try XCTUnwrap(WorldComponents(json: list))
        XCTAssertEqual(components.areas.map(\.id), [V2.bathroomID])
        XCTAssertEqual(components.shortStretches.map(\.id), [V2.bedID, V2.crumbID])
        XCTAssertEqual(components.areas[0].reason, "a-reason-from-2027")
    }

    /// An entry that disagrees with the contract does not become a walk with a
    /// piece silently missing: the whole list is not computed.
    func testAMalformedEntryMakesTheWholeListNotComputed() {
        for (key, value) in [("keyframes", "66" as Any), ("has_geometry", NSNull()),
                             ("capture_spans_s", [[3.0]]), ("capture_spans_s", [[5.0, 2.0]]),
                             ("capture_spans_s", "0:29–0:33"), ("id", 7)] {
            var list = V2.components
            list[1][key] = value
            XCTAssertNil(WorldComponents(json: list), "\(key) = \(value)")
        }
        var list = V2.components
        list[1]["id"] = "A1B2C3D4E5F60718"
        XCTAssertNil(WorldComponents(json: list), "an area id that cannot address its routes as sent")
    }

    /// A walk the gate kept whole: only the room. Nothing new appears (§8).
    func testAWholeWalkIsWhole() throws {
        let components = try XCTUnwrap(WorldComponents(json: [V2.components[0]]))
        XCTAssertTrue(components.isWhole)
        XCTAssertNil(WorldComponentsPresentation.areasHeading(components))
        XCTAssertNil(WorldComponentsPresentation.footer(components))
        XCTAssertNil(WorldComponentsPresentation.roomCaptionSuffix(components))
    }

    /// §3.1: the row carries it; an older row has no key and decodes exactly
    /// as before.
    func testTheListingRowCarriesComponentsAndAnOlderRowDoesNot() throws {
        let row = try XCTUnwrap(WorldListingSession(json: V2.row(components: V2.components)))
        XCTAssertEqual(row.components?.areas.count, 2)
        let older = try XCTUnwrap(WorldListingSession(json: V2.row(components: nil)))
        XCTAssertNil(older.components)
        let explicitNull = try XCTUnwrap(WorldListingSession(json: V2.row(components: NSNull())))
        XCTAssertNil(explicitNull.components)
    }

    /// §3.2: the revision body carries it beside, not inside, `revision`.
    func testTheRevisionBodyCarriesComponentsOutsideTheRevision() throws {
        let body: [String: Any] = [
            "session_id": "s1", "representation": "appearance", "revision": "s1/appearance:1@3",
            "live": false, "components": V2.components,
        ]
        let data = try JSONSerialization.data(withJSONObject: body)
        let revision = try WorldRenderClient.decodeRevision(data)
        XCTAssertEqual(revision.revision, "s1/appearance:1@3")
        XCTAssertEqual(revision.components?.areas.count, 2)
        let bare = try WorldRenderClient.decodeRevision(Data(#"{"revision": "s1/sparse"}"#.utf8))
        XCTAssertNil(bare.components)
    }
}

// MARK: - The words (§8)

@MainActor
final class WorldComponentsWordsTests: XCTestCase {

    func testSpansAreMinutesAndSecondsOfTheWalkWithAnEnDash() {
        let spans = [WorldCaptureSpan(start: 29.4, end: 33.2), WorldCaptureSpan(start: 109.0, end: 131.6)]
        // Truncated, as the Tower's `walk_time` and the contract's examples are.
        XCTAssertEqual(WorldComponentsPresentation.spans(spans), "0:29–0:33 and 1:49–2:11")
        // §8's own example, from the target's bed corner [[29.6, 33.1], [109.7, 132.4]].
        XCTAssertEqual(
            WorldComponentsPresentation.spans([WorldCaptureSpan(start: 29.6, end: 33.1),
                                               WorldCaptureSpan(start: 109.7, end: 132.4)]),
            "0:29–0:33 and 1:49–2:12")
        XCTAssertEqual(WorldComponentsPresentation.spans([WorldCaptureSpan(start: 86.0, end: 109.0)]), "1:26–1:49")
        XCTAssertEqual(
            WorldComponentsPresentation.spans([WorldCaptureSpan(start: 0, end: 5), WorldCaptureSpan(start: 60, end: 61),
                                               WorldCaptureSpan(start: 120, end: 125)]),
            "0:00–0:05, 1:00–1:01 and 2:00–2:05")
        XCTAssertNil(WorldComponentsPresentation.spans([]))
        // No hours, and minutes are not wrapped past 59.
        XCTAssertEqual(WorldComponentsPresentation.span(WorldCaptureSpan(start: 3725.0, end: 3726.2)), "62:05–62:06")
    }

    func testTheAreasRowFooterAndCaptionSuffix() throws {
        let components = try XCTUnwrap(WorldComponents(json: V2.components))
        XCTAssertEqual(WorldComponentsPresentation.areasHeading(components),
                       "2 more areas — captured on this walk but not placed in this room.")
        XCTAssertEqual(WorldComponentsPresentation.areaLine(number: 1, area: components.areas[0]),
                       "Area 1 · 0:29–0:33 and 1:49–2:11 · 66 photos")
        XCTAssertEqual(WorldComponentsPresentation.footer(components),
                       "1 short stretch (21 photos) could not be placed or shown.")
        XCTAssertEqual(WorldComponentsPresentation.roomCaptionSuffix(components), " · 2 more areas shown separately")
        XCTAssertNil(WorldComponentsPresentation.roomCaptionSuffix(nil), "null draws exactly today's caption")

        let one = try XCTUnwrap(WorldComponents(json: Array(V2.components.prefix(2))))
        XCTAssertEqual(WorldComponentsPresentation.areasHeading(one),
                       "1 more area — captured on this walk but not placed in this room.")
        XCTAssertEqual(WorldComponentsPresentation.roomCaptionSuffix(one), " · 1 more area shown separately")
    }

    /// §8: an entry opens only with something to draw; otherwise it says why.
    func testEachAreaSaysWhetherItOpens() throws {
        let components = try XCTUnwrap(WorldComponents(json: V2.components))
        XCTAssertEqual(WorldComponentsPresentation.availability(of: components.areas[0]), .opens)
        XCTAssertEqual(WorldComponentsPresentation.availability(of: components.areas[1]), .improving)
        XCTAssertEqual(WorldComponentsPresentation.availabilityWord(.improving), "Improving")

        var list = V2.components
        list[2]["photographic"] = ["state": "failed", "stage": "surface", "detail": "raised"]
        let failed = try XCTUnwrap(WorldComponents(json: list))
        XCTAssertEqual(WorldComponentsPresentation.availability(of: failed.areas[1]), .couldNotBeBuilt)
        XCTAssertEqual(WorldComponentsPresentation.availabilityWord(.couldNotBeBuilt), "could not be built")
        XCTAssertNil(WorldComponentsPresentation.availabilityWord(.opens))
    }

    /// §5.4: the area viewer's header and caption, and never a scale, a name
    /// or a position.
    func testTheAreaViewerSaysItIsNotPlaced() {
        XCTAssertEqual(WorldComponentsPresentation.areaHeader(number: 2, of: 2), "Area 2 of 2 — not placed in the room")
        let opening = WorldAreaOpening(number: 2, total: 2, spans: [])
        XCTAssertEqual(opening.header, "Area 2 of 2 — not placed in the room")
        XCTAssertEqual(opening.numberLine, "Area 2 of 2")
        XCTAssertEqual(WorldAreaOpening.notPlacedLine, "not placed in the room")
        let spans = [WorldCaptureSpan(start: 86.0, end: 109.0)]
        let appearance = WorldComponentsPresentation.areaCaption(representation: .appearance, spans: spans)
        // §5.4's sentence exactly: "from 1:26 to 1:49 of this walk".
        XCTAssertEqual(appearance, "The camera's own images, faces redacted, from 1:26 to 1:49 of this walk. "
            + "This area could not be placed relative to the room, so it is shown on its own: its "
            + "position, direction and size are not comparable with the room's. Not to scale.")
        let surface = WorldComponentsPresentation.areaCaption(representation: .surface, spans: spans)
        XCTAssertTrue(surface.hasPrefix("Surfaces the Tower reconstructed from the walk, from 1:26 to 1:49"), surface)
        // §5.4's own example: the target's bathroom, several spans, ONE range --
        // first start to last end, truncated, exactly as the Tower's page says it.
        let bathroom = [WorldCaptureSpan(start: 86.7, end: 97.1), WorldCaptureSpan(start: 106.4, end: 109.6)]
        XCTAssertEqual(WorldComponentsPresentation.envelopeInProse(bathroom), "1:26 to 1:49")
        XCTAssertTrue(WorldComponentsPresentation.areaCaption(representation: .appearance, spans: bathroom)
            .contains("from 1:26 to 1:49 of this walk."))
        XCTAssertNil(WorldComponentsPresentation.envelopeInProse([]))
        let tilted = WorldComponentsPresentation.areaCaption(representation: .surface, spans: spans, levelled: false)
        XCTAssertTrue(tilted.contains("Its vertical could not be estimated, so it may look tilted."), tilted)
        for caption in [appearance, surface, tilted] {
            XCTAssertTrue(caption.hasSuffix("Not to scale."), caption)
            for word in ["metre", "meter", " m ", "bathroom", "closet", "left of", "behind"] {
                XCTAssertFalse(caption.contains(word), "\(word): \(caption)")
            }
        }
    }
}

// MARK: - `photographic.scope` (§3.4, C1 E3/E13)

@MainActor
final class WorldPhotographicScopeTests: XCTestCase {

    private var snapshot: WorldSnapshot {
        WorldSnapshot(worldID: "w1", keyframeCount: 249,
                      geometry: WorldGeometryReport(representation: "sparse point cloud", elementCount: 9_000,
                                                    isIncremental: false),
                      trajectory: WorldTrajectoryReport(poseCount: 249))
    }

    private func areaWord(_ state: String) -> WorldPhotographicReport {
        WorldPhotographicReport(state: WorldPhotographicState(rawValue: state), stage: "surface",
                                detail: "an area of this walk", scope: .area)
    }

    func testScopeDecodesAndDefaultsToTheRoom() throws {
        XCTAssertEqual(WorldPhotographicReport(json: ["state": "owed", "scope": "area"])?.scope, .area)
        XCTAssertEqual(WorldPhotographicReport(json: ["state": "owed"])?.scope, .room)
        let unknown = try XCTUnwrap(WorldPhotographicReport(json: ["state": "owed", "scope": "garden"]))
        XCTAssertEqual(unknown.scope.rawValue, "garden")
        XCTAssertEqual(unknown.standing, .owed, "an unknown scope is not an area; the word keeps its meaning")
    }

    func testAnAreasUnfinishedWordLeavesTheRoomFinished() {
        for word in ["running", "owed", "unobservable"] {
            XCTAssertEqual(areaWord(word).standing, .areaStillFinishing, word)
            XCTAssertFalse(areaWord(word).standing.isUnfinished, word)
        }
        XCTAssertEqual(areaWord("failed").standing, .settled, "an area's failure never reaches the row (§3.4)")
    }

    /// E13 makes `build_in_progress` true while an area builds. The room on
    /// screen is still Saved, with no spinner and no "figures are not final".
    func testARoomWithAnAreaStillBuildingIsSavedWithoutASpinner() {
        let solved = WorldFinalizationReport(state: .complete, finalSolve: "solved")
        let evidence = WorldEvidence(snapshot: snapshot)
        for building in [true, false, nil] as [Bool?] {
            let stage = WorldStage.stage(
                for: .finalizing(snapshot, buildInProgress: building), evidence: evidence,
                finalization: solved, photographic: areaWord("running"))
            XCTAssertEqual(stage, .saved, "\(String(describing: building))")
        }
        let presentation = WorldPresentation(
            stage: .saved, evidence: evidence, finalSolve: .solved,
            reconstruction: .final(WorldRenderTarget(worldID: "w1", sessionID: "s1")),
            photographic: areaWord("running"))
        XCTAssertEqual(presentation.headline, "Saved")
        XCTAssertFalse(presentation.showsLiveBuild(buildInProgress: true))
        XCTAssertEqual(presentation.viewerNote, "Saved. An area of this walk is still being finished.")
        let sentence = WorldPresentation.finalizingDetail(buildInProgress: true, photographic: areaWord("running"))
        XCTAssertEqual(sentence, "Saved. An area of this walk is still being finished.")
        XCTAssertFalse(sentence.contains("figures are not final"))
        // The room's own live build still draws the spinner.
        XCTAssertTrue(WorldPresentation(stage: .improving).showsLiveBuild(buildInProgress: true))
    }

    /// The Saved Worlds row: complete as far as its room goes, and it says how
    /// many areas are still being finished when the listing knows.
    func testAFinishingRowWhoseRoomIsDoneSaysAnAreaIsStillBeingFinished() throws {
        let row = try XCTUnwrap(WorldListingSession(json: V2.row(
            components: V2.components, state: "finalizing",
            photographic: ["state": "running", "stage": "surface", "detail": "an area of this walk", "scope": "area"])))
        XCTAssertEqual(WorldListingPresentation.stateBadge(for: row), "Complete")
        XCTAssertEqual(WorldListingPresentation.photographicCaption(for: row),
                       "1 area of this walk is still being finished.")
        let note = WorldPickerView.note(
            forOpened: row, target: WorldRenderTarget(worldID: "w1", sessionID: "s1"),
            pinnedStage: nil, pinnedReconstruction: nil)
        XCTAssertEqual(note, "Saved. 1 area of this walk is still being finished.")
        XCTAssertFalse(note?.contains("very different") ?? true)
    }

    func testTheCountIsSaidOnlyWhenKnown() {
        XCTAssertEqual(WorldPhotographicCopy.areaStillFinishing(), "Saved. An area of this walk is still being finished.")
        XCTAssertEqual(WorldPhotographicCopy.areaStillFinishing(count: 3), "Saved. 3 areas of this walk are still being finished.")
        XCTAssertEqual(WorldPhotographicCopy.areasStillFinishingCaption(count: 1), "1 area of this walk is still being finished.")
    }
}

// MARK: - The area viewer's whitelist (§5.5, C1 M1)

@MainActor
final class WorldAreaWhitelistTests: XCTestCase {

    private let area = WorldAssetScope.area(sessionID: "s1", areaID: V2.bathroomID)
    private let digest = "0123456789abcdef0123456789abcdef"

    private func parse(_ path: String, scope: WorldAssetScope, session: String? = "s1") -> WorldAssetRequest? {
        WorldAssetRequest.parse(URL(string: "glasses-world://tower" + path), method: "GET",
                                worldID: "w1", sessionID: session, scope: scope)
    }

    func testAnAreaViewerAnswersExactlyItsFiveRoutes() {
        let base = "/worlds/w1/areas/s1/\(V2.bathroomID)"
        XCTAssertEqual(parse("\(base)/render", scope: area), .page)
        XCTAssertEqual(parse("\(base)/render/revision", scope: area), .renderRevision)
        XCTAssertEqual(parse("\(base)/appearance/manifest", scope: area), .appearanceManifest)
        XCTAssertEqual(parse("\(base)/appearance/chunk/\(digest)", scope: area), .appearanceChunk(digest: digest))
        XCTAssertEqual(parse("\(base)/appearance/proxy/\(digest)", scope: area), .appearanceProxy(digest: digest))
    }

    func testAnAreaViewerAnswersNothingElse() {
        let base = "/worlds/w1/areas/s1/\(V2.bathroomID)"
        let refused = [
            // Every room route.
            "/worlds/w1/render", "/worlds/w1/render/revision?session_id=s1",
            "/worlds/w1/appearance/s1/manifest", "/worlds/w1/appearance/s1/chunk/\(digest)",
            // Another area, another session, another world.
            "/worlds/w1/areas/s1/\(V2.bedID)/render", "/worlds/w1/areas/s2/\(V2.bathroomID)/render",
            "/worlds/w2/areas/s1/\(V2.bathroomID)/render",
            // Any query, even the room's harmless ones.
            "\(base)/render?viewer=appearance-1", "\(base)/render/revision?session_id=s1",
            "\(base)/appearance/manifest?x=1",
            // Malformed or extra.
            "\(base)/appearance/chunk/NOTHEX", "\(base)/appearance/chunk/\(digest.uppercased())",
            "\(base)/appearance/other/\(digest)", "\(base)/render/revision/extra", "\(base)",
            "\(base)/../\(V2.bedID)/render", "/worlds/w1/areas/s1/\(V2.bathroomID.uppercased())/render",
        ]
        for path in refused {
            XCTAssertNil(parse(path, scope: area), path)
        }
        XCTAssertNil(WorldAssetRequest.parse(
            URL(string: "glasses-world://tower\(base)/render"), method: "POST", worldID: "w1",
            sessionID: "s1", scope: area), "only GET")
    }

    /// The room's whitelist is exactly what it was: it answers no `/areas/`
    /// path at all.
    func testARoomViewerAnswersNoAreaRoute() {
        let base = "/worlds/w1/areas/s1/\(V2.bathroomID)"
        for suffix in ["/render", "/render/revision", "/appearance/manifest", "/appearance/chunk/\(digest)"] {
            XCTAssertNil(parse(base + suffix, scope: .room), suffix)
        }
        XCTAssertEqual(parse("/worlds/w1/render", scope: .room), .page)
        XCTAssertEqual(parse("/worlds/w1/render/revision?session_id=s1", scope: .room), .renderRevision)
    }

    /// Proxied to the same path on the Tower, with no query -- the area routes
    /// take none and ignore `viewer` (§5.1, C1 M12).
    func testAreaRequestsAreProxiedToTheAreaPathsWithNoQuery() {
        let host = URL(string: "http://tower.invalid:8000")!
        let base = "/worlds/w1/areas/s1/\(V2.bathroomID)"
        let cases: [(WorldAssetRequest, String)] = [
            (.renderRevision, "\(base)/render/revision"),
            (.appearanceManifest, "\(base)/appearance/manifest"),
            (.appearanceChunk(digest: digest), "\(base)/appearance/chunk/\(digest)"),
            (.appearanceProxy(digest: digest), "\(base)/appearance/proxy/\(digest)"),
        ]
        for (request, path) in cases {
            let url = request.towerURL(baseURL: host, worldID: "w1", sessionID: "s1", scope: area)
            XCTAssertEqual(url?.path, path)
            XCTAssertNil(url?.query, path)
        }
        XCTAssertNil(WorldAssetRequest.page.towerURL(baseURL: host, worldID: "w1", sessionID: "s1", scope: area))
        // The room's proxy still adds its `viewer` declaration, as before.
        XCTAssertEqual(WorldAssetRequest.renderRevision.towerURL(baseURL: host, worldID: "w1", sessionID: "s1")?.query,
                       "session_id=s1&viewer=appearance-1")
    }

    func testThePageAddressesForTheRoomAndForAnArea() {
        XCTAssertEqual(WorldAssetScheme.pageURL(worldID: "w1", scope: area)?.absoluteString,
                       "glasses-world://tower/worlds/w1/areas/s1/\(V2.bathroomID)/render")
        XCTAssertEqual(WorldAssetScheme.pageURL(worldID: "w1", scope: .room)?.absoluteString,
                       "glasses-world://tower/worlds/w1/render")
        XCTAssertNil(WorldAssetScheme.pageURL(worldID: "w1", scope: .area(sessionID: "s1", areaID: "not-an-id")))
    }

    func testTheScopeATargetOpens() {
        XCTAssertEqual(WorldAssetScope.of(WorldRenderTarget(worldID: "w1", sessionID: "s1")), .room)
        XCTAssertEqual(WorldAssetScope.of(WorldRenderTarget(worldID: "w1", sessionID: "s1", areaID: V2.bathroomID)),
                       area)
        let handler = WorldAssetSchemeHandler(worldID: "w1", scope: area)
        XCTAssertEqual(handler.sessionID, "s1", "an area viewer's session is fixed at open, never taught by the page")
    }

    /// The page and the native revision poll: the area paths, with no query
    /// at all -- no `session_id` (it is in the path) and no `viewer` (C1 M12).
    func testTheAreaPageAndRevisionAddresses() {
        let host = URL(string: "http://tower.invalid:8000")!
        let target = WorldRenderTarget(worldID: "w1", sessionID: "s1", areaID: V2.bathroomID)
        let page = WorldRenderClient.url(for: target, baseURL: host)
        XCTAssertEqual(page?.path, "/worlds/w1/areas/s1/\(V2.bathroomID)/render")
        XCTAssertNil(page?.query)
        let revision = WorldRenderClient.revisionURL(for: target, baseURL: host)
        XCTAssertEqual(revision?.path, "/worlds/w1/areas/s1/\(V2.bathroomID)/render/revision")
        XCTAssertNil(revision?.query)
        XCTAssertNil(WorldRenderClient.url(
            for: WorldRenderTarget(worldID: "w1", sessionID: nil, areaID: V2.bathroomID), baseURL: host),
            "an area needs its session")
        XCTAssertNotEqual(target.id, target.room.id, "the room and an area are different screens")
        XCTAssertEqual(target.room.areaID, nil)
        XCTAssertEqual(target.room.area(V2.bathroomID), target)
    }
}

// MARK: - The area viewer (§5.1, §5.2, C1 E4, M10)

@MainActor
final class WorldAreaViewerTests: XCTestCase {

    private static let host = URL(string: "http://stub.invalid")!
    private static let areaBase = "/worlds/w1/areas/s1/\(V2.bathroomID)"
    private let target = WorldRenderTarget(worldID: "w1", sessionID: "s1", areaID: V2.bathroomID)

    private func client() -> WorldRenderClient {
        WorldRenderClient(baseURL: Self.host, session: StubbedGeometryProtocol.makeSession())
    }

    private func areaPage(area: String = V2.bathroomID, session: String = "s1", declareArea: Bool = true) -> String {
        "<!doctype html><html><head><meta charset=\"utf-8\">"
            + "<meta name=\"wb-representation\" content=\"appearance\">"
            + "<meta name=\"wb-revision\" content=\"\(session)/area:\(area)/appearance:1@1\">"
            + (declareArea ? "<meta name=\"wb-area\" content=\"\(area)\">" : "")
            + "</head><body></body></html>"
    }

    private func waitUntil(timeout: TimeInterval = 3, _ condition: @MainActor () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    func testAnAreaViewerDrawsOnlyThePageOfItsOwnArea() {
        XCTAssertNil(WorldRenderViewerModel.areaPageMismatch(html: areaPage(), target: target))
        XCTAssertNotNil(WorldRenderViewerModel.areaPageMismatch(html: areaPage(area: V2.bedID), target: target))
        XCTAssertNotNil(WorldRenderViewerModel.areaPageMismatch(html: areaPage(session: "s2"), target: target))
        XCTAssertNotNil(WorldRenderViewerModel.areaPageMismatch(html: areaPage(declareArea: false), target: target))
        XCTAssertNil(WorldRenderViewerModel.areaPageMismatch(html: areaPage(area: V2.bedID), target: target.room),
                     "the room is never refused by this check")
    }

    func testAPageOfAnotherAreaIsNotDrawn() async {
        StubbedGeometryProtocol.reset(routes: ["\(Self.areaBase)/render": (200, areaPage(area: V2.bedID))])
        let model = WorldRenderViewerModel(target: target, client: client())
        await model.load()
        XCTAssertNil(model.state.html)
        XCTAssertNotNil(model.state.failureMessage)
    }

    func testTheAreaPageIsFetchedFromTheAreaRoute() async {
        StubbedGeometryProtocol.reset(routes: ["\(Self.areaBase)/render": (200, areaPage())])
        let model = WorldRenderViewerModel(target: target, client: client())
        await model.load()
        XCTAssertNotNil(model.state.html)
        XCTAssertEqual(StubbedGeometryProtocol.requestCount(for: "\(Self.areaBase)/render"), 1)
        XCTAssertEqual(model.assets.scope, .area(sessionID: "s1", areaID: V2.bathroomID))
        XCTAssertNil(model.components, "an area viewer has no areas row")
    }

    /// The four sentences are identifiers: three are terminal, one transient.
    func testTheFourAreaAnswersAreStableIdentifiers() {
        XCTAssertEqual(WorldAreaRouteAnswer.noSuchArea, "no such area in this session")
        XCTAssertEqual(WorldAreaRouteAnswer.couldNotBeBuilt, "this area could not be built")
        XCTAssertEqual(WorldAreaRouteAnswer.sessionHasNoAreas, "this session has no areas")
        XCTAssertEqual(WorldAreaRouteAnswer.notBuiltYet, "this area has not been built yet")
        for terminal in [WorldAreaRouteAnswer.noSuchArea, WorldAreaRouteAnswer.couldNotBeBuilt,
                         WorldAreaRouteAnswer.sessionHasNoAreas] {
            XCTAssertTrue(WorldAreaRouteAnswer.isTerminal(terminal))
            XCTAssertFalse(WorldRenderFetchError.absent(detail: terminal).isRetryable, terminal)
            XCTAssertEqual(WorldRenderFetchError.absent(detail: terminal).message,
                           "The Tower has no picture of this area: \(terminal).")
        }
        XCTAssertFalse(WorldAreaRouteAnswer.isTerminal(WorldAreaRouteAnswer.notBuiltYet))
        XCTAssertTrue(WorldRenderFetchError.absent(detail: WorldAreaRouteAnswer.notBuiltYet).isRetryable)
        XCTAssertFalse(WorldAreaRouteAnswer.isTerminal("No Such Area In This Session"), "equality, not similarity")
    }

    private func readyAreaModel(revisionAnswer: (Int, String)) async -> WorldRenderViewerModel {
        StubbedGeometryProtocol.reset(routes: [
            "\(Self.areaBase)/render": (200, areaPage()),
            "\(Self.areaBase)/render/revision": revisionAnswer,
        ])
        let model = WorldRenderViewerModel(target: target, client: client())
        model.revisionPollInterval = .milliseconds(20)
        await model.load()
        model.pageEvent(.rendered)
        return model
    }

    /// §5.2: `no such area in this session` is terminal for that id -- keep the
    /// picture, stop following, offer the way back.
    func testATerminalAnswerStopsTheFollowerAndKeepsThePicture() async {
        let model = await readyAreaModel(
            revisionAnswer: (404, #"{"detail": "no such area in this session"}"#))
        let follow = Task { await model.followRevisions() }
        let stopped = await waitUntil { model.areaNoLongerServed }
        XCTAssertTrue(stopped)
        XCTAssertNotNil(model.state.html, "the picture stays")
        let settled = await waitUntil(timeout: 1) { false }
        XCTAssertFalse(settled)
        XCTAssertEqual(StubbedGeometryProtocol.requestCount(for: "\(Self.areaBase)/render/revision"), 1,
                       "the follower asked once and stopped")
        follow.cancel()
        await follow.value
    }

    /// `this area has not been built yet` is transient: ask again.
    func testATransientAnswerKeepsFollowing() async {
        let model = await readyAreaModel(
            revisionAnswer: (404, #"{"detail": "this area has not been built yet"}"#))
        let follow = Task { await model.followRevisions() }
        let askedAgain = await waitUntil {
            StubbedGeometryProtocol.requestCount(for: "\(Self.areaBase)/render/revision") >= 3
        }
        XCTAssertTrue(askedAgain)
        XCTAssertFalse(model.areaNoLongerServed)
        // Gone before the next test resets the stub (the reason
        // `WorldRenderRevisionTests.stop` exists).
        follow.cancel()
        await follow.value
    }
}

// MARK: - The room's areas row is kept current (§3.2)

@MainActor
final class WorldRoomComponentsTests: XCTestCase {

    private static let host = URL(string: "http://stub.invalid")!

    private func client() -> WorldRenderClient {
        WorldRenderClient(baseURL: Self.host, session: StubbedGeometryProtocol.makeSession())
    }

    private func revision(components: Any?) throws -> String {
        var body: [String: Any] = ["session_id": "s1", "representation": "surface",
                                   "revision": "s1/surface:1", "live": false]
        if let components { body["components"] = components }
        return String(decoding: try JSONSerialization.data(withJSONObject: body), as: UTF8.self)
    }

    func testTheRoomReadsItsAreasFromTheRevisionAsItOpens() async throws {
        StubbedGeometryProtocol.reset(routes: [
            "/worlds/w1/render/revision": (200, try revision(components: V2.components)),
        ])
        let model = WorldRenderViewerModel(target: WorldRenderTarget(worldID: "w1", sessionID: "s1"), client: client())
        XCTAssertNil(model.components)
        await model.refreshComponents()
        XCTAssertEqual(model.components?.areas.count, 2)
    }

    /// An older Tower's revision has no key, and the room is exactly today's.
    func testAnOlderTowerLeavesTheRoomAsToday() async throws {
        StubbedGeometryProtocol.reset(routes: ["/worlds/w1/render/revision": (200, try revision(components: nil))])
        let model = WorldRenderViewerModel(target: WorldRenderTarget(worldID: "w1", sessionID: "s1"), client: client())
        await model.refreshComponents()
        XCTAssertNil(model.components)
    }

    /// Seeded from the listing; only a change a reader can see republishes.
    func testTheListIsReplacedOnlyWhenSomethingVisibleMoved() throws {
        let seeded = try XCTUnwrap(WorldComponents(json: V2.components))
        let model = WorldRenderViewerModel(
            target: WorldRenderTarget(worldID: "w1", sessionID: "s1"), client: client(), components: seeded)
        var publishes = 0
        let cancellable = model.$components.dropFirst().sink { _ in publishes += 1 }
        defer { cancellable.cancel() }

        var proseOnly = V2.components
        proseOnly[2]["photographic"] = ["state": "running", "stage": "surface", "detail": "other prose"]
        model.adoptComponents(WorldComponents(json: proseOnly))
        XCTAssertEqual(publishes, 0, "detail prose is not a visible change")

        var built = V2.components
        built[2]["has_geometry"] = true
        built[2]["photographic"] = ["state": "complete", "stage": "appearance", "detail": "done"]
        model.adoptComponents(WorldComponents(json: built))
        XCTAssertEqual(publishes, 1)
        XCTAssertEqual(WorldComponentsPresentation.availability(of: model.components!.areas[1]), .opens)

        let area = WorldRenderViewerModel(
            target: WorldRenderTarget(worldID: "w1", sessionID: "s1", areaID: V2.bathroomID),
            client: client(), components: seeded)
        XCTAssertNil(area.components, "an area viewer never shows an areas row")
    }
}

// MARK: - `finalization.notice` (contract v6, G1-F2)

@MainActor
final class WorldFinalizationNoticeTests: XCTestCase {

    /// §3.1's sentence for masks that are off, as a row would carry it.
    private let masksOff = "masks were not applied (they are off on this Tower: TOWER_WORLD_SOLVE_MASKS); "
        + "an operator can turn them on, then an owner can re-finish this walk"

    private func row(finalization: Any?, components: Any? = nil) throws -> WorldListingSession {
        var json = V2.row(components: components)
        json["finalization"] = finalization
        return try XCTUnwrap(WorldListingSession(json: json))
    }

    func testTheNoticeIsReadVerbatim() throws {
        let joined = masksOff + "; the evidence gate could not measure metric scale (depth did not finish); "
            + "the Tower re-runs the gate when it is idle"
        let session = try row(finalization: ["state": "complete", "final_solve": "solved", "detail": NSNull(),
                                             "notice": joined])
        XCTAssertEqual(session.finalization?.notice, joined, "verbatim: joined causes, punctuation and all")
        XCTAssertEqual(session.finalization?.finalSolve, "solved")
    }

    /// Absent is today's row, byte for byte (§7 rule 1).
    func testAbsentOrUnusableIsNoNotice() throws {
        XCTAssertNil(try row(finalization: ["state": "complete", "final_solve": "solved"]).finalization?.notice)
        for value in [NSNull(), "", "   \n", 7, ["text": "x"], true] as [Any] {
            let session = try row(finalization: ["state": "complete", "final_solve": "solved", "notice": value])
            XCTAssertNil(session.finalization?.notice, "\(value)")
            XCTAssertNotNil(session.finalization, "a bad notice does not cost the finalization record")
        }
        XCTAssertNil(try row(finalization: NSNull()).finalization)
    }

    /// `detail` keeps its meaning and is never shown in the notice's place.
    func testDetailIsNotTheNotice() throws {
        let session = try row(finalization: ["state": "complete", "final_solve": "solved",
                                             "detail": "Traceback (most recent call last): …"])
        XCTAssertNil(session.finalization?.notice)
        XCTAssertNil(WorldPickerView.notice(forOpened: session, target: WorldRenderTarget(worldID: "w1", sessionID: "s1")))
    }

    /// Shown for the ROOM, whatever `components` says -- a walk whose gate
    /// raised has `components: null` and can still carry one -- and never on
    /// an area's screen.
    func testTheNoticeBelongsToTheRoomRegardlessOfComponents() throws {
        let room = WorldRenderTarget(worldID: "w1", sessionID: "s1")
        let gateRaised = try row(finalization: ["state": "complete", "final_solve": "solved",
                                                "notice": "the evidence gate failed (a Tower error); the Tower re-runs it when it is idle"],
                                 components: nil)
        XCTAssertNil(gateRaised.components)
        XCTAssertEqual(WorldPickerView.notice(forOpened: gateRaised, target: room),
                       "the evidence gate failed (a Tower error); the Tower re-runs it when it is idle")
        let withAreas = try row(finalization: ["state": "complete", "final_solve": "solved", "notice": masksOff],
                                components: V2.components)
        XCTAssertEqual(WorldPickerView.notice(forOpened: withAreas, target: room), masksOff)
        XCTAssertNil(WorldPickerView.notice(forOpened: withAreas, target: room.area(V2.bathroomID)),
                     "an area's screen does not repeat the walk's notice")
        XCTAssertNil(WorldPickerView.notice(forOpened: nil, target: room), "a world row that named no session")
    }

    /// The same record on the status channel (`lifecycle.finalization`) decodes
    /// the notice too, for the live screen's room.
    func testTheStatusChannelsFinalizationCarriesItToo() {
        let payload: [String: Any] = ["lifecycle": ["finalization": [
            "state": "complete", "final_solve": "solved", "notice": masksOff]]]
        XCTAssertEqual(WorldBuilderResultDecoder.finalization(from: payload)?.notice, masksOff)
    }

    /// A notice arriving or leaving is a change the view model republishes
    /// (the report is `Equatable` over it), so it cannot stick on screen after
    /// the owed work is done.
    func testANoticeArrivingOrLeavingIsAChange() {
        let owed = WorldFinalizationReport(state: .complete, finalSolve: "solved", notice: masksOff)
        let done = WorldFinalizationReport(state: .complete, finalSolve: "solved")
        XCTAssertNotEqual(owed, done)
    }
}

// MARK: - Contract v7's reason `seed-unstable` (rule 8: unknown reasons are generic)

@MainActor
final class WorldComponentsUnknownReasonTests: XCTestCase {

    /// v7 (H2 consensus) adds `seed-unstable`. This build has no copy for any
    /// reason -- §8 shows none -- so a new one changes nothing a wearer sees:
    /// the entry decodes, the area opens, the line is the same.
    func testSeedUnstableIsCarriedAndChangesNothingOnScreen() throws {
        var list = V2.components
        list[1]["reason"] = "seed-unstable"
        list[1]["reasons"] = ["seed-unstable", "single-unconfirmed-link"]
        list[3]["reason"] = "seed-unstable"
        list[3]["reasons"] = ["seed-unstable"]
        let components = try XCTUnwrap(WorldComponents(json: list))
        XCTAssertEqual(components.areas.map(\.id), [V2.bedID, V2.bathroomID])
        XCTAssertEqual(components.areas[0].reason, "seed-unstable")
        XCTAssertEqual(components.areas[0].reasons, ["seed-unstable", "single-unconfirmed-link"])
        XCTAssertEqual(WorldComponentsPresentation.availability(of: components.areas[0]), .opens)
        let baseline = try XCTUnwrap(WorldComponents(json: V2.components))
        XCTAssertEqual(WorldComponentsPresentation.areaLine(number: 1, area: components.areas[0]),
                       WorldComponentsPresentation.areaLine(number: 1, area: baseline.areas[0]))
        XCTAssertEqual(WorldComponentsPresentation.footer(components), WorldComponentsPresentation.footer(baseline))
        XCTAssertEqual(WorldComponentsPresentation.areasHeading(components),
                       WorldComponentsPresentation.areasHeading(baseline))
    }

    /// A `placed` entry is only ever the room: a reason on it (which the Tower
    /// never sends) does not stop the list being trusted.
    func testAReasonWordNeverDecidesWhetherTheListIsTrusted() throws {
        var list = V2.components
        list[0]["reason"] = "seed-unstable"
        XCTAssertNotNil(WorldComponents(json: list))
    }
}

// MARK: - The notice guard (G1-F3: defence in depth)

@MainActor
final class WorldNoticeGuardTests: XCTestCase {

    /// Every §3.1 sentence (v6), with plausible `<why>` phrases, alone and
    /// joined as the Tower joins them: shown exactly as sent.
    private let contractSentences = [
        "masks were not applied (GPU out of memory); an owner can re-finish this walk",
        "masks were not applied (they are off on this Tower: TOWER_WORLD_SOLVE_MASKS); an operator can turn them on, then an owner can re-finish this walk",
        "masks were not applied (the transient detector did not load); an operator can make the transient detector run on this Tower, then an owner can re-finish this walk",
        "masks were applied by OneFormer alone, not by the union rule the evidence gate needs; an operator can make Grounding DINO and SAM available on this Tower, then an owner can re-finish this walk",
        "the evidence gate could not measure metric scale (depth did not finish); the Tower re-runs the gate when it is idle",
        "the evidence gate had too little metric scale to place pieces by it (too few cameras had a level); the depth stage ran to the end, so re-running the gate would not change this; an owner can re-capture this walk",
        "the evidence gate failed (an error in the gate); the Tower re-runs it when it is idle",
        "the evidence gate could not measure metric scale (depth did not finish); the idle Tower re-ran the gate 3 times without finishing it and has stopped trying; an owner can re-finish this walk",
        "the evidence gate failed (RuntimeError); the Tower re-runs it when it is idle",
        "the phone and/or the Tower lost 1/2 of the frames; an owner can re-capture this walk",
    ]

    func testEveryOwnerFacingSentenceIsShownVerbatim() {
        for sentence in contractSentences {
            XCTAssertEqual(WorldNoticeGuard.displayText(sentence), sentence, sentence)
        }
        // The worst honest case: every cause joined.
        let joined = contractSentences[3] + "; " + contractSentences[5] + "; " + contractSentences[7]
        XCTAssertEqual(WorldNoticeGuard.displayText(joined), joined)
        XCTAssertLessThan(joined.count, WorldNoticeGuard.maximumLength)
    }

    func testMachineOutputIsReplacedByTheGenericSentence() {
        let machine = [
            #"masks were not applied (C:\Users\tvllo\Projects\Glasses\tower\data\w1\solve); an owner can re-finish this walk"#,
            #"could not read \\tower\share\worlds\w1"#,
            "the evidence gate failed (/Users/tristan/Projects/Glasses/tower/tower/world_builder/gate.py); the Tower re-runs it",
            "the evidence gate failed (see ~/Projects/Glasses-scratch/run/log.txt)",
            "failed opening path=/var/lib/tower/worlds/w1",
            "Traceback (most recent call last):\n  File \"gate.py\", line 12, in run\nValueError: bad",
            "gate crashed: File \"/opt/tower/gate.py\", line 88, in solve",
            "ValueError: shapes (3,4) and (5,) not aligned",
            "RuntimeError(CUDA out of memory)",
            "tower.world_builder.coherence.GateError: the gate raised",
            "Error: the depth stage returned 0 cameras",
            "Exception in the depth stage",
            "masks were not applied\nsecond line",
            String(repeating: "masks were not applied; ", count: 40),
        ]
        for text in machine {
            XCTAssertEqual(WorldNoticeGuard.displayText(text), WorldNoticeGuard.genericSentence, text)
        }
    }

    func testTheGenericSentenceIsOwnerFacingAndStable() {
        let generic = WorldNoticeGuard.genericSentence
        XCTAssertFalse(WorldNoticeGuard.looksLikeMachineOutput(generic))
        XCTAssertEqual(WorldNoticeGuard.displayText(generic), generic, "idempotent")
        XCTAssertTrue(generic.contains("owner"))
    }

    func testNoNoticeStaysNoNotice() {
        for value in [nil, NSNull(), "", "  ", 5] as [Any?] {
            XCTAssertNil(WorldNoticeGuard.displayText(value))
        }
    }

    /// The room's notice goes through the guard wherever it is shown.
    func testTheRoomShowsTheGuardedText() throws {
        var json = V2.row(components: nil)
        json["finalization"] = ["state": "complete", "final_solve": "solved",
                                "notice": #"the evidence gate failed (C:\tower\gate.py)"#]
        let session = try XCTUnwrap(WorldListingSession(json: json))
        XCTAssertEqual(session.finalization?.notice, #"the evidence gate failed (C:\tower\gate.py)"#,
                       "the decoded record keeps what was sent")
        XCTAssertEqual(WorldPickerView.notice(forOpened: session, target: WorldRenderTarget(worldID: "w1", sessionID: "s1")),
                       WorldNoticeGuard.genericSentence)
    }
}
