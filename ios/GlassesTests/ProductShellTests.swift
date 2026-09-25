//
//  ProductShellTests.swift
//  GlassesTests
//
//  Guards the truthfulness invariants of the product shell and of the cartridge
//  integration layer built on it. Rule 3 (Truthful State Only) and Rule 16
//  (Epistemic Honesty) in docs/02-DEVELOPMENT-RULES.md, plus the display rules
//  in docs/06-PRIVACY-DATA.md and docs/07-PLATFORM-CONSTRAINTS.md.
//
//  In rough order: the cartridge catalog and its roadmap statuses; workspace
//  selection; the shared availability/phase layer; the four cartridge clients
//  and view models; redaction; provenance and observation time; and the four
//  cartridges' own domain rules.
//
//  Everything here lives in one file because GlassesTests is NOT a
//  filesystem-synchronized group — its members are listed explicitly in
//  project.pbxproj — so adding a test file means hand-editing that file, which
//  was judged the worse risk. New suites are appended rather than split out.
//

import Combine
import MWDATCore
import XCTest
#if DEBUG
// For `MWDATCamera.StreamState`, which `isCaptureSessionClaimed` switches over.
// The app imports MWDATCamera only under DEBUG, and everything here that names
// it is DEBUG-only for the same reason.
import MWDATCamera
// For the DAT 1.0.0 Motion spike at the end of this file: Mock Device Kit's
// `MockMotionKit`. It declares its own `MotionSample` and `Vector3`, distinct
// from MWDATMotion's, so the tests qualify them.
import MWDATMockDevice
// Test-only, and only to write a silent, video-only file for the mock camera
// feed in that same suite. The audio-free invariant is about the app: its
// sources under Glasses/ and its built binaries, which this file is neither.
import AVFoundation
// For the IMU recorder suite: UIKit's protected-data notification names, which
// the recorder restates by value, and `MotionSamplingRate`.
import UIKit
import MWDATMotion
#endif

@testable import Glasses

// `Combine` is imported for the cartridge client protocols' `stateUpdates`
// requirement, whose type is `AnyPublisher`. The test doubles below satisfy it
// through the protocol extensions' default rather than implementing it, so this
// import is belt-and-braces — but the target enables
// SWIFT_UPCOMING_FEATURE_MEMBER_IMPORT_VISIBILITY, and being explicit about a
// module whose types appear in a conformance costs nothing.

final class CartridgeCatalogTests: XCTestCase {

    func testCatalogIsNotEmpty() {
        XCTAssertFalse(Cartridge.catalog.isEmpty)
    }

    func testCartridgeIdentifiersAreUnique() {
        let ids = Cartridge.catalog.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count, "Duplicate cartridge id would break ForEach identity")
    }

    /// **This test was the tripwire that guarded the old badge vocabulary, and
    /// it has been answered rather than deleted.**
    ///
    /// It used to read: *"nothing is runnable yet. If a future change adds an
    /// 'available'/'active' status, this test should fail and force a
    /// deliberate decision about whether the Tower actually supports it (module
    /// container is V0.8, first module V0.9)."*
    ///
    /// It fired, and here is the deliberate decision. The Tower **does**
    /// support it: probed live on 2026-08-26 it reports `module_state:
    /// "active"` with `module_id: "experimental-cv"`, and `GET /cartridges`
    /// declares `world_builder.status/2026-08-25` with `available: true` — a
    /// contract this build implements and has drawn live geometry from on
    /// hardware. The premise the tripwire defended is gone.
    ///
    /// What replaces it is the invariant that is still worth defending: **a
    /// badge may not claim more than the app can open.** A cartridge with no
    /// workspace has nothing for a person to try, so it can never be
    /// `.readyToTest`, and one with a workspace is never `.notBuilt`. That is
    /// asserted in both directions so neither half can drift.
    func testABadgeNeverClaimsMoreThanTheAppCanOpen() {
        for cartridge in Cartridge.catalog {
            if cartridge.workspace == nil {
                XCTAssertEqual(
                    cartridge.status,
                    .notBuilt,
                    "\(cartridge.name) advertises '\(cartridge.status.badge)' with no workspace to open"
                )
            } else {
                XCTAssertNotEqual(
                    cartridge.status,
                    .notBuilt,
                    "\(cartridge.name) ships a workspace but calls itself not built"
                )
            }
        }
    }

    /// The badge is a fact about this build, not about the Tower on the other
    /// end of the socket.
    ///
    /// `CartridgeStatus` must consult nothing at runtime — a live Tower's
    /// ability to serve a cartridge is resolved per-connection by
    /// `CartridgeAvailability.resolve` and rendered inside the workspace. If
    /// this ever became connection-dependent, "Ready to test" would start
    /// flickering with the network and the drawer would stop being a stable
    /// answer to "what is in this build".
    func testTheBadgeIsStaticAndDoesNotTrackTheTower() {
        let first = Cartridge.catalog.map(\.status)
        let second = Cartridge.catalog.map(\.status)
        XCTAssertEqual(first, second)
        XCTAssertEqual(
            Cartridge.catalog.first { $0.id == "object-memory" }?.status,
            .readyToTest,
            "Object Memory stays ready to test even though the live Tower answers 404 — that is one Tower's configuration, not this build's capability"
        )
    }

    func testEveryCartridgeCitesASpec() {
        for cartridge in Cartridge.catalog {
            XCTAssertTrue(
                cartridge.specPath.hasPrefix("docs/modules/"),
                "\(cartridge.name) must point at a real spec file"
            )
        }
    }
}

final class StateDisplayTests: XCTestCase {

    /// `PermissionStatus` has no `notDetermined` case, so a nil value means
    /// "not asked yet" — it must never render as denied or unknown.
    func testUncheckedCameraPermissionIsDistinctFromDenied() {
        XCTAssertEqual(StateDisplay.cameraPermission(nil), "Not checked yet")
        XCTAssertEqual(StateDisplay.cameraPermission(.denied), "Denied")
        XCTAssertEqual(StateDisplay.cameraPermission(.granted), "Allowed")
    }

    func testRegistrationStatesAreHumanReadable() {
        XCTAssertEqual(StateDisplay.registration(.registered), "Registered")
        XCTAssertEqual(StateDisplay.registration(.registering), "Registering…")
        XCTAssertEqual(StateDisplay.registration(.available), "Not registered")
        XCTAssertEqual(StateDisplay.registration(.unavailable), "Meta AI unavailable")
    }

    /// A failed Tower status carries a URLSession message that can run to a
    /// full sentence. The pill shows a fixed short string; the detail is
    /// routed to the banner instead.
    func testTowerFailureDetailIsSeparatedFromTheStatusLabel() {
        let status = TowerStatus.failed("Connection lost: The network connection was lost.")
        XCTAssertEqual(StateDisplay.tower(status), "Disconnected")
        XCTAssertEqual(
            StateDisplay.towerFailureDetail(status),
            "Connection lost: The network connection was lost."
        )
    }

    func testTowerFailureDetailIsNilWhenHealthy() {
        XCTAssertNil(StateDisplay.towerFailureDetail(.online))
        XCTAssertNil(StateDisplay.towerFailureDetail(.offline))
        XCTAssertNil(StateDisplay.towerFailureDetail(.connecting))
    }

    func testTowerStatusLabels() {
        XCTAssertEqual(StateDisplay.tower(.offline), "Offline")
        XCTAssertEqual(StateDisplay.tower(.connecting), "Connecting…")
        XCTAssertEqual(StateDisplay.tower(.online), "Connected")
    }
}

// MARK: - Workspaces

/// Guards the separation that lets a World Builder workspace exist without any
/// of it claiming the Tower can build a world.
@MainActor
final class CartridgeWorkspaceTests: XCTestCase {

    /// The load-bearing invariant of the whole Product Shell V2 design. Giving
    /// a cartridge a workspace is a statement about *this app*; `status` is a
    /// statement about the *Tower's* roadmap. If having a screen ever silently
    /// promoted a module's status, the drawer would start advertising a runtime
    /// that still does not exist.
    func testHavingAWorkspaceDoesNotPromoteACartridgesStatus() {
        let withWorkspaces = Cartridge.catalog.filter { $0.workspace != nil }
        XCTAssertFalse(withWorkspaces.isEmpty, "this test is vacuous if nothing has a workspace")

        // Every status in the catalog, pinned against docs/03-ROADMAP.md and
        // the module specs — not merely "not promoted to next".
        //
        // ## Why this replaced a narrower assertion
        //
        // Product Shell V2 asserted two things: that World Builder was still
        // `.future`, and that the `.next` cartridge had no workspace at all.
        // The second held only because Experimental CV Lab was the one
        // cartridge with a roadmap position and no screen — an accident of the
        // ordering in which workspaces were built, not the invariant.
        // Experimental CV Lab has a workspace now, and that assertion would
        // fail while nothing it was protecting had gone wrong.
        //
        // The invariant it was protecting is "a screen never changes a
        // roadmap position", and the honest way to state that is the whole
        // map. This is strictly stronger than what it replaced: it catches a
        // drift on *any* cartridge, workspace or not, and a new cartridge
        // added without a deliberate decision about its status fails here
        // rather than passing silently.
        let expected: [String: CartridgeStatus] = [
            // Each value is justified by evidence, cited here so a future
            // change has to argue with the evidence rather than with a habit.
            "experimental-cv": .readyToTest,      // the Tower's only running module; module_state "active"
            "object-memory": .readyToTest,        // two live HTTP routes, decoder pinned against a real Tower
            "world-build": .readyToTest,          // the one offered contract; device-validated walk, 7,086 points
            "document-memory": .readyToTest,      // declares status/2026-08-27 + library under http_contracts
            "scene-understanding": .readyToTest,  // declares live/2026-08-27; `live` answers the persistence objection
            "visual-qa": .notBuilt,               // zero Tower code
            "accessibility": .notBuilt,           // zero Tower code
            "environmental-memory": .notBuilt,    // zero Tower code; its own design says do not begin
        ]
        XCTAssertEqual(
            Set(Cartridge.catalog.map(\.id)),
            Set(expected.keys),
            "a cartridge was added or removed without updating this roadmap pin"
        )
        for cartridge in Cartridge.catalog {
            XCTAssertEqual(
                cartridge.status,
                expected[cartridge.id],
                "\(cartridge.name)'s status drifted from the evidence recorded beside it"
            )
        }

        // Restated positively. This count moved from three to five on
        // 2026-08-27 when the Tower unified four lanes, and the number is
        // asserted rather than derived so that moving it stays a decision
        // somebody makes on evidence.
        //
        // `.awaitingTower` now has **no members**, and that is the honest
        // state rather than a gap: the case meant "the Tower implements this
        // and offers no contract for it", and there is no longer a cartridge
        // in that position — every one with Tower code behind it now declares
        // something. The case is kept because it is the correct answer the
        // next time a Tower lane finishes an engine before its contract, which
        // is the ordinary order of work.
        XCTAssertEqual(
            Cartridge.catalog.filter { $0.status == .readyToTest }.count,
            5,
            """
            five cartridges reach a person today: World Builder, Object Memory, \
            Experimental CV Lab, Document Memory, Scene Understanding
            """
        )
        XCTAssertTrue(
            Cartridge.catalog.filter { $0.status == .awaitingTower }.isEmpty,
            "a cartridge is waiting on a Tower contract again; say which and why beside it"
        )
    }

    /// The availability guard, asserted rather than implied.
    ///
    /// `testNoCartridgeClaimsToBeAvailable` and
    /// `testWorkspacesDidNotIntroduceAnAvailabilityClaim` both compare
    /// `status.badge` against `["Up next", "Planned", "Future"]` — which is the
    /// **exhaustive** set of badges `CartridgeStatus` can currently emit. Neither
    /// can fail while the enum has three cases, and a new `.available` case
    /// whose badge happened to read "Planned" would sail through both. The
    /// invariant they advertise — that no `available`/`active` case exists — was
    /// never actually asserted.
    ///
    /// This asserts it directly, via an exhaustive switch. A new case makes the
    /// switch non-exhaustive and the compiler stops the build, which is a louder
    /// failure than a red test and exactly the deliberate decision point
    /// `08-IOS-CARTRIDGE-SHELL.md` wants.
    ///
    /// The two pre-existing tests are left untouched. They were Mac-validated,
    /// they are not wrong, and they still guard the badge strings.
    func testEveryStatusIsADeliberateDecisionWithItsOwnBadge() {
        for status in CartridgeStatus.allCases {
            switch status {
            case .readyToTest, .awaitingTower, .notBuilt:
                continue
            // A case added here makes this switch non-exhaustive and stops the
            // build, which is louder than a red test and is the point. Before
            // adding one, answer the question this vocabulary exists to answer:
            // what can a person DO with it in this build? If the answer is a
            // roadmap position, it does not belong here — roadmap position
            // lives in docs/03-ROADMAP.md and each cartridge's specPath, and
            // rendering it in the drawer is what produced a catalog where
            // Visual Q&A read "Planned" with no backend while World Builder
            // read "Future" with a device-validated walk behind it.
            }
        }
        XCTAssertEqual(
            Set(CartridgeStatus.allCases.map(\.badge)).count,
            CartridgeStatus.allCases.count,
            "two statuses render the same badge, so the drawer cannot distinguish them"
        )
        XCTAssertEqual(
            CartridgeStatus.allCases.filter(\.isProminent),
            [.readyToTest],
            "exactly one status may draw the eye, or the drawer stops answering 'which can I try' in a glance"
        )
    }

    /// Every cartridge cites a spec path this repository actually contains.
    ///
    /// `testEveryCartridgeCitesASpec` asserts only the `docs/modules/` **prefix**,
    /// so it passes for `docs/modules/DOES-NOT-EXIST.md`. A test bundle cannot
    /// stat the repository, so the honest substitute is to pin the exact set —
    /// which catches a typo in a new entry and forces a deliberate edit when a
    /// spec is genuinely added.
    func testEveryCartridgeCitesASpecThatExists() {
        let known: Set<String> = [
            "docs/modules/EXPERIMENTAL-CV.md",
            "docs/modules/OBJECT-MEMORY.md",
            "docs/modules/VISUAL-QA.md",
            "docs/modules/WORLD-BUILD.md",
            "docs/modules/ACCESSIBILITY.md",
            "docs/modules/ENVIRONMENTAL-MEMORY.md",
            "docs/modules/DOCUMENT-MEMORY.md",
            "docs/modules/SCENE-UNDERSTANDING.md",
        ]
        for cartridge in Cartridge.catalog {
            XCTAssertTrue(
                known.contains(cartridge.specPath),
                "\(cartridge.name) cites \(cartridge.specPath), which is not a spec file in this repository"
            )
        }
    }

    /// Every workspace this app compiles must belong to a cartridge, or it is
    /// unreachable code that no drawer row can open.
    ///
    /// The reason `CartridgeWorkspace` is `CaseIterable`: without iterating the
    /// cases, a workspace added to the enum and to `ContentView` but never
    /// attached to a catalog entry compiles, passes every other test, and can
    /// never be opened.
    func testEveryWorkspaceIsReachableFromTheCatalog() {
        let attached = Set(Cartridge.catalog.compactMap(\.workspace))
        for workspace in CartridgeWorkspace.allCases {
            XCTAssertTrue(
                attached.contains(workspace),
                "\(workspace) has no cartridge, so nothing can open it"
            )
        }
    }

    /// Openability and the badge are now two views of one fact, so they must
    /// agree exactly.
    ///
    /// The drawer decides tappability from `CartridgeDrawerRow`; the badge is
    /// read off `status`. If those disagreed, a row would either invite a tap
    /// that goes nowhere or hide a cartridge a person was just told is ready.
    func testTheOpenableRowsAreExactlyTheCartridgesThatAreNotUnbuilt() {
        XCTAssertEqual(
            Cartridge.selectable.map(\.id),
            Cartridge.catalog.filter { $0.status != .notBuilt }.map(\.id)
        )
    }

    func testExactlyTheCartridgesWithWorkspacesAreSelectable() {
        let expected = Cartridge.catalog.filter { $0.workspace != nil }.map(\.id)
        XCTAssertEqual(Cartridge.selectable.map(\.id), expected)
        XCTAssertFalse(Cartridge.selectable.isEmpty)
    }

    func testWorldBuilderIsTheWorkspaceThatExists() {
        let worldBuilder = Cartridge.catalog.first { $0.workspace == .worldBuilder }
        XCTAssertNotNil(worldBuilder)
        XCTAssertEqual(worldBuilder?.id, "world-build")
    }

    // MARK: Restoring a persisted selection

    func testAStoredIdentifierResolvesToItsCartridge() {
        let resolved = Cartridge.workspaceCartridge(forID: "world-build")
        XCTAssertEqual(resolved?.workspace, .worldBuilder)
    }

    func testAnUnknownStoredIdentifierResolvesToNothing() {
        XCTAssertNil(Cartridge.workspaceCartridge(forID: "not-a-cartridge"))
        XCTAssertNil(Cartridge.workspaceCartridge(forID: ""))
        XCTAssertNil(Cartridge.workspaceCartridge(forID: nil))
    }

    /// A stored identifier outlives the build that wrote it. If a cartridge
    /// loses its workspace in a later release, a persisted selection must not
    /// resurrect a screen that no longer exists — it must fall back to Home.
    func testAStoredCartridgeWithoutAWorkspaceDoesNotReopen() throws {
        // Unwrapped rather than optional-chained: if every cartridge gained a
        // workspace this test would otherwise pass while asserting nothing.
        let withoutWorkspace = try XCTUnwrap(
            Cartridge.catalog.first { $0.workspace == nil },
            "no workspace-less cartridge left to test the fallback with"
        )
        XCTAssertNil(Cartridge.workspaceCartridge(forID: withoutWorkspace.id))
    }

    // MARK: One answer to "is this cartridge openable?"

    /// `Cartridge.selectable` is documented as "every cartridge the drawer may
    /// present as openable", but for most of this shell's life the drawer did
    /// not consult it — or anything derived from it. It iterated
    /// `Cartridge.catalog` and re-derived openability inline as
    /// `cartridge.workspace != nil`, twice: once to decide whether to wrap the
    /// row in a `Button`, and once inside the row to pick the accessibility
    /// hint. `selectable`'s only callers were in this file.
    ///
    /// Two code paths answering one question agreed only by coincidence of
    /// implementation. Nothing made them agree, so nothing would have caught
    /// them diverging — a cartridge could have been openable to the drawer and
    /// absent from `selectable`, and every test here would still have passed
    /// while asserting the wrong list.
    ///
    /// `Cartridge.drawerRows` is now the single answer. The drawer renders it;
    /// `selectable` is defined as the openable rows of it. These tests pin that
    /// definition, so the identity below is not a coincidence being observed —
    /// it is the construction being checked.
    func testTheDrawerRendersEveryCatalogEntryInCatalogOrder() {
        XCTAssertEqual(
            Cartridge.drawerRows.map(\.cartridge.id),
            Cartridge.catalog.map(\.id),
            "the drawer shows all catalog entries, openable or not — dropping the informational rows would hide three modules"
        )
    }

    func testTheDrawersOpenableRowsAreExactlyTheSelectableCartridges() {
        let openable = Cartridge.drawerRows.filter(\.isOpenable).map(\.cartridge.id)
        XCTAssertEqual(
            openable,
            Cartridge.selectable.map(\.id),
            "the drawer and Cartridge.selectable disagree about which cartridges may be opened"
        )
        XCTAssertFalse(openable.isEmpty, "this test is vacuous if nothing is openable")
    }

    /// The other half of the same guarantee: a row is openable **if and only
    /// if** its cartridge has a workspace. The `.openable` case carries a
    /// non-optional `CartridgeWorkspace`, so the compiler already forbids an
    /// openable row with nothing to open; this pins the converse, that a
    /// cartridge with a workspace cannot be filed as informational.
    func testARowIsOpenableExactlyWhenItsCartridgeHasAWorkspace() {
        for row in Cartridge.drawerRows {
            switch row {
            case .openable(let cartridge, let workspace):
                XCTAssertEqual(
                    cartridge.workspace,
                    workspace,
                    "\(cartridge.name) opens a workspace that is not its own"
                )
            case .informational(let cartridge):
                XCTAssertNil(
                    cartridge.workspace,
                    "\(cartridge.name) has a workspace but the drawer renders it as informational"
                )
            }
        }
    }

    /// The three rows with no workspace, named. `testAStoredCartridgeWithout…`
    /// only needs one of them to exist; this pins which three, so a cartridge
    /// silently losing its workspace fails here rather than quietly becoming an
    /// informational row in a shipped build.
    func testTheThreeCartridgesWithoutAWorkspaceStayInformational() {
        let informational = Cartridge.drawerRows.filter { !$0.isOpenable }.map(\.cartridge.id)
        XCTAssertEqual(informational, ["visual-qa", "accessibility", "environmental-memory"])
    }

    /// Both hint strings, pinned to the same decision that decides tappability.
    ///
    /// These are the only two sentences VoiceOver reads that state whether a
    /// row does anything. If the hint and the `Button` were ever derived
    /// separately, one of them would eventually say a row opens something it
    /// cannot open — a lie told only to the users who cannot see that nothing
    /// happened (Rule 3, Truthful State Only).
    func testTheAccessibilityHintFollowsTheSameOpenabilityDecision() {
        for row in Cartridge.drawerRows {
            XCTAssertEqual(
                row.accessibilityHint,
                row.isOpenable ? "Opens this workspace" : "No workspace in this app yet",
                "\(row.cartridge.name)'s hint does not match its openability"
            )
        }
        XCTAssertEqual(
            Set(Cartridge.drawerRows.map(\.accessibilityHint)),
            ["Opens this workspace", "No workspace in this app yet"],
            "both hints must still be reachable — a drawer with one hint has stopped distinguishing the two kinds of row"
        )
    }
}

// MARK: - World model boundary

/// The Tower has no world builder. These tests pin what the app is allowed to
/// say about that, and pin the display rule `docs/modules/WORLD-BUILD.md`
/// imposes on any spatial figure the Tower eventually sends.
@MainActor
final class WorldModelTests: XCTestCase {

    // MARK: Scale provenance

    /// The monocular-depth rule, as a test. WORLD-BUILD.md forbids ever
    /// presenting inferred depth as ground truth, so exactly one case may be
    /// rendered as an estimate and the others must not be silently folded into
    /// it.
    func testOnlyInferredDepthIsMarkedAsAnEstimate() {
        XCTAssertTrue(WorldScaleSemantics.inferredMetric.isEstimate)
        XCTAssertFalse(WorldScaleSemantics.relative.isEstimate)
        XCTAssertFalse(WorldScaleSemantics.measuredMetric.isEstimate)
        XCTAssertFalse(WorldScaleSemantics.unknown.isEstimate)
    }

    /// Every provenance has to be sayable. A missing string would leave a view
    /// to compose its own explanation, which is how an inferred figure ends up
    /// described as measured.
    func testEveryScaleCanBeExplainedToAPerson() {
        // `allCases`, not a hand-written list: a future case would otherwise
        // slip past this unnoticed, which is the one thing the display rule
        // cannot tolerate.
        XCTAssertGreaterThanOrEqual(WorldScaleSemantics.allCases.count, 4)
        for scale in WorldScaleSemantics.allCases {
            XCTAssertFalse(scale.displayName.isEmpty, "\(scale) has no label")
            XCTAssertFalse(scale.explanation.isEmpty, "\(scale) has no explanation")
        }
    }

    /// Unknown scale must read as unknown, never as a benign-looking default.
    func testUnknownScaleIsNotDressedUpAsRelative() {
        XCTAssertNotEqual(WorldScaleSemantics.unknown.displayName, WorldScaleSemantics.relative.displayName)
        XCTAssertEqual(WorldScaleSemantics.unknown.displayName, "Unknown")
    }

    // MARK: State

    /// The three states that must never imply a world exists. `.unsupported` is
    /// today's only reachable state; `.awaitingFirstUpdate` is the one most
    /// likely to be misdrawn later as "a world is appearing".
    func testStatesWithoutAWorldDoNotClaimOne() {
        let empty: [WorldModelState] = [
            .unsupported(reason: "no Tower support"),
            .idle,
            .awaitingFirstUpdate,
            .failed(CartridgeFailure(kind: .transport, message: "boom")),
        ]
        for state in empty {
            XCTAssertFalse(state.hasWorld, "\(state) claims a world")
            XCTAssertFalse(state.isReceivingUpdates, "\(state) claims live updates")
            XCTAssertNil(state.snapshot, "\(state) produced a snapshot from nothing")
        }
    }

    func testOnlyReceivingCountsAsLive() {
        let snapshot = WorldSnapshot(keyframeCount: 12)
        XCTAssertTrue(WorldModelState.receiving(snapshot).isReceivingUpdates)
        XCTAssertFalse(
            WorldModelState.finalized(snapshot).isReceivingUpdates,
            "a finished world is inspectable, not live"
        )
    }

    func testAWorldSurvivesFinalisationSoItStaysInspectable() {
        let snapshot = WorldSnapshot(name: "Bedroom", keyframeCount: 37)
        XCTAssertTrue(WorldModelState.finalized(snapshot).hasWorld)
        XCTAssertEqual(WorldModelState.finalized(snapshot).snapshot, snapshot)
    }

    // MARK: Snapshot emptiness

    /// A Tower that reports nothing must not be rendered as a world with zeroes
    /// in it. `nil` and `0` are different claims: one is "not reported", the
    /// other is "reported as none".
    func testADefaultSnapshotReportsNothingRatherThanZero() {
        let snapshot = WorldSnapshot()
        XCTAssertTrue(snapshot.isEmpty)
        XCTAssertNil(snapshot.keyframeCount)
        XCTAssertNil(snapshot.mappingSeconds)
        XCTAssertEqual(snapshot.tracking, .unavailable)
        XCTAssertEqual(snapshot.scale, .unknown)
    }

    func testASnapshotWithAnyRealFieldIsNotEmpty() {
        XCTAssertFalse(WorldSnapshot(keyframeCount: 0).isEmpty, "zero keyframes is still a report")
        XCTAssertFalse(WorldSnapshot(name: "Bedroom").isEmpty)
        XCTAssertFalse(WorldSnapshot(tracking: .lost).isEmpty)
        XCTAssertFalse(WorldSnapshot(scale: .relative).isEmpty)
    }

    // MARK: The only source that exists

    /// Today's whole truth: there is no Tower world builder, and the app says
    /// exactly that and nothing more.
    ///
    /// `UnavailableWorldModelSource` was renamed `UnavailableWorldBuilderClient`
    /// so all four cartridges name this layer the same way. The contract is
    /// unchanged.
    func testTheOnlyClientReportsTheCapabilityIsAbsent() {
        let client = UnavailableWorldBuilderClient()
        guard case .unsupported(let reason) = client.state else {
            return XCTFail("expected .unsupported, got \(client.state)")
        }
        XCTAssertFalse(reason.isEmpty, "an unsupported state must explain itself")
        XCTAssertFalse(client.state.hasWorld)
        XCTAssertFalse(client.state.isReceivingUpdates)
    }
}

// MARK: - Shared cartridge integration layer

/// Guards the small shared layer the four cartridges sit on: what the Tower has
/// declared (nothing), how availability is resolved from that, and the rule that
/// a phase without data may not carry data.
///
/// These are the tests that would notice if a future change made the app *look*
/// integrated before the Tower was.
@MainActor
final class CartridgeIntegrationTests: XCTestCase {

    // MARK: What the Tower has declared

    /// The state of Tower integration, asserted rather than assumed.
    ///
    /// This test used to assert both tables were empty, and its own comment
    /// said that the first real contract landing should make it fail as a
    /// signal to review every consumer. That happened: the Tower now declares
    /// `world_builder.status/2026-08-25` over the socket, and every consumer
    /// was reviewed. What it pins now is the same property in its new form —
    /// **exactly one contract is implemented, and it is that one** — so a
    /// second one appearing is still a review and not a silent widening.
    func testTheImplementedContractsAreExactlyTheFiveThisBuildDecodes() {
        XCTAssertEqual(
            TowerCapabilities.supported,
            [
                WorldBuilderResultContract.identifier,
                ExperimentalCVContract.status,
                SceneUnderstandingContract.identifier,
                DocumentMemoryContract.statusIdentifier,
                DocumentMemoryContract.libraryIdentifier
            ],
            "this build's implemented contracts changed; every cartridge's client must be reviewed"
        )
        XCTAssertTrue(
            TowerCapabilities.declared.isEmpty,
            """
            a contract was hardcoded into the local table. Every one this app \
            implements arrives over the wire, and a compile-time copy is a second \
            answer that can disagree with the Tower's.
            """
        )
    }

    /// Object Memory has a complete client and no declaration, deliberately.
    ///
    /// This is the one asymmetry in the system that looks like a bug from every
    /// angle, so it is pinned from the iOS side rather than left to be
    /// rediscovered. The Tower's own contract §9 says: *"Object Memory is NOT
    /// in `GET /cartridges`, and that is deliberate… Reach Object Memory over
    /// HTTP. Learn nothing about it from the declaration."* A
    /// `result_subscribe` for it is refused `unknown_cartridge`, and the Tower
    /// pins that refusal on its side too.
    ///
    /// So the absence below is the assertion, not an omission in it. If Object
    /// Memory ever gains a socket declaration, this test fails and **that is
    /// the review** — the two halves must land together, and the Tower lane
    /// recorded the decision as a human's to make.
    func testObjectMemoryIsReachedWithoutADeclaration() {
        XCTAssertNil(
            TowerCapabilities.towerCartridgeNames["object-memory"],
            """
            Object Memory gained a Tower name. It is undeclared by design and \
            reached over HTTP; a name here would make it look subscribable.
            """
        )
        XCTAssertFalse(
            TowerCapabilities.supported.contains(where: { $0.hasPrefix("object_memory.") }),
            "an object_memory contract entered the subscription set; it has none on the socket"
        )
    }

    /// The Tower's name for a cartridge is not this app's, and only cartridges
    /// with a client that decodes a declared contract have both.
    ///
    /// The hyphen/underscore split is not cosmetic: `experimental-cv` against
    /// `experimental_cv` is the mapping without which a complete client, a
    /// correct decoder and a live offer still resolve to "nothing here".
    func testTheCartridgeNameMappingCoversTheDeclaredCartridges() {
        XCTAssertEqual(
            TowerCapabilities.towerCartridgeNames,
            [
                "world-build": "world_builder",
                "experimental-cv": "experimental_cv",
                "scene-understanding": "scene_understanding",
                "document-memory": "document_memory"
            ]
        )
        let mapped = Set(TowerCapabilities.towerCartridgeNames.keys)
        for cartridge in Cartridge.catalog where !mapped.contains(cartridge.id) {
            XCTAssertNil(
                TowerCapabilities.towerCartridgeNames[cartridge.id],
                "\(cartridge.name) gained a Tower name without a client to use it"
            )
        }
    }

    // MARK: Availability against a live declaration

    private func declaration(
        cartridge: String = "world_builder",
        contract: String = WorldBuilderResultContract.identifier,
        available: Bool = true,
        reason: String? = nil
    ) -> TowerCartridgeDeclaration {
        TowerCartridgeDeclaration(
            envelopeContract: "cartridge_results.envelope/2026-08-23",
            offers: [
                TowerCartridgeOffer(
                    json: [
                        "cartridge": cartridge,
                        "result_type": "status",
                        "contract": contract,
                        "available": available,
                        "unavailable_reason": reason as Any,
                        "snapshot_only": true,
                    ]
                )!
            ]
        )
    }

    /// A Tower that has declared nothing is indistinguishable from one that
    /// never will — which is correct, because from here it is.
    func testWorldBuilderIsUnavailableUntilTheTowerDeclaresIt() {
        XCTAssertEqual(
            TowerCapabilities.availability(
                for: "world-build",
                declaredBy: nil,
                isTowerReachable: true
            ),
            .noContract
        )
    }

    /// The declaration is what makes it available, and connectivity is the
    /// second gate rather than the first.
    func testADeclaredWorldBuilderContractBecomesAvailableWhenReachable() {
        let declared = declaration()
        XCTAssertEqual(
            TowerCapabilities.availability(
                for: "world-build",
                declaredBy: declared,
                isTowerReachable: true
            ),
            .available(
                CartridgeContract(
                    cartridgeID: "world-build",
                    identifier: WorldBuilderResultContract.identifier
                )
            )
        )
        XCTAssertEqual(
            TowerCapabilities.availability(
                for: "world-build",
                declaredBy: declared,
                isTowerReachable: false
            ),
            .towerUnreachable,
            "a declared contract must read as disconnected, not as absent, while the socket is down"
        )
    }

    /// A Tower speaking a different dated contract is a disagreement, not a
    /// version to compare. It must reach `.unsupportedContract`, which tells a
    /// person to update the app rather than to reconnect.
    func testAnUndatedOrLaterContractIsNotDecodedOnAGuess() {
        for identifier in ["world_builder.status/2027-01-01", "world_builder.status/2026-01-01", "v2"] {
            XCTAssertEqual(
                TowerCapabilities.availability(
                    for: "world-build",
                    declaredBy: declaration(contract: identifier),
                    isTowerReachable: true
                ),
                .unsupportedContract(
                    declared: CartridgeContract(cartridgeID: "world-build", identifier: identifier)
                ),
                "contract \(identifier) was treated as compatible"
            )
        }
    }

    /// `available: false` is an offer, not silence. Collapsing it to
    /// `.noContract` would render "no world root is configured" as "this Tower
    /// will never do this" — a different and wrong claim, calling for a
    /// different response from a person.
    func testAnUnavailableOfferIsStillAnOffer() {
        XCTAssertEqual(
            TowerCapabilities.availability(
                for: "world-build",
                declaredBy: declaration(available: false, reason: "no world root is configured"),
                isTowerReachable: true
            ),
            .available(
                CartridgeContract(
                    cartridgeID: "world-build",
                    identifier: WorldBuilderResultContract.identifier
                )
            )
        )
    }

    /// A declaration naming some other cartridge must not make World Builder
    /// available, and a cartridge this build has no name for stays absent.
    ///
    /// This test used to use `scene_understanding` as its "other cartridge",
    /// and that stopped being a valid choice on 2026-08-27: Scene Understanding
    /// now has a name mapping and a client, so a declaration for it resolves to
    /// something. `visual-qa` is the replacement because it is the case the
    /// test is actually about — a cartridge with **no Tower code anywhere**,
    /// which no amount of declaring can change.
    func testADeclarationForAnotherCartridgeChangesNothing() {
        let other = declaration(cartridge: "visual_qa", contract: "visual_qa.v1")
        XCTAssertEqual(
            TowerCapabilities.availability(
                for: "world-build",
                declaredBy: other,
                isTowerReachable: true
            ),
            .noContract
        )
        XCTAssertEqual(
            TowerCapabilities.availability(
                for: "visual-qa",
                declaredBy: other,
                isTowerReachable: true
            ),
            .noContract,
            "a cartridge with no name mapping and no client became available"
        )
    }

    /// The contract's **third state**, which this app could not previously
    /// express and which is the whole reason `GET /cartridges` exists over HTTP
    /// as well as the socket.
    ///
    /// The Tower's §2 puts three outcomes side by side and says they call for
    /// *opposite* instructions to a person:
    ///
    /// | the declaration says | show |
    /// |---|---|
    /// | absent from every list | "not built yet" |
    /// | present, contract this build cannot read | **"update the app"** |
    /// | present, `available: false` | "connect" / the reason |
    ///
    /// The middle row is the one worth a test, because collapsing it into the
    /// first tells someone a feature does not exist when they are one update
    /// away from it. Before the name mapping existed, every unmapped cartridge
    /// fell into the first row regardless of what the Tower said — which is why
    /// this could not be asserted until now.
    func testADeclaredContractThisBuildCannotReadAsksForAnUpdateNotForPatience() {
        let futureTower = declaration(
            cartridge: "scene_understanding",
            contract: "scene_understanding.live/2027-01-01"
        )
        XCTAssertEqual(
            TowerCapabilities.availability(
                for: "scene-understanding",
                declaredBy: futureTower,
                isTowerReachable: true
            ),
            .unsupportedContract(
                declared: CartridgeContract(
                    cartridgeID: "scene-understanding",
                    identifier: "scene_understanding.live/2027-01-01"
                )
            ),
            """
            a Tower speaking a newer contract resolved to "not built" rather than \
            "update the app". Those are opposite instructions to a person.
            """
        )
    }

    /// No cartridge becomes usable by connecting **alone**. Connectivity is the
    /// second gate, never the first: a declaration has to arrive as well, and
    /// a UI that suggested reconnecting would fix a missing contract would send
    /// a user round a loop that cannot terminate.
    ///
    /// Asserted through the no-declaration entry point.
    ///
    /// ## Why this no longer expects one answer for all eight
    ///
    /// It used to assert `.noContract` for every cartridge in both reachability
    /// states, which was right when the Tower declared nothing. **With no
    /// declaration and no connection, the honest answer now depends on whether
    /// this build could ever receive one:**
    ///
    /// - A cartridge with a client and a Tower-name mapping — World Builder,
    ///   the CV Lab, Scene, Document — gets `.towerUnreachable`. Saying "this
    ///   Tower has not declared it" on a cold launch is a claim about a machine
    ///   nobody has spoken to, and the shipped `.noContract` string goes
    ///   further and denies being about the connection.
    /// - A cartridge with **no Tower code anywhere** — Visual Q&A,
    ///   Accessibility, Environmental Memory — stays `.noContract` in both
    ///   states, because "the Tower is unreachable" would be its own false
    ///   story: reconnecting cannot help.
    /// - Object Memory stays `.noContract` here too, and that is not an
    ///   oversight: it is undeclared by design and resolves its real
    ///   availability by probing over HTTP, not through this path.
    ///
    /// What the test still pins is the property in its name — **connectivity
    /// alone never makes anything available.** `.isAvailable` is false in every
    /// one of the sixteen combinations.
    func testNoCartridgeIsAvailableWhetherOrNotTheTowerIsReachable() {
        for cartridge in Cartridge.catalog {
            let thisBuildCanReceiveADeclaration =
                TowerCapabilities.towerCartridgeNames[cartridge.id] != nil
            for reachable in [true, false] {
                let availability = TowerCapabilities.availability(
                    for: cartridge.id,
                    isTowerReachable: reachable
                )
                let expected: CartridgeAvailability =
                    (thisBuildCanReceiveADeclaration && !reachable)
                        ? .towerUnreachable
                        : .noContract
                XCTAssertEqual(
                    availability,
                    expected,
                    "\(cartridge.name) resolved wrongly with reachable=\(reachable)"
                )
                XCTAssertFalse(
                    availability.isAvailable,
                    "\(cartridge.name) claimed availability with reachable=\(reachable)"
                )
                XCTAssertEqual(
                    availability.forcedPhase,
                    expected == .towerUnreachable ? .disconnected : .unsupported
                )
            }
        }
    }

    /// The cold-launch case, stated on its own because it is what a person
    /// actually meets first.
    ///
    /// Before the fix this returned `.noContract` for all four, whose shipped
    /// explanation reads *"That is a statement about what the Tower offers, not
    /// about this connection"* — false in precisely the situation that produced
    /// it, and the first thing anyone sees if the Tower is not up yet.
    func testAColdLaunchWithNoTowerBlamesTheConnectionNotTheTower() {
        for cartridgeID in TowerCapabilities.towerCartridgeNames.keys {
            XCTAssertEqual(
                TowerCapabilities.availability(for: cartridgeID, isTowerReachable: false),
                .towerUnreachable,
                "\(cartridgeID) told a cold-launched app the Tower had never heard of it"
            )
        }
    }

    // MARK: Contract resolution

    /// A Tower speaking a contract this build does not implement must produce an
    /// explicit unsupported state — not a silent empty screen, and not an
    /// attempt to decode it anyway.
    func testAnUnknownContractIsExplicitlyUnsupported() {
        let declared = CartridgeContract(cartridgeID: "world-build", identifier: "v99")
        let availability = CartridgeAvailability.resolve(
            declared: declared,
            supported: [],
            isTowerReachable: true
        )
        XCTAssertEqual(availability, .unsupportedContract(declared: declared))
        XCTAssertFalse(availability.isAvailable)
        // `.needsUpdate`, not `.unsupported`. The Tower *declared* this
        // cartridge — it can do it — and this build is the half that cannot
        // read the agreement. That is the one empty state on this screen a
        // person can end themselves, and "Nothing yet" told them a feature
        // did not exist when they were one update away from it.
        XCTAssertEqual(availability.forcedPhase, .needsUpdate)
        XCTAssertNotEqual(availability.forcedPhase, .unsupported)
    }

    /// The precedence rule. A contract mismatch is not fixed by reconnecting,
    /// so it must outrank an unreachable Tower — otherwise the UI advises a
    /// reconnect that cannot help, forever.
    func testAContractMismatchOutranksAnUnreachableTower() {
        let declared = CartridgeContract(cartridgeID: "world-build", identifier: "v99")
        XCTAssertEqual(
            CartridgeAvailability.resolve(declared: declared, supported: [], isTowerReachable: false),
            .unsupportedContract(declared: declared),
            "an unreachable Tower masked a contract this build cannot speak"
        )
    }

    /// An implemented contract still needs a reachable Tower — and that is a
    /// connection state, not an error.
    func testAnImplementedContractStillNeedsAReachableTower() {
        let declared = CartridgeContract(cartridgeID: "world-build", identifier: "v1")
        XCTAssertEqual(
            CartridgeAvailability.resolve(declared: declared, supported: ["v1"], isTowerReachable: false),
            .towerUnreachable
        )
        XCTAssertEqual(
            CartridgeAvailability.resolve(declared: declared, supported: ["v1"], isTowerReachable: true),
            .available(declared)
        )
    }

    /// Availability stops forcing a phase exactly when the cartridge is usable,
    /// and not before. This is what lets a cartridge's own state show through
    /// only once there is something behind it.
    func testOnlyAnAvailableCartridgeLetsItsOwnStateThrough() {
        let declared = CartridgeContract(cartridgeID: "x", identifier: "v1")
        XCTAssertNil(CartridgeAvailability.available(declared).forcedPhase)
        XCTAssertEqual(CartridgeAvailability.noContract.forcedPhase, .unsupported)
        // `.disconnected`, not `.unsupported`. A Tower that is merely
        // unreachable may well be able to do this, and the shared panel now
        // says so in its headline and glyph rather than only in its prose.
        XCTAssertEqual(CartridgeAvailability.towerUnreachable.forcedPhase, .disconnected)
        // `.needsUpdate`, not `.unsupported`, for the same reason `.disconnected`
        // is not `.unsupported`: opposite responses. Nothing fixes `.noContract`
        // but a Tower change; this one is fixed by updating the app.
        XCTAssertEqual(
            CartridgeAvailability.unsupportedContract(declared: declared).forcedPhase,
            .needsUpdate
        )
    }

    // MARK: Explanations

    /// Every unavailable state has to be sayable, and an available one must not
    /// produce a reassuring sentence a caller might show by accident.
    func testEveryUnavailableStateExplainsItselfAndAnAvailableOneDoesNot() {
        let declared = CartridgeContract(cartridgeID: "x", identifier: "v9")
        let unavailable: [CartridgeAvailability] = [
            .noContract,
            .unsupportedContract(declared: declared),
            .towerUnreachable,
        ]
        for availability in unavailable {
            let explanation = availability.explanation(cartridgeName: "Test Cartridge")
            XCTAssertNotNil(explanation, "\(availability) cannot be explained")
            XCTAssertFalse(explanation?.isEmpty ?? true)
            XCTAssertTrue(
                explanation?.contains("Test Cartridge") ?? false,
                "an explanation that does not name the cartridge is not actionable"
            )
        }
        XCTAssertNil(CartridgeAvailability.available(declared).explanation(cartridgeName: "Test Cartridge"))
    }

    /// The unsupported-contract explanation must name the contract. Without it,
    /// "update the app" is advice with nothing behind it and nothing to report.
    func testAnUnsupportedContractExplanationNamesTheContract() {
        let declared = CartridgeContract(cartridgeID: "x", identifier: "world-v3")
        let explanation = CartridgeAvailability.unsupportedContract(declared: declared)
            .explanation(cartridgeName: "World Builder")
        XCTAssertTrue(explanation?.contains("world-v3") ?? false)
    }

    // MARK: The joined explanation

    /// The two-sentence join, decided once here rather than restated in four
    /// workspaces.
    ///
    /// The ordering matters: the shared sentence about the Tower comes first,
    /// then whatever the cartridge adds about itself. Either may be absent, and
    /// a blank paragraph between them reads as a missing string — which is how
    /// a reader starts to distrust the rest of the panel.
    func testTheSharedAndClientSentencesAreJoinedInOrder() {
        let shared = CartridgeAvailability.noContract
            .explanation(cartridgeName: "World Builder") ?? ""
        XCTAssertFalse(shared.isEmpty)

        let joined = CartridgeAvailability.noContract.explanation(
            cartridgeName: "World Builder",
            clientReason: "And here is the cartridge speaking."
        )
        XCTAssertTrue(joined.hasPrefix(shared), "the shared sentence must come first")
        XCTAssertTrue(joined.hasSuffix("And here is the cartridge speaking."))
        XCTAssertFalse(joined.contains("\n\n\n"), "a blank paragraph opened between the two sentences")
    }

    /// An absent client sentence leaves the shared one intact and adds no
    /// trailing whitespace.
    func testAnAbsentClientSentenceLeavesNoGap() {
        let shared = CartridgeAvailability.noContract
            .explanation(cartridgeName: "World Builder") ?? ""
        for reason in [nil, ""] as [String?] {
            XCTAssertEqual(
                CartridgeAvailability.noContract
                    .explanation(cartridgeName: "World Builder", clientReason: reason),
                shared
            )
        }
    }

    /// An available cartridge has no shared sentence, so the client stands
    /// alone rather than being prefixed by an empty one.
    func testAnAvailableCartridgeShowsOnlyTheClientSentence() {
        let contract = CartridgeContract(cartridgeID: "x", identifier: "v1")
        XCTAssertEqual(
            CartridgeAvailability.available(contract)
                .explanation(cartridgeName: "X", clientReason: "Only this."),
            "Only this."
        )
        XCTAssertEqual(
            CartridgeAvailability.available(contract)
                .explanation(cartridgeName: "X", clientReason: nil),
            ""
        )
    }

    // MARK: Failure wrapping

    /// A cartridge failure passes through unchanged; anything else becomes
    /// `.transport`, which is the only honest attribution available without
    /// knowing where an unrecognised error came from.
    func testWrappingPreservesACartridgeFailureAndClassifiesAnythingElse() {
        let original = CartridgeFailure(kind: .notSupported, message: "no contract")
        XCTAssertEqual(CartridgeFailure.wrapping(original), original)

        struct Other: Error {}
        let wrapped = CartridgeFailure.wrapping(Other())
        XCTAssertEqual(wrapped.kind, .transport)
        XCTAssertFalse(wrapped.message.isEmpty)
    }

    // MARK: Phases

    /// The load-bearing property of `CartridgePhase`, stated once here and
    /// applied to every cartridge's own state in `testNoClientProducesTowerData`.
    func testOnlyLiveAndSettledPhasesMayCarryData() {
        XCTAssertTrue(CartridgePhase.live.mayCarryData)
        XCTAssertTrue(CartridgePhase.settled.mayCarryData)
        for phase in [CartridgePhase.unsupported, .disconnected, .idle, .waiting, .failed] {
            XCTAssertFalse(phase.mayCarryData, "\(phase) is permitted to carry data")
        }
        // Restated over `allCases` so a future phase cannot be added on the
        // permissive side of this line without being noticed.
        for phase in CartridgePhase.allCases where phase != .live && phase != .settled {
            XCTAssertFalse(phase.mayCarryData, "\(phase) is permitted to carry data")
        }
    }

    /// A spinner is only ever honest in one phase.
    ///
    /// Asserted case by case rather than as `showsProgress == (phase ==
    /// .waiting)`. That formula is the implementation verbatim, so it would
    /// have re-derived its own expectation from any change to the definition
    /// and passed regardless — a mirror, not a test.
    func testProgressIsOnlyHonestWhileWaiting() {
        XCTAssertTrue(CartridgePhase.waiting.showsProgress)
        // `.unsupported` matters most: nothing is underway and nothing will be
        // without a Tower change, so a spinner there is the most convincing and
        // least true indicator the app could draw.
        XCTAssertFalse(CartridgePhase.unsupported.showsProgress)
        XCTAssertFalse(CartridgePhase.disconnected.showsProgress)
        XCTAssertFalse(CartridgePhase.idle.showsProgress)
        XCTAssertFalse(CartridgePhase.live.showsProgress)
        XCTAssertFalse(CartridgePhase.settled.showsProgress)
        XCTAssertFalse(CartridgePhase.failed.showsProgress)
        // Decided: nothing is in flight while the app is the thing that is
        // behind, so a spinner would be as untrue here as in `.unsupported`.
        XCTAssertFalse(CartridgePhase.needsUpdate.showsProgress)
        XCTAssertEqual(CartridgePhase.allCases.count, 8, "a phase was added without a decision here")
    }

    /// An unreachable Tower and an absent capability are different situations
    /// with opposite remedies, and must stay different phases.
    func testAnUnreachableTowerIsNotAMissingCapability() {
        XCTAssertNotEqual(CartridgePhase.disconnected, CartridgePhase.unsupported)
        XCTAssertFalse(CartridgePhase.disconnected.mayCarryData)
    }

    /// The third member of the same family. All four empty phases mean "there
    /// is nothing to show", and each one implies a different next move: wait
    /// for the Tower to gain the capability, wait for the network, update the
    /// app, or press the button. Collapsing any two of them hands a person the
    /// wrong instruction.
    func testTheFourEmptyPhasesAreDistinctBecauseTheirRemediesAre() {
        let empty: [CartridgePhase] = [.unsupported, .disconnected, .needsUpdate, .idle]
        XCTAssertEqual(Set(empty).count, empty.count, "two empty phases collapsed into one")
        for phase in empty {
            XCTAssertFalse(phase.mayCarryData, "\(phase) must not carry data")
            XCTAssertFalse(phase.showsProgress, "\(phase) must not claim work is underway")
        }
    }

    // MARK: Failures

    /// A failure with no message renders as a blank panel attached to nothing.
    /// The substitution keeps the guarantee absolute rather than by convention.
    func testAFailureAlwaysCarriesAnExplanation() {
        for kind in CartridgeFailure.Kind.allCases {
            let failure = CartridgeFailure(kind: kind, message: "")
            XCTAssertFalse(failure.message.isEmpty, "\(kind) produced a blank failure")
        }
        XCTAssertEqual(
            CartridgeFailure(kind: .transport, message: "socket closed").message,
            "socket closed",
            "a real message must survive untouched"
        )
    }
}

// MARK: - The four unavailable cartridge clients

/// One table, four cartridges, one invariant: **these four clients produce no
/// Tower data, because the Tower produces none for them.**
///
/// Written as a table rather than four suites so that a cartridge added without
/// a truthful client fails here rather than passing by omission.
///
/// ## Why Object Memory is not in the table
///
/// It is the fifth cartridge and the first whose Tower half genuinely answers:
/// two read-only HTTP routes, serving a real store. Its client therefore
/// *should* be able to reach `.settled` with records in it, which is precisely
/// what `testNoClientProducesTowerData` forbids — the invariant here is "no
/// data from a Tower that produces none", and for Object Memory the premise is
/// false.
///
/// So it is covered by `ObjectMemoryTests` instead, which asserts the stronger
/// property that actually applies to it: that what it produces is decoded from
/// what the Tower sent, and that nothing it says about a record overclaims.
/// `testEveryOpenableCartridgeHasAClient` below still includes it, because
/// "every screen has a client" is true of all five.
@MainActor
final class CartridgeClientTests: XCTestCase {

    /// Every client, paired with the catalog id it must answer for and whether
    /// it is exposing any Tower data at all.
    private struct ClientCase {
        let cartridgeID: String
        let phase: CartridgePhase
        let reason: String
        let hasData: Bool
    }

    private func allClients() -> [ClientCase] {
        let world = UnavailableWorldBuilderClient()
        let cv = UnavailableExperimentalCVClient()
        let documents = UnavailableDocumentMemoryClient()
        let scene = UnavailableSceneUnderstandingClient()

        return [
            ClientCase(
                cartridgeID: world.cartridgeID,
                phase: world.state.phase,
                reason: UnavailableWorldBuilderClient.reason,
                hasData: world.state.snapshot != nil || world.state.hasWorld
            ),
            ClientCase(
                cartridgeID: cv.cartridgeID,
                phase: cv.state.phase,
                reason: UnavailableExperimentalCVClient.reason,
                hasData: cv.state.run != nil
            ),
            ClientCase(
                cartridgeID: documents.cartridgeID,
                phase: documents.state.phase,
                reason: UnavailableDocumentMemoryClient.reason,
                hasData: documents.state.result != nil
            ),
            ClientCase(
                cartridgeID: scene.cartridgeID,
                phase: scene.state.phase,
                reason: UnavailableSceneUnderstandingClient.reason,
                // `observation`, not the old `snapshot`. `SceneSnapshot` was
                // removed with the entity/relationship types it held: the wire
                // has no key that could carry an entity, so a type shaped like
                // a list of them could never be filled from a real payload.
                hasData: scene.state.observation != nil
            ),
        ]
    }

    /// The single most important assertion in this file.
    func testNoClientProducesTowerData() {
        let cases = allClients()
        XCTAssertEqual(cases.count, 4, "a cartridge client was added without being covered here")
        for client in cases {
            XCTAssertEqual(
                client.phase,
                .unsupported,
                "\(client.cartridgeID) is not in the unsupported phase"
            )
            XCTAssertFalse(
                client.phase.mayCarryData,
                "\(client.cartridgeID)'s phase permits data it cannot have"
            )
            XCTAssertFalse(
                client.hasData,
                "\(client.cartridgeID) produced Tower data from a Tower that produces none"
            )
        }
    }

    /// An unsupported state that does not explain itself becomes a blank screen
    /// the user reads as a bug.
    func testEveryClientExplainsWhyItHasNothing() {
        for client in allClients() {
            XCTAssertFalse(client.reason.isEmpty, "\(client.cartridgeID) gives no reason")
        }
    }

    /// **No client may make a claim about what the Tower stores.**
    ///
    /// This app has no channel through which it could know that. It sends
    /// frames over a transport `07-PLATFORM-CONSTRAINTS.md` Limitation 11
    /// describes as unauthenticated and unencrypted, and the protocol says
    /// nothing about storage in either direction — so "the Tower keeps none of
    /// this" is not a conservative guess, it is a fabrication (Rule 3), and it
    /// is worst on Scene Understanding, whose subject is bystanders.
    ///
    /// A length check would not have caught it. The offending sentence was
    /// long, fluent, and wrong.
    func testNoClientClaimsToKnowWhatTheTowerStores() {
        let forbidden = [
            "stores nothing",
            "store nothing",
            "keeps nothing",
            "is not stored",
            "are not stored",
            "never stored",
            "nothing is stored",
        ]
        for client in allClients() {
            let reason = client.reason.lowercased()
            for phrase in forbidden {
                XCTAssertFalse(
                    reason.contains(phrase),
                    "\(client.cartridgeID) claims to know what the Tower stores: \"\(phrase)\""
                )
            }
        }
    }

    /// Cartridge isolation, as far as a runtime test can reach it: a client
    /// answers for exactly one cartridge, that cartridge is in the catalog, and
    /// no two clients answer for the same one.
    func testEachClientAnswersForExactlyItsOwnCartridge() {
        let ids = allClients().map(\.cartridgeID)
        XCTAssertEqual(ids.count, Set(ids).count, "two clients claim the same cartridge")

        let catalogIDs = Set(Cartridge.catalog.map(\.id))
        for id in ids {
            XCTAssertTrue(catalogIDs.contains(id), "\(id) is not a cartridge in the catalog")
        }

        // Stated exactly, so a copy-paste that leaves the wrong id in a new
        // client is caught rather than merely being "some catalog id".
        XCTAssertEqual(
            Set(ids),
            ["world-build", "experimental-cv", "document-memory", "scene-understanding"]
        )
    }

    /// Every workspace-bearing cartridge has a client. A screen with no client
    /// would have to invent its own state, which is where fabricated data
    /// enters an app.
    func testEveryOpenableCartridgeHasAClient() {
        // The four unavailable clients, plus Object Memory's — which is not in
        // `allClients()` because the invariant that table asserts does not
        // apply to it. Read off `CartridgeClients` rather than hardcoded, so a
        // sixth cartridge cannot be satisfied by a string in this file.
        let clients = CartridgeClients()
        var clientIDs = Set(allClients().map(\.cartridgeID))
        clientIDs.insert(clients.objectMemory.cartridgeID)

        for cartridge in Cartridge.selectable {
            XCTAssertTrue(
                clientIDs.contains(cartridge.id),
                "\(cartridge.name) has a workspace but no client"
            )
        }
    }

    // MARK: Requests are refused, not swallowed

    /// A silent no-op leaves a control that appears to work. Both cartridges
    /// that accept a request must refuse it audibly — docs/04-MODULE-SYSTEM.md
    /// requires an unsupported request to "produce a clear degraded/failed
    /// state rather than silently pretending" it applied.
    func testRunningAnExperimentIsRefusedRatherThanIgnored() {
        let client = UnavailableExperimentalCVClient()
        XCTAssertThrowsError(
            try client.run(CVExperiment(id: "any", name: "Any"))
        ) { error in
            guard let failure = error as? CartridgeFailure else {
                return XCTFail("expected a CartridgeFailure, got \(error)")
            }
            XCTAssertFalse(failure.message.isEmpty)
            // Not `.towerReportedFailure`. The Tower reported nothing — there
            // may not even be a socket open — and attributing a local refusal
            // to the other machine is a fabricated claim about it that a later
            // log or telemetry consumer would read back as fact.
            XCTAssertEqual(failure.kind, .notSupported)
        }
    }

    func testSearchingDocumentsIsRefusedRatherThanReturningEmpty() {
        let client = UnavailableDocumentMemoryClient()
        XCTAssertThrowsError(
            try client.search(.recent(limit: 10), origin: .appText)
        ) { error in
            guard let failure = error as? CartridgeFailure else {
                return XCTFail("expected a CartridgeFailure, got \(error)")
            }
            // The distinction this protects: an empty result would read as
            // "you have no documents", which is a false statement about the
            // user's own memory rather than about the Tower.
            XCTAssertFalse(failure.message.isEmpty)
            XCTAssertEqual(failure.kind, .notSupported)
        }
    }

    /// The Experimental CV client declares no experiments. A populated picker —
    /// even of plausible ones from the module spec's candidate list — would be
    /// the app asserting those experiments exist.
    func testNoExperimentsAreDeclared() {
        // Through the view model, which is the only thing that ever reads the
        // list. Pattern-matching the client's `let` state for `.idle` could
        // never match and made the test unable to fail for the reason it names.
        let lab = ExperimentalCVViewModel(client: UnavailableExperimentalCVClient())
        XCTAssertTrue(
            lab.availableExperiments.isEmpty,
            "the app is offering experiments the Tower has not declared"
        )
        XCTAssertNil(lab.state.run)
        XCTAssertEqual(lab.state.phase, .unsupported)
    }
}

// MARK: - View models

/// Guards what the four workspace view models may own and may do.
///
/// The runtime-ownership half of this — that constructing them opens no socket
/// and disturbs no live stream — is in `TowerClientTests`, where a real socket
/// and a mock server are available to prove it against.
@MainActor
final class CartridgeViewModelTests: XCTestCase {

    /// With no declaration, the phase now depends on whether the Tower is
    /// reachable — and that difference is the point, not an inconsistency.
    ///
    /// `.unsupported` says "this Tower will never do this"; `.disconnected`
    /// says "ask again when connected". All four of these cartridges have a
    /// client and a Tower-name mapping, so on a cold launch the second is the
    /// true one. Asserting `.unsupported` in both states — which this test did
    /// — is what let a disconnected app tell a person the Tower had never heard
    /// of four cartridges it serves.
    func testEveryViewModelSeparatesAnAbsentContractFromAnAbsentConnection() {
        for (name, phaseFor) in Self.viewModelPhases {
            XCTAssertEqual(
                phaseFor(true), .unsupported,
                "\(name) with a reachable Tower and no declaration is not 'unsupported'"
            )
            XCTAssertEqual(
                phaseFor(false), .disconnected,
                "\(name) blamed the Tower for what is a missing connection"
            )
        }
    }

    private static let viewModelPhases: [(String, (Bool) -> CartridgePhase)] = [
        ("World Builder", { WorldBuilderViewModel(client: UnavailableWorldBuilderClient()).phase(isTowerReachable: $0) }),
        ("Experimental CV Lab", { ExperimentalCVViewModel(client: UnavailableExperimentalCVClient()).phase(isTowerReachable: $0) }),
        ("Document Memory", { DocumentMemoryViewModel(client: UnavailableDocumentMemoryClient()).phase(isTowerReachable: $0) }),
        ("Scene Understanding", { SceneUnderstandingViewModel(client: UnavailableSceneUnderstandingClient()).phase(isTowerReachable: $0) })
    ]

    /// Availability must outrank the client's own state, or a cartridge whose
    /// Tower cannot serve it would render `.idle` and invite a user to start
    /// something that cannot run.
    func testAvailabilityOutranksTheClientState() {
        let lab = ExperimentalCVViewModel(client: IdleExperimentalCVClient())
        XCTAssertEqual(lab.state.phase, .idle, "the double must actually report idle")
        XCTAssertEqual(
            lab.phase(isTowerReachable: true),
            .unsupported,
            "an idle client showed through a Tower that declares no contract"
        )
    }

    /// A refused request has to reach the screen. Swallowing it is how a button
    /// comes to look like it works.
    func testARefusedExperimentIsSurfacedRatherThanSwallowed() {
        let lab = ExperimentalCVViewModel(client: UnavailableExperimentalCVClient())
        XCTAssertNil(lab.lastRequestFailure)
        lab.run(CVExperiment(id: "any", name: "Any"))
        XCTAssertNotNil(lab.lastRequestFailure, "a refusal vanished")
        XCTAssertFalse(lab.lastRequestFailure?.message.isEmpty ?? true)
        // The refusal must not invent a state transition either.
        XCTAssertEqual(lab.state.phase, .unsupported)
    }

    func testARefusedSearchIsSurfacedRatherThanSwallowed() {
        let memory = DocumentMemoryViewModel(client: UnavailableDocumentMemoryClient())
        memory.queryText = "the parking notice"
        memory.submitTypedQuery()
        XCTAssertNotNil(memory.lastRequestFailure)
        XCTAssertEqual(memory.state.phase, .unsupported)
    }

    /// Whitespace is not a question. Submitting it would produce a "nothing
    /// found" answer to a query nobody asked.
    func testAnEmptyOrBlankQueryIsNotSubmitted() {
        let client = RecordingDocumentMemoryClient()
        let memory = DocumentMemoryViewModel(client: client)

        memory.queryText = ""
        memory.submitTypedQuery()
        memory.queryText = "   \n  "
        memory.submitTypedQuery()

        XCTAssertTrue(client.searches.isEmpty, "a blank query was submitted")
    }

    /// Typed text becomes a semantic query from the app, and the origin is
    /// carried rather than assumed — that parameter is the seam a Siri intent
    /// or wake-word layer would submit through without this cartridge growing a
    /// dependency on speech.
    func testTypedTextBecomesASemanticQueryTaggedAsAppInput() {
        let client = RecordingDocumentMemoryClient()
        let memory = DocumentMemoryViewModel(client: client)

        memory.queryText = "  the parking notice  "
        memory.submitTypedQuery()

        XCTAssertEqual(client.searches.count, 1)
        XCTAssertEqual(client.searches.first?.query, .semantic("the parking notice"))
        XCTAssertEqual(client.searches.first?.origin, .appText)
    }

    /// An input layer that is not this app can submit the same query type. No
    /// such layer exists; this asserts the seam is usable without one.
    func testAQueryCanArriveFromOutsideTheApp() {
        let client = RecordingDocumentMemoryClient()
        let memory = DocumentMemoryViewModel(client: client)

        memory.submit(.recent(limit: 5), origin: .externalIntent)

        XCTAssertEqual(client.searches.first?.origin, .externalIntent)
        XCTAssertEqual(client.searches.first?.query, .recent(limit: 5))
        XCTAssertTrue(memory.queryText.isEmpty, "an external query must not need the text field")
    }

    /// The available-experiment list is empty in every state but `.idle`.
    func testAvailableExperimentsAreOnlyListedWhenSomethingCanBeRun() {
        XCTAssertTrue(ExperimentalCVViewModel(client: UnavailableExperimentalCVClient()).availableExperiments.isEmpty)
        let idle = ExperimentalCVViewModel(client: IdleExperimentalCVClient())
        XCTAssertEqual(idle.availableExperiments.map(\.id), ["fixture"])
    }
}

// MARK: Test doubles

/// Reports `.idle` so tests can prove availability outranks a client's state.
/// Lives in the test target, where it can never reach a device.
@MainActor
private final class IdleExperimentalCVClient: ExperimentalCVClient {
    let cartridgeID = "experimental-cv"
    let state: ExperimentalCVState = .idle(
        available: [CVExperiment(id: "fixture", name: "Fixture")]
    )
    func run(_ experiment: CVExperiment) throws {}
}

/// Records what it was asked, so query routing can be asserted without a Tower.
@MainActor
private final class RecordingDocumentMemoryClient: DocumentMemoryClient {
    struct Search: Equatable {
        let query: DocumentQuery
        let origin: DocumentQueryOrigin
    }

    let cartridgeID = "document-memory"
    let state: DocumentMemoryState = .idle
    private(set) var searches: [Search] = []

    func search(_ query: DocumentQuery, origin: DocumentQueryOrigin) throws {
        searches.append(Search(query: query, origin: origin))
    }
}

// MARK: - Privacy: redaction and artifacts

/// Guards the last link in the privacy pipeline:
///
/// ```text
/// raw sensor data → ephemeral perception → derived structured state
///                 → redaction → persistence / display
/// ```
///
/// iOS applies no redaction and cannot. What it can do — and what these pin —
/// is refuse to display anything whose producer did not state that it was
/// treated, and say which is which.
@MainActor
final class ArtifactRedactionTests: XCTestCase {

    /// The single decision this whole area exists to make.
    func testOnlyRedactedImageryMayBeShownOnAPersistedSurface() {
        XCTAssertTrue(RedactionState.redacted.isDisplayableWhenPersisted)
        XCTAssertFalse(RedactionState.rawEphemeral.isDisplayableWhenPersisted)
        XCTAssertFalse(RedactionState.unknown.isDisplayableWhenPersisted)
    }

    /// An unstated treatment is not a treatment. `.unknown` must be handled
    /// exactly as strictly as raw — a lenient default here is how untreated
    /// imagery reaches a screen.
    func testAnUnstatedTreatmentIsAsStrictAsRaw() {
        XCTAssertEqual(
            RedactionState.unknown.isDisplayableWhenPersisted,
            RedactionState.rawEphemeral.isDisplayableWhenPersisted
        )
        XCTAssertFalse(VisualArtifactState.available(.unknown).isDisplayable)
        XCTAssertFalse(VisualArtifactState.available(.rawEphemeral).isDisplayable)
        XCTAssertTrue(VisualArtifactState.available(.redacted).isDisplayable)
    }

    /// Arriving is not the same as being showable. Both conditions are
    /// required, and a test that only covered the redacted case would not have
    /// noticed if the fetch state were dropped from the check.
    func testAnArtifactMustBothArriveAndBeRedacted() {
        XCTAssertFalse(VisualArtifactState.notFetched(.redacted).isDisplayable)
        XCTAssertFalse(VisualArtifactState.fetching(.redacted).isDisplayable)
        XCTAssertFalse(VisualArtifactState.absent.isDisplayable)
        XCTAssertFalse(
            VisualArtifactState.failed(CartridgeFailure(kind: .transport, message: "x")).isDisplayable
        )
    }

    /// A withheld image must say why, or it renders as a blank square the user
    /// reads as a bug rather than as a deliberate refusal.
    func testWithheldImageryExplainsItself() {
        XCTAssertNotNil(VisualArtifactState.available(.rawEphemeral).withheldReason)
        XCTAssertNotNil(VisualArtifactState.available(.unknown).withheldReason)
        XCTAssertNotNil(
            VisualArtifactState.failed(CartridgeFailure(kind: .transport, message: "dropped")).withheldReason
        )
        // Nothing to explain in the two cases where nothing is being withheld.
        XCTAssertNil(VisualArtifactState.available(.redacted).withheldReason)
        XCTAssertNil(VisualArtifactState.absent.withheldReason)
    }

    /// The redaction state has to survive the fetch. Losing it between "arrived"
    /// and "rendered" is exactly how an untreated image gets drawn.
    func testRedactionTravelsWithTheArtifactThroughEveryState() {
        XCTAssertEqual(VisualArtifactState.notFetched(.redacted).redaction, .redacted)
        XCTAssertEqual(VisualArtifactState.fetching(.unknown).redaction, .unknown)
        XCTAssertEqual(VisualArtifactState.available(.rawEphemeral).redaction, .rawEphemeral)
        XCTAssertNil(VisualArtifactState.absent.redaction)
    }

    /// Every treatment must be sayable.
    func testEveryRedactionStateCanBeDescribed() {
        for state in RedactionState.allCases {
            XCTAssertFalse(state.explanation.isEmpty, "\(state) has no explanation")
        }
    }

    /// The redacted explanation must attribute the claim to whoever made it.
    ///
    /// It read "People in this image were obscured before it was stored" — a
    /// specific, checkable privacy guarantee about a step **no Tower contract
    /// defines**. Redaction could be face masking, text masking, or the
    /// producer's own definition of the word. Stating what was removed turns
    /// an opaque flag into exactly the assurance the Tower cannot honour.
    func testTheRedactedExplanationAttributesItsClaimRatherThanDescribingIt() {
        let explanation = RedactionState.redacted.explanation
        XCTAssertTrue(
            explanation.contains("producer"),
            "the redaction claim is stated as fact rather than attributed: \(explanation)"
        )
        XCTAssertFalse(
            explanation.lowercased().contains("people in this image"),
            "the app described what redaction removed, which no contract defines"
        )
    }
}

// MARK: - Provenance and time

/// Guards Core Principles 2, 4 and 5 from `07-PLATFORM-CONSTRAINTS.md` at the
/// point they are most easily lost: display.
@MainActor
final class ObservationProvenanceTests: XCTestCase {

    /// A measurement carries no model confidence, and a model output carries one
    /// or explicitly carries none — which is a worse state than a low
    /// confidence, not a better one.
    func testConfidenceBelongsOnlyToInference() {
        XCTAssertNil(ObservationProvenance.measured.confidence)
        XCTAssertNil(ObservationProvenance.unknown.confidence)
        XCTAssertEqual(ObservationProvenance.inferred(confidence: 0.8).confidence, 0.8)
        XCTAssertNil(ObservationProvenance.inferred(confidence: nil).confidence)
    }

    /// An inference owes the reader a caveat; a measurement owes none. A caveat
    /// on everything is noise, and noise gets ignored.
    func testOnlyUncertainValuesCarryACaveat() {
        XCTAssertNil(ObservationProvenance.measured.caveat)
        XCTAssertNotNil(ObservationProvenance.inferred(confidence: 0.9).caveat)
        XCTAssertNotNil(ObservationProvenance.unknown.caveat)
    }

    /// An inference with no reported confidence must not read as certainty, and
    /// must not read as zero either — neither is what the Tower said.
    func testAnInferenceWithoutAConfidenceSaysSo() {
        let caveat = ObservationProvenance.inferred(confidence: nil).caveat ?? ""
        XCTAssertTrue(caveat.contains("did not report"), "got: \(caveat)")
        XCTAssertFalse(caveat.contains("0%"), "a missing confidence rendered as zero")
    }

    func testOnlyInferenceIsMarkedAsInference() {
        XCTAssertTrue(ObservationProvenance.inferred(confidence: nil).isInference)
        XCTAssertFalse(ObservationProvenance.measured.isInference)
        XCTAssertFalse(ObservationProvenance.unknown.isInference)
    }

    /// A malformed confidence from the Tower must not render as "170%". Clamping
    /// is display hygiene, not a fix — the decode site is where an out-of-range
    /// value should be noticed.
    func testConfidenceIsClampedForDisplay() {
        XCTAssertEqual(ObservationProvenance.percent(1.7), "100%")
        XCTAssertEqual(ObservationProvenance.percent(-0.5), "0%")
        XCTAssertEqual(ObservationProvenance.percent(0.5), "50%")
    }
}

/// The rule that "metric" is not "metres", as a test.
///
/// `ReportedFigure` was extracted because two screens were rendering
/// `String(format: "%.1f m", …)` from a bare `Double` — inventing the unit —
/// while `CVMetric.unit` two files away had already taken the correct position
/// ("never assumed"). It is the smallest shared helper in the change and it was
/// the only one with no coverage, which is a poor combination for something
/// whose entire job is a display rule.
@MainActor
final class ReportedFigureTests: XCTestCase {

    /// A figure with no unit renders bare. That is not a degraded rendering —
    /// it is the honest one, because an unlabelled quantity is what the Tower
    /// sent.
    func testAFigureWithNoUnitRendersBare() {
        XCTAssertEqual(ReportedFigure.format(14.2, unit: nil), "14.2")
        XCTAssertEqual(ReportedFigure.format(3, unit: nil), "3")
    }

    /// An empty unit string is treated as no unit, not as a trailing space.
    func testAnEmptyUnitIsTreatedAsNoUnit() {
        XCTAssertEqual(ReportedFigure.format(14.2, unit: ""), "14.2")
    }

    /// The Tower's unit is used verbatim — never translated, never normalised.
    /// iOS does not know that "cm" and "m" are related, and must not act as
    /// though it does.
    func testTheTowerUnitIsUsedVerbatim() {
        XCTAssertEqual(ReportedFigure.format(14.2, unit: "m"), "14.2 m")
        XCTAssertEqual(ReportedFigure.format(1420, unit: "cm"), "1420 cm")
        XCTAssertEqual(ReportedFigure.format(7, unit: "keyframes"), "7 keyframes")
    }

    /// A whole number renders without a decimal point, so a count does not read
    /// as a measurement.
    func testWholeNumbersDoNotGrowADecimalPoint() {
        XCTAssertEqual(ReportedFigure.format(40, unit: nil), "40")
        XCTAssertEqual(ReportedFigure.format(-3, unit: nil), "-3")
        XCTAssertEqual(ReportedFigure.format(0, unit: nil), "0")
    }

    /// **The regression this exists to prevent.** No output may contain a unit
    /// the caller did not supply — in particular not the metres two screens
    /// used to print from a scale that only ever claimed to be metric *in
    /// kind*.
    func testNoUnitIsEverInvented() {
        for value in [0.0, 1.0, 14.2, -7.5, 1_000.0] {
            let rendered = ReportedFigure.format(value, unit: nil)
            for invented in ["m", "cm", "metre", "meter", "ft"] {
                XCTAssertFalse(
                    rendered.contains(invented),
                    "\(rendered) carries a unit nobody supplied"
                )
            }
        }
    }
}

@MainActor
final class ObservationTimeTests: XCTestCase {

    /// Core Principle 5. Arrival time must never be promoted to observation
    /// time, because a view labelling one as the other is the whole failure the
    /// principle names.
    func testArrivalTimeIsNeverShownAsObservationTime() {
        let arrivalOnly = ObservationTime(observedAt: nil, receivedAt: Date())
        XCTAssertNil(arrivalOnly.displayableObservationTime)
        XCTAssertTrue(arrivalOnly.isObservationTimeUnknown)
    }

    func testAReportedObservationTimeIsUsed() {
        let observed = Date(timeIntervalSince1970: 1_000)
        let time = ObservationTime(observedAt: observed, receivedAt: Date())
        XCTAssertEqual(time.displayableObservationTime, observed)
        XCTAssertFalse(time.isObservationTimeUnknown)
    }

    /// A default carries neither, which is the honest starting point.
    func testADefaultTimeClaimsNothing() {
        let time = ObservationTime()
        XCTAssertNil(time.displayableObservationTime)
        XCTAssertNil(time.receivedAt)
        XCTAssertTrue(time.isObservationTimeUnknown)
    }
}

/// Limitation 8 — camera FOV is not attention — enforced as a naming rule.
@MainActor
final class ObservedDurationTests: XCTestCase {

    /// The label must describe the camera, not the wearer. "Viewed", "read" and
    /// "looked" are all claims the platform cannot support at any confidence.
    func testTheLabelDescribesTheCameraAndNotTheWearer() {
        let labels = [
            ObservedDuration(seconds: 4).label,
            ObservedDuration(seconds: 65).label,
            ObservedDuration(seconds: 120).label,
        ]
        for label in labels {
            XCTAssertTrue(label.hasPrefix("In view"), "got: \(label)")
            let lowered = label.lowercased()
            for forbidden in ["viewed", "read", "looked", "watched", "seen"] {
                XCTAssertFalse(lowered.contains(forbidden), "\(label) claims attention")
            }
        }
    }

    func testDurationsAreFormattedForPeople() {
        XCTAssertEqual(ObservedDuration(seconds: 4).label, "In view 4s")
        XCTAssertEqual(ObservedDuration(seconds: 120).label, "In view 2m")
        XCTAssertEqual(ObservedDuration(seconds: 65).label, "In view 1m 5s")
    }

    /// A negative duration is meaningless and would render as "In view -3s".
    func testANegativeDurationIsFloored() {
        XCTAssertEqual(ObservedDuration(seconds: -3).seconds, 0)
    }

    /// The caveat must say what the glasses cannot do, not merely soften what
    /// they can.
    func testTheCaveatDeniesAttentionOutright() {
        let caveat = ObservedDuration.attentionCaveat.lowercased()
        XCTAssertTrue(caveat.contains("cannot tell"), "got: \(ObservedDuration.attentionCaveat)")
    }
}

// MARK: - World Builder additions

/// The fields Product Shell V2 did not have: calibration, geometry, trajectory,
/// persistence, and the `.finalizing` state.
@MainActor
final class WorldModelIntegrationTests: XCTestCase {

    // MARK: Finalizing

    /// `.finalizing` exists because the Tower working is not the same as the
    /// Tower observing. A live badge while the camera is off is a lie about the
    /// sensor, whatever the compute is doing.
    func testFinalizingIsWorkButNotLiveObservation() {
        let state = WorldModelState.finalizing(WorldSnapshot(keyframeCount: 9))
        XCTAssertFalse(state.isReceivingUpdates, "finalizing claimed live observation")
        XCTAssertTrue(state.hasWorld)
        XCTAssertNotNil(state.snapshot)
        XCTAssertEqual(state.phase, .live, "finalizing is the Tower working")
    }

    /// Every state's coarse phase, pinned. A drift here would let a shared
    /// panel draw a spinner over a world that is finished, or a live badge over
    /// one that is not being observed.
    func testEveryWorldStateMapsToTheRightPhase() {
        let snapshot = WorldSnapshot(keyframeCount: 1)
        let expected: [(WorldModelState, CartridgePhase)] = [
            (.unsupported(reason: "x"), .unsupported),
            (.idle, .idle),
            (.awaitingFirstUpdate, .waiting),
            (.receiving(snapshot), .live),
            (.finalizing(snapshot), .live),
            (.finalizing(snapshot, buildInProgress: true), .live),
            (.finalized(snapshot), .settled),
            // Settled, not failed: nothing more will arrive, and there is
            // data to draw — which `.failed` may not carry.
            (.interrupted(snapshot, reason: "the builder exited"), .settled),
            (.failed(CartridgeFailure(kind: .transport, message: "x")), .failed),
        ]
        for (state, phase) in expected {
            XCTAssertEqual(state.phase, phase, "\(state) mapped to the wrong phase")
        }
    }

    /// Restated for the states that must never carry data — the invariant the
    /// shared `CartridgePhase` exists to make checkable.
    func testStatesWithoutAWorldStillCarryNoData() {
        let empty: [WorldModelState] = [
            .unsupported(reason: "x"), .idle, .awaitingFirstUpdate,
            .failed(CartridgeFailure(kind: .transport, message: "x")),
        ]
        for state in empty {
            XCTAssertFalse(state.phase.mayCarryData)
            XCTAssertNil(state.snapshot)
            XCTAssertFalse(state.hasWorld)
        }
    }

    // MARK: Calibration and metric display

    /// Both gates, together. An uncalibrated figure is not a rough figure, it is
    /// an unanchored one; and a relative figure is not a distance at all.
    func testAMetricFigureNeedsBothCalibrationAndMetricScale() {
        XCTAssertTrue(
            WorldSnapshot(scale: .inferredMetric, calibration: .calibrated).permitsMetricDisplay
        )
        XCTAssertFalse(
            WorldSnapshot(scale: .inferredMetric, calibration: .calibrating).permitsMetricDisplay,
            "an uncalibrated pipeline produced a displayable distance"
        )
        XCTAssertFalse(
            WorldSnapshot(scale: .relative, calibration: .calibrated).permitsMetricDisplay,
            "a relative figure was treated as a distance"
        )
        XCTAssertFalse(
            WorldSnapshot(scale: .unknown, calibration: .calibrated).permitsMetricDisplay
        )
    }

    /// Only a calibrated pipeline permits spatial figures, and "not reported" is
    /// not a permission.
    func testOnlyCalibratedPermitsSpatialFigures() {
        XCTAssertTrue(WorldCalibrationState.calibrated.permitsSpatialFigures)
        for state in [WorldCalibrationState.unknown, .uncalibrated, .calibrating] {
            XCTAssertFalse(state.permitsSpatialFigures, "\(state) permitted a spatial figure")
        }
    }

    /// Silence and a report are different claims and must read differently.
    func testUnknownCalibrationIsNotDressedUpAsUncalibrated() {
        XCTAssertNotEqual(
            WorldCalibrationState.unknown.displayName,
            WorldCalibrationState.uncalibrated.displayName
        )
        for state in WorldCalibrationState.allCases {
            XCTAssertFalse(state.displayName.isEmpty)
        }
    }

    // MARK: Geometry

    /// The representation is opaque: whatever the Tower calls it survives
    /// untouched. Nothing in this app matches on it, so nothing can drop a name
    /// it does not recognise.
    func testAnyGeometryRepresentationSurvivesVerbatim() {
        for name in ["point_cloud", "pose_graph", "gaussian splats", "something-new"] {
            let report = WorldGeometryReport(representation: name, elementCount: 12)
            XCTAssertEqual(report.representation, name)
            XCTAssertTrue(report.hasReport)
        }
    }

    /// A count alone is still a report — it just cannot be labelled, and the
    /// view is what decides how to present it.
    func testGeometryWithOnlyACountIsStillAReport() {
        XCTAssertTrue(WorldGeometryReport(elementCount: 5).hasReport)
        XCTAssertTrue(WorldGeometryReport(representation: "mesh").hasReport)
        XCTAssertFalse(WorldGeometryReport().hasReport)
        XCTAssertFalse(
            WorldGeometryReport(isIncremental: true).hasReport,
            "an update mode is not geometry"
        )
    }

    // MARK: Trajectory

    /// A path length under relative scale is a number in arbitrary units.
    /// Printing it as metres would be the fabrication WORLD-BUILD.md names.
    func testPathLengthIsWithheldUnlessTheScaleClaimsMetres() {
        XCTAssertTrue(
            WorldTrajectoryReport(pathLength: 14.2, scale: .inferredMetric).distanceDisplayable
        )
        XCTAssertTrue(
            WorldTrajectoryReport(pathLength: 14.2, scale: .measuredMetric).distanceDisplayable
        )
        XCTAssertFalse(
            WorldTrajectoryReport(pathLength: 14.2, scale: .relative).distanceDisplayable,
            "a relative path length was offered as a distance"
        )
        XCTAssertFalse(
            WorldTrajectoryReport(pathLength: 14.2, scale: .unknown).distanceDisplayable
        )
        XCTAssertFalse(
            WorldTrajectoryReport(poseCount: 40, scale: .inferredMetric).distanceDisplayable,
            "a distance was claimed with no length reported"
        )
    }

    func testAPoseCountIsAReportWithoutADistance() {
        XCTAssertTrue(WorldTrajectoryReport(poseCount: 40).hasReport)
        XCTAssertFalse(WorldTrajectoryReport().hasReport)
    }

    // MARK: Emptiness, extended

    /// The new fields must participate in emptiness, or a snapshot carrying only
    /// a geometry report would claim to be empty and be omitted from the UI.
    func testEmptinessAccountsForEveryNewField() {
        XCTAssertTrue(WorldSnapshot().isEmpty)
        XCTAssertFalse(WorldSnapshot(calibration: .calibrating).isEmpty)
        XCTAssertFalse(WorldSnapshot(geometry: WorldGeometryReport(elementCount: 0)).isEmpty)
        XCTAssertFalse(WorldSnapshot(trajectory: WorldTrajectoryReport(poseCount: 0)).isEmpty)
        XCTAssertFalse(WorldSnapshot(persistence: .session).isEmpty)
    }

    /// Silence about storage is not a promise that the world is discarded.
    func testUnknownPersistenceIsNotSessionOnly() {
        XCTAssertNotEqual(
            WorldPersistenceState.unknown.displayName,
            WorldPersistenceState.session.displayName
        )
    }

    /// Inspecting a stored world is a different mode from watching one being
    /// built, and the identifier travels with it so the claim stays checkable.
    func testInspectionModeDistinguishesAStoredWorld() {
        XCTAssertFalse(WorldInspectionMode.live.isInspecting)
        XCTAssertTrue(WorldInspectionMode.inspecting(worldID: "abc").isInspecting)
        XCTAssertNotEqual(
            WorldInspectionMode.inspecting(worldID: "abc"),
            WorldInspectionMode.inspecting(worldID: "def")
        )
    }
}

// MARK: - Experimental CV Lab

/// Guards the rule `docs/modules/EXPERIMENTAL-CV.md` puts on results:
/// inference must be distinguishable from measurement.
///
/// The other rule it used to guard — nothing may be called "better" without a
/// baseline to be better than — is no longer testable here, because there is
/// no longer any code that could break it. `CVMetric` carries no `baseline`,
/// no `higherIsBetter` and no `comparison`: the Tower sends those fields as
/// `null` on every metric, always, and the machinery was deleted rather than
/// left dormant behind them. The four tests that lived here asserted the
/// behaviour of a verdict renderer that no longer exists.
@MainActor
final class ExperimentalCVModelTests: XCTestCase {

    /// A unit the Tower did not send is omitted, not substituted.
    func testAMissingUnitIsOmittedRatherThanInvented() {
        let bare = CVMetric(label: "count", value: 12, provenance: .measured)
        XCTAssertEqual(bare.displayValue, "12")
        let withUnit = CVMetric(label: "latency", value: 12, unit: "ms", provenance: .measured)
        XCTAssertEqual(withUnit.displayValue, "12 ms")
    }

    /// The run-level caveat has to fire whenever any single metric is an
    /// inference — one measured metric in the set must not suppress it.
    func testARunKnowsWhenAnyOfItsMetricsIsAnInference() {
        let experiment = CVExperiment(id: "x", name: "X")
        let measured = CVMetric(label: "frames", value: 30, provenance: .measured)
        let inferred = CVMetric(label: "depth", value: 1.2, provenance: .inferred(confidence: 0.6))

        XCTAssertFalse(CVExperimentRun(experiment: experiment, metrics: [measured]).containsInference)
        XCTAssertTrue(
            CVExperimentRun(experiment: experiment, metrics: [measured, inferred]).containsInference
        )
        XCTAssertFalse(CVExperimentRun(experiment: experiment).containsInference)
    }

    /// Zero annotations is a result — "found nothing" — and must not merge with
    /// "did not say".
    func testZeroAnnotationsIsAReportAndSilenceIsNot() {
        XCTAssertTrue(CVAnnotationReport(count: 0).hasReport)
        XCTAssertFalse(CVAnnotationReport().hasReport)
        XCTAssertTrue(CVAnnotationReport(artifact: .available(.redacted)).hasReport)
    }

    /// An annotated frame gets no privacy exemption for being a debug surface.
    func testAnUnredactedAnnotatedFrameIsWithheld() {
        let report = CVAnnotationReport(count: 3, artifact: .available(.rawEphemeral))
        XCTAssertFalse(report.artifact.isDisplayable)
        XCTAssertNotNil(report.artifact.withheldReason)
    }

    func testEveryCVStateMapsToTheRightPhase() {
        let experiment = CVExperiment(id: "x", name: "X")
        let run = CVExperimentRun(experiment: experiment)
        let expected: [(ExperimentalCVState, CartridgePhase)] = [
            (.unsupported(reason: "x"), .unsupported),
            (.idle(available: []), .idle),
            (.starting(experiment), .waiting),
            (.running(run), .live),
            (.completed(run), .settled),
            (.failed(CartridgeFailure(kind: .timedOut, message: "x")), .failed),
        ]
        for (state, phase) in expected {
            XCTAssertEqual(state.phase, phase, "\(state) mapped to the wrong phase")
        }
        XCTAssertTrue(ExperimentalCVState.running(run).isRunning)
        XCTAssertFalse(ExperimentalCVState.completed(run).isRunning)
    }

    // MARK: The frame channel
    //
    // The Tower's per-frame reply carries the running experiment's own answer —
    // `result_value`, `result_label`, `mean_intensity`, `processing_ms`,
    // `metrics` — and it arrives on the frame path, not on the cartridge
    // contract channel. `CVFrameReading` is that reply projected for display.
    // The tests below guard the two things that are easy to get wrong about it:
    // that a bare number never escapes without the Tower's own name for it, and
    // that showing it does not turn the cartridge layer's `.unsupported` state
    // into a state that carries data.

    private func frameResult(
        sequence: Int? = 41,
        meanIntensity: Double? = nil,
        processingMs: Double? = nil,
        resultValue: Double? = nil,
        resultLabel: String? = nil,
        metrics: [String: Double] = [:]
    ) -> TowerFrameResult {
        TowerFrameResult(
            sequence: sequence,
            meanIntensity: meanIntensity,
            processingMs: processingMs,
            resultValue: resultValue,
            resultLabel: resultLabel,
            stageMs: [:],
            metrics: metrics,
            // A Tower running no CV Lab attaches no `cv_lab` block, which is
            // the case these fixtures describe: they exercise the pair rule on
            // the message's own fields.
            cvLab: nil
        )
    }

    /// The pair rule, in both directions. `result_value` is a bare number whose
    /// meaning belongs to the experiment; rendering it under a caption this app
    /// chose would invent a unit nobody promised. A label with no number under
    /// it is the mirror-image fabrication — a row that asserts the experiment
    /// measured something it did not report.
    func testAHeadlineNumberAndItsLabelAreOnlyReadableTogether() {
        XCTAssertNil(
            CVFrameReading(frameResult(resultValue: 0.42)).headline,
            "a bare number escaped without the Tower's own name for it"
        )
        XCTAssertNil(
            CVFrameReading(frameResult(resultLabel: "edge density")).headline,
            "a label was offered with no number under it"
        )
        // Absent and empty are indistinguishable on the wire, and a row
        // captioned with whitespace is a bare number with extra steps.
        XCTAssertNil(
            CVFrameReading(frameResult(resultValue: 0.42, resultLabel: "  ")).headline,
            "whitespace was accepted as a label"
        )

        let both = CVFrameReading(frameResult(resultValue: 0.42, resultLabel: "edge density"))
        XCTAssertEqual(both.headline?.label, "edge density")
        XCTAssertEqual(both.headline?.value ?? -1, 0.42, accuracy: 0.0001)
    }

    /// Rule 3: absence renders as absence. A field the Tower omitted is unknown,
    /// and a zero it actually sent is a result — the two must not merge in
    /// either direction.
    func testAnOmittedFigureStaysAbsentAndAReportedZeroStaysAResult() {
        let silent = CVFrameReading(frameResult())
        XCTAssertNil(silent.headline)
        XCTAssertNil(silent.meanIntensity, "silence about intensity became a dark frame")
        XCTAssertNil(silent.processingMs, "silence about timing became instant work")
        XCTAssertTrue(silent.measurements.isEmpty)
        XCTAssertFalse(
            silent.hasAnything,
            "a reply carrying only a sequence number claimed to have a result in it"
        )

        let dark = CVFrameReading(frameResult(meanIntensity: 0, resultValue: 0, resultLabel: "edges"))
        XCTAssertEqual(dark.meanIntensity, 0)
        XCTAssertEqual(dark.headline?.value, 0)
        XCTAssertTrue(dark.hasAnything, "a reported zero was discarded as if it were silence")
    }

    /// The Tower's additive measurements arrive as a dictionary, which has no
    /// stable order. A list of numbers that reshuffles itself twelve times a
    /// second is unreadable, and an unnamed one is a bare number again.
    func testMeasurementsKeepTheirTowerGivenNamesInAStableOrder() {
        let reading = CVFrameReading(
            frameResult(metrics: ["edges": 4210, "": 7, "contrast": 0.5])
        )
        XCTAssertEqual(reading.measurements.map(\.label), ["contrast", "edges"])
        XCTAssertEqual(reading.measurements.first?.displayValue, "0.500")
        XCTAssertTrue(reading.hasAnything)
    }

    /// Two wire keys that differ only in whitespace are two entries the Tower
    /// sent, and they have to stay two distinguishable rows.
    ///
    /// The display label is trimmed — a row captioned with whitespace is a bare
    /// number with extra steps — so `{"edges": 1, " edges": 2}` collapses to one
    /// caption. When the row identity was that trimmed caption, `ForEach` saw
    /// duplicate ids and the sort key was equal for both; `sorted(by:)` is not
    /// stable, so the two rows could swap places on every reply. That is the
    /// exact twelve-times-a-second reshuffle the sort was added to prevent.
    func testKeysDifferingOnlyInWhitespaceStayDistinctAndTotallyOrdered() {
        let reading = CVFrameReading(frameResult(metrics: ["edges": 1, " edges": 2]))

        XCTAssertEqual(reading.measurements.count, 2, "the Tower sent two entries and one was dropped")
        XCTAssertEqual(
            Set(reading.measurements.map(\.id)).count, 2,
            "two rows share one identity, which is a duplicate ForEach id"
        )
        XCTAssertEqual(reading.measurements.map(\.label), ["edges", "edges"])
        // Ordered on the untrimmed key once the captions tie, so the order is
        // total and does not depend on the dictionary's iteration order.
        XCTAssertEqual(reading.measurements.map(\.id), [" edges", "edges"])
        XCTAssertEqual(reading.measurements.map(\.value), [2, 1])
    }

    /// Rule 16. `frame_result` has no provenance field, and silence must not be
    /// read as "measured" — so the reading states `.unknown` and owes the caveat
    /// that goes with it wherever its figures are drawn.
    func testTheFrameChannelClaimsNoProvenance() {
        XCTAssertEqual(CVFrameReading.provenance, .unknown)
        XCTAssertFalse(CVFrameReading.provenance.isInference)
        XCTAssertNotNil(
            CVFrameReading.provenance.caveat,
            "figures of unstated provenance would be drawn with nothing said about it"
        )
    }

    /// The invariant this whole design exists to protect, in the two halves it
    /// actually has — one checkable at runtime and one only at compile time.
    ///
    /// The frame reading is frame-path output. It must not become a case of
    /// `ExperimentalCVState`, because the Tower offers no typed contract for
    /// `experimental_cv` and that state is therefore `.unsupported` — a phase
    /// `CartridgePhase.mayCarryData` forbids from carrying anything. Showing the
    /// Tower's real answer must not cost that guarantee.
    ///
    /// **What no assertion can reach.** Swift cannot be asked at runtime what
    /// payload types an enum's cases carry, and assertions can only speak about
    /// values that exist — so an earlier version of this test, which built a
    /// `lab` and a `reading` that never interacted and then checked facts about
    /// the `lab` alone, would have stayed green if `case
    /// frameReading(CVFrameReading)` had been added to `ExperimentalCVState`
    /// and rendered. It asserted that the *current* state is `.unsupported`,
    /// which is true because `UnavailableExperimentalCVClient.state` is a
    /// hardcoded literal.
    ///
    /// **What does reach it**: `assertCarriesOnlyContractData` below switches
    /// over every case without a `default`. Adding a case stops this test
    /// target compiling, at the one line whose comment says why the case must
    /// not exist — which is the only mechanism available that a new case cannot
    /// pass silently.
    func testNoCartridgeStateCarriesAFrameReading() {
        let experiment = CVExperiment(id: "e", name: "E")
        let run = CVExperimentRun(experiment: experiment)
        for state: ExperimentalCVState in [
            .unsupported(reason: "no runner"),
            .idle(available: []),
            .idle(available: [experiment]),
            .starting(experiment),
            .running(run),
            .completed(run),
            .failed(CartridgeFailure(kind: .notSupported, message: "no")),
        ] {
            assertCarriesOnlyContractData(state)
        }
    }

    /// Every case of `ExperimentalCVState`, and what each may carry.
    ///
    /// No `default`, deliberately. A case added here to carry `CVFrameReading`
    /// — the frame channel's output, which has no contract behind it and no
    /// provenance attached — breaks the build rather than passing, and whoever
    /// adds it has to argue with this comment first. The frame channel belongs
    /// beside this enum (`CVFrameReading`, read straight off `TowerClient`),
    /// never inside it.
    private func assertCarriesOnlyContractData(_ state: ExperimentalCVState) {
        switch state {
        case .unsupported, .idle, .starting, .failed:
            XCTAssertNil(
                state.run,
                "a state with no run produced one: \(state)"
            )
            XCTAssertFalse(
                state.phase.mayCarryData,
                "\(state) claims a phase that permits data it does not have"
            )
        case .running(let run), .paused(let run), .completed(let run):
            // The only three cases that carry results, and what they carry is
            // a `CVExperimentRun` — metrics with `provenance` required by the
            // type, which is exactly what the frame channel
            // cannot supply.
            XCTAssertEqual(state.run, run)
            XCTAssertTrue(state.phase.mayCarryData)
        }
    }

    /// The runtime half: with the only client that exists, nothing the frame
    /// channel produces reaches the cartridge state — before or after a reading
    /// with a real result in it has been built and read.
    func testReadingTheFrameChannelLeavesTheCartridgeStateUntouched() {
        let lab = ExperimentalCVViewModel(client: UnavailableExperimentalCVClient())
        let before = lab.state

        let reading = CVFrameReading(frameResult(resultValue: 0.42, resultLabel: "edge density"))
        XCTAssertTrue(reading.hasAnything, "the fixture must actually carry a result")
        XCTAssertNotNil(reading.headline)

        XCTAssertEqual(lab.state, before, "the frame channel moved the cartridge state")
        XCTAssertEqual(lab.state.phase, .unsupported)
        XCTAssertFalse(lab.state.phase.mayCarryData)
        XCTAssertNil(lab.state.run, "the frame channel produced a cartridge run")
        XCTAssertEqual(
            lab.availability(isTowerReachable: true),
            .noContract,
            "experimental-cv gained a contract it was never offered"
        )
        XCTAssertEqual(lab.phase(isTowerReachable: true), .unsupported)
    }

    /// Reading the experiment's answer is not the same as being able to choose
    /// it. There is no wire message to list, start or stop an experiment, so the
    /// list stays empty and the request stays refused.
    func testShowingTheAnswerDoesNotMakeAnythingRunnable() {
        let client = UnavailableExperimentalCVClient()
        XCTAssertThrowsError(try client.run(CVExperiment(id: "any", name: "Any"))) { error in
            XCTAssertEqual((error as? CartridgeFailure)?.kind, .notSupported)
        }

        let lab = ExperimentalCVViewModel(client: client)
        XCTAssertTrue(
            lab.availableExperiments.isEmpty,
            "the app is offering experiments the Tower has not declared"
        )
    }
}

// MARK: - Document Memory

/// Guards the retrieval-truthfulness rules from `ENVIRONMENTAL-MEMORY.md` and
/// Core Principle 3.
@MainActor
final class DocumentMemoryModelTests: XCTestCase {

    /// "Nothing matched" and "never observed" are different answers about
    /// different things, and merging them lets a gap in what the glasses
    /// happened to see read as a statement about the world.
    func testNothingMatchedAndNeverObservedAreDifferentAnswers() {
        XCTAssertNotEqual(DocumentQueryEvidence.notFound, .noObservation)
        XCTAssertNotEqual(
            DocumentQueryEvidence.notFound.explanation,
            DocumentQueryEvidence.noObservation.explanation
        )
        XCTAssertTrue(
            DocumentQueryEvidence.noObservation.explanation.contains("not the same"),
            "a never-observed answer failed to disclaim absence"
        )
    }

    /// A result cannot claim a match while carrying nothing — and the safe
    /// direction from that is a failure, never a stronger claim.
    ///
    /// An earlier version quietly rewrote it to `.notFound`, whose user-facing
    /// sentence is "Nothing in the Tower's document memory matched." That is a
    /// **definite negative statement about the user's own memory, manufactured
    /// from a decode failure**, with the Tower's confidence discarded on the
    /// way — Core Principle 3 in exactly the forbidden direction.
    func testAMatchWithNoDocumentsIsUndecodableRatherThanNotFound() {
        XCTAssertThrowsError(
            try DocumentQueryResult(
                query: .text("parking"),
                origin: .appText,
                documents: [],
                evidence: .matched(confidence: 0.9)
            )
        ) { error in
            guard let failure = error as? CartridgeFailure else {
                return XCTFail("expected a CartridgeFailure, got \(error)")
            }
            XCTAssertEqual(failure.kind, .undecodableResponse)
        }
    }

    /// The confidence must survive when the match is genuine — Core Principle 4.
    func testAGenuineMatchKeepsItsConfidence() throws {
        let result = try DocumentQueryResult(
            query: .text("parking"),
            origin: .appText,
            documents: [RememberedDocument(id: "1")],
            evidence: .matched(confidence: 0.42)
        )
        XCTAssertEqual(result.evidence, .matched(confidence: 0.42))
        XCTAssertTrue(result.evidence.explanation.contains("42%"))
    }

    /// A never-observed answer must not be turned into a match by the coercion
    /// above, in either direction.
    func testAnEmptyNeverObservedAnswerIsPreserved() throws {
        let result = try DocumentQueryResult(
            query: .observedWithin(DateInterval(start: Date(timeIntervalSince1970: 0), duration: 60)),
            origin: .appText,
            documents: [],
            evidence: .noObservation
        )
        XCTAssertEqual(result.evidence, .noObservation)
    }

    /// A record with no title is described, not named. An invented title is a
    /// claim about the document's contents.
    func testADocumentWithoutATitleIsDescribedRatherThanNamed() {
        XCTAssertEqual(RememberedDocument(id: "1").displayTitle, "Untitled document")
        XCTAssertEqual(RememberedDocument(id: "1", title: "Parking notice").displayTitle, "Parking notice")
    }

    /// Silence about text and a verdict of unreadable are different claims.
    func testUnknownTextIsNotAVerdictOfUnreadable() {
        XCTAssertNotEqual(
            DocumentTextAvailability.unknown.displayName,
            DocumentTextAvailability.notReadable.displayName
        )
        XCTAssertFalse(DocumentTextAvailability.unknown.hasText)
        XCTAssertFalse(DocumentTextAvailability.notReadable.hasText)
        XCTAssertTrue(DocumentTextAvailability.extracted(characterCount: nil).hasText)
    }

    /// The list carries a count, never the text. A list of documents must not
    /// also be a bulk transfer of every document's contents into the phone.
    func testExtractedTextIsReportedAsACountNotAsContent() {
        XCTAssertEqual(
            DocumentTextAvailability.extracted(characterCount: 1_200).displayName,
            "1200 characters"
        )
        XCTAssertEqual(
            DocumentTextAvailability.extracted(characterCount: nil).displayName,
            "Text available"
        )
    }

    /// A document's thumbnail obeys the same rule as every other stored image.
    func testAnUnredactedThumbnailIsWithheld() {
        let raw = RememberedDocument(id: "1", thumbnail: .available(.rawEphemeral))
        XCTAssertFalse(raw.isThumbnailDisplayable)
        let redacted = RememberedDocument(id: "1", thumbnail: .available(.redacted))
        XCTAssertTrue(redacted.isThumbnailDisplayable)
    }

    /// Observation time never falls back to arrival time, at the document level.
    func testADocumentWithNoObservationTimeSaysSo() {
        let document = RememberedDocument(
            id: "1",
            time: ObservationTime(observedAt: nil, receivedAt: Date())
        )
        XCTAssertNil(document.time.displayableObservationTime)
        XCTAssertTrue(document.time.isObservationTimeUnknown)
    }

    /// Time queries are ranges. "This morning" is not an instant, and answering
    /// it as one answers a different question.
    func testTimeQueriesAreRangesRatherThanInstants() {
        let interval = DateInterval(start: Date(timeIntervalSince1970: 0), duration: 3_600)
        XCTAssertEqual(DocumentQuery.observedWithin(interval), .observedWithin(interval))
        XCTAssertNotEqual(
            DocumentQuery.observedWithin(interval),
            .observedWithin(DateInterval(start: Date(timeIntervalSince1970: 0), duration: 60))
        )
    }

    /// The origin exists so a future input layer needs no new path. No such
    /// layer exists; this asserts the seam does.
    func testAnInputSourceBeyondTheAppIsRepresentable() {
        XCTAssertEqual(DocumentQueryOrigin.allCases.count, 2)
        XCTAssertTrue(DocumentQueryOrigin.allCases.contains(.externalIntent))
    }

    func testEveryDocumentStateMapsToTheRightPhase() throws {
        let result = try DocumentQueryResult(
            query: .recent(limit: 5), origin: .appText, evidence: .notFound
        )
        let expected: [(DocumentMemoryState, CartridgePhase)] = [
            (.unsupported(reason: "x"), .unsupported),
            (.idle, .idle),
            (.searching(.recent(limit: 5)), .waiting),
            (.results(result), .settled),
            (.failed(CartridgeFailure(kind: .transport, message: "x")), .failed),
        ]
        for (state, phase) in expected {
            XCTAssertEqual(state.phase, phase, "\(state) mapped to the wrong phase")
        }
    }
}

// MARK: - Camera readiness

/// A scriptable `WearablesInterface`, the seam this suite has not had before.
///
/// It exists for one question the rest of the file cannot ask: what happens to
/// camera readiness when DAT answers a permission query *before* it has
/// discovered a device. On real hardware that is not an edge case — it is what
/// happens on every cold launch, and it is what
/// `GlassesConnection.refreshCameraPermissionForAvailableDevice()` repairs.
///
/// Two DAT types cannot be faked: `Device` and `DeviceSession` both carry
/// `@_hasMissingDesignatedInitializers`. So `deviceForIdentifier` returns nil
/// (the link-state trigger is therefore device-only, and is covered physically
/// rather than here) and `createSession` throws — which is enough to assert the
/// property that matters most, that nothing automatic ever asks for a session.
final class ScriptedWearables: WearablesInterface, @unchecked Sendable {
    struct Token: AnyListenerToken {
        func cancel() async {}
    }

    private let lock = NSLock()
    private var _devices: [DeviceIdentifier] = []
    private var _permissionResults: [Result<PermissionStatus, PermissionError>]
    private var _permissionCheckCount = 0
    private var _createSessionCount = 0
    private var devicesContinuation: AsyncStream<[DeviceIdentifier]>.Continuation?
    private var _isSubscribed = false

    /// Results for successive `checkPermissionStatus` calls; the final entry
    /// repeats once the script is exhausted.
    init(permissionResults: [Result<PermissionStatus, PermissionError>]) {
        _permissionResults = permissionResults
    }

    /// Whether `devicesStream()` has been subscribed. `GlassesConnection`
    /// subscribes from a `Task` in `init`, so a test that emits immediately
    /// after construction would otherwise race the subscription and lose.
    var isSubscribed: Bool { lock.withLock { _isSubscribed } }
    var permissionCheckCount: Int { lock.withLock { _permissionCheckCount } }
    var createSessionCount: Int { lock.withLock { _createSessionCount } }

    /// Publishes a new device list, as `devicesStream()` would.
    func emitDevices(_ devices: [DeviceIdentifier]) {
        let continuation: AsyncStream<[DeviceIdentifier]>.Continuation? = lock.withLock {
            _devices = devices
            return devicesContinuation
        }
        continuation?.yield(devices)
    }

    var registrationState: RegistrationState { .registered }
    var devices: [DeviceIdentifier] { lock.withLock { _devices } }

    /// Replays the current list on subscription, as DAT's own stream does —
    /// which is also what makes a warm launch (devices already known before
    /// anyone subscribes) reproducible here.
    func devicesStream() -> AsyncStream<[DeviceIdentifier]> {
        AsyncStream { continuation in
            let current: [DeviceIdentifier] = lock.withLock {
                devicesContinuation = continuation
                _isSubscribed = true
                return _devices
            }
            if !current.isEmpty { continuation.yield(current) }
        }
    }

    func checkPermissionStatus(_ permission: Permission) async throws(PermissionError) -> PermissionStatus {
        let result: Result<PermissionStatus, PermissionError> = lock.withLock {
            _permissionCheckCount += 1
            let next = _permissionResults.first ?? .failure(.internalError)
            if _permissionResults.count > 1 { _permissionResults.removeFirst() }
            return next
        }
        switch result {
        case .success(let status): return status
        case .failure(let error): throw error
        }
    }

    func createSession(deviceSelector: any DeviceSelector) throws(DeviceSessionError) -> DeviceSession {
        lock.withLock { _createSessionCount += 1 }
        throw .noEligibleDevice
    }

    // Unused by these tests, but required by the protocol.
    func addRegistrationStateListener(_ listener: @escaping @Sendable (RegistrationState) -> Void) -> any AnyListenerToken { Token() }
    func addDevicesListener(_ listener: @escaping @Sendable ([DeviceIdentifier]) -> Void) -> any AnyListenerToken { Token() }
    func deviceForIdentifier(_ identifier: DeviceIdentifier) -> Device? { nil }
    func requestPermission(_ permission: Permission) async throws(PermissionError) -> PermissionStatus { .granted }
    func startRegistration() async throws(RegistrationError) {}
    func startUnregistration() async throws(UnregistrationError) {}
    func handleUrl(_ url: URL) async throws(WearablesHandleURLError) -> Bool { false }
    func openFirmwareUpdate() async throws(NavigationError) {}
    func openDATGlassesAppUpdate() async throws(NavigationError) {}
}

/// The regression this suite exists for.
///
/// Physically observed on an iPhone 16 Pro with Ray-Ban Meta glasses: the
/// launch permission read threw `No wearable devices have been discovered or
/// registered`, the glasses arrived a moment later, and `cameraPermissionStatus`
/// stayed `nil` for the rest of the process — so every capture session was
/// refused for a permission that had in fact been granted.
@MainActor
final class CameraReadinessTests: XCTestCase {

    /// Polls rather than sleeps a fixed interval: the refresh is driven by a
    /// `Task` hop off the `devicesStream` loop, not by a known delay.
    private func waitUntil(
        _ condition: @MainActor () -> Bool,
        timeout: TimeInterval = 2
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return condition()
    }

    /// The regression itself. A permission read that fails because DAT has no
    /// device yet must not leave readiness permanently unknown.
    func testAPermissionCheckBeforeDiscoveryDoesNotPoisonLaterReadiness() async {
        let wearables = ScriptedWearables(permissionResults: [
            .failure(.noDevice),
            .success(.granted),
        ])
        let connection = GlassesConnection(wearables: wearables)

        // The launch read, as `ProjectManager.startAutomaticConnections()`
        // makes it: no devices yet, so DAT refuses to answer.
        connection.checkCameraPermission(reportErrors: false)
        let refused = await waitUntil { wearables.permissionCheckCount >= 1 }
        XCTAssertTrue(refused, "the launch read should have been attempted")
        XCTAssertNil(
            connection.cameraPermissionStatus,
            "a failed read must leave the status unknown rather than guessing"
        )
        XCTAssertNil(
            connection.errorMessage,
            "the automatic read must not raise an alert nobody's action caused"
        )

        // The glasses arrive.
        wearables.emitDevices(["device-1"])

        let recovered = await waitUntil { connection.cameraPermissionStatus == .granted }
        XCTAssertTrue(
            recovered,
            "device arrival must make readiness truthful again; it stayed \(String(describing: connection.cameraPermissionStatus))"
        )
    }

    /// Auto-connect != auto-camera. The whole point of refreshing readiness on
    /// device arrival is that it must buy nothing else.
    func testDeviceArrivalRefreshesReadinessButNeverStartsTheCamera() async {
        let wearables = ScriptedWearables(permissionResults: [
            .failure(.noDevice),
            .success(.granted),
        ])
        let connection = GlassesConnection(wearables: wearables)
        connection.checkCameraPermission(reportErrors: false)
        _ = await waitUntil { wearables.permissionCheckCount >= 1 }

        wearables.emitDevices(["device-1"])
        _ = await waitUntil { connection.cameraPermissionStatus == .granted }

        XCTAssertEqual(
            wearables.createSessionCount, 0,
            "a device becoming available must not create a capture session"
        )
        #if DEBUG
        XCTAssertEqual(connection.cameraStreamState, .stopped, "the camera must stay off")
        XCTAssertEqual(connection.frameCount, 0, "no frames may be captured without an explicit start")
        #endif
    }

    /// The refresh is bounded by the unknown status, not by an edge, so it must
    /// stop asking the moment it has an answer — otherwise device churn turns
    /// into an unbounded query loop.
    func testReadinessIsNotReReadOnceItIsKnown() async {
        let wearables = ScriptedWearables(permissionResults: [.success(.granted)])
        let connection = GlassesConnection(wearables: wearables)
        _ = await waitUntil { wearables.isSubscribed }

        wearables.emitDevices(["device-1"])
        let known = await waitUntil { connection.cameraPermissionStatus == .granted }
        XCTAssertTrue(known)
        let afterFirstAnswer = wearables.permissionCheckCount

        // Repeated arrivals and departures, as glasses reconnecting would.
        for _ in 0..<5 {
            wearables.emitDevices([])
            wearables.emitDevices(["device-1"])
        }
        try? await Task.sleep(nanoseconds: 200_000_000)

        XCTAssertEqual(
            wearables.permissionCheckCount, afterFirstAnswer,
            "a known permission must not be re-read on every device change"
        )
        XCTAssertEqual(connection.cameraPermissionStatus, .granted)
    }

    /// A warm relaunch: `init` seeds `devices` from `wearables.devices`, so the
    /// first stream yield is not an empty->non-empty transition. An
    /// edge-triggered refresh would never fire here.
    func testReadinessIsRefreshedWhenDevicesWereAlreadyKnownAtLaunch() async {
        let wearables = ScriptedWearables(permissionResults: [.success(.granted)])
        wearables.emitDevices(["device-1"])
        let connection = GlassesConnection(wearables: wearables)
        XCTAssertEqual(connection.devices, ["device-1"], "the fixture must reproduce a warm launch")

        // No new emission: the only yield is the stream replaying a list that
        // was already known. An empty->non-empty edge never occurs here.
        let known = await waitUntil { connection.cameraPermissionStatus == .granted }
        XCTAssertTrue(known, "a pre-seeded device list must still produce a readiness read")
    }

    /// A permission that is genuinely denied is a truthful answer, and must be
    /// recorded as one rather than retried into an alert loop.
    func testADeniedPermissionIsRecordedAndNotRetried() async {
        let wearables = ScriptedWearables(permissionResults: [.success(.denied)])
        let connection = GlassesConnection(wearables: wearables)
        _ = await waitUntil { wearables.isSubscribed }

        wearables.emitDevices(["device-1"])
        let settled = await waitUntil { connection.cameraPermissionStatus == .denied }
        XCTAssertTrue(settled)
        let afterAnswer = wearables.permissionCheckCount

        wearables.emitDevices([])
        wearables.emitDevices(["device-1"])
        try? await Task.sleep(nanoseconds: 200_000_000)

        XCTAssertEqual(wearables.permissionCheckCount, afterAnswer)
        XCTAssertNil(connection.errorMessage, "an automatic read must never raise an alert")
    }
}

// MARK: - Connection lifetime

@MainActor
final class ConnectionLifetimeTests: XCTestCase {

    /// The retain cycle, made a failing test rather than an argument.
    ///
    /// `GlassesConnection` stores the three `Task`s it creates in `init`. If a
    /// task body holds `self` strongly for the life of an unbounded
    /// `for await` — which `guard let self else { return }` placed *outside*
    /// the loop does — then the object owns the task and the task owns the
    /// object, and the `isolated deinit` that stops the camera and the device
    /// session can never run.
    ///
    /// Written after the fix, and verified against the code before it: this
    /// test fails on the old shape and passes on the new one. Without it
    /// nothing in the suite would notice the cycle being reintroduced, and the
    /// pattern is idiomatic enough to come back by accident.
    func testGlassesConnectionDeallocatesWhenReleased() async {
        weak var weakConnection: GlassesConnection?
        do {
            let wearables = ScriptedWearables(permissionResults: [.success(.granted)])
            let connection = GlassesConnection(wearables: wearables)
            weakConnection = connection
            // The `init` tasks subscribe asynchronously, and the probe only
            // means anything once they are actually suspended on their
            // streams — that suspension is where the strong reference would
            // be held. Asserted rather than merely waited for, so this fails
            // loudly instead of passing over streams that were never live.
            for _ in 0..<20 where !wearables.isSubscribed {
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
            XCTAssertTrue(wearables.isSubscribed, "the probe needs live streams to be meaningful")
            withExtendedLifetime(connection) {}
        }
        // Deallocation follows task cancellation in `deinit`, which is not
        // synchronous with the scope exit.
        for _ in 0..<50 {
            if weakConnection == nil { break }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertNil(
            weakConnection,
            "GlassesConnection outlived its last strong reference — isolated deinit never ran"
        )
    }
}

// MARK: - Capture resolution

/// Pins the DEBUG-only capture-resolution control.
///
/// The control exists because the rung was a hardcoded `.low` and Document
/// Memory's premise cannot be tested at 360x640 — its measured word recall is
/// 0.429-0.810 there against 0.957-1.000 at 1280x720. What these tests actually
/// guard is the *other* side of that: World Builder's path is physically proven
/// at `.low`, so the default must not move, and the label must not invent a
/// size DAT did not declare.
///
/// DEBUG-only because `CaptureResolutionPreference` is, and because
/// `MWDATCamera` is imported under `#if DEBUG` in `GlassesConnection.swift`.
#if DEBUG
final class CaptureResolutionPreferenceTests: XCTestCase {

    /// The rung every existing measurement was taken at, including the P3 clean
    /// walk. If this moves, every prior figure silently stops being comparable
    /// and World Builder's proven path changes underneath it.
    func testTheDefaultIsLow() {
        XCTAssertEqual(CaptureResolutionPreference.default, .low)
    }

    /// A fresh connection starts at the default rather than at whatever the
    /// picker last showed — there is no persistence here and there should not
    /// be one, because a rung silently surviving a relaunch is how a walk gets
    /// recorded at the wrong resolution without anyone choosing that.
    @MainActor
    func testAFreshConnectionStartsAtTheDefault() {
        let connection = GlassesConnection(wearables: ScriptedWearables(permissionResults: []))
        XCTAssertEqual(connection.captureResolution, .default)
    }

    /// Three rungs, each mapping to a distinct `StreamingResolution`. A
    /// collapsed mapping would make the picker move while the stream did not.
    func testEachRungMapsToADistinctStreamingResolution() {
        let all = CaptureResolutionPreference.allCases
        XCTAssertEqual(all.count, 3)
        XCTAssertEqual(Set(all.map(\.id)).count, 3, "rung ids must be unique")

        let resolutions = all.map(\.streamingResolution)
        XCTAssertEqual(resolutions.count, 3)
        for (index, lhs) in resolutions.enumerated() {
            for rhs in resolutions[(index + 1)...] {
                XCTAssertNotEqual(lhs, rhs, "two rungs collapsed onto one StreamingResolution")
            }
        }
    }

    /// The displayed size must come from DAT, not from a literal in this app.
    ///
    /// Asserted against `StreamingResolution.videoFrameSize` rather than
    /// against the string "360x640", so that if the SDK ever changes a rung the
    /// label follows it instead of quietly going stale — which is the whole
    /// reason `declaredSizeDescription` reads the SDK at all.
    func testTheDeclaredSizeIsReadFromDATRatherThanHardcoded() {
        for rung in CaptureResolutionPreference.allCases {
            let size = rung.streamingResolution.videoFrameSize
            XCTAssertEqual(
                rung.declaredSizeDescription,
                "\(size.width)x\(size.height)",
                "\(rung.rawValue) must render the size DAT declares"
            )
        }
    }

    /// Ties the default rung to the physical evidence. The P3 clean walk
    /// recorded 108 frames, every one 360x640
    /// (`docs/evidence/2026-08-26-p3-clean-walk-console.txt`). If DAT's `.low`
    /// ever stops meaning that, the corpus and every figure derived from it
    /// stop being comparable, and this test is the tripwire.
    func testLowIsStillTheRungTheWalkWasMeasuredAt() {
        XCTAssertEqual(CaptureResolutionPreference.low.declaredSizeDescription, "360x640")
    }

    /// Every rung needs a picker segment. An empty label renders as a blank
    /// segment the user cannot identify.
    func testEveryRungHasANonEmptyShortLabel() {
        for rung in CaptureResolutionPreference.allCases {
            XCTAssertFalse(rung.shortLabel.isEmpty, "\(rung.rawValue) has no label")
        }
    }
}
#endif


// MARK: - Capture session claim

/// Pins the predicate that decides whether the capture rung may be changed.
///
/// This exists because of a defect an adversarial review found and this suite
/// could not have caught: the picker was originally gated on
/// `isCaptureEngaged`, which answers *"should the button read Stop?"* and is
/// deliberately **false** during `.paused` and `.stopping`. Both DAT enums have
/// those cases and the original allow-lists covered neither, so a cap-touch
/// pause — documented as device-initiated in `07-PLATFORM-CONSTRAINTS.md` §146
/// — unlocked the picker while the session and camera were both still held. A
/// change made there is accepted, displayed, and then silently discarded by
/// `beginCameraStream`'s `guard camera == nil` on resume, leaving the panel
/// asserting a rung that was never requested.
///
/// Every combination is exercised rather than a chosen few, because the defect
/// was precisely a case nobody thought to enumerate.
#if DEBUG
final class CaptureSessionClaimTests: XCTestCase {

    private let allSessionStates: [DeviceSessionState] =
        [.idle, .starting, .started, .paused, .stopping, .stopped]
    private let allStreamStates: [MWDATCamera.StreamState] =
        [.stopped, .stopping, .waitingForDevice, .starting, .streaming, .paused]

    /// The regression. Neither of these may report "no session".
    func testAPausedSessionStillCountsAsClaimed() {
        XCTAssertTrue(
            GlassesConnection.isCaptureSessionClaimed(session: .paused, stream: .paused),
            "a cap-touch pause holds the session and camera; the rung cannot change under it"
        )
        XCTAssertTrue(
            GlassesConnection.isCaptureSessionClaimed(session: .started, stream: .paused),
            "the stream pausing alone still leaves a camera that beginCameraStream will refuse to replace"
        )
    }

    /// `stopCameraSession()` sets both to `.stopping` synchronously, but
    /// `deviceSession` stays non-nil until the `.stopped` callback runs
    /// cleanup. In that window `startCameraSession()` refuses, so "applies at
    /// the next start" describes a start the app is declining to perform.
    func testAStoppingSessionStillCountsAsClaimed() {
        XCTAssertTrue(
            GlassesConnection.isCaptureSessionClaimed(session: .stopping, stream: .stopping)
        )
    }

    /// The only combination that may report "no session" is the fully torn-down
    /// one. Asserted over every pairing so a future case cannot be quietly
    /// admitted to the not-claimed side.
    func testOnlyAFullyTornDownSessionIsUnclaimed() {
        for session in allSessionStates {
            for stream in allStreamStates {
                let claimed = GlassesConnection.isCaptureSessionClaimed(session: session, stream: stream)
                let shouldBeIdle = (session == .idle || session == .stopped) && stream == .stopped
                XCTAssertEqual(
                    claimed,
                    !shouldBeIdle,
                    "session=\(session) stream=\(stream) decided wrongly"
                )
            }
        }
    }

    /// The two predicates must genuinely differ, and differ in the direction
    /// that matters — `isCaptureSessionClaimed` is the strictly more
    /// conservative one. If a refactor ever collapsed them, the defect returns
    /// silently.
    func testTheClaimPredicateIsStrictlyBroaderThanTheEngagedPredicate() {
        var foundADisagreement = false
        for session in allSessionStates {
            for stream in allStreamStates {
                let claimed = GlassesConnection.isCaptureSessionClaimed(session: session, stream: stream)
                // `isCaptureEngaged`'s rule, restated here rather than called,
                // so this test still fails if that property is widened to match
                // instead of the two staying deliberately different.
                let engagedBySession: Bool
                switch session {
                case .starting, .started: engagedBySession = true
                default: engagedBySession = false
                }
                let engagedByStream: Bool
                switch stream {
                case .streaming, .starting, .waitingForDevice: engagedByStream = true
                default: engagedByStream = false
                }
                let engaged = engagedBySession || engagedByStream
                if engaged { XCTAssertTrue(claimed, "engaged must imply claimed") }
                if claimed != engaged { foundADisagreement = true }
            }
        }
        XCTAssertTrue(
            foundADisagreement,
            "the two predicates agree everywhere, which means the paused/stopping gap is back"
        )
    }

    // MARK: The four-way reading

    /// **A device-initiated pause is its own answer, not "not running".**
    ///
    /// This is the half of the defect above that a `Bool` could not express.
    /// `isCaptureEngaged` is false during `.paused`, so Home and World Builder
    /// flip their primary control back to "Start capture" — and a tap on it
    /// hits `startCameraSession()`'s `guard deviceSession == nil` and does
    /// nothing observable at all. `CaptureClaim` exists so a caller can tell
    /// "nothing is running, start one" apart from "a session is held and the
    /// glasses stopped delivering, and nothing in this app can change that".
    func testADevicePausedSessionIsItsOwnClaimRatherThanNoSession() {
        XCTAssertEqual(
            GlassesConnection.captureClaim(session: .paused, stream: .paused), .devicePaused
        )
        XCTAssertEqual(
            GlassesConnection.captureClaim(session: .started, stream: .paused),
            .devicePaused,
            "the stream pausing alone is still the device having stopped delivery"
        )
        XCTAssertEqual(
            GlassesConnection.captureClaim(session: .paused, stream: .streaming),
            .devicePaused,
            "the session's own pause outranks a stream state that has not caught up"
        )
    }

    /// Only a fully torn-down pair is startable, and the claim says so for every
    /// combination rather than for the few anybody thought to check.
    func testOnlyAFullyTornDownPairIsUnclaimed() {
        for session in allSessionStates {
            for stream in allStreamStates {
                let claim = GlassesConnection.captureClaim(session: session, stream: stream)
                let tornDown = (session == .idle || session == .stopped) && stream == .stopped
                XCTAssertEqual(
                    claim == .unclaimed, tornDown,
                    "session=\(session) stream=\(stream) decided wrongly"
                )
            }
        }
    }

    /// The two readings must agree about the one thing they both answer: a
    /// claim of `.unclaimed` is exactly the case `isCaptureSessionClaimed`
    /// calls unclaimed. If they ever disagree, one of them is lying about
    /// whether a `DeviceSession` is held.
    func testTheClaimAndTheBooleanAgreeAboutWhatIsHeld() {
        for session in allSessionStates {
            for stream in allStreamStates {
                let claim = GlassesConnection.captureClaim(session: session, stream: stream)
                let held = GlassesConnection.isCaptureSessionClaimed(
                    session: session, stream: stream
                )
                XCTAssertEqual(
                    held, claim != .unclaimed,
                    "session=\(session) stream=\(stream): the two readings disagree"
                )
            }
        }
    }

    /// A teardown in progress is neither running nor startable. Reporting it as
    /// `.unclaimed` would offer a Start that `startCameraSession()` refuses in
    /// exactly that window.
    func testATeardownInProgressIsNeitherRunningNorStartable() {
        XCTAssertEqual(
            GlassesConnection.captureClaim(session: .stopping, stream: .stopping), .ending
        )
        XCTAssertEqual(
            GlassesConnection.captureClaim(session: .stopped, stream: .stopping), .ending
        )
    }
}
#endif

// MARK: - Motion probe (DAT 1.0.0 spike)

#if DEBUG
/// The probe's buffer arithmetic, from fixed numbers. No DAT involved.
final class MotionSampleRingTests: XCTestCase {

    /// `i`-th sample of a stream whose device clock and receipt clock both
    /// tick at `hz`.
    private func sample(_ i: Int, hz: Double = 30, gyro: Float? = 1) -> MotionProbeSample {
        MotionProbeSample(
            deviceTimestampNs: Int64((Double(i) / hz * 1_000_000_000).rounded()),
            receivedAt: 100 + Double(i) / hz,
            gyroMagnitude: gyro
        )
    }

    /// The two rates come from two different clocks and must not be blended:
    /// the glasses measured at 30 Hz, the phone received at 25 Hz.
    func testEachRateComesFromItsOwnClock() throws {
        var ring = MotionSampleRing()
        for i in 0..<60 {
            ring.append(MotionProbeSample(
                deviceTimestampNs: Int64(i) * 33_333_333,
                receivedAt: 100 + Double(i) / 25,
                gyroMagnitude: 0.5
            ))
        }
        XCTAssertEqual(try XCTUnwrap(ring.deviceRateHz), 30, accuracy: 0.01)
        XCTAssertEqual(try XCTUnwrap(ring.arrivalRateHz), 25, accuracy: 0.01)
    }

    /// Ten seconds in, only the last three are kept; the total still counts
    /// everything.
    func testTheBufferKeepsOnlyTheLastFewSeconds() throws {
        var ring = MotionSampleRing(window: 3)
        for i in 0..<300 { ring.append(sample(i)) }
        XCTAssertEqual(ring.totalReceived, 300)
        XCTAssertTrue((90...91).contains(ring.samples.count), "kept \(ring.samples.count)")
        let first = try XCTUnwrap(ring.samples.first), last = try XCTUnwrap(ring.samples.last)
        XCTAssertLessThanOrEqual(last.receivedAt - first.receivedAt, 3 + 1e-9)
        XCTAssertEqual(try XCTUnwrap(ring.arrivalRateHz), 30, accuracy: 0.01)
    }

    /// A burst faster than any configured rate cannot grow it without bound.
    func testABurstIsCappedAtTheCapacity() {
        var ring = MotionSampleRing()
        for i in 0..<1_000 {
            ring.append(MotionProbeSample(deviceTimestampNs: Int64(i), receivedAt: 100, gyroMagnitude: nil))
        }
        XCTAssertEqual(ring.samples.count, MotionSampleRing.capacity)
        XCTAssertEqual(ring.samples.last?.deviceTimestampNs, 999, "the newest samples are the ones kept")
        XCTAssertNil(ring.arrivalRateHz, "no receipt interval, so no receipt rate")
    }

    /// A sample without a gyroscope reading is missing, not zero.
    func testAMissingGyroscopeIsNeverReadAsZero() {
        var ring = MotionSampleRing()
        XCTAssertNil(ring.latestGyroMagnitude)
        XCTAssertNil(ring.deviceRateHz)
        ring.append(sample(0, gyro: 2.5))
        ring.append(sample(1, gyro: 0.25))
        ring.append(sample(2, gyro: nil))
        XCTAssertEqual(ring.latestGyroMagnitude, 0.25)
        XCTAssertEqual(ring.peakGyroMagnitude, 2.5)
    }
}

/// End to end through DAT 1.0.0's Mock Device Kit: a real `DeviceSession`, a
/// real `Motion` capability, and `MockMotionKit` replaying a recording.
///
/// The recording is synthesized, not captured -- no glasses IMU recording
/// exists in this repository yet -- in the shape DAT documents: accelerometer
/// at rest reads gravity (`mockMotionGravityMetersPerSecondSquared`), and a
/// 1.5 s head turn about the vertical axis peaks at 3 rad/s. Mock replay
/// "follows the configured sampling rate rather than the intervals between
/// recorded timestamps", so what the app receives measures the app's
/// configuration, not the recording's.
///
/// `MockMotionKit.isStreaming` is the independent witness for both tests: it is
/// the mock glasses' own statement that a Motion capability is streaming from
/// them, read from outside `GlassesConnection`.
@MainActor
final class MotionProbeMockDeviceTests: XCTestCase {

    private var mockGlasses: (any MockGlasses)?

    override func setUp() async throws {
        MockDeviceKit.shared.enable()
        let glasses = try MockDeviceKit.shared.pairGlasses(model: .rayBanMeta)
        glasses.powerOn()
        glasses.don()
        mockGlasses = glasses
    }

    override func tearDown() async throws {
        if let mockGlasses {
            await MockDeviceKit.shared.unpairDevice(mockGlasses)
        }
        await MockDeviceKit.shared.disable()
        mockGlasses = nil
    }

    static let recordingHz = 30.0

    static func headTurnRecording(seconds: Double = 6) -> [MWDATMockDevice.MotionSample] {
        let gravity = mockMotionGravityMetersPerSecondSquared
        return (0..<Int(recordingHz * seconds)).map { i in
            let t = Double(i) / recordingHz
            let turning = t >= 1.5 && t < 3.0
            let yawRate = turning ? Float(3 * sin((t - 1.5) / 1.5 * .pi)) : 0
            return MWDATMockDevice.MotionSample(
                timestampNs: Int64((t * 1_000_000_000).rounded()),
                accelerometer: MWDATMockDevice.Vector3(x: 0, y: gravity, z: 0),
                gyroscope: MWDATMockDevice.Vector3(x: 0, y: yawRate, z: 0),
                source: .glasses
            )
        }
    }

    private func waitUntil(
        timeout: TimeInterval,
        _ condition: @MainActor () -> Bool
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return condition()
    }

    /// Brings a `GlassesConnection` on the real `Wearables.shared` to the
    /// point where a Start is a real start, then starts capture.
    private func startCapture(_ connection: GlassesConnection) async throws {
        let ready = await waitUntil(timeout: 15) {
            connection.hasActiveDevice && connection.cameraPermissionStatus == .granted
        }
        XCTAssertTrue(ready, "mock glasses never became an active device with camera permission (active=\(connection.hasActiveDevice), permission=\(String(describing: connection.cameraPermissionStatus)))")
        connection.startCameraSession()
        XCTAssertNil(connection.lastCaptureStartRefusal, "start was refused: \(String(describing: connection.lastCaptureStartRefusal))")
        let started = await waitUntil(timeout: 15) {
            connection.deviceSessionState == .started && connection.cameraStreamState == .streaming
        }
        XCTAssertTrue(started, "session \(connection.deviceSessionState), camera \(connection.cameraStreamState), refusal \(String(describing: connection.lastCaptureStartRefusal))")
    }

    /// A short, silent, video-only clip for `MockCameraKit.setCameraFeed`, so
    /// the mock camera really streams while Motion runs. Without a feed the
    /// Simulator's mock camera fails its stream at once ("Critical error"),
    /// which would leave "beside the camera" unproven. Four seconds of 24 fps
    /// at DAT's `.low` size, each frame a different grey. HEVC, because the
    /// mock camera reads "H.265 frames from video file". `seconds` lengthens
    /// it for a suite that streams longer than four seconds.
    static func makeCameraFeed(seconds: Int = 4) async throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("motion-probe-feed-\(UUID().uuidString).mov")
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: 360,
            AVVideoHeightKey: 640,
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: 360,
            kCVPixelBufferHeightKey as String: 640,
        ])
        writer.add(input)
        XCTAssertTrue(writer.startWriting(), "could not start the feed writer: \(String(describing: writer.error))")
        writer.startSession(atSourceTime: .zero)
        for frame in 0..<(24 * seconds) {
            while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, try XCTUnwrap(adaptor.pixelBufferPool), &buffer)
            let pixels = try XCTUnwrap(buffer)
            CVPixelBufferLockBaseAddress(pixels, [])
            memset(CVPixelBufferGetBaseAddress(pixels), Int32(frame * 2 % 256), CVPixelBufferGetDataSize(pixels))
            CVPixelBufferUnlockBaseAddress(pixels, [])
            adaptor.append(pixels, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: 24))
        }
        input.markAsFinished()
        await writer.finishWriting()
        XCTAssertEqual(writer.status, .completed, "feed writer: \(String(describing: writer.error))")
        return url
    }

    private func stopCapture(_ connection: GlassesConnection) async {
        connection.stopCameraSession()
        _ = await waitUntil(timeout: 10) { connection.captureClaim == .unclaimed }
    }

    /// The proof: with the developer switch on, Motion starts beside the camera
    /// at 30 Hz and the app receives the replayed samples at about that rate,
    /// carrying the recording's gyroscope values.
    func testWithTheSwitchOnTheAppReceivesMotionAtTheConfiguredRate() async throws {
        let glasses = try XCTUnwrap(mockGlasses)
        let recording = Self.headTurnRecording()
        // Staged first: on iOS a feed set while Motion is streaming is ignored.
        glasses.services.motion.setMotionFeed(recording)
        let feed = try await Self.makeCameraFeed()
        defer { try? FileManager.default.removeItem(at: feed) }
        glasses.services.camera.setCameraFeed(fileURL: feed)

        let connection = GlassesConnection()
        connection.motionProbeEnabled = true
        try await startCapture(connection)

        let streaming = await waitUntil(timeout: 12) { glasses.services.motion.isStreaming }
        XCTAssertTrue(streaming, "the mock glasses never reported Motion streaming (probe: \(connection.motionProbe.snapshot))")
        XCTAssertTrue(connection.isMotionAttached)

        let framesAtMotionStart = connection.frameCount
        let received = await waitUntil(timeout: 15) { connection.motionProbe.totalReceived >= 90 }
        let samples = connection.motionProbe.samples
        let framesDuringMotion = connection.frameCount - framesAtMotionStart
        let cameraState = connection.cameraStreamState
        XCTAssertTrue(received, "received \(connection.motionProbe.totalReceived) samples (probe: \(connection.motionProbe.snapshot))")
        // Beside the camera, not instead of it: frames kept arriving over the
        // same seconds the samples did, and the stream is still up.
        XCTAssertGreaterThan(framesDuringMotion, 0, "no camera frames arrived while Motion was streaming")
        XCTAssertEqual(cameraState, .streaming, "the camera stream was \(cameraState) while Motion ran")
        await stopCapture(connection)

        // The rate the app received them at, measured on the phone's clock over
        // what the probe kept. Generous bounds: this is a Simulator, and the
        // claim is "at about the configured rate", not a timing benchmark.
        var ring = MotionSampleRing(window: .infinity)
        samples.forEach { ring.append($0) }
        let arrival = try XCTUnwrap(ring.arrivalRateHz)
        XCTAssertEqual(arrival, 30, accuracy: 6, "configured 30 Hz, received \(arrival) Hz")

        // The values are the recording's, not something else's.
        let recorded = Set(recording.compactMap { $0.gyroscope.map { abs($0.y) } })
        let magnitudes = samples.compactMap(\.gyroMagnitude)
        XCTAssertEqual(magnitudes.count, samples.count, "every replayed sample carried a gyroscope reading")
        for magnitude in magnitudes {
            XCTAssertTrue(recorded.contains { abs($0 - magnitude) < 1e-4 }, "gyro |\(magnitude)| is not in the recording")
        }

        // Evidence for the report, not assertions: the device-clock rate and
        // the first timestamps, which say whether the mock re-stamps samples.
        print("[MotionProbeTest] cameraFramesDuringMotion=\(framesDuringMotion) received=\(connection.motionProbe.totalReceived) arrivalHz=\(arrival) deviceHz=\(String(describing: ring.deviceRateHz)) firstDeviceNs=\(samples.prefix(3).map(\.deviceTimestampNs)) peakGyro=\(String(describing: ring.peakGyroMagnitude))")
    }

    /// And the other half: with the switch at its default, OFF, a capture
    /// session starts with no Motion capability at all.
    func testWithTheSwitchOffNoMotionCapabilityIsStarted() async throws {
        let glasses = try XCTUnwrap(mockGlasses)
        glasses.services.motion.setMotionFeed(Self.headTurnRecording())
        let feed = try await Self.makeCameraFeed()
        defer { try? FileManager.default.removeItem(at: feed) }
        glasses.services.camera.setCameraFeed(fileURL: feed)

        let connection = GlassesConnection()
        XCTAssertFalse(connection.motionProbeEnabled, "the Motion switch must default to OFF")
        try await startCapture(connection)

        // Long enough for a Motion start to have reached the glasses: the ON
        // test sees `isStreaming` well inside this.
        let everStreamed = await waitUntil(timeout: 3) { glasses.services.motion.isStreaming }
        XCTAssertFalse(everStreamed, "a Motion capability streamed with the switch off")
        XCTAssertFalse(connection.isMotionAttached, "a Motion capability was attached with the switch off")
        XCTAssertEqual(connection.motionProbe.totalReceived, 0)
        XCTAssertEqual(connection.motionProbe.snapshot.state, "off")

        await stopCapture(connection)
    }
}
#endif

// MARK: - DAT 1.0.0's nonblocking compatibility warning

#if DEBUG
/// `dwaOutOfStuRange` is documented as nonblocking ("the session can
/// continue"), and it arrives through the same session error listener as the
/// terminal errors, which present the modal "Something went wrong". It must
/// reach the log and Developer Tools, and nothing else may change.
@MainActor
final class DATNonblockingWarningTests: XCTestCase {

    private func connection() -> GlassesConnection {
        GlassesConnection(wearables: ScriptedWearables(permissionResults: []))
    }

    func testTheOutOfRangeWarningIsLoggedNotAlerted() throws {
        let glasses = connection()
        glasses.handleSessionError(.dwaOutOfStuRange)
        XCTAssertNil(glasses.errorMessage, "a nonblocking warning raised the modal alert")
        let warning = try XCTUnwrap(glasses.datNonblockingWarning, "the warning never reached Developer Tools")
        XCTAssertEqual(warning.count, 1)
        XCTAssertEqual(warning.source, "session error listener")
        XCTAssertEqual(warning.description, DeviceSessionError.dwaOutOfStuRange.description)

        glasses.handleSessionError(.dwaOutOfStuRange)
        XCTAssertEqual(glasses.datNonblockingWarning?.count, 2, "repeats are counted, not replaced")
        XCTAssertNil(glasses.errorMessage)
    }

    /// Every other case behaves exactly as before: `errorMessage`, verbatim.
    func testEveryOtherSessionErrorStillReachesTheAlert() {
        let others: [DeviceSessionError] = [
            .noEligibleDevice, .sessionAlreadyStopped, .sessionAlreadyExists, .sessionIdle,
            .capabilityAlreadyActive, .capabilityNotFound, .unexpectedError(description: "x"),
            .thermalCritical, .thermalEmergency, .peakPowerShutdown, .batteryCritical,
            .datAppOnTheGlassesUpdateRequired, .dwaUnavailable, .insufficientSDKVersion,
        ]
        for error in others {
            let glasses = connection()
            XCTAssertFalse(GlassesConnection.isNonblockingWarning(error), "\(error) was classed as nonblocking")
            glasses.handleSessionError(error)
            XCTAssertEqual(glasses.errorMessage, error.localizedDescription, "\(error) no longer reaches the alert")
            XCTAssertNil(glasses.datNonblockingWarning, "\(error) was counted as a nonblocking warning")
        }
        XCTAssertTrue(GlassesConnection.isNonblockingWarning(DeviceSessionError.dwaOutOfStuRange))
        XCTAssertTrue(GlassesConnection.isNonblockingWarning(DeviceSessionError.dwaOutOfStuRange as any Error))
        XCTAssertFalse(GlassesConnection.isNonblockingWarning(CocoaError(.fileNoSuchFile) as any Error))
    }
}
#endif

// MARK: - IMU recorder (walk-5 evidence)

#if DEBUG
/// Shared by the IMU recorder suites: reading a log back.
enum IMULogTestSupport {
    static func temporaryBase() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("imu-rec-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func lines(_ url: URL) throws -> [[String: Any]] {
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.isEmpty || text.hasSuffix("\n"), "the log ends mid-line")
        return try text.split(separator: "\n").map { line in
            let object = try JSONSerialization.jsonObject(with: Data(line.utf8))
            return try XCTUnwrap(object as? [String: Any], "not a JSON object: \(line)")
        }
    }

    static func logFiles(under base: URL) -> [URL] {
        let folder = base.appendingPathComponent(IMURecorder.directoryName)
        return ((try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "jsonl" }
    }

    static func header(glassesModel: String? = "rayBanMeta", actualHz: Int? = 60, recorder: IMURecorder) -> IMULogHeader {
        IMULogHeader(
            sessionID: recorder.sessionID,
            fileName: recorder.relativePath,
            startMonoNs: recorder.startMonoNs,
            startWall: recorder.startWall,
            motionRequestedHz: 60,
            motionActualHz: actualHz,
            motionAttempts: [IMUMotionAttempt(hz: 60, error: nil)],
            glassesModel: glassesModel,
            captureResolution: "low",
            cameraFPSRequested: 24,
            towerTargetFPS: 12
        )
    }

    static func reading(_ i: Int, hz: Int = 60, source: IMUMotionReading.Source = .glasses) -> IMUMotionReading {
        IMUMotionReading(
            timestampNs: Int64(i) * 1_000_000_000 / Int64(hz),
            accelerometer: SIMD3(0, 9.80665, 0),
            gyroscope: SIMD3(0.01, Float(i) * 0.001, -0.02),
            orientation: SIMD4(1, 0, 0, 0),
            source: source
        )
    }

    /// Closes and waits for the file to be final.
    static func close(_ recorder: IMURecorder, reason: String = "test") -> IMURecorderSummary {
        let done = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: IMURecorderSummary?
        recorder.close(reason: reason) { summary in
            result = summary
            done.signal()
        }
        _ = done.wait(timeout: .now() + 10)
        return result!
    }

    /// A real `CMSampleBuffer` carrying `pts`, as DAT's `VideoFrame` does.
    static func sampleBuffer(pts: CMTime) throws -> CMSampleBuffer {
        var pixels: CVPixelBuffer?
        CVPixelBufferCreate(nil, 36, 64, kCVPixelFormatType_32BGRA, nil, &pixels)
        let image = try XCTUnwrap(pixels)
        var format: CMVideoFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: image, formatDescriptionOut: &format)
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: 24), presentationTimeStamp: pts, decodeTimeStamp: .invalid)
        var buffer: CMSampleBuffer?
        CMSampleBufferCreateReadyWithImageBuffer(
            allocator: nil, imageBuffer: image, formatDescription: try XCTUnwrap(format),
            sampleTiming: &timing, sampleBufferOut: &buffer
        )
        return try XCTUnwrap(buffer)
    }
}

/// The four line types, from fixed numbers.
final class IMULogFormatTests: XCTestCase {

    func testAMotionLineCarriesBothClocksAndWritesAMissingSensorAsNull() throws {
        let reading = IMUMotionReading(
            timestampNs: 1_234_567_890_123,
            accelerometer: SIMD3(0.5, 9.80665, -0.25),
            gyroscope: nil,
            orientation: SIMD4(0.9, 0.1, 0.2, 0.3),
            source: .glasses
        )
        let line = IMULogFormat.motionLine(reading, rxMonoNs: 987_654_321)
        XCTAssertTrue(line.hasPrefix(#"{"t":"m","ts_ns":1234567890123,"rx_mono_ns":987654321,"#), line)
        XCTAssertTrue(line.hasSuffix("}\n"))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
        // Written in Float's shortest exact form, which reads back as the same Float.
        XCTAssertEqual((object["a"] as? [Double])?.map { Float($0) }, [0.5, 9.80665, -0.25])
        XCTAssertTrue(object["g"] is NSNull, "a missing gyroscope must be null, never zero")
        XCTAssertEqual((object["q"] as? [Double])?.map { Float($0) }, [0.9, 0.1, 0.2, 0.3], "q is [w,x,y,z]")
        XCTAssertEqual(object["src"] as? String, "glasses")
    }

    func testAFrameLineCarriesPTSEpochReceiptSentAndSeq() throws {
        let epoch = UUID()
        let line = IMULogFormat.frameLine(IMUFrameStamp(ptsUs: 41_666, rxMonoNs: 5_000), epoch: epoch, sent: true, seq: 7)
        XCTAssertEqual(line, #"{"t":"f","pts_us":41666,"epoch":"\#(epoch.uuidString)","rx_mono_ns":5000,"sent":true,"seq":7}"# + "\n")
        let missing = IMULogFormat.frameLine(IMUFrameStamp(ptsUs: nil, rxMonoNs: 1), epoch: nil, sent: false, seq: 8)
        XCTAssertEqual(missing, #"{"t":"f","pts_us":null,"epoch":null,"rx_mono_ns":1,"sent":false,"seq":8}"# + "\n")
    }

    /// The PTS is `CMSampleBufferGetPresentationTimeStamp`, in microseconds.
    func testTheFrameStampReadsTheSampleBuffersPTSInMicroseconds() throws {
        let buffer = try IMULogTestSupport.sampleBuffer(pts: CMTime(value: 12_345, timescale: 24))
        let stamp = IMUFrameStamp(sampleBuffer: buffer, rxMonoNs: 42)
        XCTAssertEqual(stamp.ptsUs, 514_375_000, "12345/24 s")
        XCTAssertEqual(stamp.rxMonoNs, 42)
        XCTAssertEqual(IMUFrameStamp.microseconds(CMTime(value: 1_500, timescale: 1_000_000_000)), 2, "rounded, not truncated")
        XCTAssertNil(IMUFrameStamp.microseconds(.invalid))
        XCTAssertNil(IMUFrameStamp.microseconds(.indefinite))
    }

    func testStringsAreEscapedAndNonFiniteNumbersAreNull() throws {
        let line = IMULogFormat.eventLine("x", monoNs: 1, fields: [
            ("s", .string("a \"quoted\"\nline\\ \u{01}")),
            ("nan", .float(.nan)),
            ("inf", .double(.infinity)),
        ])
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
        XCTAssertEqual(object["s"] as? String, "a \"quoted\"\nline\\ \u{01}")
        XCTAssertTrue(object["nan"] is NSNull)
        XCTAssertTrue(object["inf"] is NSNull)
        XCTAssertEqual(object["ev"] as? String, "x")
        XCTAssertEqual(object["mono_ns"] as? Int, 1)
    }

    /// The header names the build, the SDK, both Motion rates and both clocks
    /// at start -- and the SDK version is read from the linked framework, and
    /// agrees with the project's pin.
    func testTheHeaderRecordsBuildSDKRatesAndClocks() throws {
        var header = IMULogHeader(
            sessionID: UUID(), fileName: "imu-logs/x.jsonl", startMonoNs: 123, startWall: Date(timeIntervalSince1970: 1_790_000_000.25),
            motionRequestedHz: 60, motionActualHz: 30,
            motionAttempts: [IMUMotionAttempt(hz: 60, error: "refused"), IMUMotionAttempt(hz: 30, error: nil)],
            glassesModel: "rayBanMeta", captureResolution: "low", cameraFPSRequested: 24, towerTargetFPS: 12
        )
        header.build = IMULogHeader.BuildInfo(
            appBuildSHA: "abc1234", appVersion: "1.0 (1)", appBinaryModified: nil,
            datSDKVersion: "1.0.0", datMotionVersion: "1.0.0", phoneModel: "iPhone17,1", phoneOS: "Version 26.5"
        )
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(header.line().utf8)) as? [String: Any])
        XCTAssertEqual(object["t"] as? String, "header")
        XCTAssertEqual(object["app_build_sha"] as? String, "abc1234")
        XCTAssertEqual(object["dat_sdk_version"] as? String, "1.0.0")
        XCTAssertEqual(object["motion_rate_requested_hz"] as? Int, 60)
        XCTAssertEqual(object["motion_rate_actual_hz"] as? Int, 30)
        XCTAssertEqual((object["motion_rate_attempts"] as? [[String: Any]])?.first?["error"] as? String, "refused")
        XCTAssertEqual(object["phone_model"] as? String, "iPhone17,1")
        XCTAssertEqual(object["start_mono_ns"] as? Int, 123)
        XCTAssertEqual(object["start_wall_unix_ms"] as? Int, 1_790_000_000_250)
        XCTAssertNotNil(object["start_wall_iso"] as? String)
        XCTAssertNotNil(object["seq_semantics"] as? String)

        // The live values, as a phone would write them.
        let live = IMULogHeader.BuildInfo.current
        XCTAssertEqual(live.datSDKVersion, "1.0.0", "read from MWDATCore's own Info.plist")
        XCTAssertEqual(live.datMotionVersion, "1.0.0")
        XCTAssertNotNil(live.phoneModel)
        let project = try String(
            contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Glasses.xcodeproj/project.pbxproj"),
            encoding: .utf8
        )
        XCTAssertTrue(project.contains("version = \(live.datSDKVersion ?? "?");"), "the linked DAT is not the pinned one")
    }

    func testTheRecordingIndicatorFollowsTheStatus() {
        var status = IMURecorderStatus()
        XCTAssertEqual(IMURecordingIndicator(status), .off)
        status.phase = .recording
        XCTAssertEqual(IMURecordingIndicator(status), .writing)
        status.capReason = "duration"
        XCTAssertEqual(IMURecordingIndicator(status), .stopped("duration cap"))
        status.phase = .failed("disk")
        XCTAssertEqual(IMURecordingIndicator(status), .stopped("error"))
        status.phase = .closed
        XCTAssertEqual(IMURecordingIndicator(status), .off)
    }

    @MainActor
    func testTheProtectedDataNamesAreUIKits() {
        XCTAssertEqual(IMURecorder.protectedDataWillBecomeUnavailable, UIApplication.protectedDataWillBecomeUnavailableNotification)
        XCTAssertEqual(IMURecorder.protectedDataDidBecomeAvailable, UIApplication.protectedDataDidBecomeAvailableNotification)
    }
}

/// The recorder against a real file in a temporary directory: ordering, caps,
/// epochs, the locked phone, and where the I/O runs.
final class IMURecorderFileTests: XCTestCase {

    private var base: URL!

    override func setUpWithError() throws {
        base = try IMULogTestSupport.temporaryBase()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: base)
    }

    func testTheHeaderIsFirstOnDiskEvenWhenSamplesArriveBeforeIt() throws {
        let recorder = IMURecorder(baseDirectory: base)
        // Samples before the header, as Motion can deliver before `open`.
        for i in 0..<5 { recorder.recordMotion(IMULogTestSupport.reading(i), rxMonoNs: UInt64(1_000 + i)) }
        recorder.noteCameraStart(reason: "stream.start()")
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        for seq in 1...10 {
            recorder.recordFrame(IMUFrameStamp(ptsUs: Int64(seq) * 41_667, rxMonoNs: UInt64(seq)), seq: seq, sent: seq % 2 == 1)
        }
        let summary = IMULogTestSupport.close(recorder)
        let url = try XCTUnwrap(summary.fileURL)
        XCTAssertTrue(url.lastPathComponent.hasSuffix("-\(recorder.sessionID.uuidString.prefix(8)).jsonl"))
        XCTAssertEqual(url.deletingLastPathComponent().lastPathComponent, "imu-logs")

        let lines = try IMULogTestSupport.lines(url)
        XCTAssertEqual(lines.first?["t"] as? String, "header")
        XCTAssertEqual(lines.filter { $0["t"] as? String == "header" }.count, 1)
        XCTAssertEqual(lines.filter { $0["t"] as? String == "m" }.count, 5)
        let frames = lines.filter { $0["t"] as? String == "f" }
        XCTAssertEqual(frames.compactMap { $0["seq"] as? Int }, Array(1...10))
        XCTAssertEqual(frames.filter { $0["sent"] as? Bool == true }.count, 5)
        XCTAssertEqual(lines.last?["ev"] as? String, "close")
        XCTAssertEqual(summary.counts.frames, 10)
        XCTAssertEqual(summary.counts.framesSent, 5)
        XCTAssertEqual(summary.bytesHeld, 0)
        XCTAssertEqual(summary.bytesWritten, try Data(contentsOf: url).count)
    }

    /// Every file operation ran on the recorder's own queue and none on the
    /// main thread, although every call into the recorder came from it.
    func testEveryFileOperationRunsOnTheRecordersQueueNotTheMainThread() throws {
        XCTAssertTrue(Thread.isMainThread, "this test drives the recorder from the main thread on purpose")
        let recorder = IMURecorder(baseDirectory: base, limits: {
            var limits = IMURecorder.Limits()
            limits.flushThresholdBytes = 512
            return limits
        }())
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        recorder.noteCameraStart(reason: "test")
        for i in 0..<400 {
            recorder.recordMotion(IMULogTestSupport.reading(i), rxMonoNs: UInt64(i))
            if i % 3 == 0 { recorder.recordFrame(IMUFrameStamp(ptsUs: Int64(i) * 13_889, rxMonoNs: UInt64(i)), seq: i / 3 + 1, sent: true) }
        }
        let summary = IMULogTestSupport.close(recorder)
        XCTAssertGreaterThan(summary.io.operations, 10, "the threshold flushes did not happen")
        XCTAssertEqual(summary.io.onMainThread, 0, "file I/O ran on the main thread")
        XCTAssertEqual(summary.io.offQueue, 0, "file I/O ran off the recorder's queue")
        XCTAssertNil(summary.failure)
    }

    /// Private by construction: complete protection, and never in a backup.
    func testTheFileIsProtectedAndExcludedFromBackup() throws {
        let recorder = IMURecorder(baseDirectory: base)
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        let url = try XCTUnwrap(IMULogTestSupport.close(recorder).fileURL)
        XCTAssertEqual(try url.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup, true)
        XCTAssertEqual(
            try url.deletingLastPathComponent().resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup,
            true
        )
        let protection = try FileManager.default.attributesOfItem(atPath: url.path)[.protectionKey] as? FileProtectionType
        // The Simulator does not always report a class; when it does, it must be this one.
        if let protection { XCTAssertEqual(protection, .complete) }
    }

    func testSamplesFromAnotherSourceAreCountedNotWritten() throws {
        let recorder = IMURecorder(baseDirectory: base)
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        recorder.recordMotion(IMULogTestSupport.reading(0), rxMonoNs: 1)
        recorder.recordMotion(IMULogTestSupport.reading(1, source: .neuralBand), rxMonoNs: 2)
        recorder.recordMotion(IMULogTestSupport.reading(2, source: .unknown), rxMonoNs: 3)
        recorder.recordMotion(IMULogTestSupport.reading(3, source: .neuralBand), rxMonoNs: 4)
        let summary = IMULogTestSupport.close(recorder)
        let lines = try IMULogTestSupport.lines(try XCTUnwrap(summary.fileURL))
        XCTAssertEqual(lines.filter { $0["t"] as? String == "m" }.count, 1)
        XCTAssertEqual(summary.counts.motionOtherBySource, ["neural_band": 2, "unknown": 1])
        XCTAssertEqual(lines.filter { $0["ev"] as? String == "first_non_glasses_sample" }.count, 1)
        let counts = try XCTUnwrap(lines.last { $0["ev"] as? String == "counts" })
        XCTAssertEqual(counts["m_other"] as? Int, 3)
    }

    func testAGapOnTheGlassesClockIsCountedAsMissingSamples() throws {
        let recorder = IMURecorder(baseDirectory: base)
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        recorder.noteMotionStart(hz: 60, requestedHz: 60, attempts: [])
        for i in [0, 1, 2, 6, 7] { recorder.recordMotion(IMULogTestSupport.reading(i), rxMonoNs: UInt64(i)) }
        let summary = IMULogTestSupport.close(recorder)
        XCTAssertEqual(summary.counts.motionGapEvents, 1)
        XCTAssertEqual(summary.counts.motionEstimatedMissing, 3)
    }

    /// 200 MB in the app; 8 KB here. Data lines stop at the cap, it is logged,
    /// and the closing lines still land.
    func testTheSizeCapStopsDataLinesAndSaysSo() throws {
        var limits = IMURecorder.Limits()
        limits.maxBytes = 8 * 1024
        limits.eventReserveBytes = 2 * 1024
        let recorder = IMURecorder(baseDirectory: base, limits: limits)
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        for i in 0..<500 { recorder.recordMotion(IMULogTestSupport.reading(i), rxMonoNs: UInt64(i)) }
        let summary = IMULogTestSupport.close(recorder)
        let url = try XCTUnwrap(summary.fileURL)
        let size = try Data(contentsOf: url).count
        XCTAssertLessThanOrEqual(size, limits.maxBytes)
        XCTAssertEqual(summary.capReason, "size")
        XCTAssertGreaterThan(summary.counts.droppedAfterCap, 0)
        let lines = try IMULogTestSupport.lines(url)
        let cap = try XCTUnwrap(lines.first { $0["ev"] as? String == "cap_reached" })
        XCTAssertEqual(cap["cap"] as? String, "size")
        XCTAssertEqual(lines.last?["ev"] as? String, "close", "the closing line was lost to the cap")
        XCTAssertLessThan(lines.filter { $0["t"] as? String == "m" }.count, 500)
    }

    /// 30 minutes in the app; a fraction of a second here.
    func testTheDurationCapStopsDataLinesAndSaysSo() throws {
        var limits = IMURecorder.Limits()
        limits.maxDuration = 0.2
        let recorder = IMURecorder(baseDirectory: base, limits: limits)
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        recorder.recordMotion(IMULogTestSupport.reading(0), rxMonoNs: 1)
        recorder.waitUntilIdle()
        Thread.sleep(forTimeInterval: 0.3)
        recorder.recordMotion(IMULogTestSupport.reading(1), rxMonoNs: 2)
        recorder.recordFrame(IMUFrameStamp(ptsUs: 1, rxMonoNs: 3), seq: 1, sent: true)
        let summary = IMULogTestSupport.close(recorder)
        XCTAssertEqual(summary.capReason, "duration")
        XCTAssertEqual(summary.counts.motionGlasses, 1)
        XCTAssertEqual(summary.counts.frames, 0)
        let lines = try IMULogTestSupport.lines(try XCTUnwrap(summary.fileURL))
        XCTAssertEqual(lines.first { $0["ev"] as? String == "cap_reached" }?["cap"] as? String, "duration")
    }

    /// 1 GB for the folder in the app. A file opens with whatever budget is
    /// left, and not at all when there is none.
    func testTheFolderCapBoundsANewFileAndRefusesOneWithNoRoom() throws {
        let folder = base.appendingPathComponent(IMURecorder.directoryName)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data(count: 20_000).write(to: folder.appendingPathComponent("older.jsonl"))

        var limits = IMURecorder.Limits()
        limits.folderMaxBytes = 20_000 + 12 * 1024
        limits.eventReserveBytes = 2 * 1024
        let recorder = IMURecorder(baseDirectory: base, limits: limits)
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        for i in 0..<500 { recorder.recordMotion(IMULogTestSupport.reading(i), rxMonoNs: UInt64(i)) }
        let summary = IMULogTestSupport.close(recorder)
        XCTAssertEqual(summary.capReason, "folder")
        XCTAssertEqual(summary.effectiveMaxBytes, 12 * 1024)
        XCTAssertLessThanOrEqual(IMULogStore.totalBytes(in: folder), limits.folderMaxBytes)

        limits.folderMaxBytes = IMULogStore.totalBytes(in: folder) + 1_000
        let refused = IMURecorder(baseDirectory: base, limits: limits)
        refused.open(header: IMULogTestSupport.header(recorder: refused))
        refused.recordMotion(IMULogTestSupport.reading(0), rxMonoNs: 1)
        let refusal = IMULogTestSupport.close(refused)
        XCTAssertNotNil(refusal.failure, "a full folder must refuse the new file")
        XCTAssertNil(refusal.fileURL)
        XCTAssertEqual(IMULogStore.logFiles(in: folder).count, 2, "a file was created in a full folder")
    }

    /// A fresh epoch at each camera start, and at a return to streaming after a
    /// pause; frames carry the epoch they arrived in.
    func testTheEpochChangesAtEveryCameraStartAndResume() throws {
        let recorder = IMURecorder(baseDirectory: base)
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        var seq = 0
        func frames(_ n: Int) {
            for _ in 0..<n {
                seq += 1
                recorder.recordFrame(IMUFrameStamp(ptsUs: Int64(seq) * 41_667, rxMonoNs: UInt64(seq)), seq: seq, sent: false)
            }
        }
        recorder.noteCameraStart(reason: "stream.start()")
        recorder.noteStreamState("starting")
        recorder.noteStreamState("streaming")   // the first streaming is not a restart
        frames(3)
        recorder.noteStreamState("paused")
        recorder.noteStreamState("streaming")   // resume: a new epoch
        frames(3)
        recorder.noteCameraStart(reason: "stream.start()")   // a camera restart
        frames(3)
        let summary = IMULogTestSupport.close(recorder)
        XCTAssertEqual(summary.epochs, 3)
        let lines = try IMULogTestSupport.lines(try XCTUnwrap(summary.fileURL))
        let starts = lines.filter { $0["ev"] as? String == "camera_start" }
        XCTAssertEqual(starts.count, 3)
        let epochs = starts.compactMap { $0["epoch"] as? String }
        XCTAssertEqual(Set(epochs).count, 3, "an epoch was reused")
        let frameEpochs = lines.filter { $0["t"] as? String == "f" }.compactMap { $0["epoch"] as? String }
        XCTAssertEqual(frameEpochs, [String](repeating: epochs[0], count: 3) + [String](repeating: epochs[1], count: 3) + [String](repeating: epochs[2], count: 3))
        XCTAssertEqual(starts[1]["reason"] as? String, "resume from paused")
        XCTAssertEqual(starts[1]["prev_epoch"] as? String, epochs[0])
    }

    func testAPTSThatRunsBackwardsIsLoggedAsADiscontinuity() throws {
        let recorder = IMURecorder(baseDirectory: base)
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        recorder.noteCameraStart(reason: "test")
        for (seq, pts) in [(1, 100_000), (2, 141_667), (3, 50_000), (4, 91_667)] {
            recorder.recordFrame(IMUFrameStamp(ptsUs: Int64(pts), rxMonoNs: UInt64(seq)), seq: seq, sent: true)
        }
        let summary = IMULogTestSupport.close(recorder)
        XCTAssertEqual(summary.counts.ptsDiscontinuities, 1)
        let lines = try IMULogTestSupport.lines(try XCTUnwrap(summary.fileURL))
        XCTAssertEqual(lines.first { $0["ev"] as? String == "pts_discontinuity" }?["seq"] as? Int, 3)
    }

    /// `.complete` protection makes the file unwritable while the phone is
    /// locked. Lines are held in memory then, written at the unlock, and a close
    /// asked for while locked waits for it rather than dropping them.
    func testALockedPhoneHoldsLinesAndWritesThemAtTheUnlock() throws {
        let recorder = IMURecorder(baseDirectory: base)
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        for i in 0..<10 { recorder.recordMotion(IMULogTestSupport.reading(i), rxMonoNs: UInt64(i)) }
        recorder.protectedDataWillBecomeUnavailable()
        for i in 10..<30 { recorder.recordMotion(IMULogTestSupport.reading(i), rxMonoNs: UInt64(i)) }

        let closed = expectation(description: "closed after the unlock")
        recorder.close(reason: "locked") { _ in closed.fulfill() }
        recorder.waitUntilIdle()
        let whileLocked = recorder.summary()
        XCTAssertFalse(whileLocked.closed, "a close while locked must wait for the unlock")
        XCTAssertGreaterThan(whileLocked.bytesHeld, 0)
        let url = try XCTUnwrap(whileLocked.fileURL)
        XCTAssertEqual(try IMULogTestSupport.lines(url).filter { $0["t"] as? String == "m" }.count, 10,
                       "lines were written while the file was locked")

        recorder.protectedDataDidBecomeAvailable()
        wait(for: [closed], timeout: 5)
        let lines = try IMULogTestSupport.lines(url)
        XCTAssertEqual(lines.filter { $0["t"] as? String == "m" }.count, 30)
        XCTAssertNotNil(lines.first { $0["ev"] as? String == "protected_data_unavailable" })
        XCTAssertNotNil(lines.first { $0["ev"] as? String == "protected_data_available" })
        XCTAssertEqual(lines.last?["ev"] as? String, "close")
        XCTAssertEqual(recorder.summary().bytesHeld, 0)
    }

    /// The camera path's cost. What the frame listener adds per frame is a
    /// PTS read and a clock read on DAT's thread, and on the main actor one
    /// `recordFrame` -- a single enqueue. Both are measured here, from the main
    /// thread, against a real `CMSampleBuffer`.
    func testTheRecordersPerFrameCostOnTheCallingThread() throws {
        let recorder = IMURecorder(baseDirectory: base)
        recorder.open(header: IMULogTestSupport.header(recorder: recorder))
        recorder.noteCameraStart(reason: "bench")
        recorder.waitUntilIdle()
        let buffer = try IMULogTestSupport.sampleBuffer(pts: CMTime(value: 1, timescale: 24))
        let n = 20_000

        var stamps: [IMUFrameStamp] = []
        stamps.reserveCapacity(n)
        let stampStart = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<n {
            stamps.append(IMUFrameStamp(sampleBuffer: buffer, rxMonoNs: DispatchTime.now().uptimeNanoseconds))
        }
        let stampNs = Double(DispatchTime.now().uptimeNanoseconds - stampStart) / Double(n)

        var worst: UInt64 = 0
        var perCall: [UInt64] = []
        perCall.reserveCapacity(n)
        let recordStart = DispatchTime.now().uptimeNanoseconds
        for (i, stamp) in stamps.enumerated() {
            let t0 = DispatchTime.now().uptimeNanoseconds
            recorder.recordFrame(stamp, seq: i + 1, sent: i % 2 == 0)
            let dt = DispatchTime.now().uptimeNanoseconds - t0
            perCall.append(dt)
            worst = max(worst, dt)
        }
        let recordNs = Double(DispatchTime.now().uptimeNanoseconds - recordStart) / Double(n)
        perCall.sort()
        let p99 = perCall[n * 99 / 100]

        // And what the recorder's own queue spends per frame line, off the main thread.
        let drainStart = DispatchTime.now().uptimeNanoseconds
        recorder.waitUntilIdle()
        let drainNs = Double(DispatchTime.now().uptimeNanoseconds - drainStart)
        let summary = IMULogTestSupport.close(recorder)
        XCTAssertEqual(summary.counts.frames, n)

        print(String(format: "[IMURecBench] per frame: stamp on DAT's thread %.0f ns; recordFrame on the calling thread mean %.0f ns, p99 %llu ns, worst %llu ns; queue drain after the loop %.1f ms", stampNs, recordNs, p99, worst, drainNs / 1_000_000))
        // A frame arrives every 41.7 ms. Generous bounds for a Debug build on a
        // shared Simulator host; the printed numbers are the report.
        XCTAssertLessThan(stampNs, 5_000)
        XCTAssertLessThan(recordNs, 20_000)
    }
}

/// The purge: local, confirmed in the UI, and refused while a log is open.
@MainActor
final class IMULogPurgeTests: XCTestCase {

    func testThePurgeDeletesEveryLogAndNothingElse() async throws {
        let base = try IMULogTestSupport.temporaryBase()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appendingPathComponent(IMURecorder.directoryName)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for i in 0..<3 { try Data(count: 1_000 * (i + 1)).write(to: folder.appendingPathComponent("\(i).jsonl")) }
        let bystander = base.appendingPathComponent("not-a-log.txt")
        try Data("keep".utf8).write(to: bystander)

        let glasses = GlassesConnection(wearables: ScriptedWearables(permissionResults: []))
        glasses.imuLogBaseDirectory = base
        let deleted = await glasses.purgeIMULogs()
        XCTAssertEqual(deleted, IMULogInventory(files: 3, bytes: 6_000))
        XCTAssertEqual(glasses.imuLogInventory, IMULogInventory(files: 0, bytes: 0))
        XCTAssertTrue(IMULogStore.logFiles(in: folder).isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: bystander.path), "the purge reached outside imu-logs")
    }
}

/// End to end on Mock Device Kit: a real `DeviceSession`, a real `Camera` and
/// `Motion`, the real `GlassesConnection`, and -- in the first test -- the real
/// `ProjectManager` bridge to a `TowerClient` on a local mock Tower, so the
/// log's `seq` is compared with what actually went out on the wire.
@MainActor
final class IMURecorderMockDeviceTests: XCTestCase {

    private var mockGlasses: (any MockGlasses)?
    private var base: URL!

    override func setUp() async throws {
        base = try IMULogTestSupport.temporaryBase()
        MockDeviceKit.shared.enable()
        let glasses = try MockDeviceKit.shared.pairGlasses(model: .rayBanMeta)
        glasses.powerOn()
        glasses.don()
        mockGlasses = glasses
    }

    override func tearDown() async throws {
        if let mockGlasses {
            await MockDeviceKit.shared.unpairDevice(mockGlasses)
        }
        await MockDeviceKit.shared.disable()
        mockGlasses = nil
        try? FileManager.default.removeItem(at: base)
    }

    /// A 60 Hz recording: mock replay follows the configured rate, so a
    /// recording at any other rate would put the glasses' clock at that rate.
    static func recording(hz: Double = 60, seconds: Double = 20) -> [MWDATMockDevice.MotionSample] {
        let gravity = mockMotionGravityMetersPerSecondSquared
        return (0..<Int(hz * seconds)).map { i in
            let t = Double(i) / hz
            let yawRate = Float(2 * sin(t * .pi / 2))
            return MWDATMockDevice.MotionSample(
                timestampNs: Int64((t * 1_000_000_000).rounded()),
                accelerometer: MWDATMockDevice.Vector3(x: 0, y: gravity, z: 0),
                gyroscope: MWDATMockDevice.Vector3(x: 0, y: yawRate, z: 0),
                source: .glasses
            )
        }
    }

    private func waitUntil(timeout: TimeInterval, _ condition: @MainActor () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return condition()
    }

    private func startCapture(_ connection: GlassesConnection) async throws {
        let ready = await waitUntil(timeout: 15) {
            connection.hasActiveDevice && connection.cameraPermissionStatus == .granted
        }
        XCTAssertTrue(ready, "mock glasses never became an active device with camera permission")
        connection.startCameraSession()
        XCTAssertNil(connection.lastCaptureStartRefusal)
        let started = await waitUntil(timeout: 15) {
            connection.deviceSessionState == .started && connection.cameraStreamState == .streaming
        }
        XCTAssertTrue(started, "session \(connection.deviceSessionState), camera \(connection.cameraStreamState)")
    }

    /// Stops capture and returns the closed log's summary.
    private func stopCapture(_ connection: GlassesConnection, recorder: IMURecorder?) async -> IMURecorderSummary? {
        connection.stopCameraSession()
        _ = await waitUntil(timeout: 10) { connection.captureClaim == .unclaimed }
        guard let recorder else { return nil }
        _ = await waitUntil(timeout: 10) { recorder.summary().closed }
        return recorder.summary()
    }

    private func streamingConnection(feedSeconds: Int = 12, motionFeed: Bool = true) async throws -> (GlassesConnection, URL) {
        let glasses = try XCTUnwrap(mockGlasses)
        if motionFeed { glasses.services.motion.setMotionFeed(Self.recording()) }
        let feed = try await MotionProbeMockDeviceTests.makeCameraFeed(seconds: feedSeconds)
        glasses.services.camera.setCameraFeed(fileURL: feed)
        let connection = GlassesConnection()
        connection.imuLogBaseDirectory = base
        return (connection, feed)
    }

    private static func rateHz(_ values: [Int]) -> Double? {
        guard let first = values.first, let last = values.last, values.count >= 2, last > first else { return nil }
        return Double(values.count - 1) / (Double(last - first) / 1_000_000_000)
    }

    /// The proof. Switch on: one file, the header first and saying 60 Hz, `m`
    /// lines at about 60 Hz by both clocks, an `f` line for every frame the
    /// phone received with rising `pts_us` and `sent` exactly where the frame
    /// was forwarded -- and every frame the mock Tower received carries a `seq`
    /// the log marks as sent.
    func testWithTheSwitchOnTheLogHasHeaderMotionAt60HzAndEveryFrame() async throws {
        let (connection, feed) = try await streamingConnection()
        defer { try? FileManager.default.removeItem(at: feed) }

        // The mock Tower, and the app's own bridge to it.
        let server = try MockTowerServer()
        let port = try await server.start()
        let wire = MessageRecorder()
        server.onText = { text in
            wire.record(text)
            if text.contains(#""type":"ping""#) { server.send(text: #"{"type":"pong"}"#) }
        }
        defer { server.stop() }
        let tower = TowerClient()
        let project = ProjectManager(glassesConnection: connection, towerClient: tower)
        tower.connect(to: URL(string: "ws://127.0.0.1:\(port)/"))
        let online = await waitUntil(timeout: 10) { tower.status == .online }
        XCTAssertTrue(online, "the mock Tower never came online (\(tower.status))")

        // Every frame this connection forwarded, as `ProjectManager` saw it.
        var forwarded: [Int] = []
        let sink = connection.$latestCapturedFrame.compactMap { $0?.sequence }.sink { forwarded.append($0) }
        defer { sink.cancel() }

        connection.imuRecorderEnabled = true
        try await startCapture(connection)
        let recorder = try XCTUnwrap(connection.imuRecorder, "the switch was on and no recorder was made")
        XCTAssertEqual(connection.motionConfiguredRate?.hertz, 60)
        let enough = await waitUntil(timeout: 20) {
            let counts = recorder.summary().counts
            return counts.motionGlasses >= 240 && counts.frames >= 72
                && wire.all.filter { $0.contains(#""type":"frame""#) }.count >= 10
        }
        XCTAssertTrue(enough, "not enough data: \(recorder.summary().counts), tower frames \(wire.all.filter { $0.contains(#""type":"frame""#) }.count)")
        let badgeWhileRecording = connection.imuRecorderReadout.indicator
        _ = await stopCapture(connection, recorder: recorder)
        // Frames DAT delivered before the stop can still be queued for the main
        // actor; each reaches the closed recorder and is counted there. Wait for
        // the count to settle, then read the recorder, whose read drains its
        // queue behind every enqueue already made.
        var framesReceived = connection.frameCount
        var settled = false
        while !settled {
            try await Task.sleep(nanoseconds: 300_000_000)
            settled = connection.frameCount == framesReceived
            framesReceived = connection.frameCount
        }
        let summary = recorder.summary()
        withExtendedLifetime(project) {}

        XCTAssertEqual(badgeWhileRecording, .writing, "no recording badge while the log was written")
        let files = IMULogTestSupport.logFiles(under: base)
        XCTAssertEqual(files.count, 1, "one file per capture session")
        let url = try XCTUnwrap(summary.fileURL)
        XCTAssertEqual(files.first?.lastPathComponent, url.lastPathComponent)
        XCTAssertEqual(summary.io.onMainThread, 0)
        XCTAssertNil(summary.failure)
        let lines = try IMULogTestSupport.lines(url)

        // Header.
        let header = try XCTUnwrap(lines.first)
        XCTAssertEqual(header["t"] as? String, "header")
        XCTAssertEqual(header["motion_rate_requested_hz"] as? Int, 60)
        XCTAssertEqual(header["motion_rate_actual_hz"] as? Int, 60)
        XCTAssertEqual(header["dat_sdk_version"] as? String, "1.0.0")
        XCTAssertEqual(header["glasses_model"] as? String, "Ray-Ban Meta")
        XCTAssertNotNil(header["start_mono_ns"] as? Int)
        XCTAssertNotNil(header["start_wall_unix_ms"] as? Int)
        XCTAssertNotNil(header["phone_model"] as? String)

        // Motion, at about the configured rate by the glasses' clock and by
        // the phone's.
        let motion = lines.filter { $0["t"] as? String == "m" }
        XCTAssertGreaterThanOrEqual(motion.count, 240)
        XCTAssertTrue(motion.allSatisfy { $0["src"] as? String == "glasses" })
        let deviceHz = try XCTUnwrap(Self.rateHz(motion.compactMap { $0["ts_ns"] as? Int }))
        let arrivalHz = try XCTUnwrap(Self.rateHz(motion.compactMap { $0["rx_mono_ns"] as? Int }))
        XCTAssertEqual(deviceHz, 60, accuracy: 3, "configured 60 Hz, the glasses' clock says \(deviceHz)")
        XCTAssertEqual(arrivalHz, 60, accuracy: 12, "configured 60 Hz, received at \(arrivalHz)")
        XCTAssertTrue(motion.allSatisfy { ($0["g"] as? [Double])?.count == 3 && ($0["a"] as? [Double])?.count == 3 })

        // Frames: every one, in order, with a rising PTS in one epoch.
        let frames = lines.filter { $0["t"] as? String == "f" }
        let seqs = frames.compactMap { $0["seq"] as? Int }
        XCTAssertEqual(seqs, Array(1...seqs.count), "an f line is missing or out of order")
        XCTAssertEqual(seqs.count + summary.counts.framesAfterClose, framesReceived,
                       "every received frame is either logged or counted as arriving after the close")
        let pts = frames.compactMap { $0["pts_us"] as? Int }
        XCTAssertEqual(pts.count, frames.count, "a frame without a PTS")
        let discontinuities = lines.filter { $0["ev"] as? String == "pts_discontinuity" }.compactMap { $0["seq"] as? Int }
        for i in 1..<pts.count where !discontinuities.contains(seqs[i]) {
            XCTAssertGreaterThan(pts[i], pts[i - 1], "pts_us did not rise at seq \(seqs[i])")
        }
        let startEpoch = lines.first { $0["ev"] as? String == "camera_start" }?["epoch"] as? String
        XCTAssertNotNil(startEpoch)
        XCTAssertTrue(frames.allSatisfy { $0["epoch"] as? String == startEpoch })

        // `sent` is true exactly for the frames handed to the Tower client.
        let sent = Set(frames.filter { $0["sent"] as? Bool == true }.compactMap { $0["seq"] as? Int })
        XCTAssertEqual(sent, Set(forwarded).intersection(Set(seqs)), "sent flags disagree with what was forwarded")
        XCTAssertGreaterThan(sent.count, 0)
        XCTAssertLessThan(sent.count, seqs.count, "the 12 fps gate forwarded every frame of a 24 fps stream")

        // And the wire: every frame the Tower received carries a `seq` the log
        // marks as sent, and no `source_seq`, so the Tower's `frames.py` stores
        // `source_seq = seq` -- the log's `seq`.
        let towerFrames = wire.all.compactMap { text -> [String: Any]? in
            guard text.contains(#""type":"frame""#) else { return nil }
            return try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        }
        XCTAssertGreaterThanOrEqual(towerFrames.count, 10)
        for message in towerFrames {
            let seq = try XCTUnwrap(message["seq"] as? Int)
            XCTAssertNil(message["source_seq"], "the app now sends source_seq; the Tower would no longer fall back to seq")
            XCTAssertTrue(sent.contains(seq), "the Tower received seq \(seq), which the log does not mark as sent")
        }
        print("[IMURecTest] 60 Hz: m=\(motion.count) deviceHz=\(deviceHz) arrivalHz=\(arrivalHz) f=\(frames.count) sent=\(sent.count) towerFrames=\(towerFrames.count) towerSeqs=\(towerFrames.compactMap { $0["seq"] as? Int }.prefix(8)) payloadKeys=\(towerFrames.first.map { $0.keys.sorted() } ?? [])")

        // Events.
        let events = lines.filter { $0["t"] as? String == "ev" }.compactMap { $0["ev"] as? String }
        for name in ["camera_start", "stream_state", "session_state", "motion_start", "motion_state", "camera_stop", "counts", "close"] {
            XCTAssertTrue(events.contains(name), "no \(name) event")
        }
        XCTAssertEqual(lines.first { $0["ev"] as? String == "motion_start" }?["rate_hz"] as? Int, 60)
        let badgeAfter = await waitUntil(timeout: 3) { connection.imuRecorderReadout.indicator == .off }
        XCTAssertTrue(badgeAfter, "the recording badge outlived the log")

        // Opt-in, for checking the Mac-side join against a log this app really
        // wrote: `TEST_RUNNER_IMU_TEST_KEEP_DIR=<dir>` keeps the log and writes
        // the frames the mock Tower received as a Tower-shaped frames.jsonl.
        if let keep = ProcessInfo.processInfo.environment["IMU_TEST_KEEP_DIR"], !keep.isEmpty {
            let dir = URL(fileURLWithPath: keep, isDirectory: true)
            let capture = dir.appendingPathComponent("capture", isDirectory: true)
            try FileManager.default.createDirectory(at: capture, withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: dir.appendingPathComponent(url.lastPathComponent))
            try FileManager.default.copyItem(at: url, to: dir.appendingPathComponent(url.lastPathComponent))
            let now = Date().timeIntervalSince1970
            let journal = towerFrames.map { message -> String in
                let seq = message["seq"] as? Int ?? -1
                return #"{"schema_version":1,"source_seq":\#(seq),"wire_seq":\#(seq),"tx_seq":\#(message["tx_seq"] as? Int ?? -1),"received_at":\#(now),"time_basis":"tower-receipt","relpath":"frames/\#(String(format: "%08d", seq)).jpg"}"#
            }.joined(separator: "\n") + "\n"
            try journal.write(to: capture.appendingPathComponent("frames.jsonl"), atomically: true, encoding: .utf8)
        }
    }

    /// Switch off, the default: no file, and no Motion capability at all.
    func testWithTheSwitchOffThereIsNoFileAndNoMotion() async throws {
        let (connection, feed) = try await streamingConnection(feedSeconds: 6)
        defer { try? FileManager.default.removeItem(at: feed) }
        XCTAssertFalse(connection.imuRecorderEnabled, "the IMU log switch must default to OFF")
        XCTAssertFalse(connection.motionProbeEnabled)
        try await startCapture(connection)
        let everStreamed = await waitUntil(timeout: 3) { self.mockGlasses?.services.motion.isStreaming == true }
        XCTAssertFalse(everStreamed, "Motion streamed with both switches off")
        XCTAssertFalse(connection.isMotionAttached)
        XCTAssertNil(connection.imuRecorder)
        XCTAssertEqual(connection.imuRecorderReadout.indicator, .off)
        _ = await stopCapture(connection, recorder: nil)
        XCTAssertFalse(FileManager.default.fileExists(atPath: base.appendingPathComponent(IMURecorder.directoryName).path),
                       "an imu-logs folder was created with the switch off")
    }

    /// 60 Hz refused by `addMotion`: the recorder falls back to 30 Hz, and the
    /// header and the events say so.
    func testA60HzRefusalFromAddMotionFallsBackTo30() async throws {
        let (connection, feed) = try await streamingConnection()
        defer { try? FileManager.default.removeItem(at: feed) }
        connection.imuRecorderEnabled = true
        connection.motionRatesRefusedForTesting = [60]
        try await startCapture(connection)
        let recorder = try XCTUnwrap(connection.imuRecorder)
        XCTAssertEqual(connection.motionConfiguredRate?.hertz, 30)
        let enough = await waitUntil(timeout: 15) { recorder.summary().counts.motionGlasses >= 90 }
        XCTAssertTrue(enough)
        let stopped = await stopCapture(connection, recorder: recorder)
        let summary = try XCTUnwrap(stopped)
        let lines = try IMULogTestSupport.lines(try XCTUnwrap(summary.fileURL))
        let header = try XCTUnwrap(lines.first)
        XCTAssertEqual(header["motion_rate_requested_hz"] as? Int, 60)
        XCTAssertEqual(header["motion_rate_actual_hz"] as? Int, 30)
        let attempts = try XCTUnwrap(header["motion_rate_attempts"] as? [[String: Any]])
        XCTAssertEqual(attempts.compactMap { $0["hz"] as? Int }, [60, 30])
        XCTAssertNotNil(attempts[0]["error"] as? String)
        XCTAssertTrue(attempts[1]["error"] is NSNull)
        // Mock replay paces delivery at the configured rate and passes the
        // recording's own timestamps through, so the rate to check is arrival.
        let motion = lines.filter { $0["t"] as? String == "m" }
        let arrivalHz = try XCTUnwrap(Self.rateHz(motion.compactMap { $0["rx_mono_ns"] as? Int }))
        XCTAssertEqual(arrivalHz, 30, accuracy: 6, "fell back to 30 Hz, received at \(arrivalHz)")
        XCTAssertNotNil(lines.first { $0["ev"] as? String == "motion_add_failed" })
    }

    /// 60 Hz accepted but failing at run time before a single sample: one retry
    /// at 30 Hz, on the live session -- `removeMotion`, then `addMotion` again
    /// -- and Motion streams again at the new rate. Delivered through the same
    /// handler as Motion's own error listener, since Mock Device Kit cannot
    /// publish a Motion error; the "before any sample" condition is
    /// `runtimeFallbackRate`'s and is tested below.
    func testA60HzRefusalAtRunTimeFallsBackTo30Once() async throws {
        let (connection, feed) = try await streamingConnection()
        defer { try? FileManager.default.removeItem(at: feed) }
        connection.imuRecorderEnabled = true
        try await startCapture(connection)
        let recorder = try XCTUnwrap(connection.imuRecorder)
        XCTAssertEqual(connection.motionConfiguredRate?.hertz, 60)

        connection.injectMotionRefusalForTesting()
        XCTAssertEqual(connection.motionConfiguredRate?.hertz, 30, "no fallback to 30 Hz")
        XCTAssertTrue(connection.isMotionAttached)
        connection.injectMotionRefusalForTesting()
        XCTAssertEqual(connection.motionConfiguredRate?.hertz, 30, "the fallback is taken once, not repeatedly")

        let resumed = await waitUntil(timeout: 15) { connection.motionSamplesSinceAttach >= 60 }
        XCTAssertTrue(resumed, "Motion did not stream again after the fallback (\(connection.motionSamplesSinceAttach) samples)")
        let stopped = await stopCapture(connection, recorder: recorder)
        let summary = try XCTUnwrap(stopped)
        let lines = try IMULogTestSupport.lines(try XCTUnwrap(summary.fileURL))
        let fallback = try XCTUnwrap(lines.first { $0["ev"] as? String == "motion_fallback" })
        XCTAssertEqual(fallback["from_hz"] as? Int, 60)
        XCTAssertEqual(fallback["to_hz"] as? Int, 30)
        XCTAssertEqual(lines.filter { $0["ev"] as? String == "motion_start" }.compactMap { $0["rate_hz"] as? Int }, [60, 30])
        XCTAssertEqual(lines.filter { $0["ev"] as? String == "motion_fallback" }.count, 1)
        // After the fallback, samples arrive at about 30 Hz.
        let restartedAt = try XCTUnwrap(lines.last { $0["ev"] as? String == "motion_start" }?["mono_ns"] as? Int)
        let after = lines.filter { $0["t"] as? String == "m" }.compactMap { $0["rx_mono_ns"] as? Int }.filter { $0 > restartedAt }
        let arrivalHz = try XCTUnwrap(Self.rateHz(after))
        XCTAssertEqual(arrivalHz, 30, accuracy: 6, "after the fallback, received at \(arrivalHz)")
        print("[IMURecTest] runtime fallback: \(after.count) samples after the retry at \(arrivalHz) Hz")
    }

    /// The decision itself: only with the recorder on, only before any sample,
    /// only once, only to a later rate.
    func testTheRunTimeFallbackDecision() {
        let rates = GlassesConnection.imuRecorderMotionRates
        func decide(_ rate: MotionSamplingRate, on: Bool = true, fell: Bool = false, samples: Int = 0) -> Int? {
            GlassesConnection.runtimeFallbackRate(
                after: rate, in: rates, recorderOn: on, alreadyFellBack: fell, samplesSinceAttach: samples
            )?.hertz
        }
        XCTAssertEqual(rates.map(\.hertz), [60, 30])
        XCTAssertEqual(decide(.hz60), 30)
        XCTAssertNil(decide(.hz60, samples: 1), "a rate that delivered was not refused")
        XCTAssertNil(decide(.hz60, fell: true), "only once")
        XCTAssertNil(decide(.hz60, on: false), "the probe alone never falls back")
        XCTAssertNil(decide(.hz30), "nothing after the last rate")
    }
}
#endif
