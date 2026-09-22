//
//  TowerSmokeUITests.swift
//  GlassesUITests
//
//  The app in the Simulator, against a real Tower on the branch under test.
//
//  Opt-in, and skipped otherwise: set GLASSES_UITEST_TOWER_AUTHORITY to the
//  `host:port` of a running Tower (a Mac-local one is `127.0.0.1:8010`) and
//  the app is launched with `GLASSES_TOWER_AUTHORITY` pointing at it. With
//  the variable unset every test here reports as skipped, never as passed,
//  so a green run without a Tower cannot be mistaken for a validated one.
//
//  What these prove is the seam the unit tests cannot: that the screens
//  reach a Tower over a socket and HTTP, that a saved world can be chosen
//  and its picture fetched and drawn by a WKWebView, and that the CV Lab's
//  connection row and experiment rows act on the Tower and reflect what it
//  answers. What they do not prove is anything the glasses do -- the
//  Simulator has no camera -- and each test says so where that matters.
//
//      TEST_RUNNER_GLASSES_UITEST_TOWER_AUTHORITY=127.0.0.1:8010 xcodebuild \
//          -project Glasses.xcodeproj -scheme Glasses \
//          -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
//          -only-testing:GlassesUITests test
//
//  (xcodebuild forwards a variable to the test runner only under the
//  TEST_RUNNER_ prefix; the runner sees it without the prefix.)
//
//  Debug configuration only: the app honours GLASSES_TOWER_AUTHORITY under
//  `#if DEBUG`. A Release-configured run would talk to the development
//  Tower instead, so `setUp` refuses to run against anything but a build
//  that shows the DEBUG-only capture surface.
//

import XCTest

final class TowerSmokeUITests: XCTestCase {

    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        guard let authority = ProcessInfo.processInfo.environment["GLASSES_UITEST_TOWER_AUTHORITY"],
              !authority.isEmpty
        else {
            throw XCTSkip("Set GLASSES_UITEST_TOWER_AUTHORITY=host:port to run against a Tower.")
        }
        app = XCUIApplication()
        app.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = authority
        // A system alert (location, notifications) would otherwise sit over
        // the app and every wait below would time out on it.
        addUIInterruptionMonitor(withDescription: "system alert") { alert in
            for title in ["Allow", "Don't Allow", "OK", "Not Now"] {
                let button = alert.buttons[title]
                if button.exists { button.tap(); return true }
            }
            return false
        }
        app.launch()
        // The DEBUG-only Home capture control is the proof this build reads
        // the override at all; without it the tests would be steering the
        // wrong Tower and any result would be about something else.
        let debugOnlyControl = app.buttons["Start session"]
        try XCTSkipUnless(debugOnlyControl.waitForExistence(timeout: 10),
                          "not a DEBUG build; the Tower override is not read, so these tests cannot steer it.")
    }

    private func attach(_ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    /// Swipe the screen up until `element` is on it and can be hit. The
    /// workspaces are long scroll views and the drawer is a half-height
    /// sheet, so most of what these tests touch starts below the fold.
    ///
    /// **Settle before swiping.** This used to swipe the moment the first
    /// hittability check failed, and that is a trap for anything near the top
    /// of a workspace. A sheet dismissing over the target reads as
    /// not-hittable for a few hundred milliseconds; one `swipeUp` then scrolls
    /// a control that was already on screen up past the top, and because this
    /// only ever swipes *up* it can never come back — so the loop spends its
    /// whole timeout scrolling away from the thing it is looking for.
    ///
    /// That is not hypothetical: it is what made
    /// `testOpeningASavedWorldShowsThe3DWorld` (then named
    /// `testASavedWorldsPictureOpensInsideTheApp`) fail at `reveal(picture)`
    /// straight after the picker closed. Instrumenting the same step showed
    /// the button `exists`, `isEnabled` and `isHittable` once the dismissal
    /// animation had finished — the product was right and the helper was
    /// racing it. The swipe loop below is unchanged, so nothing that was
    /// asserted is weakened: an element that is absent, disabled or genuinely
    /// occluded still fails. The cost is a flat `settle` seconds on any call
    /// where the element really is below the fold, which raises this helper's
    /// worst case from `timeout` to `timeout + settle`.
    ///
    /// **And swipe back down for the second half of the budget.** The same
    /// trap in the other direction, found at the 2026-09-14 Mac gate: the
    /// saved-world test reveals Diagnostics (below the fold, several swipes
    /// down the workspace) and then "Back to live", which sits in the banner
    /// at the TOP. An up-only loop had scrolled it off the screen to find the
    /// first and could never bring it back for the second. An element that
    /// is absent, disabled or genuinely occluded still fails: both halves
    /// require `exists && isHittable`, and nothing is asserted more weakly.
    @discardableResult
    private func reveal(_ element: XCUIElement, timeout: TimeInterval = 10,
                        settle: TimeInterval = 2) -> Bool {
        let settleDeadline = Date().addingTimeInterval(settle)
        while Date() < settleDeadline {
            if element.exists && element.isHittable { return true }
            Thread.sleep(forTimeInterval: 0.1)
        }
        let halfway = Date().addingTimeInterval(timeout / 2)
        while Date() < halfway {
            if element.exists && element.isHittable { return true }
            app.swipeUp(velocity: .slow)
        }
        let deadline = Date().addingTimeInterval(timeout / 2)
        while Date() < deadline {
            if element.exists && element.isHittable { return true }
            app.swipeDown(velocity: .slow)
        }
        return element.exists && element.isHittable
    }

    /// Tap `element` until `effect` is true. A tap that lands while a sheet
    /// is still animating in can be swallowed; the effect, not the tap, is
    /// what the test is about.
    @discardableResult
    private func tap(_ element: XCUIElement, until effect: @autoclosure () -> Bool,
                     attempts: Int = 3, wait: TimeInterval = 3) -> Bool {
        for _ in 0..<attempts {
            if element.exists && element.isHittable { element.tap() }
            let deadline = Date().addingTimeInterval(wait)
            while Date() < deadline {
                if effect() { return true }
                Thread.sleep(forTimeInterval: 0.25)
            }
        }
        return effect()
    }

    private func open(cartridge name: String) {
        let cartridges = app.buttons["Cartridges"]
        XCTAssertTrue(cartridges.waitForExistence(timeout: 10), "the shell's Cartridges button")
        let drawerDone = app.buttons["Done"]
        XCTAssertTrue(tap(cartridges, until: drawerDone.exists), "the cartridge drawer opened")
        let row = app.buttons.containing(NSPredicate(format: "label BEGINSWITH %@", name)).firstMatch
        XCTAssertTrue(reveal(row), "a drawer row for \(name)")
        // The workspace has arrived when the drawer is gone and the shell's
        // title names it -- not when a static text with that name exists,
        // which the drawer row itself satisfies.
        XCTAssertTrue(tap(row, until: !drawerDone.exists && app.navigationBars[name].exists),
                      "the \(name) workspace opened")
    }

    // MARK: World Builder

    /// Saved worlds -> choose a session -> **the 3D world is what opens** ->
    /// back -> Close -> the workspace offers the same world, with the sparse
    /// solver output behind Diagnostics.
    ///
    /// ## What changed here, and why the old assertions were pinning a defect
    ///
    /// This test used to tap a session, assert the picker DISMISSED, then hunt
    /// the workspace for a small bordered "Picture" button and tap that. It was
    /// an accurate description of the product and the product was wrong:
    /// opening a saved world dropped the reader onto a gallery of top-down
    /// sparse point clouds captioned with the solver's registration vocabulary,
    /// with the 3D reconstruction two taps and a scroll away.
    ///
    /// A tap now pushes the 3D world onto the picker's own navigation stack.
    /// The picker stays up underneath, so a person comparing two walks does not
    /// reopen it each time, and Close still lands on a workspace pinned to the
    /// world they chose.
    func testOpeningASavedWorldShowsThe3DWorld() throws {
        open(cartridge: "World Builder")
        attach("world-builder-live")

        // Nothing named yet: the header's picture control is offered and
        // disabled. Unchanged, and still the honest answer for a live screen
        // with no world on it.
        let picture = app.buttons["Interactive picture of the world"]
        XCTAssertTrue(reveal(picture))
        XCTAssertFalse(picture.isEnabled, "no world has been named, so there is nothing to open")

        XCTAssertTrue(tap(app.buttons["Saved worlds"], until: app.navigationBars["Saved worlds"].exists))
        let sessionTag = app.staticTexts.containing(NSPredicate(format: "label ENDSWITH %@", "session")).firstMatch
        let sessionsTag = app.staticTexts.containing(NSPredicate(format: "label ENDSWITH %@", "sessions")).firstMatch
        XCTAssertTrue(sessionTag.waitForExistence(timeout: 15) || sessionsTag.waitForExistence(timeout: 1),
                      "the Tower listed at least one world")
        attach("saved-worlds")

        // A SESSION row, chosen by its badge, rather than the world's own row.
        // Opening a world without naming a session asks the Tower for its
        // `latest` selection, and if that session has no geometry there is no
        // picture to show. "Complete" and "Partial" are the two badges with
        // something to draw: since 2026-09-14 a finished session whose final
        // solve was skipped or failed is badged "Partial" on the row, the
        // same word the canvas gives it, and it opens exactly like a
        // "Complete" one. The Mac fixture's finished session records
        // `final_solve: skipped`, so it is a "Partial" row.
        let complete = app.staticTexts.containing(
            NSPredicate(format: "label IN %@", ["Complete", "Partial"])
        ).firstMatch
        try XCTSkipUnless(complete.waitForExistence(timeout: 15),
                          "the Tower's list has no finished session to picture")
        XCTAssertTrue(reveal(complete))

        // The 3D world opens from the tap itself. No second control.
        //
        // Pinned on "not to scale", the one phrase every rung's caption
        // shares: the caption now follows the page, and a surface is not "Not
        // a surface". What must hold on every rung is that no size is claimed.
        let caption = app.staticTexts.containing(
            NSPredicate(format: "label CONTAINS[c] %@", "not to scale")
        ).firstMatch
        XCTAssertTrue(tap(complete, until: caption.exists),
                      "tapping a session opens its 3D world, with no further tap")

        // The page arrived and the web view drew it. Asserted on the presence
        // of the Tower's page rather than on its wording: the render page's own
        // caption is the Tower's to write, and asserting a phrase from it here
        // would pin a page that is being rewritten on the other side.
        let webView = app.webViews.firstMatch
        XCTAssertTrue(webView.waitForExistence(timeout: 30), "the Tower's page rendered inside the WKWebView")

        // The caption above proves the push, not the page: its "not to scale"
        // predicate also matches the caption shown while the page is still
        // being fetched. The rung caption is only on screen once the app has
        // read `wb-representation` out of the Tower's page, which is the one
        // thing only a real Tower can prove. The prefixes are the first words
        // of `WorldRenderRepresentation.caption(for:)` for **every** rung it
        // can return: appearance, surface, dense and sparse, in that order.
        //
        // The appearance prefix was missing and this assertion was a certain
        // failure on the happy path (review 2, M-9). `WorldRenderClient.url`
        // sends `viewer=appearance-1` unconditionally, so a Tower that has an
        // appearance artifact — which the canonical validation world has — is
        // asked for and serves the appearance page, whose caption begins with
        // none of the three rungs this predicate used to list.
        let rungCaption = app.staticTexts.containing(NSPredicate(
            format: "label BEGINSWITH %@ OR label BEGINSWITH %@ OR label BEGINSWITH %@"
                + " OR label BEGINSWITH %@",
            "The camera's own images", "Surfaces the Tower reconstructed",
            "Points the Tower measured densely", "Points the Tower measured from the walk"
        )).firstMatch
        XCTAssertTrue(rungCaption.waitForExistence(timeout: 30), "the caption read the page's rung")
        attach("3d-world")

        // There is always a way to ask for the world again, whatever the page
        // is showing (review 2, M-0): a page that boots with no imagery reports
        // `didFinish` like any other, so the screen is ready and the failure
        // view's "Try again" is not on it. Queried by identifier, because the
        // word in the toolbar is the app's to change.
        XCTAssertTrue(
            app.descendants(matching: .any).matching(identifier: "world-render-reload")
                .firstMatch.waitForExistence(timeout: 10),
            "the 3D world always offers a way to fetch it again"
        )

        // The canvas takes a gesture without the screen moving under it.
        webView.swipeLeft()
        webView.pinch(withScale: 1.5, velocity: 1)
        attach("3d-world-after-gestures")
        XCTAssertTrue(caption.exists)

        // Details holds what used to be in the caption: the identifiers, and
        // the switch to the Tower's diagnostics rendering. Both still
        // reachable, neither in the way.
        let details = app.buttons["Details"].firstMatch
        if reveal(details, timeout: 4, settle: 1) {
            details.tap()
            XCTAssertTrue(app.staticTexts.containing(
                NSPredicate(format: "label BEGINSWITH %@", "World ")
            ).firstMatch.waitForExistence(timeout: 5),
                          "the world id is reachable in Details")
            attach("3d-world-details")
        }

        // Back to the picker, then out of it: the workspace is pinned to the
        // world that was opened.
        // The back button by its LABEL, which UIKit sets to the previous
        // screen's title. `app.navigationBars.buttons.element(boundBy: 0)` is
        // unscoped across every navigation bar on screen and can pick the
        // wrong one while a push is still animating; a labelled query cannot.
        // The index form stays only as a last resort, for the case where a
        // long title makes UIKit collapse the label to "Back".
        let named = app.navigationBars.buttons["Saved worlds"]
        let generic = app.navigationBars.buttons["Back"]
        let backToWorlds = named.exists ? named
            : (generic.exists ? generic : app.navigationBars.firstMatch.buttons.element(boundBy: 0))
        XCTAssertTrue(tap(backToWorlds, until: app.navigationBars["Saved worlds"].exists),
                      "the 3D world pops back to the list it was opened from")
        XCTAssertTrue(tap(app.buttons["Close"], until: !app.navigationBars["Saved worlds"].exists))

        let looking = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Looking at saved world")).firstMatch
        XCTAssertTrue(looking.waitForExistence(timeout: 10), "the workspace names the world that was opened")
        attach("inspecting-saved-world")

        // The 3D world is the workspace's primary control for this world too.
        let open3D = app.buttons["Open the 3D world"].firstMatch
        XCTAssertTrue(reveal(open3D), "the workspace leads with the 3D world, not with the gallery")

        // The sentence that was wrong in the field must not be anywhere on a
        // screen describing a world the Tower built.
        XCTAssertFalse(
            app.staticTexts.containing(
                NSPredicate(format: "label CONTAINS %@", "have not mapped anything")
            ).firstMatch.exists,
            "a world with geometry must never be described as unmapped"
        )

        // And the sparse solver output is still reachable, behind Diagnostics.
        //
        // Queried by the identifier the view actually sets
        // (`WorldCanvasView.diagnostics`), with the label as a fallback: a
        // `DisclosureGroup` is published as a button on some OS versions and as
        // a container on others, and a hard assertion on one of those spellings
        // burns the full `reveal` timeout when the other is what shipped.
        //
        // Tolerated rather than asserted, for the same reason the Details
        // disclosure above is: this test's subject is that the 3D world is what
        // opens, and the gallery's own presence is pinned by
        // `WorldPresentationTests` without a simulator. A miss is reported so
        // it is visible in the log rather than silently passing.
        let byIdentifier = app.descendants(matching: .any)
            .matching(identifier: "world-diagnostics").firstMatch
        let byLabel = app.buttons["Diagnostics"].firstMatch
        let diagnostics = byIdentifier.exists ? byIdentifier : byLabel
        if reveal(diagnostics, timeout: 4, settle: 1) {
            attach("diagnostics-available")
        } else {
            XCTContext.runActivity(named: "Diagnostics disclosure not found by identifier or label") { _ in }
        }

        XCTAssertTrue(reveal(app.buttons["Back to live"]))
        XCTAssertTrue(tap(app.buttons["Back to live"], until: !looking.exists))
        attach("back-to-live")
    }

    /// A session the Tower says has no geometry is described on the row, and
    /// is not a control.
    ///
    /// ## Why this test stopped tapping
    ///
    /// It used to tap the row and read the Tower's 404 prose with a "Try again"
    /// beneath it. That is a control for an operation that cannot work:
    /// `GET /worlds/{id}/render?session_id=` answers 404 for a session with no
    /// geometry, `WorldRenderFetchError.absent` is classified retryable, and
    /// nothing reachable from this screen builds geometry for a session that
    /// has none. The same defect was removed one level up for worlds with no
    /// sessions; this is it applied to session rows.
    ///
    /// So the assertion is now about the row: the Tower's word is on it, the
    /// reason is spelled out, and the tap does nothing because there is nothing
    /// to tap.
    func testASessionWithoutGeometrySaysSoOnTheRow() throws {
        open(cartridge: "World Builder")
        XCTAssertTrue(reveal(app.buttons["Saved worlds"]))
        XCTAssertTrue(tap(app.buttons["Saved worlds"], until: app.navigationBars["Saved worlds"].exists))
        // "No geometry" since 2026-09-06: the picker words the badge from the
        // Tower's own `state`, and `unbuilt` reads "No geometry". The old
        // lower-case "no geometry" was the fallback for a Tower that sent no
        // state at all, and it is still reachable, so both are accepted.
        let noGeometry = app.staticTexts.containing(
            NSPredicate(format: "label ==[c] %@", "no geometry")
        ).firstMatch
        try XCTSkipUnless(noGeometry.waitForExistence(timeout: 15), "the Tower's list has no session without geometry")
        XCTAssertTrue(reveal(noGeometry))

        // The row says why, in one of the two sentences the picker can use.
        let reason = app.staticTexts.containing(
            NSPredicate(format: "label CONTAINS %@", "nothing to open")
        ).firstMatch
        let stillComing = app.staticTexts.containing(
            NSPredicate(format: "label CONTAINS %@", "may still build it")
        ).firstMatch
        XCTAssertTrue(reason.waitForExistence(timeout: 5) || stillComing.exists,
                      "a session with no geometry must say why it cannot be opened")
        attach("no-geometry")

        // And tapping it leads nowhere: no 3D screen, so no caption from one.
        noGeometry.tap()
        let caption = app.staticTexts.containing(
            NSPredicate(format: "label CONTAINS[c] %@", "not to scale")
        ).firstMatch
        XCTAssertFalse(caption.waitForExistence(timeout: 3),
                       "a session with nothing to draw must not push a 3D screen")
        XCTAssertTrue(app.navigationBars["Saved worlds"].exists, "still on the list")
    }

    // MARK: CV Lab

    /// The Tower row connects, disconnects and reconnects from inside the
    /// Lab; an experiment chosen from inside the Lab is armed on the Tower
    /// and the row reflects the answer. The camera card is present and
    /// truthful about having no glasses.
    func testTheLabConnectsAndSwitchesExperimentsFromItsOwnScreen() throws {
        open(cartridge: "Experimental CV Lab")

        let connected = app.staticTexts["Connected"]
        XCTAssertTrue(connected.waitForExistence(timeout: 20), "the shell's auto-connect reached the Tower")
        attach("cv-lab-connected")

        // The camera card exists and, with no glasses, says so rather than
        // offering a start that would do nothing.
        XCTAssertTrue(app.staticTexts["Waiting for the glasses to become active."].exists)

        // The catalog came down the socket.
        let baseline = app.buttons.containing(NSPredicate(format: "label BEGINSWITH %@", "Baseline")).firstMatch
        let edge = app.buttons.containing(NSPredicate(format: "label BEGINSWITH[c] %@", "Edge detection")).firstMatch
        XCTAssertTrue(reveal(edge, timeout: 20), "the Tower's experiment catalog")
        attach("cv-lab-catalog")

        // Switch twice. Each arm is acknowledged by the Tower within the
        // client's ten-second bound, after which the row is a button again.
        let armedEdge = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH[c] %@ AND label CONTAINS %@", "Edge detection", "armed")).firstMatch
        XCTAssertTrue(tap(edge, until: armedEdge.exists, wait: 15), "the run header names Edge detection as armed")
        attach("cv-lab-edge")
        XCTAssertTrue(reveal(baseline, timeout: 15))
        let armedBaseline = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@ AND label CONTAINS %@", "Baseline", "armed")).firstMatch
        XCTAssertTrue(tap(baseline, until: armedBaseline.exists, wait: 15))
        attach("cv-lab-baseline")
        XCTAssertTrue(connected.exists, "the socket never dropped across the switches")

        // Pause / Resume / Stop act on the run. The run panel sits above the
        // catalog, so scroll back up to it.
        app.swipeDown(velocity: .slow); app.swipeDown(velocity: .slow)
        XCTAssertTrue(reveal(app.buttons["Pause"]))
        let paused = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Paused.")).firstMatch
        XCTAssertTrue(tap(app.buttons["Pause"], until: paused.exists, wait: 15))
        XCTAssertTrue(tap(app.buttons["Resume"], until: app.buttons["Pause"].exists, wait: 15))
        let stopped = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Stopped.")).firstMatch
        XCTAssertTrue(tap(app.buttons["Stop"], until: stopped.exists, wait: 15))
        attach("cv-lab-stopped")

        // Disconnect and reconnect from the row.
        app.swipeDown(velocity: .slow); app.swipeDown(velocity: .slow)
        XCTAssertTrue(reveal(app.buttons["Disconnect"]))
        XCTAssertTrue(tap(app.buttons["Disconnect"], until: app.staticTexts["Disconnected"].exists, wait: 10))
        attach("cv-lab-disconnected")
        XCTAssertTrue(tap(app.buttons["Connect"], until: connected.exists, wait: 20))
        XCTAssertTrue(reveal(edge, timeout: 20), "the catalog came back after the reconnect")
        attach("cv-lab-reconnected")
    }
}
