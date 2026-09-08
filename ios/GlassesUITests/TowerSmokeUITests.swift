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
    /// `testASavedWorldsPictureOpensInsideTheApp` fail at `reveal(picture)`
    /// straight after the picker closed. Instrumenting the same step showed
    /// the button `exists`, `isEnabled` and `isHittable` once the dismissal
    /// animation had finished — the product was right and the helper was
    /// racing it. The swipe loop below is unchanged, so nothing that was
    /// asserted is weakened: an element that is absent, disabled or genuinely
    /// occluded still fails. The cost is a flat `settle` seconds on any call
    /// where the element really is below the fold, which raises this helper's
    /// worst case from `timeout` to `timeout + settle`.
    @discardableResult
    private func reveal(_ element: XCUIElement, timeout: TimeInterval = 10,
                        settle: TimeInterval = 2) -> Bool {
        let settleDeadline = Date().addingTimeInterval(settle)
        while Date() < settleDeadline {
            if element.exists && element.isHittable { return true }
            Thread.sleep(forTimeInterval: 0.1)
        }
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if element.exists && element.isHittable { return true }
            app.swipeUp(velocity: .slow)
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

    /// Saved worlds -> choose a world -> the picture opens and draws inside
    /// the app -> Close -> Back to live.
    func testASavedWorldsPictureOpensInsideTheApp() throws {
        open(cartridge: "World Builder")
        attach("world-builder-live")

        // Nothing named yet: the picture button is offered and disabled.
        let picture = app.buttons["Interactive picture of the world"]
        XCTAssertTrue(reveal(picture))
        XCTAssertFalse(picture.isEnabled, "no world has been named, so there is nothing to open")

        XCTAssertTrue(tap(app.buttons["Saved worlds"], until: app.navigationBars["Saved worlds"].exists))
        // A world is listed at all.
        let sessionTag = app.staticTexts.containing(NSPredicate(format: "label ENDSWITH %@", "session")).firstMatch
        let sessionsTag = app.staticTexts.containing(NSPredicate(format: "label ENDSWITH %@", "sessions")).firstMatch
        XCTAssertTrue(sessionTag.waitForExistence(timeout: 15) || sessionsTag.waitForExistence(timeout: 1),
                      "the Tower listed at least one world")
        attach("saved-worlds")

        // A SESSION row, chosen by its badge, rather than the world's own row.
        //
        // This test used to tap the world row and expect a picture. Since
        // 2026-09-06 that is the wrong expectation, and deliberately so.
        // Opening a world without naming a session asks the Tower for its
        // `latest` selection, which is the most recently updated session --
        // and if that session has no geometry, there is no picture to offer
        // and the control is correctly disabled. Drawing an older session's
        // geometry under a newer session's name is exactly the "geometry that
        // is not this world's, presented as if it were" defect the World
        // Builder lane set out to close, so a test that demanded it was
        // pinning the bug.
        //
        // The picture path therefore goes through a session that HAS
        // geometry, which is what a person does: the picker shows a badge per
        // session, and "Complete" is the one with something to draw.
        let complete = app.staticTexts["Complete"].firstMatch
        try XCTSkipUnless(complete.waitForExistence(timeout: 15),
                          "the Tower's list has no completed session to picture")
        XCTAssertTrue(reveal(complete))

        // The header now names the world, and the picture is offered.
        let looking = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Looking at saved world")).firstMatch
        XCTAssertTrue(tap(complete, until: looking.exists), "the session opened and the picker closed")
        XCTAssertTrue(reveal(picture))
        XCTAssertTrue(picture.isEnabled)
        attach("inspecting-saved-world")

        XCTAssertTrue(tap(picture, until: app.navigationBars["Reconstruction"].exists))
        // The page arrived and the web view drew it: the page's own title
        // bar names the world, and its caption says what the picture is.
        let pageTitle = app.webViews.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "world ")).firstMatch
        XCTAssertTrue(pageTitle.waitForExistence(timeout: 30), "the Tower's page rendered inside the WKWebView")
        let pageCaption = app.webViews.staticTexts.containing(NSPredicate(format: "label CONTAINS %@", "structure-from-motion")).firstMatch
        XCTAssertTrue(pageCaption.exists, "the page's own caption is on screen")
        XCTAssertTrue(app.staticTexts.containing(NSPredicate(format: "label CONTAINS %@", "Not a surface")).firstMatch.exists,
                      "the sheet's caption is on screen")
        attach("picture")

        // The canvas takes a gesture without the sheet or page moving.
        let canvas = app.webViews.firstMatch
        canvas.swipeLeft()
        canvas.pinch(withScale: 1.5, velocity: 1)
        attach("picture-after-gestures")
        XCTAssertTrue(app.navigationBars["Reconstruction"].exists)

        XCTAssertTrue(tap(app.buttons["Close"], until: !app.navigationBars["Reconstruction"].exists))
        XCTAssertTrue(looking.waitForExistence(timeout: 5))
        XCTAssertTrue(reveal(app.buttons["Back to live"]))
        XCTAssertTrue(tap(app.buttons["Back to live"], until: !looking.exists))
        attach("back-to-live")
    }

    /// A session with no geometry is told to the reader in the Tower's own
    /// words, with a way to try again -- not a blank page.
    func testASessionWithoutGeometrySaysSo() throws {
        open(cartridge: "World Builder")
        XCTAssertTrue(reveal(app.buttons["Saved worlds"]))
        XCTAssertTrue(tap(app.buttons["Saved worlds"], until: app.navigationBars["Saved worlds"].exists))
        // "No geometry" since 2026-09-06: the picker now words the badge from
        // the Tower's own `state`, and `unbuilt` reads "No geometry". The old
        // lower-case "no geometry" was the fallback for a Tower that sent no
        // state at all, and it is still reachable, so both are accepted --
        // matching on the words rather than on one spelling of them.
        let noGeometry = app.staticTexts.containing(
            NSPredicate(format: "label ==[c] %@", "no geometry")
        ).firstMatch
        try XCTSkipUnless(noGeometry.waitForExistence(timeout: 15), "the Tower's list has no session without geometry")
        XCTAssertTrue(reveal(noGeometry))
        let looking = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Looking at saved world")).firstMatch
        XCTAssertTrue(tap(noGeometry, until: looking.exists), "the session opened and the picker closed")
        let picture = app.buttons["Interactive picture of the world"]
        XCTAssertTrue(reveal(picture))

        // The picture is REFUSED, not offered and then apologised for.
        //
        // This test used to open the viewer here and read the Tower's 404
        // prose ("session ... has no geometry yet") with a Try again button.
        // Since 2026-09-06 that sheet is unreachable for this session, and
        // deliberately: the Tower says `geometry.available: false`, the app
        // therefore names no render target, and the control is disabled --
        // "a control that appears from nowhere is one nobody looks for, and
        // a disabled one says 'not yet' truthfully"
        // (WorldBuilderWorkspaceView). Offering a button whose only outcome
        // is a 404 was the weaker behaviour, so this asserts the better one.
        //
        // The session is still openable and still named, which is the half
        // that matters: a session with nothing to draw is a session a person
        // can look at and be told about, not one that disappears.
        XCTAssertFalse(picture.isEnabled,
                       "a session the Tower says has no geometry must not offer a picture")
        attach("no-geometry")

        // The Tower's own 404 prose is still the viewer's answer when a
        // render fails for a session that DID claim geometry. That path is
        // covered by the Tower's route tests; it is not reachable from this
        // screen for this session any more, and pretending otherwise here
        // would be pinning a control that no longer exists.
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
