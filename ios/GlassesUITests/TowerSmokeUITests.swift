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
    @discardableResult
    private func reveal(_ element: XCUIElement, timeout: TimeInterval = 10) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if element.exists && element.isHittable { return true }
            app.swipeUp(velocity: .slow)
        }
        return element.exists && element.isHittable
    }

    private func open(cartridge name: String) {
        let cartridges = app.buttons["Cartridges"]
        XCTAssertTrue(cartridges.waitForExistence(timeout: 10), "the shell's Cartridges button")
        cartridges.tap()
        let row = app.buttons.containing(NSPredicate(format: "label BEGINSWITH %@", name)).firstMatch
        XCTAssertTrue(reveal(row), "a drawer row for \(name)")
        row.tap()
    }

    // MARK: World Builder

    /// Saved worlds -> choose a world -> the picture opens and draws inside
    /// the app -> Close -> Back to live.
    func testASavedWorldsPictureOpensInsideTheApp() throws {
        open(cartridge: "World Builder")
        XCTAssertTrue(app.staticTexts["World Builder"].waitForExistence(timeout: 5))
        attach("world-builder-live")

        // Nothing named yet: the picture button is offered and disabled.
        let picture = app.buttons["Interactive picture of the world"]
        XCTAssertTrue(reveal(picture))

        app.buttons["Saved worlds"].tap()
        XCTAssertTrue(app.navigationBars["Saved worlds"].waitForExistence(timeout: 5))
        // The first world's own row (a section header button), whatever it is
        // called: the Tower's list, not a fixture name.
        let sessionTag = app.staticTexts.containing(NSPredicate(format: "label ENDSWITH %@", "session")).firstMatch
        let sessionsTag = app.staticTexts.containing(NSPredicate(format: "label ENDSWITH %@", "sessions")).firstMatch
        XCTAssertTrue(sessionTag.waitForExistence(timeout: 15) || sessionsTag.waitForExistence(timeout: 1),
                      "the Tower listed at least one world")
        attach("saved-worlds")
        let worldRow = sessionTag.exists ? sessionTag : sessionsTag
        XCTAssertTrue(reveal(worldRow))
        worldRow.tap()

        // The header now names the world, and the picture is offered.
        let looking = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Looking at saved world")).firstMatch
        XCTAssertTrue(looking.waitForExistence(timeout: 5))
        XCTAssertTrue(reveal(picture))
        XCTAssertTrue(picture.isEnabled)
        attach("inspecting-saved-world")

        picture.tap()
        XCTAssertTrue(app.navigationBars["Reconstruction"].waitForExistence(timeout: 5))
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

        app.buttons["Close"].tap()
        XCTAssertTrue(looking.waitForExistence(timeout: 5))
        XCTAssertTrue(reveal(app.buttons["Back to live"]))
        app.buttons["Back to live"].tap()
        XCTAssertTrue(looking.waitForNonExistence(timeout: 5))
        attach("back-to-live")
    }

    /// A session with no geometry is told to the reader in the Tower's own
    /// words, with a way to try again -- not a blank page.
    func testASessionWithoutGeometrySaysSo() throws {
        open(cartridge: "World Builder")
        XCTAssertTrue(reveal(app.buttons["Saved worlds"]))
        app.buttons["Saved worlds"].tap()
        XCTAssertTrue(app.navigationBars["Saved worlds"].waitForExistence(timeout: 5))
        let noGeometry = app.staticTexts["no geometry"].firstMatch
        try XCTSkipUnless(noGeometry.waitForExistence(timeout: 15), "the Tower's list has no session without geometry")
        XCTAssertTrue(reveal(noGeometry))
        noGeometry.tap()
        let picture = app.buttons["Interactive picture of the world"]
        XCTAssertTrue(reveal(picture))
        picture.tap()
        let message = app.staticTexts.containing(NSPredicate(format: "label CONTAINS %@", "no geometry yet")).firstMatch
        XCTAssertTrue(message.waitForExistence(timeout: 15), "the Tower's 404 detail is shown")
        XCTAssertTrue(app.buttons["Try again"].exists)
        attach("no-geometry")
        app.buttons["Close"].tap()
    }

    // MARK: CV Lab

    /// The Tower row connects, disconnects and reconnects from inside the
    /// Lab; an experiment chosen from inside the Lab is armed on the Tower
    /// and the row reflects the answer. The camera card is present and
    /// truthful about having no glasses.
    func testTheLabConnectsAndSwitchesExperimentsFromItsOwnScreen() throws {
        open(cartridge: "Experimental CV Lab")
        XCTAssertTrue(app.staticTexts["Experimental CV Lab"].waitForExistence(timeout: 5))

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
        edge.tap()
        let armedEdge = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH[c] %@ AND label CONTAINS %@", "Edge detection", "armed")).firstMatch
        XCTAssertTrue(armedEdge.waitForExistence(timeout: 15), "the run header names Edge detection as armed")
        attach("cv-lab-edge")
        XCTAssertTrue(reveal(baseline, timeout: 15))
        baseline.tap()
        let armedBaseline = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@ AND label CONTAINS %@", "Baseline", "armed")).firstMatch
        XCTAssertTrue(armedBaseline.waitForExistence(timeout: 15))
        attach("cv-lab-baseline")
        XCTAssertTrue(connected.exists, "the socket never dropped across the switches")

        // Pause / Resume / Stop act on the run. The run panel sits above the
        // catalog, so scroll back up to it.
        app.swipeDown(velocity: .slow); app.swipeDown(velocity: .slow)
        XCTAssertTrue(app.buttons["Pause"].waitForExistence(timeout: 5))
        app.buttons["Pause"].tap()
        XCTAssertTrue(app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Paused.")).firstMatch.waitForExistence(timeout: 15))
        app.buttons["Resume"].tap()
        XCTAssertTrue(app.buttons["Pause"].waitForExistence(timeout: 15))
        app.buttons["Stop"].tap()
        XCTAssertTrue(app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Stopped.")).firstMatch.waitForExistence(timeout: 15))
        attach("cv-lab-stopped")

        // Disconnect and reconnect from the row.
        app.swipeDown(velocity: .slow); app.swipeDown(velocity: .slow)
        XCTAssertTrue(app.buttons["Disconnect"].waitForExistence(timeout: 5))
        app.buttons["Disconnect"].tap()
        XCTAssertTrue(app.staticTexts["Disconnected"].waitForExistence(timeout: 10))
        attach("cv-lab-disconnected")
        app.buttons["Connect"].tap()
        XCTAssertTrue(connected.waitForExistence(timeout: 20))
        XCTAssertTrue(reveal(edge, timeout: 20), "the catalog came back after the reconnect")
        attach("cv-lab-reconnected")
    }
}
