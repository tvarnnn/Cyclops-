//
//  TowerSettingsUITests.swift
//  GlassesUITests
//
//  Settings' Tower address, end to end in the Simulator, against two local
//  mocks served from this process (`MockTowerHTTPServer`): one that answers as
//  a Tower and one that answers 404 to everything.
//
//  Runs by default: it needs no Tower. It never lets the app reach a real one
//  either. Launch 1 points the app's own socket at a closed loopback port with
//  GLASSES_TOWER_AUTHORITY; launch 2 has no override, and its socket goes to
//  the address launch 1 saved, which is the mock. The built-in address is only
//  ever *saved as the next launch's*, never connected to.
//
//  Screenshots, opt-in: TEST_RUNNER_U02_SHOTS_DIR=<dir> writes PNGs there,
//  named with TEST_RUNNER_U02_SHOT_SUFFIX; TEST_RUNNER_U02_CONTENT_SIZE=<UIKit
//  content size category name> launches at that text size.
//

import XCTest

final class TowerSettingsUITests: XCTestCase {

    private var tower: MockTowerHTTPServer!
    private var notATower: MockTowerHTTPServer!
    private var towerAuthority = ""
    private var notATowerAuthority = ""
    /// Nothing listens on port 9 of this Mac, so a connection is refused at
    /// once — the app's own socket fails fast and lands on "Tower unreachable".
    private let closedAuthority = "127.0.0.1:9"

    override func setUpWithError() throws {
        continueAfterFailure = false
        tower = try MockTowerHTTPServer(.tower)
        towerAuthority = "127.0.0.1:\(try tower.start())"
        notATower = try MockTowerHTTPServer(.notATower)
        notATowerAuthority = "127.0.0.1:\(try notATower.start())"
    }

    override func tearDownWithError() throws {
        // Leave nothing saved for the next run, even after a failure half-way:
        // a launch with the reset argument clears it before anything reads it.
        let cleanup = XCUIApplication()
        cleanup.launchArguments = ["-UITestResetTowerAddress", "-UITestSkipOnboarding"]
        cleanup.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = closedAuthority
        cleanup.launch()
        cleanup.terminate()
        tower?.stop()
        notATower?.stop()
    }

    // MARK: The test

    func testSettingsTestsAnAddressReadOnlyAndSavesItForTheNextLaunch() throws {
        // Launch 1: the developer override points the socket at a closed port.
        var app = launch(override: closedAuthority, reset: true)

        // "Tower unreachable" leads to Settings.
        let banner = app.buttons["Tower settings"]
        XCTAssertTrue(banner.waitForExistence(timeout: 20), "the failed Tower's banner offers Settings")
        XCTAssertTrue(reveal(app, banner), "the banner's action is on the screen")
        shoot(app, "01-home-tower-failed")
        XCTAssertTrue(tap(banner, until: app.navigationBars["Settings"].exists), "the banner opens Settings")

        let inUse = element(app, "tower-in-use")
        XCTAssertTrue(inUse.waitForExistence(timeout: 5))
        XCTAssertTrue(inUse.label.contains(closedAuthority), inUse.label)
        XCTAssertTrue(inUse.label.contains("Developer override"), inUse.label)
        XCTAssertTrue(reveal(app, element(app, "tower-unencrypted-note")), "the unencrypted note is on the screen")

        // Not an address: said so, and Test is off.
        enter(app, "http://\(towerAuthority)")
        let problem = element(app, "tower-address-problem")
        XCTAssertTrue(problem.waitForExistence(timeout: 5))
        XCTAssertTrue(problem.label.contains("Leave out http://"), problem.label)
        shoot(app, "02-settings-invalid")
        let test = app.buttons["tower-test"]
        XCTAssertTrue(reveal(app, test))
        XCTAssertFalse(test.isEnabled)

        // Something that answers, and is not a Tower.
        enter(app, notATowerAuthority)
        XCTAssertEqual(runTest(app), "Reached something, but it isn't a Glasses Tower")
        shoot(app, "03-settings-not-a-tower")
        shootFurther(app, "03-settings-not-a-tower")

        // Nothing there at all.
        enter(app, closedAuthority)
        XCTAssertEqual(runTest(app), "Not reachable")

        // A Tower, with its contract.
        enter(app, towerAuthority)
        XCTAssertEqual(runTest(app), "Connected to a Tower")
        XCTAssertTrue(element(app, "tower-test-result").label.contains(MockTowerHTTPServer.contract))
        shoot(app, "04-settings-connected")
        shootFurther(app, "04-settings-connected")

        // Exactly one request per Test, and it is the read-only one.
        XCTAssertEqual(tower.requestLines, ["GET /cartridges HTTP/1.1"])
        XCTAssertEqual(notATower.requestLines, ["GET /cartridges HTTP/1.1"])

        // Save it. The override still wins in this run, and Settings says so.
        let save = app.buttons["tower-save"]
        XCTAssertTrue(reveal(app, save))
        save.tap()
        XCTAssertFalse(save.isEnabled, "saved")
        let override = element(app, "tower-override-notice")
        XCTAssertTrue(reveal(app, override))
        XCTAssertTrue(override.label.contains(towerAuthority), override.label)
        shoot(app, "05-settings-saved-under-override")
        app.terminate()

        // Launch 2: no override. The saved address is the one in use, and the
        // app's own socket went there (a WebSocket upgrade on /ws), not to the
        // built-in Tower.
        app = launch(override: nil, reset: false)
        XCTAssertTrue(waitFor { self.tower.requestLines.contains("GET /ws HTTP/1.1") },
                      "the app connected to the saved address: \(tower.requestLines)")
        XCTAssertTrue(tap(app.buttons["Settings"], until: app.navigationBars["Settings"].exists),
                      "the toolbar opens Settings")
        let inUseNow = element(app, "tower-in-use")
        XCTAssertTrue(inUseNow.waitForExistence(timeout: 5))
        XCTAssertTrue(inUseNow.label.contains(towerAuthority), inUseNow.label)
        XCTAssertTrue(inUseNow.label.contains("Saved in Settings"), inUseNow.label)
        XCTAssertFalse(element(app, "tower-restart-notice").exists, "nothing pending yet")

        // Back to the built-in address: applied at the next launch, and said.
        let useBuiltIn = app.buttons["tower-use-built-in"]
        XCTAssertTrue(reveal(app, useBuiltIn))
        useBuiltIn.tap()
        let restart = element(app, "tower-restart-notice")
        XCTAssertTrue(reveal(app, restart), "a restart is asked for")
        XCTAssertTrue(restart.label.contains("100.110.156.55:8000"), restart.label)
        XCTAssertTrue(restart.label.contains(towerAuthority), restart.label)
        shoot(app, "06-settings-restart-needed")
        shootFurther(app, "06-settings-restart-needed")

        // Nothing but reads ever reached either mock.
        for line in tower.requestLines + notATower.requestLines {
            XCTAssertTrue(line.hasPrefix("GET "), line)
            XCTAssertFalse(line.contains("session"), line)
        }
        app.terminate()
    }

    /// Opt-in: TEST_RUNNER_U02_REAL_TOWER=<host:port> of a real Tower (or the
    /// read-only proxy in front of one). "Test connection" against it must say
    /// it is a Tower and name the contract the Tower declares. Only
    /// `GET /cartridges` is sent; the app's own socket stays on a closed port.
    func testTestConnectionRecognisesARealTower() throws {
        guard let real = env("U02_REAL_TOWER") else {
            throw XCTSkip("Set TEST_RUNNER_U02_REAL_TOWER=host:port to test against a real Tower.")
        }
        let app = launch(override: closedAuthority, reset: true)
        XCTAssertTrue(tap(app.buttons["Settings"], until: app.navigationBars["Settings"].exists))
        enter(app, real)
        XCTAssertEqual(runTest(app), "Connected to a Tower")
        let result = element(app, "tower-test-result")
        XCTAssertTrue(result.label.contains("contract"), result.label)
        shoot(app, "07-settings-real-tower")
        app.terminate()
    }

    // MARK: Helpers

    private func launch(override: String?, reset: Bool) -> XCUIApplication {
        let app = XCUIApplication()
        // The first-run cards would cover the banner and the toolbar.
        app.launchArguments.append("-UITestSkipOnboarding")
        if reset { app.launchArguments.append("-UITestResetTowerAddress") }
        if let size = env("U02_CONTENT_SIZE") {
            app.launchArguments += ["-UIPreferredContentSizeCategoryName", size]
        }
        if let override { app.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = override }
        app.launch()
        return app
    }

    private func env(_ name: String) -> String? {
        let value = ProcessInfo.processInfo.environment[name]
        return value?.isEmpty == false ? value : nil
    }

    private func element(_ app: XCUIApplication, _ identifier: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    /// Replace the address field's text.
    private func enter(_ app: XCUIApplication, _ text: String) {
        let field = element(app, "tower-address-field")
        XCTAssertTrue(reveal(app, field, builtAbove: true), "the address field")
        // Bottom-right of a field that wraps: the end of its text.
        field.coordinate(withNormalizedOffset: CGVector(dx: 0.97, dy: 0.9)).tap()
        let current = (field.value as? String) ?? ""
        field.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: current.count + 4))
        field.typeText(text + "\n")
        XCTAssertEqual(field.value as? String, text, "the field holds what was typed")
    }

    /// Taps Test connection and returns the first line of the answer.
    private func runTest(_ app: XCUIApplication) -> String? {
        let test = app.buttons["tower-test"]
        XCTAssertTrue(reveal(app, test), "the Test connection button")
        XCTAssertTrue(test.isEnabled, "Test is on for a usable address")
        test.tap()
        let result = element(app, "tower-test-result")
        guard waitFor(timeout: 8, { result.exists }) else { return nil }
        reveal(app, result)
        let titles = ["Connected to a Tower", "Reached something, but it isn't a Glasses Tower", "Not reachable"]
        return titles.first { result.label.hasPrefix($0) }
    }

    /// Scroll until `element` can be tapped and is clear of the navigation
    /// bar, moving towards it: a short drag up when it is below, a short drag
    /// down when it is above. A lazy list may not have built a row that is
    /// off screen, so `builtAbove` says which way to look for one that does
    /// not exist yet. Never a drag down with Settings already at its top,
    /// which would pull the sheet away instead of scrolling it.
    @discardableResult
    private func reveal(_ app: XCUIApplication, _ element: XCUIElement,
                        builtAbove: Bool = false, attempts: Int = 12) -> Bool {
        for _ in 0..<attempts {
            let clearTop = topInset(app)
            if element.exists && element.isHittable && element.frame.minY >= clearTop - 1 {
                settleFullyOnScreen(app, element)
                return true
            }
            let window = app.windows.firstMatch.frame
            let isAbove = element.exists
                ? element.frame.minY < clearTop || element.frame.midY < window.midY
                : builtAbove
            if isAbove && settingsIsAtTop(app) { break }
            let from = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: isAbove ? 0.4 : 0.7))
            let to = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: isAbove ? 0.6 : 0.5))
            from.press(forDuration: 0.1, thenDragTo: to, withVelocity: .slow, thenHoldForDuration: 0.1)
        }
        return element.exists && element.isHittable && element.frame.minY >= topInset(app) - 1
    }

    /// The bottom of whichever navigation bar is in front.
    private func topInset(_ app: XCUIApplication) -> CGFloat {
        let settings = app.navigationBars["Settings"]
        if settings.exists { return settings.frame.maxY }
        let any = app.navigationBars.firstMatch
        return any.exists ? any.frame.maxY : 0
    }

    private func settingsIsAtTop(_ app: XCUIApplication) -> Bool {
        let first = element(app, "tower-in-use")
        return first.exists && first.frame.minY >= topInset(app) - 1
    }

    /// A row that is hittable can still be cut off at the bottom, which at the
    /// accessibility sizes hides most of a note. Nudge it up, a little at a
    /// time, while it overhangs and would fit.
    private func settleFullyOnScreen(_ app: XCUIApplication, _ element: XCUIElement) {
        let window = app.windows.firstMatch.frame
        for _ in 0..<4 {
            let frame = element.frame
            guard frame.maxY > window.maxY - 20, frame.height < window.height * 0.7 else { return }
            let from = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.7))
            let to = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.55))
            from.press(forDuration: 0.1, thenDragTo: to, withVelocity: .slow, thenHoldForDuration: 0.1)
        }
    }

    @discardableResult
    private func tap(_ element: XCUIElement, until effect: @autoclosure () -> Bool, attempts: Int = 4) -> Bool {
        for _ in 0..<attempts {
            if element.waitForExistence(timeout: 5), element.isHittable { element.tap() }
            if waitFor(timeout: 3, effect) { return true }
        }
        return effect()
    }

    private func waitFor(timeout: TimeInterval = 10, _ condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            Thread.sleep(forTimeInterval: 0.2)
        }
        return condition()
    }

    /// At an accessibility text size one row can fill the screen, so the
    /// row a shot is about continues below it: one more, a screen further on.
    private func shootFurther(_ app: XCUIApplication, _ name: String) {
        guard env("U02_CONTENT_SIZE") != nil, env("U02_SHOTS_DIR") != nil else { return }
        let from = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.8))
        let to = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.35))
        from.press(forDuration: 0.1, thenDragTo: to, withVelocity: .slow, thenHoldForDuration: 0.1)
        shoot(app, name + "-more")
    }

    private func shoot(_ app: XCUIApplication, _ name: String) {
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        guard let dir = env("U02_SHOTS_DIR") else { return }
        let suffix = env("U02_SHOT_SUFFIX").map { "-\($0)" } ?? ""
        let url = URL(fileURLWithPath: dir).appendingPathComponent("\(name)\(suffix).png")
        try? screenshot.pngRepresentation.write(to: url)
    }
}
