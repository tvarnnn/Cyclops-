//
//  OnboardingUITests.swift
//  GlassesUITests
//
//  U0.3: "How Glasses works", end to end in the Simulator. A first launch
//  shows the cards; Skip and Get started both close them for good; Settings
//  reopens them; the second card's links open Connections and Settings over
//  them; VoiceOver reads a card before its controls; and every card passes
//  the accessibility audit.
//
//  Runs by default and needs no Tower. The app's own socket is pointed at a
//  closed loopback port, so no launch here reaches a real Tower.
//
//  Screenshots, opt-in: TEST_RUNNER_U03_SHOTS_DIR=<dir> writes PNGs there,
//  named with TEST_RUNNER_U03_SHOT_SUFFIX; TEST_RUNNER_U03_CONTENT_SIZE=<UIKit
//  content size category name> launches every test at that text size.
//

import XCTest

final class OnboardingUITests: XCTestCase {

    /// Nothing listens on port 9 of this Mac.
    private let closedAuthority = "127.0.0.1:9"
    private let slugs = ["what-it-does", "what-you-need", "how-to-walk", "what-you-see"]

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    // MARK: The tests

    func testAFirstLaunchShowsTheCardsAndGetStartedClosesThemForGood() throws {
        var app = launch(reset: true)
        XCTAssertTrue(waitForPage(app, 1), "a first launch opens on the first card")
        XCTAssertEqual(closeButton(app).label, "Skip")

        for page in 1...4 {
            XCTAssertTrue(waitForPage(app, page))
            XCTAssertTrue(reveal(app, title(app, slugs[page - 1])), "card \(page)'s title")
            if page < 4 {
                XCTAssertTrue(closeButton(app).isHittable, "Skip is on card \(page)")
                XCTAssertFalse(app.buttons["onboarding-finish"].exists)
                XCTAssertTrue(next(app, to: page + 1), "Continue shows card \(page + 1)")
            }
        }
        let finish = app.buttons["onboarding-finish"]
        XCTAssertTrue(finish.waitForExistence(timeout: 5))
        XCTAssertEqual(finish.label, "Get started")
        XCTAssertTrue(tap(finish, until: !self.page(app).exists), "Get started closes the cards")
        XCTAssertTrue(app.buttons["Cartridges"].waitForExistence(timeout: 5), "Home is underneath")
        XCTAssertTrue(app.buttons["Cartridges"].isHittable)
        app.terminate()

        // Seen: the next launch opens on Home.
        app = launch(reset: false)
        XCTAssertTrue(app.buttons["Cartridges"].waitForExistence(timeout: 10))
        XCTAssertFalse(page(app).waitForExistence(timeout: 3), "the cards do not come back")
        app.terminate()
    }

    func testSkipClosesThemForGoodFromAnyCard() throws {
        var app = launch(reset: true)
        XCTAssertTrue(waitForPage(app, 1))
        // A swipe turns the page both ways, as well as Continue.
        XCTAssertTrue(swipe(app, towardsNext: true, to: 2), "a swipe to the left shows the next card")
        XCTAssertTrue(swipe(app, towardsNext: false, to: 1), "a swipe to the right shows the one before")
        XCTAssertTrue(next(app, to: 2))
        XCTAssertTrue(tap(closeButton(app), until: !self.page(app).exists), "Skip closes the cards")
        XCTAssertTrue(app.buttons["Cartridges"].isHittable, "Home is underneath")
        app.terminate()

        app = launch(reset: false)
        XCTAssertTrue(app.buttons["Cartridges"].waitForExistence(timeout: 10))
        XCTAssertFalse(page(app).waitForExistence(timeout: 3), "skipped counts as seen")
        app.terminate()
    }

    func testSettingsReopensTheCards() throws {
        let app = launch(reset: false, skip: true)
        XCTAssertTrue(tap(app.buttons["Settings"], until: app.navigationBars["Settings"].exists),
                      "the toolbar opens Settings")
        let row = app.buttons["settings-how-it-works"]
        XCTAssertTrue(revealBelow(app, row), "Settings offers How Glasses works")
        XCTAssertEqual(row.label, "How Glasses works")
        XCTAssertTrue(row.isEnabled, "no capture is running")
        shoot(app, "05-settings-how-it-works")

        XCTAssertTrue(tap(row, until: self.page(app).exists), "the row opens the cards")
        XCTAssertTrue(waitForPage(app, 1), "the cards open on the first card")
        XCTAssertFalse(app.navigationBars["Settings"].exists, "Settings closed first")
        XCTAssertEqual(closeButton(app).label, "Close")
        for page in 2...4 { XCTAssertTrue(next(app, to: page)) }
        let done = app.buttons["onboarding-finish"]
        XCTAssertEqual(done.label, "Done")
        XCTAssertTrue(tap(done, until: !self.page(app).exists), "Done closes the cards")
        XCTAssertTrue(app.buttons["Cartridges"].isHittable)

        // And Close, from the middle.
        XCTAssertTrue(tap(app.buttons["Settings"], until: app.navigationBars["Settings"].exists))
        XCTAssertTrue(revealBelow(app, row))
        XCTAssertTrue(tap(row, until: self.page(app).exists))
        XCTAssertTrue(waitForPage(app, 1))
        XCTAssertTrue(next(app, to: 2))
        XCTAssertTrue(tap(closeButton(app), until: !self.page(app).exists), "Close closes the cards")
        app.terminate()
    }

    func testTheSecondCardOpensConnectionsAndSettingsOverTheCards() throws {
        let app = launch(reset: true)
        XCTAssertTrue(waitForPage(app, 1))
        XCTAssertTrue(next(app, to: 2))

        let settings = app.buttons["onboarding-open-settings"]
        XCTAssertTrue(reveal(app, settings))
        XCTAssertTrue(tap(settings, until: app.navigationBars["Settings"].exists), "the link opens Settings")
        XCTAssertTrue(element(app, "tower-in-use").waitForExistence(timeout: 5), "with the Tower address")
        XCTAssertFalse(app.buttons["settings-how-it-works"].exists, "no way back into the cards from inside them")
        XCTAssertTrue(tap(app.navigationBars["Settings"].buttons["Done"],
                          until: !app.navigationBars["Settings"].exists))
        XCTAssertTrue(waitForPage(app, 2), "back on the same card")

        let connections = app.buttons["onboarding-open-connections"]
        XCTAssertTrue(reveal(app, connections))
        XCTAssertTrue(tap(connections, until: app.navigationBars["Connections"].exists),
                      "the link opens Connections")
        XCTAssertTrue(tap(app.navigationBars["Connections"].buttons["Done"],
                          until: !app.navigationBars["Connections"].exists))
        XCTAssertTrue(waitForPage(app, 2), "back on the same card")
        app.terminate()
    }

    /// Contrast, clipped text, Dynamic Type, descriptions and hit regions, on
    /// every card, at whatever text size and appearance the run uses. Each
    /// card is also shot when TEST_RUNNER_U03_SHOTS_DIR is set.
    func testEveryCardPassesTheAccessibilityAudit() throws {
        let app = launch(reset: true)
        XCTAssertTrue(waitForPage(app, 1))
        for page in 1...4 {
            XCTAssertTrue(waitForPage(app, page))
            // The indicator changes at once; the page and the button's label
            // are still sliding and cross-fading. Shoot and audit the settled card.
            Thread.sleep(forTimeInterval: 1)
            let name = String(format: "%02d-%@", page, slugs[page - 1])
            shoot(app, name)
            shootFurther(app, name, card: page)
            try audit(app, card: page)
            if page < 4 {
                XCTAssertTrue(next(app, to: page + 1))
            }
        }
        app.terminate()
    }

    /// What VoiceOver meets, in order: the card's title (a header), its
    /// words, each point as one element that says both its title and its
    /// sentence, then where you are ("Card, 1 of 4"), then Continue, then Skip.
    func testVoiceOverReadsTheCardBeforeItsControls() throws {
        let app = launch(reset: true)
        XCTAssertTrue(waitForPage(app, 1))
        XCTAssertTrue(next(app, to: 2))
        XCTAssertTrue(next(app, to: 3))
        Thread.sleep(forTimeInterval: 1)

        let title = title(app, "how-to-walk")
        XCTAssertTrue(title.exists)
        XCTAssertEqual(title.label, "How to walk")
        XCTAssertTrue(title.elementType == .staticText)

        let pocket = app.descendants(matching: .any)
            .matching(NSPredicate(format: "label BEGINSWITH %@", "Keep your phone in your pocket")).firstMatch
        XCTAssertTrue(pocket.exists, "the point is one element")
        XCTAssertTrue(pocket.label.contains("confuses the world"), pocket.label)

        XCTAssertEqual(page(app).label, "Card")
        XCTAssertEqual(page(app).value as? String, "3 of 4")

        // Reading order is the tree's depth-first order, which is how
        // `debugDescription` prints it. (`allElementsBoundByIndex` is not:
        // it lists shallower elements first, so the footer would come
        // before a card that sits deeper, inside the pager.)
        let tree = app.debugDescription
        let positions = ["onboarding-title-how-to-walk", "onboarding-page",
                         "onboarding-continue", "onboarding-close"]
            .map { tree.range(of: "'\($0)'")?.lowerBound }
        XCTAssertFalse(positions.contains(nil), "every element is in the tree")
        let found = positions.compactMap { $0 }
        XCTAssertEqual(found, found.sorted(), "title, then Card, then Continue, then Skip")
        XCTAssertEqual(closeButton(app).label, "Skip")
        app.terminate()
    }

    // MARK: Helpers

    /// Every issue on the card, each named with its element, rather than the
    /// first one alone.
    private func audit(_ app: XCUIApplication, card: Int) throws {
        var found: [String] = []
        try app.performAccessibilityAudit { issue in
            let element = issue.element.map { "\($0.elementType.rawValue) '\($0.identifier)' '\($0.label)'" } ?? "-"
            found.append("card \(card): \(issue.compactDescription) [\(issue.detailedDescription)] on \(element)")
            return true
        }
        XCTAssertEqual(found, [], found.joined(separator: "\n"))
    }

    private func launch(reset: Bool, skip: Bool = false) -> XCUIApplication {
        let app = XCUIApplication()
        if reset { app.launchArguments.append("-UITestResetOnboarding") }
        if skip { app.launchArguments.append("-UITestSkipOnboarding") }
        if let size = env("U03_CONTENT_SIZE") {
            app.launchArguments += ["-UIPreferredContentSizeCategoryName", size]
        }
        app.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = closedAuthority
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

    private func page(_ app: XCUIApplication) -> XCUIElement { element(app, "onboarding-page") }
    private func closeButton(_ app: XCUIApplication) -> XCUIElement { app.buttons["onboarding-close"] }
    private func title(_ app: XCUIApplication, _ slug: String) -> XCUIElement {
        element(app, "onboarding-title-\(slug)")
    }

    /// The page indicator reads "N of 4": the card that is on screen, rather
    /// than one the pager keeps built beside it.
    private func waitForPage(_ app: XCUIApplication, _ number: Int, timeout: TimeInterval = 10) -> Bool {
        waitFor(timeout: timeout) {
            let indicator = self.page(app)
            return indicator.exists && (indicator.value as? String) == "\(number) of 4"
        }
    }

    /// Continue, until card `number` is showing.
    ///
    /// Tapped again only if the page has not turned, so it can never skip a
    /// card. The retry is for the first launch after xcodebuild boots the
    /// Simulator, where a first tap can highlight the button and deliver
    /// nothing. Instrumented on the 8 GB Mac (deep in swap): UIKit logged the
    /// touch dispatched to the window and "send gesture actions", and the
    /// Button's action never ran; the second tap ran it, just before UIKit
    /// logged "System gesture gate timed out". Every later launch in the same
    /// run takes the first tap.
    private func next(_ app: XCUIApplication, to number: Int) -> Bool {
        let button = app.buttons["onboarding-continue"]
        for attempt in 0..<3 {
            if attempt > 0 {
                XCTContext.runActivity(named: "Continue tapped again: card \(number) did not show") { _ in }
            }
            if button.waitForExistence(timeout: 5), button.isHittable { button.tap() }
            if waitForPage(app, number, timeout: 4) { return true }
        }
        return false
    }

    /// Swipe the pager, until card `number` is showing.
    private func swipe(_ app: XCUIApplication, towardsNext: Bool, to number: Int) -> Bool {
        for _ in 0..<3 {
            if towardsNext { app.swipeLeft() } else { app.swipeRight() }
            if waitForPage(app, number, timeout: 4) { return true }
        }
        return false
    }

    /// Scroll the card up until `element` can be hit, then back down.
    @discardableResult
    private func reveal(_ app: XCUIApplication, _ element: XCUIElement, attempts: Int = 8) -> Bool {
        if element.waitForExistence(timeout: 5), element.isHittable { return true }
        for _ in 0..<attempts {
            drag(app, from: 0.7, to: 0.4)
            if element.exists && element.isHittable { return true }
        }
        for _ in 0..<attempts {
            drag(app, from: 0.3, to: 0.6)
            if element.exists && element.isHittable { return true }
        }
        return element.exists && element.isHittable
    }

    /// Scroll down only, until `element` can be hit: for the last row of
    /// Settings, which at the accessibility sizes is many screens down and
    /// not built until it is near. Never a drag down, which at the top of a
    /// sheet would pull the sheet away.
    private func revealBelow(_ app: XCUIApplication, _ element: XCUIElement, attempts: Int = 40) -> Bool {
        // All of it, clear of the home indicator: a row at the very bottom
        // edge is already "hittable", and would be shot half off the screen.
        func onScreen() -> Bool {
            guard element.exists, element.isHittable else { return false }
            return element.frame.maxY <= app.windows.firstMatch.frame.maxY - 40
        }
        _ = element.waitForExistence(timeout: 3)
        for _ in 0..<attempts {
            if onScreen() { return true }
            drag(app, from: 0.7, to: 0.4)
        }
        return onScreen()
    }

    /// A vertical drag, which the pager leaves to the card's own scroll view.
    private func drag(_ app: XCUIApplication, from: CGFloat, to: CGFloat) {
        let start = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: from))
        let end = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: to))
        start.press(forDuration: 0.1, thenDragTo: end, withVelocity: .slow, thenHoldForDuration: 0.1)
    }

    /// Tap `element` until `effect` is true. Tapped again only while the
    /// effect is missing, so a tap is never doubled; see `next(_:to:)` for
    /// why a first tap can go missing.
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

    /// At the accessibility sizes a card is taller than the screen: shoot it
    /// again a screen further down, until the bottom has been seen, then
    /// scroll back so the next card starts from the top.
    private func shootFurther(_ app: XCUIApplication, _ name: String, card: Int) {
        guard env("U03_CONTENT_SIZE") != nil, env("U03_SHOTS_DIR") != nil else { return }
        // Dragged inside the card, above the footer, which at these sizes
        // takes the bottom quarter of the screen.
        for more in 1...10 {
            let before = app.screenshot().pngRepresentation
            drag(app, from: 0.6, to: 0.15)
            Thread.sleep(forTimeInterval: 1)
            let after = app.screenshot()
            if after.pngRepresentation == before { break }
            shoot(app, "\(name)-more\(more)", screenshot: after)
        }
        for _ in 1...12 { drag(app, from: 0.15, to: 0.6) }
    }

    private func shoot(_ app: XCUIApplication, _ name: String, screenshot: XCUIScreenshot? = nil) {
        let screenshot = screenshot ?? app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        guard let dir = env("U03_SHOTS_DIR") else { return }
        let suffix = env("U03_SHOT_SUFFIX").map { "-\($0)" } ?? ""
        let url = URL(fileURLWithPath: dir).appendingPathComponent("\(name)\(suffix).png")
        try? screenshot.pngRepresentation.write(to: url)
    }
}
