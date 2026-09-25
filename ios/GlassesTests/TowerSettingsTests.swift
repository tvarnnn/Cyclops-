//
//  TowerSettingsTests.swift
//  GlassesTests
//
//  U0.2: Settings' Tower address. What an address may be (per build), which
//  address wins (override > saved > built-in), that a saved one persists and
//  takes effect only at the next launch, that "Test connection" sends one
//  read-only request with a 5 s bound and says exactly one of three things,
//  and that the App Transport Security in Info.plist is the policy's twin.
//

import Foundation
import XCTest

@testable import Glasses

// MARK: - Validation

final class TowerAddressPolicyTests: XCTestCase {

    func testEveryBuildRefusesWhatIsNotAHostAndPort() {
        for policy in [TowerAddressPolicy.anyHost, .privateNetworks] {
            XCTAssertEqual(policy.check(""), .empty)
            XCTAssertEqual(policy.check("  \n"), .empty)
            XCTAssertEqual(policy.check("http://192.168.1.20:8000"), .includesScheme)
            XCTAssertEqual(policy.check("ws://192.168.1.20:8000/ws"), .includesScheme)
            for malformed in ["192.168.1.20:8000/ws", "user@192.168.1.20:1", "192.168.1.20:notaport",
                              "192.168.1.20:0", "192.168.1.20:65536", "192.168.1.20:99999",
                              ":8000", "192.168.1.20:8000?x", "%2F:1"] {
                XCTAssertEqual(policy.check(malformed), .malformed, "\(policy) \(malformed)")
            }
        }
    }

    func testADebugBuildAcceptsAnyWellFormedHost() {
        for candidate in ["127.0.0.1:8000", "8.8.8.8:8000", "tower.example.com:8000", "tower.local",
                          "[::1]:8000", " 192.168.1.20:8000 "] {
            guard case .accepted = TowerAddressPolicy.anyHost.check(candidate) else {
                return XCTFail("anyHost refused \(candidate)")
            }
        }
        XCTAssertEqual(TowerAddressPolicy.anyHost.check(" 192.168.1.20:8000\n"), .accepted("192.168.1.20:8000"))
    }

    func testAReleaseBuildAcceptsPrivateTailscaleLocalAndTsNetAddresses() {
        for candidate in ["10.1.2.3:8000", "172.16.0.1:8000", "172.31.255.255", "192.168.1.20:8000",
                          "169.254.10.20:8000", "100.64.0.1:8000", "100.127.255.254:8000",
                          "100.110.156.55:8000", "[fd7a:115c:a1e0::1]:8000", "[fc00::1]:8000",
                          "tower.local:8000", "Tower.Local", "desktop.tail1234.ts.net:8000"] {
            guard case .accepted = TowerAddressPolicy.privateNetworks.check(candidate) else {
                return XCTFail("privateNetworks refused \(candidate)")
            }
        }
    }

    func testAReleaseBuildRefusesTheInternetAndTheLookalikes() {
        for candidate in ["8.8.8.8:8000", "1.1.1.1", "172.15.255.255", "172.32.0.1", "100.63.255.255",
                          "100.128.0.1", "192.169.0.1", "127.0.0.1:8000", "[::1]:8000",
                          "[2001:db8::1]:8000", "example.com:8000", "ts.net", "evil-ts.net",
                          "tower.local.example.com", "10.0.0.1.example.com", "010.0.0.1",
                          "10.0.0.256", "tower", "[::ffff:10.0.0.1]:8000"] {
            XCTAssertEqual(TowerAddressPolicy.privateNetworks.check(candidate), .notPrivate, candidate)
        }
    }

    func testTheBuiltInAddressPassesBothPolicies() {
        let builtIn = TowerConfiguration.defaultAuthority
        XCTAssertEqual(TowerAddressPolicy.anyHost.check(builtIn), .accepted(builtIn))
        XCTAssertEqual(TowerAddressPolicy.privateNetworks.check(builtIn), .accepted(builtIn))
    }

    func testThisBuildsPolicyIsTheDebugOne() {
        #if DEBUG
        XCTAssertEqual(TowerAddressPolicy.current, .anyHost)
        #else
        XCTAssertEqual(TowerAddressPolicy.current, .privateNetworks)
        #endif
    }

    func testEveryRangeParsesAndMatchesOnlyItsOwnAddresses() {
        for cidr in TowerAddressPolicy.privateRanges {
            XCTAssertNotNil(TowerAddressRange(cidr), cidr)
        }
        let cgnat = TowerAddressRange("100.64.0.0/10")!
        XCTAssertTrue(cgnat.contains([100, 64, 0, 0]))
        XCTAssertTrue(cgnat.contains([100, 127, 255, 255]))
        XCTAssertFalse(cgnat.contains([100, 128, 0, 0]))
        XCTAssertFalse(cgnat.contains([100, 63, 255, 255]))
        XCTAssertFalse(cgnat.contains([UInt8](repeating: 0, count: 16)), "an IPv6 address is never in an IPv4 range")
        XCTAssertNil(TowerAddressRange("10.0.0.0/33"))
        XCTAssertNil(TowerAddressRange("10.0.0.0"))
        XCTAssertNil(TowerAddressRange("tower.local/8"))
    }

    func testTheWordsForEachProblem() {
        XCTAssertNil(TowerSettingsText.problem(.accepted("192.168.1.20:8000")))
        for check in [TowerAddressCheck.empty, .includesScheme, .malformed, .notPrivate] {
            let words = TowerSettingsText.problem(check)
            XCTAssertNotNil(words, "\(check)")
            XCTAssertTrue(words?.contains("192.168.1.20:8000") == true || check == .notPrivate, "\(check) gives an example")
        }
        XCTAssertTrue(TowerSettingsText.problem(.notPrivate)!.contains(".ts.net"))
    }
}

// MARK: - Precedence

final class TowerAddressPrecedenceTests: XCTestCase {

    private let override = "127.0.0.1:8010"
    private let saved = "192.168.1.20:8000"

    func testTheOverrideWinsOverASavedAddress() {
        let resolution = TowerConfiguration.resolve(overrideValue: override, savedValue: saved, policy: .anyHost)
        XCTAssertEqual(resolution, .init(authority: override, source: .developerOverride))
    }

    func testASavedAddressWinsOverTheBuiltInOne() {
        let resolution = TowerConfiguration.resolve(overrideValue: nil, savedValue: saved, policy: .anyHost)
        XCTAssertEqual(resolution, .init(authority: saved, source: .saved))
    }

    func testNothingSavedAndNoOverrideIsExactlyTheBuiltInAddress() {
        for policy in [TowerAddressPolicy.anyHost, .privateNetworks] {
            let resolution = TowerConfiguration.resolve(overrideValue: nil, savedValue: nil, policy: policy)
            XCTAssertEqual(resolution, .init(authority: "100.110.156.55:8000", source: .builtIn))
        }
    }

    /// A value present but unusable is skipped — never trusted, never fatal —
    /// and the next source down decides.
    func testAnUnusableValueFallsThroughToTheNextSource() {
        XCTAssertEqual(
            TowerConfiguration.resolve(overrideValue: "http://x", savedValue: saved, policy: .anyHost),
            .init(authority: saved, source: .saved)
        )
        XCTAssertEqual(
            TowerConfiguration.resolve(overrideValue: "  ", savedValue: "bad/path", policy: .anyHost),
            .init(authority: TowerConfiguration.defaultAuthority, source: .builtIn)
        )
        // An internet address saved by a Debug build is not used by a Release
        // one, whose ATS would refuse it.
        XCTAssertEqual(
            TowerConfiguration.resolve(overrideValue: nil, savedValue: "8.8.8.8:8000", policy: .privateNetworks),
            .init(authority: TowerConfiguration.defaultAuthority, source: .builtIn)
        )
        XCTAssertEqual(
            TowerConfiguration.resolve(overrideValue: nil, savedValue: "8.8.8.8:8000", policy: .anyHost),
            .init(authority: "8.8.8.8:8000", source: .saved)
        )
    }

    /// The override is syntax-checked only, as it always was: it is how the
    /// Simulator reaches `127.0.0.1`, which no Release policy would accept.
    func testTheOverrideIsNotSubjectToTheReleasePolicy() {
        XCTAssertEqual(
            TowerConfiguration.resolve(overrideValue: "127.0.0.1:8010", savedValue: nil, policy: .privateNetworks),
            .init(authority: "127.0.0.1:8010", source: .developerOverride)
        )
    }

    /// The running process followed the rule: what it uses is what `resolve`
    /// gives for its own environment and its own saved value.
    func testThisProcessResolvedByTheRule() {
        XCTAssertEqual(
            TowerConfiguration.resolution,
            TowerConfiguration.resolve(
                overrideValue: TowerConfiguration.launchOverrideValue,
                savedValue: TowerAddressStore().saved,
                policy: .current
            )
        )
        XCTAssertEqual(TowerConfiguration.authority, TowerConfiguration.resolution.authority)
        XCTAssertEqual(TowerConfiguration.httpBaseURL.absoluteString, "http://\(TowerConfiguration.authority)")
        XCTAssertEqual(TowerConfiguration.webSocketURL.absoluteString, "ws://\(TowerConfiguration.authority)/ws")
        #if DEBUG
        XCTAssertEqual(
            TowerConfiguration.launchOverrideValue,
            ProcessInfo.processInfo.environment["GLASSES_TOWER_AUTHORITY"]
        )
        #else
        XCTAssertNil(TowerConfiguration.launchOverrideValue)
        #endif
    }
}

// MARK: - Persistence and the Settings model

final class TowerSettingsModelTests: XCTestCase {

    private var suiteName = ""
    private var defaults: UserDefaults!

    override func setUp() {
        super.setUp()
        suiteName = "TowerSettingsModelTests.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suiteName)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suiteName)
        defaults = nil
        super.tearDown()
    }

    private var store: TowerAddressStore { TowerAddressStore(defaults: defaults) }

    @MainActor
    private func model(
        inUse: TowerConfiguration.Resolution = .init(authority: TowerConfiguration.defaultAuthority, source: .builtIn),
        override: String? = nil,
        policy: TowerAddressPolicy = .anyHost,
        tester: TowerConnectionTester = TowerConnectionTester(session: TowerProbeStubProtocol.makeSession())
    ) -> TowerSettingsModel {
        TowerSettingsModel(inUse: inUse, overrideValue: override, policy: policy, store: store, tester: tester)
    }

    func testTheStoreSavesReadsAndForgetsOneAddress() {
        XCTAssertNil(store.saved)
        store.save("192.168.1.20:8000")
        XCTAssertEqual(store.saved, "192.168.1.20:8000")
        XCTAssertEqual(defaults.string(forKey: "GlassesTowerAuthority"), "192.168.1.20:8000")
        // A second store over the same defaults — the next launch — reads it.
        XCTAssertEqual(TowerAddressStore(defaults: defaults).saved, "192.168.1.20:8000")
        store.clear()
        XCTAssertNil(store.saved)
    }

    /// With nothing saved, Settings offers nothing to do: the field holds the
    /// address in use, Save is off, and no restart is asked for.
    @MainActor
    func testNothingSavedChangesNothing() {
        let settings = model()
        XCTAssertEqual(settings.draft, "100.110.156.55:8000")
        XCTAssertNil(settings.saved)
        XCTAssertFalse(settings.canSave)
        XCTAssertFalse(settings.canUseBuiltIn)
        XCTAssertFalse(settings.needsRestart)
        XCTAssertTrue(settings.canTest)
        XCTAssertNil(store.saved, "opening Settings saves nothing")
    }

    @MainActor
    func testSavingPersistsTheNormalisedAddressAndAsksForARestart() {
        let settings = model()
        settings.draft = "  192.168.1.20:8000 \n"
        XCTAssertTrue(settings.canSave)
        settings.save()
        XCTAssertEqual(store.saved, "192.168.1.20:8000")
        XCTAssertEqual(settings.saved, "192.168.1.20:8000")
        XCTAssertEqual(settings.draft, "192.168.1.20:8000")
        XCTAssertFalse(settings.canSave, "saving the same address again would change nothing")
        XCTAssertTrue(settings.needsRestart)
        XCTAssertEqual(settings.inUse.authority, "100.110.156.55:8000", "the running app keeps its address")
        XCTAssertEqual(settings.nextLaunch, .init(authority: "192.168.1.20:8000", source: .saved))
        let words = TowerSettingsText.restartBody(inUse: settings.inUse.authority, next: settings.nextLaunch.authority)
        XCTAssertTrue(words.contains("100.110.156.55:8000") && words.contains("192.168.1.20:8000"))
        XCTAssertTrue(words.contains("app switcher"))

        // The next launch's Settings reads the saved address back.
        let relaunched = model(inUse: .init(authority: "192.168.1.20:8000", source: .saved))
        XCTAssertEqual(relaunched.draft, "192.168.1.20:8000")
        XCTAssertFalse(relaunched.needsRestart)
        XCTAssertTrue(relaunched.canUseBuiltIn)
    }

    @MainActor
    func testAnUnusableDraftCanNeitherBeTestedNorSaved() {
        let settings = model(policy: .privateNetworks)
        for draft in ["", "http://192.168.1.20:8000", "192.168.1.20:8000/ws", "8.8.8.8:8000"] {
            settings.draft = draft
            XCTAssertFalse(settings.canSave, draft)
            XCTAssertFalse(settings.canTest, draft)
            settings.save()
            settings.test()
            XCTAssertNil(store.saved, draft)
            XCTAssertEqual(settings.probe, .idle, draft)
        }
    }

    @MainActor
    func testUsingTheBuiltInAddressForgetsTheSavedOne() {
        store.save("192.168.1.20:8000")
        let settings = model(inUse: .init(authority: "192.168.1.20:8000", source: .saved))
        settings.useBuiltIn()
        XCTAssertNil(store.saved)
        XCTAssertNil(settings.saved)
        XCTAssertEqual(settings.draft, TowerConfiguration.defaultAuthority)
        XCTAssertTrue(settings.needsRestart, "the running app still uses the saved address")
        XCTAssertEqual(settings.nextLaunch.source, .builtIn)
    }

    /// With the developer override in force, a save is kept for later and no
    /// restart is promised: the override would win again.
    @MainActor
    func testUnderTheOverrideASaveIsForALaunchWithoutIt() {
        let settings = model(inUse: .init(authority: "127.0.0.1:8010", source: .developerOverride),
                             override: "127.0.0.1:8010")
        settings.draft = "192.168.1.20:8000"
        settings.save()
        XCTAssertEqual(store.saved, "192.168.1.20:8000")
        XCTAssertFalse(settings.needsRestart)
        XCTAssertEqual(settings.withoutOverride, .init(authority: "192.168.1.20:8000", source: .saved))
    }

    @MainActor
    func testTestingReportsTheOutcomeForTheAddressTested() async throws {
        TowerProbeStubProtocol.reset(answer: .init(status: 200, body: TowerProbeStubProtocol.declaration))
        let settings = model()
        settings.draft = "192.168.1.20:8000"
        settings.test()
        XCTAssertEqual(settings.probe, .testing("192.168.1.20:8000"))
        XCTAssertFalse(settings.canTest, "one probe at a time")
        try await waitUntil { settings.probe != .testing("192.168.1.20:8000") }
        XCTAssertEqual(settings.probe, .finished(
            "192.168.1.20:8000", .tower(contract: "cartridge_results.envelope/2026-08-23")
        ))
        XCTAssertNil(store.saved, "testing saves nothing")

        // Editing the address retires the answer, which was about the old one.
        settings.draft = "192.168.1.21:8000"
        XCTAssertEqual(settings.probe, .idle)
    }

    @MainActor
    private func waitUntil(_ condition: @MainActor () -> Bool, timeout: TimeInterval = 5) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() {
            if Date() > deadline { return XCTFail("timed out") }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
    }
}

// MARK: - Test connection

final class TowerConnectionTesterTests: XCTestCase {

    func testItSendsOneReadOnlyGetForTheDeclarationWithAFiveSecondBound() throws {
        let request = try XCTUnwrap(TowerConnectionTester.request(for: "192.168.1.20:8000"))
        XCTAssertEqual(request.httpMethod, "GET")
        XCTAssertEqual(request.url?.absoluteString, "http://192.168.1.20:8000/cartridges")
        XCTAssertEqual(request.timeoutInterval, 5)
        XCTAssertNil(request.httpBody)
        XCTAssertEqual(request.cachePolicy, .reloadIgnoringLocalCacheData)

        let configuration = TowerConnectionTester.sessionConfiguration()
        XCTAssertEqual(configuration.timeoutIntervalForRequest, 5)
        XCTAssertEqual(configuration.timeoutIntervalForResource, 5, "five seconds for the whole exchange")
        XCTAssertFalse(configuration.waitsForConnectivity)
        XCTAssertNil(configuration.urlCache)
    }

    func testATowersDeclarationIsATowerWithItsContract() async {
        TowerProbeStubProtocol.reset(answer: .init(status: 200, body: TowerProbeStubProtocol.declaration))
        let outcome = await TowerConnectionTester(session: TowerProbeStubProtocol.makeSession())
            .probe(authority: "192.168.1.20:8000")
        XCTAssertEqual(outcome, .tower(contract: "cartridge_results.envelope/2026-08-23"))
        let sent = TowerProbeStubProtocol.recorded()
        XCTAssertEqual(sent.map(\.line), ["GET /cartridges"], "exactly one request, and it is the read-only one")
        XCTAssertEqual(sent.first?.timeout, 5)
    }

    func testEverythingElseThatAnswersIsNotATower() async {
        let cases: [(TowerProbeStubProtocol.Answer, TowerProbeOutcome)] = [
            (.init(status: 404, body: #"{"detail":"Not Found"}"#), .notATower(.status(404))),
            (.init(status: 500, body: ""), .notATower(.status(500))),
            (.init(status: 200, body: "<html>router login</html>"), .notATower(.notADeclaration)),
            (.init(status: 200, body: #"{"status":"ok"}"#), .notATower(.notADeclaration)),
            (.init(status: 200, body: #"{"type":"cartridges"}"#), .notATower(.notADeclaration)),
            (.init(status: 200, body: #"{"type":"cartridges","cartridges":[]}"#), .tower(contract: nil)),
        ]
        for (answer, expected) in cases {
            TowerProbeStubProtocol.reset(answer: answer)
            let outcome = await TowerConnectionTester(session: TowerProbeStubProtocol.makeSession())
                .probe(authority: "192.168.1.20:8000")
            XCTAssertEqual(outcome, expected, "\(answer)")
        }
    }

    /// A redirect is never followed, so the one request cannot become a
    /// request somewhere else; the 3xx itself is the answer.
    func testARedirectIsAnswerNotFollowed() {
        let response = HTTPURLResponse(url: URL(string: "http://192.168.1.20:8000/cartridges")!,
                                       statusCode: 302, httpVersion: "HTTP/1.1",
                                       headerFields: ["Location": "http://example.com/"])!
        XCTAssertEqual(TowerConnectionTester.classify(data: Data(), response: response), .notATower(.status(302)))
    }

    func testTransportFailuresAreNotReachableUnlessSomethingAnswered() {
        let cases: [(URLError.Code, TowerProbeOutcome)] = [
            (.timedOut, .notReachable(.timedOut)),
            (.cannotConnectToHost, .notReachable(.refused)),
            (.cannotFindHost, .notReachable(.nameNotFound)),
            (.notConnectedToInternet, .notReachable(.noNetwork)),
            (.appTransportSecurityRequiresSecureConnection, .notReachable(.blockedByTransportSecurity)),
            (.networkConnectionLost, .notReachable(.other)),
            (.badServerResponse, .notATower(.notHTTP)),
            (.cannotParseResponse, .notATower(.notHTTP)),
        ]
        for (code, expected) in cases {
            XCTAssertEqual(TowerConnectionTester.classify(error: URLError(code)), expected, "\(code)")
        }
    }

    /// A real socket, no stub: nothing listens on port 1 of this Mac.
    func testAClosedPortIsNotReachable() async {
        let started = Date()
        let outcome = await TowerConnectionTester().probe(authority: "127.0.0.1:1")
        XCTAssertEqual(outcome, .notReachable(.refused))
        XCTAssertLessThan(Date().timeIntervalSince(started), 5.5)
    }

    func testTheThreeAnswersAreWordedExactly() {
        XCTAssertEqual(TowerSettingsText.outcomeTitle(.notReachable(.timedOut)), "Not reachable")
        XCTAssertEqual(TowerSettingsText.outcomeTitle(.notATower(.status(404))),
                       "Reached something, but it isn't a Glasses Tower")
        XCTAssertEqual(TowerSettingsText.outcomeTitle(.tower(contract: nil)), "Connected to a Tower")
        XCTAssertTrue(TowerSettingsText.outcomeDetail(.tower(contract: "c/1"), authority: "h:1").contains("c/1"))
        XCTAssertTrue(TowerSettingsText.outcomeDetail(.notReachable(.timedOut), authority: "h:1").contains("5 seconds"))
    }

    /// Settings cannot start anything on a Tower. Its files name no socket
    /// client, no session route and no verb but GET.
    func testSettingsSourcesReachNothingButTheProbe() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Glasses")
        for file in ["TowerConnectionTester.swift", "Views/SettingsView.swift", "TowerAddress.swift"] {
            let url = root.appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: url.path) else {
                #if targetEnvironment(simulator)
                return XCTFail("sources not readable at \(url.path)")
                #else
                throw XCTSkip("sources are not on a device")
                #endif
            }
            // Code only: the doc comments name what the code never does.
            let text = try String(contentsOf: url, encoding: .utf8)
                .components(separatedBy: "\n")
                .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
                .joined(separator: "\n")
            for forbidden in ["TowerClient(", "tower.connect", "\"POST\"", "\"PUT\"", "\"DELETE\"",
                              "session/start", "WorldBuilderWorkspaceView", "NavigationLink"] {
                XCTAssertFalse(text.contains(forbidden), "\(file) contains \(forbidden)")
            }
        }
    }
}

// MARK: - App Transport Security

/// `Info.plist` is the Release policy; the Debug build derives from it. Both
/// are checked here against `TowerAddressPolicy`, so the list Settings accepts
/// and the list ATS lets through cannot drift apart.
final class TowerTransportSecurityTests: XCTestCase {

    private func sourceInfoPlist() throws -> [String: Any] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Glasses/Info.plist")
        guard FileManager.default.fileExists(atPath: url.path) else {
            #if targetEnvironment(simulator)
            throw NSError(domain: "TowerTransportSecurityTests", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Info.plist not readable at \(url.path)"])
            #else
            throw XCTSkip("sources are not on a device")
            #endif
        }
        let data = try Data(contentsOf: url)
        return try XCTUnwrap(PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any])
    }

    private func ats(_ plist: [String: Any]) throws -> [String: Any] {
        try XCTUnwrap(plist["NSAppTransportSecurity"] as? [String: Any])
    }

    private func exceptionDomains(_ ats: [String: Any]) throws -> [String: [String: Any]] {
        try XCTUnwrap(ats["NSExceptionDomains"] as? [String: [String: Any]])
    }

    private var builtInHost: String {
        String(TowerConfiguration.defaultAuthority.split(separator: ":").first!)
    }

    func testTheReleasePolicyIsLocalNetworkingAndThePrivateRangesOnly() throws {
        let ats = try ats(try sourceInfoPlist())
        XCTAssertEqual(ats["NSAllowsLocalNetworking"] as? Bool, true)
        XCTAssertNil(ats["NSAllowsArbitraryLoads"], "Release must not allow arbitrary loads")
        XCTAssertNil(ats["NSAllowsArbitraryLoadsInWebContent"])

        let domains = try exceptionDomains(ats)
        let expected = Set(TowerAddressPolicy.privateRanges + ["ts.net", builtInHost])
        XCTAssertEqual(Set(domains.keys), expected, "the exceptions and TowerAddressPolicy disagree")
        for (domain, settings) in domains {
            XCTAssertEqual(settings["NSExceptionAllowsInsecureHTTPLoads"] as? Bool, true, domain)
        }
        XCTAssertEqual(domains["ts.net"]?["NSIncludesSubdomains"] as? Bool, true)
        XCTAssertEqual(TowerAddressPolicy.privateNameSuffixes, ["local", "ts.net"],
                       "`.local` is NSAllowsLocalNetworking's; ts.net is the one name exception")
    }

    /// The app this suite runs in is the Debug build: arbitrary loads on, and
    /// NSAllowsLocalNetworking gone, because its presence would make iOS
    /// ignore arbitrary loads. Everything else is the source file's.
    func testTheDebugBuildAllowsArbitraryLoads() throws {
        let built = try XCTUnwrap(Bundle.main.infoDictionary)
        XCTAssertEqual(Bundle.main.bundleIdentifier, "com.tristanvarner.Glasses", "read the host app's Info.plist")
        let builtATS = try ats(built)
        #if DEBUG
        XCTAssertEqual(builtATS["NSAllowsArbitraryLoads"] as? Bool, true)
        XCTAssertNil(builtATS["NSAllowsLocalNetworking"], "its presence disables NSAllowsArbitraryLoads")
        #else
        XCTAssertNil(builtATS["NSAllowsArbitraryLoads"])
        XCTAssertEqual(builtATS["NSAllowsLocalNetworking"] as? Bool, true)
        #endif
        let sourceDomains = try exceptionDomains(try ats(try sourceInfoPlist()))
        XCTAssertEqual(Set(try exceptionDomains(builtATS).keys), Set(sourceDomains.keys))
        // The rest of the plist came through the derive phase untouched.
        XCTAssertEqual((built["MWDAT"] as? [String: Any])?["AppLinkURLScheme"] as? String, "glasses://")
        XCTAssertEqual(built["UISupportedExternalAccessoryProtocols"] as? [String], ["com.meta.ar.wearable"])
    }
}

// MARK: - A network stand-in

/// Answers every request with one canned response and records what was sent.
final class TowerProbeStubProtocol: URLProtocol {
    struct Answer: CustomStringConvertible {
        let status: Int
        let body: String
        var description: String { "\(status) \(body.prefix(40))" }
    }

    static let declaration = #"""
        {"type":"cartridges","envelope_contract":"cartridge_results.envelope/2026-08-23",
         "cartridges":[],"not_offered":[],"http_contracts":[]}
        """#

    private static let lock = NSLock()
    private static var answer = Answer(status: 500, body: "")
    private static var requests: [(line: String, timeout: TimeInterval)] = []

    static func reset(answer: Answer) {
        lock.lock()
        defer { lock.unlock() }
        self.answer = answer
        requests = []
    }

    static func recorded() -> [(line: String, timeout: TimeInterval)] {
        lock.lock()
        defer { lock.unlock() }
        return requests
    }

    static func makeSession() -> URLSession {
        let configuration = TowerConnectionTester.sessionConfiguration()
        configuration.protocolClasses = [TowerProbeStubProtocol.self]
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.lock.lock()
        Self.requests.append(("\(request.httpMethod ?? "GET") \(request.url?.path ?? "")", request.timeoutInterval))
        let answer = Self.answer
        Self.lock.unlock()
        guard let url = request.url,
              let response = HTTPURLResponse(url: url, statusCode: answer.status, httpVersion: "HTTP/1.1",
                                             headerFields: ["Content-Type": "application/json"])
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(answer.body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
