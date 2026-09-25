//
//  MockTowerHTTPServer.swift
//  GlassesUITests
//
//  The HTTP half of a Tower, as far as Settings' "Test connection" can see it,
//  served from the UI-test process on the Mac's loopback — which the app in
//  the Simulator shares. Built like `GlassesTests/MockTowerServer.swift` (the
//  WebSocket mock), on Network.framework's `NWListener`, and for the same
//  reason: no third-party server on either side of the test. That one speaks
//  only WebSocket frames, which a plain `GET` never reaches, so this is its
//  sibling rather than a reuse.
//
//  It records every request line it is sent, which is how a test proves the
//  app sent exactly one `GET /cartridges` and nothing that could change a
//  Tower.
//

import Foundation
import Network

final class MockTowerHTTPServer: @unchecked Sendable {
    enum Personality {
        /// Answers `GET /cartridges` with a Tower's declaration; 404 otherwise.
        case tower
        /// Answers everything 404, like a web server that is not a Tower.
        case notATower
    }

    static let contract = "cartridge_results.envelope/2026-08-23"
    private static let declaration = """
        {"type":"cartridges","envelope_contract":"\(contract)","cartridges":[],"not_offered":[],"http_contracts":[]}
        """

    private let listener: NWListener
    private let personality: Personality
    private let queue = DispatchQueue(label: "MockTowerHTTPServer")
    private var lines: [String] = []
    private var startFailure: Error?

    init(_ personality: Personality) throws {
        self.personality = personality
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        // Loopback only: nothing off this Mac can reach the mock.
        parameters.requiredInterfaceType = .loopback
        listener = try NWListener(using: parameters)
    }

    /// Starts listening and returns the system-assigned port.
    func start(timeout: TimeInterval = 5) throws -> UInt16 {
        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready: ready.signal()
            case .failed(let error): self?.startFailure = error; ready.signal()
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
        listener.start(queue: queue)
        guard ready.wait(timeout: .now() + timeout) == .success else {
            throw NSError(domain: "MockTowerHTTPServer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "the listener never became ready"])
        }
        if let failure = queue.sync(execute: { startFailure }) { throw failure }
        guard let port = listener.port?.rawValue else {
            throw NSError(domain: "MockTowerHTTPServer", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "the listener has no port"])
        }
        return port
    }

    /// Every request line received so far, like `GET /cartridges HTTP/1.1`.
    var requestLines: [String] { queue.sync { lines } }

    func stop() {
        listener.cancel()
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(on: connection, buffered: Data())
    }

    private func receive(on connection: NWConnection, buffered: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffered = buffered
            if let data { buffered.append(data) }
            if let end = buffered.range(of: Data("\r\n\r\n".utf8)) {
                let head = String(decoding: buffered[..<end.lowerBound], as: UTF8.self)
                let line = head.components(separatedBy: "\r\n").first ?? ""
                self.lines.append(line)
                self.respond(on: connection, to: line)
            } else if error == nil, !isComplete {
                self.receive(on: connection, buffered: buffered)
            } else {
                connection.cancel()
            }
        }
    }

    private func respond(on connection: NWConnection, to line: String) {
        let isDeclarationRequest = line.hasPrefix("GET /cartridges ")
        let (status, body) = personality == .tower && isDeclarationRequest
            ? ("200 OK", Self.declaration)
            : ("404 Not Found", #"{"detail":"Not Found"}"#)
        let response = "HTTP/1.1 \(status)\r\nContent-Type: application/json\r\n"
            + "Content-Length: \(body.utf8.count)\r\nConnection: close\r\n\r\n\(body)"
        connection.send(content: Data(response.utf8), completion: .contentProcessed { _ in
            connection.cancel()
        })
    }
}
