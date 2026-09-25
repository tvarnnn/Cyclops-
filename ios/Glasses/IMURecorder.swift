//
//  IMURecorder.swift
//  Glasses
//

// Entire file is DEBUG-only, like the Motion probe beside it and the Developer
// Tools switch that arms it. It is walk-5 evidence tooling: the glasses IMU and
// the timing of every camera frame, written to a file on this phone and joined
// offline with the Tower's own capture. Nothing here is sent to the Tower (no
// contract change), shown to a wearer, or played as a sound or a haptic.
#if DEBUG

import Combine
import CoreMedia
import Foundation
import MWDATCore
import MWDATMotion

// MARK: - Values

/// One Motion reading, reduced to the fields the log keeps.
///
/// The app's own type rather than `MotionSample`, so the log format can be
/// tested from fixed numbers without the SDK, and so Mock Device Kit's
/// same-named `MotionSample` never has to be disambiguated in a test.
nonisolated struct IMUMotionReading: Equatable, Sendable {
    /// `MotionSource`, spelled the way the log spells it.
    enum Source: String, Sendable {
        case glasses
        case neuralBand = "neural_band"
        case unknown
    }

    /// `MotionSample.timestampNs`: the glasses' own monotonic clock.
    let timestampNs: Int64
    let accelerometer: SIMD3<Float>?
    let gyroscope: SIMD3<Float>?
    /// The fused orientation as (w, x, y, z), the order the log writes it in.
    let orientation: SIMD4<Float>?
    let source: Source

    init(
        timestampNs: Int64,
        accelerometer: SIMD3<Float>?,
        gyroscope: SIMD3<Float>?,
        orientation: SIMD4<Float>?,
        source: Source
    ) {
        self.timestampNs = timestampNs
        self.accelerometer = accelerometer
        self.gyroscope = gyroscope
        self.orientation = orientation
        self.source = source
    }

    init(_ sample: MotionSample) {
        let source: Source
        switch sample.source {
        case .glasses: source = .glasses
        case .neuralBand: source = .neuralBand
        case .unknown: source = .unknown
        }
        self.init(
            timestampNs: sample.timestampNs,
            accelerometer: sample.accelerometer.map { SIMD3($0.x, $0.y, $0.z) },
            gyroscope: sample.gyroscope.map { SIMD3($0.x, $0.y, $0.z) },
            orientation: sample.orientation.map { SIMD4($0.w, $0.x, $0.y, $0.z) },
            source: source
        )
    }
}

/// The two clocks read off one camera frame **on DAT's callback thread**,
/// before the main-actor hop, for the same reason `FramePTSProbe` reads there:
/// a receipt time taken after the hop carries main-actor queueing as well as
/// transport.
nonisolated struct IMUFrameStamp: Equatable, Sendable {
    /// `CMSampleBufferGetPresentationTimeStamp` in microseconds, or `nil` when
    /// DAT's buffer carried an invalid or indefinite time.
    let ptsUs: Int64?
    /// This phone's `DispatchTime` (`mach_absolute_time`) in nanoseconds: the
    /// same base as `MonotonicClock` and as the Motion lines' `rx_mono_ns`.
    let rxMonoNs: UInt64

    init(ptsUs: Int64?, rxMonoNs: UInt64) {
        self.ptsUs = ptsUs
        self.rxMonoNs = rxMonoNs
    }

    init(sampleBuffer: CMSampleBuffer, rxMonoNs: UInt64) {
        self.init(
            ptsUs: Self.microseconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)),
            rxMonoNs: rxMonoNs
        )
    }

    static func microseconds(_ time: CMTime) -> Int64? {
        guard time.isValid, !time.isIndefinite, time.isNumeric, time.timescale > 0 else { return nil }
        return CMTimeConvertScale(time, timescale: 1_000_000, method: .roundHalfAwayFromZero).value
    }
}

/// One `addMotion` attempt, for the header: which rate was asked for, and why
/// it was refused if it was.
nonisolated struct IMUMotionAttempt: Equatable, Sendable {
    let hz: Int
    let error: String?
}

/// The glasses' `DeviceState`, as strings, so a change can be detected by
/// comparison and written without the SDK's types.
nonisolated struct IMUDeviceStateSnapshot: Equatable, Sendable {
    let thermal: String
    let batteryPercent: Int?
    let charging: String
    let don: String
    let hinge: String
    let link: String
    let compatibility: String

    init(thermal: String, batteryPercent: Int?, charging: String, don: String, hinge: String, link: String, compatibility: String) {
        self.thermal = thermal
        self.batteryPercent = batteryPercent
        self.charging = charging
        self.don = don
        self.hinge = hinge
        self.link = link
        self.compatibility = compatibility
    }

    init(_ state: DeviceState) {
        self.init(
            thermal: "\(state.thermalLevel)",
            batteryPercent: state.batteryLevel,
            charging: "\(state.chargingState)",
            don: "\(state.donState)",
            hinge: "\(state.hingeState)",
            link: "\(state.linkState)",
            compatibility: "\(state.compatibility)"
        )
    }
}

extension MotionSamplingRate {
    /// The rate in hertz, for the log and for gap detection.
    nonisolated var hertz: Int {
        switch self {
        case .hz5: return 5
        case .hz10: return 10
        case .hz15: return 15
        case .hz24: return 24
        case .hz30: return 30
        case .hz60: return 60
        }
    }
}

// MARK: - JSON lines

/// A JSON value as the log writes it. Hand-rolled rather than
/// `JSONSerialization` so the keys come out in the documented order, the
/// floats in their shortest exact form, and every line costs one string build.
nonisolated indirect enum IMUJSON: Sendable {
    case string(String)
    case int(Int)
    case uint(UInt64)
    case double(Double)
    case float(Float)
    case bool(Bool)
    case null
    case array([IMUJSON])
    case object([(String, IMUJSON)])

    static func optionalString(_ value: String?) -> IMUJSON { value.map(IMUJSON.string) ?? .null }
    static func optionalInt(_ value: Int?) -> IMUJSON { value.map { .int($0) } ?? .null }

    func write(into out: inout String) {
        switch self {
        case .string(let value): Self.writeString(value, into: &out)
        case .int(let value): out += String(value)
        case .uint(let value): out += String(value)
        case .double(let value): out += value.isFinite ? "\(value)" : "null"
        case .float(let value): out += value.isFinite ? "\(value)" : "null"
        case .bool(let value): out += value ? "true" : "false"
        case .null: out += "null"
        case .array(let values):
            out += "["
            for (index, value) in values.enumerated() {
                if index > 0 { out += "," }
                value.write(into: &out)
            }
            out += "]"
        case .object(let fields):
            Self.writeObject(fields, into: &out)
        }
    }

    static func writeObject(_ fields: [(String, IMUJSON)], into out: inout String) {
        out += "{"
        for (index, field) in fields.enumerated() {
            if index > 0 { out += "," }
            writeString(field.0, into: &out)
            out += ":"
            field.1.write(into: &out)
        }
        out += "}"
    }

    static func writeString(_ value: String, into out: inout String) {
        out += "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
    }

    /// One complete log line, newline-terminated.
    static func line(_ fields: [(String, IMUJSON)]) -> String {
        var out = ""
        out.reserveCapacity(160)
        writeObject(fields, into: &out)
        out += "\n"
        return out
    }
}

/// The four line types. Pure functions of their inputs, so the format is
/// tested without a file, a queue or the SDK.
nonisolated enum IMULogFormat {
    /// Bumped on any change a reader must know about.
    static let schema = 1

    /// `{"t":"m","ts_ns":…,"rx_mono_ns":…,"a":[x,y,z]|null,"g":[x,y,z]|null,"q":[w,x,y,z]|null,"src":"glasses"}`
    static func motionLine(_ reading: IMUMotionReading, rxMonoNs: UInt64) -> String {
        IMUJSON.line([
            ("t", .string("m")),
            ("ts_ns", .int(Int(reading.timestampNs))),
            ("rx_mono_ns", .uint(rxMonoNs)),
            ("a", vector(reading.accelerometer.map { [$0.x, $0.y, $0.z] })),
            ("g", vector(reading.gyroscope.map { [$0.x, $0.y, $0.z] })),
            // Stored and written as (w, x, y, z).
            ("q", vector(reading.orientation.map { [$0[0], $0[1], $0[2], $0[3]] })),
            ("src", .string(reading.source.rawValue)),
        ])
    }

    /// `{"t":"f","pts_us":…,"epoch":"<uuid>","rx_mono_ns":…,"sent":true|false,"seq":N}`
    static func frameLine(_ stamp: IMUFrameStamp, epoch: UUID?, sent: Bool, seq: Int) -> String {
        IMUJSON.line([
            ("t", .string("f")),
            ("pts_us", stamp.ptsUs.map { .int(Int($0)) } ?? .null),
            ("epoch", .optionalString(epoch?.uuidString)),
            ("rx_mono_ns", .uint(stamp.rxMonoNs)),
            ("sent", .bool(sent)),
            ("seq", .int(seq)),
        ])
    }

    /// `{"t":"ev","ev":"<name>","mono_ns":…, …fields}`
    static func eventLine(_ name: String, monoNs: UInt64, fields: [(String, IMUJSON)]) -> String {
        IMUJSON.line([("t", .string("ev")), ("ev", .string(name)), ("mono_ns", .uint(monoNs))] + fields)
    }

    private static func vector(_ values: [Float]?) -> IMUJSON {
        guard let values else { return .null }
        return .array(values.map(IMUJSON.float))
    }
}

/// Everything the header line records. Built by `GlassesConnection` once the
/// Motion attempt has an answer, so the header can say which rate was granted.
nonisolated struct IMULogHeader: Sendable {
    var sessionID: UUID
    var fileName: String
    var startMonoNs: UInt64
    var startWall: Date
    var motionRequestedHz: Int?
    var motionActualHz: Int?
    var motionAttempts: [IMUMotionAttempt]
    var glassesModel: String?
    var captureResolution: String
    var cameraFPSRequested: Int
    var towerTargetFPS: Double
    /// `nil` reads `.current` when the line is built, which is on the
    /// recorder's queue: it stats the app binary, and that is file I/O.
    var build: BuildInfo? = nil

    /// What this binary and this phone are. Read once per header.
    struct BuildInfo: Sendable {
        var appBuildSHA: String?
        var appVersion: String?
        var appBinaryModified: Date?
        var datSDKVersion: String?
        var datMotionVersion: String?
        var phoneModel: String?
        var phoneOS: String

        static var current: BuildInfo {
            let main = Bundle.main
            // `GlassesBuildSHA` is `$(GLASSES_BUILD_SHA)` in Info.plist: empty
            // unless the build passed `GLASSES_BUILD_SHA=<sha>` to xcodebuild.
            let sha = (main.object(forInfoDictionaryKey: "GlassesBuildSHA") as? String)
                .flatMap { $0.trimmingCharacters(in: .whitespaces).isEmpty ? nil : $0 }
            let short = main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
            let build = main.object(forInfoDictionaryKey: "CFBundleVersion") as? String
            let modified = main.executableURL.flatMap {
                (try? FileManager.default.attributesOfItem(atPath: $0.path))?[.modificationDate] as? Date
            }
            return BuildInfo(
                appBuildSHA: sha,
                appVersion: [short, build.map { "(\($0))" }].compactMap { $0 }.joined(separator: " "),
                appBinaryModified: modified,
                datSDKVersion: frameworkVersion("com.facebook.MWDATCore"),
                datMotionVersion: frameworkVersion("com.facebook.MWDATMotion"),
                phoneModel: hardwareModel(),
                phoneOS: ProcessInfo.processInfo.operatingSystemVersionString
            )
        }

        /// DAT ships as dynamic frameworks, so each one's own Info.plist says
        /// which release is linked, rather than this file restating the pin.
        static func frameworkVersion(_ identifier: String) -> String? {
            Bundle(identifier: identifier)?.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        }

        /// `iPhone17,1` and the like; on the Simulator, the simulated model.
        static func hardwareModel() -> String? {
            if let simulated = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
                return simulated
            }
            var info = utsname()
            uname(&info)
            let machine = withUnsafeBytes(of: &info.machine) { raw in
                String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
            }
            return machine.isEmpty ? nil : machine
        }
    }

    func line() -> String {
        let build = self.build ?? .current
        return IMUJSON.line([
            ("t", .string("header")),
            ("schema", .int(IMULogFormat.schema)),
            ("file", .string(fileName)),
            ("session_id", .string(sessionID.uuidString)),
            ("app_build_sha", .optionalString(build.appBuildSHA)),
            ("app_version", .optionalString(build.appVersion)),
            ("app_binary_mtime", .optionalString(build.appBinaryModified.map(IMURecorder.isoExtended))),
            ("dat_sdk_version", .optionalString(build.datSDKVersion)),
            ("dat_motion_version", .optionalString(build.datMotionVersion)),
            ("motion_rate_requested_hz", .optionalInt(motionRequestedHz)),
            ("motion_rate_actual_hz", .optionalInt(motionActualHz)),
            ("motion_rate_attempts", .array(motionAttempts.map {
                .object([("hz", .int($0.hz)), ("error", .optionalString($0.error))])
            })),
            ("phone_model", .optionalString(build.phoneModel)),
            ("phone_os", .string(build.phoneOS)),
            ("glasses_model", .optionalString(glassesModel)),
            ("start_mono_ns", .uint(startMonoNs)),
            ("start_wall_unix_ms", .int(Int((startWall.timeIntervalSince1970 * 1000).rounded()))),
            ("start_wall_iso", .string(IMURecorder.isoExtended(startWall))),
            ("capture_resolution", .string(captureResolution)),
            ("camera_fps_requested", .int(cameraFPSRequested)),
            ("tower_target_fps", .double(towerTargetFPS)),
            ("clocks", .object([
                ("ts_ns", .string("MotionSample.timestampNs: the glasses' monotonic clock")),
                ("rx_mono_ns", .string("phone DispatchTime.uptimeNanoseconds (mach_absolute_time; stops while the phone sleeps), read on DAT's callback thread before any main-actor hop")),
                ("pts_us", .string("CMSampleBufferGetPresentationTimeStamp of VideoFrame.sampleBuffer, in microseconds")),
                ("mono_ns", .string("phone DispatchTime.uptimeNanoseconds when the event was noted")),
            ])),
            ("seq_semantics", .string("the app's 1-based DAT frame ordinal for this capture session: GlassesConnection.frameCount, sent to the Tower as frame.seq, which the Tower stores as source_seq in frames.jsonl (the app sends no source_seq)")),
            ("sent_semantics", .string("true when the 12 fps gate selected the frame and it was decoded and handed to TowerClient.sendFrame; the Tower may still not have received it (offline, send window, paused) -- frames.jsonl is the truth for receipt")),
            ("epoch_semantics", .string("a new UUID at each camera start: stream.start(), and each return to streaming after a pause or a stop with frames already in the epoch")),
        ])
    }
}

// MARK: - Recorder

/// Counters the recorder keeps. Also the body of the periodic `counts` event
/// and of the Developer Tools readout.
nonisolated struct IMURecorderCounts: Equatable, Sendable {
    /// Motion lines written: glasses-source samples.
    var motionGlasses = 0
    /// Samples not written because their source was not the glasses, by source.
    var motionOtherBySource: [String: Int] = [:]
    /// Consecutive glasses samples more than 1.5 intervals apart, on the
    /// glasses' clock, and how many samples those gaps are estimated to hold.
    /// DAT's sample stream keeps the newest 256 and drops silently, so a gap
    /// is the only trace an overflow leaves.
    var motionGapEvents = 0
    var motionEstimatedMissing = 0
    var motionMaxGapNs: Int64 = 0
    /// A timestamp that did not advance.
    var motionNonMonotonic = 0
    /// Frame lines written, and how many of them were forwarded.
    var frames = 0
    var framesSent = 0
    var framesWithoutPTS = 0
    var ptsDiscontinuities = 0
    /// Lines refused after a cap, after close, or because the phone stayed
    /// locked long enough to fill the in-memory buffer.
    var droppedAfterCap = 0
    var droppedAfterClose = 0
    var droppedWhileLocked = 0
    /// The frames among `droppedAfterClose`: received by the phone after the
    /// session's log closed (still queued for the main actor at the stop).
    var framesAfterClose = 0

    var motionOther: Int { motionOtherBySource.values.reduce(0, +) }

    var jsonFields: [(String, IMUJSON)] {
        [
            ("m_glasses", .int(motionGlasses)),
            ("m_other", .int(motionOther)),
            ("m_other_by_src", .object(motionOtherBySource.sorted { $0.key < $1.key }.map { ($0.key, .int($0.value)) })),
            ("m_gap_events", .int(motionGapEvents)),
            ("m_est_missing", .int(motionEstimatedMissing)),
            ("m_max_gap_ms", .double(Double(motionMaxGapNs) / 1_000_000)),
            ("m_nonmonotonic", .int(motionNonMonotonic)),
            ("f_total", .int(frames)),
            ("f_sent", .int(framesSent)),
            ("f_no_pts", .int(framesWithoutPTS)),
            ("f_pts_discontinuities", .int(ptsDiscontinuities)),
            ("dropped_after_cap", .int(droppedAfterCap)),
            ("dropped_after_close", .int(droppedAfterClose)),
            ("f_after_close", .int(framesAfterClose)),
            ("dropped_while_locked", .int(droppedWhileLocked)),
        ]
    }
}

/// What Developer Tools and the recording badge show. Republished at most
/// once a second, and on every change of phase or cap.
nonisolated struct IMURecorderStatus: Equatable, Sendable {
    enum Phase: Equatable, Sendable {
        case idle
        case recording
        case closed
        case failed(String)
    }

    var phase: Phase = .idle
    var fileName = ""
    var bytes = 0
    var counts = IMURecorderCounts()
    var epochs = 0
    var motionConfiguredHz: Int?
    /// Glasses samples per second over the last tick, by the glasses' clock.
    var motionRateHz: Double?
    /// Which cap stopped the data lines, if one has: `size`, `duration` or
    /// `folder`.
    var capReason: String?
    /// The phone is locked; lines are held in memory until it unlocks.
    var heldWhileLocked = false
}

/// The recording badge's three states.
nonisolated enum IMURecordingIndicator: Equatable, Sendable {
    /// No recorder: no badge.
    case off
    /// A recorder is writing this capture's IMU log.
    case writing
    /// A recorder exists but is not writing data lines: a cap was reached, or
    /// the file could not be written. The reason is short enough for a badge.
    case stopped(String)

    init(_ status: IMURecorderStatus) {
        switch status.phase {
        case .idle, .closed: self = .off
        case .failed: self = .stopped("error")
        case .recording: self = status.capReason.map { .stopped("\($0) cap") } ?? .writing
        }
    }
}

/// Which file operations ran where. The recorder's promise is that every one of
/// them runs on its own serial queue and none on the main thread, and a test
/// reads this to hold it to that.
nonisolated struct IMUIOAudit: Equatable, Sendable {
    var operations = 0
    var onMainThread = 0
    var offQueue = 0
}

/// Writes one capture session's IMU log: `Documents/imu-logs/<start>-<id>.jsonl`.
///
/// ## The one rule: nothing here runs on the caller's thread but an enqueue
///
/// The main actor already carries the frame path (gate, JPEG encode, base64,
/// JSON, viewfinder), so every public method does exactly one thing on the
/// calling thread -- `queue.async` with a small value -- and everything else,
/// formatting included, happens on this recorder's own serial utility queue.
/// Lines are buffered and written when 64 KB accumulate or once a second,
/// whichever is first, so a crash loses at most about a second.
///
/// ## Bounded, three ways
///
/// A file stops taking data lines (`m`, `f`) at 200 MB or 30 minutes, whichever
/// comes first, and the whole `imu-logs` folder is held to 1 GB: a file that
/// opens with less than 200 MB of folder budget left is capped at what is left,
/// and one that opens with almost none is not created at all. Events may use a
/// small reserve beyond the data cap, so the `cap_reached` line and the closing
/// `counts` still land. The first refusal is printed and logged; every later
/// one is counted.
///
/// ## Private
///
/// Motion is raw sensor data (docs/06-PRIVACY-DATA.md), so the file is written
/// with `FileProtectionType.complete` and excluded from backup. `.complete`
/// makes the file unreadable while the phone is locked, including to this
/// process, so when protected data is about to go away the recorder flushes and
/// closes its handle, holds new lines in memory (bounded), and reopens and
/// flushes when the phone is unlocked. A close requested while locked waits for
/// the unlock rather than losing the held lines.
///
/// State below `queue` is confined to it; that is what the `@unchecked` asserts.
nonisolated final class IMURecorder: @unchecked Sendable {

    struct Limits: Equatable, Sendable {
        /// Per file.
        var maxBytes = 200_000_000
        /// Per file, on this phone's monotonic clock from the recorder's start.
        var maxDuration: TimeInterval = 30 * 60
        /// The whole `imu-logs` folder, this file included.
        var folderMaxBytes = 1_000_000_000
        var eventReserveBytes = 64 * 1024
        var flushThresholdBytes = 64 * 1024
        var flushInterval: TimeInterval = 1.0
        /// A `counts` event every this many flush ticks (10 s by default).
        var countsEveryTicks = 10
        /// How much may be held in memory while the phone is locked: about
        /// half an hour of 60 Hz Motion plus 24 fps frame lines.
        var maxBufferedBytesWhileLocked = 32 * 1024 * 1024

        static let standard = Limits()
    }

    /// Directory under Documents that holds every log. `pull_imu_logs.sh`
    /// copies exactly this.
    static let directoryName = "imu-logs"

    /// `UIApplication.protectedDataWillBecomeUnavailableNotification` and its
    /// pair, by value, so this nonisolated file need not reach into UIKit's
    /// main-actor class for two constants. A test holds them equal to UIKit's.
    static let protectedDataWillBecomeUnavailable = Notification.Name("UIApplicationProtectedDataWillBecomeUnavailable")
    static let protectedDataDidBecomeAvailable = Notification.Name("UIApplicationProtectedDataDidBecomeAvailable")

    let sessionID: UUID
    let startWall: Date
    let startMonoNs: UInt64
    let limits: Limits
    /// `imu-logs/<ISO-start>-<first 8 of the session id>.jsonl`, relative to
    /// the base directory.
    let relativePath: String

    private let baseDirectory: URL?
    private let onStatus: (@Sendable (IMURecorderStatus) -> Void)?
    private let queue: DispatchQueue
    private let queueKey = DispatchSpecificKey<UInt8>()

    // MARK: Queue-confined state

    private var header: String?
    private var fileURL: URL?
    private var handle: FileHandle?
    private var buffer = Data()
    private var bytesAccepted = 0
    private var bytesWritten = 0
    /// `min(limits.maxBytes, folder budget left)`, fixed when the file opens.
    private var effectiveMaxBytes: Int
    private var capIsFolder = false
    private var capReason: String?
    private var closed = false
    private var closePending: (reason: String, monoNs: UInt64)?
    private var closeCompletions: [@Sendable (IMURecorderSummary) -> Void] = []
    private var failure: String?
    private var timer: DispatchSourceTimer?
    private var ticks = 0
    private var observers: [NSObjectProtocol] = []
    private var protectedDataAvailable = true
    private var counts = IMURecorderCounts()
    private var io = IMUIOAudit()

    private var epoch: UUID?
    private var epochFrames = 0
    private var epochCount = 0
    private var lastStreamState: String?
    private var lastPTSUs: Int64?
    private var lastDeviceState: IMUDeviceStateSnapshot?
    private var motionIntervalNs: Int64?
    private var motionConfiguredHz: Int?
    private var lastMotionTsNs: Int64?
    private var tickFirstTsNs: Int64?
    private var tickLastTsNs: Int64?
    private var tickSamples = 0
    private var lastRateHz: Double?
    private var reportedFirstOtherSource = false

    /// - Parameters:
    ///   - baseDirectory: where `imu-logs/` is created; `nil` is this app's
    ///     Documents directory, resolved on the recorder's queue.
    ///   - onStatus: called on the recorder's queue, at most once a second and
    ///     on every change of phase or cap.
    init(
        sessionID: UUID = UUID(),
        startWall: Date = Date(),
        startMonoNs: UInt64 = DispatchTime.now().uptimeNanoseconds,
        baseDirectory: URL? = nil,
        limits: Limits = .standard,
        onStatus: (@Sendable (IMURecorderStatus) -> Void)? = nil
    ) {
        self.sessionID = sessionID
        self.startWall = startWall
        self.startMonoNs = startMonoNs
        self.baseDirectory = baseDirectory
        self.limits = limits
        self.onStatus = onStatus
        self.effectiveMaxBytes = limits.maxBytes
        let shortID = String(sessionID.uuidString.prefix(8))
        self.relativePath = "\(Self.directoryName)/\(Self.isoBasic(startWall))-\(shortID).jsonl"
        self.queue = DispatchQueue(label: "Glasses.IMURecorder.\(shortID)", qos: .utility)
        queue.setSpecific(key: queueKey, value: 1)
    }

    // MARK: Called from any thread; each is one enqueue

    /// Writes the header, creates the file and starts the flush timer. Lines
    /// recorded before this are held in memory and written after the header,
    /// so the header is always the first line on disk.
    func open(header: IMULogHeader) {
        queue.async { self.openOnQueue(header.line()) }
    }

    func recordFrame(_ stamp: IMUFrameStamp, seq: Int, sent: Bool) {
        queue.async { self.frameOnQueue(stamp, seq: seq, sent: sent) }
    }

    func recordMotion(_ reading: IMUMotionReading, rxMonoNs: UInt64) {
        queue.async { self.motionOnQueue(reading, rxMonoNs: rxMonoNs) }
    }

    /// A camera start: a fresh epoch, which every later `f` line carries.
    func noteCameraStart(reason: String, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async { self.rotateEpoch(reason: reason, monoNs: monoNs) }
    }

    func noteCameraStop(reason: String, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async {
            self.eventOnQueue("camera_stop", monoNs: monoNs, fields: [
                ("epoch", .optionalString(self.epoch?.uuidString)),
                ("reason", .string(reason)),
                ("epoch_frames", .int(self.epochFrames)),
            ])
        }
    }

    /// Every camera `StreamState`. A return to streaming after a pause or a
    /// stop, in an epoch that already has frames, is a camera start.
    func noteStreamState(_ state: String, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async {
            self.eventOnQueue("stream_state", monoNs: monoNs, fields: [("state", .string(state))])
            if state == "streaming", let previous = self.lastStreamState,
               previous != "streaming", previous != "starting", self.epochFrames > 0 {
                self.rotateEpoch(reason: "resume from \(previous)", monoNs: monoNs)
            }
            self.lastStreamState = state
        }
    }

    /// Every `DeviceSessionState`: pause and resume arrive here.
    func noteSessionState(_ state: String, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async { self.eventOnQueue("session_state", monoNs: monoNs, fields: [("state", .string(state))]) }
    }

    /// Motion attached at `hz`. Also arms gap detection at that rate.
    func noteMotionStart(hz: Int, requestedHz: Int, attempts: [IMUMotionAttempt], monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async {
            self.motionConfiguredHz = hz
            self.motionIntervalNs = hz > 0 ? 1_000_000_000 / Int64(hz) : nil
            self.lastMotionTsNs = nil
            self.eventOnQueue("motion_start", monoNs: monoNs, fields: [
                ("rate_hz", .int(hz)),
                ("requested_hz", .int(requestedHz)),
                ("attempts", .array(attempts.map { .object([("hz", .int($0.hz)), ("error", .optionalString($0.error))]) })),
            ])
        }
    }

    func noteEvent(_ name: String, fields: [(String, IMUJSON)] = [], monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async { self.eventOnQueue(name, monoNs: monoNs, fields: fields) }
    }

    /// Written only when it differs from the last one written.
    func noteDeviceState(_ state: IMUDeviceStateSnapshot, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async {
            guard state != self.lastDeviceState else { return }
            self.lastDeviceState = state
            self.eventOnQueue("device_state", monoNs: monoNs, fields: [
                ("thermal", .string(state.thermal)),
                ("battery_pct", .optionalInt(state.batteryPercent)),
                ("charging", .string(state.charging)),
                ("don", .string(state.don)),
                ("hinge", .string(state.hinge)),
                ("link", .string(state.link)),
                ("compatibility", .string(state.compatibility)),
            ])
        }
    }

    /// The phone is locking: flush now, while the file is still writable, and
    /// hold everything after this in memory. Called by the notification this
    /// recorder observes, and by tests.
    func protectedDataWillBecomeUnavailable() {
        queue.async { self.lockOnQueue() }
    }

    /// The phone unlocked: reopen, write what was held, and finish a close
    /// that was waiting for this.
    func protectedDataDidBecomeAvailable() {
        queue.async { self.unlockOnQueue() }
    }

    /// Final counts, flush, fsync, close. Idempotent. `completion` runs on the
    /// recorder's queue once the file is closed -- which, if the phone is
    /// locked, is after it next unlocks.
    func close(reason: String, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds, completion: (@Sendable (IMURecorderSummary) -> Void)? = nil) {
        queue.async {
            if let completion { self.closeCompletions.append(completion) }
            self.requestCloseOnQueue(reason: reason, monoNs: monoNs)
        }
    }

    // MARK: Test and diagnostics reads (block the caller until the queue drains)

    func waitUntilIdle() { queue.sync {} }

    func summary() -> IMURecorderSummary { queue.sync { summaryOnQueue() } }

    // MARK: Queue side

    private func openOnQueue(_ headerLine: String) {
        guard header == nil, !closed, failure == nil else { return }
        header = headerLine
        let headerData = Data(headerLine.utf8)
        buffer = headerData + buffer
        bytesAccepted += headerData.count

        var existingFolderBytes = 0
        performIO {
            let base = try baseDirectory ?? FileManager.default.url(
                for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true
            )
            let url = base.appendingPathComponent(relativePath)
            var folder = url.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            var excluded = URLResourceValues()
            excluded.isExcludedFromBackup = true
            try folder.setResourceValues(excluded)
            existingFolderBytes = IMULogStore.totalBytes(in: folder)

            // The folder's budget, fixed now. A file that would open with less
            // room than its own event reserve is not created at all.
            let budget = limits.folderMaxBytes - existingFolderBytes
            guard budget >= limits.eventReserveBytes * 2 else {
                throw IMURecorderError.folderFull(existingBytes: existingFolderBytes, limit: limits.folderMaxBytes)
            }
            if budget < limits.maxBytes {
                effectiveMaxBytes = budget
                capIsFolder = true
            }

            guard FileManager.default.createFile(
                atPath: url.path, contents: nil,
                attributes: [.protectionKey: FileProtectionType.complete]
            ) else {
                throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path])
            }
            var fileURL = url
            try fileURL.setResourceValues(excluded)
            self.fileURL = fileURL
            handle = try FileHandle(forWritingTo: fileURL)
        }
        guard failure == nil else { return }
        print("[Glasses][IMURec] recording to \(fileURL?.path ?? relativePath) (folder already holds \(existingFolderBytes) bytes; this file may take \(effectiveMaxBytes))")
        flushOnQueue()

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(
            deadline: .now() + limits.flushInterval,
            repeating: limits.flushInterval,
            leeway: .milliseconds(100)
        )
        timer.setEventHandler { [weak self] in self?.tick() }
        timer.resume()
        self.timer = timer

        // Strong captures on purpose: a close that arrives while the phone is
        // locked has to outlive its owner until the unlock that finishes it.
        // `finishCloseOnQueue` removes these, which breaks the cycle.
        let center = NotificationCenter.default
        observers = [
            center.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil) { [self] _ in
                noteEvent("phone_thermal", fields: [("state", .string(Self.name(of: ProcessInfo.processInfo.thermalState)))])
            },
            center.addObserver(forName: Self.protectedDataWillBecomeUnavailable, object: nil, queue: nil) { [self] _ in
                protectedDataWillBecomeUnavailable()
            },
            center.addObserver(forName: Self.protectedDataDidBecomeAvailable, object: nil, queue: nil) { [self] _ in
                protectedDataDidBecomeAvailable()
            },
        ]
        eventOnQueue("phone_thermal", monoNs: DispatchTime.now().uptimeNanoseconds, fields: [
            ("state", .string(Self.name(of: ProcessInfo.processInfo.thermalState))),
        ])
        publishStatus()
    }

    private func frameOnQueue(_ stamp: IMUFrameStamp, seq: Int, sent: Bool) {
        if let pts = stamp.ptsUs {
            // A capture clock that runs backwards, or jumps by more than any
            // frame interval, inside one epoch. Recorded, not repaired.
            if let last = lastPTSUs, pts <= last || pts - last > 1_000_000 {
                counts.ptsDiscontinuities += 1
                eventOnQueue("pts_discontinuity", monoNs: stamp.rxMonoNs, fields: [
                    ("epoch", .optionalString(epoch?.uuidString)),
                    ("seq", .int(seq)),
                    ("prev_pts_us", .int(Int(last))),
                    ("pts_us", .int(Int(pts))),
                ])
            }
            lastPTSUs = pts
        }
        guard append(IMULogFormat.frameLine(stamp, epoch: epoch, sent: sent, seq: seq), isEvent: false) else {
            if closed { counts.framesAfterClose += 1 }
            return
        }
        epochFrames += 1
        counts.frames += 1
        if sent { counts.framesSent += 1 }
        if stamp.ptsUs == nil { counts.framesWithoutPTS += 1 }
    }

    private func motionOnQueue(_ reading: IMUMotionReading, rxMonoNs: UInt64) {
        guard reading.source == .glasses else {
            counts.motionOtherBySource[reading.source.rawValue, default: 0] += 1
            if !reportedFirstOtherSource {
                reportedFirstOtherSource = true
                eventOnQueue("first_non_glasses_sample", monoNs: rxMonoNs, fields: [
                    ("src", .string(reading.source.rawValue)),
                    ("ts_ns", .int(Int(reading.timestampNs))),
                ])
            }
            return
        }
        guard append(IMULogFormat.motionLine(reading, rxMonoNs: rxMonoNs), isEvent: false) else { return }
        counts.motionGlasses += 1

        if let last = lastMotionTsNs {
            let delta = reading.timestampNs - last
            if delta <= 0 {
                counts.motionNonMonotonic += 1
            } else {
                counts.motionMaxGapNs = max(counts.motionMaxGapNs, delta)
                if let interval = motionIntervalNs, delta * 2 > interval * 3 {
                    counts.motionGapEvents += 1
                    let missing = Int((Double(delta) / Double(interval)).rounded()) - 1
                    counts.motionEstimatedMissing += max(0, missing)
                }
            }
        }
        lastMotionTsNs = reading.timestampNs
        if tickFirstTsNs == nil { tickFirstTsNs = reading.timestampNs }
        tickLastTsNs = reading.timestampNs
        tickSamples += 1
    }

    private func rotateEpoch(reason: String, monoNs: UInt64) {
        let previous = epoch
        let next = UUID()
        epoch = next
        epochFrames = 0
        epochCount += 1
        lastPTSUs = nil
        eventOnQueue("camera_start", monoNs: monoNs, fields: [
            ("epoch", .string(next.uuidString)),
            ("reason", .string(reason)),
            ("prev_epoch", .optionalString(previous?.uuidString)),
        ])
    }

    private func eventOnQueue(_ name: String, monoNs: UInt64, fields: [(String, IMUJSON)]) {
        append(IMULogFormat.eventLine(name, monoNs: monoNs, fields: fields), isEvent: true)
    }

    /// Accepts one line into the buffer, or refuses it: for a cap, because the
    /// log is closed, or because the phone has been locked so long that the
    /// in-memory hold is full.
    @discardableResult
    private func append(_ line: String, isEvent: Bool) -> Bool {
        guard !closed else {
            counts.droppedAfterClose += 1
            return false
        }
        if !isEvent, capReason == nil,
           Double(DispatchTime.now().uptimeNanoseconds &- startMonoNs) / 1_000_000_000 >= limits.maxDuration {
            reachCap("duration")
        }
        if !isEvent, capReason != nil {
            counts.droppedAfterCap += 1
            return false
        }
        let data = Data(line.utf8)
        let cap = isEvent ? effectiveMaxBytes : effectiveMaxBytes - limits.eventReserveBytes
        guard bytesAccepted + data.count <= cap else {
            counts.droppedAfterCap += 1
            if !isEvent { reachCap(capIsFolder ? "folder" : "size") }
            return false
        }
        if !protectedDataAvailable, buffer.count + data.count > limits.maxBufferedBytesWhileLocked {
            counts.droppedWhileLocked += 1
            return false
        }
        buffer.append(data)
        bytesAccepted += data.count
        if header != nil, protectedDataAvailable, buffer.count >= limits.flushThresholdBytes {
            flushOnQueue()
        }
        return true
    }

    /// Stops the data lines for good, says so in the console and the log, and
    /// tells the badge.
    private func reachCap(_ reason: String) {
        guard capReason == nil else { return }
        capReason = reason
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds &- startMonoNs) / 1_000_000_000
        print("[Glasses][IMURec] \(reason) cap reached after \(String(format: "%.0f", elapsed)) s and \(bytesAccepted) bytes; no more m/f lines are written to \(relativePath)")
        eventOnQueue("cap_reached", monoNs: DispatchTime.now().uptimeNanoseconds, fields: [
            ("cap", .string(reason)),
            ("bytes", .int(bytesAccepted)),
            ("elapsed_s", .double(elapsed)),
            ("max_bytes", .int(effectiveMaxBytes)),
            ("max_duration_s", .double(limits.maxDuration)),
            ("folder_max_bytes", .int(limits.folderMaxBytes)),
        ])
        publishStatus()
    }

    private func flushOnQueue() {
        guard let handle, protectedDataAvailable, !buffer.isEmpty, failure == nil else { return }
        let pending = buffer
        performIO { try handle.write(contentsOf: pending) }
        guard failure == nil else { return }
        bytesWritten += pending.count
        buffer.removeAll(keepingCapacity: true)
    }

    private func lockOnQueue() {
        guard protectedDataAvailable, !closed else { return }
        eventOnQueue("protected_data_unavailable", monoNs: DispatchTime.now().uptimeNanoseconds, fields: [])
        flushOnQueue()
        if let handle {
            performIO {
                try handle.synchronize()
                try handle.close()
            }
        }
        handle = nil
        protectedDataAvailable = false
        print("[Glasses][IMURec] phone locking: flushed and closed; holding lines in memory until unlock")
        publishStatus()
    }

    private func unlockOnQueue() {
        guard !protectedDataAvailable else { return }
        protectedDataAvailable = true
        eventOnQueue("protected_data_available", monoNs: DispatchTime.now().uptimeNanoseconds, fields: [
            ("held_bytes", .int(buffer.count)),
        ])
        if let fileURL, failure == nil {
            performIO {
                let reopened = try FileHandle(forWritingTo: fileURL)
                try reopened.seekToEnd()
                handle = reopened
            }
        }
        flushOnQueue()
        print("[Glasses][IMURec] phone unlocked: reopened and wrote what was held")
        publishStatus()
        if let pending = closePending {
            closePending = nil
            finishCloseOnQueue(reason: pending.reason, monoNs: pending.monoNs)
        }
    }

    private func tick() {
        flushOnQueue()
        ticks += 1
        if let first = tickFirstTsNs, let last = tickLastTsNs, tickSamples >= 2, last > first {
            lastRateHz = Double(tickSamples - 1) / (Double(last - first) / 1_000_000_000)
        } else {
            lastRateHz = tickSamples == 0 ? 0 : nil
        }
        tickFirstTsNs = nil
        tickLastTsNs = nil
        tickSamples = 0
        if limits.countsEveryTicks > 0, ticks % limits.countsEveryTicks == 0 {
            eventOnQueue("counts", monoNs: DispatchTime.now().uptimeNanoseconds, fields: counts.jsonFields)
        }
        publishStatus()
    }

    private func requestCloseOnQueue(reason: String, monoNs: UInt64) {
        guard !closed, closePending == nil else { return }
        guard protectedDataAvailable || fileURL == nil || failure != nil else {
            // Locked: the held lines can only be written after the unlock.
            closePending = (reason, monoNs)
            print("[Glasses][IMURec] close requested while locked; finishing \(relativePath) at the next unlock")
            return
        }
        finishCloseOnQueue(reason: reason, monoNs: monoNs)
    }

    private func finishCloseOnQueue(reason: String, monoNs: UInt64) {
        // Last, so they are the file's last lines even after a locked wait.
        eventOnQueue("counts", monoNs: monoNs, fields: counts.jsonFields)
        eventOnQueue("close", monoNs: monoNs, fields: [
            ("reason", .string(reason)),
            ("bytes", .int(bytesAccepted)),
            ("epochs", .int(epochCount)),
            ("cap", .optionalString(capReason)),
        ])
        timer?.cancel()
        timer = nil
        observers.forEach { NotificationCenter.default.removeObserver($0) }
        observers = []
        flushOnQueue()
        if let handle {
            performIO {
                try handle.synchronize()
                try handle.close()
            }
        }
        handle = nil
        closed = true
        if header != nil {
            print("[Glasses][IMURec] closed \(relativePath) (\(reason)): \(bytesWritten) bytes, m=\(counts.motionGlasses) f=\(counts.frames) sent=\(counts.framesSent) epochs=\(epochCount) cap=\(capReason ?? "none")")
        }
        publishStatus()
        let summary = summaryOnQueue()
        let completions = closeCompletions
        closeCompletions = []
        completions.forEach { $0(summary) }
    }

    /// Every file operation goes through here, so the audit is complete.
    private func performIO(_ body: () throws -> Void) {
        io.operations += 1
        if Thread.isMainThread { io.onMainThread += 1 }
        if DispatchQueue.getSpecific(key: queueKey) == nil { io.offQueue += 1 }
        do {
            try body()
        } catch {
            guard failure == nil else { return }
            failure = error.localizedDescription
            print("[Glasses][IMURec] recording stopped: \(error.localizedDescription)")
            try? handle?.close()
            handle = nil
            timer?.cancel()
            timer = nil
            publishStatus()
        }
    }

    private func publishStatus() {
        guard let onStatus else { return }
        onStatus(statusOnQueue())
    }

    private func statusOnQueue() -> IMURecorderStatus {
        let phase: IMURecorderStatus.Phase
        if let failure {
            phase = .failed(failure)
        } else if closed {
            phase = .closed
        } else if header != nil {
            phase = .recording
        } else {
            phase = .idle
        }
        return IMURecorderStatus(
            phase: phase,
            fileName: relativePath,
            bytes: bytesAccepted,
            counts: counts,
            epochs: epochCount,
            motionConfiguredHz: motionConfiguredHz,
            motionRateHz: lastRateHz,
            capReason: capReason,
            heldWhileLocked: !protectedDataAvailable
        )
    }

    private func summaryOnQueue() -> IMURecorderSummary {
        IMURecorderSummary(
            fileURL: fileURL,
            bytesWritten: bytesWritten,
            bytesHeld: buffer.count,
            counts: counts,
            epochs: epochCount,
            capReason: capReason,
            effectiveMaxBytes: effectiveMaxBytes,
            closed: closed,
            failure: failure,
            io: io
        )
    }

    // MARK: Formatting helpers

    /// `20260925T143012Z`: ISO 8601 basic format, safe in a file name.
    static func isoBasic(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        return formatter.string(from: date)
    }

    static func isoExtended(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }

    static func name(of state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

nonisolated enum IMURecorderError: LocalizedError {
    case folderFull(existingBytes: Int, limit: Int)

    var errorDescription: String? {
        switch self {
        case .folderFull(let existing, let limit):
            return "imu-logs already holds \(existing) of its \(limit) bytes; delete the logs in Developer Tools to record again"
        }
    }
}

/// What a closed (or draining) recorder reports, for tests and the console.
nonisolated struct IMURecorderSummary: Equatable, Sendable {
    let fileURL: URL?
    let bytesWritten: Int
    /// Accepted but not yet on disk: the buffer, or what is held while locked.
    let bytesHeld: Int
    let counts: IMURecorderCounts
    let epochs: Int
    let capReason: String?
    let effectiveMaxBytes: Int
    let closed: Bool
    let failure: String?
    let io: IMUIOAudit
}

// MARK: - The folder

/// How many logs there are and how much they hold.
nonisolated struct IMULogInventory: Equatable, Sendable {
    var files = 0
    var bytes = 0
}

/// Reads and purges `imu-logs/`. Every function does file I/O, so callers run
/// them off the main thread.
nonisolated enum IMULogStore {
    static func folder(base: URL?) throws -> URL {
        let base = try base ?? FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false
        )
        return base.appendingPathComponent(IMURecorder.directoryName, isDirectory: true)
    }

    static func logFiles(in folder: URL) -> [URL] {
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]
        )) ?? []
        return contents.filter { $0.pathExtension == "jsonl" }
    }

    static func totalBytes(in folder: URL) -> Int {
        logFiles(in: folder).reduce(0) { total, url in
            total + ((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
    }

    static func inventory(base: URL?) -> IMULogInventory {
        guard let folder = try? folder(base: base) else { return IMULogInventory() }
        let files = logFiles(in: folder)
        return IMULogInventory(files: files.count, bytes: totalBytes(in: folder))
    }

    /// Deletes every log in the folder -- on this phone only -- and returns what
    /// was deleted.
    static func purge(base: URL?) -> IMULogInventory {
        guard let folder = try? folder(base: base) else { return IMULogInventory() }
        var deleted = IMULogInventory()
        for url in logFiles(in: folder) {
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            if (try? FileManager.default.removeItem(at: url)) != nil {
                deleted.files += 1
                deleted.bytes += size
            }
        }
        return deleted
    }
}

/// The recorder's readout for Developer Tools and the recording badge, on its
/// own object for the reason `MotionProbe` gives: nearly every screen observes
/// `GlassesConnection`, and only these two should redraw when the recorder
/// reports.
@MainActor
final class IMURecorderReadout: ObservableObject {
    @Published private(set) var status = IMURecorderStatus()
    @Published private(set) var indicator: IMURecordingIndicator = .off

    func update(_ status: IMURecorderStatus) {
        if status != self.status { self.status = status }
        setIndicator(IMURecordingIndicator(status))
    }

    func setIndicator(_ indicator: IMURecordingIndicator) {
        if indicator != self.indicator { self.indicator = indicator }
    }
}

#endif
