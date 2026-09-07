//
//  SceneUnderstandingWorkspaceView.swift
//  Glasses
//

import Foundation
import SwiftUI

/// The Scene Understanding workspace: what the Tower can currently see, stated
/// as carefully as the measurements require.
///
/// ## The three things this screen exists to get right
///
/// 1. **Every count is a floor.** `count_is_lower_bound` is `true` on every
///    payload, and it appeared **zero times** in this app until this screen was
///    rewritten. An undercount published without disclosure looks exactly like
///    a quiet room, so the disclosure sits with the counts rather than under a
///    disclosure triangle, and the measured recall figures are shown beside it.
/// 2. **Five silences, five sentences.** "Stopped", "still loading", "failed",
///    "running but no frame yet" and "looked and saw nothing" are five
///    different facts. Only the last is about a room. Rendering them alike
///    would tell a person there is nobody there when in fact nobody has pressed
///    Start.
/// 3. **A paused scene is not a current one.** `.lastKnown` gets its own
///    heading, its own age and its own colour. `.observing` is the only state
///    that may be drawn as now.
///
/// ## What it will never show
///
/// A list of people. There is no per-person row on this wire, no track handle,
/// no box, and no field that could hold one — so there is nothing to draw a row
/// from, and this screen is built so that adding one would mean changing the
/// model rather than adding a `ForEach`.
///
/// ## No camera controls, and no session controls either
///
/// As with Experimental CV Lab: the app's session controls stay on Home and
/// World Builder, so the set of places that can start the camera does not grow.
/// This screen additionally sends **nothing** to the Tower — see
/// `SceneUnderstandingContract.phoneSendsNothingNote`.
struct SceneUnderstandingWorkspaceView: View {
    /// A value, not the connection — see `TowerReachabilityReader`.
    let isTowerReachable: Bool

    @StateObject private var scene: SceneUnderstandingViewModel

    /// The client is injected and owned by `ProjectManager`; see
    /// `CartridgeClients`.
    init(isTowerReachable: Bool, client: any SceneUnderstandingClient) {
        self.isTowerReachable = isTowerReachable
        _scene = StateObject(wrappedValue: SceneUnderstandingViewModel(client: client))
    }

    private var availability: CartridgeAvailability {
        scene.availability(isTowerReachable: isTowerReachable)
    }

    var body: some View {
        VStack(spacing: 16) {
            header

            if let forcedPhase = availability.forcedPhase {
                CartridgeStatePanel(
                    title: "The scene",
                    phase: forcedPhase,
                    explanation: scene.unavailableExplanation(isTowerReachable: isTowerReachable),
                    futureDescription: Self.futureDescription
                )
            } else {
                scenePanel
            }
        }
        // Appearing is what opens the live subscription, and disappearing is
        // what closes it. On the Tower that subscription is the difference
        // between a people detector running and not running, so the screen
        // — not the connection — decides. See
        // `TowerSceneUnderstandingClient.workspaceVisible`.
        .onAppear { scene.workspaceVisibilityChanged(isVisible: true) }
        .onDisappear { scene.workspaceVisibilityChanged(isVisible: false) }
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Scene Understanding")
                .font(.title2.weight(.semibold))
            // "Observed", never "present"; "how many", never "who". The count
            // is the whole product here, and the sentence has to set up that
            // it is a floor before the number appears below it.
            Text("An anonymous count of what the camera can see in front of you, and roughly where. Nobody is identified, nothing is kept, and every count is a floor rather than a total.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    private static let futureDescription = """
        When this Tower offers a scene contract, this panel shows how many of \
        thirteen kinds of thing the camera can confirm in front of you and \
        which side of your view they are on. It never shows who, and there is \
        no field on the wire that could carry that.
        """

    // MARK: The panel

    @ViewBuilder
    private var scenePanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("The scene")

            VStack(alignment: .leading, spacing: 14) {
                switch scene.state {
                case .unsupported(let reason):
                    Text(reason)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                case .idle(let reading):
                    // Four of the five silences land here. Which one is decided
                    // by the reading, never by this view.
                    SceneSilenceView(
                        silence: scene.state.silence ?? .stopped,
                        towerWording: reading?.unavailableReasonText
                    )
                    if let reading { SceneFlowFootnote(reading: reading) }

                case .awaitingFirstScene(let reading):
                    SceneSilenceView(
                        silence: scene.state.silence ?? .runningButNoFrameYet,
                        towerWording: reading.unavailableReasonText
                    )
                    if reading.lifecycle.loadOverdue {
                        // An overdue load is not a failure and must not be
                        // drawn as one.
                        Text(SceneLifecycle.loadOverdueNote)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    SceneFlowFootnote(reading: reading)

                case .observing(let reading):
                    SceneReadingView(reading: reading, isCurrent: true)

                case .lastKnown(let reading):
                    SceneReadingView(reading: reading, isCurrent: false)

                case .failed(let failure):
                    Label("Scene reading failed", systemImage: "exclamationmark.triangle.fill")
                        .font(.headline)
                    Text(failure.message)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        }
    }
}

// MARK: - Silence

/// One of the five ways there is nothing to show.
///
/// The Tower's own sentence is preferred over this app's wherever it exists —
/// it names the variable, the state or the exception, and is more specific than
/// anything written here can be. The headline is this app's, because a headline
/// has to be short and the Tower writes paragraphs.
struct SceneSilenceView: View {
    let silence: SceneSilence
    /// `scene_unavailable_reason`, verbatim. `nil` for the fifth case, which is
    /// not an unavailability and which the Tower writes no sentence for.
    var towerWording: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(silence.headline, systemImage: symbol)
                .font(.headline)
            Text(towerWording ?? silence.explanation)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    /// Deliberately five glyphs for five states. A shared icon would undo the
    /// distinction the five sentences exist to draw.
    private var symbol: String {
        switch silence {
        case .stopped: return "circle.dashed"
        case .stillLoading: return "clock"
        case .towerFailed: return "exclamationmark.triangle.fill"
        case .runningButNoFrameYet: return "hourglass"
        case .lookedAndSawNothing: return "eye"
        }
    }
}

// MARK: - A reading

/// One scene reading, current or last-known.
struct SceneReadingView: View {
    let reading: SceneReading
    /// `false` when this is the scene as it was at the moment of a pause.
    let isCurrent: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if !isCurrent { pausedBanner }

            if let observation = reading.observation {
                if observation.sawNothing {
                    SceneSilenceView(silence: .lookedAndSawNothing, towerWording: nil)
                } else {
                    counts(observation)
                }

                lowerBoundDisclosure

                people(observation.people)

                positions(observation)
            }

            SceneFlowFootnote(reading: reading)

            anonymityFootnote
        }
    }

    // MARK: Paused

    /// A stale scene announces its staleness **before** its contents.
    ///
    /// Not after, and not in a caption below the numbers: a reader takes the
    /// first thing on the panel as the subject of everything under it. When the
    /// Tower reported no observation time, that is said outright rather than
    /// filled in from the phone's clock.
    private var pausedBanner: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(staleHeadline, systemImage: "pause.circle")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.orange)
            Text("""
                Paused. These counts are the last reading before the pause, not \
                what is in front of you now.
                """)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    private var staleHeadline: String {
        guard let age = reading.stalenessSeconds else {
            return "Last reading. The Tower did not say when."
        }
        let whole = Int(age.rounded())
        if whole < 60 { return "Last reading, \(whole)s ago" }
        return "Last reading, \(whole / 60)m \(whole % 60)s ago"
    }

    // MARK: Counts

    private func counts(_ observation: SceneObservation) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            // People first and separately, because the people block is an
            // aggregate with three qualifications on it and the object counts
            // are not.
            ForEach(observation.present) { entry in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text("\(entry.count)")
                        .font(.title3.weight(.semibold))
                        .monospacedDigit()
                    Text(Self.everydayName(for: entry.label, count: entry.count))
                        .font(.subheadline)
                    Spacer(minLength: 8)
                    Text("at least")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel(
                    "At least \(entry.count) \(Self.everydayName(for: entry.label, count: entry.count)) observed"
                )
            }
        }
    }

    /// COCO class names, in words a reader will not misread.
    ///
    /// `mouse` is the pointing device and `tv` is any large display. Both carry
    /// COCO's meanings rather than the everyday ones, and a reader who has not
    /// been told will supply the everyday one — "a mouse" in a room is not a
    /// trackpad accessory to most people.
    static func everydayName(for label: String, count: Int) -> String {
        let singular: String
        switch label {
        case "mouse": singular = "computer mouse"
        case "tv": singular = "screen or TV"
        case "cell phone": singular = "phone"
        case "dining table": singular = "table"
        // "persons" is a word, and not the one anybody says.
        case "person": return count == 1 ? "person" : "people"
        default: singular = label
        }
        guard count != 1 else { return singular }
        if singular.hasSuffix("h") || singular.hasSuffix("s") { return singular + "es" }
        return singular + "s"
    }

    // MARK: The disclosure

    /// `count_is_lower_bound`, rendered where a person sees it.
    ///
    /// This is the obligation §12.3 of the contract names, and the reason it is
    /// not a footnote: an undercount published without disclosure looks exactly
    /// like a quiet room, and a footnote below three limitation paragraphs is
    /// the same as no disclosure at all.
    /// The Tower's single-person note, shown only when it applies.
    ///
    /// The privacy review accepted one limitation: with exactly one person in
    /// view, the side, size and facing aggregates describe that person for as
    /// long as they are in view. The Tower publishes the sentence that says
    /// so in `single_person_note`, and until now nothing on this side read
    /// it, so the disclosure the acceptance rested on reached nobody.
    ///
    /// Shown only at a count of one, because that is the case it is about;
    /// beside a count of four it would be noise, and noise is how a
    /// disclosure stops being read. Verbatim, because it is the Tower's claim
    /// about its own payload.
    @ViewBuilder
    private var singlePersonDisclosure: some View {
        if reading.describesOnePerson, let note = reading.singlePersonNote {
            VStack(alignment: .leading, spacing: 6) {
                Label("One person in view", systemImage: "person.fill.viewfinder")
                    .font(.subheadline.weight(.medium))
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.vertical, 4)
        }
    }

    @ViewBuilder
    private var lowerBoundDisclosure: some View {
        if reading.countIsLowerBound {
            VStack(alignment: .leading, spacing: 6) {
                singlePersonDisclosure
                Label("A floor, not a total", systemImage: "arrow.down.to.line")
                    .font(.subheadline.weight(.medium))
                Text(SceneReading.countCaveat)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                // The Tower's own measured limits, in its own words. Every one
                // is a sentence a person can be shown, and none of them is
                // summarised here — a summary of a measurement is a new claim.
                ForEach(reading.countLimitations) { limitation in
                    Text(limitation.detail)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if let measured = reading.countMeasurement.measuredAt {
                    Text(measurementFootnote(measured))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(12)
            .background(Color(.tertiarySystemGroupedBackground), in: .rect(cornerRadius: 12))
            .accessibilityElement(children: .combine)
        }
    }

    /// `is_current: false` said in words.
    ///
    /// The figures describe the frames they were measured on. A rate asserted
    /// in the present tense would read as current state, which is why the Tower
    /// publishes the flag and why this sentence exists.
    private func measurementFootnote(_ measuredAt: String) -> String {
        var text = "Measured \(measuredAt)"
        if let frames = reading.countMeasurement.corpusFrames {
            text += " on \(frames) frames"
        }
        if !reading.countMeasurement.isCurrent {
            text += ". Not re-derived since; these describe the frames they were measured on."
        } else {
            text += "."
        }
        return text
    }

    // MARK: People

    /// The people block: a count and an aggregate, never a list.
    private func people(_ people: ScenePeople) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            SectionLabel("People")

            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("\(people.count)")
                    .font(.title3.weight(.semibold))
                    .monospacedDigit()
                Text(people.count == 1 ? "person observed" : "people observed")
                    .font(.subheadline)
            }
            .accessibilityElement(children: .combine)

            if people.mayIncludeWearer {
                // The single most important qualification on this number. Every
                // `person` box in this platform's only real corpus is the
                // wearer's own torso, so a bare count reads as a count of other
                // people and is usually a count of one's own chest.
                Text("This can include your own body in frame, and often does.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !people.validated {
                Text("Never checked against a known answer — nobody has yet worn these glasses in a room with a bystander and compared.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if people.partialBottomEdge > 0 {
                // Kept out of the count on purpose, and said out loud: from a
                // camera worn at head height a figure cut off by the bottom
                // edge with no head in view is most often the wearer.
                Text(people.partialBottomEdge == 1
                     ? "1 partial figure at the bottom edge, not counted — from a camera at head height that is usually your own body."
                     : "\(people.partialBottomEdge) partial figures at the bottom edge, not counted — from a camera at head height that is usually your own body.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let sizes = Self.sizesText(people.byApparentSize) {
                // Sizes in the picture, never distances. The Tower's own note
                // says why, and it is shown rather than paraphrased.
                Text(sizes)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let note = people.apparentSizeNote {
                    Text(note)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            facing(people)
        }
    }

    /// Orientation, with `nil` and `0` drawn differently.
    ///
    /// This is the field on this payload where the difference is most likely to
    /// be lost. `0` means the Tower measured and found nobody facing the
    /// wearer; `nil` means it never measured. A screen that renders both as
    /// "0 facing you" has turned an observation gap into an observation of
    /// absence, which is Core Principle 3 in the forbidden direction.
    @ViewBuilder
    private func facing(_ people: ScenePeople) -> some View {
        if let facingWearer = people.facingWearer {
            VStack(alignment: .leading, spacing: 2) {
                // "Appears to be": the Tower establishes that the front of a
                // head is visible, and nothing more.
                Text(facingWearer == 1
                     ? "1 appears to be facing your direction"
                     : "\(facingWearer) appear to be facing your direction")
                    .font(.caption)
                if people.orientationStatus == "experimental" {
                    Text("Experimental: checked only on still photographs, never on a person seen through these glasses.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let unknown = people.facingUnknown, unknown > 0 {
                    Text("\(unknown) with orientation unknown")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                if let remainder = people.undifferentiatedRemainder, remainder > 0 {
                    // Not "facing away". The two states that would name it are
                    // withheld and have no bucket, so this is a remainder and
                    // is called one.
                    Text("\(remainder) neither — the Tower publishes no bucket for which way those are facing.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Text(SceneFacing.gazeCaveat)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        } else {
            VStack(alignment: .leading, spacing: 2) {
                // "Not measured", never "0". The wording has to make clear that
                // no number was withheld — there was no number.
                Text("Which way anyone is facing was not measured.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let reason = people.facingUnavailableReason {
                    Text(reason)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    // MARK: Where

    /// Side counts per label, and the exclusion that goes with them.
    @ViewBuilder
    private func positions(_ observation: SceneObservation) -> some View {
        let occupied = observation.positions.filter { !$0.sides.isEmpty }
        if !occupied.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                SectionLabel("Roughly where")

                ForEach(occupied) { entry in
                    HStack(spacing: 8) {
                        Text(SceneReadingView.everydayName(for: entry.label, count: entry.sides.total))
                            .font(.caption)
                        Spacer(minLength: 8)
                        Text(Self.sidesText(entry.sides))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .accessibilityElement(children: .combine)
                }

                if let convention = reading.sideConvention {
                    // The convention is declared on the payload precisely
                    // because a left and a right with no stated convention is a
                    // silent presumption, and a Tower signing the other way
                    // would put everything on the wrong side of the wearer.
                    Text(convention)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let excludesReason = reading.refusals.whereExcludesReason,
                   !reading.refusals.whereExcludes.isEmpty {
                    Text(excludesReason)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    /// Apparent sizes in words, or nil when there is nobody to size.
    ///
    /// "In view" rather than "away": these are fractions of the frame, and a
    /// seated person or a child is "small" at any distance.
    static func sizesText(_ sizes: [String: Int]) -> String? {
        var parts: [String] = []
        for key in ["large", "medium", "small"] {
            if let n = sizes[key], n > 0 { parts.append("\(n) \(key)") }
        }
        guard !parts.isEmpty else { return nil }
        return "Size in view: " + parts.joined(separator: ", ")
    }

    /// Side counts in words. Buckets with nothing in them are omitted rather
    /// than shown as "0 left", which would fill the row with non-answers.
    static func sidesText(_ sides: SceneSideCounts) -> String {
        var parts: [String] = []
        if sides.left > 0 { parts.append("\(sides.left) left") }
        if sides.centre > 0 { parts.append("\(sides.centre) centre") }
        if sides.right > 0 { parts.append("\(sides.right) right") }
        // Its own word, never folded into "centre": a frame whose size was
        // never learned has not placed anything in the middle of the view, it
        // has placed nothing.
        if sides.unknown > 0 { parts.append("\(sides.unknown) side unknown") }
        return parts.joined(separator: ", ")
    }

    // MARK: Anonymity

    /// Said out loud rather than left to be noticed.
    ///
    /// The deliberate contrast with Document Memory: that cartridge publishes a
    /// `provenance` block it labels `joinable: true`, because a document is a
    /// record. This one publishes nothing joinable, because a scene is not.
    private var anonymityFootnote: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(SceneRefusals.joinabilityNote)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
            if !reading.refusals.refusedEntityFields.isEmpty {
                Text(
                    "Refused outright: "
                        + reading.refusals.refusedEntityFields.map(\.field).joined(separator: ", ")
                        + "."
                )
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Flow

/// The counters that say whether the count on screen could keep up.
///
/// `frames_skipped` is on the wire deliberately — a silently dropped frame is
/// indistinguishable from a quiet room — and it is shown here for the same
/// reason. A sustained non-zero value also stretches the tracker's departure
/// bound, which is a frame count rather than a duration, so a count can include
/// someone who has already left.
struct SceneFlowFootnote: View {
    let reading: SceneReading

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            if reading.framesSkipped > 0 {
                Text("\(reading.framesSkipped) frames were skipped because the Tower was busy. Counts are less stable than they look while that is climbing.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let note = reading.observedAtNote, reading.observedAtTowerReceipt != nil {
                // The Tower's own sentence about what its timestamp is. Carried
                // rather than paraphrased, because the paraphrase everybody
                // reaches for — "when the glasses saw it" — is the exact
                // substitution it exists to prevent.
                Text(note)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
