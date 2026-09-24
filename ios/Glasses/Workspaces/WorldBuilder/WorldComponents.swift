//
//  WorldComponents.swift
//  Glasses
//

import Foundation

// MARK: - The pieces of one walk: the room, and the areas it could not place

/// `components[].state`: `placed` (the room) or `unplaced`
/// (`WORLD-BUILDER-COMPONENTS.md` §2.1). A `RawRepresentable` struct, for the
/// reason `WorldSelectionMode` gives.
nonisolated struct WorldComponentState: RawRepresentable, Equatable, Sendable, Hashable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    static let placed = WorldComponentState(rawValue: "placed")
    static let unplaced = WorldComponentState(rawValue: "unplaced")
}

/// `components[].shown_as`: how the Tower shows a piece (§2.3). The phone
/// reads this and never recomputes it from `keyframes` or the spans.
nonisolated struct WorldComponentShownAs: RawRepresentable, Equatable, Sendable, Hashable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    /// The placed component: drawn by the room's own route, exactly as today.
    static let room = WorldComponentShownAs(rawValue: "room")
    /// Unplaced, and big enough to build and show on its own (§5).
    static let area = WorldComponentShownAs(rawValue: "area")
    /// Unplaced, below both floors: counted in the footer, never drawn.
    static let none = WorldComponentShownAs(rawValue: "none")
}

/// One `[start, end]` from `capture_spans_s`: seconds since the session's
/// `started_at`, on the Tower's receipt clock. An envelope -- "captured between
/// A and B" -- not a claim of continuous presence (§2.4 rule 5).
nonisolated struct WorldCaptureSpan: Equatable, Hashable, Sendable {
    let start: Double
    let end: Double
}

/// One entry of `components[]` (§2.1).
///
/// Every field the contract names, and no other: in particular no extent,
/// distance or name, because the Tower sends none (§2.4 rules 6 and 7) and the
/// phone never derives one.
nonisolated struct WorldComponent: Equatable, Sendable {
    /// 16 lower-hex characters: a path segment of the area routes. Compared for
    /// equality only, and not stable across a re-finish (§2.4 rule 3).
    let id: String
    let state: WorldComponentState
    /// The first of `reasons`, or `nil` for the room. Not shown on screen: the
    /// phone's copy is the same for every reason (§8), so a new reason word
    /// from a newer Tower changes nothing here (§2.4 rule 8).
    let reason: String?
    let reasons: [String]
    let shownAs: WorldComponentShownAs
    /// Published keyframes: what the phone calls **photos**.
    let keyframes: Int
    let keyframesPhone: Int?
    let captureSpans: [WorldCaptureSpan]
    /// Whether opening this component would show something now. For an area:
    /// whether the area's render route would answer 200.
    let hasGeometry: Bool
    /// This component's OWN photographic build (`null` for a `none` entry).
    /// The same seven words, through the same decoder, as everywhere else.
    let photographic: WorldPhotographicReport?

    /// Whether `id` can be put into an area route as it stands: exactly 16
    /// lower-hex characters, the alphabet §2.1 fixes so it "must need no
    /// encoding". Checked like `WorldAssetRequest.isDigest`.
    nonisolated static func isAreaID(_ value: String) -> Bool {
        value.utf8.count == 16
            && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
}

// In an extension so the memberwise initialiser survives.
nonisolated extension WorldComponent {
    /// `nil` when the entry disagrees with the contract: a missing or mistyped
    /// required field, or a malformed span. The caller then treats the whole
    /// list as not computed, which is today's behaviour, rather than showing a
    /// walk with a piece silently missing.
    init?(json: [String: Any]) {
        guard
            let id = json["id"] as? String, !id.isEmpty,
            let state = json["state"] as? String, !state.isEmpty,
            let shownAs = json["shown_as"] as? String, !shownAs.isEmpty,
            let keyframes = json["keyframes"] as? Int,
            let hasGeometry = json["has_geometry"] as? Bool
        else { return nil }
        var spans: [WorldCaptureSpan] = []
        if let raw = json["capture_spans_s"] {
            guard let pairs = raw as? [Any] else { return nil }
            for pair in pairs {
                guard
                    let values = pair as? [Any], values.count == 2,
                    let start = (values[0] as? NSNumber)?.doubleValue,
                    let end = (values[1] as? NSNumber)?.doubleValue,
                    start.isFinite, end.isFinite, start >= 0, end >= start
                else { return nil }
                spans.append(WorldCaptureSpan(start: start, end: end))
            }
        }
        self.id = id
        self.state = WorldComponentState(rawValue: state)
        self.reason = json["reason"] as? String
        self.reasons = (json["reasons"] as? [Any])?.compactMap { $0 as? String } ?? []
        self.shownAs = WorldComponentShownAs(rawValue: shownAs)
        self.keyframes = keyframes
        self.keyframesPhone = json["keyframes_phone"] as? Int
        self.captureSpans = spans
        self.hasGeometry = hasGeometry
        self.photographic = WorldPhotographicReport(json: json["photographic"])
    }
}

/// A session's `components`, as the phone uses them: the room, the areas it
/// can offer, and the short stretches it can only count.
///
/// ## `nil` is "not computed", and so is anything the phone cannot trust
///
/// §2.4 rule 1: `components: null` -- every world built before this change --
/// means not computed, never "no areas", and the phone shows exactly today's
/// screens. Rule 8 widens that: a list with no `placed` entry the phone
/// understands is treated as `null`. This type's failable initialiser is where
/// both happen, so no screen ever has to ask whether a list is trustworthy:
/// if it exists, it has a room first, and every area in it can be opened by
/// its id.
nonisolated struct WorldComponents: Equatable, Sendable {
    /// Every entry, in the Tower's order (§2.4 rule 2).
    let entries: [WorldComponent]

    /// The placed component: the room. Always `entries[0]`.
    var room: WorldComponent { entries[0] }

    /// The entries shown as areas, in list order. "Area k" is position k-1
    /// here (§2.4 rule 2) -- a number, not a name, and not stable across a
    /// re-finish.
    var areas: [WorldComponent] { entries.filter { $0.shownAs == .area } }

    /// The entries counted but never drawn: `none`, and -- rule 8 -- any
    /// `shown_as` word this build does not know. Never the room.
    var shortStretches: [WorldComponent] {
        entries.dropFirst().filter { $0.shownAs != .area && $0.shownAs != .room }
    }

    /// The footer's `P`: the photos in the short stretches.
    var shortStretchKeyframes: Int { shortStretches.reduce(0) { $0 + $1.keyframes } }

    /// Areas whose own photographic build is not finished (`running`, `owed`,
    /// `unobservable`). Counted for "N areas of this walk are still being
    /// finished".
    var areasStillFinishing: Int {
        areas.filter { $0.photographic?.standing.isUnfinished == true }.count
    }

    /// Whether this list would change nothing on screen: no areas, no short
    /// stretches (§8: "a walk the gate kept whole looks exactly as today").
    var isWhole: Bool { areas.isEmpty && shortStretches.isEmpty }

    /// Only what the screens show, for deciding whether a newer list is worth
    /// redrawing: ids, `shown_as`, `has_geometry` and the photographic word
    /// (§3.2: "comparing the array by value"). Detail prose moving is not a
    /// change a reader can see.
    var displayKey: [String] {
        entries.map {
            "\($0.id)|\($0.shownAs.rawValue)|\($0.hasGeometry)|\($0.photographic?.state.rawValue ?? "-")"
                + "|\($0.keyframes)"
        }
    }

    init(entries: [WorldComponent]) {
        self.entries = entries
    }

    /// `nil` for `null`, an absent key, a non-list, an empty list (rule 1 says
    /// a computed list "is never empty"), a list whose first entry is not the
    /// one `placed` room, a second `placed` entry, any entry that does not
    /// decode, and an area whose id could not address its routes.
    init?(json: Any?) {
        guard let raw = json as? [Any], !raw.isEmpty else { return nil }
        var entries: [WorldComponent] = []
        for item in raw {
            guard let object = item as? [String: Any], let entry = WorldComponent(json: object) else {
                return nil
            }
            entries.append(entry)
        }
        guard
            entries[0].state == .placed,
            entries.dropFirst().allSatisfy({ $0.state != .placed }),
            entries.allSatisfy({ $0.shownAs != .area || WorldComponent.isAreaID($0.id) })
        else { return nil }
        self.entries = entries
    }
}

// MARK: - The words

/// How the phone describes a walk's components (§8). Pure and `nonisolated`,
/// like `WorldListingPresentation`, so every sentence is tested without a view.
nonisolated enum WorldComponentsPresentation {

    /// `m:ss` of the walk: minutes are not wrapped past 59, hours are never used
    /// (§8, C1 E14), and seconds are **truncated** -- exactly the Tower's
    /// `surface_render.walk_time`, which writes the area page's own caption,
    /// so the phone and the page never differ by a second. It is also what
    /// the contract's own examples show: §8's "0:29–0:33 and 1:49–2:12" and
    /// §5.4's "from 1:26 to 1:49" for spans ending at 33.1 s, 132.4 s and
    /// 109.6 s. (It was ceilinged at the end before G1, which put "0:34" and
    /// "2:13" beside a page saying "0:33" and "2:12".)
    static func clock(_ seconds: Double) -> String {
        let whole = max(0, Int(seconds.rounded(.down)))
        return "\(whole / 60):" + String(format: "%02d", whole % 60)
    }

    /// `0:29–0:33`, with an en dash.
    static func span(_ span: WorldCaptureSpan) -> String {
        clock(span.start) + "–" + clock(span.end)
    }

    /// `0:29–0:33 and 1:49–2:12`; three or more are joined with commas and a
    /// final "and". `nil` for no spans. The ROWS' form (§8): every span.
    static func spans(_ spans: [WorldCaptureSpan]) -> String? {
        let parts = spans.map(span)
        switch parts.count {
        case 0: return nil
        case 1: return parts[0]
        case 2: return "\(parts[0]) and \(parts[1])"
        default: return parts.dropLast().joined(separator: ", ") + " and " + parts[parts.count - 1]
        }
    }

    /// The area viewer's range, exactly as §5.4 words it and the Tower's page
    /// writes it (`surface_render.area_caption`): ONE envelope, from the first
    /// span's start to the last span's end -- *from 1:26 to 1:49* for the
    /// target's bathroom `[[86.7, 97.1], [106.4, 109.6]]`. The spans are
    /// ascending (§2.1). `nil` for none. (G1-F1: the phone listed every span
    /// here, so the native caption and the page's disagreed on one screen.)
    static func envelopeInProse(_ spans: [WorldCaptureSpan]) -> String? {
        guard let first = spans.first, let last = spans.last else { return nil }
        return clock(first.start) + " to " + clock(last.end)
    }

    static func photos(_ count: Int) -> String {
        count == 1 ? "1 photo" : "\(count) photos"
    }

    /// The areas row's heading: *N more areas — captured on this walk but not
    /// placed in this room.* `nil` with no areas.
    static func areasHeading(_ components: WorldComponents) -> String? {
        let count = components.areas.count
        guard count > 0 else { return nil }
        let noun = count == 1 ? "1 more area" : "\(count) more areas"
        return "\(noun) — captured on this walk but not placed in this room."
    }

    /// One area's line: *Area k · 0:29–0:33 and 1:49–2:12 · 66 photos*.
    static func areaLine(number: Int, area: WorldComponent) -> String {
        var parts = ["Area \(number)"]
        if let spans = spans(area.captureSpans) { parts.append(spans) }
        parts.append(photos(area.keyframes))
        return parts.joined(separator: " · ")
    }

    /// What an area entry can do, from its own `has_geometry` and word (§8).
    enum AreaAvailability: Equatable {
        /// Something to draw: the entry opens the area viewer.
        case opens
        /// Nothing to draw yet, and its build is `running` or `owed` (or the
        /// Tower could not tell): *Improving*, and it does not open.
        case improving
        /// Nothing to draw and a settled word: *could not be built*.
        case couldNotBeBuilt
    }

    static func availability(of area: WorldComponent) -> AreaAvailability {
        if area.hasGeometry { return .opens }
        if area.photographic?.standing.isUnfinished == true { return .improving }
        return .couldNotBeBuilt
    }

    /// The trailing word on an area's line, or `nil` for one that opens.
    static func availabilityWord(_ availability: AreaAvailability) -> String? {
        switch availability {
        case .opens: return nil
        case .improving: return "Improving"
        case .couldNotBeBuilt: return "could not be built"
        }
    }

    /// The footer: *K short stretches (P photos) could not be placed or
    /// shown.* `nil` with none.
    static func footer(_ components: WorldComponents) -> String? {
        let count = components.shortStretches.count
        guard count > 0 else { return nil }
        let noun = count == 1 ? "1 short stretch" : "\(count) short stretches"
        return "\(noun) (\(photos(components.shortStretchKeyframes))) could not be placed or shown."
    }

    /// The room caption's suffix: ` · N more areas shown separately`, or `nil`.
    static func roomCaptionSuffix(_ components: WorldComponents?) -> String? {
        guard let count = components?.areas.count, count > 0 else { return nil }
        return count == 1 ? " · 1 more area shown separately" : " · \(count) more areas shown separately"
    }

    /// The area viewer's header: *Area k of N — not placed in the room*.
    static func areaHeader(number: Int, of total: Int) -> String {
        "Area \(number) of \(total) — not placed in the room"
    }

    /// The area viewer's native caption, one line per rung (§5.4). The
    /// appearance rung names the imagery and when it was captured; the surface
    /// rung names the surfaces. Both say it is not placed and not comparable,
    /// and neither claims a scale.
    static func areaCaption(
        representation: WorldRenderRepresentation?, spans spanList: [WorldCaptureSpan], levelled: Bool? = nil
    ) -> String {
        let when = envelopeInProse(spanList).map { " from \($0) of this walk" } ?? " of this walk"
        let first: String
        switch representation {
        case .appearance: first = "The camera's own images, faces redacted,\(when)."
        case .surface: first = "Surfaces the Tower reconstructed from the walk,\(when)."
        default: first = "What the Tower reconstructed\(when)."
        }
        var caption = first + " This area could not be placed relative to the room, so it is shown on "
            + "its own: its position, direction and size are not comparable with the room's."
        if levelled == false {
            caption += " Its vertical could not be estimated, so it may look tilted."
        }
        return caption + " Not to scale."
    }
}

// MARK: - The area routes' own answers

/// The four sentences an area route answers 404 with
/// (`WORLD-BUILDER-COMPONENTS.md` §5.1, C1 E4). **Stable identifiers,
/// compared for equality**, exactly as `WorldRenderFetchError
/// .unmatchedRouteDetail` compares FastAPI's `Not Found`: their text never
/// changes without a contract change, and a Tower test pins it.
nonisolated enum WorldAreaRouteAnswer {
    /// Terminal for that id: a malformed id, one from an earlier solve, the
    /// room's id, a `none` entry's id. The session was re-finished; the areas
    /// row in the room is current.
    static let noSuchArea = "no such area in this session"
    /// Terminal: its build failed or was declined and nothing is drawable.
    static let couldNotBeBuilt = "this area could not be built"
    /// Terminal: `components` is `null` -- every older world.
    static let sessionHasNoAreas = "this session has no areas"
    /// Transient: its build is owed or running and nothing is drawable yet.
    static let notBuiltYet = "this area has not been built yet"

    static let terminal: Set<String> = [noSuchArea, couldNotBeBuilt, sessionHasNoAreas]

    /// Whether a 404's `detail` says this area will not be served, whatever
    /// the phone does: stop following and offer the way back to the room.
    static func isTerminal(_ detail: String?) -> Bool {
        detail.map(terminal.contains) ?? false
    }

    /// Whether a 404's `detail` is one of the four at all.
    static func isAreaAnswer(_ detail: String?) -> Bool {
        isTerminal(detail) || detail == notBuiltYet
    }
}
