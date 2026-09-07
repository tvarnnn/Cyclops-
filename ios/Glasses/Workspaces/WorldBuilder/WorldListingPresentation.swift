//
//  WorldListingPresentation.swift
//  Glasses
//

import Foundation

/// The labelling and grouping the saved-worlds picker draws from, with no
/// view in it.
///
/// ## Why it is a type and not a few computed properties on the view
///
/// The picker was written against a two-world fixture and first met a real
/// root on 2026-09-06: 162 worlds, 96 of them session-less shells offered as
/// openable rows, 156 unnamed and drawn as hex ids, 29 abandoned sessions
/// labelled "still open". None of that was a decoding error — every row was
/// true — it was a presentation that had never been asked what 228 true rows
/// look like. Everything here is a pure function of the decoded listing, so
/// a synthetic root of that shape can be asserted on without rendering.
nonisolated enum WorldListingPresentation {

    // MARK: Titles

    /// The Tower's name when it gave one; otherwise a dated title. Never the
    /// bare id.
    static func title(for entry: WorldListingEntry) -> String {
        if let name = entry.displayName, !name.isEmpty { return name }
        return datedTitle(updatedAt: entry.updatedAt)
    }

    /// `"Walk · 6 Sep 2026, 20:14"`, from a Tower-clock timestamp.
    ///
    /// The word is "Walk" because that is what a World Builder session is —
    /// the wearer walked and the Tower mapped — and not "World", which every
    /// row on the screen already is. The time is in the phone's zone, since
    /// a person reading it is asking "was that this evening?", and a fixed
    /// format rather than the locale's so two rows never sort by eye
    /// differently from how they sort by clock.
    static func datedTitle(updatedAt: Double) -> String {
        "Walk · " + dateFormatter().string(from: Date(timeIntervalSince1970: updatedAt))
    }

    /// Built per call rather than held: `DateFormatter` is not `Sendable`,
    /// and this type is deliberately usable from any isolation.
    private static func dateFormatter() -> DateFormatter {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "d MMM yyyy, HH:mm"
        return formatter
    }

    // MARK: Grouping

    /// The listing split for the picker: worlds worth a section, and the
    /// shells collapsed behind one line at the bottom.
    nonisolated struct Grouped: Equatable, Sendable {
        /// Worlds with at least one session, in the Tower's order.
        let primary: [WorldListingEntry]
        /// Worlds with none, in the Tower's order. Not offered as primary
        /// rows; a disclosure names how many there are.
        let empty: [WorldListingEntry]

        /// `"3 worlds with no sessions"`, or `nil` when there are none to
        /// disclose — an empty disclosure is a control that opens onto
        /// nothing.
        var emptyHeading: String? {
            switch empty.count {
            case 0: return nil
            case 1: return "1 world with no sessions"
            default: return "\(empty.count) worlds with no sessions"
            }
        }
    }

    static func grouped(_ worlds: [WorldListingEntry]) -> Grouped {
        Grouped(
            primary: worlds.filter(\.hasSessions),
            empty: worlds.filter { !$0.hasSessions }
        )
    }

    // MARK: Session rows

    /// One word for what the session is, from the Tower's `state` when it
    /// sent one, else from what an older Tower did send.
    ///
    /// The vocabulary is the panel's: a row that says "Interrupted" opens
    /// onto a canvas headlined "Interrupted", never onto one that says
    /// something else about the same session. An unknown state word is shown
    /// as itself rather than mapped to the nearest known one.
    static func stateBadge(for session: WorldListingSession) -> String? {
        if let state = session.state {
            switch state {
            case .receiving: return "Building"
            case .finalizing: return "Finishing"
            case .complete: return "Complete"
            case .interrupted: return "Interrupted"
            case .unbuilt: return "No geometry"
            default: return state.rawValue
            }
        }
        // No `state`: a Tower older than 2026-09-06. `abandoned` may still be
        // there (it landed a little earlier), and it is the one word that
        // stops a dead builder's record reading as a live one.
        if session.abandoned == true { return "unfinished" }
        if session.isStillOpen { return "still open" }
        if !session.hasGeometry { return "no geometry" }
        return nil
    }

    /// `"467 keyframes"`, or `nil` when neither count was sent. The journal's
    /// count stands in when the record's is the start-of-session zero — see
    /// `WorldListingSession.keyframeCount`.
    static func keyframeCaption(for session: WorldListingSession) -> String? {
        guard let count = session.keyframeCount else { return nil }
        return count == 1 ? "1 keyframe" : "\(count) keyframes"
    }

    /// `"1 session"` / `"4 sessions"`.
    static func sessionCount(_ count: Int) -> String {
        count == 1 ? "1 session" : "\(count) sessions"
    }
}
