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
            case .finalizing:
                // The room is finished and only one of the walk's areas is
                // still being made (`scope: "area"`, COMPONENTS §3.4): the
                // walk is complete as far as its room goes, and the caption
                // says an area is not.
                if session.photographic?.standing.isAreaStillFinishing == true { return "Complete" }
                return "Finishing"
            case .complete:
                // The same rule `WorldStage.stage` applies to `.finalized`:
                // a record whose final solve was skipped, failed or
                // unavailable is what the walk produced WITHOUT its last,
                // best pass, and the canvas headlines it "Partial". The
                // listing has no such state word — the Tower's lifecycle
                // says `ready` regardless — but the row carries
                // `finalization`, so the row can say what the canvas will.
                // Before this, a Tower shut down mid-walk listed "Complete ·
                // no final pass" over a canvas reading "Partial".
                if WorldFinalSolve(word: session.finalization?.finalSolve).deniesAFinishedWorld {
                    return "Partial"
                }
                // Complete, and its photographic build failed: saved, but not
                // what the walk was for. Plain "Complete" over it is the T3
                // defect the Tower's `photographic` block exists to end
                // (`WORLD-BUILDER-IOS.md` §3a); the caption under the title
                // carries the full sentence.
                if session.photographic?.standing.isFailed == true {
                    return WorldPhotographicCopy.failedBadge
                }
                return "Complete"
            case .interrupted:
                // "Interrupted" with geometry opens onto a canvas headlined
                // "Interrupted"; without geometry the canvas says "Needs
                // retry" (`WorldStage.stage`, `.interrupted` arm), because
                // there is nothing to open and walking again is the only
                // thing that produces another one. The row says the same.
                return session.hasGeometry ? "Interrupted" : "Needs retry"
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

    /// The caption under a row that cannot be opened, or `nil` for one that
    /// can. Says which of the three "nothing to open" situations this is,
    /// because the Tower tells them apart on purpose: a walk still open may
    /// yet build; a walk that built and whose output is gone is
    /// `interrupted`; a walk that built nothing is `unbuilt`. The first
    /// version of this caption said "No geometry was built for this walk"
    /// under every one of them — including the row whose own badge said a
    /// build had run.
    static func noGeometryCaption(for session: WorldListingSession) -> String? {
        guard !session.hasGeometry else { return nil }
        if session.isStillOpen {
            return "No geometry yet. The Tower may still build it for this walk."
        }
        if session.state == .interrupted {
            return "This walk's geometry is no longer on the Tower, so there is nothing to open."
        }
        return "No geometry was built for this walk, so there is nothing to open."
    }

    /// What to call one session on a row: when the walk started, in the
    /// phone's zone.
    ///
    /// The session's own id used to be the row's primary label — 32 hex
    /// characters, monospaced, on every session of every world. It identifies
    /// the record and tells a person nothing about which walk it was. The start
    /// time does, and it is the one thing a reader choosing between two walks
    /// of the same world actually has to go on.
    ///
    /// `"Walk · "` matches `datedTitle`, so a world's own row and its sessions'
    /// rows read in the same vocabulary.
    static func sessionTitle(for session: WorldListingSession) -> String {
        datedTitle(updatedAt: session.startedAt)
    }

    /// `"no final pass"`, or `nil`.
    ///
    /// ## The field this reads had never been read
    ///
    /// `WorldListingSession.finalization` is decoded off `GET /worlds` and was
    /// consulted by nothing. Its `final_solve` word is the difference between a
    /// walk that finished and a walk whose last, best pass was skipped or
    /// failed — and the picker is where a person chooses between two walks, so
    /// it is the place that difference is worth the two words it costs.
    ///
    /// **Only the words that say the pass did not happen.** `solved` gets no
    /// caption (a finished walk does not need a badge saying it finished),
    /// `pending` gets none (it may yet run), and a `nil` record gets none at
    /// all — silence is not a finding, which is the rule
    /// `WorldFinalSolve.notReported` states in one place for the whole app.
    static func finalSolveCaption(for session: WorldListingSession) -> String? {
        let solve = WorldFinalSolve(word: session.finalization?.finalSolve)
        guard solve.deniesAFinishedWorld else { return nil }
        return "no final pass"
    }

    /// The row's photographic sentence, or `nil`: only for a session the
    /// Tower lists `complete` whose photographic build failed, and only where
    /// the final pass ran (a row that already says "Partial" has said the
    /// truer thing). Every other word adds nothing to a row, and silence keeps
    /// an older world's row exactly as it was.
    static func photographicCaption(for session: WorldListingSession) -> String? {
        if session.state == .finalizing, session.photographic?.standing.isAreaStillFinishing == true {
            let count = session.components.map(\.areasStillFinishing).flatMap { $0 > 0 ? $0 : nil }
            return WorldPhotographicCopy.areasStillFinishingCaption(count: count)
        }
        guard session.state == .complete,
              session.photographic?.standing.isFailed == true,
              !WorldFinalSolve(word: session.finalization?.finalSolve).deniesAFinishedWorld
        else { return nil }
        return WorldPhotographicCopy.failedHeadline
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
