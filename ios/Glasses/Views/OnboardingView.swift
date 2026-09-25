//
//  OnboardingView.swift
//  Glasses
//

import SwiftUI

/// "How Glasses works": four cards, shown full-screen on the first launch and
/// reopened from Settings.
///
/// It decides nothing about *when* it is shown; that is
/// `OnboardingPresentation`, in `ContentView`, which also keeps it off the
/// screen during a capture. It talks to nothing: the two links on the second
/// card are closures, and the host opens Connections or Settings over it.
struct OnboardingView: View {
    let mode: OnboardingMode
    let onOpenConnections: () -> Void
    let onOpenSettings: () -> Void
    let onEnd: (OnboardingEnd) -> Void

    @State private var page: OnboardingCard.ID = .whatItDoes
    @AccessibilityFocusState private var focusedTitle: OnboardingCard.ID?
    @State private var focusMove: Task<Void, Never>?
    @Environment(\.accessibilityVoiceOverEnabled) private var isVoiceOverRunning
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    private let cards = OnboardingText.cards

    private var index: Int {
        cards.firstIndex { $0.id == page } ?? 0
    }

    private var isLast: Bool { index == cards.count - 1 }

    var body: some View {
        VStack(spacing: 0) {
            // The system pager: a swipe follows the finger, and VoiceOver's
            // three-finger swipe turns the page.
            TabView(selection: $page) {
                ForEach(cards) { card in
                    OnboardingCardView(
                        card: card,
                        focusedTitle: $focusedTitle,
                        onOpenConnections: onOpenConnections,
                        onOpenSettings: onOpenSettings
                    )
                    .tag(card.id)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))

            footer
        }
        // Home stays in the window under the cover; VoiceOver must not wander
        // into it past the last control here.
        .accessibilityElement(children: .contain)
        .accessibilityAddTraits(.isModal)
        .background(Color(.systemGroupedBackground))
        // The VoiceOver "escape" gesture closes the cards, like Skip.
        .accessibilityAction(.escape) { onEnd(.skipped) }
        .onDisappear { focusMove?.cancel() }
    }

    /// Continue, then Skip beneath it, where a thumb is: the order of the
    /// system's own setup screens. Both scale with Dynamic Type, which a
    /// navigation-bar button does not.
    private var footer: some View {
        VStack(spacing: 4) {
            PageIndicator(
                count: cards.count,
                index: index,
                // VoiceOver stays on the indicator, as on the system's page
                // control, and reads the new value.
                onStep: { step in move(by: step, movingFocus: false) }
            )

            Button {
                if isLast {
                    onEnd(.finished)
                } else {
                    move(by: 1)
                }
            } label: {
                // Title 3 rather than Headline: white on the system blue is
                // 3.5:1, which the accessibility audit passes only for large
                // text.
                Text(isLast ? OnboardingText.finishButton(mode) : OnboardingText.continueButton)
                    .font(.title3.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .accessibilityIdentifier(isLast ? "onboarding-finish" : "onboarding-continue")

            // On the last card the primary button already ends them. A
            // hidden stand-in keeps the height, so the pages above do not
            // jump; `.hidden()` also takes it out of VoiceOver and the audit.
            if isLast {
                skipLabel.hidden()
            } else {
                Button {
                    onEnd(.skipped)
                } label: {
                    skipLabel
                }
                .buttonStyle(.borderless)
                .accessibilityHint(OnboardingText.closeHint)
                .accessibilityIdentifier("onboarding-close")
            }
        }
        .padding(.horizontal, dynamicTypeSize.isAccessibilitySize ? 16 : 24)
        .padding(.top, 2)
    }

    // Title 3 for the same reason as Continue: the tint on this background
    // is 3.2:1, enough only for large text.
    private var skipLabel: some View {
        Text(OnboardingText.closeButton(mode))
            .font(.title3)
            .frame(maxWidth: .infinity, minHeight: 44)
            .contentShape(.rect)
    }

    /// Turns the page. With VoiceOver on, it then moves VoiceOver to the new
    /// card's title rather than leaving it on the button.
    ///
    /// Only with VoiceOver on, and only once the page has settled: focus sent
    /// to a card the pager is still scrolling in asks it to scroll to that
    /// card a second time, mid-flight. `withAnimation`'s completion is no
    /// measure of settling, because the pager's scroll is UIKit's. A swipe
    /// needs none of this; VoiceOver follows its own three-finger scroll.
    private func move(by step: Int, movingFocus: Bool = true) {
        let target = cards[min(max(index + step, 0), cards.count - 1)].id
        guard target != page else { return }
        withAnimation { page = target }
        focusMove?.cancel()
        guard movingFocus, isVoiceOverRunning else { return }
        focusMove = Task {
            try? await Task.sleep(for: .milliseconds(600))
            guard !Task.isCancelled, page == target else { return }
            focusedTitle = target
        }
    }
}

// MARK: - A card

private struct OnboardingCardView: View {
    let card: OnboardingCard
    var focusedTitle: AccessibilityFocusState<OnboardingCard.ID?>.Binding
    let onOpenConnections: () -> Void
    let onOpenSettings: () -> Void

    /// The card's picture, scaled with the text but kept from filling the
    /// screen at the largest sizes.
    @ScaledMetric(relativeTo: .largeTitle) private var symbolSize: CGFloat = 36
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    /// Narrower at the accessibility sizes, where every point of line length
    /// keeps a long word whole.
    private var sideMargin: CGFloat { dynamicTypeSize.isAccessibilitySize ? 16 : 24 }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Image(systemName: card.symbol)
                    .font(.system(size: min(symbolSize, 72)))
                    .foregroundStyle(.tint)
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 10) {
                    Text(card.title)
                        .font(.largeTitle.bold())
                        .accessibilityAddTraits(.isHeader)
                        .accessibilityFocused(focusedTitle, equals: card.id)
                        .accessibilityIdentifier("onboarding-title-\(card.id.slug)")

                    // The label colour: secondary grey on the grouped
                    // background is under 4.5:1 at this size.
                    Text(card.lead)
                        .font(.body)
                }
                .fixedSize(horizontal: false, vertical: true)

                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(card.points.enumerated()), id: \.offset) { offset, point in
                        if offset > 0 {
                            Divider().padding(.leading, 16)
                        }
                        OnboardingPointRow(point: point)
                    }
                }
                .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))

                if card.id == .whatYouNeed {
                    links
                }
            }
            .frame(maxWidth: 560, alignment: .leading)
            .padding(.horizontal, sideMargin)
            .padding(.top, 4)
            .padding(.bottom, 24)
            .frame(maxWidth: .infinity)
        }
        .scrollBounceBehavior(.basedOnSize)
        // A card taller than the screen says so as it appears.
        .scrollIndicatorsFlash(onAppear: true)
    }

    /// Rows like the system's own grouped lists: the words in the label
    /// colour, which reads at every size, and a chevron for "opens a screen".
    private var links: some View {
        VStack(alignment: .leading, spacing: 0) {
            OnboardingLinkRow(
                title: OnboardingText.openConnections,
                symbol: "link",
                hint: OnboardingText.openConnectionsHint,
                identifier: "onboarding-open-connections",
                action: onOpenConnections
            )
            Divider().padding(.leading, 16)
            OnboardingLinkRow(
                title: OnboardingText.openSettings,
                symbol: "gearshape",
                hint: OnboardingText.openSettingsHint,
                identifier: "onboarding-open-settings",
                action: onOpenSettings
            )
        }
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }
}

private struct OnboardingLinkRow: View {
    let title: String
    let symbol: String
    let hint: String
    let identifier: String
    let action: () -> Void

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        Button(action: action) {
            Group {
                // At the accessibility sizes the symbol goes above the words
                // and the chevron goes, or "Connections" breaks mid-word.
                if dynamicTypeSize.isAccessibilitySize {
                    VStack(alignment: .leading, spacing: 6) {
                        symbolView
                        Text(title)
                            .foregroundStyle(.primary)
                    }
                } else {
                    HStack(spacing: 12) {
                        Label {
                            Text(title)
                                .foregroundStyle(.primary)
                        } icon: {
                            symbolView
                        }
                        Spacer(minLength: 0)
                        Image(systemName: "chevron.forward")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(.tertiary)
                            .accessibilityHidden(true)
                    }
                }
            }
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .accessibilityHint(hint)
        .accessibilityIdentifier(identifier)
    }

    private var symbolView: some View {
        Image(systemName: symbol)
            .foregroundStyle(.tint)
            .accessibilityHidden(true)
    }
}

/// A symbol, a title and a sentence. The symbol moves above the words at the
/// accessibility sizes, where a column beside them would squeeze a word onto
/// two lines.
private struct OnboardingPointRow: View {
    let point: OnboardingPoint

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @ScaledMetric(relativeTo: .title2) private var symbolColumn: CGFloat = 32

    var body: some View {
        Group {
            if dynamicTypeSize.isAccessibilitySize {
                VStack(alignment: .leading, spacing: 8) {
                    symbol
                    words
                }
            } else {
                HStack(alignment: .firstTextBaseline, spacing: 14) {
                    symbol
                        .frame(width: symbolColumn)
                    words
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .accessibilityElement(children: .combine)
    }

    private var symbol: some View {
        Image(systemName: point.symbol)
            .font(.title2)
            .foregroundStyle(.tint)
            .accessibilityHidden(true)
    }

    private var words: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(point.title)
                .font(.headline)
            // The label colour, as for the lead: the secondary grey is
            // 3.5:1 on the card, short of 4.5:1 for text this size.
            Text(point.text)
                .font(.subheadline)
        }
        .fixedSize(horizontal: false, vertical: true)
    }
}

/// Where the reader is, as dots, and to VoiceOver as "Card, 2 of 4",
/// adjustable with a swipe up or down like the system's page control.
private struct PageIndicator: View {
    let count: Int
    let index: Int
    let onStep: (Int) -> Void

    @ScaledMetric(relativeTo: .caption) private var dot: CGFloat = 7

    var body: some View {
        HStack(spacing: dot * 1.2) {
            ForEach(0..<count, id: \.self) { position in
                Circle()
                    .fill(position == index ? Color.primary : Color(.tertiaryLabel))
                    .frame(width: dot, height: dot)
            }
        }
        // VoiceOver can adjust it, so it is a control, and a control needs a
        // target a finger can find: 44 points, like the system's.
        .frame(minWidth: 88, minHeight: 44)
        .contentShape(.rect)
        .accessibilityElement()
        .accessibilityLabel(OnboardingText.pageLabel)
        .accessibilityValue(OnboardingText.pageValue(index, of: count))
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: onStep(1)
            case .decrement: onStep(-1)
            @unknown default: break
            }
        }
        .accessibilityIdentifier("onboarding-page")
    }
}
