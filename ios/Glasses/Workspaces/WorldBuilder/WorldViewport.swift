//
//  WorldViewport.swift
//  Glasses
//

import CoreGraphics
import Foundation

/// Where the reader has panned, zoomed and turned a canvas to.
///
/// Pure value, no SwiftUI in it, so the arithmetic that decides where a point
/// lands can be tested without rendering. The view (`InteractiveWorldCanvas`)
/// owns two of these — the committed one and the one a gesture is currently
/// moving — and asks this for the matrix.
///
/// **2D only, on purpose.** `up_axis` is `"unknown"` on every manifest this
/// build has seen, so the only honest projection is the top-down `(x, z)` the
/// tiles already draw, and the only honest interactions are the three that
/// keep it a plan view: slide it, scale it, turn it. There is no tilt here
/// because a tilt would have to guess which way is up.
///
/// The transform is about the **canvas centre**, so a pinch zooms into the
/// middle of what is on screen rather than into the top-left corner, and a
/// rotation turns the picture in place rather than swinging it off-canvas.
nonisolated struct WorldViewportState: Equatable, Sendable {
    /// Uniform. Clamped to `scaleRange` wherever it is set from a gesture.
    var scale: CGFloat = 1
    /// In canvas points, applied after the rotation and scale.
    var offset: CGSize = .zero
    /// Radians. Positive turns the picture the way a `RotateGesture` value
    /// reports, so the two are added without a sign flip.
    var rotation: CGFloat = 0

    /// Wide enough that a 30,000-unit cluster beside a 3-unit one can be
    /// examined either way; bounded so a runaway pinch cannot scale the
    /// picture to a single pixel or to a single point filling the screen.
    static let scaleRange: ClosedRange<CGFloat> = 0.25...20

    static let identity = WorldViewportState()

    /// `scale`, held inside `scaleRange`.
    static func clampedScale(_ scale: CGFloat) -> CGFloat {
        Swift.min(Swift.max(scale, scaleRange.lowerBound), scaleRange.upperBound)
    }

    /// The same state with its scale held inside `scaleRange`.
    func clamped() -> WorldViewportState {
        var copy = self
        copy.scale = Self.clampedScale(scale)
        return copy
    }

    /// The matrix a `GraphicsContext` concatenates before drawing.
    ///
    /// Reads, for a point `p` and the canvas centre `c`:
    /// `p' = R(rotation) · (scale · (p − c)) + c + offset`.
    ///
    /// Built with CoreGraphics' own operations rather than a hand-written
    /// matrix. Each `…By` call **pre-concatenates** — the new operation is
    /// applied to the point *before* the transform it was called on — so the
    /// chain below is read bottom-up as what happens to the point: move the
    /// centre to the origin, scale, rotate, move the centre back plus the pan.
    func transform(in size: CGSize) -> CGAffineTransform {
        let centre = CGPoint(x: size.width / 2, y: size.height / 2)
        var transform = CGAffineTransform.identity
        transform = transform.translatedBy(
            x: centre.x + offset.width, y: centre.y + offset.height
        )
        transform = transform.rotated(by: rotation)
        transform = transform.scaledBy(x: scale, y: scale)
        transform = transform.translatedBy(x: -centre.x, y: -centre.y)
        return transform
    }

    /// Where a canvas point lands under this state. The testable face of
    /// `transform(in:)`.
    func apply(_ point: CGPoint, in size: CGSize) -> CGPoint {
        point.applying(transform(in: size))
    }
}
