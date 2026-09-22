//
//  InteractiveWorldCanvas.swift
//  Glasses
//

import SwiftUI

/// A `Canvas` the reader can pan, pinch and turn, and double-tap to reset.
///
/// It draws nothing itself. The caller's `draw` closure receives the context
/// with `WorldViewportState.transform(in:)` already concatenated, so a drawing
/// written against the canvas's own coordinates — exactly as `FragmentCanvas`
/// writes one — comes out slid, scaled and turned without knowing it was.
///
/// ## Two states, not one
///
/// A gesture reports its *total* translation, magnification or rotation since
/// it began, on every change. Adding each report to the live state would
/// compound them; the live state is therefore always rebuilt as
/// `committed ⊕ this gesture's total`, and the gesture's end promotes it to
/// `committed`. Three gestures run `simultaneously`, and each rebuilds only its
/// own component from the same committed base, so a pinch that drifts does not
/// erase a pan that finished a moment ago.
///
/// Why not `@GestureState`: it resets to its initial value when the gesture
/// ends, which is the behaviour wanted for the *in-progress* half and exactly
/// the wrong behaviour for the pan that should stay where the finger left it.
/// Two `@State`s say what is meant.
///
/// 2D only, and see `WorldViewportState` for why.
struct InteractiveWorldCanvas: View {
    let draw: (inout GraphicsContext, CGSize) -> Void

    @State private var committed = WorldViewportState.identity
    @State private var live = WorldViewportState.identity

    var body: some View {
        Canvas { context, size in
            context.concatenate(live.transform(in: size))
            draw(&context, size)
        }
        // A hit anywhere on the canvas, not only on a drawn pixel — most of a
        // point cloud's tile is empty space.
        .contentShape(Rectangle())
        .onTapGesture(count: 2) {
            committed = .identity
            live = .identity
        }
        .gesture(pan.simultaneously(with: pinch).simultaneously(with: turn))
    }

    private var pan: some Gesture {
        DragGesture()
            .onChanged { value in
                live.offset = CGSize(
                    width: committed.offset.width + value.translation.width,
                    height: committed.offset.height + value.translation.height
                )
            }
            .onEnded { _ in committed = live }
    }

    private var pinch: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                live.scale = WorldViewportState.clampedScale(
                    committed.scale * value.magnification
                )
            }
            .onEnded { _ in committed = live }
    }

    private var turn: some Gesture {
        RotateGesture()
            .onChanged { value in
                live.rotation = committed.rotation + CGFloat(value.rotation.radians)
            }
            .onEnded { _ in committed = live }
    }
}
