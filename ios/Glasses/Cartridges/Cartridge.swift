//
//  Cartridge.swift
//  Glasses
//

import Foundation

/// What a person can do with a cartridge in this build.
///
/// ## Why this stopped being a roadmap vocabulary
///
/// This enum used to be `.next` / `.planned` / `.future` — "Up next",
/// "Planned", "Future" — and its doc said an `.available` case would violate
/// Rule 3 because *"no module runtime exists on either side yet: the Tower's
/// module container is V0.8 and the first module is V0.9."*
///
/// **That premise is refuted by the wire.** A Tower probed on 2026-08-26
/// reports `module_state: "active"` with `module_id: "experimental-cv"` — the
/// container exists and a module is running in it — and `GET /cartridges`
/// declares `world_builder.status/2026-08-25` with `available: true`, which
/// this build implements and has drawn 7,086 live points from during a
/// physical walk. Three cartridges now reach a person; refusing to say so was
/// the Rule 3 violation, not the reverse.
///
/// ## Why the axis changed, and not just the words
///
/// The old values answered *"where does this module sit on the Tower
/// roadmap?"* The drawer is a product surface, and a person reading a badge
/// there is asking *"is there anything here for me right now?"* Those are
/// different questions, and answering the first in the second's place produced
/// a badge set that was actively inverted: **Visual Q&A read "Planned" — a
/// readier-sounding word — while having no Tower code at all, and World
/// Builder read "Future" while being the one cartridge with a device-validated
/// walk behind it.**
///
/// Roadmap position has not been lost. It lives in `docs/03-ROADMAP.md` and
/// each cartridge's `specPath`, which is where a roadmap belongs. It is no
/// longer rendered to a wearer as though it were a capability.
///
/// ## What these values are NOT
///
/// **This is static catalog data about *this build*, not a live claim about
/// the Tower you happen to be connected to.** Whether a given Tower can serve
/// a cartridge right now is resolved per-connection by
/// `CartridgeAvailability.resolve` and rendered inside each workspace. A badge
/// reading "Ready to test" beside a workspace saying "Nothing yet" is not a
/// contradiction: the app is ready and that Tower is not. Keeping those two
/// separate is the reason this type consults nothing at runtime.
nonisolated enum CartridgeStatus: Equatable, CaseIterable {
    /// This build implements the cartridge and can show real Tower data for it.
    ///
    /// Not a promise that any particular Tower is serving it this second —
    /// see the type doc. It is a promise that there is something here to try.
    case readyToTest
    /// The Tower implements it; no contract is offered on the wire yet.
    ///
    /// The workspace opens and refuses truthfully. This is deliberately
    /// distinct from `notBuilt`: the two call for opposite responses — one is
    /// waiting on a Tower decision that is already costed, the other is waiting
    /// on a backend nobody has written.
    case awaitingTower
    /// No backend exists anywhere. A specification only.
    case notBuilt

    var badge: String {
        switch self {
        case .readyToTest: return "Ready to test"
        case .awaitingTower: return "Awaiting Tower"
        case .notBuilt: return "Not built"
        }
    }

    /// Whether the badge should draw the eye.
    ///
    /// Exactly one status earns it. The drawer's job in one glance is to answer
    /// "which of these can I try?", and eight identically-styled capsules make
    /// a reader parse all eight to find out. Tinting more than one would put
    /// the question back.
    var isProminent: Bool { self == .readyToTest }
}

/// A module the platform intends to host.
///
/// Static catalog data. `status` describes what a person can do with the
/// cartridge in this build and carries no runtime behaviour; `workspace`
/// describes whether *this app* has a screen for it. The two are no longer
/// independent — a cartridge with nothing to open cannot be ready to test, and
/// `CartridgeCatalogTests` pins that correspondence in both directions.
struct Cartridge: Identifiable, Equatable {
    let id: String
    let name: String
    let summary: String
    let status: CartridgeStatus
    /// Where this module is defined, so the drawer can cite a real source.
    let specPath: String
    /// The workspace this app can open for the cartridge, or `nil` if the app
    /// has no screen for it — in which case its drawer row stays informational,
    /// exactly as every row was before workspaces existed.
    var workspace: CartridgeWorkspace? = nil
}

extension Cartridge {
    /// Names and ordering follow the module specs in docs/modules/.
    ///
    /// **`status` is taken from evidence, not from the specs.** Each value
    /// below is justified by what the Tower actually serves and what this build
    /// actually implements, verified against a live Tower and recorded in
    /// `docs/agent-handoffs/MAC-MEGA-INTEGRATION-FINAL.md`. Nothing here is
    /// aspirational copy, and nothing here is a roadmap position any more —
    /// a spec saying "future" no longer makes a badge say it.
    static let catalog: [Cartridge] = [
        // `.readyToTest`, and it is the least arguable entry in the catalog:
        // this is the **only module the Tower is actually running**. A live
        // Tower reports `module_state: "active"`, `module_id: "experimental-cv"`,
        // and its chosen experiment's answer arrives in `frame_result` on every
        // frame — now rendered in this cartridge's own workspace rather than
        // only on Home.
        //
        // What a person cannot do is choose, start or stop an experiment: the
        // Tower picks one at its own startup and offers no message to ask for
        // another. The workspace says exactly that. "Ready to test" is still
        // the honest badge — there is something here to try — and the nuance
        // belongs on the screen, not in a capsule.
        Cartridge(
            id: "experimental-cv",
            name: "Experimental CV Lab",
            summary: "A sandbox for running and comparing vision models on the glasses feed.",
            status: .readyToTest,
            specPath: "docs/modules/EXPERIMENTAL-CV.md",
            workspace: .experimentalCV
        ),
        // The summary used to read "Remembers where objects were last seen",
        // which is the claim the cartridge's own contract spends four sections
        // refusing. `spatial_ref` is null in every payload it can produce and
        // is actively nulled on read: nothing here knows where anything is.
        // The drawer row is the first sentence a person reads about this
        // module, so it is the first place the overclaim had to go.
        //
        // `.readyToTest`. The old comment here said "the Tower serving two
        // read-only routes is not the module runtime arriving" — which was
        // about the roadmap, and the roadmap is no longer what this field
        // answers. Two live HTTP routes, a real decoder pinned against a real
        // Tower's bytes, and a workspace that queries them is precisely
        // something a person can try.
        //
        // Note the current Tower answers 404 (`no object memory root is
        // configured`) and the workspace renders that truthfully, by test. That
        // is a fact about one Tower's configuration, not about this build —
        // see `CartridgeStatus`'s doc on why the badge does not track it.
        Cartridge(
            id: "object-memory",
            name: "Object Memory",
            summary: "Records that a category of thing was visible to the camera, and when.",
            status: .readyToTest,
            specPath: "docs/modules/OBJECT-MEMORY.md",
            workspace: .objectMemory
        ),
        Cartridge(
            id: "visual-qa",
            name: "Visual Q&A",
            summary: "Answers questions about what the camera can see, including reading text.",
            status: .notBuilt,
            specPath: "docs/modules/VISUAL-QA.md"
        ),
        // `.readyToTest`, and this is the entry whose old value was most
        // wrong. It read "Future" while being **the only cartridge in this
        // catalog with physical evidence behind it**: it is the one contract
        // `GET /cartridges` offers, and a clean walk on 2026-08-26 drew 7,086
        // points across 28 segments live, with a 92.2% geometry cache hit rate
        // and zero transport failures.
        //
        // Named "World Builder" here, while the spec file it cites is titled
        // "World Build". Same module; the product surface and the Tower work
        // both use the longer name, and two names for one thing on screen would
        // be worse than the small mismatch with the document title.
        Cartridge(
            id: "world-build",
            name: "World Builder",
            summary: "Reconstructs a persistent spatial model of a space over time.",
            status: .readyToTest,
            specPath: "docs/modules/WORLD-BUILD.md",
            workspace: .worldBuilder
        ),
        Cartridge(
            id: "accessibility",
            name: "Accessibility",
            summary: "Experimental assistive descriptions. Not a safety or navigation device.",
            status: .notBuilt,
            specPath: "docs/modules/ACCESSIBILITY.md"
        ),
        Cartridge(
            id: "environmental-memory",
            name: "Environmental Memory",
            summary: "Search the physical world you have already walked through.",
            status: .notBuilt,
            specPath: "docs/modules/ENVIRONMENTAL-MEMORY.md"
        ),
        // The two entries below are narrower first versions of modules already
        // in this catalog, not new ambitions. Environmental Memory's own spec
        // asks a first version to "choose one constrained memory type, such as
        // […] searchable OCR history", and Object Memory's pipeline produces
        // the live tracks Scene Understanding reads. Each cites its own concept
        // seed, which says what it takes from its parent and what it leaves
        // behind.
        //
        // Both were `.awaitingTower`, on the grounds that the Tower listed them
        // under `not_offered` — Document Memory having no typed contract, and
        // Scene Understanding persisting nothing so the channel that reads
        // persisted state had nothing to read.
        //
        // **That Tower decision was made on 2026-08-27 and both premises are
        // gone.** `not_offered` is now `[]`. Document Memory declares
        // `document_memory.status/2026-09-07` on the socket *and*
        // `document_memory.library/2026-09-07` under a new `http_contracts`
        // block. Scene Understanding declares `scene_understanding.live/…` —
        // and the persistence objection turned out to be answered by the
        // result *type*: `live` rather than `status`, where the payload IS the
        // answer rather than progress toward a stored one. A cartridge that
        // persists nothing can still say what it can see right now.
        //
        // Both now have clients that decode those contracts, so both are
        // `.readyToTest` on the same terms as the other three: this build
        // implements them and can show real Tower data. Neither badge promises
        // the Tower you happen to be connected to is configured for it — that
        // is resolved per-connection by `CartridgeAvailability.resolve` and
        // rendered inside the workspace, which is the separation this whole
        // type exists to keep.
        //
        // What has NOT changed is the distinction from `.notBuilt`, which is
        // still load-bearing for the three entries above that have no Tower
        // code anywhere. Those are a backend away, not a decision away.
        Cartridge(
            id: "document-memory",
            name: "Document Memory",
            summary: "Documents you passed, kept as text and summaries rather than as pictures.",
            status: .readyToTest,
            specPath: "docs/modules/DOCUMENT-MEMORY.md",
            workspace: .documentMemory
        ),
        Cartridge(
            id: "scene-understanding",
            name: "Scene Understanding",
            summary: "An anonymous read of how many people and objects the camera can see, and roughly where.",
            status: .readyToTest,
            specPath: "docs/modules/SCENE-UNDERSTANDING.md",
            workspace: .sceneUnderstanding
        ),
    ]
}
