"""The Tower -> iOS structured cartridge result channel.

A READ-ONLY reporting surface over state that has already been produced
elsewhere. Nothing in this package runs a cartridge, drives a module,
touches a frame, or writes anything. It answers one question -- "what does
the Tower currently know about cartridge X?" -- from files on disk, or,
since 2026-08-27, from a live session object it is handed and may only
read.

That restraint is the architecture, not an accident of scope. The obvious
alternative was to register World Builder as a live production module and
publish its results as they were computed; `TOWER-TO-IOS.md` 6.1 lists the
four blockers that stop, and every one of them is real. But 6.1 also
conflates two things that turn out to be separable:

    a World Builder TRANSPORT        <- achievable now, this package
    World Builder as a LIVE MODULE   <- blocked at V1.0/V1.1

The blockers are all properties of the second. `process()` being
synchronous and `bytes`-only, `ExperimentResult` being scalar-shaped,
`_build_cv_module` being a registry of one, `LIFECYCLE_TIMEOUT_S` being
too short for a build -- none of them bear on a reader that never joins
the frame path at all.

**Scene Understanding is the case that does not fit that sentence**, and
it is worth saying how it was resolved rather than leaving the header
above quietly false. It persists nothing by design -- enforced, not
intended -- so there is no file for a reader to read, and the
journal-follower pattern that gave Object Memory a route is unavailable to
it. It is nonetheless served here, because the separation that matters
survived: the LIVE part is `tower/scene/live.py`, which owns a worker
thread and a model and is constructed by `tower/cartridge_runtime.py`;
this package is handed the resulting session object and may only call
`status()` and `latest()` on it. `test_the_result_channel_never_writes`
forbids a call named `observe` or `build` anywhere under `tower/results/`
and so mechanically prevents that handle from becoming a second frame
path.
"""

import time

from tower.results.contracts import (
    CARTRIDGE_DOCUMENT_MEMORY,
    CARTRIDGE_EXPERIMENTAL_CV,
    CARTRIDGE_SCENE_UNDERSTANDING,
    CARTRIDGE_WORLD_BUILDER,
    EXPERIMENTAL_CV_STATUS_CONTRACT,
    RESULT_TYPE_LIVE,
    RESULT_TYPE_STATUS,
    WORLD_BUILDER_STATUS_CONTRACT,
)
from tower.results.envelope import Snapshot, compute_revision
from tower.results.publisher import ResultHub


def make_snapshot_for(
    world_root,
    clock=time.time,
    *,
    document_root=None,
    scene_source=None,
    document_source=None,
    cv_lab=None,
    document_unavailable_reason=None,
):
    """A callable turning (cartridge, result_type, world, session) into a Snapshot.

    Built once and handed to the hub, so the hub itself imports no
    cartridge and can be tested against a stub. Producers are constructed
    lazily and reused, because the World Builder producer holds a small
    per-target cache that would be thrown away on every poll otherwise.

    `scene_source` and `document_source` are LIVE SESSION OBJECTS rather
    than roots, and the asymmetry with `world_root` is the whole point of
    the header above: those two cartridges are not reading a file that
    another process wrote, they are reporting on work happening in this
    process. They are passed in rather than constructed here so this
    module still imports nothing that owns a thread or a model, and so a
    test can drive the whole channel with a stub that has never seen
    torch.

    `cv_lab` is the third of those and the same shape: the live Lab
    object or None. It is the first thing this channel reported that was
    not a file -- see `tower/results/experimental_cv.py` for why that is
    still read-only, and for the concurrency it introduces.

    All three are optional and default to None, which produces an
    explicit `unavailable` snapshot rather than an exception -- the same
    under-promise-on-omission rule `registry.declare` follows.
    """
    producers: dict = {}

    def snapshot_for(cartridge, result_type, world_id, session_id) -> Snapshot:
        if (
            cartridge == CARTRIDGE_EXPERIMENTAL_CV
            and result_type == RESULT_TYPE_STATUS
        ):
            if cv_lab is None:
                # Function-local import, the same pattern the World
                # Builder producer uses below: `tower/results/__init__.py`
                # is in the cartridge-blind core list, and a module-level
                # import would put a cartridge in the shared surface.
                from tower.results.experimental_cv import unavailable_payload

                payload = unavailable_payload(
                    "this Tower is running without a CV Lab module"
                )
                return Snapshot(payload=payload, revision=compute_revision(payload))
            producer = producers.get(cartridge)
            if producer is None:
                from tower.results.experimental_cv import (
                    ExperimentalCVStatusProducer,
                )

                producer = ExperimentalCVStatusProducer(cv_lab)
                producers[cartridge] = producer
            return producer.snapshot()
        if cartridge == CARTRIDGE_WORLD_BUILDER and result_type == RESULT_TYPE_STATUS:
            if world_root is None:
                return _unavailable_snapshot(
                    "no world root is configured on this Tower"
                )
            producer = producers.get(cartridge)
            if producer is None:
                from tower.results.world_builder import WorldBuilderStatusProducer

                producer = WorldBuilderStatusProducer(world_root, clock)
                producers[cartridge] = producer
            return producer.snapshot(world_id, session_id)

        if (
            cartridge == CARTRIDGE_SCENE_UNDERSTANDING
            and result_type == RESULT_TYPE_LIVE
        ):
            if scene_source is None:
                return _unavailable_snapshot(
                    "Scene Understanding is not enabled on this Tower"
                )
            from tower.results.scene_understanding import VOLATILE_PATHS, live_payload

            state, _observed_at, _computed_at = scene_source.latest()
            payload = live_payload(scene_source.status(), state)
            return Snapshot(
                payload=payload,
                revision=compute_revision(payload, VOLATILE_PATHS),
                volatile_fields=VOLATILE_PATHS,
            )

        if (
            cartridge == CARTRIDGE_DOCUMENT_MEMORY
            and result_type == RESULT_TYPE_STATUS
        ):
            if document_root is None:
                return _unavailable_snapshot(
                    "no document root is configured on this Tower"
                )
            from tower.results.document_memory import (
                STATUS_VOLATILE_PATHS,
                DocumentStatusProducer,
            )

            producer = producers.get(cartridge)
            if producer is None:
                producer = DocumentStatusProducer(
                    document_root,
                    document_source,
                    clock=clock,
                    unavailable_reason=document_unavailable_reason,
                )
                producers[cartridge] = producer
            payload = producer.payload()
            return Snapshot(
                payload=payload,
                revision=compute_revision(payload, STATUS_VOLATILE_PATHS),
                volatile_fields=STATUS_VOLATILE_PATHS,
            )

        # Unreachable through the wire: registry.find_offer refuses an
        # unknown pair before a subscription is ever created. Present so
        # that a future producer added to the registry and forgotten here
        # fails as an explicit unavailable rather than a KeyError inside
        # the poll loop.
        return _unavailable_snapshot(
            f"no producer is wired for {cartridge}/{result_type}"
        )

    return snapshot_for


def _unavailable_snapshot(reason: str) -> Snapshot:
    """The Tower cannot serve this cartridge at all.

    Distinct from "there is nothing to show yet": this is a Tower
    limitation, so it projects to iOS's `.unsupported`, which tells a
    person the Tower cannot do this rather than inviting them to wait.
    """
    payload = {
        "lifecycle": {
            "state": "unavailable",
            "evidence": "nothing to read",
            "reason": reason,
            "build_in_progress": None,
            "build_in_progress_unavailable_reason": reason,
        },
        "model_state": "unsupported",
        "model_state_reason": reason,
        "world_snapshot": None,
    }
    return Snapshot(payload=payload, revision=compute_revision(payload))


def build_hub(
    world_root,
    clock=time.time,
    *,
    document_root=None,
    scene_source=None,
    document_source=None,
    cv_lab=None,
    document_unavailable_reason=None,
) -> ResultHub:
    return ResultHub(
        make_snapshot_for(
            world_root,
            clock,
            document_root=document_root,
            scene_source=scene_source,
            document_source=document_source,
            cv_lab=cv_lab,
            document_unavailable_reason=document_unavailable_reason,
        ),
        clock=clock,
    )


__all__ = [
    "CARTRIDGE_EXPERIMENTAL_CV",
    "EXPERIMENTAL_CV_STATUS_CONTRACT",
    "CARTRIDGE_WORLD_BUILDER",
    "RESULT_TYPE_STATUS",
    "WORLD_BUILDER_STATUS_CONTRACT",
    "ResultHub",
    "build_hub",
    "make_snapshot_for",
]
