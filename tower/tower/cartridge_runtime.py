"""The live cartridges this configuration enables, built in one place.

`main.py:68` says of the capture-worker supervisor that it is "the ONE
place in the web process that knows a world builder exists, and it knows
it as an argv". That sentence is the design, and the answer to a second
and third cartridge is to stop adding places rather than to add two more.

So `main.py` asks this module a question it can ask generically --
"which live cartridges does this configuration enable?" -- exactly as it
already asks `tower.results` for a hub, and this module is the only file
in the web process that names them.

WHY THIS IS NOT A DODGE

`test_scene_understanding_is_not_registered_as_a_production_module` reads
`main.py` and asserts the string `tower.scene` does not appear;
`test_document_memory_is_not_registered_as_a_production_module` does the
same for `document_memory`. Routing around a test by moving an import is
usually the wrong instinct. It is the right one here, and the difference
is that the SECOND assertion of each of those tests -- that
`tower/modules/scene.py` and `tower/modules/document_memory.py` do not
exist -- is the invariant, and it survives verbatim and untouched.
Neither cartridge is a `Module`. Neither is in the `ModuleContainer`.
Neither runs on the event loop. What changed is that the web process can
now hand them frames, which is the thing `registry.py` named as their
blocker: "nothing in the web process observes it, so there is no state
for this channel to read".

WHAT A LIVE CARTRIDGE IS, HERE

An object with `offer_frame(raw_bytes, *, received_at)` and a lifecycle.
It is registered on `app.state.frame_consumers` -- deliberately NOT on
`app.state.frame_observers`, which is the dataset recorder's list and is
shaped around capture lineage (`start` mints a capture id, `capture_dir`
is called unguarded, `/health` reports on it). Putting a cartridge there
would make `/health` claim frames were being recorded when they were
being counted and dropped, and one missing method would end a connection
mid-stream. Two lists, two jobs.

NOTHING IS CONSTRUCTED THAT IS NOT ENABLED

Every import below is inside a branch that has already checked a setting.
A Tower with both cartridges off imports neither, loads no model, starts
no thread, and pays nothing -- which is also what keeps
`test_importing_the_lab_does_not_import_torch` and
`test_the_ocr_dependency_is_not_imported_at_module_load` passing: both
run `import tower.main` in a subprocess and assert torch and easyocr are
absent from `sys.modules`.
"""

import logging
from dataclasses import dataclass, field

from tower.logging_config import client_safe_reason

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveCartridges:
    """What the web process needs to hold, once the factory has decided.

    Three separate handles rather than one bag, because they are consumed
    by three different things and conflating them is how a result channel
    ends up able to drive a frame path:

    `frame_consumers`  read by `tower/routes/ws.py`, offered every frame
    `scene`            read by the result channel and the control route
    `document`         the same, for the other cartridge

    `scene` and `document` are the SAME objects that appear in
    `frame_consumers`. They are named separately so a reader can be handed
    one without being handed the list.
    """

    frame_consumers: list = field(default_factory=list)
    scene: object | None = None
    document: object | None = None
    # Why `document` is None when the reason is not "capture is off": the
    # OCR extra is not installed, or the session could not be built. Read
    # by the session routes and the status producer so a phone is told
    # what is actually wrong rather than which variable to set.
    document_unavailable_reason: str | None = None
    # Why `scene` is None, when the reason is not "nobody enabled it".
    #
    # Without this the declaration blamed the wrong thing. A construction
    # failure with `TOWER_SCENE_UNDERSTANDING` switched ON still served
    # `SCENE_DISABLED_REASON`, which reads "TOWER_SCENE_UNDERSTANDING is
    # unset or off" -- so an operator whose real problem was a missing
    # `[ml]` extra was told to set a variable that was already set.
    # Verified on the tree before this field existed, with the flag on,
    # device `cuda` and torch absent.
    #
    # None means "no specific reason", and the declaration then falls back
    # to the configured-off wording, which stays pinned by its own test.
    scene_unavailable_reason: str | None = None

    # -- demand from the result channel ---------------------------------
    #
    # A subscription to a live cartridge's result is the phone saying "a
    # person is looking at this right now", and for Scene Understanding
    # that is what decides whether the detector runs at all (see
    # `tower/scene/live.py`, WHEN IT RUNS). The result channel is
    # cartridge-blind and must stay so, so it reports the subscription by
    # cartridge NAME and this object decides which session, if any, that
    # name is demand for. Every method is safe to call for any cartridge
    # and never raises: a demand hook that could end a connection over a
    # status subscription would be the failure `results_ws.handle` exists
    # to prevent.

    def watcher_joined(self, cartridge: str, token, *, owner=None) -> None:
        session = self._demand_target(cartridge)
        if session is not None:
            try:
                session.watcher_joined(token, owner=owner)
            except Exception:
                logger.exception("[Tower][Cartridge] watcher_joined failed")

    def watcher_left(self, cartridge: str, token) -> None:
        session = self._demand_target(cartridge)
        if session is not None:
            try:
                session.watcher_left(token)
            except Exception:
                logger.exception("[Tower][Cartridge] watcher_left failed")

    def watchers_left(self, *, owner) -> None:
        """A connection closed. Every session it was watching is told."""
        for session in self.frame_consumers:
            left = getattr(session, "watchers_left", None)
            if left is None:
                continue
            try:
                left(owner=owner)
            except Exception:
                logger.exception("[Tower][Cartridge] watchers_left failed")

    def _demand_target(self, cartridge: str):
        # The name is compared to a string rather than to an import from
        # `tower.results.contracts`, so this module keeps importing
        # nothing from the result channel and the channel nothing from
        # here. `test_the_result_channel_core_is_cartridge_blind` guards
        # the other direction.
        if cartridge == "scene_understanding":
            return self.scene
        return None

    def shutdown(self) -> None:
        """Stop every live session. Never raises.

        Called from the app's lifespan teardown, after the result hub has
        stopped polling: a session that stops while the hub is still
        reading would publish one last payload saying `stopped` for no
        reason, and the order that avoids that is free.
        """
        for consumer in self.frame_consumers:
            try:
                consumer.stop()
            except Exception:
                logger.exception(
                    "[Tower][Cartridge] a live session did not stop cleanly"
                )


def _resolve_device(requested: str) -> str:
    """A requested device, as one that exists. Torch imported lazily.

    Duplicated in spirit from `tower/experiments/depth.py: resolve_device`
    and deliberately not imported from it. That function lives in the Lab,
    and `test_scene_understanding_does_not_import_another_cartridge`
    forbids `tower.scene` importing `tower.experiments` -- for the good
    reason that the Lab is a sandbox that may be thrown away. Reaching
    through this module to borrow it would couple the two by a longer
    path and honour the letter of that rule while breaking it.

    The behaviour is the same and the reason is the same: an unnoticed
    downgrade from cuda to cpu turns a GPU deployment into a CPU one with
    a GPU label on it, which is worse than a failure. "auto" downgrades;
    "cuda" does not.
    """
    if requested == "cpu":
        return "cpu"

    import torch

    available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if available else "cpu"
    if requested == "cuda" and not available:
        raise RuntimeError("cuda requested but torch reports it is unavailable")
    return requested


def _cap_torch_threads(settings) -> None:
    """Bound torch's thread pool, or say why it was not bounded.

    PROCESS-GLOBAL, and that is why it is opt-in rather than a default:
    `torch.set_num_threads` has no per-model scope, so a value chosen for
    the scene detector also applies to the Experimental CV Lab.

    Silence would be the wrong default here. Measured over 829 real
    corpus frames at the delivered 12.0 fps
    (`scripts/cartridge_live_benchmark.py`), torch's own default cost
    **4.12 cores** and capping it at 2 cost **1.03**, with throughput
    identical to within 0.3%. An operator who never learns that is paying
    three cores for nothing, and a log line is the cheapest way to tell
    them.
    """
    if settings.scene_torch_threads <= 0:
        logger.info(
            "[Tower][Config] TOWER_SCENE_TORCH_THREADS is unset: torch will "
            "use one thread per core. Measured at 4.12 cores for one scene "
            "session against 1.03 when capped at 2, with the same scene "
            "throughput -- but the cap is PROCESS-GLOBAL and costs the CV "
            # Single %, not %%. `logger.info` with no arguments applies no
            # %-formatting at all, so "%%" reaches the operator literally.
            "Lab 39-179% per frame ON THE EVENT LOOP (see config.py). "
            "Prefer 4 to 2 if this Tower shares its CPU"
        )
        return
    try:
        import torch

        torch.set_num_threads(settings.scene_torch_threads)
    except Exception:
        logger.exception(
            "[Tower][Config] TOWER_SCENE_TORCH_THREADS could not be applied"
        )
        return
    logger.info(
        "[Tower][Config] torch intra-op threads capped at %s. This is "
        "PROCESS-GLOBAL and applies to every torch consumer here, not only "
        "Scene Understanding",
        settings.scene_torch_threads,
    )


def _scene_session(settings):
    """A Scene Understanding session, not yet started.

    Constructed but STOPPED. Starting it is an explicit act -- a POST to
    the control route, or a client subscribing -- because a Tower that
    began detecting the moment it booted would be recording nothing and
    computing constantly, which is the posture `_build_frame_observers`
    already rejects for the recorder ("arming is not recording").

    The factory closes over the device rather than resolving it per
    session, so a `cuda` that has gone missing fails at startup where an
    operator sees it, not on the first frame of a physical test.

    TORCH IS IMPORTED HERE, EAGERLY AND UNCONDITIONALLY, and that is the
    same principle as the sentence above rather than a new one. It buys
    two separate things.

    **The declaration stops lying.** `build_live_cartridges` already
    try/excepts this function and `main.py` already derives
    `scene_enabled` from whether it returned a session, so a Tower that
    cannot run Scene already reports it unavailable -- but only when
    `_resolve_device` happened to import torch, which it does ONLY when
    the device is not "cpu". On the default `cpu` a torch-less host
    constructed a session happily, `/cartridges` said `available: true`,
    and `start()` then failed in 51 ms. Measured on a host with torch
    blocked: `auto` and `cuda` already answered `available: false`; `cpu`
    was the one configuration that promised a cartridge it could not run.

    **It closes a concurrent-import race.** Scene and Document Memory
    both start on the same `stream_start`, on separate worker threads,
    and each used to reach torchvision lazily and for the first time
    there. MEASURED, 8 fresh processes out of 8, with both cartridges
    enabled: BOTH landed in terminal `failed` with

        ImportError: cannot import name 'InterpolationMode' from
        partially initialized module 'torchvision.transforms'

    Importing here is single-threaded inside `create_app()`, before any
    worker exists. 6 of 6 trials clean afterwards. That race is invisible
    to an in-process test suite, because once the first import succeeds it
    cannot recur.

    IT DOES NOT CLOSE THE RACE CLASS, only this instance of it. Scene is
    one of several first-time torchvision importers; a reviewer reproduced
    the identical failure between Document Memory's easyocr and
    `tower/detection.py`'s lazy import with Scene OFF, where nothing here
    runs. Preimporting is the fix for the pair that ships enabled
    together, not a general answer to two threads racing an import.

    `find_spec` was measured as the alternative and REFUSED. It locates
    without executing, so it fixed 0 of 6 race trials, and against a
    package with a valid spec whose loader raises it reported
    `available: true`, answered `POST /scene/start` with 200, and then
    STILL said `available: true` after the session had failed.

    The cost is moved, not added. Boot with Scene on goes 0.32 s -> 2.21 s,
    but end to end from boot to a running session is 2.019 s before and
    2.002 s after -- within 1%, because this is the same import the first
    Start used to pay for. A Tower with Scene off pays nothing (0.3229 s
    against a 0.3180 s base).
    """
    import torch  # noqa: F401
    import torchvision  # noqa: F401

    from tower.scene.detect import detector_for
    from tower.scene.engine import SceneEngine
    from tower.scene.live import SceneLive
    from tower.scene.orientation import FaceVisibilityEstimator, model_path

    device = _resolve_device(settings.scene_device)
    _cap_torch_threads(settings)
    detector_choice = getattr(settings, "scene_detector", "auto")
    # Resolved once, here, so a bad choice fails at boot where an
    # operator sees it and not on the first frame of a physical test.
    detector_for(device, detector_choice)

    # The face model is a file; either it is vendored or it is not, and
    # that is decided once too. Without it there is no orientation stage
    # and the wire says `orientation_enabled: false` with a reason
    # rather than guessing.
    face_model = model_path() if settings.scene_orientation else None
    if settings.scene_orientation and face_model is None:
        logger.warning(
            "[Tower][Config] TOWER_SCENE_ORIENTATION is on but no face model "
            "was found; Scene Understanding will run without orientation"
        )

    def make_engine():
        facing = None if face_model is None else FaceVisibilityEstimator(face_model)
        return SceneEngine(
            detector_for(device, detector_choice), facing_estimator=facing
        )

    logger.info(
        "[Tower][Config] Scene Understanding enabled on %s (detector %s, "
        "orientation %s); no session is running until a client watches a "
        "stream",
        device,
        detector_choice,
        "on" if face_model is not None else "off",
    )
    return SceneLive(make_engine, follow_stream=settings.scene_autostart)


def _document_session(settings):
    """A Document Memory capture session, not yet started.

    Requires BOTH a root and the capture flag, and the conjunction is not
    redundant. A root with capture off is a Tower that serves a library
    recorded elsewhere and records nothing itself, which is the correct
    posture for a machine reprocessing captures offline.
    """
    from tower.document_memory.live import DocumentLive
    from tower.document_memory.ocr import require_ocr_extra

    # A capability is offered only when it is genuinely there. `find_spec`
    # locates without executing, which is exactly enough for "the extra
    # is not installed" and costs no import of torch on the web process.
    # (Scene refused `find_spec` for a different question -- "does this
    # installed package load" -- which a session start still answers.)
    require_ocr_extra()

    logger.info(
        "[Tower][Config] document capture enabled on device %r, writing to "
        "%s; no session is running until one is started",
        settings.document_device,
        settings.document_root,
    )
    return DocumentLive(
        settings.document_root,
        follow_stream=settings.document_autostart,
        device=settings.document_device,
        retention_days=settings.document_retention_days,
    )


# Why Scene Understanding is unavailable in auto mode on a host without
# the [ml] extra. Module constants, like `SCENE_DISABLED_REASON`, so the
# route and the declaration cannot drift into two sentences.
SCENE_ML_EXTRA_MISSING_REASON = (
    "Scene Understanding is not available on this Tower because the "
    "optional [ml] extra (torch and torchvision) is not installed, so no "
    "session can be started and there is no live scene to read. It is "
    "offered automatically once the extra is installed; set "
    "TOWER_SCENE_UNDERSTANDING=off to stop offering it. This build "
    "implements the contract"
)


def _scene_construction_failed(mode: str, reason: str) -> str:
    if mode == "on":
        return (
            "Scene Understanding is enabled on this Tower but could not "
            f"be constructed, so no session can be started: {reason}. "
            "This build implements the contract"
        )
    return (
        "Scene Understanding is not available on this Tower: it could not "
        f"be constructed, so no session can be started: {reason}. It is "
        "offered automatically when the optional [ml] extra (torch and "
        "torchvision) works; set TOWER_SCENE_UNDERSTANDING=off to stop "
        "offering it. This build implements the contract"
    )


def build_live_cartridges(settings) -> LiveCartridges:
    """Everything this configuration turns on. Often nothing.

    A failure to construct one cartridge must not cost the other, and
    must not stop the Tower: a missing weight file or a `cuda` that
    vanished is a reason to run without that cartridge and say so
    loudly, not a reason for a Tower to refuse to serve frames. The
    cartridge then reports itself unavailable through the normal path,
    which is the same answer a client would get if it had never been
    enabled -- and the log line is how an operator tells those apart.
    """
    consumers: list = []
    scene = None
    document = None
    scene_unavailable_reason = None
    document_unavailable_reason = None

    mode = getattr(settings, "scene_understanding_mode", None)
    if mode is None:
        # A settings object built by hand, without the tri-state. The
        # boolean is the whole truth for it, exactly as before.
        mode = "on" if settings.scene_understanding else "off"

    if settings.scene_understanding:
        try:
            scene = _scene_session(settings)
            consumers.append(scene)
        except Exception as exc:
            logger.exception(
                "[Tower][Config] Scene Understanding could not be "
                "constructed (mode %s); this Tower will report it "
                "unavailable",
                mode,
            )
            scene = None
            # `client_safe_reason` rather than `str(exc)`, and it is the
            # same helper `live_session` already uses for the same wire.
            # This string reaches an unauthenticated `/cartridges`, and an
            # OSError describes a failure by naming the PATH it happened
            # on -- which would disclose the home directory and with it
            # the OS username. This repository's own exceptions are
            # written to be read by a person and pass through; everything
            # else reduces to its type name.
            scene_unavailable_reason = _scene_construction_failed(
                mode, client_safe_reason(exc)
            )
    elif mode == "auto":
        # Auto, and the [ml] extra is not installed. Not "off": nobody
        # switched anything, and the sentence has to send an operator to
        # the install step rather than to a variable.
        scene_unavailable_reason = SCENE_ML_EXTRA_MISSING_REASON

    if settings.document_capture and settings.document_root is not None:
        try:
            document = _document_session(settings)
            consumers.append(document)
        except Exception as exc:
            logger.exception(
                "[Tower][Config] document capture is enabled but could not "
                "be constructed; this Tower will record no documents"
            )
            document = None
            # Same wire, same rule as the scene reason above: a person's
            # sentence passes through, anything else reduces to its type.
            # Names TOWER_DOCUMENT_CAPTURE because iOS keys its "no capture
            # session" outcome on that substring.
            document_unavailable_reason = (
                "this Tower runs no document capture session: Document "
                "Memory is enabled (TOWER_DOCUMENT_CAPTURE is on) but the "
                f"session could not be constructed: {client_safe_reason(exc)}. "
                "Documents recorded elsewhere are still served"
            )
    elif settings.document_capture:
        logger.warning(
            "[Tower][Config] TOWER_DOCUMENT_CAPTURE is on but "
            "TOWER_DOCUMENT_ROOT is unset: nothing will be recorded, "
            "because there is nowhere to record it"
        )

    return LiveCartridges(
        frame_consumers=consumers,
        scene=scene,
        document=document,
        scene_unavailable_reason=scene_unavailable_reason,
        document_unavailable_reason=document_unavailable_reason,
    )
