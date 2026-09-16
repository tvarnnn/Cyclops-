import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tower.capture import DEFAULT_MAX_IDLE_POLLS, CaptureRecorder
from tower.capture_workers import CaptureWorkerSupervisor, WorkerSpec
from tower.cartridge_runtime import build_live_cartridges
from tower.cartridge_session import CartridgeSession, STOP_POLICY_REQUEST
from tower.config import KNOWN_VERIFIERS, TOWER_ROOT, Settings, get_settings
from tower.cv_lab.preview import PreviewPolicy
from tower.experiments import ExperimentSettings
from tower.logging_config import configure_logging
from tower.modules.base import Module
from tower.modules.container import ModuleContainer
from tower.modules.experimental_cv import ExperimentalCVModule
from tower.results import build_hub
from tower.results.contracts import CARTRIDGE_OBJECT_MEMORY, CARTRIDGE_WORLD_BUILDER
from tower.results.object_memory import (
    build_face_filter,
    keyframe_store_from_root,
    recorded_classes_for,
)
from tower.routes import (
    cartridges,
    cv_lab,
    cv_lab_preview,
    documents,
    geometry,
    health,
    observations,
    scene,
    sessions,
    ws,
)
from tower.session import ConnectionTracker

logger = logging.getLogger(__name__)

# TOWER_ROOT moved to `tower/config.py` and is imported above. It is the
# directory holding `scripts/`, `models/` and, by default, `data/`, and
# it is resolved from a file rather than from the working directory
# because a builder started with the wrong CWD finds no YuNet weights
# (`world_builder/redaction.py` resolves them relatively) and silently
# records its redaction as `none`. It lives beside the settings now
# because a DEFAULT PATH is a setting, and two modules resolving the
# same root independently is how the producer and the reader came to
# disagree about where observations live.

# The worker names this Tower can attach to a capture. Strings, and
# strings only: `capture_workers.py` addresses a spec by name and knows
# nothing else about it, and `test_the_capture_worker_supervisor_is_
# cartridge_blind` is what keeps that true.
WORLD_BUILD_WORKER = "world-build-session"
OBJECT_MEMORY_WORKER = "object-memory-session"


def _build_cv_module(settings: Settings, connection_count=None) -> Module:
    """The one module slot.

    There used to be a branch here selecting a different Module subclass
    for the depth experiment, because that experiment holds a model.
    Experiment state now lives behind the Experiment protocol, so the
    module is the same one whichever experiment is selected -- which is
    what the module doc always said: one Lab slot, many experiments.

    `settings.cv_experiment` is now the STARTUP DEFAULT and nothing more.
    It is what this Tower arms at boot so that a client which knows
    nothing about the CV Lab still receives a `frame_result` for every
    frame, exactly as before; a client that does know sends
    `cv_lab_start` and the environment variable stops mattering until the
    next restart. That is the whole of "remove the product dependence on
    TOWER_CV_EXPERIMENT": it survives as a developer default, not as the
    only way to choose.

    `connection_count` lets the Lab report how many clients are attached,
    which is what turns "I pressed Start and nothing happened" from a
    guess into a reading.
    """
    return ExperimentalCVModule(
        settings.cv_experiment,
        ExperimentSettings(
            device=settings.cv_device,
            torch_threads=settings.cv_torch_threads,
        ),
        connection_count=connection_count,
        # Whether this Tower draws anything at all is an operator's
        # decision, so it arrives from `Settings` rather than being a
        # default the Lab picked for itself. See `Settings.cv_preview`.
        preview=PreviewPolicy.from_settings(settings),
    )


def _build_frame_observers(settings: Settings) -> list:
    """Register the dataset recorder, or nothing at all.

    A LIST because ws.py reads a list -- more than one consumer may
    eventually want raw frames, and a singleton would force the second
    one to displace the first.

    Arming is not recording. A configured root creates no directory and
    writes no byte until a `stream_start` arrives, so this stays an
    Explicit Dataset-Recording Session under 06-PRIVACY-DATA.md rather
    than becoming incidental capture. Unset by default, which is why
    every Tower that has ever run recorded nothing.
    """
    if settings.capture_root is None:
        return []
    return [CaptureRecorder(settings.capture_root)]


def _world_build_spec(settings: Settings, gate=None) -> WorkerSpec | None:
    """The builder that follows a capture WHILE World Builder is active, or nothing.

    This function and its neighbour are the ONLY places in the web
    process that know a world builder and an object memory exist, and
    they know them as an argv -- a script path and some flags -- not as
    an import. That is deliberate and it is load-bearing:
    `test_shared_code_does_not_import_a_cartridge` forbids transport,
    config and the module system from importing a cartridge, on the
    grounds that "the next cartridge inherits its assumptions". A command
    line inherits nothing, and `CaptureWorkerSupervisor` stays a thing
    that runs processes rather than a thing that builds worlds.

    The web process therefore still does not build. It supervises a child
    that does, which is what keeps an expensive rebuild off the frame
    path.

    GATED since 2026-09-06, like the object-memory producer, and for a
    resource reason rather than a privacy one. Ungated, a builder -- with
    its background global solves at all-cores-but-two -- attached to every
    capture on the Tower, including a CV Lab camera session that never
    asked for a world, and the only way to run CV Lab without it was
    `TOWER_WORLD_AUTOBUILD=false` and a restart. The gate is the
    `world_builder` cartridge session: the phone opens it when the World
    Builder workspace is on screen and closes it when it leaves.
    `TOWER_WORLD_AUTOBUILD` keeps its meaning -- whether a builder may run
    at all -- and stops being the way to switch cartridges.
    """
    if settings.world_root is None or not settings.world_autobuild:
        return None

    register = ("--register",) if settings.world_register else ()
    solve = ("--solve",) if settings.world_solve else ()
    # Dense reconstruction needs the solve: it is anchored to the global
    # solution's cameras and sparse points, and there is nothing to anchor
    # to without one. Asking for dense without solve is a configuration
    # mistake, and passing it anyway would make the builder refuse per
    # session rather than here, once.
    densify = ("--densify",) if (settings.world_densify and settings.world_solve) else ()

    return WorkerSpec(
        argv=(
            sys.executable,
            str(TOWER_ROOT / "scripts" / "world_build_session.py"),
            "--follow-capture",
            "{capture_dir}",
            "--root",
            settings.world_root,
            "--rebuild-every",
            str(settings.world_rebuild_every),
            # Place the segments once the walk ends. It is a flag on the
            # child's argv rather than anything this process does,
            # because the web process must keep knowing the builder only
            # as a command line -- and because registration is seconds of
            # solving that the frame path must never see.
            *register,
            # The global solver, same reasoning: a child of the builder,
            # never the web process, and seconds to minutes of solving the
            # frame path must never see.
            *solve,
            # And the dense stage, which is minutes of GPU and runs after the
            # world lock is released. Off unless TOWER_WORLD_DENSIFY says
            # otherwise; see Settings.world_densify for why the default is off.
            *densify,
            # So a producer whose Tower died without closing the manifest
            # stops following instead of polling that directory forever.
            # See DEFAULT_MAX_IDLE_POLLS: the bound has always existed and
            # neither spec passed it, so the invariant `follow()`'s
            # docstring promises was never actually armed in production.
            "--max-idle-polls",
            str(DEFAULT_MAX_IDLE_POLLS),
            # The half of the stop agreement that lives in the child: stdin
            # EOF is a SOFT stop (end the session, write the final build,
            # skip the final solve), and the builder installs handlers so a
            # console control event is a HARD one (wrap up now). Paired
            # with `stop_via_stdin` below, as the producer's is.
            "--stop-on-stdin-close",
        ),
        cwd=str(TOWER_ROOT),
        name=WORLD_BUILD_WORKER,
        gate=gate,
        stop_via_stdin=True,
        # Once asked, the builder ends within one final build; on a long
        # walk that is tens of seconds, not the shared 10 s default that
        # shot it mid-finalization. The grace is the bound, not the wait.
        stop_grace_seconds=30.0,
    )


def _observation_spec(settings: Settings, gate) -> WorkerSpec | None:
    """The producer that remembers objects, and the gate that permits it.

    GATED, unlike the builder, and the difference is a privacy decision
    rather than a symmetry oversight. A world is geometry; a memory of
    which objects were around is a record of a wearer's surroundings that
    outlives the walk. So it attaches only while a session is ACTIVE --
    something a person started, and can pause -- and a Tower that has
    just booted starts nothing.

    `{attach_mode}` is substituted by the supervisor. A producer attached
    at capture open is told it saw the whole capture; one attached in the
    middle is told it arrived late, and must not go back and remember the
    part of the walk that happened before anybody asked.
    """
    if settings.observation_root is None:
        return None

    return WorkerSpec(
        argv=(
            sys.executable,
            str(TOWER_ROOT / "scripts" / "object_memory_session.py"),
            "--follow-capture",
            "{capture_dir}",
            # The SAME value the read routes are given, from the same
            # settings object. There is no second default to drift.
            "--root",
            settings.observation_root,
            "--attach-mode",
            "{attach_mode}",
            "--device",
            settings.observation_device,
            "--retention-days",
            str(settings.observation_retention_days),
            "--verifier",
            settings.observation_verifier,
            "--verifier-device",
            settings.observation_verifier_device,
            # Whether each record gets a small filtered crop of its own.
            # Passed explicitly in both directions rather than relying on
            # the script's default, for the same reason every other value
            # here is: the read routes serve those keyframes from the
            # same root, and a Tower that serves what it did not ask to
            # be written is a Tower with two answers to one question.
            (
                "--keep-imagery"
                if settings.observation_keep_imagery
                else "--no-keep-imagery"
            ),
            # Same bound, same reason as the builder's. This producer
            # additionally writes an observation store, so an orphan that
            # polls forever is holding a root a later session will reuse.
            "--max-idle-polls",
            str(DEFAULT_MAX_IDLE_POLLS),
            # The half of the stop agreement that lives in the child. The
            # other half is `stop_via_stdin` below, and the two are here
            # together so neither can be set without the other.
            "--stop-on-stdin-close",
        ),
        cwd=str(TOWER_ROOT),
        name=OBJECT_MEMORY_WORKER,
        gate=gate,
        # Pause and Stop must let this producer finish writing what it
        # saw: `engine.release()` is what closes the sightings still
        # open, and the sighting still open when a wearer presses Stop is
        # the object they had been looking at longest. See
        # `capture_workers._ask_to_stop`.
        stop_via_stdin=True,
    )


def _build_capture_worker_supervisor(settings: Settings, gates: dict):
    """Decide what, if anything, follows a capture.

    `gates` maps a worker name to the predicate that says whether it may
    run. It is passed in rather than built here because a gate reads a
    session's state and a session needs a supervisor to attach through --
    the two are mutually referential, and the wiring point resolves that
    by handing over a closure that looks the session up when asked
    instead of capturing it at construction.
    """
    specs = [
        spec
        for spec in (
            _world_build_spec(settings, gates.get(WORLD_BUILD_WORKER)),
            _observation_spec(settings, gates.get(OBJECT_MEMORY_WORKER)),
        )
        if spec is not None
    ]
    return CaptureWorkerSupervisor(specs)


def _recorded_classes(settings: Settings) -> tuple[str, ...]:
    """The classes this Tower will actually write, as a tuple of strings.

    Resolved through the result-channel ADAPTER, which is the one module
    outside the cartridge allowed to import its policy. This file knows
    the world builder as an argv and knows object memory the same way;
    the only thing it takes from either is a tuple of strings.

    The route reads the answer off `app.state`. Neither the route nor the
    wiring point holds a policy, and neither can drift from what the
    producer was told, because both come from one `Settings`.
    """
    return recorded_classes_for(settings.observation_verifier)


def _open_capture_lookup(frame_observers):
    """What is recording right now, as `(capture_id, capture_dir)` or None.

    A closure over the observers rather than a reach into `app.state`,
    so `CartridgeSession` stays testable without an app -- and so a Tower
    with no recorder configured makes Start a no-op that waits, rather
    than an AttributeError on somebody's button.
    """

    def lookup():
        for observer in frame_observers:
            # `status` is a PROPERTY on CaptureRecorder, not a method --
            # `tower/routes/health.py` reads it the same way. Calling it
            # raises TypeError, which the session catches and reports as
            # "nothing is recording": a Start that silently waits forever
            # instead of attaching to the walk in progress.
            status = observer.status
            if status is None or not status.is_open:
                continue
            return status.capture_id, observer.capture_dir(status.capture_id)
        return None

    return lookup


def _log_effective_configuration(
    settings: Settings, supervisor: CaptureWorkerSupervisor
) -> None:
    """Say what this Tower will and will not do, at startup, once.

    Every setting that decides whether a cartridge works at all is
    optional and every one of them used to fail SILENTLY when unset: no
    capture root meant no recorder, no world root meant the result
    channel reported the cartridge unavailable, no observation root meant
    a producer wrote 64 records into a directory every HTTP request
    answered 404 about. On 2026-08-24 the first pair produced a Tower
    that answered every frame, recorded nothing anyone could find, and
    told the phone there was no world -- with nothing in the log saying
    why. On 2026-08-26 the third produced the same shape of surprise for
    a different cartridge. These lines fix that permanently.
    """
    if settings.capture_root is None:
        logger.warning(
            "[Tower][Config] TOWER_CAPTURE_ROOT is unset: NO frames will be "
            "recorded and /health will report capture: null"
        )
    else:
        logger.info(
            "[Tower][Config] capture root %s (armed; records nothing until "
            "stream_start)",
            settings.capture_root,
        )

    if settings.world_root is None:
        logger.warning(
            "[Tower][Config] TOWER_WORLD_ROOT is unset: World Builder is "
            "declared but reported unavailable, and iOS will show it as "
            "unsupported"
        )
    else:
        # ABSOLUTE, and whether it holds anything. `.env` carries the
        # relative `data/world_builder`; a Tower launched from any directory
        # but `tower/` resolves it somewhere else, answers `GET /worlds`
        # with an empty list, and the phone reads "no saved worlds yet".
        # The line that would have said so used to print the relative
        # string.
        resolved = Path(settings.world_root).resolve()
        worlds_dir = resolved / "worlds"
        if worlds_dir.is_dir():
            logger.info("[Tower][Config] world root %s (%s)", resolved, settings.world_root)
        else:
            logger.warning(
                "[Tower][Config] world root %s (%s) holds no worlds/ directory yet: "
                "GET /worlds will answer an empty list. If saved worlds were "
                "expected, check the directory this Tower was started from",
                resolved,
                settings.world_root,
            )

    if settings.scene_understanding:
        logger.info(
            "[Tower][Config] Scene Understanding is enabled (mode %s); it "
            "observes nothing until a client subscribes to the live scene "
            "while a stream is open, and persists nothing ever",
            settings.scene_understanding_mode,
        )
    elif settings.scene_understanding_mode == "auto":
        logger.info(
            "[Tower][Config] Scene Understanding is unavailable: the "
            "optional [ml] extra is not installed. The contract is "
            "declared and reported unavailable"
        )
    else:
        logger.info(
            "[Tower][Config] TOWER_SCENE_UNDERSTANDING is off: the contract "
            "is declared and reported unavailable"
        )

    if settings.document_root is None:
        logger.info(
            "[Tower][Config] TOWER_DOCUMENT_ENABLED is off: /documents/* "
            "will answer 404 and the contract is reported unavailable"
        )
    else:
        logger.info(
            "[Tower][Config] document root %s (capture %s, OCR device %s, "
            "retention %s days)",
            settings.document_root,
            "on" if settings.document_capture else "off",
            settings.document_device,
            settings.document_retention_days,
        )

    if settings.observation_root is None:
        logger.warning(
            "[Tower][Config] TOWER_OBSERVATION_ENABLED is off: nothing will "
            "produce object-memory observations and /object-memory/* will "
            "answer 404"
        )
    else:
        logger.info(
            "[Tower][Config] observation root %s (one path for the producer "
            "AND the read routes; the web process never writes or deletes "
            "observations). Producer device %s, retention %s days, verifier "
            "%s on %s -- recording %s.",
            settings.observation_root,
            settings.observation_device,
            settings.observation_retention_days,
            settings.observation_verifier,
            settings.observation_verifier_device,
            ", ".join(_recorded_classes(settings)),
        )
        # Said at startup rather than only in the producer's report,
        # because it is the difference between a memory that keeps its
        # picture for thirty days and one whose picture belongs to a
        # capture directory nobody promised to keep.
        logger.info(
            "[Tower][Config] owned keyframes %s (TOWER_OBSERVATION_KEEP_"
            "IMAGERY). On, each record keeps a small filtered crop under "
            "the observation root, deleted when the record expires. Off, "
            "no NEW crop is written -- crops already on disk are still "
            "served and still pruned with their records, because deleting "
            "them on a config change would be a deletion nobody asked "
            "for; scripts/object_query.py --purge-all removes them.",
            "on" if settings.observation_keep_imagery else "off",
        )
        # `auto` is a request, not an answer, and this process cannot
        # answer it: resolving it needs torch, which the web process
        # deliberately does not import. The producer resolves it and
        # prints the concrete device on its first line, which lands in
        # this console because workers inherit stdio -- so say where to
        # look rather than logging a word that is not a device.
        if "auto" in (
            settings.observation_device,
            settings.observation_verifier_device,
        ):
            logger.info(
                "[Tower][Config] a device above reads 'auto': it is resolved "
                "by the producer, which prints the device it actually got as "
                "its first line when a session starts."
            )
        # The verifier decides whether this Tower records two classes or
        # fourteen, so where the value came from is worth one line. It is
        # a default now rather than something an operator typed, and a
        # default that changed what a Tower remembers should not be
        # silent about being a default.
        if not os.environ.get("TOWER_OBSERVATION_VERIFIER", "").strip():
            logger.info(
                "[Tower][Config] TOWER_OBSERVATION_VERIFIER is unset, so the "
                "built-in default %r is in force. Set it to 'none' to record "
                "only the classes the detector is trusted on. A host that "
                "cannot load the weights records those two anyway -- the "
                "producer says so and does not stop.",
                settings.observation_verifier,
            )
        raw_verifier = os.environ.get("TOWER_OBSERVATION_VERIFIER", "")
        if raw_verifier.strip() and raw_verifier.strip().lower() != (
            settings.observation_verifier
        ):
            # Loud, because the alternative is a Tower that quietly
            # records less than the operator asked for.
            logger.warning(
                "[Tower][Config] TOWER_OBSERVATION_VERIFIER=%r is not a "
                "verifier this build has; running with %r instead. Known: "
                "%s",
                raw_verifier,
                settings.observation_verifier,
                ", ".join(KNOWN_VERIFIERS),
            )

    logger.info(
        "[Tower][Config] CV Lab startup default is %r on device %r with torch "
        "threads %r; a client may select another with cv_lab_start, no "
        "restart required",
        settings.cv_experiment,
        settings.cv_device,
        settings.cv_torch_threads,
    )

    attached = supervisor.worker_names()
    if WORLD_BUILD_WORKER in attached:
        logger.info(
            "[Tower][Config] a builder will be attached to a capture WHILE World "
            "Builder is active on the phone (POST /cartridges/%s/session/start; "
            "stop releases it), rebuilding every %s keyframes. It is stopped at "
            "startup.",
            CARTRIDGE_WORLD_BUILDER,
            settings.world_rebuild_every,
        )
    elif settings.world_root is not None:
        logger.warning(
            "[Tower][Config] TOWER_WORLD_AUTOBUILD is off: captures will be "
            "recorded but NOTHING will build a world from them"
        )

    if OBJECT_MEMORY_WORKER in attached:
        logger.info(
            "[Tower][Config] an object-memory producer will be attached to "
            "each capture WHILE A SESSION IS ACTIVE. It is stopped at "
            "startup: POST /cartridges/%s/session/start begins one.",
            CARTRIDGE_OBJECT_MEMORY,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # The result hub first: it holds a polling task, and stopping it
    # before the module container means no snapshot can be built against
    # an app that is half torn down. Guarded with getattr because most of
    # this repo's tests construct the app without running lifespan at all
    # (see the comment in create_app).
    hub = getattr(app.state, "result_hub", None)
    if hub is not None:
        await hub.shutdown()
    # Workers before the module container, and after the hub: a worker
    # holds a world's writer lock, and the honest order is to let it
    # finish and release before this process stops being able to report
    # what it did.
    supervisor = getattr(app.state, "capture_workers", None)
    if supervisor is not None:
        await asyncio.to_thread(supervisor.shutdown)
    # Live sessions after the hub, for the same reason as the workers: a
    # session stopped while the hub was still polling would publish one
    # last payload saying "stopped" that nobody asked for. Off-thread
    # because stopping a session joins a worker thread, and a bounded
    # join on the event loop is still a join on the event loop.
    live = getattr(app.state, "live_cartridges", None)
    if live is not None:
        await asyncio.to_thread(live.shutdown)
    # Before the container, and awaited. An experiment may be mid-load in
    # a background task; `ModuleContainer.shutdown()` would release the
    # Lab synchronously and leave that task to be cancelled by a loop
    # that is about to close. This is the one place with both a running
    # loop and the authority to wait for it.
    lab = getattr(app.state, "cv_lab", None)
    if lab is not None:
        await lab.shutdown()
    await app.state.module_container.shutdown()


def create_app() -> FastAPI:
    configure_logging(get_settings())

    app = FastAPI(title="Glasses Tower", lifespan=lifespan)
    app.state.session = ConnectionTracker()
    settings = get_settings()
    cv_module = _build_cv_module(
        settings,
        # Read lazily, so the Lab reports the count at the moment it is
        # asked rather than the count at startup, which is always zero.
        connection_count=lambda: app.state.session.live_connections,
    )
    app.state.module_container = ModuleContainer(cv_module)
    # The SAME object the module holds -- not a second Lab, and not a
    # copy of its state. Two Labs sharing one slot would be the "two
    # experiments at once" failure the whole design exists to prevent, and
    # a copy would be a second answer to "what is running" that starts
    # disagreeing the moment somebody switches.
    app.state.cv_lab = cv_module.lab
    app.state.frame_observers = _build_frame_observers(settings)
    # Read-only, and read by the result channel alone. The web process
    # never builds a world; world_build_session.py does, in its own
    # process, and this is only where to look for what it wrote.
    app.state.world_root = settings.world_root
    # Reads no world and writes no world. It starts the process that
    # does, at the moment a capture id comes into existence -- which is
    # the moment nobody outside this process can know it.
    # Read-only, and read by one HTTP route. The web process never
    # observes and never deletes: the producer is its own script, and
    # deletion is a CLI a human types. Unset means that route answers 404.
    app.state.object_memory_root = settings.observation_root
    # Which classes the READ routes may claim this Tower records. It
    # depends on whether a verifier is configured, and it is derived from
    # the same `Settings` object that builds the producer's argv -- so
    # the surface that answers "have you ever looked for a remote?"
    # cannot disagree with the process that would have written one.
    #
    # A tuple of strings, not an import: `tower/routes/observations.py`
    # is not allowed to know what a verifier is.
    app.state.object_memory_recorded_classes = _recorded_classes(settings)
    # Where the pictures behind the records are. Read-only, and read by
    # one route family: this process serves frames out of the capture
    # tree and never writes to it.
    app.state.capture_root = settings.capture_root
    # One filter for the whole app, so the ONNX session is built once
    # rather than per request. A Tower whose weights are missing gets a
    # filter that reports itself unavailable, and the imagery routes
    # then refuse -- they never fall back to an unfiltered frame.
    app.state.object_memory_face_filter = build_face_filter()
    # The crops this cartridge OWNS, under the same root as the records.
    #
    # Read-only from here, exactly like the store: the web process never
    # writes a keyframe (the producer does, in its own process) and never
    # deletes one (retention does, through `ObservationStore`). This is
    # only where to look. None when object memory is switched off, which
    # is the same condition that makes the routes answer 404.
    app.state.object_memory_keyframes = keyframe_store_from_root(
        settings.observation_root
    )
    # Mutually referential, resolved by a lookup rather than by an
    # ordering trick: the worker spec's gate asks a session whether it is
    # active, and the session needs the supervisor the spec is registered
    # with in order to attach and detach. The dict is populated below,
    # and the gate reads it when a capture opens -- which is always after
    # startup, so it is never empty when it is asked.
    cartridge_sessions: dict[str, CartridgeSession] = {}

    def _object_memory_gate() -> bool:
        session = cartridge_sessions.get(CARTRIDGE_OBJECT_MEMORY)
        return session is not None and session.is_active()

    def _world_build_gate() -> bool:
        session = cartridge_sessions.get(CARTRIDGE_WORLD_BUILDER)
        return session is not None and session.is_active()

    app.state.capture_workers = _build_capture_worker_supervisor(
        settings,
        {
            OBJECT_MEMORY_WORKER: _object_memory_gate,
            WORLD_BUILD_WORKER: _world_build_gate,
        },
    )
    cartridge_sessions[CARTRIDGE_OBJECT_MEMORY] = CartridgeSession(
        cartridge=CARTRIDGE_OBJECT_MEMORY,
        worker=OBJECT_MEMORY_WORKER,
        supervisor=app.state.capture_workers,
        open_capture=_open_capture_lookup(app.state.frame_observers),
        clock=time.time,
    )
    # World Builder's session is INTENT TO BUILD, not a recording consent:
    # "the World Builder workspace is on the phone's screen". Its Stop asks
    # rather than terminates (STOP_POLICY_REQUEST), because a builder that
    # is finalizing a walk the wearer just finished must be allowed to.
    cartridge_sessions[CARTRIDGE_WORLD_BUILDER] = CartridgeSession(
        cartridge=CARTRIDGE_WORLD_BUILDER,
        worker=WORLD_BUILD_WORKER,
        supervisor=app.state.capture_workers,
        open_capture=_open_capture_lookup(app.state.frame_observers),
        clock=time.time,
        stop_policy=STOP_POLICY_REQUEST,
    )
    # Deliberately NOT persisted anywhere. A Tower that restarts comes
    # back with every cartridge stopped, because resuming a memory of
    # what a camera sees without anybody asking again is the wrong
    # direction to fail in.
    app.state.cartridge_sessions = cartridge_sessions
    # The live cartridges this configuration enables, built in one place
    # that knows their names so this file does not have to. Often empty:
    # both are off by default, and an empty list costs nothing on the
    # frame path.
    live = build_live_cartridges(settings)
    app.state.live_cartridges = live
    # A SECOND list, beside `frame_observers`. That one is the dataset
    # recorder's and is shaped around capture lineage -- it mints capture
    # ids, `/health` reports on it, and `ws.py` calls `capture_dir()` on
    # its members unguarded. A cartridge counting frames belongs nowhere
    # near it.
    app.state.frame_consumers = live.frame_consumers
    # Read by the declaration and by two HTTP routes. The web process
    # records a document only when a session is started; unset means the
    # routes answer 404.
    app.state.document_root = settings.document_root
    # The window the session writes under, so a read with no window of
    # its own sees what the writer promised rather than forever. 0 means
    # the operator chose forever, and None is what the routes read that as.
    app.state.document_retention_days = (
        settings.document_retention_days
        if settings.document_retention_days and settings.document_retention_days > 0
        else None
    )
    # Whether a live session exists is a SEPARATE question from whether
    # the library can be read, and the two must not be conflated. A Tower
    # reprocessing captures offline has a library and no session; that is
    # a normal configuration and `/documents` must serve it.
    # Whether the contract may be offered at all, which is a question
    # about configuration and not about whether a session is running.
    app.state.scene_enabled = bool(live.scene is not None)
    # Why it is unavailable, when the reason is not "nobody enabled it".
    # `None` keeps the configured-off wording, so the common case is
    # unchanged; a construction failure replaces it with what actually
    # went wrong instead of naming a variable that is already set.
    app.state.scene_unavailable_reason = live.scene_unavailable_reason
    _log_effective_configuration(settings, app.state.capture_workers)
    # One shared reader for the whole app. It starts no task until a
    # client subscribes and stops again when the last one goes, so a Tower
    # nobody is watching does no polling and no disk IO on its behalf.
    app.state.result_hub = build_hub(
        settings.world_root,
        document_root=settings.document_root,
        scene_source=live.scene,
        document_source=live.document,
        cv_lab=app.state.cv_lab,
        document_unavailable_reason=live.document_unavailable_reason,
    )
    # Started here, not in `lifespan` above: TestClient(create_app()) used
    # without `with client:` (every pre-existing test in this repo) never
    # runs ASGI lifespan events, leaving the module UNLOADED forever. See
    # docs/superpowers/specs/2026-08-19-v0.8-module-container-design.md, "Wiring" Amendment.
    asyncio.run(app.state.module_container.load_and_start())
    app.include_router(health.router)
    app.include_router(cartridges.router)
    app.include_router(cv_lab.router)
    # Its own module rather than more routes on `cv_lab.router`: that one
    # serves one JSON document and this one serves image bytes with an
    # ETag, a conditional GET and a no-store header, and the two have
    # nothing in common but a path prefix.
    app.include_router(cv_lab_preview.router)
    app.include_router(geometry.router)
    app.include_router(observations.router)
    app.include_router(sessions.router)
    app.include_router(scene.router)
    app.include_router(documents.router)
    app.include_router(ws.router)
    return app


app = create_app()
