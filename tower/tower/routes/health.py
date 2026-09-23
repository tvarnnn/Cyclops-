import logging

from fastapi import APIRouter, Request

SERVICE_NAME = "glasses-tower"
API_VERSION = "0.1.0"

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    container = request.app.state.module_container
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": API_VERSION,
        "module_state": container.state.value,
        "module_id": container.descriptor.id,
        "capture": _capture_state(request.app),
        "capture_workers": _worker_state(request.app),
        "cartridge_sessions": _session_state(request.app),
        "background_chore": _chore_state(request.app),
    }


def _chore_state(app) -> dict | None:
    """What the photographic finisher is doing, from another machine.

    ADDED AFTER A 95-MINUTE STALL THAT NOTHING COULD SEE. On 2026-09-22 the
    phone read "Finishing" for over an hour while the finisher sat at 0% CPU,
    and the only way to learn that was a remote shell and py-spy. `running`
    with `seconds_since_progress` climbing is that shape; `owed` with
    `next_run_not_before_seconds` says when it will be tried again; `last`
    says how the previous run ended -- including `stalled`.

    None when this Tower runs no chore. Never raises.
    """
    chore = getattr(app.state, "world_finish_chore", None)
    snapshot = getattr(chore, "snapshot", None)
    if snapshot is None:
        return None
    try:
        return snapshot()
    except Exception:
        logger.exception("[Tower][Health] could not read the background chore")
        return {"error": "unavailable"}


def _session_state(app) -> dict | None:
    """What each controllable cartridge was asked to do, and what happened.

    Separate from `capture_workers`, and the separation is the point:
    that block reports processes, this one reports INTENT. An operator
    debugging "the phone says it is remembering and the store is not
    growing" needs to see both, because the answer is the gap between
    them.

    Never raises. /health is how an operator learns the Tower is unwell;
    it must not itself fail because a subsystem is broken.
    """
    sessions = getattr(app.state, "cartridge_sessions", None)
    if not sessions:
        return None
    state = {}
    for cartridge, session in sessions.items():
        try:
            state[cartridge] = session.snapshot()
        except Exception:
            logger.exception(
                "[Tower][Health] could not read the %s session", cartridge
            )
            state[cartridge] = {"error": "unavailable"}
    return state


def _worker_state(app) -> dict | None:
    """Whether anything is turning captures into anything.

    Answers "why isn't World Builder changing?" from another machine.
    The Tower is normally operated over Tailscale, where a server-side
    log line is invisible, and on 2026-08-24 the only way to establish
    that nothing was following the capture was to notice that no world
    directory had appeared.

    `enabled: false` -- nothing is configured to follow a capture.
    `workers: []` with `enabled: true` -- configured, nothing running
    right now, which is correct between walks and wrong during one.

    `configured` names WHICH workers this Tower knows how to run, which
    `enabled` cannot: with more than one spec, "something is configured"
    stopped being an answer to "is a builder configured". A worker that
    appears in `configured` and never in `workers` during a walk is the
    shape of every failure this block exists to make visible.
    """
    supervisor = getattr(app.state, "capture_workers", None)
    if supervisor is None:
        return None
    try:
        return {
            "enabled": supervisor.enabled,
            "configured": list(supervisor.worker_names()),
            "workers": supervisor.status(),
        }
    except Exception:
        logger.exception("[Tower][Health] could not read worker state")
        return {"error": "unavailable"}


def _capture_state(app) -> dict | None:
    """Whether raw imagery is being written, and how much of it.

    06-PRIVACY-DATA.md requires an Explicit Dataset-Recording Session to
    indicate that it is recording. Until this existed that indication was
    a server-side log line, which nobody operating the Tower from another
    machine -- the normal case, over Tailscale -- can see.

    `null` means no recorder is registered at all, which is different from
    a registered recorder that is idle. Collapsing the two would make "we
    are definitely not recording" indistinguishable from "we are armed and
    one stream_start away".

    Never raises. /health is how an operator learns the Tower is unwell;
    it must not itself fail because a subsystem is broken.
    """
    observers = getattr(app.state, "frame_observers", None) or []
    if not observers:
        return None

    try:
        recording = any(observer.is_recording for observer in observers)
        statuses = [
            observer.status
            for observer in observers
            if getattr(observer, "status", None) is not None
        ]
        latest = statuses[-1] if statuses else None
        return {
            "armed": True,
            "recording": recording,
            "capture_id": None if latest is None else latest.capture_id,
            "frames_written": 0 if latest is None else latest.frames_written,
            "bytes_written": 0 if latest is None else latest.bytes_written,
        }
    except Exception:
        logger.exception("[Tower][Health] could not read capture state")
        return {"armed": True, "error": "unavailable"}
