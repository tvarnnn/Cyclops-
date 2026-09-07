"""Scene Understanding as a live session: Start, Pause, Stop, and a clock.

`SceneEngine` is pull-shaped -- hand it a decoded frame, get a state back,
keep nothing. That is the right shape for a script and the wrong shape for
a web process, which owns neither the loop nor the timing.

The lifecycle, the single-slot frame path and the abandoned-load latch are
all `tower/live_session.py`, shared with Document Memory because both
cartridges need exactly the same three properties around an engine that is
synchronous, blocking and expensive. What is HERE is only what is specific
to this cartridge, and two things are.

**Stop must mean stopped.** On `stop()` the last scene is DISCARDED, not
retained. A scene held past the end of a session and served to a client is
the failure this cartridge is least able to survive: it is a claim about
what is around a person right now, and it would be wrong. No staleness
number is large enough to make it safe, because a client that renders
counts above staleness shows the room first.

`pause()` is the deliberately different case -- the scene survives and the
payload says it is no longer being refreshed. `IOS-to-Tower.md` 4.7 asks
the Tower to keep `observing` and `lastKnown` apart rather than flatten
them, and this is that distinction.

Document Memory's Stop does the opposite and keeps what it recorded. That
asymmetry is the difference between the two cartridges, not an
inconsistency: a scene expires the moment nobody is looking, and a record
of what was read does not.

WHEN IT RUNS: STREAM AND WATCHER, OR OPERATOR

A scene session runs while **somebody is streaming AND somebody is
watching**. Two facts, tracked apart, neither sufficient alone:

  * the STREAM is the feed. `stream_opened` / `stream_closed` arrive from
    the connection that sends frames. With no stream there is nothing to
    observe, so the last stream closing stops the session whoever started
    it -- which closes integration finding 15, where a Stop -> Start over
    HTTP left a session owned by nobody that its own stream could no
    longer end.
  * a WATCHER is the demand. `watcher_joined` / `watcher_left` arrive from
    the result channel when a client subscribes to the live scene. A phone
    streaming for World Builder or the CV Lab is not a request to detect
    people in the room; a subscription to this cartridge's live result is.
    When the last watcher leaves, the detector is released and the GPU is
    handed back.

The OPERATOR path is separate and deliberately so. `start()` -- reached
from `POST /scene/start` -- runs the session at once, with or without a
stream, so a physical test can be driven from a Mac or curl without a
phone build; `stop()` ends that hold. `follow_stream=False`
(`TOWER_SCENE_AUTOSTART=false`) keeps ONLY the operator path.

No demand event ever resumes a Pause. A Pause is a person's decision; a
socket connecting or a screen opening is not.

PERSISTS NOTHING, ENFORCED

`test_scene_understanding_persists_nothing` AST-walks every file on this
cartridge's wire path -- this one, the adapter, the route, the runtime
factory and the shared session base -- and fails on any call that could
write. Frames arrive here as bytes, are decoded into an array, are
counted, and are dropped.
"""

import threading

from tower.live_session import (
    LIFECYCLE_STATES,
    LOAD_OVERDUE_S,
    STATE_FAILED,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    STOP_JOIN_TIMEOUT_S,
    LiveSession,
    decode_frame,
)

# Published on the status so a client can see WHY a session is or is not
# running, in one word it can pin.
RUNS_WHEN = "stream-and-watcher-or-operator"

__all__ = [
    "LIFECYCLE_STATES",
    "RUNS_WHEN",
    "LOAD_OVERDUE_S",
    "STATE_FAILED",
    "STATE_PAUSED",
    "STATE_RUNNING",
    "STATE_STARTING",
    "STATE_STOPPED",
    "STOP_JOIN_TIMEOUT_S",
    "SceneLive",
    "decode_frame",
]


class SceneLive(LiveSession):
    """One Scene Understanding session, driven by frames from elsewhere.

    Constructed with a factory rather than an engine, because `start()`
    after a `stop()` must build a fresh engine: `SceneEngine.release()`
    releases the models and resets the tracker, and re-loading a released
    detector is not a path this repository supports -- `tower/loading.py`
    is explicit that "FAILED is terminal in this repo's lifecycle". A new
    session being a new object graph is also what "session-scoped track
    ids" already promised.
    """

    name = "Scene"

    def __init__(
        self,
        engine_factory,
        *,
        decode=decode_frame,
        follow_stream: bool = True,
        **kwargs,
    ) -> None:
        # `follow_stream` defaults ON here and OFF on the base, and the
        # difference is the cartridge. Enabling Scene Understanding is
        # already the opt-in, it writes nothing, and requiring a second
        # out-of-band step was not a safety property -- it was a dead
        # product path, because iOS sends nothing when a cartridge is
        # opened. TOWER_SCENE_AUTOSTART=false restores manual control,
        # and the five routes work either way.
        #
        # Document Memory deliberately defaults it OFF: it writes to
        # disk, and a session that persists gets an explicit start.
        super().__init__(follow_stream=follow_stream, **kwargs)
        self._engine_factory = engine_factory
        self._decode = decode
        self._latest = None
        self._latest_observed_at: float | None = None
        self._latest_computed_at: float | None = None
        self._decode_failures = 0
        # Demand. Held under `_demand`, a lock of their own, and acted on
        # with that lock still held so two demand events cannot interleave
        # into a decision neither of them made -- a watcher leaving and
        # another joining, resolved as "stop" then "start", must not land
        # as "start" then "stop". `stop()` and `start()` take it first
        # too, so the order is always demand -> lifecycle -> condition and
        # can never invert.
        self._demand = threading.RLock()
        # Connections currently streaming frames. NOT cleared by a stop:
        # a stream that is open stays open whether or not anyone is
        # watching it, which is exactly the fact the old owner set lost.
        self._open_streams: set = set()
        # Result-channel subscriptions to the live scene, token -> owner
        # connection, so a connection closing can drop every watcher it
        # held without knowing their tokens.
        self._watchers: dict = {}
        # Whether an operator started this by hand. Cleared by `stop()`.
        self._operator_hold = False

    def latest(self):
        """The most recent `SceneState`, or None, with when it was taken.

        Returns `(state, observed_at, computed_at)`. `None` is a real
        answer and means one of four things the caller must keep apart:
        the session is stopped, it is still loading, it failed, or it is
        running and has not finished a frame yet. `status()` says which.

        The state is safe to hold: `SceneEngine.observe` copies the
        tracker's tracks (`tests/test_scene_snapshot_isolation.py`), so
        nothing this returns changes underneath a publisher.
        """
        with self._condition:
            return (self._latest, self._latest_observed_at, self._latest_computed_at)

    # -- demand --------------------------------------------------------

    def start(self, *, resume_paused: bool = True) -> dict:
        """The operator's Start. Runs at once, stream or no stream."""
        with self._demand:
            self._operator_hold = True
            return super().start(resume_paused=resume_paused)

    def stop(self) -> dict:
        """The operator's Stop, and every other stop. Releases the hold."""
        with self._demand:
            self._operator_hold = False
            return super().stop()

    def stream_opened(self, owner=None) -> None:
        """A connection began streaming frames. Run if somebody is watching."""
        with self._demand:
            self._open_streams.add(owner)
            self._reconcile()

    def stream_closed(self, owner=None) -> None:
        """A streaming connection ended. The last one out stops the session.

        Whoever started it. Frames arrive only from a stream, so a session
        with no stream has nothing left to do -- and, for this cartridge,
        a scene kept past its feed is a claim about a room the wearer has
        left. A connection that never streamed is not an owner and does
        not count, so a passing socket cannot end an operator's session.
        """
        with self._demand:
            was_streaming = owner in self._open_streams
            self._open_streams.discard(owner)
            if was_streaming and not self._open_streams:
                super().stop()

    def watcher_joined(self, token, *, owner=None) -> None:
        """A client subscribed to the live scene. Run if a stream is open."""
        with self._demand:
            self._watchers[token] = owner
            self._reconcile()

    def watcher_left(self, token) -> None:
        """That subscription ended. The last watcher out stops the session.

        Unless an operator is holding it: a Mac driving a physical test
        keeps its session across the phone's screen changes.
        """
        with self._demand:
            self._watchers.pop(token, None)
            self._stop_if_unwatched()

    def watchers_left(self, *, owner) -> None:
        """A connection closed; every watcher it held leaves with it."""
        with self._demand:
            for token in [t for t, o in self._watchers.items() if o == owner]:
                del self._watchers[token]
            self._stop_if_unwatched()

    def _reconcile(self) -> None:
        """Start if the two facts are both true. Under `_demand`.

        `resume_paused=False`: a demand event is a socket or a screen, not
        a person, and must not undo a Pause. Starting from FAILED is
        allowed because each call here is one discrete request -- a new
        watcher, a new stream -- and gets one new attempt; nothing here
        loops, so a load that keeps failing fails once per request.
        """
        if not self._follow_stream:
            return
        if self._open_streams and self._watchers:
            super().start(resume_paused=False)

    def _stop_if_unwatched(self) -> None:
        if self._operator_hold or self._watchers:
            return
        super().stop()

    # -- hooks ---------------------------------------------------------

    def _create(self):
        engine = self._engine_factory()
        engine.load()
        return engine

    def _engine_name(self, engine):
        return getattr(getattr(engine, "_detector", None), "name", None)

    def _consume(self, engine, raw_bytes, received_at, source_seq):
        frame = self._decode(raw_bytes)
        if frame is None:
            with self._condition:
                self._decode_failures += 1
            # `None` travels on like any other result and `_publish`
            # ignores it. Counting the failure and continuing is the same
            # answer `SceneEngine._detect` already gives a bad frame.
            return None
        return engine.observe(frame, received_at=received_at)

    def _publish(self, result, received_at: float, now: float) -> None:
        if result is None:
            # A frame that would not decode is not an observation. Undo
            # the increment the base made on our behalf, so
            # `frames_observed` counts frames that produced a scene
            # rather than frames that were taken off the slot.
            self._frames_observed -= 1
            return
        self._latest = result
        self._latest_observed_at = received_at
        self._latest_computed_at = now

    def _on_start_locked(self) -> None:
        self._decode_failures = 0
        self._latest = None
        self._latest_observed_at = None
        self._latest_computed_at = None

    def _on_stop_locked(self) -> None:
        """The load-bearing three lines in this class.

        See the module header. A cartridge whose entire claim is "what is
        around you NOW" must not answer that question after it stopped
        looking, and a state left in a field is exactly how it would.
        """
        self._latest = None
        self._latest_observed_at = None
        self._latest_computed_at = None

    def _extra_status(self) -> dict:
        now = self._clock()
        return {
            "detector": getattr(self, "_engine_label", None),
            "decode_failures": self._decode_failures,
            # When the FRAME was received, not when the detector finished
            # with it. A 30 ms detection would otherwise make every scene
            # look 30 ms fresher than it is, and the error grows with the
            # cost of the model.
            "observed_at": self._latest_observed_at,
            "computed_at": self._latest_computed_at,
            "staleness_seconds": (
                None
                if self._latest_observed_at is None
                else max(now - self._latest_observed_at, 0.0)
            ),
            "has_state": bool(self._latest is not None),
            "follows_stream": bool(self._follow_stream),
            # Why the session is, or is not, running. Counts rather than
            # tokens: a subscription id or a connection token on the wire
            # would be a handle to something this payload should not name.
            "demand": {
                "streams": len(self._open_streams),
                "watchers": len(self._watchers),
                "operator_hold": bool(self._operator_hold),
                "runs_when": RUNS_WHEN,
            },
        }
