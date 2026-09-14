"""The incremental-update stream, as an append-only journal.

An append-only JSONL journal IS an incremental update stream: a reader
tails it and sees each keyframe and each state change as a line, in order.
It needs no socket, no subscription lifecycle, no backpressure policy, and
carries no risk of an unbounded queue.

Deliberately NOT an in-process pub/sub bus. Nothing subscribes in-process
today, because World Builder is not registered as a production module --
the wire contract is scalar-only and the lifecycle work that would change
that is blocked. A bus would be machinery with no consumer. When the
module contract opens up, this journal is what a publisher reads from, and
the events already have stable ids and a closed kind set.

`event_id` is dense within a session on purpose: a gap means an event was
genuinely dropped, not that the writer skipped a number. A consumer can
therefore always tell that it missed something.
"""

import logging

from tower.world_builder.schema import SCHEMA_VERSION, TIME_BASIS

logger = logging.getLogger(__name__)

# Closed set. A kind outside it is a bug, not an extension point -- adding
# one is a schema change, because consumers switch on these.
EVENT_KINDS = frozenset(
    {
        "session_started",
        "keyframe_accepted",
        "frame_rejected",
        "tracking_lost",
        # The SOLVE chain failed, not tracking. Distinct from
        # tracking_lost on purpose: the tracker is healthy and its
        # reference is kept, so a consumer must not read this as the
        # wearer having lost the world. It does not set last_tracking.
        "solve_chain_broken",
        "segment_started",
        "backend_downgraded",
        "mapping_stalled",
        "build_completed",
        "session_stopped",
        # The sender's `source_seq` went backwards mid-session (the
        # glasses restarted inside a capture lineage) and the engine
        # renumbered from there. Payload: `source_seq`, `renumbered_to`.
        # Informational; no consumer switches on it.
        "source_seq_restarted",
    }
)


class WorldEvent:
    __slots__ = ("event_id", "kind", "at", "payload")

    def __init__(self, event_id: int, kind: str, at: float, payload: dict):
        self.event_id = event_id
        self.kind = kind
        self.at = at
        self.payload = payload

    def to_json_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": self.event_id,
            "kind": self.kind,
            "at": self.at,
            "time_basis": TIME_BASIS,
            "payload": dict(self.payload),
        }


class EventLog:
    """Appends session events to the durable journal."""

    def __init__(self, store, world_id: str, session_id: str, clock) -> None:
        self._store = store
        self._world_id = world_id
        self._session_id = session_id
        self._clock = clock
        self._next_id = 0

    def append(self, kind: str, payload: dict | None = None) -> WorldEvent:
        if kind not in EVENT_KINDS:
            raise ValueError(
                f"unknown event kind {kind!r}; the set is closed because "
                "consumers switch on it"
            )
        event = WorldEvent(
            event_id=self._next_id,
            kind=kind,
            at=self._clock(),
            payload=payload or {},
        )
        self._next_id += 1
        self._store.append_event(self._world_id, self._session_id, event)
        return event
