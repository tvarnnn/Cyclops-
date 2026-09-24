"""Reading a saved world back, cold, without the session that made it.

Everything here is read-only and process-independent: a world written by
one process is opened by another with no shared state, which is the whole
point of the persistence design.

The reporting discipline is that an unavailable figure prints as
``unknown`` and never as ``0``. Zero is a measurement; unknown is the
absence of one, and collapsing the two is the exact conflation Rule 16
forbids -- "no poses were solved" and "we never tried to solve poses" look
identical once both render as 0.
"""

from dataclasses import dataclass

from tower.world_builder.records import format_distance
from tower.world_builder.schema import (
    POSE_STATUS_ANCHOR,
    POSE_STATUS_SOLVED,
    SCALE_UNKNOWN,
)
from tower.world_builder.store import WorldStore

UNKNOWN = "unknown"


@dataclass(frozen=True)
class SessionView:
    session_id: str
    started_at: float
    ended_at: float | None
    end_reason: str | None
    frames_observed: int
    keyframes_accepted: int
    rejected_by_reason: dict
    intrinsics_source: str
    backend_id: str | None
    downgraded_from: str | None
    downgrade_reason: str | None
    redaction: str
    keyframes: int
    edges: int
    pose_status_counts: dict
    degeneracy_counts: dict
    segments: int
    points: int

    @property
    def has_geometry(self) -> bool:
        return self.points > 0


class WorldView:
    """A loaded world. Cheap to construct; reads journals on demand."""

    def __init__(self, store: WorldStore, world_id: str) -> None:
        self._store = store
        self._world = store.read_world(world_id)

    @property
    def world(self):
        return self._world

    @property
    def world_id(self) -> str:
        return self._world.world_id

    @property
    def scale(self):
        return self._world.scale

    def format_distance(self, value: float) -> str:
        """Never prints metres unless the scale genuinely licenses it."""
        return format_distance(value, self._world.scale)

    def session_ids(self) -> list[str]:
        return self._store.list_session_ids(self.world_id)

    def session(self, session_id: str) -> SessionView:
        store = self._store
        session = store.read_session(self.world_id, session_id)
        keyframes = store.read_keyframes(self.world_id, session_id)
        edges = store.read_edges(self.world_id, session_id)
        derived = store.read_derived(self.world_id, session_id)

        pose_status_counts: dict[str, int] = {}
        degeneracy_counts: dict[str, int] = {}
        points = 0
        if derived is not None:
            for pose in derived["poses"]:
                status = pose["status"]
                pose_status_counts[status] = pose_status_counts.get(status, 0) + 1
                reason = pose.get("degeneracy") or ""
                if reason:
                    degeneracy_counts[reason] = degeneracy_counts.get(reason, 0) + 1
            points = len(derived["points"])

        segments = (
            len({keyframe.segment_index for keyframe in keyframes})
            if keyframes
            else 0
        )

        return SessionView(
            session_id=session_id,
            started_at=session.started_at,
            ended_at=session.ended_at,
            end_reason=session.end_reason,
            frames_observed=session.frames_observed,
            keyframes_accepted=session.keyframes_accepted,
            rejected_by_reason=dict(session.rejected_by_reason),
            intrinsics_source=session.intrinsics.source,
            backend_id=session.backend_id,
            downgraded_from=session.backend_downgraded_from,
            downgrade_reason=session.backend_downgrade_reason,
            redaction=session.redaction,
            keyframes=len(keyframes),
            edges=len(edges),
            pose_status_counts=pose_status_counts,
            degeneracy_counts=degeneracy_counts,
            segments=segments,
            points=points,
        )

    def trajectory(self, session_id: str) -> list[dict]:
        """Ordered camera poses, for overview and recorded-path replay.

        Returns every keyframe in order with its pose when one was
        solved, its image path, and its segment. A consumer walking this
        list has everything needed to show the reconstructed path and to
        step into the wearer's original view at any point along it --
        which is why no Tower-side "replay" feature is needed beyond
        persisting what is already persisted.

        Entries whose pose was refused carry ``pose: None`` and keep their
        degeneracy reason, so a viewer can show the gap honestly rather
        than interpolating across it.
        """
        keyframes = self._store.read_keyframes(self.world_id, session_id)
        derived = self._store.read_derived(self.world_id, session_id)
        poses_by_id = {}
        if derived is not None:
            poses_by_id = {row["keyframe_id"]: row for row in derived["poses"]}

        trajectory = []
        for keyframe in keyframes:
            row = poses_by_id.get(keyframe.keyframe_id)
            solved = bool(
                row
                and row["status"] in (POSE_STATUS_SOLVED, POSE_STATUS_ANCHOR)
                and row["translation"] is not None
            )
            trajectory.append(
                {
                    "keyframe_id": keyframe.keyframe_id,
                    "source_seq": keyframe.source_seq,
                    # THE SEGMENT THE POSE IS IN -- the derived row's, which a gated solve
                    # may have split off the tracker's (review V7, M1); the pose is only
                    # meaningful in that segment's frame. The tracker's when there is no row.
                    "segment_index": (int(row["segment_index"]) if row and row.get("segment_index")
                                      is not None else keyframe.segment_index),
                    "tracker_segment_index": keyframe.segment_index,
                    "image_relpath": keyframe.image_relpath,
                    "status": row["status"] if row else UNKNOWN,
                    "degeneracy": (row.get("degeneracy") or "") if row else "",
                    "pose": (
                        {
                            "translation": row["translation"],
                            "rotation_wxyz": row["rotation"],
                        }
                        if solved
                        else None
                    ),
                }
            )
        return trajectory

    def events(
        self, session_id: str, after_event_id: int | None = None
    ) -> list[dict]:
        """The incremental update stream, optionally from a cursor.

        Works on a session that has not stopped. This is the read half of
        live viewing: the journal is written as the walk happens, so a
        second process can follow along without any push channel, any
        subscription lifecycle, or any change to the module contract.
        """
        return self._store.read_events(self.world_id, session_id, after_event_id)

    def points(self, session_id: str) -> list[list[float]]:
        derived = self._store.read_derived(self.world_id, session_id)
        return [] if derived is None else derived["points"]

    def to_report(self) -> dict:
        sessions = [
            _session_report(self.session(session_id))
            for session_id in self.session_ids()
        ]
        return {
            "world_id": self.world_id,
            "display_name": self._world.display_name or UNKNOWN,
            "schema_version": self._world.schema_version,
            "frame_revision": self._world.frame_revision,
            "pose_convention": self._world.pose_convention,
            "scale": {
                "state": self._world.scale.state,
                "meters_per_unit": (
                    self._world.scale.meters_per_unit
                    if self._world.scale.meters_per_unit is not None
                    else UNKNOWN
                ),
                "allows_metres": self._world.scale.allows_metres,
            },
            "images_purged": self._world.images_purged,
            "storage_bytes": self._store.world_bytes(self.world_id),
            "sessions": sessions,
        }


def _downgrade_report(view: SessionView) -> str:
    """"Not downgraded" and "never built" are different answers.

    `backend_downgraded_from` is None in both cases, and rendering both
    as "unknown" is how a session that ran the calibrated backend cleanly
    came out reading `downgraded_from unknown` -- which an operator
    checking a fresh calibration reads as a downgrade from a backend
    nobody can name. That is the exact question this report exists to
    answer, so it answers it.

    A session that has never been built has no `backend_id` either, and
    for that one "unknown" is the truth.
    """
    if view.downgraded_from:
        return view.downgraded_from
    if view.backend_id is None:
        return UNKNOWN
    return "none"


def _session_report(view: SessionView) -> dict:
    return {
        "session_id": view.session_id,
        "end_reason": view.end_reason or UNKNOWN,
        "ended_at": view.ended_at if view.ended_at is not None else UNKNOWN,
        "frames_observed": view.frames_observed,
        "keyframes_accepted": view.keyframes_accepted,
        "rejected_by_reason": view.rejected_by_reason or UNKNOWN,
        "intrinsics_source": view.intrinsics_source,
        "backend_id": view.backend_id or UNKNOWN,
        "backend_downgraded_from": _downgrade_report(view),
        "redaction": view.redaction,
        "keyframes": view.keyframes,
        "edges": view.edges,
        "segments": view.segments,
        "pose_status_counts": view.pose_status_counts or UNKNOWN,
        "degeneracy_counts": view.degeneracy_counts or UNKNOWN,
        "points": view.points if view.has_geometry else UNKNOWN,
    }


def open_world(root, world_id: str) -> WorldView:
    return WorldView(WorldStore(root), world_id)


def render_text(report: dict) -> str:
    lines = [
        "=== World ===",
        f"world_id        {report['world_id']}",
        f"name            {report['display_name']}",
        f"schema_version  {report['schema_version']}",
        f"frame_revision  {report['frame_revision']}",
        f"images_purged   {report['images_purged']}",
        "",
        "=== Scale ===",
        f"state           {report['scale']['state']}",
        f"meters_per_unit {report['scale']['meters_per_unit']}",
        f"metres allowed  {report['scale']['allows_metres']}",
    ]
    if report["scale"]["state"] == SCALE_UNKNOWN:
        # States the RULE rather than one of its two causes. `unknown` is
        # set both by "no solved pose" and by "more than one segment,
        # each with its own arbitrary unit" -- so a world with 29 solved
        # poses across 6 segments used to be told it had no solved pose.
        lines.append(
            "note            no unit at all -- that needs solved poses "
            "inside a SINGLE segment"
        )
    elif not report["scale"]["allows_metres"]:
        lines.append(
            "note            internally consistent, arbitrary unit -- NOT metric"
        )

    storage = report["storage_bytes"]
    lines += [
        "",
        "=== Storage ===",
        f"total           {storage['total'] / 1e6:.2f} MB",
        f"  images        {storage['images'] / 1e6:.2f} MB  (authoritative)",
        f"  journals      {storage['journals'] / 1e6:.2f} MB  (authoritative)",
        f"  derived       {storage['derived'] / 1e6:.2f} MB  (reclaimable)",
    ]

    for session in report["sessions"]:
        lines += [
            "",
            f"=== Session {session['session_id']} ===",
            f"end_reason           {session['end_reason']}",
            f"frames_observed      {session['frames_observed']}",
            f"keyframes_accepted   {session['keyframes_accepted']}",
            f"segments             {session['segments']}",
            f"edges                {session['edges']}",
            f"intrinsics_source    {session['intrinsics_source']}",
            f"backend              {session['backend_id']}",
            f"downgraded_from      {session['backend_downgraded_from']}",
            f"redaction            {session['redaction']}",
            f"points               {session['points']}",
            f"pose_status          {session['pose_status_counts']}",
            f"degeneracy           {session['degeneracy_counts']}",
            f"rejected_by_reason   {session['rejected_by_reason']}",
        ]
    return "\n".join(lines)
