"""Following things across frames, so a count means something.

This is the module the brief singles out: *person counting must use
tracking rather than naively summing detections*. Two failure modes make
that non-negotiable, and they pull in opposite directions:

- a detector that misses a person on one frame in five reports a count
  flickering between 2 and 3 while nothing in the room changed;
- a detector that fires twice on one person reports two people.

A tracker with a hit streak fixes the first; association fixes the
second. Neither is fixed by a better detector alone.

There is a third, and it is quieter than both: a confirmed track dropped
while its person is behind a doorframe comes back as a NEW `track_id`
and is counted as somebody new. And a fourth, pulling the other way: a
confirmed track kept alive after its person has LEFT is counted as
somebody still there. The first version of this module bought the third
with one constant and paid for it with the fourth. They are now two
constants -- how long a track is KEPT (`MAX_ABSENCE_S`, continuity) and
how long it is COUNTED (`MAX_COUNT_ABSENCE_S`, presence) -- because the
2026-09-07 labelled sequences showed the split is what the count needs.

**Association is by IoU only.** Not by appearance. Matching by how
something looks is the first step toward recognising it again, and this
cartridge must never do that -- `track_id` means "the same blob one frame
later" and nothing more.

**Matching maximises how many tracks get a detection FIRST, and the
total overlap SECOND.** Greedy shipped first and an adversarial review
broke it in one frame: given IoU(T1,D1)=1.00, IoU(T1,D2)=0.33,
IoU(T2,D1)=0.25, IoU(T2,D2)=0.00, a complete matching exists (T1<-D2,
T2<-D1) and greedy takes T1<-D1 instead, stealing T2's only qualifying
detection; T2 then starves and a phantom confirms in its place.
Maximum-cardinality matching (Kuhn, augmenting paths with candidates in
IoU order) replaced it and fixed that case -- but among the matchings of
equal size it picked one by the ORDER it visited tracks, not by how
well the pairs overlap, and on the 2026-09-07 labelled camera-motion
sequences that swapped adjacent people's ids under a head turn. The
assignment is now Hungarian over `1 + IoU` for every qualifying pair:
the constant makes every extra matched pair worth more than any
overlap, so cardinality still wins first, and the IoU decides among the
complete matchings. Track ids are never published, so what a switch
costs here is orientation history and a spurious "somebody new".
"""

from dataclasses import dataclass

from tower.scene.records import BoundingBox, Detection, FacingEstimate, Track

# The interval at which frames actually arrive, in seconds. Measured from
# the corpus's own `frames.jsonl` receipt timestamps: a median gap of
# 83.5 ms across 9,145 frames in the 14 captures with more than 50 of
# them, i.e. 12.0 fps, with per-capture medians of 68.5-87.6 ms.
#
# It lives HERE, at the tracker, rather than in the engine that also
# needs it, because thresholds below are derived from it and the engine
# imports this module. A constant a threshold depends on must sit no
# higher than the threshold.
DELIVERED_FRAME_INTERVAL_S = 0.0835


def frames_in(seconds: float) -> int:
    """How many delivered frames fit in a duration.

    The whole defect this module carried was a duration written as a
    frame count: `max_misses = 5` was justified as "roughly 1.5 seconds"
    against an assumed ~3.3 fps, and when the real rate turned out to be
    12.0 fps the number silently became 0.42 s while its comment went on
    claiming 1.5. A duration that is converted rather than counted
    cannot drift like that again.
    """
    return max(1, round(seconds / DELIVERED_FRAME_INTERVAL_S))


# How long a thing may be absent and still be the same thing when it
# comes back. A wall-clock duration, because an occlusion is one: a
# person passing behind a doorframe at walking pace is hidden for about
# half a second, behind another person for a little less, and through a
# head turn away and back for about one.
#
# Measured on 9,145 corpus frames
# (`docs/superpowers/research/2026-08-26-tracker-retune.md`): raising
# the budget from 5 frames to 12 cut the `person` ids created from 134
# to 104, and count stability at 18 and 24 frames is identical to 12.
# 1.0 s is where continuity stops improving.
MAX_ABSENCE_S = 1.0

# How long a confirmed thing may be unseen and still be COUNTED as in
# view. Shorter than `MAX_ABSENCE_S`, deliberately. On the 2026-09-07
# labelled sequences (40 COCO scenes under synthetic head motion, 10,800
# frames, two real detectors, five dropout/occlusion conditions) the
# count's error was dominated by DEPARTURE LAG -- people who had panned
# out of view still counted for a second -- and counting only tracks
# seen within 0.5 s took the exact-count rate from 0.315 to 0.370 and
# the mean error from 1.46 to 1.19 people, while leaving continuity at
# 1.0 s untouched (ids per person unchanged). 0.25 s helped the count a
# little more (0.415) and doubled the flicker (0.08 -> 0.18); 0.5 s is
# where those meet.
MAX_COUNT_ABSENCE_S = 0.5


@dataclass(frozen=True)
class TrackerPolicy:
    """Every threshold, with its reason.

    A value object rather than constants because the benchmark sweeps
    them, and a threshold that cannot be swept cannot be chosen from data.

    Two of these are frame counts and two are durations, and knowing
    which is which is what the 3.6x frame-rate error turned on.
    `min_hits` and `min_iou` describe what happens BETWEEN two frames --
    detector noise and object motion -- so they are per-frame quantities.
    `max_misses` and `max_count_misses` describe durations in the room,
    so they are derived from the rate.
    """

    # Below this two boxes are not the same thing one frame later.
    #
    # Derived from the corpus: the 1st percentile of IoU between the same
    # object's boxes in consecutive frames is 0.525 for `person`, 0.613
    # for `laptop` and 0.386 for `cell phone` (medians 0.96, 0.96, 0.91 --
    # at 12 fps a box barely moves). 0.25 is the largest floor that keeps
    # at least 99.5% of every measured label's true associations. The
    # 2026-09-07 sweep over 0.2/0.25/0.3 moved nothing decisively.
    min_iou: float = 0.25
    # How many CONSECUTIVE frames a track must be seen before it counts.
    # One detection is a flicker; so is one every six frames, which is why
    # this is a streak and not a lifetime total. Swept both ways on the
    # corpus, and 3 is where both neighbours are worse.
    min_hits: int = 3
    # How many consecutive misses before a track is DROPPED: 12, which is
    # `MAX_ABSENCE_S` at the measured frame interval and nothing else.
    max_misses: int = frames_in(MAX_ABSENCE_S)
    # How many consecutive misses before a confirmed track stops being
    # COUNTED: 6, which is `MAX_COUNT_ABSENCE_S` at the same interval. A
    # policy whose `max_misses` is smaller than this counts every
    # confirmed track it keeps, which is the pre-2026-09-07 behaviour.
    max_count_misses: int = frames_in(MAX_COUNT_ABSENCE_S)


class Tracker:
    """Anonymous multi-object tracking. No appearance, no identity.

    Per class, by IoU, with a maximum-weight assignment. Still no motion
    model: a constant-velocity Kalman filter was benchmarked on the
    2026-09-07 sequences and bought fewer switches at the price of more
    phantom tracks (408 against 273) and no count accuracy, at three
    times the cost. **Identity through a symmetric crossing is not
    preserved**, and cannot be from boxes alone; for this cartridge that
    is a small cost, because it must never have identity in the first
    place.
    """

    def __init__(self, policy: TrackerPolicy | None = None) -> None:
        self._policy = policy or TrackerPolicy()
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def policy(self) -> TrackerPolicy:
        return self._policy

    @property
    def tracks(self) -> list[Track]:
        """Every live track, confirmed or not."""
        return list(self._tracks)

    def confirmed(self) -> list[Track]:
        """The tracks that have earned confirmation and are still alive."""
        return [
            track
            for track in self._tracks
            if track.confirmed(self._policy.min_hits)
        ]

    def counted(self) -> list[Track]:
        """The tracks a count may be taken from: confirmed AND recently seen."""
        return [
            track
            for track in self._tracks
            if track.counted(self._policy.min_hits, self._policy.max_count_misses)
        ]

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def update(self, detections: list[Detection], *, at: float) -> list[Track]:
        """Associate this frame's detections and return the live tracks.

        Association is **within a class**: a chair box overlapping a
        person box is not evidence that the chair became a person, and
        cross-class association would let one label flip rewrite a
        track's identity.
        """
        assignment = self._match(detections)

        matched_detections: set[int] = set()
        for track_index, detection_index in assignment.items():
            track = self._tracks[track_index]
            detection = detections[detection_index]
            matched_detections.add(detection_index)
            reacquired = track.misses > 0
            track.box = detection.box
            track.score = detection.score
            track.last_seen_at = at
            track.hits += 1
            track.streak += 1
            track.misses = 0
            if track.streak >= self._policy.min_hits:
                track.is_confirmed = True
            if reacquired:
                # Re-matched after a gap. Whatever was behind that gap --
                # an occlusion, or a DIFFERENT PERSON stepping into the
                # same spot -- this box is no longer evidence for the
                # orientation measured before it. Carrying it forward
                # reported one person's facing as another's.
                track.facing = FacingEstimate()
                track.facing_estimated_at = None
                track.facing_history = ()

        for index, track in enumerate(self._tracks):
            if index not in assignment:
                track.misses += 1
                # A streak is consecutive by definition. Confirmation
                # already latched stays latched; earning it starts over.
                track.streak = 0

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            self._tracks.append(
                Track(
                    track_id=self._next_id,
                    label=detection.label,
                    box=detection.box,
                    score=detection.score,
                    first_seen_at=at,
                    last_seen_at=at,
                    is_confirmed=self._policy.min_hits <= 1,
                )
            )
            self._next_id += 1

        self._tracks = [
            track
            for track in self._tracks
            if track.misses <= self._policy.max_misses
        ]
        return self.tracks

    def _match(self, detections: list[Detection]) -> dict:
        """Maximum-cardinality, then maximum-IoU matching of tracks to detections.

        Weight is `1 + IoU`, zero below the floor and across classes; a
        zero weight is never assigned. Small by construction -- a handful of
        tracks and detections per frame -- so an exact O(n^3) method
        costs microseconds. Deliberately no scipy:
        `linear_sum_assignment` would do this in one line, but scipy
        arrived here as an OCR dependency and this cartridge must not
        acquire it by accident.
        """
        if not self._tracks or not detections:
            return {}
        weights = []
        for track in self._tracks:
            row = []
            for detection in detections:
                if detection.label != track.label:
                    row.append(0.0)
                    continue
                score = track.box.iou(detection.box)
                # 1 + IoU: any matched pair outweighs any overlap, so the
                # assignment maximises the number of matched tracks
                # before it maximises how well they overlap. A zero is
                # "not a candidate" and is never assigned.
                row.append(1.0 + score if score >= self._policy.min_iou else 0.0)
            weights.append(row)
        assignment = {}
        for track_index, detection_index in maximum_weight_assignment(weights):
            if weights[track_index][detection_index] > 0.0:
                assignment[track_index] = detection_index
        return assignment

    def count(self, label: str) -> int:
        """How many COUNTED tracks carry this label.

        The one number the brief insists must not come from detections.
        """
        return sum(1 for track in self.counted() if track.label == label)


def maximum_weight_assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Pairs `(row, column)` maximising the total weight. Hungarian, O(n^3).

    Rectangular matrices are padded with zero-weight rows or columns, so
    every row and column is assigned once and the caller drops the
    zero-weight pairs. Deterministic: ties resolve by index order.
    """
    rows = len(weights)
    columns = len(weights[0]) if rows else 0
    if rows == 0 or columns == 0:
        return []
    size = max(rows, columns)
    # Minimise negated weights over a square, 1-indexed matrix.
    cost = [[0.0] * (size + 1) for _ in range(size + 1)]
    for r in range(rows):
        for c in range(columns):
            cost[r + 1][c + 1] = -weights[r][c]

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    row_of_column = [0] * (size + 1)
    for r in range(1, size + 1):
        row_of_column[0] = r
        minimum = [float("inf")] * (size + 1)
        visited = [False] * (size + 1)
        previous = [0] * (size + 1)
        column = 0
        while True:
            visited[column] = True
            row = row_of_column[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, size + 1):
                if visited[candidate]:
                    continue
                reduced = cost[row][candidate] - u[row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if visited[candidate]:
                    u[row_of_column[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if row_of_column[column] == 0:
                break
        while column:
            row_of_column[column] = row_of_column[previous[column]]
            column = previous[column]

    pairs = []
    for c in range(1, size + 1):
        r = row_of_column[c]
        if 1 <= r <= rows and 1 <= c <= columns:
            pairs.append((r - 1, c - 1))
    pairs.sort()
    return pairs


def detections_from_boxes(label: str, boxes, score: float = 0.9):
    """Convenience for tests and fixtures: boxes are (x0, y0, x1, y1)."""
    return [
        Detection(label=label, score=score, box=BoundingBox(*box)) for box in boxes
    ]
