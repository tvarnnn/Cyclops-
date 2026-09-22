"""A camera that is not looking at the shared geometry must not vote.

THE DEFECT, stated once so the tests below are readable.

`_pnp_observations` places any target keyframe for which twelve of the
source segment's landmarks survive PnP RANSAC. On repetitive indoor
texture, a keyframe sharing no physical view with the source clears that
bar on aliased matches -- and the fabricated cameras agree with each
other, because they all collapse toward the origin together. A Sim3's
scale IS the ratio of the placed constellation's span to the target's own
span, so a partly collapsed constellation reports a scale that is
confidently, sharply wrong.

Measured with a known answer, by splitting one real segment into halves
that share a frame and a unit by construction (truth: scale exactly 1.0):
segment 29 of the 2026-08-29 drawer walk returned 0.30 -- 3.04x wrong --
at 0.62 deg rotation agreement, 2.48 px reprojection and 2.04 scale
ambiguity. Every clause in `admit()` except reciprocity passed it. Seven
of its eight target cameras were fabricated.

That same signature is what refused pair (14,29) on the real walk -- the
return leg meeting the outbound leg, 20,267 verified inliers between the
two best-conditioned segments in the capture.

These tests are synthetic and arithmetic on purpose. The real-corpus
evidence lives in `tests/test_world_registration.py::TestTheRealWalk`,
which skips on a host without the walk; the mechanism must be pinned
somewhere that never skips.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.world_registration import (  # noqa: E402
    MAX_CAMERA_SCALE_DEVIATION,
    MIN_CAMERAS_FOR_CONSENSUS,
    MIN_CORRESPONDENCES_FOR_CAMERA_SCALE,
    SegmentGeometry,
    _camera_scale,
    _consensus_observations,
    _Observation,
)

IDENTITY = np.eye(3)
ORIGIN = np.zeros(3)
INTRINSICS = np.array([[400.0, 0.0, 180.0], [0.0, 400.0, 320.0],
                       [0.0, 0.0, 1.0]])
POINTS = 12


def _cloud(count=POINTS):
    """Points in front of the camera, spread in x, y and depth."""
    return np.array(
        [[i % 4 - 1.5, (i // 4) - 1.0, 4.0 + 0.5 * i] for i in range(count)],
        dtype=float,
    )


def _segments(scale, count=POINTS):
    """Two segments whose units differ by exactly `scale`.

    `source.points` are `scale` times `target.points`, so a camera that
    genuinely sees the same geometry must measure a depth ratio of
    exactly `scale`.
    """
    target_points = _cloud(count)
    source_points = scale * target_points

    def build(index, points):
        return SegmentGeometry(
            index=index,
            keypoints=[np.zeros((count, 2))],
            descriptors=[None],
            points=points,
            poses={0: (IDENTITY, ORIGIN)},
            observed={(0, i): i for i in range(count)},
            intrinsics=INTRINSICS,
        )

    return build(1, source_points), build(2, target_points)


def _observation(frame, source, target, count=POINTS):
    """One placed camera, at the origin of both frames."""
    return _Observation(
        frame=frame,
        object_points=source.points[:count],
        image_points=np.zeros((count, 2)),
        r_target=IDENTITY,
        t_target=ORIGIN.copy(),
        r_pnp=IDENTITY,
        t_pnp=ORIGIN.copy(),
    )


def _world_of_cameras(ratios, votes=POINTS):
    """Two segments and one placed camera per entry in `ratios`.

    Each camera is given its OWN block of landmarks, and the source's
    copy of block j sits at `ratios[j]` times the target's -- so camera j
    measures a depth ratio of exactly `ratios[j]` through the real
    `_camera_scale` arithmetic, with nothing stubbed and no field
    fabricated. That is faithfully the failure being modelled: a
    fabricated camera is one whose landmarks are at the wrong apparent
    depth, so it reports a scale the honest cameras do not share.

    Returns (source, target, matches, observations).
    """
    blocks = [_cloud(votes) for _ in ratios]
    target_points = np.vstack(blocks)
    source_points = np.vstack(
        [ratio * block for ratio, block in zip(ratios, blocks)]
    )

    def index_of(camera, feature):
        return camera * votes + feature

    source = SegmentGeometry(
        index=1,
        keypoints=[np.zeros((len(target_points), 2))],
        descriptors=[None],
        points=source_points,
        poses={0: (IDENTITY, ORIGIN)},
        observed={(0, index_of(c, f)): index_of(c, f)
                  for c in range(len(ratios)) for f in range(votes)},
        intrinsics=INTRINSICS,
    )
    target = SegmentGeometry(
        index=2,
        keypoints=[np.zeros((len(target_points), 2))
                   for _ in range(len(ratios))],
        descriptors=[None] * len(ratios),
        points=target_points,
        poses={c: (IDENTITY, ORIGIN) for c in range(len(ratios))},
        observed={(c, index_of(c, f)): index_of(c, f)
                  for c in range(len(ratios)) for f in range(votes)},
        intrinsics=INTRINSICS,
    )
    matches = [
        (0, c, [(index_of(c, f), index_of(c, f)) for f in range(votes)])
        for c in range(len(ratios))
    ]
    observations = [
        _observation(c, source, target, count=len(target_points))
        for c in range(len(ratios))
    ]
    return source, target, matches, observations


class TestACameraStatesTheScaleItself:
    def test_a_genuine_camera_measures_the_true_ratio(self):
        source, target = _segments(scale=2.5)
        landmarks = {i: i for i in range(POINTS)}
        observation = _observation(0, source, target)

        ratio, votes = _camera_scale(source, target, observation, landmarks)

        assert ratio == pytest.approx(2.5, rel=1e-9)
        assert votes == POINTS

    def test_a_camera_seeing_nothing_shared_has_no_opinion(self):
        source, target = _segments(scale=2.5)
        observation = _observation(0, source, target)

        ratio, votes = _camera_scale(source, target, observation, {})

        assert ratio is None
        assert votes == 0

    def test_a_landmark_behind_a_camera_does_not_vote(self):
        """A negative depth is a different solution, not a small number."""
        source, target = _segments(scale=1.0)
        behind = source.points.copy()
        behind[:, 2] *= -1.0
        source = SegmentGeometry(
            index=source.index, keypoints=source.keypoints,
            descriptors=source.descriptors, points=behind,
            poses=source.poses, observed=source.observed,
            intrinsics=source.intrinsics,
        )
        observation = _observation(0, source, target)

        ratio, votes = _camera_scale(
            source, target, observation, {i: i for i in range(POINTS)}
        )

        assert ratio is None
        assert votes == 0


class TestTheConsensusDropsTheFabricatedCameras:
    """The whole point: keep the majority that agrees, drop the rest."""

    def _run(self, ratios, votes=POINTS):
        source, target, matches, observations = _world_of_cameras(
            ratios, votes=votes
        )
        kept = _consensus_observations(source, target, matches, observations)
        return {o.frame for o in kept}

    def test_a_lone_disagreeing_camera_is_dropped(self):
        kept = self._run([1.0, 1.0, 1.0, 0.3])

        assert kept == {0, 1, 2}, (
            "the camera measuring a third of everyone else's scale is the "
            "fabricated-camera signature and must not enter the fit"
        )

    def test_the_majority_wins_even_when_it_is_the_smaller_scale(self):
        """Nothing here privileges a scale near 1.0.

        The filter has no idea what the true scale is -- the two segments
        have unrelated units. It knows only that the cameras must agree.
        """
        kept = self._run([0.3, 0.3, 0.3, 1.0])

        assert kept == {0, 1, 2}

    def test_cameras_that_all_agree_are_all_kept(self):
        assert self._run([1.0, 1.0, 1.0, 1.0]) == {0, 1, 2, 3}

    def test_ordinary_drift_inside_the_tolerance_survives(self):
        """A segment's own scale drifts along its length; that is not a lie.

        Measured on the real captures: 0.5% between the halves of one
        segment, up to ~7% between adjacent thirds of another. The filter
        must not mistake that for fabrication, which is why the tolerance
        sits well above it.
        """
        inside = 1.0 + MAX_CAMERA_SCALE_DEVIATION / 2.0
        kept = self._run([1.0, 1.0, 1.0, inside])

        assert kept == {0, 1, 2, 3}

    def test_too_few_opinions_leaves_the_fit_exactly_as_it_was(self):
        """Below a quorum there is no majority to be in.

        Filtering on a one- or two-camera "consensus" is filtering on the
        first camera, so the sparse case is handed on untouched and
        judged by the gate's other clauses -- `min_cameras` above all.
        """
        ratios = [1.0] * (MIN_CAMERAS_FOR_CONSENSUS - 2) + [0.3]
        assert len(ratios) < MIN_CAMERAS_FOR_CONSENSUS
        kept = self._run(ratios)

        assert kept == set(range(len(ratios)))

    def test_a_camera_with_too_little_evidence_does_not_vote(self):
        """Its median is a coin toss; letting it vote scatters the honest ones."""
        thin = MIN_CORRESPONDENCES_FOR_CAMERA_SCALE - 1
        source, target, matches, observations = _world_of_cameras(
            [1.0, 1.0, 1.0, 1.0]
        )
        # Starve the last camera by taking away all but `thin` of its
        # correspondences, leaving everything else untouched.
        frame, target_frame, pairs = matches[3]
        matches[3] = (frame, target_frame, pairs[:thin])

        kept = _consensus_observations(source, target, matches, observations)

        assert {o.frame for o in kept} == {0, 1, 2}


class TestFitDirectionActuallyUsesIt:
    """The call site, not just the function.

    Found by mutation: deleting the `_consensus_observations(...)` line
    from `fit_direction` -- leaving the function itself intact, so every
    other test in this file still passed -- reverted the entire
    behavioural change (corpus-wide, 17 segments and 20,453 points back
    to 12 and 15,571) with a green suite. Every other test here calls
    the filter directly, so none of them could see it.
    """

    def _fit(self, monkeypatch, ratios):
        from scripts import world_registration as wr

        source, target, matches, observations = _world_of_cameras(ratios)
        monkeypatch.setattr(
            wr, "_pnp_observations",
            lambda *_args, **_kwargs: list(observations),
        )
        return wr.fit_direction(source, target, matches), observations

    def test_a_fabricated_camera_does_not_reach_the_fit(self, monkeypatch):
        fit, observations = self._fit(monkeypatch, [1.0, 1.0, 1.0, 0.3])

        assert fit is not None
        assert fit.cameras_considered == len(observations) == 4
        assert fit.cameras == 3, (
            "fit_direction was handed four cameras, one measuring a third "
            "of the others' scale, and used all four -- the consensus "
            "filter is not on the path"
        )
        assert {frame for _, frame in fit.provenance} == {0, 1, 2}, (
            "provenance names a camera the filter dropped, so "
            "MutualEvidence would be reasoning about a fit that never ran"
        )

    def test_nothing_is_dropped_when_the_cameras_agree(self, monkeypatch):
        """The positive control: the filter is not simply truncating."""
        fit, observations = self._fit(monkeypatch, [1.0, 1.0, 1.0, 1.0])

        assert fit.cameras == fit.cameras_considered == len(observations)


class TestTheToleranceIsNotArbitrary:
    """A bound, so `MAX_CAMERA_SCALE_DEVIATION = 5.0` cannot pass quietly.

    Both ends are measured. Below ~0.30 the canonical world loses pair
    (5,32) to ordinary intra-segment drift -- its three forward cameras
    read 4.23, 4.12 and 5.36. At 0.90 the drawer walk loses (14,29),
    because segment 29's ten fabricated cameras stop being separated
    from its genuine ones.
    """

    def test_the_tolerance_stays_between_its_measured_ends(self):
        assert 0.30 <= MAX_CAMERA_SCALE_DEVIATION < 0.90, (
            f"MAX_CAMERA_SCALE_DEVIATION = {MAX_CAMERA_SCALE_DEVIATION} is "
            "outside the range the corpus supports: below 0.30 honest "
            "drift is mistaken for fabrication, and at 0.90 the filter "
            "stops separating the drawer walk's fabricated cameras"
        )


class TestTheGateReadsTheBaselineTheFitActuallyHad:
    """`target_span_over_depth` must describe the placed cameras.

    Scale enters a Sim3 only through the baseline between the target
    cameras a fit placed, and that is routinely a subset of the segment
    -- more so since the consensus filter. Reading the whole segment's
    span credits an estimate with parallax it never saw: on the drawer
    walk the forward fit of (14,29) uses cameras spanning 0.1699 while
    segment 29 as a whole spans 0.7335, a 4.3x overstatement.
    """

    def test_the_span_shrinks_to_the_cameras_that_were_used(self):
        from scripts.world_registration import _placed_span_over_depth

        _, target, _, observations = _world_of_cameras([1.0] * 4)
        spread = {0: (IDENTITY, np.array([0.0, 0.0, 0.0])),
                  1: (IDENTITY, np.array([-6.0, 0.0, 0.0])),
                  2: (IDENTITY, np.array([-12.0, 0.0, 0.0])),
                  3: (IDENTITY, np.array([-18.0, 0.0, 0.0]))}
        target.poses.update(spread)

        whole = _placed_span_over_depth(target, observations)
        clustered = _placed_span_over_depth(target, observations[:2])

        assert clustered < whole, (
            "a fit that placed two adjacent cameras must not be scored "
            "against the baseline of four spread ones"
        )

    def test_a_fit_with_no_placeable_camera_has_no_baseline(self):
        from scripts.world_registration import _placed_span_over_depth

        _, target, _, observations = _world_of_cameras([1.0] * 3)
        target.poses.clear()

        assert _placed_span_over_depth(target, observations) == 0.0


class TestTheFilterCanOnlyNarrow:
    """It removes cameras. It never adds one, and never moves a threshold.

    This is the property that makes the change safe to ship: fewer
    cameras can only make `min_cameras` harder to satisfy, so no pair is
    admitted on less evidence than before. The extra recall comes from
    measuring the RIGHT cameras, not from a lower bar.
    """

    def test_the_result_is_always_a_subset_of_what_it_was_given(self):
        source, target, matches, observations = _world_of_cameras(
            [1.0, 1.0, 1.0, 0.3, 3.0, 1.05]
        )

        kept = _consensus_observations(source, target, matches, observations)

        assert len(kept) <= len(observations)
        assert all(any(k is o for o in observations) for k in kept)
