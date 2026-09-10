"""Registration runs when a walk ends, and cannot take the walk down with it.

The defect these cover is not a wrong answer, it is a question never
asked. `scripts/world_registration.py` -- the Sim3 fit, the mutual
evidence rule, the cycle check, the digest-bound persistence, the
serving-side staleness refusal -- was complete, tested and inert. Nothing
called it. Every physical walk therefore finalised with no
`placements.json`, and the phone drew every segment as its own island:
22 disconnected fragments on the 2026-08-29 drawer walk, whose segments
were not refused so much as never considered.

So these tests are about wiring and blast radius, not about geometry.
The geometry is `tests/test_world_registration.py`'s subject and is
unchanged.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.world_build_session import register_session  # noqa: E402
from scripts.world_registration import (  # noqa: E402
    NO_VISUAL_LINK,
    SupportMissingError,
)


def _run(script, *args):
    return subprocess.run(
        [sys.executable, f"scripts/{script}", *args],
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def registered_world(tmp_path_factory):
    """A real synthetic walk, built and registered through the driver.

    The `derived_world` conftest fixture cannot be used here: it writes
    no `support.json`, and registration refuses outright without the
    2-D/3-D association. Driving the script end to end is also the only
    way to exercise the wiring these tests are about.
    """
    root = tmp_path_factory.mktemp("reg")
    result = _run(
        "world_build_session.py",
        "--synthetic", "--synthetic-frames", "16",
        "--root", str(root), "--format", "json", "--register",
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)

    from tower.world_builder.store import WorldStore

    return (
        WorldStore(root),
        report["world_id"],
        report["session_id"],
        report,
    )


REAL_ROOT = Path("data/world_builder")


@pytest.fixture(scope="module")
def linked_world():
    """A saved world with at least two segments carrying geometry.

    Registration's pair accounting is only observable where pairs exist,
    and `--synthetic` yields one segment however many frames it renders.
    Resolved by world id order so the choice is stable, and skipped
    rather than faked where the corpus is absent -- the same shape
    `TestTheRealWalk` uses, and for the same reason.
    """
    from scripts.world_registration import SupportMissingError, register
    from tower.world_builder.store import WorldStore

    if not (REAL_ROOT / "worlds").is_dir():
        pytest.skip(f"no world corpus at {REAL_ROOT} on this host")
    store = WorldStore(REAL_ROOT)
    for path in sorted((REAL_ROOT / "worlds").iterdir()):
        try:
            world = store.read_world(path.name)
        except Exception:
            continue
        for session in world.session_ids:
            try:
                report = register(store, path.name, session)
            except SupportMissingError:
                continue
            except Exception:
                continue
            if report["segments_with_geometry"] >= 2:
                return report, (path.name, session)
    pytest.skip("no saved world has two segments carrying geometry")


class _Boom:
    """A store whose every read fails, standing in for any solver fault."""

    def __init__(self, error):
        self._error = error

    def __getattr__(self, name):
        def explode(*_args, **_kwargs):
            raise self._error

        return explode


class TestTheDriverAsksTheQuestion:
    def test_a_run_without_the_flag_still_does_not_register(self, tmp_path):
        """The flag is opt-in at the script, and the Tower opts in.

        Kept explicit so a batch reprocess -- which may be exploring a
        threshold -- cannot silently overwrite what a world serves.
        """
        result = _run(
            "world_build_session.py",
            "--synthetic", "--synthetic-frames", "10",
            "--root", str(tmp_path), "--format", "json",
        )

        assert result.returncode == 0
        assert "registration" not in json.loads(result.stdout)

    def test_registering_reports_what_it_placed_and_writes_it_down(
        self, tmp_path
    ):
        result = _run(
            "world_build_session.py",
            "--synthetic", "--synthetic-frames", "12",
            "--root", str(tmp_path), "--format", "json", "--register",
        )

        assert result.returncode == 0
        report = json.loads(result.stdout)
        registration = report["registration"]
        assert registration["attempted"] is True
        assert registration["wrote_placements"] is True

        placements = (
            Path(tmp_path) / "worlds" / report["world_id"] / "derived"
            / report["session_id"] / "placements.json"
        )
        assert placements.exists(), (
            "registration reported success and wrote nothing, so the "
            "serving layer still has no placement to read"
        )

        # Refusal is the default answer and a perfectly good outcome; what
        # must never happen is a placement claimed without a transform.
        rows = json.loads(placements.read_text())["placements"]
        assert rows
        for row in rows:
            if row["state"] == "registered":
                assert row["rotation_wxyz"] is not None
                assert row["translation"] is not None
                assert row["scale"] is not None
                assert row["reference_segment"] is not None
            else:
                assert row["rotation_wxyz"] is None
                assert row["translation"] is None
                assert row["scale"] is None
                assert row["refusal_reason"]


class TestAFailedRegistrationCostsNothingButTheRegistration:
    """A reconstruction is worth keeping even when it cannot be placed.

    Registration runs after the last build, so by the time it can fail
    the poses, points and support are already on disk. Letting it
    propagate would throw away the whole walk to lose a transform -- and
    it is the newest, least exercised step in the pipeline.
    """

    def test_a_solver_fault_is_reported_not_raised(self, caplog):
        outcome = register_session(
            _Boom(RuntimeError("the solver fell over")), "w" * 32, "s" * 32
        )

        assert outcome["attempted"] is True
        assert outcome["wrote_placements"] is False
        assert "the solver fell over" in outcome["error"]

    def test_a_world_with_no_support_says_so_rather_than_failing(self):
        """Worlds built before support.json existed cannot be registered.

        That is a refusal with a remedy in it, not an error, and it must
        read differently from a crash.
        """
        outcome = register_session(
            _Boom(SupportMissingError("world w has no support.json")),
            "w" * 32,
            "s" * 32,
        )

        assert outcome["attempted"] is True
        assert outcome["wrote_placements"] is False
        assert "support.json" in outcome["refusal"]
        assert "error" not in outcome


class TestTheTransformIsBoundToTheBuildItWasSolvedAgainst:
    """A rebuild must invalidate every placement, and nothing tested it.

    `write_derived` rewrites poses and points wholesale and never touches
    `placements.json`, so a Sim3 outlives the reconstruction it was
    fitted to. `usable_placements` refuses on `input_digest` for exactly
    that reason -- and forcing `placements_from_report(..., digest=None)`
    left this file green at 6 passed while `usable_placements` silently
    dropped to zero registered rows. The guard the whole design rests on
    was asserted nowhere.
    """

    def test_a_placement_solved_against_another_build_is_not_served(
        self, registered_world, tmp_path
    ):
        """A REBUILD WRITES BOTH MANIFESTS, so simulating one has to.

        This edited only the world's `derived/manifest.json`. That was a
        complete simulation while there was one copy; `write_derived` now
        writes a second beside the poses and points it describes, and
        `usable_placements` reads THAT one -- because after a crash between
        the two writes it is the copy that matches the geometry actually on
        disk. Editing one file no longer means "a rebuild happened", it
        means "the two copies disagree", which is a different question and
        the next test asks it.

        Caught by the suite, as a regression, from the change that gave
        `usable_placements` the session id.
        """
        from tower.results.world_builder_geometry import usable_placements

        store, world_id, session_id, _ = registered_world
        before = usable_placements(store, world_id, session_id)
        assert before, "the fixture served no placements at all"

        paths = [
            # `world_path` names world.json itself, not the directory.
            store.world_path(world_id).parent / "derived" / "manifest.json",
            store.session_manifest_path(world_id, session_id),
        ]
        originals = {}
        for path in paths:
            assert path.exists(), path
            originals[path] = path.read_text()
            manifest = json.loads(originals[path])
            manifest["input_digest"] = "0" * 64
            path.write_text(json.dumps(manifest))
        try:
            after = usable_placements(store, world_id, session_id)
        finally:
            for path, text in originals.items():
                path.write_text(text)

        assert after == {}, (
            "a placement solved against a different build was still "
            "served; a rebuild replaces poses and points wholesale, so "
            "the transform now describes geometry that does not exist"
        )

    def test_the_copy_beside_the_geometry_decides_when_the_two_disagree(
        self, registered_world
    ):
        """A crash between the two manifest writes, both directions.

        `write_derived` writes poses, points, support, then the SESSION
        manifest, then the world's. A crash in that last gap leaves new
        geometry, a new session manifest and a stale world manifest -- so
        the session's copy is the one that matches what is on disk, and
        judging placements by it is what keeps them bound to the geometry
        they were solved against.

        The reverse is the conservative case: a session copy that says the
        geometry moved refuses the placements even if the world's copy
        still agrees with them.
        """
        from tower.results.world_builder_geometry import usable_placements

        store, world_id, session_id, _ = registered_world
        world_manifest = store.world_path(world_id).parent / "derived" / "manifest.json"
        session_manifest = store.session_manifest_path(world_id, session_id)
        originals = {p: p.read_text() for p in (world_manifest, session_manifest)}

        def with_digest(path, digest):
            manifest = json.loads(originals[path])
            manifest["input_digest"] = digest
            path.write_text(json.dumps(manifest))

        try:
            # The world's copy lagged; the session's matches the geometry.
            with_digest(world_manifest, "0" * 64)
            assert usable_placements(store, world_id, session_id), (
                "placements bound to the geometry on disk were refused "
                "because a second, staler copy of the manifest disagreed"
            )

            # And the other way: the session's copy says the geometry moved.
            world_manifest.write_text(originals[world_manifest])
            with_digest(session_manifest, "0" * 64)
            assert usable_placements(store, world_id, session_id) == {}, (
                "the copy beside the geometry said the build moved and the "
                "placements were served anyway"
            )
        finally:
            for path, text in originals.items():
                path.write_text(text)


class TestEveryCandidatePairIsAccountedFor:
    """A pair the matcher could not link must still produce a row.

    This branch used to `continue`. On the 2026-08-29 drawer walk that
    left `candidate_pairs` at 228 of the 253 pairs over 23 segments with
    geometry, and the missing 25 were indistinguishable in the report
    from pairs that were never enumerated. The distinction matters
    because it is the one that says whether a walk's problem is
    RETRIEVAL -- we never found the shared view -- or ESTIMATION -- we
    found it and could not agree about it. Those want opposite work.

    These need a world with at least two segments carrying geometry.
    `--synthetic` produces exactly one, so a fixture built from it makes
    every assertion here vacuously true -- `candidate_pairs == 0` really
    does equal `0 * -1 // 2`. They run against the saved corpus and skip
    where it is absent, in the same shape as `TestTheRealWalk`.
    """

    def test_the_report_covers_the_whole_upper_triangle(self, linked_world):
        report, _ = linked_world

        n = report["segments_with_geometry"]
        assert n >= 2
        assert report["candidate_pairs"] == n * (n - 1) // 2, (
            "a pair vanished from the report; a pair nobody could link is "
            "a measurement, not an absence"
        )

    def test_an_unlinkable_pair_names_the_matcher_as_the_reason(
        self, linked_world
    ):
        report, _ = linked_world

        unlinked = [p for p in report["pairs"] if p["reason"] == NO_VISUAL_LINK]
        if not unlinked:
            pytest.skip("every pair on this world shares a verified view")
        for pair in unlinked:
            assert pair["registered"] is False
            assert pair["clauses"]["verified_frame_pairs"] == 0
            assert pair["clauses"]["inliers"] == 0

    def test_a_linked_pair_reports_how_much_evidence_it_had(self, linked_world):
        """Refusals were legible; the evidence behind them was not.

        A pair refused with 4,449 verified inliers and one refused with
        16 are different situations, and before this the report said the
        same thing about both.
        """
        report, _ = linked_world

        linked = [
            p for p in report["pairs"]
            if p["clauses"].get("verified_frame_pairs", 0) > 0
        ]
        assert linked, "no pair on this world shares a verified view"
        for pair in linked:
            assert pair["clauses"]["inliers"] > 0

    def test_a_fit_says_how_many_cameras_it_started_with(self, linked_world):
        """`cameras` alone cannot distinguish 4-of-4 from 4-of-23."""
        report, _ = linked_world

        solved = [
            p for p in report["pairs"] if "cameras_considered" in p["clauses"]
        ]
        assert solved, "no pair on this world solved in both directions"
        for pair in solved:
            assert (
                pair["clauses"]["cameras_considered"]
                >= pair["clauses"]["cameras"]
            ), "the filter reported keeping more cameras than it was given"
