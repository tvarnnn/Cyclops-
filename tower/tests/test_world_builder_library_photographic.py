"""The Saved Worlds row tells the truth about the photographic room.

T3 of the Mac/iOS validation of 2026-09-22 was found on the status channel
and fixed there first, and the listing is the OTHER surface the phone reads
-- the one a person actually chooses a walk from, and then shuts the Tower
down. Its row had the identical defect and one worse property: the only
question it asked was `_photographic_build_running`, "is a stage running
this millisecond", and that probe FAILED OPEN in two directions at once.
A stage that failed is not running; a stage that is owed is not running; a
probe that raised returned `False`. All three rendered as **complete**, and
the function's own warning said so about itself:

    this row keeps its settled word and may say 'complete' over a world
    that is still being built

These tests pin the four shapes that used to read `complete` and must not,
the one shape that read `complete` and MUST GO ON READING IT -- 165 of the
166 worlds on the machine this was written for -- and the agreement between
this surface and the panel it opens, for every one of them.
"""

import dataclasses

import pytest

from tests.result_channel_fixtures import build_world
from tower.results.world_builder import WorldBuilderStatusProducer
from tower.results.world_builder_library import build_world_listing
from tower.world_builder.store import WorldStore

# A finalization that SUCCEEDED. Every session below carries it, because
# that is the point: the capture ran, the final solve solved, the derived
# tree is on disk and the render route serves it. Nothing that follows is a
# claim about the walk -- it is a claim about the photographic room built on
# top of a walk that went fine, which is exactly the distinction the row's
# one word cannot hold and the `photographic` block exists to carry.
FINALIZED = {
    "state": "complete", "final_solve": "solved",
    "started_at": 1.0, "updated_at": 2.0, "detail": "",
}

# What the panel's lifecycle word means in the picker's vocabulary. The two
# vocabularies are deliberately not the same list -- `idle` and
# `unavailable` only mean anything against a live subscription -- but where
# both have a word for a state they must be the same word, which is
# `session_state`'s own opening comment: "A row in the picker and the panel
# it opens must not disagree about what a session is."
LIFECYCLE_TO_ROW = {
    "receiving": "receiving",
    "finalizing": "finalizing",
    "ready": "complete",
    "interrupted": "interrupted",
    "stopped_unbuilt": "unbuilt",
}


@pytest.fixture
def built_session(tmp_path):
    """A real world, built by the real engine, stopped and finalized.

    Returns `(store, world_id, session_id, rewrite)`, where `rewrite`
    replaces fields on the session record in place. No hand-written world:
    the thing under test is what the Tower says about what it persisted.
    """
    root = tmp_path / "worlds"
    world_id, session_id = build_world(root, frames=6)
    store = WorldStore(root)

    def rewrite(**fields):
        session = store.read_session(world_id, session_id)
        store.write_session(dataclasses.replace(session, **fields))

    return store, world_id, session_id, rewrite


def _row(store):
    return build_world_listing(store)["worlds"][0]["sessions"][0]


def _lifecycle(store, world_id, session_id):
    """The panel's answer about the same session, from the same disk."""
    snapshot = WorldBuilderStatusProducer(store.root, lambda: 1e9).snapshot(
        world_id=world_id, session_id=session_id
    )
    return getattr(snapshot, "payload", snapshot)["lifecycle"]


def _break_the_liveness_probe(monkeypatch):
    """Make the stage-liveness probe raise, as a locked file or a bad
    status.json on a real disk would."""
    import tower.results.world_builder_render as render

    def boom(*args, **kwargs):
        raise OSError("the liveness probe is broken")

    monkeypatch.setattr(render, "_stage_running", boom)


# -- the four shapes that used to read `complete` --------------------------


def test_a_failed_photographic_build_is_not_reported_as_photographic_success(
    built_session,
):
    """T3 proper: `failed` was indistinguishable from `complete`.

    The row's WORD stays `complete`, and that is deliberate and argued in
    `session_state`: the session IS saved, the geometry is on disk, and
    `interrupted` is contract-defined as a claim about the capture, which
    did not fail. `finalizing` would be worse still -- it tells the wearer
    to wait for a build that is not coming back on its own.

    So the claim that must not be made is the PHOTOGRAPHIC one, and the
    only place the row can make it is the `photographic` block. It says
    `failed`, it names the stage, and it carries the stage's own detail.
    """
    store, world_id, session_id, rewrite = built_session
    rewrite(finalization=FINALIZED, stages={
        "surface": {"state": "ok"},
        "appearance": {"state": "failed", "detail": "no keyframe survived"},
    })

    row = _row(store)
    photographic = row["photographic"]
    assert photographic["state"] == "failed", (
        "the row said nothing at all about a photographic build that "
        "failed, which is the whole of T3"
    )
    assert photographic["stage"] == "appearance"
    assert photographic["detail"] == "no keyframe survived"
    # The world is still openable and the row still says so.
    assert row["has_geometry"] is True
    assert row["state"] == "complete"
    # And the panel says the same thing about the same session.
    assert _lifecycle(store, world_id, session_id)["photographic"] == photographic


def test_an_interrupted_stage_with_no_live_process_is_not_complete(built_session):
    """A stage recorded `running` with nothing alive is ABANDONED, not running.

    This is the shape `world_finish_pending.py` picks up at the next Tower
    start, so the world genuinely is owed a photographic room and a wearer
    genuinely should wait. The old row asked "is a process running now",
    got `False`, and said **complete** over it.
    """
    store, world_id, session_id, rewrite = built_session
    rewrite(finalization=FINALIZED, stages={
        "surface": {"state": "ok"},
        "appearance": {"state": "running"},
    })

    row = _row(store)
    assert row["photographic"]["state"] == "owed"
    assert row["state"] == "finalizing", (
        "a world that still owes a photographic room was offered to the "
        "wearer as finished"
    )

    # `stopped` is the other interrupted record state, and the surface is
    # the other photographic stage. Both reach the same answer.
    rewrite(finalization=FINALIZED, stages={"surface": {"state": "stopped"}})
    row = _row(store)
    assert row["photographic"]["state"] == "owed"
    assert row["photographic"]["stage"] == "surface"
    assert row["state"] == "finalizing"


def test_a_broken_photographic_probe_does_not_produce_complete(
    built_session, monkeypatch
):
    """T4, on the listing. A probe that cannot answer must not answer "fine".

    Two different breakages, because the row has two layers that can fail:
    the stage-liveness probe underneath `photographic_state`, and
    `photographic_state` itself. Before this change the first returned
    `False` (`_photographic_build_running`'s `except` arm, which warned
    that the row "may say 'complete' over a world that is still being
    built") and the second did not exist.
    """
    store, world_id, session_id, rewrite = built_session
    rewrite(finalization=FINALIZED, stages={
        "surface": {"state": "ok"},
        "appearance": {"state": "running"},
    })

    # 1. The liveness probe raises. The record still says a stage did not
    #    finish, so "nothing is running" is a claim on no evidence.
    _break_the_liveness_probe(monkeypatch)
    row = _row(store)
    assert row["photographic"]["state"] == "unobservable"
    assert row["state"] == "finalizing"

    # 2. `photographic_state` itself raises, which is the row's own
    #    `except` arm. It must reach the same word, not fall back to the
    #    settled one.
    import tower.world_builder.photographic as photographic_module

    def boom(*args, **kwargs):
        raise RuntimeError("the photographic state is broken")

    monkeypatch.setattr(photographic_module, "photographic_state", boom)
    row = _row(store)
    assert row["photographic"]["state"] == "unobservable"
    assert row["state"] == "finalizing"


def test_a_broken_probe_does_not_relabel_a_settled_world(
    built_session, monkeypatch
):
    """The OTHER failure, which is just as forbidden as the false "Saved".

    An earlier draft of the state machine asked liveness for every session.
    With the probe broken that sends all 165 historical worlds to
    `unobservable` at once, and a conservative mapping then parks every one
    of them on "Improving" forever. `photographic_state` only consults the
    probe where the RECORD is ambiguous, so a world whose record is settled
    is answered without it -- and this pins that the listing inherits the
    property rather than re-deriving it.
    """
    store, _world_id, _session_id, rewrite = built_session
    _break_the_liveness_probe(monkeypatch)

    rewrite(finalization=FINALIZED)  # no stages at all: the historical shape
    assert _row(store)["state"] == "complete"
    assert _row(store)["photographic"]["state"] == "never_recorded"

    rewrite(finalization=FINALIZED, stages={
        "surface": {"state": "ok"}, "appearance": {"state": "ok"},
    })
    assert _row(store)["state"] == "complete"
    assert _row(store)["photographic"]["state"] == "complete"


# -- the shape that must NOT move -----------------------------------------


def test_a_historical_world_with_no_stage_record_still_reads_complete(
    built_session,
):
    """THE MOST IMPORTANT TEST IN THIS FILE.

    166 worlds on the machine this was written for; 165 of them were built
    by a Tower that had no photographic stages at all. They are not broken,
    they are owed nothing, and relabelling them would turn one honest bug
    report into a hundred and sixty-five false ones. "Anything without an
    appearance is unfinished" is the obvious rule and it is the wrong one.

    Two historical shapes, because `photographic_state` distinguishes them
    and the row must not: a finalized session with no stage record at all
    (`never_recorded`), and a session with no finished global solve to
    build a photographic room from (`unattempted`). Both are settled, both
    read `complete`, and neither costs a probe.
    """
    store, world_id, session_id, rewrite = built_session

    # 1. The 165: finalized, solved, geometry on disk, and no `stages` key
    #    because the Tower that built it had never heard of one.
    rewrite(finalization=FINALIZED)
    row = _row(store)
    assert row["photographic"]["state"] == "never_recorded"
    assert row["state"] == "complete", (
        "a world built before the photographic stages existed was "
        "relabelled as unfinished"
    )
    assert _lifecycle(store, world_id, session_id)["state"] == "ready"

    # 2. Older still: no finalization record either. Nothing can be owed a
    #    photographic room it has no solve to build from.
    rewrite(finalization=None)
    row = _row(store)
    assert row["photographic"]["state"] == "unattempted"
    assert row["state"] == "complete"

    # 3. And a Tower with the appearance stage switched off, which records
    #    the stages as `unavailable`. Nothing is wrong and nothing is
    #    coming, so nothing should be promised.
    rewrite(finalization=FINALIZED, stages={
        "surface": {"state": "unavailable"},
        "appearance": {"state": "unavailable", "detail": "appearance is off"},
    })
    row = _row(store)
    assert row["photographic"]["state"] == "unattempted"
    assert row["state"] == "complete"


# -- the two surfaces say one thing ---------------------------------------


def test_the_row_and_the_panel_agree_about_every_photographic_state(
    built_session, monkeypatch
):
    """The picker and the panel, over the whole new vocabulary.

    `test_the_picker_and_the_panel_agree_about_every_session` does this for
    the geometry shapes and `test_the_listing_states_each_session_the_way_
    the_status_channel_does` for the lock shapes. This is the same
    obligation for the states the photographic record introduced -- and it
    is the one that would have caught the defect being fixed, because the
    status channel learned `photographic_state` first and the listing had
    not, so for a while the panel said "Improving" while the row for the
    SAME session said "Complete".

    Both surfaces are computed from the same disk by separate readers with
    no shared call, which is the only thing that makes the assertion mean
    anything.
    """
    store, world_id, session_id, rewrite = built_session

    shapes = [
        ("never_recorded", dict(finalization=FINALIZED)),
        ("complete", dict(finalization=FINALIZED, stages={
            "surface": {"state": "ok"}, "appearance": {"state": "ok"}})),
        ("failed", dict(finalization=FINALIZED, stages={
            "surface": {"state": "ok"},
            "appearance": {"state": "failed", "detail": "boom"}})),
        ("owed", dict(finalization=FINALIZED, stages={
            "surface": {"state": "ok"}, "appearance": {"state": "running"}})),
        ("unattempted", dict(finalization=FINALIZED, stages={
            "surface": {"state": "unavailable"},
            "appearance": {"state": "unavailable"}})),
    ]
    for expected_state, fields in shapes:
        rewrite(**fields)
        row = _row(store)
        lifecycle = _lifecycle(store, world_id, session_id)
        assert row["photographic"]["state"] == expected_state, fields
        assert lifecycle["photographic"] == row["photographic"], (
            f"the two surfaces describe the photographic room differently "
            f"for {expected_state}: {lifecycle['photographic']} vs "
            f"{row['photographic']}"
        )
        assert row["state"] == LIFECYCLE_TO_ROW[lifecycle["state"]], (
            f"for {expected_state} the picker says {row['state']!r} and the "
            f"panel says {lifecycle['state']!r}"
        )

    # And `unobservable`, which needs the probe broken to reach.
    rewrite(finalization=FINALIZED, stages={
        "surface": {"state": "ok"}, "appearance": {"state": "running"}})
    _break_the_liveness_probe(monkeypatch)
    row = _row(store)
    lifecycle = _lifecycle(store, world_id, session_id)
    assert row["photographic"]["state"] == "unobservable"
    assert lifecycle["photographic"]["state"] == "unobservable"
    assert row["state"] == LIFECYCLE_TO_ROW[lifecycle["state"]] == "finalizing"


def test_the_photographic_block_is_additive_and_the_contract_id_holds():
    """A new key, not a new contract.

    iOS reads these rows key by key out of a `[String: Any]`, so a key it
    does not know is a key it never looks at -- but `WorldListingDecoder`
    equality-tests `contract` on the first line of every guard, so bumping
    it would empty Saved Worlds on every build that is not rebuilt from
    this branch. The block is worth having; it is not worth that.
    """
    from tower.results.world_builder_library import WORLDS_CONTRACT

    assert WORLDS_CONTRACT == "world_builder.worlds/2026-09-10"


def test_the_row_carries_the_block_for_a_session_nobody_probed(built_session):
    """Every row has the key, including the ones the word did not need it for.

    A client reading `row["photographic"]["state"]` must not have to handle
    the key being absent on some rows and present on others -- that is the
    shape that makes a decoder guess, and guessing "fine" is how T3
    happened. Three keys, always, on every session in the listing.
    """
    store, _world_id, _session_id, rewrite = built_session
    rewrite(finalization=FINALIZED)
    row = _row(store)
    assert set(row["photographic"]) == {"state", "stage", "detail"}
    assert isinstance(row["photographic"]["detail"], str)
